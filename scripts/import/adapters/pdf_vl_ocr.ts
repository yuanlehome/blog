/**
 * PaddleOCR-VL API Client
 *
 * Handles communication with PaddleOCR-VL layout parsing API
 *
 * Network stability enhancements:
 * - Force IPv4 to avoid IPv6 connectivity issues in CI
 * - Proper connection timeout via undici Agent
 * - Enhanced diagnostics for DNS/TCP/TLS
 * - Exponential backoff retry with detailed error logging
 */

import type { Logger } from '../../logger/types.js';
import * as dnsPromises from 'dns/promises';
import * as net from 'net';
import * as https from 'https';
import { Agent } from 'undici';

/**
 * OCR result structure
 */
export interface PaddleOcrVlResult {
  markdown: string;
  images: Record<string, string>; // img_path -> img_url
  outputImages?: string[];
}

/**
 * PaddleOCR-VL API response structure
 */
interface PaddleOcrVlResponse {
  result?: {
    layoutParsingResults?: Array<{
      markdown?: {
        text?: string;
        images?: Record<string, string>;
      };
      outputImages?: string[];
    }>;
  };
  error?: string;
  message?: string;
}

/**
 * Custom error for OCR API failures
 */
export class OcrApiError extends Error {
  constructor(
    message: string,
    public statusCode: number,
    public responseBody?: string,
    public requestId?: string,
    public cause?: Error,
  ) {
    super(message);
    this.name = 'OcrApiError';
    if (cause) {
      this.cause = cause;
    }
  }
}

/**
 * Custom error for OCR response parsing failures
 */
export class OcrParseError extends Error {
  constructor(
    message: string,
    public responseBody?: string,
  ) {
    super(message);
    this.name = 'OcrParseError';
  }
}

/**
 * Error code classification for quick diagnosis
 */
export type OcrErrorCode =
  | 'OCR_NET_INVALID_IP'
  | 'OCR_NET_DNS_FAIL'
  | 'OCR_NET_TIMEOUT'
  | 'OCR_NET_TLS'
  | 'OCR_NET_CONNECTION'
  | 'OCR_HTTP_NON_2XX'
  | 'OCR_RESPONSE_PARSE_FAIL'
  | 'OCR_RESULT_EMPTY'
  | 'OCR_PDF_FETCH_FAIL'
  | 'OCR_PDF_PARSE_FAIL'
  | 'OCR_UNKNOWN';

/**
 * Classify error into diagnostic code for quick identification
 * This helps diagnose failures like "Invalid IP address: undefined"
 * Priority order matters - more specific checks should come before generic ones
 */
export function classifyOcrError(error: any, statusCode: number): OcrErrorCode {
  // Check error.cause and nested causes for network codes
  const causeCode = error?.cause?.code || error?.cause?.cause?.code;
  const errorCode = error?.code;
  const errorMessage = error?.message || '';

  // Invalid IP address (common in CI with undefined env vars)
  if (
    causeCode === 'ERR_INVALID_IP_ADDRESS' ||
    errorCode === 'ERR_INVALID_IP_ADDRESS' ||
    errorMessage.includes('Invalid IP address')
  ) {
    return 'OCR_NET_INVALID_IP';
  }

  // DNS failures
  if (
    causeCode === 'ENOTFOUND' ||
    causeCode === 'EAI_AGAIN' ||
    errorCode === 'ENOTFOUND' ||
    errorCode === 'EAI_AGAIN' ||
    errorMessage.includes('getaddrinfo')
  ) {
    return 'OCR_NET_DNS_FAIL';
  }

  // TLS/SSL errors (check before timeout, since "TLS handshake timeout" should be TLS not timeout)
  if (
    causeCode === 'EPROTO' ||
    causeCode === 'UNABLE_TO_VERIFY_LEAF_SIGNATURE' ||
    errorCode === 'EPROTO' ||
    errorMessage.includes('SSL') ||
    errorMessage.includes('TLS') ||
    errorMessage.includes('certificate')
  ) {
    return 'OCR_NET_TLS';
  }

  // Timeout errors (after TLS check)
  if (
    causeCode === 'ETIMEDOUT' ||
    causeCode === 'ESOCKETTIMEDOUT' ||
    errorCode === 'ETIMEDOUT' ||
    errorCode === 'ESOCKETTIMEDOUT' ||
    error?.name === 'AbortError' ||
    (errorMessage.includes('timeout') && !errorMessage.includes('TLS')) ||
    errorMessage.includes('abort')
  ) {
    return 'OCR_NET_TIMEOUT';
  }

  // HTTP non-2xx status codes (check early for priority)
  if (statusCode >= 300 && statusCode < 600) {
    return 'OCR_HTTP_NON_2XX';
  }

  // Parse failures
  if (error instanceof OcrParseError || error?.name === 'OcrParseError') {
    return 'OCR_RESPONSE_PARSE_FAIL';
  }

  // Empty result
  if (errorMessage.includes('empty') || errorMessage.includes('no markdown')) {
    return 'OCR_RESULT_EMPTY';
  }

  // PDF fetch/parse failures (check before generic connection errors)
  if (errorMessage.includes('PDF')) {
    if (errorMessage.includes('download') || errorMessage.includes('fetch')) {
      return 'OCR_PDF_FETCH_FAIL';
    }
    if (
      errorMessage.includes('parse') ||
      errorMessage.includes('invalid') ||
      errorMessage.includes('format')
    ) {
      return 'OCR_PDF_PARSE_FAIL';
    }
  }

  // Connection errors (generic - check last)
  if (
    causeCode === 'ECONNRESET' ||
    causeCode === 'ECONNREFUSED' ||
    causeCode === 'EPIPE' ||
    causeCode === 'EHOSTUNREACH' ||
    causeCode === 'ENETUNREACH' ||
    errorCode === 'ECONNRESET' ||
    errorCode === 'ECONNREFUSED' ||
    errorCode === 'EPIPE' ||
    errorCode === 'EHOSTUNREACH' ||
    errorCode === 'ENETUNREACH' ||
    errorMessage.includes('fetch failed')
  ) {
    return 'OCR_NET_CONNECTION';
  }

  return 'OCR_UNKNOWN';
}

/**
 * Configuration for OCR API calls
 */
interface OcrConfig {
  retries: number;
  timeoutMs: number;
  connectTimeoutMs: number;
  enableDiag: boolean;
  overrideIp?: string;
}

/**
 * Generate a short OCR job ID for correlation
 * Format: ocr-{8 random hex chars}
 */
export function generateOcrJobId(): string {
  const randomHex = Math.random().toString(16).substring(2, 10);
  return `ocr-${randomHex}`;
}

/**
 * Get OCR configuration from environment
 */
function getOcrConfig(): OcrConfig {
  // Check for IP override configuration
  // Support multiple environment variable names for compatibility
  // Priority order (first defined wins):
  // 1. PADDLE_OCR_VL_API_IP (most specific, recommended)
  // 2. PADDLE_OCR_VL_IP (shorter variant)
  // 3. PDF_OCR_API_IP (generic PDF OCR variant)
  // 4. PADDLEOCR_VL_IP (legacy, for backward compatibility)
  const ipOverride =
    process.env.PADDLE_OCR_VL_API_IP ||
    process.env.PADDLE_OCR_VL_IP ||
    process.env.PDF_OCR_API_IP ||
    process.env.PADDLEOCR_VL_IP;

  return {
    retries: parseInt(process.env.PDF_OCR_RETRY || '3', 10),
    timeoutMs: parseInt(process.env.PDF_OCR_TIMEOUT_MS || '90000', 10),
    connectTimeoutMs: parseInt(process.env.PDF_OCR_CONNECT_TIMEOUT_MS || '15000', 10),
    enableDiag: process.env.PDF_OCR_DIAG === '1' || process.env.CI === 'true',
    overrideIp: ipOverride?.trim() || undefined,
  };
}

/**
 * Validate API URL according to official documentation
 * URL must be obtained from https://aistudio.baidu.com/paddleocr/task page
 */
function validateApiUrl(apiUrl: string): void {
  if (!apiUrl || apiUrl.trim() === '') {
    throw new Error(
      'API URL is required. Please obtain it from https://aistudio.baidu.com/paddleocr/task ' +
        'and set it via PDF_OCR_API_URL or PADDLEOCR_VL_API_URL environment variable.',
    );
  }

  let url: URL;
  try {
    url = new URL(apiUrl);
  } catch (error) {
    throw new Error(
      `Invalid API URL format: ${apiUrl}. ` +
        'Please verify the URL from https://aistudio.baidu.com/paddleocr/task',
    );
  }

  // Must be HTTPS
  if (url.protocol !== 'https:') {
    throw new Error(
      `API URL must use HTTPS protocol, got: ${url.protocol}. ` +
        'Please verify the URL from https://aistudio.baidu.com/paddleocr/task',
    );
  }

  // Must include /layout-parsing in path (as per official documentation)
  if (!url.pathname.includes('/layout-parsing')) {
    throw new Error(
      `API URL path must include '/layout-parsing', got: ${url.pathname}. ` +
        'Please verify the URL from https://aistudio.baidu.com/paddleocr/task',
    );
  }
}

/**
 * Network diagnostic results
 */
interface DiagnosticResult {
  dns?: {
    success: boolean;
    addresses?: Array<{ address: string; family: number }>;
    error?: string;
    hasIPv6?: boolean;
  };
  tcp?: {
    success: boolean;
    address?: string;
    port?: number;
    error?: string;
  };
  tls?: {
    success: boolean;
    statusCode?: number;
    headers?: Record<string, string>;
    error?: string;
  };
}

/**
 * Run network diagnostics for API host
 */
async function runNetworkDiagnostics(apiUrl: string, logger?: Logger): Promise<DiagnosticResult> {
  const result: DiagnosticResult = {};

  try {
    const url = new URL(apiUrl);
    const host = url.hostname;
    const port = url.port ? parseInt(url.port, 10) : 443;
    const path = url.pathname;

    // DNS lookup - get all addresses (both IPv4 and IPv6)
    try {
      const addresses = await dnsPromises.lookup(host, { all: true });
      const hasIPv6 = addresses.some((addr) => addr.family === 6);
      result.dns = {
        success: true,
        addresses: addresses.map((a) => ({ address: a.address, family: a.family })),
        hasIPv6,
      };
      logger?.debug('DNS lookup succeeded', {
        module: 'pdf_vl_ocr',
        stage: 'diagnostics',
        host,
        addresses: result.dns.addresses,
        hasIPv6,
      });
    } catch (error) {
      result.dns = {
        success: false,
        error: error instanceof Error ? error.message : String(error),
      };
      logger?.warn('DNS lookup failed', {
        module: 'pdf_vl_ocr',
        stage: 'diagnostics',
        host,
        error: result.dns.error,
      });
    }

    // TCP connection test - use IPv4 address if available
    if (result.dns?.success && result.dns.addresses && result.dns.addresses.length > 0) {
      // Prefer IPv4 address
      const ipv4Addr = result.dns.addresses.find((a) => a.family === 4);
      const targetAddr = ipv4Addr || result.dns.addresses[0];

      try {
        await new Promise<void>((resolve, reject) => {
          const socket = net.connect(
            {
              host: targetAddr.address,
              port,
              timeout: 3000,
            },
            () => {
              socket.end();
              resolve();
            },
          );
          socket.on('error', reject);
          socket.on('timeout', () => {
            socket.destroy();
            reject(new Error('Connection timeout'));
          });
        });
        result.tcp = {
          success: true,
          address: targetAddr.address,
          port,
        };
        logger?.debug('TCP connection succeeded', {
          module: 'pdf_vl_ocr',
          stage: 'diagnostics',
          address: targetAddr.address,
          port,
        });
      } catch (error) {
        result.tcp = {
          success: false,
          address: targetAddr.address,
          port,
          error: error instanceof Error ? error.message : String(error),
        };
        logger?.warn('TCP connection failed', {
          module: 'pdf_vl_ocr',
          stage: 'diagnostics',
          address: targetAddr.address,
          port,
          error: result.tcp.error,
        });
      }
    }

    // TLS handshake test - test the actual API path
    if (result.tcp?.success && url.protocol === 'https:') {
      try {
        await new Promise<void>((resolve, reject) => {
          const req = https.request(
            {
              method: 'HEAD',
              hostname: host,
              port,
              path, // Use actual API path
              timeout: 3000,
            },
            (res) => {
              // Capture response metadata
              result.tls = {
                success: true,
                statusCode: res.statusCode,
                headers: {
                  location: res.headers.location || '',
                  server: Array.isArray(res.headers.server)
                    ? res.headers.server.join(', ')
                    : res.headers.server || '',
                  via: Array.isArray(res.headers.via)
                    ? res.headers.via.join(', ')
                    : res.headers.via || '',
                  'content-type': Array.isArray(res.headers['content-type'])
                    ? res.headers['content-type'].join(', ')
                    : res.headers['content-type'] || '',
                },
              };
              res.resume();
              resolve();
            },
          );
          req.on('error', reject);
          req.on('timeout', () => {
            req.destroy();
            reject(new Error('TLS handshake timeout'));
          });
          req.end();
        });
        logger?.debug('TLS handshake succeeded', {
          module: 'pdf_vl_ocr',
          stage: 'diagnostics',
          host,
          path,
          statusCode: result.tls?.statusCode,
          headers: result.tls?.headers,
        });
      } catch (error) {
        result.tls = {
          success: false,
          error: error instanceof Error ? error.message : String(error),
        };
        logger?.warn('TLS handshake failed', {
          module: 'pdf_vl_ocr',
          stage: 'diagnostics',
          host,
          path,
          error: result.tls.error,
        });
      }
    }
  } catch (error) {
    logger?.warn('Diagnostics failed', {
      module: 'pdf_vl_ocr',
      stage: 'diagnostics',
      error: error instanceof Error ? error.message : String(error),
    });
  }

  return result;
}

/**
 * Check if error is retryable
 */
function isRetryableError(error: any, statusCode: number): boolean {
  // Network errors
  const retryableErrorCodes = [
    'ECONNRESET',
    'ETIMEDOUT',
    'ENOTFOUND',
    'EAI_AGAIN',
    'ECONNREFUSED',
    'EPIPE',
    'EHOSTUNREACH',
    'ENETUNREACH',
  ];

  if (error?.code && retryableErrorCodes.includes(error.code)) {
    return true;
  }

  if (error?.cause?.code && retryableErrorCodes.includes(error.cause.code)) {
    return true;
  }

  // If error is OcrApiError, check its cause
  if (error instanceof OcrApiError && error.cause) {
    const cause = error.cause as any;
    if (cause.code && retryableErrorCodes.includes(cause.code)) {
      return true;
    }
    // Check nested cause
    if (cause.cause?.code && retryableErrorCodes.includes(cause.cause.code)) {
      return true;
    }
  }

  // HTTP status codes
  const retryableStatusCodes = [429, 502, 503, 504];
  if (retryableStatusCodes.includes(statusCode)) {
    return true;
  }

  // Timeout errors
  if (error?.name === 'AbortError' || error?.message?.includes('abort')) {
    return true;
  }

  return false;
}

/**
 * Calculate exponential backoff delay with jitter
 */
function calculateBackoff(attempt: number): number {
  const baseDelays = [500, 1500, 3500];
  const baseDelay = baseDelays[Math.min(attempt, baseDelays.length - 1)];
  const jitter = Math.random() * 200 - 100; // ±100ms jitter
  return baseDelay + jitter;
}

/**
 * IP override configuration for fetch requests
 */
interface IpOverrideConfig {
  enabled: boolean;
  ip?: string;
  ipVersion?: 4 | 6; // IP version: 4 for IPv4, 6 for IPv6
  state: 'missing' | 'invalid' | 'enabled';
  dispatcher?: Agent; // Only present when enabled=true
}

/**
 * Validate and prepare IP override configuration
 * Returns configuration that indicates whether IP override should be used
 * Creates undici dispatcher only when IP is valid
 *
 * @param overrideIp - The IP address from configuration (may be undefined/empty/invalid)
 * @param connectTimeoutMs - Connection timeout in milliseconds
 * @param logger - Optional logger for diagnostics
 * @returns IP override configuration indicating whether to use custom dispatcher
 */
function prepareIpOverride(
  overrideIp: string | undefined,
  connectTimeoutMs: number,
  logger?: Logger,
): IpOverrideConfig {
  // No IP provided - use system DNS
  if (!overrideIp || overrideIp.trim() === '') {
    return {
      enabled: false,
      state: 'missing',
    };
  }

  const ip = overrideIp.trim();

  // Validate IP address format using Node's net.isIP
  // Returns 0 if invalid, 4 for IPv4, 6 for IPv6
  const ipVersion = net.isIP(ip);
  if (ipVersion === 0) {
    // Invalid IP - log warning and fall back to system DNS
    logger?.warn('Invalid IP override configuration, falling back to system DNS', {
      module: 'pdf_vl_ocr',
      stage: 'network_config',
      overrideIp: ip,
      reason: 'net.isIP returned 0 (invalid IP format)',
    });
    return {
      enabled: false,
      ip,
      state: 'invalid',
    };
  }

  // Valid IP address - create dispatcher with custom lookup
  logger?.debug('IP override enabled, creating custom dispatcher', {
    module: 'pdf_vl_ocr',
    stage: 'network_config',
    ip,
    ipVersion: ipVersion === 4 ? 'IPv4' : 'IPv6',
  });

  // Create undici Agent with custom lookup that returns the override IP
  const agentConfig: any = {
    connect: {
      timeout: connectTimeoutMs,
      lookup: (_hostname: string, _options: any, callback: any) => {
        // Return the override IP directly, bypassing DNS resolution
        // undici expects callback(error, address, family)
        callback(null, ip, ipVersion);
      },
    },
  };

  const dispatcher = new Agent(agentConfig);

  return {
    enabled: true,
    ip,
    ipVersion: ipVersion as 4 | 6,
    state: 'enabled',
    dispatcher,
  };
}

/**
 * Extract detailed error information from fetch error
 * Ensures no empty error={} in logs by extracting all available fields
 */
function extractErrorDetails(error: any): Record<string, any> {
  const details: Record<string, any> = {
    name: error?.name || 'Error',
    message: error?.message || String(error),
  };

  if (error?.stack) {
    details.stack = error.stack.split('\n').slice(0, 3).join('\n');
  }

  // Extract cause details (important for network errors)
  if (error?.cause) {
    const cause = error.cause;
    details.cause = {};

    // Extract all standard error fields
    if (cause.name) details.cause.name = cause.name;
    if (cause.message) details.cause.message = cause.message;
    if (cause.code) details.cause.code = cause.code;
    if (cause.errno !== undefined) details.cause.errno = cause.errno;
    if (cause.syscall) details.cause.syscall = cause.syscall;
    if (cause.address) details.cause.address = cause.address;
    if (cause.port !== undefined) details.cause.port = cause.port;
    if (cause.hostname) details.cause.hostname = cause.hostname;

    // Remove empty cause object
    if (Object.keys(details.cause).length === 0) {
      delete details.cause;
    }
  }

  // Direct error fields (for cases where cause is not used)
  if (error?.code) {
    details.code = error.code;
  }

  if (error?.errno !== undefined) {
    details.errno = error.errno;
  }

  if (error?.syscall) {
    details.syscall = error.syscall;
  }

  if (error?.address) {
    details.address = error.address;
  }

  if (error?.port !== undefined) {
    details.port = error.port;
  }

  return details;
}

/**
 * Call PaddleOCR-VL API with a single attempt (no retry logic)
 */
async function callPaddleOcrVlOnce(
  pdfBuffer: Buffer,
  apiUrl: string,
  token: string,
  config: OcrConfig,
  attempt: number,
  ocrJobId: string,
  logger?: Logger,
): Promise<PaddleOcrVlResult> {
  const base64Pdf = pdfBuffer.toString('base64');

  // Check PDF size limit (25MB)
  const maxSizeBytes = 25 * 1024 * 1024;
  if (pdfBuffer.length > maxSizeBytes) {
    throw new OcrApiError(
      `PDF size exceeds 25MB limit (${Math.round(pdfBuffer.length / 1024 / 1024)}MB)`,
      0,
      undefined,
      undefined,
    );
  }

  // Prepare request payload (consistent with official documentation)
  let payload: string;
  try {
    const payloadObj = {
      file: base64Pdf, // Base64-encoded PDF content
      fileType: 0, // 0 = PDF, 1 = image (as per official docs)
      useDocOrientationClassify: false, // Optional: document orientation detection
      useDocUnwarping: false, // Optional: document unwarping
      useChartRecognition: false, // Optional: chart/table recognition
    };
    payload = JSON.stringify(payloadObj);
  } catch (error) {
    throw new OcrApiError(
      `Failed to serialize request payload: ${error instanceof Error ? error.message : String(error)}`,
      0,
      undefined,
      undefined,
      error instanceof Error ? error : undefined,
    );
  }

  // Create abort controller for total timeout
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), config.timeoutMs);

  // Prepare IP override configuration (creates dispatcher only when IP is valid)
  const ipOverride = prepareIpOverride(config.overrideIp, config.connectTimeoutMs, logger);

  // Check for proxy environment variables (any proxy-related var indicates proxy may be in use)
  // Note: NO_PROXY indicates which hosts bypass the proxy, suggesting proxy is configured
  const hasHttpProxy = !!(process.env.HTTP_PROXY || process.env.http_proxy);
  const hasHttpsProxy = !!(process.env.HTTPS_PROXY || process.env.https_proxy);
  const hasNoProxy = !!(process.env.NO_PROXY || process.env.no_proxy);

  // Parse API URL for logging
  const apiUrlObj = new URL(apiUrl);

  // B. OCR Request Preparation - Log before fetch
  logger?.info('OCR request starting', {
    module: 'pdf_vl_ocr',
    event: 'ocr-request-prepare',
    ocrJobId,
    attempt,
    endpointHost: apiUrlObj.host,
    endpointPath: apiUrlObj.pathname,
    timeoutMs: config.timeoutMs,
    retryPolicy: {
      maxAttempts: config.retries + 1,
      backoff: 'exponential',
    },
    requestPayloadMeta: {
      sendPdfAs: 'base64',
      payloadBytesApprox: payload.length,
      pdfSizeKb: Math.round(pdfBuffer.length / 1024),
    },
    networkMeta: {
      overrideIpState: ipOverride.state,
      proxyPresent: {
        http: hasHttpProxy,
        https: hasHttpsProxy,
        noProxy: hasNoProxy,
      },
      userAgent: 'blog-content-import/1.0',
      dnsStrategy: ipOverride.enabled ? 'customLookup' : 'systemDefault',
    },
  });

  let response: Response;
  let responseText: string = '';
  const requestStartTime = Date.now();

  try {
    // Prepare fetch options
    const fetchOptions: RequestInit = {
      method: 'POST',
      headers: {
        // Authorization format as per official documentation
        Authorization: `token ${token}`,
        'Content-Type': 'application/json', // Required by API
        Accept: 'application/json',
        'User-Agent': 'blog-content-import/1.0',
      },
      body: payload,
      signal: controller.signal,
    };

    // Only use custom dispatcher when IP override is enabled
    // When disabled (missing/invalid), use system default DNS
    if (ipOverride.enabled && ipOverride.dispatcher) {
      // @ts-ignore - undici dispatcher is valid but TypeScript doesn't recognize it
      fetchOptions.dispatcher = ipOverride.dispatcher;
    }

    // Make API request
    response = await fetch(apiUrl, fetchOptions);

    const fetchDuration = Date.now() - requestStartTime;

    responseText = await response.text();

    logger?.debug('OCR API response received', {
      module: 'pdf_vl_ocr',
      stage: 'response',
      statusCode: response.status,
      responseLength: responseText.length,
      durationMs: fetchDuration,
    });
  } catch (error) {
    const errorDetails = extractErrorDetails(error);
    const elapsedMs = Date.now() - requestStartTime;
    const errorCode = classifyOcrError(error, 0);

    // C. OCR Request Result - Failure (network exception)
    logger?.error('OCR request failed', {
      module: 'pdf_vl_ocr',
      event: 'ocr-request-fail',
      ocrJobId,
      attempt,
      statusCode: 0,
      errorCode,
      errName: errorDetails.name,
      errMessage: errorDetails.message,
      undiciCauseName: errorDetails.cause?.name,
      undiciCauseCode: errorDetails.cause?.code,
      causeSummary: errorDetails.cause
        ? `${errorDetails.cause.name || 'Error'}: ${errorDetails.cause.code || errorDetails.cause.message || 'unknown'}`
        : undefined,
      endpointHost: apiUrlObj.host,
      elapsedMs,
      shouldRetry: isRetryableError(error, 0) && attempt <= config.retries,
      networkMeta: {
        overrideIpState: ipOverride.state,
        proxyPresent: { http: hasHttpProxy, https: hasHttpsProxy, noProxy: hasNoProxy },
      },
    });

    const errorMsg = error instanceof Error ? error.message : String(error);
    // Truncate error details to <= 2KB for responseBody
    const errorDetailsStr = JSON.stringify(errorDetails);
    const truncatedDetails =
      errorDetailsStr.length > 2048
        ? errorDetailsStr.slice(0, 2048) + '...(truncated)'
        : errorDetailsStr;

    throw new OcrApiError(
      `Failed to call PaddleOCR-VL API: ${errorMsg}`,
      0,
      truncatedDetails,
      undefined,
      error instanceof Error ? error : undefined,
    );
  } finally {
    clearTimeout(timeoutId);
    // Clean up dispatcher if it was created
    if (ipOverride.dispatcher) {
      ipOverride.dispatcher.destroy();
    }
  }

  const elapsedMs = Date.now() - requestStartTime;

  // Check HTTP status
  if (!response.ok) {
    const errorCode = classifyOcrError(null, response.status);

    // Sanitize response body for logging (truncate to 2KB max)
    const bodySample =
      responseText.length > 2048 ? responseText.slice(0, 2048) + '...(truncated)' : responseText;

    // C. OCR Request Result - HTTP non-2xx
    logger?.error('OCR request failed with HTTP error', {
      module: 'pdf_vl_ocr',
      event: 'ocr-request-fail',
      ocrJobId,
      attempt,
      statusCode: response.status,
      errorCode,
      errMessage: `HTTP ${response.status} ${response.statusText}`,
      endpointHost: apiUrlObj.host,
      elapsedMs,
      shouldRetry: isRetryableError(null, response.status) && attempt <= config.retries,
      responseBodySnippet: bodySample.slice(0, 200),
    });

    throw new OcrApiError(
      `PaddleOCR-VL API returned error: HTTP ${response.status} ${response.statusText}`,
      response.status,
      bodySample,
    );
  }

  // Parse response
  let parsedResponse: PaddleOcrVlResponse;
  try {
    parsedResponse = JSON.parse(responseText);
  } catch (error) {
    const errorCode = classifyOcrError(error, response.status);
    const bodySample =
      responseText.length > 500 ? responseText.slice(0, 500) + '...' : responseText;

    // C. OCR Request Result - Parse failure
    logger?.error('OCR response parse failed', {
      module: 'pdf_vl_ocr',
      event: 'ocr-request-fail',
      ocrJobId,
      attempt,
      statusCode: response.status,
      errorCode,
      errName: 'OcrParseError',
      errMessage: error instanceof Error ? error.message : String(error),
      endpointHost: apiUrlObj.host,
      elapsedMs,
      shouldRetry: false,
    });

    throw new OcrParseError(
      `Failed to parse PaddleOCR-VL API response as JSON: ${error instanceof Error ? error.message : String(error)}`,
      bodySample,
    );
  }

  // Check for API-level errors
  if (parsedResponse.error) {
    const errorCode = classifyOcrError(parsedResponse, response.status);

    logger?.error('OCR API returned error', {
      module: 'pdf_vl_ocr',
      event: 'ocr-request-fail',
      ocrJobId,
      attempt,
      statusCode: response.status,
      errorCode,
      errMessage: parsedResponse.error,
      endpointHost: apiUrlObj.host,
      elapsedMs,
      shouldRetry: false,
    });

    throw new OcrApiError(
      `PaddleOCR-VL API error: ${parsedResponse.error}`,
      response.status,
      parsedResponse.message,
    );
  }

  // Extract result
  const layoutResults = parsedResponse.result?.layoutParsingResults;
  if (!layoutResults || layoutResults.length === 0) {
    const errorCode = classifyOcrError(new Error('empty layoutParsingResults'), response.status);

    logger?.error('OCR response empty', {
      module: 'pdf_vl_ocr',
      event: 'ocr-request-fail',
      ocrJobId,
      attempt,
      statusCode: response.status,
      errorCode,
      errMessage: 'Empty layoutParsingResults',
      endpointHost: apiUrlObj.host,
      elapsedMs,
      shouldRetry: false,
    });

    throw new OcrParseError(
      'PaddleOCR-VL API returned empty layoutParsingResults',
      JSON.stringify(parsedResponse).slice(0, 500),
    );
  }

  const firstResult = layoutResults[0];
  const markdown = firstResult.markdown?.text;
  const images = firstResult.markdown?.images;

  if (!markdown) {
    const errorCode = classifyOcrError(new Error('no markdown text'), response.status);

    logger?.error('OCR result missing markdown', {
      module: 'pdf_vl_ocr',
      event: 'ocr-request-fail',
      ocrJobId,
      attempt,
      statusCode: response.status,
      errorCode,
      errMessage: 'No markdown text in result',
      endpointHost: apiUrlObj.host,
      elapsedMs,
      shouldRetry: false,
    });

    throw new OcrParseError(
      'PaddleOCR-VL API did not return markdown text',
      JSON.stringify(firstResult).slice(0, 500),
    );
  }

  // C. OCR Request Result - Success
  logger?.info('OCR request succeeded', {
    module: 'pdf_vl_ocr',
    event: 'ocr-request-success',
    ocrJobId,
    attempt,
    statusCode: response.status,
    requestId: undefined, // API doesn't provide request ID
    elapsedMs,
    responseMeta: {
      markdownChars: markdown.length,
      imagesCount: images ? Object.keys(images).length : 0,
      outputImagesCount: firstResult.outputImages?.length || 0,
    },
  });

  return {
    markdown,
    images: images || {},
    outputImages: firstResult.outputImages,
  };
}

/**
 * Call PaddleOCR-VL API to parse PDF (with retry logic)
 */
export async function callPaddleOcrVl(
  pdfBuffer: Buffer,
  apiUrl: string,
  token: string,
  logger?: Logger,
  ocrJobId?: string,
): Promise<PaddleOcrVlResult> {
  const config = getOcrConfig();
  const jobId = ocrJobId || generateOcrJobId();

  // Validate API URL first
  validateApiUrl(apiUrl);

  // Log environment info
  logger?.info('OCR API call starting', {
    module: 'pdf_vl_ocr',
    stage: 'init',
    ocrJobId: jobId,
    apiUrlDomain: new URL(apiUrl).hostname,
    nodeVersion: process.version,
    platform: process.platform,
    pdfSizeKb: Math.round(pdfBuffer.length / 1024),
    config: {
      retries: config.retries,
      timeoutMs: config.timeoutMs,
      connectTimeoutMs: config.connectTimeoutMs,
      enableDiag: config.enableDiag,
    },
  });

  // Run diagnostics if enabled
  if (config.enableDiag) {
    logger?.info('Running network diagnostics', {
      module: 'pdf_vl_ocr',
      stage: 'diagnostics',
      ocrJobId: jobId,
    });
    const diagResult = await runNetworkDiagnostics(apiUrl, logger);
    logger?.info('Network diagnostics complete', {
      module: 'pdf_vl_ocr',
      stage: 'diagnostics',
      ocrJobId: jobId,
      result: diagResult,
    });
  }

  // Retry loop
  let lastError: Error | undefined;
  let lastStatusCode = 0;
  let retryCount = 0;

  for (let attempt = 1; attempt <= config.retries + 1; attempt++) {
    if (attempt > 1) {
      retryCount++;
      const delay = calculateBackoff(attempt - 2);
      const nextBackoffMs = Math.round(delay);

      // Extract error details for retry logging
      const errorInfo = lastError ? extractErrorDetails(lastError) : {};
      const errorCode = lastError ? classifyOcrError(lastError, lastStatusCode) : 'OCR_UNKNOWN';

      logger?.info('Retrying OCR API call', {
        module: 'pdf_vl_ocr',
        stage: 'retry',
        ocrJobId: jobId,
        attempt,
        maxAttempts: config.retries + 1,
        delayMs: nextBackoffMs,
        nextBackoffMs,
        lastErrorCode: errorCode,
        lastCauseCode: errorInfo.code || errorInfo.cause?.code,
        lastErrorMessage: errorInfo.message,
      });
      await new Promise((resolve) => setTimeout(resolve, delay));
    }

    try {
      return await callPaddleOcrVlOnce(pdfBuffer, apiUrl, token, config, attempt, jobId, logger);
    } catch (error) {
      lastError = error instanceof Error ? error : new Error(String(error));

      // Extract status code for retry decision
      if (error instanceof OcrApiError) {
        lastStatusCode = error.statusCode;
      }

      // Check if error is retryable
      const shouldRetry = attempt <= config.retries && isRetryableError(error, lastStatusCode);

      if (!shouldRetry) {
        // Log final failure
        const errorDetails = extractErrorDetails(error);
        const errorCode = classifyOcrError(error, lastStatusCode);
        logger?.error('OCR API call failed permanently', {
          module: 'pdf_vl_ocr',
          stage: 'final_error',
          ocrJobId: jobId,
          attempt,
          maxAttempts: config.retries + 1,
          statusCode: lastStatusCode,
          errorCode,
          error: errorDetails,
        });
        throw error;
      }
    }
  }

  // All retries exhausted
  throw lastError || new Error('OCR API call failed after all retries');
}

/**
 * Call local mock OCR provider (for testing)
 */
export async function callLocalMockOcr(logger?: Logger): Promise<PaddleOcrVlResult> {
  logger?.info('Using local mock OCR provider', {
    module: 'pdf_vl_ocr',
    stage: 'mock',
  });

  try {
    // Load mock fixture
    const fs = await import('fs/promises');
    const path = await import('path');
    const fixtureDir = path.join(process.cwd(), 'tests/fixtures/ocr');
    const fixturePath = path.join(fixtureDir, 'paddle_mock.json');

    const fixtureData = await fs.readFile(fixturePath, 'utf-8');
    const mockResponse: PaddleOcrVlResponse = JSON.parse(fixtureData);

    // Extract result (same logic as real API)
    const layoutResults = mockResponse.result?.layoutParsingResults;
    if (!layoutResults || layoutResults.length === 0) {
      throw new OcrParseError('Mock fixture has empty layoutParsingResults');
    }

    const firstResult = layoutResults[0];
    const markdown = firstResult.markdown?.text;
    const images = firstResult.markdown?.images;

    if (!markdown) {
      throw new OcrParseError('Mock fixture does not contain markdown text');
    }

    logger?.info('Mock OCR successful', {
      module: 'pdf_vl_ocr',
      stage: 'mock',
      markdownLength: markdown.length,
      imageCount: images ? Object.keys(images).length : 0,
    });

    return {
      markdown,
      images: images || {},
      outputImages: firstResult.outputImages,
    };
  } catch (error) {
    logger?.error('Mock OCR failed', {
      module: 'pdf_vl_ocr',
      stage: 'mock',
      error: error instanceof Error ? error.message : String(error),
    });
    throw error;
  }
}
