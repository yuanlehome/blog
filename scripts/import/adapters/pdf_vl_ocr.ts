/**
 * PaddleOCR-VL API Client
 *
 * Handles communication with PaddleOCR-VL layout parsing API
 */

import type { Logger } from '../../logger/types.js';
import * as dns from 'dns/promises';
import * as net from 'net';
import * as https from 'https';

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
 * Configuration for OCR API calls
 */
interface OcrConfig {
  retries: number;
  timeoutMs: number;
  connectTimeoutMs: number;
  enableDiag: boolean;
}

/**
 * Get OCR configuration from environment
 */
function getOcrConfig(): OcrConfig {
  return {
    retries: parseInt(process.env.PDF_OCR_RETRY || '3', 10),
    timeoutMs: parseInt(process.env.PDF_OCR_TIMEOUT_MS || '90000', 10),
    connectTimeoutMs: parseInt(process.env.PDF_OCR_CONNECT_TIMEOUT_MS || '15000', 10),
    enableDiag: process.env.PDF_OCR_DIAG === '1' || process.env.CI === 'true',
  };
}

/**
 * Network diagnostic results
 */
interface DiagnosticResult {
  dns?: { success: boolean; address?: string; error?: string };
  tcp?: { success: boolean; error?: string };
  tls?: { success: boolean; error?: string };
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

    // DNS lookup
    try {
      const addresses = await dns.lookup(host);
      result.dns = { success: true, address: addresses.address };
      logger?.debug('DNS lookup succeeded', {
        module: 'pdf_vl_ocr',
        stage: 'diagnostics',
        host,
        address: addresses.address,
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

    // TCP connection test
    if (result.dns?.success) {
      try {
        await new Promise<void>((resolve, reject) => {
          const socket = net.connect({ host, port, timeout: 3000 }, () => {
            socket.end();
            resolve();
          });
          socket.on('error', reject);
          socket.on('timeout', () => {
            socket.destroy();
            reject(new Error('Connection timeout'));
          });
        });
        result.tcp = { success: true };
        logger?.debug('TCP connection succeeded', {
          module: 'pdf_vl_ocr',
          stage: 'diagnostics',
          host,
          port,
        });
      } catch (error) {
        result.tcp = {
          success: false,
          error: error instanceof Error ? error.message : String(error),
        };
        logger?.warn('TCP connection failed', {
          module: 'pdf_vl_ocr',
          stage: 'diagnostics',
          host,
          port,
          error: result.tcp.error,
        });
      }
    }

    // TLS handshake test (HEAD request)
    if (result.tcp?.success && url.protocol === 'https:') {
      try {
        await new Promise<void>((resolve, reject) => {
          const req = https.request(
            { method: 'HEAD', hostname: host, port, path: '/', timeout: 3000 },
            (res) => {
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
        result.tls = { success: true };
        logger?.debug('TLS handshake succeeded', {
          module: 'pdf_vl_ocr',
          stage: 'diagnostics',
          host,
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
 * Extract detailed error information from fetch error
 */
function extractErrorDetails(error: any): Record<string, any> {
  const details: Record<string, any> = {
    name: error?.name || 'Error',
    message: error?.message || String(error),
  };

  if (error?.stack) {
    details.stack = error.stack.split('\n').slice(0, 3).join('\n');
  }

  if (error?.cause) {
    details.cause = {
      name: error.cause.name,
      message: error.cause.message,
      code: error.cause.code,
      errno: error.cause.errno,
      syscall: error.cause.syscall,
      address: error.cause.address,
      port: error.cause.port,
    };
  }

  if (error?.code) {
    details.code = error.code;
  }

  if (error?.errno) {
    details.errno = error.errno;
  }

  if (error?.syscall) {
    details.syscall = error.syscall;
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
  logger?: Logger,
): Promise<PaddleOcrVlResult> {
  const base64Pdf = pdfBuffer.toString('base64');

  logger?.debug('Preparing OCR request', {
    module: 'pdf_vl_ocr',
    stage: 'prepare',
    base64Length: base64Pdf.length,
    base64Bytes: base64Pdf.length,
    pdfSizeKb: Math.round(pdfBuffer.length / 1024),
  });

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

  // Prepare request payload
  let payload: string;
  try {
    const payloadObj = {
      file: base64Pdf,
      fileType: 0, // 0 = PDF, 1 = image
      useDocOrientationClassify: false,
      useDocUnwarping: false,
      useChartRecognition: false,
    };
    payload = JSON.stringify(payloadObj);
    logger?.debug('Payload prepared', {
      module: 'pdf_vl_ocr',
      stage: 'prepare',
      payloadBytes: payload.length,
    });
  } catch (error) {
    throw new OcrApiError(
      `Failed to serialize request payload: ${error instanceof Error ? error.message : String(error)}`,
      0,
      undefined,
      undefined,
      error instanceof Error ? error : undefined,
    );
  }

  // Create abort controller for timeout
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), config.timeoutMs);

  let response: Response;
  let responseText: string = '';

  try {
    // Make API request
    const startTime = Date.now();
    response = await fetch(apiUrl, {
      method: 'POST',
      headers: {
        Authorization: `token ${token}`,
        'Content-Type': 'application/json',
        Accept: 'application/json',
        'User-Agent': 'blog-content-import/1.0',
      },
      body: payload,
      signal: controller.signal,
    });

    const fetchDuration = Date.now() - startTime;

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

    logger?.error('OCR API request failed', {
      module: 'pdf_vl_ocr',
      stage: 'request',
      ...errorDetails,
    });

    const errorMsg = error instanceof Error ? error.message : String(error);
    throw new OcrApiError(
      `Failed to call PaddleOCR-VL API: ${errorMsg}`,
      0,
      JSON.stringify(errorDetails),
      undefined,
      error instanceof Error ? error : undefined,
    );
  } finally {
    clearTimeout(timeoutId);
  }

  // Check HTTP status
  if (!response.ok) {
    // Sanitize response body for logging (truncate if too long)
    const bodySample =
      responseText.length > 2048 ? responseText.slice(0, 2048) + '...' : responseText;

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
    const bodySample =
      responseText.length > 500 ? responseText.slice(0, 500) + '...' : responseText;
    throw new OcrParseError(
      `Failed to parse PaddleOCR-VL API response as JSON: ${error instanceof Error ? error.message : String(error)}`,
      bodySample,
    );
  }

  // Check for API-level errors
  if (parsedResponse.error) {
    throw new OcrApiError(
      `PaddleOCR-VL API error: ${parsedResponse.error}`,
      response.status,
      parsedResponse.message,
    );
  }

  // Extract result
  const layoutResults = parsedResponse.result?.layoutParsingResults;
  if (!layoutResults || layoutResults.length === 0) {
    throw new OcrParseError(
      'PaddleOCR-VL API returned empty layoutParsingResults',
      JSON.stringify(parsedResponse).slice(0, 500),
    );
  }

  const firstResult = layoutResults[0];
  const markdown = firstResult.markdown?.text;
  const images = firstResult.markdown?.images;

  if (!markdown) {
    throw new OcrParseError(
      'PaddleOCR-VL API did not return markdown text',
      JSON.stringify(firstResult).slice(0, 500),
    );
  }

  logger?.info('OCR parsing successful', {
    module: 'pdf_vl_ocr',
    stage: 'success',
    markdownLength: markdown.length,
    imageCount: images ? Object.keys(images).length : 0,
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
): Promise<PaddleOcrVlResult> {
  const config = getOcrConfig();

  // Log environment info
  logger?.info('OCR API call starting', {
    module: 'pdf_vl_ocr',
    stage: 'init',
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
    });
    const diagResult = await runNetworkDiagnostics(apiUrl, logger);
    logger?.info('Network diagnostics complete', {
      module: 'pdf_vl_ocr',
      stage: 'diagnostics',
      result: diagResult,
    });
  }

  // Retry loop
  let lastError: Error | undefined;
  let lastStatusCode = 0;

  for (let attempt = 0; attempt <= config.retries; attempt++) {
    if (attempt > 0) {
      const delay = calculateBackoff(attempt - 1);
      logger?.info('Retrying OCR API call', {
        module: 'pdf_vl_ocr',
        stage: 'retry',
        attempt,
        maxRetries: config.retries,
        delayMs: Math.round(delay),
      });
      await new Promise((resolve) => setTimeout(resolve, delay));
    }

    try {
      return await callPaddleOcrVlOnce(pdfBuffer, apiUrl, token, config, logger);
    } catch (error) {
      lastError = error instanceof Error ? error : new Error(String(error));

      // Extract status code for retry decision
      if (error instanceof OcrApiError) {
        lastStatusCode = error.statusCode;
      }

      // Check if error is retryable
      const shouldRetry = attempt < config.retries && isRetryableError(error, lastStatusCode);

      logger?.warn('OCR API call failed', {
        module: 'pdf_vl_ocr',
        stage: 'error',
        attempt: attempt + 1,
        maxAttempts: config.retries + 1,
        shouldRetry,
        error: extractErrorDetails(error),
      });

      if (!shouldRetry) {
        // Log 4xx errors with response body snippet
        if (lastStatusCode >= 400 && lastStatusCode < 500 && error instanceof OcrApiError) {
          logger?.error('Client error from OCR API', {
            module: 'pdf_vl_ocr',
            stage: 'client_error',
            statusCode: lastStatusCode,
            responseBodySnippet: error.responseBody,
          });
        }
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
