/**
 * PDF Download and Validation Utilities
 *
 * Handles downloading PDF files with retries and validation
 */

import type { Logger } from '../../logger/types.js';

const PDF_MIN_SIZE_BYTES = 50 * 1024; // 50KB minimum
const PDF_MAGIC_BYTES = '%PDF-';
const DOWNLOAD_TIMEOUT_MS = 60000; // 60 seconds
const MAX_RETRIES = 3;

/**
 * Custom error classes for better error handling
 */
export class PdfDownloadError extends Error {
  constructor(
    message: string,
    public url: string,
    public statusCode?: number,
    public contentType?: string,
  ) {
    super(message);
    this.name = 'PdfDownloadError';
  }
}

export class NotPdfError extends Error {
  constructor(
    message: string,
    public contentType?: string,
    public actualHeader?: string,
  ) {
    super(message);
    this.name = 'NotPdfError';
  }
}

/**
 * Exponential backoff delay
 */
function getBackoffDelay(attempt: number): number {
  const baseDelay = 1000; // 1 second
  const maxDelay = 10000; // 10 seconds
  const delay = baseDelay * Math.pow(2, attempt);
  return Math.min(delay, maxDelay);
}

/**
 * Sleep for specified milliseconds
 */
function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * Download PDF with retries and validation
 */
export async function downloadPdf(
  url: string,
  maxSizeMb: number,
  logger?: Logger,
): Promise<Buffer> {
  let lastError: Error | null = null;

  for (let attempt = 0; attempt < MAX_RETRIES; attempt++) {
    try {
      if (attempt > 0) {
        const delay = getBackoffDelay(attempt - 1);
        logger?.debug('Retrying PDF download', {
          module: 'pdf_vl',
          attempt: attempt + 1,
          maxRetries: MAX_RETRIES,
          delayMs: delay,
        });
        await sleep(delay);
      }

      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), DOWNLOAD_TIMEOUT_MS);

      try {
        const response = await fetch(url, {
          signal: controller.signal,
          redirect: 'follow', // Follow redirects
          headers: {
            'User-Agent':
              'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
          },
        });

        clearTimeout(timeout);

        if (!response.ok) {
          throw new PdfDownloadError(
            `Failed to download PDF: HTTP ${response.status} ${response.statusText}`,
            url,
            response.status,
            response.headers.get('content-type') || undefined,
          );
        }

        const contentType = response.headers.get('content-type') || '';
        const contentLength = response.headers.get('content-length');

        // Check content length if available
        if (contentLength) {
          const sizeBytes = parseInt(contentLength, 10);
          const sizeMb = sizeBytes / (1024 * 1024);

          if (sizeMb > maxSizeMb) {
            throw new PdfDownloadError(
              `PDF file too large: ${sizeMb.toFixed(2)}MB exceeds limit of ${maxSizeMb}MB`,
              url,
              response.status,
              contentType,
            );
          }
        }

        // Download content
        const arrayBuffer = await response.arrayBuffer();
        const buffer = Buffer.from(arrayBuffer);

        // Validate size
        if (buffer.length < PDF_MIN_SIZE_BYTES) {
          throw new NotPdfError(
            `File too small: ${buffer.length} bytes (minimum ${PDF_MIN_SIZE_BYTES} bytes)`,
            contentType,
          );
        }

        const sizeMb = buffer.length / (1024 * 1024);
        if (sizeMb > maxSizeMb) {
          throw new PdfDownloadError(
            `PDF file too large: ${sizeMb.toFixed(2)}MB exceeds limit of ${maxSizeMb}MB`,
            url,
            response.status,
            contentType,
          );
        }

        logger?.debug('PDF downloaded successfully', {
          module: 'pdf_vl',
          url,
          sizeBytes: buffer.length,
          contentType,
          attempt: attempt + 1,
        });

        return buffer;
      } finally {
        clearTimeout(timeout);
      }
    } catch (error) {
      lastError = error instanceof Error ? error : new Error(String(error));

      // Don't retry on validation errors
      if (error instanceof NotPdfError || error instanceof PdfDownloadError) {
        // Don't retry 4xx errors (client errors - won't succeed on retry)
        if (error instanceof PdfDownloadError && error.statusCode && error.statusCode >= 400) {
          throw error;
        }
        // NotPdfError means the content is invalid, no point retrying
        if (error instanceof NotPdfError) {
          throw error;
        }
        // For other PdfDownloadError (network, timeout, 5xx), continue to retry
      }

      logger?.warn('PDF download attempt failed', {
        module: 'pdf_vl',
        url,
        attempt: attempt + 1,
        maxRetries: MAX_RETRIES,
        error: lastError.message,
      });

      if (attempt === MAX_RETRIES - 1) {
        break;
      }
    }
  }

  throw new PdfDownloadError(
    `Failed to download PDF after ${MAX_RETRIES} attempts: ${lastError?.message || 'Unknown error'}`,
    url,
  );
}

/**
 * Validate that buffer is a valid PDF
 */
export function validatePdf(buffer: Buffer): void {
  // Check PDF magic bytes
  const header = buffer.subarray(0, 5).toString('latin1');

  if (!header.startsWith(PDF_MAGIC_BYTES)) {
    throw new NotPdfError(
      `Not a valid PDF file: expected header "${PDF_MAGIC_BYTES}" but got "${header}"`,
      undefined,
      header,
    );
  }
}
