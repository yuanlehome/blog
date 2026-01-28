/**
 * PDF VL OCR Tests
 *
 * Tests for PaddleOCR-VL client with network failure scenarios
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import {
  callPaddleOcrVl,
  callLocalMockOcr,
  OcrApiError,
} from '../../scripts/import/adapters/pdf_vl_ocr.js';
import fs from 'fs';
import path from 'path';

// Mock logger
const mockLogger = {
  debug: vi.fn(),
  info: vi.fn(),
  warn: vi.fn(),
  error: vi.fn(),
  summary: vi.fn(),
  span: vi.fn(() => ({
    start: vi.fn(),
    end: vi.fn(),
  })),
  child: vi.fn(function (this: any) {
    return this;
  }),
};

// Load test fixtures
const fixturesDir = path.join(process.cwd(), 'tests/fixtures/pdf');
const samplePdfBuffer = fs.readFileSync(path.join(fixturesDir, 'sample.pdf'));

// Mock fetch globally
const originalFetch = global.fetch;

describe('PDF VL OCR Client', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  afterEach(() => {
    global.fetch = originalFetch;
    delete process.env.PDF_OCR_RETRY;
    delete process.env.PDF_OCR_TIMEOUT_MS;
    delete process.env.PDF_OCR_DIAG;
  });

  describe('Network failure retry', () => {
    it('should retry on ENOTFOUND error', async () => {
      process.env.PDF_OCR_RETRY = '2';
      process.env.PDF_OCR_TIMEOUT_MS = '5000';
      process.env.PDF_OCR_DIAG = '0'; // Disable diagnostics for speed

      let callCount = 0;

      // Mock fetch to simulate ENOTFOUND error
      global.fetch = vi.fn(() => {
        callCount++;
        const error: any = new TypeError('fetch failed');
        error.cause = {
          code: 'ENOTFOUND',
          errno: -3008,
          syscall: 'getaddrinfo',
          hostname: 'invalid.example.com',
        };
        return Promise.reject(error);
      }) as any;

      await expect(
        callPaddleOcrVl(
          samplePdfBuffer,
          'https://invalid.example.com/api',
          'test-token',
          mockLogger as any,
        ),
      ).rejects.toThrow(OcrApiError);

      // Should have retried (1 initial + 2 retries = 3 total)
      expect(callCount).toBe(3);

      // Verify error contains cause information
      try {
        await callPaddleOcrVl(
          samplePdfBuffer,
          'https://invalid.example.com/api',
          'test-token',
          mockLogger as any,
        );
      } catch (error) {
        if (error instanceof OcrApiError) {
          expect(error.cause).toBeDefined();
          expect(error.responseBody).toContain('ENOTFOUND');
        }
      }
    });

    it('should retry on ETIMEDOUT error', async () => {
      process.env.PDF_OCR_RETRY = '1';
      process.env.PDF_OCR_TIMEOUT_MS = '5000';
      process.env.PDF_OCR_DIAG = '0';

      let callCount = 0;

      global.fetch = vi.fn(() => {
        callCount++;
        const error: any = new TypeError('fetch failed');
        error.cause = {
          code: 'ETIMEDOUT',
          errno: -60,
          syscall: 'connect',
        };
        return Promise.reject(error);
      }) as any;

      await expect(
        callPaddleOcrVl(
          samplePdfBuffer,
          'https://example.com/api',
          'test-token',
          mockLogger as any,
        ),
      ).rejects.toThrow(OcrApiError);

      expect(callCount).toBe(2); // 1 initial + 1 retry
    });

    it('should retry on HTTP 503 error', async () => {
      process.env.PDF_OCR_RETRY = '2';
      process.env.PDF_OCR_DIAG = '0';

      let callCount = 0;

      global.fetch = vi.fn(() => {
        callCount++;
        return Promise.resolve({
          ok: false,
          status: 503,
          statusText: 'Service Unavailable',
          text: () => Promise.resolve('Service temporarily unavailable'),
        } as Response);
      }) as any;

      await expect(
        callPaddleOcrVl(
          samplePdfBuffer,
          'https://example.com/api',
          'test-token',
          mockLogger as any,
        ),
      ).rejects.toThrow(OcrApiError);

      expect(callCount).toBe(3); // 1 initial + 2 retries
    });

    it('should NOT retry on HTTP 400 error', async () => {
      process.env.PDF_OCR_RETRY = '3';
      process.env.PDF_OCR_DIAG = '0';

      let callCount = 0;

      global.fetch = vi.fn(() => {
        callCount++;
        return Promise.resolve({
          ok: false,
          status: 400,
          statusText: 'Bad Request',
          text: () => Promise.resolve('Invalid request payload'),
        } as Response);
      }) as any;

      await expect(
        callPaddleOcrVl(
          samplePdfBuffer,
          'https://example.com/api',
          'test-token',
          mockLogger as any,
        ),
      ).rejects.toThrow(OcrApiError);

      expect(callCount).toBe(1); // Should not retry on 4xx errors
    });

    it('should succeed after retry', async () => {
      process.env.PDF_OCR_RETRY = '2';
      process.env.PDF_OCR_DIAG = '0';

      let callCount = 0;

      global.fetch = vi.fn(() => {
        callCount++;
        if (callCount < 2) {
          // First call fails
          const error: any = new TypeError('fetch failed');
          error.cause = { code: 'ECONNRESET' };
          return Promise.reject(error);
        }
        // Second call succeeds
        const mockResponse = {
          result: {
            layoutParsingResults: [
              {
                markdown: {
                  text: '# Test\n\nContent here.',
                  images: {},
                },
              },
            ],
          },
        };
        return Promise.resolve({
          ok: true,
          status: 200,
          text: () => Promise.resolve(JSON.stringify(mockResponse)),
        } as Response);
      }) as any;

      const result = await callPaddleOcrVl(
        samplePdfBuffer,
        'https://example.com/api',
        'test-token',
        mockLogger as any,
      );

      expect(callCount).toBe(2);
      expect(result.markdown).toContain('Test');
    });
  });

  describe('Timeout abort', () => {
    it('should abort on timeout', async () => {
      process.env.PDF_OCR_TIMEOUT_MS = '1000'; // 1 second timeout
      process.env.PDF_OCR_RETRY = '0'; // No retries
      process.env.PDF_OCR_DIAG = '0';

      // Mock fetch that respects abort signal
      global.fetch = vi.fn((url: any, options: any) => {
        return new Promise((resolve, reject) => {
          // Listen for abort signal
          if (options?.signal) {
            options.signal.addEventListener('abort', () => {
              reject(new DOMException('The operation was aborted', 'AbortError'));
            });
          }
          // Never resolve otherwise
        });
      }) as any;

      const startTime = Date.now();

      await expect(
        callPaddleOcrVl(
          samplePdfBuffer,
          'https://example.com/api',
          'test-token',
          mockLogger as any,
        ),
      ).rejects.toThrow();

      const duration = Date.now() - startTime;

      // Should timeout within reasonable time (allow some overhead)
      expect(duration).toBeLessThan(2000);
      expect(duration).toBeGreaterThan(900);
    }, 15000); // Set test timeout to 15 seconds for safety
  });

  describe('Local mock provider', () => {
    it('should load mock fixture successfully', async () => {
      const result = await callLocalMockOcr(mockLogger as any);

      expect(result.markdown).toBeDefined();
      expect(result.markdown.length).toBeGreaterThan(100);
      expect(result.markdown).toContain('Mock PDF Document Title');
      expect(result.images).toBeDefined();
    });

    it('should return valid markdown structure', async () => {
      const result = await callLocalMockOcr(mockLogger as any);

      // Check for sufficient content (minimum 20 lines requirement)
      const lines = result.markdown.split('\n').filter((line) => line.trim().length > 0);
      expect(lines.length).toBeGreaterThanOrEqual(20);

      // Check for headings
      expect(result.markdown).toMatch(/^#\s+/m);
    });
  });

  describe('Error details extraction', () => {
    it('should extract cause details from fetch error', async () => {
      process.env.PDF_OCR_RETRY = '0';
      process.env.PDF_OCR_DIAG = '0';

      const mockCause = {
        code: 'ECONNREFUSED',
        errno: -61,
        syscall: 'connect',
        address: '127.0.0.1',
        port: 443,
      };

      global.fetch = vi.fn(() => {
        const error: any = new TypeError('connect ECONNREFUSED');
        error.cause = mockCause;
        return Promise.reject(error);
      }) as any;

      try {
        await callPaddleOcrVl(
          samplePdfBuffer,
          'https://example.com/api',
          'test-token',
          mockLogger as any,
        );
        expect.fail('Should have thrown');
      } catch (error) {
        if (error instanceof OcrApiError) {
          expect(error.responseBody).toContain('ECONNREFUSED');
          expect(error.responseBody).toContain('connect');
        }
      }
    });
  });

  describe('PDF size validation', () => {
    it('should reject PDF larger than 25MB', async () => {
      process.env.PDF_OCR_RETRY = '0';
      process.env.PDF_OCR_DIAG = '0';

      // Create a buffer larger than 25MB
      const largePdfBuffer = Buffer.alloc(26 * 1024 * 1024);
      largePdfBuffer.write('%PDF-1.4', 0); // Valid PDF header

      await expect(
        callPaddleOcrVl(largePdfBuffer, 'https://example.com/api', 'test-token', mockLogger as any),
      ).rejects.toThrow('exceeds 25MB limit');
    });
  });
});
