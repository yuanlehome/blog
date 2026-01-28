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
          'https://invalid.example.com/layout-parsing',
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
          'https://invalid.example.com/layout-parsing',
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
          'https://example.com/layout-parsing',
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
          'https://example.com/layout-parsing',
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
          'https://example.com/layout-parsing',
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
        'https://example.com/layout-parsing',
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
          'https://example.com/layout-parsing',
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
          'https://example.com/layout-parsing',
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
        callPaddleOcrVl(
          largePdfBuffer,
          'https://example.com/layout-parsing',
          'test-token',
          mockLogger as any,
        ),
      ).rejects.toThrow('exceeds 25MB limit');
    });
  });

  describe('API URL validation', () => {
    it('should reject empty API URL', async () => {
      process.env.PDF_OCR_RETRY = '0';
      process.env.PDF_OCR_DIAG = '0';

      await expect(
        callPaddleOcrVl(samplePdfBuffer, '', 'test-token', mockLogger as any),
      ).rejects.toThrow('API URL is required');
    });

    it('should reject non-HTTPS API URL', async () => {
      process.env.PDF_OCR_RETRY = '0';
      process.env.PDF_OCR_DIAG = '0';

      await expect(
        callPaddleOcrVl(
          samplePdfBuffer,
          'http://example.com/layout-parsing',
          'test-token',
          mockLogger as any,
        ),
      ).rejects.toThrow('must use HTTPS protocol');
    });

    it('should reject API URL without /layout-parsing path', async () => {
      process.env.PDF_OCR_RETRY = '0';
      process.env.PDF_OCR_DIAG = '0';

      await expect(
        callPaddleOcrVl(
          samplePdfBuffer,
          'https://example.com/api/v1',
          'test-token',
          mockLogger as any,
        ),
      ).rejects.toThrow("must include '/layout-parsing'");
    });

    it('should accept valid API URL with /layout-parsing', async () => {
      process.env.PDF_OCR_RETRY = '0';
      process.env.PDF_OCR_DIAG = '0';

      // Mock successful response
      global.fetch = vi.fn(() => {
        const mockResponse = {
          result: {
            layoutParsingResults: [
              {
                markdown: {
                  text: '# Test\n\nValid URL accepted.',
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
        'https://example.aistudio-app.com/layout-parsing',
        'test-token',
        mockLogger as any,
      );

      expect(result.markdown).toContain('Test');
    });
  });

  describe('IPv4 forcing and connection timeout', () => {
    it('should use undici dispatcher with IPv4 forcing', async () => {
      process.env.PDF_OCR_RETRY = '0';
      process.env.PDF_OCR_DIAG = '0';

      let fetchOptions: any = null;

      // Mock fetch to capture options
      global.fetch = vi.fn((url: any, options: any) => {
        fetchOptions = options;
        const mockResponse = {
          result: {
            layoutParsingResults: [
              {
                markdown: {
                  text: '# Test\n\nIPv4 dispatcher applied.',
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

      await callPaddleOcrVl(
        samplePdfBuffer,
        'https://example.com/layout-parsing',
        'test-token',
        mockLogger as any,
      );

      // Verify dispatcher was provided (undici Agent)
      expect(fetchOptions).toBeDefined();
      expect(fetchOptions.dispatcher).toBeDefined();
      expect(typeof fetchOptions.dispatcher.destroy).toBe('function');
    });

    it('should handle connection timeout error', async () => {
      process.env.PDF_OCR_RETRY = '1';
      process.env.PDF_OCR_DIAG = '0';
      process.env.PDF_OCR_CONNECT_TIMEOUT_MS = '1000';

      let callCount = 0;

      // Mock fetch to simulate connection timeout
      global.fetch = vi.fn(() => {
        callCount++;
        const error: any = new TypeError('fetch failed');
        error.cause = {
          code: 'ETIMEDOUT',
          errno: -60,
          syscall: 'connect',
          address: '1.2.3.4',
          port: 443,
        };
        return Promise.reject(error);
      }) as any;

      try {
        await callPaddleOcrVl(
          samplePdfBuffer,
          'https://example.com/layout-parsing',
          'test-token',
          mockLogger as any,
        );
        expect.fail('Should have thrown');
      } catch (error) {
        if (error instanceof OcrApiError) {
          // Verify error contains connection details
          expect(error.responseBody).toContain('ETIMEDOUT');
          expect(error.responseBody).toContain('1.2.3.4');
          expect(error.responseBody).toContain('443');
        }
      }

      // Should have retried
      expect(callCount).toBe(2);
    });
  });

  describe('Error details extraction', () => {
    it('should extract all error fields including cause', async () => {
      process.env.PDF_OCR_RETRY = '0';
      process.env.PDF_OCR_DIAG = '0';

      const mockCause = {
        code: 'ECONNREFUSED',
        errno: -61,
        syscall: 'connect',
        address: '192.168.1.1',
        port: 443,
        hostname: 'example.com',
      };

      global.fetch = vi.fn(() => {
        const error: any = new TypeError('connect ECONNREFUSED');
        error.cause = mockCause;
        return Promise.reject(error);
      }) as any;

      try {
        await callPaddleOcrVl(
          samplePdfBuffer,
          'https://example.com/layout-parsing',
          'test-token',
          mockLogger as any,
        );
        expect.fail('Should have thrown');
      } catch (error) {
        if (error instanceof OcrApiError) {
          // Verify all fields are extracted
          const body = error.responseBody || '';
          expect(body).toContain('ECONNREFUSED');
          expect(body).toContain('-61');
          expect(body).toContain('connect');
          expect(body).toContain('192.168.1.1');
          expect(body).toContain('443');
        }
      }
    });

    it('should not have empty error object in logs', async () => {
      process.env.PDF_OCR_RETRY = '0';
      process.env.PDF_OCR_DIAG = '0';

      global.fetch = vi.fn(() => {
        return Promise.reject(new Error('Simple error'));
      }) as any;

      try {
        await callPaddleOcrVl(
          samplePdfBuffer,
          'https://example.com/layout-parsing',
          'test-token',
          mockLogger as any,
        );
      } catch (error) {
        // Verify logger.error was called with non-empty error object
        const errorCalls = mockLogger.error.mock.calls;
        expect(errorCalls.length).toBeGreaterThan(0);

        const lastErrorCall = errorCalls[errorCalls.length - 1];
        const errorParam = lastErrorCall[1];
        expect(errorParam).toBeDefined();
        expect(errorParam.error).toBeDefined();
        expect(errorParam.error.message).toBeDefined();
        expect(errorParam.error.message).not.toBe('');
      }
    });

    it('should truncate large response body to 2KB', async () => {
      process.env.PDF_OCR_RETRY = '0';
      process.env.PDF_OCR_DIAG = '0';

      // Create a large error with details > 2KB
      const largeMessage = 'x'.repeat(3000);
      const error: any = new TypeError(largeMessage);
      error.cause = { code: 'LARGE', data: 'y'.repeat(3000) };

      global.fetch = vi.fn(() => {
        return Promise.reject(error);
      }) as any;

      try {
        await callPaddleOcrVl(
          samplePdfBuffer,
          'https://example.com/layout-parsing',
          'test-token',
          mockLogger as any,
        );
      } catch (error) {
        if (error instanceof OcrApiError) {
          const body = error.responseBody || '';
          // Should be truncated
          expect(body.length).toBeLessThanOrEqual(2048 + 50); // +50 for "(truncated)" suffix
          expect(body).toContain('truncated');
        }
      }
    });
  });

  describe('IP override configuration', () => {
    beforeEach(() => {
      // Clear any IP override env vars
      delete process.env.PADDLE_OCR_VL_API_IP;
      delete process.env.PADDLE_OCR_VL_IP;
      delete process.env.PDF_OCR_API_IP;
      delete process.env.PADDLEOCR_VL_IP;
    });

    afterEach(() => {
      delete process.env.PADDLE_OCR_VL_API_IP;
      delete process.env.PADDLE_OCR_VL_IP;
      delete process.env.PDF_OCR_API_IP;
      delete process.env.PADDLEOCR_VL_IP;
    });

    it('should not create IP override when config is missing', async () => {
      process.env.PDF_OCR_RETRY = '0';
      process.env.PDF_OCR_DIAG = '0';

      let fetchOptions: any = null;

      global.fetch = vi.fn((url: any, options: any) => {
        fetchOptions = options;
        const mockResponse = {
          result: {
            layoutParsingResults: [
              {
                markdown: {
                  text: '# Test\n\nNo IP override.',
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

      await callPaddleOcrVl(
        samplePdfBuffer,
        'https://example.com/layout-parsing',
        'test-token',
        mockLogger as any,
      );

      // Should have dispatcher (always created for IPv4 forcing)
      expect(fetchOptions.dispatcher).toBeDefined();

      // Verify debug log shows override not enabled
      const debugCalls = mockLogger.debug.mock.calls;
      const fetchConfigLog = debugCalls.find((call: any) => call[1]?.stage === 'fetch_config');
      expect(fetchConfigLog).toBeDefined();
      expect(fetchConfigLog![1].overrideEnabled).toBe(false);
      expect(fetchConfigLog![1].overrideIpState).toBe('missing');
    });

    it('should handle empty IP override string', async () => {
      process.env.PDF_OCR_API_IP = '   '; // Whitespace only
      process.env.PDF_OCR_RETRY = '0';
      process.env.PDF_OCR_DIAG = '0';

      global.fetch = vi.fn(() => {
        const mockResponse = {
          result: {
            layoutParsingResults: [
              {
                markdown: {
                  text: '# Test\n\nEmpty IP.',
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

      await callPaddleOcrVl(
        samplePdfBuffer,
        'https://example.com/layout-parsing',
        'test-token',
        mockLogger as any,
      );

      // Verify debug log shows override not enabled
      const debugCalls = mockLogger.debug.mock.calls;
      const fetchConfigLog = debugCalls.find((call: any) => call[1]?.stage === 'fetch_config');
      expect(fetchConfigLog).toBeDefined();
      expect(fetchConfigLog![1].overrideEnabled).toBe(false);
      expect(fetchConfigLog![1].overrideIpState).toBe('missing');
    });

    it('should handle invalid IP override', async () => {
      process.env.PDF_OCR_API_IP = 'not-an-ip-address';
      process.env.PDF_OCR_RETRY = '0';
      process.env.PDF_OCR_DIAG = '0';

      global.fetch = vi.fn(() => {
        const mockResponse = {
          result: {
            layoutParsingResults: [
              {
                markdown: {
                  text: '# Test\n\nInvalid IP.',
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

      await callPaddleOcrVl(
        samplePdfBuffer,
        'https://example.com/layout-parsing',
        'test-token',
        mockLogger as any,
      );

      // Verify warning was logged
      const warnCalls = mockLogger.warn.mock.calls;
      const invalidIpWarn = warnCalls.find(
        (call: any) => call[0]?.includes('Invalid IP override') || call[1]?.reason,
      );
      expect(invalidIpWarn).toBeDefined();

      // Verify debug log shows override as invalid
      const debugCalls = mockLogger.debug.mock.calls;
      const fetchConfigLog = debugCalls.find((call: any) => call[1]?.stage === 'fetch_config');
      expect(fetchConfigLog).toBeDefined();
      expect(fetchConfigLog![1].overrideEnabled).toBe(false);
      expect(fetchConfigLog![1].overrideIpState).toBe('invalid');
    });

    it('should use valid IPv4 override', async () => {
      process.env.PDF_OCR_API_IP = '192.168.1.100';
      process.env.PDF_OCR_RETRY = '0';
      process.env.PDF_OCR_DIAG = '0';

      global.fetch = vi.fn(() => {
        const mockResponse = {
          result: {
            layoutParsingResults: [
              {
                markdown: {
                  text: '# Test\n\nValid IPv4 override.',
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

      await callPaddleOcrVl(
        samplePdfBuffer,
        'https://example.com/layout-parsing',
        'test-token',
        mockLogger as any,
      );

      // Verify debug log shows override enabled
      const debugCalls = mockLogger.debug.mock.calls;
      const networkConfigLog = debugCalls.find(
        (call: any) => call[1]?.stage === 'network_config' && call[1]?.ip === '192.168.1.100',
      );
      expect(networkConfigLog).toBeDefined();
      expect(networkConfigLog![1].ipVersion).toBe('IPv4');

      const fetchConfigLog = debugCalls.find((call: any) => call[1]?.stage === 'fetch_config');
      expect(fetchConfigLog).toBeDefined();
      expect(fetchConfigLog![1].overrideEnabled).toBe(true);
      expect(fetchConfigLog![1].overrideIpState).toBe('enabled');
    });

    it('should use valid IPv6 override', async () => {
      process.env.PADDLE_OCR_VL_IP = '2001:0db8:85a3::8a2e:0370:7334';
      process.env.PDF_OCR_RETRY = '0';
      process.env.PDF_OCR_DIAG = '0';

      global.fetch = vi.fn(() => {
        const mockResponse = {
          result: {
            layoutParsingResults: [
              {
                markdown: {
                  text: '# Test\n\nValid IPv6 override.',
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

      await callPaddleOcrVl(
        samplePdfBuffer,
        'https://example.com/layout-parsing',
        'test-token',
        mockLogger as any,
      );

      // Verify debug log shows override enabled with IPv6
      const debugCalls = mockLogger.debug.mock.calls;
      const networkConfigLog = debugCalls.find(
        (call: any) => call[1]?.stage === 'network_config' && call[1]?.ipVersion === 'IPv6',
      );
      expect(networkConfigLog).toBeDefined();

      const fetchConfigLog = debugCalls.find((call: any) => call[1]?.stage === 'fetch_config');
      expect(fetchConfigLog).toBeDefined();
      expect(fetchConfigLog![1].overrideEnabled).toBe(true);
      expect(fetchConfigLog![1].overrideIpState).toBe('enabled');
    });

    it('should handle undefined IP without ERR_INVALID_IP_ADDRESS', async () => {
      // This is the critical test for the bug fix
      // When IP override is undefined, should NOT pass it to network layer
      process.env.PDF_OCR_RETRY = '0';
      process.env.PDF_OCR_DIAG = '0';

      let fetchCalled = false;

      global.fetch = vi.fn(() => {
        fetchCalled = true;
        const mockResponse = {
          result: {
            layoutParsingResults: [
              {
                markdown: {
                  text: '# Test\n\nNo IP error.',
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

      // Should succeed without ERR_INVALID_IP_ADDRESS
      const result = await callPaddleOcrVl(
        samplePdfBuffer,
        'https://example.com/layout-parsing',
        'test-token',
        mockLogger as any,
      );

      expect(result.markdown).toContain('Test');
      expect(fetchCalled).toBe(true);

      // Should not have any errors about invalid IP
      const errorCalls = mockLogger.error.mock.calls;
      const ipErrors = errorCalls.filter((call: any) =>
        JSON.stringify(call).includes('ERR_INVALID_IP_ADDRESS'),
      );
      expect(ipErrors.length).toBe(0);
    });

    it('should prioritize PADDLE_OCR_VL_API_IP over other env vars', async () => {
      process.env.PADDLE_OCR_VL_API_IP = '10.0.0.1';
      process.env.PDF_OCR_API_IP = '10.0.0.2';
      process.env.PADDLEOCR_VL_IP = '10.0.0.3';
      process.env.PDF_OCR_RETRY = '0';
      process.env.PDF_OCR_DIAG = '0';

      global.fetch = vi.fn(() => {
        const mockResponse = {
          result: {
            layoutParsingResults: [
              {
                markdown: {
                  text: '# Test\n\nPriority test.',
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

      await callPaddleOcrVl(
        samplePdfBuffer,
        'https://example.com/layout-parsing',
        'test-token',
        mockLogger as any,
      );

      // Verify the first env var was used
      const debugCalls = mockLogger.debug.mock.calls;
      const networkConfigLog = debugCalls.find(
        (call: any) => call[1]?.stage === 'network_config' && call[1]?.ip,
      );
      expect(networkConfigLog).toBeDefined();
      expect(networkConfigLog![1].ip).toBe('10.0.0.1');
    });
  });

  describe('Proxy environment detection', () => {
    beforeEach(() => {
      delete process.env.HTTP_PROXY;
      delete process.env.http_proxy;
      delete process.env.HTTPS_PROXY;
      delete process.env.https_proxy;
      delete process.env.NO_PROXY;
      delete process.env.no_proxy;
    });

    afterEach(() => {
      delete process.env.HTTP_PROXY;
      delete process.env.http_proxy;
      delete process.env.HTTPS_PROXY;
      delete process.env.https_proxy;
      delete process.env.NO_PROXY;
      delete process.env.no_proxy;
    });

    it('should detect HTTP_PROXY presence', async () => {
      process.env.HTTP_PROXY = 'http://proxy.example.com:8080';
      process.env.PDF_OCR_RETRY = '0';
      process.env.PDF_OCR_DIAG = '0';

      global.fetch = vi.fn(() => {
        const mockResponse = {
          result: {
            layoutParsingResults: [
              {
                markdown: {
                  text: '# Test\n\nProxy present.',
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

      await callPaddleOcrVl(
        samplePdfBuffer,
        'https://example.com/layout-parsing',
        'test-token',
        mockLogger as any,
      );

      // Verify proxy presence is logged (but not the URL itself)
      const debugCalls = mockLogger.debug.mock.calls;
      const fetchConfigLog = debugCalls.find((call: any) => call[1]?.stage === 'fetch_config');
      expect(fetchConfigLog).toBeDefined();
      expect(fetchConfigLog![1].proxyPresent).toBe(true);

      // Ensure proxy URL is NOT logged (security)
      const allLogs = JSON.stringify(mockLogger.debug.mock.calls);
      expect(allLogs).not.toContain('proxy.example.com');
    });

    it('should detect NO_PROXY presence', async () => {
      process.env.NO_PROXY = 'localhost,127.0.0.1';
      process.env.PDF_OCR_RETRY = '0';
      process.env.PDF_OCR_DIAG = '0';

      global.fetch = vi.fn(() => {
        const mockResponse = {
          result: {
            layoutParsingResults: [
              {
                markdown: {
                  text: '# Test\n\nNo proxy config.',
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

      await callPaddleOcrVl(
        samplePdfBuffer,
        'https://example.com/layout-parsing',
        'test-token',
        mockLogger as any,
      );

      const debugCalls = mockLogger.debug.mock.calls;
      const fetchConfigLog = debugCalls.find((call: any) => call[1]?.stage === 'fetch_config');
      expect(fetchConfigLog).toBeDefined();
      expect(fetchConfigLog![1].proxyPresent).toBe(true);
    });

    it('should show proxyPresent=false when no proxy vars set', async () => {
      process.env.PDF_OCR_RETRY = '0';
      process.env.PDF_OCR_DIAG = '0';

      global.fetch = vi.fn(() => {
        const mockResponse = {
          result: {
            layoutParsingResults: [
              {
                markdown: {
                  text: '# Test\n\nNo proxy.',
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

      await callPaddleOcrVl(
        samplePdfBuffer,
        'https://example.com/layout-parsing',
        'test-token',
        mockLogger as any,
      );

      const debugCalls = mockLogger.debug.mock.calls;
      const fetchConfigLog = debugCalls.find((call: any) => call[1]?.stage === 'fetch_config');
      expect(fetchConfigLog).toBeDefined();
      expect(fetchConfigLog![1].proxyPresent).toBe(false);
    });
  });
});
