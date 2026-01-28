/**
 * PDF VL OCR Error Classification Tests
 *
 * Tests for classifyOcrError function to ensure proper error categorization
 */

import { describe, it, expect } from 'vitest';
import { classifyOcrError } from '../../scripts/import/adapters/pdf_vl_ocr.js';

describe('classifyOcrError', () => {
  describe('OCR_NET_INVALID_IP', () => {
    it('should classify ERR_INVALID_IP_ADDRESS in cause.code', () => {
      const error = {
        message: 'fetch failed',
        cause: {
          code: 'ERR_INVALID_IP_ADDRESS',
          message: 'Invalid IP address: undefined',
        },
      };
      expect(classifyOcrError(error, 0)).toBe('OCR_NET_INVALID_IP');
    });

    it('should classify ERR_INVALID_IP_ADDRESS in error.code', () => {
      const error = {
        code: 'ERR_INVALID_IP_ADDRESS',
        message: 'Invalid IP address: undefined',
      };
      expect(classifyOcrError(error, 0)).toBe('OCR_NET_INVALID_IP');
    });

    it('should classify Invalid IP address in message', () => {
      const error = {
        message: 'Invalid IP address: undefined',
      };
      expect(classifyOcrError(error, 0)).toBe('OCR_NET_INVALID_IP');
    });

    it('should classify ERR_INVALID_IP_ADDRESS in nested cause', () => {
      const error = {
        message: 'fetch failed',
        cause: {
          message: 'connection error',
          cause: {
            code: 'ERR_INVALID_IP_ADDRESS',
          },
        },
      };
      expect(classifyOcrError(error, 0)).toBe('OCR_NET_INVALID_IP');
    });
  });

  describe('OCR_NET_DNS_FAIL', () => {
    it('should classify ENOTFOUND error', () => {
      const error = {
        message: 'fetch failed',
        cause: {
          code: 'ENOTFOUND',
          errno: -3008,
          syscall: 'getaddrinfo',
        },
      };
      expect(classifyOcrError(error, 0)).toBe('OCR_NET_DNS_FAIL');
    });

    it('should classify EAI_AGAIN error', () => {
      const error = {
        code: 'EAI_AGAIN',
        message: 'getaddrinfo EAI_AGAIN',
      };
      expect(classifyOcrError(error, 0)).toBe('OCR_NET_DNS_FAIL');
    });

    it('should classify getaddrinfo in message', () => {
      const error = {
        message: 'getaddrinfo failed',
      };
      expect(classifyOcrError(error, 0)).toBe('OCR_NET_DNS_FAIL');
    });
  });

  describe('OCR_NET_TIMEOUT', () => {
    it('should classify ETIMEDOUT error', () => {
      const error = {
        cause: {
          code: 'ETIMEDOUT',
          message: 'Connection timeout',
        },
      };
      expect(classifyOcrError(error, 0)).toBe('OCR_NET_TIMEOUT');
    });

    it('should classify AbortError', () => {
      const error = {
        name: 'AbortError',
        message: 'The operation was aborted',
      };
      expect(classifyOcrError(error, 0)).toBe('OCR_NET_TIMEOUT');
    });

    it('should classify timeout in message', () => {
      const error = {
        message: 'Request timeout after 30000ms',
      };
      expect(classifyOcrError(error, 0)).toBe('OCR_NET_TIMEOUT');
    });

    it('should classify abort in message', () => {
      const error = {
        message: 'The user aborted a request',
      };
      expect(classifyOcrError(error, 0)).toBe('OCR_NET_TIMEOUT');
    });
  });

  describe('OCR_NET_TLS', () => {
    it('should classify EPROTO error', () => {
      const error = {
        cause: {
          code: 'EPROTO',
          message: 'SSL handshake failed',
        },
      };
      expect(classifyOcrError(error, 0)).toBe('OCR_NET_TLS');
    });

    it('should classify SSL in message', () => {
      const error = {
        message: 'SSL certificate verification failed',
      };
      expect(classifyOcrError(error, 0)).toBe('OCR_NET_TLS');
    });

    it('should classify TLS in message', () => {
      const error = {
        message: 'TLS handshake timeout',
      };
      expect(classifyOcrError(error, 0)).toBe('OCR_NET_TLS');
    });

    it('should classify certificate in message', () => {
      const error = {
        message: 'unable to verify certificate',
      };
      expect(classifyOcrError(error, 0)).toBe('OCR_NET_TLS');
    });
  });

  describe('OCR_NET_CONNECTION', () => {
    it('should classify ECONNRESET error', () => {
      const error = {
        code: 'ECONNRESET',
        message: 'socket hang up',
      };
      expect(classifyOcrError(error, 0)).toBe('OCR_NET_CONNECTION');
    });

    it('should classify ECONNREFUSED error', () => {
      const error = {
        cause: {
          code: 'ECONNREFUSED',
          errno: -61,
        },
      };
      expect(classifyOcrError(error, 0)).toBe('OCR_NET_CONNECTION');
    });

    it('should classify EHOSTUNREACH error', () => {
      const error = {
        cause: {
          code: 'EHOSTUNREACH',
        },
      };
      expect(classifyOcrError(error, 0)).toBe('OCR_NET_CONNECTION');
    });

    it('should classify fetch failed in message', () => {
      const error = {
        message: 'fetch failed',
      };
      expect(classifyOcrError(error, 0)).toBe('OCR_NET_CONNECTION');
    });
  });

  describe('OCR_HTTP_NON_2XX', () => {
    it('should classify 400 Bad Request', () => {
      expect(classifyOcrError({}, 400)).toBe('OCR_HTTP_NON_2XX');
    });

    it('should classify 401 Unauthorized', () => {
      expect(classifyOcrError({}, 401)).toBe('OCR_HTTP_NON_2XX');
    });

    it('should classify 404 Not Found', () => {
      expect(classifyOcrError({}, 404)).toBe('OCR_HTTP_NON_2XX');
    });

    it('should classify 500 Internal Server Error', () => {
      expect(classifyOcrError({}, 500)).toBe('OCR_HTTP_NON_2XX');
    });

    it('should classify 502 Bad Gateway', () => {
      expect(classifyOcrError({}, 502)).toBe('OCR_HTTP_NON_2XX');
    });

    it('should classify 503 Service Unavailable', () => {
      expect(classifyOcrError({}, 503)).toBe('OCR_HTTP_NON_2XX');
    });

    it('should not classify 2xx as HTTP error', () => {
      expect(classifyOcrError({}, 200)).not.toBe('OCR_HTTP_NON_2XX');
    });
  });

  describe('OCR_RESPONSE_PARSE_FAIL', () => {
    it('should classify OcrParseError by name', () => {
      const error = {
        name: 'OcrParseError',
        message: 'Failed to parse response',
      };
      expect(classifyOcrError(error, 200)).toBe('OCR_RESPONSE_PARSE_FAIL');
    });
  });

  describe('OCR_RESULT_EMPTY', () => {
    it('should classify empty in message', () => {
      const error = {
        message: 'Result is empty',
      };
      expect(classifyOcrError(error, 200)).toBe('OCR_RESULT_EMPTY');
    });

    it('should classify no markdown in message', () => {
      const error = {
        message: 'API returned no markdown text',
      };
      expect(classifyOcrError(error, 200)).toBe('OCR_RESULT_EMPTY');
    });
  });

  describe('OCR_PDF_FETCH_FAIL', () => {
    it('should classify PDF download error', () => {
      const error = {
        message: 'Failed to download PDF from URL',
      };
      expect(classifyOcrError(error, 0)).toBe('OCR_PDF_FETCH_FAIL');
    });

    it('should classify PDF fetch error', () => {
      const error = {
        message: 'PDF fetch failed',
      };
      expect(classifyOcrError(error, 0)).toBe('OCR_PDF_FETCH_FAIL');
    });
  });

  describe('OCR_PDF_PARSE_FAIL', () => {
    it('should classify PDF parse error', () => {
      const error = {
        message: 'Failed to parse PDF',
      };
      expect(classifyOcrError(error, 0)).toBe('OCR_PDF_PARSE_FAIL');
    });

    it('should classify invalid PDF error', () => {
      const error = {
        message: 'Invalid PDF format',
      };
      expect(classifyOcrError(error, 0)).toBe('OCR_PDF_PARSE_FAIL');
    });
  });

  describe('OCR_UNKNOWN', () => {
    it('should classify unknown errors', () => {
      const error = {
        message: 'Something unexpected happened',
      };
      expect(classifyOcrError(error, 0)).toBe('OCR_UNKNOWN');
    });

    it('should classify null errors', () => {
      expect(classifyOcrError(null, 0)).toBe('OCR_UNKNOWN');
    });

    it('should classify undefined errors', () => {
      expect(classifyOcrError(undefined, 0)).toBe('OCR_UNKNOWN');
    });
  });

  describe('Priority order', () => {
    it('should prioritize INVALID_IP over generic connection error', () => {
      const error = {
        message: 'fetch failed',
        cause: {
          code: 'ERR_INVALID_IP_ADDRESS',
        },
      };
      // Should be OCR_NET_INVALID_IP, not OCR_NET_CONNECTION
      expect(classifyOcrError(error, 0)).toBe('OCR_NET_INVALID_IP');
    });

    it('should prioritize DNS_FAIL over generic connection error', () => {
      const error = {
        message: 'fetch failed',
        cause: {
          code: 'ENOTFOUND',
        },
      };
      // Should be OCR_NET_DNS_FAIL, not OCR_NET_CONNECTION
      expect(classifyOcrError(error, 0)).toBe('OCR_NET_DNS_FAIL');
    });

    it('should prioritize HTTP status over parse error in message', () => {
      const error = {
        message: 'Failed to parse response after HTTP 500',
      };
      // With statusCode 500, should be OCR_HTTP_NON_2XX
      expect(classifyOcrError(error, 500)).toBe('OCR_HTTP_NON_2XX');
    });
  });
});
