/**
 * Tests for redaction utilities
 */

import { describe, it, expect } from 'vitest';
import {
  isSensitiveKey,
  redactValue,
  redactFields,
  truncateString,
  sanitizeError,
} from '../../../scripts/logger/redaction';

describe('Redaction', () => {
  describe('isSensitiveKey', () => {
    it('should detect default sensitive keys (case-insensitive)', () => {
      expect(isSensitiveKey('token')).toBe(true);
      expect(isSensitiveKey('TOKEN')).toBe(true);
      expect(isSensitiveKey('apiKey')).toBe(true);
      expect(isSensitiveKey('API_KEY')).toBe(true);
      expect(isSensitiveKey('password')).toBe(true);
      expect(isSensitiveKey('secret')).toBe(true);
      expect(isSensitiveKey('cookie')).toBe(true);
      expect(isSensitiveKey('authorization')).toBe(true);
      expect(isSensitiveKey('accessToken')).toBe(true);
      expect(isSensitiveKey('refresh_token')).toBe(true);
      expect(isSensitiveKey('session')).toBe(true);
      expect(isSensitiveKey('auth')).toBe(true);
    });

    it('should detect partial matches', () => {
      expect(isSensitiveKey('myToken')).toBe(true);
      expect(isSensitiveKey('user_password')).toBe(true);
      expect(isSensitiveKey('apiKeyValue')).toBe(true);
    });

    it('should not detect non-sensitive keys', () => {
      expect(isSensitiveKey('username')).toBe(false);
      expect(isSensitiveKey('email')).toBe(false);
      expect(isSensitiveKey('status')).toBe(false);
      expect(isSensitiveKey('message')).toBe(false);
    });

    it('should support custom sensitive keys', () => {
      expect(isSensitiveKey('myCustomField')).toBe(false);
      expect(isSensitiveKey('myCustomField', ['custom'])).toBe(true);
      expect(isSensitiveKey('MYCUSTOMFIELD', ['custom'])).toBe(true);
    });
  });

  describe('redactValue', () => {
    it('should redact Bearer tokens', () => {
      const input = 'Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.long-token-value-here';
      const output = redactValue(input);
      expect(output).toContain('Bearer eyJhbG...here');
      expect(output).not.toContain('long-token-value');
    });

    it('should preserve short tokens', () => {
      const input = 'Bearer short';
      const output = redactValue(input);
      expect(output).toBe('Bearer short');
    });

    it('should redact cookie values', () => {
      const input = 'session=abc123def456ghi789jklmno; path=/';
      const output = redactValue(input);
      expect(output).toContain('session=abc123...lmno');
      expect(output).not.toContain('def456ghi');
    });

    it('should preserve short cookie values', () => {
      const input = 'test=short';
      const output = redactValue(input);
      expect(output).toBe('test=short');
    });

    it('should redact sensitive URL parameters', () => {
      const input = 'https://api.example.com/data?token=secret123&key=mykey456&user=alice';
      const output = redactValue(input);
      expect(output).toContain('token=[REDACTED]');
      expect(output).toContain('key=[REDACTED]');
      expect(output).toContain('user=alice');
      expect(output).not.toContain('secret123');
      expect(output).not.toContain('mykey456');
    });

    it('should handle non-string values', () => {
      expect(redactValue(123)).toBe(123);
      expect(redactValue(true)).toBe(true);
      expect(redactValue(null)).toBe(null);
      expect(redactValue(undefined)).toBe(undefined);
    });
  });

  describe('redactFields', () => {
    it('should redact sensitive field names', () => {
      const input = {
        username: 'alice',
        password: 'secret123',
        apiKey: 'key-abc-123',
        status: 'active',
      };
      const output = redactFields(input);
      expect(output.username).toBe('alice');
      expect(output.password).toBe('[REDACTED]');
      expect(output.apiKey).toBe('[REDACTED]');
      expect(output.status).toBe('active');
    });

    it('should apply string redaction to non-sensitive fields', () => {
      const input = {
        url: 'https://api.com?token=secret',
        message: 'Authorization: Bearer longtoken123456789',
      };
      const output = redactFields(input);
      expect(output.url).toContain('[REDACTED]');
      expect(output.message).toContain('Bearer longto...6789');
    });

    it('should recursively redact nested objects', () => {
      const input = {
        user: {
          name: 'alice',
          credentials: {
            password: 'secret',
            token: 'token123',
          },
        },
        data: {
          value: 42,
        },
      };
      const output = redactFields(input);
      expect(output.user.name).toBe('alice');
      expect(output.user.credentials.password).toBe('[REDACTED]');
      expect(output.user.credentials.token).toBe('[REDACTED]');
      expect(output.data.value).toBe(42);
    });

    it('should support custom redact keys', () => {
      const input = {
        username: 'alice',
        customField: 'sensitive-data',
        normalField: 'public-data',
      };
      const output = redactFields(input, ['customField']);
      expect(output.username).toBe('alice');
      expect(output.customField).toBe('[REDACTED]');
      expect(output.normalField).toBe('public-data');
    });

    it('should preserve arrays', () => {
      const input = {
        items: [1, 2, 3],
        tags: ['a', 'b'],
      };
      const output = redactFields(input);
      expect(output.items).toEqual([1, 2, 3]);
      expect(output.tags).toEqual(['a', 'b']);
    });
  });

  describe('truncateString', () => {
    it('should not truncate short strings', () => {
      const input = 'Short string';
      const output = truncateString(input, 100);
      expect(output).toBe(input);
    });

    it('should truncate long strings and add hash', () => {
      const input = 'a'.repeat(300);
      const output = truncateString(input, 200);
      expect(output.length).toBeLessThan(input.length);
      expect(output).toContain('truncated');
      expect(output).toContain('hash:');
      expect(output.startsWith('aaa')).toBe(true);
    });

    it('should use default max length of 200', () => {
      const input = 'x'.repeat(250);
      const output = truncateString(input);
      expect(output).toContain('truncated');
      expect(output.length).toBeLessThan(input.length);
    });
  });

  describe('sanitizeError', () => {
    it('should extract basic error properties', () => {
      const error = new Error('Test error');
      const sanitized = sanitizeError(error);
      expect(sanitized.message).toBe('Test error');
      expect(sanitized.name).toBe('Error');
      expect(sanitized.stack).toBeDefined();
    });

    it('should optionally exclude stack trace', () => {
      const error = new Error('Test error');
      const sanitized = sanitizeError(error, false);
      expect(sanitized.message).toBe('Test error');
      expect(sanitized.stack).toBeUndefined();
    });

    it('should include custom error properties', () => {
      const error: any = new Error('Test error');
      error.code = 'ENOTFOUND';
      error.statusCode = 404;
      const sanitized = sanitizeError(error);
      expect(sanitized.code).toBe('ENOTFOUND');
      expect(sanitized.statusCode).toBe(404);
    });

    it('should redact sensitive strings in custom properties', () => {
      const error: any = new Error('Test error');
      error.url = 'https://api.com?token=secret123';
      const sanitized = sanitizeError(error);
      expect(sanitized.url).toContain('[REDACTED]');
      expect(sanitized.url).not.toContain('secret123');
    });
  });
});
