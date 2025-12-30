/**
 * Sensitive data redaction utilities
 */

/** Default sensitive field names to redact */
const DEFAULT_REDACT_KEYS = [
  'token',
  'secret',
  'cookie',
  'authorization',
  'password',
  'passwd',
  'pwd',
  'apikey',
  'api_key',
  'access_token',
  'refresh_token',
  'session',
  'auth',
];

/** Pattern to detect Bearer tokens */
const BEARER_TOKEN_PATTERN = /Bearer\s+[\w\-._~+/]+=*/gi;

/** Pattern to detect cookie values (must not match URL parameters) */
const COOKIE_PATTERN = /([a-zA-Z0-9_-]+)=([^;,\s&=]{11,})/g;

/** Pattern to detect sensitive URL parameters */
const SENSITIVE_URL_PARAMS = /([?&])(token|secret|password|auth|apikey|api_key|key)=([^&\s]+)/gi;

/**
 * Check if a field name is sensitive
 */
export function isSensitiveKey(key: string, customKeys: string[] = []): boolean {
  const lowerKey = key.toLowerCase();
  const allKeys = [...DEFAULT_REDACT_KEYS, ...customKeys.map((k) => k.toLowerCase())];
  return allKeys.some((sensitiveKey) => lowerKey.includes(sensitiveKey));
}

/**
 * Redact a single value
 */
export function redactValue(value: any): any {
  if (typeof value === 'string') {
    let redacted = value;

    // Redact sensitive URL parameters FIRST (before cookie pattern which is more greedy)
    redacted = redacted.replace(SENSITIVE_URL_PARAMS, (match, prefix, param, value) => {
      return `${prefix}${param}=[REDACTED]`;
    });

    // Redact Bearer tokens
    redacted = redacted.replace(BEARER_TOKEN_PATTERN, (match) => {
      const token = match.substring('Bearer '.length);
      if (token.length <= 10) return match;
      return `Bearer ${token.substring(0, 6)}...${token.substring(token.length - 4)}`;
    });

    // Redact cookie values
    redacted = redacted.replace(COOKIE_PATTERN, (match, name, value) => {
      if (value.length <= 10) return match;
      return `${name}=${value.substring(0, 6)}...${value.substring(value.length - 4)}`;
    });

    return redacted;
  }

  return value;
}

/**
 * Redact sensitive fields in an object
 */
export function redactFields(
  obj: Record<string, any>,
  customKeys: string[] = [],
): Record<string, any> {
  const redacted: Record<string, any> = {};

  for (const [key, value] of Object.entries(obj)) {
    if (isSensitiveKey(key, customKeys)) {
      redacted[key] = '[REDACTED]';
    } else if (typeof value === 'object' && value !== null && !Array.isArray(value)) {
      // Recursively redact nested objects
      redacted[key] = redactFields(value, customKeys);
    } else if (typeof value === 'string') {
      // Apply string redaction patterns
      redacted[key] = redactValue(value);
    } else {
      redacted[key] = value;
    }
  }

  return redacted;
}

/**
 * Truncate long strings to prevent logging massive content
 */
export function truncateString(value: string, maxLength: number = 200): string {
  if (value.length <= maxLength) {
    return value;
  }
  const hash = hashString(value);
  return `${value.substring(0, maxLength)}... [truncated, hash: ${hash}]`;
}

/**
 * Simple hash function for truncated content
 */
function hashString(str: string): string {
  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    const char = str.charCodeAt(i);
    hash = (hash << 5) - hash + char;
    hash = hash & hash; // Convert to 32bit integer
  }
  return Math.abs(hash).toString(16).substring(0, 8);
}

/**
 * Sanitize an error object for logging
 */
export function sanitizeError(error: Error, includeStack: boolean = true): Record<string, any> {
  const sanitized: Record<string, any> = {
    message: error.message,
    name: error.name,
  };

  if (includeStack && error.stack) {
    sanitized.stack = error.stack;
  }

  // Include any custom error properties
  for (const key of Object.keys(error)) {
    if (key !== 'message' && key !== 'name' && key !== 'stack') {
      const value = (error as any)[key];
      if (typeof value === 'string') {
        sanitized[key] = redactValue(value);
      } else {
        sanitized[key] = value;
      }
    }
  }

  return sanitized;
}
