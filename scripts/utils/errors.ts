/**
 * Error serialization utilities for consistent logging
 */

import { redactValue } from '../logger/redaction.js';

/**
 * Serialized error structure
 */
export interface SerializedError {
  message: string;
  name: string;
  stack?: string;
  cause?: SerializedError;
  [key: string]: any;
}

/**
 * Serialize an error for logging with consistent structure
 * Ensures all errors have message, name, stack, and cause
 */
export function serializeError(error: unknown): SerializedError {
  // Handle null/undefined
  if (error === null || error === undefined) {
    return {
      message: String(error),
      name: 'UnknownError',
    };
  }

  // Handle Error objects
  if (error instanceof Error) {
    const serialized: SerializedError = {
      message: error.message || 'Unknown error',
      name: error.name || 'Error',
    };

    // Include stack trace
    if (error.stack) {
      serialized.stack = error.stack;
    }

    // Include cause if present (Error.cause is ES2022 feature)
    if ((error as any).cause) {
      serialized.cause = serializeError((error as any).cause);
    }

    // Include any additional properties from the error
    for (const key of Object.keys(error)) {
      if (key !== 'message' && key !== 'name' && key !== 'stack' && key !== 'cause') {
        const value = (error as any)[key];
        // Redact sensitive values in string properties
        if (typeof value === 'string') {
          serialized[key] = redactValue(value);
        } else if (typeof value === 'object' && value !== null && !Array.isArray(value)) {
          // Recursively redact nested objects
          serialized[key] = redactNestedObject(value);
        } else {
          serialized[key] = value;
        }
      }
    }

    return serialized;
  }

  // Handle objects with message property
  if (typeof error === 'object' && error !== null) {
    const obj = error as any;
    return {
      message: obj.message || obj.msg || String(error),
      name: obj.name || 'Error',
      ...(obj.stack && { stack: obj.stack }),
      ...(obj.cause && { cause: serializeError(obj.cause) }),
      ...Object.keys(obj).reduce(
        (acc, key) => {
          if (!['message', 'msg', 'name', 'stack', 'cause'].includes(key)) {
            const value = obj[key];
            if (typeof value === 'string') {
              acc[key] = redactValue(value);
            } else if (typeof value === 'object' && value !== null && !Array.isArray(value)) {
              // Recursively redact nested objects
              acc[key] = redactNestedObject(value);
            } else {
              acc[key] = value;
            }
          }
          return acc;
        },
        {} as Record<string, any>,
      ),
    };
  }

  // Handle primitive types (string, number, boolean)
  return {
    message: String(error),
    name: 'UnknownError',
  };
}

/**
 * Helper function to recursively redact nested objects
 */
function redactNestedObject(obj: Record<string, any>): Record<string, any> {
  const redacted: Record<string, any> = {};

  for (const [key, value] of Object.entries(obj)) {
    if (typeof value === 'string') {
      redacted[key] = redactValue(value);
    } else if (typeof value === 'object' && value !== null && !Array.isArray(value)) {
      // Recursively redact nested objects
      redacted[key] = redactNestedObject(value);
    } else {
      redacted[key] = value;
    }
  }

  return redacted;
}
