/**
 * Utilities for logger implementation
 */

import crypto from 'crypto';

/** ANSI color codes for pretty output */
export const colors = {
  reset: '\x1b[0m',
  bold: '\x1b[1m',
  dim: '\x1b[2m',
  
  // Level colors
  debug: '\x1b[36m', // cyan
  info: '\x1b[32m',  // green
  warn: '\x1b[33m',  // yellow
  error: '\x1b[31m', // red
  
  // Semantic colors
  gray: '\x1b[90m',
  blue: '\x1b[34m',
  magenta: '\x1b[35m',
} as const;

/**
 * Check if output should be colorized
 */
export function shouldUseColor(forceColor?: boolean): boolean {
  if (forceColor !== undefined) {
    return forceColor;
  }
  
  // Disable colors in CI environments
  if (process.env.CI === 'true' || process.env.NO_COLOR === '1') {
    return false;
  }
  
  // Enable colors if stdout is a TTY
  return process.stdout.isTTY ?? false;
}

/**
 * Apply color to text
 */
export function colorize(text: string, color: string, enabled: boolean): string {
  if (!enabled) {
    return text;
  }
  return `${color}${text}${colors.reset}`;
}

/**
 * Generate a short run ID
 */
export function generateRunId(): string {
  return crypto.randomBytes(4).toString('hex');
}

/**
 * Get current timestamp in ISO format
 */
export function timestamp(): string {
  return new Date().toISOString();
}

/**
 * Get log level numeric priority
 */
export function getLevelPriority(level: string): number {
  const priorities: Record<string, number> = {
    debug: 0,
    info: 1,
    warn: 2,
    error: 3,
  };
  return priorities[level] ?? 1;
}

/**
 * Check if a log should be emitted based on level
 */
export function shouldLog(messageLevel: string, minLevel: string): boolean {
  return getLevelPriority(messageLevel) >= getLevelPriority(minLevel);
}

/**
 * Format fields for pretty output
 */
export function formatFieldsPretty(fields: Record<string, any>, useColor: boolean): string {
  if (Object.keys(fields).length === 0) {
    return '';
  }
  
  const formatted = Object.entries(fields)
    .map(([key, value]) => {
      const valueStr = typeof value === 'string' ? value : JSON.stringify(value);
      const keyColored = colorize(key, colors.blue, useColor);
      return `${keyColored}=${valueStr}`;
    })
    .join(' ');
  
  return ` ${formatted}`;
}

/**
 * Format duration in human-readable format
 */
export function formatDuration(ms: number): string {
  if (ms < 1000) {
    return `${ms.toFixed(0)}ms`;
  }
  if (ms < 60000) {
    return `${(ms / 1000).toFixed(2)}s`;
  }
  return `${(ms / 60000).toFixed(2)}m`;
}
