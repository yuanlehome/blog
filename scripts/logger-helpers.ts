/**
 * Integration helpers for using logger in scripts
 */

import { createLogger, generateRunId } from './logger/index.js';
import type { Logger } from './logger/types.js';

/**
 * Create a script logger with standard fields
 */
export function createScriptLogger(
  scriptName: string,
  options: { url?: string; slug?: string; provider?: string } = {},
): Logger {
  const runId = generateRunId();
  const fields: Record<string, any> = {
    runId,
    script: scriptName,
  };

  if (options.url) fields.url = options.url;
  if (options.slug) fields.slug = options.slug;
  if (options.provider) fields.provider = options.provider;

  return createLogger({ fields });
}

/**
 * Get current timestamp for duration calculations
 */
export function now(): number {
  return Date.now();
}

/**
 * Calculate duration in milliseconds
 */
export function duration(start: number): number {
  return Date.now() - start;
}
