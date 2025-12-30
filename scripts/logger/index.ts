/**
 * Unified Logging System
 * 
 * A lightweight, structured logging system for script execution
 * 
 * Features:
 * - Standard logging levels (debug/info/warn/error)
 * - Structured logging with fields (JSON support)
 * - Span-based timing (start/end/duration)
 * - Summary logging for script completion
 * - Sensitive data redaction
 * - Multiple output formats (pretty/json)
 * - File output support (JSON Lines)
 * - Child loggers with inherited context
 * 
 * Usage:
 * ```typescript
 * import { createLogger, generateRunId } from './logger/index.js';
 * 
 * const logger = createLogger({
 *   fields: { runId: generateRunId(), script: 'my-script' }
 * });
 * 
 * logger.info('Starting script');
 * 
 * const span = logger.time('fetch-data');
 * // ... do work ...
 * span.end({ status: 'ok' });
 * 
 * logger.summary({ status: 'ok', durationMs: 1234 });
 * ```
 * 
 * Environment Variables:
 * - LOG_LEVEL: debug|info|warn|error (default: info)
 * - LOG_FORMAT: pretty|json (default: pretty)
 * - LOG_FILE: path to log file (optional, JSON Lines format)
 * - LOG_COLOR: 0|1 (default: auto-detect TTY)
 * - LOG_SILENT: 1 (suppress console output, for testing)
 */

export { createLogger } from './logger.js';
export { generateRunId } from './utils.js';
export type {
  Logger,
  LoggerOptions,
  LogFields,
  LogLevel,
  LogFormat,
  Span,
  SpanOptions,
  SpanEndOptions,
  SpanStatus,
  SummaryFields,
} from './types.js';
