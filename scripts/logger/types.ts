/**
 * Type definitions for unified logging system
 */

export type LogLevel = 'debug' | 'info' | 'warn' | 'error';

export type LogFormat = 'pretty' | 'json';

export type SpanStatus = 'ok' | 'fail' | 'pending';

export interface LoggerOptions {
  /** Minimum log level to output */
  level?: LogLevel;
  /** Output format: pretty (human-readable) or json (machine-readable) */
  format?: LogFormat;
  /** Optional file path to write JSON Lines logs */
  filePath?: string;
  /** Enable colored output (default: auto-detect based on TTY) */
  color?: boolean;
  /** Silent mode for testing (suppress console output) */
  silent?: boolean;
  /** Additional fields to include in all logs (e.g., script name, runId) */
  fields?: Record<string, any>;
  /** Custom keys to redact (in addition to defaults) */
  redactKeys?: string[];
}

export interface LogFields {
  [key: string]: any;
}

export interface SpanOptions {
  /** Span name */
  name: string;
  /** Additional fields for the span */
  fields?: LogFields;
}

export interface SpanEndOptions {
  /** Span status: ok, fail, or pending */
  status?: SpanStatus;
  /** Additional fields to log at span end */
  fields?: LogFields;
}

export interface Span {
  /** Start the span and log the start event */
  start(): void;
  /** End the span and log the end event with duration */
  end(options?: SpanEndOptions): void;
}

export interface Logger {
  /** Log info message */
  info(message: string, fields?: LogFields): void;
  /** Log warning message */
  warn(message: string, fields?: LogFields): void;
  /** Log error message (supports Error objects) */
  error(message: string | Error, fields?: LogFields): void;
  /** Log debug message */
  debug(message: string, fields?: LogFields): void;
  /** Create a child logger with additional fields */
  child(fields: LogFields): Logger;
  /** Create and return a span for timing operations */
  span(options: SpanOptions): Span;
  /** Shorthand for creating and starting a timer */
  time(name: string): Span;
  /** Log a summary at the end of script execution */
  summary(fields: LogFields): void;
}

export interface SummaryFields extends LogFields {
  /** Summary status: ok or fail */
  status: 'ok' | 'fail';
  /** Total duration in milliseconds */
  durationMs?: number;
  /** Generated file paths */
  files?: string[];
  /** Any statistics to report */
  stats?: Record<string, number>;
}
