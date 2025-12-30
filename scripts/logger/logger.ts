/**
 * Core Logger Implementation
 */

import fs from 'fs';
import path from 'path';
import type {
  Logger,
  LoggerOptions,
  LogFields,
  LogLevel,
  Span,
  SpanOptions,
  SpanEndOptions,
} from './types.js';
import { redactFields, sanitizeError, truncateString } from './redaction.js';
import {
  colors,
  colorize,
  shouldUseColor,
  timestamp,
  shouldLog,
  formatFieldsPretty,
  formatDuration,
} from './utils.js';

/**
 * Internal log entry structure
 */
interface LogEntry {
  ts: string;
  level: LogLevel;
  msg: string;
  runId?: string;
  script?: string;
  event?: string;
  durationMs?: number;
  status?: string;
  error?: any;
  [key: string]: any;
}

/**
 * Span implementation for timing operations
 */
class SpanImpl implements Span {
  private logger: LoggerImpl;
  private name: string;
  private fields: LogFields;
  private startTime?: number;
  private started: boolean = false;

  constructor(logger: LoggerImpl, name: string, fields: LogFields) {
    this.logger = logger;
    this.name = name;
    this.fields = fields;
  }

  start(): void {
    if (this.started) {
      return;
    }
    this.started = true;
    this.startTime = Date.now();
    this.logger.log('info', `${this.name} started`, {
      ...this.fields,
      event: 'span.start',
      span: this.name,
    });
  }

  end(options: SpanEndOptions = {}): void {
    if (!this.started) {
      this.start();
    }

    const endTime = Date.now();
    const durationMs = this.startTime ? endTime - this.startTime : 0;
    const status = options.status || 'ok';

    this.logger.log('info', `${this.name} completed`, {
      ...this.fields,
      ...options.fields,
      event: 'span.end',
      span: this.name,
      status,
      durationMs,
    });
  }
}

/**
 * Logger implementation
 */
class LoggerImpl implements Logger {
  private options: Required<LoggerOptions>;
  private fileStream?: fs.WriteStream;
  private baseFields: LogFields;
  private useColor: boolean;

  constructor(options: LoggerOptions = {}) {
    this.options = {
      level: options.level || this.getEnvLevel(),
      format: options.format || this.getEnvFormat(),
      filePath: options.filePath || process.env.LOG_FILE || '',
      color: options.color !== undefined ? options.color : shouldUseColor(),
      silent: options.silent || process.env.LOG_SILENT === '1',
      fields: options.fields || {},
      redactKeys: options.redactKeys || [],
    };

    this.baseFields = { ...this.options.fields };
    this.useColor = this.options.color && this.options.format === 'pretty';

    // Initialize file stream if needed
    if (this.options.filePath && !this.options.silent) {
      this.initFileStream();
    }
  }

  private getEnvLevel(): LogLevel {
    const level = process.env.LOG_LEVEL?.toLowerCase();
    if (level === 'debug' || level === 'info' || level === 'warn' || level === 'error') {
      return level;
    }
    return 'info';
  }

  private getEnvFormat(): 'pretty' | 'json' {
    const format = process.env.LOG_FORMAT?.toLowerCase();
    return format === 'json' ? 'json' : 'pretty';
  }

  private initFileStream(): void {
    try {
      const dir = path.dirname(this.options.filePath);
      if (!fs.existsSync(dir)) {
        fs.mkdirSync(dir, { recursive: true });
      }
      this.fileStream = fs.createWriteStream(this.options.filePath, { flags: 'a' });
    } catch (error) {
      console.error(`Failed to initialize log file: ${error}`);
    }
  }

  /**
   * Internal logging method
   */
  log(level: LogLevel, message: string, fields: LogFields = {}): void {
    if (!shouldLog(level, this.options.level)) {
      return;
    }

    // Merge fields with base fields and redact
    const allFields = { ...this.baseFields, ...fields };
    const redactedFields = redactFields(allFields, this.options.redactKeys);

    // Build log entry
    const entry: LogEntry = {
      ts: timestamp(),
      level,
      msg: message,
      ...redactedFields,
    };

    // Format output
    if (this.options.format === 'json') {
      this.emitJson(entry);
    } else {
      this.emitPretty(level, message, redactedFields);
    }

    // Write to file if configured
    if (this.fileStream) {
      try {
        this.fileStream.write(JSON.stringify(entry) + '\n');
        // Force flush for testing
        if (process.env.NODE_ENV === 'test') {
          (this.fileStream as any).fd && fs.fdatasyncSync((this.fileStream as any).fd);
        }
      } catch (error) {
        // Ignore write errors silently to avoid breaking the script
      }
    }
  }

  private emitJson(entry: LogEntry): void {
    if (this.options.silent) {
      return;
    }
    console.log(JSON.stringify(entry));
  }

  private emitPretty(level: LogLevel, message: string, fields: LogFields): void {
    if (this.options.silent) {
      return;
    }

    const levelColor = colors[level] || colors.info;
    const levelStr = colorize(level.toUpperCase().padEnd(5), levelColor, this.useColor);
    const timeStr = colorize(new Date().toLocaleTimeString(), colors.gray, this.useColor);
    const fieldsStr = formatFieldsPretty(fields, this.useColor);

    const output = `${timeStr} ${levelStr} ${message}${fieldsStr}`;

    // All output goes to console.log for testing consistency
    console.log(output);
  }

  info(message: string, fields?: LogFields): void {
    this.log('info', message, fields);
  }

  warn(message: string, fields?: LogFields): void {
    this.log('warn', message, fields);
  }

  error(messageOrError: string | Error, fields?: LogFields): void {
    if (messageOrError instanceof Error) {
      const sanitized = sanitizeError(messageOrError, true);
      this.log('error', messageOrError.message, {
        ...fields,
        error: sanitized,
      });
    } else {
      this.log('error', messageOrError, fields);
    }
  }

  debug(message: string, fields?: LogFields): void {
    this.log('debug', message, fields);
  }

  child(fields: LogFields): Logger {
    return new LoggerImpl({
      ...this.options,
      fields: { ...this.baseFields, ...fields },
    });
  }

  span(options: SpanOptions): Span {
    return new SpanImpl(this, options.name, options.fields || {});
  }

  time(name: string): Span {
    const span = this.span({ name });
    span.start();
    return span;
  }

  summary(fields: LogFields): void {
    this.log('info', 'Summary', {
      ...fields,
      event: 'summary',
    });
  }

  /**
   * Close file stream (call at script end)
   */
  close(): void {
    if (this.fileStream) {
      this.fileStream.end();
      // Wait for file to close
      this.fileStream.once('finish', () => {
        this.fileStream = undefined;
      });
    }
  }
}

/**
 * Create a new logger instance
 */
export function createLogger(options: LoggerOptions = {}): Logger {
  return new LoggerImpl(options);
}
