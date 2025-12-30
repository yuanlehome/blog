/**
 * Tests for core Logger implementation
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import fs from 'fs';
import path from 'path';
import { createLogger } from '../../../scripts/logger/logger';

describe('Logger', () => {
  let originalEnv: NodeJS.ProcessEnv;
  let consoleLogSpy: ReturnType<typeof vi.spyOn>;
  let consoleErrorSpy: ReturnType<typeof vi.spyOn>;

  beforeEach(() => {
    originalEnv = { ...process.env };
    consoleLogSpy = vi.spyOn(console, 'log').mockImplementation(() => {});
    consoleErrorSpy = vi.spyOn(console, 'error').mockImplementation(() => {});
  });

  afterEach(() => {
    process.env = originalEnv;
    consoleLogSpy.mockRestore();
    consoleErrorSpy.mockRestore();
  });

  describe('createLogger', () => {
    it('should create a logger with default options', () => {
      const logger = createLogger();
      expect(logger).toBeDefined();
      expect(logger.info).toBeDefined();
      expect(logger.warn).toBeDefined();
      expect(logger.error).toBeDefined();
      expect(logger.debug).toBeDefined();
    });

    it('should respect LOG_LEVEL environment variable', () => {
      process.env.LOG_LEVEL = 'error';
      const logger = createLogger();

      logger.info('info message');
      logger.warn('warn message');
      logger.error('error message');

      // Only error should be logged
      expect(consoleLogSpy).toHaveBeenCalledTimes(1);
      expect(consoleLogSpy.mock.calls[0][0]).toContain('ERROR');
    });

    it('should respect LOG_FORMAT environment variable', () => {
      process.env.LOG_FORMAT = 'json';
      const logger = createLogger();

      logger.info('test message', { normalKey: 'value' });

      expect(consoleLogSpy).toHaveBeenCalled();
      const output = consoleLogSpy.mock.calls[0][0];
      const parsed = JSON.parse(output);
      expect(parsed.level).toBe('info');
      expect(parsed.msg).toBe('test message');
      expect(parsed.normalKey).toBe('value');
    });

    it('should respect LOG_SILENT environment variable', () => {
      process.env.LOG_SILENT = '1';
      const logger = createLogger();

      logger.info('test message');
      logger.warn('test warning');
      logger.error('test error');

      expect(consoleLogSpy).not.toHaveBeenCalled();
      expect(consoleErrorSpy).not.toHaveBeenCalled();
    });
  });

  describe('Logging Methods', () => {
    it('should log info messages', () => {
      const logger = createLogger({ format: 'json', silent: false });
      logger.info('info message');

      expect(consoleLogSpy).toHaveBeenCalled();
      const output = JSON.parse(consoleLogSpy.mock.calls[0][0]);
      expect(output.level).toBe('info');
      expect(output.msg).toBe('info message');
    });

    it('should log warn messages', () => {
      const logger = createLogger({ format: 'json', silent: false });
      logger.warn('warn message');

      expect(consoleLogSpy).toHaveBeenCalled();
      const output = JSON.parse(consoleLogSpy.mock.calls[0][0]);
      expect(output.level).toBe('warn');
      expect(output.msg).toBe('warn message');
    });

    it('should log error messages', () => {
      const logger = createLogger({ format: 'json', silent: false });
      logger.error('error message');

      expect(consoleLogSpy).toHaveBeenCalled();
      const output = JSON.parse(consoleLogSpy.mock.calls[0][0]);
      expect(output.level).toBe('error');
      expect(output.msg).toBe('error message');
    });

    it('should log debug messages', () => {
      const logger = createLogger({ level: 'debug', format: 'json', silent: false });
      logger.debug('debug message');

      expect(consoleLogSpy).toHaveBeenCalled();
      const output = JSON.parse(consoleLogSpy.mock.calls[0][0]);
      expect(output.level).toBe('debug');
      expect(output.msg).toBe('debug message');
    });

    it('should log Error objects', () => {
      const logger = createLogger({ format: 'json', silent: false });
      const error = new Error('Test error');
      logger.error(error);

      expect(consoleLogSpy).toHaveBeenCalled();
      const output = JSON.parse(consoleLogSpy.mock.calls[0][0]);
      expect(output.level).toBe('error');
      expect(output.msg).toBe('Test error');
      expect(output.error).toBeDefined();
      expect(output.error.message).toBe('Test error');
      expect(output.error.stack).toBeDefined();
    });

    it('should include additional fields', () => {
      const logger = createLogger({ format: 'json', silent: false });
      logger.info('test message', { userId: 123, action: 'login' });

      const output = JSON.parse(consoleLogSpy.mock.calls[0][0]);
      expect(output.userId).toBe(123);
      expect(output.action).toBe('login');
    });
  });

  describe('Level Filtering', () => {
    it('should filter by log level', () => {
      const logger = createLogger({ level: 'warn', silent: false });

      logger.debug('debug');
      logger.info('info');
      logger.warn('warn');
      logger.error('error');

      expect(consoleLogSpy).toHaveBeenCalledTimes(2); // warn + error
    });
  });

  describe('Pretty Format', () => {
    it('should output human-readable format', () => {
      const logger = createLogger({ format: 'pretty', color: false, silent: false });
      logger.info('test message', { normalKey: 'value' });

      expect(consoleLogSpy).toHaveBeenCalled();
      const output = consoleLogSpy.mock.calls[0][0];
      expect(output).toContain('INFO');
      expect(output).toContain('test message');
      expect(output).toContain('normalKey=value');
    });

    it('should not include ANSI codes when color is disabled', () => {
      const logger = createLogger({ format: 'pretty', color: false, silent: false });
      logger.info('test');

      const output = consoleLogSpy.mock.calls[0][0];
      expect(output).not.toMatch(/\x1b\[\d+m/);
    });
  });

  describe('JSON Format', () => {
    it('should output valid JSON', () => {
      const logger = createLogger({ format: 'json', silent: false });
      logger.info('test', { key: 'value' });

      const output = consoleLogSpy.mock.calls[0][0];
      expect(() => JSON.parse(output)).not.toThrow();
    });

    it('should include timestamp', () => {
      const logger = createLogger({ format: 'json', silent: false });
      logger.info('test');

      const output = JSON.parse(consoleLogSpy.mock.calls[0][0]);
      expect(output.ts).toMatch(/^\d{4}-\d{2}-\d{2}T/);
    });

    it('should include all standard fields', () => {
      const logger = createLogger({ format: 'json', silent: false });
      logger.info('test message', { custom: 'field' });

      const output = JSON.parse(consoleLogSpy.mock.calls[0][0]);
      expect(output.ts).toBeDefined();
      expect(output.level).toBe('info');
      expect(output.msg).toBe('test message');
      expect(output.custom).toBe('field');
    });
  });

  describe('Sensitive Data Redaction', () => {
    it('should redact password fields', () => {
      const logger = createLogger({ format: 'json', silent: false });
      logger.info('login', { username: 'alice', password: 'secret123' });

      const output = JSON.parse(consoleLogSpy.mock.calls[0][0]);
      expect(output.username).toBe('alice');
      expect(output.password).toBe('[REDACTED]');
    });

    it('should redact token fields', () => {
      const logger = createLogger({ format: 'json', silent: false });
      logger.info('auth', { token: 'abc123', apiKey: 'key456' });

      const output = JSON.parse(consoleLogSpy.mock.calls[0][0]);
      expect(output.token).toBe('[REDACTED]');
      expect(output.apiKey).toBe('[REDACTED]');
    });

    it('should redact Bearer tokens in strings', () => {
      const logger = createLogger({ format: 'json', silent: false });
      logger.info('request', {
        headers: 'Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.longtoken',
      });

      const output = JSON.parse(consoleLogSpy.mock.calls[0][0]);
      expect(output.headers).toContain('Bearer eyJhbG');
      expect(output.headers).not.toContain('longtoken');
    });

    it('should support custom redact keys', () => {
      const logger = createLogger({
        format: 'json',
        silent: false,
        redactKeys: ['customSecret'],
      });
      logger.info('test', { customSecret: 'value', normalField: 'public' });

      const output = JSON.parse(consoleLogSpy.mock.calls[0][0]);
      expect(output.customSecret).toBe('[REDACTED]');
      expect(output.normalField).toBe('public');
    });
  });

  describe('Child Logger', () => {
    it('should create child logger with inherited fields', () => {
      const parent = createLogger({ format: 'json', silent: false, fields: { runId: 'abc123' } });
      const child = parent.child({ script: 'test-script' });

      child.info('test message');

      const output = JSON.parse(consoleLogSpy.mock.calls[0][0]);
      expect(output.runId).toBe('abc123');
      expect(output.script).toBe('test-script');
    });

    it('should not affect parent logger', () => {
      const parent = createLogger({ format: 'json', silent: false, fields: { runId: 'abc123' } });
      const child = parent.child({ script: 'test-script' });

      parent.info('parent message');

      const output = JSON.parse(consoleLogSpy.mock.calls[0][0]);
      expect(output.runId).toBe('abc123');
      expect(output.script).toBeUndefined();
    });

    it('should allow multiple levels of nesting', () => {
      const root = createLogger({ format: 'json', silent: false, fields: { rootField: 'root' } });
      const child = root.child({ level2: 'child' });
      const grandchild = child.child({ level3: 'grandchild' });

      grandchild.info('test');

      const output = JSON.parse(consoleLogSpy.mock.calls[0][0]);
      expect(output.level).toBe('info'); // log level
      expect(output.rootField).toBe('root');
      expect(output.level2).toBe('child');
      expect(output.level3).toBe('grandchild');
    });
  });

  describe('Span Timing', () => {
    beforeEach(() => {
      vi.useFakeTimers();
    });

    afterEach(() => {
      vi.useRealTimers();
    });

    it('should log span start and end', () => {
      const logger = createLogger({ format: 'json', silent: false });
      const span = logger.span({ name: 'test-operation' });

      span.start();
      vi.advanceTimersByTime(1000);
      span.end();

      expect(consoleLogSpy).toHaveBeenCalledTimes(2);

      const startLog = JSON.parse(consoleLogSpy.mock.calls[0][0]);
      expect(startLog.event).toBe('span.start');
      expect(startLog.span).toBe('test-operation');

      const endLog = JSON.parse(consoleLogSpy.mock.calls[1][0]);
      expect(endLog.event).toBe('span.end');
      expect(endLog.span).toBe('test-operation');
      expect(endLog.durationMs).toBeGreaterThanOrEqual(1000);
    });

    it('should include span status', () => {
      const logger = createLogger({ format: 'json', silent: false });
      const span = logger.span({ name: 'test-operation' });

      span.start();
      span.end({ status: 'fail' });

      const endLog = JSON.parse(consoleLogSpy.mock.calls[1][0]);
      expect(endLog.status).toBe('fail');
    });

    it('should include additional fields', () => {
      const logger = createLogger({ format: 'json', silent: false });
      const span = logger.span({ name: 'test-operation', fields: { type: 'fetch' } });

      span.start();
      span.end({ fields: { count: 42 } });

      const startLog = JSON.parse(consoleLogSpy.mock.calls[0][0]);
      expect(startLog.type).toBe('fetch');

      const endLog = JSON.parse(consoleLogSpy.mock.calls[1][0]);
      expect(endLog.type).toBe('fetch');
      expect(endLog.count).toBe(42);
    });

    it('should auto-start span on end if not started', () => {
      const logger = createLogger({ format: 'json', silent: false });
      const span = logger.span({ name: 'test-operation' });

      span.end();

      expect(consoleLogSpy).toHaveBeenCalledTimes(2); // start + end
    });
  });

  describe('time() shorthand', () => {
    beforeEach(() => {
      vi.useFakeTimers();
    });

    afterEach(() => {
      vi.useRealTimers();
    });

    it('should auto-start the span', () => {
      const logger = createLogger({ format: 'json', silent: false });
      const span = logger.time('quick-operation');

      expect(consoleLogSpy).toHaveBeenCalledTimes(1);
      const startLog = JSON.parse(consoleLogSpy.mock.calls[0][0]);
      expect(startLog.event).toBe('span.start');

      vi.advanceTimersByTime(500);
      span.end();

      const endLog = JSON.parse(consoleLogSpy.mock.calls[1][0]);
      expect(endLog.durationMs).toBeGreaterThanOrEqual(500);
    });
  });

  describe('summary()', () => {
    it('should log summary event', () => {
      const logger = createLogger({ format: 'json', silent: false });
      logger.summary({
        status: 'ok',
        durationMs: 5000,
        files: ['output.md'],
        stats: { images: 5, paragraphs: 20 },
      });

      expect(consoleLogSpy).toHaveBeenCalled();
      const output = JSON.parse(consoleLogSpy.mock.calls[0][0]);
      expect(output.event).toBe('summary');
      expect(output.status).toBe('ok');
      expect(output.durationMs).toBe(5000);
      expect(output.files).toEqual(['output.md']);
      expect(output.stats).toEqual({ images: 5, paragraphs: 20 });
    });
  });

  describe('File Output', () => {
    const testLogFile = '/tmp/test-logger.jsonl';

    beforeEach(() => {
      // Clean up before each test
      if (fs.existsSync(testLogFile)) {
        fs.unlinkSync(testLogFile);
      }
    });

    afterEach(() => {
      if (fs.existsSync(testLogFile)) {
        fs.unlinkSync(testLogFile);
      }
      const deepDir = '/tmp/deep';
      if (fs.existsSync(deepDir)) {
        fs.rmSync(deepDir, { recursive: true, force: true });
      }
    });

    it('should write logs to file in JSON Lines format', async () => {
      const logger = createLogger({ filePath: testLogFile, silent: false });

      logger.info('message 1');
      logger.info('message 2');

      // Close logger to flush file
      (logger as any).close();

      // Wait a bit for file operations
      await new Promise((resolve) => setTimeout(resolve, 100));

      expect(fs.existsSync(testLogFile)).toBe(true);
      const content = fs.readFileSync(testLogFile, 'utf-8');
      const lines = content
        .trim()
        .split('\n')
        .filter((l) => l.length > 0);

      expect(lines.length).toBeGreaterThanOrEqual(2);
      const log1 = JSON.parse(lines[0]);
      const log2 = JSON.parse(lines[1]);
      expect(log1.msg).toBe('message 1');
      expect(log2.msg).toBe('message 2');
    });

    it('should create directory if it does not exist', async () => {
      const deepPath = '/tmp/deep/nested/dir/test.jsonl';
      const logger = createLogger({ filePath: deepPath, silent: false });

      logger.info('test');
      (logger as any).close();

      // Wait for file operations
      await new Promise((resolve) => setTimeout(resolve, 100));

      expect(fs.existsSync(deepPath)).toBe(true);
    });

    it('should append to existing file', async () => {
      fs.writeFileSync(testLogFile, '{"existing":"log"}\n');

      const logger = createLogger({ filePath: testLogFile, silent: false });
      logger.info('new message');
      (logger as any).close();

      // Wait for file operations
      await new Promise((resolve) => setTimeout(resolve, 100));

      const content = fs.readFileSync(testLogFile, 'utf-8');
      const lines = content
        .trim()
        .split('\n')
        .filter((l) => l.length > 0);
      expect(lines.length).toBeGreaterThanOrEqual(2);
      expect(lines[0]).toBe('{"existing":"log"}');
      const newLog = JSON.parse(lines[1]);
      expect(newLog.msg).toBe('new message');
    });
  });
});
