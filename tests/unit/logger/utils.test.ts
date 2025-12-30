/**
 * Tests for logger utilities
 */

import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import {
  colors,
  shouldUseColor,
  colorize,
  generateRunId,
  timestamp,
  getLevelPriority,
  shouldLog,
  formatFieldsPretty,
  formatDuration,
} from '../../../scripts/logger/utils';

describe('Logger Utils', () => {
  describe('shouldUseColor', () => {
    const originalEnv = { ...process.env };

    beforeEach(() => {
      // Reset environment
      delete process.env.CI;
      delete process.env.NO_COLOR;
    });

    afterEach(() => {
      process.env = { ...originalEnv };
    });

    it('should respect explicit true', () => {
      expect(shouldUseColor(true)).toBe(true);
    });

    it('should respect explicit false', () => {
      expect(shouldUseColor(false)).toBe(false);
    });

    it('should disable colors in CI', () => {
      process.env.CI = 'true';
      expect(shouldUseColor()).toBe(false);
    });

    it('should disable colors when NO_COLOR is set', () => {
      process.env.NO_COLOR = '1';
      expect(shouldUseColor()).toBe(false);
    });
  });

  describe('colorize', () => {
    it('should apply color when enabled', () => {
      const result = colorize('test', colors.blue, true);
      expect(result).toContain(colors.blue);
      expect(result).toContain(colors.reset);
      expect(result).toContain('test');
    });

    it('should not apply color when disabled', () => {
      const result = colorize('test', colors.blue, false);
      expect(result).toBe('test');
      expect(result).not.toContain(colors.blue);
    });
  });

  describe('generateRunId', () => {
    it('should generate 8 character hex string', () => {
      const runId = generateRunId();
      expect(runId).toMatch(/^[a-f0-9]{8}$/);
    });

    it('should generate unique IDs', () => {
      const id1 = generateRunId();
      const id2 = generateRunId();
      expect(id1).not.toBe(id2);
    });
  });

  describe('timestamp', () => {
    it('should return ISO 8601 format', () => {
      const ts = timestamp();
      expect(ts).toMatch(/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$/);
    });

    it('should be a valid date', () => {
      const ts = timestamp();
      const date = new Date(ts);
      expect(date.toISOString()).toBe(ts);
    });
  });

  describe('getLevelPriority', () => {
    it('should return correct priorities', () => {
      expect(getLevelPriority('debug')).toBe(0);
      expect(getLevelPriority('info')).toBe(1);
      expect(getLevelPriority('warn')).toBe(2);
      expect(getLevelPriority('error')).toBe(3);
    });

    it('should default to info for unknown levels', () => {
      expect(getLevelPriority('unknown')).toBe(1);
    });
  });

  describe('shouldLog', () => {
    it('should allow equal or higher priority levels', () => {
      expect(shouldLog('error', 'debug')).toBe(true);
      expect(shouldLog('warn', 'debug')).toBe(true);
      expect(shouldLog('info', 'info')).toBe(true);
      expect(shouldLog('error', 'error')).toBe(true);
    });

    it('should block lower priority levels', () => {
      expect(shouldLog('debug', 'info')).toBe(false);
      expect(shouldLog('info', 'warn')).toBe(false);
      expect(shouldLog('warn', 'error')).toBe(false);
    });
  });

  describe('formatFieldsPretty', () => {
    it('should format empty fields', () => {
      const result = formatFieldsPretty({}, false);
      expect(result).toBe('');
    });

    it('should format single field', () => {
      const result = formatFieldsPretty({ key: 'value' }, false);
      expect(result).toBe(' key=value');
    });

    it('should format multiple fields', () => {
      const result = formatFieldsPretty({ key1: 'value1', key2: 'value2' }, false);
      expect(result).toContain('key1=value1');
      expect(result).toContain('key2=value2');
    });

    it('should JSON stringify non-string values', () => {
      const result = formatFieldsPretty({ count: 42, active: true }, false);
      expect(result).toContain('count=42');
      expect(result).toContain('active=true');
    });

    it('should apply color when enabled', () => {
      const result = formatFieldsPretty({ key: 'value' }, true);
      expect(result).toContain(colors.blue);
    });
  });

  describe('formatDuration', () => {
    it('should format milliseconds', () => {
      expect(formatDuration(0)).toBe('0ms');
      expect(formatDuration(500)).toBe('500ms');
      expect(formatDuration(999)).toBe('999ms');
    });

    it('should format seconds', () => {
      expect(formatDuration(1000)).toBe('1.00s');
      expect(formatDuration(1500)).toBe('1.50s');
      expect(formatDuration(59999)).toBe('60.00s');
    });

    it('should format minutes', () => {
      expect(formatDuration(60000)).toBe('1.00m');
      expect(formatDuration(90000)).toBe('1.50m');
      expect(formatDuration(3600000)).toBe('60.00m');
    });
  });
});
