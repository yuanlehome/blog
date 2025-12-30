/**
 * Tests for typography configuration loader
 */

import { describe, it, expect } from 'vitest';
import {
  loadTypographyConfig,
  getTypographyConfig,
  typographyConfigSchema,
  defaultTypographyConfig,
} from '../../../src/config/loaders/typography';

describe('Typography Configuration', () => {
  describe('loadTypographyConfig', () => {
    it('should load and validate typography configuration', () => {
      const config = loadTypographyConfig();

      expect(config).toBeDefined();
      expect(config.fontFamily).toBeDefined();
      expect(config.fontSize).toBeDefined();
      expect(config.lineHeight).toBeDefined();
      expect(config.fontWeight).toBeDefined();
      expect(config.prose).toBeDefined();
    });

    it('should have correct font family stacks', () => {
      const config = loadTypographyConfig();

      expect(config.fontFamily.sans).toBeInstanceOf(Array);
      expect(config.fontFamily.serif).toBeInstanceOf(Array);
      expect(config.fontFamily.mono).toBeInstanceOf(Array);
      expect(config.fontFamily.mono).toContain('Fira Code');
    });

    it('should have correct font size scale', () => {
      const config = loadTypographyConfig();

      expect(config.fontSize.base).toBe('1rem');
      expect(config.fontSize.xs).toBe('0.75rem');
      expect(config.fontSize['4xl']).toBe('2.25rem');
    });
  });

  describe('getTypographyConfig (cached)', () => {
    it('should return cached configuration', () => {
      const config1 = getTypographyConfig();
      const config2 = getTypographyConfig();

      expect(config1).toBe(config2);
    });
  });

  describe('typographyConfigSchema', () => {
    it('should accept valid configuration', () => {
      const validConfig = {
        fontFamily: {
          sans: ['Arial', 'sans-serif'],
          serif: ['Georgia', 'serif'],
          mono: ['Courier', 'monospace'],
        },
        fontSize: {
          xs: '0.75rem',
          sm: '0.875rem',
          base: '1rem',
          lg: '1.125rem',
          xl: '1.25rem',
          '2xl': '1.5rem',
          '3xl': '1.875rem',
          '4xl': '2.25rem',
        },
        lineHeight: {
          body: 1.75,
          heading: 1.3,
          code: 1.65,
          tight: 1.25,
        },
        fontWeight: {
          normal: 400,
          medium: 500,
          semibold: 600,
          bold: 700,
        },
        prose: {
          maxWidth: 'none',
          headingSpacing: 1.0,
          paragraphSpacing: 1.0,
        },
      };

      const result = typographyConfigSchema.safeParse(validConfig);
      expect(result.success).toBe(true);
    });

    it('should validate lineHeight ranges', () => {
      const invalidConfig = {
        lineHeight: {
          body: 4, // exceeds max of 3
        },
      };

      const result = typographyConfigSchema.safeParse(invalidConfig);
      expect(result.success).toBe(false);
    });

    it('should validate lineHeight minimum', () => {
      const invalidConfig = {
        lineHeight: {
          body: 0.5, // below min of 1
        },
      };

      const result = typographyConfigSchema.safeParse(invalidConfig);
      expect(result.success).toBe(false);
    });

    it('should validate fontWeight ranges', () => {
      const invalidConfig = {
        fontWeight: {
          bold: 1000, // exceeds max of 900
        },
      };

      const result = typographyConfigSchema.safeParse(invalidConfig);
      expect(result.success).toBe(false);
    });

    it('should validate fontWeight minimum', () => {
      const invalidConfig = {
        fontWeight: {
          normal: 50, // below min of 100
        },
      };

      const result = typographyConfigSchema.safeParse(invalidConfig);
      expect(result.success).toBe(false);
    });

    it('should validate prose spacing ranges', () => {
      const invalidConfig = {
        prose: {
          headingSpacing: 3, // exceeds max of 2
        },
      };

      const result = typographyConfigSchema.safeParse(invalidConfig);
      expect(result.success).toBe(false);
    });

    it('should apply defaults for missing fields', () => {
      const minimalConfig = {};

      const result = typographyConfigSchema.safeParse(minimalConfig);
      expect(result.success).toBe(true);
      if (result.success) {
        expect(result.data.fontSize.base).toBe('1rem');
        expect(result.data.lineHeight.body).toBe(1.75);
        expect(result.data.fontFamily.mono).toContain('Fira Code');
      }
    });
  });

  describe('defaultTypographyConfig', () => {
    it('should have sensible defaults', () => {
      expect(defaultTypographyConfig.fontSize.base).toBe('1rem');
      expect(defaultTypographyConfig.lineHeight.body).toBe(1.75);
      expect(defaultTypographyConfig.fontWeight.bold).toBe(700);
      expect(defaultTypographyConfig.fontFamily.mono).toContain('Fira Code');
    });

    it('should pass schema validation', () => {
      const result = typographyConfigSchema.safeParse(defaultTypographyConfig);
      expect(result.success).toBe(true);
    });
  });
});
