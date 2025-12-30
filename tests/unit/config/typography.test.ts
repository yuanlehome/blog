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
          sans: ['system-ui', 'sans-serif'],
          serif: ['Georgia', 'serif'],
          mono: ['Menlo', 'monospace'],
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
          tight: 1.25,
          snug: 1.375,
          normal: 1.5,
          relaxed: 1.625,
          loose: 1.75,
        },
        fontWeight: {
          normal: 400,
          medium: 500,
          semibold: 600,
          bold: 700,
        },
        prose: {
          maxWidth: '65ch',
          useSerif: false,
          paragraphSpacing: '1.25em',
          headingSpacing: {
            before: '1.5em',
            after: '0.5em',
          },
        },
      };

      const result = typographyConfigSchema.safeParse(validConfig);
      expect(result.success).toBe(true);
    });

    it('should validate font size units', () => {
      const invalidConfig = {
        fontSize: {
          base: '16',
        },
      };

      const result = typographyConfigSchema.safeParse(invalidConfig);
      expect(result.success).toBe(false);
    });

    it('should accept valid font size units', () => {
      const validConfig = {
        fontSize: {
          base: '16px',
          lg: '1.125rem',
          xl: '1.25em',
        },
      };

      const result = typographyConfigSchema.safeParse(validConfig);
      expect(result.success).toBe(true);
    });

    it('should validate line height boundaries', () => {
      const invalidConfigTooSmall = {
        lineHeight: {
          tight: 0.5,
        },
      };

      const invalidConfigTooLarge = {
        lineHeight: {
          loose: 3,
        },
      };

      expect(typographyConfigSchema.safeParse(invalidConfigTooSmall).success).toBe(false);
      expect(typographyConfigSchema.safeParse(invalidConfigTooLarge).success).toBe(false);
    });

    it('should validate font weight boundaries', () => {
      const invalidConfigTooSmall = {
        fontWeight: {
          normal: 50,
        },
      };

      const invalidConfigTooLarge = {
        fontWeight: {
          bold: 1000,
        },
      };

      expect(typographyConfigSchema.safeParse(invalidConfigTooSmall).success).toBe(false);
      expect(typographyConfigSchema.safeParse(invalidConfigTooLarge).success).toBe(false);
    });
  });

  describe('defaultTypographyConfig', () => {
    it('should have sensible defaults', () => {
      expect(defaultTypographyConfig.fontSize.base).toBe('1rem');
      expect(defaultTypographyConfig.lineHeight.normal).toBe(1.5);
      expect(defaultTypographyConfig.fontWeight.normal).toBe(400);
      expect(defaultTypographyConfig.prose.useSerif).toBe(false);
    });

    it('should pass schema validation', () => {
      const result = typographyConfigSchema.safeParse(defaultTypographyConfig);
      expect(result.success).toBe(true);
    });
  });
});
