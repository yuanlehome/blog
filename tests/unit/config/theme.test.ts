/**
 * Tests for theme configuration loader
 */

import { describe, it, expect } from 'vitest';
import {
  loadThemeConfig,
  getThemeConfig,
  themeConfigSchema,
  defaultThemeConfig,
} from '../../../src/config/loaders/theme';

describe('Theme Configuration', () => {
  describe('loadThemeConfig', () => {
    it('should load and validate theme configuration', () => {
      const config = loadThemeConfig();

      expect(config).toBeDefined();
      expect(config.colorMode).toBeDefined();
      expect(config.themes).toBeInstanceOf(Array);
      expect(config.storageKey).toBeDefined();
      expect(config.colors).toBeDefined();
      expect(config.darkColors).toBeDefined();
    });

    it('should have correct color mode settings', () => {
      const config = loadThemeConfig();

      expect(config.colorMode.default).toBe('system');
      expect(config.colorMode.allowToggle).toBe(true);
      expect(config.colorMode.persist).toBe(true);
    });

    it('should have correct color palettes', () => {
      const config = loadThemeConfig();

      expect(config.colors.brand).toBeDefined();
      expect(config.colors.code).toBeDefined();
      expect(config.darkColors.brand).toBeDefined();
      expect(config.darkColors.code).toBeDefined();
    });

    it('should have correct code block settings', () => {
      const config = loadThemeConfig();

      expect(config.codeBlock.showLineNumbers).toBe(true);
      expect(config.codeBlock.showCopyButton).toBe(true);
      expect(config.codeBlock.theme.light).toBe('github-light');
      expect(config.codeBlock.theme.dark).toBe('github-dark');
    });
  });

  describe('getThemeConfig (cached)', () => {
    it('should return cached configuration', () => {
      const config1 = getThemeConfig();
      const config2 = getThemeConfig();

      expect(config1).toBe(config2);
    });
  });

  describe('themeConfigSchema', () => {
    it('should accept valid configuration with all fields', () => {
      const validConfig = {
        colorMode: {
          default: 'light' as const,
          allowToggle: true,
          persist: true,
        },
        themes: ['light', 'dark'],
        storageKey: 'theme',
        icons: {
          light: '☀️',
          dark: '🌙',
        },
        labels: {
          light: 'Light',
          dark: 'Dark',
        },
        colors: {
          brand: '#3b82f6',
          accent: '#8b5cf6',
          background: '#ffffff',
          foreground: '#111827',
          muted: '#6b7280',
          border: '#e5e7eb',
          card: '#f9fafb',
          code: {
            background: '#f8fafc',
            foreground: '#0f172a',
            border: '#e2e8f0',
            keyword: '#3b82f6',
            string: '#10b981',
            comment: '#94a3b8',
            function: '#8b5cf6',
          },
        },
        darkColors: {
          brand: '#60a5fa',
          accent: '#a78bfa',
          background: '#111827',
          foreground: '#f9fafb',
          muted: '#9ca3af',
          border: '#374151',
          card: '#1f2937',
          code: {
            background: '#0b1221',
            foreground: '#e2e8f0',
            border: '#334155',
            keyword: '#60a5fa',
            string: '#34d399',
            comment: '#64748b',
            function: '#a78bfa',
          },
        },
        emphasis: {
          linkUnderline: 'hover' as const,
          focusRing: true,
        },
        codeBlock: {
          theme: {
            light: 'github-light',
            dark: 'github-dark',
          },
          showLineNumbers: true,
          showCopyButton: true,
          wrapLongLines: false,
          inlineCodeStyle: 'subtle' as const,
          radius: '0.9rem',
          enableHighlight: true,
        },
        header: {
          variant: 'default' as const,
          backgroundOpacity: 0.92,
          blurStrength: '10px',
        },
        animations: {
          respectReducedMotion: true,
          enableScrollEffects: true,
        },
      };

      const result = themeConfigSchema.safeParse(validConfig);
      expect(result.success).toBe(true);
    });

    it('should validate colorMode.default enum', () => {
      const invalidConfig = {
        colorMode: {
          default: 'invalid',
        },
      };

      const result = themeConfigSchema.safeParse(invalidConfig);
      expect(result.success).toBe(false);
    });

    it('should validate color format - accept hex', () => {
      const validConfig = {
        colors: {
          brand: '#3b82f6',
        },
      };

      const result = themeConfigSchema.safeParse(validConfig);
      expect(result.success).toBe(true);
    });

    it('should validate color format - accept rgb', () => {
      const validConfig = {
        colors: {
          brand: 'rgb(59, 130, 246)',
        },
      };

      const result = themeConfigSchema.safeParse(validConfig);
      expect(result.success).toBe(true);
    });

    it('should validate color format - accept hsl', () => {
      const validConfig = {
        colors: {
          brand: 'hsl(217, 91%, 60%)',
        },
      };

      const result = themeConfigSchema.safeParse(validConfig);
      expect(result.success).toBe(true);
    });

    it('should reject invalid color format', () => {
      const invalidConfig = {
        colors: {
          brand: 'not-a-color',
        },
      };

      const result = themeConfigSchema.safeParse(invalidConfig);
      expect(result.success).toBe(false);
    });

    it('should validate emphasis.linkUnderline enum', () => {
      const invalidConfig = {
        emphasis: {
          linkUnderline: 'invalid',
        },
      };

      const result = themeConfigSchema.safeParse(invalidConfig);
      expect(result.success).toBe(false);
    });

    it('should validate codeBlock.inlineCodeStyle enum', () => {
      const invalidConfig = {
        codeBlock: {
          inlineCodeStyle: 'invalid',
        },
      };

      const result = themeConfigSchema.safeParse(invalidConfig);
      expect(result.success).toBe(false);
    });

    it('should validate header.variant enum', () => {
      const invalidConfig = {
        header: {
          variant: 'invalid',
        },
      };

      const result = themeConfigSchema.safeParse(invalidConfig);
      expect(result.success).toBe(false);
    });

    it('should validate header.backgroundOpacity range', () => {
      const invalidConfig = {
        header: {
          backgroundOpacity: 1.5, // exceeds max of 1
        },
      };

      const result = themeConfigSchema.safeParse(invalidConfig);
      expect(result.success).toBe(false);
    });

    it('should validate header.backgroundOpacity minimum', () => {
      const invalidConfig = {
        header: {
          backgroundOpacity: -0.1, // below min of 0
        },
      };

      const result = themeConfigSchema.safeParse(invalidConfig);
      expect(result.success).toBe(false);
    });

    it('should apply defaults for missing fields', () => {
      const minimalConfig = {};

      const result = themeConfigSchema.safeParse(minimalConfig);
      expect(result.success).toBe(true);
      if (result.success) {
        expect(result.data.colorMode.default).toBe('system');
        expect(result.data.codeBlock.showLineNumbers).toBe(true);
        expect(result.data.colors.brand).toBe('#0284c7');
      }
    });
  });

  describe('defaultThemeConfig', () => {
    it('should have sensible defaults', () => {
      expect(defaultThemeConfig.colorMode.default).toBe('system');
      expect(['light', 'dark', 'system']).toContain(defaultThemeConfig.colorMode.default);
      expect(defaultThemeConfig.themes).toBeInstanceOf(Array);
      expect(defaultThemeConfig.colors.brand).toBe('#0284c7');
      expect(defaultThemeConfig.darkColors.brand).toBe('#38bdf8');
    });

    it('should pass schema validation', () => {
      const result = themeConfigSchema.safeParse(defaultThemeConfig);
      expect(result.success).toBe(true);
    });
  });
});
