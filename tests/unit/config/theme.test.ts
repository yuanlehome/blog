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
      expect(config.defaultTheme).toBeDefined();
      expect(config.themes).toBeInstanceOf(Array);
      expect(config.storageKey).toBeDefined();
      expect(config.colors).toBeDefined();
      expect(config.darkColors).toBeDefined();
      expect(config.codeTheme).toBeDefined();
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
    it('should accept valid configuration', () => {
      const validConfig = {
        defaultTheme: 'light' as const,
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
          background: {
            base: '#ffffff',
            subtle: '#f8fafc',
            muted: '#f1f5f9',
          },
          foreground: {
            base: '#0f172a',
            muted: '#64748b',
          },
          border: {
            default: '#e2e8f0',
            subtle: '#f1f5f9',
          },
          card: {
            background: '#ffffff',
            border: '#e2e8f0',
          },
          code: {
            background: '#f8fafc',
            foreground: '#0f172a',
            border: '#e5e7eb',
            scrollbar: '#cbd5e1',
          },
        },
        darkColors: {
          brand: '#60a5fa',
          accent: '#a78bfa',
          background: {
            base: '#0f172a',
            subtle: '#1e293b',
            muted: '#334155',
          },
          foreground: {
            base: '#f1f5f9',
            muted: '#94a3b8',
          },
          border: {
            default: '#334155',
            subtle: '#1e293b',
          },
          card: {
            background: '#1e293b',
            border: '#334155',
          },
          code: {
            background: '#0b1221',
            foreground: '#e2e8f0',
            border: '#334155',
            scrollbar: '#475569',
          },
        },
        codeTheme: {
          light: 'github-light',
          dark: 'github-dark',
          showLineNumbers: true,
          showCopyButton: true,
          wrapLongLines: false,
          inlineCodeStyle: 'subtle' as const,
        },
        emphasis: {
          linkUnderline: 'hover' as const,
          focusRing: true,
        },
        animations: {
          respectReducedMotion: true,
          enableScrollEffects: true,
        },
      };

      const result = themeConfigSchema.safeParse(validConfig);
      expect(result.success).toBe(true);
    });

    it('should validate defaultTheme enum', () => {
      const invalidConfig = {
        defaultTheme: 'invalid',
      };

      const result = themeConfigSchema.safeParse(invalidConfig);
      expect(result.success).toBe(false);
    });

    it('should validate hex color format', () => {
      const validHexColors = {
        colors: {
          brand: '#3b82f6',
          accent: '#8b5cf6',
        },
      };

      const result = themeConfigSchema.safeParse(validHexColors);
      expect(result.success).toBe(true);
    });

    it('should validate rgb color format', () => {
      const validRgbColors = {
        colors: {
          brand: 'rgb(59, 130, 246)',
          accent: 'rgba(139, 92, 246, 0.8)',
        },
      };

      const result = themeConfigSchema.safeParse(validRgbColors);
      expect(result.success).toBe(true);
    });

    it('should validate hsl color format', () => {
      const validHslColors = {
        colors: {
          brand: 'hsl(217, 91%, 60%)',
          accent: 'hsla(258, 90%, 66%, 0.8)',
        },
      };

      const result = themeConfigSchema.safeParse(validHslColors);
      expect(result.success).toBe(true);
    });

    it('should reject invalid color format', () => {
      const invalidColors = {
        colors: {
          brand: 'blue',
        },
      };

      const result = themeConfigSchema.safeParse(invalidColors);
      expect(result.success).toBe(false);
    });

    it('should validate codeTheme.inlineCodeStyle enum', () => {
      const invalidConfig = {
        codeTheme: {
          inlineCodeStyle: 'invalid',
        },
      };

      const result = themeConfigSchema.safeParse(invalidConfig);
      expect(result.success).toBe(false);
    });

    it('should validate emphasis.linkUnderline enum', () => {
      const invalidConfig = {
        emphasis: {
          linkUnderline: 'sometimes',
        },
      };

      const result = themeConfigSchema.safeParse(invalidConfig);
      expect(result.success).toBe(false);
    });
  });

  describe('defaultThemeConfig', () => {
    it('should have sensible defaults', () => {
      expect(defaultThemeConfig.defaultTheme).toBeDefined();
      expect(['light', 'dark', 'system']).toContain(defaultThemeConfig.defaultTheme);
      expect(defaultThemeConfig.themes).toBeInstanceOf(Array);
      expect(defaultThemeConfig.colors).toBeDefined();
      expect(defaultThemeConfig.darkColors).toBeDefined();
    });

    it('should pass schema validation', () => {
      const result = themeConfigSchema.safeParse(defaultThemeConfig);
      expect(result.success).toBe(true);
    });

    it('should have valid color values', () => {
      expect(defaultThemeConfig.colors.brand).toMatch(/^#[0-9A-Fa-f]{6}$/);
      expect(defaultThemeConfig.colors.accent).toMatch(/^#[0-9A-Fa-f]{6}$/);
      expect(defaultThemeConfig.darkColors.brand).toMatch(/^#[0-9A-Fa-f]{6}$/);
      expect(defaultThemeConfig.darkColors.accent).toMatch(/^#[0-9A-Fa-f]{6}$/);
    });
  });
});
