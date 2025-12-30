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
        defaultTheme: "light" as const,
        themes: ["light", "dark"],
        storageKey: "theme",
        icons: {
          light: "☀️",
          dark: "🌙",
        },
        labels: {
          light: "Light",
          dark: "Dark",
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
        defaultTheme: "invalid",
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
    });

    it('should pass schema validation', () => {
      const result = themeConfigSchema.safeParse(defaultThemeConfig);
      expect(result.success).toBe(true);
    });
  });
});
