/**
 * Tests for components configuration loader
 */

import { describe, it, expect } from 'vitest';
import {
  loadComponentsConfig,
  getComponentsConfig,
  componentsConfigSchema,
  defaultComponentsConfig,
} from '../../../src/config/loaders/components';

describe('Components Configuration', () => {
  describe('loadComponentsConfig', () => {
    it('should load and validate components configuration', () => {
      const config = loadComponentsConfig();

      expect(config).toBeDefined();
      expect(config.radius).toBeDefined();
      expect(config.shadow).toBeDefined();
      expect(config.border).toBeDefined();
      expect(config.motion).toBeDefined();
    });

    it('should have correct radius settings', () => {
      const config = loadComponentsConfig();

      expect(config.radius.sm).toBe('0.35rem');
      expect(config.radius.lg).toBe('0.9rem');
    });

    it('should have correct shadow settings', () => {
      const config = loadComponentsConfig();

      expect(config.shadow.card).toBe('md');
      expect(config.shadow.codeBlock).toBe('md');
    });

    it('should have correct motion settings', () => {
      const config = loadComponentsConfig();

      expect(config.motion.enabled).toBe(true);
      expect(config.motion.level).toBe('normal');
      expect(config.motion.respectReducedMotion).toBe(true);
    });
  });

  describe('getComponentsConfig (cached)', () => {
    it('should return cached configuration', () => {
      const config1 = getComponentsConfig();
      const config2 = getComponentsConfig();

      expect(config1).toBe(config2);
    });
  });

  describe('componentsConfigSchema', () => {
    it('should accept valid configuration', () => {
      const validConfig = {
        radius: {
          sm: '0.25rem',
          md: '0.5rem',
          lg: '1rem',
          xl: '1.5rem',
        },
        shadow: {
          card: 'md' as const,
          codeBlock: 'lg' as const,
          header: 'sm' as const,
          hoverLift: true,
        },
        shadowValues: {
          none: 'none',
          sm: '0 1px 2px 0 rgb(0 0 0 / 0.05)',
          md: '0 4px 6px -1px rgb(0 0 0 / 0.1)',
          lg: '0 10px 15px -3px rgb(0 0 0 / 0.1)',
          xl: '0 20px 25px -5px rgb(0 0 0 / 0.1)',
        },
        shadowValuesDark: {
          none: 'none',
          sm: '0 1px 2px 0 rgb(0 0 0 / 0.3)',
          md: '0 10px 30px rgb(0 0 0 / 0.28)',
          lg: '0 20px 40px rgb(0 0 0 / 0.35)',
          xl: '0 25px 50px rgb(0 0 0 / 0.4)',
        },
        border: {
          style: 'solid' as const,
          width: '1px',
        },
        motion: {
          enabled: true,
          level: 'normal' as const,
          respectReducedMotion: true,
        },
        motionTiming: {
          subtle: {
            duration: '100ms',
            easing: 'ease',
          },
          normal: {
            duration: '160ms',
            easing: 'ease',
          },
          energetic: {
            duration: '240ms',
            easing: 'ease-in-out',
          },
        },
        spacingScale: 'comfortable' as const,
        spacingMultiplier: {
          compact: 0.75,
          comfortable: 1.0,
          relaxed: 1.25,
        },
      };

      const result = componentsConfigSchema.safeParse(validConfig);
      expect(result.success).toBe(true);
    });

    it('should validate shadow enum', () => {
      const invalidConfig = {
        shadow: {
          card: 'invalid',
        },
      };

      const result = componentsConfigSchema.safeParse(invalidConfig);
      expect(result.success).toBe(false);
    });

    it('should validate border style enum', () => {
      const invalidConfig = {
        border: {
          style: 'invalid',
        },
      };

      const result = componentsConfigSchema.safeParse(invalidConfig);
      expect(result.success).toBe(false);
    });

    it('should validate motion level enum', () => {
      const invalidConfig = {
        motion: {
          level: 'invalid',
        },
      };

      const result = componentsConfigSchema.safeParse(invalidConfig);
      expect(result.success).toBe(false);
    });

    it('should validate spacingScale enum', () => {
      const invalidConfig = {
        spacingScale: 'invalid',
      };

      const result = componentsConfigSchema.safeParse(invalidConfig);
      expect(result.success).toBe(false);
    });

    it('should validate spacing multiplier range', () => {
      const invalidConfig = {
        spacingMultiplier: {
          compact: 2.0, // exceeds max of 1.5
        },
      };

      const result = componentsConfigSchema.safeParse(invalidConfig);
      expect(result.success).toBe(false);
    });

    it('should validate spacing multiplier minimum', () => {
      const invalidConfig = {
        spacingMultiplier: {
          comfortable: 0.3, // below min of 0.5
        },
      };

      const result = componentsConfigSchema.safeParse(invalidConfig);
      expect(result.success).toBe(false);
    });

    it('should apply defaults for missing fields', () => {
      const minimalConfig = {};

      const result = componentsConfigSchema.safeParse(minimalConfig);
      expect(result.success).toBe(true);
      if (result.success) {
        expect(result.data.shadow.card).toBe('md');
        expect(result.data.motion.level).toBe('normal');
        expect(result.data.spacingScale).toBe('comfortable');
      }
    });
  });

  describe('defaultComponentsConfig', () => {
    it('should have sensible defaults', () => {
      expect(defaultComponentsConfig.radius.lg).toBe('0.9rem');
      expect(defaultComponentsConfig.shadow.card).toBe('md');
      expect(defaultComponentsConfig.motion.enabled).toBe(true);
      expect(defaultComponentsConfig.spacingScale).toBe('comfortable');
    });

    it('should pass schema validation', () => {
      const result = componentsConfigSchema.safeParse(defaultComponentsConfig);
      expect(result.success).toBe(true);
    });
  });
});
