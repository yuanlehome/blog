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
      expect(config.componentRadius).toBeDefined();
      expect(config.shadow).toBeDefined();
      expect(config.componentShadow).toBeDefined();
      expect(config.border).toBeDefined();
      expect(config.motion).toBeDefined();
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
          none: '0',
          sm: '0.375rem',
          default: '0.5rem',
          md: '0.75rem',
          lg: '0.9rem',
          xl: '0.75rem',
          full: '9999px',
        },
        componentRadius: {
          card: '0.75rem',
          button: '0.5rem',
          image: '0.75rem',
          code: '0.9rem',
          input: '0.5rem',
        },
        shadow: {
          none: 'none',
          sm: '0 1px 2px 0 rgb(0 0 0 / 0.05)',
          default: '0 1px 3px 0 rgb(0 0 0 / 0.1)',
          md: '0 4px 6px -1px rgb(0 0 0 / 0.1)',
          lg: '0 10px 15px -3px rgb(0 0 0 / 0.1)',
          xl: '0 20px 25px -5px rgb(0 0 0 / 0.1)',
          '2xl': '0 25px 50px -12px rgb(0 0 0 / 0.25)',
        },
        componentShadow: {
          card: '0 8px 24px rgb(15 23 42 / 0.08)',
          cardDark: '0 10px 30px rgb(0 0 0 / 0.28)',
          header: '0 8px 24px rgb(15 23 42 / 0.08)',
          headerDark: '0 10px 32px rgb(0 0 0 / 0.28)',
          hoverLift: true,
        },
        border: {
          style: 'solid' as const,
          width: '1px',
          opacity: 0.2,
        },
        motion: {
          enabled: true,
          level: 'normal' as const,
          duration: {
            fast: 150,
            normal: 200,
            slow: 300,
          },
          easing: {
            default: 'ease',
            in: 'ease-in',
            out: 'ease-out',
            inOut: 'ease-in-out',
          },
        },
        spacingScale: 'comfortable' as const,
        spacingMultiplier: {
          compact: 0.875,
          comfortable: 1.0,
          relaxed: 1.25,
        },
      };

      const result = componentsConfigSchema.safeParse(validConfig);
      expect(result.success).toBe(true);
    });

    it('should validate border style enum', () => {
      const invalidConfig = {
        border: {
          style: 'double',
        },
      };

      const result = componentsConfigSchema.safeParse(invalidConfig);
      expect(result.success).toBe(false);
    });

    it('should validate motion level enum', () => {
      const invalidConfig = {
        motion: {
          level: 'extreme',
        },
      };

      const result = componentsConfigSchema.safeParse(invalidConfig);
      expect(result.success).toBe(false);
    });

    it('should validate spacing scale enum', () => {
      const invalidConfig = {
        spacingScale: 'huge',
      };

      const result = componentsConfigSchema.safeParse(invalidConfig);
      expect(result.success).toBe(false);
    });

    it('should validate motion duration boundaries', () => {
      const invalidConfigTooSmall = {
        motion: {
          duration: {
            fast: 10,
          },
        },
      };

      const invalidConfigTooLarge = {
        motion: {
          duration: {
            slow: 2000,
          },
        },
      };

      expect(componentsConfigSchema.safeParse(invalidConfigTooSmall).success).toBe(false);
      expect(componentsConfigSchema.safeParse(invalidConfigTooLarge).success).toBe(false);
    });

    it('should validate border opacity boundaries', () => {
      const invalidConfigNegative = {
        border: {
          opacity: -0.1,
        },
      };

      const invalidConfigTooLarge = {
        border: {
          opacity: 1.5,
        },
      };

      expect(componentsConfigSchema.safeParse(invalidConfigNegative).success).toBe(false);
      expect(componentsConfigSchema.safeParse(invalidConfigTooLarge).success).toBe(false);
    });

    it('should validate spacing multiplier boundaries', () => {
      const invalidConfig = {
        spacingMultiplier: {
          compact: 0.3,
        },
      };

      const result = componentsConfigSchema.safeParse(invalidConfig);
      expect(result.success).toBe(false);
    });
  });

  describe('defaultComponentsConfig', () => {
    it('should have sensible defaults', () => {
      expect(defaultComponentsConfig.spacingScale).toBe('comfortable');
      expect(defaultComponentsConfig.motion.enabled).toBe(true);
      expect(defaultComponentsConfig.componentShadow.hoverLift).toBe(true);
    });

    it('should pass schema validation', () => {
      const result = componentsConfigSchema.safeParse(defaultComponentsConfig);
      expect(result.success).toBe(true);
    });
  });
});
