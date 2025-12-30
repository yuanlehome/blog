import { describe, expect, it } from 'vitest';
import {
  getComponentsConfig,
  componentsConfigSchema,
} from '../../../src/config/loaders/components';

describe('HeaderMascot Configuration', () => {
  it('should load headerMascot config with default values', () => {
    const config = getComponentsConfig();

    expect(config.headerMascot).toBeDefined();
    expect(config.headerMascot.enabled).toBe(true);
    expect(config.headerMascot.speed).toBe(1.0);
    expect(config.headerMascot.interactive).toBe(true);
    expect(config.headerMascot.hideOnMobile).toBe(true);
  });

  it('should validate headerMascot enabled as boolean', () => {
    const validConfig = {
      headerMascot: {
        enabled: false,
        speed: 1.0,
        interactive: true,
        hideOnMobile: true,
      },
    };

    const result = componentsConfigSchema.safeParse(validConfig);
    expect(result.success).toBe(true);
  });

  it('should validate headerMascot speed within range', () => {
    const validSpeeds = [0.1, 0.5, 1.0, 2.0, 5.0];

    validSpeeds.forEach((speed) => {
      const config = {
        headerMascot: {
          enabled: true,
          speed,
          interactive: true,
          hideOnMobile: true,
        },
      };

      const result = componentsConfigSchema.safeParse(config);
      expect(result.success).toBe(true);
    });
  });

  it('should reject headerMascot speed outside valid range', () => {
    const invalidSpeeds = [0, 0.05, 6.0, 10.0];

    invalidSpeeds.forEach((speed) => {
      const config = {
        headerMascot: {
          enabled: true,
          speed,
          interactive: true,
          hideOnMobile: true,
        },
      };

      const result = componentsConfigSchema.safeParse(config);
      expect(result.success).toBe(false);
    });
  });

  it('should validate headerMascot interactive as boolean', () => {
    const config = {
      headerMascot: {
        enabled: true,
        speed: 1.5,
        interactive: false,
        hideOnMobile: false,
      },
    };

    const result = componentsConfigSchema.safeParse(config);
    expect(result.success).toBe(true);
    if (result.success) {
      expect(result.data.headerMascot.interactive).toBe(false);
      expect(result.data.headerMascot.hideOnMobile).toBe(false);
    }
  });

  it('should use default values when headerMascot is not provided', () => {
    const config = {};

    const result = componentsConfigSchema.safeParse(config);
    expect(result.success).toBe(true);
    if (result.success) {
      expect(result.data.headerMascot).toBeDefined();
      expect(result.data.headerMascot.enabled).toBe(true);
      expect(result.data.headerMascot.speed).toBe(1.0);
    }
  });

  it('should reject non-boolean values for enabled', () => {
    const invalidConfig = {
      headerMascot: {
        enabled: 'yes',
        speed: 1.0,
        interactive: true,
        hideOnMobile: true,
      },
    };

    const result = componentsConfigSchema.safeParse(invalidConfig);
    expect(result.success).toBe(false);
  });

  it('should reject non-numeric values for speed', () => {
    const invalidConfig = {
      headerMascot: {
        enabled: true,
        speed: 'fast',
        interactive: true,
        hideOnMobile: true,
      },
    };

    const result = componentsConfigSchema.safeParse(invalidConfig);
    expect(result.success).toBe(false);
  });
});
