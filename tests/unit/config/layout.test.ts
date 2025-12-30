/**
 * Tests for layout configuration loader
 */

import { describe, it, expect } from 'vitest';
import {
  loadLayoutConfig,
  getLayoutConfig,
  layoutConfigSchema,
  defaultLayoutConfig,
} from '../../../src/config/loaders/layout';

describe('Layout Configuration', () => {
  describe('loadLayoutConfig', () => {
    it('should load and validate layout configuration', () => {
      const config = loadLayoutConfig();

      expect(config).toBeDefined();
      expect(config.container).toBeDefined();
      expect(config.layoutMode).toBeDefined();
      expect(config.sidebar).toBeDefined();
      expect(config.toc).toBeDefined();
      expect(config.alignment).toBeDefined();
    });
  });

  describe('getLayoutConfig (cached)', () => {
    it('should return cached configuration', () => {
      const config1 = getLayoutConfig();
      const config2 = getLayoutConfig();

      expect(config1).toBe(config2);
    });
  });

  describe('layoutConfigSchema', () => {
    it('should accept valid configuration', () => {
      const validConfig = {
        container: {
          width: '72rem',
          paddingX: {
            mobile: '1rem',
            tablet: '1.5rem',
            desktop: '2rem',
          },
        },
        layoutMode: 'rightSidebar' as const,
        sidebar: {
          enabled: true,
          position: 'right' as const,
          width: '18rem',
          sticky: true,
          gap: '3rem',
        },
        toc: {
          enabled: true,
          position: 'sidebar' as const,
          mobileBehavior: 'drawer' as const,
          defaultOpen: false,
          stickyOffset: 96,
        },
        alignment: {
          headerAlign: 'left' as const,
          footerAlign: 'center' as const,
          postMetaAlign: 'left' as const,
          contentAlign: 'left' as const,
        },
      };

      const result = layoutConfigSchema.safeParse(validConfig);
      expect(result.success).toBe(true);
    });

    it('should validate layoutMode enum', () => {
      const invalidConfig = {
        layoutMode: 'invalid',
      };

      const result = layoutConfigSchema.safeParse(invalidConfig);
      expect(result.success).toBe(false);
    });

    it('should validate sidebar position enum', () => {
      const invalidConfig = {
        sidebar: {
          position: 'top',
        },
      };

      const result = layoutConfigSchema.safeParse(invalidConfig);
      expect(result.success).toBe(false);
    });

    it('should validate TOC position enum', () => {
      const invalidConfig = {
        toc: {
          position: 'bottom',
        },
      };

      const result = layoutConfigSchema.safeParse(invalidConfig);
      expect(result.success).toBe(false);
    });

    it('should validate stickyOffset boundaries', () => {
      const invalidConfigNegative = {
        toc: {
          stickyOffset: -10,
        },
      };

      const invalidConfigTooLarge = {
        toc: {
          stickyOffset: 300,
        },
      };

      expect(layoutConfigSchema.safeParse(invalidConfigNegative).success).toBe(false);
      expect(layoutConfigSchema.safeParse(invalidConfigTooLarge).success).toBe(false);
    });

    it('should accept valid CSS units', () => {
      const validUnits = {
        container: {
          width: '1200px',
        },
        sidebar: {
          width: '20%',
          gap: '2em',
        },
      };

      const result = layoutConfigSchema.safeParse(validUnits);
      expect(result.success).toBe(true);
    });

    it('should reject invalid CSS units', () => {
      const invalidUnits = {
        container: {
          width: 'invalid',
        },
      };

      const result = layoutConfigSchema.safeParse(invalidUnits);
      expect(result.success).toBe(false);
    });
  });

  describe('defaultLayoutConfig', () => {
    it('should have sensible defaults', () => {
      expect(defaultLayoutConfig.layoutMode).toBe('rightSidebar');
      expect(defaultLayoutConfig.sidebar.enabled).toBe(true);
      expect(defaultLayoutConfig.toc.enabled).toBe(true);
    });

    it('should pass schema validation', () => {
      const result = layoutConfigSchema.safeParse(defaultLayoutConfig);
      expect(result.success).toBe(true);
    });
  });
});
