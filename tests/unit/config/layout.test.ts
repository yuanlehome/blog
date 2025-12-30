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

    it('should have correct container settings', () => {
      const config = loadLayoutConfig();

      expect(config.container.width).toBe('72rem');
      expect(config.container.paddingX).toBeDefined();
      expect(config.container.paddingX.mobile).toBeDefined();
    });

    it('should have correct sidebar settings', () => {
      const config = loadLayoutConfig();

      expect(config.sidebar.enabled).toBe(true);
      expect(config.sidebar.position).toBe('right');
      expect(config.sidebar.width).toBe('18rem');
      expect(config.sidebar.sticky).toBe(true);
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
            tablet: '1rem',
            desktop: '1rem',
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
          position: 'right' as const,
          mobileBehavior: 'drawer' as const,
          defaultOpen: false,
          offset: 96,
        },
        alignment: {
          headerAlign: 'left' as const,
          footerAlign: 'left' as const,
          postMetaAlign: 'left' as const,
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
          position: 'invalid',
        },
      };

      const result = layoutConfigSchema.safeParse(invalidConfig);
      expect(result.success).toBe(false);
    });

    it('should validate toc position enum', () => {
      const invalidConfig = {
        toc: {
          position: 'invalid',
        },
      };

      const result = layoutConfigSchema.safeParse(invalidConfig);
      expect(result.success).toBe(false);
    });

    it('should validate toc mobileBehavior enum', () => {
      const invalidConfig = {
        toc: {
          mobileBehavior: 'invalid',
        },
      };

      const result = layoutConfigSchema.safeParse(invalidConfig);
      expect(result.success).toBe(false);
    });

    it('should validate toc offset range', () => {
      const invalidConfig = {
        toc: {
          offset: 250, // exceeds max of 200
        },
      };

      const result = layoutConfigSchema.safeParse(invalidConfig);
      expect(result.success).toBe(false);
    });

    it('should allow valid toc offset', () => {
      const validConfig = {
        toc: {
          offset: 100,
        },
      };

      const result = layoutConfigSchema.safeParse(validConfig);
      expect(result.success).toBe(true);
    });

    it('should validate alignment enums', () => {
      const invalidConfig = {
        alignment: {
          headerAlign: 'invalid',
        },
      };

      const result = layoutConfigSchema.safeParse(invalidConfig);
      expect(result.success).toBe(false);
    });

    it('should apply defaults for missing fields', () => {
      const minimalConfig = {};

      const result = layoutConfigSchema.safeParse(minimalConfig);
      expect(result.success).toBe(true);
      if (result.success) {
        expect(result.data.layoutMode).toBe('rightSidebar');
        expect(result.data.sidebar.enabled).toBe(true);
        expect(result.data.toc.offset).toBe(96);
      }
    });
  });

  describe('defaultLayoutConfig', () => {
    it('should have sensible defaults', () => {
      expect(defaultLayoutConfig.layoutMode).toBe('rightSidebar');
      expect(defaultLayoutConfig.sidebar.enabled).toBe(true);
      expect(defaultLayoutConfig.toc.enabled).toBe(true);
      expect(defaultLayoutConfig.container.width).toBe('72rem');
    });

    it('should pass schema validation', () => {
      const result = layoutConfigSchema.safeParse(defaultLayoutConfig);
      expect(result.success).toBe(true);
    });
  });
});
