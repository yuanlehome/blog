/**
 * Tests for navigation configuration loader
 */

import { describe, it, expect } from 'vitest';
import {
  loadNavConfig,
  getNavConfig,
  navConfigSchema,
  defaultNavConfig,
} from '../../../src/config/loaders/nav';

describe('Navigation Configuration', () => {
  describe('loadNavConfig', () => {
    it('should load and validate navigation configuration', () => {
      const config = loadNavConfig();

      expect(config).toBeDefined();
      expect(config.header).toBeDefined();
      expect(config.header.brandText).toBeDefined();
      expect(config.header.menuItems).toBeInstanceOf(Array);
      expect(config.theme).toBeDefined();
    });

    it('should have menu items with required fields', () => {
      const config = loadNavConfig();

      if (config.header.menuItems.length > 0) {
        const firstItem = config.header.menuItems[0];
        expect(firstItem.label).toBeDefined();
        expect(firstItem.href).toBeDefined();
        expect(typeof firstItem.isExternal).toBe('boolean');
      }
    });
  });

  describe('getNavConfig (cached)', () => {
    it('should return cached configuration', () => {
      const config1 = getNavConfig();
      const config2 = getNavConfig();

      expect(config1).toBe(config2);
    });
  });

  describe('navConfigSchema', () => {
    it('should accept valid configuration', () => {
      const validConfig = {
        header: {
          brandText: 'Test Blog',
          menuItems: [
            { label: 'Home', href: '/', isExternal: false },
            { label: 'About', href: '/about', isExternal: false },
          ],
        },
        theme: {
          enableToggle: true,
          showLabel: true,
          icons: {
            light: '☀️',
            dark: '🌙',
            default: '🖥️',
          },
        },
      };

      const result = navConfigSchema.safeParse(validConfig);
      expect(result.success).toBe(true);
    });

    it('should use defaults for missing theme', () => {
      const minimalConfig = {
        header: {
          brandText: 'Test Blog',
          menuItems: [],
        },
      };

      const result = navConfigSchema.parse(minimalConfig);

      expect(result.theme).toBeDefined();
      expect(result.theme.enableToggle).toBe(true);
    });
  });

  describe('defaultNavConfig', () => {
    it('should have all required fields', () => {
      expect(defaultNavConfig.header).toBeDefined();
      expect(defaultNavConfig.header.brandText).toBeDefined();
      expect(defaultNavConfig.header.menuItems).toBeInstanceOf(Array);
      expect(defaultNavConfig.theme).toBeDefined();
    });

    it('should pass schema validation', () => {
      const result = navConfigSchema.safeParse(defaultNavConfig);
      expect(result.success).toBe(true);
    });
  });
});
