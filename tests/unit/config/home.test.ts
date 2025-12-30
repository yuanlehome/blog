/**
 * Tests for home page configuration loader
 */

import { describe, it, expect } from 'vitest';
import {
  loadHomeConfig,
  getHomeConfig,
  homeConfigSchema,
  defaultHomeConfig,
} from '../../../src/config/loaders/home';

describe('Home Page Configuration', () => {
  describe('loadHomeConfig', () => {
    it('should load and validate home configuration', () => {
      const config = loadHomeConfig();

      expect(config).toBeDefined();
      expect(config.title).toBeDefined();
      expect(config.pagination).toBeDefined();
      expect(config.pagination.pageSize).toBeGreaterThan(0);
      expect(config.pagination.windowSize).toBeGreaterThan(0);
    });
  });

  describe('getHomeConfig (cached)', () => {
    it('should return cached configuration', () => {
      const config1 = getHomeConfig();
      const config2 = getHomeConfig();

      expect(config1).toBe(config2);
    });
  });

  describe('homeConfigSchema', () => {
    it('should accept valid configuration', () => {
      const validConfig = {
        title: 'Recent Posts',
        showPostCount: true,
        postCountText: 'posts',
        pagination: {
          pageSize: 10,
          windowSize: 5,
        },
        navigation: {
          newerText: '← Newer',
          olderText: 'Older →',
          pageLabel: 'Page',
        },
      };

      const result = homeConfigSchema.safeParse(validConfig);
      expect(result.success).toBe(true);
    });

    it('should reject invalid page size', () => {
      const invalidConfig = {
        pagination: {
          pageSize: 0,
          windowSize: 5,
        },
      };

      const result = homeConfigSchema.safeParse(invalidConfig);
      expect(result.success).toBe(false);
    });
  });

  describe('defaultHomeConfig', () => {
    it('should have sensible defaults', () => {
      expect(defaultHomeConfig.pagination.pageSize).toBeGreaterThan(0);
      expect(defaultHomeConfig.pagination.windowSize).toBeGreaterThan(0);
    });

    it('should pass schema validation', () => {
      const result = homeConfigSchema.safeParse(defaultHomeConfig);
      expect(result.success).toBe(true);
    });
  });
});
