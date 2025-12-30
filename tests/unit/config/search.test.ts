import { describe, it, expect } from 'vitest';
import {
  loadSearchConfig,
  getSearchConfig,
  defaultSearchConfig,
  searchConfigSchema,
} from '../../../src/config/loaders/search';

describe('search config loader', () => {
  describe('defaultSearchConfig', () => {
    it('has all required fields', () => {
      expect(defaultSearchConfig.enabled).toBe(true);
      expect(defaultSearchConfig.shortcut).toBe(true);
      expect(defaultSearchConfig.provider).toBe('fuse');
      expect(defaultSearchConfig.lazyLoad).toBe(true);
      expect(defaultSearchConfig.useWorker).toBe(true);
      expect(defaultSearchConfig.maxResults).toBe(12);
    });

    it('has snippet configuration', () => {
      expect(defaultSearchConfig.snippet.window).toBe(80);
      expect(defaultSearchConfig.snippet.maxLines).toBe(2);
    });

    it('has weight configuration', () => {
      expect(defaultSearchConfig.weights.title).toBe(6);
      expect(defaultSearchConfig.weights.headings).toBe(3);
      expect(defaultSearchConfig.weights.tags).toBe(3);
      expect(defaultSearchConfig.weights.summary).toBe(2);
      expect(defaultSearchConfig.weights.body).toBe(1);
    });

    it('has fuse configuration', () => {
      expect(defaultSearchConfig.fuse.threshold).toBe(0.35);
      expect(defaultSearchConfig.fuse.includeMatches).toBe(true);
      expect(defaultSearchConfig.fuse.includeScore).toBe(true);
    });

    it('has filter configuration', () => {
      expect(defaultSearchConfig.filters.tags).toBe(true);
      expect(defaultSearchConfig.filters.year).toBe(false);
      expect(defaultSearchConfig.filters.source).toBe(false);
    });

    it('has UI configuration', () => {
      expect(defaultSearchConfig.ui.placement).toBe('modal');
      expect(defaultSearchConfig.ui.showTags).toBe(true);
      expect(defaultSearchConfig.ui.showDate).toBe(true);
      expect(defaultSearchConfig.ui.placeholder).toBe('搜索文章...');
    });
  });

  describe('searchConfigSchema', () => {
    it('validates valid config', () => {
      const validConfig = {
        search: {
          enabled: true,
          shortcut: true,
          provider: 'fuse',
          lazyLoad: true,
          useWorker: true,
          maxResults: 10,
        },
      };

      const result = searchConfigSchema.safeParse(validConfig);
      expect(result.success).toBe(true);
    });

    it('applies defaults for missing fields', () => {
      const minimalConfig = {
        search: {},
      };

      const result = searchConfigSchema.safeParse(minimalConfig);
      expect(result.success).toBe(true);
      if (result.success) {
        expect(result.data.search.enabled).toBe(true);
        expect(result.data.search.provider).toBe('fuse');
      }
    });

    it('rejects invalid provider', () => {
      const invalidConfig = {
        search: {
          provider: 'invalid',
        },
      };

      const result = searchConfigSchema.safeParse(invalidConfig);
      expect(result.success).toBe(false);
    });

    it('rejects maxResults out of range', () => {
      const invalidConfig = {
        search: {
          maxResults: 500,
        },
      };

      const result = searchConfigSchema.safeParse(invalidConfig);
      expect(result.success).toBe(false);
    });
  });

  describe('loadSearchConfig', () => {
    it('loads and validates config from YAML', () => {
      const config = loadSearchConfig();

      expect(config).toBeDefined();
      expect(config.enabled).toBe(true);
      expect(config.provider).toBe('fuse');
    });
  });

  describe('getSearchConfig', () => {
    it('returns cached config', () => {
      const config1 = getSearchConfig();
      const config2 = getSearchConfig();

      expect(config1).toEqual(config2);
    });
  });
});
