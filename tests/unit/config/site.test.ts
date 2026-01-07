/**
 * Tests for site configuration loader
 */

import { describe, it, expect } from 'vitest';
import {
  loadSiteConfig,
  getSiteConfig,
  siteConfigSchema,
  defaultSiteConfig,
  type SiteConfig,
} from '../../../src/config/loaders/site';

describe('Site Configuration', () => {
  describe('loadSiteConfig', () => {
    it('should load and validate site configuration', () => {
      const config = loadSiteConfig();

      expect(config).toBeDefined();
      expect(config.siteName).toBeDefined();
      expect(config.title).toBeDefined();
      expect(config.description).toBeDefined();
      expect(config.author).toBeDefined();
    });

    it('should have correct types', () => {
      const config = loadSiteConfig();

      expect(typeof config.siteName).toBe('string');
      expect(typeof config.title).toBe('string');
      expect(typeof config.description).toBe('string');
      expect(typeof config.author).toBe('string');
      expect(typeof config.copyrightYear).toBe('number');
      expect(typeof config.copyrightText).toBe('string');
      expect(typeof config.enableRSS).toBe('boolean');
      expect(typeof config.enableSitemap).toBe('boolean');
    });
  });

  describe('getSiteConfig (cached)', () => {
    it('should return cached configuration', () => {
      const config1 = getSiteConfig();
      const config2 = getSiteConfig();

      expect(config1).toBe(config2);
    });
  });

  describe('siteConfigSchema', () => {
    it('should accept valid configuration', () => {
      const validConfig: SiteConfig = {
        siteName: 'Test Blog',
        title: 'Test Blog',
        description: 'A test blog',
        author: 'Test Author',
        copyrightYear: 2024,
        copyrightText: 'All rights reserved.',
        defaultLanguage: 'en',
        dateFormat: 'YYYY-MM-DD',
        enableRSS: true,
        enableSitemap: true,
        busuanzi: {
          enabled: true,
          scriptUrl: 'https://busuanzi.ibruce.info/busuanzi/2.3/busuanzi.pure.mini.js',
          debug: false,
        },
        socialImage: 'test.jpg',
      };

      const result = siteConfigSchema.safeParse(validConfig);
      expect(result.success).toBe(true);
    });

    it('should use defaults for missing optional fields', () => {
      const minimalConfig = {
        siteName: 'Test Blog',
        title: 'Test Blog',
        description: 'A test blog',
        author: 'Test Author',
      };

      const result = siteConfigSchema.parse(minimalConfig);

      expect(result.copyrightYear).toBeDefined();
      expect(result.copyrightText).toBe('All rights reserved.');
      expect(result.enableRSS).toBe(true);
      expect(result.enableSitemap).toBe(true);
    });

    it('should reject invalid configuration', () => {
      const invalidConfig = {
        siteName: '',
      };

      const result = siteConfigSchema.safeParse(invalidConfig);
      expect(result.success).toBe(false);
    });
  });

  describe('defaultSiteConfig', () => {
    it('should have all required fields', () => {
      expect(defaultSiteConfig.siteName).toBeDefined();
      expect(defaultSiteConfig.title).toBeDefined();
      expect(defaultSiteConfig.description).toBeDefined();
      expect(defaultSiteConfig.author).toBeDefined();
      expect(defaultSiteConfig.copyrightYear).toBeDefined();
      expect(defaultSiteConfig.copyrightText).toBeDefined();
      expect(defaultSiteConfig.defaultLanguage).toBeDefined();
      expect(defaultSiteConfig.dateFormat).toBeDefined();
      expect(defaultSiteConfig.enableRSS).toBeDefined();
      expect(defaultSiteConfig.enableSitemap).toBeDefined();
      expect(defaultSiteConfig.socialImage).toBeDefined();
    });

    it('should pass schema validation', () => {
      const result = siteConfigSchema.safeParse(defaultSiteConfig);
      expect(result.success).toBe(true);
    });
  });
});
