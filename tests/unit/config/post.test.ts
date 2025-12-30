/**
 * Tests for post page configuration loader
 */

import { describe, it, expect } from 'vitest';
import {
  loadPostConfig,
  getPostConfig,
  postConfigSchema,
  defaultPostConfig,
} from '../../../src/config/loaders/post';

describe('Post Page Configuration', () => {
  describe('loadPostConfig', () => {
    it('should load and validate post configuration', () => {
      const config = loadPostConfig();
      
      expect(config).toBeDefined();
      expect(config.metadata).toBeDefined();
      expect(config.tableOfContents).toBeDefined();
      expect(config.comments).toBeDefined();
      expect(config.floatingActions).toBeDefined();
      expect(config.readingProgress).toBeDefined();
      expect(config.prevNext).toBeDefined();
      expect(config.relatedPosts).toBeDefined();
    });

    it('should have giscus configuration', () => {
      const config = loadPostConfig();
      
      expect(config.comments.giscus).toBeDefined();
      expect(config.comments.giscus.repo).toBeDefined();
      expect(config.comments.giscus.repoId).toBeDefined();
    });
  });

  describe('getPostConfig (cached)', () => {
    it('should return cached configuration', () => {
      const config1 = getPostConfig();
      const config2 = getPostConfig();
      
      expect(config1).toBe(config2);
    });
  });

  describe('postConfigSchema', () => {
    it('should accept valid configuration', () => {
      const validConfig = {
        metadata: {
          showPublishedDate: true,
          showUpdatedDate: true,
          showReadingTime: true,
          showWordCount: true,
          publishedLabel: "Published",
          updatedLabel: "Updated",
          icons: {
            published: "📅",
            updated: "🔄",
            wordCount: "✍️",
            readingTime: "⏱️",
          },
        },
        tableOfContents: {
          enable: true,
          defaultExpanded: false,
          showOnMobile: true,
          mobileTrigger: false,
        },
        floatingActions: {
          enableToc: true,
          enableTop: true,
          enableBottom: true,
        },
        readingProgress: {
          enable: true,
        },
        prevNext: {
          enable: true,
        },
        relatedPosts: {
          enable: true,
          maxCount: 3,
        },
        comments: {
          enable: true,
          defaultEnabled: true,
          provider: "giscus",
          giscus: {
            repo: "owner/repo",
            repoId: "id",
            category: "General",
            categoryId: "id",
            mapping: "pathname",
            strict: "0",
            reactionsEnabled: "1",
            emitMetadata: "0",
            inputPosition: "bottom",
            theme: "preferred_color_scheme",
            lang: "en",
          },
          themeMapping: {
            light: "light",
            dark: "dark",
          },
        },
        sourceAttribution: {
          enable: true,
          prefix: "Source:",
          authorPrefix: "Author",
          linkText: "View original",
        },
      };

      const result = postConfigSchema.safeParse(validConfig);
      expect(result.success).toBe(true);
    });
  });

  describe('defaultPostConfig', () => {
    it('should have all required sections', () => {
      expect(defaultPostConfig.metadata).toBeDefined();
      expect(defaultPostConfig.tableOfContents).toBeDefined();
      expect(defaultPostConfig.comments).toBeDefined();
      expect(defaultPostConfig.floatingActions).toBeDefined();
      expect(defaultPostConfig.readingProgress).toBeDefined();
      expect(defaultPostConfig.prevNext).toBeDefined();
      expect(defaultPostConfig.relatedPosts).toBeDefined();
      expect(defaultPostConfig.sourceAttribution).toBeDefined();
    });

    it('should pass schema validation', () => {
      const result = postConfigSchema.safeParse(defaultPostConfig);
      expect(result.success).toBe(true);
    });
  });
});
