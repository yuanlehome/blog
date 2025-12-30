/**
 * Tests for profile configuration loader
 */

import { describe, it, expect } from 'vitest';
import {
  loadProfileConfig,
  getProfileConfig,
  profileConfigSchema,
  defaultProfileConfig,
} from '../../../src/config/loaders/profile';

describe('Profile Configuration', () => {
  describe('loadProfileConfig', () => {
    it('should load and validate profile configuration', () => {
      const config = loadProfileConfig();
      
      expect(config).toBeDefined();
      expect(config.name).toBeDefined();
      expect(config.bio).toBeDefined();
      expect(config.socialLinks).toBeInstanceOf(Array);
      expect(config.whatIDo).toBeDefined();
      expect(config.techStack).toBeDefined();
      expect(config.journey).toBeDefined();
    });

    it('should have social links with valid URLs', () => {
      const config = loadProfileConfig();
      
      config.socialLinks.forEach(link => {
        expect(link.name).toBeDefined();
        expect(link.url).toBeDefined();
        expect(link.url).toMatch(/^https?:\/\//);
      });
    });
  });

  describe('getProfileConfig (cached)', () => {
    it('should return cached configuration', () => {
      const config1 = getProfileConfig();
      const config2 = getProfileConfig();
      
      expect(config1).toBe(config2);
    });
  });

  describe('profileConfigSchema', () => {
    it('should accept valid configuration', () => {
      const validConfig = {
        name: "Test User",
        bio: "A test bio",
        socialLinks: [
          { name: "GitHub", url: "https://github.com/test" },
        ],
        whatIDo: {
          title: "What I Do",
          description: "Building software",
        },
        techStack: {
          title: "Tech Stack",
          skills: ["TypeScript", "Node.js"],
        },
        journey: {
          title: "My Journey",
          items: [
            {
              year: "2020",
              role: "Developer",
              description: "Built things",
            },
          ],
        },
      };

      const result = profileConfigSchema.safeParse(validConfig);
      expect(result.success).toBe(true);
    });

    it('should reject invalid URLs', () => {
      const invalidConfig = {
        socialLinks: [
          { name: "Test", url: "not-a-url" },
        ],
      };

      const result = profileConfigSchema.safeParse(invalidConfig);
      expect(result.success).toBe(false);
    });
  });

  describe('defaultProfileConfig', () => {
    it('should have all required fields', () => {
      expect(defaultProfileConfig.name).toBeDefined();
      expect(defaultProfileConfig.bio).toBeDefined();
      expect(defaultProfileConfig.socialLinks).toBeInstanceOf(Array);
      expect(defaultProfileConfig.whatIDo).toBeDefined();
      expect(defaultProfileConfig.techStack).toBeDefined();
      expect(defaultProfileConfig.journey).toBeDefined();
    });

    it('should pass schema validation', () => {
      const result = profileConfigSchema.safeParse(defaultProfileConfig);
      expect(result.success).toBe(true);
    });
  });
});
