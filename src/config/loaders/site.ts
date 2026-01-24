/**
 * Site configuration loader
 *
 * This module loads and validates the site-wide configuration from site.yml.
 * It provides global settings like site name, title, description, and SEO options.
 *
 * @module src/config/loaders/site
 */

import { z } from 'zod';
import { loadConfig } from './base';
import siteConfigData from '../yaml/site.yml';

/**
 * Site configuration schema
 */
export const siteConfigSchema = z.object({
  siteName: z.string().min(1).default("Yuanle Liu's Blog"),
  title: z.string().min(1).default("Yuanle Liu's Blog"),
  description: z.string().default('A minimal Astro blog'),
  author: z.string().default('Yuanle Liu'),

  copyrightYear: z.number().default(new Date().getFullYear()),
  copyrightText: z.string().default('All rights reserved.'),

  defaultLanguage: z.string().default('en'),
  dateFormat: z.string().default('YYYY-MM-DD'),

  enableSitemap: z.boolean().default(true),

  socialImage: z.string().default('placeholder-social.jpg'),
});

export type SiteConfig = z.infer<typeof siteConfigSchema>;

/**
 * Default site configuration
 */
export const defaultSiteConfig: SiteConfig = {
  siteName: "Yuanle Liu's Blog",
  title: "Yuanle Liu's Blog",
  description: 'A minimal Astro blog',
  author: 'Yuanle Liu',
  copyrightYear: new Date().getFullYear(),
  copyrightText: 'All rights reserved.',
  defaultLanguage: 'en',
  dateFormat: 'YYYY-MM-DD',
  enableSitemap: true,
  socialImage: 'placeholder-social.jpg',
};

/**
 * Load and validate site configuration
 *
 * @returns Validated site configuration
 * @throws Error if configuration is invalid
 */
export function loadSiteConfig(): SiteConfig {
  const parsed = loadConfig(siteConfigData, siteConfigSchema, 'site.yml');
  return parsed as SiteConfig;
}

/**
 * Cached site configuration instance
 */
let cachedConfig: SiteConfig | null = null;

/**
 * Get site configuration (cached)
 *
 * @returns Site configuration
 */
export function getSiteConfig(): SiteConfig {
  if (!cachedConfig) {
    cachedConfig = loadSiteConfig();
  }
  return cachedConfig;
}
