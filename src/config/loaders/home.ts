/**
 * Home page configuration loader
 *
 * This module loads and validates the home page configuration from home.yml.
 * It provides settings for the blog homepage including pagination and section titles.
 *
 * @module src/config/loaders/home
 */

import { z } from 'zod';
import { loadConfig } from './base';
import homeConfigData from '../yaml/home.yml';

/**
 * Home page configuration schema
 */
export const homeConfigSchema = z.object({
  title: z.string().default('Recent Posts'),
  showPostCount: z.boolean().default(true),
  postCountText: z.string().default('published posts'),

  pagination: z
    .object({
      pageSize: z.number().min(1).default(5),
      windowSize: z.number().min(1).default(5),
    })
    .default({}),

  navigation: z
    .object({
      newerText: z.string().default('← Newer'),
      olderText: z.string().default('Older →'),
      pageLabel: z.string().default('Page'),
    })
    .default({}),
});

export type HomeConfig = z.infer<typeof homeConfigSchema>;

/**
 * Default home page configuration
 */
export const defaultHomeConfig: HomeConfig = {
  title: 'Recent Posts',
  showPostCount: true,
  postCountText: 'published posts',
  pagination: {
    pageSize: 5,
    windowSize: 5,
  },
  navigation: {
    newerText: '← Newer',
    olderText: 'Older →',
    pageLabel: 'Page',
  },
};

/**
 * Load and validate home page configuration
 *
 * @returns Validated home page configuration
 * @throws Error if configuration is invalid
 */
export function loadHomeConfig(): HomeConfig {
  return loadConfig(homeConfigData, homeConfigSchema, 'home.yml');
}

/**
 * Cached home configuration instance
 */
let cachedConfig: HomeConfig | null = null;

/**
 * Get home page configuration (cached)
 *
 * @returns Home page configuration
 */
export function getHomeConfig(): HomeConfig {
  if (!cachedConfig) {
    cachedConfig = loadHomeConfig();
  }
  return cachedConfig;
}
