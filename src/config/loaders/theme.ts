/**
 * Theme configuration loader
 *
 * This module loads and validates the theme configuration from theme.yml.
 * It provides settings for visual themes and appearance preferences.
 *
 * @module src/config/loaders/theme
 */

import { z } from 'zod';
import { loadConfig } from './base';
import themeConfigData from '../yaml/theme.yml';

/**
 * Theme configuration schema
 */
export const themeConfigSchema = z.object({
  defaultTheme: z.enum(['light', 'dark', 'system']).default('system'),
  themes: z.array(z.string()).default(['light', 'dark']),
  storageKey: z.string().default('theme'),
  icons: z
    .object({
      light: z.string().default('☀️'),
      dark: z.string().default('🌙'),
    })
    .default({}),
  labels: z
    .object({
      light: z.string().default('Light'),
      dark: z.string().default('Dark'),
    })
    .default({}),
  animations: z
    .object({
      respectReducedMotion: z.boolean().default(true),
      enableScrollEffects: z.boolean().default(true),
    })
    .default({}),
});

export type ThemeConfig = z.infer<typeof themeConfigSchema>;

/**
 * Default theme configuration
 */
export const defaultThemeConfig: ThemeConfig = {
  defaultTheme: 'system',
  themes: ['light', 'dark'],
  storageKey: 'theme',
  icons: {
    light: '☀️',
    dark: '🌙',
  },
  labels: {
    light: 'Light',
    dark: 'Dark',
  },
  animations: {
    respectReducedMotion: true,
    enableScrollEffects: true,
  },
};

/**
 * Load and validate theme configuration
 *
 * @returns Validated theme configuration
 * @throws Error if configuration is invalid
 */
export function loadThemeConfig(): ThemeConfig {
  const parsed = loadConfig(themeConfigData, themeConfigSchema, 'theme.yml');
  return parsed as ThemeConfig;
}

/**
 * Cached theme configuration instance
 */
let cachedConfig: ThemeConfig | null = null;

/**
 * Get theme configuration (cached)
 *
 * @returns Theme configuration
 */
export function getThemeConfig(): ThemeConfig {
  if (!cachedConfig) {
    cachedConfig = loadThemeConfig();
  }
  return cachedConfig;
}
