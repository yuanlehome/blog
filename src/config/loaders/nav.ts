/**
 * Navigation configuration loader
 *
 * This module loads and validates the navigation configuration from nav.yml.
 * It provides settings for header navigation menu items and theme toggle.
 *
 * @module src/config/loaders/nav
 */

import { z } from 'zod';
import { loadConfig } from './base';
import navConfigData from '../yaml/nav.yml';

/**
 * Navigation menu item schema
 */
export const navMenuItemSchema = z.object({
  label: z.string().min(1),
  href: z.string().min(1),
  isExternal: z.boolean().default(false),
  openInNewTab: z.boolean().optional(),
});

export type NavMenuItem = z.infer<typeof navMenuItemSchema>;

/**
 * Theme configuration schema
 */
export const navThemeSchema = z.object({
  enableToggle: z.boolean().default(true),
  showLabel: z.boolean().default(true),
  icons: z
    .object({
      light: z.string().default('☀️'),
      dark: z.string().default('🌙'),
      default: z.string().default('🖥️'),
    })
    .default({}),
});

export type NavTheme = z.infer<typeof navThemeSchema>;

/**
 * Navigation configuration schema
 */
export const navConfigSchema = z.object({
  header: z.object({
    brandText: z.string().min(1).default("Yuanle Liu's Blog"),
    menuItems: z.array(navMenuItemSchema).default([]),
  }),
  theme: navThemeSchema.default({}),
});

export type NavConfig = z.infer<typeof navConfigSchema>;

/**
 * Default navigation configuration
 */
export const defaultNavConfig: NavConfig = {
  header: {
    brandText: "Yuanle Liu's Blog",
    menuItems: [
      { label: 'Home', href: '/', isExternal: false },
      { label: 'Archive', href: '/archive/', isExternal: false },
      { label: 'About', href: '/about/', isExternal: false },
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

/**
 * Load and validate navigation configuration
 *
 * @returns Validated navigation configuration
 * @throws Error if configuration is invalid
 */
export function loadNavConfig(): NavConfig {
  const parsed = loadConfig(navConfigData, navConfigSchema, 'nav.yml');
  return parsed as NavConfig;
}

/**
 * Cached navigation configuration instance
 */
let cachedConfig: NavConfig | null = null;

/**
 * Get navigation configuration (cached)
 *
 * @returns Navigation configuration
 */
export function getNavConfig(): NavConfig {
  if (!cachedConfig) {
    cachedConfig = loadNavConfig();
  }
  return cachedConfig;
}
