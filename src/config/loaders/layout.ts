/**
 * Layout configuration loader
 *
 * This module loads and validates the layout configuration from layout.yml.
 * It provides settings for container width, sidebar, layout modes, and alignment.
 *
 * @module src/config/loaders/layout
 */

import { z } from 'zod';
import { loadConfig } from './base';
import layoutConfigData from '../yaml/layout.yml';

/**
 * Layout configuration schema
 */
export const layoutConfigSchema = z.object({
  container: z
    .object({
      width: z.string().default('72rem'),
      paddingX: z
        .object({
          mobile: z.string().default('1rem'),
          tablet: z.string().default('1rem'),
          desktop: z.string().default('1rem'),
        })
        .default({}),
    })
    .default({}),
  layoutMode: z.enum(['centered', 'rightSidebar', 'leftSidebar']).default('rightSidebar'),
  sidebar: z
    .object({
      enabled: z.boolean().default(true),
      position: z.enum(['left', 'right']).default('right'),
      width: z.string().default('18rem'),
      sticky: z.boolean().default(true),
      gap: z.string().default('3rem'),
    })
    .default({}),
  toc: z
    .object({
      enabled: z.boolean().default(true),
      position: z.enum(['left', 'right', 'inline']).default('right'),
      mobileBehavior: z.enum(['drawer', 'inline', 'hidden']).default('drawer'),
      defaultOpen: z.boolean().default(false),
      offset: z.number().min(0).max(200).default(96),
    })
    .default({}),
  alignment: z
    .object({
      headerAlign: z.enum(['left', 'center']).default('left'),
      footerAlign: z.enum(['left', 'center']).default('left'),
      postMetaAlign: z.enum(['left', 'center']).default('left'),
    })
    .default({}),
});

export type LayoutConfig = z.infer<typeof layoutConfigSchema>;

/**
 * Default layout configuration
 */
export const defaultLayoutConfig: LayoutConfig = {
  container: {
    width: '72rem',
    paddingX: {
      mobile: '1rem',
      tablet: '1rem',
      desktop: '1rem',
    },
  },
  layoutMode: 'rightSidebar',
  sidebar: {
    enabled: true,
    position: 'right',
    width: '18rem',
    sticky: true,
    gap: '3rem',
  },
  toc: {
    enabled: true,
    position: 'right',
    mobileBehavior: 'drawer',
    defaultOpen: false,
    offset: 96,
  },
  alignment: {
    headerAlign: 'left',
    footerAlign: 'left',
    postMetaAlign: 'left',
  },
};

/**
 * Load and validate layout configuration
 *
 * @returns Validated layout configuration
 * @throws Error if configuration is invalid
 */
export function loadLayoutConfig(): LayoutConfig {
  const parsed = loadConfig(layoutConfigData, layoutConfigSchema, 'layout.yml');
  return parsed as LayoutConfig;
}

/**
 * Cached layout configuration instance
 */
let cachedConfig: LayoutConfig | null = null;

/**
 * Get layout configuration (cached)
 *
 * @returns Layout configuration
 */
export function getLayoutConfig(): LayoutConfig {
  if (!cachedConfig) {
    cachedConfig = loadLayoutConfig();
  }
  return cachedConfig;
}
