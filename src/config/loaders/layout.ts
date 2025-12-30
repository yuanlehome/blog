/**
 * Layout configuration loader
 *
 * This module loads and validates the layout configuration from layout.yml.
 * It provides settings for page layout, container widths, and sidebar positioning.
 *
 * @module src/config/loaders/layout
 */

import { z } from 'zod';
import { loadConfig } from './base';
import layoutConfigData from '../yaml/layout.yml';

/**
 * CSS unit validation (rem, px, %, vh, vw)
 */
const cssUnitSchema = z
  .string()
  .refine((val) => /^\d+(\.\d+)?(rem|px|%|vh|vw|em|ch)$/.test(val), {
    message: 'Invalid CSS unit. Use rem, px, %, vh, vw, em, or ch with a numeric value.',
  });

/**
 * Layout configuration schema
 */
export const layoutConfigSchema = z.object({
  container: z
    .object({
      width: cssUnitSchema.default('72rem'),
      paddingX: z
        .object({
          mobile: cssUnitSchema.default('1rem'),
          tablet: cssUnitSchema.default('1.5rem'),
          desktop: cssUnitSchema.default('2rem'),
        })
        .default({}),
    })
    .default({}),
  layoutMode: z.enum(['centered', 'rightSidebar', 'leftSidebar']).default('rightSidebar'),
  sidebar: z
    .object({
      enabled: z.boolean().default(true),
      position: z.enum(['left', 'right']).default('right'),
      width: cssUnitSchema.default('18rem'),
      sticky: z.boolean().default(true),
      gap: cssUnitSchema.default('3rem'),
    })
    .default({}),
  toc: z
    .object({
      enabled: z.boolean().default(true),
      position: z.enum(['sidebar', 'inline', 'hidden']).default('sidebar'),
      mobileBehavior: z.enum(['drawer', 'inline', 'hidden']).default('drawer'),
      defaultOpen: z.boolean().default(false),
      stickyOffset: z.number().min(0).max(200).default(96),
    })
    .default({}),
  alignment: z
    .object({
      headerAlign: z.enum(['left', 'center']).default('left'),
      footerAlign: z.enum(['left', 'center']).default('center'),
      postMetaAlign: z.enum(['left', 'center']).default('left'),
      contentAlign: z.enum(['left', 'center']).default('left'),
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
      tablet: '1.5rem',
      desktop: '2rem',
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
    position: 'sidebar',
    mobileBehavior: 'drawer',
    defaultOpen: false,
    stickyOffset: 96,
  },
  alignment: {
    headerAlign: 'left',
    footerAlign: 'center',
    postMetaAlign: 'left',
    contentAlign: 'left',
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
