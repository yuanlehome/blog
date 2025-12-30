/**
 * Components configuration loader
 *
 * This module loads and validates the components configuration from components.yml.
 * It provides settings for border radius, shadows, borders, motion, and spacing.
 *
 * @module src/config/loaders/components
 */

import { z } from 'zod';
import { loadConfig } from './base';
import componentsConfigData from '../yaml/components.yml';

/**
 * Components configuration schema
 */
export const componentsConfigSchema = z.object({
  radius: z
    .object({
      sm: z.string().default('0.35rem'),
      md: z.string().default('0.65rem'),
      lg: z.string().default('0.9rem'),
      xl: z.string().default('0.75rem'),
    })
    .default({}),
  shadow: z
    .object({
      card: z.enum(['none', 'sm', 'md', 'lg']).default('md'),
      codeBlock: z.enum(['none', 'sm', 'md', 'lg']).default('md'),
      header: z.enum(['none', 'sm', 'md', 'lg']).default('md'),
      hoverLift: z.boolean().default(false),
    })
    .default({}),
  shadowValues: z
    .object({
      none: z.string().default('none'),
      sm: z.string().default('0 1px 2px 0 rgb(0 0 0 / 0.05)'),
      md: z.string().default('0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1)'),
      lg: z.string().default('0 10px 15px -3px rgb(0 0 0 / 0.1), 0 4px 6px -4px rgb(0 0 0 / 0.1)'),
      xl: z.string().default('0 20px 25px -5px rgb(0 0 0 / 0.1), 0 8px 10px -6px rgb(0 0 0 / 0.1)'),
    })
    .default({}),
  shadowValuesDark: z
    .object({
      none: z.string().default('none'),
      sm: z.string().default('0 1px 2px 0 rgb(0 0 0 / 0.3)'),
      md: z.string().default('0 10px 30px rgb(0 0 0 / 0.28)'),
      lg: z.string().default('0 20px 40px rgb(0 0 0 / 0.35)'),
      xl: z.string().default('0 25px 50px rgb(0 0 0 / 0.4)'),
    })
    .default({}),
  border: z
    .object({
      style: z.enum(['solid', 'dashed', 'dotted']).default('solid'),
      width: z.string().default('1px'),
    })
    .default({}),
  motion: z
    .object({
      enabled: z.boolean().default(true),
      level: z.enum(['subtle', 'normal', 'energetic']).default('normal'),
      respectReducedMotion: z.boolean().default(true),
    })
    .default({}),
  motionTiming: z
    .object({
      subtle: z
        .object({
          duration: z.string().default('100ms'),
          easing: z.string().default('ease'),
        })
        .default({}),
      normal: z
        .object({
          duration: z.string().default('160ms'),
          easing: z.string().default('ease'),
        })
        .default({}),
      energetic: z
        .object({
          duration: z.string().default('240ms'),
          easing: z.string().default('ease-in-out'),
        })
        .default({}),
    })
    .default({}),
  spacingScale: z.enum(['compact', 'comfortable', 'relaxed']).default('comfortable'),
  spacingMultiplier: z
    .object({
      compact: z.number().min(0.5).max(1.5).default(0.75),
      comfortable: z.number().min(0.5).max(1.5).default(1.0),
      relaxed: z.number().min(0.5).max(2).default(1.25),
    })
    .default({}),
});

export type ComponentsConfig = z.infer<typeof componentsConfigSchema>;

/**
 * Default components configuration
 */
export const defaultComponentsConfig: ComponentsConfig = {
  radius: {
    sm: '0.35rem',
    md: '0.65rem',
    lg: '0.9rem',
    xl: '0.75rem',
  },
  shadow: {
    card: 'md',
    codeBlock: 'md',
    header: 'md',
    hoverLift: false,
  },
  shadowValues: {
    none: 'none',
    sm: '0 1px 2px 0 rgb(0 0 0 / 0.05)',
    md: '0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1)',
    lg: '0 10px 15px -3px rgb(0 0 0 / 0.1), 0 4px 6px -4px rgb(0 0 0 / 0.1)',
    xl: '0 20px 25px -5px rgb(0 0 0 / 0.1), 0 8px 10px -6px rgb(0 0 0 / 0.1)',
  },
  shadowValuesDark: {
    none: 'none',
    sm: '0 1px 2px 0 rgb(0 0 0 / 0.3)',
    md: '0 10px 30px rgb(0 0 0 / 0.28)',
    lg: '0 20px 40px rgb(0 0 0 / 0.35)',
    xl: '0 25px 50px rgb(0 0 0 / 0.4)',
  },
  border: {
    style: 'solid',
    width: '1px',
  },
  motion: {
    enabled: true,
    level: 'normal',
    respectReducedMotion: true,
  },
  motionTiming: {
    subtle: {
      duration: '100ms',
      easing: 'ease',
    },
    normal: {
      duration: '160ms',
      easing: 'ease',
    },
    energetic: {
      duration: '240ms',
      easing: 'ease-in-out',
    },
  },
  spacingScale: 'comfortable',
  spacingMultiplier: {
    compact: 0.75,
    comfortable: 1.0,
    relaxed: 1.25,
  },
};

/**
 * Load and validate components configuration
 *
 * @returns Validated components configuration
 * @throws Error if configuration is invalid
 */
export function loadComponentsConfig(): ComponentsConfig {
  const parsed = loadConfig(componentsConfigData, componentsConfigSchema, 'components.yml');
  return parsed as ComponentsConfig;
}

/**
 * Cached components configuration instance
 */
let cachedConfig: ComponentsConfig | null = null;

/**
 * Get components configuration (cached)
 *
 * @returns Components configuration
 */
export function getComponentsConfig(): ComponentsConfig {
  if (!cachedConfig) {
    cachedConfig = loadComponentsConfig();
  }
  return cachedConfig;
}
