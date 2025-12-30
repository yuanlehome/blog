/**
 * Components configuration loader
 *
 * This module loads and validates the components configuration from components.yml.
 * It provides settings for visual styling (radius, shadows, borders, spacing).
 *
 * @module src/config/loaders/components
 */

import { z } from 'zod';
import { loadConfig } from './base';
import componentsConfigData from '../yaml/components.yml';

/**
 * CSS unit validation for radius and spacing
 */
const cssUnitSchema = z
  .string()
  .refine(
    (val) =>
      /^\d+(\.\d+)?(rem|px|em)$/.test(val) || val === '0' || val === 'none' || val === '9999px',
    { message: 'Invalid CSS unit. Use rem, px, em, or special values like "0", "none", "9999px".' },
  );

/**
 * Box shadow validation
 */
const shadowSchema = z.string();

/**
 * Components configuration schema
 */
export const componentsConfigSchema = z.object({
  radius: z
    .object({
      none: z.string().default('0'),
      sm: cssUnitSchema.default('0.375rem'),
      default: cssUnitSchema.default('0.5rem'),
      md: cssUnitSchema.default('0.75rem'),
      lg: cssUnitSchema.default('0.9rem'),
      xl: cssUnitSchema.default('0.75rem'),
      full: z.string().default('9999px'),
    })
    .default({}),
  componentRadius: z
    .object({
      card: cssUnitSchema.default('0.75rem'),
      button: cssUnitSchema.default('0.5rem'),
      image: cssUnitSchema.default('0.75rem'),
      code: cssUnitSchema.default('0.9rem'),
      input: cssUnitSchema.default('0.5rem'),
    })
    .default({}),
  shadow: z
    .object({
      none: z.string().default('none'),
      sm: shadowSchema.default('0 1px 2px 0 rgb(0 0 0 / 0.05)'),
      default: shadowSchema.default(
        '0 1px 3px 0 rgb(0 0 0 / 0.1), 0 1px 2px -1px rgb(0 0 0 / 0.1)',
      ),
      md: shadowSchema.default('0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1)'),
      lg: shadowSchema.default(
        '0 10px 15px -3px rgb(0 0 0 / 0.1), 0 4px 6px -4px rgb(0 0 0 / 0.1)',
      ),
      xl: shadowSchema.default(
        '0 20px 25px -5px rgb(0 0 0 / 0.1), 0 8px 10px -6px rgb(0 0 0 / 0.1)',
      ),
      '2xl': shadowSchema.default('0 25px 50px -12px rgb(0 0 0 / 0.25)'),
    })
    .default({}),
  componentShadow: z
    .object({
      card: shadowSchema.default('0 8px 24px rgb(15 23 42 / 0.08)'),
      cardDark: shadowSchema.default('0 10px 30px rgb(0 0 0 / 0.28)'),
      header: shadowSchema.default('0 8px 24px rgb(15 23 42 / 0.08)'),
      headerDark: shadowSchema.default('0 10px 32px rgb(0 0 0 / 0.28)'),
      hoverLift: z.boolean().default(true),
    })
    .default({}),
  border: z
    .object({
      style: z.enum(['solid', 'dashed', 'dotted']).default('solid'),
      width: z.string().default('1px'),
      opacity: z.number().min(0).max(1).default(0.2),
    })
    .default({}),
  motion: z
    .object({
      enabled: z.boolean().default(true),
      level: z.enum(['subtle', 'normal', 'energetic']).default('normal'),
      duration: z
        .object({
          fast: z.number().min(50).max(500).default(150),
          normal: z.number().min(50).max(500).default(200),
          slow: z.number().min(50).max(1000).default(300),
        })
        .default({}),
      easing: z
        .object({
          default: z.string().default('ease'),
          in: z.string().default('ease-in'),
          out: z.string().default('ease-out'),
          inOut: z.string().default('ease-in-out'),
        })
        .default({}),
    })
    .default({}),
  spacingScale: z.enum(['compact', 'comfortable', 'relaxed']).default('comfortable'),
  spacingMultiplier: z
    .object({
      compact: z.number().min(0.5).max(1.5).default(0.875),
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
    none: '0',
    sm: '0.375rem',
    default: '0.5rem',
    md: '0.75rem',
    lg: '0.9rem',
    xl: '0.75rem',
    full: '9999px',
  },
  componentRadius: {
    card: '0.75rem',
    button: '0.5rem',
    image: '0.75rem',
    code: '0.9rem',
    input: '0.5rem',
  },
  shadow: {
    none: 'none',
    sm: '0 1px 2px 0 rgb(0 0 0 / 0.05)',
    default: '0 1px 3px 0 rgb(0 0 0 / 0.1), 0 1px 2px -1px rgb(0 0 0 / 0.1)',
    md: '0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1)',
    lg: '0 10px 15px -3px rgb(0 0 0 / 0.1), 0 4px 6px -4px rgb(0 0 0 / 0.1)',
    xl: '0 20px 25px -5px rgb(0 0 0 / 0.1), 0 8px 10px -6px rgb(0 0 0 / 0.1)',
    '2xl': '0 25px 50px -12px rgb(0 0 0 / 0.25)',
  },
  componentShadow: {
    card: '0 8px 24px rgb(15 23 42 / 0.08)',
    cardDark: '0 10px 30px rgb(0 0 0 / 0.28)',
    header: '0 8px 24px rgb(15 23 42 / 0.08)',
    headerDark: '0 10px 32px rgb(0 0 0 / 0.28)',
    hoverLift: true,
  },
  border: {
    style: 'solid',
    width: '1px',
    opacity: 0.2,
  },
  motion: {
    enabled: true,
    level: 'normal',
    duration: {
      fast: 150,
      normal: 200,
      slow: 300,
    },
    easing: {
      default: 'ease',
      in: 'ease-in',
      out: 'ease-out',
      inOut: 'ease-in-out',
    },
  },
  spacingScale: 'comfortable',
  spacingMultiplier: {
    compact: 0.875,
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
