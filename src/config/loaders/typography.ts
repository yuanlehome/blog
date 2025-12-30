/**
 * Typography configuration loader
 *
 * This module loads and validates the typography configuration from typography.yml.
 * It provides settings for font families, sizes, line heights, and weights.
 *
 * @module src/config/loaders/typography
 */

import { z } from 'zod';
import { loadConfig } from './base';
import typographyConfigData from '../yaml/typography.yml';

/**
 * Typography configuration schema
 */
export const typographyConfigSchema = z.object({
  fontFamily: z
    .object({
      sans: z
        .array(z.string())
        .default([
          'Inter',
          'ui-sans-serif',
          'system-ui',
          '-apple-system',
          'BlinkMacSystemFont',
          'Segoe UI',
          'Roboto',
          'Helvetica Neue',
          'Arial',
          'Noto Sans SC',
          'PingFang SC',
          'Microsoft YaHei',
          'sans-serif',
        ]),
      serif: z
        .array(z.string())
        .default(['ui-serif', 'Georgia', 'Cambria', 'Times New Roman', 'Times', 'serif']),
      mono: z
        .array(z.string())
        .default([
          'JetBrains Mono',
          'Fira Code',
          'SFMono-Regular',
          'Menlo',
          'Monaco',
          'Consolas',
          'Liberation Mono',
          'Courier New',
          'monospace',
        ]),
    })
    .default({}),
  fontSize: z
    .object({
      xs: z.string().default('0.75rem'),
      sm: z.string().default('0.875rem'),
      base: z.string().default('1rem'),
      lg: z.string().default('1.125rem'),
      xl: z.string().default('1.25rem'),
      '2xl': z.string().default('1.5rem'),
      '3xl': z.string().default('1.875rem'),
      '4xl': z.string().default('2.25rem'),
    })
    .default({}),
  lineHeight: z
    .object({
      body: z.number().min(1).max(3).default(1.75),
      heading: z.number().min(1).max(2).default(1.3),
      code: z.number().min(1).max(3).default(1.65),
      tight: z.number().min(1).max(2).default(1.25),
    })
    .default({}),
  fontWeight: z
    .object({
      normal: z.number().min(100).max(900).default(400),
      medium: z.number().min(100).max(900).default(500),
      semibold: z.number().min(100).max(900).default(600),
      bold: z.number().min(100).max(900).default(700),
    })
    .default({}),
  prose: z
    .object({
      maxWidth: z.string().default('none'),
      headingSpacing: z.number().min(0.5).max(2).default(1.0),
      paragraphSpacing: z.number().min(0.5).max(2).default(1.0),
    })
    .default({}),
});

export type TypographyConfig = z.infer<typeof typographyConfigSchema>;

/**
 * Default typography configuration
 */
export const defaultTypographyConfig: TypographyConfig = {
  fontFamily: {
    sans: [
      'Inter',
      'ui-sans-serif',
      'system-ui',
      '-apple-system',
      'BlinkMacSystemFont',
      'Segoe UI',
      'Roboto',
      'Helvetica Neue',
      'Arial',
      'Noto Sans SC',
      'PingFang SC',
      'Microsoft YaHei',
      'sans-serif',
    ],
    serif: ['ui-serif', 'Georgia', 'Cambria', 'Times New Roman', 'Times', 'serif'],
    mono: [
      'JetBrains Mono',
      'Fira Code',
      'SFMono-Regular',
      'Menlo',
      'Monaco',
      'Consolas',
      'Liberation Mono',
      'Courier New',
      'monospace',
    ],
  },
  fontSize: {
    xs: '0.75rem',
    sm: '0.875rem',
    base: '1rem',
    lg: '1.125rem',
    xl: '1.25rem',
    '2xl': '1.5rem',
    '3xl': '1.875rem',
    '4xl': '2.25rem',
  },
  lineHeight: {
    body: 1.75,
    heading: 1.3,
    code: 1.65,
    tight: 1.25,
  },
  fontWeight: {
    normal: 400,
    medium: 500,
    semibold: 600,
    bold: 700,
  },
  prose: {
    maxWidth: 'none',
    headingSpacing: 1.0,
    paragraphSpacing: 1.0,
  },
};

/**
 * Load and validate typography configuration
 *
 * @returns Validated typography configuration
 * @throws Error if configuration is invalid
 */
export function loadTypographyConfig(): TypographyConfig {
  const parsed = loadConfig(typographyConfigData, typographyConfigSchema, 'typography.yml');
  return parsed as TypographyConfig;
}

/**
 * Cached typography configuration instance
 */
let cachedConfig: TypographyConfig | null = null;

/**
 * Get typography configuration (cached)
 *
 * @returns Typography configuration
 */
export function getTypographyConfig(): TypographyConfig {
  if (!cachedConfig) {
    cachedConfig = loadTypographyConfig();
  }
  return cachedConfig;
}
