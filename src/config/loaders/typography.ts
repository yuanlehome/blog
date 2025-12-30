/**
 * Typography configuration loader
 *
 * This module loads and validates the typography configuration from typography.yml.
 * It provides settings for fonts, sizes, and text rendering.
 *
 * @module src/config/loaders/typography
 */

import { z } from 'zod';
import { loadConfig } from './base';
import typographyConfigData from '../yaml/typography.yml';

/**
 * CSS unit validation for font sizes
 */
const fontSizeSchema = z
  .string()
  .refine((val) => /^\d+(\.\d+)?(rem|px|em)$/.test(val), {
    message: 'Invalid font size. Use rem, px, or em with a numeric value.',
  });

/**
 * Typography configuration schema
 */
export const typographyConfigSchema = z.object({
  fontFamily: z
    .object({
      sans: z
        .array(z.string())
        .default([
          'ui-sans-serif',
          'system-ui',
          '-apple-system',
          'BlinkMacSystemFont',
          '"Segoe UI"',
          'Roboto',
          '"Helvetica Neue"',
          'Arial',
          '"Noto Sans"',
          'sans-serif',
          '"Apple Color Emoji"',
          '"Segoe UI Emoji"',
          '"Segoe UI Symbol"',
          '"Noto Color Emoji"',
        ]),
      serif: z
        .array(z.string())
        .default(['ui-serif', 'Georgia', 'Cambria', '"Times New Roman"', 'Times', 'serif']),
      mono: z
        .array(z.string())
        .default([
          '"Fira Code"',
          '"SF Mono"',
          'SFMono-Regular',
          'Menlo',
          'Monaco',
          'Consolas',
          '"Liberation Mono"',
          '"Courier New"',
          'monospace',
        ]),
    })
    .default({}),
  fontSize: z
    .object({
      xs: fontSizeSchema.default('0.75rem'),
      sm: fontSizeSchema.default('0.875rem'),
      base: fontSizeSchema.default('1rem'),
      lg: fontSizeSchema.default('1.125rem'),
      xl: fontSizeSchema.default('1.25rem'),
      '2xl': fontSizeSchema.default('1.5rem'),
      '3xl': fontSizeSchema.default('1.875rem'),
      '4xl': fontSizeSchema.default('2.25rem'),
    })
    .default({}),
  lineHeight: z
    .object({
      tight: z.number().min(1).max(2).default(1.25),
      snug: z.number().min(1).max(2).default(1.375),
      normal: z.number().min(1).max(2).default(1.5),
      relaxed: z.number().min(1).max(2).default(1.625),
      loose: z.number().min(1).max(2.5).default(1.75),
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
      maxWidth: z.string().default('65ch'),
      useSerif: z.boolean().default(false),
      paragraphSpacing: z.string().default('1.25em'),
      headingSpacing: z
        .object({
          before: z.string().default('1.5em'),
          after: z.string().default('0.5em'),
        })
        .default({}),
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
      'ui-sans-serif',
      'system-ui',
      '-apple-system',
      'BlinkMacSystemFont',
      '"Segoe UI"',
      'Roboto',
      '"Helvetica Neue"',
      'Arial',
      '"Noto Sans"',
      'sans-serif',
      '"Apple Color Emoji"',
      '"Segoe UI Emoji"',
      '"Segoe UI Symbol"',
      '"Noto Color Emoji"',
    ],
    serif: ['ui-serif', 'Georgia', 'Cambria', '"Times New Roman"', 'Times', 'serif'],
    mono: [
      '"Fira Code"',
      '"SF Mono"',
      'SFMono-Regular',
      'Menlo',
      'Monaco',
      'Consolas',
      '"Liberation Mono"',
      '"Courier New"',
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
    tight: 1.25,
    snug: 1.375,
    normal: 1.5,
    relaxed: 1.625,
    loose: 1.75,
  },
  fontWeight: {
    normal: 400,
    medium: 500,
    semibold: 600,
    bold: 700,
  },
  prose: {
    maxWidth: '65ch',
    useSerif: false,
    paragraphSpacing: '1.25em',
    headingSpacing: {
      before: '1.5em',
      after: '0.5em',
    },
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
