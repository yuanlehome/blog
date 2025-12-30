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
 * Color validation helper - accepts hex, rgb, rgba, hsl, hsla
 */
const colorSchema = z
  .string()
  .regex(
    /^(#[0-9a-fA-F]{3,8}|rgb\(.*\)|rgba\(.*\)|hsl\(.*\)|hsla\(.*\))$/,
    'Invalid color format. Use hex (#abc or #aabbcc), rgb(), rgba(), hsl(), or hsla()',
  );

/**
 * Theme configuration schema
 */
export const themeConfigSchema = z.object({
  colorMode: z
    .object({
      default: z.enum(['light', 'dark', 'system']).default('system'),
      allowToggle: z.boolean().default(true),
      persist: z.boolean().default(true),
    })
    .default({}),
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
  colors: z
    .object({
      brand: colorSchema.default('#0284c7'),
      accent: colorSchema.default('#0284c7'),
      background: colorSchema.default('#fafafa'),
      foreground: colorSchema.default('#18181b'),
      muted: colorSchema.default('#71717a'),
      border: colorSchema.default('#e4e4e7'),
      card: colorSchema.default('#f4f4f5'),
      code: z
        .object({
          background: colorSchema.default('#f4f4f5'),
          foreground: colorSchema.default('#18181b'),
          border: colorSchema.default('#e4e4e7'),
          keyword: colorSchema.default('#0284c7'),
          string: colorSchema.default('#059669'),
          comment: colorSchema.default('#a1a1aa'),
          function: colorSchema.default('#0284c7'),
        })
        .default({}),
    })
    .default({}),
  darkColors: z
    .object({
      brand: colorSchema.default('#38bdf8'),
      accent: colorSchema.default('#38bdf8'),
      background: colorSchema.default('#18181b'),
      foreground: colorSchema.default('#fafafa'),
      muted: colorSchema.default('#a1a1aa'),
      border: colorSchema.default('#3f3f46'),
      card: colorSchema.default('#27272a'),
      code: z
        .object({
          background: colorSchema.default('#1f1f23'),
          foreground: colorSchema.default('#e4e4e7'),
          border: colorSchema.default('#3f3f46'),
          keyword: colorSchema.default('#38bdf8'),
          string: colorSchema.default('#34d399'),
          comment: colorSchema.default('#71717a'),
          function: colorSchema.default('#38bdf8'),
        })
        .default({}),
    })
    .default({}),
  emphasis: z
    .object({
      linkUnderline: z.enum(['never', 'hover', 'always']).default('hover'),
      focusRing: z.boolean().default(true),
    })
    .default({}),
  codeBlock: z
    .object({
      theme: z
        .object({
          light: z.string().default('github-light'),
          dark: z.string().default('github-dark'),
        })
        .default({}),
      showLineNumbers: z.boolean().default(true),
      showCopyButton: z.boolean().default(true),
      wrapLongLines: z.boolean().default(false),
      inlineCodeStyle: z.enum(['subtle', 'boxed']).default('subtle'),
      radius: z.string().default('0.9rem'),
      enableHighlight: z.boolean().default(true),
    })
    .default({}),
  header: z
    .object({
      variant: z.enum(['default', 'subtle', 'frosted', 'elevated']).default('default'),
      backgroundOpacity: z.number().min(0).max(1).default(0.92),
      blurStrength: z.string().default('10px'),
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
  colorMode: {
    default: 'system',
    allowToggle: true,
    persist: true,
  },
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
  colors: {
    brand: '#0284c7',
    accent: '#0284c7',
    background: '#fafafa',
    foreground: '#18181b',
    muted: '#71717a',
    border: '#e4e4e7',
    card: '#f4f4f5',
    code: {
      background: '#f4f4f5',
      foreground: '#18181b',
      border: '#e4e4e7',
      keyword: '#0284c7',
      string: '#059669',
      comment: '#a1a1aa',
      function: '#0284c7',
    },
  },
  darkColors: {
    brand: '#38bdf8',
    accent: '#38bdf8',
    background: '#18181b',
    foreground: '#fafafa',
    muted: '#a1a1aa',
    border: '#3f3f46',
    card: '#27272a',
    code: {
      background: '#1f1f23',
      foreground: '#e4e4e7',
      border: '#3f3f46',
      keyword: '#38bdf8',
      string: '#34d399',
      comment: '#71717a',
      function: '#38bdf8',
    },
  },
  emphasis: {
    linkUnderline: 'hover',
    focusRing: true,
  },
  codeBlock: {
    theme: {
      light: 'github-light',
      dark: 'github-dark',
    },
    showLineNumbers: true,
    showCopyButton: true,
    wrapLongLines: false,
    inlineCodeStyle: 'subtle',
    radius: '0.9rem',
    enableHighlight: true,
  },
  header: {
    variant: 'default',
    backgroundOpacity: 0.92,
    blurStrength: '10px',
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
