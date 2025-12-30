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
      brand: colorSchema.default('#3b82f6'),
      accent: colorSchema.default('#8b5cf6'),
      background: colorSchema.default('#ffffff'),
      foreground: colorSchema.default('#111827'),
      muted: colorSchema.default('#6b7280'),
      border: colorSchema.default('#e5e7eb'),
      card: colorSchema.default('#f9fafb'),
      code: z
        .object({
          background: colorSchema.default('#f8fafc'),
          foreground: colorSchema.default('#0f172a'),
          border: colorSchema.default('#e2e8f0'),
          keyword: colorSchema.default('#3b82f6'),
          string: colorSchema.default('#10b981'),
          comment: colorSchema.default('#94a3b8'),
          function: colorSchema.default('#8b5cf6'),
        })
        .default({}),
    })
    .default({}),
  darkColors: z
    .object({
      brand: colorSchema.default('#60a5fa'),
      accent: colorSchema.default('#a78bfa'),
      background: colorSchema.default('#111827'),
      foreground: colorSchema.default('#f9fafb'),
      muted: colorSchema.default('#9ca3af'),
      border: colorSchema.default('#374151'),
      card: colorSchema.default('#1f2937'),
      code: z
        .object({
          background: colorSchema.default('#0b1221'),
          foreground: colorSchema.default('#e2e8f0'),
          border: colorSchema.default('#334155'),
          keyword: colorSchema.default('#60a5fa'),
          string: colorSchema.default('#34d399'),
          comment: colorSchema.default('#64748b'),
          function: colorSchema.default('#a78bfa'),
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
    brand: '#3b82f6',
    accent: '#8b5cf6',
    background: '#ffffff',
    foreground: '#111827',
    muted: '#6b7280',
    border: '#e5e7eb',
    card: '#f9fafb',
    code: {
      background: '#f8fafc',
      foreground: '#0f172a',
      border: '#e2e8f0',
      keyword: '#3b82f6',
      string: '#10b981',
      comment: '#94a3b8',
      function: '#8b5cf6',
    },
  },
  darkColors: {
    brand: '#60a5fa',
    accent: '#a78bfa',
    background: '#111827',
    foreground: '#f9fafb',
    muted: '#9ca3af',
    border: '#374151',
    card: '#1f2937',
    code: {
      background: '#0b1221',
      foreground: '#e2e8f0',
      border: '#334155',
      keyword: '#60a5fa',
      string: '#34d399',
      comment: '#64748b',
      function: '#a78bfa',
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
