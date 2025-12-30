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
 * Color format validation (hex, rgb, hsl)
 */
const colorSchema = z.string().refine(
  (val) => {
    // Allow hex colors: #RGB, #RRGGBB, #RRGGBBAA
    if (/^#([0-9A-Fa-f]{3}|[0-9A-Fa-f]{6}|[0-9A-Fa-f]{8})$/.test(val)) return true;
    // Allow rgb/rgba: rgb(r,g,b) or rgba(r,g,b,a)
    if (/^rgba?\(\s*\d+\s*,\s*\d+\s*,\s*\d+\s*(,\s*[\d.]+\s*)?\)$/.test(val)) return true;
    // Allow hsl/hsla: hsl(h,s%,l%) or hsla(h,s%,l%,a)
    if (/^hsla?\(\s*\d+\s*,\s*\d+%\s*,\s*\d+%\s*(,\s*[\d.]+\s*)?\)$/.test(val)) return true;
    return false;
  },
  { message: 'Invalid color format. Use hex (#RRGGBB), rgb(), or hsl() format.' },
);

/**
 * Color system schema
 */
const colorSystemSchema = z.object({
  brand: colorSchema.default('#3b82f6'),
  accent: colorSchema.default('#8b5cf6'),
  background: z
    .object({
      base: colorSchema.default('#ffffff'),
      subtle: colorSchema.default('#f8fafc'),
      muted: colorSchema.default('#f1f5f9'),
    })
    .default({}),
  foreground: z
    .object({
      base: colorSchema.default('#0f172a'),
      muted: colorSchema.default('#64748b'),
    })
    .default({}),
  border: z
    .object({
      default: colorSchema.default('#e2e8f0'),
      subtle: colorSchema.default('#f1f5f9'),
    })
    .default({}),
  card: z
    .object({
      background: colorSchema.default('#ffffff'),
      border: colorSchema.default('#e2e8f0'),
    })
    .default({}),
  code: z
    .object({
      background: colorSchema.default('#f8fafc'),
      foreground: colorSchema.default('#0f172a'),
      border: colorSchema.default('#e5e7eb'),
      scrollbar: colorSchema.default('#cbd5e1'),
    })
    .default({}),
});

/**
 * Code theme schema
 */
const codeThemeSchema = z.object({
  light: z.string().default('github-light'),
  dark: z.string().default('github-dark'),
  showLineNumbers: z.boolean().default(true),
  showCopyButton: z.boolean().default(true),
  wrapLongLines: z.boolean().default(false),
  inlineCodeStyle: z.enum(['subtle', 'boxed']).default('subtle'),
});

/**
 * Emphasis schema
 */
const emphasisSchema = z.object({
  linkUnderline: z.enum(['never', 'hover', 'always']).default('hover'),
  focusRing: z.boolean().default(true),
});

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
  colors: colorSystemSchema.default({}),
  darkColors: colorSystemSchema.default({}),
  codeTheme: codeThemeSchema.default({}),
  emphasis: emphasisSchema.default({}),
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
  colors: {
    brand: '#3b82f6',
    accent: '#8b5cf6',
    background: {
      base: '#ffffff',
      subtle: '#f8fafc',
      muted: '#f1f5f9',
    },
    foreground: {
      base: '#0f172a',
      muted: '#64748b',
    },
    border: {
      default: '#e2e8f0',
      subtle: '#f1f5f9',
    },
    card: {
      background: '#ffffff',
      border: '#e2e8f0',
    },
    code: {
      background: '#f8fafc',
      foreground: '#0f172a',
      border: '#e5e7eb',
      scrollbar: '#cbd5e1',
    },
  },
  darkColors: {
    brand: '#60a5fa',
    accent: '#a78bfa',
    background: {
      base: '#0f172a',
      subtle: '#1e293b',
      muted: '#334155',
    },
    foreground: {
      base: '#f1f5f9',
      muted: '#94a3b8',
    },
    border: {
      default: '#334155',
      subtle: '#1e293b',
    },
    card: {
      background: '#1e293b',
      border: '#334155',
    },
    code: {
      background: '#0b1221',
      foreground: '#e2e8f0',
      border: '#334155',
      scrollbar: '#475569',
    },
  },
  codeTheme: {
    light: 'github-light',
    dark: 'github-dark',
    showLineNumbers: true,
    showCopyButton: true,
    wrapLongLines: false,
    inlineCodeStyle: 'subtle',
  },
  emphasis: {
    linkUnderline: 'hover',
    focusRing: true,
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
