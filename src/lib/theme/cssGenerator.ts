/**
 * Theme CSS Variable Generator
 *
 * This module generates CSS variables from configuration files.
 * It transforms YAML-based theme settings into CSS custom properties.
 *
 * @module src/lib/theme/cssGenerator
 */

import type { ThemeConfig } from '../../config/loaders/theme';
import type { TypographyConfig } from '../../config/loaders/typography';
import type { ComponentsConfig } from '../../config/loaders/components';

/**
 * Generate CSS custom properties for colors
 *
 * @param colors - Color configuration object
 * @param prefix - CSS variable prefix (default: '--color')
 * @returns CSS variable declarations as a string
 */
function generateColorVariables(
  colors: Record<string, string | Record<string, string>>,
  prefix = '--color',
): string {
  const variables: string[] = [];

  for (const [key, value] of Object.entries(colors)) {
    if (typeof value === 'string') {
      variables.push(`  ${prefix}-${key}: ${value};`);
    } else if (typeof value === 'object') {
      // Handle nested objects (e.g., code colors)
      for (const [subKey, subValue] of Object.entries(value)) {
        variables.push(`  ${prefix}-${key}-${subKey}: ${subValue};`);
      }
    }
  }

  return variables.join('\n');
}

/**
 * Generate CSS custom properties for typography
 *
 * @param typography - Typography configuration
 * @returns CSS variable declarations as a string
 */
function generateTypographyVariables(typography: TypographyConfig): string {
  const variables: string[] = [];

  // Font families
  variables.push(`  --font-sans: ${typography.fontFamily.sans.join(', ')};`);
  variables.push(`  --font-serif: ${typography.fontFamily.serif.join(', ')};`);
  variables.push(`  --font-mono: ${typography.fontFamily.mono.join(', ')};`);

  // Font sizes
  for (const [key, value] of Object.entries(typography.fontSize)) {
    variables.push(`  --font-size-${key}: ${value};`);
  }

  // Line heights
  variables.push(`  --line-height-body: ${typography.lineHeight.body};`);
  variables.push(`  --line-height-heading: ${typography.lineHeight.heading};`);
  variables.push(`  --line-height-code: ${typography.lineHeight.code};`);
  variables.push(`  --line-height-tight: ${typography.lineHeight.tight};`);

  // Font weights
  variables.push(`  --font-weight-normal: ${typography.fontWeight.normal};`);
  variables.push(`  --font-weight-medium: ${typography.fontWeight.medium};`);
  variables.push(`  --font-weight-semibold: ${typography.fontWeight.semibold};`);
  variables.push(`  --font-weight-bold: ${typography.fontWeight.bold};`);

  return variables.join('\n');
}

/**
 * Generate CSS custom properties for components
 *
 * @param components - Components configuration
 * @returns CSS variable declarations as a string
 */
function generateComponentVariables(components: ComponentsConfig): string {
  const variables: string[] = [];

  // Border radius
  variables.push(`  --radius-sm: ${components.radius.sm};`);
  variables.push(`  --radius-md: ${components.radius.md};`);
  variables.push(`  --radius-lg: ${components.radius.lg};`);
  variables.push(`  --radius-xl: ${components.radius.xl};`);

  // Shadow values
  const shadowLevel = components.shadow.card;
  variables.push(`  --shadow-card: ${components.shadowValues[shadowLevel]};`);

  const codeBlockShadow = components.shadow.codeBlock;
  variables.push(`  --shadow-code-block: ${components.shadowValues[codeBlockShadow]};`);

  const headerShadow = components.shadow.header;
  variables.push(`  --shadow-header: ${components.shadowValues[headerShadow]};`);

  // Border
  variables.push(`  --border-style: ${components.border.style};`);
  variables.push(`  --border-width: ${components.border.width};`);

  // Motion timing
  const motionLevel = components.motion.level;
  const timing = components.motionTiming[motionLevel];
  variables.push(`  --motion-duration: ${timing.duration};`);
  variables.push(`  --motion-easing: ${timing.easing};`);

  // Spacing
  const spacingMultiplier = components.spacingMultiplier[components.spacingScale];
  variables.push(`  --spacing-multiplier: ${spacingMultiplier};`);

  return variables.join('\n');
}

/**
 * Generate complete CSS stylesheet with all theme variables
 *
 * @param theme - Theme configuration
 * @param typography - Typography configuration
 * @param components - Components configuration
 * @returns Complete CSS stylesheet as a string
 */
export function generateThemeCSS(
  theme: ThemeConfig,
  typography: TypographyConfig,
  components: ComponentsConfig,
): string {
  const lightColors = generateColorVariables(theme.colors);
  const darkColors = generateColorVariables(theme.darkColors);
  const typographyVars = generateTypographyVariables(typography);
  const componentVars = generateComponentVariables(components);

  // Also generate dark mode shadow values
  const darkShadowLevel = components.shadow.card;
  const darkCardShadow = components.shadowValuesDark[darkShadowLevel];

  const darkCodeBlockShadow = components.shadowValuesDark[components.shadow.codeBlock];
  const darkHeaderShadow = components.shadowValuesDark[components.shadow.header];

  return `:root {
  /* Typography */
${typographyVars}

  /* Components */
${componentVars}

  /* Light Mode Colors */
${lightColors}

  /* Cyberpunk Neon Effects - Light Mode */
  --neon-glow-cyan: 0 0 10px rgb(6 182 212 / 0.5), 0 0 20px rgb(6 182 212 / 0.3);
  --neon-glow-purple: 0 0 10px rgb(168 85 247 / 0.5), 0 0 20px rgb(168 85 247 / 0.3);
  --neon-glow-pink: 0 0 10px rgb(236 72 153 / 0.5), 0 0 20px rgb(236 72 153 / 0.3);
  --neon-glow-subtle: 0 0 8px rgb(6 182 212 / 0.3);
  --glass-bg: rgba(39, 39, 42, 0.75);
  --glass-border: rgba(63, 63, 70, 0.5);
  --scanline-opacity: 0.02;
  --noise-opacity: 0.03;

  /* Code block specific (legacy support) */
  --code-font: var(--font-mono);
  --code-radius: ${theme.codeBlock.radius};
  --code-border: 1px solid var(--color-code-border);
  --code-bg: var(--color-code-background);
  --code-fg: var(--color-code-foreground);
  --code-scrollbar: rgb(82 82 91 / 0.6);
  --code-highlight-bg: rgb(6 182 212 / 0.15);
  --code-highlight-accent: rgb(6 182 212);

  /* Header specific (legacy support) */
  --header-bg: rgba(26, 26, 31, 0.85);
  --header-bg-scrolled: rgba(26, 26, 31, 0.95);
  --header-border: rgb(63 63 70 / 0.7);
  --header-shadow: var(--shadow-header);
  --header-shadow-strong: ${components.shadowValues.lg};
}

.dark {
  /* Dark Mode Colors */
${darkColors}

  /* Dark mode shadows */
  --shadow-card: ${darkCardShadow};
  --shadow-code-block: ${darkCodeBlockShadow};
  --shadow-header: ${darkHeaderShadow};

  /* Cyberpunk Neon Effects - Dark Mode (more intense) */
  --neon-glow-cyan: 0 0 15px rgb(34 211 238 / 0.6), 0 0 30px rgb(34 211 238 / 0.4), 0 0 45px rgb(34 211 238 / 0.2);
  --neon-glow-purple: 0 0 15px rgb(192 132 252 / 0.6), 0 0 30px rgb(192 132 252 / 0.4), 0 0 45px rgb(192 132 252 / 0.2);
  --neon-glow-pink: 0 0 15px rgb(244 114 182 / 0.6), 0 0 30px rgb(244 114 182 / 0.4), 0 0 45px rgb(244 114 182 / 0.2);
  --neon-glow-subtle: 0 0 12px rgb(34 211 238 / 0.4), 0 0 24px rgb(34 211 238 / 0.2);
  --glass-bg: rgba(24, 24, 27, 0.65);
  --glass-border: rgba(82, 82, 91, 0.4);
  --scanline-opacity: 0.03;
  --noise-opacity: 0.04;

  /* Code block specific (legacy support) */
  --code-border: 1px solid var(--color-code-border);
  --code-bg: var(--color-code-background);
  --code-fg: var(--color-code-foreground);
  --code-scrollbar: rgb(63 63 70 / 0.7);
  --code-highlight-bg: rgb(34 211 238 / 0.2);
  --code-highlight-accent: rgb(34 211 238);

  /* Header specific (legacy support) */
  --header-bg: rgba(10, 10, 15, 0.75);
  --header-bg-scrolled: rgba(10, 10, 15, 0.92);
  --header-border: rgb(82 82 91 / 0.6);
  --header-shadow: var(--shadow-header);
  --header-shadow-strong: ${components.shadowValuesDark.lg};
}
`;
}

/**
 * Generate inline CSS for use in Astro components
 *
 * @param theme - Theme configuration
 * @param typography - Typography configuration
 * @param components - Components configuration
 * @returns CSS string ready to be used in a style tag
 */
export function generateInlineThemeCSS(
  theme: ThemeConfig,
  typography: TypographyConfig,
  components: ComponentsConfig,
): string {
  return `<style>${generateThemeCSS(theme, typography, components)}</style>`;
}
