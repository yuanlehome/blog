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

  /* Code block specific (legacy support) */
  --code-font: var(--font-mono);
  --code-radius: ${theme.codeBlock.radius};
  --code-border: 1px solid var(--color-code-border);
  --code-bg: var(--color-code-background);
  --code-fg: var(--color-code-foreground);
  --code-scrollbar: rgb(203 213 225 / 0.75);
  --code-highlight-bg: rgb(59 130 246 / 0.12);
  --code-highlight-accent: rgb(59 130 246);

  /* Header specific (legacy support) */
  --header-bg: rgb(248 250 252 / ${theme.header.backgroundOpacity});
  --header-bg-scrolled: rgb(248 250 252 / ${Math.min(theme.header.backgroundOpacity + 0.06, 1)});
  --header-border: rgb(226 232 240 / 0.9);
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

  /* Code block specific (legacy support) */
  --code-border: 1px solid var(--color-code-border);
  --code-bg: var(--color-code-background);
  --code-fg: var(--color-code-foreground);
  --code-scrollbar: rgb(71 85 105 / 0.8);
  --code-highlight-bg: rgb(59 130 246 / 0.2);
  --code-highlight-accent: rgb(96 165 250);

  /* Header specific (legacy support) */
  --header-bg: rgb(15 23 42 / ${theme.header.backgroundOpacity - 0.2});
  --header-bg-scrolled: rgb(15 23 42 / ${Math.min(theme.header.backgroundOpacity - 0.08, 0.92)});
  --header-border: rgb(71 85 105 / 0.8);
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
