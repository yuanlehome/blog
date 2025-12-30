/**
 * CSS Variables Generator
 *
 * This module generates CSS custom properties (variables) from configuration.
 * The generated CSS is injected into the page to apply theme colors, typography,
 * and component styles.
 *
 * @module src/lib/style/css-vars
 */

import type { ThemeConfig } from '../../config/loaders/theme';
import type { TypographyConfig } from '../../config/loaders/typography';
import type { ComponentsConfig } from '../../config/loaders/components';

/**
 * Generate CSS variables for theme colors
 */
export function generateColorVars(colors: ThemeConfig['colors'], prefix = ''): string {
  const vars: string[] = [];

  if (colors.brand) vars.push(`--color-brand${prefix}: ${colors.brand};`);
  if (colors.accent) vars.push(`--color-accent${prefix}: ${colors.accent};`);

  if (colors.background) {
    if (colors.background.base)
      vars.push(`--color-background-base${prefix}: ${colors.background.base};`);
    if (colors.background.subtle)
      vars.push(`--color-background-subtle${prefix}: ${colors.background.subtle};`);
    if (colors.background.muted)
      vars.push(`--color-background-muted${prefix}: ${colors.background.muted};`);
  }

  if (colors.foreground) {
    if (colors.foreground.base)
      vars.push(`--color-foreground-base${prefix}: ${colors.foreground.base};`);
    if (colors.foreground.muted)
      vars.push(`--color-foreground-muted${prefix}: ${colors.foreground.muted};`);
  }

  if (colors.border) {
    if (colors.border.default) vars.push(`--color-border-default${prefix}: ${colors.border.default};`);
    if (colors.border.subtle) vars.push(`--color-border-subtle${prefix}: ${colors.border.subtle};`);
  }

  if (colors.card) {
    if (colors.card.background)
      vars.push(`--color-card-background${prefix}: ${colors.card.background};`);
    if (colors.card.border) vars.push(`--color-card-border${prefix}: ${colors.card.border};`);
  }

  if (colors.code) {
    if (colors.code.background)
      vars.push(`--color-code-background${prefix}: ${colors.code.background};`);
    if (colors.code.foreground)
      vars.push(`--color-code-foreground${prefix}: ${colors.code.foreground};`);
    if (colors.code.border) vars.push(`--color-code-border${prefix}: ${colors.code.border};`);
    if (colors.code.scrollbar) vars.push(`--color-code-scrollbar${prefix}: ${colors.code.scrollbar};`);
  }

  return vars.join('\n  ');
}

/**
 * Generate CSS variables for typography
 */
export function generateTypographyVars(typography: TypographyConfig): string {
  const vars: string[] = [];

  // Font families
  if (typography.fontFamily) {
    if (typography.fontFamily.sans)
      vars.push(`--font-sans: ${typography.fontFamily.sans.join(', ')};`);
    if (typography.fontFamily.serif)
      vars.push(`--font-serif: ${typography.fontFamily.serif.join(', ')};`);
    if (typography.fontFamily.mono)
      vars.push(`--font-mono: ${typography.fontFamily.mono.join(', ')};`);
  }

  // Font sizes
  if (typography.fontSize) {
    Object.entries(typography.fontSize).forEach(([key, value]) => {
      vars.push(`--font-size-${key}: ${value};`);
    });
  }

  // Line heights
  if (typography.lineHeight) {
    Object.entries(typography.lineHeight).forEach(([key, value]) => {
      vars.push(`--line-height-${key}: ${value};`);
    });
  }

  // Font weights
  if (typography.fontWeight) {
    Object.entries(typography.fontWeight).forEach(([key, value]) => {
      vars.push(`--font-weight-${key}: ${value};`);
    });
  }

  // Prose
  if (typography.prose) {
    if (typography.prose.maxWidth) vars.push(`--prose-max-width: ${typography.prose.maxWidth};`);
    if (typography.prose.paragraphSpacing)
      vars.push(`--prose-paragraph-spacing: ${typography.prose.paragraphSpacing};`);
    if (typography.prose.headingSpacing) {
      if (typography.prose.headingSpacing.before)
        vars.push(`--prose-heading-spacing-before: ${typography.prose.headingSpacing.before};`);
      if (typography.prose.headingSpacing.after)
        vars.push(`--prose-heading-spacing-after: ${typography.prose.headingSpacing.after};`);
    }
  }

  return vars.join('\n  ');
}

/**
 * Generate CSS variables for component styles
 */
export function generateComponentVars(components: ComponentsConfig): string {
  const vars: string[] = [];

  // Border radius
  if (components.radius) {
    Object.entries(components.radius).forEach(([key, value]) => {
      vars.push(`--radius-${key}: ${value};`);
    });
  }

  // Component-specific radius
  if (components.componentRadius) {
    Object.entries(components.componentRadius).forEach(([key, value]) => {
      vars.push(`--radius-${key}: ${value};`);
    });
  }

  // Shadows
  if (components.shadow) {
    Object.entries(components.shadow).forEach(([key, value]) => {
      vars.push(`--shadow-${key}: ${value};`);
    });
  }

  // Component-specific shadows
  if (components.componentShadow) {
    if (components.componentShadow.card)
      vars.push(`--shadow-card: ${components.componentShadow.card};`);
    if (components.componentShadow.cardDark)
      vars.push(`--shadow-card-dark: ${components.componentShadow.cardDark};`);
    if (components.componentShadow.header)
      vars.push(`--shadow-header: ${components.componentShadow.header};`);
    if (components.componentShadow.headerDark)
      vars.push(`--shadow-header-dark: ${components.componentShadow.headerDark};`);
  }

  // Border
  if (components.border) {
    if (components.border.style) vars.push(`--border-style: ${components.border.style};`);
    if (components.border.width) vars.push(`--border-width: ${components.border.width};`);
    if (typeof components.border.opacity === 'number')
      vars.push(`--border-opacity: ${components.border.opacity};`);
  }

  // Motion
  if (components.motion) {
    if (components.motion.duration) {
      Object.entries(components.motion.duration).forEach(([key, value]) => {
        vars.push(`--duration-${key}: ${value}ms;`);
      });
    }
    if (components.motion.easing) {
      Object.entries(components.motion.easing).forEach(([key, value]) => {
        vars.push(`--easing-${key}: ${value};`);
      });
    }
  }

  // Spacing multiplier
  if (components.spacingMultiplier && components.spacingScale) {
    const multiplier = components.spacingMultiplier[components.spacingScale];
    if (multiplier !== undefined) {
      vars.push(`--spacing-multiplier: ${multiplier};`);
    }
  }

  return vars.join('\n  ');
}

/**
 * Generate complete CSS stylesheet from configuration
 */
export function generateConfigCSS(
  theme: ThemeConfig,
  typography: TypographyConfig,
  components: ComponentsConfig,
): string {
  const lightVars = generateColorVars(theme.colors);
  const darkVars = generateColorVars(theme.darkColors);
  const typographyVars = generateTypographyVars(typography);
  const componentVars = generateComponentVars(components);

  return `/* Auto-generated CSS Variables from Configuration */
/* Do not edit manually - modify YAML config files instead */

:root {
  /* Light mode colors */
  ${lightVars}
  
  /* Typography */
  ${typographyVars}
  
  /* Components */
  ${componentVars}
}

.dark {
  /* Dark mode colors */
  ${darkVars}
}
`;
}
