/**
 * Tests for CSS generator utility
 */

import { describe, it, expect } from 'vitest';
import { generateThemeCSS } from '../../../src/lib/theme/cssGenerator';
import { defaultThemeConfig } from '../../../src/config/loaders/theme';
import { defaultTypographyConfig } from '../../../src/config/loaders/typography';
import { defaultComponentsConfig } from '../../../src/config/loaders/components';

describe('CSS Generator', () => {
  describe('generateThemeCSS', () => {
    it('should generate valid CSS', () => {
      const css = generateThemeCSS(
        defaultThemeConfig,
        defaultTypographyConfig,
        defaultComponentsConfig,
      );

      expect(css).toBeDefined();
      expect(css).toContain(':root');
      expect(css).toContain('.dark');
    });

    it('should include typography variables', () => {
      const css = generateThemeCSS(
        defaultThemeConfig,
        defaultTypographyConfig,
        defaultComponentsConfig,
      );

      expect(css).toContain('--font-sans:');
      expect(css).toContain('--font-mono:');
      expect(css).toContain('--font-size-base:');
      expect(css).toContain('--line-height-body:');
      expect(css).toContain('--font-weight-bold:');
    });

    it('should include component variables', () => {
      const css = generateThemeCSS(
        defaultThemeConfig,
        defaultTypographyConfig,
        defaultComponentsConfig,
      );

      expect(css).toContain('--radius-sm:');
      expect(css).toContain('--radius-lg:');
      expect(css).toContain('--shadow-card:');
      expect(css).toContain('--border-style:');
      expect(css).toContain('--motion-duration:');
    });

    it('should include color variables', () => {
      const css = generateThemeCSS(
        defaultThemeConfig,
        defaultTypographyConfig,
        defaultComponentsConfig,
      );

      expect(css).toContain('--color-brand:');
      expect(css).toContain('--color-background:');
      expect(css).toContain('--color-code-background:');
      expect(css).toContain('--color-code-keyword:');
    });

    it('should include legacy code block variables', () => {
      const css = generateThemeCSS(
        defaultThemeConfig,
        defaultTypographyConfig,
        defaultComponentsConfig,
      );

      expect(css).toContain('--code-font:');
      expect(css).toContain('--code-radius:');
      expect(css).toContain('--code-bg:');
      expect(css).toContain('--code-fg:');
    });

    it('should include legacy header variables', () => {
      const css = generateThemeCSS(
        defaultThemeConfig,
        defaultTypographyConfig,
        defaultComponentsConfig,
      );

      expect(css).toContain('--header-bg:');
      expect(css).toContain('--header-shadow:');
    });

    it('should include dark mode variables', () => {
      const css = generateThemeCSS(
        defaultThemeConfig,
        defaultTypographyConfig,
        defaultComponentsConfig,
      );

      const darkSection = css.split('.dark')[1];
      expect(darkSection).toBeDefined();
      expect(darkSection).toContain('--color-brand:');
      expect(darkSection).toContain('--shadow-card:');
    });

    it('should use correct font family from typography config', () => {
      const css = generateThemeCSS(
        defaultThemeConfig,
        defaultTypographyConfig,
        defaultComponentsConfig,
      );

      expect(css).toContain('Fira Code');
      expect(css).toContain('system-ui');
    });

    it('should use correct shadow values from components config', () => {
      const css = generateThemeCSS(
        defaultThemeConfig,
        defaultTypographyConfig,
        defaultComponentsConfig,
      );

      // Default shadow is 'md'
      expect(css).toContain('--shadow-card: 0 4px 6px -1px rgb(0 0 0 / 0.1)');
    });

    it('should use correct radius values from theme config', () => {
      const css = generateThemeCSS(
        defaultThemeConfig,
        defaultTypographyConfig,
        defaultComponentsConfig,
      );

      expect(css).toContain(`--code-radius: ${defaultThemeConfig.codeBlock.radius}`);
    });

    it('should handle custom theme colors', () => {
      const customTheme = {
        ...defaultThemeConfig,
        colors: {
          ...defaultThemeConfig.colors,
          brand: '#ff0000',
        },
      };

      const css = generateThemeCSS(customTheme, defaultTypographyConfig, defaultComponentsConfig);

      expect(css).toContain('--color-brand: #ff0000');
    });

    it('should handle custom typography', () => {
      const customTypography = {
        ...defaultTypographyConfig,
        fontSize: {
          ...defaultTypographyConfig.fontSize,
          base: '18px',
        },
      };

      const css = generateThemeCSS(defaultThemeConfig, customTypography, defaultComponentsConfig);

      expect(css).toContain('--font-size-base: 18px');
    });

    it('should handle custom component settings', () => {
      const customComponents = {
        ...defaultComponentsConfig,
        radius: {
          ...defaultComponentsConfig.radius,
          lg: '2rem',
        },
      };

      const css = generateThemeCSS(defaultThemeConfig, defaultTypographyConfig, customComponents);

      expect(css).toContain('--radius-lg: 2rem');
    });

    it('should generate valid CSS syntax (basic validation)', () => {
      const css = generateThemeCSS(
        defaultThemeConfig,
        defaultTypographyConfig,
        defaultComponentsConfig,
      );

      // Check for basic CSS structure
      expect(css).toMatch(/:root\s*\{/);
      expect(css).toMatch(/\.dark\s*\{/);
      // Check that all declarations have colons and semicolons
      const declarations = css.match(/--[\w-]+:\s*[^;]+;/g);
      expect(declarations).toBeTruthy();
      expect(declarations!.length).toBeGreaterThan(0);
    });
  });
});
