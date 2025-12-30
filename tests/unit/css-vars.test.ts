/**
 * Tests for CSS variables generator
 */

import { describe, it, expect } from 'vitest';
import {
  generateColorVars,
  generateTypographyVars,
  generateComponentVars,
  generateConfigCSS,
} from '../../src/lib/style/css-vars';
import { defaultThemeConfig } from '../../src/config/loaders/theme';
import { defaultTypographyConfig } from '../../src/config/loaders/typography';
import { defaultComponentsConfig } from '../../src/config/loaders/components';

describe('CSS Variables Generator', () => {
  describe('generateColorVars', () => {
    it('should generate color variables from theme config', () => {
      const vars = generateColorVars(defaultThemeConfig.colors);

      expect(vars).toContain('--color-brand:');
      expect(vars).toContain('--color-accent:');
      expect(vars).toContain('--color-background-base:');
      expect(vars).toContain('--color-foreground-base:');
      expect(vars).toContain('--color-border-default:');
      expect(vars).toContain('--color-card-background:');
      expect(vars).toContain('--color-code-background:');
    });

    it('should handle empty colors object', () => {
      const vars = generateColorVars({
        brand: '#000000',
        accent: '#ffffff',
        background: { base: '#fff', subtle: '#fff', muted: '#fff' },
        foreground: { base: '#000', muted: '#000' },
        border: { default: '#000', subtle: '#000' },
        card: { background: '#fff', border: '#000' },
        code: { background: '#fff', foreground: '#000', border: '#000', scrollbar: '#000' },
      });

      expect(vars).toBeTruthy();
      expect(vars.length).toBeGreaterThan(0);
    });
  });

  describe('generateTypographyVars', () => {
    it('should generate typography variables', () => {
      const vars = generateTypographyVars(defaultTypographyConfig);

      expect(vars).toContain('--font-sans:');
      expect(vars).toContain('--font-serif:');
      expect(vars).toContain('--font-mono:');
      expect(vars).toContain('--font-size-base:');
      expect(vars).toContain('--line-height-normal:');
      expect(vars).toContain('--font-weight-normal:');
      expect(vars).toContain('--prose-max-width:');
    });

    it('should handle all font sizes', () => {
      const vars = generateTypographyVars(defaultTypographyConfig);

      expect(vars).toContain('--font-size-xs:');
      expect(vars).toContain('--font-size-sm:');
      expect(vars).toContain('--font-size-lg:');
      expect(vars).toContain('--font-size-xl:');
    });
  });

  describe('generateComponentVars', () => {
    it('should generate component style variables', () => {
      const vars = generateComponentVars(defaultComponentsConfig);

      expect(vars).toContain('--radius-');
      expect(vars).toContain('--shadow-');
      expect(vars).toContain('--border-style:');
      expect(vars).toContain('--duration-');
      expect(vars).toContain('--easing-');
      expect(vars).toContain('--spacing-multiplier:');
    });

    it('should include component-specific radius', () => {
      const vars = generateComponentVars(defaultComponentsConfig);

      expect(vars).toContain('--radius-card:');
      expect(vars).toContain('--radius-button:');
      expect(vars).toContain('--radius-code:');
    });

    it('should include motion variables', () => {
      const vars = generateComponentVars(defaultComponentsConfig);

      expect(vars).toContain('--duration-fast:');
      expect(vars).toContain('--duration-normal:');
      expect(vars).toContain('--duration-slow:');
      expect(vars).toContain('--easing-default:');
    });
  });

  describe('generateConfigCSS', () => {
    it('should generate complete CSS with all sections', () => {
      const css = generateConfigCSS(
        defaultThemeConfig,
        defaultTypographyConfig,
        defaultComponentsConfig,
      );

      expect(css).toContain(':root {');
      expect(css).toContain('.dark {');
      expect(css).toContain('/* Light mode colors */');
      expect(css).toContain('/* Dark mode colors */');
      expect(css).toContain('/* Typography */');
      expect(css).toContain('/* Components */');
    });

    it('should include all color variables', () => {
      const css = generateConfigCSS(
        defaultThemeConfig,
        defaultTypographyConfig,
        defaultComponentsConfig,
      );

      expect(css).toContain('--color-brand:');
      expect(css).toContain('--color-accent:');
    });

    it('should include typography variables', () => {
      const css = generateConfigCSS(
        defaultThemeConfig,
        defaultTypographyConfig,
        defaultComponentsConfig,
      );

      expect(css).toContain('--font-sans:');
      expect(css).toContain('--font-size-base:');
    });

    it('should include component variables', () => {
      const css = generateConfigCSS(
        defaultThemeConfig,
        defaultTypographyConfig,
        defaultComponentsConfig,
      );

      expect(css).toContain('--radius-');
      expect(css).toContain('--shadow-');
    });

    it('should be valid CSS format', () => {
      const css = generateConfigCSS(
        defaultThemeConfig,
        defaultTypographyConfig,
        defaultComponentsConfig,
      );

      // Check basic CSS structure
      expect(css).toMatch(/:root\s*\{/);
      expect(css).toMatch(/\.dark\s*\{/);
      expect(css).toMatch(/--[\w-]+:\s*[^;]+;/);
    });
  });
});
