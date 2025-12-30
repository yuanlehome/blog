/**
 * Tests for LLM Math Fixer Module
 */

import { describe, it, expect } from 'vitest';
import { LLMMathFixer } from '../../scripts/markdown/llm-math-fixer';

describe('LLM Math Fixer', () => {
  describe('Mock Provider', () => {
    const fixer = new LLMMathFixer({ provider: 'mock' });

    it('should remove all $ characters from block math', async () => {
      const badSample = '$\\displaystyle\\sum_{i=1}^{n}$ x_i';
      const result = await fixer.fixMathBlock(badSample);

      expect(result.fixed).not.toContain('$');
      expect(result.confidence).toBe('high');
      expect(result.notes).toContain('Removed 2 $ delimiter(s)');
    });

    it('should fix the specific bad sample from issue', async () => {
      const badSample =
        '\\displaystyle\\exp(m_{[0,L)}-m_{[0,L+1)})$}\\colorbox{orange}{$\\displaystyle\\sum_{l\\in[0,L)}\\exp(s_l-m_{[0,L)})$} + \\colorbox{lime}{$\\displaystyle\\exp(s_L-m_{[0,L+1)})$}$';

      const result = await fixer.fixMathBlock(badSample);

      // Should remove all $
      expect(result.fixed).not.toContain('$');
      // Should remove \displaystyle
      expect(result.fixed).not.toContain('\\displaystyle');
      // Should indicate fixing happened
      expect(result.notes.length).toBeGreaterThan(0);
    });

    it('should handle $} pattern (dollar before closing brace)', async () => {
      const badSample = '\\colorbox{red}{content$}';
      const result = await fixer.fixMathBlock(badSample);

      expect(result.fixed).not.toContain('$');
      expect(result.fixed).toContain('\\colorbox{red}{content}');
    });

    it('should handle {$...$ } pattern (dollars inside braces)', async () => {
      const badSample = '\\colorbox{blue}{$x + y$}';
      const result = await fixer.fixMathBlock(badSample);

      expect(result.fixed).not.toContain('$');
      expect(result.fixed).toContain('\\colorbox{blue}{x + y}');
    });

    it('should handle extra $ at the end', async () => {
      const badSample = '\\sum_{i=1}^{n} x_i$';
      const result = await fixer.fixMathBlock(badSample);

      expect(result.fixed).not.toContain('$');
      expect(result.fixed).toBe('\\sum_{i=1}^{n} x_i');
    });

    it('should balance missing closing braces', async () => {
      const badSample = '\\frac{a}{b + \\frac{c}{d}';
      const result = await fixer.fixMathBlock(badSample);

      expect(result.fixed).toContain('}');
      // Should have closing braces added
      expect(result.notes.some((note) => note.includes('closing brace'))).toBe(true);
    });

    it('should preserve escaped dollars \\$', async () => {
      const content = 'Price is \\$100';
      const result = await fixer.fixMathBlock(content);

      expect(result.fixed).toContain('\\$100');
    });

    it('should not modify already valid math', async () => {
      const validMath = '\\sum_{i=1}^{n} x_i = \\frac{1}{n} \\int_{0}^{\\infty} f(x) dx';
      const result = await fixer.fixMathBlock(validMath);

      expect(result.fixed).toBe(validMath.trim());
      expect(result.confidence).toBe('high');
    });

    it('should handle complex nested colorbox with inline $', async () => {
      const badSample = '\\colorbox{red}{$a$} + \\colorbox{blue}{$b$} + \\colorbox{green}{$c$}';
      const result = await fixer.fixMathBlock(badSample);

      expect(result.fixed).not.toContain('$');
      expect(result.fixed).toContain('\\colorbox{red}{a}');
      expect(result.fixed).toContain('\\colorbox{blue}{b}');
      expect(result.fixed).toContain('\\colorbox{green}{c}');
    });

    it('should report low confidence for badly unbalanced braces', async () => {
      const badSample = 'x = y}}}}}'; // Too many closing braces
      const result = await fixer.fixMathBlock(badSample);

      expect(result.confidence).toBe('low');
      expect(result.notes.some((note) => note.includes('Unbalanced braces'))).toBe(true);
    });

    it('should handle empty math blocks', async () => {
      const empty = '';
      const result = await fixer.fixMathBlock(empty);

      expect(result.fixed).toBe('');
      expect(result.confidence).toBe('high');
    });

    it('should handle math blocks with only whitespace', async () => {
      const whitespace = '   \n\n   ';
      const result = await fixer.fixMathBlock(whitespace);

      expect(result.fixed.trim()).toBe('');
    });
  });

  describe('Batch Processing', () => {
    const fixer = new LLMMathFixer({ provider: 'mock' });

    it('should deduplicate identical math blocks', async () => {
      const contents = [
        '$\\sum_{i}$',
        '$\\sum_{i}$', // duplicate
        '$\\int_{a}^{b}$',
        '$\\sum_{i}$', // duplicate
      ];

      const resultMap = await fixer.fixMathBlocks(contents);

      // Should only process unique contents
      expect(resultMap.size).toBeLessThanOrEqual(2);

      // All results should have no $
      resultMap.forEach((result) => {
        expect(result.fixed).not.toContain('$');
      });
    });

    it('should handle large batches', async () => {
      const contents = Array.from({ length: 50 }, (_, i) => `$x_{${i}}$`);
      const resultMap = await fixer.fixMathBlocks(contents);

      expect(resultMap.size).toBeGreaterThan(0);
      resultMap.forEach((result) => {
        expect(result.fixed).not.toContain('$');
      });
    });
  });

  describe('Real-world Bad Samples', () => {
    const fixer = new LLMMathFixer({ provider: 'mock' });

    it('should fix the complete problem example from the issue', async () => {
      // The exact bad sample from the issue description
      const problematicMath = `\\displaystyle\\exp(m_{[0,L)}-m_{[0,L+1)})$}\\colorbox{orange}{$\\displaystyle\\sum_{l\\in[0,L)}\\exp(s_l-m_{[0,L)})$} + \\colorbox{lime}{$\\displaystyle\\exp(s_L-m_{[0,L+1)})$}$`;

      const result = await fixer.fixMathBlock(problematicMath);

      // Critical assertions
      expect(result.fixed).not.toContain('$'); // No $ at all
      expect(result.fixed).not.toContain('\\displaystyle'); // No \displaystyle
      expect(result.fixed).toContain('\\colorbox{orange}'); // Preserve colorbox
      expect(result.fixed).toContain('\\colorbox{lime}'); // Preserve colorbox
      expect(result.notes.length).toBeGreaterThan(0); // Should have fix notes
    });

    it('should fix colorbox with displaystyle and inline math pattern', async () => {
      const badSample = '\\colorbox{red}{$\\displaystyle E = mc^2$}';
      const result = await fixer.fixMathBlock(badSample);

      expect(result.fixed).not.toContain('$');
      expect(result.fixed).not.toContain('\\displaystyle');
      expect(result.fixed).toContain('\\colorbox{red}{E = mc^2}');
    });

    it('should fix multiple inline $ with colorbox in sequence', async () => {
      const badSample = '$\\alpha$\\colorbox{yellow}{$\\beta$}$\\gamma$\\colorbox{cyan}{$\\delta$}';
      const result = await fixer.fixMathBlock(badSample);

      expect(result.fixed).not.toContain('$');
      expect(result.fixed).toContain('\\colorbox{yellow}');
      expect(result.fixed).toContain('\\colorbox{cyan}');
    });

    it('should handle nested colorbox with mixed $ patterns', async () => {
      const badSample = '\\colorbox{a}{outer $inner$ content}$';
      const result = await fixer.fixMathBlock(badSample);

      expect(result.fixed).not.toContain('$');
      expect(result.fixed).toContain('\\colorbox{a}{outer inner content}');
    });
  });
});
