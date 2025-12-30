/**
 * Tests for Math Fixer Module
 */

import { describe, it, expect } from 'vitest';
import { fixMathBlock, degradeToTexBlock } from '../../scripts/markdown/math-fixer';

describe('Math Fixer', () => {
  describe('fixMathBlock', () => {
    it('should fix the example with \\colorbox{red}{ $$ causing block split', () => {
      const broken = `
\\sum_{l\\in[0,L+1)}\\exp(s_l-m_{[0,L+1)}) = \\colorbox{red}{
$$

\\displaystyle\\exp(m_{[0,L)}-m_{[0,L+1)})$}\\colorbox{orange}{$\\displaystyle\\sum_{l\\in[0,L)}\\exp(s_l-m_{[0,L)})$} + \\colorbox{lime}{$\\displaystyle\\exp(s_L-m_{[0,L+1)})$}
`;

      const result = fixMathBlock(broken);

      // Should remove stray $$ inside the block
      expect(result.fixed).not.toContain('$$');
      // Should remove nested $ delimiters
      expect(result.changed).toBe(true);
      expect(result.issues).toContain('Removed stray $$ delimiters inside block math');
      expect(result.issues).toContain('Removed nested inline $ delimiters');
    });

    it('should remove stray $$ delimiters inside block math', () => {
      const broken = 'x = y $$ z = w';
      const result = fixMathBlock(broken);

      expect(result.fixed).not.toContain('$$');
      expect(result.changed).toBe(true);
      expect(result.issues).toContain('Removed stray $$ delimiters inside block math');
    });

    it('should remove nested inline $ delimiters', () => {
      const broken = '$\\displaystyle\\sum_{i=1}^{n}$ x_i';
      const result = fixMathBlock(broken);

      expect(result.fixed).not.toContain('$');
      expect(result.changed).toBe(true);
      expect(result.issues).toContain('Removed nested inline $ delimiters');
    });

    it('should preserve escaped dollars \\$', () => {
      const content = 'Price is \\$100';
      const result = fixMathBlock(content);

      expect(result.fixed).toContain('\\$100');
    });

    it('should balance missing closing braces', () => {
      const broken = '\\frac{a}{b + \\frac{c}{d}';
      const result = fixMathBlock(broken);

      expect(result.changed).toBe(true);
      expect(result.fixed).toContain('}');
      expect(result.issues).toContain('Balanced brackets/braces');
    });

    it('should balance missing closing square brackets', () => {
      const broken = 'x = [a, b, c';
      const result = fixMathBlock(broken);

      expect(result.changed).toBe(true);
      expect(result.fixed).toContain(']');
    });

    it('should balance missing closing parentheses', () => {
      const broken = 'f(x + g(y)';
      const result = fixMathBlock(broken);

      expect(result.changed).toBe(true);
    });

    it('should detect pseudo-math (plain text in $$)', () => {
      const plainText = `
这是一段普通的中文文字，不包含任何数学符号或者LaTeX命令，只是普通文本而已。
这只是被错误地包裹在数学块中的普通段落而已，这里有很多文字但是没有数学内容。
应该被检测出来并降级处理，因为这根本不是数学公式，而是纯文本内容，包含大量的中文字符但是没有任何数学命令或符号。
`;

      const result = fixMathBlock(plainText);

      expect(result.confidence).toBe('low');
      expect(result.issues.some((issue) => issue.includes('pseudo-math'))).toBe(true);
    });

    it('should NOT detect real math as pseudo-math', () => {
      const realMath = `
\\sum_{i=1}^{n} x_i = \\frac{1}{n} \\int_{0}^{\\infty} f(x) dx
`;

      const result = fixMathBlock(realMath);

      expect(result.issues.some((issue) => issue.includes('pseudo-math'))).toBe(false);
    });

    it('should fix broken \\colorbox commands', () => {
      const broken = '\\colorbox{red}{incomplete content';
      const result = fixMathBlock(broken);

      // Should attempt to close the second brace
      expect(result.changed).toBe(true);
    });

    it('should normalize invisible characters', () => {
      const withInvisible = 'x\u00a0=\u00a0y'; // non-breaking spaces
      const result = fixMathBlock(withInvisible);

      expect(result.changed).toBe(true);
      expect(result.fixed).toBe('x = y');
      expect(result.issues).toContain('Normalized invisible characters');
    });

    it('should validate fixed math blocks', () => {
      const validMath = '\\frac{a}{b} + \\sqrt{c}';
      const result = fixMathBlock(validMath);

      // Should pass validation
      expect(result.confidence).toBe('high');
    });

    it('should detect unbalanced brackets in validation', () => {
      const unbalanced = '\\frac{a}{b';
      const result = fixMathBlock(unbalanced);

      // Should fix by adding missing brace
      expect(result.changed).toBe(true);
      expect(result.fixed).toContain('}');
    });

    it('should not modify already valid math', () => {
      const validMath = `
\\begin{align}
  \\sum_{i=1}^{n} x_i &= \\frac{1}{n} \\int_{0}^{\\infty} f(x) dx \\\\
  E[X] &= \\mu
\\end{align}
`;

      const result = fixMathBlock(validMath);

      expect(result.changed).toBe(false);
      expect(result.confidence).toBe('high');
    });

    it('should handle complex nested structures', () => {
      const complex = `
\\frac{
  \\sum_{i=1}^{n} \\left( x_i - \\bar{x} \\right)^2
}{
  n - 1
}
`;

      const result = fixMathBlock(complex);

      // Should not break valid complex math
      expect(result.confidence).toBe('high');
    });

    it('should remove both stray $$ and nested $ in same block', () => {
      const broken = 'x = $y$ $$ z = $w$';
      const result = fixMathBlock(broken);

      expect(result.fixed).not.toContain('$$');
      expect(result.fixed).not.toContain('$');
      expect(result.changed).toBe(true);
    });

    it('should handle empty math blocks', () => {
      const empty = '';
      const result = fixMathBlock(empty);

      expect(result.fixed).toBe('');
      expect(result.changed).toBe(false);
    });

    it('should handle math blocks with only whitespace', () => {
      const whitespace = '   \n\n   ';
      const result = fixMathBlock(whitespace);

      expect(result.changed).toBe(false);
    });

    it('should not treat math with Chinese characters as pseudo-math if has LaTeX', () => {
      const mathWithChinese = '设函数 f(x) = \\sin(x) 在区间 [0, \\pi] 上连续';
      const result = fixMathBlock(mathWithChinese);

      // Has LaTeX command \sin, so should not be pseudo-math
      expect(result.issues.some((issue) => issue.includes('pseudo-math'))).toBe(false);
    });

    it('should handle multiple \\colorbox commands', () => {
      const multiple = '\\colorbox{red}{a} + \\colorbox{blue}{b}';
      const result = fixMathBlock(multiple);

      // Should remain valid
      expect(result.confidence).toBe('high');
    });

    it('should fix \\colorbox with missing second brace', () => {
      const broken = '\\colorbox{red}{content without close';
      const result = fixMathBlock(broken);

      expect(result.changed).toBe(true);
    });

    it('should handle mathematical symbols (Greek letters, operators)', () => {
      const symbols = '∑ α β γ ∫ ∂ ∇ ≤ ≥';
      const result = fixMathBlock(symbols);

      // Has math symbols, not pseudo-math
      expect(result.issues.some((issue) => issue.includes('pseudo-math'))).toBe(false);
    });

    it('should detect too many extra closing braces as low confidence', () => {
      const extraClosing = 'x = y}}}';
      const result = fixMathBlock(extraClosing);

      // Can't fix extra closing braces safely
      expect(result.confidence).toBe('low');
    });
  });

  describe('degradeToTexBlock', () => {
    it('should wrap content in tex code block with note', () => {
      const content = 'broken math content';
      const reason = 'unbalanced brackets';
      const result = degradeToTexBlock(content, reason);

      expect(result).toContain('```tex');
      expect(result).toContain('broken math content');
      expect(result).toContain(reason);
      expect(result).toContain('*Note:');
    });

    it('should trim content before wrapping', () => {
      const content = '\n\n  broken math  \n\n';
      const result = degradeToTexBlock(content, 'test');

      expect(result).toContain('```tex\nbroken math\n```');
    });
  });
});

describe('Math Fixer Integration', () => {
  it('should handle the complete problem example from issue', () => {
    // The exact example from the issue
    const problematicMath = `
\\sum_{l\\in[0,L+1)}\\exp(s_l-m_{[0,L+1)}) = \\colorbox{red}{
$$

\\displaystyle\\exp(m_{[0,L)}-m_{[0,L+1)})$}\\colorbox{orange}{$\\displaystyle\\sum_{l\\in[0,L)}\\exp(s_l-m_{[0,L)})$} + \\colorbox{lime}{$\\displaystyle\\exp(s_L-m_{[0,L+1)})$}$
`;

    const result = fixMathBlock(problematicMath);

    // Should fix multiple issues
    expect(result.changed).toBe(true);
    expect(result.issues.length).toBeGreaterThan(0);

    // Should remove stray $$
    expect(result.fixed).not.toContain('$$');

    // The issues should include multiple fixes
    expect(result.issues.some((issue) => issue.includes('$$'))).toBe(true);
  });
});
