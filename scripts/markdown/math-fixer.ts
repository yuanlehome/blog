/**
 * Math Fixer Module
 *
 * Fixes common LaTeX/math syntax issues in imported markdown using LLM-based cleaning.
 *
 * Strategy (NEW - LLM-based):
 * 1. LLM-based math fixing (removes inline $ in block math, fixes \colorbox, etc.)
 * 2. Enhanced validation after fixing (strict $ checking)
 * 3. Safe fallback: convert unfixable math to tex code blocks
 *
 * Key Fix Target: Inline math `$` mixed inside block math, intertwined with
 * `\colorbox{...}{...}` / `\displaystyle` commands.
 */

import { getConfiguredMathFixer, type LLMMathFixer } from './llm-math-fixer.js';

export interface MathFixResult {
  fixed: string;
  changed: boolean;
  issues: string[];
  confidence: 'high' | 'low';
  degraded: boolean; // true if converted to code block
}

// Singleton LLM fixer instance
let llmFixerInstance: LLMMathFixer | null = null;

/**
 * Get or create LLM math fixer instance
 */
function getLLMFixer(): LLMMathFixer {
  if (!llmFixerInstance) {
    llmFixerInstance = getConfiguredMathFixer();
  }
  return llmFixerInstance;
}

/**
 * Fix a block math node (content between $$ ... $$)
 * Uses LLM-based fixing as primary strategy
 */
export async function fixMathBlock(raw: string): Promise<MathFixResult> {
  const result: MathFixResult = {
    fixed: raw,
    changed: false,
    issues: [],
    confidence: 'high',
    degraded: false,
  };

  let content = raw;

  // 1. Normalize invisible characters (quick pre-processing)
  content = normalizeInvisibleChars(content);
  if (content !== raw) {
    result.changed = true;
    result.issues.push('Normalized invisible characters');
  }

  // 2. Check if this is pseudo-math (plain text wrapped in $$)
  if (isPseudoMath(content)) {
    result.issues.push('Detected pseudo-math block (plain text in $$)');
    result.confidence = 'low';
    result.fixed = content;
    return result;
  }

  // 3. LLM-based fixing (primary strategy)
  try {
    const llmFixer = getLLMFixer();
    const llmResult = await llmFixer.fixMathBlock(content);

    if (llmResult.fixed !== content) {
      content = llmResult.fixed;
      result.changed = true;

      // Only add notes if there are meaningful notes (not empty)
      if (llmResult.notes.length > 0) {
        result.issues.push(`LLM fixed: ${llmResult.notes.join(', ')}`);
      }

      // Map LLM confidence to our confidence
      if (llmResult.confidence === 'low') {
        result.confidence = 'low';
      }
    } else if (llmResult.notes.length > 0) {
      // LLM tried to fix but resulted in same content - unusual
      // Still record notes but don't mark as changed
      result.issues.push(`LLM notes: ${llmResult.notes.join(', ')}`);
    }
  } catch (error) {
    // LLM fixing failed, log and continue with validation
    result.issues.push(`LLM fixing failed: ${error}`);
    result.confidence = 'low';
  }

  // 4. Validate the result with enhanced rules
  const validation = validateMathBlock(content);
  if (!validation.valid) {
    result.confidence = 'low';
    result.issues.push(...validation.errors);
  }

  result.fixed = content;
  return result;
}

/**
 * Normalize invisible and unusual whitespace characters
 */
function normalizeInvisibleChars(text: string): string {
  const replacements: Record<string, string> = {
    '\u00a0': ' ', // non-breaking space
    '\u2000': ' ',
    '\u2001': ' ',
    '\u2002': ' ',
    '\u2003': ' ',
    '\u2004': ' ',
    '\u2005': ' ',
    '\u2006': ' ',
    '\u2007': ' ',
    '\u2008': ' ',
    '\u2009': ' ',
    '\u200a': ' ',
    '\u202f': ' ',
    '\u3000': ' ',
    '\u200b': '', // zero width space
    '\u200c': '',
    '\u200d': '',
    '\u2061': '', // function application
    '\ufeff': '', // byte order mark
  };

  let normalized = text;
  for (const [char, replacement] of Object.entries(replacements)) {
    if (normalized.includes(char)) {
      normalized = normalized.split(char).join(replacement);
    }
  }

  return normalized;
}

/**
 * Detect if content is pseudo-math (plain text, not real LaTeX)
 * Heuristics:
 * - Contains long natural language sentences
 * - Lacks typical LaTeX commands or symbols
 * - High ratio of common words vs math operators
 */
function isPseudoMath(content: string): boolean {
  const trimmed = content.trim();

  // Empty or very short content is not pseudo-math
  if (trimmed.length < 20) {
    return false;
  }

  // Check for typical LaTeX commands/environments
  const hasLatexCommands =
    /\\(frac|sum|int|prod|sqrt|begin|end|left|right|displaystyle|text|mathbb|mathbf|cdot|times|alpha|beta|gamma|theta|lambda|mu|sigma|pi|infty|partial|nabla|leq|geq|approx|equiv|neq)/.test(
      trimmed,
    );

  // Check for math symbols
  const hasMathSymbols = /[∑∫∏√∞∂∇≤≥≈≡≠±×÷∈∉⊂⊃∪∩αβγδεθλμσπ]/.test(trimmed);

  // If has LaTeX commands or math symbols, it's real math
  if (hasLatexCommands || hasMathSymbols) {
    return false;
  }

  // Check for long sentences (typical of plain text)
  // Split by period and check for sentences > 80 chars
  const sentences = trimmed.split(/[.。!！?？]/);
  const longSentences = sentences.filter((s) => s.trim().length > 80);

  // If has multiple long sentences without math, it's pseudo-math
  if (longSentences.length >= 2) {
    return true;
  }

  // Check for high ratio of Chinese characters (if content is mostly Chinese prose)
  const chineseChars = (trimmed.match(/[\u4e00-\u9fa5]/g) || []).length;
  const totalChars = trimmed.length;
  if (totalChars > 100 && chineseChars / totalChars > 0.5) {
    // If it's mostly Chinese and no math symbols, likely pseudo-math
    return true;
  }

  return false;
}

/**
 * Validate that fixed math block is potentially valid
 * Enhanced validation with strict $ checking
 */
function validateMathBlock(content: string): { valid: boolean; errors: string[] } {
  const errors: string[] = [];

  // Check 1: No stray $$ delimiters (STRICT - block math cannot contain $$)
  if (content.includes('$$')) {
    errors.push('Contains stray $$ delimiter');
  }

  // Check 2: Count unescaped single $ (STRICT - should not exist in block math at all)
  let dollarCount = 0;
  for (let i = 0; i < content.length; i++) {
    if (content[i] === '$' && (i === 0 || content[i - 1] !== '\\')) {
      dollarCount++;
    }
  }
  if (dollarCount > 0) {
    errors.push(`Contains ${dollarCount} unescaped $ delimiter(s) - not allowed in block math`);
  }

  // Check 3: No LaTeX display delimiters \[ \] (these are alternative block math delimiters)
  if (content.includes('\\[') || content.includes('\\]')) {
    errors.push('Contains \\[ or \\] delimiters - not allowed in block math');
  }

  // Check 4: No HTML tags (common corruption from imports)
  const htmlTagPattern = /<(img|figure|div|span|p|br|a)\b[^>]*>/i;
  if (htmlTagPattern.test(content)) {
    errors.push('Contains HTML tags - not valid LaTeX');
  }

  // Check 5: Bracket balance
  const brackets = { '{': 0, '[': 0, '(': 0 };
  for (let i = 0; i < content.length; i++) {
    const char = content[i];
    if (i > 0 && content[i - 1] === '\\') continue;

    if (char === '{') brackets['{']++;
    else if (char === '}') brackets['{']--;
    else if (char === '[') brackets['[']++;
    else if (char === ']') brackets['[']--;
    else if (char === '(') brackets['(']++;
    else if (char === ')') brackets['(']--;
  }

  if (brackets['{'] !== 0) {
    errors.push(`Unbalanced braces: ${brackets['{']}`);
  }
  if (brackets['['] !== 0) {
    errors.push(`Unbalanced square brackets: ${brackets['[']}`);
  }
  if (brackets['('] !== 0) {
    errors.push(`Unbalanced parentheses: ${brackets['(']}`);
  }

  // Check 6: No truncated commands (backslash at end without command)
  if (content.trim().endsWith('\\') && !content.trim().endsWith('\\\\')) {
    errors.push('Truncated command at end');
  }

  return {
    valid: errors.length === 0,
    errors,
  };
}

/**
 * Degrade unfixable math to safe tex code block
 */
export function degradeToTexBlock(content: string, reason: string): string {
  return `\`\`\`tex\n${content.trim()}\n\`\`\`\n\n*Note: Math block could not be automatically fixed (${reason}). Showing as code.*`;
}
