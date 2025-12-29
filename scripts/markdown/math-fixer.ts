/**
 * Math Fixer Module
 *
 * Fixes common LaTeX/math syntax issues in imported markdown:
 * - Removes stray $$ delimiters inside block math
 * - Removes nested inline $ delimiters inside block math
 * - Balances brackets/braces with high-confidence heuristics
 * - Cleans up broken \colorbox and similar commands
 * - Detects and downgrades pseudo-math blocks (plain text in $$)
 *
 * Strategy:
 * 1. Deterministic rule-based fixes (no LLM, high confidence)
 * 2. Validation after fixing
 * 3. Safe fallback: convert unfixable math to tex code blocks
 */

export interface MathFixResult {
  fixed: string;
  changed: boolean;
  issues: string[];
  confidence: 'high' | 'low';
  degraded: boolean; // true if converted to code block
}

/**
 * Fix a block math node (content between $$ ... $$)
 */
export function fixMathBlock(raw: string): MathFixResult {
  const result: MathFixResult = {
    fixed: raw,
    changed: false,
    issues: [],
    confidence: 'high',
    degraded: false,
  };

  let content = raw;

  // 1. Remove invisible/unusual characters
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

  // 3. Remove stray $$ inside block math
  const withoutDoubleDollar = removeStrayDoubleDollar(content);
  if (withoutDoubleDollar !== content) {
    content = withoutDoubleDollar;
    result.changed = true;
    result.issues.push('Removed stray $$ delimiters inside block math');
  }

  // 4. Remove nested inline $ delimiters inside block math
  const withoutInlineDollar = removeNestedInlineDollar(content);
  if (withoutInlineDollar !== content) {
    content = withoutInlineDollar;
    result.changed = true;
    result.issues.push('Removed nested inline $ delimiters');
  }

  // 5. Fix broken \colorbox and similar commands
  const withFixedCommands = fixBrokenLatexCommands(content);
  if (withFixedCommands !== content) {
    content = withFixedCommands;
    result.changed = true;
    result.issues.push('Fixed broken LaTeX commands');
  }

  // 6. Balance brackets with high confidence
  const balanceResult = balanceBrackets(content);
  if (balanceResult.changed) {
    content = balanceResult.fixed;
    result.changed = true;
    result.issues.push('Balanced brackets/braces');
    if (balanceResult.confidence === 'low') {
      result.confidence = 'low';
    }
  }

  // 7. Validate the result
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
 * Remove stray $$ delimiters inside block math
 * Example: "content $$ more content" -> "content more content"
 */
function removeStrayDoubleDollar(content: string): string {
  // Split by $$ and check if there are odd occurrences
  // If we find $$ in the middle, remove them
  let result = content;

  // Simple approach: replace all $$ with empty string
  // This works because we're already inside a block math (delimited by outer $$)
  result = result.replace(/\$\$/g, '');

  return result;
}

/**
 * Remove nested inline $ delimiters inside block math
 * Example: "$\displaystyle\sum_{i}$" -> "\displaystyle\sum_{i}"
 * Strategy: Remove unescaped $ that appear to be inline delimiters
 */
function removeNestedInlineDollar(content: string): string {
  let result = '';
  let i = 0;

  while (i < content.length) {
    const char = content[i];

    // Check for escaped dollar
    if (char === '\\' && i + 1 < content.length && content[i + 1] === '$') {
      result += '\\$';
      i += 2;
      continue;
    }

    // Check for unescaped $
    if (char === '$') {
      // Skip this dollar (don't add to result)
      i++;
      continue;
    }

    result += char;
    i++;
  }

  return result;
}

/**
 * Fix broken LaTeX commands like \colorbox{color}{content}
 * Ensures both pairs of braces are balanced
 */
function fixBrokenLatexCommands(content: string): string {
  let result = content;

  // Pattern: \colorbox{...}{...} or similar commands
  // Find commands that might be broken
  const commandPattern = /\\(colorbox|textcolor|fcolorbox|boxed|fbox)\s*\{/g;

  let match;
  const fixes: Array<{ start: number; end: number; replacement: string }> = [];

  while ((match = commandPattern.exec(content)) !== null) {
    const cmdStart = match.index;
    const firstBraceStart = match.index + match[0].length - 1;

    // Find the end of first brace
    const firstBraceEnd = findMatchingBrace(content, firstBraceStart);
    if (firstBraceEnd === -1) {
      // First brace not closed, skip
      continue;
    }

    // Check if there's a second brace immediately after
    let secondBraceStart = firstBraceEnd + 1;
    // Skip whitespace
    while (secondBraceStart < content.length && /\s/.test(content[secondBraceStart])) {
      secondBraceStart++;
    }

    if (secondBraceStart >= content.length || content[secondBraceStart] !== '{') {
      // No second brace, this command might be incomplete
      continue;
    }

    // Find the end of second brace
    const secondBraceEnd = findMatchingBrace(content, secondBraceStart);
    if (secondBraceEnd === -1) {
      // Second brace not closed, try to close it
      // Find a reasonable place to close (end of line or next command)
      let closePos = content.indexOf('\n', secondBraceStart);
      if (closePos === -1) {
        closePos = content.length;
      }

      // Check if there's another backslash command before the newline
      const nextCmd = content.indexOf('\\', secondBraceStart + 1);
      if (nextCmd !== -1 && nextCmd < closePos) {
        closePos = nextCmd;
      }

      // Insert closing brace
      const fixed = content.slice(cmdStart, closePos) + '}';
      fixes.push({ start: cmdStart, end: closePos, replacement: fixed });
    }
  }

  // Apply fixes in reverse order to maintain indices
  fixes.reverse().forEach((fix) => {
    result = result.slice(0, fix.start) + fix.replacement + result.slice(fix.end);
  });

  return result;
}

/**
 * Find matching closing brace for an opening brace
 * Returns -1 if not found
 */
function findMatchingBrace(text: string, openPos: number): number {
  if (text[openPos] !== '{') {
    return -1;
  }

  let depth = 1;
  let i = openPos + 1;

  while (i < text.length && depth > 0) {
    const char = text[i];

    // Check for escaped braces
    if (i > 0 && text[i - 1] === '\\') {
      i++;
      continue;
    }

    if (char === '{') {
      depth++;
    } else if (char === '}') {
      depth--;
      if (depth === 0) {
        return i;
      }
    }

    i++;
  }

  return -1; // Not found
}

/**
 * Balance brackets and braces with high-confidence heuristics
 */
function balanceBrackets(content: string): {
  fixed: string;
  changed: boolean;
  confidence: 'high' | 'low';
} {
  const result = {
    fixed: content,
    changed: false,
    confidence: 'high' as 'high' | 'low',
  };

  // Count brackets
  const counts = {
    '{': 0,
    '}': 0,
    '[': 0,
    ']': 0,
    '(': 0,
    ')': 0,
  };

  let i = 0;
  while (i < content.length) {
    const char = content[i];

    // Skip escaped characters
    if (char === '\\' && i + 1 < content.length) {
      i += 2;
      continue;
    }

    if (char in counts) {
      counts[char as keyof typeof counts]++;
    }

    i++;
  }

  // Check balance
  let fixed = content;
  let changed = false;

  // Fix braces {}
  const braceDiff = counts['{'] - counts['}'];
  if (braceDiff > 0 && braceDiff <= 3) {
    // Missing closing braces, add them at the end
    fixed += '}'.repeat(braceDiff);
    changed = true;
  } else if (braceDiff < 0) {
    // Extra closing braces, harder to fix safely
    result.confidence = 'low';
  }

  // Fix square brackets []
  const squareDiff = counts['['] - counts[']'];
  if (squareDiff > 0 && squareDiff <= 3) {
    fixed += ']'.repeat(squareDiff);
    changed = true;
  } else if (squareDiff < 0) {
    result.confidence = 'low';
  }

  // Fix parentheses ()
  const parenDiff = counts['('] - counts[')'];
  if (parenDiff > 0 && parenDiff <= 3) {
    fixed += ')'.repeat(parenDiff);
    changed = true;
  } else if (parenDiff < 0) {
    result.confidence = 'low';
  }

  if (changed) {
    result.fixed = fixed;
    result.changed = true;
  }

  return result;
}

/**
 * Validate that fixed math block is potentially valid
 * Lightweight validation (no KaTeX execution)
 */
function validateMathBlock(content: string): { valid: boolean; errors: string[] } {
  const errors: string[] = [];

  // Check 1: No stray $$ or $ delimiters
  if (content.includes('$$')) {
    errors.push('Contains stray $$ delimiter');
  }

  // Count unescaped single $ (should not exist in block math)
  let dollarCount = 0;
  for (let i = 0; i < content.length; i++) {
    if (content[i] === '$' && (i === 0 || content[i - 1] !== '\\')) {
      dollarCount++;
    }
  }
  if (dollarCount > 0) {
    errors.push(`Contains ${dollarCount} unescaped $ delimiter(s)`);
  }

  // Check 2: Bracket balance
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

  // Check 3: No truncated commands (backslash at end without command)
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
