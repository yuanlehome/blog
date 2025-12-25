import fs from 'fs';
import path from 'path';

const targetPath = process.argv[2];

if (!targetPath) {
  console.error('Usage: npx tsx scripts/fix-math.ts <file-or-directory-path>');
  process.exit(1);
}

const fullPath = path.resolve(targetPath);

if (!fs.existsSync(fullPath)) {
  console.error(`Path not found: ${fullPath}`);
  process.exit(1);
}

function processFile(filePath: string) {
    const content = fs.readFileSync(filePath, 'utf-8');
    const fixed = fixMath(content);
    if (fixed !== content) {
        fs.writeFileSync(filePath, fixed, 'utf-8');
        console.log(`✅ Fixed math in ${filePath}`);
    } else {
        // console.log(`✨ No changes needed for ${filePath}`);
    }
}

function processDirectory(dirPath: string) {
    const files = fs.readdirSync(dirPath);
    for (const file of files) {
        const p = path.join(dirPath, file);
        const stat = fs.statSync(p);
        if (stat.isDirectory()) {
            processDirectory(p);
        } else if (file.endsWith('.md') || file.endsWith('.mdx')) {
            processFile(p);
        }
    }
}

// Heuristic:
// 1. Split by `$$` to identify existing block math (odd indices) vs text/inline (even indices).
// 2. In text parts, look for `$` ... `$` pairs.
// 3. If a pair contains `\n` OR `\begin{`, promote it to `$$`.

function normalizeInvisibleCharacters(text: string): string {
  const replacements: Record<string, string> = {
    // Spaces and special widths
    '\u00a0': ' ', // non-breaking space
    '\u2000': ' ', // en quad
    '\u2001': ' ', // em quad
    '\u2002': ' ', // en space
    '\u2003': ' ', // em space
    '\u2004': ' ', // three-per-em space
    '\u2005': ' ', // four-per-em space
    '\u2006': ' ', // six-per-em space
    '\u2007': ' ', // figure space
    '\u2008': ' ', // punctuation space
    '\u2009': ' ', // thin space
    '\u200a': ' ', // hair space
    // Invisible glyphs that should be removed entirely
    '\u2061': '', // function application
  };

  let normalized = text;

  for (const [char, replacement] of Object.entries(replacements)) {
    if (normalized.includes(char)) {
      normalized = normalized.split(char).join(replacement);
    }
  }

  return normalized;
}

function fixMath(originalText: string): string {
  const text = normalizeInvisibleCharacters(originalText);
  const tokens: { type: 'text' | 'inline' | 'block'; content: string; raw: string }[] = [];
  let buffer = '';
  let i = 0;

  while (i < text.length) {
    const char = text[i];
    const next = text[i + 1];

    // Check for escaped dollar
    if (char === '\\' && next === '$') {
      buffer += '\\$';
      i += 2;
      continue;
    }

    // Check for Block Math $$
    if (char === '$' && next === '$') {
      if (buffer) {
        tokens.push({ type: 'text', content: buffer, raw: buffer });
        buffer = '';
      }
      
      // Find end of block
      let j = i + 2;
      let blockContent = '';
      let closed = false;
      while (j < text.length) {
        if (text[j] === '$' && text[j + 1] === '$') {
            closed = true;
            break;
        }
        blockContent += text[j];
        j++;
      }

      if (closed) {
        tokens.push({ type: 'block', content: blockContent, raw: `$$${blockContent}$$` });
        i = j + 2;
      } else {
        // Unclosed, treat as text
        buffer += '$$';
        i += 2;
      }
      continue;
    }

    // Check for Inline Math $
    if (char === '$') {
      if (buffer) {
        tokens.push({ type: 'text', content: buffer, raw: buffer });
        buffer = '';
      }

      let j = i + 1;
      let inlineContent = '';
      let closed = false;
      while (j < text.length) {
        if (text[j] === '\\' && text[j+1] === '$') {
            inlineContent += '\\$';
            j += 2;
            continue;
        }
        if (text[j] === '$') {
            closed = true;
            break;
        }
        inlineContent += text[j];
        j++;
      }

      if (closed) {
        tokens.push({ type: 'inline', content: inlineContent, raw: `$${inlineContent}$` });
        i = j + 1;
      } else {
        buffer += '$';
        i++;
      }
      continue;
    }

    buffer += char;
    i++;
  }
  
  if (buffer) {
      tokens.push({ type: 'text', content: buffer, raw: buffer });
  }

  // Reconstruct
  return tokens.map(token => {
      if (token.type === 'block') return token.raw;
      if (token.type === 'text') return token.raw;
      
      // Analyze Inline Math
      const inner = token.content;
      const needsBlock = inner.includes('\n') || inner.includes('\\begin{') || inner.includes('\\[');

      // Trim whitespace from inner for inline math consistency
      // But only if it's NOT a block-like promotion
      if (needsBlock) {
          console.log(`Promoting inline math to block:\n${inner.substring(0, 50)}...`);
          // Trim whitespace from inner
          const cleanInner = inner.trim();
          return `\n$$\n${cleanInner}\n$$\n`;
      }

      // Fix inline spacing: $ x $ -> $x$
      if (inner.startsWith(' ') || inner.endsWith(' ')) {
          console.log(`Trimming inline math whitespace: "${inner}"`);
          return `$${inner.trim()}$`;
      }
      
      return token.raw;
  }).join('');
}

const stat = fs.statSync(fullPath);
if (stat.isDirectory()) {
    processDirectory(fullPath);
} else {
    processFile(fullPath);
}
