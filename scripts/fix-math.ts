import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

function processFile(filePath: string) {
  const content = fs.readFileSync(filePath, "utf-8");
  const fixed = fixMath(content);
  if (fixed !== content) {
    fs.writeFileSync(filePath, fixed, "utf-8");
    console.log(`✅ Fixed math in ${filePath}`);
  }
}

function processDirectory(dirPath: string) {
  const files = fs.readdirSync(dirPath);
  for (const file of files) {
    const p = path.join(dirPath, file);
    const stat = fs.statSync(p);
    if (stat.isDirectory()) {
      processDirectory(p);
    } else if (file.endsWith(".md") || file.endsWith(".mdx")) {
      processFile(p);
    }
  }
}

// Heuristic:
// 1. Skip code fences, inline code, and YAML frontmatter so math tokens in code samples stay intact.
// 2. Normalize invisible characters from Notion exports.
// 3. In text parts, promote multi-line inline math into block math and trim inline math padding.

const INVISIBLE_REPLACEMENTS: Record<string, string> = {
  // Spaces and special widths
  "\u00a0": " ", // non-breaking space
  "\u2000": " ", // en quad
  "\u2001": " ", // em quad
  "\u2002": " ", // en space
  "\u2003": " ", // em space
  "\u2004": " ", // three-per-em space
  "\u2005": " ", // four-per-em space
  "\u2006": " ", // six-per-em space
  "\u2007": " ", // figure space
  "\u2008": " ", // punctuation space
  "\u2009": " ", // thin space
  "\u200a": " ", // hair space
  "\u202f": " ", // narrow no-break space (common in CJK text)
  "\u3000": " ", // ideographic space
  "\u200b": "", // zero width space
  "\u200c": "", // zero width non-joiner
  "\u200d": "", // zero width joiner
  // Invisible glyphs that should be removed entirely
  "\u2061": "", // function application
  "\ufeff": "", // byte order mark
};

type Segment = { type: "text" | "code" | "frontmatter"; content: string };
type InlineSegment = { type: "text" | "code"; content: string };

function normalizeInvisibleCharacters(text: string): string {
  let normalized = text;

  for (const [char, replacement] of Object.entries(INVISIBLE_REPLACEMENTS)) {
    if (normalized.includes(char)) {
      normalized = normalized.split(char).join(replacement);
    }
  }

  return normalized;
}

function splitFrontmatter(text: string): {
  frontmatter?: string;
  body: string;
} {
  if (!text.startsWith("---")) return { body: text };

  const end = text.indexOf("\n---", 3);
  if (end === -1) return { body: text };

  const fmEnd = end + "\n---".length;
  return {
    frontmatter: text.slice(0, fmEnd + 1),
    body: text.slice(fmEnd + 1),
  };
}

function splitCodeFences(text: string): Segment[] {
  const segments: Segment[] = [];
  const { frontmatter, body } = splitFrontmatter(text);

  if (frontmatter) {
    segments.push({ type: "frontmatter", content: frontmatter });
  }

  const lines = body.split("\n");
  let buffer = "";
  let inFence = false;
  let fenceMarker = "";

  const flush = (type: Segment["type"]) => {
    if (buffer.length) {
      segments.push({ type, content: buffer });
      buffer = "";
    }
  };

  for (let idx = 0; idx < lines.length; idx++) {
    const line = lines[idx];
    const suffix = idx < lines.length - 1 ? "\n" : "";
    const fenceMatch = line.match(/^\s*(`{3,}|~{3,})/);

    if (fenceMatch) {
      const marker = fenceMatch[1];
      if (!inFence) {
        flush("text");
        inFence = true;
        fenceMarker = marker;
        buffer = line + suffix;
      } else if (line.trim().startsWith(fenceMarker)) {
        buffer += line + suffix;
        flush("code");
        inFence = false;
        fenceMarker = "";
      } else {
        buffer += line + suffix;
      }
      continue;
    }

    buffer += line + suffix;
  }

  flush(inFence ? "code" : "text");
  return segments;
}

function splitInlineCode(text: string): InlineSegment[] {
  const segments: InlineSegment[] = [];
  let i = 0;
  let buffer = "";

  const pushText = () => {
    if (buffer) {
      segments.push({ type: "text", content: buffer });
      buffer = "";
    }
  };

  while (i < text.length) {
    if (text[i] === "`") {
      const ticks = countBackticks(text, i);
      const closeIndex = text.indexOf("`".repeat(ticks), i + ticks);

      if (closeIndex !== -1) {
        pushText();
        const codeContent = text.slice(i, closeIndex + ticks);
        segments.push({ type: "code", content: codeContent });
        i = closeIndex + ticks;
        continue;
      }
    }

    buffer += text[i];
    i++;
  }

  if (buffer) {
    segments.push({ type: "text", content: buffer });
  }

  return segments;
}

function countBackticks(text: string, start: number): number {
  let count = 0;
  while (text[start + count] === "`") {
    count++;
  }
  return count;
}

function fixMath(originalText: string): string {
  const text = normalizeInvisibleCharacters(originalText);
  const segments = splitCodeFences(text);

  return segments
    .map((segment) => {
      if (segment.type !== "text") return segment.content;

      const inlineSegments = splitInlineCode(segment.content);
      return inlineSegments
        .map((inline) =>
          inline.type === "text"
            ? fixMathTokens(inline.content)
            : inline.content,
        )
        .join("");
    })
    .join("");
}

function fixMathTokens(text: string): string {
  const tokens: {
    type: "text" | "inline" | "block";
    content: string;
    raw: string;
  }[] = [];
  let buffer = "";
  let i = 0;

  while (i < text.length) {
    const char = text[i];
    const next = text[i + 1];

    // Check for escaped dollar
    if (char === "\\" && next === "$") {
      buffer += "\\$";
      i += 2;
      continue;
    }

    // Check for Block Math $$
    if (char === "$" && next === "$") {
      if (buffer) {
        tokens.push({ type: "text", content: buffer, raw: buffer });
        buffer = "";
      }

      // Find end of block
      let j = i + 2;
      let blockContent = "";
      let closed = false;
      while (j < text.length) {
        if (text[j] === "$" && text[j + 1] === "$") {
          closed = true;
          break;
        }
        blockContent += text[j];
        j++;
      }

      if (closed) {
        tokens.push({
          type: "block",
          content: blockContent,
          raw: `$$${blockContent}$$`,
        });
        i = j + 2;
      } else {
        // Unclosed, treat as text
        buffer += "$$";
        i += 2;
      }
      continue;
    }

    // Check for Inline Math $
    if (char === "$") {
      if (buffer) {
        tokens.push({ type: "text", content: buffer, raw: buffer });
        buffer = "";
      }

      let j = i + 1;
      let inlineContent = "";
      let closed = false;
      while (j < text.length) {
        if (text[j] === "\\" && text[j + 1] === "$") {
          inlineContent += "\\$";
          j += 2;
          continue;
        }
        if (text[j] === "$") {
          closed = true;
          break;
        }
        inlineContent += text[j];
        j++;
      }

      if (closed) {
        tokens.push({
          type: "inline",
          content: inlineContent,
          raw: `$${inlineContent}$`,
        });
        i = j + 1;
      } else {
        buffer += "$";
        i++;
      }
      continue;
    }

    buffer += char;
    i++;
  }

  if (buffer) {
    tokens.push({ type: "text", content: buffer, raw: buffer });
  }

  // Reconstruct
  return tokens
    .map((token) => {
      if (token.type === "block") return token.raw;
      if (token.type === "text") return token.raw;

      // Analyze Inline Math
      const inner = token.content;
      const needsBlock =
        inner.includes("\n") ||
        inner.includes("\\begin{") ||
        inner.includes("\\[");

      // Trim whitespace from inner for inline math consistency
      // But only if it's NOT a block-like promotion
      if (needsBlock) {
        console.log(
          `Promoting inline math to block:\n${inner.substring(0, 50)}...`,
        );
        const cleanInner = inner.trim();
        return `\n$$\n${cleanInner}\n$$\n`;
      }

      // Fix inline spacing: $ x $ -> $x$
      if (inner.startsWith(" ") || inner.endsWith(" ")) {
        console.log(`Trimming inline math whitespace: "${inner}"`);
        return `$${inner.trim()}$`;
      }

      return token.raw;
    })
    .join("");
}

export function runFixMath(targetPath: string) {
  const fullPath = path.resolve(targetPath);

  if (!fs.existsSync(fullPath)) {
    throw new Error(`Path not found: ${fullPath}`);
  }

  const stat = fs.statSync(fullPath);
  if (stat.isDirectory()) {
    processDirectory(fullPath);
  } else {
    processFile(fullPath);
  }
}

export { normalizeInvisibleCharacters, splitCodeFences, fixMath };

const modulePath = fileURLToPath(import.meta.url);
if (process.argv[1] && path.resolve(process.argv[1]) === modulePath) {
  const targetPath = process.argv[2];
  if (!targetPath) {
    console.error(
      "Usage: npx tsx scripts/fix-math.ts <file-or-directory-path>",
    );
    process.exit(1);
  }

  try {
    runFixMath(targetPath);
  } catch (error) {
    console.error((error as Error).message);
    process.exit(1);
  }
}
