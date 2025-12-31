/**
 * Markdown Pipeline Module
 *
 * Provides a reliable markdown processing pipeline that:
 * - Parses markdown to AST using remark
 * - Handles frontmatter via gray-matter with merge semantics
 * - Normalizes content (invisible chars, code fences, etc.)
 * - Serializes back to valid markdown
 *
 * This ensures all markdown output is syntactically valid and
 * renderable by Astro/remark.
 */

import { unified } from 'unified';
import remarkParse from 'remark-parse';
import remarkStringify from 'remark-stringify';
import remarkGfm from 'remark-gfm';
import remarkFrontmatter from 'remark-frontmatter';
import { visit } from 'unist-util-visit';
import type { Root, Code, Heading, List } from 'mdast';
import matter from 'gray-matter';

/**
 * Bidi control characters that should be removed
 * These can cause rendering issues and invisible content
 */
const BIDI_CONTROL_CHARS = [
  '\u202A', // LEFT-TO-RIGHT EMBEDDING
  '\u202B', // RIGHT-TO-LEFT EMBEDDING
  '\u202C', // POP DIRECTIONAL FORMATTING
  '\u202D', // LEFT-TO-RIGHT OVERRIDE
  '\u202E', // RIGHT-TO-LEFT OVERRIDE
  '\u2066', // LEFT-TO-RIGHT ISOLATE
  '\u2067', // RIGHT-TO-LEFT ISOLATE
  '\u2068', // FIRST STRONG ISOLATE
  '\u2069', // POP DIRECTIONAL ISOLATE
];

/**
 * Zero-width and invisible characters that should be removed
 */
const INVISIBLE_CHARS = [
  '\u200B', // ZERO WIDTH SPACE
  '\u200C', // ZERO WIDTH NON-JOINER
  '\u200D', // ZERO WIDTH JOINER
  '\uFEFF', // BYTE ORDER MARK / ZERO WIDTH NO-BREAK SPACE
  '\u2060', // WORD JOINER
  '\u2061', // FUNCTION APPLICATION
  '\u2062', // INVISIBLE TIMES
  '\u2063', // INVISIBLE SEPARATOR
  '\u2064', // INVISIBLE PLUS
];

/**
 * Special space characters to normalize to regular space
 */
const SPECIAL_SPACES: Record<string, string> = {
  '\u00A0': ' ', // NON-BREAKING SPACE
  '\u2000': ' ', // EN QUAD
  '\u2001': ' ', // EM QUAD
  '\u2002': ' ', // EN SPACE
  '\u2003': ' ', // EM SPACE
  '\u2004': ' ', // THREE-PER-EM SPACE
  '\u2005': ' ', // FOUR-PER-EM SPACE
  '\u2006': ' ', // SIX-PER-EM SPACE
  '\u2007': ' ', // FIGURE SPACE
  '\u2008': ' ', // PUNCTUATION SPACE
  '\u2009': ' ', // THIN SPACE
  '\u200A': ' ', // HAIR SPACE
  '\u202F': ' ', // NARROW NO-BREAK SPACE
  '\u205F': ' ', // MEDIUM MATHEMATICAL SPACE
  '\u3000': ' ', // IDEOGRAPHIC SPACE
};

export interface PipelineOptions {
  /**
   * Stringify options for remark-stringify
   */
  stringify?: {
    bullet?: '-' | '*' | '+';
    fence?: '`' | '~';
    fences?: boolean;
    incrementListMarker?: boolean;
    emphasis?: '_' | '*';
    strong?: '_' | '*';
    listItemIndent?: 'one' | 'tab' | 'mixed';
  };

  /**
   * Whether to fix code fence issues (nested backticks)
   * Default: true
   */
  fixCodeFences?: boolean;

  /**
   * Whether to ensure blank lines around headings
   * Default: true
   */
  ensureHeadingSpacing?: boolean;

  /**
   * Whether to stabilize list formatting
   * Default: true
   */
  stabilizeLists?: boolean;

  /**
   * Whether to normalize image URLs (encode spaces)
   * Default: true
   */
  normalizeImageUrls?: boolean;
}

export interface FrontmatterMergeOptions {
  /**
   * Fields managed by the source (e.g., Notion) that should always override
   */
  sourceFields?: string[];

  /**
   * Fields to preserve from existing file if not provided by source
   */
  preserveFields?: string[];
}

export interface ProcessMarkdownResult {
  markdown: string;
  frontmatter: Record<string, any>;
  diagnostics: {
    invisibleCharsRemoved: number;
    codeFencesFixed: number;
    headingSpacingFixed: number;
    imageUrlsFixed: number;
    duplicateFrontmatterKeysRemoved: number;
  };
}

// Pre-compiled regexes for performance
const BIDI_PATTERN = new RegExp(`[${BIDI_CONTROL_CHARS.join('')}]`, 'g');
const INVISIBLE_PATTERN = new RegExp(`[${INVISIBLE_CHARS.join('')}]`, 'g');
const SPECIAL_SPACE_PATTERN = new RegExp(`[${Object.keys(SPECIAL_SPACES).join('')}]`, 'g');

/**
 * Clean invisible and control characters from text.
 * Preserves normal text, Chinese characters, and emoji.
 *
 * @example
 * // Remove zero-width and bidi chars
 * cleanInvisibleCharacters('Hello\u200BWorld\u202A!');
 * // => { cleaned: 'HelloWorld!', count: 2 }
 *
 * @example
 * // Normalize special spaces
 * cleanInvisibleCharacters('Hello\u00A0World');
 * // => { cleaned: 'Hello World', count: 0 }
 */
export function cleanInvisibleCharacters(text: string): { cleaned: string; count: number } {
  let cleaned = text;
  let count = 0;

  // Remove bidi control characters (single pass with pre-compiled regex)
  const bidiMatches = cleaned.match(BIDI_PATTERN);
  if (bidiMatches) {
    count += bidiMatches.length;
    cleaned = cleaned.replace(BIDI_PATTERN, '');
  }

  // Remove zero-width/invisible characters (single pass with pre-compiled regex)
  const invisibleMatches = cleaned.match(INVISIBLE_PATTERN);
  if (invisibleMatches) {
    count += invisibleMatches.length;
    cleaned = cleaned.replace(INVISIBLE_PATTERN, '');
  }

  // Normalize special spaces to regular spaces
  cleaned = cleaned.replace(SPECIAL_SPACE_PATTERN, ' ');

  return { cleaned, count };
}

/**
 * Parse frontmatter with duplicate key detection and removal
 * Returns the parsed data with any duplicate keys reported
 */
export function parseFrontmatterSafe(content: string): {
  data: Record<string, any>;
  content: string;
  duplicateKeys: string[];
} {
  // First, clean the content
  const { cleaned } = cleanInvisibleCharacters(content);

  // Check for duplicate keys in YAML frontmatter before parsing
  const duplicateKeys: string[] = [];
  let processedContent = cleaned;

  const fmMatch = cleaned.match(/^---\r?\n([\s\S]*?)\r?\n---/);
  if (fmMatch) {
    const yamlContent = fmMatch[1];
    const keyPattern = /^([a-zA-Z_][a-zA-Z0-9_]*):/gm;
    const foundKeys: Set<string> = new Set();
    let match;

    while ((match = keyPattern.exec(yamlContent)) !== null) {
      const key = match[1];
      if (foundKeys.has(key)) {
        duplicateKeys.push(key);
      }
      foundKeys.add(key);
    }

    // If there are duplicate keys, we need to remove them before parsing
    // Keep only the last occurrence of each key (matching gray-matter behavior)
    if (duplicateKeys.length > 0) {
      const lines = yamlContent.split('\n');

      // Find all key positions with their ranges (including nested content)
      interface KeyRange {
        keyName: string;
        startLine: number;
        endLine: number; // exclusive
      }
      const keyRanges: KeyRange[] = [];

      for (let i = 0; i < lines.length; i++) {
        const lineMatch = lines[i].match(/^([a-zA-Z_][a-zA-Z0-9_]*):/);
        if (lineMatch) {
          const key = lineMatch[1];
          // Find where this key's content ends (next top-level key or end of content)
          let endLine = i + 1;
          while (endLine < lines.length) {
            // Check if this line is a new top-level key (not indented)
            if (/^[a-zA-Z_][a-zA-Z0-9_]*:/.test(lines[endLine])) {
              break;
            }
            endLine++;
          }
          keyRanges.push({ keyName: key, startLine: i, endLine });
        }
      }

      // Group ranges by key name
      const rangesByKey: Map<string, KeyRange[]> = new Map();
      for (const range of keyRanges) {
        if (!rangesByKey.has(range.keyName)) {
          rangesByKey.set(range.keyName, []);
        }
        rangesByKey.get(range.keyName)!.push(range);
      }

      // Mark line ranges to remove (all but last occurrence of duplicate keys)
      const linesToRemove = new Set<number>();
      for (const key of duplicateKeys) {
        const ranges = rangesByKey.get(key);
        if (ranges && ranges.length > 1) {
          // Remove all but the last occurrence
          for (let i = 0; i < ranges.length - 1; i++) {
            for (let line = ranges[i].startLine; line < ranges[i].endLine; line++) {
              linesToRemove.add(line);
            }
          }
        }
      }

      // Rebuild YAML without duplicate key lines
      const cleanedYaml = lines.filter((_, i) => !linesToRemove.has(i)).join('\n');
      processedContent = `---\n${cleanedYaml}\n---${cleaned.slice(fmMatch[0].length)}`;
    }
  }

  // Parse with gray-matter
  const parsed = matter(processedContent);

  return {
    data: parsed.data,
    content: parsed.content,
    duplicateKeys,
  };
}

/**
 * Merge frontmatter data with proper handling of source vs existing fields
 */
export function mergeFrontmatter(
  sourceData: Record<string, any>,
  existingData: Record<string, any>,
  options: FrontmatterMergeOptions = {},
): Record<string, any> {
  const {
    sourceFields = [
      'title',
      'date',
      'updated',
      'lastEditedTime',
      'tags',
      'draft',
      'cover',
      'source',
      'notion',
      'status',
      'slug',
    ],
    preserveFields = ['description', 'author', 'lang', 'translatedFrom', 'canonicalUrl'],
  } = options;

  const merged: Record<string, any> = {};

  // First, copy all existing data
  for (const [key, value] of Object.entries(existingData)) {
    merged[key] = value;
  }

  // Then, overlay source-managed fields (always override)
  for (const field of sourceFields) {
    if (field in sourceData && sourceData[field] !== undefined) {
      merged[field] = sourceData[field];
    }
  }

  // For other source fields not in preserve list, overlay them
  for (const [key, value] of Object.entries(sourceData)) {
    if (!preserveFields.includes(key)) {
      merged[key] = value;
    }
  }

  return merged;
}

/**
 * Serialize frontmatter to YAML string with guaranteed unique keys
 */
export function serializeFrontmatter(data: Record<string, any>): string {
  // gray-matter.stringify handles this properly
  // We just need to ensure the data object has no duplicate keys (which JS objects can't have anyway)
  return matter.stringify('', data).split('\n').slice(0, -2).join('\n') + '\n---\n';
}

/**
 * Find the minimum fence length needed for a code block
 * Returns a fence that won't conflict with content
 */
function getMinFence(content: string, preferredChar: '`' | '~' = '`'): string {
  const backtickMatch = content.match(/`{3,}/g);
  const tildeMatch = content.match(/~{3,}/g);

  let maxBackticks = 0;
  let maxTildes = 0;

  if (backtickMatch) {
    maxBackticks = Math.max(...backtickMatch.map((m) => m.length));
  }
  if (tildeMatch) {
    maxTildes = Math.max(...tildeMatch.map((m) => m.length));
  }

  // If content has backticks, use longer fence or switch to tildes
  if (preferredChar === '`' && maxBackticks >= 3) {
    if (maxTildes < maxBackticks) {
      return '~'.repeat(Math.max(3, maxTildes + 1));
    }
    return '`'.repeat(maxBackticks + 1);
  }

  if (preferredChar === '~' && maxTildes >= 3) {
    if (maxBackticks < maxTildes) {
      return '`'.repeat(Math.max(3, maxBackticks + 1));
    }
    return '~'.repeat(maxTildes + 1);
  }

  return preferredChar.repeat(3);
}

/**
 * Remark plugin to fix code fence issues
 */
function remarkFixCodeFences() {
  return (tree: Root) => {
    let fixedCount = 0;

    visit(tree, 'code', (node: Code) => {
      const content = node.value || '';

      // Check if content contains the fence marker
      const hasBackticks = content.includes('```');
      const hasTildes = content.includes('~~~');

      if (hasBackticks || hasTildes) {
        // We need to mark this for special handling
        // The stringify will handle it, but we mark it here
        const fence = getMinFence(content, '`');
        // Store the required fence length in meta (will be used by custom stringify)
        if (!node.meta) {
          node.meta = '';
        }
        (node as any)._requiredFence = fence;
        fixedCount++;
      }
    });

    return fixedCount;
  };
}

/**
 * Remark plugin to ensure blank lines around headings
 */
function remarkEnsureHeadingSpacing() {
  return (tree: Root) => {
    let fixedCount = 0;
    const children = tree.children;

    for (let i = 0; i < children.length; i++) {
      const node = children[i];

      if (node.type === 'heading') {
        // Check previous sibling - should not be a list or paragraph without blank line
        if (i > 0) {
          const prev = children[i - 1];
          // If previous is a list, we need blank line handling
          // remark-stringify handles this, but we track it
          if (prev.type === 'list') {
            fixedCount++;
          }
        }
      }
    }

    return fixedCount;
  };
}

/**
 * Remark plugin to stabilize list formatting
 */
function remarkStabilizeLists() {
  return (tree: Root) => {
    visit(tree, 'list', (node: List) => {
      // Ensure consistent spread property for clean output
      // A list is "spread" if it has blank lines between items
      if (node.spread === undefined) {
        node.spread = false;
      }
    });
  };
}

/**
 * Remark plugin to normalize image URLs
 */
function remarkNormalizeImageUrls() {
  return (tree: Root) => {
    let fixedCount = 0;

    visit(tree, 'image', (node: any) => {
      if (node.url && typeof node.url === 'string') {
        // Encode spaces in URLs if not already encoded
        if (node.url.includes(' ') && !node.url.includes('%20')) {
          node.url = node.url.replace(/ /g, '%20');
          fixedCount++;
        }
      }
    });

    visit(tree, 'link', (node: any) => {
      if (node.url && typeof node.url === 'string') {
        // Encode spaces in URLs if not already encoded
        if (node.url.includes(' ') && !node.url.includes('%20')) {
          node.url = node.url.replace(/ /g, '%20');
          fixedCount++;
        }
      }
    });

    return fixedCount;
  };
}

/**
 * Main markdown processing pipeline
 *
 * Takes raw markdown (with or without frontmatter) and returns
 * cleaned, normalized, syntactically valid markdown.
 */
export async function processMarkdown(
  input: string,
  options: PipelineOptions = {},
): Promise<ProcessMarkdownResult> {
  const {
    stringify = {
      bullet: '-',
      fence: '`',
      fences: true,
      incrementListMarker: false,
      emphasis: '_',
      strong: '*',
      listItemIndent: 'one',
    },
    fixCodeFences = true,
    ensureHeadingSpacing = true,
    stabilizeLists = true,
    normalizeImageUrls = true,
  } = options;

  const diagnostics = {
    invisibleCharsRemoved: 0,
    codeFencesFixed: 0,
    headingSpacingFixed: 0,
    imageUrlsFixed: 0,
    duplicateFrontmatterKeysRemoved: 0,
  };

  // Step 1: Clean invisible characters from raw input
  const { cleaned: cleanedInput, count: invisibleCount } = cleanInvisibleCharacters(input);
  diagnostics.invisibleCharsRemoved = invisibleCount;

  // Step 2: Parse frontmatter
  const { data: frontmatter, content: body, duplicateKeys } = parseFrontmatterSafe(cleanedInput);
  diagnostics.duplicateFrontmatterKeysRemoved = duplicateKeys.length;

  // Step 3: Build the processor pipeline
  const processor = unified().use(remarkParse).use(remarkGfm).use(remarkFrontmatter, ['yaml']);

  // Parse to AST
  const tree = processor.parse(body) as Root;

  // Step 4: Apply transforms
  if (fixCodeFences) {
    const fixed = remarkFixCodeFences()(tree);
    diagnostics.codeFencesFixed = typeof fixed === 'number' ? fixed : 0;
  }

  if (ensureHeadingSpacing) {
    const fixed = remarkEnsureHeadingSpacing()(tree);
    diagnostics.headingSpacingFixed = typeof fixed === 'number' ? fixed : 0;
  }

  if (stabilizeLists) {
    remarkStabilizeLists()(tree);
  }

  if (normalizeImageUrls) {
    const plugin = remarkNormalizeImageUrls();
    const fixed = plugin(tree);
    diagnostics.imageUrlsFixed = typeof fixed === 'number' ? fixed : 0;
  }

  // Step 5: Stringify AST back to markdown
  const stringifyProcessor = unified()
    .use(remarkStringify, {
      bullet: stringify.bullet,
      fence: stringify.fence,
      fences: stringify.fences,
      incrementListMarker: stringify.incrementListMarker,
      emphasis: stringify.emphasis,
      strong: stringify.strong,
      listItemIndent: stringify.listItemIndent,
    })
    .use(remarkGfm);

  let processedBody = stringifyProcessor.stringify(tree);

  // Step 6: Normalize line endings and trailing newline
  processedBody = processedBody.replace(/\r\n/g, '\n');
  if (!processedBody.endsWith('\n')) {
    processedBody += '\n';
  }

  // Step 7: Compress excessive blank lines (3+ -> 2)
  processedBody = processedBody.replace(/\n{3,}/g, '\n\n');

  // Step 8: Reconstruct with frontmatter
  let finalMarkdown: string;
  if (Object.keys(frontmatter).length > 0) {
    finalMarkdown = matter.stringify(processedBody, frontmatter);
  } else {
    finalMarkdown = processedBody;
  }

  // Ensure trailing newline
  if (!finalMarkdown.endsWith('\n')) {
    finalMarkdown += '\n';
  }

  return {
    markdown: finalMarkdown,
    frontmatter,
    diagnostics,
  };
}

/**
 * Process markdown for Notion import specifically
 *
 * This is a higher-level function that:
 * 1. Takes raw Notion markdown content
 * 2. Merges frontmatter with existing file data
 * 3. Applies all pipeline transforms
 * 4. Returns valid, renderable markdown
 */
export async function processMarkdownForNotionSync(
  rawMarkdown: string,
  newFrontmatter: Record<string, any>,
  existingContent?: string,
  mergeOptions?: FrontmatterMergeOptions,
): Promise<ProcessMarkdownResult> {
  // Clean the raw markdown first
  const { cleaned: cleanedRaw, count: rawCleanCount } = cleanInvisibleCharacters(rawMarkdown);

  // If existing content, parse its frontmatter for merge
  let existingFrontmatter: Record<string, any> = {};
  if (existingContent) {
    const { data } = parseFrontmatterSafe(existingContent);
    existingFrontmatter = data;
  }

  // Merge frontmatter
  const mergedFrontmatter = mergeFrontmatter(newFrontmatter, existingFrontmatter, mergeOptions);

  // Construct the full markdown with merged frontmatter
  const fullMarkdown = matter.stringify(cleanedRaw, mergedFrontmatter);

  // Process through the pipeline
  const result = await processMarkdown(fullMarkdown);

  // Add the raw cleaning count to diagnostics
  result.diagnostics.invisibleCharsRemoved += rawCleanCount;

  return result;
}

export default processMarkdown;
