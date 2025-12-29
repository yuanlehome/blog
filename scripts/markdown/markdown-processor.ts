/**
 * Markdown Processor Module
 *
 * Core processing pipeline for markdown enhancement:
 * - Language detection and translation
 * - Code fence language inference
 * - Image caption normalization
 * - Markdown syntax cleanup
 */

import { unified } from 'unified';
import remarkParse from 'remark-parse';
import remarkStringify from 'remark-stringify';
import remarkGfm from 'remark-gfm';
import { visit } from 'unist-util-visit';
import type { Root, Code, Paragraph, Image } from 'mdast';
import matter from 'gray-matter';

import { detectLanguage, shouldTranslate } from './language-detector.js';
import { type Translator, type TranslationNode, getConfiguredTranslator } from './translator.js';
import { detectCodeLanguage, isGitHubActionsWorkflow } from './code-fence-fixer.js';

export interface ProcessingOptions {
  slug?: string;
  source?: string;
  translator?: Translator;
  enableTranslation?: boolean;
  enableCodeFenceFix?: boolean;
  enableImageCaptionFix?: boolean;
  enableMarkdownCleanup?: boolean;
  enableMathDelimiterFix?: boolean;
}

export interface ProcessingDiagnostics {
  changed: boolean;
  translated: boolean;
  detectedLanguage?: string;
  codeFencesFixed: number;
  imageCaptionsFixed: number;
  emptyLinesCompressed: number;
  mathDelimitersFixed: number;
  frontmatterUpdated: boolean;
  translationProvider?: string;
  translationModel?: string;
  translationBatches?: number;
  translationSuccessBatches?: number;
  translationFailedBatches?: number;
  translationCacheHits?: number;
}

export interface ProcessingResult {
  markdown: string;
  diagnostics: ProcessingDiagnostics;
}

/**
 * Generate stable node ID for translation tracking
 */
function generateNodeId(node: any, index: number): string {
  return `node_${index}_${node.type}`;
}

/**
 * Extract translatable nodes from AST
 */
function extractTranslatableNodes(tree: Root): TranslationNode[] {
  const nodes: TranslationNode[] = [];
  let nodeIndex = 0;

  visit(tree, (node: any, _index: number | undefined, parent: any) => {
    // Skip code nodes
    if (node.type === 'code' || node.type === 'inlineCode') {
      return;
    }

    // Extract text from text nodes
    if (node.type === 'text' && node.value && node.value.trim()) {
      const nodeId = generateNodeId(node, nodeIndex++);
      nodes.push({
        nodeId,
        text: node.value,
        context: parent?.type,
      });
      // Store nodeId on node for later patching
      (node as any)._nodeId = nodeId;
    }

    // Extract heading text
    if (node.type === 'heading') {
      const text = extractTextFromNode(node);
      if (text.trim()) {
        const nodeId = generateNodeId(node, nodeIndex++);
        nodes.push({
          nodeId,
          text,
          context: 'heading',
        });
        (node as any)._nodeId = nodeId;
      }
    }

    // Extract image alt text for translation
    if (node.type === 'image' && node.alt && node.alt.trim()) {
      const nodeId = generateNodeId(node, nodeIndex++);
      nodes.push({
        nodeId,
        text: node.alt,
        context: 'image-alt',
      });
      (node as any)._altNodeId = nodeId;
    }
  });

  return nodes;
}

/**
 * Extract text content from a node recursively
 */
function extractTextFromNode(node: any): string {
  if (node.type === 'text') {
    return node.value || '';
  }
  if (node.children) {
    return node.children.map(extractTextFromNode).join('');
  }
  return '';
}

/**
 * Apply translation patches to AST
 */
function applyTranslationPatches(tree: Root, patches: Record<string, string>): void {
  visit(tree, (node: any) => {
    if ((node as any)._nodeId && patches[(node as any)._nodeId]) {
      const translatedText = patches[(node as any)._nodeId];

      if (node.type === 'text') {
        node.value = translatedText;
      } else if (node.type === 'heading' && node.children) {
        // Replace heading text while preserving structure
        node.children = [{ type: 'text', value: translatedText }];
      }
    }

    // Apply translation to image alt text
    if (node.type === 'image' && (node as any)._altNodeId && patches[(node as any)._altNodeId]) {
      node.alt = patches[(node as any)._altNodeId];
    }
  });
}

/**
 * Fix code fence languages
 */
function fixCodeFences(tree: Root): number {
  let fixedCount = 0;

  visit(tree, 'code', (node: Code) => {
    if (!node.lang || node.lang.trim() === '') {
      const detectedLang = isGitHubActionsWorkflow(node.value)
        ? 'yaml'
        : detectCodeLanguage(node.value);

      node.lang = detectedLang;
      fixedCount++;
    }
  });

  return fixedCount;
}

/**
 * Fix image captions by converting to Markdown-compatible format
 *
 * Strategy: Keep Markdown image syntax and add caption as italic text below.
 * This avoids HTML figure tags which may not render correctly in Astro.
 */
function fixImageCaptions(tree: Root): number {
  let fixedCount = 0;
  const nodesToInsert: Array<{
    parent: any;
    index: number;
    newNode: any;
  }> = [];

  visit(tree, (node: any, index: number | undefined, parent: any) => {
    if (
      node.type === 'paragraph' &&
      node.children?.length === 1 &&
      node.children[0].type === 'image'
    ) {
      const image = node.children[0] as Image;
      const nextSibling = parent && typeof index === 'number' && parent.children[index + 1];

      // Check if next sibling is a potential caption
      if (nextSibling && nextSibling.type === 'paragraph' && isCaptionParagraph(nextSibling)) {
        const caption = extractTextFromNode(nextSibling);

        // Convert the caption paragraph to italic emphasis
        const captionNode = {
          type: 'paragraph',
          children: [
            {
              type: 'emphasis',
              children: [{ type: 'text', value: caption }],
            },
          ],
        };

        // Replace the next sibling with italic caption
        nodesToInsert.push({
          parent,
          index: index! + 1,
          newNode: captionNode,
        });

        fixedCount++;
      }
      // Note: We no longer create figures for standalone images with alt text
      // Alt text is already preserved in the image syntax
    }
  });

  // Apply replacements
  nodesToInsert.reverse().forEach(({ parent, index, newNode }) => {
    parent.children[index] = newNode;
  });

  return fixedCount;
}

/**
 * Check if a paragraph looks like a caption
 *
 * Captions typically:
 * - Start with "Figure", "Fig.", "Table", "Image", etc.
 * - Start with a number followed by colon (e.g., "1: Description")
 * - Are relatively short (max 120 chars)
 * - Don't look like regular prose
 */
function isCaptionParagraph(node: Paragraph): boolean {
  const text = extractTextFromNode(node);

  // Caption should not be too long (max 120 chars)
  if (text.length > 120) {
    return false;
  }

  // Caption should not start with markdown syntax
  if (/^[#\-*>`]/.test(text)) {
    return false;
  }

  // Should have some content
  if (text.trim().length === 0) {
    return false;
  }

  // Check for common caption patterns
  const captionPatterns = [
    /^Figure\s+\d+/i, // "Figure 1", "Figure 2:", etc.
    /^Fig\.\s*\d+/i, // "Fig. 1", "Fig.2:", etc.
    /^Table\s+\d+/i, // "Table 1", "Table 2:", etc.
    /^Image\s+\d+/i, // "Image 1", etc.
    /^图\s*\d+/, // Chinese "图1", "图 1", etc.
    /^表\s*\d+/, // Chinese "表1", "表 1", etc.
    /^\d+[:.：]\s*/, // Starts with number and colon "1: Description"
  ];

  return captionPatterns.some((pattern) => pattern.test(text.trim()));
}

/**
 * Normalize invisible characters from imports (e.g., Notion exports)
 */
const INVISIBLE_REPLACEMENTS: Record<string, string> = {
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
  '\u202f': ' ', // narrow no-break space (common in CJK text)
  '\u3000': ' ', // ideographic space
  '\u200b': '', // zero width space
  '\u200c': '', // zero width non-joiner
  '\u200d': '', // zero width joiner
  // Invisible glyphs that should be removed entirely
  '\u2061': '', // function application
  '\ufeff': '', // byte order mark
};

function normalizeInvisibleCharacters(text: string): string {
  let normalized = text;

  for (const [char, replacement] of Object.entries(INVISIBLE_REPLACEMENTS)) {
    if (normalized.includes(char)) {
      normalized = normalized.split(char).join(replacement);
    }
  }

  return normalized;
}

/**
 * Fix math delimiters in markdown content
 * - Normalize invisible characters
 * - Skip code fences and inline code
 * - Fix math tokens: trim whitespace from inline math, promote multi-line inline math to block math
 */
function fixMathDelimiters(markdown: string): { content: string; changes: number } {
  let content = normalizeInvisibleCharacters(markdown);
  let changes = 0;

  // Process text outside of code blocks and inline code
  const segments = splitIntoSegments(content);

  content = segments
    .map((segment) => {
      if (segment.type !== 'text') return segment.content;

      const inlineSegments = splitInlineCode(segment.content);
      return inlineSegments
        .map((inline) => {
          if (inline.type === 'code') return inline.content;

          const fixed = fixMathTokens(inline.content);
          if (fixed !== inline.content) changes++;
          return fixed;
        })
        .join('');
    })
    .join('');

  return { content, changes };
}

/**
 * Split markdown into segments (frontmatter, code fences, text)
 */
type Segment = { type: 'text' | 'code' | 'frontmatter'; content: string };

function splitIntoSegments(text: string): Segment[] {
  const segments: Segment[] = [];

  // Handle frontmatter
  let body = text;
  if (text.startsWith('---')) {
    const end = text.indexOf('\n---', 3);
    if (end !== -1) {
      const fmEnd = end + '\n---'.length;
      segments.push({ type: 'frontmatter', content: text.slice(0, fmEnd + 1) });
      body = text.slice(fmEnd + 1);
    }
  }

  // Split by code fences
  const lines = body.split('\n');
  let buffer = '';
  let inFence = false;
  let fenceMarker = '';

  const flush = (type: Segment['type']) => {
    if (buffer.length) {
      segments.push({ type, content: buffer });
      buffer = '';
    }
  };

  for (let idx = 0; idx < lines.length; idx++) {
    const line = lines[idx];
    const suffix = idx < lines.length - 1 ? '\n' : '';
    const fenceMatch = line.match(/^\s*(`{3,}|~{3,})/);

    if (fenceMatch) {
      const marker = fenceMatch[1];
      if (!inFence) {
        flush('text');
        inFence = true;
        fenceMarker = marker;
        buffer = line + suffix;
      } else if (line.trim().startsWith(fenceMarker)) {
        buffer += line + suffix;
        flush('code');
        inFence = false;
        fenceMarker = '';
      } else {
        buffer += line + suffix;
      }
      continue;
    }

    buffer += line + suffix;
  }

  flush(inFence ? 'code' : 'text');
  return segments;
}

/**
 * Split text by inline code
 */
type InlineSegment = { type: 'text' | 'code'; content: string };

function splitInlineCode(text: string): InlineSegment[] {
  const segments: InlineSegment[] = [];
  let i = 0;
  let buffer = '';

  const pushText = () => {
    if (buffer) {
      segments.push({ type: 'text', content: buffer });
      buffer = '';
    }
  };

  while (i < text.length) {
    if (text[i] === '`') {
      const ticks = countBackticks(text, i);
      const closeIndex = text.indexOf('`'.repeat(ticks), i + ticks);

      if (closeIndex !== -1) {
        pushText();
        const codeContent = text.slice(i, closeIndex + ticks);
        segments.push({ type: 'code', content: codeContent });
        i = closeIndex + ticks;
        continue;
      }
    }

    buffer += text[i];
    i++;
  }

  if (buffer) {
    segments.push({ type: 'text', content: buffer });
  }

  return segments;
}

function countBackticks(text: string, start: number): number {
  let count = 0;
  while (text[start + count] === '`') {
    count++;
  }
  return count;
}

/**
 * Fix math tokens in text (handle inline and block math)
 */
function fixMathTokens(text: string): string {
  const tokens: {
    type: 'text' | 'inline' | 'block';
    content: string;
    raw: string;
  }[] = [];
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
        tokens.push({
          type: 'block',
          content: blockContent,
          raw: `$$${blockContent}$$`,
        });
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
        if (text[j] === '\\' && text[j + 1] === '$') {
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
        tokens.push({
          type: 'inline',
          content: inlineContent,
          raw: `$${inlineContent}$`,
        });
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

  // Reconstruct with fixes
  return tokens
    .map((token) => {
      if (token.type === 'block') return token.raw;
      if (token.type === 'text') return token.raw;

      // Analyze Inline Math
      const inner = token.content;
      const needsBlock =
        inner.includes('\n') || inner.includes('\\begin{') || inner.includes('\\[');

      // Promote to block math if needed
      if (needsBlock) {
        const cleanInner = inner.trim();
        return `\n$$\n${cleanInner}\n$$\n`;
      }

      // Fix inline spacing: $ x $ -> $x$
      if (inner.startsWith(' ') || inner.endsWith(' ')) {
        return `$${inner.trim()}$`;
      }

      return token.raw;
    })
    .join('');
}

/**
 * Normalize markdown formatting
 */
function normalizeMarkdown(markdown: string): { content: string; changes: number } {
  let content = markdown;
  let changes = 0;

  // Compress 3+ consecutive empty lines to 2
  const before = content;
  content = content.replace(/\n{3,}/g, '\n\n');
  if (content !== before) {
    changes += (before.match(/\n{3,}/g) || []).length;
  }

  // Ensure LF line endings
  content = content.replace(/\r\n/g, '\n');

  return { content, changes };
}

/**
 * Main processing function for markdown import enhancement
 */
export async function processMarkdownForImport(
  input: { markdown: string; slug?: string; source?: string },
  options: ProcessingOptions = {},
): Promise<ProcessingResult> {
  const {
    translator = getConfiguredTranslator(),
    enableTranslation = true,
    enableCodeFenceFix = true,
    enableImageCaptionFix = true,
    enableMarkdownCleanup = true,
    enableMathDelimiterFix = true,
  } = options;

  const diagnostics: ProcessingDiagnostics = {
    changed: false,
    translated: false,
    codeFencesFixed: 0,
    imageCaptionsFixed: 0,
    emptyLinesCompressed: 0,
    mathDelimitersFixed: 0,
    frontmatterUpdated: false,
  };

  // Parse frontmatter
  const { data: frontmatter, content: markdownBody } = matter(input.markdown);

  // Detect language
  const langDetection = detectLanguage(markdownBody);
  diagnostics.detectedLanguage = langDetection.language;

  let processedMarkdown = markdownBody;
  let needsTranslation = false;

  // Check if translation is needed
  if (
    enableTranslation &&
    translator &&
    shouldTranslate(markdownBody) &&
    langDetection.language === 'en'
  ) {
    needsTranslation = true;
  }

  // Parse markdown to AST
  const processor = unified().use(remarkParse).use(remarkGfm);

  const tree = processor.parse(processedMarkdown) as Root;

  // Fix code fences BEFORE translation (so they are properly marked as non-translatable)
  if (enableCodeFenceFix) {
    const fixed = fixCodeFences(tree);
    diagnostics.codeFencesFixed = fixed;
    if (fixed > 0) {
      diagnostics.changed = true;
    }
  }

  // Fix image captions BEFORE translation (convert to markdown-native italic format)
  // This ensures captions are properly structured before translation extracts text
  if (enableImageCaptionFix) {
    const fixed = fixImageCaptions(tree);
    diagnostics.imageCaptionsFixed = fixed;
    if (fixed > 0) {
      diagnostics.changed = true;
    }
  }

  // Apply translation if needed
  if (needsTranslation && translator) {
    try {
      const translatableNodes = extractTranslatableNodes(tree);
      if (translatableNodes.length > 0) {
        const translationResult = await translator.translate(translatableNodes);
        const patchMap: Record<string, string> = {};
        translationResult.patches.forEach((patch) => {
          patchMap[patch.nodeId] = patch.translatedText;
        });

        applyTranslationPatches(tree, patchMap);
        diagnostics.translated = true;
        diagnostics.changed = true;

        // Add translation metadata to diagnostics
        if (translationResult.metadata) {
          diagnostics.translationProvider = translationResult.metadata.provider;
          diagnostics.translationModel = translationResult.metadata.model;
          diagnostics.translationBatches = translationResult.metadata.batches;
          diagnostics.translationSuccessBatches = translationResult.metadata.successBatches;
          diagnostics.translationFailedBatches = translationResult.metadata.failedBatches;
          diagnostics.translationCacheHits = translationResult.metadata.cacheHits;
        }

        // Update frontmatter
        if (!frontmatter.lang || frontmatter.lang === 'en') {
          frontmatter.lang = 'zh';
          frontmatter.translatedFrom = 'en';
          diagnostics.frontmatterUpdated = true;
        }
      }
    } catch (error) {
      console.warn('Translation failed, continuing with other fixes:', error);
    }
  }

  // Convert AST back to markdown
  const stringifyProcessor = unified()
    .use(remarkStringify, {
      bullet: '-',
      fence: '`',
      fences: true,
      incrementListMarker: false,
    })
    .use(remarkGfm);

  processedMarkdown = stringifyProcessor.stringify(tree);

  // Fix math delimiters (must be done after AST conversion)
  if (enableMathDelimiterFix) {
    const { content, changes } = fixMathDelimiters(processedMarkdown);
    processedMarkdown = content;
    diagnostics.mathDelimitersFixed = changes;
    if (changes > 0) {
      diagnostics.changed = true;
    }
  }

  // Normalize markdown formatting
  if (enableMarkdownCleanup) {
    const { content, changes } = normalizeMarkdown(processedMarkdown);
    processedMarkdown = content;
    diagnostics.emptyLinesCompressed = changes;
    if (changes > 0) {
      diagnostics.changed = true;
    }
  }

  // Reconstruct final markdown with frontmatter
  const finalMarkdown = matter.stringify(processedMarkdown, frontmatter);

  return {
    markdown: finalMarkdown,
    diagnostics,
  };
}
