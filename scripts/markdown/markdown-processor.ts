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
import remarkMath from 'remark-math';
import { visit } from 'unist-util-visit';
import type { Root, Code, Paragraph } from 'mdast';
import matter from 'gray-matter';

import { detectLanguage, shouldTranslate } from './language-detector.js';
import {
  type Translator,
  type TranslationNode,
  type TranslationPatch,
  getConfiguredTranslator,
} from './translator.js';
import { detectCodeLanguage, isGitHubActionsWorkflow } from './code-fence-fixer.js';
import type { Logger } from '../logger/types.js';
import { createLogger } from '../logger/index.js';

export interface ProcessingOptions {
  slug?: string;
  source?: string;
  translator?: Translator;
  enableTranslation?: boolean;
  enableCodeFenceFix?: boolean;
  enableImageCaptionFix?: boolean;
  enableMarkdownCleanup?: boolean;
  enableMathDelimiterFix?: boolean;
  logger?: Logger;
}

export interface ProcessingDiagnostics {
  changed: boolean;
  translated: boolean;
  detectedLanguage?: string;
  codeFencesFixed: number;
  imageCaptionsFixed: number;
  emptyLinesCompressed: number;
  mathDelimitersFixed: number;
  mathPatched: number; // Math blocks fixed via translator
  mathFallbackToCodeFence: number; // Math blocks degraded to code
  invalidMathPatchReasons: string[]; // Reasons for fallback
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
 * Includes both text nodes and math nodes
 */
function extractTranslatableNodes(tree: Root): TranslationNode[] {
  const nodes: TranslationNode[] = [];
  let nodeIndex = 0;

  visit(tree, (node: any, _index: number | undefined, parent: any) => {
    // Skip code nodes
    if (node.type === 'code' || node.type === 'inlineCode') {
      return;
    }

    // Extract block math nodes (remark-math)
    if (node.type === 'math' && node.value) {
      const nodeId = generateNodeId(node, nodeIndex++);
      nodes.push({
        kind: 'math',
        nodeId,
        latex: node.value, // Raw LaTeX without $$ delimiters
      });
      // Store nodeId on node for later patching
      (node as any)._nodeId = nodeId;
      return; // Don't process children of math nodes
    }

    // Extract text from text nodes
    if (node.type === 'text' && node.value && node.value.trim()) {
      const nodeId = generateNodeId(node, nodeIndex++);
      nodes.push({
        kind: 'text',
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
          kind: 'text',
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
        kind: 'text',
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
 * Validate a fixed math block
 * Returns true if valid, false if should fallback to code block
 */
function validateMathPatch(latex: string): { valid: boolean; reason?: string } {
  // Check 1: No $ or $$ delimiters allowed
  if (latex.includes('$$')) {
    return { valid: false, reason: 'Contains $$ delimiter' };
  }
  if (latex.includes('$')) {
    return { valid: false, reason: 'Contains $ delimiter' };
  }

  // Check 2: No \[ or \] delimiters
  if (latex.includes('\\[') || latex.includes('\\]')) {
    return { valid: false, reason: 'Contains \\[ or \\] delimiter' };
  }

  // Check 3: No HTML tags
  if (/<[^>]+>/.test(latex)) {
    return { valid: false, reason: 'Contains HTML tags' };
  }

  // Check 4: Basic bracket balance
  const brackets = { '{': 0, '[': 0, '(': 0 };
  for (let i = 0; i < latex.length; i++) {
    const char = latex[i];
    // Skip escaped characters
    if (i > 0 && latex[i - 1] === '\\') continue;

    if (char === '{') brackets['{']++;
    else if (char === '}') brackets['{']--;
    else if (char === '[') brackets['[']++;
    else if (char === ']') brackets['[']--;
    else if (char === '(') brackets['(']++;
    else if (char === ')') brackets['(']--;
  }

  if (brackets['{'] !== 0) {
    return { valid: false, reason: `Unbalanced braces: ${brackets['{']}` };
  }
  if (brackets['['] !== 0) {
    return { valid: false, reason: `Unbalanced square brackets: ${brackets['[']}` };
  }
  if (brackets['('] !== 0) {
    return { valid: false, reason: `Unbalanced parentheses: ${brackets['(']}` };
  }

  return { valid: true };
}

/**
 * Apply translation patches to AST
 * Handles both text and math patches with validation and fallback
 */
function applyTranslationPatches(
  tree: Root,
  patches: TranslationPatch[],
): {
  mathPatched: number;
  mathFallbackToCodeFence: number;
  invalidMathPatchReasons: string[];
} {
  const stats = {
    mathPatched: 0,
    mathFallbackToCodeFence: 0,
    invalidMathPatchReasons: [] as string[],
  };

  // Create a map for quick lookup
  const patchMap = new Map<string, TranslationPatch>();
  for (const patch of patches) {
    patchMap.set(patch.nodeId, patch);
  }

  // Track nodes to replace (for math fallback)
  const nodesToReplace: Array<{ parent: any; index: number; newNode: any }> = [];

  visit(tree, (node: any, index: number | undefined, parent: any) => {
    const nodeId = (node as any)._nodeId;
    const altNodeId = (node as any)._altNodeId;

    // Apply text patches
    if (nodeId && patchMap.has(nodeId)) {
      const patch = patchMap.get(nodeId)!;

      if (patch.kind === 'text') {
        if (node.type === 'text') {
          node.value = patch.text;
        } else if (node.type === 'heading' && node.children) {
          // Replace heading text while preserving structure
          node.children = [{ type: 'text', value: patch.text }];
        }
      } else if (patch.kind === 'math' && node.type === 'math') {
        // Validate math patch
        const validation = validateMathPatch(patch.latex);

        if (validation.valid && patch.confidence === 'high') {
          // Apply the fix
          node.value = patch.latex;
          stats.mathPatched++;
        } else {
          // Fallback: convert to code block
          const reason = validation.reason || `Low confidence (${patch.confidence})`;
          stats.invalidMathPatchReasons.push(reason);
          stats.mathFallbackToCodeFence++;

          // Create code block with original latex
          const codeNode = {
            type: 'code',
            lang: 'tex',
            value: node.value.trim(), // Use original value
          };

          const noteNode = {
            type: 'paragraph',
            children: [
              {
                type: 'emphasis',
                children: [
                  {
                    type: 'text',
                    value: `Note: Math block could not be automatically fixed (${reason}). Showing as code.`,
                  },
                ],
              },
            ],
          };

          if (parent && typeof index === 'number') {
            nodesToReplace.push({
              parent,
              index,
              newNode: [codeNode, noteNode],
            });
          }
        }
      }
    }

    // Apply translation to image alt text
    if (node.type === 'image' && altNodeId && patchMap.has(altNodeId)) {
      const patch = patchMap.get(altNodeId)!;
      if (patch.kind === 'text') {
        node.alt = patch.text;
      }
    }
  });

  // Apply node replacements (in reverse order to maintain indices)
  nodesToReplace.reverse().forEach(({ parent, index, newNode }) => {
    parent.children.splice(index, 1, ...newNode);
  });

  return stats;
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
    logger: parentLogger,
  } = options;

  // Create child logger with context
  const logger =
    parentLogger?.child({
      module: 'markdown',
      slug: input.slug,
      source: input.source,
    }) ?? createLogger({ silent: true });

  const diagnostics: ProcessingDiagnostics = {
    changed: false,
    translated: false,
    codeFencesFixed: 0,
    imageCaptionsFixed: 0,
    emptyLinesCompressed: 0,
    mathDelimitersFixed: 0,
    mathPatched: 0,
    mathFallbackToCodeFence: 0,
    invalidMathPatchReasons: [],
    frontmatterUpdated: false,
  };

  const processingSpan = logger.span({ name: 'markdown-processing', fields: {} });
  processingSpan.start();

  try {
    // Parse frontmatter
    const { data: frontmatter, content: markdownBody } = matter(input.markdown);

    // Detect language
    const langDetection = detectLanguage(markdownBody);
    diagnostics.detectedLanguage = langDetection.language;
    logger.info('Language detected', {
      step: 'language-detection',
      language: langDetection.language,
      confidence: langDetection.confidence,
    });

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
      logger.info('Translation needed', {
        step: 'translation-check',
        needsTranslation: true,
      });
    }

    // Parse markdown to AST (with remark-math to parse math nodes)
    const processor = unified().use(remarkParse).use(remarkGfm).use(remarkMath);

    const tree = processor.parse(processedMarkdown) as Root;

    // Fix code fences BEFORE translation (so they are properly marked as non-translatable)
    if (enableCodeFenceFix) {
      const codeFenceSpan = logger.span({
        name: 'code-fence-fix',
        fields: { step: 'codeFenceFix' },
      });
      codeFenceSpan.start();
      const fixed = fixCodeFences(tree);
      diagnostics.codeFencesFixed = fixed;
      if (fixed > 0) {
        diagnostics.changed = true;
      }
      codeFenceSpan.end({ status: 'ok', fields: { fixedCount: fixed } });
    }

    // Fix image captions BEFORE translation (convert to markdown-native italic format)
    // This ensures captions are properly structured before translation extracts text
    if (enableImageCaptionFix) {
      const captionSpan = logger.span({
        name: 'image-caption-fix',
        fields: { step: 'captionFix' },
      });
      captionSpan.start();
      const fixed = fixImageCaptions(tree);
      diagnostics.imageCaptionsFixed = fixed;
      if (fixed > 0) {
        diagnostics.changed = true;
      }
      captionSpan.end({ status: 'ok', fields: { fixedCount: fixed } });
    }

    // Apply translation AND math fixing if needed (combined in one LLM call)
    if (needsTranslation && translator) {
      const translationSpan = logger.span({ name: 'translation', fields: { step: 'translate' } });
      translationSpan.start();
      try {
        const translatableNodes = extractTranslatableNodes(tree);
        logger.debug('Extracted translatable nodes', {
          step: 'translate',
          nodesCount: translatableNodes.length,
        });

        if (translatableNodes.length > 0) {
          const translationResult = await translator.translate(translatableNodes, {
            logger: logger.child({ step: 'translate', provider: translator.name }),
          });

          const applyStats = applyTranslationPatches(tree, translationResult.patches);
          diagnostics.mathPatched = applyStats.mathPatched;
          diagnostics.mathFallbackToCodeFence = applyStats.mathFallbackToCodeFence;
          diagnostics.invalidMathPatchReasons = applyStats.invalidMathPatchReasons;

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

          translationSpan.end({
            status: 'ok',
            fields: {
              nodesTranslated: translatableNodes.length,
              mathPatched: diagnostics.mathPatched,
              mathFallbackToCodeFence: diagnostics.mathFallbackToCodeFence,
              batches: diagnostics.translationBatches,
            },
          });
        } else {
          translationSpan.end({ status: 'ok', fields: { nodesTranslated: 0 } });
        }
      } catch (error) {
        logger.warn('Translation failed, continuing with other fixes', {
          step: 'translate',
          error: error instanceof Error ? error.message : String(error),
          fallbackUsed: true,
        });
        translationSpan.end({ status: 'fail', fields: { reason: 'translation-error' } });
      }
    }

    // Convert AST back to markdown (with remark-math to serialize math nodes)
    const stringifyProcessor = unified()
      .use(remarkStringify, {
        bullet: '-',
        fence: '`',
        fences: true,
        incrementListMarker: false,
      })
      .use(remarkGfm)
      .use(remarkMath);

    processedMarkdown = stringifyProcessor.stringify(tree);

    // Fix math delimiters (must be done after AST conversion)
    if (enableMathDelimiterFix) {
      const mathDelimiterSpan = logger.span({
        name: 'math-delimiter-fix',
        fields: { step: 'mathDelimiterFix' },
      });
      mathDelimiterSpan.start();
      const { content, changes } = fixMathDelimiters(processedMarkdown);
      processedMarkdown = content;
      diagnostics.mathDelimitersFixed = changes;
      if (changes > 0) {
        diagnostics.changed = true;
      }
      mathDelimiterSpan.end({ status: 'ok', fields: { fixedCount: changes } });
    }

    // Normalize markdown formatting
    if (enableMarkdownCleanup) {
      const cleanupSpan = logger.span({ name: 'markdown-cleanup', fields: { step: 'cleanup' } });
      cleanupSpan.start();
      const { content, changes } = normalizeMarkdown(processedMarkdown);
      processedMarkdown = content;
      diagnostics.emptyLinesCompressed = changes;
      if (changes > 0) {
        diagnostics.changed = true;
      }
      cleanupSpan.end({ status: 'ok', fields: { changesCount: changes } });
    }

    // Reconstruct final markdown with frontmatter
    const finalMarkdown = matter.stringify(processedMarkdown, frontmatter);

    processingSpan.end({ status: 'ok', fields: { changed: diagnostics.changed } });

    logger.summary({
      status: 'ok',
      changed: diagnostics.changed,
      translated: diagnostics.translated,
      codeFencesFixed: diagnostics.codeFencesFixed,
      imageCaptionsFixed: diagnostics.imageCaptionsFixed,
      mathPatched: diagnostics.mathPatched,
      mathFallbackToCodeFence: diagnostics.mathFallbackToCodeFence,
    });

    return {
      markdown: finalMarkdown,
      diagnostics,
    };
  } catch (error) {
    processingSpan.end({ status: 'fail' });
    logger.error(error instanceof Error ? error : new Error(String(error)), {
      step: 'processing-error',
    });
    throw error;
  }
}
