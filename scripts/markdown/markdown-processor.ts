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
}

export interface ProcessingDiagnostics {
  changed: boolean;
  translated: boolean;
  detectedLanguage?: string;
  codeFencesFixed: number;
  imageCaptionsFixed: number;
  emptyLinesCompressed: number;
  frontmatterUpdated: boolean;
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
 * Fix image captions by converting to HTML figure elements
 */
function fixImageCaptions(tree: Root): number {
  let fixedCount = 0;
  const nodesToReplace: Array<{
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
        const figureHtml = createFigureHtml(image, caption);

        nodesToReplace.push({
          parent,
          index: index!,
          newNode: {
            type: 'html',
            value: figureHtml,
          },
        });

        // Mark next sibling for removal
        nodesToReplace.push({
          parent,
          index: index! + 1,
          newNode: null as any,
        });

        fixedCount++;
      } else if (image.alt) {
        // Convert standalone image with alt text to figure
        const figureHtml = createFigureHtml(image, image.alt);
        nodesToReplace.push({
          parent,
          index: index!,
          newNode: {
            type: 'html',
            value: figureHtml,
          },
        });
        fixedCount++;
      }
    }
  });

  // Apply replacements (in reverse order to maintain indices)
  nodesToReplace.reverse().forEach(({ parent, index, newNode }) => {
    if (newNode) {
      parent.children[index] = newNode;
    } else {
      parent.children.splice(index, 1);
    }
  });

  return fixedCount;
}

/**
 * Check if a paragraph looks like a caption
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
  return text.trim().length > 0;
}

/**
 * Create HTML figure element with image and caption
 */
function createFigureHtml(image: Image, caption?: string): string {
  const img = `<img src="${image.url}" alt="${image.alt || ''}" />`;
  if (caption && caption.trim()) {
    return `<figure>\n  ${img}\n  <figcaption>${caption}</figcaption>\n</figure>`;
  }
  return `<figure>\n  ${img}\n</figure>`;
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
  } = options;

  const diagnostics: ProcessingDiagnostics = {
    changed: false,
    translated: false,
    codeFencesFixed: 0,
    imageCaptionsFixed: 0,
    emptyLinesCompressed: 0,
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

  // Fix code fences
  if (enableCodeFenceFix) {
    const fixed = fixCodeFences(tree);
    diagnostics.codeFencesFixed = fixed;
    if (fixed > 0) {
      diagnostics.changed = true;
    }
  }

  // Fix image captions
  if (enableImageCaptionFix) {
    const fixed = fixImageCaptions(tree);
    diagnostics.imageCaptionsFixed = fixed;
    if (fixed > 0) {
      diagnostics.changed = true;
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
