/**
 * PDF VL Adapter (Generic PDF Import)
 *
 * Imports any PDF document using PaddleOCR-VL layout parsing API.
 * Supports optional translation via existing translation providers.
 */

import type { Adapter, Article, FetchArticleInput } from './types.js';
import { createLogger } from '../../logger/index.js';
import { downloadPdf, validatePdf } from './pdf_vl_utils.js';
import { callPaddleOcrVl } from './pdf_vl_ocr.js';
import { processOcrMarkdown, downloadOcrImages } from './pdf_vl_markdown.js';
import { processMarkdownForImport } from '../../markdown/index.js';
import path from 'path';

/**
 * Configuration for PDF import
 */
interface PdfImportConfig {
  // PaddleOCR-VL settings
  apiUrl: string;
  token: string;
  maxPdfMb: number;

  // Translation settings
  translateEnabled: boolean;
  translateProvider?: string;
}

/**
 * Get PDF import configuration from environment
 */
function getPdfConfig(): PdfImportConfig {
  return {
    apiUrl:
      process.env.PADDLEOCR_VL_API_URL ||
      'https://xbe1mb28fa0dz7kb.aistudio-app.com/layout-parsing',
    token: process.env.PADDLEOCR_VL_TOKEN || '',
    maxPdfMb: parseInt(process.env.PDF_MAX_MB || '50', 10),
    translateEnabled: process.env.MARKDOWN_TRANSLATE_ENABLED === '1',
    translateProvider: process.env.MARKDOWN_TRANSLATE_PROVIDER || 'deepseek',
  };
}

/**
 * Check if URL is a PDF based on extension or content type
 */
function isPdfUrl(url: string): boolean {
  try {
    const urlObj = new URL(url);
    const pathname = urlObj.pathname.toLowerCase();
    return pathname.endsWith('.pdf');
  } catch {
    return false;
  }
}

/**
 * Extract title from markdown (first h1) or fallback to filename
 */
function extractTitle(markdown: string, url: string): string {
  // Try to find first H1
  const h1Match = markdown.match(/^#\s+(.+)$/m);
  if (h1Match && h1Match[1].trim()) {
    return h1Match[1].trim();
  }

  // Fallback: extract filename from URL
  try {
    const urlObj = new URL(url);
    const filename = path.basename(urlObj.pathname);
    return filename.replace(/\.pdf$/i, '').replace(/[-_]/g, ' ');
  } catch {
    return 'Imported PDF Document';
  }
}

/**
 * Count effective content lines (non-empty, non-symbol-only)
 */
function countEffectiveLines(markdown: string): number {
  const lines = markdown.split('\n');
  let effectiveCount = 0;

  for (const line of lines) {
    const trimmed = line.trim();
    // Skip empty lines
    if (!trimmed) continue;
    // Skip lines with only symbols/punctuation
    if (!/[a-zA-Z0-9\u4e00-\u9fa5]/.test(trimmed)) continue;
    effectiveCount++;
  }

  return effectiveCount;
}

/**
 * PDF VL Adapter implementation
 */
export const pdfVlAdapter: Adapter = {
  id: 'others',
  name: 'PDF VL (Generic PDF Import)',

  canHandle(url: string): boolean {
    // Only handle PDF URLs based on extension
    // Note: In the actual registry, we need to check content-type for non-.pdf URLs
    return isPdfUrl(url);
  },

  async fetchArticle(input: FetchArticleInput): Promise<Article> {
    const { url, options = {} } = input;
    // Note: PDF import doesn't use the page object since we download directly
    const {
      slug = 'pdf-article',
      imageRoot = '/tmp/images',
      publicBasePath,
      logger: parentLogger,
    } = options;

    const config = getPdfConfig();

    if (!config.token) {
      throw new Error(
        'PADDLEOCR_VL_TOKEN environment variable is required for PDF import. ' +
          'Please set it in your .env.local or GitHub Secrets.',
      );
    }

    // Create child logger with context
    const logger =
      parentLogger?.child({
        module: 'import',
        adapter: 'pdf_vl',
        url,
        slug,
      }) ?? createLogger({ silent: true });

    const extractionSpan = logger.span({
      name: 'pdf-vl-extraction',
      fields: { adapter: 'pdf_vl', url },
    });
    extractionSpan.start();

    try {
      logger.info('Starting PDF import', {
        adapter: 'pdf_vl',
        url,
        maxPdfMb: config.maxPdfMb,
      });

      // Step 1: Download PDF
      logger.info('Downloading PDF', { adapter: 'pdf_vl', stage: 'download' });
      const pdfBuffer = await downloadPdf(url, config.maxPdfMb, logger);
      logger.info('PDF downloaded', {
        adapter: 'pdf_vl',
        stage: 'download',
        sizeKb: Math.round(pdfBuffer.length / 1024),
      });

      // Step 2: Validate PDF
      validatePdf(pdfBuffer);
      logger.info('PDF validated', { adapter: 'pdf_vl', stage: 'validate' });

      // Step 3: Call PaddleOCR-VL
      logger.info('Calling PaddleOCR-VL API', {
        adapter: 'pdf_vl',
        stage: 'ocr',
        // Don't log token
      });
      const ocrResult = await callPaddleOcrVl(pdfBuffer, config.apiUrl, config.token, logger);
      logger.info('OCR completed', {
        adapter: 'pdf_vl',
        stage: 'ocr',
        markdownLength: ocrResult.markdown.length,
        imageCount: Object.keys(ocrResult.images).length,
      });

      // Step 4: Process markdown and download images
      logger.info('Processing markdown', { adapter: 'pdf_vl', stage: 'process' });
      let markdown = processOcrMarkdown(ocrResult.markdown);

      // Check effective content quality
      const effectiveLines = countEffectiveLines(markdown);
      if (effectiveLines < 20) {
        throw new Error(
          `Insufficient content quality: only ${effectiveLines} effective lines found (minimum 20 required). ` +
            'This may indicate the PDF is a scanned image or the OCR failed. ' +
            'Please verify the PDF is text-based and try again.',
        );
      }

      logger.info('Downloading images', {
        adapter: 'pdf_vl',
        stage: 'images',
        count: Object.keys(ocrResult.images).length,
      });
      const localImageMap = await downloadOcrImages(
        ocrResult.images,
        imageRoot,
        slug,
        publicBasePath || `/images/pdf/${slug}`,
        logger,
      );

      // Update image references in markdown
      for (const [imgPath, localPath] of Object.entries(localImageMap)) {
        // Replace image references - handle both relative and absolute paths
        const patterns = [
          new RegExp(
            `!\\[([^\\]]*)\\]\\(${imgPath.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}\\)`,
            'g',
          ),
          new RegExp(
            `!\\[([^\\]]*)\\]\\(\\.\\/${imgPath.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}\\)`,
            'g',
          ),
        ];

        for (const pattern of patterns) {
          markdown = markdown.replace(pattern, `![$1](${localPath})`);
        }
      }

      logger.info('Markdown processed', {
        adapter: 'pdf_vl',
        stage: 'process',
        effectiveLines,
      });

      // Step 5: Optional translation
      if (config.translateEnabled && config.translateProvider) {
        logger.info('Translating markdown', {
          adapter: 'pdf_vl',
          stage: 'translate',
          provider: config.translateProvider,
        });

        const translationResult = await processMarkdownForImport(
          { markdown, slug, source: 'pdf' },
          {
            logger,
            enableTranslation: true,
          },
        );

        markdown = translationResult.markdown;
        logger.info('Translation completed', {
          adapter: 'pdf_vl',
          stage: 'translate',
          translated: translationResult.diagnostics.translated,
        });
      }

      // Extract title
      const title = extractTitle(markdown, url);

      extractionSpan.end({
        status: 'ok',
        fields: {
          title,
          effectiveLines,
          imagesCount: Object.keys(localImageMap).length,
          markdownLength: markdown.length,
        },
      });

      logger.summary({
        status: 'ok',
        adapter: 'pdf_vl',
        title,
        effectiveLines,
        imagesCount: Object.keys(localImageMap).length,
        markdownLength: markdown.length,
      });

      return {
        title,
        markdown,
        canonicalUrl: url,
        source: 'others',
        publishedAt: new Date().toISOString(),
        images: Object.values(localImageMap).map((localPath) => ({
          url: '',
          localPath,
        })),
        diagnostics: {
          extractionMethod: 'paddleocr-vl',
          warnings: effectiveLines < 50 ? ['Content may be sparse (< 50 effective lines)'] : [],
        },
      };
    } catch (error) {
      extractionSpan.end({ status: 'fail' });
      logger.error(error instanceof Error ? error : new Error(String(error)), {
        adapter: 'pdf_vl',
        url,
        stage: 'general',
      });
      throw error;
    }
  },
};
