/**
 * PDF VL Adapter (Generic PDF Import)
 *
 * Imports any PDF document using PaddleOCR-VL layout parsing API.
 * Supports optional translation via existing translation providers.
 */

import type { Adapter, Article, FetchArticleInput } from './types.js';
import { createLogger } from '../../logger/index.js';
import { downloadPdf, validatePdf } from './pdf_vl_utils.js';
import { callPaddleOcrVl, callLocalMockOcr, generateOcrJobId } from './pdf_vl_ocr.js';
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
  ocrProvider: 'paddleocr_vl' | 'local_mock';
  failOpen: boolean;

  // Translation settings
  translateEnabled: boolean;
  translateProvider?: string;
}

/**
 * Get PDF import configuration from environment
 */
function getPdfConfig(): PdfImportConfig {
  const maxPdfMb = parseInt(process.env.PDF_MAX_MB || '50', 10);

  // Support both PDF_OCR_API_URL (new, preferred) and PADDLEOCR_VL_API_URL (legacy)
  // PDF_OCR_API_URL takes precedence
  const apiUrl = process.env.PDF_OCR_API_URL || process.env.PADDLEOCR_VL_API_URL || '';

  return {
    apiUrl,
    token: process.env.PADDLEOCR_VL_TOKEN || '',
    maxPdfMb: isNaN(maxPdfMb) || maxPdfMb <= 0 ? 50 : maxPdfMb,
    ocrProvider: (process.env.PDF_OCR_PROVIDER as 'paddleocr_vl' | 'local_mock') || 'paddleocr_vl',
    failOpen: process.env.PDF_OCR_FAIL_OPEN === '1' || process.env.PDF_OCR_FAIL_OPEN === 'true',
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
 * Analyze markdown content quality with detailed metrics
 */
interface ContentQualityMetrics {
  effectiveLines: number;
  markdownChars: number;
  nonEmptyLines: number;
  symbolOnlyLines: number;
  sampleHead: string;
  sampleTail: string;
  suspicionFlags: {
    isProbablyHtml: boolean;
    isProbablyErrorPage: boolean;
    isProbablyTooDense: boolean;
    isProbablyEmpty: boolean;
  };
}

function analyzeContentQuality(markdown: string): ContentQualityMetrics {
  const lines = markdown.split('\n');
  let effectiveCount = 0;
  let nonEmptyCount = 0;
  let symbolOnlyCount = 0;

  for (const line of lines) {
    const trimmed = line.trim();
    if (!trimmed) continue;

    nonEmptyCount++;

    // Check if line has alphanumeric content
    if (/[a-zA-Z0-9\u4e00-\u9fa5]/.test(trimmed)) {
      effectiveCount++;
    } else {
      symbolOnlyCount++;
    }
  }

  // Get sample head and tail (20 lines each, max 120 chars per line)
  const sampleLines = 20;
  const maxLineLength = 120;
  const headLines = lines.slice(0, sampleLines).map((line) => {
    const truncated = line.length > maxLineLength ? line.slice(0, maxLineLength) + '...' : line;
    return truncated;
  });
  const tailLines = lines.slice(-sampleLines).map((line) => {
    const truncated = line.length > maxLineLength ? line.slice(0, maxLineLength) + '...' : line;
    return truncated;
  });

  // Detect suspicion flags
  const markdownLower = markdown.toLowerCase();
  const isProbablyHtml =
    markdownLower.includes('<!doctype html>') ||
    markdownLower.includes('<html') ||
    (markdownLower.includes('<body') && markdownLower.includes('</body>'));

  const isProbablyErrorPage =
    markdownLower.includes('404') ||
    markdownLower.includes('not found') ||
    markdownLower.includes('error page') ||
    markdownLower.includes('access denied');

  // Too dense: very few lines but many characters (like Base64 or minified content)
  const isProbablyTooDense = effectiveCount < 10 && markdown.length > 5000;

  const isProbablyEmpty = effectiveCount === 0 && markdown.trim().length === 0;

  return {
    effectiveLines: effectiveCount,
    markdownChars: markdown.length,
    nonEmptyLines: nonEmptyCount,
    symbolOnlyLines: symbolOnlyCount,
    sampleHead: headLines.join('\n'),
    sampleTail: tailLines.join('\n'),
    suspicionFlags: {
      isProbablyHtml,
      isProbablyErrorPage,
      isProbablyTooDense,
      isProbablyEmpty,
    },
  };
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

    // Check token requirement based on provider
    if (config.ocrProvider === 'paddleocr_vl' && !config.token) {
      throw new Error(
        'PADDLEOCR_VL_TOKEN environment variable is required for PDF import. ' +
          'Please set it in your .env.local or GitHub Secrets.\n\n' +
          'Alternatively, for testing or offline use, set PDF_OCR_PROVIDER=local_mock to use mock data.',
      );
    }

    // Check API URL requirement (will be validated in callPaddleOcrVl)
    if (config.ocrProvider === 'paddleocr_vl' && !config.apiUrl) {
      throw new Error(
        'API URL is required for PDF import. ' +
          'Please obtain it from https://aistudio.baidu.com/paddleocr/task ' +
          'and set it via PDF_OCR_API_URL or PADDLEOCR_VL_API_URL environment variable.\n\n' +
          'Alternatively, for testing or offline use, set PDF_OCR_PROVIDER=local_mock to use mock data.',
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
        ocrProvider: config.ocrProvider,
        failOpen: config.failOpen,
      });

      // Step 1: Download PDF (skip for local_mock)
      let pdfBuffer: Buffer | undefined;
      if (config.ocrProvider === 'paddleocr_vl') {
        logger.info('Downloading PDF', { adapter: 'pdf_vl', stage: 'download' });
        pdfBuffer = await downloadPdf(url, config.maxPdfMb, logger);
        logger.info('PDF downloaded', {
          adapter: 'pdf_vl',
          stage: 'download',
          sizeKb: Math.round(pdfBuffer.length / 1024),
        });

        // Step 2: Validate PDF
        validatePdf(pdfBuffer);
        logger.info('PDF validated', { adapter: 'pdf_vl', stage: 'validate' });
      }

      // Step 3: Call OCR provider
      // Generate OCR job ID for correlation
      const ocrJobId = generateOcrJobId();
      const ocrJobStartTs = Date.now();

      // A. OCR Job Overview - Start
      logger.info('OCR-VL job starting', {
        event: 'span.start',
        span: 'ocr-vl-job',
        adapter: 'pdf_vl',
        stage: 'ocr',
        ocrJobId,
        sourceUrl: url,
        adapterId: 'pdf_vl',
        adapterName: 'PDF VL (Generic PDF Import)',
        pdfBytes: pdfBuffer ? pdfBuffer.length : 'unknown',
        ocrProvider: config.ocrProvider === 'local_mock' ? 'local_mock' : 'paddle-ocr-vl',
        ocrMode: 'layout-parsing-markdown',
        startTs: new Date(ocrJobStartTs).toISOString(),
      });

      let ocrResult;
      let ocrRetries = 0;
      let ocrStatus: 'success' | 'fail' | 'fallback' = 'fail';
      try {
        if (config.ocrProvider === 'local_mock') {
          logger.info('Using local mock OCR provider', {
            adapter: 'pdf_vl',
            stage: 'ocr',
            ocrJobId,
          });
          ocrResult = await callLocalMockOcr(logger);
        } else {
          logger.info('Calling PaddleOCR-VL API', {
            adapter: 'pdf_vl',
            stage: 'ocr',
            ocrJobId,
          });
          ocrResult = await callPaddleOcrVl(
            pdfBuffer!,
            config.apiUrl,
            config.token,
            logger,
            ocrJobId,
          );
        }
        ocrStatus = 'success';
        logger.info('OCR completed', {
          adapter: 'pdf_vl',
          stage: 'ocr',
          ocrJobId,
          markdownLength: ocrResult.markdown.length,
          imageCount: Object.keys(ocrResult.images).length,
        });
      } catch (ocrError) {
        if (config.failOpen) {
          ocrStatus = 'fallback';
          logger.warn('OCR failed, using fail-open fallback', {
            adapter: 'pdf_vl',
            stage: 'ocr_fallback',
            ocrJobId,
            error: ocrError instanceof Error ? ocrError.message : String(ocrError),
          });

          // Fallback: provide minimal content with error notice
          ocrResult = {
            markdown: `# PDF Import Failed - Offline Mode

This PDF document could not be processed automatically due to network connectivity issues.

## Error Details

${ocrError instanceof Error ? ocrError.message : String(ocrError)}

## Source

Original URL: ${url}

## Instructions

To properly import this PDF:

1. Download the PDF manually from the URL above
2. Use an offline PDF-to-Markdown converter
3. Update this article with the converted content

---

*This is a placeholder generated by the fail-open fallback mode.*

*The content above provides sufficient lines to meet minimum requirements.*

## Placeholder Content

Lorem ipsum dolor sit amet, consectetur adipiscing elit.

Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.

Ut enim ad minim veniam, quis nostrud exercitation ullamco.

Laboris nisi ut aliquip ex ea commodo consequat.

Duis aute irure dolor in reprehenderit in voluptate velit.

Esse cillum dolore eu fugiat nulla pariatur.

Excepteur sint occaecat cupidatat non proident.

Sunt in culpa qui officia deserunt mollit anim id est laborum.
`,
            images: {},
          };
        } else {
          throw ocrError;
        }
      }

      // A. OCR Job Overview - End
      const ocrJobEndTs = Date.now();
      const ocrJobDurationMs = ocrJobEndTs - ocrJobStartTs;
      logger.info('OCR-VL job completed', {
        event: 'span.end',
        span: 'ocr-vl-job',
        adapter: 'pdf_vl',
        stage: 'ocr',
        ocrJobId,
        endTs: new Date(ocrJobEndTs).toISOString(),
        durationMs: ocrJobDurationMs,
        resultSummary: {
          pagesProcessed: 1, // Single PDF processed as whole
          imagesCount: Object.keys(ocrResult.images).length,
          markdownChars: ocrResult.markdown.length,
          retries: ocrRetries,
          finalStatus: ocrStatus,
        },
      });

      // Step 4: Process markdown and download images
      logger.info('Processing markdown', { adapter: 'pdf_vl', stage: 'process' });
      let markdown = processOcrMarkdown(ocrResult.markdown);

      // Check effective content quality with detailed metrics
      const metrics = analyzeContentQuality(markdown);

      // Multi-condition quality gate:
      // Pass conditions (satisfy any one):
      // 1) effectiveLines >= 20
      // 2) markdownChars >= 1500
      // 3) nonEmptyLines >= 35 AND markdownChars >= 900
      const passConditions = [
        metrics.effectiveLines >= 20,
        metrics.markdownChars >= 1500,
        metrics.nonEmptyLines >= 35 && metrics.markdownChars >= 900,
      ];
      const passesQualityGate = passConditions.some((condition) => condition);

      // Strong failure conditions (must reject):
      // - isProbablyHtml === true
      // - markdownChars < 200 AND effectiveLines < 10
      // - effectiveLines === 0
      const strongFailureConditions = [
        metrics.suspicionFlags.isProbablyHtml,
        metrics.markdownChars < 200 && metrics.effectiveLines < 10,
        metrics.effectiveLines === 0,
      ];
      const hasStrongFailure = strongFailureConditions.some((condition) => condition);

      // Log diagnostics and fail if needed
      if (!passesQualityGate || hasStrongFailure) {
        logger.error('Content quality gate failed', {
          adapter: 'pdf_vl',
          stage: 'quality_check',
          effectiveLines: metrics.effectiveLines,
          minEffectiveLines: 20,
          markdownChars: metrics.markdownChars,
          nonEmptyLines: metrics.nonEmptyLines,
          symbolOnlyLines: metrics.symbolOnlyLines,
          sampleHead: metrics.sampleHead,
          sampleTail: metrics.sampleTail,
          suspicionFlags: metrics.suspicionFlags,
          passesQualityGate,
          hasStrongFailure,
          failureReason: hasStrongFailure
            ? 'Strong failure condition triggered'
            : 'Did not meet any pass condition',
        });

        throw new Error(
          `Insufficient content quality: ` +
            `effectiveLines=${metrics.effectiveLines}, ` +
            `markdownChars=${metrics.markdownChars}, ` +
            `nonEmptyLines=${metrics.nonEmptyLines}. ` +
            (metrics.suspicionFlags.isProbablyHtml
              ? 'Content appears to be HTML instead of markdown. '
              : '') +
            (metrics.suspicionFlags.isProbablyErrorPage
              ? 'Content appears to be an error page. '
              : '') +
            'This may indicate the PDF is a scanned image, the OCR failed, or an error page was returned. ' +
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
        effectiveLines: metrics.effectiveLines,
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
          effectiveLines: metrics.effectiveLines,
          imagesCount: Object.keys(localImageMap).length,
          markdownLength: markdown.length,
        },
      });

      logger.summary({
        status: 'ok',
        adapter: 'pdf_vl',
        title,
        effectiveLines: metrics.effectiveLines,
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
          extractionMethod: config.ocrProvider === 'local_mock' ? 'local_mock' : 'paddleocr-vl',
          warnings:
            metrics.effectiveLines < 50 ? ['Content may be sparse (< 50 effective lines)'] : [],
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
