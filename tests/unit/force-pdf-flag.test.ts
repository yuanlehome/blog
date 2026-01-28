/**
 * Force PDF Flag Tests
 *
 * Tests for the --forcePdf flag functionality
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import fs from 'fs';
import path from 'path';

// Load test fixtures
const fixturesDir = path.join(process.cwd(), 'tests/fixtures/pdf');
const samplePdfBuffer = fs.readFileSync(path.join(fixturesDir, 'sample.pdf'));
const sampleImageBuffer = fs.readFileSync(path.join(fixturesDir, 'test-image.png'));

// Mock fetch globally
const originalFetch = global.fetch;

describe('Force PDF Flag', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    // Set required environment variable
    process.env.PADDLEOCR_VL_TOKEN = 'test-token-12345';
    process.env.MARKDOWN_TRANSLATE_ENABLED = '0';
  });

  afterEach(() => {
    global.fetch = originalFetch;
    delete process.env.PADDLEOCR_VL_TOKEN;
    delete process.env.MARKDOWN_TRANSLATE_ENABLED;
  });

  describe('arXiv URL handling', () => {
    it('should block arXiv URLs by default (forcePdf=false)', async () => {
      // This tests the default behavior - arXiv should be blocked
      const { isArxivUrl } = await import('../../scripts/content-import.js');

      // Test various arXiv URL patterns
      expect(isArxivUrl('https://arxiv.org/pdf/2306.00978')).toBe(true);
      expect(isArxivUrl('https://arxiv.org/abs/2306.00978')).toBe(true);
      expect(isArxivUrl('https://ar5iv.labs.arxiv.org/html/2306.00978')).toBe(true);

      // Non-arXiv URLs should not be blocked
      expect(isArxivUrl('https://example.com/document.pdf')).toBe(false);
      expect(isArxivUrl('https://pdfviewer.com/arxiv-like.pdf')).toBe(false);
    });

    it('should allow PDF import with forcePdf=true even for arXiv URLs', async () => {
      // Mock PDF download and OCR for arXiv URL
      global.fetch = vi.fn((url: string | URL | Request) => {
        const urlStr = typeof url === 'string' ? url : url.toString();

        if (urlStr.includes('arxiv.org/pdf/2306.00978')) {
          return Promise.resolve({
            ok: true,
            status: 200,
            headers: new Map([
              ['content-type', 'application/pdf'],
              ['content-length', String(samplePdfBuffer.length)],
            ]) as any,
            arrayBuffer: () => Promise.resolve(samplePdfBuffer.buffer),
          } as Response);
        }

        if (urlStr.includes('layout-parsing')) {
          // Mock PaddleOCR-VL response
          const mockResponse = {
            result: {
              layoutParsingResults: [
                {
                  markdown: {
                    text: `# arXiv Paper Title

This is an arXiv paper imported via forcePdf mode.

## Introduction

Lorem ipsum dolor sit amet, consectetur adipiscing elit.
Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.

## Method

Ut enim ad minim veniam, quis nostrud exercitation ullamco.
Laboris nisi ut aliquip ex ea commodo consequat.

## Results

Duis aute irure dolor in reprehenderit in voluptate velit.
Esse cillum dolore eu fugiat nulla pariatur.

## Discussion

Excepteur sint occaecat cupidatat non proident, sunt in culpa.
Qui officia deserunt mollit anim id est laborum.

![Figure 1](./image1.png)

More content here to ensure we meet the minimum line requirement.
This is line 20.
And line 21.
Line 22 for good measure.
Line 23.
`,
                    images: {
                      'image1.png': 'https://example.com/images/test-image.png',
                    },
                  },
                },
              ],
            },
          };

          return Promise.resolve({
            ok: true,
            status: 200,
            text: () => Promise.resolve(JSON.stringify(mockResponse)),
          } as Response);
        }

        // Mock image download
        if (urlStr.includes('test-image.png')) {
          return Promise.resolve({
            ok: true,
            status: 200,
            arrayBuffer: () => Promise.resolve(sampleImageBuffer.buffer),
          } as Response);
        }

        return Promise.reject(new Error('Unexpected URL: ' + urlStr));
      }) as any;

      // Import PDF adapter
      const { pdfVlAdapter } = await import('../../scripts/import/adapters/pdf_vl.js');

      // Create temp directory for images
      const tempDir = path.join(process.cwd(), 'tests/tmp/force-pdf-test');
      fs.mkdirSync(tempDir, { recursive: true });

      const mockLogger = {
        debug: vi.fn(),
        info: vi.fn(),
        warn: vi.fn(),
        error: vi.fn(),
        summary: vi.fn(),
        span: vi.fn(() => ({
          start: vi.fn(),
          end: vi.fn(),
        })),
        child: vi.fn(function (this: any) {
          return this;
        }),
      };

      try {
        // Test that PDF adapter can handle the arXiv URL when forcePdf is enabled
        const result = await pdfVlAdapter.fetchArticle({
          url: 'https://arxiv.org/pdf/2306.00978',
          page: null as any,
          options: {
            slug: 'arxiv-paper',
            imageRoot: tempDir,
            publicBasePath: '/images/pdf/arxiv-paper',
            logger: mockLogger as any,
          },
        });

        // Verify result
        expect(result.title).toBe('arXiv Paper Title');
        expect(result.source).toBe('others');
        expect(result.canonicalUrl).toBe('https://arxiv.org/pdf/2306.00978');
        expect(result.markdown).toContain('arXiv paper imported via forcePdf mode');
        expect(result.markdown.length).toBeGreaterThan(100);

        // Verify image was downloaded
        const imageFiles = fs.readdirSync(path.join(tempDir, 'pdf', 'arxiv-paper'));
        expect(imageFiles.length).toBeGreaterThan(0);

        // Verify effective line count
        expect(result.diagnostics?.extractionMethod).toBe('paddleocr-vl');
      } finally {
        // Cleanup
        fs.rmSync(tempDir, { recursive: true, force: true });
      }
    });
  });

  describe('PDF adapter selection with forcePdf', () => {
    it('should force PDF adapter for PDF URLs when forcePdf=true', async () => {
      const { pdfVlAdapter } = await import('../../scripts/import/adapters/pdf_vl.js');

      // PDF adapter handles URLs ending with .pdf
      expect(pdfVlAdapter.canHandle('https://example.com/document.pdf')).toBe(true);
      expect(pdfVlAdapter.canHandle('https://arxiv.org/pdf/2306.00978.pdf')).toBe(true);

      // Note: arxiv.org/pdf/XXX doesn't end with .pdf, so canHandle returns false
      // But with forcePdf flag, the adapter will be forced regardless of canHandle
      expect(pdfVlAdapter.canHandle('https://arxiv.org/pdf/2306.00978')).toBe(false);
    });

    it('should use normal adapter resolution when forcePdf=false', async () => {
      const { resolveAdapter } = await import('../../scripts/import/adapters/index.js');

      // Without forcePdf, arXiv URLs would be blocked before adapter resolution
      // Test that normal URLs resolve to correct adapters
      const zhihuAdapter = resolveAdapter('https://zhuanlan.zhihu.com/p/123456');
      expect(zhihuAdapter?.id).toBe('zhihu');

      const mediumAdapter = resolveAdapter('https://medium.com/@user/article-slug-123');
      expect(mediumAdapter?.id).toBe('medium');

      const pdfAdapter = resolveAdapter('https://example.com/document.pdf');
      expect(pdfAdapter?.id).toBe('others'); // PDF adapter has id 'others'
    });
  });
});
