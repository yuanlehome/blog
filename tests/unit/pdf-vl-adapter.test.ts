/**
 * PDF VL Adapter Tests
 *
 * Tests for PDF import adapter with mocked network calls
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { pdfVlAdapter } from '../../scripts/import/adapters/pdf_vl.js';
import fs from 'fs';
import path from 'path';

// Mock dependencies
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

// Load test fixtures
const fixturesDir = path.join(process.cwd(), 'tests/fixtures/pdf');
const samplePdfBuffer = fs.readFileSync(path.join(fixturesDir, 'sample.pdf'));
const sampleImageBuffer = fs.readFileSync(path.join(fixturesDir, 'test-image.png'));

// Mock fetch globally
const originalFetch = global.fetch;

describe('PDF VL Adapter', () => {
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

  describe('canHandle', () => {
    it('should handle URLs ending with .pdf', () => {
      expect(pdfVlAdapter.canHandle('https://example.com/document.pdf')).toBe(true);
      expect(pdfVlAdapter.canHandle('https://example.com/paper.PDF')).toBe(true);
    });

    it('should not handle non-PDF URLs', () => {
      expect(pdfVlAdapter.canHandle('https://example.com/article.html')).toBe(false);
      expect(pdfVlAdapter.canHandle('https://zhuanlan.zhihu.com/p/123456')).toBe(false);
    });
  });

  describe('fetchArticle', () => {
    it('should successfully import a PDF', async () => {
      // Mock PDF download
      global.fetch = vi.fn((url: string | URL | Request) => {
        const urlStr = typeof url === 'string' ? url : url.toString();

        if (urlStr.includes('example.com/test.pdf')) {
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
                    text: `# Test Article Title

This is the first paragraph of the test article. It contains meaningful content.

## Section 1

Lorem ipsum dolor sit amet, consectetur adipiscing elit.
Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.

## Section 2

Ut enim ad minim veniam, quis nostrud exercitation ullamco.
Laboris nisi ut aliquip ex ea commodo consequat.

### Subsection 2.1

Duis aute irure dolor in reprehenderit in voluptate velit.
Esse cillum dolore eu fugiat nulla pariatur.

## Section 3

Excepteur sint occaecat cupidatat non proident, sunt in culpa.
Qui officia deserunt mollit anim id est laborum.

![Test Image](./image1.png)

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

      // Create temp directory for images
      const tempDir = path.join(process.cwd(), 'tests/tmp/pdf-test');
      fs.mkdirSync(tempDir, { recursive: true });

      try {
        const result = await pdfVlAdapter.fetchArticle({
          url: 'https://example.com/test.pdf',
          page: null as any, // PDF adapter doesn't use page
          options: {
            slug: 'test-article',
            imageRoot: tempDir,
            publicBasePath: '/images/pdf/test-article',
            logger: mockLogger as any,
          },
        });

        // Verify result
        expect(result.title).toBe('Test Article Title');
        expect(result.source).toBe('others');
        expect(result.canonicalUrl).toBe('https://example.com/test.pdf');
        expect(result.markdown).toContain('Test Article Title');
        expect(result.markdown).toContain('Section 1');
        expect(result.markdown.length).toBeGreaterThan(100);

        // Verify image was downloaded
        const imageFiles = fs.readdirSync(path.join(tempDir, 'pdf', 'test-article'));
        expect(imageFiles.length).toBeGreaterThan(0);

        // Verify effective line count
        expect(result.diagnostics?.extractionMethod).toBe('paddleocr-vl');
      } finally {
        // Cleanup
        fs.rmSync(tempDir, { recursive: true, force: true });
      }
    });

    it('should fail when PADDLEOCR_VL_TOKEN is missing', async () => {
      delete process.env.PADDLEOCR_VL_TOKEN;

      await expect(
        pdfVlAdapter.fetchArticle({
          url: 'https://example.com/test.pdf',
          page: null as any,
          options: {
            slug: 'test',
            imageRoot: '/tmp',
          },
        }),
      ).rejects.toThrow('PADDLEOCR_VL_TOKEN environment variable is required');
    });

    it('should fail when PDF download fails', async () => {
      global.fetch = vi.fn(() =>
        Promise.resolve({
          ok: false,
          status: 404,
          statusText: 'Not Found',
          headers: new Map() as any,
        } as Response),
      ) as any;

      await expect(
        pdfVlAdapter.fetchArticle({
          url: 'https://example.com/missing.pdf',
          page: null as any,
          options: {
            slug: 'test',
            imageRoot: '/tmp',
            logger: mockLogger as any,
          },
        }),
      ).rejects.toThrow();
    });

    it('should fail when content is not a PDF', async () => {
      // Create a buffer that's large enough but not a PDF
      const fakePdfBuffer = Buffer.alloc(60000, 'x');

      global.fetch = vi.fn(() =>
        Promise.resolve({
          ok: true,
          status: 200,
          headers: new Map([
            ['content-type', 'text/html'],
            ['content-length', String(fakePdfBuffer.length)],
          ]) as any,
          arrayBuffer: () => Promise.resolve(fakePdfBuffer.buffer),
        } as Response),
      ) as any;

      await expect(
        pdfVlAdapter.fetchArticle({
          url: 'https://example.com/fake.pdf',
          page: null as any,
          options: {
            slug: 'test',
            imageRoot: '/tmp',
            logger: mockLogger as any,
          },
        }),
      ).rejects.toThrow('Not a valid PDF');
    });

    it('should fail when OCR returns insufficient content', async () => {
      // Load sparse PDF
      const sparsePdfBuffer = fs.readFileSync(path.join(fixturesDir, 'sparse-sample.pdf'));

      // Mock PDF download
      global.fetch = vi.fn((url: string | URL | Request) => {
        const urlStr = typeof url === 'string' ? url : url.toString();

        if (urlStr.includes('example.com/sparse.pdf')) {
          return Promise.resolve({
            ok: true,
            status: 200,
            headers: new Map([
              ['content-type', 'application/pdf'],
              ['content-length', String(sparsePdfBuffer.length)],
            ]) as any,
            arrayBuffer: () => Promise.resolve(sparsePdfBuffer.buffer),
          } as Response);
        }

        if (urlStr.includes('layout-parsing')) {
          // Mock sparse OCR response
          const mockResponse = {
            result: {
              layoutParsingResults: [
                {
                  markdown: {
                    text: '# Title\n\nOnly a few lines of content here.',
                    images: {},
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

        return Promise.reject(new Error('Unexpected URL'));
      }) as any;

      await expect(
        pdfVlAdapter.fetchArticle({
          url: 'https://example.com/sparse.pdf',
          page: null as any,
          options: {
            slug: 'test',
            imageRoot: '/tmp',
            logger: mockLogger as any,
          },
        }),
      ).rejects.toThrow('Insufficient content quality');
    });
  });
});
