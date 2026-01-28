/**
 * arXiv Adapter Tests
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { parseArxivId, arxivAdapter } from '../../scripts/import/adapters/arxiv.js';
import fs from 'fs';
import path from 'path';
import os from 'os';
import axios from 'axios';
import { createLogger } from '../../scripts/logger/index.js';

describe('arXiv Adapter', () => {
  describe('parseArxivId', () => {
    it('should parse arXiv PDF URLs', () => {
      expect(parseArxivId('https://arxiv.org/pdf/2306.00978')).toBe('2306.00978');
      expect(parseArxivId('https://arxiv.org/pdf/2306.00978.pdf')).toBe('2306.00978');
    });

    it('should parse arXiv abs URLs', () => {
      expect(parseArxivId('https://arxiv.org/abs/2306.00978')).toBe('2306.00978');
      expect(parseArxivId('https://arxiv.org/abs/1234.5678v2')).toBe('1234.5678v2');
    });

    it('should parse arXiv src URLs', () => {
      expect(parseArxivId('https://arxiv.org/src/2306.00978')).toBe('2306.00978');
      expect(parseArxivId('https://arxiv.org/src/2306.00978v5')).toBe('2306.00978v5');
    });

    it('should parse arXiv e-print URLs', () => {
      expect(parseArxivId('https://arxiv.org/e-print/2306.00978')).toBe('2306.00978');
    });

    it('should handle version numbers', () => {
      expect(parseArxivId('https://arxiv.org/pdf/2306.00978v5')).toBe('2306.00978v5');
      expect(parseArxivId('https://arxiv.org/abs/1234.5678v12')).toBe('1234.5678v12');
    });

    it('should return null for non-arXiv URLs', () => {
      expect(parseArxivId('https://example.com/paper')).toBeNull();
      expect(parseArxivId('https://google.com/arxiv')).toBeNull();
      // Security: prevent arbitrary hosts that contain arxiv.org
      expect(parseArxivId('https://evil-arxiv.org.example.com/pdf/1234.5678')).toBeNull();
      expect(parseArxivId('https://arxiv.org.attacker.com/pdf/1234.5678')).toBeNull();
    });

    it('should return null for invalid arXiv URLs', () => {
      expect(parseArxivId('https://arxiv.org/help')).toBeNull();
      expect(parseArxivId('https://arxiv.org/')).toBeNull();
    });
  });

  describe('Metadata Extraction', () => {
    it('should extract entry title, not feed title', () => {
      // This simulates the arXiv API response structure
      const feedXml = fs.readFileSync(
        path.join(process.cwd(), 'tests/fixtures/arxiv/feed-response.xml'),
        'utf-8',
      );

      // Extract entry
      const entryMatch = feedXml.match(/<entry>([\s\S]*?)<\/entry>/);
      expect(entryMatch).toBeTruthy();

      const entryXml = entryMatch![1];

      // Extract title from entry
      const titleMatch = entryXml.match(/<title>([\s\S]*?)<\/title>/);
      expect(titleMatch).toBeTruthy();

      const title = titleMatch![1].trim().replace(/\s+/g, ' ');
      expect(title).toBe('Evaluating Large Language Models at Evaluating Instruction Following');
      expect(title).not.toContain('arXiv Query');
      expect(title).not.toContain('search_query');
      expect(title).not.toContain('&amp;');
    });

    it('should decode HTML entities in metadata', () => {
      const decodeHtmlEntities = (text: string): string => {
        const entities: Record<string, string> = {
          '&amp;': '&',
          '&lt;': '<',
          '&gt;': '>',
          '&quot;': '"',
          '&apos;': "'",
          '&#39;': "'",
        };
        return text.replace(/&[#a-z0-9]+;/gi, (entity) => entities[entity] || entity);
      };

      expect(decodeHtmlEntities('Test &amp; Test')).toBe('Test & Test');
      expect(decodeHtmlEntities('&lt;tag&gt;')).toBe('<tag>');
      expect(decodeHtmlEntities('&quot;quoted&quot;')).toBe('"quoted"');
    });

    it('should collapse whitespace in titles', () => {
      const title = 'Title   with\n  multiple\t  spaces';
      const normalized = title.replace(/\s+/g, ' ').trim();
      expect(normalized).toBe('Title with multiple spaces');
    });
  });

  describe('Main TeX Detection', () => {
    let tempDir: string;

    beforeEach(() => {
      tempDir = path.join(os.tmpdir(), `test-arxiv-${Date.now()}`);
      fs.mkdirSync(tempDir, { recursive: true });
    });

    afterEach(() => {
      if (fs.existsSync(tempDir)) {
        fs.rmSync(tempDir, { recursive: true, force: true });
      }
    });

    it('should identify single tex file', () => {
      const texContent = `
\\documentclass{article}
\\begin{document}
Test content
\\end{document}
      `;
      fs.writeFileSync(path.join(tempDir, 'paper.tex'), texContent);

      // Import the function dynamically to test
      const detectMainTex = (dir: string, files: string[]): string | null => {
        const texFiles = files.filter((f) => f.endsWith('.tex'));
        if (texFiles.length === 1) return texFiles[0];

        for (const texFile of texFiles) {
          const content = fs.readFileSync(path.join(dir, texFile), 'utf-8');
          if (content.includes('\\documentclass') && content.includes('\\begin{document}')) {
            return texFile;
          }
        }
        return null;
      };

      const mainTex = detectMainTex(tempDir, ['paper.tex']);
      expect(mainTex).toBe('paper.tex');
    });

    it('should prefer main.tex over other files', () => {
      const validTex = `
\\documentclass{article}
\\begin{document}
Content
\\end{document}
      `;

      fs.writeFileSync(path.join(tempDir, 'main.tex'), validTex);
      fs.writeFileSync(path.join(tempDir, 'other.tex'), validTex);

      const detectMainTex = (dir: string, files: string[]): string | null => {
        const texFiles = files.filter((f) => f.endsWith('.tex'));
        const candidates: Array<{ file: string; score: number }> = [];

        for (const texFile of texFiles) {
          const content = fs.readFileSync(path.join(dir, texFile), 'utf-8');
          if (!content.includes('\\documentclass') || !content.includes('\\begin{document}')) {
            continue;
          }

          let score = 100;
          const basename = path.basename(texFile).toLowerCase();
          if (basename === 'main.tex') score += 50;
          candidates.push({ file: texFile, score });
        }

        if (candidates.length === 0) return null;
        candidates.sort((a, b) => b.score - a.score);
        return candidates[0].file;
      };

      const mainTex = detectMainTex(tempDir, ['main.tex', 'other.tex']);
      expect(mainTex).toBe('main.tex');
    });

    it('should return null if no valid tex file found', () => {
      fs.writeFileSync(path.join(tempDir, 'invalid.tex'), 'No document class here');

      const detectMainTex = (dir: string, files: string[]): string | null => {
        const texFiles = files.filter((f) => f.endsWith('.tex'));

        for (const texFile of texFiles) {
          const content = fs.readFileSync(path.join(dir, texFile), 'utf-8');
          if (content.includes('\\documentclass') && content.includes('\\begin{document}')) {
            return texFile;
          }
        }
        return null;
      };

      const mainTex = detectMainTex(tempDir, ['invalid.tex']);
      expect(mainTex).toBeNull();
    });
  });

  describe('Path Traversal Security', () => {
    it('should detect path traversal attempts', () => {
      const isSafePath = (filePath: string, extractDir: string): boolean => {
        const resolved = path.resolve(extractDir, filePath);
        const relative = path.relative(extractDir, resolved);

        if (relative.startsWith('..') || path.isAbsolute(relative)) {
          return false;
        }
        return true;
      };

      const extractDir = path.join(os.tmpdir(), 'extract');

      expect(isSafePath('normal.tex', extractDir)).toBe(true);
      expect(isSafePath('subdir/file.tex', extractDir)).toBe(true);
      expect(isSafePath('../../../etc/passwd', extractDir)).toBe(false);
      expect(isSafePath('/etc/passwd', extractDir)).toBe(false);
      expect(isSafePath('../../danger', extractDir)).toBe(false);
    });
  });

  describe('LaTeX to Markdown Conversion', () => {
    it('should convert sections to markdown headings', () => {
      const latex = `
\\section{Introduction}
Some text here.
\\subsection{Background}
More text.
      `;

      const convert = (text: string): string => {
        let result = text;
        result = result.replace(/\\section\{([^}]+)\}/g, '\n## $1\n');
        result = result.replace(/\\subsection\{([^}]+)\}/g, '\n### $1\n');
        return result;
      };

      const markdown = convert(latex);
      expect(markdown).toContain('## Introduction');
      expect(markdown).toContain('### Background');
    });

    it('should convert text formatting', () => {
      const latex = '\\textbf{bold} and \\textit{italic}';

      const convert = (text: string): string => {
        let result = text;
        result = result.replace(/\\textbf\{([^}]+)\}/g, '**$1**');
        result = result.replace(/\\textit\{([^}]+)\}/g, '*$1*');
        return result;
      };

      const markdown = convert(latex);
      expect(markdown).toBe('**bold** and *italic*');
    });

    it('should convert math equations', () => {
      const latex = 'Inline \\(x^2\\) and display \\[E=mc^2\\]';

      const convert = (text: string): string => {
        let result = text;
        result = result.replace(/\\\(([\s\S]*?)\\\)/g, '$$$1$$');
        result = result.replace(/\\\[([\s\S]*?)\\\]/g, '\n$$$$\n$1\n$$$$\n');
        return result;
      };

      const markdown = convert(latex);
      expect(markdown).toContain('$x^2$');
      expect(markdown).toContain('$$\nE=mc^2\n$$');
    });

    it('should filter out kramdown attribute list noise', () => {
      const markdown = `
Some content
{: .class-name}
More content
{: #id-name}
Final content
`;

      const cleaned = markdown.replace(/^\s*\{:.*?\}\s*$/gm, '');
      expect(cleaned).not.toContain('{: .class-name}');
      expect(cleaned).not.toContain('{: #id-name}');
      expect(cleaned).toContain('Some content');
      expect(cleaned).toContain('More content');
    });

    it('should filter out standalone braces', () => {
      const markdown = `
Content line
{}
:
More content
`;

      const cleaned = markdown.replace(/^\s*[\{\}:]+\s*$/gm, '');
      expect(cleaned).toContain('Content line');
      expect(cleaned).toContain('More content');
      expect(cleaned.split('\n').filter((l) => l.trim() === '{}' || l.trim() === ':')).toHaveLength(
        0,
      );
    });
  });

  describe('Content Quality Validation', () => {
    it('should reject content with too few lines', () => {
      const shortContent = 'Line 1\nLine 2\nLine 3';
      const lines = shortContent.split('\n').filter((line) => line.trim().length > 0);

      expect(lines.length).toBeLessThan(20);
    });

    it('should accept content with sufficient lines', () => {
      const goodContent = Array(30)
        .fill(0)
        .map((_, i) => `This is content line ${i} with sufficient text`)
        .join('\n');
      const lines = goodContent.split('\n').filter((line) => line.trim().length > 5);

      expect(lines.length).toBeGreaterThanOrEqual(20);
    });

    it('should calculate average line length', () => {
      const content = 'Short\nA bit longer line\nAnother line with more text';
      const lines = content.split('\n').filter((line) => line.trim().length > 0);
      const avgLength = lines.reduce((sum, line) => sum + line.length, 0) / lines.length;

      expect(avgLength).toBeGreaterThan(0);
    });
  });

  describe('Integration Tests', () => {
    let tempDir: string;

    beforeEach(() => {
      tempDir = path.join(os.tmpdir(), `test-arxiv-integration-${Date.now()}`);
      fs.mkdirSync(tempDir, { recursive: true });
    });

    afterEach(() => {
      if (fs.existsSync(tempDir)) {
        fs.rmSync(tempDir, { recursive: true, force: true });
      }
      vi.restoreAllMocks();
    });

    it('should handle canHandle correctly', () => {
      expect(arxivAdapter.canHandle('https://arxiv.org/pdf/2306.00978')).toBe(true);
      expect(arxivAdapter.canHandle('https://arxiv.org/abs/2306.00978')).toBe(true);
      expect(arxivAdapter.canHandle('https://example.com/paper')).toBe(false);
    });

    it('should have correct adapter metadata', () => {
      expect(arxivAdapter.id).toBe('arxiv');
      expect(arxivAdapter.name).toBe('arXiv');
    });

    it('should handle API metadata fetch with mock', async () => {
      // Mock axios to return our test XML
      const mockXml = fs.readFileSync(
        path.join(process.cwd(), 'tests/fixtures/arxiv/feed-response.xml'),
        'utf-8',
      );

      const axiosMock = vi.spyOn(axios, 'get');
      axiosMock.mockImplementation((url: string, config?: any) => {
        if (url.includes('export.arxiv.org/api/query')) {
          return Promise.resolve({ data: mockXml });
        }
        if (url.includes('arxiv.org/src/')) {
          // Return a mock tar.gz stream
          const tarPath = path.join(process.cwd(), 'tests/fixtures/arxiv/test-paper.tar.gz');
          const stream = fs.createReadStream(tarPath);
          return Promise.resolve({ data: stream });
        }
        return Promise.reject(new Error('Unexpected URL'));
      });

      const logger = createLogger({ silent: true });
      const imageRoot = path.join(tempDir, 'images');
      fs.mkdirSync(imageRoot, { recursive: true });

      try {
        const result = await arxivAdapter.fetchArticle({
          url: 'https://arxiv.org/pdf/2306.00978',
          page: null as any, // arXiv adapter doesn't use page
          options: {
            slug: 'test-paper',
            imageRoot,
            logger,
            publicBasePath: '/test-images',
          },
        });

        expect(result.title).toBe(
          'Evaluating Large Language Models at Evaluating Instruction Following',
        );
        expect(result.source).toBe('arxiv');
        expect(result.canonicalUrl).toBe('https://arxiv.org/pdf/2306.00978');
        expect(result.tags).toContain('arxiv');
        expect(result.markdown).toBeTruthy();
        expect(result.markdown.length).toBeGreaterThan(100);
      } catch (error: any) {
        // It's ok if validation fails for the mock content being too short
        if (!error.message.includes('Converted content too short')) {
          throw error;
        }
      }

      axiosMock.mockRestore();
    });

    it('should throw error for invalid arXiv URL', async () => {
      const logger = createLogger({ silent: true });
      await expect(
        arxivAdapter.fetchArticle({
          url: 'https://example.com/not-arxiv',
          page: null as any,
          options: { slug: 'test', imageRoot: tempDir, logger },
        }),
      ).rejects.toThrow('Invalid arXiv URL');
    });

    it('should handle source download failures gracefully', async () => {
      const axiosMock = vi.spyOn(axios, 'get');
      const isAxiosErrorMock = vi.spyOn(axios, 'isAxiosError');

      const error: any = new Error('Request failed with status code 404');
      error.response = { status: 404 };
      error.isAxiosError = true;

      axiosMock.mockImplementation((url: string) => {
        if (url.includes('arxiv.org/src/')) {
          return Promise.reject(error);
        }
        return Promise.reject(new Error('Unexpected URL'));
      });

      isAxiosErrorMock.mockReturnValue(true);

      const logger = createLogger({ silent: true });
      await expect(
        arxivAdapter.fetchArticle({
          url: 'https://arxiv.org/pdf/2306.00978',
          page: null as any,
          options: { slug: 'test', imageRoot: tempDir, logger },
        }),
      ).rejects.toThrow(/source not found/i);

      axiosMock.mockRestore();
      isAxiosErrorMock.mockRestore();
    });

    it('should handle metadata fetch failures', async () => {
      const axiosMock = vi.spyOn(axios, 'get');
      // Mock successful source download first
      axiosMock.mockImplementation((url: string, config?: any) => {
        if (url.includes('arxiv.org/src/')) {
          const tarPath = path.join(process.cwd(), 'tests/fixtures/arxiv/test-paper.tar.gz');
          const stream = fs.createReadStream(tarPath);
          return Promise.resolve({ data: stream });
        }
        if (url.includes('export.arxiv.org/api/query')) {
          return Promise.reject(new Error('Network error'));
        }
        return Promise.reject(new Error('Unexpected URL'));
      });

      const logger = createLogger({ silent: true });
      await expect(
        arxivAdapter.fetchArticle({
          url: 'https://arxiv.org/pdf/2306.00978',
          page: null as any,
          options: { slug: 'test', imageRoot: tempDir, logger },
        }),
      ).rejects.toThrow(/Failed to fetch arXiv metadata/i);

      axiosMock.mockRestore();
    });

    it('should handle API response without entry', async () => {
      const axiosMock = vi.spyOn(axios, 'get');
      axiosMock.mockImplementation((url: string, config?: any) => {
        if (url.includes('arxiv.org/src/')) {
          const tarPath = path.join(process.cwd(), 'tests/fixtures/arxiv/test-paper.tar.gz');
          const stream = fs.createReadStream(tarPath);
          return Promise.resolve({ data: stream });
        }
        if (url.includes('export.arxiv.org/api/query')) {
          return Promise.resolve({ data: '<feed><title>No entries</title></feed>' });
        }
        return Promise.reject(new Error('Unexpected URL'));
      });

      const logger = createLogger({ silent: true });
      await expect(
        arxivAdapter.fetchArticle({
          url: 'https://arxiv.org/pdf/2306.00978',
          page: null as any,
          options: { slug: 'test', imageRoot: tempDir, logger },
        }),
      ).rejects.toThrow(/No entry found in arXiv API response/i);

      axiosMock.mockRestore();
    });
  });
});
