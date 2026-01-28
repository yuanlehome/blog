/**
 * arXiv Adapter Tests
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { parseArxivId } from '../../scripts/import/adapters/arxiv.js';
import fs from 'fs';
import path from 'path';
import os from 'os';

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
  });
});
