import { describe, expect, it, vi } from 'vitest';
import fs from 'fs';
import path from 'path';
import {
  processMdFiles,
  normalizeInvisibleCharacters,
  splitCodeFences,
} from '../../scripts/utils.ts';
import { processMarkdownForImport } from '../../scripts/markdown/index.js';

const fixturesDir = path.join(process.cwd(), 'tests/fixtures/posts');

describe('math delimiter fixing (from process-md-files)', () => {
  it('normalizes inline and block math tokens', () => {
    const fixturePath = path.join(fixturesDir, 'fixture-alpha.md');
    const content = fs.readFileSync(fixturePath, 'utf-8');
    const fixed = processMdFiles(content);

    expect(fixed).toContain('$a + b = c$');
    expect(fixed).toMatch(/\n?\$\$/); // block promotion retained
  });

  it('normalizes invisible characters and trims inline math spacing', () => {
    const input = 'A\u00a0B $ x $ and \\$kept$';
    const fixed = processMdFiles(input);

    expect(normalizeInvisibleCharacters(input)).toContain('A B');
    expect(fixed).toContain('$x$');
    expect(fixed).toContain('\\$kept$');
  });

  it('splits frontmatter and code fences safely', () => {
    const text = `---\ntitle: demo\n---\npara\n\`\`\`js\n$code$\n\`\`\`\n~~~py\ncode\n`;
    const segments = splitCodeFences(text);

    expect(segments[0].type).toBe('frontmatter');
    const codeSegments = segments.filter((s) => s.type === 'code');
    expect(codeSegments.some((s) => s.content.includes('```js'))).toBe(true);
    expect(codeSegments.some((s) => s.content.includes('~~~py'))).toBe(true);
  });

  it('handles inline code, promotions, and unclosed math blocks', () => {
    const input = 'Text ```$kept$``` $ spaced $ $line\nwith newline$ $$unclosed';
    const fixed = processMdFiles(input);

    expect(fixed).toContain('```$kept$```');
    expect(fixed).toContain('$spaced$');
    expect(fixed).toContain('$$\nline\nwith newline\n$$');
    expect(fixed).toContain('$$unclosed');
  });

  it('handles unclosed frontmatter and escaped inline dollars', () => {
    const text = '---\ntitle: test\ncontent without end $a \\$ b$';
    const fixed = processMdFiles(text);

    expect(fixed).toContain('$a \\$ b$');
  });

  it('keeps mixed fence markers within an open fence', () => {
    const content = '```js\ncode\n~~~\nstill code';
    const segments = splitCodeFences(content);

    expect(segments.some((s) => s.type === 'code' && s.content.includes('~~~'))).toBe(true);
  });
});

describe('markdown processor integration', () => {
  it('processes markdown with math delimiter fix enabled', async () => {
    const input = 'Some text $ x $ here';
    const result = await processMarkdownForImport(
      { markdown: input },
      {
        enableMathDelimiterFix: true,
        enableTranslation: false,
        enableCodeFenceFix: false,
        enableImageCaptionFix: false,
        enableMarkdownCleanup: false,
      },
    );

    expect(result.markdown).toContain('$x$');
    expect(result.diagnostics.mathDelimitersFixed).toBeGreaterThan(0);
  });

  it('processes files through markdown processor', async () => {
    const tmpFile = path.join(process.cwd(), 'tmp-math-test.md');
    const input = '---\ntitle: test\n---\n\nNumber $ spaced $ here';
    fs.writeFileSync(tmpFile, input);

    const content = fs.readFileSync(tmpFile, 'utf-8');
    const result = await processMarkdownForImport(
      { markdown: content },
      { enableMathDelimiterFix: true },
    );

    fs.writeFileSync(tmpFile, result.markdown);
    const updated = fs.readFileSync(tmpFile, 'utf-8');

    expect(updated).toContain('$spaced$');
    expect(result.diagnostics.mathDelimitersFixed).toBeGreaterThan(0);

    fs.rmSync(tmpFile, { force: true });
  });
});
