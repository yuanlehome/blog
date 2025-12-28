import { describe, expect, it, vi } from 'vitest';
import fs from 'fs';
import path from 'path';
import {
  fixMath,
  normalizeInvisibleCharacters,
  runCli,
  runFixMath,
  splitCodeFences,
} from '../../scripts/fix-math';

const fixturesDir = path.join(process.cwd(), 'tests/fixtures/posts');

describe('fix-math script', () => {
  it('normalizes inline and block math tokens', () => {
    const fixturePath = path.join(fixturesDir, 'fixture-alpha.md');
    const content = fs.readFileSync(fixturePath, 'utf-8');
    const fixed = fixMath(content);

    expect(fixed).toContain('$a + b = c$');
    expect(fixed).toMatch(/\n?\$\$/); // block promotion retained
  });

  it('can process directories recursively', () => {
    const tmpDir = fs.mkdtempSync(path.join(process.cwd(), 'tmp-math-'));
    const nestedDir = path.join(tmpDir, 'nested');
    const tmpFile = path.join(nestedDir, 'sample.md');
    fs.mkdirSync(nestedDir, { recursive: true });
    fs.copyFileSync(path.join(fixturesDir, 'fixture-beta.md'), tmpFile);

    runFixMath(tmpDir);

    const output = fs.readFileSync(tmpFile, 'utf-8');
    expect(output).toContain('$\\nabla \\times \\vec{F} = \\mu_0\\vec{J}$');

    fs.rmSync(tmpDir, { recursive: true, force: true });
  });

  it('normalizes invisible characters and trims inline math spacing', () => {
    const input = 'A\u00a0B $ x $ and \\$kept$';
    const fixed = fixMath(input);

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
    const fixed = fixMath(input);

    expect(fixed).toContain('```$kept$```');
    expect(fixed).toContain('$spaced$');
    expect(fixed).toContain('$$\nline\nwith newline\n$$');
    expect(fixed).toContain('$$unclosed');
  });

  it('throws when path is missing', () => {
    expect(() => runFixMath('/non-existent-path/file.md')).toThrow();
  });

  it('writes files when called directly with a file path', () => {
    const tmpFile = path.join(process.cwd(), 'tmp-math-file.md');
    fs.writeFileSync(tmpFile, 'Number $ spaced $ here');
    const logSpy = vi.spyOn(console, 'log').mockImplementation(() => {});

    runFixMath(tmpFile);

    const updated = fs.readFileSync(tmpFile, 'utf-8');
    expect(updated).toContain('$spaced$');

    fs.rmSync(tmpFile, { force: true });
    logSpy.mockRestore();
  });

  it('handles unclosed frontmatter and escaped inline dollars', () => {
    const text = '---\ntitle: test\ncontent without end $a \\$ b$';
    const fixed = fixMath(text);

    expect(fixed).toContain('$a \\$ b$');
  });

  it('keeps mixed fence markers within an open fence', () => {
    const content = '```js\ncode\n~~~\nstill code';
    const segments = splitCodeFences(content);

    expect(segments.some((s) => s.type === 'code' && s.content.includes('~~~'))).toBe(true);
  });

  it('cli runner shows usage when target is missing', () => {
    const exitSpy = vi.spyOn(process, 'exit').mockImplementation(() => {
      throw new Error('exit');
    }) as any;
    const errorSpy = vi.spyOn(console, 'error').mockImplementation(() => {});
    const argv = [process.argv[0], path.join(process.cwd(), 'scripts/fix-math.ts')];

    expect(() => runCli(argv)).toThrow('exit');

    exitSpy.mockRestore();
    errorSpy.mockRestore();
  });

  it('cli runner surfaces errors from runFixMath', () => {
    const exitSpy = vi.spyOn(process, 'exit').mockImplementation(() => {
      throw new Error('exit');
    }) as any;
    const errorSpy = vi.spyOn(console, 'error').mockImplementation(() => {});
    const argv = [process.argv[0], path.join(process.cwd(), 'scripts/fix-math.ts'), '/missing.md'];

    expect(() => runCli(argv)).toThrow('exit');

    exitSpy.mockRestore();
    errorSpy.mockRestore();
  });
});
