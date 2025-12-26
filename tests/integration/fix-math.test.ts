import { describe, expect, it } from 'vitest';
import fs from 'fs';
import path from 'path';
import { fixMath, runFixMath } from '../../scripts/fix-math';

const fixturesDir = path.join(process.cwd(), 'tests/fixtures/posts');

describe('fix-math script', () => {
  it('normalizes inline and block math tokens', () => {
    const fixturePath = path.join(fixturesDir, 'fixture-alpha.md');
    const content = fs.readFileSync(fixturePath, 'utf-8');
    const fixed = fixMath(content);

    expect(fixed).toContain('$a + b = c$');
    expect(fixed).toMatch(/\\n?\$\$/); // block promotion retained
  });

  it('can process directories recursively', () => {
    const tmpDir = fs.mkdtempSync(path.join(process.cwd(), 'tmp-math-'));
    const tmpFile = path.join(tmpDir, 'sample.md');
    fs.copyFileSync(path.join(fixturesDir, 'fixture-beta.md'), tmpFile);

    runFixMath(tmpDir);

    const output = fs.readFileSync(tmpFile, 'utf-8');
    expect(output).toContain('$\\nabla \\times \\vec{F} = \\mu_0\\vec{J}$');

    fs.rmSync(tmpDir, { recursive: true, force: true });
  });
});
