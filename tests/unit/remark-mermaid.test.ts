import { mkdtemp, mkdir, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { afterEach, describe, expect, it, vi } from 'vitest';
import type { Root } from 'mdast';
import { VFile } from 'vfile';
import remarkMermaid, {
  mermaidImagePath,
  parseMermaidOptions,
  resolveMermaidSlug,
} from '../../src/lib/markdown/remarkMermaid';

const cleanupDirs: string[] = [];

afterEach(async () => {
  await Promise.all(
    cleanupDirs
      .splice(0)
      .map(async (dir) =>
        import('node:fs/promises').then((fs) => fs.rm(dir, { recursive: true, force: true })),
      ),
  );
});

describe('remarkMermaid', () => {
  it('emits base-prefixed image src when rendered assets exist', async () => {
    const rootDir = await mkdtemp(join(tmpdir(), 'remark-mermaid-'));
    cleanupDirs.push(rootDir);
    const publicDir = join(rootDir, 'public');
    const filePath = join(rootDir, 'src/content/blog/notion/test.md');
    const code = 'graph TD\nA-->B';
    const slug = resolveMermaidSlug(filePath);
    const options = parseMermaidOptions(undefined);

    const lightPath = mermaidImagePath(code, options, slug, 'light').replace(/^\//, '');
    const darkPath = mermaidImagePath(code, options, slug, 'dark').replace(/^\//, '');
    await mkdir(join(publicDir, lightPath.replace(/\/[^/]+$/, '')), { recursive: true });
    await writeFile(join(publicDir, lightPath), '<svg></svg>');
    await writeFile(join(publicDir, darkPath), '<svg></svg>');

    const tree: Root = {
      type: 'root',
      children: [{ type: 'code', lang: 'mermaid', value: code }],
    };

    const file = new VFile({ path: filePath, cwd: rootDir, data: {} });
    const transformer = (remarkMermaid as any)({ base: '/blog/' }) as any;
    transformer(tree, file, () => {});

    expect(tree.children[0].type).toBe('html');
    const value = (tree.children[0] as any).value as string;
    expect(value).toContain('src="/blog/generated/mermaid/notion/');
    expect(value).toContain('.light.svg');
    expect(value).toContain('.dark.svg');
  });

  it('keeps original code block when rendered assets are missing', () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const tree: Root = {
      type: 'root',
      children: [{ type: 'code', lang: 'mermaid', value: 'graph TD\nA-->B' }],
    };
    const file = new VFile({
      path: '/tmp/src/content/blog/notion/missing.md',
      cwd: '/tmp',
      data: {},
    });

    const transformer = (remarkMermaid as any)({ base: '/blog/' }) as any;
    transformer(tree, file, () => {});

    expect(tree.children[0].type).toBe('code');
    expect(warn).toHaveBeenCalledOnce();
  });
});
