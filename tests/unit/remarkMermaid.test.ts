/**
 * Tests for remarkMermaid plugin and mermaidCover utility
 */

import { describe, it, expect } from 'vitest';
import { unified } from 'unified';
import remarkParse from 'remark-parse';
import remarkStringify from 'remark-stringify';
import { VFile } from 'vfile';
import remarkMermaid, { mermaidImagePath } from '../../src/lib/markdown/remarkMermaid';
import { getMermaidCoverFromBody } from '../../src/lib/content/mermaidCover';
import type { Root, Image } from 'mdast';

// -------------------------------------------------------------------------
// mermaidImagePath — deterministic hash-based path generation
// -------------------------------------------------------------------------

describe('mermaidImagePath', () => {
  it('returns a path starting with /mermaid-generated/', () => {
    const path = mermaidImagePath('graph TD\nA-->B');
    expect(path).toMatch(/^\/mermaid-generated\//);
  });

  it('ends with .png', () => {
    const path = mermaidImagePath('graph TD\nA-->B');
    expect(path).toMatch(/\.png$/);
  });

  it('is deterministic — same code always yields the same path', () => {
    const code = 'sequenceDiagram\n  Alice->>Bob: Hello!';
    expect(mermaidImagePath(code)).toBe(mermaidImagePath(code));
  });

  it('is content-addressable — different code yields different paths', () => {
    expect(mermaidImagePath('graph TD\nA-->B')).not.toBe(mermaidImagePath('graph TD\nA-->C'));
  });

  it('is whitespace-insensitive (trims the code before hashing)', () => {
    const codeA = '  graph TD\n  A-->B  \n';
    const codeB = 'graph TD\n  A-->B';
    expect(mermaidImagePath(codeA)).toBe(mermaidImagePath(codeB));
  });
});

// -------------------------------------------------------------------------
// remarkMermaid — AST transformation
// -------------------------------------------------------------------------

async function processMarkdown(
  input: string,
  frontmatter?: Record<string, string>,
): Promise<{ tree: Root; mermaidCover?: string }> {
  const file = await unified()
    .use(remarkParse)
    .use(remarkMermaid)
    .use(remarkStringify)
    .process(input);

  // Set up the astro data the same way the Astro markdown renderer does
  // (we need to pre-populate so our plugin can read/write frontmatter)
  return {
    tree: unified().use(remarkParse).parse(input) as Root,
    mermaidCover: (file.data as any)?.astro?.frontmatter?.mermaidCover,
  };
}

describe('remarkMermaid plugin', () => {
  it('replaces a mermaid code block with an image node', async () => {
    const code = 'graph TD\n  A-->B';
    const input = `\`\`\`mermaid\n${code}\n\`\`\`\n`;

    // Run just the tree transform (no stringify) to inspect the AST
    const processor = unified().use(remarkParse).use(remarkMermaid);
    const tree = unified().use(remarkParse).parse(input) as Root;
    processor.run(tree);
    // The visit happens synchronously, so the tree is already modified
    // Find image node
    const imgNode = tree.children.find((c) => c.type === 'image') as Image | undefined;
    expect(imgNode).toBeDefined();
    expect(imgNode?.url).toBe(mermaidImagePath(code));
  });

  it('sets the image alt from the title meta', async () => {
    const code = 'graph TD\n  A-->B';
    const input = `\`\`\`mermaid title="My Chart"\n${code}\n\`\`\`\n`;

    const processor = unified().use(remarkParse).use(remarkMermaid);
    const tree = unified().use(remarkParse).parse(input) as Root;
    processor.run(tree);

    const imgNode = tree.children.find((c) => c.type === 'image') as Image | undefined;
    expect(imgNode?.alt).toBe('My Chart');
    expect(imgNode?.title).toBe('My Chart');
  });

  it('sets mermaidCover in file.data for the first mermaid block', async () => {
    const code = 'sequenceDiagram\n  Alice->>Bob: Hi';
    const input = `\`\`\`mermaid\n${code}\n\`\`\`\n`;

    const file = new VFile({ value: input });
    (file.data as any).astro = { frontmatter: {} };

    const processor = unified().use(remarkParse).use(remarkMermaid);
    const tree = unified().use(remarkParse).parse(input) as Root;
    await processor.run(tree, file);

    expect((file.data as any).astro.frontmatter.mermaidCover).toBe(mermaidImagePath(code));
  });

  it('does NOT overwrite an existing cover with mermaidCover', async () => {
    const code = 'graph TD\n  A-->B';
    const input = `\`\`\`mermaid\n${code}\n\`\`\`\n`;
    const existingCover = '/images/my-cover.jpg';

    const file = new VFile({ value: input });
    (file.data as any).astro = { frontmatter: { cover: existingCover } };

    const processor = unified().use(remarkParse).use(remarkMermaid);
    const tree = unified().use(remarkParse).parse(input) as Root;
    await processor.run(tree, file);

    expect((file.data as any).astro.frontmatter.mermaidCover).toBeUndefined();
  });

  it('leaves non-mermaid code blocks untouched', async () => {
    const input = '```js\nconsole.log("hi");\n```\n';

    const processor = unified().use(remarkParse).use(remarkMermaid);
    const tree = unified().use(remarkParse).parse(input) as Root;
    processor.run(tree);

    expect(tree.children[0].type).toBe('code');
    expect((tree.children[0] as any).lang).toBe('js');
  });

  it('handles multiple mermaid blocks in one document', async () => {
    const code1 = 'graph TD\n  A-->B';
    const code2 = 'graph LR\n  X-->Y';
    const input = `\`\`\`mermaid\n${code1}\n\`\`\`\n\n\`\`\`mermaid\n${code2}\n\`\`\`\n`;

    const processor = unified().use(remarkParse).use(remarkMermaid);
    const tree = unified().use(remarkParse).parse(input) as Root;
    processor.run(tree);

    const images = tree.children.filter((c) => c.type === 'image') as Image[];
    expect(images).toHaveLength(2);
    expect(images[0].url).toBe(mermaidImagePath(code1));
    expect(images[1].url).toBe(mermaidImagePath(code2));
  });
});

// -------------------------------------------------------------------------
// getMermaidCoverFromBody — raw markdown body scanner
// -------------------------------------------------------------------------

describe('getMermaidCoverFromBody', () => {
  it('returns undefined when there is no mermaid block', () => {
    const body = '# Hello\n\nSome text\n\n```js\nconsole.log("hi");\n```\n';
    expect(getMermaidCoverFromBody(body)).toBeUndefined();
  });

  it('returns the expected image path for a mermaid block', () => {
    const code = 'graph TD\n  A-->B';
    const body = `# Title\n\n\`\`\`mermaid\n${code}\n\`\`\`\n\nMore text.\n`;
    const result = getMermaidCoverFromBody(body);
    expect(result).toBeDefined();
    expect(result).toMatch(/^\/mermaid-generated\/.+\.png$/);
  });

  it('matches the path that mermaidImagePath generates for the same code', () => {
    const code = 'sequenceDiagram\n  Alice->>Bob: Hi';
    const body = `\`\`\`mermaid\n${code}\n\`\`\`\n`;
    const fromBody = getMermaidCoverFromBody(body);
    const direct = mermaidImagePath(code);
    expect(fromBody).toBe(direct);
  });

  it('uses the FIRST mermaid block when there are multiple', () => {
    const code1 = 'graph TD\n  A-->B';
    const code2 = 'graph TD\n  X-->Y';
    const body = `\`\`\`mermaid\n${code1}\n\`\`\`\n\n\`\`\`mermaid\n${code2}\n\`\`\`\n`;
    const result = getMermaidCoverFromBody(body);
    expect(result).toBe(mermaidImagePath(code1));
    expect(result).not.toBe(mermaidImagePath(code2));
  });

  it('ignores meta on the opening fence (e.g., title="…")', () => {
    const code = 'graph LR\n  A-->B';
    const body = `\`\`\`mermaid title="My diagram"\n${code}\n\`\`\`\n`;
    const result = getMermaidCoverFromBody(body);
    expect(result).toBe(mermaidImagePath(code));
  });

  it('handles empty body gracefully', () => {
    expect(getMermaidCoverFromBody('')).toBeUndefined();
  });

  it('handles body with only non-mermaid fences', () => {
    const body = '```python\nprint("hello")\n```\n';
    expect(getMermaidCoverFromBody(body)).toBeUndefined();
  });
});
