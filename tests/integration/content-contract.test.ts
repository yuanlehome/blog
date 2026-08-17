import { describe, it, expect, vi } from 'vitest';

// Mock astro:content so that src/content/config can be imported in Vitest
vi.mock('astro:content', async () => {
  const zod = await import('zod');
  return {
    z: zod.z,
    defineCollection: (opts: { schema: any }) => ({ schema: opts.schema }),
  };
});

describe('content frontmatter contracts', () => {
  async function getBlogSchema() {
    const { collections } = await import('../../src/content/config');
    // In tests we mock defineCollection to return an object with a schema property
    return (collections as any).blog.schema as { parse: (value: unknown) => unknown };
  }

  async function getScalingBookSchema() {
    const { collections } = await import('../../src/content/config');
    return (collections as any).scalingBook.schema as { parse: (value: unknown) => unknown };
  }

  it('accepts Notion-synced frontmatter shape', async () => {
    const schema = await getBlogSchema();

    const notionFrontmatter = {
      title: 'Test Notion Post',
      date: '2024-01-01',
      updated: '2024-01-02',
      lastEditedTime: '2024-01-02T00:00:00.000Z',
      tags: ['notion', 'sync'],
      cover: '/images/notion/test-post/cover.jpg',
      status: 'published',
      notion: { id: 'page-123' },
      source: 'notion',
    };

    expect(() => schema.parse(notionFrontmatter)).not.toThrow();
  });

  it('accepts imported-article frontmatter shape', async () => {
    const schema = await getBlogSchema();

    const importedFrontmatter = {
      title: 'Imported Article',
      date: '2024-02-10',
      updated: '2024-02-12',
      tags: ['imported', 'external'],
      status: 'published',
      source_url: 'https://example.com/article',
      source_author: 'Example Author',
      imported_at: '2024-02-12T10:00:00.000Z',
      source: {
        title: 'Example Site',
        url: 'https://example.com/article',
      },
    };

    expect(() => schema.parse(importedFrontmatter)).not.toThrow();
  });

  it('rejects frontmatter missing required title field', async () => {
    const schema = await getBlogSchema();

    const invalidFrontmatter = {
      // title is missing
      date: '2024-03-01',
      tags: [],
      status: 'published',
    };

    expect(() => schema.parse(invalidFrontmatter)).toThrow();
  });

  it('accepts the dedicated Scaling Book chapter frontmatter', async () => {
    const schema = await getScalingBookSchema();

    const chapterFrontmatter = {
      title: 'Roofline 模型详解',
      description: '从计算与带宽两个方向估算算子性能。',
      chapter: 1,
      order: 1,
      part: 1,
      partTitle: 'Roofline 模型详解',
      sourcePath: 'docs/part1.md',
      sourceCommit: '44109cacac9c5a9809a81c68ae4d45d7d2632ea6',
    };

    expect(() => schema.parse(chapterFrontmatter)).not.toThrow();
  });
});
