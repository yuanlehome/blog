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
});
