import { describe, expect, it } from 'vitest';
import type { CollectionEntry } from 'astro:content';
import {
  getAllTags,
  getPostsByTagSlug,
  getTagStats,
  normalizeTag,
  normalizeTags,
  slugifyTag,
} from '../../src/lib/content/tags';

const makePost = (slug: string, date: string, tags: string[]): CollectionEntry<'blog'> =>
  ({
    slug,
    id: slug,
    collection: 'blog',
    data: {
      title: slug,
      date: new Date(date),
      tags,
      status: 'published',
    },
    body: `${slug} body`,
    render: async () => ({ Content: () => null, headings: [] as any[] }),
  }) as unknown as CollectionEntry<'blog'>;

describe('tag utilities', () => {
  it('normalizes single tag', () => {
    expect(normalizeTag('  AI  ')).toBe('AI');
    expect(normalizeTag('')).toBe('');
    expect(normalizeTag('a'.repeat(40)).length).toBeLessThanOrEqual(32);
  });

  it('normalizes and deduplicates tag list', () => {
    expect(normalizeTags(['AI', ' ai ', 'ML'])).toEqual(['AI', 'ML']);
    expect(normalizeTags(['', '  '])).toEqual([]);
  });

  it('slugifies tags with chinese and ascii safely', () => {
    expect(slugifyTag('CUDA')).toBe('cuda');
    expect(slugifyTag('推理 优化')).toBe('推理-优化');
    expect(slugifyTag('AI Infra')).toBe('ai-infra');
  });

  it('computes tag stats sorted by count then name', () => {
    const posts = [
      makePost('a', '2024-01-01', ['AI', 'ML']),
      makePost('b', '2024-01-02', ['AI', '推理']),
    ];
    const stats = getTagStats(posts);
    expect(stats[0]).toMatchObject({ tag: 'AI', slug: 'ai', count: 2 });
    expect(stats.find((s) => s.tag === '推理')).toBeTruthy();
  });

  it('filters posts by tag slug', () => {
    const posts = [
      makePost('a', '2024-01-01', ['AI', 'ML']),
      makePost('b', '2024-01-02', ['AI', '推理']),
      makePost('c', '2024-01-03', ['Other']),
    ];
    const filtered = getPostsByTagSlug(posts, '推理');
    expect(filtered.map((p) => p.slug)).toEqual(['b']);
  });

  it('exposes all tags', () => {
    const posts = [makePost('a', '2024-01-01', ['AI']), makePost('b', '2024-01-02', ['ML'])];
    expect(getAllTags(posts).map((t) => t.tag)).toEqual(['AI', 'ML']);
  });
});
