import { describe, expect, it, vi } from 'vitest';
import type { CollectionEntry } from 'astro:content';
import { findPrevNext, findRelated, groupByYearMonth } from '../../src/lib/content/posts';

vi.mock('astro:content', () => ({
  getCollection: vi.fn(),
}));

const makePost = (slug: string, date: string, tags: string[] = ['a']): CollectionEntry<'blog'> =>
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

describe('posts utils', () => {
  it('filters and sorts published posts', async () => {
    const posts = [
      {
        ...makePost('b', '2024-01-01'),
        data: { ...makePost('b', '2024-01-01').data, status: 'draft' },
      },
      makePost('a', '2024-03-01'),
      makePost('c', '2024-02-01'),
    ];

    const { getCollection } = await import('astro:content');
    (getCollection as any).mockResolvedValue(posts);

    const { getPublishedPosts } = await import('../../src/lib/content/posts');
    const result = await getPublishedPosts();

    expect(getCollection).toHaveBeenCalledOnce();
    expect(result.map((p) => p.slug)).toEqual(['a', 'c']);
  });

  it('finds previous and next posts', () => {
    const posts = [
      makePost('first', '2024-01-01'),
      makePost('second', '2024-02-01'),
      makePost('third', '2024-03-01'),
    ];
    const { prev, next } = findPrevNext(posts, 'second');

    expect(prev?.slug).toBe('first');
    expect(next?.slug).toBe('third');
  });

  it('handles edges and missing slugs when finding prev/next', () => {
    const posts = [makePost('only', '2024-01-01'), makePost('second', '2024-02-01')];
    const first = findPrevNext(posts, 'only');
    const missing = findPrevNext(posts, 'absent');

    expect(first.prev).toBeUndefined();
    expect(first.next?.slug).toBe('second');
    expect(missing.prev).toBeUndefined();
    expect(missing.next).toBeUndefined();
  });

  it('groups posts by year and month', () => {
    const posts = [
      makePost('jan', '2024-01-15'),
      makePost('feb', '2024-02-01'),
      makePost('jan-old', '2024-01-01'),
    ];
    const grouped = groupByYearMonth(posts);

    expect(grouped[0].key).toBe('2024-02');
    expect(grouped[1].posts.map((p) => p.slug)).toEqual(['jan', 'jan-old']);
  });

  it('finds related posts by shared tags', () => {
    const posts = [
      makePost('first', '2024-01-01', ['js', 'astro']),
      makePost('second', '2024-02-01', ['astro']),
      makePost('third', '2024-02-15', ['rust']),
    ];

    const related = findRelated(posts, posts[0], 2);
    expect(related.map((p) => p.slug)).toEqual(['second']);
  });
});
