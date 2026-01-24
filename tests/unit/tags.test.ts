import { describe, it, expect } from 'vitest';
import {
  slugifyTag,
  buildTagIndex,
  getPostsByTagSlug,
  sortTags,
  type TagStats,
} from '../../src/lib/content/tags';
import type { CollectionEntry } from 'astro:content';

// Helper to create mock posts
function createMockPost(
  slug: string,
  title: string,
  tags: string[],
  date: Date,
): CollectionEntry<'blog'> {
  return {
    slug,
    id: `${slug}.md`,
    collection: 'blog',
    data: {
      title,
      tags,
      date,
      status: 'published' as const,
    },
    body: '',
    render: async () => ({
      Content: null as any,
      headings: [],
      remarkPluginFrontmatter: {},
    }),
  };
}

describe('slugifyTag', () => {
  it('converts basic strings to lowercase slugs', () => {
    expect(slugifyTag('AI Infra')).toBe('ai-infra');
    expect(slugifyTag('CUDA')).toBe('cuda');
    expect(slugifyTag('Machine Learning')).toBe('machine-learning');
  });

  it('handles Chinese characters', () => {
    // Chinese characters get encoded
    const result = slugifyTag('推理优化');
    expect(result).toBeTruthy();
    expect(result).toContain('%');
  });

  it('handles mixed Chinese and English', () => {
    const result = slugifyTag('AI 推理');
    // Should handle mixed content
    expect(result).toBeTruthy();
  });

  it('trims and collapses whitespace', () => {
    expect(slugifyTag('  Multiple   Spaces  ')).toBe('multiple-spaces');
    expect(slugifyTag('Tab\tTest')).toBe('tab-test');
  });

  it('handles special characters', () => {
    // Slashes and special chars
    const result = slugifyTag('推理优化/部署');
    expect(result).toBeTruthy();
  });

  it('handles empty and whitespace-only strings', () => {
    expect(slugifyTag('')).toBe('');
    expect(slugifyTag('   ')).toBe('');
    expect(slugifyTag('\t\n')).toBe('');
  });

  it('handles numbers and symbols', () => {
    expect(slugifyTag('C++')).toBe('c');
    expect(slugifyTag('Web3.0')).toBe('web30');
    expect(slugifyTag('Node.js')).toBe('nodejs');
  });

  it('is case-insensitive', () => {
    expect(slugifyTag('AI')).toBe(slugifyTag('ai'));
    expect(slugifyTag('CUDA')).toBe(slugifyTag('cuda'));
  });

  it('handles duplicate characters', () => {
    expect(slugifyTag('Test---Slug')).toBe('test-slug');
    expect(slugifyTag('Hello___World')).toBe('helloworld');
  });
});

describe('buildTagIndex', () => {
  it('builds empty index for no posts', () => {
    const { allTags, tagMap } = buildTagIndex([]);
    expect(allTags).toEqual([]);
    expect(Object.keys(tagMap)).toHaveLength(0);
  });

  it('aggregates tags correctly', () => {
    const posts = [
      createMockPost('post1', 'Post 1', ['AI', 'ML'], new Date('2024-01-01')),
      createMockPost('post2', 'Post 2', ['AI', 'Deep Learning'], new Date('2024-01-02')),
      createMockPost('post3', 'Post 3', ['ML'], new Date('2024-01-03')),
    ];

    const { allTags, tagMap } = buildTagIndex(posts);

    expect(allTags).toHaveLength(3);
    expect(Object.keys(tagMap)).toHaveLength(3);

    // Check AI tag
    const aiTag = allTags.find((t) => t.name === 'AI');
    expect(aiTag).toBeDefined();
    expect(aiTag!.count).toBe(2);
    expect(aiTag!.slug).toBe('ai');

    // Check ML tag
    const mlTag = allTags.find((t) => t.name === 'ML');
    expect(mlTag).toBeDefined();
    expect(mlTag!.count).toBe(2);
  });

  it('sorts tags by count descending, then name ascending', () => {
    const posts = [
      createMockPost('post1', 'Post 1', ['Zebra', 'Alpha'], new Date('2024-01-01')),
      createMockPost('post2', 'Post 2', ['Beta', 'Alpha'], new Date('2024-01-02')),
      createMockPost('post3', 'Post 3', ['Alpha'], new Date('2024-01-03')),
    ];

    const { allTags } = buildTagIndex(posts);

    // Alpha appears 3 times, Beta 1 time, Zebra 1 time
    expect(allTags[0].name).toBe('Alpha');
    expect(allTags[0].count).toBe(3);

    // Beta and Zebra both have 1, so alphabetical order
    expect(allTags[1].name).toBe('Beta');
    expect(allTags[2].name).toBe('Zebra');
  });

  it('calculates latest date correctly', () => {
    const posts = [
      createMockPost('post1', 'Post 1', ['AI'], new Date('2024-01-01')),
      createMockPost('post2', 'Post 2', ['AI'], new Date('2024-01-05')),
      createMockPost('post3', 'Post 3', ['AI'], new Date('2024-01-03')),
    ];

    const { allTags } = buildTagIndex(posts);
    const aiTag = allTags.find((t) => t.name === 'AI');

    expect(aiTag!.latestDate).toEqual(new Date('2024-01-05'));
  });

  it('handles tag name case sensitivity', () => {
    const posts = [
      createMockPost('post1', 'Post 1', ['AI'], new Date('2024-01-01')),
      createMockPost('post2', 'Post 2', ['ai'], new Date('2024-01-02')),
    ];

    const { allTags, tagMap } = buildTagIndex(posts);

    // Should treat 'AI' and 'ai' as different tags
    expect(allTags).toHaveLength(2);
    expect(allTags.find((t) => t.name === 'AI')).toBeDefined();
    expect(allTags.find((t) => t.name === 'ai')).toBeDefined();
  });

  it('handles disambiguation for duplicate slugs', () => {
    const posts = [createMockPost('post1', 'Post 1', ['C++', 'C  '], new Date('2024-01-01'))];

    const { allTags, tagMap } = buildTagIndex(posts);

    // Both 'C++' and 'C  ' might normalize to 'c'
    // First one should get 'c', second should get 'c-2'
    expect(allTags).toHaveLength(2);

    const slugs = allTags.map((t) => t.slug);
    // Check that slugs are unique
    expect(new Set(slugs).size).toBe(allTags.length);
  });

  it('trims empty tags', () => {
    const posts = [
      createMockPost('post1', 'Post 1', ['AI', '', '  ', 'ML'], new Date('2024-01-01')),
    ];

    const { allTags } = buildTagIndex(posts);

    // Should only have 'AI' and 'ML', empty strings trimmed
    expect(allTags).toHaveLength(2);
    expect(allTags.map((t) => t.name)).toEqual(expect.arrayContaining(['AI', 'ML']));
  });

  it('sorts posts within each tag by date descending', () => {
    const posts = [
      createMockPost('post1', 'Post 1', ['AI'], new Date('2024-01-01')),
      createMockPost('post2', 'Post 2', ['AI'], new Date('2024-01-05')),
      createMockPost('post3', 'Post 3', ['AI'], new Date('2024-01-03')),
    ];

    const { tagMap } = buildTagIndex(posts);
    const aiPosts = tagMap['ai'].posts;

    expect(aiPosts[0].slug).toBe('post2'); // 2024-01-05
    expect(aiPosts[1].slug).toBe('post3'); // 2024-01-03
    expect(aiPosts[2].slug).toBe('post1'); // 2024-01-01
  });
});

describe('getPostsByTagSlug', () => {
  it('returns posts for valid tag slug', () => {
    const posts = [
      createMockPost('post1', 'Post 1', ['AI'], new Date('2024-01-01')),
      createMockPost('post2', 'Post 2', ['ML'], new Date('2024-01-02')),
    ];

    const { tagMap } = buildTagIndex(posts);
    const aiPosts = getPostsByTagSlug(tagMap, 'ai');

    expect(aiPosts).toBeDefined();
    expect(aiPosts).toHaveLength(1);
    expect(aiPosts![0].slug).toBe('post1');
  });

  it('returns undefined for invalid tag slug', () => {
    const posts = [createMockPost('post1', 'Post 1', ['AI'], new Date('2024-01-01'))];

    const { tagMap } = buildTagIndex(posts);
    const result = getPostsByTagSlug(tagMap, 'nonexistent');

    expect(result).toBeUndefined();
  });
});

describe('sortTags', () => {
  const mockTags: TagStats[] = [
    { name: 'Zebra', slug: 'zebra', count: 1, latestDate: new Date('2024-01-01') },
    { name: 'Alpha', slug: 'alpha', count: 3, latestDate: new Date('2024-01-05') },
    { name: 'Beta', slug: 'beta', count: 2, latestDate: new Date('2024-01-10') },
  ];

  it('sorts by count descending', () => {
    const sorted = sortTags(mockTags, 'count');
    expect(sorted[0].name).toBe('Alpha'); // count: 3
    expect(sorted[1].name).toBe('Beta'); // count: 2
    expect(sorted[2].name).toBe('Zebra'); // count: 1
  });

  it('sorts by name ascending', () => {
    const sorted = sortTags(mockTags, 'name');
    expect(sorted[0].name).toBe('Alpha');
    expect(sorted[1].name).toBe('Beta');
    expect(sorted[2].name).toBe('Zebra');
  });

  it('sorts by recent date descending', () => {
    const sorted = sortTags(mockTags, 'recent');
    expect(sorted[0].name).toBe('Beta'); // 2024-01-10
    expect(sorted[1].name).toBe('Alpha'); // 2024-01-05
    expect(sorted[2].name).toBe('Zebra'); // 2024-01-01
  });

  it('does not mutate original array', () => {
    const original = [...mockTags];
    sortTags(mockTags, 'name');
    expect(mockTags).toEqual(original);
  });
});
