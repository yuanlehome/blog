import { describe, it, expect } from 'vitest';
import {
  normalizeSlug,
  slugFromTitle,
  slugFromFileStem,
  shortHash,
  ensureUniqueSlug,
  ensureUniqueSlugs,
  normalizeBase,
  buildPostUrl,
  deriveSlug,
} from '../../src/lib/slug';

describe('normalizeSlug', () => {
  it('converts basic strings to lowercase slugs', () => {
    expect(normalizeSlug('Hello World')).toBe('hello-world');
    expect(normalizeSlug('UPPERCASE')).toBe('uppercase');
    expect(normalizeSlug('MixedCase')).toBe('mixedcase');
  });

  it('handles Chinese characters', () => {
    // Note: slugify with strict:true removes non-Latin characters
    // This is intentional for URL safety. Chinese titles should use fallback or explicit slugs.
    expect(normalizeSlug('你好世界')).toBe('');
    expect(normalizeSlug('中文测试')).toBe('');
    // Mixed content keeps Latin parts
    expect(normalizeSlug('你好 Hello 世界')).toBe('hello');
  });

  it('handles emoji', () => {
    expect(normalizeSlug('😀 Emoji Test')).toBe('emoji-test');
    expect(normalizeSlug('🚀 Rocket Launch 🎉')).toBe('rocket-launch');
  });

  it('handles mixed scripts', () => {
    // Non-Latin characters are removed with strict:true
    expect(normalizeSlug('Hello 世界 World')).toBe('hello-world');
    expect(normalizeSlug('Test 测试 123')).toBe('test-123');
  });

  it('collapses consecutive spaces and special characters', () => {
    expect(normalizeSlug('Hello   World')).toBe('hello-world');
    expect(normalizeSlug('Test---Slug')).toBe('test-slug');
    // Note: underscores are treated as word separators but removed, not converted to hyphens
    expect(normalizeSlug('Under_score_test')).toBe('underscoretest');
    expect(normalizeSlug('under-score-test')).toBe('under-score-test'); // Hyphens preserved
  });

  it('removes special characters', () => {
    expect(normalizeSlug('Hello! World?')).toBe('hello-world');
    // Special chars converted to words with strict:true
    expect(normalizeSlug('Test@#$%Slug')).toBe('testdollarpercentslug');
    expect(normalizeSlug('Price: $100')).toBe('price-dollar100');
  });

  it('handles empty and whitespace-only strings', () => {
    expect(normalizeSlug('')).toBe('');
    expect(normalizeSlug('   ')).toBe('');
    expect(normalizeSlug('\t\n')).toBe('');
  });

  it('handles leading and trailing special characters', () => {
    expect(normalizeSlug('---slug---')).toBe('slug');
    expect(normalizeSlug('__test__')).toBe('test');
    expect(normalizeSlug('!!!important!!!')).toBe('important');
  });

  it('handles numbers and alphanumeric combinations', () => {
    expect(normalizeSlug('2024-01-01')).toBe('2024-01-01');
    // Dots are removed, not converted
    expect(normalizeSlug('v1.2.3')).toBe('v123');
    expect(normalizeSlug('v1-2-3')).toBe('v1-2-3'); // Hyphens preserved
    expect(normalizeSlug('Post123')).toBe('post123');
  });
});

describe('slugFromTitle', () => {
  it('uses explicit slug if provided', () => {
    expect(
      slugFromTitle({
        explicitSlug: 'custom-slug',
        title: 'My Title',
        fallbackId: 'page-123',
      }),
    ).toBe('custom-slug');
  });

  it('uses title if explicit slug is not provided', () => {
    expect(
      slugFromTitle({
        title: 'My Title',
        fallbackId: 'page-123',
      }),
    ).toBe('my-title');
  });

  it('uses fallback ID if title is empty', () => {
    expect(
      slugFromTitle({
        title: '',
        fallbackId: 'page-123',
      }),
    ).toBe('page-123');

    expect(
      slugFromTitle({
        fallbackId: 'default-slug',
      }),
    ).toBe('default-slug');
  });

  it('ignores null or empty explicit slug', () => {
    expect(
      slugFromTitle({
        explicitSlug: null,
        title: 'My Title',
      }),
    ).toBe('my-title');

    expect(
      slugFromTitle({
        explicitSlug: '',
        title: 'My Title',
      }),
    ).toBe('my-title');
  });

  it('handles all options empty', () => {
    expect(
      slugFromTitle({
        explicitSlug: '',
        title: '',
        fallbackId: '',
      }),
    ).toBe('');
  });

  it('normalizes all inputs', () => {
    expect(
      slugFromTitle({
        explicitSlug: 'CUSTOM SLUG',
        title: 'ignored',
      }),
    ).toBe('custom-slug');

    expect(
      slugFromTitle({
        title: 'My Title!!!',
      }),
    ).toBe('my-title');
  });
});

describe('slugFromFileStem', () => {
  it('normalizes file stems', () => {
    expect(slugFromFileStem('hello-world')).toBe('hello-world');
    expect(slugFromFileStem('Hello World')).toBe('hello-world');
    expect(slugFromFileStem('UPPERCASE')).toBe('uppercase');
  });

  it('handles date-prefixed filenames', () => {
    expect(slugFromFileStem('2024-01-01-post')).toBe('2024-01-01-post');
    // Dots are removed
    expect(slugFromFileStem('2024.01.01 Post')).toBe('20240101-post');
  });

  it('handles special characters in filenames', () => {
    // Underscores are removed
    expect(slugFromFileStem('my_file_name')).toBe('myfilename');
    expect(slugFromFileStem('my-file-name')).toBe('my-file-name'); // Hyphens preserved
    expect(slugFromFileStem('test@file#name')).toBe('testfilename');
  });
});

describe('shortHash', () => {
  it('generates consistent hashes', () => {
    const hash1 = shortHash('test-input');
    const hash2 = shortHash('test-input');
    expect(hash1).toBe(hash2);
  });

  it('generates different hashes for different inputs', () => {
    const hash1 = shortHash('input1');
    const hash2 = shortHash('input2');
    expect(hash1).not.toBe(hash2);
  });

  it('respects length parameter', () => {
    expect(shortHash('test', 4)).toHaveLength(4);
    expect(shortHash('test', 8)).toHaveLength(8);
    expect(shortHash('test', 12)).toHaveLength(12);
  });

  it('default length is 6', () => {
    expect(shortHash('test')).toHaveLength(6);
  });
});

describe('ensureUniqueSlug', () => {
  it('returns original slug if not used', () => {
    const used = new Map();
    expect(ensureUniqueSlug('my-slug', 'id1', used)).toBe('my-slug');
    expect(used.get('my-slug')).toBe('id1');
  });

  it('returns original slug if owned by same entity', () => {
    const used = new Map([['my-slug', 'id1']]);
    expect(ensureUniqueSlug('my-slug', 'id1', used)).toBe('my-slug');
    expect(used.get('my-slug')).toBe('id1');
  });

  it('generates unique slug with hash suffix on conflict', () => {
    const used = new Map([['my-slug', 'id1']]);
    const result = ensureUniqueSlug('my-slug', 'id2', used);

    expect(result).toMatch(/^my-slug-[a-f0-9]{6}$/);
    expect(used.get(result)).toBe('id2');
  });

  it('handles multiple conflicts with counter', () => {
    const used = new Map();

    const slug1 = ensureUniqueSlug('post', 'id1', used);
    expect(slug1).toBe('post');

    const slug2 = ensureUniqueSlug('post', 'id2', used);
    expect(slug2).toMatch(/^post-[a-f0-9]{6}$/);

    // Simulate conflict with hash-based slug
    const slug3 = ensureUniqueSlug('post', 'id3', used);

    // Should get different result since hash conflicts
    // (In real scenario, hash collision is extremely rare)
    expect(slug3).not.toBe(slug1);
    expect(slug3).not.toBe(slug2);
  });

  it('handles empty desired slug', () => {
    const used = new Map();
    const result = ensureUniqueSlug('', 'page-123', used);
    expect(result).toBe('page-123');
  });
});

describe('ensureUniqueSlugs', () => {
  it('handles batch with no conflicts', () => {
    const items = [
      { id: '1', slug: 'post-1' },
      { id: '2', slug: 'post-2' },
      { id: '3', slug: 'post-3' },
    ];

    const result = ensureUniqueSlugs(items);
    expect(result.get('1')).toBe('post-1');
    expect(result.get('2')).toBe('post-2');
    expect(result.get('3')).toBe('post-3');
  });

  it('resolves conflicts in batch', () => {
    const items = [
      { id: 'id1', slug: 'post' },
      { id: 'id2', slug: 'post' },
      { id: 'id3', slug: 'article' },
    ];

    const result = ensureUniqueSlugs(items);
    expect(result.get('id1')).toBe('post');
    expect(result.get('id2')).toMatch(/^post-[a-f0-9]{6}$/);
    expect(result.get('id3')).toBe('article');
  });

  it('handles three-way conflict', () => {
    const items = [
      { id: 'id1', slug: 'post' },
      { id: 'id2', slug: 'post' },
      { id: 'id3', slug: 'post' },
    ];

    const result = ensureUniqueSlugs(items);
    const values = Array.from(result.values());
    expect(values[0]).toBe('post');
    expect(values[1]).toMatch(/^post-[a-f0-9]{6}$/);
    expect(values[2]).toMatch(/^post-[a-f0-9]{6}(-\d+)?$/);

    // All should be unique
    expect(new Set(values).size).toBe(3);
  });

  it('handles empty batch', () => {
    const result = ensureUniqueSlugs([]);
    expect(result.size).toBe(0);
  });
});

describe('normalizeBase', () => {
  it('adds trailing slash if missing', () => {
    expect(normalizeBase('/blog')).toBe('/blog/');
    expect(normalizeBase('/path/to/blog')).toBe('/path/to/blog/');
  });

  it('preserves trailing slash if present', () => {
    expect(normalizeBase('/blog/')).toBe('/blog/');
    expect(normalizeBase('/')).toBe('/');
  });

  it('handles empty string', () => {
    expect(normalizeBase('')).toBe('/');
  });
});

describe('buildPostUrl', () => {
  it('builds URL with default base', () => {
    // Note: Default base depends on siteBase from config
    // We test with explicit base to avoid environment dependencies
    const url = buildPostUrl('my-post', '/blog/');
    expect(url).toBe('/blog/my-post/');
  });

  it('builds URL with custom base', () => {
    expect(buildPostUrl('my-post', '/custom/')).toBe('/custom/my-post/');
    expect(buildPostUrl('my-post', '/custom')).toBe('/custom/my-post/');
  });

  it('handles root base', () => {
    expect(buildPostUrl('my-post', '/')).toBe('/my-post/');
  });

  it('removes trailing slash from slug', () => {
    expect(buildPostUrl('my-post/', '/blog/')).toBe('/blog/my-post/');
  });

  it('handles complex slugs', () => {
    expect(buildPostUrl('2024-01-01-post', '/blog/')).toBe('/blog/2024-01-01-post/');
    expect(buildPostUrl('hello-world-123', '/blog/')).toBe('/blog/hello-world-123/');
  });

  it('normalizes base without trailing slash', () => {
    expect(buildPostUrl('post', '/blog')).toBe('/blog/post/');
  });
});

describe('deriveSlug (legacy)', () => {
  it('works as alias for slugFromTitle', () => {
    expect(
      deriveSlug({
        explicitSlug: 'custom',
        title: 'My Title',
        fallbackId: 'id-123',
      }),
    ).toBe('custom');

    expect(
      deriveSlug({
        title: 'My Title',
        fallbackId: 'id-123',
      }),
    ).toBe('my-title');

    expect(
      deriveSlug({
        fallbackId: 'id-123',
      }),
    ).toBe('id-123');
  });
});

describe('Edge Cases and Integration', () => {
  it('handles full workflow: title → slug → unique → URL', () => {
    // Step 1: Generate slug from title (non-Latin chars removed)
    const slug = slugFromTitle({
      title: '你好 World! 测试 Post 🚀',
      fallbackId: 'page-123',
    });
    expect(slug).toBe('world-post');

    // Step 2: Ensure uniqueness
    const used = new Map();
    const uniqueSlug = ensureUniqueSlug(slug, 'page-123', used);
    expect(uniqueSlug).toBe('world-post');

    // Step 3: Build URL
    const url = buildPostUrl(uniqueSlug, '/blog/');
    expect(url).toBe('/blog/world-post/');
  });

  it('handles Chinese-only titles with fallback', () => {
    // Pure Chinese title falls back to ID
    const slug = slugFromTitle({
      title: '你好世界',
      fallbackId: 'page-123',
    });
    expect(slug).toBe('page-123');

    const url = buildPostUrl(slug, '/blog/');
    expect(url).toBe('/blog/page-123/');
  });

  it('handles Notion-style workflow with conflicts', () => {
    const pages = [
      { id: 'notion-1', title: 'My Post' },
      { id: 'notion-2', title: 'My Post' },
      { id: 'notion-3', title: 'Another Post' },
    ];

    const items = pages.map((p) => ({
      id: p.id,
      slug: slugFromTitle({ title: p.title, fallbackId: p.id }),
    }));

    const uniqueSlugs = ensureUniqueSlugs(items);

    expect(uniqueSlugs.get('notion-1')).toBe('my-post');
    expect(uniqueSlugs.get('notion-2')).toMatch(/^my-post-[a-f0-9]{6}$/);
    expect(uniqueSlugs.get('notion-3')).toBe('another-post');

    // Build URLs
    const urls = Array.from(uniqueSlugs.values()).map((slug) => buildPostUrl(slug, '/blog/'));
    expect(urls[0]).toBe('/blog/my-post/');
    expect(urls[1]).toMatch(/^\/blog\/my-post-[a-f0-9]{6}\/$/);
    expect(urls[2]).toBe('/blog/another-post/');
  });

  it('handles URL import workflow', () => {
    // Simulate importing from external URL
    const title = 'Inside NVIDIA GPUs: Anatomy of High-Performance MatMul Kernels';
    const slug = slugFromTitle({ title });
    expect(slug).toBe('inside-nvidia-gpus-anatomy-of-high-performance-matmul-kernels');

    const url = buildPostUrl(slug, '/blog/');
    expect(url).toBe('/blog/inside-nvidia-gpus-anatomy-of-high-performance-matmul-kernels/');
  });

  it('handles local markdown workflow', () => {
    // Local file: hello-world.md
    const slug = slugFromFileStem('hello-world');
    expect(slug).toBe('hello-world');

    const url = buildPostUrl(slug, '/blog/');
    expect(url).toBe('/blog/hello-world/');
  });
});
