import { describe, expect, it } from 'vitest';
import { slugFromTitle as deriveSlug, ensureUniqueSlug, normalizeSlug } from '../../src/lib/slug';

describe('slug utilities', () => {
  it('derives slug from explicit value, title, or fallback', () => {
    expect(
      deriveSlug({
        explicitSlug: 'Custom Slug',
        title: 'Ignored',
        fallbackId: 'id',
      }),
    ).toBe('custom-slug');

    expect(
      deriveSlug({
        explicitSlug: '',
        title: 'Hello World!',
        fallbackId: 'id',
      }),
    ).toBe('hello-world');

    expect(
      deriveSlug({
        explicitSlug: '',
        title: '',
        fallbackId: '123',
      }),
    ).toBe('123');
  });

  it('ensures unique slugs using owner id hash', () => {
    const used = new Map<string, string>();
    const first = ensureUniqueSlug('post', 'owner-a', used);
    expect(first).toBe('post');

    const second = ensureUniqueSlug('post', 'owner-b', used);
    expect(second.startsWith('post-')).toBe(true);
    expect(second).not.toBe(first);

    const third = ensureUniqueSlug('post', 'owner-b', used);
    expect(third).toBe(second);
  });

  it('resolves repeated conflicts by incrementing suffix', () => {
    const used = new Map<string, string>();
    used.set('post', 'owner-a');
    const firstConflict = ensureUniqueSlug('post', 'owner-c', used);
    used.set(firstConflict, 'other-owner');
    const resolved = ensureUniqueSlug('post', 'owner-c', used);
    expect(resolved).not.toBe(firstConflict);
    expect(resolved.startsWith('post-')).toBe(true);
  });

  it('normalizes slugs consistently', () => {
    expect(normalizeSlug('Hello World')).toBe('hello-world');
    expect(normalizeSlug('Café 123')).toBe('cafe-123');
  });
});
