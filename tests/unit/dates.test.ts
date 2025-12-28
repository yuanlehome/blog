import { describe, expect, it } from 'vitest';
import type { CollectionEntry } from 'astro:content';
import { getDisplayDates } from '../../src/utils/dates';

type BlogData = CollectionEntry<'blog'>['data'];

const baseData = (): BlogData =>
  ({
    title: 'Sample',
    date: new Date('2024-01-01'),
    tags: [],
    status: 'published',
  }) as BlogData;

const withData = (overrides: Partial<BlogData>): BlogData =>
  ({
    ...baseData(),
    ...overrides,
  }) as BlogData;

describe('date display helpers', () => {
  it('shows published and updated when updated is later', () => {
    const dates = getDisplayDates(withData({ updated: new Date('2024-01-05') }));

    expect(dates.publishedLabel).toBe('2024-01-01');
    expect(dates.updatedLabel).toBe('2024-01-05');
  });

  it('hides updated when it matches the published day', () => {
    const dates = getDisplayDates(withData({ updated: new Date('2024-01-01T12:00:00Z') }));

    expect(dates.updatedLabel).toBe('');
  });

  it('uses last edited time as a fallback', () => {
    const dates = getDisplayDates(
      withData({
        updated: undefined,
        updatedAt: undefined,
        lastmod: undefined,
        lastEditedTime: new Date('2024-02-10'),
      }),
    );

    expect(dates.updatedLabel).toBe('2024-02-10');
  });
});
