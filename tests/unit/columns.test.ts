import { describe, expect, it } from 'vitest';
import {
  COLUMN_CATALOG,
  COLUMNS_INDEX_PATH,
  findColumnBySlug,
} from '../../src/lib/content/columns';

describe('columns catalog', () => {
  it('uses a dedicated index and unique internal routes', () => {
    expect(COLUMNS_INDEX_PATH).toBe('/columns/');
    expect(new Set(COLUMN_CATALOG.map((column) => column.slug)).size).toBe(COLUMN_CATALOG.length);
    expect(new Set(COLUMN_CATALOG.map((column) => column.href)).size).toBe(COLUMN_CATALOG.length);

    for (const column of COLUMN_CATALOG) {
      expect(column.href).toMatch(/^\/[a-z0-9-]+\/$/);
      expect(column.coverWidth).toBeGreaterThan(0);
      expect(column.coverHeight).toBeGreaterThan(0);
    }
  });

  it('registers Scaling Book as a column', () => {
    expect(findColumnBySlug('scaling-book')).toEqual(
      expect.objectContaining({
        href: '/scaling-book/',
        shortTitle: 'Scaling Book 中文版',
      }),
    );
  });
});
