import { describe, expect, it } from 'vitest';
import {
  COLUMN_CATALOG,
  COLUMNS_INDEX_PATH,
  MODERN_GPU_PROGRAMMING_ZH_URL,
  findColumnBySlug,
} from '../../src/lib/content/columns';

describe('columns catalog', () => {
  it('uses a dedicated index and unique routes', () => {
    expect(COLUMNS_INDEX_PATH).toBe('/columns/');
    expect(new Set(COLUMN_CATALOG.map((column) => column.slug)).size).toBe(COLUMN_CATALOG.length);
    expect(new Set(COLUMN_CATALOG.map((column) => column.href)).size).toBe(COLUMN_CATALOG.length);

    for (const column of COLUMN_CATALOG) {
      if (column.isExternal) {
        expect(column.href).toMatch(/^https:\/\//);
      } else {
        expect(column.href).toMatch(/^\/[a-z0-9-]+\/$/);
      }
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

  it('registers the official Modern GPU Programming Chinese edition as an external column', () => {
    expect(findColumnBySlug('modern-gpu-programming-for-mlsys')).toEqual(
      expect.objectContaining({
        href: MODERN_GPU_PROGRAMMING_ZH_URL,
        isExternal: true,
        shortTitle: 'Modern GPU Programming For MLSys 中文版',
      }),
    );
  });
});
