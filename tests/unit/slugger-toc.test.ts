import { describe, expect, it } from 'vitest';
import { createHeadingSlugger } from '../../src/utils/slugger';
import { buildTocForest, flattenForest, type Heading } from '../../src/utils/tocTree';

describe('slugger and toc utilities', () => {
  it('creates unique, stable slugs', () => {
    const slug = createHeadingSlugger();
    expect(slug('Hello World')).toBe('hello-world');
    expect(slug('Hello World')).toBe('hello-world-1');
    expect(slug('你好')).toBe('你好');
  });

  it('builds forest and flattens in depth-first order', () => {
    const headings: Heading[] = [
      { depth: 1, slug: 'root-1', text: 'Root 1' },
      { depth: 2, slug: 'child-1', text: 'Child' },
      { depth: 1, slug: 'root-2', text: 'Root 2' },
      { depth: 3, slug: 'grandchild', text: 'Grandchild' },
    ];

    const forest = buildTocForest(headings);
    expect(forest).toHaveLength(2);
    expect(forest[0].children[0].slug).toBe('child-1');
    expect(forest[1].children[0]?.slug).toBe('grandchild');

    const flattened = flattenForest(forest);
    expect(flattened.map((h) => h.slug)).toEqual(['root-1', 'child-1', 'root-2', 'grandchild']);
  });
});
