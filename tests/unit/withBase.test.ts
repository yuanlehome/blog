import { describe, expect, it } from 'vitest';
import { normalizeBase, withBase } from '../../src/lib/site/withBase';

describe('withBase helpers', () => {
  it('normalizes base values', () => {
    expect(normalizeBase('blog')).toBe('/blog/');
    expect(normalizeBase('/blog')).toBe('/blog/');
    expect(normalizeBase('/')).toBe('/');
  });

  it('prefixes absolute paths with non-root base', () => {
    expect(withBase('/generated/diagrams/a.svg', '/blog/')).toBe('/blog/generated/diagrams/a.svg');
  });

  it('keeps root base unchanged', () => {
    expect(withBase('/generated/diagrams/a.svg', '/')).toBe('/generated/diagrams/a.svg');
  });

  it('does not double-prefix when path already contains base', () => {
    expect(withBase('/blog/generated/diagrams/a.svg', '/blog/')).toBe(
      '/blog/generated/diagrams/a.svg',
    );
  });

  it('does not rewrite external or relative paths', () => {
    expect(withBase('https://example.com/a.svg', '/blog/')).toBe('https://example.com/a.svg');
    expect(withBase('generated/diagrams/a.svg', '/blog/')).toBe('generated/diagrams/a.svg');
  });
});
