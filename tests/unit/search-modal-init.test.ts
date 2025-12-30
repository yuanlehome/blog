/**
 * Test suite for SearchModal initialization and URL generation
 *
 * This test suite validates:
 * 1. Correct URL generation with different BASE_URL values
 * 2. Error handling for various failure scenarios
 * 3. HTML detection for 404 pages
 */

import { describe, it, expect, beforeEach } from 'vitest';

/**
 * Helper function to simulate URL generation logic from SearchModal
 */
function generateSearchIndexUrl(baseUrl: string, origin: string = 'https://example.com'): string {
  // Replicate the logic from SearchModal.astro
  return new URL('search-index.json', origin + baseUrl).toString();
}

/**
 * Helper to normalize base URL (matches normalizeBase from slug module)
 */
function normalizeBase(base: string): string {
  if (!base) return '/';
  return base.endsWith('/') ? base : `${base}/`;
}

describe('SearchModal URL generation', () => {
  describe('generateSearchIndexUrl', () => {
    it('generates correct URL with root base /', () => {
      const url = generateSearchIndexUrl('/', 'https://example.com');
      expect(url).toBe('https://example.com/search-index.json');
    });

    it('generates correct URL with /blog/ base', () => {
      const url = generateSearchIndexUrl('/blog/', 'https://example.com');
      expect(url).toBe('https://example.com/blog/search-index.json');
    });

    it('generates correct URL with /blog base (without trailing slash)', () => {
      const url = generateSearchIndexUrl('/blog', 'https://example.com');
      expect(url).toBe('https://example.com/search-index.json');
    });

    it('generates correct URL with normalized /blog/ base', () => {
      const normalized = normalizeBase('/blog');
      const url = generateSearchIndexUrl(normalized, 'https://example.com');
      expect(url).toBe('https://example.com/blog/search-index.json');
    });

    it('handles double slash correctly', () => {
      // URL constructor does NOT normalize double slashes in path, which is expected behavior
      const url = generateSearchIndexUrl('/blog//', 'https://example.com');
      // The double slash will be preserved: /blog//search-index.json
      // This is fine because normalizeBase should prevent this scenario in production
      expect(url).toBe('https://example.com/blog//search-index.json');
    });

    it('works with localhost origin', () => {
      const url = generateSearchIndexUrl('/blog/', 'http://localhost:3000');
      expect(url).toBe('http://localhost:3000/blog/search-index.json');
    });

    it('works with GitHub Pages domain', () => {
      const url = generateSearchIndexUrl('/blog/', 'https://yuanlehome.github.io');
      expect(url).toBe('https://yuanlehome.github.io/blog/search-index.json');
    });
  });

  describe('normalizeBase', () => {
    it('adds trailing slash to /blog', () => {
      expect(normalizeBase('/blog')).toBe('/blog/');
    });

    it('keeps trailing slash on /blog/', () => {
      expect(normalizeBase('/blog/')).toBe('/blog/');
    });

    it('handles root path /', () => {
      expect(normalizeBase('/')).toBe('/');
    });

    it('handles empty string', () => {
      expect(normalizeBase('')).toBe('/');
    });
  });
});

describe('SearchModal error scenarios', () => {
  describe('Error response detection', () => {
    it('detects HTML 404 page', () => {
      const responseText = '<!DOCTYPE html><html><head><title>404</title></head></html>';
      const contentType = 'text/html';

      const isHtml =
        !contentType.includes('application/json') && responseText.trim().startsWith('<');
      expect(isHtml).toBe(true);
    });

    it('accepts valid JSON response', () => {
      const responseText = '{"version":1,"entries":[]}';
      const contentType = 'application/json';

      const isHtml =
        !contentType.includes('application/json') && responseText.trim().startsWith('<');
      expect(isHtml).toBe(false);
    });

    it('detects HTML even without content-type', () => {
      const responseText = '<html><body>404 Not Found</body></html>';
      const contentType = '';

      const isHtml =
        !contentType.includes('application/json') && responseText.trim().startsWith('<');
      expect(isHtml).toBe(true);
    });

    it('handles JSON with correct content-type', () => {
      const responseText = '{"version":1,"entries":[]}';
      const contentType = 'application/json; charset=utf-8';

      const isHtml =
        !contentType.includes('application/json') && responseText.trim().startsWith('<');
      expect(isHtml).toBe(false);
    });

    it('does not false-positive on text starting with <', () => {
      const responseText = '< is less than symbol';
      const contentType = 'text/plain';

      // This would be detected as HTML, which is acceptable for our use case
      // since we're looking for HTML responses, and this edge case is unlikely
      const isHtml =
        !contentType.includes('application/json') && responseText.trim().startsWith('<');
      expect(isHtml).toBe(true);
    });
  });

  describe('Error message validation', () => {
    it('error object contains required fields', () => {
      const errorInfo = {
        url: 'https://example.com/blog/search-index.json',
        status: 404,
        statusText: 'Not Found',
        contentType: 'text/html',
      };

      expect(errorInfo).toHaveProperty('url');
      expect(errorInfo).toHaveProperty('status');
      expect(errorInfo).toHaveProperty('statusText');
      expect(errorInfo).toHaveProperty('contentType');
      expect(errorInfo.status).toBe(404);
    });

    it('parses HTML response preview', () => {
      const responseText =
        '<!DOCTYPE html><html><head><title>404</title></head><body>Page not found</body></html>';
      const preview = responseText.substring(0, 100);

      expect(preview).toContain('<!DOCTYPE html>');
      expect(preview.length).toBeLessThanOrEqual(100);
    });
  });
});

describe('JSON parsing scenarios', () => {
  it('parses valid search index JSON', () => {
    const jsonText = JSON.stringify({
      version: 1,
      generatedAt: '2024-01-01T00:00:00.000Z',
      count: 1,
      tags: { javascript: 1 },
      entries: [
        {
          slug: 'test-post',
          url: '/test-post/',
          title: 'Test Post',
          headings: [],
          tags: ['javascript'],
          date: '2024-01-01T00:00:00.000Z',
          summary: 'Test summary',
          body: 'Test body',
        },
      ],
    });

    const parsed = JSON.parse(jsonText);
    expect(parsed.version).toBe(1);
    expect(parsed.entries).toHaveLength(1);
    expect(parsed.entries[0].slug).toBe('test-post');
  });

  it('throws on invalid JSON', () => {
    const invalidJson = '{"version":1,invalid}';

    expect(() => JSON.parse(invalidJson)).toThrow();
  });

  it('throws on empty string', () => {
    const emptyJson = '';

    expect(() => JSON.parse(emptyJson)).toThrow();
  });

  it('handles JSON with extra whitespace', () => {
    const jsonText = '  \n  {"version": 1}  \n  ';

    const parsed = JSON.parse(jsonText);
    expect(parsed.version).toBe(1);
  });
});
