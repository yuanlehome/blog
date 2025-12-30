import { describe, it, expect, beforeEach } from 'vitest';
import {
  filterByTags,
  filterByYear,
  applyFilters,
  sortResults,
  deduplicateResults,
  escapeHtml,
  highlightText,
  extractSnippet,
  generateHighlights,
  SearchEngine,
} from '../../src/lib/search/engine';
import type { SearchIndexEntry, SearchResult } from '../../src/lib/search/types';

const createMockEntry = (overrides: Partial<SearchIndexEntry> = {}): SearchIndexEntry => ({
  slug: 'test-post',
  url: '/test-post/',
  title: 'Test Post',
  headings: ['Heading 1', 'Heading 2'],
  tags: ['javascript', 'typescript'],
  date: '2024-01-15T00:00:00.000Z',
  summary: 'This is a test summary',
  body: 'This is the full body content of the test post',
  ...overrides,
});

const createMockResult = (overrides: Partial<SearchResult> = {}): SearchResult => ({
  item: createMockEntry(),
  score: 0.5,
  matches: [],
  ...overrides,
});

describe('search engine', () => {
  describe('filterByTags', () => {
    const entries: SearchIndexEntry[] = [
      createMockEntry({ slug: 'a', tags: ['js', 'ts'] }),
      createMockEntry({ slug: 'b', tags: ['js'] }),
      createMockEntry({ slug: 'c', tags: ['python'] }),
    ];

    it('returns all entries when no tags specified', () => {
      expect(filterByTags(entries, [])).toEqual(entries);
    });

    it('filters by single tag', () => {
      const result = filterByTags(entries, ['js']);
      expect(result.map((e) => e.slug)).toEqual(['a', 'b']);
    });

    it('filters by multiple tags (AND logic)', () => {
      const result = filterByTags(entries, ['js', 'ts']);
      expect(result.map((e) => e.slug)).toEqual(['a']);
    });

    it('returns empty array when no matches', () => {
      const result = filterByTags(entries, ['rust']);
      expect(result).toEqual([]);
    });
  });

  describe('filterByYear', () => {
    const entries: SearchIndexEntry[] = [
      createMockEntry({ slug: 'a', date: '2024-01-01T00:00:00.000Z' }),
      createMockEntry({ slug: 'b', date: '2024-06-01T00:00:00.000Z' }),
      createMockEntry({ slug: 'c', date: '2023-01-01T00:00:00.000Z' }),
    ];

    it('filters by year', () => {
      const result = filterByYear(entries, 2024);
      expect(result.map((e) => e.slug)).toEqual(['a', 'b']);
    });

    it('returns empty array for year with no posts', () => {
      const result = filterByYear(entries, 2022);
      expect(result).toEqual([]);
    });
  });

  describe('applyFilters', () => {
    const entries: SearchIndexEntry[] = [
      createMockEntry({ slug: 'a', tags: ['js'], date: '2024-01-01T00:00:00.000Z' }),
      createMockEntry({ slug: 'b', tags: ['js'], date: '2023-01-01T00:00:00.000Z' }),
      createMockEntry({ slug: 'c', tags: ['python'], date: '2024-01-01T00:00:00.000Z' }),
    ];

    it('applies no filters when none specified', () => {
      const result = applyFilters(entries, { query: 'test' });
      expect(result).toEqual(entries);
    });

    it('applies tag filter', () => {
      const result = applyFilters(entries, { query: 'test', tags: ['js'] });
      expect(result.map((e) => e.slug)).toEqual(['a', 'b']);
    });

    it('applies year filter', () => {
      const result = applyFilters(entries, { query: 'test', year: 2024 });
      expect(result.map((e) => e.slug)).toEqual(['a', 'c']);
    });

    it('applies combined filters', () => {
      const result = applyFilters(entries, { query: 'test', tags: ['js'], year: 2024 });
      expect(result.map((e) => e.slug)).toEqual(['a']);
    });
  });

  describe('sortResults', () => {
    it('sorts by score (lower is better)', () => {
      const results: SearchResult[] = [
        createMockResult({ score: 0.5 }),
        createMockResult({ score: 0.1 }),
        createMockResult({ score: 0.8 }),
      ];

      const sorted = sortResults(results);
      expect(sorted.map((r) => r.score)).toEqual([0.1, 0.5, 0.8]);
    });

    it('sorts by date when scores are similar', () => {
      const results: SearchResult[] = [
        createMockResult({
          score: 0.5,
          item: createMockEntry({ date: '2024-01-01T00:00:00.000Z' }),
        }),
        createMockResult({
          score: 0.5,
          item: createMockEntry({ date: '2024-06-01T00:00:00.000Z' }),
        }),
      ];

      const sorted = sortResults(results);
      // Newer first when scores are close
      expect(new Date(sorted[0].item.date).getMonth()).toBe(5); // June
    });
  });

  describe('deduplicateResults', () => {
    it('removes duplicate entries by slug', () => {
      const results: SearchResult[] = [
        createMockResult({ item: createMockEntry({ slug: 'a' }) }),
        createMockResult({ item: createMockEntry({ slug: 'b' }) }),
        createMockResult({ item: createMockEntry({ slug: 'a' }) }),
      ];

      const deduped = deduplicateResults(results);
      expect(deduped.map((r) => r.item.slug)).toEqual(['a', 'b']);
    });

    it('keeps first occurrence', () => {
      const results: SearchResult[] = [
        createMockResult({ item: createMockEntry({ slug: 'a', title: 'First' }), score: 0.1 }),
        createMockResult({ item: createMockEntry({ slug: 'a', title: 'Second' }), score: 0.5 }),
      ];

      const deduped = deduplicateResults(results);
      expect(deduped[0].item.title).toBe('First');
    });
  });

  describe('escapeHtml', () => {
    it('escapes special characters', () => {
      expect(escapeHtml('<script>')).toBe('&lt;script&gt;');
      expect(escapeHtml('a & b')).toBe('a &amp; b');
      expect(escapeHtml('"quoted"')).toBe('&quot;quoted&quot;');
    });

    it('preserves normal text', () => {
      expect(escapeHtml('hello world')).toBe('hello world');
    });
  });

  describe('highlightText', () => {
    it('wraps matches in mark tags', () => {
      const text = 'hello world';
      const indices: Array<[number, number]> = [[0, 4]];
      expect(highlightText(text, indices)).toBe('<mark>hello</mark> world');
    });

    it('handles multiple matches', () => {
      const text = 'foo bar foo';
      const indices: Array<[number, number]> = [
        [0, 2],
        [8, 10],
      ];
      expect(highlightText(text, indices)).toBe('<mark>foo</mark> bar <mark>foo</mark>');
    });

    it('merges overlapping indices', () => {
      const text = 'hello';
      const indices: Array<[number, number]> = [
        [0, 2],
        [1, 4],
      ];
      expect(highlightText(text, indices)).toBe('<mark>hello</mark>');
    });

    it('escapes HTML in highlighted text', () => {
      const text = '<script> alert()';
      const indices: Array<[number, number]> = [[9, 13]];
      expect(highlightText(text, indices)).toBe('&lt;script&gt; <mark>alert</mark>()');
    });

    it('returns escaped text when no indices', () => {
      expect(highlightText('hello', [])).toBe('hello');
    });
  });

  describe('extractSnippet', () => {
    it('extracts snippet around match', () => {
      const text = 'a'.repeat(50) + ' match ' + 'b'.repeat(50);
      const indices: Array<[number, number]> = [[51, 55]];
      const result = extractSnippet(text, indices, 20);

      expect(result.text).toContain('match');
      expect(result.text.length).toBeLessThan(text.length);
    });

    it('adds ellipsis when truncated', () => {
      const text = 'a'.repeat(100) + ' match ' + 'b'.repeat(100);
      const indices: Array<[number, number]> = [[101, 105]];
      const result = extractSnippet(text, indices, 20);

      expect(result.text.startsWith('...')).toBe(true);
      expect(result.text.endsWith('...')).toBe(true);
    });

    it('adjusts highlight indices for snippet', () => {
      const text = 'prefix match suffix';
      const indices: Array<[number, number]> = [[7, 11]];
      const result = extractSnippet(text, indices, 50);

      expect(result.highlightIndices.length).toBeGreaterThan(0);
    });

    it('handles no matches', () => {
      const text = 'some text content';
      const result = extractSnippet(text, [], 10);

      // Text shorter than 2*window is returned as-is
      expect(result.text).toBe('some text content');
      expect(result.highlightIndices).toEqual([]);
    });
  });

  describe('generateHighlights', () => {
    it('generates highlighted title and snippet', () => {
      const result = createMockResult({
        matches: [
          {
            key: 'title',
            indices: [[0, 3]],
            value: 'Test Post',
          },
          {
            key: 'body',
            indices: [[0, 3]],
            value: 'This is content',
          },
        ],
      });

      const highlights = generateHighlights(result, 50);

      expect(highlights.highlightedTitle).toContain('<mark>');
      expect(highlights.highlightedSnippet).toBeTruthy();
    });

    it('uses summary when body has no matches', () => {
      const result = createMockResult({
        item: createMockEntry({ summary: 'Summary text here' }),
        matches: [
          {
            key: 'summary',
            indices: [[0, 6]],
            value: 'Summary text here',
          },
        ],
      });

      const highlights = generateHighlights(result, 50);

      expect(highlights.highlightedSnippet).toContain('<mark>Summary</mark>');
    });
  });

  describe('SearchEngine', () => {
    let engine: SearchEngine;
    let entries: SearchIndexEntry[];

    beforeEach(() => {
      entries = [
        createMockEntry({
          slug: 'javascript-guide',
          title: 'JavaScript Guide',
          tags: ['javascript'],
          body: 'Learn JavaScript programming',
        }),
        createMockEntry({
          slug: 'typescript-intro',
          title: 'TypeScript Introduction',
          tags: ['typescript', 'javascript'],
          body: 'TypeScript is a typed superset of JavaScript',
        }),
        createMockEntry({
          slug: 'python-basics',
          title: 'Python Basics',
          tags: ['python'],
          body: 'Learn Python programming language',
        }),
      ];

      engine = new SearchEngine({ title: 6, headings: 3, tags: 3, summary: 2, body: 1 }, 80);
      engine.initialize(entries);
    });

    it('initializes correctly', () => {
      expect(engine.isInitialized()).toBe(true);
      expect(engine.getEntries()).toEqual(entries);
    });

    it('searches by title', () => {
      const response = engine.search({ query: 'JavaScript' });

      expect(response.results.length).toBeGreaterThan(0);
      expect(response.results[0].item.title).toContain('JavaScript');
    });

    it('searches by body content', () => {
      const response = engine.search({ query: 'typed superset' });

      expect(response.results.length).toBe(1);
      expect(response.results[0].item.slug).toBe('typescript-intro');
    });

    it('returns title matches before body matches', () => {
      const response = engine.search({ query: 'JavaScript' });

      // JavaScript Guide should rank higher than TypeScript Introduction
      // because title match is weighted higher
      const jsGuideIndex = response.results.findIndex((r) => r.item.slug === 'javascript-guide');
      const tsIntroIndex = response.results.findIndex((r) => r.item.slug === 'typescript-intro');

      expect(jsGuideIndex).toBeLessThan(tsIntroIndex);
    });

    it('filters by tags', () => {
      const response = engine.search({ query: 'programming', tags: ['python'] });

      expect(response.results.length).toBe(1);
      expect(response.results[0].item.slug).toBe('python-basics');
    });

    it('limits results', () => {
      const response = engine.search({ query: 'JavaScript', maxResults: 1 });

      expect(response.results.length).toBe(1);
    });

    it('returns empty for no query', () => {
      const response = engine.search({ query: '' });

      expect(response.results).toEqual([]);
    });

    it('includes search time', () => {
      const response = engine.search({ query: 'JavaScript' });

      expect(response.searchTime).toBeGreaterThanOrEqual(0);
    });

    it('generates highlights for results', () => {
      const response = engine.search({ query: 'JavaScript' });

      expect(response.results[0].highlightedTitle).toContain('<mark>');
    });

    it('handles fuzzy matching', () => {
      // "Javascrpt" with typo should still match "JavaScript"
      const response = engine.search({ query: 'Javascrpt' });

      expect(response.results.length).toBeGreaterThan(0);
    });

    it('handles Chinese content', () => {
      const chineseEntries: SearchIndexEntry[] = [
        createMockEntry({
          slug: 'chinese-post',
          title: '深度学习入门',
          body: '本文介绍深度学习的基础知识',
        }),
      ];

      const chineseEngine = new SearchEngine();
      chineseEngine.initialize(chineseEntries);

      const response = chineseEngine.search({ query: '深度学习' });

      expect(response.results.length).toBe(1);
    });
  });
});
