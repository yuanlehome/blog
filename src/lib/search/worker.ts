/**
 * Search Web Worker
 *
 * This worker handles search operations off the main thread
 * to prevent UI blocking during search.
 *
 * @module src/lib/search/worker
 */

import Fuse, { type FuseResult, type IFuseOptions } from 'fuse.js';
import type {
  SearchIndexEntry,
  SearchIndex,
  SearchQuery,
  SearchResult,
  SearchResponse,
  SearchMatch,
} from './types';

// Worker state
let fuse: Fuse<SearchIndexEntry> | null = null;
let entries: SearchIndexEntry[] = [];
let weights = { title: 6, headings: 3, tags: 3, summary: 2, body: 1 };
let snippetWindow = 80;

/**
 * Get Fuse.js options
 */
function getFuseOptions(): IFuseOptions<SearchIndexEntry> {
  return {
    keys: [
      { name: 'title', weight: weights.title },
      { name: 'headings', weight: weights.headings },
      { name: 'tags', weight: weights.tags },
      { name: 'summary', weight: weights.summary },
      { name: 'body', weight: weights.body },
    ],
    threshold: 0.35,
    ignoreLocation: true,
    includeMatches: true,
    includeScore: true,
    minMatchCharLength: 1,
    useExtendedSearch: false,
    ignoreFieldNorm: false,
    fieldNormWeight: 1,
  };
}

/**
 * Escape HTML special characters
 */
function escapeHtml(text: string): string {
  const escapeMap: Record<string, string> = {
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    '"': '&quot;',
    "'": '&#39;',
  };
  return text.replace(/[&<>"']/g, (char) => escapeMap[char] || char);
}

/**
 * Generate highlighted text from match indices
 */
function highlightText(text: string, indices: Array<[number, number]>): string {
  if (!indices || indices.length === 0) {
    return escapeHtml(text);
  }

  const sortedIndices = [...indices].sort((a, b) => a[0] - b[0]);
  const merged: Array<[number, number]> = [];

  for (const [start, end] of sortedIndices) {
    if (merged.length === 0 || start > merged[merged.length - 1][1] + 1) {
      merged.push([start, end]);
    } else {
      merged[merged.length - 1][1] = Math.max(merged[merged.length - 1][1], end);
    }
  }

  let result = '';
  let lastIndex = 0;

  for (const [start, end] of merged) {
    result += escapeHtml(text.substring(lastIndex, start));
    result += `<mark>${escapeHtml(text.substring(start, end + 1))}</mark>`;
    lastIndex = end + 1;
  }

  result += escapeHtml(text.substring(lastIndex));
  return result;
}

/**
 * Extract a snippet around the first match
 */
function extractSnippet(
  text: string,
  indices: Array<[number, number]>,
  window: number = 80,
): { text: string; highlightIndices: Array<[number, number]> } {
  if (!indices || indices.length === 0 || !text) {
    const snippet = text.substring(0, window * 2);
    return {
      text: snippet + (text.length > window * 2 ? '...' : ''),
      highlightIndices: [],
    };
  }

  const firstMatch = indices[0];
  const matchStart = firstMatch[0];
  const matchEnd = firstMatch[1];

  let snippetStart = Math.max(0, matchStart - window);
  let snippetEnd = Math.min(text.length, matchEnd + window);

  if (snippetStart > 0) {
    const spaceIndex = text.indexOf(' ', snippetStart);
    if (spaceIndex !== -1 && spaceIndex < matchStart) {
      snippetStart = spaceIndex + 1;
    }
  }

  if (snippetEnd < text.length) {
    const spaceIndex = text.lastIndexOf(' ', snippetEnd);
    if (spaceIndex !== -1 && spaceIndex > matchEnd) {
      snippetEnd = spaceIndex;
    }
  }

  const snippet = text.substring(snippetStart, snippetEnd);
  const prefix = snippetStart > 0 ? '...' : '';
  const suffix = snippetEnd < text.length ? '...' : '';

  const adjustedIndices: Array<[number, number]> = [];
  for (const [start, end] of indices) {
    if (start >= snippetStart && end <= snippetEnd) {
      adjustedIndices.push([
        start - snippetStart + prefix.length,
        end - snippetStart + prefix.length,
      ]);
    }
  }

  return {
    text: prefix + snippet + suffix,
    highlightIndices: adjustedIndices,
  };
}

/**
 * Generate highlights for a result
 */
function generateHighlights(result: SearchResult): {
  highlightedTitle: string;
  highlightedSnippet: string;
} {
  let highlightedTitle = escapeHtml(result.item.title);
  let highlightedSnippet = '';

  const titleMatch = result.matches.find((m) => m.key === 'title');
  if (titleMatch && titleMatch.indices) {
    highlightedTitle = highlightText(result.item.title, titleMatch.indices);
  }

  const bodyMatch = result.matches.find((m) => m.key === 'body');
  const summaryMatch = result.matches.find((m) => m.key === 'summary');

  if (bodyMatch && bodyMatch.indices && bodyMatch.indices.length > 0) {
    const { text, highlightIndices } = extractSnippet(
      result.item.body,
      bodyMatch.indices,
      snippetWindow,
    );
    highlightedSnippet = highlightText(text, highlightIndices);
  } else if (summaryMatch && summaryMatch.indices) {
    highlightedSnippet = highlightText(result.item.summary, summaryMatch.indices);
  } else {
    highlightedSnippet = escapeHtml(
      result.item.summary.length > snippetWindow * 2
        ? result.item.summary.substring(0, snippetWindow * 2) + '...'
        : result.item.summary,
    );
  }

  return { highlightedTitle, highlightedSnippet };
}

/**
 * Filter entries by tags
 */
function filterByTags(entries: SearchIndexEntry[], tags: string[]): SearchIndexEntry[] {
  if (!tags || tags.length === 0) return entries;
  return entries.filter((entry) => tags.every((tag) => entry.tags.includes(tag)));
}

/**
 * Filter entries by year
 */
function filterByYear(entries: SearchIndexEntry[], year: number): SearchIndexEntry[] {
  return entries.filter((entry) => new Date(entry.date).getFullYear() === year);
}

/**
 * Convert Fuse result to SearchResult
 */
function convertResult(result: FuseResult<SearchIndexEntry>): SearchResult {
  return {
    item: result.item,
    score: result.score ?? 1,
    matches: (result.matches || []).map((m) => ({
      key: m.key || '',
      indices: m.indices as Array<[number, number]>,
      value: m.value,
    })),
  };
}

/**
 * Perform search
 */
function search(query: SearchQuery): SearchResponse {
  const startTime = performance.now();

  if (!fuse || !query.query.trim()) {
    return {
      results: [],
      totalMatches: 0,
      searchTime: performance.now() - startTime,
    };
  }

  // Apply filters
  let searchEntries = entries;
  if (query.tags && query.tags.length > 0) {
    searchEntries = filterByTags(searchEntries, query.tags);
  }
  if (query.year) {
    searchEntries = filterByYear(searchEntries, query.year);
  }

  // Create filtered Fuse instance if needed
  let searchFuse = fuse;
  if (searchEntries.length !== entries.length) {
    searchFuse = new Fuse(searchEntries, getFuseOptions());
  }

  // Search
  const fuseResults = searchFuse.search(query.query);

  // Convert and process
  let results = fuseResults.map(convertResult);

  // Deduplicate
  const seen = new Set<string>();
  results = results.filter((r) => {
    if (seen.has(r.item.slug)) return false;
    seen.add(r.item.slug);
    return true;
  });

  // Sort
  results.sort((a, b) => {
    const scoreDiff = a.score - b.score;
    if (Math.abs(scoreDiff) > 0.01) return scoreDiff;
    return new Date(b.item.date).getTime() - new Date(a.item.date).getTime();
  });

  const totalMatches = results.length;

  // Limit
  results = results.slice(0, query.maxResults || 12);

  // Generate highlights
  results = results.map((result) => {
    const highlights = generateHighlights(result);
    return { ...result, ...highlights };
  });

  return {
    results,
    totalMatches,
    searchTime: performance.now() - startTime,
  };
}

/**
 * Initialize the worker with index data
 */
function initialize(
  index: SearchIndex,
  config?: { weights?: typeof weights; snippetWindow?: number },
): void {
  entries = index.entries;
  if (config?.weights) weights = config.weights;
  if (config?.snippetWindow) snippetWindow = config.snippetWindow;
  fuse = new Fuse(entries, getFuseOptions());
}

// Message handler
self.onmessage = (event: MessageEvent) => {
  const { type, payload } = event.data;

  switch (type) {
    case 'init':
      try {
        const { index, config } = payload as {
          index: SearchIndex;
          config?: { weights?: typeof weights; snippetWindow?: number };
        };
        initialize(index, config);
        self.postMessage({ type: 'ready' });
      } catch (error) {
        self.postMessage({ type: 'error', payload: String(error) });
      }
      break;

    case 'search':
      try {
        const response = search(payload as SearchQuery);
        self.postMessage({ type: 'results', payload: response });
      } catch (error) {
        self.postMessage({ type: 'error', payload: String(error) });
      }
      break;

    default:
      self.postMessage({ type: 'error', payload: `Unknown message type: ${type}` });
  }
};
