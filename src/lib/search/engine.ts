/**
 * Search engine implementation using Fuse.js
 *
 * This module provides the core search functionality including
 * fuzzy matching, result ranking, and snippet generation.
 *
 * @module src/lib/search/engine
 */

import Fuse, { type FuseResult, type IFuseOptions, type FuseResultMatch } from 'fuse.js';
import type {
  SearchIndexEntry,
  SearchResult,
  SearchMatch,
  SearchQuery,
  SearchResponse,
} from './types';

/**
 * Default Fuse.js options for search
 */
export function getDefaultFuseOptions(weights: {
  title: number;
  headings: number;
  tags: number;
  summary: number;
  body: number;
}): IFuseOptions<SearchIndexEntry> {
  return {
    // Keys to search with weights
    keys: [
      { name: 'title', weight: weights.title },
      { name: 'headings', weight: weights.headings },
      { name: 'tags', weight: weights.tags },
      { name: 'summary', weight: weights.summary },
      { name: 'body', weight: weights.body },
    ],
    // Fuzzy matching settings
    threshold: 0.35,
    ignoreLocation: true,
    // Include match positions for highlighting
    includeMatches: true,
    includeScore: true,
    // Minimum characters to start matching
    minMatchCharLength: 1,
    // Use extended search for AND/OR
    useExtendedSearch: false,
    // Ignore field length norm for better results
    ignoreFieldNorm: false,
    // Limit per-field matches
    fieldNormWeight: 1,
  };
}

/**
 * Convert Fuse.js match to our SearchMatch format
 */
function convertFuseMatch(match: FuseResultMatch): SearchMatch {
  return {
    key: match.key || '',
    indices: match.indices as Array<[number, number]>,
    value: match.value,
  };
}

/**
 * Convert Fuse.js result to our SearchResult format
 */
function convertFuseResult(result: FuseResult<SearchIndexEntry>): SearchResult {
  return {
    item: result.item,
    score: result.score ?? 1,
    matches: (result.matches || []).map(convertFuseMatch),
  };
}

/**
 * Filter entries by tags (AND logic - entry must have all specified tags)
 */
export function filterByTags(entries: SearchIndexEntry[], tags: string[]): SearchIndexEntry[] {
  if (!tags || tags.length === 0) {
    return entries;
  }
  return entries.filter((entry) => tags.every((tag) => entry.tags.includes(tag)));
}

/**
 * Filter entries by year
 */
export function filterByYear(entries: SearchIndexEntry[], year: number): SearchIndexEntry[] {
  return entries.filter((entry) => {
    const entryYear = new Date(entry.date).getFullYear();
    return entryYear === year;
  });
}

/**
 * Apply filters to entries
 */
export function applyFilters(entries: SearchIndexEntry[], query: SearchQuery): SearchIndexEntry[] {
  let filtered = entries;

  if (query.tags && query.tags.length > 0) {
    filtered = filterByTags(filtered, query.tags);
  }

  if (query.year) {
    filtered = filterByYear(filtered, query.year);
  }

  return filtered;
}

/**
 * Sort results by relevance and date
 * - Primary: score (lower is better)
 * - Secondary: date (newer first)
 */
export function sortResults(results: SearchResult[]): SearchResult[] {
  return results.sort((a, b) => {
    // First by score (lower is better)
    const scoreDiff = a.score - b.score;
    if (Math.abs(scoreDiff) > 0.01) {
      return scoreDiff;
    }
    // Then by date (newer first)
    return new Date(b.item.date).getTime() - new Date(a.item.date).getTime();
  });
}

/**
 * Deduplicate results by slug
 */
export function deduplicateResults(results: SearchResult[]): SearchResult[] {
  const seen = new Set<string>();
  return results.filter((result) => {
    if (seen.has(result.item.slug)) {
      return false;
    }
    seen.add(result.item.slug);
    return true;
  });
}

/**
 * Escape HTML special characters
 */
export function escapeHtml(text: string): string {
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
export function highlightText(
  text: string,
  indices: Array<[number, number]>,
  maxLength?: number,
): string {
  if (!indices || indices.length === 0) {
    const escaped = escapeHtml(text);
    return maxLength && escaped.length > maxLength
      ? escaped.substring(0, maxLength) + '...'
      : escaped;
  }

  // Sort indices by start position
  const sortedIndices = [...indices].sort((a, b) => a[0] - b[0]);

  // Merge overlapping indices
  const merged: Array<[number, number]> = [];
  for (const [start, end] of sortedIndices) {
    if (merged.length === 0 || start > merged[merged.length - 1][1] + 1) {
      merged.push([start, end]);
    } else {
      merged[merged.length - 1][1] = Math.max(merged[merged.length - 1][1], end);
    }
  }

  // Build highlighted string
  let result = '';
  let lastIndex = 0;

  for (const [start, end] of merged) {
    // Add text before match
    result += escapeHtml(text.substring(lastIndex, start));
    // Add highlighted match
    result += `<mark>${escapeHtml(text.substring(start, end + 1))}</mark>`;
    lastIndex = end + 1;
  }

  // Add remaining text
  result += escapeHtml(text.substring(lastIndex));

  if (maxLength && result.length > maxLength) {
    // Try to find a good cut point
    const cutPoint = result.substring(0, maxLength).lastIndexOf(' ');
    if (cutPoint > maxLength * 0.7) {
      return result.substring(0, cutPoint) + '...';
    }
    return result.substring(0, maxLength) + '...';
  }

  return result;
}

/**
 * Extract a snippet around the first match
 */
export function extractSnippet(
  text: string,
  indices: Array<[number, number]>,
  windowSize: number = 80,
): { text: string; highlightIndices: Array<[number, number]> } {
  if (!indices || indices.length === 0 || !text) {
    const snippet = text.substring(0, windowSize * 2);
    return {
      text: snippet + (text.length > windowSize * 2 ? '...' : ''),
      highlightIndices: [],
    };
  }

  // Find the first match
  const firstMatch = indices[0];
  const matchStart = firstMatch[0];
  const matchEnd = firstMatch[1];

  // Calculate window around match
  let snippetStart = Math.max(0, matchStart - windowSize);
  let snippetEnd = Math.min(text.length, matchEnd + windowSize);

  // Adjust to word boundaries
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

  // Adjust indices for the snippet
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
 * Generate highlighted title and snippet for a search result
 */
export function generateHighlights(
  result: SearchResult,
  snippetWindow: number = 80,
): { highlightedTitle: string; highlightedSnippet: string } {
  let highlightedTitle = escapeHtml(result.item.title);
  let highlightedSnippet = '';

  // Find title match
  const titleMatch = result.matches.find((m) => m.key === 'title');
  if (titleMatch && titleMatch.indices) {
    highlightedTitle = highlightText(result.item.title, titleMatch.indices);
  }

  // Find body or summary match for snippet
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
    // No body/summary match, show summary
    highlightedSnippet = escapeHtml(
      result.item.summary.length > snippetWindow * 2
        ? result.item.summary.substring(0, snippetWindow * 2) + '...'
        : result.item.summary,
    );
  }

  return { highlightedTitle, highlightedSnippet };
}

/**
 * Search engine class
 */
export class SearchEngine {
  private fuse: Fuse<SearchIndexEntry> | null = null;
  private entries: SearchIndexEntry[] = [];
  private weights: {
    title: number;
    headings: number;
    tags: number;
    summary: number;
    body: number;
  };
  private snippetWindow: number;

  constructor(
    weights = { title: 6, headings: 3, tags: 3, summary: 2, body: 1 },
    snippetWindow = 80,
  ) {
    this.weights = weights;
    this.snippetWindow = snippetWindow;
  }

  /**
   * Initialize the search engine with entries
   */
  initialize(entries: SearchIndexEntry[]): void {
    this.entries = entries;
    const options = getDefaultFuseOptions(this.weights);
    this.fuse = new Fuse(entries, options);
  }

  /**
   * Check if the engine is initialized
   */
  isInitialized(): boolean {
    return this.fuse !== null;
  }

  /**
   * Perform a search
   */
  search(query: SearchQuery): SearchResponse {
    const startTime = performance.now();

    if (!this.fuse || !query.query.trim()) {
      return {
        results: [],
        totalMatches: 0,
        searchTime: performance.now() - startTime,
      };
    }

    // Apply filters first
    let searchEntries = applyFilters(this.entries, query);

    // If filters were applied, create a temporary Fuse instance
    let fuse = this.fuse;
    if (searchEntries.length !== this.entries.length) {
      const options = getDefaultFuseOptions(this.weights);
      fuse = new Fuse(searchEntries, options);
    }

    // Perform search
    const fuseResults = fuse.search(query.query);

    // Convert and process results
    let results = fuseResults.map(convertFuseResult);

    // Deduplicate
    results = deduplicateResults(results);

    // Sort by relevance and date
    results = sortResults(results);

    const totalMatches = results.length;

    // Apply max results limit
    const maxResults = query.maxResults || 12;
    results = results.slice(0, maxResults);

    // Generate highlights for final results
    results = results.map((result) => {
      const highlights = generateHighlights(result, this.snippetWindow);
      return {
        ...result,
        highlightedTitle: highlights.highlightedTitle,
        highlightedSnippet: highlights.highlightedSnippet,
      };
    });

    return {
      results,
      totalMatches,
      searchTime: performance.now() - startTime,
    };
  }

  /**
   * Get all entries (for tag list, etc.)
   */
  getEntries(): SearchIndexEntry[] {
    return this.entries;
  }
}
