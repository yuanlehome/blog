/**
 * Search types and interfaces
 *
 * This module defines the types used for the search functionality.
 *
 * @module src/lib/search/types
 */

/**
 * A single entry in the search index
 */
export interface SearchIndexEntry {
  /** Unique identifier (slug) */
  slug: string;
  /** Post URL path */
  url: string;
  /** Post title */
  title: string;
  /** Post headings (H2, H3, etc.) */
  headings: string[];
  /** Post tags */
  tags: string[];
  /** Post date (ISO string) */
  date: string;
  /** Post summary/description if available */
  summary: string;
  /** Plain text content (truncated) */
  body: string;
  /** Source type (notion, wechat, others) */
  source?: string;
}

/**
 * The complete search index
 */
export interface SearchIndex {
  /** Version of the index format */
  version: number;
  /** When the index was generated */
  generatedAt: string;
  /** Total number of entries */
  count: number;
  /** All available tags with counts */
  tags: Record<string, number>;
  /** The search entries */
  entries: SearchIndexEntry[];
}

/**
 * A match position within a string
 */
export interface MatchPosition {
  /** Start index */
  start: number;
  /** End index */
  end: number;
}

/**
 * Match information for highlighting
 */
export interface SearchMatch {
  /** The field that was matched */
  key: string;
  /** Match positions within the value */
  indices: Array<[number, number]>;
  /** The matched value */
  value?: string;
}

/**
 * A single search result
 */
export interface SearchResult {
  /** The original index entry */
  item: SearchIndexEntry;
  /** Search relevance score (lower is better for Fuse.js) */
  score: number;
  /** Match information for highlighting */
  matches: SearchMatch[];
  /** Pre-computed highlighted title */
  highlightedTitle?: string;
  /** Pre-computed snippet with highlights */
  highlightedSnippet?: string;
}

/**
 * Search query options
 */
export interface SearchQuery {
  /** The search query string */
  query: string;
  /** Tag filters (if any) */
  tags?: string[];
  /** Year filter (if any) */
  year?: number;
  /** Maximum results to return */
  maxResults?: number;
}

/**
 * Search response from worker
 */
export interface SearchResponse {
  /** The search results */
  results: SearchResult[];
  /** Total number of matches (before limit) */
  totalMatches: number;
  /** Time taken to search in ms */
  searchTime: number;
}

/**
 * Message types for worker communication
 */
export type WorkerMessageType = 'init' | 'search' | 'ready' | 'results' | 'error';

/**
 * Message sent to the search worker
 */
export interface WorkerMessage {
  type: WorkerMessageType;
  payload?: SearchIndex | SearchQuery | SearchResponse | string;
}
