/**
 * Type definitions for the import adapter architecture
 */

import type { Page } from '@playwright/test';
import type { Logger } from '../../logger/types.js';

/**
 * Standardized article structure output by all adapters
 */
export interface Article {
  /** Article title */
  title: string;
  /** Markdown content */
  markdown: string;
  /** Original/canonical URL */
  canonicalUrl: string;
  /** Source identifier (zhihu, medium, wechat, arxiv, others) */
  source: 'zhihu' | 'medium' | 'wechat' | 'arxiv' | 'others';
  /** Article author */
  author?: string;
  /** Publication date (ISO 8601) */
  publishedAt?: string;
  /** Last update date (ISO 8601) */
  updatedAt?: string;
  /** Cover image */
  cover?: {
    url: string;
    alt?: string;
  } | null;
  /** Tags */
  tags?: string[];
  /** Images extracted from article */
  images?: Array<{
    url: string;
    localPath?: string;
  }>;
  /** Diagnostic information */
  diagnostics?: {
    extractionMethod?: string;
    warnings?: string[];
    retries?: number;
  };
}

/**
 * Input parameters for adapter fetchArticle method
 */
export interface FetchArticleInput {
  url: string;
  page: Page;
  options?: {
    slug?: string;
    imageRoot?: string;
    publicBasePath?: string;
    downloadImage?: DownloadImageFunction;
    logger?: Logger;
  };
}

/**
 * Image download function type
 */
export type DownloadImageFunction = (
  url: string,
  provider: string,
  slug: string,
  imageRoot: string,
  index: number,
  articleUrl?: string,
  publicBasePath?: string,
) => Promise<string | null>;

/**
 * Adapter interface that all site adapters must implement
 */
export interface Adapter {
  /** Unique adapter identifier */
  id: 'zhihu' | 'medium' | 'wechat' | 'arxiv' | 'others';

  /** Human-readable adapter name */
  name: string;

  /** Check if this adapter can handle the given URL */
  canHandle(url: string): boolean;

  /** Fetch and convert article to standardized format */
  fetchArticle(input: FetchArticleInput): Promise<Article>;

  /** Optional: normalize article fields after extraction */
  normalize?(article: Article): Article;

  /** Diagnostic information for logging */
  diagnostics?: {
    lastUrl?: string;
    lastError?: string;
    attempts?: number;
  };
}

/**
 * Adapter registry interface
 */
export interface AdapterRegistry {
  /** Register an adapter */
  register(adapter: Adapter): void;

  /** Resolve the best adapter for a given URL */
  resolve(url: string, logger?: Logger): Adapter | null;

  /** Get all registered adapters */
  getAll(): Adapter[];

  /** Get adapter by ID */
  getById(id: string): Adapter | null;
}
