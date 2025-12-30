/**
 * Search configuration loader
 *
 * This module loads and validates the search configuration from search.yml.
 * It provides settings for full-text search functionality including fuzzy matching,
 * keyboard navigation, and result filtering.
 *
 * @module src/config/loaders/search
 */

import { z } from 'zod';
import { loadConfig } from './base';
import searchConfigData from '../yaml/search.yml';

/**
 * Search configuration schema
 */
export const searchConfigSchema = z.object({
  search: z.object({
    enabled: z.boolean().default(true),
    shortcut: z.boolean().default(true),
    provider: z.enum(['fuse', 'minisearch']).default('fuse'),
    lazyLoad: z.boolean().default(true),
    useWorker: z.boolean().default(true),
    maxResults: z.number().min(1).max(100).default(12),

    snippet: z
      .object({
        window: z.number().min(20).max(200).default(80),
        maxLines: z.number().min(1).max(5).default(2),
      })
      .default({}),

    weights: z
      .object({
        title: z.number().min(0).max(10).default(6),
        headings: z.number().min(0).max(10).default(3),
        tags: z.number().min(0).max(10).default(3),
        summary: z.number().min(0).max(10).default(2),
        body: z.number().min(0).max(10).default(1),
      })
      .default({}),

    fuse: z
      .object({
        threshold: z.number().min(0).max(1).default(0.35),
        minMatchCharLength: z.number().min(1).max(10).default(1),
        maxPatternLength: z.number().min(1).max(100).default(32),
        ignoreLocation: z.boolean().default(true),
        includeMatches: z.boolean().default(true),
        includeScore: z.boolean().default(true),
      })
      .default({}),

    filters: z
      .object({
        tags: z.boolean().default(true),
        year: z.boolean().default(false),
        source: z.boolean().default(false),
      })
      .default({}),

    ui: z
      .object({
        placement: z.enum(['modal', 'drawer']).default('modal'),
        showTags: z.boolean().default(true),
        showDate: z.boolean().default(true),
        placeholder: z.string().default('搜索文章...'),
        noResultsText: z.string().default('未找到相关文章'),
        loadingText: z.string().default('加载中...'),
        recentTitle: z.string().default('最近文章'),
        tagsTitle: z.string().default('热门标签'),
        closeText: z.string().default('关闭'),
        hintText: z.string().default('按 ESC 关闭'),
      })
      .default({}),
  }),
});

export type SearchConfig = z.infer<typeof searchConfigSchema>['search'];

/**
 * Default search configuration
 */
export const defaultSearchConfig: SearchConfig = {
  enabled: true,
  shortcut: true,
  provider: 'fuse',
  lazyLoad: true,
  useWorker: true,
  maxResults: 12,
  snippet: {
    window: 80,
    maxLines: 2,
  },
  weights: {
    title: 6,
    headings: 3,
    tags: 3,
    summary: 2,
    body: 1,
  },
  fuse: {
    threshold: 0.35,
    minMatchCharLength: 1,
    maxPatternLength: 32,
    ignoreLocation: true,
    includeMatches: true,
    includeScore: true,
  },
  filters: {
    tags: true,
    year: false,
    source: false,
  },
  ui: {
    placement: 'modal',
    showTags: true,
    showDate: true,
    placeholder: '搜索文章...',
    noResultsText: '未找到相关文章',
    loadingText: '加载中...',
    recentTitle: '最近文章',
    tagsTitle: '热门标签',
    closeText: '关闭',
    hintText: '按 ESC 关闭',
  },
};

/**
 * Load and validate search configuration
 *
 * @returns Validated search configuration
 * @throws Error if configuration is invalid
 */
export function loadSearchConfig(): SearchConfig {
  const parsed = loadConfig(searchConfigData, searchConfigSchema, 'search.yml');
  return parsed.search as SearchConfig;
}

/**
 * Cached search configuration instance
 */
let cachedConfig: SearchConfig | null = null;

/**
 * Get search configuration (cached)
 *
 * @returns Search configuration
 */
export function getSearchConfig(): SearchConfig {
  if (!cachedConfig) {
    cachedConfig = loadSearchConfig();
  }
  return cachedConfig;
}
