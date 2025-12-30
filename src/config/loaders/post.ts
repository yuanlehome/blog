/**
 * Post page configuration loader
 *
 * This module loads and validates the post page configuration from post.yml.
 * It provides settings for individual blog post pages including metadata display,
 * table of contents, comments, and related features.
 *
 * @module src/config/loaders/post
 */

import { z } from 'zod';
import { loadConfig } from './base';
import postConfigData from '../yaml/post.yml';

/**
 * Post metadata configuration schema
 */
export const postMetadataSchema = z.object({
  showPublishedDate: z.boolean().default(true),
  showUpdatedDate: z.boolean().default(true),
  showReadingTime: z.boolean().default(true),
  showWordCount: z.boolean().default(true),
  publishedLabel: z.string().default('发布于'),
  updatedLabel: z.string().default('更新于'),
  icons: z
    .object({
      published: z.string().default('📅'),
      updated: z.string().default('🔄'),
      wordCount: z.string().default('✍️'),
      readingTime: z.string().default('⏱️'),
    })
    .default({}),
});

export type PostMetadata = z.infer<typeof postMetadataSchema>;

/**
 * Table of contents configuration schema
 */
export const tocConfigSchema = z.object({
  enable: z.boolean().default(true),
  defaultExpanded: z.boolean().default(false),
  showOnMobile: z.boolean().default(true),
  mobileTrigger: z.boolean().default(false),
});

export type TocConfig = z.infer<typeof tocConfigSchema>;

/**
 * Floating actions configuration schema
 */
export const floatingActionsSchema = z.object({
  enableToc: z.boolean().default(true),
  enableTop: z.boolean().default(true),
  enableBottom: z.boolean().default(true),
});

export type FloatingActions = z.infer<typeof floatingActionsSchema>;

/**
 * Reading progress configuration schema
 */
export const readingProgressSchema = z.object({
  enable: z.boolean().default(true),
});

export type ReadingProgress = z.infer<typeof readingProgressSchema>;

/**
 * Prev/Next navigation configuration schema
 */
export const prevNextSchema = z.object({
  enable: z.boolean().default(true),
});

export type PrevNext = z.infer<typeof prevNextSchema>;

/**
 * Related posts configuration schema
 */
export const relatedPostsSchema = z.object({
  enable: z.boolean().default(true),
  maxCount: z.number().min(1).default(3),
});

export type RelatedPosts = z.infer<typeof relatedPostsSchema>;

/**
 * Giscus configuration schema
 */
export const giscusConfigSchema = z.object({
  repo: z.string().default('yuanlehome/blog'),
  repoId: z.string().default('R_kgDOQu2Jjw'),
  category: z.string().default('General'),
  categoryId: z.string().default('DIC_kwDOQu2Jj84C0QDN'),
  mapping: z.string().default('pathname'),
  strict: z.string().default('0'),
  reactionsEnabled: z.string().default('1'),
  emitMetadata: z.string().default('0'),
  inputPosition: z.string().default('bottom'),
  theme: z.string().default('preferred_color_scheme'),
  lang: z.string().default('zh-CN'),
});

export type GiscusConfig = z.infer<typeof giscusConfigSchema>;

/**
 * Comments configuration schema
 */
export const commentsConfigSchema = z.object({
  enable: z.boolean().default(true),
  defaultEnabled: z.boolean().default(true),
  provider: z.string().default('giscus'),
  giscus: giscusConfigSchema.default({}),
  themeMapping: z
    .object({
      light: z.string().default('light'),
      dark: z.string().default('dark_dimmed'),
    })
    .default({}),
});

export type CommentsConfig = z.infer<typeof commentsConfigSchema>;

/**
 * Source attribution configuration schema
 */
export const sourceAttributionSchema = z.object({
  enable: z.boolean().default(true),
  prefix: z.string().default('📝 转载 / 引用来源：'),
  authorPrefix: z.string().default('作者'),
  linkText: z.string().default('查看原文'),
});

export type SourceAttribution = z.infer<typeof sourceAttributionSchema>;

/**
 * Post page configuration schema
 */
export const postConfigSchema = z.object({
  metadata: postMetadataSchema.default({}),
  tableOfContents: tocConfigSchema.default({}),
  floatingActions: floatingActionsSchema.default({}),
  readingProgress: readingProgressSchema.default({}),
  prevNext: prevNextSchema.default({}),
  relatedPosts: relatedPostsSchema.default({}),
  comments: commentsConfigSchema.default({}),
  sourceAttribution: sourceAttributionSchema.default({}),
});

export type PostConfig = z.infer<typeof postConfigSchema>;

/**
 * Default post page configuration
 */
export const defaultPostConfig: PostConfig = {
  metadata: {
    showPublishedDate: true,
    showUpdatedDate: true,
    showReadingTime: true,
    showWordCount: true,
    publishedLabel: '发布于',
    updatedLabel: '更新于',
    icons: {
      published: '📅',
      updated: '🔄',
      wordCount: '✍️',
      readingTime: '⏱️',
    },
  },
  tableOfContents: {
    enable: true,
    defaultExpanded: false,
    showOnMobile: true,
    mobileTrigger: false,
  },
  floatingActions: {
    enableToc: true,
    enableTop: true,
    enableBottom: true,
  },
  readingProgress: {
    enable: true,
  },
  prevNext: {
    enable: true,
  },
  relatedPosts: {
    enable: true,
    maxCount: 3,
  },
  comments: {
    enable: true,
    defaultEnabled: true,
    provider: 'giscus',
    giscus: {
      repo: 'yuanlehome/blog',
      repoId: 'R_kgDOQu2Jjw',
      category: 'General',
      categoryId: 'DIC_kwDOQu2Jj84C0QDN',
      mapping: 'pathname',
      strict: '0',
      reactionsEnabled: '1',
      emitMetadata: '0',
      inputPosition: 'bottom',
      theme: 'preferred_color_scheme',
      lang: 'zh-CN',
    },
    themeMapping: {
      light: 'light',
      dark: 'dark_dimmed',
    },
  },
  sourceAttribution: {
    enable: true,
    prefix: '📝 转载 / 引用来源：',
    authorPrefix: '作者',
    linkText: '查看原文',
  },
};

/**
 * Load and validate post page configuration
 *
 * @returns Validated post page configuration
 * @throws Error if configuration is invalid
 */
export function loadPostConfig(): PostConfig {
  const parsed = loadConfig(postConfigData, postConfigSchema, 'post.yml');
  return parsed as PostConfig;
}

/**
 * Cached post configuration instance
 */
let cachedConfig: PostConfig | null = null;

/**
 * Get post page configuration (cached)
 *
 * @returns Post page configuration
 */
export function getPostConfig(): PostConfig {
  if (!cachedConfig) {
    cachedConfig = loadPostConfig();
  }
  return cachedConfig;
}
