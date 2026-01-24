/**
 * Tag Module - Tag aggregation and slug generation
 *
 * This module provides tag-related utilities for the blog:
 * - Tag slug generation with URL safety
 * - Tag index building (aggregating all tags from posts)
 * - Post filtering by tag
 * - Disambiguation for duplicate tag slugs
 */

import type { CollectionEntry } from 'astro:content';
import { normalizeSlug } from '../slug';

export interface TagStats {
  name: string;
  slug: string;
  count: number;
  latestDate: Date;
}

export interface TagMap {
  [slug: string]: {
    name: string;
    slug: string;
    posts: CollectionEntry<'blog'>[];
  };
}

/**
 * Generate URL-safe slug from tag name
 *
 * Rules:
 * - Trim and collapse whitespace
 * - Convert to lowercase
 * - Handle Chinese/English/numbers/symbols
 * - Ensure URL-safe characters
 *
 * @param tagName - Tag name to slugify
 * @returns URL-safe slug
 *
 * @example
 * slugifyTag('AI Infra') // 'ai-infra'
 * slugifyTag('CUDA') // 'cuda'
 * slugifyTag('推理优化/部署') // '%E6%8E%A8%E7%90%86%E4%BC%98%E5%8C%96-%E9%83%A8%E7%BD%B2'
 * slugifyTag('  Multiple   Spaces  ') // 'multiple-spaces'
 */
export function slugifyTag(tagName: string): string {
  if (!tagName) return '';

  // Trim and normalize whitespace
  const normalized = tagName.trim().replace(/\s+/g, ' ');

  // Use normalizeSlug which handles all the complex cases
  const slug = normalizeSlug(normalized);

  // If normalizeSlug returns empty (e.g., Chinese-only), encode the original
  if (!slug && normalized) {
    // For Chinese/non-Latin characters, use encodeURIComponent
    return encodeURIComponent(normalized.toLowerCase().replace(/\s+/g, '-'));
  }

  return slug;
}

/**
 * Build tag index from posts
 *
 * Aggregates all tags from post frontmatter only. Does not auto-generate tags.
 * Handles disambiguation for tags that generate the same slug.
 *
 * @param posts - Array of blog posts
 * @returns Object with allTags array and tagMap object
 *
 * @example
 * const { allTags, tagMap } = buildTagIndex(posts);
 * // allTags: [{ name: 'AI', slug: 'ai', count: 5, latestDate: ... }, ...]
 * // tagMap: { 'ai': { name: 'AI', slug: 'ai', posts: [...] }, ... }
 */
export function buildTagIndex(posts: CollectionEntry<'blog'>[]) {
  // Group posts by tag name
  // Tags ONLY come from post.data.tags frontmatter field - no auto-generation
  const tagPostsMap = new Map<string, CollectionEntry<'blog'>[]>();

  posts.forEach((post) => {
    // Only use tags from frontmatter, filter out empty/invalid tags
    const tags = (post.data.tags || []).map((tag) => tag.trim()).filter((tag) => tag.length > 0);

    tags.forEach((tagName) => {
      const existing = tagPostsMap.get(tagName) || [];
      existing.push(post);
      tagPostsMap.set(tagName, existing);
    });
  });

  // Generate slugs and handle duplicates
  const slugUsage = new Map<string, string[]>(); // slug -> [tagName1, tagName2, ...]

  for (const [tagName] of tagPostsMap.entries()) {
    const slug = slugifyTag(tagName);
    const existing = slugUsage.get(slug) || [];
    existing.push(tagName);
    slugUsage.set(slug, existing);
  }

  // Create final tag data with unique slugs
  const tagMap: TagMap = {};
  const allTags: TagStats[] = [];

  // Sort tag names for consistent disambiguation
  const sortedTagNames = Array.from(tagPostsMap.keys()).sort();

  sortedTagNames.forEach((tagName) => {
    const posts = tagPostsMap.get(tagName)!;
    let slug = slugifyTag(tagName);

    // Handle disambiguation if multiple tags map to same slug
    const duplicates = slugUsage.get(slug) || [];
    if (duplicates.length > 1) {
      const index = duplicates.indexOf(tagName);
      if (index > 0) {
        // Add suffix for duplicates (keep first one clean)
        slug = `${slug}-${index + 1}`;
      }
    }

    // Calculate latest date
    const latestDate = posts.reduce((latest, post) => {
      const postDate = post.data.date;
      return postDate > latest ? postDate : latest;
    }, new Date(0));

    tagMap[slug] = {
      name: tagName,
      slug,
      posts: posts.sort((a, b) => b.data.date.valueOf() - a.data.date.valueOf()),
    };

    allTags.push({
      name: tagName,
      slug,
      count: posts.length,
      latestDate,
    });
  });

  // Sort tags by count (descending), then by name (ascending)
  allTags.sort((a, b) => {
    if (b.count !== a.count) {
      return b.count - a.count;
    }
    return a.name.localeCompare(b.name);
  });

  return { allTags, tagMap };
}

/**
 * Get posts by tag slug
 *
 * @param tagMap - Tag map from buildTagIndex
 * @param slug - Tag slug
 * @returns Array of posts for the tag, or undefined if not found
 */
export function getPostsByTagSlug(tagMap: TagMap, slug: string) {
  return tagMap[slug]?.posts;
}

/**
 * Sort tags by different criteria
 *
 * @param tags - Array of tag stats
 * @param sortBy - Sort criteria
 * @returns Sorted array of tags
 */
export function sortTags(tags: TagStats[], sortBy: 'count' | 'name' | 'recent'): TagStats[] {
  const sorted = [...tags];

  switch (sortBy) {
    case 'count':
      sorted.sort((a, b) => {
        if (b.count !== a.count) {
          return b.count - a.count;
        }
        return a.name.localeCompare(b.name);
      });
      break;
    case 'name':
      sorted.sort((a, b) => a.name.localeCompare(b.name));
      break;
    case 'recent':
      sorted.sort((a, b) => {
        if (b.latestDate.valueOf() !== a.latestDate.valueOf()) {
          return b.latestDate.valueOf() - a.latestDate.valueOf();
        }
        return a.name.localeCompare(b.name);
      });
      break;
  }

  return sorted;
}
