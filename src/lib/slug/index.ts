/**
 * Slug Module - Single Source of Truth for All Slug Operations
 *
 * This module provides a unified API for slug generation, normalization,
 * and URL construction across the entire codebase. All slug-related logic
 * must go through this module to ensure consistency.
 *
 * @module src/lib/slug
 */

import crypto from 'crypto';
import slugify from 'slugify';
import { siteBase } from '../../config/site';

/**
 * Normalize a string into a valid slug
 *
 * Handles:
 * - Chinese characters, emoji, mixed scripts
 * - Consecutive spaces, hyphens, underscores
 * - Special characters and symbols
 * - Lowercase conversion
 *
 * @param value - Input string to normalize
 * @returns Normalized slug string (lowercase, alphanumeric + hyphens)
 *
 * @example
 * normalizeSlug('Hello World') // 'hello-world'
 * normalizeSlug('你好世界') // '' (Chinese removed with strict:true)
 * normalizeSlug('Hello  World!!!') // 'hello-world'
 * normalizeSlug('😀 Emoji Test') // 'emoji-test'
 */
export function normalizeSlug(value: string): string {
  if (!value) return '';
  return slugify(value, { lower: true, strict: true }) || '';
}

/**
 * Generate slug from title with fallback options
 *
 * Priority order:
 * 1. Explicit slug (if provided)
 * 2. Title-derived slug
 * 3. Fallback ID
 *
 * @param options - Slug derivation options
 * @returns Derived slug
 *
 * @example
 * slugFromTitle({ title: 'My Post' }) // 'my-post'
 * slugFromTitle({ explicitSlug: 'custom', title: 'My Post' }) // 'custom'
 * slugFromTitle({ title: '', fallbackId: 'page-123' }) // 'page-123'
 */
export function slugFromTitle(options: {
  explicitSlug?: string | null;
  title?: string;
  fallbackId?: string;
}): string {
  const { explicitSlug, title, fallbackId = '' } = options;

  // Priority 1: Explicit slug
  const explicit = explicitSlug ? normalizeSlug(explicitSlug) : '';
  if (explicit) return explicit;

  // Priority 2: Title
  const fromTitle = title ? normalizeSlug(title) : '';
  if (fromTitle) return fromTitle;

  // Priority 3: Fallback ID
  return normalizeSlug(fallbackId) || fallbackId;
}

/**
 * Generate slug from file stem (filename without extension)
 *
 * Ensures compatibility with Astro's default behavior where
 * filename becomes the route slug.
 *
 * @param stem - File stem (e.g., 'hello-world' from 'hello-world.md')
 * @returns Normalized slug
 *
 * @example
 * slugFromFileStem('hello-world') // 'hello-world'
 * slugFromFileStem('Hello World') // 'hello-world'
 * slugFromFileStem('2024-01-01-post') // '2024-01-01-post'
 */
export function slugFromFileStem(stem: string): string {
  return normalizeSlug(stem);
}

/**
 * Generate short hash for conflict resolution
 *
 * Used internally by ensureUniqueSlug to create unique suffixes.
 *
 * @param input - Input string to hash
 * @param length - Hash length (default: 6)
 * @returns Hex string of specified length
 *
 * @internal
 */
export function shortHash(input: string, length = 6): string {
  return crypto.createHash('sha256').update(input).digest('hex').slice(0, length);
}

/**
 * Ensure slug is unique among existing slugs
 *
 * If the desired slug conflicts with an existing one:
 * 1. Tries `slug-{hash}` where hash is derived from ownerId
 * 2. Falls back to `slug-{hash}-{counter}` if still conflicting
 *
 * @param desired - Desired slug
 * @param ownerId - Unique identifier for this content (e.g., Notion page ID)
 * @param used - Map of slug → ownerId for existing content
 * @returns Unique slug, registered in the used map
 *
 * @example
 * const used = new Map();
 * ensureUniqueSlug('post', 'id1', used) // 'post'
 * ensureUniqueSlug('post', 'id2', used) // 'post-a1b2c3' (hash of id2)
 * ensureUniqueSlug('post', 'id1', used) // 'post' (same owner)
 */
export function ensureUniqueSlug(
  desired: string,
  ownerId: string,
  used: Map<string, string>,
): string {
  let slug = desired || normalizeSlug(ownerId);
  const existingOwner = used.get(slug);

  // If slug is available or owned by same entity, use it
  if (!existingOwner || existingOwner === ownerId) {
    used.set(slug, ownerId);
    return slug;
  }

  // Try with hash suffix
  const candidate = `${slug}-${shortHash(ownerId)}`;
  const candidateOwner = used.get(candidate);
  if (!candidateOwner || candidateOwner === ownerId) {
    used.set(candidate, ownerId);
    return candidate;
  }

  // Fall back to counter suffix
  let counter = 2;
  let finalSlug = candidate;
  while (used.has(finalSlug) && used.get(finalSlug) !== ownerId) {
    finalSlug = `${candidate}-${counter}`;
    counter += 1;
  }
  used.set(finalSlug, ownerId);
  return finalSlug;
}

/**
 * Ensure uniqueness for batch of slugs
 *
 * Takes an array of items with slugs and ensures all are unique.
 * Returns a map of original slug → final unique slug.
 *
 * @param items - Array of items with slug and id
 * @returns Map of original slug → final unique slug
 *
 * @example
 * const items = [
 *   { id: '1', slug: 'post' },
 *   { id: '2', slug: 'post' },
 *   { id: '3', slug: 'article' }
 * ];
 * const result = ensureUniqueSlugs(items);
 * // Map {
 * //   'post' => 'post',
 * //   'post' => 'post-abc123',
 * //   'article' => 'article'
 * // }
 */
export function ensureUniqueSlugs(items: Array<{ id: string; slug: string }>): Map<string, string> {
  const used = new Map<string, string>();
  const result = new Map<string, string>();

  for (const item of items) {
    const originalSlug = item.slug;
    const uniqueSlug = ensureUniqueSlug(originalSlug, item.id, used);
    result.set(item.id, uniqueSlug);
  }

  return result;
}

/**
 * Normalize BASE_URL to ensure it ends with /
 *
 * @param base - Base URL (e.g., import.meta.env.BASE_URL)
 * @returns Normalized base with trailing slash
 *
 * @example
 * normalizeBase('/blog') // '/blog/'
 * normalizeBase('/blog/') // '/blog/'
 * normalizeBase('/') // '/'
 */
export function normalizeBase(base: string): string {
  if (!base) return '/';
  return base.endsWith('/') ? base : `${base}/`;
}

/**
 * Build full post URL from slug
 *
 * Combines BASE_URL with slug and ensures proper trailing slash.
 * This is the ONLY function that should be used for constructing post URLs.
 *
 * @param slug - Post slug
 * @param base - Optional base URL override (defaults to siteBase from config)
 * @returns Full post URL with trailing slash
 *
 * @example
 * buildPostUrl('my-post') // '/blog/my-post/' (assuming siteBase = '/blog/')
 * buildPostUrl('my-post', '/') // '/my-post/'
 * buildPostUrl('my-post', '/custom') // '/custom/my-post/'
 */
export function buildPostUrl(slug: string, base?: string): string {
  const normalizedBase = normalizeBase(base ?? siteBase);
  const normalizedSlug = slug.endsWith('/') ? slug.slice(0, -1) : slug;

  // Handle edge case where base is '/' and slug already has it
  if (normalizedBase === '/') {
    return `/${normalizedSlug}/`;
  }

  return `${normalizedBase}${normalizedSlug}/`;
}
