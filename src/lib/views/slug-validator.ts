/**
 * Slug validation utilities for views API
 */

/**
 * Validate slug format
 * Slugs should be lowercase alphanumeric with hyphens
 */
export function isValidSlug(slug: string): boolean {
  if (!slug || typeof slug !== 'string') {
    return false;
  }

  // Must be between 1 and 200 characters
  if (slug.length === 0 || slug.length > 200) {
    return false;
  }

  // Must match pattern: lowercase letters, numbers, hyphens, forward slashes
  // Allow forward slashes for nested paths
  const slugPattern = /^[a-z0-9-/]+$/;
  if (!slugPattern.test(slug)) {
    return false;
  }

  // Should not start or end with hyphen or slash
  if (slug.startsWith('-') || slug.endsWith('-') || slug.startsWith('/') || slug.endsWith('/')) {
    return false;
  }

  // Should not have consecutive hyphens or slashes
  if (slug.includes('--') || slug.includes('//')) {
    return false;
  }

  return true;
}

/**
 * Sanitize slug to ensure it's safe for API calls
 */
export function sanitizeSlug(slug: string): string {
  if (!slug || typeof slug !== 'string') {
    return '';
  }

  return slug
    .toLowerCase()
    .trim()
    .replace(/[^a-z0-9-/]/g, '-')
    .replace(/--+/g, '-')
    .replace(/\/\/+/g, '/')
    .replace(/^[-/]+|[-/]+$/g, '');
}
