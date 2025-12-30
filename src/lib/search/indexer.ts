/**
 * Search index generator utilities
 *
 * This module provides functions for generating the search index
 * from blog posts, including text extraction and processing.
 *
 * @module src/lib/search/indexer
 */

import type { SearchIndexEntry, SearchIndex } from './types';

/**
 * Maximum length of body text to include in index (characters)
 */
const MAX_BODY_LENGTH = 20000;

/**
 * Remove markdown code blocks from text
 */
export function removeCodeBlocks(text: string): string {
  // Remove fenced code blocks (``` or ~~~)
  return text.replace(/```[\s\S]*?```|~~~[\s\S]*?~~~/g, '');
}

/**
 * Remove markdown frontmatter from text
 */
export function removeFrontmatter(text: string): string {
  // Remove YAML frontmatter at the start
  return text.replace(/^---[\s\S]*?---\n?/, '');
}

/**
 * Remove HTML tags from text
 */
export function removeHtmlTags(text: string): string {
  return text.replace(/<[^>]+>/g, '');
}

/**
 * Remove markdown inline formatting but keep text
 */
export function removeMarkdownFormatting(text: string): string {
  return (
    text
      // Remove inline code
      .replace(/`[^`]+`/g, '')
      // Remove images
      .replace(/!\[([^\]]*)\]\([^)]+\)/g, '')
      // Remove links but keep text
      .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1')
      // Remove bold/italic
      .replace(/\*\*([^*]+)\*\*/g, '$1')
      .replace(/\*([^*]+)\*/g, '$1')
      .replace(/__([^_]+)__/g, '$1')
      .replace(/_([^_]+)_/g, '$1')
      // Remove strikethrough
      .replace(/~~([^~]+)~~/g, '$1')
      // Remove blockquotes
      .replace(/^>\s*/gm, '')
      // Remove horizontal rules
      .replace(/^[-*_]{3,}\s*$/gm, '')
      // Remove list markers
      .replace(/^[\s]*[-*+]\s+/gm, '')
      .replace(/^[\s]*\d+\.\s+/gm, '')
  );
}

/**
 * Extract headings from markdown content
 */
export function extractHeadings(text: string): string[] {
  const headings: string[] = [];
  const headingRegex = /^#{1,6}\s+(.+)$/gm;
  let match;

  while ((match = headingRegex.exec(text)) !== null) {
    const heading = match[1].trim();
    if (heading) {
      headings.push(heading);
    }
  }

  return headings;
}

/**
 * Convert markdown to plain text for indexing
 */
export function markdownToPlainText(markdown: string): string {
  let text = markdown;

  // Remove frontmatter
  text = removeFrontmatter(text);

  // Remove code blocks
  text = removeCodeBlocks(text);

  // Remove HTML tags
  text = removeHtmlTags(text);

  // Remove markdown formatting
  text = removeMarkdownFormatting(text);

  // Normalize whitespace
  text = text.replace(/\n+/g, ' ').replace(/\s+/g, ' ').trim();

  return text;
}

/**
 * Truncate text to a maximum length, preserving word boundaries
 */
export function truncateText(text: string, maxLength: number): string {
  if (text.length <= maxLength) {
    return text;
  }

  // Find the last space before maxLength
  const truncated = text.substring(0, maxLength);
  const lastSpace = truncated.lastIndexOf(' ');

  if (lastSpace > maxLength * 0.8) {
    return truncated.substring(0, lastSpace) + '...';
  }

  return truncated + '...';
}

/**
 * Extract source type from the post path
 */
export function extractSourceType(filePath: string): string | undefined {
  if (filePath.includes('/notion/')) return 'notion';
  if (filePath.includes('/wechat/')) return 'wechat';
  if (filePath.includes('/others/')) return 'others';
  return undefined;
}

/**
 * Create a search index entry from a blog post
 */
export function createSearchEntry(
  slug: string,
  url: string,
  title: string,
  body: string,
  tags: string[],
  date: Date,
  summary?: string,
  source?: string,
): SearchIndexEntry {
  // Remove frontmatter first
  const cleanBody = removeFrontmatter(body);

  // Extract headings before removing all formatting
  const headings = extractHeadings(cleanBody);

  // Convert to plain text
  const plainText = markdownToPlainText(cleanBody);

  // Truncate body
  const truncatedBody = truncateText(plainText, MAX_BODY_LENGTH);

  return {
    slug,
    url,
    title,
    headings,
    tags,
    date: date.toISOString(),
    summary: summary || truncateText(plainText, 200),
    body: truncatedBody,
    source,
  };
}

/**
 * Calculate tag counts from entries
 */
export function calculateTagCounts(entries: SearchIndexEntry[]): Record<string, number> {
  const counts: Record<string, number> = {};

  for (const entry of entries) {
    for (const tag of entry.tags) {
      counts[tag] = (counts[tag] || 0) + 1;
    }
  }

  return counts;
}

/**
 * Create a complete search index from entries
 */
export function createSearchIndex(entries: SearchIndexEntry[]): SearchIndex {
  return {
    version: 1,
    generatedAt: new Date().toISOString(),
    count: entries.length,
    tags: calculateTagCounts(entries),
    entries,
  };
}
