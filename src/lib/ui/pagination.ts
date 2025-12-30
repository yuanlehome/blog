/**
 * Pagination utility for generating page number arrays with ellipsis support
 *
 * This module provides windowed pagination to prevent UI crowding when
 * there are many pages. It always shows first and last pages, with a
 * window around the current page.
 *
 * @module src/lib/ui/pagination
 */

export type PaginationItem = number | 'ellipsis';

export interface PaginationOptions {
  currentPage: number;
  totalPages: number;
  windowSize?: number;
}

/**
 * Generate an array of page numbers with ellipsis for windowed pagination
 *
 * Rules:
 * - If totalPages ≤ windowSize: show all pages
 * - If totalPages > windowSize: show window around current page with ellipsis
 * - Always show first and last page
 * - Use 'ellipsis' to represent gaps
 *
 * @param options - Pagination configuration
 * @returns Array of page numbers and ellipsis indicators
 *
 * @example
 * getPaginationPages({ currentPage: 1, totalPages: 10, windowSize: 5 })
 * // [1, 2, 3, 4, 5, 'ellipsis', 10]
 *
 * @example
 * getPaginationPages({ currentPage: 5, totalPages: 10, windowSize: 5 })
 * // [1, 'ellipsis', 3, 4, 5, 6, 7, 'ellipsis', 10]
 *
 * @example
 * getPaginationPages({ currentPage: 9, totalPages: 10, windowSize: 5 })
 * // [1, 'ellipsis', 6, 7, 8, 9, 10]
 */
export function getPaginationPages(options: PaginationOptions): PaginationItem[] {
  const { currentPage, totalPages, windowSize = 5 } = options;

  // Validate inputs
  if (totalPages < 1) {
    throw new Error(`totalPages must be at least 1, got ${totalPages}`);
  }
  if (windowSize < 1) {
    throw new Error(`windowSize must be at least 1, got ${windowSize}`);
  }
  if (currentPage < 1 || currentPage > totalPages) {
    throw new Error(`currentPage ${currentPage} is out of range [1, ${totalPages}]`);
  }

  // If total pages fit in window, show all
  if (totalPages <= windowSize) {
    return Array.from({ length: totalPages }, (_, i) => i + 1);
  }

  const pages: PaginationItem[] = [];

  // Calculate the window range around current page
  // We want to show windowSize pages total in the middle section
  const halfWindow = Math.floor(windowSize / 2);
  let windowStart = Math.max(currentPage - halfWindow, 1);
  let windowEnd = Math.min(currentPage + halfWindow, totalPages);

  // Adjust window to maintain windowSize if near edges
  if (windowEnd - windowStart + 1 < windowSize) {
    if (windowStart === 1) {
      windowEnd = Math.min(windowSize, totalPages);
    } else if (windowEnd === totalPages) {
      windowStart = Math.max(totalPages - windowSize + 1, 1);
    }
  }

  // Always show first page
  pages.push(1);

  // Add ellipsis if there's a gap after first page
  if (windowStart > 2) {
    pages.push('ellipsis');
  }

  // Add pages in the window (excluding first and last if they're in range)
  for (let i = windowStart; i <= windowEnd; i++) {
    if (i > 1 && i < totalPages) {
      pages.push(i);
    }
  }

  // Add ellipsis if there's a gap before last page
  if (windowEnd < totalPages - 1) {
    pages.push('ellipsis');
  }

  // Always show last page (if not already shown)
  if (totalPages > 1) {
    pages.push(totalPages);
  }

  return pages;
}
