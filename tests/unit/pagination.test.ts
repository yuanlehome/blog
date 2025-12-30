import { describe, expect, it } from 'vitest';
import { getPaginationPages } from '../../src/lib/ui/pagination';

describe('getPaginationPages', () => {
  describe('input validation', () => {
    it('throws error when currentPage is less than 1', () => {
      expect(() => getPaginationPages({ currentPage: 0, totalPages: 10 })).toThrow(
        'currentPage 0 is out of range [1, 10]',
      );
    });

    it('throws error when currentPage exceeds totalPages', () => {
      expect(() => getPaginationPages({ currentPage: 11, totalPages: 10 })).toThrow(
        'currentPage 11 is out of range [1, 10]',
      );
    });

    it('throws error when totalPages is less than 1', () => {
      expect(() => getPaginationPages({ currentPage: 1, totalPages: 0 })).toThrow(
        'totalPages must be at least 1, got 0',
      );
    });

    it('throws error when windowSize is less than 1', () => {
      expect(() => getPaginationPages({ currentPage: 1, totalPages: 10, windowSize: 0 })).toThrow(
        'windowSize must be at least 1, got 0',
      );
    });
  });

  describe('when totalPages ≤ windowSize', () => {
    it('returns all pages when totalPages equals windowSize', () => {
      const result = getPaginationPages({ currentPage: 3, totalPages: 5, windowSize: 5 });
      expect(result).toEqual([1, 2, 3, 4, 5]);
    });

    it('returns all pages when totalPages is less than windowSize', () => {
      const result = getPaginationPages({ currentPage: 2, totalPages: 3, windowSize: 5 });
      expect(result).toEqual([1, 2, 3]);
    });

    it('returns single page when totalPages is 1', () => {
      const result = getPaginationPages({ currentPage: 1, totalPages: 1, windowSize: 5 });
      expect(result).toEqual([1]);
    });

    it('returns two pages when totalPages is 2', () => {
      const result = getPaginationPages({ currentPage: 1, totalPages: 2, windowSize: 5 });
      expect(result).toEqual([1, 2]);
    });
  });

  describe('when currentPage is at the start', () => {
    it('shows window at start with ellipsis before last page (page 1)', () => {
      const result = getPaginationPages({ currentPage: 1, totalPages: 10, windowSize: 5 });
      expect(result).toEqual([1, 2, 3, 4, 5, 'ellipsis', 10]);
    });

    it('shows window at start with ellipsis before last page (page 2)', () => {
      const result = getPaginationPages({ currentPage: 2, totalPages: 10, windowSize: 5 });
      expect(result).toEqual([1, 2, 3, 4, 5, 'ellipsis', 10]);
    });

    it('shows window at start with ellipsis before last page (page 3)', () => {
      const result = getPaginationPages({ currentPage: 3, totalPages: 10, windowSize: 5 });
      expect(result).toEqual([1, 2, 3, 4, 5, 'ellipsis', 10]);
    });

    it('shows window without ellipsis when near the transition point', () => {
      const result = getPaginationPages({ currentPage: 4, totalPages: 10, windowSize: 5 });
      expect(result).toEqual([1, 2, 3, 4, 5, 6, 'ellipsis', 10]);
    });
  });

  describe('when currentPage is in the middle', () => {
    it('shows window with ellipsis on both sides (page 5)', () => {
      const result = getPaginationPages({ currentPage: 5, totalPages: 10, windowSize: 5 });
      expect(result).toEqual([1, 'ellipsis', 3, 4, 5, 6, 7, 'ellipsis', 10]);
    });

    it('shows window with ellipsis on both sides (page 6)', () => {
      const result = getPaginationPages({ currentPage: 6, totalPages: 10, windowSize: 5 });
      expect(result).toEqual([1, 'ellipsis', 4, 5, 6, 7, 8, 'ellipsis', 10]);
    });

    it('shows centered window for page 5 of 12', () => {
      const result = getPaginationPages({ currentPage: 5, totalPages: 12, windowSize: 5 });
      expect(result).toEqual([1, 'ellipsis', 3, 4, 5, 6, 7, 'ellipsis', 12]);
    });
  });

  describe('when currentPage is at the end', () => {
    it('shows window at end with ellipsis after first page (page 10)', () => {
      const result = getPaginationPages({ currentPage: 10, totalPages: 10, windowSize: 5 });
      expect(result).toEqual([1, 'ellipsis', 6, 7, 8, 9, 10]);
    });

    it('shows window at end with ellipsis after first page (page 9)', () => {
      const result = getPaginationPages({ currentPage: 9, totalPages: 10, windowSize: 5 });
      expect(result).toEqual([1, 'ellipsis', 6, 7, 8, 9, 10]);
    });

    it('shows window at end with ellipsis after first page (page 8)', () => {
      const result = getPaginationPages({ currentPage: 8, totalPages: 10, windowSize: 5 });
      expect(result).toEqual([1, 'ellipsis', 6, 7, 8, 9, 10]);
    });

    it('shows window without ellipsis when near the transition point', () => {
      const result = getPaginationPages({ currentPage: 7, totalPages: 10, windowSize: 5 });
      expect(result).toEqual([1, 'ellipsis', 5, 6, 7, 8, 9, 10]);
    });
  });

  describe('different windowSize values', () => {
    it('works with windowSize of 3', () => {
      const result = getPaginationPages({ currentPage: 5, totalPages: 10, windowSize: 3 });
      expect(result).toEqual([1, 'ellipsis', 4, 5, 6, 'ellipsis', 10]);
    });

    it('works with windowSize of 7', () => {
      const result = getPaginationPages({ currentPage: 8, totalPages: 15, windowSize: 7 });
      expect(result).toEqual([1, 'ellipsis', 5, 6, 7, 8, 9, 10, 11, 'ellipsis', 15]);
    });

    it('works with windowSize of 1 (minimal)', () => {
      const result = getPaginationPages({ currentPage: 5, totalPages: 10, windowSize: 1 });
      expect(result).toEqual([1, 'ellipsis', 5, 'ellipsis', 10]);
    });
  });

  describe('edge cases', () => {
    it('handles total pages exactly windowSize + 1', () => {
      const result = getPaginationPages({ currentPage: 3, totalPages: 6, windowSize: 5 });
      expect(result).toEqual([1, 2, 3, 4, 5, 6]);
    });

    it('handles total pages exactly windowSize + 2', () => {
      const result = getPaginationPages({ currentPage: 4, totalPages: 7, windowSize: 5 });
      expect(result).toEqual([1, 2, 3, 4, 5, 6, 7]);
    });

    it('uses default windowSize of 5 when not provided', () => {
      const result = getPaginationPages({ currentPage: 5, totalPages: 10 });
      expect(result).toEqual([1, 'ellipsis', 3, 4, 5, 6, 7, 'ellipsis', 10]);
    });
  });

  describe('realistic blog scenarios', () => {
    it('handles a blog with 25 posts (5 per page)', () => {
      const result = getPaginationPages({ currentPage: 3, totalPages: 5 });
      expect(result).toEqual([1, 2, 3, 4, 5]);
    });

    it('handles a blog with 50 posts (10 pages)', () => {
      const result = getPaginationPages({ currentPage: 5, totalPages: 10 });
      expect(result).toEqual([1, 'ellipsis', 3, 4, 5, 6, 7, 'ellipsis', 10]);
    });

    it('handles a blog with 100 posts (20 pages) at the beginning', () => {
      const result = getPaginationPages({ currentPage: 2, totalPages: 20 });
      expect(result).toEqual([1, 2, 3, 4, 5, 'ellipsis', 20]);
    });

    it('handles a blog with 100 posts (20 pages) in the middle', () => {
      const result = getPaginationPages({ currentPage: 10, totalPages: 20 });
      expect(result).toEqual([1, 'ellipsis', 8, 9, 10, 11, 12, 'ellipsis', 20]);
    });

    it('handles a blog with 100 posts (20 pages) near the end', () => {
      const result = getPaginationPages({ currentPage: 18, totalPages: 20 });
      expect(result).toEqual([1, 'ellipsis', 16, 17, 18, 19, 20]);
    });
  });
});
