/**
 * Tests for alignment utility functions
 */

import { describe, it, expect } from 'vitest';
import {
  alignToTextClass,
  alignToJustifyClass,
  alignToItemsClass,
} from '../../src/lib/ui/alignment';

describe('Alignment Utilities', () => {
  describe('alignToTextClass', () => {
    it('should return text-left for left alignment', () => {
      expect(alignToTextClass('left')).toBe('text-left');
    });

    it('should return text-center for center alignment', () => {
      expect(alignToTextClass('center')).toBe('text-center');
    });
  });

  describe('alignToJustifyClass', () => {
    it('should return justify-start for left alignment', () => {
      expect(alignToJustifyClass('left')).toBe('justify-start');
    });

    it('should return justify-center for center alignment', () => {
      expect(alignToJustifyClass('center')).toBe('justify-center');
    });
  });

  describe('alignToItemsClass', () => {
    it('should return items-start for left alignment', () => {
      expect(alignToItemsClass('left')).toBe('items-start');
    });

    it('should return items-center for center alignment', () => {
      expect(alignToItemsClass('center')).toBe('items-center');
    });
  });
});
