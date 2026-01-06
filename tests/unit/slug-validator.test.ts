/**
 * Tests for slug validation utilities
 */

import { describe, it, expect } from 'vitest';
import { isValidSlug, sanitizeSlug } from '../../src/lib/views/slug-validator';

describe('Slug Validator', () => {
  describe('isValidSlug', () => {
    it('should accept valid slugs', () => {
      expect(isValidSlug('hello-world')).toBe(true);
      expect(isValidSlug('my-post-123')).toBe(true);
      expect(isValidSlug('post')).toBe(true);
      expect(isValidSlug('a-b-c-d')).toBe(true);
      expect(isValidSlug('2024-01-15')).toBe(true);
    });

    it('should accept nested paths', () => {
      expect(isValidSlug('category/post')).toBe(true);
      expect(isValidSlug('blog/2024/my-post')).toBe(true);
    });

    it('should reject empty strings', () => {
      expect(isValidSlug('')).toBe(false);
    });

    it('should reject non-string values', () => {
      expect(isValidSlug(null as any)).toBe(false);
      expect(isValidSlug(undefined as any)).toBe(false);
      expect(isValidSlug(123 as any)).toBe(false);
    });

    it('should reject slugs with uppercase letters', () => {
      expect(isValidSlug('Hello-World')).toBe(false);
      expect(isValidSlug('myPost')).toBe(false);
    });

    it('should reject slugs with special characters', () => {
      expect(isValidSlug('hello world')).toBe(false);
      expect(isValidSlug('hello_world')).toBe(false);
      expect(isValidSlug('hello@world')).toBe(false);
      expect(isValidSlug('hello.world')).toBe(false);
    });

    it('should reject slugs starting or ending with hyphen', () => {
      expect(isValidSlug('-hello')).toBe(false);
      expect(isValidSlug('hello-')).toBe(false);
      expect(isValidSlug('-hello-')).toBe(false);
    });

    it('should reject slugs starting or ending with slash', () => {
      expect(isValidSlug('/hello')).toBe(false);
      expect(isValidSlug('hello/')).toBe(false);
      expect(isValidSlug('/hello/')).toBe(false);
    });

    it('should reject slugs with consecutive hyphens', () => {
      expect(isValidSlug('hello--world')).toBe(false);
    });

    it('should reject slugs with consecutive slashes', () => {
      expect(isValidSlug('hello//world')).toBe(false);
    });

    it('should reject very long slugs', () => {
      const longSlug = 'a'.repeat(201);
      expect(isValidSlug(longSlug)).toBe(false);
    });

    it('should accept slugs up to 200 characters', () => {
      const maxLengthSlug = 'a'.repeat(200);
      expect(isValidSlug(maxLengthSlug)).toBe(true);
    });
  });

  describe('sanitizeSlug', () => {
    it('should convert to lowercase', () => {
      expect(sanitizeSlug('Hello-World')).toBe('hello-world');
      expect(sanitizeSlug('MY-POST')).toBe('my-post');
    });

    it('should replace spaces with hyphens', () => {
      expect(sanitizeSlug('hello world')).toBe('hello-world');
      expect(sanitizeSlug('my post title')).toBe('my-post-title');
    });

    it('should replace special characters with hyphens', () => {
      expect(sanitizeSlug('hello_world')).toBe('hello-world');
      expect(sanitizeSlug('hello@world')).toBe('hello-world');
      expect(sanitizeSlug('hello.world')).toBe('hello-world');
    });

    it('should remove consecutive hyphens', () => {
      expect(sanitizeSlug('hello--world')).toBe('hello-world');
      expect(sanitizeSlug('hello---world')).toBe('hello-world');
    });

    it('should remove consecutive slashes', () => {
      expect(sanitizeSlug('hello//world')).toBe('hello/world');
      expect(sanitizeSlug('hello///world')).toBe('hello/world');
    });

    it('should trim leading and trailing hyphens', () => {
      expect(sanitizeSlug('-hello-')).toBe('hello');
      expect(sanitizeSlug('--hello--')).toBe('hello');
    });

    it('should trim leading and trailing slashes', () => {
      expect(sanitizeSlug('/hello/')).toBe('hello');
      expect(sanitizeSlug('//hello//')).toBe('hello');
    });

    it('should handle empty strings', () => {
      expect(sanitizeSlug('')).toBe('');
    });

    it('should handle non-string values', () => {
      expect(sanitizeSlug(null as any)).toBe('');
      expect(sanitizeSlug(undefined as any)).toBe('');
    });

    it('should handle mixed issues', () => {
      expect(sanitizeSlug('  Hello__World  ')).toBe('hello-world');
      expect(sanitizeSlug('/My/Post/Title/')).toBe('my/post/title');
    });
  });
});
