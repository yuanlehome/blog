/**
 * Tests for views client
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import { HttpViewsProvider, MockViewsProvider } from '../../src/lib/views/views-client';

describe('Views Client', () => {
  describe('MockViewsProvider', () => {
    let provider: MockViewsProvider;

    beforeEach(() => {
      provider = new MockViewsProvider();
    });

    it('should return 0 views for new slug', async () => {
      const result = await provider.getViews('test-post');
      expect(result).toEqual({
        slug: 'test-post',
        views: 0,
      });
    });

    it('should increment views for first visit', async () => {
      const result = await provider.incrementViews('test-post', 'client-123');
      expect(result).toEqual({
        slug: 'test-post',
        views: 1,
        counted: true,
      });
    });

    it('should not increment views within 24 hours', async () => {
      // First visit
      await provider.incrementViews('test-post', 'client-123');

      // Second visit immediately
      const result = await provider.incrementViews('test-post', 'client-123');
      expect(result).toEqual({
        slug: 'test-post',
        views: 1,
        counted: false,
      });
    });

    it('should increment views from different clients', async () => {
      await provider.incrementViews('test-post', 'client-123');
      const result = await provider.incrementViews('test-post', 'client-456');

      expect(result).toEqual({
        slug: 'test-post',
        views: 2,
        counted: true,
      });
    });

    it('should track views separately for different slugs', async () => {
      await provider.incrementViews('post-1', 'client-123');
      await provider.incrementViews('post-2', 'client-123');

      const result1 = await provider.getViews('post-1');
      const result2 = await provider.getViews('post-2');

      expect(result1.views).toBe(1);
      expect(result2.views).toBe(1);
    });

    it('should reject invalid slugs', async () => {
      await expect(provider.getViews('Invalid Slug')).rejects.toThrow('Invalid slug');
      await expect(provider.incrementViews('Invalid Slug', 'client-123')).rejects.toThrow(
        'Invalid slug',
      );
    });

    it('should reject empty client ID', async () => {
      await expect(provider.incrementViews('test-post', '')).rejects.toThrow(
        'Client ID is required',
      );
    });
  });

  describe('HttpViewsProvider', () => {
    let provider: HttpViewsProvider;
    const mockFetch = vi.fn();

    beforeEach(() => {
      global.fetch = mockFetch;
      mockFetch.mockReset();
      provider = new HttpViewsProvider({ apiEndpoint: 'https://api.example.com' });
    });

    it('should fetch views successfully', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ slug: 'test-post', views: 42 }),
      });

      const result = await provider.getViews('test-post');

      expect(result).toEqual({
        slug: 'test-post',
        views: 42,
      });

      expect(mockFetch).toHaveBeenCalledWith(
        'https://api.example.com/api/views?slug=test-post',
        expect.objectContaining({
          signal: expect.any(AbortSignal),
        }),
      );
    });

    it('should increment views successfully', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ slug: 'test-post', views: 43, counted: true }),
      });

      const result = await provider.incrementViews('test-post', 'client-123');

      expect(result).toEqual({
        slug: 'test-post',
        views: 43,
        counted: true,
      });

      expect(mockFetch).toHaveBeenCalledWith(
        'https://api.example.com/api/views/incr?slug=test-post',
        expect.objectContaining({
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ clientId: 'client-123' }),
          signal: expect.any(AbortSignal),
        }),
      );
    });

    it('should handle HTTP errors', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 404,
        statusText: 'Not Found',
      });

      await expect(provider.getViews('test-post')).rejects.toThrow('HTTP 404: Not Found');
    });

    it('should reject invalid slugs', async () => {
      await expect(provider.getViews('Invalid Slug')).rejects.toThrow('Invalid slug');
      await expect(provider.incrementViews('Invalid Slug', 'client-123')).rejects.toThrow(
        'Invalid slug',
      );
    });

    it('should reject empty client ID on increment', async () => {
      await expect(provider.incrementViews('test-post', '')).rejects.toThrow(
        'Client ID is required',
      );
    });

    it('should handle network errors', async () => {
      mockFetch.mockRejectedValueOnce(new Error('Network error'));

      await expect(provider.getViews('test-post')).rejects.toThrow('Network error');
    });

    it('should encode slug in URL', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ slug: 'test/post', views: 1 }),
      });

      await provider.getViews('test/post');

      expect(mockFetch).toHaveBeenCalledWith(
        'https://api.example.com/api/views?slug=test%2Fpost',
        expect.anything(),
      );
    });
  });
});
