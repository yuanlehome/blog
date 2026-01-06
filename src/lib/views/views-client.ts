/**
 * Views API client implementation
 */

import type { ViewsProvider, ViewsProviderConfig, ViewsResponse, ViewsIncrementResponse } from './types';
import { isValidSlug } from './slug-validator';

/**
 * Default timeout for API calls (5 seconds)
 */
const DEFAULT_TIMEOUT = 5000;

/**
 * Create an AbortSignal with timeout
 */
function createTimeoutSignal(timeout: number): AbortSignal {
  const controller = new AbortController();
  setTimeout(() => controller.abort(), timeout);
  return controller.signal;
}

/**
 * HTTP-based views provider implementation
 */
export class HttpViewsProvider implements ViewsProvider {
  private apiEndpoint: string;
  private timeout: number;

  constructor(config: ViewsProviderConfig) {
    this.apiEndpoint = config.apiEndpoint.replace(/\/$/, ''); // Remove trailing slash
    this.timeout = config.timeout || DEFAULT_TIMEOUT;
  }

  async getViews(slug: string): Promise<ViewsResponse> {
    if (!isValidSlug(slug)) {
      throw new Error(`Invalid slug: ${slug}`);
    }

    const url = `${this.apiEndpoint}/api/views?slug=${encodeURIComponent(slug)}`;
    const signal = createTimeoutSignal(this.timeout);

    try {
      const response = await fetch(url, { signal });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const data = await response.json();
      return {
        slug: data.slug,
        views: Number(data.views) || 0,
      };
    } catch (error) {
      if (error instanceof Error && error.name === 'AbortError') {
        throw new Error('Request timeout');
      }
      throw error;
    }
  }

  async incrementViews(slug: string, clientId: string): Promise<ViewsIncrementResponse> {
    if (!isValidSlug(slug)) {
      throw new Error(`Invalid slug: ${slug}`);
    }

    if (!clientId) {
      throw new Error('Client ID is required');
    }

    const url = `${this.apiEndpoint}/api/views/incr?slug=${encodeURIComponent(slug)}`;
    const signal = createTimeoutSignal(this.timeout);

    try {
      const response = await fetch(url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ clientId }),
        signal,
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const data = await response.json();
      return {
        slug: data.slug,
        views: Number(data.views) || 0,
        counted: Boolean(data.counted),
      };
    } catch (error) {
      if (error instanceof Error && error.name === 'AbortError') {
        throw new Error('Request timeout');
      }
      throw error;
    }
  }
}

/**
 * Mock views provider for development/testing
 * Stores data in memory only
 */
export class MockViewsProvider implements ViewsProvider {
  private views: Map<string, number> = new Map();
  private lastViewed: Map<string, number> = new Map(); // clientId:slug -> timestamp

  async getViews(slug: string): Promise<ViewsResponse> {
    if (!isValidSlug(slug)) {
      throw new Error(`Invalid slug: ${slug}`);
    }

    return {
      slug,
      views: this.views.get(slug) || 0,
    };
  }

  async incrementViews(slug: string, clientId: string): Promise<ViewsIncrementResponse> {
    if (!isValidSlug(slug)) {
      throw new Error(`Invalid slug: ${slug}`);
    }

    if (!clientId) {
      throw new Error('Client ID is required');
    }

    const key = `${clientId}:${slug}`;
    const now = Date.now();
    const lastView = this.lastViewed.get(key) || 0;
    const hoursSinceLastView = (now - lastView) / (1000 * 60 * 60);

    let counted = false;

    // Only count if more than 24 hours have passed
    if (hoursSinceLastView >= 24 || lastView === 0) {
      const currentViews = this.views.get(slug) || 0;
      this.views.set(slug, currentViews + 1);
      this.lastViewed.set(key, now);
      counted = true;
    }

    return {
      slug,
      views: this.views.get(slug) || 0,
      counted,
    };
  }
}

/**
 * Create a views provider based on environment
 */
export function createViewsProvider(config?: Partial<ViewsProviderConfig>): ViewsProvider {
  // If no API endpoint is provided, use mock provider
  if (!config?.apiEndpoint) {
    return new MockViewsProvider();
  }

  return new HttpViewsProvider({
    apiEndpoint: config.apiEndpoint,
    timeout: config.timeout,
  });
}
