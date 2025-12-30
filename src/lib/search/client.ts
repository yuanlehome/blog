/**
 * Search client for browser usage
 *
 * This module provides a client-side interface to the search functionality,
 * managing the worker and providing fallback for non-worker environments.
 *
 * @module src/lib/search/client
 */

import type { SearchIndex, SearchQuery, SearchResponse } from './types';
import { SearchEngine } from './engine';

/**
 * Search client configuration
 */
export interface SearchClientConfig {
  useWorker: boolean;
  weights: {
    title: number;
    headings: number;
    tags: number;
    summary: number;
    body: number;
  };
  snippetWindow: number;
  maxResults: number;
}

/**
 * Default configuration
 */
const defaultConfig: SearchClientConfig = {
  useWorker: true,
  weights: { title: 6, headings: 3, tags: 3, summary: 2, body: 1 },
  snippetWindow: 80,
  maxResults: 12,
};

/**
 * Search client class
 */
export class SearchClient {
  private worker: Worker | null = null;
  private engine: SearchEngine | null = null;
  private config: SearchClientConfig;
  private initialized = false;
  private initPromise: Promise<void> | null = null;
  private pendingSearches = new Map<
    number,
    { resolve: (value: SearchResponse) => void; reject: (reason: unknown) => void }
  >();
  private searchId = 0;

  constructor(config: Partial<SearchClientConfig> = {}) {
    this.config = { ...defaultConfig, ...config };
  }

  /**
   * Check if Web Workers are supported
   */
  private supportsWorker(): boolean {
    return typeof Worker !== 'undefined';
  }

  /**
   * Create and setup the worker
   */
  private setupWorker(): Worker {
    // Create worker from the worker file
    const worker = new Worker(new URL('./worker.ts', import.meta.url), {
      type: 'module',
    });

    worker.onmessage = (event: MessageEvent) => {
      const { type, payload } = event.data;

      switch (type) {
        case 'ready':
          this.initialized = true;
          break;

        case 'results': {
          const search = this.pendingSearches.get(this.searchId - 1);
          if (search) {
            search.resolve(payload as SearchResponse);
            this.pendingSearches.delete(this.searchId - 1);
          }
          break;
        }

        case 'error': {
          const search = this.pendingSearches.get(this.searchId - 1);
          if (search) {
            search.reject(new Error(payload as string));
            this.pendingSearches.delete(this.searchId - 1);
          }
          break;
        }
      }
    };

    worker.onerror = (error) => {
      console.error('Search worker error:', error);
      // Fall back to main thread
      this.worker = null;
      this.engine = new SearchEngine(this.config.weights, this.config.snippetWindow);
    };

    return worker;
  }

  /**
   * Initialize the search client with the index
   */
  async initialize(index: SearchIndex): Promise<void> {
    if (this.initPromise) {
      return this.initPromise;
    }

    this.initPromise = new Promise((resolve, reject) => {
      try {
        if (this.config.useWorker && this.supportsWorker()) {
          this.worker = this.setupWorker();

          // Set up ready handler
          const originalHandler = this.worker.onmessage;
          this.worker.onmessage = (event: MessageEvent) => {
            if (event.data.type === 'ready') {
              this.initialized = true;
              this.worker!.onmessage = originalHandler;
              resolve();
            } else if (event.data.type === 'error') {
              reject(new Error(event.data.payload));
            }
            // Call original handler
            if (originalHandler && this.worker) {
              originalHandler.call(this.worker, event);
            }
          };

          // Send init message
          this.worker.postMessage({
            type: 'init',
            payload: {
              index,
              config: {
                weights: this.config.weights,
                snippetWindow: this.config.snippetWindow,
              },
            },
          });
        } else {
          // Use main thread engine
          this.engine = new SearchEngine(this.config.weights, this.config.snippetWindow);
          this.engine.initialize(index.entries);
          this.initialized = true;
          resolve();
        }
      } catch (error) {
        // Fall back to main thread
        this.worker = null;
        this.engine = new SearchEngine(this.config.weights, this.config.snippetWindow);
        this.engine.initialize(index.entries);
        this.initialized = true;
        resolve();
      }
    });

    return this.initPromise;
  }

  /**
   * Check if the client is initialized
   */
  isInitialized(): boolean {
    return this.initialized;
  }

  /**
   * Perform a search
   */
  async search(query: SearchQuery): Promise<SearchResponse> {
    if (!this.initialized) {
      throw new Error('Search client not initialized');
    }

    // Apply default max results
    query.maxResults = query.maxResults || this.config.maxResults;

    if (this.worker) {
      return new Promise((resolve, reject) => {
        const id = this.searchId++;
        this.pendingSearches.set(id, { resolve, reject });
        this.worker!.postMessage({ type: 'search', payload: query });
      });
    }

    if (this.engine) {
      return this.engine.search(query);
    }

    throw new Error('Search client not properly initialized');
  }

  /**
   * Terminate the worker
   */
  terminate(): void {
    if (this.worker) {
      this.worker.terminate();
      this.worker = null;
    }
    this.engine = null;
    this.initialized = false;
    this.initPromise = null;
    this.pendingSearches.clear();
  }
}

/**
 * Global search client instance
 */
let globalClient: SearchClient | null = null;

/**
 * Get the global search client
 */
export function getSearchClient(config?: Partial<SearchClientConfig>): SearchClient {
  if (!globalClient) {
    globalClient = new SearchClient(config);
  }
  return globalClient;
}

/**
 * Reset the global search client (useful for testing)
 */
export function resetSearchClient(): void {
  if (globalClient) {
    globalClient.terminate();
    globalClient = null;
  }
}
