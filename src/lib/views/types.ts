/**
 * Types for page views functionality
 */

/**
 * Response from GET /api/views?slug=<slug>
 */
export interface ViewsResponse {
  slug: string;
  views: number;
}

/**
 * Response from POST /api/views/incr?slug=<slug>
 */
export interface ViewsIncrementResponse {
  slug: string;
  views: number;
  counted: boolean;
}

/**
 * Configuration for views provider
 */
export interface ViewsProviderConfig {
  apiEndpoint: string;
  timeout?: number;
}

/**
 * Abstract interface for views provider
 * Allows switching between different backend implementations
 */
export interface ViewsProvider {
  /**
   * Get current view count for a slug
   */
  getViews(slug: string): Promise<ViewsResponse>;

  /**
   * Increment view count for a slug
   * Returns updated count and whether the view was counted
   */
  incrementViews(slug: string, clientId: string): Promise<ViewsIncrementResponse>;
}
