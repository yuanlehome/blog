import { withBaseFromEnv } from './withBase';

/**
 * Resolves asset URL by prefixing with BASE_URL if needed
 * Handles absolute paths that should be prefixed with the site base
 *
 * @param url - The asset URL (can be relative, absolute, or external)
 * @returns The resolved URL with BASE_URL prefix if applicable
 */
export function resolveAssetUrl(url: string | undefined | null): string | undefined {
  if (!url) return undefined;
  return withBaseFromEnv(url);
}
