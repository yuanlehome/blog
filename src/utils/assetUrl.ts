/**
 * Resolves asset URL by prefixing with BASE_URL if needed
 * Handles absolute paths that should be prefixed with the site base
 * 
 * @param url - The asset URL (can be relative, absolute, or external)
 * @returns The resolved URL with BASE_URL prefix if applicable
 */
export function resolveAssetUrl(url: string | undefined | null): string | undefined {
  if (!url) return undefined;
  
  // External URLs (http://, https://, //) should not be prefixed
  if (url.startsWith('http://') || url.startsWith('https://') || url.startsWith('//')) {
    return url;
  }
  
  // If it's an absolute path starting with /, prefix with BASE_URL
  if (url.startsWith('/')) {
    const base = import.meta.env.BASE_URL;
    const normalizedBase = base.endsWith('/') ? base.slice(0, -1) : base;
    return `${normalizedBase}${url}`;
  }
  
  // Relative paths are returned as-is
  return url;
}
