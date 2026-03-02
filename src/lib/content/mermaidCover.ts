/**
 * Utilities for detecting Mermaid-generated cover images from raw post bodies.
 */

import {
  mermaidImagePath,
  parseMermaidOptions,
  resolveMermaidSlug,
} from '../markdown/remarkMermaid';

/**
 * Extracts the expected Mermaid-generated light cover image URL from a raw post body.
 *
 * @param body - Raw post body content
 * @param postId - Content collection entry ID (e.g. "notion/my-post.md"). Used to
 *   derive the same directory-based slug that `render-mermaid.mjs` uses when writing
 *   SVG files to `public/generated/mermaid/<dirSlug>/`.  Passing `post.slug` (the URL
 *   slug) instead would produce a mismatched path and a broken cover image.
 */
export function getMermaidCoverFromBody(body: string, postId?: string): string | undefined {
  const match = body.match(/^```mermaid([^\n]*)\n([\s\S]*?)^```/m);
  if (!match?.[2]) return undefined;
  const meta = match[1]?.trim();
  const code = match[2].trimEnd();
  if (!code) return undefined;
  // Derive the same directory slug that render-mermaid.mjs uses (e.g. "notion")
  // from the content-collection entry id (e.g. "notion/my-post.md").
  const slug = postId ? resolveMermaidSlug(`/src/content/blog/${postId}`) : 'shared';
  return mermaidImagePath(code, parseMermaidOptions(meta), slug, 'light');
}
