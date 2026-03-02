/**
 * Utilities for detecting Mermaid-generated cover images from raw post bodies.
 *
 * This module is intentionally dependency-free (no mermaid import) so it can
 * be called from Astro components (PostList, slug page) without triggering the
 * heavy jsdom / mermaid initialization.
 */

import { mermaidImagePath } from '../markdown/remarkMermaid';

/**
 * Extracts the expected mermaid cover image URL from a raw markdown body.
 *
 * Scans for the first ` ```mermaid ` fenced code block and returns the
 * deterministic PNG path that `remarkMermaid` would generate for it.
 * Returns `undefined` when no mermaid block is found.
 *
 * @param body - Raw markdown string (post.body from Astro content collection)
 */
export function getMermaidCoverFromBody(body: string): string | undefined {
  // Match ``` followed by "mermaid" and an optional meta string, then content
  const match = body.match(/^```mermaid[^\n]*\n([\s\S]*?)^```/m);
  if (!match?.[1]) return undefined;
  const code = match[1].trimEnd();
  if (!code) return undefined;
  return mermaidImagePath(code);
}
