/**
 * Utilities for detecting Mermaid-generated cover images from raw post bodies.
 */

import { mermaidImagePath, parseMermaidOptions } from '../markdown/remarkMermaid';

/**
 * Extracts the expected Mermaid-generated light cover image URL from a raw post body.
 */
export function getMermaidCoverFromBody(body: string, slug?: string): string | undefined {
  const match = body.match(/^```mermaid([^\n]*)\n([\s\S]*?)^```/m);
  if (!match?.[2]) return undefined;
  const meta = match[1]?.trim();
  const code = match[2].trimEnd();
  if (!code) return undefined;
  return mermaidImagePath(code, parseMermaidOptions(meta), slug ?? 'shared', 'light');
}
