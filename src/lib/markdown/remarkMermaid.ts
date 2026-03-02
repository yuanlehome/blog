/**
 * remarkMermaid — build-time Mermaid code block transformer.
 *
 * This plugin only handles the **AST transformation** step:
 *  1. Replaces ```mermaid fenced code blocks with PNG <img> nodes.
 *  2. Sets `file.data.astro.frontmatter.mermaidCover` for the first Mermaid
 *     block in a post that has no explicit `cover` frontmatter field.
 *
 * **Rendering** (diagram → PNG file) is deliberately separated and performed
 * by `scripts/render-mermaid.mjs` (run via the `prebuild` npm script) so that
 * it executes in a plain Node.js process — not inside Vite's SSR module runner
 * which closes before async dynamic `import()` calls can resolve.
 *
 * Supported meta options on the opening fence:
 *   title="My Diagram"   → used as the image alt text
 *   caption="…"          → same as title
 */

import type { Plugin } from 'unified';
import type { Root, Image, Code } from 'mdast';
import type { VFile } from 'vfile';
import { visit } from 'unist-util-visit';
import { createHash } from 'crypto';
import { join } from 'path';

// --------------------------------------------------------------------------
// Shared path helpers (also imported by mermaidCover utility and the
// render-mermaid.mjs prebuild script)
// --------------------------------------------------------------------------

const GENERATED_DIR = 'mermaid-generated';

/**
 * Deterministic public path for a Mermaid diagram.
 * The same code always maps to the same file name, enabling incremental builds.
 */
export function mermaidImagePath(code: string): string {
  const hash = createHash('md5').update(code.trim()).digest('hex').slice(0, 12);
  return `/${GENERATED_DIR}/${hash}.png`;
}

/** Absolute filesystem path under `publicDir` for a given diagram code. */
export function mermaidAbsolutePath(code: string, publicDir: string): string {
  const rel = mermaidImagePath(code).replace(/^\//, '');
  return join(publicDir, rel);
}

// --------------------------------------------------------------------------
// Meta string parser — mirrors the format used by remarkCodeMeta
// --------------------------------------------------------------------------

const titlePattern = /(?:title|caption)=((?:"[^"]+"|'[^']+'|\S+))/i;

function parseTitle(meta?: string | null): string | undefined {
  if (!meta) return undefined;
  const m = meta.match(titlePattern);
  const raw = m?.[1]?.trim();
  return raw?.replace(/^['"]|['"]$/g, '');
}

// --------------------------------------------------------------------------
// remark plugin
// --------------------------------------------------------------------------

const remarkMermaid: Plugin<[], Root> = () => {
  return (tree: Root, file: VFile): void => {
    // Collect mermaid nodes
    const targets: Array<{ node: Code; index: number; parent: Root['children'][0] }> = [];

    visit(tree, 'code', (node: Code, index, parent) => {
      if (node.lang?.toLowerCase() === 'mermaid' && parent && index != null) {
        targets.push({ node, index, parent: parent as unknown as Root['children'][0] });
      }
    });

    if (targets.length === 0) return;

    // Detect if frontmatter already has an explicit cover
    const existingCover: string | undefined = (file.data as any)?.astro?.frontmatter?.cover;
    let coverSet = Boolean(existingCover);

    for (const { node, index, parent } of targets) {
      const code = node.value.trim();
      const title = parseTitle(node.meta) ?? 'Mermaid Diagram';
      const imgPath = mermaidImagePath(code);

      // Replace the code block with an image node
      const imageNode: Image = {
        type: 'image',
        url: imgPath,
        alt: title,
        title,
      };
      (parent as any).children.splice(index, 1, imageNode);

      // Promote as cover candidate if no explicit cover exists yet
      if (!coverSet) {
        (file.data as any).astro ??= {};
        (file.data as any).astro.frontmatter ??= {};
        (file.data as any).astro.frontmatter.mermaidCover = imgPath;
        coverSet = true;
      }
    }
  };
};

export default remarkMermaid;
