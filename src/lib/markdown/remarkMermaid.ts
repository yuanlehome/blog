/**
 * remarkMermaid — build-time Mermaid code block transformer.
 *
 * This plugin only handles the **AST transformation** step:
 *  1. Replaces ```mermaid fenced code blocks with dual-theme SVG <img> nodes.
 *  2. Sets `file.data.astro.frontmatter.mermaidCover` for the first Mermaid
 *     block in a post that has no explicit `cover` frontmatter field.
 *
 * Rendering (diagram → SVG files) is performed by `scripts/render-mermaid.mjs`.
 */

import type { Plugin } from 'unified';
import type { Root, Code, HTML } from 'mdast';
import type { VFile } from 'vfile';
import { visit } from 'unist-util-visit';
import { createHash } from 'crypto';
import { existsSync } from 'fs';
import { dirname, join } from 'path';
import { withBase } from '../site/withBase';

const GENERATED_ROOT = 'generated/mermaid';
const DEFAULT_SEQUENCE_MODE = 'loose' as const;

export type MermaidSequenceMode = 'tight' | 'normal' | 'loose';

export interface MermaidRenderOptions {
  sequenceMode: MermaidSequenceMode;
  scale: number;
  width?: number;
  fontSize: number;
  wrap: boolean;
  theme?: string;
}

const DEFAULT_OPTIONS: MermaidRenderOptions = {
  sequenceMode: DEFAULT_SEQUENCE_MODE,
  scale: 1,
  width: undefined,
  fontSize: 14,
  wrap: true,
  theme: undefined,
};

const META_ENTRY_RE = /(\w+)=((?:"[^"]*"|'[^']*'|\S+))/g;

function asNumber(value: string | undefined, fallback: number): number {
  if (!value) return fallback;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function asBoolean(value: string | undefined, fallback: boolean): boolean {
  if (!value) return fallback;
  const lowered = value.toLowerCase();
  if (['true', '1', 'yes', 'on'].includes(lowered)) return true;
  if (['false', '0', 'no', 'off'].includes(lowered)) return false;
  return fallback;
}

function sanitizeSlugSegment(input: string): string {
  return input
    .toLowerCase()
    .replace(/[^a-z0-9/-]/g, '-')
    .replace(/\/+/g, '/')
    .replace(/-+/g, '-')
    .replace(/^\/+|\/+$/g, '');
}

export function parseMermaidMeta(meta?: string | null): Record<string, string> {
  if (!meta) return {};
  const out: Record<string, string> = {};
  META_ENTRY_RE.lastIndex = 0;
  let match: RegExpExecArray | null;
  while ((match = META_ENTRY_RE.exec(meta)) !== null) {
    out[match[1]] = match[2].replace(/^['"]|['"]$/g, '');
  }
  return out;
}

export function parseMermaidOptions(meta?: string | null): MermaidRenderOptions {
  const entries = parseMermaidMeta(meta);
  const rawMode = (entries.sequenceMode ?? DEFAULT_SEQUENCE_MODE).toLowerCase();
  const sequenceMode: MermaidSequenceMode =
    rawMode === 'tight' || rawMode === 'normal' || rawMode === 'loose' ? rawMode : 'loose';

  return {
    sequenceMode,
    scale: Math.min(Math.max(asNumber(entries.scale, DEFAULT_OPTIONS.scale), 0.5), 2.5),
    width: entries.width ? Math.max(400, Math.min(2400, asNumber(entries.width, 1200))) : undefined,
    fontSize: Math.max(
      11,
      Math.min(24, Math.round(asNumber(entries.fontSize, DEFAULT_OPTIONS.fontSize))),
    ),
    wrap: asBoolean(entries.wrap, DEFAULT_OPTIONS.wrap),
    theme: entries.theme,
  };
}

export function parseTitle(meta?: string | null): string | undefined {
  const entries = parseMermaidMeta(meta);
  return entries.title ?? entries.caption;
}

export function resolveMermaidSlug(filePath?: string): string {
  if (!filePath) return 'shared';
  const normalized = filePath.replace(/\\/g, '/');
  const marker = '/src/content/blog/';
  const idx = normalized.lastIndexOf(marker);
  if (idx === -1) return 'shared';
  const relativePath = normalized.slice(idx + marker.length);
  const folder = dirname(relativePath);
  return sanitizeSlugSegment(folder === '.' ? 'root' : folder);
}

export function mermaidHash(code: string, options: MermaidRenderOptions): string {
  const payload = JSON.stringify({
    code: code.trim(),
    options,
    version: 5,
  });
  return createHash('md5').update(payload).digest('hex').slice(0, 12);
}

export function mermaidImagePath(
  code: string,
  options: MermaidRenderOptions = DEFAULT_OPTIONS,
  slug = 'shared',
  theme: 'light' | 'dark' = 'light',
): string {
  const hash = mermaidHash(code, options);
  const safeSlug = sanitizeSlugSegment(slug) || 'shared';
  return `/${GENERATED_ROOT}/${safeSlug}/${hash}.${theme}.svg`;
}

export function mermaidAbsolutePath(
  code: string,
  publicDir: string,
  options: MermaidRenderOptions = DEFAULT_OPTIONS,
  slug = 'shared',
  theme: 'light' | 'dark' = 'light',
): string {
  const rel = mermaidImagePath(code, options, slug, theme).replace(/^\//, '');
  return join(publicDir, rel);
}

function buildDualImageHtml(lightSrc: string, darkSrc: string, alt: string): string {
  const escapedAlt = alt.replace(/"/g, '&quot;');
  return `<span class="mermaid-dual-image"><img src="${lightSrc}" alt="${escapedAlt}" loading="lazy" decoding="async" data-theme="light" class="mermaid-dual-image__img mermaid-dual-image__img--light" /><img src="${darkSrc}" alt="${escapedAlt}" loading="lazy" decoding="async" data-theme="dark" class="mermaid-dual-image__img mermaid-dual-image__img--dark" /></span>`;
}

interface RemarkMermaidOptions {
  base?: string;
}

const remarkMermaid: Plugin<[RemarkMermaidOptions?], Root> = (pluginOptions = {}) => {
  return (tree: Root, file: VFile): void => {
    const targets: Array<{ node: Code; index: number; parent: Root['children'][0] }> = [];

    visit(tree, 'code', (node: Code, index, parent) => {
      if (node.lang?.toLowerCase() === 'mermaid' && parent && index != null) {
        targets.push({ node, index, parent: parent as unknown as Root['children'][0] });
      }
    });

    if (targets.length === 0) return;

    const existingCover: string | undefined = (file.data as any)?.astro?.frontmatter?.cover;
    let coverSet = Boolean(existingCover);
    const slug = resolveMermaidSlug(file.path);
    const base = pluginOptions.base ?? '/';
    const debugEnabled = process.env.DEBUG_MERMAID === '1';

    for (const { node, index, parent } of targets) {
      const code = node.value.trim();
      const options = parseMermaidOptions(node.meta);
      const title = parseTitle(node.meta) ?? 'Mermaid Diagram';
      const lightPath = mermaidImagePath(code, options, slug, 'light');
      const darkPath = mermaidImagePath(code, options, slug, 'dark');

      const lightPublicPath = mermaidAbsolutePath(
        code,
        join(file.cwd ?? process.cwd(), 'public'),
        options,
        slug,
        'light',
      );
      const darkPublicPath = mermaidAbsolutePath(
        code,
        join(file.cwd ?? process.cwd(), 'public'),
        options,
        slug,
        'dark',
      );

      if (!existsSync(lightPublicPath) || !existsSync(darkPublicPath)) {
        console.warn(
          `[remarkMermaid] Missing rendered assets for ${file.path ?? 'unknown file'}: ${lightPublicPath} | ${darkPublicPath}. Keeping code block fallback.`,
        );
        continue;
      }

      const lightSrc = withBase(lightPath, base);
      const darkSrc = withBase(darkPath, base);

      if (debugEnabled) {
        console.debug(`[remarkMermaid] abs=${lightPublicPath} rel=${lightPath} src=${lightSrc}`);
        console.debug(`[remarkMermaid] abs=${darkPublicPath} rel=${darkPath} src=${darkSrc}`);
      }

      const htmlNode: HTML = {
        type: 'html',
        value: buildDualImageHtml(lightSrc, darkSrc, title),
      };
      (parent as any).children.splice(index, 1, htmlNode);

      if (!coverSet) {
        (file.data as any).astro ??= {};
        (file.data as any).astro.frontmatter ??= {};
        (file.data as any).astro.frontmatter.mermaidCover = lightPath;
        coverSet = true;
      }
    }
  };
};

export default remarkMermaid;
