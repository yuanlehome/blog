#!/usr/bin/env node
/**
 * render-mermaid.mjs — prebuild script to render Mermaid diagrams to PNG.
 *
 * Run automatically via `npm run prebuild` before `astro build`.
 * Scans all Markdown files in `src/content/blog/` for ```mermaid fenced code
 * blocks, renders each unique diagram to a PNG file under `public/mermaid-generated/`,
 * and skips diagrams whose output file already exists (incremental builds).
 *
 * This script runs in a plain Node.js process, which avoids the
 * "Vite module runner has been closed" error that occurs when async dynamic
 * `import()` calls are made from inside remark/rehype plugins during
 * Astro's content-collection compilation phase.
 */

import { readdir, readFile, mkdir, writeFile, access } from 'node:fs/promises';
import { join, resolve } from 'node:path';
import { createHash } from 'node:crypto';
import { JSDOM } from 'jsdom';
import sharp from 'sharp';

// ---------------------------------------------------------------------------
// Constants — must match mermaidImagePath() in src/lib/markdown/remarkMermaid.ts
// ---------------------------------------------------------------------------

const PROJECT_ROOT = resolve(import.meta.dirname, '..');
const CONTENT_DIR = join(PROJECT_ROOT, 'src', 'content', 'blog');
const PUBLIC_DIR = join(PROJECT_ROOT, 'public');
const GENERATED_DIR = join(PUBLIC_DIR, 'mermaid-generated');

function mermaidImagePath(code) {
  const hash = createHash('md5').update(code.trim()).digest('hex').slice(0, 12);
  return join(GENERATED_DIR, `${hash}.png`);
}

// ---------------------------------------------------------------------------
// Mermaid renderer — runs in THIS process (not inside Vite's SSR runner)
// ---------------------------------------------------------------------------

let _mermaidRender = null;

async function getMermaidRenderer() {
  if (_mermaidRender) return _mermaidRender;

  const dom = new JSDOM('<!DOCTYPE html><html><body></body></html>', {
    pretendToBeVisual: true,
  });
  const { window } = dom;

  // Polyfill getBBox / getComputedTextLength for Mermaid's layout engine.
  // We approximate text width at 6 px/char (capped to 150 px) to keep the
  // generated viewBox within sharp's 32767×32767 px limit.
  window.SVGElement.prototype.getBBox = function () {
    const tag = (this.tagName ?? '').toLowerCase();
    if (tag === 'text' || tag === 'tspan') {
      const w = Math.min((this.textContent?.length ?? 0) * 6, 150);
      return { x: 0, y: 0, width: w || 50, height: 14 };
    }
    return { x: 0, y: 0, width: 50, height: 20 };
  };
  window.SVGElement.prototype.getComputedTextLength = function () {
    return Math.min((this.textContent?.length ?? 0) * 6, 150);
  };

  const props = {
    window,
    document: window.document,
    DOMParser: window.DOMParser,
    Element: window.Element,
    SVGElement: window.SVGElement,
    HTMLElement: window.HTMLElement,
    Text: window.Text,
    Comment: window.Comment,
    localStorage: {
      getItem: () => null,
      setItem: () => {},
      removeItem: () => {},
      clear: () => {},
      length: 0,
      key: () => null,
    },
  };
  for (const [key, value] of Object.entries(props)) {
    try {
      Object.defineProperty(globalThis, key, { value, writable: true, configurable: true });
    } catch {
      // Some globals (navigator, etc.) are read-only — skip them
    }
  }

  const { default: mermaid } = await import('mermaid');
  mermaid.initialize({
    startOnLoad: false,
    theme: 'neutral',
    securityLevel: 'loose',
    fontFamily: 'trebuchet ms, verdana, arial, sans-serif',
  });

  _mermaidRender = async (code, id) => {
    const { svg } = await mermaid.render(id, code);
    return svg;
  };

  return _mermaidRender;
}

// ---------------------------------------------------------------------------
// SVG → PNG conversion
// ---------------------------------------------------------------------------

/**
 * Compute the actual bounding box of all rendered elements from the SVG by
 * scanning translate(), rect, polygon, path, and circle coordinates.
 * This is needed because jsdom's getBBox() mock returns wrong dimensions,
 * causing mermaid to produce a bad viewBox.
 */
function computeSvgBounds(svg) {
  let minX = Infinity,
    minY = Infinity,
    maxX = -Infinity,
    maxY = -Infinity;

  // Collect all translate(x, y) contexts from node groups
  const translateRe = /transform="translate\((-?\d+\.?\d*),\s*(-?\d+\.?\d*)\)"/g;
  const rectRe =
    /x="(-?\d+\.?\d*)"\s+y="(-?\d+\.?\d*)"\s+width="(\d+\.?\d*)"\s+height="(\d+\.?\d*)"/g;
  const circleRe = /cx="(-?\d+\.?\d*)"\s+cy="(-?\d+\.?\d*)"\s+r="(\d+\.?\d*)"/g;

  // Simple coordinate extraction: find all numeric coordinate pairs in path d= attributes
  const dCoordRe = /[ML]\s*(-?\d+\.?\d*),?(-?\d+\.?\d*)/g;

  let m;
  while ((m = translateRe.exec(svg)) !== null) {
    const tx = parseFloat(m[1]),
      ty = parseFloat(m[2]);
    // Assume node size ~80x40 centred at translate point
    minX = Math.min(minX, tx - 80);
    maxX = Math.max(maxX, tx + 80);
    minY = Math.min(minY, ty - 20);
    maxY = Math.max(maxY, ty + 20);
  }
  while ((m = rectRe.exec(svg)) !== null) {
    const x = parseFloat(m[1]),
      y = parseFloat(m[2]),
      w = parseFloat(m[3]),
      h = parseFloat(m[4]);
    if (w > 0 && h > 0 && w < 5000 && h < 5000) {
      minX = Math.min(minX, x);
      maxX = Math.max(maxX, x + w);
      minY = Math.min(minY, y);
      maxY = Math.max(maxY, y + h);
    }
  }
  while ((m = circleRe.exec(svg)) !== null) {
    const cx = parseFloat(m[1]),
      cy = parseFloat(m[2]),
      r = parseFloat(m[3]);
    minX = Math.min(minX, cx - r);
    maxX = Math.max(maxX, cx + r);
    minY = Math.min(minY, cy - r);
    maxY = Math.max(maxY, cy + r);
  }
  while ((m = dCoordRe.exec(svg)) !== null) {
    const x = parseFloat(m[1]),
      y = parseFloat(m[2]);
    if (Math.abs(x) < 5000 && Math.abs(y) < 5000) {
      minX = Math.min(minX, x);
      maxX = Math.max(maxX, x);
      minY = Math.min(minY, y);
      maxY = Math.max(maxY, y);
    }
  }

  if (!isFinite(minX)) return null;
  const pad = 12;
  return {
    x: minX - pad,
    y: minY - pad,
    w: maxX - minX + 2 * pad,
    h: maxY - minY + 2 * pad,
  };
}

async function svgToPng(svg) {
  // Fix the viewBox when mermaid renders it incorrectly (jsdom getBBox mock limitation)
  const bounds = computeSvgBounds(svg);
  let patchedSvg = svg;
  if (bounds && bounds.w > 10 && bounds.h > 10) {
    const targetW = 800;
    const targetH = Math.round((bounds.h / bounds.w) * targetW);
    patchedSvg = patchedSvg
      .replace(/viewBox="[^"]*"/, `viewBox="${bounds.x} ${bounds.y} ${bounds.w} ${bounds.h}"`)
      .replace(/width="[^"]*"/, `width="${targetW}"`)
      .replace(/height="[^"]*"/, `height="${targetH}"`);
  }
  return sharp(Buffer.from(patchedSvg), { density: 150 }).png().toBuffer();
}

// ---------------------------------------------------------------------------
// Markdown scanner — extracts unique mermaid blocks with their alt titles
// ---------------------------------------------------------------------------

const MERMAID_FENCE_RE = /^```mermaid([^\n]*)\n([\s\S]*?)^```/gm;
const TITLE_RE = /(?:title|caption)=((?:"[^"]+"|'[^']+'|\S+))/i;

function extractMermaidBlocks(content) {
  const blocks = [];
  let m;
  MERMAID_FENCE_RE.lastIndex = 0;
  while ((m = MERMAID_FENCE_RE.exec(content)) !== null) {
    const meta = m[1].trim();
    const code = m[2].trimEnd();
    if (code) {
      const titleMatch = meta.match(TITLE_RE);
      const raw = titleMatch?.[1]?.trim();
      const title = raw ? raw.replace(/^['"]|['"]$/g, '') : undefined;
      blocks.push({ code, title });
    }
  }
  return blocks;
}

async function collectMarkdownFiles(dir) {
  const entries = await readdir(dir, { withFileTypes: true });
  const files = [];
  for (const entry of entries) {
    const fullPath = join(dir, entry.name);
    if (entry.isDirectory()) {
      files.push(...(await collectMarkdownFiles(fullPath)));
    } else if (entry.name.endsWith('.md') || entry.name.endsWith('.mdx')) {
      files.push(fullPath);
    }
  }
  return files;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

async function main() {
  const files = await collectMarkdownFiles(CONTENT_DIR);
  console.log(`[render-mermaid] Scanning ${files.length} Markdown files…`);

  // Collect all unique mermaid diagrams across all files
  const diagrams = new Map(); // code.trim() → { code, title, outputPath }
  for (const filePath of files) {
    const content = await readFile(filePath, 'utf-8');
    for (const { code, title } of extractMermaidBlocks(content)) {
      const key = code.trim();
      if (!diagrams.has(key)) {
        diagrams.set(key, { code: key, title, outputPath: mermaidImagePath(key) });
      }
    }
  }

  if (diagrams.size === 0) {
    console.log('[render-mermaid] No Mermaid diagrams found.');
    return;
  }

  console.log(`[render-mermaid] Found ${diagrams.size} unique diagram(s).`);

  // Create output directory
  await mkdir(GENERATED_DIR, { recursive: true });

  // Initialise renderer once
  const render = await getMermaidRenderer();
  let rendered = 0;
  let skipped = 0;

  for (const { code, outputPath } of diagrams.values()) {
    // Skip if PNG already exists (incremental build)
    try {
      await access(outputPath);
      skipped++;
      continue;
    } catch {
      // File does not exist — render it
    }

    const diagramId = `mermaid-${createHash('md5').update(code).digest('hex').slice(0, 8)}`;
    try {
      const svg = await render(code, diagramId);
      const png = await svgToPng(svg);
      await writeFile(outputPath, png);
      rendered++;
      console.log(`[render-mermaid] ✓ ${outputPath.replace(PROJECT_ROOT + '/', '')}`);
    } catch (err) {
      console.error(`[render-mermaid] ✗ Failed to render diagram: ${err.message}`);
      console.error(`  Code: ${code.slice(0, 80)}…`);
    }
  }

  console.log(
    `[render-mermaid] Done. ${rendered} rendered, ${skipped} skipped (already up-to-date).`,
  );
}

main().catch((err) => {
  console.error('[render-mermaid] Fatal error:', err);
  process.exit(1);
});
