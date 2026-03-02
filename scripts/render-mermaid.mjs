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

async function svgToPng(svg) {
  const vbMatch = svg.match(/viewBox="(-?\d+\.?\d*)\s+(-?\d+\.?\d*)\s+(\d+\.?\d*)\s+(\d+\.?\d*)"/);
  let scaledSvg = svg;
  if (vbMatch) {
    const vbW = parseFloat(vbMatch[3]);
    const vbH = parseFloat(vbMatch[4]);
    const scale = Math.max(800 / (vbW || 1), 300 / (vbH || 1));
    const targetW = Math.round(vbW * scale);
    scaledSvg = svg.replace(/width="100%"/, `width="${targetW}"`);
  }
  return sharp(Buffer.from(scaledSvg), { density: 150 }).png().toBuffer();
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
