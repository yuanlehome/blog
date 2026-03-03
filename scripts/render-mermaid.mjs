#!/usr/bin/env node
/**
 * Render Mermaid diagrams to dual-theme SVG files used by remarkMermaid.
 */

import { readdir, readFile, mkdir, access, writeFile, rename } from 'node:fs/promises';
import { dirname, join, relative, resolve } from 'node:path';
import { createHash } from 'node:crypto';
import { pathToFileURL } from 'node:url';
import { JSDOM } from 'jsdom';

const PROJECT_ROOT = resolve(import.meta.dirname, '..');
const CONTENT_DIR = join(PROJECT_ROOT, 'src', 'content', 'blog');
const PUBLIC_DIR = join(PROJECT_ROOT, 'public');

const GENERATED_ROOT = 'generated/mermaid';
const META_ENTRY_RE = /(\w+)=((?:"[^"]*"|'[^']*'|\S+))/g;
// Flowchart rendering under jsdom can occasionally emit a tiny viewBox while edge points are valid.
// These bounds and padding values are used only for that malformed tiny-viewBox recovery path.
const TINY_VIEWBOX_WIDTH_THRESHOLD = 240;
const TINY_VIEWBOX_HEIGHT_THRESHOLD = 180;
const TINY_VIEWBOX_PAD_X = 80;
const TINY_VIEWBOX_PAD_Y = 80;
const TINY_VIEWBOX_EXPANSION_FACTOR = 2;
const EDGE_VIEWBOX_PADDING = 24;
const EDGE_VIEWBOX_MAX_EXPANSION_FACTOR = 1.6;
const DEFAULT_MAX_OUTPUT_HEIGHT = 960;
// Character-width heuristics for jsdom SVG text measurement (rough px per glyph).
const CHAR_WIDTH_SPACE = 4;
const CHAR_WIDTH_CJK = 14;
const CHAR_WIDTH_UPPERCASE = 8.5;
const CHAR_WIDTH_ALNUM = 7;
const CHAR_WIDTH_SYMBOL = 8;

function getAttrNumber(node, name, fallback = 0) {
  const value = Number(node.getAttribute(name));
  return Number.isFinite(value) ? value : fallback;
}

function parseTranslate(transform = '') {
  const match = transform.match(/translate\(([^)]+)\)/i);
  if (!match) return { x: 0, y: 0 };
  const [x = '0', y = '0'] = match[1].split(/[\s,]+/).filter(Boolean);
  return {
    x: Number.isFinite(Number(x)) ? Number(x) : 0,
    y: Number.isFinite(Number(y)) ? Number(y) : 0,
  };
}

function mergeBounds(bounds) {
  if (!bounds.length) return null;
  let minX = Number.POSITIVE_INFINITY;
  let minY = Number.POSITIVE_INFINITY;
  let maxX = Number.NEGATIVE_INFINITY;
  let maxY = Number.NEGATIVE_INFINITY;
  for (const b of bounds) {
    if (!b || !Number.isFinite(b.width) || !Number.isFinite(b.height)) continue;
    minX = Math.min(minX, b.x);
    minY = Math.min(minY, b.y);
    maxX = Math.max(maxX, b.x + b.width);
    maxY = Math.max(maxY, b.y + b.height);
  }
  if (![minX, minY, maxX, maxY].every(Number.isFinite)) return null;
  return { x: minX, y: minY, width: maxX - minX, height: maxY - minY };
}

function normalizeLabelText(text = '') {
  return text
    .replace(/<br\s*\/?>/gi, '\n')
    .replace(/\\n/g, '\n')
    .replace(/\s*\n\s*/g, '\n')
    .trim();
}

function estimateTextMetrics(text = '') {
  const normalized = normalizeLabelText(text);
  const lines = normalized ? normalized.split('\n') : [''];
  const lineHeight = 18;
  const maxWidth = Math.max(
    ...lines.map((line) =>
      Array.from(line).reduce((sum, ch) => {
        if (/\s/.test(ch)) return sum + CHAR_WIDTH_SPACE;
        if (/[\u3400-\u9FFF\uF900-\uFAFF]/u.test(ch)) return sum + CHAR_WIDTH_CJK;
        if (/[A-Z]/.test(ch)) return sum + CHAR_WIDTH_UPPERCASE;
        if (/[0-9a-z]/.test(ch)) return sum + CHAR_WIDTH_ALNUM;
        return sum + CHAR_WIDTH_SYMBOL;
      }, 0),
    ),
  );
  return {
    width: Math.max(60, Math.ceil(maxWidth) + 8),
    height: Math.max(24, lines.length * lineHeight),
    lines: lines.length,
  };
}

function parseMermaidMeta(meta = '') {
  const out = {};
  META_ENTRY_RE.lastIndex = 0;
  let match;
  while ((match = META_ENTRY_RE.exec(meta)) !== null) {
    out[match[1]] = match[2].replace(/^['"]|['"]$/g, '');
  }
  return out;
}

function parseMermaidOptions(meta = '') {
  const entries = parseMermaidMeta(meta);
  const sequenceMode = ['tight', 'normal', 'loose'].includes(
    (entries.sequenceMode || '').toLowerCase(),
  )
    ? entries.sequenceMode.toLowerCase()
    : 'loose';
  const scale = Number(entries.scale);
  const width = Number(entries.width);
  const fontSize = Number(entries.fontSize);
  return {
    sequenceMode,
    scale: Number.isFinite(scale) ? Math.min(Math.max(scale, 0.5), 2.5) : 1,
    width: Number.isFinite(width) ? Math.max(400, Math.min(width, 2400)) : undefined,
    fontSize: Number.isFinite(fontSize) ? Math.max(11, Math.min(Math.round(fontSize), 24)) : 14,
    wrap: entries.wrap ? ['true', '1', 'yes', 'on'].includes(entries.wrap.toLowerCase()) : true,
    theme: entries.theme,
  };
}

function parseTitle(meta = '') {
  const entries = parseMermaidMeta(meta);
  return entries.title || entries.caption;
}

function resolveMermaidSlug(filePath) {
  const normalized = filePath.replace(/\\/g, '/');
  const marker = '/src/content/blog/';
  const idx = normalized.lastIndexOf(marker);
  if (idx === -1) return 'shared';
  const relPath = normalized.slice(idx + marker.length);
  const folder = dirname(relPath);
  return (folder === '.' ? 'root' : folder)
    .toLowerCase()
    .replace(/[^a-z0-9/-]/g, '-')
    .replace(/-+/g, '-');
}

function mermaidHash(code, options) {
  return createHash('md5')
    .update(JSON.stringify({ code: code.trim(), options, version: 6 }))
    .digest('hex')
    .slice(0, 12);
}

function mermaidAbsolutePath(code, publicDir, options, slug, theme) {
  const hash = mermaidHash(code, options);
  const safeSlug = (slug || 'shared').replace(/^\/+|\/+$/g, '');
  return join(publicDir, GENERATED_ROOT, safeSlug, `${hash}.${theme}.svg`);
}

const THEME_PRESETS = {
  light: {
    theme: 'default',
    background: '#ffffff',
    themeVariables: {
      background: '#ffffff',
      primaryColor: '#e2e8f0',
      primaryTextColor: '#0f172a',
      lineColor: '#334155',
      noteBkgColor: '#f8fafc',
      noteTextColor: '#0f172a',
      actorBkg: '#f1f5f9',
      actorTextColor: '#0f172a',
      signalColor: '#1f2937',
      signalTextColor: '#111827',
    },
  },
  dark: {
    theme: 'dark',
    background: '#0b0f19',
    themeVariables: {
      background: '#0b0f19',
      primaryColor: '#1f2937',
      primaryTextColor: '#e5e7eb',
      lineColor: '#9ca3af',
      noteBkgColor: '#1f2937',
      noteTextColor: '#e5e7eb',
      actorBkg: '#111827',
      actorTextColor: '#e5e7eb',
      signalColor: '#cbd5e1',
      signalTextColor: '#e5e7eb',
    },
  },
};

const SEQUENCE_PRESETS = {
  tight: {
    diagramMarginX: 30,
    diagramMarginY: 20,
    actorMargin: 40,
    messageMargin: 18,
    noteMargin: 12,
  },
  normal: {
    diagramMarginX: 48,
    diagramMarginY: 30,
    actorMargin: 60,
    messageMargin: 28,
    noteMargin: 20,
  },
  loose: {
    diagramMarginX: 72,
    diagramMarginY: 44,
    actorMargin: 86,
    messageMargin: 38,
    noteMargin: 28,
  },
};

let _mermaid;
async function getMermaid() {
  if (_mermaid) return _mermaid;

  const dom = new JSDOM('<!DOCTYPE html><html><body></body></html>', { pretendToBeVisual: true });
  const { window } = dom;

  window.SVGElement.prototype.getBBox = function () {
    const rawTag = this.tagName ?? '';
    const tag = (this.tagName ?? '').toLowerCase();
    if (tag === 'text' || tag === 'tspan' || rawTag === 'foreignObject') {
      const { width, height } = estimateTextMetrics(this.textContent ?? '');
      // Keep text-local bbox origin stable in jsdom; using absolute x/y here can
      // push labels outside their container during Mermaid layout and make box text disappear.
      return { x: 0, y: 0, width, height };
    }
    if (tag === 'rect') {
      const width = getAttrNumber(this, 'width', Number.NaN);
      const height = getAttrNumber(this, 'height', Number.NaN);
      if (Number.isFinite(width) && width > 0 && Number.isFinite(height) && height > 0) {
        const x = getAttrNumber(this, 'x', 0);
        const y = getAttrNumber(this, 'y', 0);
        return { x, y, width, height };
      }
    }
    if (tag === 'g' || tag === 'svg') {
      const childBounds = Array.from(this.children)
        .map((child) => {
          if (!child || typeof child.getBBox !== 'function') return null;
          try {
            const box = child.getBBox();
            const { x, y } = parseTranslate(child.getAttribute('transform') ?? '');
            return { ...box, x: box.x + x, y: box.y + y };
          } catch {
            return null;
          }
        })
        .filter(Boolean);
      const merged = mergeBounds(childBounds);
      if (merged) return merged;
    }
    return { x: 0, y: 0, width: 70, height: 24 };
  };
  window.SVGElement.prototype.getComputedTextLength = function () {
    return estimateTextMetrics(this.textContent ?? '').width;
  };
  window.HTMLElement.prototype.getBoundingClientRect = function () {
    const text = normalizeLabelText(this.textContent ?? '');
    const { width, height } = estimateHtmlLabelMetrics(text);
    return {
      x: 0,
      y: 0,
      top: 0,
      left: 0,
      right: width,
      bottom: height,
      width,
      height,
      toJSON() {
        return this;
      },
    };
  };

  for (const [key, value] of Object.entries({
    window,
    document: window.document,
    DOMParser: window.DOMParser,
    Element: window.Element,
    SVGElement: window.SVGElement,
    HTMLElement: window.HTMLElement,
    Text: window.Text,
    Comment: window.Comment,
  })) {
    try {
      Object.defineProperty(globalThis, key, { value, writable: true, configurable: true });
    } catch {
      // ignore readonly globals
    }
  }

  const { default: mermaid } = await import('mermaid');
  _mermaid = mermaid;
  return mermaid;
}

function isSequenceDiagram(code) {
  return /^\s*sequenceDiagram\b/m.test(code);
}

function makeRenderConfig(options, themeMode) {
  const themePreset = THEME_PRESETS[themeMode];
  const sequence = SEQUENCE_PRESETS[options.sequenceMode] ?? SEQUENCE_PRESETS.loose;
  return {
    startOnLoad: false,
    securityLevel: 'loose',
    theme: options.theme || themePreset.theme,
    fontSize: options.fontSize,
    wrap: options.wrap,
    flowchart: {
      padding: options.sequenceMode === 'loose' ? 24 : 16,
      nodeSpacing: options.sequenceMode === 'loose' ? 72 : 50,
      rankSpacing: options.sequenceMode === 'loose' ? 72 : 50,
      useMaxWidth: false,
      htmlLabels: false,
      wrap: options.wrap,
    },
    sequence,
    themeVariables: themePreset.themeVariables,
    fontFamily: 'Inter, -apple-system, Segoe UI, Roboto, sans-serif',
  };
}

function ensureViewBox(document) {
  const svg = document.querySelector('svg');
  if (!svg) return null;
  const current = svg.getAttribute('viewBox');
  if (current) return svg;
  const width = Number(svg.getAttribute('width') ?? 1200);
  const height = Number(svg.getAttribute('height') ?? 800);
  svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
  return svg;
}

function injectBackground(document, fill) {
  const svg = ensureViewBox(document);
  if (!svg) return;
  const hasOpaqueRect = Array.from(svg.querySelectorAll(':scope > rect')).some((rect) => {
    const f = (rect.getAttribute('fill') || '').toLowerCase();
    return f && f !== 'none' && f !== 'transparent';
  });
  if (hasOpaqueRect) return;
  const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
  rect.setAttribute('x', '0');
  rect.setAttribute('y', '0');
  rect.setAttribute('width', '100%');
  rect.setAttribute('height', '100%');
  rect.setAttribute('fill', fill);
  svg.insertBefore(rect, svg.firstChild);
}

function addViewBoxPadding(document, padding) {
  const svg = ensureViewBox(document);
  if (!svg) return;
  const viewBox = (svg.getAttribute('viewBox') ?? '').split(/\s+/).map(Number);
  if (viewBox.length !== 4 || viewBox.some((v) => !Number.isFinite(v))) return;
  const [x, y, w, h] = viewBox;
  svg.setAttribute(
    'viewBox',
    `${x - padding} ${y - padding} ${w + padding * 2} ${h + padding * 2}`,
  );
}

function normalizeTinyViewBox(document) {
  const svg = ensureViewBox(document);
  if (!svg) return;
  const viewBox = (svg.getAttribute('viewBox') ?? '').split(/\s+/).map(Number);
  if (viewBox.length !== 4 || viewBox.some((v) => !Number.isFinite(v))) return;
  const [, , width, height] = viewBox;
  if (width > TINY_VIEWBOX_WIDTH_THRESHOLD || height > TINY_VIEWBOX_HEIGHT_THRESHOLD) return;

  let minX = Number.POSITIVE_INFINITY;
  let minY = Number.POSITIVE_INFINITY;
  let maxX = Number.NEGATIVE_INFINITY;
  let maxY = Number.NEGATIVE_INFINITY;

  for (const path of svg.querySelectorAll('path[data-points]')) {
    const raw = path.getAttribute('data-points');
    if (!raw) continue;
    try {
      const points = JSON.parse(Buffer.from(raw, 'base64').toString('utf-8'));
      for (const point of points) {
        const x = Number(point?.x);
        const y = Number(point?.y);
        if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
        minX = Math.min(minX, x);
        minY = Math.min(minY, y);
        maxX = Math.max(maxX, x);
        maxY = Math.max(maxY, y);
      }
    } catch {
      // ignore invalid points payload
    }
  }

  if (![minX, minY, maxX, maxY].every(Number.isFinite)) return;
  if (
    maxX - minX <= width * TINY_VIEWBOX_EXPANSION_FACTOR &&
    maxY - minY <= height * TINY_VIEWBOX_EXPANSION_FACTOR
  ) {
    return;
  }

  svg.setAttribute(
    'viewBox',
    `${Math.floor(minX - TINY_VIEWBOX_PAD_X)} ${Math.floor(minY - TINY_VIEWBOX_PAD_Y)} ${Math.ceil(maxX - minX + TINY_VIEWBOX_PAD_X * 2)} ${Math.ceil(maxY - minY + TINY_VIEWBOX_PAD_Y * 2)}`,
  );
}

function decodeEdgePoints(raw) {
  try {
    const points = JSON.parse(Buffer.from(raw, 'base64').toString('utf-8'));
    if (!Array.isArray(points)) return [];
    return points
      .map((point) => ({ x: Number(point?.x), y: Number(point?.y) }))
      .filter((point) => Number.isFinite(point.x) && Number.isFinite(point.y));
  } catch {
    return [];
  }
}

function expandViewBoxToFitEdges(document, padding = EDGE_VIEWBOX_PADDING) {
  const svg = ensureViewBox(document);
  if (!svg) return;
  const viewBox = (svg.getAttribute('viewBox') ?? '').split(/\s+/).map(Number);
  if (viewBox.length !== 4 || viewBox.some((v) => !Number.isFinite(v))) return;
  const [x, y, w, h] = viewBox;
  let minX = x;
  let minY = y;
  let maxX = x + w;
  let maxY = y + h;
  let changed = false;

  for (const path of svg.querySelectorAll('path[data-points]')) {
    const raw = path.getAttribute('data-points');
    if (!raw) continue;
    const points = decodeEdgePoints(raw);
    for (const point of points) {
      minX = Math.min(minX, point.x - padding);
      minY = Math.min(minY, point.y - padding);
      maxX = Math.max(maxX, point.x + padding);
      maxY = Math.max(maxY, point.y + padding);
      changed = true;
    }
  }

  if (!changed) return;
  const expandedWidth = Math.ceil(maxX - minX);
  const expandedHeight = Math.ceil(maxY - minY);
  if (
    expandedWidth > w * EDGE_VIEWBOX_MAX_EXPANSION_FACTOR ||
    expandedHeight > h * EDGE_VIEWBOX_MAX_EXPANSION_FACTOR
  ) {
    return;
  }
  svg.setAttribute('viewBox', `${Math.floor(minX)} ${Math.floor(minY)} ${expandedWidth} ${expandedHeight}`);
}

function applyScaleAndWidth(document, options) {
  const svg = ensureViewBox(document);
  if (!svg) return;
  const viewBox = (svg.getAttribute('viewBox') ?? '').split(/\s+/).map(Number);
  if (viewBox.length !== 4 || viewBox.some((v) => !Number.isFinite(v))) return;
  const [, , vbWidth, vbHeight] = viewBox;
  const baseWidth = options.width ?? Math.round(vbWidth * options.scale);
  let width = Math.max(320, Math.round(baseWidth));
  let height = Math.max(220, Math.round((vbHeight / vbWidth) * width));
  if (!options.width && height > DEFAULT_MAX_OUTPUT_HEIGHT) {
    const ratio = DEFAULT_MAX_OUTPUT_HEIGHT / height;
    width = Math.max(320, Math.round(width * ratio));
    height = Math.max(220, Math.round(height * ratio));
  }
  svg.setAttribute('width', String(width));
  svg.setAttribute('height', String(height));
}

function normalizeMermaidCode(code = '') {
  if (!/^\s*flowchart\b|^\s*graph\b/m.test(code)) return code;
  return code.replace(/\\n/g, '<br/>');
}

function wrapLongSequenceText(document, options) {
  if (!options.wrap) return;
  const texts = Array.from(
    document.querySelectorAll('text.messageText, text.noteText, g.actor text'),
  );
  for (const node of texts) {
    const raw = (node.textContent ?? '').trim();
    if (raw.length < 28) continue;
    const segments = raw
      .match(/.{1,24}(?:\s+|$)/g)
      ?.map((s) => s.trim())
      .filter(Boolean) ?? [raw];
    if (segments.length < 2 || segments.length > 4) continue;
    node.textContent = '';
    segments.forEach((segment, idx) => {
      const tspan = document.createElementNS('http://www.w3.org/2000/svg', 'tspan');
      tspan.setAttribute('x', node.getAttribute('x') ?? '0');
      tspan.setAttribute('dy', idx === 0 ? '0' : '1.15em');
      tspan.textContent = segment;
      node.appendChild(tspan);
    });
    if (!node.getAttribute('font-size')) {
      node.setAttribute('font-size', String(Math.max(12, options.fontSize - 1)));
    }
  }
}


function extractForeignObjectLabelText(foreignObject) {
  const markup = foreignObject.innerHTML ?? '';
  const normalized = markup
    .replace(/<br\s*\/?>/gi, '\n')
    .replace(/<\/p>/gi, '\n')
    .replace(/<\/div>/gi, '\n')
    .replace(/<[^>]+>/g, '')
    .replace(/&nbsp;/gi, ' ');
  const fallback = foreignObject.textContent ?? '';
  return normalizeLabelText(normalized || fallback);
}

function estimateHtmlLabelMetrics(text = '') {
  const { width, height } = estimateTextMetrics(text);
  return {
    width: Math.max(110, width + 34),
    height: Math.max(44, height + 22),
  };
}

function repairForeignObjectLabels(document) {
  const foreignObjects = Array.from(document.querySelectorAll('foreignObject'));
  for (const fo of foreignObjects) {
    const width = Number(fo.getAttribute('width') ?? 0);
    const height = Number(fo.getAttribute('height') ?? 0);
    if (width > 1 && height > 1) continue;

    const labelText = extractForeignObjectLabelText(fo);
    if (!labelText) continue;
    const metrics = estimateHtmlLabelMetrics(labelText);

    fo.setAttribute('width', String(metrics.width));
    fo.setAttribute('height', String(metrics.height));
    if (!fo.hasAttribute('x')) fo.setAttribute('x', String(-metrics.width / 2));
    if (!fo.hasAttribute('y')) fo.setAttribute('y', String(-metrics.height / 2));

    const nodeGroup = fo.closest('g.node');
    if (!nodeGroup) continue;
    const rect = nodeGroup.querySelector('rect.label-container');
    if (!rect) continue;
    rect.setAttribute('x', String(-metrics.width / 2));
    rect.setAttribute('y', String(-metrics.height / 2));
    rect.setAttribute('width', String(metrics.width));
    rect.setAttribute('height', String(metrics.height));
  }
}

function postprocessSvg(rawSvg, options, themeMode) {
  const dom = new JSDOM(rawSvg, { contentType: 'image/svg+xml' });
  const { document } = dom.window;
  normalizeTinyViewBox(document);
  repairForeignObjectLabels(document);
  expandViewBoxToFitEdges(document);
  injectBackground(document, THEME_PRESETS[themeMode].background);
  addViewBoxPadding(document, 24);
  if (isSequenceDiagram(rawSvg)) {
    wrapLongSequenceText(document, options);
  }
  applyScaleAndWidth(document, options);
  return document.documentElement.outerHTML;
}

const MERMAID_FENCE_RE = /^```mermaid([^\n]*)\n([\s\S]*?)^```/gm;

function extractMermaidBlocks(content) {
  const blocks = [];
  let m;
  MERMAID_FENCE_RE.lastIndex = 0;
  while ((m = MERMAID_FENCE_RE.exec(content)) !== null) {
    const meta = m[1].trim();
    const code = m[2].trimEnd();
    if (!code) continue;
    blocks.push({ code, meta, title: parseTitle(meta) });
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

async function writeAtomic(targetPath, content) {
  const tempPath = `${targetPath}.${process.pid}.${Date.now()}.tmp`;
  await mkdir(dirname(targetPath), { recursive: true });
  await writeFile(tempPath, content, 'utf-8');
  await rename(tempPath, targetPath);
}

async function exists(path) {
  try {
    await access(path);
    return true;
  } catch {
    return false;
  }
}

async function renderForTheme(mermaid, code, id, options, themeMode) {
  mermaid.initialize(makeRenderConfig(options, themeMode));
  const normalizedCode = normalizeMermaidCode(code);
  const { svg } = await mermaid.render(`${id}-${themeMode}`, normalizedCode);
  return postprocessSvg(svg, options, themeMode);
}

async function main() {
  const files = await collectMarkdownFiles(CONTENT_DIR);
  console.log(`[render-mermaid] Scanning ${files.length} Markdown files…`);

  const diagrams = new Map();
  for (const filePath of files) {
    const content = await readFile(filePath, 'utf-8');
    const slug = resolveMermaidSlug(filePath);
    for (const { code, meta } of extractMermaidBlocks(content)) {
      const options = parseMermaidOptions(meta);
      const lightPath = mermaidAbsolutePath(code, PUBLIC_DIR, options, slug, 'light');
      const darkPath = mermaidAbsolutePath(code, PUBLIC_DIR, options, slug, 'dark');
      const key = `${slug}::${createHash('md5')
        .update(code + JSON.stringify(options))
        .digest('hex')}`;
      diagrams.set(key, { code, slug, options, lightPath, darkPath });
    }
  }

  if (diagrams.size === 0) {
    console.log('[render-mermaid] No Mermaid diagrams found.');
    return;
  }

  const mermaid = await getMermaid();
  let rendered = 0;
  let skipped = 0;

  for (const { code, slug, options, lightPath, darkPath } of diagrams.values()) {
    const hasLight = await exists(lightPath);
    const hasDark = await exists(darkPath);
    if (hasLight && hasDark) {
      skipped++;
      continue;
    }

    const diagramId = `mermaid-${createHash('md5')
      .update(slug + code)
      .digest('hex')
      .slice(0, 8)}`;
    try {
      if (!hasLight) {
        const lightSvg = await renderForTheme(mermaid, code, diagramId, options, 'light');
        await writeAtomic(lightPath, lightSvg);
      }
      if (!hasDark) {
        const darkSvg = await renderForTheme(mermaid, code, diagramId, options, 'dark');
        await writeAtomic(darkPath, darkSvg);
      }
      rendered++;
      console.log(
        `[render-mermaid] ✓ ${relative(PROJECT_ROOT, lightPath)} / ${relative(PROJECT_ROOT, darkPath)}`,
      );
    } catch (err) {
      console.error(`[render-mermaid] ✗ Failed: ${err.message}`);
      console.error(`  Code: ${code.slice(0, 120)}…`);
    }
  }

  console.log(`[render-mermaid] Done. ${rendered} rendered, ${skipped} skipped.`);
}

const isDirectRun =
  process.argv[1] != null && import.meta.url === pathToFileURL(process.argv[1]).href;

if (isDirectRun) {
  main().catch((err) => {
    console.error('[render-mermaid] Fatal error:', err);
    process.exit(1);
  });
}

export {
  decodeEdgePoints,
  estimateTextMetrics,
  expandViewBoxToFitEdges,
  normalizeLabelText,
  normalizeMermaidCode,
};
