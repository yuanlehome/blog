#!/usr/bin/env node
/**
 * Post-build asset consistency check.
 *
 * Scans every HTML file under `dist/` and verifies that every same-origin
 * `<img src>` / `<source srcset>` URL maps to a real file inside `dist/`.
 * Exits with code 1 (and prints details) if any file is missing so that CI
 * catches broken images before deployment.
 *
 * Usage:
 *   node scripts/check-assets.mjs [--dist <path>] [--base <basePath>]
 *
 * Defaults: --dist ./dist  --base /blog
 */

import { readdir, readFile } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { join, resolve } from 'node:path';

const args = process.argv.slice(2);
function getArg(flag, fallback) {
  const idx = args.indexOf(flag);
  return idx !== -1 && args[idx + 1] ? args[idx + 1] : fallback;
}

const DIST_DIR = resolve(getArg('--dist', 'dist'));
const SITE_BASE = (getArg('--base', process.env.SITE_BASE ?? '/blog') || '/').replace(/\/$/, '');

async function collectHtmlFiles(dir) {
  const entries = await readdir(dir, { withFileTypes: true });
  const files = [];
  for (const entry of entries) {
    const fullPath = join(dir, entry.name);
    if (entry.isDirectory()) {
      files.push(...(await collectHtmlFiles(fullPath)));
    } else if (entry.name.endsWith('.html')) {
      files.push(fullPath);
    }
  }
  return files;
}

/** Extract all local image src / srcset values from raw HTML. */
function extractImageSrcs(html) {
  const srcs = new Set();
  // <img src="...">
  for (const m of html.matchAll(/\bsrc="([^"]+)"/g)) {
    srcs.add(m[1]);
  }
  // <source srcset="...">
  for (const m of html.matchAll(/\bsrcset="([^"]+)"/g)) {
    // srcset may contain multiple comma-separated entries like "url 1x, url2 2x"
    for (const part of m[1].split(',')) {
      srcs.add(part.trim().split(/\s+/)[0]);
    }
  }
  return srcs;
}

/** Convert a URL path like /blog/generated/... to a dist-relative file path. */
function urlToDistPath(src, distDir, siteBase) {
  if (!src) return null;
  // Ignore external URLs, data URIs, protocol-relative URLs, and anchors
  if (/^(https?:|\/\/|data:|#)/.test(src)) return null;
  // Only check absolute paths (starts with /)
  if (!src.startsWith('/')) return null;

  let rel = src;
  if (siteBase && siteBase !== '/' && rel.startsWith(siteBase)) {
    rel = rel.slice(siteBase.length);
  }
  // Strip leading slash
  rel = rel.replace(/^\/+/, '');
  if (!rel) return null;
  return join(distDir, rel);
}

async function main() {
  if (!existsSync(DIST_DIR)) {
    console.error(`[check-assets] dist directory not found: ${DIST_DIR}`);
    console.error('  Run `npm run build` first.');
    process.exit(1);
  }

  const htmlFiles = await collectHtmlFiles(DIST_DIR);
  console.log(`[check-assets] Scanning ${htmlFiles.length} HTML files in ${DIST_DIR} …`);

  const errors = [];

  for (const htmlFile of htmlFiles) {
    const html = await readFile(htmlFile, 'utf-8');
    const srcs = extractImageSrcs(html);

    for (const src of srcs) {
      const filePath = urlToDistPath(src, DIST_DIR, SITE_BASE);
      if (!filePath) continue;
      if (!existsSync(filePath)) {
        const relativePage = htmlFile.replace(DIST_DIR, '').replace(/^[/\\]/, '');
        errors.push({ page: relativePage, src, expected: filePath });
      }
    }
  }

  if (errors.length === 0) {
    console.log('[check-assets] ✓ All image assets resolved successfully.');
    return;
  }

  console.error(`[check-assets] ✗ ${errors.length} broken image reference(s) found:\n`);
  for (const { page, src, expected } of errors) {
    console.error(`  Page : ${page}`);
    console.error(`  src  : ${src}`);
    console.error(`  File : ${expected} (NOT FOUND)`);
    console.error('');
  }
  process.exit(1);
}

main().catch((err) => {
  console.error('[check-assets] Fatal:', err);
  process.exit(1);
});
