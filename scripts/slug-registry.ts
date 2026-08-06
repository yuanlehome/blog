import fs from 'fs';
import path from 'path';
import matter from 'gray-matter';

const TRACKING_QUERY_PARAMS = new Set([
  'share_code',
  'utm_source',
  'utm_medium',
  'utm_campaign',
  'utm_content',
  'utm_term',
  'utm_psn',
  'utm_id',
  'utm_oi',
]);

function walkMarkdownFiles(dir: string, files: string[] = []): string[] {
  const entries = fs.readdirSync(dir, { withFileTypes: true });
  for (const entry of entries) {
    const fullPath = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      walkMarkdownFiles(fullPath, files);
      continue;
    }
    if (entry.isFile() && entry.name.endsWith('.md')) {
      files.push(fullPath);
    }
  }
  return files;
}

/**
 * Normalize an imported article URL for stable ownership comparisons.
 */
export function normalizeImportSourceUrl(sourceUrl: string): string {
  const trimmedUrl = sourceUrl.trim();

  try {
    const url = new URL(trimmedUrl);
    for (const key of [...url.searchParams.keys()]) {
      const normalizedKey = key.toLowerCase();
      if (normalizedKey.startsWith('utm_') || TRACKING_QUERY_PARAMS.has(normalizedKey)) {
        url.searchParams.delete(key);
      }
    }
    url.searchParams.sort();
    return url.toString();
  } catch {
    return trimmedUrl;
  }
}

/**
 * Build the stable owner ID shared by imported content on disk and new imports.
 */
export function buildImportOwnerId(provider: string, sourceUrl: string): string {
  return `import:${provider.trim().toLowerCase()}:${normalizeImportSourceUrl(sourceUrl)}`;
}

/**
 * Build slug ownership map from all blog markdown files.
 * slug -> ownerId
 */
export function buildSlugOwnerMap(contentRoot: string): Map<string, string> {
  const map = new Map<string, string>();
  if (!fs.existsSync(contentRoot)) return map;

  const files = walkMarkdownFiles(contentRoot);
  for (const file of files) {
    const raw = fs.readFileSync(file, 'utf-8');
    const { data } = matter(raw);
    const slug = String(data.slug || path.basename(file, '.md')).trim();
    if (!slug) continue;

    const notionId = data?.notion?.id || data.notionId;
    const relativePath = path.relative(contentRoot, file);
    const [provider] = relativePath.split(path.sep);
    const sourceUrl =
      typeof data.source_url === 'string' && data.source_url.trim() ? data.source_url : null;
    const ownerId = notionId
      ? String(notionId)
      : sourceUrl && provider && provider !== path.basename(relativePath)
        ? buildImportOwnerId(provider, sourceUrl)
        : `file:${relativePath}`;
    if (!map.has(slug)) {
      map.set(slug, ownerId);
    }
  }

  return map;
}
