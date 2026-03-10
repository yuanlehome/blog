import fs from 'fs';
import path from 'path';
import matter from 'gray-matter';

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
    const ownerId = notionId ? String(notionId) : `file:${path.relative(contentRoot, file)}`;
    if (!map.has(slug)) {
      map.set(slug, ownerId);
    }
  }

  return map;
}
