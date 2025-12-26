import fs from 'fs';
import path from 'path';
import matter from 'gray-matter';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const CONTENT_DIR = path.resolve(__dirname, '../src/content/blog');
const OUTPUT = path.resolve(__dirname, '../public/search-index.json');

interface SearchEntry {
  title: string;
  slug: string;
  date: string;
  tags: string[];
  excerpt: string;
}

function readPosts(dir: string): SearchEntry[] {
  const entries: SearchEntry[] = [];
  for (const file of fs.readdirSync(dir)) {
    const full = path.join(dir, file);
    const stat = fs.statSync(full);
    if (stat.isDirectory()) {
      entries.push(...readPosts(full));
      continue;
    }
    if (!file.endsWith('.md') && !file.endsWith('.mdx')) continue;
    const raw = fs.readFileSync(full, 'utf-8');
    const { data, content } = matter(raw);
    if (data.status && data.status !== 'published') continue;
    const slug = data.slug || path.basename(file, path.extname(file));
    const excerpt = content.replace(/\n+/g, ' ').slice(0, 260);
    entries.push({
      title: data.title || slug,
      slug,
      date: data.date || new Date().toISOString(),
      tags: data.tags || [],
      excerpt,
    });
  }
  return entries.sort((a, b) => new Date(b.date).getTime() - new Date(a.date).getTime());
}

function main() {
  const posts = readPosts(CONTENT_DIR);
  fs.mkdirSync(path.dirname(OUTPUT), { recursive: true });
  fs.writeFileSync(OUTPUT, JSON.stringify(posts, null, 2));
  console.log(`Saved ${posts.length} entries to ${OUTPUT}`);
}

main();
