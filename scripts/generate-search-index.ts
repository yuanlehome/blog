import fs from "fs";
import path from "path";
import matter from "gray-matter";
import { fileURLToPath } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT_DIR = path.join(__dirname, "..");
const OUTPUT_FILE = path.join(ROOT_DIR, "public", "search-index.json");

const KNOWN_CONTENT_DIRS = [
  path.join(ROOT_DIR, "content", "posts"),
  path.join(ROOT_DIR, "content", "notion"),
  path.join(ROOT_DIR, "src", "content", "blog"),
  path.join(ROOT_DIR, "src", "content", "blog", "notion"),
];

const MAX_BODY_LENGTH = 1000;
const MAX_EXCERPT_LENGTH = 200;

interface SearchRecord {
  id: string;
  title: string;
  slug: string;
  date: string;
  tags: string[];
  excerpt: string;
  content: string;
  source?: string;
}

const stripMarkdown = (markdown: string) => {
  return markdown
    .replace(/```[\s\S]*?```/g, " ")
    .replace(/`([^`]*)`/g, "$1")
    .replace(/\!\[[^\]]*\]\([^)]*\)/g, " ")
    .replace(/\[[^\]]*\]\([^)]*\)/g, "$1")
    .replace(/<[^>]+>/g, " ")
    .replace(/[#>*_~`>-]+/g, " ")
    .replace(/\s+/g, " ")
    .trim();
};

const ensureDirectories = (dirs: string[]) => {
  return dirs.filter((dir) => fs.existsSync(dir));
};

const walkMarkdownFiles = (dir: string): string[] => {
  const entries = fs.readdirSync(dir, { withFileTypes: true });
  const files: string[] = [];

  for (const entry of entries) {
    const entryPath = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      files.push(...walkMarkdownFiles(entryPath));
    } else if (entry.isFile() && /(\.md|\.mdx)$/.test(entry.name)) {
      files.push(entryPath);
    }
  }

  return files;
};

const safeSlug = (slug: string) =>
  slug
    .toLowerCase()
    .replace(/[^a-z0-9\u4e00-\u9fa5-]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .replace(/-+/g, "-")
    .trim();

const uniqueSlugger = () => {
  const seen = new Map<string, number>();
  return (raw: string) => {
    const base = raw || "post";
    const normalized = safeSlug(base) || "post";
    const count = seen.get(normalized) ?? 0;
    seen.set(normalized, count + 1);
    if (count === 0) return normalized;
    return `${normalized}-${count + 1}`;
  };
};

const buildExcerpt = (text: string, fallback?: string) => {
  if (fallback && fallback.trim()) return fallback.trim();
  if (!text) return "";
  const snippet = text.slice(0, MAX_EXCERPT_LENGTH);
  return snippet + (text.length > MAX_EXCERPT_LENGTH ? "…" : "");
};

const truncateBody = (text: string) => {
  if (!text) return "";
  return text.slice(0, MAX_BODY_LENGTH);
};

const collectRecords = (contentDirs: string[]): SearchRecord[] => {
  const records: SearchRecord[] = [];
  const makeSlug = uniqueSlugger();

  for (const dir of contentDirs) {
    const files = walkMarkdownFiles(dir);
    for (const filePath of files) {
      const raw = fs.readFileSync(filePath, "utf-8");
      const { data, content } = matter(raw);
      const status = (data.status as string) || "published";
      if (status !== "published") continue;

      const baseName = path.basename(filePath, path.extname(filePath));
      const slug = makeSlug((data.slug as string) || baseName);
      const plain = stripMarkdown(String(content));

      const record: SearchRecord = {
        id: slug,
        title: String(data.title || baseName),
        slug,
        date: data.date ? new Date(data.date).toISOString() : new Date().toISOString(),
        tags: Array.isArray(data.tags) ? data.tags.map(String) : [],
        excerpt: buildExcerpt(plain, data.excerpt as string | undefined),
        content: truncateBody(plain),
        source: data.source || undefined,
      };

      records.push(record);
    }
  }

  return records.sort((a, b) => b.date.localeCompare(a.date));
};

const run = () => {
  const availableDirs = ensureDirectories(KNOWN_CONTENT_DIRS);
  if (availableDirs.length === 0) {
    console.warn("No content directories found. Skipping search index generation.");
    return;
  }

  const records = collectRecords(availableDirs);
  const payload = {
    generatedAt: new Date().toISOString(),
    count: records.length,
    items: records,
  };

  fs.mkdirSync(path.dirname(OUTPUT_FILE), { recursive: true });
  fs.writeFileSync(OUTPUT_FILE, JSON.stringify(payload, null, 2), "utf-8");

  console.log(`Search index generated with ${records.length} entries at ${OUTPUT_FILE}`);
};

run();
