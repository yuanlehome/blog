import fs from 'fs';
import path from 'path';
import matter from 'gray-matter';
import { fileURLToPath } from 'url';
import { MiniSearchLite } from '../src/utils/mini-search-lite';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const CONTENT_DIR = path.join(__dirname, '../src/content/blog');
const OUTPUT_DIR = path.join(__dirname, '../public/search');

const searchFields = ['title', 'headings', 'paragraph_text', 'tags', 'date'];

function walkMarkdownFiles(dir: string): string[] {
  const entries = fs.readdirSync(dir, { withFileTypes: true });
  const files: string[] = [];

  for (const entry of entries) {
    const fullPath = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      files.push(...walkMarkdownFiles(fullPath));
    } else if (/\.(md|mdx)$/i.test(entry.name)) {
      files.push(fullPath);
    }
  }

  return files;
}

function stripMarkdown(content: string): string {
  return content
    .replace(/```[\s\S]*?```/g, ' ')
    .replace(/`([^`]*)`/g, '$1')
    .replace(/\[([^\]]+)\]\([^\)]+\)/g, '$1')
    .replace(/\!\[[^\]]*\]\([^\)]*\)/g, ' ')
    .replace(/[>*_~#`]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

function relativeSlug(filePath: string): string {
  const relPath = path.relative(CONTENT_DIR, filePath).replace(/\\/g, '/');
  return relPath.replace(/\.(md|mdx)$/i, '');
}

function cleanParagraph(paragraph: string): string {
  return stripMarkdown(paragraph).replace(/\s+/g, ' ').trim();
}

function collectDocumentsFromBody(body: string, baseSlug: string, title: string, tags: string[], date: string) {
  const lines = body.split(/\r?\n/);
  let paragraphBuffer: string[] = [];
  let documents: any[] = [];
  let metadata: any[] = [];
  let paragraphIndex = 0;
  const headingStack: string[] = [];

  const flushParagraph = () => {
    const raw = paragraphBuffer.join(' ').trim();
    const paragraph = cleanParagraph(raw);
    paragraphBuffer = [];
    if (!paragraph) return;

    paragraphIndex += 1;
    const docId = `${baseSlug}#p${paragraphIndex}`;
    const headingTrail = headingStack.join(' › ');

    const doc = {
      id: docId,
      title,
      headings: headingTrail || title,
      paragraph_text: paragraph,
      tags,
      date,
    };
    documents.push(doc);
    metadata.push({
      id: docId,
      slug: baseSlug,
      title,
      heading: headingTrail,
      paragraph,
      tags,
      date,
    });
  };

  for (const line of lines) {
    const headingMatch = /^(#{1,6})\s+(.*)/.exec(line);
    if (headingMatch) {
      flushParagraph();
      const level = headingMatch[1].length;
      const text = cleanParagraph(headingMatch[2]);
      headingStack.splice(level - 1);
      headingStack[level - 1] = text;
      continue;
    }

    if (line.trim() === '') {
      flushParagraph();
      continue;
    }

    paragraphBuffer.push(line);
  }

  flushParagraph();

  return { documents, metadata };
}

function buildIndex() {
  if (!fs.existsSync(CONTENT_DIR)) {
    console.warn(`Content directory not found: ${CONTENT_DIR}`);
    return;
  }

  if (!fs.existsSync(OUTPUT_DIR)) {
    fs.mkdirSync(OUTPUT_DIR, { recursive: true });
  }

  const files = walkMarkdownFiles(CONTENT_DIR);
  const indexer = new MiniSearchLite({
    fields: searchFields,
    weights: {
      title: 4,
      headings: 3,
      paragraph_text: 2,
      tags: 1.5,
      date: 1,
    },
  });

  let allDocs: any[] = [];
  let allMetadata: any[] = [];

  for (const file of files) {
    const raw = fs.readFileSync(file, 'utf-8');
    const { data, content } = matter(raw);
    if (data.status && data.status !== 'published') continue;

    const title = data.title ?? relativeSlug(file);
    const slug = relativeSlug(file);
    const tags = Array.isArray(data.tags) ? data.tags : [];
    const date = data.date ? new Date(data.date).toISOString().slice(0, 10) : '';

    const { documents, metadata } = collectDocumentsFromBody(content, slug, title, tags, date);
    allDocs = allDocs.concat(documents);
    allMetadata = allMetadata.concat(metadata);
  }

  indexer.addAll(allDocs);

  const indexPath = path.join(OUTPUT_DIR, 'index.json');
  const metaPath = path.join(OUTPUT_DIR, 'metadata.json');

  fs.writeFileSync(indexPath, JSON.stringify(indexer.toJSON()));
  fs.writeFileSync(metaPath, JSON.stringify(allMetadata, null, 2));

  console.log(`Generated search index with ${allDocs.length} documents across ${files.length} posts.`);
}

buildIndex();
