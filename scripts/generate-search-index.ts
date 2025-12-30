/**
 * Search index generation script
 *
 * This script generates the search index from blog posts at build time.
 * Run with: npm run build (as part of the build process)
 *
 * @module scripts/generate-search-index
 */

import * as fs from 'fs';
import * as path from 'path';
import { fileURLToPath } from 'url';

// Get directory paths
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const ROOT_DIR = path.resolve(__dirname, '..');
const CONTENT_DIR = path.join(ROOT_DIR, 'src/content/blog');
const OUTPUT_DIR = path.join(ROOT_DIR, 'public');
const OUTPUT_FILE = path.join(OUTPUT_DIR, 'search-index.json');

// Import indexer utilities - we'll use the compiled version or raw TS
// For build-time, we parse markdown directly

/**
 * Remove markdown code blocks from text
 */
function removeCodeBlocks(text: string): string {
  return text.replace(/```[\s\S]*?```|~~~[\s\S]*?~~~/g, '');
}

/**
 * Remove markdown frontmatter from text
 */
function removeFrontmatter(text: string): string {
  return text.replace(/^---[\s\S]*?---\n?/, '');
}

/**
 * Remove HTML tags from text
 */
function removeHtmlTags(text: string): string {
  return text.replace(/<[^>]+>/g, '');
}

/**
 * Remove markdown inline formatting but keep text
 */
function removeMarkdownFormatting(text: string): string {
  return text
    .replace(/`[^`]+`/g, '')
    .replace(/!\[([^\]]*)\]\([^)]+\)/g, '')
    .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1')
    .replace(/\*\*([^*]+)\*\*/g, '$1')
    .replace(/\*([^*]+)\*/g, '$1')
    .replace(/__([^_]+)__/g, '$1')
    .replace(/_([^_]+)_/g, '$1')
    .replace(/~~([^~]+)~~/g, '$1')
    .replace(/^>\s*/gm, '')
    .replace(/^[-*_]{3,}\s*$/gm, '')
    .replace(/^[\s]*[-*+]\s+/gm, '')
    .replace(/^[\s]*\d+\.\s+/gm, '');
}

/**
 * Extract headings from markdown content
 */
function extractHeadings(text: string): string[] {
  const headings: string[] = [];
  const headingRegex = /^#{1,6}\s+(.+)$/gm;
  let match;

  while ((match = headingRegex.exec(text)) !== null) {
    const heading = match[1].trim();
    if (heading) {
      headings.push(heading);
    }
  }

  return headings;
}

/**
 * Convert markdown to plain text for indexing
 */
function markdownToPlainText(markdown: string): string {
  let text = removeFrontmatter(markdown);
  text = removeCodeBlocks(text);
  text = removeHtmlTags(text);
  text = removeMarkdownFormatting(text);
  text = text.replace(/\n+/g, ' ').replace(/\s+/g, ' ').trim();
  return text;
}

/**
 * Truncate text to a maximum length
 */
function truncateText(text: string, maxLength: number): string {
  if (text.length <= maxLength) {
    return text;
  }
  const truncated = text.substring(0, maxLength);
  const lastSpace = truncated.lastIndexOf(' ');
  if (lastSpace > maxLength * 0.8) {
    return truncated.substring(0, lastSpace) + '...';
  }
  return truncated + '...';
}

/**
 * Parse YAML frontmatter from markdown content
 */
function parseFrontmatter(content: string): { data: Record<string, unknown>; content: string } {
  const frontmatterRegex = /^---\n([\s\S]*?)\n---\n([\s\S]*)$/;
  const match = content.match(frontmatterRegex);

  if (!match) {
    return { data: {}, content };
  }

  const yamlContent = match[1];
  const bodyContent = match[2];

  // Simple YAML parser for basic frontmatter
  const data: Record<string, unknown> = {};

  const lines = yamlContent.split('\n');
  let currentKey = '';
  let inArray = false;
  let arrayValues: string[] = [];

  for (const line of lines) {
    // Handle array items
    if (line.match(/^\s+-\s+(.+)$/)) {
      const arrayMatch = line.match(/^\s+-\s+(.+)$/);
      if (arrayMatch && inArray) {
        arrayValues.push(arrayMatch[1].replace(/^['"]|['"]$/g, ''));
      }
      continue;
    }

    // End of array
    if (inArray && currentKey) {
      data[currentKey] = arrayValues;
      inArray = false;
      arrayValues = [];
    }

    // Key-value pairs
    const kvMatch = line.match(/^(\w+):\s*(.*)$/);
    if (kvMatch) {
      const key = kvMatch[1];
      const value = kvMatch[2].trim();

      if (value === '' || value === '[]') {
        // Possible array start or empty value
        currentKey = key;
        if (value === '[]') {
          data[key] = [];
        } else {
          inArray = true;
          arrayValues = [];
        }
      } else if (value.startsWith('[') && value.endsWith(']')) {
        // Inline array
        const items = value
          .slice(1, -1)
          .split(',')
          .map((s) => s.trim().replace(/^['"]|['"]$/g, ''));
        data[key] = items.filter(Boolean);
      } else if (value.startsWith("'") || value.startsWith('"')) {
        data[key] = value.replace(/^['"]|['"]$/g, '');
      } else if (value === 'true') {
        data[key] = true;
      } else if (value === 'false') {
        data[key] = false;
      } else if (!isNaN(Number(value))) {
        data[key] = Number(value);
      } else {
        data[key] = value;
      }
    }
  }

  // Handle any remaining array
  if (inArray && currentKey) {
    data[currentKey] = arrayValues;
  }

  return { data, content: bodyContent };
}

interface SearchIndexEntry {
  slug: string;
  url: string;
  title: string;
  headings: string[];
  tags: string[];
  date: string;
  summary: string;
  body: string;
  source?: string;
}

interface SearchIndex {
  version: number;
  generatedAt: string;
  count: number;
  tags: Record<string, number>;
  entries: SearchIndexEntry[];
}

/**
 * Calculate tag counts from entries
 */
function calculateTagCounts(entries: SearchIndexEntry[]): Record<string, number> {
  const counts: Record<string, number> = {};
  for (const entry of entries) {
    for (const tag of entry.tags) {
      counts[tag] = (counts[tag] || 0) + 1;
    }
  }
  return counts;
}

/**
 * Extract source type from the file path
 */
function extractSourceType(filePath: string): string | undefined {
  if (filePath.includes('/notion/') || filePath.includes('\\notion\\')) return 'notion';
  if (filePath.includes('/wechat/') || filePath.includes('\\wechat\\')) return 'wechat';
  if (filePath.includes('/others/') || filePath.includes('\\others\\')) return 'others';
  return undefined;
}

/**
 * Get all markdown files recursively
 */
function getMarkdownFiles(dir: string): string[] {
  const files: string[] = [];

  if (!fs.existsSync(dir)) {
    return files;
  }

  const entries = fs.readdirSync(dir, { withFileTypes: true });

  for (const entry of entries) {
    const fullPath = path.join(dir, entry.name);

    if (entry.isDirectory()) {
      files.push(...getMarkdownFiles(fullPath));
    } else if (entry.isFile() && (entry.name.endsWith('.md') || entry.name.endsWith('.mdx'))) {
      files.push(fullPath);
    }
  }

  return files;
}

/**
 * Process a single markdown file
 */
function processFile(filePath: string): SearchIndexEntry | null {
  try {
    const content = fs.readFileSync(filePath, 'utf-8');
    const { data, content: body } = parseFrontmatter(content);

    // Skip drafts
    if (data.status === 'draft') {
      return null;
    }

    // Extract slug from filename or frontmatter
    const slug = (data.slug as string) || path.basename(filePath, path.extname(filePath));
    const title = (data.title as string) || slug;
    const tags = (data.tags as string[]) || [];
    const date = data.date ? new Date(data.date as string).toISOString() : new Date().toISOString();
    const source = extractSourceType(filePath);

    // Process content
    const cleanBody = removeFrontmatter(body);
    const headings = extractHeadings(cleanBody);
    const plainText = markdownToPlainText(cleanBody);
    const truncatedBody = truncateText(plainText, 20000);
    const summary = truncateText(plainText, 200);

    return {
      slug,
      url: `/${slug}/`,
      title,
      headings,
      tags,
      date,
      summary,
      body: truncatedBody,
      source,
    };
  } catch (error) {
    console.error(`Error processing ${filePath}:`, error);
    return null;
  }
}

/**
 * Main function to generate the search index
 */
async function generateSearchIndex(): Promise<void> {
  console.log('🔍 Generating search index...');

  // Get all markdown files
  const files = getMarkdownFiles(CONTENT_DIR);
  console.log(`📄 Found ${files.length} markdown files`);

  // Process each file
  const entries: SearchIndexEntry[] = [];

  for (const file of files) {
    const entry = processFile(file);
    if (entry) {
      entries.push(entry);
    }
  }

  console.log(`✅ Processed ${entries.length} published posts`);

  // Sort by date (newest first)
  entries.sort((a, b) => new Date(b.date).getTime() - new Date(a.date).getTime());

  // Create index
  const index: SearchIndex = {
    version: 1,
    generatedAt: new Date().toISOString(),
    count: entries.length,
    tags: calculateTagCounts(entries),
    entries,
  };

  // Ensure output directory exists
  if (!fs.existsSync(OUTPUT_DIR)) {
    fs.mkdirSync(OUTPUT_DIR, { recursive: true });
  }

  // Write index file
  fs.writeFileSync(OUTPUT_FILE, JSON.stringify(index), 'utf-8');

  // Calculate file size
  const stats = fs.statSync(OUTPUT_FILE);
  const sizeKB = (stats.size / 1024).toFixed(2);

  console.log(`📦 Search index written to ${OUTPUT_FILE} (${sizeKB} KB)`);
  console.log(`🏷️  Tags: ${Object.keys(index.tags).length}`);
}

// Run if called directly
generateSearchIndex().catch((error) => {
  console.error('Failed to generate search index:', error);
  process.exit(1);
});
