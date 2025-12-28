import { Client } from '@notionhq/client';
import { NotionToMarkdown } from 'notion-to-md';
import type { PageObjectResponse } from '@notionhq/client/build/src/api-endpoints';
import dotenv from 'dotenv';
import fs from 'fs';
import path from 'path';
import matter from 'gray-matter';
import type { BlockObjectResponse } from '@notionhq/client/build/src/api-endpoints';
import crypto from 'crypto';
import { deriveSlug, ensureUniqueSlug } from './slug';
import { NOTION_CONTENT_DIR, NOTION_PUBLIC_IMG_DIR, ensureDir } from '../src/config/paths';

dotenv.config({ path: '.env.local' });

const NOTION_TOKEN = process.env.NOTION_TOKEN;
const DATABASE_ID = process.env.NOTION_DATABASE_ID;

if (!NOTION_TOKEN || !DATABASE_ID) {
  console.error('Missing NOTION_TOKEN or NOTION_DATABASE_ID in environment variables.');
  process.exit(1);
}

const notion = new Client({ auth: NOTION_TOKEN });
const n2m = new NotionToMarkdown({ notionClient: notion });

const MIME_EXTENSION_MAP: Record<string, string> = {
  'image/jpeg': '.jpg',
  'image/jpg': '.jpg',
  'image/png': '.png',
  'image/webp': '.webp',
  'image/gif': '.gif',
  'image/avif': '.avif',
};

// Ensure directories exist
ensureDir(NOTION_CONTENT_DIR);
ensureDir(NOTION_PUBLIC_IMG_DIR);

export function resolveExtension(url: string, contentType?: string | null) {
  const pathnameExt = path.extname(new URL(url).pathname);
  if (pathnameExt) return pathnameExt;
  if (contentType) {
    const mime = contentType.split(';')[0]?.trim().toLowerCase();
    if (mime && MIME_EXTENSION_MAP[mime]) return MIME_EXTENSION_MAP[mime];
  }
  return '.png';
}

// Helper: Download Image
export async function downloadImage(
  url: string,
  slug: string,
  imageId: string,
): Promise<string | null> {
  const dir = path.join(NOTION_PUBLIC_IMG_DIR, slug);
  if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });

  const normalizedId = imageId.replace(/[^a-zA-Z0-9-_]/g, '');
  const safeImageId =
    normalizedId || crypto.createHash('md5').update(imageId).digest('hex').slice(0, 12);
  const existingFile = fs.readdirSync(dir).find((f) => f.startsWith(safeImageId));
  if (existingFile) {
    return `/images/notion/${slug}/${existingFile}`;
  }

  let lastError: unknown = null;
  try {
    for (let attempt = 1; attempt <= 2; attempt++) {
      try {
        const controller = new AbortController();
        const timer = setTimeout(() => controller.abort(), 30000);
        try {
          const response = await fetch(url, {
            // Fail fast on very slow responses
            signal: controller.signal,
          });
          clearTimeout(timer);
          if (!response.ok) throw new Error(`Unexpected status ${response.status}`);

          const contentType = response.headers.get('content-type');
          const buffer = Buffer.from(await response.arrayBuffer());
          const ext = resolveExtension(url, contentType);
          const filename = `${safeImageId}${ext}`;
          const localPath = path.join(dir, filename);
          fs.writeFileSync(localPath, buffer);
          return `/images/notion/${slug}/${filename}`;
        } catch (error) {
          clearTimeout(timer);
          throw error;
        }
      } catch (error) {
        lastError = error;
        if (attempt < 2) {
          await new Promise((resolve) => setTimeout(resolve, 300 * attempt));
        }
      }
    }
  } catch (error) {
    lastError = error;
  }

  console.error(`Failed to download image: ${url}`, lastError);
  return null;
}

export function extractCoverUrl(props: any): string | null {
  const prop = props.cover || props.Cover || props.COVER;
  if (!prop) return null;

  if (prop.type === 'files' && Array.isArray(prop.files) && prop.files.length > 0) {
    const first = prop.files[0];
    if (first.type === 'external') return first.external.url;
    if (first.type === 'file') return first.file.url;
  }
  if (prop.type === 'url' && prop.url) return prop.url;
  if (prop.type === 'file' && prop.file?.url) return prop.file.url;
  if (prop.type === 'rich_text' && prop.rich_text?.[0]?.plain_text) {
    return prop.rich_text[0].plain_text;
  }
  return null;
}

export function getImageUrlFromBlock(
  block: BlockObjectResponse,
): { url: string; blockId: string } | null {
  if (block.type !== 'image') return null;
  const image = block.image;
  const url = image.type === 'external' ? image.external.url : image.file.url;
  if (!url) return null;
  return { url, blockId: block.id };
}

export async function findFirstImageBlock(
  blockId: string,
): Promise<{ url: string; blockId: string } | null> {
  let start_cursor: string | undefined;
  do {
    const response = await notion.blocks.children.list({ block_id: blockId, start_cursor });
    for (const block of response.results) {
      if (!('type' in block)) continue;
      const imageData = getImageUrlFromBlock(block as BlockObjectResponse);
      if (imageData) return imageData;
      if ('has_children' in block && (block as BlockObjectResponse).has_children) {
        const nested = await findFirstImageBlock(block.id);
        if (nested) return nested;
      }
    }
    start_cursor = response.has_more ? (response.next_cursor as string | undefined) : undefined;
  } while (start_cursor);
  return null;
}

// Remove the first dummy transformer and only keep the second one
let currentPageSlug = '';

export async function transformImageBlock(block: any, slug?: string, downloader = downloadImage) {
  const { image } = block as any;
  const url = image.type === 'external' ? image.external.url : image.file.url;
  const caption = image.caption?.map((c: any) => c.plain_text).join('') || '';
  const blockId = block.id;
  const targetSlug = slug || currentPageSlug;

  if (!targetSlug) {
    // Fallback if context missing
    return `![${caption}](${url})`;
  }

  const localUrl = await downloader(url, targetSlug, blockId);
  if (localUrl) {
    return `![${caption}](${localUrl})`;
  }
  return `![${caption}](${url})`;
}

n2m.setCustomTransformer('image', async (block) =>
  transformImageBlock(block, currentPageSlug, downloadImage),
);

type ExistingPostMeta = {
  slug: string;
  lastEdited?: string;
  path: string;
  notionId?: string;
};

async function getExistingPosts() {
  const files = fs.readdirSync(NOTION_CONTENT_DIR).filter((f) => f.endsWith('.md'));
  const bySlug = new Map<string, ExistingPostMeta>();
  const byNotionId = new Map<string, ExistingPostMeta>();

  for (const file of files) {
    const content = fs.readFileSync(path.join(NOTION_CONTENT_DIR, file), 'utf-8');
    const { data } = matter(content);
    const slug = data.slug || path.basename(file, '.md');
    const notionId = data.notion?.id || data.notionId;
    const meta: ExistingPostMeta = {
      slug,
      lastEdited: data.lastEditedTime,
      path: file,
      notionId,
    };
    bySlug.set(slug, meta);
    if (notionId) byNotionId.set(notionId, meta);
  }
  return { bySlug, byNotionId };
}

function moveDirContents(source: string, target: string) {
  if (!fs.existsSync(source)) return;
  if (source === target) return;
  if (!fs.existsSync(target)) fs.mkdirSync(target, { recursive: true });

  let skipped = false;
  for (const entry of fs.readdirSync(source, { withFileTypes: true })) {
    const from = path.join(source, entry.name);
    const to = path.join(target, entry.name);
    if (fs.existsSync(to)) {
      skipped = true;
      continue;
    }
    if (entry.isDirectory()) {
      fs.mkdirSync(to, { recursive: true });
      moveDirContents(from, to);
    } else if (entry.isFile()) {
      fs.copyFileSync(from, to);
    }
  }
  if (!skipped) {
    fs.rmSync(source, { recursive: true, force: true });
  }
}

function migrateImageDir(oldKey: string, newSlug: string) {
  if (!oldKey || oldKey === newSlug) return;
  const source = path.join(NOTION_PUBLIC_IMG_DIR, oldKey);
  const target = path.join(NOTION_PUBLIC_IMG_DIR, newSlug);
  moveDirContents(source, target);
}

// Type Guard
function isFullPage(page: any): page is PageObjectResponse {
  return 'properties' in page;
}

export async function sync() {
  console.log('Starting Notion Sync...');

  const existingPosts = await getExistingPosts();
  const usedSlugs = new Map<string, string>();
  for (const [slug, meta] of existingPosts.bySlug.entries()) {
    usedSlugs.set(slug, meta.notionId ?? `existing-${slug}`);
  }

  // Fetch all pages (filter in memory to avoid schema mismatch errors)
  const response = await notion.databases.query({
    database_id: DATABASE_ID!,
  });

  const pages = response.results.filter((page): page is PageObjectResponse => {
    if (!isFullPage(page)) return false;

    // Try to find a property that looks like "Status" or "status"
    const props = page.properties as any;
    const statusProp = props.Status || props.status;

    // If no status property exists, assume published (or log warning)
    if (!statusProp) return true;

    // Handle Select or Status type
    if (statusProp.type === 'select') {
      return statusProp.select?.name === 'Published';
    }
    if (statusProp.type === 'status') {
      return statusProp.status?.name === 'Published';
    }
    return false;
  });

  console.log(`Found ${pages.length} published pages.`);

  for (const page of pages) {
    // isFullPage check is redundant due to filter but good for safety if logic changes
    if (!isFullPage(page)) continue;

    const pageId = page.id;
    const lastEditedTime = page.last_edited_time;
    const props = page.properties as any;

    console.log(`Processing ${pageId}...`);

    // Extract Frontmatter fields
    const titleProp = props.Name || props.title || props.Title;
    const title = titleProp?.title?.[0]?.plain_text || 'Untitled';

    const propSlug =
      props.slug?.rich_text?.[0]?.plain_text || props.Slug?.rich_text?.[0]?.plain_text || null;
    const baseSlug = deriveSlug({ explicitSlug: propSlug, title, fallbackId: pageId });
    let slug = ensureUniqueSlug(baseSlug, pageId, usedSlugs);
    if (slug !== baseSlug) {
      console.log(`Slug conflict detected for ${pageId}: using ${slug} (base ${baseSlug})`);
    }

    const previousMeta = existingPosts.byNotionId.get(pageId);
    const previousSlug = previousMeta?.slug;
    if (previousSlug && previousSlug !== slug) {
      console.log(`Slug changed for ${pageId}: ${previousSlug} -> ${slug}`);
      migrateImageDir(previousSlug, slug);
    }
    migrateImageDir(pageId, slug);

    currentPageSlug = slug; // Set context for image transformer

    const existingBySlug = existingPosts.bySlug.get(slug);
    if (existingBySlug && existingBySlug.lastEdited === lastEditedTime) {
      console.log(`Skipping ${slug} (up to date).`);
      continue;
    }

    const date = props.date?.date?.start || new Date().toISOString().split('T')[0];
    const tags = props.tags?.multi_select?.map((t: any) => t.name) || [];

    let cover = '';
    const pageCoverUrl =
      page.cover && page.cover.type === 'external'
        ? page.cover.external.url
        : page.cover && page.cover.type === 'file'
          ? page.cover.file.url
          : null;
    const propertyCoverUrl = extractCoverUrl(props);

    const preferredCover = pageCoverUrl || propertyCoverUrl;
    if (preferredCover) {
      const localCover = await downloadImage(preferredCover, slug, 'cover');
      if (localCover) cover = localCover;
    }
    if (!cover) {
      const firstImage = await findFirstImageBlock(pageId);
      if (firstImage) {
        const fallbackCover = await downloadImage(firstImage.url, slug, firstImage.blockId);
        if (fallbackCover) cover = fallbackCover;
      }
    }

    // Convert Body to MD
    const mdblocks = await n2m.pageToMarkdown(pageId);
    const mdString = await n2m.toMarkdownString(mdblocks);

    // Construct Frontmatter
    const frontmatter = {
      title,
      slug,
      date,
      tags,
      status: 'published',
      cover,
      lastEditedTime, // Important for incremental sync
      updated: lastEditedTime,
      source: 'notion',
      notion: { id: pageId },
    };

    const fileContent = matter.stringify(mdString.parent || '', frontmatter);

    const filePath = path.join(NOTION_CONTENT_DIR, `${slug}.md`);
    fs.writeFileSync(filePath, fileContent);
    console.log(`Saved ${slug}.md`);

    if (previousSlug && previousSlug !== slug) {
      const oldPath = path.join(NOTION_CONTENT_DIR, `${previousSlug}.md`);
      if (fs.existsSync(oldPath) && oldPath !== filePath) {
        fs.rmSync(oldPath);
        console.log(`Removed old file for renamed slug: ${previousSlug}.md`);
      }
    }
  }

  console.log('Sync complete.');
}

if (process.env.NODE_ENV !== 'test') {
  sync().catch(console.error);
}
