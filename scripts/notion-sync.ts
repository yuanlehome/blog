import { Client } from '@notionhq/client';
import { NotionToMarkdown } from 'notion-to-md';
import type { PageObjectResponse, PartialPageObjectResponse, DatabaseObjectResponse } from '@notionhq/client/build/src/api-endpoints';
import dotenv from 'dotenv';
import fs from 'fs';
import path from 'path';
import axios from 'axios';
import slugify from 'slugify';
import matter from 'gray-matter';
import { fileURLToPath } from 'url';

dotenv.config({ path: '.env.local' });

const NOTION_TOKEN = process.env.NOTION_TOKEN;
const DATABASE_ID = process.env.NOTION_DATABASE_ID;

if (!NOTION_TOKEN || !DATABASE_ID) {
  console.error('Missing NOTION_TOKEN or NOTION_DATABASE_ID in environment variables.');
  process.exit(1);
}

const notion = new Client({ auth: NOTION_TOKEN });
const n2m = new NotionToMarkdown({ notionClient: notion });

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const CONTENT_DIR = path.join(__dirname, '../src/content/blog/notion');
const PUBLIC_IMG_DIR = path.join(__dirname, '../public/images/notion');

// Ensure directories exist
if (!fs.existsSync(CONTENT_DIR)) fs.mkdirSync(CONTENT_DIR, { recursive: true });
if (!fs.existsSync(PUBLIC_IMG_DIR)) fs.mkdirSync(PUBLIC_IMG_DIR, { recursive: true });

// Helper: Download Image
async function downloadImage(url: string, pageId: string, imageId: string): Promise<string | null> {
  try {
    const ext = path.extname(new URL(url).pathname).split('?')[0] || '.png'; // Default to png if no ext
    const dir = path.join(PUBLIC_IMG_DIR, pageId);
    if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
    
    const filename = `${imageId}${ext}`;
    const localPath = path.join(dir, filename);
    const publicPath = `/images/notion/${pageId}/${filename}`;

    // Check if file exists (optional: could check size/hash, but simple existence is fast)
    if (fs.existsSync(localPath)) {
        return publicPath;
    }

    const response = await axios({
      url,
      method: 'GET',
      responseType: 'stream',
    });

    const writer = fs.createWriteStream(localPath);
    response.data.pipe(writer);

    return new Promise((resolve, reject) => {
      writer.on('finish', () => resolve(publicPath));
      writer.on('error', reject);
    });
  } catch (error) {
    console.error(`Failed to download image: ${url}`, error);
    return null;
  }
}

// Remove the first dummy transformer and only keep the second one
let currentPageId = '';

n2m.setCustomTransformer('image', async (block) => {
    const { image } = block as any;
    const url = image.type === 'external' ? image.external.url : image.file.url;
    const caption = image.caption?.map((c: any) => c.plain_text).join('') || '';
    const blockId = block.id;
    
    if (!currentPageId) {
        // Fallback if context missing
        return `![${caption}](${url})`; 
    }

    const localUrl = await downloadImage(url, currentPageId, blockId);
    if (localUrl) {
        return `![${caption}](${localUrl})`;
    }
    return `![${caption}](${url})`;
});


async function getExistingPosts() {
  const files = fs.readdirSync(CONTENT_DIR).filter(f => f.endsWith('.md'));
  const map = new Map<string, { lastEdited: string, path: string }>();
  
  for (const file of files) {
    const content = fs.readFileSync(path.join(CONTENT_DIR, file), 'utf-8');
    const { data } = matter(content);
    if (data.notionId && data.lastEditedTime) {
      map.set(data.notionId, { lastEdited: data.lastEditedTime, path: file });
    }
  }
  return map;
}

// Type Guard
function isFullPage(page: any): page is PageObjectResponse {
    return 'properties' in page;
}

async function sync() {
  console.log('Starting Notion Sync...');
  
  const existingPosts = await getExistingPosts();

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
    
    // Check incremental
    const existing = existingPosts.get(pageId);
    if (existing && existing.lastEdited === lastEditedTime) {
      console.log(`Skipping ${pageId} (up to date).`);
      continue;
    }

    console.log(`Processing ${pageId}...`);
    currentPageId = pageId; // Set context for image transformer

    // Extract Frontmatter
    // Look for property named 'Name' (standard) or 'title' or 'Title'
    const titleProp = props.Name || props.title || props.Title;
    const title = titleProp?.title?.[0]?.plain_text || 'Untitled';
    
    let slug = props.slug?.rich_text?.[0]?.plain_text || props.Slug?.rich_text?.[0]?.plain_text;
    if (!slug) {
        slug = slugify(title, { lower: true, strict: true }) || pageId;
    }
    
    const date = props.date?.date?.start || new Date().toISOString().split('T')[0];
    const tags = props.tags?.multi_select?.map((t: any) => t.name) || [];
    const excerpt = props.excerpt?.rich_text[0]?.plain_text || '';
    
    let cover = '';
    if (page.cover) {
        const coverUrl = page.cover.type === 'external' ? page.cover.external.url : page.cover.file.url;
        const localCover = await downloadImage(coverUrl, pageId, 'cover');
        if (localCover) cover = localCover;
    }

    // Convert Body to MD
    const mdblocks = await n2m.pageToMarkdown(pageId);
    const mdString = n2m.toMarkdownString(mdblocks);

    // Construct Frontmatter
    const frontmatter = {
      title,
      slug,
      date,
      tags,
      status: 'published',
      excerpt,
      cover,
      notionId: pageId,
      lastEditedTime, // Important for incremental sync
    };

    const fileContent = matter.stringify(mdString.parent || '', frontmatter);
    
    const filePath = path.join(CONTENT_DIR, `${slug}.md`);
    fs.writeFileSync(filePath, fileContent);
    console.log(`Saved ${slug}.md`);
  }
  
  console.log('Sync complete.');
}

sync().catch(console.error);
