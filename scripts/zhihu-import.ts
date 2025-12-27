import { chromium } from '@playwright/test';
import fs from 'fs';
import matter from 'gray-matter';
import path from 'path';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import remarkStringify from 'remark-stringify';
import rehypeParse from 'rehype-parse';
import rehypeRaw from 'rehype-raw';
import rehypeRemark from 'rehype-remark';
import slugify from 'slugify';
import { unified } from 'unified';
import { visit } from 'unist-util-visit';
import { fileURLToPath } from 'url';

type HastElement = {
  type?: string;
  tagName?: string;
  properties?: Record<string, any>;
  children?: HastElement[];
  value?: string;
};

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const CONTENT_DIR = path.join(__dirname, '../src/content/blog/zhihu');
const IMAGE_ROOT = path.join(__dirname, '../public/images/zhihu');

function parseArgs(): string {
  const url =
    process.argv.find((arg) => arg.startsWith('--url='))?.replace('--url=', '') ||
    process.env.URL ||
    process.env.url ||
    process.argv[2];

  if (!url) {
    throw new Error('Usage: npm run zhihu:import -- --url=<ZH_URL>');
  }

  return url;
}

function hasClass(node: HastElement, className: string) {
  const classNames = node.properties?.className;
  if (Array.isArray(classNames)) return classNames.includes(className);
  if (typeof classNames === 'string') return classNames.split(/\s+/).includes(className);
  return false;
}

function getTextContent(node: HastElement): string {
  if (!node) return '';
  if (node.type === 'text' && typeof node.value === 'string') return node.value;
  if (node.children?.length) {
    return node.children.map((child) => getTextContent(child)).join('');
  }
  return '';
}

function normalizeUrl(url: string, base?: string) {
  if (!url) return '';
  if (url.startsWith('//')) return `https:${url}`;
  if (/^https?:\/\//i.test(url)) return url;
  if (base) {
    try {
      return new URL(url, base).toString();
    } catch {
      return '';
    }
  }
  return '';
}

function resolveImageSrc(node: HastElement, base?: string) {
  const props = node.properties || {};
  const candidates = [
    props['data-original'],
    props['data-actualsrc'],
    props['data-src'],
    props.src,
    props['data-actual-url'],
  ];
  const url = candidates.find((u) => typeof u === 'string' && u.trim().length > 0);
  if (!url || url.startsWith('data:')) return '';
  return normalizeUrl(url, base);
}

async function downloadImage(url: string, slug: string, index: number): Promise<string | null> {
  try {
    const finalUrl = normalizeUrl(url);
    if (!finalUrl) return null;

    const extFromUrl = path.extname(new URL(finalUrl).pathname).split('?')[0];
    const ext = extFromUrl || '.jpg';
    const dir = path.join(IMAGE_ROOT, slug);
    fs.mkdirSync(dir, { recursive: true });

    const filename = `${String(index + 1).padStart(3, '0')}${ext}`;
    const localPath = path.join(dir, filename);

    if (fs.existsSync(localPath)) {
      return `/images/zhihu/${slug}/${filename}`;
    }

    const res = await fetch(finalUrl, {
      headers: {
        'User-Agent':
          'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36',
        Referer: 'https://www.zhihu.com',
      },
    });

    if (!res.ok) {
      console.warn(`Failed to fetch image ${finalUrl}: ${res.status}`);
      return null;
    }

    const buffer = Buffer.from(await res.arrayBuffer());
    fs.writeFileSync(localPath, buffer);
    return `/images/zhihu/${slug}/${filename}`;
  } catch (error) {
    console.warn(`Failed to download image ${url}`, error);
    return null;
  }
}

function transformMath() {
  return (tree: HastElement) => {
    visit(tree as any, 'element', (node: HastElement, index: number | null, parent: any) => {
      if (!parent || index === null || index === undefined) return;
      if (node.tagName === 'span' && hasClass(node, 'ztext-math')) {
        const latex = (node.properties?.['data-tex'] as string) || getTextContent(node);
        const value = `$${latex.trim()}$`;
        parent.children.splice(index, 1, { type: 'text', value });
      }
      if (node.tagName === 'div' && hasClass(node, 'ztext-math')) {
        const latex = (node.properties?.['data-tex'] as string) || getTextContent(node);
        const value = `$$${latex.trim()}$$`;
        parent.children.splice(index, 1, { type: 'text', value });
      }
    });
  };
}

function localizeImages(options: { slug: string; baseUrl?: string; collected: string[] }) {
  return async (tree: HastElement) => {
    const imageNodes: { node: HastElement; url: string }[] = [];

    visit(tree as any, 'element', (node: HastElement) => {
      if (node.tagName !== 'img') return;
      const src = resolveImageSrc(node, options.baseUrl);
      if (!src) return;
      imageNodes.push({ node, url: src });
    });

    const mapping = new Map<string, string>();
    let index = 0;
    for (const { url } of imageNodes) {
      if (mapping.has(url)) continue;
      const local = await downloadImage(url, options.slug, index);
      if (local) {
        mapping.set(url, local);
        index += 1;
        if (!options.collected.includes(local)) {
          options.collected.push(local);
        }
      }
    }

    imageNodes.forEach(({ node, url }) => {
      const local = mapping.get(url);
      if (local) {
        node.properties = { ...(node.properties || {}), src: local };
        delete node.properties?.['data-original'];
        delete node.properties?.['data-actualsrc'];
        delete node.properties?.['data-src'];
        delete node.properties?.srcset;
      }
    });
  };
}

async function htmlToMdx(html: string, slug: string, baseUrl?: string) {
  const images: string[] = [];
  const file = await unified()
    .use(rehypeParse, { fragment: true })
    .use(rehypeRaw)
    .use(transformMath())
    .use(localizeImages({ slug, baseUrl, collected: images }))
    .use(rehypeRemark, { allowDangerousHtml: true })
    .use(remarkMath)
    .use(remarkGfm)
    .use(remarkStringify, {
      allowDangerousHtml: true,
      fences: true,
      bullet: '-',
      rule: '-',
    })
    .process(html);

  return { markdown: String(file).trim(), images };
}

async function fetchArticle(url: string) {
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage({
    userAgent:
      'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36',
  });

  try {
    await page.goto(url, { waitUntil: 'networkidle', timeout: 120000 });
    await page.waitForSelector('.Post-RichText, .RichText, article', { timeout: 30000 });

    const result = await page.evaluate(() => {
      const pickText = (selectors: string[]) => {
        for (const sel of selectors) {
          const el = document.querySelector(sel);
          const text = el?.textContent?.trim();
          if (text) return text;
        }
        return '';
      };

      const pickMeta = (selectors: string[]) => {
        for (const sel of selectors) {
          const meta = document.querySelector(sel) as HTMLMetaElement | null;
          const content = meta?.getAttribute('content')?.trim();
          if (content) return content;
        }
        return '';
      };

      const contentSelectors = ['.Post-RichText', '.RichText', 'article'];
      let html = '';
      for (const sel of contentSelectors) {
        const el = document.querySelector(sel);
        if (el) {
          html = (el as HTMLElement).innerHTML;
          break;
        }
      }

      return {
        title:
          pickText(['h1.Post-Title', 'h1.RichText-Title', '.Post-Title', 'h1']) ||
          document.title ||
          'Zhihu Article',
        author:
          pickMeta(['meta[name=\"author\"]']) ||
          pickText(['.AuthorInfo-name', '.ContentItem-author .UserLink-link', '.UserLink-link']),
        published: pickMeta([
          'meta[itemprop=\"datePublished\"]',
          'meta[property=\"article:published_time\"]',
          'meta[name=\"publish_date\"]',
        ]),
        html,
      };
    });

    if (!result.html?.trim()) {
      throw new Error('Failed to extract article content.');
    }

    return result;
  } finally {
    await browser.close();
  }
}

async function main() {
  fs.mkdirSync(CONTENT_DIR, { recursive: true });
  fs.mkdirSync(IMAGE_ROOT, { recursive: true });

  const targetUrl = parseArgs();
  const { title, author, published, html } = await fetchArticle(targetUrl);

  const slugFromTitle = slugify(title, { lower: true, strict: true });
  const fallbackSlug = slugify(new URL(targetUrl).pathname.split('/').filter(Boolean).pop() || '', {
    lower: true,
    strict: true,
  });
  const slug = slugFromTitle || fallbackSlug || `zhihu-${Date.now()}`;

  const { markdown, images } = await htmlToMdx(html, slug, targetUrl);

  const publishedDate = published ? new Date(published) : new Date();
  const safeDate = Number.isNaN(publishedDate.valueOf()) ? new Date() : publishedDate;

  const frontmatter: Record<string, any> = {
    title: title || 'Zhihu Article',
    date: safeDate.toISOString().split('T')[0],
    tags: ['zhihu'],
    status: 'published',
    source: 'zhihu',
    source_url: targetUrl,
    source_author: author || 'Zhihu',
    imported_at: new Date().toISOString(),
  };

  if (images[0]) {
    frontmatter.cover = images[0];
  }

  const fileContent = matter.stringify(markdown, frontmatter);
  const filepath = path.join(CONTENT_DIR, `${slug}.mdx`);
  fs.writeFileSync(filepath, fileContent);

  console.log(`Saved article to ${filepath}`);
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
