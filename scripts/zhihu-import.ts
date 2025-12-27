import { chromium, type BrowserContext, type Page } from '@playwright/test';
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

type ExtractedArticle = {
  title: string;
  author?: string;
  published?: string;
  html: string;
  baseUrl?: string;
};

type Provider = {
  name: string;
  match: (url: string) => boolean;
  extract: (page: Page, url: string) => Promise<ExtractedArticle>;
};

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const CONTENT_ROOT = path.join(__dirname, '../src/content/blog');
const IMAGE_ROOT = path.join(__dirname, '../public/images');

function parseArgs(): string {
  const url =
    process.argv.find((arg) => arg.startsWith('--url='))?.replace('--url=', '') ||
    process.env.URL ||
    process.env.url ||
    process.argv[2];

  if (!url) {
    throw new Error('Usage: npm run zhihu:import -- --url=<URL>');
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

async function downloadImage(
  url: string,
  provider: string,
  slug: string,
  imageRoot: string,
  index: number,
): Promise<string | null> {
  try {
    const finalUrl = normalizeUrl(url);
    if (!finalUrl) return null;

    const extFromUrl = path.extname(new URL(finalUrl).pathname).split('?')[0];
    const ext = extFromUrl || '.jpg';
    const dir = path.join(imageRoot, slug);
    fs.mkdirSync(dir, { recursive: true });

    const filename = `${String(index + 1).padStart(3, '0')}${ext}`;
    const localPath = path.join(dir, filename);

    if (fs.existsSync(localPath)) {
      return `/images/${provider}/${slug}/${filename}`;
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
    return `/images/${provider}/${slug}/${filename}`;
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

function localizeImages(options: {
  slug: string;
  baseUrl?: string;
  collected: string[];
  provider: string;
  imageRoot: string;
}) {
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
      const local = await downloadImage(url, options.provider, options.slug, options.imageRoot, index);
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

async function htmlToMdx(
  html: string,
  options: { slug: string; provider: string; baseUrl?: string; imageRoot: string },
) {
  const images: string[] = [];
  const file = await unified()
    .use(rehypeParse, { fragment: true })
    .use(rehypeRaw)
    .use(transformMath())
    .use(localizeImages({ slug: options.slug, baseUrl: options.baseUrl, collected: images, provider: options.provider, imageRoot: options.imageRoot }))
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

async function withBrowser<T>(fn: (context: BrowserContext) => Promise<T>) {
  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext({
    userAgent:
      'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36',
  });
  try {
    return await fn(context);
  } finally {
    await context.close();
    await browser.close();
  }
}

const providers: Provider[] = [
  {
    name: 'zhihu',
    match: (url) => /zhihu\.com/.test(url),
    extract: async (page, url) => {
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
            pickMeta(['meta[name="author"]']) ||
            pickText(['.AuthorInfo-name', '.ContentItem-author .UserLink-link', '.UserLink-link']),
          published: pickMeta([
            'meta[itemprop="datePublished"]',
            'meta[property="article:published_time"]',
            'meta[name="publish_date"]',
          ]),
          html,
        };
      });

      if (!result.html?.trim()) {
        throw new Error('Failed to extract article content.');
      }

      return { ...result, baseUrl: url };
    },
  },
  {
    name: 'medium',
    match: (url) => /medium\.com/.test(url),
    extract: async (page, url) => {
      await page.goto(url, { waitUntil: 'networkidle', timeout: 120000 });
      await page.waitForSelector('article', { timeout: 30000 });

      const origin = new URL(url).origin;
      const result = await page.evaluate(() => {
        const pickMeta = (selectors: string[]) => {
          for (const sel of selectors) {
            const meta = document.querySelector(sel) as HTMLMetaElement | null;
            const content = meta?.getAttribute('content')?.trim();
            if (content) return content;
          }
          return '';
        };

        const article = document.querySelector('article');
        return {
          title: document.querySelector('h1')?.textContent?.trim() || document.title || 'Medium Article',
          author:
            pickMeta(['meta[name="author"]']) ||
            document.querySelector('a[rel="author"]')?.textContent?.trim() ||
            '',
          published: pickMeta([
            'meta[property="article:published_time"]',
            'meta[name="publish_date"]',
            'meta[name="date"]',
          ]),
          html: (article as HTMLElement | null)?.innerHTML || '',
        };
      });

      if (!result.html?.trim()) {
        throw new Error('Failed to extract article content.');
      }

      return { ...result, baseUrl: origin };
    },
  },
  {
    name: 'wechat',
    match: (url) => /mp\.weixin\.qq\.com/.test(url),
    extract: async (page, url) => {
      await page.goto(url, { waitUntil: 'networkidle', timeout: 120000 });
      await page.waitForSelector('#js_content, .rich_media_content', { timeout: 30000 });

      const result = await page.evaluate(() => {
        const content =
          (document.querySelector('#js_content') as HTMLElement | null) ||
          (document.querySelector('.rich_media_content') as HTMLElement | null);
        return {
          title:
            document.querySelector('#activity-name')?.textContent?.trim() ||
            document.querySelector('h1')?.textContent?.trim() ||
            document.title ||
            'WeChat Article',
          author:
            document.querySelector('#js_name')?.textContent?.trim() ||
            document.querySelector('.profile_nickname')?.textContent?.trim() ||
            '',
          published: document.querySelector('#publish_time')?.textContent?.trim() || '',
          html: content?.innerHTML || '',
        };
      });

      if (!result.html?.trim()) {
        throw new Error('Failed to extract article content.');
      }

      return { ...result, baseUrl: 'https://mp.weixin.qq.com' };
    },
  },
];

async function fetchArticle(provider: Provider, url: string) {
  return withBrowser(async (context) => {
    const page = await context.newPage();
    return provider.extract(page, url);
  });
}

async function main() {
  const targetUrl = parseArgs();
  const provider =
    providers.find((p) => p.match(targetUrl)) ||
    providers.find((p) => p.name === 'zhihu');

  if (!provider) {
    throw new Error('No provider matched the given URL.');
  }

  const contentDir = path.join(CONTENT_ROOT, provider.name);
  const imageRoot = path.join(IMAGE_ROOT, provider.name);
  fs.mkdirSync(contentDir, { recursive: true });
  fs.mkdirSync(imageRoot, { recursive: true });

  const { title, author, published, html, baseUrl } = await fetchArticle(provider, targetUrl);

  const slugFromTitle = slugify(title, { lower: true, strict: true });
  const fallbackSlug = slugify(new URL(targetUrl).pathname.split('/').filter(Boolean).pop() || '', {
    lower: true,
    strict: true,
  });
  const slug = slugFromTitle || fallbackSlug || `${provider.name}-${Date.now()}`;

  const { markdown, images } = await htmlToMdx(html, {
    slug,
    provider: provider.name,
    baseUrl: baseUrl || targetUrl,
    imageRoot,
  });

  const publishedDate = published ? new Date(published) : new Date();
  const safeDate = Number.isNaN(publishedDate.valueOf()) ? new Date() : publishedDate;

  const frontmatter: Record<string, any> = {
    title: title || 'Imported Article',
    date: safeDate.toISOString().split('T')[0],
    tags: [provider.name],
    status: 'published',
    source: provider.name,
    source_url: targetUrl,
    source_author: author || provider.name,
    imported_at: new Date().toISOString(),
  };

  if (images[0]) {
    frontmatter.cover = images[0];
  }

  const fileContent = matter.stringify(markdown, frontmatter);
  const filepath = path.join(contentDir, `${slug}.mdx`);
  fs.writeFileSync(filepath, fileContent);

  console.log(`Saved article to ${filepath}`);
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
