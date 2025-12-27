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
import { unified, type Plugin } from 'unified';
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
const ARTIFACTS_ROOT = path.join(__dirname, '../artifacts');

// Constants for retry and timing
const MAX_RETRIES = 3;
const BASE_BACKOFF_MS = 1000;
const MAX_BACKOFF_MS = 10000;
const JS_INITIALIZATION_DELAY = 2000;
const MIN_CONTENT_LENGTH = 100;
const CONTENT_WAIT_TIMEOUT = 30000;
const MAX_URL_LENGTH_FOR_FILENAME = 50;
const WECHAT_PLACEHOLDER_THRESHOLD = 60 * 1024; // ~60KB placeholder guard

const MIME_TYPE_EXTENSION_MAP: Record<string, string> = {
  'image/jpeg': '.jpg',
  'image/jpg': '.jpg',
  'image/png': '.png',
  'image/webp': '.webp',
  'image/gif': '.gif',
  'image/bmp': '.bmp',
};

function parseArgs(): string {
  const url =
    process.argv.find((arg) => arg.startsWith('--url='))?.replace('--url=', '') ||
    process.env.URL ||
    process.env.url ||
    process.argv[2];

  if (!url) {
    throw new Error('Usage: npm run import:content -- --url=<URL>');
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

/**
 * Sanitize Zhihu URL by removing tracking and share parameters
 */
function sanitizeZhihuUrl(url: string): string {
  try {
    const urlObj = new URL(url);
    // Remove common tracking and share parameters
    const paramsToRemove = [
      'share_code',
      'utm_source',
      'utm_medium',
      'utm_campaign',
      'utm_content',
      'utm_term',
      'utm_psn',
      'utm_id',
      'utm_oi',
    ];
    paramsToRemove.forEach((param) => urlObj.searchParams.delete(param));
    return urlObj.toString();
  } catch {
    return url;
  }
}

/**
 * Check if a URL is from a specific domain (secure hostname validation)
 */
function isFromDomain(url: string, domain: string): boolean {
  try {
    const urlObj = new URL(url);
    const hostname = urlObj.hostname.toLowerCase();
    // Must be exact match or subdomain of the target domain
    return hostname === domain || hostname.endsWith(`.${domain}`);
  } catch {
    return false;
  }
}

/**
 * Save debug artifacts (screenshot, HTML, logs) when scraping fails
 */
async function saveDebugArtifacts(
  page: Page,
  url: string,
  error: Error,
  logs: { type: string; text: string; timestamp: number }[],
): Promise<void> {
  try {
    fs.mkdirSync(ARTIFACTS_ROOT, { recursive: true });
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    const safeUrl = url.replace(/[^a-zA-Z0-9]/g, '_').substring(0, MAX_URL_LENGTH_FOR_FILENAME);
    const prefix = `${timestamp}_${safeUrl}`;

    // Save screenshot
    const screenshotPath = path.join(ARTIFACTS_ROOT, `${prefix}_screenshot.png`);
    await page.screenshot({ path: screenshotPath, fullPage: true }).catch((e) => {
      console.warn(`Failed to save screenshot: ${e.message}`);
    });
    console.log(`Screenshot saved: ${screenshotPath}`);

    // Save HTML
    const htmlPath = path.join(ARTIFACTS_ROOT, `${prefix}_page.html`);
    const html = await page.content().catch(() => '<html>Failed to get page content</html>');
    fs.writeFileSync(htmlPath, html);
    console.log(`HTML saved: ${htmlPath}`);

    // Save logs
    const logsPath = path.join(ARTIFACTS_ROOT, `${prefix}_logs.json`);
    const logData = {
      url,
      error: {
        message: error.message,
        stack: error.stack,
      },
      timestamp: new Date().toISOString(),
      logs,
    };
    fs.writeFileSync(logsPath, JSON.stringify(logData, null, 2));
    console.log(`Logs saved: ${logsPath}`);
  } catch (artifactError) {
    console.error('Failed to save debug artifacts:', artifactError);
  }
}

/**
 * Detect if the page is a login/captcha/blocked page
 */
async function detectBlockedPage(page: Page): Promise<string | null> {
  try {
    const pageContent = await page.content();
    const title = await page.title();
    const url = page.url();

    // Check for login page
    if (
      /登录|login|sign.?in/i.test(title) ||
      /登录|login|sign.?in/i.test(pageContent) ||
      url.includes('/signin') ||
      url.includes('/login')
    ) {
      return 'Zhihu blocked request (login page detected)';
    }

    // Check for captcha/verification page
    if (
      /验证|captcha|security.?check|human.?verification/i.test(title) ||
      /验证|captcha|security.?check/i.test(pageContent)
    ) {
      return 'Zhihu blocked request (captcha/security check detected)';
    }

    // Check for anti-spider page
    if (/安全验证|反作弊|access.?denied/i.test(pageContent)) {
      return 'Zhihu blocked request (anti-spider protection)';
    }

    return null;
  } catch {
    return null;
  }
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
    const buildHeaders = (): Record<string, string> => {
      if (provider === 'wechat') {
        return {
          'User-Agent':
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
          Referer: 'https://mp.weixin.qq.com/',
          Accept: 'image/avif,image/webp,image/apng,image/*,*/*;q=0.8',
          'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
        };
      }
      return {
        'User-Agent':
          'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36',
        Referer: 'https://www.zhihu.com',
      };
    };

    const res = await fetch(finalUrl, { headers: buildHeaders() });

    if (!res.ok) {
      console.warn(`Failed to fetch image ${finalUrl}: ${res.status}`);
      return null;
    }

    const contentType = res.headers.get('content-type') || '';
    const buffer = Buffer.from(await res.arrayBuffer());

    if (provider === 'wechat') {
      const contentTypeIsImage = contentType.toLowerCase().startsWith('image/');
      const isPlaceholder =
        buffer.length === 0 || buffer.length < WECHAT_PLACEHOLDER_THRESHOLD || !contentTypeIsImage;
      if (isPlaceholder) {
        console.warn(
          `WeChat image suspected placeholder (size=${buffer.length} bytes, content-type=${contentType})`,
        );
        return null;
      }
    }

    const mimeType = (contentType.split(';')[0] || '').trim().toLowerCase();
    const extFromMime = MIME_TYPE_EXTENSION_MAP[mimeType];
    const ext = extFromUrl || extFromMime || '.jpg';
    const dir = path.join(imageRoot, slug);
    const filename = `${String(index + 1).padStart(3, '0')}${ext}`;
    const localPath = path.join(dir, filename);

    if (fs.existsSync(localPath)) {
      return `/images/${provider}/${slug}/${filename}`;
    }

    fs.mkdirSync(dir, { recursive: true });
    fs.writeFileSync(localPath, buffer);
    return `/images/${provider}/${slug}/${filename}`;
  } catch (error) {
    console.warn(`Failed to download image ${url}`, error);
    return null;
  }
}

/**
 * Playwright fallback design (manual use):
 * --------------------------------------
 * When direct HTTP download returns a suspected placeholder, launch Playwright,
 * open the WeChat article URL with a trusted Referer, listen for network
 * responses from mmbiz.qpic.cn, and persist the real response body:
 *
 * await withBrowser(async (context) => {
 *   const page = await context.newPage();
 *   let imageBody: Buffer | null = null;
 *   page.on('response', async (resp) => {
 *     if (resp.url().includes('mmbiz.qpic.cn') && resp.request().resourceType() === 'image') {
 *       imageBody = Buffer.from(await resp.body());
 *     }
 *   });
 *   await page.goto(articleUrl, { waitUntil: 'networkidle', referer: 'https://mp.weixin.qq.com/' });
 *   await page.waitForTimeout(2000);
 *   if (imageBody) fs.writeFileSync(targetPath, imageBody);
 * });
 *
 * This stays out of CI/build paths and can be opted in manually when needed.
 */

const transformMath: Plugin<[], any> = () => (tree: any) => {
  visit(tree, 'element', (node: HastElement, index: number | null | undefined, parent: any) => {
    const idx = typeof index === 'number' ? index : null;
    if (!parent || idx === null) return;
    if (node.tagName === 'span' && hasClass(node, 'ztext-math')) {
      const latex = (node.properties?.['data-tex'] as string) || getTextContent(node);
      const value = `$${latex.trim()}$`;
      parent.children.splice(idx, 1, { type: 'text', value });
    }
    if (node.tagName === 'div' && hasClass(node, 'ztext-math')) {
      const latex = (node.properties?.['data-tex'] as string) || getTextContent(node);
      const value = `$$${latex.trim()}$$`;
      parent.children.splice(idx, 1, { type: 'text', value });
    }
  });
};

const localizeImages = (options: {
  slug: string;
  baseUrl?: string;
  collected: string[];
  provider: string;
  imageRoot: string;
}): Plugin<[], any> => {
  return function plugin() {
    return async (tree: any) => {
      const imageNodes: { node: HastElement; url: string }[] = [];

      visit(tree, 'element', (node: HastElement) => {
        if (node.tagName !== 'img') return;
        const src = resolveImageSrc(node, options.baseUrl);
        if (!src) return;
        imageNodes.push({ node, url: src });
      });

      const mapping = new Map<string, string>();
      let index = 0;
      for (const { url } of imageNodes) {
        if (mapping.has(url)) continue;
        const local = await downloadImage(
          url,
          options.provider,
          options.slug,
          options.imageRoot,
          index,
        );
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
  };
};

async function htmlToMdx(
  html: string,
  options: { slug: string; provider: string; baseUrl?: string; imageRoot: string },
) {
  const images: string[] = [];
  const imagePlugin = localizeImages({
    slug: options.slug,
    baseUrl: options.baseUrl,
    collected: images,
    provider: options.provider,
    imageRoot: options.imageRoot,
  });
  const processor: any = unified();
  const file = await processor
    .use(rehypeParse, { fragment: true })
    .use(rehypeRaw)
    .use(transformMath)
    .use(imagePlugin)
    .use(rehypeRemark as any)
    .use(remarkMath)
    .use(remarkGfm)
    .use(remarkStringify, {
      fences: true,
      bullet: '-',
      rule: '-',
    } as any)
    .process(html);

  return { markdown: String(file).trim(), images };
}

async function withBrowser<T>(fn: (context: BrowserContext) => Promise<T>) {
  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext({
    userAgent:
      'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    locale: 'zh-CN',
    viewport: { width: 1920, height: 1080 },
    extraHTTPHeaders: {
      'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
    },
  });
  try {
    return await fn(context);
  } finally {
    await context.close();
    await browser.close();
  }
}

/**
 * Wait for content with three-phase strategy:
 * 1. Wait for selector to be attached to DOM
 * 2. Wait for content to have meaningful text
 */
async function waitForContent(
  page: Page,
  selectors: string[],
  options: { minTextLength?: number; timeout?: number } = {},
): Promise<void> {
  const { minTextLength = MIN_CONTENT_LENGTH, timeout = CONTENT_WAIT_TIMEOUT } = options;
  const startTime = Date.now();

  for (const selector of selectors) {
    try {
      // Phase 1: Wait for element to be attached (not necessarily visible)
      await page.waitForSelector(selector, { state: 'attached', timeout: 5000 });

      // Phase 2: Wait for element to have meaningful content
      await page.waitForFunction(
        ({ sel, minLen }) => {
          const el = document.querySelector(sel);
          if (!el) return false;
          const text = el.textContent?.trim() || '';
          return text.length >= minLen;
        },
        { sel: selector, minLen: minTextLength },
        { timeout: Math.max(5000, timeout - (Date.now() - startTime)) },
      );

      // Success!
      return;
    } catch (error) {
      // Try next selector
      continue;
    }
  }

  // All selectors failed
  throw new Error(
    `Zhihu DOM structure changed: None of the expected content selectors found: ${selectors.join(', ')}`,
  );
}

/**
 * Enhanced Zhihu extraction with retry logic and comprehensive error handling
 */
async function extractZhihuWithRetry(
  page: Page,
  url: string,
  maxRetries = MAX_RETRIES,
): Promise<ExtractedArticle> {
  // Sanitize URL first
  const sanitizedUrl = sanitizeZhihuUrl(url);
  console.log(`Sanitized URL: ${sanitizedUrl}`);

  const logs: { type: string; text: string; timestamp: number }[] = [];

  // Setup logging listeners
  page.on('console', (msg) => {
    logs.push({ type: 'console', text: msg.text(), timestamp: Date.now() });
  });
  page.on('pageerror', (error) => {
    logs.push({ type: 'error', text: error.message, timestamp: Date.now() });
  });
  page.on('response', (response) => {
    if (isFromDomain(response.url(), 'zhihu.com')) {
      logs.push({
        type: 'response',
        text: `${response.status()} ${response.url()}`,
        timestamp: Date.now(),
      });
    }
  });

  let lastError: Error | null = null;

  for (let attempt = 1; attempt <= maxRetries; attempt++) {
    try {
      console.log(`Attempt ${attempt}/${maxRetries} to extract from ${sanitizedUrl}`);

      // Phase 1: Navigate with domcontentloaded
      await page.goto(sanitizedUrl, {
        waitUntil: 'domcontentloaded',
        timeout: 60000,
      });

      // Small delay to let JS initialize
      await page.waitForTimeout(JS_INITIALIZATION_DELAY);

      // Check for blocked page
      const blockReason = await detectBlockedPage(page);
      if (blockReason) {
        throw new Error(blockReason);
      }

      // Phase 2 & 3: Wait for content with robust selectors
      const contentSelectors = [
        '.Post-RichText',
        '.RichText',
        'article',
        '.ztext',
        '[data-za-detail-view-element_name="Article"]',
        '.Post-Main .RichContent',
      ];

      await waitForContent(page, contentSelectors, {
        minTextLength: MIN_CONTENT_LENGTH,
        timeout: CONTENT_WAIT_TIMEOUT,
      });

      // Extract content
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

        const contentSelectors = [
          '.Post-RichText',
          '.RichText',
          'article',
          '.ztext',
          '[data-za-detail-view-element_name="Article"]',
          '.Post-Main .RichContent',
        ];
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
        throw new Error('Zhihu DOM structure changed: Failed to extract article content');
      }

      console.log(`Successfully extracted article: ${result.title}`);
      return { ...result, baseUrl: sanitizedUrl };
    } catch (error) {
      lastError = error as Error;
      console.error(`Attempt ${attempt} failed:`, error);

      // Save artifacts on last attempt
      if (attempt === maxRetries) {
        console.error('All retry attempts exhausted. Saving debug artifacts...');
        await saveDebugArtifacts(page, sanitizedUrl, lastError, logs);
      } else {
        // Exponential backoff: wait before retry
        const backoffMs = Math.min(BASE_BACKOFF_MS * Math.pow(2, attempt - 1), MAX_BACKOFF_MS);
        console.log(`Waiting ${backoffMs}ms before retry...`);
        await page.waitForTimeout(backoffMs);
      }
    }
  }

  // All retries failed
  throw lastError || new Error('Failed to extract Zhihu article after all retries');
}

const providers: Provider[] = [
  {
    name: 'zhihu',
    match: (url) => isFromDomain(url, 'zhihu.com'),
    extract: async (page, url) => {
      return extractZhihuWithRetry(page, url);
    },
  },
  {
    name: 'medium',
    match: (url) => isFromDomain(url, 'medium.com'),
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
          title:
            document.querySelector('h1')?.textContent?.trim() || document.title || 'Medium Article',
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
    match: (url) => isFromDomain(url, 'mp.weixin.qq.com'),
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
  const provider = providers.find((p) => p.match(targetUrl));

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
    slug: slug,
    date: safeDate.toISOString().split('T')[0],
    tags: [],
    status: 'published',
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
