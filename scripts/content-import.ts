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
import { fileURLToPath, pathToFileURL } from 'url';
import crypto from 'crypto';
import sharp from 'sharp';
import { JSDOM } from 'jsdom';
import readline from 'readline';

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
  updated?: string;
  html: string;
  baseUrl?: string;
  sourceTitle?: string;
  sourceUrl?: string;
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

// Image download constants
const IMAGE_MAX_RETRIES = 5;
const IMAGE_BASE_BACKOFF_MS = 500;
const IMAGE_MAX_BACKOFF_MS = 8000;
const WECHAT_REQUEST_DELAY_MIN_MS = 150;
const WECHAT_REQUEST_DELAY_MAX_MS = 400;
const WECHAT_CONCURRENCY_LIMIT = 2;
const WECHAT_PLACEHOLDER_MIN_WIDTH = 200; // WeChat placeholder is typically small
const WECHAT_PLACEHOLDER_MIN_HEIGHT = 150;

const MIME_TYPE_EXTENSION_MAP: Record<string, string> = {
  'image/jpeg': '.jpg',
  'image/jpg': '.jpg',
  'image/png': '.png',
  'image/webp': '.webp',
  'image/gif': '.gif',
  'image/bmp': '.bmp',
  'image/avif': '.avif',
  'image/svg+xml': '.svg',
};

const WECHAT_IMAGE_HEADERS = {
  'User-Agent':
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
  Referer: 'https://mp.weixin.qq.com/',
  Origin: 'https://mp.weixin.qq.com',
  Accept: 'image/avif,image/webp,image/apng,image/*,*/*;q=0.8',
  'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
  'Cache-Control': 'no-cache',
  Pragma: 'no-cache',
};

const DEFAULT_IMAGE_HEADERS = {
  'User-Agent':
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
  Referer: 'https://www.zhihu.com',
};

type ProviderConfig = {
  headers: Record<string, string>;
  placeholderGuard?: (buffer: Buffer, contentType: string) => Promise<boolean>;
  defaultExt?: string;
  maxRetries?: number;
  enableRateLimiting?: boolean;
  concurrencyLimit?: number;
  requestDelayMs?: { min: number; max: number };
};

const PROVIDER_CONFIGS: Record<string, ProviderConfig> = {
  wechat: {
    headers: WECHAT_IMAGE_HEADERS,
    placeholderGuard: isSuspectedWechatPlaceholder,
    defaultExt: '.jpg',
    maxRetries: IMAGE_MAX_RETRIES,
    enableRateLimiting: true,
    concurrencyLimit: WECHAT_CONCURRENCY_LIMIT,
    requestDelayMs: { min: WECHAT_REQUEST_DELAY_MIN_MS, max: WECHAT_REQUEST_DELAY_MAX_MS },
  },
  zhihu: {
    headers: DEFAULT_IMAGE_HEADERS,
    defaultExt: '.jpg',
    maxRetries: MAX_RETRIES,
  },
};

function getProviderConfig(provider: string): ProviderConfig {
  return (
    PROVIDER_CONFIGS[provider] || {
      headers: DEFAULT_IMAGE_HEADERS,
      defaultExt: '.jpg',
      maxRetries: MAX_RETRIES,
    }
  );
}

type ImportArgs = {
  url: string;
  force: boolean;
  dryRun: boolean;
  useFirstImageAsCover: boolean;
};

async function parseArgs(): Promise<ImportArgs> {
  const argUrl =
    process.argv.find((arg) => arg.startsWith('--url='))?.replace('--url=', '') ||
    process.env.URL ||
    process.env.url ||
    process.argv[2];

  const force = process.argv.includes('--force');
  const dryRun = process.argv.includes('--dry-run');
  const useFirstImageAsCover = process.argv.includes('--use-first-image-as-cover');
  let url = argUrl;

  if (!url) {
    try {
      const stdin = fs.readFileSync(0, 'utf8').trim();
      if (stdin) {
        url = stdin;
      }
    } catch {
      // ignore
    }
  }

  if (!url && process.stdin.isTTY) {
    url = await new Promise((resolve) => {
      const rl = readline.createInterface({ input: process.stdin, output: process.stdout });
      rl.question('Enter article URL: ', (answer) => {
        rl.close();
        resolve(answer.trim());
      });
    });
  }

  if (!url) {
    throw new Error('Usage: npm run import:content -- --url=<URL>');
  }

  return { url, force, dryRun, useFirstImageAsCover };
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

function pickFirstText(document: Document, selectors: string[]) {
  for (const selector of selectors) {
    const el = document.querySelector(selector);
    const text = el?.textContent?.trim();
    if (text) return text;
  }
  return '';
}

function pickFirstAttr(document: Document, selectors: string[], attr: string) {
  for (const selector of selectors) {
    const el = document.querySelector(selector) as HTMLElement | HTMLMetaElement | null;
    const value = el?.getAttribute(attr)?.trim();
    if (value) return value;
  }
  return '';
}

function removeNodes(root: Document | Element, selectors: string[]) {
  selectors.forEach((selector) => {
    root.querySelectorAll(selector).forEach((node) => node.remove());
  });
}

type ContentCandidate = { el: HTMLElement; score: number };

function selectMainContent(document: Document): HTMLElement | null {
  const candidates = [
    'article',
    'main article',
    'main',
    '[data-article]',
    '.article',
    '.article-body',
    '.post',
    '.post-content',
    '.blog-post',
    '.markdown',
    '.mdx',
    '.prose',
    '.content',
    '.entry-content',
  ];

  let best: ContentCandidate | null = null;

  const evaluate = (el: Element) => {
    const textLength = el.textContent?.trim().length || 0;
    if (textLength < MIN_CONTENT_LENGTH) return;
    if (!best || textLength > best.score) {
      best = { el: el as HTMLElement, score: textLength };
    }
  };

  for (const selector of candidates) {
    document.querySelectorAll(selector).forEach((el) => evaluate(el));
  }

  if (!best) {
    document.querySelectorAll('div').forEach((el) => evaluate(el));
  }

  const winner = best as ContentCandidate | null;
  return winner ? winner.el : null;
}

function stripNoise(root: Element) {
  const NOISE_SELECTORS = [
    'nav',
    'footer',
    'aside',
    '.toc',
    '#toc',
    '.table-of-contents',
    '.TableOfContents',
    '.toc-container',
    '.TableOfContents__root',
    '.breadcrumb',
    '.breadcrumbs',
    '.comment',
    '.comments',
    '#comments',
    '[data-component="comments"]',
    '.related',
    '.related-posts',
    '.post-navigation',
    '.next-post',
    '.prev-post',
    '.subscribe',
    '.newsletter',
    '.cookie',
    '.cookie-banner',
    '.share',
    '.social',
    '.ad',
    '.ads',
    '.advert',
    '.promo',
  ];
  removeNodes(root, NOISE_SELECTORS);
}

export function extractArticleFromHtml(html: string, url: string): ExtractedArticle {
  const dom = new JSDOM(html);
  const { document } = dom.window;

  removeNodes(document, ['script', 'style', 'noscript', 'template']);
  const sourceUrl = new URL(url);
  const title =
    pickFirstText(document, ['main h1', 'article h1', 'h1']) ||
    pickFirstAttr(document, ['meta[property="og:title"]', 'meta[name="title"]'], 'content') ||
    document.title ||
    sourceUrl.hostname;

  const author =
    pickFirstAttr(
      document,
      ['meta[name="author"]', 'meta[property="article:author"]'],
      'content',
    ) || pickFirstText(document, ['[rel="author"]', '.author', '.post-author', '.byline']);

  const published =
    pickFirstAttr(
      document,
      [
        'meta[property="article:published_time"]',
        'meta[name="publish_date"]',
        'meta[name="date"]',
        'time[datetime]',
      ],
      'content',
    ) || pickFirstAttr(document, ['time[datetime]'], 'datetime');

  const updated =
    pickFirstAttr(
      document,
      ['meta[property="article:modified_time"]', 'meta[name="lastmod"]'],
      'content',
    ) || pickFirstAttr(document, ['time[datetime][itemprop="dateModified"]'], 'datetime');

  const main = selectMainContent(document) || document.body;
  stripNoise(main);

  return {
    title,
    author,
    published,
    updated,
    html: main.innerHTML,
    baseUrl: sourceUrl.origin,
    sourceTitle: sourceUrl.hostname,
    sourceUrl: sourceUrl.toString(),
  };
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
  const pickFromSrcset = (value?: string) => {
    if (!value) return '';
    const candidates = value
      .split(',')
      .map((part) => part.trim())
      .map((part) => {
        const [urlPart, size] = part.split(/\s+/);
        const width = parseInt(size?.replace('w', '') || '0', 10);
        return { url: urlPart, width };
      })
      .filter((item) => item.url);
    if (!candidates.length) return '';
    candidates.sort((a, b) => b.width - a.width);
    return candidates[0].url;
  };
  // Priority order for WeChat lazy-loaded images:
  // 1. data-src (primary lazy-load attribute)
  // 2. data-original (alternative lazy-load attribute)
  // 3. data-backup-src (backup image URL)
  // 4. src (fallback for already-loaded images)
  const candidates = [
    props['data-src'],
    props['data-original'],
    props['data-backup-src'],
    props['data-actualsrc'],
    props.src,
    props['data-actual-url'],
  ];
  let url = candidates.find((u) => typeof u === 'string' && u.trim().length > 0);

  if (!url && typeof props.srcset === 'string') {
    url = pickFromSrcset(props.srcset);
  }
  if (!url && typeof props['data-srcset'] === 'string') {
    url = pickFromSrcset(props['data-srcset'] as string);
  }

  // Filter out data URLs (base64 encoded images) and empty URLs
  if (!url || url.startsWith('data:')) return '';
  return normalizeUrl(url, base);
}

async function isSuspectedWechatPlaceholder(buffer: Buffer, contentType: string): Promise<boolean> {
  const normalizedContentType = contentType.toLowerCase();
  const contentTypeIsImage = normalizedContentType.startsWith('image/');

  // First check: content-type must be an image
  if (!contentTypeIsImage) {
    console.warn(`Non-image content-type detected: ${contentType}`);
    return true;
  }

  // Second check: empty buffer
  if (buffer.length === 0) {
    console.warn('Empty image buffer detected');
    return true;
  }

  // Third check: size threshold (WeChat placeholder is typically < 60KB)
  if (buffer.length < WECHAT_PLACEHOLDER_THRESHOLD) {
    console.warn(`Small image detected (${buffer.length} bytes), checking dimensions...`);

    // Fourth check: use sharp to validate image dimensions
    try {
      const metadata = await sharp(buffer).metadata();
      const width = metadata.width || 0;
      const height = metadata.height || 0;

      if (width < WECHAT_PLACEHOLDER_MIN_WIDTH || height < WECHAT_PLACEHOLDER_MIN_HEIGHT) {
        console.warn(`Image dimensions too small (${width}x${height}), suspected placeholder`);
        return true;
      }

      // If size is small but dimensions are reasonable, it might be a valid small image
      console.log(
        `Image is small (${buffer.length} bytes) but has valid dimensions (${width}x${height})`,
      );
      return false;
    } catch (error) {
      console.warn(`Failed to read image metadata, treating as placeholder: ${error}`);
      return true;
    }
  }

  return false;
}

/**
 * Sleep for a random duration within the specified range
 */
async function randomDelay(minMs: number, maxMs: number): Promise<void> {
  const delayMs = Math.floor(Math.random() * (maxMs - minMs + 1)) + minMs;
  await new Promise((resolve) => setTimeout(resolve, delayMs));
}

/**
 * Calculate exponential backoff with jitter
 */
function calculateBackoff(attempt: number, baseMs: number, maxMs: number): number {
  const exponentialMs = baseMs * Math.pow(2, attempt - 1);
  const jitter = Math.random() * 0.3 * exponentialMs; // 30% jitter
  return Math.min(exponentialMs + jitter, maxMs);
}

/**
 * Semaphore for limiting concurrent operations
 */
class Semaphore {
  private permits: number;
  private queue: Array<() => void> = [];

  constructor(permits: number) {
    this.permits = permits;
  }

  async acquire(): Promise<void> {
    if (this.permits > 0) {
      this.permits--;
      return;
    }
    return new Promise((resolve) => {
      this.queue.push(resolve);
    });
  }

  release(): void {
    const nextResolve = this.queue.shift();
    if (nextResolve) {
      nextResolve();
    } else {
      this.permits++;
    }
  }
}

// Global semaphore for WeChat image downloads
const wechatImageSemaphore = new Semaphore(WECHAT_CONCURRENCY_LIMIT);

/**
 * Download image via HTTP with retry logic
 */
async function downloadImageViaHttp(
  url: string,
  config: ProviderConfig,
  provider: string,
): Promise<{ buffer: Buffer; contentType: string } | null> {
  const maxRetries = config.maxRetries || MAX_RETRIES;

  for (let attempt = 1; attempt <= maxRetries; attempt++) {
    try {
      // Rate limiting for WeChat
      if (config.enableRateLimiting && config.requestDelayMs) {
        await randomDelay(config.requestDelayMs.min, config.requestDelayMs.max);
      }

      const res = await fetch(url, {
        headers: config.headers,
        signal: AbortSignal.timeout(30000), // 30s timeout
      });

      const contentType = res.headers.get('content-type') || '';
      const normalizedContentType = contentType.toLowerCase();

      // Strict validation: response must be OK
      if (!res.ok) {
        const preview = await res
          .text()
          .then((text) => text.substring(0, 200))
          .catch(() => '[Unable to read response body]');
        console.warn(
          `[${provider}] HTTP ${res.status} for ${url}, content-type: ${contentType}, preview: ${preview}`,
        );

        // Retry on 429 (rate limit) or 5xx (server errors)
        if (res.status === 429 || res.status >= 500) {
          if (attempt < maxRetries) {
            const backoffMs = calculateBackoff(
              attempt,
              IMAGE_BASE_BACKOFF_MS,
              IMAGE_MAX_BACKOFF_MS,
            );
            console.log(
              `[${provider}] Retrying after ${backoffMs}ms (attempt ${attempt + 1}/${maxRetries})...`,
            );
            await new Promise((resolve) => setTimeout(resolve, backoffMs));
            continue;
          }
        }
        return null;
      }

      // Strict validation: content-type must be image/*
      if (!normalizedContentType.startsWith('image/')) {
        console.warn(
          `[${provider}] Non-image content-type: ${contentType} for ${url}, treating as failure`,
        );
        return null;
      }

      const buffer = Buffer.from(await res.arrayBuffer());
      console.log(
        `[${provider}] Downloaded ${buffer.length} bytes from ${url} (content-type: ${contentType})`,
      );
      return { buffer, contentType };
    } catch (error: any) {
      console.warn(
        `[${provider}] Download attempt ${attempt}/${maxRetries} failed for ${url}: ${error.message}`,
      );

      // Retry on network errors
      if (attempt < maxRetries) {
        const backoffMs = calculateBackoff(attempt, IMAGE_BASE_BACKOFF_MS, IMAGE_MAX_BACKOFF_MS);
        console.log(`[${provider}] Retrying after ${backoffMs}ms...`);
        await new Promise((resolve) => setTimeout(resolve, backoffMs));
        continue;
      }
    }
  }

  console.error(`[${provider}] All ${maxRetries} download attempts failed for ${url}`);
  return null;
}

/**
 * Download WeChat image via Playwright fallback
 * This is called when HTTP download fails or returns a placeholder
 */
async function downloadWechatImageViaPlaywright(
  articleUrl: string,
  imageUrl: string,
): Promise<Buffer | null> {
  console.log(
    `[wechat] Attempting Playwright fallback for image: ${imageUrl} from article: ${articleUrl}`,
  );

  try {
    return await withBrowser(async (context) => {
      const page = await context.newPage();
      let capturedBuffer: Buffer | null = null;

      // Listen for image responses from mmbiz.qpic.cn
      page.on('response', async (response) => {
        try {
          const respUrl = response.url();
          if (
            respUrl.includes('mmbiz.qpic.cn') &&
            response.request().resourceType() === 'image' &&
            respUrl.includes(new URL(imageUrl).pathname.split('/').pop() || '')
          ) {
            console.log(`[wechat] Captured image response: ${respUrl}`);
            capturedBuffer = Buffer.from(await response.body());
          }
        } catch (error) {
          console.warn(`[wechat] Failed to capture response body: ${error}`);
        }
      });

      // Navigate to the article with proper referer
      await page.goto(articleUrl, {
        waitUntil: 'networkidle',
        timeout: 60000,
        referer: 'https://mp.weixin.qq.com/',
      });

      // Wait a bit for images to load
      await page.waitForTimeout(3000);

      if (capturedBuffer) {
        console.log(
          `[wechat] Successfully captured ${(capturedBuffer as Buffer).length} bytes via Playwright`,
        );
        return capturedBuffer as Buffer;
      }

      console.warn('[wechat] Playwright fallback did not capture the image');
      return null;
    });
  } catch (error) {
    console.error(`[wechat] Playwright fallback failed: ${error}`);
    return null;
  }
}

async function downloadImage(
  url: string,
  provider: string,
  slug: string,
  imageRoot: string,
  index: number,
  articleUrl?: string,
  publicBasePath?: string,
): Promise<string | null> {
  const finalUrl = normalizeUrl(url);
  if (!finalUrl) {
    console.warn(`[${provider}] Invalid image URL: ${url}`);
    return null;
  }

  const config = getProviderConfig(provider);
  const { placeholderGuard, defaultExt = '.jpg' } = config;

  // Generate stable filename using hash
  const urlHash = crypto.createHash('md5').update(finalUrl).digest('hex').substring(0, 8);
  const filenameBase = `${String(index + 1).padStart(3, '0')}-${urlHash}`;
  const dir = path.join(imageRoot, slug);
  fs.mkdirSync(dir, { recursive: true });
  const publicBase = publicBasePath || `/images/${provider}/${slug}`;

  // Check if image already exists with any extension
  const existingFiles = fs.existsSync(dir) ? fs.readdirSync(dir) : [];
  const existingFile = existingFiles.find((f) => f.startsWith(filenameBase));
  if (existingFile) {
    console.log(`[${provider}] Image already exists: ${existingFile}`);
    return path.posix.join(publicBase, existingFile);
  }

  // Acquire semaphore for rate-limited providers
  let semaphoreAcquired = false;
  if (config.enableRateLimiting) {
    await wechatImageSemaphore.acquire();
    semaphoreAcquired = true;
  }

  try {
    // Attempt HTTP download
    const httpResult = await downloadImageViaHttp(finalUrl, config, provider);

    if (httpResult) {
      const { buffer, contentType } = httpResult;

      // Check for placeholder (WeChat-specific)
      if (placeholderGuard) {
        const isPlaceholder = await placeholderGuard(buffer, contentType);
        if (isPlaceholder) {
          console.warn(
            `[${provider}] Placeholder detected for ${finalUrl} (size=${buffer.length} bytes, content-type=${contentType})`,
          );

          // Try Playwright fallback for WeChat
          if (provider === 'wechat' && articleUrl) {
            console.log(`[${provider}] Attempting Playwright fallback...`);
            const playwrightBuffer = await downloadWechatImageViaPlaywright(articleUrl, finalUrl);

            if (playwrightBuffer) {
              // Re-check the playwright buffer
              const playwrightContentType = 'image/jpeg'; // Assume JPEG from Playwright
              const isStillPlaceholder = await placeholderGuard(
                playwrightBuffer,
                playwrightContentType,
              );

              if (!isStillPlaceholder) {
                // Success! Save the Playwright buffer
                const ext = defaultExt;
                const filename = `${filenameBase}${ext}`;
                const localPath = path.join(dir, filename);
                fs.writeFileSync(localPath, playwrightBuffer);
                console.log(`[${provider}] Saved via Playwright: ${localPath}`);
                return path.posix.join(publicBase, filename);
              } else {
                console.warn(
                  `[${provider}] Playwright fallback also returned placeholder for ${finalUrl}`,
                );
              }
            }
          }

          return null;
        }
      }

      // Determine extension from content-type
      const mimeType = (contentType.split(';')[0] || '').trim().toLowerCase();
      const extFromMime = MIME_TYPE_EXTENSION_MAP[mimeType];
      const extFromUrl = path.extname(new URL(finalUrl).pathname).split('?')[0];
      const ext = extFromMime || extFromUrl || defaultExt;

      const filename = `${filenameBase}${ext}`;
      const localPath = path.join(dir, filename);

      fs.writeFileSync(localPath, buffer);
      console.log(`[${provider}] Successfully saved: ${localPath}`);
      return path.posix.join(publicBase, filename);
    }

    // HTTP download failed, try Playwright fallback for WeChat
    if (provider === 'wechat' && articleUrl) {
      console.log(`[${provider}] HTTP download failed, attempting Playwright fallback...`);
      const playwrightBuffer = await downloadWechatImageViaPlaywright(articleUrl, finalUrl);

      if (playwrightBuffer) {
        // Check if Playwright buffer is valid
        if (placeholderGuard) {
          const isPlaceholder = await placeholderGuard(playwrightBuffer, 'image/jpeg');
          if (isPlaceholder) {
            console.warn(`[${provider}] Playwright fallback returned placeholder for ${finalUrl}`);
            return null;
          }
        }

        const ext = defaultExt;
        const filename = `${filenameBase}${ext}`;
        const localPath = path.join(dir, filename);
        fs.writeFileSync(localPath, playwrightBuffer);
        console.log(`[${provider}] Saved via Playwright fallback: ${localPath}`);
        return path.posix.join(publicBase, filename);
      }
    }

    console.error(`[${provider}] All download methods failed for ${finalUrl}`);
    return null;
  } catch (error) {
    console.error(`[${provider}] Unexpected error downloading ${finalUrl}:`, error);
    return null;
  } finally {
    if (semaphoreAcquired) {
      wechatImageSemaphore.release();
    }
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
    const extractAnnotation = (target: HastElement): string | null => {
      if (target.tagName === 'annotation' && target.properties?.encoding === 'application/x-tex') {
        const firstChild = target.children?.[0];
        if (firstChild && typeof firstChild.value === 'string') {
          return firstChild.value;
        }
      }
      if (target.children?.length) {
        for (const child of target.children) {
          const result = extractAnnotation(child);
          if (result) return result;
        }
      }
      return null;
    };

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

    if (hasClass(node, 'katex') || hasClass(node, 'katex-display')) {
      const latex = extractAnnotation(node);
      if (latex) {
        const display = hasClass(node, 'katex-display');
        const value = display ? `$$${latex.trim()}$$` : `$${latex.trim()}$`;
        parent.children.splice(idx, 1, { type: 'text', value });
      }
    }
  });
};

const localizeImages = (options: {
  slug: string;
  baseUrl?: string;
  collected: string[];
  provider: string;
  imageRoot: string;
  articleUrl?: string;
  publicBasePath?: string;
  downloadImage?: typeof downloadImage;
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
      const downloader = options.downloadImage ?? downloadImage;
      const publicBasePath =
        options.publicBasePath || `/images/${options.provider}/${options.slug}`;
      let index = 0;
      for (const { url } of imageNodes) {
        if (mapping.has(url)) continue;
        const local = await downloader(
          url,
          options.provider,
          options.slug,
          options.imageRoot,
          index,
          options.articleUrl,
          publicBasePath,
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

export async function htmlToMdx(
  html: string,
  options: {
    slug: string;
    provider: string;
    baseUrl?: string;
    imageRoot: string;
    articleUrl?: string;
    publicBasePath?: string;
    downloadImage?: typeof downloadImage;
  },
) {
  const images: string[] = [];
  const imagePlugin = localizeImages({
    slug: options.slug,
    baseUrl: options.baseUrl,
    collected: images,
    provider: options.provider,
    imageRoot: options.imageRoot,
    articleUrl: options.articleUrl,
    publicBasePath: options.publicBasePath,
    downloadImage: options.downloadImage,
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

  let markdown = String(file).trim();
  markdown = markdown.replace(/\\\$\\\$/g, '$$').replace(/\\\$/g, '$');

  return { markdown, images };
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

      // Fix lazy-loaded images: replace data:image/svg+xml placeholders with actual URLs
      // This ensures real image URLs (https://mmbiz.qpic.cn/...) are captured in the HTML
      await page.evaluate(() => {
        const root =
          document.querySelector('#js_content') || document.querySelector('.rich_media_content');
        if (root) {
          const images = root.querySelectorAll('img');
          for (const img of images) {
            // Priority order for WeChat lazy-loaded images
            const realUrl =
              img.getAttribute('data-src') ||
              img.getAttribute('data-original') ||
              img.getAttribute('data-backup-src') ||
              img.getAttribute('data-actualsrc') ||
              img.getAttribute('data-actual-url');

            // Replace placeholder with actual URL if found
            if (realUrl && /^https?:\/\//i.test(realUrl)) {
              img.setAttribute('src', realUrl);
              // Clean up lazy-load attributes to avoid confusion
              img.removeAttribute('data-src');
              img.removeAttribute('data-original');
              img.removeAttribute('data-backup-src');
            }
          }
        }
      });

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
  {
    name: 'imported',
    match: () => true,
    extract: async (page, url) => {
      await page.goto(url, { waitUntil: 'networkidle', timeout: 120000 });
      const content = await page.content();
      const article = extractArticleFromHtml(content, url);

      if (!article.html?.trim()) {
        throw new Error('Failed to extract article content.');
      }

      return {
        title: article.title,
        author: article.author,
        published: article.published,
        updated: article.updated,
        html: article.html,
        baseUrl: article.baseUrl || new URL(url).origin,
        sourceTitle: article.sourceTitle,
        sourceUrl: article.sourceUrl,
      };
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
  const options = await parseArgs();
  const targetUrl = options.url;
  const provider = providers.find((p) => p.match(targetUrl));

  if (!provider) {
    throw new Error('No provider matched the given URL.');
  }

  const contentDir = path.join(CONTENT_ROOT, provider.name);
  const imageRoot = path.join(IMAGE_ROOT, provider.name);
  fs.mkdirSync(contentDir, { recursive: true });
  fs.mkdirSync(imageRoot, { recursive: true });

  const { title, author, published, updated, html, baseUrl, sourceTitle } = await fetchArticle(
    provider,
    targetUrl,
  );

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
    articleUrl: targetUrl, // Pass article URL for Playwright fallback
    publicBasePath: `/images/${provider.name}/${slug}`,
    downloadImage: options.dryRun ? async () => null : undefined,
  });

  // Safety check: Ensure no WeChat lazy-load placeholders in final markdown
  // This prevents publishing articles with broken image placeholders
  if (provider.name === 'wechat' && /data:image\/svg\+xml/i.test(markdown)) {
    throw new Error(
      'WeChat image placeholder detected (data:image/svg+xml) in generated markdown. ' +
        'Image extraction failed - real image URLs were not properly captured from lazy-loaded attributes.',
    );
  }

  const publishedDate = published ? new Date(published) : new Date();
  const safeDate = Number.isNaN(publishedDate.valueOf()) ? new Date() : publishedDate;
  const updatedDate = updated ? new Date(updated) : safeDate;
  const safeUpdatedDate = Number.isNaN(updatedDate.valueOf()) ? safeDate : updatedDate;

  const frontmatter: Record<string, any> = {
    title: title || 'Imported Article',
    slug: slug,
    date: safeDate.toISOString().split('T')[0],
    updated: safeUpdatedDate ? safeUpdatedDate.toISOString().split('T')[0] : undefined,
    tags: [],
    status: 'published',
    source_url: targetUrl,
    source_author: author || provider.name,
    imported_at: new Date().toISOString(),
    source: {
      title: sourceTitle || new URL(targetUrl).hostname,
      url: targetUrl,
    },
  };

  if (options.useFirstImageAsCover && images[0]) {
    frontmatter.cover = images[0];
  }

  const fileContent = matter.stringify(markdown, frontmatter);
  const filepath = path.join(contentDir, `${slug}.mdx`);

  if (fs.existsSync(filepath) && !options.force) {
    console.log(`File already exists at ${filepath}. Use --force to overwrite.`);
    return;
  }

  if (options.dryRun) {
    console.log('Dry run mode enabled. Preview frontmatter + markdown:\n');
    console.log(fileContent);
    return;
  }

  fs.writeFileSync(filepath, fileContent);

  console.log(`Saved article to ${filepath}`);
}

if (import.meta.url === pathToFileURL(process.argv[1]).href) {
  main().catch((error) => {
    console.error(error);
    process.exit(1);
  });
}
