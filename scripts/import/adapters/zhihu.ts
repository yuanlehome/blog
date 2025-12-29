/**
 * Zhihu Adapter
 *
 * Handles article import from Zhihu Column (zhuanlan.zhihu.com)
 */

import type { Adapter, Article, FetchArticleInput } from './types.js';
import { htmlToMdx } from '../../content-import.js';

// Constants for retry and timing
const MAX_RETRIES = 3;
const BASE_BACKOFF_MS = 1000;
const MAX_BACKOFF_MS = 10000;
const JS_INITIALIZATION_DELAY = 2000;
const MIN_CONTENT_LENGTH = 100;
const CONTENT_WAIT_TIMEOUT = 30000;

/**
 * Check if URL is from Zhihu domain
 */
function isFromDomain(url: string, domain: string): boolean {
  try {
    const urlObj = new URL(url);
    const hostname = urlObj.hostname.toLowerCase();
    return hostname === domain || hostname.endsWith(`.${domain}`);
  } catch {
    return false;
  }
}

/**
 * Sanitize Zhihu URL by removing tracking and share parameters
 */
function sanitizeZhihuUrl(url: string): string {
  try {
    const urlObj = new URL(url);
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
 * Detect if the page is a login/captcha/blocked page
 */
async function detectBlockedPage(page: any): Promise<string | null> {
  try {
    const pageContent = await page.content();
    const title = await page.title();
    const url = page.url();

    if (
      /登录|login|sign.?in/i.test(title) ||
      /登录|login|sign.?in/i.test(pageContent) ||
      url.includes('/signin') ||
      url.includes('/login')
    ) {
      return 'Zhihu blocked request (login page detected)';
    }

    if (
      /验证|captcha|security.?check|human.?verification/i.test(title) ||
      /验证|captcha|security.?check/i.test(pageContent)
    ) {
      return 'Zhihu blocked request (captcha/security check detected)';
    }

    if (/安全验证|反作弊|access.?denied/i.test(pageContent)) {
      return 'Zhihu blocked request (anti-spider protection)';
    }

    return null;
  } catch {
    return null;
  }
}

/**
 * Wait for content with retry strategy
 */
async function waitForContent(
  page: any,
  selectors: string[],
  options: { minTextLength?: number; timeout?: number } = {},
): Promise<void> {
  const { minTextLength = MIN_CONTENT_LENGTH, timeout = CONTENT_WAIT_TIMEOUT } = options;
  const startTime = Date.now();

  for (const selector of selectors) {
    try {
      await page.waitForSelector(selector, { state: 'attached', timeout: 5000 });

      await page.waitForFunction(
        ({ sel, minLen }: { sel: string; minLen: number }) => {
          const el = document.querySelector(sel);
          if (!el) return false;
          const text = el.textContent?.trim() || '';
          return text.length >= minLen;
        },
        { sel: selector, minLen: minTextLength },
        { timeout: Math.max(5000, timeout - (Date.now() - startTime)) },
      );

      return;
    } catch {
      continue;
    }
  }

  throw new Error(
    `Zhihu DOM structure changed: None of the expected content selectors found: ${selectors.join(', ')}`,
  );
}

/**
 * Enhanced Zhihu extraction with retry logic
 */
async function extractZhihuWithRetry(
  page: any,
  url: string,
  maxRetries = MAX_RETRIES,
): Promise<{ title: string; author: string; published: string; html: string }> {
  const sanitizedUrl = sanitizeZhihuUrl(url);
  let lastError: Error | null = null;

  for (let attempt = 1; attempt <= maxRetries; attempt++) {
    try {
      console.log(`Attempt ${attempt}/${maxRetries} to extract from ${sanitizedUrl}`);

      await page.goto(sanitizedUrl, {
        waitUntil: 'domcontentloaded',
        timeout: 60000,
      });

      await page.waitForTimeout(JS_INITIALIZATION_DELAY);

      const blockReason = await detectBlockedPage(page);
      if (blockReason) {
        throw new Error(blockReason);
      }

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
      return result;
    } catch (error) {
      lastError = error as Error;
      console.error(`Attempt ${attempt} failed:`, error);

      if (attempt < maxRetries) {
        const backoffMs = Math.min(BASE_BACKOFF_MS * Math.pow(2, attempt - 1), MAX_BACKOFF_MS);
        console.log(`Waiting ${backoffMs}ms before retry...`);
        await page.waitForTimeout(backoffMs);
      }
    }
  }

  throw lastError || new Error('Failed to extract Zhihu article after all retries');
}

/**
 * Zhihu adapter implementation
 */
export const zhihuAdapter: Adapter = {
  id: 'zhihu',
  name: 'Zhihu Column',

  canHandle(url: string): boolean {
    return isFromDomain(url, 'zhihu.com') && url.includes('zhuanlan.zhihu.com/p/');
  },

  async fetchArticle(input: FetchArticleInput): Promise<Article> {
    const { url, page, options = {} } = input;
    const { slug = 'zhihu-article', imageRoot = '/tmp/images', publicBasePath } = options;

    // Extract article with retry logic
    const result = await extractZhihuWithRetry(page, url);

    // Convert HTML to Markdown
    const { markdown, images } = await htmlToMdx(result.html, {
      slug,
      provider: 'zhihu',
      baseUrl: sanitizeZhihuUrl(url),
      imageRoot,
      articleUrl: url,
      publicBasePath: publicBasePath || `/images/zhihu/${slug}`,
      downloadImage: options.downloadImage,
    });

    return {
      title: result.title,
      markdown,
      canonicalUrl: sanitizeZhihuUrl(url),
      source: 'zhihu',
      author: result.author,
      publishedAt: result.published || undefined,
      images: images.map((localPath) => ({ url: '', localPath })),
    };
  },
};
