/**
 * Zhihu Adapter
 *
 * Handles article import from Zhihu Column (zhuanlan.zhihu.com)
 */

import { JSDOM } from 'jsdom';
import type { Adapter, Article, FetchArticleFromHtmlInput, FetchArticleInput } from './types.js';
import { htmlToMdx } from '../../content-import.js';
import type { Logger } from '../../logger/types.js';
import { createLogger } from '../../logger/index.js';

// Constants for retry and timing
const MAX_RETRIES = 3;
const BASE_BACKOFF_MS = 1000;
const MAX_BACKOFF_MS = 10000;
const JS_INITIALIZATION_DELAY = 2000;
const MIN_CONTENT_LENGTH = 100;
const CONTENT_WAIT_TIMEOUT = 30000;
const CONTENT_SELECTORS = [
  '.Post-RichText',
  '.RichText',
  'article',
  '.ztext',
  '[data-za-detail-view-element_name="Article"]',
  '.Post-Main .RichContent',
];

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

type RawExtractedZhihuArticle = {
  title: string;
  author: string;
  published: string;
  html: string;
  keywords?: string;
};

type ExtractedZhihuArticle = Omit<RawExtractedZhihuArticle, 'keywords'> & {
  tags: string[];
};

function pickFirstText(document: Document, selectors: string[]): string {
  for (const selector of selectors) {
    const text = document.querySelector(selector)?.textContent?.trim();
    if (text) return text;
  }
  return '';
}

function pickFirstAttribute(document: Document, selectors: string[], attribute: string): string {
  for (const selector of selectors) {
    const value = document.querySelector(selector)?.getAttribute(attribute)?.trim();
    if (value) return value;
  }
  return '';
}

function parseZhihuKeywords(keywords?: string): string[] {
  const seen = new Set<string>();
  const tags: string[] = [];

  for (const keyword of (keywords || '').split(/[,，]/)) {
    const tag = keyword.trim();
    if (!tag || seen.has(tag)) continue;
    seen.add(tag);
    tags.push(tag);
  }

  return tags;
}

function formatDateInShanghai(date: Date): string {
  const parts = new Intl.DateTimeFormat('en-US', {
    timeZone: 'Asia/Shanghai',
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
  }).formatToParts(date);
  const values = Object.fromEntries(parts.map((part) => [part.type, part.value]));
  return `${values.year}-${values.month}-${values.day}`;
}

function normalizeZhihuPublishedDate(published: string): string {
  const value = published.trim();
  const datePrefix = value.match(/^(\d{4}-\d{2}-\d{2})(?:$|T)/)?.[1];
  if (!datePrefix) return value;
  if (value === datePrefix) return datePrefix;

  // An ISO value without an offset represents Zhihu's local wall-clock time.
  if (!/(?:Z|[+-]\d{2}:?\d{2})$/i.test(value)) {
    return datePrefix;
  }

  const parsed = new Date(value);
  return Number.isNaN(parsed.getTime()) ? value : formatDateInShanghai(parsed);
}

function isZhihuEntitySearchLink(href: string): boolean {
  try {
    const url = new URL(href, 'https://zhuanlan.zhihu.com');
    return (
      url.hostname.toLowerCase() === 'zhida.zhihu.com' &&
      url.pathname.replace(/\/+$/, '') === '/search' &&
      (url.searchParams.has('zd_token') ||
        url.searchParams.get('zhida_source')?.toLowerCase() === 'entity')
    );
  } catch {
    return false;
  }
}

/**
 * Remove short-lived Zhihu entity links and align in-article heading depth with
 * the blog title, which is rendered separately from frontmatter.
 */
function cleanZhihuContentHtml(html: string): string {
  const dom = new JSDOM('');
  const { document } = dom.window;
  const container = document.createElement('div');
  container.innerHTML = html;

  container
    .querySelectorAll('script, style, noscript, template')
    .forEach((element) => element.remove());

  container.querySelectorAll('a[href]').forEach((anchor) => {
    const href = anchor.getAttribute('href');
    if (href && isZhihuEntitySearchLink(href)) {
      anchor.replaceWith(...Array.from(anchor.childNodes));
    }
  });

  const headings = Array.from(container.querySelectorAll('h1, h2, h3, h4, h5, h6'));
  const topHeadingLevel = headings.reduce((minimum, heading) => {
    const level = Number(heading.tagName.slice(1));
    return Math.min(minimum, level);
  }, 7);

  if (topHeadingLevel > 2 && topHeadingLevel <= 6) {
    const offset = topHeadingLevel - 2;
    for (const heading of headings) {
      const level = Number(heading.tagName.slice(1));
      const replacement = document.createElement(`h${level - offset}`);
      for (const attribute of Array.from(heading.attributes)) {
        replacement.setAttribute(attribute.name, attribute.value);
      }
      replacement.append(...Array.from(heading.childNodes));
      heading.replaceWith(replacement);
    }
  }

  return container.innerHTML;
}

function normalizeExtractedZhihuArticle(article: RawExtractedZhihuArticle): ExtractedZhihuArticle {
  return {
    title: article.title,
    author: article.author,
    published: normalizeZhihuPublishedDate(article.published),
    html: cleanZhihuContentHtml(article.html),
    tags: parseZhihuKeywords(article.keywords),
  };
}

/**
 * Extract a Zhihu article from a complete page snapshot.
 *
 * This deliberately uses DOM parsing instead of Playwright so saved pages can
 * bypass an anti-spider response while still using the normal Markdown and
 * image-localization pipeline.
 */
export function extractZhihuArticleFromHtml(html: string): ExtractedZhihuArticle {
  if (!html.trim()) {
    throw new Error('Zhihu HTML file is empty');
  }

  const dom = new JSDOM(html);
  const { document } = dom.window;
  let content: Element | null = null;

  for (const selector of CONTENT_SELECTORS) {
    const candidate = document.querySelector(selector);
    if (candidate?.innerHTML.trim()) {
      content = candidate;
      break;
    }
  }

  if (!content) {
    throw new Error(
      `Zhihu HTML file does not contain article content; expected one of: ${CONTENT_SELECTORS.join(', ')}`,
    );
  }

  const contentClone = content.cloneNode(true) as Element;

  const title =
    pickFirstText(document, ['h1.Post-Title', 'h1.RichText-Title', '.Post-Title', 'article h1']) ||
    pickFirstAttribute(document, ['meta[property="og:title"]', 'meta[name="title"]'], 'content') ||
    document.title.trim() ||
    'Zhihu Article';

  const author =
    pickFirstAttribute(
      document,
      ['meta[name="author"]', 'meta[property="article:author"]'],
      'content',
    ) ||
    pickFirstText(document, [
      '.AuthorInfo-name',
      '.ContentItem-author .UserLink-link',
      '.UserLink-link',
      '[rel="author"]',
    ]);

  const published =
    pickFirstAttribute(
      document,
      [
        'meta[itemprop="datePublished"]',
        'meta[property="article:published_time"]',
        'meta[name="publish_date"]',
        'meta[name="date"]',
      ],
      'content',
    ) || pickFirstAttribute(document, ['time[datetime]'], 'datetime');

  const keywords = pickFirstAttribute(document, ['meta[name="keywords" i]'], 'content');

  return normalizeExtractedZhihuArticle({
    title,
    author,
    published,
    keywords,
    html: contentClone.innerHTML,
  });
}

/**
 * Detect if the page is a login/captcha/blocked page
 */
async function detectBlockedPage(page: any): Promise<string | null> {
  try {
    const contentState = await page.evaluate(
      ({ selectors, minTextLength }: { selectors: string[]; minTextLength: number }) => {
        const hasArticleContent = selectors.some((selector) => {
          const text = document.querySelector(selector)?.textContent?.trim() || '';
          return text.length >= minTextLength;
        });

        return {
          hasArticleContent,
          visibleText: (document.body?.innerText || '').slice(0, 20000),
        };
      },
      { selectors: CONTENT_SELECTORS, minTextLength: MIN_CONTENT_LENGTH },
    );

    // A rendered article is stronger evidence than generic login text in Zhihu's header.
    if (contentState?.hasArticleContent === true) {
      return null;
    }

    const title = await page.title();
    const url = page.url();
    const visibleText =
      typeof contentState?.visibleText === 'string' ? contentState.visibleText : '';
    let pathname = '';

    try {
      pathname = new URL(url).pathname;
    } catch {
      pathname = url;
    }

    if (
      /登录|login|sign.?in/i.test(title) ||
      /\/(?:signin|login)(?:\/|$)/i.test(pathname) ||
      /请先登录|登录后(?:继续|查看|访问|浏览)|登录知乎|使用(?:知乎)?账号登录|手机号登录/i.test(
        visibleText,
      )
    ) {
      return 'Zhihu blocked request (login page detected)';
    }

    if (
      /验证|captcha|security.?check|human.?verification/i.test(title) ||
      /\/(?:captcha|security-check|account\/unhuman)(?:\/|$)/i.test(pathname) ||
      /安全验证|请完成.{0,10}验证|拖动.{0,20}滑块|验证码|captcha|security.?check|human.?verification|verify you are human/i.test(
        visibleText,
      )
    ) {
      return 'Zhihu blocked request (captcha/security check detected)';
    }

    if (/反作弊|访问异常|请求存在异常|access.?denied/i.test(visibleText)) {
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
  logger?: Logger,
): Promise<ExtractedZhihuArticle> {
  const sanitizedUrl = sanitizeZhihuUrl(url);
  let lastError: Error | null = null;

  for (let attempt = 1; attempt <= maxRetries; attempt++) {
    try {
      logger?.info('Attempting extraction', {
        adapter: 'zhihu',
        attempt,
        maxRetries,
        url: sanitizedUrl,
      });

      await page.goto(sanitizedUrl, {
        waitUntil: 'domcontentloaded',
        timeout: 60000,
      });

      await page.waitForTimeout(JS_INITIALIZATION_DELAY);

      const blockReason = await detectBlockedPage(page);
      if (blockReason) {
        logger?.warn('Zhihu blocked request', {
          adapter: 'zhihu',
          attempt,
          reason: blockReason,
          blockedDetected: true,
        });
        throw new Error(blockReason);
      }

      await waitForContent(page, CONTENT_SELECTORS, {
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
          keywords: pickMeta(['meta[name="keywords" i]']),
          html,
        };
      });

      if (!result.html?.trim()) {
        throw new Error('Zhihu DOM structure changed: Failed to extract article content');
      }

      const normalizedResult = normalizeExtractedZhihuArticle(result);

      logger?.info('Successfully extracted article', {
        adapter: 'zhihu',
        attempt,
        title: normalizedResult.title,
        hasAuthor: Boolean(normalizedResult.author),
        hasPublished: Boolean(normalizedResult.published),
        htmlLength: normalizedResult.html.length,
      });
      return normalizedResult;
    } catch (error) {
      lastError = error as Error;
      logger?.warn('Extraction attempt failed', {
        adapter: 'zhihu',
        attempt,
        maxRetries,
        error: error instanceof Error ? error.message : String(error),
      });

      if (attempt < maxRetries) {
        const backoffMs = Math.min(BASE_BACKOFF_MS * Math.pow(2, attempt - 1), MAX_BACKOFF_MS);
        logger?.info('Waiting before retry', {
          adapter: 'zhihu',
          attempt,
          backoffMs,
        });
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
    const {
      slug = 'zhihu-article',
      imageRoot = '/tmp/images',
      publicBasePath,
      logger: parentLogger,
    } = options;

    // Create child logger with context
    const logger =
      parentLogger?.child({
        module: 'import',
        adapter: 'zhihu',
        url,
        slug,
      }) ?? createLogger({ silent: true });

    const extractionSpan = logger.span({ name: 'zhihu-extraction', fields: { adapter: 'zhihu' } });
    extractionSpan.start();

    try {
      // Extract article with retry logic
      const result = await extractZhihuWithRetry(page, url, MAX_RETRIES, logger);

      logger.info('Converting HTML to Markdown', {
        adapter: 'zhihu',
        htmlLength: result.html.length,
      });

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

      extractionSpan.end({
        status: 'ok',
        fields: {
          imagesCount: images.length,
          markdownLength: markdown.length,
        },
      });

      logger.summary({
        status: 'ok',
        adapter: 'zhihu',
        title: result.title,
        imagesCount: images.length,
        markdownLength: markdown.length,
      });

      return {
        title: result.title,
        markdown,
        canonicalUrl: sanitizeZhihuUrl(url),
        source: 'zhihu',
        author: result.author,
        publishedAt: result.published || undefined,
        tags: result.tags,
        images: images.map((localPath) => ({ url: '', localPath })),
      };
    } catch (error) {
      extractionSpan.end({ status: 'fail' });
      logger.error(error instanceof Error ? error : new Error(String(error)), {
        adapter: 'zhihu',
        url,
      });
      throw error;
    }
  },

  async fetchArticleFromHtml(input: FetchArticleFromHtmlInput): Promise<Article> {
    const { url, html, options = {} } = input;
    const {
      slug = 'zhihu-article',
      imageRoot = '/tmp/images',
      publicBasePath,
      logger: parentLogger,
    } = options;
    const logger =
      parentLogger?.child({
        module: 'import',
        adapter: 'zhihu',
        input: 'html-file',
        url,
        slug,
      }) ?? createLogger({ silent: true });
    const extractionSpan = logger.span({
      name: 'zhihu-html-extraction',
      fields: { adapter: 'zhihu', input: 'html-file' },
    });
    extractionSpan.start();

    try {
      const result = extractZhihuArticleFromHtml(html);
      logger.info('Converting local HTML to Markdown', {
        adapter: 'zhihu',
        htmlLength: result.html.length,
      });

      const { markdown, images } = await htmlToMdx(result.html, {
        slug,
        provider: 'zhihu',
        baseUrl: sanitizeZhihuUrl(url),
        imageRoot,
        articleUrl: url,
        publicBasePath: publicBasePath || `/images/zhihu/${slug}`,
        downloadImage: options.downloadImage,
      });

      extractionSpan.end({
        status: 'ok',
        fields: {
          imagesCount: images.length,
          markdownLength: markdown.length,
        },
      });
      logger.summary({
        status: 'ok',
        adapter: 'zhihu',
        input: 'html-file',
        title: result.title,
        imagesCount: images.length,
        markdownLength: markdown.length,
      });

      return {
        title: result.title,
        markdown,
        canonicalUrl: sanitizeZhihuUrl(url),
        source: 'zhihu',
        author: result.author,
        publishedAt: result.published || undefined,
        tags: result.tags,
        images: images.map((localPath) => ({ url: '', localPath })),
      };
    } catch (error) {
      extractionSpan.end({ status: 'fail' });
      logger.error(error instanceof Error ? error : new Error(String(error)), {
        adapter: 'zhihu',
        input: 'html-file',
        url,
      });
      throw error;
    }
  },
};
