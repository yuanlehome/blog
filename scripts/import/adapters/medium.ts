/**
 * Medium Adapter
 *
 * Handles article import from Medium.com and custom Medium publications
 */

import type { Adapter, Article, FetchArticleInput } from './types.js';
import { htmlToMdx } from '../../content-import.js';

/**
 * Check if URL is from Medium domain
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
 * Medium adapter implementation
 */
export const mediumAdapter: Adapter = {
  id: 'medium',
  name: 'Medium',

  canHandle(url: string): boolean {
    return isFromDomain(url, 'medium.com');
  },

  async fetchArticle(input: FetchArticleInput): Promise<Article> {
    const { url, page, options = {} } = input;
    const { slug = 'medium-article', imageRoot = '/tmp/images', publicBasePath } = options;

    // Navigate to page
    await page.goto(url, { waitUntil: 'networkidle', timeout: 120000 });
    await page.waitForSelector('article', { timeout: 30000 });

    const origin = new URL(url).origin;

    // Extract article metadata and content
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
      throw new Error('Failed to extract Medium article content');
    }

    // Convert HTML to Markdown
    const { markdown, images } = await htmlToMdx(result.html, {
      slug,
      provider: 'medium',
      baseUrl: origin,
      imageRoot,
      articleUrl: url,
      publicBasePath: publicBasePath || `/images/medium/${slug}`,
      downloadImage: options.downloadImage,
    });

    return {
      title: result.title,
      markdown,
      canonicalUrl: url,
      source: 'medium',
      author: result.author,
      publishedAt: result.published || undefined,
      images: images.map((localPath) => ({ url: '', localPath })),
    };
  },
};
