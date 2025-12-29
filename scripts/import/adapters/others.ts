/**
 * Others Adapter (Fallback)
 * 
 * Generic adapter for any website not handled by specific adapters
 * Uses Readability-based extraction as fallback
 */

import type { Adapter, Article, FetchArticleInput } from './types.js';
import { extractArticleFromHtml, htmlToMdx } from '../../content-import.js';

/**
 * Others adapter - always returns true (lowest priority)
 */
export const othersAdapter: Adapter = {
  id: 'others',
  name: 'Others (Generic)',

  canHandle(): boolean {
    // Always returns true - this is the fallback adapter
    return true;
  },

  async fetchArticle(input: FetchArticleInput): Promise<Article> {
    const { url, page, options = {} } = input;
    const { slug = 'article', imageRoot = '/tmp/images', publicBasePath } = options;

    // Navigate to page
    await page.goto(url, { waitUntil: 'networkidle', timeout: 120000 });
    
    // Get page content
    const content = await page.content();
    
    // Use generic extraction
    const article = extractArticleFromHtml(content, url);

    if (!article.html?.trim()) {
      throw new Error('Failed to extract article content');
    }

    // Convert HTML to Markdown
    const { markdown, images } = await htmlToMdx(article.html, {
      slug,
      provider: 'others',
      baseUrl: article.baseUrl || new URL(url).origin,
      imageRoot,
      articleUrl: url,
      publicBasePath: publicBasePath || `/images/others/${slug}`,
      downloadImage: options.downloadImage,
    });

    return {
      title: article.title,
      markdown,
      canonicalUrl: url,
      source: 'others',
      author: article.author,
      publishedAt: article.published || undefined,
      updatedAt: article.updated || undefined,
      images: images.map((localPath) => ({ url: '', localPath })),
    };
  },
};
