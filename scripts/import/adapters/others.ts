/**
 * Others Adapter (Fallback)
 *
 * Generic adapter for any website not handled by specific adapters
 * Uses Readability-based extraction as fallback
 */

import type { Adapter, Article, FetchArticleInput } from './types.js';
import { extractArticleFromHtml, htmlToMdx } from '../../content-import.js';
import type { Logger } from '../../logger/types.js';
import { createLogger } from '../../logger/index.js';

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
    const { slug = 'article', imageRoot = '/tmp/images', publicBasePath, logger: parentLogger } = options;

    // Create child logger with context
    const logger =
      parentLogger?.child({
        module: 'import',
        adapter: 'others',
        url,
        slug,
      }) ?? createLogger({ silent: true });

    const extractionSpan = logger.span({ name: 'others-extraction', fields: { adapter: 'others' } });
    extractionSpan.start();

    try {
      logger.info('Navigating to page', { adapter: 'others', url, strategy: 'readability' });

      // Navigate to page
      await page.goto(url, { waitUntil: 'networkidle', timeout: 120000 });

      // Get page content
      const content = await page.content();

      logger.debug('Extracting article using Readability', { adapter: 'others' });

      // Use generic extraction
      const article = extractArticleFromHtml(content, url);

      if (!article.html?.trim()) {
        throw new Error('Failed to extract article content');
      }

      logger.info('Converting HTML to Markdown', {
        adapter: 'others',
        htmlLength: article.html.length,
      });

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

      extractionSpan.end({
        status: 'ok',
        fields: {
          imagesCount: images.length,
          markdownLength: markdown.length,
        },
      });

      logger.summary({
        status: 'ok',
        adapter: 'others',
        title: article.title,
        imagesCount: images.length,
        markdownLength: markdown.length,
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
    } catch (error) {
      extractionSpan.end({ status: 'fail' });
      logger.error(error instanceof Error ? error : new Error(String(error)), {
        adapter: 'others',
        url,
      });
      throw error;
    }
  },
};
