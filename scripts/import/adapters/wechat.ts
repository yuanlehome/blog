/**
 * WeChat Adapter
 *
 * Handles article import from WeChat Official Account articles (mp.weixin.qq.com)
 */

import type { Adapter, Article, FetchArticleInput } from './types.js';
import { htmlToMdx } from '../../content-import.js';
import { createLogger } from '../../logger/index.js';

/**
 * Check if URL is from WeChat domain
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
 * WeChat adapter implementation
 */
export const wechatAdapter: Adapter = {
  id: 'wechat',
  name: 'WeChat',

  canHandle(url: string): boolean {
    return isFromDomain(url, 'mp.weixin.qq.com');
  },

  async fetchArticle(input: FetchArticleInput): Promise<Article> {
    const { url, page, options = {} } = input;
    const {
      slug = 'wechat-article',
      imageRoot = '/tmp/images',
      publicBasePath,
      logger: parentLogger,
    } = options;

    // Create child logger with context
    const logger =
      parentLogger?.child({
        module: 'import',
        adapter: 'wechat',
        url,
        slug,
      }) ?? createLogger({ silent: true });

    const extractionSpan = logger.span({
      name: 'wechat-extraction',
      fields: { adapter: 'wechat' },
    });
    extractionSpan.start();

    try {
      logger.info('Navigating to WeChat page', { adapter: 'wechat', url });

      // Navigate to page
      await page.goto(url, { waitUntil: 'networkidle', timeout: 120000 });
      await page.waitForSelector('#js_content, .rich_media_content', { timeout: 30000 });

      logger.debug('Fixing lazy-loaded images', { adapter: 'wechat' });

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

      // Extract article metadata and content
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
        throw new Error('Failed to extract WeChat article content');
      }

      logger.info('Converting HTML to Markdown', {
        adapter: 'wechat',
        htmlLength: result.html.length,
      });

      // Convert HTML to Markdown
      const { markdown, images } = await htmlToMdx(result.html, {
        slug,
        provider: 'wechat',
        baseUrl: 'https://mp.weixin.qq.com',
        imageRoot,
        articleUrl: url,
        publicBasePath: publicBasePath || `/images/wechat/${slug}`,
        downloadImage: options.downloadImage,
      });

      // Safety check: Ensure no WeChat lazy-load placeholders in final markdown
      // This prevents publishing articles with broken image placeholders
      if (/data:image\/svg\+xml/i.test(markdown)) {
        throw new Error(
          'WeChat image placeholder detected (data:image/svg+xml) in generated markdown. ' +
            'Image extraction failed - real image URLs were not properly captured from lazy-loaded attributes.',
        );
      }

      extractionSpan.end({
        status: 'ok',
        fields: {
          imagesCount: images.length,
          markdownLength: markdown.length,
        },
      });

      logger.summary({
        status: 'ok',
        adapter: 'wechat',
        title: result.title,
        imagesCount: images.length,
        markdownLength: markdown.length,
      });

      return {
        title: result.title,
        markdown,
        canonicalUrl: url,
        source: 'wechat',
        author: result.author,
        publishedAt: result.published || undefined,
        images: images.map((localPath) => ({ url: '', localPath })),
      };
    } catch (error) {
      extractionSpan.end({ status: 'fail' });
      logger.error(error instanceof Error ? error : new Error(String(error)), {
        adapter: 'wechat',
        url,
      });
      throw error;
    }
  },
};
