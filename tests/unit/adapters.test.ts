/**
 * Adapter Tests
 *
 * Tests for import adapters with mocked browser and network calls
 */

import fs from 'fs';
import path from 'path';
import { describe, it, expect, vi } from 'vitest';
import { extractZhihuArticleFromHtml, zhihuAdapter } from '../../scripts/import/adapters/zhihu.js';
import { wechatAdapter } from '../../scripts/import/adapters/wechat.js';
import { mediumAdapter } from '../../scripts/import/adapters/medium.js';
import { othersAdapter } from '../../scripts/import/adapters/others.js';
import { createLogger } from '../../scripts/logger/index.js';
import type { LogFields } from '../../scripts/logger/types.js';

// Mock page object
function createMockPage() {
  return {
    goto: vi.fn().mockResolvedValue(undefined),
    waitForSelector: vi.fn().mockResolvedValue(undefined),
    waitForTimeout: vi.fn().mockResolvedValue(undefined),
    waitForFunction: vi.fn().mockResolvedValue(undefined),
    content: vi.fn().mockResolvedValue('<html><body>Mock content</body></html>'),
    title: vi.fn().mockResolvedValue('Mock Title'),
    url: vi.fn().mockReturnValue('https://example.com/article'),
    evaluate: vi.fn().mockResolvedValue({
      title: 'Test Article',
      author: 'Test Author',
      published: '2024-01-01',
      html: '<p>Test content</p>',
    }),
  };
}

const zhihuNormalizationHtml = `
  <!doctype html>
  <html>
    <head>
      <title>Zhihu normalization</title>
      <meta name="author" content="Test Author">
      <meta name="keywords" content="大模型, 推理优化，系统，大模型">
      <meta itemprop="datePublished" content="2026-08-06T00:15:00+08:00">
    </head>
    <body>
      <h1 class="Post-Title">Zhihu normalization</h1>
      <div class="Post-RichText">
        <h3 class="section">第一节</h3>
        <p>
          <a href="https://zhida.zhihu.com/search?q=实体&amp;zd_token=temporary-token"><strong>实体术语</strong></a>
          和 <a href="https://zhida.zhihu.com/search?q=无令牌实体&amp;zhida_source=entity"><em>无令牌实体术语</em></a>
          和 <a href="https://example.com/reference">正常链接</a>
        </p>
        <h4>细节</h4>
      </div>
    </body>
  </html>
`;

describe('Zhihu Adapter', () => {
  it('should extract metadata and article innerHTML from a complete page snapshot', () => {
    const html = fs.readFileSync(
      path.join(process.cwd(), 'tests/fixtures/zhihu/article.html'),
      'utf8',
    );

    const article = extractZhihuArticleFromHtml(html);

    expect(article.title).toBe('Rust 2021 Edition');
    expect(article.author).toBe('Test Author');
    expect(article.published).toBe('2023-06-15');
    expect(article.html).toContain('这是一篇关于 Rust 2021 Edition 的文章');
    expect(article.html).not.toContain('<h1');
  });

  it('should normalize saved-page metadata, entity links, and heading depth', () => {
    const article = extractZhihuArticleFromHtml(zhihuNormalizationHtml);

    expect(article.published).toBe('2026-08-06');
    expect(article.tags).toEqual(['大模型', '推理优化', '系统']);
    expect(article.html).toContain('<h2 class="section">第一节</h2>');
    expect(article.html).toContain('<h3>细节</h3>');
    expect(article.html).toContain('<strong>实体术语</strong>');
    expect(article.html).toContain('<em>无令牌实体术语</em>');
    expect(article.html).not.toContain('zhida.zhihu.com');
    expect(article.html).toContain('href="https://example.com/reference"');
  });

  describe('canHandle', () => {
    it('should handle zhuanlan.zhihu.com URLs', () => {
      expect(zhihuAdapter.canHandle('https://zhuanlan.zhihu.com/p/123456')).toBe(true);
    });

    it('should not handle general zhihu.com URLs without zhuanlan', () => {
      expect(zhihuAdapter.canHandle('https://www.zhihu.com/question/123456/answer/789')).toBe(
        false,
      );
    });

    it('should not handle non-Zhihu URLs', () => {
      expect(zhihuAdapter.canHandle('https://example.com/article')).toBe(false);
    });
  });

  describe('fetchArticle', () => {
    it('should import a complete local HTML snapshot without a page', async () => {
      const html = fs.readFileSync(
        path.join(process.cwd(), 'tests/fixtures/zhihu/article.html'),
        'utf8',
      );
      const mockDownloadImage = vi.fn().mockResolvedValue('/images/zhihu/local/001-image.jpg');

      const result = await zhihuAdapter.fetchArticleFromHtml!({
        url: 'https://zhuanlan.zhihu.com/p/123456?share_code=abc&utm_psn=456',
        html,
        options: {
          slug: 'local',
          imageRoot: '/tmp/test',
          publicBasePath: '/images/zhihu/local',
          downloadImage: mockDownloadImage,
        },
      });

      expect(result.title).toBe('Rust 2021 Edition');
      expect(result.author).toBe('Test Author');
      expect(result.publishedAt).toBe('2023-06-15');
      expect(result.canonicalUrl).toBe('https://zhuanlan.zhihu.com/p/123456');
      expect(result.markdown).toContain('这是一篇关于 Rust 2021 Edition 的文章');
      expect(mockDownloadImage).toHaveBeenCalledWith(
        'https://example.com/image.jpg',
        'zhihu',
        'local',
        '/tmp/test',
        0,
        'https://zhuanlan.zhihu.com/p/123456?share_code=abc&utm_psn=456',
        '/images/zhihu/local',
      );
    });

    it('should apply Zhihu normalization when importing a local HTML snapshot', async () => {
      const result = await zhihuAdapter.fetchArticleFromHtml!({
        url: 'https://zhuanlan.zhihu.com/p/123456',
        html: zhihuNormalizationHtml,
        options: { slug: 'local', imageRoot: '/tmp/test' },
      });

      expect(result.publishedAt).toBe('2026-08-06');
      expect(result.tags).toEqual(['大模型', '推理优化', '系统']);
      expect(result.markdown).toContain('## 第一节');
      expect(result.markdown).toContain('### 细节');
      expect(result.markdown).toContain('**实体术语**');
      expect(result.markdown).not.toContain('zhida.zhihu.com');
      expect(result.markdown).toContain('[正常链接](https://example.com/reference)');
    });

    it('should extract article from Zhihu', async () => {
      const mockPage = createMockPage();
      const mockDownloadImage = vi.fn().mockResolvedValue('/images/test.jpg');

      // Mock successful extraction
      mockPage.evaluate.mockResolvedValue({
        title: 'Zhihu Article',
        author: 'Zhihu Author',
        published: '2024-01-01',
        html: '<div class="Post-RichText"><p>Zhihu content</p></div>',
      });

      const result = await zhihuAdapter.fetchArticle({
        url: 'https://zhuanlan.zhihu.com/p/123456',
        page: mockPage as any,
        options: {
          slug: 'test-slug',
          imageRoot: '/tmp/test',
          downloadImage: mockDownloadImage,
        },
      });

      expect(result.title).toBe('Zhihu Article');
      expect(result.source).toBe('zhihu');
      expect(result.markdown).toBeTruthy();
      expect(mockPage.goto).toHaveBeenCalled();
    });

    it('should apply Zhihu normalization to live extraction', async () => {
      const mockPage = createMockPage();
      mockPage.evaluate
        .mockResolvedValueOnce({
          hasArticleContent: true,
          visibleText: 'A'.repeat(150),
        })
        .mockResolvedValueOnce({
          title: 'Live Zhihu Article',
          author: 'Live Author',
          published: '2026-08-06T00:15:00+08:00',
          keywords: '大模型，推理优化, 大模型',
          html: `
            <h4>实时正文</h4>
            <p><a href="https://zhida.zhihu.com/search?q=实体&zd_token=short-lived">实时实体</a></p>
          `,
        });

      const result = await zhihuAdapter.fetchArticle({
        url: 'https://zhuanlan.zhihu.com/p/123456',
        page: mockPage as any,
        options: { slug: 'live', imageRoot: '/tmp/test' },
      });

      expect(result.publishedAt).toBe('2026-08-06');
      expect(result.tags).toEqual(['大模型', '推理优化']);
      expect(result.markdown).toContain('## 实时正文');
      expect(result.markdown).toContain('实时实体');
      expect(result.markdown).not.toContain('zhida.zhihu.com');
    });

    it('should not treat the login button as a blocked page when article content is present', async () => {
      const mockPage = createMockPage();
      const articleText = 'A'.repeat(150);
      mockPage.content.mockResolvedValue(`
        <html>
          <body>
            <header><button>登录</button></header>
            <article class="Post-RichText">${articleText}</article>
          </body>
        </html>
      `);
      mockPage.evaluate
        .mockResolvedValueOnce({
          hasArticleContent: true,
          visibleText: `登录 ${articleText}`,
        })
        .mockResolvedValueOnce({
          title: 'Zhihu Article',
          author: 'Zhihu Author',
          published: '2024-01-01',
          html: `<p>${articleText}</p>`,
        });

      const result = await zhihuAdapter.fetchArticle({
        url: 'https://zhuanlan.zhihu.com/p/123456',
        page: mockPage as any,
        options: { slug: 'test-slug', imageRoot: '/tmp/test' },
      });

      expect(result.title).toBe('Zhihu Article');
      expect(result.markdown).toContain(articleText);
    });

    it('should still reject a genuine login page without article content', async () => {
      const mockPage = createMockPage();
      mockPage.title.mockResolvedValue('知乎');
      mockPage.url.mockReturnValue('https://zhuanlan.zhihu.com/p/123456');
      mockPage.evaluate.mockResolvedValue({
        hasArticleContent: false,
        visibleText: '登录知乎，登录后继续浏览',
      });

      await expect(
        zhihuAdapter.fetchArticle({
          url: 'https://zhuanlan.zhihu.com/p/123456',
          page: mockPage as any,
          options: { slug: 'test-slug', imageRoot: '/tmp/test' },
        }),
      ).rejects.toThrow('Zhihu blocked request (login page detected)');

      expect(mockPage.waitForSelector).not.toHaveBeenCalled();
    });

    it('should handle extraction errors gracefully', async () => {
      const mockPage = createMockPage();
      mockPage.evaluate.mockRejectedValue(new Error('Extraction failed'));

      await expect(
        zhihuAdapter.fetchArticle({
          url: 'https://zhuanlan.zhihu.com/p/123456',
          page: mockPage as any,
          options: { slug: 'test-slug', imageRoot: '/tmp/test' },
        }),
      ).rejects.toThrow();
    });
  });
});

describe('WeChat Adapter', () => {
  describe('canHandle', () => {
    it('should handle mp.weixin.qq.com URLs', () => {
      expect(wechatAdapter.canHandle('https://mp.weixin.qq.com/s/abc123')).toBe(true);
    });

    it('should not handle non-WeChat URLs', () => {
      expect(wechatAdapter.canHandle('https://example.com/article')).toBe(false);
    });
  });

  describe('fetchArticle', () => {
    it('should extract article from WeChat', async () => {
      const mockPage = createMockPage();
      const mockDownloadImage = vi.fn().mockResolvedValue('/images/test.jpg');

      mockPage.evaluate.mockResolvedValue({
        title: 'WeChat Article',
        author: 'WeChat Author',
        published: '2024-01-01',
        html: '<div id="js_content"><p>WeChat content</p></div>',
      });

      const result = await wechatAdapter.fetchArticle({
        url: 'https://mp.weixin.qq.com/s/abc123',
        page: mockPage as any,
        options: {
          slug: 'test-slug',
          imageRoot: '/tmp/test',
          downloadImage: mockDownloadImage,
        },
      });

      expect(result.title).toBe('WeChat Article');
      expect(result.source).toBe('wechat');
      expect(result.markdown).toBeTruthy();
      expect(mockPage.waitForSelector).toHaveBeenCalled();
    });

    it('should handle missing content selector', async () => {
      const mockPage = createMockPage();
      mockPage.waitForSelector.mockRejectedValue(new Error('Selector not found'));

      await expect(
        wechatAdapter.fetchArticle({
          url: 'https://mp.weixin.qq.com/s/abc123',
          page: mockPage as any,
          options: { slug: 'test-slug', imageRoot: '/tmp/test' },
        }),
      ).rejects.toThrow();
    });
  });
});

describe('Medium Adapter', () => {
  describe('canHandle', () => {
    it('should handle medium.com URLs', () => {
      expect(mediumAdapter.canHandle('https://medium.com/@user/article-title-123')).toBe(true);
    });

    it('should handle custom Medium domains', () => {
      expect(mediumAdapter.canHandle('https://blog.medium.com/article')).toBe(true);
    });

    it('should not handle non-Medium URLs', () => {
      expect(mediumAdapter.canHandle('https://example.com/article')).toBe(false);
    });
  });

  describe('fetchArticle', () => {
    it('should extract article from Medium', async () => {
      const mockPage = createMockPage();
      const mockDownloadImage = vi.fn().mockResolvedValue('/images/test.jpg');

      mockPage.evaluate.mockResolvedValue({
        title: 'Medium Article',
        author: 'Medium Author',
        published: '2024-01-01',
        html: '<article><p>Medium content</p></article>',
      });

      const result = await mediumAdapter.fetchArticle({
        url: 'https://medium.com/@user/article',
        page: mockPage as any,
        options: {
          slug: 'test-slug',
          imageRoot: '/tmp/test',
          downloadImage: mockDownloadImage,
        },
      });

      expect(result.title).toBe('Medium Article');
      expect(result.source).toBe('medium');
      expect(result.markdown).toBeTruthy();
    });

    it('should handle empty content', async () => {
      const mockPage = createMockPage();
      mockPage.evaluate.mockResolvedValue({
        title: 'Empty Article',
        author: '',
        published: '',
        html: '',
      });

      await expect(
        mediumAdapter.fetchArticle({
          url: 'https://medium.com/@user/article',
          page: mockPage as any,
          options: { slug: 'test-slug', imageRoot: '/tmp/test' },
        }),
      ).rejects.toThrow('Failed to extract Medium article content');
    });
  });
});

describe('Others Adapter', () => {
  describe('canHandle', () => {
    it('should handle any URL as fallback', () => {
      expect(othersAdapter.canHandle('https://example.com/article')).toBe(true);
      expect(othersAdapter.canHandle('https://blog.example.org/post/123')).toBe(true);
    });
  });

  describe('fetchArticle', () => {
    it('should extract article from generic site', async () => {
      const mockPage = createMockPage();
      const mockDownloadImage = vi.fn().mockResolvedValue('/images/test.jpg');

      mockPage.content.mockResolvedValue(`
        <html>
          <body>
            <article>
              <h1>Generic Article</h1>
              <p>Generic content</p>
            </article>
          </body>
        </html>
      `);

      const result = await othersAdapter.fetchArticle({
        url: 'https://example.com/article',
        page: mockPage as any,
        options: {
          slug: 'test-slug',
          imageRoot: '/tmp/test',
          downloadImage: mockDownloadImage,
        },
      });

      expect(result.title).toBeTruthy();
      expect(result.source).toBe('others');
      expect(result.markdown).toBeTruthy();
      expect(mockPage.goto).toHaveBeenCalled();
    });

    it('should handle pages with no article content', async () => {
      const mockPage = createMockPage();
      mockPage.content.mockResolvedValue('<html><body><div>Not much here</div></body></html>');

      const result = await othersAdapter.fetchArticle({
        url: 'https://example.com/sparse',
        page: mockPage as any,
        options: {
          slug: 'test-slug',
          imageRoot: '/tmp/test',
          downloadImage: async () => null,
        },
      });

      expect(result.title).toBeTruthy();
      expect(result.source).toBe('others');
    });
  });
});

describe('Adapters with Logger', () => {
  describe('Zhihu Adapter', () => {
    it('should log extraction attempts and results', async () => {
      const mockPage = createMockPage();
      const mockDownloadImage = vi.fn().mockResolvedValue('/images/test.jpg');

      mockPage.evaluate.mockResolvedValue({
        title: 'Zhihu Article with Logging',
        author: 'Test Author',
        published: '2024-01-01',
        html: '<p>Test content</p>',
      });

      const logCalls: Array<{ level: string; message: string; fields?: LogFields }> = [];
      const spanCalls: Array<{ name: string; status?: string; fields?: LogFields }> = [];

      const mockLogger = createLogger({ silent: true });

      // Wrap child to track all calls
      const originalChild = mockLogger.child.bind(mockLogger);
      mockLogger.child = (fields: LogFields) => {
        const childLogger = originalChild(fields);
        const originalInfo = childLogger.info.bind(childLogger);
        childLogger.info = (message: string, infoFields?: LogFields) => {
          logCalls.push({ level: 'info', message, fields: { ...fields, ...infoFields } });
          return originalInfo(message, infoFields);
        };
        const originalSpan = childLogger.span.bind(childLogger);
        childLogger.span = (opts) => {
          const span = originalSpan(opts);
          spanCalls.push({ name: opts.name, fields: { ...fields, ...opts.fields } });
          const originalEnd = span.end.bind(span);
          span.end = (endOpts) => {
            spanCalls.push({
              name: opts.name,
              status: endOpts?.status,
              fields: { ...fields, ...opts.fields, ...endOpts?.fields },
            });
            return originalEnd(endOpts);
          };
          return span;
        };
        return childLogger;
      };

      await zhihuAdapter.fetchArticle({
        url: 'https://zhuanlan.zhihu.com/p/123456',
        page: mockPage as any,
        options: {
          slug: 'test-slug',
          imageRoot: '/tmp/test',
          downloadImage: mockDownloadImage,
          logger: mockLogger,
        },
      });

      // Verify logger was called with adapter context
      expect(logCalls.filter((c) => c.fields?.adapter === 'zhihu').length).toBeGreaterThan(0);
      expect(spanCalls.filter((c) => c.name === 'zhihu-extraction').length).toBeGreaterThan(0);
      expect(
        spanCalls.filter((c) => c.name === 'zhihu-extraction' && c.status === 'ok').length,
      ).toBeGreaterThan(0);
    });

    it('should log retry attempts with backoff info', async () => {
      const mockPage = createMockPage();
      let attemptCount = 0;

      // First attempt fails, second succeeds
      mockPage.evaluate.mockImplementation((_pageFunction, argument) => {
        if (argument?.selectors) {
          return Promise.resolve({ hasArticleContent: false, visibleText: '' });
        }

        attemptCount++;
        if (attemptCount === 1) {
          return Promise.reject(new Error('Temporary failure'));
        }
        return Promise.resolve({
          title: 'Article',
          author: 'Author',
          published: '2024-01-01',
          html: '<p>Content</p>',
        });
      });

      const warnCalls: Array<{ message: string; fields?: LogFields }> = [];
      const infoCalls: Array<{ message: string; fields?: LogFields }> = [];
      const mockLogger = createLogger({ silent: true });

      // Wrap child to track warnings and info
      const originalChild = mockLogger.child.bind(mockLogger);
      mockLogger.child = (fields: LogFields) => {
        const childLogger = originalChild(fields);
        const originalWarn = childLogger.warn.bind(childLogger);
        childLogger.warn = (message: string, warnFields?: LogFields) => {
          warnCalls.push({ message, fields: { ...fields, ...warnFields } });
          return originalWarn(message, warnFields);
        };
        const originalInfo = childLogger.info.bind(childLogger);
        childLogger.info = (message: string, infoFields?: LogFields) => {
          infoCalls.push({ message, fields: { ...fields, ...infoFields } });
          return originalInfo(message, infoFields);
        };
        return childLogger;
      };

      await zhihuAdapter.fetchArticle({
        url: 'https://zhuanlan.zhihu.com/p/123456',
        page: mockPage as any,
        options: {
          slug: 'test',
          imageRoot: '/tmp/test',
          downloadImage: async () => null,
          logger: mockLogger,
        },
      });

      // Verify retry was logged with attempt and backoff info
      expect(
        warnCalls.filter((c) => c.fields?.attempt === 1 && c.fields?.backoffMs).length +
          infoCalls.filter((c) => c.fields?.attempt === 1 && c.fields?.backoffMs).length,
      ).toBeGreaterThan(0);
    });
  });

  describe('Medium Adapter', () => {
    it('should log extraction strategy and stats', async () => {
      const mockPage = createMockPage();
      const mockDownloadImage = vi.fn().mockResolvedValue('/images/test.jpg');

      mockPage.evaluate.mockResolvedValue({
        title: 'Medium Article',
        author: 'Author',
        published: '2024-01-01',
        html: '<p>Content</p>',
      });

      const logCalls: Array<{ level: string; message: string; fields?: LogFields }> = [];
      const summaryCalls: Array<{ fields?: LogFields }> = [];
      const mockLogger = createLogger({ silent: true });

      // Wrap child to track info and summary calls
      const originalChild = mockLogger.child.bind(mockLogger);
      mockLogger.child = (fields: LogFields) => {
        const childLogger = originalChild(fields);
        const originalInfo = childLogger.info.bind(childLogger);
        childLogger.info = (message: string, infoFields?: LogFields) => {
          logCalls.push({ level: 'info', message, fields: { ...fields, ...infoFields } });
          return originalInfo(message, infoFields);
        };
        const originalSummary = childLogger.summary.bind(childLogger);
        childLogger.summary = (summaryFields: LogFields) => {
          summaryCalls.push({ fields: { ...fields, ...summaryFields } });
          return originalSummary(summaryFields);
        };
        return childLogger;
      };

      await mediumAdapter.fetchArticle({
        url: 'https://medium.com/@user/article',
        page: mockPage as any,
        options: {
          slug: 'test-slug',
          imageRoot: '/tmp/test',
          downloadImage: mockDownloadImage,
          logger: mockLogger,
        },
      });

      // Verify logger captured adapter and stats
      expect(logCalls.filter((c) => c.fields?.adapter === 'medium').length).toBeGreaterThan(0);

      // Check either in info logs or summary
      const hasStatsInInfo =
        logCalls.filter(
          (c) => c.fields?.imagesCount !== undefined && c.fields?.markdownLength !== undefined,
        ).length > 0;
      const hasStatsInSummary =
        summaryCalls.filter(
          (c) => c.fields?.imagesCount !== undefined && c.fields?.markdownLength !== undefined,
        ).length > 0;

      expect(hasStatsInInfo || hasStatsInSummary).toBe(true);
    });
  });
});
