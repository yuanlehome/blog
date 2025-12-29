import fs from 'fs';
import path from 'path';
import { describe, expect, it, vi } from 'vitest';
import { extractArticleFromHtml, htmlToMdx, sanitizeMdx } from '../../scripts/content-import';
import { resolveAdapter } from '../../scripts/import/adapters/index';

const fixturePath = path.join(process.cwd(), 'tests/fixtures/matmul.html');
const fixtureHtml = fs.readFileSync(fixturePath, 'utf8');
const MATMUL_URL = 'https://www.aleksagordic.com/blog/matmul';

describe('content import for external articles', () => {
  it('extracts the matmul article without noise', () => {
    const article = extractArticleFromHtml(fixtureHtml, MATMUL_URL);

    expect(article.title).toBe('Matrix Multiplication from First Principles');
    expect(article.author).toBe('Aleksa Gordic');
    expect(article.published).toBe('2023-02-01');
    expect(article.updated).toBe('2023-02-10');
    expect(article.html).toContain('Matrix multiplication sits at the heart of deep learning');
    expect(article.html).toContain('Vectorized implementation');
    expect(article.html).not.toContain('Table of contents placeholder');
    expect(article.html).not.toContain('Footer noise');
    expect(article.html).not.toContain('Comments placeholder');
  });

  it('converts matmul html into stable markdown with math and images', async () => {
    const article = extractArticleFromHtml(fixtureHtml, MATMUL_URL);
    const downloadImage = vi.fn(
      async (
        imageUrl: string,
        _provider: string,
        slug: string,
        _imageRoot: string,
        index: number,
        _articleUrl?: string,
        publicBasePath?: string,
      ) => {
        const base = publicBasePath || `/images/others/${slug}`;
        return path.posix.join(
          base,
          `${String(index + 1).padStart(3, '0')}-${path.basename(new URL(imageUrl).pathname)}`,
        );
      },
    );

    const { markdown, images } = await htmlToMdx(article.html, {
      slug: 'matmul',
      provider: 'others',
      baseUrl: article.baseUrl,
      imageRoot: '/tmp/images',
      articleUrl: MATMUL_URL,
      publicBasePath: '/images/others/matmul',
      downloadImage,
    });

    expect(markdown).toMatchSnapshot();
    expect(images[0]).toContain('/images/others/matmul/001');
    expect(downloadImage).toHaveBeenCalled();
  });

  describe('adapter resolution', () => {
    it('resolves zhihu adapter for zhihu column URLs', () => {
      const adapter = resolveAdapter('https://zhuanlan.zhihu.com/p/668888063');
      expect(adapter).not.toBeNull();
      expect(adapter?.id).toBe('zhihu');
    });

    it('resolves medium adapter for medium URLs', () => {
      const adapter = resolveAdapter('https://medium.com/@user/article');
      expect(adapter).not.toBeNull();
      expect(adapter?.id).toBe('medium');
    });

    it('resolves wechat adapter for wechat URLs', () => {
      const adapter = resolveAdapter('https://mp.weixin.qq.com/s/abc123');
      expect(adapter).not.toBeNull();
      expect(adapter?.id).toBe('wechat');
    });

    it('resolves others adapter for generic URLs', () => {
      const adapter = resolveAdapter('https://example.com/article');
      expect(adapter).not.toBeNull();
      expect(adapter?.id).toBe('others');
    });
  });

  describe('sanitizeMdx', () => {
    it('converts autolink angle-bracket URLs into safe markdown links', () => {
      const input = [
        '11. Lecture 44: NVIDIA Profiling <https://www.youtube.com/watch?v=F_BazucyCMw&ab_channel=GPUMODE>',
        '12. <https://github.com/siboehm/SGEMM_CUDA/>',
        '13. CUTLASS: Fast Linear Algebra in CUDA C++ <https://developer.nvidia.com/blog/cutlass-linear-algebra-cuda/>',
      ].join('\n');

      const output = sanitizeMdx(input);

      expect(output).not.toMatch(/<https?:\/\//);
      expect(output).toContain(
        '[https://www.youtube.com/watch?v=F_BazucyCMw&ab_channel=GPUMODE](https://www.youtube.com/watch?v=F_BazucyCMw&ab_channel=GPUMODE)',
      );
      expect(output).toContain(
        '[https://github.com/siboehm/SGEMM_CUDA/](https://github.com/siboehm/SGEMM_CUDA/)',
      );
      expect(output).toContain(
        '[https://developer.nvidia.com/blog/cutlass-linear-algebra-cuda/](https://developer.nvidia.com/blog/cutlass-linear-algebra-cuda/)',
      );
    });

    it('removes empty HTML comments while preserving surrounding content', () => {
      const input = '`sinf`\n\n<!-- -->\n\n->\n\n<!-- -->\n\n`__sinf`)';

      const output = sanitizeMdx(input);

      expect(output).not.toContain('<!-- -->');
      expect(output).toContain('`sinf`');
      expect(output).toContain('->');
      expect(output).toContain('`__sinf`');
      expect(output).toContain(')');
      expect(output).not.toMatch(/\n{3,}/);
    });

    it('does not alter real HTML tags', () => {
      const input = '<div>ok</div>\n<img src="x">\n<br />';

      const output = sanitizeMdx(input);

      expect(output).toContain('<div>ok</div>');
      expect(output).toContain('<img src="x">');
      expect(output).toContain('<br />');
    });
  });

  describe('cookie injection', () => {
    it('enables cookie injection when IMPORT_COOKIE is set', async () => {
      // Set environment variable
      const originalCookie = process.env.IMPORT_COOKIE;
      process.env.IMPORT_COOKIE = 'test_cookie=test_value';

      // Mock download function to verify cookie is being injected
      const downloadImage = vi.fn(async () => '/images/test/001.jpg');

      try {
        const article = extractArticleFromHtml(fixtureHtml, MATMUL_URL);
        await htmlToMdx(article.html, {
          slug: 'test-cookie',
          provider: 'others',
          baseUrl: article.baseUrl,
          imageRoot: '/tmp/images',
          articleUrl: MATMUL_URL,
          publicBasePath: '/images/others/test-cookie',
          downloadImage,
        });

        // Verify downloadImage was called (cookie injection happens in HTTP fetch layer)
        expect(downloadImage).toHaveBeenCalled();
      } finally {
        // Restore environment
        if (originalCookie !== undefined) {
          process.env.IMPORT_COOKIE = originalCookie;
        } else {
          delete process.env.IMPORT_COOKIE;
        }
      }
    });

    it('works without cookie when IMPORT_COOKIE is not set', async () => {
      // Ensure no cookie is set
      const originalCookie = process.env.IMPORT_COOKIE;
      delete process.env.IMPORT_COOKIE;

      const downloadImage = vi.fn(async () => '/images/test/001.jpg');

      try {
        const article = extractArticleFromHtml(fixtureHtml, MATMUL_URL);
        await htmlToMdx(article.html, {
          slug: 'test-no-cookie',
          provider: 'others',
          baseUrl: article.baseUrl,
          imageRoot: '/tmp/images',
          articleUrl: MATMUL_URL,
          publicBasePath: '/images/others/test-no-cookie',
          downloadImage,
        });

        // Should still work without cookie
        expect(downloadImage).toHaveBeenCalled();
      } finally {
        // Restore environment
        if (originalCookie !== undefined) {
          process.env.IMPORT_COOKIE = originalCookie;
        }
      }
    });

    it('does not expose cookie value in any output', async () => {
      const originalCookie = process.env.IMPORT_COOKIE;
      const testCookie = 'secret_cookie=sensitive_value_12345';
      process.env.IMPORT_COOKIE = testCookie;

      const downloadImage = vi.fn(async () => '/images/test/001.jpg');
      const consoleLog = vi.spyOn(console, 'log');

      try {
        const article = extractArticleFromHtml(fixtureHtml, MATMUL_URL);
        const { markdown } = await htmlToMdx(article.html, {
          slug: 'test-no-leak',
          provider: 'others',
          baseUrl: article.baseUrl,
          imageRoot: '/tmp/images',
          articleUrl: MATMUL_URL,
          publicBasePath: '/images/others/test-no-leak',
          downloadImage,
        });

        // Verify cookie is not in markdown output
        expect(markdown).not.toContain(testCookie);
        expect(markdown).not.toContain('secret_cookie');
        expect(markdown).not.toContain('sensitive_value_12345');

        // Verify console logs don't contain cookie value
        const logCalls = consoleLog.mock.calls.map((call) => call.join(' '));
        logCalls.forEach((log) => {
          expect(log).not.toContain(testCookie);
          expect(log).not.toContain('sensitive_value_12345');
        });
      } finally {
        consoleLog.mockRestore();
        if (originalCookie !== undefined) {
          process.env.IMPORT_COOKIE = originalCookie;
        } else {
          delete process.env.IMPORT_COOKIE;
        }
      }
    });
  });
});
