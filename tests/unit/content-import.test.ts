import fs from 'fs';
import path from 'path';
import { describe, expect, it, vi } from 'vitest';
import { extractArticleFromHtml, htmlToMdx, sanitizeMdx } from '../../scripts/content-import';

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
});
