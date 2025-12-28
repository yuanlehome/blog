import fs from 'fs';
import path from 'path';
import { describe, expect, it, vi } from 'vitest';
import { extractArticleFromHtml, htmlToMdx } from '../../scripts/content-import';

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
        const base = publicBasePath || `/images/imported/${slug}`;
        return path.posix.join(
          base,
          `${String(index + 1).padStart(3, '0')}-${path.basename(new URL(imageUrl).pathname)}`,
        );
      },
    );

    const { markdown, images } = await htmlToMdx(article.html, {
      slug: 'matmul',
      provider: 'imported',
      baseUrl: article.baseUrl,
      imageRoot: '/tmp/images',
      articleUrl: MATMUL_URL,
      publicBasePath: '/images/imported/matmul',
      downloadImage,
    });

    expect(markdown).toMatchSnapshot();
    expect(images[0]).toContain('/images/imported/matmul/001');
    expect(downloadImage).toHaveBeenCalled();
  });
});
