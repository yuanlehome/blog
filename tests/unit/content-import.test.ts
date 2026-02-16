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

  describe('failed image downloads', () => {
    it('handles failed image downloads gracefully (e.g., HTTP 403)', async () => {
      const html = `
        <div>
          <p>Article content with images</p>
          <img src="https://example.com/image1.jpg" alt="Success" />
          <img src="https://blocked-oss.example.com/403-image.svg" alt="Blocked" />
          <img src="https://example.com/image3.png" alt="Another Success" />
        </div>
      `;

      // Mock downloadImage that simulates a 403 failure for one image
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
          // Simulate 403 failure for blocked OSS URL
          if (imageUrl.includes('blocked-oss')) {
            return null; // Failed download
          }
          
          const base = publicBasePath || `/images/others/${slug}`;
          return path.posix.join(
            base,
            `${String(index + 1).padStart(3, '0')}-${path.basename(new URL(imageUrl).pathname)}`,
          );
        },
      );

      const { markdown, images } = await htmlToMdx(html, {
        slug: 'test-article',
        provider: 'others',
        baseUrl: 'https://example.com',
        imageRoot: '/tmp/images',
        publicBasePath: '/images/others/test-article',
        downloadImage,
      });

      // Verify downloadImage was called for all 3 images
      expect(downloadImage).toHaveBeenCalledTimes(3);

      // Only 2 images should be successfully downloaded (403 blocked one should fail)
      expect(images).toHaveLength(2);
      expect(images[0]).toContain('001-image1.jpg');
      expect(images[1]).toContain('002-image3.png');

      // Failed image should not be in the images array
      expect(images.some((img) => img.includes('403-image'))).toBe(false);

      // Markdown should contain the successful images
      expect(markdown).toContain('/images/others/test-article/001-image1.jpg');
      expect(markdown).toContain('/images/others/test-article/002-image3.png');
    });

    it('continues processing when all images fail to download', async () => {
      const html = `
        <div>
          <p>Article content</p>
          <img src="https://blocked1.example.com/image1.jpg" alt="Blocked 1" />
          <img src="https://blocked2.example.com/image2.jpg" alt="Blocked 2" />
        </div>
      `;

      // Mock downloadImage that always fails (simulating all 403 errors)
      const downloadImage = vi.fn(async () => null);

      const { markdown, images } = await htmlToMdx(html, {
        slug: 'failed-article',
        provider: 'others',
        baseUrl: 'https://example.com',
        imageRoot: '/tmp/images',
        publicBasePath: '/images/others/failed-article',
        downloadImage,
      });

      // Verify downloadImage was called
      expect(downloadImage).toHaveBeenCalledTimes(2);

      // No images should be downloaded
      expect(images).toHaveLength(0);

      // Markdown should still be generated with article content
      expect(markdown).toContain('Article content');
      expect(markdown.length).toBeGreaterThan(0);
    });
  });

  describe('slug consistency for images', () => {
    it('handles consistent slug for image paths (tempSlug == finalSlug)', async () => {
      const article = extractArticleFromHtml(fixtureHtml, MATMUL_URL);
      const slug = 'matrix-multiplication-from-first-principles';

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
        slug,
        provider: 'others',
        baseUrl: article.baseUrl,
        imageRoot: '/tmp/images',
        articleUrl: MATMUL_URL,
        publicBasePath: `/images/others/${slug}`,
        downloadImage,
      });

      // All image paths should use the consistent slug
      expect(images.length).toBeGreaterThan(0);
      images.forEach((imgPath) => {
        expect(imgPath).toContain(`/images/others/${slug}/`);
        expect(imgPath).not.toContain('/images/others/others-');
      });

      // Markdown should contain correct image references
      const imageReferences = markdown.match(/!\[.*?\]\((\/images\/[^)]+)\)/g) || [];
      expect(imageReferences.length).toBeGreaterThan(0);
      imageReferences.forEach((ref) => {
        expect(ref).toContain(`/images/others/${slug}/`);
      });
    });

    it('handles different tempSlug vs finalSlug scenarios', async () => {
      // Simulate the bug scenario where URL path differs from title-derived slug
      const article = extractArticleFromHtml(fixtureHtml, MATMUL_URL);
      const tempSlug = 'matmul'; // from URL path
      const finalSlug = 'matrix-multiplication-from-first-principles'; // from title

      // First: simulate downloading with tempSlug
      const downloadImageWithTemp = vi.fn(
        async (
          imageUrl: string,
          _provider: string,
          _slug: string,
          _imageRoot: string,
          index: number,
        ) => {
          return `/images/others/${tempSlug}/${String(index + 1).padStart(3, '0')}-${path.basename(new URL(imageUrl).pathname)}`;
        },
      );

      const { markdown: markdownWithTemp, images: imagesWithTemp } = await htmlToMdx(article.html, {
        slug: tempSlug,
        provider: 'others',
        baseUrl: article.baseUrl,
        imageRoot: '/tmp/images',
        articleUrl: MATMUL_URL,
        publicBasePath: `/images/others/${tempSlug}`,
        downloadImage: downloadImageWithTemp,
      });

      // Verify images initially use tempSlug
      expect(imagesWithTemp.length).toBeGreaterThan(0);
      imagesWithTemp.forEach((imgPath) => {
        expect(imgPath).toContain(`/images/others/${tempSlug}/`);
      });

      // Simulate the migration: rewrite paths from tempSlug to finalSlug
      const tempPublicPath = `/images/others/${tempSlug}`;
      const finalPublicPath = `/images/others/${finalSlug}`;
      const oldPathPattern = new RegExp(tempPublicPath.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), 'g');
      const migratedMarkdown = markdownWithTemp.replace(oldPathPattern, finalPublicPath);
      const migratedImages = imagesWithTemp.map((imgPath) =>
        imgPath.replace(tempPublicPath, finalPublicPath),
      );

      // After migration: all paths should use finalSlug
      expect(migratedImages.length).toBeGreaterThan(0);
      migratedImages.forEach((imgPath) => {
        expect(imgPath).toContain(`/images/others/${finalSlug}/`);
        expect(imgPath).not.toContain(`/images/others/${tempSlug}/`);
      });

      // Markdown should have updated references
      const imageReferences = migratedMarkdown.match(/!\[.*?\]\((\/images\/[^)]+)\)/g) || [];
      expect(imageReferences.length).toBeGreaterThan(0);
      imageReferences.forEach((ref) => {
        expect(ref).toContain(`/images/others/${finalSlug}/`);
        expect(ref).not.toContain(`/images/others/${tempSlug}/`);
      });
    });

    it('verifies image path rewriting is complete and accurate', async () => {
      const html = `
        <div>
          <p>Article content with images</p>
          <img src="https://example.com/image1.jpg" alt="First" />
          <img src="https://example.com/image2.png" alt="Second" />
          <p>More content</p>
          <img src="https://example.com/image3.gif" alt="Third" />
        </div>
      `;

      const tempSlug = 'temp-article';
      const finalSlug = 'final-article-title';

      const downloadImage = vi.fn(
        async (
          _url: string,
          _provider: string,
          _slug: string,
          _imageRoot: string,
          index: number,
        ) => {
          return `/images/others/${tempSlug}/${String(index + 1).padStart(3, '0')}-image${index + 1}.jpg`;
        },
      );

      const { markdown, images } = await htmlToMdx(html, {
        slug: tempSlug,
        provider: 'others',
        baseUrl: 'https://example.com',
        imageRoot: '/tmp/images',
        publicBasePath: `/images/others/${tempSlug}`,
        downloadImage,
      });

      // Verify all 3 images were processed with tempSlug
      expect(images).toHaveLength(3);
      expect(images[0]).toBe(`/images/others/${tempSlug}/001-image1.jpg`);
      expect(images[1]).toBe(`/images/others/${tempSlug}/002-image2.jpg`);
      expect(images[2]).toBe(`/images/others/${tempSlug}/003-image3.jpg`);

      // Simulate migration
      const migratedMarkdown = markdown.replace(
        new RegExp(`/images/others/${tempSlug}`, 'g'),
        `/images/others/${finalSlug}`,
      );

      // Verify no tempSlug remains in markdown
      expect(migratedMarkdown).not.toContain(`/images/others/${tempSlug}`);
      expect(migratedMarkdown).toContain(`/images/others/${finalSlug}`);

      // Count occurrences - should be exactly 3 (one per image)
      const finalSlugMatches = migratedMarkdown.match(
        new RegExp(`/images/others/${finalSlug}`, 'g'),
      );
      expect(finalSlugMatches).toHaveLength(3);
    });
  });
});
