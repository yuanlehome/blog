/**
 * Verification script to test image download functionality
 *
 * This script verifies that:
 * 1. Image directories are created correctly
 * 2. Images are downloaded with proper naming
 * 3. MDX files reference images correctly
 *
 * Usage:
 *   npx tsx scripts/verify-image-download.ts
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const IMAGE_ROOT = path.join(__dirname, '../public/images');
const CONTENT_ROOT = path.join(__dirname, '../src/content/blog');

interface VerificationResult {
  provider: string;
  articles: Array<{
    slug: string;
    mdxPath: string;
    imageDir: string;
    imageCount: number;
    imagesFound: string[];
    imagesReferenced: string[];
    hasMatchingImages: boolean;
  }>;
}

function extractImageReferencesFromMdx(mdxPath: string): string[] {
  const content = fs.readFileSync(mdxPath, 'utf-8');
  const imageRegex = /!\[.*?\]\((\/images\/[^)]+)\)|src=["'](\/images\/[^"']+)["']/g;
  const images: string[] = [];

  let match;
  while ((match = imageRegex.exec(content)) !== null) {
    images.push(match[1] || match[2]);
  }

  return images;
}

function verifyProvider(provider: string): VerificationResult {
  const providerContentDir = path.join(CONTENT_ROOT, provider);
  const providerImageDir = path.join(IMAGE_ROOT, provider);

  const result: VerificationResult = {
    provider,
    articles: [],
  };

  if (!fs.existsSync(providerContentDir)) {
    console.log(`⚠️  No content directory found for ${provider}`);
    return result;
  }

  const mdxFiles = fs
    .readdirSync(providerContentDir)
    .filter((f) => f.endsWith('.mdx') || f.endsWith('.md'));

  for (const mdxFile of mdxFiles) {
    const slug = path.basename(mdxFile, path.extname(mdxFile));
    const mdxPath = path.join(providerContentDir, mdxFile);
    const imageDir = path.join(providerImageDir, slug);

    const imagesReferenced = extractImageReferencesFromMdx(mdxPath);
    const imagesFound = fs.existsSync(imageDir) ? fs.readdirSync(imageDir) : [];

    const hasMatchingImages =
      imagesReferenced.length === 0 ||
      (imagesReferenced.length > 0 &&
        imagesReferenced.every((ref) => {
          // Extract filename from reference path
          const filename = ref.split('/').pop();
          return filename && imagesFound.includes(filename);
        }));

    result.articles.push({
      slug,
      mdxPath,
      imageDir,
      imageCount: imagesFound.length,
      imagesFound,
      imagesReferenced,
      hasMatchingImages,
    });
  }

  return result;
}

function main() {
  console.log('🔍 Verifying image download functionality...\n');

  const providers = ['wechat', 'zhihu', 'notion', 'medium'];
  const results: VerificationResult[] = [];

  for (const provider of providers) {
    const result = verifyProvider(provider);
    results.push(result);
  }

  // Print results
  let hasIssues = false;

  for (const result of results) {
    if (result.articles.length === 0) {
      continue;
    }

    console.log(`\n📁 Provider: ${result.provider}`);
    console.log('─'.repeat(60));

    for (const article of result.articles) {
      console.log(`\n  📄 Article: ${article.slug}`);
      console.log(`     Path: ${article.mdxPath}`);
      console.log(`     Images directory: ${article.imageDir}`);
      console.log(`     Images found: ${article.imageCount}`);

      if (article.imagesReferenced.length > 0) {
        console.log(`     Images referenced in MDX: ${article.imagesReferenced.length}`);

        if (article.hasMatchingImages) {
          console.log(`     ✅ All referenced images exist`);
        } else {
          console.log(`     ❌ Some referenced images are missing!`);
          console.log(`        Referenced: ${article.imagesReferenced.join(', ')}`);
          console.log(`        Found: ${article.imagesFound.join(', ')}`);
          hasIssues = true;
        }
      } else {
        console.log(`     ℹ️  No images referenced in this article`);
      }

      if (article.imageCount > 0) {
        console.log(`     📦 Downloaded images:`);
        article.imagesFound.slice(0, 5).forEach((img) => {
          console.log(`        - ${img}`);
        });
        if (article.imagesFound.length > 5) {
          console.log(`        ... and ${article.imagesFound.length - 5} more`);
        }
      }
    }
  }

  console.log('\n' + '='.repeat(60));
  if (hasIssues) {
    console.log('❌ Verification found issues - some images are missing');
    process.exit(1);
  } else {
    console.log('✅ All checks passed - image download functionality is working correctly');
    console.log('\n💡 To test with a new article, run:');
    console.log('   npm run import:content -- --url="<ARTICLE_URL>"');
    process.exit(0);
  }
}

main();
