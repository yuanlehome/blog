#!/usr/bin/env tsx
/**
 * Rename all .mdx files to .md in the content directory
 * This script is idempotent and can be run multiple times safely
 */
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const CONTENT_ROOT = path.join(__dirname, '../src/content/blog');

function findMdxFiles(dir: string): string[] {
  const results: string[] = [];
  const entries = fs.readdirSync(dir, { withFileTypes: true });

  for (const entry of entries) {
    const fullPath = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      results.push(...findMdxFiles(fullPath));
    } else if (entry.isFile() && entry.name.endsWith('.mdx')) {
      results.push(fullPath);
    }
  }

  return results;
}

function renameMdxToMd() {
  console.log('Starting MDX to MD rename process...');
  console.log(`Content root: ${CONTENT_ROOT}`);

  const mdxFiles = findMdxFiles(CONTENT_ROOT);

  if (mdxFiles.length === 0) {
    console.log('✅ No .mdx files found. All files are already .md');
    return;
  }

  console.log(`Found ${mdxFiles.length} .mdx file(s) to rename:`);

  let successCount = 0;
  let errorCount = 0;

  for (const mdxPath of mdxFiles) {
    const mdPath = mdxPath.replace(/\.mdx$/, '.md');
    const relativePath = path.relative(CONTENT_ROOT, mdxPath);

    try {
      // Check if target .md file already exists
      if (fs.existsSync(mdPath)) {
        console.warn(
          `⚠️  Skipping ${relativePath}: target ${path.basename(mdPath)} already exists`,
        );
        errorCount++;
        continue;
      }

      // Rename the file
      fs.renameSync(mdxPath, mdPath);
      console.log(`✅ Renamed: ${relativePath} → ${path.basename(mdPath)}`);
      successCount++;
    } catch (error) {
      console.error(`❌ Error renaming ${relativePath}:`, error);
      errorCount++;
    }
  }

  console.log('\n📊 Summary:');
  console.log(`  ✅ Successfully renamed: ${successCount}`);
  console.log(`  ❌ Errors/Skipped: ${errorCount}`);
  console.log(`  📁 Total processed: ${mdxFiles.length}`);

  if (successCount > 0) {
    console.log('\n✨ Rename completed successfully!');
  }
}

// Run the script
renameMdxToMd();
