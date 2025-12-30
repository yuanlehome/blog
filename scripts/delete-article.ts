import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';
import matter from 'gray-matter';
import { BLOG_CONTENT_DIR, PUBLIC_DIR, PUBLIC_IMAGES_DIR, ROOT_DIR } from '../src/config/paths';
import { createScriptLogger, now, duration } from './logger-helpers.js';

type Options = {
  target: string;
  deleteImages: boolean;
  dryRun: boolean;
};

const REPO_ROOT = ROOT_DIR;
const BLOG_ROOT = BLOG_CONTENT_DIR;
const PUBLIC_ROOT = PUBLIC_DIR;
const IMAGES_ROOT = PUBLIC_IMAGES_DIR;

// Maximum number of image directories that can be matched before requiring confirmation
// This prevents accidental mass deletion due to overly broad slug matches
const MAX_IMAGE_DIRS_MATCH = 20;

function toBoolean(value?: string | boolean): boolean | undefined {
  if (typeof value === 'boolean') return value;
  if (typeof value !== 'string') return undefined;
  const normalized = value.trim().toLowerCase();
  if (['true', '1', 'yes', 'y', 'on'].includes(normalized)) return true;
  if (['false', '0', 'no', 'n', 'off'].includes(normalized)) return false;
  return undefined;
}

function isPathTarget(target: string): boolean {
  return target.includes('/') || target.endsWith('.md');
}

async function fileExists(filePath: string): Promise<boolean> {
  try {
    await fs.access(filePath);
    return true;
  } catch {
    return false;
  }
}

async function extractCoverPath(articlePath: string): Promise<string | undefined> {
  const content = await fs.readFile(articlePath, 'utf-8');
  const parsed = matter(content);
  const cover = parsed.data?.cover;
  if (typeof cover !== 'string') return undefined;
  const trimmed = cover.trim();
  if (!trimmed) return undefined;
  try {
    new URL(trimmed);
    return undefined;
  } catch {
    // Not a fully qualified URL; continue.
  }

  // Normalize leading slashes to keep resolution inside the repo.
  const normalized = trimmed.replace(/^\/+/, '');
  const resolved = path.resolve(REPO_ROOT, normalized);
  if (resolved === PUBLIC_ROOT || resolved.startsWith(`${PUBLIC_ROOT}${path.sep}`)) {
    return resolved;
  }
  return undefined;
}

async function findArticlesBySlug(slug: string, dir: string): Promise<string[]> {
  const matches: string[] = [];
  const entries = await fs.readdir(dir, { withFileTypes: true });

  for (const entry of entries) {
    const entryPath = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      matches.push(...(await findArticlesBySlug(slug, entryPath)));
    } else if (entry.isFile() && entry.name.endsWith('.md')) {
      if (path.basename(entry.name, '.md') === slug) {
        matches.push(entryPath);
      }
    }
  }

  return matches;
}

function ensureInsideBlog(articlePath: string) {
  const resolved = path.resolve(articlePath);
  const blogWithSep = `${BLOG_ROOT}${path.sep}`;
  if (resolved !== BLOG_ROOT && resolved.startsWith(blogWithSep)) return;
  throw new Error(`Target must be inside ${BLOG_ROOT}`);
}

async function resolveArticle(target: string): Promise<{ articlePath: string; slug: string }> {
  if (isPathTarget(target)) {
    const resolvedPath = path.resolve(REPO_ROOT, target);
    ensureInsideBlog(resolvedPath);

    if (path.extname(resolvedPath) !== '.md') {
      throw new Error('Only .md articles are supported for deletion.');
    }

    if (!(await fileExists(resolvedPath))) {
      throw new Error(`Article not found at path: ${resolvedPath}`);
    }

    return { articlePath: resolvedPath, slug: path.basename(resolvedPath, '.md') };
  }

  const matches = await findArticlesBySlug(target, BLOG_ROOT);
  if (matches.length === 0) {
    throw new Error(`No article found for slug "${target}". Please provide a valid path or slug.`);
  }
  if (matches.length > 1) {
    const list = matches.map((articlePath) => path.relative(REPO_ROOT, articlePath)).join('\n - ');
    throw new Error(
      `Multiple articles matched slug "${target}". Please provide an explicit path.\n - ${list}`,
    );
  }

  return { articlePath: matches[0], slug: target };
}

function parseOptions(): Options {
  const args = process.argv.slice(2);
  let target: string | undefined;
  let deleteImages: boolean | undefined;
  let dryRun: boolean | undefined;

  for (let i = 0; i < args.length; i += 1) {
    const arg = args[i];
    if (arg === '--target') {
      const next = args[i + 1];
      if (next && !next.startsWith('--')) {
        target = next;
        i += 1;
      } else {
        throw new Error('Missing value for --target');
      }
    } else if (arg === '--delete-images') {
      deleteImages = true;
    } else if (arg === '--dry-run') {
      dryRun = true;
    }
  }

  if (!target && process.env.TARGET) {
    target = process.env.TARGET;
  }

  const envDeleteImages = toBoolean(process.env.DELETE_IMAGES);
  if (deleteImages === undefined && envDeleteImages !== undefined) {
    deleteImages = envDeleteImages;
  }

  const envDryRun = toBoolean(process.env.DRY_RUN);
  if (dryRun === undefined && envDryRun !== undefined) {
    dryRun = envDryRun;
  }

  if (!target) {
    throw new Error('Missing required target. Provide --target or TARGET env.');
  }

  return {
    target,
    deleteImages: deleteImages ?? false,
    dryRun: dryRun ?? false,
  };
}

async function removeFile(targetPath: string, dryRun: boolean) {
  const rel = path.relative(REPO_ROOT, targetPath);
  if (dryRun) {
    console.log(`[dry-run] Would delete file: ${rel}`);
    return;
  }
  await fs.rm(targetPath);
  console.log(`Deleted file: ${rel}`);
}

async function removeDirectory(targetPath: string, dryRun: boolean) {
  const rel = path.relative(REPO_ROOT, targetPath);
  if (!(await fileExists(targetPath))) {
    console.log(`Skip missing directory: ${rel}`);
    return;
  }
  if (dryRun) {
    console.log(`[dry-run] Would delete directory: ${rel}`);
    return;
  }
  await fs.rm(targetPath, { recursive: true, force: true });
  console.log(`Deleted directory: ${rel}`);
}

/**
 * Check if a directory basename matches the slug.
 * Matches if basename equals slug OR starts with "slug-"
 */
export function matchesSlugPattern(slug: string, basename: string): boolean {
  return basename === slug || basename.startsWith(`${slug}-`);
}

/**
 * Find all image directories matching the slug.
 * Matches directories where:
 * - basename === slug (exact match)
 * - basename.startsWith(slug + '-') (prefix match for conflict suffixes)
 */
async function findImageDirsBySlug(slug: string): Promise<string[]> {
  const matches: string[] = [];

  // Ensure IMAGES_ROOT exists
  if (!(await fileExists(IMAGES_ROOT))) {
    return matches;
  }

  const imagesRootResolved = path.resolve(IMAGES_ROOT);
  const imagesWithSep = `${imagesRootResolved}${path.sep}`;

  // Check if path is inside IMAGES_ROOT
  function isInsideImagesRoot(resolved: string): boolean {
    return resolved === imagesRootResolved || resolved.startsWith(imagesWithSep);
  }

  // Recursively scan IMAGES_ROOT for matching directories
  async function scanDir(dir: string) {
    const resolvedDir = path.resolve(dir);
    if (!isInsideImagesRoot(resolvedDir)) {
      return;
    }

    const entries = await fs.readdir(dir, { withFileTypes: true });
    for (const entry of entries) {
      if (!entry.isDirectory()) continue;

      const entryPath = path.join(dir, entry.name);
      const basename = entry.name;

      // Match logic: exact match or starts with "slug-"
      if (matchesSlugPattern(slug, basename)) {
        matches.push(entryPath);
      }

      // Recursively scan subdirectories
      await scanDir(entryPath);
    }
  }

  await scanDir(IMAGES_ROOT);
  return matches;
}

async function main() {
  const scriptStart = now();
  const options = parseOptions();
  const logger = createScriptLogger('delete-article', { target: options.target });
  
  logger.info('Starting article deletion', {
    target: options.target,
    deleteImages: options.deleteImages,
    dryRun: options.dryRun,
  });

  try {
    const resolveSpan = logger.time('resolve-article');
    let articlePath: string;
    let slug: string;
    try {
      const result = await resolveArticle(options.target);
      articlePath = result.articlePath;
      slug = result.slug;
      const relativeArticle = path.relative(REPO_ROOT, articlePath);
      resolveSpan.end({ status: 'ok', fields: { articlePath: relativeArticle, slug } });
      logger.info('Resolved article', { articlePath: relativeArticle, slug });
    } catch (error) {
      resolveSpan.end({ status: 'fail' });
      throw error;
    }

    const coverPath = await extractCoverPath(articlePath);
    if (coverPath) {
      logger.debug('Detected cover in public/', { coverPath: path.relative(REPO_ROOT, coverPath) });
    }

    const deleteSpan = logger.time('delete-files');
    let deletedFiles = 0;
    let deletedDirs = 0;

    try {
      await removeFile(articlePath, options.dryRun);
      deletedFiles++;

      if (options.deleteImages) {
        // Find all matching image directories
        const imageDirs = await findImageDirsBySlug(slug);

        if (imageDirs.length === 0) {
          logger.info('No image directories found for slug', { slug });
        } else {
          // Safety check: prevent deletion of too many directories
          if (imageDirs.length > MAX_IMAGE_DIRS_MATCH) {
            throw new Error(
              `Too many image directories matched (${imageDirs.length} > ${MAX_IMAGE_DIRS_MATCH}). ` +
                `This may indicate an overly broad match pattern. ` +
                `Matched directories:\n${imageDirs.map((d) => path.relative(REPO_ROOT, d)).join('\n')}`,
            );
          }

          logger.info('Matched image directories', {
            count: imageDirs.length,
            directories: imageDirs.map((d) => path.relative(REPO_ROOT, d)),
          });

          // Delete matched directories
          for (const dir of imageDirs) {
            await removeDirectory(dir, options.dryRun);
            deletedDirs++;
          }
        }

        // Delete cover file if it exists
        if (coverPath) {
          if (await fileExists(coverPath)) {
            await removeFile(coverPath, options.dryRun);
            deletedFiles++;
          } else {
            logger.debug('Cover file not found, skipping', {
              coverPath: path.relative(REPO_ROOT, coverPath),
            });
          }
        }
      } else {
        logger.info('Delete images disabled');
      }

      deleteSpan.end({
        status: 'ok',
        fields: { deletedFiles, deletedDirs },
      });
    } catch (error) {
      deleteSpan.end({ status: 'fail' });
      throw error;
    }

    if (options.dryRun) {
      logger.info('Dry-run complete, no files were deleted');
      logger.summary({
        status: 'ok',
        durationMs: duration(scriptStart),
        dryRun: true,
        slug,
        deletedFiles: 0,
        deletedDirs: 0,
      });
    } else {
      logger.info('Deletion completed successfully');
      logger.summary({
        status: 'ok',
        durationMs: duration(scriptStart),
        slug,
        deletedFiles,
        deletedDirs,
      });
    }
  } catch (error) {
    logger.error('Delete article failed', { error });
    logger.summary({
      status: 'fail',
      durationMs: duration(scriptStart),
      error: error instanceof Error ? error.message : String(error),
    });
    throw error;
  }
}

// Only run main if this script is executed directly (not imported during tests)
if (process.env.NODE_ENV !== 'test') {
  // Check if this module is being run directly
  const isMain =
    process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url);
  if (isMain) {
    main().catch((error: unknown) => {
      const message = error instanceof Error ? error.message : String(error);
      console.error(`Delete article failed: ${message}`);
      process.exit(1);
    });
  }
}
