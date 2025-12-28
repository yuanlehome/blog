import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';
import matter from 'gray-matter';

type Options = {
  target: string;
  deleteImages: boolean;
  dryRun: boolean;
};

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const REPO_ROOT = path.resolve(__dirname, '..');
const BLOG_ROOT = path.join(REPO_ROOT, 'src', 'content', 'blog');
const PUBLIC_ROOT = path.join(REPO_ROOT, 'public');
const IMPORTED_IMAGES_ROOT = path.join(PUBLIC_ROOT, 'images', 'imported');

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

async function main() {
  const options = parseOptions();
  console.log(`Starting delete-article with target="${options.target}"`);
  const { articlePath, slug } = await resolveArticle(options.target);
  const relativeArticle = path.relative(REPO_ROOT, articlePath);
  console.log(`Resolved article: ${relativeArticle}`);

  const coverPath = await extractCoverPath(articlePath);
  if (coverPath) {
    console.log(`Detected cover in public/: ${path.relative(REPO_ROOT, coverPath)}`);
  }

  await removeFile(articlePath, options.dryRun);

  if (options.deleteImages) {
    const imagesDir = path.join(IMPORTED_IMAGES_ROOT, slug);
    await removeDirectory(imagesDir, options.dryRun);
    if (coverPath) {
      if (await fileExists(coverPath)) {
        await removeFile(coverPath, options.dryRun);
      } else {
        console.log(`Cover file not found, skipping: ${path.relative(REPO_ROOT, coverPath)}`);
      }
    }
  } else {
    console.log('Skipping image deletion (delete-images disabled).');
  }

  if (options.dryRun) {
    console.log('Dry-run complete. No files were deleted.');
  } else {
    console.log('Deletion completed successfully.');
  }
}

main().catch((error: unknown) => {
  const message = error instanceof Error ? error.message : String(error);
  console.error(`Delete article failed: ${message}`);
  process.exit(1);
});
