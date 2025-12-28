import fs from 'fs/promises';
import path from 'path';
import matter from 'gray-matter';
import { fileURLToPath } from 'url';
import { deriveSlug, ensureUniqueSlug } from './slug';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const REPO_ROOT = path.resolve(__dirname, '..');
const NOTION_CONTENT_DIR =
  process.env.NOTION_CONTENT_DIR && process.env.NOTION_CONTENT_DIR.trim().length > 0
    ? path.resolve(process.env.NOTION_CONTENT_DIR)
    : path.join(REPO_ROOT, 'src', 'content', 'blog', 'notion');
const NOTION_IMAGES_DIR =
  process.env.NOTION_PUBLIC_IMG_DIR && process.env.NOTION_PUBLIC_IMG_DIR.trim().length > 0
    ? path.resolve(process.env.NOTION_PUBLIC_IMG_DIR)
    : path.join(REPO_ROOT, 'public', 'images', 'notion');

export type MigrationTarget = {
  filePath: string;
  dir: string;
  slug: string;
  desiredSlug: string;
  notionId?: string;
  cover?: string;
  content: string;
  data: Record<string, any>;
  aliases: Set<string>;
};

function parseArgs() {
  const dryRun = process.argv.includes('--dry-run');
  return { dryRun };
}

async function fileExists(p: string) {
  try {
    await fs.access(p);
    return true;
  } catch {
    return false;
  }
}

export async function moveDirContents(source: string, target: string, dryRun: boolean) {
  if (source === target) return;
  if (!(await fileExists(source))) return;
  if (dryRun) {
    console.log(
      `[dry-run] Would move ${path.relative(REPO_ROOT, source)} -> ${path.relative(REPO_ROOT, target)}`,
    );
    return;
  }
  await fs.mkdir(target, { recursive: true });
  const entries = await fs.readdir(source, { withFileTypes: true });
  for (const entry of entries) {
    const from = path.join(source, entry.name);
    const to = path.join(target, entry.name);
    if (await fileExists(to)) continue;
    if (entry.isDirectory()) {
      await moveDirContents(from, to, dryRun);
    } else if (entry.isFile()) {
      await fs.copyFile(from, to);
    }
  }
  await fs.rm(source, { recursive: true, force: true });
}

export function replaceNotionImagePaths(input: string, fromKeys: Set<string>, toSlug: string) {
  let output = input;
  for (const key of fromKeys) {
    if (!key) continue;
    const re = new RegExp(`/images/notion/${key}/`, 'g');
    output = output.replace(re, `/images/notion/${toSlug}/`);
  }
  return output;
}

async function loadTargets(): Promise<MigrationTarget[]> {
  const entries = await fs.readdir(NOTION_CONTENT_DIR);
  const targets: MigrationTarget[] = [];

  for (const entry of entries) {
    if (!entry.endsWith('.md')) continue;
    const filePath = path.join(NOTION_CONTENT_DIR, entry);
    const raw = await fs.readFile(filePath, 'utf-8');
    const parsed = matter(raw);
    const data = parsed.data as Record<string, any>;
    const notionId = data.notionId || data.notion?.id;
    const desiredSlug = deriveSlug({
      explicitSlug: data.slug,
      title: data.title,
      fallbackId: notionId || path.basename(entry, '.md'),
    });
    const target: MigrationTarget = {
      filePath,
      dir: NOTION_CONTENT_DIR,
      slug: data.slug || path.basename(entry, '.md'),
      desiredSlug,
      notionId,
      cover: data.cover,
      content: parsed.content,
      data,
      aliases: new Set<string>(),
    };
    if (notionId) target.aliases.add(notionId);
    target.aliases.add(target.slug);
    targets.push(target);
  }

  return targets;
}

export async function migrateTargets(targets: MigrationTarget[], dryRun: boolean) {
  const used = new Map<string, string>();

  for (const target of targets) {
    try {
      const ownerId = target.notionId || path.basename(target.filePath);
      const slug = ensureUniqueSlug(target.desiredSlug, ownerId, used);
      const aliases = new Set(target.aliases);
      target.slug = slug;

      // Move image directories
      for (const alias of aliases) {
        const source = path.join(NOTION_IMAGES_DIR, alias);
        const dest = path.join(NOTION_IMAGES_DIR, slug);
        await moveDirContents(source, dest, dryRun);
      }

      // Rewrite content paths
      let cover = target.cover;
      if (typeof cover === 'string') {
        const replaced = replaceNotionImagePaths(cover, aliases, slug);
        if (replaced !== cover) cover = replaced;
      }

      const updatedContent = replaceNotionImagePaths(target.content, aliases, slug);

      // Update frontmatter
      const data = { ...target.data };
      delete (data as any).notionId;
      data.slug = slug;
      if (target.notionId) {
        data.notion = { id: target.notionId };
      }
      if (cover !== undefined) {
        data.cover = cover;
      }

      const newFileContent = matter.stringify(updatedContent, data);
      const newPath = path.join(target.dir, `${slug}.md`);

      if (dryRun) {
        console.log(`[dry-run] Would write ${path.relative(REPO_ROOT, newPath)}`);
        if (newPath !== target.filePath) {
          console.log(
            `[dry-run] Would remove old file ${path.relative(REPO_ROOT, target.filePath)}`,
          );
        }
        continue;
      }

      await fs.writeFile(newPath, newFileContent);
      if (newPath !== target.filePath && (await fileExists(target.filePath))) {
        await fs.rm(target.filePath);
      }
      console.log(`Migrated ${path.relative(REPO_ROOT, newPath)}`);
    } catch (error) {
      console.error(`Failed to migrate ${path.relative(REPO_ROOT, target.filePath)}:`, error);
      throw error;
    }
  }
}

async function main() {
  const { dryRun } = parseArgs();
  const targets = await loadTargets();
  await migrateTargets(targets, dryRun);
  console.log('Migration complete.');
}

const isMain = process.argv[1] && path.resolve(process.argv[1]) === __filename;
if (isMain && process.env.NODE_ENV !== 'test') {
  main().catch((error) => {
    console.error(error);
    process.exit(1);
  });
}
