import fs from 'fs';
import path from 'path';

const resolveWhenSet = (value?: string | null) =>
  value && value.trim().length > 0 ? path.resolve(value) : undefined;

export const ROOT_DIR = path.resolve(process.env.PROJECT_ROOT ?? process.cwd());

export const CONTENT_ROOT = path.join(ROOT_DIR, 'src', 'content');
export const BLOG_CONTENT_DIR =
  resolveWhenSet(process.env.BLOG_CONTENT_DIR) ?? path.join(CONTENT_ROOT, 'blog');
export const NOTION_CONTENT_DIR =
  resolveWhenSet(process.env.NOTION_CONTENT_DIR) ?? path.join(BLOG_CONTENT_DIR, 'notion');

export const PUBLIC_DIR = path.join(ROOT_DIR, 'public');
export const PUBLIC_IMAGES_DIR =
  resolveWhenSet(process.env.PUBLIC_IMAGES_DIR) ?? path.join(PUBLIC_DIR, 'images');
export const NOTION_PUBLIC_IMG_DIR =
  resolveWhenSet(process.env.NOTION_PUBLIC_IMG_DIR) ?? path.join(PUBLIC_IMAGES_DIR, 'notion');

export const ARTIFACTS_DIR =
  resolveWhenSet(process.env.ARTIFACTS_DIR) ?? path.join(ROOT_DIR, 'artifacts');

export const DIST_DIR = path.join(ROOT_DIR, 'dist');

export const ensureDir = (dir: string) => {
  if (!dir) return;
  if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir, { recursive: true });
  }
};
