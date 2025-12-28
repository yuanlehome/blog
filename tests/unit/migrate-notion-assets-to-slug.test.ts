import fs from 'fs';
import path from 'path';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { MigrationTarget } from '../../scripts/migrate-notion-assets-to-slug';

let tmpDir: string;

beforeEach(() => {
  tmpDir = fs.mkdtempSync(path.join(process.cwd(), 'tmp-migrate-'));
  vi.resetModules();
});

afterEach(() => {
  fs.rmSync(tmpDir, { recursive: true, force: true });
  delete process.env.NOTION_CONTENT_DIR;
  delete process.env.NOTION_PUBLIC_IMG_DIR;
});

describe('migrate notion assets helpers', () => {
  it('replaces notion image paths from multiple aliases', async () => {
    const { replaceNotionImagePaths } = await import('../../scripts/migrate-notion-assets-to-slug');
    const content =
      '![img](/images/notion/old/a.png) and ![b](/images/notion/id/b.png) and absolute';
    const result = replaceNotionImagePaths(content, new Set(['old', 'id']), 'new-slug');
    expect(result).toContain('/images/notion/new-slug/a.png');
    expect(result).toContain('/images/notion/new-slug/b.png');
    expect(result.includes('old')).toBe(false);
    expect(result.includes('id/b.png')).toBe(false);
  });

  it('moves directory contents idempotently', async () => {
    const { moveDirContents } = await import('../../scripts/migrate-notion-assets-to-slug');
    const source = path.join(tmpDir, 'src');
    const target = path.join(tmpDir, 'dest');
    fs.mkdirSync(source, { recursive: true });
    fs.writeFileSync(path.join(source, 'file.txt'), 'data');
    fs.mkdirSync(path.join(source, 'nested'), { recursive: true });
    fs.writeFileSync(path.join(source, 'nested', 'nested.txt'), 'nested');

    await moveDirContents(source, target, false);
    expect(fs.existsSync(path.join(target, 'file.txt'))).toBe(true);
    expect(fs.existsSync(source)).toBe(false);

    // run again to ensure idempotent
    await moveDirContents(source, target, false);
    expect(fs.existsSync(path.join(target, 'file.txt'))).toBe(true);
  });

  it('supports dry-run moves without touching files', async () => {
    const { moveDirContents } = await import('../../scripts/migrate-notion-assets-to-slug');
    const source = path.join(tmpDir, 'src-dry');
    const target = path.join(tmpDir, 'dest-dry');
    fs.mkdirSync(source, { recursive: true });
    fs.writeFileSync(path.join(source, 'file.txt'), 'data');

    await moveDirContents(source, target, true);
    expect(fs.existsSync(source)).toBe(true);
    expect(fs.existsSync(target)).toBe(false);
  });

  it('migrates targets and rewrites frontmatter and paths', async () => {
    const imagesDir = path.join(tmpDir, 'images');
    const contentDir = path.join(tmpDir, 'content');
    fs.mkdirSync(imagesDir, { recursive: true });
    fs.mkdirSync(contentDir, { recursive: true });
    const sourceImgDir = path.join(imagesDir, 'nid');
    fs.mkdirSync(sourceImgDir, { recursive: true });
    fs.writeFileSync(path.join(sourceImgDir, 'img.png'), 'img');

    process.env.NOTION_CONTENT_DIR = contentDir;
    process.env.NOTION_PUBLIC_IMG_DIR = imagesDir;
    const mod = await import('../../scripts/migrate-notion-assets-to-slug');
    const target: MigrationTarget = {
      filePath: path.join(contentDir, 'old.md'),
      dir: contentDir,
      slug: 'old',
      desiredSlug: 'new-slug',
      notionId: 'nid',
      cover: '/images/notion/nid/img.png',
      content: '![img](/images/notion/nid/img.png)',
      data: { title: 'Title', slug: 'old', notionId: 'nid' },
      aliases: new Set(['nid', 'old']),
    };
    fs.writeFileSync(target.filePath, 'content');

    await mod.migrateTargets([target], false);
    const newPath = path.join(contentDir, 'new-slug.md');
    expect(fs.existsSync(newPath)).toBe(true);
    const fm = (await import('gray-matter')).read(newPath).data;
    expect(fm.notion.id).toBe('nid');
    expect(String(fm.cover)).toContain('/images/notion/new-slug/');
    expect(fs.existsSync(path.join(imagesDir, 'new-slug', 'img.png'))).toBe(true);
    expect(fs.existsSync(sourceImgDir)).toBe(false);
  });

  it('supports dry-run migration without writing files', async () => {
    const imagesDir = path.join(tmpDir, 'images-dry');
    const contentDir = path.join(tmpDir, 'content-dry');
    fs.mkdirSync(imagesDir, { recursive: true });
    fs.mkdirSync(contentDir, { recursive: true });

    process.env.NOTION_CONTENT_DIR = contentDir;
    process.env.NOTION_PUBLIC_IMG_DIR = imagesDir;
    const mod = await import('../../scripts/migrate-notion-assets-to-slug');
    const target: MigrationTarget = {
      filePath: path.join(contentDir, 'old.md'),
      dir: contentDir,
      slug: 'old',
      desiredSlug: 'new-slug',
      notionId: 'nid',
      cover: '/images/notion/nid/img.png',
      content: '![img](/images/notion/nid/img.png)',
      data: { title: 'Title', slug: 'old', notionId: 'nid' },
      aliases: new Set(['nid', 'old']),
    };

    await mod.migrateTargets([target], true);
    expect(fs.existsSync(path.join(contentDir, 'new-slug.md'))).toBe(false);
  });
});
