import fs from 'fs';
import path from 'path';
import matter from 'gray-matter';
import { beforeEach, afterEach, describe, expect, it, vi } from 'vitest';

const pageId = 'page-123';
const imageBlockId = 'block-abc';
const imageUrl = 'https://example.com/first-image';

const mockPage: any = {
  id: pageId,
  last_edited_time: '2024-01-01T00:00:00.000Z',
  cover: null,
  properties: {
    Name: { title: [{ plain_text: 'Test Post' }] },
    tags: { multi_select: [] },
    date: { date: { start: '2024-01-02' } },
    Status: { type: 'select', select: { name: 'Published' } },
  },
};

const imageBlock: any = {
  object: 'block',
  id: imageBlockId,
  type: 'image',
  image: {
    type: 'external',
    external: { url: imageUrl },
    caption: [{ plain_text: 'caption' }],
  },
  has_children: false,
};

const blocksListMock = vi.fn();

vi.mock('@notionhq/client', () => {
  class Client {
    databases = {
      query: async () => ({ results: [mockPage] }),
    };
    blocks = {
      children: {
        list: (...args: any[]) => blocksListMock(...args),
      },
    };
    constructor() {
      // no-op
    }
  }
  return {
    Client,
  };
});

class NotionToMarkdownMock {
  private transformers: Record<string, (block: any) => any> = {};
  setCustomTransformer(type: string, fn: any) {
    this.transformers[type] = fn;
  }
  async pageToMarkdown() {
    return [imageBlock];
  }
  async toMarkdownString(blocks: any[]) {
    const parts = await Promise.all(
      blocks.map(async (block) => {
        const transformer = this.transformers[block.type];
        if (transformer) {
          return await transformer(block);
        }
        return '';
      }),
    );
    return { parent: parts.join('\n') };
  }
}

vi.mock('notion-to-md', () => ({
  NotionToMarkdown: NotionToMarkdownMock,
}));

describe('notion sync cover fallback', () => {
  const contentDir = path.join(process.cwd(), 'tmp/notion-content');
  const imageDir = path.join(process.cwd(), 'tmp/notion-images');

  beforeEach(() => {
    fs.rmSync(contentDir, { recursive: true, force: true });
    fs.rmSync(imageDir, { recursive: true, force: true });
    process.env.NOTION_CONTENT_DIR = contentDir;
    process.env.NOTION_PUBLIC_IMG_DIR = imageDir;
    process.env.NOTION_TOKEN = 'dummy';
    process.env.NOTION_DATABASE_ID = 'dummy-db';
    blocksListMock.mockReset();
    blocksListMock.mockResolvedValue({
      results: [imageBlock],
      has_more: false,
      next_cursor: null,
    });
    vi.resetModules();
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    delete process.env.NOTION_CONTENT_DIR;
    delete process.env.NOTION_PUBLIC_IMG_DIR;
    delete process.env.NOTION_TOKEN;
    delete process.env.NOTION_DATABASE_ID;
  });

  it('uses first body image as cover when explicit cover is missing', async () => {
    const imageBuffer = Buffer.from('image-data');
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(imageBuffer, {
        status: 200,
        headers: { 'content-type': 'image/jpeg' },
      }),
    );
    vi.stubGlobal('fetch', fetchMock);

    const { sync } = await import('../../scripts/notion-sync');
    await sync();

    const generatedFile = path.join(contentDir, 'test-post.md');
    expect(fs.existsSync(generatedFile)).toBe(true);
    const { data } = matter(fs.readFileSync(generatedFile, 'utf-8'));
    const expectedCover = `/images/notion/test-post/${imageBlockId}.jpg`;

    expect(data.cover).toBe(expectedCover);
    expect(fs.existsSync(path.join(imageDir, 'test-post', `${imageBlockId}.jpg`))).toBe(true);
    expect(fetchMock).toHaveBeenCalled();
  });

  it('preserves original publication date when updating post', async () => {
    const imageBuffer = Buffer.from('image-data');
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(imageBuffer, {
        status: 200,
        headers: { 'content-type': 'image/jpeg' },
      }),
    );
    vi.stubGlobal('fetch', fetchMock);

    // First sync - create the post
    const { sync } = await import('../../scripts/notion-sync');
    await sync();

    const generatedFile = path.join(contentDir, 'test-post.md');
    expect(fs.existsSync(generatedFile)).toBe(true);
    const firstSync = matter(fs.readFileSync(generatedFile, 'utf-8'));
    const originalDate = firstSync.data.date;
    expect(originalDate).toBe('2024-01-02'); // From mockPage.properties.date

    // Simulate an update by changing last_edited_time
    mockPage.last_edited_time = '2024-02-15T00:00:00.000Z';

    // Second sync - update the post
    vi.resetModules();
    const { sync: sync2 } = await import('../../scripts/notion-sync');
    await sync2();

    // Verify date is preserved but updated field changes
    const secondSync = matter(fs.readFileSync(generatedFile, 'utf-8'));
    expect(secondSync.data.date).toBe(originalDate); // Date should be preserved
    expect(secondSync.data.updated).toBe('2024-02-15T00:00:00.000Z'); // Updated field should change
  });
});
