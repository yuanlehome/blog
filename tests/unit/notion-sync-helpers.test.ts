import fs from 'fs';
import path from 'path';
import matter from 'gray-matter';
import { beforeEach, afterEach, describe, expect, it, vi } from 'vitest';
import { shortHash } from '../../src/lib/slug';

const queryMock = vi.fn();
const blocksListMock = vi.fn();
const lastN2mInstance: { current?: any } = {};

class NotionToMarkdownMock {
  transformers: Record<string, (block: any) => any> = {};
  blocks: any[] = [];
  constructor() {
    lastN2mInstance.current = this;
  }
  setCustomTransformer(type: string, fn: any) {
    this.transformers[type] = fn;
  }
  async pageToMarkdown() {
    return this.blocks;
  }
  async toMarkdownString(blocks: any[]) {
    const rendered = await Promise.all(
      blocks.map(async (block) => {
        const transformer = this.transformers[block.type];
        return transformer ? await transformer(block) : '';
      }),
    );
    return { parent: rendered.join('\n') };
  }
}

vi.mock('@notionhq/client', () => ({
  Client: class Client {
    databases = { query: (...args: any[]) => queryMock(...args) };
    blocks = { children: { list: (...args: any[]) => blocksListMock(...args) } };
  },
}));

vi.mock('notion-to-md', () => ({
  NotionToMarkdown: NotionToMarkdownMock,
}));

let tmpRoot: string;

beforeEach(() => {
  vi.resetModules();
  vi.clearAllMocks();
  tmpRoot = fs.mkdtempSync(path.join(process.cwd(), 'tmp-notion-'));
  process.env.NOTION_CONTENT_DIR = path.join(tmpRoot, 'content');
  process.env.NOTION_PUBLIC_IMG_DIR = path.join(tmpRoot, 'images');
  process.env.NOTION_TOKEN = 'token';
  process.env.NOTION_DATABASE_ID = 'db';
});

afterEach(() => {
  fs.rmSync(tmpRoot, { recursive: true, force: true });
  delete process.env.NOTION_CONTENT_DIR;
  delete process.env.NOTION_PUBLIC_IMG_DIR;
  delete process.env.NOTION_TOKEN;
  delete process.env.NOTION_DATABASE_ID;
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

describe('notion sync helpers', () => {
  it('fails fast when env variables are missing', async () => {
    delete process.env.NOTION_TOKEN;
    delete process.env.NOTION_DATABASE_ID;
    const exitMock = vi.spyOn(process, 'exit').mockImplementation(((code?: number) => {
      throw new Error(`exit ${code}`);
    }) as any);
    const consoleSpy = vi.spyOn(console, 'error').mockImplementation(() => {});

    await expect(import('../../scripts/notion-sync')).rejects.toThrow('exit 1');
    expect(consoleSpy).toHaveBeenCalled();

    exitMock.mockRestore();
    consoleSpy.mockRestore();
  });

  it('resolves extensions from url, content-type, or defaults', async () => {
    const mod = await import('../../scripts/notion-sync');
    expect(mod.resolveExtension('https://example.com/image.png')).toBe('.png');
    expect(mod.resolveExtension('https://example.com/raw', 'image/webp; charset=utf-8')).toBe(
      '.webp',
    );
    expect(mod.resolveExtension('https://example.com/raw')).toBe('.png');
  });

  it('returns existing image paths and logs when download fails', async () => {
    const mod = await import('../../scripts/notion-sync');
    const dir = path.join(process.env.NOTION_PUBLIC_IMG_DIR!, 'slug');
    fs.mkdirSync(dir, { recursive: true });
    const existing = path.join(dir, 'imageid.png');
    fs.writeFileSync(existing, 'data');

    const fetchMock = vi.fn().mockRejectedValue(new Error('network'));
    vi.stubGlobal('fetch', fetchMock);
    vi.useFakeTimers();

    const existingUrl = await mod.downloadImage('https://example.com/a', 'slug', 'imageid');
    expect(existingUrl).toBe('/images/notion/slug/imageid.png');
    expect(fetchMock).not.toHaveBeenCalled();

    const missingPromise = mod.downloadImage('https://example.com/b', 'slug', 'new-image');
    await vi.runAllTimersAsync();
    const missingUrl = await missingPromise;
    expect(missingUrl).toBeNull();
  });

  it('saves downloaded images with inferred extensions', async () => {
    const mod = await import('../../scripts/notion-sync');
    const buffer = Buffer.from('image');
    const fetchMock = vi
      .fn()
      .mockResolvedValue(
        new Response(buffer, { headers: { 'content-type': 'image/webp' }, status: 200 }),
      );
    vi.stubGlobal('fetch', fetchMock);

    const url = await mod.downloadImage('https://example.com/noext', 'slug', 'img-123');
    expect(url).toMatch(/img-123\.webp$/);
    const saved = fs.readdirSync(path.join(process.env.NOTION_PUBLIC_IMG_DIR!, 'slug'));
    expect(saved.some((f) => f.endsWith('.webp'))).toBe(true);
  });

  it('extracts cover urls from different property shapes', async () => {
    const mod = await import('../../scripts/notion-sync');
    expect(
      mod.extractCoverUrl({
        cover: { type: 'files', files: [{ type: 'external', external: { url: 'https://files' } }] },
      }),
    ).toBe('https://files');
    expect(
      mod.extractCoverUrl({
        Cover: { type: 'files', files: [{ type: 'file', file: { url: 'https://file' } }] },
      }),
    ).toBe('https://file');
    expect(mod.extractCoverUrl({ cover: { type: 'url', url: 'https://url' } })).toBe('https://url');
    expect(mod.extractCoverUrl({ cover: { type: 'file', file: { url: 'https://file2' } } })).toBe(
      'https://file2',
    );
    expect(
      mod.extractCoverUrl({
        cover: { type: 'rich_text', rich_text: [{ plain_text: 'https://rt' }] },
      }),
    ).toBe('https://rt');
    expect(mod.extractCoverUrl({})).toBeNull();
  });

  it('handles download errors with hashed identifiers and http failures', async () => {
    const mod = await import('../../scripts/notion-sync');
    const fetchMock = vi.fn().mockResolvedValue(new Response(Buffer.from('x'), { status: 500 }));
    vi.stubGlobal('fetch', fetchMock);
    vi.useFakeTimers();

    const resultPromise = mod.downloadImage('https://example.com/image', 'slug', '###');
    await vi.runAllTimersAsync();
    expect(await resultPromise).toBeNull();
    const files = fs.readdirSync(path.join(process.env.NOTION_PUBLIC_IMG_DIR!, 'slug'));
    expect(files.length).toBe(0);
  });

  it('finds first image block through recursion and pagination', async () => {
    queryMock.mockResolvedValue({ results: [] });
    blocksListMock
      .mockResolvedValueOnce({
        results: [{ id: 'p1', type: 'paragraph', has_children: false }],
        has_more: true,
        next_cursor: 'cursor-1',
      })
      .mockResolvedValueOnce({
        results: [
          {
            id: 'child',
            type: 'image',
            image: { type: 'external', external: { url: 'https://img' } },
          },
        ],
        has_more: false,
        next_cursor: null,
      })
      .mockResolvedValueOnce({
        results: [
          {
            id: 'nested',
            type: 'callout',
            has_children: true,
          },
        ],
        has_more: false,
        next_cursor: null,
      })
      .mockResolvedValueOnce({
        results: [
          {
            id: 'nested-img',
            type: 'image',
            image: { type: 'external', external: { url: 'https://nested' } },
          },
        ],
        has_more: false,
        next_cursor: null,
      });

    const mod = await import('../../scripts/notion-sync');
    const paged = await mod.findFirstImageBlock('root');
    expect(paged?.url).toBe('https://img');

    const nested = await mod.findFirstImageBlock('nested');
    expect(nested?.url).toBe('https://nested');
  });

  it('skips blocks without type in findFirstImageBlock', async () => {
    blocksListMock.mockResolvedValue({ results: [{}], has_more: false, next_cursor: null });
    const mod = await import('../../scripts/notion-sync');
    const result = await mod.findFirstImageBlock('root');
    expect(result).toBeNull();
  });

  it('handles child blocks without nested images', async () => {
    blocksListMock
      .mockResolvedValueOnce({
        results: [{ id: 'nested', type: 'paragraph', has_children: true }],
        has_more: false,
        next_cursor: null,
      })
      .mockResolvedValueOnce({ results: [], has_more: false, next_cursor: null });
    const mod = await import('../../scripts/notion-sync');
    const result = await mod.findFirstImageBlock('root');
    expect(result).toBeNull();
  });

  it('transforms image blocks using page context or falls back', async () => {
    const mod = await import('../../scripts/notion-sync');
    const downloadSpy = vi.fn().mockResolvedValue('/images/notion/p/id.png');
    const block = {
      id: 'id',
      image: {
        type: 'external',
        external: { url: 'https://img' },
        caption: [{ plain_text: 'cap' }],
      },
    };

    const fallback = await mod.transformImageBlock(block, '', downloadSpy);
    expect(fallback).toBe('![cap](https://img)');

    const localized = await mod.transformImageBlock(block, 'page', downloadSpy);
    expect(localized).toBe('![cap](/images/notion/p/id.png)');
    expect(downloadSpy).toHaveBeenCalled();

    const failed = await mod.transformImageBlock(block, 'page', vi.fn().mockResolvedValue(null));
    expect(failed).toBe('![cap](https://img)');

    const fileBlock = {
      ...block,
      image: { type: 'file', file: { url: 'https://file' }, caption: [] },
    };
    const fileResult = await mod.transformImageBlock(fileBlock, 'page', downloadSpy);
    expect(fileResult).toContain('/images/notion/p/id.png');
  });

  it('sync filters statuses, skips up-to-date posts, and builds slugs', async () => {
    const contentDir = process.env.NOTION_CONTENT_DIR!;
    fs.mkdirSync(contentDir, { recursive: true });
    const existing = matter.stringify('old', {
      slug: 'should-skip',
      notion: { id: 'page-status' },
      lastEditedTime: '2024-02-02T00:00:00.000Z',
    });
    fs.writeFileSync(path.join(contentDir, 'should-skip.md'), existing);

    const pages = [
      {
        id: 'page-select',
        last_edited_time: '2024-03-01T00:00:00.000Z',
        cover: { type: 'external', external: { url: 'https://cover' } },
        properties: {
          Name: { title: [{ plain_text: 'Select Post' }] },
          Status: { type: 'select', select: { name: 'Published' } },
          tags: { multi_select: [] },
          date: { date: { start: '2024-03-01' } },
        },
      },
      {
        id: 'page-draft',
        last_edited_time: '2024-03-02T00:00:00.000Z',
        properties: {
          Name: { title: [{ plain_text: 'Draft' }] },
          Status: { type: 'status', status: { name: 'Draft' } },
          tags: { multi_select: [] },
          date: { date: { start: '2024-03-02' } },
        },
      },
      {
        id: 'page-status',
        last_edited_time: '2024-02-02T00:00:00.000Z',
        properties: {
          Name: { title: [{ plain_text: 'Should Skip' }] },
          Status: { type: 'status', status: { name: 'Published' } },
          tags: { multi_select: [] },
        },
      },
      {
        id: 'page-nostatus',
        last_edited_time: '2024-03-03T00:00:00.000Z',
        cover: null,
        properties: {
          Title: { title: [{ plain_text: 'No Status' }] },
          tags: { multi_select: [{ name: 'astro' }] },
          date: { date: { start: '2024-03-03' } },
        },
      },
    ];

    queryMock.mockResolvedValue({ results: pages });
    blocksListMock.mockResolvedValue({ results: [], has_more: false, next_cursor: null });
    if (lastN2mInstance.current) {
      lastN2mInstance.current.blocks = [];
    }

    const mod = await import('../../scripts/notion-sync');
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        new Response(Buffer.from('img'), {
          status: 200,
          headers: { 'content-type': 'image/png' },
        }),
      ),
    );
    vi.spyOn(mod, 'downloadImage');
    await mod.sync();

    const files = fs.readdirSync(contentDir);
    expect(files.some((f) => f.startsWith('select-post'))).toBe(true);
    expect(files.some((f) => f.includes('no-status'))).toBe(true);
    expect(files.some((f) => f.includes('should-skip'))).toBe(true);
    expect(files.some((f) => f.includes('draft'))).toBe(false);
  });

  it('sync falls back to first image block when cover is missing', async () => {
    const pages = [
      {
        id: 'page-fallback',
        last_edited_time: '2024-04-01T00:00:00.000Z',
        cover: null,
        properties: {
          Name: { title: [{ plain_text: 'Fallback Cover' }] },
          tags: { multi_select: [] },
        },
      },
    ];

    queryMock.mockResolvedValue({ results: pages });
    blocksListMock.mockResolvedValue({
      results: [
        { id: 'img', type: 'image', image: { type: 'external', external: { url: 'https://img' } } },
      ],
      has_more: false,
      next_cursor: null,
    });

    const mod = await import('../../scripts/notion-sync');
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        new Response(Buffer.from('img'), {
          status: 200,
          headers: { 'content-type': 'image/png' },
        }),
      ),
    );
    await mod.sync();

    const files = fs.readdirSync(process.env.NOTION_CONTENT_DIR!);
    const content = matter.read(path.join(process.env.NOTION_CONTENT_DIR!, files[0]));
    expect(content.data.cover).toContain('/images/notion/fallback-cover/img');
  });

  it('skips unsupported pages and respects provided slugs', async () => {
    const contentDir = process.env.NOTION_CONTENT_DIR!;
    fs.mkdirSync(contentDir, { recursive: true });
    fs.writeFileSync(
      path.join(contentDir, 'ignored.md'),
      matter.stringify('body', { title: 'no id' }),
    );

    const pages = [
      { id: 'not-full' },
      {
        id: 'file-cover',
        last_edited_time: '2024-05-01T00:00:00.000Z',
        cover: { type: 'file', file: { url: 'https://file-cover' } },
        properties: {
          Name: { title: [{ plain_text: 'File Cover' }] },
          Status: { type: 'status', status: { name: 'Published' } },
          slug: { rich_text: [{ plain_text: 'custom-slug' }] },
          date: { date: { start: '2024-05-01' } },
          tags: { multi_select: [] },
        },
      },
      {
        id: 'unknown-status',
        last_edited_time: '2024-05-02T00:00:00.000Z',
        properties: {
          Name: { title: [{ plain_text: 'Unknown' }] },
          Status: { type: 'multi_select' },
          tags: { multi_select: [] },
        },
      },
    ];

    queryMock.mockResolvedValue({ results: pages });
    blocksListMock.mockResolvedValue({ results: [], has_more: false, next_cursor: null });
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        new Response(Buffer.from('img'), {
          status: 200,
          headers: { 'content-type': 'image/png' },
        }),
      ),
    );

    const mod = await import('../../scripts/notion-sync');
    await mod.sync();

    const files = fs.readdirSync(contentDir);
    expect(files.some((f) => f.startsWith('custom-slug'))).toBe(true);
    expect(files.some((f) => f.includes('unknown-status'))).toBe(false);
  });

  it('initializes default directories when env paths are absent', async () => {
    delete process.env.NOTION_CONTENT_DIR;
    delete process.env.NOTION_PUBLIC_IMG_DIR;
    const defaultContent = path.join(process.cwd(), 'src/content/blog/notion');
    const defaultImages = path.join(process.cwd(), 'public/images/notion');
    fs.mkdirSync(defaultContent, { recursive: true });
    fs.mkdirSync(defaultImages, { recursive: true });
    vi.resetModules();
    await import('../../scripts/notion-sync');

    expect(fs.existsSync(defaultContent)).toBe(true);
    expect(fs.existsSync(defaultImages)).toBe(true);
  });

  it('handles downloadImage returning null for preferred and fallback covers', async () => {
    queryMock.mockResolvedValue({
      results: [
        {
          id: 'cover-null',
          last_edited_time: '2024-06-01T00:00:00.000Z',
          properties: {
            Name: { title: [{ plain_text: 'Null Cover' }] },
            tags: { multi_select: [] },
          },
        },
      ],
    });
    blocksListMock.mockResolvedValue({
      results: [
        { id: 'img', type: 'image', image: { type: 'external', external: { url: 'https://img' } } },
      ],
      has_more: false,
      next_cursor: null,
    });
    const mod = await import('../../scripts/notion-sync');
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(new Response(Buffer.from('x'), { status: 500 })),
    );

    await mod.sync();
    const file = fs.readFileSync(
      path.join(process.env.NOTION_CONTENT_DIR!, 'null-cover.md'),
      'utf-8',
    );
    const frontmatter = matter(file).data;
    expect(frontmatter.cover).toBe('');
  });

  it('resolves slug conflicts by hashing notion id', async () => {
    const contentDir = process.env.NOTION_CONTENT_DIR!;
    fs.mkdirSync(contentDir, { recursive: true });
    fs.mkdirSync(process.env.NOTION_PUBLIC_IMG_DIR!, { recursive: true });
    queryMock.mockResolvedValue({
      results: [
        {
          id: 'page-dup-a',
          last_edited_time: '2024-07-01T00:00:00.000Z',
          cover: null,
          properties: {
            Name: { title: [{ plain_text: 'Dup' }] },
            tags: { multi_select: [] },
            slug: { rich_text: [{ plain_text: 'dup' }] },
            date: { date: { start: '2024-07-01' } },
          },
        },
        {
          id: 'page-dup-b',
          last_edited_time: '2024-07-02T00:00:00.000Z',
          cover: null,
          properties: {
            Name: { title: [{ plain_text: 'Dup' }] },
            tags: { multi_select: [] },
            slug: { rich_text: [{ plain_text: 'dup' }] },
            date: { date: { start: '2024-07-02' } },
          },
        },
      ],
    });
    blocksListMock.mockResolvedValue({ results: [], has_more: false, next_cursor: null });
    const mod = await import('../../scripts/notion-sync');
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        new Response(Buffer.from('img'), {
          status: 200,
          headers: { 'content-type': 'image/png' },
        }),
      ),
    );
    await mod.sync();
    const files = fs.readdirSync(contentDir);
    expect(files).toContain('dup.md');
    expect(files).toContain(`dup-${shortHash('page-dup-b')}.md`);
  });

  it('migrates images and files when slug changes for the same notion page', async () => {
    const contentDir = process.env.NOTION_CONTENT_DIR!;
    const imagesDir = process.env.NOTION_PUBLIC_IMG_DIR!;
    fs.mkdirSync(contentDir, { recursive: true });
    fs.mkdirSync(imagesDir, { recursive: true });
    const existingPath = path.join(contentDir, 'old-slug.md');
    fs.writeFileSync(
      existingPath,
      matter.stringify('body', {
        slug: 'old-slug',
        notion: { id: 'page-rename' },
        lastEditedTime: '2024-01-01T00:00:00.000Z',
      }),
    );
    const sourceImageDir = path.join(imagesDir, 'page-rename');
    fs.mkdirSync(sourceImageDir, { recursive: true });
    fs.writeFileSync(path.join(sourceImageDir, 'img.png'), 'img');

    queryMock.mockResolvedValue({
      results: [
        {
          id: 'page-rename',
          last_edited_time: '2024-02-01T00:00:00.000Z',
          cover: null,
          properties: {
            Name: { title: [{ plain_text: 'Renamed' }] },
            tags: { multi_select: [] },
            slug: { rich_text: [{ plain_text: 'new-slug' }] },
            date: { date: { start: '2024-02-01' } },
          },
        },
      ],
    });
    blocksListMock.mockResolvedValue({ results: [], has_more: false, next_cursor: null });
    const mod = await import('../../scripts/notion-sync');
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(new Response(Buffer.from('img'), { status: 200 })),
    );
    await mod.sync();

    expect(fs.existsSync(path.join(contentDir, 'new-slug.md'))).toBe(true);
    expect(fs.existsSync(existingPath)).toBe(false);
    expect(fs.existsSync(path.join(imagesDir, 'new-slug', 'img.png'))).toBe(true);
    expect(fs.existsSync(sourceImageDir)).toBe(false);
  });

  it('runs autorun branch when NODE_ENV is not test', async () => {
    const originalEnv = process.env.NODE_ENV;
    try {
      process.env.NODE_ENV = 'production';
      queryMock.mockResolvedValue({ results: [] });
      vi.resetModules();
      await import('../../scripts/notion-sync');
    } finally {
      process.env.NODE_ENV = originalEnv;
    }
  });

  it('preserves existing local tags when updating articles from Notion', async () => {
    const contentDir = process.env.NOTION_CONTENT_DIR!;
    fs.mkdirSync(contentDir, { recursive: true });

    // Create an existing file with local tags
    const existingContent = matter.stringify('existing content', {
      title: 'Test Article',
      slug: 'test-article',
      notion: { id: 'page-preserve-tags' },
      lastEditedTime: '2024-01-01T00:00:00.000Z',
      tags: ['Local Tag 1', 'Local Tag 2'],
      date: '2024-01-01',
    });
    fs.writeFileSync(path.join(contentDir, 'test-article.md'), existingContent);

    // Mock a page update from Notion with different tags
    queryMock.mockResolvedValue({
      results: [
        {
          id: 'page-preserve-tags',
          last_edited_time: '2024-02-01T00:00:00.000Z',
          cover: null,
          properties: {
            Name: { title: [{ plain_text: 'Test Article' }] },
            tags: { multi_select: [{ name: 'Notion Tag 1' }, { name: 'Notion Tag 2' }] },
            date: { date: { start: '2024-02-01' } },
            Status: { type: 'select', select: { name: 'Published' } },
          },
        },
      ],
    });
    blocksListMock.mockResolvedValue({ results: [], has_more: false, next_cursor: null });

    const mod = await import('../../scripts/notion-sync');
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(new Response(Buffer.from('img'), { status: 200 })),
    );
    await mod.sync();

    // Verify that both local and Notion tags are preserved
    const updatedFile = fs.readFileSync(path.join(contentDir, 'test-article.md'), 'utf-8');
    const { data } = matter(updatedFile);

    expect(data.tags).toEqual(
      expect.arrayContaining(['Local Tag 1', 'Local Tag 2', 'Notion Tag 1', 'Notion Tag 2']),
    );
    expect(data.tags.length).toBe(4);
  });
});
