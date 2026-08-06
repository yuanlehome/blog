import fs from 'fs';
import os from 'os';
import path from 'path';
import { afterEach, describe, expect, it } from 'vitest';
import { ensureUniqueSlug } from '../../src/lib/slug/index.js';
import {
  buildImportOwnerId,
  buildSlugOwnerMap,
  normalizeImportSourceUrl,
} from '../../scripts/slug-registry.js';

const tempDirs: string[] = [];

afterEach(() => {
  for (const dir of tempDirs) {
    fs.rmSync(dir, { recursive: true, force: true });
  }
  tempDirs.length = 0;
});

describe('buildSlugOwnerMap', () => {
  it('collects slugs from nested markdown files', () => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), 'slug-map-'));
    tempDirs.push(root);
    fs.mkdirSync(path.join(root, 'notion'), { recursive: true });
    fs.mkdirSync(path.join(root, 'wechat'), { recursive: true });

    fs.writeFileSync(
      path.join(root, 'notion', 'a.md'),
      `---\nslug: custom-slug\nnotion:\n  id: notion-123\n---\ncontent`,
    );
    fs.writeFileSync(path.join(root, 'wechat', 'b.md'), `---\ntitle: test\n---\ncontent`);

    const map = buildSlugOwnerMap(root);
    expect(map.get('custom-slug')).toBe('notion-123');
    expect(map.get('b')).toBe('file:wechat/b.md');
  });

  it('uses the canonical source URL and provider as the owner for imported content', () => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), 'slug-map-'));
    tempDirs.push(root);
    fs.mkdirSync(path.join(root, 'zhihu'), { recursive: true });

    fs.writeFileSync(
      path.join(root, 'zhihu', 'article.md'),
      [
        '---',
        'slug: imported-article',
        "source_url: 'https://zhuanlan.zhihu.com/p/123?share_code=abc&utm_psn=456'",
        '---',
        'content',
      ].join('\n'),
    );

    const map = buildSlugOwnerMap(root);
    const canonicalOwner = buildImportOwnerId('zhihu', 'https://zhuanlan.zhihu.com/p/123');

    expect(map.get('imported-article')).toBe(canonicalOwner);
    expect(ensureUniqueSlug('imported-article', canonicalOwner, map)).toBe('imported-article');
  });
});

describe('import owner URL normalization', () => {
  it('removes tracking parameters case-insensitively and keeps meaningful parameters', () => {
    expect(
      normalizeImportSourceUrl(
        'https://example.com/article?view=full&UTM_Source=feed&share_code=abc',
      ),
    ).toBe('https://example.com/article?view=full');
  });

  it('creates the same owner from tracked and canonical URLs', () => {
    expect(
      buildImportOwnerId(
        'Zhihu',
        'https://zhuanlan.zhihu.com/p/123?utm_medium=social&share_code=abc',
      ),
    ).toBe(buildImportOwnerId('zhihu', 'https://zhuanlan.zhihu.com/p/123'));
  });
});
