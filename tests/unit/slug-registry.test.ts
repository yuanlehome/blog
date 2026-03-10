import fs from 'fs';
import os from 'os';
import path from 'path';
import { afterEach, describe, expect, it } from 'vitest';
import { buildSlugOwnerMap } from '../../scripts/slug-registry.js';

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
});
