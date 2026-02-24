import fs from 'fs';
import path from 'path';
import { describe, it, expect } from 'vitest';

describe('configuration and env contracts', () => {
  const repoRoot = process.cwd();

  it('has required config files for CI and tooling', () => {
    const requiredFiles = [
      '.lychee.toml',
      'astro.config.mjs',
      'tailwind.config.mjs',
      'vitest.config.ts',
    ];

    for (const rel of requiredFiles) {
      const fullPath = path.join(repoRoot, rel);
      expect(fs.existsSync(fullPath)).toBe(true);
    }
  });

  it('exposes paths configuration without throwing at import time', async () => {
    // Ensure src/config/paths can be imported in a Node/Vitest environment
    const mod = await import('../../src/config/paths');
    expect(mod).toHaveProperty('ROOT_DIR');
    expect(mod).toHaveProperty('BLOG_CONTENT_DIR');
    expect(mod).toHaveProperty('PUBLIC_DIR');
  });
});

