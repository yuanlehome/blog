import fs from 'node:fs';
import path from 'node:path';
import { describe, expect, it } from 'vitest';

describe('import:content npm lifecycle', () => {
  it('forwards CLI arguments only to the importer and lints after a successful import', () => {
    const packageJsonPath = path.join(process.cwd(), 'package.json');
    const packageJson = JSON.parse(fs.readFileSync(packageJsonPath, 'utf8'));

    expect(packageJson.scripts['import:content']).toBe('tsx scripts/content-import.ts');
    expect(packageJson.scripts['postimport:content']).toBe('npm run lint');
  });
});
