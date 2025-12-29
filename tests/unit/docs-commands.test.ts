import fs from 'fs';
import path from 'path';
import { beforeAll, describe, expect, it } from 'vitest';

const repoRoot = path.resolve(__dirname, '..', '..');
const readme = fs.readFileSync(path.join(repoRoot, 'README.md'), 'utf8');
let helpTexts: {
  CONTENT_IMPORT_HELP: string;
  DELETE_ARTICLE_HELP: string;
  NOTION_SYNC_HELP: string;
  PROCESS_MD_FILES_HELP: string;
};

beforeAll(async () => {
  process.env.NOTION_TOKEN ||= 'test-token';
  process.env.NOTION_DATABASE_ID ||= 'test-db';

  const notion = await import('../../scripts/notion-sync');
  const importer = await import('../../scripts/content-import');
  const deleter = await import('../../scripts/delete-article');
  const processor = await import('../../scripts/process-md-files');

  helpTexts = {
    CONTENT_IMPORT_HELP: importer.CONTENT_IMPORT_HELP,
    DELETE_ARTICLE_HELP: deleter.DELETE_ARTICLE_HELP,
    NOTION_SYNC_HELP: notion.NOTION_SYNC_HELP,
    PROCESS_MD_FILES_HELP: processor.PROCESS_MD_FILES_HELP,
  };
});

describe('docs command snippets', () => {
  it('README 覆盖核心命令与参数', () => {
    const snippets = [
      'npm run notion:sync',
      'npm run import:content -- --url=',
      '--allow-overwrite',
      '--dry-run',
      '--use-first-image-as-cover',
      'npm run delete:article -- --target=',
      '--delete-images',
      'npm run check',
      'npm run lint',
      'npm run test',
      'npm run test:e2e',
    ];

    snippets.forEach((snippet) => {
      expect(readme).toContain(snippet);
    });
  });

  it('help 文案列出真实参数', () => {
    expect(helpTexts.CONTENT_IMPORT_HELP).toContain('--allow-overwrite');
    expect(helpTexts.CONTENT_IMPORT_HELP).toContain('--use-first-image-as-cover');
    expect(helpTexts.CONTENT_IMPORT_HELP).toContain('--dry-run');
    expect(helpTexts.DELETE_ARTICLE_HELP).toContain('--target');
    expect(helpTexts.DELETE_ARTICLE_HELP).toContain('--delete-images');
    expect(helpTexts.PROCESS_MD_FILES_HELP).toContain('process-md-files');
    expect(helpTexts.NOTION_SYNC_HELP).toContain('NOTION_TOKEN');
  });
});
