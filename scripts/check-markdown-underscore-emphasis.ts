import { readdir, readFile, stat } from 'node:fs/promises';
import path from 'node:path';

import { findSuspiciousUnderscoreEmphasis } from './markdown/underscore-emphasis-guard.js';

const IGNORED_DIRECTORIES = new Set([
  '.astro',
  '.git',
  '.pytest_cache',
  'coverage',
  'dist',
  'node_modules',
  'public',
  'test-results',
  'tmp',
]);

const IGNORED_PATH_PREFIXES = ['tests/fixtures/'];

function toPosixPath(value: string): string {
  return value.split(path.sep).join('/');
}

function shouldIgnore(relativePath: string): boolean {
  return IGNORED_PATH_PREFIXES.some((prefix) => relativePath.startsWith(prefix));
}

async function collectMarkdownFiles(rootPath: string, relativePath = ''): Promise<string[]> {
  const directoryPath = path.join(rootPath, relativePath);
  const entries = await readdir(directoryPath, { withFileTypes: true });
  const files: string[] = [];

  for (const entry of entries) {
    const entryRelativePath = toPosixPath(path.join(relativePath, entry.name));
    if (entry.isDirectory()) {
      if (IGNORED_DIRECTORIES.has(entry.name) || shouldIgnore(`${entryRelativePath}/`)) {
        continue;
      }

      files.push(...(await collectMarkdownFiles(rootPath, entryRelativePath)));
      continue;
    }

    if (!entry.isFile() || !entry.name.endsWith('.md') || shouldIgnore(entryRelativePath)) {
      continue;
    }

    files.push(path.join(rootPath, entryRelativePath));
  }

  return files;
}

async function expandTargets(rootPath: string, targets: string[]): Promise<string[]> {
  if (targets.length === 0) {
    return collectMarkdownFiles(rootPath);
  }

  const files: string[] = [];
  for (const target of targets) {
    const resolvedPath = path.resolve(rootPath, target);
    const targetStat = await stat(resolvedPath);
    if (targetStat.isDirectory()) {
      files.push(...(await collectMarkdownFiles(resolvedPath)));
      continue;
    }

    if (targetStat.isFile() && resolvedPath.endsWith('.md')) {
      files.push(resolvedPath);
    }
  }

  return files.sort();
}

async function main(): Promise<void> {
  const rootPath = process.cwd();
  const targets = process.argv.slice(2);
  const markdownFiles = await expandTargets(rootPath, targets);
  const failures: string[] = [];

  for (const filePath of markdownFiles) {
    const markdown = await readFile(filePath, 'utf8');
    const findings = await findSuspiciousUnderscoreEmphasis(markdown);
    const relativePath = toPosixPath(path.relative(rootPath, filePath));

    for (const finding of findings) {
      failures.push(
        `${relativePath}:${finding.line}:${finding.column} ${finding.nodeType} parsed from underscore delimiters inside an identifier-like token. Wrap code-like names in backticks or escape the underscores.\n` +
          `  raw: ${finding.raw}\n` +
          `  line: ${finding.excerpt}`,
      );
    }
  }

  if (failures.length === 0) {
    return;
  }

  console.error(failures.join('\n\n'));
  process.exitCode = 1;
}

void main();
