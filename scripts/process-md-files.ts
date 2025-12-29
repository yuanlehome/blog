import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { processMdFiles, processFile, processDirectory } from './utils.js';

export function runProcessMdFiles(targetPath: string) {
  const fullPath = path.resolve(targetPath);

  if (!fs.existsSync(fullPath)) {
    throw new Error(`Path not found: ${fullPath}`);
  }

  const stat = fs.statSync(fullPath);
  if (stat.isDirectory()) {
    processDirectory(fullPath, (file) => file.endsWith('.md'), processMdFiles);
  } else {
    processFile(fullPath, processMdFiles);
  }
}

// Re-export for backward compatibility
export { normalizeInvisibleCharacters, splitCodeFences, processMdFiles } from './utils.js';

export function runCli(argv = process.argv) {
  const modulePath = fileURLToPath(import.meta.url);
  if (!argv[1] || path.resolve(argv[1]) !== modulePath) return;

  const targetPath = argv[2];
  if (!targetPath) {
    console.error('Usage: npx tsx scripts/process-md-files.ts <file-or-directory-path>');
    process.exit(1);
  }

  try {
    runProcessMdFiles(targetPath);
  } catch (error) {
    console.error((error as Error).message);
    process.exit(1);
  }
}

runCli();
