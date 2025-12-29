import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { processMdFiles, processFile, processDirectory } from './utils.js';

export const PROCESS_MD_FILES_HELP = [
  '用法: npx tsx scripts/process-md-files.ts <file-or-directory>',
  '说明: 修正常见数学/不可见字符，原地改写 Markdown',
  '提示: --help 或 -h 显示本说明',
].join('\n');

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
  if (argv.includes('--help') || argv.includes('-h')) {
    console.log(PROCESS_MD_FILES_HELP);
    process.exit(0);
  }
  if (!targetPath) {
    console.error(PROCESS_MD_FILES_HELP);
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
