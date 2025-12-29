import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { fixMath, processFile, processDirectory } from './utils.js';

export function runFixMath(targetPath: string) {
  const fullPath = path.resolve(targetPath);

  if (!fs.existsSync(fullPath)) {
    throw new Error(`Path not found: ${fullPath}`);
  }

  const stat = fs.statSync(fullPath);
  if (stat.isDirectory()) {
    processDirectory(fullPath, (file) => file.endsWith('.md'), fixMath);
  } else {
    processFile(fullPath, fixMath);
  }
}

// Re-export for backward compatibility
export { normalizeInvisibleCharacters, splitCodeFences, fixMath } from './utils.js';

export function runCli(argv = process.argv) {
  const modulePath = fileURLToPath(import.meta.url);
  if (!argv[1] || path.resolve(argv[1]) !== modulePath) return;

  const targetPath = argv[2];
  if (!targetPath) {
    console.error('Usage: npx tsx scripts/fix-math.ts <file-or-directory-path>');
    process.exit(1);
  }

  try {
    runFixMath(targetPath);
  } catch (error) {
    console.error((error as Error).message);
    process.exit(1);
  }
}

runCli();
