import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { fixMath, normalizeInvisibleCharacters, splitCodeFences } from './lib/shared/math-fix.js';

function processFile(filePath: string) {
  const content = fs.readFileSync(filePath, 'utf-8');
  const fixed = fixMath(content);
  if (fixed !== content) {
    fs.writeFileSync(filePath, fixed, 'utf-8');
    console.log(`✅ Fixed math in ${filePath}`);
  }
}

function processDirectory(dirPath: string) {
  const files = fs.readdirSync(dirPath);
  for (const file of files) {
    const p = path.join(dirPath, file);
    const stat = fs.statSync(p);
    if (stat.isDirectory()) {
      processDirectory(p);
    } else if (file.endsWith('.md')) {
      processFile(p);
    }
  }
}

export function runFixMath(targetPath: string) {
  const fullPath = path.resolve(targetPath);

  if (!fs.existsSync(fullPath)) {
    throw new Error(`Path not found: ${fullPath}`);
  }

  const stat = fs.statSync(fullPath);
  if (stat.isDirectory()) {
    processDirectory(fullPath);
  } else {
    processFile(fullPath);
  }
}

// Re-export for backward compatibility
export { normalizeInvisibleCharacters, splitCodeFences, fixMath };

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
