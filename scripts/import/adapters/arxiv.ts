/**
 * arXiv Adapter
 *
 * Handles article import from arXiv.org papers by downloading LaTeX source
 * and converting to Markdown/MDX
 */

import type { Adapter, Article, FetchArticleInput } from './types.js';
import { createLogger } from '../../logger/index.js';
import fs from 'fs';
import path from 'path';
import os from 'os';
import axios from 'axios';
import * as tar from 'tar';
import { promisify } from 'util';
import { pipeline } from 'stream';

const streamPipeline = promisify(pipeline);

// Security limits
const MAX_EXTRACT_SIZE = 200 * 1024 * 1024; // 200MB
const MAX_FILE_COUNT = 5000;
const EXTRACT_TIMEOUT = 60000; // 60 seconds

// Allowed file extensions for extraction
const ALLOWED_EXTENSIONS = new Set([
  '.tex',
  '.bib',
  '.bst',
  '.sty',
  '.cls',
  '.png',
  '.jpg',
  '.jpeg',
  '.pdf',
  '.eps',
  '.svg',
]);

/**
 * Parse arXiv paper ID from various URL formats
 */
export function parseArxivId(url: string): string | null {
  try {
    const urlObj = new URL(url);
    // Only allow arxiv.org or its subdomains, not any hostname containing arxiv.org
    if (urlObj.hostname !== 'arxiv.org' && !urlObj.hostname.endsWith('.arxiv.org')) {
      return null;
    }

    // Match patterns: /pdf/<id>, /abs/<id>, /src/<id>
    // Paper ID format: YYMM.NNNNN or YYMM.NNNNNvN
    const match = urlObj.pathname.match(/\/(pdf|abs|src|e-print)\/(\d{4}\.\d{4,5}(?:v\d+)?)/);
    if (match) {
      return match[2];
    }

    return null;
  } catch {
    return null;
  }
}

/**
 * Check if path is safe (no path traversal)
 */
function isSafePath(filePath: string, extractDir: string): boolean {
  const resolved = path.resolve(extractDir, filePath);
  const relative = path.relative(extractDir, resolved);

  // Check for path traversal
  if (relative.startsWith('..') || path.isAbsolute(relative)) {
    return false;
  }

  return true;
}

/**
 * Download arXiv source package
 */
async function downloadSource(
  paperId: string,
  destPath: string,
  logger: ReturnType<typeof createLogger>,
): Promise<void> {
  const sourceUrl = `https://arxiv.org/src/${paperId}`;
  logger.info('Downloading arXiv source', { paperId, sourceUrl });

  try {
    const response = await axios.get(sourceUrl, {
      responseType: 'stream',
      timeout: 120000,
      headers: {
        'User-Agent': 'Mozilla/5.0 (compatible; BlogImporter/1.0)',
      },
    });

    await streamPipeline(response.data, fs.createWriteStream(destPath));
    logger.info('Source download complete', { paperId, size: fs.statSync(destPath).size });
  } catch (error) {
    if (axios.isAxiosError(error) && error.response?.status === 404) {
      throw new Error(
        `arXiv source not found for ${paperId}. The paper may not have source files.`,
      );
    }
    throw error;
  }
}

/**
 * Safely extract tar.gz with security checks
 */
async function extractTarGz(
  tarPath: string,
  extractDir: string,
  logger: ReturnType<typeof createLogger>,
): Promise<string[]> {
  logger.info('Extracting source package', { tarPath, extractDir });

  let totalSize = 0;
  let fileCount = 0;
  const extractedFiles: string[] = [];

  return new Promise((resolve, reject) => {
    let timeoutId: NodeJS.Timeout | null = null;

    const extract = tar.extract({
      cwd: extractDir,
      filter: (entryPath: string, entry: any) => {
        fileCount++;

        // Check file count limit
        if (fileCount > MAX_FILE_COUNT) {
          reject(new Error(`Too many files in archive (>${MAX_FILE_COUNT})`));
          return false;
        }

        // Check size limit
        totalSize += entry.size || 0;
        if (totalSize > MAX_EXTRACT_SIZE) {
          reject(new Error(`Archive too large (>${MAX_EXTRACT_SIZE} bytes)`));
          return false;
        }

        // Security: check for path traversal
        if (!isSafePath(entryPath, extractDir)) {
          logger.warn('Skipping unsafe path', { path: entryPath });
          return false;
        }

        // Only extract allowed file types
        const ext = path.extname(entryPath).toLowerCase();
        if (ext && !ALLOWED_EXTENSIONS.has(ext)) {
          return false;
        }

        return true;
      },
      onentry: (entry: any) => {
        extractedFiles.push(entry.path);
      },
    });

    const readStream = fs.createReadStream(tarPath);
    readStream.pipe(extract);

    extract.on('end', () => {
      if (timeoutId) clearTimeout(timeoutId);
      logger.info('Extraction complete', { fileCount: extractedFiles.length, totalSize });
      resolve(extractedFiles);
    });

    extract.on('error', (err: Error) => {
      if (timeoutId) clearTimeout(timeoutId);
      reject(err);
    });

    readStream.on('error', (err: Error) => {
      if (timeoutId) clearTimeout(timeoutId);
      reject(err);
    });

    // Timeout with cleanup
    timeoutId = setTimeout(() => {
      readStream.destroy();
      // Extract stream will be cleaned up by error handler
      reject(new Error('Extraction timeout'));
    }, EXTRACT_TIMEOUT);
  });
}

/**
 * Detect main TeX file
 */
function detectMainTex(extractDir: string, extractedFiles: string[]): string | null {
  const texFiles = extractedFiles.filter((f) => f.endsWith('.tex'));

  if (texFiles.length === 0) {
    return null;
  }

  if (texFiles.length === 1) {
    return texFiles[0];
  }

  // Score each tex file
  const candidates: Array<{ file: string; score: number }> = [];

  for (const texFile of texFiles) {
    let score = 0;
    const fullPath = path.join(extractDir, texFile);

    try {
      const content = fs.readFileSync(fullPath, 'utf-8');

      // Must have documentclass
      if (!content.includes('\\documentclass')) {
        continue;
      }

      // Must have begin{document}
      if (!content.includes('\\begin{document}')) {
        continue;
      }

      score += 100;

      // Prefer certain filenames
      const basename = path.basename(texFile).toLowerCase();
      if (basename === 'main.tex') score += 50;
      if (basename === 'paper.tex') score += 40;
      if (basename.includes('main')) score += 30;

      // Prefer files in root directory
      if (!texFile.includes('/')) {
        score += 20;
      }

      candidates.push({ file: texFile, score });
    } catch (error) {
      // Skip files we can't read
      continue;
    }
  }

  if (candidates.length === 0) {
    return null;
  }

  // Sort by score descending
  candidates.sort((a, b) => b.score - a.score);

  return candidates[0].file;
}

/**
 * Fetch metadata from arXiv API
 */
async function fetchArxivMetadata(
  paperId: string,
  logger: ReturnType<typeof createLogger>,
): Promise<{
  title: string;
  authors: string[];
  abstract: string;
  published: string;
  updated: string;
  categories: string[];
}> {
  // Remove version number for API query
  const baseId = paperId.replace(/v\d+$/, '');
  const apiUrl = `https://export.arxiv.org/api/query?id_list=${baseId}`;

  logger.info('Fetching arXiv metadata', { paperId, apiUrl });

  try {
    const response = await axios.get(apiUrl, {
      timeout: 30000,
      headers: {
        'User-Agent': 'Mozilla/5.0 (compatible; BlogImporter/1.0)',
      },
    });

    const xml = response.data;

    // Parse XML (simple regex-based parsing for Atom feed)
    const titleMatch = xml.match(/<title>([^<]+)<\/title>/);
    const title = titleMatch ? titleMatch[1].trim() : `arXiv Paper ${paperId}`;

    const authorsMatches = xml.matchAll(/<author>\s*<name>([^<]+)<\/name>/g);
    const authors = Array.from(authorsMatches, (m: RegExpMatchArray) => m[1].trim());

    const summaryMatch = xml.match(/<summary>([^<]+)<\/summary>/);
    const abstract = summaryMatch ? summaryMatch[1].trim() : '';

    const publishedMatch = xml.match(/<published>([^<]+)<\/published>/);
    const published = publishedMatch ? publishedMatch[1].trim() : '';

    const updatedMatch = xml.match(/<updated>([^<]+)<\/updated>/);
    const updated = updatedMatch ? updatedMatch[1].trim() : '';

    const categoryMatches = xml.matchAll(/<category[^>]+term="([^"]+)"/g);
    const categories = Array.from(categoryMatches, (m: RegExpMatchArray) => m[1]);

    logger.info('Metadata fetched', {
      title,
      authors: authors.length,
      categories: categories.length,
    });

    return {
      title: title.replace(/^arXiv:\d+\.\d+v?\d*\s+/i, ''), // Remove arXiv prefix
      authors,
      abstract,
      published,
      updated,
      categories,
    };
  } catch (error) {
    logger.warn('Failed to fetch metadata, using defaults', { error });
    return {
      title: `arXiv Paper ${paperId}`,
      authors: [],
      abstract: '',
      published: new Date().toISOString(),
      updated: new Date().toISOString(),
      categories: [],
    };
  }
}

/**
 * Convert LaTeX to Markdown and copy images to destination
 */
function latexToMarkdown(
  texContent: string,
  extractDir: string,
  texFilePath: string,
  imageDestDir: string,
  publicBasePath: string,
): { markdown: string; copiedImages: number } {
  let markdown = texContent;
  let copiedImages = 0;

  // Remove comments
  markdown = markdown.replace(/(?<!\\)%.*$/gm, '');

  // Extract and remove preamble
  const documentMatch = markdown.match(/\\begin{document}([\s\S]*?)\\end{document}/);
  if (documentMatch) {
    markdown = documentMatch[1];
  }

  // Convert sections
  markdown = markdown.replace(/\\section\{([^}]+)\}/g, '\n## $1\n');
  markdown = markdown.replace(/\\subsection\{([^}]+)\}/g, '\n### $1\n');
  markdown = markdown.replace(/\\subsubsection\{([^}]+)\}/g, '\n#### $1\n');

  // Convert text formatting
  markdown = markdown.replace(/\\textbf\{([^}]+)\}/g, '**$1**');
  markdown = markdown.replace(/\\textit\{([^}]+)\}/g, '*$1*');
  markdown = markdown.replace(/\\emph\{([^}]+)\}/g, '*$1*');

  // Convert math environments
  markdown = markdown.replace(/\\\[([\s\S]*?)\\\]/g, '\n$$$$\n$1\n$$$$\n');
  markdown = markdown.replace(/\\\(([\s\S]*?)\\\)/g, '$$$1$$');

  // Convert equation environments
  markdown = markdown.replace(
    /\\begin{equation\*?}([\s\S]*?)\\end{equation\*?}/g,
    '\n$$$$\n$1\n$$$$\n',
  );
  markdown = markdown.replace(/\\begin{align\*?}([\s\S]*?)\\end{align\*?}/g, '\n$$$$\n$1\n$$$$\n');

  // Extract and copy images
  const imageMatches = markdown.matchAll(/\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}/g);
  for (const match of imageMatches) {
    let imagePath = match[1];

    // Try to find the actual image file
    const possibleExtensions = ['', '.png', '.jpg', '.jpeg', '.pdf', '.eps'];
    let foundPath: string | null = null;

    for (const ext of possibleExtensions) {
      const testPath = path.join(extractDir, path.dirname(texFilePath), imagePath + ext);
      if (fs.existsSync(testPath)) {
        foundPath = testPath;
        break;
      }
    }

    if (foundPath) {
      // Copy image to destination with unique name to avoid collisions
      const imageExt = path.extname(foundPath);
      const imageBasename = path.basename(foundPath, imageExt);
      let destFilename = path.basename(foundPath);
      let destPath = path.join(imageDestDir, destFilename);

      // Handle filename collisions by appending index
      let counter = 1;
      while (fs.existsSync(destPath)) {
        destFilename = `${imageBasename}-${counter}${imageExt}`;
        destPath = path.join(imageDestDir, destFilename);
        counter++;
      }

      // Ensure dest directory exists
      fs.mkdirSync(path.dirname(destPath), { recursive: true });

      // Copy the file
      fs.copyFileSync(foundPath, destPath);
      copiedImages++;

      // Update markdown with public path
      const publicPath = `${publicBasePath}/${destFilename}`;
      markdown = markdown.replace(match[0], `\n![${destFilename}](${publicPath})\n`);
    } else {
      // Remove if not found
      markdown = markdown.replace(match[0], '');
    }
  }

  // Clean up LaTeX commands we don't handle
  markdown = markdown.replace(/\\[a-zA-Z]+(?:\[[^\]]*\])?(?:\{[^}]*\})?/g, '');

  // Clean up extra whitespace
  markdown = markdown.replace(/\n{3,}/g, '\n\n');
  markdown = markdown.trim();

  return { markdown, copiedImages };
}

/**
 * arXiv adapter implementation
 */
export const arxivAdapter: Adapter = {
  id: 'arxiv',
  name: 'arXiv',

  canHandle(url: string): boolean {
    return parseArxivId(url) !== null;
  },

  async fetchArticle(input: FetchArticleInput): Promise<Article> {
    const { url, options = {} } = input;
    const { slug = 'arxiv-article', imageRoot = '/tmp/images', logger: parentLogger } = options;

    // Create child logger with context
    const logger =
      parentLogger?.child({
        module: 'import',
        adapter: 'arxiv',
        url,
        slug,
      }) ?? createLogger({ silent: true });

    const extractionSpan = logger.span({
      name: 'arxiv-extraction',
      fields: { adapter: 'arxiv' },
    });
    extractionSpan.start();

    const paperId = parseArxivId(url);
    if (!paperId) {
      throw new Error(`Invalid arXiv URL: ${url}`);
    }

    logger.info('Processing arXiv paper', { paperId, url });

    // Create temp directories using os.tmpdir() for cross-platform compatibility
    const tmpDir = path.join(os.tmpdir(), `arxiv-${paperId}-${Date.now()}`);
    const extractDir = path.join(tmpDir, 'extracted');
    fs.mkdirSync(extractDir, { recursive: true });

    // Create destination directory for images
    const imageDestDir = path.join(imageRoot, slug);
    const publicBasePath = options.publicBasePath || `/images/arxiv/${slug}`;

    try {
      // Download source
      const tarPath = path.join(tmpDir, 'source.tar.gz');
      await downloadSource(paperId, tarPath, logger);

      // Extract source
      const extractedFiles = await extractTarGz(tarPath, extractDir, logger);

      // Detect main TeX file
      const mainTex = detectMainTex(extractDir, extractedFiles);
      if (!mainTex) {
        const texFiles = extractedFiles.filter((f) => f.endsWith('.tex'));
        throw new Error(
          `No compilable main TeX file found. Available .tex files: ${texFiles.join(', ') || 'none'}`,
        );
      }

      logger.info('Main TeX file detected', { mainTex });

      // Read and convert TeX to Markdown
      const texPath = path.join(extractDir, mainTex);
      const texContent = fs.readFileSync(texPath, 'utf-8');
      const { markdown, copiedImages } = latexToMarkdown(
        texContent,
        extractDir,
        mainTex,
        imageDestDir,
        publicBasePath,
      );

      // Fetch metadata
      const metadata = await fetchArxivMetadata(paperId, logger);

      // Format date
      const publishedDate = metadata.published
        ? new Date(metadata.published).toISOString().split('T')[0]
        : new Date().toISOString().split('T')[0];

      extractionSpan.end({
        status: 'ok',
        fields: {
          paperId,
          mainTex,
          imagesCount: copiedImages,
          markdownLength: markdown.length,
        },
      });

      logger.summary({
        status: 'ok',
        adapter: 'arxiv',
        paperId,
        title: metadata.title,
        imagesCount: copiedImages,
        markdownLength: markdown.length,
      });

      return {
        title: metadata.title,
        markdown: `# ${metadata.title}\n\n${markdown}`,
        canonicalUrl: url,
        source: 'arxiv',
        author: metadata.authors.join(', '),
        publishedAt: publishedDate,
        tags: ['arxiv', ...metadata.categories.slice(0, 3)],
        images: [],
      };
    } catch (error) {
      extractionSpan.end({ status: 'fail' });
      logger.error(error instanceof Error ? error : new Error(String(error)), {
        adapter: 'arxiv',
        url,
        paperId,
      });
      throw error;
    } finally {
      // Cleanup temp directory
      try {
        fs.rmSync(tmpDir, { recursive: true, force: true });
      } catch (cleanupError) {
        logger.warn('Failed to cleanup temp directory', { tmpDir });
      }
    }
  },
};
