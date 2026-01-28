/**
 * Markdown Processing and Image Download for PDF OCR
 *
 * Cleans up OCR-generated markdown and downloads associated images
 */

import type { Logger } from '../../logger/types.js';
import path from 'path';
import fs from 'fs';

const IMAGE_DOWNLOAD_TIMEOUT_MS = 30000; // 30 seconds
const IMAGE_MAX_RETRIES = 3;

/**
 * Custom error for image download failures
 */
export class ImageDownloadError extends Error {
  constructor(
    message: string,
    public url: string,
    public statusCode?: number,
  ) {
    super(message);
    this.name = 'ImageDownloadError';
  }
}

/**
 * Sleep helper
 */
function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * Exponential backoff delay
 */
function getBackoffDelay(attempt: number): number {
  const baseDelay = 500;
  const maxDelay = 5000;
  const delay = baseDelay * Math.pow(2, attempt);
  return Math.min(delay, maxDelay);
}

/**
 * Process OCR markdown for MDX compatibility
 *
 * - Fix unclosed code fences
 * - Normalize list indentation
 * - Remove excessive blank lines
 * - Clean up suspicious table of contents blocks
 */
export function processOcrMarkdown(markdown: string): string {
  let processed = markdown;

  // Fix unclosed code fences
  processed = fixCodeFences(processed);

  // Normalize excessive blank lines (more than 2 consecutive)
  processed = processed.replace(/\n{4,}/g, '\n\n\n');

  // Normalize list indentation (ensure proper spacing)
  processed = normalizeListIndentation(processed);

  return processed;
}

/**
 * Fix unclosed code fences
 */
function fixCodeFences(markdown: string): string {
  const lines = markdown.split('\n');
  const result: string[] = [];
  let inCodeBlock = false;

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const isCodeFence = /^```/.test(line);

    if (isCodeFence) {
      if (!inCodeBlock) {
        // Opening fence
        inCodeBlock = true;
      } else {
        // Closing fence
        inCodeBlock = false;
      }
    }

    result.push(line);
  }

  // If still in code block at end, close it
  if (inCodeBlock) {
    result.push('```');
  }

  return result.join('\n');
}

/**
 * Normalize list indentation
 */
function normalizeListIndentation(markdown: string): string {
  const lines = markdown.split('\n');
  const result: string[] = [];

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const prevLine = i > 0 ? lines[i - 1] : '';

    // Check if this is a list item
    const isListItem = /^\s*[-*+]\s/.test(line) || /^\s*\d+\.\s/.test(line);
    const prevIsListItem = /^\s*[-*+]\s/.test(prevLine) || /^\s*\d+\.\s/.test(prevLine);

    // Ensure blank line before list start (unless already there or at start)
    if (isListItem && !prevIsListItem && prevLine.trim() !== '' && i > 0) {
      if (result.length > 0 && result[result.length - 1].trim() !== '') {
        result.push('');
      }
    }

    result.push(line);
  }

  return result.join('\n');
}

/**
 * Download images from OCR result
 *
 * @param images - Map of image paths to URLs from OCR
 * @param imageRoot - Root directory for saving images
 * @param slug - Article slug for organization
 * @param publicBasePath - Public URL path for images
 * @returns Map of original image paths to local relative paths
 */
export async function downloadOcrImages(
  images: Record<string, string>,
  imageRoot: string,
  slug: string,
  publicBasePath: string,
  logger?: Logger,
): Promise<Record<string, string>> {
  const localImageMap: Record<string, string> = {};

  // Create image directory
  const imageDir = path.join(imageRoot, 'pdf', slug);
  fs.mkdirSync(imageDir, { recursive: true });

  const entries = Object.entries(images);
  let successCount = 0;
  let failCount = 0;

  for (const [imgPath, imgUrl] of entries) {
    try {
      // Sanitize image path to prevent path traversal
      const sanitizedPath = sanitizeImagePath(imgPath);
      if (!sanitizedPath) {
        logger?.warn('Skipping invalid image path', {
          module: 'pdf_vl',
          imgPath,
        });
        failCount++;
        continue;
      }

      // Download image with retries
      const imageBuffer = await downloadImageWithRetry(imgUrl, logger);

      // Determine file extension from buffer or URL
      const ext = getImageExtension(imageBuffer, imgUrl);

      // Generate filename
      const filename = `${sanitizedPath.replace(/[^a-zA-Z0-9_-]/g, '_')}${ext}`;
      const localPath = path.join(imageDir, filename);

      // Save image
      fs.writeFileSync(localPath, imageBuffer);

      // Store relative path for markdown
      const relativePath = `${publicBasePath}/${filename}`;
      localImageMap[imgPath] = relativePath;

      successCount++;
      logger?.debug('Image downloaded', {
        module: 'pdf_vl',
        imgPath,
        localPath: relativePath,
      });
    } catch (error) {
      failCount++;
      logger?.warn('Failed to download image', {
        module: 'pdf_vl',
        imgPath,
        imgUrl,
        error: error instanceof Error ? error.message : String(error),
      });
      // Continue with other images
    }
  }

  logger?.info('Image download completed', {
    module: 'pdf_vl',
    total: entries.length,
    success: successCount,
    failed: failCount,
  });

  return localImageMap;
}

/**
 * Download image with retry logic
 */
async function downloadImageWithRetry(url: string, logger?: Logger): Promise<Buffer> {
  let lastError: Error | null = null;

  for (let attempt = 0; attempt < IMAGE_MAX_RETRIES; attempt++) {
    try {
      if (attempt > 0) {
        const delay = getBackoffDelay(attempt - 1);
        await sleep(delay);
      }

      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), IMAGE_DOWNLOAD_TIMEOUT_MS);

      try {
        const response = await fetch(url, {
          signal: controller.signal,
          headers: {
            'User-Agent':
              'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
          },
        });

        clearTimeout(timeout);

        if (!response.ok) {
          throw new ImageDownloadError(
            `HTTP ${response.status} ${response.statusText}`,
            url,
            response.status,
          );
        }

        const arrayBuffer = await response.arrayBuffer();
        return Buffer.from(arrayBuffer);
      } finally {
        clearTimeout(timeout);
      }
    } catch (error) {
      lastError = error instanceof Error ? error : new Error(String(error));

      if (attempt === IMAGE_MAX_RETRIES - 1) {
        break;
      }
    }
  }

  throw new ImageDownloadError(
    `Failed after ${IMAGE_MAX_RETRIES} attempts: ${lastError?.message || 'Unknown error'}`,
    url,
  );
}

/**
 * Sanitize image path to prevent path traversal
 */
function sanitizeImagePath(imgPath: string): string | null {
  // Remove leading ./
  let cleaned = imgPath.replace(/^\.\//, '');

  // Check for path traversal attempts
  if (cleaned.includes('..') || path.isAbsolute(cleaned)) {
    return null;
  }

  // Get basename only (ignore directory structure)
  cleaned = path.basename(cleaned);

  // Remove extension
  const ext = path.extname(cleaned);
  cleaned = cleaned.slice(0, -ext.length);

  return cleaned;
}

/**
 * Get image extension from buffer or URL
 */
function getImageExtension(buffer: Buffer, url: string): string {
  // Check magic bytes (need at least 12 bytes for WebP)
  if (buffer.length >= 12) {
    const header = buffer.slice(0, 12);

    // PNG: 89 50 4E 47
    if (header[0] === 0x89 && header[1] === 0x50 && header[2] === 0x4e && header[3] === 0x47) {
      return '.png';
    }

    // JPEG: FF D8 FF
    if (header[0] === 0xff && header[1] === 0xd8 && header[2] === 0xff) {
      return '.jpg';
    }

    // GIF: 47 49 46
    if (header[0] === 0x47 && header[1] === 0x49 && header[2] === 0x46) {
      return '.gif';
    }

    // WebP: 52 49 46 46 ... 57 45 42 50
    if (
      header[0] === 0x52 &&
      header[1] === 0x49 &&
      header[2] === 0x46 &&
      header[3] === 0x46 &&
      header[8] === 0x57 &&
      header[9] === 0x45 &&
      header[10] === 0x42 &&
      header[11] === 0x50
    ) {
      return '.webp';
    }
  }

  // Fallback: try to extract from URL
  try {
    const urlExt = path.extname(new URL(url).pathname).toLowerCase();
    if (['.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp'].includes(urlExt)) {
      return urlExt === '.jpeg' ? '.jpg' : urlExt;
    }
  } catch {
    // Invalid URL, ignore and use default
  }

  // Default
  return '.jpg';
}
