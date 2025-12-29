/**
 * Language Detection Module
 *
 * Detects the primary language of markdown content to determine if translation is needed.
 * Uses character-based heuristics to identify English vs Chinese text.
 */

export type LanguageCode = 'en' | 'zh' | 'unknown';

export interface LanguageDetectionResult {
  language: LanguageCode;
  confidence: number;
  englishRatio: number;
  chineseRatio: number;
}

/**
 * Character ranges for language detection
 */
const CJK_RANGES = [
  [0x4e00, 0x9fff], // CJK Unified Ideographs
  [0x3400, 0x4dbf], // CJK Extension A
  [0xf900, 0xfaff], // CJK Compatibility Ideographs
];

const ENGLISH_PATTERN = /[a-zA-Z]/;

/**
 * Check if a character is CJK (Chinese/Japanese/Korean)
 */
function isCJKChar(char: string): boolean {
  const code = char.charCodeAt(0);
  return CJK_RANGES.some(([start, end]) => code >= start && code <= end);
}

/**
 * Extract text content from markdown, excluding code blocks and URLs
 */
function extractTextContent(markdown: string): string {
  let text = markdown;

  // Remove frontmatter
  text = text.replace(/^---\n[\s\S]*?\n---\n/, '');

  // Remove code blocks
  text = text.replace(/```[\s\S]*?```/g, '');
  text = text.replace(/`[^`\n]+`/g, '');

  // Remove URLs
  text = text.replace(/https?:\/\/[^\s)]+/g, '');

  // Remove image/link markdown syntax
  text = text.replace(/!\[([^\]]*)\]\([^)]+\)/g, '$1');
  text = text.replace(/\[([^\]]*)\]\([^)]+\)/g, '$1');

  return text;
}

/**
 * Detect the primary language of markdown content
 *
 * @param markdown - Markdown content to analyze
 * @param threshold - Minimum English ratio to consider content as English (default: 0.6)
 * @returns Language detection result
 */
export function detectLanguage(markdown: string, threshold = 0.6): LanguageDetectionResult {
  const textContent = extractTextContent(markdown);

  let englishCount = 0;
  let chineseCount = 0;
  let totalChars = 0;

  for (const char of textContent) {
    // Skip whitespace and punctuation
    if (/\s/.test(char) || /[^\w\u4e00-\u9fff]/.test(char)) {
      continue;
    }

    totalChars++;

    if (isCJKChar(char)) {
      chineseCount++;
    } else if (ENGLISH_PATTERN.test(char)) {
      englishCount++;
    }
  }

  // If no meaningful content, return unknown
  if (totalChars < 10) {
    return {
      language: 'unknown',
      confidence: 0,
      englishRatio: 0,
      chineseRatio: 0,
    };
  }

  const englishRatio = englishCount / totalChars;
  const chineseRatio = chineseCount / totalChars;

  // Determine language based on ratios
  let language: LanguageCode = 'unknown';
  let confidence = 0;

  if (englishRatio >= threshold) {
    language = 'en';
    confidence = englishRatio;
  } else if (chineseRatio >= threshold) {
    language = 'zh';
    confidence = chineseRatio;
  } else if (englishRatio > chineseRatio) {
    language = 'en';
    confidence = englishRatio;
  } else if (chineseRatio > englishRatio) {
    language = 'zh';
    confidence = chineseRatio;
  }

  return {
    language,
    confidence,
    englishRatio,
    chineseRatio,
  };
}

/**
 * Check if content should be translated based on language detection
 *
 * @param markdown - Markdown content to check
 * @param threshold - Minimum English ratio (default: 0.6)
 * @returns True if content is primarily English and should be translated
 */
export function shouldTranslate(markdown: string, threshold = 0.6): boolean {
  const result = detectLanguage(markdown, threshold);
  return result.language === 'en' && result.confidence >= threshold;
}
