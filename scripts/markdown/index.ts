/**
 * Markdown Enhancement Pipeline
 *
 * Main entry point for markdown import enhancements:
 * - Automatic translation from English to Chinese
 * - Code fence language detection
 * - Image caption normalization
 * - Markdown formatting cleanup
 *
 * Usage:
 *   import { processMarkdownForImport } from './scripts/markdown';
 *
 * Configuration via environment variables:
 *   - MARKDOWN_TRANSLATE_ENABLED=1 (enable translation)
 *   - MARKDOWN_TRANSLATE_PROVIDER=mock|identity (translator to use)
 */

export {
  processMarkdownForImport,
  type ProcessingOptions,
  type ProcessingResult,
  type ProcessingDiagnostics,
} from './markdown-processor.js';

export {
  detectLanguage,
  shouldTranslate,
  type LanguageCode,
  type LanguageDetectionResult,
} from './language-detector.js';

export {
  createTranslator,
  getConfiguredTranslator,
  MockTranslator,
  IdentityTranslator,
  type Translator,
  type TranslationNode,
  type TranslationPatch,
  type TranslationResult,
} from './translator.js';

export { DeepSeekTranslator } from './deepseek-translator.js';

export { detectCodeLanguage, isGitHubActionsWorkflow } from './code-fence-fixer.js';

export { fixMathBlock, degradeToTexBlock, type MathFixResult } from './math-fixer.js';
