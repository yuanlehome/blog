/**
 * Translation Module
 *
 * Provides translation interfaces and implementations for markdown content.
 * Supports multiple translation providers with a mock implementation for testing.
 */

import { DeepSeekTranslator } from './deepseek-translator.js';
import type { Logger } from '../logger/types.js';

/**
 * Input node for translation (text content)
 */
export interface TextTranslationNode {
  kind: 'text';
  nodeId: string;
  text: string;
  context?: string; // Optional context for better translation
}

/**
 * Input node for math fixing (LaTeX content)
 */
export interface MathTranslationNode {
  kind: 'math';
  nodeId: string;
  latex: string; // Raw LaTeX content (without $$ delimiters)
}

/**
 * Union type for all translation nodes
 */
export type TranslationNode = TextTranslationNode | MathTranslationNode;

/**
 * Output patch for translated text
 */
export interface TextTranslationPatch {
  kind: 'text';
  nodeId: string;
  text: string;
}

/**
 * Output patch for fixed math
 */
export interface MathTranslationPatch {
  kind: 'math';
  nodeId: string;
  latex: string; // Fixed LaTeX (no $$, no $, valid syntax)
  confidence: 'high' | 'low';
}

/**
 * Union type for all translation patches
 */
export type TranslationPatch = TextTranslationPatch | MathTranslationPatch;

export interface TranslationResult {
  patches: TranslationPatch[];
  metadata?: {
    provider: string;
    model?: string;
    timestamp: string;
    batches?: number;
    successBatches?: number;
    failedBatches?: number;
    cacheHits?: number;
  };
}

/**
 * Translation options
 */
export interface TranslationOptions {
  logger?: Logger;
}

/**
 * Base translator interface
 */
export interface Translator {
  name: string;
  translate(nodes: TranslationNode[], options?: TranslationOptions): Promise<TranslationResult>;
}

/**
 * Mock translator for testing
 * - For text nodes: Prepends "[ZH] " to simulate translation
 * - For math nodes: Removes all $ and $$ delimiters to simulate fixing
 */
export class MockTranslator implements Translator {
  name = 'mock';

  async translate(
    nodes: TranslationNode[],
    options?: TranslationOptions,
  ): Promise<TranslationResult> {
    const logger = options?.logger;
    logger?.info('Mock translation started', { nodesCount: nodes.length });

    const patches: TranslationPatch[] = nodes.map((node) => {
      if (node.kind === 'text') {
        return {
          kind: 'text',
          nodeId: node.nodeId,
          text: `[ZH] ${node.text}`,
        };
      } else {
        // Math node - remove all $ delimiters
        const fixed = node.latex.replace(/\$+/g, '');
        return {
          kind: 'math',
          nodeId: node.nodeId,
          latex: fixed,
          confidence: 'high',
        };
      }
    });

    logger?.info('Mock translation completed', {
      patchesCount: patches.length,
    });

    return {
      patches,
      metadata: {
        provider: 'mock',
        timestamp: new Date().toISOString(),
      },
    };
  }
}

/**
 * Identity translator - returns original content unchanged
 * Useful for disabling translation while keeping the pipeline intact
 */
export class IdentityTranslator implements Translator {
  name = 'identity';

  async translate(
    nodes: TranslationNode[],
    options?: TranslationOptions,
  ): Promise<TranslationResult> {
    const logger = options?.logger;
    logger?.debug('Identity translation (no-op)', { nodesCount: nodes.length });

    const patches: TranslationPatch[] = nodes.map((node) => {
      if (node.kind === 'text') {
        return {
          kind: 'text',
          nodeId: node.nodeId,
          text: node.text,
        };
      } else {
        return {
          kind: 'math',
          nodeId: node.nodeId,
          latex: node.latex,
          confidence: 'high',
        };
      }
    });

    return {
      patches,
      metadata: {
        provider: 'identity',
        timestamp: new Date().toISOString(),
      },
    };
  }
}

/**
 * Create a translator instance based on configuration
 */
export function createTranslator(provider: string = 'mock'): Translator {
  switch (provider.toLowerCase()) {
    case 'mock':
      return new MockTranslator();
    case 'identity':
    case 'none':
      return new IdentityTranslator();
    case 'deepseek': {
      // Check if API key is available
      if (!process.env.DEEPSEEK_API_KEY) {
        console.warn('DEEPSEEK_API_KEY not configured, falling back to identity translator');
        return new IdentityTranslator();
      }
      return new DeepSeekTranslator();
    }
    // Future: Add OpenAI, Claude, etc.
    // case 'openai':
    //   return new OpenAITranslator();
    default:
      console.warn(`Unknown translator provider: ${provider}, using mock translator`);
      return new MockTranslator();
  }
}

/**
 * Get translator from environment configuration
 */
export function getConfiguredTranslator(): Translator | null {
  const enabled = process.env.MARKDOWN_TRANSLATE_ENABLED === '1';
  if (!enabled) {
    return null;
  }

  const provider = process.env.MARKDOWN_TRANSLATE_PROVIDER || 'mock';
  return createTranslator(provider);
}
