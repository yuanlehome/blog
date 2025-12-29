/**
 * Translation Module
 *
 * Provides translation interfaces and implementations for markdown content.
 * Supports multiple translation providers with a mock implementation for testing.
 */

export interface TranslationNode {
  nodeId: string;
  text: string;
  context?: string; // Optional context for better translation
}

export interface TranslationPatch {
  nodeId: string;
  translatedText: string;
}

export interface TranslationResult {
  patches: TranslationPatch[];
  metadata?: {
    provider: string;
    model?: string;
    timestamp: string;
  };
}

/**
 * Base translator interface
 */
export interface Translator {
  name: string;
  translate(nodes: TranslationNode[]): Promise<TranslationResult>;
}

/**
 * Mock translator for testing
 * Prepends "[ZH] " to each text to simulate translation
 */
export class MockTranslator implements Translator {
  name = 'mock';

  async translate(nodes: TranslationNode[]): Promise<TranslationResult> {
    const patches: TranslationPatch[] = nodes.map((node) => ({
      nodeId: node.nodeId,
      translatedText: `[ZH] ${node.text}`,
    }));

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
 * Identity translator - returns original text unchanged
 * Useful for disabling translation while keeping the pipeline intact
 */
export class IdentityTranslator implements Translator {
  name = 'identity';

  async translate(nodes: TranslationNode[]): Promise<TranslationResult> {
    const patches: TranslationPatch[] = nodes.map((node) => ({
      nodeId: node.nodeId,
      translatedText: node.text,
    }));

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
    // Future: Add OpenAI, DeepSeek, Claude, etc.
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
