/**
 * Tests for DeepSeek Translator
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { DeepSeekTranslator } from '../../scripts/markdown/deepseek-translator';
import type { TranslationNode } from '../../scripts/markdown/translator';
import * as fs from 'node:fs';

// Mock fetch globally
const originalFetch = global.fetch;

describe('DeepSeekTranslator', () => {
  let translator: DeepSeekTranslator;
  const testCacheDir = '.cache/test-markdown-translate';

  beforeEach(() => {
    // Setup test environment
    process.env.DEEPSEEK_API_KEY = 'test-api-key';
    process.env.DEEPSEEK_MODEL = 'deepseek-chat';
    process.env.DEEPSEEK_BASE_URL = 'https://api.deepseek.com';
    process.env.DEEPSEEK_REQUEST_TIMEOUT_MS = '5000';
    process.env.DEEPSEEK_MAX_BATCH_CHARS = '100';
    process.env.DEEPSEEK_MAX_CONCURRENCY = '2';
    process.env.DEEPSEEK_CACHE_ENABLED = '1';
    process.env.DEEPSEEK_CACHE_DIR = testCacheDir;

    translator = new DeepSeekTranslator();

    // Clean up test cache directory
    if (fs.existsSync(testCacheDir)) {
      fs.rmSync(testCacheDir, { recursive: true, force: true });
    }
  });

  afterEach(() => {
    // Restore fetch
    global.fetch = originalFetch;

    // Clean up
    if (fs.existsSync(testCacheDir)) {
      fs.rmSync(testCacheDir, { recursive: true, force: true });
    }

    // Reset environment
    delete process.env.DEEPSEEK_API_KEY;
    delete process.env.DEEPSEEK_MODEL;
    delete process.env.DEEPSEEK_BASE_URL;
    delete process.env.DEEPSEEK_REQUEST_TIMEOUT_MS;
    delete process.env.DEEPSEEK_MAX_BATCH_CHARS;
    delete process.env.DEEPSEEK_MAX_CONCURRENCY;
    delete process.env.DEEPSEEK_CACHE_ENABLED;
    delete process.env.DEEPSEEK_CACHE_DIR;
  });

  describe('Configuration', () => {
    it('should throw error when API key is missing', async () => {
      delete process.env.DEEPSEEK_API_KEY;
      const noKeyTranslator = new DeepSeekTranslator();

      const nodes: TranslationNode[] = [{ kind: 'text', nodeId: 'node1', text: 'Hello' }];

      await expect(noKeyTranslator.translate(nodes)).rejects.toThrow(
        'DEEPSEEK_API_KEY is not configured',
      );
    });

    it('should use default model when not specified', () => {
      delete process.env.DEEPSEEK_MODEL;
      const defaultTranslator = new DeepSeekTranslator();
      expect(defaultTranslator.name).toBe('deepseek');
    });

    it('should accept custom configuration', () => {
      const customTranslator = new DeepSeekTranslator({
        apiKey: 'custom-key',
        model: 'custom-model',
      });
      expect(customTranslator.name).toBe('deepseek');
    });
  });

  describe('Batching', () => {
    it('should create single batch for small content', async () => {
      const nodes: TranslationNode[] = [
        { kind: 'text', nodeId: 'node1', text: 'Hello' },
        { kind: 'text', nodeId: 'node2', text: 'World' },
      ];

      let fetchCallCount = 0;
      global.fetch = vi.fn(async () => {
        fetchCallCount++;
        return new Response(
          JSON.stringify({
            choices: [
              {
                message: {
                  content: JSON.stringify({
                    patches: {
                      node1: { kind: 'text', text: '你好' },
                      node2: { kind: 'text', text: '世界' },
                    },
                  }),
                },
              },
            ],
          }),
          { status: 200 },
        );
      }) as any;

      const result = await translator.translate(nodes);

      expect(fetchCallCount).toBe(1);
      expect(result.patches).toHaveLength(2);
      expect(result.metadata?.batches).toBe(1);
    });

    it('should split into multiple batches when exceeding max chars', async () => {
      // Max batch chars is 100, so these nodes should create multiple batches
      const nodes: TranslationNode[] = [
        { kind: 'text', nodeId: 'node1', text: 'A'.repeat(60) },
        { kind: 'text', nodeId: 'node2', text: 'B'.repeat(60) },
        { kind: 'text', nodeId: 'node3', text: 'C'.repeat(60) },
      ];

      let fetchCallCount = 0;
      global.fetch = vi.fn(async (url: string, options: any) => {
        fetchCallCount++;
        const body = JSON.parse(options.body);
        const userContent = body.messages[1].content;

        // Extract node IDs from prompt
        const nodeIds =
          userContent.match(/"(node\d+)":/g)?.map((m: string) => m.slice(1, -2)) || [];

        const patches: Record<string, { kind: 'text'; text: string }> = {};
        nodeIds.forEach((id: string) => {
          patches[id] = { kind: 'text', text: `Translated ${id}` };
        });

        return new Response(
          JSON.stringify({
            choices: [
              {
                message: {
                  content: JSON.stringify({ patches }),
                },
              },
            ],
          }),
          { status: 200 },
        );
      }) as any;

      const result = await translator.translate(nodes);

      expect(fetchCallCount).toBeGreaterThan(1);
      expect(result.patches).toHaveLength(3);
      expect(result.metadata?.batches).toBeGreaterThan(1);
    });

    it('should handle single oversized node in dedicated batch', async () => {
      const nodes: TranslationNode[] = [
        { kind: 'text', nodeId: 'node1', text: 'A'.repeat(150) }, // Exceeds max batch chars
        { kind: 'text', nodeId: 'node2', text: 'Small' },
      ];

      let fetchCallCount = 0;
      global.fetch = vi.fn(async (url: string, options: any) => {
        fetchCallCount++;
        const body = JSON.parse(options.body);
        const userContent = body.messages[1].content;

        const nodeIds =
          userContent.match(/"(node\d+)":/g)?.map((m: string) => m.slice(1, -2)) || [];

        const patches: Record<string, { kind: 'text'; text: string }> = {};
        nodeIds.forEach((id: string) => {
          patches[id] = { kind: 'text', text: `Translated ${id}` };
        });

        return new Response(
          JSON.stringify({
            choices: [
              {
                message: {
                  content: JSON.stringify({ patches }),
                },
              },
            ],
          }),
          { status: 200 },
        );
      }) as any;

      const result = await translator.translate(nodes);

      expect(fetchCallCount).toBe(2); // One for oversized, one for small
      expect(result.patches).toHaveLength(2);
    });
  });

  describe('JSON Schema Validation', () => {
    it('should handle valid JSON response', async () => {
      const nodes: TranslationNode[] = [{ kind: 'text', nodeId: 'node1', text: 'Hello' }];

      global.fetch = vi.fn(async () => {
        return new Response(
          JSON.stringify({
            choices: [
              {
                message: {
                  content: JSON.stringify({
                    patches: {
                      node1: { kind: 'text', text: '你好' },
                    },
                  }),
                },
              },
            ],
          }),
          { status: 200 },
        );
      }) as any;

      const result = await translator.translate(nodes);

      expect(result.patches).toHaveLength(1);
      expect((result.patches[0] as any).text).toBe('你好');
      expect(result.metadata?.successBatches).toBe(1);
      expect(result.metadata?.failedBatches).toBe(0);
    });

    it('should handle non-JSON response with fallback', async () => {
      const nodes: TranslationNode[] = [{ kind: 'text', nodeId: 'node1', text: 'Hello' }];

      global.fetch = vi.fn(async () => {
        return new Response(
          JSON.stringify({
            choices: [
              {
                message: {
                  content: 'This is not valid JSON',
                },
              },
            ],
          }),
          { status: 200 },
        );
      }) as any;

      const result = await translator.translate(nodes);

      // Should fallback to original text
      expect(result.patches).toHaveLength(1);
      expect((result.patches[0] as any).text).toBe('Hello');
      expect(result.metadata?.failedBatches).toBe(1);
    });

    it('should handle response missing patches field', async () => {
      const nodes: TranslationNode[] = [{ kind: 'text', nodeId: 'node1', text: 'Hello' }];

      global.fetch = vi.fn(async () => {
        return new Response(
          JSON.stringify({
            choices: [
              {
                message: {
                  content: JSON.stringify({ result: 'invalid structure' }),
                },
              },
            ],
          }),
          { status: 200 },
        );
      }) as any;

      const result = await translator.translate(nodes);

      // Should fallback to original text
      expect(result.patches).toHaveLength(1);
      expect((result.patches[0] as any).text).toBe('Hello');
      expect(result.metadata?.failedBatches).toBe(1);
    });

    it('should handle response missing node IDs', async () => {
      const nodes: TranslationNode[] = [
        { kind: 'text', nodeId: 'node1', text: 'Hello' },
        { kind: 'text', nodeId: 'node2', text: 'World' },
      ];

      global.fetch = vi.fn(async () => {
        return new Response(
          JSON.stringify({
            choices: [
              {
                message: {
                  content: JSON.stringify({
                    patches: {
                      node1: { kind: 'text', text: '你好' },
                      // Missing node2
                    },
                  }),
                },
              },
            ],
          }),
          { status: 200 },
        );
      }) as any;

      const result = await translator.translate(nodes);

      // Should add fallback for missing node
      expect(result.patches).toHaveLength(2);
      const patch1 = result.patches.find((p) => p.nodeId === 'node1') as any;
      const patch2 = result.patches.find((p) => p.nodeId === 'node2') as any;
      expect(patch1?.text).toBe('你好');
      expect(patch2?.text).toBe('World'); // Falls back to original
    });
  });

  describe('Timeout and Network Errors', () => {
    it('should handle timeout with fallback', async () => {
      const nodes: TranslationNode[] = [{ kind: 'text', nodeId: 'node1', text: 'Hello' }];

      global.fetch = vi.fn(async (url: string, options: any) => {
        // Simulate a long-running request that gets aborted
        return new Promise((resolve, reject) => {
          const timeoutId = setTimeout(() => {
            resolve(
              new Response(
                JSON.stringify({
                  choices: [{ message: { content: '{"patches": {"node1": "你好"}}' } }],
                }),
                { status: 200 },
              ),
            );
          }, 10000); // Much longer than timeout

          if (options.signal) {
            options.signal.addEventListener('abort', () => {
              clearTimeout(timeoutId);
              const abortError = new Error('The operation was aborted');
              abortError.name = 'AbortError';
              reject(abortError);
            });
          }
        });
      }) as any;

      const result = await translator.translate(nodes);

      // Should fallback to original text
      expect(result.patches).toHaveLength(1);
      expect((result.patches[0] as any).text).toBe('Hello');
      expect(result.metadata?.failedBatches).toBe(1);
    }, 10000); // Increase test timeout

    it('should handle HTTP error with fallback', async () => {
      const nodes: TranslationNode[] = [{ kind: 'text', nodeId: 'node1', text: 'Hello' }];

      global.fetch = vi.fn(async () => {
        return new Response('Internal Server Error', { status: 500 });
      }) as any;

      const result = await translator.translate(nodes);

      // Should fallback to original text
      expect(result.patches).toHaveLength(1);
      expect((result.patches[0] as any).text).toBe('Hello');
      expect(result.metadata?.failedBatches).toBe(1);
    });

    it('should handle network error with fallback', async () => {
      const nodes: TranslationNode[] = [{ kind: 'text', nodeId: 'node1', text: 'Hello' }];

      global.fetch = vi.fn(async () => {
        throw new Error('Network error');
      }) as any;

      const result = await translator.translate(nodes);

      // Should fallback to original text
      expect(result.patches).toHaveLength(1);
      expect((result.patches[0] as any).text).toBe('Hello');
      expect(result.metadata?.failedBatches).toBe(1);
    });
  });

  describe('Caching', () => {
    it('should cache successful translations', async () => {
      const nodes: TranslationNode[] = [{ kind: 'text', nodeId: 'node1', text: 'Hello World' }];

      let fetchCallCount = 0;
      global.fetch = vi.fn(async () => {
        fetchCallCount++;
        return new Response(
          JSON.stringify({
            choices: [
              {
                message: {
                  content: JSON.stringify({
                    patches: { node1: { kind: 'text', text: '你好世界' } },
                  }),
                },
              },
            ],
          }),
          { status: 200 },
        );
      }) as any;

      // First call should hit API
      const result1 = await translator.translate(nodes);
      expect(fetchCallCount).toBe(1);
      expect(result1.metadata?.cacheHits).toBe(0);

      // Second call should use cache
      const result2 = await translator.translate(nodes);
      expect(fetchCallCount).toBe(1); // No additional fetch
      expect(result2.metadata?.cacheHits).toBe(1);
      expect((result2.patches[0] as any).text).toBe('你好世界');
    });

    it('should not cache when caching is disabled', async () => {
      process.env.DEEPSEEK_CACHE_ENABLED = '0';
      const noCacheTranslator = new DeepSeekTranslator();

      const nodes: TranslationNode[] = [{ kind: 'text', nodeId: 'node1', text: 'Hello' }];

      let fetchCallCount = 0;
      global.fetch = vi.fn(async () => {
        fetchCallCount++;
        return new Response(
          JSON.stringify({
            choices: [
              {
                message: {
                  content: JSON.stringify({
                    patches: { node1: { kind: 'text', text: '你好' } },
                  }),
                },
              },
            ],
          }),
          { status: 200 },
        );
      }) as any;

      await noCacheTranslator.translate(nodes);
      await noCacheTranslator.translate(nodes);

      // Should make two API calls
      expect(fetchCallCount).toBe(2);
    });

    it('should use different cache keys for different content', async () => {
      const nodes1: TranslationNode[] = [{ kind: 'text', nodeId: 'node1', text: 'Hello' }];
      const nodes2: TranslationNode[] = [{ kind: 'text', nodeId: 'node1', text: 'World' }];

      let fetchCallCount = 0;
      global.fetch = vi.fn(async (url: string, options: any) => {
        fetchCallCount++;
        const body = JSON.parse(options.body);
        const userContent = body.messages[1].content;

        return new Response(
          JSON.stringify({
            choices: [
              {
                message: {
                  content: JSON.stringify({
                    patches: {
                      node1: { kind: 'text', text: `Translated: ${userContent.slice(0, 20)}...` },
                    },
                  }),
                },
              },
            ],
          }),
          { status: 200 },
        );
      }) as any;

      await translator.translate(nodes1);
      await translator.translate(nodes2);

      // Should make two API calls for different content
      expect(fetchCallCount).toBe(2);
    });
  });

  describe('Concurrency Control', () => {
    it('should respect max concurrency limit', async () => {
      // Create many batches
      const nodes: TranslationNode[] = Array.from({ length: 10 }, (_, i) => ({
        kind: 'text',
        nodeId: `node${i}`,
        text: 'A'.repeat(60), // Each creates its own batch
      }));

      let concurrentRequests = 0;
      let maxConcurrent = 0;

      global.fetch = vi.fn(async () => {
        concurrentRequests++;
        maxConcurrent = Math.max(maxConcurrent, concurrentRequests);

        // Simulate network delay
        await new Promise((resolve) => setTimeout(resolve, 10));

        const result = new Response(
          JSON.stringify({
            choices: [
              {
                message: {
                  content: JSON.stringify({
                    patches: { node0: { kind: 'text', text: 'Translated' } },
                  }),
                },
              },
            ],
          }),
          { status: 200 },
        );

        concurrentRequests--;
        return result;
      }) as any;

      await translator.translate(nodes);

      // Max concurrency is set to 2 in beforeEach
      expect(maxConcurrent).toBeLessThanOrEqual(2);
    });
  });

  describe('Empty Input', () => {
    it('should handle empty node list', async () => {
      const nodes: TranslationNode[] = [];

      const result = await translator.translate(nodes);

      expect(result.patches).toHaveLength(0);
      expect(result.metadata?.provider).toBe('deepseek');
    });
  });

  describe('Partial Batch Failure', () => {
    it('should handle partial batch failures gracefully', async () => {
      const nodes: TranslationNode[] = [
        { kind: 'text', nodeId: 'node1', text: 'A'.repeat(60) },
        { kind: 'text', nodeId: 'node2', text: 'B'.repeat(60) },
        { kind: 'text', nodeId: 'node3', text: 'C'.repeat(60) },
      ];

      let callCount = 0;
      global.fetch = vi.fn(async (url: string, options: any) => {
        callCount++;
        const body = JSON.parse(options.body);
        const userContent = body.messages[1].content;
        const nodeIds =
          userContent.match(/"(node\d+)":/g)?.map((m: string) => m.slice(1, -2)) || [];

        // Fail first batch, succeed others
        if (callCount === 1) {
          throw new Error('First batch failed');
        }

        const patches: Record<string, { kind: 'text'; text: string }> = {};
        nodeIds.forEach((id: string) => {
          patches[id] = { kind: 'text', text: `Translated ${id}` };
        });

        return new Response(
          JSON.stringify({
            choices: [
              {
                message: {
                  content: JSON.stringify({ patches }),
                },
              },
            ],
          }),
          { status: 200 },
        );
      }) as any;

      const result = await translator.translate(nodes);

      // All nodes should have patches (failed ones fallback to original)
      expect(result.patches.length).toBeGreaterThan(0);
      expect(result.metadata?.failedBatches).toBeGreaterThan(0);
      expect(result.metadata?.successBatches).toBeGreaterThan(0);
    });
  });
});
