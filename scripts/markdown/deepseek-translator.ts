/**
 * DeepSeek Translation Provider
 *
 * Implements translation using DeepSeek's most advanced model with:
 * - AST + JSON patch translation strategy
 * - Automatic batching and rate limiting
 * - Timeout and error handling with graceful degradation
 * - File-based caching for cost optimization
 */

import * as crypto from 'node:crypto';
import * as fs from 'node:fs';
import * as path from 'node:path';
import type { Translator, TranslationNode, TranslationResult, TranslationPatch } from './translator.js';

/**
 * DeepSeek API configuration
 */
interface DeepSeekConfig {
  apiKey: string;
  model: string;
  baseUrl: string;
  requestTimeout: number;
  maxBatchChars: number;
  maxConcurrency: number;
  cacheEnabled: boolean;
  cacheDir: string;
}

/**
 * DeepSeek API request payload
 */
interface DeepSeekRequest {
  model: string;
  messages: Array<{
    role: 'system' | 'user';
    content: string;
  }>;
  temperature?: number;
  response_format?: { type: 'json_object' };
}

/**
 * DeepSeek API response
 */
interface DeepSeekResponse {
  id: string;
  object: string;
  created: number;
  model: string;
  choices: Array<{
    index: number;
    message: {
      role: string;
      content: string;
    };
    finish_reason: string;
  }>;
  usage?: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
}

/**
 * Translation batch for processing
 */
interface TranslationBatch {
  nodes: TranslationNode[];
  totalChars: number;
}

/**
 * Translation diagnostics
 */
interface TranslationDiagnostics {
  totalBatches: number;
  successBatches: number;
  failedBatches: number;
  cacheHits: number;
}

/**
 * Expected JSON structure from DeepSeek
 */
interface DeepSeekPatchResponse {
  patches: Record<string, string>;
}

/**
 * Get configuration from environment variables
 */
function getConfig(): DeepSeekConfig {
  return {
    apiKey: process.env.DEEPSEEK_API_KEY || '',
    model: process.env.DEEPSEEK_MODEL || 'deepseek-chat',
    baseUrl: process.env.DEEPSEEK_BASE_URL || 'https://api.deepseek.com',
    requestTimeout: parseInt(process.env.DEEPSEEK_REQUEST_TIMEOUT_MS || '60000', 10),
    maxBatchChars: parseInt(process.env.DEEPSEEK_MAX_BATCH_CHARS || '6000', 10),
    maxConcurrency: parseInt(process.env.DEEPSEEK_MAX_CONCURRENCY || '2', 10),
    cacheEnabled: process.env.DEEPSEEK_CACHE_ENABLED !== '0',
    cacheDir: process.env.DEEPSEEK_CACHE_DIR || '.cache/markdown-translate',
  };
}

/**
 * Simple semaphore for concurrency control
 */
class Semaphore {
  private available: number;
  private waiting: Array<() => void> = [];

  constructor(max: number) {
    this.available = max;
  }

  async acquire(): Promise<void> {
    if (this.available > 0) {
      this.available--;
      return;
    }

    return new Promise<void>((resolve) => {
      this.waiting.push(resolve);
    });
  }

  release(): void {
    if (this.waiting.length > 0) {
      const resolve = this.waiting.shift();
      resolve?.();
    } else {
      this.available++;
    }
  }
}

/**
 * DeepSeek Translator Implementation
 */
export class DeepSeekTranslator implements Translator {
  name = 'deepseek';
  private config: DeepSeekConfig;
  private diagnostics: TranslationDiagnostics;

  constructor(config?: Partial<DeepSeekConfig>) {
    const envConfig = getConfig();
    this.config = { ...envConfig, ...config };
    this.diagnostics = {
      totalBatches: 0,
      successBatches: 0,
      failedBatches: 0,
      cacheHits: 0,
    };

    // Ensure cache directory exists if caching is enabled
    if (this.config.cacheEnabled && this.config.cacheDir) {
      this.ensureCacheDir();
    }
  }

  /**
   * Translate nodes using DeepSeek API
   */
  async translate(nodes: TranslationNode[]): Promise<TranslationResult> {
    if (!this.config.apiKey) {
      throw new Error('DEEPSEEK_API_KEY is not configured');
    }

    if (nodes.length === 0) {
      return {
        patches: [],
        metadata: {
          provider: 'deepseek',
          model: this.config.model,
          timestamp: new Date().toISOString(),
        },
      };
    }

    // Split nodes into batches
    const batches = this.createBatches(nodes);
    this.diagnostics.totalBatches = batches.length;

    // Process batches with concurrency control
    const semaphore = new Semaphore(this.config.maxConcurrency);
    const batchPromises = batches.map((batch) => this.processBatch(batch, semaphore));
    const batchResults = await Promise.all(batchPromises);

    // Merge results
    const patches: TranslationPatch[] = [];
    const failedNodeIds = new Set<string>();

    for (const result of batchResults) {
      if (result.success) {
        patches.push(...result.patches);
      } else {
        // Track failed nodes for fallback
        result.nodeIds.forEach((id) => failedNodeIds.add(id));
      }
    }

    // Fallback: Add identity patches for failed nodes
    for (const node of nodes) {
      if (failedNodeIds.has(node.nodeId) && !patches.find((p) => p.nodeId === node.nodeId)) {
        patches.push({
          nodeId: node.nodeId,
          translatedText: node.text, // Keep original text
        });
      }
    }

    return {
      patches,
      metadata: {
        provider: 'deepseek',
        model: this.config.model,
        timestamp: new Date().toISOString(),
        batches: this.diagnostics.totalBatches,
        successBatches: this.diagnostics.successBatches,
        failedBatches: this.diagnostics.failedBatches,
        cacheHits: this.diagnostics.cacheHits,
      },
    };
  }

  /**
   * Split nodes into batches based on character count
   */
  private createBatches(nodes: TranslationNode[]): TranslationBatch[] {
    const batches: TranslationBatch[] = [];
    let currentBatch: TranslationNode[] = [];
    let currentChars = 0;

    for (const node of nodes) {
      const nodeChars = node.text.length;

      // If single node exceeds limit, create dedicated batch
      if (nodeChars > this.config.maxBatchChars) {
        if (currentBatch.length > 0) {
          batches.push({ nodes: currentBatch, totalChars: currentChars });
          currentBatch = [];
          currentChars = 0;
        }
        batches.push({ nodes: [node], totalChars: nodeChars });
        continue;
      }

      // Check if adding this node would exceed limit
      if (currentChars + nodeChars > this.config.maxBatchChars && currentBatch.length > 0) {
        batches.push({ nodes: currentBatch, totalChars: currentChars });
        currentBatch = [];
        currentChars = 0;
      }

      currentBatch.push(node);
      currentChars += nodeChars;
    }

    // Add remaining nodes
    if (currentBatch.length > 0) {
      batches.push({ nodes: currentBatch, totalChars: currentChars });
    }

    return batches;
  }

  /**
   * Process a single batch with concurrency control
   */
  private async processBatch(
    batch: TranslationBatch,
    semaphore: Semaphore,
  ): Promise<{ success: boolean; patches: TranslationPatch[]; nodeIds: string[] }> {
    const nodeIds = batch.nodes.map((n) => n.nodeId);

    try {
      await semaphore.acquire();

      // Check cache first
      const cacheKey = this.getCacheKey(batch.nodes);
      const cached = this.getCachedTranslation(cacheKey);
      if (cached) {
        this.diagnostics.cacheHits++;
        this.diagnostics.successBatches++;
        return { success: true, patches: cached, nodeIds };
      }

      // Make API request
      const patches = await this.translateBatch(batch.nodes);

      // Cache successful result
      if (patches.length > 0) {
        this.cacheTranslation(cacheKey, patches);
        this.diagnostics.successBatches++;
        return { success: true, patches, nodeIds };
      }

      this.diagnostics.failedBatches++;
      return { success: false, patches: [], nodeIds };
    } catch (error) {
      console.warn(`DeepSeek translation batch failed:`, error instanceof Error ? error.message : error);
      this.diagnostics.failedBatches++;
      return { success: false, patches: [], nodeIds };
    } finally {
      semaphore.release();
    }
  }

  /**
   * Translate a batch of nodes via DeepSeek API
   */
  private async translateBatch(nodes: TranslationNode[]): Promise<TranslationPatch[]> {
    // Build prompt
    const nodeMap: Record<string, string> = {};
    nodes.forEach((node) => {
      nodeMap[node.nodeId] = node.text;
    });

    const systemPrompt = this.buildSystemPrompt();
    const userPrompt = this.buildUserPrompt(nodeMap);

    const requestPayload: DeepSeekRequest = {
      model: this.config.model,
      messages: [
        { role: 'system', content: systemPrompt },
        { role: 'user', content: userPrompt },
      ],
      temperature: 0.3,
      response_format: { type: 'json_object' },
    };

    // Make request with timeout
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.config.requestTimeout);

    try {
      const response = await fetch(`${this.config.baseUrl}/v1/chat/completions`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${this.config.apiKey}`,
        },
        body: JSON.stringify(requestPayload),
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        throw new Error(`DeepSeek API returned ${response.status}: ${response.statusText}`);
      }

      const data = (await response.json()) as DeepSeekResponse;

      // Extract and validate response
      const content = data.choices?.[0]?.message?.content;
      if (!content) {
        throw new Error('No content in DeepSeek response');
      }

      // Parse JSON
      const parsed = this.parseAndValidateResponse(content, Object.keys(nodeMap));
      if (!parsed) {
        throw new Error('Invalid response format from DeepSeek');
      }

      // Convert to patches
      const patches: TranslationPatch[] = [];
      for (const [nodeId, translatedText] of Object.entries(parsed.patches)) {
        patches.push({ nodeId, translatedText });
      }

      return patches;
    } catch (error) {
      clearTimeout(timeoutId);
      if (error instanceof Error && error.name === 'AbortError') {
        throw new Error('DeepSeek API request timeout');
      }
      throw error;
    }
  }

  /**
   * Build system prompt for DeepSeek
   */
  private buildSystemPrompt(): string {
    return `You are a professional technical translator. Your task is to translate English technical content to Chinese.

CRITICAL RULES:
1. Output MUST be valid JSON with this exact structure: {"patches": {"node-id": "translated text", ...}}
2. Do NOT output markdown, explanations, or extra fields
3. The "patches" object MUST contain ALL node IDs from the input
4. Preserve technical terms: First occurrence as "中文（English）", then just "中文"
5. Do NOT translate: code, variable names, function names, URLs, file paths
6. Keep semantic meaning identical - no expansion or reduction
7. Preserve original punctuation style, adjust spaces only in natural language
8. If a node cannot be translated, return the original text unchanged

If you cannot produce valid JSON, output the minimal valid structure: {"patches": {}}`;
  }

  /**
   * Build user prompt with nodes to translate
   */
  private buildUserPrompt(nodeMap: Record<string, string>): string {
    const nodeList = Object.entries(nodeMap)
      .map(([id, text]) => `"${id}": "${this.escapeJsonString(text)}"`)
      .join(',\n  ');

    return `Translate the following nodes to Chinese. Return JSON with {"patches": {...}} structure.

Input nodes:
{
  ${nodeList}
}

Remember: Output only valid JSON, no markdown or explanations.`;
  }

  /**
   * Escape string for JSON
   */
  private escapeJsonString(str: string): string {
    return str
      .replace(/\\/g, '\\\\')
      .replace(/"/g, '\\"')
      .replace(/\n/g, '\\n')
      .replace(/\r/g, '\\r')
      .replace(/\t/g, '\\t');
  }

  /**
   * Parse and validate DeepSeek response
   */
  private parseAndValidateResponse(
    content: string,
    expectedNodeIds: string[],
  ): DeepSeekPatchResponse | null {
    try {
      const parsed = JSON.parse(content) as DeepSeekPatchResponse;

      // Check structure
      if (!parsed || typeof parsed !== 'object') {
        return null;
      }

      if (!parsed.patches || typeof parsed.patches !== 'object') {
        return null;
      }

      // Check all expected node IDs are present
      for (const nodeId of expectedNodeIds) {
        if (!(nodeId in parsed.patches)) {
          console.warn(`DeepSeek response missing node: ${nodeId}`);
          // Add original text as fallback
          parsed.patches[nodeId] = '';
        }

        // Validate value is string
        if (typeof parsed.patches[nodeId] !== 'string') {
          parsed.patches[nodeId] = String(parsed.patches[nodeId] || '');
        }
      }

      return parsed;
    } catch (error) {
      console.warn('Failed to parse DeepSeek response as JSON:', error);
      return null;
    }
  }

  /**
   * Generate cache key for a batch of nodes
   */
  private getCacheKey(nodes: TranslationNode[]): string {
    const content = nodes.map((n) => `${n.nodeId}:${n.text}`).join('|');
    const hash = crypto.createHash('sha256').update(content).update(this.config.model).digest('hex');
    return hash;
  }

  /**
   * Get cached translation if available
   */
  private getCachedTranslation(cacheKey: string): TranslationPatch[] | null {
    if (!this.config.cacheEnabled) {
      return null;
    }

    try {
      const cachePath = path.join(this.config.cacheDir, `${cacheKey}.json`);
      if (!fs.existsSync(cachePath)) {
        return null;
      }

      const cached = JSON.parse(fs.readFileSync(cachePath, 'utf-8')) as TranslationPatch[];
      return Array.isArray(cached) ? cached : null;
    } catch {
      return null;
    }
  }

  /**
   * Cache translation result
   */
  private cacheTranslation(cacheKey: string, patches: TranslationPatch[]): void {
    if (!this.config.cacheEnabled) {
      return;
    }

    try {
      // Ensure cache directory exists
      if (!fs.existsSync(this.config.cacheDir)) {
        fs.mkdirSync(this.config.cacheDir, { recursive: true });
      }

      const cachePath = path.join(this.config.cacheDir, `${cacheKey}.json`);
      fs.writeFileSync(cachePath, JSON.stringify(patches, null, 2), 'utf-8');
    } catch (error) {
      // Silent fail - caching is optional
      console.warn('Failed to cache translation:', error instanceof Error ? error.message : error);
    }
  }

  /**
   * Ensure cache directory exists
   */
  private ensureCacheDir(): void {
    try {
      if (!fs.existsSync(this.config.cacheDir)) {
        fs.mkdirSync(this.config.cacheDir, { recursive: true });
      }
    } catch (error) {
      console.warn('Failed to create cache directory:', error instanceof Error ? error.message : error);
      this.config.cacheEnabled = false;
    }
  }
}
