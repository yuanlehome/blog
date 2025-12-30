/**
 * LLM Math Fixer Module
 *
 * Uses LLM to fix broken LaTeX math blocks with:
 * - Strict block math `$` cleaning (no `$` allowed in fixed output)
 * - Automatic batching and deduplication
 * - Timeout and degradation strategy
 * - Mock provider support for CI (no external API calls)
 *
 * Key Fix Target: Inline math `$` mixed inside block math, intertwined with
 * `\colorbox{...}{...}` / `\displaystyle` commands, causing patterns like:
 * - `$}` (dollar before closing brace)
 * - `{$\displaystyle...$}` (dollar inside braces)
 * - Extra `$` at the end
 */

import * as crypto from 'node:crypto';

/**
 * Math fix request - content to fix
 */
export interface MathFixRequest {
  id: string; // Unique identifier for deduplication
  content: string; // Raw LaTeX content (without outer $$)
}

/**
 * Math fix result
 */
export interface MathFixResult {
  id: string;
  fixed: string; // Fixed LaTeX content
  confidence: 'high' | 'medium' | 'low';
  notes: string[]; // Human-readable fix notes
}

/**
 * LLM Math Fixer configuration
 */
export interface LLMMathFixerConfig {
  provider: 'mock' | 'deepseek'; // Which LLM provider to use
  apiKey?: string; // API key for real provider
  model?: string; // Model name
  baseUrl?: string; // API base URL
  requestTimeout?: number; // Request timeout in ms
  maxBatchSize?: number; // Max items per batch
  maxConcurrency?: number; // Max concurrent requests
}

/**
 * LLM provider interface for math fixing
 */
export interface MathFixProvider {
  name: string;
  fixBatch(requests: MathFixRequest[]): Promise<MathFixResult[]>;
}

/**
 * Mock provider for testing/CI - applies deterministic rules
 */
class MockMathFixProvider implements MathFixProvider {
  name = 'mock';

  async fixBatch(requests: MathFixRequest[]): Promise<MathFixResult[]> {
    return requests.map((req) => {
      let fixed = req.content;
      const notes: string[] = [];
      let hasChanges = false;

      // Remove all unescaped $ characters (but preserve \$)
      let dollarCount = 0;
      let result = '';
      for (let i = 0; i < fixed.length; i++) {
        if (fixed[i] === '\\' && i + 1 < fixed.length && fixed[i + 1] === '$') {
          // Preserve escaped dollar
          result += '\\$';
          i++; // Skip next character
          continue;
        }
        if (fixed[i] === '$') {
          dollarCount++;
          hasChanges = true;
          // Skip unescaped dollar
          continue;
        }
        result += fixed[i];
      }
      fixed = result;

      if (dollarCount > 0) {
        notes.push(`Removed ${dollarCount} $ delimiter(s)`);
      }

      // Remove \displaystyle (not needed in block math, can cause issues)
      if (fixed.includes('\\displaystyle')) {
        fixed = fixed.replace(/\\displaystyle\s*/g, '');
        notes.push('Removed \\displaystyle commands');
        hasChanges = true;
      }

      // Basic brace balancing
      const braceBalance = this.countBraces(fixed);
      if (braceBalance > 0 && braceBalance <= 3) {
        fixed = fixed + '}'.repeat(braceBalance);
        notes.push(`Added ${braceBalance} closing brace(s)`);
        hasChanges = true;
      }

      // Determine confidence
      let confidence: 'high' | 'medium' | 'low' = 'high';
      if (braceBalance < 0 || braceBalance > 3) {
        confidence = 'low';
        notes.push('Unbalanced braces detected');
      }

      // Only trim if we made changes, preserve original whitespace otherwise
      const finalFixed = hasChanges ? fixed.trim() : fixed;

      return {
        id: req.id,
        fixed: finalFixed,
        confidence,
        notes,
      };
    });
  }

  private countBraces(text: string): number {
    let count = 0;
    let i = 0;
    while (i < text.length) {
      if (text[i] === '\\' && i + 1 < text.length) {
        i += 2; // Skip escaped characters
        continue;
      }
      if (text[i] === '{') count++;
      else if (text[i] === '}') count--;
      i++;
    }
    return count;
  }
}

/**
 * DeepSeek provider for real LLM-based fixing
 */
class DeepSeekMathFixProvider implements MathFixProvider {
  name = 'deepseek';
  private config: Required<LLMMathFixerConfig>;

  constructor(config: LLMMathFixerConfig) {
    this.config = {
      provider: 'deepseek',
      apiKey: config.apiKey || '',
      model: config.model || 'deepseek-chat',
      baseUrl: config.baseUrl || 'https://api.deepseek.com',
      requestTimeout: config.requestTimeout || 60000,
      maxBatchSize: config.maxBatchSize || 10,
      maxConcurrency: config.maxConcurrency || 2,
    };

    if (!this.config.apiKey) {
      throw new Error('DeepSeek API key is required');
    }
  }

  async fixBatch(requests: MathFixRequest[]): Promise<MathFixResult[]> {
    const systemPrompt = this.buildSystemPrompt();
    const userPrompt = this.buildUserPrompt(requests);

    try {
      const response = await this.callDeepSeekAPI(systemPrompt, userPrompt);
      const parsed = this.parseResponse(response);
      return this.matchResults(requests, parsed);
    } catch (error) {
      console.error('DeepSeek math fix failed:', error);
      // Fallback: return original content with low confidence
      return requests.map((req) => ({
        id: req.id,
        fixed: req.content,
        confidence: 'low' as const,
        notes: ['LLM call failed, returning original content'],
      }));
    }
  }

  private buildSystemPrompt(): string {
    return `You are a LaTeX math block fixer. Your task is to fix broken block math expressions.

CRITICAL RULES FOR BLOCK MATH:
1. Input is block math content (between $$ ... $$), WITHOUT the outer $$ delimiters
2. You MUST output valid block-math LaTeX
3. **NEVER output any $ character** - this will break block math parsing
4. For patterns like \\colorbox{X}{ $ ... $ }, {$..., ...$}:
   - Remove ALL $ characters
   - Preserve the mathematical content inside
   - Fix any brace imbalance caused by removing $
5. Remove \\displaystyle commands - they're redundant in block math
6. Do NOT change mathematical meaning
7. Do NOT add explanations or expand formulas
8. Output MUST be strict JSON: { "fixes": { "id": {"fixed": "...", "confidence": "high|medium|low", "notes": ["..."]}, ... } }
9. The "fixed" field MUST NOT contain $$ or any $ character

Example bad input:
$}\\colorbox{orange}{$\\displaystyle\\sum_{i}$}

Example good output:
}\\colorbox{orange}{\\sum_{i}}

(Note: all $ removed, \\displaystyle removed, braces balanced)`;
  }

  private buildUserPrompt(requests: MathFixRequest[]): string {
    const items = requests.map((req) => `"${req.id}": ${JSON.stringify(req.content)}`).join(',\n  ');
    return `Fix these block math expressions. Remove ALL $ characters and fix brace imbalances:\n\n{\n  ${items}\n}\n\nReturn JSON with fixes for each id.`;
  }

  private async callDeepSeekAPI(systemPrompt: string, userPrompt: string): Promise<string> {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), this.config.requestTimeout);

    try {
      const response = await fetch(`${this.config.baseUrl}/v1/chat/completions`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${this.config.apiKey}`,
        },
        body: JSON.stringify({
          model: this.config.model,
          messages: [
            { role: 'system', content: systemPrompt },
            { role: 'user', content: userPrompt },
          ],
          temperature: 0.1,
          response_format: { type: 'json_object' },
        }),
        signal: controller.signal,
      });

      if (!response.ok) {
        throw new Error(`DeepSeek API error: ${response.status} ${response.statusText}`);
      }

      const data = await response.json();
      return data.choices?.[0]?.message?.content || '';
    } finally {
      clearTimeout(timeout);
    }
  }

  private parseResponse(response: string): Record<string, MathFixResult> {
    try {
      const parsed = JSON.parse(response);
      const fixes = parsed.fixes || {};
      return fixes;
    } catch (error) {
      throw new Error(`Failed to parse DeepSeek response: ${error}`);
    }
  }

  private matchResults(
    requests: MathFixRequest[],
    parsed: Record<string, MathFixResult>,
  ): MathFixResult[] {
    return requests.map((req) => {
      const result = parsed[req.id];
      if (result && result.fixed !== undefined) {
        return {
          id: req.id,
          fixed: result.fixed,
          confidence: result.confidence || 'medium',
          notes: result.notes || [],
        };
      }
      // Fallback for missing result
      return {
        id: req.id,
        fixed: req.content,
        confidence: 'low' as const,
        notes: ['LLM did not return fix for this item'],
      };
    });
  }
}

/**
 * LLM Math Fixer - main interface
 */
export class LLMMathFixer {
  private provider: MathFixProvider;
  private config: LLMMathFixerConfig;

  constructor(config: LLMMathFixerConfig = { provider: 'mock' }) {
    this.config = config;

    if (config.provider === 'deepseek') {
      this.provider = new DeepSeekMathFixProvider(config);
    } else {
      this.provider = new MockMathFixProvider();
    }
  }

  /**
   * Fix multiple math blocks with deduplication and batching
   */
  async fixMathBlocks(contents: string[]): Promise<Map<string, MathFixResult>> {
    // Deduplicate by content hash
    const uniqueMap = new Map<string, MathFixRequest>();
    const contentToHash = new Map<string, string>();

    for (const content of contents) {
      const hash = this.hashContent(content);
      contentToHash.set(content, hash);

      if (!uniqueMap.has(hash)) {
        uniqueMap.set(hash, { id: hash, content });
      }
    }

    // Process all unique requests in one batch
    const requests = Array.from(uniqueMap.values());
    const results = await this.provider.fixBatch(requests);

    // Build result map
    const resultMap = new Map<string, MathFixResult>();
    for (const result of results) {
      resultMap.set(result.id, result);
    }

    return resultMap;
  }

  /**
   * Fix a single math block
   */
  async fixMathBlock(content: string): Promise<MathFixResult> {
    const hash = this.hashContent(content);
    const request: MathFixRequest = { id: hash, content };
    const results = await this.provider.fixBatch([request]);
    return results[0];
  }

  /**
   * Hash content for deduplication
   */
  private hashContent(content: string): string {
    return crypto.createHash('sha256').update(content).digest('hex').substring(0, 16);
  }
}

/**
 * Get configured LLM math fixer from environment
 */
export function getConfiguredMathFixer(): LLMMathFixer {
  const provider = (process.env.MATH_FIX_PROVIDER || 'mock') as 'mock' | 'deepseek';

  if (provider === 'deepseek') {
    const apiKey = process.env.DEEPSEEK_API_KEY;
    if (!apiKey) {
      console.warn('DEEPSEEK_API_KEY not configured, falling back to mock math fixer');
      return new LLMMathFixer({ provider: 'mock' });
    }

    return new LLMMathFixer({
      provider: 'deepseek',
      apiKey,
      model: process.env.DEEPSEEK_MODEL || 'deepseek-chat',
      baseUrl: process.env.DEEPSEEK_BASE_URL || 'https://api.deepseek.com',
      requestTimeout: parseInt(process.env.DEEPSEEK_REQUEST_TIMEOUT_MS || '60000', 10),
      maxBatchSize: parseInt(process.env.MATH_FIX_MAX_BATCH_SIZE || '10', 10),
      maxConcurrency: parseInt(process.env.MATH_FIX_MAX_CONCURRENCY || '2', 10),
    });
  }

  return new LLMMathFixer({ provider: 'mock' });
}
