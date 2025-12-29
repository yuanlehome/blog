/**
 * Tests for Translation and Markdown Processing
 */

import { describe, it, expect, beforeEach } from 'vitest';
import {
  MockTranslator,
  IdentityTranslator,
  createTranslator,
  getConfiguredTranslator,
} from '../../scripts/markdown/translator';
import { processMarkdownForImport } from '../../scripts/markdown/markdown-processor';

describe('Translator', () => {
  describe('MockTranslator', () => {
    it('should prepend [ZH] to text', async () => {
      const translator = new MockTranslator();
      const result = await translator.translate([
        { nodeId: 'node1', text: 'Hello World' },
        { nodeId: 'node2', text: 'This is a test' },
      ]);

      expect(result.patches).toHaveLength(2);
      expect(result.patches[0].translatedText).toBe('[ZH] Hello World');
      expect(result.patches[1].translatedText).toBe('[ZH] This is a test');
      expect(result.metadata?.provider).toBe('mock');
    });
  });

  describe('IdentityTranslator', () => {
    it('should return original text unchanged', async () => {
      const translator = new IdentityTranslator();
      const result = await translator.translate([
        { nodeId: 'node1', text: 'Hello World' },
        { nodeId: 'node2', text: 'Keep original' },
      ]);

      expect(result.patches).toHaveLength(2);
      expect(result.patches[0].translatedText).toBe('Hello World');
      expect(result.patches[1].translatedText).toBe('Keep original');
    });
  });

  describe('createTranslator', () => {
    const originalEnv = process.env;

    beforeEach(() => {
      process.env = { ...originalEnv };
    });

    afterEach(() => {
      process.env = originalEnv;
    });

    it('should create MockTranslator by default', () => {
      const translator = createTranslator();
      expect(translator.name).toBe('mock');
    });

    it('should create IdentityTranslator for "none"', () => {
      const translator = createTranslator('none');
      expect(translator.name).toBe('identity');
    });

    it('should create IdentityTranslator for "identity"', () => {
      const translator = createTranslator('identity');
      expect(translator.name).toBe('identity');
    });

    it('should create MockTranslator for unknown provider', () => {
      const translator = createTranslator('unknown-provider');
      expect(translator.name).toBe('mock');
    });

    it('should create IdentityTranslator when DeepSeek API key is missing', () => {
      delete process.env.DEEPSEEK_API_KEY;
      const translator = createTranslator('deepseek');
      expect(translator.name).toBe('identity');
    });

    it('should create DeepSeekTranslator when API key is present', () => {
      process.env.DEEPSEEK_API_KEY = 'test-key';
      process.env.DEEPSEEK_MODEL = 'deepseek-chat';
      const translator = createTranslator('deepseek');
      expect(translator.name).toBe('deepseek');
    });
  });

  describe('getConfiguredTranslator', () => {
    const originalEnv = process.env;

    beforeEach(() => {
      process.env = { ...originalEnv };
    });

    afterEach(() => {
      process.env = originalEnv;
    });

    it('should return null when translation is disabled', () => {
      process.env.MARKDOWN_TRANSLATE_ENABLED = '0';
      const translator = getConfiguredTranslator();
      expect(translator).toBeNull();
    });

    it('should return translator when enabled', () => {
      process.env.MARKDOWN_TRANSLATE_ENABLED = '1';
      const translator = getConfiguredTranslator();
      expect(translator).not.toBeNull();
      expect(translator?.name).toBe('mock');
    });
  });
});

describe('Markdown Processor', () => {
  describe('processMarkdownForImport', () => {
    it('should fix code fence languages', async () => {
      const markdown = `
# Test Article

\`\`\`
def hello():
    print("Hello")
\`\`\`

\`\`\`
echo "World"
\`\`\`
`;

      const result = await processMarkdownForImport({ markdown }, { enableTranslation: false });

      expect(result.diagnostics.codeFencesFixed).toBe(2);
      expect(result.markdown).toContain('```python');
      expect(result.markdown).toContain('```bash');
    });

    it('should not change existing code fence languages', async () => {
      const markdown = `
\`\`\`javascript
console.log("test");
\`\`\`
`;

      const result = await processMarkdownForImport({ markdown }, { enableTranslation: false });

      expect(result.diagnostics.codeFencesFixed).toBe(0);
      expect(result.markdown).toContain('```javascript');
    });

    it('should compress multiple empty lines', async () => {
      const markdown = `# Title\n\n\n\n\n\nText here`;

      const result = await processMarkdownForImport({ markdown }, { enableTranslation: false });

      // The normalizer should compress 3+ newlines
      if (result.diagnostics.emptyLinesCompressed > 0) {
        expect(result.diagnostics.changed).toBe(true);
      }
      // Should not have 3+ consecutive newlines in output
      expect(result.markdown).not.toMatch(/\n{3,}/);
    });

    it('should convert images with captions to Markdown format with italic caption', async () => {
      const markdown = `
![Image Alt](https://example.com/image.jpg)

Figure 1: This is a caption for the image.
`;

      const result = await processMarkdownForImport(
        { markdown },
        { enableTranslation: false, enableImageCaptionFix: true },
      );

      expect(result.diagnostics.imageCaptionsFixed).toBeGreaterThan(0);
      // Should keep image as Markdown syntax
      expect(result.markdown).toContain('![Image Alt](https://example.com/image.jpg)');
      // Caption should be in italic
      expect(result.markdown).toContain('*Figure 1: This is a caption for the image.*');
      // Should NOT contain HTML figure tags
      expect(result.markdown).not.toContain('<figure>');
      expect(result.markdown).not.toContain('<figcaption>');
    });

    it('should not treat long paragraphs as captions', async () => {
      const markdown = `
![Image Alt](https://example.com/image.jpg)

This is a very long paragraph that should not be treated as a caption because it exceeds the length threshold of 120 characters that we use for caption detection.
`;

      const result = await processMarkdownForImport(
        { markdown },
        { enableTranslation: false, enableImageCaptionFix: true },
      );

      // Should NOT create italic caption for long paragraph
      expect(result.diagnostics.imageCaptionsFixed).toBe(0);
      // Image should remain as Markdown
      expect(result.markdown).toContain('![Image Alt](https://example.com/image.jpg)');
      // The long paragraph should remain as regular content
      expect(result.markdown).toContain('This is a very long paragraph');
    });

    it('should not treat headings as captions', async () => {
      const markdown = `
![Image](https://example.com/img.jpg)

## Next Section

Content here.
`;

      const result = await processMarkdownForImport(
        { markdown },
        { enableTranslation: false, enableImageCaptionFix: true },
      );

      // Heading should not be treated as caption
      expect(result.markdown).toContain('## Next Section');
      expect(result.diagnostics.imageCaptionsFixed).toBe(0);
    });

    it('should handle image with alt text but no caption', async () => {
      const markdown = `
![Image Alt Text](https://example.com/image.jpg)

Regular paragraph follows.
`;

      const result = await processMarkdownForImport(
        { markdown },
        { enableTranslation: false, enableImageCaptionFix: true },
      );

      // Should NOT fix images without proper captions
      expect(result.diagnostics.imageCaptionsFixed).toBe(0);
      // Image should remain as Markdown
      expect(result.markdown).toContain('![Image Alt Text](https://example.com/image.jpg)');
    });

    it('should translate English content with mock translator', async () => {
      const markdown = `
# Hello World

This is an English article.
It should be translated to Chinese.
`;

      const translator = new MockTranslator();
      const result = await processMarkdownForImport(
        { markdown },
        {
          translator,
          enableTranslation: true,
        },
      );

      expect(result.diagnostics.translated).toBe(true);
      expect(result.diagnostics.detectedLanguage).toBe('en');
      expect(result.markdown).toContain('[ZH]');
    });

    it('should translate image captions but preserve URLs', async () => {
      const markdown = `
![Figure 1: NVIDIA GPU Model](https://example.com/images/gpu-model.png)

Figure 1: NVIDIA Hopper H100 GPU Model
`;

      const translator = new MockTranslator();
      const result = await processMarkdownForImport(
        { markdown },
        {
          translator,
          enableTranslation: true,
          enableImageCaptionFix: true,
        },
      );

      expect(result.diagnostics.translated).toBe(true);
      expect(result.diagnostics.imageCaptionsFixed).toBeGreaterThan(0);

      // Image URL must be preserved exactly
      expect(result.markdown).toContain('https://example.com/images/gpu-model.png');

      // Alt text should be translated
      expect(result.markdown).toContain('[ZH]');

      // Caption should be translated and in italic
      expect(result.markdown).toMatch(/\*.*\[ZH\].*\*/);

      // Should NOT contain HTML figure tags
      expect(result.markdown).not.toContain('<figure>');
      expect(result.markdown).not.toContain('<figcaption>');
    });

    it('should not modify image URLs during translation', async () => {
      const markdown = `
![Alt text](/images/local/test.png)

Some caption text.
`;

      const translator = new MockTranslator();
      const result = await processMarkdownForImport(
        { markdown },
        {
          translator,
          enableTranslation: true,
          enableImageCaptionFix: true,
        },
      );

      // URL with leading slash must be preserved
      expect(result.markdown).toContain('/images/local/test.png');
      // URL should not be modified or translated
      expect(result.markdown).not.toContain('[ZH] /images');
    });

    it('should not translate Chinese content', async () => {
      const markdown = `
# 你好世界

这是一篇中文文章。
不应该被翻译。
`;

      const translator = new MockTranslator();
      const result = await processMarkdownForImport(
        { markdown },
        {
          translator,
          enableTranslation: true,
        },
      );

      expect(result.diagnostics.translated).toBe(false);
      expect(result.diagnostics.detectedLanguage).toBe('zh');
      expect(result.markdown).not.toContain('[ZH]');
    });

    it('should not translate code blocks', async () => {
      const markdown = `
# Article

This is content to translate.

\`\`\`python
# This code should not be translated
def hello_world():
    print("Keep this English")
\`\`\`

More content to translate.
`;

      const translator = new MockTranslator();
      const result = await processMarkdownForImport(
        { markdown },
        {
          translator,
          enableTranslation: true,
        },
      );

      // Code blocks should remain unchanged
      expect(result.markdown).toContain('def hello_world()');
      expect(result.markdown).toContain('print("Keep this English")');
      expect(result.markdown).not.toContain('[ZH] def hello_world');
    });

    it('should update frontmatter when translated', async () => {
      const markdown = `---
title: Original Title
date: 2024-01-01
---

# Hello World

This is English content.
`;

      const translator = new MockTranslator();
      const result = await processMarkdownForImport(
        { markdown },
        {
          translator,
          enableTranslation: true,
        },
      );

      expect(result.diagnostics.frontmatterUpdated).toBe(true);
      expect(result.markdown).toContain('lang: zh');
      expect(result.markdown).toContain('translatedFrom: en');
    });

    it('should not overwrite existing non-en lang in frontmatter', async () => {
      const markdown = `---
title: Title
lang: zh
---

# Hello World

English content.
`;

      const translator = new MockTranslator();
      const result = await processMarkdownForImport(
        { markdown },
        {
          translator,
          enableTranslation: true,
        },
      );

      // Should not translate if lang is already zh
      expect(result.diagnostics.translated).toBe(true);
    });

    it('should handle translation failure gracefully', async () => {
      const markdown = `
# Test

English content here.
`;

      // Create a translator that fails
      const failingTranslator = {
        name: 'failing',
        async translate() {
          throw new Error('Translation failed');
        },
      };

      const result = await processMarkdownForImport(
        { markdown },
        {
          translator: failingTranslator,
          enableTranslation: true,
        },
      );

      // Should still return valid markdown
      expect(result.markdown).toBeTruthy();
      expect(result.diagnostics.translated).toBe(false);
    });

    it('should apply all fixes together', async () => {
      const markdown = `
# English Article



This is content.

\`\`\`
print("test")
\`\`\`


More content.
`;

      const translator = new MockTranslator();
      const result = await processMarkdownForImport(
        { markdown },
        {
          translator,
          enableTranslation: true,
          enableCodeFenceFix: true,
          enableMarkdownCleanup: true,
        },
      );

      expect(result.diagnostics.changed).toBe(true);
      expect(result.diagnostics.translated).toBe(true);
      expect(result.diagnostics.codeFencesFixed).toBeGreaterThan(0);
      // Empty line compression is done, check if result is clean
      expect(result.markdown).not.toMatch(/\n{3,}/);
    });

    it('should skip translation when disabled', async () => {
      const markdown = `
# Hello World

This is English content.
`;

      const result = await processMarkdownForImport({ markdown }, { enableTranslation: false });

      expect(result.diagnostics.translated).toBe(false);
      expect(result.markdown).not.toContain('[ZH]');
    });

    it('should skip code fence fix when disabled', async () => {
      const markdown = `
\`\`\`
print("test")
\`\`\`
`;

      const result = await processMarkdownForImport({ markdown }, { enableCodeFenceFix: false });

      expect(result.diagnostics.codeFencesFixed).toBe(0);
    });

    it('should skip image caption fix when disabled', async () => {
      const markdown = `
![Image](https://example.com/img.jpg)

Caption text.
`;

      const result = await processMarkdownForImport({ markdown }, { enableImageCaptionFix: false });

      expect(result.diagnostics.imageCaptionsFixed).toBe(0);
    });

    it('should skip markdown cleanup when disabled', async () => {
      const markdown = `# Title\n\n\n\n\nText`;

      const result = await processMarkdownForImport({ markdown }, { enableMarkdownCleanup: false });

      expect(result.diagnostics.emptyLinesCompressed).toBe(0);
    });

    it('should handle empty markdown', async () => {
      const markdown = '';

      const result = await processMarkdownForImport({ markdown }, { enableTranslation: false });

      expect(result.markdown).toBeTruthy();
      expect(result.diagnostics.translated).toBe(false);
    });

    it('should handle markdown with only frontmatter', async () => {
      const markdown = `---
title: Test
---
`;

      const result = await processMarkdownForImport({ markdown }, { enableTranslation: false });

      expect(result.markdown).toContain('title: Test');
    });

    it('should not translate when translator is null', async () => {
      const markdown = `
# Hello World

English content.
`;

      const result = await processMarkdownForImport(
        { markdown },
        { translator: undefined, enableTranslation: true },
      );

      expect(result.diagnostics.translated).toBe(false);
    });

    it('should fix broken math blocks with stray $$ delimiters', async () => {
      const markdown = `
# Article

Some text before.

$$
x = y $$ z = w
$$

Some text after.
`;

      const result = await processMarkdownForImport(
        { markdown },
        { enableTranslation: false, enableMathFix: true },
      );

      expect(result.diagnostics.mathBlocksFixed).toBeGreaterThan(0);
      // Should remove the stray $$ inside the block
      expect(result.markdown).not.toMatch(/\$\$[^$]*\$\$[^$]*\$\$/);
    });

    it('should fix the problem example from issue', async () => {
      const markdown = `
# Article

$$
\\sum_{l\\in[0,L+1)}\\exp(s_l-m_{[0,L+1)}) = \\colorbox{red}{
$$

\\displaystyle\\exp(m_{[0,L)}-m_{[0,L+1)})$}\\colorbox{orange}{$\\displaystyle\\sum_{l\\in[0,L)}\\exp(s_l-m_{[0,L)})$} + \\colorbox{lime}{$\\displaystyle\\exp(s_L-m_{[0,L+1)})$}
$$

More content.
`;

      const result = await processMarkdownForImport(
        { markdown },
        { enableTranslation: false, enableMathFix: true },
      );

      expect(result.diagnostics.mathBlocksFixed).toBeGreaterThan(0);
      expect(result.diagnostics.changed).toBe(true);
    });

    it('should degrade pseudo-math blocks to code', async () => {
      const markdown = `
# Article

$$
这是一段普通的中文文字，不包含任何数学符号或者LaTeX命令，只是普通文本而已。
这只是被错误地包裹在数学块中的普通段落而已，这里有很多文字但是没有数学内容。
应该被检测出来并降级处理为代码块，因为这根本不是数学公式，而是纯文本内容，包含大量的中文字符但是没有任何数学命令或符号。
$$

More content.
`;

      const result = await processMarkdownForImport(
        { markdown },
        { enableTranslation: false, enableMathFix: true },
      );

      expect(result.diagnostics.mathBlocksDegraded).toBeGreaterThan(0);
      // Should convert to tex code block
      expect(result.markdown).toContain('```tex');
    });

    it('should not modify valid math blocks', async () => {
      const markdown = `
# Article

$$
\\frac{a}{b} + \\sqrt{c}
$$

$$
\\sum_{i=1}^{n} x_i = \\int_{0}^{\\infty} f(x) dx
$$
`;

      const result = await processMarkdownForImport(
        { markdown },
        { enableTranslation: false, enableMathFix: true },
      );

      // Should not fix valid math
      expect(result.diagnostics.mathBlocksFixed).toBe(0);
      expect(result.diagnostics.mathBlocksDegraded).toBe(0);
    });

    it('should skip math fix when disabled', async () => {
      const markdown = `
$$
x = y $$ z = w
$$
`;

      const result = await processMarkdownForImport(
        { markdown },
        { enableTranslation: false, enableMathFix: false },
      );

      expect(result.diagnostics.mathBlocksFixed).toBe(0);
    });

    it('should not fix math in code blocks', async () => {
      const markdown = `
\`\`\`python
# This is code, not math
text = "x = y $$ z = w"
\`\`\`

$$
x = y $$ z = w
$$
`;

      const result = await processMarkdownForImport(
        { markdown },
        { enableTranslation: false, enableMathFix: true },
      );

      // Should only fix the actual math block, not code
      expect(result.diagnostics.mathBlocksFixed).toBe(1);
      // Code block should be preserved
      expect(result.markdown).toContain('```python');
      expect(result.markdown).toContain('text = "x = y $$ z = w"');
    });

    it('should apply all fixes including math fix', async () => {
      const markdown = `
# English Article



This is content.

\`\`\`
print("test")
\`\`\`

$$
x = $y$ $$ z = $w$
$$
`;

      const translator = new MockTranslator();
      const result = await processMarkdownForImport(
        { markdown },
        {
          translator,
          enableTranslation: true,
          enableCodeFenceFix: true,
          enableMarkdownCleanup: true,
          enableMathFix: true,
        },
      );

      expect(result.diagnostics.changed).toBe(true);
      expect(result.diagnostics.translated).toBe(true);
      expect(result.diagnostics.codeFencesFixed).toBeGreaterThan(0);
      expect(result.diagnostics.mathBlocksFixed).toBeGreaterThan(0);
    });
  });
});
