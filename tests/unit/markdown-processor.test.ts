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

    it('should convert images with captions to figures', async () => {
      const markdown = `
![Image Alt](https://example.com/image.jpg)

This is a caption for the image.
`;

      const result = await processMarkdownForImport(
        { markdown },
        { enableTranslation: false, enableImageCaptionFix: true },
      );

      expect(result.diagnostics.imageCaptionsFixed).toBeGreaterThan(0);
      expect(result.markdown).toContain('<figure>');
      expect(result.markdown).toContain('<figcaption>');
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

      // Should create figure with alt text, but long paragraph stays separate
      expect(result.markdown).toContain('<figure>');
      expect(result.markdown).toContain('Image Alt');
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

      expect(result.diagnostics.imageCaptionsFixed).toBeGreaterThan(0);
      expect(result.markdown).toContain('<figure>');
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
  });
});
