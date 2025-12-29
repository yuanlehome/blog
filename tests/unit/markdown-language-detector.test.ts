/**
 * Tests for Language Detection Module
 */

import { describe, it, expect } from 'vitest';
import { detectLanguage, shouldTranslate } from '../../scripts/markdown/language-detector';

describe('Language Detector', () => {
  describe('detectLanguage', () => {
    it('should detect English content', () => {
      const markdown = `
# Hello World

This is an article written in English.
It contains multiple paragraphs with English text.
The detection should work even with some numbers 123 and symbols.
`;

      const result = detectLanguage(markdown);
      expect(result.language).toBe('en');
      expect(result.confidence).toBeGreaterThan(0.6);
      expect(result.englishRatio).toBeGreaterThan(0.6);
    });

    it('should detect Chinese content', () => {
      const markdown = `
# 你好世界

这是一篇用中文写的文章。
它包含多个段落的中文文本。
检测应该能够正确识别中文内容。
`;

      const result = detectLanguage(markdown);
      expect(result.language).toBe('zh');
      expect(result.confidence).toBeGreaterThan(0.6);
      expect(result.chineseRatio).toBeGreaterThan(0.6);
    });

    it('should ignore code blocks when detecting language', () => {
      const markdown = `
# 你好世界

这是中文内容。

\`\`\`python
def hello_world():
    print("Hello World")
    return "This is English in code"
\`\`\`

更多中文内容在这里。
`;

      const result = detectLanguage(markdown);
      expect(result.language).toBe('zh');
    });

    it('should ignore URLs when detecting language', () => {
      const markdown = `
# 文章标题

这是一篇中文文章，包含一些链接：

- https://example.com/some-english-path
- https://github.com/username/repository

中文内容继续。
`;

      const result = detectLanguage(markdown);
      expect(result.language).toBe('zh');
    });

    it('should ignore inline code when detecting language', () => {
      const markdown = `
# 配置说明

使用 \`npm install\` 命令安装依赖。
然后运行 \`npm run dev\` 启动开发服务器。
这些是中文说明，包含英文命令。
`;

      const result = detectLanguage(markdown);
      expect(result.language).toBe('zh');
    });

    it('should return unknown for very short content', () => {
      const markdown = 'Hi';
      const result = detectLanguage(markdown);
      expect(result.language).toBe('unknown');
      expect(result.confidence).toBe(0);
    });

    it('should handle mixed content based on majority', () => {
      const markdown = `
# Article Title

This article has some English text at the beginning.

但是大部分内容都是中文的。
中文段落有很多很多。
这样检测应该识别为中文。
中文继续中文继续中文继续。
更多中文内容更多中文内容。
`;

      const result = detectLanguage(markdown);
      expect(result.language).toBe('zh');
    });

    it('should handle frontmatter correctly', () => {
      const markdown = `---
title: Hello World
date: 2024-01-01
tags: [english, article]
---

# Hello World

This is an English article with frontmatter.
The frontmatter should be ignored during detection.
`;

      const result = detectLanguage(markdown);
      expect(result.language).toBe('en');
    });
  });

  describe('shouldTranslate', () => {
    it('should return true for English content', () => {
      const markdown = 'This is an English article that needs translation.';
      expect(shouldTranslate(markdown)).toBe(true);
    });

    it('should return false for Chinese content', () => {
      const markdown = '这是一篇中文文章，不需要翻译。';
      expect(shouldTranslate(markdown)).toBe(false);
    });

    it('should return false for unknown/short content', () => {
      const markdown = 'Hi';
      expect(shouldTranslate(markdown)).toBe(false);
    });

    it('should respect custom threshold', () => {
      const markdown = `
# Title
Some English text here.
一些中文在这里。
`;
      // With default threshold (0.6), might not be clear
      // With lower threshold (0.4), English might win
      expect(shouldTranslate(markdown, 0.4)).toBe(true);
    });
  });
});
