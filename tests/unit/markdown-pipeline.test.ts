/**
 * Tests for Markdown Pipeline
 *
 * Tests the unified markdown processing pipeline for:
 * - Frontmatter uniqueness and parsing
 * - Invisible character removal
 * - Code fence fixing
 * - Markdown parsability/re-parsability
 * - Notion content regression cases
 */

import { describe, it, expect } from 'vitest';
import matter from 'gray-matter';
import { unified } from 'unified';
import remarkParse from 'remark-parse';

import {
  processMarkdown,
  processMarkdownForNotionSync,
  cleanInvisibleCharacters,
  parseFrontmatterSafe,
  mergeFrontmatter,
} from '../../scripts/lib/markdown/pipeline';

describe('cleanInvisibleCharacters', () => {
  it('should remove bidi control characters', () => {
    const input = 'Hello\u202AWorld\u202E!';
    const { cleaned, count } = cleanInvisibleCharacters(input);

    expect(cleaned).toBe('HelloWorld!');
    expect(count).toBe(2);
  });

  it('should remove zero-width characters', () => {
    const input = 'Hello\u200BWorld\u200C\u200D!';
    const { cleaned, count } = cleanInvisibleCharacters(input);

    expect(cleaned).toBe('HelloWorld!');
    expect(count).toBe(3);
  });

  it('should remove BOM character', () => {
    const input = '\uFEFFHello World';
    const { cleaned, count } = cleanInvisibleCharacters(input);

    expect(cleaned).toBe('Hello World');
    expect(count).toBe(1);
  });

  it('should normalize special spaces to regular space', () => {
    const input = 'Hello\u00A0World\u2003!';
    const { cleaned } = cleanInvisibleCharacters(input);

    expect(cleaned).toBe('Hello World !');
  });

  it('should preserve normal Chinese text', () => {
    const input = '你好世界 Hello 🚀';
    const { cleaned, count } = cleanInvisibleCharacters(input);

    expect(cleaned).toBe('你好世界 Hello 🚀');
    expect(count).toBe(0);
  });

  it('should preserve emoji', () => {
    const input = 'Hello 🎉🔥💡 World';
    const { cleaned, count } = cleanInvisibleCharacters(input);

    expect(cleaned).toBe('Hello 🎉🔥💡 World');
    expect(count).toBe(0);
  });
});

describe('parseFrontmatterSafe', () => {
  it('should parse valid frontmatter', () => {
    const input = `---
title: Test Post
date: 2024-01-01
---

Content here.`;

    const { data, content, duplicateKeys } = parseFrontmatterSafe(input);

    expect(data.title).toBe('Test Post');
    // gray-matter parses dates as Date objects or strings depending on format
    expect(data.date).toBeDefined();
    expect(content.trim()).toBe('Content here.');
    expect(duplicateKeys).toHaveLength(0);
  });

  it('should detect duplicate keys in frontmatter', () => {
    const input = `---
title: First Title
date: 2024-01-01
title: Second Title
---

Content here.`;

    const { data, duplicateKeys } = parseFrontmatterSafe(input);

    expect(duplicateKeys).toContain('title');
    // After de-duplication, uses the last value
    expect(data.title).toBe('Second Title');
  });

  it('should handle frontmatter with arrays', () => {
    const input = `---
title: Test
tags:
  - tag1
  - tag2
  - tag3
---

Content.`;

    const { data } = parseFrontmatterSafe(input);

    expect(data.tags).toEqual(['tag1', 'tag2', 'tag3']);
  });

  it('should clean invisible chars from frontmatter', () => {
    const input = `---
title: Test\u200BPost
date: 2024-01-01
---

Content here.`;

    const { data } = parseFrontmatterSafe(input);

    expect(data.title).toBe('TestPost');
  });
});

describe('mergeFrontmatter', () => {
  it('should merge source fields over existing fields', () => {
    const source = { title: 'New Title', date: '2024-02-01' };
    const existing = { title: 'Old Title', date: '2024-01-01', author: 'John' };

    const merged = mergeFrontmatter(source, existing);

    expect(merged.title).toBe('New Title');
    expect(merged.date).toBe('2024-02-01');
    expect(merged.author).toBe('John');
  });

  it('should preserve existing preserve fields not in source', () => {
    const source = { title: 'New Title' };
    const existing = { title: 'Old Title', description: 'Existing description' };

    const merged = mergeFrontmatter(source, existing);

    expect(merged.title).toBe('New Title');
    expect(merged.description).toBe('Existing description');
  });

  it('should handle empty existing data', () => {
    const source = { title: 'New Title', tags: ['a', 'b'] };
    const existing = {};

    const merged = mergeFrontmatter(source, existing);

    expect(merged.title).toBe('New Title');
    expect(merged.tags).toEqual(['a', 'b']);
  });

  it('should handle custom source and preserve fields', () => {
    const source = { customField: 'value1', anotherField: 'value2' };
    const existing = { customField: 'old', preservedField: 'keep' };

    const merged = mergeFrontmatter(source, existing, {
      sourceFields: ['customField'],
      preserveFields: ['preservedField'],
    });

    expect(merged.customField).toBe('value1');
    expect(merged.preservedField).toBe('keep');
  });
});

describe('processMarkdown', () => {
  describe('basic processing', () => {
    it('should process simple markdown', async () => {
      const input = `# Hello World

This is a paragraph.

- Item 1
- Item 2
`;

      const result = await processMarkdown(input);

      expect(result.markdown).toContain('# Hello World');
      expect(result.markdown).toContain('This is a paragraph.');
      expect(result.markdown).toContain('- Item 1');
      expect(result.markdown).toContain('- Item 2');
    });

    it('should preserve frontmatter', async () => {
      const input = `---
title: Test Post
date: 2024-01-01
tags:
  - test
  - example
---

# Hello World
`;

      const result = await processMarkdown(input);

      expect(result.frontmatter.title).toBe('Test Post');
      expect(result.frontmatter.tags).toEqual(['test', 'example']);
      expect(result.markdown).toContain('title: Test Post');
    });

    it('should ensure trailing newline', async () => {
      const input = '# Hello World';

      const result = await processMarkdown(input);

      expect(result.markdown.endsWith('\n')).toBe(true);
    });

    it('should compress multiple empty lines', async () => {
      const input = `# Title



Too many blank lines.`;

      const result = await processMarkdown(input);

      // Should not have 3+ consecutive newlines
      expect(result.markdown).not.toMatch(/\n{3,}/);
    });
  });

  describe('invisible character removal', () => {
    it('should remove invisible characters from content', async () => {
      const input = `# Hello\u200BWorld

This has\u202A invisible\u202E chars.`;

      const result = await processMarkdown(input);

      expect(result.markdown).not.toContain('\u200B');
      expect(result.markdown).not.toContain('\u202A');
      expect(result.markdown).not.toContain('\u202E');
      expect(result.diagnostics.invisibleCharsRemoved).toBeGreaterThan(0);
    });

    it('should remove invisible characters from frontmatter', async () => {
      const input = `---
title: Hello\u200BWorld
---

Content.`;

      const result = await processMarkdown(input);

      expect(result.frontmatter.title).toBe('HelloWorld');
    });
  });

  describe('code fence handling', () => {
    it('should preserve code blocks', async () => {
      const input = `
\`\`\`python
def hello():
    print("Hello World")
\`\`\`
`;

      const result = await processMarkdown(input);

      expect(result.markdown).toContain('```python');
      expect(result.markdown).toContain('def hello():');
      expect(result.markdown).toContain('print("Hello World")');
    });

    it('should handle code blocks with nested backticks', async () => {
      const input = `
\`\`\`\`markdown
Here is some code:

\`\`\`python
print("nested")
\`\`\`

End.
\`\`\`\`
`;

      const result = await processMarkdown(input);

      // Should be parseable
      const parsed = matter(result.markdown);
      expect(parsed.content).toBeTruthy();
    });

    it('should convert indented code blocks to fenced', async () => {
      const input = `---
title: Test
---

# Code Example

    indented code block
    line 2
    line 3
`;

      const result = await processMarkdown(input);

      // remark-stringify with fences:true converts to fenced
      expect(result.markdown).toContain('```');
    });
  });

  describe('image URL normalization', () => {
    it('should handle image URLs with spaces using angle brackets', async () => {
      const input = `
![Alt text](https://example.com/path with spaces/image.png)
`;

      const result = await processMarkdown(input);

      // remark-stringify wraps URLs with spaces in angle brackets, which is valid markdown
      // Either encoding or angle brackets is acceptable
      const hasEncodedUrl = result.markdown.includes('%20');
      const hasAngleBrackets =
        result.markdown.includes('<https://') && result.markdown.includes('>');
      expect(hasEncodedUrl || hasAngleBrackets).toBe(true);
    });

    it('should not double-encode already encoded URLs', async () => {
      const input = `
![Alt text](https://example.com/path%20with%20spaces/image.png)
`;

      const result = await processMarkdown(input);

      expect(result.markdown).toContain('%20');
      expect(result.markdown).not.toContain('%2520'); // Double encoding
    });

    it('should handle link URLs with spaces', async () => {
      const input = `
[Link text](https://example.com/path with spaces/page)
`;

      const result = await processMarkdown(input);

      // remark-stringify wraps URLs with spaces in angle brackets
      const hasEncodedUrl = result.markdown.includes('%20');
      const hasAngleBrackets =
        result.markdown.includes('<https://') && result.markdown.includes('>');
      expect(hasEncodedUrl || hasAngleBrackets).toBe(true);
    });
  });

  describe('parsability', () => {
    it('should produce reparseable markdown', async () => {
      const input = `---
title: Complex Post
tags:
  - markdown
  - test
---

# Heading 1

Some paragraph with **bold** and *italic* text.

## Heading 2

- List item 1
- List item 2
  - Nested item
  - Another nested

\`\`\`javascript
const x = 1;
console.log(x);
\`\`\`

> Blockquote here

| Column 1 | Column 2 |
| -------- | -------- |
| Cell 1   | Cell 2   |
`;

      const result = await processMarkdown(input);

      // Should be parseable by gray-matter
      const parsed = matter(result.markdown);
      expect(parsed.data.title).toBe('Complex Post');
      expect(parsed.content).toBeTruthy();

      // Should be parseable by remark
      const processor = unified().use(remarkParse);
      const tree = processor.parse(parsed.content);
      expect(tree.type).toBe('root');
      expect(tree.children.length).toBeGreaterThan(0);
    });

    it('should handle edge case: markdown with only frontmatter', async () => {
      const input = `---
title: Empty Post
---
`;

      const result = await processMarkdown(input);

      expect(result.frontmatter.title).toBe('Empty Post');
      expect(result.markdown).toContain('title: Empty Post');
    });

    it('should handle edge case: empty input', async () => {
      const input = '';

      const result = await processMarkdown(input);

      expect(result.markdown).toBe('\n');
      expect(Object.keys(result.frontmatter)).toHaveLength(0);
    });
  });

  describe('duplicate frontmatter keys', () => {
    it('should detect and report duplicate keys', async () => {
      const input = `---
title: First
date: 2024-01-01
title: Second
---

Content.`;

      const result = await processMarkdown(input);

      expect(result.diagnostics.duplicateFrontmatterKeysRemoved).toBe(1);
      // Output should only have one title
      expect(result.markdown.match(/title:/g)?.length).toBe(1);
    });

    it('should produce valid output even with duplicate keys', async () => {
      const input = `---
title: First
tags:
  - a
title: Second
tags:
  - b
---

Content.`;

      const result = await processMarkdown(input);

      // Should be parseable
      const parsed = matter(result.markdown);
      expect(parsed.data.title).toBeDefined();
      expect(parsed.data.tags).toBeDefined();
    });
  });
});

describe('processMarkdownForNotionSync', () => {
  it('should merge frontmatter from new and existing content', async () => {
    const rawMarkdown = `# New Content

This is the new article body.`;

    const newFrontmatter = {
      title: 'Updated Title',
      date: '2024-02-01',
      source: 'notion',
    };

    const existingContent = `---
title: Old Title
date: 2024-01-01
description: Existing description
author: John Doe
---

# Old Content

This is the old body.`;

    const result = await processMarkdownForNotionSync(rawMarkdown, newFrontmatter, existingContent);

    expect(result.frontmatter.title).toBe('Updated Title');
    expect(result.frontmatter.date).toBe('2024-02-01');
    expect(result.frontmatter.source).toBe('notion');
    // Preserved field
    expect(result.frontmatter.description).toBe('Existing description');
  });

  it('should work without existing content', async () => {
    const rawMarkdown = `# New Post

Content here.`;

    const newFrontmatter = {
      title: 'New Post',
      date: '2024-02-01',
      tags: ['new'],
    };

    const result = await processMarkdownForNotionSync(rawMarkdown, newFrontmatter);

    expect(result.frontmatter.title).toBe('New Post');
    expect(result.frontmatter.tags).toEqual(['new']);
  });

  it('should clean invisible characters from raw markdown', async () => {
    const rawMarkdown = `# Hello\u200BWorld

Content with\u202A invisible\u202E chars.`;

    const newFrontmatter = { title: 'Test' };

    const result = await processMarkdownForNotionSync(rawMarkdown, newFrontmatter);

    expect(result.markdown).not.toContain('\u200B');
    expect(result.markdown).not.toContain('\u202A');
    expect(result.diagnostics.invisibleCharsRemoved).toBeGreaterThan(0);
  });
});

describe('Notion content regression tests', () => {
  describe('code blocks', () => {
    it('should handle Python code block', async () => {
      const input = `---
title: Python Example
---

\`\`\`python
import numpy as np

def calculate(x):
    return np.sum(x ** 2)
\`\`\`
`;

      const result = await processMarkdown(input);

      expect(result.markdown).toContain('```python');
      expect(result.markdown).toContain('import numpy as np');
      expect(result.markdown).toContain('def calculate(x):');
    });

    it('should handle code block with special characters', async () => {
      const input = `---
title: Special Chars
---

\`\`\`bash
echo "Hello $USER"
if [[ $? -eq 0 ]]; then
    echo "Success"
fi
\`\`\`
`;

      const result = await processMarkdown(input);

      expect(result.markdown).toContain('echo "Hello $USER"');
      expect(result.markdown).toContain('[[ $? -eq 0 ]]');
    });
  });

  describe('images', () => {
    it('should handle image with spaces in URL', async () => {
      const input = `---
title: Image Test
---

![Screenshot of the app](https://example.com/images/my screenshot.png)

Some text after.
`;

      const result = await processMarkdown(input);

      // remark-stringify uses angle brackets for URLs with spaces
      const hasEncodedUrl = result.markdown.includes('%20');
      const hasAngleBrackets = result.markdown.includes('<https://');
      expect(hasEncodedUrl || hasAngleBrackets).toBe(true);
    });

    it('should preserve image alt text', async () => {
      const input = `---
title: Image Test
---

![This is a detailed description of the image](https://example.com/image.png)
`;

      const result = await processMarkdown(input);

      expect(result.markdown).toContain('![This is a detailed description of the image]');
    });
  });

  describe('tables', () => {
    it('should handle GFM tables', async () => {
      const input = `---
title: Table Test
---

| Feature | Support |
| ------- | ------- |
| Tables  | Yes     |
| Lists   | Yes     |
`;

      const result = await processMarkdown(input);

      expect(result.markdown).toContain('| Feature | Support |');
      expect(result.markdown).toContain('| Tables');
    });

    it('should handle tables with alignment', async () => {
      const input = `---
title: Aligned Table
---

| Left | Center | Right |
| :--- | :----: | ----: |
| L    |   C    |     R |
`;

      const result = await processMarkdown(input);

      // Table alignment markers should be preserved
      // remark-gfm may normalize the number of dashes
      expect(result.markdown).toContain(':--'); // left align
      expect(result.markdown).toMatch(/:--+:/); // center align (one or more dashes)
      expect(result.markdown).toMatch(/--+:/); // right align
    });
  });

  describe('nested lists', () => {
    it('should handle nested unordered lists', async () => {
      const input = `---
title: List Test
---

- Level 1 item
  - Level 2 item
    - Level 3 item
  - Another level 2
- Back to level 1
`;

      const result = await processMarkdown(input);

      expect(result.markdown).toContain('- Level 1 item');
      // Nested items should be indented
      expect(result.markdown).toMatch(/\s{2,}- Level 2 item/);
    });

    it('should handle mixed ordered and unordered lists', async () => {
      const input = `---
title: Mixed List
---

1. First ordered
   - Unordered sub
   - Another sub
2. Second ordered
   1. Ordered sub
   2. Another ordered sub
`;

      const result = await processMarkdown(input);

      expect(result.markdown).toContain('1.');
      expect(result.markdown).toContain('-');
    });

    it('should handle task lists', async () => {
      const input = `---
title: Task List
---

- [ ] Unchecked task
- [x] Checked task
- [ ] Another unchecked
`;

      const result = await processMarkdown(input);

      expect(result.markdown).toContain('- [ ]');
      expect(result.markdown).toContain('- [x]');
    });
  });
});
