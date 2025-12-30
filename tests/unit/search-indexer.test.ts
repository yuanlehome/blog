import { describe, it, expect } from 'vitest';
import {
  removeCodeBlocks,
  removeFrontmatter,
  removeHtmlTags,
  removeMarkdownFormatting,
  extractHeadings,
  markdownToPlainText,
  truncateText,
  extractSourceType,
  createSearchEntry,
  calculateTagCounts,
  createSearchIndex,
} from '../../src/lib/search/indexer';
import type { SearchIndexEntry } from '../../src/lib/search/types';

describe('search indexer', () => {
  describe('removeCodeBlocks', () => {
    it('removes fenced code blocks with backticks', () => {
      const input = 'before\n```js\nconst x = 1;\n```\nafter';
      expect(removeCodeBlocks(input)).toBe('before\n\nafter');
    });

    it('removes fenced code blocks with tildes', () => {
      const input = 'before\n~~~python\nprint("hi")\n~~~\nafter';
      expect(removeCodeBlocks(input)).toBe('before\n\nafter');
    });

    it('handles multiple code blocks', () => {
      const input = 'a\n```\ncode1\n```\nb\n```\ncode2\n```\nc';
      expect(removeCodeBlocks(input)).toBe('a\n\nb\n\nc');
    });

    it('preserves text without code blocks', () => {
      const input = 'no code here';
      expect(removeCodeBlocks(input)).toBe('no code here');
    });
  });

  describe('removeFrontmatter', () => {
    it('removes YAML frontmatter', () => {
      const input = '---\ntitle: Test\ndate: 2024-01-01\n---\nContent here';
      expect(removeFrontmatter(input)).toBe('Content here');
    });

    it('preserves content without frontmatter', () => {
      const input = 'Just content';
      expect(removeFrontmatter(input)).toBe('Just content');
    });
  });

  describe('removeHtmlTags', () => {
    it('removes HTML tags', () => {
      expect(removeHtmlTags('<p>Hello</p>')).toBe('Hello');
      expect(removeHtmlTags('<a href="url">Link</a>')).toBe('Link');
    });

    it('removes self-closing tags', () => {
      expect(removeHtmlTags('before<br/>after')).toBe('beforeafter');
    });

    it('handles nested/malformed tags', () => {
      // Handles nested cases through iteration
      expect(removeHtmlTags('<scr<b>ipt>')).toBe('ipt>');
    });
  });

  describe('removeMarkdownFormatting', () => {
    it('removes inline code', () => {
      expect(removeMarkdownFormatting('use `code` here')).toBe('use  here');
    });

    it('removes images', () => {
      expect(removeMarkdownFormatting('![alt](url)')).toBe('');
    });

    it('removes links but keeps text', () => {
      expect(removeMarkdownFormatting('[click here](url)')).toBe('click here');
    });

    it('removes bold formatting', () => {
      expect(removeMarkdownFormatting('**bold** text')).toBe('bold text');
      expect(removeMarkdownFormatting('__bold__ text')).toBe('bold text');
    });

    it('removes italic formatting', () => {
      expect(removeMarkdownFormatting('*italic* text')).toBe('italic text');
      expect(removeMarkdownFormatting('_italic_ text')).toBe('italic text');
    });

    it('removes strikethrough', () => {
      expect(removeMarkdownFormatting('~~deleted~~ text')).toBe('deleted text');
    });

    it('removes blockquotes', () => {
      expect(removeMarkdownFormatting('> quoted text')).toBe('quoted text');
    });
  });

  describe('extractHeadings', () => {
    it('extracts headings at different levels', () => {
      const input = '# H1\n## H2\n### H3\nContent\n#### H4';
      expect(extractHeadings(input)).toEqual(['H1', 'H2', 'H3', 'H4']);
    });

    it('returns empty array for no headings', () => {
      expect(extractHeadings('No headings here')).toEqual([]);
    });

    it('handles headings with special characters', () => {
      const input = '## Hello, World!\n### Test & Demo';
      expect(extractHeadings(input)).toEqual(['Hello, World!', 'Test & Demo']);
    });
  });

  describe('markdownToPlainText', () => {
    it('converts markdown to plain text', () => {
      const input = `---
title: Test
---

# Heading

This is **bold** and *italic* text.

\`\`\`js
const x = 1;
\`\`\`

More text here.`;
      const result = markdownToPlainText(input);
      expect(result).not.toContain('```');
      expect(result).not.toContain('---');
      expect(result).not.toContain('**');
      expect(result).toContain('bold');
      expect(result).toContain('More text here');
    });

    it('normalizes whitespace', () => {
      const input = 'line1\n\n\nline2\n\nline3';
      const result = markdownToPlainText(input);
      expect(result).toBe('line1 line2 line3');
    });
  });

  describe('truncateText', () => {
    it('preserves text shorter than max', () => {
      expect(truncateText('short', 100)).toBe('short');
    });

    it('truncates at word boundary', () => {
      const input = 'hello world this is long text';
      const result = truncateText(input, 18);
      // The function tries to find a word boundary; "hello world this" is 16 chars
      expect(result).toBe('hello world this...');
    });

    it('adds ellipsis when truncated', () => {
      const result = truncateText('a'.repeat(100), 50);
      expect(result.endsWith('...')).toBe(true);
    });
  });

  describe('extractSourceType', () => {
    it('identifies notion source', () => {
      expect(extractSourceType('/content/blog/notion/post.md')).toBe('notion');
    });

    it('identifies wechat source', () => {
      expect(extractSourceType('/content/blog/wechat/post.md')).toBe('wechat');
    });

    it('identifies others source', () => {
      expect(extractSourceType('/content/blog/others/post.md')).toBe('others');
    });

    it('returns undefined for unknown source', () => {
      expect(extractSourceType('/content/blog/post.md')).toBeUndefined();
    });
  });

  describe('createSearchEntry', () => {
    it('creates a complete search entry', () => {
      const entry = createSearchEntry(
        'test-post',
        '/test-post/',
        'Test Post Title',
        '# Heading\n\nSome content here.',
        ['tag1', 'tag2'],
        new Date('2024-01-01'),
        'Summary text',
        'notion',
      );

      expect(entry.slug).toBe('test-post');
      expect(entry.url).toBe('/test-post/');
      expect(entry.title).toBe('Test Post Title');
      expect(entry.tags).toEqual(['tag1', 'tag2']);
      expect(entry.headings).toContain('Heading');
      expect(entry.source).toBe('notion');
    });

    it('generates summary from body if not provided', () => {
      const entry = createSearchEntry(
        'test',
        '/test/',
        'Title',
        'This is the body content for the post.',
        [],
        new Date(),
      );

      expect(entry.summary).toBeTruthy();
      expect(entry.summary).toContain('body content');
    });

    it('removes code blocks from body', () => {
      const entry = createSearchEntry(
        'test',
        '/test/',
        'Title',
        '```js\nconst x = 1;\n```\nPlain text',
        [],
        new Date(),
      );

      expect(entry.body).not.toContain('const x');
      expect(entry.body).toContain('Plain text');
    });
  });

  describe('calculateTagCounts', () => {
    it('counts tags across entries', () => {
      const entries: SearchIndexEntry[] = [
        {
          slug: '1',
          url: '/1/',
          title: 'A',
          headings: [],
          tags: ['js', 'ts'],
          date: '',
          summary: '',
          body: '',
        },
        {
          slug: '2',
          url: '/2/',
          title: 'B',
          headings: [],
          tags: ['js'],
          date: '',
          summary: '',
          body: '',
        },
        {
          slug: '3',
          url: '/3/',
          title: 'C',
          headings: [],
          tags: ['python'],
          date: '',
          summary: '',
          body: '',
        },
      ];

      const counts = calculateTagCounts(entries);
      expect(counts.js).toBe(2);
      expect(counts.ts).toBe(1);
      expect(counts.python).toBe(1);
    });

    it('returns empty object for no entries', () => {
      expect(calculateTagCounts([])).toEqual({});
    });
  });

  describe('createSearchIndex', () => {
    it('creates a valid search index', () => {
      const entries: SearchIndexEntry[] = [
        {
          slug: '1',
          url: '/1/',
          title: 'A',
          headings: [],
          tags: ['js'],
          date: '2024-01-01',
          summary: '',
          body: '',
        },
      ];

      const index = createSearchIndex(entries);

      expect(index.version).toBe(1);
      expect(index.count).toBe(1);
      expect(index.entries).toEqual(entries);
      expect(index.tags).toEqual({ js: 1 });
      expect(index.generatedAt).toBeTruthy();
    });
  });
});
