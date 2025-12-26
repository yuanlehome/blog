import type { Plugin } from 'unified';
import { visit } from 'unist-util-visit';
import type { Root } from 'mdast';

const invisibleReplacements: Record<string, string> = {
  '\u00a0': ' ',
  '\u2000': ' ',
  '\u2001': ' ',
  '\u2002': ' ',
  '\u2003': ' ',
  '\u2004': ' ',
  '\u2005': ' ',
  '\u2006': ' ',
  '\u2007': ' ',
  '\u2008': ' ',
  '\u2009': ' ',
  '\u200a': ' ',
  '\u202f': ' ',
  '\u3000': ' ',
  '\u200b': '',
  '\u200c': '',
  '\u200d': '',
  '\ufeff': '',
};

const notionCalloutEmoji = /^(💡|⚠️|📌|❗|✅|ℹ️|📣|📝)\s+/;

const remarkNotionCompat: Plugin<[], Root> = () => {
  return (tree) => {
    // Normalize invisible characters and stray escapes
    visit(tree, 'text', (node) => {
      Object.entries(invisibleReplacements).forEach(([char, replacement]) => {
        if (node.value.includes(char)) {
          node.value = node.value.split(char).join(replacement);
        }
      });
      // Notion sometimes escapes underscores in math/table content
      node.value = node.value.replace(/\\_/g, '_');
    });

    // Ensure code blocks always have a language for Shiki
    visit(tree, 'code', (node) => {
      if (!node.lang || node.lang.trim() === '') {
        node.lang = 'text';
      }
      // Trim extra blank lines that sneak in from Notion export
      node.value = node.value.replace(/\n{3,}/g, '\n\n').trimEnd();
    });

    // Collapse excessive blank paragraphs from Notion
    if (Array.isArray(tree.children)) {
      const compacted: Root['children'] = [];
      let blankStreak = 0;
      for (const child of tree.children) {
        const isBlankParagraph =
          child.type === 'paragraph' &&
          child.children?.every((c: any) => c.type === 'text' && c.value.trim() === '');
        if (isBlankParagraph) {
          blankStreak += 1;
          if (blankStreak > 1) continue;
        } else {
          blankStreak = 0;
        }
        compacted.push(child);
      }
      tree.children = compacted;
    }

    // Degrade callout/toggle to blockquote with a label so it renders consistently
    visit(tree, 'blockquote', (node) => {
      const first = node.children?.[0];
      if (first && first.type === 'paragraph') {
        const textNode = first.children?.find((c: any) => c.type === 'text');
        if (textNode && typeof textNode.value === 'string') {
          textNode.value = textNode.value.replace(notionCalloutEmoji, '').trimStart();
        }
      }
    });

    // Tables from Notion sometimes lose alignment; ensure align array exists
    visit(tree, 'table', (node: any) => {
      if (!node.align) node.align = [];
    });
  };
};

export default remarkNotionCompat;
