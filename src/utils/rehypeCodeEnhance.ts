import type { Plugin } from 'unified';
import { visit } from 'unist-util-visit';
import type { Element } from 'hast';

const rehypeCodeEnhance: Plugin = () => {
  return (tree) => {
    visit(tree, 'element', (node: Element) => {
      if (node.tagName !== 'pre') return;
      const code = node.children.find(
        (child): child is Element => child.type === 'element' && child.tagName === 'code'
      );
      if (!code) return;

      const classList = (code.properties?.className as string[] | undefined) || [];
      const langClass = classList.find((cls) => cls.startsWith('language-')) || 'language-text';
      const language = langClass.replace('language-', '') || 'text';

      node.properties = {
        ...node.properties,
        className: ['code-block', ...(node.properties?.className as string[] | undefined || [])],
      };

      const copyButton: Element = {
        type: 'element',
        tagName: 'button',
        properties: {
          className: ['code-copy'],
          type: 'button',
          'data-lang': language,
        },
        children: [{ type: 'text', value: 'Copy' }],
      };

      const label: Element = {
        type: 'element',
        tagName: 'span',
        properties: {
          className: ['code-lang'],
        },
        children: [{ type: 'text', value: language }],
      };

      const toolbar: Element = {
        type: 'element',
        tagName: 'div',
        properties: {
          className: ['code-toolbar'],
        },
        children: [label, copyButton],
      };

      node.children = [toolbar, ...node.children];
    });
  };
};

export default rehypeCodeEnhance;
