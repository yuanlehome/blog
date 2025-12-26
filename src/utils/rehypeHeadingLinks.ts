import type { Plugin } from 'unified';
import { visit } from 'unist-util-visit';
import type { Element } from 'hast';
import slugify from 'slugify';

const rehypeHeadingLinks: Plugin = () => {
  return (tree) => {
    visit(tree, 'element', (node: Element) => {
      if (!/^h[1-6]$/.test(node.tagName)) return;
      const text = extractText(node).trim();
      if (!text) return;
      const id = slugify(text, { lower: true, strict: true });
      node.properties = { ...(node.properties || {}), id };

      const anchor: Element = {
        type: 'element',
        tagName: 'a',
        properties: {
          href: `#${id}`,
          className: ['heading-anchor'],
          ariaLabel: `Link to ${text}`,
        },
        children: [{ type: 'text', value: '¶' }],
      };

      node.children = [...(node.children || []), anchor];
    });
  };
};

function extractText(node: Element): string {
  if (!node.children) return '';
  return node.children
    .map((child) => {
      if (child.type === 'text') return child.value;
      if ((child as Element).children) return extractText(child as Element);
      return '';
    })
    .join(' ');
}

export default rehypeHeadingLinks;
