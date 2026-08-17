import type { Element, Root, Text } from 'hast';
import type { Plugin } from 'unified';
import { visit } from 'unist-util-visit';

const rehypeLocalizedFootnotes: Plugin<[], Root> = () => (tree) => {
  visit(tree, 'element', (node: Element) => {
    if (node.tagName !== 'h2' || node.properties?.id !== 'footnote-label') return;

    const label = node.children.find((child): child is Text => child.type === 'text');
    if (label) label.value = '脚注';
  });
};

export default rehypeLocalizedFootnotes;
