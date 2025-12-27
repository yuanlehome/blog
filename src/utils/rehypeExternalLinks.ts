import type { Plugin } from 'unified';
import { visit } from 'unist-util-visit';
import type { Element } from 'hast';

interface Options {
  target?: string;
  rel?: string[];
}

const rehypeExternalLinks: Plugin<[Options?]> = (options = {}) => {
  const target = options.target ?? '_blank';
  const rel = options.rel ?? ['noopener', 'noreferrer'];

  return (tree) => {
    visit(tree, 'element', (node: Element) => {
      if (node.tagName !== 'a') return;
      const href = (node.properties?.href as string) || '';
      if (
        !href ||
        href.startsWith('#') ||
        href.startsWith('/') ||
        href.startsWith(import.meta.env.BASE_URL)
      ) {
        return;
      }
      node.properties = {
        ...node.properties,
        target,
        rel: rel.join(' '),
      };
    });
  };
};

export default rehypeExternalLinks;
