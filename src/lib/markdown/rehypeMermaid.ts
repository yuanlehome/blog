import type { Element, Root } from 'hast';
import type { Plugin } from 'unified';
import { visit } from 'unist-util-visit';

const getClassList = (node: Element): string[] => {
  const className = node.properties?.className;
  if (Array.isArray(className)) {
    return className.filter((item): item is string => typeof item === 'string');
  }
  if (typeof className === 'string') {
    return className.split(/\s+/).filter(Boolean);
  }
  return [];
};

const encodeSource = (value: string): string => Buffer.from(value, 'utf-8').toString('base64');

const rehypeMermaid: Plugin<[], Root> = () => {
  return (tree) => {
    let mermaidIndex = 0;

    visit(tree, 'element', (node: Element, index, parent) => {
      if (!parent || typeof index !== 'number' || node.tagName !== 'pre') return;

      const codeNode = node.children.find(
        (child): child is Element => child.type === 'element' && child.tagName === 'code',
      );
      if (!codeNode) return;

      const classNames = getClassList(codeNode);
      const dataLang = (codeNode.properties as Record<string, unknown> | undefined)?.dataLang;
      const isMermaid =
        classNames.includes('language-mermaid') ||
        (typeof dataLang === 'string' && dataLang.toLowerCase() === 'mermaid');

      if (!isMermaid) return;

      const rawCode =
        (codeNode.properties as Record<string, unknown> | undefined)?.dataRawCode ||
        codeNode.children.map((child) => ('value' in child ? child.value : '')).join('');
      const code = typeof rawCode === 'string' ? rawCode : '';
      const encoded = encodeSource(code);
      const idx = String(mermaidIndex);
      mermaidIndex += 1;

      const mermaidFigure: Element = {
        type: 'element',
        tagName: 'figure',
        properties: {
          className: ['mermaid-block', 'not-prose'],
          dataMermaid: encoded,
          'data-mermaid': encoded,
          dataIdx: idx,
          'data-idx': idx,
        },
        children: [
          {
            type: 'element',
            tagName: 'div',
            properties: { className: ['mermaid-loading'] },
            children: [{ type: 'text', value: 'Rendering diagram…' }],
          },
          {
            type: 'element',
            tagName: 'pre',
            properties: { className: ['mermaid-source'], hidden: true },
            children: [{ type: 'text', value: code }],
          },
          {
            type: 'element',
            tagName: 'div',
            properties: { className: ['mermaid-render'] },
            children: [],
          },
          {
            type: 'element',
            tagName: 'div',
            properties: { className: ['mermaid-error'], hidden: true },
            children: [],
          },
        ],
      };

      parent.children.splice(index, 1, mermaidFigure);
    });
  };
};

export default rehypeMermaid;
