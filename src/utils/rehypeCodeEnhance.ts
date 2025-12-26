import type { Plugin } from 'unified';
import { visit } from 'unist-util-visit';
import type { Element } from 'hast';

type ClassName = string | string[] | undefined;

const toClassList = (className: ClassName): string[] => {
  if (!className) return [];
  if (Array.isArray(className)) return className.filter(Boolean).map(String);
  return className
    .split(/\s+/)
    .map((cls) => cls.trim())
    .filter(Boolean);
};

const findLanguage = (pre: Element, code: Element): string => {
  const classLanguages = [...toClassList(pre.properties?.className), ...toClassList(code.properties?.className)]
    .map((cls) => (cls.startsWith('language-') ? cls.replace('language-', '') : undefined))
    .filter(Boolean) as string[];

  const dataLanguages = [
    pre.properties?.dataLanguage as string | undefined,
    pre.properties?.['data-language'] as string | undefined,
    code.properties?.dataLanguage as string | undefined,
    code.properties?.['data-language'] as string | undefined,
  ].filter(Boolean) as string[];

  const language = [...dataLanguages, ...classLanguages].find(Boolean);
  return language && language.trim() ? language.trim() : 'text';
};

const createToolbar = (language: string): Element => ({
  type: 'element',
  tagName: 'div',
  properties: {
    className: ['code-toolbar'],
  },
  children: [
    {
      type: 'element',
      tagName: 'span',
      properties: {
        className: ['code-lang'],
      },
      children: [{ type: 'text', value: language }],
    },
  ],
});

const rehypeCodeEnhance: Plugin = () => {
  return (tree) => {
    visit(tree, 'element', (node: Element, index, parent) => {
      if (!parent || node.tagName !== 'pre') return;
      const code = node.children.find(
        (child): child is Element => child.type === 'element' && child.tagName === 'code'
      );
      if (!code) return;

      if (parent.type === 'element' && toClassList((parent as Element).properties?.className).includes('code-block')) {
        return;
      }

      const language = findLanguage(node, code);

      const preClassList = new Set(toClassList(node.properties?.className));
      preClassList.delete('code-block');

      node.properties = {
        ...node.properties,
        className: Array.from(preClassList),
        dataLanguage: language,
        'data-language': language,
      };

      code.properties = {
        ...code.properties,
        dataLanguage: language,
        'data-language': language,
      };

      const wrapper: Element = {
        type: 'element',
        tagName: 'div',
        properties: {
          className: ['code-block'],
          dataLanguage: language,
          'data-language': language,
        },
        children: [createToolbar(language), node],
      };

      parent.children.splice(index!, 1, wrapper);
    });
  };
};

export default rehypeCodeEnhance;
