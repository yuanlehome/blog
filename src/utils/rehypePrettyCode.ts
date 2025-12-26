import type { Plugin } from 'unified';
import { visit } from 'unist-util-visit';
import { codeToHast, bundledLanguages, bundledThemes } from 'shiki';
import type { Element, Root, Text } from 'hast';

const lightTheme = bundledThemes['github-light'];
const darkTheme = bundledThemes['github-dark'];

const aliases: Record<string, string> = {
  js: 'javascript',
  ts: 'typescript',
  tsx: 'tsx',
  jsx: 'jsx',
  py: 'python',
  sh: 'bash',
  shell: 'bash',
  md: 'markdown',
  yml: 'yaml',
  cplusplus: 'cpp',
};

const normalizeLanguage = (lang?: string): string => {
  if (!lang) return 'text';
  const normalized = lang.toLowerCase();
  const mapped = aliases[normalized] || normalized;
  return bundledLanguages[mapped as keyof typeof bundledLanguages] ? mapped : 'text';
};

const parseHighlightSet = (value?: string): Set<number> => {
  if (!value) return new Set();
  return new Set(
    value
      .split(',')
      .map((v) => Number.parseInt(v.trim(), 10))
      .filter((v) => Number.isFinite(v) && v > 0)
  );
};

const ensureLineText = (line: Element) => {
  const hasTextChild = line.children.some((child) => child.type === 'text' || child.type === 'element');
  if (!hasTextChild) {
    line.children = [{ type: 'text', value: ' ' } as Text];
  }
};

const setClassName = (element: Element, className: string) => {
  const current = element.properties?.className;
  if (Array.isArray(current)) {
    if (!current.includes(className)) current.push(className);
    element.properties!.className = current;
    return;
  }

  const cls = typeof current === 'string' ? current.split(/\s+/).filter(Boolean) : [];
  if (!cls.includes(className)) cls.push(className);
  element.properties = { ...element.properties, className: cls };
};

const clonePreFromHast = async (code: string, lang: string, theme: 'light' | 'dark'): Promise<Element> => {
  const hast = await codeToHast(code, {
    lang,
    theme: theme === 'light' ? lightTheme : darkTheme,
  });
  const pre = (hast.children.find((child): child is Element => child.type === 'element' && child.tagName === 'pre') ||
    ({ type: 'element', tagName: 'pre', properties: {}, children: [] } as Element));
  pre.properties = {
    ...pre.properties,
    className: ['code-block__pre'],
    dataTheme: theme,
    'data-theme': theme,
    dataLanguage: lang,
    'data-language': lang,
    tabIndex: 0,
  };

  // Remove shiki inline background; styling handled in CSS
  if (pre.properties && 'style' in pre.properties) {
    delete (pre.properties as any).style;
  }

  const codeNode = pre.children.find((child): child is Element => child.type === 'element' && child.tagName === 'code');
  if (codeNode) {
    codeNode.properties = {
      ...codeNode.properties,
      dataLanguage: lang,
      'data-language': lang,
    };
  }

  return pre;
};

const decorateLines = (pre: Element, highlights: Set<number>, showLineNumbers: boolean) => {
  let lineNumber = 1;
  visit(pre, 'element', (node: Element) => {
    const classes = node.properties?.className || node.properties?.class;
    const classList = Array.isArray(classes)
      ? classes
      : typeof classes === 'string'
        ? classes.split(/\s+/)
        : [];

    if (node.tagName === 'span' && classList.includes('line')) {
      ensureLineText(node);
      if (showLineNumbers) {
        (node.properties ||= {});
        (node.properties as any)['data-line-number'] = lineNumber;
      }
      if (highlights.has(lineNumber)) {
        setClassName(node, 'line--highlighted');
      }
      lineNumber += 1;
    }
  });
};

const rehypePrettyCode: Plugin = () => {
  return async (tree: Root) => {
    const transforms: Promise<void>[] = [];

    visit(tree, 'element', (node: Element, index, parent) => {
      if (!parent || node.tagName !== 'pre') return;
      const codeNode = node.children.find((child): child is Element => child.type === 'element' && child.tagName === 'code');
      if (!codeNode) return;

      const raw = (codeNode.properties as any)?.dataRawCode || codeNode.children.map((child: any) => child.value || '').join('');
      const langFromData = (codeNode.properties as any)?.dataLang as string | undefined;
      const langFromClass = Array.isArray(codeNode.properties?.className)
        ? (codeNode.properties?.className as string[]).find((cls) => cls.startsWith('language-'))?.replace('language-', '')
        : undefined;
      const lang = normalizeLanguage(langFromData || langFromClass || 'text');
      const highlightLines = parseHighlightSet((codeNode.properties as any)?.dataHighlightLines as string | undefined);
      const showLineNumbers = (codeNode.properties as any)?.dataLineNumbers !== 'false';
      const title = (codeNode.properties as any)?.dataTitle as string | undefined;

      transforms.push(
        (async () => {
          const lightPre = await clonePreFromHast(raw, lang, 'light');
          const darkPre = await clonePreFromHast(raw, lang, 'dark');

          decorateLines(lightPre, highlightLines, showLineNumbers);
          decorateLines(darkPre, highlightLines, showLineNumbers);

          const header: Element = {
            type: 'element',
            tagName: 'div',
            properties: { className: ['code-block__header'] },
            children: [
              {
                type: 'element',
                tagName: 'div',
                properties: { className: ['code-block__title'] },
                children: [
                  {
                    type: 'text',
                    value: title || lang,
                  },
                ],
              },
              {
                type: 'element',
                tagName: 'button',
                properties: {
                  type: 'button',
                  className: ['code-block__copy'],
                  'aria-label': '复制代码',
                },
                children: [
                  { type: 'text', value: 'Copy' },
                ],
              },
            ],
          };

          const figure: Element = {
            type: 'element',
            tagName: 'figure',
            properties: {
              className: ['code-block', 'not-prose'],
              dataLanguage: lang,
              'data-language': lang,
              dataRawCode: raw,
              'data-raw-code': raw,
              dataLineNumbers: showLineNumbers ? 'true' : 'false',
              'data-line-numbers': showLineNumbers ? 'true' : 'false',
            },
            children: [header, lightPre, darkPre],
          };

          parent.children.splice(index!, 1, figure);
        })()
      );
    });

    await Promise.all(transforms);
  };
};

export default rehypePrettyCode;
