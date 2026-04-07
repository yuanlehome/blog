import prettier from 'prettier';

export interface SuspiciousUnderscoreEmphasisFinding {
  line: number;
  column: number;
  excerpt: string;
  raw: string;
  nodeType: 'emphasis' | 'strong';
}

interface MarkdownNode {
  type?: string;
  children?: MarkdownNode[];
  position?: {
    start?: { offset?: number };
    end?: { offset?: number };
  };
}

const prettierDebug = prettier as typeof prettier & {
  __debug: {
    parse(text: string, options: { parser: 'markdown' }): Promise<{ ast: MarkdownNode }>;
  };
};

function getBodyStartOffset(markdown: string): number {
  if (!markdown.startsWith('---\n') && !markdown.startsWith('---\r\n')) {
    return 0;
  }

  let lineStart = markdown.indexOf('\n') + 1;
  while (lineStart > 0 && lineStart < markdown.length) {
    const lineEnd = markdown.indexOf('\n', lineStart);
    const line = markdown
      .slice(lineStart, lineEnd === -1 ? markdown.length : lineEnd)
      .replace(/\r$/, '');

    if (line === '---' || line === '...') {
      return lineEnd === -1 ? markdown.length : lineEnd + 1;
    }

    if (lineEnd === -1) {
      return 0;
    }

    lineStart = lineEnd + 1;
  }

  return 0;
}

function buildLineStarts(text: string): number[] {
  const starts = [0];
  for (let index = 0; index < text.length; index += 1) {
    if (text[index] === '\n') {
      starts.push(index + 1);
    }
  }
  return starts;
}

function offsetToLineColumn(
  lineStarts: number[],
  offset: number,
): { line: number; column: number } {
  let low = 0;
  let high = lineStarts.length - 1;

  while (low <= high) {
    const mid = Math.floor((low + high) / 2);
    const lineStart = lineStarts[mid];
    const nextLineStart =
      mid + 1 < lineStarts.length ? lineStarts[mid + 1] : Number.POSITIVE_INFINITY;

    if (offset < lineStart) {
      high = mid - 1;
      continue;
    }

    if (offset >= nextLineStart) {
      low = mid + 1;
      continue;
    }

    return {
      line: mid + 1,
      column: offset - lineStart + 1,
    };
  }

  return {
    line: lineStarts.length,
    column: 1,
  };
}

function getLineExcerpt(markdown: string, lineStarts: number[], line: number): string {
  const start = lineStarts[line - 1] ?? 0;
  const nextStart = line < lineStarts.length ? lineStarts[line] : markdown.length;
  return markdown.slice(start, nextStart).replace(/[\r\n]+$/, '');
}

function isAsciiIdentifierChar(char: string | undefined): boolean {
  return char !== undefined && /[A-Za-z0-9]/.test(char);
}

function isUnderscoreDelimited(nodeType: 'emphasis' | 'strong', raw: string): boolean {
  if (nodeType === 'strong') {
    return raw.startsWith('__') && raw.endsWith('__');
  }

  return raw.startsWith('_') && raw.endsWith('_');
}

function normalizeRaw(raw: string): string {
  return raw.replace(/\s+/g, ' ').trim();
}

function visitNodes(node: MarkdownNode, visitor: (node: MarkdownNode) => void): void {
  visitor(node);
  for (const child of node.children ?? []) {
    visitNodes(child, visitor);
  }
}

export async function findSuspiciousUnderscoreEmphasis(
  markdown: string,
): Promise<SuspiciousUnderscoreEmphasisFinding[]> {
  const bodyStartOffset = getBodyStartOffset(markdown);
  const body = markdown.slice(bodyStartOffset);
  const { ast } = await prettierDebug.__debug.parse(body, { parser: 'markdown' });
  const lineStarts = buildLineStarts(markdown);
  const findings: SuspiciousUnderscoreEmphasisFinding[] = [];

  visitNodes(ast, (node) => {
    if (node.type !== 'emphasis' && node.type !== 'strong') {
      return;
    }

    const startOffset = node.position?.start.offset;
    const endOffset = node.position?.end.offset;
    if (startOffset === undefined || endOffset === undefined) {
      return;
    }

    const raw = body.slice(startOffset, endOffset);
    if (!isUnderscoreDelimited(node.type, raw)) {
      return;
    }

    const absoluteStart = bodyStartOffset + startOffset;
    const absoluteEnd = bodyStartOffset + endOffset;
    const previousChar = markdown[absoluteStart - 1];
    const nextChar = markdown[absoluteEnd];

    if (!isAsciiIdentifierChar(previousChar) && !isAsciiIdentifierChar(nextChar)) {
      return;
    }

    const { line, column } = offsetToLineColumn(lineStarts, absoluteStart);
    findings.push({
      line,
      column,
      excerpt: getLineExcerpt(markdown, lineStarts, line),
      raw: normalizeRaw(raw),
      nodeType: node.type,
    });
  });

  return findings;
}
