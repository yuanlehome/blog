import { describe, expect, it } from 'vitest';
import { unified } from 'unified';
import remarkParse from 'remark-parse';
import remarkRehype from 'remark-rehype';
import rehypeStringify from 'rehype-stringify';
import rehypeMermaid from '../src/lib/markdown/rehypeMermaid';
import remarkCodeMeta from '../src/lib/markdown/remarkCodeMeta';

describe('rehypeMermaid', () => {
  it('replaces mermaid code blocks with mermaid placeholders only', async () => {
    const input = [
      '```mermaid',
      'graph TD',
      '  A-->B',
      '```',
      '',
      '```ts',
      'const value = 1;',
      '```',
    ].join('\n');

    const result = await unified()
      .use(remarkParse)
      .use(remarkCodeMeta)
      .use(remarkRehype)
      .use(rehypeMermaid)
      .use(rehypeStringify)
      .process(input);

    const html = String(result);

    expect(html).toContain('class="mermaid-block not-prose"');
    expect(html).toContain('data-mermaid=');
    expect(html).toContain('Rendering diagram…');
    expect(html).toContain('<code class="language-ts" data-lang="ts"');
    expect(html).toContain('const value = 1;');
  });
});
