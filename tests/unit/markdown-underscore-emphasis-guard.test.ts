import prettier from 'prettier';
import { describe, expect, it, vi } from 'vitest';

import { findSuspiciousUnderscoreEmphasis } from '../../scripts/markdown/underscore-emphasis-guard';

const prettierDebug = prettier as typeof prettier & {
  __debug: {
    parse(text: string, options: { parser: 'markdown' }): Promise<{ ast: Record<string, unknown> }>;
  };
};

describe('findSuspiciousUnderscoreEmphasis', () => {
  it('flags identifier-like underscores when the parser reports emphasis', async () => {
    const markdown = `---
title: Test
---

18. V1 中 model_runner 为非首段 rank 预分配固定地址的 \`intermediate_tensors\` 缓冲区（用于 CUDA Graph 捕获）；执行时通过 copy_ 将接收数据写入该预分配缓冲区，避免动态分配。
`;
    const raw =
      '_runner 为非首段 rank 预分配固定地址的 `intermediate_tensors` 缓冲区（用于 CUDA Graph 捕获）；执行时通过 copy_';
    const body = markdown.slice(markdown.indexOf('\n---\n') + '\n---\n'.length);
    const startOffset = body.indexOf(raw);
    expect(startOffset).toBeGreaterThanOrEqual(0);
    const parseSpy = vi.spyOn(prettierDebug.__debug, 'parse').mockResolvedValueOnce({
      ast: {
        type: 'root',
        children: [
          {
            type: 'emphasis',
            position: {
              start: { offset: startOffset },
              end: { offset: startOffset + raw.length },
            },
          },
        ],
      },
    } as Awaited<ReturnType<typeof prettierDebug.__debug.parse>>);

    try {
      const findings = await findSuspiciousUnderscoreEmphasis(markdown);

      expect(parseSpy).toHaveBeenCalledWith(body, { parser: 'markdown' });
      expect(findings).toHaveLength(1);
      expect(findings[0]).toMatchObject({
        line: 5,
        nodeType: 'emphasis',
        raw,
      });
    } finally {
      parseSpy.mockRestore();
    }
  });

  it('ignores inline code spans with underscores', async () => {
    const markdown = 'Use `model_runner` and `copy_` for the PP receive buffer.\n';

    await expect(findSuspiciousUnderscoreEmphasis(markdown)).resolves.toEqual([]);
  });

  it('ignores intended underscore emphasis in prose', async () => {
    const markdown = 'This is _important_ for CUDA Graph capture.\n';

    await expect(findSuspiciousUnderscoreEmphasis(markdown)).resolves.toEqual([]);
  });

  it('ignores fenced code blocks and frontmatter values', async () => {
    const markdown = [
      '---',
      'title: model_runner copy_',
      '---',
      '',
      '```python',
      'model_runner()',
      'copy_ = tensor',
      '```',
      '',
    ].join('\n');

    await expect(findSuspiciousUnderscoreEmphasis(markdown)).resolves.toEqual([]);
  });
});
