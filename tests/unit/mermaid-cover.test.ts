import { describe, expect, it } from 'vitest';
import { getMermaidCoverFromBody } from '../../src/lib/content/mermaidCover';
import {
  mermaidImagePath,
  parseMermaidOptions,
  resolveMermaidSlug,
} from '../../src/lib/markdown/remarkMermaid';

const SIMPLE_MERMAID = 'graph TD\nA-->B';
const BODY_WITH_MERMAID = `# Title\n\n\`\`\`mermaid\n${SIMPLE_MERMAID}\n\`\`\`\n\nsome text`;
const BODY_NO_MERMAID = '# Title\n\nJust some text without mermaid.';

describe('getMermaidCoverFromBody', () => {
  it('returns undefined when there is no mermaid block', () => {
    expect(getMermaidCoverFromBody(BODY_NO_MERMAID, 'notion/post.md')).toBeUndefined();
  });

  it('returns undefined for empty body', () => {
    expect(getMermaidCoverFromBody('', 'notion/post.md')).toBeUndefined();
  });

  it('uses directory slug derived from postId (not the URL slug)', () => {
    const result = getMermaidCoverFromBody(BODY_WITH_MERMAID, 'notion/my-post.md');
    expect(result).toBeDefined();
    // Should use the directory "notion" as the slug, not the post filename
    expect(result).toContain('/generated/mermaid/notion/');
    expect(result).toContain('.light.svg');
  });

  it('produces the same path as render-mermaid would write', () => {
    // Simulate what render-mermaid.mjs does: resolveMermaidSlug on the full file path
    const fullPath = '/repo/src/content/blog/notion/my-post.md';
    const dirSlug = resolveMermaidSlug(fullPath); // → 'notion'
    const options = parseMermaidOptions(undefined);
    const expected = mermaidImagePath(SIMPLE_MERMAID, options, dirSlug, 'light');

    const result = getMermaidCoverFromBody(BODY_WITH_MERMAID, 'notion/my-post.md');
    expect(result).toBe(expected);
  });

  it('falls back to "shared" slug when postId is undefined', () => {
    const result = getMermaidCoverFromBody(BODY_WITH_MERMAID, undefined);
    expect(result).toBeDefined();
    expect(result).toContain('/generated/mermaid/shared/');
  });

  it('handles nested directory IDs correctly (e.g. wechat/sub/post.md)', () => {
    const result = getMermaidCoverFromBody(BODY_WITH_MERMAID, 'wechat/subdir/post.md');
    expect(result).toBeDefined();
    // Should use 'wechat/subdir' as the slug
    expect(result).toContain('/generated/mermaid/wechat/subdir/');
  });

  it('returns a path using root slug for top-level posts', () => {
    const result = getMermaidCoverFromBody(BODY_WITH_MERMAID, 'my-top-level-post.md');
    expect(result).toBeDefined();
    expect(result).toContain('/generated/mermaid/root/');
  });
});
