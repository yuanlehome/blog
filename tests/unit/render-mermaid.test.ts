import { describe, expect, it } from 'vitest';
import { JSDOM } from 'jsdom';
import {
  decodeEdgePoints,
  estimateTextMetrics,
  expandViewBoxToFitEdges,
  normalizeLabelText,
  normalizeMermaidCode,
} from '../../scripts/render-mermaid.mjs';

describe('render-mermaid helpers', () => {
  it('estimates multiline text metrics for mixed CJK/ASCII labels', () => {
    const metrics = estimateTextMetrics('Prompt Dataset\\n复杂标签 Label'); // intentionally escaped \n
    expect(metrics.lines).toBe(2);
    expect(metrics.width).toBeGreaterThan(90);
    expect(metrics.height).toBeGreaterThanOrEqual(36);
  });

  it('expands viewBox to include edge points when they exceed bounds', () => {
    const points = Buffer.from(
      JSON.stringify([
        { x: 20, y: 10 },
        { x: 130, y: 120 },
      ]),
      'utf-8',
    ).toString('base64');
    const dom = new JSDOM(`<svg viewBox="0 0 100 100"><path data-points="${points}" /></svg>`, {
      contentType: 'image/svg+xml',
    });
    const { document } = dom.window;

    expandViewBoxToFitEdges(document, 16);

    const viewBox = document
      .querySelector('svg')
      ?.getAttribute('viewBox')
      ?.split(/\s+/)
      .map(Number);
    expect(viewBox).toBeDefined();
    expect(viewBox?.[2]).toBeGreaterThan(100);
    expect(viewBox?.[3]).toBeGreaterThan(100);
  });

  it('safely decodes malformed edge payloads', () => {
    expect(decodeEdgePoints('not-base64')).toEqual([]);
  });

  it('normalizes escaped newlines for flowchart labels', () => {
    const code = 'flowchart TD\nA["foo\\nbar"] --> B["baz"]';
    expect(normalizeMermaidCode(code)).toContain('foo<br/>bar');
  });

  it('normalizes label text with br and escaped newlines', () => {
    expect(normalizeLabelText('  a<br/>b\\n c  ')).toBe('a\nb\nc');
  });
});
