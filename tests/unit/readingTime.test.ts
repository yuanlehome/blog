import { describe, expect, it } from 'vitest';
import { getReadingTime } from '../../src/lib/content/readingTime';

describe('reading time', () => {
  it('counts mixed language tokens and strips html', () => {
    const result = getReadingTime('<p>你好 world 123</p>');
    expect(result.wordCount).toBe(4);
    expect(result.readingTime).toBe(1);
    expect(result.text).toBe('1 min read');
  });

  it('handles empty and whitespace-only content', () => {
    const result = getReadingTime('   ');
    expect(result.wordCount).toBe(0);
    expect(result.readingTime).toBe(0);
  });
});
