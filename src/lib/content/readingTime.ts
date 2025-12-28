export function getReadingTime(content: string) {
  const wordsPerMinute = 450;
  // Remove HTML tags but keep spacing so that inline text stays separable
  const cleanContent = content.replace(/<[^>]*>/g, ' ');

  // Count tokens in a way that works for Chinese, English, numbers, and code:
  // - Each Han character counts as 1
  // - Continuous runs of ASCII letters/digits (including common apostrophes) count as 1
  const tokens = cleanContent.match(/\p{Script=Han}|[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?/gu);
  const wordCount = tokens ? tokens.length : 0;
  const readingTime = wordCount > 0 ? Math.max(1, Math.ceil(wordCount / wordsPerMinute)) : 0;

  return {
    wordCount,
    readingTime,
    text: `${readingTime} min read`,
  };
}
