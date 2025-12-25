export function getReadingTime(content: string) {
  const wordsPerMinute = 200;
  // Strip HTML tags and newlines, then split by whitespace
  const cleanContent = content.replace(/<[^>]*>/g, '').replace(/\s+/g, ' ').trim();
  const wordCount = cleanContent.split(' ').length;
  const readingTime = Math.ceil(wordCount / wordsPerMinute);
  
  return {
    wordCount,
    readingTime,
    text: `${readingTime} min read`
  };
}
