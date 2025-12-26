/**
 * Pre-process raw Markdown before Astro parses it.
 * Especially useful for Notion exports that occasionally contain stray fences or invisible characters.
 */
export function sanitizeMarkdown(markdown: string): string {
  if (!markdown) return '';

  // Normalize zero-width/invisible characters
  const cleaned = markdown
    .replace(/[\u00a0\u2000-\u200b\ufeff]/g, (char) => (char === '\u00a0' ? ' ' : ''))
    .replace(/\r\n?/g, '\n');

  // Fix unmatched triple backticks that sometimes appear in Notion exports
  const fenceCount = (cleaned.match(/```/g) || []).length;
  if (fenceCount % 2 === 1) {
    return `${cleaned}\n\u200b\n\`\`\``; // close the last fence with a zero-width spacer
  }

  // Collapse excessive blank lines
  return cleaned.replace(/\n{3,}/g, '\n\n').trimStart();
}
