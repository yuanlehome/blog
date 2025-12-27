import GithubSlugger from 'github-slugger';

export function createHeadingSlugger() {
  const slugger = new GithubSlugger();
  return (text: string) => slugger.slug(text, false);
}
