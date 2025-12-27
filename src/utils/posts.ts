import { getCollection, type CollectionEntry } from 'astro:content';

export async function getPublishedPosts() {
  const posts = await getCollection('blog');
  return posts
    .filter((post) => post.data.status === 'published')
    .sort((a, b) => b.data.date.valueOf() - a.data.date.valueOf());
}

export function findPrevNext(posts: CollectionEntry<'blog'>[], slug: string) {
  const index = posts.findIndex((p) => p.slug === slug);
  return {
    prev: index > 0 ? posts[index - 1] : undefined,
    next: index >= 0 && index < posts.length - 1 ? posts[index + 1] : undefined,
  };
}

export function findRelated(
  posts: CollectionEntry<'blog'>[],
  current: CollectionEntry<'blog'>,
  limit = 4,
) {
  const tags = new Set(current.data.tags);
  return posts
    .filter((p) => p.slug !== current.slug && p.data.tags.some((t) => tags.has(t)))
    .slice(0, limit);
}

export function groupByYearMonth(posts: CollectionEntry<'blog'>[]) {
  const map = new Map<string, CollectionEntry<'blog'>[]>();
  posts.forEach((post) => {
    const d = post.data.date;
    const key = `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}`;
    const list = map.get(key) ?? [];
    list.push(post);
    map.set(key, list);
  });
  return Array.from(map.entries())
    .sort((a, b) => (a[0] < b[0] ? 1 : -1))
    .map(([key, list]) => ({
      key,
      posts: list.sort((a, b) => b.data.date.valueOf() - a.data.date.valueOf()),
    }));
}
