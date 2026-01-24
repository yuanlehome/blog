import type { CollectionEntry } from 'astro:content';
import { shortHash } from '../slug';

const MAX_TAG_LENGTH = 32;

export function normalizeTag(tag: string): string {
  if (!tag) return '';
  const collapsed = tag
    .trim()
    .replace(/\s+/g, ' ')
    .normalize('NFKC');
  if (!collapsed) return '';
  const limited = collapsed.length > MAX_TAG_LENGTH ? collapsed.slice(0, MAX_TAG_LENGTH) : collapsed;
  return limited;
}

export function normalizeTags(tags: string[]): string[] {
  const seen = new Set<string>();
  const result: string[] = [];
  for (const raw of tags) {
    const normalized = normalizeTag(raw);
    if (!normalized) continue;
    const key = normalized.toLowerCase();
    if (seen.has(key)) continue;
    seen.add(key);
    result.push(normalized);
  }
  return result;
}

export function slugifyTag(tag: string): string {
  const normalized = normalizeTag(tag);
  if (!normalized) return '';
  const lower = normalized.toLowerCase();
  const withHyphen = lower.replace(/\s+/g, '-');
  const cleaned = withHyphen
    .replace(/[^\p{L}\p{N}-]+/gu, '-')
    .replace(/-+/g, '-')
    .replace(/^-|-$/g, '');
  if (cleaned) return cleaned;
  return encodeURIComponent(lower);
}

function ensureUniqueTagSlug(baseSlug: string, tag: string, used: Map<string, string>): string {
  let slug = baseSlug;
  const existing = used.get(slug);
  if (!existing || existing === tag) {
    used.set(slug, tag);
    return slug;
  }
  let candidate = `${baseSlug}-${shortHash(tag)}`;
  let counter = 2;
  while (used.has(candidate) && used.get(candidate) !== tag) {
    candidate = `${baseSlug}-${shortHash(`${tag}-${counter}`)}`;
    counter += 1;
  }
  used.set(candidate, tag);
  return candidate;
}

export type TagStat = { tag: string; slug: string; count: number };

export function buildTagMaps(posts: CollectionEntry<'blog'>[]) {
  const tagToSlug = new Map<string, string>();
  const slugToTag = new Map<string, string>();
  const usedSlugs = new Map<string, string>();

  for (const post of posts) {
    const perPostTags = normalizeTags(post.data.tags ?? []);
    for (const tag of perPostTags) {
      if (tagToSlug.has(tag)) continue;
      const baseSlug = slugifyTag(tag);
      if (!baseSlug) continue;
      const unique = ensureUniqueTagSlug(baseSlug, tag, usedSlugs);
      tagToSlug.set(tag, unique);
      slugToTag.set(unique, tag);
    }
  }

  return { tagToSlug, slugToTag };
}

export function getTagStats(posts: CollectionEntry<'blog'>[]): TagStat[] {
  const stats = new Map<string, TagStat>();
  const { tagToSlug } = buildTagMaps(posts);

  for (const post of posts) {
    const perPostTags = normalizeTags(post.data.tags ?? []);
    for (const tag of perPostTags) {
      const slug = tagToSlug.get(tag);
      if (!slug) continue;
      const existing = stats.get(slug);
      if (existing) {
        existing.count += 1;
      } else {
        stats.set(slug, { tag, slug, count: 1 });
      }
    }
  }

  return Array.from(stats.values()).sort((a, b) => {
    if (b.count !== a.count) return b.count - a.count;
    return a.tag.localeCompare(b.tag, 'en');
  });
}

export function getPostsByTagSlug(posts: CollectionEntry<'blog'>[], slug: string) {
  const { slugToTag } = buildTagMaps(posts);
  const targetTag = slugToTag.get(slug);
  if (!targetTag) return [];

  return posts
    .filter((post) => normalizeTags(post.data.tags ?? []).includes(targetTag))
    .sort((a, b) => b.data.date.valueOf() - a.data.date.valueOf());
}

export function getAllTags(posts: CollectionEntry<'blog'>[]): TagStat[] {
  return getTagStats(posts);
}
