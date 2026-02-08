import type { CollectionEntry } from 'astro:content';

type BlogData = CollectionEntry<'blog'>['data'];

const normalizeDate = (value: Date | string | undefined | null) => {
  if (!value) return null;
  const date = value instanceof Date ? value : new Date(value);
  return Number.isNaN(date.valueOf()) ? null : date;
};

export const formatDate = (value: Date | string | undefined | null) => {
  const date = normalizeDate(value);
  if (!date) return '';
  return date.toISOString().slice(0, 10);
};

export const resolveUpdatedDate = (data: BlogData) =>
  normalizeDate(data.updatedAt) ?? normalizeDate(data.updated) ?? normalizeDate(data.lastmod);

export const getDisplayDates = (data: BlogData) => {
  const published = normalizeDate(data.date);
  const updated = resolveUpdatedDate(data);

  const publishedLabel = formatDate(published);
  const updatedLabel = formatDate(updated);
  const isUpdatedAfterPublish =
    Boolean(updated && !published) ||
    Boolean(updated && published && updated.valueOf() > published.valueOf());
  const shouldShowUpdated = Boolean(
    updatedLabel && updatedLabel !== publishedLabel && isUpdatedAfterPublish,
  );

  return {
    published,
    updated,
    publishedLabel,
    updatedLabel: shouldShowUpdated ? updatedLabel : '',
  };
};
