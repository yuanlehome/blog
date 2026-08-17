import { SCALING_BOOK_SHORT_TITLE, SCALING_BOOK_TITLE } from './scalingBookMeta';

export const COLUMNS_INDEX_PATH = '/columns/';

export interface ColumnDefinition {
  slug: string;
  href: `/${string}/`;
  eyebrow: string;
  title: string;
  shortTitle: string;
  description: string;
  statusLabel: string;
  coverImage: `/${string}`;
  coverAlt: string;
  coverWidth: number;
  coverHeight: number;
  topics: readonly string[];
}

/**
 * Single source of truth for the dedicated columns shown on the columns index.
 * Add the next column here and its card and navigation active state appear automatically.
 */
export const COLUMN_CATALOG = [
  {
    slug: 'scaling-book',
    href: '/scaling-book/',
    eyebrow: '大模型系统 · 完整译著',
    title: SCALING_BOOK_TITLE,
    shortTitle: SCALING_BOOK_SHORT_TITLE,
    description:
      '从 Roofline、TPU 与分片矩阵出发，系统讲解 Transformer 的训练、推理、性能剖析和 GPU 架构。',
    statusLabel: '13 章已完整翻译',
    coverImage: '/images/scaling-book/img/dragon.png',
    coverAlt: 'Scaling Book 原书中的龙形插图',
    coverWidth: 1784,
    coverHeight: 631,
    topics: ['LLM 系统', 'TPU', 'GPU', '训练与推理'],
  },
] as const satisfies readonly ColumnDefinition[];

export function findColumnBySlug(slug: string): ColumnDefinition | undefined {
  return COLUMN_CATALOG.find((column) => column.slug === slug);
}
