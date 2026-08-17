import { SCALING_BOOK_SHORT_TITLE, SCALING_BOOK_TITLE } from './scalingBookMeta';

export const COLUMNS_INDEX_PATH = '/columns/';
export const MODERN_GPU_PROGRAMMING_ZH_URL = 'https://mlc.ai/modern-gpu-programming-for-mlsys/zh/';

export interface ColumnDefinition {
  slug: string;
  href: `/${string}/` | `https://${string}`;
  isExternal: boolean;
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
    isExternal: false,
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
  {
    slug: 'modern-gpu-programming-for-mlsys',
    href: MODERN_GPU_PROGRAMMING_ZH_URL,
    isExternal: true,
    eyebrow: 'MLC Community · 官方教程',
    title: '面向机器学习系统的现代 GPU 编程',
    shortTitle: 'Modern GPU Programming For MLSys 中文版',
    description:
      '围绕 Blackwell GPU、TIRx、GEMM 与 Flash Attention 4 展开的现代 GPU 编程教程，点击后前往 MLC 官方中文版。',
    statusLabel: '官方中文版 · 外部链接',
    coverImage: '/images/columns/modern-gpu-programming-for-mlsys.svg',
    coverAlt: '由 GPU 计算单元和数据流组成的抽象专栏封面',
    coverWidth: 1200,
    coverHeight: 675,
    topics: ['Blackwell', 'TIRx', 'GEMM', 'Flash Attention 4'],
  },
] as const satisfies readonly ColumnDefinition[];

export function findColumnBySlug(slug: string): ColumnDefinition | undefined {
  return COLUMN_CATALOG.find((column) => column.slug === slug);
}
