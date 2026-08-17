import { getCollection, type CollectionEntry } from 'astro:content';

export {
  SCALING_BOOK_REPOSITORY_URL,
  SCALING_BOOK_SHORT_TITLE,
  SCALING_BOOK_SOURCE_COMMIT,
  SCALING_BOOK_SOURCE_URL,
  SCALING_BOOK_TITLE,
} from './scalingBookMeta';

export type ScalingBookEntry = CollectionEntry<'scalingBook'>;

export interface ScalingBookReference {
  key: string;
  title: string;
  authors: string;
  year: number;
  url?: string;
}

export interface ScalingBookAuthor {
  name: string;
  url: string;
}

export const SCALING_BOOK_AUTHORS: ScalingBookAuthor[] = [
  { name: 'Jacob Austin', url: 'https://www.jacobaustin.org/' },
  { name: 'Sholto Douglas', url: 'https://x.com/_sholtodouglas' },
  { name: 'Roy Frostig', url: 'https://cs.stanford.edu/~rfrostig/' },
  { name: 'Anselm Levskaya', url: 'https://anselmlevskaya.com/' },
  { name: 'Charlie Chen', url: 'https://x.com/charliexychen' },
  { name: 'Sharad Vikram', url: 'https://sharadvikram.com/' },
  { name: 'Federico Lebron', url: 'https://fedelebron.com/' },
  { name: 'Peter Choy', url: 'https://x.com/pchoy95' },
  { name: 'Vinay Ramasesh', url: 'https://x.com/vinayramasesh' },
  { name: 'Albert Webson', url: 'https://representation.ai/' },
  { name: 'Reiner Pope', url: 'https://x.com/reinerpope' },
];

export const SCALING_BOOK_CHAPTER_CONTRIBUTORS = [
  {
    chapter: 10,
    authors: [{ name: 'Yash Katariya', url: 'https://x.com/yashk2810' }],
    affiliation: undefined,
  },
  {
    chapter: 12,
    authors: [
      { name: 'Jacob Austin', url: 'https://www.jacobaustin.org/' },
      { name: 'Swapnil Patil', url: 'https://www.linkedin.com/in/swapnil-patil-5b47a068' },
      { name: 'Adam Paszke', url: 'https://x.com/apaszke' },
      { name: 'Reiner Pope', url: 'https://x.com/reinerpope' },
    ],
    affiliation: '本章作者标注的机构为 Google DeepMind 与 MatX。',
  },
] satisfies Array<{
  chapter: number;
  authors: ScalingBookAuthor[];
  affiliation?: string;
}>;

export const SCALING_BOOK_REFERENCES: ScalingBookReference[] = [
  {
    key: 'transformers',
    title: 'Attention Is All You Need',
    authors: 'Vaswani et al.',
    year: 2017,
    url: 'https://arxiv.org/abs/1706.03762',
  },
  {
    key: 'tpu_paper',
    title:
      'TPU v4: An Optically Reconfigurable Supercomputer for Machine Learning with Hardware Support for Embeddings',
    authors: 'Jouppi et al.',
    year: 2023,
    url: 'https://arxiv.org/abs/2304.01433',
  },
  {
    key: 'glu',
    title: 'GLU Variants Improve Transformer',
    authors: 'Shazeer',
    year: 2020,
    url: 'https://arxiv.org/abs/2002.05202',
  },
  {
    key: 'mqa',
    title: 'Fast Transformer Decoding: One Write-Head Is All You Need',
    authors: 'Shazeer',
    year: 2019,
    url: 'https://arxiv.org/abs/1911.02150',
  },
  {
    key: 'gmqa',
    title: 'GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints',
    authors: 'Ainslie et al.',
    year: 2023,
    url: 'https://arxiv.org/abs/2305.13245',
  },
  {
    key: 'moe',
    title: 'Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer',
    authors: 'Shazeer et al.',
    year: 2017,
    url: 'https://arxiv.org/abs/1701.06538',
  },
  {
    key: 'zero',
    title: 'ZeRO: Memory Optimizations Toward Training Trillion Parameter Models',
    authors: 'Rajbhandari et al.',
    year: 2019,
    url: 'https://arxiv.org/abs/1910.02054',
  },
  {
    key: 'megatron',
    title: 'Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism',
    authors: 'Shoeybi et al.',
    year: 2019,
    url: 'https://arxiv.org/abs/1909.08053',
  },
  {
    key: 'DeepSeek3',
    title: 'DeepSeek-V3 Technical Report',
    authors: 'DeepSeek-AI et al.',
    year: 2024,
    url: 'https://arxiv.org/abs/2412.19437',
  },
  {
    key: 'llama3',
    title: 'The Llama 3 Herd of Models',
    authors: 'Grattafiori et al.',
    year: 2024,
    url: 'https://arxiv.org/abs/2407.21783',
  },
  {
    key: 'esti',
    title: 'Efficiently Scaling Transformer Inference',
    authors: 'Pope et al.',
    year: 2022,
    url: 'https://arxiv.org/abs/2211.05102',
  },
  {
    key: 'paged',
    title: 'Efficient Memory Management for Large Language Model Serving with PagedAttention',
    authors: 'Kwon et al.',
    year: 2023,
    url: 'https://arxiv.org/abs/2309.06180',
  },
  {
    key: 'spec1',
    title: 'Fast Inference from Transformers via Speculative Decoding',
    authors: 'Leviathan et al.',
    year: 2022,
    url: 'https://arxiv.org/abs/2211.17192',
  },
  {
    key: 'spec2',
    title: 'Accelerating Large Language Model Decoding with Speculative Sampling',
    authors: 'Chen et al.',
    year: 2023,
    url: 'https://arxiv.org/abs/2302.01318',
  },
  {
    key: 'eagle',
    title: 'EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty',
    authors: 'Li et al.',
    year: 2024,
    url: 'https://arxiv.org/abs/2401.15077',
  },
  {
    key: 'medusa',
    title: 'Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads',
    authors: 'Cai et al.',
    year: 2024,
    url: 'https://arxiv.org/abs/2401.10774',
  },
];

export async function getScalingBookChapters(): Promise<ScalingBookEntry[]> {
  const chapters = await getCollection('scalingBook');
  return chapters.sort((a, b) => a.data.order - b.data.order);
}

export function findScalingBookPrevNext(chapters: ScalingBookEntry[], slug: string) {
  const index = chapters.findIndex((chapter) => chapter.slug === slug);
  return {
    prev: index > 0 ? chapters[index - 1] : undefined,
    next: index >= 0 && index < chapters.length - 1 ? chapters[index + 1] : undefined,
  };
}
