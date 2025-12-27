import { defineConfig } from 'astro/config';
import mdx from '@astrojs/mdx';
import tailwind from '@astrojs/tailwind';
import sitemap from '@astrojs/sitemap';
import remarkMath from 'remark-math';
import remarkGfm from 'remark-gfm';
import rehypeKatex from 'rehype-katex';
import yaml from '@rollup/plugin-yaml';
import remarkPrefixImages from './src/utils/remarkPrefixImages';
import remarkNotionCompat from './src/utils/remarkNotionCompat';
import remarkCodeMeta from './src/utils/remarkCodeMeta';
import rehypePrettyCode from './src/utils/rehypePrettyCode';
import rehypeHeadingLinks from './src/utils/rehypeHeadingLinks';
import rehypeExternalLinks from './src/utils/rehypeExternalLinks';

const siteBase = process.env.SITE_BASE ?? (process.env.NODE_ENV === 'production' ? '/blog' : '/');
const siteUrl = process.env.SITE_URL ?? 'https://yuanlehome.github.io/blog/';

// https://astro.build/config
export default defineConfig({
  site: siteUrl,
  base: siteBase,
  trailingSlash: 'always',
  integrations: [mdx(), tailwind(), sitemap()],
  vite: {
    plugins: [yaml()]
  },
  markdown: {
    remarkPlugins: [
      remarkMath,
      remarkGfm,
      remarkNotionCompat,
      remarkCodeMeta,
      [remarkPrefixImages, { base: siteBase }],
    ],
    rehypePlugins: [
      rehypeKatex,
      rehypePrettyCode,
      rehypeHeadingLinks,
      [rehypeExternalLinks, { target: '_blank', rel: ['noopener', 'noreferrer'] }],
    ],
    syntaxHighlight: false,
  },
});
