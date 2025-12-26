import { defineConfig } from 'astro/config';
import tailwind from '@astrojs/tailwind';
import sitemap from '@astrojs/sitemap';
import remarkMath from 'remark-math';
import remarkGfm from 'remark-gfm';
import rehypeKatex from 'rehype-katex';
import yaml from '@rollup/plugin-yaml';
import remarkPrefixImages from './src/utils/remarkPrefixImages';
import remarkNotionCompat from './src/utils/remarkNotionCompat';
import rehypeCodeEnhance from './src/utils/rehypeCodeEnhance';
import rehypeHeadingLinks from './src/utils/rehypeHeadingLinks';
import rehypeExternalLinks from './src/utils/rehypeExternalLinks';

// https://astro.build/config
export default defineConfig({
  site: 'https://yuanlehome.github.io/blog/',
  base: process.env.NODE_ENV === 'production' ? '/blog' : '/',
  trailingSlash: 'always',
  integrations: [tailwind(), sitemap()],
  vite: {
    plugins: [yaml()]
  },
  markdown: {
    remarkPlugins: [
      remarkMath,
      remarkGfm,
      remarkNotionCompat,
      [remarkPrefixImages, { base: process.env.NODE_ENV === 'production' ? '/blog' : '/' }],
    ],
    rehypePlugins: [
      rehypeKatex,
      rehypeHeadingLinks,
      [rehypeExternalLinks, { target: '_blank', rel: ['noopener', 'noreferrer'] }],
      rehypeCodeEnhance,
    ],
    shikiConfig: {
      theme: 'github-dark',
      wrap: true,
    },
  },
});
