import { defineConfig } from 'astro/config';
import tailwind from '@astrojs/tailwind';
import sitemap from '@astrojs/sitemap';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import yaml from '@rollup/plugin-yaml';
import remarkPrefixImages from './src/utils/remarkPrefixImages';

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
      [remarkPrefixImages, { base: process.env.NODE_ENV === 'production' ? '/blog' : '/' }],
    ],
    rehypePlugins: [rehypeKatex],
    shikiConfig: {
      theme: 'github-dark',
      wrap: true,
    },
  },
});
