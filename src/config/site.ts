import { stringFromEnv } from './env';

const defaultBase = stringFromEnv(
  'SITE_BASE',
  process.env.NODE_ENV === 'production' ? '/blog' : '/',
);
const normalizedBase = defaultBase?.startsWith('/') ? defaultBase : `/${defaultBase ?? ''}`;

export const siteUrl = stringFromEnv('SITE_URL', 'https://yuanlehome.github.io/blog/');
export const siteBase = normalizedBase || '/';

export const siteConfig = {
  url: siteUrl,
  base: siteBase,
  trailingSlash: 'always' as const,
};
