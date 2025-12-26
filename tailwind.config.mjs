/** @type {import('tailwindcss').Config} */
export default {
  content: ['./src/**/*.{astro,html,js,jsx,md,mdx,svelte,ts,tsx,vue,yml,yaml}'],
  darkMode: 'class',
  theme: {
    extend: {
      fontFamily: {
        sans: ['"Inter var"', 'Inter', '"SF Pro Text"', '"Segoe UI"', '"Noto Sans SC"', '"PingFang SC"', 'system-ui', 'sans-serif'],
        mono: ['"JetBrains Mono"', 'SFMono-Regular', 'Menlo', 'Consolas', 'monospace'],
      },
      typography: (theme) => ({
        DEFAULT: {
          css: {
            color: theme('colors.slate.800'),
            fontSize: '1.05rem',
            lineHeight: '1.8',
            maxWidth: '72ch',
            '--tw-prose-headings': theme('colors.slate.900'),
            '--tw-prose-body': theme('colors.slate.800'),
            '--tw-prose-links': theme('colors.blue.700'),
            '--tw-prose-bold': theme('colors.slate.900'),
            '--tw-prose-code': theme('colors.slate.900'),
            '--tw-prose-quotes': theme('colors.slate.900'),
            '--tw-prose-th-borders': theme('colors.slate.200'),
            '--tw-prose-td-borders': theme('colors.slate.200'),
            h1: {
              fontWeight: '700',
              letterSpacing: '-0.03em',
              lineHeight: '1.2',
            },
            h2: {
              fontWeight: '700',
              letterSpacing: '-0.02em',
              lineHeight: '1.25',
            },
            h3: {
              fontWeight: '600',
              letterSpacing: '-0.01em',
              lineHeight: '1.35',
            },
            'code::before': { content: '""' },
            'code::after': { content: '""' },
            code: {
              color: theme('colors.slate.900'),
              backgroundColor: 'rgba(148, 163, 184, 0.16)',
              padding: '0.15em 0.35em',
              borderRadius: '0.35rem',
              fontWeight: '500',
              fontFamily: 'var(--code-font)',
            },
            'code:not(pre code)': {
              color: theme('colors.slate.900'),
            },
            pre: {
              backgroundColor: 'transparent',
              color: 'inherit',
              padding: '0',
            },
            'pre code': {
              color: 'inherit',
              backgroundColor: 'transparent',
              padding: '0',
            },
            'blockquote p': {
              fontSize: '1.05rem',
              color: theme('colors.slate.800'),
            },
          },
        },
        invert: {
          css: {
            color: theme('colors.slate.200'),
            '--tw-prose-headings': theme('colors.slate.50'),
            '--tw-prose-body': theme('colors.slate.200'),
            '--tw-prose-links': theme('colors.sky.200'),
            '--tw-prose-bold': theme('colors.slate.50'),
            '--tw-prose-code': theme('colors.slate.100'),
            '--tw-prose-quotes': theme('colors.slate.100'),
            '--tw-prose-th-borders': theme('colors.slate.700'),
            '--tw-prose-td-borders': theme('colors.slate.700'),
            code: {
              backgroundColor: 'rgba(148, 163, 184, 0.16)',
              color: theme('colors.slate.100'),
            },
          },
        },
      }),
    },
  },
  plugins: [require('@tailwindcss/typography')],
};
