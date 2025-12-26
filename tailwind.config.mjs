/** @type {import('tailwindcss').Config} */
export default {
	content: ['./src/**/*.{astro,html,js,jsx,md,mdx,svelte,ts,tsx,vue,yml,yaml}'],
	darkMode: 'class',
	theme: {
		extend: {
            typography: (theme) => ({
                DEFAULT: {
                    css: {
                        'code::before': { content: '""' },
                        'code::after': { content: '""' },
                        code: {
                            color: theme('colors.sky.800'),
                            backgroundColor: theme('colors.sky.50'),
                            padding: '0.2em 0.35em',
                            borderRadius: '0.35rem',
                            fontWeight: '500',
                        },
                        'code:not(pre code)': {
                            color: theme('colors.sky.800'),
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
                        '.dark code': {
                            color: theme('colors.sky.200'),
                            backgroundColor: theme('colors.slate.800'),
                        },
                    },
                },
            }),
        },
        },
	plugins: [
		require('@tailwindcss/typography'),
	],
}
