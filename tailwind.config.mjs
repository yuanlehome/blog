/** @type {import('tailwindcss').Config} */
export default {
	content: ['./src/**/*.{astro,html,js,jsx,md,svelte,ts,tsx,vue,yml,yaml}'],
	darkMode: 'class',
	theme: {
		extend: {
            typography: (theme) => ({
                DEFAULT: {
                    css: {
                        'code::before': { content: '""' },
                        'code::after': { content: '""' },
                        code: {
                            color: theme('colors.zinc.700'),
                            backgroundColor: theme('colors.zinc.100'),
                            padding: '0.2em 0.4em',
                            borderRadius: '0.25rem',
                            fontWeight: '500',
                            fontSize: '0.875em',
                            fontFamily: 'var(--code-font)',
                        },
                        'code:not(pre code)': {
                            color: theme('colors.zinc.700'),
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
                            color: theme('colors.zinc.200'),
                            backgroundColor: theme('colors.zinc.800'),
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
