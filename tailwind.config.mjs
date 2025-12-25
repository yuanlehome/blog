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
                            color: theme('colors.blue.600'),
                            backgroundColor: theme('colors.gray.100'),
                            padding: '0.2em 0.4em',
                            borderRadius: '0.25rem',
                            fontWeight: '500',
                        },
                        'code:not(pre code)': {
                            color: theme('colors.blue.600'),
                        },
                        // Dark mode overrides
                        '.dark code': {
                            color: theme('colors.blue.400'),
                            backgroundColor: theme('colors.gray.800'),
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
