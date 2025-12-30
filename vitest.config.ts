import { defineConfig } from 'vitest/config';
import yaml from '@rollup/plugin-yaml';

export default defineConfig({
  plugins: [yaml()],
  test: {
    globals: true,
    environment: 'node',
    include: ['tests/unit/**/*.test.ts', 'tests/integration/**/*.test.ts'],
    coverage: {
      provider: 'v8',
      reporter: ['text', 'lcov'],
      exclude: [
        'scripts/content-import.ts',
        'scripts/delete-article.ts',
        'src/config/env.ts',
        'scripts/markdown/index.ts',
      ],
      thresholds: {
        lines: 80,
        functions: 80,
        branches: 75,
        statements: 80,
      },
    },
  },
});
