/**
 * Dependency-cruiser configuration to enforce layering rules between
 * Runtime (`src/`) and Scripts (`scripts/`) as described in docs/architecture.md.
 *
 * Key rules:
 * - Runtime code (`src/**`) MUST NOT import anything from `scripts/**`.
 * - Scripts (`scripts/**`) MAY import only:
 *   - `src/config/paths.ts`
 *   - `src/lib/slug/**`
 */

/** @type {import('dependency-cruiser').IConfiguration} */
module.exports = {
  options: {
    tsPreCompilationDeps: true,
    exclude: {
      path: [
        '^node_modules',
        '^dist',
        '^artifacts',
        '^tests',
        '^public',
        '^.github',
      ],
    },
    doNotFollow: {
      path: ['node_modules'],
    },
    reporterOptions: {
      dot: {
        collapsePattern: 'node_modules/[^/]+',
      },
    },
  },
  forbidden: [
    {
      name: 'no-scripts-from-runtime',
      severity: 'error',
      from: {
        path: '^src/',
      },
      to: {
        path: '^scripts/',
      },
    },
    {
      name: 'scripts-only-use-whitelisted-runtime-modules',
      severity: 'error',
      from: {
        path: '^scripts/',
      },
      to: {
        path: '^src/',
        pathNot: '^src/config/paths\\.ts$|^src/lib/slug/',
      },
    },
  ],
};

