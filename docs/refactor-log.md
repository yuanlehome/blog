# Directory Simplification Refactor Log

## Objective

Simplify project structure by:

1. Removing `src/utils/` (which only re-exports from `src/lib`)
2. Consolidating script utilities into `scripts/utils.ts`
3. Removing `scripts/lib/` directory

## Phase 0: Baseline Verification (2024-12-29)

### Commands Run

```bash
npm run check
npm run lint
npm run test
```

### Results

- ✅ **npm run check**: PASSED (0 errors, 5 warnings about deprecated deriveSlug)
- ✅ **npm run lint**: PASSED (all files formatted, 0 markdown errors)
- ✅ **npm run test**: PASSED (120/120 tests, 97.7% coverage)

### Current Structure

#### `src/utils/` Directory (14 files)

All files are simple re-exports from `src/lib`:

- assetUrl.ts → `../lib/site/assetUrl`
- code-blocks.ts → `../lib/ui/code-blocks`
- dates.ts → `../lib/content/dates`
- floatingActionStack.ts → `../lib/ui/floatingActionStack`
- posts.ts → `../lib/content/posts`
- readingTime.ts → `../lib/content/readingTime`
- rehypeExternalLinks.ts → `../lib/markdown/rehypeExternalLinks`
- rehypeHeadingLinks.ts → `../lib/markdown/rehypeHeadingLinks`
- rehypePrettyCode.ts → `../lib/markdown/rehypePrettyCode`
- remarkCodeMeta.ts → `../lib/markdown/remarkCodeMeta`
- remarkNotionCompat.ts → `../lib/markdown/remarkNotionCompat`
- remarkPrefixImages.ts → `../lib/markdown/remarkPrefixImages`
- slugger.ts → `../lib/content/slugger`
- tocTree.ts → `../lib/content/tocTree`

#### `scripts/lib/` Directory (1 file)

- `shared/math-fix.ts`: Pure utility functions for math delimiter fixing

#### References to `src/utils/`

**Astro Config (astro.config.mjs)**:

- `./src/utils/remarkPrefixImages`
- `./src/utils/remarkNotionCompat`
- `./src/utils/remarkCodeMeta`
- `./src/utils/rehypePrettyCode`
- `./src/utils/rehypeHeadingLinks`
- `./src/utils/rehypeExternalLinks`

**Astro Files (14 files)**:

- `src/pages/index.astro`: posts
- `src/pages/[...slug].astro`: readingTime, assetUrl, posts, dates
- `src/pages/page/[page].astro`: posts
- `src/pages/archive.astro`: posts (groupByYearMonth)
- `src/components/FloatingActionStack.astro`: floatingActionStack
- `src/components/PostList.astro`: readingTime, assetUrl, dates
- `src/components/MobileToc.astro`: tocTree
- `src/components/TocTree.astro`: tocTree
- `src/components/TableOfContents.astro`: tocTree

**Test Files (7 files)**:

- `tests/unit/floating-action-stack.test.ts`
- `tests/unit/assetUrl.test.ts`
- `tests/unit/readingTime.test.ts`
- `tests/unit/posts.test.ts`
- `tests/unit/slugger-toc.test.ts`
- `tests/unit/dates.test.ts`

**Script Files**:

- `scripts/notion-sync.ts`: imports from `../src/lib/slug` and `../src/config/paths` (not utils)
- `scripts/content-import.ts`: imports from `../src/lib/slug` and `../src/config/paths` (not utils)
- `scripts/fix-math.ts`: imports from `./lib/shared/math-fix.js`

## Phase 1: Migration Strategy

### Strategy for `src/utils/` Removal

1. **Create `src/lib/index.ts`** as unified export point (optional, for simplicity)
2. **Replace imports in Astro config** with direct paths to `src/lib/*`
3. **Replace imports in Astro files** using relative paths or create barrel exports
4. **Replace imports in test files** with proper paths to `src/lib/*`
5. **Delete `src/utils/` directory**

### Strategy for `scripts/lib/` Consolidation

1. **Create `scripts/utils.ts`** with:
   - Math fixing utilities (from scripts/lib/shared/math-fix.ts)
   - File I/O helpers (ensureDir, recursive file processing)
   - Logging utilities with prefixes
   - Error handling wrapper (runMain)
2. **Refactor scripts** to import from `./utils` instead of `./lib/*`

3. **Delete `scripts/lib/` directory**

## Migration Progress

(To be filled in as we progress through phases)
