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

### Phase 2: Remove `src/utils/` Directory ✅ COMPLETED

**Date**: 2024-12-29

**Changes Made**:

1. Updated all imports in Astro config (astro.config.mjs)
2. Updated all imports in Astro files (9 files):
   - Pages: index.astro, [...slug].astro, page/[page].astro, archive.astro
   - Components: FloatingActionStack.astro, PostList.astro, MobileToc.astro, TocTree.astro, TableOfContents.astro
3. Updated all imports in test files (7 files)
4. Deleted `src/utils/` directory (14 files removed)

**Migration Examples**:

```typescript
// Before
import { getPublishedPosts } from '../utils/posts';
import { resolveAssetUrl } from '../utils/assetUrl';
import { buildTocForest } from '../utils/tocTree';

// After
import { getPublishedPosts } from '../lib/content/posts';
import { resolveAssetUrl } from '../lib/site/assetUrl';
import { buildTocForest } from '../lib/content/tocTree';
```

**Test Results**:

- ✅ npm run check: PASSED (0 errors)
- ✅ npm run lint: PASSED
- ✅ npm run test: PASSED (120/120 tests)

### Phase 3: Create `scripts/utils.ts` ✅ COMPLETED

**Date**: 2024-12-29

**Changes Made**:

1. Created `scripts/utils.ts` with:
   - Directory & File I/O utilities (ensureDir, processFile, processDirectory)
   - Error handling wrapper (runMain)
   - Math fixing utilities (migrated from scripts/lib/shared/math-fix.ts)
2. Updated `scripts/fix-math.ts` to import from `./utils`
3. Deleted `scripts/lib/` directory (1 file removed)
4. Created `scripts/README.md` documenting the new structure

**Utilities Provided by `scripts/utils.ts`**:

- `ensureDir(dir)`: Ensure directory exists
- `processFile(filePath, processFn)`: Process single file
- `processDirectory(dirPath, filterFn, processFn)`: Recursively process files
- `runMain(mainFn)`: Async error handling wrapper
- `fixMath(text)`: Fix math delimiters in markdown
- `normalizeInvisibleCharacters(text)`: Normalize invisible Unicode
- `splitCodeFences(text)`: Split markdown into segments

**Test Results**:

- ✅ npm run check: PASSED (0 errors)
- ✅ npm run lint: PASSED
- ✅ npm run test: PASSED (120/120 tests, 96.3% coverage)

### Phase 4: Scripts Already Using Proper Structure ✅

**Analysis**:

- `scripts/notion-sync.ts`: Already imports from `../src/config/paths` and `../src/lib/slug` ✅
- `scripts/content-import.ts`: Already imports from `../src/config/paths` and `../src/lib/slug` ✅
- `scripts/fix-math.ts`: Now imports from `./utils` ✅
- `scripts/delete-article.ts`: No shared utilities needed ✅

**No additional refactoring needed** - scripts were already well-structured!

## Final Summary

### Files Changed
- **35 files changed**: 443 insertions(+), 88 deletions(-)

### Files Deleted
**`src/utils/` directory (14 files)**:
- assetUrl.ts, code-blocks.ts, dates.ts, floatingActionStack.ts
- posts.ts, readingTime.ts, slugger.ts, tocTree.ts
- rehypeExternalLinks.ts, rehypeHeadingLinks.ts, rehypePrettyCode.ts
- remarkCodeMeta.ts, remarkNotionCompat.ts, remarkPrefixImages.ts

**`scripts/lib/` directory (1 file)**:
- shared/math-fix.ts

### Files Created
- `scripts/utils.ts` (328 lines)
- `scripts/README.md` (139 lines)
- `docs/refactor-log.md` (this file)

### Files Modified
- astro.config.mjs (6 import paths)
- 4 Astro pages, 6 Astro components, 1 Astro layout
- 6 test files, 1 script file

### Migration Examples

#### Example 1: Simple Import Update
```typescript
// Before
import { getPublishedPosts } from '../utils/posts';
// After
import { getPublishedPosts } from '../lib/content/posts';
```

#### Example 2: Script Utilities
```typescript
// Before
import { fixMath } from './lib/shared/math-fix.js';
// After
import { fixMath, processFile, processDirectory } from './utils.js';
```

### Test Results - All Gates Passing ✅

#### Before Refactor
- ✅ npm run check: 0 errors, 5 hints
- ✅ npm run lint: All formatted
- ✅ npm run test: 120/120 tests, 97.7% coverage

#### After Refactor
- ✅ npm run check: 0 errors, 5 hints
- ✅ npm run lint: All formatted
- ✅ npm run test: 120/120 tests, 96.3% coverage
- ✅ npm run build: 20 pages generated

### Behavior Compatibility ✅

**All paths remain unchanged**:
- Notion content: `src/content/blog/notion/`
- Notion images: `public/images/notion/`
- Platform imports: `src/content/blog/<platform>/`
- Platform images: `public/images/<platform>/<slug>/`

### Conclusion

✅ **Refactor completed successfully!**

- Cleaner structure: Removed redundant `src/utils/` re-export layer
- Better organization: `src/lib/` has clear subdirectories by purpose
- Consolidated scripts: Single `scripts/utils.ts` for shared utilities
- No behavior changes: All tests passing, build successful
- Well documented: README files and refactor log completed
