# Refactor Plan: Comprehensive Repository Restructure

**Date**: 2025-12-28  
**Repository**: yuanlehome/blog  
**Purpose**: Consolidate slug logic, restructure scripts for testability, organize utils/lib properly

---

## 0. Pre-Flight Validation ✅

All baseline checks MUST pass before refactoring begins:

- ✅ `npm run check` - Passes (0 errors, 0 warnings, 1 hint)
- ✅ `npm run lint` - Passes (prettier + markdownlint clean)
- ✅ `npm run test` - Passes (74 tests, 97.59% coverage)
- ⏸️ `npm run test:e2e` - Deferred (requires build + playwright setup)

---

## 1. Scripts Inventory

### Current Scripts in `scripts/`

| Script              | Purpose                               | Inputs                                          | Outputs                                                           | Side Effects                   | Called By                                            | Issues                                                             |
| ------------------- | ------------------------------------- | ----------------------------------------------- | ----------------------------------------------------------------- | ------------------------------ | ---------------------------------------------------- | ------------------------------------------------------------------ |
| `notion-sync.ts`    | Sync Notion DB to markdown            | `.env.local` (NOTION_TOKEN, NOTION_DATABASE_ID) | `src/content/blog/notion/*.md`, `public/images/notion/**`         | Writes files, downloads images | `npm run notion:sync`                                | Mixes orchestration + business logic, not easily testable          |
| `content-import.ts` | Import from URL (WeChat/Zhihu/Medium) | CLI args (`--url`, `--allow-overwrite`, etc.)   | `src/content/blog/<provider>/*.md`, `public/images/<provider>/**` | Writes files, downloads images | `npm run import:content`                             | Very large (1590 lines), mixes CLI + logic, slug generation inline |
| `fix-math.ts`       | Fix math delimiters in markdown       | File/directory path                             | Modified markdown files in place                                  | Overwrites files               | Called by `notion:sync` and `import:content` scripts | Needs to be pure function + CLI wrapper                            |
| `slug.ts`           | Slug generation helpers               | N/A                                             | N/A                                                               | N/A                            | Imported by `notion-sync.ts`                         | Good structure but limited scope; needs expansion                  |
| `delete-article.ts` | Delete article and optionally images  | CLI args (`--target`, `--delete-images`)        | N/A                                                               | Deletes files/directories      | `npm run delete:article`                             | Reasonable structure                                               |

### Scripts Dependencies Map

```text
notion:sync → notion-sync.ts → fix-math.ts → lint
                            ↘ slug.ts
                            ↘ ../src/config/paths

import:content → content-import.ts → fix-math.ts → lint
                                   ↘ slugify (inline usage)
                                   ↘ ../src/config/paths

delete:article → delete-article.ts → ../src/config/paths
```

### Issues Identified

1. **Slug logic scattered**:
   - `scripts/slug.ts` has `deriveSlug`, `normalizeSlug`, `ensureUniqueSlug`
   - `content-import.ts` uses `slugify` directly (lines 1511-1516)
   - No unified entry point
   - File stem → slug conversion not abstracted

2. **Scripts are monolithic**:
   - `content-import.ts` is 1590 lines with CLI + business logic mixed
   - `notion-sync.ts` mixes orchestration with image download/conversion
   - Hard to unit test without mocking file system

3. **Fix-math not reusable**:
   - `fix-math.ts` has CLI logic mixed with core `fixMath()` function
   - Should be pure function + separate CLI wrapper

---

## 2. Slug Logic Map

### Current Slug Occurrences

#### In Scripts

**scripts/slug.ts** (56 lines):

- `normalizeSlug(value: string): string` - uses `slugify`
- `deriveSlug({ explicitSlug, title, fallbackId })` - priority: explicit → title → fallbackId
- `ensureUniqueSlug(desired, ownerId, used)` - conflict resolution with hash suffix

**scripts/notion-sync.ts**:

- Imports `deriveSlug`, `ensureUniqueSlug` from `./slug`
- Line 296-300: Derives slug from Notion property or title, ensures uniqueness
- Line 52-104: `downloadImage(url, slug, imageId)` - uses slug for directory structure
- Line 156-177: `transformImageBlock()` - uses `currentPageSlug` global state
- Line 194: Extracts slug from frontmatter or filename
- Line 311: Sets `currentPageSlug` for image transformer context
- Line 364: Uses slug for file path: `${slug}.md`

**scripts/content-import.ts**:

- Line 11: Imports `slugify` directly
- Line 1511-1516: Inline slug generation (NOT using `scripts/slug.ts`!)

  ```ts
  const slugFromTitle = slugify(title, { lower: true, strict: true });
  const fallbackSlug = slugify(new URL(targetUrl).pathname.split('/').filter(Boolean).pop() || '', {
    lower: true,
    strict: true,
  });
  const slug = slugFromTitle || fallbackSlug || `${provider.name}-${Date.now()}`;
  ```

- Line 801-822: `downloadImage()` - uses slug for directory structure
- Line 1519-1524: Uses slug in htmlToMdx options

**scripts/delete-article.ts**:

- Line 67-119: `findArticlesBySlug()`, `resolveArticle()` - slug-based file resolution
- Line 198-199: `matchesSlugPattern()` - matches exact or prefixed slugs
- Line 208-250: `findImageDirsBySlug()` - finds image dirs by slug pattern

#### In Source Code

**src/pages/[...slug].astro**:

- Line 20: `params: { slug: post.slug }`
- Line 28: `findPrevNext(all, post.slug)`

**src/lib/content/posts.ts**:

- Line 10-11: `findPrevNext(posts, slug)` - finds post by slug
- Line 25: Filters related posts excluding current slug

**src/lib/content/slugger.ts**:

- TOC heading slug generation (NOT related to post slugs)
- Uses `github-slugger` for heading IDs

**Multiple Astro files** (BASE_URL handling):

- `src/pages/*.astro`, `src/components/*.astro`, `src/layouts/Layout.astro`
- All have duplicate BASE_URL normalization:

  ```ts
  const BASE = import.meta.env.BASE_URL.endsWith('/')
    ? import.meta.env.BASE_URL
    : `${import.meta.env.BASE_URL}/`;
  ```

- Used to construct post URLs: `${BASE}${post.slug}/`

### Slug Call Chain

Notion Sync:
notion-sync.ts → deriveSlug() → normalizeSlug()
→ ensureUniqueSlug()
→ downloadImage(url, slug, ...)
→ Output: ${slug}.md, /images/notion/${slug}/

Content Import:
content-import.ts → slugify() (DIRECT, NOT using scripts/slug.ts!)
→ downloadImage(url, slug, ...)
→ Output: ${slug}.md, /images/<provider>/${slug}/

Local Markdown:
Filename: hello-world.md → Route: /hello-world/
(No explicit slug generation; filename becomes slug via Astro's slug property)

URL Construction:
Astro pages → ${BASE}${post.slug}/ (BASE_URL normalization repeated)

````text

### Slug Issues to Address

1. **No single source of truth**: `scripts/slug.ts` exists but `content-import.ts` doesn't use it
2. **Filename → slug not abstracted**: Local markdown relies on Astro's default behavior
3. **BASE_URL handling duplicated**: 10+ files have identical normalization code
4. **No `buildPostUrl()` helper**: URL construction is scattered
5. **No `ensureUniqueSlugs()` batch function**: Only handles one-at-a-time conflicts

---

## 3. Utils Current State Audit

### Directory Structure

```text
src/
├── utils/              # Re-export layer (thin wrappers)
│   ├── assetUrl.ts     → export * from '../lib/site/assetUrl'
│   ├── dates.ts        → export * from '../lib/content/dates'
│   ├── posts.ts        → export * from '../lib/content/posts'
│   ├── slugger.ts      → export * from '../lib/content/slugger'
│   ├── tocTree.ts      → export * from '../lib/content/tocTree'
│   ├── readingTime.ts  → export * from '../lib/content/readingTime'
│   ├── floatingActionStack.ts → export * from '../lib/ui/floatingActionStack'
│   ├── code-blocks.ts  (actual implementation, should move to lib)
│   ├── rehype*.ts      (actual implementations, should move to lib/markdown/)
│   └── remark*.ts      (actual implementations, should move to lib/markdown/)
├── lib/
│   ├── content/        # Content domain utilities
│   │   ├── dates.ts, posts.ts, readingTime.ts, slugger.ts, tocTree.ts
│   ├── markdown/       # Markdown processing plugins
│   │   ├── rehype*.ts, remark*.ts
│   ├── site/           # Site configuration
│   │   └── assetUrl.ts
│   └── ui/             # UI utilities
│       ├── code-blocks.ts, floatingActionStack.ts
```text

### Reference Count Analysis

| Module           | Import Count | Domain   | Notes                                              |
| ---------------- | ------------ | -------- | -------------------------------------------------- |
| `posts.ts`       | ~10          | Content  | Core content operations, correctly placed          |
| `slugger.ts`     | ~5           | Content  | TOC heading slugs, NOT post slugs (confusing name) |
| `dates.ts`       | ~8           | Content  | Date formatting, correctly placed                  |
| `assetUrl.ts`    | ~15          | Site     | BASE_URL handling, but duplicated in Astro files   |
| `tocTree.ts`     | ~3           | Content  | TOC tree building, correctly placed                |
| `readingTime.ts` | ~5           | Content  | Reading time calculation, correctly placed         |
| `rehype*.ts`     | ~1-2 each    | Markdown | Should be in lib/markdown, some in utils           |
| `remark*.ts`     | ~1-2 each    | Markdown | Should be in lib/markdown, some in utils           |

### Utils/Lib Issues to Address (Phase 4)

1. **Duplicate implementations**:
   - `src/utils/code-blocks.ts` has actual code, not a re-export
   - `src/utils/rehype*.ts` and `src/utils/remark*.ts` have actual code

2. **"Pseudo-utils" (actually domain logic)**:
   - `slugger.ts` is confusingly named (it's for TOC headings, not post slugs)
   - `posts.ts` is domain logic, not a utility

3. **Missing abstractions**:
   - No `src/lib/slug/` for post slug operations
   - No `src/lib/utils/` for truly generic utilities (path, string, etc.)
   - BASE_URL normalization not centralized

4. **Import inconsistencies**:
   - Some files import from `src/utils/`, others from `src/lib/`
   - No clear rule on which to use

---

## 4. README Inconsistencies

### Comparison: README vs Reality

| README Claims                                                                 | Actual Behavior                                                 | Consistent?   |
| ----------------------------------------------------------------------------- | --------------------------------------------------------------- | ------------- |
| "本地首次运行 E2E 请先执行一次 `npx playwright install --with-deps chromium`" | Correct, package.json has this in `ci` script                   | ✅            |
| "npm run notion:sync"                                                         | Runs sync, then fix-math, then lint (correct)                   | ✅            |
| "npm run import:content -- --url='...'"                                       | Correct, supports --url flag                                    | ✅            |
| "输出到 `src/content/blog/notion/`"                                           | Correct                                                         | ✅            |
| "下载图片到 `public/images/notion/`"                                          | Correct                                                         | ✅            |
| "本地 Markdown 在 src/content/blog/ 下"                                       | Actually organized as notion/, wechat/, zhihu/, others/ subdirs | ⚠️ Incomplete |
| "文件名会成为路由的一部分"                                                    | True, but not documented how slug conflicts are handled         | ⚠️ Incomplete |
| "微信公众号图片下载包含占位符检测、重试机制和浏览器回退策略"                  | True, implemented in content-import.ts                          | ✅            |
| Scripts commands listed in table                                              | Missing `delete:article` script                                 | ⚠️ Incomplete |
| "详细的模块边界、依赖方向与脚本/工作流入口说明见 docs/architecture.md"        | docs/architecture.md exists                                     | ✅            |

### Missing Documentation

1. **Slug conflict resolution**: How Notion sync handles slug conflicts with hash suffix
2. **Image directory structure**: Pattern is `public/images/<provider>/<slug>/`
3. **delete:article script**: Not listed in commands table
4. **BASE_URL handling**: Not documented how to configure or how it affects routes
5. **Slug generation rules**: Not documented how titles become slugs (Chinese, emoji, symbols)

---

## 5. Refactor Roadmap (8 Steps)

### Step 1: Create Comprehensive Assessment ✅ CURRENT STEP

**Scope**: Create this document
**Risk**: None (documentation only)
**Rollback**: Delete docs/refactor-plan.md
**Validation**: Document exists and is complete

---

### Step 2: Consolidate Slug Logic (CORE)

**Scope**:

- Create `src/lib/slug/index.ts` with all slug functions
- Move `scripts/slug.ts` logic to `src/lib/slug/`
- Add missing functions:
  - `slugFromFileStem(stem: string): string`
  - `buildPostUrl(slug: string, base?: string): string`
  - `ensureUniqueSlugs(slugs: string[]): Map<string, string>`
- Replace all direct `slugify` usage in `content-import.ts`
- Update `notion-sync.ts` to import from `src/lib/slug/`
- Delete `scripts/slug.ts`

**Files Changed**:

- **New**: `src/lib/slug/index.ts`, `src/lib/slug/index.test.ts`
- **Modified**: `scripts/notion-sync.ts`, `scripts/content-import.ts`
- **Deleted**: `scripts/slug.ts`

**Risk**: Medium (changes critical sync/import logic)
**Rollback Point**: Commit before slug consolidation
**Validation**:

```bash
npm run check    # No type errors
npm run test     # All existing tests pass
npm run lint     # Code style consistent
# Manually test: npm run notion:sync (dry-run if possible)
# Manually test: npm run import:content -- --url=... --dry-run
````

---

### Step 3: Centralize BASE_URL Handling

**Scope**:

- Add `buildPostUrl(slug: string): string` to `src/lib/slug/`
- Add `normalizeBase(base: string): string` to `src/lib/site/assetUrl.ts`
- Replace all duplicate BASE_URL normalization in Astro files
- Update all `${BASE}${post.slug}/` to use `buildPostUrl(post.slug)`

**Files Changed**:

- **Modified**: `src/lib/slug/index.ts`, `src/lib/site/assetUrl.ts`
- **Modified**: 10+ Astro files (pages/, components/, layouts/)

**Risk**: Low (URL construction is well-tested in E2E)  
**Rollback Point**: Commit before BASE_URL changes  
**Validation**:

```bash
npm run check
npm run test
npm run build
npm run test:e2e   # Critical for URL validation
```

#### Implementation Notes (Step 3)

**Status**: ✅ Functions already exist in `src/lib/slug/index.ts`

- `normalizeBase(base: string)` - line 203-206
- `buildPostUrl(slug: string, base?: string)` - line 223-233

**Implementation approach**:

1. Import `buildPostUrl` from `src/lib/slug` into Astro files
2. Remove duplicate BASE_URL normalization code
3. Replace `${BASE}${post.slug}/` with `buildPostUrl(post.slug)`
4. For non-post URLs (like /about, /archive), keep BASE usage or use `normalizeBase()` if needed

**Files to modify** (9 files with BASE_URL.endsWith):

- `src/layouts/Layout.astro`
- `src/components/PostList.astro`
- `src/components/Header.astro`
- `src/components/PrevNext.astro`
- `src/components/RelatedPosts.astro`
- `src/pages/about.astro`
- `src/pages/archive.astro`
- `src/pages/index.astro`
- `src/pages/page/[page].astro`

**Checklist**:

- [x] Import `buildPostUrl` into Astro files
- [x] Replace post URL construction with `buildPostUrl(post.slug)`
- [x] Remove duplicate BASE normalization where no longer needed
- [x] Run `npm run check` - ✅ PASS
- [x] Run `npm run lint` - ✅ PASS
- [x] Run `npm run test` - ✅ PASS (120 tests, 97.7% coverage)
- [x] Run `npm run build` - ✅ PASS
- [x] Run `npm run test:e2e` - ✅ PASS (6 tests)
- [x] Commit: `refactor(step3): centralize BASE_URL handling`

---

---

### Step 4: Refactor Scripts into CLI + Logic Layers

**Scope**:

- Create `scripts/lib/` directory structure:

  ```text
  scripts/lib/
  ├── notion/
  │   ├── sync.ts         # Core sync logic
  │   ├── download.ts     # Image download
  │   └── transform.ts    # Markdown transform
  ├── import/
  │   ├── providers.ts    # Provider definitions
  │   ├── extract.ts      # Content extraction
  │   └── convert.ts      # HTML to MDX
  └── shared/
      ├── math-fix.ts     # Pure fixMath() function
      └── file-utils.ts   # File system wrappers
  ```

- Refactor `notion-sync.ts` to thin CLI wrapper
- Refactor `content-import.ts` to thin CLI wrapper
- Refactor `fix-math.ts` to export pure function + CLI wrapper
- Add unit tests for all `scripts/lib/` modules

**Files Changed**:

- **New**: `scripts/lib/**/*.ts`, `tests/unit/scripts-lib/**/*.test.ts`
- **Modified**: `scripts/notion-sync.ts`, `scripts/content-import.ts`, `scripts/fix-math.ts`

**Risk**: High (major restructure of critical scripts)  
**Rollback Point**: Commit before scripts refactor  
**Validation**:

```bash
npm run check
npm run test      # Must include new scripts/lib tests
npm run notion:sync --help   # Verify CLI still works
npm run import:content --help
npm run test:e2e  # Full integration test
```

#### Implementation Notes (Step 4)

**Analysis of Current State**:

- `content-import.ts`: 1590 lines - Very large CLI + logic mix
- `notion-sync.ts`: 382 lines - Moderate CLI + logic mix
- `fix-math.ts`: 364 lines - Already has good structure with `fixMath()` function

**Implementation Strategy** (Three-phase approach to minimize risk):

**Phase 4.1**: Extract `fix-math.ts` (Lowest risk)

- `fix-math.ts` already has a pure `fixMath()` function (lines ~30-300)
- Move pure function to `scripts/lib/shared/math-fix.ts`
- Keep CLI wrapper in `scripts/fix-math.ts` as thin layer
- This is the template for other refactors

**Phase 4.2**: Refactor `notion-sync.ts` (Medium risk)

- Extract image download logic to `scripts/lib/notion/download.ts`
- Extract markdown transform to `scripts/lib/notion/transform.ts`
- Extract core sync logic to `scripts/lib/notion/sync.ts`
- Keep `scripts/notion-sync.ts` as CLI wrapper with argument parsing

**Phase 4.3**: Refactor `content-import.ts` (Highest risk - 1590 lines)

- Extract provider definitions to `scripts/lib/import/providers.ts`
- Extract content extraction logic to `scripts/lib/import/extract.ts`
- Extract HTML to MDX conversion to `scripts/lib/import/convert.ts`
- Keep `scripts/content-import.ts` as CLI wrapper

**Checklist**:

- [x] **Phase 4.1**: Extract fix-math
  - [x] Create `scripts/lib/shared/math-fix.ts` with pure function
  - [x] Update `scripts/fix-math.ts` to import and use it
  - [x] Re-export functions for backward compatibility
  - [x] Existing tests pass (integration/fix-math.test.ts)
  - [x] Run gate checks - ✅ ALL PASS
- [ ] **Phase 4.2**: Refactor notion-sync
  - [ ] Create `scripts/lib/notion/` structure
  - [ ] Extract and test each module
  - [ ] Update `scripts/notion-sync.ts` to use lib modules
  - [ ] Run gate checks including `npm run notion:sync --help`
- [ ] **Phase 4.3**: Refactor content-import
  - [ ] Create `scripts/lib/import/` structure
  - [ ] Extract and test each module
  - [ ] Update `scripts/content-import.ts` to use lib modules
  - [ ] Run gate checks including `npm run import:content --help`
- [ ] Run full validation
- [ ] Commit: `refactor(step4): separate scripts into CLI + logic layers`

---

### Step 5: Organize Utils/Lib Structure

**Scope**:

- Move actual implementations from `src/utils/` to appropriate `src/lib/` subdirs:
  - `code-blocks.ts` → `src/lib/ui/code-blocks.ts`
  - `rehype*.ts` → `src/lib/markdown/`
  - `remark*.ts` → `src/lib/markdown/`
- Keep `src/utils/` as thin re-export layer (backward compatibility)
- Create `src/lib/utils/` for truly generic utilities
- Update all imports to use `src/lib/` directly (phase out `src/utils/`)
- Add ESLint rule or convention doc to prevent deep path imports

**Files Changed**:

- **Moved**: 6-8 files from `src/utils/` to `src/lib/`
- **Modified**: All files that import from moved modules
- **New**: `src/lib/utils/index.ts`

**Risk**: Low (mostly moving files, not changing logic)  
**Rollback Point**: Commit before file moves  
**Validation**:

```bash
npm run check
npm run test
npm run build
# Verify no broken imports
```

---

### Step 6: Documentation Updates

**Scope**:

- Create `docs/slug.md` with slug specification:
  - Normalization rules (Chinese, emoji, symbols)
  - Conflict resolution strategy
  - Examples from all three sources (Notion, URL import, local)
- Create `scripts/README.md` with:
  - Each script's purpose, inputs/outputs, examples
  - Orchestration flow diagrams
  - Common issues and solutions
- Update root `README.md`:
  - Add delete:article command
  - Clarify content source directory structure
  - Document BASE_URL configuration
  - Add slug conflict handling
  - Update common issues section

**Files Changed**:

- **New**: `docs/slug.md`, `scripts/README.md`
- **Modified**: `README.md`

**Risk**: None (documentation only)  
**Rollback Point**: Any commit  
**Validation**: Manual review of documentation accuracy

---

### Step 7: Test Coverage Completion

**Scope**:

- Add comprehensive tests for `src/lib/slug/`:
  - Chinese characters, emoji, mixed scripts
  - Consecutive spaces, hyphens, underscores
  - Conflict resolution (batch and incremental)
  - URL building (with various BASE_URL formats)
- Add tests for moved `scripts/lib/` modules
- Ensure coverage stays above 95%

**Files Changed**:

- **New**: `tests/unit/slug.test.ts` (expand existing)
- **New**: `tests/unit/scripts-lib/**/*.test.ts`

**Risk**: None (tests only)  
**Rollback Point**: Any commit  
**Validation**:

```bash
npm run test -- --coverage
# Verify coverage ≥ 95%
```

---

### Step 8: Final Validation & Code Review

**Scope**:

- Run full test suite (check, lint, test, test:e2e)
- Request code review
- Run security scan (codeql)
- Update `docs/refactor-plan.md` with results
- Create deliverables summary

**Files Changed**:

- **Modified**: `docs/refactor-plan.md`

**Risk**: None (validation only)  
**Rollback Point**: Any commit  
**Validation**:

```bash
npm run check     # Must pass
npm run lint      # Must pass
npm run test      # Must pass, coverage ≥ 95%
npm run test:e2e  # Must pass
```

---

## 6. Risk Assessment

### High Risk Changes

1. **Slug consolidation** (Step 2): Affects notion-sync and content-import
   - Mitigation: Extensive unit tests, manual dry-run testing
   - Rollback: Revert to commit before Step 2

2. **Scripts refactor** (Step 4): Major restructure of critical automation
   - Mitigation: Keep CLI interface identical, add comprehensive tests
   - Rollback: Revert to commit before Step 4

### Medium Risk Changes

1. **BASE_URL centralization** (Step 3): Changes URL construction
   - Mitigation: E2E tests verify routing, test with GitHub Pages
   - Rollback: Revert to commit before Step 3

### Low Risk Changes

1. **Utils/Lib organization** (Step 5): Mostly file moves
   - Mitigation: TypeScript will catch broken imports
   - Rollback: Revert to commit before Step 5

2. **Documentation** (Step 6): No code changes
   - Rollback: Not needed

---

## 7. Success Criteria

### Must Have (Blocking)

- ✅ All baseline tests pass (check, lint, test)
- ✅ E2E tests pass
- ✅ Slug logic exists in single module (`src/lib/slug/`)
- ✅ No direct `slugify` usage outside `src/lib/slug/`
- ✅ Scripts have CLI + logic separation
- ✅ Test coverage ≥ 95%

### Should Have (Non-Blocking)

- ✅ `docs/slug.md` and `scripts/README.md` created
- ✅ README.md updated with accurate information
- ✅ Utils organized into domain-specific lib/ subdirectories

### Nice to Have

- ⏸️ ESLint rule for preventing deep path imports (can be added later)
- ⏸️ GitHub Actions workflow for running tests on PR (already exists)

---

## 8. Deliverables Checklist

- [ ] **docs/refactor-plan.md** - This document, updated with execution results
- [ ] **docs/slug.md** - Slug specification and examples
- [ ] **scripts/README.md** - Scripts documentation
- [ ] **src/lib/slug/** - Consolidated slug module with tests
- [ ] **scripts/lib/** - Refactored script logic with tests
- [ ] **Updated README.md** - Accurate project documentation
- [ ] **Test coverage report** - ≥95% coverage maintained
- [ ] **Validation evidence** - Screenshots/logs of passing test suite

---

## 9. Timeline Estimate

| Phase                      | Time    | Risk   |
| -------------------------- | ------- | ------ |
| Step 1: Assessment         | 1h      | None   |
| Step 2: Slug consolidation | 3h      | Medium |
| Step 3: BASE_URL           | 2h      | Medium |
| Step 4: Scripts refactor   | 5h      | High   |
| Step 5: Utils/Lib org      | 2h      | Low    |
| Step 6: Documentation      | 2h      | None   |
| Step 7: Test coverage      | 2h      | None   |
| Step 8: Final validation   | 1h      | None   |
| **Total**                  | **18h** |        |

---

## 10. Execution Notes

### Phase 1 Complete ✅

- Created comprehensive assessment document
- Identified all slug occurrences and usage patterns
- Mapped current utils/lib structure
- Documented all scripts and their dependencies
- Listed README inconsistencies

### Next Steps

- Proceed to Step 2: Consolidate slug logic into `src/lib/slug/`
- Create comprehensive test suite for slug module
- Replace all direct slugify usage
