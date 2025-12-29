# Architecture Documentation

> **Design Principle**: Code is the single source of truth. This document reflects the current repository structure and explains design decisions based on what actually exists.

---

## 1. Overall Architecture

This is a **content-pipeline-driven static blog** where content generation and runtime rendering are strictly separated:

```text
┌─────────────────────────────────────────────┐
│         Content Sources                     │
│  ┌─────────┐ ┌──────────┐ ┌──────────────┐ │
│  │ Notion  │ │ External │ │    Local     │ │
│  │Database │ │   URLs   │ │  Markdown    │ │
│  └────┬────┘ └─────┬────┘ └──────┬───────┘ │
└───────┼────────────┼─────────────┼─────────┘
        │            │             │
        v            v             v
┌───────────────────────────────────────────────┐
│         Scripts Layer (Node.js)               │
│  ┌──────────────┐  ┌─────────────────────┐   │
│  │ notion-sync  │  │  content-import     │   │
│  │     .ts      │  │      .ts            │   │
│  └──────┬───────┘  └──────┬──────────────┘   │
│         │                 │                   │
│  ┌──────┴─────────────────┴──────────┐       │
│  │      scripts/utils.ts              │       │
│  │  (Shared utility functions)        │       │
│  └────────────────────────────────────┘       │
└───────────────────┬───────────────────────────┘
                    │ writes
                    v
        ┌───────────────────────────┐
        │   Content Artifacts       │
        │  src/content/blog/        │
        │  public/images/           │
        └───────────┬───────────────┘
                    │ reads at build time
                    v
┌───────────────────────────────────────────────┐
│         Runtime Layer (Astro)                 │
│                                               │
│  ┌────────────────────────────────────────┐  │
│  │          src/lib/                      │  │
│  │  ┌──────┐ ┌──────┐ ┌────────┐         │  │
│  │  │ slug │ │content│ │markdown│ ...     │  │
│  │  └──────┘ └──────┘ └────────┘         │  │
│  └────────────────────────────────────────┘  │
│                                               │
│  ┌────────────────────────────────────────┐  │
│  │  Components / Layouts / Pages          │  │
│  └────────────────────────────────────────┘  │
└───────────────────┬───────────────────────────┘
                    │ astro build
                    v
        ┌───────────────────────────┐
        │   Static Site Output      │
        │       (dist/)             │
        └───────────────────────────┘
```

### Key Architectural Principles

1. **Unidirectional Data Flow**: Content flows from external sources → scripts → artifacts → runtime → static output
2. **Strict Layer Isolation**: Scripts never import from `src/lib`; runtime never fetches external content
3. **Content as Data**: All content in `src/content/blog/` is treated as build-time data, not application logic
4. **Single Responsibility**: Each layer has a clear boundary and purpose

---

## 2. Runtime Layer (`src/`)

The runtime layer contains all code that executes during Astro build or in the browser. It is **completely isolated from scripts** and operates only on pre-generated content artifacts.

### 2.1 Configuration (`src/config/`)

Centralized configuration modules that serve as the single source of truth for paths and settings:

- **`paths.ts`**: All filesystem paths (content directories, image directories, build output)
  - Exports: `ROOT_DIR`, `BLOG_CONTENT_DIR`, `NOTION_CONTENT_DIR`, `PUBLIC_IMAGES_DIR`, `NOTION_PUBLIC_IMG_DIR`, `ARTIFACTS_DIR`, etc.
  - Used by both runtime and scripts to ensure consistency
  - Supports environment variable overrides for testing

- **`site.ts`**: Site metadata (base URL, site URL, title, description)
  - Used by RSS, sitemap, and page metadata generation

- **`env.ts`**: Environment variable parsing utilities

- **`features.ts`**: Feature flags (boolean toggles controlled via environment variables)

**Design Rationale**: By centralizing all paths in `src/config/paths.ts`, we eliminate hardcoded paths and make it easy to reconfigure the project for different environments or deployment targets.

### 2.2 Business Logic (`src/lib/`)

This is the **only location for runtime business logic**. Each subdirectory represents a domain with clear responsibilities:

#### `src/lib/slug/`

**Responsibility**: Single source of truth for all slug generation and validation logic

- Provides `slugFromTitle()` for converting titles to URL-safe slugs
- Provides `ensureUniqueSlug()` for detecting and resolving slug conflicts
- Used by both scripts (during content sync) and runtime (for route generation)

**Why centralized**: Slug consistency is critical for URL stability. Having one module ensures:

- No divergence between script-generated slugs and runtime-expected slugs
- Easy to modify slug algorithm globally
- Slug conflict detection works identically everywhere

#### `src/lib/content/`

**Responsibility**: Content querying, transformation, and metadata extraction

Key modules:

- `posts.ts`: Fetches published posts from Astro's content collection
- `dates.ts`: Date formatting and parsing utilities
- `readingTime.ts`: Calculates estimated reading time from content
- `tocTree.ts`: Builds table-of-contents tree structure from headings
- `slugger.ts`: Creates heading sluggers for anchor link generation

**Why separate from scripts**: This logic operates on _already-synced_ content. It doesn't know or care whether content came from Notion, external URLs, or local Markdown.

#### `src/lib/markdown/`

**Responsibility**: Markdown processing plugins for Astro's unified/remark/rehype pipeline

Key modules:

- `rehypeHeadingLinks.ts`: Adds anchor links to headings
- `rehypeExternalLinks.ts`: Adds `target="_blank"` and security attributes to external links
- `rehypePrettyCode.ts`: Syntax highlighting via Shiki
- `remarkNotionCompat.ts`: Notion-specific Markdown fixes (e.g., callout blocks)
- `remarkPrefixImages.ts`: Prefixes image paths with base URL
- `remarkCodeMeta.ts`: Parses code block metadata

**Why a separate domain**: These are runtime transformations that happen during page rendering, not during content import. They are stateless transformations that work on any Markdown content.

#### `src/lib/site/`

**Responsibility**: Site-level utilities

- `assetUrl.ts`: Resolves asset URLs with proper base path prefixing

#### `src/lib/ui/`

**Responsibility**: Client-side interaction logic

- `code-blocks.ts`: Copy-to-clipboard functionality for code blocks
- `floatingActionStack.ts`: Floating action button stack calculation (scroll-to-top, TOC, etc.)

**Why separate**: Pure client-side JavaScript that has no dependency on content structure.

### 2.4 Content Collection (`src/content/`)

- **`src/content/blog/`**: Contains all blog post Markdown/MDX files
  - `notion/`: Synced from Notion via `notion-sync.ts`
  - `wechat/`: Imported from WeChat articles via `content-import.ts`
  - `others/`: Imported from other platforms (Zhihu, Medium, etc.)
  - Can also contain local Markdown files placed directly by developers

- **`src/content/config.ts`**: Astro content collection schema definition

**Important**: Content files are **data, not logic**. They are generated by scripts and consumed by runtime. Manual edits to synced content (e.g., `notion/`) will be overwritten on next sync.

### 2.5 Dependency Flow in Runtime

```text
src/config/
    ↓
src/lib/
    ↓
src/components/ + src/layouts/
    ↓
src/pages/
```

- **`config`** has no dependencies on other runtime code
- **`lib`** may depend on `config` but not on components/layouts/pages
- **Components/Layouts** may depend on `lib` and `config`
- **Pages** orchestrate everything but contain minimal logic

---

## 3. Scripts Layer (`scripts/`)

The scripts layer is responsible for **content acquisition and preparation**. Scripts run **outside of Astro build** via Node.js and have no knowledge of Astro internals.

### 3.1 Scripts Positioning

**What scripts do:**

- Fetch content from external sources (Notion API, web scraping)
- Download and process images
- Generate Markdown/MDX files with proper frontmatter
- Fix common content issues (math formatting, invisible characters)
- Maintain slug uniqueness and detect conflicts

**What scripts do NOT do:**

- Render content to HTML (that's Astro's job)
- Implement business logic (that belongs in `src/lib/`)
- Get imported by runtime code (strict isolation)

### 3.2 `scripts/utils.ts` - The Shared Utility Layer

**Design Decision**: Use a **single file** for shared script utilities instead of `scripts/lib/`

**Why a single file?**

1. **Scripts are entry points, not a library**: Each script is a standalone command-line tool. They don't form a complex dependency graph that requires modular structure.

2. **Prevents premature abstraction**: Creating `scripts/lib/` invites over-engineering. With a single file, you think carefully before adding utilities.

3. **Clear ownership**: `scripts/utils.ts` contains **only script-specific utilities** that should never be used in runtime. Examples:
   - File I/O helpers (`ensureDir`, `processFile`, `processDirectory`)
   - Process execution wrappers
   - Math delimiter fixing (whitespace normalization in `$ x $` → `$x$`)

4. **Easier to maintain**: One file to review when cleaning up utilities, versus navigating multiple directories.

**What goes in `scripts/utils.ts`:**

- Generic file system operations
- String processing utilities for content fixing
- Process spawning helpers
- Utilities used by 2+ scripts

**What does NOT go in `scripts/utils.ts`:**

- Business logic (slug generation → `src/lib/slug/`)
- Runtime transformations (Markdown plugins → `src/lib/markdown/`)
- Configuration (paths → `src/config/paths.ts`)

### 3.3 Core Scripts

#### `notion-sync.ts`

**Responsibility**: Sync published Notion pages to Markdown files

**Workflow:**

1. Connects to Notion API using credentials from `.env.local`
2. Queries Notion database for pages with `status = "Published"`
3. Downloads page content via `notion-to-md` library
4. Downloads cover images and inline images to `public/images/notion/<pageId>/`
5. Generates slug using `src/lib/slug/slugFromTitle()`
6. Checks for slug conflicts using `src/lib/slug/ensureUniqueSlug()`
7. Writes Markdown files to `src/content/blog/notion/`
8. Calls `process-md-files.ts` to fix math formatting
9. Runs `npm run lint` to format the generated files

**Input:**

- Notion API credentials (`NOTION_TOKEN`, `NOTION_DATABASE_ID`)
- Notion database with Published pages

**Output:**

- `src/content/blog/notion/*.md` files
- `public/images/notion/<pageId>/*` image files

**Relationship with `scripts/utils.ts`:**

- Uses `ensureDir()` for directory creation
- Does NOT contain slug logic (delegates to `src/lib/slug/`)

#### `content-import.ts`

**Responsibility**: Import articles from external URLs (WeChat, Zhihu, Medium)

**Workflow:**

1. Accepts URL via CLI argument (`--url="..."`)
2. Detects platform (WeChat, Zhihu, Medium) based on URL pattern
3. Uses Playwright to scrape article content
4. Downloads all images to `public/images/<platform>/<slug>/`
5. Converts HTML to Markdown using rehype/remark pipeline
6. Generates frontmatter with title, date, author, etc.
7. Writes MDX file to `src/content/blog/<platform>/`
8. Calls `process-md-files.ts` to fix formatting
9. Runs `npm run lint`

**Input:**

- URL of article to import
- Optional flags: `--overwrite`, `--preview`, `--cover-first-image`

**Output:**

- `src/content/blog/<platform>/<slug>.mdx`
- `public/images/<platform>/<slug>/*` image files

**Platform-specific logic:**

- **WeChat**: Handles placeholder images, retries failed downloads, uses browser fallback for stubborn images
- **Zhihu**: Extracts author and publish date
- **Medium**: Similar extraction logic

**Relationship with `scripts/utils.ts`:**

- Uses `ensureDir()` for directory creation
- Uses file processing utilities for post-processing

#### `process-md-files.ts`

**Responsibility**: Fix common formatting issues in Markdown files

**Operations:**

- Remove unnecessary whitespace around inline math (`$ x $` → `$x$`)
- Normalize invisible Unicode characters
- Fix code fence formatting

**Usage:**

- Can be run standalone: `npx tsx scripts/process-md-files.ts <path>`
- Called automatically by `notion-sync.ts` and `content-import.ts`

**Input:**

- File path or directory path

**Output:**

- Modified files in place

**Relationship with `scripts/utils.ts`:**

- Defined as exported function in `utils.ts` for reuse
- Uses `processFile()` and `processDirectory()` helpers

#### `delete-article.ts`

**Responsibility**: Delete articles and associated images

**Workflow:**

1. Accepts target (slug or file path) via CLI argument
2. Finds matching article file
3. Optionally extracts cover image path from frontmatter
4. Deletes article file
5. Optionally deletes associated image directory
6. Supports dry-run mode for safety

**Input:**

- Article slug or file path (`--target=<value>`)
- Optional flags: `--delete-images`, `--dry-run`

**Output:**

- Deleted files (or dry-run report)

---

## Summary

This architecture achieves **maintainability through clarity**:

1. **Two separate worlds**: Scripts (Node.js CLI tools) and Runtime (Astro build)
2. **Strict layer boundaries**: Config → Lib → Components → Pages (runtime); Scripts write artifacts
3. **Domain-driven structure**: Each module has a single, well-defined purpose
4. **Minimal abstraction**: Concrete implementations over speculative frameworks
5. **Single source of truth**: Paths in `src/config/paths`, slugs in `src/lib/slug`, content from external sources

**When in doubt:**

- If it runs during `npm run dev` or `npm run build` → it belongs in `src/`
- If it runs via `npm run notion:sync` or `npm run import:content` → it belongs in `scripts/`
- If it's used by both → it belongs in `src/config/` or `src/lib/slug/` (the only shared modules)
