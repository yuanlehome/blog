# Astro Static Blog

A production-ready static blog built with **Astro**, designed for maintainability and content flexibility. Content comes from three sources: **Notion**, **external URLs** (WeChat, Zhihu, Medium), and **local Markdown** files.

**Core Design Philosophy**: Content acquisition (scripts) and content rendering (Astro runtime) are strictly separated, enabling reproducible builds and clear boundaries between data and logic.

---

## 🎯 Project Overview

This is an **Astro-powered static blog** with the following characteristics:

- **Multiple Content Sources**: Import from Notion databases, scrape from external URLs, or write local Markdown
- **Build-Time Content Generation**: All content is pre-fetched and stored as Markdown/MDX files before Astro build
- **Math Support**: KaTeX rendering for both inline (`$...$`) and block (`$$...$$`) math equations
- **Modern Tooling**: TypeScript, Tailwind CSS, Vitest, Playwright, automated CI/CD

**Key Design Choice**: Content is **acquired by scripts** (Node.js CLI tools) and **rendered by Astro** (static site generator). This separation ensures:

- Reproducible builds (same content files → same output)
- Fast builds (no API calls during `astro build`)
- Clear ownership (content sync = scripts, content display = Astro)

---

## 📁 Directory Structure

```
blog/
├── src/
│   ├── lib/              # Runtime business logic (slug, content, markdown plugins)
│   ├── config/           # Configuration (paths, site metadata, env variables)
│   ├── content/          # Content collection (blog posts in Markdown/MDX)
│   │   └── blog/
│   │       ├── notion/   # Synced from Notion (auto-generated)
│   │       ├── wechat/   # Imported from WeChat articles (auto-generated)
│   │       ├── others/   # Imported from other platforms (auto-generated)
│   │       └── [local]   # Local Markdown files (manually written)
│   ├── components/       # Astro/React components
│   ├── layouts/          # Page layouts
│   └── pages/            # Astro routes
├── scripts/
│   ├── notion-sync.ts       # Notion → Markdown sync
│   ├── content-import.ts    # External URL → Markdown import
│   ├── process-md-files.ts  # Math formatting fixes
│   ├── delete-article.ts    # Delete articles and images
│   └── utils.ts             # Shared script utilities (NOT for runtime)
├── public/
│   └── images/
│       ├── notion/       # Downloaded Notion images
│       ├── wechat/       # Downloaded WeChat images
│       └── others/       # Downloaded images from other platforms
├── docs/
│   ├── architecture.md   # Architecture and design decisions
│   └── ci-workflow.md # CI/CD workflow documentation
└── tests/
    ├── unit/             # Unit tests (Vitest)
    ├── integration/      # Integration tests
    └── e2e/              # End-to-end tests (Playwright)
```

### Key Directories Explained

- **`src/lib/`**: Runtime business logic organized by domain (slug, content, markdown, site, ui). **No `src/utils/`** — each module has a clear, single responsibility.
- **`scripts/`**: Content acquisition scripts. Run independently of Astro. **No `scripts/lib/`** — shared utilities live in a single `utils.ts` file.
- **`src/config/`**: Centralized configuration (paths, site URL, feature flags). Used by both scripts and runtime.
- **`src/content/blog/`**: All blog posts. Subdirectories indicate content source (notion, wechat, others, or root for local files).

---

## 🚀 Quick Start

### Prerequisites

- **Node.js 22+**
- **Notion account** (if using Notion sync)

### Setup

1. **Clone and Install**

   ```bash
   git clone <repository-url>
   cd blog
   npm install
   ```

2. **Configure Environment Variables**

   Copy `.env.local.example` to `.env.local`:

   ```bash
   cp .env.local.example .env.local
   ```

   Fill in Notion credentials (if using Notion sync):

   ```env
   NOTION_TOKEN=secret_your_token_here
   NOTION_DATABASE_ID=your_database_id_here
   ```

   - **Token**: Create an integration at [Notion Integrations](https://www.notion.so/my-integrations)
   - **Database ID**: Found in your Notion database URL (32-character string after `notion.so/`)
   - **Connect Integration**: In Notion database, click `...` → `Connect to` → select your integration

3. **Start Development Server**

   ```bash
   npm run dev
   ```

   Opens at `http://localhost:4321/blog/`

---

## ✍️ Content Workflows

### Workflow 1: Notion → Blog

**Use Case**: Write articles in Notion, sync to blog as Markdown

**Steps:**

1. Create or update pages in your Notion database
2. Set page status to **"Published"** (supports both `select` and `status` property types)
3. Run sync command:

   ```bash
   npm run notion:sync
   ```

**What Happens:**

- Fetches all Published pages from Notion API
- Converts pages to Markdown using `notion-to-md`
- Downloads cover images and inline images to `public/images/notion/<pageId>/`
- Generates URL-safe slugs from titles (detects conflicts)
- Writes Markdown files to `src/content/blog/notion/`
- Fixes math formatting (removes spaces in `$ x $` → `$x$`)
- Runs linting/formatting

**Output:**

- `src/content/blog/notion/<slug>.md`
- `public/images/notion/<pageId>/*.{jpg,png,webp}`

**Idempotency**: Safe to run multiple times. Existing Notion files are overwritten; other content sources are untouched.

⚠️ **Important**: Do not manually edit files in `src/content/blog/notion/` — changes will be overwritten on next sync. Edit in Notion instead.

---

### Workflow 2: External URL → Blog

**Use Case**: Import articles from WeChat, Zhihu, Medium, etc.

**Command:**

```bash
npm run import:content -- --url="<article-url>"
```

**Examples:**

```bash
# WeChat article
npm run import:content -- --url="https://mp.weixin.qq.com/s/Pe5rITX7srkWOoVHTtT4yw"

# Zhihu article
npm run import:content -- --url="https://zhuanlan.zhihu.com/p/123456789"

# Medium article
npm run import:content -- --url="https://medium.com/@author/article-slug"
```

**Optional Flags:**

- `--overwrite`: Allow overwriting existing article with same slug
- `--preview`: Show extracted content without saving
- `--cover-first-image`: Use first image as cover if no cover found

**What Happens:**

- Detects platform (WeChat, Zhihu, Medium) from URL
- Launches headless browser (Playwright) to scrape content
- Downloads all images to `public/images/<platform>/<slug>/`
- Converts HTML to Markdown using unified/remark/rehype
- Generates frontmatter (title, date, author, cover)
- Writes MDX file to `src/content/blog/<platform>/`
- Fixes math formatting and runs linting

**Output:**

- `src/content/blog/<platform>/<slug>.mdx`
- `public/images/<platform>/<slug>/*.{jpg,png,webp}`

**Platform-Specific Notes:**

- **WeChat**: Handles image placeholders, retries failed downloads, uses browser fallback for stubborn images
- **Zhihu**: Extracts author and publish date from page metadata
- **Medium**: Similar extraction with platform-specific DOM selectors

⚠️ **Important**: Imported articles should be edited in their original platform or locally (if overwritten with `--overwrite`). Re-importing overwrites local changes unless `--overwrite` is omitted.

---

### Workflow 3: Local Markdown

**Use Case**: Write articles directly in the repository

**Steps:**

1. Create a `.md` or `.mdx` file in `src/content/blog/` (not in a subdirectory)
2. Add required frontmatter:

   ```yaml
   ---
   title: Your Article Title
   date: 2025-01-15
   status: published # or draft
   cover: /blog/images/your-cover.png # optional
   ---
   ```

3. Write content using Markdown
4. Build or dev to see changes

**Output:**

- Article appears at `/blog/<filename>/` (filename becomes slug)

**Benefits:**

- Full control over content and metadata
- Git-tracked changes
- No external dependencies
- Coexists with Notion and imported content

---

## 🛠️ Scripts Reference

All scripts are defined in `package.json` and run via `npm run <script>`.

> **完整的 Scripts 使用说明**：所有脚本的详细功能、参数、使用场景，请参见 **[scripts/README.md](scripts/README.md)**

| Script                | Command                                   | Description                                               |
| --------------------- | ----------------------------------------- | --------------------------------------------------------- |
| **Development**       |
| `dev`                 | `astro dev`                               | Start development server at `http://localhost:4321/blog/` |
| `start`               | `astro dev`                               | Alias for `dev`                                           |
| **Building**          |
| `build`               | `astro build`                             | Build static site to `dist/`                              |
| `preview`             | `astro preview`                           | Preview production build locally                          |
| **Content Sync**      |
| `notion:sync`         | `tsx scripts/notion-sync.ts && ...`       | Sync Notion pages, fix formatting, run linting            |
| `import:content`      | `tsx scripts/content-import.ts && ...`    | Import from URL, fix formatting, run linting              |
| `delete:article`      | `tsx scripts/delete-article.ts`           | Delete article and optionally associated images           |
| **Quality Assurance** |
| `check`               | `astro check`                             | TypeScript and Astro component validation                 |
| `lint`                | `npm run format:check && npm run md:lint` | Format and lint all files (auto-fixes)                    |
| `format:check`        | `prettier --check --write ...`            | Format code and Markdown                                  |
| `md:lint`             | `markdownlint-cli2`                       | Lint Markdown files                                       |
| `test`                | `vitest run --coverage`                   | Run unit tests with coverage                              |
| `test:watch`          | `vitest watch`                            | Run tests in watch mode                                   |
| `test:e2e`            | Build and run Playwright tests            | End-to-end browser tests                                  |
| `test:ci`             | All quality checks + build                | Full CI validation pipeline                               |

### Quick Examples

```bash
# Sync Notion content
npm run notion:sync

# Import article from URL
npm run import:content -- --url="<article-url>"

# Delete article by slug
npm run delete:article -- --target=<article-slug>

# Run all quality checks
npm run check && npm run lint && npm run test
```

**详细参数与用法**：参见 [scripts/README.md](scripts/README.md)

---

## 🧪 Development & Quality Assurance

### Local Development

```bash
npm run dev
```

- Hot module replacement (HMR) for fast development
- Available at `http://localhost:4321/blog/`
- Changes to `src/` rebuild automatically

### Type Checking

```bash
npm run check
```

Validates TypeScript types and Astro component props using `@astrojs/check`.

### Linting & Formatting

```bash
npm run lint
```

Runs:

1. **Prettier** on all code and Markdown (auto-fixes formatting)
2. **Markdownlint** on all Markdown files (enforces style rules)

⚠️ **Note**: This command **modifies files** to fix issues. Recommended to run before committing.

### Testing

#### Unit Tests (Vitest)

```bash
npm run test          # Run once with coverage
npm run test:watch    # Watch mode for TDD
```

- Tests in `tests/unit/`
- Coverage report in `coverage/`
- Tests for `src/lib/` modules (slug, content, markdown plugins, etc.)

#### End-to-End Tests (Playwright)

```bash
npm run test:e2e
```

- Tests in `tests/e2e/`
- Builds site first, then runs browser tests
- **First-time setup**: `npx playwright install --with-deps chromium` (CI does this automatically)

### Pre-Merge Quality Gate

**All PRs must pass these checks before merging to `main`:**

```bash
npm run check    # ✓ TypeScript types valid
npm run lint     # ✓ Code formatted, Markdown linted
npm run test     # ✓ Unit tests pass
npm run test:e2e # ✓ E2E tests pass
npm run build    # ✓ Site builds successfully
```

These are enforced by `.github/workflows/validation.yml`.

---

## 📚 Documentation

- **[架构文档 (Architecture Guide)](docs/architecture.md)**: 仓库架构、模块职责、设计规范的详细说明
- **[CI / Workflow 文档](docs/ci-workflow.md)**: GitHub Actions 工作流的完整说明与触发关系
- **[Scripts 使用说明](scripts/README.md)**: 所有脚本的功能、参数、使用场景的权威文档

**推荐阅读顺序**：

1. **新用户**：先看本 README，了解项目概况和快速开始
2. **开发者**：阅读 [`docs/architecture.md`](docs/architecture.md) 理解架构设计与开发规范
3. **内容管理者**：阅读 [`scripts/README.md`](scripts/README.md) 学习如何使用脚本同步和导入内容
4. **CI 维护者**：阅读 [`docs/ci-workflow.md`](docs/ci-workflow.md) 了解工作流配置与触发逻辑

---

## 🔗 CI/CD Workflows

This repository uses GitHub Actions for continuous integration and deployment:

- **`validation.yml`**: Runs on all PRs and pushes (check, lint, test, build, E2E)
- **`deploy.yml`**: Deploys to GitHub Pages on merge to `main`
- **`sync-notion.yml`**: Scheduled Notion sync (creates PR with updated content)
- **`import-content.yml`**: Manual workflow to import articles from URLs (creates PR)
- **`delete-article.yml`**: Manual workflow to delete articles (creates PR)
- **`post-deploy-smoke-test.yml`**: Verifies live site after deployment
- **`link-check.yml`**: Checks for broken links
- **`pr-preview.yml`**: Deploys PR preview to GitHub Pages

See [CI Workflow Map](docs/ci-workflow.md) for detailed workflow relationships and permissions.

---

## 📄 License

This project is licensed under the [ISC License](LICENSE). Free to use and modify within license terms.

---

## 🙏 Contributing

Contributions are welcome! Please:

1. Read [`docs/architecture.md`](docs/architecture.md) to understand design principles
2. Run quality checks before submitting PR:
   ```bash
   npm run check && npm run lint && npm run test && npm run test:e2e
   ```
3. Follow existing code organization (domain-driven modules, no util directories)
4. Update documentation if adding new features

**Questions?** Open an issue or discussion.
