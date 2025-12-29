# Scripts Directory

This directory contains various utility scripts for content management and maintenance.

## Available Scripts

### `notion-sync.ts`
Syncs content from Notion database to the blog.

**Usage:**
```bash
npm run notion:sync
```

**What it does:**
- Fetches published pages from Notion database
- Downloads images and saves them locally
- Converts Notion content to Markdown
- Handles slug management and conflict resolution
- Performs incremental sync (only updates changed content)

**Environment variables required:**
- `NOTION_TOKEN`: Notion API token
- `NOTION_DATABASE_ID`: Notion database ID

### `content-import.ts`
Imports articles from external platforms (WeChat, Zhihu, Medium, etc.).

**Usage:**
```bash
npm run import:content -- --url=<URL>
```

**Options:**
- `--url=<URL>`: URL of the article to import (required)
- `--allow-overwrite`: Overwrite existing files
- `--dry-run`: Preview without writing files
- `--use-first-image-as-cover`: Use the first image as the cover

**Supported platforms:**
- WeChat (mp.weixin.qq.com)
- Zhihu (zhihu.com)
- Medium (medium.com)
- Generic HTML articles (fallback)

### `fix-math.ts`
Fixes math delimiters in Markdown files.

**Usage:**
```bash
npm run fix-math <file-or-directory>
```

**What it does:**
- Normalizes invisible characters from Notion exports
- Fixes inline math delimiters ($...$)
- Promotes multi-line inline math to block math ($$...$$)
- Trims whitespace from math expressions

### `delete-article.ts`
Deletes an article and its associated images.

**Usage:**
```bash
npm run delete:article
```

(Interactive prompt will ask for the article slug)

## Shared Utilities (`utils.ts`)

The `scripts/utils.ts` module provides shared utilities for all scripts:

### Directory & File I/O
- `ensureDir(dir)`: Ensure a directory exists
- `processFile(filePath, processFn)`: Process a single file
- `processDirectory(dirPath, filterFn, processFn)`: Recursively process files in a directory

### Error Handling
- `runMain(mainFn)`: Run an async function with error handling and proper exit codes

### Math Delimiter Fixing
- `fixMath(text)`: Fix math delimiters in markdown
- `normalizeInvisibleCharacters(text)`: Normalize invisible Unicode characters
- `splitCodeFences(text)`: Split markdown into segments (frontmatter, code, text)

## Important Notes

### Script Utilities vs Runtime Code

**The utilities in `scripts/utils.ts` are ONLY for scripts.**

- ✅ Scripts can import from `./utils`
- ❌ Runtime code (`src/`) should NOT import from `scripts/utils.ts`
- ✅ Runtime code should use utilities from `src/lib/` or `src/config/`

### Path Configuration

Scripts import path configuration from `src/config/paths.ts`:
- `NOTION_CONTENT_DIR`: Where Notion articles are saved
- `NOTION_PUBLIC_IMG_DIR`: Where Notion images are saved
- `BLOG_CONTENT_DIR`: Blog content root
- `PUBLIC_IMAGES_DIR`: Public images root
- `ARTIFACTS_DIR`: Debug artifacts for troubleshooting

### Adding New Scripts

When creating new scripts:

1. Import utilities from `./utils` for common operations
2. Import paths from `../src/config/paths`
3. Import slug utilities from `../src/lib/slug`
4. Use `runMain()` wrapper for proper error handling
5. Document the script in this README

### Best Practices

- **Keep scripts focused**: Each script should do one thing well
- **Reuse utilities**: Use `scripts/utils.ts` for common operations
- **Don't mix concerns**: Keep script utilities separate from runtime code
- **Document behavior**: Add comments for complex logic
- **Test changes**: Run the script after modifications to ensure it works
