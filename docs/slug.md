# Slug Specification

**Module**: `src/lib/slug/`  
**Purpose**: Single source of truth for all slug generation and URL construction

---

## Overview

This document defines how slugs are generated, normalized, and used throughout the blog system. All slug-related operations **must** go through the `src/lib/slug` module to ensure consistency across:

- Notion synchronization (`scripts/notion-sync.ts`)
- URL imports (`scripts/content-import.ts`)
- Local markdown files (filename-based routing)
- Post URL construction (Astro pages/components)

---

## Core Functions

### `normalizeSlug(value: string): string`

Converts any string into a valid URL slug.

**Rules**:

- Lowercase conversion
- Non-Latin characters (Chinese, emoji, etc.) are **removed** (not transliterated)
- Special characters (`@#$%` etc.) may be converted to words (e.g., `$` → `dollar`)
- Underscores are removed (not converted to hyphens)
- Hyphens are preserved
- Consecutive spaces/hyphens are collapsed
- Leading/trailing special chars are trimmed

**Examples**:

```javascript
normalizeSlug('Hello World'); // 'hello-world'
normalizeSlug('你好 World'); // 'world' (Chinese removed)
normalizeSlug('Test_File_Name'); // 'testfilename' (underscores removed)
normalizeSlug('Price: $100'); // 'price-dollar100'
normalizeSlug('2024-01-01-post'); // '2024-01-01-post' (hyphens preserved)
normalizeSlug('!!!Important!!!'); // 'important'
```

**Why Chinese is removed**: The underlying `slugify` library with `strict: true` ensures URL-safe ASCII-only slugs. For Chinese-titled content, use explicit slugs or rely on fallback IDs.

---

### `slugFromTitle(options): string`

Generates slug from title with fallback priority.

**Priority Order**:

1. `explicitSlug` (if provided) - direct override
2. `title` (if provided) - normalized to slug
3. `fallbackId` (if provided) - used when title is empty or non-Latin

**Parameters**:

```typescript
{
  explicitSlug?: string | null;  // Explicit slug override
  title?: string;                // Post title
  fallbackId?: string;           // Fallback identifier (e.g., Notion page ID)
}
```

**Examples**:

```javascript
// Priority 1: Explicit slug
slugFromTitle({
  explicitSlug: 'custom-slug',
  title: 'Ignored',
  fallbackId: 'id-123',
});
// → 'custom-slug'

// Priority 2: Title
slugFromTitle({
  title: 'Inside NVIDIA GPUs',
  fallbackId: 'page-456',
});
// → 'inside-nvidia-gpus'

// Priority 3: Fallback ID
slugFromTitle({
  title: '你好世界', // Pure Chinese, normalizes to ''
  fallbackId: 'notion-abc123',
});
// → 'notion-abc123'
```

---

### `slugFromFileStem(stem: string): string`

Converts filename (without extension) to slug. Ensures compatibility with Astro's default routing behavior where `hello-world.md` becomes `/hello-world/`.

**Examples**:

```javascript
slugFromFileStem('hello-world'); // 'hello-world'
slugFromFileStem('2024-01-01-post'); // '2024-01-01-post'
slugFromFileStem('My File Name'); // 'my-file-name'
```

**Usage**: Apply this to local markdown files in `src/content/blog/` to ensure consistent slug behavior.

---

### `ensureUniqueSlug(desired, ownerId, used): string`

Ensures slug uniqueness by adding hash suffixes on conflicts.

**Conflict Resolution**:

1. If slug is available → return as-is
2. If slug is used by same owner → return as-is (idempotent)
3. If slug conflicts → try `slug-{hash}` (6-char hash of ownerId)
4. If still conflicts → try `slug-{hash}-2`, `slug-{hash}-3`, etc.

**Parameters**:

- `desired`: Desired slug
- `ownerId`: Unique identifier for content (e.g., Notion page ID)
- `used`: Map of `slug → ownerId` tracking existing content

**Examples**:

```javascript
const used = new Map();

// First post
ensureUniqueSlug('post', 'id-1', used); // → 'post'

// Conflict!
ensureUniqueSlug('post', 'id-2', used); // → 'post-a1b2c3' (hash of id-2)

// Same owner, no conflict
ensureUniqueSlug('post', 'id-1', used); // → 'post' (unchanged)
```

**Why hash suffix?**: Provides stable, predictable slugs for the same content across syncs, even if titles collide.

---

### `ensureUniqueSlugs(items): Map<string, string>`

Batch version of `ensureUniqueSlug` for processing multiple items at once.

**Parameters**:

```typescript
items: Array<{ id: string; slug: string }>;
```

**Returns**: `Map<id, uniqueSlug>`

**Example**:

```javascript
const items = [
  { id: 'notion-1', slug: 'my-post' },
  { id: 'notion-2', slug: 'my-post' }, // Conflict!
  { id: 'notion-3', slug: 'other' },
];

const result = ensureUniqueSlugs(items);
// Map {
//   'notion-1' => 'my-post',
//   'notion-2' => 'my-post-abc123',
//   'notion-3' => 'other'
// }
```

---

### `buildPostUrl(slug, base?): string`

Constructs full post URL with proper BASE_URL handling.

**Rules**:

- Automatically adds trailing slashes
- Handles root (`/`) and subpath (`/blog/`) bases
- Ensures no double slashes

**Parameters**:

- `slug`: Post slug
- `base`: Optional base URL override (defaults to `siteBase` from config)

**Examples**:

```javascript
// Default base: /blog/
buildPostUrl('my-post'); // '/blog/my-post/'
buildPostUrl('my-post', '/'); // '/my-post/'
buildPostUrl('my-post', '/docs'); // '/docs/my-post/'

// Trailing slash normalization
buildPostUrl('my-post/', '/blog'); // '/blog/my-post/'
buildPostUrl('my-post', '/blog/'); // '/blog/my-post/'
```

**Usage**: **ONLY** function for constructing post URLs. Replaces all instances of `${BASE}${post.slug}/` in Astro files.

---

## Slug Sources

### 1. Notion Sync

**Script**: `scripts/notion-sync.ts`

**Slug Priority**:

1. Notion property `slug` or `Slug` (if set)
2. Page title (via `slugFromTitle`)
3. Notion page ID (fallback)

**Output**:

- Markdown: `src/content/blog/notion/{slug}.md`
- Images: `public/images/notion/{slug}/`

**Conflict Handling**: `ensureUniqueSlug` with Notion page ID as owner

**Example Flow**:

```typescript
const title = 'My Notion Post';
const pageId = 'abc123-def456';
const propSlug = null; // No explicit slug set

const baseSlug = slugFromTitle({ title, fallbackId: pageId });
// → 'my-notion-post'

const slug = ensureUniqueSlug(baseSlug, pageId, usedSlugs);
// → 'my-notion-post' (if available)
//   OR 'my-notion-post-a1b2c3' (if conflict)
```

---

### 2. URL Import

**Script**: `scripts/content-import.ts`

**Slug Priority**:

1. Article title (via `slugFromTitle`)
2. URL path component (fallback)
3. `{provider}-{timestamp}` (last resort)

**Output**:

- Markdown: `src/content/blog/{provider}/{slug}.md`
- Images: `public/images/{provider}/{slug}/`

**Example Flow**:

```typescript
const title = 'Inside NVIDIA GPUs: Anatomy of High-Performance MatMul Kernels';
const url = 'https://example.com/posts/nvidia-gpus';

const urlPath = 'nvidia-gpus'; // Extracted from URL
const slug = slugFromTitle({ title, fallbackId: urlPath });
// → 'inside-nvidia-gpus-anatomy-of-high-performance-matmul-kernels'

// If title were pure Chinese:
const cnTitle = '英伟达 GPU 剖析';
const cnSlug = slugFromTitle({ title: cnTitle, fallbackId: urlPath });
// → 'nvidia-gpus' (fallback to URL path)
```

---

### 3. Local Markdown

**Location**: `src/content/blog/*.md` (or subdirectories)

**Slug Source**: Filename stem (without extension)

**Routing**: Handled by Astro's content collections

- `hello-world.md` → `/hello-world/`
- `2024-01-01-post.md` → `/2024-01-01-post/`

**Best Practice**: Use `slugFromFileStem` when programmatically generating local files to ensure consistency.

**Example**:

```typescript
const filename = '2024-12-28-post.md';
const stem = path.basename(filename, '.md');
const slug = slugFromFileStem(stem);
// → '2024-12-28-post'

// Astro will route this to: /2024-12-28-post/
```

---

## Common Issues & Solutions

### Issue: Chinese-titled posts get fallback slugs

**Cause**: `slugify` with `strict: true` removes non-Latin characters for URL safety

**Solution**:

1. **Preferred**: Set explicit slug in Notion property
2. **Fallback**: Use Notion page ID or URL path component

**Example**:

```typescript
// Notion page: title="深度学习入门", slug property not set
slugFromTitle({ title: '深度学习入门', fallbackId: 'page-123' });
// → 'page-123'

// With explicit slug:
slugFromTitle({
  explicitSlug: 'deep-learning-intro',
  title: '深度学习入门',
});
// → 'deep-learning-intro'
```

---

### Issue: Slug conflicts between sources

**Cause**: Title overlap or manual filename collision

**Detection**: `ensureUniqueSlug` or `ensureUniqueSlugs` adds hash suffix

**Example**:

```typescript
// Notion post: "My Post" (id: notion-123)
// Local file: my-post.md

// Notion sync:
ensureUniqueSlug('my-post', 'notion-123', used);
// → 'my-post-a1b2c3' (conflict with local file)
```

**Prevention**: Review existing slugs before sync, use explicit slugs in Notion

---

### Issue: Underscores in filenames

**Cause**: `slugify` removes underscores

**Example**:

```javascript
slugFromFileStem('my_file_name'); // → 'myfilename' (unexpected!)
```

**Solution**: Use hyphens in filenames: `my-file-name.md`

---

### Issue: Special characters in imported titles

**Cause**: Some special chars convert to words

**Example**:

```javascript
slugFromTitle({ title: 'Price: $100' });
// → 'price-dollar100'
```

**Solution**: Usually acceptable; if not, set explicit slug during import

---

## URL Construction

### Bad Practice ❌

```astro
<!-- Duplicated BASE_URL normalization -->
<script>
  const BASE = import.meta.env.BASE_URL.endsWith('/')
    ? import.meta.env.BASE_URL
    : `${import.meta.env.BASE_URL}/`;
</script>
<a href={`${BASE}${post.slug}/`}>Link</a>
```

### Good Practice ✅

```astro
---
import { buildPostUrl } from '@/lib/slug';
---

<a href={buildPostUrl(post.slug)}>Link</a>
```

**Benefits**:

- Single normalization point
- Handles all edge cases (root base, missing slashes, etc.)
- Consistent across entire codebase

---

## Testing

All slug functions are comprehensively tested in `tests/unit/slug-consolidated.test.ts` (46 tests):

- **normalizeSlug**: Chinese, emoji, special chars, edge cases
- **slugFromTitle**: Priority order, fallback behavior
- **slugFromFileStem**: Filename conventions
- **ensureUniqueSlug**: Conflict resolution, idempotence
- **ensureUniqueSlugs**: Batch processing
- **buildPostUrl**: BASE_URL handling, trailing slashes
- **Integration tests**: Full workflows (Notion, import, local)

---

## Migration Checklist

If you're updating code to use the slug module:

- [ ] Replace direct `slugify()` calls with `slugFromTitle()`
- [ ] Replace manual BASE_URL normalization with `buildPostUrl()`
- [ ] Use `ensureUniqueSlug` for single items, `ensureUniqueSlugs` for batches
- [ ] Import from `src/lib/slug` (not `scripts/slug`)
- [ ] Add tests for slug-related logic
- [ ] Verify no duplicate slug generation code remains

---

## Future Considerations

1. **Transliteration**: Consider adding Chinese → Pinyin transliteration as an option (requires additional library)
2. **Custom slugify options**: Allow per-source customization of `slugify` behavior if needed
3. **Slug aliases**: Support multiple slugs pointing to same content (redirects)
4. **Slug history**: Track slug changes for 301 redirects

---

**Last Updated**: 2025-12-28  
**Module Version**: 1.0.0
