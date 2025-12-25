# Astro + Notion Static Blog

A modern, fast static blog powered by **Astro** and **Notion**.
It supports writing in both **Notion** (CMS-style) and **Markdown** (Git-style).

## 🚀 Quick Start

### 1. Prerequisites
- Node.js 18+
- A Notion Account
- A GitHub Account

### 2. Setup

1. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   cd blog
   npm install
   ```

2. **Configure Environment**:
   Copy `.env.local.example` to `.env.local` and fill in your keys:
   ```ini
   NOTION_TOKEN=secret_...
   NOTION_DATABASE_ID=...
   ```
   *   **Token**: [Create Integration](https://www.notion.so/my-integrations).
   *   **Database ID**: From your Notion Database URL.
   *   **Important**: Share your Notion Database with the integration (Top right `...` > `Connect to` > Your Integration).

3. **Start Development**:
   ```bash
   npm run dev
   ```

## ✍️ Writing Content

### Option A: Notion (CMS)
1. Write your post in Notion.
2. Set status to **Published**.
3. **Sync**:
   - Locally: `npm run notion:sync`
   - GitHub: Go to Actions > "Sync Notion Content" > Run workflow.
   - *Note*: This runs automatically every day at 00:00 UTC.

### Option B: Local Markdown
1. Create a file in `src/content/blog/local/my-post.md`.
2. Add frontmatter:
   ```yaml
   ---
   title: "My Post"
   date: 2023-10-01
   status: "published"
   ---
   ```
3. Commit and push.

## 🧮 Math Support (LaTeX)

This blog supports math rendering via KaTeX.

- **Inline Math**: Use `$ E=mc^2 $`.
- **Block Math**:
  ```latex
  $$
  \sum_{i=0}^n i^2 = \frac{(n^2+n)(2n+1)}{6}
  $$
  ```

**Automatic Math Fixer**:
The sync script automatically runs a fixer (`scripts/fix-math.ts`) that:
1. Trims whitespace in inline math (`$ x $` -> `$x$`).
2. Promotes multi-line inline math to block math.

To run it manually on a specific file:
```bash
npx tsx scripts/fix-math.ts src/content/blog/local/my-post.md
```

## 🛠 Commands

| Command | Description |
| :--- | :--- |
| `npm run dev` | Start local dev server at `localhost:4321` |
| `npm run build` | Build for production |
| `npm run notion:sync` | Fetch Notion posts, download images, and fix math |
| `npm run preview` | Preview the built site |

## 📂 Project Structure

```
├── .github/workflows/    # CI/CD (Deploy & Sync)
├── public/images/notion/ # Synced Notion images
├── scripts/
│   ├── notion-sync.ts    # Notion -> Markdown converter
│   └── fix-math.ts       # Math syntax corrector
├── src/
│   ├── content/blog/
│   │   ├── local/        # Your manual markdown files
│   │   └── notion/       # Auto-generated from Notion (don't edit)
│   ├── pages/            # Routes (index, about, [...slug])
│   └── layouts/          # Base HTML layouts
```
