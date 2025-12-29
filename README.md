# Astro 静态博客

## 1. 项目是什么

- 基于 Astro 的静态博客，内容源自 Notion、外部 URL（微信/知乎/Medium 等）或本地 Markdown。
- 所有内容在构建前由 Node 脚本落盘，运行态只读取 `src/content` 与 `public/images`。
- slug 与 frontmatter 统一由 `src/lib/slug` 生成，图片路径固定在 `/images/<source>/<slug>/`。
- 设计准则：脚本负责采集与清洗，Astro 负责渲染，层间不交叉调用。

## 2. 目录速览

- `src/`：运行时代码与内容集合（含 `lib/`、`content/`、组件、页面）
- `scripts/`：采集脚本，含 `notion-sync.ts`、`content-import.ts`、`delete-article.ts`、`process-md-files.ts`
- `public/`：静态资源与已下载的图片（按来源分目录）
- `docs/`：架构说明与 CI 流程
- `tests/`：Vitest 与 Playwright 测试

## 3. 内容工作流

### Notion → Markdown

- 输入：Notion 数据库 Published 条目；需 `.env.local` 中 `NOTION_TOKEN`、`NOTION_DATABASE_ID`
- 命令：`npm run notion:sync`（`--help` 查看说明）
- 输出：`src/content/blog/notion/*.md`，图片存 `public/images/notion/<slug>/`
- 行为：生成 slug、覆盖同名文件、封面优先取属性/首图，最后自动执行 `npm run lint`
- 注意：手改 `notion/` 会被下一次同步覆盖

### 外部 URL 抓取

- 输入：文章 URL；必填参数 `--url=<URL>`（可用环境变量 URL），支持 `--allow-overwrite`、`--dry-run`、`--use-first-image-as-cover`
- 命令：`npm run import:content -- --url=<URL> [--allow-overwrite] [--dry-run] [--use-first-image-as-cover]`
- 输出：`src/content/blog/<provider>/<slug>.md`，图片落 `public/images/<provider>/<slug>/`
- 行为：Playwright 抓取 HTML → Markdown，本地化图片，frontmatter 写来源与时间，默认不覆盖已有文件
- 注意：WeChat 会防占位图，`--help` 可查看参数与示例

### 本地 Markdown

- 输入：在 `src/content/blog/` 新建 `.md`/`.mdx`（文件名即 slug）
- 可选：`npx tsx scripts/process-md-files.ts <path>` 修正常见公式/空白
- 输出：随 Astro 构建直接渲染；图片请放入 `public/images/<custom>/<slug>/`
- 注意：frontmatter 至少包含 `title`、`date`、`status`

## 4. 常用命令（与 package.json 一致）

- `npm run notion:sync`：同步 Notion；依赖 `NOTION_TOKEN`、`NOTION_DATABASE_ID`；`--help` 查看说明
- `npm run import:content -- --url=<URL> [--allow-overwrite] [--dry-run] [--use-first-image-as-cover]`：URL 导入；同名环境变量可替代参数
- `npm run delete:article -- --target=<slug|path> [--delete-images] [--dry-run]`：删除文章，可清理图片；支持环境变量 `TARGET`、`DELETE_IMAGES`、`DRY_RUN`
- `npm run dev` / `npm run build` / `npm run preview`：本地开发与预览
- `npm run check`、`npm run lint`、`npm run test`、`npm run test:e2e`：类型检查、格式校验、单测、E2E
- `npm run docs:verify`：校验 README 是否包含核心命令片段

## 5. 质量门禁

- 需要通过：`npm run check`
- 需要通过：`npm run lint`
- 需要通过：`npm run test`
- 需要通过：`npm run test:e2e`
- PR 合入要求：以上四项全绿，必要时附加 `npm run build` 产物检查

## 6. FAQ（短）

- **为什么没有 `src/utils`？** 逻辑按领域放在 `src/lib/*`，避免公用工具箱变成垃圾场。
- **scripts 能被运行时代码引用吗？** 不能，`scripts/` 只做采集；运行态只用 `src/lib` 与 `src/config`。
- **slug/路径规则在哪定义？** `src/lib/slug` 负责 slug，图片目录固定 `public/images/<source>/<slug>/`，内容入口统一在 `src/content/blog/`。
- **删除或覆盖内容的边界？** 同步会覆盖 `notion/`，导入默认不覆盖除非 `--allow-overwrite`，删除脚本支持 `--delete-images` 与 `--dry-run`。
- **为什么脚本有 `--help`？** 作为参数真相源，帮助文案与解析逻辑同源，避免文档写错。
