# scripts 目录说明

## 核心脚本

- **`notion-sync.ts`**
  - 用法：`npm run notion:sync`（`--help` 查看参数）
  - 环境：`NOTION_TOKEN`、`NOTION_DATABASE_ID` 必填。
  - 作用：拉取 Published 页面，生成 Markdown，下载封面/图片到 `public/images/notion/<slug>/`，最后自动运行 `npm run lint`。

- **`content-import.ts`**
  - 用法：`npm run import:content -- --url=<URL> [--allow-overwrite] [--dry-run] [--use-first-image-as-cover]`，可用环境变量 `URL`/`ALLOW_OVERWRITE`/`DRY_RUN`/`USE_FIRST_IMAGE_AS_COVER`。
  - 支持平台：微信、知乎、Medium 及通用 HTML。
  - 作用：Playwright 抓取 → 本地化图片 → 生成 `src/content/blog/<provider>/<slug>.md` 与 `public/images/<provider>/<slug>/`。

- **`delete-article.ts`**
  - 用法：`npm run delete:article -- --target=<slug|path> [--delete-images] [--dry-run]`（亦可用环境变量 `TARGET`、`DELETE_IMAGES`、`DRY_RUN`）。
  - 作用：删除文章文件；可选删除匹配图片目录；dry-run 仅打印。

- **`process-md-files.ts`**
  - 用法：`npx tsx scripts/process-md-files.ts <file-or-directory>`；`--help` 查看说明。
  - 作用：清理不可见字符、修正公式分隔符，原地修改文件。

## 约束与约定

- `scripts/utils.ts` 只为脚本复用，运行态禁止引用；共享常量只来自 `src/config/paths` 与 `src/lib/slug`。
- 脚本均提供 `--help`/`-h`，帮助文案与解析逻辑同源，文档需以此为准。
- 内容与图片目录按来源分层：`src/content/blog/<provider>/`，`public/images/<provider>/<slug>/`。
- 变更脚本时请跑 `npm run test` 或至少对应的子测试，避免破坏采集链路。
