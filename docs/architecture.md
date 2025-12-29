# 架构说明

## 1. 总体架构图（ASCII）

```text
内容源 (Notion / 外部 URL / 本地 Markdown)
          │
          v
scripts 层 (notion-sync / content-import / process-md-files / delete-article)
          │ 写入
          v
内容产物 (src/content/blog/* 与 public/images/*)
          │ 只读
          v
Astro runtime (src/lib + 组件/页面)
          │ astro build
          v
静态站点输出 dist/
```

## 2. 分层与边界

- **Runtime：`src/`**
  - `src/lib` 是运行时域逻辑与 slug 入口；组件/页面只消费 `src/content` 数据。
  - 没有 `src/utils`，公共逻辑按领域分布。
- **Content Artifacts：`src/content` + `public/images`**
  - 均视为数据产物；`notion/`、`wechat/`、`others/` 等目录由脚本写入。
  - 运行态只读不写，构建期间按 frontmatter 生成页面。
- **Scripts：`scripts/`**
  - `scripts/utils.ts` 仅供脚本复用，禁止运行态引用。
  - CLI 入口：`notion-sync.ts`、`content-import.ts`、`delete-article.ts`，`process-md-files.ts` 负责内容清洗。

## 3. 关键约定

- slug 统一由 `src/lib/slug` 生成与去重，脚本与 runtime 共用。
- 图片路径：`public/images/<provider>/<slug>/`；Notion 重命名会搬迁旧目录。
- Notion 同步仅拉取 Published，若 lastEdited 未变会跳过；冲突 slug 会带哈希。
- URL 导入输出 `src/content/blog/<provider>/<slug>.md`，默认拒绝覆盖，`--allow-overwrite` 才会重写。
- 导入与同步都会在末尾运行 `process-md-files` + `npm run lint` 以修正公式与格式。
- 本地 Markdown 至少包含 `title`、`date`、`status` frontmatter，文件名即访问 slug。
- 删除脚本按 `--target` 定位文章，可用 `--delete-images` 清理匹配的图片目录。
- 运行态绝不调用脚本，也不访问外部网络；脚本只通过 `src/config/paths`、`src/lib/slug` 共享常量。

## 4. 我们刻意不做的事

- 不在运行态请求 Notion 或外网；所有抓取前置到 scripts。
- 不在 runtime 引用 `scripts/`，也不把脚本工具挪到 `src/utils`。
- 不为未来来源做抽象框架；有新来源按现有模式落盘。
- 不手改 `src/content/blog/notion/` 与对应图片目录。
- 不跳过 slug 冲突与覆盖提示，必须显式传入覆盖参数。
