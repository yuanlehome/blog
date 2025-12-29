# 架构与模块边界（Astro 博客）

## 目录划分

- `src/config/`：统一配置入口
  - `paths.ts`：仓库根路径、内容目录、公共图片目录、构建输出、调试产物目录等（支持环境变量覆盖）。
  - `site.ts`：站点 `site/base` 元信息。
  - `env.ts` / `features.ts`：环境变量读取与特性开关的布尔/数字解析。
- `src/lib/`：纯逻辑与可复用模块（无 Astro 依赖）
  - `content/`：文章读取、日期/slug/toc、阅读时长等内容域工具。
  - `markdown/`：remark/rehype 插件及代码高亮定制。
  - `ui/`：纯前端交互逻辑（如代码块复制、浮动按钮栈计算）。
  - `site/`：站点级工具（如 `assetUrl`）。
- `src/components/` & `src/layouts/`：UI 组件与页面骨架，仅依赖 `lib`/`config`。
- `src/pages/`：Astro 路由页面，负责组合组件与数据。
- `scripts/`：命令入口（content import / notion sync / delete 等），路径等配置统一来自 `src/config`。
- `tests/`：单测、集成与 e2e，依赖 `lib`/`config` 暴露的接口。

依赖方向约束：`config` → `lib` → `components/layouts/pages` → `scripts`，避免反向耦合。

## 配置与环境

- 路径相关：`src/config/paths.ts` 暴露 `ROOT_DIR`、`BLOG_CONTENT_DIR`、`NOTION_CONTENT_DIR`、`PUBLIC_IMAGES_DIR`、`NOTION_PUBLIC_IMG_DIR`、`ARTIFACTS_DIR` 等，统一用于脚本与工具函数。
- 站点元信息：`src/config/site.ts`（`siteBase`、`siteUrl`）。
- 特性开关：`src/config/features.ts`，通过 `FEATURE_*` 环境变量控制。
- 习惯用法：在新模块中优先使用 `config` 提供的路径/开关，避免硬编码。

## 脚本与工具链

- 入口位于 `scripts/`，仅做参数解析与调用，核心逻辑依赖 `src/config` 路径。
- 主要命令：
  - `npm run import:content` → `scripts/content-import.ts`（外部文章抓取、图片下载、MDX 生成）。
  - `npm run notion:sync` → `scripts/notion-sync.ts`（Notion 同步、封面迁移、slug 冲突处理）。
  - `npm run delete:article` → `scripts/delete-article.ts`（文章与资源清理）。
  - `scripts/process-md-files.ts` 为辅助格式化，可被上述命令复用。
- 参数解析：兼容 `--flag=value` 与 `--flag value`，并读取对应的环境变量兜底。

## 工作流概览

- `validation.yml`：PR/Main 验证（check/lint/unit/build/e2e）并带缓存+并发控制。
- `import-content.yml` / `sync-notion.yml`：内容导入/同步，跑脚本后发起 PR。
- `deploy.yml`：主干部署。
- 复用：Node 安装、Playwright 缓存、并发组均保持一致；权限最小化。

## 常见开发任务

- **新增工具函数**：放入 `src/lib/{domain}`，如与内容相关则置于 `content/`；需要页面使用时从 `src/utils/*` 或组件直接导入。
- **修改站点/路径配置**：更新 `src/config/site.ts` 或 `src/config/paths.ts`，避免在业务代码中硬编码。
- **新增 importer/脚本**：在 `scripts/` 创建薄入口，核心逻辑放入 `src/lib`/`src/tooling`，参数解析复用 `config`。
- **Workflow 调整**：将公共步骤提炼为复用 action/workflow_call，保持缓存 key 与并发前缀一致。
- **调试内容/目录变更**：使用 `BLOG_CONTENT_DIR`、`PUBLIC_IMAGES_DIR` 等环境变量快速切换输出位置，测试前清理或重建相关目录即可。
