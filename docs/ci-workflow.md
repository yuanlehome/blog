# CI/Workflow 文档

本文档描述仓库中的 **GitHub Actions 工作流**，包括触发条件、职责和调用关系。

> **相关文档**
>
> - Scripts 详细说明 → [scripts/README.md](../scripts/README.md)
> - 仓库架构 → [architecture.md](./architecture.md)

---

## 一、Workflow 清单

| Workflow 文件                | 职责                    | 触发条件                              |
| ---------------------------- | ----------------------- | ------------------------------------- |
| `validation.yml`             | 质量门禁（含烟测）       | PR → main / Push → main               |
| `deploy.yml`                 | 构建并发布 GitHub Pages | Push → main / 手动                    |
| `sync-notion.yml`            | 同步 Notion 内容        | 每日 00:00 UTC / 手动                 |
| `import-content.yml`         | 导入外部文章/PDF        | 手动                                  |
| `delete-article.yml`         | 删除文章                | 手动                                  |
| `post-deploy-smoke-test.yml` | 部署后烟测              | `deploy.yml` 成功后                   |
| `link-check.yml`             | 链接有效性检查          | PR → main / Push → main / 每周一 03:00 UTC |
| `pr-preview.yml`             | PR 预览站点             | PR 打开/同步/关闭                      |
| `copilot-fix-posts.yml`      | Copilot 修复文章        | 手动                                  |

---

## 二、Workflow 详解

### 2.1 `validation.yml` — 质量门禁

**触发条件**：

- PR 指向 main 分支
- Push 到 main 分支

**执行内容**：

1. `validate` job：
   - 类型检查（`npm run check`）
   - Lint 与格式化（`npm run lint`）
   - 单元测试（`npm run test`，含覆盖率）
   - 构建验证（`npm run build`）
   - E2E 测试（`npm run test:e2e`，自动安装 chromium）

2. `smoke-test` job（依赖 validate）：
   - 启动静态服务器
   - 访问首页、文章页
   - 验证 Sitemap
   - 断言内容正确性

**权限**：`contents: read`

**并发控制**：同一 PR/分支的多次触发会取消旧的运行

---

### 2.2 `deploy.yml` — 部署

**触发条件**：

- Push 到 main 分支
- 手动触发（`workflow_dispatch`）

**执行内容**：

1. 检出代码、安装依赖
2. 运行 `npm run build`
3. 上传 `dist/` 作为 Pages artifact
4. 部署到 GitHub Pages（使用 actions/deploy-pages）

**权限**：`contents: read`、`pages: write`、`id-token: write`

**并发控制**：同一部署任务会取消旧的运行

---

### 2.3 `sync-notion.yml` — Notion 同步

**触发条件**：

- 定时：每日 00:00 UTC
- 手动触发（`workflow_dispatch`）

**输入参数（手动触发）**：

| 参数                          | 类型    | 默认值     | 说明                            |
| ----------------------------- | ------- | ---------- | ------------------------------- |
| `markdown_translate_enabled`  | boolean | `false`    | 启用 Markdown 翻译              |
| `markdown_translate_provider` | choice  | `identity` | 翻译提供商（identity/deepseek） |

**调用 Scripts**：`npm run notion:sync` → `scripts/notion-sync.ts`

**执行内容**：

1. 运行同步脚本
2. 检测变更
3. 如有变更，创建 PR（分支：`chore/sync-notion-<run_id>`）
4. 启用自动合并

**权限**：`contents: write`、`pull-requests: write`

**Secrets**：`NOTION_TOKEN`（必需）、`NOTION_DATABASE_ID`（必需）、`DEEPSEEK_API_KEY`（可选）

> **Scripts 参数详情** → [scripts/README.md](../scripts/README.md#21-notion-syncts)

---

### 2.4 `import-content.yml` — 导入外部文章

**触发条件**：手动触发（`workflow_dispatch`）

**输入参数**：

| 参数                          | 类型    | 默认值     | 说明                                            |
| ----------------------------- | ------- | ---------- | ----------------------------------------------- |
| `url`                         | string  | —          | 必填，文章 URL（支持知乎/微信/Medium/PDF/其他） |
| `allow_overwrite`             | boolean | `false`    | 覆盖已存在文章                                   |
| `dry_run`                     | boolean | `false`    | 预览模式（不写入文件）                           |
| `use_first_image_as_cover`    | boolean | `true`     | 首图作为封面                                     |
| `force_pdf`                   | boolean | `false`    | 强制 PDF 导入（绕过受限域名如 arXiv）            |
| `markdown_translate_enabled`  | boolean | `false`    | 启用翻译                                         |
| `markdown_translate_provider` | choice  | `deepseek` | 翻译提供商（identity/deepseek）                  |

**调用 Scripts**：`npm run import:content` → `scripts/content-import.ts`

**执行内容**：

1. 安装依赖和 Playwright
2. 运行导入脚本
3. 如有变更且非 dry-run，创建 PR（分支：`chore/import-content-<run_id>`）
4. 启用自动合并

**权限**：`contents: write`、`pull-requests: write`

**Secrets**：`DEEPSEEK_API_KEY`（翻译可选）、`PADDLEOCR_VL_TOKEN`（PDF 导入可选）、`PADDLEOCR_VL_API_URL`（PDF 导入可选）

> **Scripts 参数详情** → [scripts/README.md](../scripts/README.md#22-content-importts)

---

### 2.5 `delete-article.yml` — 删除文章

**触发条件**：手动触发（`workflow_dispatch`）

**输入参数**：

| 参数            | 类型    | 默认值  | 说明                  |
| --------------- | ------- | ------- | --------------------- |
| `target`        | string  | —       | 必填，slug 或文件路径 |
| `delete_images` | boolean | `false` | 删除关联图片          |
| `dry_run`       | boolean | `false` | 预览模式              |

**调用 Scripts**：`npm run delete:article` → `scripts/delete-article.ts`

**执行内容**：

1. 运行删除脚本
2. 如有变更且非 dry-run，创建 PR（分支：`chore/delete-article-<run_id>`）

**权限**：`contents: write`、`pull-requests: write`

> **Scripts 参数详情** → [scripts/README.md](../scripts/README.md#23-delete-articlets)

---

### 2.6 `post-deploy-smoke-test.yml` — 部署后烟测

**触发条件**：`deploy.yml` 成功完成后（`workflow_run`）

**执行内容**：

1. 等待 GitHub Pages 部署生效
2. 访问首页、文章页、Sitemap
3. 验证响应状态和内容

**权限**：`contents: read`

---

### 2.7 `link-check.yml` — 链接检查

**触发条件**：

- PR 指向 main
- Push 到 main
- 定时：每周一 03:00 UTC

**执行内容**：使用 lychee 检查 README、src、public、docs 中的链接

**权限**：`contents: read`

---

### 2.8 `pr-preview.yml` — PR 预览

**触发条件**：

- PR 打开/同步（新 push）
- PR 关闭（清理）

**执行内容**：

- **PR 打开/同步**：构建站点，部署到外部预览仓库，在 PR 中评论预览链接
- **PR 关闭**：清理预览分支

**权限**：`contents: write`、`pull-requests: write`

---

## 三、Workflow 依赖关系

```text
┌─────────────────┐     ┌─────────────────┐
│   PR → main     │────▶│  validation.yml │
│   Push → main   │     │  (validate +    │
└─────────────────┘     │   smoke-test)   │
                        └────────┬────────┘
                                 │
        ┌────────────────────────┴────────────────────────┐
        │                                                 │
        ▼                                                 ▼
┌───────────────────┐                          ┌─────────────────┐
│   PR → main       │                          │   Push → main   │
│   link-check.yml  │                          │   deploy.yml    │
└───────────────────┘                          └────────┬────────┘
                                                        │
                                                        ▼
                                            ┌─────────────────────────┐
                                            │ post-deploy-smoke-test  │
                                            │      .yml               │
                                            └─────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                     手动触发 Workflow                               │
├─────────────────────────────────────────────────────────────────────┤
│  sync-notion.yml / import-content.yml / delete-article.yml          │
│              ↓ 创建 PR                                              │
│        ┌─────────────────┐                                          │
│        │  validation.yml │  (自动触发)                              │
│        └────────┬────────┘                                          │
│                 ↓ 合并                                              │
│        ┌─────────────────┐                                          │
│        │   deploy.yml    │                                          │
│        └─────────────────┘                                          │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 四、CI 安全策略

### 4.1 无需 Secrets 的 Workflow（PR 自动运行）

以下 workflow 只使用 `contents: read`，可在来自 fork 的 PR 上自动运行：

- `validation.yml`
- `link-check.yml`

### 4.2 需要 Secrets 的 Workflow（仅手动触发）

以下 workflow 需要 secrets，只能手动触发：

| Workflow             | 需要的 Secrets                                                   |
| -------------------- | ---------------------------------------------------------------- |
| `sync-notion.yml`    | `NOTION_TOKEN`、`NOTION_DATABASE_ID`、`DEEPSEEK_API_KEY`（可选） |
| `import-content.yml` | `DEEPSEEK_API_KEY`（可选）                                       |
| `delete-article.yml` | 无                                                               |

---

## 五、Workflow 与 Scripts 的关系

| Workflow             | npm script               | 脚本文件                    |
| -------------------- | ------------------------ | --------------------------- |
| `sync-notion.yml`    | `npm run notion:sync`    | `scripts/notion-sync.ts`    |
| `import-content.yml` | `npm run import:content` | `scripts/content-import.ts` |
| `delete-article.yml` | `npm run delete:article` | `scripts/delete-article.ts` |

**职责边界**：

- Workflow：触发时机、环境配置、权限管理、PR 创建
- Scripts：具体的内容获取、转换、写入逻辑

---

## 六、后续调整指南

### 添加新 Workflow

1. 创建 `.github/workflows/<name>.yml`
2. 定义触发条件（`on:`）和权限（`permissions:`）
3. 如需调用 scripts，参考现有 workflow 模式
4. 在本文档中补充说明

### 修改现有 Workflow

1. 确认是否影响 PR 合并策略
2. 测试修改是否破坏 CI 流程
3. 更新本文档中的相关说明

### 调整 Scripts

如需修改 scripts 的参数或行为：

1. 修改 scripts 源码
2. 更新 [scripts/README.md](../scripts/README.md)
3. 如有必要，调整调用该 script 的 workflow
4. 本文档通常不需修改（除非触发条件或 job 结构变化）
