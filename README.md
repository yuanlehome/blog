# Astro 静态博客

基于 **Astro** 的生产级静态博客，支持 Notion 同步、外部文章导入和本地 Markdown。

---

## 快速导航

| 文档                              | 说明                      |
| --------------------------------- | ------------------------- |
| [架构说明](docs/architecture.md)  | 仓库结构与模块设计        |
| [CI 工作流](docs/ci-workflow.md)  | GitHub Actions 触发与流程 |
| [站点配置](docs/configuration.md) | YAML 配置文件详解         |
| [配置治理](docs/config-audit.md)  | 配置入口与一致性规范      |
| [Scripts 指南](scripts/README.md) | 脚本参数与用例            |

---

## 核心能力

- **多源内容**：Notion 数据库、外部文章抓取（知乎/微信/Medium/PDF）、本地 Markdown
- **预取构建**：脚本在构建前生成内容文件，构建期不访问外部接口
- **自动化 CI**：内容同步、部署、质量检查全自动化（每日定时/手动触发）
- **数学公式**：KaTeX 支持行内/块级公式
- **暗黑模式**：支持亮/暗主题切换，自动同步代码高亮与评论主题
- **标签系统**：标签完全来自 frontmatter `tags` 字段，不自动生成

---

## 开发者常用命令

```bash
# 开发
npm run dev              # 启动开发服务器（默认 http://localhost:4321/blog/）
npm run start            # 同 npm run dev

# 内容操作
npm run notion:sync      # 同步 Notion 内容（自动格式化）
npm run import:content -- --url="<URL>"  # 导入外部文章（自动格式化）
npm run delete:article -- --target=<slug>  # 删除文章

# 质量检查
npm run check            # Astro/TypeScript 类型检查
npm run lint             # Prettier + Markdownlint 格式化与检查
npm run format:check     # 仅格式化检查
npm run md:lint          # 仅 Markdownlint 检查
npm run test             # Vitest 单元测试（含覆盖率）
npm run test:watch       # Vitest 监听模式
npm run test:e2e         # Playwright E2E 测试（自动安装 chromium + 构建 + 测试）
npm run test:ci          # CI 完整测试（check + test + test:e2e + build）
npm run ci               # 完整 CI（安装依赖 + test:ci）

# 构建与预览
npm run build            # 构建静态站点
npm run preview          # 本地预览构建结果

# 配置审计
npm run config:audit     # 检查 YAML 配置生效性
```

---

## 快速开始

### 前置条件

- Node.js **22+**
- 使用 Notion 同步时需要 Notion 账号

### 初始化

```bash
git clone <repository-url>
cd blog
npm install
```

### 配置环境变量

```bash
cp .env.local.example .env.local
```

如需 Notion 同步，编辑 `.env.local`：

```env
NOTION_TOKEN=secret_your_token_here
NOTION_DATABASE_ID=your_database_id_here
```

### 启动开发

```bash
npm run dev
```

---

## 目录结构

```text
blog/
├── src/              # Astro 站点逻辑
│   ├── config/       # 集中式配置
│   ├── lib/          # 业务逻辑（按领域组织）
│   ├── content/blog/ # 内容集合
│   ├── components/   # UI 组件
│   └── pages/        # 路由
├── scripts/          # 内容获取脚本
├── docs/             # 设计文档
├── .github/workflows/# CI/CD
├── tests/            # 测试
└── public/           # 静态资源
```

> **详细说明** → [docs/architecture.md](docs/architecture.md)

---

## CI/CD

| Workflow                     | 触发条件              | 职责              |
| ---------------------------- | --------------------- | ----------------- |
| `validation.yml`             | PR / Push → main      | 质量门禁（含烟测） |
| `deploy.yml`                 | Push → main / 手动    | 部署 GitHub Pages |
| `sync-notion.yml`            | 每日 00:00 UTC / 手动 | 同步 Notion       |
| `import-content.yml`         | 手动                  | 导入外部文章/PDF  |
| `delete-article.yml`         | 手动                  | 删除文章          |
| `post-deploy-smoke-test.yml` | 部署后                | 烟测              |
| `link-check.yml`             | PR / Push / 每周一    | 链接有效性检查    |
| `pr-preview.yml`             | PR 打开/同步/关闭     | PR 预览站点       |

> **详细说明** → [docs/ci-workflow.md](docs/ci-workflow.md)

---

## 内容写作

### Frontmatter 字段

博客文章使用 Markdown 文件存储在 `src/content/blog/` 目录（按来源分类：`notion/`、`wechat/`、`zhihu/`、`medium/`、`others/`）。

**必需字段**：

```yaml
---
title: '文章标题'
slug: article-slug  # URL 路径，自动生成或手动指定
date: '2025-12-30'  # 发布日期 (YYYY-MM-DD)
tags: ['标签1', '标签2']  # 标签列表（标签系统完全依赖此字段）
status: published  # 或 draft
---
```

**可选字段**：

```yaml
cover: /images/folder/image.png  # 封面图（相对 public/ 路径）
source_url: https://example.com  # 原文 URL
source_author: 作者名  # 原文作者
imported_at: '2025-12-30T10:00:00Z'  # 导入时间
lang: zh  # 语言（zh/en）
translatedFrom: en  # 翻译源语言
comments: false  # 禁用评论（默认启用）
```

### 标签系统

- 标签**完全来自** frontmatter `tags` 字段
- 不自动生成，不从内容中提取
- 新增标签直接写入 `tags` 列表即可

### 本地写作

```bash
# 1. 在 src/content/blog/ 创建 .md 文件（推荐放入 others/ 子目录）
# 2. 添加 frontmatter（至少包含 title/slug/date/tags/status）
# 3. 运行开发服务器预览
npm run dev
# 4. 格式化检查
npm run lint
```

---

## 贡献

欢迎贡献！提交前请：

1. 阅读 [docs/architecture.md](docs/architecture.md) 了解设计约定
2. 运行质量检查：`npm run check && npm run lint && npm run test`
3. 遵循现有代码组织方式

---

## 许可证

遵循 [ISC License](LICENSE)。
