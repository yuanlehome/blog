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

- **多源内容**：Notion 数据库、外部文章抓取（知乎/微信/Medium）、本地 Markdown
- **预取构建**：脚本在构建前生成内容文件，构建期不访问外部接口
- **自动化 CI**：内容同步、部署、质量检查全自动化
- **数学公式**：KaTeX 支持行内/块级公式

---

## 开发者常用命令

```bash
# 开发
npm run dev              # 启动开发服务器（默认 http://localhost:4321/blog/）

# 内容操作
npm run notion:sync      # 同步 Notion 内容
npm run import:content -- --url="<URL>"  # 导入外部文章
npm run delete:article -- --target=<slug>  # 删除文章

# 质量检查
npm run check            # Astro/TypeScript 类型检查
npm run lint             # Prettier + Markdownlint
npm run test             # Vitest 单元测试
npm run test:e2e         # Playwright 端到端测试（首次运行需先安装浏览器：npx playwright install chromium）

# 构建与预览
npm run build            # 构建静态站点
npm run preview          # 本地预览构建结果
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

| Workflow                     | 触发条件         | 职责              |
| ---------------------------- | ---------------- | ----------------- |
| `validation.yml`             | PR / Push → main | 质量门禁          |
| `deploy.yml`                 | Push → main      | 部署 GitHub Pages |
| `sync-notion.yml`            | 每日 / 手动      | 同步 Notion       |
| `import-content.yml`         | 手动             | 导入外部文章      |
| `post-deploy-smoke-test.yml` | 部署后           | 烟测              |

> **详细说明** → [docs/ci-workflow.md](docs/ci-workflow.md)

---

## 贡献

欢迎贡献！提交前请：

1. 阅读 [docs/architecture.md](docs/architecture.md) 了解设计约定
2. 运行质量检查：`npm run check && npm run lint && npm run test`
3. 遵循现有代码组织方式

---

## 许可证

遵循 [ISC License](LICENSE)。
