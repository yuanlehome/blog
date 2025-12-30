# Astro 静态博客

基于 **Astro** 的生产级静态博客，内容来源可来自 **Notion**、**外部链接**（微信 / 知乎 / Medium 等）及 **本地 Markdown**。核心理念：内容获取脚本与 Astro 渲染彻底解耦，构建可复现、边界清晰。

---

## 🎯 项目概览

- 多源内容：Notion 数据库、外部文章抓取、本地 Markdown/MDX
- 预取 + 静态构建：脚本在 `astro build` 前生成内容文件，构建阶段不再访问外部接口
- 数学公式：KaTeX 支持行内/块级公式
- 工具链：TypeScript、Tailwind CSS、Vitest、Playwright，全自动 CI/CD

---

## 📁 目录速览

```
blog/
├── src/
│   ├── lib/          # 运行时业务逻辑（slug、内容、markdown 插件）
│   ├── config/       # 统一配置（路径、站点信息、特性开关）
│   ├── content/      # 内容集合（Markdown/MDX）
│   │   └── blog/
│   │       ├── notion/  # Notion 同步内容（自动生成）
│   │       ├── wechat/  # 微信导入内容（自动生成）
│   │       ├── others/  # 其他平台导入内容（自动生成）
│   │       └── [root]   # 本地撰写内容
│   ├── components/   # 组件
│   ├── layouts/      # 页面布局
│   └── pages/        # 路由
├── scripts/          # 内容获取脚本（独立于 Astro 运行）
├── public/           # 静态资源与下载的图片
├── docs/             # 架构与 CI 文档
└── tests/            # Vitest / Playwright 测试
```

**约定**：`src/lib/` 按领域组织，不设通用 `utils/`；脚本共享工具集中在 `scripts/utils.ts`。

---

## 🚀 快速开始

### 前置条件

- Node.js **22+**
- 使用 Notion 同步时需要 Notion 账号

### 初始化

1. 安装依赖

```bash
git clone <repository-url>
cd blog
npm install
```

2. 配置环境变量

```bash
cp .env.local.example .env.local
```

如需 Notion 同步，填写：

```env
NOTION_TOKEN=secret_your_token_here
NOTION_DATABASE_ID=your_database_id_here
```

Notion 配置流程（需同步内容时）：

- 在 https://www.notion.so/my-integrations 创建集成，获得 `NOTION_TOKEN`
- 打开文章数据库，复制 URL 最后一个 `/` 之后到 `?`（如有）之前的 32 位字符串作为 `NOTION_DATABASE_ID`
- 在数据库右上角 `...` → **Connect to** 选择刚创建的集成，授予访问权限
- 确认页面状态字段支持 Published（select 或 status 均可）

3. 本地开发

```bash
npm run dev
```

默认访问 `http://localhost:4321/blog/`。

---

## ✍️ 内容工作流

### 1) Notion → Blog

- 在数据库中将页面状态设为 **Published**
- 执行：
  ```bash
  npm run notion:sync
  ```
- 自动拉取页面、生成 slug、下载图片到 `public/images/notion/<pageId>/`，并写入 `src/content/blog/notion/`。该目录文件会被下次同步覆盖，请在 Notion 内编辑。

### 2) 外部链接 → Blog（微信 / 知乎 / Medium 等）

- 执行：
  ```bash
  npm run import:content -- --url="<article-url>"
  ```
- 自动识别平台、使用 Playwright 抓取、转为 Markdown/MDX，图片保存在 `public/images/<platform>/<slug>/`，内容写入 `src/content/blog/<platform>/`。支持 `--allow-overwrite`、`--dry-run` 等参数。

### 3) 本地 Markdown

- 在 `src/content/blog/` 根目录新增 `.md`/`.mdx`：
  ```yaml
  ---
  title: 文章标题
  date: 2025-01-15
  status: published # 或 draft
  cover: /blog/images/cover.png # 可选
  ---
  ```
- 文件名即访问路径 `/blog/<filename>/`，与其他来源内容并存。

---

## 🛠️ 常用脚本

所有脚本均以 `npm run <script>` 运行，完整说明见 [scripts/README.md](scripts/README.md)。

| 类型 | 脚本                  | 作用                           |
| ---- | --------------------- | ------------------------------ |
| 开发 | `dev` / `start`       | 启动开发服务器                 |
| 构建 | `build` / `preview`   | 构建静态站点并本地预览         |
| 内容 | `notion:sync`         | 同步 Notion 内容并修正公式格式 |
|      | `import:content`      | 抓取外部文章生成 Markdown/MDX  |
|      | `delete:article`      | 删除文章及关联图片             |
| 质量 | `check`               | Astro/TS 类型检查              |
|      | `lint`                | Prettier + Markdownlint        |
|      | `test` / `test:watch` | Vitest 单测                    |
|      | `test:e2e`            | Playwright 端到端测试          |
|      | `test:ci`             | CI 全量校验                    |

常用示例：

```bash
npm run notion:sync
npm run import:content -- --url="<article-url>"
npm run delete:article -- --target=<slug>
npm run check && npm run lint && npm run test
```

---

## 🧪 开发与质量

- 开发：`npm run dev`，支持 HMR
- 类型检查：`npm run check`
- 格式与 Markdown 规范：`npm run lint`（会自动修复）
- 单元测试：`npm run test`，覆盖率在 `coverage/`
- 端到端测试：`npm run test:e2e`（首次需安装 Playwright 依赖）
- 预合并建议：`npm run check && npm run lint && npm run test && npm run test:e2e && npm run build`

---

## 📚 文档

- [架构说明](docs/architecture.md)：模块职责与设计规范
- [配置指南](docs/configuration.md)：**YAML 配置文件详解，自定义博客 UI 无需改代码**
- [CI 工作流](docs/ci-workflow.md)：GitHub Actions 触发与关系
- [Scripts 指南](scripts/README.md)：脚本参数与用例

---

## 🔗 CI/CD

仓库使用 GitHub Actions：

- `validation.yml`：PR 与 push 的检查（类型、lint、测试、构建、E2E）
- `deploy.yml`：合并 `main` 后部署到 GitHub Pages
- `sync-notion.yml`：定时同步 Notion
- `import-content.yml`、`delete-article.yml`：手动触发内容导入/删除
- `post-deploy-smoke-test.yml`：部署后烟测
- `link-check.yml`、`pr-preview.yml`：链接检查与 PR 预览

详情见 [docs/ci-workflow.md](docs/ci-workflow.md)。

---

## 📄 许可证

遵循 [ISC License](LICENSE)，可在许可范围内自由使用与修改。

---

## 🙏 贡献

欢迎贡献！提交前请：

1. 阅读 [`docs/architecture.md`](docs/architecture.md) 了解设计约定
2. 运行质量检查：
   ```bash
   npm run check && npm run lint && npm run test && npm run test:e2e
   ```
3. 遵循现有代码组织方式（按领域拆分，无通用 util 目录）
4. 添加新功能时同步更新相关文档

有疑问可提 Issue 或 Discussion。
