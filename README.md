# Astro + Notion Static Blog

一个由 **Astro** 和 **Notion** 驱动的静态博客示例。通过同步 Notion 数据库里的页面生成 Markdown 内容，并在构建时完成数学公式渲染，适合想要用 Notion 作为内容源的个人或团队。

## 🎯 特性

- **Notion 写作流程**：用 Notion Database 管理文章，状态为 Published 的页面会被拉取并转成 Markdown。
- **图片与封面下载**：同步时自动下载 Notion 中的图片与封面到 `public/images/notion/<pageId>/`。
- **数学公式支持**：结合 `remark-math` 与 `rehype-katex` 渲染公式，同步后还会用脚本修正常见格式问题。
- **现代前端**：Astro + Tailwind 构建，提供 RSS、Sitemap 与基础的文章列表/详情页。

## 🚀 快速开始

### 1. 环境要求

- Node.js 22
- 可访问的 Notion 账号与数据库

### 2. 安装与配置

1. **克隆仓库并安装依赖**

   ```bash
   git clone <your-repo-url>
   cd blog
   npm install
   ```

2. **配置环境变量**
   复制 `.env.local.example` 为 `.env.local` 并填写 Notion 信息：

   ```ini
   NOTION_TOKEN=secret_your_token_here
   NOTION_DATABASE_ID=your_database_id_here
   ```

   - **Token**：前往 [Create Integration](https://www.notion.so/my-integrations) 创建并获取。
   - **Database ID**：来自 Notion 数据库 URL（`notion.so/` 后的 32 位字符串）。
   - **权限**：在数据库右上角 `...` → `Connect` → 选择你的 Integration，否则无法读取数据。

3. **本地开发**
   ```bash
   npm run dev
   ```
   默认在 `http://localhost:4321` 提供预览。
   （如需运行 E2E，请先执行一次 `npx playwright install --with-deps chromium` 安装浏览器。）

### 3. 内容同步与写作

支持两种方式：

1. **Notion 驱动**：在 Notion 数据库中写作并将状态设为 **Published**（支持 `select` 或 `status` 属性），然后运行同步脚本：

   ```bash
   npm run notion:sync
   ```

   - 会将页面转换为 Markdown，输出到 `src/content/blog/notion/`。
   - 自动下载页面中的图片与封面到 `public/images/notion/`，并为引用生成本地路径。
   - 自动运行 `scripts/fix-math.ts` 修正常见数学公式格式（如去除 `$ x $` 中的空格，将多行行内公式提升为块级）。

2. **从 URL 导入**：支持从知乎、微信公众号、Medium 等平台导入文章，支持覆盖、预览、封面自动取首图等可选参数：

   ```bash
   npm run import:content -- --url="<文章URL>"
   ```

   示例：

   ```bash
   # 导入微信公众号文章
   npm run import:content -- --url="https://mp.weixin.qq.com/s/Pe5rITX7srkWOoVHTtT4yw"

   # 导入知乎文章
   npm run import:content -- --url="https://zhuanlan.zhihu.com/p/123456789"
   ```

   - 自动识别平台（微信、知乎、Medium）并提取内容
   - 自动下载文章中的所有图片到 `public/images/<平台>/<文章slug>/`
   - 生成 MDX 文件到 `src/content/blog/<平台>/`
   - 微信公众号图片下载包含占位符检测、重试机制和浏览器回退策略
   - 自动运行数学公式修正和代码格式化

3. **本地 Markdown**：在 `src/content/blog/` 下添加 `.md/.mdx` 文件，满足以下 Frontmatter 即可：
   ```yaml
   ---
   title: 文章标题
   date: 2025-01-01
   status: published # 或 draft
   cover: /images/your-cover.png # 可选，指向 public 下资源或远程 URL
   ---
   ```
   文件名会成为路由的一部分，例如 `hello-world.md` 生成 `/hello-world/`，与 Notion 同步的文章并列展示。

构建或部署前请先同步内容，确保最新文章被包含在站点中。

### 4. 数学公式

- **行内**：`$E=mc^2$`
- **块级**：
  ```latex
  $$
  \sum_{i=0}^n i^2 = \frac{(n^2+n)(2n+1)}{6}
  $$
  ```

如需单独处理指定文件，可直接运行：

```bash
npx tsx scripts/fix-math.ts src/content/blog/notion/<file>.md
```

## 🧭 项目结构

```
├── .github/workflows/        # CI / 部署（如果启用）
├── public/images/
│   ├── notion/               # 同步的 Notion 图片与封面
│   ├── wechat/               # 微信公众号文章图片
│   └── zhihu/                # 知乎文章图片（如有）
├── scripts/
│   ├── notion-sync.ts        # Notion → Markdown 转换与图片下载（会串联 fix-math 与 lint）
│   ├── content-import.ts     # 从 URL 导入文章（支持微信、知乎、Medium，串联 fix-math 与 lint）
│   └── fix-math.ts           # 数学公式修正
├── src/
│   ├── content/blog/local/   # 手写 Markdown（可选自行创建）
│   ├── content/blog/notion/  # Notion 同步生成的 Markdown（自动写入）
│   ├── content/blog/wechat/  # 微信公众号导入的文章
│   ├── pages/                # 路由（index, about, [...slug]）
│   └── layouts/              # 基础页面布局
```

## 🔧 常用命令

| 命令                                      | 说明                                                            |
| :---------------------------------------- | :-------------------------------------------------------------- |
| `npm run dev`                             | 启动开发服务器（默认 `localhost:4321`）                         |
| `npm run build`                           | 生成生产构建                                                    |
| `npm run preview`                         | 预览生产构建                                                    |
| `npm run notion:sync`                     | 拉取 Notion 文章、下载图片、修复公式并运行 lint（会改写内容）   |
| `npm run import:content -- --url="<URL>"` | 从指定 URL 导入文章并本地化资源，随后修复公式并运行 lint        |
| `npm run check`                           | Astro 类型检查                                                  |
| `npm run test`                            | Vitest 单元测试（含覆盖率）                                     |
| `npm run test:e2e`                        | Playwright 端到端测试（需先 `npx playwright install --with-deps chromium`） |
| `npm run lint`                            | prettier + markdownlint（含自动修复，可能改写文件）             |

## 🛠️ CI / Workflow 对齐
- PR / Push：`validation.yml` 统一执行 check、lint、test、build、E2E 与 smoke job。
- 部署：`deploy.yml` 发布 GitHub Pages，成功后由 `post-deploy-smoke-test.yml` 做线上探活。
- 内容：`import-content.yml` 手动导入、`sync-notion.yml` 定时同步，均通过 PR 写入内容与图片。
- 辅助：`pr-preview.yml` 提供 PR 预览，`link-check.yml` 做死链检测。
- 详细关系图与说明见 [`docs/ci-workflow-map.md`](docs/ci-workflow-map.md)。

## 📄 许可证

本项目基于 [ISC License](LICENSE) 开源，欢迎在许可范围内自由使用与修改。
