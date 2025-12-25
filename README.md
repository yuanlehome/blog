# Astro + Notion Static Blog

一个由 **Astro** 和 **Notion** 驱动的现代静态博客。支持在 Notion（类 CMS）和 Markdown（Git 工作流）中写作，同时提供数学公式渲染、站内搜索等特性，适合个人或团队搭建内容网站。

## 🎯 特性
- **双写作模式**：Notion 数据库与本地 Markdown 并行，满足不同习惯。
- **自动同步**：通过脚本或 GitHub Actions 定期同步 Notion 内容并修正公式格式。
- **数学公式**：内置 KaTeX，支持行内与块级公式。
- **现代前端**：基于 Astro + Tailwind，生成快速、可扩展的静态站点。
- **可部署性**：适配主流静态托管，提供 RSS、Sitemap 等 SEO 基础能力。

## 🚀 快速开始

### 1. 环境要求
- Node.js 18+
- Notion 账号与数据库
- GitHub 账号（可选，用于自动同步）

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
   NOTION_TOKEN=secret_...
   NOTION_DATABASE_ID=...
   ```
   - **Token**：前往 [Create Integration](https://www.notion.so/my-integrations) 创建并获取。
   - **Database ID**：来自 Notion 数据库 URL。
   - **权限**：在数据库右上角 `...` → `Connect to` → 选择你的 Integration。

3. **本地开发**
   ```bash
   npm run dev
   ```
   访问 `http://localhost:4321` 查看效果。

### 3. 写作方式

#### 选项 A：使用 Notion（推荐）
1. 在 Notion 数据库中撰写文章并将状态设为 **Published**。
2. 同步内容：
   - 本地运行：`npm run notion:sync`
   - GitHub Actions：打开 Actions → “Sync Notion Content” → Run workflow。
   - 默认每天 00:00 UTC 自动执行。

#### 选项 B：使用本地 Markdown
1. 在 `src/content/blog/local/` 下创建文件，如 `my-post.md`。
2. 添加 frontmatter：
   ```yaml
   ---
   title: "My Post"
   date: 2023-10-01
   status: "published"
   ---
   ```
3. 提交并推送代码。

### 4. 数学公式支持
- **行内公式**：`$ E=mc^2 $`
- **块级公式**：
  ```latex
  $$
  \sum_{i=0}^n i^2 = \frac{(n^2+n)(2n+1)}{6}
  $$
  ```

**自动修正脚本**：同步脚本会调用 `scripts/fix-math.ts`：
1. 去除行内公式空格（`$ x $` → `$x$`）。
2. 将多行行内公式提升为块级公式。

手动修正指定文件：
```bash
npx tsx scripts/fix-math.ts src/content/blog/local/my-post.md
```

## 🧭 项目结构
```
├── .github/workflows/    # CI/CD（部署与同步）
├── public/images/notion/ # 同步的 Notion 图片
├── scripts/
│   ├── notion-sync.ts    # Notion → Markdown 转换
│   └── fix-math.ts       # 数学公式修正
├── src/
│   ├── content/blog/
│   │   ├── local/        # 手动维护的 Markdown 文件
│   │   └── notion/       # Notion 自动生成内容（勿手动编辑）
│   ├── pages/            # 路由（index, about, [...slug]）
│   └── layouts/          # 基础页面布局
```

## 🔧 常用命令

| 命令 | 说明 |
| :--- | :--- |
| `npm run dev` | 启动开发服务器（默认 `localhost:4321`） |
| `npm run build` | 生成生产构建 |
| `npm run preview` | 预览生产构建 |
| `npm run notion:sync` | 拉取 Notion 文章、下载图片并修复公式 |
| `npm run format` | 使用 Prettier 格式化 `scripts/` 与 `src/` 代码 |

## 🤝 贡献指南
- 请先阅读 [CONTRIBUTING.md](CONTRIBUTING.md) 了解分支、提交与测试流程。
- 提交 Issue 前请提供复现步骤、期望结果和环境信息。
- 行为需遵守 [行为准则](CODE_OF_CONDUCT.md)。

## 📄 许可证

本项目基于 [ISC License](LICENSE) 开源，欢迎在许可范围内自由使用与修改。
