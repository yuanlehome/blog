# 架构文档

本文档描述仓库的 **模块划分与职责边界**，是理解代码结构的入口。

> **相关文档**
>
> - Scripts 使用说明 → [scripts/README.md](../scripts/README.md)
> - CI 工作流 → [ci-workflow.md](./ci-workflow.md)
> - 站点配置 → [configuration.md](./configuration.md)
> - 配置治理 → [config-audit.md](./config-audit.md)

---

## 一、仓库模块划分

```text
blog/
├── src/              # Runtime 层：Astro 站点逻辑
│   ├── config/       # 集中式配置（路径、站点信息、功能开关）
│   ├── lib/          # 业务逻辑（按领域组织）
│   ├── content/      # 内容集合（Markdown/MDX 文件）
│   ├── components/   # UI 组件
│   ├── layouts/      # 页面布局
│   └── pages/        # 路由入口
├── scripts/          # Scripts 层：内容获取与预处理工具
│   ├── import/       # 平台适配器（adapter 架构）
│   ├── markdown/     # Markdown 处理管线（翻译、修复）
│   └── logger/       # 脚本日志工具
├── docs/             # 设计文档
├── .github/workflows/# CI/CD 自动化
├── tests/            # Vitest / Playwright 测试
└── public/           # 静态资源与下载的图片
```

---

## 二、模块职责边界

### 2.1 Runtime 层（`src/`）

Astro 构建期和浏览器运行期执行的代码。**完全隔离于 scripts**，仅操作预生成的内容制品。

| 子模块              | 职责                                           |
| ------------------- | ---------------------------------------------- |
| `src/config/`       | 路径、站点元数据、环境变量、功能开关           |
| `src/lib/slug/`     | Slug 生成与冲突检测（**可被 scripts 导入**）   |
| `src/lib/content/`  | 内容查询、日期格式化、阅读时间、目录树         |
| `src/lib/markdown/` | Remark/Rehype 插件（标题锚点、外链、语法高亮） |
| `src/lib/site/`     | 资源 URL 解析                                  |
| `src/lib/ui/`       | 客户端交互（代码复制、浮动按钮）               |
| `src/lib/theme/`    | 主题切换逻辑（CSS 生成）                       |
| `src/content/blog/` | 博客文章（notion/、wechat/、others/、本地）    |

### 2.2 Scripts 层（`scripts/`）

内容获取与预处理，在 **Astro 构建之外** 通过 Node.js 运行。

| 脚本                | 职责                                 |
| ------------------- | ------------------------------------ |
| `notion-sync.ts`    | 同步 Notion 数据库已发布页面         |
| `content-import.ts` | 从外部 URL（知乎、微信、Medium）导入 |
| `delete-article.ts` | 删除文章及关联图片                   |
| `config-audit.ts`   | 配置生效性审计                       |

> **详细参数与用法** → [scripts/README.md](../scripts/README.md)

### 2.3 CI/Workflows（`.github/workflows/`）

自动化触发与执行。

| Workflow                     | 职责                      |
| ---------------------------- | ------------------------- |
| `validation.yml`             | PR/push 质量门禁（含烟测） |
| `deploy.yml`                 | 构建并发布到 GitHub Pages |
| `sync-notion.yml`            | 定时/手动同步 Notion      |
| `import-content.yml`         | 手动导入外部文章/PDF      |
| `delete-article.yml`         | 手动删除文章              |
| `post-deploy-smoke-test.yml` | 部署后烟测                |
| `link-check.yml`             | 链接有效性检查            |
| `pr-preview.yml`             | PR 预览站点               |
| `copilot-fix-posts.yml`      | Copilot 修复文章          |

> **详细说明** → [ci-workflow.md](./ci-workflow.md)

### 2.4 Docs（`docs/`）

| 文档               | 职责                         |
| ------------------ | ---------------------------- |
| `architecture.md`  | 仓库结构与模块设计（本文档） |
| `ci-workflow.md`   | CI/Workflow 整体工作流       |
| `configuration.md` | 用户/站点层面配置            |
| `config-audit.md`  | 配置治理与一致性             |

---

## 三、依赖方向与边界规则

### 3.1 依赖关系

```text
┌─────────────────────────────────────────────────────────────┐
│                      外部来源                               │
│    Notion 数据库 │ 外部 URL（微信/知乎）│ 本地 Markdown     │
└──────────────────────────┬──────────────────────────────────┘
                           │ 获取
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                    Scripts 层                               │
│  notion-sync.ts │ content-import.ts │ delete-article.ts    │
│                                                             │
│  可导入: src/config/paths.ts, src/lib/slug/                 │
└──────────────────────────┬──────────────────────────────────┘
                           │ 写入
                           ▼
              ┌─────────────────────────────┐
              │       内容制品              │
              │  src/content/blog/          │
              │  public/images/             │
              └──────────────┬──────────────┘
                             │ 读取
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                    Runtime 层                               │
│  src/config/ → src/lib/ → src/components/ → src/pages/      │
└──────────────────────────┬──────────────────────────────────┘
                           │ astro build
                           ▼
              ┌─────────────────────────────┐
              │     静态站点 (dist/)        │
              └─────────────────────────────┘
```

### 3.2 边界规则

| 方向               | 允许 | 说明                                              |
| ------------------ | ---- | ------------------------------------------------- |
| Scripts → Runtime  | 部分 | 只能导入 `src/config/paths.ts` 和 `src/lib/slug/` |
| Runtime → Scripts  | 禁止 | Runtime 不得依赖 scripts                          |
| Scripts → utils.ts | 允许 | Scripts 内部共享工具                              |
| Runtime → utils.ts | 禁止 | `scripts/utils.ts` 仅供 scripts 使用              |

### 3.3 共享模块

**可被 Scripts 和 Runtime 共同使用**：

- `src/config/paths.ts`：路径配置（单一数据源）
- `src/lib/slug/`：Slug 生成与验证

**仅供 Runtime 使用**：

- `src/lib/content/`、`src/lib/markdown/`、`src/lib/site/`、`src/lib/ui/`、`src/lib/theme/`

---

## 四、新增功能/脚本的规范

### 4.1 在 `src/` 中添加功能

| 功能类型          | 放置位置                           |
| ----------------- | ---------------------------------- |
| 配置（路径、URL） | `src/config/`                      |
| Markdown 转换     | `src/lib/markdown/`                |
| 内容查询/转换     | `src/lib/content/`                 |
| URL/Slug 相关     | `src/lib/slug/` 或 `src/lib/site/` |
| 客户端交互        | `src/lib/ui/`                      |
| 新领域（如搜索）  | `src/lib/<domain>/`                |

**禁止**：创建 `src/lib/utils/` 或 `src/lib/helpers/`

### 4.2 在 `scripts/` 中添加脚本

1. 创建 `scripts/<script-name>.ts`
2. 导入必要模块（`./utils`、`../src/config/paths`、`../src/lib/slug`）
3. 在 `package.json` 添加 npm script
4. 在 [scripts/README.md](../scripts/README.md) 添加说明

### 4.3 添加 Workflow

1. 创建 `.github/workflows/<workflow-name>.yml`
2. 定义触发条件（`on:`）和权限（`permissions:`）
3. 如需调用 scripts，参考现有 workflow 模式
4. 在 [ci-workflow.md](./ci-workflow.md) 添加说明

---

## 五、设计原则

1. **单向数据流**：内容从外部来源 → scripts → 制品 → runtime → 静态输出
2. **严格层隔离**：Scripts 不导入 `src/lib`（slug 除外）；Runtime 不获取外部内容
3. **领域驱动结构**：按功能领域组织代码（slug、content、markdown），不按类型（utils、helpers）
4. **单一数据源**：路径在 `src/config/paths`，Slug 在 `src/lib/slug`
5. **最小抽象**：具体实现优于推测性框架，只在需要时抽象
