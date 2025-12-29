# 架构文档

本文档专注于 **仓库架构与设计规范**。

> **关于 Scripts 的使用说明**：请参见 [scripts/README.md](../scripts/README.md)  
> **关于 CI / Workflow**：请参见 [ci-workflow.md](./ci-workflow.md)  
> **设计原则**：代码是唯一的真相来源。本文档反映当前仓库结构，并基于实际存在的代码解释设计决策。

---

## 一、整体架构

这是一个 **内容管道驱动的静态博客**，内容生成与运行时渲染严格分离：

```text
┌─────────────────────────────────────────────┐
│         内容来源                            │
│  ┌─────────┐ ┌──────────┐ ┌──────────────┐ │
│  │ Notion  │ │ 外部 URL │ │  本地 MD     │ │
│  │ 数据库  │ │（微信等）│ │  文件        │ │
│  └────┬────┘ └─────┬────┘ └──────┬───────┘ │
└───────┼────────────┼─────────────┼─────────┘
        │            │             │
        v            v             v
┌───────────────────────────────────────────────┐
│         Scripts 层（Node.js）                 │
│  ┌──────────────┐  ┌─────────────────────┐   │
│  │ notion-sync  │  │  content-import     │   │
│  │     .ts      │  │      .ts            │   │
│  └──────┬───────┘  └──────┬──────────────┘   │
│         │                 │                   │
│  ┌──────┴─────────────────┴──────────┐       │
│  │      scripts/utils.ts              │       │
│  │  （共享工具函数）                  │       │
│  └────────────────────────────────────┘       │
└───────────────────┬───────────────────────────┘
                    │ 写入
                    v
        ┌───────────────────────────┐
        │   内容制品（Artifacts）   │
        │  src/content/blog/        │
        │  public/images/           │
        └───────────┬───────────────┘
                    │ 构建时读取
                    v
┌───────────────────────────────────────────────┐
│         Runtime 层（Astro）                   │
│                                               │
│  ┌────────────────────────────────────────┐  │
│  │          src/lib/                      │  │
│  │  ┌──────┐ ┌──────┐ ┌────────┐         │  │
│  │  │ slug │ │content│ │markdown│ ...     │  │
│  │  └──────┘ └──────┘ └────────┘         │  │
│  └────────────────────────────────────────┘  │
│                                               │
│  ┌────────────────────────────────────────┐  │
│  │  Components / Layouts / Pages          │  │
│  └────────────────────────────────────────┘  │
└───────────────────┬───────────────────────────┘
                    │ astro build
                    v
        ┌───────────────────────────┐
        │   静态站点输出            │
        │       (dist/)             │
        └───────────────────────────┘
```

### 关键架构原则

1. **单向数据流**：内容从外部来源 → scripts → 制品 → runtime → 静态输出
2. **严格层隔离**：Scripts 不导入 `src/lib`；runtime 不获取外部内容
3. **内容即数据**：`src/content/blog/` 中的所有内容被视为构建时数据，而非应用逻辑
4. **单一职责**：每一层都有清晰的边界和目标

---

## 二、Runtime 层（`src/`）

Runtime 层包含所有在 Astro 构建或浏览器中执行的代码。它 **完全隔离于 scripts**，仅操作预生成的内容制品。

### 2.1 配置（`src/config/`）

集中式配置模块，作为路径和设置的单一数据源：

- **`paths.ts`**：所有文件系统路径（内容目录、图片目录、构建输出）
  - 导出：`ROOT_DIR`、`BLOG_CONTENT_DIR`、`NOTION_CONTENT_DIR`、`PUBLIC_IMAGES_DIR`、`NOTION_PUBLIC_IMG_DIR`、`ARTIFACTS_DIR` 等
  - 被 runtime 和 scripts 共同使用，确保一致性
  - 支持通过环境变量覆盖（用于测试）

- **`site.ts`**：站点元数据（base URL、站点 URL、标题、描述）
  - 用于 RSS、sitemap、页面元数据生成

- **`env.ts`**：环境变量解析工具

- **`features.ts`**：功能开关（通过环境变量控制的布尔开关）

**设计理由**：通过在 `src/config/paths.ts` 中集中所有路径，我们消除了硬编码路径，并使项目易于为不同环境或部署目标重新配置。

### 2.2 业务逻辑（`src/lib/`）

这是 **runtime 业务逻辑的唯一位置**。每个子目录代表一个具有明确职责的领域：

#### `src/lib/slug/`

**职责**：所有 slug 生成与验证逻辑的单一数据源

- 提供 `slugFromTitle()` 用于将标题转换为 URL 安全的 slug
- 提供 `ensureUniqueSlug()` 用于检测和解决 slug 冲突
- 被 scripts（内容同步期间）和 runtime（路由生成）共同使用

**为什么集中化**：Slug 一致性对 URL 稳定性至关重要。拥有一个模块确保：

- Scripts 生成的 slug 与 runtime 期望的 slug 不会分歧
- 易于全局修改 slug 算法
- Slug 冲突检测在所有地方都以相同方式工作

#### `src/lib/content/`

**职责**：内容查询、转换、元数据提取

关键模块：

- `posts.ts`：从 Astro 的内容集合获取已发布文章
- `dates.ts`：日期格式化与解析工具
- `readingTime.ts`：根据内容计算估计阅读时间
- `tocTree.ts`：从标题构建目录树结构
- `slugger.ts`：为锚点链接生成标题 slugger

**为什么与 scripts 分离**：这些逻辑操作的是 _已同步_ 的内容。它不知道也不关心内容来自 Notion、外部 URL 还是本地 Markdown。

#### `src/lib/markdown/`

**职责**：用于 Astro 的 unified/remark/rehype 管道的 Markdown 处理插件

关键模块：

- `rehypeHeadingLinks.ts`：为标题添加锚点链接
- `rehypeExternalLinks.ts`：为外部链接添加 `target="_blank"` 和安全属性
- `rehypePrettyCode.ts`：通过 Shiki 进行语法高亮
- `remarkNotionCompat.ts`：Notion 特定的 Markdown 修复（如 callout 块）
- `remarkPrefixImages.ts`：为图片路径添加 base URL 前缀
- `remarkCodeMeta.ts`：解析代码块元数据

**为什么是单独的领域**：这些是在页面渲染期间发生的 runtime 转换，而不是在内容导入期间。它们是无状态转换，适用于任何 Markdown 内容。

#### `src/lib/site/`

**职责**：站点级工具

- `assetUrl.ts`：解析资源 URL，正确添加 base path 前缀

#### `src/lib/ui/`

**职责**：客户端交互逻辑

- `code-blocks.ts`：代码块的复制到剪贴板功能
- `floatingActionStack.ts`：浮动操作按钮堆栈计算（回到顶部、目录等）

**为什么分离**：纯客户端 JavaScript，与内容结构无依赖。

### 2.3 内容集合（`src/content/`）

- **`src/content/blog/`**：包含所有博客文章的 Markdown/MDX 文件
  - `notion/`：通过 `notion-sync.ts` 从 Notion 同步
  - `wechat/`：通过 `content-import.ts` 从微信文章导入
  - `others/`：从其他平台（知乎、Medium 等）导入
  - 也可以包含开发者直接放置的本地 Markdown 文件

- **`src/content/config.ts`**：Astro 内容集合的 schema 定义

**重要**：内容文件是 **数据，而非逻辑**。它们由 scripts 生成，被 runtime 消费。对已同步内容（如 `notion/`）的手动编辑将在下次同步时被覆盖。

### 2.4 Runtime 的依赖流

```text
src/config/
    ↓
src/lib/
    ↓
src/components/ + src/layouts/
    ↓
src/pages/
```

- **`config`** 不依赖其他 runtime 代码
- **`lib`** 可以依赖 `config`，但不依赖 components/layouts/pages
- **Components/Layouts** 可以依赖 `lib` 和 `config`
- **Pages** 协调一切，但包含最少的逻辑

---

## 三、Scripts 层（`scripts/`）

Scripts 层负责 **内容获取与准备**。Scripts 在 **Astro 构建之外** 通过 Node.js 运行，不了解 Astro 内部。

### 3.1 Scripts 的定位

**Scripts 做什么**：

- 从外部来源获取内容（Notion API、网页抓取）
- 下载和处理图片
- 生成带有正确 frontmatter 的 Markdown/MDX 文件
- 修复常见内容问题（数学公式格式、不可见字符）
- 维护 slug 唯一性并检测冲突

**Scripts 不做什么**：

- 渲染内容为 HTML（那是 Astro 的工作）
- 实现业务逻辑（属于 `src/lib/`）
- 被 runtime 代码导入（严格隔离）

### 3.2 `scripts/utils.ts` - 共享工具层

**设计决策**：使用 **单文件** 作为共享 script 工具，而不是 `scripts/lib/`

**为什么使用单文件？**

1. **Scripts 是入口点，不是库**：每个 script 是独立的命令行工具。它们不形成需要模块化结构的复杂依赖图。

2. **防止过早抽象**：创建 `scripts/lib/` 会邀请过度工程化。使用单文件，你会仔细考虑是否添加工具。

3. **清晰的所有权**：`scripts/utils.ts` 包含 **仅 script 特定的工具**，永远不应在 runtime 中使用。示例：
   - 文件 I/O 辅助（`ensureDir`、`processFile`、`processDirectory`）
   - 进程执行包装器
   - 数学分隔符修复（在 `$ x $` → `$x$` 中规范化空格）

4. **更易维护**：一个文件在清理工具时查看，而不是导航多个目录。

**`scripts/utils.ts` 中放什么**：

- 通用文件系统操作
- 内容修复的字符串处理工具
- 进程生成辅助
- 被 2+ 个 scripts 使用的工具

**`scripts/utils.ts` 中不放什么**：

- 业务逻辑（slug 生成 → `src/lib/slug/`）
- Runtime 转换（Markdown 插件 → `src/lib/markdown/`）
- 配置（路径 → `src/config/paths.ts`）

> **Scripts 的完整使用说明**：参见 [scripts/README.md](../scripts/README.md)

### 3.3 核心 Scripts（概念级说明）

#### `notion-sync.ts`

**职责**：将已发布的 Notion 页面同步到 Markdown 文件

**概念流程**：

1. 连接 Notion API
2. 查询已发布页面
3. 下载内容和图片
4. 使用 `src/lib/slug/` 生成 slug
5. 写入 Markdown 到 `src/content/blog/notion/`

> **详细参数与用法**：参见 [scripts/README.md](../scripts/README.md#21-notion-syncts)

#### `content-import.ts`

**职责**：从外部 URL（微信、知乎、Medium）导入文章

**概念流程**：

1. 接受 URL
2. 使用 Playwright 抓取内容
3. 下载图片
4. 转换为 Markdown
5. 写入 MDX 到 `src/content/blog/<platform>/`

> **详细参数与用法**：参见 [scripts/README.md](../scripts/README.md#22-content-importts)

#### `delete-article.ts`

**职责**：删除文章及其关联图片

**概念流程**：

1. 查找文章
2. 删除文章文件
3. 可选删除图片目录

> **详细参数与用法**：参见 [scripts/README.md](../scripts/README.md#24-delete-articlets)

---

## 四、模块之间的边界与依赖方向

### 4.1 依赖关系表

| 从 → 到            | 是否允许 | 说明                                              |
| ------------------ | -------- | ------------------------------------------------- |
| Scripts → Runtime  | ✅ 部分  | 只能导入 `src/config/paths.ts` 和 `src/lib/slug/` |
| Runtime → Scripts  | ❌ 禁止  | Runtime 不得依赖 scripts                          |
| Scripts → utils.ts | ✅ 允许  | Scripts 的共享工具层                              |
| Runtime → utils.ts | ❌ 禁止  | `utils.ts` 仅供 scripts 使用                      |

### 4.2 共享模块的选择

**可被 Scripts 和 Runtime 共同使用**：

- `src/config/paths.ts`：路径配置（单一数据源）
- `src/lib/slug/`：slug 生成与验证（保证 URL 一致性）

**仅供 Runtime 使用**：

- `src/lib/content/`、`src/lib/markdown/`、`src/lib/site/`、`src/lib/ui/`

**仅供 Scripts 使用**：

- `scripts/utils.ts`

### 4.3 为什么这样设计

**单向数据流的优势**：

- **可重现构建**：相同的内容文件 → 相同的输出
- **快速构建**：`astro build` 期间无 API 调用
- **清晰所有权**：内容同步 = scripts，内容显示 = Astro

**严格边界的优势**：

- **易于测试**：每一层可以独立测试
- **易于重构**：修改 scripts 不影响 runtime
- **易于理解**：新人可以快速定位代码位置

---

## 五、设计原则

### 5.1 核心原则

1. **领域驱动结构**：按功能领域组织代码（slug、content、markdown），而不是按类型（utils、helpers）

2. **最小抽象**：具体实现优于推测性框架。只在需要时抽象。

3. **单一数据源**：
   - 路径 → `src/config/paths`
   - Slug → `src/lib/slug`
   - 内容 → 外部来源（Notion、URL、本地文件）

4. **边界清晰**：每个模块有明确的职责，不越界

5. **无"utils"目录**：
   - `src/` 中没有 `utils/` 目录 → 按领域组织
   - `scripts/` 中没有 `lib/` 目录 → 使用单文件 `utils.ts`

### 5.2 为什么没有 `src/utils/` 或 `scripts/lib/`

**问题**：`utils/` 和 `lib/` 目录容易变成"什么都放"的垃圾桶

**解决方案**：

- **在 `src/` 中**：按领域组织（`slug/`、`content/`、`markdown/`）
  - 每个模块有明确的职责
  - 容易决定新代码应该放哪里

- **在 `scripts/` 中**：使用单文件 `utils.ts`
  - 迫使思考是否真的需要抽象
  - 防止过早优化

### 5.3 何时考虑重构

**`src/lib/` 添加新模块的条件**：

- 有明确的领域边界（不是"misc"或"helpers"）
- 有 3+ 个文件（不为单个函数创建目录）
- 职责单一且清晰

**`scripts/utils.ts` 拆分的条件**：

- 文件超过 500 行
- 出现多个独立的工具类别（如网络请求、图片处理）
- 有 3+ 个 scripts 共享工具

**当前阶段**：复杂度足够简单，当前结构足够。

---

## 六、后续开发规范

### 6.1 添加新的 Runtime 功能

**问题**：我需要在 `src/` 中添加新功能，应该放在哪里？

**决策树**：

1. **是配置吗？**（路径、URL、功能开关）
   → 放在 `src/config/`

2. **是 Markdown 转换吗？**（remark/rehype 插件）
   → 放在 `src/lib/markdown/`

3. **是内容查询/转换吗？**（获取文章、格式化日期）
   → 放在 `src/lib/content/`

4. **是 URL/Slug 相关吗？**
   → 放在 `src/lib/slug/` 或 `src/lib/site/`

5. **是客户端交互吗？**
   → 放在 `src/lib/ui/`

6. **是新的领域吗？**（如"评论"、"搜索"）
   → 创建新目录 `src/lib/<domain>/`

**不要**：创建 `src/lib/utils/` 或 `src/lib/helpers/`

### 6.2 添加新的 Script

**步骤**：

1. 在 `scripts/` 创建 `<script-name>.ts`
2. 导入 `./utils`、`../src/config/paths`、`../src/lib/slug`
3. 实现主逻辑
4. 在 `package.json` 添加 npm script
5. 在 `scripts/README.md` 添加说明

> **详细指南**：参见 [scripts/README.md](../scripts/README.md#五添加新-script-的指南)

### 6.3 添加新的 Workflow

**步骤**：

1. 在 `.github/workflows/` 创建 `<workflow-name>.yml`
2. 定义触发条件和权限
3. 编写 jobs 和 steps
4. 如需调用 scripts，参考现有 workflow
5. 在 `docs/ci-workflow.md` 添加说明

> **详细指南**：参见 [docs/ci-workflow.md](./ci-workflow.md#六后续调整指南)

### 6.4 修改共享模块

**修改 `src/config/paths.ts`**：

- 谨慎修改，会影响 scripts 和 runtime
- 确保路径仍然正确指向实际位置
- 更新依赖该路径的所有代码

**修改 `src/lib/slug/`**：

- 谨慎修改，会影响 URL 稳定性
- 考虑向后兼容（旧 URL 仍能工作）
- 测试 scripts 和 runtime 中的 slug 生成

---

## 七、总结

本架构通过 **清晰性实现可维护性**：

1. **两个独立世界**：Scripts（Node.js CLI 工具）和 Runtime（Astro 构建）
2. **严格层边界**：Config → Lib → Components → Pages（runtime）；Scripts 写入制品
3. **领域驱动结构**：每个模块有单一、明确定义的目标
4. **最小抽象**：具体实现优于推测性框架
5. **单一数据源**：路径在 `src/config/paths`，slug 在 `src/lib/slug`，内容来自外部来源

**不确定时**：

- 如果在 `npm run dev` 或 `npm run build` 期间运行 → 属于 `src/`
- 如果通过 `npm run notion:sync` 或 `npm run import:content` 运行 → 属于 `scripts/`
- 如果被两者使用 → 属于 `src/config/` 或 `src/lib/slug/`（唯一的共享模块）
