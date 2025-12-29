# Scripts 使用说明

本文档是 **scripts 的权威说明文档**，包含所有脚本的功能、参数、使用场景。

> **关于 CI / Workflow**：请参见 [docs/ci-workflow.md](../docs/ci-workflow.md)  
> **关于仓库架构设计**：请参见 [docs/architecture.md](../docs/architecture.md)

---

## 一、Scripts 在仓库中的定位

### 1.1 Scripts 是什么

Scripts 是 **内容获取与预处理工具**，运行在 Astro 构建之外，负责：

- 从外部数据源（Notion API、网页）获取内容
- 下载并本地化图片资源
- 转换内容为 Markdown/MDX 格式
- 修正常见格式问题（数学公式、不可见字符）
- 管理 slug 唯一性，避免冲突

### 1.2 Scripts 不是什么

- **不是 runtime 代码**：不会被 `src/` 中的 Astro 代码 import
- **不参与页面渲染**：仅负责内容准备，不涉及 HTML 生成
- **不是构建工具**：不替代 `astro build`，仅生成输入数据

### 1.3 Scripts 与仓库其它模块的关系

**与 Runtime 的关系**：

- Scripts 写入的文件（`src/content/blog/`、`public/images/`）是 Runtime 的数据输入
- Scripts 可以导入 `src/config/paths.ts` 和 `src/lib/slug/` 中的共享配置和工具
- Runtime **不得** 导入 `scripts/` 中的任何代码

**与 CI / Workflow 的关系**：

- Scripts 被 GitHub Actions workflow 调用（通过 `npm run` 命令）
- Workflow 负责触发时机、环境配置、PR 创建
- Scripts 负责具体的内容获取逻辑

> **详细的 Workflow 说明**：参见 [docs/ci-workflow.md](../docs/ci-workflow.md)

**与 Architecture 的关系**：

- Scripts 是"内容获取层"，位于架构的最上游
- Scripts 输出是 Runtime 的输入（单向数据流）

> **详细的架构设计**：参见 [docs/architecture.md](../docs/architecture.md)

---

## 二、脚本功能详解

### 2.1 `notion-sync.ts`

#### 功能

从 Notion 数据库同步已发布页面到博客，转换为 Markdown 文件并下载所有图片。

#### 使用场景

- 在 Notion 中撰写或更新文章
- 将文章状态设为"Published"
- 运行此脚本同步到博客仓库

#### 输入

**环境变量（必需）**：

- `NOTION_TOKEN`：Notion API 集成 token
  - 获取方式：在 [Notion Integrations](https://www.notion.so/my-integrations) 创建集成
- `NOTION_DATABASE_ID`：Notion 数据库 ID
  - 获取方式：数据库 URL 中的 32 字符串（`notion.so/` 后面）

**Notion 数据库要求**：

- 必须包含 `status` 属性（类型为 `select` 或 `status`）
- 只同步 status = "Published" 的页面
- 页面必须有标题

#### 输出目录

- **Markdown 文件**：`src/content/blog/notion/<slug>.md`
- **图片文件**：`public/images/notion/<pageId>/<imageId>.<ext>`

#### 参数

**无命令行参数**，仅通过环境变量配置。

#### 使用方法

```bash
# 确保 .env.local 中配置了 NOTION_TOKEN 和 NOTION_DATABASE_ID
npm run notion:sync
```

#### 执行流程

1. 连接 Notion API，查询 status = "Published" 的页面
2. 遍历每个页面：
   - 生成 slug（基于标题，检测冲突）
   - 下载封面图和正文中的所有图片
   - 使用 `notion-to-md` 转换为 Markdown
   - 写入 `src/content/blog/notion/<slug>.md`
3. 调用 `process-md-files.ts` 修正数学公式格式
4. 运行 `npm run lint` 格式化生成的文件

#### 幂等性

- 可安全地多次运行
- 已存在的 Notion 文件会被覆盖（基于 slug）
- 其他来源（wechat、others、本地文件）不受影响

⚠️ **重要**：不要手动编辑 `src/content/blog/notion/` 中的文件，下次同步会覆盖。应在 Notion 中编辑。

---

### 2.2 `content-import.ts`

#### 功能

从外部 URL（微信、知乎、Medium 等）导入文章，转换为 MDX 文件并下载所有图片。

#### 使用场景

- 导入微信公众号文章
- 导入知乎专栏文章
- 导入 Medium 文章
- 导入其他平台的 HTML 文章

#### 输入

**必需参数**：

- `--url`：文章的完整 URL

**可选参数**：

- `--allow-overwrite`：允许覆盖已存在的同 slug 文章（默认：不覆盖，会报错）
- `--dry-run`：预览模式，不实际写入文件，仅输出将要执行的操作
- `--use-first-image-as-cover`：如果文章没有封面，使用正文第一张图片作为封面（默认：启用）

**环境变量**（可替代命令行参数）：

- `URL`：等同于 `--url`
- `ALLOW_OVERWRITE`：`true` / `false`
- `DRY_RUN`：`true` / `false`
- `USE_FIRST_IMAGE_AS_COVER`：`true` / `false`

#### 输出目录

- **MDX 文件**：`src/content/blog/<platform>/<slug>.mdx`
  - `<platform>` 根据 URL 自动识别：`wechat`、`zhihu`、`medium`、`others`
- **图片文件**：`public/images/<platform>/<slug>/<imageId>.<ext>`

#### 参数详解（100% 与源码一致）

根据源码 `scripts/content-import.ts` 第 679-700 行：

| 参数名称                     | 源码位置   | 类型    | 默认值  | 说明                                                      |
| ---------------------------- | ---------- | ------- | ------- | --------------------------------------------------------- |
| `--url`                      | 行 680-687 | string  | 无      | 文章 URL（必填）。支持 `--url=<value>` 或 `--url <value>` |
| `--allow-overwrite`          | 行 689-690 | boolean | `false` | 是否允许覆盖已存在文章                                    |
| `--dry-run`                  | 行 692     | boolean | `false` | 预览模式，不写入文件                                      |
| `--use-first-image-as-cover` | 行 694-695 | boolean | `false` | 将正文首图作为封面（如果没有封面）                        |

**注意**：源码中 `use-first-image-as-cover` 默认值为 `false`，但在实际使用中通常启用。

#### 使用方法

```bash
# 基本用法：导入微信文章
npm run import:content -- --url="https://mp.weixin.qq.com/s/Pe5rITX7srkWOoVHTtT4yw"

# 导入知乎文章
npm run import:content -- --url="https://zhuanlan.zhihu.com/p/123456789"

# 覆盖已存在的文章
npm run import:content -- --url="<URL>" --allow-overwrite

# 预览模式（不实际写入）
npm run import:content -- --url="<URL>" --dry-run

# 使用首图作为封面
npm run import:content -- --url="<URL>" --use-first-image-as-cover
```

#### 执行流程

1. 根据 URL 识别平台（微信、知乎、Medium、其他）
2. 启动 Playwright 无头浏览器访问 URL
3. 提取文章元数据（标题、作者、发布时间）
4. 提取文章 HTML 内容
5. 下载所有图片到 `public/images/<platform>/<slug>/`
6. 将 HTML 转换为 Markdown（使用 rehype/remark 管道）
7. 生成 frontmatter（标题、日期、作者、封面等）
8. 写入 MDX 文件到 `src/content/blog/<platform>/<slug>.mdx`
9. 调用 `process-md-files.ts` 修正格式
10. 运行 `npm run lint` 格式化

#### 平台特性

**微信（WeChat）**：

- 处理图片占位符（防止下载失败）
- 重试机制（最多 5 次，指数退避）
- 浏览器回退下载（对付顽固图片）
- 占位符检测（通过文件大小和图片尺寸）

**知乎（Zhihu）**：

- 提取作者和发布日期
- 处理知乎特有的 DOM 结构

**Medium**：

- 类似知乎，提取平台特定元数据

**其他平台**：

- 通用 HTML 提取逻辑
- 自动检测标题、作者、日期

⚠️ **重要**：

- 导入的文章应在原平台编辑，或使用 `--allow-overwrite` 本地编辑后覆盖
- 重新导入会覆盖本地修改（除非不使用 `--allow-overwrite`）

---

### 2.3 `process-md-files.ts`

#### 功能

修正 Markdown 文件中的常见格式问题，主要针对数学公式格式。

#### 使用场景

- 修正从 Notion 导出的数学公式空格问题（`$ x $` → `$x$`）
- 清理不可见 Unicode 字符
- 作为其他脚本的后处理步骤（自动调用）

#### 输入

- 文件路径或目录路径

#### 输出目录

- 原地修改文件（in-place update）

#### 参数

```bash
npx tsx scripts/process-md-files.ts <file-or-directory-path>
```

- `<file-or-directory-path>`：要处理的文件或目录的绝对/相对路径

#### 使用方法

```bash
# 处理单个文件
npx tsx scripts/process-md-files.ts src/content/blog/notion/my-article.md

# 处理整个目录（递归）
npx tsx scripts/process-md-files.ts src/content/blog/notion/

# 通常不需要手动运行，由 notion-sync 和 import:content 自动调用
```

#### 执行流程

1. 识别输入是文件还是目录
2. 对每个 Markdown 文件：
   - 分离 frontmatter、代码块、文本内容
   - 清理不可见 Unicode 字符
   - 修正行内数学公式（去除 `$` 前后的空格）
   - 将多行行内数学公式提升为块级数学公式（`$$...$$`）
   - 重新组装文件内容
   - 写回原文件

#### 处理规则

**不可见字符规范化**：

- U+2060（WORD JOINER）→ 空字符串
- U+FEFF（ZERO WIDTH NO-BREAK SPACE）→ 空字符串

**数学公式修正**：

- `$ x $` → `$x$`
- `$  x  $` → `$x$`
- 多行的 `$ ... $` → `$$ ... $$`（块级公式）

⚠️ **重要**：不处理代码块内的内容，只处理普通文本。

---

### 2.4 `delete-article.ts`

#### 功能

删除指定文章及其关联的图片目录。

#### 使用场景

- 删除不再需要的文章
- 清理重复或错误导入的文章
- 同时清理文章关联的图片资源

#### 输入

**必需参数**：

- `--target`：文章的 slug 或文件路径

**可选参数**：

- `--delete-images`：同时删除文章关联的图片目录（默认：不删除）
- `--dry-run`：预览模式，不实际删除（默认：不启用）

**环境变量**（可替代命令行参数）：

- `TARGET`：等同于 `--target`
- `DELETE_IMAGES`：`true` / `false`
- `DRY_RUN`：`true` / `false`

#### 输出目录

**删除的文件**：

- 文章文件：`src/content/blog/**/<slug>.md`
- 图片目录（如果启用 `--delete-images`）：
  - 从 frontmatter `cover` 提取的图片目录
  - 匹配 slug 的图片目录（在 `public/images/` 下）

#### 参数详解（100% 与源码一致）

根据源码 `scripts/delete-article.ts` 第 155-184 行：

| 参数名称          | 源码位置   | 类型    | 默认值  | 说明                           |
| ----------------- | ---------- | ------- | ------- | ------------------------------ |
| `--target`        | 行 157-165 | string  | 无      | 文章的 slug 或相对路径（必填） |
| `--delete-images` | 行 166-167 | boolean | `false` | 是否同时删除关联图片目录       |
| `--dry-run`       | 行 168-169 | boolean | `false` | 预览模式，不实际删除文件       |

#### 使用方法

```bash
# 通过 slug 删除文章
npm run delete:article -- --target=my-article-slug

# 通过路径删除文章
npm run delete:article -- --target=src/content/blog/wechat/my-article.mdx

# 删除文章并清理图片
npm run delete:article -- --target=my-article-slug --delete-images

# 预览模式（查看将删除什么）
npm run delete:article -- --target=my-article-slug --delete-images --dry-run
```

#### 执行流程

1. 解析 `target` 参数：
   - 如果包含 `/` 或以 `.md` 结尾 → 视为文件路径
   - 否则 → 视为 slug
2. 根据 slug 或路径查找文章文件
3. 如果启用 `--delete-images`：
   - 提取文章 frontmatter 中的 `cover` 字段
   - 查找匹配的图片目录
   - 最多匹配 20 个目录（防止误删）
4. 删除文章文件
5. 删除图片目录（如果启用）
6. 输出删除日志

#### 安全机制

- **路径限制**：只能删除 `src/content/blog/` 内的文件
- **slug 冲突保护**：如果多个文件匹配同一 slug，会提示选择
- **批量删除保护**：如果匹配超过 20 个图片目录，需要确认
- **dry-run 模式**：预览将要删除的文件，不实际执行

⚠️ **重要**：

- 删除操作不可逆，建议先使用 `--dry-run` 预览
- 图片目录匹配基于 slug 和 cover 路径，可能不完全准确

---

## 三、`scripts/utils.ts` 的定位

### 3.1 为什么存在 `utils.ts`

Scripts 需要共享一些通用的文件操作和字符串处理逻辑，但这些逻辑：

- **不属于 Runtime**（不应在 `src/lib/` 中）
- **不适合放在单个 script 中**（会重复代码）
- **非常简单，不需要复杂的模块结构**（不需要 `scripts/lib/`）

因此，使用 **单个文件** `scripts/utils.ts` 作为 scripts 的共享工具层。

### 3.2 `utils.ts` 放什么

**适合放入 `utils.ts` 的内容**：

- 文件系统操作封装（`ensureDir`、`processFile`、`processDirectory`）
- 字符串处理工具（`normalizeInvisibleCharacters`、数学公式修正）
- 进程执行辅助函数（`runMain`）
- 被 2 个及以上 script 使用的工具函数

**示例函数**（当前已存在）：

```typescript
// 目录操作
export function ensureDir(dir: string): void;

// 文件处理
export function processFile(filePath: string, processFn: (text: string) => string): void;

export function processDirectory(
  dirPath: string,
  filterFn: (filename: string) => boolean,
  processFn: (text: string) => string,
): void;

// 错误处理
export function runMain(mainFn: () => Promise<void>): void;

// Markdown 处理
export function processMdFiles(text: string): string;
export function normalizeInvisibleCharacters(text: string): string;
export function splitCodeFences(text: string): {
  frontmatter: string;
  segments: Array<{ type: 'code' | 'text'; content: string }>;
};
```

### 3.3 `utils.ts` 不放什么

**不适合放入 `utils.ts` 的内容**：

- **业务逻辑**：slug 生成 → `src/lib/slug/`
- **Runtime 转换**：Markdown 插件 → `src/lib/markdown/`
- **配置**：路径定义 → `src/config/paths.ts`
- **单个 script 特有的逻辑**：应保留在对应 script 中

### 3.4 为什么不用 `scripts/lib/` 目录

**设计决策**：使用单个 `utils.ts` 而不是 `scripts/lib/` 目录

**原因**：

1. **Scripts 是入口，不是库**：每个 script 是独立的命令行工具，不需要复杂的模块结构
2. **防止过度抽象**：单文件迫使我们思考是否真的需要抽象，避免过早优化
3. **清晰的边界**：`scripts/utils.ts` 明确表示"仅供 scripts 使用的工具"
4. **易于维护**：一个文件，易于查看所有工具函数，减少查找成本

**何时考虑拆分**：

- 如果 `utils.ts` 超过 500 行
- 如果出现多个独立的工具类别（如网络请求、图片处理）

当前的复杂度下，单文件足够。

---

## 四、Scripts 与 Runtime 的边界

### 4.1 明确的边界规则

| 方向                    | 是否允许 | 说明                                                 |
| ----------------------- | -------- | ---------------------------------------------------- |
| Scripts import Runtime  | ✅ 部分  | 只能 import `src/config/paths.ts` 和 `src/lib/slug/` |
| Runtime import Scripts  | ❌ 禁止  | Runtime 不得依赖 scripts                             |
| Scripts import utils.ts | ✅ 允许  | Scripts 的共享工具层                                 |
| Runtime import utils.ts | ❌ 禁止  | `utils.ts` 仅供 scripts 使用                         |

### 4.2 共享模块的选择

**可被 Scripts 和 Runtime 共同使用**：

- `src/config/paths.ts`：路径配置（单一数据源）
- `src/lib/slug/`：slug 生成与验证（保证 URL 一致性）

**仅供 Runtime 使用**：

- `src/lib/content/`、`src/lib/markdown/`、`src/lib/site/`、`src/lib/ui/`

**仅供 Scripts 使用**：

- `scripts/utils.ts`

### 4.3 违反边界的后果

如果 Runtime 导入 Scripts：

- 破坏单向数据流
- 引入不必要的依赖（Playwright、Notion SDK 等）
- 增大构建产物体积
- 混淆职责边界

如果 Scripts 导入过多 Runtime 代码：

- 耦合过紧，难以独立演进
- 增加测试复杂度

---

## 五、添加新 Script 的指南

### 5.1 创建新脚本的步骤

1. **在 `scripts/` 目录创建新文件**（如 `scripts/my-script.ts`）

2. **导入必要的模块**：

   ```typescript
   import { ensureDir } from './utils';
   import { BLOG_CONTENT_DIR, PUBLIC_IMAGES_DIR } from '../src/config/paths';
   import { slugFromTitle } from '../src/lib/slug';
   ```

3. **实现主函数**：

   ```typescript
   async function main() {
     // 脚本逻辑
   }
   ```

4. **添加命令行参数解析**（如需要）：

   ```typescript
   function parseArgs() {
     const target = process.argv
       .find((arg) => arg.startsWith('--target='))
       ?.slice('--target='.length);
     // ...
     return { target };
   }
   ```

5. **使用 `runMain` 包装**（可选，统一错误处理）：

   ```typescript
   import { runMain } from './utils';
   runMain(main);
   ```

6. **在 `package.json` 添加 npm script**：

   ```json
   {
     "scripts": {
       "my-script": "tsx scripts/my-script.ts"
     }
   }
   ```

7. **在本文档中添加说明**（本 README.md）

### 5.2 脚本设计原则

- **单一职责**：每个脚本只做一件事
- **幂等性**：可安全地多次运行
- **错误处理**：清晰的错误信息，适当的退出码
- **参数验证**：检查必需参数，提供有用的帮助信息
- **日志输出**：输出进度和结果，便于调试
- **文件安全**：使用 `ensureDir` 创建目录，避免覆盖重要文件

### 5.3 何时添加到 `utils.ts`

当满足以下条件时，考虑将逻辑提取到 `utils.ts`：

- 被 2 个及以上 script 使用
- 是纯函数（无副作用）或封装的文件操作
- 与特定业务逻辑无关
- 不超过 50 行代码

**不要**：

- 将单个 script 的主逻辑放入 `utils.ts`
- 将 Runtime 逻辑放入 `utils.ts`

---

## 六、常见问题

### Q1：为什么 Notion sync 会覆盖我的本地修改？

**A**：Notion sync 将 Notion 视为单一数据源，每次同步会重写 `src/content/blog/notion/` 中的所有文件。

**解决方案**：

- 在 Notion 中编辑内容，不要本地编辑
- 如果需要本地编辑，将文章移出 `notion/` 目录

### Q2：如何避免导入重复的文章？

**A**：默认情况下，`content-import.ts` 不会覆盖已存在的同 slug 文章。

**如需覆盖**：使用 `--allow-overwrite` 参数

### Q3：数学公式格式为什么需要修正？

**A**：Notion 导出的 Markdown 中，数学公式可能包含多余空格（`$ x $`），导致 KaTeX 无法正确渲染。

**修正后**：`$x$` 可以正确渲染。

### Q4：删除文章后能否恢复？

**A**：删除操作不可逆，建议：

- 先使用 `--dry-run` 预览
- 确保有 Git 提交记录，可以通过 Git 回滚

### Q5：Scripts 可以在本地运行吗？

**A**：可以。Scripts 设计为本地可运行：

- 配置 `.env.local` 中的环境变量
- 运行 `npm run <script-name>`

---

## 七、后续维护指南

### 修改现有 Script

1. **修改源码**（在 `scripts/` 目录）
2. **更新本文档中的对应章节**（参数说明必须与源码一致）
3. **如有必要，更新调用该 script 的 workflow**（参见 [docs/ci-workflow.md](../docs/ci-workflow.md)）
4. **本地测试**：确保修改不破坏现有功能

### 删除废弃 Script

1. 从 `scripts/` 目录删除文件
2. 从 `package.json` 删除对应的 npm script
3. 从本文档删除说明
4. 检查是否有 workflow 调用该 script，如有则一并删除

### 重构 Scripts

如果 scripts 复杂度增加，考虑：

- 将 `utils.ts` 拆分为多个模块（如 `utils/fs.ts`、`utils/markdown.ts`）
- 引入专门的 CLI 参数解析库（如 `commander`）
- 添加单元测试（在 `tests/unit/scripts/` 中）

**当前阶段**：保持简单，单文件 `utils.ts` 足够。
