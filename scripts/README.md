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
   - 通过 markdown pipeline 自动处理（翻译、代码语言检测、数学公式修正等）
   - 写入 `src/content/blog/notion/<slug>.md`
3. 运行 `npm run lint` 格式化生成的文件

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
8. 通过 markdown pipeline 自动处理（翻译、代码语言检测、数学公式修正等）
9. 写入 MDX 文件到 `src/content/blog/<platform>/<slug>.mdx`
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

### 2.3 `delete-article.ts`

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

### 2.5 Markdown 导入增强：翻译与格式化

#### 功能

自动增强导入/同步的 Markdown 文章，包括：

- **自动翻译**：检测英文文章并翻译为中文
- **代码块语言标注**：自动推断并补齐缺失的语言标识符
- **图片 caption 规范化**：转换为标准 HTML figure 结构
- **格式清理**：压缩多余空行、统一换行符

#### 使用场景

- 导入英文技术文章时自动翻译
- 修复从 Notion 或其他平台导入的格式问题
- 统一文章格式规范

#### 何时触发

**自动触发**：集成在以下脚本中，无需手动调用

- `notion-sync.ts`：Notion 同步时自动应用
- `content-import.ts`：内容导入时自动应用

#### 配置

**环境变量**：

- `MARKDOWN_TRANSLATE_ENABLED`：是否启用翻译功能
  - `1`：启用翻译（默认）
  - `0` 或未设置：禁用翻译
- `MARKDOWN_TRANSLATE_PROVIDER`：翻译提供商
  - `mock`：测试用翻译器（默认）
  - `identity`/`none`：不翻译，保持原文
  - `deepseek`：使用 DeepSeek 真实翻译（需配置 API key）

**配置示例**：

```bash
# .env.local - 测试环境
MARKDOWN_TRANSLATE_ENABLED=1
MARKDOWN_TRANSLATE_PROVIDER=mock

# .env.local - 生产环境（使用 DeepSeek）
MARKDOWN_TRANSLATE_ENABLED=1
MARKDOWN_TRANSLATE_PROVIDER=deepseek
DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxx
```

#### DeepSeek 翻译提供商

**功能特性**：

- 使用 DeepSeek 最先进模型进行高质量翻译
- AST + JSON patch 策略，严格保护代码和 URL
- 自动分批处理长文档
- 并发控制和超时保护
- 失败自动降级（保留原文）
- 可选的文件缓存（避免重复翻译）

**环境变量配置**：

| 变量名                        | 必需 | 默认值                      | 说明                               |
| ----------------------------- | ---- | --------------------------- | ---------------------------------- |
| `DEEPSEEK_API_KEY`            | 是   | 无                          | DeepSeek API 密钥                  |
| `DEEPSEEK_MODEL`              | 否   | `deepseek-chat`             | DeepSeek 模型名称                  |
| `DEEPSEEK_BASE_URL`           | 否   | `https://api.deepseek.com`  | DeepSeek API 基础 URL              |
| `DEEPSEEK_REQUEST_TIMEOUT_MS` | 否   | `60000`                     | 请求超时时间（毫秒）               |
| `DEEPSEEK_MAX_BATCH_CHARS`    | 否   | `6000`                      | 单批次最大字符数                   |
| `DEEPSEEK_MAX_CONCURRENCY`    | 否   | `2`                         | 最大并发请求数                     |
| `DEEPSEEK_CACHE_ENABLED`      | 否   | `1`                         | 是否启用缓存（`1` 启用，`0` 禁用） |
| `DEEPSEEK_CACHE_DIR`          | 否   | `.cache/markdown-translate` | 缓存目录路径                       |

**使用方法**：

```bash
# 1. 配置环境变量（.env.local）
MARKDOWN_TRANSLATE_ENABLED=1
MARKDOWN_TRANSLATE_PROVIDER=deepseek
DEEPSEEK_API_KEY=sk-your-api-key-here
DEEPSEEK_MODEL=deepseek-chat

# 2. 导入或同步内容（自动使用 DeepSeek 翻译）
npm run notion:sync
# 或
npm run import:content -- --url="https://example.com/article"

# 3. 查看翻译诊断信息（输出示例）
Enhanced my-article:
  - Translated from en (provider: deepseek, model: deepseek-chat)
  - Batches: 3, Success: 3, Failed: 0, Cache hits: 0
  - Fixed 2 code fences
  - Fixed 1 image captions
```

**翻译策略**：

DeepSeek 翻译器遵循严格的翻译规则：

1. **严格 JSON 输出**：要求模型返回 `{"patches": {"node-id": "译文", ...}}` 格式
2. **保护代码和特殊内容**：
   - 不翻译代码块、行内代码
   - 不翻译 URL、路径、变量名、函数名
   - 保留 frontmatter 不变
3. **技术术语处理**：首次出现使用"中文（English）"，后续只用中文
4. **失败降级**：任何批次失败都回退为原文，不影响其他部分
5. **缓存优化**：相同内容不重复翻译，节省 API 费用

**安全性**：

- API key 不会在日志中打印
- 日志只包含 provider、model、批次数、失败原因（不含文章内容）
- 缓存文件存储在本地 `.cache/` 目录（已加入 `.gitignore`）

**注意事项**：

- **CI/测试环境**：默认不使用 DeepSeek（避免产生 API 费用），使用 `mock` 翻译器
- **本地开发**：需手动配置 `DEEPSEEK_API_KEY` 才能使用
- **网络依赖**：DeepSeek 翻译需要互联网连接，若请求失败会自动降级
- **成本控制**：启用缓存（默认启用）可避免重复翻译相同内容
- **模型选择**：`deepseek-chat` 为推荐模型，成本和质量平衡较好

#### 功能详解

##### A. 语言检测

- 自动分析文章主体语言（英文/中文）
- 排除代码块、URL、行内代码的干扰
- 英文字符占比 ≥ 60% 时触发翻译

##### B. 翻译（仅在检测到英文时）

- 只翻译自然语言内容：标题、段落、列表文本
- 严格保护：代码块、行内代码、URL、图片链接、frontmatter 不翻译
- 使用 AST + patch 策略确保结构稳定
- 失败降级：翻译失败时保留原文并继续其他修复

##### C. 代码块语言标注

支持自动检测的语言（15+ 种）：

- **Shebang 检测**：Python、Bash、Node.js
- **关键字检测**：
  - `python`：def, class, import, from...import
  - `javascript`：function, const, let, var, console.log
  - `typescript`：interface, type, enum, 类型注解
  - `bash`：echo, cd, if [[, ${}
  - `go`：package, func, defer, chan
  - `rust`：fn, let mut, impl, trait
  - `cpp`：#include, std::, namespace
  - `java`：public class, System.out
  - `dockerfile`：FROM, RUN, COPY, CMD
  - `sql`：SELECT, FROM, WHERE, JOIN
  - `yaml`：结构化 key-value（包括 GitHub Actions）
  - `json`：JSON 结构
  - `html`：HTML 标签
  - `css`：CSS 选择器和属性

##### D. 图片 caption 处理

- 识别图片后紧跟的短文本（≤ 120 字符）作为 caption
- 转换为 HTML `<figure>` 结构：

  ```html
  <figure>
    <img src="..." alt="..." />
    <figcaption>caption text</figcaption>
  </figure>
  ```

- 误判保护：
  - 超长段落不作为 caption
  - 标题、列表、代码块不作为 caption
  - 避免破坏文档结构

##### E. Markdown 格式清理

- 压缩 3+ 连续空行为 2 行
- 统一为 LF 换行符（`\n`）
- 规范标题和代码块间距

##### F. Frontmatter 增强

翻译后自动添加元数据：

```yaml
lang: zh
translatedFrom: en
```

#### 诊断输出

增强完成后会输出诊断信息：

```text
Enhanced my-article:
  - Translated from en
  - Fixed 3 code fences
  - Fixed 2 image captions
```

#### 模块架构

实现位于 `scripts/markdown/`：

- `language-detector.ts`：语言检测
- `translator.ts`：翻译接口与实现
- `code-fence-fixer.ts`：代码语言检测
- `markdown-processor.ts`：主处理管线
- `index.ts`：导出接口

#### 编程接口

可在其他脚本中使用：

```typescript
import { processMarkdownForImport } from './markdown';

const result = await processMarkdownForImport(
  { markdown: content, slug: 'my-article', source: 'notion' },
  {
    enableTranslation: true, // 启用翻译
    enableCodeFenceFix: true, // 修复代码块
    enableImageCaptionFix: true, // 修复图片 caption
    enableMarkdownCleanup: true, // 清理格式
    enableMathDelimiterFix: true, // 修复数学公式（如 `$ x $` → `$x$`）
  },
);

console.log(result.diagnostics); // 查看修改统计
```

#### 注意事项

- **测试环境**：默认使用 `mock` 翻译器，不依赖外部 API
- **生产环境**：需要配置真实翻译提供商（未来支持）
- **翻译质量**：Mock 翻译器仅用于测试，生产环境需使用真实 LLM
- **失败降级**：任何步骤失败都不会中断流程，会继续其他修复

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
