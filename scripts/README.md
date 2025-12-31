# Scripts 使用说明

本文档是 **scripts 目录的权威说明**，包含所有脚本的功能、参数和使用场景。

> **相关文档**
>
> - CI 工作流 → [docs/ci-workflow.md](../docs/ci-workflow.md)
> - 仓库架构 → [docs/architecture.md](../docs/architecture.md)
> - 站点配置 → [docs/configuration.md](../docs/configuration.md)

---

## 一、Scripts 目录结构

```text
scripts/
├── notion-sync.ts       # Notion 内容同步
├── content-import.ts    # 外部文章导入
├── delete-article.ts    # 文章删除
├── config-audit.ts      # 配置审计
├── utils.ts             # 共享工具函数
├── logger-helpers.ts    # 日志辅助工具
├── import/              # 导入适配器
│   └── adapters/        # 平台适配器（zhihu、wechat、medium、others）
├── lib/                 # 通用库
│   └── markdown/        # 统一 Markdown 管线
│       └── pipeline.ts  # AST 解析与序列化
├── markdown/            # Markdown 处理管线
│   ├── index.ts         # 入口
│   ├── translator.ts    # 翻译接口
│   ├── deepseek-translator.ts  # DeepSeek 翻译器
│   ├── language-detector.ts    # 语言检测
│   └── code-fence-fixer.ts     # 代码块修复
└── logger/              # 日志工具
```

---

## 二、脚本功能详解

### 2.1 `notion-sync.ts`

**功能**：从 Notion 数据库同步已发布页面到博客。

**npm 命令**：`npm run notion:sync`

**环境变量（必需）**：

| 变量名               | 说明             |
| -------------------- | ---------------- |
| `NOTION_TOKEN`       | Notion API token |
| `NOTION_DATABASE_ID` | Notion 数据库 ID |

**环境变量（可选）**：

| 变量名                        | 默认值     | 说明                |
| ----------------------------- | ---------- | ------------------- |
| `MARKDOWN_TRANSLATE_ENABLED`  | `0`        | 启用翻译（`1`/`0`） |
| `MARKDOWN_TRANSLATE_PROVIDER` | `identity` | 翻译提供商          |
| `DEEPSEEK_API_KEY`            | —          | DeepSeek API key    |

**输出**：

- Markdown 文件：`src/content/blog/notion/<slug>.md`
- 图片文件：`public/images/notion/<slug>/<imageId>.<ext>`

**执行流程**：

1. 连接 Notion API，查询 status = "Published" 的页面
2. 生成 slug（基于标题，检测冲突）
3. 下载封面图和正文图片
4. 使用 `notion-to-md` 转换为 Markdown
5. 通过 markdown 管线处理（翻译、代码修复、数学公式修正）
6. 写入文件，运行 `npm run lint` 格式化

**幂等性**：可安全多次运行，已存在的文件会被覆盖。

**CI 调用** → [sync-notion.yml](../docs/ci-workflow.md#23-sync-notionyml--notion-同步)

---

### 2.2 `content-import.ts`

**功能**：从外部 URL 导入文章，支持知乎、微信、Medium 等平台。

**npm 命令**：`npm run import:content`

**命令行参数**：

| 参数                         | 类型    | 必需 | 默认值  | 说明           |
| ---------------------------- | ------- | ---- | ------- | -------------- |
| `--url`                      | string  | ✓    | —       | 文章 URL       |
| `--allow-overwrite`          | boolean | ✗    | `false` | 覆盖已存在文章 |
| `--dry-run`                  | boolean | ✗    | `false` | 预览模式       |
| `--use-first-image-as-cover` | boolean | ✗    | `false` | 首图作为封面   |

**环境变量**（可替代命令行参数）：

| 变量名                     | 对应参数                     |
| -------------------------- | ---------------------------- |
| `URL`                      | `--url`                      |
| `ALLOW_OVERWRITE`          | `--allow-overwrite`          |
| `DRY_RUN`                  | `--dry-run`                  |
| `USE_FIRST_IMAGE_AS_COVER` | `--use-first-image-as-cover` |

**环境变量（翻译，可选）**：

| 变量名                        | 默认值     | 说明             |
| ----------------------------- | ---------- | ---------------- |
| `MARKDOWN_TRANSLATE_ENABLED`  | `0`        | 启用翻译         |
| `MARKDOWN_TRANSLATE_PROVIDER` | `identity` | 翻译提供商       |
| `DEEPSEEK_API_KEY`            | —          | DeepSeek API key |

**支持的平台**：

| 平台   | URL 模式                 | 输出目录                   |
| ------ | ------------------------ | -------------------------- |
| 知乎   | `zhuanlan.zhihu.com/p/*` | `src/content/blog/zhihu/`  |
| 微信   | `mp.weixin.qq.com/s/*`   | `src/content/blog/wechat/` |
| Medium | `*.medium.com/*`         | `src/content/blog/medium/` |
| 其他   | 任意 URL                 | `src/content/blog/others/` |

**使用示例**：

```bash
# 导入知乎文章
npm run import:content -- --url="https://zhuanlan.zhihu.com/p/668888063"

# 覆盖已存在的文章
npm run import:content -- --url="<URL>" --allow-overwrite

# 预览模式
npm run import:content -- --url="<URL>" --dry-run
```

**CI 调用** → [import-content.yml](../docs/ci-workflow.md#24-import-contentyml--导入外部文章)

---

### 2.3 `delete-article.ts`

**功能**：删除指定文章及其关联图片目录。

**npm 命令**：`npm run delete:article`

**命令行参数**：

| 参数              | 类型    | 必需 | 默认值  | 说明             |
| ----------------- | ------- | ---- | ------- | ---------------- |
| `--target`        | string  | ✓    | —       | slug 或文件路径  |
| `--delete-images` | boolean | ✗    | `false` | 删除关联图片目录 |
| `--dry-run`       | boolean | ✗    | `false` | 预览模式         |

**环境变量**（可替代命令行参数）：

| 变量名          | 对应参数          |
| --------------- | ----------------- |
| `TARGET`        | `--target`        |
| `DELETE_IMAGES` | `--delete-images` |
| `DRY_RUN`       | `--dry-run`       |

**使用示例**：

```bash
# 通过 slug 删除
npm run delete:article -- --target=my-article-slug

# 通过路径删除
npm run delete:article -- --target=src/content/blog/wechat/my-article.md

# 删除文章及图片
npm run delete:article -- --target=my-slug --delete-images

# 预览模式
npm run delete:article -- --target=my-slug --delete-images --dry-run
```

**安全机制**：

- 只能删除 `src/content/blog/` 内的文件
- 如果多个文件匹配 slug，会提示选择
- 最多匹配 20 个图片目录（防止误删）

**CI 调用** → [delete-article.yml](../docs/ci-workflow.md#25-delete-articleyml--删除文章)

---

### 2.4 `config-audit.ts`

**功能**：检查 YAML 配置项的生效状态。

**npm 命令**：`npm run config:audit`

**输出**：配置项生效性报告，退出码 0 表示所有配置有效，1 表示发现问题。

> **详细说明** → [docs/config-audit.md](../docs/config-audit.md)

---

## 三、Markdown 处理管线

导入和同步时自动应用的 Markdown 增强功能。

### 3.1 统一管线 (`scripts/lib/markdown/pipeline.ts`)

Notion 同步使用基于 AST（抽象语法树）的统一管线，确保输出的 Markdown 语法合法、可渲染：

| 功能             | 说明                                      |
| ---------------- | ----------------------------------------- |
| 不可见字符清理   | 移除 bidi 控制字符、零宽字符、BOM         |
| Frontmatter 合并 | 解析-合并-序列化，保证 key 唯一           |
| 重复 key 处理    | 自动检测并移除重复的 frontmatter 键       |
| AST 解析与序列化 | 基于 remark-parse/stringify，保证语法正确 |
| 代码块稳定化     | 使用 fenced 格式，处理嵌套反引号          |
| URL 规范化       | 编码空格或使用尖括号语法                  |
| 空行压缩         | 3+ 连续空行压缩为 2                       |

**使用方式**：Notion 同步自动调用，无需手动配置。

### 3.2 增强功能列表

| 功能              | 说明                            |
| ----------------- | ------------------------------- |
| 语言检测          | 分析文章主体语言（英文/中文）   |
| 翻译              | 英文文章翻译为中文（可选）      |
| 代码块语言标注    | 自动推断并补齐缺失的语言标识符  |
| 图片 caption 处理 | 转换为 Markdown 斜体格式        |
| 数学公式修复      | 修正 `$ x $` → `$x$` 等格式问题 |
| 格式清理          | 压缩多余空行、统一换行符        |

### 3.3 翻译提供商

| Provider   | 说明                     |
| ---------- | ------------------------ |
| `identity` | 不翻译，保持原文（默认） |
| `deepseek` | 使用 DeepSeek API 翻译   |

**DeepSeek 环境变量**：

| 变量名                   | 必需 | 默认值                     |
| ------------------------ | ---- | -------------------------- |
| `DEEPSEEK_API_KEY`       | 是   | —                          |
| `DEEPSEEK_MODEL`         | 否   | `deepseek-chat`            |
| `DEEPSEEK_BASE_URL`      | 否   | `https://api.deepseek.com` |
| `DEEPSEEK_CACHE_ENABLED` | 否   | `1`                        |

---

## 四、Scripts 与仓库模块的关系

### 4.1 可导入的模块

| 模块                  | 用途            |
| --------------------- | --------------- |
| `src/config/paths.ts` | 路径配置        |
| `src/lib/slug/`       | Slug 生成与验证 |

### 4.2 依赖边界

- ✅ Scripts → `src/config/paths.ts`、`src/lib/slug/`
- ❌ Runtime → Scripts（禁止）
- ✅ Scripts → `scripts/utils.ts`（内部共享）

### 4.3 与 CI 的关系

| Workflow             | 调用的 Script               |
| -------------------- | --------------------------- |
| `sync-notion.yml`    | `scripts/notion-sync.ts`    |
| `import-content.yml` | `scripts/content-import.ts` |
| `delete-article.yml` | `scripts/delete-article.ts` |

> **详细 CI 说明** → [docs/ci-workflow.md](../docs/ci-workflow.md)

---

## 五、添加新 Script 的指南

### 5.1 步骤

1. 创建 `scripts/<script-name>.ts`
2. 导入必要模块：

   ```typescript
   import { BLOG_CONTENT_DIR } from '../src/config/paths';
   import { slugFromTitle } from '../src/lib/slug';
   ```

3. 实现主函数和参数解析
4. 在 `package.json` 添加 npm script：

   ```json
   {
     "scripts": {
       "my-script": "tsx scripts/my-script.ts"
     }
   }
   ```

5. 在本文档中添加说明

### 5.2 设计原则

- **单一职责**：每个脚本只做一件事
- **幂等性**：可安全多次运行
- **错误处理**：清晰的错误信息，适当的退出码
- **参数验证**：检查必需参数，提供帮助信息
- **日志输出**：输出进度和结果

---

## 六、常见问题

### Q1：为什么 Notion sync 会覆盖本地修改？

Notion sync 将 Notion 视为单一数据源。如需本地编辑，将文章移出 `notion/` 目录。

### Q2：如何避免导入重复文章？

默认不覆盖同 slug 文章。如需覆盖，使用 `--allow-overwrite`。

### Q3：删除文章后能否恢复？

删除不可逆，建议先用 `--dry-run` 预览，或通过 Git 回滚。

### Q4：Scripts 可以在本地运行吗？

可以。配置 `.env.local` 中的环境变量，然后运行 `npm run <script-name>`。
