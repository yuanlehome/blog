# 搜索功能指南 (Search Feature Guide)

本指南介绍博客的全文搜索功能配置与使用。

## 目录

1. [功能概述](#功能概述)
2. [快速开始](#快速开始)
3. [配置说明](#配置说明)
4. [搜索索引生成](#搜索索引生成)
5. [性能优化](#性能优化)
6. [常见问题](#常见问题)

## 功能概述

博客搜索功能提供以下特性：

- **快捷键支持**：全站可用 `⌘K` (Mac) 或 `Ctrl+K` (Windows/Linux) 打开搜索
- **模糊匹配**：支持拼写容错，1-2 个字符差错仍能搜到
- **中英文混合**：完整支持中文和英文内容搜索
- **智能排序**：标题匹配优先于正文匹配
- **标签过滤**：支持按标签筛选结果
- **高亮显示**：搜索结果中高亮命中词
- **键盘导航**：支持上下键选择，回车跳转
- **懒加载**：仅在打开搜索时加载索引，不影响首屏性能
- **Web Worker**：搜索在后台线程执行，不阻塞主线程

## 快速开始

### 启用搜索

搜索功能默认启用。如需关闭，编辑 `src/config/yaml/search.yml`：

```yaml
search:
  enabled: false
```

### 使用搜索

1. 按 `⌘K` 或 `Ctrl+K` 打开搜索面板
2. 输入关键词进行搜索
3. 使用 `↑` `↓` 键选择结果
4. 按 `Enter` 跳转到选中的文章
5. 按 `ESC` 关闭搜索面板

## 配置说明

搜索配置文件位于 `src/config/yaml/search.yml`。

### 基础配置

| 字段         | 类型    | 默认值 | 说明                       |
| ------------ | ------- | ------ | -------------------------- |
| `enabled`    | boolean | true   | 是否启用搜索功能           |
| `shortcut`   | boolean | true   | 是否启用快捷键 (⌘K/Ctrl+K) |
| `provider`   | string  | "fuse" | 搜索引擎：fuse             |
| `lazyLoad`   | boolean | true   | 懒加载搜索索引             |
| `useWorker`  | boolean | true   | 使用 Web Worker            |
| `maxResults` | number  | 12     | 最大结果数量 (1-100)       |

### 摘要配置

```yaml
snippet:
  window: 80 # 匹配词周围显示的字符数 (20-200)
  maxLines: 2 # 摘要最大行数 (1-5)
```

### 权重配置

权重值越高，匹配结果排名越靠前。

```yaml
weights:
  title: 6 # 标题权重（最高）
  headings: 3 # 标题权重
  tags: 3 # 标签权重
  summary: 2 # 摘要权重
  body: 1 # 正文权重（最低）
```

### Fuse.js 配置

```yaml
fuse:
  threshold: 0.35 # 模糊匹配阈值 (0=精确, 1=模糊)
  minMatchCharLength: 1 # 最小匹配字符数
  maxPatternLength: 32 # 最大模式长度
  ignoreLocation: true # 忽略匹配位置
  includeMatches: true # 包含匹配信息（用于高亮）
  includeScore: true # 包含分数（用于排序）
```

### 过滤配置

```yaml
filters:
  tags: true # 启用标签过滤
  year: false # 启用年份过滤（暂不支持）
  source: false # 启用来源过滤（暂不支持）
```

### UI 配置

```yaml
ui:
  placement: 'modal' # 面板类型：modal（弹窗）
  showTags: true # 结果中显示标签
  showDate: true # 结果中显示日期
  placeholder: '搜索文章...' # 输入框占位文本
  noResultsText: '未找到相关文章' # 无结果提示
  loadingText: '加载中...' # 加载提示
  recentTitle: '最近文章' # 最近文章标题
  tagsTitle: '热门标签' # 标签区域标题
  closeText: '关闭' # 关闭按钮文本
  hintText: '按 ESC 关闭' # 提示文本
```

### 完整配置示例

```yaml
search:
  enabled: true
  shortcut: true
  provider: 'fuse'
  lazyLoad: true
  useWorker: true
  maxResults: 12

  snippet:
    window: 80
    maxLines: 2

  weights:
    title: 6
    headings: 3
    tags: 3
    summary: 2
    body: 1

  fuse:
    threshold: 0.35
    minMatchCharLength: 1
    maxPatternLength: 32
    ignoreLocation: true
    includeMatches: true
    includeScore: true

  filters:
    tags: true

  ui:
    placement: 'modal'
    showTags: true
    showDate: true
    placeholder: '搜索文章...'
```

## 搜索索引生成

### 自动生成

搜索索引在构建时自动生成：

```bash
npm run build
```

这会先执行 `npm run search:index` 生成索引，然后构建站点。

### 手动生成

单独生成索引：

```bash
npm run search:index
```

输出示例：

```text
🔍 Generating search index...
📄 Found 14 markdown files
✅ Processed 14 published posts
📦 Search index written to public/search-index.json (347.83 KB)
🏷️  Tags: 5
```

### 索引内容

索引包含以下信息：

- `slug`：文章唯一标识
- `url`：文章 URL
- `title`：标题
- `headings`：所有标题（H1-H6）
- `tags`：标签列表
- `date`：发布日期
- `summary`：摘要（前 200 字符）
- `body`：正文纯文本（最多 20000 字符）
- `source`：来源（notion/wechat/others）

### 索引优化

生成索引时会自动：

1. 移除代码块（不参与搜索）
2. 移除 HTML 标签
3. 移除 Markdown 格式符号
4. 截断过长的正文
5. 提取所有标题用于搜索

## 性能优化

### 懒加载

默认启用懒加载，搜索索引仅在用户首次打开搜索面板时加载。

### Web Worker

搜索计算在 Web Worker 中执行，不阻塞主线程，保证 UI 流畅。

### 索引体积控制

- 正文最多保留 20000 字符
- 移除代码块减少体积
- 摘要最多 200 字符

### 首屏不加载

搜索相关资源不会在首屏加载，不影响首页性能。

## 常见问题

### 中文输入时搜索异常

搜索组件已处理 IME 组合输入（compositionstart/compositionend），正常情况下不会出现问题。如遇异常，请检查浏览器版本。

### 搜索无结果

1. 检查关键词是否正确
2. 尝试使用更短的关键词
3. 检查是否有标签过滤生效
4. 确认文章状态为 `published`

### 索引体积过大

如果索引文件过大（超过 500KB）：

1. 减少 `maxPatternLength`
2. 在构建脚本中调整正文截断长度
3. 考虑只索引标题和摘要

### 搜索结果不准确

调整 `threshold` 值：

- 降低值（如 0.2）：更精确的匹配
- 提高值（如 0.5）：更宽松的匹配

### 浏览器不支持 Web Worker

组件会自动回退到主线程搜索，并添加防抖处理。

### 移动端使用

移动端搜索面板会全屏显示，支持触摸滚动。

### 禁用快捷键

如果快捷键与其他功能冲突：

```yaml
search:
  shortcut: false
```

用户仍可通过点击导航栏搜索按钮打开搜索。

## 技术实现

搜索功能基于以下技术栈：

- **Fuse.js**：模糊搜索引擎
- **Web Worker**：后台搜索处理
- **构建时索引**：预生成 JSON 索引
- **Astro 组件**：SearchModal.astro

相关文件：

- `src/config/yaml/search.yml`：配置文件
- `src/config/loaders/search.ts`：配置加载器
- `src/lib/search/`：搜索库
- `src/components/SearchModal.astro`：搜索组件
- `scripts/generate-search-index.ts`：索引生成脚本
