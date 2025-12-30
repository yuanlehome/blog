# 站点配置指南

本文档介绍如何通过 YAML 配置文件自定义博客 UI 和功能。

> **相关文档**
>
> - 配置治理 → [config-audit.md](./config-audit.md)
> - 仓库架构 → [architecture.md](./architecture.md)

---

## 一、配置文件位置

所有 YAML 配置文件位于 `src/config/yaml/` 目录：

```text
src/config/yaml/
├── site.yml         # 站点全局配置
├── nav.yml          # 导航菜单配置
├── home.yml         # 首页配置
├── post.yml         # 文章页配置
├── theme.yml        # 主题与配色配置
├── layout.yml       # 布局与结构配置
├── typography.yml   # 字体与排版配置
├── components.yml   # 组件样式配置
└── profile.yml      # 个人资料配置
```

---

## 二、配置文件说明

### 2.1 站点配置（`site.yml`）

| 字段              | 类型    | 默认值                   | 说明                 |
| ----------------- | ------- | ------------------------ | -------------------- |
| `siteName`        | string  | "Yuanle Liu's Blog"      | 网站名称             |
| `title`           | string  | "Yuanle Liu's Blog"      | 网站标题             |
| `description`     | string  | "A minimal Astro blog"   | 网站描述（SEO）      |
| `author`          | string  | "Yuanle Liu"             | 作者名称             |
| `copyrightYear`   | number  | 2025                     | 版权年份             |
| `copyrightText`   | string  | "All rights reserved."   | 版权文本             |
| `defaultLanguage` | string  | "en"                     | 默认语言             |
| `dateFormat`      | string  | "YYYY-MM-DD"             | 日期格式             |
| `enableRSS`       | boolean | true                     | 启用 RSS             |
| `enableSitemap`   | boolean | true                     | 启用站点地图         |

### 2.2 导航配置（`nav.yml`）

```yaml
header:
  brandText: '品牌文字'
  menuItems:
    - label: '首页'
      href: '/'
      isExternal: false
    - label: 'GitHub'
      href: 'https://github.com/user'
      isExternal: true
      openInNewTab: true

theme:
  enableToggle: true
  showLabel: true
```

### 2.3 首页配置（`home.yml`）

| 字段                       | 类型    | 默认值            | 说明             |
| -------------------------- | ------- | ----------------- | ---------------- |
| `title`                    | string  | "Recent Posts"    | 首页标题         |
| `showPostCount`            | boolean | true              | 显示文章总数     |
| `postCountText`            | string  | "published posts" | 文章计数后缀     |
| `pagination.pageSize`      | number  | 5                 | 每页文章数       |
| `pagination.windowSize`    | number  | 5                 | 分页窗口大小     |
| `navigation.newerText`     | string  | "← Newer"         | 更新按钮文本     |
| `navigation.olderText`     | string  | "Older →"         | 更早按钮文本     |

### 2.4 文章页配置（`post.yml`）

| 字段                        | 类型    | 默认值   | 说明           |
| --------------------------- | ------- | -------- | -------------- |
| `metadata.showPublishedDate`| boolean | true     | 显示发布日期   |
| `metadata.showUpdatedDate`  | boolean | true     | 显示更新日期   |
| `metadata.showReadingTime`  | boolean | true     | 显示阅读时间   |
| `metadata.showWordCount`    | boolean | true     | 显示字数       |
| `tableOfContents.enable`    | boolean | true     | 启用目录       |
| `tableOfContents.defaultExpanded` | boolean | false | 默认展开     |
| `floatingActions.enableToc` | boolean | true     | 目录浮动按钮   |
| `floatingActions.enableTop` | boolean | true     | 返回顶部按钮   |
| `comments.enable`           | boolean | true     | 启用评论       |
| `comments.provider`         | string  | "giscus" | 评论服务商     |

### 2.5 布局配置（`layout.yml`）

| 字段                        | 类型    | 默认值         | 说明               |
| --------------------------- | ------- | -------------- | ------------------ |
| `layoutMode`                | enum    | "rightSidebar" | 布局模式           |
| `sidebar.enabled`           | boolean | true           | 启用侧边栏         |
| `sidebar.position`          | enum    | "right"        | 侧边栏位置         |
| `alignment.headerAlign`     | enum    | "left"         | 头部对齐           |
| `alignment.footerAlign`     | enum    | "left"         | 底部对齐           |
| `alignment.postMetaAlign`   | enum    | "left"         | 文章元信息对齐     |

**layoutMode 可选值**：`centered`、`rightSidebar`、`leftSidebar`

### 2.6 主题配置（`theme.yml`）

| 字段                           | 类型    | 默认值         | 说明               |
| ------------------------------ | ------- | -------------- | ------------------ |
| `colorMode.default`            | enum    | "system"       | 默认主题           |
| `colorMode.allowToggle`        | boolean | true           | 允许切换           |
| `colors.brand`                 | color   | "#3b82f6"      | 品牌色             |
| `colors.accent`                | color   | "#8b5cf6"      | 强调色             |
| `codeBlock.showLineNumbers`    | boolean | true           | 代码行号           |
| `codeBlock.showCopyButton`     | boolean | true           | 复制按钮           |
| `header.variant`               | enum    | "default"      | 页头样式           |

---

## 三、环境变量

在 `.env.local` 中配置敏感信息和环境相关配置：

```bash
# Notion 同步（本地开发时需要）
NOTION_TOKEN=secret_your_token_here
NOTION_DATABASE_ID=your_database_id_here

# 翻译功能（可选）
DEEPSEEK_API_KEY=sk-your-key-here

# 站点配置（可选覆盖）
SITE_BASE=/blog/
SITE_URL=https://example.github.io/blog
```

> **CI 环境变量配置** → [ci-workflow.md](./ci-workflow.md#四ci-安全策略)

---

## 四、配置验证

所有配置使用 Zod schema 验证。构建时会显示详细错误信息。

### 验证配置

```bash
npm run check
```

### 常见错误

1. **类型错误**：确保数字字段使用数字，不加引号
2. **必填字段缺失**：添加缺失的必填字段
3. **枚举值错误**：使用文档中列出的有效值
4. **URL 格式错误**：URL 需以 `http://` 或 `https://` 开头

---

## 五、常见问题

### 如何修改每页文章数？

编辑 `home.yml`：

```yaml
pagination:
  pageSize: 10
```

### 如何添加导航菜单项？

编辑 `nav.yml`，在 `menuItems` 中添加：

```yaml
- label: '新页面'
  href: '/new-page/'
  isExternal: false
```

### 如何关闭评论？

编辑 `post.yml`：

```yaml
comments:
  enable: false
```

或在单篇文章 frontmatter 中：

```yaml
---
comments: false
---
```

### 修改配置后需要重启吗？

是的，YAML 配置在构建时加载，修改后需重启 `npm run dev`。
