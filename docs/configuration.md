# 博客配置指南 (Blog Configuration Guide)

本指南介绍如何通过 YAML 配置文件自定义博客的 UI 和功能，无需修改源代码。

## 目录 (Table of Contents)

1. [配置文件位置](#配置文件位置)
2. [站点配置 (site.yml)](#站点配置-siteyml)
3. [导航配置 (nav.yml)](#导航配置-navyml)
4. [首页配置 (home.yml)](#首页配置-homeyml)
5. [文章页配置 (post.yml)](#文章页配置-postyml)
6. [主题配置 (theme.yml)](#主题配置-themeyml)
7. [布局配置 (layout.yml)](#布局配置-layoutyml)
8. [排版配置 (typography.yml)](#排版配置-typographyyml)
9. [组件配置 (components.yml)](#组件配置-componentsyml)
10. [个人资料配置 (profile.yml)](#个人资料配置-profileyml)
11. [配置验证](#配置验证)
12. [常用场景示例](#常用场景示例)
13. [常见问题](#常见问题)

## 配置文件位置

所有配置文件位于 `src/config/yaml/` 目录下：

```text
src/config/yaml/
├── site.yml        # 站点全局配置
├── nav.yml         # 导航菜单配置
├── home.yml        # 首页配置
├── post.yml        # 文章页配置
├── theme.yml       # 主题配色与代码高亮
├── layout.yml      # 布局与间距（NEW）
├── typography.yml  # 字体与排版（NEW）
├── components.yml  # 组件样式（NEW）
└── profile.yml     # 个人资料配置
```

## 站点配置 (site.yml)

站点级别的全局设置。

### 字段说明

| 字段              | 类型    | 默认值                   | 说明                 |
| ----------------- | ------- | ------------------------ | -------------------- |
| `siteName`        | string  | "Yuanle Liu's Blog"      | 网站名称             |
| `title`           | string  | "Yuanle Liu's Blog"      | 网站标题             |
| `description`     | string  | "A minimal Astro blog"   | 网站描述（用于 SEO） |
| `author`          | string  | "Yuanle Liu"             | 作者名称             |
| `copyrightYear`   | number  | 2025                     | 版权年份             |
| `copyrightText`   | string  | "All rights reserved."   | 版权文本             |
| `defaultLanguage` | string  | "en"                     | 默认语言             |
| `dateFormat`      | string  | "YYYY-MM-DD"             | 日期格式             |
| `enableRSS`       | boolean | true                     | 是否启用 RSS 订阅    |
| `enableSitemap`   | boolean | true                     | 是否启用站点地图     |
| `socialImage`     | string  | "placeholder-social.jpg" | 社交媒体分享图片     |

### 示例

```yaml
siteName: '我的技术博客'
title: '我的技术博客'
description: '分享技术文章和编程经验'
author: '张三'
copyrightYear: 2025
copyrightText: '保留所有权利。'
defaultLanguage: 'zh-CN'
enableRSS: true
enableSitemap: true
```

## 导航配置 (nav.yml)

配置网站顶部导航菜单。

### 字段说明

#### header

| 字段        | 类型   | 默认值              | 说明               |
| ----------- | ------ | ------------------- | ------------------ |
| `brandText` | string | "Yuanle Liu's Blog" | 品牌文字（左上角） |
| `menuItems` | array  | [...]               | 菜单项数组         |

#### menuItems 项

| 字段           | 类型    | 必填 | 说明               |
| -------------- | ------- | ---- | ------------------ |
| `label`        | string  | ✓    | 菜单项显示文本     |
| `href`         | string  | ✓    | 链接地址           |
| `isExternal`   | boolean | ✗    | 是否外部链接       |
| `openInNewTab` | boolean | ✗    | 是否在新标签页打开 |

#### theme

| 字段           | 类型    | 默认值 | 说明                 |
| -------------- | ------- | ------ | -------------------- |
| `enableToggle` | boolean | true   | 是否显示主题切换按钮 |
| `showLabel`    | boolean | true   | 是否显示主题标签     |
| `icons`        | object  | {...}  | 主题图标配置         |

### 示例

```yaml
header:
  brandText: '我的博客'
  menuItems:
    - label: '首页'
      href: '/'
      isExternal: false
    - label: '归档'
      href: '/archive/'
      isExternal: false
    - label: '关于'
      href: '/about/'
      isExternal: false
    - label: 'GitHub'
      href: 'https://github.com/username'
      isExternal: true
      openInNewTab: true

theme:
  enableToggle: true
  showLabel: true
  icons:
    light: '☀️'
    dark: '🌙'
    default: '🖥️'
```

## 首页配置 (home.yml)

配置博客首页和文章列表页面。

### 字段说明

| 字段            | 类型    | 默认值            | 说明             |
| --------------- | ------- | ----------------- | ---------------- |
| `title`         | string  | "Recent Posts"    | 首页标题         |
| `showPostCount` | boolean | true              | 是否显示文章总数 |
| `postCountText` | string  | "published posts" | 文章计数后缀文本 |

#### pagination

| 字段         | 类型   | 默认值 | 说明           |
| ------------ | ------ | ------ | -------------- |
| `pageSize`   | number | 5      | 每页显示文章数 |
| `windowSize` | number | 5      | 分页窗口大小   |

#### navigation

| 字段        | 类型   | 默认值    | 说明           |
| ----------- | ------ | --------- | -------------- |
| `newerText` | string | "← Newer" | "更新"按钮文本 |
| `olderText` | string | "Older →" | "更早"按钮文本 |
| `pageLabel` | string | "Page"    | 页码标签       |

### 示例

```yaml
title: '最新文章'
showPostCount: true
postCountText: '篇文章'

pagination:
  pageSize: 10 # 每页显示 10 篇文章
  windowSize: 7 # 显示 7 个页码

navigation:
  newerText: '← 较新'
  olderText: '较旧 →'
  pageLabel: '第'
```

## 文章页配置 (post.yml)

配置单篇文章页面的功能和显示。

### 字段说明

#### metadata

| 字段                | 类型    | 默认值   | 说明         |
| ------------------- | ------- | -------- | ------------ |
| `showPublishedDate` | boolean | true     | 显示发布日期 |
| `showUpdatedDate`   | boolean | true     | 显示更新日期 |
| `showReadingTime`   | boolean | true     | 显示阅读时间 |
| `showWordCount`     | boolean | true     | 显示字数     |
| `publishedLabel`    | string  | "发布于" | 发布日期标签 |
| `updatedLabel`      | string  | "更新于" | 更新日期标签 |

#### tableOfContents

| 字段              | 类型    | 默认值 | 说明           |
| ----------------- | ------- | ------ | -------------- |
| `enable`          | boolean | true   | 是否启用目录   |
| `defaultExpanded` | boolean | false  | 默认是否展开   |
| `showOnMobile`    | boolean | true   | 移动端是否显示 |
| `mobileTrigger`   | boolean | false  | 移动端触发器   |

#### floatingActions

| 字段           | 类型    | 默认值 | 说明             |
| -------------- | ------- | ------ | ---------------- |
| `enableToc`    | boolean | true   | 启用目录浮动按钮 |
| `enableTop`    | boolean | true   | 启用返回顶部按钮 |
| `enableBottom` | boolean | true   | 启用到底部按钮   |

#### comments (Giscus)

| 字段                | 类型    | 默认值    | 说明             |
| ------------------- | ------- | --------- | ---------------- |
| `enable`            | boolean | true      | 是否启用评论     |
| `defaultEnabled`    | boolean | true      | 文章默认开启评论 |
| `provider`          | string  | "giscus"  | 评论服务提供商   |
| `giscus.repo`       | string  | -         | GitHub 仓库      |
| `giscus.repoId`     | string  | -         | 仓库 ID          |
| `giscus.category`   | string  | "General" | 讨论分类         |
| `giscus.categoryId` | string  | -         | 分类 ID          |
| `giscus.lang`       | string  | "zh-CN"   | 界面语言         |

### 示例

```yaml
metadata:
  showPublishedDate: true
  showUpdatedDate: true
  showReadingTime: true
  showWordCount: true
  publishedLabel: '发布于'
  updatedLabel: '更新于'

tableOfContents:
  enable: true
  defaultExpanded: false
  showOnMobile: true

comments:
  enable: true
  defaultEnabled: true
  provider: 'giscus'
  giscus:
    repo: 'username/repo'
    repoId: 'YOUR_REPO_ID'
    category: 'General'
    categoryId: 'YOUR_CATEGORY_ID'
    lang: 'zh-CN'
```

## 主题配置 (theme.yml)

配置网站主题、配色方案和代码高亮。**此配置已扩展以支持完整的颜色系统和代码主题定制。**

### 字段说明

#### 主题模式

| 字段           | 类型   | 默认值            | 说明                        |
| -------------- | ------ | ----------------- | --------------------------- |
| `defaultTheme` | enum   | "system"          | 默认主题：light/dark/system |
| `themes`       | array  | ["light", "dark"] | 可用主题列表                |
| `storageKey`   | string | "theme"           | LocalStorage 键名           |

#### 图标与标签

| 字段    | 类型   | 默认值 | 说明         |
| ------- | ------ | ------ | ------------ |
| `light` | string | "☀️"   | 亮色主题图标 |
| `dark`  | string | "🌙"   | 暗色主题图标 |

#### 颜色系统 (colors)

完整的配色方案，支持 hex、rgb、hsl 格式：

| 字段                | 类型   | 默认值（亮色） | 说明         |
| ------------------- | ------ | -------------- | ------------ |
| `brand`             | string | #3b82f6        | 主品牌色     |
| `accent`            | string | #8b5cf6        | 强调色       |
| `background.base`   | string | #ffffff        | 页面背景     |
| `background.subtle` | string | #f8fafc        | 次级背景     |
| `background.muted`  | string | #f1f5f9        | 柔和背景     |
| `foreground.base`   | string | #0f172a        | 正文文本     |
| `foreground.muted`  | string | #64748b        | 次级文本     |
| `border.default`    | string | #e2e8f0        | 默认边框     |
| `border.subtle`     | string | #f1f5f9        | 柔和边框     |
| `card.background`   | string | #ffffff        | 卡片背景     |
| `card.border`       | string | #e2e8f0        | 卡片边框     |
| `code.background`   | string | #f8fafc        | 代码块背景   |
| `code.foreground`   | string | #0f172a        | 代码块文本   |
| `code.border`       | string | #e5e7eb        | 代码块边框   |
| `code.scrollbar`    | string | #cbd5e1        | 代码块滚动条 |

#### 暗色模式颜色 (darkColors)

暗色模式下的配色（结构与 colors 相同）

#### 代码主题 (codeTheme)

| 字段              | 类型    | 默认值       | 说明                         |
| ----------------- | ------- | ------------ | ---------------------------- |
| `light`           | string  | github-light | 亮色模式代码高亮主题         |
| `dark`            | string  | github-dark  | 暗色模式代码高亮主题         |
| `showLineNumbers` | boolean | true         | 是否显示行号                 |
| `showCopyButton`  | boolean | true         | 是否显示复制按钮             |
| `wrapLongLines`   | boolean | false        | 是否自动换行                 |
| `inlineCodeStyle` | enum    | subtle       | 行内代码样式：subtle / boxed |

#### 强调样式 (emphasis)

| 字段            | 类型    | 默认值 | 说明                               |
| --------------- | ------- | ------ | ---------------------------------- |
| `linkUnderline` | enum    | hover  | 链接下划线：never / hover / always |
| `focusRing`     | boolean | true   | 是否显示焦点环                     |

#### 动画 (animations)

| 字段                   | 类型    | 默认值 | 说明                 |
| ---------------------- | ------- | ------ | -------------------- |
| `respectReducedMotion` | boolean | true   | 尊重系统减少动画设置 |
| `enableScrollEffects`  | boolean | true   | 启用滚动效果         |

### 示例

```yaml
defaultTheme: 'system'
themes:
  - 'light'
  - 'dark'

# 自定义颜色
colors:
  brand: '#3b82f6'
  accent: '#8b5cf6'
  background:
    base: '#ffffff'
    subtle: '#f8fafc'
  foreground:
    base: '#0f172a'
    muted: '#64748b'

# 暗色模式颜色
darkColors:
  brand: '#60a5fa'
  accent: '#a78bfa'
  background:
    base: '#0f172a'
    subtle: '#1e293b'

# 代码高亮
codeTheme:
  light: 'github-light'
  dark: 'github-dark'
  showLineNumbers: true
  showCopyButton: true

emphasis:
  linkUnderline: 'hover'
  focusRing: true

animations:
  respectReducedMotion: true
  enableScrollEffects: true
```

## 布局配置 (layout.yml)

**NEW** 控制页面布局、容器宽度、侧边栏位置和对齐方式。

### 字段说明

#### 容器 (container)

| 字段               | 类型   | 默认值 | 说明             |
| ------------------ | ------ | ------ | ---------------- |
| `width`            | string | 72rem  | 最大内容宽度     |
| `paddingX.mobile`  | string | 1rem   | 移动端左右内边距 |
| `paddingX.tablet`  | string | 1.5rem | 平板左右内边距   |
| `paddingX.desktop` | string | 2rem   | 桌面端左右内边距 |

#### 布局模式 (layoutMode)

| 值             | 说明          |
| -------------- | ------------- |
| `centered`     | 单列居中布局  |
| `rightSidebar` | 内容 + 右侧栏 |
| `leftSidebar`  | 内容 + 左侧栏 |

#### 侧边栏 (sidebar)

| 字段       | 类型    | 默认值 | 说明                     |
| ---------- | ------- | ------ | ------------------------ |
| `enabled`  | boolean | true   | 是否启用侧边栏           |
| `position` | enum    | right  | 侧边栏位置：left / right |
| `width`    | string  | 18rem  | 侧边栏宽度               |
| `sticky`   | boolean | true   | 是否固定在屏幕           |
| `gap`      | string  | 3rem   | 与内容区的间距           |

#### 目录 (toc)

| 字段             | 类型    | 默认值  | 说明                                 |
| ---------------- | ------- | ------- | ------------------------------------ |
| `enabled`        | boolean | true    | 是否启用目录                         |
| `position`       | enum    | sidebar | 目录位置：sidebar / inline / hidden  |
| `mobileBehavior` | enum    | drawer  | 移动端行为：drawer / inline / hidden |
| `defaultOpen`    | boolean | false   | 默认是否打开（抽屉模式）             |
| `stickyOffset`   | number  | 96      | 固定时距顶部偏移（像素）             |

#### 对齐 (alignment)

| 字段            | 类型 | 默认值 | 说明                          |
| --------------- | ---- | ------ | ----------------------------- |
| `headerAlign`   | enum | left   | 页头对齐：left / center       |
| `footerAlign`   | enum | center | 页脚对齐：left / center       |
| `postMetaAlign` | enum | left   | 文章元信息对齐：left / center |
| `contentAlign`  | enum | left   | 内容对齐：left / center       |

### 示例

```yaml
# 右侧栏布局 + 目录
layoutMode: 'rightSidebar'

container:
  width: '72rem'
  paddingX:
    mobile: '1rem'
    tablet: '1.5rem'
    desktop: '2rem'

sidebar:
  enabled: true
  position: 'right'
  width: '18rem'
  sticky: true
  gap: '3rem'

toc:
  enabled: true
  position: 'sidebar'
  mobileBehavior: 'drawer'
  defaultOpen: false
  stickyOffset: 96

alignment:
  headerAlign: 'left'
  footerAlign: 'center'
  postMetaAlign: 'left'
  contentAlign: 'left'
```

## 排版配置 (typography.yml)

**NEW** 字体系统、字号、行高和排版规则。

### 字段说明

#### 字体族 (fontFamily)

| 字段    | 类型  | 说明               |
| ------- | ----- | ------------------ |
| `sans`  | array | 无衬线字体栈       |
| `serif` | array | 衬线字体栈         |
| `mono`  | array | 等宽字体栈（代码） |

#### 字号 (fontSize)

| 字段   | 默认值   | 说明     |
| ------ | -------- | -------- |
| `xs`   | 0.75rem  | 特小文本 |
| `sm`   | 0.875rem | 小文本   |
| `base` | 1rem     | 基础文本 |
| `lg`   | 1.125rem | 大文本   |
| `xl`   | 1.25rem  | 特大文本 |
| `2xl`  | 1.5rem   | 2X 大    |
| `3xl`  | 1.875rem | 3X 大    |
| `4xl`  | 2.25rem  | 4X 大    |

#### 行高 (lineHeight)

| 字段      | 默认值 | 说明         |
| --------- | ------ | ------------ |
| `tight`   | 1.25   | 紧凑（标题） |
| `snug`    | 1.375  | 略紧         |
| `normal`  | 1.5    | 正常         |
| `relaxed` | 1.625  | 宽松（正文） |
| `loose`   | 1.75   | 很宽松       |

#### 字重 (fontWeight)

| 字段       | 默认值 | 说明 |
| ---------- | ------ | ---- |
| `normal`   | 400    | 常规 |
| `medium`   | 500    | 中等 |
| `semibold` | 600    | 半粗 |
| `bold`     | 700    | 粗体 |

#### 文章排版 (prose)

| 字段                    | 默认值 | 说明             |
| ----------------------- | ------ | ---------------- |
| `maxWidth`              | 65ch   | 最大宽度         |
| `useSerif`              | false  | 是否使用衬线字体 |
| `paragraphSpacing`      | 1.25em | 段落间距         |
| `headingSpacing.before` | 1.5em  | 标题前间距       |
| `headingSpacing.after`  | 0.5em  | 标题后间距       |

### 示例

```yaml
fontFamily:
  sans:
    - 'ui-sans-serif'
    - 'system-ui'
    - 'sans-serif'
  mono:
    - '"Fira Code"'
    - 'Menlo'
    - 'monospace'

fontSize:
  base: '1rem'
  lg: '1.125rem'
  xl: '1.25rem'

lineHeight:
  normal: 1.5
  relaxed: 1.625

fontWeight:
  normal: 400
  bold: 700

prose:
  maxWidth: '65ch'
  useSerif: false
  paragraphSpacing: '1.25em'
```

## 组件配置 (components.yml)

**NEW** 组件视觉样式（圆角、阴影、边框、动画）。

### 字段说明

#### 圆角 (radius)

| 字段      | 默认值   | 说明     |
| --------- | -------- | -------- |
| `none`    | 0        | 无圆角   |
| `sm`      | 0.375rem | 小圆角   |
| `default` | 0.5rem   | 默认     |
| `md`      | 0.75rem  | 中等     |
| `lg`      | 0.9rem   | 大圆角   |
| `xl`      | 0.75rem  | 特大     |
| `full`    | 9999px   | 完全圆角 |

#### 组件圆角 (componentRadius)

| 字段     | 默认值  | 说明   |
| -------- | ------- | ------ |
| `card`   | 0.75rem | 卡片   |
| `button` | 0.5rem  | 按钮   |
| `image`  | 0.75rem | 图片   |
| `code`   | 0.9rem  | 代码块 |
| `input`  | 0.5rem  | 输入框 |

#### 阴影 (shadow)

| 字段      | 说明     |
| --------- | -------- |
| `none`    | 无阴影   |
| `sm`      | 小阴影   |
| `default` | 默认阴影 |
| `md`      | 中等阴影 |
| `lg`      | 大阴影   |
| `xl`      | 特大阴影 |
| `2xl`     | 超大阴影 |

#### 组件阴影 (componentShadow)

| 字段         | 说明             |
| ------------ | ---------------- |
| `card`       | 卡片阴影（亮色） |
| `cardDark`   | 卡片阴影（暗色） |
| `header`     | 页头阴影（亮色） |
| `headerDark` | 页头阴影（暗色） |
| `hoverLift`  | 是否启用悬停上浮 |

#### 边框 (border)

| 字段      | 类型   | 默认值 | 说明                              |
| --------- | ------ | ------ | --------------------------------- |
| `style`   | enum   | solid  | 边框样式：solid / dashed / dotted |
| `width`   | string | 1px    | 边框宽度                          |
| `opacity` | number | 0.2    | 边框透明度（0-1）                 |

#### 动画 (motion)

| 字段              | 类型    | 默认值 | 说明                              |
| ----------------- | ------- | ------ | --------------------------------- |
| `enabled`         | boolean | true   | 是否启用动画                      |
| `level`           | enum    | normal | 强度：subtle / normal / energetic |
| `duration.fast`   | number  | 150    | 快速动画时长（毫秒）              |
| `duration.normal` | number  | 200    | 正常动画时长（毫秒）              |
| `duration.slow`   | number  | 300    | 慢速动画时长（毫秒）              |
| `easing.default`  | string  | ease   | 缓动函数                          |

#### 间距缩放 (spacingScale)

| 值            | 说明 |
| ------------- | ---- |
| `compact`     | 紧凑 |
| `comfortable` | 舒适 |
| `relaxed`     | 宽松 |

### 示例

```yaml
# 圆角配置
componentRadius:
  card: '0.75rem'
  button: '0.5rem'
  code: '0.9rem'

# 阴影配置
componentShadow:
  card: '0 8px 24px rgb(15 23 42 / 0.08)'
  cardDark: '0 10px 30px rgb(0 0 0 / 0.28)'
  hoverLift: true

# 边框配置
border:
  style: 'solid'
  width: '1px'
  opacity: 0.2

# 动画配置
motion:
  enabled: true
  level: 'normal'
  duration:
    fast: 150
    normal: 200
    slow: 300

# 间距缩放
spacingScale: 'comfortable'
```

## 个人资料配置 (profile.yml)

配置关于页面的个人信息。

### 字段说明

| 字段          | 类型   | 必填 | 说明           |
| ------------- | ------ | ---- | -------------- |
| `name`        | string | ✓    | 姓名           |
| `bio`         | string | ✓    | 个人简介       |
| `socialLinks` | array  | ✓    | 社交链接数组   |
| `whatIDo`     | object | ✓    | "我做什么"部分 |
| `techStack`   | object | ✓    | 技术栈部分     |
| `journey`     | object | ✓    | 个人经历时间线 |

#### socialLinks 项

| 字段         | 类型         | 必填 | 说明       |
| ------------ | ------------ | ---- | ---------- |
| `name`       | string       | ✓    | 链接名称   |
| `url`        | string (URL) | ✓    | 链接地址   |
| `icon`       | string       | ✗    | 图标标识   |
| `colorClass` | string       | ✗    | CSS 颜色类 |

#### journey.items 项

| 字段          | 类型   | 必填 | 说明              |
| ------------- | ------ | ---- | ----------------- |
| `year`        | string | ✓    | 年份或时间段      |
| `role`        | string | ✓    | 角色/职位         |
| `description` | string | ✓    | 描述              |
| `color`       | string | ✗    | 时间点颜色 CSS 类 |

### 示例

```yaml
name: '张三'
bio: '全栈开发工程师，热爱开源'

socialLinks:
  - name: 'GitHub'
    url: 'https://github.com/username'
    colorClass: 'bg-gray-900 text-white hover:bg-gray-800'
  - name: 'Twitter'
    url: 'https://twitter.com/username'
    colorClass: 'bg-blue-500 text-white hover:bg-blue-600'

whatIDo:
  title: '我的工作'
  description: '构建高性能的 Web 应用和分布式系统'

techStack:
  title: '技术栈'
  skills:
    - 'TypeScript'
    - 'React'
    - 'Node.js'
    - 'Go'

journey:
  title: '我的经历'
  items:
    - year: '2023 - 至今'
      role: '高级工程师 @ 某公司'
      description: '负责核心系统架构设计'
      color: 'bg-blue-500'
    - year: '2020 - 2023'
      role: '软件工程师 @ 另一公司'
      description: '参与多个项目开发'
      color: 'bg-gray-300'
```

## 常用场景示例

### 场景 1：右侧栏 + 目录抽屉（当前默认）

保持现有 UI 效果，右侧显示目录，移动端使用抽屉：

```yaml
# layout.yml
layoutMode: 'rightSidebar'
sidebar:
  enabled: true
  position: 'right'
  width: '18rem'

toc:
  enabled: true
  position: 'sidebar'
  mobileBehavior: 'drawer'
```

### 场景 2：单列居中 + 宽内容

去除侧边栏，内容单列居中显示，适合纯阅读体验：

```yaml
# layout.yml
layoutMode: 'centered'
container:
  width: '80rem' # 更宽的容器

sidebar:
  enabled: false

toc:
  enabled: true
  position: 'inline' # 目录嵌入文章内部
  mobileBehavior: 'inline'
```

### 场景 3：自定义主色 + 紫色强调

使用自定义品牌色和强调色：

```yaml
# theme.yml
colors:
  brand: '#0ea5e9' # 天蓝色
  accent: '#a855f7' # 紫色

darkColors:
  brand: '#38bdf8' # 亮蓝色
  accent: '#c084fc' # 亮紫色
```

### 场景 4：更紧凑的间距

使用紧凑间距模式，适合信息密集型博客：

```yaml
# components.yml
spacingScale: 'compact'

# layout.yml
container:
  paddingX:
    mobile: '0.75rem'
    tablet: '1rem'
    desktop: '1.5rem'
```

### 场景 5：更舒适的阅读体验

使用更大的字号和宽松的行高：

```yaml
# typography.yml
fontSize:
  base: '1.125rem' # 18px
  lg: '1.25rem' # 20px

lineHeight:
  normal: 1.625 # 更宽松
  relaxed: 1.75

prose:
  maxWidth: '70ch' # 更宽的文章宽度
  paragraphSpacing: '1.5em' # 段落间距更大
```

### 场景 6：代码块自定义

调整代码块样式，隐藏行号，启用自动换行：

```yaml
# theme.yml
codeTheme:
  light: 'github-light'
  dark: 'one-dark-pro' # 使用不同的暗色主题
  showLineNumbers: false # 隐藏行号
  showCopyButton: true
  wrapLongLines: true # 启用自动换行

# components.yml
componentRadius:
  code: '0.5rem' # 更小的圆角
```

### 场景 7：极简风格

最小化视觉装饰，专注内容：

```yaml
# components.yml
componentRadius:
  card: '0.25rem' # 更小的圆角
  button: '0.25rem'
  code: '0.25rem'

componentShadow:
  card: '0 1px 3px rgb(0 0 0 / 0.1)' # 更淡的阴影
  hoverLift: false # 禁用悬停效果

border:
  opacity: 0.1 # 更淡的边框

# components.yml
motion:
  level: 'subtle' # 更微妙的动画
```

### 场景 8：左侧栏布局

将侧边栏和目录移到左侧：

```yaml
# layout.yml
layoutMode: 'leftSidebar'

sidebar:
  enabled: true
  position: 'left'
  width: '18rem'

toc:
  enabled: true
  position: 'sidebar'
```

## 配置验证

所有配置文件都使用 Zod 进行 schema 验证。如果配置无效，构建时会显示详细的错误信息。

### 常见验证错误

1. **字段类型错误**

   ```text
   Invalid configuration in site.yml:
     - copyrightYear: Expected number, received string
   ```

   解决：确保数字字段使用数字，不要用引号。

2. **必填字段缺失**

   ```text
   Invalid configuration in profile.yml:
     - name: Required
   ```

   解决：添加缺失的必填字段。

3. **URL 格式错误**

   ```text
   Invalid configuration in profile.yml:
     - socialLinks.0.url: Invalid url
   ```

   解决：确保 URL 以 `http://` 或 `https://` 开头。

4. **颜色格式错误**

   ```text
   Invalid configuration in theme.yml:
     - colors.brand: Invalid color format
   ```

   解决：使用有效的颜色格式（hex、rgb、hsl），例如 `#3b82f6`、`rgb(59, 130, 246)`、`hsl(217, 91%, 60%)`。

5. **枚举值无效**

   ```text
   Invalid configuration in layout.yml:
     - layoutMode: Invalid enum value
   ```

   解决：检查允许的枚举值，使用文档中列出的有效选项。

### 验证配置

运行以下命令检查配置是否有效：

```bash
npm run check
```

## 常见问题

### 如何修改每页显示的文章数？

编辑 `src/config/yaml/home.yml`：

```yaml
pagination:
  pageSize: 10 # 改为你想要的数字
```

### 如何添加新的导航菜单项？

编辑 `src/config/yaml/nav.yml`，在 `menuItems` 数组中添加：

```yaml
header:
  menuItems:
    # ... 现有项目
    - label: '新页面'
      href: '/new-page/'
      isExternal: false
```

### 如何关闭评论功能？

编辑 `src/config/yaml/post.yml`：

```yaml
comments:
  enable: false
```

或者在单篇文章的 frontmatter 中设置：

```yaml
---
comments: false
---
```

### 如何修改分页窗口大小？

编辑 `src/config/yaml/home.yml`：

```yaml
pagination:
  windowSize: 7 # 显示的页码数量
```

### 如何自定义主题颜色？

编辑 `src/config/yaml/theme.yml`：

```yaml
colors:
  brand: '#0ea5e9' # 自定义主色
  accent: '#a855f7' # 自定义强调色
  # ... 其他颜色

darkColors:
  brand: '#38bdf8' # 暗色模式主色
  accent: '#c084fc' # 暗色模式强调色
```

颜色会自动生成 CSS 变量并应用到整个站点。

### 如何切换布局模式？

编辑 `src/config/yaml/layout.yml`：

```yaml
# 单列居中（无侧边栏）
layoutMode: 'centered'

# 右侧栏（默认）
layoutMode: 'rightSidebar'

# 左侧栏
layoutMode: 'leftSidebar'
```

### 如何调整代码块样式？

编辑 `src/config/yaml/theme.yml`：

```yaml
codeTheme:
  showLineNumbers: true # 显示/隐藏行号
  showCopyButton: true # 显示/隐藏复制按钮
  wrapLongLines: false # 是否自动换行
  inlineCodeStyle: 'subtle' # 或 'boxed'
```

编辑 `src/config/yaml/components.yml`：

```yaml
componentRadius:
  code: '0.9rem' # 代码块圆角大小
```

### 如何使用不同的字体？

编辑 `src/config/yaml/typography.yml`：

```yaml
fontFamily:
  sans:
    - '"Custom Font"' # 你的自定义字体
    - 'system-ui'
    - 'sans-serif'
  mono:
    - '"Fira Code"' # 代码字体
    - 'monospace'
```

**注意**：需要确保字体已在系统中安装或通过 `@font-face` 加载。

### 如何调整容器宽度？

编辑 `src/config/yaml/layout.yml`：

```yaml
container:
  width: '80rem' # 更宽的容器（默认 72rem）
```

### 配置文件可以用环境变量吗？

配置文件在构建时加载，不支持运行时环境变量。如需动态配置，请使用 `.env` 文件配合代码逻辑。

### 修改配置后需要重启开发服务器吗？

是的。YAML 配置在构建时加载，修改后需要重启 `npm run dev`。

### 默认配置值是什么？

所有配置项都有默认值，对应当前 UI 的样式。如果不修改配置文件，网站将保持原有外观。

默认值已在各配置节的"字段说明"表格中列出。

### 如何重置配置为默认值？

删除或注释掉自定义的配置项，系统将自动使用默认值。或者参考 YAML 文件中的注释查看默认值。

### 颜色格式有什么要求？

支持以下格式：

- Hex: `#3b82f6` 或 `#3b82f6ff`（带 alpha）
- RGB: `rgb(59, 130, 246)` 或 `rgba(59, 130, 246, 0.8)`
- HSL: `hsl(217, 91%, 60%)` 或 `hsla(217, 91%, 60%, 0.8)`

**不支持**颜色关键字（如 `blue`、`red`）。

---

## 相关链接

- [Astro 文档](https://docs.astro.build/)
- [Zod 文档](https://zod.dev/)
- [Tailwind CSS 文档](https://tailwindcss.com/)

## 反馈

如有问题或建议，请在 GitHub 仓库提 issue。
