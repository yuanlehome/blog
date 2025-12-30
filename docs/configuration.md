# 博客配置指南 (Blog Configuration Guide)

本指南介绍如何通过 YAML 配置文件自定义博客的 UI 和功能，无需修改源代码。

## 目录 (Table of Contents)

1. [配置文件位置](#配置文件位置)
2. [站点配置 (site.yml)](#站点配置-siteyml)
3. [导航配置 (nav.yml)](#导航配置-navyml)
4. [首页配置 (home.yml)](#首页配置-homeyml)
5. [文章页配置 (post.yml)](#文章页配置-postyml)
6. [主题配置 (theme.yml)](#主题配置-themeyml)
7. [布局配置 (layout.yml)](#布局配置-layoutyml) ⭐ **新增**
8. [排版配置 (typography.yml)](#排版配置-typographyyml) ⭐ **新增**
9. [组件配置 (components.yml)](#组件配置-componentsyml) ⭐ **新增**
10. [个人资料配置 (profile.yml)](#个人资料配置-profileyml)
11. [配置验证](#配置验证)
12. [自定义示例](#自定义示例) ⭐ **新增**
13. [常见问题](#常见问题)

## 配置文件位置

所有配置文件位于 `src/config/yaml/` 目录下：

```text
src/config/yaml/
├── site.yml         # 站点全局配置
├── nav.yml          # 导航菜单配置
├── home.yml         # 首页配置
├── post.yml         # 文章页配置
├── theme.yml        # 主题与色彩配置
├── layout.yml       # 布局与结构配置 ⭐ 新增
├── typography.yml   # 字体与排版配置 ⭐ 新增
├── components.yml   # 组件样式配置 ⭐ 新增
└── profile.yml      # 个人资料配置
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

配置网站主题和外观。

### 字段说明

| 字段           | 类型   | 默认值            | 说明                        |
| -------------- | ------ | ----------------- | --------------------------- |
| `defaultTheme` | enum   | "system"          | 默认主题：light/dark/system |
| `themes`       | array  | ["light", "dark"] | 可用主题列表                |
| `storageKey`   | string | "theme"           | LocalStorage 键名           |

#### icons

| 字段    | 类型   | 默认值 | 说明         |
| ------- | ------ | ------ | ------------ |
| `light` | string | "☀️"   | 亮色主题图标 |
| `dark`  | string | "🌙"   | 暗色主题图标 |

#### animations

| 字段                   | 类型    | 默认值 | 说明                 |
| ---------------------- | ------- | ------ | -------------------- |
| `respectReducedMotion` | boolean | true   | 尊重系统减少动画设置 |
| `enableScrollEffects`  | boolean | true   | 启用滚动效果         |

### 示例

```yaml
defaultTheme: 'light' # 默认使用亮色主题
themes:
  - 'light'
  - 'dark'

icons:
  light: '☀️'
  dark: '🌙'

animations:
  respectReducedMotion: true
  enableScrollEffects: true
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

### 配置文件可以用环境变量吗？

配置文件在构建时加载，不支持运行时环境变量。如需动态配置，请使用 `.env` 文件配合代码逻辑。

### 如何自定义主题颜色？

主题颜色主要通过 Tailwind CSS 配置。配置文件主要控制功能开关和文本内容。

要修改颜色，请编辑：

- `tailwind.config.mjs` - Tailwind 配置
- `src/styles/global.css` - 全局样式

### 修改配置后需要重启开发服务器吗？

是的。YAML 配置在构建时加载，修改后需要重启 `npm run dev`。

---

## 相关链接

- [Astro 文档](https://docs.astro.build/)
- [Zod 文档](https://zod.dev/)
- [Tailwind CSS 文档](https://tailwindcss.com/)

## 反馈

如有问题或建议，请在 GitHub 仓库提 issue。

## 布局配置 (layout.yml)

**⭐ 新增配置**：控制页面布局、侧边栏、目录和对齐方式。

### 字段说明

#### container

| 字段       | 类型   | 默认值  | 说明                 |
| ---------- | ------ | ------- | -------------------- |
| `width`    | string | "72rem" | 主内容区最大宽度     |
| `paddingX` | object | {...}   | 水平内边距（响应式） |

#### layoutMode

| 值             | 说明                     |
| -------------- | ------------------------ |
| `centered`     | 单列居中布局（无侧边栏） |
| `rightSidebar` | 右侧边栏布局（默认）     |
| `leftSidebar`  | 左侧边栏布局             |

#### sidebar

| 字段       | 类型    | 默认值  | 说明             |
| ---------- | ------- | ------- | ---------------- |
| `enabled`  | boolean | true    | 是否启用侧边栏   |
| `position` | enum    | "right" | 位置：left/right |
| `width`    | string  | "18rem" | 侧边栏宽度       |
| `sticky`   | boolean | true    | 是否固定定位     |
| `gap`      | string  | "3rem"  | 与内容区间距     |

#### toc（目录）

| 字段             | 类型    | 默认值   | 说明                             |
| ---------------- | ------- | -------- | -------------------------------- |
| `enabled`        | boolean | true     | 是否启用目录                     |
| `position`       | enum    | "right"  | 位置：left/right/inline          |
| `mobileBehavior` | enum    | "drawer" | 移动端行为：drawer/inline/hidden |
| `defaultOpen`    | boolean | false    | 默认是否展开                     |
| `offset`         | number  | 96       | 顶部偏移量（px）                 |

#### alignment

| 字段            | 类型 | 默认值 | 说明                  |
| --------------- | ---- | ------ | --------------------- |
| `headerAlign`   | enum | "left" | 头部对齐：left/center |
| `footerAlign`   | enum | "left" | 底部对齐：left/center |
| `postMetaAlign` | enum | "left" | 文章元信息对齐        |

### 示例

```yaml
# 单列居中布局
container:
  width: '65rem'
layoutMode: 'centered'
alignment:
  headerAlign: 'center'

# 左侧边栏布局
layoutMode: 'leftSidebar'
sidebar:
  position: 'left'
  width: '16rem'
toc:
  position: 'left'
```

## 排版配置 (typography.yml)

**⭐ 新增配置**：控制字体、字号、行高等排版设置。

### 字段说明

#### fontFamily

| 字段    | 类型  | 说明               |
| ------- | ----- | ------------------ |
| `sans`  | array | 无衬线字体栈       |
| `serif` | array | 衬线字体栈         |
| `mono`  | array | 等宽字体栈（代码） |

#### fontSize

字号预设，支持的键：`xs`, `sm`, `base`, `lg`, `xl`, `2xl`, `3xl`, `4xl`

#### lineHeight

| 字段      | 类型   | 默认值 | 说明     |
| --------- | ------ | ------ | -------- |
| `body`    | number | 1.75   | 正文行高 |
| `heading` | number | 1.3    | 标题行高 |
| `code`    | number | 1.65   | 代码行高 |
| `tight`   | number | 1.25   | 紧凑行高 |

#### fontWeight

| 字段       | 类型   | 默认值 | 说明     |
| ---------- | ------ | ------ | -------- |
| `normal`   | number | 400    | 普通字重 |
| `medium`   | number | 500    | 中等字重 |
| `semibold` | number | 600    | 次粗字重 |
| `bold`     | number | 700    | 粗字重   |

### 示例

```yaml
# 使用衬线字体作为正文
fontFamily:
  sans:
    - 'Georgia'
    - 'serif'

# 调整字号
fontSize:
  base: '1.125rem' # 18px，更大的正文
  lg: '1.25rem'

# 更紧凑的行距
lineHeight:
  body: 1.6
  code: 1.5
```

## 组件配置 (components.yml)

**⭐ 新增配置**：控制圆角、阴影、边框、动画等组件样式。

### 字段说明

#### radius（圆角）

| 字段 | 类型   | 默认值    | 用途         |
| ---- | ------ | --------- | ------------ |
| `sm` | string | "0.35rem" | 行内代码等   |
| `md` | string | "0.65rem" | 按钮等       |
| `lg` | string | "0.9rem"  | 代码块等     |
| `xl` | string | "0.75rem" | 卡片、图片等 |

#### shadow（阴影）

| 字段        | 类型    | 默认值 | 说明                 |
| ----------- | ------- | ------ | -------------------- |
| `card`      | enum    | "md"   | 卡片阴影级别         |
| `codeBlock` | enum    | "md"   | 代码块阴影级别       |
| `header`    | enum    | "md"   | 头部阴影级别         |
| `hoverLift` | boolean | false  | 悬停时是否有抬升效果 |

阴影级别：`none`, `sm`, `md`, `lg`

#### border

| 字段    | 类型   | 默认值  | 说明     |
| ------- | ------ | ------- | -------- |
| `style` | enum   | "solid" | 边框样式 |
| `width` | string | "1px"   | 边框宽度 |

边框样式：`solid`, `dashed`, `dotted`

#### motion（动画）

| 字段                   | 类型    | 默认值   | 说明                 |
| ---------------------- | ------- | -------- | -------------------- |
| `enabled`              | boolean | true     | 是否启用动画         |
| `level`                | enum    | "normal" | 动画强度             |
| `respectReducedMotion` | boolean | true     | 尊重系统减少动画设置 |

动画强度：`subtle`（100ms），`normal`（160ms），`energetic`（240ms）

#### spacingScale

| 值            | 倍数 | 说明             |
| ------------- | ---- | ---------------- |
| `compact`     | 0.75 | 紧凑间距         |
| `comfortable` | 1.0  | 舒适间距（默认） |
| `relaxed`     | 1.25 | 宽松间距         |

### 示例

```yaml
# 更圆润的设计
radius:
  sm: '0.5rem'
  md: '0.75rem'
  lg: '1rem'
  xl: '1.25rem'

# 更明显的阴影
shadow:
  card: 'lg'
  codeBlock: 'lg'
  hoverLift: true

# 更快的动画
motion:
  level: 'subtle'

# 更紧凑的间距
spacingScale: 'compact'
```

## 主题配置 (theme.yml) - 增强版

**🔄 已扩展**：新增色彩、代码块和头部样式配置。

### 新增字段说明

#### colorMode

| 字段          | 类型    | 默认值   | 说明                        |
| ------------- | ------- | -------- | --------------------------- |
| `default`     | enum    | "system" | 默认主题：light/dark/system |
| `allowToggle` | boolean | true     | 允许切换主题                |
| `persist`     | boolean | true     | 保存用户选择到 localStorage |

#### colors（亮色模式）

| 字段         | 类型   | 默认值    | 说明         |
| ------------ | ------ | --------- | ------------ |
| `brand`      | color  | "#3b82f6" | 品牌主色     |
| `accent`     | color  | "#8b5cf6" | 强调色       |
| `background` | color  | "#ffffff" | 页面背景     |
| `foreground` | color  | "#111827" | 文本颜色     |
| `muted`      | color  | "#6b7280" | 次要文本     |
| `border`     | color  | "#e5e7eb" | 边框颜色     |
| `card`       | color  | "#f9fafb" | 卡片背景     |
| `code.*`     | object | {...}     | 代码相关颜色 |

**颜色格式**：支持 hex（`#3b82f6`）、rgb（`rgb(59, 130, 246)`）、hsl（`hsl(217, 91%, 60%)`）

#### darkColors（暗色模式）

与 `colors` 结构相同，用于暗色模式的配色方案。

#### emphasis（强调样式）

| 字段            | 类型    | 默认值  | 说明                           |
| --------------- | ------- | ------- | ------------------------------ |
| `linkUnderline` | enum    | "hover" | 链接下划线：never/hover/always |
| `focusRing`     | boolean | true    | 显示焦点环                     |

#### codeBlock（代码块）

| 字段              | 类型    | 默认值         | 说明                       |
| ----------------- | ------- | -------------- | -------------------------- |
| `theme.light`     | string  | "github-light" | 亮色模式语法主题           |
| `theme.dark`      | string  | "github-dark"  | 暗色模式语法主题           |
| `showLineNumbers` | boolean | true           | 显示行号                   |
| `showCopyButton`  | boolean | true           | 显示复制按钮               |
| `wrapLongLines`   | boolean | false          | 换行显示长代码             |
| `inlineCodeStyle` | enum    | "subtle"       | 行内代码样式：subtle/boxed |
| `radius`          | string  | "0.9rem"       | 代码块圆角                 |
| `enableHighlight` | boolean | true           | 启用行高亮                 |

#### header（页头样式）

| 字段                | 类型   | 默认值    | 说明                                  |
| ------------------- | ------ | --------- | ------------------------------------- |
| `variant`           | enum   | "default" | 变体：default/subtle/frosted/elevated |
| `backgroundOpacity` | number | 0.92      | 背景不透明度（0-1）                   |
| `blurStrength`      | string | "10px"    | 毛玻璃效果强度（frosted 变体）        |

### 完整示例

```yaml
# 自定义配色方案
colorMode:
  default: 'light'
  allowToggle: true

colors:
  brand: '#0066cc'
  accent: '#ff6b6b'
  background: '#fafafa'
  code:
    background: '#f5f5f5'
    keyword: '#0066cc'

darkColors:
  brand: '#4da6ff'
  accent: '#ff8787'
  background: '#0a0a0a'

# 代码块配置
codeBlock:
  showLineNumbers: true
  showCopyButton: true
  wrapLongLines: false
  inlineCodeStyle: 'subtle'

# 页头样式
header:
  variant: 'frosted'
  backgroundOpacity: 0.85
  blurStrength: '12px'

emphasis:
  linkUnderline: 'always'
  focusRing: true
```

## 自定义示例

以下是一些常见的自定义场景示例。

### 示例 1：宽屏居中布局

适合喜欢简洁、专注阅读体验的用户。

```yaml
# layout.yml
container:
  width: '80rem' # 更宽的内容区
layoutMode: 'centered'
sidebar:
  enabled: false
toc:
  position: 'inline' # 目录放在文章内
alignment:
  headerAlign: 'center'
  postMetaAlign: 'center'
```

### 示例 2：紧凑深色主题

适合代码密集型博客。

```yaml
# theme.yml
colorMode:
  default: 'dark'

darkColors:
  background: '#0d1117'
  foreground: '#c9d1d9'
  code:
    background: '#161b22'

codeBlock:
  theme:
    dark: 'monokai'
  showLineNumbers: true
  wrapLongLines: true

# components.yml
spacingScale: 'compact'
radius:
  lg: '0.5rem' # 更少圆角

# typography.yml
fontSize:
  base: '0.95rem' # 稍小的字体
lineHeight:
  body: 1.6
```

### 示例 3：左侧边栏 + 大字体

适合阅读性优先的技术文章。

```yaml
# layout.yml
layoutMode: 'leftSidebar'
sidebar:
  position: 'left'
  width: '20rem'
toc:
  position: 'left'

# typography.yml
fontSize:
  base: '1.125rem' # 18px 基础字体
  lg: '1.375rem'
lineHeight:
  body: 1.8 # 更宽松的行高

fontFamily:
  sans:
    - 'Source Sans Pro'
    - 'system-ui'
    - 'sans-serif'

# components.yml
spacingScale: 'relaxed'
```

### 示例 4：彩色视觉风格

适合设计类、创意类博客。

```yaml
# theme.yml
colors:
  brand: '#8b5cf6' # 紫色主题
  accent: '#ec4899' # 粉色强调
  card: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)'

# components.yml
radius:
  xl: '1.5rem' # 更圆润
shadow:
  card: 'lg'
  hoverLift: true

motion:
  level: 'energetic' # 更有活力的动画

# typography.yml
fontFamily:
  sans:
    - 'Inter'
    - 'system-ui'
```

### 示例 5：极简黑白

适合文学、哲学类博客。

```yaml
# theme.yml
colors:
  brand: '#000000'
  accent: '#333333'
  background: '#ffffff'
  foreground: '#000000'
  border: '#e0e0e0'

emphasis:
  linkUnderline: 'always'

# components.yml
radius:
  sm: '0'
  md: '0'
  lg: '0'
  xl: '0' # 无圆角

shadow:
  card: 'none'
  codeBlock: 'none'

border:
  style: 'solid'
  width: '2px'

# typography.yml
fontFamily:
  sans:
    - 'Merriweather'
    - 'Georgia'
    - 'serif'

fontSize:
  base: '1.1rem'

lineHeight:
  body: 1.8
```

## 配置验证与错误排查

所有配置文件都使用 Zod 进行 schema 验证。如果配置无效，构建时会显示详细的错误信息。

### 常见验证错误

1. **颜色格式错误**

   ```text
   Invalid configuration in theme.yml:
     - colors.brand: Invalid color format. Use hex (#abc or #aabbcc), rgb(), rgba(), hsl(), or hsla()
   ```

   解决：确保颜色值使用正确的格式，如 `#3b82f6` 或 `rgb(59, 130, 246)`。

2. **枚举值错误**

   ```text
   Invalid configuration in layout.yml:
     - layoutMode: Invalid enum value. Expected 'centered' | 'rightSidebar' | 'leftSidebar'
   ```

   解决：使用配置文档中列出的有效值。

3. **数值范围错误**

   ```text
   Invalid configuration in typography.yml:
     - lineHeight.body: Number must be less than or equal to 3
   ```

   解决：确保数值在允许的范围内。

4. **字段类型错误**

   ```text
   Invalid configuration in components.yml:
     - radius.lg: Expected string, received number
   ```

   解决：确保字段类型正确，尺寸值需要带单位（如 `"1rem"` 而不是 `1`）。

### 验证配置

运行以下命令检查配置是否有效：

```bash
npm run check
npm run test
```
