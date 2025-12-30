# 博客配置指南 (Blog Configuration Guide)

本指南介绍如何通过 YAML 配置文件自定义博客的 UI 和功能，无需修改源代码。

## 目录 (Table of Contents)

1. [配置文件位置](#配置文件位置)
2. [站点配置 (site.yml)](#站点配置-siteyml)
3. [导航配置 (nav.yml)](#导航配置-navyml)
4. [首页配置 (home.yml)](#首页配置-homeyml)
5. [文章页配置 (post.yml)](#文章页配置-postyml)
6. [主题配置 (theme.yml)](#主题配置-themeyml)
7. [个人资料配置 (profile.yml)](#个人资料配置-profileyml)
8. [配置验证](#配置验证)
9. [常见问题](#常见问题)

## 配置文件位置

所有配置文件位于 `src/config/yaml/` 目录下：

```text
src/config/yaml/
├── site.yml       # 站点全局配置
├── nav.yml        # 导航菜单配置
├── home.yml       # 首页配置
├── post.yml       # 文章页配置
├── theme.yml      # 主题配置
└── profile.yml    # 个人资料配置
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
