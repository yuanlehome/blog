# 配置生效性审计报告 (Config Effectiveness Audit Report)

生成时间: 2025-12-30T14:09:37.542Z

## 一、审计摘要 (Summary)

- **总配置项**: 6
- **已生效 (USED)**: 0
- **仅读取未影响渲染 (READ_ONLY)**: 3
- **被硬编码覆盖 (SHADOWED)**: 3
- **完全未使用 (UNUSED)**: 0

⚠️ **发现问题**: 存在未生效或被覆盖的配置项，需要修复！

## 二、配置项清单 (Config Inventory)

| 配置路径 | 类型 | 默认值 | 状态 | 问题 |
|---------|------|--------|------|------|
| `layout.alignment.headerAlign` | enum | `left` | ⚠️ SHADOWED | Hardcoded class "text-center" found in /home/runner/work/blog/blog/src/components/Footer.astro - may shadow config; Hardcoded class "items-center" found in /home/runner/work/blog/blog/src/components/Header.astro - may shadow config; Hardcoded class "items-center" found in /home/runner/work/blog/blog/src/components/MobileToc.astro - may shadow config; Hardcoded class "items-center" found in /home/runner/work/blog/blog/src/components/PostList.astro - may shadow config; Hardcoded class "items-start" found in /home/runner/work/blog/blog/src/components/TocTree.astro - may shadow config; Hardcoded class "items-start" found in /home/runner/work/blog/blog/src/pages/[...slug].astro - may shadow config; Hardcoded class "items-center" found in /home/runner/work/blog/blog/src/pages/[...slug].astro - may shadow config; Hardcoded class "text-left" found in /home/runner/work/blog/blog/src/pages/about.astro - may shadow config; Hardcoded class "justify-start" found in /home/runner/work/blog/blog/src/pages/about.astro - may shadow config; Hardcoded class "text-center" found in /home/runner/work/blog/blog/src/pages/about.astro - may shadow config; Hardcoded class "justify-center" found in /home/runner/work/blog/blog/src/pages/about.astro - may shadow config; Hardcoded class "items-center" found in /home/runner/work/blog/blog/src/pages/about.astro - may shadow config; Hardcoded class "items-center" found in /home/runner/work/blog/blog/src/pages/archive.astro - may shadow config; Hardcoded class "justify-center" found in /home/runner/work/blog/blog/src/pages/index.astro - may shadow config; Hardcoded class "items-center" found in /home/runner/work/blog/blog/src/pages/index.astro - may shadow config; Hardcoded class "justify-center" found in /home/runner/work/blog/blog/src/pages/page/[page].astro - may shadow config; Hardcoded class "items-center" found in /home/runner/work/blog/blog/src/pages/page/[page].astro - may shadow config |
| `layout.alignment.footerAlign` | enum | `left` | ⚠️ SHADOWED | Hardcoded class "text-center" found in /home/runner/work/blog/blog/src/components/Footer.astro - may shadow config; Hardcoded class "items-center" found in /home/runner/work/blog/blog/src/components/Header.astro - may shadow config; Hardcoded class "items-center" found in /home/runner/work/blog/blog/src/components/MobileToc.astro - may shadow config; Hardcoded class "items-center" found in /home/runner/work/blog/blog/src/components/PostList.astro - may shadow config; Hardcoded class "items-start" found in /home/runner/work/blog/blog/src/components/TocTree.astro - may shadow config; Hardcoded class "items-start" found in /home/runner/work/blog/blog/src/pages/[...slug].astro - may shadow config; Hardcoded class "items-center" found in /home/runner/work/blog/blog/src/pages/[...slug].astro - may shadow config; Hardcoded class "text-left" found in /home/runner/work/blog/blog/src/pages/about.astro - may shadow config; Hardcoded class "justify-start" found in /home/runner/work/blog/blog/src/pages/about.astro - may shadow config; Hardcoded class "text-center" found in /home/runner/work/blog/blog/src/pages/about.astro - may shadow config; Hardcoded class "justify-center" found in /home/runner/work/blog/blog/src/pages/about.astro - may shadow config; Hardcoded class "items-center" found in /home/runner/work/blog/blog/src/pages/about.astro - may shadow config; Hardcoded class "items-center" found in /home/runner/work/blog/blog/src/pages/archive.astro - may shadow config; Hardcoded class "justify-center" found in /home/runner/work/blog/blog/src/pages/index.astro - may shadow config; Hardcoded class "items-center" found in /home/runner/work/blog/blog/src/pages/index.astro - may shadow config; Hardcoded class "justify-center" found in /home/runner/work/blog/blog/src/pages/page/[page].astro - may shadow config; Hardcoded class "items-center" found in /home/runner/work/blog/blog/src/pages/page/[page].astro - may shadow config |
| `layout.alignment.postMetaAlign` | enum | `left` | ⚠️ SHADOWED | Hardcoded class "text-center" found in /home/runner/work/blog/blog/src/components/Footer.astro - may shadow config; Hardcoded class "items-center" found in /home/runner/work/blog/blog/src/components/Header.astro - may shadow config; Hardcoded class "items-center" found in /home/runner/work/blog/blog/src/components/MobileToc.astro - may shadow config; Hardcoded class "items-center" found in /home/runner/work/blog/blog/src/components/PostList.astro - may shadow config; Hardcoded class "items-start" found in /home/runner/work/blog/blog/src/components/TocTree.astro - may shadow config; Hardcoded class "items-start" found in /home/runner/work/blog/blog/src/pages/[...slug].astro - may shadow config; Hardcoded class "items-center" found in /home/runner/work/blog/blog/src/pages/[...slug].astro - may shadow config; Hardcoded class "text-left" found in /home/runner/work/blog/blog/src/pages/about.astro - may shadow config; Hardcoded class "justify-start" found in /home/runner/work/blog/blog/src/pages/about.astro - may shadow config; Hardcoded class "text-center" found in /home/runner/work/blog/blog/src/pages/about.astro - may shadow config; Hardcoded class "justify-center" found in /home/runner/work/blog/blog/src/pages/about.astro - may shadow config; Hardcoded class "items-center" found in /home/runner/work/blog/blog/src/pages/about.astro - may shadow config; Hardcoded class "items-center" found in /home/runner/work/blog/blog/src/pages/archive.astro - may shadow config; Hardcoded class "justify-center" found in /home/runner/work/blog/blog/src/pages/index.astro - may shadow config; Hardcoded class "items-center" found in /home/runner/work/blog/blog/src/pages/index.astro - may shadow config; Hardcoded class "justify-center" found in /home/runner/work/blog/blog/src/pages/page/[page].astro - may shadow config; Hardcoded class "items-center" found in /home/runner/work/blog/blog/src/pages/page/[page].astro - may shadow config |
| `layout.layoutMode` | enum | `rightSidebar` | ⚪ READ_ONLY | - |
| `layout.sidebar.enabled` | boolean | `true` | ⚪ READ_ONLY | - |
| `layout.sidebar.position` | enum | `right` | ⚪ READ_ONLY | - |

## 三、问题详情 (Issues Detail)

### layout.alignment.headerAlign

- **文件**: layout.yml
- **状态**: SHADOWED
- **问题**:
  - Hardcoded class "text-center" found in /home/runner/work/blog/blog/src/components/Footer.astro - may shadow config
  - Hardcoded class "items-center" found in /home/runner/work/blog/blog/src/components/Header.astro - may shadow config
  - Hardcoded class "items-center" found in /home/runner/work/blog/blog/src/components/MobileToc.astro - may shadow config
  - Hardcoded class "items-center" found in /home/runner/work/blog/blog/src/components/PostList.astro - may shadow config
  - Hardcoded class "items-start" found in /home/runner/work/blog/blog/src/components/TocTree.astro - may shadow config
  - Hardcoded class "items-start" found in /home/runner/work/blog/blog/src/pages/[...slug].astro - may shadow config
  - Hardcoded class "items-center" found in /home/runner/work/blog/blog/src/pages/[...slug].astro - may shadow config
  - Hardcoded class "text-left" found in /home/runner/work/blog/blog/src/pages/about.astro - may shadow config
  - Hardcoded class "justify-start" found in /home/runner/work/blog/blog/src/pages/about.astro - may shadow config
  - Hardcoded class "text-center" found in /home/runner/work/blog/blog/src/pages/about.astro - may shadow config
  - Hardcoded class "justify-center" found in /home/runner/work/blog/blog/src/pages/about.astro - may shadow config
  - Hardcoded class "items-center" found in /home/runner/work/blog/blog/src/pages/about.astro - may shadow config
  - Hardcoded class "items-center" found in /home/runner/work/blog/blog/src/pages/archive.astro - may shadow config
  - Hardcoded class "justify-center" found in /home/runner/work/blog/blog/src/pages/index.astro - may shadow config
  - Hardcoded class "items-center" found in /home/runner/work/blog/blog/src/pages/index.astro - may shadow config
  - Hardcoded class "justify-center" found in /home/runner/work/blog/blog/src/pages/page/[page].astro - may shadow config
  - Hardcoded class "items-center" found in /home/runner/work/blog/blog/src/pages/page/[page].astro - may shadow config

### layout.alignment.footerAlign

- **文件**: layout.yml
- **状态**: SHADOWED
- **问题**:
  - Hardcoded class "text-center" found in /home/runner/work/blog/blog/src/components/Footer.astro - may shadow config
  - Hardcoded class "items-center" found in /home/runner/work/blog/blog/src/components/Header.astro - may shadow config
  - Hardcoded class "items-center" found in /home/runner/work/blog/blog/src/components/MobileToc.astro - may shadow config
  - Hardcoded class "items-center" found in /home/runner/work/blog/blog/src/components/PostList.astro - may shadow config
  - Hardcoded class "items-start" found in /home/runner/work/blog/blog/src/components/TocTree.astro - may shadow config
  - Hardcoded class "items-start" found in /home/runner/work/blog/blog/src/pages/[...slug].astro - may shadow config
  - Hardcoded class "items-center" found in /home/runner/work/blog/blog/src/pages/[...slug].astro - may shadow config
  - Hardcoded class "text-left" found in /home/runner/work/blog/blog/src/pages/about.astro - may shadow config
  - Hardcoded class "justify-start" found in /home/runner/work/blog/blog/src/pages/about.astro - may shadow config
  - Hardcoded class "text-center" found in /home/runner/work/blog/blog/src/pages/about.astro - may shadow config
  - Hardcoded class "justify-center" found in /home/runner/work/blog/blog/src/pages/about.astro - may shadow config
  - Hardcoded class "items-center" found in /home/runner/work/blog/blog/src/pages/about.astro - may shadow config
  - Hardcoded class "items-center" found in /home/runner/work/blog/blog/src/pages/archive.astro - may shadow config
  - Hardcoded class "justify-center" found in /home/runner/work/blog/blog/src/pages/index.astro - may shadow config
  - Hardcoded class "items-center" found in /home/runner/work/blog/blog/src/pages/index.astro - may shadow config
  - Hardcoded class "justify-center" found in /home/runner/work/blog/blog/src/pages/page/[page].astro - may shadow config
  - Hardcoded class "items-center" found in /home/runner/work/blog/blog/src/pages/page/[page].astro - may shadow config

### layout.alignment.postMetaAlign

- **文件**: layout.yml
- **状态**: SHADOWED
- **问题**:
  - Hardcoded class "text-center" found in /home/runner/work/blog/blog/src/components/Footer.astro - may shadow config
  - Hardcoded class "items-center" found in /home/runner/work/blog/blog/src/components/Header.astro - may shadow config
  - Hardcoded class "items-center" found in /home/runner/work/blog/blog/src/components/MobileToc.astro - may shadow config
  - Hardcoded class "items-center" found in /home/runner/work/blog/blog/src/components/PostList.astro - may shadow config
  - Hardcoded class "items-start" found in /home/runner/work/blog/blog/src/components/TocTree.astro - may shadow config
  - Hardcoded class "items-start" found in /home/runner/work/blog/blog/src/pages/[...slug].astro - may shadow config
  - Hardcoded class "items-center" found in /home/runner/work/blog/blog/src/pages/[...slug].astro - may shadow config
  - Hardcoded class "text-left" found in /home/runner/work/blog/blog/src/pages/about.astro - may shadow config
  - Hardcoded class "justify-start" found in /home/runner/work/blog/blog/src/pages/about.astro - may shadow config
  - Hardcoded class "text-center" found in /home/runner/work/blog/blog/src/pages/about.astro - may shadow config
  - Hardcoded class "justify-center" found in /home/runner/work/blog/blog/src/pages/about.astro - may shadow config
  - Hardcoded class "items-center" found in /home/runner/work/blog/blog/src/pages/about.astro - may shadow config
  - Hardcoded class "items-center" found in /home/runner/work/blog/blog/src/pages/archive.astro - may shadow config
  - Hardcoded class "justify-center" found in /home/runner/work/blog/blog/src/pages/index.astro - may shadow config
  - Hardcoded class "items-center" found in /home/runner/work/blog/blog/src/pages/index.astro - may shadow config
  - Hardcoded class "justify-center" found in /home/runner/work/blog/blog/src/pages/page/[page].astro - may shadow config
  - Hardcoded class "items-center" found in /home/runner/work/blog/blog/src/pages/page/[page].astro - may shadow config

## 四、修复建议 (Fix Recommendations)

- **layout.alignment.headerAlign**: 移除硬编码的 class，改为从配置读取并映射到 Tailwind class
- **layout.alignment.footerAlign**: 移除硬编码的 class，改为从配置读取并映射到 Tailwind class
- **layout.alignment.postMetaAlign**: 移除硬编码的 class，改为从配置读取并映射到 Tailwind class
