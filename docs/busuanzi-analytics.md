# Busuanzi 浏览量统计配置文档

## 概述

本博客集成了 [Busuanzi（不蒜子）](https://busuanzi.ibruce.info/) 浏览量统计服务，可以展示文章页面的访问次数（PV）和站点整体的访问量统计（UV/PV）。

## 功能特性

- ✅ **文章页 PV 统计**：每篇文章详情页显示"阅读次数"
- ✅ **纯前端实现**：不需要后端服务器或数据库
- ✅ **优雅降级**：脚本加载失败时不影响页面显示
- ✅ **可开关配置**：通过环境变量控制是否启用
- ✅ **暗黑模式支持**：与博客主题保持一致
- ✅ **性能优化**：脚本异步加载，不阻塞页面渲染

## 配置方法

### 1. 环境变量配置

在项目根目录创建 `.env.local` 文件（可以参考 `.env.local.example`），添加以下配置：

```bash
# 启用 Busuanzi 浏览量统计
PUBLIC_BUSUANZI_ENABLED=true

# 可选：自定义 Busuanzi 脚本地址（不配置则使用默认地址）
# PUBLIC_BUSUANZI_SCRIPT_URL=https://busuanzi.ibruce.info/busuanzi/2.3/busuanzi.pure.mini.js

# 可选：启用调试日志（默认关闭）
# PUBLIC_BUSUANZI_DEBUG=true
```

### 2. 配置说明

| 环境变量                      | 必填 | 默认值                                                         | 说明                                       |
| ----------------------------- | ---- | -------------------------------------------------------------- | ------------------------------------------ |
| `PUBLIC_BUSUANZI_ENABLED`     | 否   | `false`                                                        | 是否启用 Busuanzi，设置为 `true` 或 `1`    |
| `PUBLIC_BUSUANZI_SCRIPT_URL`  | 否   | `https://busuanzi.ibruce.info/busuanzi/2.3/busuanzi.pure.mini.js` | Busuanzi 脚本地址，可配置为自建或镜像地址 |
| `PUBLIC_BUSUANZI_DEBUG`       | 否   | `false`                                                        | 是否在浏览器控制台输出调试信息             |

## 使用示例

### 启用 Busuanzi

在 `.env.local` 文件中设置：

```bash
PUBLIC_BUSUANZI_ENABLED=true
```

### 禁用 Busuanzi

在 `.env.local` 文件中设置：

```bash
PUBLIC_BUSUANZI_ENABLED=false
```

或者删除 `PUBLIC_BUSUANZI_ENABLED` 配置项（默认为禁用）。

### 使用自定义脚本地址

如果官方 Busuanzi 服务不可用，可以配置自建或第三方镜像：

```bash
PUBLIC_BUSUANZI_ENABLED=true
PUBLIC_BUSUANZI_SCRIPT_URL=https://your-cdn.com/busuanzi.js
```

### 开启调试模式

开发时可以开启调试模式，在浏览器控制台查看 Busuanzi 加载状态：

```bash
PUBLIC_BUSUANZI_ENABLED=true
PUBLIC_BUSUANZI_DEBUG=true
```

## 显示位置

### 文章页面

在文章详情页的元信息区域（标题下方），会显示：

```
👀 阅读 123 次
```

位置与发布日期、更新日期、字数统计等信息在同一区域。

### 暗黑模式

Busuanzi 统计信息会根据当前主题自动调整颜色，与博客整体风格保持一致。

## 工作原理

1. **脚本加载**：页面加载时，如果 `PUBLIC_BUSUANZI_ENABLED=true`，会异步加载 Busuanzi 脚本
2. **占位符识别**：Busuanzi 脚本会自动识别页面中的特定 DOM 元素（`id="busuanzi_value_page_pv"`）
3. **数值填充**：脚本从服务端获取访问统计数据并填充到页面
4. **显示控制**：数值填充成功后，容器元素会自动显示

## 降级处理

当 Busuanzi 服务不可用时：

- **配置未启用**：不会加载任何脚本，页面不会显示统计信息
- **脚本加载失败**：不会影响页面正常显示，不会有控制台错误
- **数据获取失败**：统计容器保持隐藏状态，不会显示 `0` 或错误信息

## 测试

### 单元测试

运行单元测试以验证 Busuanzi 集成逻辑：

```bash
npm run test
```

相关测试文件：`tests/unit/busuanzi.test.ts`

### E2E 测试

E2E 测试会自动 mock Busuanzi 脚本，避免依赖外部服务：

```bash
npm run test:e2e
```

## 隐私说明

Busuanzi 是一个第三方统计服务：

- 会收集页面访问信息（URL、访问次数）
- 使用 Cookie 或 localStorage 识别访客
- 详细隐私政策请参考 [Busuanzi 官方文档](https://busuanzi.ibruce.info/)

如果您对隐私有顾虑，可以通过设置 `PUBLIC_BUSUANZI_ENABLED=false` 完全禁用此功能。

## 常见问题

### Q: 为什么统计数据显示不出来？

A: 请检查：

1. 确认 `.env.local` 中 `PUBLIC_BUSUANZI_ENABLED=true`
2. 重新构建项目 `npm run build`
3. 检查浏览器控制台是否有网络错误
4. 如果使用自定义脚本地址，确认地址可访问

### Q: 本地开发时看不到统计数据？

A: 本地开发环境下，Busuanzi 可能无法正常统计。这是正常现象，部署到生产环境后即可正常显示。

### Q: 统计数据是否准确？

A: Busuanzi 使用简单的访问计数，可能受到以下因素影响：

- 浏览器缓存
- Cookie 清理
- 不同设备访问
- 爬虫访问

如需更精确的统计，建议使用专业的分析工具（如 Google Analytics）。

### Q: 如何重置统计数据？

A: Busuanzi 的统计数据存储在其服务端，无法通过配置重置。如需重新开始统计，需要更改页面 URL 或联系 Busuanzi 官方支持。

## 相关文件

- 核心库：`src/lib/analytics/busuanzi.ts`
- 显示组件：`src/components/BusuanziViews.astro`
- 布局集成：`src/layouts/Layout.astro`
- 页面集成：`src/pages/[...slug].astro`
- 单元测试：`tests/unit/busuanzi.test.ts`
- E2E 测试：`tests/e2e/blog.spec.ts`

## 更新日志

### 2026-01-07

- ✅ 初始实现 Busuanzi 集成
- ✅ 支持文章页 PV 统计
- ✅ 添加配置选项和环境变量
- ✅ 实现优雅降级机制
- ✅ 添加单元测试和 E2E 测试
- ✅ 支持暗黑模式
