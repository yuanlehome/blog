# 持久化浏览量 (PV) 功能文档

## 概述

持久化浏览量（Page Views, PV）功能为博客文章提供浏览次数统计，支持持久化存储和防刷机制。该功能设计为前端静态站点 + 后端 API 的架构，可部署在 GitHub Pages 等静态托管平台。

## 功能特性

### 核心能力

1. **浏览量展示**：在文章详情页的元数据区域展示浏览量（例如：👀 1234）
2. **自动累计**：每次真实用户访问文章时，浏览量自动 +1
3. **防刷机制**：基于 client ID 和 24 小时时间窗口的去重策略
4. **持久化存储**：浏览量数据长期保存，支持外部 API 或本地 mock
5. **优雅降级**：API 失败时不影响页面渲染，自动隐藏浏览量显示
6. **本地开发支持**：内置 mock provider，无需配置即可本地开发

### 技术特点

- **类型安全**：完整的 TypeScript 类型定义
- **抽象设计**：ViewsProvider 接口，支持切换不同后端实现
- **客户端增强**：使用 requestIdleCallback 延迟执行，不阻塞首屏渲染
- **SEO 友好**：纯客户端执行，不影响 SSR/SSG
- **可测试**：完整的单元测试和 E2E 测试覆盖

## 架构设计

### 目录结构

```text
src/lib/views/          # 浏览量核心库
├── types.ts            # TypeScript 类型定义
├── client-id.ts        # Client ID 生成与持久化
├── slug-validator.ts   # Slug 校验工具
├── views-client.ts     # Views API 客户端实现
└── index.ts            # 模块导出

src/components/
└── Views.astro         # 浏览量展示组件

tests/unit/             # 单元测试
├── client-id.test.ts
├── slug-validator.test.ts
└── views-client.test.ts

tests/e2e/              # E2E 测试
└── views.spec.ts
```

### 抽象层设计

#### ViewsProvider 接口

```typescript
interface ViewsProvider {
  getViews(slug: string): Promise<ViewsResponse>;
  incrementViews(slug: string, clientId: string): Promise<ViewsIncrementResponse>;
}
```

#### 实现类

1. **HttpViewsProvider**: HTTP API 客户端，用于生产环境
2. **MockViewsProvider**: 内存存储实现，用于开发/测试

### API 规范

#### GET /api/views?slug=\<slug\>

获取指定文章的浏览量。

**请求参数**：
- `slug`: 文章 slug（必需）

**响应**：
```json
{
  "slug": "my-post",
  "views": 1234
}
```

#### POST /api/views/incr?slug=\<slug\>

增加指定文章的浏览量。

**请求参数**：
- `slug`: 文章 slug（必需，query 参数）

**请求体**：
```json
{
  "clientId": "uuid-client-id"
}
```

**响应**：
```json
{
  "slug": "my-post",
  "views": 1235,
  "counted": true
}
```

- `counted`: 布尔值，表示本次访问是否被计入

## 防刷策略

### Client ID 生成

1. **优先级**：localStorage > sessionStorage > 内存
2. **格式**：UUID v4 (例如：`550e8400-e29b-41d4-a716-446655440000`)
3. **存储键**：`blog_views_client_id`

### 去重逻辑

- **键**：`${clientId}:${slug}`
- **时间窗口**：24 小时
- **规则**：同一 client ID 在 24 小时内对同一文章只计数 1 次

## 使用指南

### 前端集成

在文章页面中使用 `Views` 组件：

```astro
---
import Views from '../components/Views.astro';
---

<Views slug={post.slug} />
```

可选配置 API endpoint：

```astro
<Views slug={post.slug} apiEndpoint="https://api.example.com" />
```

### 后端配置

#### 方案 A：使用 Mock Provider（默认）

无需配置，组件会自动使用内存存储的 MockViewsProvider。适合：
- 本地开发
- 测试环境
- 演示环境

#### 方案 B：配置外部 API

设置 `apiEndpoint` prop 指向你的 API 服务：

```astro
<Views slug={post.slug} apiEndpoint="https://your-api.example.com" />
```

### 部署后端 API

#### Cloudflare Workers + KV 示例

```javascript
// workers/views.js
export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    const slug = url.searchParams.get('slug');

    if (url.pathname === '/api/views') {
      // GET: 获取浏览量
      const views = await env.VIEWS_KV.get(slug) || 0;
      return new Response(JSON.stringify({ slug, views: parseInt(views) }), {
        headers: { 'Content-Type': 'application/json' }
      });
    }

    if (url.pathname === '/api/views/incr' && request.method === 'POST') {
      // POST: 增加浏览量
      const { clientId } = await request.json();
      const key = `${slug}:${clientId}`;
      const lastView = await env.VIEWS_KV.get(key);
      const now = Date.now();

      let counted = false;
      if (!lastView || now - parseInt(lastView) > 24 * 60 * 60 * 1000) {
        const currentViews = parseInt(await env.VIEWS_KV.get(slug) || 0);
        await env.VIEWS_KV.put(slug, String(currentViews + 1));
        await env.VIEWS_KV.put(key, String(now), { expirationTtl: 86400 });
        counted = true;
      }

      const views = parseInt(await env.VIEWS_KV.get(slug) || 0);
      return new Response(JSON.stringify({ slug, views, counted }), {
        headers: { 'Content-Type': 'application/json' }
      });
    }

    return new Response('Not Found', { status: 404 });
  }
};
```

配置文件 `wrangler.toml`:

```toml
name = "blog-views-api"
main = "workers/views.js"
compatibility_date = "2024-01-01"

[[kv_namespaces]]
binding = "VIEWS_KV"
id = "your-kv-namespace-id"
```

部署命令：

```bash
npx wrangler deploy
```

#### Vercel Serverless 示例

```typescript
// api/views.ts
import { VercelRequest, VercelResponse } from '@vercel/node';
import { createClient } from '@vercel/kv';

const kv = createClient({
  url: process.env.KV_REST_API_URL!,
  token: process.env.KV_REST_API_TOKEN!,
});

export default async function handler(req: VercelRequest, res: VercelResponse) {
  const slug = req.query.slug as string;

  if (req.method === 'GET') {
    const views = await kv.get(slug) || 0;
    return res.json({ slug, views: Number(views) });
  }

  if (req.method === 'POST') {
    const { clientId } = req.body;
    const key = `${slug}:${clientId}`;
    const lastView = await kv.get(key);
    const now = Date.now();

    let counted = false;
    if (!lastView || now - Number(lastView) > 24 * 60 * 60 * 1000) {
      await kv.incr(slug);
      await kv.set(key, now, { ex: 86400 });
      counted = true;
    }

    const views = await kv.get(slug) || 0;
    return res.json({ slug, views: Number(views), counted });
  }

  return res.status(405).json({ error: 'Method not allowed' });
}
```

## 测试

### 运行单元测试

```bash
npm run test -- tests/unit/client-id.test.ts
npm run test -- tests/unit/slug-validator.test.ts
npm run test -- tests/unit/views-client.test.ts
```

### 运行 E2E 测试

```bash
npm run test:e2e
```

### 测试覆盖

- ✅ Client ID 生成与持久化
- ✅ Slug 校验与清理
- ✅ Views API 调用（GET/POST）
- ✅ Mock Provider 行为
- ✅ 24 小时去重逻辑
- ✅ 错误处理与优雅降级
- ✅ 页面显示与交互

## 配置选项

### Views 组件 Props

| 属性          | 类型   | 必需 | 默认值     | 说明                    |
| ------------- | ------ | ---- | ---------- | ----------------------- |
| `slug`        | string | 是   | -          | 文章 slug               |
| `apiEndpoint` | string | 否   | undefined  | API 端点，不提供则使用 mock |

### ViewsProvider 配置

| 选项          | 类型   | 默认值 | 说明                  |
| ------------- | ------ | ------ | --------------------- |
| `apiEndpoint` | string | -      | API 基础 URL          |
| `timeout`     | number | 5000   | 请求超时时间（毫秒）  |

## 扩展指南

### 添加新的 Provider

1. 实现 `ViewsProvider` 接口：

```typescript
import type { ViewsProvider, ViewsResponse, ViewsIncrementResponse } from './types';

export class MyCustomProvider implements ViewsProvider {
  async getViews(slug: string): Promise<ViewsResponse> {
    // 实现获取逻辑
  }

  async incrementViews(slug: string, clientId: string): Promise<ViewsIncrementResponse> {
    // 实现增量逻辑
  }
}
```

2. 在 `createViewsProvider` 中添加选择逻辑：

```typescript
export function createViewsProvider(config?: Partial<ViewsProviderConfig>): ViewsProvider {
  if (config?.providerType === 'custom') {
    return new MyCustomProvider(config);
  }
  // ...
}
```

### 自定义显示样式

修改 `src/components/Views.astro` 中的模板：

```astro
<div class="custom-views-style">
  <span>🔥</span>
  <span data-views-count>—</span>
  <span>次浏览</span>
</div>
```

## 故障排查

### 问题：浏览量不显示

**可能原因**：
1. API 服务不可用
2. Slug 格式不正确
3. 网络请求被阻止

**解决方法**：
- 检查浏览器控制台错误信息
- 验证 API endpoint 配置
- 确认 slug 符合格式要求（小写字母、数字、连字符）

### 问题：浏览量不增加

**可能原因**：
1. 24 小时内重复访问
2. Client ID 相同
3. 后端去重逻辑生效

**解决方法**：
- 清除 localStorage 中的 `blog_views_client_id`
- 使用隐私模式/无痕模式
- 等待 24 小时后重试

### 问题：首屏渲染变慢

**不应该发生**：Views 组件使用 `requestIdleCallback` 延迟执行，不应阻塞渲染。

**检查**：
- 确认组件正确使用延迟加载
- 检查 API 响应时间
- 考虑增加超时时间

## 性能指标

- **首屏渲染**：不受影响（延迟执行）
- **API 超时**：5 秒（可配置）
- **存储开销**：localStorage 约 36 字节（UUID）
- **网络请求**：每篇文章 2 次（GET + POST）

## 安全考虑

1. **Slug 校验**：防止注入攻击
2. **Client ID 隔离**：每个客户端独立 ID
3. **限流保护**：后端应实现 rate limiting
4. **CORS 配置**：API 应正确配置 CORS 头

## 未来规划

- [ ] 支持批量查询浏览量
- [ ] 添加浏览量排行榜
- [ ] 支持更多后端存储（Supabase、Firebase）
- [ ] 添加管理界面
- [ ] 支持浏览量趋势图

## 参考资源

- [Cloudflare Workers 文档](https://developers.cloudflare.com/workers/)
- [Vercel Serverless Functions](https://vercel.com/docs/functions)
- [Astro 组件文档](https://docs.astro.build/en/core-concepts/astro-components/)

---

**最后更新**：2024-01
**维护者**：Blog Team
