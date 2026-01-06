# 浏览量 API 部署示例

本目录包含不同平台的浏览量 API 后端实现示例。

## 可用部署方案

### 1. Cloudflare Workers + KV

**推荐指数**: ⭐⭐⭐⭐⭐

**优势**:

- 全球边缘网络，低延迟
- 免费套餐慷慨（100,000 请求/天）
- 简单的 KV 存储
- 易于部署和维护

**查看**: [cloudflare-workers/](./cloudflare-workers/)

### 2. Vercel Serverless (计划中)

适合已经在 Vercel 上部署博客的用户。

### 3. Supabase + Edge Functions (计划中)

适合需要更复杂数据查询和分析的场景。

## 选择建议

- **首次部署**: 推荐 Cloudflare Workers，免费额度高，部署简单
- **已用 Vercel**: 推荐 Vercel Serverless，统一平台管理
- **需要数据分析**: 推荐 Supabase，支持 SQL 查询

## 本地开发

博客内置 Mock Provider，无需部署后端即可在本地开发和测试：

```bash
npm run dev
```

浏览量数据会存储在内存中，刷新后重置。

## 贡献

欢迎贡献更多部署方案的示例！请确保：

1. 遵循 API 规范（见 [docs/page-views.md](../docs/page-views.md)）
2. 包含完整的部署文档
3. 测试过能正常工作

## 参考

- [浏览量功能文档](../docs/page-views.md)
- [API 规范](../docs/page-views.md#api-规范)
