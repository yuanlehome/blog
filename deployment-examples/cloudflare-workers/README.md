# Cloudflare Workers 部署指南

## 前置条件

1. Cloudflare 账户
2. 已安装 Node.js 和 npm
3. 安装 Wrangler CLI：`npm install -g wrangler`

## 步骤 1: 登录 Cloudflare

```bash
wrangler login
```

## 步骤 2: 创建 KV 命名空间

```bash
# 生产环境
wrangler kv:namespace create "VIEWS_KV"

# 预览环境（可选）
wrangler kv:namespace create "VIEWS_KV" --preview
```

命令会输出 namespace ID，将其复制到 `wrangler.toml` 文件中。

## 步骤 3: 配置 wrangler.toml

编辑 `wrangler.toml` 文件，替换以下内容：

```toml
[[kv_namespaces]]
binding = "VIEWS_KV"
id = "你的实际-KV-namespace-ID"
```

## 步骤 4: 部署

```bash
# 部署到生产环境
wrangler deploy

# 部署到指定环境
wrangler deploy --env production
wrangler deploy --env staging
```

## 步骤 5: 测试 API

```bash
# 获取浏览量
curl "https://your-worker.your-subdomain.workers.dev/api/views?slug=test-post"

# 增加浏览量
curl -X POST "https://your-worker.your-subdomain.workers.dev/api/views/incr?slug=test-post" \
  -H "Content-Type: application/json" \
  -d '{"clientId":"test-client-123"}'
```

## 步骤 6: 在博客中配置

在 Astro 博客中使用部署的 API：

```astro
<Views 
  slug={post.slug} 
  apiEndpoint="https://your-worker.your-subdomain.workers.dev" 
/>
```

## 高级配置

### 自定义域名

在 Cloudflare Dashboard 中为 Worker 添加自定义域名：

1. Workers & Pages → 选择你的 worker
2. Settings → Triggers → Add Custom Domain
3. 输入域名（例如：api.yourblog.com）

### CORS 限制

在 `index.js` 中修改 CORS 配置：

```javascript
const corsHeaders = {
  'Access-Control-Allow-Origin': 'https://yourblog.com',  // 限制为你的博客域名
  // ...
};
```

### 速率限制

添加基于 IP 的速率限制：

```javascript
// 在 handleIncrementViews 函数开始处添加
const clientIP = request.headers.get('CF-Connecting-IP');
const rateLimitKey = `ratelimit:${clientIP}`;
const requestCount = await env.VIEWS_KV.get(rateLimitKey);

if (requestCount && parseInt(requestCount) > 100) {
  return new Response(JSON.stringify({ error: 'Rate limit exceeded' }), {
    status: 429,
    headers: { 'Content-Type': 'application/json', ...corsHeaders },
  });
}

await env.VIEWS_KV.put(rateLimitKey, String(parseInt(requestCount || '0') + 1), {
  expirationTtl: 3600,  // 1 hour
});
```

## 监控

查看 Worker 日志和分析：

```bash
wrangler tail
```

或在 Cloudflare Dashboard 中查看：Workers & Pages → 选择你的 worker → Metrics

## 成本

- Workers：免费套餐包含 100,000 请求/天
- KV：免费套餐包含 100,000 读取/天，1,000 写入/天
- 超出部分按使用量计费

详见：https://developers.cloudflare.com/workers/platform/pricing/

## 故障排查

### 问题：部署失败

```bash
# 检查配置
wrangler whoami
wrangler kv:namespace list
```

### 问题：CORS 错误

确保 `corsHeaders` 配置正确，并且 OPTIONS 请求被正确处理。

### 问题：KV 写入失败

检查 KV namespace 绑定是否正确，ID 是否匹配。

## 参考资源

- [Cloudflare Workers 文档](https://developers.cloudflare.com/workers/)
- [Wrangler CLI 文档](https://developers.cloudflare.com/workers/wrangler/)
- [KV 存储文档](https://developers.cloudflare.com/workers/runtime-apis/kv/)
