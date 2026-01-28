# PDF 导入功能

本文档介绍基于 PaddleOCR-VL 的 PDF 导入功能，用于解析布局和提取文本。

## 功能概述

PDF 导入功能可将任意 PDF 文档导入为博客文章，流程如下：

1. 从 URL 下载 PDF
2. 调用 PaddleOCR-VL API 解析布局和提取文本
3. 转换为 Markdown 格式
4. 下载并保存所有图片
5. 可选翻译（使用 DeepSeek）
6. 保存为带 frontmatter 的博客文章

## 配置

### 环境变量

在 `.env.local` 中添加：

```env
# 必需 - PaddleOCR-VL API token
PADDLEOCR_VL_TOKEN=your_token_here

# 可选 - API URL（默认使用 PaddleOCR-VL 端点）
PADDLEOCR_VL_API_URL=https://xbe1mb28fa0dz7kb.aistudio-app.com/layout-parsing

# 可选 - PDF 文件大小限制（默认 50MB）
PDF_MAX_MB=50

# 可选 - 启用翻译（0 = 禁用，1 = 启用）
MARKDOWN_TRANSLATE_ENABLED=1
MARKDOWN_TRANSLATE_PROVIDER=deepseek

# 可选 - DeepSeek API key（翻译时必需）
DEEPSEEK_API_KEY=sk-your-api-key-here
```

### GitHub Actions

导入 workflow 已支持 PDF 导入：

1. 在 GitHub Secrets 中设置 `PADDLEOCR_VL_TOKEN`
2. 在 workflow 输入中提供 PDF URL
3. 系统自动检测 PDF 并使用相应适配器

## 使用方法

### 命令行

```bash
# 导入 PDF（不翻译）
npm run import:content -- --url https://example.com/paper.pdf

# 导入 PDF（启用翻译）
MARKDOWN_TRANSLATE_ENABLED=1 npm run import:content -- --url https://example.com/paper.pdf

# 导入受限域名（如 arXiv）使用 --forcePdf 参数
npm run import:content -- --url https://arxiv.org/pdf/2306.00978 --forcePdf

# 注意：不加 --forcePdf 时，arXiv URL 默认被阻止
# --forcePdf 参数强制使用通用 PDF 导入器
```

### GitHub Actions

1. 进入 Actions → Import Content
2. 点击 "Run workflow"
3. 输入 PDF URL（如 `https://example.com/document.pdf`）
4. 受限域名（如 arXiv）需勾选 "Force PDF import mode"
5. 配置选项：
   - 可选启用翻译
   - 选择翻译提供商（推荐 deepseek）
6. 运行 workflow

## 核心功能

### PDF 下载与验证

- 自动跟随重定向
- 失败重试（3 次，指数退避）
- 验证文件为有效 PDF（检查魔术字节 `%PDF-`）
- 文件大小限制（默认最大 50MB）
- 最小尺寸验证（50KB，防止下载不完整）

### OCR 处理

- 使用 PaddleOCR-VL 布局解析 API
- 提取文本并保持格式
- 识别和映射图片
- 保留文档结构（标题、列表、表格）

### Markdown 处理

- 修复未闭合代码围栏
- 规范化列表缩进
- 删除多余空行
- 验证内容质量（最少 20 行有效内容）
- 确保 MDX 兼容性

### 图片处理

- 下载 OCR 结果中引用的所有图片
- 保存到 `/images/pdf/{slug}/` 目录
- 更新 markdown 引用为本地路径
- 支持多种格式（PNG、JPG、GIF、WebP）
- 通过魔术字节验证图片格式
- 防止路径遍历攻击

### 可选翻译

- 集成现有翻译系统
- 使用 DeepSeek 翻译
- 保留代码块、URL 和技术术语
- 保持 markdown 结构

### 安全性

- 不记录 token 或敏感数据
- 验证所有文件路径防止目录遍历
- 强制文件大小限制
- 验证文件类型

## 错误处理

常见问题的错误提示：

- **缺少 token**：提示需要 `PADDLEOCR_VL_TOKEN`
- **下载失败**：显示 HTTP 状态码和错误详情
- **无效 PDF**：文件不以 `%PDF-` 开头
- **文件过大**：显示实际大小与限制对比
- **内容不足**：OCR 返回少于 20 行有效内容
- **OCR API 错误**：包含 HTTP 状态和错误消息

## 测试

包含完整单元测试：

```bash
# 运行 PDF 适配器测试
npm test -- tests/unit/pdf-vl-adapter.test.ts

# 运行所有测试
npm test
```

测试覆盖：

- URL 检测（PDF vs. 非 PDF）
- 完整 PDF 导入流程
- 错误场景（缺少 token、下载失败、无效 PDF）
- 内容质量验证
- 图片下载与处理

## 架构

### 文件

- `scripts/import/adapters/pdf_vl.ts` - 主适配器实现
- `scripts/import/adapters/pdf_vl_utils.ts` - PDF 下载与验证
- `scripts/import/adapters/pdf_vl_ocr.ts` - PaddleOCR-VL API 客户端
- `scripts/import/adapters/pdf_vl_markdown.ts` - Markdown 处理与图片处理
- `tests/unit/pdf-vl-adapter.test.ts` - 完整单元测试

### 流程

1. **检测**：适配器检查 URL 是否以 `.pdf` 结尾
2. **下载**：下载 PDF 并验证，带重试机制
3. **验证**：检查是否为有效 PDF 并符合尺寸要求
4. **OCR**：发送到 PaddleOCR-VL API 解析布局
5. **处理**：清理 markdown，验证内容质量
6. **图片**：下载所有图片并更新引用
7. **翻译**（可选）：翻译内容并保持结构
8. **输出**：生成带 frontmatter 的博客文章

## 限制

- 最大 PDF 尺寸：50MB（可配置）
- 最少内容要求：20 行有效内容
- 仅支持 PDF 文件（不支持其他文档格式）
- 需要网络连接访问 OCR API
- 需要 PaddleOCR-VL API token

## 常见问题

### "PADDLEOCR_VL_TOKEN environment variable is required"

解决方案：在 `.env.local` 或 GitHub Secrets 中添加 `PADDLEOCR_VL_TOKEN`。

### "File too small" 或 "Not a valid PDF file"

下载的文件不是有效 PDF。检查：

- URL 确实指向 PDF 文件
- 服务器可访问
- 不需要身份验证

### "Insufficient content quality"

OCR 返回少于 20 行有效内容。可能原因：

- PDF 是扫描图片（OCR 难以识别）
- PDF 主要是图片，文本很少
- OCR 服务出现问题

解决方案：确认 PDF 包含文本（而非仅扫描图片）。

### "Failed to download PDF after 3 attempts"

网络或服务器问题。检查：

- URL 可访问
- 网络连接稳定
- 服务器正在响应
