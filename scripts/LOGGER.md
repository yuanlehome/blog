# 统一日志系统文档

## 功能概述

Scripts 使用统一的日志系统，提供结构化、可观测、对 CI 友好的日志记录能力。

**核心功能**：

- 标准日志级别：debug/info/warn/error
- 结构化日志：支持附加字段（JSON）
- 阶段化日志：span/time（开始/结束/耗时/状态）
- 诊断汇总：summary（成功/失败、统计、耗时）
- 敏感信息脱敏：自动屏蔽 token/cookie/password 等
- 多种输出格式：pretty（人类可读）/ json（机器分析）
- 文件输出：支持写入 JSON Lines 格式日志文件

## 环境变量配置

| 环境变量     | 可选值                        | 默认值   | 说明                                      |
| ------------ | ----------------------------- | -------- | ----------------------------------------- |
| `LOG_LEVEL`  | `debug`/`info`/`warn`/`error` | `info`   | 最小日志级别                              |
| `LOG_FORMAT` | `pretty`/`json`               | `pretty` | 输出格式（pretty 适合本地，json 适合 CI） |
| `LOG_FILE`   | 文件路径                      | 无       | 可选，写入 JSON Lines 格式日志文件        |
| `LOG_COLOR`  | `0`/`1`                       | 自动检测 | 是否启用彩色输出（TTY 自动检测）          |
| `LOG_SILENT` | `1`                           | 无       | 静默模式（测试时抑制控制台输出）          |

## 使用示例

### 基础用法

```typescript
import { createScriptLogger, now, duration } from './logger-helpers.js';

const scriptStart = now();
const logger = createScriptLogger('my-script', {
  url: 'https://example.com',
  provider: 'example',
});

logger.info('Starting script');

// 使用 span 记录阶段耗时
const fetchSpan = logger.time('fetch-data');
try {
  // ... 执行操作
  fetchSpan.end({ status: 'ok', fields: { size: 12345 } });
} catch (error) {
  fetchSpan.end({ status: 'fail' });
  logger.error(error);
  throw error;
}

// 记录摘要
logger.summary({
  status: 'ok',
  durationMs: duration(scriptStart),
  files: ['output.md'],
  stats: { images: 5, codeBlocks: 3 },
});
```

### 本地开发（Pretty 格式）

```bash
# 默认使用 pretty 格式（彩色，人类可读）
npm run import:content -- --url="https://example.com/article"

# 启用 debug 级别日志
LOG_LEVEL=debug npm run import:content -- --url="https://example.com/article"
```

输出示例：

```text
9:30:15 AM INFO  Starting content import url=https://example.com/article runId=a1b2c3d4
9:30:16 AM INFO  fetch-article started event=span.start span=fetch-article
9:30:18 AM INFO  fetch-article completed event=span.end span=fetch-article status=ok durationMs=2143
9:30:20 AM INFO  Summary status=ok durationMs=5234 files=["content/blog/article.md"]
```

### CI/自动化（JSON 格式）

```bash
# JSON 格式输出（机器可解析）
LOG_FORMAT=json npm run notion:sync

# 同时写入日志文件
LOG_FORMAT=json LOG_FILE=.logs/import.jsonl npm run import:content -- --url="..."
```

输出示例（每行一个 JSON 对象）：

```json
{"ts":"2025-01-01T09:30:15.123Z","level":"info","msg":"Starting content import","runId":"a1b2c3d4","script":"content-import","url":"https://example.com/article"}
{"ts":"2025-01-01T09:30:18.456Z","level":"info","msg":"fetch-article completed","event":"span.end","span":"fetch-article","status":"ok","durationMs":2143}
```

## 敏感信息脱敏

日志系统自动屏蔽敏感字段和值：

**字段名脱敏**（大小写不敏感）：

- `token`, `apiKey`, `api_key`
- `secret`, `password`, `passwd`
- `cookie`, `authorization`, `session`

**值脱敏模式**：

- Bearer Token：`Bearer eyJhbGciO...` → `Bearer eyJhbG...xyz`
- Cookie：`session=abc123def456ghi789jklmno` → `session=abc123...lmno`
- URL 参数：`?token=secret123` → `?token=[REDACTED]`

**自定义脱敏**：

```typescript
const logger = createLogger({
  redactKeys: ['customSecret', 'internalId'],
});
```

## 日志查看与分析

### 本地查看

```bash
# 实时查看日志文件
tail -f .logs/import.jsonl

# 过滤特定 runId 的日志
grep 'runId":"a1b2c3d4"' .logs/import.jsonl | jq .

# 查看所有错误日志
jq 'select(.level == "error")' .logs/import.jsonl
```

### CI 中查看

在 GitHub Actions 中：

```yaml
# 启用 JSON 格式便于 CI 解析
- name: Run import
  env:
    LOG_FORMAT: json
    LOG_LEVEL: info
  run: npm run import:content -- --url="${{ inputs.url }}"

# 查看 summary
- name: Extract summary
  run: |
    grep '"event":"summary"' logs/*.jsonl | jq .
```

## 最佳实践

### ✅ 推荐做法

1. **本地开发**：使用 `pretty` 格式，启用彩色输出
2. **CI/自动化**：使用 `json` 格式，写入日志文件
3. **长任务**：使用 `span` 记录各阶段耗时
4. **错误处理**：捕获异常并记录详细 stack trace
5. **脚本结束**：输出 `summary` 汇总执行结果

### ❌ 不要做

1. **不要**在日志中直接输出敏感信息（token/cookie/password）
2. **不要**在 GitHub Workflow inputs 中明文填写敏感信息
3. **不要**打印超长内容（如整篇 Markdown），使用摘要或 hash
4. **不要**混用 `console.log` 和 logger（保持一致性）

## 注意事项

**安全性**：

- 日志系统自动脱敏常见敏感字段
- 但不能保证 100% 覆盖所有场景
- 请人工审查日志输出，避免泄露敏感信息

**性能**：

- JSON 序列化是轻量的，不会显著影响性能
- 文件写入是异步的，不会阻塞主线程
- 建议在 CI 中启用文件输出，本地开发可禁用

**调试**：

- 使用 `LOG_LEVEL=debug` 查看详细日志
- 使用 `LOG_SILENT=1` 在测试中抑制输出
- 在 CI 中查看日志文件（`.logs/` 目录）

## 集成示例

参见 `scripts/examples/content-import-with-logger.ts`，展示了如何在脚本中集成日志系统。

完整示例包括：

- 脚本启动时创建 logger
- 使用 span 记录各阶段耗时
- 捕获并记录错误
- 输出最终 summary

## 覆盖率要求

本仓库要求测试覆盖率 **≥ 80%**（lines, statements），branches ≥ 75%。

配置位置：`vitest.config.ts`

```typescript
coverage: {
  thresholds: {
    lines: 80,
    functions: 80,
    branches: 75,
    statements: 80,
  },
}
```

### 查看覆盖率

```bash
# 运行测试并生成覆盖率报告
npm run test

# 查看 HTML 报告
open coverage/index.html
```

### 覆盖率最佳实践

1. **新增代码**：确保覆盖率 ≥ 90%
2. **难以覆盖的分支**：通过 mock/fake timers 覆盖
3. **不要**：简单用 `/* istanbul ignore */` 逃避覆盖
4. **例外情况**：仅当确有必要（如 TTY 检测、文件系统错误），且必须说明理由
