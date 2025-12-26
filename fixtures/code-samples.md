---
title: "代码块渲染验收样例"
---

## TypeScript + 行高亮

```ts {1,4-6} title="hello.ts"
import { defineConfig } from 'astro/config'

const greeting = (name: string) => `Hello, ${name}!`
const users = ['Ada', 'Linus', 'Grace']

console.log(users.map(greeting).join(', '))
```

## 长行 + 滚动

```json title="data.json"
{
  "long_line": "https://example.com/this/is/a/very/long/path/that/should/scroll/nicely/without/wrapping/or/breaking-prose-styles",
  "nested": { "message": "Keep horizontal scroll and no prose padding" }
}
```

## Bash + 无行号

```bash nolines
npm run build && npm run preview
```

## Markdown 代码块（内嵌反引号）

```markdown {2}
Here is `inline code` and a fenced block:

```
console.log('nested fence')
```
```

## 未指定语言（fallback）

```
This block has no language but should still render with a header and copy button.
```

## Python 示例

```python {3}
def fib(n: int):
    if n <= 1:
        return n
    return fib(n - 1) + fib(n - 2)
```
