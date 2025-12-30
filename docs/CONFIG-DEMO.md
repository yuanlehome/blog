# 配置生效性演示 (Config Effectiveness Demo)

本文档展示配置修复前后的对比，以及如何验证配置确实生效。

## 修复前 vs 修复后

### 1. Footer 对齐配置

#### 修复前 ❌

```astro
<!-- src/components/Footer.astro -->
<footer class="py-8 mt-auto border-t">
  <div class="container mx-auto px-4 text-center text-gray-500">
    <!-- 硬编码 text-center，无法通过配置改变 -->
  </div>
</footer>
```

#### 修复后 ✅

```astro
<!-- src/components/Footer.astro -->
import { getSiteConfig, getLayoutConfig } from '../config/loaders';
import { alignToTextClass } from '../lib/ui/alignment';

const layoutConfig = getLayoutConfig();
const footerAlignClass = alignToTextClass(layoutConfig.alignment.footerAlign);

<footer class="py-8 mt-auto border-t">
  <div class={`container mx-auto px-4 ${footerAlignClass} text-gray-500`}>
    <!-- 从配置读取对齐方式 -->
  </div>
</footer>
```

#### 配置文件

```yaml
# src/config/yaml/layout.yml
alignment:
  footerAlign: 'left' # 改为 'center' 可立即生效
```

### 2. Header 对齐配置

#### 修复前 ❌

```astro
<!-- src/components/Header.astro -->
<header>
  <div class="flex justify-between items-center">
    <!-- 硬编码 justify-between items-center -->
  </div>
</header>
```

#### 修复后 ✅

```astro
<!-- src/components/Header.astro -->
import { getLayoutConfig } from '../config/loaders';
import { alignToJustifyClass, alignToItemsClass } from '../lib/ui/alignment';

const layoutConfig = getLayoutConfig();
const headerAlign = layoutConfig.alignment.headerAlign;
const justifyClass = alignToJustifyClass(headerAlign);
const itemsClass = alignToItemsClass(headerAlign);

<header>
  <div class={`flex ${justifyClass} ${itemsClass}`}>
    <!-- 从配置读取对齐方式 -->
  </div>
</header>
```

#### 配置文件

```yaml
# src/config/yaml/layout.yml
alignment:
  headerAlign: 'left' # 改为 'center' 可立即生效
```

### 3. 文章元数据对齐配置

#### 修复前 ❌

```astro
<!-- src/pages/[...slug].astro -->
<h1 class="text-3xl font-bold">Title</h1>
<div class="flex items-center gap-3">
  <!-- 硬编码 items-center -->
</div>
```

#### 修复后 ✅

```astro
<!-- src/pages/[...slug].astro -->
import { getLayoutConfig } from '../config/loaders';
import { alignToTextClass, alignToItemsClass } from '../lib/ui/alignment';

const layoutConfig = getLayoutConfig();
const postMetaAlign = layoutConfig.alignment.postMetaAlign;
const metaTextClass = alignToTextClass(postMetaAlign);
const metaItemsClass = alignToItemsClass(postMetaAlign);

<h1 class={`text-3xl font-bold ${metaTextClass}`}>Title</h1>
<div class={`flex ${metaItemsClass} gap-3`}>
  <!-- 从配置读取对齐方式 -->
</div>
```

#### 配置文件

```yaml
# src/config/yaml/layout.yml
alignment:
  postMetaAlign: 'left' # 改为 'center' 可立即生效
```

## 如何验证配置生效

### 方法 1: 修改配置并构建

```bash
# 1. 修改配置文件
vim src/config/yaml/layout.yml

# 将 footerAlign 从 'left' 改为 'center'
alignment:
  footerAlign: 'center'

# 2. 构建项目
npm run build

# 3. 预览
npm run preview

# 4. 查看 Footer 现在应该是居中对齐
```

### 方法 2: 运行单元测试

```bash
# 测试对齐映射函数
npm test -- alignment.test.ts

# 输出应该显示：
# ✓ alignToTextClass('left') -> 'text-left'
# ✓ alignToTextClass('center') -> 'text-center'
# ✓ alignToJustifyClass('left') -> 'justify-start'
# ✓ alignToJustifyClass('center') -> 'justify-center'
```

### 方法 3: 运行配置审计

```bash
# 运行审计脚本
npm run config:audit

# 输出应该显示：
# ✅ layout.alignment.headerAlign - USED
# ✅ layout.alignment.footerAlign - USED
# ✅ layout.alignment.postMetaAlign - USED
# ✅ PASS: All configs are effective!
```

## 配置映射函数

新增的工具函数位于 `src/lib/ui/alignment.ts`：

```typescript
export type AlignmentValue = 'left' | 'center';

// 映射到 text-* class
export function alignToTextClass(align: AlignmentValue): string {
  return align === 'center' ? 'text-center' : 'text-left';
}

// 映射到 justify-* class
export function alignToJustifyClass(align: AlignmentValue): string {
  return align === 'center' ? 'justify-center' : 'justify-start';
}

// 映射到 items-* class
export function alignToItemsClass(align: AlignmentValue): string {
  return align === 'center' ? 'items-center' : 'items-start';
}
```

## 配置守护系统

### 自动审计脚本

```bash
npm run config:audit
```

该脚本会：

1. 扫描所有配置项
2. 检查每个配置是否被使用
3. 检测硬编码覆盖
4. 生成审计报告到 `docs/config-audit.md`

### 退出码

- **0**: 所有配置有效 ✅
- **1**: 发现问题（UNUSED 或 SHADOWED）❌

### CI 集成

在 GitHub Actions 中添加：

```yaml
- name: Config Audit
  run: npm run config:audit
```

如果有配置问题，CI 会失败，确保问题被及时发现。

## 效果演示

### 左对齐 (left) - 默认

```yaml
alignment:
  footerAlign: 'left'
```

渲染结果：

```html
<div class="text-left">© 2025 Blog. All rights reserved.</div>
```

### 居中对齐 (center)

```yaml
alignment:
  footerAlign: 'center'
```

渲染结果：

```html
<div class="text-center">© 2025 Blog. All rights reserved.</div>
```

## 总结

✅ 配置现在**真正生效**了！

- 修改 YAML 配置文件会立即影响 UI
- 不存在硬编码覆盖配置的情况
- 有自动化审计确保配置有效性
- 有单元测试保障映射函数正确性

这就是"配置驱动系统"的正确实现方式！
