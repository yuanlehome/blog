# 配置治理文档

本文档描述仓库的 **配置入口与治理策略**，确保配置项的一致性和可维护性。

> **相关文档**
>
> - 站点配置指南 → [configuration.md](./configuration.md)
> - 仓库架构 → [architecture.md](./architecture.md)

---

## 一、配置入口清单

仓库中存在以下配置入口：

| 类别           | 路径                          | 用途                           |
| -------------- | ----------------------------- | ------------------------------ |
| YAML 配置      | `src/config/yaml/*.yml`       | 站点 UI 与功能配置             |
| 路径配置       | `src/config/paths.ts`         | 文件系统路径定义               |
| 站点配置       | `src/config/site.ts`          | 站点元数据（URL、标题）        |
| 功能开关       | `src/config/features.ts`      | 环境变量控制的布尔开关         |
| 环境变量       | `.env.local`                  | 本地开发密钥和配置             |
| Astro 配置     | `astro.config.mjs`            | Astro 构建配置                 |
| Tailwind 配置  | `tailwind.config.mjs`         | CSS 框架配置                   |
| TypeScript     | `tsconfig.json`               | TypeScript 编译选项            |
| 测试配置       | `vitest.config.ts`            | 单元测试配置                   |
| E2E 配置       | `playwright.config.ts`        | 端到端测试配置                 |
| Lint 配置      | `.prettierrc.json`、`.markdownlint*.jsonc` | 代码格式化与 Markdown 规范 |

---

## 二、YAML 配置入口（`src/config/yaml/`）

用户可通过 YAML 文件自定义站点 UI，无需修改代码。

| 文件             | 用途                               |
| ---------------- | ---------------------------------- |
| `site.yml`       | 站点名称、描述、版权信息           |
| `nav.yml`        | 导航菜单、品牌文字                 |
| `home.yml`       | 首页标题、分页配置                 |
| `post.yml`       | 文章页功能（目录、评论、元数据）   |
| `theme.yml`      | 主题、配色、代码块样式             |
| `layout.yml`     | 布局模式、侧边栏、对齐方式         |
| `typography.yml` | 字体、字号、行高                   |
| `components.yml` | 圆角、阴影、动画                   |
| `profile.yml`    | 关于页个人信息                     |

> **详细字段说明** → [configuration.md](./configuration.md)

---

## 三、环境变量入口

### 3.1 构建期环境变量

在 `.env.local` 或 CI 中配置，影响构建输出：

| 变量名                | 用途                   | 默认值                    |
| --------------------- | ---------------------- | ------------------------- |
| `SITE_BASE`           | 站点 base 路径         | `/blog/`                  |
| `SITE_URL`            | 站点完整 URL           | `https://.../blog`        |
| `PROJECT_ROOT`        | 项目根目录             | `process.cwd()`           |

### 3.2 Scripts 环境变量

用于脚本执行，不影响构建输出：

| 变量名                        | 用途                       | 必需性     |
| ----------------------------- | -------------------------- | ---------- |
| `NOTION_TOKEN`                | Notion API token           | 同步时必需 |
| `NOTION_DATABASE_ID`          | Notion 数据库 ID           | 同步时必需 |
| `DEEPSEEK_API_KEY`            | DeepSeek 翻译 API key      | 翻译时必需 |
| `MARKDOWN_TRANSLATE_ENABLED`  | 启用翻译（`1`/`0`）        | 可选       |
| `MARKDOWN_TRANSLATE_PROVIDER` | 翻译提供商                 | 可选       |

### 3.3 CI 专用变量

在 GitHub Actions 中配置为 Secrets 或 Variables：

| 类型     | 变量名              | 用途                     |
| -------- | ------------------- | ------------------------ |
| Secret   | `NOTION_TOKEN`      | Notion API token         |
| Secret   | `NOTION_DATABASE_ID`| Notion 数据库 ID         |
| Secret   | `DEEPSEEK_API_KEY`  | DeepSeek API key         |
| Secret   | `GH_PAT`            | GitHub PAT（PR 自动合并）|
| Secret   | `PREVIEW_TOKEN`     | 预览仓库部署 token       |
| Secret   | `PREVIEW_REPO`      | 预览仓库名               |
| Variable | `SITE_URL`          | 部署后站点 URL           |

---

## 四、配置生效性审计

### 4.1 审计脚本

```bash
npm run config:audit
```

执行 `scripts/config-audit.ts`，检查 YAML 配置项的生效状态。

### 4.2 审计状态

| 状态       | 含义                             |
| ---------- | -------------------------------- |
| ✅ USED    | 配置项已生效，影响页面渲染       |
| ⚪ READ_ONLY | 配置项被读取，但未影响渲染     |
| 🟡 SHADOWED | 配置项被硬编码覆盖             |
| ❌ UNUSED  | 配置项未被使用                   |

### 4.3 退出码

- `0`：所有配置有效
- `1`：发现问题（SHADOWED 或 UNUSED）

---

## 五、配置治理规则

### 5.1 新增配置的原则

1. **优先使用 YAML**：用户可配置项应放入 `src/config/yaml/`
2. **类型安全**：所有 YAML 配置必须有对应的 Zod schema
3. **默认值**：所有配置项必须有合理的默认值
4. **文档化**：新配置项必须在 [configuration.md](./configuration.md) 中说明

### 5.2 环境变量使用规范

1. **敏感信息**：使用 `.env.local`（本地）或 Secrets（CI）
2. **构建期配置**：使用 `src/config/env.ts` 解析
3. **命名规范**：使用 `SCREAMING_SNAKE_CASE`

### 5.3 防止 Dead Config

1. **定期审计**：运行 `npm run config:audit`
2. **使用映射函数**：禁止硬编码配置对应的样式
3. **删除废弃配置**：及时清理不再使用的配置项

---

## 六、配置文件间的关系

```text
┌─────────────────────────────────────────────────────────────┐
│                     src/config/yaml/                        │
│   site.yml  nav.yml  home.yml  post.yml  theme.yml  ...     │
└──────────────────────────┬──────────────────────────────────┘
                           │ 加载
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                   src/config/loaders/                       │
│   使用 Zod schema 验证并解析 YAML                           │
└──────────────────────────┬──────────────────────────────────┘
                           │ 导出
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                    src/config/index.ts                      │
│   统一导出所有配置（siteConfig, navConfig, ...）            │
└──────────────────────────┬──────────────────────────────────┘
                           │ 使用
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              src/components/ + src/layouts/                 │
│   读取配置，渲染页面                                        │
└─────────────────────────────────────────────────────────────┘
```

---

## 七、当前配置状态

根据最新审计结果：

- **总配置项**：6
- **已生效**：3
- **仅读取**：3
- **被覆盖**：0
- **未使用**：0

✅ 状态良好，所有配置项都在正常使用中。

---

## 八、后续维护

### 添加新配置项

1. 在 `src/config/yaml/<file>.yml` 中添加字段
2. 更新对应的 Zod schema
3. 在组件中使用配置
4. 运行 `npm run config:audit` 验证
5. 更新 [configuration.md](./configuration.md)

### 移除废弃配置

1. 从 YAML 文件中删除
2. 更新 Zod schema
3. 清理引用该配置的代码
4. 运行 `npm run config:audit` 确认
