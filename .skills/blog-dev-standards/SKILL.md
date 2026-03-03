---
name: blog-dev
description: >
  为 yuanlehome/blog（Astro 静态博客）提供可复用的开发规范与执行流程，确保改动可落地、最小化、可验证、CI 必过。
  Use this when users ask to add features, fix bugs, optimize UX, adjust rendering pipelines, modify scripts/workflows, or refactor code in the blog repo.
license: ISC (repo LICENSE) + project conventions in this file
---

# Blog Dev Skill（yuanlehome/blog）

本 Skill 约束任何对 `yuanlehome/blog` 的变更：**先基于真实代码建立事实 → 最小改动实现 → 可验证验收 → CI 必过**。仓库结构、边界与工作流以仓库内现有 docs 为准（见下方 Resources）。

---

## Use when

- 新功能：主题/布局/侧边栏/TOC/图片交互/搜索/标签/评论/代码高亮/文章元信息等
- 渲染链路：Mermaid、Markdown 插件、首图/封面识别、图片处理、暗黑模式可读性
- 内容管线：Notion 同步、外部文章导入（知乎/微信/Medium/PDF/others）、删除文章、翻译/Markdown 修复
- CI/Workflow：GitHub Actions、质量门禁、部署、烟测、链接检查、PR Preview、自动修文
- 性能/可靠性：滚动/事件监听、移动端触控、SSR/构建期稳定性、失败兜底

---

## Critical constraints (non-negotiable)

### 1) CI 必须通过
交付的实现方案或 agent prompt 必须以仓库脚本为准，提交最终代码前，默认保证至少这些命令通过（同仓库 README/CI 约定）：
- `npm run check`
- `npm run lint`
- `npm run format`
- `npm run test`
- `npm run test:e2e`
- `npm run ci`

> 若仓库脚本名发生变化，以 `package.json` 为准，但质量门槛不降低。

### 2) 严格遵循仓库分层与依赖边界（最重要）
以 `docs/architecture.md` 为准：
- **Runtime 层**：`src/`（Astro 构建期 + 浏览器运行期），只读取**预生成内容制品**
- **Scripts 层**：`scripts/`（Node.js 运行），负责内容获取与预处理
- **禁止** Runtime 依赖 Scripts（`src` 不得 import `scripts/*`）
- Scripts 仅允许导入少量共享模块（以仓库 docs 约定为准，典型是 `src/config/paths.ts`、`src/lib/slug/`）

### 3) 最小改动原则（Minimal change surface)
- 优先修复/扩展既有机制；避免“顺手重构”
- 不引入大体量新依赖；如必须引入，说明原因、替代方案、影响面（bundle/维护成本）

### 4) 文档变更：允许，但必须克制
- 文档（README / docs / 注释）**可以**为完成任务做必要调整，但必须：
  - **只改与任务强相关**的小段落/小节
  - **短、准、可执行**（怎么用/怎么验收/注意事项），禁止长篇大论
  - 不新增大量章节/流程图/长列表造成维护负担
- 若用户明确要求写/改文档：**中文撰写**，同样遵循最小改动

### 5) 代码组织：禁止“通用 utils 垃圾桶”
- 不新增通用 `utils/` 目录（用户偏好）
- 复用逻辑必须贴近业务域放置（feature 内部最小抽象）
- 允许 Scripts 内部已有的共享工具文件（见 `scripts/README.md`）

---

## Canonical repo references (must follow)

这些是仓库内已成型的“权威 docs”，agent 需要先读后改：

- `docs/architecture.md`：模块划分、职责边界、依赖方向、边界规则、新增功能/脚本规范
- `docs/ci-workflow.md`：工作流清单、触发条件、职责、参数与 secrets
- `docs/configuration.md`：YAML 配置文件位置、字段说明、环境变量、验证方式
- `docs/config-audit.md`：配置治理与审计规范（含映射函数与测试要求）
- `scripts/README.md`：脚本功能/参数/用例（Notion sync、导入、删除、配置审计、Markdown 管线等）
- `README.md`：项目总览、常用命令、目录结构、CI/CD 总表、写作 frontmatter 约定

---

## Default workflow (agent execution)

### Step 0 — Establish facts from real code
必须完成：
- 找到触发入口（页面路由/组件/脚本/Workflow）与最小可复现路径
- 明确影响面：桌面/移动、亮/暗主题、构建期/运行期、内容来源（notion/wechat/zhihu/medium/others）
- 产出：问题定位（文件路径/函数/组件）+ 复现步骤 + 期望行为（验收标准）

### Step 1 — Propose minimal change plan
输出必须包含：
- **Goal**：可观测验收标准（“打开/关闭/滚动/缩放/导入/构建”的具体表现）
- **Non-goals**：明确不做什么（防止 scope creep）
- **Change list**：到文件路径级别（必要时到函数/组件）
- **Risks & mitigations**：主题/移动端/SSR/性能/可访问性/回归风险
- **Rollback**：如何回滚（小 commit、feature flag、可逆改动）

### Step 2 — Implement with repo conventions
- 沿用现有结构：`src/config/*`、`src/lib/*`、`src/components/*`、`src/pages/*`
- 交互/性能：避免滚动监听风暴；必要时使用 `requestAnimationFrame` 节流；事件监听注意 passive；触控手势与页面滚动冲突要处理
- 失败兜底：图片/渲染失败、外部资源失败、数据为空、慢网速均需合理降级

### Step 3 — Quality gate
必须做到：
- 跑完：`npm run check && npm run lint && npm run test && npm run test:e2e && npm run ci`
- 手动验收：桌面 + 移动端、亮/暗主题、关键交互（关闭恢复滚动位置、Esc/遮罩点击等）
- 若涉及配置：按 `docs/config-audit.md` 的“映射函数 + 单测”规范，避免硬编码遮蔽配置

### Step 4 — Deliverable format (what to output)
交付给用户/下游 agent 的内容必须包含：
- 变更摘要（why/what）
- 可执行步骤（按顺序）
- 文件级改动点（到路径粒度）
- 验收 checklist（用户可照着点）
- CI 命令清单（再次强调必跑）

---

## Playbooks (common tasks)

### A) UI/交互：图片放大/拖拽/缩放、TOC、侧边栏
默认要求：
- 同时覆盖鼠标 + 触控（drag/pinch/double tap）+ 滚轮（如适用）
- 关闭逻辑完整：Esc/遮罩/按钮；阻止背景滚动穿透；关闭后恢复滚动位置
- 暗黑模式可读：背景/边框/对比度
- 避免：滚动跳回、事件冒泡冲突、过度重渲染导致卡顿

### B) 渲染链路：Mermaid、Markdown 插件、首图/封面
默认要求：
- 渲染失败有降级（原始代码/占位/提示）且不阻塞正文
- 文本不重叠：合理 padding、字号、节点间距、自动换行策略
- 暗黑模式清晰：导出图背景与前景对比正确

### C) 内容脚本：Notion 同步 / 外部导入 / PDF OCR / 删除文章
默认要求（以 `scripts/README.md` 为准）：
- 幂等：重复运行不产生不可控副作用
- 参数/环境变量清晰可查；失败信息可定位
- CI/Workflow 入参保持向后兼容（除非明确破坏性变更）

### D) 配置治理
以 `docs/configuration.md` + `docs/config-audit.md` 为准：
- 配置必须通过映射函数消费，禁止硬编码遮蔽
- 新增配置：同步补 schema 校验 + 映射函数 + 单测（覆盖枚举值）
- 可用 `npm run config:audit` 做回归检查

---

## Do not

- 不要在没有需求的情况下做“顺手重构”
- 不要引入通用 utils 目录/文件当垃圾桶
- 不要做长篇文档扩写；如需改文档必须与任务强相关且最小改动
- 不要交付“看起来对”的方案：必须可落地、可验证、可跑 CI

---

## Quick self-check (copy/paste)

- [ ] 我已阅读并遵循：`docs/architecture.md`、`docs/ci-workflow.md`、`docs/configuration.md`、`docs/config-audit.md`、`scripts/README.md`、`README.md`
- [ ] 我已基于真实代码定位入口与调用链，并给出最小复现步骤
- [ ] 方案是最小改动面，且包含风险与回滚方式
- [ ] 没有新增通用 utils 目录/文件
- [ ] 文档改动（若有）是短小且任务相关的
- [ ] 我已确保：`npm run check && npm run lint && npm run test && npm run test:e2e && npm run ci` 可通过
- [ ] 我提供了：可执行步骤 + 文件级改动点 + 验收 checklist

---

## Resources (repo-internal)

- `docs/architecture.md`
- `docs/ci-workflow.md`
- `docs/configuration.md`
- `docs/config-audit.md`
- `scripts/README.md`
- `README.md`
