# Refactor Plan

## 1. 当前目录结构（关键节点）

```text
/home/runner/work/blog/blog
├── scripts/
│   ├── content-import.ts        # Playwright 抓取并转 MDX（知乎/Medium/微信/兜底）
│   ├── delete-article.ts        # 按 slug/path 删除文章与关联图片
│   ├── fix-math.ts              # 批量修正常见数学公式格式
│   ├── notion-sync.ts           # Notion → Markdown + 下载图片 + slug 去重
│   └── slug.ts                  # slug 生成/去重工具（仅脚本层）
├── src/
│   ├── components/              # UI 组件（列表/页眉/浮动按钮等）
│   ├── config/                  # 站点配置与路径常量（base、site、paths）
│   ├── content/                 # 内容集合与 frontmatter 校验
│   ├── lib/
│   │   ├── content/             # posts/readingTime/tocTree/slugger 等
│   │   ├── markdown/            # remark/rehype 插件
│   │   ├── site/assetUrl.ts
│   │   └── ui/                  # code-blocks、floatingActionStack
│   ├── pages/                   # 路由（index / archive / about / [...slug] / rss）
│   ├── styles/                  # 全局样式
│   └── utils/                   # 仅 re-export src/lib/*，无独立实现
├── tests/
│   ├── unit/                    # slug/notion-sync/helpers/assetUrl 等单测
│   └── e2e/                     # Playwright（base 依赖 /blog 前缀）
├── docs/architecture.md, ci-workflow-map.md
├── astro.config.mjs             # base/site 配置，remark/rehype 插件挂载
├── package.json                 # npm scripts（notion:sync / import:content / delete:article 等）
└── playwright.config.ts         # E2E 入口（需先安装浏览器）
```

## 2. `scripts/` 脚本梳理

| 文件                | 作用                                                                                           | 输入/输出 & 调用                                                                                                                          | 存在的问题                                                                                                                                                                   |
| ------------------- | ---------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `notion-sync.ts`    | 拉取 Notion DB，转 Markdown，下载图片/封面，写入 `src/content/blog/notion`，并迁移旧 slug 目录 | 依赖环境变量 `NOTION_TOKEN`/`NOTION_DATABASE_ID`，由 `npm run notion:sync` 调用；输出 md 与 `/public/images/notion/<slug>`                | CLI、IO 与逻辑紧耦合；顶层 `dotenv` + `process.exit` 增加可测性难度；图片下载/slug 处理散落；缺少明确的 logger/返回值；部分逻辑重复于 `scripts/slug.ts`                      |
| `content-import.ts` | 用 Playwright 抓取外部文章（知乎/Medium/微信/兜底），本地化图片并生成 MDX                      | CLI 读取 `--url` 等参数或环境变量，由 `npm run import:content` 调用；输出到 `src/content/blog/<provider>/` 与 `public/images/<provider>/` | 解析、抓取、写文件混在同一层；slug 生成自 `slugify`，未复用脚本 slug 工具且无冲突处理；缺少测试与可插拔 logger；多处 `process.exit`/`console`；图片本地化与 HTML 转 MDX 混杂 |
| `fix-math.ts`       | 遍历目录/文件修正常见数学公式（去除空格、行内/块级规范等）                                     | CLI 参数为文件/目录路径，`npm run notion:sync` 和 `npm run import:content` 作为后置步骤调用                                               | 纯逻辑与文件遍历未拆分；无测试；直接写文件与打印日志，缺少可组合接口                                                                                                         |
| `delete-article.ts` | 按 slug 或路径删除文章，可选删除关联图片目录                                                   | 读取 `--target`/`--delete-images`/`--dry-run` 或 env，独立 `npm run delete:article` 调用                                                  | CLI 与业务耦合；slug 匹配规则散落；无测试；logger/退出码不统一                                                                                                               |
| `slug.ts`           | `normalizeSlug`/`deriveSlug`/`ensureUniqueSlug`/`shortHash`                                    | 供 `notion-sync.ts` 使用，单测存在                                                                                                        | 仅服务脚本层；与 `content-import.ts`、页面路由的 slug 逻辑未对齐；不在 `src/lib`，无法作为统一入口                                                                           |

## 3. slug 逻辑散点

- `scripts/slug.ts`：slug 规范化（`slugify` lower+strict）、显式/标题/ID 兜底、去重（owner hash+递增），被 `notion-sync` 调用并在单测覆盖。
- `scripts/notion-sync.ts`：使用 `deriveSlug` + `ensureUniqueSlug`，并在 slug 变更时迁移图片目录；图片下载路径依赖当前 slug。
- `scripts/content-import.ts`：独立用 `slugify` 生成 slug，兜底为 URL path 或时间戳，无冲突检测，也未复用脚本 slug 工具。
- `src/lib/content/slugger.ts` 与 `src/utils/slugger.ts`：为 Markdown heading 生成 anchor（GithubSlugger），与文章 slug 无关。
- 页面路由 (`src/pages/[...slug].astro`) 与组件依赖 content collection 的 `slug` 字段，未有集中处理函数；RSS/Archive/PrevNext 等都直接使用该 slug。
- 测试：`tests/unit/slug.test.ts` 仅覆盖 `scripts/slug.ts`；Notion 同步测试中也间接验证 slug 去重/迁移；未覆盖 content-import 与页面层 slug 一致性。

## 4. `utils/` 现状

- `src/utils/*` 基本是对 `src/lib/*` 的 re-export（如 `assetUrl`, `dates`, `readingTime`, `tocTree`, `slugger`, `floatingActionStack` 等），无独立实现。
- `src/lib/` 已按 domain 粗分 `content/`、`markdown/`、`site/`、`ui/`，但缺少统一出口与导入约束，`utils/` 形同别名层，易造成混用。
- 建议：收敛到 `src/lib` 为唯一工具层，提供 `src/lib/utils/index.ts`（或 `src/lib/index.ts`）统一出口；`src/utils` 退场或仅保留兼容 re-export，并清理跨层相对路径导入。

## 5. README 与实际不一致点

- 结构描述遗漏 `scripts/delete-article.ts` 与 `scripts/slug.ts`，且 `src/content/blog` 现有 `notion/`, `wechat/`, `others/`，与 README 所列 `local/` 示例不完全匹配。
- 未说明运行时 `SITE_BASE`/`SITE_URL`（生产默认 `/blog`），导致本地/预览与线上路径差异可能令人困惑；E2E 依赖 `/blog/` 前缀未强调。
- 未覆盖 `npm run delete:article`、`npm run ci`、`npx playwright install --with-deps chromium` 的前置安装要求；脚本参数/输出也未集中文档化。
- `scripts/` 缺少 README，根 README 的脚本说明难与实际命令保持同步。

## 6. 重构路线（4~8 步，可回滚）

1. **建立 slug 单一入口**
   - 改动：在 `src/lib/slug/` 实现 `normalizeSlug`/`slugFromTitle`/`ensureUniqueSlugs`/`buildPostUrl`/`parsePostSlugFromPath`，补文档与单测；迁移 `scripts/slug.ts` 逻辑并更新 Notion/导入调用。
   - 风险：slug 生成差异导致路由/文件名变更。
   - 回滚：保留迁移前 commit（仅文档+模块新增），回退即可恢复旧行为。
   - 验证：`npm run test`（含 slug 单测）、针对 slug 场景的新增用例。

2. **脚本拆层与目录规范** (`scripts/{notion,content,release,lib}`，其中 `release` 预留发布/构建相关脚本位)
   - 改动：为 `notion-sync`/`content-import`/`fix-math`/`delete-article` 划分 CLI 层 + `scripts/lib/*` 纯逻辑；统一参数解析与 logger 接口；移动至对应子目录并新增 `scripts/README.md`。
   - 风险：路径/导入错误导致脚本失效。
   - 回滚：逐脚本独立提交，必要时还原单个脚本的 commit。
   - 验证：`npm run notion:sync`、`npm run import:content -- --url=https://example.com/demo`（或仓库内 e2e 使用的示例页）、`npm run lint`（覆盖脚本格式）。

3. **slug 应用端对齐**
   - 改动：`content-import`、`notion-sync`、文件命名、图片目录迁移均通过 `src/lib/slug` API；必要时为 slug 冲突返回结构化结果。
   - 风险：历史内容 slug 变化或图片路径错位。
   - 回滚：保留迁移前内容备份；按 commit 还原并还原生成目录。
   - 验证：针对 slug 冲突/中文/emoji/连续符号的单测；跑 `npm run test`。

4. **稳定工具层与导入规范**
   - 改动：在 `src/lib/utils` 聚合通用函数，更新组件/页面/脚本的导入路径；`src/utils` 仅保留兼容出口或移除。
   - 风险：相对路径或别名更新遗漏。
   - 回滚：逐文件 revert；保持 lib 导出旧别名以平滑过渡。
   - 验证：`npm run check` + `npm run lint`。

5. **脚本文档与根 README 对齐**
   - 改动：补充 `scripts/README.md`（用途/输入/输出/示例/常见问题），更新根 `README.md`（项目定位、BASE_URL、Notion 同步、脚本、测试、贡献指南）。
   - 风险：文档与脚本再次偏离。
   - 回滚：文档单独提交，可独立撤销。
   - 验证：`npm run md:lint`。

6. **回归与质量门禁**
   - 改动：无代码，只运行全量质量命令并补充遗漏的 Playwright 浏览器安装说明。
   - 风险：CI 时间/资源。
   - 回滚：不涉及代码；如有问题重跑或在 CI 配置补充安装步骤。
   - 验证：依次执行 `npm run check && npm run lint && npm run test && npx playwright install --with-deps chromium && npm run test:e2e`。
