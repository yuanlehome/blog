# 贡献指南

感谢你愿意为本项目贡献力量！请按照以下流程提交修改，以确保协作顺畅。

## 开发流程
1. **Fork & Clone**：Fork 仓库并在本地克隆。
2. **创建分支**：为每个改动创建独立分支：
   ```bash
   git checkout -b feat/your-feature
   ```
3. **安装依赖**：
   ```bash
   npm install
   ```
4. **编写与测试**：完成开发后至少运行一次构建或相关脚本，确保基本可用：
   ```bash
   npm run build
   # 或针对内容同步
   npm run notion:sync
   ```
5. **格式化代码**：
   ```bash
   npm run format
   ```
6. **提交信息**：使用清晰的提交说明（如 `feat: add rss feed`、`docs: update readme`）。
7. **发起 Pull Request**：说明变更目的、测试情况和相关 Issue 链接。

## Issue 规范
- 标题简洁明了，正文包含：
  - 复现步骤 / 需求背景
  - 预期行为 vs. 实际行为
  - 运行环境（系统、浏览器、Node 版本等）
- 提问前可先搜索现有 Issue，避免重复。

## 提交内容建议
- 文档：保持 README 与配置文件同步更新，必要时增加使用示例。
- 代码：
  - 遵循已有代码风格，避免引入未使用的依赖。
  - 需要新增脚本或命令时，请更新 README 的“常用命令”或“快速开始”。
- 测试：若改动会影响构建或内容同步，请附上相关命令的运行结果。

## 行为准则
请阅读并遵守 [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)。感谢你的尊重与合作！
