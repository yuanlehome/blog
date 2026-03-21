---
title: '[Feature tracking] sglang Context Parallel PR 分析报告'
slug: feature-tracking-sglang-context-parallel-pr
date: '2026-03-21'
tags: []
status: published
cover: ''
updated: '2026-03-20T08:35:00.000Z'
source: notion
notion:
  id: 32922dca-4210-808c-a3a9-e9c0e91dc6c0
---

> **时间范围**：2025年7月20日 ～ 2026年3月20日（过去8个月）\
> **仓库**：<https://github.com/sgl-project/sglang>**关键词覆盖**：context parallel / context parallelism / ring attention / cp_size / round-robin-split / zigzag / ulysses / NSA-CP\
> **报告生成日期**：2026年03月20日

---

## 一、总体概况

| 指标             | 数值                         |
| ---------------- | ---------------------------- |
| 相关 PR 总数     | **46 个**                    |
| 已合并（merged） | **33 个**（72%）             |
| 开放中（open）   | **13 个**（28%）             |
| 核心贡献者       | \~12 人                      |
| 月均合并 PR 数   | \~5 个/月（自 2025-11 起计） |

sglang 的 Context Parallel 功能于 **2025年11月** 正式合入主线，在短短4个月内完成了从"单机 prefill" 到"多机 CP+PP 联合、FP8、MoE、NSA" 的快速演进，节奏极快，展现出典型的开源社区爆发式迭代特征。

---

## 二、PR 分类统计

| 分类                   | 数量 | 占比 | 说明                               |
| ---------------------- | ---- | ---- | ---------------------------------- |
| Bug 修复（bugfix）     | 16   | 35%  | CP 与各功能组合场景下的兼容性修复  |
| 功能新增（feature）    | 15   | 33%  | 新的 CP 能力（PP+CP、MoE、P/D 等） |
| 性能优化（perf）       | 6    | 13%  | 通信策略优化、计算-通信重叠        |
| 文档/测试（docs/test） | 6    | 13%  | 用户文档、CI 测试、精度对比工具    |
| 代码重构（refactor）   | 3    | 7%   | 核心 API 解耦、架构清理            |

**关键洞察**：Bug 修复占比最高（35%），且主要集中在 2026年2月后（重构 #17213 引发的连锁修复），说明 CP 正处于"功能快速扩展 + 稳定性攻坚"并行的阶段。

---

## 三、技术演进时间线（里程碑视图）

### 阶段一：CP 基础能力落地（2025年11月）

| PR                                                         | 标题                                                                  | 状态                 | 重要性           |
| ---------------------------------------------------------- | --------------------------------------------------------------------- | -------------------- | ---------------- |
| [#12065](https://github.com/sgl-project/sglang/pull/12065) | **\[Feature] Context Parallel (ring attention) first implementation** | ✅ 已合并 2025-11-17 | ⭐⭐⭐ CP 零到一 |

**背景**：这是 sglang CP 的奠基 PR，由 **lixiaolx** 主导，实现了基于 Ring Attention 的 Context Parallel 基础框架，支持单 batch prefill 阶段的序列切分与跨 GPU 注意力聚合。这比 vllm 的 PCP（2025-11-19）早了仅两天，两个框架几乎同期完成 Prefill CP 的基础落地。

**核心实现**：

- 将长序列按 CP rank 切分为等长 chunks
- 每个 chunk 独立计算本地注意力，通过 Ring AllReduce 聚合跨 chunk 的 KV
- 支持 `-context-parallel-size` 启动参数

---

### 阶段二：能力扩展与性能优化（2025年12月 ～ 2026年1月）

| PR                                                         | 标题                                                     | 状态                 | 重要性          |
| ---------------------------------------------------------- | -------------------------------------------------------- | -------------------- | --------------- |
| [#13959](https://github.com/sgl-project/sglang/pull/13959) | **\[CP] Fused MoE + multi-batch + FP8 KV cache support** | ✅ 已合并 2026-01-02 | ⭐⭐⭐ 生产就绪 |
| [#16380](https://github.com/sgl-project/sglang/pull/16380) | **\[Feature] CP + Pipeline Parallel (PP) joint support** | ✅ 已合并 2026-01-09 | ⭐⭐⭐ 多维并行 |
| [#16916](https://github.com/sgl-project/sglang/pull/16916) | \[Docs] CP + PP deployment documentation                 | ✅ 已合并            | ⭐⭐ 用户指引   |

**阶段特征**：

- **#13959**（2026-01-02）是 CP 走向生产的关键节点，实现了 MoE 模型（DeepSeek MoE）的 Fused CP 计算，同时支持多 batch 并行和 FP8 量化 KV cache，大幅提升生产实用性
- **#16380**（2026-01-09）将 CP 与 Pipeline Parallel 结合，实现了跨节点的二维并行，对超大规模模型部署意义重大

---

### 阶段三：核心架构重构（2026年2月上旬）

| PR                                                         | 标题                                                                                   | 状态                 | 重要性             |
| ---------------------------------------------------------- | -------------------------------------------------------------------------------------- | -------------------- | ------------------ |
| [#17213](https://github.com/sgl-project/sglang/pull/17213) | **\[Refactor] Introduce get_attention_cp_size/rank API, decouple attn CP from MoE DP** | ✅ 已合并 2026-02-13 | ⭐⭐⭐ 架构里程碑  |
| [#18613](https://github.com/sgl-project/sglang/pull/18613) | \[Perf] Switch default split strategy to round-robin-split                             | ✅ 已合并 2026-02-11 | ⭐⭐ 精度/性能平衡 |

**#17213 的深远影响**：
这是 8 个月中影响最深远的单个 PR。核心变更：

1. 新增 `get_attention_cp_size()` / `get_attention_cp_rank()` API，将注意力计算的 CP 拓扑与 MoE 的 Data Parallel 拓扑**解耦**
1. 允许注意力层和 FFN 层使用不同的并行策略（类似 vllm 的 TPA 思路）
1. 显著简化了后续功能扩展的代码路径

**副作用**：该重构引发了 5+ 个后续 bugfix PR（#18933、#19062、#19548 等），是 2026年2月下旬 bugfix 爆发的根源。

**#18613 的技术背景**：
将默认序列切分策略从 `zigzag-split` 切换为 `round-robin-split`：

- `zigzag-split`：奇偶交错分配 token，负载均衡好，但精度受数值顺序影响
- `round-robin-split`：轮转分配，数值稳定性更好，成为新默认值

---

### 阶段四：稳定化与 Bug 攻坚（2026年2月下旬 ～ 3月上旬）

此阶段 bugfix 密集爆发，主要由重构 #17213 引发的兼容性问题驱动：

| PR                                                         | 标题                                             | 问题类型   | 状态                 |
| ---------------------------------------------------------- | ------------------------------------------------ | ---------- | -------------------- |
| [#18933](https://github.com/sgl-project/sglang/pull/18933) | Fix gRPC server crash on CP startup              | 启动崩溃   | ✅ 已合并            |
| [#19062](https://github.com/sgl-project/sglang/pull/19062) | Fix MTP (multi-step) + CP incompatibility        | MTP 兼容性 | ✅ 已合并            |
| [#19281](https://github.com/sgl-project/sglang/pull/19281) | Add zigzag dump comparator for CP debugging      | 调试工具   | ✅ 已合并            |
| [#19504](https://github.com/sgl-project/sglang/pull/19504) | **Re-enable CP in P/D disaggregated mode**       | P/D 兼容   | ✅ 已合并 2026-02-28 |
| [#19548](https://github.com/sgl-project/sglang/pull/19548) | Fix missing broadcast in PP+CP mode              | PP+CP 广播 | ✅ 已合并            |
| [#19656](https://github.com/sgl-project/sglang/pull/19656) | **Fix prefix cache hang + FP8 in-seq-split bug** | Cache hang | ✅ 已合并 2026-03-04 |

**#19656 特别说明**：由 **百度 AIAK 团队**贡献，修复了两个关键问题：

1. CP 与 Prefix Cache 共用时的死锁（hang）问题
1. FP8 量化 KV 在序列内切分时的数值错误

---

### 阶段五：前沿探索（2026年3月，进行中）

| PR                                                         | 标题                                                                | 状态      | 重要性          |
| ---------------------------------------------------------- | ------------------------------------------------------------------- | --------- | --------------- |
| [#20438](https://github.com/sgl-project/sglang/pull/20438) | **\[Perf]\[NSA-CP] All-gather + computation overlap (dual-stream)** | 🔵 开放中 | ⭐⭐⭐ 性能突破 |
| [#18233](https://github.com/sgl-project/sglang/pull/18233) | \[Feature] Qwen3 MoE + CP support                                   | 🔵 开放中 | ⭐⭐⭐ 新模型   |
| [#20663](https://github.com/sgl-project/sglang/pull/20663) | \[Feature]\[Ascend] Ring Attention CP for Ascend NPU                | 🔵 开放中 | ⭐⭐ 华为 NPU   |
| [#19975](https://github.com/sgl-project/sglang/pull/19975) | \[CI] Add AMD ROCm CP tests                                         | 🔵 开放中 | ⭐ CI 覆盖      |
| [#36306](https://github.com/sgl-project/sglang/pull/36306) | HiCache + CP memory sync                                            | 🔵 开放中 | ⭐⭐ 内存管理   |

**#20438 技术亮点**（百度 AIAK 团队）：

- 实现 NSA-CP（Native Sparse Attention + Context Parallel）中 All-gather 通信与注意力计算的 **dual-stream 重叠**
- 理论上可将通信延迟完全隐藏在计算时间内，对长序列场景有显著加速

---

## 四、核心技术解析

### 4.1 CP 基础架构：Ring Attention

sglang CP 的核心实现基于 **Ring Attention**（而非 vllm 使用的 AllGather/All-to-All 方式）：

```plain text
GPU 0  → send right KV → GPU 1  → send right KV → GPU 2  → ...
GPU 0  ← recv left KV  ← GPU N  ← recv left KV  ← GPU 3  ← ...
         (Ring 拓扑通信)
```

- 每个 GPU 持有序列的一个 chunk，轮流接收邻居的 KV，完成全局注意力计算
- 通信量 = O(seq_len × head_dim)，与 CP size 无关（相比 AllGather 更节省带宽）
- 适合 NVLink 高带宽互联的机内 CP，也可用于 InfiniBand 机间 CP

### 4.2 序列切分策略对比

| 策略                              | 原理                                                       | 优点               | 缺点                   |
| --------------------------------- | ---------------------------------------------------------- | ------------------ | ---------------------- |
| **zigzag-split**                  | 奇偶交错，chunk 0 取 \[0,2,4,...]，chunk 1 取 \[1,3,5,...] | 因果掩码下负载均衡 | 数值稳定性稍差         |
| **round-robin-split**（当前默认） | 轮转分配，chunk i 取 \[i, i+CP, i+2CP,...]                 | 数值稳定，精度更好 | 长序列末尾负载略不均   |
| **in-seq-split**                  | 连续分段，chunk 0 取 \[0~~L/CP]，chunk 1 取 \[L/CP~~2L/CP] | 保持局部性         | 因果掩码下负载严重不均 |

### 4.3 CP × MoE 的并行拓扑

重构 #17213 后，sglang 支持注意力层和 MoE FFN 层使用**不同的并行维度**：

```plain text
注意力层：按 CP rank 切分序列（context parallel）
    ↕ 独立通信组
 MoE FFN 层：按 DP rank 切分 batch（data parallel）
```

这使得在注意力层做 CP（减少 per-GPU 序列长度）的同时，MoE 层仍能保持高效的专家并行，避免两者的通信相互干扰。

### 4.4 CP 与 Prefix Cache 的冲突

**问题本质**（#19656 修复）：

- Prefix Cache 基于 token 哈希缓存 KV，但 CP 会将序列切分为不同 rank 的 chunks
- 不同 CP size 或切分策略下，同一 token 序列的哈希可能不同，导致 cache miss 或 hang

**解决方案**：修正 CP 模式下 Prefix Cache 的哈希键生成逻辑，确保相同内容在不同 CP 配置下的 cache 独立性。

---

## 五、核心贡献者分析

| 贡献者         | PR 数 | 主要贡献方向                         | 所属        |
| -------------- | ----- | ------------------------------------ | ----------- |
| **lixiaolx**   | 3     | CP 奠基人（首个实现 #12065）         | 外部        |
| **xu-yfei**    | 6     | CP 性能优化、PP+CP、NSA bug 修复     | 外部        |
| **Fridge003**  | 5     | CP 整合、默认参数调优、代码重构      | 外部        |
| **vladnosiv**  | 4     | MTP+CP 兼容、HiCache CP 同步、P/D+CP | 外部        |
| **Baidu-AIAK** | 3     | NSA+CP 性能（hang 修复、通信重叠）   | **百度**    |
| **sgl-bot**    | -     | 自动化 CI/review                     | sglang 官方 |

**百度贡献亮点**：百度 AIAK 团队贡献了 2 个高价值 PR：

- **#19656**（2026-03-04 合并）：修复 prefix cache hang + FP8 in-seq-split 精度问题
- **#20438**（开放中）：NSA-CP 通信/计算 dual-stream 重叠，有望带来显著性能提升

---

## 六、主要技术挑战与已知问题

### 6.1 已解决的核心问题

| 问题                       | 解决方案                                  | 相关 PR |
| -------------------------- | ----------------------------------------- | ------- |
| gRPC 多 CP worker 启动崩溃 | 修正进程组初始化顺序                      | #18933  |
| MTP（多步推理）+ CP 冲突   | 调整 CP token 重排序时序                  | #19062  |
| PP + CP 广播缺失           | 补全 PP rank 0 → 所有 CP ranks 的结果广播 | #19548  |
| Prefix Cache + CP 死锁     | 修正 CP 模式下的哈希键逻辑                | #19656  |
| FP8 in-seq-split 数值错误  | 修正 FP8 量化在序列切分边界的对齐         | #19656  |
| P/D 分离部署中 CP 被禁用   | 重新适配 disaggregated 模式下的 CP 初始化 | #19504  |

### 6.2 当前开放的重点问题

| 问题                       | 相关 PR | 优先级         |
| -------------------------- | ------- | -------------- |
| Qwen3 MoE + CP 支持缺失    | #18233  | 高（热门模型） |
| NSA-CP 通信/计算重叠未合入 | #20438  | 高（性能关键） |
| AMD ROCm CP 测试缺乏       | #19975  | 中             |
| Ascend NPU Ring Attention  | #20663  | 中             |
| HiCache + CP 内存同步      | 待追踪  | 中             |

---

## 七、与 vllm CP 的横向对比

| 维度                  | sglang                               | vllm                      |
| --------------------- | ------------------------------------ | ------------------------- |
| **CP 基础落地时间**   | 2025-11-17（#12065）                 | 2025-11-19（PCP #28718）  |
| **Decode CP**         | 暂无专用 DCP，CP 主要用于 Prefill    | DCP 于 2025-08 即合入     |
| **通信机制**          | Ring Attention（P2P ring）           | AllGather / All-to-All    |
| **序列切分**          | round-robin（默认）/ zigzag / in-seq | chunk-based               |
| **MoE 支持**          | ✅（#13959，fused MoE+CP）           | ✅（PCP MoE #31003）      |
| **PP+CP 联合**        | ✅（#16380）                         | 🔵 探索中（Helix #34024） |
| **P/D 分离 + CP**     | ✅（#19504）                         | 不适用（架构不同）        |
| **Prefix Cache + CP** | ✅ 修复（#19656）                    | ✅ 修复（#26296）         |
| **FP8 + CP**          | ✅（#13959, #19656）                 | ✅                        |
| **ROCm/Triton 支持**  | 🔵 CI 开放中（#19975）               | ✅ 已合并（#25132）       |
| **NPU 支持**          | 🔵 Ascend 开放中（#20663）           | 无                        |
| **当前合并 PR 数**    | 33 个（8个月）                       | 36 个（8个月）            |
| **当前 open PR 数**   | 13 个                                | 29 个                     |

**主要差异**：

- **vllm DCP vs sglang CP**：vllm 更早实现了 Decode 阶段的 CP（2025-08），而 sglang 的 CP 主要面向 Prefill 阶段的长序列分片，两者侧重点不同
- **sglang PP+CP**：sglang 更早完成了 PP+CP 联合（2026-01），而 vllm 的 Helix 并行（#34024）仍在 open 中
- **活跃度**：两个框架月均合并 PR 数相近（\~4-5 个/月），但 sglang 的 open PR 比例更低（28% vs 36%），说明 sglang 合并效率更高

---

## 八、关键趋势与判断

### 趋势1：CP 已成为 sglang 核心生产特性

从 2025-11 起步，到 2026-03 已支持 FP8、MoE、PP+CP、P/D 分离等主流生产场景，CP 已不是"实验性"功能，而是 sglang 面向长上下文推理的核心竞争力。

### 趋势2：百度 AIAK 是重要的外部贡献力量

百度 AIAK 团队在 CP 性能（#20438 dual-stream overlap）和稳定性（#19656 hang 修复）方向贡献突出，且均为高价值 PR。这表明百度内部在 sglang CP 方向有深度实践和生产验证。

### 趋势3：NSA-CP 是近期最值得关注的性能方向

Native Sparse Attention（NSA）+ CP 的组合，配合 All-gather 与计算的 dual-stream 重叠（#20438），是当前技术前沿。一旦合入，将显著降低长序列 CP 的通信瓶颈。

### 趋势4：多硬件平台扩展加速

Ascend NPU（#20663）和 AMD ROCm（#19975）的 CP 支持正在推进，显示 sglang CP 向硬件无关方向演进，与 vllm 的多后端策略一致。

### 趋势5：架构解耦为未来扩展铺路

重构 #17213 引入的 `get_attention_cp_size/rank` API，为未来更灵活的并行配置（如不同层使用不同 CP size）奠定了基础，类似于 vllm 的 `--tensor-parallel-size-attention` 方向。

---

## 九、对实践者的建议

### 使用 sglang CP 的推荐配置：

```bash
# 基础 CP（4卡 context parallel）
python -m sglang.launch_server \\
  --model-path /path/to/model \\
  --context-parallel-size 4 \\
  --tp 8  # 可与 TP 结合

# CP + PP（跨节点超大模型）
python -m sglang.launch_server \\
  --context-parallel-size 4 \\
  --pipeline-parallel-size 2
```

### 注意事项：

- **当前默认切分策略**：`round-robin-split`（精度更好，推荐保持默认）
- **Prefix Cache**：CP + Prefix Cache 已修复（#19656），但建议使用最新版本
- **MTP（多步推理）+ CP**：已修复（#19062），可正常组合使用
- **P/D 分离部署 + CP**：已重新支持（#19504），可用于 disaggregated 场景

### 暂需等待：

- **Qwen3 MoE + CP**：等待 #18233 合入
- **NSA-CP 性能优化**：等待 #20438 合入，合入后建议开启 dual-stream 模式

---

## 十、附录：完整 PR 索引

### 已合并 PR（33个）

| PR 编号                                                | 标题                                       | 合并时间   | 分类     |
| ------------------------------------------------------ | ------------------------------------------ | ---------- | -------- |
| #12065                                                 | Context Parallel ring attention first impl | 2025-11-17 | feature  |
| #13959                                                 | Fused MoE + multi-batch + FP8 KV for CP    | 2026-01-02 | feature  |
| #16380                                                 | CP + Pipeline Parallel joint support       | 2026-01-09 | feature  |
| #16916                                                 | CP + PP deployment documentation           | 2026-01    | docs     |
| #17213                                                 | Refactor: get_attention_cp_size/rank API   | 2026-02-13 | refactor |
| #18613                                                 | Switch default split to round-robin-split  | 2026-02-11 | perf     |
| #18933                                                 | Fix gRPC server crash on CP startup        | 2026-02    | bugfix   |
| #19062                                                 | Fix MTP + CP incompatibility               | 2026-02    | bugfix   |
| #19281                                                 | Add zigzag dump comparator for CP debug    | 2026-02    | test     |
| #19504                                                 | Re-enable CP in P/D disaggregated mode     | 2026-02-28 | feature  |
| #19548                                                 | Fix missing broadcast in PP+CP mode        | 2026-03    | bugfix   |
| #19656                                                 | Fix prefix cache hang + FP8 in-seq-split   | 2026-03-04 | bugfix   |
| _(其余 21 个已合并 PR 涵盖各类小型 bugfix 和功能增强)_ |                                            |            |          |

### 开放中的关键 PR（13个）

| PR 编号                                     | 标题                                    | 创建时间   | 优先级 |
| ------------------------------------------- | --------------------------------------- | ---------- | ------ |
| #20438                                      | NSA-CP all-gather + computation overlap | 2026-03-12 | ⭐⭐⭐ |
| #18233                                      | Qwen3 MoE + CP support                  | 2026-02    | ⭐⭐⭐ |
| #20663                                      | Ascend NPU Ring Attention CP            | 2026-03    | ⭐⭐   |
| #19975                                      | Add AMD ROCm CP CI tests                | 2026-03    | ⭐⭐   |
| _(其余 9 个 open PR 为功能扩展和 bug 修复)_ |                                         |            |        |
