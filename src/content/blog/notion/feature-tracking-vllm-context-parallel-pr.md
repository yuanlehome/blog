---
title: '[Feature tracking] vllm Context Parallel PR 分析报告'
slug: feature-tracking-vllm-context-parallel-pr
date: '2026-03-21'
tags: []
status: published
cover: ''
updated: '2026-03-20T08:35:00.000Z'
source: notion
notion:
  id: 32122dca-4210-80af-8e2d-ecc520f301b1
---

> **时间范围**：2025年7月20日 ～ 2026年3月20日（过去8个月）\
> **仓库**：<https://github.com/vllm-project/vllm>**关键词覆盖**：context parallel、context parallelism、DCP（Decode Context Parallelism）、PCP（Prefill Context Parallelism）、ring attention、sequence parallel\
> **报告生成日期**：2026年03月20日

---

## 一、总体概况

| 指标                   | 数值             |
| ---------------------- | ---------------- |
| 相关 PR 总数           | **80 个**        |
| 已合并（merged）       | **36 个**（45%） |
| 开放中（open）         | **29 个**（36%） |
| 已关闭未合并（closed） | **15 个**（19%） |
| 核心贡献者             | \~15 人          |
| 月均合并 PR 数         | \~4.5 个/月      |

过去8个月，vllm 社区在 Context Parallel 方向投入了大量精力，从无到有地构建了完整的 DCP（Decode Context Parallelism）体系，并在 2025 年 11 月正式引入 PCP（Prefill Context Parallelism）。整个演进节奏清晰：**先 Decode 侧铺底，再 Prefill 侧跟进，最后迈向统一并行框架**。

---

## 二、PR 分类统计

| 类别                    | 数量 | 占比 | 说明                                |
| ----------------------- | ---- | ---- | ----------------------------------- |
| Bug 修复（bugfix）      | 23   | 29%  | 修复 DCP/PCP 在各场景下的正确性问题 |
| 功能扩展（enhancement） | 22   | 28%  | 扩展 CP 的硬件后端、模型架构支持    |
| 新功能（feature）       | 19   | 24%  | 全新的 CP 功能（PCP、Helix 并行等） |
| 用户体验（UX）          | 8    | 10%  | 错误提示、配置简化、文档            |
| 性能优化（perf）        | 3    | 4%   | 通信性能、内存访问优化              |
| 测试/CI（test）         | 3    | 4%   | 集成测试、CI 覆盖                   |
| 文档（docs）            | 2    | 3%   | 用户手册、API 文档                  |

**关键洞察**：bugfix 占比最高（29%），说明 CP 功能整体仍处于快速迭代 + 稳定化阶段，功能边界在持续扩展的同时，正确性保证是核心挑战。

---

## 三、技术演进时间线（里程碑视图）

### 阶段一：DCP 基础能力落地（2025年8月）

这一阶段是整个 Context Parallel 体系的起点，主要针对 **DeepSeek MLA 架构**实现 Decode 阶段的 Context Parallelism。

| PR                                                        | 标题                                          | 状态                 | 重要性              |
| --------------------------------------------------------- | --------------------------------------------- | -------------------- | ------------------- |
| [#23791](https://github.com/vllm-project/vllm/pull/23791) | Cuda kernels for upcoming DCP feature         | ✅ 已合并 2025-08-28 | ⭐⭐⭐ 底层内核基础 |
| [#23734](https://github.com/vllm-project/vllm/pull/23734) | Support Decode Context Parallel (DCP) for MLA | ✅ 已合并 2025-09-06 | ⭐⭐⭐ DCP 首个特性 |

**背景**：DeepSeek v2/v3 使用 MLA（Multi-head Latent Attention）架构，其 KV Cache 压缩特性使得跨节点的 KV 分片和 AllGather 更加经济。DCP 首先在 MLA 上实现，是合理的技术选择。

---

### 阶段二：DCP 多后端扩展（2025年9月）

MLA DCP 落地后，社区迅速跟进扩展到多个注意力后端和硬件平台。

| PR                                                        | 标题                                    | 状态                 | 重要性                |
| --------------------------------------------------------- | --------------------------------------- | -------------------- | --------------------- |
| [#24385](https://github.com/vllm-project/vllm/pull/24385) | DCP on Blackwell with CUTLASS MLA       | ✅ 已合并 2025-09-08 | ⭐⭐⭐ H200/B200 支持 |
| [#24453](https://github.com/vllm-project/vllm/pull/24453) | DCP for FLASH_ATTN_MLA backend          | ✅ 已合并 2025-09-10 | ⭐⭐ FlashAttn 后端   |
| [#24864](https://github.com/vllm-project/vllm/pull/24864) | Support DCP for GQA with FlashAttention | ✅ 已合并 2025-10-14 | ⭐⭐⭐ GQA 架构支持   |
| [#25132](https://github.com/vllm-project/vllm/pull/25132) | DCP for Triton backend (ROCm)           | ✅ 已合并 2025-09-25 | ⭐⭐ AMD GPU 支持     |
| [#25414](https://github.com/vllm-project/vllm/pull/25414) | Remove contiguous output req for CP MLA | ✅ 已合并 2025-09-23 | ⭐ 正确性修复         |

**关键判断**：DCP 从 MLA（DeepSeek）扩展到 GQA（Llama/Qwen 等主流架构），并覆盖 FlashAttention 和 Triton（ROCm）两大后端，标志着 DCP 从"实验性"走向"通用化"。

---

### 阶段三：DCP 稳定化与生产就绪（2025年10月-11月）

这一阶段以大量 bugfix 为主，同时完善了用户文档和部署指南。

| PR                                                        | 标题                                           | 状态                 | 问题类型            |
| --------------------------------------------------------- | ---------------------------------------------- | -------------------- | ------------------- |
| [#26296](https://github.com/vllm-project/vllm/pull/26296) | Fix block_size of hash in DCP prefix caching   | ✅ 已合并            | Prefix Cache 兼容性 |
| [#26509](https://github.com/vllm-project/vllm/pull/26509) | CP: fix correct_attn_out for 4-D views         | ✅ 已合并            | 注意力输出修正      |
| [#26574](https://github.com/vllm-project/vllm/pull/26574) | Set default CUDAGraphMode to PIECEWISE for DCP | ✅ 已合并            | CUDA Graph 兼容性   |
| [#25049](https://github.com/vllm-project/vllm/pull/25049) | DCP with query length > 1 (MTP) with FA3       | ✅ 已合并 2025-10-09 | MTP（多步推理）支持 |
| [#27518](https://github.com/vllm-project/vllm/pull/27518) | Fix wrong dcp_local_seq_lens calc              | ✅ 已合并            | 序列长度计算 Bug    |
| [#27929](https://github.com/vllm-project/vllm/pull/27929) | DCP: check return_lse for all layers           | ✅ 已合并            | 多层 LSE 计算 Bug   |
| [#28100](https://github.com/vllm-project/vllm/pull/28100) | Fix DCP Assert (reorder_batch_threshold)       | ✅ 已合并            | 断言错误            |
| [#26696](https://github.com/vllm-project/vllm/pull/26696) | DCP kv_cache interleave size > 1               | ✅ 已合并 2025-11-08 | KV Cache 交错支持   |
| [#26877](https://github.com/vllm-project/vllm/pull/26877) | DCP deployment documentation                   | ✅ 已合并            | 用户手册            |

**主要痛点集中在**：① Prefix Caching 与 CP 的兼容性；② CUDA Graph 的 PIECEWISE 模式适配；③ 多步推理（MTP）下的序列长度计算。

---

### 阶段四：PCP 正式引入（2025年11月-12月）

PCP（Prefill Context Parallelism）与 DCP 原理相近，但作用于 Prefill 阶段，对长上下文场景（如 128K+ tokens）尤为关键。

| PR                                                        | 标题                                                        | 状态                 | 重要性               |
| --------------------------------------------------------- | ----------------------------------------------------------- | -------------------- | -------------------- |
| [#28718](https://github.com/vllm-project/vllm/pull/28718) | **\[Feature] Prefill Context Parallel (PCP) basic support** | ✅ 已合并 2025-11-19 | ⭐⭐⭐ PCP 里程碑    |
| [#29094](https://github.com/vllm-project/vllm/pull/29094) | Fix block size in block_table with PCP                      | ✅ 已合并 2025-11-22 | ⭐⭐ 正确性修复      |
| [#29487](https://github.com/vllm-project/vllm/pull/29487) | Fix num_q_heads on DCP for Flashinfer                       | ✅ 已合并 2025-12-05 | ⭐ Flashinfer 兼容   |
| [#29952](https://github.com/vllm-project/vllm/pull/29952) | PCP\&DCP CUDA graph check refactor                          | ✅ 已合并 2025-12-05 | ⭐⭐ CUDA Graph 重构 |
| [#30309](https://github.com/vllm-project/vllm/pull/30309) | Fix DCP accuracy with FLASH_ATTN_MLA                        | ✅ 已合并 2025-12-09 | ⭐⭐ 精度问题        |
| [#31003](https://github.com/vllm-project/vllm/pull/31003) | PCP basic support for MoE model                             | ✅ 已合并 2025-12-31 | ⭐⭐⭐ MoE 支持      |
| [#28723](https://github.com/vllm-project/vllm/pull/28723) | Support PCP for GQA flashinfer                              | 🔵 开放中            | 后端扩展             |
| [#28988](https://github.com/vllm-project/vllm/pull/28988) | Support PCP with MLA                                        | 🔵 开放中            | MLA 扩展             |

**重要意义**：PCP 使得 vllm 在 Prefill 阶段也能横向扩展算力，这对长上下文推理（128K-1M tokens）至关重要。MoE 模型的 PCP 支持则直接服务于 DeepSeek MoE 等大规模生产模型。

---

### 阶段五：架构重构与次世代并行（2026年1月-3月）

这一阶段以大规模重构和下一代并行框架探索为特征。

| PR                                                        | 标题                                             | 状态                 | 重要性                   |
| --------------------------------------------------------- | ------------------------------------------------ | -------------------- | ------------------------ |
| [#33239](https://github.com/vllm-project/vllm/pull/33239) | Move DCP validation to ParallelConfig            | ✅ 已合并 2026-01-30 | ⭐⭐ 配置解耦            |
| [#34179](https://github.com/vllm-project/vllm/pull/34179) | **DCP support for GPU model runner v2**          | ✅ 已合并 2026-02-18 | ⭐⭐⭐ 新执行引擎        |
| [#34786](https://github.com/vllm-project/vllm/pull/34786) | DCP simplification for model runner v2           | ✅ 已合并 2026-02-18 | ⭐⭐ 代码简化            |
| [#34883](https://github.com/vllm-project/vllm/pull/34883) | **Add All-to-All communication backend for DCP** | ✅ 已合并 2026-03-04 | ⭐⭐⭐ 通信后端优化      |
| [#34024](https://github.com/vllm-project/vllm/pull/34024) | **Add Helix (Context + Tensor) Parallelism**     | 🔵 开放中            | ⭐⭐⭐ 联合并行框架      |
| [#33403](https://github.com/vllm-project/vllm/pull/33403) | \[WIP] pcp alternative impl                      | 🔵 开放中            | ⭐⭐ PCP 替代实现        |
| [#36306](https://github.com/vllm-project/vllm/pull/36306) | PCP alternative implementation                   | 🔵 开放中            | ⭐⭐ PCP 替代实现        |
| [#36287](https://github.com/vllm-project/vllm/pull/36287) | Add --tensor-parallel-size-attention for TPA     | 🔵 开放中            | ⭐⭐ Tensor+Context 分离 |
| [#35206](https://github.com/vllm-project/vllm/pull/35206) | Support Sequence Parallel for Model Runner v2    | 🔵 开放中            | ⭐⭐ 序列并行            |

---

## 四、核心技术解析

### 4.1 DCP（Decode Context Parallelism）

**原理**：在 Decode 阶段（自回归生成），将不同请求的 KV Cache 分布在多个 GPU 上，通过 AllGather 或 All-to-All 通信聚合注意力计算。

**技术实现要点**：

- **通信方式**：从 AllGather（简单，带宽需求高）→ All-to-All（#34883，更高效）演进
- **KV Cache 分片**：`dcp_local_seq_lens` 管理每个 GPU 上的本地 KV 长度
- **注意力后端**：支持 FlashAttention v2/v3、FLASH_ATTN_MLA、Triton（ROCm）、Flashinfer
- **LSE 聚合**：各 GPU 的 log-sum-exp 需要合并才能得到正确的 softmax 结果
- **CUDA Graph 兼容**：使用 PIECEWISE 模式（而非整图模式）以支持动态形状

**适用场景**：大 Batch + 长历史 KV（高 KV Cache 压力场景），如多轮对话、RAG 推理

### 4.2 PCP（Prefill Context Parallelism）

**原理**：在 Prefill 阶段（Prompt 处理），将长输入序列切分为多个 chunks，分布在不同 GPU 上，各 GPU 独立计算本地注意力后通过 Ring Attention 或 All-to-All 聚合。

**技术实现要点**：

- **Causal Mask 处理**：Prefill 使用因果注意力，跨块的因果关系需要特殊处理（local chunk 的 KV 必须能访问之前 chunks 的 KV）
- **块表（block_table）兼容**：PCP 改变了 token 在物理内存中的分布，`block_table` 需要相应调整（#29094）
- **MoE 支持**：MoE 模型的 FFN 计算不涉及跨序列依赖，PCP 对 MoE 更加友好（#31003）
- **竞争实现**：目前有两条技术路线并行探索（#33403 vs #36306），预计将在 Q2 2026 收敛

**适用场景**：超长 Prompt 处理（128K+ tokens），如文档理解、代码分析、长上下文推理

### 4.3 Helix 并行（Context + Tensor 联合并行）

**原理**（#34024）：将 Context Parallelism 和 Tensor Parallelism 结合，形成二维并行网格，进一步提升大规模集群的利用率。

- **TPA（Tensor Parallel Attention）**：注意力层和 FFN 层使用不同的并行度（#36287，`-tensor-parallel-size-attention` 参数）
- **Helix 网格**：CP Rank × TP Rank 构成二维通信拓扑，优化节点内/节点间通信

这是 vllm 向类 Megatron 的"3D 并行"演进的重要信号。

---

## 五、核心贡献者分析

| 贡献者             | 主要贡献方向                                | PR 数量（估计） |
| ------------------ | ------------------------------------------- | --------------- |
| **youzhedian**     | DCP for MLA 首席贡献者，CUDA kernel         | 8+              |
| **FENP**           | DCP GQA FlashAttention，PCP MLA 扩展        | 5+              |
| **pisceskkk**      | PCP 首个实现，GQA Flashinfer                | 4+              |
| **sungsooha**      | DCP Model Runner v2，Helix 并行，All-to-All | 5+              |
| **LucasWilkinson** | PCP 替代实现，底层优化                      | 3+              |
| **qiruiyangmeta**  | Context Parallel 基础设施（token sharding） | 3               |
| **aws-bowencc**    | DCP prefill→non-DCP decode 混合推理         | 2+              |

**观察**：贡献者来自百度、AWS、Meta 等多家公司，体现了社区的多元化；其中 `youzhedian`、`FENP` 等来自百度的工程师主导了 DCP/MLA 方向的核心工作。

---

## 六、主要技术挑战与已知问题

### 6.1 已解决的核心问题

| 问题                          | 解决方案                                | 相关 PR |
| ----------------------------- | --------------------------------------- | ------- |
| Prefix Caching + DCP 兼容性   | 修正 KV block hash 中的 block_size 计算 | #26296  |
| CUDA Graph 与 DCP 不兼容      | 强制使用 PIECEWISE 模式                 | #26574  |
| MTP（多步推理）下 query_len>1 | FA3 后端适配                            | #25049  |
| LSE 多层聚合错误              | 所有层统一检查 return_lse               | #27929  |
| DCP + FLASH_ATTN_MLA 精度问题 | 修复注意力输出累加精度                  | #30309  |

### 6.2 当前开放的已知问题

- **PCP 双实现收敛**：目前有 #33403（WIP）和 #36306 两个 PCP 替代实现同时开放，存在设计决策未定的风险
- **PCP + MLA 未完成**：#28988（PCP with MLA）仍处于 Open 状态，DeepSeek 系列模型的 Prefill CP 尚不完整
- **PCP + GQA Flashinfer**：#28723 仍处于 Open 状态
- **DCP CUDA Graph 修复**：#36070 等近期 bugfix 仍在评审中
- **Helix 并行**：#34024 是大型 PR（联合并行框架），Review 复杂度高，合并周期可能较长

---

## 七、关键趋势与判断

### 趋势1：DCP 已进入"生产就绪"阶段

自 2026 年 2 月 DCP 适配 Model Runner v2（#34179）并引入 All-to-All 通信后端（#34883）后，DCP 已具备在最新执行引擎上稳定运行的能力，进入生产就绪阶段。

### 趋势2：PCP 处于"快速迭代"中，尚未稳定

PCP 于 2025 年 11 月合入基础版本后，持续出现 bugfix 和设计调整。目前存在两个并行的 PCP 替代实现（#33403 vs #36306），表明社区对 PCP 的最佳实现路径仍有争论，**预计 Q2 2026 前后完成收敛**。

### 趋势3：向"统一并行框架"演进

Helix 并行（#34024）和 TPA 分离（#36287）的出现，表明 vllm 正在从"单一并行维度"向"多维并行的统一框架"演进，目标是在大规模集群（256+ GPU）上实现最优的计算-通信 overlap。

### 趋势4：DeepSeek 架构驱动

纵观整个 CP 演进，MLA 架构（DeepSeek）是最核心的驱动力。DCP 首先在 MLA 上实现，PCP 的 MLA 扩展也在积极推进，表明 vllm 社区高度关注 DeepSeek 系列模型的推理效率。

### 趋势5：多硬件平台并重

CP 功能先后适配了 NVIDIA（CUDA / FlashAttention）、AMD（ROCm / Triton 后端）和 Blackwell（H200/B200 / CUTLASS），显示出 vllm 在硬件无关性上的持续投入。

---

## 八、各阶段 PR 合并率分析

| 阶段           | 时间段        | PR 数 | 合并率 |
| -------------- | ------------- | ----- | ------ |
| DCP 基础落地   | 2025-08       | 2     | 100%   |
| DCP 多后端扩展 | 2025-09       | 8     | 88%    |
| DCP 稳定化     | 2025-10 \~ 11 | 20    | 75%    |
| PCP 引入       | 2025-11 \~ 12 | 15    | 53%    |
| 架构重构       | 2026-01 \~ 02 | 18    | 44%    |
| 次世代并行     | 2026-03       | 17    | 12%    |

**趋势**：越到近期，Open PR 比例越高，这符合"最新功能仍在评审"的规律，并非停滞信号。

---

## 九、对实践者的建议

### 使用 DCP：

- 推荐搭配 `CUDAGraphMode=PIECEWISE`（现已为默认值）
- 在 FlashAttention v3（FA3）后端和 Blackwell GPU 上效果最佳
- 与 Prefix Caching 联用需注意 KV block 大小对齐问题（已修复）

### 使用 PCP：

- 当前建议使用 **主线版本**（#28718 已合并），等待 #33403/#36306 的替代实现稳定后再评估切换
- MoE 模型（如 DeepSeek）PCP 已支持（#31003）
- MLA 架构的 PCP（#28988）尚未合并，建议关注但暂不用于生产

### 关注 Helix：

- \#34024 是下一个值得重点关注的 PR，将带来 Context+Tensor 联合并行的统一接口

---

## 十、附录：完整 PR 索引

### 已合并 PR（36个）

| PR 编号 | 标题                                             | 合并时间   | 分类        |
| ------- | ------------------------------------------------ | ---------- | ----------- |
| #23791  | Cuda kernels for upcoming DCP feature            | 2025-08-28 | feature     |
| #23734  | Support DCP for MLA                              | 2025-09-06 | feature     |
| #24385  | DCP on Blackwell with CUTLASS MLA                | 2025-09-08 | enhancement |
| #24453  | DCP for FLASH_ATTN_MLA backend                   | 2025-09-10 | enhancement |
| #25414  | Remove contiguous output req for CP MLA          | 2025-09-23 | bugfix      |
| #25132  | DCP for Triton backend (ROCm)                    | 2025-09-25 | enhancement |
| #24864  | Support DCP for GQA with FlashAttention          | 2025-10-14 | feature     |
| #26296  | Fix block_size of hash in DCP prefix caching     | 2025-10-10 | bugfix      |
| #26509  | CP: fix correct_attn_out for 4-D views           | 2025-10-11 | bugfix      |
| #26574  | Set default CUDAGraphMode to PIECEWISE for DCP   | 2025-10-12 | bugfix      |
| #25049  | DCP with query length > 1 (MTP) with FA3         | 2025-10-09 | enhancement |
| #26877  | DCP deployment documentation                     | 2025-10    | docs        |
| #26696  | DCP kv_cache interleave size > 1                 | 2025-11-08 | enhancement |
| #27518  | Fix wrong dcp_local_seq_lens calc                | 2025-11-05 | bugfix      |
| #27929  | DCP: check return_lse for all layers             | 2025-11-05 | bugfix      |
| #28100  | Fix DCP Assert (reorder_batch_threshold)         | 2025-11-05 | bugfix      |
| #28526  | Fix local_chunk_len for DCP in reorg_kvcache     | 2025-11-13 | bugfix      |
| #25438  | DCP for GQA with Flashinfer                      | 2025-11-14 | enhancement |
| #28718  | **PCP basic support**                            | 2025-11-19 | feature     |
| #28751  | Fix DCP AssertionError (reorder_batch_threshold) | 2025-11-15 | bugfix      |
| #29094  | Fix block size in block_table with PCP           | 2025-11-22 | bugfix      |
| #29487  | Fix num_q_heads on DCP for Flashinfer            | 2025-12-05 | bugfix      |
| #29952  | PCP\&DCP CUDA graph check refactor               | 2025-12-05 | enhancement |
| #30050  | Relocate PCP feature check                       | 2025-12-11 | enhancement |
| #30309  | Fix DCP accuracy with FLASH_ATTN_MLA             | 2025-12-09 | bugfix      |
| #31003  | PCP basic support for MoE model                  | 2025-12-31 | feature     |
| #33239  | Move DCP validation to ParallelConfig            | 2026-01-30 | enhancement |
| #34179  | **DCP support for GPU model runner v2**          | 2026-02-18 | feature     |
| #34786  | DCP simplification for model runner v2           | 2026-02-18 | enhancement |
| #34883  | **Add All-to-All communication backend for DCP** | 2026-03-04 | feature     |
| #35082  | Fix DCP + FA3 crash (missing num_splits)         | 2026-02-27 | bugfix      |
| #31277  | Support ViT SP parallelism in qwen2.5vl/qwen3vl  | 2025-12\~  | feature     |

### 开放中的关键 PR（29个，节选重要）

| PR 编号 | 标题                                          | 创建时间   | 优先级 |
| ------- | --------------------------------------------- | ---------- | ------ |
| #34024  | Add Helix (Context + Tensor) Parallelism      | 2026-02-06 | ⭐⭐⭐ |
| #36306  | PCP alternative implementation                | 2026-03-07 | ⭐⭐⭐ |
| #36287  | Add --tensor-parallel-size-attention          | 2026-03-06 | ⭐⭐⭐ |
| #35206  | Support Sequence Parallel for Model Runner v2 | 2026-02-24 | ⭐⭐   |
| #34975  | dcp prefill -> non-dcp decode prototype       | 2026-02-20 | ⭐⭐   |
| #28988  | Support PCP with MLA                          | 2025-11-19 | ⭐⭐   |
| #28723  | PCP for GQA flashinfer                        | 2025-11-14 | ⭐⭐   |
| #33403  | \[WIP] pcp alternative impl                   | 2026-01-30 | ⭐⭐   |
| #23545  | DCCP supported                                | 2025-08-25 | ⭐     |
