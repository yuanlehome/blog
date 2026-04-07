---
title: 深入 vLLM Pipeline Parallelism：架构、源码与性能取舍
slug: inside-vllm-pipeline-parallelism
date: '2026-04-03'
tags: ['Source Code Analysis', 'Distributed Parallel']
status: published
source: original
---

## 结论摘要

1. vLLM 的 PP 实现采用“只实例化本 stage 层、其余占位”的设计，通过 `PPMissingLayer` 充当不属于本 rank 的 layer 占位符，实际不分配权重。

2. 层分区的核心逻辑在 `get_pp_indices()` 函数中，默认采用均匀切分，remainder 层分配给**中间 stage**（既不给第一个也不给最后一个），以平衡首尾 stage 的额外 embedding/norm 负载。

3. vLLM 支持通过环境变量 `VLLM_PP_LAYER_PARTITION` 手动指定各 stage 的层数（如 `"4,6,6,4"`），实现非均匀切分。

4. PP 的 stage 间通信使用 NCCL 的 P2P 异步 `isend`/`irecv` 操作，传输内容是 `IntermediateTensors`（通常包含 `hidden_states` 和 `residual` 两个张量）。

5. V1 引擎中非末尾 PP rank 不做 sampling，而是通过 `pp_broadcast()` / `pp_receive()` 从最后一个 rank 广播采样结果回所有 rank。

6. `SupportsPP` 是一个 Python Protocol，模型必须声明 `supports_pp = True`、实现 `make_empty_intermediate_tensors()` 和在 `forward()` 中接受 `intermediate_tensors` 参数。

7. vLLM 的 `world_size = pipeline_parallel_size × tensor_parallel_size × prefill_context_parallel_size`，即 PP、TP、CP 正交组合。

8. PP 与 Elastic EP 互斥，不能同时启用。

9. 在 V1 架构中，GPU worker 进程数 = DP × PP × TP，每个 GPU 对应一个 worker 进程。

10. PP 的 `IntermediateTensors` 是一个 dataclass，内含 `tensors: dict[str, torch.Tensor]` 和可选的 `kv_connector_output`，用于 stage 间传递激活值。

11. `make_layers()` 函数构建完整层列表但只实例化本 rank 的层，前后用 `PPMissingLayer` 填充，保持全模型结构一致。

12. vLLM V1 引擎通过 `step_with_batch_queue` 和大小为 PP stage 数的 `batch_queue` 实现 GPipe 风格的 pipeline 填充；在稳态下多个 batch 同时在 pipeline 中流动，充分利用各 stage 计算能力，消除大部分填充 bubble。

13. PP 对 prefill 和 decode 的影响不对称：prefill 阶段传输的 `IntermediateTensors` 数据量大（token 数多），bubble 比例相对较小；decode 阶段每个 step 只有少量 token，stage 间通信延迟在总时间中占比更高。

14. 首段 stage（rank 0）负责 embedding 层、末段 stage（最后一个 rank）负责 RMSNorm 和 lm_head，中间 stage 只有 decoder layers（以 Llama 模型为例）。

15. 权重加载时通过 `is_pp_missing_parameter()` 检查跳过 `PPMissingLayer` 对应的参数，避免加载不属于本 rank 的权重。

16. vLLM 支持的 PP executor backend 包括 `ray`、`mp`（multiprocessing）和 `external_launcher`。

17. PP 的主要价值场景是“能部署”而非“更快”：当单张 GPU 显存不够放下整个模型时，PP 是最简单的跨卡方案；但因为引入 bubble 和通信，吞吐量通常低于等价的 TP。

18. V1 中 `model_runner` 为非首段 rank 预分配固定地址的 `intermediate_tensors` 缓冲区（用于 CUDA Graph 捕获）；执行时通过 `copy_` 将接收数据写入该预分配缓冲区，避免动态分配。

19. V1 中 PP 通信使用 `AsyncIntermediateTensors` 实现惰性同步——`irecv` 在 GPU 计算开始前就发起，只在 `model_runner` 真正访问 `.tensors` 时才等待通信完成，实现通信与计算重叠。

20. PP+TP 组合时，stage 间通信是 PP 的 P2P send/recv，stage 内通信是 TP 的 all-reduce/all-gather。两者正交但在模型执行的关键路径上串行叠加，对延迟影响显著。

21. PP+TP 联合优化时，`isend_tensor_dict` / `irecv_tensor_dict` 支持 TP 分片通信：每个 TP rank 只发送自己的 hidden_states 分片，接收端再通过本地 NVLink `all_gather` 恢复完整张量，将跨节点通信量降低 tp_size 倍。

22. 调度器（Scheduler）只有一个实例（CPU 端），通过 `use_pp` 标志感知 PP；各 stage 的 KV cache 大小通过 `collective_rpc` 取各 worker 最小值来同步，确保所有 PP stage 使用相同数量的 KV cache blocks。

23. 非末尾 PP stage 接收 sampled tokens 并非仅为了首 stage 的 embedding 需要（`last_sampled_tokens → input_ids`），中间 stage 同样必须更新 `num_computed_tokens` 和 `total_len`，以便正确计算 `positions`（RoPE 编码），缺少此更新会导致所有中间 stage 的 attention 位置编码错误。

24. `postprocess` 调用 `post_update` Triton kernel，对每个请求原子地执行 5 项 GPU 端状态更新：`last_sampled_tokens`（下步 input_ids 来源）、`all_token_ids`（追加新 token）、`total_len`、`num_computed_tokens`（支持 spec decode rejection 回退）、`output_bin_counts`（仅末尾 rank，用于惩罚计数）。

---

## 一、推理 PP 的理论模型

### 1.1 什么是 Pipeline Parallelism

Pipeline Parallelism 将模型按层（layer）维度切分为多个 **stage**，每个 stage 放置在不同的设备（GPU）上。数据（一个 batch 或 micro-batch）按顺序流过各 stage，每个 stage 完成自己负责的层的计算后，将中间激活值（intermediate tensors）传递给下一个 stage。

```text
Stage 0 (GPU 0)     Stage 1 (GPU 1)     Stage 2 (GPU 2)     Stage 3 (GPU 3)
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│ Embedding    │     │ Layers 8-15 │     │ Layers 16-23│     │ Layers 24-31│
│ Layers 0-7   │────>│             │────>│             │────>│ Norm + Head │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
     P2P send            P2P send            P2P send
```

### 1.2 PP 的计算模型

设模型总层数为 $L$，PP 并行度为 $P$，每个 stage 分到 $L/P$ 层。

- **计算量**：每个 stage 的计算量约为总量的 $1/P$（假设均匀切分且忽略首尾 stage 的额外计算）
- **理想加速比**：如果没有 bubble 和通信开销，PP 的加速比接近 $P$

但现实中，PP 存在两个核心开销：

1. **Pipeline bubble**：当一个 batch 在 pipeline 中流动时，部分 stage 处于空闲状态
2. **Stage 间通信**：中间激活值需要通过 P2P 通信传递

### 1.3 Bubble 分析

对于**推理场景**（而非训练），bubble 分析与训练有本质区别：

**训练中的 PP（1F1B 调度）**：

- 训练用 micro-batch pipeline，可以通过 1F1B（one forward, one backward）调度来重叠不同 micro-batch 的 forward 和 backward，减小 bubble
- Bubble 比例 ≈ $(P-1)/M$，其中 $M$ 是 micro-batch 数

**推理中的 PP（无 backward，batch queue 调度）**：

- 推理只有 forward pass，没有 backward
- vLLM V1 通过大小为 PP stage 数的 `batch_queue` 实现类 GPipe 的填充，稳态下 pipeline 满载（详见 4.10 节）
- 单 batch 从 stage 0 到 stage $P-1$ 串行执行，总时间 = $\sum_{i=0}^{P-1} T_{\text{compute},i} + \sum_{i=0}^{P-2} T_{\text{comm},i}$
- 在批量化稳态下，多个 batch 同时在流水线中流动，填充空闲 stage，bubble 大幅降低

### 1.4 Prefill vs Decode 的差异

| 维度               | Prefill                               | Decode                          |
| ------------------ | ------------------------------------- | ------------------------------- |
| Token 数/步        | 大（整个 prompt）                     | 小（通常 1 token/request）      |
| 计算强度           | 高（计算密集）                        | 低（显存带宽密集）              |
| 通信数据量         | 大（`batch × seq_len × hidden_size`） | 小（`batch × 1 × hidden_size`） |
| 通信/计算比        | 低（通信时间占比小）                  | 高（通信时间占比大）            |
| PP bubble 相对影响 | 小                                    | 大                              |

**关键洞察**：Decode 阶段每个 step 的计算量极小（只处理 1 个新 token），但 PP 的 stage 间通信延迟是固定的。因此 decode 时 PP 的效率损失远大于 prefill。

### 1.5 PP 解决什么、不解决什么

**PP 解决**：

- **显存不足**：当单卡放不下完整模型时，PP 将模型按层切分到多卡
- **跨机推理**：PP 的 P2P 通信只需要相邻 stage 之间通信，跨机带宽需求比 TP 低得多

**PP 不解决**：

- **计算效率**：PP 不减少总计算量，反而因 bubble 增加总计算时间
- **延迟优化**：PP 增加了 pipeline 延迟（请求必须通过所有 stage）
- **负载均衡**：首尾 stage 的额外 embedding/norm 计算导致天然不均衡

### 1.6 为什么不能直接套训练 1F1B 心智模型

1. **无 backward pass**：推理只有 forward，不存在 1F1B 的重叠机会
2. **Continuous batching**：推理系统持续接收新请求，batch 组成动态变化，不适合固定 micro-batch 数量的调度
3. **KV cache 管理**：每个 stage 都需要管理自己的 KV cache，prefill/decode 混合 batch 让调度更复杂
4. **延迟敏感**：推理对 TTFT（首 token 延迟）和 TPOT（每 token 延迟）敏感，不能像训练那样只关心吞吐

---

## 二、推理 PP 的深层思考

### 2.1 为什么很多人只懂训练 PP、不懂推理 PP

训练 PP 的核心是 1F1B 调度和 micro-batch pipeline，这是一个被充分研究和实现的问题（GPipe、PipeDream 等）。而推理 PP 面临完全不同的挑战：

- 没有 backward pass，1F1B 调度失去意义
- Continuous batching 导致 batch 大小动态变化
- 延迟要求严格，不能简单追求吞吐
- KV cache 管理与 PP 交叉，复杂度指数增长

### 2.2 推理 PP 常先解决“能部署”再谈“更快”

在实际部署中，PP 的首要动机通常是：

1. 模型太大，单卡放不下 → PP 是最简单的跨卡方案
2. 跨机节点部署，节点间带宽有限 → PP 的 P2P 通信比 TP 的 all-reduce 更适合低带宽环境
3. 异构 GPU 部署 → PP 可以将不同层分配到不同型号的 GPU

只有在“能跑”的前提下，才考虑 PP 的性能优化。

### 2.3 首尾 stage 不公平

以 Llama 为例（`vllm/model_executor/models/llama.py`）：

- **Stage 0**（首段）：`VocabParallelEmbedding` + decoder layers
- **中间 stage**：只有 decoder layers
- **最后一个 stage**（末段）：decoder layers + `RMSNorm` + `ParallelLMHead` + `LogitsProcessor`

末段的额外计算（norm + lm_head + logits processing + sampling）使其成为瓶颈。vLLM 的 `get_pp_indices()` 通过将 remainder 层分配给中间 stage 来部分缓解这个问题，但无法完全消除首尾不公平。

### 2.4 Continuous batching 和 decode 为何让 PP 更复杂

1. **动态 batch**：Continuous batching 意味着每个 step 的 batch 组成不同，prefill 和 decode 请求混合，token 数量变化大，导致 stage 间传输的数据量不稳定
2. **KV cache 分布**：每个 stage 的 KV cache 独立管理，决定能接纳多少新请求时需要考虑所有 stage 的显存状态
3. **Decode 的效率问题**：decode 时每个 stage 的计算量极小，但通信延迟固定，pipeline bubble 比例极高

### 2.5 “可并发多个 batch 波次” ≠ “吞吐一定高”

即使可以交错执行多个 batch 来填充 pipeline bubble（类似训练中的 micro-batch），在推理场景中也未必有效：

1. **显存限制**：多个 batch 同时在 pipeline 中意味着更大的 KV cache 和激活内存占用
2. **延迟约束**：批量化增加了单个请求的等待时间
3. **调度复杂性**：Continuous batching 下的 pipeline 调度远比固定 micro-batch 复杂
4. **Decode 粒度太细**：decode 的单 step 计算量太小，pipeline 充分利用率很难保证

### 2.6 PP vs TP：什么时候更合理，什么时候只是妥协

**PP 更合理的场景**：

- 跨机节点部署，节点间带宽有限（如 InfiniBand 100Gbps vs NVLink 900GB/s）
- 模型层数多但每层参数量不大
- 需要在不同型号 GPU 间分配负载

**TP 更合理的场景**：

- 单机多卡，NVLink 连接
- 需要最低延迟
- 模型每层参数量大（如大 hidden_size）

**PP 只是妥协的场景**：

- 单纯因为单卡显存不够，但有 NVLink 高速互连 → 应优先用 TP
- 追求最大吞吐但 PP 带来的 bubble 损失超过了分布式收益

### 2.7 统一理解 TTFT、TPOT、吞吐、显存利用率与 PP 的关系

| 指标                      | PP 的影响                                                                        |
| ------------------------- | -------------------------------------------------------------------------------- |
| **TTFT**（首 token 延迟） | **增加**。Prefill 必须通过所有 stage，增加 $(P-1)$ 个 stage 的通信延迟           |
| **TPOT**（每 token 延迟） | **增加**。每个 decode step 必须通过所有 stage，通信开销在 decode 中占比更大      |
| **吞吐**（tokens/s）      | **通常降低或持平**。Bubble 和通信抵消了并行化收益                                |
| **单卡显存占用**          | **降低**。模型权重按层分布，每卡只加载 $1/P$ 的层                                |
| **总 KV cache 容量**      | **取决于分配**。KV cache 在每个 stage 独立管理，但只有本 stage 的层需要 cache    |
| **总显存利用率**          | **可能降低**。因为首尾 stage 的 embedding/lm_head 不参与 KV cache 分配但占用显存 |

### 2.8 vLLM 当前的“推理系统特有” PP 设计

1. **异步 P2P 通信**：使用 `isend_tensor_dict` / `irecv_tensor_dict` 实现非阻塞通信，并通过 `AsyncIntermediateTensors` 的惰性同步机制实现通信与计算的重叠
2. **采样结果广播**：只有最后一个 rank 执行 sampling，然后将 sampled_token_ids 通过 `pp_broadcast` 广播给所有 rank
3. **KV connector 整合**：`IntermediateTensors` 包含 `kv_connector_output` 字段，用于支持 KV transfer 等高级特性
4. **与 TP 分片通信配合**：`isend/irecv` 支持 TP 分片传输，每个 TP rank 只发送自己的 hidden_states 分片，减少跨节点通信量
5. **Pipeline 填充**：V1 引擎通过 `batch_queue` 维持多个并发 batch 在 pipeline 中流动，消除填充阶段 bubble

---

## 三、vLLM 中 PP 的整体架构

### 3.1 架构总览

```text
用户配置: --pipeline-parallel-size=4 --tensor-parallel-size=2
                        │
                        ▼
              ┌─────────────────┐
              │  ParallelConfig │
              │  pp_size=4      │  world_size = pp × tp × cp = 4 × 2 × 1 = 8
              └─────────────────┘
                        │
                        ▼
              ┌─────────────────────────┐
              │ initialize_model_parallel│
              │ 创建 PP GroupCoordinator │  _PP = init_model_parallel_group(...)
              └─────────────────────────┘
                        │
             ┌──────────┴──────────┐
             ▼                     ▼
    ┌──────────────┐      ┌──────────────┐
    │  GPU Worker   │      │  GPU Worker   │
    │  rank 0-1     │      │  rank 2-3     │     ...每组 tp_size 个 worker
    │  PP stage 0   │      │  PP stage 1   │     构成一个 PP stage
    └──────────────┘      └──────────────┘
             │                     │
             ▼                     ▼
    ┌──────────────┐      ┌──────────────┐
    │ Model Runner  │      │ Model Runner  │
    │ make_layers() │      │ make_layers() │
    │ stage 0 layers│      │ stage 1 layers│
    └──────────────┘      └──────────────┘
```

### 3.2 进程与通信拓扑

根据官方文档（Architecture Overview），V1 架构中：

- GPU Worker 进程数 = `DP × PP × TP`
- 每个 GPU 一个 worker 进程
- PP stage 间通过 NCCL P2P 通信
- TP group 内通过 NCCL all-reduce/all-gather 通信

PP group 和 TP group 的关系是正交的：

```text
假设 pp_size=2, tp_size=4, 共 8 个 rank:

PP group 0: [rank 0, rank 4]  (stage 0 → stage 1)
PP group 1: [rank 1, rank 5]
PP group 2: [rank 2, rank 6]
PP group 3: [rank 3, rank 7]

TP group 0: [rank 0, rank 1, rank 2, rank 3]  (stage 0 内)
TP group 1: [rank 4, rank 5, rank 6, rank 7]  (stage 1 内)
```

**全局 rank 布局的底层实现**：在 `initialize_model_parallel()` 中，所有 rank 按以下方式组织：

```python
# all_ranks layout: ExternalDP × DP × PP × PCP × TP
all_ranks = torch.arange(world_size).reshape(-1, dp_size, pp_size, pcp_size, tp_size)
```

以 TP=2, PP=4, world_size=8 为例：

```text
all_ranks reshape 为 (1, 1, 4, 1, 2)：
  Stage 0: [0, 1]    ← TP group 0
  Stage 1: [2, 3]    ← TP group 1
  Stage 2: [4, 5]    ← TP group 2
  Stage 3: [6, 7]    ← TP group 3

PP group（TP rank 0）: [0, 2, 4, 6]
PP group（TP rank 1）: [1, 3, 5, 7]
```

PP group 通过对 `all_ranks` 转置（交换 PP 维和 TP 维）、重塑、unbind 生成：

```python
group_ranks = (
    all_ranks.transpose(2, 4)   # 交换 PP 维和 TP 维
    .reshape(-1, pipeline_model_parallel_size)
    .unbind(0)
)
_PP = init_model_parallel_group(group_ranks, ..., group_name="pp")
```

---

## 四、vLLM PP 源码深度解析

### 4.1 配置入口与参数传递链路

#### 4.1.1 ParallelConfig

- **文件**：`vllm/config/parallel.py`
- **定义**：`pipeline_parallel_size: int = 1`
- **world_size 计算**：
  ```python
  self.world_size = (
      self.pipeline_parallel_size
      * self.tensor_parallel_size
      * self.prefill_context_parallel_size
  )
  ```
- **约束**：与 Elastic EP 互斥
- **executor backend**：`distributed_executor_backend` 可选 `"ray"`, `"mp"`, `"external_launcher"` 或自定义 Executor 类

#### 4.1.2 Engine 层面验证

- **文件**：`vllm/engine/arg_utils.py`
- 当 `pipeline_parallel_size > 1` 时，检查 executor backend 是否支持 PP
- 有效的后端：`ray`, `mp`, `external_launcher`，或自定义 Executor 声明了 `supports_pp = True`

### 4.2 Worker / Rank / Group / Stage 的关系

#### 4.2.1 GroupCoordinator

- **文件**：`vllm/distributed/parallel_state.py`
- **关键属性**：
  - `rank`: 全局 rank
  - `ranks`: group 内所有全局 rank 的列表
  - `world_size`: group 大小（= PP size）
  - `rank_in_group`: 在 PP group 内的 rank（即 stage 编号）
  - `is_first_rank`: `rank_in_group == 0`
  - `is_last_rank`: `rank_in_group == world_size - 1`
  - `next_rank`: 下一个 PP stage 的全局 rank
  - `prev_rank`: 上一个 PP stage 的全局 rank
  - `device_group`: 设备通信组（NCCL）
  - `cpu_group`: CPU 通信组（Gloo）

#### 4.2.2 PP Group 初始化

- **文件**：`vllm/distributed/parallel_state.py`
- 在 `initialize_model_parallel()` 函数中创建
- PP group 通过对 `all_ranks` 张量的转置和重塑生成（与 3.2 节描述一致，详见该节代码）

#### 4.2.3 PP Group 获取

- **文件**：`vllm/distributed/parallel_state.py`
- `get_pp_group() -> GroupCoordinator` 返回全局 `_PP` 单例

#### 4.2.4 关系映射

```text
Global Rank → PP Stage (rank_in_group)
Global Rank → TP Position (rank in TP group)
Global Rank → Local GPU (local_rank)

PP Stage = get_pp_group().rank_in_group
TP Position = get_tp_group().rank_in_group
```

### 4.3 Layer Partition 的核心逻辑

#### 4.3.1 get_pp_indices()

- **文件**：`vllm/distributed/utils.py`
- **签名**：`get_pp_indices(num_hidden_layers, pp_rank, pp_size) -> tuple[int, int]`
- **返回**：`(start_layer, end_layer)` 左闭右开区间

**默认均匀切分算法**：

```python
layers_per_partition = num_hidden_layers // pp_size
partitions = [layers_per_partition for _ in range(pp_size)]

if remaining_layers := num_hidden_layers % pp_size:
    for i in range(2, remaining_layers + 2):
        partitions[-i] += 1
```

**Remainder 分配策略**：

- 额外的层从**倒数第二个** partition 开始往前分配
- **排除最后一个** partition（因为它通常包含 norm 层，已经有额外计算）
- 当 `pp_size > 2` 时，也**排除第一个** partition（因为它包含 embedding 层）
- 目标：平衡各 stage 的计算负载

**分配示例**：

| 总层数 | PP Size | 分配结果     | 说明                                   |
| ------ | ------- | ------------ | -------------------------------------- |
| 32     | 4       | [8, 8, 8, 8] | 均匀，无 remainder                     |
| 22     | 4       | [5, 6, 6, 5] | remainder=2，分给中间                  |
| 5      | 3       | [2, 2, 1]    | remainder=2，不够中间分                |
| 4      | 3       | [1, 2, 1]    | remainder=1，给中间                    |
| 3      | 2       | [2, 1]       | remainder=1，给倒数第二个（即 rank 0） |

**手动覆盖**：设置 `VLLM_PP_LAYER_PARTITION="5,6,5,6"` 环境变量可以完全自定义分区。

**测试用例**：`tests/distributed/test_pipeline_partition.py`

#### 4.3.2 make_layers()

- **文件**：`vllm/model_executor/models/utils.py`
- **签名**：`make_layers(num_hidden_layers, layer_fn, prefix) -> tuple[int, int, ModuleList]`

**核心逻辑**：

```python
start_layer, end_layer = get_pp_indices(
    num_hidden_layers, get_pp_group().rank_in_group, get_pp_group().world_size
)

modules = torch.nn.ModuleList(
    [PPMissingLayer() for _ in range(start_layer)]           # 前面的占位
    + get_offloader().wrap_modules(
        layer_fn(prefix=f"{prefix}.{idx}")
        for idx in range(start_layer, end_layer)
    )                                                         # 本 rank 的真实层
    + [PPMissingLayer() for _ in range(end_layer, num_hidden_layers)]  # 后面的占位
)
```

**设计要点**：

- `ModuleList` 保持全模型结构（长度 = num_hidden_layers），但只有 `start_layer:end_layer` 是真实 layer
- 前后用 `PPMissingLayer` 填充，不分配任何参数/权重
- `get_offloader().wrap_modules()` 支持权重 offload 到 CPU

### 4.4 SupportsPP 接口体系

#### 4.4.1 SupportsPP Protocol

- **文件**：`vllm/model_executor/models/interfaces.py`
- **要求**：
  1. `supports_pp: ClassVar[Literal[True]] = True`
  2. `make_empty_intermediate_tensors(batch_size, dtype, device) -> IntermediateTensors`
  3. `forward(input_ids, positions, *, intermediate_tensors) -> IntermediateTensors | None`

#### 4.4.2 supports_pp() 验证函数

- **文件**：`vllm/model_executor/models/interfaces.py`
- 同时检查属性和签名两个维度
- 如果设置了 `supports_pp=True` 但 forward 不接受 `intermediate_tensors`，会发出警告

#### 4.4.3 IntermediateTensors

- **文件**：`vllm/sequence.py`
- **字段**：
  - `tensors: dict[str, torch.Tensor]` - 键值对形式的中间张量
  - `kv_connector_output: KVConnectorOutput | None` - 可选的 KV connector 输出
- **支持的操作**：`__getitem__`（支持 str key 或 slice）、`__setitem__`、`items()`、`empty_like()` 等
- **典型内容**（以 Llama 为例）：`{"hidden_states": tensor, "residual": tensor}`

#### 4.4.4 PPMissingLayer

- **文件**：`vllm/model_executor/models/utils.py`
- 继承 `torch.nn.Identity`，不分配任何参数
- `forward()` 返回第一个 arg 或第一个 kwarg value（作为 pass-through）

#### 4.4.5 make_empty_intermediate_tensors_factory()

- **文件**：`vllm/model_executor/models/utils.py`
- 工厂函数，创建生成空 `IntermediateTensors` 的闭包
- 用于非首段 rank 的 profiling（CUDA Graph 捕获等需要知道张量形状）

#### 4.4.6 Transformers 后端的 PP 处理

对于基于 HuggingFace Transformers 格式的模型，`vllm/model_executor/models/transformers/base.py` 提供了通用的 PP 支持：

**`pipeline_parallel` 方法**：

- 读取 HuggingFace 的 `_pp_plan` 属性确定层划分
- 非本 stage 的层替换为 `PPMissingLayer()`
- 首 stage 保留 embedding；末 stage 保留 norm/lm_head

**`forward` 方法**：

```python
def forward(self, input_ids, positions, intermediate_tensors=None, ...):
    if not self.pp_group.is_first_rank:
        # 非首 stage：不做 embedding，直接用上游传来的 hidden_states
        input_ids = None
        inputs_embeds = intermediate_tensors["hidden_states"]

    outputs = self.model(...)
    hidden_states = outputs[0][0, ...]

    if not self.pp_group.is_last_rank:
        # 非末 stage：输出 IntermediateTensors，不计算 logits
        return IntermediateTensors({"hidden_states": hidden_states})
    return hidden_states
```

### 4.5 案例分析：Llama 模型的 PP 实现

以 `LlamaForCausalLM` 为例，追踪完整的 PP 调用链。

**文件**：`vllm/model_executor/models/llama.py`

#### 4.5.1 LlamaForCausalLM（外层包装）

- **类声明**：`class LlamaForCausalLM(nn.Module, SupportsLoRA, SupportsPP, SupportsEagle, SupportsEagle3)`
- **构造函数**：

  ```python
  self.model = self._init_model(...)  # 创建 LlamaModel

  # lm_head 只在最后一个 PP rank 上创建
  if get_pp_group().is_last_rank:
      self.lm_head = ParallelLMHead(...)
      self.logits_processor = LogitsProcessor(...)
  else:
      self.lm_head = PPMissingLayer()

  # 委托给内部模型
  self.make_empty_intermediate_tensors = self.model.make_empty_intermediate_tensors
  ```

#### 4.5.2 LlamaModel（内层模型，核心 PP 逻辑）

- **构造函数**：

  **Embedding 层**：

  ```python
  if get_pp_group().is_first_rank or (
      config.tie_word_embeddings and get_pp_group().is_last_rank
  ):
      self.embed_tokens = VocabParallelEmbedding(...)
  else:
      self.embed_tokens = PPMissingLayer()
  ```

  - 首段 rank 必须有 embedding
  - 如果 `tie_word_embeddings=True`，末段 rank 也需要（与 lm_head 共享权重）

  **Decoder Layers**：

  ```python
  self.start_layer, self.end_layer, self.layers = make_layers(
      config.num_hidden_layers,
      lambda prefix: layer_type(vllm_config=vllm_config, prefix=prefix),
      prefix=f"{prefix}.layers",
  )
  ```

  **Norm 层**：

  ```python
  if get_pp_group().is_last_rank:
      self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
  else:
      self.norm = PPMissingLayer()
  ```

  **空中间张量工厂**：

  ```python
  self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
      ["hidden_states", "residual"], config.hidden_size
  )
  ```

#### 4.5.3 Forward 路径分析

**LlamaModel.forward()**：

**首段 (Stage 0)**：

```python
if get_pp_group().is_first_rank:
    if inputs_embeds is not None:
        hidden_states = inputs_embeds
    else:
        hidden_states = self.embed_input_ids(input_ids)  # Embedding lookup
    residual = None
```

- 输入：`input_ids`（token IDs）
- 输出：`hidden_states` 张量

**中段 (Stage 1..P-2)**：

```python
else:
    assert intermediate_tensors is not None
    hidden_states = intermediate_tensors["hidden_states"]
    residual = intermediate_tensors["residual"]
```

- 输入：从上一个 stage 接收的 `IntermediateTensors`
- 解包得到 `hidden_states` 和 `residual`

**所有 stage 共同的层计算**：

```python
for idx, layer in enumerate(
    islice(self.layers, self.start_layer, self.end_layer)
):
    hidden_states, residual = layer(
        positions, hidden_states, residual, **extra_layer_kwargs
    )
```

- 只迭代 `start_layer:end_layer` 范围（避免执行 PPMissingLayer）

**非末段返回**：

```python
if not get_pp_group().is_last_rank:
    return IntermediateTensors(
        {"hidden_states": hidden_states, "residual": residual}
    )
```

**末段处理**：

```python
hidden_states, _ = self.norm(hidden_states, residual)
return hidden_states
```

- 应用 RMSNorm
- 返回最终 hidden_states（给 lm_head 和 LogitsProcessor 使用）

#### 4.5.4 各 Stage 的组件分布

```text
Stage 0 (首段):
├── VocabParallelEmbedding ✓
├── DecoderLayers[0:start_1] ✓
├── DecoderLayers[start_1:] → PPMissingLayer
├── RMSNorm → PPMissingLayer
└── lm_head → PPMissingLayer

Stage k (中段):
├── VocabParallelEmbedding → PPMissingLayer
├── DecoderLayers[:start_k] → PPMissingLayer
├── DecoderLayers[start_k:end_k] ✓
├── DecoderLayers[end_k:] → PPMissingLayer
├── RMSNorm → PPMissingLayer
└── lm_head → PPMissingLayer

Stage P-1 (末段):
├── VocabParallelEmbedding → PPMissingLayer (除非 tie_word_embeddings)
├── DecoderLayers[:start_P-1] → PPMissingLayer
├── DecoderLayers[start_P-1:L] ✓
├── RMSNorm ✓
├── ParallelLMHead ✓
└── LogitsProcessor ✓
```

### 4.6 运行时下批次的流动

#### 4.6.1 V1 GPU Worker 的执行流程

- **文件**：`vllm/v1/worker/gpu_worker.py`

**完整流程**：

```text
Step 1: 等待上一 iter 的异步 PP send 完成（确保上条 pipeline 数据已发出）
         ↓
Step 2: 非首段 rank：发起异步 irecv 接收上一 stage 的中间张量（非阻塞，后台进行）
         ↓
Step 3: 执行模型前向（GPU 计算）
         ↓
Step 4a（末段 rank）：返回 ModelRunnerOutput（正常输出）
Step 4b（非末段 rank）：异步发送中间张量给下一 stage，return None
```

```python
def execute_model(self, scheduler_output):
    # Step 1: 等上一 iter 发送完成
    for handle in self._pp_send_work:
        handle.wait()
    self._pp_send_work = []

    # Step 2: 非首段 rank 异步接收 activation
    if not get_pp_group().is_first_rank:
        tensor_dict, comm_handles, comm_postprocess = get_pp_group().irecv_tensor_dict(...)
        intermediate_tensors = AsyncIntermediateTensors(
            tensor_dict, comm_handles=comm_handles, comm_postprocess=comm_postprocess
        )

    # Step 3: model forward
    output = self.model_runner.execute_model(scheduler_output, intermediate_tensors)

    # Step 4b: 非末段 rank 异步发送 activation
    if isinstance(output, IntermediateTensors):
        self._pp_send_work = get_pp_group().isend_tensor_dict(output.tensors, ...)
        return None
```

**`isend_tensor_dict` 实现要点**（`vllm/distributed/parallel_state.py`）：

```python
def isend_tensor_dict(self, tensor_dict, dst=None, ...):
    # Step 1: 同步发送 metadata（shape/dtype，CPU 小对象）
    self.send_object(metadata_list, dst=dst)

    # Step 2: 异步发送 GPU tensor（activation 数据）
    handles = []
    for key, tensor in zip(tensor_keys, tensor_list):
        handle = torch.distributed.isend(tensor, dst=self.ranks[dst], group=comm_group)
        tensor.record_stream(torch.cuda.current_stream(tensor.device))
        handles.append(handle)

    return handles  # 调用者稍后 wait
```

**`irecv_tensor_dict` 实现要点**（`vllm/distributed/parallel_state.py`）：

```python
def irecv_tensor_dict(self, src=None, ...):
    # Step 1: 接收 metadata，获取 shape/dtype
    recv_metadata_list = self.recv_object(src=src)

    # Step 2: 预分配 buffer，异步接收 GPU tensor
    for key, value in recv_metadata_list:
        if isinstance(value, TensorMetadata):
            full_tensor = torch.empty(value.size, ...)
            handle = torch.distributed.irecv(full_tensor, ...)
            handles.append(handle)

    return tensor_dict, handles, postprocess
```

#### 4.6.2 V1 Model Runner 的 PP 处理

- **文件**：`vllm/v1/worker/gpu/model_runner.py`

**初始化**：

```python
self.use_pp = self.parallel_config.pipeline_parallel_size > 1
self.is_first_pp_rank = get_pp_group().is_first_rank
self.is_last_pp_rank = get_pp_group().is_last_rank
self.intermediate_tensors: IntermediateTensors | None = None  # 持久化缓冲区
```

**Profiling 阶段创建空中间张量**：

```python
if not self.is_first_pp_rank:
    self.intermediate_tensors = self.model.make_empty_intermediate_tensors(
        batch_size=self.max_num_tokens,
        dtype=self.model_config.dtype,
        device=self.device,
    )
```

**execute_model 中的 PP 处理**：

```python
if not self.is_first_pp_rank:
    model_inputs["input_ids"] = None
    model_inputs["inputs_embeds"] = None

    assert intermediate_tensors is not None
    n = input_batch.num_tokens_after_padding
    model_inputs["intermediate_tensors"] = IntermediateTensors({
        k: v[:n].copy_(intermediate_tensors.tensors[k][:n])
        for k, v in self.intermediate_tensors.tensors.items()
    })
```

- 使用持久化缓冲区 `self.intermediate_tensors` 避免每次分配
- `copy_` 将接收的数据拷贝到预分配的缓冲区中（固定地址，CUDA Graph 需要）

**输出处理**：

```python
if self.is_last_pp_rank:
    hidden_states = model_output  # 最终 hidden states
    output_intermediate_tensors = None
else:
    output_intermediate_tensors = model_output  # IntermediateTensors
```

#### 4.6.3 采样结果广播与非末尾 Stage 的状态维护

- **文件**：`vllm/v1/worker/gpu/pp_utils.py`

**末段 rank 广播**（`pp_broadcast`）：

```python
def pp_broadcast(sampled_token_ids, num_sampled, num_rejected):
    pp = get_pp_group()
    torch.distributed.broadcast(
        sampled_token_ids.contiguous(), src=pp.last_rank, group=pp.device_group
    )
    combined = torch.stack((num_sampled, num_rejected), dim=0)
    torch.distributed.broadcast(combined, src=pp.last_rank, group=pp.device_group)
```

**非末段 rank 接收**（`pp_receive`）：

```python
def pp_receive(num_reqs, max_sample_len=1):
    pp = get_pp_group()
    sampled_tokens = torch.empty(num_reqs, max_sample_len, dtype=torch.int64, device=pp.device)
    torch.distributed.broadcast(sampled_tokens, src=pp.last_rank, group=pp.device_group)
    combined = torch.empty(2, num_reqs, dtype=torch.int32, device=pp.device)
    torch.distributed.broadcast(combined, src=pp.last_rank, group=pp.device_group)
    return sampled_tokens, *combined.unbind(dim=0)
```

**在 `sample_tokens` 中的使用**（`vllm/v1/worker/gpu/model_runner.py`）：

```python
if not self.is_last_pp_rank:
    sampled, num_sampled, num_rejected = pp_receive(
        input_batch.num_reqs, max_sample_len=self.num_speculative_steps + 1
    )
    self.postprocess(input_batch, sampled, num_sampled, num_rejected)
    return None

# 末段 rank：执行 sampling
sampler_output, num_sampled, num_rejected = self.sample(
    hidden_states, input_batch, grammar_output
)
if self.use_pp:
    pp_broadcast(sampler_output.sampled_token_ids, num_sampled, num_rejected)
```

#### 4.6.4 为什么所有 Stage 都必须接收采样结果

首 stage 和中间 stage 接收 sampled tokens 的原因**本质不同**，不能混为一谈：

**首 stage 的需求**：负责 embedding，下一步的 `input_ids` 必须包含上一步采样出的 token。`last_sampled_tokens` 直接被 `combine_sampled_and_draft_tokens` 写入 `input_ids`，送入 embedding layer。缺失则自回归链路断裂。

**中间 stage 的需求（不同！）**：中间 stage 在 `execute_model` 中会立即用 `input_ids = None` 覆盖 token id（故 embedding 无意义）。中间 stage 接收 sampled tokens 的真实目的是：

**维护正确的 `num_computed_tokens` 和 `total_len`，以便下一步计算出正确的 `positions` 和 `seq_lens`，供每个 attention layer 的 RoPE 使用。**

每个中间 stage 的每一个 attention layer 都需要 `positions` 做旋转位置编码：

```python
# llama.py（所有 PP rank 的每个 attention layer 都执行）
def forward(self, positions, hidden_states):
    q, k = self.rotary_emb(positions, q, k)   # ← positions 是 RoPE 的核心输入
    attn_output = self.attn(q, k, v, ...)
```

`positions` 由 `num_computed_tokens` 计算（`input_batch.py`，`_prepare_pos_seq_lens_kernel`）：

```python
# Triton kernel
num_computed = num_computed_tokens[req_state_idx]
seq_len      = num_computed + query_len          # seq_lens
positions[i] = num_computed + i                  # positions = num_computed + 偏移
```

所有 stage 必须在本地同步更新的状态字段：

| 状态字段                       | 首 stage 用途                       | 中间 stage 用途                            |
| ------------------------------ | ----------------------------------- | ------------------------------------------ |
| `last_sampled_tokens[req_idx]` | **关键**：下一步 `input_ids` 的来源 | 更新但被 `input_ids=None` 覆盖，实际无用   |
| `num_computed_tokens[req_idx]` | 计算 `positions`（RoPE）            | **关键**：同样用于计算 `positions`（RoPE） |
| `total_len[req_idx]`           | `seq_lens` 计算                     | 同左                                       |
| `all_token_ids[req_idx, :]`    | chunked prefill 时读取 token        | 同左（跟踪 prefill 完成进度）              |

#### 4.6.5 `postprocess` 与 `post_update` Triton Kernel

**文件**：`vllm/v1/worker/gpu/model_runner.py`（`postprocess`），`vllm/v1/worker/gpu/input_batch.py`（`_post_update_kernel`）

```python
def postprocess(self, input_batch, sampled_tokens, num_sampled, num_rejected):
    output_bin_counts = (
        self.sampler.penalties_state.output_bin_counts
        if self.is_last_pp_rank else None   # 非末尾 rank 不更新惩罚计数
    )
    post_update(
        input_batch.idx_mapping,                   # batch_idx -> req_state_idx
        self.req_states.num_computed_tokens.gpu,
        self.req_states.last_sampled_tokens,        # ← 更新后下步作为 input_ids
        output_bin_counts,
        sampled_tokens,
        num_sampled,
        num_rejected,
        input_batch.query_start_loc,
        self.req_states.all_token_ids.gpu,          # ← 追加新 token
        self.req_states.total_len.gpu,
    )
    # CPU 端更新 num_computed_prefill_tokens
    computed_prefill = self.req_states.num_computed_prefill_tokens
    computed_prefill[idx_mapping_np] += input_batch.num_scheduled_tokens
    np.minimum(computed_prefill, self.req_states.prefill_len.np, out=computed_prefill)
```

Triton kernel 对每个请求执行的 **5 项状态更新**：

```text
① last_sampled_tokens[req_idx] = sampled_tokens[req_id, num_sampled-1]
   → 下一步 prepare_inputs 时被 combine_sampled_and_draft_tokens 写入 input_ids

② all_token_ids[req_idx, total_len : total_len+num_sampled] = sampled_tokens[req_id, :]
   → 维护完整 token 历史，chunked prefill 时从此读取

③ total_len[req_idx] += num_sampled

④ num_computed_tokens[req_idx] += query_len - num_rejected
   → spec decode rejection 时 num_rejected > 0，回退 positions

⑤ output_bin_counts（仅末尾 rank）：frequency/repetition penalty 计数
```

#### 4.6.6 下一步 input_ids 的构造路径

**文件**：`vllm/v1/worker/gpu/model_runner.py`（`prepare_inputs`），`vllm/v1/worker/gpu/input_batch.py`（`combine_sampled_and_draft_tokens`）

```python
# model_runner.py prepare_inputs（所有 PP stage 都执行此步）
logits_indices = combine_sampled_and_draft_tokens(
    self.input_buffers.input_ids,
    idx_mapping,
    self.req_states.last_sampled_tokens,   # ← postprocess 写入的值
    query_start_loc,
    seq_lens,
    ...
)

# combine_sampled_and_draft_tokens Triton kernel（input_batch.py）
# 对每个 decode 请求：
last_token_id = last_sampled_tokens[req_state_idx]
input_ids[query_end - num_logits] = last_token_id  # decode token = 上一步采样结果
```

**若非末尾 stage 不更新 `last_sampled_tokens`，则首 stage 送入 embedding 的 `input_ids` 将是错误的 token，整个自回归链路断裂。**

#### 4.6.7 调度器与 GPU Worker 的分工

调度器和 GPU worker 是**并行的两条更新路径**：

| 主体                              | 获取 sampled tokens 的方式                        | 用途                                                              |
| --------------------------------- | ------------------------------------------------- | ----------------------------------------------------------------- |
| **非末尾 PP stage**（GPU worker） | `pp_receive` broadcast（GPU tensor）              | 更新本地 `req_states`，构造下一步 `input_ids`                     |
| **末尾 PP stage**（GPU worker）   | 本地采样结果                                      | 同上 + D2H copy 到 `ModelRunnerOutput`                            |
| **调度器**（CPU，EngineCore）     | `ModelRunnerOutput.sampled_token_ids`（CPU list） | `append_output_token_ids`、`check_stop`、grammar 推进、返回给前端 |

worker 在 GPU 上维护 token 状态用于下一步推理，scheduler 在 CPU 上维护 token 状态用于停止判断和结果输出。

#### 4.6.8 完整调用链（PP=2 示例）

```text
[EngineCore.step]
  │
  ├─ executor.execute_model(scheduler_output)
  │   ├─ Stage 0: prepare_inputs (last_sampled_tokens → input_ids)
  │   │           → model forward → IntermediateTensors → isend
  │   └─ Stage 1: irecv → model forward → hidden_states（存入 execute_model_state）
  │
  ├─ executor.sample_tokens(grammar_output)
  │   ├─ Stage 1 (末尾):
  │   │   sample(hidden_states) → sampled_token_ids
  │   │   pp_broadcast(sampled_token_ids, num_sampled, num_rejected)
  │   │   postprocess → post_update（更新 last_sampled_tokens 等）
  │   │   return ModelRunnerOutput（via D2H copy）
  │   │
  │   └─ Stage 0 (首):
  │       pp_receive() ← 接收 Stage 1 广播
  │       postprocess → post_update:
  │         ① last_sampled_tokens[req] = sampled_token   ← 下步 input_ids 来源
  │         ② all_token_ids[req] 追加新 token
  │         ③ total_len[req] += num_sampled
  │         ④ num_computed_tokens[req] += query_len - num_rejected
  │       return None
  │
  └─ scheduler.update_from_output(scheduler_output, model_runner_output)
      ← 仅使用末尾 stage 的 ModelRunnerOutput（CPU 端）
      → request.append_output_token_ids(token_id)
      → check_stop(request)
      → grammar.accept_tokens(req_id, new_token_ids)
      → 返回 EngineCoreOutputs 给前端
```

### 4.7 权重加载的 PP 感知

#### 4.7.1 跳过 PPMissingLayer 参数

- **文件**：`vllm/model_executor/models/utils.py`

```python
def is_pp_missing_parameter(name: str, model: torch.nn.Module) -> bool:
    if isinstance(model, (StageMissingLayer, PPMissingLayer)):
        return True
    return any(
        name.startswith(missing_layer_name)
        for missing_layer_name in get_pp_missing_layer_names(model)
    )
```

- `get_pp_missing_layer_names()` 遍历模型，收集所有 `PPMissingLayer` 实例的名称前缀
- 结果被缓存在 `_model_to_pp_missing_layer_names` 字典中
- 权重加载器使用此函数跳过不属于本 rank 的参数

### 4.8 计算与通信重叠：AsyncIntermediateTensors

V1 引擎通过 `AsyncIntermediateTensors` 实现惰性同步，将 PP 通信与 CPU 调度准备工作重叠：

**文件**：`vllm/v1/worker/gpu_worker.py`

```python
class AsyncIntermediateTensors(IntermediateTensors):
    def __init__(self, tensors, comm_handles=None, comm_postprocess=None):
        self._comm_handles = comm_handles
        self._comm_postprocess = comm_postprocess
        self._comm_waited = False

    def __getattribute__(self, name):
        # 仅在真正访问 .tensors 时才等待通信完成（惰性同步）
        if name == "tensors" and not self._comm_waited:
            self.wait_for_comm()
        return object.__getattribute__(self, name)
```

**重叠时序**：

```text
时间轴 →

Stage N-1:  [compute_k]  ──────── [isend activation_k]
Stage N  :  [irecv activation_k]  ...  [wait recv]  [compute_k]  [isend ...]
Stage N+1:                              [irecv activation_k]      ...
```

`irecv` 在 compute 开始前就发起，GPU 上的 NCCL 传输可以与调度准备、输入预处理等 CPU 工作并发进行。只有当 `model_runner` 实际访问 `.tensors` 时才会真正等待通信完成（`wait_for_comm`），从而充分重叠通信与计算。

### 4.9 流水线气泡消除：批次队列机制

V1 引擎通过 GPipe 风格的 `batch_queue` 机制消除填充阶段的流水线气泡：

**文件**：`vllm/v1/engine/core.py`

```python
# batch_queue_size = PP stage 数
self.batch_queue_size = self.model_executor.max_concurrent_batches

if self.batch_queue_size > 1:
    self.batch_queue = deque(maxlen=self.batch_queue_size)
    self.step_fn = self.step_with_batch_queue  # PP 模式走带队列的 step
```

`max_concurrent_batches` 的计算（`vllm/v1/executor/multiproc_executor.py`）：

```python
@cached_property
def max_concurrent_batches(self) -> int:
    pp_size = self.parallel_config.pipeline_parallel_size
    # PP 时队列大小 = stage 数，使得 pipeline 始终满载
    return 2 if pp_size <= 1 and self.scheduler_config.async_scheduling else pp_size
```

**`step_with_batch_queue` 流程**：

```python
def step_with_batch_queue(self):
    # 1. 调度新 batch，非阻塞提交给 executor
    new_scheduler_output = scheduler.schedule()
    future = executor.execute_model_async(new_scheduler_output, non_block=True)
    batch_queue.append((future, new_scheduler_output))

    # 2. 若队列未满，直接返回（继续填充 pipeline）
    if len(batch_queue) < batch_queue_size:
        return []

    # 3. 队列满时，pop 最老的 future 并等待完成
    oldest_future, oldest_output = batch_queue.popleft()
    model_output = oldest_future.get()  # 阻塞等待最老的 batch

    # 4. 更新调度器状态
    scheduler.update_from_output(oldest_output, model_output)
    return outputs
```

**Pipeline 填充示意（PP=4）**：

```text
Step 1: [Batch A → Stage 0]
Step 2: [Batch A → Stage 1] [Batch B → Stage 0]
Step 3: [Batch A → Stage 2] [Batch B → Stage 1] [Batch C → Stage 0]
Step 4: [Batch A → Stage 3] [Batch B → Stage 2] [Batch C → Stage 1] [Batch D → Stage 0]
         ↑ 此时 Batch A 完成，输出结果，同时 pipeline 已满载（稳态）
```

**Executor 的输出 rank**（仅从 PP 最末 stage 且 TP rank=0 的 worker 收集输出）：

```python
def _get_output_rank(self) -> int:
    return (
        self.world_size
        - self.parallel_config.tensor_parallel_size
        * self.parallel_config.prefill_context_parallel_size
    )
```

---

## 五、vLLM PP 的性能模型与工程取舍

### 5.1 Stage 计算负载分析

**均匀切分下的不均衡**：

假设 Llama-70B (80 层)，PP=4：

- Stage 0: Embedding + Layers 0-19 (20 layers)
- Stage 1: Layers 20-39 (20 layers)
- Stage 2: Layers 40-59 (20 layers)
- Stage 3: Layers 60-79 + Norm + LM Head (20 layers + extra)

末段 Stage 3 的额外负载：

- RMSNorm：相对较小
- ParallelLMHead：矩阵乘法 `hidden_size × vocab_size`，对于大词表（如 128K）比较显著
- LogitsProcessor：softmax 和 top-k/top-p 采样
- Sampling：token 生成

末段的额外延迟可能占整个 stage 计算的 5-15%（取决于 vocab_size 和 batch_size），这直接映射为 pipeline 的 bubble。

### 5.2 激活传输开销

**传输内容**：`IntermediateTensors` 包含 `hidden_states` 和 `residual`

传输数据量 = `2 × batch_size × seq_len × hidden_size × sizeof(dtype)`

| 场景         | Batch | Tokens | Hidden | dtype | 数据量 |
| ------------ | ----- | ------ | ------ | ----- | ------ |
| Prefill (短) | 32    | 512    | 8192   | bf16  | 512 MB |
| Prefill (长) | 4     | 4096   | 8192   | bf16  | 256 MB |
| Decode       | 256   | 1      | 8192   | bf16  | 8 MB   |

**关键观察**：

- Prefill 传输量大但计算量也大，通信/计算比较低
- Decode 传输量小但计算量更小，通信延迟在总时间中占比高
- NVLink (900 GB/s) 下 8MB 传输约 9μs，但 NCCL P2P 的实际延迟（包括 kernel launch 等）可能达到 50-100μs
- 跨节点 InfiniBand (100 Gbps = 12.5 GB/s) 下 8MB 需要约 640μs

### 5.3 Pipeline Bubble 分析

**简化模型**：假设每个 stage 的计算时间为 $T_c$，通信时间为 $T_{comm}$

单个 batch 的端到端延迟：
$$T_{total} = P \times T_c + (P-1) \times T_{comm}$$

如果不用 PP（所有层在一张卡上）：
$$T_{single} = P \times T_c \quad \text{（计算量不变，只是在一张卡上串行）}$$

所以 PP 的延迟开销 = $(P-1) \times T_{comm}$

**Bubble 比例**（在流水线未充分填充时）：
$$\text{Bubble ratio} = \frac{(P-1) \times T_c}{P \times T_c + (P-1) \times T_{comm}} \approx \frac{P-1}{P}$$

对于 PP=4，理论 bubble 比例 ≈ 75%（如果只有一个 batch 在流动）。

### 5.4 PP 对 KV Cache 的影响

**KV cache 分布**：

- 每个 stage 只缓存本 stage 层的 KV cache
- Stage 0 的 KV cache = 层 0 到 层 $L/P-1$ 的 cache
- 总 KV cache 大小不变，但分布到各卡

**显存分配差异**：

- 首段：Embedding 权重 + KV cache + 少量激活
- 中段：Layer 权重 + KV cache
- 末段：Layer 权重 + Norm 权重 + LM Head 权重 + KV cache + Logits 缓存

末段因为额外的 LM Head（`hidden_size × vocab_size` 参数）可能显存压力更大，减少了可用于 KV cache 的空间，影响最大并发请求数。

**KV cache 独立管理的实现**（`vllm/v1/worker/gpu/attn_utils.py`）：

```python
def get_kv_cache_spec(vllm_config: VllmConfig) -> dict[str, KVCacheSpec]:
    # 只获取本 stage 实例化的 attention layers（PPMissingLayer 被过滤掉）
    attn_layers = get_layers_from_vllm_config(vllm_config, AttentionLayerBase)
    for layer_name, attn_module in attn_layers.items():
        if spec := attn_module.get_kv_cache_spec(vllm_config):
            kv_cache_spec[layer_name] = spec
    return kv_cache_spec
```

`PPMissingLayer` 不会出现在 `static_forward_context`，因此不会被分配 KV cache。各 stage 层数相同（尽量均匀划分），KV cache 大小基本一致。`determine_available_memory` 通过 `collective_rpc` 在所有 workers 执行，executor 取各 stage 最小值，确保所有 PP stage 使用相同数量的 KV cache blocks。

### 5.5 PP+TP 的通信协同优化

PP+TP 组合时，两种通信模式正交叠加：

- **TP 通信**：每层的 all-reduce/all-gather，在 stage 内
- **PP 通信**：stage 间的 P2P send/recv

两者在关键路径上**串行叠加**：

```text
Stage k 的一步执行:
[TP all-reduce] → [Layer compute] → [TP all-reduce] → ... → [PP send] → [等待 PP recv]
```

在 PP+TP 场景下：

- TP 的 all-reduce 延迟 × 每层次数 = TP 通信开销
- PP 的 P2P 延迟 × 每步次数 = PP 通信开销
- 总通信开销 = TP 开销 + PP 开销（不可重叠）

**TP 分片优化**：`isend_tensor_dict` / `irecv_tensor_dict` 内置了一项重要优化——当 PP 与 TP 组合时，每个 TP rank 只发送 `hidden_states` 的 1/tp_size 分片，接收端通过本地 NVLink `all_gather` 恢复完整张量：

```python
# 发送端（vllm/distributed/parallel_state.py）
if self._should_use_all_gather(key, tensor.numel(), all_gather_group, ...):
    # 每个 TP rank 只发送 1/tp_size 分片
    tensor = tensor.reshape(all_gather_size, -1)[all_gather_rank]
handle = torch.distributed.isend(tensor, ...)

# 接收端
if self._should_use_all_gather(key, full_tensor.numel(), all_gather_group, ...):
    slice_tensor = full_tensor.reshape(all_gather_size, -1)[all_gather_rank]
    handle = torch.distributed.irecv(slice_tensor, ...)
    # postprocess: 在本地 NVLink 上 all_gather 恢复完整 tensor
    postprocess.append(lambda: all_gather(full_tensor, ...))
```

效果对比：

```text
无优化：每个 TP rank 发送完整 hidden_states（seq_len × hidden_dim）
有优化：每个 TP rank 只发送 hidden_states / tp_size
        接收后在本地 NVLink 上 all_gather（带宽更高、延迟更低）
→ 跨节点通信量减少 tp_size 倍，把慢速跨节点链路换成快速 NVLink
```

### 5.6 什么拓扑下 PP 可能优于 TP

1. **跨节点多机场景**：
   - 节点间带宽有限（如 100Gbps InfiniBand）
   - TP 的 all-reduce 需要所有参与者通信，对带宽要求高
   - PP 只需相邻 stage P2P，跨节点通信量更少
   - 推荐拓扑：节点内 TP，节点间 PP

2. **GPU 数量 > TP 最大值**：
   - TP 受限于 attention head 数量（PP 无此限制）
   - 如 Llama-7B 只有 32 个 head，TP 最大 32-way
   - 需要更多 GPU 时必须用 PP

3. **异构 GPU 部署**：
   - TP 要求所有 GPU 完全同构
   - PP 允许不同 stage 用不同 GPU（虽然 vLLM 目前不直接支持）

### 5.7 什么情况下 PP “能跑”但未必划算

1. **单机 NVLink 连接下用 PP**：NVLink 带宽足以支撑 TP 的 all-reduce，PP 的 bubble 反而是浪费
2. **PP 度数过高**：PP=8 意味着 87.5% 的 bubble（单 batch），严重影响单请求延迟
3. **Decode 场景**：每步只处理极少 token，PP 通信延迟在总时间中占主导
4. **小模型**：模型足够小能放进单卡，用 PP 纯粹增加延迟

---

## 六、vLLM PP 与 TP/DP/EP 的关系

### 6.1 PP × TP 正交组合

- **World size**：`PP × TP × CP`
- **通信组**：PP group 和 TP group 完全独立
- **实践中的组合**：
  - PP=2, TP=4: 8 GPU，2 个 stage 每 4 卡做 TP
  - PP=4, TP=2: 8 GPU，4 个 stage 每 2 卡做 TP
  - 通常建议节点内用 TP，节点间用 PP

### 6.2 PP × DP

- **V1 架构中**：DP 是在 PP × TP 之上的复制
- 每个 DP rank 是一个完整的 PP × TP pipeline
- GPU 总数 = DP × PP × TP
- **调度**：每个 DP rank 有独立的 Engine Core 和 Scheduler

### 6.3 PP × EP

- PP 与 Elastic EP 互斥，不能同时启用（`vllm/config/parallel.py`）
- PP 与标准 EP 的兼容性取决于具体模型实现，MoE 模型的 PP 支持需要特殊处理

### 6.4 PP × Context Parallelism

- world_size 计算包含 `prefill_context_parallel_size`
- PP 与 CP 可以组合使用，但实际场景较少，可能存在未发现的兼容性问题

### 6.5 切分策略选择指南

```text
决策树：
│
├─ 单卡能放下模型？
│   ├─ 是 → 不用并行
│   └─ 否 → 需要切分
│       │
│       ├─ 单机多卡 + NVLink？
│       │   ├─ 是 → 优先 TP
│       │   │   ├─ TP 够用？→ 只用 TP
│       │   │   └─ TP 不够？→ TP + PP（PP 尽量小）
│       │   └─ 否 → TP（节点内）+ PP（节点间）
│       │
│       ├─ MoE 模型？
│       │   ├─ 是 → 考虑 EP（但与 PP 互斥）
│       │   └─ 否 → TP + PP
│       │
│       └─ 需要提高吞吐？
│           └─ DP + (TP/PP)
```

---

## 七、对推理 PP 的进一步思考

### 7.1 未来优化方向

1. **更细粒度的微批次流水线调度（Micro-batch Pipeline Scheduling）**：当前 `batch_queue` 机制基于整 batch 粒度；更进一步的优化是 micro-batch 级别的交错调度，使 pipeline 在 continuous batching 下更充分
2. **非均匀 Stage 切分**：根据实际 profiling 数据自动调整各 stage 的层数
3. **Prefill/Decode 分离 + PP**：将 prefill 和 decode 分到不同的 PP pipeline
4. **Decode 专项优化**：decode 阶段每步计算量极小，PP 通信延迟在总时间中占比大，需针对性优化

### 7.2 PP 的工程挑战

1. **调试困难**：PP 跨多个进程，故障定位复杂
2. **性能剖析不直观**：需要同时分析多个 rank 的 trace
3. **KV cache 管理**：调度器需要感知所有 stage 的显存状态
4. **错误传播**：一个 stage 出错会导致整个 pipeline 停滞

### 7.3 PP 在实际生产中的定位

在当前 vLLM 实现中，PP 的主要价值是：

1. **可部署性**：让超大模型能在有限 GPU 上运行
2. **跨机适配**：作为跨节点部署的桥梁
3. **TP 的补充**：当 TP 不足时作为额外维度

而非作为性能优化手段。改善 PP 的 bubble 损失和通信开销仍需持续的工程优化。

---

## 八、参考资料

### 源码文件

| 文件路径                                          | 关键内容                                                                                                    |
| ------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| `vllm/config/parallel.py`                         | `pipeline_parallel_size` 定义, world_size 计算, Elastic EP 互斥检查                                         |
| `vllm/distributed/utils.py`                       | `get_pp_indices()` 层分区算法                                                                               |
| `vllm/distributed/parallel_state.py`              | `GroupCoordinator`, `get_pp_group()`, PP group 初始化, `isend_tensor_dict`, `irecv_tensor_dict`             |
| `vllm/model_executor/models/utils.py`             | `PPMissingLayer`, `make_layers()`, `is_pp_missing_parameter()`, `make_empty_intermediate_tensors_factory()` |
| `vllm/model_executor/models/interfaces.py`        | `SupportsPP` Protocol, `supports_pp()` 函数                                                                 |
| `vllm/model_executor/models/transformers/base.py` | Transformers 后端 PP 支持（`pipeline_parallel`, `forward`）                                                 |
| `vllm/sequence.py`                                | `IntermediateTensors` 数据结构                                                                              |
| `vllm/model_executor/models/llama.py`             | Llama PP 实现（案例分析）                                                                                   |
| `vllm/v1/worker/gpu_worker.py`                    | Worker 层面 PP 通信（isend/irecv）, `AsyncIntermediateTensors`                                              |
| `vllm/v1/worker/gpu/model_runner.py`              | Model Runner PP 状态管理、intermediate_tensors 预分配缓冲区、`postprocess` 方法                             |
| `vllm/v1/worker/gpu/pp_utils.py`                  | `pp_broadcast`, `pp_receive`                                                                                |
| `vllm/v1/worker/gpu/input_batch.py`               | `_post_update_kernel`（Triton 5 状态更新）、`combine_sampled_and_draft_tokens`（input_ids 构造）            |
| `vllm/v1/worker/gpu/states.py`                    | `last_sampled_tokens`、`all_token_ids`、`num_computed_tokens` 状态字段                                      |
| `vllm/v1/core/sched/scheduler.py`                 | `update_from_output`（调度器端 CPU 状态更新）                                                               |
| `vllm/v1/worker/gpu/attn_utils.py`                | `get_kv_cache_spec`（PP stage 的 KV cache 独立管理）                                                        |
| `vllm/v1/engine/core.py`                          | `step_with_batch_queue`, `batch_queue_size`（Pipeline Bubble 消除）                                         |
| `vllm/v1/executor/multiproc_executor.py`          | `max_concurrent_batches`, `_get_output_rank`                                                                |
| `vllm/engine/arg_utils.py`                        | PP 支持检查（executor backend 验证）                                                                        |
| `tests/distributed/test_pipeline_partition.py`    | PP 分区测试用例                                                                                             |
| `vllm/envs.py`                                    | `VLLM_PP_LAYER_PARTITION` 环境变量                                                                          |

### 官方文档

| 文档                       | URL                                                                        |
| -------------------------- | -------------------------------------------------------------------------- |
| Architecture Overview      | https://docs.vllm.ai/en/latest/design/arch_overview/                       |
| Parallelism and Scaling    | https://docs.vllm.ai/en/latest/serving/parallelism_scaling/                |
| Engine Arguments           | https://docs.vllm.ai/en/latest/configuration/engine_args/                  |
| Multi-Node Serving Example | https://docs.vllm.ai/en/latest/examples/online_serving/multi-node-serving/ |
