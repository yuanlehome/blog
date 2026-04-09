---
title: 深入 SGLang Overlap 调度：CPU-GPU Overlap、SBO 与 TBO 源码分析
slug: sglang-batch-overlap-analysis
date: '2026-04-09'
tags: ['Source Code Analysis']
status: published
source: original
---

本文基于 SGLang 源码（`main` 分支，commit `ab8b83f71`）梳理 CPU-GPU Overlap、Single Batch Overlap（SBO）与 Two Batch Overlap（TBO）的整体设计、关键数据结构与执行路径。

## 结论摘要

1. SGLang 一共实现了三层可组合的 overlap 机制：调度层的 CPU-GPU Overlap、单批次内的 SBO，以及双批次交错执行的 TBO。

2. CPU-GPU Overlap 默认开启，本质是在 GPU 执行当前 batch 前向时，让 CPU 同步处理上一批结果；它依赖 `forward_stream`、`copy_stream`、`schedule_stream` 以及 `FutureMap` 协同完成。

3. `FutureMap` 用“负数占位符 + 环形缓冲区”的方式延迟解析下一批 `input_ids`，解决 overlap 调度下 sampled token 尚未落地的问题。

4. CPU-GPU Overlap 并非总是可用；SGLang 会在 PP、多种特殊设备、特定推测解码路径和部分特殊模型上自动禁用它。

5. SBO 聚焦单个 MoE 层内部，通过双 CUDA Stream、SM 分区、CUDA Event 和 signal tensor，让 combine 通信与 Down GEMM 或 shared experts 计算重叠。

6. `compute_overlap_args()` 是 SBO 的资源分配核心：Blackwell 默认给通信侧 32 个 SM，Hopper 默认给通信侧 3 个 SM，其余 SM 留给计算。

7. `SboFlags` 将 SBO 细分为三种模式：Combine-Down GEMM 双流重叠、Combine-Shared Experts 双流重叠，以及 Dispatch-Shared Experts 单流重叠。

8. TBO 的核心不是多 stream，而是把一个 batch 拆成两个子 batch，再用 stage 级交错执行隐藏 dispatch/combine 的通信延迟。

9. 在 decode 路径下，TBO 的 `delta_stages = 2`，让子批次 B 相对 A 滞后两个 stage；在 prefill 路径下，`delta_stages = 0`，两个子批次同步推进。

10. TBO 的拆分策略并不固定：extend 模式优先做按序列的平衡拆分，失衡严重时会退化到 two-chunk，把单条序列切到两个子批次里。

11. `YieldOperation`、`_convert_operations_to_stages()` 与 `_StageExecutor` 组成了 TBO 的执行框架，负责 stage 切分、状态传递与 DP buffer 长度同步。

12. TBO 还会影响 attention backend、dispatcher、CUDA Graph 与 DP Attention 一致性，因此它是一个跨 scheduler、model executor 与 MoE dispatcher 的系统级设计。

13. SBO 和 TBO 可以同时开启：TBO 负责子批次之间的交错，SBO 负责每个子批次内部 MoE 层的通信-计算重叠。

14. 从收益边界看，CPU-GPU Overlap 主要隐藏调度与结果处理开销，SBO 主要隐藏单层 combine 延迟，而 TBO 以更高的实现复杂度和额外内存开销换取更大范围的通信隐藏。

---

## 一、概述

SGLang 中实现了 **三个层次** 的 Overlap 机制，它们是正交的，可以组合使用：

| 层次   | 名称                       | 启用方式                          | 默认状态     | 核心目标                                                                                              |
| ------ | -------------------------- | --------------------------------- | ------------ | ----------------------------------------------------------------------------------------------------- |
| **L1** | CPU-GPU Overlap Schedule   | `--disable-overlap-schedule` 关闭 | **默认开启** | 将 CPU 调度与 GPU 前向计算重叠                                                                        |
| **L2** | Single Batch Overlap (SBO) | `--enable-single-batch-overlap`   | 默认关闭     | **单个 batch 内部**，将 MoE 通信（combine）与计算（Down GEMM / Shared Experts）重叠（双 CUDA stream） |
| **L3** | Two Batch Overlap (TBO)    | `--enable-two-batch-overlap`      | 默认关闭     | 将 batch 拆分为两个 micro-batch，**交错执行**以隐藏通信延迟                                           |

三者的定位关系（包含关系）：

```text
┌─────────────────────────────────────────────────────┐
│  L1: CPU-GPU Overlap (Scheduler 级别)                │
│  ┌───────────────────────────────────────────────┐  │
│  │  L3: TBO (Forward 内部，batch 级别拆分)        │  │
│  │  ┌─────────────────────────────────────────┐  │  │
│  │  │  L2: SBO (MoE layer 内部，stream 级别)   │  │  │
│  │  └─────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘
```

在大规模 MoE（Mixture-of-Experts）模型推理中（如 DeepSeek-V2/V3），Expert Parallelism（EP）涉及大量的 All-to-All 通信，通信延迟是关键性能瓶颈。三层 Overlap 机制分别从调度、层内和跨批次三个维度来隐藏这一延迟。

---

## 二、配置与启用方式

### 2.1 Server Args 配置（`python/sglang/srt/server_args.py`）

```python
disable_overlap_schedule: bool = False          # 禁用调度层 overlap（默认开启）
enable_two_batch_overlap: bool = False           # 启用 TBO
enable_single_batch_overlap: bool = False        # 启用 SBO
tbo_token_distribution_threshold: float = 0.48   # TBO 拆分平衡阈值（控制 two-chunk split）
```

### 2.2 环境变量

| 环境变量                                              | 默认值                 | 说明                                                         |
| ----------------------------------------------------- | ---------------------- | ------------------------------------------------------------ |
| `SGLANG_DISABLE_CONSECUTIVE_PREFILL_OVERLAP`          | `false`                | 禁止连续两个 prefill 之间的 CPU-GPU overlap（优化 TTFT）     |
| `SGLANG_DEEPEP_LL_COMBINE_SEND_NUM_SMS`               | Blackwell: 32, 其他: 3 | SBO combine 通信使用的 SM 数量                               |
| `SGLANG_BLACKWELL_OVERLAP_SHARED_EXPERTS_OUTSIDE_SBO` | `false`                | Blackwell 架构：在 SBO 外部的备用 stream 运行 shared experts |
| `SGLANG_TBO_DEBUG`                                    | `false`                | 开启 TBO 调试日志                                            |
| `SGLANG_OPERATIONS_ENABLE_PROFILE`                    | `0`                    | 开启 operations 层 NVTX profiling                            |
| `SGLANG_NPU_USE_MULTI_STREAM`                         | —                      | NPU 使用多 stream                                            |

---

## 三、CPU-GPU Overlap Schedule（基线 Overlap）

**源码**：`python/sglang/srt/managers/scheduler.py`

### 3.1 核心思想

在 LLM 推理服务中，每次 forward pass 后，CPU 需要进行结果处理（token 采样、请求状态更新、调度下一个 batch 等）。**CPU-GPU Overlap** 的核心思想是：在 GPU 执行当前 batch 的前向计算时，CPU 同时处理**上一个 batch** 的结果，从而将 CPU 处理时间完全隐藏在 GPU 计算时间内。

### 3.2 CUDA Stream 设计

Scheduler 初始化时创建三个 CUDA Stream（`scheduler.py`）：

```python
def init_overlap(self):
    # 1. forward_stream：用于 GPU 前向计算（来自 ModelRunner）
    self.forward_stream_ctx = self.device_module.stream(self.forward_stream)
    # 2. copy_stream：用于 D2H 非阻塞内存拷贝
    self.copy_stream = self.device_module.Stream()
    # 3. schedule_stream：整个事件循环运行于此（在 run_event_loop 中创建）

    if self.enable_overlap:
        self.future_map = FutureMap(...)    # 存储"未来" token id 的环形缓冲区
        self.batch_record_buf = [None] * 2  # 双缓冲，保持 batch 引用防止 GC
        self.batch_record_ct = 0
```

### 3.3 Overlap 事件循环

**普通循环**（`event_loop_normal`）：串行执行，GPU 完成后才处理结果

```text
recv → schedule → forward (GPU 阻塞) → process_result → recv → ...
```

**Overlap 循环**（`event_loop_overlap`，`scheduler.py`）：CPU 调度与 GPU 前向并行

```text
时间线（GPU 视角）:
  GPU:  [  Forward(batch N-1)  ] [  Forward(batch N)  ] [  Forward(batch N+1)  ]
  CPU:            [Process(N-2)] [Schedule(N)][Process(N-1)] [Schedule(N+1)]
                                      ↑ overlap ↑
```

核心逻辑：

```python
def event_loop_overlap(self):
    self.result_queue = deque()  # 缓存等待处理的 (batch, result) 对

    while True:
        # ① CPU 工作：接收请求、调度下一个 batch（在 schedule_stream 上）
        recv_reqs = self.recv_requests()
        self.process_input_requests(recv_reqs)
        batch = self.get_next_batch_to_run()
        disable_overlap_for_batch = self.is_disable_overlap_for_batch(batch)

        # ② 特殊情况：如果需要禁止 overlap（如连续 prefill），先处理上一批结果
        if disable_overlap_for_batch:
            pop_and_process()

        # ③ 启动当前批次 GPU 前向（非阻塞，在 forward_stream 上）
        if batch:
            batch_result = self.run_batch(batch)
            self.result_queue.append((batch.copy(), batch_result))

        # ④ 在 GPU 计算当前批次时，处理上一批结果（overlap 核心）
        if self.last_batch and not disable_overlap_for_batch:
            pop_and_process()

        # ⑤ 延迟采样（用于 grammar 约束场景）
        self.launch_batch_sample_if_needed(batch_result)
```

**禁用 Overlap 的条件**（`is_disable_overlap_for_batch`）：

```python
def is_disable_overlap_for_batch(self, batch):
    # 1. 连续 prefill 时禁用（优化 TTFT）
    disable = (SGLANG_DISABLE_CONSECUTIVE_PREFILL_OVERLAP
               and batch_is_extend and last_batch_is_extend)

    # 2. 推测解码 + grammar 时禁用（尚未支持）
    need_grammar_sync = (batch.is_spec_v2 and batch.has_grammar
                         and batch.forward_mode.is_decode()
                         and len(self.result_queue) > 0)

    return disable or need_grammar_sync
```

### 3.4 Future Map 机制

**源码**：`python/sglang/srt/managers/overlap_utils.py`

由于 overlap 模式下，CPU 调度 batch N+1 时，batch N 的 `next_token_ids` 尚未就绪（GPU 尚在计算中）。SGLang 使用 `FutureMap`（`overlap_utils.py`）解决这一问题：

```python
class FutureMap:
    """循环缓冲区，存储 future token IDs"""
    def __init__(self, ...):
        self.future_ct = 0
        self.future_limit = max_running_requests * (3 + max_num_chunks)
        # 环形缓冲区，存放真实 token id
        self.token_ids_buf = torch.empty((self.future_buffer_len,), dtype=torch.int64, device=device)

    def alloc_future_indices(self, bs):
        """分配负数索引作为占位符"""
        cur_future_ct = self.future_ct
        self.future_ct = (cur_future_ct + bs) % self.future_limit
        indices = torch.arange(cur_future_ct + 1, cur_future_ct + 1 + bs, ...)
        return FutureIndices(indices=indices, interval=slice(...))

    def resolve_future(self, model_worker_batch):
        """在 forward 开始前，将 input_ids 中的负数占位符替换为真实 token id"""
        # 使用 JIT CUDA kernel 高效替换
        _resolve_future_token_ids(model_worker_batch.input_ids, self.token_ids_buf)

    def store_to_map(self, future_indices, batch_result):
        """forward 完成后，将实际 token id 存入缓冲区"""
        self.token_ids_buf[intv] = batch_result.next_token_ids
```

**完整流程**：

1. batch N 前向启动前，分配 `future_indices`（负数索引）
2. batch N+1 调度时，使用 `-future_indices` 作为 `input_ids` 的占位符
3. batch N+1 前向开始时，在 `forward_stream` 上调用 `resolve_future()` 将占位符替换为真实值
4. batch N 前向完成后，调用 `store_to_map()` 将真实 token id 写入缓冲区

### 3.5 run_batch 的 Overlap 路径

```python
def run_batch(self, batch):
    if self.enable_overlap:
        model_worker_batch = batch.get_model_worker_batch()
        self.record_batch_in_overlap(model_worker_batch)  # 防止 GPU tensor 被 GC

        bs = len(model_worker_batch.seq_lens)
        future_indices = self.future_map.alloc_future_indices(bs)

        with self.forward_stream_ctx:  # 切换到 forward_stream
            self.forward_stream.wait_stream(self.schedule_stream)  # 同步点：确保调度完成
            self.future_map.resolve_future(model_worker_batch)     # 解析 future 占位符
            batch_result = self.model_worker.forward_batch_generation(model_worker_batch)

            # 非阻塞 D2H 拷贝（在 forward_stream 上记录 event，copy_stream 等待后拷贝）
            batch_result.copy_done = self.device_module.Event()
            self.future_map.store_to_map(future_indices, batch_result)
            batch_result.copy_to_cpu(...)
```

### 3.6 自动禁用条件

以下场景会自动禁用 CPU-GPU Overlap（`server_args.py`）：

- Pipeline 并行（`pp_size > 1`）
- MPS / XPU 设备
- NGRAM 推测解码
- Sparse head embedding 模型
- Mamba 模型（no_buffer 策略）
- Diffusion LLM 推理

---

## 四、Single Batch Overlap（SBO）详解

**核心源码**：`python/sglang/srt/batch_overlap/single_batch_overlap.py`

### 4.1 设计目标与核心思想

SBO 针对的是**单个批次内** MoE 层的通信-计算重叠。在 DeepEP（Expert Parallelism）的 MoE 前向中，关键步骤是：

```text
Gate → Select → Dispatch(通信) → Expert GEMM(up+gate) → Down GEMM → Combine(通信) → Shared Experts → Output
```

SBO 的核心思想：

- **不拆分 batch**，而是在 CUDA kernel 级别分区
- **将 GPU 的 SM 分为两组**：一组给通信（DeepEP combine），一组给计算（Down GEMM 或 Shared Experts）
- **使用两个 CUDA Stream**：主 stream 执行计算，`alt_stream` 执行通信
- **使用 CUDA Event + Signal Tensor** 进行细粒度生产者-消费者同步

### 4.2 SBO 标志位与开关

```python
# server_args.py
enable_single_batch_overlap: bool = False  # CLI: --enable-single-batch-overlap

# moe/utils.py
IS_SBO_ENABLED = server_args.enable_single_batch_overlap
def is_sbo_enabled() -> bool:
    return IS_SBO_ENABLED
```

`single_batch_overlap.py` 定义了 `SboFlags` 类，控制三种具体的 Overlap 模式：

```python
class SboFlags:
    @classmethod
    def enable_combine_down_gemm_two_stream_overlap(cls):
        """模式1: Combine 通信与 Down-Projection GEMM 重叠（双 stream）"""
        return (
            is_sbo_enabled()
            and (
                get_moe_runner_backend().is_flashinfer_cutedsl()
                or (get_moe_runner_backend().is_deep_gemm() and not is_blackwell())
            )
        )

    @classmethod
    def enable_combine_shared_two_stream_overlap(cls):
        """模式2: Combine 通信与 Shared Experts 计算重叠（双 stream）"""
        return (
            is_sbo_enabled()
            and not cls.enable_dispatch_shared_one_stream_overlap()
            and not envs.SGLANG_BLACKWELL_OVERLAP_SHARED_EXPERTS_OUTSIDE_SBO
        )

    @classmethod
    def enable_dispatch_shared_one_stream_overlap(cls):
        """模式3: Dispatch 通信与 Shared Experts 重叠（单 stream，非 Blackwell）"""
        return is_sbo_enabled() and not is_blackwell()

    @classmethod
    def fuse_shared_experts_inside_sbo(cls):
        """是否将 Shared Experts 融合到 SBO 流程中"""
        return (
            cls.enable_combine_shared_two_stream_overlap()
            or cls.enable_dispatch_shared_one_stream_overlap()
        )
```

### 4.3 核心数据结构

```python
@dataclass
class CombineOverlapArgs:
    """Combine 通信端的 overlap 参数"""
    overlap: bool                       # 是否与 down gemm 重叠
    stream: torch.cuda.Stream           # 备用 stream（通信在此 stream 上执行）
    wait_event: torch.cuda.Event        # 同步事件
    num_sms: Optional[int] = None       # 分配给通信的 SM 数量
    signal: Optional[torch.Tensor] = None  # 细粒度同步 signal tensor
    block_m: Optional[int] = 64         # block 大小
    threshold: Optional[int] = 0        # 计算侧 SM 数阈值

@dataclass
class DownGemmOverlapArgs:
    """Down-Projection GEMM 计算端的 overlap 参数"""
    num_sms: int                        # 分配给计算的 SM 数量
    signal: torch.Tensor                # 与 combine 共享的 signal tensor
    start_event: torch.cuda.Event       # 开始事件
```

### 4.4 SM 分区机制

`compute_overlap_args()` 函数（`single_batch_overlap.py`）实现了 GPU SM 的动态分区：

```python
def compute_overlap_args(dispatch_output, alt_stream):
    total_num_sms = torch.cuda.get_device_properties("cuda").multi_processor_count

    # SM 分配策略
    if envs.SGLANG_DEEPEP_LL_COMBINE_SEND_NUM_SMS.is_set():
        communicate_num_sms = envs.SGLANG_DEEPEP_LL_COMBINE_SEND_NUM_SMS.get()
    else:
        communicate_num_sms = 32 if is_blackwell() else 3
    compute_num_sms = total_num_sms - communicate_num_sms

    # 创建同步原语
    combine_wait_event = torch.cuda.Event()
    combine_overlap_args = CombineOverlapArgs(
        overlap=False,
        num_sms=communicate_num_sms,
        stream=alt_stream,          # 通信在备用 stream 执行
        wait_event=combine_wait_event,
    )

    if SboFlags.enable_combine_down_gemm_two_stream_overlap():
        # 创建 signal tensor 用于细粒度生产者-消费者同步
        combine_signal = torch.zeros(num_local_experts, dtype=torch.uint32, ...)
        down_gemm_overlap_args = DownGemmOverlapArgs(
            signal=combine_signal,
            start_event=combine_wait_event,
            num_sms=compute_num_sms,
        )
        combine_overlap_args.overlap = True
        combine_overlap_args.signal = combine_signal
```

| GPU 架构            | 通信 SM 数 | 计算 SM 数 | 备注                       |
| ------------------- | ---------- | ---------- | -------------------------- |
| Blackwell (B200 等) | 32         | total - 32 | SM 资源充足，通信需更多 SM |
| Hopper (H100/H200)  | 3          | total - 3  | 通信只需少量 SM            |

### 4.5 三种 SBO 模式

#### 模式 1：Combine-Down GEMM 双流重叠

**条件**：`enable_combine_down_gemm_two_stream_overlap() = True`

```text
主 Stream:    [Expert Up/Gate GEMM] → [Down GEMM (compute_num_sms 个 SM)]
                                              ↑ start_event
alt Stream:                           [Combine 通信 (communicate_num_sms 个 SM)]
                                        ↑ wait_event (等待 Down GEMM 就绪)

共享: signal 张量用于实现 SM 级同步（Down GEMM 写完一块，Combine 立即读）
```

Signal tensor 大小取决于架构：

- **Blackwell**：`num_local_experts` 个 `uint32` 元素
- **Hopper**：`num_local_experts × ceil(num_tokens / 64)` 个 `int32` 元素（按 `block_m=64` 细分）

Down GEMM 通过 `DownGemmOverlapArgs` 限制使用 `compute_num_sms` 个 SM，同时通过 `signal` 张量在 block 级别与 combine 通信同步。

#### 模式 2：Combine-Shared Experts 双流重叠

**条件**：`enable_combine_shared_two_stream_overlap() = True`

```text
主 Stream:    [Down GEMM] → record event → [Shared Experts (compute_num_sms 个 SM)]
alt Stream:                  wait event  → [Combine 通信 (communicate_num_sms 个 SM)]

两者在 Down GEMM 完成后并行执行。
```

#### 模式 3：Dispatch-Shared Experts 单流重叠

**条件**：`enable_dispatch_shared_one_stream_overlap() = True`（Hopper 及以前架构）

```text
主 Stream: [Dispatch 通信] → hook 触发 → [Shared Experts] → [Expert GEMM] → [Combine]

在 Dispatch 通信触发 hook 回调时执行 Shared Experts（利用通信等待时间）。
```

### 4.6 执行流程图

**完整 SBO 流程（以 `forward_deepep` 为例，`deepseek_v2.py`）：**

```text
                         ┌─────────────┐
                         │   Gate(路由) │
                         └──────┬──────┘
                                ▼
                         ┌─────────────┐
                         │  Select TopK │
                         └──────┬──────┘
                                ▼
              ┌─────────────────┼──────────────────┐
    SBO 模式1 │                 │                   │ SBO 模式3
  (dispatch)  │                 │(非 SBO)           │(dispatch-shared)
              ▼                 ▼                   ▼
   ┌────────────────┐  ┌──────────────┐   ┌────────────────┐
   │ Dispatch (通信) │  │ alt_stream:  │   │ Dispatch + hook │
   │ + hook:         │  │ shared_exp   │   │ → Shared Experts│
   │   shared_exp    │  └──────────────┘   └────────────────┘
   └────────┬───────┘                              │
            ▼                                      ▼
     ┌──────────────┐                      ┌──────────────┐
     │ Expert GEMM   │                      │ Expert GEMM   │
     │ (Up+Gate→Down)│                      │ (Up+Gate→Down)│
     └──────┬───────┘                      └──────┬───────┘
            ▼                                      ▼
  ┌────────────────────┐                  ┌──────────────────┐
  │ Down GEMM + Combine│                  │     Combine      │
  │ (SM 分区并行执行)   │                  │                  │
  └────────┬───────────┘                  └──────┬──────────┘
            ▼                                      ▼
     ┌──────────────┐                      ┌──────────────┐
     │    Output     │                      │    Output     │
     └──────────────┘                      └──────────────┘
```

### 4.7 CUDA Stream 与 Event 同步

**Alt Stream 创建**（`deepseek_v2.py` 模型初始化时）：

```python
self.alt_stream = (
    torch.cuda.Stream()
    if _is_cuda or envs.SGLANG_NPU_USE_MULTI_STREAM.get()
    else None
)
```

**同步机制**（以 DeepEP Dispatcher 的 `combine_a` / `combine_b` 拆分为例）：

```python
# combine_a（主流程）：在 alt_stream 上启动 combine 通信
def combine_a(self, ...):
    hidden_states, event, hook = self._combine_core(...)
    return hidden_states, event, hook

# _combine_core：切换到 alt_stream
def _combine_core(self, ...):
    if overlap_args is not None:
        overlap_args.stream.wait_event(overlap_args.wait_event)  # 等待 Down GEMM 就绪
        ctx = torch.cuda.stream(overlap_args.stream)              # 切换到 alt_stream
    with ctx:
        combined, event, hook = buffer.low_latency_combine(
            overlap=overlap_args.overlap,
            src_signals=overlap_args.signal,      # SM 级信号同步
            num_sms=overlap_args.num_sms,         # 限制通信使用的 SM 数
            ...
        )

# combine_b（同步点）：等待两路完成
def combine_b(self, ...):
    if overlap_args is not None:
        overlap_args.stream.wait_stream(current_stream())   # alt_stream 等主流
    hook()  # 等待通信完成
    if overlap_args is not None:
        current_stream().wait_stream(overlap_args.stream)    # 主流等 alt_stream
    return hidden_states
```

### 4.8 Dispatcher 中的使用

SBO 的 overlap 参数通过 dispatcher 的 hook 机制注入（`token_dispatcher/base.py`）：

```python
class BaseDispatcher:
    def set_overlap_args(self, combine_overlap_args, meta_overlap_args):
        """注入 SBO 参数（在 MoE forward 开始时调用）"""
        self.combine_overlap_args = combine_overlap_args
        self.meta_overlap_args = meta_overlap_args

    def clear_overlap_args(self):
        """清除 SBO 参数（在 MoE forward 结束时调用）"""
        self.combine_overlap_args = None
        self.meta_overlap_args = None
```

支持 SBO 的 dispatcher：

- **DeepEP Dispatcher**（`token_dispatcher/deepep.py`）
- **MoriEP Dispatcher**（`token_dispatcher/moriep.py`）

---

## 五、Two Batch Overlap（TBO）详解

**核心源码**：`python/sglang/srt/batch_overlap/two_batch_overlap.py`、`operations.py`、`operations_strategy.py`

### 5.1 设计目标与核心思想

TBO 的目标是实现**跨子批次**的流水线并行：

1. 将一个批次**拆分为两个子批次** A 和 B
2. 将每个子批次的 MoE 层分解为多个 **stage**（通过 `YieldOperation` 分隔）
3. **交错执行**两个子批次的 stage：当 A 执行 dispatch 通信时，B 执行 attention 计算
4. 通过 `delta_stages` 控制 A 提前 B 几个 stage

核心洞察：MoE 层中通信（dispatch/combine）和计算（attention/GEMM）可以在**同一 CUDA stream 上交错执行**而不需要多 stream —— 因为两个子批次在数据上完全独立，自然形成隐式同步。

```text
                  Micro-batch A        Micro-batch B
Time ─────────────────────────────────────────────────►

Stage 0: [  Attn_Prepare_A   ]
Stage 1: [  Attn_Core_A      ] [  Attn_Prepare_B   ]
Stage 2: [  Dispatch_A       ] [  Attn_Core_B       ]
Stage 3: [  Experts_A        ] [  Dispatch_B        ]
Stage 4: [  Combine_A        ] [  Experts_B         ]
Stage 5: [  Output_A         ] [  Combine_B         ]
Stage 6:                       [  Output_B          ]
```

**配置与前置条件**：

```python
# server_args.py
enable_two_batch_overlap: bool = False     # CLI: --enable-two-batch-overlap
tbo_token_distribution_threshold: float = 0.48  # CLI: --tbo-token-distribution-threshold

# moe/utils.py
IS_TBO_ENABLED = server_args.enable_two_batch_overlap
def is_tbo_enabled() -> bool:
    return IS_TBO_ENABLED
```

TBO 需要 MoE A2A backend 不为 none（`server_args.py`），即需要配合 Expert Parallelism 使用。

### 5.2 整体架构

TBO 的实现分为以下几个关键层次：

```text
┌─────────────────────────────────────────────────────┐
│  Model Layer (deepseek_v2.py, qwen2_moe.py, ...)    │
│  └── model_forward_maybe_tbo()                       │
│      ├── Batch Splitting (TboForwardBatchPreparer)   │
│      ├── Operations Strategy (operations_strategy.py)│
│      └── Overlapped Execution (operations.py)        │
├─────────────────────────────────────────────────────┤
│  Attention Backend (tbo_backend.py)                  │
│  └── TboAttnBackend (primary + 2 children)           │
├─────────────────────────────────────────────────────┤
│  MoE Dispatcher (two_batch_overlap.py)               │
│  └── MaybeTboDeepEPDispatcher (2 inner dispatchers)  │
├─────────────────────────────────────────────────────┤
│  DP Attention Coordinator (scheduler_dp_attn_mixin)  │
│  └── TboDPAttentionPreparer (全局一致性协调)          │
├─────────────────────────────────────────────────────┤
│  CUDA Graph Support (cuda_graph_runner.py)           │
│  └── TboCudaGraphRunnerPlugin                        │
└─────────────────────────────────────────────────────┘
```

### 5.3 批次拆分策略

#### 5.3.1 拆分入口

```python
def compute_split_seq_index(forward_mode, num_tokens, extend_lens, token_num_per_seq):
    if forward_mode == ForwardMode.EXTEND:
        return _split_extend_seqs(extend_lens)    # Prefill 按 token 数平衡拆分
    elif forward_mode.is_decode():
        return (num_tokens // token_num_per_seq) // 2  # Decode 直接对半分
    elif forward_mode.is_target_verify():
        return (num_tokens // token_num_per_seq) // 2  # Target verify 对半分
```

#### 5.3.2 Extend 模式的拆分策略

**策略 1：按序列平衡拆分**（默认）

```python
def _split_array_by_balanced_sum(arr):
    """找到使左右 token 总数差最小的拆分点"""
    overall_sum = sum(arr)
    left_sum = 0
    for i in range(1, len(arr)):
        left_sum += arr[i - 1]
        right_sum = overall_sum - left_sum
        diff = abs(left_sum - right_sum)
        if diff <= min_diff:
            min_diff = diff
            best_index = i
    return best_index
```

**策略 2：Two-Chunk 拆分**（当 token 分布极度不均时启用）

```python
def _is_two_chunk_split_enabled(extend_lens):
    """当一侧占比超过 threshold 时启用 two-chunk"""
    threshold = get_tbo_token_distribution_threshold()  # 默认 0.48
    vanilla_split = _split_array_by_balanced_sum(extend_lens)
    left_sum = sum(extend_lens[:vanilla_split])
    overall_sum = sum(extend_lens)
    return left_sum < overall_sum * threshold or left_sum > overall_sum * (1 - threshold)
```

Two-Chunk 模式允许将单个序列拆分到两个子批次中（一部分 token 在 A，剩余在 B），使两侧负载更平衡：

```python
def _split_array_by_cum_less_than_half(arr):
    # 前半部分 token 归 A，后半部分归 B，允许在序列中间截断
    ...
```

#### 5.3.3 Token 级别拆分

```python
def compute_split_token_index(split_seq_index, forward_mode, extend_seq_lens, token_num_per_seq):
    if forward_mode == ForwardMode.EXTEND:
        if _is_two_chunk_split_enabled(extend_seq_lens):
            return sum(extend_seq_lens) // 2  # Two-chunk: 按总 token 数对半
        return sum(extend_seq_lens[:split_seq_index])   # 普通: 按序列边界切
    elif forward_mode.is_decode():
        return split_seq_index * token_num_per_seq
```

### 5.4 Operations 与 Stage 执行框架

**源码**：`python/sglang/srt/batch_overlap/operations.py`

TBO 将层的前向计算分解为 **Operations（操作）** 和 **Stages（阶段）**：

```python
class YieldOperation: pass           # 阶段分隔符
class ExecutionOperation:            # 可执行操作
    debug_name: str
    fn: Callable

# 将操作列表按 YieldOperation 分割为多个 Stage
def _convert_operations_to_stages(operations) -> List[Stage]:
    # [op1, op2, Yield, op3, op4, Yield, op5] → [[op1,op2], [op3,op4], [op5]]
    return list(_chunk_by_separator(operations, lambda op: isinstance(op, YieldOperation)))
```

**StageExecutor**：

```python
class _StageExecutor:
    def __init__(self, debug_name, stages, inputs):
        self._stages = stages
        self._stage_output = inputs  # 每个 stage 的输出作为下一个 stage 的输入
        self._stage_state = _StateDict()  # 跨 stage 的状态存储（如 dispatch 结果）
        # DP Attention 相关
        self._global_dp_buffer_len = inputs["forward_batch"].global_dp_buffer_len
        self._local_dp_buffer_len = inputs["forward_batch"].tbo_padded_len

    def next(self):
        """执行下一个 stage 中的所有操作"""
        stage = self._stages[self._index]
        # 设置 DP buffer 长度（子 batch 可能有不同的 padded 长度）
        set_dp_buffer_len(self._global_dp_buffer_len, self._local_dp_buffer_len, ...)
        # 依次执行 stage 中的所有操作，前一个操作的输出传给下一个
        for op in stage:
            self._stage_output = op.fn(state=self._stage_state, **self._stage_output)
        self._index += 1
```

**关键设计点**：

- 每个 operation 接收上一个 operation 的输出作为 `**kwargs`
- `_StateDict` 提供跨 stage 的状态共享（dispatch 结果在后续 combine 中使用）
- `set_dp_buffer_len` 在每次 stage 切换时更新，因为两个子 batch 可能有不同的 padded 长度

### 5.5 DeepSeek 策略定义

**源码**：`python/sglang/srt/batch_overlap/operations_strategy.py`

#### Decode 策略（`delta_stages = 2`）

```python
def _compute_moe_deepseek_blog_decode(layer):
    return OperationsStrategy(
        tbo_delta_stages=2,  # micro-batch B 比 A 延迟 2 个 stage
        operations=[
            layer.op_comm_prepare_attn,       # Stage 0
            layer.self_attn.op_prepare,
            YieldOperation(),                  # ─── Stage 0/1 ───
            layer.self_attn.op_core,          # Stage 1
            layer.op_comm_prepare_mlp,
            layer.mlp.op_gate,
            layer.mlp.op_select_experts,
            YieldOperation(),                  # ─── Stage 1/2 ───
            layer.mlp.op_dispatch_a,          # Stage 2
            layer.mlp.op_shared_experts,
            YieldOperation(),                  # ─── Stage 2/3 ───
            layer.mlp.op_dispatch_b,          # Stage 3
            layer.mlp.op_experts,
            layer.mlp.op_combine_a,
            YieldOperation(),                  # ─── Stage 3/4 ───
            layer.mlp.op_combine_b,           # Stage 4
            YieldOperation(),                  # ─── Stage 4/5 ───
            layer.mlp.op_output,              # Stage 5
            layer.op_comm_postprocess_layer,
        ],
    )
```

**Decode 阶段的交错执行**（`tbo_delta_stages=2`）：

```text
时间 →

Phase 1（A 先行 2 stage）:
  A: [Stage0: comm_prep + attn_prep] [Stage1: attn_core + gate + select]

Phase 2（交错执行）:
  A: [Stage2: dispatch_a + shared]   [Stage3: dispatch_b + experts + combine_a]   [Stage4: combine_b]   [Stage5: output]
  B: [Stage0: comm_prep + attn_prep] [Stage1: attn_core + gate + select]          [Stage2: dispatch_a]  [Stage3: dispatch_b + experts]

Phase 3（B 收尾）:
  B: [Stage4: combine_b]   [Stage5: output + postprocess]

关键效果: A 的 dispatch 通信（Stage2-3）与 B 的 attention 计算（Stage0-1）时间重叠
```

#### Prefill 策略（`delta_stages = 0`）

```python
def _compute_moe_deepseek_blog_prefill(layer):
    return OperationsStrategy(
        tbo_delta_stages=0,  # Prefill 无延迟偏移：两个子批次同步开始
        deep_gemm_num_sms=total_num_sms - DeepEPConfig.num_sms,
        operations=[
            # Stage 0: Attention + Gate + Dispatch Phase A
            layer.op_comm_prepare_attn,
            layer.self_attn.op_prepare,
            layer.self_attn.op_core,
            layer.op_comm_prepare_mlp,
            layer.mlp.op_gate,
            layer.mlp.op_select_experts,
            layer.mlp.op_dispatch_a,
            YieldOperation(),
            # Stage 1: Dispatch Phase B + Expert 计算 + Combine Phase A
            layer.mlp.op_dispatch_b,
            layer.mlp.op_experts,
            layer.mlp.op_combine_a,
            YieldOperation(),
            # Stage 2: Shared Experts + Combine Phase B + Output
            layer.mlp.op_shared_experts,
            layer.mlp.op_combine_b,
            layer.mlp.op_output,
            layer.op_comm_postprocess_layer,
        ],
    )
```

Prefill 阶段 `tbo_delta_stages=0`，两个 micro-batch 同步开始，stage 数量较少（3 个 stage）。

#### 支持的模型

| 模型           | 类名                     | 文件                      |
| -------------- | ------------------------ | ------------------------- |
| DeepSeek V2/V3 | `DeepseekV2DecoderLayer` | `models/deepseek_v2.py`   |
| Qwen3 MoE      | `Qwen3MoeDecoderLayer`   | `models/qwen2_moe.py`     |
| MiMo V2        | `MiMoV2DecoderLayer`     | `models/mimo_v2_flash.py` |
| MiniMax M2     | —                        | `models/minimax_m2.py`    |
| GLM4 MoE       | —                        | `models/glm4_moe.py`      |

### 5.6 TBO 前向执行流程

**入口**：`model_forward_maybe_tbo()`（`two_batch_overlap.py`）

```python
def model_forward_maybe_tbo(layers, enable_tbo, positions, forward_batch, hidden_states, ...):
    operations_strategy = OperationsStrategy.init_new_tbo(layers, forward_batch.global_forward_mode)

    if enable_tbo:
        return _model_forward_tbo(inputs, operations_strategy, ...)
    else:
        return _model_forward_non_tbo(inputs, operations_strategy)
```

**TBO 前向**：

```python
def _model_forward_tbo(inputs, operations_strategy, ...):
    # 1. 拆分输入为两个子批次（按 tbo_children 切片）
    inputs_arr = _model_forward_tbo_split_inputs(**inputs, ...)

    # 2. 配置 Deep GEMM 使用的 SM 数量（与 TBO 协同限制）
    with deep_gemm_wrapper.configure_deep_gemm_num_sms(operations_strategy.deep_gemm_num_sms):
        # 3. 交错执行两个子批次
        outputs_arr = execute_overlapped_operations(
            inputs_arr=inputs_arr,
            operations_arr=[operations_strategy.operations] * 2,  # 两个子批次用相同操作
            delta_stages=[0, operations_strategy.tbo_delta_stages],  # A 先跑，B 延后
        )

    # 4. 合并输出
    return _model_forward_tbo_merge_outputs(*outputs_arr, original_hidden_states_len)
```

**输入拆分**：

```python
def _model_forward_filter_inputs(hidden_states, residual, positions, output_forward_batch, ...):
    token_slice = slice(*output_forward_batch.tbo_parent_token_range)
    hidden_states = hidden_states[token_slice]
    residual = residual[token_slice] if residual is not None else None
    positions = positions[token_slice]
    # 对齐到 attention_tp_size 的倍数（零填充）
    hidden_states = _pad(hidden_states, output_forward_batch.tbo_padded_len)
    return dict(hidden_states=hidden_states, residual=residual, positions=positions, ...)
```

**输出合并**：

```python
def _model_forward_tbo_merge_outputs(output_a, output_b, original_len):
    res = torch.zeros((original_len, *value_a.shape[1:]), ...)
    res[slice(s0, t0)] = value_a[:t0 - s0]  # A 的有效部分（去掉 padding）
    res[slice(s1, t1)] = value_b[:t1 - s1]  # B 的有效部分（去掉 padding）
    return res
```

在模型的 `forward()` 中（以 DeepSeek V2 为例）：

```python
def forward(self, input_ids, positions, forward_batch, ...):
    # Dense 层正常执行（前 first_k_dense_replace 层）
    for i in range(self.start_layer, normal_end_layer):
        hidden_states, residual = self.layers[i](hidden_states, positions, forward_batch, residual)

    # Sparse MoE 层使用 TBO
    if forward_batch.can_run_tbo:
        hidden_states, residual = model_forward_maybe_tbo(
            layers=self.layers[normal_end_layer:self.end_layer],
            enable_tbo=True, ...
        )
```

### 5.7 流水线式交错执行

**核心**：`execute_overlapped_operations()`（`operations.py`）

```python
def execute_overlapped_operations(inputs_arr, operations_arr, delta_stages):
    inputs_a, inputs_b = inputs_arr
    delta_stage = delta_stages[1]  # delta_stages[0] 始终为 0

    stages_a = _convert_operations_to_stages(operations_a)
    stages_b = _convert_operations_to_stages(operations_b)
    executor_a = _StageExecutor("a", stages_a, inputs=inputs_a)
    executor_b = _StageExecutor("b", stages_b, inputs=inputs_b)

    # Phase 1: A 先行 delta_stage 步（B 尚未开始）
    for _ in range(delta_stage):
        executor_a.next()

    # Phase 2: A 和 B 交替执行（核心 overlap 阶段）
    for _ in range(executor_a.num_stages - delta_stage):
        executor_a.next()
        executor_b.next()

    # Phase 3: B 完成剩余 delta_stage 步
    for _ in range(delta_stage):
        executor_b.next()

    return [executor_a.output, executor_b.output]
```

**Decode 模式可视化**（`delta_stages = 2`，共 6 个 stage）：

```text
时间 →

Phase 1 (A 先行):
  A: [Stage0: comm_prep + attn_prep] [Stage1: attn_core + gate + select]

Phase 2 (交错执行):
  A: [Stage2: dispatch_a + shared]   [Stage3: dispatch_b + experts + combine_a]   [Stage4: combine_b]   [Stage5: output]
  B: [Stage0: comm_prep + attn_prep] [Stage1: attn_core + gate + select]          [Stage2: dispatch_a]  [Stage3: dispatch_b + experts]

Phase 3 (B 收尾):
  B: [Stage4: combine_b]   [Stage5: output + postprocess]
```

关键效果：**A 的 dispatch 通信（Stage2-3）与 B 的 attention 计算（Stage0-1）在时间上重叠**，反之亦然。

### 5.8 TBO Attention Backend

**源码**：`python/sglang/srt/layers/attention/tbo_backend.py`

```python
class TboAttnBackend(AttentionBackend):
    def __init__(self, primary, children: List[AttentionBackend]):
        self.primary = primary       # 原始 attention backend（用于完整 batch）
        self.children = children     # 2 个子 attention backend（对应两个子批次）

    @classmethod
    def init_new(cls, creator):
        return cls(
            primary=creator(),
            children=[creator() for _ in range(2)],
        )

    def init_forward_metadata(self, forward_batch):
        # Primary 处理完整 batch
        self.primary.init_forward_metadata(forward_batch=forward_batch)
        # 每个 child 处理对应的子 batch
        if forward_batch.tbo_children is not None:
            for child, fb_child in zip(self.children, forward_batch.tbo_children):
                if fb_child.batch_size > 0:
                    child.init_forward_metadata(forward_batch=fb_child)
```

### 5.9 TboForwardBatchPreparer

**源码**：`two_batch_overlap.py` 中的 `TboForwardBatchPreparer`

负责将 `ForwardBatch` 拆分为两个子 `ForwardBatch`（`two_batch_overlap.py`）：

```python
class TboForwardBatchPreparer:
    @classmethod
    def prepare_raw(cls, batch, tbo_children_num_token_non_padded):
        tbo_split_token_index = cls._compute_split_token_index(batch)

        # 创建子批次 A: token [0, split)，seq [0, split_seq)
        child_a = cls.filter_batch(batch,
            start_token_index=0, end_token_index=tbo_split_token_index,
            start_seq_index=0, end_seq_index=batch.tbo_split_seq_index,
            output_attn_backend=attn_backend_child_a, ...)

        # 创建子批次 B: token [split, end)，seq [split_seq, batch_size)
        child_b = cls.filter_batch(batch,
            start_token_index=tbo_split_token_index, end_token_index=batch.input_ids.shape[0],
            start_seq_index=batch.tbo_split_seq_index, end_seq_index=batch.batch_size,
            output_attn_backend=attn_backend_child_b, ...)

        # Two-chunk 模式：处理被拆分到两侧的序列
        if is_enable_two_chunk:
            cls.derive_fields_related_to_seq_len_for_two_chunk(batch, child_a, child_b, ...)

        batch.tbo_children = [child_a, child_b]
```

`filter_batch` 方法对 `ForwardBatch` 的每个字段进行切片：

- **Token 维度字段**（`input_ids`, `positions`, `out_cache_loc`）→ `[start_token:end_token]`
- **Sequence 维度字段**（`req_pool_indices`, `seq_lens`, `extend_seq_lens` 等）→ `[start_seq:end_seq]`
- **全局字段**（`forward_mode`, `return_logprob` 等）→ 直接复制引用
- **TBO 专属字段**：设置 `tbo_parent_token_range`、`tbo_padded_len`

每个子 batch 的 `tbo_padded_len` 对齐到 `attention_tp_size` 的倍数：

```python
output_dict["tbo_padded_len"] = (
    (end_token_index - start_token_index - 1) // attention_tp_size + 1
) * attention_tp_size
```

**Two-chunk 拆分细节**（当一个序列被拆分到两个子批次时）：

```python
def derive_fields_related_to_seq_len_for_two_chunk(batch, child_a, child_b, tbo_split_seq_index):
    half = overall_sum // 2
    left_last_seq_tokens = half - sum(extend_lens[:split_seq_index])
    right_first_seq_tokens = extend_lens[split_seq_index] - left_last_seq_tokens

    child_a.extend_seq_lens_cpu[-1] = left_last_seq_tokens     # A 的最后一个序列截断
    child_b.extend_seq_lens_cpu[0] = right_first_seq_tokens    # B 的第一个序列取剩余部分
    # 相应更新 seq_lens, extend_prefix_lens 等
```

### 5.10 TBO 与 CUDA Graph

**源码**：`two_batch_overlap.py` 中的 `TboCudaGraphRunnerPlugin`

```python
class TboCudaGraphRunnerPlugin:
    def capture_one_batch_size(self, batch, num_tokens):
        """CUDA Graph capture 时预计算 TBO 拆分信息"""
        batch.tbo_split_seq_index = compute_split_seq_index(...)
        self._tbo_children_num_token_non_padded[...] = (
            TboForwardBatchPreparer.compute_tbo_children_num_token_non_padded(batch)
        )
        TboForwardBatchPreparer.prepare_raw(batch, ...)

    def replay_prepare(self, forward_mode, bs, num_token_non_padded, spec_info):
        """CUDA Graph replay 时更新拆分参数"""
        tbo_split_seq_index, tbo_split_token_index = (
            compute_split_indices_for_cuda_graph_replay(...)
        )
        self._tbo_children_num_token_non_padded[...] = (
            TboForwardBatchPreparer.compute_tbo_children_num_token_non_padded_raw(...)
        )
```

在 `cuda_graph_runner.py` 中：

- 当 `forward_batch.can_run_tbo = True` 时，不使用 CUDA Graph 直接 replay，改用 TBO 路径
- CUDA Graph capture 的 batch size 会翻倍（`cuda_graph_runner.py`），因为需要为拆分后的两个子 batch 预留空间

### 5.11 TBO DP Attention 支持

**源码**：`two_batch_overlap.py` 中的 `TboDPAttentionPreparer`

当使用 Data Parallelism (DP) Attention 时，需要**所有 DP rank 对 TBO 决策达成一致**：

```python
class TboDPAttentionPreparer:
    def prepare_all_gather(self, local_batch):
        # 1. 计算本地 TBO split index
        self.local_tbo_split_seq_index = compute_split_seq_index(...)

        # 2. 判断本地是否可以运行 TBO
        local_can_run_tbo = (self.local_tbo_split_seq_index is not None) and not (
            # EXTEND + low_latency DeepEP 不支持 TBO
            local_batch.forward_mode.is_extend() and enable_a2a_moe
            and resolved_deepep_mode.is_low_latency()
        )
        return local_can_run_tbo, local_forward_mode

    def compute_output(self, partial_global_info):
        # 3. 聚合所有 rank 的决策
        local_can_run_tbo_aggregated = min(cpu_data[:, 0].tolist())  # 所有 rank 都同意才启用
        forward_mode_agree = self._is_all_same(forward_modes)         # 所有 rank 模式一致
        can_run_tbo = (self.enable_two_batch_overlap
                       and local_can_run_tbo_aggregated
                       and forward_mode_agree)
```

### 5.12 MaybeTboDeepEPDispatcher

**源码**：`two_batch_overlap.py`

当 TBO 启用时，每个子批次需要**独立的 dispatcher 实例**（避免通信 buffer 冲突）：

```python
class MaybeTboDeepEPDispatcher(BaseDispatcher):
    def __init__(self, **kwargs):
        num_inner_dispatchers = 2 if is_tbo_enabled() else 1
        if get_moe_a2a_backend().is_deepep():
            self._inners = [DeepEPDispatcher(**kwargs) for _ in range(num_inner_dispatchers)]
        elif get_moe_a2a_backend().is_mooncake():
            self._inners = [MooncakeEPDispatcher(**kwargs) for _ in range(num_inner_dispatchers)]
        elif get_moe_a2a_backend().is_mori():
            self._inners = [MoriEPDispatcher(instance_id=i, **kwargs) for i in range(num_inner_dispatchers)]
        elif get_moe_a2a_backend().is_nixl():
            self._inners = [NixlEPDispatcher(**kwargs) for _ in range(num_inner_dispatchers)]

    def _execute(self, name, tbo_subbatch_index=None, **kwargs):
        """通过 tbo_subbatch_index 路由到对应的 dispatcher"""
        return getattr(self._inners[tbo_subbatch_index or 0], name)(**kwargs)

    # dispatch/combine 用 tbo_subbatch_index 路由到对应的 dispatcher
    def dispatch(self, **kwargs): return self._execute("dispatch", **kwargs)
    def dispatch_a(self, **kwargs): return self._execute("dispatch_a", **kwargs)
    def dispatch_b(self, **kwargs): return self._execute("dispatch_b", **kwargs)
    def combine(self, **kwargs): return self._execute("combine", **kwargs)
    def combine_a(self, **kwargs): return self._execute("combine_a", **kwargs)
    def combine_b(self, **kwargs): return self._execute("combine_b", **kwargs)
```

支持的后端：**DeepEP**、**Mooncake**、**MoriEP**、**Nixl**

### 5.13 ForwardBatch 中的 TBO 字段

```python
# forward_batch_info.py
@dataclass
class ForwardBatch:
    tbo_split_seq_index: Optional[int] = None            # sequence 拆分点
    tbo_parent_token_range: Optional[Tuple[int,int]] = None  # 子 batch 在父 batch 中的 token 范围
    tbo_padded_len: Optional[int] = None                 # 对齐后的 token 长度
    tbo_children: Optional[List[ForwardBatch]] = None    # 两个子 batch

    @property
    def can_run_tbo(self):
        """是否可以执行 TBO（tbo_split_seq_index 非空即可运行）"""
        return self.tbo_split_seq_index is not None
```

---

## 六、SBO 与 TBO 对比

### 设计理念对比

| 维度                | SBO (Single Batch Overlap)                               | TBO (Two Batch Overlap)                           |
| ------------------- | -------------------------------------------------------- | ------------------------------------------------- |
| **核心思路**        | 同一 batch 内，通过双 stream + SM 分区实现通信与计算并行 | 将 batch 拆为两个 micro-batch，交错执行以隐藏通信 |
| **作用范围**        | 单个批次内的单个 MoE 层                                  | 跨所有 MoE 层的两个子批次                         |
| **Overlap 对象**    | Combine 通信 ↔ Down GEMM 或 Shared Experts               | 子批次 A 的通信 ↔ 子批次 B 的计算                 |
| **拆分粒度**        | 不拆分 batch，在 CUDA kernel 级别分区                    | 在 sequence 级别拆分为两个子 batch                |
| **并行方式**        | 双 CUDA Stream + SM 分区                                 | 单 CUDA Stream，stage 级别交错                    |
| **同步机制**        | CUDA Event + Signal Tensor（细粒度 block 级）            | stage-by-stage 交替（隐式同步）                   |
| **适用模型**        | DeepEP MoE（DeepSeek V2/V3）                             | DeepSeek, Qwen3 MoE, MiMo V2 等                   |
| **启动参数**        | `--enable-single-batch-overlap`                          | `--enable-two-batch-overlap`                      |
| **硬件要求**        | 需要 DeepEP low-latency 后端                             | 需要 EP > 1 的 MoE 模型                           |
| **CUDA Graph 兼容** | 透明兼容                                                 | 需要特殊处理（batch size 翻倍、Plugin 支持）      |
| **可组合性**        | 可在 TBO 的每个子批次内部使用                            | TBO 使用 op\_\* 方法，其中可包含 SBO 逻辑         |

### 性能特征对比

| 维度                | SBO                                | TBO                                                     |
| ------------------- | ---------------------------------- | ------------------------------------------------------- |
| **通信隐藏范围**    | 部分隐藏（仅 combine 阶段）        | 大范围隐藏（dispatch + combine 都可与计算重叠）         |
| **额外内存开销**    | 极低（仅 signal tensor）           | 较高（双份 attention metadata、双 dispatcher、padding） |
| **计算效率影响**    | SM 分区导致计算和通信都用更少的 SM | 子 batch 更小，kernel 效率可能降低                      |
| **适用 batch size** | 任意大小                           | 需要足够大的 batch（太小会导致子 batch 效率低下）       |

### 执行模型对比

**SBO 执行模型（双 Stream）**：

```text
Main Stream:   [Dispatch] [Expert GEMM] [Down-Proj GEMM (N-k SMs)]
Alt Stream:                             [Combine A2A (k SMs)      ]
                                         ← Signal 同步 →
```

**TBO 执行模型（单 Stream 交错）**：

```text
Single Stream: [A:Attn] [B:Attn][A:Dispatch] [B:Attn_Core][A:Expert] [B:Dispatch][A:Combine] [B:Expert]...
```

### 组合使用

SBO 和 TBO 可以**同时启用**，此时：

- TBO 负责在两个 micro-batch 之间交错 attention 和 MoE 操作
- SBO 负责在每个 micro-batch 的 MoE 层内部，进一步重叠 combine 和 down-projection
- `operations_strategy.py` 定义的 `op_*` 方法内部会触发 `_post_dispatch_hook` → `compute_overlap_args()` → SBO 逻辑

```bash
# 启动示例（同时开启 TBO + SBO）
python -m sglang.launch_server \
    --model deepseek-ai/DeepSeek-V3 \
    --enable-two-batch-overlap \
    --enable-single-batch-overlap \
    --moe-a2a-backend deepep
```

### 限制和约束

**SBO 限制**：

- 需要特定的 MoE runner backend（`flashinfer_cutedsl` 或 `deep_gemm`）
- Blackwell GPU 上部分策略不同（SM 分配更多给通信）
- SM 分区策略是静态的，无法动态适应负载

**TBO 限制**：

- 需要 MoE A2A backend（Expert Parallelism）
- 仅支持 Sparse MoE 层（Dense 层不使用 TBO）
- Prefill + Low-Latency DeepEP 模式下不支持 TBO
- 需要所有 DP rank forward_mode 一致
- batch size 太小时可能反而降低性能

---

## 七、关键源码文件索引

### Batch Overlap 核心包

| 文件                                                      | 说明                                                                                         |
| --------------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| `python/sglang/srt/batch_overlap/single_batch_overlap.py` | SBO 实现：SboFlags、CombineOverlapArgs、DownGemmOverlapArgs、compute_overlap_args（SM 分区） |
| `python/sglang/srt/batch_overlap/two_batch_overlap.py`    | TBO 实现：batch 拆分、合并、model_forward_maybe_tbo、MaybeTboDeepEPDispatcher、DP 协调       |
| `python/sglang/srt/batch_overlap/operations.py`           | 操作执行引擎：YieldOperation、execute_overlapped_operations、\_StageExecutor                 |
| `python/sglang/srt/batch_overlap/operations_strategy.py`  | 操作策略：DeepSeek/Qwen3/MiMo 的 prefill/decode 策略与 delta_stages 配置                     |

### Scheduler / Manager 层

| 文件                                                    | 说明                                                        |
| ------------------------------------------------------- | ----------------------------------------------------------- |
| `python/sglang/srt/managers/scheduler.py`               | Scheduler 主体：event_loop_overlap、run_batch、init_overlap |
| `python/sglang/srt/managers/overlap_utils.py`           | FutureMap：循环缓冲区管理 future token IDs                  |
| `python/sglang/srt/managers/scheduler_dp_attn_mixin.py` | DP Attention TBO 协调                                       |
| `python/sglang/srt/managers/tp_worker.py`               | TpModelWorker：forward_batch_generation、延迟采样           |

### Model Executor 层

| 文件                                                     | 说明                                                                        |
| -------------------------------------------------------- | --------------------------------------------------------------------------- |
| `python/sglang/srt/model_executor/forward_batch_info.py` | ForwardBatch：TBO 字段（tbo_split_seq_index、tbo_children、can_run_tbo 等） |
| `python/sglang/srt/model_executor/model_runner.py`       | ModelRunner：forward_stream 创建、TboAttnBackend 初始化                     |
| `python/sglang/srt/model_executor/cuda_graph_runner.py`  | CudaGraphRunner：TBO CUDA Graph 支持                                        |

### Attention / MoE 层

| 文件                                                      | 说明                                                             |
| --------------------------------------------------------- | ---------------------------------------------------------------- |
| `python/sglang/srt/layers/attention/tbo_backend.py`       | TboAttnBackend：primary + 2 children attention backend           |
| `python/sglang/srt/layers/moe/utils.py`                   | is_tbo_enabled()、is_sbo_enabled() 全局状态                      |
| `python/sglang/srt/layers/moe/token_dispatcher/deepep.py` | DeepEP Dispatcher：combine_a/combine_b 拆分，SBO alt_stream 使用 |
| `python/sglang/srt/layers/moe/token_dispatcher/moriep.py` | MoriEP Dispatcher：SBO overlap args 使用                         |
| `python/sglang/srt/layers/moe/token_dispatcher/base.py`   | BaseDispatcher：set_overlap_args / clear_overlap_args            |
| `python/sglang/srt/layers/moe/moe_runner/deep_gemm.py`    | Down GEMM 中使用 DownGemmOverlapArgs 限制 SM 并传递 signal       |
| `python/sglang/srt/layers/moe/moe_runner/runner.py`       | MoeRunner：持有并传递 overlap args                               |

### 模型文件

| 文件                                        | 说明                                                                               |
| ------------------------------------------- | ---------------------------------------------------------------------------------- |
| `python/sglang/srt/models/deepseek_v2.py`   | DeepSeek V2/V3：alt*stream 创建、forward_deepep SBO hook、op*\* 操作定义、TBO 入口 |
| `python/sglang/srt/models/qwen2_moe.py`     | Qwen3 MoE：TBO 支持                                                                |
| `python/sglang/srt/models/mimo_v2_flash.py` | MiMo V2：TBO 支持                                                                  |
| `python/sglang/srt/models/minimax_m2.py`    | MiniMax M2：TBO 支持                                                               |
| `python/sglang/srt/models/glm4_moe.py`      | GLM4 MoE：TBO 支持                                                                 |

### 配置和测试

| 文件                                                     | 说明                                          |
| -------------------------------------------------------- | --------------------------------------------- |
| `python/sglang/srt/server_args.py`                       | 所有 overlap 相关的 ServerArgs 定义和验证逻辑 |
| `docs/advanced_features/expert_parallelism.md`           | TBO 和 SBO 的官方文档                         |
| `test/manual/test_two_batch_overlap.py`                  | TBO 手动测试                                  |
| `test/registered/scheduler/test_no_overlap_scheduler.py` | 禁用 overlap 的回归测试                       |

### 全部 Overlap 相关配置汇总

| 配置项                                                | 类型       | 默认值                 | 说明                                        |
| ----------------------------------------------------- | ---------- | ---------------------- | ------------------------------------------- |
| `--disable-overlap-schedule`                          | Server arg | `False`                | 禁用 CPU-GPU overlap scheduler              |
| `--enable-two-batch-overlap`                          | Server arg | `False`                | 启用 TBO                                    |
| `--enable-single-batch-overlap`                       | Server arg | `False`                | 启用 SBO                                    |
| `--tbo-token-distribution-threshold`                  | Server arg | `0.48`                 | TBO token 分布阈值（控制 two-chunk split）  |
| `SGLANG_DISABLE_CONSECUTIVE_PREFILL_OVERLAP`          | 环境变量   | `false`                | 禁用连续 prefill batch 的 overlap           |
| `SGLANG_DEEPEP_LL_COMBINE_SEND_NUM_SMS`               | 环境变量   | Blackwell: 32, 其他: 3 | SBO combine 通信 SM 数量                    |
| `SGLANG_BLACKWELL_OVERLAP_SHARED_EXPERTS_OUTSIDE_SBO` | 环境变量   | `false`                | Blackwell 上 shared experts 放到备用 stream |
| `SGLANG_TBO_DEBUG`                                    | 环境变量   | `false`                | TBO 调试模式                                |
| `SGLANG_OPERATIONS_ENABLE_PROFILE`                    | 环境变量   | `0`                    | 启用 NVTX profiling                         |
