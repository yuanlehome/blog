---
title: Efficient RL Training - Optimizing Memory Usage in verl
slug: efficient-rl-training-optimizing-memory-usage-in-verl
date: '2026-02-05'
tags: []
status: published
source_url: 'https://hebiao064.github.io/rl-memory-management'
source_author: hebiao064.github.io
imported_at: '2026-02-05T18:38:50.057Z'
source:
  title: hebiao064.github.io
  url: 'https://hebiao064.github.io/rl-memory-management'
cover: >-
  /images/others/efficient-rl-training-optimizing-memory-usage-in-verl/004-7d38101b.png
lang: zh
translatedFrom: en
---

# Efficient RL Training - Optimizing Memory Usage in verl

## 1. Introduction

[Reinforcement learning](https://en.wikipedia.org/wiki/Reinforcement_learning) (RL) for large language models (LLMs) presents unique challenges due to its integration of inference and training in each step, demanding significant scalability and resource efficiency. The [verl](https://github.com/volcengine/verl) library, designed for RL training of LLMs, combines advanced training strategies like Fully Sharded Data Parallel ([FSDP](https://pytorch.org/docs/stable/fsdp.html)) and [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) with inference engines such as [SGLang](https://github.com/sgl-project/sglang) for efficient rollout generation. This blog post details the SGLang RL team’s efforts to optimize memory usage in [verl](https://github.com/volcengine/verl), focusing on techniques that reduce peak memory demands and enable training larger models on limited GPU resources.

## 2. High-Level RL Training Workflow

![An example flow of online RL training](/images/others/efficient-rl-training-optimizing-memory-usage-in-verl/004-7d38101b.png)

The diagram above illustrates the **online RL training** proces, simplified by omitting the reference and critic models and assuming a basic reward function (common in code and reasoning tasks) instead of a reward model. The policy model exists in two instances: one optimized for training (using [FSDP](https://pytorch.org/docs/stable/fsdp.html) or [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)) and another for inference (using [SGLang](https://github.com/sgl-project/sglang) or [vLLM](https://github.com/vllm-project/vllm)).[1](#fn:1)

#### Simplified PPO Example

Below is a simplified implementation using [Proximal Policy Optimization](https://en.wikipedia.org/wiki/Proximal_policy_optimization) (PPO):

```text
for prompts, pretrain_batch in dataloader:
    # Stage 1: Rollout generation (inference)
    batch = actor.generate_sequences(prompts)
    # Stage 2: Prepare experience
    batch = reference.compute_log_prob(batch)
    batch = reward.compute_reward(batch)  # Reward function or model
    batch = compute_advantages(batch, algo_type)
    # Stage 3: Actor training
    actor_metrics = actor.update_actor(batch)
```

Each iteration involves a rollout (inference) phase using the actor model, followed by training. [verl](https://github.com/volcengine/verl)’s design co-locates both the rollout and training versions of the actor model on the same GPUs, optimizing resource sharing but complicating memory management. This post focuses on addressing the actor model’s memory challenges.

## 3. 内存挑战

verl 中的 RL 训练需要在 rollout 和训练阶段之间无缝切换，这两个阶段都是内存密集型的。在同一 GPU 上共同放置这些阶段会面临内存不足（OOM）错误的风险，尤其是对于大型模型。以下是在 H200 GPU 节点（8 个 GPU，每个约 141 GB VRAM）上使用 FSDP 进行训练和 SGLang 进行 rollout 的 **Qwen2.5-7B-Instruct** 的内存分解。

### 训练阶段内存分解

使用 FSDP 在 8 个 GPU 上进行分片，并启用完全分片模式和完全激活检查点，每个 GPU 持有：

![训练阶段内存分解](/images/others/efficient-rl-training-optimizing-memory-usage-in-verl/005-86e69d17.png)

**训练峰值内存**：每个 GPU 约 48 GB

### Rollout 阶段内存分解

在推理期间，通常加载完整模型（不分片）：

- **模型权重**：约 15.4 GB（为推理效率而加载的完整模型）
- **KV 缓存**：约 60-90 GB（主导因素，可通过 SGLang 中的 `mem-fraction` 调整，假设比率为 0.7-0.9）
- **CUDA Graph**：约 1-3 GB（捕获计算图以加速推理）
- **输入/输出缓冲区**：约 3-7 GB（请求批处理和响应生成）

**Rollout 总内存**：每个 GPU 约 80-115 GB

在同一 GPU 上管理这些内存需求需要仔细优化，以避免阶段转换期间的 OOM 错误。

## 4. Memory Optimization Journey

### 4.1: The Naive Approach

In our initial approach, we kept both training model weights and the inference engine ([SGLang](https://github.com/sgl-project/sglang)) in GPU memory without offloading.

![v0: The Naive Approach](/images/others/efficient-rl-training-optimizing-memory-usage-in-verl/006-82fec675.png)

However, [SGLang](https://github.com/sgl-project/sglang)’s significant memory footprint made it impossible to start training. This was a conceptual baseline and was never implemented.

### 4.2: Offloading Weights to CPU and Relaunch Inference Engine

To address this, we offloaded training model weights to CPU after training, serializing them to disk. During the rollout phase, we relaunched the [SGLang](https://github.com/sgl-project/sglang) engine, loading weights from disk.

![v1: Offloading Weights to CPU and Relaunch Inference Engine](/images/others/efficient-rl-training-optimizing-memory-usage-in-verl/007-bf1d84c5.png)

This reduced GPU memory usage during rollout but introduced significant delays:

- Slow Disk I/O: Loading weights from disk was time-consuming.
- Recapture CUDA Graph: Recapturing CUDA Graphs added overhead.

While this was an improvement, it was too slow for practical use.

### 4.3: Sleeping the Inference Engine

We explored keeping the [CUDA Graph](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-graph) alive while freeing weights and KV cache memory during training. The challenge was that recreating these tensors broke [CUDA Graph replay](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-graph-replay) due to changes in virtual memory addresses.

Hence the goal can be rephrased to:

- Free physical memory during training to allocate space.
- Reallocate GPU memory for weights and KV cache at the same virtual memory addresses during rollout.

The SGLang RL team (credit to [Tom](https://github.com/fzyzcjy)) developed the [torch_memory_saver](https://github.com/fzyzcjy/torch_memory_saver) library [2](#fn:2), enabling memory pausing and resuming while preserving CUDA Graph compatibility.

Here’s how it works:

```python
import torch_memory_saver

memory_saver = torch_memory_saver.torch_memory_saver

# Create tensors in a pausable region
with memory_saver.region():
    pauseable_tensor = torch.full((1_000_000_000,), 100, dtype=torch.uint8, device='cuda')

# Pause to free CUDA memory
memory_saver.pause()

# Resume to reallocate memory at the same virtual address
memory_saver.resume()
```

#### Implementation Using CUDA Virtual Memory APIs

Before **CUDA 10.2**, memory management relied on `cudaMalloc`, `cudaFree`, and `cudaMemcpy`, which lacked control over virtual memory addresses. **CUDA 10.2**[3](#fn:3) introduced APIs for fine-grained virtual memory management:

- `cuMemCreate`: Creates a physical memory handle.
- `cuMemAddressReserve`: Reserves a virtual address range.
- `cuMemMap`: Maps a physical memory handle to a virtual address range.

These APIs enabled a custom memory allocator to preserve virtual memory addresses. And in [SGLang](https://github.com/sgl-project/sglang) and [verl](https://github.com/volcengine/verl) system, we utilized `LD_PRELOAD` [4](#fn:4) to replace the default cuda malloc and free with our custom allocator.

#### Modified CUDA Malloc

![cuda malloc](/images/others/efficient-rl-training-optimizing-memory-usage-in-verl/008-80b3ab26.png)

1. Create a `CUmemGenericAllocationHandle` and allocate physical memory with `cuMemCreate`, the handler contains the properties of the memory to allocate, like where is this memory physically located or what kind of shareable handles should be available. [3](#fn:3)
1. Reserve a virtual address range using `cuMemAddressReserve`.
1. Map the physical memory to the virtual address using `cuMemMap`.
1. Store the virtual memory pointer and physical memory handle in a **Metadata Map**.

#### Pausing Tensors

![pause tensor](/images/others/efficient-rl-training-optimizing-memory-usage-in-verl/009-5abe350c.png)

1. Unmap memory from the virtual address range using `cuMemUnmap`
1. Retrieve the physical memory handle from the **Metadata Map** and free it with `cuMemRelease`.

This releases physical memory while retaining virtual addresses.

#### 恢复张量

![恢复张量](/images/others/efficient-rl-training-optimizing-memory-usage-in-verl/010-098c4127.png)

1. 使用 `cuMemCreate` 创建一个新的物理内存句柄。
2. 使用 `cuMemAlloc` 分配物理内存。
3. 使用 `cuMemMap` 将新的物理内存映射到存储的虚拟地址。
4. 更新**元数据映射**以包含新的句柄。

到目前为止，我们针对内存挑战已经有了一个相当不错的解决方案。

![v2：休眠推理引擎](/images/others/efficient-rl-training-optimizing-memory-usage-in-verl/011-a8399ff5.png)

#### 权重加载优化

为了解决权重加载缓慢的问题，我们避免了磁盘序列化。相反，我们将训练模型权重加载到 GPU 上，并通过 CUDA 进程间通信更新 rollout 引擎的权重。这显著减少了训练到 rollout 切换的时间（例如，对于 7B 模型，时间小于 0.5 秒）。

### 4.4：多阶段唤醒

尽管有这些改进，我们的用户报告在使用更大模型或高 KV 缓存比率（>0.7）时，在训练-rollout 切换期间出现内存不足（OOM）错误。我们发现在恢复过程中存在内存浪费（上图中的红色块）。为了优化，我们将恢复过程分为多个阶段：

1. 将训练模型权重加载到 GPU 上。
2. 恢复推理模型权重。
3. 同步权重。
4. 卸载训练模型。
5. 为 rollout 恢复 KV 缓存。

最初，`torch_memory_saver` 的单例设计不支持选择性暂停/恢复内存区域。我们探索了两种解决方案：

- 多个 `torch_memory_saver` 实例。
- 基于标签的暂停/恢复 API。

我们选择了基于标签的方法，以最小化对 SGLang 代码库的更改，因为 SGLang 严重依赖单例设计。您可以在 RFC 中找到两种实现的详细信息。

#### 基于标签的内存管理

我们向张量元数据添加了一个标签参数，实现了选择性暂停/恢复。

![基于标签的恢复](/images/others/efficient-rl-training-optimizing-memory-usage-in-verl/012-14e0c9d9.png)

**暂停过程：**

1. 检查每个张量的元数据以匹配标签。
2. 如果匹配，使用 `cuMemUnmap` 取消映射内存。
3. 使用 `cuMemRelease` 释放物理内存。

**新接口：**

```python
import torch_memory_saver

memory_saver = torch_memory_saver.torch_memory_saver

# 使用特定标签创建张量
with torch_memory_saver.region(tag="weights"):
    tensor1 = torch.full((5_000_000_000,), 100, dtype=torch.uint8, device='cuda')

with torch_memory_saver.region(tag="kv_cache"):
    tensor2 = torch.full((5_000_000_000,), 100, dtype=torch.uint8, device='cuda')

# 选择性暂停和恢复
torch_memory_saver.pause("weights")
torch_memory_saver.pause("kv_cache")

torch_memory_saver.resume("weights")
# 同步权重并卸载训练模型
torch_memory_saver.resume("kv_cache")
```

**多阶段恢复过程：**

![v3：多阶段恢复](/images/others/efficient-rl-training-optimizing-memory-usage-in-verl/013-33b532fa.png)

这种方法最小化了内存浪费，解决了 OOM 问题，并提高了大模型和高 KV 缓存比率的效率。

## 5. 结论

通过本文概述的优化，我们成功地在 **Qwen 32B** 上训练了模型，KV 缓存内存比率为 **0.9**，使用了 **8 个 H200 GPU** - 这一成就最初是无法实现的。本文总结了 SGLang RL 团队的内存优化努力，为强化学习（RL）训练的高效内存管理提供了见解。我们希望它能作为理解和应对类似挑战的宝贵资源。

## 6. 致谢

我们向 SGLang RL 团队和 verl 团队表示诚挚的感谢，特别感谢 Tom 开发了紧凑而强大的 `torch_memory_saver` 库，并为 SGLang 奠定了基础，以及 Chenyang 领导 SGLang RL 计划并提供关键指导和支持。
