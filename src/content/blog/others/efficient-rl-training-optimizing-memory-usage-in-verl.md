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

## 3. The Memory Challenge

RL training in [verl](https://github.com/volcengine/verl) requires seamless transitions between rollout and training phases, both of which are memory-intensive. Co-locating these phases on the same GPUs risks out-of-memory (OOM) errors, especially with large models. Below is the memory breakdown for **[Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)** on an H200 GPU node (8 GPUs, \~141 GB VRAM each) using [FSDP](https://pytorch.org/docs/stable/fsdp.html) for training and [SGLang](https://github.com/sgl-project/sglang) for rollout.

#### Training Phase Memory Breakdown

With [FSDP](https://pytorch.org/docs/stable/fsdp.html) sharding across 8 GPUs, and enable FULLY SHARDED mode with Full Activation Checkpointing, each GPU holds:

![training phase memory breakdown](/images/others/efficient-rl-training-optimizing-memory-usage-in-verl/005-86e69d17.png)

**Peak Training Memory**: \~48 GB per GPU

#### Rollout Phase Memory Breakdown

During inference, the full model is typically loaded (not sharded):

- **Model Weights**: \~15.4 GB (full model for inference efficiency)
- **KV Cache**: \~60-90 GB (dominant factor, can be tuned by `mem-fraction` in SGLang, assuming `0.7-0.9` ratio)
- **CUDA Graph**: \~1-3 GB (captures computation graph for inference acceleration)
- **Input/Output Buffers**: \~3-7 GB (request batching and response generation)

**Total Rollout Memory**: \~80-115 GB per GPU

Managing these memory demands on the same GPUs requires careful optimization to avoid OOM errors during phase transitions.

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

#### Resuming Tensors

![恢复张量](/images/others/efficient-rl-training-optimizing-memory-usage-in-verl/010-098c4127.png)

1. 使用 `cuMemCreate` 创建一个新的物理内存句柄。
1. 使用 `cuMemAlloc` 分配物理内存。
1. 使用 `cuMemMap` 将新的物理内存映射到存储的虚拟地址。
1. 更新 **元数据映射** 以包含新的句柄。

到目前为止，我们针对内存挑战已经有了一个相当不错的解决方案。

![v2：休眠推理引擎](/images/others/efficient-rl-training-optimizing-memory-usage-in-verl/011-a8399ff5.png)

#### 权重加载优化

为了解决权重加载缓慢的问题，我们避免了磁盘序列化。相反，我们将训练模型权重加载到GPU上，并通过CUDA进程间通信更新rollout引擎的权重。这显著减少了训练到rollout切换的时间（例如，对于7B模型，时间小于0.5秒）。

### 4.4：多阶段唤醒

尽管有这些改进，我们的用户报告在使用更大模型或高KV缓存比率（>0.7）时，在训练-rollout切换期间出现内存不足（OOM）错误。我们发现在恢复过程中存在内存浪费（上图中的红色块）。为了优化，我们将恢复过程分为多个阶段：

1. 将训练模型权重加载到GPU上。
1. 恢复推理模型权重。
1. 同步权重。
1. 卸载训练模型。
1. 为rollout恢复KV缓存。

最初，`torch_memory_saver` 的单例设计不支持选择性暂停/恢复内存区域。我们探索了两种解决方案：

- 多个 `torch_memory_saver` 实例。
- 基于标签的暂停/恢复API。

我们选择了基于标签的方法，以最小化对SGLang代码库的更改，因为SGLang严重依赖单例设计。您可以在 [RFC](https://github.com/sgl-project/sglang/issues/7009) 中找到两种实现的详细信息。

#### 基于标签的内存管理

我们向张量元数据添加了一个标签参数，实现了选择性暂停/恢复。

![基于标签的恢复](/images/others/efficient-rl-training-optimizing-memory-usage-in-verl/012-14e0c9d9.png)

**暂停过程：**

1. 检查每个张量的元数据以匹配标签。
1. 如果匹配，使用 `cuMemUnmap` 取消映射内存。
1. 使用 `cuMemRelease` 释放物理内存。

**新接口：**

```python
import torch_memory_saver

memory_saver = torch_memory_saver.torch_memory_saver

# Create tensors with specific tags
with torch_memory_saver.region(tag="weights"):
    tensor1 = torch.full((5_000_000_000,), 100, dtype=torch.uint8, device='cuda')

with torch_memory_saver.region(tag="kv_cache"):
    tensor2 = torch.full((5_000_000_000,), 100, dtype=torch.uint8, device='cuda')

# Pause and resume selectively
torch_memory_saver.pause("weights")
torch_memory_saver.pause("kv_cache")

torch_memory_saver.resume("weights")
# Sync weights and offload training model
torch_memory_saver.resume("kv_cache")
```

**多阶段恢复过程：**

![v3：多阶段恢复](/images/others/efficient-rl-training-optimizing-memory-usage-in-verl/013-33b532fa.png)

这种方法最小化了内存浪费，解决了OOM问题，并提高了大模型和高KV缓存比率的效率。

## 5. 结论

通过本旅程中概述的优化，我们成功地在 **Qwen 32B** 上训练了模型，KV缓存内存比率为 **0.9**，使用了 **8个H200 GPU**——这一成就最初是无法实现的。这篇博客文章总结了SGLang RL团队的内存优化努力，为强化学习（RL）训练的高效内存管理提供了见解。我们希望它能作为理解和应对类似挑战的宝贵资源。

## 6. 致谢

我们向SGLang RL团队和verl团队表示诚挚的感谢，特别感谢 [Tom](https://github.com/fzyzcjy) 开发了紧凑而强大的 `torch_memory_saver` 库，并为SGLang和 [Chenyang](https://www.linkedin.com/in/chayennezhao/) 领导SGLang RL计划并提供关键指导和支持奠定了基础。

## 7. 脚注

1. [LlamaRL论文](https://arxiv.org/pdf/2505.24034) [↩](#fnref:1)

1. [Torch Memory Saver：一个PyTorch库，允许张量内存被临时释放并在稍后恢复](https://github.com/fzyzcjy/torch_memory_saver) [↩](#fnref:2)

1. [CUDA 10.2：引入低级GPU虚拟内存管理](https://developer.nvidia.com/blog/introducing-low-level-gpu-virtual-memory-management/) [↩](#fnref:3) [↩2](#fnref:3:1)

1. [LD_PRELOAD](https://catonmat.net/simple-ld-preload-tutorial) [↩](#fnref:4)
