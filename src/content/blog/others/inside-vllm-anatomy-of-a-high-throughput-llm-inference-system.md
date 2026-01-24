---
title: 'Inside vLLM: Anatomy of a High-Throughput LLM Inference System'
slug: inside-vllm-anatomy-of-a-high-throughput-llm-inference-system
date: '2025-12-29'
tags: ['Source Code Analysis', 'Architecture Design']
status: published
source_url: 'https://www.aleksagordic.com/blog/vllm'
source_author: www.aleksagordic.com
imported_at: '2025-12-29T14:07:49.833Z'
source:
  title: www.aleksagordic.com
  url: 'https://www.aleksagordic.com/blog/vllm'
cover: >-
  /images/others/inside-vllm-anatomy-of-a-high-throughput-llm-inference-system/001-f5b11a75.png
lang: zh
translatedFrom: en
---

# 深入 vLLM：剖析一个高吞吐量 LLM 推理系统

## 从分页注意力（paged attention）、连续批处理（continuous batching）、前缀缓存（prefix caching）、推测解码（specdec）等，到多 GPU、多节点的动态大规模服务

2025年8月29日

在这篇文章中，我将逐步介绍构成现代高吞吐量 LLM 推理系统的所有核心系统组件和高级功能。特别是，我将详细剖析 vLLM [\[1\]](#ref-1) 的工作原理。

这是系列文章的第一篇。它从宏观开始，然后层层深入细节（采用倒金字塔方法），以便您能形成对整个系统准确的高层次心智模型，而不会淹没在细枝末节中。

后续文章将深入探讨特定子系统。

本文分为五个部分：

1. [LLM 引擎与引擎核心](#cpt1)：vLLM 的基础（调度、分页注意力（paged attention）、连续批处理（continuous batching）等）
1. [高级功能](#cpt2)：分块预填充（chunked prefill）、前缀缓存（prefix caching）、引导与推测解码（guided & speculative decoding）、解耦的 P/D（disaggregated P/D）
1. [扩展规模](#cpt3)：从单 GPU 到多 GPU 执行
1. [服务层](#cpt4)：分布式/并发网络框架
1. [基准测试与自动调优](#cpt5)：测量延迟和吞吐量

📝笔记

- 分析基于 [commit 42172ad](https://github.com/vllm-project/vllm/tree/42172ad)（2025年8月9日）。
- 目标受众：任何对最先进 LLM 引擎工作原理感到好奇的人，以及有兴趣为 vLLM、SGLang 等做出贡献的人。
- 我将专注于 [V1 引擎](https://docs.vllm.ai/en/latest/usage/v1_guide.html)。我还探索了 V0（[现已弃用](https://github.com/vllm-project/vllm/issues/18571)），这对于理解项目如何演变很有价值，许多概念仍然适用。
- 关于 LLM 引擎/引擎核心的第一部分可能有点令人不知所措/枯燥——但博客的其余部分有很多示例和视觉效果。 :)

## LLM 引擎与引擎核心

LLM 引擎是 vLLM 的基本构建块。仅凭它本身，它已经能够实现高吞吐量推理——但仅限于离线设置。您还不能通过网络将其服务提供给客户。

我们将使用以下离线推理代码片段作为我们的运行示例（改编自 [basic.py](https://github.com/vllm-project/vllm/blob/main/examples/offline_inference/basic/basic.py)）。

```python
from vllm import LLM, SamplingParams

prompts = [
    "Hello, my name is",
    "The president of the United States is",
]

sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

def main():
    llm = LLM(model="TinyLlama/TinyLlama-1.1B-Chat-v1.0")

    outputs = llm.generate(prompts, sampling_params)

if __name__ == "__main__":
    main()
```

📝环境变量：

- VLLM_USE_V1="1" # 我们正在使用引擎 V1
- VLLM_ENABLE_V1_MULTIPROCESSING="0" # 我们在单个进程中运行

此配置为：

- 离线（无网络/分布式系统框架）
- 同步（所有执行都在单个阻塞进程中发生）
- 单 GPU（无数据/模型/流水线/专家并行；DP/TP/PP/EP = 1）
- 使用标准 Transformer [\[2\]](#ref-2)（支持像 Jamba 这样的混合模型需要更复杂的混合 KV 缓存内存分配器）

从这里开始，我们将逐步构建一个在线、异步、多 GPU、多节点的推理系统——但仍然服务一个标准 Transformer。

在这个例子中，我们做两件事：

1. 实例化一个引擎
1. 调用 `generate` 来从给定的提示中采样

让我们开始分析构造函数。

## LLM 引擎构造函数

引擎的主要组件包括：

- vLLM 配置（包含所有用于配置模型、缓存、并行性等的旋钮）
- 处理器（将原始输入转换为 `EngineCoreRequests`，通过验证、分词和处理）
- 引擎核心客户端（在我们的运行示例中，我们使用 `InprocClient`，这基本上等于 `EngineCore`；我们将逐步构建到 `DPLBAsyncMPClient`，它允许大规模服务）
- 输出处理器（将原始 `EngineCoreOutputs` 转换为用户看到的 `RequestOutput`）

📝注意：

随着 V0 引擎被弃用，类名和细节可能会变化。我将强调核心思想而非确切的签名。我会抽象掉一些但不是所有这些细节。

引擎核心本身由几个子组件组成：

- 模型执行器（驱动模型的前向传递，我们目前处理的是 `UniProcExecutor`，它在单个 GPU 上有一个 `Worker` 进程）。我们将逐步构建到 `MultiProcExecutor`，它支持多个 GPU

- 结构化输出管理器（用于引导解码——我们稍后会介绍）

- 调度器（决定哪些请求进入下一个引擎步骤）——它进一步包含：
  1. 策略设置——可以是 **FCFS**（先到先服务）或 **priority**（优先级较高的请求优先服务）
  1. `waiting` 和 `running` 队列
  1. KV 缓存管理器——分页注意力（paged attention）的核心 [\[3\]](#ref-3)

KV 缓存管理器维护一个 `free_block_queue`——一个可用的 KV 缓存块池（通常数量在数十万级别，取决于 VRAM 大小和块大小）。在分页注意力（paged attention）期间，这些块作为索引结构，将令牌映射到它们计算的 KV 缓存块。

![LLM 引擎构造函数](/images/others/inside-vllm-anatomy-of-a-high-throughput-llm-inference-system/001-f5b11a75.png)

本节描述的核心组件及其关系

标准 Transformer 层（非 MLA [\[4\]](#ref-4)）的块大小计算如下：\
2（键/值） \* `block_size`（默认=16） \* `num_kv_heads` \* `head_size` \* `dtype_num_bytes`（例如，bf16 为 2）

在模型执行器构造期间，创建一个 `Worker` 对象，并执行三个关键过程。（稍后，使用 `MultiProcExecutor` 时，这些相同的过程在不同 GPU 上的每个工作进程中独立运行。）

1. 初始化设备：
   - 分配一个 CUDA 设备（例如 "cuda:0"）给工作进程，并检查模型数据类型是否受支持（例如 bf16）
   - 验证是否有足够的 VRAM，给定请求的 `gpu_memory_utilization`（例如 0.8 → 总 VRAM 的 80%）
   - 设置分布式设置（DP / TP / PP / EP 等）
   - 实例化一个 `model_runner`（持有采样器、KV 缓存和前向传递缓冲区，如 `input_ids`、`positions` 等）
   - 实例化一个 `InputBatch` 对象（持有 CPU 端前向传递缓冲区、KV 缓存索引的块表、采样元数据等）

1. 加载模型：
   - 实例化模型架构
   - 加载模型权重
   - 调用 model.eval()（PyTorch 的推理模式）
   - 可选：在模型上调用 torch.compile()

1. 初始化 KV 缓存
   - 获取每层 KV 缓存规范。历史上这总是 `FullAttentionSpec`（同质 Transformer），但使用混合模型（滑动窗口、像 Jamba 这样的 Transformer/SSM）时变得更复杂（见 Jenga [\[5\]](#ref-5)）
   - 运行一个虚拟/性能分析前向传递，并拍摄 GPU 内存快照，以计算有多少 KV 缓存块适合可用 VRAM
   - 分配、重塑并将 KV 缓存张量绑定到注意力层
   - 准备注意力元数据（例如，将后端设置为 FlashAttention），稍后在前向传递期间由内核使用
   - 除非 `--enforce-eager`如果提供了预热批次大小，对每个预热批次大小执行一次虚拟运行并捕获CUDA图。CUDA图将整个GPU工作序列记录到一个DAG中。之后在前向传递期间，我们启动/重放预烘焙的图，减少内核启动开销，从而改善延迟。

我在这里抽象了许多底层细节——但这些是我现在要介绍的核心部分，因为我将在后续章节中反复引用它们。

现在引擎已初始化，让我们继续到`generate`函数。

## 生成函数

第一步是验证并将请求输入引擎。对于每个提示，我们：

1. 创建唯一的请求ID并捕获其到达时间
1. 调用输入预处理器，对提示进行分词，并返回一个包含`prompt`、`prompt_token_ids`和`type`（文本、令牌、嵌入等）的字典
1. 将此信息打包到一个`EngineCoreRequest`中，添加优先级、采样参数和其他元数据
1. 将请求传递给引擎核心，核心将其包装在一个`Request`对象中，并将其状态设置为`WAITING`。然后此请求被添加到调度器的`waiting`队列中（如果是FCFS则追加，如果是优先级则堆推入）

此时引擎已接收输入，可以开始执行。在同步引擎示例中，这些初始提示是我们将处理的唯一请求——没有机制在运行中注入新请求。相比之下，异步引擎支持此功能（即**连续批处理** [\[6\]](#ref-6)）：在每一步之后，新旧请求都会被考虑。

由于前向传递将批次展平为单个序列，且自定义内核高效处理，连续批处理在同步引擎中本质上得到支持。

接下来，只要有请求要处理，引擎就重复调用其`step()`函数。每一步有三个阶段：

1. 调度：选择在此步骤中运行哪些请求（解码和/或（分块）预填充）
1. 前向传递：运行模型并采样令牌
1. 后处理：将采样的令牌ID附加到每个`Request`，进行反分词，并检查停止条件。如果请求完成，则清理（例如，将其KV缓存块返回到`free_block_queue`）并提前返回输出

📝停止条件包括：

- 请求超过其长度限制（`max_model_length`或其自身的`max_tokens`）
- 采样的令牌是EOS ID（除非`ignore_eos`启用——

  \>

  在基准测试中很有用，当我们想强制生成一定数量的输出令牌时）

- 采样的令牌匹配采样参数中指定的任何`stop_token_ids` specified in the sampling parameters
- 停止字符串出现在输出中——我们将输出截断到第一个停止字符串出现处，并在引擎中中止请求（注意`stop_token_ids`将出现在输出中，但停止字符串不会）。

![引擎循环](/images/others/inside-vllm-anatomy-of-a-high-throughput-llm-inference-system/002-f0c7c57b.png)

引擎循环

在流式模式下，我们会发送生成的中间令牌，但我们现在忽略这一点。

接下来，我们将更详细地检查调度。

## 调度器

推理引擎处理两种主要类型的工作负载：

1. **预填充**请求——对所有提示令牌的前向传递。这些通常是**计算受限**（阈值取决于硬件和提示长度）。最后，我们从最终令牌位置的概率分布中采样一个令牌。
1. **解码**请求——仅对最近令牌的前向传递。所有早期的KV向量已缓存。这些是**内存带宽受限**，因为我们仍需要加载所有LLM权重（和KV缓存）来计算一个令牌。

在[基准测试部分](#cpt5)，我们将分析GPU性能的所谓屋顶线模型。这将更详细地探讨预填充/解码性能概况。

V1调度器可以在同一步骤中混合两种类型的请求，这得益于更智能的设计选择。相比之下，V0引擎一次只能处理预填充或解码。

调度器优先处理解码请求——即那些已在`running`队列中的请求。对于每个此类请求，它：

1. 计算要生成的新令牌数量（不总是1，由于推测解码和异步调度——稍后详述）。
1. 调用KV缓存管理器的`allocate_slots`函数（详情如下）。
1. 通过减去步骤1中的令牌数量来更新令牌预算。

之后，它处理来自`waiting`队列的预填充请求，它：

1. 检索计算块的数量（如果前缀缓存禁用则返回0——稍后介绍）。
1. 调用KV缓存管理器的`allocate_slots`函数。
1. 将请求从等待队列弹出并移动到运行队列，将其状态设置为`RUNNING`。
1. 更新令牌预算。

现在让我们看看`allocate_slots`做什么，它：

1. **计算块数量**——确定需要分配多少新的KV缓存块（`n`）。每个块默认存储16个令牌。例如，如果预填充请求有17个新令牌，我们需要`ceil(17/16) = 2`个块。
1. **检查可用性**——如果管理器池中没有足够的块，则提前退出。根据是解码还是预填充请求，引擎可能尝试重新计算抢占（V0中支持交换抢占），通过驱逐低优先级请求（调用`kv_cache_manager.free`，将KV块返回到块池），或者可能跳过调度并继续执行。
1. **分配块**——通过KV缓存管理器的协调器，从块池中获取前`n`个块（之前提到的`free_block_queue`双向链表）。存储到`req_to_blocks`，这是映射每个`request_id`到其KV缓存块列表的字典。

![KV缓存块](/images/others/inside-vllm-anatomy-of-a-high-throughput-llm-inference-system/003-92ff8d14.png)

KV缓存块列表

我们终于准备好进行前向传递了！

## 运行前向传递

我们调用模型执行器的`execute_model`，它委托给`Worker`，后者又委托给模型运行器。

以下是主要步骤：

1. **更新状态**——从`input_batch`中修剪完成的请求；更新与前向传递相关的杂项元数据（例如，每个请求的KV缓存块，将用于索引到分页KV缓存内存）。
1. **准备输入**——从CPU→GPU复制缓冲区；计算位置；构建`slot_mapping`（在示例中详述）；构建注意力元数据。
1. **前向传递**——使用自定义分页注意力内核运行模型。所有序列被展平并连接成一个长的“超级序列”。位置索引和注意力掩码确保每个序列仅关注其自身的令牌，从而实现无需右填充的连续批处理。
1. **收集最后令牌状态**——提取每个序列最终位置的隐藏状态并计算逻辑值。
1. **样本** — 根据采样配置（贪婪、温度、top-p、top-k等）从计算出的logits中采样token。

前向传播步骤本身有两种执行模式：

1. **Eager模式** — 当启用eager执行时，运行标准的PyTorch前向传播。
1. **"捕获"模式** — 当未强制使用eager时，执行/重放预捕获的CUDA图（记得我们在引擎构建的初始化KV缓存过程中捕获了这些图）。

这里有一个具体示例，应该能让连续批处理和分页注意力变得清晰：

![fwd pass - continuous batching & paged attn](/images/others/inside-vllm-anatomy-of-a-high-throughput-llm-inference-system/004-3e6a70d8.png)

前向传播：连续批处理和分页注意力

## 高级功能 — 扩展核心引擎逻辑

有了基本的引擎流程，我们现在可以看看高级功能。

我们已经讨论了抢占、分页注意力和连续批处理。

接下来，我们将深入探讨：

1. 分块预填充
1. 前缀缓存
1. 引导解码（通过语法约束的有限状态机）
1. 推测解码
1. 分离的P/D（预填充/解码）

## 分块预填充

分块预填充是一种通过将长提示的预填充步骤拆分为更小的块来处理长提示的技术。没有它，我们可能会遇到一个非常长的请求独占一个引擎步骤，阻止其他预填充请求运行。这会推迟所有其他请求并增加它们的延迟。

例如，让每个块包含`n`（=8）个token，用小写字母标记并用"-"分隔。一个长提示`P`可能看起来像`x-y-z`，其中`z`是一个不完整的块（例如2个token）。执行`P`的完整预填充将需要≥3个引擎步骤（>可能发生，如果它未在某个步骤中调度执行），并且只有在最后一个分块预填充步骤中我们才会采样一个新token。

这里是同一个示例的视觉表示：

![Chunked prefilling - pt 1](/images/others/inside-vllm-anatomy-of-a-high-throughput-llm-inference-system/005-e0b3ede8.png)

实现很简单：限制每个步骤的新token数量。如果请求的数量超过`long_prefill_token_threshold`，则将其重置为该值。底层的索引逻辑（如前所述）会处理其余部分。

在vLLM V1中，您可以通过将`long_prefill_token_threshold`设置为正整数来启用分块预填充。（技术上，无论是否设置，如果提示长度超过token预算，我们会截断它并运行分块预填充。）

## 前缀缓存

为了解释前缀缓存的工作原理，让我们拿原始代码示例并稍作调整：

```python
from vllm import LLM, SamplingParams

long_prefix = "<a piece of text that is encoded into more than block_size tokens>"

prompts = [
    "Hello, my name is",
    "The president of the United States is",
]

sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

def main():
    llm = LLM(model="TinyLlama/TinyLlama-1.1B-Chat-v1.0")

    outputs = llm.generate(long_prefix + prompts[0], sampling_params)
    outputs = llm.generate(long_prefix + prompts[1], sampling_params)

if __name__ == "__main__":
    main()
```

前缀缓存避免重新计算多个提示在开头共享的token — 因此称为**前缀**。

关键部分是`long_prefix`：它被定义为任何长于KV缓存块（默认为16个token）的前缀。为了简化我们的示例，假设`long_prefix`的长度正好是`n x block_size`（其中`n ≥ 1`）。

即，它完美对齐块边界 — 否则我们将不得不重新计算`long_prefix_len % block_size`个token，因为我们无法缓存不完整的块。

没有前缀缓存，每次我们处理具有相同`long_prefix`的新请求时，我们会重新计算所有`n x block_size`个token。

有了前缀缓存，这些token只计算一次（它们的KV存储在KV缓存分页内存中），然后被重用，因此只有新的提示token需要处理。这加速了预填充请求（尽管对解码没有帮助）。

这在vLLM中是如何工作的？

在第一次`generate`调用期间，在调度阶段，在`kv_cache_manager.get_computed_blocks`内部，引擎调用`hash_request_tokens`：

1. 此函数将`long_prefix + prompts[0]`拆分为16个token的块。
1. 对于每个完整块，它计算一个哈希（使用内置哈希或SHA-256，后者较慢但碰撞较少）。哈希结合了前一个块的哈希、当前token和可选的元数据。
1. 可选元数据包括：MM哈希、LoRA ID、缓存盐（注入第一个块的哈希中，确保只有具有此缓存盐的请求可以重用块）。

   每个结果存储为一个`BlockHash`对象，包含哈希和其token ID。我们返回一个块哈希列表。

该列表存储在`self.req_to_block_hashes[request_id]`中。

接下来，引擎调用`find_longest_cache_hit`来检查这些哈希是否已存在于`cached_block_hash_to_block`中。在第一个请求上，未找到匹配项。

![Prefix caching logic - pt 1](/images/others/inside-vllm-anatomy-of-a-high-throughput-llm-inference-system/006-498abd38.png)

然后我们调用`allocate_slots`，它调用`coordinator.cache_blocks`，将新的`BlockHash`条目与分配的KV块关联，并在`cached_block_hash_to_block`中记录它们。

之后，前向传播将填充分页KV缓存内存中的KV，对应于我们上面分配的KV缓存块。

经过许多引擎步骤后，它将分配更多KV缓存块，但这对我们的示例无关紧要，因为前缀在`long_prefix`之后立即分叉。

![Prefix caching logic - pt 2](/images/others/inside-vllm-anatomy-of-a-high-throughput-llm-inference-system/007-c5c73ae4.png)

在第二次具有相同前缀的`generate`调用上，步骤1-3重复，但现在`find_longest_cache_hit`找到所有`n`块的匹配项（通过线性搜索）。引擎可以直接重用那些KV块。

![Prefix caching logic - pt 3](/images/others/inside-vllm-anatomy-of-a-high-throughput-llm-inference-system/008-2a7fe4e7.png)

如果原始请求仍然存活，那些块的引用计数会增加（例如到2）。在这个示例中，第一个请求已经完成，所以块被释放回池中，它们的引用计数重置为0。因为我们能够从`cached_block_hash_to_block`中检索它们，我们知道它们是有效的（KV缓存管理器的逻辑设置如此），所以我们只是再次将它们从`free_block_queue`中移除。

📝高级说明：

KV缓存块仅在即将从`free_block_queue`（从左弹出）重新分配时变得无效，并且我们发现该块仍然有关联的哈希并存在于`cached_block_hash_to_block`中。在那一刻，我们清除块的哈希并从`cached_block_hash_to_block`中移除其条目，确保它不能通过前缀缓存被重用（至少对于那个旧前缀）。

这就是前缀缓存的要点：不要重新计算你已经见过的前缀 — 只需重用它们的KV缓存！

如果你理解了这个示例，你也理解了分页注意力是如何工作的。

前缀缓存默认启用。要禁用它：`enable_prefix_caching = False`。

## 引导解码（FSM）

引导解码是一种技术，在每个解码步骤中，logits受到基于语法的有限状态机的约束。这确保只有语法允许的token可以被采样。

这是一个强大的设置：你可以强制执行从正则语法（乔姆斯基类型-3，例如任意正则表达式模式）一直到上下文无关语法（类型-2，涵盖大多数编程语言）的任何内容。

为了让这更具体，让我们从最简单的例子开始，基于我们之前的代码：

```python
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams

prompts = [
    "This sucks",
    "The weather is beautiful",
]

guided_decoding_params = GuidedDecodingParams(choice=["Positive", "Negative"])
sampling_params = SamplingParams(guided_decoding=guided_decoding_params)

def main():
    llm = LLM(model="TinyLlama/TinyLlama-1.1B-Chat-v1.0")

    outputs = llm.generate(prompts, sampling_params)

if __name__ == "__main__":
    main()
```

在我给出的玩具示例中（假设字符级分词）：在预填充阶段，FSM（有限状态机）掩码logits，使得只有"P"或"N"是可行的。如果采样到"P"，FSM移动到"Positive"分支；下一步只允许"o"，依此类推。

![FSM](/images/others/inside-vllm-anatomy-of-a-high-throughput-llm-inference-system/009-44c19ee7.png)

玩具示例FSM（有限状态机）

这在vLLM中如何工作：

1. 在LLM引擎构建时，创建一个`StructuredOutputManager`；它可以访问分词器并维护一个`_grammar_bitmask`张量。
1. 当添加请求时，其状态设置为`WAITING_FOR_FSM`，并且`grammar_init`选择后端编译器（例如，`xgrammar` [\[7\]](#ref-7)；注意后端是第三方代码）。
1. 该请求的语法被异步编译。
1. 在调度期间，如果异步编译已完成，状态切换到`WAITING`，并且`request_id`被添加到`structured_output_request_ids`；否则它被放入`skipped_waiting_requests`以在下一个引擎步骤重试。
1. 在调度循环之后（仍在调度内部），如果有FSM请求，`StructuredOutputManager`要求后端准备/更新`_grammar_bitmask`。
1. 在前向传递产生logits后，xgr_torch_compile的函数将位掩码扩展到词汇表大小（32倍扩展比，因为我们使用32位整数），并将不允许的logits掩码为–∞。
1. 在采样下一个token后，请求的FSM通过`accept_tokens`前进。视觉上，我们在FSM图上移动到下一个状态。

步骤6值得进一步澄清。

如果`vocab_size = 32`，`_grammar_bitmask`是单个整数；其二进制表示编码哪些token是允许的（"1"）与不允许的（"0"）。例如，"101…001"扩展为长度为32的数组`[1, 0, 1, …, 0, 0, 1]`；位置为0的logits被设置为–∞。对于更大的词汇表，使用多个32位字并相应扩展/连接。后端（例如，`xgrammar`）负责使用当前FSM状态产生这些位模式。

📝注意：

这里的大部分复杂性隐藏在第三方库中，如xgrammar。

这是一个更简单的例子，词汇表大小=8且使用8位整数（适合喜欢我视觉化的人）：

![FSM](/images/others/inside-vllm-anatomy-of-a-high-throughput-llm-inference-system/010-d812ac8e.png)

玩具示例

您可以在vLLM中通过传入所需的`guided_decoding`配置来启用此功能。

## 推测解码

在自回归生成中，每个新token都需要大型语言模型的前向传递。这很昂贵——每一步都重新加载并应用所有模型权重，仅为了计算单个token！（假设批次大小==1，通常为`B`）

推测解码[\[8\]](#ref-8)通过引入较小的草稿语言模型来加速此过程。草稿廉价地提出`k`个token。但我们最终不想从较小的模型采样——它仅用于猜测候选延续。大型模型仍然决定什么是有效的。

以下是步骤：

1. **草稿：**&#x5728;上下文中运行小模型并提出`k`个token

1. **验证：**&#x5728;上下文+`k`个草稿token上运行大型模型一次。这为这些`k`个位置加上一个额外位置产生概率（所以我们得到`k+1`个候选）

1. **接受/拒绝：**&#x4ECE;左到右遍历`k`个草稿token：
   - 如果大型模型对草稿token的概率≥草稿的概率，接受它
   - 否则，以概率`p_large(token)/p_draft(token)`
   - 接受它`k`在第一次拒绝时停止，或接受所有
     - 个草稿token。`k`如果所有`(k+1)`个草稿token都被接受，也从大型模型中“免费”采样额外的第
     - 个token（我们已经计算了该分布）。`p_large - p_draft`如果有拒绝，在该位置创建新的重新平衡分布（

**，最小值为0，归一化总和为1）并从其中采样最后一个token。**&#x4E3A;什么这有效：`k+1`尽管我们使用小模型提出候选，但接受/拒绝规则保证在期望中，序列的分布与我们从大型模型逐个token采样完全相同。这意味着推测解码在统计上等同于标准自回归解码——但可能快得多，因为单个大型模型传递可以产生多达

个token。

📝注意：[gpt-fast](https://github.com/meta-pytorch/gpt-fast)查看简单实现，以及[原始论文](https://arxiv.org/abs/2302.01318)了解数学细节和与完整模型采样等价的证明。

vLLM V1不支持LLM草稿模型方法，而是实现更快——但准确性较低——的提议方案：n-gram、EAGLE[\[9\]](#ref-9)和Medusa[\[10\]](#ref-10)。

每个的简要说明：

1. **n-gram：**&#x53D6;最后`prompt_lookup_max`个token；在序列中找到先前的匹配；如果找到，提出跟随该匹配的`k`个token；否则减小窗口并重试，直到`prompt_lookup_min`
1. 当前实现返回`k`个token，在**第一次**匹配之后。引入最近性偏差并反转搜索方向感觉更自然吗？（即最后一次匹配）

   **Eagle：**&#x5BF9;大型语言模型进行“模型手术”——保留嵌入和LM头，用轻量级MLP替换transformer堆栈；微调它作为廉价草稿

1. **Medusa：**&#x5728;大型模型顶部（LM头之前的嵌入）训练辅助线性头，以并行预测下一个`k`个token；使用这些头更高效地提出token，比运行单独的小语言模型

以下是如何在vLLM中使用`ngram`作为草稿方法调用推测解码：

```python
from vllm import LLM, SamplingParams

prompts = [
    "Hello, my name is",
    "The president of the United States is",
]

sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

speculative_config={
    "method": "ngram",
    "prompt_lookup_max": 5,
    "prompt_lookup_min": 3,
    "num_speculative_tokens": 3,
}

def main():
    llm = LLM(model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", speculative_config=speculative_config)

    outputs = llm.generate(prompts, sampling_params)

if __name__ == "__main__":
    main()
```

这在vLLM中如何工作？

**设置（在引擎构建期间）：**

1. 初始化设备：创建一个`drafter`（草稿模型，例如，`NgramProposer`）和一个`rejection_sampler`（部分用Triton编写）。
1. 加载模型：加载草稿模型权重（对于n-gram无操作）。

**之后在`generate`函数中**（假设我们获得全新请求）：

1. 使用大型模型运行常规预填充步骤。
1. 在前向传递和标准采样后，调用`propose_draft_token_ids(k)`从草稿模型中采样`k`个草稿token。
1. 将这些存储在`request.spec_token_ids`中（更新请求元数据）。
1. 在下一个引擎步骤，当请求在运行队列中时，添加`len(request.spec_token_ids)`到“新token”计数，以便`allocate_slots`为前向传递保留足够的KV块。
1. 复制`spec_token_ids`到`input_batch.token_ids_cpu`以形成（上下文+草稿）token。
1. 通过`_calc_spec_decode_metadata`计算元数据（这从`input_batch.token_ids_cpu`复制token，准备logits等），然后在草稿token上运行大型模型前向传递。
1. 代替从logits进行常规采样，使用`rejection_sampler`从左到右接受/拒绝并产生`output_token_ids`。
1. 重复步骤2-7，直到满足停止条件。

理解此过程的最佳方式是启动调试器并逐步执行代码库，但本节希望给您一个初步印象。还有这个：

![Drafting stage](/images/others/inside-vllm-anatomy-of-a-high-throughput-llm-inference-system/011-0ba74f4c.png)

![Verify stage & rejection sampling stage](/images/others/inside-vllm-anatomy-of-a-high-throughput-llm-inference-system/012-f94f19bc.png)

## 解耦P/D

我之前已经暗示过解耦P/D（预填充/解码）背后的动机。

预填充和解码具有非常不同的性能特征（计算密集型与内存带宽密集型），因此分离它们的执行是一个合理的设计。这提供了对延迟的更严格控制——包括`TFTT`（首令牌时间）和`ITL`（令牌间延迟）——更多细节将在[基准测试](#cpt5)部分讨论。

在实践中，我们运行`N`vLLM预填充实例和`M`vLLM解码实例，根据实时请求混合自动扩展它们。预填充工作器将KV写入专用的KV缓存服务；解码工作器从中读取。这隔离了长而突发的预填充与稳定、对延迟敏感的解码。

这在vLLM中是如何工作的？

为清晰起见，下面的示例依赖于`SharedStorageConnector`，一个用于说明机制的调试连接器实现。

连接器是vLLM处理实例间KV交换的抽象。连接器接口尚未稳定，计划进行一些近期改进，可能涉及更改，有些可能是破坏性的。

我们启动2个vLLM实例（GPU 0用于预填充，GPU 1用于解码），然后在它们之间传输KV缓存：

```python

import os
import time
from multiprocessing import Event, Process
import multiprocessing as mp

from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig

prompts = [
    "Hello, my name is",
    "The president of the United States is",
]

def run_prefill(prefill_done):
  os.environ["CUDA_VISIBLE_DEVICES"] = "0"

  sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=1)

  ktc=KVTransferConfig(
      kv_connector="SharedStorageConnector",
      kv_role="kv_both",
      kv_connector_extra_config={"shared_storage_path": "local_storage"},
  )

  llm = LLM(model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", kv_transfer_config=ktc)
  llm.generate(prompts, sampling_params)

  prefill_done.set()  # notify decode instance that KV cache is ready

  # To keep the prefill node running in case the decode node is not done;
  # otherwise, the script might exit prematurely, causing incomplete decoding.
  try:
      while True:
          time.sleep(1)
  except KeyboardInterrupt:
      print("Script stopped by user.")

def run_decode(prefill_done):
  os.environ["CUDA_VISIBLE_DEVICES"] = "1"

  sampling_params = SamplingParams(temperature=0, top_p=0.95)

  ktc=KVTransferConfig(
      kv_connector="SharedStorageConnector",
      kv_role="kv_both",
      kv_connector_extra_config={"shared_storage_path": "local_storage"},
  )

  llm = LLM(model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", kv_transfer_config=ktc)

  prefill_done.wait()  # block waiting for KV cache from prefill instance

  # Internally it'll first fetch KV cache before starting the decoding loop
  outputs = llm.generate(prompts, sampling_params)

if __name__ == "__main__":
  prefill_done = Event()
  prefill_process = Process(target=run_prefill, args=(prefill_done,))
  decode_process = Process(target=run_decode, args=(prefill_done,))

  prefill_process.start()
  decode_process.start()

  decode_process.join()
  prefill_process.terminate()
```

📝注意：

我还尝试过`LMCache` [\[11\]](#ref-11)，最快的生产就绪连接器（使用NVIDIA的NIXL作为后端），但它仍处于前沿，我遇到了一些错误。由于其大部分复杂性位于外部仓库中，`SharedStorageConnector`是更好的解释选择。

以下是vLLM中的步骤：

1. **实例化**——在引擎构建期间，连接器在两个地方创建：
   - 在工作器的初始化设备过程中（在初始化工作器分布式环境函数下），角色为“工作器”。
   - 在调度器构造函数中，角色为“调度器”。

1. **缓存查找**——当调度器处理来自`waiting`队列的预填充请求（在本地前缀缓存检查后）时，它调用连接器的`get_num_new_matched_tokens`。这检查KV缓存服务器中是否有外部缓存的令牌。预填充总是看到0；解码可能有缓存命中。结果在调用`allocate_slots`之前添加到本地计数中。

1. **状态更新**——调度器然后调用`connector.update_state_after_alloc`，记录有缓存的请求（对预填充无操作）。

1. **元构建**——在调度结束时，调度器调用`meta = connector.build_connector_meta`：
   - 预填充添加所有具有`is_store=True`的请求（用于上传KV）。
   - 解码添加具有`is_store=False`的请求（用于获取KV）。

1. **上下文管理器**——在前向传递之前，引擎进入KV连接器上下文管理器：
   - 进入时：调用`kv_connector.start_load_kv`。对于解码，这从外部服务器加载KV并注入到分页内存中。对于预填充，这是无操作。
   - 退出时：调用`kv_connector.wait_for_save`。对于预填充，这会阻塞直到KV上传到外部服务器。对于解码，这是无操作。

这是一个可视化示例：

![disaggregated P/D](/images/others/inside-vllm-anatomy-of-a-high-throughput-llm-inference-system/013-c27c3e57.png)

解耦P/D

📝附加说明：

- 对于`SharedStorageConnector`，“外部服务器”只是本地文件系统。
- 根据配置，KV传输也可以逐层进行（在每个注意力层之前/之后）。
- 解码仅在请求的第一步加载外部KV；之后在本地计算/存储。

## 从UniprocExecutor到MultiProcExecutor

有了核心技术，我们现在可以讨论扩展。

假设您的模型权重不再适合单个GPU的VRAM。

第一个选项是使用张量并行（例如`TP=8`）将模型分片到同一节点上的多个GPU上。如果模型仍然不适合，下一步是在节点间进行流水线并行。

📝注意：

- 节点内带宽显著高于节点间，这就是为什么张量并行（TP）通常优于流水线并行（PP）。（PP通信的数据量也少于TP。）
- 我不涵盖专家并行（EP），因为我们专注于标准Transformer而非MoE，也不涵盖序列并行，因为TP和PP在实践中最常用。

在这个阶段，我们需要多个GPU进程（工作器）和一个编排层来协调它们。这正是`MultiProcExecutor`提供的。

![MultiProcExecutor](/images/others/inside-vllm-anatomy-of-a-high-throughput-llm-inference-system/014-63e1eb31.png)

MultiProcExecutor在TP=8设置中（驱动工作器为rank 0）

这在vLLM中如何工作：

1. `MultiProcExecutor`初始化一个`rpc_broadcast_mq`消息队列（底层使用共享内存实现）。

1. 构造函数循环遍历`world_size`（例如`TP=8 ⇒ world_size=8`）并通过`WorkerProc.make_worker_process`为每个rank生成一个守护进程。

1. 对于每个工作器，父进程首先创建一个读取器和写入器管道。

1. 新进程运行`WorkerProc.worker_main`，它实例化一个工作器（经过与`UniprocExecutor`中相同的“初始化设备”、“加载模型”等步骤）。

1. 每个工作器确定它是驱动（TP组中的rank 0）还是常规工作器。每个工作器设置两个队列：
   - `rpc_broadcast_mq`（与父进程共享）用于接收工作。
   - `worker_response_mq`用于发送响应回。

1. 在初始化期间，每个子进程通过管道将其`worker_response_mq`句柄发送给父进程。一旦全部接收，父进程解除阻塞——这完成协调。

1. 工作器然后进入忙循环，阻塞在`rpc_broadcast_mq.dequeue`上。当工作项到达时，它们执行它（就像在`UniprocExecutor`中一样，但现在有TP/PP特定的分区工作）。结果通过`worker_response_mq.enqueue`发送回。

1. 在运行时，当请求到达时，`MultiProcExecutor`将其入队到`rpc_broadcast_mq`（非阻塞）中，用于所有子工作器。然后它等待指定输出rank的`worker_response_mq.dequeue`以收集最终结果。

从引擎的角度看，没有任何变化——所有这些多进程复杂性都通过调用模型执行器的`execute_model`被抽象掉。

- 在`UniProcExecutor`情况下：execute_model直接导致在工作器上调用execute_model
- 在`MultiProcExecutor`情况下：execute_model间接通过`rpc_broadcast_mq`

在每个工作器上调用execute_model

此时，我们可以使用相同的引擎接口运行资源允许的任意大模型。`DP > 1`下一步是扩展：启用数据并行（

## 分布式系统服务vLLM

设置服务基础设施有很多方法，但为具体起见，这里有一个示例：假设我们有两个H100节点，并希望在其上运行四个vLLM引擎。

如果模型需要`TP=4`，我们可以这样配置节点。

![服务器配置包含2个8xH100节点](/images/others/inside-vllm-anatomy-of-a-high-throughput-llm-inference-system/015-b9dd52a1.png)

服务器配置包含2个8xH100节点（1个无头节点，1个API服务器节点）

在第一个节点上，以无头模式（无API服务器）运行引擎，使用以下参数：

```python
vllm serve <model-name>
  --tensor-parallel-size 4
  --data-parallel-size 4
  --data-parallel-size-local 2
  --data-parallel-start-rank 0
  --data-parallel-address <master-ip>
  --data-parallel-rpc-port 13345
  --headless
```

并在另一个节点上运行相同的命令，稍作调整：

- 无`--headless`
- 修改DP起始排名

```python
vllm serve <model-name>
  --tensor-parallel-size 4
  --data-parallel-size 4
  --data-parallel-size-local 2
  --data-parallel-start-rank 2
  --data-parallel-address <master-ip>
  --data-parallel-rpc-port 13345
```

📝注意：

这假设网络已配置，所有节点都能访问指定的IP和端口。

这在VLLM中如何工作？

## 在无头服务器节点上

在无头节点上，一个`CoreEngineProcManager`启动2个进程（每个`--data-parallel-size-local`）各运行`EngineCoreProc.run_engine_core`。这些函数中的每一个都创建一个`DPEngineCoreProc`（引擎核心），然后进入其忙循环。

`DPEngineCoreProc`初始化其父`EngineCoreProc`（的子`EngineCore`），它：

1. 创建一个`input_queue`和`output_queue`（`queue.Queue`）。
1. 使用一个`DEALER` ZMQ套接字（异步消息库）与另一个节点上的前端执行初始握手，并接收协调地址信息。
1. 初始化DP组（例如使用NCCL后端）。
1. 用`EngineCore`初始化`MultiProcExecutor`（`TP=4`在4个GPU上，如前所述）。
1. 创建一个`ready_event`（`threading.Event`）。
1. 启动一个输入守护线程（`threading.Thread`）运行`process_input_sockets(…, ready_event)`。类似地启动一个输出线程。
1. 仍在主线程中，等待`ready_event`，直到所有4个进程（跨越2个节点）的输入线程完成协调握手，最终执行`ready_event.set()`。
1. 一旦解除阻塞，向前端发送“就绪”消息，包含元数据（例如，`num_gpu_blocks`在分页KV缓存内存中可用）。
1. 主线程、输入线程和输出线程随后进入各自的忙循环。

TL;DR：我们最终得到4个子进程（每个DP副本一个），每个运行一个主线程、输入线程和输出线程。它们与DP协调器和前端完成协调握手，然后每个进程的所有三个线程在稳态忙循环中运行。

![分布式系统包含4个DPEngineCoreProc](/images/others/inside-vllm-anatomy-of-a-high-throughput-llm-inference-system/016-65b08a2c.png)

分布式系统包含4个DP副本运行4个DPEngineCoreProc

**当前稳态：**

- **输入线程**— 在输入套接字上阻塞，直到请求从API服务器路由过来；收到后，解码有效载荷，通过`input_queue.put_nowait(...)`入队一个工作项，然后返回在套接字上阻塞。
- **主线程**— 在`input_queue.get(...)`上唤醒，将请求馈送到引擎；`MultiProcExecutor`运行前向传递并将结果入队到`output_queue`。
- **输出线程**— 在`output_queue.get(...)`上唤醒，将结果发送回API服务器，然后恢复阻塞。

**额外机制：**

- **DP波计数器**— 系统跟踪“波”；当所有引擎变为空闲时，它们静止，计数器在新工作到达时递增（用于协调/指标）。
- **控制消息**— API服务器可以发送不仅仅是推理请求（例如，中止和实用/控制RPC）。
- **用于锁步的虚拟步骤**— 如果任何DP副本有工作，所有副本执行一个前向步骤；没有请求的副本执行一个虚拟步骤以参与所需的同步点（避免阻塞活动副本）。

锁步澄清：这实际上仅对MoE模型是必需的，其中专家层形成EP或TP组，而注意力层仍然是DP。目前总是用DP完成—这只是因为“内置”非MoE DP的用途有限，因为你可以以正常方式运行多个独立的vLLM并在它们之间进行负载均衡。

现在对于第二部分，API服务器节点上发生了什么？

## 在API服务器节点上

我们实例化一个`AsyncLLM`对象（LLM引擎的asyncio包装器）。内部创建一个`DPLBAsyncMPClient`（数据并行、负载均衡、异步、多处理客户端）。

在`MPClient`的父类中，`launch_core_engines`函数运行并：

1. 创建用于启动握手的ZMQ地址（如在无头节点上所见）。
1. 生成一个`DPCoordinator`进程。
1. 创建一个`CoreEngineProcManager`（与无头节点上相同）。

在`AsyncMPClient`（的子`MPClient`）内部，我们：

1. 创建一个`outputs_queue`（`asyncio.Queue`）。
1. 我们创建一个asyncio任务`process_outputs_socket`，它（通过输出套接字）与所有4个`DPEngineCoreProc`的输出线程通信，并写入`outputs_queue`。
1. 随后，另一个asyncio任务`output_handler`从`AsyncLLM`读取此队列，并最终将信息发送到`create_completion`函数。

在`DPAsyncMPClient`内部，我们创建一个asyncio任务`run_engine_stats_update_task`，它与DP协调器通信。

DP协调器在前端（API服务器）和后端（引擎核心）之间调解。它：

- 定期向前端的`run_engine_stats_update_task`发送负载均衡信息（队列大小、等待/运行请求）。
- 处理来自前端的`SCALE_ELASTIC_EP`命令，通过动态更改引擎数量（仅适用于Ray后端）。
- 向后端发送`START_DP_WAVE`事件（当由前端触发）并报告波状态更新。

回顾一下，前端（`AsyncLLM`）运行几个asyncio任务（记住：并发，非并行）：

- 一类任务通过`generate`路径处理输入请求（每个新客户端请求生成一个新的asyncio任务）。
- 两个任务（`process_outputs_socket`，`output_handler`）处理来自底层引擎的输出消息。
- 一个任务（`run_engine_stats_update_task`）维护与DP协调器的通信：发送波触发器、轮询LB状态和处理动态缩放请求。

最后，主服务器进程创建一个FastAPI应用并挂载端点，如`OpenAIServingCompletion`和`OpenAIServingChat`，它们暴露`/completion`、`/chat/completion`等。然后通过Uvicorn提供堆栈。

所以，将所有内容放在一起，这是完整的请求生命周期！

您从终端发送：

```bash
curl -X POST http://localhost:8000/v1/completions -H "Content-Type: application/json" -d '{
  "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
  "prompt": "The capital of France is",
  "max_tokens": 50,
  "temperature": 0.7
}'
```

接下来发生什么：

1. 请求命中API服务器上的`OpenAIServingCompletion`的`create_completion`路由。

1. 该函数异步标记化提示，并准备元数据（请求ID、采样参数、时间戳等）。

1. 然后调用`AsyncLLM.generate`，它遵循与同步引擎相同的流程，最终调用`DPAsyncMPClient.add_request_async`。

1. 这反过来调用`get_core_engine_for_request`，它基于DP协调器的状态在引擎之间进行负载均衡（选择具有最小分数/最低负载的引擎：`score = len(waiting) * 4 + len(running)`）。

1. 将`ADD`请求发送到所选引擎的`input_socket`。

1. 在该引擎上：
   - 输入线程—解除阻塞，从输入套接字解码数据，并将工作项放置在`input_queue`上供主线程使用。
   - 主线程—在`input_queue`上解除阻塞，将请求添加到引擎，并重复调用`engine_core.step()`，将中间结果入队到`output_queue`，直到满足停止条件。
   - 提醒：`step()`调用调度器、模型执行器（这又可以是`MultiProcExecutor`！）等。我们已经见过这个了！

     输出线程—在`output_queue`上解除阻塞，并通过输出套接字发送结果。

1. 这些结果触发`AsyncLLM`输出asyncio任务（`process_outputs_socket`和`output_handler`），它们将令牌传播回FastAPI的`create_completion`路由。

1. FastAPI 附加元数据（完成原因、对数概率、使用信息等）并返回一个`JSONResponse`通过 Uvicorn 到您的终端！

就这样，您的完成结果返回了——整个分布式机制隐藏在一个简单的`curl`命令之后！ :) 太有趣了！！！

📝附加说明：

- 当添加更多 API 服务器时，负载均衡在操作系统/套接字级别处理。从应用程序的角度看，没有显著变化——复杂性被隐藏了。
- 使用 Ray 作为 DP 后端，您可以暴露一个 URL 端点（`/scale_elastic_ep`），支持自动向上或向下扩展引擎副本的数量。

## 基准测试和自动调优 - 延迟与吞吐量

到目前为止，我们一直在分析“气体粒子”——请求如何通过引擎/系统内部流动的细节。现在是时候放大视角，将系统作为一个整体来看，并问：我们如何衡量推理系统的性能？

在最高级别，有两个相互竞争的指标：

1. **延迟**——从提交请求到返回令牌的时间
1. **吞吐量**——系统每秒可以生成/处理的令牌/请求数量

**延迟**对于交互式应用程序最重要，用户正在等待响应。

**吞吐量**在离线工作负载中很重要，如用于预训练/后训练运行的合成数据生成、数据清洗/处理，以及一般任何类型的离线批量推理作业。

在解释为什么延迟和吞吐量相互竞争之前，让我们定义一些常见的推理指标：

| 指标                          | 定义                                                                                                        |
| ----------------------------- | ----------------------------------------------------------------------------------------------------------- |
| `TTFT`（首令牌时间）          | 从提交请求到接收到第一个输出令牌的时间                                                                      |
| `ITL`（令牌间延迟）           | 两个连续令牌之间的时间（例如，从令牌 i-1 到令牌 i）                                                         |
| `TPOT`（每个输出令牌时间）    | 请求中所有输出令牌的平均 ITL                                                                                |
| `Latency / E2E`（端到端延迟） | 处理请求的总时间，即 TTFT + 所有 ITL 的总和，或等效于提交请求和接收到最后一个输出令牌之间的时间             |
| `Throughput`                  | 每秒处理的总令牌数（输入、输出或两者），或每秒请求数                                                        |
| `Goodput`                     | 满足服务水平目标（SLOs）的吞吐量，如最大 TTFT、TPOT 或 e2e 延迟。例如，只计算来自满足这些 SLOs 的请求的令牌 |

![ttft, itl, e2e latency](/images/others/inside-vllm-anatomy-of-a-high-throughput-llm-inference-system/017-fc530c2a.png)

ttft, itl, e2e latency

这是一个简化模型，解释了这两个指标的竞争性质。

假设：权重 I/O 而非 KV 缓存 I/O 占主导；即我们处理的是短序列。

当查看批次大小`B`如何影响单个解码步骤时，权衡变得清晰。随着`B ↓`趋近于 1，ITL 下降：每个步骤的工作量减少，令牌不与其他令牌“竞争”。随着`B ↑`趋近于无穷大，ITL 上升，因为每个步骤执行更多 FLOPs——但吞吐量提高（直到达到峰值性能），因为权重 I/O 在更多令牌上分摊。

屋顶线模型有助于理解这里：在饱和批次`B_sat`以下，步骤时间由 HBM 带宽主导（逐层将权重流式传输到片上内存），因此步骤延迟几乎平坦——计算 1 个与 10 个令牌可能花费相似时间。超过`B_sat`，内核变得计算受限，步骤时间大致随`B`增长；每个额外令牌都会增加 ITL。

![roofline perf model](/images/others/inside-vllm-anatomy-of-a-high-throughput-llm-inference-system/018-f7e689f7.png)

roofline perf model

📝注意：

为了更严格的处理，我们必须考虑内核自动调优：随着`B`增长，运行时可能切换到针对该形状更高效的内核，改变实现的性能`P_kernel`。步骤延迟是`t = FLOPs_step / P_kernel`，其中`FLOPs_step`是步骤中的工作量。您可以看到，当`P_kernel`达到`P_peak`时，每个步骤的更多计算将直接导致延迟增加。

## 如何在 vLLM 中进行基准测试

vLLM 提供了一个`vllm bench {serve,latency,throughput}` CLI，包装了 vllm / benchmarks / {server,latency,throughput}.py。

以下是脚本的功能：

- **latency**——使用短输入（默认 32 个令牌）并以小批次（默认 8）采样 128 个输出令牌。它运行多次迭代并报告批次的 e2e 延迟。
- **throughput**——一次性提交一组固定的提示（默认：1000 个 ShareGPT 样本）（也称为`QPS=Inf`模式），并报告运行期间的输入/输出/总令牌数和每秒请求数。
- **serve**——启动一个 vLLM 服务器，并通过从泊松（或更一般地，伽马）分布采样请求到达间隔时间来模拟真实世界工作负载。它在时间窗口内发送请求，测量我们讨论过的所有指标，并可以选择强制执行服务器端最大并发数（通过信号量，例如将服务器限制为 64 个并发请求）。

以下是运行延迟脚本的示例：

```bash
vllm bench latency
  --model <model-name>
  --input-tokens 32
  --output-tokens 128
  --batch-size 8
```

CI 中使用的基准测试配置位于`.buildkite/nightly-benchmarks/tests`。

还有一个自动调优脚本，驱动 serve 基准测试以找到满足目标 SLOs 的参数设置（例如，“在保持 p99 e2e < 500 ms 的同时最大化吞吐量”），返回建议的配置。

## 尾声

我们从基本引擎核心（`UniprocExecutor`）开始，添加了高级功能如推测解码和前缀缓存，扩展到`MultiProcExecutor`（使用`TP/PP > 1`），最后扩展出去，将所有内容包装在异步引擎和分布式服务堆栈中——以如何衡量系统性能结束。

vLLM 还包括我跳过的专门处理。例如：

- **多样化硬件后端：** TPUs、AWS Neuron（Trainium/Inferentia）等。
- **架构/技术：** `MLA`、`MoE`、编码器-解码器（例如，Whisper）、池化/嵌入模型、`EPLB`、`m-RoPE`、`LoRA`、`ALiBi`、无注意力变体、滑动窗口注意力、多模态 LMs 和状态空间模型（例如，Mamba/Mamba-2、Jamba）
- **TP/PP/SP**
- **混合 KV 缓存逻辑**（Jenga）、更复杂的采样方法如束采样等
- **实验性**：异步调度

好处是，这些大多数与上述主要流程正交——您几乎可以将它们视为“插件”（实际上有一些耦合，当然）。

我喜欢理解系统。话虽如此，在这个高度上，分辨率肯定受到了影响。在接下来的帖子中，我将放大特定子系统并深入细节。

💡联系我：

如果您发现帖子中有任何错误，请私信我——欢迎在[X](https://x.com/gordic_aleksa)或[LinkedIn](https://www.linkedin.com/in/aleksagordic/)上给我留言，或通过[匿名反馈](https://docs.google.com/forms/d/1z1fEirrN2xtGxAsJvptpM7yV4ByT5SF25S-XiMPrXNA/edit)。

## 致谢

非常感谢 [Hyperstack](https://www.hyperstack.cloud/) 在过去一年为我提供H100进行实验！

感谢 [Nick Hill](https://www.linkedin.com/in/nickhillprofile/)（vLLM核心贡献者，RedHat）、[Mark Saroufim](https://x.com/marksaroufim)（PyTorch）、[Kyle Krannen](https://www.linkedin.com/in/kyle-kranen/)（NVIDIA，Dynamo）和 [Ashish Vaswani](https://www.linkedin.com/in/ashish-vaswani-99892181/) 阅读此博客文章的预发布版本并提供反馈！

## 参考文献

1. vLLM <https://github.com/vllm-project/vllm>
1. 《Attention Is All You Need》，<https://arxiv.org/abs/1706.03762>
1. 《Efficient Memory Management for Large Language Model Serving with PagedAttention》，<https://arxiv.org/abs/2309.06180>
1. 《DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model》，<https://arxiv.org/abs/2405.04434>
1. 《Jenga: Effective Memory Management for Serving LLM with Heterogeneity》，<https://arxiv.org/abs/2503.18292>
1. 《Orca: A Distributed Serving System for Transformer-Based Generative Models》，<https://www.usenix.org/conference/osdi22/presentation/yu>
1. 《XGrammar: Flexible and Efficient Structured Generation Engine for Large Language Models》，<https://arxiv.org/abs/2411.15100>
1. 《Accelerating Large Language Model Decoding with Speculative Sampling》，<https://arxiv.org/abs/2302.01318>
1. 《EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty》，<https://arxiv.org/abs/2401.15077>
1. 《Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads》，<https://arxiv.org/abs/2401.10774>
1. LMCache，<https://github.com/LMCache/LMCache>
