---
title: 'Pipeline Parallelism in SGLang: Scaling to Million-Token Contexts and Beyond'
slug: pipeline-parallelism-in-sglang-scaling-to-million-token-contexts-and-beyond
date: '2026-03-24'
tags: ['Distributed Parallel']
status: published
source_url: 'https://lmsys.org/blog/2026-01-15-chunked-pipeline/'
source_author: lmsys.org
imported_at: '2026-03-24T06:50:54.711Z'
source:
  title: lmsys.org
  url: 'https://lmsys.org/blog/2026-01-15-chunked-pipeline/'
cover: >-
  /images/others/pipeline-parallelism-in-sglang-scaling-to-million-token-contexts-and-beyond/001-a1729821.png
lang: zh
translatedFrom: en
---
## TL;DR

我们很高兴地介绍SGLang高度优化的流水线并行（Pipeline Parallelism，PP）实现，专门设计用于应对超长上下文推理的挑战。通过整合**分块流水线并行（Chunked Pipeline Parallelism）**、**异步点对点通信（Asynchronous P2P Communication）**，以及一个简单而有效的**动态分块机制（Dynamic Chunking mechanism）**，这种PP设计实现了行业领先的性能，同时确保与其他并行策略、PD解耦（PD Disaggregation）和HiCache的无缝兼容。在多节点部署中，使用此实现扩展到PP4 TP8，与TP8相比，在分块预填充大小设置为12K时，在H20集群上为DeepSeek-V3.1带来了**3.31倍的预填充吞吐量（Prefill Throughput for DeepSeek-V3.1）**，显著优于TP32解决方案（2.54倍），优势达**30.5%**。这突显了PP在大规模跨节点扩展方面相对于纯TP的固有架构优势。此外，我们的实现还能实现高达**67.9%的首令牌时间（TTFT）减少**，同时保持**82.8%的强扩展效率（strong scaling efficiency）**，为扩展万亿参数模型以处理超长上下文提供了一条高效、开源的路径。

![DeepSeek-V3.1在H20上的预填充吞吐量（批量大小=1）（越高越好）](/images/others/pipeline-parallelism-in-sglang-scaling-to-million-token-contexts-and-beyond/001-a1729821.png)

DeepSeek-V3.1在H20上的预填充吞吐量（批量大小=1）（越高越好）\
注：DCK 12288（σ=0.65）表示启用动态分块，初始分块预填充大小设置为12K，平滑因子设置为0.65。

**👉 查看[PP路线图（PP Roadmap）](https://github.com/sgl-project/sglang/issues/11857)。**

## 引言（Introduction）

随着大语言模型（Large Language Models，LLMs）向万亿参数架构和“无限”上下文窗口扩展，底层服务基础设施必须向更细粒度、跨节点的并行化策略演进。虽然KV缓存技术有效减轻了冗余计算，但它们无法规避超长序列中因极大初始输入令牌长度（Input Token Length，ITL）而固有的高昂首令牌时间（Time to First Token，TTFT）。尽管张量并行（Tensor Parallelism，TP）仍然是节点内扩展的传统方法，但在多节点部署中经常遇到通信瓶颈。另一方面，尽管传统流水线并行（Pipeline Parallelism，PP）通过减少通信量来解决这一瓶颈，但在处理如此庞大的提示时，它却面临资源利用不足和气泡开销的问题。

受开源创新和学术研究的启发，SGLang引入了一种高度优化的流水线并行实现，具有异步通信和动态分块预填充功能，有效最小化了流水线气泡。通过整合这些技术，SGLang探索并重构了超长提示的处理方式——有效扩展了长序列预填充的高昂延迟，并将其转化为高吞吐量、计算可扩展的流式工作流。

实证基准测试表明，SGLang的PP实现实现了行业领先的性能。在大规模部署中，它在扩展到PP4时，为各种模型架构保持**超过80%的扩展效率（scaling efficiency）**，并且在H20上部署Qwen3-235B-A22B-FP8时，使用PP8还能为超长提示实现**高达81%的TTFT减少**。

## 背景：为什么需要流水线并行？（Background: Why Pipeline Parallelism?）

为了验证流水线并行（Pipeline Parallelism，PP）对于长上下文预填充的必要性，必须将其与现有范式——特别是张量并行（Tensor Parallelism，TP）和上下文并行（Context Parallelism，CP）——进行评估。虽然TP和CP具有不同的优势，但对它们的通信量、气泡比率和实现复杂性的理论和实证分解表明，PP在多节点扩展中占据独特且最优的位置。以下分析概述了每种方法固有的具体权衡。

### 1. 通信量与可扩展性分析（Communication Volume and Scalability Analysis）

分布式推理扩展的主要瓶颈是设备间通信。随着模型深度和序列长度的增加，设备间传输的数据量成为一个限制因素，尤其是在扩展到大规模和多节点部署时。

假设B代表批量大小（对于超长上下文推理通常为1），S代表总序列长度，H代表隐藏状态维度，L代表总层数，M代表微批次大小，激活精度为FP8（1字节）。基于此，我们分析了不同并行策略的通信量。

- **TP：** TP将单个权重张量在单层内跨多个设备分割。因此，由于需要在注意力块（Attention Block）和MLP块（MLP Block）之后进行同步，TP会产生高通信开销。因此，通信量随层数线性增长。这种频繁的**全归约（All-Reduce）**&#x540C;步使TP受带宽限制，限制了其在大型集群中的可扩展性。（注：每个全归约在基于环的实现中涉及2倍数据大小。每层涉及2次全归约操作，一次在注意力块之后，一次在MLP块之后。）

- **CP：** 类似地，CP需要大量的同步通信来跨设备聚合键值（Key-Value，KV）状态。通常，CP在每一层使用**全收集（All-Gather）**，导致在带宽受限环境中产生显著的延迟惩罚。（注：假设CP使用基于环注意力（Ring-Attention）的解决方案。对于使用GQA的模型，H\_kv小于H，这减少了CP的通信量。）

- **PP：** 相比之下，PP表现出显著减少的通信足迹。数据仅在流水线阶段的**边界处**传输，使用**点对点（Point-to-Point，P2P）**&#x539F;语而非集体操作。由于一个阶段通常包含多个层，通信频率由阶段数（而非总层数）决定。关键的是，对于固定模型，当我们增加每个阶段的层数时，边界处的通信量保持恒定。（注意：在多节点部署中，与TP相比，PP实现了总通信量近一个数量级的减少。）

### 2. 气泡率权衡

虽然PP优化了通信，但它引入了流水线气泡——设备等待数据依赖的空闲期。这带来了通信效率与设备利用率之间的权衡。

- **TP和CP：**&#x8FD9;两种方法理论上都实现了零气泡率，因为所有设备同时计算同一张量或序列的不同部分。这最大化了计算强度，假设通信不会阻塞计算。

- **PP：**&#x50;P不可避免地会产生气泡率，由PP大小和微批次数之间的相互作用量化：然而，对于工作负载较大的长上下文预填充场景，该比率显著下降，使得效率损失与通信增益相比可忽略不计。在[**性能影响**](#performance-impact)部分，我们将评估我们PP实现的**强扩展效率**（即处理器数量增加而问题规模保持不变）。

值得注意的是，虽然PP在跨节点扩展方面具有明显优势，其中通信带宽常成为主要瓶颈，但纯高程度PP配置通常不推荐。这是因为对于固定工作负载，流水线气泡率随PP大小成比例增加。相反，更好的策略是利用无气泡并行方法，如TP或CP，进行节点内扩展。由于节点内通信通常使用高带宽互连（如NVLink），这些集体操作远不如跨节点传输可能成为性能瓶颈，允许系统最大化计算利用率而不产生额外流水线开销。

### 3. 实现复杂性与架构通用性

新功能的实现复杂性和架构通用性对于现代推理系统至关重要，尤其是对于开源项目。

- **TP：**&#x54;P易于实现且广泛支持。然而，大规模TP配置本质上不适用，因为量化块所需的粒度有时无法与MoE FFN权重施加的分区约束对齐。因此，即使不考虑通信量和开销，由于与量化（一种关键且不可或缺的优化技术）不兼容，在多节点扩展场景中通常无法使用更大的TP。
- **CP：**&#x43;P复杂，需要对注意力机制进行特定且通常是侵入性的修改（例如Ring Attention）。这些更改必须针对每个注意力变体和特定模型定制，降低了通用性。
- **PP：**&#x50;P代表中等复杂性。它需要对模型进行分区，但对层的内部机制保持不可知。这使得PP成为适用于所有模型架构的通用解决方案，无需为特定注意力变体重写内核级代码。在某种程度上，消除PP气泡比实现PP本身更困难。

| 指标        | 张量并行（TP）      | 上下文并行（CP）     | 流水线并行（PP）  |
| --------- | ------------- | ------------- | ---------- |
| **分割维度**  | 隐藏状态          | 序列            | 层          |
| **通信模式**  | AllReduce（每层） | AllGather（每层） | P2P（发送/接收） |
| **通信量**   | 高             | 中             | **低**      |
| **气泡率**   | **0**         | **0**         |            |
| **实现复杂性** | **低**         | 高（注意力变体特定）    | 中          |
| **架构通用性** | **高**         | 低             | **高**      |

总之，通用性与扩展效率的平衡使PP不仅是一种替代方案，而且是**必要组件**，用于将长上下文预填充扩展到大规模多节点集群，其中TP和CP遇到带宽上限。同时，CP有潜力补充TP以实现节点内无气泡扩展和加速。**PP × CP**已在开发中（[未来路线图](#future-roadmap)），将包含在本博客的第二部分。

## 挑战：“气泡”与“墙”

在传统流水线并行设置中，模型层跨GPU分区（阶段1到阶段N）。当处理标准请求（例如<4K令牌）时，通常工作良好。然而，当处理超过**128K甚至1M令牌**的提示时，两个关键问题出现：

1. **流水线气泡：**&#x5C06;提示作为整体批次处理迫使下游GPU进入长时间空闲状态，产生巨大的“流水线气泡”，严重降低吞吐量。
1. **内存墙：**&#x5728;单次传递中处理100万令牌提示需要存储和通信整个序列的中间隐藏状态，导致显著开销和峰值内存占用。

## SGLang流水线并行架构

SGLang的流水线实现超越了标准的“顺序”方法。我们引入了几个高级功能以最小化“气泡”（即GPU空闲时间）并最大化硬件利用率。

### 1. 分块流水线并行（CPP）

在单次前向传递中处理100万令牌提示将导致巨大气泡，因为后续阶段等待第一阶段完成。受Mooncake[\[1\]](https://dl.acm.org/doi/pdf/10.1145/3773772)、BladeLLM[\[2\]](https://arxiv.org/pdf/2501.15383?)和TeraPipe[\[3\]](http://proceedings.mlr.press/v139/li21y/li21y.pdf)，SGLang 支持分块流水线并行（Chunked Pipeline Parallelism）。不同于将完整提示词输入流水线，SGLang 将提示词划分为更小的“块”（例如 4K 或 6K 个令牌）。这些块像微批次一样流过流水线各阶段。通过将长提示词分解为小块，系统可以对预填充（prefill）阶段进行“流水线化”。一旦第一阶段完成对块 1 的隐藏状态计算并启动流水线并行（PP）通信，它立即转向处理块 2，而第二阶段同时开始处理块 1。这将流水线启动延迟从与总序列长度成正比，降低到仅与第一个块的大小成正比。

从工程角度看，这一方法是应对超长上下文挑战的关键第一步。值得注意的是，SGLang 早在六个多月前就率先支持了此功能（[#5724](https://github.com/sgl-project/sglang/pull/5724)，[#8846](https://github.com/sgl-project/sglang/pull/8846)），突显了其长期以来对优化现实世界长上下文推理的承诺。

### 2. 更好的重叠：微批次与异步点对点通信

尽管结合流水线并行（Pipeline Parallelism）与分块预填充（Chunked Prefill）相比张量并行（tensor parallelism）能显著减少通信量，但它常受流水线气泡（pipeline bubbles）困扰，即 GPU 在等待 CPU 元数据处理或网络传输时阻塞。为消除此性能隐患，SGLang 实现了一个微批次事件循环（Micro-batching Event Loop），配合非阻塞的异步点对点（P2P）通信，以重叠 GPU 计算与 CPU 元数据处理及流水线并行（PP）通信。这确保当一个微批次在 GPU 上计算时，下一个微批次已在准备并有效就位，从而尽可能保持流水线饱和。代码可在[此处](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/scheduler_pp_mixin.py)获取。

实现的关键机制包括：

- **事件循环中的解耦同步/异步逻辑：**&#x8C03;度器在`async_send`中使用`_pp_send_pyobj_to_next_stage`。它不等待传输完成，而是返回一个`P2PWork`句柄。实际的同步（`P2PWork.work.wait()`）被推迟到调用`_pp_commit_comm_work`时进行，允许 CPU 在数据传输过程中执行其他工作——例如调度下一个批次或处理元数据。
- **多流执行：**&#x9664;了作为同步流的主`default_stream`外，SGLang 还利用专用的`forward_stream`和`copy_stream`来分别执行前向传递 GPU 计算和数据到主机（D2H）内存传输，以实现更好的重叠。当`_pp_launch_batch`在当前阶段的 GPU 上执行当前微批次时，CPU 使用`_pp_process_batch_result`处理上一个微批次的结果。

### 3. 高级选项：动态分块

通过分块流水线并行（Chunked Pipeline Parallelism）和异步点对点（P2P）通信，SGLang 在流水线并行（PP）规模增加到 4 时已实现超过 80% 的强扩展效率。然而，使用固定大小的分块预填充仍可能在流水线中产生气泡，且随着流水线并行（PP）度增加，这种低效性变得更加明显。此现象背后的主要原因是模型在相同大小的块上表现出不均匀的执行延迟，这主要归因于自注意力（self-attention）的增量性质。**随着前缀序列长度增长，每个块的处理时间非线性增加。这些时间不匹配在流水线中传播，在更高的流水线并行（PP）等级上加剧效率损失。**

![图 1：固定分块预填充大小的流水线示意图](/images/others/pipeline-parallelism-in-sglang-scaling-to-million-token-contexts-and-beyond/002-b0f29cf7.jpg)

*图 1：固定分块预填充大小的流水线示意图*

我们使用较大的流水线并行（PP）规模测试了不同模型，发现它们都符合此结论。以下是典型情况的性能分析结果。

![图 2：流水线并行（PP）等级 7 使用固定分块预填充大小的性能分析结果](/images/others/pipeline-parallelism-in-sglang-scaling-to-million-token-contexts-and-beyond/003-69d0a754.png)

*图 2：流水线并行（PP）等级 7 使用固定分块预填充大小的性能分析结果*

因此，如果 SGLang 在分块流水线并行（CPP）中仍**使用固定的分块预填充大小，流水线气泡比率将大于理论预期（即**）。

为解决此问题，SGLang 引入了动态分块机制来预测下一个块的最优大小，使其满足以下条件：

其中表示前缀序列长度（Prefix Sequence Length），表示下一个块大小（Next Chunk Size）。通过分析一系列具有不同初始令牌长度（ITL）的请求，我们将累积运行时间建模为序列长度的二次函数。利用此模型，我们求解出对于任意给定前缀长度的最优下一个块大小。由于注意力（Attention）机制的计算/通信复杂度随缩放，下一个块大小将随着增长而逐渐减小，以保持流水线各阶段间块执行时间的对齐。

基于此方法，调度器可以在运行时预测并动态减小块大小，以最小化由阶段错位引起的气泡。需要注意的是，调度器不使用原始预测值。为便于高效的键值缓存（KVCache）内存管理并确保与硬件执行效率的亲和性，该值向下对齐到最接近的 max(`--page-size`, 64) 的倍数。

![图 3：完美动态分块的流水线示意图](/images/others/pipeline-parallelism-in-sglang-scaling-to-million-token-contexts-and-beyond/004-9aeef911.jpg)

*图 3：完美动态分块的流水线示意图*

然而，由于硬件、模型和目标工作负载的差异，静态配置很少能在所有场景下达到最优。因此，切换到动态分块模式时，实现峰值性能需要一定程度的超参数调优。此外，我们发现由于不同形状的内核性能变化，很难完美拟合二次函数。因此，我们引入了一个环境变量（`SGLANG_DYNAMIC_CHUNKING_SMOOTH_FACTOR`）来平滑动态分块算法的缩减，默认值为 0.75，它决定了在预填充阶段块大小可以变化的程度。较大的值导致更激进的块大小缩减，可能提高性能但增加总块数（末尾的块大小可能变得非常小，这可能导致性能下降）。

**动态分块预填充调优指南**

- **步骤 1 - 迭代寻找针对目标 PP 大小的最优固定分块预填充大小**：针对不同目标 ITL 的不同 PP 大小可能具有不同的最优分块预填充大小。因此，用户应根据可用的扩展资源进行迭代以获得基线。

- **步骤 2 - 动态分块的初始分块大小选择**：将初始大小设置为最优固定分块预填充大小的 2 倍或 3 倍。这减少了总分块数，并防止“尾部分块”未充分利用硬件。为了在极大输入令牌长度（ITL）下保持效率，动态预测器自动确保后续分块至少为此初始大小的 1/4。此外，对于此类情况，也建议使用更大的初始分块大小（例如，最优固定分块预填充大小的 4 倍）。

- **步骤 3 - 平滑因子调整**：此因子控制分块大小调整二次性能拟合模型给出的预测的严格程度。

  - 1.0：严格遵循模型。
  - **0.6 – 0.85（推荐）**：实现动态扩展与硬件稳定性最佳平衡的典型范围。通过实验，我们发现 0.6 到 0.85 之间的范围通常能为动态分块带来最佳性能，如图 4 和图 5 所示。
  - 0：禁用动态调整，恢复为传统的固定大小分块。

![图 4：为 DeepSeek-V3.1 调整平滑因子的示例（数值越低越好）](/images/others/pipeline-parallelism-in-sglang-scaling-to-million-token-contexts-and-beyond/005-8cd73d80.png)

*图 4：为 DeepSeek-V3.1 调整平滑因子的示例（数值越低越好）*

![图 5：为 Qwen3-235B-A22B-FP8 调整平滑因子的示例（数值越低越好）](/images/others/pipeline-parallelism-in-sglang-scaling-to-million-token-contexts-and-beyond/006-dab6a644.png)

*图 5：为 Qwen3-235B-A22B-FP8 调整平滑因子的示例（数值越低越好）*

- **另一个小的优化技巧：**&#x5F53;层无法在多个 rank 间均匀分配时，将较大的分区放在较高的 PP rank 上。当较高的 PP rank 等待前一阶段的结果时，这可以提高 GPU 利用率，从而减少较高 PP rank 上的气泡。如果我们以 DeepSeek-V3.1 为例，`SGLANG_PP_LAYER_PARTITION=15,15,15,16`通常比`16,15,15,15`表现更好。

为了验证这些组合策略的有效性，我们分析了使用动态分块的 DeepSeek-V3.1 的执行情况。如下文 PP rank 3 的分析结果所示，与静态分块方法相比，流水线气泡显著减少，从而实现了更饱和的执行。

![图 6：使用动态分块的 PP rank 3 的分析结果（DeepSeek-V3.1）](/images/others/pipeline-parallelism-in-sglang-scaling-to-million-token-contexts-and-beyond/007-35c6aff3.png)

*图 6：使用动态分块的 PP rank 3 的分析结果（DeepSeek-V3.1）*

### 4. 生产就绪：与 PD 解耦和 HiCache 的兼容性

SGLang 的一个独特优势是其原生支持流水线设置中的**预填充-解码（PD）解耦**。在解耦集群中，预填充节点可以使用高程度的 PP 来处理极长上下文提示，而解码节点可以利用不同的并行策略（如高程度的 TP）来最大化令牌生成速度。

- **逐分块 KVCache 传输：**&#x53;GLang 支持像 mooncake 这样的传输引擎后端，当启用 PD 解耦时，它可以在一个分块完成后立即将该分块的 KVCache 从预填充节点传输到解码节点，而不是等待所有分块完成。此功能大大减少了 KVCache 传输开销。
- **灵活的混合策略：**&#x53;GLang 允许用户混合并利用多种并行性与 PD 解耦。您可以在预填充的一组节点上为繁重的预填充任务运行 PP8 TP8，并为高吞吐量解码应用其他组合，例如 PP1 TP8、PP8 TP1 和 PP1 DP16 EP16，从而针对推理生命周期的不同阶段进行优化。这允许用户以高度可定制的方式满足生产中预期的首令牌时间（TTFT）和每令牌时间（TPOT）目标。
- **内存效率：**&#x901A;过跨设备分布模型权重，PP 减少了每 GPU 的内存占用，允许更大的 KV 缓存和更高的并发性。因此，在某些情况下，它可以用于扩展最大上下文长度。

当处理超过 128K 令牌的上下文时，SGLang 还支持分块流水线并行与**HiCache，**&#x4E00;个分布式分层 KV 缓存系统，以进一步降低多轮问答和智能体应用在超长初始 ITL 情况下的 TTFT：

- **语义前缀匹配：**&#x53;GLang 的 HiCache 使用基于基数树的层次结构在分块级别匹配前缀。当长上下文请求到达时，SGLang 可以执行分层缓存查找。如果前缀令牌（在先前分块中处理过）已经缓存在 HiCache 的“存储”层（例如，主机内存或本地磁盘）中，PP 流水线可以完全跳过这些分块，从而大幅降低 TTFT。

## 性能影响

本节对 DeepSeek-V3.1 和 Qwen3-235B-A22B-FP8 模型的 PP 性能特征进行了严格的定量评估。分析侧重于 PP 大小、动态分块（DCK）和硬件可扩展性之间的相互作用。

我们的实验测试平台是一个由 6 个 H20 节点（8 × 96GB VRAM GPU）组成的小型集群。由于测试资源有限，未对 DeepSeek-V3.1 进行 PP 度为 8 的实验。此外，对于 DeepSeek-V3.1 的 PP 大小 = 1 配置，我们使用了一个独立的 H20 节点（8 × 141GB VRAM GPU）来获得 128K 输入令牌长度的基线性能（在 96GB VRAM 版本上会发生 OOM）。为了更好地验证流水线饱和时的吞吐性能，我们在吞吐测试中对 16 个连续请求进行了基准测试并测量了平均值。

注意：我们使用符号**DCK**来表示启用动态分块时的分块预填充大小设置，以及**σ**代表动态分块（dynamic chunking）的平滑因子。为了进行极长上下文的实验，我们将上述模型的上下文长度覆盖为100万，仅用于性能分析。此外，我们尝试为DeepSeek-V3.1配置TP32和Qwen3-235B-A22B-FP8配置TP8进行实验，但不幸的是，大的TP配置本质上不受支持，因为权重量化块无法被FFN（MoE）层的权重整除（[参考问题](https://github.com/sgl-project/sglang/issues/3345)）。为了在多节点扩展场景中彻底比较TP和PP的差异，我们修改了模型实现文件（在`load_weights`）和config.json（[变通问题](https://github.com/sgl-project/sglang/issues/3491#issuecomment-2650779851)）中跳过部分权重加载），并设法使TP32仅用于性能验证目的而可运行。

### 输入令牌吞吐量与强扩展效率

吞吐量（Throughput）与PP大小的分析表明，在两个模型系列中都表现出强大的水平可扩展性，尽管扩展效率的程度因配置而异。

- **PP vs. TP**：从实验数据中观察到的一个关键现象是，当将张量并行（Tensor Parallelism，TP）扩展到16时，与混合并行方法（PP2 TP8）相比，出现了性能下降。尽管使用了相同的总GPU数量，PP2 TP8在吞吐量和延迟指标上始终优于PP1 TP16。此外，对于所有分块大小配置，PP4 TP8在吞吐量和延迟指标上也始终优于PP1 TP32。值得注意的是，在固定分块大小为12288的情况下，这种设置在PP4 TP8的所有分块策略设置中表现出最低的性能。然而，PP4 TP8的最差性能仍然显著优于PP1 TP32（其固定分块大小也等于12288），优势幅度为**18.4%**，尽管这种设置已经代表了测试的纯TP配置中的最佳性能。而使用动态分块时，这一优势幅度增加到**30.5%**。这些结果突显了PP方法的内在优势。
- **DCK的卓越可扩展性**：**Qwen DCK 18K**配置表现出最高的可扩展性，在PP8（32个GPU）时相比PP1（4个GPU）实现了**6.14×**&#x7684;加速因子。这一性能表明，分块大小的动态调整优化了计算强度与节点间通信延迟之间的平衡。
- **架构比较**：DeepSeek模型在达到PP4阈值之前，表现出与Qwen相当的可扩展轨迹。值得注意的是，**DeepSeek DCK 12K (3.31×)**&#x7565;微优于静态4K变体（3.20×），验证了动态分块（Dynamic Chunking）策略在提升吞吐量方面的跨架构鲁棒性。

![图7：DeepSeek-V3.1的吞吐量分析（越高越好）](/images/others/pipeline-parallelism-in-sglang-scaling-to-million-token-contexts-and-beyond/001-a1729821.png)

*图7：DeepSeek-V3.1的吞吐量分析（越高越好）*

![图8：Qwen3-235B-A22B-FP8的吞吐量分析（越高越好）](/images/others/pipeline-parallelism-in-sglang-scaling-to-million-token-contexts-and-beyond/008-dc3ef7af.png)

*图8：Qwen3-235B-A22B-FP8的吞吐量分析（越高越好）*

强扩展效率曲线说明了随着系统扩展，硬件利用率的下降（对于强扩展效率分析，ITL保持为128K，而增加，因此根据公式，气泡率下限肯定会更高）。所有配置在PP大小（GPU数量）增加时都表现出效率的单调衰减。然而，**Qwen DCK 18K**在PP8规模下仍保持**76.9%**&#x7684;优越效率，而静态6K配置则降至**69.6%**。这证实了更大、动态管理的分块对由流水线气泡引起的性能下降更具弹性。由于资源限制，DeepSeek-V3.1仅评估到PP大小=4，保持了**82.8%**&#x7684;效率。根据当前斜率推断，DeepSeek可能会遵循与Qwen类似的效率轨迹，其中DCK预计将优于固定分块策略。

![图9：强扩展效率与PP大小分析（越高越好）](/images/others/pipeline-parallelism-in-sglang-scaling-to-million-token-contexts-and-beyond/009-990c200f.png)

*图9：强扩展效率与PP大小分析（越高越好）*

### 降低TTFT及为100万ITL的扩展

从图10和图11中，我们可以观察到，将PP大小从PP1增加到PP4可以显著降低固定分块设置和动态分块的TTFT。但动态分块在不同的PP设置下表现更好。对于Qwen3-235B-A22B-FP8，基线TTFT&#x4E3A;**\~55.5s**（PP1 TP4），在PP8 TP4配置下减少&#x5230;**\~10.5s**，代表了约**81.1%**&#x7684;延迟改进。而对于DeepSeek-V3.1，基线TTFT&#x4E3A;**\~48.5s**（PP1 TP8），在PP4 TP8配置下减少&#x5230;**\~15.5s**，描绘了约**67.9%**&#x7684;延迟改进。这些结果表明，分块流水线并行（Chunked Pipeline Parallelism）对于降低TTFT非常有效。

![图10：DeepSeek-V3.1的TTFT分析（越低越好）](/images/others/pipeline-parallelism-in-sglang-scaling-to-million-token-contexts-and-beyond/010-57be5142.png)

*图10：DeepSeek-V3.1的TTFT分析（越低越好）*

![图11：Qwen3-235B-A22B-FP8的TTFT分析（越低越好）](/images/others/pipeline-parallelism-in-sglang-scaling-to-million-token-contexts-and-beyond/011-d233d1b9.png)

*图11：Qwen3-235B-A22B-FP8的TTFT分析（越低越好）*

为了展示SGLang与这种优化的分块流水线并行的可扩展性，我们针对Qwen3-235B-A22B-FP8在PP8（32个NVIDIA H20 GPU）配置下，对不同输入令牌长度的TTFT进行了基准测试。如下表所示，该系统能够高效扩展以处理大规模上下文。即使在**100万令牌**的极端边缘，SGLang在NVIDIA H20上仍保持高稳定性和可接受的延迟，展示了其应对最苛刻长上下文应用的能力。

表1：Qwen3-235B-A22B-FP8在H20上PP8 TP4配置的TTFT与输入令牌长度对比

| 输入令牌长度   | 128K  | 256K  | 512K   | 1M     |
| -------- | ----- | ----- | ------ | ------ |
| TTFT (s) | 10.54 | 32.68 | 114.33 | 420.91 |

利用比H20具有更高计算能力和带宽的硬件，或扩展到更多节点上的更大PP规模（例如，DeepSeek-V3.1模型的PP8 TP16），可以进一步降低百万令牌上下文的首令牌时间（TTFT）。我们邀请社区在各种硬件配置上尝试这一新功能。请分享您的性能发现并报告遇到的任何错误。我们很乐意听取您的意见——欢迎在[PP路线图](https://github.com/sgl-project/sglang/issues/11857)的问题中提出任何问题。您的反馈对于帮助我们完善这些长上下文优化至关重要！此外，请关注我们即将推出的CP × PP实现——DeepSeek-V3.2的初始支持已在主分支上可用。

## 入门指南

要利用这些功能，您只需配置`--pp-size`和`--chunked-prefill-size`。要进一步使用动态分块解决方案，请使用`--enable-dynamic-chunking`并设置环境变量`SGLANG_DYNAMIC_CHUNKING_SMOOTH_FACTOR`。

注意：所需的SGLang发布版本`>= v0.5.7`

示例：

```bash
# Example: Serving DeepSeek-V3.1 with 128K Input Token Length (32 GPUs total)
# Using 8-way Tensor Parallelism and 4-way Pipeline Parallelism

# prefill node 0 (fixed chunked prefill size)
python3 -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V3.1 --trust-remote-code \
  --nnodes 4 --node-rank 0 --tp 8 --pp-size 4 \
  --port 30000 --dist-init-addr <MASTER_NODE_IP> \
  --mem-fraction-static 0.8 --attention-backend fa3 \
  --host 0.0.0.0 --watchdog-timeout 3600 \
  --max-running-requests 128 --chunked-prefill-size 4096

# prefill node 0 (with dynamic chunking)
export SGLANG_DYNAMIC_CHUNKING_SMOOTH_FACTOR=0.65
python3 -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V3.1 --trust-remote-code \
  --nnodes 4 --node-rank 0 --tp 8 --pp-size 4 \
  --port 30000 --dist-init-addr <MASTER_NODE_IP> \
  --mem-fraction-static 0.8 --attention-backend fa3 \
  --host 0.0.0.0 --watchdog-timeout 3600 \
  --max-running-requests 128 --chunked-prefill-size 12288 --enable-dynamic-chunking

# Example: Serving Qwen3-235B-A22B-FP8 with 128K Input Token Length (32 GPUs total)
# Using 4-way Tensor Parallelism and 8-way Pipeline Parallelism

# prefill node 0 (fixed chunked prefill size)
python3 -m sglang.launch_server \
  --model-path Qwen/Qwen3-235B-A22B-FP8 --trust-remote-code \
  --nnodes 4 --node-rank 0 --tp 4 --pp-size 8 \
  --port 30000 --dist-init-addr <MASTER_NODE_IP> \
  --mem-fraction-static 0.8 --attention-backend fa3 \
  --host 0.0.0.0 --watchdog-timeout 3600 \
  --max-running-requests 128 --chunked-prefill-size 6144

# prefill node 0 (with dynamic chunking)
export SGLANG_DYNAMIC_CHUNKING_SMOOTH_FACTOR=0.8
python3 -m sglang.launch_server \
  --model-path Qwen/Qwen3-235B-A22B-FP8 --trust-remote-code \
  --nnodes 4 --node-rank 0 --tp 4 --pp-size 8 \
  --port 30000 --dist-init-addr <MASTER_NODE_IP> \
  --mem-fraction-static 0.8 --attention-backend fa3 \
  --host 0.0.0.0 --watchdog-timeout 3600 \
  --max-running-requests 128 --chunked-prefill-size 18432 --enable-dynamic-chunking
```

## 未来路线图：

我们正在不断完善PP堆栈。我们的2026年上半年PP路线图包括以下重要任务：

- 与上下文并行（Context Parallelism）兼容以进一步降低TTFT
- 解码端的流水线并行（Pipeline Parallelism）
  - 性能优化和最佳实践调优
- 动态分块的更好拟合和分块策略

**👉 查看[PP路线图](https://github.com/sgl-project/sglang/issues/11857)。**

## 结论

SGLang的流水线并行（Pipeline Parallelism）实现不仅仅是模型分割；它是为长上下文时代对推理生命周期的全面重新设计。通过结合分块预填充、异步通信和动态分块，SGLang为长上下文下服务和加速万亿参数模型提供了最高效且开源的道路。

## 致谢

- 我们要感谢SGLang团队和社区的实施和慷慨支持，特别是**Shangming Cai**、**Xuchun Shang**、**Yanbo Yang**、**Leon Gao**、**Ying Sheng**、Zhiqiang Xie、Lianmin Zheng以及许多其他人。
- 我们要感谢**Jianhao Fu**（来自AntGroup SCT网络团队）、**Kevin Li**（来自TikTok）、Siyu Liu（来自阿里云计算）、Xiaolei Zhang（来自字节跳动）、Teng Ma（来自阿里云计算）、Chao Wang（来自美团）和Xiaowei Wang（来自NVIDIA）在代码改进和测试方面的突出贡献。
- 我们从[SGLang](https://github.com/sgl-project/sglang)、Mooncake[\[1\]](https://dl.acm.org/doi/pdf/10.1145/3773772)和TeraPipe[\[3\]](http://proceedings.mlr.press/v139/li21y/li21y.pdf)的系统设计中学到了很多，它们共同帮助改进了这个流水线并行（Pipeline Parallelism）实现。

## 参考文献

\[1] Qin, Ruoyu, et al. "Mooncake: A kvcache-centric disaggregated architecture for llm serving." ACM Transactions on Storage (2024).\
\[2] Yang, An, et al. "Qwen2. 5-1m technical report." arXiv preprint arXiv:2501.15383 (2025).\
\[3] Li, Zhuohan, et al. "Terapipe: Token-level pipeline parallelism for training large-scale language models." International Conference on Machine Learning. PMLR, 2021.
