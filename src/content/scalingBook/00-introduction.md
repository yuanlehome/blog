---
title: "导论"
description: "训练大语言模型常常给人一种炼金术般的感觉，但理解和优化模型性能并非一定如此。本书旨在揭开语言模型扩展科学的神秘面纱：TPU（以及 GPU）如何工作、它们如何彼此通信，LLM 如何在真实硬件上运行，以及如何在训练和推理期间并行化模型，使其在超大规模下仍能高效运行。如果你曾想过“训练这个 LLM 应该有多贵”“我自己部署这个模型需要多少内存”或“什么是 AllGather”，我们希望本书能对你有所帮助。"
chapter: 0
order: 0
part: 0
partTitle: "导论"
sourcePath: "index.md"
sourceCommit: "44109cacac9c5a9809a81c68ae4d45d7d2632ea6"
---

<span id="introduction"></span>

# 导论

![](/images/scaling-book/img/dragon.png)

深度学习的许多部分至今仍像某种黑魔法，但模型性能优化并非如此——即使规模极其庞大也一样！从单个加速器到数以万计的加速器，相对简单的原理始终适用。理解这些原理后，你就能完成许多实用的事情：

- 粗略估计模型各部分距离理论最优性能还有多远。
- 在不同规模下，对不同并行方案（即如何把计算拆分到多个设备上）做出有依据的选择。
- 估算训练和运行大型 Transformer 模型所需的成本与时间。
- 设计能够利用[特定](https://arxiv.org/abs/2205.14135)[硬件](https://arxiv.org/abs/1911.02150)[特性](https://arxiv.org/abs/2007.00072)的算法。
- 在明确理解当前算法性能瓶颈的基础上设计硬件。

**预期背景：** 我们假设你对大语言模型（LLM）和 Transformer 架构有基本了解，但不一定熟悉它们如何在大规模系统上运行。你应当了解 LLM 训练的基础知识，最好还对 JAX 有一些基本认识。实用的背景读物包括介绍 Transformer 架构的[这篇博客文章](https://jalammar.github.io/illustrated-transformer/)和[原始 Transformer 论文](https://arxiv.org/abs/1706.03762)。你还可以查看[这份列表](../11-conclusion/#further-reading)，其中收录了更多适合同时或日后阅读的资料。

**目标与反馈：** 读完本书后，你应当能够较有把握地估算：对于给定硬件平台上的 Transformer 模型，哪种并行方案最好，以及训练和推理大致需要多长时间。如果你仍然做不到，请给我们发邮件或留言！我们非常希望知道怎样才能把这些内容讲得更清楚。

你或许也会喜欢新增的 NVIDIA GPU [第 12 章](../12-gpus/#how-to-think-about-gpus)！

<span id="why-should-you-care"></span>

### 为什么你应当关心这些？

三四年前，我认为大多数机器学习研究人员都不需要理解本书中的任何内容。但如今，即使是“小”模型也已如此接近硬件极限，以至于要开展创新研究，就必须考虑大规模系统的效率问题。[^ch0-1] **如果基准成绩提高 20% 的代价是 Roofline 效率下降 20%，那么这项提升就毫无意义。** 很有前景的模型架构常常失败，要么因为它们根本<em>无法</em>在大规模系统上高效运行，要么因为没有人愿意投入精力让它们高效运行。

**“模型扩展”的目标，是在增加训练或推理所用芯片数量时，让吞吐量也按比例线性增加。** 这称为“_强扩展_（strong scaling）”。增加芯片（即增加“并行度”）通常会缩短计算时间，但也会带来额外的芯片间通信。当通信耗时超过计算耗时时，系统便进入“通信受限”状态，无法继续实现强扩展。[^ch0-2] 如果我们对硬件有足够深入的理解，能预判这些瓶颈会在哪里出现，就可以通过设计或重新配置模型来避开它们。[^ch0-3]

*本书旨在解释 TPU（以及 GPU）硬件如何工作，以及 Transformer 架构为了在当代硬件上高效运行而经历了怎样的演变。我们希望，这既能帮助设计新架构的研究人员，也能帮助致力于让这一代 LLM 高速运行的工程师。*

<span id="high-level-outline"></span>

## 全书概览

本书的整体结构如下：

[第 1 章](../01-roofline/#all-about-rooflines)介绍 Roofline 分析，以及可能限制系统扩展能力的因素（通信、计算和内存）。[第 2 章](../02-tpus/#how-to-think-about-tpus)和[第 3 章](../03-sharding/#sharded-matrices-and-how-to-multiply-them)详细讨论 TPU 如何工作：既把它看作单独的芯片，也——这一点至关重要——把它看作由带宽和延迟都有限的芯片间链路连接起来的系统。我们将回答如下问题：

* 一次给定规模的矩阵乘法应当耗时多久？它在什么情况下受计算能力、内存带宽或通信带宽限制？
* TPU 如何互连成训练集群？系统各部分的带宽有多大？
* 在多个 TPU 之间聚合、分散或重新分布数组需要多长时间？
* 如何高效地将以不同方式分布在多个设备上的矩阵相乘？

![图：第 2 章中的示意图，展示 TPU 如何执行逐元素乘法。根据数组大小和不同链路的带宽，运算可能处于计算受限状态（充分使用硬件计算能力），也可能处于内存受限状态（瓶颈在于从内存加载数据）。](/images/scaling-book/img/pointwise-product.gif)

详见[第 2 章](../02-tpus/#how-to-think-about-tpus)。

五年前，机器学习领域还有丰富多彩的架构版图——ConvNet、LSTM、MLP、Transformer——但如今基本只剩下 Transformer[[transformers]](../#ref-transformers)。我们坚信，Transformer 架构的每个组成部分都值得深入理解：每个矩阵的确切大小、归一化发生的位置，以及各部分包含多少参数和 FLOPs[^ch0-4]。[第 4 章](../04-transformers/#all-the-transformer-math-you-need-to-know)将细致讲解这些“Transformer 数学”，说明如何计算训练与推理过程中的参数量和 FLOPs。由此，我们便能知道模型会占用多少内存、计算或通信分别会消耗多少时间，以及注意力相对于前馈块会在什么时候变得重要。

![图：标准 Transformer 层，其中每次矩阵乘法（matmul）都画成圆圈中的一个点。所有参数（归一化参数除外）均以紫色标出。第 4 章将更详细地讲解这张图。](/images/scaling-book/img/transformer-diagram.png)

详见[第 4 章](../04-transformers/#all-the-transformer-math-you-need-to-know)。

[第 5 章：训练](../05-training/#how-to-parallelize-a-transformer-for-training)和[第 7 章：推理](../07-inference/#all-about-transformer-inference)是本书的核心，我们将在其中讨论一个根本问题：给定某种规模的模型和一定数量的芯片，如何并行化模型，才能让系统保持在“强扩展”区间？这是一个看似简单、答案却出乎意料地复杂的问题。从高层来看，把模型拆分到多个芯片上主要有四种并行技术（**数据并行**、**张量并行**、**流水线并行**和**专家并行**），此外还有许多用于降低内存需求的技术（**重计算（rematerialization）**、**优化器/模型分片（即 ZeRO）**、**主机卸载**、**梯度累积**）。这些内容都会在相应章节中讨论。

我们希望，在读完这些章节后，面对新的架构或设置，你能够自行选择合适的方法。[第 6 章](../06-llama3-training/#training-llama-3-on-tpus)和[第 8 章](../08-llama3-inference/#serving-llama-3-70b-on-tpus)则是实践教程，将这些概念应用于广受欢迎的开源模型 LLaMA 3。

最后，[第 9 章](../09-profiling/#how-to-profile-tpu-programs)和[第 10 章](../10-jax/#programming-tpus-in-jax)介绍如何在 JAX 中实现其中一些思想，以及在出现问题时如何对代码进行性能剖析和调试。[第 12 章](../12-gpus/#how-to-think-about-gpus)是新增内容，将深入讨论 GPU。

全书各处都安排了练习题，供你亲自思考。你完全不必有压力，不需要读完所有章节，也不必按顺序阅读。还请留下反馈。目前本书仍是一份草稿，会继续修订。谢谢！

*在此特别感谢 James Bradbury 和 Blake Hechtman，他们推导出了本书中的许多思想。*

<span id="without-further-ado-here-is-section-1-about-tpu-rooflines"></span>

### 闲话少说，[第 1 章](../01-roofline/#all-about-rooflines)将从 TPU 的 Roofline 模型讲起。

<span id="links-to-sections"></span>

## 章节导航

*这一系列文章可能比实际需要的更长，但我们希望这不会让你望而却步。前三章是预备知识；如果你已经熟悉这些材料，可以跳过，不过其中会引入后续章节使用的记号。最后三部分或许最具实践价值，因为它们会说明如何处理真实模型。*

**第一部分：预备知识**

* [**第 1 章：Roofline 分析简介**](../01-roofline/#all-about-rooflines)。算法受三类因素限制：计算、通信和内存。我们可以据此近似估算算法的运行速度。

* [**第 2 章：如何理解 TPU**](../02-tpus/#how-to-think-about-tpus)。TPU 如何工作？这会怎样影响我们能够训练和部署推理服务的模型？

* [**第 3 章：分片矩阵及其乘法**](../03-sharding/#sharded-matrices-and-how-to-multiply-them)。本章借助我们最喜欢的运算——（分片）矩阵乘法——来解释模型分片和多 TPU 并行。

**第二部分：Transformer**

* [**第 4 章：你需要掌握的全部 Transformer 数学**](../04-transformers/#all-the-transformer-math-you-need-to-know)。Transformer 的前向传播和反向传播分别需要多少 FLOPs？你能计算它的参数量吗？KV 缓存有多大？本章将逐步推导这些计算。

* [**第 5 章：如何并行化 Transformer 训练**](../05-training/#how-to-parallelize-a-transformer-for-training)。FSDP、Megatron 分片、流水线并行：给定一定数量的芯片，如何用给定批大小尽可能高效地训练给定规模的模型？

* [**第 6 章：在 TPU 上训练 LLaMA 3**](../06-llama3-training/#training-llama-3-on-tpus)。如何在 TPU 上训练 LLaMA 3？需要多长时间？成本是多少？

* [**第 7 章：Transformer 推理详解**](../07-inference/#all-about-transformer-inference)。训练完成后，我们还需要部署模型。推理引入了一个新的考量因素——延迟——并改变了内存使用格局。本章将讨论解耦式推理服务如何工作，以及应如何理解 KV 缓存。

* [**第 8 章：在 TPU 上部署 LLaMA 3 推理服务**](../08-llama3-inference/#serving-llama-3-70b-on-tpus)。在 TPU v5e 上部署 LLaMA 3 推理服务需要多少成本？延迟与吞吐量之间有怎样的权衡？

**第三部分：实践教程**

* [**第 9 章：如何对 TPU 代码进行性能剖析**](../09-profiling/#how-to-profile-tpu-programs)。真实的 LLM 从来不像前面的理论那样简单。本章介绍 JAX + XLA 软件栈，以及如何使用 JAX/TensorBoard 性能剖析器调试和修复真实问题。

* [**第 10 章：使用 JAX 为 TPU 编程**](../10-jax/#programming-tpus-in-jax)。JAX 提供了一系列仿佛魔法般的计算并行 API，但你需要知道如何使用它们。本章包含有趣的示例和附有解答的练习题。

**第四部分：总结与附加内容**

* [**第 11 章：总结与延伸阅读**](../11-conclusion/#conclusions-and-further-reading)。关于 TPU 和 LLM 的总结思考与延伸阅读。

* [**第 12 章：如何理解 GPU**](../12-gpus/#how-to-think-about-gpus)。介绍 GPU 的附加章节，讨论 GPU 的工作方式、互连方式，以及它的 Roofline 模型与 TPU 有何不同。

[^ch0-1]: 从历史上看，机器学习研究大体遵循着系统创新与软件改进交替推进的“滴答”周期。Alex Krizhevsky 当年不得不编写堪称邪术的 CUDA 代码来加速 CNN，但几年之内，Theano 和 TensorFlow 等库便让你不再需要这样做。也许这里同样会发生这种变化，几年后，本书中的所有内容都会被抽象层完全隐藏起来。不过，规模定律不断把模型推向硬件能力的最前沿；至少在可预见的未来，前沿研究很可能仍与高效地把模型扩展到大型硬件拓扑这一能力密不可分。
[^ch0-2]: 随着计算时间缩短，你通常也会在单芯片层面遇到瓶颈。崭新的 TPU 或 GPU 也许号称每秒可执行 500 万亿次运算，但如果不够谨慎，让它被内存中的参数搬运拖住，它同样可能只能发挥十分之一的性能。单芯片计算能力、内存带宽和总内存容量之间的相互作用，是理解扩展问题的关键。
[^ch0-3]: 硬件设计者面临的恰好是反问题：在尽量降低成本的同时，为算法提供刚刚足够的计算能力、带宽和内存。可以想象，这种“协同设计”问题会带来多大压力：你必须押注在首批芯片真正可用时——往往是两三年后——算法会是什么样子。TPU 的发展堪称这场博弈中的巨大成功。矩阵乘法是一种独特的算法，它每搬运一个字节所执行的 FLOPs（每字节 N FLOPs）远多于几乎所有其他算法；早期 TPU 及其脉动阵列架构在当时实现了远优于 GPU 的单位成本性能。TPU 是为机器学习工作负载设计的，而配备 Tensor Core 的 GPU 也正在迅速演进以填补这一领域。但可以想象：如果神经网络没有兴起，或者发生了某种根本变化，使本就不如 GPU 灵活的 TPU 无法应对，代价会有多么高昂。
[^ch0-4]: FLOPs 是浮点运算次数（FLoating point OPerations），基本上就是所需加法和乘法的总次数。许多资料把 FLOPs 当作“每秒运算次数”，而本书会明确使用 FLOPs/s 表示速率。
