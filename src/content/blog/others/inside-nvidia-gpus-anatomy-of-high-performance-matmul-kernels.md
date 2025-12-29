---
title: 'Inside NVIDIA GPUs: Anatomy of high performance matmul kernels'
slug: inside-nvidia-gpus-anatomy-of-high-performance-matmul-kernels
date: '2025-12-29'
tags: []
status: published
source_url: 'https://www.aleksagordic.com/blog/matmul'
source_author: www.aleksagordic.com
imported_at: '2025-12-29T11:48:48.995Z'
source:
  title: www.aleksagordic.com
  url: 'https://www.aleksagordic.com/blog/matmul'
cover: >-
  /images/others/inside-nvidia-gpus-anatomy-of-high-performance-matmul-kernels/001-8ee6a76e.png
lang: zh
translatedFrom: en
---

# 深入NVIDIA GPU：高性能矩阵乘法（matmul）内核的剖析

## 从GPU架构和PTX/SASS到warp-tiling和深度异步张量核心（tensor core）流水线

2025年9月29日

在这篇文章中，我将逐步介绍支撑最先进（SOTA）NVIDIA GPU矩阵乘法（matmul）内核的所有核心硬件概念和编程技术。

**为什么关注矩阵乘法（matmul）？**&#x54;ransformer在训练和推理过程中，大部分浮点运算（FLOPs）都发生在矩阵乘法（matmul）中（如MLP中的线性层、注意力QKV投影、输出投影等）。这些操作具有极高的并行性，天然适合GPU。最后，理解矩阵乘法（matmul）内核的工作原理，能为你提供设计几乎所有其他高性能GPU内核的工具包。

本文分为四个部分：

1. [NVIDIA GPU架构基础](#cpt1)：全局内存（global memory）、共享内存（shared memory）、L1/L2缓存，以及功率限制对SOL的影响等。
1. [GPU汇编语言](#cpt2)：SASS和PTX
1. [设计接近SOTA的同步矩阵乘法（matmul）内核](#cpt3)：warp-tiling方法
1. [在Hopper上设计SOTA异步矩阵乘法（matmul）内核](#cpt4)：利用张量核心（tensor cores）、TMA、计算与加载/存储的重叠、Hilbert曲线等。

我的目标是让这篇文章自成一体：足够详细以独立存在，又足够简洁以避免成为教科书。

这是更广泛系列的第一部分。在后续文章中，我（理想地）计划涵盖：

- 在Blackwell GPU上设计SOTA矩阵乘法（matmul）内核
- 通过微基准测试实验探索GPU架构
- 设计SOTA多GPU内核
- 揭秘内存一致性模型（GPU中的tokenizer等价物：默默支撑系统运行但令大多数开发者困惑的关键组件）

## NVIDIA GPU架构基础

要编写高性能GPU内核，你需要对硬件有坚实的心理模型。随着我们深入硬件架构，这一点将很快变得清晰。

在本文中，我专注于Hopper H100 GPU。如果你深入理解Hopper，将知识适配到未来架构（Blackwell、Rubin）或早期架构（Ampere、Volta）会变得直接。

[Hopper](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/) [\[1\]](#ref-1)和[Ampere](https://developer.nvidia.com/blog/nvidia-ampere-architecture-in-depth/) [\[2\]](#ref-2)白皮书是很好的信息来源。

在最高层面，GPU执行两个基本任务：

1. 移动和存储数据（内存系统）
1. 对数据进行有用工作（计算流水线）

下面的H100框图反映了这种划分：蓝色组件代表内存或数据移动，而红色组件是计算（热）单元。

<figure>
  <img src="/images/others/inside-nvidia-gpus-anatomy-of-high-performance-matmul-kernels/001-8ee6a76e.png" alt="Figure 1: Model of the NVIDIA Hopper H100 GPU" />
  <figcaption>图1：NVIDIA Hopper H100 GPU模型</figcaption>
</figure>

如果你发现文章中有任何错误，请私信我——欢迎在[X](https://x.com/gordic_aleksa)或[LinkedIn](https://www.linkedin.com/in/aleksagordic/)上给我留言，或通过[匿名反馈](https://docs.google.com/forms/d/1z1fEirrN2xtGxAsJvptpM7yV4ByT5SF25S-XiMPrXNA/edit)。

## 内存

GPU中的内存系统是高度分层的，类似于CPU架构。

这种层次结构由物理和电路设计决定：SRAM单元更快但更大（实现其速度的控制电路也增加了面积），而DRAM单元更小/更密集但更慢。结果是快速内存容量较小且昂贵，而较慢的内存可以提供更大的容量。我们将在后面更详细地介绍DRAM单元/内存。

这种容量与延迟之间的权衡正是缓存层次结构存在的原因。在理想世界中，每个计算单元旁边都会有一大池超快内存。由于这在物理上不可能，GPU设计者妥协：将少量快速内存放置在计算单元附近，并由逐渐更大、更慢的内存池支持。这种组织方式最大化整体系统吞吐量。

GPU内存系统包括：

1. **设备内存（device memory）**（VRAM）。在CUDA术语中，“设备”内存指片外DRAM——物理上与GPU芯片分离但封装在同一板上——实现为堆叠HBM。它承载全局内存（GMEM）、每线程“本地”内存（寄存器溢出空间）等。

1. **L2缓存**。一个大型的k路组关联SRAM缓存。它物理上分为两部分；每个SM直接连接到一个分区，并通过交叉开关间接连接到另一个分区。

1. **分布式共享内存（DSMEM）**。物理上接近的一组SM（一个GPC）的池化共享内存（SMEM）。

1. L1缓存和共享内存
   1. **L1缓存**。一个较小的k路组关联SRAM缓存，专用于每个SM。
   1. **共享内存（SMEM）**。程序员管理的片上内存。SMEM和L1共享相同的物理存储，它们的相对分割可以在软件中配置。

1. **寄存器文件（RMEM）**。最快的存储，位于计算单元旁边。寄存器是各个线程私有的。与CPU相比，GPU包含更多寄存器，总RMEM容量与L1/SMEM存储的总和大小相同。

<figure>
  <img src="/images/others/inside-nvidia-gpus-anatomy-of-high-performance-matmul-kernels/002-a9f61b85.png" alt="Figure 2: Memory hierarchy of the H100 (SXM5) GPU" />
  <figcaption>图2：H100（SXM5）GPU的内存层次结构</figcaption>
</figure>

📝注意：

还有其他一些较小的指令缓存，以及常量内存等，我将忽略它们，因为它们对我们的理解不重要。

从设备内存向下到寄存器（级别1-5），你可以看到一个清晰的趋势：带宽按数量级增加，而延迟和容量按类似数量级减少。

由此得出几个直接含义：

1. 将最频繁访问的数据尽可能靠近计算单元。
1. 最小化对层次结构较低级别的访问，尤其是设备内存（GMEM）。

另一个值得注意的组件是**张量内存加速器（TMA）**，随Hopper引入。TMA支持全局内存和共享内存之间以及集群内共享内存之间的异步数据传输。它还支持swizzling以减少bank冲突——我们将适时（双关语）介绍这些细节。

## 计算

从内存切换到计算，基本单位是**流式多处理器（SM）**. Hopper H100（SXM5）总共集成了132个流式多处理器（SMs）。

流式多处理器（SMs）被分组为图形处理集群（GPCs）：每个GPC包含18个SMs，GPU上有8个GPCs。四个GPCs直接连接到一个L2分区，另外四个连接到第二个分区。

📝备注：

GPC也是支撑CUDA中线程块集群（thread-block cluster）抽象的硬件单元——我们稍后会回到编程模型。

关于集群的一个要点：之前我说每个GPC有18个SMs，所以有8个GPCs时，我们预计有144个SMs。但SXM/PCIe外形规格暴露了132或114个SMs。差异在哪里？这是因为18×8的布局仅适用于完整的GH100芯片——在实际产品中，一些SMs被熔断关闭。这直接影响我们编写内核时如何选择集群配置。例如，你不能使用所有SMs来创建跨越超过2个SMs的集群。

最后，请注意图形处理集群（GPC）中的“图形”是一个遗留术语。在现代服务器级GPU中，这些集群纯粹用作计算/AI加速单元，而不是图形引擎。GPU也是如此，去掉G，它们就是AI加速器。

除了已经提到的L1/SMEM/TMA/RMEM组件（所有这些都物理上位于SM内），每个SM还包含：

1. **张量核心（Tensor Cores）。** 执行小矩阵块（例如，`64x16 @ 16x256`）上矩阵乘法的专用单元，具有高吞吐量。大型矩阵乘法被分解为许多这样的矩阵块操作，因此有效利用它们对于达到峰值性能至关重要。
1. **CUDA核心和SFUs。** 所谓的“CUDA核心”（营销术语）执行标准浮点运算，如FMA（融合乘加：`c = a * b + c`）。特殊功能单元（SFUs）处理超越函数，如`sin`、`cos`、`exp`、`log`，但也处理代数函数，如`sqrt`、`rsqrt`等。
1. **加载/存储（LD/ST）单元。** 服务加载和存储指令的电路，与TMA引擎互补。
1. **线程束调度器（Warp schedulers）。** 每个SM包含调度器，为32个线程的组（在CUDA中称为线程束）发出指令。一个线程束调度器每个周期可以发出一个线程束指令。

每个SM物理上分为四个象限，每个象限容纳上述计算单元的一个子集。

这引出了以下见解：

📝并行性与并发性

一个SM最多可以同时发出四个线程束的指令（即，在给定周期内，真正并行执行128个线程）。

然而，一个SM可以容纳多达2048个并发线程（64个线程束）。这些线程束驻留并随时间调度进出，允许硬件隐藏内存/流水线延迟。

换句话说，指令并行性（有多少线程在给定周期开始执行指令）限制为每个SM一次128个线程（4个32宽的线程束指令），而并发性（调度器中跟踪并符合运行条件的线程数）扩展到2048个线程。

## 光速与功率节流

既然我们购买NVIDIA GPU是为了计算，很自然地会问：天花板是什么——GPU的最大计算吞吐量？这通常被称为“光速”（SoL）性能：由芯片物理特性决定的上限。

根据数据类型有多个天花板。在LLM训练工作负载中，bfloat16（`bf16`）近年来一直是主导格式，尽管`fp8`和4位格式变得越来越重要（对于推理，fp8相当标准）。

峰值吞吐量计算为：`perf = freq_clk_max * num_tc * flop_per_tc_per_clk`

或用文字表述：最大时钟频率 × 张量核心数量 × 每个张量核心每个周期的FLOPs。

<figure>
  <img src="/images/others/inside-nvidia-gpus-anatomy-of-high-performance-matmul-kernels/003-9cf3af00.png" alt="Figure 3: H100 SXM5 BF16 speed-of-light derivation" />
  <figcaption>图3：H100 SXM5 BF16光速推导</figcaption>
</figure>

📝FLOP vs FLOPs vs FLOPS vs FLOP/s

- FLOP = 单个浮点运算。
- FLOP/s = 吞吐量单位：每秒浮点运算次数。
- FLOPs（小写s）= FLOP的复数形式（运算）。
- FLOPS（全大写）常被误用表示吞吐量，但严格来说应仅读作“FLOPs”（FLOP的复数）。FLOPS用作FLOP/s是马虎的！ :)

我在上图留下了一个提示：“光速”实际上不是恒定的（我猜这是类比失效的地方）。

在实践中，峰值吞吐量取决于实际时钟频率，这可能在功率或热节流下变化。如果GPU时钟下降，有效光速也会下降：

<figure>
  <img src="/images/others/inside-nvidia-gpus-anatomy-of-high-performance-matmul-kernels/004-821585f3.png" alt="Figure 4: Power throttling reduces clock frequency and lowers the effective “speed of light”" />
  <figcaption>图4：功率节流降低时钟频率并降低有效“光速”</figcaption>
</figure>

📝进一步阅读：

Horace He在他的[博客文章](https://www.thonking.ai/p/strangely-matrix-multiplications) [\[3\]](#ref-3)中更深入地探讨了这一现象。

这就是我们目前需要的硬件细节。

接下来，我们将把重点转向CUDA编程模型，然后深入一层硬件，最终上升到CUDA C++领域。

## CUDA编程模型

CUDA编程模型自然地映射到GPU硬件和内存层次结构。

关键抽象是：

1. 线程（thread）
1. 线程束（warp，32个线程）
1. 线程块（thread block）
1. 线程块集群（thread block cluster）
1. 网格（grid，线程块或集群的集合）

<figure>
  <img src="/images/others/inside-nvidia-gpus-anatomy-of-high-performance-matmul-kernels/005-0d49d664.png" alt="Figure 5: CUDA Programming Model: threads, warps, blocks, clusters, grids" />
  <figcaption>图5：CUDA编程模型：线程、线程束、块、集群、网格</figcaption>
</figure>

每个线程通过变量如`gridDim`、`blockIdx`、`blockDim`和`threadIdx`“感知”其在CUDA层次结构中的位置。内部，这些存储在特殊寄存器中，并在内核启动时由CUDA运行时初始化。

这种位置信息使得在GPU上分配工作变得容易。例如，假设我们要处理一个1024×1024的图像。我们可以将其划分为32×32的线程块，每个块包含32×32的线程排列。

然后每个线程可以计算其全局坐标，例如

```c
const int x = blockIdx.x * blockDim.x + threadIdx.x
const int y = blockIdx.y * blockDim.y + threadIdx.y
```

，并使用这些坐标从全局内存（`image[x][y]`）中获取其分配的像素，执行一些逐点操作，并将结果存储回去。

以下是这些变量如何相互关联：

<figure>
  <img src="/images/others/inside-nvidia-gpus-anatomy-of-high-performance-matmul-kernels/006-a3c6cd90.png" alt="Figure 6: CUDA's built-in variables: how threads know where they are" />
  <figcaption>图6：CUDA内置变量：线程如何知道它们的位置</figcaption>
</figure>

如图中所述，在实践中我们大多使用1D或2D网格/集群/块形状。但内部，它们总是可以根据需要逻辑重组。

例如，如果`threadIdx.x`从0到1023运行（一个1024个线程的1D块），我们可以将其拆分为`x = threadIdx.x % 32`和`y = threadIdx.x / 32`，有效地将块重塑为32×32的逻辑二维布局。

将CUDA模型连接回硬件，现在应该清楚一个事实：**一个线程块应至少包含4个warp（即128个线程）。**

为什么？

1. 一个线程块驻留在单个SM上。
1. 每个SM有4个warp调度器——因此为了充分利用硬件，您不希望它们闲置。

📝更多关于4个warp的原因：

我们稍后将深入探讨这一点，但请注意，在Hopper架构上，warp组（4个warp）是WGMMA（矩阵乘法）张量核心指令的执行单元。

此外，对于持久内核，我们通常每个SM只启动一个线程块，因此重要的是结构化工作，以保持所有warp调度器忙碌。

掌握了CUDA编程模型的术语后，我们现在可以继续深入GPU的架构。

## GMEM模型

让我们深入探讨GMEM。如前所述，它被实现为一堆DRAM层，底部有一个逻辑层（HBM）。但DRAM到底是什么？

<figure>
  <img src="/images/others/inside-nvidia-gpus-anatomy-of-high-performance-matmul-kernels/007-0c5a4437.png" alt="Figure 7: Inside a DRAM cell: transistor + capacitor, wordline + bitline" />
  <figcaption>图7：DRAM单元内部：晶体管+电容器，字线+位线</figcaption>
</figure>

现在我们理解了单个位是如何存储的，让我们放大到整个内存矩阵。从高层次看，它看起来像这样：

<figure>
  <img src="/images/others/inside-nvidia-gpus-anatomy-of-high-performance-matmul-kernels/008-dbb04e9b.png" alt="Figure 8: GMEM model" />
  <figcaption>图8：GMEM模型</figcaption>
</figure>

📝关于HBM的进一步阅读：

如果您想更深入地了解HBM，我发现论文[“Demystifying the Characteristics of High Bandwidth Memory for Real-Time Systems”](https://upcommons.upc.edu/server/api/core/bitstreams/b843de39-f32f-4069-8843-48f74c030213/content) [\[21\]](#ref-21)相当有信息量。

因此我们得出结论：访问模式很重要，因为DRAM单元的物理特性。这里是一个例子：

<figure>
  <img src="/images/others/inside-nvidia-gpus-anatomy-of-high-performance-matmul-kernels/009-9ffead5b.png" alt="Figure 9: Effect of access pattern in GMEM" />
  <figcaption>图9：GMEM中访问模式的影响</figcaption>
</figure>

Stephen Jones的演讲[“How CUDA Programming Works”](https://www.nvidia.com/en-us/on-demand/session/gtcfall22-a41101/) [\[4\]](#ref-4)值得一看。

如果我们示例中的矩阵是列主序，情况会反转：列中的元素将连续存储，因此高效的选择是在内循环中遍历行以避免DRAM惩罚。

所以当人们说“GMEM合并非常重要”时，这就是他们的意思：线程应访问连续的内存位置，以最小化触及的DRAM行数。

接下来，让我们将注意力转向SMEM的工作原理。

## SMEM模型

共享内存（SMEM）具有**非常**不同于GMEM的特性。它由SRAM单元而非DRAM构建，这赋予了它根本不同的速度和容量权衡。

SRAM单元的确切设计并不重要——足以说存储单个位信息需要更多晶体管。请随意搜索“SRAM cell”。

SMEM被组织成32个bank，每个bank宽32位（4字节）：

<figure>
  <img src="/images/others/inside-nvidia-gpus-anatomy-of-high-performance-matmul-kernels/010-349393f4.png" alt="Figure 10: SMEM model" />
  <figcaption>图10：SMEM模型</figcaption>
</figure>

SMEM可以在单个周期内从所有32个bank（128B）提供数据——但前提是遵守一个规则：

**一个warp中的线程不能访问同一bank内的不同地址。否则，这些请求将在多个周期内串行化。**

这种情况被称为**bank冲突**。如果N个线程访问同一bank的不同地址，结果是N路bank冲突，warp的内存请求需要N个周期完成。

在最坏情况下，所有32个线程针对同一bank的不同地址，吞吐量下降32倍。

为了说明，假设warp大小为5。以下两种访问模式将分别需要3个周期和1个周期来服务：

<figure>
  <img src="/images/others/inside-nvidia-gpus-anatomy-of-high-performance-matmul-kernels/011-651002e5.png" alt="Figure 11: SMEM: good vs. bad access patterns" />
  <figcaption>图11：SMEM：良好与不良访问模式</figcaption>
</figure>

重要的是：如果warp中的多个线程访问同一bank内的相同地址，SMEM可以广播（或多播）该值给所有线程。

在下面的示例中，请求在单个周期内服务：

- Bank 1可以向2个线程多播一个值。
- Bank 2可以向3个线程多播一个值。

<figure>
  <img src="/images/others/inside-nvidia-gpus-anatomy-of-high-performance-matmul-kernels/012-2c95f4f4.png" alt="Figure 12: SMEM: multicasting (served in a single cycle)" />
  <figcaption>图12：SMEM：多播（在单个周期内服务）</figcaption>
</figure>

现在，对于硬件拼图的最后一块：L1缓存。

这是Axel关于SMEM微基准测试的一篇优秀[博客文章](https://feldmann.nyc/blog/smem-microbenchmarks) [\[5\]](#ref-5)。

## L1模型

我们已经看到L1和SMEM共享相同的物理存储，但L1在该存储周围添加了一个硬件管理的脚手架层。

从高层次看，L1缓存的逻辑流程是：

1. 一个warp发出内存请求（到SMEM或GMEM）。
1. 请求进入MIO管道并被分派到LSUIN路由器。
1. 路由器引导请求：SMEM访问立即从数据数组服务，而GMEM访问进入标签比较阶段。
1. 在标签阶段，GMEM地址标签与存储在目标集中的标签进行比较，以确定数据是否驻留在L1中。
1. 在**命中**时，请求直接从数据数组服务（就像SMEM一样）。
1. 在**未命中**时，请求传播到L2（以及更远，如有必要，直到GMEM或对等GPU内存）。当数据返回时，它被缓存在L1中，驱逐现有行，并并行发送回请求warp。

这是我刚刚描述的系统：

<figure>
  <img src="/images/others/inside-nvidia-gpus-anatomy-of-high-performance-matmul-kernels/013-8e982c04.png" alt="Figure 13: L1 cache model" />
  <figcaption>图13：L1缓存模型</figcaption>
</figure>

让我们深入一层，详细查看标签阶段和数据阶段：

<figure>
  <img src="/images/others/inside-nvidia-gpus-anatomy-of-high-performance-matmul-kernels/014-c6c094a8.png" alt="Figure 14: Breakdown of k-way set-associative cache organization" />
  <figcaption>图14：k路组相联缓存组织的分解</figcaption>
</figure>

当GMEM地址进入标签阶段时，命中/未命中逻辑展开如下：

1. 标签阶段接收GMEM地址。

1. 提取集合ID位，并检查该集合中的所有缓存行（标签）。

1. 如果找到标签匹配（潜在缓存命中）：
   - 检查行的有效性标志。
     - 如果无效→视为缓存未命中（继续到步骤4）。
     - 如果有效→从数据数组获取请求的扇区并交付到warp的寄存器。

1. 如果未找到匹配（缓存未命中），请求被路由到内存层次结构的其余部分（L2及更远）。
   - 当数据从L2返回时，它被存储在集合中，根据替换策略（例如，伪LRU）驱逐现有行，并并行交付到请求warp。

请注意，L2与L1没有太大不同，除了它是全局的（相对于每SM），更大（具有更高的相联度），分成两个通过交叉开关连接的切片，并支持更细微的持久性和缓存策略。

至此，我们已经涵盖了理解后续部分所需的关键GPU硬件组件。

📝GPU代际间的梯度：

我之前提到，理解Hopper（Hopper）是理解NVIDIA GPU未来和过去世代的绝佳基础。

迄今为止最大的世代跃迁是从Ampere（安培）到Hopper（Hopper），引入了：

- 分布式共享内存（DSMEM，Distributed Shared Memory）：用于在整个GPC的SMEM之间进行直接SM到SM通信，支持加载、存储和原子操作。
- TMA（Tensor Memory Accelerator）：用于异步张量数据移动（GMEM ↔ SMEM，SMEM ↔ SMEM）的硬件单元。
- 线程块集群（Thread Block Clusters）：一种新的CUDA编程模型抽象，用于跨SM分组块。
- 异步事务屏障（Asynchronous transaction barriers）：拆分屏障，计数事务（字节）而不仅仅是线程。

Ampere（安培）（例如A100）本身引入了几个关键特性：

- Tensor Core（张量核心）中的tf32和bf16支持。
- 异步复制（GMEM → SMEM），具有两种模式：绕过L1和访问L1。
- 异步屏障（在共享内存中硬件加速）。
- CUDA任务图（CUDA task graphs），支撑PyTorch中的CUDA图，并减少CPU启动和网格初始化开销。
- 通过CUDA Cooperative Groups（CUDA协作组）暴露的warp级归约指令（支持warp范围内、整数数据类型的单步归约，无需shuffle模式）。

## GPU汇编语言：PTX和SASS

让我们向上移动一层到硬件之上的ISA（指令集架构）。ISA简单来说就是处理器（例如NVIDIA GPU）可以执行的指令集，包括其二进制编码（操作码、操作数等）和行为语义。这些共同定义了程序员如何指导硬件执行有用工作。

ISA的人类可读形式被称为**汇编**：程序员使用助记符如`0x1fff…3B`，而不是像`FMA R12, R13, R14, R15`那样编写原始二进制。

在NVIDIA GPU上，原生ISA称为SASS。不幸的是，它的文档很少——尤其是对于最新的GPU世代。一些较旧的世代已被部分或完全逆向工程，但官方文档仍然有限。您可以在此处[here](https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html) [\[6\]](#ref-6)找到文档。

PTX是NVIDIA的**虚拟ISA：**&#x4E00;种抽象GPU的指令集。PTX代码不直接执行；而是由`ptxas`编译为原生ISA（SASS）。

PTX的关键优势是前向兼容性。十年前编译为PTX的CUDA程序仍然可以在现代GPU如Blackwell（布莱克韦尔）上运行。它可能无法高效利用最新的硬件特性，但会正确执行。

这是因为PTX嵌入到CUDA二进制文件中，与原生SASS一起。当二进制文件在未来GPU上运行时，如果匹配的SASS代码不存在，PTX会被JIT编译为目标架构的SASS：

<figure>
  <img src="/images/others/inside-nvidia-gpus-anatomy-of-high-performance-matmul-kernels/015-aa3bd47e.png" alt="Figure 15: CUDA compilation flow: from CUDA C++ → PTX → SASS" />
  <figcaption>图15：CUDA编译流程：从CUDA C++ → PTX → SASS</figcaption>
</figure>

为什么关心PTX/SASS？

因为这是可以找到最后几个百分点性能的地方。在今天的规模下，这些“几个百分点”是巨大的：如果您在30,000个H100上训练LLM，即使将核心内核性能提高1%，也能节省数百万美元。

正如我的朋友[Aroun](https://github.com/ademeure)喜欢说的：在编写大规模训练/推理内核时，我们关心`O(NR)`，而不是`O(N)`。（这里，NR = 核反应堆。）换句话说，可能没有新的渐近复杂度类等待被发现——大的收益（大部分）已经消失。但在数百万GPU上挤出约1%的效率，相当于节省几个SMR（小型模块化反应堆）的能量。

要深入了解SASS，我推荐Aroun的["Introduction to SASS & GPU Microarchitecture"](https://www.youtube.com/watch?v=we3i5VuoPWk) [\[7\]](#ref-7)视频。

理解SASS并不意味着您会开始直接用SASS编写CUDA内核。相反，在编写CUDA C++时，您希望紧密耦合到编译器的输出（PTX/SASS）。这让您可以双重检查您的提示（例如`#pragma unroll`用于展开循环，或向量化加载）是否确实被降低为预期的指令（例如`LDG.128`）。

这些低级细节中隐藏性能的一个很好例子来自现在著名的Citadel论文，["Dissecting the NVIDIA Volta GPU Architecture via Microbenchmarking"](https://arxiv.org/abs/1804.06826) [\[8\]](#ref-8)。作者调整SASS以避免内存bank冲突，并将性能从132 GFLOP/s提升到152 GFLOP/s——提高了15.4%。

还要注意，一些指令在CUDA C++中没有等效物；您只需编写内联PTX！我们将在第4章后面看到这方面的例子。

现在（希望）我已经说服您PTX/SASS很重要，让我们介绍最简单的matmul内核，它将作为本章其余部分的运行示例。之后我们将深入分析其汇编。

让我们从最简单的情况开始：针对“串行处理器”如CPU的朴素矩阵乘法内核：

```c
for (int m = 0; m < M; m++) {
    for (int n = 0; n < N; n++) {
        float tmp = 0.0f;  // accumulator for dot product
        for (int k = 0; k < K; k++) {
            tmp += A[m][k] * B[k][n];  // A and B are input matrices
        }
        C[m][n] = tmp;  // C is the output matrix
    }
}
```

我们循环遍历输出矩阵（`m`）的行（`n`）和列（`C`），并在每个位置计算点积（`C[m,n] = dot(a[m,k],b[k,n])`）。这是matmul的教科书定义：

<figure>
  <img src="/images/others/inside-nvidia-gpus-anatomy-of-high-performance-matmul-kernels/016-6bf0d586.png" alt="Figure 16: Naive CPU matmul example" />
  <figcaption>图16：朴素CPU matmul示例</figcaption>
</figure>

总共，矩阵乘法需要`M × N`个点积。每个点积执行`K`次乘加，所以总工作量是`2 × M × N × K` FLOPs（因子2是因为，按照惯例，我们计数FMA = 乘 + 加）。

并行性在哪里？

所有这些点积都是独立的。没有理由计算`C[0,1]`应该等待`C[0,0]`。这种独立性意味着我们可以跨两个外部循环（遍历`m`和`n`）并行化。

有了这个见解，让我们看看最简单的GPU内核。我们将使用一个稍微更通用的形式：`C = alpha * A @ B + beta * C`。这是经典的GEMM（通用矩阵乘法）。设置`alpha = 1.0`和`beta = 0.0`恢复为更简单的`C = A @ B`。

内核代码：

```c
// __global__ keyword declares a GPU kernel
__global__ void naive_kernel(int M, int N, int K, float alpha,
                                          const float *A, const float *B,
                                          float beta, float *C) {
  int BLOCKSIZE=32;

  const int row = blockIdx.x * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
  const int col = blockIdx.y * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);

  if (row < M && col < N) {  // guard in case some threads are outside the range
    float tmp = 0.0;
    // compute dot product
    for (int i = 0; i < K; ++i) {
      tmp += A[row * K + i] * B[i * N + col];
    }
    // GEMM: C = alpha * A @ B + beta * C
    C[row * N + col] = alpha * tmp + beta * C[row * N + col];
  }
}
```

我们这样启动它：

```sql
// create as many blocks as necessary to map all of C
dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32), 1);
// 32 * 32 = 1024 thread per block
dim3 blockDim(32 * 32);
// launch the asynchronous execution of the kernel on the device
// the function call returns immediately on the host
naive_kernel<<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
```

您可以在这里观察到几点：

- 内核是从单个线程的角度编写的。这遵循SIMT（单指令多线程）模型：程序员编写一个线程的工作，而CUDA处理网格、集群和块的启动和初始化。（其他编程模型，如OpenAI的[Triton](https://github.com/triton-lang/triton) [\[22\]](#ref-22)，让您从**tile**的角度编写。）
- 每个线程使用其块和线程索引（我们之前讨论的变量）来计算其在`row`中的（`col`）坐标。`C`并写出对应的点积。
- 我们使用尽可能多的32×32线程块（1024个线程）来平铺输出矩阵。
- 如果`M`或`N`不能被32整除，一些线程会落在有效输出区域之外。`C`这就是为什么我们在代码中包含一个防护条件。

最后两点结合导致了一个通常被称为**平铺量化（tile quantization）的现象：**

<figure>
  <img src="/images/others/inside-nvidia-gpus-anatomy-of-high-performance-matmul-kernels/017-20e073fb.png" alt="Figure 17: Tile quantization" />
  <figcaption>图17：平铺量化</figcaption>
</figure>

当平铺相对于输出矩阵较大时，这种效应尤其明显。在我们的例子中，由于32能整除4096，所以没有问题。但如果矩阵大小是，比如33×33，那么大约75%的线程最终会做无用功。

代码本可以通过传递2D块而不是1D块来更简单地编写。那样的话，我们就不需要硬编码块大小为32，并且可以使用`threadIdx.x`和`threadIdx.y`。在内部，1D结构通过索引算术有效地转换为2D：`threadIdx.x / BLOCKSIZE`和`threadIdx.x % BLOCKSIZE`，所以在实践中没有太大区别。

我最初从[Simon的博客（Simon's blog）](https://siboehm.com/articles/22/CUDA-MMM) [\[9\]](#ref-9)改编了这段代码，并专注于对其进行深入的PTX/SASS分析（即将到来），所以我不想重复这些辛苦工作，因为轻微的代码更改会导致不同的PTX/SASS。

让我们更仔细地看看这个内核实际上做了什么。在本文的其余部分，我们将假设`M = N = 4096`。本示例中的所有矩阵都是行主序格式（在一些后续示例中，`B`将是列主序——标准约定）。

线程的逻辑组织如下：

<figure>
  <img src="/images/others/inside-nvidia-gpus-anatomy-of-high-performance-matmul-kernels/018-9c5036fe.png" alt="Figure 18: Thread organization in naive matmul kernel" />
  <figcaption>图18：朴素矩阵乘法内核中的线程组织</figcaption>
</figure>

矩阵乘法逻辑本身如下：

<figure>
  <img src="/images/others/inside-nvidia-gpus-anatomy-of-high-performance-matmul-kernels/019-63cad355.png" alt="Figure 19: Naive matmul kernel" />
  <figcaption>图19：朴素矩阵乘法内核</figcaption>
</figure>

当我们的GMEM访问被合并时，硬件中会自动发生一些有趣的优化：

- （矩阵A）对于一个从`A`读取的warp，32个每线程`LDG.32`指令（都来自同一地址）被合并为一个warp级别的`LDG.32`，其结果被广播到warp中的所有线程。
- （矩阵B）对于一个从`B`读取的warp，32个连续的每线程`LDG.32`指令被组合成一个128B的warp级别加载。这依赖于线程沿着连续维度读取。如果它们读取列（非连续），硬件将需要发出多个warp级别指令。

注意，我们总共启动了(4096/32) \* (4096/32) = 16,384个线程块。然而，H100 PCIe（我使用的卡）只有114个SM。

这就提出了一个问题：每个SM上可以同时运行多少个块？

一般来说，三种资源限制并发性：

1. 寄存器
1. 共享内存（SMEM）
1. 线程/warp

从Nsight Compute分析器（`ncu --set full -o out.ncu-rep naive_kernel`，也见下图）中，我们看到内核每个线程使用32个寄存器。每个块有1024个线程，那就是1024×32=32,768个寄存器每块。由于每个SM有65,536个寄存器（你可以在[CUDA C编程指南（CUDA C programming guide）](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#features-and-technical-specifications-technical-specifications-per-compute-capability) [\[10\]](#ref-10)中找到这些常量），这限制我们每个SM最多2个块。

📝注意：

提示。你也可以在编译时传递`--ptxas-options=-v`让编译器报告寄存器使用情况和其他资源计数。`nvdisasm`也是一个有用的小工具。

在Hopper（计算能力9.0）上，每个SM的最大线程数是2048。每个块有1024个线程，这再次限制我们每个SM最多2个块。

回想硬件章节，即使内核没有显式使用SMEM，每个块总是有1024B的系统级开销。在默认的SMEM分配为每个SM 8192B（不将拨盘调到228 KiB）的情况下，这将允许最多8个块。

综合起来：`max blocks/SM = min(2,2,8) = 2`。

所以，在任何给定时间，这个内核最多可以有114×2 = 228个线程块驻留在GPU上。

这意味着我们需要16,384 / 228 = \~71.86个所谓的**波（waves）**&#x6765;完成矩阵乘法操作。

📝占用率（Occupancy）

在CUDA术语中，占用率通常指可以在SM上同时运行的块数。还有一个密切相关的定义：

占用率（warp）：活动warp与每个SM最大warp数的比率。

这里，“活动warp”指线程块在启动时分配资源（寄存器、SMEM等）后的warp。

<figure>
  <img src="/images/others/inside-nvidia-gpus-anatomy-of-high-performance-matmul-kernels/020-08298cc8.png" alt="Figure 20: Nsight Compute: Occupancy, Waves info" />
  <figcaption>图20：Nsight Compute：占用率、波信息</figcaption>
</figure>

这里有一个[优秀教程（excellent tutorial）](https://www.youtube.com/watch?v=F_BazucyCMw&ab_channel=GPUMODE) [\[11\]](#ref-11)关于使用Nsight Compute分析器。

值得在此提及：就像**平铺量化（tile quantization）**&#x4E00;样，也有一个**波量化（wave quantization）**&#x7684;概念。当波数较小时，这种效应尤其明显。

例如，假设我启动一个内核，有114个块（正好是我的H100 PCIe上的SM数）。并假设我们一次只能运行1个块/SM。每个SM只有一个块，内核在一个波中完成。现在想象我将启动增加到115个块。突然，执行时间几乎翻倍——因为我们需要两个波——但第二个波中的大部分资源闲置，只有一个块运行：

<figure>
  <img src="/images/others/inside-nvidia-gpus-anatomy-of-high-performance-matmul-kernels/021-90e8eef8.png" alt="Figure 21: Wave quantization" />
  <figcaption>图21：波量化</figcaption>
</figure>

有了对朴素矩阵乘法内核的基本分析，现在让我们转向PTX/SASS视图。以下是我使用的编译设置（[Godbolt](<https://godbolt.org/#g:!((g:!((g:!((h:codeEditor,i:(filename:'1',fontScale:14,fontUsePx:'0',j:1,lang:cuda,selection:(endColumn:42,endLineNumber:16,positionColumn:42,positionLineNumber:16,selectionStartColumn:42,selectionStartLineNumber:16,startColumn:42,startLineNumber:16),source:'//+__global__+keyword+declares+a+GPU+kernel%0A__global__+void+naive_kernel(int+M,+int+N,+int+K,+float+alpha,%0A++++++++++++++++++++++++++++++++++++++++++const+float+*A,+const+float+*B,%0A++++++++++++++++++++++++++++++++++++++++++float+beta,+float+*C)+%7B%0A++int+BLOCKSIZE%3D32%3B%0A%0A++const+int+row+%3D+blockIdx.x+*+BLOCKSIZE+%2B+(threadIdx.x+/+BLOCKSIZE)%3B%0A++const+int+col+%3D+blockIdx.y+*+BLOCKSIZE+%2B+(threadIdx.x+%25+BLOCKSIZE)%3B%0A%0A++if+(row+%3C+M+%26%26+col+%3C+N)+%7B++//+guard+in+case+some+threads+are+outside+the+range%0A++++float+tmp+%3D+0.0%3B%0A++++//+compute+dot+product%0A++++for+(int+i+%3D+0%3B+i+%3C+K%3B+%2B%2Bi)+%7B%0A++++++tmp+%2B%3D+A%5Brow+*+K+%2B+i%5D+*+B%5Bi+*+N+%2B+col%5D%3B%0A++++%7D%0A++++//+GEMM:+C+%3D+alpha+*+A+@+B+%2B+beta+*+C%0A++++C%5Brow+*+N+%2B+col%5D+%3D+alpha+*+tmp+%2B+beta+*+C%5Brow+*+N+%2B+col%5D%3B%0A++%7D%0A%7D'),l:'5',n:'0',o:'CUDA+C%2B%2B+source+%231',t:'0')),header:(),k:31.19733490103861,l:'4',m:100,n:'0',o:'',s:0,t:'0'),(g:!((h:compiler,i:(compiler:nvcc125u1,filters:(b:'0',binary:'1',binaryObject:'1',commentOnly:'0',debugCalls:'1',demangle:'0',directives:'0',execute:'1',intel:'0',libraryCode:'0',trim:'1',verboseDemangling:'0'),flagsViewOpen:'1',fontScale:14,fontUsePx:'0',j:1,lang:cuda,libs:!(),options:'+-O3+-DNDEBUG+--generate-code%3Darch%3Dcompute_90,code%3D%5Bcompute_90,sm_90a%5D+--ptxas-options%3D-v+-std%3Dc%2B%2B17',overrides:!(),selection:(endColumn:1,endLineNumber:1,positionColumn:1,positionLineNumber:1,selectionStartColumn:1,selectionStartLineNumber:1,startColumn:1,startLineNumber:1),source:1),l:'5',n:'0',o:'+NVCC+12.5.1+(Editor+%231)',t:'0')),header:(),k:35.46933176562806,l:'4',n:'0',o:'',s:0,t:'0'),(g:!((h:device,i:(compilerName:'NVCC+12.5.1',device:PTX,editorid:1,fontScale:14,fontUsePx:'0',j:1,selection:(endColumn:1,endLineNumber:1,positionColumn:1,positionLineNumber:1,selectionStartColumn:1,selectionStartLineNumber:1,startColumn:1,startLineNumber:1),treeid:0),l:'5',n:'0',o:'Device+Viewer+NVCC+12.5.1+(Editor+%231,+Compiler+%231)',t:'0')),k:33.33333333333333,l:'4',n:'0',o:'',s:0,t:'0')),l:'2',n:'0',o:'',t:'0')),version:4>)）：

```text
compilation settings:
nvcc 12.5.1

-O3  # the most aggressive standard-compliant optimization level, add loop unrolling, etc.
-DNDEBUG  # turn assert() into noop, doesn't matter for our simple kernel
--generate-code=arch=compute_90,code=[compute_90,sm_90a]  # embed PTX/SASS for H100
--ptxas-options=-v  # makes ptxas print per-kernel resource usage during compilation
-std=c++17  # compile the code according to the ISO C++17 standard, doesn't matter
# --fast-math  # not using, less important for this kernel
```

另一个重要设置是`--use_fast_math`。它用数值精度换取速度，主要影响fp32操作。例如，它将标准数学函数替换为快速、近似的内部函数（例如`sinf`

->

`__sinf`），启用对非正规数（低于最小“正规”可表示幅度的非常小的浮点数）的flush-to-zero（ftz）等。

下面是上面显示的CUDA C++内核的带注释PTX。我手动解码它以更好地内化ISA。请随意放大并花点时间消化结构（或者直接跳到图后阅读我的总结，然后回到图）：

<figure>
  <img src="/images/others/inside-nvidia-gpus-anatomy-of-high-performance-matmul-kernels/022-6ddbe5f8.png" alt="Figure 22: PTX code corresponding to naive matmul CUDA kernel" />
  <figcaption>图22：对应朴素矩阵乘法CUDA内核的PTX代码</figcaption>
</figure>

总结一下，以下是PTX代码的高级流程：

- 计算`row`和`col`变量。有趣的是，编译器使用`bfi`（位字段插入）指令来计算`col`，而不是简单地将寄存器`r2`和`r3`相加。这可能是试图通过将工作路由到较少使用的单元来平衡执行流水线——但请注意`bfi`本身并不比加法指令快。
- 如果此线程在`C`的有效范围之外，则提前退出（防护逻辑）。
- 如果`K < 1`，直接跳转到存储到`C`（`tmp`将为0.0）。
- 如果`K <= 3`，跳转到尾部循环。
- 否则，如果`K > 3`：计算`A`和`B`的基本偏移量，然后进入主循环。
- 主循环（展开×4）。每次迭代执行4个FMA步骤，与加载和地址算术交错进行。
- 尾部循环（`<= 3`次迭代）。执行剩余的点积步骤，不进行展开。
- 尾声：加载`C`的输出值，应用GEMM更新（`alpha * A @ B + beta * C`），并使用`st.global.f32`将结果写回全局内存。

这里可以看到一些编译器优化：提前退出、循环展开、分割为主循环和尾部循环，以及看起来像是流水线负载平衡（假设我的`bfi`假设正确）。

展开尤其重要，因为它暴露了ILP（指令级并行性）。warp不需要那么快被换出，因为它仍有独立的指令可以发出——这有助于隐藏延迟。

什么是ILP（指令级并行性）？

指令级并行性（ILP）是指单个warp通过连续发出独立指令可以同时保持“在飞行中”的工作量。高ILP允许warp调度器在每个周期发出新指令，而较早的指令仍在等待其延迟。

考虑这两个指令流（假设FMA需要4个周期）：

1\) 低ILP（完全依赖链）

```text
y = a * b + 1.0;     // uses a,b
z = y * c + 1.0;     // depends on y
w = z * c + 1.0;     // depends on z
```

每个FMA依赖于前一个结果 => 无法并行调度 => 总延迟 = 12（3\*4）个周期。

2\) 高ILP（独立操作）

```text
c0 = a0 * b0 + 1.0;
c1 = a1 * b1 + 1.0;
c2 = a2 * b2 + 1.0;
```

三个独立的FMA => 调度器可以在连续周期中发出它们。在周期0、1、2发出，结果在4、5、6就绪 => 总延迟 = 6个周期。

这就是为什么循环展开/ILP很重要。

对于调试，您可能希望禁用循环展开以使PTX/SASS分析更容易。只需添加：`#pragma unroll 1`。

展开还减少了分支（`bra`）指令的数量，使程序更简洁/高效。

我还观察到一些编译器低效之处，例如：

- 不必要的变量初始化为0。
- 过于复杂的`A`地址计算。
- 冗余的部分偏移量计算，其中两个指令本可以合并为一个。

有趣！现在让我们看看对应的SASS代码：

<figure>
  <img src="/images/others/inside-nvidia-gpus-anatomy-of-high-performance-matmul-kernels/023-8a52ed55.png" alt="Figure 23: SASS code corresponding to naive matmul CUDA kernel" />
  <figcaption>图23：对应朴素矩阵乘法CUDA内核的SASS代码</figcaption>
</figure>

我只强调与PTX代码的差异：

- 循环现在展开×16！
- LDG指令移动到循环顶部，将计算与数据加载重叠。FMA大多聚集在每个展开块的末尾。
- 有2个尾部循环：一个展开8倍，一个展开4倍，最终循环覆盖最后3次迭代。

我在SASS中也发现了有趣的编译器怪癖和低效之处：

- 程序计数器（`R1`寄存器）被加载但从未使用。不清楚为什么？
- 冗余的零初始化仍然存在。
- 一个谓词是无操作：它总是为真，所以跳转到标签`L_x_2`（4倍展开循环）永远不会执行。
- 4倍展开循环包含一个多余的`BRA`指令——它永远不会迭代超过一次。
- 在最终的`EXIT`之后，代码陷入无限while循环。是虚假的实现细节还是故障？
- 最后（不是故障），代码用`NOPs`填充以实现内存对齐。

有趣！我们了解了编译器在幕后做了什么。

现在，有了所有这些背景知识，让我们换个话题，深入一些SOTA（最先进）内核。

📝下一章的补充阅读：

我强烈推荐Simon的优秀[博客文章](https://siboehm.com/articles/22/CUDA-MMM)。它是我最初深入内核的灵感来源。在这一章中，我将使用他的[内核10](https://github.com/siboehm/SGEMM_CUDA/blob/master/src/kernels/10_kernel_warptiling.cuh) [\[12\]](#ref-12)代码作为参考。虽然代码本身似乎是基于CUTLASS的（参见[这个](https://developer.nvidia.com/blog/cutlass-linear-algebra-cuda/) [\[13\]](#ref-13)和[这个](https://github.com/NVIDIA/cutlass/blob/b0e09d7cd371eded41f7c1e057caf1593c27ba55/media/docs/efficient_gemm.md) [\[14\]](#ref-14)例如），但我首先分析了Simon的版本——所以这里我将遵循那个版本。

## 设计接近SOTA的同步矩阵乘法内核

在这一章中，我们将分解一个在以下约束下接近SOTA的fp32内核：

- 无TMA
- 无异步内存指令
- 无张量核心
- 仅fp32（无bf16）

换句话说，这是在Volta前GPU模型下的SOTA（在Volta/Ampere上接近SOTA）：

- Volta引入了张量核心
- Ampere引入了异步内存指令
- Hopper引入了TMA

我们将研究的技术称为**warp-tiling**。

在深入之前，让我们用一个小修改重新审视之前的内核，看看会发生什么。具体来说，我们将改变`row`和`col`变量的计算方式。

原始版本：

```c
const int row = blockIdx.x * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
const int col = blockIdx.y * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);
```

修改版本：

```c
const int row = blockIdx.x * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);
const int col = blockIdx.y * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
```

换句话说，我们只是交换了`%`和`/`运算符。

交换`row2`和`col2`是与之前示例相比逻辑结构中的唯一变化：

<figure>
  <img src="/images/others/inside-nvidia-gpus-anatomy-of-high-performance-matmul-kernels/024-a1578a89.png" alt="Figure 24: New logical organization of row2 and col2 variables" />
  <figcaption>图24：row2和col2变量的新逻辑组织</figcaption>
</figure>

这是修改后内核现在所做的：

<figure>
  <img src="/images/others/inside-nvidia-gpus-anatomy-of-high-performance-matmul-kernels/025-a98a3759.png" alt="Figure 25: Naive kernel with uncoalesced GMEM access" />
  <figcaption>图25：具有非合并GMEM访问的朴素内核</figcaption>
</figure>

这个看似无害的调整使我们的GMEM访问非合并。

在我的H100 PCIe卡上，性能从3171 GFLOP/s下降到仅243 GFLOP/s——13倍的减速。这正是我们之前在GMEM部分看到的惩罚（Stephen Jones的跨步GMEM访问实验）。

从外部看，这似乎只是两个运算符之间的简单交换。但如果您没有硬件的心理模型，您永远不会预料到如此戏剧性的效果。

<figure>
  <img src="/images/others/inside-nvidia-gpus-anatomy-of-high-performance-matmul-kernels/026-5d1c148e.png" alt="Figure 26: Roofline model" />
  <figcaption>图26：屋顶线模型</figcaption>
</figure>

查看屋顶线模型，您可以看到我们的内核位于图的深内存带宽限制区域。我们为计算支付了NVIDIA大笔费用，所以我们应该瞄准计算限制区域。

📝屋顶线模型

屋顶线模型将**性能（FLOP/s）**&#x7ED8;制在y轴上，将**算术强度（AI）**&#x7ED8;制在x轴上。

算术强度定义为从设备内存/GMEM加载的每字节执行的FLOP数（默认情况下）。

“脊点”出现在：`peak perf / GMEM bw`。对于我的H100 PCIe，这大约是\~410。只有当AI超过这个值时，内核才能进入计算限制区域。

让我们在继续之前重新审视顺序矩阵乘法代码。作为参考：

```c
for (int m = 0; m < M; m++) {
    for (int n = 0; n < N; n++) {
        float tmp = 0.0f;  // accumulator for dot product
        for (int k = 0; k < K; k++) {
            tmp += A[m][k] * B[k][n];
        }
        C[m][n] = tmp;
    }
}
```

这里我想强调的关键点是语义对循环顺序是不变的。换句话说，我们可以以3! = 6种方式中的任意一种排列三个嵌套循环，结果仍然是正确的矩阵乘法。

在这六种排列中，最有趣的是`K`作为最外层循环。（m和n的相对顺序不太重要，所以让我们假设“规范”`m-n`顺序）：

```c
for (int k = 0; k < K; k++) {
    for (int m = 0; m < M; m++) {
        float a = A[m][k];  // reuse this load across N (think GMEM access minimization)
        for (int n = 0; n < N; n++) {
            C[m][n] += a * B[k][n];
        }
    }
}
```

如果这些加载来自GMEM，我们通过将加载次数从`A`从`N^3`减少到`N^2`，大约节省了2倍带宽。

但更重要的见解是算法性的：这个版本将矩阵乘法计算为**外积的部分和（partial sum of outer products）**。这个视角对于理解warp-tiling方法至关重要，我们接下来将深入探讨：

<figure>
  <img src="/images/others/inside-nvidia-gpus-anatomy-of-high-performance-matmul-kernels/027-d86b0b62.png" alt="Figure 27: Matmul as a sum of partial outer products" />
  <figcaption>图27：矩阵乘法作为部分外积的和</figcaption>
</figure>

这可能显而易见，但值得强调：点积等价于部分点积的和：

<figure>
  <img src="/images/others/inside-nvidia-gpus-anatomy-of-high-performance-matmul-kernels/028-ba9329f3.png" alt="Figure 28: Dot product is equivalent to a sum of partial dot products" />
  <figcaption>图28：点积等价于部分点积的和</figcaption>
</figure>

这很重要，因为它让我们将计算分解为一系列块矩阵乘法（每个产生部分点积）。通过在执行计算前将这些块移动到SMEM中，我们可以减少GMEM流量并显著加速。

如果不分块，我们不可能将其放入SMEM中。

还要回想，我们初始的内核算术强度非常低——它们每加载的字节做的工作很少。为了提高它，我们需要：

1. 每个线程计算多个输出元素。
1. 使输出瓦片尽可能接近正方形。

这里有一个直观的解释为什么这很重要：

<figure>
  <img src="/images/others/inside-nvidia-gpus-anatomy-of-high-performance-matmul-kernels/029-438eb3c2.png" alt="Figure 29: Arithmetic intensity improves when each thread computes multiple outputs and when tiles approach a square shape" />
  <figcaption>图29：当每个线程计算多个输出且瓦片接近正方形时，算术强度提高</figcaption>
</figure>

至此，我们已经收集了理解warp-tiling所需的大部分部分。让我们把它们组合起来。

我们知道两个关键点：

- 输出瓦片应该是正方形的（以最大化算术强度）。
- 计算应该分解为子步骤，以便中间块可以放入SMEM。

考虑到这一点，算法的高层结构如下：

<figure>
  <img src="/images/others/inside-nvidia-gpus-anatomy-of-high-performance-matmul-kernels/030-5306a1ba.png" alt="Figure 30: High-level structure of the warp-tiling algorithm, also referred to as block tiling." />
  <figcaption>图30：warp-tiling算法的高层结构，也称为块平铺（block tiling）。</figcaption>
</figure>

参考代码[在这里](https://github.com/siboehm/SGEMM_CUDA/blob/master/src/kernels/10_kernel_warptiling.cuh)。我建议从我的图表开始，然后打开代码来连接所有点。

📝注意：

我将使用与Simon博客文章中相同的瓦片大小（未针对我的H100进行自动调优）：

`Bm = Bn = 128, Bk = 16`

由于每个块的计算是独立的——并且我们已经确信部分点积累积为完整点积——我们只需要关注单个块的单个步骤。其余部分（其他1023个块，4096/128 \* 4096/128 = 32 \* 32 = 1024总计）将遵循相同的逻辑。

📝给自己提醒

出于某种原因，我很难忽略其他块。所以，是时候念咒语了：“其他一切都是正确的；我只需要专注于下一步。局部正确性导致全局正确性。” :)

带着这种心态，让我们放大到蓝色块的第一个步骤（红色箭头转换前的计算），它对应于输出瓦片`C[0,0]`（注意 - 瓦片 - 不是元素）。

块维度是`Bm × Bk`对于矩阵`A`和`Bk × Bn`对于矩阵`B`。这些被加载到SMEM缓冲区`As`和`Bs`中。

加载/存储`B`到`Bs`是直接的，因为`Bs`未转置。4个warp中的每个从GMEM获取一行`B`，每个线程发出向量化加载（`LDG.128`），然后是向量化存储（`STS.128`）。每个warp循环4次，步长为4行。

对应代码（我添加了注释并移除了Simon注释掉的代码）：

```text
for (uint offset = 0; offset + rowStrideB <= BK; offset += rowStrideB) {
    // we need reinterpret_cast to force LDG.128 instructions (128b = 4 4B floats)
    reinterpret_cast<float4 *>(
        &Bs[(innerRowB + offset) * BN + innerColB * 4])[0] =
        reinterpret_cast<const float4 *>(
            &B[(innerRowB + offset) * N + innerColB * 4])[0];
  }
```

<figure>
  <img src="/images/others/inside-nvidia-gpus-anatomy-of-high-performance-matmul-kernels/031-636ace81.png" alt="Figure 31: Loading a chunk of B (GMEM) into Bs (SMEM)" />
  <figcaption>图31：将B的块（GMEM）加载到Bs（SMEM）</figcaption>
</figure>

加载`A`→`As`。这个步骤更棘手，因为`As`是转置的。转置的原因是在计算阶段启用向量化加载（`LDS.128`）。

权衡是存储不能向量化：从`A`的一行获取的4个浮点数现在必须分散到`As`的一列中，这映射到相同的内存bank。这是可以接受的，因为我们优先考虑快速加载——`As`的每个元素在计算期间将被多次访问，而存储只发生一次。

图中的`innerRowX`和`innerColX`注释确切显示了每个线程负责的工作。

对应代码：

```text
for (uint offset = 0; offset + rowStrideA <= BM; offset += rowStrideA) {
  // we need reinterpret_cast to force LDG.128 instructions
  const float4 tmp = reinterpret_cast<const float4 *>(
      &A[(innerRowA + offset) * K + innerColA * 4])[0];
  As[(innerColA * 4 + 0) * BM + innerRowA + offset] = tmp.x;
  As[(innerColA * 4 + 1) * BM + innerRowA + offset] = tmp.y;
  As[(innerColA * 4 + 2) * BM + innerRowA + offset] = tmp.z;
  As[(innerColA * 4 + 3) * BM + innerRowA + offset] = tmp.w;
}
```

<figure>
  <img src="/images/others/inside-nvidia-gpus-anatomy-of-high-performance-matmul-kernels/032-c438d970.png" alt="Figure 32: Loading a chunk of A (GMEM) into As (SMEM)" />
  <figcaption>图32：将A的块（GMEM）加载到As（SMEM）</figcaption>
</figure>

加载后，我们同步线程块（`__syncthreads()`）以确保所有数据在`As`和`Bs`中可用。

现在是计算阶段。

对应代码（我建议略读它，并查看带有几次传递的绘图：））：

```sql
for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {  // dotIdx is the outer most loop
  // WM = 64, that's why As is broken into 2x64 parts
  // TM = 8, that's why thread processes 8 rows from As
  // WMITER = 1, that's why only single slice in As (2 in the appendix of the drawing)
  for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
    // load from As into register regM
    for (uint i = 0; i < TM; ++i) {
      regM[wSubRowIdx * TM + i] =
          As[(dotIdx * BM) + warpRow * WM + wSubRowIdx * WSUBM +
              threadRowInWarp * TM + i];
    }
  }
  // WN = 64, that's why Bs is broken into 2x64 parts
  // TN = 4, that's why 4 columns per slice of Bs
  // WNITER = 4, that's why four slices in Bs
  // WSUBN = WN/WNITER = 16 (used to iterate over slices)
  for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
    for (uint i = 0; i < TN; ++i) {
      // load from Bs into register regN
      regN[wSubColIdx * TN + i] =
          Bs[(dotIdx * BN) + warpCol * WN + wSubColIdx * WSUBN +
              threadColInWarp * TN + i];
    }
  }

  // execute warptile matmul via a sum of partial outer products
  for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
    for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
      for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
        for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
          threadResults[(wSubRowIdx * TM + resIdxM) * (WNITER * TN) +
                        (wSubColIdx * TN) + resIdxN] +=
              regM[wSubRowIdx * TM + resIdxM] *
              regN[wSubColIdx * TN + resIdxN];
        }
      }
    }
  }
}
```

<figure>
  <img src="/images/others/inside-nvidia-gpus-anatomy-of-high-performance-matmul-kernels/033-130a37f6.png" alt="Figure 33: Performing matmul between As and Bs as a series of thread-level outer products (warp-tiling + thread-tiling)." />
  <figcaption>图33：在As和Bs之间执行矩阵乘法作为一系列线程级外积（warp-tiling + thread-tiling）。</figcaption>
</figure>

一旦块被处理，我们再次同步。这防止了竞态条件——没有它，一些线程可能开始将下一个块写入`As`和`Bs`，而其他线程仍在处理当前块。

同步后，我们将`A`和`B`的指针前进`Bk`，算法重复直到所有块被处理。

```text
A += BK;     // move BK columns to right
B += BK * N; // move BK rows down
```

最后，一旦循环完成，128个线程将它们的私有`threadResults`寄存器刷新到矩阵`C`的对应输出瓦片中（现在包含完整的点积！）。

在实践中，您会为特定GPU自动调优此算法的参数。但如前所述，这种风格的内核不再是首选方法——现代GPU具有异步内存机制和张量核心，将性能推远超出仅warp-tiling所能提供的。

至此，让我们转向Hopper上的真正SOTA。

📝下一章的补充阅读：

我高度推荐Pranjal的优秀[博客文章](https://cudaforfun.substack.com/p/outperforming-cublas-on-h100-a-worklog) [\[15\]](#ref-15)，它读起来更像工作日志。在本章中，我将遵循他工作日志中的内核。与Simon的工作一样，大部分代码似乎是受CUTLASS启发的（参见，例如这些帖子：CUTLASS[ping pong kernel](https://pytorch.org/blog/cutlass-ping-pong-gemm-kernel/) [\[16\]](#ref-16)和[efficient GEMM](https://github.com/NVIDIA/cutlass/blob/b0e09d7cd371eded41f7c1e057caf1593c27ba55/media/docs/efficient_gemm.md)）。

值得注意的是，细节决定成败，Pranjal成功超越了cuBLAS SOTA——在几个目标矩阵维度上达到约107%的cuBLAS性能。

## 在Hopper上设计SOTA异步矩阵乘法内核

现在是时候利用所有硬件特性，在Hopper上达到真正的SOTA。我们将使用：

- TMA同步加载/存储操作
- 张量核心（Tensor Cores）
- bf16精度（bf16 precision）

这些硬件特性不仅显著简化了warp-tiling（warp平铺）方法，还将性能提升了近一个数量级——Pranjal报告了从32 TFLOP/s到317 TFLOP/s的10倍增长。

📝参考代码：

我将以[kernel 2](https://github.com/pranjalssh/fast.cu/blob/main/examples/matmul/matmul_2.cuh) [\[17\]](#ref-17)作为这里的参考（另见我的[PR](https://github.com/pranjalssh/fast.cu/pull/8/files)）。注意，符号已从Simon的版本略有变化：`As` → `sA`和`Bs` → `sB`。

这种简化之所以有效，是因为TMA（Tensor Memory Accelerator，张量内存加速器）和Tensor Cores（张量核心）抽象了我们之前处理的大部分手动复杂性。

作为实现Hopper SOTA（State-of-the-Art，最先进技术）的第一步，让我们修改warp-tiling（warp平铺）基线。

我们保持完全相同的程序结构，除了：

- 我们现在每个线程块只需要128个线程（4个warps）。
- 平铺大小设置为`BM = BN = BK = 64`。

<figure>
  <img src="/images/others/inside-nvidia-gpus-anatomy-of-high-performance-matmul-kernels/034-5640c629.png" alt="Figure 34: We keep the same high-level structure of the warp-tiling algorithm (block-tiling)." />
  <figcaption>图34：我们保持warp-tiling（warp平铺）算法（块平铺）的相同高层结构。</figcaption>
</figure>

💡矩阵格式更改：

重要：A仍为行主序，但B现在为列主序格式。

## 通过TMA异步加载到SMEM

对于第二阶段——将数据加载到SMEM（Shared Memory，共享内存）——TMA用一个更简单的东西替换了复杂的warp级加载模式。我们只需要：

- 为`A`和`B`构建张量映射。
- 触发TMA操作（由块中的单个线程完成）。
- 使用共享内存屏障进行同步。

TMA不仅移动数据，还自动应用swizzling（混洗），这解决了我们之前在warp-tiling（warp平铺）中看到的bank冲突。（我将在后面的专门部分详细讨论swizzling。）

要形成张量映射，我们使用`cuTensorMapEncodeTiled`（见[docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html#group__CUDA__TENSOR__MEMORY_1ga7c7d2aaac9e49294304e755e6f341d7)）。此函数编码了将`A`和`B`的块从GMEM（Global Memory，全局内存）传输到SMEM所需的所有元数据。我们需要每个`A`和`B`一个张量映射，但结构上它们相同。对于`A`，我们指定：

- 数据类型：bf16
- 秩：2（矩阵）
- 指针：`A`
- 形状：`(K,M)`（最快步长维度优先）
- 行步长：`K * sizeof(bf16)`
- `sA`的形状：`(BK, BM)`
- Swizzle（混洗）模式：加载到`sA`

时使用128B模式

```python
__shared__ barrier barA;  // SMEM barriers for A and B
__shared__ barrier barB;

if (threadIdx.x == 0) {
    // initialize with all 128 threads
    init(&barA, blockDim.x);
    init(&barB, blockDim.x);
    // make initialized barrier visible to async proxy
    cde::fence_proxy_async_shared_cta();
}
__syncthreads();  // ensure barriers are visible to all threads
```

下一步：`sA`这里我们初始化SMEM屏障，用于同步写入`sB`和

。屏障用所有128个线程初始化，因为我们期望块中的每个线程在屏障切换到“就绪”状态之前到达。`cde::fence_proxy_async_shared_cta()`调用

是Hopper代理内存模型的一部分。它在CTA（块）范围内排序“异步代理”（TMA）和“通用代理”（正常线程ld/st）之间的可见性。这里我们在初始化后立即发出它，以便异步引擎看到屏障的初始化状态。（异步复制的完成将由mbarrier本身发出信号。）

完全披露：我也不声称完全理解所有内存一致性细节——官方文档也没有太大帮助。这可能值得单独写一篇后续文章。如果有人有关于这个主题的好资源——请联系我！`K`在外部

```cpp
for (int block_k_iter = 0; block_k_iter < num_blocks_k; ++block_k_iter) {
    if (threadIdx.x == 0) {  // only one thread launches TMA
        // Offsets into GMEM for this CTA's tile:
        //   A: (block_k_iter * BK, num_block_m * BM)
        cde::cp_async_bulk_tensor_2d_global_to_shared(
            &sA[0], tensorMapA, block_k_iter*BK, num_block_m*BM, barA);
        // update barrier with the number of bytes it has to wait before flipping:
        // sizeof(sA)
        tokenA = cuda::device::barrier_arrive_tx(barA, 1, sizeof(sA));
        //   B: (block_k_iter * BK, num_block_n * BN)
        cde::cp_async_bulk_tensor_2d_global_to_shared(
            &sB[0], tensorMapB, block_k_iter*BK, num_block_n*BN, barB);
        tokenB = cuda::device::barrier_arrive_tx(barB, 1, sizeof(sB));
    } else {
        tokenA = barA.arrive();  // threads-only arrival (no byte tracking)
        tokenB = barB.arrive();
    }
    barA.wait(std::move(tokenA));  // blocks until: all threads arrived AND TMA finished
    barB.wait(std::move(tokenB));
```

循环中：`A`发生了什么，逐步（对于`B`和

1. ）：`cp_async_bulk` `_tensor_2d_global_to_shared(...)`线程0启动TMA，指定SMEM目标（`sA`/`sB`）、张量映射，以及GMEM偏移指定源GMEM块。
1. 它立即调用`barrier_arrive_tx(bar, 1, sizeof(sX))`，这：
1. - 计数线程到达数（这里为1，来自线程0），并且
   - 用**预期字节数**武装屏障，以便它知道异步复制何时完成。
     所有其他线程调用`bar.arrive()`，贡献它们的到达数（无字节）。
1. 然后每个线程调用`bar.wait(token)`。此等待仅在两个条件都为真时完成：
   - 所有128个线程都已到达，并且
   - 异步引擎已将全部`sizeof(sX)`字节写入共享内存。

此加载模式是标准的Hopper惯用法——你会在现代内核中到处看到它。

在异步复制期间，TMA还使用**128B swizzle（混洗）格式**对数据进行swizzling（混洗）。

让我们花点时间解释swizzling（混洗）的实际含义。我在网上找不到清晰的解释，所以这是我的尝试——部分为你，部分为未来的我。:)

## Swizzling（混洗）

让我们从一个激励性示例开始：

<figure>
  <img src="/images/others/inside-nvidia-gpus-anatomy-of-high-performance-matmul-kernels/035-942f2ad1.png" alt="Figure 35: Swizzling example" />
  <figcaption>图35：Swizzling（混洗）示例</figcaption>
</figure>

这里发生了什么？

假设我们想加载原始GMEM矩阵第一行的所有元素。Swizzling（混洗）后，这仍然很简单：只需从SMEM矩阵读取第一行。没什么特别的。

现在，假设我们想要原始GMEM矩阵的第一列。注意，这些元素现在位于SMEM的对角线上。这意味着我们可以在单个周期内加载它们，因为没有两个线程命中同一个bank——零bank冲突。

没有swizzling（混洗），此访问会将所有这些列元素映射到同一个bank但不同地址，产生8路bank冲突并将吞吐量削减8倍。

相同属性适用于任何行或列：swizzling（混洗）后，它们都可以在单个周期内服务！

<figure>
  <img src="/images/others/inside-nvidia-gpus-anatomy-of-high-performance-matmul-kernels/036-e56ada57.png" alt="Figure 36: No bank conflicts when loading rows or columns" />
  <figcaption>图36：加载行或列时无bank冲突</figcaption>
</figure>

相同属性适用于存储。例如，如果你想在SMEM中转置矩阵，朴素的方法是：加载一行然后将其写回为一列。没有swizzling（混洗），这将导致8路bank冲突。

启用swizzling（混洗）后，我们避免了这个问题，但你必须小心索引。

📝注意

TMA在将数据从SMEM移回GMEM时会自动取消swizzling（混洗）。

既然动机清楚了，让我们问以下问题：TMA实际上如何生成swizzle（混洗）模式？

事实证明答案很简单：与特定掩码模式进行XOR（异或）。

快速回顾XOR，这是真值表：

1. 0, 0 映射到 0
1. 0, 1 映射到 1
1. 1, 0 映射到 1
1. 1, 1 映射到 0

值得注意的是：当其中一个位为1时，XOR翻转另一个位。

通常，我们可以在CUTLASS中找到[答案](https://github.com/NVIDIA/cutlass/blob/76c96b0be35cb263debe3e3d8418b80911a544ab/include/cute/swizzle.hpp#L42)。另一个Simon（不是之前的那个）也很好地解释了掩码模式如何[生成](https://veitner.bearblog.dev/understanding-cute-swizzling-the-math-behind-32b-64b-and-128b-patterns/) [\[18\]](#ref-18)——尽管没有确切说明该模式如何导致我们刚刚看到的特定swizzle（混洗）布局。

所以两个大问题仍然存在：

1. XOR掩码如何生成？
1. 掩码如何实际应用以产生swizzle（混洗）模式？

## 生成XOR掩码

NVIDIA将每个swizzle（混洗）模式与特定的“swizzle（混洗）函数”关联：

- 128B swizzle（混洗）模式与`Swizzle<3,4,3>`
- 关联`Swizzle<2,4,3>`
- 64B swizzle（混洗）模式与`Swizzle<1,4,3>`

关联`Swizzle<3,4,3>`32B swizzle（混洗）模式与

```sql
// To improve readability, I'll group bits in 8s with underscores.

// Swizzle<3, 4, 3>
// -> BBits = 3
// -> MBase = 4
// -> SShift = 3

// Given the decoded arguments from above here are the steps that the swizzling function does:

// Step 1. Compute bit_msk = (1 << BBits) - 1
bit_msk = (0b00000000_00000001 << 3) - 1 = 0b00000000_00000111  // keep 16 bit resolution

// Step 2. Compute yyy_msk = bit_msk << (MBase + max(0, SShift))
yyy_msk = 0b00000000_00000111 << 7 = 0b00000011_10000000

// Step 3. Mask the input number (annotated bits A-P for clarity)
input_number = 0bABCDEFGH_IJKLMNOP

masked = input_number & yyy_mask
  = 0bABCDEFGH_IJKLMNOP & 0b00000011_10000000 = 0b000000GH_I0000000

// Step 4. Shift right by SShift (masked >> SShift)
shifted = masked >> 3
  = 0b000000GH_I0000000 >> 3 = 0b00000000_0GHI0000

// Step 5. XOR with the original input
output = input_number ^ shifted
  = 0bABCDEFGH_IJKLMNOP ^ 0b00000000_0GHI0000 = 0bABCDEFGH_IwyzMNOP

// Replace unchanged bits with x for legibility.
// I'll also uppercase "wyz" to make it stand out and keep GHI around as they have an impact on wyz:
output = 0bxxxxxxGH_IWYZxxxx

// where WYZ = GHI ^ JKL (XOR)
```

关联
让我们解析`GHI`（位置 9、8、7，零索引）。如果其中任何一位为 1，则翻转相应的位`JKL`（位置 6、5、4）以得到`WYZ`。所有其他位保持不变。

让我们建立一些关于 swizzling 函数行为的直觉：

<figure>
  <img src="/images/others/inside-nvidia-gpus-anatomy-of-high-performance-matmul-kernels/037-faf268e7.png" alt="Figure 37: Swizzle function intuition" />
  <figcaption>图 37：Swizzle 函数直觉</figcaption>
</figure>

对于 32B 和 64B swizzling 模式，swizzling 函数是`0bxxxxxxxx_IxxZxxxx`和`0bxxxxxxxH_IxYZxxxx`。

这些遵循相同的 XOR-with-mask 思想，只是使用不同的控制位来驱动哪些低位被翻转。

这一切如何与我们开始的动机示例联系起来？

这是链接：

<figure>
  <img src="/images/others/inside-nvidia-gpus-anatomy-of-high-performance-matmul-kernels/038-16982d93.png" alt="Figure 38: Connecting the swizzle function to the matrix swizzle example" />
  <figcaption>图 38：将 swizzle 函数连接到矩阵 swizzle 示例</figcaption>
</figure>

这就是 swizzling 的 WHY 和 HOW。 :)

## Tensor Cores

回到 tensor cores。此时，我们已经将`A`和`B`的块从 GMEM 拉入`sA`和`sB`在 SMEM 中。它们已经过 swizzled，并准备好供 tensor core 使用。

NVIDIA 公开了几种矩阵乘加（MMA）指令：

- `wmma` — warp-cooperative，同步（旧世代）。
- `mma.sync` — warp-cooperative，同步（Ampere）。
- `wgmma.mma_async` — warp-group cooperative，异步（Hopper）。

📝注意：

一个**warp group** = 4 warps = 128 个线程在 CUDA 中。

我们将专注于`wgmma.mma_async`（[docs](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-instructions) [\[19\]](#ref-19)），因为它随 Hopper 引入，是目前最强大的。它是异步的，并利用 4 个协作的 warps 一起计算 matmul；这正是我们选择块大小 = 128 的原因。

对于 bf16 操作数，`wgmma`支持形状为`m64nNk16`的形式，其中`N ∈ {8, 16, 24, …, 256}`。在我们当前的示例中，我们将使用`m64n64k16`，但一般来说，更大的`N`值性能更好（只要你有足够的寄存器和 SMEM 来支持它们）。

📝注意：

`m64n64k16`意味着 tensor core 一次性计算一个`64×16` ×`16×64`的 matmul。

以下是操作数放置规则：`sA`可以驻留在寄存器或 SMEM 中，`sB`必须驻留在 SMEM 中，而累加器（`BM x BN`）始终在寄存器中。

由于这对于单个线程来说寄存器太多，累加器在 warp group 中的线程之间分区。

在我们的参考内核中，你会看到它像这样初始化：

```css
float d[WGMMA_N/16][8];  // d is the accumulator; GEMM: D = A @ B + D
memset(d, 0, sizeof(d));  // init to all 0s
```

我们设置`WGMMA_M = WGMMA_N = BM = BN = 64`。这给出：

- warp group 中的 128 个线程
- 每个线程持有`WGMMA_N/16 × 8`个寄存器
- 总计：128 × (64/16) × 8 = 64 × 64 个寄存器

...这完全匹配累加器大小（`BM × BN = 64 × 64`），只是在组中分布。

这是我们将分解的相应 tensor core 片段：

```css
asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
wgmma64<1, 1, 1, 0, 0>(d, &sA[0], &sB[0]);
wgmma64<1, 1, 1, 0, 0>(d, &sA[WGMMA_K], &sB[WGMMA_K]);
wgmma64<1, 1, 1, 0, 0>(d, &sA[2*WGMMA_K], &sB[2*WGMMA_K]);
wgmma64<1, 1, 1, 0, 0>(d, &sA[3*WGMMA_K], &sB[3*WGMMA_K]);
asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
asm volatile("wgmma.wait_group.sync.aligned %0;" ::"n"(0) : "memory");
```

📝注意：

- 一些 Hopper 指令未在 CUDA C++ 中公开，因此我们使用`asm(...);`进入内联 PTX。
- `::: "memory"`是一个内存破坏器，它防止任何内存优化围绕 asm 语句，它是给编译器的“不要将周围的内存访问移过此点”的提示；禁止编译器在此语句周围重新排列内存操作。
- `volatile`告诉编译器 asm 块\*不得\*被删除或提升，即使它看起来冗余（见[docs](https://docs.nvidia.com/cuda/inline-ptx-assembly/#incorrect-optimization)）[\[20\]](#ref-20)。

让我们首先解包围绕实际 matmul 调用的书挡指令（`wgmma.fence`、`commit_group`、`wait_group`）。

`wgmma.fence.sync.aligned;` - [docs](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-instructions-wgmma-fence)解释得很好：“wgmma.fence 在先前对任何 warpgroup 寄存器的访问和后续由 wgmma.mma_async 指令对相同寄存器的访问之间建立顺序。”

在实践中，warp-group 的所有四个 warps 必须在第一个`wgmma.mma_async`之前执行此 fence。

之后，我们就可以开始了。即使累加器寄存器在那些四个 wgmma 调用中被更新，我们之间不需要更多的 fences —— 对于相同形状、累加到相同寄存器的背靠背 MMAs 有一个特殊例外。这正是我们这里的情况。

这真的只是样板代码。事实上，如果你注释掉它，编译器会悄悄地为你插回它。

`wgmma.commit_group` - 另一个样板操作：来自[docs](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-instructions-wgmma-commit-group)的“将所有先前未提交的 wgmma.mma_async 操作提交到一个 wgmma-group”。它将我们刚刚启动的所有`wgmma.mma_async`（上面的四个调用）关闭到一个单一的“组”中。

`wgmma.wait_group 0` - 意思是：在这一点之前的所有组完成之前不要继续。由于我们只启动了一个组，它只是说“等待直到那四个 MMAs 完成并且结果实际位于累加器寄存器中”。

所以标准节奏是：fence → 启动一批异步 MMAs → 提交它们 → 等待它们完成。

现在到 wgmma 本身。`wgmma64`函数是内联 PTX 调用的包装器：

```text
wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16
```

操作码的结构使其含义相当透明：f32 是累加器数据类型，bf16 是输入`sA`和`sB`矩阵的数据类型。

语义是通常的融合乘加：`D = A @ B+D`即，GEMM 累加到现有的 fp32 tile 中。（有一个标志可以将其变为`D=A @ B`，我们稍后将使用它。）

我故意跳过`sA`和`sB`的 SMEM 描述符如何形成并传递到指令的细节。这些描述符编码 SMEM 基地址、swizzle 模式（在我们的情况下为 128B），以及`LBO`/`SBO`（前导/步长维度字节偏移）值，以便 tensor core 可以正确导航布局。在这里覆盖描述符构造将是在已经很长的帖子中的绕道；它可能值得自己写一篇专注的文章。只需知道有这个额外的元数据层，其解释我暂时省略了。

这是为什么我们需要 4 个 wgmma 调用的解释：

<figure>
  <img src="/images/others/inside-nvidia-gpus-anatomy-of-high-performance-matmul-kernels/039-c08b9966.png" alt="Figure 39: Why doing four 64x16 @ 16x64 wgmma calls is equivalent to doing a 64x64 @ 64x64 matmul" />
  <figcaption>图 39：为什么做四个 64x16 @ 16x64 wgmma 调用等价于做一个 64x64 @ 64x64 matmul</figcaption>
</figure>

这里稍微令人费解的部分是列主表示：如何`sB[0] … sB[48]`最终映射到正确的逻辑位置/切片。

但关键要点是，我们之前处理的许多 warp-tiling 和 thread-tiling 复杂性现在在硬件中被抽象掉了。过去需要跨 warps 精心编排的东西已经简化为一些样板指令和几个声明性的 wgmma 调用。

也就是说，这只是起点。我们仍然浪费 TMA 和 tensor core 周期：

<figure>
  <img src="/images/others/inside-nvidia-gpus-anatomy-of-high-performance-matmul-kernels/040-12899786.png" alt="Figure 40: We are wasting TMA and TC cycles - we can do better" />
  <figcaption>图 40：我们正在浪费 TMA 和 TC 周期 - 我们可以做得更好</figcaption>
</figure>

我们解决浪费周期的方法是通过流水线化计算和数据移动。具体来说，我们将`sA`和`sB`（SMEM 驻留的 tiles）变成一个块队列 —— 比如长度为 5。

然后我们将工作拆分到两个 warp-groups：

- 一个 warp-group 充当`producer`，负责通过将新的`A`和`B`块流式传输到队列中来保持 TMA 忙碌。
- 另一个warp组（warp-group）充当`consumer`，从队列中提取数据以保持张量核心（tensor cores）饱和。

自然，这需要协调。我们使用的机制是一个SMEM屏障（SMEM barriers）队列，每个队列槽位有一个`full[i]`/`empty[i]`对来同步生产者和消费者。

参考：[kernel 4](https://github.com/pranjalssh/fast.cu/blob/main/examples/matmul/matmul_4.cuh#L270)代码。

以下是设置：

```rust
// queue of barriers
__shared__ barrier full[QSIZE], empty[QSIZE];
// use the largest MMA shape available
constexpr int WGMMA_M = 64, WGMMA_K = 16, WGMMA_N=BN;
```

初始化与之前类似：

```rust
if (threadIdx.x == 0) {
  for (int i = 0; i < QSIZE; ++i) {
      // num_consumers == 1 in this example;
      // 128 threads from consumer wg + 1 producer thread
      init(&full[i], num_consumers * 128 + 1);
      init(&empty[i], num_consumers * 128 + 1);
  }
  cde::fence_proxy_async_shared_cta();  // same as before
}
__syncthreads();  // same as before
```

需要注意两点：

- 我们已升级到更大的张量核心MMA（从`m64n64k16`到`m64nBNk16`），因为经验表明这有助于最大化计算吞吐量。
- 由于队列是多槽位的，屏障初始化必须循环遍历所有条目。

以下是主要逻辑：

- 在生产者（`wg_idx = 0`）中，一个线程协调TMA拷贝到队列。它使用`empty[qidx].wait()`来阻塞直到缓冲区槽位空闲，然后为`cp_async_bulk_tensor` `_2d_global_to_shared`和`sA`发出`sB`。最后，它用`barrier_arrive_tx`信号完成，将屏障与拷贝的字节数绑定。
- 在消费者（`wg_idx > 0`）中，所有线程首先将每个队列槽位标记为“空”（准备填充）。然后，对于每个`K`步骤，它们等待`full[qidx]`，在该缓冲区上运行张量核心MMA，完成后再次将槽位标记为空。

```css
// Producer
if (wg_idx == 0) {  // wg_idx = threadIdx.x / 128
    if (tid == 0) {  // only thread 0 issues TMA calls
        int qidx = 0;  // index into the circular buffer
        for (int block_k_iter = 0; block_k_iter < num_blocks_k; ++block_k_iter, ++qidx) {
            if (qidx == QSIZE) qidx = 0;  // wrap around
            // wait until this buffer is marked empty (ready to be written into)
            empty[qidx].wait(empty[qidx].arrive());
            // copy over chunks from A and B
            cde::cp_async_bulk_tensor_2d_global_to_shared(
                &sA[qidx*BK*BM], tensorMapA, block_k_iter*BK, num_block_m*BM, full[qidx]);
            cde::cp_async_bulk_tensor_2d_global_to_shared(
                &sB[qidx*BK*BN], tensorMapB, block_k_iter*BK, num_block_n*BN, full[qidx]);
            // mark barrier with the expected byte count (non-blocking)
            barrier::arrival_token _ = cuda::device::barrier_arrive_tx(
              full[qidx], 1, (BK*BN+BK*BM)*sizeof(bf16));
        }
    }
} else {
    // Consumer warp-group
    for (int i = 0; i < QSIZE; ++i) {
        // i initially, all buffers are considered empty; ready for write
        // all 128 consumer threads arrive on each barrier
        barrier::arrival_token _ = empty[i].arrive();
    }
    // distributed accumulator registers, zero-initialized
    float d[BM/WGMMA_M][WGMMA_N/16][8];
    memset(d, 0, sizeof(d));

    int qidx = 0;
    for (int block_k_iter = 0; block_k_iter < num_blocks_k; ++block_k_iter, ++qidx) {
        if (qidx == QSIZE) qidx = 0;  // wrap around
        // wait until TMA has finished filling this buffer
        full[qidx].wait(full[qidx].arrive());

        // core tensor core loop
        warpgroup_arrive();  // convenience wrapper around the PTX boilerplate
        #pragma unroll  // compiler hint (we saw this in PTX/SASS section)
        // submit as many tensor core ops as needed to compute sA @ sB (see drawing)
        for (int m_it = 0; m_it < BM/WGMMA_M; ++m_it) {
            bf16 *wgmma_sA = sA + qidx*BK*BM + BK*m_it*WGMMA_M;
            #pragma unroll
            for (int k_it = 0; k_it < BK/WGMMA_K; ++k_it) {
                wgmma<WGMMA_N, 1, 1, 1, 0, 0>(
                  d[m_it], &wgmma_sA[k_it*WGMMA_K], &sB[qidx*BK*BN + k_it*WGMMA_K]);
            }
        }
        warpgroup_commit_batch();
        warpgroup_wait<0>();

        // all 128 consumer threads mark buffer as consumed so producer can reuse it
        barrier::arrival_token _ = empty[qidx].arrive();
    }

    // finally: write accumulator d back to output matrix C

}
```

可视化应该使其更清晰：

<figure>
  <img src="/images/others/inside-nvidia-gpus-anatomy-of-high-performance-matmul-kernels/041-12a260d3.png" alt="Figure 41: More efficient TC/TMA pipeline: producer warp-group streams tiles into a circular buffer; consumer warp-group drains tiles into tensor cores." />
  <figcaption>图41：更高效的TC/TMA流水线：生产者warp组（warp-group）将图块流式传输到循环缓冲区；消费者warp组（warp-group）将图块排入张量核心（tensor cores）。</figcaption>
</figure>

一个自然的调整是将输出图块从128×128增加到128×256。问题在于，在这个大小下，单个消费者warp组（warp-group）中每个线程的累加器分片变得太大——每个线程仅累加器就需要256个fp32寄存器，这超出了每线程寄存器预算（并触发寄存器溢出到设备内存，这对性能非常不利）。

解决方法是添加另一个消费者warp组（warp-group），使累加器在两个组之间分片而不是一个。我们保持单个生产者（以驱动TMA）并以3×128 = 384线程启动块/CTA：

- WG0：生产者（TMA）
- WG1：消费者A（计算128×256图块的上半部分）
- WG2：消费者B（计算下半部分）

每个消费者拥有输出的64×256半图块，因此每线程累加器占用减半，避免溢出。

以下是矩阵乘法现在如何执行：

<figure>
  <img src="/images/others/inside-nvidia-gpus-anatomy-of-high-performance-matmul-kernels/042-b5e9ea8f.png" alt="Figure 42: Two consumer warp groups let us grow the tile from 128x128 -> 128x256 without register spills" />
  <figcaption>图42：两个消费者warp组（warp-group）让我们将图块从128x128 -> 128x256增长而不发生寄存器溢出</figcaption>
</figure>

下一个重要想法是，我们也可以隐藏写入输出图块的延迟：

<figure>
  <img src="/images/others/inside-nvidia-gpus-anatomy-of-high-performance-matmul-kernels/043-688f6329.png" alt="Figure 43: Persistent kernels: overlap the output store with incoming loads by launching one long-lived block per SM that processes many tiles." />
  <figcaption>图43：持久内核（Persistent kernels）：通过为每个SM启动一个长生存期的块来处理多个图块，将输出存储与传入加载重叠。</figcaption>
</figure>

💡持久内核（Persistent kernels）

持久内核（Persistent kernels）启动少量固定数量的线程块（通常每个SM一个），并在整个工作负载期间保持它们存活。每个块运行一个内部循环，从队列中拉取新图块直到工作完成，而不是为每个图块启动一个块。

这引发了一个自然问题：每个SM应处理输出图块的哪个子集，以及以什么顺序？

这个调度策略看起来如何？

让我们从一个玩具设置开始来推理选项：

- 输出图块总数：64。
- SM数量：10。
- 因此每个SM平均需要处理约6.4个块。

第一次尝试可能看起来像这样：

<figure>
  <img src="/images/others/inside-nvidia-gpus-anatomy-of-high-performance-matmul-kernels/044-94473507.png" alt="Figure 44: Naïve schedule" />
  <figcaption>图44：朴素调度</figcaption>
</figure>

我们能做得更好吗？是的——通过使调度缓存感知：

<figure>
  <img src="/images/others/inside-nvidia-gpus-anatomy-of-high-performance-matmul-kernels/045-9f30c858.png" alt="Figure 45: Block-wise cache-aware schedule" />
  <figcaption>图45：块级缓存感知调度</figcaption>
</figure>

但我们能做得更好吗？令人惊讶的是，是的——通过使用空间填充曲线：

<figure>
  <img src="/images/others/inside-nvidia-gpus-anatomy-of-high-performance-matmul-kernels/046-22c81af3.png" alt="Figure 46: Hilbert-curve schedule" />
  <figcaption>图46：希尔伯特曲线（Hilbert-curve）调度</figcaption>
</figure>

我将深入探讨的最后一个想法是利用Hopper新的集群级CUDA执行模型来减少L2/GMEM流量：

<figure>
  <img src="/images/others/inside-nvidia-gpus-anatomy-of-high-performance-matmul-kernels/047-21a30a9b.png" alt="Figure 47: Using thread block clusters to reduce the number of L2/GMEM loads." />
  <figcaption>图47：使用线程块集群（thread block clusters）减少L2/GMEM加载次数。</figcaption>
</figure>

关键观察是，集群内的多个SM可以通过DSMEM直接共享它们的SMEM，这让我们可以将集群视为一种“超级SM”。

从调度角度来看，没有根本性变化：不是每个SM处理自己独立的输出图块，而是整个集群协作处理一个更大的“超级图块”。算法的机制保持不变，但现在这些SM协调加载并重用彼此的数据。

由于希尔伯特曲线（Hilbert-curve）遍历已经设计为最大化局部性，超级SM可以遵循相同的遍历模式——只是以更粗的粒度。

最后，要超越cuBLAS，我们必须收紧同步本身。到目前为止，我们在屏障上的到达/等待调用一直很浪费。

例如，消费者线程实际上不需要在`full[qidx]`上发出到达信号。唯一重要的条件是“所有字节已到达”。丢弃那些冗余到达每次迭代节省256个令牌。类似地，对于`empty[qidx]`：一旦带有`tid==0`的消费者到达，生产者可以安全地开始填充，因为消费者端（wgmma）在所有线程中锁步执行。

一些额外的、较低级别的技巧在实践中累积起来（本着O(NR)的精神）：

- 重新平衡寄存器：使用`asm volatile("setmaxnreg.{inc,dec}.sync.aligned.u32 %0;\n" : : "n"(RegCount));`将寄存器预算从生产者warp组（warp-group）（轻量级）转移到消费者warp组（warp-group）（wgmma期间的重度用户）。
- 避免在输出时污染缓存。要么使用`__stwt`绕过L1/L2，或者更好的是，进行异步存储：先溢出到SMEM，然后让TMA异步拷贝到GMEM。这将写回与计算重叠，就像我们在输入端所做的那样。
- 跳过冗余初始化：不是清零累加器寄存器，而是调整张量核心（tensor cores）序列，使第一个MMA执行`C = A @ B`，后续MMA执行`C = A @ B + C`。

作为参考，以下是性能数字（来自Pranjal的博客），显示每个想法如何叠加在前一个之上：

| 优化                                                         | 优化前性能（TFLOP/s） | 优化后性能（TFLOP/s） |
| ------------------------------------------------------------ | --------------------- | --------------------- |
| 基线（warp-tiling） → 张量核心（Tensor Cores）+ TMA          | 32                    | 317                   |
| 增加输出图块大小                                             | 317                   | 423                   |
| 流水线：重叠TMA加载与TC计算                                  | 423                   | 498                   |
| 图块增长：128×128 → 128×256（2个消费者warp组（warp-group）） | 498                   | 610                   |
| 持久内核（Persistent kernels）（隐藏存储延迟）               | 610                   | 660                   |
| 更快的PTX屏障                                                | 660                   | 704                   |
| 集群；TMA多播                                                | 704                   | 734                   |
| 微优化                                                       | 734                   | 747                   |
| TMA异步存储（寄存器 → SMEM → GMEM）                          | 747                   | 758                   |
| 希尔伯特曲线（Hilbert-curve）调度                            | 758                   | 764                   |

此外，Aroun提交了一个[PR](https://github.com/pranjalssh/fast.cu/pull/1)，使用`stmatrix`方法优化了异步存储，又带来了+1%的提升。一些核反应堆被节省了。

## 结语

我们首先剖析了GPU本身，重点关注内存层次结构——为GMEM（全局内存）、SMEM（共享内存）和L1（一级缓存）建立心智模型，然后将它们连接到CUDA编程模型。在此过程中，我们还探讨了“光速”，它如何受功率限制——硬件现实渗入我们的模型。

从那里，我们向上移动堆栈：学习如何通过PTX/SASS与硬件通信，以及如何引导编译器生成我们真正想要的内容。

我们沿途掌握了关键概念——瓦片（tile）和波量化（wave quantization）、占用率（occupancy）、指令级并行（ILP）、屋顶线模型（roofline model）——并围绕基本等价关系建立了直觉：点积作为部分外积的和，或作为点积的部分和，以及为什么方形瓦片能产生更高的算术强度（arithmetic intensity）。

基于这个基础，我们构建了一个接近最先进水平的内核（warp tiling），仅使用CUDA核心、寄存器和共享内存就榨取了性能。

最后，我们步入了Hopper的世界：TMA（张量内存加速器）、swizzling（混洗）、张量核心（tensor cores）和`wgmma`指令、异步加载/存储管道、调度策略如希尔伯特曲线（Hilbert curves）、带TMA多播的集群、更快的PTX屏障等。

我将以贯穿整个系列的信念结束：**计算机可以被理解**。

💡联系我：

如果您在帖子中发现任何错误，请私信我——欢迎在[X](https://x.com/gordic_aleksa)或[LinkedIn](https://www.linkedin.com/in/aleksagordic/)上给我留言，或通过[匿名反馈](https://docs.google.com/forms/d/1z1fEirrN2xtGxAsJvptpM7yV4ByT5SF25S-XiMPrXNA/edit)。

## 致谢

非常感谢[Hyperstack](https://www.hyperstack.cloud/)在过去一年为我提供H100进行实验！

感谢我的朋友[Aroun Demeure](https://github.com/ademeure)（Magic的GPU与AI专家，前苹果和Imagination的GPU架构师），以及[Mark Saroufim](https://x.com/marksaroufim)（PyTorch）阅读此博客文章的预发布版本并提供反馈！

## 参考文献

1. NVIDIA Hopper架构深度解析 <https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/>
1. NVIDIA Ampere架构深度解析 <https://developer.nvidia.com/blog/nvidia-ampere-architecture-in-depth/>
1. 奇怪的是，GPU上的矩阵乘法在给定“可预测”数据时运行更快！\[简短] <https://www.thonking.ai/p/strangely-matrix-multiplications>
1. CUDA编程如何工作 <https://www.nvidia.com/en-us/on-demand/session/gtcfall22-a41101/>
1. 关于NVIDIA GPU共享内存库的笔记 <https://feldmann.nyc/blog/smem-microbenchmarks>
1. CUDA二进制工具 <https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html>
1. 第37讲：SASS与GPU微架构简介 <https://www.youtube.com/watch?v=we3i5VuoPWk>
1. 通过微基准测试剖析NVIDIA Volta GPU架构 <https://arxiv.org/abs/1804.06826>
1. 如何优化CUDA矩阵乘法内核以达到cuBLAS类似性能：工作日志 <https://siboehm.com/articles/22/CUDA-MMM>
1. CUDA C编程指南 <https://docs.nvidia.com/cuda/cuda-c-programming-guide/>
1. 第44讲：NVIDIA性能分析 <https://www.youtube.com/watch?v=F_BazucyCMw&ab_channel=GPUMODE>
1. <https://github.com/siboehm/SGEMM_CUDA/>
1. CUTLASS：CUDA C++中的快速线性代数 <https://developer.nvidia.com/blog/cutlass-linear-algebra-cuda/>
1. CUDA中的高效GEMM <https://github.com/NVIDIA/cutlass/blob/b0e09d7cd371eded41f7c1e057caf1593c27ba55/media/docs/efficient_gemm.md>
1. 在H100上超越cuBLAS：工作日志 <https://cudaforfun.substack.com/p/outperforming-cublas-on-h100-a-worklog>
1. 深入探讨CUTLASS Ping-Pong GEMM内核 <https://pytorch.org/blog/cutlass-ping-pong-gemm-kernel/>
1. <https://github.com/pranjalssh/fast.cu/>
1. 理解CuTe Swizzling——32B、64B和128B模式背后的数学 <https://veitner.bearblog.dev/understanding-cute-swizzling-the-math-behind-32b-64b-and-128b-patterns/>
1. 并行线程执行 <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html>
1. CUDA中的内联PTX汇编 <https://docs.nvidia.com/cuda/inline-ptx-assembly/>
1. 揭秘高带宽内存对实时系统的特性 <https://upcommons.upc.edu/server/api/core/bitstreams/b843de39-f32f-4069-8843-48f74c030213/content>
1. <https://github.com/triton-lang/triton>
