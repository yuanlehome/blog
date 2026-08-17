---
title: "如何理解 GPU"
description: "我们在 Google 热爱 TPU，但 GPU 同样出色。本章将深入探索 GPU 的世界：每块芯片如何工作、它们如何通过网络连接，以及这些特性对 LLM 意味着什么，尤其是与 TPU 相比时。尽管 NVIDIA、AMD、Intel 等厂商提供了众多 GPU 架构，本章将重点关注 NVIDIA GPU。本章建立在第 2 章（https://jax-ml.github.io/scaling-book/tpus/）和第 5 章（https://jax-ml.github.io/scaling-book/training）的基础上，建议先阅读这两章。"
chapter: 12
order: 12
part: 4
partTitle: "总结与附加内容"
sourcePath: "gpus.md"
sourceCommit: "44109cacac9c5a9809a81c68ae4d45d7d2632ea6"
---

<span id="how-to-think-about-gpus"></span>

# 如何理解 GPU

<span id="what-is-a-gpu"></span>

## GPU 是什么？

现代机器学习 GPU（例如 H100、B200）基本上由一组专门执行矩阵乘法的计算核心（称为**流式多处理器（Streaming Multiprocessor，SM）**）与一条高速内存（称为**高带宽内存（HBM）**）相连而成。下图展示了它的结构：

![图：H100 或 B200 GPU 的抽象布局。H100 有 132 个 SM，而 B200 有 148 个。这里对“Warp Scheduler”一词的使用略为宽泛，用它来指一组由 32 个 CUDA SIMD 核心组成的单元以及向这些核心分派工作的调度器。注意，它看起来和 TPU 多么相似！](/images/scaling-book/gpu/gpu-diagram.png)

与 TPU 的 TensorCore 类似，每个 SM 都有一个专用的矩阵乘法核心（不巧，它也叫 **Tensor Core**[^ch12-1]）、一个向量算术单元（称为 **Warp Scheduler**[^ch12-2]），以及一块高速片上缓存（称为 **SMEM**）。TPU 最多只有 2 个独立的“TensorCore”，而现代 GPU 有 100 多个 SM（H100 上有 132 个）。每个 SM 的能力都远弱于一个 TPU TensorCore，但整个系统更加灵活。各个 SM 几乎完全独立，因此 GPU 可以同时执行数百项不同的任务。[^ch12-3]

下面更细致地看一下 H100 的一个 SM：

![图：H100 SM 的结构图（来源），其中展示了 4 个子分区；每个子分区都包含一个 Tensor Core、一个 Warp Scheduler、一个寄存器文件，以及多组支持不同精度的 CUDA 核心。靠近底部的“L1 Data Cache”就是 256kB 的 SMEM 单元。B200 的结构与之相似，但额外加入了相当可观的张量内存（Tensor Memory，TMEM），用于向体积庞大的 Tensor Core 馈送数据。](/images/scaling-book/gpu/blackwell-sm.png)

来源：[NVIDIA Hopper GH100 概览](https://wccftech.com/nvidia-hopper-gh100-gpu-official-5nm-process-worlds-fastest-hpc-chip-80-billion-transistors-hbm3-memory/)。

每个 SM 会被划分为 4 个完全相同的象限，NVIDIA 称之为 **SM 子分区（SM subpartition）**。每个子分区都包含一个 Tensor Core、16k 个 32 位寄存器，以及一个称为 Warp Scheduler 的 SIMD/SIMT 向量算术单元；NVIDIA 将其各条通道（ALU）称为 **CUDA 核心（CUDA Core）**。每个分区的核心组件可以说是 Tensor Core，它负责执行矩阵乘法，并贡献绝大部分 FLOPs/s；但它并不是唯一值得关注的组件。

* <strong>CUDA 核心：</strong>每个子分区都包含一组称为 CUDA 核心的算术逻辑单元（ALU），用于执行 SIMD/SIMT 向量运算。每个 ALU 通常能在每个周期内执行 1 次算术操作，例如 f32.add。[^ch12-4] 每个子分区包含 32 个 fp32 核心（另有数量较少的 int32 和 fp64 核心），它们在每个周期内都执行同一条指令。与 TPU 的 VPU 类似，CUDA 核心负责 ReLU、逐点向量运算和归约（求和）。[^ch12-5]

* <strong>Tensor Core（TC）：</strong>每个子分区都有自己的 Tensor Core，它是类似于 TPU MXU 的专用矩阵乘法单元。Tensor Core 贡献了 GPU 绝大部分 FLOPs/s（例如在 H100 上，bf16 TC 的算力为 990 TFLOP/s，而 CUDA 核心只有 66 TFLOPs/s）。
  * [990 bf16 TFLOPs/s](https://www.nvidia.com/en-us/data-center/h100/) 配合 132 个以 1.76GHz 运行的 SM，意味着每个 H100 TC 每周期可执行 `7.5e12 / 1.76e9 / 4 ~ 1024` 次 bf16 FLOPs，大致相当于一次 8x8x8 矩阵乘法。[^ch12-6]
  * 与 TPU 一样，GPU 能以更高吞吐量执行较低精度的矩阵乘法（例如 H100 的 fp8 FLOPs/s 是 fp16 的 2 倍）。低精度训练或推理服务可以快得多。
  * 从 Volta 开始，每一代 GPU 都比上一代增大了 TC 的规模（可参阅[这篇不错的文章](https://semianalysis.com/2025/06/23/nvidia-tensor-core-evolution-from-volta-to-blackwell/)）。到了 B200，TC 已经大到其输入无法再装进 SMEM，因此 B200 引入了一种名为 TMEM 的新内存空间。[^ch12-7]

<strong>CUDA 核心比 TPU 的 VPU 更灵活：</strong>GPU 的 CUDA 核心（从 V100 开始）采用所谓 SIMT（*Single Instruction Multiple Threads*，单指令多线程）编程模型，而 TPU 采用 SIMD（*Single Instruction Multiple Data*，单指令多数据）模型。与 TPU VPU 中的 ALU 一样，同一子分区内的 CUDA 核心必须在每个周期执行相同操作（例如，如果一个核心正在把两个浮点数相加，那么该子分区中的其他所有 CUDA 核心也必须执行加法）。不过，与 VPU 不同的是，每个 CUDA 核心（在 CUDA 编程模型中也称“线程”）都有自己的指令指针，可以被独立地<em>编程</em>。当同一个 warp 中的两个线程被要求执行不同操作时，实际上会把<em>两个</em>操作都执行一遍，同时屏蔽掉不需要执行当前分支操作的核心。

![图：一组线程中发生 warp 分支发散的示例（来源）。白色空隙表示至少有一部分物理 CUDA 核心处于停顿状态。](/images/scaling-book/gpu/warp-divergence.png)

来源：[NVIDIA Volta 架构白皮书](https://images.nvidia.com/content/volta-architecture/pdf/volta-architecture-whitepaper.pdf)。

这使得线程级编程非常灵活，但代价是：如果 warp 过于频繁地发生分支发散，性能会悄无声息地下降。线程能够访问的内存也更加灵活；VPU 只能操作连续的内存块，而 CUDA 核心可以访问共享寄存器中的单个浮点数，并维护逐线程状态。

<strong>CUDA 核心的调度也更加灵活：</strong>SM 的运行方式有点像多线程 CPU，因为它们可以并发“调度”许多程序（即 <strong>warp</strong>，每个 SM 最多 64 个），但每个 _Warp Scheduler_ 在每个时钟周期只会执行一个程序。[^ch12-8] Warp Scheduler 会在活跃的 warp 之间自动切换，以隐藏内存加载等 I/O 操作。相比之下，TPU 通常是单线程的。

<span id="memory"></span>

### 内存

除了计算单元之外，GPU 还有一套内存层次结构：最大的是 HBM（GPU 的主内存），然后依次是更小的缓存（L2、L1/SMEM、TMEM 和寄存器内存）。

* <strong>寄存器：</strong>在 H100/B200 上，每个子分区都有自己的寄存器文件，其中包含 16,384 个 32 位字（每个 SM 为 `4 * 16384 * 4 = 256kiB`），CUDA 核心可以访问这些寄存器。
  * 每个 CUDA 核心一次最多只能访问 256 个寄存器，因此，虽然每个 SM 最多可以调度 64 个“驻留 warp”，但如果每个线程都使用 256 个寄存器，就只能同时容纳 8 个（`256 * 1024 / (4 * 32 * 256)`）。

* <strong>SMEM（L1 缓存）：</strong>每个 SM 都有自己的 256kB 片上缓存，称为 SMEM；它既可以作为“共享内存”由程序员控制，也可以作为片上缓存由硬件使用。SMEM 用来存储激活值和 TC 矩阵乘法的输入。

* <strong>L2 缓存：</strong>所有 SM 共享[^ch12-9]一块相对较大的约 50MB L2 缓存，用来减少对主内存的访问。
  * 它的容量与 TPU 的 VMEM 相近，但速度**慢得多**，而且不能由程序员控制。这就产生了一点“幽灵般的超距作用”：程序员必须调整内存访问模式，才能确保 L2 缓存得到充分利用。[^ch12-10]
  * NVIDIA 并未公布其芯片的 L2 带宽，但[实测](https://chipsandcheese.com/p/nvidias-h100-funny-l2-and-tons-of-bandwidth)约为 5.5TB/s。这大约是 HBM 带宽的 1.6 倍，不过它是全双工的，因此有效双向带宽更接近 3 倍。相比之下，TPU 的 VMEM 容量是其 2 倍，带宽也高得多（约 40TB/s）。

* <strong>HBM：</strong>GPU 的主内存，用于存储模型权重、梯度、激活值等。
  * 从 Volta 的 32GB 到 Blackwell（B200）的 192GB，HBM 容量增长了很多。
  * 从 HBM 到 CUDA Tensor Core 的带宽称为 HBM 带宽或内存带宽；H100 上约为 3.35TB/s，B200 上约为 9TB/s。

<span id="summary-of-gpu-specs"></span>

### GPU 规格汇总

下面汇总了近期 GPU 型号的规格。同一种 GPU 的不同变体在 SM 数量、时钟速度和 FLOPs 上会略有差异。首先是内存容量数据：

|  GPU  | 架构代际 |   时钟速度   | 每芯片 SM 数 | 每个 SM 的 SMEM 容量 | 每芯片 L2 容量 | 每芯片 HBM 容量 |
| :---: | :--------: | :-------------: | :------: | :--------------: | :--------------: | :---------------: |
| V100  |   Volta    | 1.25GHz/1.38GHz |    80    |       96kB       |       6MB        |       32GB        |
| A100  |   Ampere   | 1.10GHz/1.41GHz |   108    |      192kB       |       40MB       |       80GB        |
| H100  |   Hopper   | 1.59GHz/1.98GHz |   132    |      256kB       |       50MB       |       80GB        |
| H200  |   Hopper   | 1.59GHz/1.98GHz |   132    |      256kB       |       50MB       |       141GB       |
| B200  | Blackwell  |        ?        |   148    |      256kB       |      126MB       |       192GB       |

所有代际的每个 SM 都有 256kB 寄存器内存。Blackwell 还为每个 SM 增加了 256kB TMEM。下面是各芯片的 FLOPs 和带宽数据：

|  GPU  | 架构代际 | 每芯片 HBM BW | 每芯片 FLOPs/s（bf16/fp16） | 每芯片 FLOPs/s（fp8/int8） | 每芯片 FLOPs/s（fp4） |
| :---: | :--------: | :---------: | :----------------------: | :---------------------: | :----------------: |
| V100  |   Volta    |   9.0e11    |            —             |            —            |         —          |
| A100  |   Ampere   |   2.0e12    |          3.1e14          |         6.2e14          |         —          |
| H100  |   Hopper   |   3.4e12    |          9.9e14          |         2.0e15          |         —          |
| H200  |   Hopper   |   4.8e12    |          9.9e14          |         2.0e15          |         —          |
| B200  | Blackwell  |   8.0e12    |          2.3e15          |         4.5e15          |       9.0e15       |

这里没有列出 B100，因为它没有大规模量产。[^ch12-11] 有些规格会因 GPU 的具体版本而略有不同，因为 NVIDIA GPU 不像 TPU 那样标准化。

下面这张速查表对比了 GPU 和 TPU 的各个组件：

|              GPU              |     TPU     |              它是什么？              |
| :---------------------------: | :---------: | :-----------------------------------: |
| 流式多处理器（SM） | TensorCore | 包含其他单元的核心“单元格” |
|        Warp Scheduler         |     VPU     |      SIMD 向量算术单元      |
|           CUDA 核心           |   VPU ALU   |               SIMD ALU                |
|        SMEM（L1 缓存）        |    VMEM     |       高速片上缓存       |
|          Tensor Core          |     MXU     |      矩阵乘法单元       |
|        HBM（又称 GMEM）         |     HBM     |  高带宽、大容量内存  |

<span id="gpus-vs-tpus-at-the-chip-level"></span>

### 芯片层面的 GPU 与 TPU 对比

GPU 最初用于渲染电子游戏，但自 2010 年代深度学习兴起以来，它们越来越像专用矩阵乘法机器——换句话说，越来越像 TPU。[^ch12-12] 在一定程度上，这段历史解释了现代 GPU 为何呈现出如今的样子。它们并非专为 LLM 或机器学习模型而设计，而是通用加速器；其硬件追求某种程度的“通用性”，这既可能是福，也可能是祸。GPU 用于新任务时更常能“直接工作”，对优秀编译器的依赖也远小于 TPU。但这也使 GPU 更难推理、更难达到 Roofline 性能，因为太多编译器特性都可能形成瓶颈。

<strong>GPU 更加模块化。</strong>TPU 有 1～2 个大型 TensorCore，而 GPU 有数百个小型 SM。类似地，每个 TC 有一个大型 VPU，由 4 个可独立编程的 8x128 单元组成（总计 4096 个 ALU）；相比之下，H100 有 132 * 4 = 528 个独立 SIMD 单元，每个宽度为 32（总计 16k 个 ALU）。下面是一张 GPU 与 TPU 的逐项对比表，突出了这一点：

|              GPU              |           TPU            | H100 数量 | TPU v5p 数量 |
| :---------------------------: | :----------------------: | :----: | :-------: |
| SM（流式多处理器） |       TensorCore        |  132   |     2     |
|        Warp Scheduler         |        VPU 槽位         |  528   |     8     |
|        SMEM（L1 缓存）        |           VMEM           |  32MB  |   128MB   |
|           寄存器           | 向量寄存器（VRegs） |  32MB  |   256kB   |
|          Tensor Core          |           MXU            |  528   |     8     |

这种模块化差异一方面让 TPU 的制造成本低得多，也更容易理解；另一方面，它也让编译器承担了更重的责任，必须做出正确选择。TPU 只有单一控制线程，并且只支持 VPU 范围的向量化指令，因此编译器需要手动流水化所有内存加载以及 MXU/VPU 工作，以避免停顿。GPU 程序员则可以直接启动数十个不同的内核，每个内核都在完全独立的 SM 上运行。反过来，这些内核的性能也可能极差，因为它们不断冲刷 L2 缓存，或者未能合并内存访问；由于运行时的许多部分都由硬件控制，理解幕后究竟发生了什么也变得困难。因此，TPU 往往能用更少的工作量，更接近峰值 Roofline 性能。

<strong>从历史上看，单个 GPU 比可比的 TPU 更强大（也更昂贵）：</strong>单个 H200 的 FLOPs/s 接近 TPU v5p 的 2 倍，HBM 容量是后者的 1.5 倍。与此同时，Google Cloud 上 H200 的标价约为每小时 \$10，而 TPU v5p 约为每小时 \$4。TPU 通常比 GPU 更依赖于把多个芯片联网协同工作。

<strong>TPU 拥有多得多的高速缓存。</strong>与 GPU 的 SMEM（加 TMEM）相比，TPU 还拥有多得多的 VMEM。这些内存可以用来存储权重和激活值，使其能够以极高速度被加载和使用。如果能持续把模型权重存入或预取到 VMEM 中，这会让 TPU 在 LLM 推理时更快。

<span id="quiz-1-gpu-hardware"></span>

### 小测 1：GPU 硬件

下面这些题目用于检验前面的内容。答案已经给出，但最好先拿起纸笔，试着自己回答问题，然后再查看答案。

<strong>问题 1［CUDA 核心］：</strong>H100 有多少个 fp32 CUDA 核心（ALU）？B200 呢？与 TPU v5p 中独立 ALU 的数量相比如何？

<details>
<summary>点击此处查看答案。</summary>


<strong>答案：</strong>H100 有 132 个 SM，每个 SM 有 4 个子分区，每个子分区包含 32 个 fp32 CUDA 核心，因此共有 `132 * 4 * 32 = 16896` 个 CUDA 核心。B200 有 `148` 个 SM，因此总计 `18944` 个。TPU v5p 有 2 个 TensorCore（通常通过 Megacore 连接），每个 TensorCore 都有一个 VPU；VPU 包含 (8, 128) 条通道，每条通道有 4 个独立 ALU，因此共有 `2 * 4 * 8 * 128 = 8192` 个 ALU。这个数量大约是 H100 向量通道数的一半，运行频率则大致相同。

</details>

**问题 2［向量 FLOPs 计算］**：单个 H100 有 132 个 SM，时钟速度为 1.59GHz（加速频率最高 1.98GHz）。假设每个 ALU 每周期可以执行一次向量操作。每秒可以执行多少次 fp32 向量 FLOPs？加速频率下呢？与矩阵乘法 FLOPs 相比如何？

<details>
<summary>点击此处查看答案。</summary>


**答案：**`132 * 4 * 32 * 1.59e9 = 26.9TFLOPs/s`。在加速频率下为 33.5 TFLOPs/s。这个数只有[规格表](https://www.nvidia.com/en-us/data-center/h100/)所报数值的一半，因为严格来说，一个周期内可以执行一次 FMA（融合乘加），而这被计作两次 FLOPs；但在多数情况下，这一点并无用处。我们可以执行 990 bfloat16 矩阵乘法 TFLOPs/s，因此忽略 FMA 时，Tensor Core 的 FLOPs/s 大约高出 30 倍。

</details>

<strong>问题 3［GPU 矩阵乘法强度］：</strong>H100 上 fp16 矩阵乘法的峰值强度是多少？B200 呢？fp8 又是多少？*这里的强度是指矩阵乘法 FLOPs/s 与内存带宽之比。*

<details>
<summary>点击此处查看答案。</summary>


<strong>答案：</strong>H100 的 fp16 峰值 FLOPs 为 990e12，带宽为每秒 3.35e12 字节。因此，临界强度为 `990e12 / 3.35e12 = 295`，与 TPU 的 240 相当接近。B200 为 `2250e12 / 8e12 = 281`，也非常接近。这意味着，与 TPU 类似，要使矩阵乘法达到计算受限，批大小需要约为 280。

H100 和 B200 的 fp8 FLOPs 都恰好是 2 倍，因此峰值强度也分别翻倍到 590 和 562；不过，如果考虑到权重很可能也以 fp8 加载，那么从某种意义上说，这个强度仍保持不变。

</details>

<strong>问题 4［矩阵乘法运行时间］：</strong>根据问题 3 的答案，你预计单个 B200 上的 `fp16[64, 4096] * fp16[4096, 8192]` 矩阵乘法要运行多久？`fp16[512, 4096] * fp16[4096, 8192]` 呢？

<details>
<summary>点击此处查看答案。</summary>


由上面的结果可知，当 token 批大小低于 281 时，我们会受到内存带宽限制。因此，第一个矩阵乘法完全受带宽限制。我们需要读取或写入 $2BD + 2DF + 2BF$ 字节（`2*64*4096 + 2*4096*8192 + 2*64*8192=69e6`）；带宽为每秒 `8e12` 字节，所以大约需要 `69e6 / 8e12 = 8.6us`。实际中，我们可能只能获得总带宽的一部分，因此耗时可能更接近 10～12us。增大批大小后，我们会完全达到计算受限，因此预计 `T=2*512*4096*8192/2.3e15=15us`。同样，我们只能期望获得总 FLOPs 的一部分，所以实际可能更接近 20us。

</details>

<strong>问题 5［L1 缓存容量］：</strong>H100 的 L1/SMEM 总容量是多少？寄存器内存呢？与 TPU 的 VMEM 容量相比如何？

<details>
<summary>点击此处查看答案。</summary>


<strong>答案：</strong>每个 SM 有 256kB SMEM 和 256kB 寄存器内存，因此两者各约 33MB（`132 * 256kB`）。二者合计约 66MB。这大约是现代 TPU 120MB VMEM 的一半，不过一整个 TPU 总共只有 256kB 寄存器内存！TPU VMEM 的延迟低于 SMEM，这也是寄存器内存在 TPU 上不那么关键的原因之一（向 VMEM 溢出和从 VMEM 回填的代价很低）。

</details>

<strong>问题 6［计算 B200 时钟频率］：</strong>NVIDIA 在[这里](https://resources.nvidia.com/en-us-blackwell-architecture)报告称，B200 可以执行 80TFLOPs/s 的 fp32 向量计算。已知每个 CUDA 核心在一次 FMA（融合乘加）操作中每周期可执行 2 次 FLOPs，请估算其峰值时钟频率。

<details>
<summary>点击此处查看答案。</summary>


<strong>答案：</strong>我们知道共有 148 * 4 * 32 = 18944 个 CUDA 核心，因此每周期可以执行 `18944 * 2 = 37888 FLOPs / cycle`。所以 `80e12 / 37888 = 2.1GHz`，这是一个很高但合理的峰值时钟速度。B200 通常使用液冷，因此较高的时钟频率也更为合理。

</details>

<strong>问题 7［估算 H100 加法运行时间］：</strong>根据上面的数据，计算在单个 H100 上把两个 `fp32[N]` 向量相加应当需要多长时间。分别计算 $T_\text{math}$ 和 $T_\text{comms}$。该操作的算术强度是多少？如果你能使用相关硬件，也请尝试在 PyTorch 或 JAX 中分别以 `N = 1024` 和 `N=1024 * 1024 * 1024` 运行此操作。结果相比如何？

<details>
<summary>点击此处查看答案。</summary>


<strong>答案：</strong>首先，将两个 `fp32[N]` 向量相加会执行 N 次 FLOPs，需要加载 `4 * N * 2` 字节，并写回 4 * N 字节，总计 `3 * 4 * N = 12N`。计算两者之比，可得 `total FLOPs / total bytes = N / 12N = 1 / 12`，这个数实在很糟糕。

如前所算，忽略 FMA 时，在加速频率下大约可以达到 33.5 TFLOPs/s。这只有在所有 CUDA 核心都被使用时才成立。对于 `N = 1024`，最多只能使用 1024 个 CUDA 核心，也就是 8 个 SM，因此耗时会更长（假设计算受限，大约长 16 倍）。此外，内存带宽为每秒 3.35e12 字节。所以硬件峰值强度为 `33.5e12 / 3.35e12 = 10`。[^ch12-13] 因此，我们会严重受内存带宽限制。运行时间就是

$$
T = \max(T_\text{comms}, T_\text{math}) = \frac{12 \cdot N}{\text{3.35e12}} = \frac{N}{\text{2.8e11}}
$$

对于 `N = 65,536`，结果约为 0.23us。实际在 JAX 中测得约 1.5us，这很合理，因为这里预计会严重受延迟限制。对于 `N = 1024 * 1024 * 1024`，Roofline 约为 3.84ms，而实测为 4.1ms，表现不错！

</details>

<span id="networking"></span>

## 网络互连

网络互连是 GPU 与 TPU 差异最大的领域之一。正如我们已经看到的，TPU 连接成二维或三维环面拓扑，每个 TPU 只与相邻 TPU 直接相连。这意味着，在两个 TPU 之间发送消息时，消息必须经过中间的每一个 TPU，也迫使我们在整个网格上只能使用均匀的通信模式。这在某些方面并不方便，但也意味着每个 TPU 的链路数量保持不变，因此可以扩展到任意大的 TPU“Pod”而不会损失带宽。

GPU 则采用一种更传统的分层树状交换网络。每组 8 个 GPU 称为一个**节点**（GB200 最多可达 72 个[^ch12-14]），节点内的 GPU 通过名为 NVLink 的高带宽互连在 1 跳之内彼此连接；这些节点再通过连接到每个 GPU 的网络接口卡（NIC），使用带宽较低的 InfiniBand（IB）或以太网连接成更大的单元（称为 **SU**，即 Scalable Unit）。这些单元还可以借助更高层交换机，进一步连接成任意规模的系统。

![图：典型 H100 网络的示意图。一组 8 个 GPU 通过 NVSwitch（也称 NVLink 交换机）连接成一个节点或 NVLink 域，这些节点再通过交换式 InfiniBand 结构彼此连接。在 NVLink 域内，每个 H100 约有 450GB/s 出方向带宽，而每个节点进入 IB 网络的出方向带宽为 400GB/s。](/images/scaling-book/gpu/superpod-diagram.png)

<span id="at-the-node-level"></span>

### 节点层级

GPU 节点是一个小型单元，通常包含 8 个 GPU（GB200 最多可达 72 个），这些 GPU 通过全互连、全带宽、低延迟的 NVLink 互连连接。[^ch12-15] 每个节点都包含多个高带宽 NVSwitch，负责在所有本地 GPU 之间交换数据包。节点层级的实际拓扑随时间发生了很大变化，包括每个节点的交换机数量；不过在 H100 中，每个节点有 4 个 NVSwitch，GPU 按 `5 + 4 + 4 + 5` 的链路模式与之相连，如下图所示：

![图：从 Pascal（P100）开始各代节点（即 NVLink 域）的结构图。从 Volta（V100）起，我们通过一组交换机在节点内实现了全互连。H100 节点有 4 个 NVSwitch，通过 25GB/s 链路连接全部 8 个 GPU。](/images/scaling-book/gpu/nvlink-nodes.png)

在 Hopper 代（NVLink 4.0）中，每条 NVLink 链路都有 25GB/s 全双工[^ch12-16]带宽（B200 为 50GB/s），因此每个 GPU 进入网络的全双工带宽为 `18 * 25=450GB/s`。巨大的 NVSwitch 最多拥有 64 个 NVLink 端口，这意味着一个带 4 个交换机的 8xH100 节点最多可以处理 `64 * 25e9 * 4=6.4TB/s` 带宽。下面概览了这些数据随 GPU 代际的变化：

| NVLink 代际 | NVSwitch 代际 | GPU 架构代际 | NVLink 带宽（GB/s，全双工） | 每个 GPU 的 NVLink 端口数 | 节点内 GPU 到 GPU 带宽（GB/s，全双工） | 节点规模（NVLink 域） | 每节点 NVSwitch 数 |
| :--------: | :----------: | :------------: | :----------------------------------: | :----------------: | :------------------------------------------: | :-----------------------: | :-----------------: |
|  **3.0**   |   **2.0**    |     Ampere     |                  25                  |         12         |                     300                      |             8             |          6          |
|  **4.0**   |   **3.0**    |     Hopper     |                  25                  |         18         |                     450                      |             8             |          4          |
|  **5.0**   |   **4.0**    |   Blackwell    |                  50                  |         18         |                     900                      |           8/72            |        2/18         |

Blackwell（B200）的节点包含 8 个 GPU。GB200NVL72 支持由 72 个 GPU 组成的更大 NVLink 域。这里同时列出了 8 GPU 系统和 72 GPU 系统的详细信息。

<span id="quiz-2-gpu-nodes"></span>

### 小测 2：GPU 节点

下面还有几道关于网络互连的问答题。我觉得这些题特别值得亲手算一遍，因为它们会迫使你实际推演通信模式。

<strong>问题 1［H100 节点的总带宽］：</strong>在一个带 4 个交换机的 8xH100 节点中，每个节点共有多少带宽？<em>提示：</em>需要同时考虑 NVLink 和 NVSwitch 带宽。

<details>
<summary>点击此处查看答案。</summary>


<strong>答案：</strong>我们有 4 个 Gen4 NVSwitch，每个交换机的单向带宽为 `64 * 25e9=1.6TB/s`。这样在交换机层级可以得到 `4 * 1.6e12=6.4e12` 的带宽。不过要注意，每个 GPU 最多只能处理 450GB/s 单向带宽，因此总带宽至多为 `450e9 * 8 = 3.6TB/s`。由于后者更小，峰值带宽为 3.6TB/s。

</details>

**问题 2［对分带宽］**：对分带宽的定义，是把网络均匀划分时任意一种划分方式所能获得的最小带宽。换句话说，如果把一个网络分成大小相等的两半，有多少带宽能够跨越这两半？你能计算 8x H100 节点的对分带宽吗？<em>提示：</em>对分带宽通常包括两个方向的流量。

<details>
<summary>点击此处查看答案。</summary>


<strong>答案：</strong>任何均匀划分都会在两侧各有 4 个 GPU，每一侧都可以向另一侧送出 `4 * 450GB/s`。把两个方向的流量都计算在内，跨越分区的字节速率为 `8 * 450GB/s`，即对分带宽为 3.6TB/s。这也是 NVIDIA 在[这里](https://hc34.hotchips.org/assets/program/conference/day2/Network%20and%20Switches/NVSwitch%20HotChips%202022%20r5.pdf)等处报告的数值。

</details>

**问题 3［AllGather 开销］**：给定一个包含 B 字节的数组，在一个 8xH100 节点上，一次（受吞吐量限制的）AllGather 要运行多久？请对 bf16[D<sub>X</sub>, F] 进行计算，其中 `D=4096`、`F=65,536`。*回答之前，值得先阅读 TPU 集合通信的[章节](../03-sharding/)。请先自行推演，下一节我们会详细讨论集合通信。*

<details>
<summary>点击此处查看答案。</summary>


<strong>答案：</strong>每个 GPU 可以送出 450GB/s，并持有 $B / N$ 字节（其中 `N=8`，即节点规模）。可以设想每个节点依次把自己的字节发送给其他 $N - 1$ 个节点，总共进行 (N - 1) 轮，每轮的 $T_\text{comms} = (B / (N * W_\text{unidirectional}))$；因此 $T_\text{comms} = (N - 1) * B / (N * W_\text{unidirectional})$。近似为 $B / W_\text{uni}$，即 $B / \text{450e9}$。

对于题目给定的数组，`B = 4096 * 65536 * 2 = 536e6` 字节，因此总时间为 `536e6 * (8 - 1) / (8 * 450e9) = 1.04ms`（使用近似式则为 `536e6 / 450e9 = 1.19ms`）。这项操作可能受延迟限制，所以实践中耗时可能更长（实际约为 1.5ms）。

</details>

<span id="beyond-the-node-level"></span>

## 跨越节点层级

跨出节点之后，GPU 网络拓扑就没有那么标准化了。NVIDIA 发布了一套[参考 DGX SuperPod 架构](https://docs.nvidia.com/dgx-superpod/reference-architecture-scalable-infrastructure-h100/latest/network-fabrics.html)，它使用 InfiniBand 连接比单个节点更多的 GPU；但客户和数据中心提供商可以根据自身需求自由定制。[^ch12-17]

下图展示了一套参考的 1024 GPU H100 系统。底行中的每个方框都是一个 8xH100 节点，包含 8 个 GPU、8 个 400Gbps CX7 NIC（每个 GPU 一个）和 4 个 NVSwitch。

![图：参考的 1024 GPU H100 DGX SuperPod 示意图。它包含 128 个节点（有时为 127 个），每个节点有 8 个 H100 GPU，并通过 InfiniBand 横向扩展网络连接。每组 32 个节点（256 个 GPU）称为一个“可扩展单元”（Scalable Unit，SU）。叶 IB 交换机和脊 IB 交换机提供了足够的带宽，保证节点之间拥有完整的对分带宽。](/images/scaling-book/gpu/h100-superpod.png)

<strong>可扩展单元：</strong>每组 32 个节点称为一个“可扩展单元”（Scalable Unit，简称 SU），位于同一组 8 个 InfiniBand 叶交换机之下。一个 SU 有 256 个 GPU，每个节点配 4 个 NVSwitch，整个 SU 配 8 个 Infiniband 叶交换机。图中所有线缆均为 InfiniBand NDR（50GB/s 全双工），交换机则是 64 端口 NDR IB 交换机（每端口同样为 50GB/s）。*请注意，IB 交换机的带宽是 NVSwitch 的 2 倍（64 个端口，使用 400 Gbps 链路）。*

<strong>SuperPod：</strong>整个 SuperPod 再通过 16 个顶层“脊”IB 交换机连接 4 个这样的 SU，从而得到 1024 个 GPU、512 个节点层级 NVSwitch、32 个叶 IB 交换机和 16 个脊 IB 交换机，总计 512 + 32 + 16 = 560 个交换机。叶交换机按每组 32 个节点的方式连接节点，因此每组 256 个 GPU 配有 8 个叶交换机。所有叶交换机都连接到所有脊交换机。

**我们有多少带宽？**InfiniBand 网络（称为“横向扩展网络”，即 scale-out network）的整体拓扑是一棵**胖树**；线缆和交换机保证在节点层级之上拥有完整的对分带宽（这里为 400GB/s）。这意味着，如果把节点分成两半，每个节点都可以同时以 400GB/s 向另一分区中的一个节点送出数据。更重要的是，这意味着在横向扩展网络中，AllReduce 带宽应当大致保持恒定！实际实现未必如此，但你可以设想在横向扩展网络中的任意多个节点上执行环形归约，因为总能构造出一个包含所有节点的环。

| 层级 | GPU 数量 | 每单元交换机数 | 交换机类型 | 每单元带宽（TB/s，全双工） | GPU 到 GPU 带宽（GB/s，全双工） | 胖树带宽（GB/s，全双工） |
| :---: | :------------: | :-------------------------: | :---------: | :------------------------------------------: | :--------------------------------------: | :---: |
| 节点  |       8        |              4              |     NVL     |                     3.6                      |                   450                    | 450
| 叶  |      256       |              8              |     IB      |                     12.8                     |                    50                    | 400 |
| 脊 |      1024      |             16              |     IB      |                     51.2                     |                    50                    | 400 |

作为对比，TPU v5p 每条链路约有 90GB/s 出方向带宽，或者说沿三维环面拓扑所有轴合计有 540GB/s 出方向带宽。它并非点到点互连，因此只能用于受限且均匀的通信模式；但它仍能提供高得多的 TPU 到 TPU 带宽，而且可以扩展到任意大的拓扑（至少能达到 8960 个 TPU）。

理论上，通过增加额外交换机或增加间接层级，可以把 GPU 交换结构扩展到任意规模；代价则是额外延迟和昂贵的网络交换机。

**要点**：在 H100 节点内，每个 GPU 都拥有 450GB/s 的完整胖树带宽；跨越节点之后，这一带宽下降为节点间 400GB/s。事实会证明，这一点对通信原语至关重要。

<strong>GB200 NVL72：</strong>NVIDIA 最近开始生产新的 GB200 NVL72 GPU 集群，在单个 NVLink 域中组合 72 个 GPU，并提供完整的 900GB/s GPU 到 GPU 带宽。随后可以把这些域连接成更大的 SuperPod，并配备按比例提高（9 倍）的 IB 胖树带宽。其拓扑如下图所示：

![图：GB200 DGX SuperPod 的示意图，共包含 576 个 GPU。底层的每个机架包含 72 个 GB200 GPU。](/images/scaling-book/gpu/gb200-superpod.png)

计算单个节点的出方向带宽（上图中的橙色线），可以得到通往叶层级的带宽为 `4 * 18 * 400 / 8 = 3.6TB/s`，是 H100 的 9 倍（节点中的 GPU 数量也恰好是 9 倍）。这意味着，关键的节点出方向带宽要高得<em>多得多</em>，而跨节点集合通信带宽实际上可能<em>低于</em>节点内带宽。
更多讨论请参阅[附录 A](#appendix-a-how-does-this-change-with-gb200)。

|  节点类型  | 每节点 GPU 数 | GPU 出方向带宽 | 节点出方向带宽 |
| :---------: | :-----------: | :------------------: | :-------------------: |
|    H100     |       8       |        450e9         |         400e9         |
|    B200     |       8       |        900e9         |         400e9         |
| GB200 NVL72 |      72       |        900e9         |        3600e9         |

**要点**：GB200 NVL72 SuperPod 大幅提高了节点规模和单个节点的出方向带宽，从而显著改变了我们的 Roofline。

<span id="quiz-3-beyond-the-node-level"></span>

### 小测 3：跨越节点层级

<strong>问题 1［胖树拓扑］：</strong>使用上面的 DGX H100 示意图，计算整个 1024 GPU Pod 在节点层级的对分带宽。证明每条链路的带宽都经过选择，可以确保完整的对分带宽。*提示：请务必同时计算链路带宽和交换机带宽。*

<details>
<summary>点击此处查看答案。</summary>


<strong>答案：</strong>我们逐个组件来计算：

* 首先，每个节点都有 8 条 400Gbps NDR IB 线缆连接到叶交换机，因此每个节点通往叶层级的带宽为 `8 * 400 / 8 = 400 GB/s`。我们有 8 个叶交换机，每个带宽为 3.2TB/s（64 条 400 GBps 链路）；但 64 个端口中只有 32 个可用于接收来自 SU 的流量，因此 32 个节点的带宽为 `32 * 400 / 8 = 12.8TB/s`，仍恰好是每节点 400GB/s。
* 接着，在脊层级，每个 SU 通过 `8 * 16 * 2` 条 400Gbps NDR IB 线缆连接到脊交换机，因此每个 SU 通往叶层级的带宽为 `8 * 16 * 2 * 400 / 8 = 12.8 TB/s`。同样，每节点为 400GB/s。我们有 16 个脊交换机，每个带宽为 3.2TB/s，总计 `16 * 3.2 = 51.2 TB/s`；分摊到 128 个节点，仍然是每节点 400GB/s。

因此，无论以何种方式把节点对分，两侧之间都能达到每节点 400GB/s。每个组件都恰好具备构成胖树所需的带宽。

</details>

<strong>问题 2［扩展到更大的 DGX Pod］：</strong>假设我们想使用 2048 个而非 1024 个 GPU 进行训练。修改上述 DGX 拓扑以支持这种规模时，最简单或最好的办法是什么？4096 个 GPU 时又该怎么做？*提示：不存在唯一正确答案，但请设法控制成本，并注意链路容量。[这份](https://docs.nvidia.com/dgx-superpod-reference-architecture-dgx-h100.pdf)文档可能会有帮助。*

<details>
<summary>点击此处查看答案。</summary>


<strong>答案：</strong>一种方案是保留 SU 结构不变（8 个交换机下连接 32 个节点），只增加更多 SU 和更多顶层交换机。我们需要 2 倍数量的脊交换机，这样用 8 个 SU 和 32 个脊交换机就能提供足够带宽。

这种方案的一个问题是，每个叶交换机只有 64 个端口，而上图中已经用完了所有端口。不过，我们很容易把每个脊交换机上的 2 条 400 Gbps NDR 线缆改成 1 条，这样可以提供相同的总带宽，同时节省一些端口。

当规模达到 4096 个 GPU 时，端口确实会耗尽，因此需要增加一级间接层——也就是说，在层次结构中再增加一层。NVIDIA 称之为“核心交换机”，并使用 128 个脊交换机和 64 个核心交换机来构建 4096 GPU 集群。你可以自行计算，证明这样能够提供足够带宽。

</details>

<span id="how-do-collectives-work-on-gpus"></span>

## GPU 上的集合通信如何工作？

GPU 可以执行与 TPU 完全相同的集合通信操作：ReduceScatter、AllGather、AllReduce 和 AllToAll。与 TPU 不同，这些操作的工作方式会因其是在节点层级（通过 NVLink）执行，还是在更高层级（通过 InfiniBand）执行而发生变化。NVIDIA 在 [NVSHMEM](https://developer.nvidia.com/nvshmem) 和 [NCCL](https://developer.nvidia.com/nccl)（读作“nickel”）库中实现了这些集合通信操作。NCCL 的开源代码在[这里](https://github.com/NVIDIA/nccl)。NCCL 会根据延迟要求和拓扑采用多种实现（[详情](https://github.com/NVIDIA/nccl/issues/1415#issuecomment-2310650081)）；从这里开始，我们将讨论交换式树状结构上的理论最优模型。

<span id="intra-node-collectives"></span>

### 节点内集合通信

<strong>AllGather 或 ReduceScatter：</strong>在节点层级执行 AllGather 或 ReduceScatter 时，可以像 TPU 一样沿环形拓扑完成，并在每一跳使用完整的 GPU 到 GPU 带宽。以任意顺序排列 GPU，然后利用完整的 GPU 到 GPU 带宽，沿环发送数组的一部分。[^ch12-18] 每一跳的开销为 $T_\text{hop} = \text{bytes} / (N * \text{GPU egress bandwidth})$，所以总开销为

$$
T_\text{AG or RS comms} = \frac{\text{bytes} \cdot (N - 1)}{N \cdot \text{GPU egress bandwidth}} \rightarrow \frac{\text{bytes}}{\text{GPU egress bandwidth}}
$$

你会注意到，这与 TPU 上的情况完全相同。对于 AllReduce，可以像往常一样组合一次 RS 和一次 AG，开销翻倍。

![图：带宽最优的一维环形 AllGather 算法。对于 B 字节的数据，该算法会通过顶层交换机发送 B / X 字节，共发送 X - 1 次。](/images/scaling-book/gpu/all-gather.gif)

如果担心延迟（例如数组非常小），可以执行树状归约：先在每 2 个一组的 GPU 内执行 AllReduce，再扩展到每 4 个、每 8 个一组；这样总共只需 $\log(N)$ 跳，而不是 $N - 1$ 跳，不过总开销仍然相同。

<strong>要点：</strong>在单个节点内对 B 字节数组执行 AllGather 或 ReduceScatter 的开销约为 $T_\text{comms} = B * (8 - 1) / (8 * W_\text{GPU egress}) \approx B / W_\text{GPU egress}$。理论上，在 H100 上约为 $B  / \text{450e9}$，在 B200 上约为 $B / \text{900e9}$。除非启用网内归约，否则 AllReduce 的开销是其 2 倍。

<strong>随堂小测 1［AllGather 时间］：</strong>使用一个全双工带宽为 450 GB/s 的 8xH100 节点时，AllGather(bf16[B<sub>X</sub>, F]) 需要多长时间？令 $B=1024$、$F=16,384$。

<details>
<summary>点击此处查看答案。</summary>


<strong>答案：</strong>总数据量为 $2 \cdot B \cdot F$ 字节，单向带宽为 450e9。粗略计算需要 $T_\text{comms} = (2 \cdot B \cdot F) / \text{450e9}$，更精确地说则是 $(2 \cdot B \cdot F \cdot (8 - 1)) / (8 \cdot \text{450e9})$。代入题目给出的数值，粗略结果为 $(2 \cdot 1024 \cdot 16384) / \text{450e9} = \text{75us}$，更精确的结果为 $\text{65us}$。

</details>

<strong>AllToAll：</strong>节点内 GPU 彼此全互连，这使 AllToAll 变得相当简单——嗯，确实如此。每个 GPU 只需直接向目标节点发送数据。对于节点内的 B 字节数据，每个 GPU 持有 $B / N$ 字节，并向目标节点各发送 $(B / N^2)$ 字节，目标节点共有 $N - 1$ 个，因此总开销为

$$
T_\text{AllToAll comms} = \frac{B \cdot (N - 1)}{W \cdot N^2} \approx \frac{B}{W \cdot N}
$$

相比之下，TPU 上的开销是 $B / (4W)$。因此，在单个节点内，理论运行时间可获得 2 倍加速（$B / 4W$ 对比 $B / 8W$）。

对于混合专家模型（Mixture of Experts，MoE），我们经常希望执行一种*稀疏或变长（ragged）AllToAll*；我们保证输出维度上最多有 $k$ 个非零分片（共 $N$ 个）。也就是说，$T_\text{AllToAll} \rightarrow K[B, N]$，并且每条轴上最多有 $k$ 个非零条目（共 $N$ 个）。其开销会按 $k/N$ 的比例下降，总计约为 $\min(k/N, 1) \cdot B / (W \cdot N)$。在 MoE 中，我们通常会独立、随机地选择非零值，因此实际非零数可能少于 $k$，从而近似得到
$(N-1)/N \cdot \min(k/N, 1) \cdot B / (W \cdot N)$。[^ch12-19]

<strong>随堂小测 2［AllToAll 时间］：</strong>使用一个单向带宽为 450 GB/s 的 8xH100 节点时，AllToAll<sub>X->N</sub>(bf16[B<sub>X</sub>, N]) 需要多长时间？如果已知 8 个条目中只有 4 个非零，结果又如何？

<details>
<summary>点击此处查看答案。</summary>


<strong>答案：</strong>请注意，这里的 $B$ 是数组的批处理维度，因此数组总大小为 $V = 2 \cdot B \cdot N$ 字节。根据前面的结论可知，在稠密情况下，开销为 $V \cdot (N-1) / (W \cdot N^2)$，近似为 $V / (W \cdot N)$。如果已知只有 $\frac{1}{2}$ 的条目不是填充值，那么可以发送 $V \cdot k/N / (W \cdot N) = V / (2 \cdot W \cdot N)$，约为总开销的一半。

</details>

<strong>要点：</strong>在单个节点内的 GPU 上，对 $B$ 字节数组执行 AllToAll 的开销约为 $T_\text{comms} = (B \cdot (8 - 1)) / (8^2 \cdot W_\text{GPU egress}) \approx B / (8 \cdot W_\text{GPU egress})$。对于变长（top-$k$）AllToAll，开销会进一步下降到 $(B \cdot k) / (64 \cdot W_\text{GPU egress})$。

<strong>实测结果：</strong>下面是在一个 8xH100 节点上测得的 AllReduce 带宽。Algo BW 是实测带宽（字节数 / 运行时间），Bus BW 则按 $2 \cdot W \cdot (8 - 1) / 8$ 计算，理论上衡量的是实际链路带宽。可以看到，我们确实达到了接近 370GB/s；虽然低于 450GB/s，但已相当接近，不过这要到每设备约 10GB 的消息大小才能实现。这意味着，尽管这些估算在理论上正确，却需要很大的消息才能真正达到相应性能。

![图：关闭 SHARP 时，8xH100 节点的 AllReduce 吞吐量。蓝色曲线是根据实测结果按 $2 * \text{bytes} * (N - 1) / (N * \text{runtime})$ 计算的实际链路带宽。请注意，即使数组大到 10GB，我们仍未特别接近宣称的 450GB/s 带宽。](/images/scaling-book/gpu/gpu-all-reduce-bw.png)

这是一个实实在在的问题，因为它使我们能够提出的任何理论断言都明显变复杂。例如，即便对大小合理的数组执行 AllReduce——比如 LLaMA-3 70B 的 MLP（大小为 `bf16[8192, 28672]`，或者在 8 路模型分片下为 `bf16[8192, 3584] = 58MB`）——与 450GB/s 峰值相比也只能达到约 150GB/s。相比之下，TPU 在消息小得多时就能达到峰值带宽（参见附录 B）。

<strong>要点：</strong>虽然 NVIDIA 宣称 H100 NVLink 的带宽约为 450GB/s，但实践中很难超过 370 GB/s，因此需要相应调整前面的估算。

<strong>网内归约：</strong>从 Hopper 代开始，NVIDIA 交换机支持 [“SHARP”（Scalable Hierarchical Aggregation and Reduction Protocol）](https://developer.nvidia.com/blog/advancing-performance-with-nvidia-sharp-in-network-computing/)，从而可以执行“网内归约”。这意味着*网络交换机本身*能够执行归约操作，并把结果多路复用或“MultiCast”到多个目标 GPU：

![图：不使用 SHARP 的 AllReduce 理论开销为 2 倍，因为数据必须两次经过每个 GPU。实践中的加速只有约 30%（使用 NCCL 2.27.5 测得）。](/images/scaling-book/gpu/sharp-algorithm.png)

理论上，这几乎能把 AllReduce 开销减半：每个 GPU 可以把数据发给顶层交换机，由交换机自身执行归约并把结果广播到每个 GPU，无需让数据两次从每个 GPU 送出，同时还能降低网络延迟。

$$
T_\text{SHARP AR comms} = \frac{\text{bytes}}{\text{GPU egress bandwidth}}
$$

请注意，这是精确结果，并没有差一个 $1/N$ 因子：每个 GPU 首先送出 $B \cdot (N - 1) / N$，然后接收其本地分片的部分归约版本（接收 $B/N$），完成归约后再次送出 $B/N$，最后接收完全归约的结果（接收 $B \cdot (N - 1) / N$），最终接收的总量恰好是 $B$ 字节。

然而在实践中，启用 SHARP 后观察到的带宽增幅约为 30%，而非预测的 75%。这只能让有效集合通信带宽达到约 480GB/s，远未接近 2 倍。

![图：节点内启用与不启用 NVIDIA SHARP 时的 AllReduce 算法带宽实测结果。峰值吞吐量提升约为 30%，尽管从算法上说，它本应能获得接近 75% 的增益。](/images/scaling-book/gpu/sharp-all-reduce-cost.png)

<strong>要点：</strong>理论上，NVIDIA SHARP（多数 NVIDIA 交换机均可用）应能把对 $B$ 字节执行 AllReduce 的开销从约 $2 * B / W$ 降低到 $B / W$。然而实践中，带宽只提升约 30%。由于纯 AllReduce 在 LLM 中相当少见，因此这并不是特别有用。

<span id="cross-node-collectives"></span>

### 跨节点集合通信

跨越节点层级之后，开销会微妙一些。在树状结构上执行归约时，可以把它看成从底向上进行：先在节点内归约，再在叶层级归约，最后在脊层级归约，每个层级都采用常规算法。对于 AllReduce 尤其如此：这种方式可以减少总通信数据量，因为在节点层级完成 AllReduce 后，向叶层级送出的只需 $B$ 字节，而不是 $B * N$。

<strong>开销有多大？</strong>作为一阶近似，由于我们拥有完整的对分带宽，AllGather 或 ReduceScatter 的开销大致等于缓冲区字节数除以节点出方向带宽（H100 上为 400GB/s），而且*与树状归约的任何具体细节都无关*。

$$
T_\text{AG or RS comms} = \frac{\text{bytes}}{W_\text{node egress}} \underset{H100}{=} \frac{\text{bytes}}{\text{400e9}}
$$

对于上述 H100 网络，$W_\text{node}$ 出方向带宽通常为 400GB/s（每个节点有 8 条向外连接的 400Gbps IB 链路）。最直观的理解方式，是设想在*集群中的每个节点*上执行环形归约。由于采用胖树拓扑，我们总能在任意两个节点之间构造一个拥有 $W_\text{node}$ 出方向带宽的环，并执行常规归约。节点层级归约（几乎）永远不会成为瓶颈，因为它拥有更高的总带宽和更好的延迟；不过一般而言，其开销为

$$
T_\text{total} = \max(T_\text{comms at node}, T_\text{comms in scale-out network}) = \max\left[\frac{\text{bytes}}{W_\text{GPU egress}}, \frac{\text{bytes}}{W_\text{node egress}}\right]
$$

<details>
<summary>可以在这里查看更精确的推导。</summary>


更精确地说，我们实际上是在网络的每个层级执行环形归约，而且这些归约大多可以重叠，因此有：

$$
T_\text{AG or RS comms} = \text{bytes} \cdot max_\text{depth i}\left[\frac{D_i - 1}{D_i \cdot W_\text{link i}}\right]
$$

其中，$D_i$ 是深度 $i$ 处的度（即深度 $i$ 处的子节点数量），$W_\text{link i}$ 是把每个子节点连接到节点 $i$ 的链路带宽。

据此，对于给定拓扑，可以把可用 AllGather/AllReduce 带宽计算为 $min_\text{depth i}(D_i * W_\text{link i} / (D_i - 1))$。在上面的情形中：

* **节点：**$D_\text{node}$ = 8，因为一个节点中有 8 个 GPU，且 Wlink i = 450GB/s。因此 AG 带宽为 `450e9 * 8 / (8 - 1) = 514GB/s`。
* **叶：**$D_\text{leaf}$ = 32，因为一个 SU 中有 32 个节点，且 Wlink i = 400GB/s（8 条 400Gbps IB 链路）。因此带宽为 `400e9 * 32 / (32 - 1) = 413GB/s`。
* **脊：**$D_\text{spine}$ = 4，因为有 4 个 SU，且 $W_\text{link i}$ = 12.8TB/s（来自上面的 `8 * 16 * 2 * 400Gbps` 条链路）。因此带宽为 `12.8e12 * 4 / (4 - 1) = 17.1TB/s`。

所以，叶层级的总体 AG 或 RS 带宽为 `min(514GB/s, 413GB/s, 17.1TB/s) = 413GB/s`；实践中 $T_\text{AG or RS comms} = B / \text{413GB/s}$。也就是说，即使在最高层级，AllReduce 带宽也约为 413GB/s。对于启用 SHARP 的 AllReduce，由于没有 $(N - 1) / N$ 因子，带宽会略低于这一数值（约为 400GB/s）。尽管如此，450GB/s 和 400GB/s 已经足够接近，可以作为近似值使用。

</details>

<strong>其他集合通信：</strong>除非启用 SHARP，否则 AllReduce 的开销仍然是上述数值的 2 倍。NVIDIA 也销售支持 SHARP 的 IB 交换机，不过并非所有提供商都会部署。跨节点时，AllToAll 的变化要大得多，因为它不像 AllReduce 那样具有“分层”特性。如果希望把数据从每个 GPU 发送到其他每个 GPU，就无法充分利用节点层级的完整对分带宽。这意味着，如果一次 N 路 AllToAll 跨越 $M = N / 8$ 个节点，其开销为

$$
T_\text{AllToAll comms} = \frac{B \cdot (M - 1)}{M^2 \cdot W_\text{node egress}} \approx \frac{B}{M \cdot W_\text{node egress}}
$$

其有效带宽是 50GB/s，而不是 400GB/s。在单个 H100 节点内，开销为 $B / (8 * \text{450e9})$；跨越 2 个节点时则变为 $B / (2 \cdot \text{400e9})$，性能下降超过 4 倍。

下面汇总了 1024-GPU DGX H100 SuperPod 的架构：

|   层级   | GPU 数量 | 度（子节点数） | 交换机带宽（全双工，TB/s） | 线缆带宽（全双工，TB/s） | 集合通信带宽（GB/s） |
| :-------: | :------------: | :-----------------: | :----------------------------------: | :---------------------------------: | :-------------------------: |
|   节点    |       8        |          8          |                 6.4                  |                 3.6                 |             450             |
| 叶（SU） |      256       |         32          |                 25.6                 |                12.8                 |             400             |
|   脊   |      1024      |          4          |                 51.2                 |                51.2                 |             400             |

这里使用“集合通信带宽”一词，表示 GPU 或节点能够向外发送数据的有效带宽。它也等于 $\text{bisection bandwidth} * 2 / N$。

<strong>要点：</strong>在节点层级之上，对 B 字节执行 AllGather 或 ReduceScatter 的开销约为 $B / W_\text{node egress}$；在 H100 DGX SuperPod 上即为 $B / \text{400e9}$。除非启用 SHARP，否则 AllReduce 的开销是其两倍。整体拓扑是一棵胖树，旨在让任意两对节点之间都能获得恒定带宽。

<strong>数组沿另一条轴分片时的归约：</strong>考虑如下归约的开销：

$$
\text{AllReduce}_X(A[I_Y, J]\ \{ U_X \})
$$

这里，我们正沿 X 对一个本身又沿另一条轴 $Y$ 分片的数组执行 AllReduce。在 TPU 上，由于每条轴发送的数据量都是原来的 $1 / Y$，所以与未分片版本相比，这项操作的总开销会缩小 $1 / Y$ 倍。在 GPU 上，开销取决于哪条轴是“内轴”（节点内还是节点间），以及每个分片是否跨越不止一个节点。假设 $Y$ 是内轴，并且数组总共有 $\text{bytes}$ 字节，则总开销实际上会缩小 $Y$ 倍，但这只在 $Y$ 跨越多个节点时成立：

$$
T_\text{comms at node} = \frac{\text{bytes}}{W_\text{GPU egress}} \cdot \frac{1}{\min(Y, D_\text{node})}
$$

$$
T_\text{comms in scale-out network} = \frac{\text{bytes}}{W_\text{node egress}} \cdot \frac{D_\text{node}}{\max(D_\text{node}, Y)}
$$

$$
T_\text{total} = \max(T_\text{comms at node}, T_\text{comms in scale-out network})
$$

其中 N 是 GPU 数量，$D_\text{node}$ 仍是一个节点中的 GPU 数量（即节点的度）。可以看到，当 $Y < D_\text{node}$ 时，我们在节点层级获得了收益，但总运行时间通常不会减少；当 $Y > D_\text{node}$ 时，则会获得与跨越节点数成正比的加速。

如果要精确考虑环形归约，对于树状 AllGather<sub>X</sub>(A<sub>Y</sub> { U<sub>X</sub> })（假设 Y 是内轴），一般规律为

$$
T_\text{AR or RS comms} = \text{bytes} \cdot \max_{\text{depth } i}\left[\frac{D_i - 1}{D_i \cdot \max(Y, S_{i-1}) \cdot W_{\text{link } i}}\right]
$$

其中，$S_i$ 是 M * N * …，也就是树中第 i 层以下子节点的规模。粗略来说，这表示我们跨越的 GPU 或节点越多，可用带宽就越高，但这种提升只发生在相应节点内。

<strong>随堂小测 3［沿两条轴分片］：</strong>假设要在单个 SU（256 个芯片）上执行 $\text{AllGather}_X(\text{bf16}[D_X, F_Y])$，其中 $Y$ 是内轴。用 $D$、$F$ 和 $Y$ 表示时，这项操作需要多长时间？

<details>
<summary>点击此处查看答案。</summary>


<strong>答案：</strong>可以分成两种情况：Y <= 8 和 Y > 8。当 $Y <= 8$ 时，我们仍受叶交换机限制，所以答案与往常一样，是 $T_\text{comms} = 2 * D * F * (32 - 1) / (32 * 400e9)$。当 Y > 8 时，根据上面的结果，近似有

$$
T_\text{comms} = \frac{2 \cdot D \cdot F \cdot 256}{Y \cdot \text{12.8e12}} = \frac{2DF}{Y \cdot \text{50GB/s}}
$$

对于 `D = 8192`、`F = 32,768`，结果如下：

![图：内轴跨越更多节点时，分片 AllGather 的理论开销。](/images/scaling-book/gpu/sharded-all-gather-cost.png)

请注意，如果恰好执行 8 路模型并行，确实会把节点层级归约的开销降低 8 倍，但总开销不变；因此它是免费的，却无助于提高总体带宽。

</details>

<strong>要点：</strong>当沿多条轴进行分片时，外层归约的开销会除以内轴所跨越的节点数。

<span id="quiz-4-collectives"></span>

### 小测 4：集合通信

<strong>问题 1［SU AllGather］：</strong>只考虑一个包含 M 个节点、每节点有 N 个 GPU 的 SU。在一次 AllGather 期间，节点层级交换机准确地接收和送出了多少字节？顶层交换机呢？

<details>
<summary>点击此处查看答案。</summary>


<strong>答案：</strong>我们逐步分析归约的各个组成部分：

1. 每个 GPU 向交换机发送 $B / MN$ 字节，总接收量为 $NB / MN = B / M$ 字节。
2. 我们向脊交换机送出完整的 $B / M$ 字节。
3. 我们从脊交换机接收 $B * (M - 1) / M$ 字节。
4. 我们把 $B - B / MN$ 字节送出 $N$ 次，总计 $N * (B - B / MN) = NB - B / M$。

接收总量为 $B$，送出总量为 $BN$，所以瓶颈应当是送出方向，总时间为 $T_\text{AllGather} = BN / W_\text{node} = B / \text{450e9}$。

对于脊交换机，计算实际上更简单。必须接收 M 次 $B / M$ 字节（总计 $B$ 字节），然后把 $B (M - 1) / M$ 送出 $M$ 次，总计送出 $B * (M - 1)$。由于后者大得多，开销为 $T_\text{AllGather} = B \cdot (M - 1) / (M \cdot W_\text{node}) = B \cdot (M - 1) / (M \cdot \text{400e9})$。

</details>

<strong>问题 2［单节点 SHARP AR］：</strong>考虑一个每节点有 N 个 GPU 的单节点。使用 SHARP（网内归约）执行 AllReduce 时，交换机准确地接收和送出了多少字节？

<details>
<summary>点击此处查看答案。</summary>


<strong>答案：</strong>与前面一样，我们逐步计算。

1. 每个 GPU 发送 $B * (N - 1) / N$ 字节，因此接收总量为 $N * B * (N - 1) / N = B * (N - 1)$。
2. 我们累加部分和，并向每个 GPU 发回 $B / N$ 字节，所以送出总量为 $N * B / N = B$ 字节。
3. 我们在本地对剩余部分求部分和，然后将其发回交换机。接收总量为 $N * B / N = B$ 字节。
4. 我们收集所有分片并对其进行组播，把 $B * (N - 1) / N$ 发送到 $N$ 个目标，因此送出总量为 $B * (N - 1) / N * N = B * (N - 1)$。

所以，接收和送出的总量都是 $B * (N - 1) + B = BN$ 字节。这也支持总吞吐量恰好为 $B / W_\text{egress}$ 的结论。

</details>

<strong>问题 3［跨节点 SHARP AR］：</strong>考虑一个在单节点 N 个 GPU 上分片的数组 bf16[D<sub>X</sub>, F<sub>Y</sub>]。AllReduce(bf16[D, F<sub>Y</sub>] { U<sub>X</sub> }) 需要多长时间？可以假设执行网内归约。请解释跨越多个节点后会有何不同。

<details>
<summary>点击此处查看答案。</summary>


<strong>答案：</strong>可以尝试修改前一问题的答案。大体上，我们首先从每个 GPU 送出 $B * (X - 1) / XY$ 字节，然后向每个 GPU 发回 $B / XY$，再把同样的数据量发回交换机，最后向每个 GPU 发回 $B * (X - 1) / XY$。接收和送出总量都是 $NB / Y$，所以总时间为 $T_\text{comms} = NB / (Y * N * W_\text{link}) = N * 2DF / (Y * N * W_\text{link}) = 2 * D * F / (Y * W_\text{link})$；因此总时间确实会随 $Y$ 增大而下降。

跨越单个节点之后，可以执行与上面大致相同的归约；但从节点层级交换机送出数据时，必须发送全部 B 字节，而不只是 $B / Y$。这是因为需要让每个分片彼此分离。

</details>

<strong>问题 4［脊层级 AR 开销］：</strong>考虑与上面相同的设置，但令 $Y = 256$（因此 AR 发生在脊层级）。AllReduce 需要多长时间？同样，可以假设执行网内归约。

<details>
<summary>点击此处查看答案。</summary>


<strong>答案：</strong>这让我们可以利用脊层级大得有些荒谬的带宽。脊层级在 4 个 SU 上共有 51.2TB/s 带宽，即每个 SU 为 12.8TB/s。使用 SHARP 时，耗时最低可达 `2 * D * F / 12.8e12` 秒。

</details>

<strong>问题 5［2 路 AllGather 开销］：</strong>计算恰好跨越 2 个节点、对 $B$ 字节执行 AllGather 的精确开销。*务必计算精确开销而不是近似值，并同时考虑节点内和跨节点开销。*

<details>
<summary>点击此处查看答案。</summary>


<strong>答案：</strong>在节点层级有 $T_\text{comms} = B * 7 / (8 * \text{450e9}) = B / \text{514e9}$；而跨出节点后，实际有 $T_\text{comms} = B * (2 - 1) / (2 * \text{400e9}) = B / \text{800e9}$。因此，真正的瓶颈是节点层级归约，而不是叶层级！这也解释了为什么 DeepSeek v3 等系统会采用 2 路数据并行。

</details>

<span id="rooflines-for-llm-scaling-on-gpus"></span>

## GPU 上 LLM 扩展的 Roofline

现在，让我们来看看前面的所有内容最终是为了什么：理解 GPU 上 LLM 扩展的 Roofline。这部分是对[这里](../05-training/#how-to-parallelize-a-transformer-for-training)的 TPU 训练章节的补充。和在那里一样，我们的目标是考察不同并行策略下的总 $T_\text{math}$ 和 $T_\text{comms}$，并理解 $T_\text{comms} > T_\text{math}$ 会在什么条件下发生。与之前一样，我们只考虑具有以下运算的 MLP 块：

$$
\text{MLP}(x) \equiv x[B, D] *_D W_\text{in}[D, F] \cdot_F W_\text{out}[F, D]
$$

其中 $B$ 是**以 token 计的**全局批大小（即 $B = \text{batch size} \cdot \text{sequence length}$）。

这里我们会再次给出上面的表格，展示 GPU 和节点层级的有效带宽：

|  节点类型  | 每节点 GPU 数 | GPU 出方向带宽 | 节点出方向带宽 |
| :---------: | :-----------: | :------------------: | :-------------------: |
|    H100     |       8       |        450e9         |         400e9         |
|    B200     |       8       |        900e9         |         400e9         |
| GB200 NVL72 |      72       |        900e9         |        3600e9         |

**注意：** GPU 和节点的出方向带宽都会决定 LLM 的 Roofline。我们将使用 $W_\text{collective}$ 这个术语来表示 GPU 或节点带宽，具体取决于我们是在节点内部还是在节点层级以上运行。

让我们像分析 TPU 时那样，考察**数据并行、张量并行、流水线并行、专家并行**及其组合的计算—通信 Roofline。在本节其余部分的具体计算中，我们将重点关注 H100 的 Roofline。GB200-NVL72 的总体 Roofline 相同，但由于其节点出方向带宽更大，我们有时可能反而会在节点层级遇到瓶颈。

<span id="data-parallelism"></span>

### 数据并行

如前所述，DP 和 ZeRO 分片在反向传播中需要进行权重 AllReduce，或者一次 ReduceScatter 加一次 AllGather。由于二者开销相同，对于*不使用网内归约*的纯数据并行或 FSDP，要做到计算受限，在大小为 X 的轴上，反向传播期间每层有：

$$
T_\text{math} = \frac{2 \cdot 2 \cdot 2 \cdot BDF}{X \cdot C}
$$

$$
T_\text{comms} = \frac{2 \cdot 2 \cdot 2 \cdot DF}{W_\text{collective}}
$$

因此，要满足 $T_\text{math} > T_\text{comms}$，我们需要 $B / (XC) > 1 / W_\text{collective}$，也就是

$$
\frac{B}{X} > \frac{C}{W_\text{collective}}
$$

其中 $W_\text{collective}$ 是 GPU 层级还是节点层级的出方向带宽，取决于我们是在节点内还是跨节点进行分片。因此：

* **在节点内**，我们只需要每 GPU 的 **token** 批大小 > $\text{990e12} / \text{450e9} = 2200$。
* **在 SU 内或脊层级**，BS > $\text{990e12} / \text{400e9} = 2475$。

这比 TPU 上高得多；在 TPU 上，使用全部三个轴时，这个数值是 850。例如，在 16000 个 H100 上训练的 LLaMA-3 至少需要 40M token 的批大小（作为参考，他们实际使用的是 16M）。DeepSeek v3 在 2048 个 H800 GPU 上训练，其带宽较低，只有 300GB/s（而不是 H100 上的 450GB/s），因此每个 GPU 需要 $\text{990e12} / \text{300e9} = 3300$ 个 token，也就是总共大约 6.7M（实践中，他们使用的是 4M）。

启用网内归约并使用纯数据并行时，理论上 AllReduce 带宽会提高到 2 倍，从而把这两个数值都减半。但在实践中，收益更接近 30%，这其实只够弥补我们通常难以达到标称数值的差距。此外，由于纯数据并行很少有用，这一点在实践中基本无关紧要。

**MoE 模型：** 对于一个混合专家（Mixture of Experts，MoE）模型，设有 E 个专家、每个 token 激活 k 个专家，上式变为

$$
T_\text{math} = \frac{2 \cdot 2 \cdot 2 \cdot k \cdot BDF}{X \cdot C}
$$

$$
T_\text{comms} = \frac{2 \cdot 2 \cdot 2 \cdot EDF}{W_\text{collective}}
$$

这会让每 GPU 的 token 批大小增大 $E/k$ 倍，即

$$
\frac{B}{X} > \frac{E}{k} \frac{C}{W_\text{collective}}
$$

例如，对于新的 OpenAI OSS 模型，其 $k=4$、$E=128$，跨节点时这个数值会增至 `32 * 2475  = 79,200`，高得有些荒谬。

**X 很小时会怎样？** 当我们只进行例如 2 节点数据并行时，会受益于 $(X - 1) / X$ 的缩放关系，从而得到

$$
T_\text{math} = \frac{2 \cdot 2 \cdot 2 \cdot BDF}{N * C}
$$

$$
T_\text{comms} = \frac{2 \cdot 2 \cdot 2 \cdot DF \cdot (X-1)}{X \cdot W_\text{collective}}
$$

其中 X 是节点数，且 $N = 8 \cdot X$。那么对于稠密模型，有 $B / N > \alpha \cdot (X - 1) / X$；例如 $B / N > \text{1237}$，只有上述数值的一半。正因如此，你会相当频繁地看到 2 路数据并行。

**要点：** 假设通信与计算完美重叠、FLOPs 利用率也达到理想水平，数据并行和 ZeRO 分片要在 H100 或 B200 上达到计算受限，每 GPU 需要约 2500 个 token 的批大小。对于 MoE 模型，这一数值会增大 $E / k$ 倍，也就是总参数量与激活参数量之比。只进行少量数据并行时，临界批大小会下降。

<span id="tensor-parallelism"></span>

### 张量并行

张量并行需要对激活值执行一次 AllGather 和一次 ReduceScatter，并且需要让这些操作与 MLP 的 FLOPs 重叠。换句话说，在前向传播中，有

$$
T_\text{math} = \frac{2\cdot 2 \cdot BDF}{Y \cdot C}
$$

$$
T_\text{comms} = \frac{2\cdot 2 \cdot BD}{W_\text{collective}}
$$

要做到计算受限，由此可以得到规则

$$
Y < \frac{F \cdot W_\text{collective}}{C}
$$

在节点内，这给出大约 $F / 2200$；超过一个节点后则为 $F / 2475$。对于像 LLaMA-3 那样的 $F=\text{28000}$，这大约是 11 路 TP（或者向下取整，约为 8 路，也就是一个节点所含 GPU 的数量）。与上面一样，当恰好跨越 2 个节点时，我们会额外获得 2 倍带宽，所以一般可以进行 16 路张量并行（$F > 2475 \cdot (Y - 8)$）；理论上，这允许我们达到最高 19 路模型并行。

**要点：** 对于大小为 Y、前馈维度为 F 的轴，张量并行会在 $Y > F / 2475$ 时变为通信受限，这通常将我们限制在节点内 TP，或者最多 2 节点 TP。

<span id="expert-parallelism"></span>

### 专家并行

正如上面已经指出的，混合专家（MoE）模型的模型权重多了 E 倍，而 FLOPs 只多了 k 倍，这让数据并行变得困难得多。我们可以沿专家维度对权重进行分片，即 W<sub>in</sub>[E<sub>Z</sub>, D, F]，从而在一定程度上缓解这个问题。为了执行 MLP 块，我们需要引入 2 次 AllToAll，把激活值发送给相应的专家。

如上所述，如果 AllToAll<sub>Z->k</sub>([B, D, k]) 跨越多个节点，其开销大约为 $T_\text{AllToAll} = 2 \cdot B \cdot D \cdot (Z-8)/Z \min(8 * k / Z, 1)$，因此对于纯专家并行，我们需要

$$
T_\text{math} = \frac{4 \cdot B \cdot k \cdot D \cdot F}{Z \cdot C}
$$

$$
T_\text{comms} = \frac{4 \cdot B \cdot D \cdot (Z-8)}{W \cdot Z} \cdot \min\left(\frac{8 \cdot k}{Z}, 1\right)
$$

我们要么需要在 $K > Z/8$ 时满足 $F > \alpha \cdot (Z - 8)/k$，要么需要在 $Z \gg K$ 时满足 $F > 8 \cdot \alpha$，其中 $\alpha = C/W$。这给出了两个可以采用专家并行的区域：一是使用少量专家并行（大约 2 个节点）且 $F$ 较小；二是 $F$ 较大，而 $Z$ 可以任意大（最高可达到 E 路专家并行）。

实践中这两种情况都能见到：要么采用少量专家并行（例如 DeepSeek v3，它的 F 非常小，跨节点专家并行也相对较小且受到限制），要么模型具有较大的 F，此时可以在 TP 之外再进行大量跨节点 EP。

**要点：** 如果 $F < 8 * C / W_\text{node}$，专家并行可以跨越 1–2 个节点，开销与 TP 相近（但略低）；或者，如果 $F > 8 * C / W_\text{node}$，我们可以用相对较低的开销进行大量专家并行（最高可达 $E$ 个节点）。

<span id="pipeline-parallelism"></span>

### 流水线并行

流水线并行把不同层拆分到不同节点上，其通信成本极低，因为我们只需每隔几层发送一小批激活值。过去，流水线一直受到“流水线气泡”的困扰，但借助新的零气泡流水线方法，通常可以消除这一问题。

流水线的总体通信成本很小：设有 $N_\text{MB}$ 个微批次和 $N_\text{stages}$ 个阶段，则有 $T_\text{comms per hop} = 2 \cdot B \cdot D / (W \cdot N_\text{MB})$，并且需要经过 $N_\text{MB} + N_\text{stages} - 2$ 跳，因此大致为

$$
T_\text{total PP comms} = \frac{2BD}{W \cdot N_\text{MB}} \cdot (N_\text{MB} + N_\text{stages} - 2)
$$

$$
T_\text{per-layer comms} \approx 1.5 \cdot \frac{2BD}{W \cdot N_\text{layers}}
$$

由于这里除以了 $N_\text{layers}$，这一成本远小于其他任何成本。换句话说，从通信角度看，流水线基本上是免费的。那为什么不干脆只用流水线呢？原因有几个：

(1) **代码复杂度：** 与其他方法相比，流水线不太容易融入自动并行框架（例如 XLA 的 GSPMD）。由于它会引入微批次来隐藏流水线气泡，因此会改变程序结构；而自定义零气泡流水线调度还要求以复杂方式交错执行前向传播与反向传播，进一步加剧了这个问题。

(2) **流水线让数据并行和 FSDP 变得困难：** 不采用流水线的最大理由，很可能是它与 FSDP 和数据并行配合得不好。尤其是 ZeRO-3 分片表现很差，因为它要求我们对每个微批次都执行权重 AllGather，而此时只有 $B / N_\text{microbatches}$ 个 token 可用于摊销 AllGather 的成本。此外，在反向传播期间，*只有当最后一个微批次通过某个阶段后，我们才能对梯度执行 AllReduce 或 ReduceScatter，这意味着会有大量无法与计算重叠的通信时间。*

![图：一个 2 阶段、2 微批次的流水线示例。F 表示某阶段的前向传播，B 表示某阶段的反向传播（成本为 2 倍）。G 表示数据并行 AllReduce，其耗时可能远长于单个微批次的处理时间。](/images/scaling-book/gpu/pipeline-bubble.png)

(3) **流水线气泡与步骤不均衡：** 从上面这个（糟糕的）流水线调度可以看出，朴素流水线调度很容易产生大量气泡（也就是浪费的计算）。在上图中，第二个阶段在第 0 步处于空闲状态，第一个阶段从第 2 步到第 3 步处于空闲状态，而第二个阶段在最后一步又再次处于空闲状态。虽然通过谨慎调度可以在一定程度上避免这些气泡，但通常还是会留下一些。我们还必须在关键路径上把激活值从一个阶段传递到下一个阶段，这会增加额外开销：

![图：一个流水线示例，其中传输成本以红色表示。这会使各阶段彼此错开，并增加流水线气泡的开销。](/images/scaling-book/gpu/pipeline-transfer.png)

这些问题各自都有变通办法，但通常实现复杂且难以维护；相较于其他方法，流水线仍然是一种通信成本较低的技术。

**关于延迟的提醒：** 如前所述，即使消息相当大，GPU 也很难达到完整的 AllReduce 带宽。这意味着，即便从理论上看，我们可以把专家并行 AllToAll 等操作扩展到多个节点，实际也可能连总带宽的 50% 都难以达到。因此，我们确实会尝试把 TP 或 EP 控制在较少的节点内，以尽量降低延迟开销。

<span id="examples"></span>

### 示例

**DeepSeek 是怎么做的？** 作为参考，[DeepSeek V3](https://arxiv.org/abs/2412.19437) 使用 2048 个 H800 GPU 训练，并采用：

* 跨越 8 个节点的 64 路专家并行（EP）
* 16 路流水线并行（PP）
* 2 路 ZeRO-1 数据并行（DP）

其稳态批大小为 `4096 * 15360 = 62,914,560` 个 token，也就是每 GPU 约 30k 个 token。可以看出，这已经相当大了；但其模型也非常稀疏（k=8，E=256），所以确实需要相当大的批大小。采用 64 路 EP 和 16 路 PP 后，总共会得到 1024 路模型并行，这意味着 AllReduce 在脊层级执行；又因为它只有 2 路，所以在实践中会得到 $2 / (2 - 1) = 2$ 倍的带宽。这也有助于降低最终数据并行 AllReduce 与最后几个流水线阶段重叠时的成本。

**LLaMA-3 是怎么做的？** LLaMA-3 在 16k 个 GPU 上以 16M token 的 BS 进行训练，即每个 GPU 约 1k 个 token。其配置为：

* 节点内 8 路张量并行（TP）
* 16 路流水线并行（PP）
* 128 路 ZeRO-1 数据并行

这也是一个稠密模型，所以总体而言这些问题都相当简单。16 路 PP 把数据并行 AllReduce 的成本降低了 16 倍，有助于降低临界批大小。

<span id="tldr-of-llm-scaling-on-gpus"></span>

### GPU 上 LLM 扩展总结

让我们退一步，对目前所学内容做一个总体总结：

* **数据并行或 FSDP（ZeRO-1/3）要求每 GPU 的局部批大小约为 2500 个 token**，不过理论上，网内归约加纯 DP 可以在一定程度上降低这一数值。
* **张量并行在最高约 8 路时是计算受限的**，但如果继续扩展，我们会在变成通信受限之前缺少足够的带宽。这基本上把我们限制在单个 NVLink 域内（也就是单节点，或者需要使用最多包含 72 个 GPU 的 GB200NVL72）。
* **任何跨越多个节点的模型并行形式都可以进一步降低 FSDP 的成本**，因此我们常常希望混合 PP、EP 和 TP，跨越许多节点并降低 FSDP 成本。
* **如果你能处理零气泡流水线的代码复杂度，并保持相当大的批大小以避免数据并行瓶颈，那么流水线并行会很好用。** 流水线通常会让 ZeRO-3 变得不可行（因为你需要在每个流水线阶段执行 AllGather），但可以改用 ZeRO-1。

**从高层次看，这为我们给出了在 GPU 上分片大型模型的一套方法：**

* 对于相对较小的稠密模型，只要批大小足够，激进的 FSDP 就非常有效；如有需要，可以再加一定程度的流水线并行或张量并行。
* 对于更大的稠密模型，组合使用 1–2 节点 TP、多节点 PP 和纯 DP 会很有效。
* 对于 MoE，上述规则同样适用，不过我们还可以使用专家并行，而且通常比起 TP，我们更偏好专家并行。如果 $F > 8 * C / W_\text{node}$，就可以进行大量多节点专家并行；否则大致只能采用 2 节点 EP。

<span id="quiz-5-llm-rooflines"></span>

### 小测 5：LLM Roofline

**问题 1［B200 Roofline］：** B200 DGX SuperPod（**不是 GB200 NVL72**）的节点内带宽是原来的 2 倍（900GB/s 出方向），但横向扩展网络的带宽不变（400GB/s）（[来源](https://docs.nvidia.com/dgx-superpod/reference-architecture-scalable-infrastructure-b200/latest/network-fabrics.html)）。其总 FLOPs 已在上文给出。这会如何改变模型并行和数据并行的 Roofline？

<details>
<summary>点击此处查看答案。</summary>


**答案：** bfloat16 下的 FLOPs/s 从 990 TFLOPs 增至 2250 TFLOPs，提升了 2.25 倍。节点内带宽提升 2 倍，因此节点内 Roofline 大致保持不变。例如，对于 TP，临界强度增至 `2250e12 / 900e9 = 2500`，因此上限为 $Y < F / 2500$，只略高一点（除非节点规模增加，否则这对我们没有帮助）。

但在节点之外，带宽并未增加，反而让我们更难达到计算受限！例如，对于数据并行，临界批大小增至 `2250e12 / 400e9 = 5625`，因为 GPU 可以在相同带宽下完成显著更多的 FLOPs。

包含 72-GPU 节点的 GB200 SuperPod 通过增加更多出方向带宽改变了这一状况（[来源](https://docs.nvidia.com/dgx-superpod/reference-architecture-scalable-infrastructure-gb200/latest/network-fabrics.html#compute-fabric-576)）。

</details>

**问题 2 [如何对 LLaMA-3 70B 进行分片]：** 考虑 LLaMA-3 70B，使用 bfloat16 训练，并为 Adam 使用 fp32 优化器状态。

1. 仅仅为了存储权重和优化器，我们至少需要多少个 H100？
2. 假设我们希望在 4096 个 H100 GPU 上训练 15T 个 token。假设达到了 45% 的 MFU（模型 FLOPs 利用率）。训练需要多长时间？
3. LLaMA-3 70B 的 `F = 28,672`，训练时使用的批大小约为 4M token。在不受通信限制的前提下，我们最多可以进行多少路模型并行？加上纯 DP 后，能否在 4k 个芯片上训练 LLaMA-3，同时保持计算受限？ZeRO-3 呢？再加上 8 路流水线又如何？*注意：需要同时考虑通信成本和 GPU 内存用量。*

<details>
<summary>点击此处查看答案。</summary>


1. 权重需要 2 字节，优化器状态需要 8 字节，因此至少需要 700GB。每个 GPU 有 80GB DRAM，所以最低至少需要 9 个 GPU，或者（向上取整）至少 2 个 8xH100 节点。这种配置要花极长时间才能完成训练，而且还放不下梯度检查点，但它给出了一个下界。
2. 总共需要 `6 * 70e9 * 15e12 = 6.3e24 bf16 FLOPs`。每个 GPU 可以完成 `990e12` FLOPs，因此在 45% MFU 下，我们可以达到 1.8e18 FLOPs/s。因此整个训练需要 3.5e6 秒，也就是 40 天。
3. 节点内带宽为 450GB/s，因此上限大致为 `F / 1995 = 28672 / 1995 = 14.372`。由于这不足以跨越 2 个节点，现实中我们最多会采用 8 路模型并行。
   1. 这样一来，我们就需要进行 512 路 DP。首先需要确认内存是否足够。由于模型只进行 8 路分片，这意味着 `700GB / 8 = 87.5GB / GPU`，放不下，所以不行！
   2. 使用 ZeRO-3 和 8 路 TP 时，我们将进行 512 路 ZeRO-3。由于所有内容都进行了激进分片，内存不会有问题。每 GPU 的批大小为 `4e6 / 4096 = 976`。这个数值非常低，甚至低于纯 DP 的上限；而这里的上限还是纯 DP 上限的两倍，因为我们必须移动权重。所以也不行。
   3. 使用 8 路流水线时，每个模型并行分片现在会跨越 8 个节点。正如前面所见，这会把叶层 AllGather 的成本降低到原来的 1/8，因此那里的总体 AllReduce/AllGather 带宽会从 400GB/s 增至 `8 * 400GB/s = 3200GB/s`。此时 Roofline 为 `990e12 / 3200e9 = 309`，所以应该没问题！我们只需要高效实现流水线即可。

</details>

**问题 3 [Megatron-LM 超参数]：** 考虑 [Megatron-LM 仓库](https://github.com/NVIDIA/Megatron-LM)中的这张图，它突出了其很高的 MFU 数值。

![](/images/scaling-book/gpu/megatron-hparams.png)

请注意，所有配置的序列长度均为 4096。对于 16B、70B 和 314B 模型，每 GPU 的 token 批大小是多少？假设数据并行是最外层轴，并假设使用 bfloat16 归约，请判断各配置理论上是计算受限还是通信受限，以及是否存在更优的配置？

<details>
<summary>点击此处查看答案。</summary>


**答案：** 先计算每 GPU 的批大小。

* **16B**：`192 * 4096 / 192 = 4096` 个 token/GPU
* **70B**：`384 * 4096 / 768 = 2048` 个 token/GPU
* **314B**：`1536 * 4096 / 3072 = 2048` 个 token/GPU

这意味着，除第一个配置外，其余配置每批都在约 2k 个 token 附近徘徊；值得注意的是，这正好接近我们为 FSDP 计算出的临界阈值。根据脊层级的归约，我们计算出的界限是每 GPU 2,472 个 token，这里应该大致会触及该界限。不过，对于 70B 和 314B，由于分别进行了 16 路和 64 路模型（PP + TP）分片，我们在脊层级可以获得 2 倍和 8 倍的吞吐量，这意味着它们分别在每步约 1k 和 300 个 token 时就应该达到计算受限。

</details>

<span id="acknowledgements-and-further-reading"></span>

## 致谢与延伸阅读

本章在很大程度上得益于许多知识渊博的 GPU 专家的帮助，包括：

* Adam Paszke，他帮助解释了 GPU 内核编程的现实情况。
* Swapnil Patil，他最先向我解释了 GPU 网络的工作方式。
* Stas Bekman，他指出 GPU 的实测情况往往与宣称的规格不同。
* Reiner Pope，他帮助澄清了 GPU 和 TPU 在硬件层面的比较方式。
* Frédéric Bastien，他对芯片层面的叙述提供了详细反馈。
* Nouamane Tazi，他在 GPU 上训练 LLM 的经验帮助改进了 Roofline 一节。
* Sanford Miller，他帮助我理解 GPU 的联网方式，以及 NVIDIA 的规格与实际部署情况之间的差异。

关于 GPU，有大量优秀的阅读材料；以下是我最喜欢的一些：

* [SemiAnalysis' History of the NVIDIA Tensor Core](https://semianalysis.com/2025/06/23/nvidia-tensor-core-evolution-from-volta-to-blackwell/)：一篇精彩的文章，讲述 GPU 如何从电子游戏引擎转变为 ML 加速器。
* [SemiAnalysis' Analysis of Blackwell Performance](https://semianalysis.com/2024/04/10/nvidia-blackwell-perf-tco-analysis/)：值得阅读，有助于理解 NVIDIA GPU 的下一代产品。
* [H100 DGX SuperPod Reference](https://docs.nvidia.com/dgx-superpod-reference-architecture-dgx-h100.pdf)：内容枯燥，但对于理解大型 GPU 集群如何联网很有帮助。[这里](https://docs.nvidia.com/dgx-superpod/reference-architecture-scalable-infrastructure-gb200/latest/network-fabrics.html#compute-fabric-576)还有一份关于 GB200 系统的类似文档。
* [Hot Chips Talk about the NVLink Switch](https://hc34.hotchips.org/assets/program/conference/day2/Network%20and%20Switches/NVSwitch%20HotChips%202022%20r5.pdf)：一份介绍 NVLink 和 NCCL 集合通信的有趣材料，尤其涵盖了网内归约。
* [DeepSeek-V3 Technical Report](https://arxiv.org/pdf/2412.19437)：一份大型半开放 LLM 训练报告的优秀示例，描述了其分片配置的选择方式。
* [How to Optimize a CUDA Matmul](https://siboehm.com/articles/22/CUDA-MMM)：一篇很棒的博客，介绍如何使用 CUDA Cores 实现高效矩阵乘法，并着眼于 GPU 上的缓存一致性。
* [HuggingFace Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook)：一份 GPU 上 LLM 并行的指南，本章的部分灵感来自它。
* [Making Deep Learning Go Brrrr From First Principles](https://horace.io/brrr_intro.html)：一篇更聚焦于 GPU 和 PyTorch 的 LLM Roofline 与性能工程教程。
* [Cornell Understanding GPU Architecture site](https://cvw.cac.cornell.edu/gpu-architecture)：一份与本书类似的指南，更具体地比较了 GPU 和 CPU 的内部结构。

<span id="appendix-a-how-does-this-change-with-gb200"></span>

## 附录 A：使用 GB200 时有何变化？

Blackwell 引入了许多重大的网络变化，包括总体 NVLink 带宽翻倍（900GB/s）的 NVLink 5。B200 和 H100 一样，仍采用 8-GPU 节点；但 GB200 系统（将 B200 GPU 与 Grace CPU 结合）引入了大得多的 NVLink 域（NVL72 中为 72 个 GPU，理论上最多可达 576 个）。更大的 NVLink 域实际上也提高了节点出方向带宽，从而降低节点层级以上的集合通信成本。

![图：GB200 NVL72 单元的构造示意图，其中包含 18 个交换机和 72 个 GPU。](/images/scaling-book/gpu/b200-node.png)

节点内带宽从 450GB/s 增至 900GB/s，但这并没有带来太大差异，因为每个 GPU 的总 FLOPs/s 也翻了一倍。我们的 Roofline 基本保持不变，不过 NVLink 的带宽高得多，使得专家并行变得更容易。

在节点之外，变化更大。下面是一张来自[这里](https://docs.nvidia.com/dgx-superpod/reference-architecture-scalable-infrastructure-gb200/latest/network-fabrics.html#compute-fabric-576)的 SuperPod 示意图。

![图：由 576 个 GPU 组成的 GB200 DGX SuperPod 示意图。](/images/scaling-book/gpu/gb200-superpod.png)

如图所示，每节点出方向带宽增至 `4 * 18 * 400 / 8 = 3.6TB/s`，高于 H100 的 400GB/s。由于每芯片 FLOPs 也翻了一倍，这会让有效的跨节点 Roofline 改善约 4 倍。现在，我们可能要开始担心瓶颈会不会出现在节点层级，而不是横向扩展层级。

**Grace Hopper：** NVIDIA 还销售把若干 GPU 与 Grace CPU 配对的 GH200 和 GB200 系统。例如，一个 GH200 包含 1 个 H200 和 1 个 Grace CPU，而一个 GB200 系统包含 2 个 B200 和 1 个 Grace CPU。这种系统的一个优势是，CPU 通过全带宽 NVLink 连接（称为 NVLink C2C）与 GPU 相连，因此 CPU 到 GPU 的带宽非常高，适合把参数卸载到主机 RAM。换句话说，对于任意一个给定 GPU，访问主机内存的带宽与访问另一个 GPU 的 HBM 相同。

<span id="appendix-b-more-networking-details"></span>

## 附录 B：更多网络互连细节

下面是一张 NVLink 4 交换机的示意图。它总共有 64 个 NVLink4 端口（每个端口使用 2 条物理通道），还有一个负责通道间交换的大型交叉开关。相比之下，TPU 使用带反射镜的光交换机，可以动态重新配置。

![图：单个 NVLink4 交换机的底层视图。](/images/scaling-book/gpu/nvlink4.png)

在每个层级上，可用链路带宽或交换机总带宽都可能成为瓶颈。

* **节点层级：** 在节点层级，我们有 4 * 1.6TB/s = 6.4TB/s 的 NVSwitch 带宽，但 8 个 GPU 中的每一个都只能以 450GB/s 的速度向交换机传出数据，这意味着节点内的峰值带宽实际上是 450e9 * 8 = 3.6TB/s（全双工）。
* **SU/叶层级：** 在 SU 层级，有 8 个交换机以全互连方式连接 32 个节点，使用 1x400 Gbps InfiniBand。由此得到节点的出方向带宽为 8 * 32 * 400 / 8 = 12.8TB/s，而交换机层级有 8 * 1.6TB/s = 12.8TB/s，因此两者恰好一致。
* **脊层级：** 在脊层级，有 16 个交换机通过 2x400 Gbps 链路连接 32 个叶交换机，因此出方向带宽为 32 * 16 * 400 * 2 / 8 = 51.2TB/s。与叶交换机不同，脊交换机的全部 64 个端口都朝向下层，因此每个交换机可以传输 64 * 400 / 8 = 3.2TB/s 的流量，在交换机层级总计得到 16 * 3.2TB/s = 51.2TB/s，再次恰好一致。

平均到每个 GPU，这意味着节点层级的 GPU 到 GPU 带宽为 450GB/s，而 SU 和脊层级均为 50GB/s。

**GPU 实测 AR 带宽：**

![图：8xH100 集群上的 AllReduce 带宽（节点内，SHARP 已禁用）。](/images/scaling-book/gpu/gpu-all-reduce-bw.png)

TPU v5p 带宽（1 个轴）：

![图：TPU v5p 4x4x4 集群上的 AllReduce 带宽（沿一个轴）。](/images/scaling-book/gpu/tpu-all-reduce-bw.png)

下面还有 AllGather 带宽：

![图：8xH100 集群上的 AllGather 带宽（节点内）。](/images/scaling-book/gpu/gpu-all-gather-bw.png)

![图：TPU v5e 8x16 集群上的 AllGather 带宽（沿一个轴）。](/images/scaling-book/gpu/tpu-all-gather-bw.png)

**关于 AllToAll 成本的更多信息：**

这里可以比较近似值 $\min(K / Z) * (Z - 1) / Z$ 与真实值 $(1 - ((Z - 1) / Z) ** K) * (Z - 1) / Z$。除 $Z$ 很小时外，两者很相似。

![图：随着分片数量增加，变长（ragged）AllToAll 的近似开销与真实开销之间的比较。](/images/scaling-book/gpu/all-to-all-approx.png)

[^ch12-1]: GPU Tensor Core 是 SM 中的矩阵乘法子单元，而 TPU TensorCore 则是包含 MXU、VPU 及其他组件的总括单元。
[^ch12-2]: NVIDIA 没有为此提供一个合适的名称，所以我们只是在几个不理想的选项中挑了一个相对最合适的。Warp Scheduler 主要是向一组 CUDA 核心分派工作的单元，但我们在这里用它来描述控制单元及其控制的那组核心。
[^ch12-3]: 虽然各个 SM 相互独立，但为了达到峰值性能，它们往往被迫进行协调，因为它们共享一个容量有限的 L2 缓存。
[^ch12-4]: 较新的 GPU 支持 FMA（Fused-Multiply Add）指令，严格来说每个周期会执行两次 FLOPs；NVIDIA 毫不客气地利用这一点，把其公布的规格翻了一倍。
[^ch12-5]: 从历史上看，在 Tensor Core 引入之前，CUDA 核心是 GPU 的主要组件，用于包括光线—三角形求交和着色在内的渲染工作。在今天的游戏 GPU 上，它们仍承担大部分渲染工作，而 Tensor Core 则用于上采样（DLSS）；这使 GPU 可以先以较低分辨率渲染（像素更少 = 工作更少），再使用 ML 进行上采样。
[^ch12-6]: NVIDIA 并未公开很多 TC 硬件细节，因此这更多是一种猜测，而非确定事实——当然，它并不能说明 TC 的具体实现方式。我们知道，V100 每个 TC 每周期可执行 256 FLOPs，A100 可执行 512，H100 可执行 1024；虽然 B200 的细节尚未公布，但它很可能约为每个 TC 每周期 2048 FLOPs，因为 `2250e12 / (148 * 4 * 1.86e9)` 约等于 2048。[这里](https://forums.developer.nvidia.com/t/how-to-calculate-the-tensor-core-fp16-performance-of-h100/244727)确认了更多细节。
[^ch12-7]: 在 Ampere 中，可以由单个 warp 向 Tensor Core 供给数据；在 Hopper 中，它需要一个完整的 SM（warpgroup）；而在 Blackwell 中，则由 2 个 SM 供给数据。Blackwell 中的矩阵乘法规模也变得如此之大，以至于参数（特别是累加器）不再能放入寄存器内存/SMEM，因此 Blackwell 新增了 TMEM 来解决这个问题。
[^ch12-8]: 调度到某个给定 SM 上的 warp 称为“驻留（resident）”warp。
[^ch12-9]: 严格来说，L2 缓存被分成两半，因此在 H100 上，一半 SM 可以访问其中的 25MB。两半之间有一条链路相连，但带宽较低。
[^ch12-10]: 尽管理论上各个 SM 是彼此独立的单元，但 L2 缓存由所有 SM 共享，这一事实实际上仍迫使程序员以相当协调的方式运行各个 SM。
[^ch12-11]: NVIDIA 确实推出过 B100 一代产品，但其销售和生产时间都很短，据称是因为设计缺陷使其无法达到接近宣称规格的运行水平。由于散热和功耗方面的问题，它们很难在不降频的情况下达到峰值 FLOPs。
[^ch12-12]: 在深度学习蓬勃发展之前，GPU（“Graphics Processing Units”）顾名思义就是用来处理图形的——主要用于电子游戏。电子游戏使用数百万个小三角形表示物体，游戏会把这些三角形渲染（或“光栅化”）成二维图像，每秒在屏幕上显示 30–60 次（这个频率称为帧率）。光栅化需要把这些三角形投影到相机坐标系中，并计算哪些三角形与哪些像素重叠，每秒执行数十亿次。可以想象，这项工作的成本非常高，而这还只是开始。随后，你还需要组合与光线相交的若干个可能半透明三角形的颜色，为每个像素着色。GPU 在设计上要极快地执行这些操作，同时兼顾通用性；你需要同时运行许多不同的 GPU 工作负载（称为“着色器（shaders）”），且不能让任何单一操作占据主导地位。因此，面向消费级图形的 GPU 可以执行矩阵乘法，但这不是它的主要功能。
[^ch12-13]: 值得注意的是，这个强度在最近几代 GPU 上保持不变。H100 为 33.5 / 3.5，B200 为 80 / 8。原因尚不清楚，但这是一个有趣的观察结果。
[^ch12-14]: “节点”一词存在含义重载，可以指两种事物：一种是 NVLink 域，即通过 NVLink 互连完全连接的一组 GPU；另一种是连接到单个 CPU 主机的一组 GPU。在 B200 之前，两者通常相同；但在 GB200 NVL72 中，一个 NVLink 域包含 72 个 GPU，而连接到每个主机的仍然只有 8 个 GPU。我们在这里使用“节点”指代 NVLink 域，但这种用法存在争议。
[^ch12-15]: 有人向我描述，NVLink 有点像强化版 PCIe 连接，具有低延迟和较低的协议开销，但并非为可扩展性/容错性而设计；而 InfiniBand 则更像 Ethernet，是为规模更大的有损网络设计的。
[^ch12-16]: 这里的全双工是指两个方向各 25GB/s，且彼此独立。你可以通过链路总共发送 50GB/s，但每个方向最多只能达到 25GB/s。
[^ch12-17]: 例如，Meta 用来训练 LLaMA-3 的数据中心网络就与这里的描述有很大不同：它使用 Ethernet、一个 3 层交换架构，以及位于顶层的超额订阅交换机。
[^ch12-18]: 你也可以把它理解为：每个 GPU 都把大小为 $\text{bytes} / N$ 的数据块发送给其他 $N - 1$ 个 GPU，因此总通信量为 $(N - 1) * N * bytes / N$ 字节，得到相同的答案。
[^ch12-19]: 实际成本其实是 $(1 - \left(\frac{Z - 1}{Z}\right)^K) \cdot \frac{Z - 1}{Z}$，也就是掷 $K$ 次骰子时不同结果数量的期望值，但它与给出的近似值非常接近。更多细节参见附录。
