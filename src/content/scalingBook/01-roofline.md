---
title: "Roofline 模型详解"
description: "在硬件上运行算法时，我们受到三项因素的限制：计算机执行数学运算的速度（OPs/秒）、用于搬运数据的可用带宽（字节/秒），以及用于存储数据的总内存容量（字节）。这些“Roofline”约束让我们能够为给定计算所需的时间确定上界和下界。"
chapter: 1
order: 1
part: 1
partTitle: "预备知识"
sourcePath: "roofline.md"
sourceCommit: "44109cacac9c5a9809a81c68ae4d45d7d2632ea6"
---

<span id="all-about-rooflines"></span>

# Roofline 模型详解

<span id="where-does-the-time-go"></span>

## 时间都花到哪里了？

我们从一个极其简单的问题开始：*为什么一个算法耗时 50ms，而不是 50s 或 5ms*？模型内部究竟发生了哪些相当耗时的事情，我们又应当预期它们花费多久？

**计算：** 深度学习模型实际上就是一系列矩阵乘法，每次矩阵乘法又由浮点乘法和加法“运算”（FLOPs）构成。加速器的速度决定了完成这些计算需要多长时间：

$$
\begin{equation}
T_\text{math} = \frac{\text{Computation FLOPs}}{\text{Accelerator FLOPs/s}}
\end{equation}
$$

例如，NVIDIA H100 每秒大约可执行 9.89e14 次 bfloat16[^ch1-1] 浮点运算，而 TPU v6e 每秒可执行 9.1e14 次浮点运算。[^ch1-2] 这意味着，在 H100 上执行 1e12 FLOPs 大约需要 `1e12 / 9.89e14 = 1.01ms`，在 TPU v6e 上则需要 `1e12 / 9.1e14 = 1.1ms`。[^ch1-3]

**芯片内部通信：** 在一个加速器<em>内部</em>，张量需要在加速器内存（HBM）与计算核心之间传输。你会看到这条链路的带宽被称为“HBM 带宽”。[^ch1-4] H100 的[这一带宽约为 3.35TB/s](https://www.nvidia.com/en-us/data-center/h100/)，TPU v6e 则[约为 1.6TB/s](https://cloud.google.com/tpu/docs/v6e)。

**芯片之间的通信：** 当我们把模型分布到多个加速器<em>上</em>时，张量经常需要在加速器之间传输。硬件通常会提供若干种选择（ICI、DCN 和 PCIe），每种链路的带宽都不同。

无论通信发生在芯片内部还是芯片之间，我们都用 bytes/s 衡量它，并用下面的公式估算总通信时间：

$$
\begin{equation}
T_\text{comms} = \frac{\text{Communication Bytes}}{\text{Network/Memory Bandwidth Bytes/s}}
\end{equation}
$$

通常情况下（但并非总是如此），单芯片内的计算可以与芯片内部和芯片之间的通信重叠。这意味着，**取计算时间与通信时间二者中的较大值，就能得到训练和推理耗时的下界**；而**将两者相加，则能得到上界**。实践中，我们针对二者的最大值进行优化，因为这样代数更简单，而且通过重叠通信与计算，通常可以接近这一界限。如果以最大值为优化目标，那么上下界最多相差 2 倍，因为 $T_\text{math} + T_\text{comms} \leq 2 * \max(T_\text{math}, T_\text{comms})$。在此基础上，我们还可以对“重叠区域”和各种开销建模，从而提高估算精度；这些模型可以由目标系统上具体模型的性能剖析结果来校准。

$$
\begin{equation}
T_\text{lower}=\max(T_\text{math}, T_\text{comms})
\end{equation}
$$

$$
\begin{equation}
T_\text{upper} = T_\text{math} + T_\text{comms}
\end{equation}
$$

如果假设通信与计算可以完美重叠，那么当 $T_\text{math} > T_\text{comms}$ 时，硬件会得到充分利用。我们称这种状态为“计算受限”。当 $T_\text{comms} > T_\text{math}$ 时，系统往往“通信受限”[^ch1-5]，加速器至少有一部分 FLOPs/s 会浪费在等待数据传输上。判断一项运算究竟受计算还是通信限制的一种方法，是考察它的“**算术强度（arithmetic intensity）**”或“**运算强度（operational intensity）**”。

**定义：** 算法的算术强度，等于它执行的 FLOPs 总数与它需要通信的字节数之比；这里的通信既可以发生在芯片内部，也可以发生在芯片之间。

$$
\begin{equation}
\text{Arithmetic Intensity} = \frac{\text{Computation FLOPs}}{\text{Communication Bytes}}
\end{equation}
$$

算术强度衡量一次运算的“每字节 FLOPs”。从一阶近似来看，算术强度较高时，$T_\text{math}$ 相对于 $T_\text{comms}$ 较大，通常能够用到大部分可用 FLOPs。反之，我们会把更多时间花在通信上，从而浪费 FLOPs。二者发生转换的位置称为硬件的“临界算术强度”，也就是加速器峰值 FLOPs/s 与加速器带宽之比。

$$
\begin{align*}
T_\text{math} > T_\text{comms} \Leftrightarrow \frac{\text{Computation FLOPs}} {\text{Accelerator FLOPs/s}} > \frac{\text{Communication Bytes}}{\text{Bandwidth Bytes/s}} & \\[0.5em]
\Leftrightarrow \frac{\text{Computation FLOPs}}{\text{Communication Bytes}} > \frac{\text{Accelerator FLOPs/s}}{\text{Bandwidth Bytes/s}} & \\[0.5em]
\Leftrightarrow \text{Intensity}(\text{Computation}) > \text{Intensity}(\text{Accelerator}) & \\
\end{align*}
$$

$\text{Intensity}(\text{Accelerator})$ 表示加速器达到峰值 FLOPs/s 时所需的算术强度。**对于 TPU v5e 的 MXU，这一数值约为 240 FLOPs/byte**，因为该 TPU 每秒可执行 `1.97e14` 次浮点运算，并以每秒 `8.2e11` 字节的速度从 HBM 加载数据。[^ch1-6] 这意味着，如果一个算法的算术强度低于 240 FLOPs/byte，它就会受字节加载速度限制，因而无法充分利用硬件。[^ch1-7] 下面来看这样一个例子：

**示例（点积）：** 为了以 bfloat16 精度计算两个向量的点积 `x • y: bf16[N], bf16[N] → bf16[1]`，我们需要从内存加载 $x$ 和 $y$，两者各占 $2 * N = 2N$ 字节；随后执行 $N$ 次乘法和 $N-1$ 次加法，并向 HBM 写回 $2$ 字节。
$$
\begin{equation}
\text{Intensity}(\text{dot product}) = \frac{\text{Total FLOPs}}{\text{Total Bytes}} = \frac{N + N - 1}{2N + 2N + 2} = \frac{2N - 1}{4N + 2} \rightarrow \frac{1}{2}
\end{equation}
$$

当 $N\rightarrow\infty$ 时，上式趋近于二分之一。因此，点积的算术强度是 $\frac{1}{2}$；换句话说，每加载一个字节，点积只执行 0.5 次浮点运算。这意味着算法的算术强度低于硬件的算术强度，因此系统会受内存带宽限制。[^ch1-8]

<span id="visualizing-rooflines"></span>

### Roofline 可视化

我们可以用 **Roofline 图**直观表示内存与计算之间的权衡。图中，纵轴是算法在硬件上理论可达到的峰值 FLOPs/s（吞吐量），横轴则是算法的算术强度。下面是一个双对数坐标示例：

![图：Roofline 图示例，展示两种算术强度不同的算法（Algo 1 和 Algo 2），以及它们在不同带宽（BW1 和 BW2）下对应的理论峰值吞吐量。在红色区域，两种带宽下算法都受带宽限制，会浪费一部分硬件峰值 FLOPs/s。黄色区域只在较低带宽 BW1 下受带宽限制。绿色区域在所有带宽下都受计算限制；此处使用的是加速器峰值 FLOPs/s，继续增加带宽或提高算术强度都不会带来收益。](/images/scaling-book/img/roofline-improved.png)

在上图中，随着强度增加（从左向右移动），算法性能（FLOPs/s）起初呈线性增长，直到达到硬件的临界算术强度；对于 TPU v5e，这一数值是 240。强度低于该值的算法会受带宽（BW）限制，其性能由峰值内存带宽决定（红色区域）。右侧的算法则会充分利用可用 FLOPs（绿色区域）。这里，Algo 1 受内存带宽限制，只使用了硬件 FLOPs/s 总量的一部分；Algo 2 受计算限制。通常，我们可以通过提高算法的算术强度，或者增加可用内存带宽（从 BW1 移动到 BW2）来提升算法性能。

<span id="matrix-multiplication"></span>

### 矩阵乘法

下面来看一个即将成为我们最喜欢的算法：矩阵乘法（简称 matmul）。我们写作 $X * Y \rightarrow Z$，其中 $X$ 的形状是 $\text{bf16}[B, D]$，$Y$ 的形状是 $\text{bf16}[D, F]$，$Z$ 的形状是 $\text{bf16}[B, F]$。为了执行这次矩阵乘法，需要加载 $2DF + 2BD$ 字节，执行 $2BDF$ FLOPs，并写回 $2BF$ 字节。[^ch1-9] [^ch1-10] 因此：

$$
\begin{equation}
\text{Intensity}(\text{matmul}) = \frac{2BDF}{2BD + 2DF + 2BF} = \frac{BDF}{BD + DF + BF}
\end{equation}
$$

如果假设“批大小” $B$ 相对于 $D$ 和 $F$ 很小，就能得到一个很漂亮的简化：

$$
\begin{equation}
\frac{BDF}{BD + DF + BF} \approx \frac{BDF}{DF} = B
\end{equation}
$$

$$
\begin{equation}
\text{Intensity}(\text{matmul}) > \text{Intensity}(\text{TPU}) \implies B > \frac{1.97e14}{8.20e11} = 240
\end{equation}
$$

对于 Transformer 矩阵乘法，这是一个合理假设，因为局部（每副本）批大小通常满足 $B < 1024$ 个 token（_不是序列_），而 $D$ 和 $F > 8000$。因此，当每个模型副本的批大小[^ch1-11]超过 240 个 token 时，系统通常会变为计算受限——这是一条非常简单的规则！

**要点：** 要让 bfloat16 矩阵乘法在大多数 TPU 上进入计算受限状态，每个模型副本的 token 批大小必须大于 240。[^ch1-12]

这条规则有几个值得注意的限定条件，下面的练习会进一步讨论，尤其是量化情形（例如，激活值已量化，但仍执行全精度 FLOPs）。不过，它仍是一条值得记住的经验法则。对于 GPU，这个数值略高一些（更接近 300），但总体结论相同。当我们[把一个大矩阵乘法分解成更小的矩阵乘法](https://docs.jax.dev/en/latest/pallas/tpu/matmul.html#your-first-matrix-multiplication-kernel)时，分块大小也很重要。[^ch1-13] 下一章将讨论更底层的 GPU 和 TPU 细节，先从[如何理解 TPU](../02-tpus/#how-to-think-about-tpus)开始。

<span id="network-communication-rooflines"></span>

### 网络通信 Roofline

到目前为止，我们讨论的 Roofline 都针对内存带宽，而且<em>全部发生在单个芯片内部</em>。但这并不意味着它是一条普遍规律。事实上，本书最关心的大多数 Roofline 都涉及芯片之间的通信：通常是对分片到多个 TPU 上的矩阵执行矩阵乘法。

举一个略显刻意的例子：假设我们要把两个大矩阵 $X\sim \text{bf16}[B, D]$ 和 $Y \sim \text{bf16}[D, F]$ 相乘，并沿 $D$ 维将它们均匀拆分到 2 个 TPU/GPU 上。为了完成乘法（将在[第 3 章](../03-sharding/#sharded-matrices-and-how-to-multiply-them)看到），可以在每个 TPU 上计算各自的一半矩阵：TPU 0 执行 `Z0 = X[:, :D // 2] @ Y[:D // 2, :]`，TPU 1 执行 `Z1 = X[:, D // 2:] @ Y[D // 2:, :]`；随后把得到的“部分和”复制到另一个 TPU，并把它们相加。假设每个方向的复制带宽是 `4.5e10` bytes/s，而每块芯片可执行 `1.97e14` FLOPs/s，那么 $T_\text{math}$ 和 $T_\text{comms}$ 分别是多少？

$T_\text{math}$ 显然是原先的一半，因为每个 TPU 只完成一半工作，即：[^ch1-14]

$$
T_\text{math} = \frac{2BDF}{2 \cdot \text{Accelerator FLOPs/s}} = \frac{BDF}{1.97e14}
$$

那么 $T_\text{comms}$ 呢？现在，它表示芯片之间的通信时间！它就是发送的总字节数除以网络带宽，即：

$$
T_\text{comms} = \frac{2BF}{\text{Network Bandwidth}} = \frac{2BF}{4.5e10}
$$

因此，当 $\text{Intensity}(\text{matmul (2-chips)}) > \text{Intensity}(\text{TPU w.r.t. inter-chip network})$ 时，系统会进入计算受限状态（这次是相对于芯片间网络而言）。等价地，需要满足 $\frac{BDF}{2BF} = \frac{D}{2} > \frac{1.97e14}{4.5e10} = 4377$，也就是 $D > 8755$。请注意，与之前不同，现在的临界阈值取决于 $D$，而不是 $B$！请思考其中的原因。这只是一个例子，但它说明：要判断一项运算能否并行扩展到多个 TPU，这类 Roofline 至关重要。

<span id="a-few-problems-to-work"></span>

## 练习题

**问题 1［int8 矩阵乘法］：** 假设我们想以 int8 精度（每个参数 1 字节）而不是 bfloat16 精度（每个参数 2 字节）执行矩阵乘法 $X[B, D] \cdot_D Y[D, F] \rightarrow Z[B, F]$[^ch1-15]，因为 TPU/GPU 能以更低精度更快地执行矩阵乘法。

1. 需要从内存加载多少字节？需要向内存写回多少字节？
2. 总共执行多少次运算？
3. 算术强度是多少？
4. $T_\text{math}$ 和 $T_\text{comms}$ 的 Roofline 估算值分别是多少？整个运算运行时间的合理上下界是什么？

假设 HBM 带宽为 `8.2e11` bytes/s，int8 峰值运算速率为 `3.94e14` OPs/s（约为 bfloat16 的 2 倍）。

<details>
<summary>点击此处查看答案。 </summary>


1. 由于参数以 int8 存储，每个参数占 1 字节，因此从 HBM 加载 $BD + DF$ 字节，并写回 $BF$ 字节。
2. 运算次数与 bfloat16 相同，但理论上 int8 的 OPs/s 应当更高。因此仍然是 $2BDF$ OPs。
3. 算术强度为 $2BDF / (BD + DF + BF)$。如果像前面一样假设 $B \ll D$ 且 $B \ll F$，算术强度就是 $2B$，因此规则变为 $B > \text{HBM int8 arithmetic intensity} / 2$。根据给定数值，int8 的硬件算术强度为 `3.94e14 / 8.2e11 = 480`，所以规则是 $B > 480 / 2 = 240$。请注意，这基本没有变化！
4. $T_\text{math} = 2BDF / 3.94e14$，$T_\text{comms} = (BD + DF + BF) / 8.2e11$，因此合理下界是 $\max(T_\text{math}, T_\text{comms})$，上界则是 $T_\text{math} + T_\text{comms}$。

</details>

**问题 2［int8 + bf16 矩阵乘法］：** 实践中，我们经常对权重和激活值采用不同的量化方案，因此可能以很低的精度存储权重，却让激活值（以及计算）保持较高精度。假设权重被量化为 int8，而激活值（以及计算）保持 bfloat16。批大小达到多少时系统会进入计算受限状态？假设 bfloat16 性能为 `1.97e14` FLOPs/s。

*提示：具体来说，就是执行 `bf16[B, D] * int8[D, F] -> bf16[B, F]`，其中 $B$ 是“批大小”。*

<details>
<summary>点击此处查看答案。 </summary>


仍然假设 B 很小，我们有 2BDF 次 bfloat16 FLOPs，却只有 DF 字节的权重（而 bfloat16 权重需要 2DF 字节）。这意味着，当 $2B > 240$，也就是 $B > 120$ 时，系统会进入计算受限状态。这个阈值低了很多：如果能够对权重进行 int8 量化（这通常很容易），同时仍执行 bfloat16 FLOPs，就能显著提高效率——尽管直接执行 int8 OPs 还会更好。

</details>

**问题 3：** 沿用问题 2 的设置，绘制峰值 FLOPs/s 随 $B$ 变化的 Roofline 图，其中分别取 $F = D = 4096$ 和 $F = D = 1024$。*请使用实际加载的准确字节数，而不是近似值。*

<details>
<summary>点击此处查看答案。 </summary>


对应的图如下：

![](/images/scaling-book/img/roofline-plot-q3.png)

请注意，两个模型最终都能达到硬件峰值 FLOPs/s，但 D/F 较大的模型会更早达到。D=F=1024 几乎让临界批大小翻倍。生成这张图的代码如下：

```py
import matplotlib.pyplot as plt
import numpy as np

bs = np.arange(1, 512)

def roofline(B, D, F):
  total_flops = 2*B*D*F
  flops_time = total_flops / 1.97e14
  comms_time = (2*B*D + D*F + 2*B*F) / 8.2e11
  total_time = np.maximum(flops_time, comms_time)
  return total_flops / total_time

roofline_big = roofline(bs, 4096, 4096)
roofline_small = roofline(bs, 1024, 1024)

plt.figure(figsize=(8, 4))
plt.plot(bs, roofline_big, label='F=D=4096')
plt.plot(bs, roofline_small, label='F=D=1024')
plt.legend()
plt.xlabel('batch size')
plt.ylabel('peak bfloat16 FLOPs/s on TPU v5e')
plt.grid()
```

</details>

**问题 4：** 如果我们要执行 $\text{int8}[B, D] \cdot_D \text{int8}[B, D, F] \rightarrow \text{int8}[B, F]$，也就是设想每个批元素都有一个不同的矩阵，那么这项运算的算术强度是多少？

<details>
<summary>点击此处查看答案。 </summary>


先来看 FLOPs 总量和通信量。

1. FLOPs 总量：FLOPs 基本相同，因为我们要执行 $B$ 个独立的 $[D] \times [D, F]$ 乘积，其总工作量与一次 $[B, D] \times [D, F]$ 矩阵乘法相同（第 4 章会进一步讨论）。所以总量就是 $2BDF$。
2. 通信总量：这里的通信量大得多，为 $BD + BDF + BF$。
3. 因此，算术强度实际上变成 $2BDF / (BD + BDF + BF)$。由于分母由 $BDF$ 主导，它近似等于 $2$。也就是说，强度不再依赖批大小，而基本是一个常数。这很糟糕，因为无论怎样调整批大小，我们几乎都会一直处于内存带宽受限状态。

</details>

**问题 5［GPU 的内存 Roofline］：** 根据 NVIDIA 提供的 [H100 SXM 规格表](https://www.nvidia.com/en-us/data-center/h100/)，计算 bfloat16 矩阵乘法进入计算受限状态时的批大小。*请注意，表中的 Tensor Core FLOPs 数字是实际值的两倍，因为只有使用结构化稀疏时才能达到该数值。*

<details>
<summary>点击此处查看答案。 </summary>


规格表给出的 bfloat16 FLOPs 是 `1.979e15` FLOPs/s，并用星号注明“with sparsity”。不使用稀疏时，真实数值是它的一半，即 `9.89e14` FLOPs/s。内存带宽是 3.35TB/s，也就是 `3.35e12` bytes/s。因此，$B_\text{crit}$ 为 `9.89e14 / 3.35e12 = 295`，与 TPU 十分接近。

</details>

<span id="thats-it-for-part-1-for-part-2-looking-at-how-real-tpus-handle-flops-and-communication-click-here"></span>

### 第一部分的第 1 章到此结束！第 2 章将考察真实 TPU 如何处理 FLOPs 与通信，[点击这里继续](../02-tpus/#how-to-think-about-tpus)。

[^ch1-1]: bf16 是 [bfloat16](https://en.wikipedia.org/wiki/Bfloat16_floating-point_format) 的简称；bfloat16 是机器学习中常用的一种 16 位浮点格式。
[^ch1-2]: H100 和 B200 通常只能达到标称峰值 FLOPs 的约 80%-85%，而 TPU 在正常使用中更接近 95%。
[^ch1-3]: 请注意，这些芯片的定价不同，此处比较没有按成本归一化。
[^ch1-4]: NVIDIA 也把它称为“内存带宽（memory bandwidth）”。
[^ch1-5]: 本书会交替使用“通信受限（communication-bound）”“通信受限（comms-bound）”“内存受限（memory-bound）”和“带宽受限（bandwidth-bound）”这几个说法。
[^ch1-6]: MXU 是 TPU 上的矩阵乘法单元。之所以在这里特别说明，是因为 TPU 还有 VPU 等其他加速单元；VPU 负责逐元素运算，峰值 FLOPs/s 与 MXU 不同。
[^ch1-7]: 只有当算法从 HBM 加载权重并在 MXU 中运行时，这一结论才成立。下一章会讨论，我们有时可以把参数存入带宽高得多的 VMEM。许多算法也会在 VPU 上运行，而 VPU 有着不同的性能特征。
[^ch1-8]: 上面的 240 并不是这里正确的比较对象，因为下一章将看到，点积在 VPU 而不是 MXU 上执行。TPU v5p 的每个 VPU 核心大约可执行 7e12 FLOPs/s，所以它的临界强度约为 3；这意味着这里仍然在一定程度上受内存带宽限制。无论如何，较低且恒定的强度意味着，在大多数硬件上都很难让它进入计算受限状态。
[^ch1-9]: 严格来说，我们执行的是 $BF \times (2D - 1)$ FLOPs，不过该近似已经足够准确。其中有 $BDF$ 次乘法和 $BF * (D-1)$ 次加法。第 4 章会进一步说明。
[^ch1-10]: 虽然矩阵乘法输出从技术上说是 float32，但在复制回 HBM 之前，我们通常会将它向下转换为 bfloat16。
[^ch1-11]: 之所以说“每个模型副本”，是因为如果采用某种模型分片来增加参与矩阵乘法的芯片数量，可用计算能力和内存带宽都会按相同比例扩展。因此，临界批大小是针对每个独立权重副本而言的。
[^ch1-12]: 请注意，这<em>不是</em>通常所说、以序列数衡量的批大小。事实证明，大多数 Roofline 只取决于 token 数量，而不在乎这些 token 属于相同还是不同序列。例如，假设 2048 块 GPU 上有 512 个序列、每个序列 4096 个 token，那么总批大小为 `512 * 4096 = 2M` 个 token，局部批大小则为 1k 个 token。
[^ch1-13]: 执行大型矩阵乘法时，需要把它拆成能装入 VMEM/SMEM/TMEM（带宽更高的片上内存）的小分块。这会导致某些块被多次加载，因此“只需加载 $O(N^2)$ 字节”不再完全成立。考虑一个 $(m, k) \cdot (k, n)$ 矩阵乘法，其分块大小为 $bm$、$bk$、$bn$。令 $tm = m / bm$，其余同理。FLOPs 总量为 $2 \cdot tm \cdot tn \cdot tk \cdot bm \cdot bn \cdot bk$，总字节数为 $2 \cdot tm \cdot tn \cdot (tk \cdot (bm \cdot bk + bk \cdot bn) + bm \cdot bn)$。忽略最后一项，算术强度为 $bm \cdot bn / (bm + bn)$，与上面的结论相似。
[^ch1-14]: 这里忽略了将两个部分和相加所需的 FLOPs（额外的 BF 次加法），但这部分开销基本可以忽略。
[^ch1-15]: 在本处及下文中，记号 $A \cdot_D B$ 表示乘法在 D 维上执行收缩。这是对 einsum 记法的一种非严格借用。
