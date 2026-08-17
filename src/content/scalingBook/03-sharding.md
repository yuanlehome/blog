---
title: "分片矩阵及其乘法"
description: "训练大型机器学习模型时，我们必须把参数或输入拆分（即“分片”）到许多加速器上。由于 LLM 主要由矩阵乘法构成，理解这个问题归根结底就是理解：当矩阵被拆分到不同设备上时，应当如何进行矩阵乘法。本章基于 TPU 通信原语的成本，建立一套简单的分片矩阵乘法理论。"
chapter: 3
order: 3
part: 1
partTitle: "预备知识"
sourcePath: "sharding.md"
sourceCommit: "44109cacac9c5a9809a81c68ae4d45d7d2632ea6"
---

<span id="sharded-matrices-and-how-to-multiply-them"></span>

# 分片矩阵及其乘法

<span id="partitioning-notation-and-collective-operations"></span>

## 分区记法与集合通信操作

当我们在一万块 TPU 或 GPU 上训练 LLM 时，从抽象层面看，所做的计算与在一块设备上训练时仍然相同。区别在于，**我们的数组无法装入单块 TPU/GPU 的 HBM**，因此必须把它们拆开。[^ch3-1] 我们把这种操作称为对数组进行“*分片*”或“*分区*”。实现扩展的艺术，就在于弄清楚如何对模型分片，同时保持计算高效。

下面是一个跨 4 块 TPU 分片的二维数组 **A**：

![图：一个形状为 A[I, J] 的示例数组被分片到 4 台设备上。两个维度都被均匀地分到 2 台设备上，分片记作 A[I_X, J_Y]。每块 TPU 保存总内存的 1/4。](/images/scaling-book/img/sharding-example.png)

请注意，分片数组仍然拥有与未分片数组相同的*全局形状*或*逻辑形状*，例如 `(4, 128)`；但它还具有*设备局部形状*，例如 `(2, 64)`，这个形状给出了每块 TPU 实际保存的数据字节数（在上图中，每块 TPU 保存整个数组的 ¼）。接下来，我们把这一概念推广到任意数组。

<span id="a-unified-notation-for-sharding"></span>

### 统一的分片记法

我们使用一种*命名轴记法*的变体，来描述张量*如何*以数据块形式分片到各台设备上：假设存在一个称为**设备网格**的二维或三维设备栅格，并且每条轴都具有**网格轴名称**，**例如 X、Y 和 Z。** 然后，我们可以通过描述数组的每个命名维度如何沿物理网格轴分区，来指定矩阵数据在设备网格上的布局。我们把这种指派称为一种**分片方案**。

**示例（上图）**：对于上图，我们有：
* **网格：** 上面的设备网格 `Mesh(devices=((0, 1), (2, 3)), axis_names=('X', 'Y'))`，它告诉我们共有 4 块 TPU，排列为 2x2 网格，两条轴分别命名为 $X$ 和 $Y$。
* **分片：** $A[I_X, J_Y]$，它告诉我们将第一条轴 $I$ 沿网格轴 $X$ 分片，并将第二条轴 $J$ 沿网格轴 $Y$ 分片。该分片方案说明，每个分片保存数组的 $1 / (\lvert X\rvert \cdot \lvert Y\rvert)$。

综合这两点，我们知道该数组的局部形状（单台设备所保存分片的大小）为 $(\lvert I\rvert / 2, \lvert J\rvert / 2)$，其中 $\lvert I\rvert$ 是 A 第一维的大小，$\lvert J\rvert$ 是 A 第二维的大小。

**小测验［沿 1 条轴进行二维分片］：** 考虑一个数组 `fp32[1024, 4096]`，其分片为 $A[I_{XY}, J]$，网格为 `{'X': 8, 'Y': 2}`。每台设备保存多少数据？在 H100 上从 HBM 加载这个数组需要多长时间（假设每块芯片的内存带宽为 `3.4e12`）？

<details>
<summary>点击此处查看答案。 </summary>


$A[I_{XY}, J]$ 将第一维（I）同时沿 X 和 Y 两条硬件轴分片。在这个例子中，局部形状为 $(\lvert I\rvert /(\lvert X\rvert \cdot \lvert Y\rvert), \lvert J\rvert)$。对于给定示例，全局形状是 `fp32[1024, 4096]`，因此局部形状是 `fp32[64, 4096]`。

由于每块 GPU 保存 `4 * 64 * 4096 = 1MiB` 字节，加载大约需要 `1e6 / 3.4e12 = 294ns`；不过，因为数据量如此之小，各种开销很可能会让实际时间显著更长。

</details>

**将这些分片方案可视化：** 我们来观察一个被拆分到 4 台设备上的二维数据数组，尝试直观理解这些分片方案：

![](/images/scaling-book/img/sharding-colored1.png)

我们把矩阵的*完全复制*形式直接写成 $A[I, J]$，不附加任何分片指派。这意味着*每台*设备都包含整个矩阵的一份完整副本。

![](/images/scaling-book/img/sharding-colored2.png)

我们可以用下标形式的网格轴来表示其中某个维度已沿一条网格轴分区。例如，$A[I_X, J]$ 表示逻辑轴 **I** 已沿网格维度 **X** 分区，但维度 **J** *没有*分区，因此这些数据块仍在网格轴 **Y** 上*部分复制*。

![](/images/scaling-book/img/sharding-colored3.png)

$A[I_X, J_Y]$ 表示逻辑轴 **I** 已沿网格轴 **X** 分区，而维度 **J** 已沿网格轴 **Y** 分区。

![](/images/scaling-book/img/sharding-colored4.png)

下图展示了其他可能性：

![](/images/scaling-book/img/sharding-colored5.png)

这里，$A[I_{XY}, J]$ 表示我们把网格轴 **X** 和 **Y** 看作一个更大的展平维度，并将命名轴 **I** 分区到所有设备上。多个网格轴下标的顺序很重要，因为它指定了在网格上遍历分区的顺序。

![](/images/scaling-book/img/sharding-colored6.png)

最后请注意，我们*不能*让多个命名轴沿*同一个*网格维度分片。例如，$A[I_X, J_X]$ 是一种没有意义且被禁止的分片。一旦某个网格维度被用于分片数组的一个维度，它在某种意义上就已经“用掉”了。

**小测验：** 设 **A** 是一个形状为 `int8[128, 2048]`、分片为 $A[I_{XY}, J]$、网格为 `Mesh({'X': 2, 'Y': 8, 'Z': 2})`（总计 32 台设备）的数组。每台设备上的 **A** 使用多少内存？跨所有设备计算，**A** 总共使用多少内存？

<details>
<summary>点击此处查看答案。 </summary>


**答案：** 数组 **A** 沿 X 和 Y 分片，并沿 Z 复制，因此它在每台设备上的形状是 `int8[128 / (2 * 8), 2048] = int8[8, 2048]`，大小为 `8 * 2048 = 16,384` 字节。由于它沿 Z 复制，而在每个 Z 平面内又完全沿 X 和 Y 分片，所以原数组共有 2 份完整副本（每个 Z 平面一份）。因此，跨所有设备的总大小为：原数组大小 × Z 方向的副本数 = 128 * 2048 * 2 = 总计 512 KiB。我们也可以这样验证：32 台设备 × 每台设备 16,384 字节 = 总计 512 KiB。

</details>

<span id="how-do-we-describe-this-in-code"></span>

### 如何用代码描述这些分片？

到目前为止，我们一直避而不谈代码，但现在正好可以先睹为快。JAX 使用的命名分片语法与我们上面描述的抽象语法非常接近。我们会在[第 10 节](../10-jax/#programming-tpus-in-jax)详细讨论这一点，不过这里先做一个快速预览。你可以在 Google Colab 中[运行这里的代码](https://colab.research.google.com/drive/15cxw66eABwZPG-V4QFmbLfiykPFf_gaP?usp=sharing)，并对结果进行性能分析，看看 JAX 如何处理不同的分片方案。这段代码完成 3 件事：

1. 创建一个 **jax.Mesh**，把 8 块 TPU 映射为 4x2 网格，并将名称 'X' 和 'Y' 分别指派给两条轴。
2. 创建矩阵 A 和 B，其中 A 的两个维度都被分片，B 则沿输出维度分片。
3. 编译并执行一个简单的矩阵乘法，返回分片数组。

```py
import jax
import jax.numpy as jnp

# Create our mesh! We're running on a TPU v2-8 4x2 slice with names 'X' and 'Y'.
# The Auto axis type tells JAX to let the XLA compiler infer intermediate shardings.
assert len(jax.devices()) == 8
Auto = jax.sharding.AxisType.Auto
mesh = jax.make_mesh(axis_shapes=(4, 2), axis_names=('X', 'Y'), axis_types=(Auto, Auto))

# A little utility function to help define our sharding. A PartitionSpec is our
# sharding (a mapping from axes to names).
def P(*args):
  return jax.NamedSharding(mesh, jax.sharding.PartitionSpec(*args))

# We shard both A and B over the non-contracting dimension and A over the contracting dim.
A = jnp.zeros((8, 2048), dtype=jnp.bfloat16, device=P('X', 'Y'))
B = jnp.zeros((2048, 8192), dtype=jnp.bfloat16, device=P(None, 'Y'))

# We can perform a matmul on these sharded arrays! out_shardings tells us how we want
# the output to be sharded. JAX/XLA handles the rest of the sharding for us.
y = jax.jit(lambda A, B: jnp.einsum('BD,DF->BF', A, B), out_shardings=P('X', 'Y'))(A, B)
```

JAX 很酷的一点是，这些数组的行为就像它们根本没有分片一样！`B.shape` 会告诉我们全局形状或逻辑形状 (2048, 8192)。我们必须真正查看 `B.addressable_shards`，才能知道它在局部如何分片。我们可以对这些数组执行操作，JAX 会尝试判断该如何广播或重塑它们以完成操作。例如在上面的示例中，**A** 的局部形状是 `[2, 1024]`，**B** 的局部形状是 `[2048, 4096]`。JAX/XLA 会根据需要，自动在这些数组之间加入通信，以完成最终的乘法。

<span id="computation-with-sharded-arrays"></span>

## 使用分片数组进行计算

如果有一个分布在许多设备上的数据数组，并希望对它执行数学运算，那么将数据与计算同时分片会带来哪些额外开销？

显然，这取决于具体计算。

* 对于*逐元素*运算，在分布式数组上执行操作**没有额外开销**。
* 当我们希望跨多个设备上的元素执行运算时，情况就复杂了。好在对于大多数机器学习任务，几乎所有计算都以矩阵乘法的形式进行，而矩阵乘法相对容易分析。

本节其余部分将讨论如何对分片矩阵做乘法。近似来看，这需要搬动矩阵的数据块，以便完整地对每个数据块执行乘法或求和。**每种分片方案都会涉及不同的通信。** 例如，$A[I_X, J] \cdot B[J, K_Y] \to C[I_X, K_Y]$ 不需要任何通信即可相乘，因为*收缩维度*（J，也就是我们实际求和的维度）没有分片。但是，如果希望输出不分片（即 $A[I_X, J] \cdot B[J, K_Y] \to C[I, K]$），就需要把 $A$ 和 $B$，或者把 $C$，复制到每台设备上（使用 *AllGather（全聚合）*）。这两种选择的通信成本不同，因此需要计算成本并选择较低者。

<details>
<summary>你可以从“分块矩阵乘法”的角度理解这一点。 </summary>


为了理解这一点，回顾“分块矩阵”这一概念会很有帮助；它就是由矩阵嵌套而成的矩阵：

$$
\begin{equation}
\begin{pmatrix}
a_{00} & a_{01} & a_{02} & a_{03} \\
a_{10} & a_{11} & a_{12} & a_{13} \\
a_{20} & a_{21} & a_{22} & a_{23} \\
a_{30} & a_{31} & a_{32} & a_{33}
\end{pmatrix}
=
\left(
\begin{matrix}
\begin{bmatrix}
a_{00} & a_{01} \\
a_{10} & a_{11}
\end{bmatrix} \\
\begin{bmatrix}
a_{20} & a_{21} \\
a_{30} & a_{31}
\end{bmatrix}
\end{matrix}
\begin{matrix}
\begin{bmatrix}
a_{02} & a_{03} \\
a_{12} & a_{13}
\end{bmatrix} \\
\begin{bmatrix}
a_{22} & a_{23} \\
a_{32} & a_{33}
\end{bmatrix}
\end{matrix}
\right)
=
\begin{pmatrix}
\mathbf{A_{00}} & \mathbf{A_{01}} \\
\mathbf{A_{10}} & \mathbf{A_{11}}
\end{pmatrix}
\end{equation}
$$

矩阵乘法有一个很好的性质：当乘数矩阵用分块形式表示时，其乘积也可以按照标准规则写成分块矩阵乘法：

$$
\begin{equation}
\begin{pmatrix}
A_{00} & A_{01} \\
A_{10} & A_{11}
\end{pmatrix}
\cdot
\begin{pmatrix}
B_{00} & B_{01} \\
B_{10} & B_{11}
\end{pmatrix}
=
\begin{pmatrix}
A_{00}B_{00} + A_{01}B_{10} & A_{00}B_{01} + A_{01}B_{11} \\
A_{10}B_{00} + A_{11}B_{10} & A_{10}B_{01} + A_{11}B_{11}
\end{pmatrix}
\end{equation}
$$

这意味着，实现分布式矩阵乘法可以归结为：在网络中搬运这些分片数据块，对数据块执行*局部*矩阵乘法，然后对结果求和。**接下来的问题，就是要加入什么通信，以及这种通信有多昂贵。**

</details>

方便的是，所有可能的分片方式大致可以归纳为 4 种需要考虑的情形；对于每种情形，需要加入何种通信都有一条相应规则
1. **[情形 1](#case-1-neither-multiplicand-has-a-sharded-contracting-dimension)：** 两个输入都未沿收缩维度分片。*我们可以直接对局部分片相乘，不需要任何通信。*
2. **[情形 2](#case-2-one-multiplicand-has-a-sharded-contracting-dimension)：** 一个输入的收缩维度已分片。*通常，我们会沿收缩维度对这个分片输入执行“AllGather”。*
3. **[情形 3](#case-3-both-multiplicands-have-sharded-contracting-dimensions)：** 两个输入都沿收缩维度分片。*我们可以先对局部分片相乘，然后对结果执行“AllReduce”。*
4. **[情形 4](#case-4-both-multiplicands-have-a-non-contracting-dimension-sharded-along-the-same-axis)：** 两个输入各有一个非收缩维度沿同一条轴分片。若不先对其中一个输入执行 AllGather，我们就无法继续。

你可以把它们当成必须遵守的规则，但理解这些规则为何成立、成本多高也同样很有价值。下面我们将逐一详细讨论。

<span id="case-1-neither-multiplicand-has-a-sharded-contracting-dimension"></span>

### 情形 1：两个乘数的收缩维度均未分片

**引理：** 在分片矩阵相乘时，只要收缩维度没有分片，并且两个矩阵没有沿同一条轴分片，那么计算就是有效的，输出也会沿用输入的分片方案。例如，下面的计算完全没有问题

$$
\begin{equation*}
\mathbf{A}[I_X, J] \cdot \mathbf{B}[J, K_Y] \rightarrow \mathbf{C}[I_X, K_Y]
\end{equation*}
$$

它完全不需要通信，所得张量沿 X 和 Y 两个硬件维度分片。请试着想想为什么。基本原因是，计算与分片是*相互独立*的：每个批次条目都拥有收缩轴的某个局部数据块，可以在这个数据块上完成乘法与归约。以下任一种情形都能正常工作，并遵循这条规则：

$$
\begin{align*}
\mathbf{A}[I, J] \cdot \mathbf{B}[J, K] \rightarrow &\ \mathbf{C}[I, K] \\
\mathbf{A}[I_X, J] \cdot \mathbf{B}[J, K] \rightarrow &\ \mathbf{C}[I_X, K]\\
\mathbf{A}[I, J] \cdot \mathbf{B}[J, K_Y] \rightarrow &\ \mathbf{C}[I, K_Y]\\
\mathbf{A}[I_X, J] \cdot \mathbf{B}[J, K_Y] \rightarrow &\ \mathbf{C}[I_X, K_Y]
\end{align*}
$$

由于 **A** 和 **B** 的收缩维度 **J** 都没有分片，我们只需执行输入的局部分块矩阵乘法，结果就会*已经*按照期望的输出分片方案完成分片。当两个乘数的非收缩维度沿同一条轴分片时，情况便不再如此（详见[无效分片](#case-4-both-multiplicands-have-a-non-contracting-dimension-sharded-along-the-same-axis)一节）。

<span id="case-2-one-multiplicand-has-a-sharded-contracting-dimension"></span>

### 情形 2：一个乘数的收缩维度已分片

考虑输入 **A** 沿收缩维度 **J** 分片，而 **B** 完全复制时该怎么办：

$$
\mathbf{A}[I, J_X] \cdot \mathbf{B}[J, K] \rightarrow \mathbf{C}[I, K]
$$

我们不能直接把 **A** 和 **B** 的局部数据块相乘，因为需要对 **A** 的完整收缩维度求和，而这个维度已沿 X 轴拆分。通常，我们先对 **A** 的分片执行“**AllGather**”，让每台设备都获得完整副本，然后才与 **B** 相乘：

$$
\textbf{AllGather}_X[I, J_X] \rightarrow \mathbf{A}[I, J]
$$

$$
\mathbf{A}[I, J] \cdot \mathbf{B}[J, K] \rightarrow \mathbf{C}[I, K]
$$

这样，实际乘法就可以完全在每台设备上完成。

**要点：** 当相乘的矩阵中有一个沿收缩维度分片时，我们通常先对它执行 AllGather，使收缩维度不再分片，然后执行局部矩阵乘法。

请注意，当 **B** 也没有沿 X 分片时，我们还可以执行局部的部分矩阵乘法，然后对分片的部分和求和（也就是执行 *AllReduce（全归约）*）；这样可以把计算分片，但通常通信成本更高。在某些情况下，这种方法可能更快，不过实践中 **B** 通常也会分片。下方的[问题 4](#some-problems-to-work)会推导它何时更优。

**什么是 AllGather？** AllGather 是我们要讨论的第一个核心 [MPI](https://en.wikipedia.org/wiki/Message_Passing_Interface) 通信原语。AllGather 会*移除*沿某条轴的分片，把散布在各台设备上的分片重新组装到该轴上的*每台*设备中。使用上面的记法，AllGather 会从一组轴中移除下标，例如

$$
\textbf{AllGather}_{XY}(A[I_{XY}, J]) \rightarrow A[I, J]
$$

我们不必移除某个维度的全部下标。例如，$A[I_{XY}, J] \rightarrow A[I_Y, J]$ 也是 AllGather，只是仅沿一条轴执行。还请注意，我们也可能希望使用 AllGather 移除*非收缩*维度上的分片，例如在下面的矩阵乘法中：

$$
A[I_X, J] \cdot B[J, K] \rightarrow C[I, K]
$$

我们既可以一开始就对 **A** 执行 AllGather 来移除输入分片，也可以先执行分片矩阵乘法，再对结果 **C** 执行 AllGather。

**AllGather 实际是如何执行的？** 要围绕单条 TPU 轴（一个环）执行一维 AllGather，我们基本上会让每块 TPU 沿环传递自己的分片，直到每台设备都拥有一份副本。[^ch3-2] 下面是一段动画：

![图：这段动画展示了如何围绕一组 8 台 TPU 或 GPU 设备执行 AllGather。开始时，每台设备各有数组的 1/8；结束时，每台设备都获得完整副本。](/images/scaling-book/img/all-gather.gif)

我们可以单向执行 AllGather，也可以双向执行（上图展示的是双向）。单向执行时，每块 TPU 都会沿环传送大小为 $\text{bytes} / N$ 的数据块，共经过 $N - 1$ 跳。双向执行时，则要经过 $\lfloor \frac{N}{2} \rfloor$ 跳，每跳的数据大小为 $2 \cdot \text{bytes} / N$。

**这需要多长时间？** 我们以双向 AllGather 为例，计算它的耗时。设 $V$ 为数组的字节数，$X$ 为收缩维度上的分片数。根据上图，每一跳都会在每个方向上发送 $V / \lvert X\rvert$ 字节，因此每跳耗时为

$$
T_{hop} = \frac{2 \cdot V}{\lvert X \rvert \cdot W_\text{ici}}
$$

其中 $W_\text{ici}$ 是**双向** ICI 带宽。[^ch3-3] 为了抵达每块 TPU，总共需要发送 $\lvert X\rvert / 2$ 跳[^ch3-4]，因此整个归约过程耗时为

$$
T_{total} = \frac{2 \cdot V \cdot X}{2 \cdot X \cdot W_\text{ici}}
$$

$$
T_{total} = \frac{V}{W_\text{ici}}
$$

请注意，这个时间**与 $X$ 无关！** 这多少有些令人惊讶，因为它意味着，尽管 TPU 只进行局部连接，连接的局部性却并不重要。我们的瓶颈只是每条链路的速度。

**要点：** 在吞吐量受限的场景中执行 AllGather（或者 ReduceScatter 或 AllReduce）时，实际通信时间只取决于数组大小和可用带宽，而与数组分片所跨的设备数量无关！

**关于 ICI 延迟的一点说明：** 无论数据量有多大，每次跨越 ICI 链路的一跳都有某种固有开销，通常约为 1us。这意味着，当数组 $A$ 很小、每跳耗时低于 1us 时，就会进入“延迟受限”状态，此时计算时间*确实*取决于 $X$。

<details>
<summary>点击此处查看完整细节。 </summary>


设 $T_\text{min}$ 为单次跳转的最短时间。那么

$$
T_{hop} = \max \left[ T_{min}, \frac{2 \cdot V}{X \cdot W_\text{ici}} \right]
$$

$$
T_{total} = \max \left[ \frac{T_{min} \cdot X}{2}, \frac{V}{W_\text{ici}} \right]
$$

因为我们总共执行 $X / 2$ 跳。对于大型归约或聚合操作，我们显然是带宽受限的。发送的数据如此之多，以至于每跳的开销几乎可以忽略不计。但对于小数组（例如从模型中采样时），这一开销不可忽略，ICI 带宽也不再相关；此时我们完全受延迟限制。换一种说法：对于某款特定 TPU，例如单向 ICI 带宽为 `4.5e10` 的 TPU v5e，发送任何小于 `4.5e10 * 1e-6 = 45kB` 的缓冲区都会受延迟限制。

</details>

下面是在 TPU v5e 8x16 切片上测得的 AllGather 带宽。数组沿大小为 16 的轴分片，因此拥有一个完整的双向环。

![图：TPU v5e 执行 AllGather 时的实测带宽与估算链路带宽。橙色 BW 是 AllGather 实际聚合的每秒字节数，蓝色曲线则是根据已知集合通信成本计算出的实测单向链路带宽。](/images/scaling-book/img/all-gather-bandwidth.png)

请注意，我们不仅达到了标称峰值带宽（`4.5e10`）的约 95%，而且在大约 10MB 时就达到了这个峰值；经过 16 路分片后，每台设备约为 625kB（*顺带一提*：这比 GPU 好得多）。

**沿多条轴执行 AllGather 时会怎样？** 沿多条轴聚合时，我们可以在 ICI 的多个维度上执行聚合。例如，AllGather<sub>XY</sub>([B, D<sub>XY</sub>]) 会跨两条硬件网格轴操作。这会使可用带宽提高 $N_\text{axes}$ 倍。

考虑延迟时，我们会得到下面这条通用规则：

$$
T_{total} = \max \left[ \frac{T_{min} \cdot \sum_{i} |X_i|}{2}, \frac{V}{W_\text{ici} \cdot N_\text{axes}} \right]
$$

其中 $\sum_i \lvert X_i \rvert / 2$ 是 TPU 网格中最长路径的长度。

**小测验 2［AllGather 时间］：** 使用[第 2 部分](../02-tpus/#how-to-think-about-tpus)中的数值，在网格为 `{'X': 8, 'Y': 4}` 的 TPU v5e 上，对 bfloat16 格式、$E = 2048$、$F = 8192$ 的数据执行 AllGather<sub>Y</sub>([E<sub>Y</sub>, F]) → [E, F] 需要多长时间？若 $E=256, F=256$，又需要多长时间？

<details>
<summary>点击此处查看答案。 </summary>


**答案：** 我们先计算几个基本量：

1) TPU v5e 的两条轴各自拥有 4.5e10 字节/秒的单向 ICI 带宽。
2) 对于 (a) 中的 bfloat16 数据，我们有 $A[E_Y, F]$，因此每台设备保存一个形状为 bf16[512, 8192] 的数组，大小为 512 * 8192 * 2 = 8.4MB。完整数组的大小为 2048 * 8192 * 2 = 34MB。

*对于第 (1) 部分*，可以使用上面的公式。因为我们沿一条轴执行 AllGather，所以 $T_{\text{comms}} = \text{34e6} / \text{9e10} = \text{377us}$。为了确认不是延迟受限，我们知道沿大小为 4 的轴最多需要 3 跳，因此延迟下界约为 3us，离这个界还很远。不过，TPU v5e 只有在某条轴大小为 16 时才有环绕连接，所以这里*实际上无法执行完全双向的 AllGather*。为了让边缘的数据到达另一侧边缘，必须走 3 跳，因此理论上更接近 $T_{\text{comms}} = 3 * \text{8.4e6} / \text{4.5e10} = 560\mu s$。[**这里**](https://imgur.com/a/RkvpRGQ)是[这个 Colab](https://colab.research.google.com/drive/15tDZMfNqm2vJjvSzw5VC9qtSwc5td-oV?usp=sharing) 得到的**实际性能分析结果**，其中显示为 $680 \mu s$；考虑到实际带宽很可能达不到理论值的 100%，这个结果很合理！*对于第 (2) 部分*，每个分片的大小为 `64 * 256 * 2 = 32kB. 32e3 / 4.5e10 = 0.7us`，因此我们受延迟限制。由于需要 3 跳，耗时大约为 3 * 1us = 3us。[实践中更接近 8us。](https://imgur.com/a/HZLQmYs)

</details>

**注意：** 当我们拥有 `{'X': 16, 'Y': 4}` 这样的二维网格时，并不要求每条轴都对应某条特定的*硬件*轴。例如，上述网格可以描述一个 4x4x4 的 TPU v5p 立方体，其中 $X$ 轴映射了 2 条硬件轴。稍后讨论跨多条轴的数据并行时，这一点会派上用场。

<span id="case-3-both-multiplicands-have-sharded-contracting-dimensions"></span>

### 情形 3：两个乘数的收缩维度均已分片

第三种基本情形是，两个乘数的收缩维度都沿同一条网格轴分片：

$$
\textbf{A}[I, J_X] \cdot \textbf{B}[J_X, K] \rightarrow C[I, K]
$$

在这种情况下，局部的分片分块矩阵乘法至少是*可以*执行的，因为它们共享同一组收缩索引。但是，每个乘积只代表完整目标乘积的一个*部分和*，并且沿 **X** 维度的每台设备都会留下这个最终目标乘积的不同*部分和*。这种情况极其常见，因此我们扩展记法，显式标记这种状态：

$$
\textbf{A}[I, J_X] \cdot_\text{LOCAL} \textbf{B}[J_X, K] \rightarrow C[I, K] \{\ U_X \}
$$

记法 **{ U<sub>X</sub> }** 读作“沿网格轴 X **尚未归约**”，表示该运算在某种意义上仍“不完整”，还要等待最后一次求和才能完成。$\cdot_\text{LOCAL}$ 语法表示我们执行局部求和，但让结果保持未归约状态。

这可以看作矩阵乘法与外积之间的如下关系：

$$
A \cdot B = \sum_{i=1}^{P} \underbrace{A_{:,i} \otimes B_{i,:}}_{\in \mathbb{R}^{n \times m}}
$$

其中 ⊗ 表示外积。因此，如果轴 **X** 上的 TPU **i** 保存 **A** 的第 **i** 列和 **B** 的第 **i** 行，我们就可以执行局部矩阵乘法，得到 $A_{:,i} \otimes B_{i,:} \in \mathbb{R}_{n\times m}$。在这个矩阵的每个条目中，都保存着 **A • B** 对应条目求和式的第 **i** 项。我们仍需对 **P** 执行这次求和——而 **P** 已沿网格轴 **X** 分片——才能得到完整的 **A • B**。如果把 **A** 和 **B** 写成分块形式（即分片），再对所得结果的每个分片求和，原理也完全相同。

我们可以沿 **X** 轴执行一次完整的 **AllReduce** 来完成这次求和：

$$
\begin{align*}
A[I, J_X] \cdot_\text{LOCAL} B[J_X, K] \rightarrow &\ C[I, K] \{ U_X \} \\
\textbf{AllReduce}_X C[I, K] \{ U_X \} \rightarrow &\ C[I, K]
\end{align*}
$$

AllReduce 会消除部分和，使沿该轴的*每台*设备都得到相同的、已完全求和的值。AllReduce 是本节将讨论的几种关键通信操作中的第二种；第一种是 AllGather，另外两种是 ReduceScatter 和 AllToAll。AllReduce 接收一个包含未归约（部分求和）轴的数组，让这些分片沿未归约轴传递并累加结果，从而完成求和。其签名为

$$
\textbf{AllReduce}_Y A[I_X, J] \{U_Y\} \rightarrow A[I_X, J]
$$

这意味着，它只是移除 $\\{U_Y\\}$ 后缀，结果的其他部分保持不变。

**AllReduce 的成本有多高？** 可以这样理解 AllReduce 的执行过程：每台设备把自己的分片发送给邻居，并把收到的所有分片相加。显然，这比 AllGather 更昂贵，因为每个“分片”的形状都与完整数组相同。一般而言，**AllReduce 的成本是 AllGather 的两倍。** 一种理解方式是，**AllReduce** 可以表示为另外两种原语的组合：一次 **ReduceScatter（归约分散）** 加一次 **AllGather**。与 AllReduce 一样，ReduceScatter 会消除数组中的部分和，但会产生沿给定维度“分散”或分区的输出。AllGather 会收集所有这些片段，并沿该物理轴“取消分区/取消分片/复制”相应逻辑轴。

$$
\begin{align*}
\textbf{ReduceScatter}_{Y,J} : A[I_X,J] \{U_Y\} \rightarrow &\ A[I_X, J_Y] \\
\textbf{AllGather}_Y : A[I_X, J_Y] \rightarrow &\ A[I_X, J]
\end{align*}
$$

**那么 ReduceScatter 呢？** 正如 AllGather 会重新组装分片数组（移除一个下标），ReduceScatter 会对尚未归约/部分求和的数组求和，然后沿同一条网格轴分散（分片）另一条逻辑轴。$X[F]\\{U_Y\\} \to X[F_Y]$。动画展示了具体过程：请注意，它与 AllGather 十分相似，只是我们不再保留每个分片，而是把它们相加。因此，除去执行归约本身所需的时间，其延迟大致相同。

![](/images/scaling-book/img/reduce-scatter.gif)

每一跳的通信时间，就是每分片字节数 $V / Y$ 除以带宽 $W_\text{ici}$，与 AllGather 相同，因此有

$$
T_{\text{comms per AllGather or ReduceScatter}} = \frac{V}{W_\text{ici}}
$$

$$
T_{\text{comms per AllReduce}} = 2 \cdot \frac{V}{W_\text{ici}}
$$

其中 $W_\text{ici}$ 是双向带宽，前提是我们拥有一个可用于归约的完整环。

<span id="case-4-both-multiplicands-have-a-non-contracting-dimension-sharded-along-the-same-axis"></span>

### 情形 4：两个乘数各有一个非收缩维度沿同一轴分片

在对张量分片时，每个网格维度最多只能出现一次。应用上述规则有时会产生违反这条约束的情况，例如：

$$
A[I_X, J] \cdot B[J, K_X] \rightarrow C[I_X, K_X]
$$

这种分片无效，因为沿维度 **X** 的某个分片（比如 **i**）将保存 **C** 的第 **(i, i)** 个分片，也就是一个对角条目。这样一来，所有分片合在一起也没有足够信息来恢复除对角条目以外的任何内容，因此我们不能允许这种分片。

解决办法是对其中某些维度执行 AllGather。这里有两种选择：

$$
\begin{align*}
\textbf{AllGather}_X A[I_X, J] \rightarrow &\ A[I, J] \\
A[I, J] \cdot B[J, K_X] \rightarrow &\ C[I, K_X]
\end{align*}
$$

或者

$$
\begin{align*}
\textbf{AllGather}_X B[J, K_X] \rightarrow &\ B[J, K] \\
A[I_X, J] \cdot B[J, K] \rightarrow &\ C[I_X, K]
\end{align*}
$$

无论哪种情况，结果的形状都只会提到一次 **X**。具体选择哪一种，取决于后续运算需要什么分片方案。

<span id="a-deeper-dive-into-tpu-communication-primitives"></span>

## 深入理解 TPU 通信原语

前面的 4 种情形引出了执行分片矩阵乘法时会用到的几种“核心通信原语”：

1. **AllGather：** 从分片中移除一个下标，收集各个分片。
2. **ReduceScatter：** 通过对某条轴上的分片求和，从数组中移除“尚未归约”后缀，同时让数组沿另一条轴保持分片状态。
3. **AllReduce：** 移除“尚未归约”后缀，使数组沿该轴不再分片。

还有一种核心通信原语值得一提，它会出现在专家混合（Mixture of Experts，MoE）模型及其他计算中：**AllToAll（全互换）**。

<span id="our-final-communication-primitive-the-alltoall"></span>

### 最后一种通信原语：AllToAll

最后一种基础集合通信在研究分片矩阵乘法时并不会自然出现，但在实践中却始终会遇到，它就是 **AllToAll**；更准确地说，是*分片转置*或重新分片操作这一特例。例如

$$
\textbf{AllToAll}_{X, J} A[I_X, J] \rightarrow A[I, J_X]
$$

在分片计算的不同区域采用彼此不兼容的布局方案时，通常需要使用 AllToAll 来重新排列分片布局。它会自然地出现在分片专家混合模型中。*你可以把 AllToAll 理解为把一个下标从一条轴移到另一条轴*。因为 AllToAll 不需要在环上复制每个分片的全部数据，所以它实际上比 AllGather *更便宜*（成本为后者的 ¼）。[^ch3-5]

![](/images/scaling-book/img/all-to-all.gif)

推广到 ND AllToAll 后，对于一个跨所有设备合计有 $V$ 字节、位于 AxBxC 网格上的数组，总成本为

$$
T_\text{comms per AllToAll} = \frac{V \cdot \max(A, B, C, ...)}{4 \cdot N \cdot W_\text{ici}}
$$

其中，照例 $W_\text{ici}$ 表示双向 ICI 带宽，而 $N = A \cdot B \cdot C \cdot \ldots$ 是设备总数。等价地，使用每台设备的字节数 $V / N$ 表示，成本为 $(V / N) \cdot \max(A, B, C, ...) / (4 \cdot W_\text{ici})$。对于一维网格，它化简为 $V / (4 \cdot W_\text{ici})$，即 AllGather 成本的 1/4。在二维情况下，成本实际上会随较短轴的大小增加而下降。

*题外话：如果你想粗略推导这一结论，可以从一维环面 $\mathbb{Z} / N\mathbb{Z}$ 开始。随机选取源节点和目标节点时，两者的平均距离为 N / 4 跳，成本因此为 $(V \cdot N) / (4 * N)$。再考虑 ND 环面时，各条轴基本彼此独立。每个节点拥有 $1 / N$ 字节，平均需要让自己的数据跳转 $\max(A, B, C, …) / 4$ 跳。也可以从对分带宽推导：在 AllToAll 中，网格的每一半都向另一半发送自己一半的数据（$V / 4$ 字节）。最窄的对分垂直切过最长轴，穿过 $2 \cdot N / \max(A, B, …)$ 条链路（两个切面，计入环绕连接），因此单向带宽为 $N \cdot W_\text{ici} / \max(A, B, …)$。用数据量除以带宽，便得到上面的公式。*

<span id="more-about-the-reducescatter"></span>

### 进一步了解 ReduceScatter

ReduceScatter 比初看起来更加基础，因为它实际上是 AllGather 的导数，反之亦然。也就是说，如果在前向传播中有：

$$
\textbf{AllGather}_X A[I_X] \rightarrow A[I]
$$

那么我们会对反向模式导数 **A'**（通常在各分片上并不相同）执行 ReduceScatter，从而得到分片后的 **A'**：

$$
\textbf{ReduceScatter}_X A'[I] \{ U_X \} \rightarrow A'[I_X]
$$

同样，前向传播中的 $\text{ReduceScatter}_X(A[I] \{U_X\}) \to A[I_X]$ 意味着，反向传播中会有 $\text{AllGather}_{X}(A'[I_X]) \to A'[I]$。

<details>
<summary>点击此处了解 AllGather 与 ReduceScatter 为何互为导数。 </summary>


原因在于，广播和归约作为线性算子互为转置，而 AllGather 和 ReduceScatter 分别是广播和归约的外积（也称为[克罗内克积](https://en.wikipedia.org/wiki/Kronecker_product)）。具体来说，如果有向量 $x \in \mathbb{R}^n$、任意数量的设备 $p \in \mathbb{N}$，并令 $u = (1, \ldots, 1) \in \mathbb{R}^p$，我们就可以按下面的方式定义广播和归约；这应该与你对它们的直观理解一致：

$$
\begin{align*}
\text{broadcast} &: \mathbb{R}^n \rightarrow \mathbb{R}^{p n} \\
\text{broadcast} &= u \otimes \mathbf{I}_n \\
\text{reduce} &: \mathbb{R}^{p n} \rightarrow \mathbb{R}^n \\
\text{reduce} &= u^T \otimes \mathbf{I}_n
\end{align*}
$$

我们通过一个 $n = 1$、$p = 2$ 的例子来看看它具体是什么样子。如果 $x = (7)$，则有 $\text{broadcast}(x) = \left(\begin{pmatrix} 1 \\ 1 \end{pmatrix} \otimes \begin{pmatrix} 1 \end{pmatrix}\right) x = \begin{pmatrix} 1 \\ 1 \end{pmatrix} x = \begin{pmatrix}  7\\  7  \end{pmatrix} \in \mathbb{R}^{p n}$。这与我们的预期一致：把 $\mathbb{R}^n$ 中的向量广播到 $\mathbb{R}^{pn}$。现在令 $y = (8, 9)$，则有 $\text{reduce}(y) = \left(\begin{pmatrix} 1 & 1 \end{pmatrix} \otimes \begin{pmatrix} 1\end{pmatrix}\right) y = \begin{pmatrix} 1 & 1  \end{pmatrix} \begin{pmatrix}  8 \\ 9  \end{pmatrix} = \begin{pmatrix}   17    \end{pmatrix}$。这同样符合预期：把 $\mathbb{R}^{p n}$ 中的向量归约为 $\mathbb{R}^{n}$ 中的向量。由于 $(A \otimes B)^T = A^T \otimes B^T$ 对任意两个矩阵 $A$ 和 $B$ 都成立，可知 $\text{reduce} = \text{broadcast}^T$。我们可以用下面的外积重新得到 AllGather 和 ReduceScatter：

$$
\begin{align*}
\text{AllGather} &: \mathbb{R}^{p n} \rightarrow \mathbb{R}^{p^2 n} \\
\text{AllGather} &= \text{broadcast} \otimes \mathbf{I}_p \\
\text{ReduceScatter} &= \mathbb{R}^{p^2 n} \rightarrow \mathbb{R}^{p n} \\
\text{ReduceScatter} &= \text{reduce} \otimes \mathbf{I}_p
\end{align*}
$$

这里，我们把 $\mathbb{R}^{p^2 n}$ 看作 $\mathbb{R}^{p \times p n}$，也就是每台设备一个 $\mathbb{R}^{p n}$ 向量，共有 $p$ 台设备。建议你用一些小例子试一试，比如 $n = 2$、$p = 3$，看看这些算子写成矩阵时是什么样子。再次利用同样的转置性质，可以得到 $\text{AllGather}^T = \text{ReduceScatter}$，当然也有 $\text{ReduceScatter}^T = \text{AllGather}$。这种转置会出现在反向传播过程中：假设有 $y = Ax$，其中 $A$ 是某个线性算子（例如 AllGather 或 ReduceScatter）；那么在反向传播时，我们会得到损失对 $y$ 的导数 $\frac{\partial L}{\partial y}$，并得到 $\frac{\partial L}{\partial x}$，即 $\frac{\partial L}{\partial x} = A^T \frac{\partial L}{\partial y}$。这说明 AllGather 的导数将是 ReduceScatter，反之亦然。

</details>

把 AllReduce 拆成 AllGather 和 ReduceScatter 还有一个方便的性质：我们可以把最后一次 AllGather 推迟到更晚。很多时候，我们并不想付出在所有设备上重新组装完整矩阵乘积副本的成本。相反，即使在两个乘数的收缩维度都已分片的情况下，我们也希望保持分片状态：

$$
A[I, J_X] \cdot B[J_X, K] \rightarrow C[I, K_X]
$$

在这种情况下，我们也可以执行 ReduceScatter 而不是 AllReduce，并选择在更晚的时候再执行 AllGather，即

$$
\begin{align*}
A[I, J_X] \cdot_{LOCAL} B[J_X, K] \rightarrow &\ C[I, K] \{ U_X \} \\
\textbf{ReduceScatter}_{X,K} C[I, K] \{ U_X \} \rightarrow &\ C[I, K_X]
\end{align*}
$$

请注意，ReduceScatter 会*引入*一个分片维度，因此在此处自然可以选择沿命名维度 **I** 或 **K** 进行分片。使用 ReduceScatter 时，我们通常需要选择要在哪个命名维度上引入新分片（不过，更大的建模上下文通常会迫使我们做出唯一选择）。正因如此，我们使用 **ReduceScatter<sub>X,K</sub>** 语法来指定要分片的轴。

<span id="how-to-overlap-matmul-communication-with-compute"></span>

### 如何重叠矩阵乘法中的通信与计算

正如我们在[第 1 部分](../01-roofline/#all-about-rooflines)中讨论的，只要通信足够快，我们通常假设总能让通信与某些有用计算重叠。本节中的集合通信通常可以与矩阵乘法计算本身重叠，但这样做并不简单。我们使用的算法称为**集合通信矩阵乘法（collective matmul）**，最早由 [Wang 等人](https://dl.acm.org/doi/pdf/10.1145/3567955.3567959)描述。下面是一段简化动画，展示了如何实现这种重叠：

![图：这段动画展示了如何让单个分片矩阵—向量乘积与随后发生的 AllReduce 重叠（即上面的情形 3）。一次完整矩阵乘法由多个矩阵—向量乘积组成。](/images/scaling-book/img/ag_matmul.gif)

简而言之，我们可以在对矩阵的一个数据块执行矩阵乘法时，开始对之前的数据块执行环形归约。在某些情况下，还可以沿批次维度或矩阵输入维度分块。我们会在[第 10 部分](../10-jax/#programming-tpus-in-jax)推导一个简单的 JAX 实现，[Mosaic 文档](https://docs.jax.dev/en/latest/pallas/gpu/collective_matmul.html)也提供了一个很好的 GPU 示例。我们鼓励你有机会自己实现一个版本。

<span id="what-have-we-learned"></span>

## 我们学到了什么？

* 数组的分片由 **Mesh** 和 **Sharding** 共同指定：**Mesh** 为 TPU 网格的物理硬件轴命名，**Sharding** 则把网格轴名称指派给数组的逻辑轴。
  * 例如，**A**[I<sub>XY</sub>, J] 描述了一个抽象数组 **A**，其第一维沿 X 和 Y 两条网格轴分片。再结合 Mesh(mesh_shape=(4, 8), axis_names=('X', 'Y'))，或缩写形式 Mesh({'X': 4, 'Y': 8})，我们就知道该数组的第一维被分成了 32 份。

* **对分片数组执行算术运算与对未分片数组执行完全相同，除非你沿分片轴执行收缩。** 在后一种情况下，必须引入某些通信。我们考虑了四种情形：

  1. *两个数组的收缩维度都未分片*：不需要通信。
  2. *一个数组的收缩维度已分片*（或者两个收缩维度沿不同轴分片）：执行运算前，我们先对其中一个输入执行 AllGather。
  3. *两个数组以相同方式沿收缩维度分片：* 我们在局部把分片相乘，然后执行 AllReduce 或 ReduceScatter。
  4. *两个数组各有一个非收缩维度沿同一条网格轴分片：* 我们先对其中一个输入执行 AllGather。

* TPU 大致使用 **4 种核心通信原语**：
  1. AllGather：$[A_X, B] \to [A, B]$
  2. ReduceScatter：$[A, B] \\{U_X\\} \to [A_X, B]$
  3. AllToAll：$[A, B_X] \to [A_X, B]$
  4. AllReduce：$[A_X, B]\\{U_Y\\} \to [A_X, B]$（严格来说不算原语，因为它把 ReduceScatter 与 AllGather 组合在一起）

![](/images/scaling-book/img/all-collectives.png)

* 这些操作各自的成本与延迟**都不取决于轴的大小（只要它们受带宽限制）**，而只取决于输入数组的大小和链路带宽。对于单向 AllGather/ReduceScatter：

$$
T_{\text{comm per AllGather or ReduceScatter}} = \frac{\text{Data volume}}{\text{bandwidth}} \cdot \frac{\text{Axis} - 1}{\text{Axis}}
\longrightarrow \frac{\text{Data volume}}{\text{bandwidth (bidirectional)}}
$$

* AllReduce 由一次 ReduceScatter 后接一次 AllGather 构成，因此成本是上式的 2 倍。AllToAll 只需让分片沿环传递部分路程，因此其成本是 AllGather 的 ¼。总结如下：

| 操作              | 说明                                                                                                                  | 语法                              | 运行时间                                         |
| :---------------- | :-------------------------------------------------------------------------------------------------------------------- | :-------------------------------- | :----------------------------------------------- |
| **AllGather**     | 收集分片数组沿某条轴的所有分片，并移除一个下标。                                                                      | $[A_X, B] \to [A, B]$            | bytes / (bidirectional ICI bandwidth * num_axes) |
| **ReduceScatter** | 沿某条轴对部分求和数组求和，并沿另一条轴将其分片（添加一个下标）。                                                    | $[A, B] \\{U_X\\} \to [A_X, B]$  | 与 AllGather 相同                                |
| **AllReduce**     | 沿某条轴对部分求和数组求和。移除一个 { U<sub>x</sub> }。由 AllGather 和 ReduceScatter 组合而成。                       | $[A_X, B]\\{U_Y\\} \to [A_X, B]$ | 2 * AllGather                                    |
| **AllToAll**      | 聚合（复制）一条轴，并沿同一条轴分片另一个维度。                                                                      | $[A, B_X] \to [A_X, B]$          | 双向环中为 AllGather / 4                         |

<span id="some-problems-to-work"></span>

## 练习题

*下面是一些基于本节内容、很有启发性的练习。暂时不会给出全部答案，但我们会尽可能继续补充答案。*

**问题 1［复制式分片］：** 一个数组的分片为 $A[I_X, J, K, \ldots]$（即只沿 $X$ 分片），网格为 `Mesh({'X': 4, 'Y': 8, 'Z': 2})`。在所有芯片上，$A$ 占用的总字节数与一份数组副本的大小之比是多少？

<details>
<summary>点击此处查看答案。 </summary>


我们的数组只沿大小为 4 的 X 分片，因此每个分片的实际大小是 $[I / 4, J, K, \ldots] = \text{sizeof}(A) / 4$。由于数组沿 Y 和 Z 复制，总大小为 $Y \cdot Z \cdot \text{sizeof}(A)$，所以总大小与单芯片大小之比为 $Y \cdot Z \cdot \text{sizeof}(A) / \text{sizeof}(A) = 16$。

</details>

**问题 2［AllGather 延迟］：** 在网格为 `Mesh({'X': 4, 'Y': 4, 'Z': 4})` 的 TPU v4p 4x4x4 切片上，如果数据类型为 bfloat16，$\text{AllGather}_X([B_X, D_Y])$ 需要多长时间？这里 $B=1024$、$D=4096$。$\text{AllGather}_{XY}([B_X, D_Y])$ 呢？$\text{AllReduce}_Z([B_X, D_Y] \{U_Z \})$ 呢？

<details>
<summary>点击此处查看答案。 </summary>


因为我们拥有完整的 `4x4x4` 立方体，所以所有轴上都有环绕链路，可使用 9e10 的双向带宽。

1. 因为我们只沿一条轴执行聚合，而另一条轴仍然分片，所以实际是在 1 条轴上聚合 $2BD / Y$ 字节。*如果只看 Y 轴上的单个分片，那么沿 X 执行的 AllGather 就像一次未分片的 AllGather，只不过字节数是其 1 / Y。* TPU v4p 的双向 ICI 带宽为 9e10 字节/秒，因此耗时为 $2BD / (\text{9e10} \cdot Y) = 2 \cdot 1024 \cdot 4096 / (\text{9e10} \cdot 4) = 23 \mu s$。

2. 与前面相比，我们拥有两倍带宽，但执行 AllGather 的是完整数组，因此 `T = 2BD / (2 * W) = 2*1024*4096 / (2 * 9e10) = 46us`。这远高于 4us 的延迟下界（每跳 1us），所以没有问题。

3. AllReduce 的成本是 AllGather 的两倍。每个分片的大小为 $2BD / (X * Y)$，因此成本大约为 $4BD / (X * Y * W)$，也就是约 `4 * 1024 * 4096 / (16 * 9e10) = 11.6us`。

*趣味事实：* 第 (1) 和第 (2) 部分实际上并非最优，因为该数组还沿未使用的 Z 轴复制，而我们可以利用这些闲置链路：可以先免费重新分片 $[B_X, D_Y] \to [B_{XZ}, D_Y]$（每台设备只需丢弃自己分片的一部分），然后执行 $\text{AllGather}_{XZ}$（或 $\text{AllGather}_{XYZ}$），以聚合更多轴的方式到达相同终态。这会把第 (1) 部分的时间缩短到 11.5us，把第 (2) 部分缩短到 31us——实践中，你只需从一开始就沿更多轴分片，便能获得这一效果；这也是应该尽可能细致地对数组分片的原因之一。

</details>

**问题 3［延迟受限的 AllGather］：** 假设我们正在执行 $\text{AllGather}_X([B_X])$，但 $B$ 很小（比如 128）。在网格为 `Mesh({'X': 4, 'Y': 4, 'Z': 4})` 的 TPU v4p 4x4x4 切片上，以 bfloat16 格式执行需要多长时间？*提示：你很可能受延迟限制。*

<details>
<summary>点击此处查看答案。 </summary>


我们的 bfloat16 数组总共只占 256 字节，每台设备仅占 64 字节。因为 TPU v4p 上的轴大小为 4，所以有环绕链路，可以双向发送数组。单向带宽为 `4.5e10`，因此每一跳大约需要 `64 / 4.5e10 ~ 0`，显然是延迟受限的。数一下跳数，我们只需 2 跳便可完成整个聚合，因此粗略估计 2us 比较合理。

</details>

**问题 4［矩阵乘法策略］：** 为执行 $X[B, D] \cdot_D Y[D_X, F] \to Z[B, F]$，本节建议先执行 $\text{AllGather}_X(Y[D_X, F])$，再对完全复制的矩阵相乘（情形 2，*策略 1*）。另一种做法是像 $X[B, D_X] \cdot_D Y[D_X, F] \to Z[B, F] \\{U_X\\}$ 那样对局部分片相乘（情形 3，*策略 2*），然后执行 $\text{AllReduce}_X(Z[B, F] \\{ U_X\\})$。两种方法各自执行多少 FLOP 和通信？哪一种更好，为什么？

<details>
<summary>点击此处查看答案。 </summary>


先从我们的基线（*策略 1*）开始。正如前文所示，AllGather 的成本是 $2DF / W_\text{ici}$。得到完全复制的数组后，总计算时间为 $2BDF / C$（其中 $C$ 是加速器的 FLOP/s，因为每块 TPU 执行相同数量的 FLOP）。所以有

$$
T_\text{total (Strategy 1)} = \max\left(\frac{2BDF}{C}, \frac{2DF}{W_\text{ici}}\right)
$$

相比之下，新策略（策略 2）会对 $2BF$ 字节执行 AllReduce，其成本为 $4BF / W_\text{ici}$；但执行的 FLOP 数量减少到原来的 $1 / X$（因为计算已分片）。这意味着我们会执行 $2\cdot B\cdot D\cdot F / X$ 个 FLOP，随后在 bfloat16 中执行的 AllReduce 会通信 $2 \cdot 2 \cdot B \cdot F$ 字节。因此，*策略 2*（不执行 AllGather，只在之后执行 AllReduce）的总时间大约为

$$
T_\text{total} = \max\left(\frac{2BDF}{X \cdot C}, \frac{4BF}{W_\text{ici}}\right)
$$

问题是：*其中哪一项更大？* 当 $D / (X \cdot C) > 2 / W_\text{ici}$，即 $D / 2X > C / W_\text{ici} \approx 2550 \rightarrow X < D / (2 * 2550)$ 时，策略 (2) 受计算限制。我们可以合理地预期 $D \approx 8k$，这意味着大约要有 $X < 2$，而这不太可能——因此，策略 2 基本上总是受通信限制。对于基线（策略 1），当 $B < C / W_\text{ici} = 2550$ 时受通信限制，这种情况很常见，但并非总是如此。

因此，如果 $B < 2550$，两种情形都受通信限制，并且有

$$
T_\text{comms for Strategy 2} < T_\text{comms for Strategy 1} \Leftrightarrow \frac{4BF}{W_\text{ici}} < \frac{2DF}{W_\text{ici}}
$$

当 $D > 2B$，也就是 $2B < 5100$ 时，上式成立。这种情况经常出现，所以当批次较小时，策略 2 有时可能更好。当批次较大（$B > 2550$）时，有

$$
T_\text{comms for Strategy 2} < T_\text{math for Strategy 1} \Leftrightarrow \frac{4BF}{W_\text{ici}} < \frac{2BDF}{C}
$$

当 $2 / W_\text{ici} < D / C$，即 $D > 2 * 2550 = 5100$ 时，上式成立；对大型模型来说通常如此。因此，对于大型模型，这种备选策略一般更好，除非 $D$ 较小。

*为什么我们不总是这样做？* 实践中有时确实会这样做，但很少出现这种情况：矩阵乘法中，一个输入的收缩维度沿某条轴分片，而另一个输入并没有沿该轴分片。例如，如果使用 FSDP（将在[第 5 节](../05-training/#how-to-parallelize-a-transformer-for-training)解释），我们会沿数据维度分片参数，但激活值*同样也会沿数据维度分片*。因此，从这个意义上说，这种情况并不常见。

</details>

**问题 5［最低延迟］：** 假设我想在 TPU v4p 4x4x4 上以尽可能低的延迟执行矩阵乘法 $A[I, J] \cdot_J B[J, K] \to C[I, K]$。假设输入可以任意分片，但结果应当完全复制。输入应如何分片？FLOP 总数和通信时间分别是多少？

<details>
<summary>点击此处查看（部分）答案。 </summary>


这里不会给出完整答案，但我们先列出最有可能的四种方案：

1. $A[I_{XYZ}, J] \cdot B[J, K]$ + 末尾执行 AG
2. $A[I, J] \cdot B[J, K_{XYZ}]$ + 末尾执行 AG
3. $A[I, J_{XYZ}] \cdot B[J_{XYZ}, K]$ + 末尾执行 AR
4. $A[I, J] \cdot B[J, K]$（完全复制）

还可以考虑让不同维度沿不同网格轴分片，但这不太可能改变最终成本。除 (4) 外，每块 TPU 的 FLOP 总数都相同，但各自通信量不同。接下来，只需计算每种方案的通信成本，看看哪一个最低。简而言之，(1) 和 (2) 同样好。

</details>

**问题 6：** 假设我们想在 TPU v5e 4x4 上执行 $A[I_X, J_Y] \cdot_J B[J_Y, K] \to C[I_X, K]$。需要执行什么通信？通信与计算各花费多少时间？

* 那么 $A[I_X, J] \cdot_J B[J_X, K_Y] \to C[I_X, K_Y]$ 呢？这是训练中最标准的场景，我们在其中组合数据分片、张量分片与 ZeRO 分片。
* 那么 $A[I_X, J] \cdot_J B[J, K_Y] \to C[I_X, K_Y]$ 呢？这是推理中的标准场景，其中执行纯张量并行（+数据并行）。

**问题 7：** 一个典型的 Transformer 块包含两个矩阵 $W_\text{in}[D, F]$ 和 $W_\text{out}[F, D]$，其中 $F \gg D$。假设批大小为 B，那么整个块就是 $In[B, D] \cdot W_\text{in}[D, F] \cdot W_\text{out}[F, D]$。取 $D=8192$、$F=32768$、$B=128$，并假设所有数据都采用 bfloat16。假设在 TPU v5e 2x2 切片上运行，但姑且设每块 TPU 只有 300MB 可用内存。应如何对 In、$W_\text{in}$、$W_\text{out}$ 和 Out 分片，才能既不超过内存限制，又尽可能缩短总时间？通信和 FLOP 分别花费多长时间？*提示：最终输出不必完全复制，但它应当与输入采用相同的分片方案，以便重复应用这个“层”。*

<details>
<summary>点击此处查看（部分）答案。 </summary>


先考虑内存。两个大矩阵各自使用 `2 * 8192 * 32768 = 536MB`。激活值 `In` 的大小是 `2 * 128 * 8192 = 2MB`（小到不必担心）。由于每台设备只有 300MB 可用内存，显然必须对矩阵乘法分片。

1. $In[B_X, D] * W_\text{in}[D_{XY}, F] * W_\text{out}[F, D_{XY}] \rightarrow Out[B_X, D]$（这通常称为 FSDP）
2. $In[B, D_{XY}] * W_\text{in}[D, F_{XY}] * W_\text{out}[F_{XY}, D] \rightarrow Out[B, D_{XY}]$（这称为张量并行）

第一种方案很糟糕，因为必须先对大权重或激活值执行 AllGather。第二种方案需要在开头执行 AllGather，在末尾执行 ReduceScatter（它比 AllReduce 更便宜）。其余计算留作练习。

</details>

**问题 8［挑战］：** 以上面的简短代码片段为模板，分配一个分片数组，并使用 pmap 或 shard_map 对 4 种主要通信原语（AllGather、AllReduce、ReduceScatter 和 AllToAll）逐一进行基准测试。你需要使用 `jax.lax.all_gather`、`jax.lax.psum`、`jax.lax.psum_scatter` 和 `jax.lax.all_to_all`。你理解这些函数的语义吗？它们各自需要多长时间？

**问题 9［分片矩阵乘法的另一种策略？］：** 我们在[上文](#case-2-one-multiplicand-has-a-sharded-contracting-dimension)声称，当矩阵乘法中只有一个输入沿收缩维度分片时，应当对分片矩阵执行 AllGather，然后在局部执行所得收缩。你或许会想到另一种策略：执行分片矩阵乘法，然后对结果执行 AllReduce（仿佛两个输入都沿收缩维度分片），也就是通过下面两步计算 $A[I, J_X] *_J B[J, K] \to C[I, K]$：

1. $C[I, K] \\{ U_X \\} = A[I, J_X] \cdot B[J_X, K]$
2. $C[I, K] = \text{AllReduce}(C[I, K] \\{ U_X\\})$

请回答：

1. 对矩阵 $A[N, M]$ 和 $B[M, K]$ 明确写出该算法，使用索引准确展示在哪台设备上执行什么计算。假设 $A$ 以 $A[I, J_X]$ 的方式分片到 ND 台设备上，并希望输出复制到所有设备。
2. 现在假设你可以接受最终结果不在每台设备上复制，而是分片（沿 N 或 K 维度均可）。上述算法应如何改变？
3. 只考虑上述策略（第 2 问中的策略，而不是第 1 问）的通信成本，它与先对 A 执行 AllGather、再执行矩阵乘法的算法相比如何？

<details>
<summary>点击此处查看答案。 </summary>



1. 首先计算外积，并把结果存入 $O[N, K]: o_{kj} = \sum_i a_{ki} b_{ij}$。请注意，重复索引并不是被收缩的那个索引，因为我们正在计算外积。这里，求和范围是在当前所用设备上存储的那组 i 值。例如，假设收缩轴大小为 16，共有 4 台设备，那么在设备 0 上，i 的范围为 {0, 1, 2, 3}；在设备 1 上，i 的范围为 {4, 5, 6, 7}；在设备 2 上，i 的范围为 {8, 9, 10, 11}；在设备 3 上，i 的范围为 {12, 13, 14, 15}。然后，对各台设备上 $O[N, K]$ 的部分和执行 AllReduce，形成完整的 $O[N, K]$。
2. 在第 2 步中，我们可以执行成本更低的 ReduceScatter，而不必执行 AllReduce；可以沿任意一条轴分片：$[N, K] \\{ U_X \\} \to [N_X, K]$ 或 $[N, K] \\{ U_X \\} \to [N, K_X]$。
3. 如上面正文所述，在吞吐量受限时，执行 AllGather 的成本与 ReduceScatter 相同；它只由所处理完整矩阵的大小决定。因此，在“先聚合、后矩阵乘法”的算法中，这一成本按 $NM$ 扩展（因为我们执行的是 $\text{AllGather}$，对象为 $A$）；在“先矩阵乘法、后归约分散”的算法中，它按 NK 扩展（因为我们对 $O$ 执行 ReduceScatter）。所以，两种算法的通信成本之比为 `M/K`。

</details>

**问题 10：AllToAll 趣题：** 上表指出，在吞吐量受限的区域内，执行 AllToAll 的时间比执行 AllGather 或 ReduceScatter 低 4 倍。在这道题中，我们将理解这个 4 倍因子从何而来，也会看看如果只有单向 ICI 链路而非双向 ICI 链路，这个因子会如何变化。

1. 先从单向情况开始。设有 *D* 台设备组成环形拓扑，并希望对 N x N 矩阵 $A[I_X, J]$ 执行 AllGather 或 ReduceScatter（为简单起见，假设 $D$ 整除 $N$）。描述这两种集合通信涉及的通信，并计算整个算法期间通过**单条** ICI 链路传输的标量（浮点数或整数）总数。
2. 接下来考虑 AllToAll，仍然采用单向 ICI。此时的算法与 AllGather 有何不同？计算该算法中通过单条 ICI 链路传输的标量数。
3. 你应该会发现，第 (a) 问与第 (b) 问答案之比是个漂亮的数字。请用简单语言解释这个因子从何而来。
4. 现在加入双向通信。这对 AllGather 所需总时间有何影响？
5. 加入双向通信对 AllToAll 所需总时间有何影响？
6. 现在请直接解释双向环中 AllGather 时间与 AllToAll 时间的比值。

<details>
<summary>点击此处查看答案。 </summary>


(1) **解答：** 过程很简单：在算法的每一步，每台设备都会向最近的邻居发送矩阵中一条由单个分片构成的“带状块”（总大小为 $\frac{N}{D} \times N$ 个元素）。这一过程会发生 $D-1$ 次，因为每个分片都需要被传到除其起始设备外的所有设备上。因此，每台设备总共传输 $\frac{N^2(D-1)}{D}$ 个标量；也就是说，这么多标量会流过单条 ICI 链路。

**答案：** $N^2 (1-\frac{1}{D})$；也可以直接写作 $N^2$（当 $D >> 1$ 时）。

(2) **解答：** 从通信角度看，AllToAll 与 AllGather 的关键区别在于：AllToAll 不需要把某台设备上分片的全部内容都传到其他每台设备。设想某台设备（称为设备 0）保存的分片是 $[A, B, C, D]$（这里 A、B、C、D 都是矩阵，我们用 4 台设备组成的环作为示例）。矩阵 $A$ 不需要发送到任何地方；矩阵 $B$ 最终要到设备 1；矩阵 $C$ 最终要到设备 2；矩阵 $D$ 最终要到设备 3。因此，算法第一步把 $B$、$C$ 和 $D$ 发送到设备 1；下一步，设备 1 继续把 $C$ 和 $D$ 发到设备 2；最后一步，设备 2 只把 $D$ 发到设备 3。此时传输的参数总数是 $(\text{size of A/B/C/D}) * (3 + 2 + 1)$。A/B/C/D 的大小（现在回到一般情形）为 $\frac{N^2}{D^2}$；同样，在一般情况下，$(3 + 2 + 1)$ 项变成 $((D-1) + (D-2) + … + 1)$，即 $\frac{(D)(D-1)}{2}$。因此，通过单条 ICI 链路传输的总字节数为 $\frac{N^2(D-1)}{D \times 2}$。

**答案：** $\frac{N^2}{2}(1-\frac{1}{D})$；也可以直接写作 $\frac{N^2}{2}$（当 $D >> 1$ 时）。

(3) **解答：** 这个因子就是 $\frac{1}{2}$；也就是说，在单向环形拓扑上，AllToAll 的成本只有 AllGather/ReduceScatter 的一半。回顾上面的推导，根本原因在于：在 AllGather 中，我们会把同样大小的数据块传输 $(D-1)$ 次，也就是计算 $ \text{tiny block size} * (D + D + D + … + D)$；而在 AllToAll 中，计算的则是 $\text{tiny block size} * (D + D-1 + D-2 + … + 1)$。这个二倍因子本质上来自 $1 + 2 + \ldots + n = n(n+1)/2$。

(4) **解答：** 现在，任一链路需要承载的标量总数都会减半，因为在双向环中，每条“分片带”可以同时向两个方向发送。

(5) **解答：** 在这种情况下，相比单向情形，我们获得了 4 倍收益。观察单条分片带中每个大小为 (N2/D2) 的数据块最终会去哪里，最容易理解这一点；不妨考虑源自设备 0 的那条分片带。我们不再像单向情形中那样，让一个数据块移动 D-1 的距离、另一个移动 D - 2 的距离，以此类推直至 1；现在，我们把分片带拆成向右或向左移动的数据块，最大移动距离为 floor(D/2)。因此，对应的求和变成 $D/2 + D/2 - 1 + D/2 - 2 + … = D/2 \cdot (D/2+1)/2$，也就是约 $D^2/8$；这是 $D$ 很大时的极限。与单向情形中的 $D^2/2$ 相比，可以看出我们获得了 4 倍收益。

(6) **解答：** 我们已经看到，在单向环中，AllToAll 的时间本来就比 AllGather 快两倍；原因是我们不需要把完整的分片带发送给每台设备。然后，加入双向传输后，AllToAll 获得了 4 倍收益，而 AllGather 只获得 2 倍收益。把这些比例合在一起，就得到我们寻找的 4 倍因子。

</details>

<span id="thats-it-for-part-3-for-part-4-about-transformer-math-click-here"></span>

### 第 3 部分到此结束！要继续阅读第 4 部分（关于 Transformer 数学），请点击[这里](../04-transformers/#all-the-transformer-math-you-need-to-know)！

[^ch3-1]: 值得注意的是，我们也可能为了速度而选择并行。即使数据能装入更少的芯片，扩展到更多芯片也只是为我们提供了更多 FLOP/s。例如在推理期间，我们有时可以装入较小拓扑，但仍会选择扩展到较大拓扑以降低延迟。同样，训练时也常扩展到更多芯片，以缩短步进时间。
[^ch3-2]: GPU AllGather 也可以这样工作：把一个节点内的 GPU 组成环，并按那个（任意的）顺序在环中传递数据块。
[^ch3-3]: 分子中的因子 2 来自我们使用了双向带宽。每个方向发送 $V / X$，总计发送 $2V / X$。
[^ch3-4]: 严格来说，是 $\lfloor X / 2 \rfloor$
[^ch3-5]: 对于设备数为偶数的双向环，每台设备会向右发送 $(N/2 + (N/2-1) + … + 1)$ 个数据块，向左发送 $((N/2-1) + … + 1)$ 个数据块，$= 0.5 \cdot (N / 2) \cdot (N/2 + 1) + 0.5 \cdot (N / 2) \cdot (N/2 - 1) = N^2/4$。每个数据块（也就是分片的分片）的大小为 $\text{bytes} / N^2$，因此每台设备的成本为 $(\text{bytes} / N^2) \cdot N^2 / 4 = \text{bytes} / 4$。由于总带宽随设备数量扩展，这一结果也可扩展到所有设备。
