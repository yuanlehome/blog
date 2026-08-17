---
title: "如何并行化 Transformer 训练"
description: "本章讨论 LLM 训练期间使用的四种主要并行方案：数据并行、完全分片数据并行（FSDP）、张量并行和流水线并行。对于每一种方案，我们都会计算系统从什么时候开始受到通信瓶颈的限制。"
chapter: 5
order: 5
part: 2
partTitle: "Transformer"
sourcePath: "training.md"
sourceCommit: "44109cacac9c5a9809a81c68ae4d45d7d2632ea6"
---

<span id="how-to-parallelize-a-transformer-for-training"></span>

# 如何并行化 Transformer 训练

<span id="what-do-we-mean-by-scaling"></span>

## 我们所说的扩展是什么意思？

“模型扩展”的目标，是在增加用于训练或推理的芯片数量时，实现吞吐量成比例的线性增长（我们称之为*强扩展*）。单芯片性能取决于内存带宽与 FLOPs 之间的权衡，而集群级性能则取决于能否让芯片间通信与有效 FLOPs 重叠，从而隐藏通信开销。这并不容易，因为增加芯片数量会提高通信负载，同时减少每台设备上可用于隐藏通信的计算量。正如我们在[第 3 节](../03-sharding/#sharded-matrices-and-how-to-multiply-them)中看到的那样，分片矩阵乘法通常需要代价高昂的 AllGather 或 ReduceScatter，而它们可能阻塞 TPU，使其无法执行有效工作。本节的目标就是找出这些操作在什么情况下会变得*过于昂贵*。

本节将讨论四种常见的并行方案：（纯）**数据并行、完全分片数据并行**（FSDP / ZeRO 分片）、**张量并行**（也称模型并行），以及（简要介绍）**流水线并行**。对于每一种方案，我们都会说明其通信成本，以及该成本从何时开始成为计算成本的瓶颈。[^ch5-1] 在本节中，你可以只关注芯片间通信成本，因为只要单芯片批大小足够大，从 HBM 向 MXU 传输数据就已经能与计算重叠。

为了简化本节中的计算，我们将使用以下记号。

| 记号 | 含义（模型参数）                                             |
| :------- | :--------------------------------------------------------------------- |
| D        | **d**<sub>model</sub>（隐藏维度／残差流维度）      |
| F        | **d**<sub>ff</sub>（前馈维度）                        |
| B        | 批维度（批中的 token 数；是总量，不是每设备数量） |
| T        | 序列长度                                                        |
| L        | 模型中的层数                                          |

| 记号 | 含义（硬件特征）                                                                 |
| :------- | :------------------------------------------------------------------------------------------------ |
| C        | 每芯片 FLOPS/s                                                                                  |
| W        | 网络带宽（双向，常以下标表示，例如 $W_{\text{ici}}$ 或 $W_{\text{dcn}}$） |
| X        | 沿网格轴 X 的芯片数量                                                                 |
| Y        | 沿另一个标为 Y 的网格轴的芯片数量                                           |
| Z        | 沿第三个标为 Z 的网格轴的芯片数量                                                |

为简单起见，**我们将 Transformer 近似为一叠 MLP 块**——正如在[第 4 节](../04-transformers/#all-the-transformer-math-you-need-to-know)中看到的，对于较大的模型，注意力只占 FLOPs 中相对较小的一部分。我们还会忽略门控矩阵乘法，于是每一层只剩下如下简单结构：

![图：简化的 Transformer 层。我们将每个 FFW 块视为两个矩阵的串联：W_in: bf16[D, F]（升维投影）和 W_out: bf16[F, D]（降维投影），输入为 In: bf16[B, D]。](/images/scaling-book/img/transformer-layer.png)

<details>
<summary>下面是这个不含任何并行的小型 Transformer 的完整算法。 </summary>


<div markdown=1 class="algorithm">

<strong>前向传播：</strong>需要计算 Loss[B]

1.  Tmp[B, F] = In[B, D] *<sub>D</sub> W<sub>in</sub>[D, F]
2.  Out[B, D] = Tmp[B, F] *<sub>F</sub> W<sub>out</sub>[F, D]
3.  Loss[B] = ...

<strong>反向传播：</strong>需要计算 dW<sub>out</sub>[F, D]、dW<sub>in</sub>[D, F]

1.  dOut[B, D] = ...
2.  dW<sub>out</sub>[F, D] = Tmp[B, F] *<sub>B</sub> dOut[B, D]
3.  dTmp[B, F] = dOut[B, D] *<sub>D</sub> W<sub>out</sub>[F, D]
4.  dW<sub>in</sub>[D, F] = In[B, D] *<sub>B</sub> dTmp[B, F]
5.  dIn[B, D] = dTmp[B, F] \*<sub>F</sub> W<sub>in</sub>[D, F]（*前面各层需要*）

</div>

我们给出这个算法，是为了与加入通信的算法进行比较。

</details>

下面是我们将要讨论的 4 种并行方案。每种方案都可以看作由上图中 **In、W<sub>in</sub>、W<sub>out</sub> 和 Out** 的分片方式唯一确定。

<strong>1. 数据并行：</strong>*激活值沿批维度分片，参数和优化器状态在每台设备上复制。通信只发生在反向传播期间。*

$$
\text{In}[B_X, D] \cdot_D W_\text{in}[D, F] \cdot_F W_\text{out}[F, D] \rightarrow \text{Out}[B_X, D]
$$

<strong>2. 完全分片数据并行（FSDP 或 ZeRO-3）：</strong>*激活值沿批维度分片（与纯数据并行相同），参数沿同一网格轴分片，并在前向传播中使用前即时进行 AllGather。优化器状态也沿批维度分片。这样可减少重复占用的内存。*

$$
\text{In}[B_X, D] \cdot_D W_\text{in}[D_X, F] \cdot_F W_\text{out}[F, D_X] \rightarrow \text{Out}[B_X, D]
$$

<strong>3. 张量并行（也称 Megatron 分片或模型并行）：</strong>*激活值沿 D（$d_\text{model}$）分片，参数沿 F（$d_{ff}$）分片。在每个块的前后分别对激活值执行 AllGather 和 ReduceScatter。可与 FSDP 配合使用。*

$$
\text{In}[B, D_Y] \cdot_D W_\text{in}[D, F_Y] \cdot_F W_\text{out}[F_Y, D] \rightarrow \text{Out}[B, D_Y]
$$

<strong>4. 流水线并行：</strong>*权重沿层维度分片，激活值被拆成微批，并沿层维度依次传递。流水线阶段之间的通信量很小（只需把激活值移动一跳）。借用一下记号：*

$$
\text{In}[L_Z, B, D][i] \cdot_D W_\text{in}[L_Z, D, F][i] \cdot_F W_\text{out}[L_Z, F, D][i] \rightarrow \text{Out}[L_Z, B, D][i]
$$

<span id="data-parallelism"></span>

### 数据并行

**写法：** $\text{In}[B_X, D] \cdot_D W_\text{in}[D, F] \cdot_F W_\text{out}[F, D] \rightarrow \text{Out}[B_X, D]$

如果模型即使以很小的批大小（>240 个 token，以确保计算受限）也能放进单芯片，**你就应该始终使用简单的数据并行。** 只要 TPU 数量少于批大小，纯数据并行就能把激活值分摊到任意数量的 TPU 上。前向传播不涉及通信，但在每一步结束时，**每个 TPU 都会对本地梯度执行 AllReduce，在更新参数之前使梯度同步。**

![图：纯数据并行（前向传播）示意图。激活值（左侧）沿批维度完全分片，而权重被完整复制，因此每个 TPU 都拥有一份完全相同的权重副本。这意味着权重占用的总内存增加到 N 倍，但前向传播不需要通信。](/images/scaling-book/img/data-parallelism.png)

<details>
<summary>下面是前向传播和反向传播的完整算法。为了简洁，我们借用记号，把 dL/dOut 写作 dOut。 </summary>


<div markdown=1 class="algorithm">

**纯数据并行算法：**

<strong>前向传播：</strong>需要计算 Loss[B<sub>X</sub>]

1.  Tmp[B<sub>X</sub>, F] = In[B<sub>X</sub>, D] \*<sub>D</sub> W<sub>in</sub>[D, F]
2.  Out[B<sub>X</sub>, D] = Tmp[B<sub>X</sub>, F] \*<sub>F</sub> W<sub>out</sub>[F, D]
3.  Loss[B<sub>X</sub>] = ...

<strong>反向传播：</strong>需要计算 dW<sub>out</sub>[F, D]、dW<sub>in</sub>[D, F]

1.  dOut[B<sub>X</sub>, D] = ...
2.  dW<sub>out</sub>[F, D] {U<sub>X</sub>} = Tmp[B<sub>X</sub>, F] \*<sub>B</sub> dOut[B<sub>X</sub>, D]
3.  dW<sub>out</sub>[F, D] = **AllReduce**(dW<sub>out</sub>[F, D] {U<sub>X</sub>})（*不在关键路径上，可以异步执行*）
4.  dTmp[B<sub>X</sub>, F] = dOut[B<sub>X</sub>, D] \*<sub>D</sub> W<sub>out</sub>[F, D]
5.  dW<sub>in</sub>[D, F] {U<sub>X</sub>} = In[B<sub>X</sub>, D] \*<sub>B</sub> dTmp[B<sub>X</sub>, F]
6.  dW<sub>in</sub>[D, F] = **AllReduce**(dW<sub>in</sub>[D, F] {U<sub>X</sub>})（*不在关键路径上，可以异步执行*）
7.  dIn[B<sub>X</sub>, D] = dTmp[B<sub>X</sub>, F] \*<sub>F</sub> W<sub>in</sub>[D, F]（*前面各层需要*）

</div>

我们忽略损失函数的细节，并将 $\text{Tmp} = W_\text{in} \cdot \text{In}$ 简写为 Tmp。请注意，尽管最终损失是平均值 **AllReduce**(Loss[B<sub>X</sub>])，但只有在反向传播中对权重梯度求平均时，我们才需要计算 AllReduce。

</details>

请注意，前向传播没有通信——**所有通信都发生在反向传播中**！反向传播还有一个很棒的特性：AllReduce 不在“关键路径”上，这意味着每个 AllReduce 都可以在方便的时候执行，而且不会阻塞后续操作。即使总体通信成本超过总计算成本，它*仍然可能成为我们的瓶颈*，但从实现角度看，这种方案宽容得多。我们将会看到，模型／张量并行并不具备这个特性。

**为什么要这样做？** 纯数据并行通过沿批维度拆分激活值来减轻激活值的内存压力；只要有更多芯片用来拆分批维度，我们几乎可以任意增大批大小。尤其是在训练期间，激活值往往主导内存使用，这一点非常有帮助。

**为什么不这样做？** 纯数据并行对模型参数或优化器状态造成的内存压力毫无帮助。这意味着，对于大规模的、真正有意思的模型，如果参数与优化器状态无法装进单个 TPU，纯数据并行就很少有用。为了让你对规模有个概念：如果使用 bf16 参数和 Adam 的 fp32 优化器状态进行训练[^ch5-2]，可容纳的最大模型参数量为 $\text{TPU memory} / 10$；例如，在配备 96GB HBM 的 TPUv5p 芯片上采用纯数据并行，大约能容纳 9B 参数。

**要点**：使用 Adam 和纯数据并行能够训练的最大模型满足 $\text{num\_params} = \text{HBM per device} / 10$。对于 TPU v5p，这大约是 9B 参数。[^ch5-3]

*为了让这种方案在训练真实模型时真正有用，我们至少需要对模型参数或优化器进行部分分片。*

**什么时候通信会成为瓶颈？** 从上面可以看到，每层有两个 AllReduce，每个大小为 $2DF$（对于 bf16 权重）。数据并行在什么时候会让我们变为通信受限？

如上表所示，令 $C$ = 每芯片 FLOPs，$W_{\text{ici}}$ = **双向**网络带宽，$X$ = 对批进行分区时使用的分片数[^ch5-4]。我们来计算执行相关矩阵乘法所需的时间 $T_\text{math}$ 和所需的通信时间 $T_\text{comms}$。由于这种并行方案的前向传播不需要通信，因此只需为反向传播计算这些量。

<em>通信时间：</em>根据前一节的结论，在一维网格中执行 AllReduce 所需的时间只取决于被 AllReduce 数组的总字节数和 ICI 带宽 $W_\text{ici}$；具体而言，AllReduce 时间为 $2 \cdot \text{total bytes} / W_\text{ici}$。由于我们需要分别对 $W_\text{in}$ 和 $W_\text{out}$ 执行 AllReduce，所以每层有 2 次 AllReduce。每次 AllReduce 的对象都是一个权重矩阵，即包含 $DF$ 个参数、占 $2DF$ 字节的数组。综合起来，单层中 AllReduce 的总时间为

$$
\begin{align}
T_\text{comms} &= \frac{2 \cdot 2 \cdot 2 \cdot D \cdot F}{W_\text{ici}}. \\
\end{align}
$$

<em>矩阵乘法时间：</em>每层在前向传播中包含两次矩阵乘法，在反向传播中包含四次矩阵乘法，每次都需要 $2(B/X)DF$ FLOPs。因此，单层反向传播有

$$
\begin{align}
T_\text{math} &= \frac{2 \cdot 2 \cdot 2 \cdot B \cdot D \cdot F}{X \cdot C} \\
\end{align}
$$

由于二者重叠，每层总时间是这两个量中的较大者：

$$
\begin{aligned}
T &\approx \max(\frac{8 \cdot B \cdot D \cdot F}{X \cdot C}, \frac{8 \cdot D \cdot F}{W_\text{ici}}) \\
T &\approx 8 \cdot D \cdot F \cdot \max(\frac{B}{X \cdot C}, \frac{1}{W_\text{ici}})
\end{aligned}
$$

当 $T_\text{math}/T_\text{comms} > 1$ 时，即满足下式时，我们变为计算受限：

$$
\begin{align}
\frac{B}{X} > \frac{C}{W_\text{ici}}.
\end{align}
$$

结论是，要让数据并行保持计算受限，每设备批大小 $B / X$ 必须超过 ICI 运算强度 $C / W_\text{ici}$。归根结底，这是因为计算时间随每设备批大小变化，而通信时间与这个量无关（因为传输的是模型权重）。请注意，条件 $B/X > C/W_\text{ici}$ 与单设备计算受限规则 $B > 240$ 十分相似；在后者中，规则同样来自计算时间随批大小变化，而数据传输量（在 $B \ll F, D$ 的范围内）与批大小无关这一事实。

让我们代入一些真实数字来感受一下规模。对于 TPUv5p，一维 ICI 数据并行时 `C=4.6e14`，`W=2 * 9e10`，因此**要避免通信受限，每芯片的批大小必须至少为 2,550**。由于可以沿多个轴执行数据并行，如果把 TPUv5p pod 的三个轴全部用于纯数据并行，我们就能把 $W_\text{ici}$ 带宽提高到 3 倍，并把下限降到每个 TPU 仅 BS=850，即每个 pod（8960 块芯片）每批 760 万个 token！**这说明纯数据并行其实很难遇到瓶颈！**

**说明［上下文并行］：**在本节中，$B$ 始终指**以 token 数计的总批大小**。不过显然，我们的批由许多不同序列组成，那么这该如何理解？对 MLP 来说，**token 就是 token**！它们属于同一个序列还是两个不同序列并不重要。因此，我们基本可以自由地同时沿批维度和序列维度进行数据并行：这称为上下文并行或序列并行，但你可以把它简单地看作另一种数据并行。注意力比 MLP 更棘手，因为我们会执行一些跨序列计算，不过可以在注意力期间收集 KV 或 Q，并仔细重叠 FLOPs 与通信来处理（通常使用一种称为“环形注意力”的方法）。本节中，我们将完全忽略序列维度，并假定使用了某种程度的批并行或序列并行。

<strong>关于多个网格轴的说明：</strong>我们应该快速说明一下，多个轴会如何影响可用带宽。为某种并行策略使用多个网格轴时，我们会获得更多带宽。

* **定义：**$M_X$（$M_Y$、$M_Z$ 等）是给定并行策略跨越的硬件网格轴数量。
* <strong>影响（带宽受限时）：</strong>使用 $M$ 个轴可提供（$\approx M$ 倍的）聚合链路带宽，因此集合通信时间按 $\propto 1/M_X$ 缩放。

<span id="fully-sharded-data-parallelism-fsdp"></span>

### 完全分片数据并行（FSDP）

**写法：** $\text{In}[B_X, D] \cdot_D W_\text{in}[D_X, F] \cdot_F W_\text{out}[F, D_X] \rightarrow \text{Out}[B_X, D]$

完全分片数据并行（通常称为 FSDP 或 ZeRO 分片[[zero]](../#ref-zero)）把模型的优化器状态和权重拆分到各数据并行分片上，并根据需要高效地收集和散播它们。**与纯数据并行相比，FSDP 可大幅降低每设备内存使用并节省反向传播 FLOPs，而额外开销极小。**

![图：FSDP 沿数据维度对 Win 的收缩维度和 Wout 的输出维度进行分片。这样可以减少内存使用，但（根据第 3 节）要求我们在执行矩阵乘法之前收集 W 的权重。请注意，激活值（左侧）并未沿收缩维度分片，因此才迫使我们执行收集。还要注意，权重的优化器状态也同样沿收缩维度分片。](/images/scaling-book/img/fsdp.png)

你应该还记得（来自[第 3 节](../03-sharding/#sharded-matrices-and-how-to-multiply-them)），AllReduce 可以分解为一次 AllGather 和一次 ReduceScatter。这意味着，我们可以不为标准数据并行执行完整的梯度 AllReduce，而是把权重和优化器状态分片到不同芯片上，在前向传播期间的每一层对其执行 AllGather，并在反向传播期间对权重执行 ReduceScatter，且不增加额外成本。

<details>
<summary>下面是 FSDP 的完整算法。 </summary>


<div markdown=1 class="algorithm">

**完全分片数据并行（FSDP）：**

<strong>前向传播：</strong>需要计算 Loss[B<sub>X</sub>]

1.  W<sub>in</sub>[D, F] = **AllGather**(W<sub>in</sub>[D<sub>X</sub>, F])（*不在关键路径上，可以在前一层期间执行*）
2.  Tmp[B<sub>X</sub>, F] = In[B<sub>X</sub>, D] \*<sub>D</sub> W<sub>in</sub>[D, F]（*现在可以丢弃 W<sub>in</sub>[D, F]*）
3.  W<sub>out</sub>[F, D] = **AllGather**(W<sub>out</sub>[F, D<sub>X</sub>])（*不在关键路径上，可以在前一层期间执行*）
4.  Out[B<sub>X</sub>, D] = Tmp[B<sub>X</sub>, F] \*<sub>F</sub> W<sub>out</sub>[F, D]
5.  Loss[B<sub>X</sub>] = ...

<strong>反向传播：</strong>需要计算 dW<sub>out</sub>[F, D<sub>X</sub>]、dW<sub>in</sub>[D<sub>X</sub>, F]

1.  dOut[B<sub>X</sub>, D] = ...
2.  dW<sub>out</sub>[F, D] {U<sub>X</sub>} = Tmp[B<sub>X</sub>, F] \*<sub>B</sub> dOut[B<sub>X</sub>, D]
3.  dW<sub>out</sub>[F, D<sub>X</sub>] = **ReduceScatter**(dW<sub>out</sub>[F, D] {U<sub>X</sub>})（*不在关键路径上，可以异步执行*）
4.  W<sub>out</sub>[F, D] = **AllGather**(W<sub>out</sub>[F, D<sub>X</sub>])（*可以提前执行*）
5.  dTmp[B<sub>X</sub>, F] = dOut[B<sub>X</sub>, D] \*<sub>D</sub> W<sub>out</sub>[F, D]（*可在此处丢弃 W<sub>out</sub>[F, D]*）
6.  dW<sub>in</sub>[D,F] {U<sub>X</sub>} = In[B<sub>X</sub>, D] \*<sub>B</sub> dTmp[B<sub>X</sub>, F]
7.  dW<sub>in</sub>[D<sub>X</sub>, F] = **ReduceScatter**(dW<sub>in</sub>[D, F] {U<sub>X</sub>})（*不在关键路径上，可以异步执行*）
8.  W<sub>in</sub>[D, F] = **AllGather**(W<sub>in</sub>[D<sub>X</sub>, F])（*可以提前执行*）
9.  dIn[B<sub>X</sub>, D] = dTmp[B<sub>X</sub>, F] \*<sub>F</sub> W<sub>in</sub>[D, F]（*前面各层需要*）（*可在此处丢弃 W<sub>in</sub>[D, F]*）

</div>

</details>

这种方法也称为“ZeRO 分片”，其名称来自“Zero Redundancy Optimizer”（零冗余优化器），因为我们既不执行任何不必要的计算，也不存储任何不必要的状态。ZeRO-{1,2,3} 分别指以这种方式对优化器状态、梯度和权重进行分片。由于它们的通信成本相同[^ch5-5]，我们基本上总是可以采用 ZeRO-3 分片，把参数、梯度和优化器状态全部分摊到一组设备上。

**为什么要这样做？** 标准数据并行包含大量重复工作。每个 TPU 都对完整梯度执行 AllReduce，然后更新完整的优化器状态（所有 TPU 上做着完全相同的工作），再更新参数（同样完全重复）。使用 ZeRO 分片（对梯度／优化器状态进行分片）时，可以不执行 AllReduce，而是对梯度执行 ReduceScatter，只更新属于自己的优化器状态分片，再更新参数分片，之后在前向传播需要时对参数执行 AllGather。

**什么时候通信会成为瓶颈？** 我们的相对 FLOPs 成本与通信成本和纯数据并行完全相同，因为反向传播中的每个 AllReduce 都变成了 AllGather + ReduceScatter。回想一下，AllReduce 由 AllGather 和 ReduceScatter 实现，二者各占一半成本。这里我们对前向传播建模，因为它的 FLOPs 与通信之比和反向传播相同：

$$
\begin{aligned}
T_\text{math} &= \frac{2 \cdot 2 \cdot B \cdot D \cdot F}{X \cdot C} \\
T_\text{comms} &= \frac{2 \cdot 2 \cdot D \cdot F}{W_\text{ici}} \\
T &\approx \max\left(\frac{4 \cdot B \cdot D \cdot F}{X \cdot C}, \frac{4 \cdot D \cdot F}{W_\text{ici}}\right) \\
T &\approx 4 \cdot D \cdot F \cdot \max\left(\frac{B}{X \cdot C}, \frac{1}{W_\text{ici}}\right)
\end{aligned}
$$

因此，与纯数据并行一样，当 $B / X > C / W_\text{ici}$ 时，也就是每设备批大小 $B/X$ 超过“ICI 运算强度”$C/W_\text{ici}$ 时（对于 v5p 为 `4.59e14 / 1.8e11 = 2550`），我们是计算受限的。这对我们非常有利，因为它意味着，只要每设备批大小足以让纯数据并行处于计算受限状态，我们就可以——无需担心脱离计算受限范围——直接升级为 FSDP，从而节省大量参数和优化器状态内存！尽管前向传播确实增加了通信，但这项成本无关紧要，因为它只需与前向传播 FLOPs 重叠即可。

<strong>要点：</strong>在 TPUv5 上，当每设备批大小小于 $2550 / M_X$ 时，FSDP 和纯数据并行都会变为带宽受限，其中 $M_X$ 是网格轴的数量。

例如，DeepSeek-V2（近期少数公开训练批大小信息的强大模型之一）使用了约 4000 万个 token 的批大小。**在碰到带宽上限之前，这可以让我们扩展到大约 47,000 块芯片，也就是约 5 个 TPUv5 pod。**

对于 LLaMA-3 70B，其训练总计约使用 `6.3e24 (15e12 * 70e9 * 6)` FLOPs。我们可以把 1600 万个 token 的一批数据拆分到约 `16e6 / (2550 / 3) = 18,823` 块芯片（大约 2 个、每个含 8960 块芯片的 pod）上，每块芯片具有 `4.59e14` FLOPs，并以 50% 峰值 FLOPs 利用率（通常称为 MFU）运行，**约 17 天即可完成训练**。还不错！不过我们继续看看还能怎样做得更好。

**关于临界批大小的说明**：有些反直觉的是，在芯片数量固定时，随着总批大小减小，我们受到通信瓶颈的影响反而会更严重！数据并行和 FSDP 允许我们扩展到任意多的芯片，前提是可以不断增大批大小。然而在实践中，随着批大小增大，由于梯度几乎不再含噪声，训练收益往往会递减。有时还会出现训练不稳定。因此，在“无限算力范围”内寻找最优分片方案时，通常先确定一个由扩展定律决定的固定批大小，以及一个已知的（很大的）芯片数量，再尝试找到一种分区方式，使这个小批大小也能放到如此多的芯片上。

<span id="tensor-parallelism"></span>

### 张量并行

**写法：** $\text{In}[B, D_Y] \cdot_D W_\text{in}[D, F_Y] \cdot_F W_\text{out}[F_Y, D] \rightarrow \text{Out}[B, D_Y]$（我们使用 $Y$，以便稍后与 FSDP 组合）

在完全分片的数据并行 AllReduce 中，我们会在芯片之间移动权重。也可以对模型的前馈维度进行分片，并在层内移动激活值——这称为“一维模型并行”或 Megatron 分片[[megatron]](../#ref-megatron)。这样可以解锁更小的高效每 pod 批大小。下图展示了以这种方式对单个矩阵进行分片的例子：

![图：基本张量并行示例。由于只沿 Y 对激活值进行分片（不同于 FSDP 中沿 X 分片），因此我们沿 X 复制激活值。用我们的标准写法表示，就是 A[B, D_Y] * B[D, F_Y] -> C[B, F_Y]。因为只沿其中一个收缩维度分片，所以通常要在矩阵乘法前对激活值 A 执行 AllGather。](/images/scaling-book/img/model-parallelism.png)

如前所述，<strong>In\[B, D<sub>Y</sub>\] \*<sub>D</sub> W<sub>in</sub>\[D, F<sub>Y</sub>\] \*<sub>F</sub> W<sub>out</sub>\[F<sub>Y</sub>, D\] \-\> Out\[B, D<sub>Y</sub>\] 意味着我们必须在第一次矩阵乘法之前收集激活值。如果激活值小于权重，这会比 ZeRO 分片更便宜。</strong>通常只有在加入一定程度的 ZeRO 分片后，这一点才成立（ZeRO 分片会减小收集操作的大小）。这也是我们往往混合使用 ZeRO 分片和张量并行的原因之一。

<details>
<summary>下面是张量并行算法！ </summary>


<div markdown=1 class="algorithm">

**张量并行：**

<strong>前向传播：</strong>需要计算 Loss[B]

1.  In[B, D] = **AllGather**(In[B, D<sub>Y</sub>])（*在关键路径上*）
2.  Tmp[B, F<sub>Y</sub>] = In[B, D] \*<sub>D</sub> W<sub>in</sub>[D, F<sub>Y</sub>]（*未沿收缩维度分片，因此不需要通信*）
3.  Out[B, D] {U<sub>Y</sub>} = Tmp[B, F<sub>Y</sub>] \*<sub>F</sub> W<sub>out</sub>[F<sub>Y</sub>, D]
4.  Out[B, D<sub>Y</sub>] = **ReduceScatter**(Out[B, D] {U<sub>Y</sub>})（*在关键路径上*）
5.  Loss[B] = ...

<strong>反向传播：</strong>需要计算 dW<sub>out</sub>[F<sub>Y</sub>, D]、dW<sub>in</sub>[D, F<sub>Y</sub>]

1.  dOut[B, D<sub>Y</sub>] = ...
2.  dOut[B, D] = **AllGather**(dOut[B, D<sub>Y</sub>])（*在关键路径上*）
3.  dW<sub>out</sub>[F<sub>Y</sub>, D] = Tmp[B, F<sub>Y</sub>] \*<sub>B</sub> dOut[B, D]
4.  dTmp[B, F<sub>Y</sub>] = dOut[B, D] \*<sub>D</sub> W<sub>out</sub>[F<sub>Y</sub>, D]（*可在此处丢弃 dOut[B, D]*）
5.  In[B, D] = **AllGather**(In[B, D<sub>Y</sub>])（*与前向传播中的 (1) 共享时可以跳过*）
6.  dW<sub>in</sub>[D, F<sub>Y</sub>] = In[B, D] \*<sub>B</sub> dTmp[B, F<sub>Y</sub>]
7.  dIn[B, D] {U<sub>Y</sub>} = dTmp[B, F<sub>Y</sub>] \*<sub>F</sub> W<sub>in</sub>[D, F<sub>Y</sub>]（*前面各层需要*）
8.  dIn[B, D<sub>Y</sub>] = **ReduceScatter**(dIn[B, D] {U<sub>Y</sub>})（*在关键路径上*）

</div>

</details>

张量并行的一个优点，是它能与 Transformer 前向传播中的两个矩阵很好地配合。朴素做法会在两个矩阵之后分别执行一次 AllReduce。但这里我们先执行 **In[B, D<sub>Y</sub>] \* W<sub>in</sub>[D, F<sub>Y</sub>] -> Tmp[B, F<sub>Y</sub>]**，再执行 **Tmp[B, F<sub>Y</sub>] \* W<sub>out</sub>[F<sub>Y</sub>, D] -> Out[B, D<sub>Y</sub>]**。这意味着，我们在开始时对 **In** 执行 AllGather，并在结束时对 **Out** 执行 ReduceScatter，而不是执行一次 AllReduce。

**成本有多高？** 我们只对前向传播建模——反向传播只是这里每项操作的转置。在一维张量并行中，我们在第一次矩阵乘法前对激活值执行 AllGather，在第二次矩阵乘法后对其执行 ReduceScatter，每次发送两个字节（bf16）。下面来算算通信何时会成为瓶颈。

$$
\begin{align}
T_\text{math} & = \frac{4 \cdot B \cdot D \cdot F}{Y \cdot C} \\
T_\text{comms} & =
\frac{2 \cdot 2 \cdot (B \cdot D)}{W_\text{ici}}\\
\textnormal{T} & \approx \max \left(\frac{4 \cdot B \cdot D \cdot F}{Y \cdot C}, \frac{2 \cdot 2 \cdot (B \cdot D)}{W_\text{ici}}\right)
\end{align}
$$

注意，我们希望计算成本大于通信成本，因此得到：

$$
\begin{align}
\frac{4 \cdot B \cdot D \cdot F}{Y \cdot C} > \frac{2 \cdot 2 \cdot (B \cdot D)}{W_\text{ici}}
\end{align}
$$

$$
\begin{align}
\frac{F}{Y \cdot C} > \frac{1}{W_\text{ici}}
\end{align}
$$

$$
\begin{align}
F > Y \cdot \frac{C}{W_\text{ici}}
\end{align}
$$

因此，例如对于 TPUv5p，bf16 下 $C / W_{ici} = 2550$，所以张量并行最多只能做到 $Y < F / 2550$。如果有多个 ICI 轴，$T_\text{comms}$ 会缩小 $M_Y$ 倍，因此得到 $Y < M_Y \cdot F / 2550$。

**要点**：当 $Y > M_Y \cdot F / 2550$ 时，张量并行会变为通信受限。对大多数模型而言，这大约是 8 路到 16 路张量并行。

**请注意，这与计算精度无关**，因为例如在 TPUv5p 上使用 int8 时，$C_\text{int8} / W_{ici}$ 是 $5100$ 而不是 $2550$，但通信量也减半，因此两个 2 倍因子相互抵消。

**让我们看几个例子：**

* 对于 TPUv5p 上的 LLaMA 3-70B，$D = 8192,$、$F \approx 30,000$，我们可以轻松进行 8 路张量并行，但 16 路张量并行会受到通信限制。8 路模型分片所需的 F 为 20k。

* 对于 Gemma 7B，$F \approx 50k$，所以 19 路张量并行时会变为通信受限。这意味着 16 路张量并行很可能仍能获得良好性能。

<span id="combining-fsdp-and-tensor-parallelism"></span>

### 组合 FSDP 与张量并行

**写法：** $\text{In}[B_X, D_Y] \cdot_D W_\text{in}[D_X, F_Y] \cdot_F W_\text{out}[F_Y, D_X] \rightarrow \text{Out}[B_X, D_Y]$

FSDP 和张量并行的妙处在于二者可以结合。通过同时沿两个轴对 **W<sub>in</sub>** 和 **W<sub>out</sub>** 分片，我们既节省内存，也节省计算。由于沿 X 对 B 分片，模型并行 AllGather 的大小会减小；由于沿 Y 对 F 分片，FSDP 的通信开销也会降低。这意味着，二者组合后可以达到比上面更低的有效批大小。

![图：组合 FSDP 与张量并行的示意图。与其他情况不同，这里没有重复的模型参数。](/images/scaling-book/img/mixed-fsdp-model-parallelism.png)

<details>
<summary>下面是混合 FSDP + 张量并行的完整算法。虽然通信操作很多，但由于激活值按批分片、权重经过更充分的张量分片，所有 AllGather 和 ReduceScatter 都更小！ </summary>


<div markdown=1 class="algorithm">

<strong>前向传播：</strong>需要计算 Loss[B]

1.  In[B<sub>X</sub>, D] = **AllGather**<sub>Y</sub>(In[B<sub>X</sub>, D<sub>Y</sub>])（*在关键路径上*）
2.  W<sub>in</sub>[D, F<sub>Y</sub>] = **AllGather**<sub>X</sub>(W<sub>in</sub>[D<sub>X</sub>, F<sub>Y</sub>])（*可以提前执行*）
3.  Tmp[B<sub>X</sub>, F<sub>Y</sub>] = In[B<sub>X</sub>, D] \*<sub>D</sub> W<sub>in</sub>[D, F<sub>Y</sub>]
4.  W<sub>out</sub>[F<sub>Y</sub>, D] = **AllGather**<sub>X</sub>(W<sub>out</sub>[F<sub>Y</sub>, D<sub>X</sub>])（*可以提前执行*）
5.  Out[B<sub>X</sub>, D] {U<sub>Y</sub>} = Tmp[B<sub>X</sub>, F<sub>Y</sub>] \*<sub>F</sub> W<sub>out</sub>[F<sub>Y</sub>, D]
6.  Out[B<sub>X</sub>, D<sub>Y</sub>] = **ReduceScatter**<sub>Y</sub>(Out[B<sub>X</sub>, D] {U<sub>Y</sub>})（*在关键路径上*）
7.  Loss[B<sub>X</sub>] = ...

<strong>反向传播：</strong>需要计算 dW<sub>out</sub>[F<sub>Y</sub>, D<sub>X</sub>]、dW<sub>in</sub>[D<sub>X</sub>, F<sub>Y</sub>]

1.  dOut[B<sub>X</sub>, D<sub>Y</sub>] = ...
2.  dOut[B<sub>X</sub>, D] = **AllGather**<sub>Y</sub>(dOut[B<sub>X</sub>, D<sub>Y</sub>])（*在关键路径上*）
3.  dW<sub>out</sub>[F<sub>Y</sub>, D] {U<sub>X</sub>} = Tmp[B<sub>X</sub>, F<sub>Y</sub>] \*<sub>B</sub> dOut[B<sub>X</sub>, D]
4.  dW<sub>out</sub>[F<sub>Y</sub>, D<sub>X</sub>] = **ReduceScatter**<sub>X</sub>(dW<sub>out</sub>[F<sub>Y</sub>, D] {U<sub>X</sub>})
5.  W<sub>out</sub>[F<sub>Y</sub>, D] = **AllGather**<sub>X</sub>(W<sub>out</sub>[F<sub>Y</sub>, D<sub>X</sub>])（*可以提前执行*）
6.  dTmp[B<sub>X</sub>, F<sub>Y</sub>] = dOut[B<sub>X</sub>, D] \*<sub>D</sub> W<sub>out</sub>[F<sub>Y</sub>, D]（*可在此处丢弃 dOut[B, D]*）
7. In[B<sub>X</sub>, D] = **AllGather**<sub>Y</sub>(In[B<sub>X</sub>, D<sub>Y</sub>])（*不在关键路径上，而且可以与前一层的 (2) 共享*）
8.  dW<sub>in</sub>[D, F<sub>Y</sub>] {U<sub>X</sub>} = In[B<sub>X</sub>, D] \*<sub>B</sub> dTmp[B<sub>X</sub>, F<sub>Y</sub>]
9.  dW<sub>in</sub>[D<sub>X</sub>, F<sub>Y</sub>] = **ReduceScatter**<sub>X</sub>(dW<sub>in</sub>[D, F<sub>Y</sub>] {U<sub>X</sub>})
10. W<sub>in</sub>[D, F<sub>Y</sub>] = **AllGather**<sub>X</sub>(W<sub>in</sub>[D<sub>X</sub>, F<sub>Y</sub>])（*可以提前执行*）
11. dIn[B<sub>X</sub>, D] {U<sub>Y</sub>} = dTmp[B<sub>X</sub>, F<sub>Y</sub>] \*<sub>F</sub> W<sub>in</sub>[D, F<sub>Y</sub>]（*前面各层需要*）
12. dIn[B<sub>X</sub>, D<sub>Y</sub>] = **ReduceScatter**<sub>Y</sub>(dIn[B<sub>X</sub>, D] {U<sub>Y</sub>})（*在关键路径上*）

</div>

</details>

**FSDP 和 TP 应该如何搭配？** 有一句简单却关键的格言：FSDP 移动权重，张量并行移动激活值。这意味着随着批大小缩小（尤其是随着数据并行程度提高），张量并行会变得更便宜，因为每个分片上的激活值更小。

* 张量并行执行 $\mathbf{AllGather}_Y([B_X, D_Y])$，其大小随 $X$ 增大而缩小。
* FSDP 执行 $\mathbf{AllGather}_X([D_X, F_Y])$，其大小随 $Y$ 增大而缩小。

因此，将二者结合起来，可以进一步降低每个副本的最小批大小。我们可以用与上文相同的方式，计算 FSDP 与 TP 的最佳配比：

令 $X$ 为分配给 FSDP 的芯片数量，$Y$ 为分配给张量并行的芯片数量。令 $N$ 为切片中的芯片总数，其中 $N=XY$。令 $M_X$ 和 $M_Y$ 分别为执行 FSDP 和 TP 所跨越的网格轴数量（二者之和应当约为 3）。由于前向传播的每 FLOP 通信量最多，我们将只对前向传播建模。把上述算法中的通信量相加，得到

$$
T_\text{FSDP comms}(B, X, Y) = \frac{2\cdot 2\cdot D \cdot F}{Y \cdot W_\text{ici} \cdot M_X}
$$

$$
T_\text{TP comms}(B, X, Y) = \frac{2 \cdot 2 \cdot B \cdot D}{X \cdot W_\text{ici} \cdot M_Y}
$$

同样，总 FLOPs 时间为

$$
T_\text{math} = \frac{2\cdot 2 \cdot B \cdot D \cdot F}{N \cdot C}.
$$

为了简化分析，我们作出两个假设：第一，允许 $X$ 和 $Y$ 取非整数值（只要它们为正且满足 $XY=N$）；第二，假设可以让 $X$ 轴和 $Y$ 轴上的通信彼此完全重叠。在第二个假设下，总通信时间为

$$
T_\text{comms} = \max\left(T_\text{FSDP comms}, T_\text{TP comms}\right)
$$

在讨论什么条件下计算受限之前，我们先找出能让总通信量最小的最优 $X$ 和 $Y$。由于 FLOPs 与 $X$ 和 $Y$ 无关，最优设置就是能让通信量最小的设置。为此，我们把上面的 $T_\text{comms}$ 改写成 $X$ 和 $N$（它保持固定，因为它是系统中的芯片数量）的函数，而不是 $X$ 和 $Y$ 的函数：

$$
T_\text{comms} (X) = \frac{4D}{W_\text{ici}} \max\left(\frac{F \cdot X}{N \cdot M_X}, \frac{B}{X \cdot M_Y}\right)
$$

由于 $T_\text{FSDP comms}$ 随 $X$ 单调递增，而 $T_\text{TP comms}$ 随 $X$ 单调递减，因此当 $T_\text{FSDP comms} = T_\text{TP comms}$ 时，二者的最大值必定达到最小。该等式成立于

$$
\begin{align*}
\frac{FX_{opt}}{M_X} = \frac{BN}{X_{opt} M_Y} \rightarrow \\
X_{opt} = \sqrt{\frac{B}{F} \frac{M_X}{M_Y} N}
\end{align*}
$$

这太有用了！它告诉我们，对于给定的 $B$、$F$ 和 $N$，多大程度的 FSDP 才是最优的。我们来感受一下规模。代入现实数值，即 $N = 64$（对应 4x4x4 芯片阵列）、$B=48,000$、$F=32768$，得到大约 $X\approx 13.9$。因此我们会选择 $X$ 为 16、$Y$ 为 4，与计算出的最优值很接近。

<strong>要点：</strong>一般而言，在训练期间，FSDP 的最优程度为 $X_{opt} = \sqrt{\frac{B}{F} \frac{M_X}{M_Y} N}$。

现在回到我们针对所有并行策略都会提出的问题：**在什么条件下，我们会处于计算受限状态？** 由于可以重叠 FLOPs 和通信，因此当下式成立时，我们是计算受限的：

$$
\max\left(T_\text{FSDP comms}, T_\text{TP comms}\right) < T_\text{math}
$$

令 ICI 算术强度 $\alpha \equiv C / W_\text{ici}$，即可化简为：

$$
\max\left(\frac{F}{Y \cdot M_X}, \frac{B}{X \cdot M_Y}\right) < \frac{B \cdot F}{N \cdot \alpha}
$$

由于我们已经算出 $X_{opt}$ 会使左侧最大值的两项相等，因此只需将它代入任意一项（注意 $Y_{opt} = N/X_{opt}$），即

$$
\frac{F}{N \cdot W_\text{ici} \cdot M_X} \sqrt{\frac{B}{F} \frac{M_X}{M_Y} N} < \frac{B \cdot F}{N \cdot C}
$$

进一步化简，得到

$$
 \sqrt{\frac{B\cdot F}{M_X \cdot M_Y \cdot N}} < \frac{B \cdot F}{N \cdot \alpha},
$$

其中左侧与通信时间成正比，右侧与计算时间成正比。请注意，计算时间随批大小线性增长（无论采用哪种并行方式都是如此），而通信时间随批大小的平方根增长。因此，计算时间与通信时间之比同样随批大小的平方根增长：

$$
 \frac{T_\text{math}}{T_\text{comms}} = \frac{\sqrt{BF}\sqrt{M_X M_Y}}{\alpha \sqrt{N}}.
$$

为了确保该比值大于 1、使我们处于计算受限状态，需要满足

$$
 \frac{B}{N} > \frac{\alpha^2}{M_X M_Y F}
$$

为了得到近似数值，再次代入 $F=32,768$、$\alpha=2550$ 和 $M_X M_Y=2$（对于三维网格必然如此），得到大约 $B/N > 99$。相比纯数据并行（或 FSDP），这大致带来 8 倍优势；对于后者，假设使用三维网格，我们算得 $B/N$ 必须大于约 $850$ 才能保持计算受限。

<strong>要点：</strong>将张量并行与 FSDP 结合，可以把 $B/N$ 降至 $2550^2 / 2F$。这样，每芯片的批大小低至 100 也能处理，相比只使用 FSDP 能达到的批大小约小 8 倍。

下图绘制了混合 FSDP + TP 的 FLOPs 与通信时间之比，并在一个代表性的 4x4x4 芯片阵列上，将其与纯张量并行（TP）和纯数据并行（FSDP）进行比较。对于很大的批大小，纯 FSDP 并行占优；但在批大小除以芯片数量约为 100 到 850 的范围内，只有混合 FSDP + TP 策略才能保持计算受限。

![图：在 TPUv5p 4x4x4 切片上，F=30k 时最优混合 FSDP/TP 的 FLOPs 与通信时间之比。与预期一致，张量并行与批大小的比值固定；理想混合 FSDP + TP 按 $\sqrt{B}$ 缩放，而 FSDP 按 $B$ 缩放。不过，在中等批大小范围内，只有 FSDP + TP 能使比值大于 1。](/images/scaling-book/img/mixed-fsdp-comms-2.png)

下面是 TPU v5p 16x16x16 的另一个示例，展示了不同分片方案下 FLOPs 和通信时间随批大小的变化。

![图：不同并行方案的通信耗时。黑色虚线是矩阵乘法 FLOPs 的耗时，因此任何位于这条线之上的曲线都受通信限制。请注意，所有策略在批大小低于 6e5 时都会变为通信受限，这与预期的 4096 * 2550^2 / (2 * 8192 * 4) = 4e5 相符。](/images/scaling-book/img/math-comms-time.png)

黑色曲线表示花在模型 FLOPs 上的时间，因此，凡是黑线低于所有通信成本的批大小，都严格受通信限制。你会注意到，黑色曲线与绿色曲线约在 `4e5` 处相交，与预测一致。

下面这个交互式动画可供你试验，它展示了不同批大小对应的总计算时间和通信时间：

<div class="scaling-book-plotly" style="position: relative; width: 100%; aspect-ratio: 16 / 9;">
  <iframe src="../../images/scaling-book/plotly/training-roofline.html" title="训练 Roofline 交互图" loading="lazy" scrolling="no" style="position: absolute; inset: 0; width: 100%; height: 100%; border: 0;"></iframe>
</div>

你会发现，它总体上与上述结果一致（最小值约在 FSDP=256、TP=16），由于每种方案所用轴数略有差别，可能会有少许波动。

<span id="pipelining"></span>

### 流水线并行

你或许已经注意到，前几节完全没有讨论流水线并行。流水线并行是 GPU 并行中的主流策略，但在 TPU 上的重要性略低。简而言之，流水线训练把模型的层拆分到多台设备上，并在前向传播和反向传播期间，在各流水线阶段之间传递激活值。算法大致如下：

1. 在 TPU 0 上初始化数据，同时沿层维度对权重分片（若流水线并行与 FSDP、张量并行结合，则为 $W_\text{in}[L_Z, D_X, F_Y]$）。
2. 在 TPU 0 上执行第一层，然后把得到的激活值复制到 TPU 1；重复此过程，直到最后一个 TPU。
3. 计算损失函数及其导数 $\partial L / \partial x_L$。
4. 对最后一个流水线阶段计算导数 $\partial L / \partial W_L$ 和 $\partial L / \partial x_{L-1}$，然后将 $\partial L / \partial x_{L-1}$ 复制到前一个流水线阶段；重复此过程，直到回到 TPU 0。

<details>
<summary>下面是一些（可以运行的）Python 伪代码 </summary>


这段伪代码应该能在 Cloud TPU VM 上运行。它虽然效率不高，也不够贴近真实实现，但能帮助你了解数据如何跨设备传播。

```python
batch_size = 32
d_model = 128
d_ff = 4 * d_model

num_layers = len(jax.devices())

key = jax.random.PRNGKey(0)

# Pretend each layer is just a single matmul.
x = jax.random.normal(key, (batch_size, d_model))
weights = jax.random.normal(key, (num_layers, d_model, d_model))

def layer_fn(x, weight):
  return x @ weight

# Assume we have num_layers == num_pipeline_stages
intermediates = [x]
for i in range(num_layers):
  x = layer_fn(x, weights[i])
  intermediates.append(x)

  if i != num_layers - 1:
    x = jax.device_put(x, jax.devices()[i+1])

def loss_fn(batch):
  return jnp.mean(batch ** 2)  # make up some fake loss function

loss, dx = jax.value_and_grad(loss_fn)(x)

for i in range(num_layers - 1, -1, -1):
  _, f_vjp = jax.vjp(layer_fn, intermediates[i], weights[i])
  dx, dw = f_vjp(dx)  # compute the jvp dx @ J(L)(x[i], W[i])
  weights[i] = weights[i] - 0.01 * dw  # update our weights

  if i != 0:
    dx = jax.device_put(dx, jax.devices()[i-1])
```

</details>

**为什么这是个好主意？** 流水线并行有许多优点：流水线阶段之间的通信成本很低，因此即使互连带宽不高，也能训练很大的模型。由于 GPU 不像 TPU 那样通过 ICI 密集连接，这一点在 GPU 上往往非常有用。

**为什么这很困难／烦人？** 你或许已经在上面的伪代码中注意到，TPU 0 几乎一直处于空闲状态！它只在流水线的第一步和最后一步执行工作。这段空闲期称为流水线气泡，处理起来非常烦人。通常我们首先尝试用微批处理来缓解这一问题：让多个小批依次通过流水线，使 TPU 0 至少在总步时的更大一部分时间内保持忙碌。

第二种方法，是仔细地重叠前向矩阵乘法 $W_i @ x_i$、反向 $dx$ 矩阵乘法 $W_i @ \partial L / \partial x_{i+1}$，以及 $dW$ 矩阵乘法 $\partial L / \partial x_{i+1} @ x_i$。由于每一项都需要一些 FLOPs，我们可以重叠它们，从而完全隐藏气泡。下面是近期 DeepSeek v3 论文[[DeepSeek3]](../#ref-DeepSeek3)中的一幅图，展示了他们的“无气泡”流水线调度：

![图：DeepSeek v3 的流水线调度。橙色表示前向矩阵乘法，绿色表示 dL/dx 矩阵乘法，蓝色表示 dL/dW 矩阵乘法。优先执行反向 dL/dx 乘法可以避免 FLOPs 被“搁浅”。](/images/scaling-book/img/deepseek-pipeline.png)

来源：[DeepSeek-V3 Technical Report](https://arxiv.org/pdf/2412.19437)。

由于这对 TPU 没那么关键（TPU 拥有更大的互连 pod），我们不会在此深入探讨，但理解流水线并行的主要瓶颈是一个很好的练习。

<span id="scaling-across-pods"></span>

### 跨 Pod 扩展

最大的 TPU 切片是包含 8960 块芯片（以及 2240 台主机）的 TPU v5p SuperPod。如果要扩展到比这更大的规模，就必须跨越数据中心网络（Data-Center Networking，DCN）边界。每台 TPU 主机都配有一块或多块 NIC（网络接口卡），通过以太网将这台主机连接到其他 TPU v5p pod。正如 [TPU 一节](../02-tpus/#how-to-think-about-tpus)中提到的，每台主机大约拥有 200Gbps（25GB/s）的全双工 DCN 带宽，折算下来每个 TPU 约有 6.25GB/s 的全双工（出口）带宽。

通常，扩展到单个 pod 之外时，我们会在 ICI 域内执行某种形式的模型并行或 FSDP，再跨多个 pod 执行纯数据并行。令 $N$ 为要扩展到的 TPU 数量，$M$ 为每个 ICI 互连切片中的 TPU 数量。为了在 DCN 上执行 AllReduce，可以在这组 pod 之间进行环形归约，于是（在反向传播中）得到：

$$
T_\text{math} = \frac{2 \cdot 2 \cdot 2 \cdot BDF}{N \cdot C}
$$

$$
T_\text{comms} = \frac{2 \cdot 2 \cdot 2 \cdot DF}{M \cdot W_\text{dcn}}
$$

通信带宽随 $M$ 增长，因为与 ICI 不同，随着 ICI 域扩大并获得更多 NIC，总带宽也会增加。化简可知，当下式成立时，$T_\text{math} > T_\text{comms}$：

$$
\frac{B}{\text{slice}} > \frac{C}{W_\text{dcn}}
$$

对于 TPU v5p，$\frac{C}{W_\text{dcn}}$ 约为 `4.59e14 / 6.25e9 = 73,440`。这告诉我们，为了高效地跨 DCN 扩展，每个 ICI 域必须有一个最小批大小，才能将每个节点的数据发出。

**这会造成多大问题？** 举个具体例子，假设要在 TPU v5p 上以 200 万个 token 的批大小训练 LLaMA-3 70B。LLaMA-3 70B 的 $F\approx 30,000$。根据前面几节，我们知道：

* 张量并行最多可以做到 $Y = M_Y \cdot F / 2550 \approx 11 \cdot M_Y$。
* 只要 $B / N > 2550 / M_X$，就可以使用 FSDP。这意味着，如果以 BS=2M 训练，并使用 3 个数据并行轴，那么最多只能使用约 $\approx 2400$ 块芯片，大约是一个 TPU v5p pod 的四分之一。
* 组合 FSDP + 张量并行时，如果 $B / N < 2550^2 / (2 \cdot 30000) = 108$，我们会变为通信受限，因此这种组合可以扩展到约 18k 块芯片！但 TPU v5p pod 最大只有 8k 块芯片，超过这个规模就必须使用 DCN。

简而言之，对于 BS=1M，我们有一个不错的训练方案：大致使用 X（FSDP）= 1024、Y（TP）= 8；但对于 BS=2M，则需要使用 DCN。如上所述，DCN 的算术强度为 $\text{73,440}$，所以只需确保每个 ICI 域的批大小大于这个值。这对我们轻而易举，因为使用 2 个 pod 时，每个 pod 的 BS 为 1M，每个 TPU 的批大小为 111，这非常不错（也许有点逼近极限，但理论上可行）。

<strong>要点：</strong>只要每个 pod 的批大小至少为 73k 个 token，使用纯数据并行跨多个 TPU pod 扩展就相当直接。

<span id="takeaways-from-llm-training-on-tpus"></span>

## TPU 上 LLM 训练的要点

* 提高并行程度或减小批大小，都会减少每芯片执行的计算量，因此往往让我们更容易受通信限制。

* 在合理的上下文长度（约 32k）以内，我们可以把 Transformer 建模为一叠 MLP 块，并用每层两／三个主要矩阵乘法的分片方式来定义各种并行方案。

* 训练期间主要考虑 4 种并行方案，每种都有各自的带宽与计算要求（数据并行、FSDP、张量并行，以及混合 FSDP + 张量并行）。

| **策略**                                 | **说明**                                                                                                                                                                            |
| -------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **数据并行**                         | 激活值按批分片，其他一切均完整复制；在反向传播期间对梯度执行 AllReduce。                                                                      |
| **FSDP**                                     | 激活值、权重和优化器按批分片；权重在使用前即时收集，梯度执行 ReduceScatter。                                                               |
| **张量并行（亦称 Megatron、模型并行）** | 激活值沿 $d_\text{model}$ 分片，权重沿 $d_{ff}$ 分片；激活值在 W<sub>in</sub> 之前收集，结果在 W<sub>out</sub> 之后执行 ReduceScatter。 |
| **混合 FSDP + 张量并行**          | 上述两种方案的组合，其中 FSDP 收集经过模型分片的权重。                                                                                                                           |

下面是每种方法的“公式”：

$$
\small
\begin{array}{cc}
\text{Strategy} & \text{Formula}\\
\hline
\text{DP} & \text{In}[B_X, D] \cdot_D W_\text{in}[D, F] \cdot_F W_\text{out}[F, D] \rightarrow \text{Out}[B_X, D] \\
\text{FSDP} & \text{In}[B_X, D] \cdot_D W_\text{in}[D_X, F] \cdot_F W_\text{out}[F, D_X] \rightarrow \text{Out}[B_X, D] \\
\text{TP} & \text{In}[B, D_Y] \cdot_D W_\text{in}[D, F_Y] \cdot_F W_\text{out}[F_Y, D] \rightarrow \text{Out}[B, D_Y] \\
\text{TP + FSDP}  & \text{In}[B_X, D_Y] \cdot_D W_\text{in}[D_X, F_Y] \cdot_F W_\text{out}[F_Y, D_X] \rightarrow \text{Out}[B_X, D_Y] \\
\hline
\end{array}
$$

* 每种策略都有一个受网络／通信限制的临界点，取决于其每设备计算量和通信量。下面列出每层的计算与通信，假设 $X$ 用于 FSDP，$Y$ 用于张量并行。

$$
\small
\begin{array}{ccc}
\text{Strategy} & \text{Compute per layer} & \text{Comms per layer} \\
& \text{(ignoring gating einsum)} & \text{(bytes, forward + backward pass)}\\
\hline
\text{DP} & 4BDF/X + 8BDF/X & 0 + 8DF \\
\text{FSDP} & 4BDF/X + 8BDF/X & 4DF + 8DF \\
\text{TP} & 4BDF/Y + 8BDF/Y & 4BD + 4BD \\
\text{FSDP + TP} & 4BDF/(XY) + 8BDF/(XY) & (4BD/X + 4DF/Y) + (8BD/X + 8DF/Y) \\
\hline
\end{array}
$$

* 纯数据并行很少有用，因为模型及其优化器状态使用的字节数是参数量的 10 倍。这意味着内存中通常只能容纳几十亿个参数。

* 当 $\text{batch size per shard} < C / W$（即网络的算术强度）时，数据并行和 FSDP 会变为通信受限。ICI 的这个值为 2,550，DCN 约为 71,000。增加并行轴可提高这一数值。

* 当 $\lvert Y\rvert > F / 2550$ 时，张量并行会变为通信受限。<strong>对大多数模型而言，大约是 8 路到 16 路。</strong>这与批大小无关。

* 混合 FSDP + 张量并行可以把批大小降至 $2550^2 / 2F \approx 100$。这个值低得惊人。

* 跨 pod 的数据并行要求每个 pod 的最小批大小约为 71,000，低于这个值就会受到 DCN 限制。

* 总体而言，如果批大小很大或模型很小，事情就很简单。你可以使用数据并行，或使用 FSDP + 跨 DCN 数据并行。中间范围才是事情变得有趣的地方。

<span id="some-problems-to-work"></span>

## 一些练习题

本节以 LLaMA-2 13B 作为基础模型。下面是模型详情：

| 超参数 | 值  |
| ---------- | ------ |
| L          | 40     |
| D          | 5,120  |
| F          | 13824  |
| N          | 40     |
| K          | 40     |
| H          | 128    |
| V          | 32,000 |

LLaMA-2 使用彼此独立的嵌入矩阵与输出矩阵，并采用门控 MLP 块。

<strong>问题 1：</strong>LLaMA-2 13B 有多少个参数（我知道这个问题有点傻，但请实际算一遍）？*请注意，正如 [Transformer 数学](../04-transformers/#all-the-transformer-math-you-need-to-know)中所述，LLaMA-3 有 3 个大型 FFW 矩阵：两个升维投影和一个降维投影。本节忽略了两个“门控”einsum 矩阵，但它们的行为与本节的 W<sub>in</sub> 相同。*

<details>
<summary>点击此处查看答案。 </summary>


* FFW 参数：$3LDF$ = `8.5e9`
* 注意力参数：$4DNHL$ = `4.2e9`
* 词表参数：$2VD$ = `0.33e9`
* 总计：`8.5e9 + 4.2e9 + 0.33e9 = 13.0e9`，与预期相符！

</details>

<strong>问题 2：</strong>假设使用 BS=16M 个 token 和 Adam 进行训练。暂时忽略并行，模型的参数、优化器状态和激活值总共会占用多少内存？*假设参数存储为 bf16、优化器状态存储为 fp32，并且每层对激活值设置三次检查点（在三个大型矩阵乘法之后）。*

<details>
<summary>点击此处查看答案。 </summary>


参数（bf16）和两个优化器状态（fp32，即一阶矩与二阶矩累加器）使用的总内存为 `(2 + 4 + 4) * 13e9 ~ 130GB`。前两次矩阵乘法后的激活值形状为 $BF$，最后一次之后为 $BD$（见上面的 Transformer 示意图），因此 bf16 的总内存为 $2 \cdot L \cdot (BD + 2 * BF) = 2LB \cdot (D + 2F)$，即 `2 * 40 * 16e6 * 5,120 * (1 + 2 * 2.7) ~ 4.2e13 = 42TB`，因为 `B=16e6`。所有其他激活值基本可以忽略。

</details>

<strong>问题 3：</strong>假设我们要在 TPUv5p 16x16x16 切片上，以 32k 序列长度和总计 300 万个 token 的批大小进行训练。与上面一样，假设使用 bfloat16 权重和 float32 优化器。

1. 能否使用纯数据并行？为什么？
2. 能否使用纯 FSDP？为什么？使用纯 FSDP 时，每台设备上会使用多少内存（假设只在 3 个大型 FFW 矩阵之后设置梯度检查点）？
3. 能否使用混合 FSDP + 张量并行？为什么？如果可以，$X$ 和 $Y$ 应分别取多少？每台设备上会存储多少内存？如果只使用 Roofline FLOPs 估算、忽略注意力，并假设 MFU 为 40%，每个训练步需要多长时间？

<details>
<summary>点击此处查看答案。 </summary>


首先写下一些数值。序列长度为 32k、批大小为 300 万时，序列批大小为 96。TPU v5p 16x16x16 切片拥有 `393TB` HBM。

1. 不能使用纯数据并行，因为它会在每块芯片上复制参数和优化器状态，而它们已经约为 130GB（来自问题 2），超过了每芯片可用的 HBM（96GB）。

2. 先只看内存。把问题 2 中的 BS=16M 换成 3M，得到总计 `~7.86e12` 字节的检查点激活值；加上 1.3e11 字节的优化器状态后，几乎正好是 8e12 = 8TB。TPUv5p 切片总共拥有 `393TB` HBM，因此远低于 HBM 上限。接下来看看受通信限制还是计算限制。使用 4096 块芯片和 3 个并行轴时，最小批大小为 `850 * 4096 = 3.48M` 个 token。这略高于 300 万的批大小。所以我们实际上受通信限制，这令人难过。因而总体答案是：**不行，不能只使用 FSDP**。

3. 现在我们知道，主要问题在于通信受限，因此来代入一些数值。首先，由上文可知，在这里，混合 FSDP + 张量并行的每芯片批大小必须高于 $2550^2 / 2F = 235$。这意味着理论上可行！下面算算二者各用多少。

规则为 $X_{opt} = \sqrt{(B / F) \cdot (M_X / M_Y) \cdot N}$，所以这里有 `sqrt(3e6 * 2 * 4096 / 13824) = 1333`，也就是说，大约使用 1024 路 DP 和 4 路 TP。每个 TPU 的内存用量与 (2) 相同，步时就是 `6 * 3e6 * 13e9 / (4096 * 4.6e14 * 0.4) = 300ms`。

</details>

<span id="thats-it-for-part-5-for-part-6-which-applies-this-content-to-real-llama-models-click-here"></span>

### 第 5 部分到此结束！第 6 部分会把这些内容应用到真实的 LLaMA 模型，请[点击此处](../06-llama3-training/#training-llama-3-on-tpus)！

<span id="appendix"></span>

## 附录

<span id="appendix-a-deriving-the-backward-pass-comms"></span>

### 附录 A：推导反向传播的通信

上面，我们把 Transformer 层的前向传播简化为 Out[B, D] = In[B, D] *<sub>D</sub> W<sub>in</sub>[D, F] *<sub>F</sub> W<sub>out</sub>[F, D]。如何推导反向传播所需的通信？

这可以相当自然地由上一节针对单次矩阵乘法 **Y = X * A** 给出的规则推出：

$$
\frac{dL}{dA} = \frac{dL}{dY}\frac{dY}{dA} = X^T \left(\frac{dL}{dY}\right)
$$

$$
\frac{dL}{dX} = \frac{dL}{dY}\frac{dY}{dX} = \left(\frac{dL}{dY}\right) A^T
$$

据此可得以下公式（用 Tmp[B, F] 表示 In[B, D] * W<sub>in</sub>[D, F]）：

<div markdown=1 class="algorithm">

1. dW<sub>out</sub>[F, D] = Tmp[B, F] *<sub>B</sub> dOut[B, D]
2. dTmp[B, F] = dOut[B, D] *<sub>D</sub> W<sub>out</sub>[F, D]
3. dW<sub>in</sub>[D, F] = In[B, D] *<sub>B</sub> dTmp[B, F]
4. dIn[B, D] = dTmp[B, F] *<sub>F</sub> W<sub>in</sub>[D, F]

</div>

请注意，这些公式只是数学陈述，没有提及分片。反向传播的任务就是计算这四个量。因此，为了确定所需的通信，我们只需取出上述四个方程中所有要参与矩阵乘法的量（Tmp、dOut、W<sub>out</sub>、W<sub>in</sub>）的分片方式——它们由并行方案规定——再使用分片矩阵乘法规则，确定必须执行哪些通信。请注意，dOut 的分片方式与 Out 相同。

[^ch5-1]: 我们将重点关注通信上限——因为内存容量约束虽然很重要，但在预训练期间采用重计算（激活值检查点）并使用大量芯片时，通常不会成为限制因素。这里也不讨论 MoE 的专家并行——它会显著扩大设计空间；我们只考虑稠密 Transformer 的基本情况。
[^ch5-2]: Adam 存储参数、一阶累加器和二阶累加器。由于参数采用 bfloat16、优化器状态采用 float32，因此每个参数占用 `2 + 8 = 10` 字节。
[^ch5-3]: 请注意，这里不包括梯度检查点，所以实际上并不实用。这是在批中只有 1 个 token 时的绝对下限。
[^ch5-4]: 我们假设这种分区是在 ICI 网格上完成的，因此相关网络带宽是 $W_\text{ici}$
[^ch5-5]: 严格来说，FSDP 在前向传播中增加了纯 DP 所没有的通信，但它与反向传播的比例相同，因此不应影响通信 Roofline。这里的关键是，ZeRO-3 把反向传播中的一次 AllReduce 变成一次 AllGather 和一次 ReduceScatter，而二者的总通信量相同。
