---
title: "你需要掌握的全部 Transformer 数学"
description: "本章将快速回顾 Transformer 架构，重点介绍如何计算 FLOPs、字节数以及其他值得关注的量。"
chapter: 4
order: 4
part: 2
partTitle: "Transformer"
sourcePath: "transformers.md"
sourceCommit: "44109cacac9c5a9809a81c68ae4d45d7d2632ea6"
---

<span id="all-the-transformer-math-you-need-to-know"></span>

# 你需要掌握的全部 Transformer 数学

<span id="counting-dots"></span>

## 数点

先从具有以下形状的向量 $x$、$y$ 和矩阵 $A$、$B$ 开始：

$$
\def \red#1{\textcolor{red}{#1}}
\def \green#1{\textcolor{green}{#1}}
\def \blue#1{\textcolor{blue}{#1}}
\def \purple#1{\textcolor{purple}{#1}}
\def \orange#1{\textcolor{orange}{#1}}
\def \gray#1{\textcolor{gray}{#1}}

\begin{array}{cc}
\textrm{array}  & \textrm{shape} \\ \hline
x               & \textrm{[P]}   \\
y               & \textrm{[P]}   \\
A               & \textrm{[N P]} \\
B               & \textrm{[P M]} \\
\hline
\end{array}
$$

- 点积 $x \cdot y$ 需要 $P$ 次*加法*和*乘法*，总计 $2P$ 次浮点运算。
- 矩阵—向量乘积 $Ax$ 会执行 $N$ 次点积，分别沿 $A$ 的各行进行，共计 $2NP$ FLOP。
- 矩阵—矩阵乘积 $AB$ 会执行 $M$ 次矩阵—向量乘法，分别对应 $B$ 的各列，总计 $2NPM$ FLOP。
- 一般而言，如果有两个高维数组 $C$ 和 $D$，其中一些维度是收缩维度，另一些是批处理维度（例如 $C[\blue{GH}IJ\red{KL}], D[\blue{GH}MN\red{KL}]$），那么这次收缩的 FLOP 成本就是 $C$ 与 $D$ 所有维度之积的两倍，其中批处理维度和收缩维度只计算一次（例如 $2\blue{GH}IJMN\red{KL}$）。请注意，只有同时出现在两个乘数中的维度才是批处理维度。（另请注意，如果没有收缩维度、这里只是逐元素乘积，那么因子 2 并不适用。）[^ch4-1]

$$
\begin{array}{ccc}
\textrm{Operation} & \textrm{FLOPs} & \textrm{Data} \\
\hline
x \cdot y  & 2P   & 2P      \\
A x        & 2NP  & NP + P  \\
AB         & 2NPM & NP + PM \\
[c_0,...,c_N] \cdot [d_0,...,d_N] &
2 \prod c_i \times \prod_{\substack{d_j \notin \blue{BATCH} \\ d_j \notin \red{CONTRACT}}} d_j
&
  \prod c_i + \prod d_j \\
\hline
\end{array}
$$

请记住：对于矩阵—矩阵乘法，*计算量*按三次方 $O(N^3)$ 扩展，而数据传输量只按二次方 $O(N^2)$ 扩展——这意味着，随着矩阵乘法规模增大，反而*更容易*达到计算饱和上限。这种情况极其罕见，也在很大程度上解释了为什么我们会采用以矩阵乘法为主的架构——它们很适合扩展！

![](/images/scaling-book/img/matmul-flops.gif)

<span id="forward-and-reverse-flops"></span>

### 前向与反向 FLOP

训练时，我们并不特别关心某次矩阵乘法的结果；真正关心的是它的导数。事实证明，计算这个导数的成本大约是只执行矩阵乘法本身的 3 倍。

假设 **B** 只是更大网络中的一个矩阵，**A** 是输入激活值，并且 **C = A B**，那么根据链式法则，损失 **L** 对 **B** 的导数为：

$$
\frac{\partial L}{\partial B} = \frac{\partial L}{\partial C}\frac{\partial C}{\partial B} = A^T \left(\frac{\partial L}{\partial C}\right)
$$

计算它需要 $2NPM$ FLOP（因为它沿 $N$ 维度收缩）。同样，损失对 **A** 的导数为

$$
\frac{\partial L}{\partial A} = \frac{\partial L}{\partial C}\frac{\partial C}{\partial A} = \left(\frac{\partial L}{\partial C}\right) B^T
$$

这同样需要 $2NPM$ FLOP，因为 **dL/dC** 是一个大小为 $[N, M]$ 的矩阵。虽然这个量不是对某个参数的导数，但它会用于计算网络前面各层的导数（例如，正如上面用 dL/dC 计算 dL/dB 一样）。

把这些成本相加，可以看到，**训练期间总共有 6NPM FLOP**，而推理期间是 2NPM：前向传播为 2NPM，反向传播为 4NPM。因为 PM 是矩阵中的参数数量，所以这就是著名的 Transformer 训练 FLOP 近似式 $6 * \text{num parameters} * \text{num tokens}$ 最简单的形式：每个 token 需要 $6 * \text{num parameters}$ FLOP。下面将给出更准确的推导。

<span id="transformer-accounting"></span>

## Transformer 账目核算

Transformer 就是未来。好吧，至少它们是现在。也许几年前，它们只是众多架构中的一种。但在今天，几乎值得了解这种架构的每一个细节。这里不会重新介绍该架构，不过可以参考[这篇博客](https://jalammar.github.io/illustrated-transformer/)和[最初的 Transformer 论文](https://arxiv.org/abs/1706.03762)。

下面是 Transformer 解码器架构的基本示意图：

![图：这张图自上而下展示了标准 Transformer 的一层。我们用单字母约定描述 Transformer 中数组的形状与布局，并继续用红色表示收缩维度、蓝色表示批处理维度。在每项运算中，左上方给出输入形状，右上方给出参数形状，下方则给出结果形状。例如，BTD 是门控 einsum 的输入形状，DF 是权重形状。](/images/scaling-book/img/transformer-diagram.png)

**注意［门控 einsum］：** 上图使用了“[门控 einsum](https://arxiv.org/abs/2002.05202)”[[glu]](../#ref-glu)，其中把上投影矩阵拆成两个矩阵（上图中的 $W_\text{In1}$ 和 $W_\text{In2}$），再将它们的输出逐元素相乘，形成一种“门控函数”。并非所有 LLM 都采用这种方式，因此你有时会看到单个 $W_\text{In}$ 矩阵，MLP 参数总数也会是 2DF，而不是 3DF。在这种情况下，通常会增大 D 和 F，以保持参数数量与使用 3 个矩阵时相同。尽管如此，LLaMA、DeepSeek 以及许多其他模型都会使用某种形式的门控 einsum。

**注意 2［MHA 注意力］：** 对自注意力而言，T 与 S 相同；但对交叉注意力而言，两者可能不同。在普通多头注意力（Multi-Head Attention，MHA）中，N 与 K 相同；而对于[多查询注意力](https://arxiv.org/abs/1911.02150)（Multi-Query Attention，MQA）[[mqa]](../#ref-mqa)，K=1；对于[分组 MQA](https://arxiv.org/abs/2305.13245)（Grouped MQA，GMQA）[[gmqa]](../#ref-gmqa)，只要求 K 能整除 N。

**注意 3［前置归一化与后置归一化］：** 上图展示的是所谓“前置归一化”（pre-norm）架构，其中归一化发生在残差连接之前，通常写作 `x + attn(norm(x))`。如今，LLaMA-3 等模型采用这种架构。最初的 Transformer 论文使用的是“后置归一化”（post-norm）架构，其中 LayerNorm 发生在残差连接之后，即 `norm(x + attn(x))`。

<span id="global-flops-and-params-calculation"></span>

## 全局 FLOP 与参数计算

我们来计算 Transformer 每层的 FLOP（这样就不用到处附加 **L** 因子了）。请注意，下面的训练 FLOP 几乎总是推理 FLOP 的 3 倍，因此把任意总数除以 3，就能得到仅前向传播的成本。

<span id="mlps"></span>

### MLP

Transformer 的 MLP 通常由两个输入矩阵乘法和一个输出矩阵乘法组成，两个输入矩阵乘法的结果会逐元素组合：

$$
\begin{array}{ccc}
\textrm{operation} & \textrm{train FLOPs} & \textrm{params} \\
\hline \\
A[B,T,\red{D}] \cdot W_{in1}[\red{D}, F] & 6BTDF & DF \\[10pt]
A[B,T,\red{D}] \cdot W_{in2}[\red{D}, F] & 6BTDF & DF \\[10pt]
\sigma\left(A_{in1}\right)[B,T, F] * A_{in2}[B,T, F] & \gray{O(BTF)} \\[10pt]
A[B,T,\red{F}] \cdot W_{out}[\red{F}, D] & 6BTDF & DF \\[10pt]
\hline \\
& \approx 18BTDF & 3DF
\end{array}
$$

<span id="attention"></span>

### 注意力

对于 **Q** 头数与 **KV** 头数不同的一般分组查询注意力情形，假设 **Q**、**K**、**V** 投影的头维度 H 相同，并估算 **QKVO** 矩阵乘法的成本：

$$
\begin{array}{ccc}
\textrm{operation} & \textrm{train FLOPs} & \textrm{params} \\
\hline \\
A[B,T,\red{D}] \cdot W_{Q}[\red{D}, N, H] & 6BTDNH & DNH \\[10pt]
A[B,T,\red{D}] \cdot W_{K}[\red{D}, K, H] & 6BTDKH & DKH \\[10pt]
A[B,T,\red{D}] \cdot W_{V}[\red{D}, K, H] & 6BTDKH & DKH \\[10pt]
A[B,T,\red{N}, \red{H}] \cdot W_{O}[\red{N}, \red{H}, D] & 6BTDNH & DNH \\[10pt]
\hline \\ & 12BTD(N+K)H & 2D(N+K)H
\end{array}
$$

点积注意力运算更加微妙：它实际上是一次 $TH \cdot HS$ 矩阵乘法，沿 $B$、$K$ 维度批处理；接着执行一次 softmax，再执行一次 $TS \cdot SH$ 矩阵乘法，同样沿 $B$、$K$ 维度批处理。下面用蓝色突出显示批处理维度：

$$
\begin{array}{cc}
\textrm{operation} & \textrm{train FLOPs} \\
\hline \\[3pt]
Q[\blue{B}, T, \blue{K}, G, \red{H}] \cdot K[\blue{B}, S, \blue{K}, \red{H}]
& 6BTSKGH = 6BTSNH  \\[3pt]
\textrm{softmax}_S \;\; L[B, T, S, K, G] & \gray{O(BTSKG) = O(BTSN)} \\[3pt]
S[\blue{B}, T, \red{S}, \blue{K}, G] \cdot V[\blue{B}, \red{S}, \blue{K}, H]
& 6BTSKGH = 6BTSNH \\[3pt]
\hline \\
& \approx 12BTSNH = 12BT^2NH \\
\end{array}
$$

**注意［因果掩码］：** 最近的大多数 Transformer 使用因果掩码，而非完整的双向注意力。在这种情况下，点积运算的有效 FLOP 会减半。为了在实践中实现这种缩减，我们需要使用注意力内核，而不能采用朴素的 einsum。

<span id="other-operations"></span>

### 其他运算

Transformer 中还会发生其他几种运算。与之相比，LayerNorm 的成本很低，在一阶成本估算中可以忽略。请注意，每层通常包含两个 LayerNorm（一个在注意力之前，一个在 MLP 之前）。此外，还有最后那个规模巨大的反嵌入矩阵乘法（不过它不是逐层出现的）。

$$
\begin{array}{ccc}
\textsf{operation} & \textsf{train FLOPs} & \textsf{params} \\
\hline \\
2 \times \textrm{layernorm}_D \;\; A[B,T,\red{D}] & \gray{O\left(BTD\right)} & \gray{2D} \\[10pt]
A[B,T,\red{D}] \cdot W_{unembed}[\red{D}, V] & 6BTDV & DV \\
\end{array}
$$

<span id="general-rule-of-thumb-for-transformer-flops"></span>

### Transformer FLOP 的通用经验法则

如果忽略点积注意力的成本（对较短上下文训练而言，这是合理的），那么所有层的总 FLOP 为

$$
\begin{align*}
(18BTDF + 12BTD(N+K)H)L = 6 *BT * (3DF + 2D(N+K)H)L \\ = 6 * \textrm{num tokens} * \textrm{parameter count}
\end{align*}
$$

由此得到一个著名的经验法则，用于估算稠密 Transformer 的 FLOP 数量，其中忽略注意力 FLOP。（反嵌入也是一个简单的矩阵乘法，具有 $6BTDV$ FLOP 和 $DV$ 个参数，同样遵循这条经验法则。）

<span id="fractional-cost-of-attention-with-context-length"></span>

### 注意力成本随上下文长度变化的占比

如果确实计入上面的点积注意力，并假设 $F=4D$、$D=NH$（这是典型设置）且 $N=K$，那么点积注意力 FLOP 与所有矩阵乘法 FLOP（包括注意力投影）之比为：

$$
\small{\frac{\textrm{attention FLOPs}}{\textrm{matmul FLOPs}} = \frac{12BT^2NH}{18BTDF + 24BTDNH} = \frac{12BT^2D}{4*18 BTD^2 + 24 BTD^2} = \frac{12BT^2D}{96 BTD^2} = \frac{T}{8D}}
$$

结论是，**训练期间只有当 T>8D 时，点积注意力 FLOP 才会占据主导。** 对于 D ~ 8k，这大约对应 64K 个 token。这也合乎情理，因为它意味着，随着 MLP 大小增加，注意力 FLOP 会变得没那么重要。对于大型模型，注意力的二次方成本实际上并不是延长上下文训练的巨大障碍。然而，对于较小模型，例如 D=4608 的 Gemma-27B，注意力会在序列长度约为 37k 时占据主导。[^ch4-2] Flash Attention 也有助于缓解长上下文的成本，我们会在[附录 A](#appendix-a-how-does-flash-attention-work)中简要讨论。

<span id="miscellaneous-math"></span>

## 其他数学知识

<span id="sparsity-and-mixture-of-experts"></span>

### 稀疏性与专家混合

如果不简要讨论专家混合（Mixture of Experts，MoE）模型[[moe]](../#ref-moe)，那将是我们的疏忽。MoE 会把标准 Transformer 中单个稠密 MLP 块替换成一组可动态路由的独立 MLP。近似来看，**MoE 就是一个普通稠密模型，只不过每层有 E 个 MLP 块**，而不是只有一个。每个 token 会激活其中 $k$ 个专家，通常 $k \ll E$。比率 $E / k$ 称为稀疏度，通常介于 8 到 64 之间（例如 [DeepSeek v3](https://arxiv.org/pdf/2412.19437) 实际上有 $k=8$、$E=256$）。与稠密版本相比，这会让参数量增加 $O(E)$，同时把每个 token 激活的参数总数乘以 $k$。

![图：一个包含 $n$ 个专家的 MoE 层示例。门控专家将每个 token 路由给其中 $k$ 个专家，再对这 $k$ 个 MLP 的输出求和。参数量是单个专家大小的 $n$ 倍，但每个 token 只使用 $k$ 个专家。](/images/scaling-book/img/moe.png)

来源：[Deepgram 的混合专家指南](https://deepgram.com/learn/mixture-of-experts-ml-model-guide)。

与稠密模型相比，MoE 会引入新的通信，主要是两次 AllToAll（一次在 MoE 块之前，一次在之后），用于把 token 路由到正确的专家，再将其送回原设备。[^ch4-3] 不过，正如上一节所见，沿单条轴执行时，每次 AllToAll 的成本仅为同类 AllGather 的 1/4（对于双向环而言）。

<span id="gradient-checkpointing"></span>

### 梯度检查点

反向传播是一种以计算换内存的算法。反向传播无需 $O(n_\text{layers}^2)$ FLOP，但**需要 $O(n_\text{layers})$ 内存**，用于保存前向传播期间产生的所有中间激活值。尽管这好于二次方计算，但内存成本极其高昂：设模型的 $B * T=4M$（每批总计 4M 个 token）、L=64、D=8192；如果要避免反向传播中所有不必要的计算，就必须保存大约 $2 * 20 * B * T * D * L = 84TB$ 的 bfloat16 激活值。这里的 20 来自（粗略）计算上面 Transformer 图中的每个中间节点，因为例如

$$
f(x) = \exp(g(x))
$$

$$
\frac{df}{dx} = \exp(g(x)) \cdot \frac{dg}{dx}
$$

所以，为了避免重新计算，我们需要保存前向传播中的 $g(x)$ 和 $\exp(g(x))$。为避免保存如此多内存，可以选择只保存一部分中间激活值。下面是我们采用的几种策略。

* **按块重计算：** 只保存每层的输入。这是我们使用的最激进方法，每层只保存 1 个检查点，意味着在上面的例子中只需保存 4.2TB。代价是，反向传播时基本要重复执行前向传播的全部 FLOP，也就是把 FLOP 从 $6 \cdot \text{num params} \cdot \text{num tokens}$ 增加到约 $8 \cdot \text{num params} \cdot \text{num tokens}$。
* **只保存大型矩阵乘法：** 另一项简单策略是只保存大型矩阵乘法的输出。这样便可避免在反向传播时重新计算任何大型矩阵乘法，但仍需重新计算其他激活函数和部分注意力。它会把上面的每层 20 个节点减少到更接近每层 7 个。

这绝不是一份全面清单。使用 JAX 时，这些操作通常由 `jax.remat`/`jax.checkpoint` 控制（可在[这里](https://jax.readthedocs.io/en/latest/_autosummary/jax.checkpoint.html)阅读更多内容）。

<span id="key-value-kv-caching"></span>

### 键—值（KV）缓存

正如[第 7 节](../07-inference/#all-about-transformer-inference)将会介绍的，LLM 推理有两个关键部分：预填充和生成。

* **预填充（Prefill）** 处理一个长提示词，并把注意力激活值保存到键—值缓存（Key-Value Cache，KV Cache）中，以供生成阶段使用；具体保存的是注意力块中的键—值投影。
* **生成（Generation）** 把多个这样的 KV 缓存批处理到一起，并从每个缓存中采样 token。

因此，每个 KV 缓存实际上都是一个大小为 $[2, S, L, K, H]$ 的数组，其中 2 代表键和值。这相当大！int8 格式键—值缓存的总大小为 $2SLKH$。对于一个中等大小的模型，若上下文长度为 8k、有 64 层，且 $KH = NH = D = 8192$，则大小为 $2 \cdot 8192 \cdot 64 \cdot 8192 = 8\text{GiB}$。现在你应该能明白，我们为什么希望使用 $K \ll N$ 的 GMQA。

<span id="what-should-you-take-away-from-this-section"></span>

## 本节应当掌握哪些要点？

* Transformer 的总参数量和 FLOP 很容易计算。假设采用 MHA（批大小为 B、词表大小为 V、序列长度为 T、D=d<sub>model</sub>、F=d<sub>ff</sub>），总结如下：


<!-- $$
\begin{array}{ccc}
\textrm{Component} & \textrm{Params per layer} & \textrm{Training FLOPs per layer} \\
\hline \\
\textbf{MLP} & 3DF & 18BTDF \\[10pt]
\textbf{Attention} & 4DNH & 24BTDNH + 12BT^2NH \\[10pt]
\textbf{Other} & D & BTD \\[10pt]
\textbf{Vocab} & DB \text{ (total, not per-layer)} & 12BTDV \\[10pt]
\end{array}
$$ -->


| 组件              | 每层参数量                | 每层训练 FLOP                |
| :---------------- | :------------------------ | :--------------------------- |
| **MLP**           | 3DF                       | 18BTDF                       |
| **注意力**        | 4DNH                      | 24BTDNH \+ 12BT<sup>2</sup>NH |
| **其他**          | 2D                        | BTD                          |
| **词表**          | DV（总计，并非每层）      | 12BTDV                       |

* 只要序列长度 $T < 8D$，MLP 块的参数量就会主导总参数量，而且 MLP 块也会主导 FLOP 预算。
* 对于合理的上下文长度，训练期间的总 FLOP 预算可以用 $6 \cdot \text{num\_params} \cdot \text{num\_tokens}$ 很好地近似。
* 推理期间，每个 KV 缓存的大小大约为 $2 \cdot S \cdot L \cdot K \cdot H$（其中 K 是 KV 头的数量），不过架构上的修改通常可以减小这个值。

<span id="a-few-problems-to-work"></span>

## 练习题

**问题 1：** 一个 $D=4096$、$F=4 \cdot D$、$V=32,000$、$L=64$ 的模型有多少参数？其中注意力参数占多大比例？每个 token 的 KV 缓存有多大？*你可以假设 $N\cdot H=D$，并采用 int8 KV 的多头注意力。*

<details>
<summary>点击此处查看答案。 </summary>


1. 总参数量大约为 $L \cdot (3DF + 4DNH + 2D) + 2DV$（计入每层的两个 LayerNorm）。代入给定数值，可得 $64 \cdot (3 \cdot 4e3 \cdot 16e3 + 4 \cdot 4e3 \cdot 4e3 + 2 \cdot 4e3) + 2 \cdot 4e3 \cdot 32e3 = 16e9$，即 16B 参数。
2. 一般情况下，注意力参数与总参数量之比为 $4DNH / (4DNH + 3DF) = 4D^2 / (4D^2 + 12D^2) = 1/4$。这意味着大约 1/4 的参数用于注意力。
3. 对每个 token，KV 缓存为 $2 \cdot L \cdot N \cdot H = 2 \cdot 64 \cdot 4096$（int8 格式），也就是 `512 KiB / token`。

</details>

**问题 2：** 在 `{'X': 4, 'Y': 8, 'Z': 4}` 上执行 A[B<sub>X</sub>, D<sub>Y</sub>] \*<sub>D</sub> W[D<sub>Y</sub>, F]，总共需要多少 FLOP？每块 TPU 执行多少 FLOP？

<details>
<summary>点击此处查看答案。 </summary>


该运算的“理论”FLOP 总数为 $2 \cdot B \cdot D \cdot F$。但是，由于计算没有沿 Z 维度分片，我们实际上额外执行了 Z 倍的 FLOP，也就是总计 $2 \cdot B \cdot D \cdot F \cdot Z$。由于计算沿其他维度分片，每台设备的总量大约为 $2 \cdot B \cdot D \cdot F / (X \cdot  Y)$。

</details>

**问题 3：** 执行 $A[I,J,K,L] * B[I,J,M,N,O] \rightarrow C[K,L,M,N,O]$ 涉及多少 FLOP？

<details>
<summary>点击此处查看答案。 </summary>


按照上面的规则，I 和 J 是收缩维度，而 K、L、M、N、O 是非收缩维度。这里没有“批处理维度”，因此结果就是 $2 \cdot I \cdot J \cdot K \cdot L \cdot M \cdot N \cdot O$，即所有轴大小的乘积。如果存在共享轴，它只会计算一次。

</details>

**问题 4：** 自注意力的算术强度是多少（忽略 Q/K/V/O 投影）？*请把答案表示为 Q 和 KV 长度 T 与 S 的函数。* 注意力在多长的上下文中会受到 FLOP 限制？给定 TPU 的 HBM 带宽，请绘制注意力相对于 FFW 块的有效成本随上下文长度增长而变化的曲线。

<details>
<summary>点击此处查看答案。 </summary>


自注意力需要加载 $Q$、$K$ 和 $V$ 激活值，然后计算 $\text{softmax}(Q \cdot K) \cdot V$，再把结果写回 HBM。这个过程会使用 Flash Attention，因此下面的数学有一些需要注意之处；但基本而言，在 bf16 中，自注意力执行

$$
\text{Q[B,T,N,H]} \rightarrow_\text{reshape} \text{Q[B, T, K, G, H]} \cdot \text{K[B, S, K, H]} \rightarrow \text{O[B, T, S, K, G]}
$$

$$
U=\text{softmax}_S(\text{O[B, T, S, K, G]})
$$

$$
\text{U[B, T, S, K, G]} \cdot \text{V[B, S, K, H]} \rightarrow \text{X[B, T, K, G, H]}
$$

因此，总字节数为 $2 * \text{sizeof}(Q) + 2 * \text{sizeof(K or V)} = 4BTNH + 4BSKH = 4BHK * (TG + S)$，FLOP 总数为 $4BTSNH + O(BTSN)$，算术强度为 $4BTSKGH / (4BHK * (TG + S))$。

所以基本上，在预填充期间有 $S=T$，算术强度为 $4BT^2KGH / 4BHKT \cdot (G+1) = TG/(G + 1) = O(T)$。生成期间 $T=1$，因此有 $4BSKGH / (4BHK \cdot (G + S)) = SG / (G + S) \rightarrow G$，这里假设 $S$ 非常大。根据你如何理解这个问题，在预填充或训练期间，如果不进行序列分片，自注意力会在 S=240 时变为计算受限。生成期间，因为 $G$ 很小，所以我们永远不会计算受限。尽管如此，可以看出，增大 $G$ 会让我们更接近计算受限。

</details>

**问题 5：** 在什么序列长度下，自注意力 FLOP 会与 QKVO 投影 FLOP 相等？

<details>
<summary>点击此处查看答案。 </summary>


这完全是在问 $24BTDNH = 12BT^2NH$ 何时成立。化简后得到 $2D = T$；例如，当 $D=4096$ 时，结果是 $8192$。这告诉我们，对于大多数合理的上下文长度，矩阵乘法 FLOP 更大。

</details>

**问题 6：** 假设在前向传播期间，我们只保存 Transformer 层中 7 次主要矩阵乘法各自的输出（Q、K、V、O \+ 三个 FFW 矩阵）。为了在反向传播中“重计算”，需要额外执行多少 FLOP？

<details>
<summary>点击此处查看答案。 </summary>


只保存七次矩阵乘法的输出（Q、K、V、O、W₁、W₂、W₃），意味着反向传播必须重新计算两次注意力矩阵乘法

$$
QK^{\top} \quad\text{and}\quad \operatorname{softmax}(QK^{\top})V
$$

才能得到 $\frac{\partial L}{\partial W_\text{O}}$。

二者都是 $T \times T$ 矩阵乘法，并沿 $B$ 个序列和 $N$ 个头进行批处理，因此额外 FLOP 为

$$
4 \; B \, T^{2} \, N \, H.
$$

其他需要重新计算的运算包括：
1. 执行 $O(BTD)$ 的运算，用于计算 $\frac{\partial L}{\partial W_\text{In1}}$ 和 $\frac{\partial L}{\partial W_\text{In2}}$。
2. 还要执行 $O(BTF)$ 的运算，用于计算 $\frac{\partial L}{\partial W_\text{Out}}$。

</details>

**问题 7：** DeepSeek v3 称其在 14.8T 个 token 上训练了 2.79M H800 小时（[来源](https://arxiv.org/pdf/2412.19437v1)）。已知它有 37B 个激活参数，它们大约实现了多高的硬件利用率？*提示：请注意，它们使用的是不带结构化稀疏的 FP8 FLOP。*

<details>
<summary>点击此处查看答案。 </summary>


从[这里](https://lenovopress.lenovo.com/lp1814.pdf)的规格表可知，带稀疏时的 FP8 性能为 3,026 TFLOP/s；不带稀疏时通常是它的一半（`1.513e15` FLOP/s）。2.79M H800 小时意味着总 FLOP 为 `2.79e6 * 1.513e15 * 60 * 60 = 1.52e25`。给定 37B 个激活参数，这次训练运行应使用约 `6 * 37e9 * 14.8e12 = 3.3e24` FLOP。这意味着 FLOP 利用率约为 `3.3e24 / 1.52e25 = 21.7%`。

</details>

**问题 8：** 专家混合（MoE）模型拥有标准稠密 MLP 块的 $E$ 份副本，每个 token 激活其中 $k$ 个专家。对于权重采用 int8 的 TPU v5e 上的 MoE，需要多大的 token 批大小才能达到计算受限？对于拥有 256 个（路由）专家且 $k=8$ 的 DeepSeek，这个数值是多少？

<details>
<summary>点击此处查看答案。 </summary>


因为每个专家有 $E$ 份副本，并采用 int8，所以对于每个权重矩阵，需要加载 $E \cdot D \cdot F$ 字节。因为每个 token 激活 $k$ 个专家，所以对于每个权重矩阵，有 $2\cdot k \cdot B \cdot D \cdot F$ FLOP。若采用 int8 权重和 bfloat16 FLOP，要达到计算受限，算术强度（每加载一个字节所执行的 FLOP）就必须超过 TPU 的约 240 FLOP/字节；当 $(2\cdot k \cdot BDF) / EDF > 240$ 或 $k \cdot B / E > 120$ 时，这一条件成立。

因此，要达到计算受限，必须有 $B > 120 \cdot E / k$。对 DeepSeek 而言，得到 $B > 120 \cdot 256 / 8 = 3840$。在生成阶段，这是一个大得惊人的批大小。

</details>

<span id="thats-it-for-part-4-for-part-5-about-scaling-transformer-training-click-here"></span>

### 第 4 部分到此结束！关于扩展 Transformer 训练的第 5 部分，请[点击这里](../05-training/#how-to-parallelize-a-transformer-for-training)！

<span id="appendix"></span>

## 附录

<span id="appendix-a-how-does-flash-attention-work"></span>

### 附录 A：Flash Attention 如何工作？

反对把 Transformer 扩展到极长上下文的传统理由是，注意力 FLOP 和内存使用量会随上下文长度按二次方增长。注意力 QK 乘积的形状确实是 $[B, T, S, N]$，其中 B 是批大小，T 和 S 是 Q 与 K 的序列维度，N 是头数；但这种说法有几个非常重要的限定条件：

1. 正如前文所述，即使成本是二次方的，也只有在 $T > 8 \cdot D$ 时，注意力 FLOP 才会占据主导；而在训练期间，单个注意力矩阵的内存与驻留在内存中的全部权重和激活值检查点相比很小，尤其是在经过分片后。
2. 为了计算注意力，我们不需要具体化完整的注意力矩阵！可以计算局部和与局部最大值，从而始终只具体化数组的一小块。尽管 FLOP 总数仍按二次方增长，但内存压力会大幅降低。

第二点最早由 [Rabe 等人（2021）](https://arxiv.org/abs/2112.05682)提出，之后又见于 [Flash Attention 论文](https://arxiv.org/abs/2205.14135)（Dao 等人，2022）。基本思想是按 K/V 数据块计算注意力：先计算局部 softmax 和一些辅助统计量，再将它们传给下一个数据块，由后者将其与自己的局部数据块合并。具体来说，我们计算

1. **M：** 序列维度上 $q \cdot k$ 的运行最大值
2. **O：** 序列维度上运行中的完整注意力 softmax
3. **L：** 运行中的分母 $\sum_i \exp(q \cdot k_i - \text{running max})$

有了这些量，我们只需恒定大小的内存，就能计算新的最大值、新的运行和以及新的输出。粗略地说，注意力大致执行下面的运算：

$$
\text{Attn}(Q, K, V) = \sum_i \frac{\exp(Q \cdot K_i - \max_j Q \cdot K_j) V_i}{\sum_l \exp(Q \cdot K_l - \max_j Q \cdot K_j)}
$$

为了数值稳定性，式中减去了最大值；这样做不会影响结果，因为 $\sum_i \exp(a_i + b) = \exp(b) \sum \exp(a)$。只看上面的分母：假设有两个连续的键向量数据块 $K^1$ 和 $K^2$，并为它们分别计算局部 softmax 和 $L^1$、$L^2$

$$
L^1 = \sum_i \exp(Q \cdot K_i^1 - \max_j Q \cdot K_j^1)
$$

$$
L^2 = \sum_i \exp(Q \cdot K_i^2 - \max_j Q \cdot K_j^2)
$$

那么，利用下面的公式，可以把它们合并成这两个数据块共同的完整 softmax 和：

$$
L^\text{combined} = \exp(M^1 - \max(M^1, M^2)) \cdot L^1 + \exp(M^2 - \max(M^1, M^2)) \cdot L^2
$$

其中

$$
M^1 = \max_j Q \cdot K_j^1 \text{ and } M^2 = \max_j Q \cdot K_j^2
$$

完整 softmax 也可以采用这种方法，从而让我们能够累积任意大的 softmax 和。下面是 Flash Attention 论文中的完整算法。

![](/images/scaling-book/img/flash-algo.png)

从硬件角度看，这让我们能够把 Q 数据块装入 VMEM（也就是上面算法所称的片上 SRAM）；这样每次迭代只需加载 KV 数据块，从而提高算术强度。运行中的统计量也可以保存在 VMEM 中。

最后还要强调一个容易忽略的细节：为了让 Flash VJP（反向模式导数）计算在训练中切实可行，会用到注意力 softmax 的一个性质。我们定义一个中间 softmax 数组：

$$
S_{ij} = \frac{e^{\tau q_i \cdot k_j}}{\sum_l e^{\tau q_i \cdot k_l}}
$$

在注意力中，我们从反向模式的 *dO* 与 *V* 数组得到 *dS*：

$$
dS_{ij} = dO_{id} \cdot_d V_{jd} = \sum_d dO_{id} V_{jd}
$$

在把这个梯度反向传播给 Q 和 K 时，

$$
d(q_i \cdot k_j) = (dS_{ij} - S_{ij} \cdot_j dS_{ij}) S_{ij}
$$

我们利用一个恒等式，将沿大型键**长度**维度的收缩，换成沿特征**深度**维度的局部收缩。

$$
\begin{align*}
S_{ij} \cdot_j dS_{ij} &= \sum_j \frac{e^{\tau q_i \cdot k_j}}{\sum_k e^{\tau q_i \cdot k_k}} \sum_d dO_{id} V_{jd} \\
&= \sum_d dO_{id} \sum_j \frac{e^{\tau q_i \cdot k_j}}{\sum_k e^{\tau q_i \cdot k_k}} V_{jd} \\
&= \sum_d dO_{id} O_{id} \\
&= dO_{id} \cdot_d O_{id}
\end{align*}
$$

这种替换对实现按序列分块的 VJP *局部*计算至关重要，也进一步支持了环形注意力等巧妙的分片方案。

[^ch4-1]: <b>收缩</b>维度是在运算期间被求和的轴（它们出现在两个输入中，但不出现在输出中），例如矩阵乘法中的内维度。<b>批处理</b>维度是同时出现在两个输入中、并原样保留到输出的共享轴；它们为彼此独立的子问题建立索引，在计算 FLOP 时不会相乘。用 einsum 的术语来说：同时出现在两个输入和输出中的标签是批处理标签；出现在两个输入中、但不出现在输出中的标签是收缩标签。
[^ch4-2]: 请注意，一些现代 OSS 模型引入了局部注意力或其他优化，以降低注意力成本并改变这一 Roofline。
[^ch4-3]: 严格来说，只有当数据或序列沿与专家相同的轴分片时，才会发生这种情况。
