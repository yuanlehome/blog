---
title: Flash Decoding - 基本原理
slug: flash-decoding
date: '2026-02-16'
tags: []
status: published
source_url: 'https://wangyu.me/posts/ml/flash-decoding/'
source_author: wangyu.me
imported_at: '2026-02-16T01:08:43.475Z'
source:
  title: wangyu.me
  url: 'https://wangyu.me/posts/ml/flash-decoding/'
cover: /images/others/flash-decoding/001-8d7d381d.gif
---

# Flash Decoding - 基本原理

分类：[机器学习](/categories/#机器学习)标签： [LLM](/tags/#LLM)创建时间：2026-01-07 21:44:00

在前面的文章中，我介绍了 [Flash Attention 的基本原理](/ml/flash-attention-overview/)，在阅读本文时，我假设你对 Flash Attention 的原理已经有了基本了解，知道如何对数据分块，如何使用 online softmax 来渐进地完成 attention 的计算。如果你尚不清楚，建议先了解 Flash Attention 的原理，这是理解 Flash Decoding 的基础。

Flash Attention 在训练阶段和推理的预填充（Prefill）阶段表现出色，这两种场景下，输入的都是完整的序列，在计算 attention 时，$Q$、$K$、$V$ 的长度相同，因为 $Q$ 中各个 token 之间相互独立，Flash Attention 对 $Q$ 进行分块并行处理，可以有效地利用 GPU 的计算资源。

但在推理的解码（Decode）阶段，decode 的输入是上一轮输出的 token，此时 Query 序列极短（通常仅一个 token），而 Key 和 Value 则包含了之前所有 token 的信息（当前 token 的 KV + KV Cache）。随着生成的 token 不断增多，K/V 的长度也会持续增长，可能达到数千甚至数万个 token。此时如果输入的 batch 较小，则会因为任务量不足，导致 GPU 的计算资源无法被充分利用。

为了解决在 decode 阶段，并行度不足的问题，Tri Dao 等人提出了 [Flash Decoding](https://pytorch.org/blog/flash-decoding/)，专门针对解码进行优化，通过在 KV 维度上进行分块并行计算，显著提升了解码阶段的注意力计算效率。在 Flash Attention 的源码中已经实现了 Flash Decoding 的算法。

本文将详细描述 Flash Decoding 的原理，同时给出 CUDA 实现的源码。另外我在 [mini-flash-attention](https://github.com/w4096/mini-flash-attention/blob/b2508f4c5508a0067b4d6e8d829d505b0694f644/csrc/mfa/decode.cuh)中实现了 Flash Decoding 的 CUDA 版本，感兴趣的读者可以参考源码。

## LLM 推理的两个阶段

在深入了解 Flash Decoding 之前，我们需要先了解大语言模型（LLM）推理的两个阶段，即预填充（Prefill）和解码阶段（Decode）。

### 预填充阶段（Prefill）

当用户输入一段 prompt 时，模型对输入的所有 token 做完整的前向计算。在这个阶段，$Q$、$K$、$V$ 的序列长度都等于 prompt 的长度。例如用户输入了 1024 个 token 的 prompt，那么 $Q$、$K$、$V$ 的行数都是 1024。此时 Flash Attention 可以在 $Q$ 的维度上做分块并行计算，GPU 的利用率很高。

这里复用我在 [Flash Attention 的基本原理](/ml/flash-attention-overview/) 中的图示：

![](https://wangyu-name.oss-cn-hangzhou.aliyuncs.com/2025/posts/1zs3x9.svg)

在计算时可以将 $Q$ 分为多个块，每个分块由一个线程块（Thread Block）处理，每个 $Q$ 块和所有的 $K$ 和 $V$ 块进行注意力计算，得到该 $Q$ 块对应的输出。这个阶段，GPU 有较多的任务可以执行，计算资源可以被充分利用。

下面是两层循环的伪代码，GPU 可以在外层循环的分块上并行执行：

```text
for bq in range(0, n, Bm):       # 外层循环：遍历 Q 的分块，可并行
    bQ = Q[bq:bq+Bm, :]
    for bk in range(0, n, Bn):   # 内层循环：遍历 K/V 的分块，串行
        bK = K[bk:bk+Bn, :]
        bV = V[bk:bk+Bn, :]
        ...
```

### 解码阶段（Decode）

预填充完成后，会得到 prompt 中所有 token 的 K/V 缓存（KV Cache），此后模型开始逐个生成新的 token。在预填充阶段，$Q$ 有很多行，外层循环可以分成很多块并行执行，GPU 的 SM（流多处理器）可以被充分利用。但在解码阶段，$Q$ 只有一行（或几行），外层循环的迭代次数为 1，只有一个线程块在工作，如果输入 batch 较小，则可能有大量 SM 处于空闲状态。而这就是 Flash Decoding 需要解决的问题。

## Flash Decoding 的核心思想

Flash Decoding 的解决方案很直观：既然 Q 维度没有足够的并行度，那就在 KV 维度上引入并行。具体做法是将内层循环对 KV 的遍历拆分为多个独立的并行任务：

1. **Split-KV 并行计算**：将 KV Cache 沿序列维度切分为多个块（split），每个块由一个独立的线程块处理。每个线程块独立计算其负责的 KV 块对应的局部注意力结果。
1. **归约（Reduce）**：所有线程块完成计算后，通过一次归约操作将各个局部结果合并为最终的全局注意力输出。

下图展示了 Flash Decoding 的计算流程：

![](https://wangyu-name.oss-cn-hangzhou.aliyuncs.com/2025/posts/8rfog4.svg)

_将 K/V 切分为多个块，Q 分别与多个 KV 块进行注意力计算，最终将结果归约合并为全局输出。_

下面还有一个来自 PyTorch 的博客中的动画示意：

![](/images/others/flash-decoding/001-8d7d381d.gif)

## Flash Decoding 的计算过程

下面我使用公式来描述 Flash Decoding 的计算过程。

**单个 KV 分块**

对于第 $k$ 个 KV 块，设其中包含的注意力分数为 $s\_i^{(k)}$（即 $Q$ 与该块中第 $i$ 个 Key 的点积除以 $\sqrt{d}$）。

该块的注意力分数最大值为：

$m^{(k)} = \max\_i\\, s\_i^{(k)}$

做了数值稳定性处理的指数和：

$l^{(k)} = \sum\_i e^{s\_i^{(k)} - m^{(k)}}$

局部加权求和为：

$O^{(k)} = \frac{\sum\_i e^{s\_i^{(k)} - m^{(k)}} \cdot v\_i^{(k)}}{l^{(k)}}$

在计算 $O^{(k)}$ 时，使用了最大值 $m^{(k)}$ 进行数值稳定化处理。但本质上 $O^{(k)}$ 的值和下面式子的结果完全相同（分子分母同时乘以 $e^{m^{(k)}}$）：

$O^{(k)} = \frac{\sum\_i e^{s\_i^{(k)}} \cdot v\_i^{(k)}}{\sum\_i e^{s\_i^{(k)}}}$

**合并多个 KV 块的结果**

现在考虑合并两个分块的输出，我们记分块 1 的输出为 $O^{(1)}$，其 softmax 分母中的指数和为 $\text{expsum}^{(1)}$，分块 2 的输出为 $O^{(2)}$，其 softmax 分母中的指数和为 $\text{expsum}^{(2)}$。则合并过程可以分为以下几步：

**第一步：计算全局指数和**

全局指数和为：

$\text{expsum}^\* = \text{expsum}^{(1)} + \text{expsum}^{(2)}$

**第二步**：修正各块的加权求和

将每个分块的输出的分母替换为全局指数和：

$O^{(1)}\_{\text{new}} = \frac{\text{expsum}^{(1)}}{\text{expsum}^\*} O^{(1)}$ $O^{(2)}\_{\text{new}} = \frac{\text{expsum}^{(2)}}{\text{expsum}^\*} O^{(2)}$

**第三步**：累加各分块的结果

合并后的输出为：

$O^\* = O^{(1)}\_{\text{new}} + O^{(2)}\_{\text{new}}$

## 使用 LSE 优化合并逻辑

前面描述的分块结果合并算法要求每个 K/V 分块保存局部的输出 $O^{(k)}$ 以及局部最大值 $m^{(k)}$ 和局部指数和 $l^{(k)}$。在 Flash Attention 的代码实现中，实际上对每个分块除了输出 $O^{(k)}$ 之外，只保存一个 LSE（Log-Sum-Exp）值。

### LSE 的定义

LSE 即 Log Sum Exp，它的定义为：

$\text{LSE}(x\_1, x\_2, \ldots, x\_n) = \log\left(\sum\_{i=1}^{n} e^{x\_i}\right)$

在编程实现时，LSE 会使用如下公式计算：

$\text{LSE}(x\_1, \ldots, x\_n) = m + \log\left(\sum\_{i=1}^{n} e^{x\_i - m}\right)$

其中 $m = \max(x\_1, \ldots, x\_n)$。

下面是推导过程：

$$
\begin{align\*} \text{LSE}(x\_1, \ldots, x\_n) &= \log\left(\sum\_{i=1}^{n} e^{x\_i}\right) \\\ &= \log\left(e^m \sum\_{i=1}^{n} e^{x\_i - m}\right) \\\ &= m + \log\left(\sum\_{i=1}^{n} e^{x\_i - m}\right) \end{align\*}
$$

LSE 本质上是 softmax 归一化分母的对数形式。阅读后文你就会明白，其实每个分块只需要这一个统计量就可以完成分块的合并操作。

### 使用 LSE 合并分块

对于分块 k，其输出为：

$O^{(k)} = \frac{\sum\_i e^{s\_i^{(k)} - m^{(k)}} \cdot v\_i^{(k)}}{\sum\_i e^{s\_i^{(k)} - m^{(k)}}}$

其中 $m^{(k)}$ 是第 $k$ 个分块的注意力分数的最大值。这里使用了最大值进行了数值稳定化处理，这是防止在编程时出现数值溢出的问题。理论上它的值等价于：

$O^{(k)} = \frac{\sum\_i e^{s\_i^{(k)}} \cdot v\_i^{(k)}}{\sum\_i e^{s\_i^{(k)}}}$

在合并时，其原理就是将分母替换为全局的指数和，从而得到全局归一化的输出。因此可以对 $O^{(k)}$ 进行如下变换：

$O^{(k)} = O^{(k)} \frac{\sum\_i e^{s\_i^{(k)}}}{\sum\_k{\sum\_i e^{s\_i^{(k)}}}}$

这样就消除掉了原来的局部指数和，并使用全局的指数和。而这个计算过程可以使用 LSE 来表示。

假设有两个块的 LSE 分别为 $LSE\_1$ 和 $LSE\_2$，对应的输出为 $O\_1$ 和 $O\_2$，则最终输出 $O$ 的计算如下：

$LSE\_\text{merge} = \log(\exp({LSE\_1}) + \exp({LSE\_2}))$ $O = O\_1 \cdot \exp({LSE\_1 - LSE\_\text{merge}}) + O\_2 \cdot \exp({LSE\_2 - LSE\_\text{merge}})$

初次看这个计算公式的时候，大概难以看明白，但只要稍微展开一下，就很容易看懂。

将 $LSE\_\text{merge}$ 展开后可得：

$$
\begin{align\*} LSE\_\text{merge} &= \log(\exp({LSE\_1}) + \exp({LSE\_2})) \\\ &= \log(\exp({\log \sum\_i e^{s\_i^{(1)}}}) + \exp({\log \sum\_i e^{s\_i^{(2)}}})) \\\ &= \log (\sum\_i e^{s\_i^{(1)}} + \sum\_i e^{s\_i^{(2)}}) \end{align\*}
$$

$\exp({LSE\_1 - LSE\_\text{merge}})$ 展开后可得：

$$
\begin{align\*} \exp({LSE\_1 - LSE\_\text{merge}}) &= \exp({ \log \sum\_i e^{s\_i^{(1)}} - \log (\sum\_i e^{s\_i^{(1)}} + \sum\_i e^{s\_i^{(2)}}) }) \\\ &= \frac{\sum\_i e^{s\_i^{(1)}}}{\sum\_i e^{s\_i^{(1)}} + \sum\_i e^{s\_i^{(2)}}} \end{align\*}
$$

这样就使用 LSE 完成了多个分块的合并操作，但是其本质就是调整每个分块的 softmax 的分母，将局部指数和替换为全局指数和。

LSE 实际上只包含了 softmax 的分母中指数和部分的信息。虽然在编程实现的时候会使用如下公式来计算：

$\text{LSE}^{(k)} = m^{(k)} + \log \sum\_i e^{s\_i^{(k)} - m^{(k)}}$

但这仅仅是为了防止数值溢出。虽然公式中出现了最大值 $m^{(k)}$，但实际上 LSE 中并没有包含最大值的信息，而且也不需要最大值。

下面两个式子是完全相同的：

$O^{(k)} = \frac{\sum\_i e^{s\_i^{(k)} - m^{(k)}} \cdot v\_i^{(k)}}{\sum\_i e^{s\_i^{(k)} - m^{(k)}}} \quad (1)$ $O^{(k)} = \frac{\sum\_i e^{s\_i^{(k)}} \cdot v\_i^{(k)}}{\sum\_i e^{s\_i^{(k)}}} \quad (2)$

虽然在编程实现时，$O^{(k)}$ 使用的是第一个式子来计算的，它使用了最大值 $m^{(k)}$ 来防止数值溢出。但在使用 LSE 进行分块合并时，$O^{(k)}$ 其实被视为第二个式子。因此只需要知道每个分块的指数和就可以完全多个分块输出的合并了。

## 总结

Flash Decoding 解决了 LLM 解码阶段 Flash Attention 并行度不足的问题，其原理其实很好理解。在 Flash Attention 的实现中，使用了 LSE（Log-Sum-Exp）值来记录每个分块的归一化信息，最后基于 LSE 对各个分块的输出进行合并。而这个过程相对而言比较难理解，本文给出了比较详细的公式推导。最终合并的计算过程就是使用如下两个公式：

$LSE\_\text{merge} = \log(\exp({LSE\_1}) + \exp({LSE\_2}))$ $O = O\_1 \cdot \exp({LSE\_1 - LSE\_\text{merge}}) + O\_2 \cdot \exp({LSE\_2 - LSE\_\text{merge}})$

你可以将其展开，然后试着推导，就很容易明白其中的原理。在 Flash Attention 的实现中，之所以对指数和取对数，保存 LSE 而非直接保存指数和，我想这是为了避免指数和过大导致的数值溢出问题。

#### 评论 （评论内容仅博主可见，不会公开显示）

邮箱（选填）

邮箱（选填）

提交
