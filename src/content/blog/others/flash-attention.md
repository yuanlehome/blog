---
title: Flash Attention 的实现原理
slug: flash-attention
date: '2026-01-27'
tags: []
status: published
source_url: 'https://wangyu.me/posts/ml/flash-attention-overview/'
source_author: wangyu.me
imported_at: '2026-01-27T11:26:14.113Z'
source:
  title: wangyu.me
  url: 'https://wangyu.me/posts/ml/flash-attention-overview/'
cover: /images/others/flash-attention/001-cfa412d2.svg
---

# Flash Attention 的实现原理

分类：[机器学习](/categories/#机器学习) 标签： [LLM](/tags/#LLM) 创建时间：2025-12-17 21:10:00

自从 _Attention is All You Need_ 论文提出以来，Transformer 被广泛应用于各类深度学习任务中。尤其是近年来，GPT 系列、LLaMA 等大规模语言模型（LLM）再度推动了 Transformer 的发展与落地。注意力机制（Attention Mechanism）作为 Transformer 的核心组件，其计算效率直接决定了模型的训练和推理速度。2022 年提出的 Flash Attention 算法大幅优化了注意力计算效率，现已成为大规模语言模型中广泛采用的核心技术之一。

在后续系列文章中，我会详细介绍 Flash Attention 的原理与实现细节，并尝试用 CUDA 实现一个简化版的 Flash Attention，以加深对其工作原理的理解。

## 注意力机制

本节先回顾注意力机制的基本原理和计算流程，为后续介绍 Flash Attention 做铺垫。对此已熟悉的读者可跳过本节。

### 注意力机制计算过程

Transformer 中注意力机制的核心思想是：通过计算查询（Query）、键（Key）和值（Value）之间的关联关系，动态调整输入序列中各位置信息的权重。具体而言，给定输入序列的表示矩阵 $X$，我们通过线性变换得到查询矩阵 $Q$、键矩阵 $K$ 和值矩阵 $V$：

$Q = XW\_Q, \quad K = XW\_K, \quad V = XW\_V$

其中，$W\_Q$、$W\_K$ 和 $W\_V$ 为可学习的权重矩阵。接下来计算查询与键的点积，得到注意力分数矩阵，最终通过 softmax 归一化后与值矩阵相乘，得到注意力输出：

$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d\_k}}\right)V$

这里的 $d\_k$ 是键的维度，用于缩放点积结果以避免数值过大。softmax 函数的定义为：

$\text{softmax}(z\_i) = \frac{e^{z\_i}}{\sum\_{j} e^{z\_j}}$

下面是注意力机制的计算流程图：

![](/images/others/flash-attention/001-cfa412d2.svg)

以输入序列 $X$ 包含 4 个 token 为例：先通过线性变换得到 $Q$、$K$、$V$，计算 $QK^T$ 得到 4×4 的注意力分数矩阵（表征每个查询与所有键的相似度）；对矩阵逐行做缩放和 softmax 归一化（确保每行权重和为 1）；最后将归一化后的注意力权重矩阵与 $V$ 相乘，本质是对 $V$ 按 $Q$ 和 $K$ 的相似度进行加权求和，最终得到注意力输出。

### 多头注意力机制

为增强模型的表达能力，Transformer 引入了多头注意力机制（Multi-Head Attention）：将 $Q$、$K$、$V$ 分别划分为多个子空间（每个子空间对应一个“注意力头”），每个头独立计算注意力输出，最后拼接所有头的输出并通过线性变换得到最终结果：

$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}\_1, \ldots, \text{head}\_h)W\_O$

具体流程为：$Q$、$K$、$V$ 被划分为 $h$ 个子空间（每个子空间维度为 $d\_k/h$），各注意力头独立完成注意力计算，最终拼接所有头的输出并通过线性变换得到最终结果。下面是多头注意力机制的计算流程图：

![](/images/others/flash-attention/002-16d387e1.svg)

上图中，输入序列 $X$ 经线性变换得到 $Q$、$K$、$V$ 后，被切分为 96 个子空间（对应 96 个注意力头）；每个头独立完成注意力计算，最终拼接所有头的输出并通过线性变换得到最终结果。

### 因果注意力

语言模型中通常要求：处理某个 token 时仅能访问其之前的上下文信息。为此，计算注意力分数时需对权重矩阵做掩码处理——对于位置 $i$ 的查询，仅允许与位置 $j \leq i$ 的键交互。实际实现时，将注意力分数矩阵中 $j > i$ 的“未来位置”设为负无穷，经 softmax 归一化后这些位置的权重会趋近于 0。这种机制被称为因果注意力（Causal Attention）或掩码注意力（Masked Attention）。

![](/images/others/flash-attention/003-83f5496e.svg)

如上图所示，注意力矩阵的上三角部分被置为负无穷，逐行 softmax 后这些位置的权重趋近于 0。以输入序列 abcde 为例：计算注意力矩阵对 $V$ 的加权求和时，a 仅能访问自身信息，b 仅能访问 a 和 b 的信息，依此类推。这确保模型生成下一个 token 时，仅能基于已有上下文。

## 注意力计算的挑战

深度学习模型的训练与推理通常依赖 GPU 等硬件加速器，而 GPU 包含寄存器、共享内存、全局内存等多层级内存（访问速度和容量差异显著）。注意力计算中，$Q$、$K$、$V$ 通常存储在全局内存，计算时需加载到共享内存/寄存器中处理。

常规注意力计算流程如下：

1. **计算注意力权重矩阵**

注意力权重矩阵 $\frac{QK^T}{\sqrt{d\_k}}$ 的计算是一个常规的矩阵乘法操作，已经有成熟且高效的计算方法，采用分块矩阵乘法（Tiling）技术，可以将 $Q$ 和 $K$ 矩阵分成多个小块，分别加载到共享内存中进行计算，最终得到注意力权重矩阵。对于一个长度为 $n$ 的序列，注意力分数矩阵是一个 $n \times n$ 的矩阵。假如序列长度为 4096，并采用 16 位浮点数存储，则需要 32MB 的内存空间来存储这个矩阵。这个矩阵会被存储在 GPU 的全局内存中以供后续计算使用。

2. **计算 softmax 归一化**

做 softmax 运算时，每一行可以独立计算，可以很好地并行化处理。而且可以采用后文描述的 online softmax 计算方法，因此单纯的 softmax 计算并不是主要的性能瓶颈。

3. **计算注意力输出**

最后一步是将归一化后的注意力权重矩阵与值矩阵 $V$ 相乘，得到最终的注意力输出。这个过程同样可以采用分块矩阵乘法技术进行计算。

上述流程的核心问题不在计算本身，而在数据传输。注意力权重矩阵需多次读写，且其大小随序列长度平方增长——长序列场景下，全局内存与共享内存间的频繁数据传输会导致内存带宽成为核心瓶颈。此外，因果注意力计算中，注意力矩阵的上三角部分为无效数据，常规流程仍会对其做无效计算，既浪费算力，又增加内存访问负担。

## Flash Attention 的核心思想

Flash Attention 的核心是融合注意力计算流程，减少注意力权重矩阵的读写操作。GPU 的计算速度远快于内存访问速度，减少内存交互可显著提升整体效率。本节将详细介绍 Flash Attention 的实现原理。由于 Flash Attention 1.0 版本和 2.0 版本在实现细节上有巨大差异，而后续版本的改进主要集中在实现细节上，因此我这里主要基于 Flash Attention 2.0 版本的核心思想进行介绍。

为了便于理解 Flash Attention 的实现原理，我会先介绍 online softmax 的原理，这个是理解 Flash Attention 的关键，然后介绍 Flash Attention 中的分块计算流程。

### Online Softmax

理解 Flash Attention 的核心是掌握 online softmax 的计算方式，以及如何将 softmax 与加权求和融合。只有理解了 online softmax 的计算原理，才能理解 Flash Attention 的实现原理。Online softmax 的原理其实很简单，但我初次接触时，看到的都是一大堆公式推导，反而让人心生畏惧难，很快就放弃了。这里我避免数学公式的堆砌，并使用更直观的方式来解释其原理。

首先，回顾 softmax 的定义：

$\text{softmax}(z\_i) = \frac{e^{x\_i}}{\sum\_{j} e^{x\_j}}$

给定一组输入 $x\_1, x\_2, \ldots, x\_n$，我们希望计算它们的 softmax 输出。传统的方法是先计算所有 $e^{x\_j}$ 的和，然后再计算每个 $e^{x\_i}$ 除以这个和。然而，这种方法有两个问题：

1. 指数函数可能导致数值溢出，尤其是当输入值较多时。
1. 需要两次遍历数据，第一遍计算指数和，第二遍计算归一化后的输出。

为了解决第一个问题，我们可以引入一个偏移量 $m = \max(x\_1, x\_2, \ldots, x\_n)$，将输入值减去这个最大值，从而避免指数函数的溢出：

$\text{softmax}(z\_i) = \frac{e^{x\_i - m}}{\sum\_{j} e^{x\_j - m}}$

给 $x\_i$ 减去 $m$ 不会改变 softmax 的输出，因为这相当于对分子和分母同时乘以 $e^{-m}$。

为了解决第二个问题，我们可以采用一种叫做 online softmax 的方法，它可以减少对数据的遍历次数。首先来看一下常规的 softmax 实现：

```cpp
void softmax(const float *x, float *out, int dim) {
    float m = std::numeric_limits<float>::min();
    for (int i = 0; i < dim; i++) {
        m = std::max(m, x[i]);
    }

    float expsum = 0;
    for (int i = 0; i < dim; i++) {
        expsum += std::exp(x[i] - m);
    }

    for (int i = 0; i < dim; i++) {
        out[i] = std::exp(x[i] - m) / expsum;
    }
}
```

这里的实现分为三步：

1. 寻找最大值 `m`。
1. 计算指数和 `expsum`。
1. 计算 softmax 输出。

第二步依赖于第一步计算出的最大值 `m`，而第三步又依赖于第一步和二步计算出的 `m` 和 `expsum`。因此，似乎一定需要三次遍历。但利用指数函数的性质，我们可以将前两步合并为一步，从而减少一次遍历。

初中数学中，我们知道指数函数满足以下性质：

$e^{a} \* e^{b} = e^{a+b}$

基于此原理我们可以在遍历过程中动态更新最大值 `m` 和指数和 `expsum`，下面是具体的实现：

```cpp
float expsum = 0;
float max = std::numeric_limits<float>::min();
for (int i = 0; i < dim; i++) {
    float m = std::max(x[i], max);
    if (m > max) {
        expsum = expsum * std::exp(max - m);
        max = m;
    }
    expsum += std::exp(x[i] - max);
}
```

在遍历过程中，我们始终保持 `max` 为当前遍历过的最大值。并使用此 `max` 来计算指数和 `expsum`。当遇到一个更大的值 `m` 时，我们需要调整 `expsum` 的值。因为此时 $\text{expsum} = \sum e^{x\_i - \text{max}}$，而我们需要将其转换为 $\text{expsum} = \sum e^{x\_i - m}$。根据指数函数的性质，我们可以通过乘以 $e^{\text{max} - m}$ 来实现这一转换。

下面是具体的数学推导：

$$
\begin{align\*} \text{expsum} &= \sum e^{x\_i - max} \* e^{\text{max} - m} \\\ &= \sum e^{x\_i - max} \* e^{\text{max}} \* e^{-m} \\\ &= \sum e^{x\_i} \* e^{-m} \\\ &= \sum e^{x\_i - m} \end{align\*}
$$

遍历完成后，我们就得到了最终的 `max` 和 `expsum`，可以用它们来计算 softmax 输出：

```cpp
for (int i = 0; i < dim; i++) {
    out[i] = std::exp(x[i] - max) / expsum;
}
```

注意力计算中，softmax 的最终目的是对 $V$ 加权求和。若将 softmax 与加权求和分开计算，需先算 softmax 输出、再与 $V$ 相乘，这需要遍历两次数据。而这两步其实可以被融合在一起，可以通过一次遍历就完成加权求和的计算。

考虑加权求和的计算：

$$
\begin{align\*} \text{out} &= \sum\_{i} \text{softmax}(x\_i) \* v\_i \\\ &= \sum\_{i} \frac{e^{x\_i - \text{max}}}{\text{expsum}} \* v\_i \\\ &= \frac{\sum\_{i} e^{x\_i - \text{max}} \* v\_i}{\text{expsum}} \end{align\*}
$$

我们同样可以在遍历过程中动态地修正上面公式中的分子部分 $\sum\_{i} e^{x\_i - \text{max}} \* v\_i$，虽然这里乘以了 $v\_i$，但和前面我们修正 `expsum` 的思路是一样的。下面是具体的实现：

```cpp
float softmax_weighted_sum(const float *x, const float *v, int dim) {
    float weighted_sum = 0;  // 初始化加权求和变量
    float expsum = 0;
    float max = std::numeric_limits<float>::min();
    for (int i = 0; i < dim; i++) {
        float m = std::max(x[i], max);
        if (m > max) {
            expsum *= std::exp(max - m);
            weighted_sum *= std::exp(max - m);     // 同步调整加权和，这与调整 expsum 类似
            max = m;
        }
        expsum += std::exp(x[i] - max);
        weighted_sum += std::exp(x[i] - max) * v[i];    // 累加当前元素的加权贡献
    }
    return weighted_sum / expsum;    // 最终归一化
}
```

给定输入向量 `x` 和对应待加权的向量 `v`，函数内部需要基于 `x` 计算 softmax，并对 `v` 做加权求和。在上述实现中，只需要一次遍历就能完成所有计算。

注意：上面的实现中，可能会担心 `weighted_sum` 累加时可能会溢出。但因为 $e^{x\_i - max}$ 的值在 0 到 1 之间，只需要用更高的精度来存储 `weighted_sum` 就不会溢出。

通过这种方式，仅需一次遍历数据就能完成加权求和，可以有效减少内存访问次数。而这正是 Flash Attention 的核心思想之一。

### 分块计算

与通用矩阵乘法（GEMM）类似，Flash Attention 将 $Q$、$K$、$V$ 划分为小块计算，下图展示了分块计算的整体流程：

![](/images/others/flash-attention/004-5f3712da.svg)

图中，$Q$ 被划分为多个 $Bm \times d$ 的分块，$K$ 和 $V$ 被划分为多个 $Bn \times d$ 的分块，计算流程包含两层循环：

1. **外层循环**

外层循环遍历 $Q$ 矩阵的小块，这里每一个块都包含 $Bm$ 个 token，其中每个 token 的维度为 $d$。因为通常使用的是多头注意力机制，每个头的维度通常为 64 到 128 这样的量级，所以这里 $d$ 通常不会很大，$Q$ 矩阵的分块可以很容易地加载到共享内存中进行计算。

2. **内层循环**

内层循环中，每次读取 $K$ 和 $V$ 矩阵的小块，这里每一个块都包含 $Bn$ 个 token。使用 Q 矩阵的小块与 K 矩阵的小块计算注意力权重，并使用 online softmax 融合加权求和的方式，计算出当前 $Q$ 矩阵小块对应的输出结果。然后不断将结果累加并修正，直到遍历完所有的 $K$ 和 $V$ 矩阵，最后得到完整的输出结果。

分块计算的伪代码如下：

```text
for bq in range(0, n, Bm):       # 遍历 Q 矩阵的小块
    bQ = Q[bq:bq+Bm, :]          # 加载 Q 矩阵的小块
    bO = np.zeros((Bm, d))       # 初始化输出小块
    softmax = Softmax()          # 初始化 online softmax 状态
    for bk in range(0, n, Bn):   # 遍历 K 矩阵的小块
        bK = K[bk:bk+Bn, :]      # 加载 K 矩阵的小块
        bV = V[bk:bk+Bn, :]      # 加载 V 矩阵的小块

        softmax.update(bQ, bK, bV, bO)  # 更新 online softmax 状态，计算加权和

    softmax.finalize(bO)         # 归一化输出
    Out[bq:bq+Bm, :] = bO        # 写回输出矩阵
```

这里的 `softmax.update` 函数实现了前文介绍的 softmax 融合加权求和的计算逻辑。`softmax.finalize` 函数则是使用最终的 `expsum` 来归一化输出结果。如果不清楚，请回顾前文对 softmax 的介绍。

因为 Attention 计算中，Q 之间各个 token 是独立的，因此我们可以将 Q 矩阵划分为多个小块进行计算。为了让讲解的跳跃性不要太大，这里先假设 K 和 V 矩阵可以一次性加载到共享内存中进行计算，示意图如下：

![](/images/others/flash-attention/005-948918a4.svg)

Q 的分块和 K 可以计算出权重矩阵，权重矩阵在和 V 相乘得到对应于 Q 的分块的输出结果。可以不断循环这个过程，完成对所有 Q 分块的计算。

然而，上述计算过程中，因为 K 和 V 矩阵比较大，无法完全载入到 GPU 的共享内存中，所以对 K 和 V 也需要按照分块进行加载计算。我们可以将 K 和 V 矩阵划分为多个 $Bn \times d$ 的小块，每次只加载一个小块进行计算。

![](/images/others/flash-attention/006-06d312a3.svg)

假如我们 Q 和 K 的分块都是 4 个 token，那么上图的计算完成后，就得到了 Q 分块中的 4 个 token 和序列中前 4 个 token 的注意力权重。下一轮迭代时，又能计算出与另外 4 个 token 的注意力权重。遍历完 K 后，就能得到 Q 分块中 4 个 token 与整个序列的注意力权重。但我们并不需要显式地存储这个注意力权重矩阵，而是直接在计算过程中使用这些计算出来的权重和 V 矩阵的分块做加权求和。这和前文介绍的 softmax 融合加权求和的计算原理是一致的，完全可以使用相同的计算逻辑。

![](/images/others/flash-attention/007-476d6ebc.svg)

### 小结

本节介绍了 Flash Attention 的核心思想，我认为在掌握了 online softmax 融合加权求和的计算原理后，在理解分块计算的流程，Flash Attention 的实现原理就变得非常直观了。

## Flash Attention 高效的原因

在做注意力计算时，注意力矩阵的大小随着序列长度的平方增长，这部分数据会占用大量的内存空间。假如序列长度为 4096，并采用 16 位浮点数存储，注意力矩阵需要占用 32MB 的空间，而 Q、K 和 V 矩阵，如果维度为 128，则每个矩阵需要占用 1MB 的空间，总共 3MB。注意力矩阵的大小远远超过 QKV 矩阵的大小。Flash Attention 采用类似 GEMM 中的分块计算方式，并使用 online softmax 在一次迭代中完成加权求和，这避免了对注意力矩阵的多次读写操作。

另外在计算因果注意力时，注意力矩阵的上三角部分是无效的，因此在计算过程中可以避免对这些无效部分进行计算。在使用 PyTorch 来做注意力计算时，通常会先计算完整的注意力矩阵，然后再对上三角部分进行掩码处理，这样会导致大量的无效计算。而 Flash Attention 在分块计算时，可以直接跳过内循环中那些无效的 K 和 V 块，从而避免了无效计算。

![](/images/others/flash-attention/008-64d5728c.svg)

## 总结

本文回顾了注意力机制的基本原理和计算流程，分析了传统注意力计算在处理长序列时面临的内存带宽瓶颈问题。随后，详细介绍了 Flash Attention 的核心思想，包括 softmax 融合加权求和的计算方法，以及分块计算的实现流程，这是两个点是理解 Flash Attention 关键点。

本文力求以通俗易懂的方式解释 Attention 机制和 Flash Attention 的实现原理。如果你仍然有没看懂的地方，这是我的错，欢迎在评论区留言讨论。而如果你看懂了，那就是你的错，你应该早点看。
