---
title: "Transformer 推理详解"
description: "对 Transformer 执行推理可能与训练大不相同，部分原因在于推理增加了一个需要考虑的新因素：延迟。本章将从使用模型采样单个新 token 开始，一路讲到如何把大型 Transformer 高效扩展到多个加速器切片上，并将其作为推理引擎的一部分运行。"
chapter: 7
order: 7
part: 2
partTitle: "Transformer"
sourcePath: "inference.md"
sourceCommit: "44109cacac9c5a9809a81c68ae4d45d7d2632ea6"
---

<span id="all-about-transformer-inference"></span>

# Transformer 推理详解

<span id="the-basics-of-transformer-inference"></span>

## Transformer 推理基础

现在，你已经训练好了一个 Transformer，并想用它生成一些新序列。*说到底，基准分数上升、损失曲线下降，都只是模型真正投入使用时能否带来有趣结果的替代指标！*[^ch7-1]

从概念上说，采样很简单。输入一个序列，我们最喜欢的 Transformer 就会输出 $\log p(\text{next token}_i \vert \text{previous tokens})$，也就是所有可能下一个 token 的对数概率。我们可以从这个分布中采样，得到一个新 token。追加这个 token 并重复上述过程，就会得到一个延续提示词的 token 序列。

![图：从 Transformer 进行朴素采样。蓝色 logits 给出了下一个 token 的分布，我们可以从中采样。请注意，每一步都会重新处理整个前缀，因此算法的运行时间为 $\Theta(n^2)$。](/images/scaling-book/img/naive-inference.png)

上面描述的是 Transformer 采样的朴素实现。它虽然可行，但**实践中绝不会这样做**，因为每生成一个 token 都要重新处理整个序列！这种算法在 FFW 上的复杂度为 $O(n^2)$，在注意力机制上的复杂度为 $O(n^3)$，而这些计算是为了生成 $n$ 个 token！

**怎样避免这个问题？** 事实证明，我们不必每次都执行完整的前向传播，而是可以保存每次前向传播中的一些中间激活，从而避免重新处理先前的 token。具体而言，由于点积注意力中的某个 token 只会关注之前的 token，我们只需把每个 token 的键投影和值投影写入一个称为 **KV 缓存**的新数据结构。保存过去 token 的这些键/值投影后，未来的 token 就可以直接计算其 $q_i \cdot k_j$ 乘积，而无需再对更早的 token 执行任何新的 FLOPs。太棒了！

基于这一点，推理包含两个关键部分：

* <strong style="color: red;">预填充</strong>：给定一个长提示词，我们同时处理提示词中的所有 token，并把得到的激活（具体来说是键值投影）保存到一个 <strong>“KV 缓存”</strong>中。还要保存最后一个 token 的 logits。
* <strong style="color: blue;">生成</strong>：给定 KV 缓存和上一轮的 logits，我们以增量方式从 logits 中采样一个 token，将该 token 再次输入 Transformer，并为下一步生成一组新的 logits。还要把这个新 token 的 KV 激活追加到 KV 缓存。重复这一过程，直到遇到特殊的 `<EOS>` token，或者达到某个最大长度限制。

下面是使用 KV 缓存进行采样的示意图：

![图：使用 KV 缓存进行高效 Transformer 采样。预填充会处理提示词，并把每个 token 的键值激活保存到缓存中。生成会利用该缓存和最后一个 token 的 logits 采样新 token；新 token 经过模型并关注缓存后，其键值投影会追加回缓存。该算法在 MLP 块中的复杂度为 $O(n)$。](/images/scaling-book/img/cached-inference.png)

借助 KV 缓存采样后，因为不再重新处理之前的 token，生成 $n$ 个 token 的时间复杂度在 FFW 上降为 $O(n)$，在注意力上降为 $O(n^2)$。不过，生成一个序列仍需进行多次前向传播——当你向 Gemini 或 ChatGPT 发出查询、结果以流式方式返回时，后台发生的就是这个过程。每个 token（通常）都对应一次针对庞大模型的、彼此独立但部分缓存的 Transformer 调用。

很快就会看到，<strong style="color: red;">预填充</strong>与 <strong style="color: blue;">生成</strong>是截然不同的两件事——Transformer 推理其实是伪装成一个任务的两个任务！与训练相比，KV 缓存也是一种全新且重要的复杂性来源。

<span id="what-do-we-actually-want-to-optimize"></span>

### 我们真正想优化什么？

继续之前，值得强调推理中一个全新的方面：延迟。训练时只关心吞吐量（每颗芯片**每秒**处理的 token 总数），推理时却必须关注 token 的产出速度，包括**首 token 延迟（Time To First Token，TTFT）**和**单 token 延迟**。例如：

* **离线批量推理**用于评估和数据生成时，只关心推理的总体成本，而不在意单个样本的延迟。
* **聊天界面/流式任务**既要以低成本大规模运行，又要保持较低的 TTFT，并以超过人类阅读速度的速率生成 token。
* **边缘推理**（例如在笔记本电脑上运行 `llama.cpp`）一次只需服务一个用户，并追求尽可能低的延迟，但硬件可能受到很大限制。

最大化硬件利用率依然至关重要，有助于降低成本和 TTFT；但与训练不同，它并不*必然*能在所有情境中为单个用户带来更好的体验。加速器、系统和模型架构层面的许多优化，都需要在延迟、吞吐量、上下文长度乃至模型质量之间做权衡。

<span id="a-more-granular-view-of-the-transformer"></span>

### 更细粒度地观察 Transformer

到目前为止，我们大多把 Transformer 看成一叠前馈块。从 FLOPs 和内存角度看，这往往合理，但不足以正确地对推理建模。[^ch7-2] 正如[第 4 部分](../04-transformers/#all-the-transformer-math-you-need-to-know)所见，Transformer 前向传播的主要组成部分包括：

1. **大量线性运算**，包括 MLP（$W_{in}$、$W_{out}$），以及注意力的 QKV 投影和输出投影（$W_Q$、$W_K$、$W_V$ 和 $W_O$）。这些运算都需要从 HBM 读取参数和一批激活，执行一些 FLOPs，再把结果写回 HBM。
2. **点积注意力**。需要从 HBM 读取一批键值投影和一批查询激活，执行若干内积与 softmax 运算，再把注意力结果写回 HBM。
3. **其他所有操作**，包括应用层归一化、激活函数、token 采样、更新 KV 缓存以及位置嵌入。这些操作确实会消耗一些 FLOPs，但要么被上述操作压倒，要么融合进上述操作。

接下来的几节将分别在预填充和生成的语境下考察这些部分，并询问最可能限制性能的是什么。在单颗加速器内部，我们究竟受计算限制还是内存限制？我们想强调，预填充与生成的答案会有多么不同。

<span id="linear-operations-what-bottlenecks-us"></span>

### 线性运算：瓶颈在哪里？

无论位于 MLP 块还是注意力中，所有线性运算从概念上看都相同。它们的算术强度取决于批大小。我们在[第 1 节](../01-roofline/#all-about-rooflines)已经做过这个计算，但值得再重复一遍。考虑一个矩阵乘法：将一个 $\text{bf16[B, D]}$ 批次乘以一个 $\text{bf16[D, F]}$ 矩阵。它可能是大型 MLP 块（$W_\text{in}$ 或 $W_\text{out}$），也可能是较小的注意力投影之一（$W_Q$、$W_K$、$W_V$、$W_O$）。为了执行这次矩阵乘法，需要把这两个数组从 HBM 载入 MXU，完成乘法，再将结果写回 HBM。和之前一样，有：

$$
T_\text{math} = \frac{\text{Computation FLOPs}}{\text{Accelerator FLOPs/s}} = \frac{2BDF}{\text{Accelerator FLOPs/s}}
$$

$$
T_\text{comms} = \frac{\text{Communication Bytes}}{\text{Bandwidth Bytes/s}} = \frac{2BD + 2FD + 2BF}{\text{Bandwidth Bytes/s}}
$$

TPU 或 GPU 可以在执行计算的同时加载数据，从而重叠这两部分。因此，要达到计算受限，就需要 $T_\text{math} \geq T_\text{comms}$，即：

$$
\frac{2BDF}{2BD + 2DF + 2BF} \geq \frac{\text{Accelerator FLOPs/s}}{\text{Bandwidth Bytes/s}} \underset{\text{TPU v5e}}{=} \frac{1.97E+14}{8.20E+11} = 240
$$

等式右侧是硬件的算术强度。现在假设 $D$ 和 $F$ 相比 $B$ 大得多（批大小通常最多为 500，而 $D$ 和 $F > 10k$），利用 $\small{2BD + 2DF + 2BF \approx 2DF}$，可以把分母简化为

$$
\begin{align*}
\frac{2BDF}{2BD + 2DF + 2BF} \approx \frac{2BDF}{2DF} \geq \frac{\text{Accelerator FLOPs/s}}{\text{Bandwidth Bytes/s}} \\
\underset{\text{TPU v5e}}{=} \frac{1.97E+14}{8.20E+11} \implies B \geq 240 = B_{\text{crit}}
\end{align*}
$$

如果量化权重，或者在矩阵乘法中使用更低精度的 FLOPs，这个临界批大小可能变化。例如，将权重量化为 int8 或 fp8 时，$B_\text{crit}$ 会减小 2 倍。如果使用 int8 或 fp8 执行 FLOPs，$B_\text{crit}$ 会增大 2 倍。因此，令 $\beta = \text{bits per param} / \text{bits per activation}$、$\alpha_\text{hbm} = C / W_\text{hbm}$，真正的临界批大小就是 $B_\text{crit} = \beta \alpha_\text{hbm}$。

**要点：** 当且仅当每个副本的 **token** 批大小大于 $B_\text{crit} = C / W_\text{hbm} \cdot (\text{bits per param} / \text{bits per activation}) = \beta \cdot \alpha_\text{hbm}$ 时，Transformer 矩阵乘法才是计算受限的。对 TPU v5e 上的 bf16 激活而言，这个值为 240 个 token；对 H100 而言，约为 280 个 token。

训练期间，所有矩阵乘法都有很高的算术强度，因为同一组权重会在很大的批次上重复使用。**这种高算术强度也延续到预填充，因为用户提示词通常有数百乃至数千个 token。** 如前所见，TPUv5e 的硬件算术强度为 240。因此，如果把一个超过 240 个 token 的序列输入到此硬件上以 bf16 运行的稠密模型，预计会达到计算受限，一切都很好。比这更短的提示词从技术上说可以合批以提高利用率，但通常没有必要。

**要点：** 预填充期间，所有矩阵乘法基本总是计算受限的。因此，只需最大化硬件利用率或 MFU（模型 FLOPs 利用率），就足以最大化每颗芯片的吞吐量（成本）并优化延迟（体现为 TTFT）。除非提示词极短，否则按提示词进行合批只会增加延迟，而对预填充吞吐量改善很小。

然而，生成期间，由于步骤之间存在顺序依赖，每个请求一次只能对一个 token 执行前向传播！因此，我们只能（相对容易地）通过将多个请求合批、沿批维度并行化来取得良好利用率。稍后会进一步讨论，但在不影响延迟的情况下把许多并发请求真正合批在一起并不容易。正因如此，**生成期间要让硬件 FLOPs 饱和困难得多。**

**要点：** 生成期间，总 token 批大小必须大于 $B_{\text{crit}}$，线性/前馈运算才会达到计算受限（TPU v5e 上的 bf16 参数对应 240）。由于生成是逐 token 串行发生的，这要求我们把多个请求合批，而这很困难！

*值得注意的是，这个规模有多大！* 生成批大小为 240，意味着 240 个并发请求同时生成；对稠密模型而言，也意味着 240 份独立的 KV 缓存。因此，除了某些批量推理场景外，实践中很难做到这一点。相比之下，预填充期间一次处理超过 240 个 token 十分常见，不过随着稀疏性提高，仍需谨慎处理。

**请注意，这个确切数值会随量化方式和硬件而变化。** 加速器通常能在较低精度下提供更高算力。例如，如果参数采用 int8、但计算采用 bf16，临界批大小会降到 120。若激活和参数都采用 int8，它又会升回 240，因为 TPUv5e 能提供 400 TOPs/s 的 int8 x int8 算力。

<span id="what-about-attention"></span>

### 注意力呢？

考察点积注意力运算时，情况会变得更复杂，尤其因为还必须把 KV 缓存纳入考虑。先只看纯多头注意力中的一个注意力头。在一次 Flash Attention 融合中，我们[^ch7-3]：

1. 从 HBM 读取 $Q$ 激活，其形状为 $\text{bf16[B, T, D]}$。
2. 从 HBM 读取 $KV$ 缓存，也就是一对 $\text{bf16[B, S, D]}$ 张量。
3. 执行 $2BSTD$ FLOPs，用于 $QK$ 矩阵乘法。使用 Flash Attention 后，不必把 $\text{bf16[B, S, T]}$ 注意力矩阵写回 HBM。
4. 执行 $2BSTD$，用于注意力 $AV$ 矩阵乘法。
5. 把得到的 $\text{bf16[B, T, D]}$ 张量写回 HBM。

把这些合在一起，得到：

$$
\text{Multiheaded Attention Arithmetic Intensity} = \frac{4BSTD}{4BSD + 4BTD} = \frac{ST}{S+T}
$$

对预填充而言，因为执行的是自注意力，所以 $S=T$；于是可简化为 $T^2 / 2T = T / 2$。这很棒，因为它意味着**预填充期间注意力的算术强度为 $\Theta(T)$**。也就是说，注意力很容易达到计算受限。只要序列长度相当大，就不会有问题！

但由于生成时序列维度微不足道，而且 $B$ 和 $D$ 维度会约掉，可以作如下近似：

$$
S \gg T = 1 \implies \frac{ST}{S+T} \approx 1
$$

这很糟，因为它意味着无法通过任何手段提高生成期间注意力的算术强度。我们只执行极少量 FLOPs，却要加载庞大的 KV 缓存。**所以注意力几乎总是受内存带宽限制！**

**要点：** 预填充期间，只要序列长度合理（大致 $\gt 480$ 个 token），注意力通常就是计算受限的；而生成期间，算术强度低且恒定，因此总是受内存带宽限制。

*从概念上说，为什么会这样？* 主要原因是，模型的线性部分之所以计算受限，是因为参数（占用大量内存带宽的部分）会被许多批项重复使用。然而，每个批项都有自己的 KV 缓存，因此更大的批大小意味着更多 KV 缓存。除非对架构进行大幅调整，否则这里几乎*总是*内存受限。

这也意味着，一旦参数内存与 KV 缓存内存变得相当，继续增大批大小所带来的吞吐量收益就会递减。这种递减收益的影响程度取决于单个序列的参数字节数与 KV 缓存字节数之比，也就是大致为 $2DF / SHK$。由于 $HK\approx D$，它大致取决于 $F$ 与序列长度 $S$ 之比。这还取决于能让 KV 缓存变小的架构修改（稍后马上会讲）。

<span id="theoretical-estimates-for-llm-latency-and-throughput"></span>

### LLM 延迟与吞吐量的理论估计

利用这些数学关系，可以为优化时应追求的步骤时间给出相当不错的界。**（注意：如果希望读者从整章只记住一件事，那就是下面这条。）** 生成期间批大小较小时（这很常见），可以假设注意力块和 MLP 块都受内存带宽限制，从而给出单步延迟的下界：

$$
\begin{equation*}
\text{Theoretical Min Step Time} = \frac{\text{Batch Size} \times \text{KV Cache Size} + \text{Parameter Size}}{\text{Total Memory Bandwidth}}
\end{equation*}
$$

类似地，对吞吐量有：

$$
\begin{equation*}
\text{Theoretical Max Tokens/s} = \frac{\text{Batch Size} \times \text{Total Memory Bandwidth}}{\text{Batch Size} \times \text{KV Cache Size} + \text{Parameter Size}}
\end{equation*}
$$

最终，随着批大小增大，FLOPs 会开始压过参数加载，因此实践中采用更一般的公式：

$$
\begin{align}
\tiny \text{Theoretical Step Time (General)} = \underbrace{\frac{\text{Batch Size} \times \text{KV Cache Size}}{\tiny \text{Total Memory Bandwidth}}}_{\text{Attention (always bandwidth-bound)}} + \underbrace{\max\left(\frac{2 \times \text{Batch Size} \times \text{Parameter Count}}{\text{Total FLOPs/s}}, \frac{\text{Parameter Size}}{\text{Total Memory Bandwidth}}\right)}_{\tiny \text{MLP (can be compute-bound)}}
\end{align}
$$

其中注意力部分（左侧）从不会达到计算受限，因此不需要 FLOPs Roofline。这些公式很适合做粗略估算，例如：

**随堂测验：** 假设要在 TPU v5e 4x4 切片上，用 int8 参数和 bf16 FLOPs，从一个 30B 参数的稠密模型执行批大小为 4 个 token 的生成步骤；上下文长度为 8192，KV 缓存为 100 kB / token。这个操作合理的延迟下界是多少？如果希望采样一批 256 个 token 呢？

<details>
<summary>点击此处查看答案。 </summary>


**答案：** 在 int8 下，参数将占用 30e9 字节；按照给定规格，每份 KV 缓存将占用 `100e3 * 8192 = 819MB`。我们有 16 颗芯片，每颗的带宽为 `8.2e11` 字节/秒，bf16 FLOPs/s 为 `1.97e14`。根据上述公式，由于批大小较小，预计步骤时间至少为 `(4 * 819e6 + 30e9) / (16 * 8.2e11) = 2.5 ms`。当批大小为 256 个 token 时，MLP 块将深入计算受限区间，因此步骤时间约为 `(256 * 819e6) / (16 * 8.2e11) + (2 * 256 * 30e9) / (16 * 1.97e14) = 21ms`。

</details>

可以看到，这里存在清晰的吞吐量与延迟权衡。小批次速度快，但硬件利用率不高；大批次速度慢，但效率高。下面是针对一些较早的 PaLM 模型计算出的延迟—吞吐量 Pareto 前沿（来自 [ESTI 论文](https://arxiv.org/pdf/2211.05102)[[esti]](../#ref-esti)）：

![图：若干 PaLM 模型的成本（可理解为吞吐量）与延迟的 Pareto 前沿。请注意，芯片数（C）和批大小（B）会让你沿 Pareto 前沿移动；唯一例外是绿色点（PaLM 540B 的 C:32 B:16），其可用内存不足以支持合适的批大小，导致吞吐量受损。还要注意，在批大小超过 240 左右之后，吞吐量总体上趋于平缓。int8 权重能提供更优的延迟—吞吐量 Pareto 最优点，但不能提高最大吞吐量。](/images/scaling-book/img/latency-cost.png)

我们不仅会以批大小为旋钮在延迟与吞吐量之间权衡；如果发现受到 HBM 限制，也可能更偏好较大拓扑，因为它能容纳更大的批次。[下一节](../08-llama3-inference/#serving-llama-3-70b-on-tpus)会对此作更详细的探讨。

**要点：** 如果关心生成吞吐量，就应使用尽可能大的每芯片批大小。任何超过 TPU 算术强度（$B_\text{crit}$，通常为 120 或 240）的每芯片批大小，都能最大化吞吐量。为做到这一点，可能需要增大拓扑。更小的批大小则能以牺牲吞吐量为代价改善延迟。

<details>
<summary>从硬件角度看，这里还有一些注意事项。点击查看细节。 </summary>


上述分析都相当理论化。实践中，我们往往看不到十分陡峭的 Roofline，原因有几个：

* 假设 HBM 读取能与 FLOPs 完美重叠并不现实，因为编译器（XLA）并非不会犯错。
* 对分片模型而言，XLA 也经常无法把模型分片矩阵乘法的 ICI 通信与 FLOPs 本身高效重叠，因此在线性层超过 $\text{BS}=32$ 时，往往就开始承受延迟损失。
* 大于理论 Roofline 的批大小仍会因重叠不完美而带来一些吞吐量改善，不过这个界仍是一条很好的经验法则。

</details>

<span id="what-about-memory"></span>

### 内存呢？

我们已经花了一些时间考察带宽和 FLOPs，但还没有讨论内存。由于出现了 KV 缓存这一新数据结构，推理时的内存图景大不相同。在本节中，选择一个真实模型（LLaMA 2-13B）来展示这些差异：

| 超参数         | 值  |
| ------------------ | ------ |
| L (num_layers)     | 40     |
| D (d_model)        | 5,120  |
| F (ffw_dimension)  | 13,824 |
| N (num_heads)      | 40     |
| K (num_kv_heads)   | 40     |
| H (qkv_dim)        | 128    |
| V (num_embeddings) | 32,000 |

推理期间是什么在占用内存？显然，首先是参数。把它们统计出来，有：

| 参数            | 公式                                                                                                          | 大小（字节）                                                |
| ---------------- | ---------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------- |
| FFW 参数       | d_model<sup>2</sup> x ffw_multiplier x 3（用于 SwiGLU 的门投影、上投影和下投影）x n_layers                                  | 5,120 x 5,120 x 2.7 x 3 x 40 = **8.5e9**                       |
| 词表参数     | 2（输入嵌入和输出嵌入）x n_embeddings x d_model                                                         | 2 x 32,000 x 5,120 = **0.3e9**                                 |
| 注意力参数 | [2（*q 和输出*）x d_model x n_heads x d_qkv + 2（*用于 k 和 v*）x d_model x n\_kv\_heads x d_qkv] x n_layers | (2 x 5,120 x 40 x 128 + 2 x 5,120 x 40 x 128) x 40 = **4.2e9** |

把这些参数相加，得到 8.5e9 + 4.2e9 + 0.3e9 = **总计 13e9 个参数**，与预期完全一致。如前几节所见，训练期间可能以 bfloat16 存储参数，并用 float32 保存优化器状态，这大约会占用 100GB 内存。与可能占用数 TB 的梯度检查点相比，这根本不值一提。

**推理有何不同？** 推理期间只存储一份参数，例如采用 bfloat16。这会占用 26GB——而实践中借助量化通常能做得好得多。此时没有优化器状态或梯度需要跟踪。由于不执行检查点操作（即不为反向传播保留激活），无论预填充[^ch7-4]还是生成，激活占用都可以忽略不计。如果预填充 8k 个 token，单个激活只占约 `8,192 x 5,120 x 2 bytes = 80MB` 内存。更长的预填充可以拆成许多次更小的前向传播，因此对更长上下文也不是问题。生成使用的 token 更少，所以激活同样可以忽略。

**主要区别是 KV 缓存**。它保存所有过往 token 的键投影和值投影，其大小只受允许的最大序列长度限制。$T$ 个 token 的总大小为

$$
\text{KV cache size} = 2 \cdot \text{bytes per float} \cdot H \cdot K \cdot L \cdot T
$$

其中 $H$ 是每个头的维度，$K$ 是 KV 头数量，$L$ 是层数；因同时存储键和值，所以还要乘以 2。

**它会非常迅速地膨胀**，即使批大小和上下文长度都不算大。对 LLaMA-13B 而言，bf16 下单个 8192 长度序列的 KV 缓存为

$$
8192\ (T) \times 40\ (K) \times 128\ (H) \times 40\ (L) \times 2\ (\text{bytes}) \times 2 = 6.7 \text{GB}
$$

**只要 4 份这样的缓存，就会超过参数的内存用量！** 需要说明的是，LLaMA 2 并未针对较长上下文下的 KV 缓存大小进行优化（情况并不总是如此糟糕，因为 $K$ 通常小得多，LLaMA-3 就是如此），但这个例子仍很有说明力。在内存或延迟估计中，不能忽略 KV 缓存。

<span id="modeling-throughput-and-latency-for-llama-2-13b"></span>

### 为 LLaMA 2-13B 的吞吐量与延迟建模

来看看在 8xTPU v5e 上以完美效率、使用不同批大小执行生成时会发生什么；批大小最大取到前面推导出的理论最大吞吐量临界值（240）。

| 批大小                        |      1 |      8 |     16 |     32 |     64 |    240 |
| :-------------------------------- | -----: | -----: | -----: | -----: | -----: | -----: |
| KV 缓存内存（GiB）             |    6.7 |   53.6 |  107.2 |  214.4 |  428.8 |   1608 |
| 总内存（GiB）                |   32.7 |   79.6 |  133.2 |  240.4 |  454.8 |   1634 |
| 理论步骤时间（ms）        |   4.98 |  12.13 |  20.30 |  36.65 |  69.33 | 249.09 |
| 理论吞吐量（tokens/s） | 200.61 | 659.30 | 787.99 | 873.21 | 923.13 | 963.53 |

8x TPU v5e 总共提供 128GiB HBM、6.5TiB/s HBM 带宽（每颗 0.82TiB/s）和 1600TF/s 算力。

对这个模型而言，增大批大小确实能提高吞吐量，但收益会迅速递减。批大小超过 16 就会发生 OOM，而要接近 240，则需要多一个数量级的内存。更大的拓扑可以改善延迟，但每颗芯片的吞吐量已经撞墙。

假设参数总数保持不变，但以某种神奇方式把 KV 缓存缩小 5 倍（比如采用 1:5 [GMQA](#tricks-for-improving-generation-throughput-and-latency)，即 40 个 Q 头共享 8 个 KV 头——更多细节见下一节）。

| 批大小                        |      1 |        8 |       16 |       32 |       64 |      240 |
| :-------------------------------- | -----: | -------: | -------: | -------: | -------: | -------: |
| KV 缓存内存（GiB）             |   1.34 |    10.72 |    21.44 |    42.88 |    85.76 |    321.6 |
| 总内存（GiB）                |  27.34 |    36.72 |    47.44 |    68.88 |   111.76 |    347.6 |
| 理论步骤时间（ms）        |   4.17 |     5.60 |     7.23 |    10.50 |    17.04 |    52.99 |
| 理论吞吐量（tokens/s） | 239.94 | 1,429.19 | 2,212.48 | 3,047.62 | 3,756.62 | 4,529.34 |

KV 缓存变小后，收益依然会递减，但每颗芯片的理论吞吐量可以一直扩展到批大小 240。此时能容纳大得多的批大小 64，而且所有批大小下的延迟也始终更低。延迟、最大吞吐量和最大批大小都得到显著改善！事实上，后续 LLaMA 世代采用的正是这一优化——LLaMA-3 8B 有 32 个查询头和 8 个 KV 头（[来源](https://huggingface.co/MaziyarPanahi/Llama-3-13B-Instruct-v0.1/blob/dfdeb40bdb2c149dfa399ea2be0d56eb120f0831/config.json)）。

**要点：** 除参数之外，KV 缓存的大小也会极大影响模型最终的推理性能。我们希望结合架构决策和运行时优化，将其控制在合理范围内。

<span id="tricks-for-improving-generation-throughput-and-latency"></span>

## 改善生成吞吐量和延迟的技巧

自最初的 [Attention is All You Need 论文](https://arxiv.org/abs/1706.03762)以来，人们已经开发出许多提高模型效率的技术，其中往往专门针对 KV 缓存。总体而言，更小的 KV 缓存能让我们更容易增大生成步骤的批大小和上下文长度而不损害延迟，也能让 Transformer 周边系统（例如请求缓存）更轻松。如果忽略对质量的影响，可以采用以下方法：

**分组多查询注意力（也称 GMQA、GQA）：** 可以减少 KV 头的数量，并让注意力机制中的许多 Q 头共享这些 KV 头。在极端情况下，甚至可以让所有 Q 头共享单个 KV 头。与纯 MHA 相比，这会按 Q:KV 比率相应缩小 KV 缓存，而且观察表明，模型性能对这种变化相对不敏感。

![](/images/scaling-book/img/gmqa.png)

这也会有效提高注意力计算的算术强度（参见[第 4 节](../04-transformers/#all-the-transformer-math-you-need-to-know)的问题 4）。

**混入一些局部注意力层：** 局部注意力把上下文限制在一个较小到中等规模的最大长度内。在训练和预填充时，这相当于把注意力矩阵掩码为一条对角带，而不是一个三角形。它实际上限制了局部层 KV 缓存的最大长度。如果在模型中把一些局部层与一些全局层混合使用，当上下文超过局部窗口时，KV 缓存大小就会大幅减小。

**跨层共享 KV：** 模型可以学习按照某种模式让不同层共享相同的 KV 缓存。这样做确实会缩小 KV 缓存，并在增大批大小、缓存、离线存储等方面带来好处；但共享的 KV 缓存可能需要从 HBM 多次读取，*所以它不一定能改善步骤时间。*

![左：多层纯全局注意力。右：全局注意力与局部注意力交错，并和相邻层共享 KV 缓存的一种示例模式。](/images/scaling-book/img/kv-sharing.png)

来源：[Character.ai 博客](https://research.character.ai/optimizing-inference/?ref=blog.character.ai)。

**量化：** 推理通常对参数和 KV 的精度不那么敏感。通过量化参数和 KV 缓存（例如量化为 int8、int4、`fp8` 等），可以节省二者的内存带宽，降低达到计算 Roofline 所需的批大小，并节省内存以便用更大的批大小运行。量化还有一个额外优点：即使模型训练时未采用量化，通常也能在训练后应用量化。

**使用不规则 HBM 读取和 Paged Attention：** 在上述计算中，我们为每份 KV 缓存分配了 8k 上下文，但通常不必从内存读取整个 KV 缓存——请求的长度分布差异很大，而且不会都用满模型的最大上下文，因此通常可以实现只读取 KV 缓存非填充部分的内核（例如 Flash Attention 的变体）。

Paged Attention[[paged]](../#ref-paged) 在此基础上进一步改进：它使用类似操作系统的页表存储 KV 缓存，基本避免了对 KV 缓存进行填充。这样会增加很多复杂性，但意味着每个批次只使用自身真正需要的内存。这是一项运行时优化，因此同样与架构无关。

![图：生成期间，单个 token（“forth”）会关注多个 KV 缓存块或页面。通过对 KV 缓存分页，可以避免加载或存储超出实际需要的内存。](/images/scaling-book/img/paged-attention.png)

来源：[PagedAttention 论文](https://arxiv.org/pdf/2309.06180)。

**全局视角：** 总的来说，与标准 MHA Transformer 相比，这些 KV 缓存优化可以把 KV 缓存缩小一个数量级以上，从而可能让 Transformer 的总体成本改善一个数量级。

<span id="distributing-inference-over-multiple-accelerators"></span>

## 将推理分布到多颗加速器上

到目前为止，我们一直含糊带过如何扩展到单颗芯片之外。沿用[第 5 节](../05-training/#how-to-parallelize-a-transformer-for-training)的思路，下面探讨可用的不同策略及其权衡。与往常一样，预填充和生成将分开考察。

<span id="prefill"></span>

### 预填充

从 Roofline 的角度看，**预填充与训练几乎完全相同**，几乎所有相同的技术和权衡都适用——模型（Megatron）并行、序列分片（上下文足够长时）、流水线，甚至完全分片数据并行（FSDP）都可行！只需把 KV 保留下来，以便之后执行生成。与训练一样，增加芯片数量能获得更多 FLOPs/s（可能降低 TTFT），但也会增加通信开销（可能降低每颗芯片的吞吐量）。

**预填充分片的一般规则：** 下面给出一组适用于预填充的通用规则。假设只对单个序列执行预填充（没有批维度）：

1. *模型分片：* 通常先采用一定程度的模型并行，直到变为 ICI 受限。正如[第 5 节](../05-training/#how-to-parallelize-a-transformer-for-training)所见，对 1 个轴而言，这一界约为 $F / 2200$（通常为 4—8 路分片）。
2. *序列并行：* 超过这一点后，采用序列并行（类似数据并行，但沿序列维度分片）。虽然序列并行会在注意力中引入一些额外通信，但在较长上下文下通常很少。与训练一样，可以重叠通信与计算（分别对 Megatron 使用集合矩阵乘法，对注意力使用环形注意力）。

**要点：** 预填充期间，几乎任何能在训练中工作的分片方式都能正常工作。先做模型并行直到 ICI 上界，再做序列并行。

<span id="generation"></span>

### 生成

生成比预填充更复杂。首先，获得较大批大小更困难，因为需要把许多请求合批。延迟目标也更低。这两点共同意味着，我们通常更受内存限制，也对通信开销更敏感，从而限制了分片策略：

1. **FSDP 不可行：** 因为从 HBM 向 MXU 加载参数和 KV 缓存时受内存限制，所以不希望通过 ICI 移动它们；ICI 比 HBM 慢几个数量级。*我们希望移动激活，而不是权重。* 这意味着类似 FSDP 的方法通常完全不适合生成。[^ch7-5]

2. **没有理由做数据并行：** 纯数据并行没有帮助，因为它会复制参数，却不能加快参数加载。更好的做法是直接启动模型的多个副本。[^ch7-6]

3. **没有序列，就没有序列分片。** 那就祝序列分片好运吧。

*因此，对稠密模型生成而言，基本只剩各种模型分片变体。* 与预填充一样，最简单的做法是采用普通模型并行（激活完全复制，MLP 的权重沿隐藏维度完全分片），直到 4—8 路时变为 ICI 受限。不过，因为我们往往受内存带宽限制，所以实际上可以超过这个上限来改善延迟！

**关于生成的 ICI 上界：** 训练期间希望达到计算受限，因此 Roofline 会考察 ICI 通信何时比 FLOPs 更耗时。然而，生成期间如果因参数加载而受内存带宽限制，就可以把模型分片扩展到这一点之外，以极小的吞吐量（tokens/sec/chip）代价改善延迟。更多模型分片会提供更多 HBM 来并行加载权重，而此时 FLOPs 并不重要。[^ch7-7] 下面看看在模型并行本身成为瓶颈之前能做到多大规模。

$$
\begin{aligned}T_\text{HBM comms} = \frac{2DF}{Y \cdot W_\text{hbm}} && T_\text{ICI comms} = \frac{2BD}{W_\text{ici}}\end{aligned}
$$

$$
T_\text{ICI comms} > T_\text{HBM comms} \rightarrow \frac{W_\text{hbm}}{W_\text{ici}} > \frac{F}{Y \cdot B} \rightarrow Y > F / (B \cdot \beta)
$$

其中 $\beta = W_\text{hbm} / W_\text{ici}$。对 TPU v5e 和 TPU v6e 而言，这个数通常约为 8。也就是说，例如当 $F$ 为 16,384、$B$ 为 32 时，理论上可以把模型并行扩展到 `16384 / (32 * 8) = 64` 路，而不会显著损害吞吐量。这假设 KV 缓存可以完全分成 64 路，而这很困难；下面会讨论这一点。

对注意力层，也采用 Megatron 风格沿头维度对注意力 $W_Q$ 和 $W_O$ 做模型分片。KV 权重相当小，当分片规模超过 $K$ 路时，复制它们往往比分片更便宜。

**要点：** 生成期间唯一的选择是各种模型并行变体。我们的目标是移动激活，而不是移动更大的 KV 缓存或参数。当批大小较大时，模型并行最多做到 FLOPs—ICI 上界（$F / \alpha$）。当批大小较小时，可以通过更多模型分片来改善延迟（只付出适度的吞吐量代价）。当希望模型分片的路数超过 KV 头数量时，还可以沿批维度对 KV 分片。

<span id="sharding-the-kv-cache"></span>

### 对 KV 缓存分片

**我们还有一个需要分片的数据结构——KV 缓存。** 同样，几乎总是希望避免复制缓存，因为它是注意力延迟的主要来源。为此，首先以 Megatron 方式沿头维度对 KV 分片。这样最多只能做 $K$ 路分片，因此对头数较少的模型，要尽可能沿头维度分片，再沿批维度分片，即 $\text{KV}[2, B_Z, S, K_Y, H]$。这样 KV 缓存就会完全分布开来。

![图：注意力机制的两种方案对比：（a）采用纯模型分片的多头注意力；（b）对 KV 缓存进行批分片的多查询注意力。请注意，我们需要两个额外的 AllToAll，把激活从模型分片转换为批分片，使其能够作用于 KV 缓存。](/images/scaling-book/img/esta-figure.png)

这样做的代价是每个注意力层执行两次 AllToAll：一次把 Q 激活转换为批分片，以便用批分片计算注意力；另一次把按批分片的注意力输出转回纯模型分片。

<details>
<summary>下面是完整算法！ </summary>


这里将完整写出同时在 $Y$ 和 $Z$ 上采用模型并行的注意力算法。很抱歉，$K$ 同时被用来表示键张量和 KV 头维度。令 $M=N/K$。

<div markdown=1 class="algorithm">

1. X[B, D] = ...（已有激活，来自上一层且未分片）
2. K[B<sub>Z</sub>, S, K<sub>Y</sub>, H], V[B<sub>Z</sub>, S, K<sub>Y</sub>, H] = ...（已有 KV 缓存，按批分片）
3. Q[B, N<sub>YZ</sub>, H] = X[B, D] \* W<sub>Q</sub>[D, N<sub>YZ</sub>, H]
4. Q[B<sub>Z</sub>, N<sub>Y</sub>, H] = **AllToAll**<sub>Z->B</sub>(Q[B, N<sub>YZ</sub>, H])
5. Q[B<sub>Z</sub>, K<sub>Y</sub>, M, H] = **Reshape**(Q[B<sub>Z</sub>, N<sub>Y</sub>, H])
6. O[B<sub>Z</sub>, S, K<sub>Y</sub>, M] = Q[B<sub>Z</sub>, K<sub>Y</sub>, M, H] \*<sub>H</sub> K[B<sub>Z</sub>, S, K<sub>Y</sub>, H]
7. O[B<sub>Z</sub>, S, K<sub>Y</sub>, M] = **Softmax**<sub>S</sub>(O[B<sub>Z</sub>, S, K<sub>Y</sub>, M])
8. O[B<sub>Z</sub>, K<sub>Y</sub>, M, H] = O[B<sub>Z</sub>, S, K<sub>Y</sub>, M] \*<sub>S</sub> V[B<sub>Z</sub>, S, K<sub>Y</sub>, H]
9. O[B, K<sub>Y</sub>, M<sub>Z</sub>, H] = **AllToAll**<sub>Z->M</sub>(O[B<sub>Z</sub>, K<sub>Y</sub>, M, H])
10. O[B, N<sub>YZ</sub>, H] = **Reshape**(O[B, K<sub>Y</sub>, M<sub>Z</sub>, H])
11. X[B, D] {U<sub>YZ</sub>} = W<sub>O</sub>[N<sub>YZ</sub>, H, D] \*<sub>N,H</sub> O[B, N<sub>YZ</sub>, H]
12. X[B, D] = **AllReduce**(X[B, D] { U<sub>YZ</sub>})

这相当复杂，但总体上可以看出它如何工作。新的通信成本适中，因为它们操作的是较小的激活；作为回报，我们大幅节省了加载（驻留的）KV 所需的内存带宽。

</div>

</details>

* **序列分片：** 如果批大小太小，或者上下文很长，可以沿序列维度对 KV 缓存分片。这里同样要为跨分片累积注意力支付集合通信成本。首先需要对 Q 激活执行 AllGather，再以类似 Flash Attention 的方式累积 KV。

<span id="designing-an-effective-inference-engine"></span>

## 设计高效的推理引擎

到目前为止，我们考察的是如何分别高效地优化和分片单独的预填充操作与生成操作。要真正高效地使用它们，就需要设计一个推理引擎，使这两种操作可以在延迟/吞吐量 Pareto 前沿上由我们选择的位置得到持续供给。

最简单的方法就是先运行一批预填充，再运行一批生成：

![图：在最简单的设置中，请求会被聚合，服务器交替运行一批预填充，并调用生成函数，直到所有序列都完成。](/images/scaling-book/img/batched-prefill.png)

这种方法易于实现，也是大多数代码库最先采用的推理设置，但它有多个缺点：

1. **延迟非常糟糕。** 预填充批大小和生成批大小被绑定在一起。预填充批大小很大时，首 token 延迟（TTFT）极差——必须完成所有预填充，用户才能看到任何 token。而批大小较小时，生成吞吐量又很糟糕。
2. **较短的生成会被较长的生成阻塞。** 许多序列会先于其他序列完成，导致生成期间出现空批槽，进一步损害生成吞吐量。随着批大小和生成长度增大，这个问题会更加严重。
3. **预填充会被填充。** 预填充会填充到最长序列，浪费大量计算。这个问题有解决方案，但 XLA 在历史上使跳过这些 FLOPs 变得相当困难。同样，批大小和预填充序列长度越大，问题就越严重。
4. **预填充和生成被迫共享同一种分片。** 二者运行在同一个切片上，这意味着它们使用相同的拓扑和分片方式（除非保留两份权重）；这通常不利于性能，例如生成希望采用更多模型分片。

因此，只建议在边缘应用中采用这种方法（这类应用通常只关心服务单个用户，并使用每字节 FLOPs 较低的硬件），或者在 Transformer 代码库生命周期早期快速迭代时采用它（因为实现简单）。

稍好一些的方法是以批大小 1 执行预填充（此时计算受限，但延迟合理），同时在生成期间把多个请求合批：

![](/images/scaling-book/img/interleaving.png)

这既避免了批量预填充浪费 TTFT，又能保持较高的生成吞吐量。我们称其为**交错式**配置，因为它会“交错”执行预填充步骤和生成步骤。对于评估这类以吞吐量为主要目标的批量生成应用，这种方法非常强大。可以把编排器配置为：只要任何生成槽位空出，就立刻优先执行预填充；如此即使生成批大小很大，也能保证较高利用率。因为预填充没有与其他请求合批，也就无需把它填充到最大长度。

主要缺点是：服务器执行预填充时，其他所有请求的生成都会暂停，因为全部计算资源都会被预填充占用。用户 A 的响应正在解码，却会被用户 B 正在执行的预填充阻塞。这意味着，虽然 TTFT 得到改善，token 生成仍会出现抖动，平均速度也很慢；对许多应用而言，这不是良好的用户体验——其他用户的预填充位于某个请求总体延迟的关键路径上。

为解决这个问题，我们将解码与预填充分离。尽管 Transformer 推理可以在一台服务器上完成，但从延迟角度看，通常最好在两组 TPU/GPU 上分别执行这两项不同任务。预填充服务器生成 KV 缓存，并经由网络将其发送给生成服务器；生成服务器把多份缓存合批，再为每份缓存生成 token。我们称之为<strong>“分离式”服务</strong>。

![](/images/scaling-book/img/disaggregation.png)

这样做有几个优点：

1. **大规模下的低延迟**：除非预填充容量不足，否则一个用户的请求永远不会被另一个用户的请求阻塞。请求应当立即完成预填充，然后被发送到生成服务器，并立刻插入生成缓冲区。如果预计会同时涌入许多请求，可以独立扩展预填充服务器数量，而无需同步扩展生成服务器数量，从而避免用户长时间滞留在预填充队列中。

2. **专门化：** 预填充和生成在延迟最优时的参数分片策略/硬件拓扑往往截然不同（例如，更多模型并行对生成有用，但对预填充无益）。强迫两种操作使用相同分片会同时损害二者的性能，而保留两组权重又会占用内存。此外，把预填充迁移到独立服务器后，它除了当前正在处理的那份 KV 缓存，不必持有任何其他 KV 缓存。这意味着会腾出大量内存，可用于历史缓存（见下一节）或优化预填充延迟。

一个缺点是，现在需要经由网络传送 KV 缓存。这通常可以接受，但也再次说明了缩小 KV 缓存的必要性。

**要点：** 对延迟敏感且高吞吐量的服务，通常必须把预填充和生成拆分到不同服务器；预填充以批大小 1 运行，生成则把许多并发请求合批。

<span id="continuous-batching"></span>

### 连续批处理

上面的第（2）个问题引出了**连续批处理**的概念。我们优化并编译：

* 一个预填充函数：处理可变的上下文长度，并把结果插入一个具有某个最大批大小和上下文长度/页数的 KV 缓冲区。
* 一个生成函数：接收 KV 缓存，并对当前所有活跃请求执行生成步骤。

随后，用一个编排器把这些函数组合起来：它对传入请求排队，根据可用生成槽位调用预填充和生成，处理历史缓存（见下一节），并以流式方式输出 token。

![](/images/scaling-book/img/continuous-batching.gif)

<span id="prefix-caching"></span>

### 前缀缓存

预填充成本高且计算受限（留给我们的余量较少），因此降低其成本的最佳方法之一就是少做一些。因为 LLM 是自回归的，查询 [“我”, “喜欢”, “狗”] 和 [“我”, “喜欢”, “猫”] 在前两个 token 上产生的 KV 缓存完全相同。这意味着，原则上，如果先计算“我喜欢狗”的缓存，再计算“我喜欢猫”的缓存，只需执行 1 / 3 的计算。复用缓存可以省下大部分工作。它在几种特定场景中尤其强大：

1. **聊天机器人**：大多数聊天机器人对话都由严格追加到既有内容的往返交流组成。这意味着，如果能保存每轮对话的 KV 缓存，就可以跳过除最新 token 之外的所有计算。
2. **少样本提示**：如果使用任何形式的少样本提示，就可以保存并免费复用。系统指令通常也采用这种形式。

这件事难做的唯一原因是内存限制。如前所见，KV 缓存很大（通常有数 GB），而要让缓存有用，就需要一直保留它们，直到后续查询到来。通常，预填充服务器上所有未使用的 HBM 都可用于本地缓存系统。此外，加速器的 CPU 主机通常有大量内存（例如，一台 8xTPUv5e 服务器有 128GiB HBM，但主机 DRAM 约有 450GiB）。这种内存比 HBM 慢得多——通常慢到无法执行生成步骤——但读取缓存已经足够快。实践中：

* 因为 KV 缓存位于处理初始请求的那组 TPU 本地，所以需要某种亲和性路由，确保后续查询到达同一副本。这可能给负载均衡带来问题。
* 更小的 KV 缓存（再次）很有帮助——它让我们能在相同空间内保存更多 KV 缓存，并缩短读取时间。
* KV 缓存及其查找可以很自然地存储在树或 trie 中，并可按 LRU 策略逐出。

![图：以 LRU trie 实现的 KV 前缀缓存。通过共享前缀，可以避免复制 KV 内存。](/images/scaling-book/img/prefix-caching-trie.png)

来源：[Character.ai 博客](https://research.character.ai/optimizing-inference/?ref=blog.character.ai)。

<span id="lets-look-at-an-implementation-jetstream"></span>

### 查看一个实现：JetStream

Google 开源了一个实现上述逻辑的库，名为 [JetStream](https://github.com/google/JetStream)。服务器包含一组“预填充引擎”和“生成引擎”，它们通常位于不同 TPU 切片上，并由单个控制器编排。预填充发生在“[预填充线程](https://github.com/AI-Hypercomputer/JetStream/blob/c0f83127c16d7861cacc560303a28404c6cbb24c/jetstream/core/orchestrator.py#L499)”中，生成发生在“[生成线程](https://github.com/AI-Hypercomputer/JetStream/blob/c0f83127c16d7861cacc560303a28404c6cbb24c/jetstream/core/orchestrator.py#L629)”中。还有一个“[传输线程](https://github.com/AI-Hypercomputer/JetStream/blob/c0f83127c16d7861cacc560303a28404c6cbb24c/jetstream/core/orchestrator.py#L592)”，负责协调把 KV 缓存从预填充切片复制到生成切片。

Engine 接口（在[这里](https://github.com/google/JetStream/blob/445f1aa8e857d0a09d72618e365daf80723bdf4c/jetstream/engine/engine_api.py#L138)实现）是任何 LLM 都必须提供的通用接口。关键方法包括：

* **prefill：** 接收一组输入 token 并生成 KV 缓存。
* **insert：** 接收一份 KV 缓存，并将其插入 generate 正在据以生成的一批 KV 缓存中。
* **generate：** 接收一组已合批的 KV 缓存，为每个批项生成一个 token，并针对每个 token，把单个 token 的 KV 缓存追加到相应解码状态。

我们还提供了 JetStream 的 PyTorch 版本，见[这里](https://github.com/google/jetstream-pytorch)。

<span id="worked-problems"></span>

## 练习题

本节将虚构一个基于 LLaMA-2 13B 的新模型。其详细信息如下：

| 超参数         | 值  |
| :----------------- | :----- |
| L (num_layers)     | 64     |
| D (d_model)        | 4,096  |
| F (ffw_dimension)  | 16,384 |
| N (num_heads)      | 32     |
| K (num_kv_heads)   | 8      |
| H (qkv_dim)        | 256    |
| V (num_embeddings) | 32,128 |

**问题 1：** 上述模型有多少参数？在 int8 下，其 KV 缓存每个 token 有多大？*可以假设输入投影矩阵和输出投影矩阵是共享的。*

<details>
<summary>点击此处查看答案。 </summary>


**参数数量：**

* MLP 参数数量：$L * D * F * 3$
* 注意力参数数量：$L * 2 * D * H * (N + K)$
* 词表参数：$D * V$（因为这些矩阵是共享的）

因此，总参数量为 $L * D * (3F + 2H * (N + K)) + D * V$。代入上述数值，得到 `64 * 4096 * (3*16384 + 2 * 256 * (32 + 8)) + 4096 * 32128 = 18.4e9`。所以，这个模型约有 184 亿个参数。

int8 下每个 token 的 KV 缓存为 $2 * L * K * H$，即每个 token `2 * 64 * 8 * 256 = 262kB`。

</details>

**问题 2：** 假设要在 TPUv5e 4x4 切片上部署这个模型，并且能在该拓扑上完全分片 KV 缓存。如果所有内容都采用 int8，并希望支持 128k 序列，能够容纳的最大批大小是多少？如果把 KV 头数量降到 1 呢？

<details>
<summary>点击此处查看答案。 </summary>


int8 下，每个 token 的 KV 缓存大小为 $2 \cdot L \cdot K \cdot H$，即 `2 * 64 * 8 * 256 = 262kB`。对于 128k 序列，这意味着每个批项占用 `262e3 * 128e3 = 33.5GB`。由于每颗 TPU 有 16GB HBM，把参数也算进去后，能够容纳的最大批大小为 `(16 * 16e9 - 18.4e9) / 33.5e9 = 7`。如果 $K=1$，则可以达到这个数的 8 倍，也就是约 56。

</details>

**问题 3：** 假设参数在 TPU v5e 4x4 切片上完全分片，把所有参数从 HBM 载入 MXU 需要多久？假设参数采用 int8。*这是单步延迟一个很好的下界。*

<details>
<summary>点击此处查看答案。 </summary>


总共有 18.4B 个参数，即 int8 下的 18.4e9 字节。每颗芯片的 HBM 带宽为 8.2e11，因此假设能充分利用 HBM 带宽，大约需要 `18e9 / (8.2e11 * 16) = 1.4ms`。

</details>

**问题 4：** 假设要在 TPUv5e 4x4 切片上用 int8 FLOPs 和 int8 参数/激活来部署这个模型。预填充和解码分别应如何分片？*提示：也许可以先回答以下问题：*

1. 4x4 上的 ICI 是什么样的？
2. 张量并行的 Roofline 上界是多少？
3. 如何对 KV 缓存分片？

采用这种分片时，生成的单步延迟大约是多少？

**问题 5：** 假设上述模型实际上是一个 MoE。MoE 模型实际上就是有 E 份 FFW 块副本的稠密模型。每个 token 会通过其中 k 个 FFW 块，并对这 `k` 个块取平均以生成输出。沿用上述设置，并令 `E=16`、`k=2`。

1. 它的总参数量和激活参数量分别是多少？*激活是指任意给定 token 实际使用的参数。*
2. 在 TPU v5e 上，要达到计算受限需要多大的批大小？
3. 每个 token 的 KV 缓存有多大？
4. 包含 T 个 token 的一次前向传播涉及多少 FLOPs？

<details>
<summary>点击此处查看答案。 </summary>


（1）作为 MoE，每个 MLP 块现在有 $3 * E * D * F$ 个参数，比稠密变体增加到 $E$ 倍。因此，它现在总共有 $L * D * (3EF + 2H * (N + K)) + D * V$ 个参数，即 `64 * 4096 * (3*16*16384 + 2 * 256 * (32 + 8)) + 4096 * 32128 = 212e9`，增加约 12 倍。对激活参数而言，激活的是 $k$ 份而不是 $E$ 份参数，总计 `64 * 4096 * (3*2*16384 + 2 * 256 * (32 + 8)) + 4096 * 32128 = 31.2e9`，相比稠密变体增加不到 2 倍。

（2）因为参数增加到 $E$ 倍，但 FLOPs 只增加到 $k$ 倍，HBM Roofline 会增加 $E/k$ 倍。这意味着在 TPU v5e 上大约需要 `240 * (16 / 2) = 1920` 个 token。

（3）KV 缓存大小保持不变，因为 MoE 特性不会改变注意力机制。

（4）仍然是 $2 \cdot \text{activated params} \cdot T$，因此为 $2 * \text{31.2e9} * T$。

</details>

**问题 6：** 对 MoE，可以采用“专家分片”，即沿网格的一个轴拆分专家。在标准记法中，第一个 FFW 权重的形状为 `[E, D, F]`，我们把它分片为 [E<sub>Z</sub>, D<sub>X</sub>, F<sub>Y</sub>]，其中 `X` 只在训练期间作为 FSDP 维度使用。假设要在 TPU v5e 上执行推理：

1. 对上述模型，在 Y=8、Z=16 的 TPU v5e 8x16 切片上，HBM 权重加载时间是多少？每颗 TPU 有多少可用 HBM？
2. 能够容纳模型的最小切片是什么？

**问题 7［二维模型分片］：** 这里将完整推导 [ESTI 论文](https://arxiv.org/pdf/2211.05102)所称的二维权重驻留分片。附录 B 会简要介绍它，但请先尝试完成这道题，看看能否自行推导数学关系。二维权重驻留分片的基本思想是：沿 $D$ 和 $F$ 两个轴对权重分片，使每个分块近似正方形。这样能减轻通信负载，并允许我们扩展得稍远一些。

下面是二维权重驻留算法：

<div markdown=1 class="algorithm">

1.  In[B, D<sub>X</sub>] = **AllGather**<sub>YZ</sub>(In[B, D<sub>XYZ</sub>])
2.  Tmp[B, F<sub>YZ</sub>] {U<sub>X</sub>} = In[B, D<sub>X</sub>] \*<sub>D</sub> W<sub>in</sub>[D<sub>X</sub>, F<sub>YZ</sub>]
3.  Tmp[B, F<sub>YZ</sub>] = **AllReduce**<sub>X</sub>(Tmp[B, F<sub>YZ</sub>] {U<sub>X</sub>})
4.  Out[B, D<sub>X</sub>] {U<sub>YZ</sub>} = Tmp[B, F<sub>YZ</sub>] \*<sub>F</sub> W<sub>out</sub>[F<sub>YZ</sub>, D<sub>X</sub>]
5.  Out[B, D<sub>XYZ</sub>] = **ReduceScatter**<sub>YZ</sub>(Out[B, D<sub>X</sub>] {U<sub>YZ</sub>})
</div>

你的目标是推导该算法的 $T_\text{math}$ 和 $T_\text{comms}$，并找出它在什么情况下会优于传统的三维模型分片。

<details>
<summary>点击此处查看答案！ </summary>


下面推导 $T_\text{math}$ 和 $T_\text{comms}$。所有 FLOPs 都完全分片，所以和之前一样，有 $T_\text{math} = 4BDF / (N \cdot C)$；但现在的通信时间为

$$
\begin{align*}
T_\text{2D comms} = \frac{2BD}{2X \cdot W_\text{ici}} + \frac{4BF}{YZ \cdot W_\text{ici}} + \frac{2BD}{2X \cdot W_\text{ici}} = \frac{2BD}{X \cdot W_\text{ici}} + \frac{4BF}{YZ \cdot W_\text{ici}}
\end{align*}
$$

这里要注意，AllReduce 的成本是两倍，并且要按照每个操作执行所跨的轴数来缩放通信。假设可以自由选择拓扑，并假设 $F=4D$（与 LLaMA-2 相同），那么通过一些基础微积分可知，$X$、$Y$ 和 $Z$ 的最优取值是 $X = \sqrt{N / 8}$、$YZ = \sqrt{8N}$，因此总通信时间为

$$
T_\text{2D comms} = \frac{2B}{W_\text{ici}} \left(\frac{D}{X} + \frac{8D}{YZ}\right) = \frac{\sqrt{128} BD}{\sqrt{N} \cdot W_\text{ici}} \approx \frac{11.3 BD}{\sqrt{N} \cdot W_\text{ici}}
$$

首先，直接沿用上面的结果，普通一维模型并行的通信时间为 $T_\text{model parallel comms} = 4BD / (3 \cdot W_\text{ici})$。那么，新方案的通信何时更少？有

$$
\begin{align*}
T_\text{model parallel comms} > T_\text{2D comms} \iff \frac{4BD}{3 \cdot W_\text{ici}} > \frac{\sqrt{128} BD}{\sqrt{N} \cdot W_\text{ici}} \\
\iff N > 128 \cdot \left(\frac{3}{4}\right)^2 = 72
\end{align*}
$$

对于一般的 $F$，可得这一条件为

$$
N > 32 \cdot \left(\frac{F}{D}\right) \cdot \left(\frac{3}{4}\right)^2
$$

这告诉我们，如果芯片数超过 72，使用新方案会更好。这个结果略显奇怪，因为以往往往在约 20 路张量并行时就发现自己受到 ICI 限制。但在这里，即使已经通信受限，总通信量仍会随着芯片总数增加而持续下降！这说明可以继续增加芯片、增大批大小、做更多参数扩展，同时仍看到延迟降低。

</details>

<span id="thats-all-for-part-7-for-part-8-with-a-look-at-how-we-might-serve-llama-3-on-tpus-click-here"></span>

### 第 7 部分到此结束！第 8 部分将探讨如何在 TPU 上部署 LLaMA 3 服务，请点击[这里](../08-llama3-inference/#serving-llama-3-70b-on-tpus)。

<span id="appendix"></span>

## 附录

<span id="appendix-a-how-real-is-the-batch-size--240-rule"></span>

### 附录 A：批大小 > 240 规则在现实中有多准确？

前面给出的简单规则——要达到计算受限，批大小必须超过 240 个 token——大致成立，但它忽略了 TPU 在其他操作没有用满全部可用 HBM 时预取权重的能力，例如执行设备间通信时。

下面是一张小型 Transformer 的实测层耗时图（单位为微秒）；其 d<sub>model</sub> 为 8192、d<sub>ff</sub> 为 32768，每层只有 2 次矩阵乘法。数据来自[这个 Colab 笔记本](https://colab.sandbox.google.com/drive/1_6krERgtolH7hbUIo7ewAMLlbA4fqEF8?usp=sharing)。可以看到，在批大小达到约 240 之前，步骤时间增长得非常缓慢，之后才开始线性增长。

![](/images/scaling-book/img/batch-scaling-latency.png)

下面是实际吞吐量，单位为 token / us。这张图相当清楚地支持了上述论点。这里每层约有 600M 参数，并以 4 路分片，因此预计最低延迟约为 365us。

![](/images/scaling-book/img/batch-scaling-throughput.png)

所以，至少在这个模型中，确实可以看到每个数据并行分片的吞吐量一直增长到约 BS240。

<span id="appendix-b-2d-weight-stationary-sharding"></span>

### 附录 B：二维权重驻留分片

随着拓扑增大，如果可以使用更高维的网格（例如 TPU 的网格），就能通过引入第二个分片轴，使用<strong>“二维权重分片”</strong>进一步改进。我们称之为<strong>“二维权重驻留”</strong>；[Efficiently Scaling Transformer Inference 论文](https://arxiv.org/abs/2211.05102)对此有更详细的描述。

在 Megatron 中，因为只沿隐藏维度 $F$ 分片，所以当一维分片的芯片数量变大时，它会显著小于 $E$（$d_\text{model}$ 维度）。这意味着在较大的批大小下，先应用 MLP 第一层，再沿隐藏维度执行一部分集合通信，可能更加经济。

![](/images/scaling-book/img/2d-weight-stationary.png)

此图展示了：

1. 一维权重驻留分片，也称纯 Megatron 分片：AllGather 之后激活被完全复制，权重则沿隐藏维度 F 完全分片。
2. 二维权重驻留分片：权重同时沿隐藏维度 F 和归约维度 E 分片，激活沿 E 维度分片。第一层之前在（yz）轴上执行 AllGather，随后在（x）轴上执行 ReduceScatter。

对注意力层而言，在芯片数量较少时，Megatron 风格分片也相对简单。不过，Megatron 分片沿 $n_\text{heads}$ 维度进行，这限制了可用的分片规模。把二维分片修改为适用于注意力（不再沿隐藏维度分片，而是沿 $n_\text{heads}$ 维度分片）之后，就能进一步扩展。

<span id="appendix-c-latency-bound-communications"></span>

### 附录 C：延迟受限通信

回顾一下，在[第 3 节](../03-sharding/#sharded-matrices-and-how-to-multiply-them)中，我们推导了在一维环上跨 X 颗芯片执行 AllGather、使每颗 TPU 上得到一个大小为 B 的张量所需的时间；链路的全双工带宽为 WICI，延迟为 Tmin。

$$
T_{total} = \max\left(\frac{T_{min} \cdot |X|}{2}, \frac{B}{W_{ICI}}\right)
$$

当 B 很大时，实际耗时相对恒定，因为向系统添加更多芯片时，完成操作所需的数据移动量与可用总带宽会同时扩展。

![](/images/scaling-book/img/all-gather.gif)

在面向延迟优化的推理中，移动的数据量相对较少，因此对激活执行的集合通信往往受延迟项限制（批大小较小时尤其如此）。只要数一数操作完成前需要经历多少跳，就能相当直观地理解延迟。

在 TPU 上，如果通信中依赖张量大小的部分每跳少于 1 微秒（一跳指两个相邻设备之间的通信），就可能受实际发起集合通信的固定开销限制。单向 ICI 带宽为 `4.5e10` 时，满足 $(\text{bytes} / n_\text{shards}) / 4.5e10 < 1e-6$，ICI 通信就会变得延迟受限。对 8 路 Megatron 分片而言，这发生在 `buffer_size < 360kB` 时。**这个值在推理中其实并没有那么小：** 当 `BS=16`、`D=8192` 且采用 int8 时，激活会占用 `16*8192=131kB`，因此已经是延迟受限的。

**要点：** 当 $\text{total bytes} < W_{ICI} \times 1e-6$ 时，通信会变为延迟受限。例如，在 $Y$ 上采用模型并行时，int8 下会在 $Y > BD / 45,000$ 时达到这一限制。

这里可以与计算 Roofline 作类比——我们正在承担某些小型操作的固定成本（通信对应延迟，矩阵乘法对应内存带宽）。

<span id="appendix-d-speculative-sampling"></span>

### 附录 D：推测采样

当我们*确实非常*关心端到端延迟时，还可以使用一种额外技巧，称为推测采样[[spec1]](../#ref-spec1)[[spec2]](../#ref-spec2)。回顾一下，通常我们会逐个从大型 Transformer 生成 token：

![](/images/scaling-book/img/spec-sampling1.png)

在推测采样中，我们使用一个更小、更便宜的模型生成 token，再用大模型检查结果。通过*贪心解码*最容易理解这一过程：

![](/images/scaling-book/img/spec-sampling2.png)

1. 从某个更小、更便宜的模型进行贪心采样。理想情况下，应使用一个经过训练、能与更大模型相匹配的模型，例如通过蒸馏得到；但它也可以简单到只使用 n-gram，或者在一个小型文本语料库中匹配 token。
2. 生成 K 个 token 后，使用大模型为到目前为止生成的所有 token 计算下一个 token 的 logits。
3. 因为采用贪心解码，只需检查小模型生成的 token 是否在所有可能 token 中概率最高。如果某个 token 错误，就取最长的正确前缀，用正确 token 替换第一个错误 token，再返回第（1）步。如果所有 token 都正确，则可以先用最后一个正确的 logit 再采样一个 token，然后返回第（1）步。

**为什么这能改善延迟？** 对每个 token 而言，这套方案仍需通过大模型执行与一次前向传播等价的 FLOPs；但由于可以把一组 token 合批，就能在一次前向传播中完成所有这些 FLOPs，并利用当前*并非*处于*计算受限*状态这一事实，免费为更多 token 评分。

按平均 FLOPs 计算，每个被接受 token 的成本都会更高（因为有些 token 会被拒绝，而且还必须调用草稿模型）；但我们从硬件中榨出了更多 FLOPs，而小模型成本很低，因此总体上仍然获益。多个步骤还会共享 KV 缓存加载，因此**对长上下文而言，推测解码也能提高吞吐量。** 由于所有内容都经过大模型检查，采样分布完全不会改变（不过非贪心情况下的确切轨迹会不同）。

传统上，推测解码依赖一个采样分布与目标模型相近的小模型，例如用 LLaMA-2 2B 服务于 LLaMA-2 70B，但这样的模型往往并不存在。即使存在，如果接受率较低，较小的草稿模型成本仍可能过高。作为替代，把草稿模型嵌入主模型可能很有帮助，例如在基础模型较靠后的某一层添加专用草稿头[[eagle]](../#ref-eagle)[[medusa]](../#ref-medusa)[[DeepSeek3]](../#ref-DeepSeek3)。由于这个头与主模型共享大部分参数，它运行得更快，也能更贴近主模型的采样分布。

对于常规自回归采样，token/s 与步骤时间直接对应。我们仍受制于这里“算术强度”一节给出的理论最小步骤时间（事实上，推测采样的步骤时间通常比常规自回归采样慢很多，但由于平均每步能得到不止 1 个 token，仍可获得高得多的 tokens/s）。

![图：此图展示了 Chinchilla（DeepMind 的一个 70B 模型）配合 4B 参数草稿模型（小模型）时的单步延迟与推测成功率。对 XSum（自然语言数据集）而言，理想的推测量是向前约 3—4 个 token；HumanEval（代码数据集）则更可预测，采用更激进的推测能获得收益。](/images/scaling-book/img/spec-sampling3.png)

**非贪心解码如何实现？** 这要复杂一些，但本质上归结为一种受 Metropolis-Hastings 启发的算法：从 logits 得到 $P_{\text{draft model}}(\text{chosen token})$ 和 $P_{\text{target model}}(\text{chosen token})$；如果这两个概率的比值小于某个阈值，就以一定概率拒绝所选 token。

这[两篇](https://arxiv.org/abs/2211.17192)[论文](https://arxiv.org/abs/2302.01318)同时推导出了这一方法，并给出了它在实践中如何工作的优秀示例。

**要点：** 推测采样是另一个强大的调节手段，可以用吞吐量换取更好的单 token 延迟。不过，当批大小受限时（例如硬件规模较小或 KV 缓存很大），它就会成为双赢方案。

[^ch7-1]: 从历史上看，即使从未接触推理，也能完成数量惊人的 Transformer 研究——基于打分的多项选择基准无需正确实现 KV 缓存或生成循环，就可以高效运行。这意味着，尤其在研究代码库中，推理代码路径往往存在许多容易取得的改进。
[^ch7-2]: 贯穿本节，你会注意到推理远不如训练宽容。通常可用 FLOPs 少得多，合批机会更少，而且对延迟敏感得多。KV 缓存也显著增加了推理的复杂性。
[^ch7-3]: 这里做了相当程度的简化，忽略了应用 softmax、掩码等操作时非矩阵乘法部分的 FLOPs。它们应当与计算或 HBM 读取重叠，但在某些 TPU 世代上，实现这种重叠并非易事。虽然这些细节不会改变主要结论——KV 缓存通常受内存限制——但仍值得关注。
[^ch7-4]: 尤其得益于 Flash Attention，它避免了把注意力矩阵具体化。
[^ch7-5]: 训练结束后意外没有关闭它，是造成数量级性能回退的一种简单且常见的方式。
[^ch7-6]: 这里的意思是，启动多台服务器，以较小批大小分别运行模型副本。模型层面的数据并行严格来说更差。
[^ch7-7]: 这里指 FLOPs 时间并不会限制我们，所以真正要担心的是 ICI 时间超过参数加载时间。
