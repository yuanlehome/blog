---
title: "在 TPU 上部署 LLaMA 3-70B 服务"
description: "让我们仔细研究如何在 TPU v5e 上部署 LLaMA 3-70B 模型。按照 Roofline 估算，部署不同模型的成本是多少？它们的 KV 缓存有多大？应该使用什么批大小？推理期间，参数和激活值如何分片？我们将通过一系列粗略估算，推导生产环境中的延迟与吞吐量。"
chapter: 8
order: 8
part: 2
partTitle: "Transformer"
sourcePath: "applied-inference.md"
sourceCommit: "44109cacac9c5a9809a81c68ae4d45d7d2632ea6"
---

<span id="serving-llama-3-70b-on-tpus"></span>

# 在 TPU 上部署 LLaMA 3-70B 服务

*本节将探讨部署 LLaMA-3 服务需要做什么，以及这件事能做到多高效。与上一节“应用”部分一样，请先尝试用纸笔自行推导答案，再查看解答！*

<span id="whats-the-llama-serving-story"></span>

## LLaMA 服务的整体情况如何？

先来回顾一下 LLaMA 3-70B 的结构（参见[第 6 节](../06-llama3-training/#training-llama-3-on-tpus)）：

| **超参数**              | **值** |
| --------------------------- | :-------: |
| $n_\text{layers}$ (L)     |    80     |
| $d_\text{model}$ (D)      |   8,192   |
| $d_{ff}$ (F)              |  28,672   |
| $n_\text{heads}$ (N)      |    64     |
| $n_\text{kv heads}$ (K)   |     8     |
| $d_\text{qkv}$ (H)        |    128    |
| $n_\text{embeddings}$ (V) |  128,256  |

先从一个简单的问题开始：**我们应该用什么硬件提供服务？** 答案基本上是：哪种硬件的每美元 FLOPs 最便宜，就用哪种。[^ch8-1] 因此，我们通常希望在 TPU v5e 上提供服务；它是目前专门用于推理的芯片（成本数据取自 2025 年 2 月的 [Google Cloud 定价](https://cloud.google.com/tpu/pricing)）：

| **TPU 类型** | **bfloat16 FLOPs/s** | **Google Cloud 美元 / 小时** | **FLOPs / 美元** |
| ------------ | :------------------: | :-------------------------: | :-----------: |
| H100         |        9.9e14        |            $10.8            |    3.3e17     |
| v5p          |       4.59e14        |            $4.2             |    3.9e17    |
| v5e          |       1.97e14        |            $1.2             |  **5.8e17**  |

每颗 TPU v5e 有 16GB HBM，这会要求我们相当激进地对模型进行分片。先来思考几个可能对我们很重要的基本量：

**问题：** LLaMA 3-70B 每个 token 的 KV 缓存有多大？*可以假设我们以 int8 存储它们。这个量决定了给定拓扑上可用的批大小。*

<details>
<summary>思考完后点击此处！ </summary>


LLaMA 3-70B 有 8 个 KV 头，因此每个 token 的大小为 `2 * K * H * L = 2 * 8 * 128 * 80 = 160kB`。

**请注意它有多大！** 如果序列长度为 32k token（这很常见），每个序列会占用 `160e3 * 32,768 = 5.3GB / sequence`。当 BS=240 时，总计为 1.3TB！由于每颗 TPU v5e 只有 16GB，我们至少需要约 `(70e9 + 1.3e12) / 16e9 = 86` 颗 TPU v5e 芯片，才能容纳这么多内存。还要注意，与 70GB 的模型参数相比，这个数有多大。

</details>

**问题：** 假设我们希望以批大小 32、序列长度 8192 来部署 L3 70B，并将所有内容（参数和 KV）都存为 int8。总共会使用多少内存？能够部署它的最小切片是什么？

<details>
<summary>答案 </summary>


因为 int8 下每个 token 的 KV 为 `160e3` 字节，所以 KV 总内存为 `160e3 * 8192 * 32 = 41.9e9` 字节。参数为 `70e9` 字节，因为每个参数占 1 字节。因此，总内存用量为 `41.9e9 + 70e9 = 112GB`。

可用的最小切片需要 `112e9 / 16e9 = 7` 颗 TPU；将其向上取整为合理的偶数规模，就是 TPU v5e `4x2`。这会非常紧张；把其他开销考虑进去后，我们可能无法真正放得下，因此至少可能需要 `4x4`（或者降低批大小）。

</details>

**问题：** 在 TPU v5e `4x2` 上采用上述批大小和量化方式时，每个解码步骤的延迟大约是多少？吞吐量（token / 秒 / 芯片）是多少？`4x4` 上又如何？*假设我们以 bfloat16 执行 FLOPs，并且所有内容都完全分片。*

<details>
<summary>答案 </summary>


可以调用上一节中的公式：

$$
\begin{aligned}
\text{Theoretical Step Time (General)} ={}& \underbrace{\frac{\text{Batch Size} \times \text{KV Cache Size}}{\text{Total Memory Bandwidth}}}_{\text{Attention (always bandwidth-bound)}} \\
&+ \underbrace{\max\left(\frac{2 \times \text{Batch Size} \times \text{Parameter Count}}{\text{Total FLOPs/s}}, \frac{\text{Parameter Size}}{\text{Total Memory Bandwidth}}\right)}_{\text{MLP (can be compute-bound)}}
\end{aligned}
$$

这里的临界批大小约为 120，因为参数为 int8，而 FLOPs 采用 bfloat16。我们也可以手动计算等式右侧的最大值，但本质上这个计算已经做过好几次了。**所以，无论对矩阵乘法还是 FLOPs 而言，我们都已深入内存带宽受限区间。**

如果严格只看内存带宽，步骤时间基本为 `(KV size + param size) / (8 * HBM bandwidth) = 112e9 / (8 * 8.2e11) = 17ms`。**因此，理论步骤时间约为 17ms。** 吞吐量为 `32 / .017 = 1882 tokens / sec`，即 `1882 / 8 = 235 tokens / sec / chip`。

这里有一个注意事项：要检查矩阵乘法是否可能受 ICI 限制。这里可以为它投入 2 个轴，因此理论上当 $Y > 2 * F / 2200 = 2 * 28672 / 2200 = 26$ 时才会受 ICI 限制，所以完全没问题！

如果改在 `4x4` 上运行，ICI 方面仍然没问题，因此延迟会降至 `17 / 2 = 8.5ms`，但每颗芯片的吞吐量保持不变。

</details>

<span id="thinking-about-throughput"></span>

### 思考吞吐量

让我们花一点时间单独思考吞吐量。优化吞吐量时，我们希望处于计算受限状态，也就是尽可能接近用满 TPU 的全部 MXU 容量。通常这意味着希望批大小尽可能大，以便完成尽可能多的工作。

**问题：** 在 TPU v5e 上使用 bfloat16 权重和激活时，批大小需要多大才能让矩阵乘法达到计算受限？如果权重采用 int8、但 FLOPs 以 bfloat16 执行呢？权重和 FLOPs 都采用 int8 又如何？

<details>
<summary>答案 </summary>


如第 7 节所述，对于任何满足 $B \ll D, F$ 的 bfloat16 矩阵乘法，都有

$$
\begin{equation*}
T_\text{math} > T_\text{comms} \leftrightarrow \frac{2BDF}{2DF} \geq \frac{\text{TPU bfloat16 FLOPs/s}}{\text{HBM bandwidth}} = 240
\end{equation*}
$$

当权重采用 int8 时，分母会少一个 2 倍因子，于是得到 $2BDF / DF = 2B > 240$，等价于 $B > 120$，也就是之前临界批大小的一半。这对我们非常有帮助！当权重和 FLOPs 都采用 int8 时，必须使用 TPU 的 int8 FLOPs/s 数值；它从 bfloat16 的 1.97e14 提升到 3.94e14，接近翻倍。这意味着我们又回到了起点，约为 $B > 240$。

int8 权重配合 bfloat16 FLOPs 的情况十分常见，因为对参数做无损量化通常比执行低精度算术更容易。

</details>

**问题：** 对 bfloat16、int8 和 int4（KV 与参数都采用相应类型）而言，在 8k 上下文下能够部署 LLaMA 3-70B 的最小 TPU v5e 拓扑分别是什么？*这个问题中可以把 KV 缓存视为小到可以忽略。*

<details>
<summary>答案 </summary>


这很简单！如果可以接受极小的批大小，那么唯一限制就是参数内存必须装入 HBM；也就是说，芯片数就是 `ceil(num_params * sizeof(dtype) / HBM per TPU)`，或者写成 `ceil(70e9 * sizeof(dtype) / 16e9)`，再向上取整到最近的合理拓扑（2 的某个倍数）：

| dtype | 参数大小 | KV 大小 / token（字节） | 最少 TPU v5e 数量 | 实际最小切片 | KV 缓存可用的剩余 HBM | 8k 下的 KV 缓存数量 |
| :---: | :--------: | :---------------------: | :----------: | :--------------: | :-------------------------: | :----------------: |
| bf16  |   140GB    |          324kB          |     8.75     |  4x4 = 16 颗芯片  |             116             |         43         |
| int8  |    70GB    |          162kB          |     4.38     |  4x2 = 8 颗芯片   |             58              |         43         |
| int4  |    35GB    |          81kB           |     2.81     |  2x2 = 4 颗芯片   |             29              |         43         |

这相当不错！它告诉我们，只要愿意，就能把 LLaMA 70B 放进 TPU v5e 2x2。只不过你会注意到 KV 缓存数量很小。那就是我们的批大小！这意味着 FLOPs 利用率会非常糟糕。为了把批大小推高到 240，我们会很乐意改用更大的拓扑。

</details>

**问题：** 假设使用这些拓扑能够容纳的最大批大小，那么每个生成步骤的延迟预计是多少？

<details>
<summary>答案 </summary>


这也很简单，因为我们选择的批大小恰好会用满全部 HBM！问题只是在于，把整颗 TPU v5e 所能容纳的字节载入 MXU 需要多久。它就是 `v5e HBM / v5e HBM memory bandwidth = 16GB / 8.2e11 = 19ms`，所以结果是**每步 19ms**。假设生成结果的中位长度为 512 个 token，则每次解码大约需要 9 秒。请注意，更小的批大小可以带来略低的延迟；例如，如果只考虑 int4 模型参数，最低延迟约为每步 10ms，因为此时 HBM 不再是满的。

</details>

**要点**：我们总能通过询问“把模型的所有参数从 HBM 载入 MXU 需要多久”来给出解码延迟的下界。当 KV 缓存较小时，可以把每一层理解为逐块载入权重、随后将其丢弃。除非使用很大的批大小或大量设备间通信，否则这通常是一个合理的界（误差在 1.5 倍以内）。批大小更大时，还需要对 KV 缓存加载建模，因为它会超过参数加载而成为主导因素。

同样，在计算受限区间（例如训练或大批量推理）中，可以使用 $\text{Total FLOPs} / (N \cdot C) = 2 \cdot \text{param count} \cdot B / (N \cdot C)$ 这一假设不存在通信的下界。

**问题：** 对上述每种情况而言，每颗芯片能达到多少吞吐量（以查询数 / 芯片表示）？*可以假设解码长度的中位数为 512 个 token。*

<details>
<summary>答案 </summary>


这是一个重要问题，因为它与每个 token 的成本完全相关。

按照对解码长度中位数的假设，吞吐量就是 $B / (\text{per-step latency} \cdot \text{median steps} \cdot N) \approx 43 / (0.019 * 512 * N)$。这大约为 $(4.42 / N)$ QPS，因此代入 $N$ 可得：

|  dtype   | QPS / 芯片 |
| :------: | :--------: |
| bfloat16 |    0.27    |
|   int8   |    0.55    |
|   int4   |    1.11    |

请注意，这个估计相当乐观，因为它完全忽略了前向传播的工作内存（分配给激活和注意力的内存）。使用 Flash Attention 时，这并非荒谬的假设，但也不现实。真实数值很可能只有这里的一半左右。要获得绝对最高的吞吐量，我们可能需要把芯片数量增加到两倍以上，并显著增大批大小。

</details>

**问题：** 如果把上述每个示例的拓扑规模翻倍，峰值吞吐量会如何变化？

<details>
<summary>答案 </summary>


如果在 bfloat16 下使用 4x8 切片，就会剩余 372GB 可供 KV 缓存使用，从而能把批大小增至 140。由于步骤时间保持不变，吞吐量将为 `14.39 / num_chips`，即：

|       dtype       | QPS / 芯片 |
| :---------------: | :--------: |
| bfloat16（在 4x8 上） |    0.44    |
|   int8（在 4x4 上）   |    0.90    |
|   int4（在 2x4 上）   |    1.80    |

进一步增加规模还会带来更大的收益！最重要的结论是：如果受到 KV 缓存大小限制，那么**最小拓扑并不总是性能最高的拓扑**。

</details>

**问题：** 现在深入研究分片问题。假设希望在 TPU v5e 4x8 上用 bfloat16 提供服务，那么生成期间会如何在 TPU v5e 4x8 上对模型分片？能否避免受到通信限制？

<details>
<summary>答案 </summary>


如上一节所述，生成期间真正可用的分片选择只有一种：模型并行。在受到通信限制之前，最多可以做多少模型并行？如上一节讨论的那样，模型大致会在下式成立时变得通信受限：

$$
Y > \frac{F \cdot M_Y}{2200}
$$

对 LLaMA 3-70B 而言，`F = 28,672`；因此，如果沿 2 个轴进行模型分片，大约得到 $Y = 28672 \cdot 2 / 2200 = 26$。所以总体上可扩展至约 16 颗芯片而不受通信限制，这允许使用 `4x4`，但不允许使用 `4x8`。一般而言，因为我们无法完美重叠计算，所以就连这个估计也过于乐观。

**要点：我们实际上无法用纯模型并行在 4x8 上提供服务。** 这里最多只能用 4x2，或者*也许*能用 4x4。

不过，如前所述，当批大小较小时，往往可以采用更多模型并行而不显著损害吞吐量，因为模型受内存带宽限制，而不是受 FLOPs 限制。前面说过，这个值大约为 $Y=F / (8\cdot B)$；因此，如果批大小为 64，理论上在变为 ICI 受限之前，可以达到 `Y = 28,672 / (8 * 64) = 56` 路模型并行。为了对结果做合理性检查，可以考察单次矩阵乘法的 $T_\text{ici comms}$、$T_\text{hbm comms}$ 和 $T_\text{math}$。显然有：

$$
\begin{aligned}T_\text{ici comms} = \frac{2BD}{W_\text{ici}} && T_\text{hbm comms} = \frac{2DF}{Y \cdot W_\text{hbm}} && T_\text{math} = \frac{2BDF}{Y \cdot C}\end{aligned}
$$

对 `4x8` 而言，$T_\text{ici comms}$ = `(2 * 64 * 8192) / 9e10 = 11us`，$T_\text{hbm comms}$ = `(2 * 8192 * 28,672) / (32 * 8.2e11) = 18us`，$T_\text{math}$ = `(2 * 64 * 8192 * 28,672) / (32 * 1.97e14) = 4us`，因此理论上仍然受 HBM 带宽限制，这很好！*请注意，从 `4x4` 扩展到 `4x8` 从吞吐量角度看可能没有帮助，但会降低延迟！*

如果考察 int8 和 int4 配置，就会发现它们*可以*用纯模型并行实现。因此，我们来到了量化带来实际优势的地方，而且这种优势超出了更快的 FLOPs：它使我们能在变为通信受限之前采用更大的批大小。**所以，这个故事的最终结论是：在 4x8 上无法达到峰值吞吐量，但对 int8 和 int4 配置而言，可以使用纯模型并行。**

</details>

**提示**：有用的模型并行最大规模取决于 $d_{ff}$，也取决于模型分片所跨的轴数。根据模型大小，这个最大值通常介于 8 和 32 之间。可以扩展到超过这个上限，以牺牲一些吞吐量为代价改善延迟。

<span id="what-about-prefill"></span>

### 预填充呢？

这里基本上一直忽略预填充，因为它简单得多。现在把几个概念放在一起，思考端到端的全局情况。

**问题：** 假设预填充期间达到 40% 的 FLOPs 利用率。在 16 颗 TPU v5e 芯片上，长度为 8192 的预填充需要多久？

<details>
<summary>答案 </summary>


在 8k token 下，我们显然处于计算受限状态，所以只需要从 FLOPs 的角度推理。已知模型有 `70e9` 个参数，因此每次前向传播使用 `2 * 70e9 * B` FLOPs。假设 MFU（FLOPs 利用率）为 40%，运行时间约为 `2 * 70e9 * 8192 / (16 * 1.97e14 * 0.4) = 0.91s`。与前面考察的数值相比，这其实相当可观！

</details>

**问题：** 假设预填充长度中位数为 8192 个 token，解码长度中位数为 4096 个 token。再假设生成批大小为 32。平均每个步骤会有多少个序列完成解码？平均每个步骤会从 KV 缓存中逐出多少个 token？

<details>
<summary>答案 </summary>


这个问题相当直接。因为解码长度中位数为 4096 个 token，所以一个序列大约每 1 / 4096 个 token 完成一次。给定批大小 32，这意味着每个步骤会逐出 `32 / 4096` 个序列。由于 KV 缓存长度约为 `8192 + 4096`，所以每个步骤会逐出 `32 * (8192 + 4096) / 4096 = 96` 个 token。通用公式为 $B * (P + G) / G$，其中 $P$ 和 $G$ 分别是预填充长度和生成长度。

</details>

**问题：** 假设采用分离式服务，预填充长度中位数为 8192，解码长度中位数为 512。采用上面针对 bfloat16 计算出的预填充和生成延迟。要让两类服务器都保持完全饱和，需要怎样的预填充服务器与生成服务器之比？

<details>
<summary>答案 </summary>


这个问题颇有意思。设 $P$ 为预填充服务器数量，$G$ 为生成服务器数量。总体来说，这是一个流水线问题：以 `P / prefill_latency` 的速率输入序列，以 `B * G / (generate_latency * median_decode_length)` 的速率消费序列。我们之前算得，每个预填充步骤为 `910ms`，批大小 43（这里就按 32 计）时每个解码步骤为 `19ms`。因此需要 `P / 0.91 = 32 * G / (0.019 * 512)`，也就是 `P = 3G`；换言之，预填充服务器数量大约要是生成服务器的 3 倍！

</details>

<span id="visualizing-the-latency-throughput-tradeoff"></span>

## 可视化延迟与吞吐量的权衡

继续以 LLaMA 70B 为例，实际看看生成期间不同批大小所对应的延迟和吞吐量。正如上一节针对 PaLM 模型展示的那样，这会给出一条吞吐量/延迟的 Pareto 前沿。我们假设采用 16 路张量并行，因为在 MLP 块中保持计算受限的前提下，这是可用规模的一个合理上界。这里使用 TPU v5e 4x4 拓扑。**滑块控制序列长度，因此可以观察更大的 KV 缓存所带来的影响。**

<div class="scaling-book-plotly" style="position: relative; width: 100%; aspect-ratio: 16 / 9;">
  <iframe src="../../images/scaling-book/plotly/pareto.html" title="延迟与吞吐量 Pareto 前沿交互图" loading="lazy" scrolling="no" style="position: absolute; inset: 0; width: 100%; height: 100%; border: 0;"></iframe>
</div>

* **请注意成本与延迟之间的权衡有多剧烈。** 以每 token 延迟翻倍为代价，可以让每 token 成本降低约 100 倍。此外，延迟范围可以从小批大小时的 5.5ms，一直到超大批大小时的 20ms。
* 请注意，在 2k 上下文下，吞吐量碰到 BS 120 的 Roofline 时，实际上会稳定在约 1 token / ms / 芯片（这里是 120，因为权重采用 int8，但 FLOPs 采用 bf16）。然而，随着序列长度增加，这个批大小不再能装入内存，因此永远无法达到完全饱和。
* 请注意，在吞吐量相同的情况下，大批大小的延迟要高出许多，因为 KV 加载（而不是参数加载）成为主导因素。

把成本和延迟的来源拆分为参数加载时间、KV 加载时间和 FLOPs 时间，就能更好地理解这一点。红色区域表示 MLP 块预计处于计算受限状态的范围。

<div class="scaling-book-plotly" style="position: relative; width: 100%; aspect-ratio: 16 / 9;">
  <iframe src="../../images/scaling-book/plotly/latency_breakdown_log.html" title="推理延迟构成交互图" loading="lazy" scrolling="no" style="position: absolute; inset: 0; width: 100%; height: 100%; border: 0;"></iframe>
</div>

这张图讲述了一个相当鲜明的故事。可以看到，起初参数加载占据了延迟的绝大部分；直到批大小变得足够大，FLOPs 和 KV 加载才变得更为显著。值得注意的是，在所有超过 2048 的序列长度下，花在 KV 缓存加载上的时间都比 FLOPs 更多！**所以，虽然增大批大小可以提高硬件利用率，但在长上下文下，KV 加载始终主导总步骤时间。**

**要点：** 对 LLaMA 3-70B 而言，在几乎所有这些配置中，我们都强烈受限于 KV 缓存内存带宽（也受 HBM 带宽限制）；这凸显出缩小 KV 缓存对于生成吞吐量有多么重要。还请注意，这里的延迟/吞吐量权衡依然极为剧烈。

<details>
<summary>实现这一计算的代码很简单。 </summary>


下面是计算这些 Roofline 的代码：

```py
import numpy as np

num_chips = 16  # we fix 16 as the amount of total model parallelism we do
bytes_per_param = 1  # int8 means 1 byte per param
param_count = 70e9
param_size = bytes_per_param * param_count
sequence_length = 8192  # can vary this

hbm_bandwidth = 8.20E+11  # v5e
flops = 1.97E+14  # v5e

def kv_cache_size(bs):
    return 2 * bs * 128 * 8 * 80

def min_topology(bytes):
    return 2 ** np.ceil(np.log2(bytes / 16e9))

def get_max_batch_size(
    num_chips: int,
    sequence_length: int,
    param_size: float,
) -> int:
  batch_sizes = np.arange(1, 1024, 4)
  kv_sizes = kv_cache_size(sequence_length * batch_sizes)
  required_chips = min_topology(kv_sizes + param_size)
  max_idx = np.where(required_chips <= num_chips)[0][-1]
  return max_idx

max_idx = get_max_batch_size(
    num_chips=num_chips,
    sequence_length=sequence_length,
    param_size=param_size,
)  # get the largest batch size that can fit
batch_sizes = np.arange(1, 512, 1)[:max_idx]
kv_sizes = kv_cache_size(sequence_length * batch_sizes)

kv_comms_time = kv_sizes / (num_chips * hbm_bandwidth)

param_comms_time = param_size / (num_chips * hbm_bandwidth)
param_comms_time = np.asarray([param_comms_time] * batch_sizes.shape[0])

flops_time = 2 * param_size * batch_sizes / (num_chips * flops)  # roughly true in a 2ND sense

mlp_time = np.maximum(flops_time, param_comms_time)
attn_time = kv_comms_time  # always bandwidth-bound for generate

latency = 1000 * (mlp_time + attn_time)
throughput = batch_sizes / (latency * num_chips)
```

请注意，我们非常明确地把延迟拆分为两个来源：KV 加载与参数加载；而延迟受 FLOPs 或通信中的较大者所限制。

</details>

<span id="worked-problems"></span>

## 练习题

下面是几道练习题。其中一些重复了上面已经完整推导过的内容，但可能有助于学习。

**问题 1：** LLaMA 3-405B 的每次前向传播对每个 token 使用多少 FLOPs？假设处于计算受限状态，在 TPU v5e 的 N 颗芯片上，单次前向传播的时间下界是多少？如果受通信限制呢？*忽略该模型无法装入单颗芯片这一事实。*

**问题 2：** 假设希望用 int8 权重和 int8 KV 缓存，以 BS240 部署 LLaMA 3-8B。（a）模型参数、（b）KV 缓存以及（c）峰值工作激活（大致）分别使用多少字节？能够运行它的最小拓扑是什么？

**问题 3：** 你会如何在 TPU v5e 上部署 LLaMA 3-405B？假设使用 int8 权重和 bfloat16 FLOPs。再假设有一个严格的 15ms / token 上限，能够达到的最高吞吐量配置是什么？理论最小步骤时间是多少？

<span id="thats-all-for-part-8-for-part-9-with-a-deep-dive-into-xla-and-tpu-profiling-click-here"></span>

### 第 8 部分到此结束！第 9 部分将深入探讨 XLA 和 TPU 性能剖析，请点击[这里](../09-profiling/#how-to-profile-tpu-programs)。

[^ch8-1]: 这并非总是成立；有时关键因素不是 FLOPs，而是更大的 HBM 或 ICI 带宽，但它是一个很好的经验法则。
