---
title: "在 TPU 上训练 LLaMA 3"
description: "让我们运用上一章学到的知识，仔细研究如何在 TPU v5p 上训练 LLaMA 3 模型。这些模型有多大？不同配置下的训练成本有多高？它们如何分片？我们将通过一系列粗略估算，具体说明前面各章的结论如何映射到真实模型上。"
chapter: 6
order: 6
part: 2
partTitle: "Transformer"
sourcePath: "applied-training.md"
sourceCommit: "44109cacac9c5a9809a81c68ae4d45d7d2632ea6"
---

<span id="training-llama-3-on-tpus"></span>

# 在 TPU 上训练 LLaMA 3

_本节的目标，是把上一节的结果应用到一个非常实际的问题上：训练 LLaMA 3 模型家族（羊群）。与前几节不同，我们希望你自己完成其中大量工作。因此，我们隐藏了每一部分的答案，让你可以先自行尝试。拿起笔，动手算一遍吧！_

<span id="what-does-llama-3-look-like"></span>

### LLaMA 3 是什么样的？

LLaMA-3 模型家族[[llama3]](../#ref-llama3)包含 3 个主要模型：LLaMA 3 8B、70B 和 405B。我们将主要关注 70B，把 8B 和 405B 留给你在末尾的习题部分探索。下面是 LLaMA 3-70B 的架构，取自 LLaMA 的 [HuggingFace 页面](https://huggingface.co/meta-llama/Meta-Llama-3-70B/blob/main/config.json)。

| **超参数**              | **值** |
| --------------------------- | --------- |
| $n_\text{layers}$ (L)     | 80        |
| $d_\text{model}$ (D)      | 8,192     |
| $d_{ff}$ (F)              | 28,672    |
| $n_\text{heads}$ (N)      | 64        |
| $n_\text{kv\_heads}$ (K)   | 8         |
| $d_\text{qkv}$ (H)        | 128       |
| $n_\text{embeddings}$ (V) | 128,256   |

为了强调这些信息有多么容易找到，下面直接给出配置本身及其对应关系：

![](/images/scaling-book/img/llama-json.png)

_为许多不同的开源 LLM 制作一张包含这些数值的大表很有用，这样就能快速比较它们所做的设计选择。_

<span id="counting-parameters-and-flops"></span>

### 计算参数量与 FLOPs

<strong>问题：</strong>根据这张表，能否算出 LLaMA 3-70B 的参数量？🤫 让我们应用[第 4 节](../04-transformers/#all-the-transformer-math-you-need-to-know)的内容，看看能不能得到 70B！

| 参数            | 公式                                                                                                                                           | 数量                                                        |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------ |
| FFW 参数       | d_model * d_ff * 3（分别用于 SwiGLU 的门控、升维和降维投影）* n_layers                                                                                         | 8,192 * 8,192 * 3.5 * 3 * 80 = **56.3e9**                    |
| 词表参数     | 2（输入嵌入和输出嵌入）* n_embeddings * d_model                                                                                          | 2 * 128,256 * 8,192 = **2.1e9**                              |
| 注意力参数 | n_layers * [ 2（用于 q 嵌入和拼接后的输出投影）* d_model * n_heads * d_qkv + 2（用于 k 和 v）* d_model * n_kv_heads * d_qkv] | 80 * (2 * 8,192 * 64 * 128 + 2 * 8,192 * 8 * 128) = **12e9** |
|                  |                                                                                                                                                   | 56.3e9 + 2.1e9 + 12e9 = **70.4e9**                           |

很好！我们得到了预期的数值。你会注意到，与预期一致，FFW 参数完全主导了总参数量，不过注意力所占的部分也不可忽略。

**要点**：MLP 块中的 3 个大型权重矩阵远大于 Transformer 中的所有其他数组，因此在推理模型内存或 FLOPs 时，通常几乎可以忽略其他所有参数。对于 LLaMA 3-70B，它们占 700 亿参数中的 560 亿。

现在来看看 FLOPs！*请记住[第 4 节](../04-transformers/#all-the-transformer-math-you-need-to-know)中关于训练的一般规则。*

<strong>问题：</strong>LLaMA-3 在每个训练步中，对每个 token 会执行多少 FLOPs？_这有助于确定整个训练过程的成本。_

<details>
<summary>思考之后，点击此处查看答案！ </summary>


**答案**：如[第 4 节](../04-transformers/#all-the-transformer-math-you-need-to-know)所示，每个 token 大约执行 $6 \cdot \text{param count}$ FLOPs，所以这里约为 `6 * 70e9 = 4.2e11` FLOPs / token。也就是每个 token 每步大约半个 TFLOP。假设我们受计算限制，并且 FLOPs 利用率完美，那么在单块 TPU v5p 芯片上应当大约需要 `4.2e11 / 4.59E+14 = 1ms`。

</details>

<strong>问题：</strong>LLaMA 3 的训练数据约有 15 万亿个 token。总共需要多少 FLOPs？

<details>
<summary>思考之后，点击此处查看答案！ </summary>


**答案**：很简单，总量就是 `4.2e11 * 15e12 = 6.3e24 FLOPs`。即 6.3 yottaFLOPs。真不少！如果只用一个 TPU，需要 `6.3e24 / 4.59E+14 = 435 years`。这也同样很久！

</details>

<strong>问题：</strong>假设要在一个完整的 TPU v5p pod 上训练，它包含 16x20x28 = 8960 块芯片。如果使用 bfloat16、MFU 为 40%，并假设计算受限，训练需要多长时间？

<details>
<summary>思考之后，点击此处查看答案！ </summary>


<strong>答案</strong>：我们知道，每块 TPU v5p 每秒可以执行 4.59e14 FLOPs。在 40% MFU 下，大约需要 `T = 6.3e24 / (8960 * 4.59e14 * 0.4) = 3.8e6 seconds`。<strong>也就是大约 44 天！</strong>假设确实能达到 40% MFU，这个时间相当合理。

</details>

<strong>问题：</strong>LLaMA 3-70B 预训练时的批大小约为 400 万个 token。使用这个批大小进行训练，最少需要多少个 TPU？_你可以假设参数采用 bfloat16、优化器状态采用 float32，并且每层设置 4 个梯度检查点。_

<details>
<summary>思考之后，点击此处查看答案！ </summary>


**答案**：这个问题主要是在问内存使用量，因为它是可用算力的唯一硬性约束。训练期间，HBM 主要有三种用途：模型参数、优化器状态和梯度检查点。如果假设权重采用 bfloat16、优化器状态采用 float32，并使用一种*非常*保守的梯度检查点方案（每层 4 次），则有：

| 项目 | 计算 | 估算内存 |
| --- | ---: | ---: |
| **参数** | 2 * 70GB | ~140GB |
| **优化器状态** | 8 * 70GB | ~560GB |
| **梯度检查点** | 2 * 8192 * 4e6 * 4 * 80 | ~20.9TB |
| **总计**                |                         | ~21.6TB |

这里的总量约为 21.6TB。你会发现，即使使用非常保守的检查点方案，梯度检查点仍然强烈主导内存用量。严格说来，我们可以减少到每层 1 个检查点，也可以进行微批处理，但以上数值是个合理的估计。按这些假设，每块 TPU v5p 有 96GB HBM，因此需要 `21.6e12 / 96e9 = 225` 块 TPU。其实并不算多！

<em>为什么不这样做？</em>因为那会需要 `44 days * 8960 / 225 = 1752 days` 才能完成训练，接近五年。<strong>这可太久了。</strong>尽管如此，它仍清楚地说明：使用这些大型集群，并不是因为受到内存限制，而是因为我们需要额外的 FLOPs。

</details>

<strong>问题：</strong>在与上一题相同的假设下，如果使用 8960 块 TPU v5p 芯片，每块芯片会使用多少内存？

<details>
<summary>思考之后，点击此处查看答案！ </summary>


**答案**：总内存仍约为 21.6TB，所以每块芯片会使用约 2.4GB，基本可以忽略不计。即使使用激进得多的检查点方案，例如每层 12 个检查点，每块芯片仍然只会用到 8GB。在这样的训练规模下，我们离内存受限还远得很。

</details>

**要点**：从技术上讲，即使是非常大的模型，也可以在非常小的拓扑上训练，但代价是可能要花很长时间。算出一次训练运行的总 FLOPs 后，我们就能根据适中的 MFU 和已知拓扑，粗略估计训练时间。

<span id="how-to-shard-llama-3-70b-for-training"></span>

### 如何为训练分片 LLaMA 3-70B

继续沿用上面的设置：假设要在包含 8960 块芯片的 TPU v5p pod 上，以 400 万个 token 的批大小（每批 1024 个长度为 4096 的序列）训练 LLaMA 3-70B。下面讨论这个模型的最佳分片策略。

<strong>问题：</strong>在上述假设下，能否只使用 FSDP 训练模型？首先，假设不能使用任何序列／上下文并行。_这应该是你最先想到的方案，因为它很简单，而且如果可行，就不会引入额外通信。_

<details>
<summary>思考之后，点击此处查看答案！ </summary>


**答案**：这个答案会有点咬文嚼字。如上所述，LLaMA 3-70B 的初始训练使用长度为 4K 的序列，因此 400 万个 token 的批大小对应 1024 的*序列批大小*。这意味着纯数据并行／FSDP 实际上最多只能扩展到 1024 块芯片，_因为可供数据并行拆分的序列就只有这么多_。所以，如果把问题简单理解为“没有额外通信的完整数据并行是否可行”，答案是否定的。下一题将回答一个没那么咬文嚼字的版本。

</details>

<strong>问题：</strong>放宽“不做任何序列分片”的要求。如果允许同时沿批轴<em>和</em>序列轴执行 FSDP，能否只使用 FSDP 在 8960 块芯片上训练 LLaMA 3-70B？

<details>
<summary>思考之后，点击此处查看答案！ </summary>


**答案**：现在也允许进行序列／上下文并行，就能扩展到大得多的规模。首先计算每设备批大小。如果进行 8960 路 FSDP，每个 TPU 的批大小最终为 `4 * 1024 * 1024 / 8960 = 468 tokens`。根据上一节，当 $\text{per device batch size} < 2550 / M_X$ 时，FSDP 会受到 ICI 限制。由于这里使用完整的三维 pod，可以分配 3 个轴，所以其下限为 850，而 468 远低于该值。**因此答案是否定的，即使使用 3 个轴也不行。我们会明确地受通信限制。**

</details>

<strong>问题：</strong>现在来看混合张量并行与 FSDP。是否存在某种组合，可以让我们保持计算受限？如果有，应分别使用多大程度的 FSDP 和张量并行？

<details>
<summary>思考之后，点击此处查看答案！ </summary>


**答案**：首先检查这是否可行。我们知道，如果每芯片批大小小于 $2550^2 / 2F = 113$，就会受通信限制。如上所见，我们略高于这个数值。所以太好了！接下来使用公式选择最优的 FSDP 程度：

$$
X_{opt} = \sqrt{\frac{2BN}{F}} = \sqrt{\frac{2 \cdot 4.19e6 \cdot 8960}{28672}} = 1618
$$

将其舍入到合理的 2 的倍数，得到大约 2048 路 FSDP 和 4 路张量并行。效果应该会很好！

</details>

**要点**：我们可以在一个完整的 TPU v5p pod 上，以 400 万个 token 的批大小，混合使用数据并行（1024 路）、序列并行（2 路）和张量并行（4 路）训练 LLaMA-3，而不会受到通信限制。若尝试使用纯 FSDP 或 FSDP + 序列并行，则会受通信限制。上一节推导出的方程非常实用。

<span id="worked-problems"></span>

## 习题

<strong>问题 1［将 LLaMA 70B 扩展到更多芯片］：</strong>假设要以相同的批大小，在 4 个 pod 上训练 LLaMA 3-70B。应采用什么并行方案？会受计算限制还是通信限制？训练大约需要多长时间？*务必使用正确的 Roofline 上限。*

**问题 2［LLaMA 405B］：**

(a) 使用 LLaMA 3-405B 的[配置](https://huggingface.co/meta-llama/Llama-3.1-405B/blob/main/config.json)（这是门控模型，因此可能需要登录并申请访问权限才能查看），像上面一样制作一张包含所有关键超参数的表。该模型总共有多少参数？每个训练步需要多少 FLOPs？如果用 15T 个 token 训练，会执行多少 FLOPs？

(b) 假设要在 8 个 TPU v5p pod 上训练。应采用什么并行方案？训练需要多长时间？会受计算限制还是通信限制？

<span id="thats-all-for-section-6-for-section-7-about-transformer-inference-click-here"></span>

### 第 6 节到此结束。第 7 节介绍 Transformer 推理，请点击[此处](../07-inference/#all-about-transformer-inference)。
