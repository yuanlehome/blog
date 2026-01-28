---
lang: zh
translatedFrom: en
title: AWQ：面向设备端LLM压缩与加速的激活感知权重量化
slug: awqllm
date: '2026-01-28'
tags: []
status: published
source_url: 'https://arxiv.org/pdf/2306.00978'
source_author: arxiv.org
imported_at: '2026-01-28T20:11:06.250Z'
source:
  title: arxiv.org
  url: 'https://arxiv.org/pdf/2306.00978'
cover: /images/others/2306.00978/img_in_image_box_615_896_1082_1062.jpg
---

# AWQ：面向设备端LLM压缩与加速的激活感知权重量化

Ji Lin$^{*1}$Jiaming Tang$^{*1,2}$Haotian Tang$^{\dagger1}$Shang Yang$^{\dagger1}$Wei-Ming Chen$^{3}$Wei-Chen Wang$^{1}$Guangxuan Xiao$^{1}$Xingyu Dang$^{1,4}$Chuang Gan$^{5,6}$Song Han$^{1,3}$

<https://github.com/mit-han-lab/llm-awq>

## 摘要

大型语言模型（LLMs）已经改变了众多人工智能应用。设备端LLM正变得越来越重要：在边缘设备上本地运行LLM可以降低云计算成本并保护用户隐私。然而，巨大的模型尺寸和有限的硬件资源带来了显著的部署挑战。我们提出了激活感知权重量化（AWQ），一种面向硬件的LLM低比特仅权重量化方法。AWQ发现并非LLM中的所有权重都同等重要。仅保护1%的关键权重就能大幅减少量化误差。为了识别关键权重通道，我们应该参考激活分布，而非权重。为了避免硬件效率低下的混合精度量化，我们通过数学推导得出，放大关键通道可以减少量化误差。AWQ采用等效变换来放大关键权重通道以保护它们。缩放因子通过离线收集激活统计量确定。AWQ不依赖任何反向传播或重构，因此能泛化到不同领域和模态，而不会过拟合校准集。AWQ在各种语言建模和领域特定基准测试（编码和数学）上优于现有工作。得益于更好的泛化能力，它在指令调优的LM上实现了优异的量化性能，并首次在多模态LM上实现。除了AWQ，我们还实现了TinyChat，一个专为4位设备端LLM/VLM设计的高效灵活推理框架。通过内核融合和平台感知的权重打包，TinyChat在桌面和移动GPU上相比Huggingface FP16实现提供了超过3倍的加速。它还使得70B Llama-2模型在移动GPU上的部署成为可能。

## 1 引言

直接在边缘设备上部署大型语言模型（LLMs）至关重要。设备端使用消除了将数据发送到云服务器造成的延迟，并使LLM能够离线运行，这对于虚拟助手、聊天机器人和自动驾驶汽车等实时应用有益。与维护和扩展集中式云基础设施相关的运营成本也可以降低。设备端LLM还通过将敏感信息保留在本地来增强数据安全性，减少数据泄露的风险。LLMs基于Transformer架构（Vaswani等人，2017年），因其在各种基准测试中的出色表现而受到广泛关注（Brown等人，2020年；Zhang等人，2022年；Touvron等人，2022年）。

<div style="text-align: center;"><img src="imgs/img_in_image_box_615_896_1082_1062.jpg" alt="Image" width="38%" /></div>

<div style="text-align: center;">Figure 1. We introduce AWQ, a versatile weight quantization method for LLM. To implement AWQ, we developed TinyChat to deploy 4-bit quantized LLMs into various edge platforms, achieving a 3-4× performance boost compared to FP16. Notably, we've also manufactured a TinyChat computer, powered by TinyChat, which contains an NVIDIA Jetson Orin Nano with only 8GB of memory and 15W power consumption. Demo: https://youtu.be/z91a8DrfgEw.</div>

LLMs的低比特权重量化可以显著减少设备端LLM推理的内存占用，但很困难。量化感知训练（QAT）由于高训练成本而不高效，而后训练量化（PTQ）在低比特设置下遭受大的精度下降。最接近的工作是GPTQ（Frantar等人，2022年），它使用二阶信息进行误差补偿。然而，它可能在重构过程中过拟合校准集，扭曲在分布外领域学习到的特征（图8），这是有问题的，因为LLMs是通用模型。

在本文中，我们提出了激活感知权重量化（AWQ），一种面向硬件的LLM低比特仅权重量化方法。我们的方法基于观察：权重对LLMs的性能并非同等重要。存在一小部分（0.1%-1%）关键权重；跳过这些关键权重的量化将显著减少量化损失（表1）。为了找到关键权重通道，洞见是尽管我们进行仅权重量化，但应参考激活分布而非权重分布：对应较大激活幅度的权重通道更关键，因为它们处理更重要的特征。为了避免硬件效率低下的混合精度实现，我们分析了权重量化的误差，并推导出放大关键通道可以减少它们的相对量化误差（方程2）。遵循这一直觉，我们设计了一种每通道缩放方法，以自动搜索在完全权重量化下最小化量化误差的最佳缩放。AWQ不依赖任何反向传播或重构，因此能很好地保留LLMs在各种领域和模态上的泛化能力，而不会过拟合校准集。

为了实现AWQ，我们设计了TinyChat，一个高效的推理框架，将4位LLM的理论内存节省转化为实测加速。我们的框架通过即时反量化显著加速线性层。我们还利用高效的4位权重打包和内核融合来最小化推理开销（例如，中间DRAM访问和内核启动开销），从而更好地实现从权重量化到4位带来的加速，尽管计算机是字节对齐的。

实验表明，AWQ在不同模型家族（例如，LLaMA（Touvron等人，2023年a）、OPT（Zhang等人，2022年））和模型尺寸的各种任务上优于现有工作。得益于更好的泛化能力，它还在指令调优的LM（例如，Vicuna）上实现了良好的量化性能，并首次在多模态LM（OpenFlamingo（Awadalla等人，2023年））上实现。TinyChat进一步将$\sim4\times$以较低的内存占用实现实测加速。在桌面、笔记本和移动GPU上，我们一致观察到，与Huggingface的FP16实现相比，在多样化的LLM频谱中平均加速达到3.2-3.3倍。此外，它使得Llama-2-70B模型能够轻松部署在单个64GB内存的NVIDIA Jetson Orin上。它还在仅8GB内存的笔记本RTX 4070 GPU上，以30 tokens/秒的交互速度普及了130亿参数LLM。AWQ已被工业和开源社区广泛采用：HuggingFace Transformers、NVIDIA TensorRT-LLM、Microsoft DirectML、Google Vertex AI、Intel Neural Compressor、Amazon Sagemaker、AMD、FastChat、vLLM、LMDploy，并使得Falcon-180B可在单个H200 GPU上部署。

## 2 相关工作

模型量化方法。量化降低了深度学习模型的比特精度（Han等人，2016；Jacob等人，2018；Nagel等人，2019；Wang等人，2019；Nagel等人，2020；Lin等人，2020），这有助于减小模型大小并加速推理。量化技术通常分为两类：量化感知训练（QAT，依赖反向传播更新量化权重）（Bengio等人，2013；Gholami等人，2021；Nagel等人，2021；Choi等人，2018）和后训练量化（Jacob等人，2018；Nagel等人，2019；2020）（PTQ，通常无需训练）。QAT方法难以扩展到像LLM这样的大型模型。因此，人们通常使用PTQ方法来量化LLM。

LLM的量化。人们研究LLM量化的两种设置：（1）W8A8量化，其中激活和权重都被量化为INT8（Dettmers等人，2022；Xiao等人，2022；Yao等人，2022；Wei等人，2022a；2023）；（2）低比特仅权重量化（例如，W4A16），其中仅权重被量化为低比特整数（Frantar等人，2022；Dettmers & Zettlemoyer，2022；Sheng等人，2023；Park等人，2022）。我们在本工作中专注于第二种设置，因为它不仅降低了硬件门槛（需要更小的内存大小），还加速了token生成（缓解内存受限工作负载）。除了普通的最近舍入基线（RTN），GPTQ（Frantar等人，2022）是与我们工作最接近的。然而，GPTQ的重建过程会导致对校准集的过拟合问题，可能无法保留LLM在其他模态和领域的通用能力。它还需要重排序技巧才能适用于某些模型（例如，LLaMA-7B（Touvron等人，2023a）和OPT-66B（Zhang等人，2022））。除了为通用硬件设计的量化方法外，SpAtten（Wang等人，2020）设计了一种渐进方法，逐步增加softmax计算中使用的比特数。

对低比特量化LLM的系统支持。低比特量化LLM已成为降低推理成本的流行设置。

<div style="text-align: center;"><img src="imgs/img_in_image_box_112_131_1083_355.jpg" alt="Image" width="79%" /></div>

<div style="text-align: center;">Figure 2. We observe that we can find 1% of the salient weights in LLMs based on the activation distribution (middle). Keeping the salient weights in FP16 can significantly improve the quantized performance (PPL from 43.2 (left) to 13.0 (middle)), but the mixed-precision format is not hardware-efficient. We follow the activation-awareness principle and propose AWQ (right). AWQ performs per-channel scaling to protect the salient weights and reduce quantization error. We measure the perplexity of OPT-6.7B under INT3-g128 quantization.</div>

有一些系统支持来实现实际的加速。GPTQ（Frantar等人，2022）为OPT模型提供INT3内核，而GPTQ-for-LLaMA借助Triton（Tillet等人，2019）扩展了对INT4重排序量化的内核支持。FlexGen（Sheng等人，2023）、llama.cpp$^{*}$和exllama$^{\dagger}$执行分组INT4量化以减少I/O成本和卸载。FasterTransformer实现了FP16×INT4 GEMM用于仅权重的每张量量化，但不支持分组量化。LUT-GEMM（Park等人，2022）借助查找表在GPU CUDA核心上执行位运算。我们的并行工作，MLC-LLM（MLC-Team，2023）凭借强大的TVM（Chen等人，2018；Feng等人，2023）后端，在多个边缘CPU和GPU平台上提供了强劲的结果。

## 3 AWO：激活感知权重量化

量化将浮点数映射到低比特整数。它是减小LLM模型大小和推理成本的有效方法（Dettmers等人，2022；Frantar等人，2022；Yao等人，2022；Xiao等人，2022）。在本节中，我们首先提出一种仅权重量化方法，通过保护更多“重要”权重来提高精度，无需训练/回归。然后开发一种数据驱动方法来搜索减少量化误差的最优缩放（图2）。

### 3.1 通过保留1%显著权重改进LLM量化

我们观察到LLM的权重并非同等重要：有一小部分显著权重对LLM性能比其他权重重要得多。跳过这些显著权重的量化可以帮助弥补由于量化损失导致的性能下降，无需任何训练或回归（图2(b)）。为了验证这一想法，我们在表1中基准测试了跳过部分权重通道时量化LLM的性能。我们测量了INT3量化模型的性能，同时保持一定比例的权重通道为FP16。确定权重重要性的常用方法是查看其幅度或$L_{2}$范数（Han等人，2015；Frankle & Carbin，2018）。但我们发现跳过具有大范数的权重通道（即基于W的FP16%）并未显著改善量化性能，导致与随机选择类似的边际改进。有趣的是，基于激活幅度选择权重可以显著提高性能，尽管仅保持0.1%-1%的通道为FP16。我们假设幅度较大的输入特征通常更重要。保持相应权重为FP16可以保留这些特征，从而有助于更好的模型性能。

局限性：尽管将0.1%的权重保留为FP16可以在不明显增加模型大小（以总比特数衡量）的情况下提升量化性能，但这种混合精度数据类型会使系统实现变得困难。我们需要提出一种方法，在不实际将重要权重保留为FP16的情况下保护它们。

### 3.2 通过激活感知缩放保护显著权重

我们提出一种替代方法，通过逐通道缩放来减少显著权重的量化误差，这种方法不受硬件效率低下的影响。

## 分析量化误差

我们首先分析仅权重量化带来的误差。考虑一组/块权重w；线性操作可以写为y = wx，量化后的对应操作为$y = Q(\mathbf{w})\mathbf{x}$。具体来说，量化

<table border=1 style='margin: auto; word-wrap: break-word;'><tr><td rowspan="2">PPL  $\downarrow$</td><td rowspan="2">FP16</td><td rowspan="2">RTN (w3-g128)</td><td colspan="3">FP16% (based on act.)</td><td colspan="3">FP16% (based on W)</td><td colspan="3">FP16% (random)</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>0.1%</td><td style='text-align: center; word-wrap: break-word;'>1%</td><td style='text-align: center; word-wrap: break-word;'>3%</td><td style='text-align: center; word-wrap: break-word;'>0.1%</td><td style='text-align: center; word-wrap: break-word;'>1%</td><td style='text-align: center; word-wrap: break-word;'>3%</td><td style='text-align: center; word-wrap: break-word;'>0.1%</td><td style='text-align: center; word-wrap: break-word;'>1%</td><td style='text-align: center; word-wrap: break-word;'>3%</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>OPT-1.3B</td><td style='text-align: center; word-wrap: break-word;'>14.62</td><td style='text-align: center; word-wrap: break-word;'>119.00</td><td style='text-align: center; word-wrap: break-word;'>25.03</td><td style='text-align: center; word-wrap: break-word;'>16.91</td><td style='text-align: center; word-wrap: break-word;'>16.68</td><td style='text-align: center; word-wrap: break-word;'>108.71</td><td style='text-align: center; word-wrap: break-word;'>98.55</td><td style='text-align: center; word-wrap: break-word;'>98.08</td><td style='text-align: center; word-wrap: break-word;'>119.76</td><td style='text-align: center; word-wrap: break-word;'>109.38</td><td style='text-align: center; word-wrap: break-word;'>61.49</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>OPT-6.7B</td><td style='text-align: center; word-wrap: break-word;'>10.86</td><td style='text-align: center; word-wrap: break-word;'>23.54</td><td style='text-align: center; word-wrap: break-word;'>11.58</td><td style='text-align: center; word-wrap: break-word;'>11.39</td><td style='text-align: center; word-wrap: break-word;'>11.36</td><td style='text-align: center; word-wrap: break-word;'>23.41</td><td style='text-align: center; word-wrap: break-word;'>22.37</td><td style='text-align: center; word-wrap: break-word;'>22.45</td><td style='text-align: center; word-wrap: break-word;'>23.54</td><td style='text-align: center; word-wrap: break-word;'>24.23</td><td style='text-align: center; word-wrap: break-word;'>24.22</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>OPT-13B</td><td style='text-align: center; word-wrap: break-word;'>10.13</td><td style='text-align: center; word-wrap: break-word;'>46.04</td><td style='text-align: center; word-wrap: break-word;'>10.51</td><td style='text-align: center; word-wrap: break-word;'>10.43</td><td style='text-align: center; word-wrap: break-word;'>10.42</td><td style='text-align: center; word-wrap: break-word;'>46.07</td><td style='text-align: center; word-wrap: break-word;'>48.96</td><td style='text-align: center; word-wrap: break-word;'>54.49</td><td style='text-align: center; word-wrap: break-word;'>44.87</td><td style='text-align: center; word-wrap: break-word;'>42.00</td><td style='text-align: center; word-wrap: break-word;'>39.71</td></tr></table>

<div style="text-align: center;">Table 1. Keeping a small fraction of weights (0.1%-1%) in FP16 significantly improves the performance of the quantized models over round-to-nearest (RTN). It is only effective when we select the important weights in FP16 by looking at activation distribution instead of weight distribution. We highlight results with a decent perplexity in green. We used INT3 quantization with a group size of 128 and measured the WikiText perplexity ( $\downarrow$ ).</div>

$s=1\ s=1.25\ s=1.5\ s=2\ s=4$

$\Delta^{^{\prime}}/\Delta$

$\Delta^{^{\prime}}\neq\Delta$

$\frac{\Delta^{^{\prime}}}{\Delta}\cdot\frac{1}{s}$

<table border=1 style='margin: auto; word-wrap: break-word;'><tr><td style='text-align: center; word-wrap: break-word;'>OPT-6.7B</td><td style='text-align: center; word-wrap: break-word;'>s=1</td><td style='text-align: center; word-wrap: break-word;'>s=1.25</td><td style='text-align: center; word-wrap: break-word;'>s=1.5</td><td style='text-align: center; word-wrap: break-word;'>s=2</td><td style='text-align: center; word-wrap: break-word;'>s=4</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>proportion of  $\Delta^{\prime}\neq\Delta$</td><td style='text-align: center; word-wrap: break-word;'>0%</td><td style='text-align: center; word-wrap: break-word;'>2.8%</td><td style='text-align: center; word-wrap: break-word;'>4.4%</td><td style='text-align: center; word-wrap: break-word;'>8.2%</td><td style='text-align: center; word-wrap: break-word;'>21.2%</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>average  $\Delta^{\prime}/\Delta$</td><td style='text-align: center; word-wrap: break-word;'>1</td><td style='text-align: center; word-wrap: break-word;'>1.005</td><td style='text-align: center; word-wrap: break-word;'>1.013</td><td style='text-align: center; word-wrap: break-word;'>1.038</td><td style='text-align: center; word-wrap: break-word;'>1.213</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>average  $\frac{\Delta^{\prime}}{\Delta}\cdot\frac{1}{s}$</td><td style='text-align: center; word-wrap: break-word;'>1</td><td style='text-align: center; word-wrap: break-word;'>0.804</td><td style='text-align: center; word-wrap: break-word;'>0.676</td><td style='text-align: center; word-wrap: break-word;'>0.519</td><td style='text-align: center; word-wrap: break-word;'>0.303</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>Wiki-2 PPL</td><td style='text-align: center; word-wrap: break-word;'>23.54</td><td style='text-align: center; word-wrap: break-word;'>12.87</td><td style='text-align: center; word-wrap: break-word;'>12.48</td><td style='text-align: center; word-wrap: break-word;'>11.92</td><td style='text-align: center; word-wrap: break-word;'>12.36</td></tr></table>

<div style="text-align: center;">Table 2. Statistics when multiplying the 1% salient channels by s > 1. Scaling up the salient channels significantly improves the perplexity (23.54 to 11.92). As s goes larger, the percentage of changed  $\Delta$  increases, and the error reduction rate for salient channels also increases. However, the best perplexity is achieved at s = 2, since further increasing s will increase the quantization error for non-salient channels.</div>

函数定义为：

$Q(\mathbf{w})=\Delta\cdot\mathrm{Round}(\frac{\mathbf{w}}{\Delta}),\quad\Delta=\frac{\max(|\mathbf{w}|)}{2^{N-1}},$

其中N是量化比特数，$\Delta$是由绝对最大值决定的量化标量。现在考虑一个权重元素$w \in w$，如果我们将w乘以s > 1并反向缩放x，我们将得到$Q(w \cdot s)(x/s)$，即：

$Q(w\cdot s)\cdot\frac{x}{s}=\Delta^{^{\prime}}\cdot\mathrm{R o u n d}(\frac{w s}{\Delta^{^{\prime}}})\cdot x\cdot\frac{1}{s},$

其中$\Delta^{\prime}$是应用s后的新量化标量。我们经验性地发现：（1）来自Round$(\cdot)$（记为RoundErr$(\cdot)$）的期望误差不变：由于round函数将浮点数映射到整数，误差大致均匀分布在$[0,0.5]$，导致平均误差为0.25；即RoundErr$(\cdot)$ $\sim$≈ 0.25。（2）放大单个元素w通常不会改变组w中的最大值。因此我们有$\Delta^{\prime} \approx \Delta$；（3）由于$\Delta$和x以FP16表示，它们没有量化误差。因此，方程1和2中的量化误差可以表示为

```tex
\begin{aligned}\mathrm{Err}(Q(w)x)&=\Delta\cdot\mathrm{RoundErr}(\frac{w}{\Delta})\cdot x\\\mathrm{Err}(Q(w\cdot s)(\frac{x}{s}))&=\Delta^{^{\prime}}\cdot\mathrm{RoundErr}(\frac{ws}{\Delta^{^{\prime}}})\cdot x\cdot\frac{1}{s}\end{aligned}
```

_Note: Math block could not be automatically fixed (Low confidence (low)). Showing as code._

$s=2$

<table border=1 style='margin: auto; word-wrap: break-word;'><tr><td style='text-align: center; word-wrap: break-word;'>OPT (PPL $\downarrow$ )</td><td style='text-align: center; word-wrap: break-word;'>1.3B</td><td style='text-align: center; word-wrap: break-word;'>2.7B</td><td style='text-align: center; word-wrap: break-word;'>6.7B</td><td style='text-align: center; word-wrap: break-word;'>13B</td><td style='text-align: center; word-wrap: break-word;'>30B</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>FP16</td><td style='text-align: center; word-wrap: break-word;'>14.62</td><td style='text-align: center; word-wrap: break-word;'>12.47</td><td style='text-align: center; word-wrap: break-word;'>10.86</td><td style='text-align: center; word-wrap: break-word;'>10.13</td><td style='text-align: center; word-wrap: break-word;'>9.56</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>RTN</td><td style='text-align: center; word-wrap: break-word;'>119.47</td><td style='text-align: center; word-wrap: break-word;'>298.00</td><td style='text-align: center; word-wrap: break-word;'>23.54</td><td style='text-align: center; word-wrap: break-word;'>46.04</td><td style='text-align: center; word-wrap: break-word;'>18.80</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>1% FP16</td><td style='text-align: center; word-wrap: break-word;'>16.91</td><td style='text-align: center; word-wrap: break-word;'>13.69</td><td style='text-align: center; word-wrap: break-word;'>11.39</td><td style='text-align: center; word-wrap: break-word;'>10.43</td><td style='text-align: center; word-wrap: break-word;'>9.85</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>s = 2</td><td style='text-align: center; word-wrap: break-word;'>18.63</td><td style='text-align: center; word-wrap: break-word;'>14.94</td><td style='text-align: center; word-wrap: break-word;'>11.92</td><td style='text-align: center; word-wrap: break-word;'>10.80</td><td style='text-align: center; word-wrap: break-word;'>10.32</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>AWQ</td><td style='text-align: center; word-wrap: break-word;'>16.32</td><td style='text-align: center; word-wrap: break-word;'>13.58</td><td style='text-align: center; word-wrap: break-word;'>11.39</td><td style='text-align: center; word-wrap: break-word;'>10.56</td><td style='text-align: center; word-wrap: break-word;'>9.77</td></tr></table>

<div style="text-align: center;">Table 3. AWQ protects salient weights and reduces quantization error by using a scaling-based method. It consistently outperforms Round-to-nearest quantization (RTN) and achieves comparable performance as mixed-precision (1% FP16) while being more hardware-friendly. We use 3-bit quantization with group size 128.</div>

新误差与原始误差的比率为$\frac{\Delta^{^{\prime}}}{\Delta} \cdot \frac{1}{s}$。给定$\Delta^{^{\prime}} \approx \Delta$且s > 1，显著权重w的相对误差更小。

为了验证这一想法，我们将OPT-6.7B模型中1%的显著通道乘以s > 1，并测量表2中每组$\Delta$的变化。我们发现放大显著通道非常有效：困惑度从s = 1（简单RTN）的23.54提升到s = 2的11.92。随着s增大，改变的$\Delta$百分比通常变得更大，但对于s < 2，百分比仍然很小（小于5%）；显著通道的相对误差随着s增加继续减小。然而，最佳PPL实际上出现在s = 2。这是因为如果使用非常大的s，当$\Delta$增加时，会增加非显著通道的相对误差（非显著通道的误差将被$\frac{\Delta'}{\Delta}$放大，在s = 4下，21.2%的通道比率大于1），这可能损害模型的整体准确性。因此，在保护显著通道时，我们还需要考虑非显著通道的误差。

搜索缩放因子。为了同时考虑显著和非显著权重，我们选择自动搜索最优的（每输入通道）缩放因子，以最小化特定层量化后的输出差异。

<div style="text-align: center;"><img src="imgs/img_in_chart_box_153_134_313_289.jpg" alt="Image" width="13%" /></div>

<div style="text-align: center;"><img src="imgs/img_in_chart_box_370_135_725_295.jpg" alt="Image" width="29%" /></div>

<div style="text-align: center;">(a) Generation stage is slower</div>

<div style="text-align: center;"><img src="imgs/img_in_chart_box_775_138_1056_299.jpg" alt="Image" width="22%" /></div>

<div style="text-align: center;">(c) Weight loading is more expensive</div>

<div style="text-align: center;">Figure 3. Bottleneck analysis for Llama-2-7B on NVIDIA RTX 4090. Left: In on-device LLM applications, generation stage is much slower than the context stage. Middle: The generation stage is memory bound and has low arithmetic intensity. W4A16 quantization can effectively improve the arithmetic intensity by 4×. Right: The amount of weight access is orders of magnitude larger than the amount of activation access. Thus, weight-only quantization is more effective for on-device LLMs.</div>

形式上，我们希望优化以下目标：

```tex
\begin{aligned}\mathbf{s}^{*}=\underset{\mathbf{s}}{\arg\min}\mathcal{L}(\mathbf{s})\\\mathcal{L}(\mathbf{s})=\|Q(\mathbf{W}\cdot\mathrm{diag}(\mathbf{s}))(\mathrm{diag}(\mathbf{s})^{-1}\cdot\mathbf{X})-\mathbf{W}\mathbf{X}\|\end{aligned}
```

_Note: Math block could not be automatically fixed (Low confidence (low)). Showing as code._

这里Q表示权重量化函数$(e.g., \operatorname{INT3/INT4}$，组大小为128$)$，W是FP16中的原始权重，X是从小型校准集缓存的输入特征（我们从预训练数据集中取一个小型校准集，以避免过拟合特定任务）。s是每（输入）通道缩放因子；对于$s^{-1} \cdot X$，它通常可以融合到前一个操作中（Wei等人，2022b；Xiao等人，2022）。由于量化函数不可微，我们无法直接通过普通反向传播优化该问题。有一些技术依赖于近似梯度（Bengio等人，2013；Esser等人，2019），但我们发现这些方法仍然存在收敛不稳定的问题。

为了使过程更稳定，我们通过分析影响缩放因子选择的因素来定义最优缩放因子的搜索空间。如上一节所示，权重通道的显著性实际上由激活尺度决定（因此称为“激活感知”）。因此，我们简单地使用一个非常简单的搜索空间：

$\mathbf{s}=\mathbf{s}_{\mathbf{X}}^{\alpha},\quad\alpha^{*}=\underset{\alpha}{\arg\min}\mathcal{L}(\mathbf{s}_{\mathbf{X}}^{\alpha})$

$s_{X}$是激活的平均幅度（每通道），我们使用单个超参数$\alpha$来平衡显著和非显著通道的保护。我们可以通过在区间$\alpha$（0表示不缩放；1对应我们搜索空间中最激进的缩放）上进行快速网格搜索来找到最佳$[0, 1]$。我们进一步应用权重裁剪以最小化量化的MSE误差。我们在表5中提供了OPT模型在INT3-g128量化下的消融研究；AWQ始终优于最近舍入量化（RTN），并达到与混合精度（1% FP16）相当的性能，同时更硬件友好。

优势。我们的方法不依赖于任何回归（Frantar等人，2022）或反向传播，这是许多量化感知训练方法所必需的。它对校准集的依赖最小，因为我们只测量每通道的平均幅度，从而防止过拟合（图8）。因此，我们的方法在量化过程中需要更少的数据，并且可以保留LLM在校准集分布之外的知识。更多细节见第5.3节。

## 4 TINYCHAT：将AWQ映射到边缘平台

AWQ 可以显著减小大型语言模型（LLM）的规模。然而，将 W4A16（4 位权重，16 位激活）量化带来的理论内存节省转化为实测加速并非易事。替代的 W8A8 量化方法，如 SmoothQuant（Xiao 等人，2022），在存储和计算中保持相同的数据精度。这使得反量化过程可以无缝集成到计算内核的收尾阶段。另一方面，W4A16 量化对内存访问和计算使用不同的数据类型。因此，为了获得最佳性能，其反量化必须纳入主计算循环中，这带来了实现上的挑战。为了解决这个问题，我们引入了 TinyChat：一个用于 AWQ 模型推理的轻巧系统。它拥有一个 PyTorch 前端和一个利用设备特定指令集（例如 CUDA/PTX、Neon、AVX）的后端。

### 4.1 为什么 AWQ 有助于加速设备端 LLM

为了理解量化 LLM 在边缘设备上的加速机会，我们首先分析 LLaMA-7B（Touvron 等人，2023a）模型在 RTX 4090 GPU 上的延迟分解。我们采用推理批次大小为 1，以适应边缘用例，并使用 NVIDIA FasterTransformer 以 FP16 实现模型。

上下文与生成延迟。如图 3(a) 所示，生成 20 个令牌需要 310 毫秒，而总结一个包含 200 个令牌的提示

<div style="text-align: center;"><img src="imgs/img_in_image_box_109_130_1083_262.jpg" alt="Image" width="79%" /></div>

<div style="text-align: center;"><img src="imgs/img_in_chart_box_824_139_1084_260.jpg" alt="Image" width="21%" /></div>

<div style="text-align: center;">Figure 4. SIMD-aware weight packing for ARM NEON with 128-bit SIMD units. Original weights are reordered and packed to align with the bit width so that the weights can be unpacked into bytes at runtime using AND and shift bitwise operations with a 128-bit mask.</div>

仅需 10 毫秒。因此，生成阶段明显慢于上下文阶段，特别是对于设备端交互式应用。

生成阶段受内存限制。为了加速生成阶段，我们在图 3(b) 中进行了屋顶线分析。4090 GPU 的峰值计算吞吐量为 165 TFLOPS，内存带宽为 1TB/s。因此，任何算术强度（FLOPs 与内存访问之比）小于 165 的工作负载在 4090 GPU 上都是内存受限的。值得注意的是，当以 FP16 执行时，设备端 LLM 的生成阶段算术强度为$\approx$1。这突显了工作负载的内存受限特性。由于给定模型的 FLOPs 是固定的，提高峰值性能的唯一方法是减少总内存流量。AWQ 将权重内存减少了四倍。

权重访问主导内存流量。因此，我们在图 3(c) 中进一步分解了权重和激活的内存访问。显然，对于设备端 LLM，权重访问主导了内存流量。将模型权重量化为 4 位整数将大致将算术强度提高到 4 FLOPs/Byte，从而在图 3(b) 中实现 4TFLOPS 的峰值性能。由于仅权重量化导致权重位宽更低（因此理论性能上限更高），AWQ 自然遵循此设置用于设备端 LLM 应用。

### 4.2 使用 TinyChat 部署 AWQ

为此，我们证明了 4 位权重量化可以带来$4 \times$理论峰值性能。我们进一步设计 TinyChat 来实现这种加速。在 GPU 上，我们只专注于实现核心组件，包括注意力、层归一化和线性投影内核。灵活的前端允许轻松定制和快速支持新模型。在 GPU 上，使用 4 位 AWQ 的 TinyChat 相比 Huggingface FP16 实现在不同系列的 LLM 上实现了超过$3 \times$的加速。在 CPU 上，我们将整个计算图降级到 C++ 以最小化开销。

即时权重反量化。对于量化层，由于硬件不提供 INT4 和 FP16 之间的乘法指令，我们需要在执行矩阵计算之前将整数反量化为 FP16。我们通过将反量化内核与矩阵乘法内核融合来避免将反量化后的权重写入 DRAM。请注意，这种融合适用于矩阵-矩阵（MM）和矩阵-向量（MV）乘积内核。

SIMD 感知权重打包。即时权重反量化减少了中间 DRAM 访问，但仍然昂贵。例如，反量化单个 4 位权重涉及 1 次移位、1 次按位与和 1 次 FMA 缩放操作，而反量化后的权重仅进行 1 次 FMA 计算。这个过程在具有 SIMD 架构的 CPU 上尤其昂贵，因为 SIMD 架构偏爱向量化指令。为了缓解这个问题，我们建议针对设备 SIMD 单元的位宽进行平台特定的权重打包。图 4 展示了我们在 ARM CPU 上的策略，其 128 位 SIMD 寄存器可提供高达$1.2\times$的加速。这里，每个寄存器保存 32 个 4 位权重，序列为$w_{0}, w_{16}, w_{1}, w_{17}, \ldots, w_{15}, w_{31}$。这种方法只需要三条 SIMD 指令来解包所有 32 个权重，而传统打包中每个权重需要 3 条标量指令（$w_{0}, w_{1}, \ldots, w_{31}$）。一般来说，对于$2^{n}$位 SIMD 寄存器，相邻权重的索引将相差$1/8 \times 2^{n}$，因为每个寄存器可以容纳$1/8 \times 2^{n}$个 8 位整数。在 GPU 上，我们发现按照（Kim 等人，2022）将每 8 个权重打包到$w_{\{0,2,4,6,1,3,5,7\}}$中更高效。

内核融合。我们还广泛应用内核融合来优化设备端 LLM 推理。对于层归一化，我们将所有操作符（例如乘法、除法和平方根）融合到单个内核中。对于注意力层，我们将 QKV 投影融合到单个内核中，并执行即时位置嵌入计算。我们还预分配 KV 缓存并在注意力内核内执行缓存更新。内核融合对于前向传递实现效率低下的模型特别有用，例如 Falcon（Penedo 等人，2023）和 StarCoder（Li 等人，2023c）。值得注意的是，每个 FP16 内核在 4090 GPU 上的计算时间约为 0.01 毫秒，与 GPU 内核启动开销相当。因此，通过内核融合减少内核调用次数可以直接带来加速。

<table border=1 style='margin: auto; word-wrap: break-word;'><tr><td rowspan="2" colspan="2">PPL $\downarrow$</td><td colspan="3">Llama-2</td><td colspan="4">LLaMA</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>7B</td><td style='text-align: center; word-wrap: break-word;'>13B</td><td style='text-align: center; word-wrap: break-word;'>70B</td><td style='text-align: center; word-wrap: break-word;'>7B</td><td style='text-align: center; word-wrap: break-word;'>13B</td><td style='text-align: center; word-wrap: break-word;'>30B</td><td style='text-align: center; word-wrap: break-word;'>65B</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>FP16</td><td style='text-align: center; word-wrap: break-word;'>-</td><td style='text-align: center; word-wrap: break-word;'>5.47</td><td style='text-align: center; word-wrap: break-word;'>4.88</td><td style='text-align: center; word-wrap: break-word;'>3.32</td><td style='text-align: center; word-wrap: break-word;'>5.68</td><td style='text-align: center; word-wrap: break-word;'>5.09</td><td style='text-align: center; word-wrap: break-word;'>4.10</td><td style='text-align: center; word-wrap: break-word;'>3.53</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>INT3</td><td style='text-align: center; word-wrap: break-word;'>RTN</td><td style='text-align: center; word-wrap: break-word;'>6.66</td><td style='text-align: center; word-wrap: break-word;'>5.52</td><td style='text-align: center; word-wrap: break-word;'>3.98</td><td style='text-align: center; word-wrap: break-word;'>7.01</td><td style='text-align: center; word-wrap: break-word;'>5.88</td><td style='text-align: center; word-wrap: break-word;'>4.88</td><td style='text-align: center; word-wrap: break-word;'>4.24</td></tr><tr><td rowspan="3">g128</td><td style='text-align: center; word-wrap: break-word;'>GPTQ</td><td style='text-align: center; word-wrap: break-word;'>6.43</td><td style='text-align: center; word-wrap: break-word;'>5.48</td><td style='text-align: center; word-wrap: break-word;'>3.88</td><td style='text-align: center; word-wrap: break-word;'>8.81</td><td style='text-align: center; word-wrap: break-word;'>5.66</td><td style='text-align: center; word-wrap: break-word;'>4.88</td><td style='text-align: center; word-wrap: break-word;'>4.17</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>GPTQ-R</td><td style='text-align: center; word-wrap: break-word;'>6.42</td><td style='text-align: center; word-wrap: break-word;'>5.41</td><td style='text-align: center; word-wrap: break-word;'>3.86</td><td style='text-align: center; word-wrap: break-word;'>6.53</td><td style='text-align: center; word-wrap: break-word;'>5.64</td><td style='text-align: center; word-wrap: break-word;'>4.74</td><td style='text-align: center; word-wrap: break-word;'>4.21</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>AWQ</td><td style='text-align: center; word-wrap: break-word;'>6.24</td><td style='text-align: center; word-wrap: break-word;'>5.32</td><td style='text-align: center; word-wrap: break-word;'>3.74</td><td style='text-align: center; word-wrap: break-word;'>6.35</td><td style='text-align: center; word-wrap: break-word;'>5.52</td><td style='text-align: center; word-wrap: break-word;'>4.61</td><td style='text-align: center; word-wrap: break-word;'>3.95</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>INT4</td><td style='text-align: center; word-wrap: break-word;'>RTN</td><td style='text-align: center; word-wrap: break-word;'>5.73</td><td style='text-align: center; word-wrap: break-word;'>4.98</td><td style='text-align: center; word-wrap: break-word;'>3.46</td><td style='text-align: center; word-wrap: break-word;'>5.96</td><td style='text-align: center; word-wrap: break-word;'>5.25</td><td style='text-align: center; word-wrap: break-word;'>4.23</td><td style='text-align: center; word-wrap: break-word;'>3.67</td></tr><tr><td rowspan="3">g128</td><td style='text-align: center; word-wrap: break-word;'>GPTQ</td><td style='text-align: center; word-wrap: break-word;'>5.69</td><td style='text-align: center; word-wrap: break-word;'>4.98</td><td style='text-align: center; word-wrap: break-word;'>3.42</td><td style='text-align: center; word-wrap: break-word;'>6.22</td><td style='text-align: center; word-wrap: break-word;'>5.23</td><td style='text-align: center; word-wrap: break-word;'>4.24</td><td style='text-align: center; word-wrap: break-word;'>3.66</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>GPTQ-R</td><td style='text-align: center; word-wrap: break-word;'>5.63</td><td style='text-align: center; word-wrap: break-word;'>4.99</td><td style='text-align: center; word-wrap: break-word;'>3.43</td><td style='text-align: center; word-wrap: break-word;'>5.83</td><td style='text-align: center; word-wrap: break-word;'>5.20</td><td style='text-align: center; word-wrap: break-word;'>4.22</td><td style='text-align: center; word-wrap: break-word;'>3.66</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>AWQ</td><td style='text-align: center; word-wrap: break-word;'>5.60</td><td style='text-align: center; word-wrap: break-word;'>4.97</td><td style='text-align: center; word-wrap: break-word;'>3.41</td><td style='text-align: center; word-wrap: break-word;'>5.78</td><td style='text-align: center; word-wrap: break-word;'>5.19</td><td style='text-align: center; word-wrap: break-word;'>4.21</td><td style='text-align: center; word-wrap: break-word;'>3.62</td></tr></table>

<div style="text-align: center;">Table 4. AWQ improves over round-to-nearest quantization (RTN) for different model sizes and different bit-precisions. It consistently achieves better perplexity than GPTQ (w/ and w/o reordering) on LLaMA & Llama-2 models.</div>

$PPL\downarrow$

<table border=1 style='margin: auto; word-wrap: break-word;'><tr><td style='text-align: center; word-wrap: break-word;'>Wikitext2 PPL $\downarrow$</td><td style='text-align: center; word-wrap: break-word;'>Mixtral-8x7B</td><td style='text-align: center; word-wrap: break-word;'>Mistral-7B</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>FP16</td><td style='text-align: center; word-wrap: break-word;'>5.94</td><td style='text-align: center; word-wrap: break-word;'>4.14</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>INT4-g128</td><td style='text-align: center; word-wrap: break-word;'>6.05</td><td style='text-align: center; word-wrap: break-word;'>4.30</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>INT3-g128</td><td style='text-align: center; word-wrap: break-word;'>6.52</td><td style='text-align: center; word-wrap: break-word;'>4.83</td></tr></table>

<div style="text-align: center;">Table 5. AWQ quantization results on Mistral-7B-Instruct-v0.2(Jiang et al., 2023) and Mixtral-8x7B-Instruct-v0.1 model (Jiang et al., 2024). The PPL result on wikitext shows that AWQ can achieve superior quantization performance on different model architectures including LLMs with GQA and Mixture-of-Experts (MoE) models.</div>

## 5 实验

### 5.1 设置

量化。在本工作中，我们专注于仅权重的分组量化。如先前工作所示（Dettmers & Zettlemoyer, 2022; Frantar et al., 2022），分组量化总是有助于改善性能/模型大小的权衡。除非另有说明，我们在整个工作中使用128的组大小。我们专注于INT4/INT3量化，因为它们能够基本保留LLMs的性能（Dettmers & Zettlemoyer, 2022）。对于AWQ，我们使用了来自Pile（Gao et al., 2020）数据集的小型校准集，以避免过拟合到特定的下游领域。我们使用20的网格大小来搜索最优的$\alpha$在方程5中。

模型。我们在LLaMA（Touvron et al., 2023a）和OPT（Zhang et al., 2022）系列上对我们的方法进行了基准测试。还有其他开放的LLMs，如BLOOM（Scao et al., 2022），但它们的质量通常较差，因此我们未将它们纳入我们的研究。我们进一步基准测试了一个指令调优模型Vicuna（Chiang et al., 2023）和视觉语言模型OpenFlamingo-9B（Awadalla et al., 2023）和LLaVA-13B（Liu et al., 2023a），以展示我们方法的通用性。

<div style="text-align: center;"><img src="imgs/img_in_chart_box_611_513_892_649.jpg" alt="Image" width="22%" /></div>

<div style="text-align: center;"><img src="imgs/img_in_chart_box_894_514_1102_650.jpg" alt="Image" width="16%" /></div>

<div style="text-align: center;">Figure 5. Comparing INT3-g128 quantized Vicuna models with FP16 counterparts under GPT-4 evaluation protocol (Chiang et al., 2023). More winning cases (in blue) indicate better performance. AWQ consistently improves the quantized performance compared to RTN and GPTQ (Frantar et al., 2022), showing generalization to instruction-tuned models.</div>

评估。遵循先前文献（Dettmers et al., 2022; Xiao et al., 2022; Frantar et al., 2022; Dettmers & Zettlemoyer, 2022; Yao et al., 2022），我们主要在语言建模任务上分析量化模型（在WikiText-2（Merity et al., 2016）上的困惑度评估），因为困惑度可以稳定地反映LLM的性能（Dettmers & Zettlemoyer, 2022）。

基线。我们的主要基线是普通的四舍五入量化（RTN）。当使用像128这样的小组大小时，它实际上相当强大（Frantar et al., 2022; Dettmers & Zettlemoyer, 2022）。我们还与LLM权重量化的最先进方法GPTQ（Frantar et al., 2022）进行了比较。对于GPTQ，我们还与使用“重新排序”技巧的更新版本（表示为GPTQ-Reorder或GPTQ-R）进行了比较。其他技术如ZeroQuant（Yao et al., 2022）、AdaRound（Nagel et al., 2020）和BRECQ（Li et al., 2021）依赖于反向传播来更新量化权重，这可能不容易扩展到大型模型大小；它们也没有超越GPTQ（Frantar et al., 2022），因此未纳入研究。

### 5.2 评估

LLaMA模型上的结果。我们专注于LLaMA模型（LLaMA（Touvron et al., 2023a）和Llama-2（Touvron et al., 2023b））

<div style="text-align: center;">AWQ: Activation-aware Weight Quantization for On-Device LLM Compression and Acceleration</div>

<table border=1 style='margin: auto; word-wrap: break-word;'><tr><td colspan="2">COCO (CIDEr  $\uparrow$ )</td><td style='text-align: center; word-wrap: break-word;'>0-shot</td><td style='text-align: center; word-wrap: break-word;'>4-shot</td><td style='text-align: center; word-wrap: break-word;'>8-shot</td><td style='text-align: center; word-wrap: break-word;'>16-shot</td><td style='text-align: center; word-wrap: break-word;'>32-shot</td><td style='text-align: center; word-wrap: break-word;'>$\Delta(32$ -shot)</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>FP16</td><td style='text-align: center; word-wrap: break-word;'>-</td><td style='text-align: center; word-wrap: break-word;'>63.73</td><td style='text-align: center; word-wrap: break-word;'>72.18</td><td style='text-align: center; word-wrap: break-word;'>76.95</td><td style='text-align: center; word-wrap: break-word;'>79.74</td><td style='text-align: center; word-wrap: break-word;'>81.70</td><td style='text-align: center; word-wrap: break-word;'>-</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>INT4</td><td style='text-align: center; word-wrap: break-word;'>RTN</td><td style='text-align: center; word-wrap: break-word;'>60.24</td><td style='text-align: center; word-wrap: break-word;'>68.07</td><td style='text-align: center; word-wrap: break-word;'>72.46</td><td style='text-align: center; word-wrap: break-word;'>74.09</td><td style='text-align: center; word-wrap: break-word;'>77.13</td><td style='text-align: center; word-wrap: break-word;'>-4.57</td></tr><tr><td rowspan="2">g128</td><td style='text-align: center; word-wrap: break-word;'>GPTQ</td><td style='text-align: center; word-wrap: break-word;'>59.72</td><td style='text-align: center; word-wrap: break-word;'>67.68</td><td style='text-align: center; word-wrap: break-word;'>72.53</td><td style='text-align: center; word-wrap: break-word;'>74.98</td><td style='text-align: center; word-wrap: break-word;'>74.98</td><td style='text-align: center; word-wrap: break-word;'>-6.72</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>AWQ</td><td style='text-align: center; word-wrap: break-word;'>62.57</td><td style='text-align: center; word-wrap: break-word;'>71.02</td><td style='text-align: center; word-wrap: break-word;'>74.75</td><td style='text-align: center; word-wrap: break-word;'>78.23</td><td style='text-align: center; word-wrap: break-word;'>80.53</td><td style='text-align: center; word-wrap: break-word;'>-1.17</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>INT3</td><td style='text-align: center; word-wrap: break-word;'>RTN</td><td style='text-align: center; word-wrap: break-word;'>46.07</td><td style='text-align: center; word-wrap: break-word;'>55.13</td><td style='text-align: center; word-wrap: break-word;'>60.46</td><td style='text-align: center; word-wrap: break-word;'>63.21</td><td style='text-align: center; word-wrap: break-word;'>64.79</td><td style='text-align: center; word-wrap: break-word;'>-16.91</td></tr><tr><td rowspan="2">g128</td><td style='text-align: center; word-wrap: break-word;'>GPTQ</td><td style='text-align: center; word-wrap: break-word;'>29.84</td><td style='text-align: center; word-wrap: break-word;'>50.77</td><td style='text-align: center; word-wrap: break-word;'>56.55</td><td style='text-align: center; word-wrap: break-word;'>60.54</td><td style='text-align: center; word-wrap: break-word;'>64.77</td><td style='text-align: center; word-wrap: break-word;'>-16.93</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>AWQ</td><td style='text-align: center; word-wrap: break-word;'>56.33</td><td style='text-align: center; word-wrap: break-word;'>64.73</td><td style='text-align: center; word-wrap: break-word;'>68.79</td><td style='text-align: center; word-wrap: break-word;'>72.86</td><td style='text-align: center; word-wrap: break-word;'>74.47</td><td style='text-align: center; word-wrap: break-word;'>-7.23</td></tr></table>

<div style="text-align: center;">Table 6. Quantization results of a visual language model OpenFlamingo-9B (Awadalla et al., 2023) on COCO Captioning datasets. Activation-aware Weight Quantization outperforms existing methods under zero-shot and various few-shot settings, demonstrating the generability to different modalities and in-context learning workloads. Activation-aware Weight Quantization reduces the quantization degradation (32-shot) from 4.57 to 1.17 under INT4-g128, providing 4× model size reduction with negligible performance loss.</div>

<table border=1 style='margin: auto; word-wrap: break-word;'><tr><td style='text-align: center; word-wrap: break-word;'>Model (Accuracy $\uparrow$ )</td><td style='text-align: center; word-wrap: break-word;'>VQAv2</td><td style='text-align: center; word-wrap: break-word;'>GQA</td><td style='text-align: center; word-wrap: break-word;'>VizWiz</td><td style='text-align: center; word-wrap: break-word;'>SQA-I</td><td style='text-align: center; word-wrap: break-word;'>VQA-T</td><td style='text-align: center; word-wrap: break-word;'>POPE</td><td style='text-align: center; word-wrap: break-word;'>MME</td><td style='text-align: center; word-wrap: break-word;'>MMB</td><td style='text-align: center; word-wrap: break-word;'>SEED</td><td style='text-align: center; word-wrap: break-word;'>llava-bench</td><td style='text-align: center; word-wrap: break-word;'>MM-Vet</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>VILA-7B</td><td style='text-align: center; word-wrap: break-word;'>80.3</td><td style='text-align: center; word-wrap: break-word;'>63.1</td><td style='text-align: center; word-wrap: break-word;'>59.6</td><td style='text-align: center; word-wrap: break-word;'>68.0</td><td style='text-align: center; word-wrap: break-word;'>62.6</td><td style='text-align: center; word-wrap: break-word;'>86.3</td><td style='text-align: center; word-wrap: break-word;'>1489.4</td><td style='text-align: center; word-wrap: break-word;'>69.8</td><td style='text-align: center; word-wrap: break-word;'>61.7</td><td style='text-align: center; word-wrap: break-word;'>75.2</td><td style='text-align: center; word-wrap: break-word;'>35.1</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>VILA-7B-AWQ</td><td style='text-align: center; word-wrap: break-word;'>80.1</td><td style='text-align: center; word-wrap: break-word;'>63.0</td><td style='text-align: center; word-wrap: break-word;'>57.8</td><td style='text-align: center; word-wrap: break-word;'>68.0</td><td style='text-align: center; word-wrap: break-word;'>61.9</td><td style='text-align: center; word-wrap: break-word;'>85.3</td><td style='text-align: center; word-wrap: break-word;'>1486.3</td><td style='text-align: center; word-wrap: break-word;'>68.8</td><td style='text-align: center; word-wrap: break-word;'>61.3</td><td style='text-align: center; word-wrap: break-word;'>75.8</td><td style='text-align: center; word-wrap: break-word;'>35.9</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>VILA-13B</td><td style='text-align: center; word-wrap: break-word;'>80.5</td><td style='text-align: center; word-wrap: break-word;'>63.6</td><td style='text-align: center; word-wrap: break-word;'>63.1</td><td style='text-align: center; word-wrap: break-word;'>70.5</td><td style='text-align: center; word-wrap: break-word;'>64.0</td><td style='text-align: center; word-wrap: break-word;'>86.3</td><td style='text-align: center; word-wrap: break-word;'>1553.6</td><td style='text-align: center; word-wrap: break-word;'>73.8</td><td style='text-align: center; word-wrap: break-word;'>62.8</td><td style='text-align: center; word-wrap: break-word;'>78.3</td><td style='text-align: center; word-wrap: break-word;'>42.6</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>VILA-13B-AWQ</td><td style='text-align: center; word-wrap: break-word;'>80.4</td><td style='text-align: center; word-wrap: break-word;'>63.6</td><td style='text-align: center; word-wrap: break-word;'>63.0</td><td style='text-align: center; word-wrap: break-word;'>71.2</td><td style='text-align: center; word-wrap: break-word;'>63.5</td><td style='text-align: center; word-wrap: break-word;'>87.0</td><td style='text-align: center; word-wrap: break-word;'>1552.9</td><td style='text-align: center; word-wrap: break-word;'>73.6</td><td style='text-align: center; word-wrap: break-word;'>62.2</td><td style='text-align: center; word-wrap: break-word;'>77.6</td><td style='text-align: center; word-wrap: break-word;'>42.0</td></tr></table>

<div style="text-align: center;">Table 7. INT4-g128 results of VILA-7B and VILA-13B (Lin et al., 2024) on 11 visual-language benchmarks. AWQ consistently shows lossless performance on all benchmarks. Benchmark names are abbreviated due to space limits. VQA-v2 (Goyal et al., 2017); GQA (Hudson & Manning, 2019); VisWiz (Gurari et al., 2018); SQA $^{1}$ : ScienceQA-IMG (Lu et al., 2022); VQA $^{2}$ : TextVQA (Singh et al., 2019); POPE (Li et al., 2023d); MME (Fu et al., 2023); MMB: MMBench (Liu et al., 2023b); MMB $^{CN}$ : MMBench-Chinese (Liu et al., 2023b); SEED: SEED-Bench (Li et al., 2023a); LLaVA $^{W}$ : LLaVA-Bench (In-the-Wild) (Liu et al., 2023a); MM-Vet (Yu et al., 2023).</div>

et al., 2023b）），因为它们与其他开源LLMs相比具有优越性能（Zhang et al., 2022; Scao et al., 2022）；它也是许多流行开源模型的基础（Taori et al., 2023; Chiang et al., 2023）。我们在表4中评估量化前后的困惑度。AWQ在不同模型规模（7B-70B）和代际中始终优于四舍五入（RTN）和GPTQ（Frantar et al., 2022）（有和没有重新排序）。

Mistral / Mixtral模型上的结果。我们还在Mistral和Mixtral模型上评估了AWQ，它们分别是最流行的开源LLMs和混合专家（MoE）模型之一（Jiang et al., 2023; 2024）。结果表明，AWQ在Mistral和Mixtral模型上都实现了优越性能。这证明AWQ在各种模型架构中都是有效的。

指令调优模型的量化。指令调优可以显著改善模型的性能和可用性（Wei et al., 2021; Sanh et al., 2021; Ouyang et al., 2022; Chung et al., 2022）。它已成为模型部署前的重要步骤。我们在图5中进一步基准测试我们的方法在流行指令调优模型Vicuna（Chiang et al., 2023）上的性能。我们使用GPT-4分数来评估量化模型相对于FP16对应模型在80个样本问题上的性能（Chiang et al., 2023）。我们比较两种顺序（量化-FP16，FP16-量化）的响应，以消除顺序效应。

<table border=1 style='margin: auto; word-wrap: break-word;'><tr><td style='text-align: center; word-wrap: break-word;'>MBPP (7B)</td><td style='text-align: center; word-wrap: break-word;'>pass@1</td><td style='text-align: center; word-wrap: break-word;'>pass@10</td><td style='text-align: center; word-wrap: break-word;'>GSM8K</td><td style='text-align: center; word-wrap: break-word;'>7B</td><td style='text-align: center; word-wrap: break-word;'>13B</td><td style='text-align: center; word-wrap: break-word;'>70B</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>FP16</td><td style='text-align: center; word-wrap: break-word;'>38.53</td><td style='text-align: center; word-wrap: break-word;'>49.77</td><td style='text-align: center; word-wrap: break-word;'>FP16</td><td style='text-align: center; word-wrap: break-word;'>13.87</td><td style='text-align: center; word-wrap: break-word;'>26.16</td><td style='text-align: center; word-wrap: break-word;'>56.41</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>RTN</td><td style='text-align: center; word-wrap: break-word;'>37.51</td><td style='text-align: center; word-wrap: break-word;'>48.49</td><td style='text-align: center; word-wrap: break-word;'>RTN</td><td style='text-align: center; word-wrap: break-word;'>11.07</td><td style='text-align: center; word-wrap: break-word;'>21.23</td><td style='text-align: center; word-wrap: break-word;'>53.98</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>GPTQ</td><td style='text-align: center; word-wrap: break-word;'>31.97</td><td style='text-align: center; word-wrap: break-word;'>44.75</td><td style='text-align: center; word-wrap: break-word;'>GPTQ</td><td style='text-align: center; word-wrap: break-word;'>12.13</td><td style='text-align: center; word-wrap: break-word;'>24.26</td><td style='text-align: center; word-wrap: break-word;'>56.03</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>AWQ</td><td style='text-align: center; word-wrap: break-word;'>40.64</td><td style='text-align: center; word-wrap: break-word;'>49.25</td><td style='text-align: center; word-wrap: break-word;'>AWQ</td><td style='text-align: center; word-wrap: break-word;'>13.57</td><td style='text-align: center; word-wrap: break-word;'>25.25</td><td style='text-align: center; word-wrap: break-word;'>56.40</td></tr></table>

<div style="text-align: center;">Table 8. INT4-g128 quantization results of CodeLlama-7b-Instruct-hf on MBPP dataset and Llama-2 (7B/13B/70B) on GSM8K dataset. AWQ outperforms existing methods on programming and math datasets, demonstrating the generability to different scenarios and evaluation settings. Notably, AWQ under the INT4-g128 configuration demonstrates comparable performance to the original FP16 model across both datasets.</div>

（我们发现GPT-4倾向于提高第一个输入的评分），导致160次试验。AWQ在两种规模（7B和13B）下始终改善INT3-g128量化的Vicuna模型，优于RTN和GPTQ，证明了对指令调优模型的通用性。

多模态语言模型的量化。大型多模态模型（LMMs）或视觉语言模型（VLMs）是增强视觉输入的LLMs（Alayrac et al., 2022; Li et al., 2023b; Koh et al., 2023; Driess et al., 2023; Zhang et al., 2023; Liu et al., 2023a）。这类模型能够基于图像/视频输入执行文本生成。由于我们的方法没有对校准集过拟合的问题，它可以直接应用于VLMs。

有时我只是看着从太空拍摄的地球照片，惊叹于这一切是多么美丽。

问题：你能详细解释这个梗吗？

<div style="text-align: center;"><img src="imgs/img_in_image_box_114_186_281_347.jpg" alt="Image" width="13%" /></div>

INT4-g128 RTN：图像中的梗展示了一盘食物，可能是炸鸡，有一个独特的转折。食物顶部放置了地球和其他行星的小图片，创造了一个有趣和引人发笑的视觉效果。盘子旁边，一个标题写道：“有时我看着从太空拍摄的地球照片，惊叹于这一切是多么美丽。”这种食物和行星图像的结合增添了一丝幽默和好奇，让观众想知道梗的背景和意图。

INT4-g128 AWQ：图像中的梗是对观看从太空拍摄的地球照片概念的一个轻松幽默的演绎。一盘炸食，特别是鸡块，配以标题：“有时我看着从太空拍摄的地球照片，惊叹于这一切是多么美丽。”食物物品和地球相关背景的俏皮结合创造了一个有趣和娱乐的视觉效果。

图6。来自LLaVA-13B模型（Liu et al., 2023a）的视觉推理示例。AWQ改进了四舍五入（RTN）基线，提供了更合理的答案。我们为文本着色以显示正确或错误的响应。

<div style="text-align: center;"><img src="imgs/img_in_image_box_111_432_287_550.jpg" alt="Image" width="14%" /></div>

W4-RTN：一架模型飞机在天空中飞行。

W4-AWQ：两架玩具飞机坐在草地上。

<div style="text-align: center;"><img src="imgs/img_in_image_box_441_432_596_550.jpg" alt="Image" width="12%" /></div>

W4-RTN：一名男子怀里抱着一只小象。

W4-AWQ：一名男子和他的女儿与一头大象合影。

<div style="text-align: center;"><img src="imgs/img_in_image_box_777_433_949_551.jpg" alt="Image" width="14%" /></div>

W4-RTN：一名男子和一只狗走过一些灌木丛。

W4-AWQ：两只狗在街上行走。

<div style="text-align: center;">Figure 7. Qualitative results of quantized OpenFlamingo-9B (Awadalla et al., 2023) on COCO captioning dataset (4-shot, INT4-g128 quantization). Our method significantly improves the captioning quality compared to the round-to-nearest (RTN) baseline. We color the text to show the correct or wrong captions.</div>

为了提供准确且高效的量化。我们使用OpenFlamingo-9B模型（Awadalla等人，2023年）（Alayrac等人，2022年的开源复现）在COCO字幕（Chen等人，2015年）数据集（表6）上进行实验。我们测量了不同少样本设置下5k个样本的平均性能。我们仅量化模型的语言部分，因为它主导了模型大小。AWQ在零样本和各种少样本设置下优于现有方法，证明了其对不同模态和上下文学习工作负载的泛化能力。在INT4-g128下，它将量化退化（32样本）从4.57降低到1.17，实现了4倍的模型大小缩减，且性能损失可忽略不计。为了进一步证明AWQ的泛化能力，我们还在一个最先进的多图像视觉语言模型VILA上评估了AWQ。表7中的结果显示，AWQ在11个视觉语言基准上实现了无损量化性能。我们进一步在图7中提供了一些定性字幕结果，以展示我们相对于RTN的优势。我们的方法为LMM/VLM量化提供了一个即用型解决方案。据我们所知，这是首次对VLM低比特量化的研究。

视觉推理结果。我们进一步在图6中提供了LLaVA-13B模型（Liu等人，2023a年）的一些定性视觉推理示例。对于INT4-g128量化，AWQ相比四舍五入（RTN）改进了响应，导致更合理的答案。在第一个示例中，AWQ模型能够理解这个梗，因为它类似于从太空看地球的样子，而RTN产生了错误的描述（用红色标记）。

<table border=1 style='margin: auto; word-wrap: break-word;'><tr><td style='text-align: center; word-wrap: break-word;'>OPT (Wiki PPL $\downarrow$ )</td><td style='text-align: center; word-wrap: break-word;'>1.3B</td><td style='text-align: center; word-wrap: break-word;'>2.7B</td><td style='text-align: center; word-wrap: break-word;'>6.7B</td><td style='text-align: center; word-wrap: break-word;'>13B</td><td style='text-align: center; word-wrap: break-word;'>30B</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>FP16</td><td style='text-align: center; word-wrap: break-word;'>14.62</td><td style='text-align: center; word-wrap: break-word;'>12.47</td><td style='text-align: center; word-wrap: break-word;'>10.86</td><td style='text-align: center; word-wrap: break-word;'>10.13</td><td style='text-align: center; word-wrap: break-word;'>9.56</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>RTN</td><td style='text-align: center; word-wrap: break-word;'>10476</td><td style='text-align: center; word-wrap: break-word;'>193210</td><td style='text-align: center; word-wrap: break-word;'>7622</td><td style='text-align: center; word-wrap: break-word;'>17564</td><td style='text-align: center; word-wrap: break-word;'>8170</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>GPTQ</td><td style='text-align: center; word-wrap: break-word;'>46.67</td><td style='text-align: center; word-wrap: break-word;'>28.15</td><td style='text-align: center; word-wrap: break-word;'>16.65</td><td style='text-align: center; word-wrap: break-word;'>16.74</td><td style='text-align: center; word-wrap: break-word;'>11.75</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>AWQ + GPTQ</td><td style='text-align: center; word-wrap: break-word;'>35.71</td><td style='text-align: center; word-wrap: break-word;'>25.70</td><td style='text-align: center; word-wrap: break-word;'>15.71</td><td style='text-align: center; word-wrap: break-word;'>13.25</td><td style='text-align: center; word-wrap: break-word;'>11.38</td></tr></table>

<div style="text-align: center;">Table 9. Our method is orthogonal to GPTQ: it further closes the performance gap under extreme low-bit quantization (INT2-g64) when combined with GPTQ. Results are WikiText-2 perplexity of OPT models.</div>

编程和数学任务的结果。为了进一步评估AWQ在涉及复杂生成任务上的性能，我们还在MBPP（Austin等人，2021年）和GSM8K（Cobbe等人，2021年）上测试了AWQ。MBPP（Austin等人，2021年）包含约1,000个Python编程问题，设计为入门级程序员可解决，涵盖编程基础、标准库功能等。GSM8K（Cobbe等人，2021年）创建用于支持需要多步推理的基本数学问题的问答任务。我们将CodeLlama-7b-Instruct-hf和Llama-2量化为INT4-g128，并在编程和数学数据集（表8）上进行实验。AWQ在两个数据集上均优于现有方法，证明了其对复杂生成的泛化能力。在INT4-g128配置下，AWQ在两个数据集上表现出与原始FP16模型相当的性能。

极低比特量化。我们进一步将LLM量化为INT2以适应有限的设备内存（表9）。

<div style="text-align: center;"><img src="imgs/img_in_chart_box_108_132_564_274.jpg" alt="Image" width="37%" /></div>

<div style="text-align: center;">(a) Our method needs a smaller calibration set</div>

<table border=1 style='margin: auto; word-wrap: break-word;'><tr><td style='text-align: center; word-wrap: break-word;'>Eval</td><td colspan="3">GPTQ</td><td colspan="2">Ours</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>Calib</td><td style='text-align: center; word-wrap: break-word;'>PubMed</td><td style='text-align: center; word-wrap: break-word;'>Enron</td><td style='text-align: center; word-wrap: break-word;'></td><td style='text-align: center; word-wrap: break-word;'>PubMed</td><td style='text-align: center; word-wrap: break-word;'>Enron</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>PubMed</td><td style='text-align: center; word-wrap: break-word;'>32.48</td><td style='text-align: center; word-wrap: break-word;'>50.41</td><td style='text-align: center; word-wrap: break-word;'>$^{+4.89}$</td><td style='text-align: center; word-wrap: break-word;'>32.56</td><td style='text-align: center; word-wrap: break-word;'>45.07</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>Enron</td><td style='text-align: center; word-wrap: break-word;'>$^{+2.33}$</td><td style='text-align: center; word-wrap: break-word;'>34.81</td><td style='text-align: center; word-wrap: break-word;'>45.52</td><td style='text-align: center; word-wrap: break-word;'>$^{+0.60}$</td><td style='text-align: center; word-wrap: break-word;'>33.16</td></tr></table>

$+4.89$

<div style="text-align: center;">(b) Our method is more robust to calibration set distribution</div>

<div style="text-align: center;">Figure 8. Left: AWQ needs a much smaller calibration set to reach a good quantized performance. It can achieve better perplexity using  $10\times$  smaller calibration set compared to GPTQ. Right: Our method is more robust to the calibration set distribution. Overall, using the same calibration and evaluation distribution works the best (PubMed-PubMed, Enron-Enron). But when using a different calibration distribution (PubMed-Enron, Enron-PubMed), AWQ only increases the perplexity by 0.5-0.6, while GPTQ has 2.3-4.9 worse perplexity. All experiments are done with the OPT-6.7B model under INT3-g128 quantization.</div>

<div style="text-align: center;"><img src="imgs/img_in_chart_box_109_445_527_624.jpg" alt="Image" width="34%" /></div>

<div style="text-align: center;"><img src="imgs/img_in_chart_box_509_444_863_624.jpg" alt="Image" width="28%" /></div>

<div style="text-align: center;"><img src="imgs/img_in_chart_box_869_469_1086_602.jpg" alt="Image" width="17%" /></div>

<div style="text-align: center;">(c) RTX 4070 laptop GPU</div>

<div style="text-align: center;">Figure 9. TinyChat provides a turn-key solution to transform the theoretical memory footprint reduction into a quantifiable speedup. As a result, TinyChat is up to  $3.9\times$  and  $3.5\times$  faster than the FP16 implementation from Huggingface on 4090 (desktop GPU) and Orin (mobile GPU), respectively. AWQ also democratizes Llama-2-13B deployment on laptop GPUs (4070) with merely 8GB memory.</div>

RTN完全失败，而AWQ在GPTQ基础上带来了显著的困惑度改进。我们的方法与GPTQ正交。我们可以将我们的方法与GPTQ结合，以进一步提高INT2量化性能，使其成为一个更实用的设置。

### 5.3 数据效率和泛化

校准集的数据效率更好。我们的方法需要较小的校准集，因为我们不依赖回归/反向传播；我们仅从校准集测量平均激活尺度，这是数据高效的。为了演示这一想法，我们在图8（a）中比较了OPT-6.7B模型在INT3-g128量化下的困惑度。AWQ需要小得多的校准集就能达到良好的量化性能；相比GPTQ（16个序列对比192个序列），它可以使用更小的校准集实现更好的困惑度。$10\times$对校准集分布的鲁棒性。我们的方法对校准集分布不太敏感，因为我们仅从校准集测量平均激活尺度，这在不同数据集分布间更具泛化性。我们进一步在图8（b）中基准测试了不同校准集分布的影响。我们从Pile数据集（Gao等人，2020年）中取了两个子集：PubMed摘要和Enron电子邮件（Klimt & Yang，2004年）。我们使用每个子集作为校准集，并在两个集上评估量化模型（校准集和评估集无重叠；我们使用了1k个样本进行评估）。总体而言，使用相同的校准和评估分布效果最佳（PubMed-PubMed，Enron-Enron）。但当使用不同的校准分布时（PubMed-Enron，Enron-PubMed），AWQ仅增加困惑度0.5-0.6，而GPTQ的困惑度差2.3-4.9。这证明了AWQ对校准集分布的鲁棒性。

5.4 加速评估

<table border=1 style='margin: auto; word-wrap: break-word;'><tr><td style='text-align: center; word-wrap: break-word;'>Model (Throughput $\uparrow$ )</td><td style='text-align: center; word-wrap: break-word;'>Precision</td><td style='text-align: center; word-wrap: break-word;'>A100</td><td style='text-align: center; word-wrap: break-word;'>4090</td><td style='text-align: center; word-wrap: break-word;'>Orin</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>VILA-7B</td><td style='text-align: center; word-wrap: break-word;'>FP16</td><td style='text-align: center; word-wrap: break-word;'>81.6</td><td style='text-align: center; word-wrap: break-word;'>58.5</td><td style='text-align: center; word-wrap: break-word;'>11.5</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>VILA-7B-AWQ</td><td style='text-align: center; word-wrap: break-word;'>W4A16</td><td style='text-align: center; word-wrap: break-word;'>155.3</td><td style='text-align: center; word-wrap: break-word;'>168.1</td><td style='text-align: center; word-wrap: break-word;'>35.6</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>VILA-13B</td><td style='text-align: center; word-wrap: break-word;'>FP16</td><td style='text-align: center; word-wrap: break-word;'>48.5</td><td style='text-align: center; word-wrap: break-word;'>OOM</td><td style='text-align: center; word-wrap: break-word;'>6.1</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>VILA-13B-AWQ</td><td style='text-align: center; word-wrap: break-word;'>W4A16</td><td style='text-align: center; word-wrap: break-word;'>102.1</td><td style='text-align: center; word-wrap: break-word;'>99.0</td><td style='text-align: center; word-wrap: break-word;'>17.5</td></tr></table>

<div style="text-align: center;">Table 10. TinyChat also enables seamless deployment of VILA (Lin et al., 2024), a state-of-the-art visual-language model, on multiple GPU platforms. Leveraging our 4-bit AWQ quantization, TinyChat accelerates VILA-7B by up to  $3.1 \times$  and VILA-13B by up to  $2.9 \times$ .</div>

5.4 加速评估

### 设置。在图9中，我们展示了TinyChat带来的系统加速结果。TinyChat优化了线性层和没有量化权重的层。我们按照exllama中描述的协议，在RTX 4090和Jetson Orin上进行基准测试实验。

结果。如图9（a）所示，在4090上，TinyChat为三个LLM家族（Llama-2、MPT和Falcon）带来了加速，相比Huggingface FP16实现。对于Llama-2-7B，我们通过FP16内核融合将推理速度从52令牌/秒提高到62令牌/秒。在更强的FP16基线之上，我们进一步从快速量化线性内核中获得了额外的加速。对于Falcon-7B，官方实现在推理时未正确支持KV缓存，因此它比其他模型慢得多。在这种情况下，我们的FP16优化带来了更大的加速。$^{\dagger}$

<div style="text-align: center;"><img src="imgs/img_in_chart_box_109_129_1062_293.jpg" alt="Image" width="77%" /></div>

<div style="text-align: center;">(a) Latency comparison on Jetson Orin (64G) mobile GPU</div>

<div style="text-align: center;">(b) Latency on Raspberry Pi 4</div>

<div style="text-align: center;">Figure 10. TinyChat offers 1.2-3.0× speedup over existing systems when running 4-bit quantized Llama models on NVIDIA Jetson Orin. It also supports a diverse range of general-purpose and coding-specific LLMs with at least 2.6× speedup over AutoGPTQ, which also supports all these workloads. Moreover, TinyChat seamlessly operates on Raspberry Pi and enables the deployment of LLMs with up to 7 billion parameters on extremely resource-constrained IoT devices.</div>

$2.7-3.9\times$$3.1\times$$1.6\times$在仅有8GB内存的笔记本电脑4070 GPU上，我们仍能以33 tokens/s的速度运行Llama-2-13B模型，而FP16实现无法容纳7B模型。我们还在表10中展示了视觉语言模型（Lin等人，2024）的加速结果。TinyChat为$3\times$在NVIDIA Jetson Orin上为VILA-7B和VILA-13B带来加速。值得注意的是，我们使用原生PyTorch API实现了所有AWQ模型的前向传播，并且此代码可在各种GPU架构上重用。因此，TinyChat提供了卓越的可扩展性。

与其他系统的比较。我们在图10中将TinyChat与现有的边缘LLM推理系统AutoGPTQ、llama.cpp和exllama进行比较。我们的系统在Orin上实现了$1.7\times$相对于llama.cpp的加速。此外，llama.cpp和exllama的适应性有限，主要针对LLaMA和Llama-2模型。相比之下，我们的TinyChat支持广泛的应用，包括StarCoder（Li等人，2023c）、StableCode（GPT-NeoX）（Black等人，2022）、Mistral（Jiang等人，2023）和Falcon（Penedo等人，2023），同时始终提供相对于AutoGPTQ的显著加速。TinyChat甚至使LLM部署在资源极度受限的Raspberry Pi 4B上成为可能，为7B模型实现了0.7 tokens/s的速度。

## 6 结论

在这项工作中，我们提出了激活感知权重量化（AWQ），一种简单而有效的低比特仅权重LLM压缩方法。基于观察到LLM中权重并非同等重要，AWQ执行逐通道缩放以减少显著权重的量化损失。AWQ不会过度拟合校准集，并保留了LLM在各个领域和模态中的通用能力。它在语言建模方面优于现有工作，并适用于指令调优的LM和多模态LM。我们的TinyChat系统进一步将AWQ实现的理论内存节省转化为$3.2-3.3\times$在桌面和移动GPU上相对于Huggingface FP16实现的实测加速，使边缘LLM部署成为可能。

## 致谢

我们感谢MIT AI硬件计划、国家科学基金会、MIT-IBM Watson AI实验室、亚马逊和MIT科学中心、微软图灵学术计划以及三星对本研究的支持。

## 参考文献

Alayrac, J.-B., Donahue, J., Luc, P., Miech, A., Barr, I., Hasson, Y., Lenc, K., Mensch, A., Millican, K., Reynolds, M., 等人。Flamingo：一种用于少样本学习的视觉语言模型。《神经信息处理系统进展》，35:23716–23736，2022。

Austin, J., Odena, A., Nye, M., Bosma, M., Michalewski, H., Dohan, D., Jiang, E., Cai, C., Terry, M., Le, Q., 和 Sutton, C. 使用大型语言模型进行程序合成，2021。

Awadalla, A., Gao, I., Gardner, J., Hessel, J., Hanafy, Y., Zhu, W., Marathe, K., Bitton, Y., Gadre, S., Jitsev, J., Kornblith, S., Koh, P. W., Ilharco, G., Wortsman, M., 和 Schmidt, L. Openflamingo，2023年3月。URL<https://doi.org/10.5281/zenodo.7733589>。

Bengio, Y., Léonard, N., 和 Courville, A. 通过随机神经元估计或传播梯度以进行条件计算。arXiv预印本arXiv:1308.3432，2013。

Black, S., Biderman, S., Hallahan, E., Anthony, Q., Gao, L., Golding, L., He, H., Leahy, C., McDonell, K., Phang, J., 等人。Gpt-neox-20b：一种开源自回归语言模型。arXiv预印本arXiv:2204.06745，2022。

Brown, T., Mann, B., Ryder, N., Subbiah, M., Kaplan, J. D., Dhariwal, P., Neelakantan, A., Shyam, P., Sastry, G., Askell, A., Agarwal, S., Herbert-Voss, A., Krueger, G., Henighan, T., Child, R., Ramesh, A., Ziegler, D., Wu, J., Winter, C., Hesse, C., Chen, M., Sigler, E., Litwin, M., Gray, S., Chess, B., Clark, J., Berner, C., McCandlish, S., Radford, A., Sutskever, I., 和 Amodei, D. 语言模型是少样本学习者。载于 Larochelle, H., Ranzato, M., Hadsell, R., Balcan, M., 和 Lin, H. (编)，《神经信息处理系统进展》，第33卷，第1877–1901页。Curran Associates, Inc., 2020。URL<https://proceedings.neurips.cc/paper/2020/file/1457c0d6bfcb4967418bfb8ac142f64a-Paper.pdf>。

Chen, T., Moreau, T., Jiang, Z., Zheng, L., Yan, E., Shen, H., Cowan, M., Wang, L., Hu, Y., Ceze, L., 等人。TVM：一种用于深度学习的自动化端到端优化编译器。载于《第13届USENIX操作系统设计与实现研讨会（OSDI）》，2018。

Chen, X., Fang, H., Lin, T.-Y., Vedantam, R., Gupta, S., Dollár, P., 和 Zitnick, C. L. Microsoft coco captions：数据收集和评估服务器。arXiv预印本arXiv:1504.00325，2015。

Chiang, W\.-L., Li, Z., Lin, Z., Sheng, Y., Wu, Z., Zhang, H., Zheng, L., Zhuang, S., Zhuang, Y., Gonzalez, J. E., Stoica, I., 和 Xing, E. P. Vicuna：一种以90% ChatGPT质量给GPT-4留下深刻印象的开源聊天机器人，2023年3月。URL<https://lmsys.org/blog/2023-03-30-vicuna/>。

Choi, J., Wang, Z., Venkataramani, S., Chuang, P. I.-J., Srinivasan, V., 和 Gopalakrishnan, K. PACT：量化神经网络的参数化裁剪激活。arXiv预印本arXiv:1805.06085，2018。

Chung, H. W., Hou, L., Longpre, S., Zoph, B., Tay, Y., Fedus, W., Li, E., Wang, X., Dehghani, M., Brahma, S., 等人。扩展指令微调的语言模型。arXiv预印本arXiv:2210.11416，2022。

Cobbe, K., Kosaraju, V., Bavarian, M., Chen, M., Jun, H., Kaiser, L., Plappert, M., Tworek, J., Hilton, J., Nakano, R., Hesse, C., 和 Schulman, J. 训练验证器以解决数学文字问题，2021。

Dettmers, T. 和 Zettlemoyer, L. 4位精度的案例：k位推理缩放定律。arXiv预印本arXiv:2212.09720，2022。

Dettmers, T., Lewis, M., Belkada, Y., 和 Zettlemoyer, L. Llm.int8()：用于大规模Transformer的8位矩阵乘法。arXiv预印本arXiv:2208.07339，2022。

Driess, D., Xia, F., Sajjadi, M. S., Lynch, C., Chowdhery, A., Ichter, B., Wahid, A., Tompson, J., Vuong, Q., Yu, T., 等人。Palm-e：一种具身多模态语言模型。arXiv预印本arXiv:2303.03378，2023。

Esser, S. K., McKinstry, J. L., Bablani, D., Appuswamy, R., 和 Modha, D. S. 学习步长量化（Learned step size quantization）。arXiv 预印本 arXiv:1902.08153，2019。

Feng, S., Hou, B., Jin, H., Lin, W., Shao, J., Lai, R., Ye, Z., Zheng, L., Yu, C. H., Yu, Y., 和 Chen, T. TensorIR：一种用于自动张量化程序优化的抽象。在 ASPLOS，2023。

Frankle, J. 和 Carbin, M. 彩票票假设：寻找稀疏、可训练的神经网络。arXiv 预印本 arXiv:1803.03635，2018。

Frantar, E., Ashkboos, S., Hoefler, T., 和 Alistarh, D. GPTQ：生成式预训练变压器的准确训练后量化。arXiv 预印本 arXiv:2210.17323，2022。

Fu, C., Chen, P., Shen, Y., Qin, Y., Zhang, M., Lin, X., Yang, J., Zheng, X., Li, K., Sun, X., Wu, Y., 和 Ji, R. MME：多模态大语言模型的综合评估基准。arXiv 预印本 arXiv:2306.13394，2023。

Gao, L., Biderman, S., Black, S., Golding, L., Hoppe, T., Foster, C., Phang, J., He, H., Thite, A., Nabeshima, N., 等。The Pile：一个用于语言建模的 800GB 多样化文本数据集。arXiv 预印本 arXiv:2101.00027，2020。

Gholami, A., Kim, S., Dong, Z., Yao, Z., Mahoney, M. W., 和 Keutzer, K. 高效神经网络推理的量化方法综述。arXiv 预印本 arXiv:2103.13630，2021。

Goyal, Y., Khot, T., Summers-Stay, D., Batra, D., 和 Parikh, D. 让 VQA 中的 V 重要：提升图像理解在视觉问答中的作用。在 IEEE 计算机视觉与模式识别会议论文集，第 6904–6913 页，2017。

Gurari, D., Li, Q., Stangl, A. J., Guo, A., Lin, C., Grauman, K., Luo, J., 和 Bigham, J. P. VizWiz 大挑战：回答盲人的视觉问题。在 IEEE 计算机视觉与模式识别会议论文集，第 3608–3617 页，2018。

Han, S., Pool, J., Tran, J., 和 Dally, W. 学习权重和连接以实现高效神经网络。神经信息处理系统进展，28，2015。

Han, S., Mao, H., 和 Dally, W. J. 深度压缩：通过剪枝、训练量化和霍夫曼编码压缩深度神经网络。在 ICLR，2016。

Hudson, D. A. 和 Manning, C. D. GQA：一个用于真实世界视觉推理和组合问答的新数据集。在 CVPR，2019。

Jacob, B., Kligys, S., Chen, B., Zhu, M., Tang, M., Howard, A., Adam, H., 和 Kalenichenko, D. 神经网络的量化和训练以实现仅整数算术的高效推理。在 IEEE 计算机视觉与模式识别会议论文集，第 2704–2713 页，2018。

Jiang, A. Q., Sablayrolles, A., Mensch, A., Bamford, C., Chaplot, D. S., Casas, D. d. l., Bressand, F., Lengyel, G., Lample, G., Saulnier, L., 等。Mistral 7B。arXiv 预印本 arXiv:2310.06825，2023。

Jiang, A. Q., Sablayrolles, A., Roux, A., Mensch, A., Savary, B., Bamford, C., Chaplot, D. S., de las Casas, D., Hanna, E. B., Bressand, F., Lengyel, G., Bour, G., Lample, G., Lavaud, L. R., Saulnier, L., Lachaux, M.-A., Stock, P., Subramanian, S., Yang, S., Antoniak, S., Scao, T. L., Gervet, T., Lavril, T., Wang, T., Lacroix, T., 和 Sayed, W. E. Mixtral of experts，2024。

Kim, Y. J., Henry, R., Fahim, R., 和 Awadalla, H. H. 谁说大象不能跑：将大规模 MoE 模型带入云规模生产。arXiv 预印本 arXiv:2211.10017，2022。

Klimt, B. 和 Yang, Y. Enron 语料库：一个用于电子邮件分类研究的新数据集。在机器学习：ECML 2004：第 15 届欧洲机器学习会议，意大利比萨，2004 年 9 月 20-24 日。会议记录 15，第 217–226 页。Springer，2004。

Koh, J. Y., Salakhutdinov, R., 和 Fried, D. 将语言模型接地到图像以进行多模态生成。arXiv 预印本 arXiv:2301.13823，2023。

Li, B., Wang, R., Wang, G., Ge, Y., Ge, Y., 和 Shan, Y. Seed-Bench：用生成式理解基准测试多模态 LLM。arXiv 预印本 arXiv:2307.16125，2023a。

Li, J., Li, D., Savarese, S., 和 Hoi, S. BLIP-2：使用冻结图像编码器和大语言模型引导语言-图像预训练。arXiv 预印本 arXiv:2301.12597，2023b。

Li, R., Allal, L. B., Zi, Y., Muennighoff, N., Kocetkov, D., Mou, C., Marone, M., Akiki, C., Li, J., Chim, J., 等。StarCoder：愿源代码与你同在！arXiv 预印本 arXiv:2305.06161，2023c。

Li, Y., Gong, R., Tan, X., Yang, Y., Hu, P., Zhang, Q., Yu, F., Wang, W., 和 Gu, S. BRECQ：通过块重建推动训练后量化的极限。arXiv 预印本 arXiv:2102.05426，2021。

Li, Y., Du, Y., Zhou, K., Wang, J., Zhao, W. X., 和 Wen, J.-R. 评估大型视觉语言模型中的物体幻觉。arXiv 预印本 arXiv:2305.10355，2023d。

Lin, J., Chen, W\.-M., Lin, Y., Gan, C., Han, S., 等。MCUNet：物联网设备上的微型深度学习。神经信息处理系统进展，33:11711–11722，2020。

Lin, J., Yin, H., Ping, W., Lu, Y., Molchanov, P., Tao, A., Mao, H., Kautz, J., Shoeybi, M., 和 Han, S. VILA：关于视觉语言模型的预训练。在 CVPR，2024。

Liu, H., Li, C., Wu, Q., 和 Lee, Y. J. 视觉指令调优。2023a。

Liu, Y., Duan, H., Zhang, Y., Li, B., Zhang, S., Zhao, W., Yuan, Y., Wang, J., He, C., Liu, Z., 等。MMBench：你的多模态模型是全能选手吗？arXiv 预印本 arXiv:2307.06281，2023b。

Lu, P., Mishra, S., Xia, T., Qiu, L., Chang, K.-W., Zhu, S.-C., Tafjord, O., Clark, P., 和 Kalyan, A. 学习解释：通过思维链进行科学问答的多模态推理。神经信息处理系统进展，35:2507–2521，2022。

Merity, S., Xiong, C., Bradbury, J., 和 Socher, R. 指针哨兵混合模型，2016。

MLC-Team。MLC-LLM，2023。URL <https://github.com/mlc-ai/mlc-llm>。

Nagel, M., Baalen, M. v., Blankevoort, T., 和 Welling, M. 通过权重均衡和偏置校正进行无数据量化。在 IEEE/CVF 国际计算机视觉会议论文集，第 1325–1334 页，2019。

Nagel, M., Amjad, R. A., Van Baalen, M., Louizos, C., and Blankevoort, T. 向上还是向下？自适应舍入用于训练后量化。In International Conference on Machine Learning, pp. 7197–7206. PMLR, 2020.

Nagel, M., Fournarakis, M., Amjad, R. A., Bondarenko, Y., Van Baalen, M., and Blankevoort, T. 神经网络量化白皮书。arXiv preprint arXiv:2106.08295, 2021.

Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C., Mishkin, P., Zhang, C., Agarwal, S., Slama, K., Ray, A., et al. 训练语言模型通过人类反馈遵循指令。Advances in Neural Information Processing Systems, 35:27730–27744, 2022.

Park, G., Park, B., Kwon, S. J., Kim, B., Lee, Y., and Lee, D. nuqmm：量化矩阵乘法用于高效推理大规模生成语言模型。arXiv preprint arXiv:2206.09557, 2022.

Penedo, G., Malartic, Q., Hesslow, D., Cojocaru, R., Cappelli, A., Alobeidli, H., Pannier, B., Almazrouei, E., and Launay, J. RefinedWeb数据集用于Falcon LLM：仅用网络数据超越精选语料库。arXiv preprint arXiv:2306.01116, 2023.

Sanh, V., Webson, A., Raffel, C., Bach, S. H., Sutawika, L., Alyafeai, Z., Chaffin, A., Stiegler, A., Scao, T. L., Raja, A., et al. 多任务提示训练实现零样本任务泛化。arXiv preprint arXiv:2110.08207, 2021.

Scao, T. L., Fan, A., Akiki, C., Pavlick, E., Ilić, S., Hesslow, D., Castagné, R., Luccioni, A. S., Yvon, F., Gallé, M., et al. Bloom：一个1760亿参数的开源多语言语言模型。arXiv preprint arXiv:2211.05100, 2022.

Sheng, Y., Zheng, L., Yuan, B., Li, Z., Ryabinin, M., Fu, D. Y., Xie, Z., Chen, B., Barrett, C., Gonzalez, J. E., et al. 使用单个GPU进行大语言模型的高吞吐量生成推理。arXiv preprint arXiv:2303.06865, 2023.

Singh, A., Natarajan, V., Shah, M., Jiang, Y., Chen, X., Batra, D., Parikh, D., and Rohrbach, M. 迈向能够阅读的VQA模型。In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 8317–8326, 2019.

Taori, R., Gulrajani, I., Zhang, T., Dubois, Y., Li, X., Guestrin, C., Liang, P., and Hashimoto, T. B. Stanford Alpaca：一个遵循指令的Llama模型。<https://github.com/tatsu-lab/stanford_alpaca>, 2023.

Tillet, P., Kung, H.-T., and Cox, D. Triton：一种用于分块神经网络计算的中间语言和编译器。In Proceedings of the 3rd ACM SIGPLAN International Workshop on Machine Learning and Programming Languages, pp. 10–19, 2019.

Touvron, H., Lavril, T., Izacard, G., Martinet, X., Lachaux, M.-A., Lacroix, T., Rozière, B., Goyal, N., Hambro, E., Azhar, F., et al. Llama：开放且高效的基础语言模型。arXiv preprint arXiv:2302.13971, 2023a.

Touvron, H., Martin, L., Stone, K., Albert, P., Almahairi, A., Babaei, Y., Bashlykov, N., Batra, S., Bhargava, P., Bhosale, S., et al. Llama 2：开放基础和微调聊天模型。arXiv preprint arXiv:2307.09288, 2023b.

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., and Polosukhin, I. 注意力机制就是一切。Advances in neural information processing systems, 30, 2017.

Wang, H., Zhang, Z., and Han, S. Spatten：通过级联令牌和头剪枝实现高效稀疏注意力架构。CoRR, abs/2012.09852, 2020. URL <https://arxiv.org/abs/2012.09852>.

Wang, K., Liu, Z., Lin, Y., Lin, J., and Han, S. HAQ：硬件感知的自动混合精度量化。In CVPR, 2019.

Wei, J., Bosma, M., Zhao, V. Y., Guu, K., Yu, A. W., Lester, B., Du, N., Dai, A. M., and Le, Q. V. 微调语言模型是零样本学习者。arXiv preprint arXiv:2109.01652, 2021.

Wei, X., Zhang, Y., Zhang, X., Gong, R., Zhang, S., Zhang, Q., Yu, F., and Liu, X. 异常值抑制：推动低比特Transformer语言模型的极限，2022a. URL <https://arxiv.org/abs/2209.13325>.

Wei, X., Zhang, Y., Zhang, X., Gong, R., Zhang, S., Zhang, Q., Yu, F., and Liu, X. 异常值抑制：推动低比特Transformer语言模型的极限。arXiv preprint arXiv:2209.13325, 2022b.

Wei, X., Zhang, Y., Li, Y., Zhang, X., Gong, R., Guo, J., and Liu, X. 异常值抑制+：通过等效和最优移位与缩放实现大语言模型的精确量化。arXiv preprint arXiv:2304.09145, 2023.

Xiao, G., Lin, J., Seznec, M., Demouth, J., and Han, S. SmoothQuant：大语言模型的准确高效训练后量化。arXiv preprint arXiv:2211.10438, 2022.

Yao, Z., Aminabadi, R. Y., Zhang, M., Wu, X., Li, C., and He, Y. ZeroQuant：高效且经济的大规模Transformer训练后量化，2022. URL <https://arxiv.org/abs/2206.01861>.

Yu, W., Yang, Z., Li, L., Wang, J., Lin, K., Liu, Z., Wang, X., and Wang, L. MM-Vet：评估大型多模态模型的综合能力。arXiv preprint arXiv:2308.02490, 2023.

Zhang, R., Han, J., Zhou, A., Hu, X., Yan, S., Lu, P., Li, H., Gao, P., and Qiao, Y. Llama-adapter：使用零初始化注意力高效微调语言模型。arXiv preprint arXiv:2303.16199, 2023.

Zhang, S., Roller, S., Goyal, N., Artetxe, M., Chen, M., Chen, S., Dewan, C., Diab, M., Li, X., Lin, X. V., Mihaylov, T., Ott, M., Shleifer, S., Shuster, K., Simig, D., Koura, P. S., Sridhar, A., Wang, T., and Zettlemoyer, L. OPT：开放预训练Transformer语言模型，2022. URL <https://arxiv.org/abs/2205.01068>.
