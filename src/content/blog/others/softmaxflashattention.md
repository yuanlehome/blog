---
lang: zh
translatedFrom: en
title: 从在线Softmax到FlashAttention
slug: softmaxflashattention
date: '2026-01-28'
tags: []
status: published
source_url: 'https://courses.cs.washington.edu/courses/cse599m/23sp/notes/flashattn.pdf'
source_author: courses.cs.washington.edu
imported_at: '2026-01-28T19:33:17.244Z'
source:
  title: courses.cs.washington.edu
  url: 'https://courses.cs.washington.edu/courses/cse599m/23sp/notes/flashattn.pdf'
---

# 从在线Softmax到FlashAttention

作者：ZIHAO YE

邮箱：<zhye@cs.washington.edu>

2023年5月11日

华盛顿大学CSE 599M 2023年春季：ML for ML Systems

FlashAttention的关键创新$[1]$是使用类似于在线Softmax（Online Softmax）的思想$[3]$来分块自注意力计算，从而我们可以在不访问GPU全局内存以获取中间logits和注意力分数的情况下，融合整个多头注意力层。在本笔记中，我将简要解释为什么分块自注意力计算是非平凡的，以及如何从在线Softmax技巧推导出FlashAttention计算。

我们感谢Andrew Gu审阅本笔记。

## 1 自注意力

自注意力的计算可以总结为（我们忽略头和批次，因为这些维度上的计算是完全并行的，为简化起见，我们也省略了注意力掩码和缩放因子$\frac{1}{\sqrt{D}}$等细节）：

$O=softmax(Q K^{T})V$

其中Q、K、V、O都是形状为$(L, D)$的2D矩阵，其中L是序列长度，D是每个头的维度（即头维度），softmax应用于最后一个维度（列）。

计算自注意力的标准方法是将计算分解为几个阶段：

$X\;=\;Q K^{T}$

$A=\operatorname{softmax}(X)$

$O=AV$

我们称X矩阵为pre-softmax logits，A矩阵为注意力分数，O矩阵为输出。

FlashAttention的一个惊人事实是，我们不需要在全局内存中具体化X和A矩阵，而是将公式1中的整个计算融合在单个CUDA内核中。这要求我们设计一个算法，仔细管理片上内存（如流算法），因为NVIDIA GPU的共享内存很小。

对于经典算法如矩阵乘法，分块用于确保片上内存不超过硬件限制。图1提供了一个例子。在内核执行期间，只有$3T^{2}$个元素存储在片上，无论矩阵形状如何。这种分块方法是有效的，因为加法是结合律的，允许将整个矩阵乘法分解为许多块状矩阵乘法的和。

然而，自注意力包含一个softmax算子，它不是直接结合律的，使得难以像图1那样简单地分块自注意力。有没有办法使softmax具有结合律？
