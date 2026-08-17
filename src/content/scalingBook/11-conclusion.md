---
title: "总结与延伸阅读"
description: "感谢阅读！本章还将提供一些参考资料，供你进一步学习。"
chapter: 11
order: 11
part: 4
partTitle: "总结与附加内容"
sourcePath: "conclusion.md"
sourceCommit: "44109cacac9c5a9809a81c68ae4d45d7d2632ea6"
---

<span id="conclusions-and-further-reading"></span>

# 总结与延伸阅读

**感谢你读完整本书，也恭喜你一路坚持到了最后。** 在结束之前，我们想先致谢几位贡献者：

<span id="acknowledgments"></span>

## 致谢

这份文档凝聚了 Google DeepMind 许多人的大量共同投入，我们想在此简要向他们致谢！

- James Bradbury、Reiner Pope、Noam Shazeer 和 Blake Hechtman 最早推导出了本书中的许多思想，也是较早从系统视角理解 Transformer 的人。
- Sholto Douglas 撰写了这份文档的第一版，并推动了项目启动。可以说，他比任何人都更深刻地塑造了本文档的整体叙事。
- Jacob Austin 牵头将最初的粗略笔记打磨成更加完善、全面的作品。他承担了文档编辑、格式整理与发布的大量工作，并协调了其他作者的贡献。
- 大多数插图和动画由 Anselm Levskaya 与 Charlie Chen 制作。
- Charlie Chen 撰写了推理部分，并绘制了许多推理相关插图。
- Roy Frostig 在发布、编辑以及整个过程中的许多其他环节提供了帮助。

我们还想感谢在整个过程中提出关键反馈的许多人，特别是 Zak Stone、Nikhil Sethi、Caitlin Stanton、Alek Dimitriev、Sridhar Lakshmanamurthy、Albert Magyar、Diwakar Gupta、Jeff Dean、Corry Wang、Matt Johnson、Peter Hawkins，以及其他许多人。感谢 Ruiqi Gao 在 HTML 格式方面提供帮助。

**感谢大家！**

在离开之前，你或许还会喜欢阅读新增的 NVIDIA GPU [第 12 章](../12-gpus/#how-to-think-about-gpus)！

<span id="further-reading"></span>

## 延伸阅读

与本书相关的文章很多，其中包括：

- [**TPU Deep Dive**](https://henryhmko.github.io/posts/tpu/tpu.html)：一篇与本书精神相通、非常精彩且深入的 TPU 架构解析。
- [**Domain specific architectures for AI inference**](https://fleetwood.dev/posts/domain-specific-architectures)：一篇与本书精神相通、深入讨论硬件和模型的文章。
- [**A Domain-Specific Supercomputer for Training Deep Neural Networks**](https://dl.acm.org/doi/pdf/10.1145/3360307)：TPU 领域最早的一批经典论文之一，包含许多本书未涉及的 Google TPU 项目细节。
- [**Making Deep Learning Go Brrrr From First Principles**](https://horace.io/brrr_intro.html)：一篇更侧重 GPU 与 PyTorch 的 LLM Roofline 模型和性能工程教程。
- [**Writing TPU Kernels with Pallas**](https://jax.readthedocs.io/en/latest/pallas/tpu/details.html)：如今，TPU 编程越来越多地需要用 Pallas 编写自定义内核。本系列讨论如何编写内核，以及许多本书没有提及的底层 TPU 细节。
- [**How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance: a Worklog**](https://siboehm.com/articles/22/CUDA-MMM)：尽管专门面向 GPU 和 CUDA，但这是一篇非常出色的博客文章，展示了如何在 CUDA 中优化矩阵乘法内核。它也可以作为深入理解 TPU 与 GPU 差异的良好材料。
- [**Distributed arrays and automatic parallelization**](https://jax.readthedocs.io/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html)：一份非常好的 JAX 并行 API 指南，也很适合用来学习如何真正实现本书讨论过的一些思想。
- [**Rafi Witten's High Performance LLMs 2024 Class**](https://github.com/rwitten/HighPerfLLMs2024)：我们以前的同事 Rafi 开设了一门很棒的 TPU 性能工程课程，所有幻灯片都发布在 GitHub 上。课程对许多主题的讲解比本书更深入。
- [**\[2211.05102\] Efficiently Scaling Transformer Inference**](https://arxiv.org/abs/2211.05102)：一篇详细讨论 Transformer 推理数学原理的论文，也是本文档许多内容的灵感来源。
- [**Huggingface Ultra-Scale Playbook**](https://huggingface.co/spaces/nanotron/ultrascale-playbook)：某种意义上可视为本书的 GPU 版本，更深入地讨论了 PyTorch 如何实现训练中的并行技术和内存节省技术。
- [**Transformer Inference Arithmetic**](https://kipp.ly/transformer-inference-arithmetic/)：一篇博客，包含许多与本书相同的思想以及一些出色的插图。
- [**Stanford CS336 Slides and Videos**](https://stanford-cs336.github.io/spring2025/index.html#coursework)：斯坦福的一门优秀课程，涵盖 LLM 训练和推理服务的许多细节，并配有实用练习。其中作业 1 和作业 2 尤其相关。
- [**Stas Bekman's ML Engineering Handbook**](https://github.com/stas00/ml-engineering)：一份实践性很强的机器学习基础设施指南，覆盖了本书未讨论的主题，例如如何与云服务商谈判、集群管理，以及如何对 GPU 吞吐量进行实测。
- [**ezyang's blog**](https://blog.ezyang.com/2026/01/computing-sharding-with-einsum/)：一位 PyTorch 负责人的博客，讨论分片与 PyTorch 的方方面面，其中包括一份 [PyTorch 内部机制指南](https://blog.ezyang.com/2019/05/pytorch-internals/)和一篇[分片矩阵乘法解析](https://blog.ezyang.com/2026/01/computing-sharding-with-einsum/)。博客里还有许多其他优质内容。
- [**The Anatomy of Collective Communication**](https://www.aleksagordic.com/blog/collective-operations)：一篇与本书精神相通、清晰讲解 GPU 和 TPU 集合通信的文章。它对 N 维集合通信和 GPU 集合通信的介绍比本书更出色。

这一领域仍然非常缺少全面而系统的写作，因此我们希望这份文稿能够鼓励更多人投入其中！我们也相信，这是一个很有前景、值得学习和研究的方向。在许多情况下，即使手头没有大量硬件加速器，也仍然可以开展相关工作。

<span id="feedback"></span>

## 反馈

欢迎留下评论或问题，帮助我们继续改进这份文档。你可以通过 jacobaustin123 [at] gmail [dot] com 联系通讯作者 Jacob Austin，也可以在 [GitHub](https://github.com/jax-ml/scaling-book) 上提交 issue、pull request 或发起 discussion 来建议修改。
