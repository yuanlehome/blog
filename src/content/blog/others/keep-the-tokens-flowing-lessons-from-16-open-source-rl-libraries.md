---
title: 'Keep the Tokens Flowing: Lessons from 16 Open-Source RL Libraries'
slug: keep-the-tokens-flowing-lessons-from-16-open-source-rl-libraries
date: '2026-03-11'
tags: []
status: published
source_url: 'https://huggingface.co/blog/async-rl-training-landscape'
source_author: huggingface.co
imported_at: '2026-03-15T08:44:25.996Z'
source:
  title: huggingface.co
  url: 'https://huggingface.co/blog/async-rl-training-landscape'
cover: >-
  /images/others/keep-the-tokens-flowing-lessons-from-16-open-source-rl-libraries/001-93bde3e0.webp
lang: zh
translatedFrom: en
---
[返回文章](/blog)

# 保持令牌流动：来自16个开源强化学习（RL）库的经验教训

发布于2026年3月10日

[在GitHub上更新](https://github.com/huggingface/blog/blob/main/async-rl-training-landscape.md)

[点赞](/login?next=%2Fblog%2Fasync-rl-training-landscape)

[49](/login?next=%2Fblog%2Fasync-rl-training-landscape)

- [![](/images/others/keep-the-tokens-flowing-lessons-from-16-open-source-rl-libraries/001-93bde3e0.webp)](/lvwerra "lvwerra")
- [![](/images/others/keep-the-tokens-flowing-lessons-from-16-open-source-rl-libraries/002-9bc33468.webp)](/clem "clem")
- [![](/images/others/keep-the-tokens-flowing-lessons-from-16-open-source-rl-libraries/003-3d36de8a.svg)](/sabman "sabman")
- [![](/images/others/keep-the-tokens-flowing-lessons-from-16-open-source-rl-libraries/004-ac0e506b.webp)](/yjernite "yjernite")
- [![](/images/others/keep-the-tokens-flowing-lessons-from-16-open-source-rl-libraries/005-0936a580.webp)](/lewtun "lewtun")
- [![](/images/others/keep-the-tokens-flowing-lessons-from-16-open-source-rl-libraries/006-5cbe078f.webp)](/FL33TW00D "FL33TW00D")
- +43

[![Amine Dirhoussi的头像](/images/others/keep-the-tokens-flowing-lessons-from-16-open-source-rl-libraries/007-e6c7cf76.webp)](/aminediroHF)

[Amine Dirhoussi](/aminediroHF)

[aminediroHF](/aminediroHF)

[关注](/aminediroHF)

[![Quentin Gallouédec的头像](/images/others/keep-the-tokens-flowing-lessons-from-16-open-source-rl-libraries/008-56f3acdf.webp)](/qgallouedec)

[Quentin Gallouédec](/qgallouedec)

[qgallouedec](/qgallouedec)

[关注](/qgallouedec)

[![Kashif Rasul的头像](/images/others/keep-the-tokens-flowing-lessons-from-16-open-source-rl-libraries/009-467464d6.webp)](/kashif)

[Kashif Rasul](/kashif)

[kashif](/kashif)

[关注](/kashif)

[![Lewis Tunstall的头像](/images/others/keep-the-tokens-flowing-lessons-from-16-open-source-rl-libraries/005-0936a580.webp)](/lewtun)

[Lewis Tunstall](/lewtun)

[lewtun](/lewtun)

[关注](/lewtun)

[![Edward Beeching的头像](/images/others/keep-the-tokens-flowing-lessons-from-16-open-source-rl-libraries/010-895c5684.webp)](/edbeeching)

[Edward Beeching](/edbeeching)

[edbeeching](/edbeeching)

[关注](/edbeeching)

[![Albert Villanova del Moral的头像](/images/others/keep-the-tokens-flowing-lessons-from-16-open-source-rl-libraries/011-2896e31c.webp)](/albertvillanova)

[Albert Villanova del Moral](/albertvillanova)

[albertvillanova](/albertvillanova)

[关注](/albertvillanova)

[![Nouamane Tazi的头像](/images/others/keep-the-tokens-flowing-lessons-from-16-open-source-rl-libraries/012-19d69556.webp)](/nouamanetazi)

[Nouamane Tazi](/nouamanetazi)

[nouamanetazi](/nouamanetazi)

[关注](/nouamanetazi)

[![Leandro von Werra的头像](/images/others/keep-the-tokens-flowing-lessons-from-16-open-source-rl-libraries/001-93bde3e0.webp)](/lvwerra)

[Leandro von Werra](/lvwerra)

[lvwerra](/lvwerra)

[关注](/lvwerra)

[![Sergio Paniego的头像](/images/others/keep-the-tokens-flowing-lessons-from-16-open-source-rl-libraries/013-dc8511e6.webp)](/sergiopaniego)

[Sergio Paniego](/sergiopaniego)

[sergiopaniego](/sergiopaniego)

[关注](/sergiopaniego)

<!-- HTML_TAG_START -->

> **TL;DR**——对于那些没时间阅读5000字关于异步强化学习（RL）基础设施文章的人（我们理解，您有模型要训练）：
>
> - **问题：**&#x5728;同步强化学习（RL）训练中，数据生成（模型推理以创建数据样本）主导了实际时间——在320亿（32B）参数模型上，单个包含32K令牌的批次可能需要*数小时*，而用于训练的GPU在此期间保持空闲。
> - **大家共同采用的解决方案：**&#x5C06;推理和训练解耦（分离）到不同的GPU池，通过rollout缓冲区（模型输出的临时存储）连接它们，并异步（无需等待）传输权重，这样双方都不必等待对方。
> - **我们调查了16个开源库**，这些库实现了这种模式，并在7个维度上进行了比较：编排原语、缓冲区设计、权重同步协议、陈旧性管理、部分rollout处理、LoRA支持以及分布式训练后端。
> - **关键发现：**&#x52;ay主导编排（在调查的16个分布式计算库中占8个）。NCCL（NVIDIA Collective Communications Library）广播是传输模型权重的默认方法。陈旧性管理指如何处理过时的数据样本，范围从简单丢弃旧样本到使用高级重要性采样校正。LoRA（Low-Rank Adaptation）训练支持较少。分布式MoE（Mixture of Experts）支持是新兴的差异化因素。
>
> 如果您想直接跳到精华部分，[这里是完整的比较表](#4-global-overview-sixteen-libraries-at-a-glance)（无需阅读，我们不会评判）。
>
> 但说真的，如果您继续阅读，可能会学到一两件事，了解为什么您的GPU有60%的时间处于空闲状态。

***

**点击展开目录**

- [1. 动机：从同步强化学习（RL）训练到异步架构](#1-motivation-from-synchronous-rl-training-to-async-architectures)

  - [1.1 TRL当前的强化学习（RL）训练方式](#11-how-trl-does-rl-training-today)
  - [1.2 共置与解耦训练](#12-colocated-vs-disaggregated-training)
  - [1.3 生成瓶颈](#13-the-generation-bottleneck)
  - [1.4 核心洞察](#14-the-core-insight)

- [2. 调查的库](#2-libraries-surveyed)

- [3. 比较框架：七个维度](#3-the-comparison-framework-seven-axes)

  - [维度1：编排与并发原语](#axis-1-orchestration--concurrency-primitive)
  - [维度2：Rollout缓冲区设计](#axis-2-rollout-buffer-design)
  - [维度3：权重同步协议](#axis-3-weight-synchronisation-protocol)
  - [维度4：陈旧性管理](#axis-4-staleness-management)
  - [维度5：部分Rollout处理](#axis-5-partial-rollout-handling)
  - [维度6：LoRA训练支持](#axis-6-lora-training-support)
  - [维度7：分布式训练后端与并行性](#axis-7-distributed-training-backend--parallelism)

- [4. 全局概览：十六个库一览](#4-global-overview-sixteen-libraries-at-a-glance)

- [5. 下一波浪潮：设计启示](#5-the-next-wave-design-implications)

  - [5.1 无批评者算法：内存释放，但权重同步压力增加](#51-critic-free-algorithms-memory-freed-but-weight-sync-pressure-increases)
  - [5.2 过程奖励：新的同步屏障](#52-process-rewards-a-new-synchronisation-barrier)
  - [5.3 多智能体协同进化：拖后腿问题加剧](#53-multi-agent-co-evolution-the-straggler-problem-compounds)
  - [5.4 训练-推理不匹配：Deepseek v3.2 MoE案例研究](#54-training-inference-mismatch-the-deepseek-v32-moe-case-study)
  - [5.5 蒸馏：同一异步问题，不同名称](#55-distillation-the-same-async-problem-under-a-different-name)

- [6. TRL异步训练器的设计选择](#6-design-choices-for-trls-async-trainer)

  - [设计原则：保持编排轻量级](#design-principle-keep-orchestration-lightweight)
  - [1. 有界队列，按令牌`model_version`（无双重缓冲）](#1-bounded-queue-with-per-token-model_version-no-double-buffering)
  - [2. 使用打包传输的NCCL权重同步](#2-nccl-weight-sync-with-packed-transfers)
  - [3. 支持智能体工作负载的部分Rollout](#3-partial-rollout-support-for-agentic-workloads)

## 1. 动机：从同步强化学习（RL）训练到异步架构

异步强化学习（RL）训练已成为大规模后训练的主导范式。现代后训练中的几个趋势使得同步训练循环几乎无法扩展：

- **推理模型的长rollout。**&#x601D;维链训练产生非常长的rollout，单个同步生成批次在单个GPU上可能需要数小时才能完成。在此期间，训练GPU完全空闲。
- **无价值函数训练器，如GRPO**使用组相对优势。这意味着每个提示生成多达G倍的rollout，并且整个批次受组中最慢完成项的制约。
- **智能体强化学习（RL）训练的兴起。**&#x5F53;模型在多轮轨迹中与工具、沙箱和外部环境交互时，rollout长度和延迟变得高度可变。一个简单的API调用可能在几秒内返回，而一个包含工具使用的复杂推理链可能运行数分钟或数小时。MiniMax的[Forge](https://www.minimax.io/news/forge-scalable-agent-rl-framework-and-algorithm)框架，用于训练MiniMax-M2.5，说明了实践中达到的规模：上下文长度高达200K令牌，超过十万个不同的智能体脚手架和环境，每日吞吐量达数百万样本。在这种规模下，生成和训练之间的任何同步屏障都会成为严重瓶颈。仅拖后腿问题（少数慢rollout阻塞整个批次）就可能导致数百个GPU空闲。

开源生态系统已汇聚于一个共同的架构响应：将推理与训练解耦到不同的GPU池，通过rollout缓冲区连接它们，并让双方并发运行。

我们正在为[TRL](https://github.com/huggingface/trl)开发一个新的异步训练器，这是最广泛使用的模型后训练库之一。为了指导我们的设计，我们调查了**十六个开源库**，这些库从一开始就围绕异步训练构建，并在**七个维度**上进行了比较：编排原语、缓冲区设计、权重同步协议、陈旧性管理、部分rollout处理、LoRA支持以及分布式训练后端。本文提炼了我们从该调查中提取的设计原则。

除了强化学习（RL）之外，对异步基础设施的需求日益明显。例如，**策略蒸馏**，其中学生生成序列，教师对其进行评分，这反映了GRPO，但将奖励函数替换为教师前向传播。认识到这种结构相似性，本调查中的所有内容同样适用于异步蒸馏。我们将在第5节回到这个更广泛的要点。

### 1.1 TRL如何进行RL训练

TRL当前的`GRPOTrainer`在一个同步的`training_step()`调用中实现了完整的GRPO循环（提示采样、生成、奖励评分、优势计算、梯度更新和权重同步）。这种设计简单且正确，但无法重叠生成与训练，导致显著的GPU利用率不足。

查看`GRPOTrainer`，在每个训练步骤中，我们依次有以下阶段：

1. **提示采样：**&#x4ECE;数据集中抽取一批提示。这里没什么特别的，我们继续。
1. **生成**，调用`model.generate()`（或向前端vLLM服务器发送请求）以生成每个提示的G个完成。这是自回归的，并主导了实际时间。
1. **奖励评分：**&#x6839;据一个或多个奖励函数评估每个完成。
1. **优势计算**
1. **前向和后向传播：**&#x8BA1;算裁剪策略梯度损失并进行反向传播。
1. **优化器步骤**，更新模型权重。
1. **权重同步**，将更新后的权重推送到推理引擎（vLLM），以便下一次生成使用新策略。

每个阶段**阻塞**直到完成，然后下一个阶段才开始。时间线如下所示：

![同步TRL训练时间线](/images/others/keep-the-tokens-flowing-lessons-from-16-open-source-rl-libraries/014-d966e2aa.png)

TRL提供`steps_per_generation`配置选项，以在多个梯度步骤中重用一组rollout（时间重用），分摊生成成本。但生成调用本身仍然是完全同步和阻塞的；训练器在批次中每个完成都完成之前无法开始梯度计算。

该库还支持以`server`模式运行vLLM作为独立进程。它在生成期间释放训练GPU，但两个硬同步屏障仍然存在：**HTTP调用直到所有完成返回**，以及权重同步在传输期间阻塞训练器和vLLM。

### 1.2 共置与分离训练

在讨论异步训练之前，理解使用独立推理引擎进行RL训练的两种部署拓扑至关重要：

- **共置模式**将推理和训练放在**同一组GPU上**。单个GPU（或TP组）同时持有训练模型（在FSDP或ZeRO下）和推理引擎（vLLM或SGLang）。一次只有一个角色处于活动状态：在生成期间，训练模型的参数可能被卸载或重新分片为推理友好的布局（例如，从FSDP分片到vLLM的张量并行布局）；在训练期间，推理引擎被暂停或休眠。权重“同步”基本上是免费的；最多是在同一GPU上进行原地重新分片，而不是网络传输。共置模式的优点是简单性和成本；您需要更少的GPU总数。根本限制是**推理和训练无法重叠**。例如，这里是Trl与vllm在`colocate_mode`：

![TRL与vLLM在共置模式](/images/others/keep-the-tokens-flowing-lessons-from-16-open-source-rl-libraries/015-fe40cd02.png)

- **分离模式**将推理和训练放在**独立的GPU池上**。推理池持续运行vLLM或SGLang；训练池持续运行优化器。两个池通过权重同步协议（NCCL广播、文件系统检查点、HTTP等）和数据传输机制（Ray对象存储、Redis流、共享内存等）进行通信。

分离模式的最大优势是**推理和训练可以并发运行**。当训练器在批次N上计算梯度时，推理池已经在为批次N+K生成rollout，从而实现异步训练。然而，这种好处是有代价的：需要额外的GPU。

并发性、异步性和并行性是常被混淆的不同概念。在本文中，当我们&#x8BF4;**“异步训练”**&#x65F6;，我们特指：**生成和训练并行运行，具有有效的重叠**；推理池在训练池计算当前批次梯度的同时，正在生成下一批rollout。这本质上是分离模式的能力。共置模式可以通过优化如睡眠/唤醒内存管理或快速原地重新分片来加速推理，但无法实现真正的同步重叠；推理和训练仍然在同一GPU上轮流进行。本调查中实现有意义异步重叠的每个库都使用分离模式作为基础。

### 1.3 生成瓶颈

在推理模型的RL训练中，**自回归生成主导实际时间**。单个数学或编码任务的rollout可以产生8K–64K令牌的思维链推理（参见[QED-Nano rollout lengths](https://huggingface.co/spaces/lm-provers/qed-nano-blogpost#outcome-reward-rl-with-long-response-lengths)）。

为了具体说明这一点，考虑[vLLM在单个H100 80GB GPU上的基准测试](https://www.databasemart.com/blog/vllm-gpu-benchmark-h100)（bf16，无量化，离线吞吐量模式）。一个**7B模型**（DeepSeek-R1-Distill-Qwen-7B）实现约6,300输出令牌/秒的聚合吞吐量；一个**32B模型**（DeepSeek-R1-Distill-Qwen-32B）降至约1,200输出令牌/秒。这些是*总*吞吐量，即推理引擎每秒可以推送的令牌数，无论有多少序列共享GPU。

现在考虑一个典型的GRPO训练步骤：**G=8个完成/提示 × 64个提示/批次 = 512个rollout**。生成需要多长时间？

| 每个rollout的输出长度 | 总输出令牌数（512个rollout） | 在1×H100上的时间（7B @ \~6K令牌/秒） | 在1×H100上的时间（32B @ \~1.2K令牌/秒） |
| :------------- | :------------------ | :------------------------- | :---------------------------- |
| 2K令牌（短CoT）     | 约1M令牌               | **约3分钟**                   | **约14分钟**                     |
| 8K令牌（中CoT）     | 约4M令牌               | **约11分钟**                  | **约56分钟**                     |
| 32K令牌（长CoT）    | 约16M令牌              | **约45分钟**                  | **约3.7小时**                    |

即使在短端（2K令牌生成，使用7B模型），仅生成就消耗每个训练步骤数分钟。在长端，前沿推理模型越来越多地在此操作，单个生成阶段可能需要*数小时*在单个GPU上。扩展到8个推理GPU可将这些时间大致除以8倍（假设吞吐量线性扩展），但即便如此，32B模型上的32K令牌展开仍需要约28分钟每步。

“**落后者问题（straggler problem）**&#x8FDB;一步加剧了这一点。在基于组的算法如GRPO中，每个提示采样G个完成。批次无法继续，直到*最慢*的完成结束。思维链输出长度变化很大；单个提示可能产生从1K到32K令牌不等的完成。批次受最长完成限制，连续批处理仅部分缓解此问题：较短序列释放槽位用于新工作，但*最后一个*GRPO组中的序列仍会阻塞该组的奖励计算和训练步骤。

### &#x20;1.4 核心洞察

本调查中的每个库都独立地收敛于相同的架构原则：**物理上将推理GPU与训练GPU分离，并异步推送权重**，使得生成永不停止，训练永不等待。

推理池持续运行，将完成的展开送入缓冲区。训练池从缓冲区拉取数据，计算梯度更新，并定期将新权重推送回推理池以保持同步。两个循环以各自的速度运行，通过缓冲区解耦。

这种设置高度可扩展，但引入了新一类问题：陈旧性（在旧策略下生成的展开）、权重同步开销、部分展开处理等。本文其余部分详细剖析了当前开源库如何解决这些问题。

***

## &#x20;2. 调查的库

| 库                 | 组织                    | 仓库                                                                                       | GitHub ⭐ (2026年3月) |
| ----------------- | --------------------- | ---------------------------------------------------------------------------------------- | -----------------: |
| **AReaL**         | inclusionAI/Ant Group | [github.com/inclusionAI/AReaL](https://github.com/inclusionAI/AReaL)                     |              4,338 |
| **ART**           | CoreWeave             | [github.com/OpenPipe/ART](https://github.com/OpenPipe/ART)                               |              8,952 |
| **Atropos**       | NousResearch          | [github.com/NousResearch/atropos](https://github.com/NousResearch/atropos)               |                878 |
| **MILES**         | radixark              | [github.com/radixark/miles](https://github.com/radixark/miles)                           |                950 |
| **NeMo-RL**       | NVIDIA                | [github.com/NVIDIA-NeMo/RL](https://github.com/NVIDIA-NeMo/RL)                           |              1,383 |
| **OAT**           | SAIL-SG               | [github.com/sail-sg/oat](https://github.com/sail-sg/oat)                                 |                637 |
| **open-instruct** | AI2 (AllenAI)         | [github.com/allenai/open-instruct](https://github.com/allenai/open-instruct)             |              3,611 |
| **PipelineRL**    | ServiceNow            | [github.com/ServiceNow/PipelineRL](https://github.com/ServiceNow/PipelineRL)             |                374 |
| **PRIME-RL**      | PrimeIntellect        | [github.com/PrimeIntellect-ai/prime-rl](https://github.com/PrimeIntellect-ai/prime-rl)   |              1,114 |
| **ROLL**          | Alibaba               | [github.com/alibaba/ROLL](https://github.com/alibaba/ROLL)                               |              2,921 |
| **SkyRL**         | NovaSky-AI            | [github.com/NovaSky-AI/SkyRL](https://github.com/NovaSky-AI/SkyRL)                       |              1,664 |
| **SLIME**         | THUDM                 | [github.com/THUDM/slime](https://github.com/THUDM/slime)                                 |              4,595 |
| **TorchForge**    | Meta                  | [github.com/meta-pytorch/torchforge](https://github.com/meta-pytorch/torchforge)         |                632 |
| **Tunix**         | Google                | [github.com/google/tunix](https://github.com/google/tunix)                               |              2,175 |
| **verl**          | ByteDance             | [github.com/verl-project/verl](https://github.com/verl-project/verl)                     |             19,673 |
| **verifiers-rl**  | PrimeIntellect        | [github.com/PrimeIntellect-ai/verifiers](https://github.com/PrimeIntellect-ai/verifiers) |              3,876 |

***

## &#x20;3. 比较框架：七个维度

为了理解快速扩展的异步RL库生态系统，我们提出了七个正交的比较维度。每个维度捕捉了一个影响库性能、复杂性和权衡的基本设计决策。

- **维度1 – 编排与并发原语：** 分布式组件如何协调（Ray actors、asyncio、pub/sub、HTTP）。
- **维度2 – 展开缓冲区设计：** 展开如何从推理流向训练。
- **维度3 – 权重同步协议：** 更新后的权重如何到达推理服务器，以及系统是否必须暂停以接受它们或继续生成。
- **维度4 – 陈旧性管理：** 如何处理离策略展开：版本拒绝、深度限制或重要性采样校正。
- **维度5 – 部分展开处理：** 当权重更新在序列中途到达时，正在进行的生成会发生什么。
- **维度6 – LoRA训练支持：** 通用LoRA支持以及是否仅适配器参数可以训练和同步，实现亚毫秒级权重传输。
- **维度7 – 分布式训练后端与并行性：** 训练使用何种并行策略，限制了最大模型大小。

### &#x20;维度1：编排与并发原语

*系统如何协调其分布式组件？*

编排框架的选择决定了编程模型、故障语义和可扩展性上限。与其列出每个库的实现细节，该领域清晰地分解为四种**编排类型（orchestration types）**，这些基本协调范式在抽象级别、故障模型和部署要求上有所不同：

| 编排类型            | 定义                                                                              | 库                                                                                       | 权衡                                                       |
| --------------- | ------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------- | -------------------------------------------------------- |
| **分布式Actor模型**  | 组件是*actors*，具有邮箱的隔离有状态进程，由运行时管理，处理调度、资源放置、容错和对象传输。通信通过异步RPC / futures / 对象存储进行。 | **Ray：** verl、SkyRL、NeMo-RL、SLIME、MILES、ROLL、OAT、open-instruct。**Monarch：** TorchForge。 | 最丰富的抽象；开箱即用地解决调度和容错。增加了非平凡的运行时依赖和框架特定的调试开销。              |
| **原生Python并发**  | 组件是线程、协程（`asyncio`）、`threading`原语、`multiprocessing`子进程和队列。无外部编排运行时。             | verifiers-rl、PipelineRL（池内）、ART（`asyncio` + 子进程代理）、AReaL（`asyncio`基于事件循环）               | 最小依赖，易于调试，完全控制。限于单节点，除非与额外IPC（Redis、HTTP、NCCL）配对用于多节点通信。 |
| **Pub/Sub消息总线** | 组件是通过仅追加流或消息队列通信的解耦生产者和消费者。本身不是编排，而是*数据传输层（data transport layer）*，连接独立运行的池。     | PipelineRL（池间：Redis`XADD`/`XREAD`流用于多节点，仅追加JSONL文件用于单节点）                                | 跨池边界清晰解耦，无需RPC。不管理进程生命周期、调度或故障恢复；必须与另一种编排类型配对。           |
| **HTTP微服务**     | 组件是通过REST API通信的独立服务。语言无关，最大解耦。                                                 | Atropos                                                                                 | 任何推理服务器，任何语言，零共享状态。最高延迟（如果使用NCCL）；无共享对象存储；容错是用户的责任。      |

> **关于Tunix的说明：** Tunix（Google）使用JAX原生网格模型，带有`ThreadPoolExecutor`用于异步重叠和`jax.device_put`用于跨网格权重传输。它在架构上与PyTorch生态系统足够不同，以至于在编排方面进行直接比较没有意义；它存在于XLA/TPU世界中，拥有自己的协调原语。

上表揭示了一个显著的模式：**调查的十六个库中有八个使用Ray作为其编排骨干**。这并非巧合；它反映了actor模型与RL训练结构之间的深层架构契合。[Anyscale（Ray背后的公司）对开源LLM RL库的调查](https://www.anyscale.com/blog/open-source-rl-libraries-for-llms)证实了这种趋同。大规模RL训练涉及本质上异构的组件（推理引擎、训练引擎、环境、奖励模型），这些组件必须在集群中编排，通常在不同硬件类型上，具有不同的扩展需求和故障模式。Ray的actor模型直接映射到这一点：

1. **Actor隔离和异构资源。**&#x6BCF;个RL组件（vLLM推理服务器、FSDP训练器、奖励模型、环境池）成为一个具有自身资源需求的Ray actor（`num_gpus`，`num_cpus`，`memory`）。放置组提供对GPU亲和性的细粒度控制，无需手动SSH/torchrun编排。

1. **调度和自动扩展。**&#x52;ay的调度器处理在集群中放置异构actor的组合问题。当生成需要比训练多8倍的GPU小时时，你可以直接告诉Ray独立扩展你的推理actor。

1. **容错性。**&#x957F;时间的RL训练运行（数天到数周）容易受到GPU故障、OOM终止和网络分区的影响。Ray的actor重启策略和对象存储复制提供了弹性，这在使用原始`asyncio`和`multiprocessing`时需要大量自定义基础设施。容错性的具体示例：`open-instruct`，例如，依赖Ray的actor监督来从vLLM引擎在rollout中途崩溃中恢复。

1. **用于零拷贝数据传输的对象存储。**&#x52;ollout数据可能很大，对于非常长上下文的推理，每批次可达数十GB。Ray的共享内存对象存储支持同一节点上actor之间的零拷贝传输，避免了通常伴随`multiprocessing.Queue`方法的序列化开销。

1. **生态系统成熟度。**&#x52;ay自2017年以来已在数千个GPU的生产部署中经过大规模实战测试。调试开销是真实的（Ray仪表板、分布式堆栈跟踪、放置组故障），但替代方案——从头构建等效协调——在多节点规模上更糟。也就是说，Ray是一个重量级依赖：它引入了自己的调度器、对象存储和仪表板，增加了并非每个团队都需要的操作复杂性。这正是为什么像PRIME-RL、PipelineRL和AReaL这样的库选择轻量级原生Python协调（asyncio、线程、Redis流）的原因——当你控制完整堆栈且部署拓扑固定时，原生Python的简单性和可调试性通常超过Ray提供的便利。

代价是对一个非平凡运行时的硬依赖。这种权衡可能是值得的，特别是对于生产规模训练（64+ GPU、多天运行、复杂奖励计算）。

虽然Ray的actor模型是领域的主要参与者，但[Monarch](https://github.com/pytorch/monarch)作为Meta推出的新PyTorch原生分布式actor框架出现，专为GPU工作负载构建。与Ray类似，Monarch基于actor模型；组件是通过消息通信的独立actor，但它是从头设计用于PyTorch/CUDA生态系统，而不是作为通用分布式运行时。

Monarch提供了几个与异步RL特别相关的能力。一个[使用Monarch的异步RL示例实现](https://allenwang28.github.io/monarch-gpu-mode/05_rl_intro.html)（来自GPU Mode讲座系列）演示了架构：生成器、回放缓冲区和训练器被建模为Monarch actor，回放缓冲区吸收来自落后rollout的延迟变化，RDMA权重同步将更新后的参数推送到生成器而不阻塞训练。该模式在结构上与基于Ray的设计（verl、SkyRL、open-instruct）相同，但使用纯PyTorch原生原语实现。

### 轴2：Rollout缓冲区设计

*生成的rollout如何从推理流向训练，以及流水线有多深？*

缓冲区是位于生成和训练之间的数据结构。其深度控制最大异步程度，因此控制最大陈旧度。

| 模式             | 深度  | 库                                                                                                                            | 特征                        |
| -------------- | --- | ---------------------------------------------------------------------------------------------------------------------------- | ------------------------- |
| **无缓冲区**（同步）   | 0   | TRL（当前）、**ART**（收集全部然后训练）                                                                                                    | 生成和训练严格交替；零陈旧度，最大空闲时间     |
| **双缓冲区**（一步提前） | 1   | verifiers-rl、SLIME（异步模式）、MILES、OAT                                                                                           | 在训练步骤N开始时提交生成N+1；恰好重叠一个批次 |
| **有界异步队列**     | 2–K | SkyRL、verl（完全异步）、NeMo-RL、ROLL、PRIME-RL、TorchForge、Tunix、**open-instruct**（`async_steps`）、**AReaL**（`max_head_offpolicyness`） | 多个批次在飞行中；陈旧度受队列容量限制       |
| **无界/流**       | 无限  | PipelineRL（Redis流）、SLIME（完全异步模式）、Atropos                                                                                     | 连续生成；陈旧度仅受显式版本控制限制        |

双缓冲区模式[是从同步训练升级到异步训练的最简单方式：它恰好重叠一个生成与一个训练步骤，并引入最多一步的策略滞后！](https://en.wikipedia.org/wiki/Multiple_buffering)另一方面，更深的队列提高了吞吐量，但需要陈旧度管理。

缓冲区控制有多少数据在飞行中。但数据只是方程的一半。另一半是在这些rollout变得陈旧之前将更新后的权重

回*送到推理服务器。这就是权重同步的用武之地！*&#x8F74;3：权重同步协议

### 轴3：权重同步协议

*范围说明：*

> **本轴专注于**解聚模&#x5F0F;**。**, where inference and training run on separate GPU pools, since that is the deployment topology where async overlap (and therefore weight sync design) actually matters. Colocated setups (same GPUs for both roles) are inherently synchronous and do not face the transport/interrupt trade-offs discussed below.

This is the most architecturally consequential axis. The protocol determines sync latency, interrupt granularity, and whether partial rollouts are possible.

There is a critical distinction to make here: the **transport mechanism** and the **interrupt model**. Most libraries pause generation at a coarse boundary, an HTTP request, a full batch, or even a full training step, before initiating weight transfer. PipelineRL is the outlier: it never stops generating at all.

**Transport mechanism:**

| Mechanism                | Latency     | Libraries                                                                           |
| :----------------------- | :---------- | :---------------------------------------------------------------------------------- |
| **NCCL Broadcast**       | \~100–500ms | PipelineRL, SkyRL, SLIME, MILES, ROLL, OAT, NeMo-RL, PRIME-RL, open-instruct, AReaL |
| **NCCL + Bucketing**     | \~20ms      | verl                                                                                |
| **KV + Shared Memory**   | Low         | TorchForge                                                                          |
| **Filesystem + HTTP**    | Medium      | PRIME-RL, AReaL, ART                                                                |
| **CUDA IPC (Zero-copy)** | Very Low    | NeMo-RL, MILES                                                                      |
| **JAX Cross-mesh**       | Low         | Tunix                                                                               |
| **HTTP PUT**             | High        | verifiers-rl                                                                        |
| **Filesystem + Restart** | Very High   | Atropos                                                                             |

**In the interrupt model, when does the generation pause to accept new weights?**

This is where PipelineRL fundamentally diverges from every other library. Rather than listing each library individually, the landscape collapses into five conceptual tiers, ordered from finest to coarsest interrupt granularity:

| Interrupt Granularity                    | What Happens                                                                                                                                     | Libraries                                                    |
| :--------------------------------------- | :----------------------------------------------------------------------------------------------------------------------------------------------- | :----------------------------------------------------------- |
| **Never** (In-flight per-forward-pass)   | Sequences never stop. The weight swap happens between token decode steps (\~1-10ms gap). Running sequences seamlessly continue with new weights. | PipelineRL, open-instruct (opt-in)                           |
| **Per HTTP Request** (Abort + Resync)    | In-flight HTTP requests are aborted. Partial tokens are resubmitted with a prefix-resume mechanism or recycled for retry.                        | SkyRL, SLIME, MILES                                          |
| **Soft Pause** (Drain in-flight)         | No new generation requests are accepted while in-progress ones finish naturally. Once drained, weights are synced and generation resumes.        | PRIME-RL, AReaL, open-instruct (default), verl (async)       |
| **Per Training Step / Batch** (Blocking) | Generation must fully complete. The trainer and inference engine take turns blocking each other.                                                 | NeMo-RL, ROLL, OAT, TorchForge, Tunix, verifiers-rl, Atropos |

The "never-stop" tier is qualitatively different from all others: PipelineRL, for example, hooks into the inference engine so that the lock is acquired and released *per transformer forward pass* (one token step for one sequence). A weight update waits at most one forward pass (\~few ms), swaps all parameters, and generation resumes immediately. Every other library stops generation at a coarser boundary, from one HTTP request (\~hundreds of ms) up to a full batch boundary (\~seconds).

Weight sync controls *when* new weights arrive. But async training means rollouts are always being generated under *some* policy version, and that *generating* policy might be several gradient steps behind the trainer. How libraries handle this policy lag is staleness management.

### &#x20;Axis 4: Staleness Management

*How does the system handle the fact that generated rollouts may come from an older policy than the one being trained?*

Once generation and training overlap, samples become off-policy. Three **orthogonal** strategies have emerged for managing this staleness, and most production systems combine more than one:

**Strategy 1: Per-sample version rejection.** Every sample is tagged with the integer policy version that generated it. At training time, samples whose version falls behind the current policy by more than a threshold are hard-dropped before entering the loss computation. Simple and correct, but wastes the precious compute spent generating discarded samples.

**Strategy 2, Depth Bounding.** The queue or buffer between generation and training has a bounded capacity (or an explicit staleness gate), which architecturally limits how far behind any sample can be. This ranges from depth=1 (one-step-ahead double buffering, where staleness is impossible by construction) to explicit capacity formulas tied to version gaps. No per-sample version tracking is required; the bound is enforced by the system's pipeline depth.

**Strategy 3, IS-weighted loss correction.** Stale samples that reach the trainer are reweighted by the importance sampling ratio π∗old(a∣s)π∗θ(a∣s), typically clipped (Truncated IS). Some libraries also apply OPSM (zero-out loss for off-policy samples with negative advantage). This preserves throughput; no samples are discarded, but there is a cost in gradient variance from the IS ratios.

These strategies are orthogonal: a system can use version rejection alone, depth bounding alone, IS correction alone, or any combination of them. Synchronous systems avoid the problem entirely by never overlapping generation and training.

| Library           | Version Rejection | Depth Bounding | IS Correction | Key Config / Notes                                                                                                   |
| ----------------- | :---------------: | :------------: | :-----------: | -------------------------------------------------------------------------------------------------------------------- |
| **AReaL**         |         ❌         |        ✅       |       ⚠️      | `max_head_offpolicyness` capacity formula; optional `use_decoupled_loss` adds IS weight capped at 5.0                |
| **ART**           |         —         |        —       |       —       | Synchronous; all rollouts collected before training; no staleness by design                                          |
| **Atropos**       |         ❌         |        ✅       |       ❌       | `max_batches_offpolicy`, ceiling on buffered batches                                                                 |
| **MILES**         |         ❌         |        ❌       |       ✅       | TIS + OPSM                                                                                                           |
| **NeMo-RL**       |         ✅         |        ❌       |       ❌       | `max_trajectory_age_steps`, per-sample version drop                                                                  |
| **OAT**           |         ❌         |        ❌       |       ✅       | Clipped TIS ratio                                                                                                    |
| **open-instruct** |         ❌         |        ✅       |       ⚠️      | `async_steps` cap (default 1, production 8); optional `--truncated_importance_sampling_ratio_cap ρ` adds clipped TIS |
| **PipelineRL**    |         ✅         |        ❌       |       ❌       | `max_lag`, integer version tag per sample; drop if age exceeds threshold                                             |
| **PRIME-RL**      |         ✅         |        ✅       |       ✅       | Full hybrid: `max_async_level` version gap + `max_off_policy_steps` cancellation + IPO trust-region IS               |
| **ROLL**          |         ❌         |        ❌       |       ✅       | Richest IS suite: TIS, TOPR, CISPO, Kimi15, six off-policy loss variants                                             |
| **SkyRL**         |         ❌         |        ✅       |       ❌       | `max_staleness_steps`, capacity gate blocks new rollouts when exceeded                                               |
| **SLIME**         |         ❌         |        ❌       |       ✅       | TIS + OPSM (off-policy masking for partial rollouts)                                                                 |
| **TorchForge**    |         ✅         |        ❌       |       ❌       | `max_policy_age`, per-sample version tag; hard drop                                                                  |
| **Tunix**         |         ❌         |        ✅       |       ❌       | Bounded queue + sync per step; staleness structurally limited                                                        |
| **verl**          |         ❌         |        ❌       |       ✅       | Clipped TIS ratio; optional OPSM                                                                                     |
| **verifiers-rl**  |         ❌         |        ✅       |       ❌       | Depth=1 FIFO + sync every step; staleness impossible by construction                                                 |

> ✅ = yes, ❌ = no, ⚠️ = optional / configurable, — = not applicable (synchronous)

- **Version rejection** is simple and correct, but wastes compute when many samples are discarded.
- **IS correction** preserves throughput at the cost of gradient variance.
- **Depth bounding** is the coarsest mechanism, but it avoids per-sample bookkeeping entirely.

The trend in production systems (PRIME-RL, AReaL, open-instruct) is toward **hybrid approaches** that combine depth bounding with optional IS correction, getting the architectural simplicity of bounded queues with the loss-level safety net of importance weighting for stable training.

Staleness management handles data that was generated under an old policy. But what about data that's *still being generated* when a weight update lands?

### &#x20;Axis 5: Partial Rollout Handling

*What happens to a generation in progress when a weight update arrives?*

This is critical for long-context tasks where a single rollout can take minutes. Four strategies:

| Strategy                                      | Libraries                         | Description                                                                                                                                                                                                                                                                  |
| --------------------------------------------- | --------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Implicit continuation**                     | PipelineRL                        | Sequences are never interrupted. Weights swap between forward passes; the sequence simply continues with new weights. Stored logprobs remain valid because training uses the *recorded* π\_old, not recomputed.                                                              |
| **Abort + retry with prefix**                 | SkyRL, SLIME                      | Active sequences are aborted. Partial tokens are accumulated, then resubmitted with a prefix-resume mechanism using the new weights.                                                                                                                                         |
| **Explicit save/resume**                      | verl (fully async)                | The rollout worker saves partial token IDs and logprobs to a buffer, waits for sync, then resumes from the saved prefix.                                                                                                                                                     |
| **Group cancellation (generation continues)** | PRIME-RL                          | Stale rollout groups have their async tasks cancelled; the inference server continues serving in-flight HTTP requests whose results are discarded. Weight sync triggers between HTTP requests without interrupting mid-request generation.                                   |
| **No partial rollout support**                | verifiers-rl, OAT, Atropos, Tunix | Weight sync only happens at batch boundaries. In-flight generations must complete before sync begins.                                                                                                                                                                        |
| **Soft pause, in-flight sequences complete**  | **AReaL**                         | A pause signal blocks new KV-cache allocations but does not abort in-progress sequences. The task dispatcher stops submitting new tasks; running tasks run to completion. After weight sync, generation dispatch resumes.                                                    |
| **Full sleep, no in-flight at sync time**     | **ART**                           | By design, training only begins after all rollouts are collected. There are never in-progress sequences when sleep is triggered. Level-1 sleep (in-progress requests exist) offloads KV cache to CPU; level-2 sleep discards it entirely.                                    |
| **Drain-or-inflight (configurable)**          | **open-instruct**                 | Default: a stop flag gates new prefetching; weight update waits for active tasks to drain. With in-flight updates enabled, drain is bypassed and weights broadcast while tokens are still being generated; sequences in progress continue with a mix of old and new weights. |

So far, every axis has assumed full-parameter training. But in LoRA training, you're only training a few million adapter parameters instead of billions, the weight sync problem nearly disappears. Let's look at how these libraries support LoRA training.

### &#x20;Axis 6: LoRA Training Support

*Does the library support parameter-efficient training via LoRA adapters, in what modes, and does it exploit adapter-only weight sync?*

LoRA is arguably the most practically consequential axis for teams with limited GPU budgets. It reduces the trainable parameter count by 99%+, halves peak activation memory, and, when the inference server is LoRA-aware, enables *adapter-only weight sync*: instead of broadcasting every parameter of a 7B+ model (\~100–500ms NCCL), only the adapter deltas are pushed to vLLM, which at rank 32 amounts to \~50 MB, a sub-millisecond transfer.

| Library           | LoRA Supported             | Mode Restriction                 | LoRA Backend                                   | Adapter-Only Sync                                                                                |
| ----------------- | -------------------------- | -------------------------------- | ---------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| **AReaL**         | ✅ Yes                      | FSDP2 only (not Megatron/Archon) | HF `peft`                                      | ✅ Yes (disk-based sync; only trainable params transferred; vLLM adapter hot-swap)                |
| **ART**           | ✅ Yes (primary design)     | Both (shared + dedicated GPU)    | Unsloth/`peft` (default); custom Megatron LoRA | ✅ Yes (only adapter saved/loaded; in-process or HTTP adapter hot-swap; base weights never moved) |
| **Atropos**       | ✅ Yes                      | Disaggregated                    | HF `peft`                                      | ✅ Yes (`lora_only` / `lora_restart` modes)                                                       |
| **MILES**         | ✅ Yes                      | Both (colocated + disaggregated) | Megatron-Bridge                                | ✅ Yes (adapter sync config for SGLang)                                                           |
| **NeMo-RL**       | ✅ Partial\*                | Both                             | Custom (not `peft`)                            | ❌ No evidence                                                                                    |
| **OAT**           | ✅ Yes                      | Both                             | HF `peft`                                      | ✅ Yes (LoRA-only sync mode)                                                                      |
| **open-instruct** | ⚠️ Code exists, not wired‡ | —                                | HF `peft` (SFT/DPO only)                       | ❌ No (LoRA not applied in the RL trainer)                                                        |
| **PipelineRL**    | ✅ Yes                      | Non-colocated                    | HF `peft`                                      | ❌ No (full NCCL broadcast)                                                                       |
| **PRIME-RL**      | ✅ Yes                      | Disaggregated                    | Custom MultiLoRA (not `peft`)                  | ✅ Yes (adapter-only state dict extraction)                                                       |
| **ROLL**          | ✅ Partial†                 | DeepSpeed backend only           | HF `peft` / TRL                                | ❌ No evidence                                                                                    |
| **SkyRL**         | ✅ Yes                      | Both                             | `peft` (FSDP) / Megatron-Bridge (Megatron)     | ✅ Yes (filesystem-based adapter sync)                                                            |
| **SLIME**         | ❌ No                       | —                                | —                                              | ❌ No                                                                                             |
| **TorchForge**    | ❌ No                       | —                                | —                                              | ❌ No                                                                                             |
| **Tunix**         | ✅ Yes                      | Both                             | qwix (JAX-native)                              | ✅ Yes (auto-detected)                                                                            |
| **verl**          | ✅ Yes (most complete)      | Both                             | `peft` (FSDP) / Megatron-Bridge (Megatron)     | ✅ Yes (unmerged adapter sync)                                                                    |
| **verifiers-rl**  | ✅ Yes (via prime-rl)       | Disaggregated                    | HF `peft` + FSDP2 + vLLM                       | ✅ Yes (vLLM LoRA serving)                                                                        |

\* NeMo-RL: LoRA for GRPO and DPO is supported only on the DTensor backend; the Megatron Core backend is SFT-only (RL LoRA listed as "coming soon"). Uses a custom DTensor-compatible LoRA module (not `peft`), optionally with Triton kernels.

† ROLL: LoRA is officially supported with the DeepSpeed training backend only. Megatron-backend LoRA appeared in the Feb 2026 changelog but remains experimental.

‡ open-instruct: The model config exposes LoRA-related fields (`use_peft`, `lora_r`, `lora_alpha`), and adapter saving is handled in the checkpoint logic. However, the `peft` model is never initialised in the RL training path; LoRA remains an SFT/DPO-only feature for the RL trainer as of March 2026.

**Three LoRA implementation families:**

1. **HuggingFace `peft`** (PipelineRL, SkyRL/FSDP, verifiers-rl, ROLL, OAT, Atropos): The most common choice. Standard checkpoint format (`adapter_model.safetensors`), compatible with any HF Transformers training loop. ZeRO-3 interactions require care: OAT, for example, needs to disable the fused LM head; ROLL must disable gradient checkpointing entirely.

1. **Megatron-Bridge** (verl/Megatron, SkyRL/Megatron, MILES): Required for 3D-parallel training (TP × PP × DP). Supports multiple LoRA types: `lora`, `canonical_lora`（将合并的QKV拆分为独立的Q/K/V适配器），`vlm_lora`，以及`dora`。该`canonical_lora`变体避免了QKV合并，从而提高了训练稳定性。MILES以HF`peft`格式和Megatron原生每秩格式保存检查点。

1. **自定义实现**（NeMo-RL、PRIME-RL、Tunix/qwix）：特定于库的LoRA模块，无法与`peft`检查点互操作。PRIME-RL独特地支持在单次运行中同时使用多个适配器，以实现多实验并行。Tunix使用Google的`qwix`JAX库，该库内置了QLoRA（NF4量化）和TPU原生梯度路由。NeMo-RL使用自定义的DTensor兼容模块，并可选Triton融合内核。

**仅适配器权重同步机会（与轴3的交互）：**

十三个库中有八个支持仅将**LoRA适配器增量**推送到推理服务器。这完全改变了权重同步问题（轴3）的性质。在使用全参数训练时，中断模型（每次前向传播锁定 vs. 每次请求中止 vs. 每批次暂停）决定了在NCCL广播期间浪费了多少生成。当使用LoRA并仅同步适配器时，传输量非常小，以至于几乎任何中断模型都能提供相当的吞吐量！即使是Atropos的暴力HTTP热交换也变得可行。

***

### 轴7：分布式训练后端与并行性

*库使用何种并行策略进行训练，这如何约束或启用异步架构？*

此轴贯穿所有其他轴。训练后端的选择决定了每个GPU能容纳多大的模型、在广播到推理服务器之前需要多少集合操作来收集权重，以及哪些模型架构能够被训练。对于团队扩展超过300亿参数或从密集模型转向专家混合（Mixture-of-Experts）模型来说，这是最具决定性的决策。

| 库                 | 训练后端                     | 并行性               | HF模型加载         | MoE / EP支持 |
| :---------------- | :----------------------- | :---------------- | :------------- | :--------- |
| **AReaL**         | FSDP2、Megatron、Archon    | DP、SP、TP、PP、CP、EP | ✅ 直接 / 转换      | ✅          |
| **ART**           | Unsloth、Megatron         | DP、TP、EP          | ✅ 直接 / 转换      | ✅          |
| **Atropos**       | PyTorch原生、TRL            | DP                | ✅ 直接           | ❌          |
| **MILES**         | Megatron、FSDP2           | DP、TP、PP          | 🔄 转换          | ✅          |
| **NeMo-RL**       | FSDP2、Megatron           | DP、SP、TP、PP、CP、EP | ✅ 直接 / 转换      | ✅          |
| **OAT**           | DeepSpeed                | DP、TP             | ✅ 直接           | ❌          |
| **open-instruct** | DeepSpeed                | DP、SP             | ✅ 直接           | ❌          |
| **PipelineRL**    | DeepSpeed                | DP、SP             | ✅ 直接           | ❌          |
| **PRIME-RL**      | FSDP2                    | DP、TP、CP、EP       | ✅ 直接           | ✅          |
| **ROLL**          | DeepSpeed、Megatron、FSDP2 | DP、SP、TP、PP、CP、EP | ✅ 直接 / 转换      | ✅          |
| **SkyRL**         | FSDP、Megatron            | DP、SP、TP、PP、EP    | ✅ 直接 / 转换      | ✅          |
| **SLIME**         | Megatron                 | DP、TP、PP、SP       | 🔄 转换          | ✅          |
| **TorchForge**    | FSDP2                    | DP、TP、CP          | ✅ 通过TorchTitan | ❌          |
| **Tunix**         | JAX/XLA                  | DP、TP             | ❌ 自定义Flax      | ❌          |
| **verl**          | FSDP、Megatron            | DP、SP、TP、PP、CP、EP | ✅ 直接 / 转换      | ✅          |
| **verifiers-rl**  | DeepSpeed                | DP                | ✅ 直接           | ❌          |

训练后端对异步RL库设计产生直接影响：

**权重同步速度直接取决于训练后端，更快的同步意味着更少的陈旧性。**

在解耦的异步设置中，权重同步*不一定*会阻塞推理。关键设计决策是**权重更新如何与进行中的生成交互**；存在四种策略，按破坏性从低到高排序：

- **原子交换，无中断。**&#x5B8C;整的权重更新作为单个阻塞RPC发送到推理引擎。每次前向传播要么看到全部旧权重，要么看到全部新权重，绝不会混合。生成最多暂停一次前向传播的间隔（约几毫秒）。（PipelineRL）
- **每参数流式传输，无中断。**&#x6BCF;个参数作为单独的RPC + NCCL广播发送。前向传播在单个参数更新之间交错，因此进行中的序列确实会在不同层看到新旧权重的混合。最大重叠，但一致性最弱。（open-instruct，飞行模式）
- **调度门，排空进行中任务，然后同步。**&#x65B0;请求被暂缓，直到进行中的序列自然完成；权重仅在流水线排空后广播。无浪费令牌，但同步气泡与最长进行中序列成正比。（PRIME-RL、AReaL、open-instruct默认、verl完全异步）
- **硬暂停或中止。**&#x63A8;理被暂停，或在权重传输开始前中止进行中的请求。最清晰的一致性，最高的计算浪费。（verl、SkyRL）

但即使在推理继续的库中，**较慢的同步意味着推理在陈旧权重上运行的时间更长**。训练器与推理池之间的策略版本差距随同步持续时间增长。这是需要考虑的因素。

\*\*随着领域向稀疏模型发展，MoE支持日益成为重要的差异化因素。\*\*\
趋势很明显：前沿模型是稀疏的（DeepSeek-V3、Qwen3-MoE、Mixtral、DBRX），开源权重的MoE正成为后训练的默认起点。训练这些模型需要专家并行（EP），将不同专家分配到不同秩，而大多数异步RL库不支持。只有基于Megatron的库（verl、SLIME、MILES、ROLL、NeMo-RL）和PRIME-RL的FSDP2+EP路径能正确处理EP。基于ZeRO的库（PipelineRL、verifiers-rl、OAT、open-instruct）可以*加载*MoE HuggingFace模型类，但如果没有EP，每个专家会跨所有ZeRO-3秩分片，而不是放置在专用秩上；每次前向传播都会AllGather每个专家，完全抵消了稀疏性优势。EP也使权重同步复杂化：在广播到vLLM/SGLang（通常从单个TP组服务所有专家）之前，训练器必须从每个EP秩AllGather专家参数，这是一个O(N∗experts×E∗size)的通信（其中E\_size是每个专家的参数数量），这在密集模型中不存在。对于一个拥有256个专家的235B MoE，这是一个显著的同步成本。希望在后训练下一代开源MoE模型时保持相关性的库需要EP感知的训练\*和\*EP感知的权重同步。

\*\*MoE LoRA是一个新兴需求，也是一个棘手的问题。\*\*\
密集模型上的LoRA是众所周知的（轴6）：将适配器附加到注意力投影，训练它们，仅同步适配器增量。MoE LoRA更难，因为自然目标是*expert FFN layers*, meaning each expert gets its own adapter. For a model with 64 experts and rank-32 LoRA on each expert's gate/up/down projections, the adapter count jumps from \~20 (dense) to \~200+ (MoE), and the adapters are distributed across EP ranks. Weight sync must gather adapters from every EP rank before pushing them to the inference server, a coordination problem that does not exist for dense LoRA. Among the surveyed libraries, only **ART** explicitly implements MoE expert LoRA layers (Megatron EP path with per-expert LoRA and manual allreduce), and **MILES** supports LoRA via Megatron-Bridge, which can target expert layers. verl's Megatron-Bridge path supports LoRA types including `vlm_lora`, but MoE-specific expert LoRA is not documented. vLLM's LoRA serving does not natively support per-expert adapters; it loads a single adapter applied uniformly, so adapter-only sync for MoE LoRA currently requires custom inference-side logic. As MoE models become the default for post-training, MoE LoRA with efficient adapter-only sync will be a key capability gap to close.

That covers the seven axes, each captures a different facet of the same underlying problem. Together, they give us a complete lens for comparing libraries. Time to put it all on one page.

***

## &#x20;4. Global Overview: Sixteen Libraries at a Glance

> **Note:** This overview reflects the state of these libraries as of March 2026. The ecosystem is evolving rapidly; specific features, backends, and integrations may change in the near future.

| Library           | Org            | Orchestration Type                                       | Inference Server       | Weight Sync                                 | Staleness Management     | Partial Rollout                     | Training Backend                      | Dist. Parallelism                                                        | LoRA Support                              |
| ----------------- | -------------- | -------------------------------------------------------- | ---------------------- | ------------------------------------------- | ------------------------ | ----------------------------------- | ------------------------------------- | ------------------------------------------------------------------------ | ----------------------------------------- |
| **AReaL**         | inclusionAI    | Native Python (asyncio + HTTP RPC); pluggable Ray/Slurm  | vLLM, SGLang           | NCCL chunked OR filesystem safetensors      | Depth + IS (optional)    | 🟧 Soft pause (in-flight complete)  | FSDP2 or Megatron-LM or Archon        | FSDP2: DP+SP+TP; Megatron: TP+SP+PP+CP+EP; Archon: FSDP2+TP+SP+PP+EP     | ✅ `peft` (Adapter-only)                   |
| **ART**           | OpenPipe       | Native Python (asyncio + mp child processes)             | vLLM                   | LoRA adapter swap (no full weight transfer) | Synchronous (none)       | ❌ No                                | Unsloth (single-GPU); Megatron-LM     | None (Unsloth); TP×EP×DP (Megatron)                                      | ✅ `peft` / Megatron LoRA (Adapter-only)   |
| **Atropos**       | NousResearch   | HTTP Microservices (FastAPI)                             | vLLM, SGLang, OpenAI   | FS checkpoint + vLLM restart                | Depth bounding           | ❌ No                                | Single-GPU PyTorch; TRL/Accelerate    | None (native); FSDP/ZeRO via TRL adapter                                 | ✅ `peft` (Adapter-only)                   |
| **MILES**         | radixark       | Distributed Actor (Ray)                                  | SGLang                 | NCCL OR CUDA IPC                            | IS correction            | 🟧 Abort + recycle to buffer        | Megatron-LM (primary); FSDP2          | Megatron: TP×PP×DP; FSDP2 available; colocated CUDA IPC                  | ✅ Megatron-Bridge (Adapter-only)          |
| **NeMo-RL**       | NVIDIA         | Distributed Actor (Ray)                                  | vLLM, SGLang, Megatron | NCCL OR CUDA IPC-ZMQ OR HTTP                | Version rejection        | ✅ In-flight continuation            | DTensor (FSDP2+TP) or Megatron-Bridge | DTensor: TP+SP+CP+FSDP2; Megatron: TP×PP×CP×EP×ETP + FSDP2               | 🟧 Custom (No adapter-only sync)          |
| **OAT**           | SAIL-SG        | Distributed Actor (Ray)                                  | vLLM                   | NCCL per-param + ZeRO-3 gather              | IS correction            | ❌ No                                | DeepSpeed ZeRO-2/3                    | ZeRO-2 / ZeRO-3 DP; AutoTP                                               | ✅ `peft` (Adapter-only)                   |
| **open-instruct** | AI2 (AllenAI)  | Distributed Actor (Ray)                                  | vLLM                   | NCCL broadcast; optional in-flight updates  | Depth + IS (optional)    | 🟧 Drain-or-inflight (configurable) | DeepSpeed ZeRO-0/2/3                  | ZeRO-3 DP + Ulysses SP; vLLM TP (inference only)                         | ❌ No                                      |
| **PipelineRL**    | ServiceNow     | Native Python + Pub/Sub (asyncio + Redis/JSONL)          | vLLM                   | NCCL pg + HTTP notify                       | Version rejection        | ✅ Implicit continuation             | DeepSpeed ZeRO-3                      | ZeRO-3 DP + Ring SP; ZeRO++ available                                    | ✅ `peft` (Full sync)                      |
| **PRIME-RL**      | PrimeIntellect | Native Python (asyncio + FS/ZMQ)                         | vLLM                   | Filesystem safetensors + HTTP OR NCCL       | Version + depth + IS     | 🟧 Group cancellation               | FSDP2 (exclusively)                   | FSDP2 per-block + TP + CP + EP; pp=1                                     | ✅ Custom MultiLoRA (Adapter-only)         |
| **ROLL**          | Alibaba        | Distributed Actor (Ray)                                  | vLLM, SGLang           | NCCL via dedicated update group             | IS correction            | ❌ No                                | DeepSpeed ZeRO or Megatron or FSDP2   | DS: ZeRO+Ulysses SP; Megatron: TP×PP×CP×EP; FSDP2: HSDP+Ulysses          | 🟧 `peft` (DeepSpeed only)                |
| **SkyRL**         | NovaSky-AI     | Distributed Actor (Ray) + Native Python                  | vLLM, SGLang           | NCCL pg                                     | Depth bounding           | 🟧 Abort + retry with prefix        | FSDP/FSDP2 or Megatron-Bridge         | FSDP: ZeRO shard + Ulysses SP; Megatron: full 5D via bridge; JAX backend | ✅ `peft` / Megatron-Bridge (Adapter-only) |
| **SLIME**         | THUDM          | Distributed Actor (Ray)                                  | SGLang                 | NCCL pg, bucketed                           | IS correction            | 🟧 Abort + recycle to buffer        | Megatron-LM                           | TP×PP×DP; Megatron→HF conversion; MoE EP all-gather                      | ❌ No                                      |
| **TorchForge**    | Meta           | Distributed Actor (Monarch)                              | vLLM                   | torchstore + shared memory prefetch         | Version rejection        | ❌ No                                | FSDP2 via TorchTitan                  | FSDP2 + TP; CP partial; PP not yet implemented                           | ❌ No                                      |
| **Tunix**         | Google         | Native Python (ThreadPoolExecutor + asyncio); JAX-native | vLLM, SGLang, JAX      | Cross-mesh reshard                          | Depth bounding           | ❌ No                                | JAX/XLA 2D mesh                       | 2D JAX mesh: FSDP + TP; no PP; TPU-primary                               | ✅ qwix / QLoRA (Adapter-only)             |
| **verl**          | ByteDance      | Distributed Actor (Ray)                                  | vLLM, SGLang           | NCCL + checkpoint-engine buckets            | IS correction            | ✅ Explicit save/resume              | FSDP1/FSDP2 or Megatron-Core          | FSDP: ZeRO-2/3/HSDP + Ulysses SP; Megatron: TP×PP×VPP×CP×EP×ETP          | ✅ `peft` / Megatron-Bridge (Adapter-only) |
| **verifiers-rl**  | PrimeIntellect | Native Python (threading + asyncio)                      | vLLM                   | PyNCCL broadcast                            | Depth bounding (depth=1) | ❌ No                                | DeepSpeed ZeRO-3 (Accelerate)         | ZeRO-3 DP only; no TP/PP                                                 | ✅ `peft` (Adapter-only)                   |

That's the current state of play. But the field is moving fast, and several emerging trends are about to stress-test these architectures in ways their designers may not have anticipated.

***

## &#x20;5. The Next Wave: Design Implications

The trends below are not a catalogue of new techniques; each one creates concrete pressure on the infrastructure and algorithmic choices made today. The question is not "what is the frontier?" but "if this trend wins, what breaks in my current stack?"

### &#x20;5.1 Critic-Free Algorithms: Memory Freed, But Weight Sync Pressure Increases

PPO's value network doubles the memory footprint of any training node. The field is converging on critic-free variants (GRPO, REINFORCE++, Online DPO) precisely because long CoT reasoning makes this overhead prohibitive at 8K–64K context lengths.

**这解锁了什么：**&#x6D88;除评论家（critic）可释放约50%的训练GPU内存。这些空闲资源可重新分配给：（a）更大的rollout批次，直接减少掉队者（straggler）方差问题，或（b）在同一GPU上共同部署推理和训练，从而完全消除对单独的NCCL权重同步进程组的需求。

**它没有解决什么：**&#x65E0;评论家（critic-free）方法仍然需要频繁地将权重推送到推理服务器。实际上，它们可能*增加*同步压力：没有价值网络（value network）提供稳定的基线，GRPO风格算法需要更大的组大小（G=8–32）以获得低方差优势估计，这意味着每个步骤有更多rollout和更快的策略漂移（policy drift）。仅在粗粒度边界（每个训练步骤或每K步）同步的库将在无评论家训练下看到陈旧性（staleness）增长更快。

**非对称轨迹过滤**（GRPO-RoC：过采样rollout，严格过滤正例，均匀下采样负例；DeepSeek-V3.2和MiniMax-M1中的CISPO/DAPO风格非对称裁剪）对陈旧性有更微妙的影响。问题不在于批次本身缩小；而是*组成*幸存批次的组成。正例轨迹（对简单提示的正确解决方案）收敛更快并被优先保留；较难的提示产生大多被丢弃的负例轨迹。结果：幸存过滤的样本系统地*更旧*于缓冲区中的平均rollout，因为它们解决的简单提示在训练早期发出。一个名义上“新鲜”的rollout缓冲区可能包含跨越广泛策略版本的幸存正例。在批次级别跟踪陈旧性的准入控制（例如，SkyRL的`max_staleness_steps`容量门（capacity gate），Atropos的`max_batches_offpolicy`）无法检测这种批次内版本扩散。每样本版本标记（Axis 4）在这种机制中不是可选的；训练器必须能够拒绝或IS校正（IS-correct）那些策略版本偏离太远的单个样本，即使它们所属的批次最近被准入。

无评论家方法简化了训练侧。但*评分*侧即将变得更加昂贵：过程奖励模型（process reward models）对中间推理步骤评分，而不仅仅是最终答案，这引入了全新的同步瓶颈。

### 5.2 过程奖励：一个新的同步屏障

结果奖励（outcome reward）是标量且廉价，在rollout结束时调用一次验证器。过程奖励模型（PRMs）对中间步骤评分，这需要（a）对完整推理轨迹进行单独的PRM前向传递，或（b）在生成期间逐令牌计算的在线效用函数（online utility function）。

**PRPO**（熵峰值分割（entropy-spike segmentation）与每段PRM评分）和**DEEP-GRPO**（通过在线效用函数识别枢纽（pivot identification））都会产生计算开销*在生成和训练之间*。在当前库生态系统中，这个阶段尴尬地映射到预处理器池（PipelineRL）或需要额外的Ray actor（verl, NeMo-RL）。两者都不是为此设计的。

**关键含义：**&#x57FA;于PRM的信用分配打破了奖励计算廉价的假设。对来自7B模型的32K令牌推理轨迹进行PRM前向传递可能非常昂贵。在G=8个完成（completion）每提示下，奖励计算可能消耗相对于生成本身不可忽略的挂钟时间。两个后果：

1. **异步奖励管道变得必要。**&#x50;RIME-RL将奖励评分作为其完全异步编排器-训练器（Orchestrator-Trainer）管道的一部分与训练并发运行；编排器处理评分，而训练器独立执行反向传播和优化器步骤。对于基于PRM的方法，这种流水线奖励计算不是可选的；同步奖励评分将主导训练挂钟时间。
1. **单独的预处理器池变得必要**。在专用GPU层上运行参考对数概率（reference logprobs）计算和PRM评分，例如，在生成和训练之间流水线化，是密集信用分配的正确架构。

**DEEP-GRPO的枢纽重采样**引入了第三代模式，与标准rollout和部分rollout恢复并列：*从序列中间状态进行局部重采样*。这需要在枢纽点保存KV缓存状态，而**当前没有异步库开箱即用地支持**。枢纽边界处的权重同步可能是一个新的正确性要求：如果权重在枢纽生成和局部重采样之间改变，优势估计会被破坏。当然，我们可以在单个预填充（prefill）中重新计算KV缓存，但这可能浪费我们训练中宝贵的计算资源。

### 5.3 多智能体协同进化：掉队者问题加剧

单智能体GRPO训练一个策略，每个提示生成G个完成。新兴的多智能体自博弈（multi-agent self-play）意味着有效的“组”跨越顺序链接的多个模型调用。奖励仅在链中所有模型完成后可用。

**掉队者（straggler）动态发生质变。**&#x5728;单智能体GRPO中，掉队者是组中最长的完成，是单峰长度分布中的尾部事件。在多智能体管道中，掉队者是*乘积*两个或更多长度分布。在提议者/求解器（Proposer/Solver）多智能体架构中，如果每个都有第90百分位完成时间（5倍中位数），联合第90百分位大约是25倍中位数。

**智能体群（swarms of agents）上的RL意味着新的工作单元。**&#x4ECA;天，每个库中的原子单元是单个（提示，完成，奖励）三元组。在多智能体训练中，原子单元变成*情节*，一个回合、工具调用和智能体间消息的有向图。缓冲区设计、陈旧性跟踪和优势计算都需要在情节上操作。重放或分叉情节也可能变得必要。

当模型至少内部一致时，跨智能体的掉队者问题已经够糟糕了。在MoE架构中，即使是单个模型也可能在推理和训练框架之间自相矛盾，这引发了强化学习训练中一系列新的涌现问题。

### 5.4 训练-推理不匹配：Deepseek v3.2 MoE案例研究

训练-推理不匹配问题在异步强化学习中普遍存在；每当在策略π∗old下生成展开数据并在π∗θ下计算梯度更新时，这两个策略就会产生分歧。大多数库通过重要性采样校正或硬版本拒绝来解决此问题。但DeepSeek-V3.2的生产经验揭示了两个**结构性**的不匹配来源，这是重要性采样校正无法修复的。

**来源1：MoE专家路由不一致。**&#x6DF7;合专家模型为每个令牌激活稀疏的专家子集。推理框架（vLLM、SGLang）和训练框架（Megatron、FSDP）独立实现路由器，门控函数中浮点数舍入的差异可能导致*对相同输入选择不同的专家*。当专家路由出现分歧时，活动参数子空间会发生不连续偏移；假设专家A处于活动状态计算的梯度步长被应用于在专家B下处于活动状态的权重。DeepSeek-V3.2发现这“导致活动参数子空间的突然偏移，从而破坏优化稳定性并加剧离策略问题。”

他们的解决方案，**保持路由**，保留了采样（推理）期间使用的确切专家路由路径，并在训练前向传播中强制执行这些路径。这要求推理框架记录并返回路由决策以及令牌对数概率，训练框架接受并强制执行它们。目前没有开源异步强化学习库实现此功能。对于任何训练MoE模型的团队（DeepSeek-V3类、Mixtral、未来的开源MoE），这是一个正确性问题，而非性能问题。

**来源2：采样截断掩码不匹配。**&#x54;op-p和top-k采样在生成时截断词汇表，将低概率令牌排除在采样分布之外。在训练期间，完整词汇表对π∗θ可见。这违反了重要性采样恒等式：π∗old（截断）和π∗θ（完整）的动作空间不同，因此对于采样期间被掩码的令牌，重要性采样比π∗θ(o∗t)/π∗old(ot)是未定义的。

DeepSeek-V3.2的**保持采样掩码**解决方案：在采样期间记录截断掩码，并在训练前向传播中将其应用于π\_θ，使两个策略在相同的词汇表子集上操作。这需要将掩码从推理服务器传递回训练器，这同样是当前库基础设施不支持的功能。

**对库设计的影响：**&#x4FDD;持路由和保持采样掩码都要求推理服务器返回*额外元数据*，包括令牌对数概率、路由决策和采样掩码。当前推理服务器（vLLM、SGLang）与训练器之间的API合约是`(token_ids, logprobs, finish_reason)`。将其扩展为`(token_ids, logprobs, finish_reason, expert_routing, sampling_mask)`是对每个库数据流的破坏性更改。

### 5.5 蒸馏：同一异步问题的不同名称

在策略蒸馏中，学生模型生成序列，教师模型用令牌级对数概率为其评分，这在结构上与GRPO中的异步协调问题相同。

本调查中的每个设计轴，包括展开缓冲区、权重同步协议、陈旧性管理和部分展开处理，都同样适用于蒸馏。生成池产生学生展开，教师为其评分（替换验证器），训练器使用优势修正的GRPO损失或独立的KL目标计算反向传播。自蒸馏增加了一个额外的协调要求：教师是学生从步骤*N−k*的冻结快照，因此系统必须定期检查点策略并在不中断流水线的情况下热交换教师服务器，这是一个目前没有库完全自动化的原语。

**对库设计的实际影响是，异步强化学习基础设施不应构建为GRPO特定系统**。生成-评分-训练流水线是一个通用模式，涵盖具有结果奖励的强化学习、具有过程奖励的强化学习、在策略蒸馏和自蒸馏。像**SLIME、MILES、PRIME-RL、AReaL和NeMo-RL**这样的库已经支持GRPO和在策略蒸馏，正是因为它们的异步脚手架将奖励/评分阶段视为可插拔组件而非硬编码的验证器调用。任何追求通用性的异步训练器都应遵循相同原则：将评分阶段定义为接口（HTTP端点、Ray actor或共置前向传播），并让缓冲区、陈旧性和权重同步机制无论填充内容如何都相同地运行。

***

## 6. TRL异步训练器的设计选择

在全面审视了编排模型、缓冲区设计、权重同步协议、陈旧性策略和部分展开处理后，我们现在可以为TRL中的异步训练器制定具体的设计选择，以及我们打算探索的未来发展方向。

### 设计原则：保持编排轻量级

当前TRL实现的一个优势是它不依赖重型编排器系统来管理训练生命周期。库内的数据保持为原生Python对象，没有外部库的着色。我们希望保持这一点：编排应尽可能简单，不依赖重型外部框架。

### 1. 带每令牌的有界队列（无双缓冲）

与其从双缓冲开始再升级到更细粒度的方案，我们直接采用**一个有界队列，其中每个令牌都带有产生它的`model_version`的标签**。这是从一开始就具备的最低可能粒度；它支持令牌级别的重要性采样校正，支持简单的准入门控（丢弃或降低超过陈旧度阈值的令牌的权重），并避免了后期将令牌级来源追溯适配到批次级缓冲区所带来的架构债务。

### 2. 使用打包传输的NCCL权重同步

NCCL进程组是必需的，并且我们已经在使用它们。添加分桶应该是下一步，因为vLLM的[`NCCLWeightTransferEngine`](https://github.com/vllm-project/vllm/blob/f3c6c9c9d794fac5e74b59bc75da6e9d1921eeac/vllm/distributed/weight_transfer/nccl_engine.py)与`packed=True`直接支持分桶广播：它将参数打包到可配置大小的`uint8`缓冲区（默认为1 GB，在CUDA流之间双缓冲），并通过一个独立于训练进程组的专用NCCL通信器进行广播。这消除了在朴素广播中占主导地位的每个参数调用开销，从而实现了巨大的同步加速。

除了vLLM内置的引擎，我们将探索用于更苛刻场景的高性能权重打包库：

- **[Awex](https://github.com/inclusionAI/asystem-awex)**（inclusionAI），一个专为RL训练设计的权重同步框架，处理跨引擎传输的难题：训练引擎（Megatron、DeepSpeed）和推理引擎（SGLang、vLLM）使用完全不同的并行策略和张量布局。Awex通过统一的转换层和确定性的P2P传输计划来抽象这一点。它支持分离GPU和共置（CUDA IPC）两种模式。

- **[Mooncake传输引擎](https://github.com/kvcache-ai/Mooncake)**，SGLang已朝着集成Mooncake传输引擎作为其高性能传输层的方向发展，集成范围涵盖PD解聚、分层KV缓存和弹性专家并行。具体针对权重同步，配套&#x7684;**[checkpoint-engine](https://github.com/MoonshotAI/checkpoint-engine)**&#x9879;目使用Mooncake的RDMA支持的P2P传输来更新万亿参数模型（Kimi-K2，256×H20 GPU），耗时约16-17秒。Mooncake现在是PyTorch生态系统的一部分，并作为[NVIDIA的NIXL传输库](https://github.com/ai-dynamo/nixl)的后端插件。

### 3. 对智能体工作负载的部分推出支持

复杂环境中的多轮工具使用任务每次推出可能需要几分钟。如果没有在权重更新期间处理进行中推出的机制，同步窗口就会成为流水线瓶颈。我们可能会实验性地探索两种策略：

- **前缀恢复**：当权重在推出中途更新时，保存KV缓存前缀，并在新策略下从检查点恢复生成。这保留了部分工作，但需要推理引擎支持中途权重交换。
- **中止并重试**：丢弃超过陈旧度阈值的进行中推出，并重新排队提示。实现更简单，但浪费的计算量与中止时平均推出长度成正比。

这就是路线图，请保持关注，我们正在TRL中开发一个具体的异步GRPO训练器，并将很快宣布 🧑🍳！

<!-- HTML_TAG_END -->

更多来自我们博客的文章

[![](/images/others/keep-the-tokens-flowing-lessons-from-16-open-source-rl-libraries/016-2ae73a5a.png)](/blog/ggml-joins-hf)

[communityopen-sourcellm](/blog/ggml-joins-hf)

[ 热门](/blog/ggml-joins-hf)

## GGML和llama.cpp加入HF以确保本地AI的长期进展

[- ![](/images/others/keep-the-tokens-flowing-lessons-from-16-open-source-rl-libraries/017-3cc80fc8.webp)- ![](/images/others/keep-the-tokens-flowing-lessons-from-16-open-source-rl-libraries/018-2aedb120.webp)- ![](/images/others/keep-the-tokens-flowing-lessons-from-16-open-source-rl-libraries/019-169b2bc1.webp)- ![](/images/others/keep-the-tokens-flowing-lessons-from-16-open-source-rl-libraries/020-fb71dc93.webp)- +2](/blog/ggml-joins-hf)

[ggerganov, et. al.](/blog/ggml-joins-hf)

[ 485](/blog/ggml-joins-hf)

[2026年2月20日 ](/blog/ggml-joins-hf)

[ggerganov, ngxson, et. al.](/blog/ggml-joins-hf)

[![](/images/others/keep-the-tokens-flowing-lessons-from-16-open-source-rl-libraries/021-1abdad50.png)](/blog/unsloth-jobs)

[llmfine-tuningtraining](/blog/unsloth-jobs)

## 使用Unsloth和Hugging Face Jobs免费训练AI模型

[- ![](/images/others/keep-the-tokens-flowing-lessons-from-16-open-source-rl-libraries/022-a81239c4.webp)- ![](/images/others/keep-the-tokens-flowing-lessons-from-16-open-source-rl-libraries/023-78ee9dcf.webp)- ![](/images/others/keep-the-tokens-flowing-lessons-from-16-open-source-rl-libraries/024-41195ad5.webp)- ![](/images/others/keep-the-tokens-flowing-lessons-from-16-open-source-rl-libraries/025-00f367f1.webp)- +2](/blog/unsloth-jobs)

[burtenshaw, et. al.](/blog/unsloth-jobs)

[ 85](/blog/unsloth-jobs)

[2026年2月20日 ](/blog/unsloth-jobs)

[burtenshaw, danielhanchen, et. al.](/blog/unsloth-jobs)

### 社区

![](/images/others/keep-the-tokens-flowing-lessons-from-16-open-source-rl-libraries/026-498fe4f4.webp)

[AINovice2005](/AINovice2005)

[4天前](#69b18d5bd1d2c8b5cd6caafb)

这是一篇非常敏锐的帖子，展示了开源生态系统中RL的现状。向所有研究和呈现这项工作的作者致敬。🙌

查看翻译

🤗

1

1

\+

回复

编辑预览

通过拖入文本输入框、粘贴或点击此处上传图像、音频和视频。

点击或粘贴此处上传图像

评论

· [注册](/join?next=%2Fblog%2Fasync-rl-training-landscape) 或 [登录](/login?next=%2Fblog%2Fasync-rl-training-landscape) 以评论

[ 点赞](/login?next=%2Fblog%2Fasync-rl-training-landscape)

[49](/login?next=%2Fblog%2Fasync-rl-training-landscape)

- [![](/images/others/keep-the-tokens-flowing-lessons-from-16-open-source-rl-libraries/001-93bde3e0.webp)](/lvwerra "lvwerra")
- [![](/images/others/keep-the-tokens-flowing-lessons-from-16-open-source-rl-libraries/002-9bc33468.webp)](/clem "clem")
- [![](/images/others/keep-the-tokens-flowing-lessons-from-16-open-source-rl-libraries/003-3d36de8a.svg)](/sabman "sabman")
- [![](/images/others/keep-the-tokens-flowing-lessons-from-16-open-source-rl-libraries/004-ac0e506b.webp)](/yjernite "yjernite")
- [![](/images/others/keep-the-tokens-flowing-lessons-from-16-open-source-rl-libraries/005-0936a580.webp)](/lewtun "lewtun")
- [![](/images/others/keep-the-tokens-flowing-lessons-from-16-open-source-rl-libraries/006-5cbe078f.webp)](/FL33TW00D "FL33TW00D")
- [![](/images/others/keep-the-tokens-flowing-lessons-from-16-open-source-rl-libraries/027-c172dcc2.webp)](/muhtasham "muhtasham")
- [![](/images/others/keep-the-tokens-flowing-lessons-from-16-open-source-rl-libraries/028-ed23d042.webp)](/davanstrien "davanstrien")
- [![](/images/others/keep-the-tokens-flowing-lessons-from-16-open-source-rl-libraries/029-5b36678a.webp)](/pcuenq "pcuenq")
- [![](/images/others/keep-the-tokens-flowing-lessons-from-16-open-source-rl-libraries/030-287c63ff.webp)](/ariG23498 "ariG23498")
- [![](/images/others/keep-the-tokens-flowing-lessons-from-16-open-source-rl-libraries/013-dc8511e6.webp)](/sergiopaniego "sergiopaniego")
- [![](/images/others/keep-the-tokens-flowing-lessons-from-16-open-source-rl-libraries/031-dfa49dee.webp)](/loleg "loleg")
- +37
