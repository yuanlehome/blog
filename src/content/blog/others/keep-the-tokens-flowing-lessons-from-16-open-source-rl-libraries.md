---
title: '保持令牌流动：来自 16 个开源强化学习（RL）库的经验教训'
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
  /images/others/keep-the-tokens-flowing-lessons-from-16-open-source-rl-libraries/014-d966e2aa.png
lang: zh
translatedFrom: en
---

# 保持令牌流动：来自 16 个开源强化学习（RL）库的经验教训

> **TL;DR**——对于那些没时间阅读 5000 字关于异步强化学习（RL）基础设施文章的人（我们理解，您有模型要训练）：
>
> - **问题：**在同步强化学习（RL）训练中，数据生成（模型推理以创建数据样本）主导了实际时间——在 320 亿（32B）参数模型上，单个包含 32K 令牌的批次可能需要*数小时*，而用于训练的 GPU 在此期间保持空闲。
> - **大家共同采用的解决方案：**将推理和训练解耦（分离）到不同的 GPU 池，通过 rollout 缓冲区（模型输出的临时存储）连接它们，并异步（无需等待）传输权重，这样双方都不必等待对方。
> - **我们调查了 16 个开源库**，这些库实现了这种模式，并在 7 个维度上进行了比较：编排原语、缓冲区设计、权重同步协议、陈旧性管理、部分 rollout 处理、LoRA 支持以及分布式训练后端。
> - **关键发现：**Ray 主导编排（在调查的 16 个分布式计算库中占 8 个）。NCCL（NVIDIA Collective Communications Library）广播是传输模型权重的默认方法。陈旧性管理指如何处理过时的数据样本，范围从简单丢弃旧样本到使用高级重要性采样校正。LoRA（Low-Rank Adaptation）训练支持较少。分布式 MoE（Mixture of Experts）支持是新兴的差异化因素。
>
> 如果您想直接跳到精华部分，[这里是完整的比较表](#4-全局概览十六个库一览)（无需阅读，我们不会评判）。
>
> 但说真的，如果您继续阅读，可能会学到一两件事，了解为什么您的 GPU 有 60%的时间处于空闲状态。

---

## 1. 动机：从同步强化学习（RL）训练到异步架构

异步强化学习（RL）训练已成为大规模后训练的主导范式。现代后训练中的几个趋势使得同步训练循环几乎无法扩展：

- **推理模型的长 rollout。**思维链训练产生非常长的 rollout，单个同步生成批次在单个 GPU 上可能需要数小时才能完成。在此期间，训练 GPU 完全空闲。
- **无价值函数训练器，如 GRPO** 使用组相对优势。这意味着每个提示生成多达 G 倍的 rollout，并且整个批次受组中最慢完成项的制约。
- **智能体强化学习（RL）训练的兴起。**当模型在多轮轨迹中与工具、沙箱和外部环境交互时，rollout 长度和延迟变得高度可变。一个简单的 API 调用可能在几秒内返回，而一个包含工具使用的复杂推理链可能运行数分钟或数小时。MiniMax 的[Forge](https://www.minimax.io/news/forge-scalable-agent-rl-framework-and-algorithm)框架，用于训练 MiniMax-M2.5，说明了实践中达到的规模：上下文长度高达 200K 令牌，超过十万个不同的智能体脚手架和环境，每日吞吐量达数百万样本。在这种规模下，生成和训练之间的任何同步屏障都会成为严重瓶颈。仅拖后腿问题（少数慢 rollout 阻塞整个批次）就可能导致数百个 GPU 空闲。

开源生态系统已汇聚于一个共同的架构响应：将推理与训练解耦到不同的 GPU 池，通过 rollout 缓冲区连接它们，并让双方并发运行。

我们正在为[TRL](https://github.com/huggingface/trl)开发一个新的异步训练器，这是最广泛使用的模型后训练库之一。为了指导我们的设计，我们调查了**十六个开源库**，这些库从一开始就围绕异步训练构建，并在**七个维度**上进行了比较：编排原语、缓冲区设计、权重同步协议、陈旧性管理、部分 rollout 处理、LoRA 支持以及分布式训练后端。本文提炼了我们从该调查中提取的设计原则。

除了强化学习（RL）之外，对异步基础设施的需求日益明显。例如，**策略蒸馏**，其中学生生成序列，教师对其进行评分，这反映了 GRPO，但将奖励函数替换为教师前向传播。认识到这种结构相似性，本调查中的所有内容同样适用于异步蒸馏。我们将在第 5 节回到这个更广泛的要点。

### 1.1 TRL 如何进行 RL 训练

TRL 当前的`GRPOTrainer`在一个同步的`training_step()`调用中实现了完整的 GRPO 循环（提示采样、生成、奖励评分、优势计算、梯度更新和权重同步）。这种设计简单且正确，但无法重叠生成与训练，导致显著的 GPU 利用率不足。

查看`GRPOTrainer`，在每个训练步骤中，我们依次有以下阶段：

1. **提示采样：**从数据集中抽取一批提示。这里没什么特别的，我们继续。
1. **生成**，调用`model.generate()`（或向前端 vLLM 服务器发送请求）以生成每个提示的 G 个完成。这是自回归的，并主导了实际时间。
1. **奖励评分：**根据一个或多个奖励函数评估每个完成。
1. **优势计算**
1. **前向和后向传播：**计算裁剪策略梯度损失并进行反向传播。
1. **优化器步骤**，更新模型权重。
1. **权重同步**，将更新后的权重推送到推理引擎（vLLM），以便下一次生成使用新策略。

每个阶段**阻塞**直到完成，然后下一个阶段才开始。时间线如下所示：

![同步 TRL 训练时间线](/images/others/keep-the-tokens-flowing-lessons-from-16-open-source-rl-libraries/014-d966e2aa.png)

TRL 提供`steps_per_generation`配置选项，以在多个梯度步骤中重用一组 rollout（时间重用），分摊生成成本。但生成调用本身仍然是完全同步和阻塞的；训练器在批次中每个完成都完成之前无法开始梯度计算。

该库还支持以`server`模式运行 vLLM 作为独立进程。它在生成期间释放训练 GPU，但两个硬同步屏障仍然存在：**HTTP 调用直到所有完成返回**，以及权重同步在传输期间阻塞训练器和 vLLM。

### 1.2 共置与分离训练

在讨论异步训练之前，理解使用独立推理引擎进行 RL 训练的两种部署拓扑至关重要：

- **共置模式**将推理和训练放在**同一组 GPU 上**。单个 GPU（或 TP 组）同时持有训练模型（在 FSDP 或 ZeRO 下）和推理引擎（vLLM 或 SGLang）。一次只有一个角色处于活动状态：在生成期间，训练模型的参数可能被卸载或重新分片为推理友好的布局（例如，从 FSDP 分片到 vLLM 的张量并行布局）；在训练期间，推理引擎被暂停或休眠。权重“同步”基本上是免费的；最多是在同一 GPU 上进行原地重新分片，而不是网络传输。共置模式的优点是简单性和成本；您需要更少的 GPU 总数。根本限制是**推理和训练无法重叠**。例如，这里是 Trl 与 vllm 在`colocate_mode`：

![TRL 与 vLLM 在共置模式](/images/others/keep-the-tokens-flowing-lessons-from-16-open-source-rl-libraries/015-fe40cd02.png)

- **分离模式**将推理和训练放在**独立的 GPU 池上**。推理池持续运行 vLLM 或 SGLang；训练池持续运行优化器。两个池通过权重同步协议（NCCL 广播、文件系统检查点、HTTP 等）和数据传输机制（Ray 对象存储、Redis 流、共享内存等）进行通信。

分离模式的最大优势是**推理和训练可以并发运行**。当训练器在批次 N 上计算梯度时，推理池已经在为批次 N+K 生成 rollout，从而实现异步训练。然而，这种好处是有代价的：需要额外的 GPU。

并发性、异步性和并行性是常被混淆的不同概念。在本文中，当我们说**“异步训练”**时，我们特指：**生成和训练并行运行，具有有效的重叠**；推理池在训练池计算当前批次梯度的同时，正在生成下一批 rollout。这本质上是分离模式的能力。共置模式可以通过优化如睡眠/唤醒内存管理或快速原地重新分片来加速推理，但无法实现真正的同步重叠；推理和训练仍然在同一 GPU 上轮流进行。本调查中实现有意义异步重叠的每个库都使用分离模式作为基础。

### 1.3 生成瓶颈

在推理模型的 RL 训练中，**自回归生成主导实际时间**。单个数学或编码任务的 rollout 可以产生 8K–64K 令牌的思维链推理（参见[QED-Nano rollout lengths](https://huggingface.co/spaces/lm-provers/qed-nano-blogpost#outcome-reward-rl-with-long-response-lengths)）。

为了具体说明这一点，考虑[vLLM 在单个 H100 80GB GPU 上的基准测试](https://www.databasemart.com/blog/vllm-gpu-benchmark-h100)（bf16，无量化，离线吞吐量模式）。一个 **7B 模型**（DeepSeek-R1-Distill-Qwen-7B）实现约 6,300 输出令牌/秒的聚合吞吐量；一个 **32B 模型**（DeepSeek-R1-Distill-Qwen-32B）降至约 1,200 输出令牌/秒。这些是*总*吞吐量，即推理引擎每秒可以推送的令牌数，无论有多少序列共享 GPU。

现在考虑一个典型的 GRPO 训练步骤：**G=8 个完成/提示 × 64 个提示/批次 = 512 个 rollout**。生成需要多长时间？

| 每个rollout的输出长度 | 总输出令牌数（512个rollout） | 在1×H100上的时间（7B @ \~6K令牌/秒） | 在1×H100上的时间（32B @ \~1.2K令牌/秒） |
| :-------------------- | :--------------------------- | :----------------------------------- | :-------------------------------------- |
| 2K令牌（短CoT）       | 约1M令牌                     | **约3分钟**                          | **约14分钟**                            |
| 8K令牌（中CoT）       | 约4M令牌                     | **约11分钟**                         | **约56分钟**                            |
| 32K令牌（长CoT）      | 约16M令牌                    | **约45分钟**                         | **约3.7小时**                           |

即使在短端（2K 令牌生成，使用 7B 模型），仅生成就消耗每个训练步骤数分钟。在长端，前沿推理模型越来越多地在此操作，单个生成阶段可能需要*数小时*在单个 GPU 上。扩展到 8 个推理 GPU 可将这些时间大致除以 8 倍（假设吞吐量线性扩展），但即便如此，32B 模型上的 32K 令牌展开仍需要约 28 分钟每步。

“**落后者问题（straggler problem）**进一步加剧了这一点。在基于组的算法如 GRPO 中，每个提示采样 G 个完成。批次无法继续，直到*最慢*的完成结束。思维链输出长度变化很大；单个提示可能产生从 1K 到 32K 令牌不等的完成。批次受最长完成限制，连续批处理仅部分缓解此问题：较短序列释放槽位用于新工作，但*最后一个* GRPO 组中的序列仍会阻塞该组的奖励计算和训练步骤。

### 1.4 核心洞察

本调查中的每个库都独立地收敛于相同的架构原则：**物理上将推理 GPU 与训练 GPU 分离，并异步推送权重**，使得生成永不停止，训练永不等待。

推理池持续运行，将完成的展开送入缓冲区。训练池从缓冲区拉取数据，计算梯度更新，并定期将新权重推送回推理池以保持同步。两个循环以各自的速度运行，通过缓冲区解耦。

这种设置高度可扩展，但引入了新一类问题：陈旧性（在旧策略下生成的展开）、权重同步开销、部分展开处理等。本文其余部分详细剖析了当前开源库如何解决这些问题。

---

## 2. 调查的库

| 库                | 组织                  | 仓库                                                                                     | GitHub ⭐ (2026年3月) |
| ----------------- | --------------------- | ---------------------------------------------------------------------------------------- | --------------------: |
| **AReaL**         | inclusionAI/Ant Group | [github.com/inclusionAI/AReaL](https://github.com/inclusionAI/AReaL)                     |                 4,338 |
| **ART**           | CoreWeave             | [github.com/OpenPipe/ART](https://github.com/OpenPipe/ART)                               |                 8,952 |
| **Atropos**       | NousResearch          | [github.com/NousResearch/atropos](https://github.com/NousResearch/atropos)               |                   878 |
| **MILES**         | radixark              | [github.com/radixark/miles](https://github.com/radixark/miles)                           |                   950 |
| **NeMo-RL**       | NVIDIA                | [github.com/NVIDIA-NeMo/RL](https://github.com/NVIDIA-NeMo/RL)                           |                 1,383 |
| **OAT**           | SAIL-SG               | [github.com/sail-sg/oat](https://github.com/sail-sg/oat)                                 |                   637 |
| **open-instruct** | AI2 (AllenAI)         | [github.com/allenai/open-instruct](https://github.com/allenai/open-instruct)             |                 3,611 |
| **PipelineRL**    | ServiceNow            | [github.com/ServiceNow/PipelineRL](https://github.com/ServiceNow/PipelineRL)             |                   374 |
| **PRIME-RL**      | PrimeIntellect        | [github.com/PrimeIntellect-ai/prime-rl](https://github.com/PrimeIntellect-ai/prime-rl)   |                 1,114 |
| **ROLL**          | Alibaba               | [github.com/alibaba/ROLL](https://github.com/alibaba/ROLL)                               |                 2,921 |
| **SkyRL**         | NovaSky-AI            | [github.com/NovaSky-AI/SkyRL](https://github.com/NovaSky-AI/SkyRL)                       |                 1,664 |
| **SLIME**         | THUDM                 | [github.com/THUDM/slime](https://github.com/THUDM/slime)                                 |                 4,595 |
| **TorchForge**    | Meta                  | [github.com/meta-pytorch/torchforge](https://github.com/meta-pytorch/torchforge)         |                   632 |
| **Tunix**         | Google                | [github.com/google/tunix](https://github.com/google/tunix)                               |                 2,175 |
| **verl**          | ByteDance             | [github.com/verl-project/verl](https://github.com/verl-project/verl)                     |                19,673 |
| **verifiers-rl**  | PrimeIntellect        | [github.com/PrimeIntellect-ai/verifiers](https://github.com/PrimeIntellect-ai/verifiers) |                 3,876 |

---

## 3. 比较框架：七个维度

为了理解快速扩展的异步 RL 库生态系统，我们提出了七个正交的比较维度。每个维度捕捉了一个影响库性能、复杂性和权衡的基本设计决策。

- **维度 1 – 编排与并发原语：** 分布式组件如何协调（Ray actors、asyncio、pub/sub、HTTP）。
- **维度 2 – 展开缓冲区设计：** 展开如何从推理流向训练。
- **维度 3 – 权重同步协议：** 更新后的权重如何到达推理服务器，以及系统是否必须暂停以接受它们或继续生成。
- **维度 4 – 陈旧性管理：** 如何处理离策略展开：版本拒绝、深度限制或重要性采样校正。
- **维度 5 – 部分展开处理：** 当权重更新在序列中途到达时，正在进行的生成会发生什么。
- **维度 6 – LoRA 训练支持：** 通用 LoRA 支持以及是否仅适配器参数可以训练和同步，实现亚毫秒级权重传输。
- **维度 7 – 分布式训练后端与并行性：** 训练使用何种并行策略，限制了最大模型大小。

### 维度 1：编排与并发原语

_系统如何协调其分布式组件？_

编排框架的选择决定了编程模型、故障语义和可扩展性上限。与其列出每个库的实现细节，该领域清晰地分解为四种**编排类型（orchestration types）**，这些基本协调范式在抽象级别、故障模型和部署要求上有所不同：

| 编排类型            | 定义                                                                                                                                   | 库                                                                                                 | 权衡                                                                                               |
| ------------------- | -------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| **分布式Actor模型** | 组件是*actors*，具有邮箱的隔离有状态进程，由运行时管理，处理调度、资源放置、容错和对象传输。通信通过异步RPC / futures / 对象存储进行。 | **Ray：** verl、SkyRL、NeMo-RL、SLIME、MILES、ROLL、OAT、open-instruct。**Monarch：** TorchForge。 | 最丰富的抽象；开箱即用地解决调度和容错。增加了非平凡的运行时依赖和框架特定的调试开销。             |
| **原生Python并发**  | 组件是线程、协程（`asyncio`）、`threading`原语、`multiprocessing`子进程和队列。无外部编排运行时。                                      | verifiers-rl、PipelineRL（池内）、ART（`asyncio` + 子进程代理）、AReaL（`asyncio`基于事件循环）    | 最小依赖，易于调试，完全控制。限于单节点，除非与额外IPC（Redis、HTTP、NCCL）配对用于多节点通信。   |
| **Pub/Sub消息总线** | 组件是通过仅追加流或消息队列通信的解耦生产者和消费者。本身不是编排，而是*数据传输层（data transport layer）*，连接独立运行的池。       | PipelineRL（池间：Redis`XADD`/`XREAD`流用于多节点，仅追加JSONL文件用于单节点）                     | 跨池边界清晰解耦，无需RPC。不管理进程生命周期、调度或故障恢复；必须与另一种编排类型配对。          |
| **HTTP微服务**      | 组件是通过REST API通信的独立服务。语言无关，最大解耦。                                                                                 | Atropos                                                                                            | 任何推理服务器，任何语言，零共享状态。最高延迟（如果使用NCCL）；无共享对象存储；容错是用户的责任。 |

> **关于 Tunix 的说明：** Tunix（Google）使用 JAX 原生网格模型，带有`ThreadPoolExecutor`用于异步重叠和`jax.device_put`用于跨网格权重传输。它在架构上与 PyTorch 生态系统足够不同，以至于在编排方面进行直接比较没有意义；它存在于 XLA/TPU 世界中，拥有自己的协调原语。

上表揭示了一个显著的模式：**调查的十六个库中有八个使用 Ray 作为其编排骨干**。这并非巧合；它反映了 actor 模型与 RL 训练结构之间的深层架构契合。[Anyscale（Ray 背后的公司）对开源 LLM RL 库的调查](https://www.anyscale.com/blog/open-source-rl-libraries-for-llms)证实了这种趋同。大规模 RL 训练涉及本质上异构的组件（推理引擎、训练引擎、环境、奖励模型），这些组件必须在集群中编排，通常在不同硬件类型上，具有不同的扩展需求和故障模式。Ray 的 actor 模型直接映射到这一点：

1. **Actor 隔离和异构资源。**每个 RL 组件（vLLM 推理服务器、FSDP 训练器、奖励模型、环境池）成为一个具有自身资源需求的 Ray actor（`num_gpus`，`num_cpus`，`memory`）。放置组提供对 GPU 亲和性的细粒度控制，无需手动 SSH/torchrun 编排。

1. **调度和自动扩展。**Ray 的调度器处理在集群中放置异构 actor 的组合问题。当生成需要比训练多 8 倍的 GPU 小时时，你可以直接告诉 Ray 独立扩展你的推理 actor。

1. **容错性。**长时间的 RL 训练运行（数天到数周）容易受到 GPU 故障、OOM 终止和网络分区的影响。Ray 的 actor 重启策略和对象存储复制提供了弹性，这在使用原始`asyncio`和`multiprocessing`时需要大量自定义基础设施。容错性的具体示例：`open-instruct`，例如，依赖 Ray 的 actor 监督来从 vLLM 引擎在 rollout 中途崩溃中恢复。

1. **用于零拷贝数据传输的对象存储。**Rollout 数据可能很大，对于非常长上下文的推理，每批次可达数十 GB。Ray 的共享内存对象存储支持同一节点上 actor 之间的零拷贝传输，避免了通常伴随`multiprocessing.Queue`方法的序列化开销。

1. **生态系统成熟度。**Ray 自 2017 年以来已在数千个 GPU 的生产部署中经过大规模实战测试。调试开销是真实的（Ray 仪表板、分布式堆栈跟踪、放置组故障），但替代方案——从头构建等效协调——在多节点规模上更糟。也就是说，Ray 是一个重量级依赖：它引入了自己的调度器、对象存储和仪表板，增加了并非每个团队都需要的操作复杂性。这正是为什么像 PRIME-RL、PipelineRL 和 AReaL 这样的库选择轻量级原生 Python 协调（asyncio、线程、Redis 流）的原因——当你控制完整堆栈且部署拓扑固定时，原生 Python 的简单性和可调试性通常超过 Ray 提供的便利。

代价是对一个非平凡运行时的硬依赖。这种权衡可能是值得的，特别是对于生产规模训练（64+ GPU、多天运行、复杂奖励计算）。

虽然 Ray 的 actor 模型是领域的主要参与者，但[Monarch](https://github.com/pytorch/monarch)作为 Meta 推出的新 PyTorch 原生分布式 actor 框架出现，专为 GPU 工作负载构建。与 Ray 类似，Monarch 基于 actor 模型；组件是通过消息通信的独立 actor，但它是从头设计用于 PyTorch/CUDA 生态系统，而不是作为通用分布式运行时。

Monarch 提供了几个与异步 RL 特别相关的能力。一个[使用 Monarch 的异步 RL 示例实现](https://allenwang28.github.io/monarch-gpu-mode/05_rl_intro.html)（来自 GPU Mode 讲座系列）演示了架构：生成器、回放缓冲区和训练器被建模为 Monarch actor，回放缓冲区吸收来自落后 rollout 的延迟变化，RDMA 权重同步将更新后的参数推送到生成器而不阻塞训练。该模式在结构上与基于 Ray 的设计（verl、SkyRL、open-instruct）相同，但使用纯 PyTorch 原生原语实现。

### 维度 2：Rollout 缓冲区设计

_生成的 rollout 如何从推理流向训练，以及流水线有多深？_

缓冲区是位于生成和训练之间的数据结构。其深度控制最大异步程度，因此控制最大陈旧度。

| 模式                     | 深度 | 库                                                                                                                                             | 特征                                           |
| ------------------------ | ---- | ---------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------- |
| **无缓冲区**（同步）     | 0    | TRL（当前）、**ART**（收集全部然后训练）                                                                                                       | 生成和训练严格交替；零陈旧度，最大空闲时间     |
| **双缓冲区**（一步提前） | 1    | verifiers-rl、SLIME（异步模式）、MILES、OAT                                                                                                    | 在训练步骤N开始时提交生成N+1；恰好重叠一个批次 |
| **有界异步队列**         | 2–K  | SkyRL、verl（完全异步）、NeMo-RL、ROLL、PRIME-RL、TorchForge、Tunix、**open-instruct**（`async_steps`）、**AReaL**（`max_head_offpolicyness`） | 多个批次在飞行中；陈旧度受队列容量限制         |
| **无界/流**              | 无限 | PipelineRL（Redis流）、SLIME（完全异步模式）、Atropos                                                                                          | 连续生成；陈旧度仅受显式版本控制限制           |

双缓冲区模式是从同步训练升级到异步训练的最简单方式：它恰好重叠一个生成与一个训练步骤，并引入最多一步的策略滞后！另一方面，更深的队列提高了吞吐量，但需要陈旧度管理。

缓冲区控制有多少数据在飞行中。但数据只是方程的一半。另一半是在这些 rollout 变得陈旧之前将更新后的权重回送到推理服务器。这就是权重同步的用武之地！

### 维度 3：权重同步协议

_范围说明：_

> **本轴专注于**分离模式（disaggregated mode），即推理和训练在独立的 GPU 池上运行，因为这是异步重叠（以及因此权重同步设计）真正重要的部署拓扑。共置模式（推理和训练使用相同 GPU）本质上是同步的，不会面临下面讨论的传输/中断权衡。

这是在架构上最具决定性的维度。协议决定同步延迟、中断粒度以及是否可以进行部分 rollout。

这里需要做一个关键区分：**传输机制**和**中断模型**。大多数库在粗粒度边界（HTTP 请求、完整批次甚至完整训练步骤）处暂停生成，然后才发起权重传输。PipelineRL 是个例外：它根本不停止生成。

**传输机制：**

| 机制                   | 延迟        | 库                                                                                  |
| :--------------------- | :---------- | :---------------------------------------------------------------------------------- |
| **NCCL 广播**          | \~100–500ms | PipelineRL, SkyRL, SLIME, MILES, ROLL, OAT, NeMo-RL, PRIME-RL, open-instruct, AReaL |
| **NCCL + 分桶**        | \~20ms      | verl                                                                                |
| **KV + 共享内存**      | 低          | TorchForge                                                                          |
| **文件系统 + HTTP**    | 中          | PRIME-RL, AReaL, ART                                                                |
| **CUDA IPC（零拷贝）** | 极低        | NeMo-RL, MILES                                                                      |
| **JAX 跨网格**         | 低          | Tunix                                                                               |
| **HTTP PUT**           | 高          | verifiers-rl                                                                        |
| **文件系统 + 重启**    | 极高        | Atropos                                                                             |

**在中断模型中，生成何时暂停以接受新权重？**

这就是 PipelineRL 从根本上与所有其他库不同的地方。与其单独列出每个库，不如将其归纳为几个概念层级，按从最细到最粗的中断粒度排序：

| 中断粒度                              | 发生了什么                                                                                      | 库                                                           |
| :------------------------------------ | :---------------------------------------------------------------------------------------------- | :----------------------------------------------------------- |
| **从不**（每次前向传播飞行中）        | 序列从不停止。权重交换发生在令牌解码步骤之间（\~1-10ms 间隙）。运行中的序列无缝地以新权重继续。 | PipelineRL, open-instruct（可选启用）                        |
| **每次 HTTP 请求**（中止 + 重新同步） | 飞行中的 HTTP 请求被中止。部分令牌通过前缀恢复机制重新提交或循环重试。                          | SkyRL, SLIME, MILES                                          |
| **软暂停**（排空进行中任务）          | 在进行中的任务自然完成时，不接受新的生成请求。排空后，同步权重并恢复生成。                      | PRIME-RL, AReaL, open-instruct（默认）, verl（异步）         |
| **每训练步骤/批次**（阻塞）           | 生成必须完全完成。训练器和推理引擎轮流阻塞对方。                                                | NeMo-RL, ROLL, OAT, TorchForge, Tunix, verifiers-rl, Atropos |

"从不停止"层级与所有其他层级在本质上不同：PipelineRL 将钩子嵌入推理引擎，使锁在*每次 Transformer 前向传播*（一个序列的一个令牌步骤）时被获取和释放。权重更新最多等待一次前向传播（约几毫秒），交换所有参数，然后立即恢复生成。其他所有库在更粗的边界处停止生成，从一次 HTTP 请求（约数百毫秒）到完整的批次边界（约数秒）不等。

权重同步控制*何时*新权重到达。但异步训练意味着 rollout 始终在*某个*策略版本下生成，而该*生成中的*策略可能落后训练器若干梯度步骤。各库如何处理这种策略滞后就是陈旧性管理。

### 维度 4：陈旧性管理

_系统如何处理生成的 rollout 可能来自比当前训练策略更旧的策略这一事实？_

一旦生成和训练重叠，样本就会变为离策略（off-policy）。已出现三种**正交**策略来管理这种陈旧性，大多数生产系统会组合使用多种策略：

**策略 1：逐样本版本拒绝。** 每个样本都标记有生成它的策略版本（整数）。在训练时，版本落后于当前策略超过阈值的样本在进入损失计算之前被硬丢弃。简单且正确，但浪费了生成被丢弃样本所花费的宝贵计算量。

**策略 2：深度限制（Depth Bounding）。** 生成与训练之间的队列或缓冲区具有有界容量（或显式陈旧性门控），从架构上限制了任何样本可以落后多远。这从 depth=1（单步提前双缓冲，构造上不可能出现陈旧性）到与版本差距绑定的显式容量公式不等。无需逐样本版本跟踪；边界由系统的流水线深度强制执行。

**策略 3：IS 加权损失校正（IS-weighted loss correction）。** 到达训练器的陈旧样本通过重要性采样比 $\pi_{\text{old}}(a|s) / \pi_\theta(a|s)$ 进行重新加权，通常进行截断（截断 IS）。一些库还应用 OPSM（对具有负优势的离策略样本将损失归零）。这保留了吞吐量；没有样本被丢弃，但 IS 比率会带来梯度方差的代价。

这些策略是正交的：一个系统可以单独使用版本拒绝、单独使用深度限制、单独使用 IS 校正，或它们的任意组合。同步系统通过从不重叠生成和训练来完全避免这个问题。

| 库                | 版本拒绝 | 深度限制 | IS 校正 | 关键配置 / 说明                                                                                                      |
| ----------------- | :------: | :------: | :-----: | -------------------------------------------------------------------------------------------------------------------- |
| **AReaL**         |    ❌    |    ✅    |   ⚠️    | `max_head_offpolicyness` capacity formula; optional `use_decoupled_loss` adds IS weight capped at 5.0                |
| **ART**           |    —     |    —     |    —    | Synchronous; all rollouts collected before training; no staleness by design                                          |
| **Atropos**       |    ❌    |    ✅    |   ❌    | `max_batches_offpolicy`, ceiling on buffered batches                                                                 |
| **MILES**         |    ❌    |    ❌    |   ✅    | TIS + OPSM                                                                                                           |
| **NeMo-RL**       |    ✅    |    ❌    |   ❌    | `max_trajectory_age_steps`, per-sample version drop                                                                  |
| **OAT**           |    ❌    |    ❌    |   ✅    | Clipped TIS ratio                                                                                                    |
| **open-instruct** |    ❌    |    ✅    |   ⚠️    | `async_steps` cap (default 1, production 8); optional `--truncated_importance_sampling_ratio_cap ρ` adds clipped TIS |
| **PipelineRL**    |    ✅    |    ❌    |   ❌    | `max_lag`, integer version tag per sample; drop if age exceeds threshold                                             |
| **PRIME-RL**      |    ✅    |    ✅    |   ✅    | Full hybrid: `max_async_level` version gap + `max_off_policy_steps` cancellation + IPO trust-region IS               |
| **ROLL**          |    ❌    |    ❌    |   ✅    | Richest IS suite: TIS, TOPR, CISPO, Kimi15, six off-policy loss variants                                             |
| **SkyRL**         |    ❌    |    ✅    |   ❌    | `max_staleness_steps`, capacity gate blocks new rollouts when exceeded                                               |
| **SLIME**         |    ❌    |    ❌    |   ✅    | TIS + OPSM (off-policy masking for partial rollouts)                                                                 |
| **TorchForge**    |    ✅    |    ❌    |   ❌    | `max_policy_age`, per-sample version tag; hard drop                                                                  |
| **Tunix**         |    ❌    |    ✅    |   ❌    | Bounded queue + sync per step; staleness structurally limited                                                        |
| **verl**          |    ❌    |    ❌    |   ✅    | Clipped TIS ratio; optional OPSM                                                                                     |
| **verifiers-rl**  |    ❌    |    ✅    |   ❌    | Depth=1 FIFO + sync every step; staleness impossible by construction                                                 |

> ✅ = 是，❌ = 否，⚠️ = 可选/可配置，— = 不适用（同步模式）

- **版本拒绝**简单且正确，但当很多样本被丢弃时会浪费计算量。
- **IS 校正**以梯度方差为代价保留了吞吐量。
- **深度限制**是最粗粒度的机制，但完全避免了逐样本的记录开销。

生产系统（PRIME-RL、AReaL、open-instruct）的趋势是采用**混合方法**，将深度限制与可选的 IS 校正相结合，获得有界队列的架构简洁性，同时通过重要性加权为稳定训练提供损失层面的安全网。

陈旧性管理处理的是在旧策略下生成的数据。但当权重更新到来时，仍在生成中的数据呢？

### 维度 5：部分 Rollout 处理

_当权重更新到来时，进行中的生成会发生什么？_

对于单个 rollout 可能需要数分钟的长上下文任务，这一点至关重要。主要有以下几种策略：

| 策略                             | 库                                | 描述                                                                                                                                          |
| -------------------------------- | --------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| **隐式延续**                     | PipelineRL                        | 序列从不被中断。权重在前向传播之间交换；序列以新权重继续。存储的 logprobs 仍然有效，因为训练使用*记录的* π_old，而非重新计算。                |
| **中止 + 前缀重试**              | SkyRL, SLIME                      | 活动序列被中止。部分令牌被累积，然后使用新权重通过前缀恢复机制重新提交。                                                                      |
| **显式保存/恢复**                | verl（完全异步）                  | rollout 工作器将部分令牌 ID 和 logprobs 保存到缓冲区，等待同步，然后从保存的前缀恢复。                                                        |
| **组取消（生成继续）**           | PRIME-RL                          | 陈旧的 rollout 组的异步任务被取消；推理服务器继续处理进行中的 HTTP 请求，但其结果被丢弃。权重同步在 HTTP 请求之间触发，不中断请求中途的生成。 |
| **不支持部分 rollout**           | verifiers-rl, OAT, Atropos, Tunix | 权重同步只在批次边界发生。进行中的生成必须在同步开始前完成。                                                                                  |
| **软暂停，进行中序列完成**       | **AReaL**                         | 暂停信号阻止新的 KV 缓存分配，但不中止进行中的序列。任务调度器停止提交新任务；运行中的任务运行至完成。权重同步后，生成调度恢复。              |
| **完全休眠，同步时无进行中请求** | **ART**                           | 按设计，训练只在所有 rollout 收集完毕后开始。触发休眠时从无进行中序列。一级休眠（存在进行中请求时）将 KV 缓存卸载到 CPU；二级休眠完全丢弃它。 |
| **排空或飞行中（可配置）**       | **open-instruct**                 | 默认：停止标志门控新的预取；权重更新等待活动任务排空。启用飞行中更新时，绕过排空，在令牌仍在生成时广播权重；进行中序列以旧新权重混合继续。    |

目前为止，每个维度都假设了全参数训练。但在 LoRA 训练中，训练的只是几百万适配器参数而非数十亿参数，权重同步问题几乎消失了。让我们看看这些库如何支持 LoRA 训练。

### 维度 6：LoRA 训练支持

_库是否支持通过 LoRA 适配器进行参数高效训练，支持哪些模式，是否利用了仅适配器权重同步？_

对于 GPU 预算有限的团队，LoRA 可以说是最具实践意义的维度。它将可训练参数量减少 99%+，将峰值激活内存减半，并且当推理服务器支持 LoRA 时，可以实现*仅适配器权重同步*：不需要广播 7B+ 模型的每个参数（NCCL 需 \~100–500ms），只需将适配器增量推送到 vLLM，在 rank 32 时约为 \~50 MB，传输时间不到一毫秒。

| 库                | LoRA 支持                  | 模式限制                         | LoRA 后端                                      | 仅适配器同步                                                                                      |
| ----------------- | -------------------------- | -------------------------------- | ---------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| **AReaL**         | ✅ Yes                     | FSDP2 only (not Megatron/Archon) | HF `peft`                                      | ✅ Yes (disk-based sync; only trainable params transferred; vLLM adapter hot-swap)                |
| **ART**           | ✅ Yes (primary design)    | Both (shared + dedicated GPU)    | Unsloth/`peft` (default); custom Megatron LoRA | ✅ Yes (only adapter saved/loaded; in-process or HTTP adapter hot-swap; base weights never moved) |
| **Atropos**       | ✅ Yes                     | Disaggregated                    | HF `peft`                                      | ✅ Yes (`lora_only` / `lora_restart` modes)                                                       |
| **MILES**         | ✅ Yes                     | Both (colocated + disaggregated) | Megatron-Bridge                                | ✅ Yes (adapter sync config for SGLang)                                                           |
| **NeMo-RL**       | ✅ Partial\*               | Both                             | Custom (not `peft`)                            | ❌ No evidence                                                                                    |
| **OAT**           | ✅ Yes                     | Both                             | HF `peft`                                      | ✅ Yes (LoRA-only sync mode)                                                                      |
| **open-instruct** | ⚠️ Code exists, not wired‡ | —                                | HF `peft` (SFT/DPO only)                       | ❌ No (LoRA not applied in the RL trainer)                                                        |
| **PipelineRL**    | ✅ Yes                     | Non-colocated                    | HF `peft`                                      | ❌ No (full NCCL broadcast)                                                                       |
| **PRIME-RL**      | ✅ Yes                     | Disaggregated                    | Custom MultiLoRA (not `peft`)                  | ✅ Yes (adapter-only state dict extraction)                                                       |
| **ROLL**          | ✅ Partial†                | DeepSpeed backend only           | HF `peft` / TRL                                | ❌ No evidence                                                                                    |
| **SkyRL**         | ✅ Yes                     | Both                             | `peft` (FSDP) / Megatron-Bridge (Megatron)     | ✅ Yes (filesystem-based adapter sync)                                                            |
| **SLIME**         | ❌ No                      | —                                | —                                              | ❌ No                                                                                             |
| **TorchForge**    | ❌ No                      | —                                | —                                              | ❌ No                                                                                             |
| **Tunix**         | ✅ Yes                     | Both                             | qwix (JAX-native)                              | ✅ Yes (auto-detected)                                                                            |
| **verl**          | ✅ Yes (most complete)     | Both                             | `peft` (FSDP) / Megatron-Bridge (Megatron)     | ✅ Yes (unmerged adapter sync)                                                                    |
| **verifiers-rl**  | ✅ Yes (via prime-rl)      | Disaggregated                    | HF `peft` + FSDP2 + vLLM                       | ✅ Yes (vLLM LoRA serving)                                                                        |

\* NeMo-RL: LoRA for GRPO and DPO is supported only on the DTensor backend; the Megatron Core backend is SFT-only (RL LoRA listed as "coming soon"). Uses a custom DTensor-compatible LoRA module (not `peft`), optionally with Triton kernels.

† ROLL: LoRA is officially supported with the DeepSpeed training backend only. Megatron-backend LoRA appeared in the Feb 2026 changelog but remains experimental.

‡ open-instruct: The model config exposes LoRA-related fields (`use_peft`, `lora_r`, `lora_alpha`), and adapter saving is handled in the checkpoint logic. However, the `peft` model is never initialised in the RL training path; LoRA remains an SFT/DPO-only feature for the RL trainer as of March 2026.

**三种 LoRA 实现家族：**

1. **HuggingFace `peft`**（PipelineRL、SkyRL/FSDP、verifiers-rl、ROLL、OAT、Atropos）：最常见的选择。标准检查点格式（`adapter_model.safetensors`），与任何 HF Transformers 训练循环兼容。ZeRO-3 交互需要注意：例如，OAT 需要禁用融合 LM head；ROLL 必须完全禁用梯度检查点。

1. **Megatron-Bridge**（verl/Megatron、SkyRL/Megatron、MILES）：3D 并行训练（TP × PP × DP）所必需。支持多种 LoRA 类型：`lora`、`canonical_lora`（将合并的 QKV 拆分为独立的 Q/K/V 适配器），`vlm_lora`，以及`dora`。该`canonical_lora`变体避免了 QKV 合并，从而提高了训练稳定性。MILES 以 HF`peft`格式和 Megatron 原生每秩格式保存检查点。

1. **自定义实现**（NeMo-RL、PRIME-RL、Tunix/qwix）：特定于库的 LoRA 模块，无法与`peft`检查点互操作。PRIME-RL 独特地支持在单次运行中同时使用多个适配器，以实现多实验并行。Tunix 使用 Google 的`qwix`JAX 库，该库内置了 QLoRA（NF4 量化）和 TPU 原生梯度路由。NeMo-RL 使用自定义的 DTensor 兼容模块，并可选 Triton 融合内核。

**仅适配器权重同步机会（与维度 3 的交互）：**

十三个库中有八个支持仅将 **LoRA 适配器增量**推送到推理服务器。这完全改变了权重同步问题（维度 3）的性质。在使用全参数训练时，中断模型（每次前向传播锁定 vs. 每次请求中止 vs. 每批次暂停）决定了在 NCCL 广播期间浪费了多少生成。当使用 LoRA 并仅同步适配器时，传输量非常小，以至于几乎任何中断模型都能提供相当的吞吐量！即使是 Atropos 的暴力 HTTP 热交换也变得可行。

---

### 维度 7：分布式训练后端与并行性

_库使用何种并行策略进行训练，这如何约束或启用异步架构？_

此轴贯穿所有其他轴。训练后端的选择决定了每个 GPU 能容纳多大的模型、在广播到推理服务器之前需要多少集合操作来收集权重，以及哪些模型架构能够被训练。对于团队扩展超过 300 亿参数或从密集模型转向专家混合（Mixture-of-Experts）模型来说，这是最具决定性的决策。

| 库                | 训练后端                   | 并行性                 | HF模型加载        | MoE / EP支持 |
| :---------------- | :------------------------- | :--------------------- | :---------------- | :----------- |
| **AReaL**         | FSDP2、Megatron、Archon    | DP、SP、TP、PP、CP、EP | ✅ 直接 / 转换    | ✅           |
| **ART**           | Unsloth、Megatron          | DP、TP、EP             | ✅ 直接 / 转换    | ✅           |
| **Atropos**       | PyTorch原生、TRL           | DP                     | ✅ 直接           | ❌           |
| **MILES**         | Megatron、FSDP2            | DP、TP、PP             | 🔄 转换           | ✅           |
| **NeMo-RL**       | FSDP2、Megatron            | DP、SP、TP、PP、CP、EP | ✅ 直接 / 转换    | ✅           |
| **OAT**           | DeepSpeed                  | DP、TP                 | ✅ 直接           | ❌           |
| **open-instruct** | DeepSpeed                  | DP、SP                 | ✅ 直接           | ❌           |
| **PipelineRL**    | DeepSpeed                  | DP、SP                 | ✅ 直接           | ❌           |
| **PRIME-RL**      | FSDP2                      | DP、TP、CP、EP         | ✅ 直接           | ✅           |
| **ROLL**          | DeepSpeed、Megatron、FSDP2 | DP、SP、TP、PP、CP、EP | ✅ 直接 / 转换    | ✅           |
| **SkyRL**         | FSDP、Megatron             | DP、SP、TP、PP、EP     | ✅ 直接 / 转换    | ✅           |
| **SLIME**         | Megatron                   | DP、TP、PP、SP         | 🔄 转换           | ✅           |
| **TorchForge**    | FSDP2                      | DP、TP、CP             | ✅ 通过TorchTitan | ❌           |
| **Tunix**         | JAX/XLA                    | DP、TP                 | ❌ 自定义Flax     | ❌           |
| **verl**          | FSDP、Megatron             | DP、SP、TP、PP、CP、EP | ✅ 直接 / 转换    | ✅           |
| **verifiers-rl**  | DeepSpeed                  | DP                     | ✅ 直接           | ❌           |

训练后端对异步 RL 库设计产生直接影响：

**权重同步速度直接取决于训练后端，更快的同步意味着更少的陈旧性。**

在解耦的异步设置中，权重同步*不一定*会阻塞推理。关键设计决策是**权重更新如何与进行中的生成交互**；存在四种策略，按破坏性从低到高排序：

- **原子交换，无中断。**完整的权重更新作为单个阻塞 RPC 发送到推理引擎。每次前向传播要么看到全部旧权重，要么看到全部新权重，绝不会混合。生成最多暂停一次前向传播的间隔（约几毫秒）。（PipelineRL）
- **每参数流式传输，无中断。**每个参数作为单独的 RPC + NCCL 广播发送。前向传播在单个参数更新之间交错，因此进行中的序列确实会在不同层看到新旧权重的混合。最大重叠，但一致性最弱。（open-instruct，飞行模式）
- **调度门，排空进行中任务，然后同步。**新请求被暂缓，直到进行中的序列自然完成；权重仅在流水线排空后广播。无浪费令牌，但同步气泡与最长进行中序列成正比。（PRIME-RL、AReaL、open-instruct 默认、verl 完全异步）
- **硬暂停或中止。**推理被暂停，或在权重传输开始前中止进行中的请求。最清晰的一致性，最高的计算浪费。（verl、SkyRL）

但即使在推理继续的库中，**较慢的同步意味着推理在陈旧权重上运行的时间更长**。训练器与推理池之间的策略版本差距随同步持续时间增长。这是需要考虑的因素。

**随着领域向稀疏模型发展，MoE 支持日益成为重要的差异化因素。**
趋势很明显：前沿模型是稀疏的（DeepSeek-V3、Qwen3-MoE、Mixtral、DBRX），开源权重的 MoE 正成为后训练的默认起点。训练这些模型需要专家并行（EP），将不同专家分配到不同秩，而大多数异步 RL 库不支持。只有基于 Megatron 的库（verl、SLIME、MILES、ROLL、NeMo-RL）和 PRIME-RL 的 FSDP2+EP 路径能正确处理 EP。基于 ZeRO 的库（PipelineRL、verifiers-rl、OAT、open-instruct）可以*加载* MoE HuggingFace 模型类，但如果没有 EP，每个专家会跨所有 ZeRO-3 秩分片，而不是放置在专用秩上；每次前向传播都会 AllGather 每个专家，完全抵消了稀疏性优势。EP 也使权重同步复杂化：在广播到 vLLM/SGLang（通常从单个 TP 组服务所有专家）之前，训练器必须从每个 EP 秩 AllGather 专家参数，这是一个 O(N∗experts×E∗size)的通信（其中 E_size 是每个专家的参数数量），这在密集模型中不存在。对于一个拥有 256 个专家的 235B MoE，这是一个显著的同步成本。希望在后训练下一代开源 MoE 模型时保持相关性的库需要 EP 感知的训练\*和\*EP 感知的权重同步。

**MoE LoRA 是一个新兴需求，也是一个棘手的问题。**
密集模型上的 LoRA 是众所周知的（维度 6）：将适配器附加到注意力投影，训练它们，仅同步适配器增量。MoE LoRA 更难，因为自然目标是*专家 FFN 层（expert FFN layers）*，即每个专家获得自己的适配器。对于一个拥有 64 个专家且每个专家的 gate/up/down 投影使用 rank-32 LoRA 的模型，适配器数量从 \~20（密集）跃升至 \~200+（MoE），且适配器分布在各个 EP 秩上。权重同步必须在将其推送到推理服务器之前从每个 EP 秩收集适配器，这是密集 LoRA 不存在的协调问题。在调查的库中，只有 **ART** 明确实现了 MoE 专家 LoRA 层（带有每专家 LoRA 和手动 allreduce 的 Megatron EP 路径），**MILES** 通过可以针对专家层的 Megatron-Bridge 支持 LoRA。verl 的 Megatron-Bridge 路径支持包括 `vlm_lora` 在内的 LoRA 类型，但 MoE 特定的专家 LoRA 没有文档说明。vLLM 的 LoRA 服务不原生支持每专家适配器；它加载均匀应用的单个适配器，因此 MoE LoRA 的仅适配器同步目前需要自定义推理侧逻辑。随着 MoE 模型成为后训练的默认选择，具有高效仅适配器同步的 MoE LoRA 将是一个亟待填补的关键能力缺口。

七个维度至此全部介绍完毕，每个维度捕捉了同一底层问题的不同侧面。综合在一起，它们为我们提供了比较这些库的完整视角。现在把所有内容汇总到一个页面上。

---

## 4. 全局概览：十六个库一览

> **说明：** 本概览反映了这些库截至 2026 年 3 月的状态。生态系统正在快速演进；具体功能、后端和集成可能在不久的将来发生变化。

| 库                | 组织           | 编排类型                                                 | 推理服务器             | 权重同步                                    | 陈旧性管理               | 部分 Rollout                        | 训练后端                              | 分布式并行性                                                             | LoRA 支持                                  |
| ----------------- | -------------- | -------------------------------------------------------- | ---------------------- | ------------------------------------------- | ------------------------ | ----------------------------------- | ------------------------------------- | ------------------------------------------------------------------------ | ------------------------------------------ |
| **AReaL**         | inclusionAI    | Native Python (asyncio + HTTP RPC); pluggable Ray/Slurm  | vLLM, SGLang           | NCCL chunked OR filesystem safetensors      | Depth + IS (optional)    | 🟧 Soft pause (in-flight complete)  | FSDP2 or Megatron-LM or Archon        | FSDP2: DP+SP+TP; Megatron: TP+SP+PP+CP+EP; Archon: FSDP2+TP+SP+PP+EP     | ✅ `peft` (Adapter-only)                   |
| **ART**           | OpenPipe       | Native Python (asyncio + mp child processes)             | vLLM                   | LoRA adapter swap (no full weight transfer) | Synchronous (none)       | ❌ No                               | Unsloth (single-GPU); Megatron-LM     | None (Unsloth); TP×EP×DP (Megatron)                                      | ✅ `peft` / Megatron LoRA (Adapter-only)   |
| **Atropos**       | NousResearch   | HTTP Microservices (FastAPI)                             | vLLM, SGLang, OpenAI   | FS checkpoint + vLLM restart                | Depth bounding           | ❌ No                               | Single-GPU PyTorch; TRL/Accelerate    | None (native); FSDP/ZeRO via TRL adapter                                 | ✅ `peft` (Adapter-only)                   |
| **MILES**         | radixark       | Distributed Actor (Ray)                                  | SGLang                 | NCCL OR CUDA IPC                            | IS correction            | 🟧 Abort + recycle to buffer        | Megatron-LM (primary); FSDP2          | Megatron: TP×PP×DP; FSDP2 available; colocated CUDA IPC                  | ✅ Megatron-Bridge (Adapter-only)          |
| **NeMo-RL**       | NVIDIA         | Distributed Actor (Ray)                                  | vLLM, SGLang, Megatron | NCCL OR CUDA IPC-ZMQ OR HTTP                | Version rejection        | ✅ In-flight continuation           | DTensor (FSDP2+TP) or Megatron-Bridge | DTensor: TP+SP+CP+FSDP2; Megatron: TP×PP×CP×EP×ETP + FSDP2               | 🟧 Custom (No adapter-only sync)           |
| **OAT**           | SAIL-SG        | Distributed Actor (Ray)                                  | vLLM                   | NCCL per-param + ZeRO-3 gather              | IS correction            | ❌ No                               | DeepSpeed ZeRO-2/3                    | ZeRO-2 / ZeRO-3 DP; AutoTP                                               | ✅ `peft` (Adapter-only)                   |
| **open-instruct** | AI2 (AllenAI)  | Distributed Actor (Ray)                                  | vLLM                   | NCCL broadcast; optional in-flight updates  | Depth + IS (optional)    | 🟧 Drain-or-inflight (configurable) | DeepSpeed ZeRO-0/2/3                  | ZeRO-3 DP + Ulysses SP; vLLM TP (inference only)                         | ❌ No                                      |
| **PipelineRL**    | ServiceNow     | Native Python + Pub/Sub (asyncio + Redis/JSONL)          | vLLM                   | NCCL pg + HTTP notify                       | Version rejection        | ✅ Implicit continuation            | DeepSpeed ZeRO-3                      | ZeRO-3 DP + Ring SP; ZeRO++ available                                    | ✅ `peft` (Full sync)                      |
| **PRIME-RL**      | PrimeIntellect | Native Python (asyncio + FS/ZMQ)                         | vLLM                   | Filesystem safetensors + HTTP OR NCCL       | Version + depth + IS     | 🟧 Group cancellation               | FSDP2 (exclusively)                   | FSDP2 per-block + TP + CP + EP; pp=1                                     | ✅ Custom MultiLoRA (Adapter-only)         |
| **ROLL**          | Alibaba        | Distributed Actor (Ray)                                  | vLLM, SGLang           | NCCL via dedicated update group             | IS correction            | ❌ No                               | DeepSpeed ZeRO or Megatron or FSDP2   | DS: ZeRO+Ulysses SP; Megatron: TP×PP×CP×EP; FSDP2: HSDP+Ulysses          | 🟧 `peft` (DeepSpeed only)                 |
| **SkyRL**         | NovaSky-AI     | Distributed Actor (Ray) + Native Python                  | vLLM, SGLang           | NCCL pg                                     | Depth bounding           | 🟧 Abort + retry with prefix        | FSDP/FSDP2 or Megatron-Bridge         | FSDP: ZeRO shard + Ulysses SP; Megatron: full 5D via bridge; JAX backend | ✅ `peft` / Megatron-Bridge (Adapter-only) |
| **SLIME**         | THUDM          | Distributed Actor (Ray)                                  | SGLang                 | NCCL pg, bucketed                           | IS correction            | 🟧 Abort + recycle to buffer        | Megatron-LM                           | TP×PP×DP; Megatron→HF conversion; MoE EP all-gather                      | ❌ No                                      |
| **TorchForge**    | Meta           | Distributed Actor (Monarch)                              | vLLM                   | torchstore + shared memory prefetch         | Version rejection        | ❌ No                               | FSDP2 via TorchTitan                  | FSDP2 + TP; CP partial; PP not yet implemented                           | ❌ No                                      |
| **Tunix**         | Google         | Native Python (ThreadPoolExecutor + asyncio); JAX-native | vLLM, SGLang, JAX      | Cross-mesh reshard                          | Depth bounding           | ❌ No                               | JAX/XLA 2D mesh                       | 2D JAX mesh: FSDP + TP; no PP; TPU-primary                               | ✅ qwix / QLoRA (Adapter-only)             |
| **verl**          | ByteDance      | Distributed Actor (Ray)                                  | vLLM, SGLang           | NCCL + checkpoint-engine buckets            | IS correction            | ✅ Explicit save/resume             | FSDP1/FSDP2 or Megatron-Core          | FSDP: ZeRO-2/3/HSDP + Ulysses SP; Megatron: TP×PP×VPP×CP×EP×ETP          | ✅ `peft` / Megatron-Bridge (Adapter-only) |
| **verifiers-rl**  | PrimeIntellect | Native Python (threading + asyncio)                      | vLLM                   | PyNCCL broadcast                            | Depth bounding (depth=1) | ❌ No                               | DeepSpeed ZeRO-3 (Accelerate)         | ZeRO-3 DP only; no TP/PP                                                 | ✅ `peft` (Adapter-only)                   |

这就是当前的态势。但该领域正在快速发展，若干新兴趋势即将以设计者可能未曾预料的方式对这些架构进行压力测试。

---

## 5. 下一波浪潮：设计启示

以下趋势并非新技术的目录；每一个都对当前的基础设施和算法选择造成具体压力。问题不是"前沿是什么？"，而是"如果这一趋势胜出，我的当前技术栈中会出现什么问题？"

### 5.1 无评论家算法：内存释放，但权重同步压力增加

PPO 的价值网络使任何训练节点的内存占用翻倍。该领域正在向无评论家（critic-free）变体（GRPO、REINFORCE++、Online DPO）收敛，正是因为长 CoT 推理使这种开销在 8K–64K 上下文长度下难以承受。

**这解锁了什么：**消除评论家（critic）可释放约 50%的训练 GPU 内存。这些空闲资源可重新分配给：（a）更大的 rollout 批次，直接减少掉队者（straggler）方差问题，或（b）在同一 GPU 上共同部署推理和训练，从而完全消除对单独的 NCCL 权重同步进程组的需求。

**它没有解决什么：**无评论家（critic-free）方法仍然需要频繁地将权重推送到推理服务器。实际上，它们可能*增加*同步压力：没有价值网络（value network）提供稳定的基线，GRPO 风格算法需要更大的组大小（G=8–32）以获得低方差优势估计，这意味着每个步骤有更多 rollout 和更快的策略漂移（policy drift）。仅在粗粒度边界（每个训练步骤或每 K 步）同步的库将在无评论家训练下看到陈旧性（staleness）增长更快。

**非对称轨迹过滤**（GRPO-RoC：过采样 rollout，严格过滤正例，均匀下采样负例；DeepSeek-V3.2 和 MiniMax-M1 中的 CISPO/DAPO 风格非对称裁剪）对陈旧性有更微妙的影响。问题不在于批次本身缩小；而是*组成*幸存批次的组成。正例轨迹（对简单提示的正确解决方案）收敛更快并被优先保留；较难的提示产生大多被丢弃的负例轨迹。结果：幸存过滤的样本系统地*更旧*于缓冲区中的平均 rollout，因为它们解决的简单提示在训练早期发出。一个名义上“新鲜”的 rollout 缓冲区可能包含跨越广泛策略版本的幸存正例。在批次级别跟踪陈旧性的准入控制（例如，SkyRL 的`max_staleness_steps`容量门（capacity gate），Atropos 的`max_batches_offpolicy`）无法检测这种批次内版本扩散。每样本版本标记（维度 4）在这种机制中不是可选的；训练器必须能够拒绝或 IS 校正（IS-correct）那些策略版本偏离太远的单个样本，即使它们所属的批次最近被准入。

无评论家方法简化了训练侧。但*评分*侧即将变得更加昂贵：过程奖励模型（process reward models）对中间推理步骤评分，而不仅仅是最终答案，这引入了全新的同步瓶颈。

### 5.2 过程奖励：一个新的同步屏障

结果奖励（outcome reward）是标量且廉价，在 rollout 结束时调用一次验证器。过程奖励模型（PRMs）对中间步骤评分，这需要（a）对完整推理轨迹进行单独的 PRM 前向传递，或（b）在生成期间逐令牌计算的在线效用函数（online utility function）。

**PRPO**（熵峰值分割（entropy-spike segmentation）与每段 PRM 评分）和 **DEEP-GRPO**（通过在线效用函数识别枢纽（pivot identification））都会产生计算开销*在生成和训练之间*。在当前库生态系统中，这个阶段尴尬地映射到预处理器池（PipelineRL）或需要额外的 Ray actor（verl, NeMo-RL）。两者都不是为此设计的。

**关键含义：**基于 PRM 的信用分配打破了奖励计算廉价的假设。对来自 7B 模型的 32K 令牌推理轨迹进行 PRM 前向传递可能非常昂贵。在 G=8 个完成（completion）每提示下，奖励计算可能消耗相对于生成本身不可忽略的挂钟时间。两个后果：

1. **异步奖励管道变得必要。**PRIME-RL 将奖励评分作为其完全异步编排器-训练器（Orchestrator-Trainer）管道的一部分与训练并发运行；编排器处理评分，而训练器独立执行反向传播和优化器步骤。对于基于 PRM 的方法，这种流水线奖励计算不是可选的；同步奖励评分将主导训练挂钟时间。
1. **单独的预处理器池变得必要**。在专用 GPU 层上运行参考对数概率（reference logprobs）计算和 PRM 评分，例如，在生成和训练之间流水线化，是密集信用分配的正确架构。

**DEEP-GRPO 的枢纽重采样**引入了第三代模式，与标准 rollout 和部分 rollout 恢复并列：_从序列中间状态进行局部重采样_。这需要在枢纽点保存 KV 缓存状态，而**当前没有异步库开箱即用地支持**。枢纽边界处的权重同步可能是一个新的正确性要求：如果权重在枢纽生成和局部重采样之间改变，优势估计会被破坏。当然，我们可以在单个预填充（prefill）中重新计算 KV 缓存，但这可能浪费我们训练中宝贵的计算资源。

### 5.3 多智能体协同进化：掉队者问题加剧

单智能体 GRPO 训练一个策略，每个提示生成 G 个完成。新兴的多智能体自博弈（multi-agent self-play）意味着有效的“组”跨越顺序链接的多个模型调用。奖励仅在链中所有模型完成后可用。

**掉队者（straggler）动态发生质变。**在单智能体 GRPO 中，掉队者是组中最长的完成，是单峰长度分布中的尾部事件。在多智能体管道中，掉队者是*乘积*两个或更多长度分布。在提议者/求解器（Proposer/Solver）多智能体架构中，如果每个都有第 90 百分位完成时间（5 倍中位数），联合第 90 百分位大约是 25 倍中位数。

**智能体群（swarms of agents）上的 RL 意味着新的工作单元。**今天，每个库中的原子单元是单个（提示，完成，奖励）三元组。在多智能体训练中，原子单元变成*情节*，一个回合、工具调用和智能体间消息的有向图。缓冲区设计、陈旧性跟踪和优势计算都需要在情节上操作。重放或分叉情节也可能变得必要。

当模型至少内部一致时，跨智能体的掉队者问题已经够糟糕了。在 MoE 架构中，即使是单个模型也可能在推理和训练框架之间自相矛盾，这引发了强化学习训练中一系列新的涌现问题。

### 5.4 训练-推理不匹配：Deepseek v3.2 MoE 案例研究

训练-推理不匹配问题在异步强化学习中普遍存在；每当在策略π∗old 下生成展开数据并在π∗θ下计算梯度更新时，这两个策略就会产生分歧。大多数库通过重要性采样校正或硬版本拒绝来解决此问题。但 DeepSeek-V3.2 的生产经验揭示了两个**结构性**的不匹配来源，这是重要性采样校正无法修复的。

**来源 1：MoE 专家路由不一致。**混合专家模型为每个令牌激活稀疏的专家子集。推理框架（vLLM、SGLang）和训练框架（Megatron、FSDP）独立实现路由器，门控函数中浮点数舍入的差异可能导致*对相同输入选择不同的专家*。当专家路由出现分歧时，活动参数子空间会发生不连续偏移；假设专家 A 处于活动状态计算的梯度步长被应用于在专家 B 下处于活动状态的权重。DeepSeek-V3.2 发现这“导致活动参数子空间的突然偏移，从而破坏优化稳定性并加剧离策略问题。”

他们的解决方案，**保持路由**，保留了采样（推理）期间使用的确切专家路由路径，并在训练前向传播中强制执行这些路径。这要求推理框架记录并返回路由决策以及令牌对数概率，训练框架接受并强制执行它们。目前没有开源异步强化学习库实现此功能。对于任何训练 MoE 模型的团队（DeepSeek-V3 类、Mixtral、未来的开源 MoE），这是一个正确性问题，而非性能问题。

**来源 2：采样截断掩码不匹配。** Top-p 和 top-k 采样在生成时截断词汇表，将低概率令牌排除在采样分布之外。在训练期间，完整词汇表对π∗θ可见。这违反了重要性采样恒等式：π∗old（截断）和π∗θ（完整）的动作空间不同，因此对于采样期间被掩码的令牌，重要性采样比π∗θ(o∗t)/π∗old(ot)是未定义的。

DeepSeek-V3.2 的**保持采样掩码**解决方案：在采样期间记录截断掩码，并在训练前向传播中将其应用于π_θ，使两个策略在相同的词汇表子集上操作。这需要将掩码从推理服务器传递回训练器，这同样是当前库基础设施不支持的功能。

**对库设计的影响：**保持路由和保持采样掩码都要求推理服务器返回*额外元数据*，包括令牌对数概率、路由决策和采样掩码。当前推理服务器（vLLM、SGLang）与训练器之间的 API 合约是`(token_ids, logprobs, finish_reason)`。将其扩展为`(token_ids, logprobs, finish_reason, expert_routing, sampling_mask)`是对每个库数据流的破坏性更改。

### 5.5 蒸馏：同一异步问题的不同名称

在策略蒸馏中，学生模型生成序列，教师模型用令牌级对数概率为其评分，这在结构上与 GRPO 中的异步协调问题相同。

本调查中的每个设计轴，包括展开缓冲区、权重同步协议、陈旧性管理和部分展开处理，都同样适用于蒸馏。生成池产生学生展开，教师为其评分（替换验证器），训练器使用优势修正的 GRPO 损失或独立的 KL 目标计算反向传播。自蒸馏增加了一个额外的协调要求：教师是学生从步骤 _N−k_ 的冻结快照，因此系统必须定期检查点策略并在不中断流水线的情况下热交换教师服务器，这是一个目前没有库完全自动化的原语。

**对库设计的实际影响是，异步强化学习基础设施不应构建为 GRPO 特定系统**。生成-评分-训练流水线是一个通用模式，涵盖具有结果奖励的强化学习、具有过程奖励的强化学习、在策略蒸馏和自蒸馏。像 **SLIME、MILES、PRIME-RL、AReaL 和 NeMo-RL** 这样的库已经支持 GRPO 和在策略蒸馏，正是因为它们的异步脚手架将奖励/评分阶段视为可插拔组件而非硬编码的验证器调用。任何追求通用性的异步训练器都应遵循相同原则：将评分阶段定义为接口（HTTP 端点、Ray actor 或共置前向传播），并让缓冲区、陈旧性和权重同步机制无论填充内容如何都相同地运行。

---

## 6. TRL 异步训练器的设计选择

在全面审视了编排模型、缓冲区设计、权重同步协议、陈旧性策略和部分展开处理后，我们现在可以为 TRL 中的异步训练器制定具体的设计选择，以及我们打算探索的未来发展方向。

### 设计原则：保持编排轻量级

当前 TRL 实现的一个优势是它不依赖重型编排器系统来管理训练生命周期。库内的数据保持为原生 Python 对象，没有外部库的着色。我们希望保持这一点：编排应尽可能简单，不依赖重型外部框架。

### 1. 带每令牌的有界队列（无双缓冲）

与其从双缓冲开始再升级到更细粒度的方案，我们直接采用**一个有界队列，其中每个令牌都带有产生它的`model_version`的标签**。这是从一开始就具备的最低可能粒度；它支持令牌级别的重要性采样校正，支持简单的准入门控（丢弃或降低超过陈旧度阈值的令牌的权重），并避免了后期将令牌级来源追溯适配到批次级缓冲区所带来的架构债务。

### 2. 使用打包传输的 NCCL 权重同步

NCCL 进程组是必需的，并且我们已经在使用它们。添加分桶应该是下一步，因为 vLLM 的[`NCCLWeightTransferEngine`](https://github.com/vllm-project/vllm/blob/f3c6c9c9d794fac5e74b59bc75da6e9d1921eeac/vllm/distributed/weight_transfer/nccl_engine.py)与`packed=True`直接支持分桶广播：它将参数打包到可配置大小的`uint8`缓冲区（默认为 1 GB，在 CUDA 流之间双缓冲），并通过一个独立于训练进程组的专用 NCCL 通信器进行广播。这消除了在朴素广播中占主导地位的每个参数调用开销，从而实现了巨大的同步加速。

除了 vLLM 内置的引擎，我们将探索用于更苛刻场景的高性能权重打包库：

- **[Awex](https://github.com/inclusionAI/asystem-awex)**（inclusionAI），一个专为 RL 训练设计的权重同步框架，处理跨引擎传输的难题：训练引擎（Megatron、DeepSpeed）和推理引擎（SGLang、vLLM）使用完全不同的并行策略和张量布局。Awex 通过统一的转换层和确定性的 P2P 传输计划来抽象这一点。它支持分离 GPU 和共置（CUDA IPC）两种模式。

- **[Mooncake 传输引擎](https://github.com/kvcache-ai/Mooncake)**，SGLang 已朝着集成 Mooncake 传输引擎作为其高性能传输层的方向发展，集成范围涵盖 PD 解聚、分层 KV 缓存和弹性专家并行。具体针对权重同步，配套的**[checkpoint-engine](https://github.com/MoonshotAI/checkpoint-engine)**项目使用 Mooncake 的 RDMA 支持的 P2P 传输来更新万亿参数模型（Kimi-K2，256×H20 GPU），耗时约 16-17 秒。Mooncake 现在是 PyTorch 生态系统的一部分，并作为[NVIDIA 的 NIXL 传输库](https://github.com/ai-dynamo/nixl)的后端插件。

### 3. 对智能体工作负载的部分推出支持

复杂环境中的多轮工具使用任务每次推出可能需要几分钟。如果没有在权重更新期间处理进行中推出的机制，同步窗口就会成为流水线瓶颈。我们可能会实验性地探索两种策略：

- **前缀恢复**：当权重在推出中途更新时，保存 KV 缓存前缀，并在新策略下从检查点恢复生成。这保留了部分工作，但需要推理引擎支持中途权重交换。
- **中止并重试**：丢弃超过陈旧度阈值的进行中推出，并重新排队提示。实现更简单，但浪费的计算量与中止时平均推出长度成正比。

这就是路线图，请保持关注，我们正在 TRL 中开发一个具体的异步 GRPO 训练器，并将很快宣布 🧑🍳！
