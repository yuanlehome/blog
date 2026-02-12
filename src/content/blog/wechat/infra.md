---
title: 图解Infra视角下的强化学习性能优化
slug: infra
date: '2026-02-12'
tags: []
status: published
source_url: 'https://mp.weixin.qq.com/s/onW4d3e-SCgO_z_16JgWEA'
source_author: InfraTech
imported_at: '2026-02-12T02:21:46.192Z'
source:
  title: mp.weixin.qq.com
  url: 'https://mp.weixin.qq.com/s/onW4d3e-SCgO_z_16JgWEA'
cover: /images/wechat/infra/001-681c5816.gif
---

![图片](/images/wechat/infra/001-681c5816.gif)

点击**蓝字**，关注我们

LLM强化学习(RL)训练过程的性能与资源效率是AI Infra重点关注的内容。RL的场景比预训练(或者SFT)的工作流更长、协同工作的模块更多，容易出现资源利用率低的情况。Infra的目标是在保证算法精度/效果满足前提下，应尽量去提升资源效率。本篇简单介绍RL训练中可能遇到的问题，并列举几个业界的解决方案。

1

**&#xA0;训练的步骤**

强化学习常见算法步骤\[1]：

![图片](/images/wechat/infra/002-4dadebfb.png)

梳理出三个层\[2]：1生成(Generation)、2准备/推理(Preparation/Inference)、3训练(Training)。

![图片](/images/wechat/infra/003-e13cf52a.png)

源自HybridFlow图片

到AI Infra关注的训练步骤大致是这样的：

![图片](/images/wechat/infra/004-2009976c.png)

上图包含了：

- 生成模块(Generator)：加载Policy模型运行推理任务，负责流程中rollout阶段，硬件资源一般是GPU。

- 奖励模块(Reward Server)：加载评价模型，模型参数固定，硬件资源CPU。

- 训练模块(Trainer)：训练Policy模型，运行参考模型。硬件资源GPU。

大致有6个处理步骤：

1. 提示词(Prompts)，Trainer生成提示词给到Generator，数据大小n；

1. 答案(Answers)：每一个prompt生成m个数据，输出数据n x m；

1. 奖励(Rewards)：对n x m数据进行打分，得到奖励值rewards;

1. 损失(Loss)：计算模型损失，函数可以是grpo、ppo等算法。

1. 梯度(Gradient)：计算模型梯度，更新模型权重。

1. 同步(Sync)：将Trainer的权重同步到Generator中。

注意：此例的算法仅考虑了Reward模型，若选取的算法改变，步骤和资源分配也将发生改变。

![图片](/images/wechat/infra/005-ee332fa9.png)

PPO算法，考虑Critic、Reference模型

2

**存在问题/挑战**

**_2.1 权重同步_**

训练与推理权重的传递理想中是这样的：

![图片](/images/wechat/infra/006-89447b7f.png)

实际上可能是这样的：

![图片](/images/wechat/infra/007-afe8d205.png)

存在问题:

- 训推共卡(task-collocated)方案，由于显存不足，权重需要卸载到CPU上；

- 训推分离(task-separated/disaggregated)方案，1、分布式并行方案不一致让权重size不同；2、跨节点带宽小，多实例传输共用一条链路；

- 权重同步产生空泡(bubble)：

![图片](/images/wechat/infra/008-64a95083.png)

- 精度差异：训练与推理的权重采用的精度格式不同。

除了上述提到的多实例传输链路带宽共用的情况外，还会出现多种实例链路并非一致情况：

![图片](/images/wechat/infra/009-f7bc4dc2.png)

产生这种问题的可能原因：

- 不是所有训、推实例都在同一个NCCL网络平面下，跨节点之间仅支持IB/RoCE或者TCP平面传输；

- 训练、推理的NCCL版本不一致；

- 系统要求实例支持扩缩容，扩缩容导致实例之间的网络建链失败(一种系统BUG)。

**_2.2 OnPolicy的挑战_**

1、Rollout的长尾/拖尾(Long-Tail)：

![图片](/images/wechat/infra/010-07badeac.png)

Rollout阶段生成数据的结束时机不一致，存在短板效应。耗时占比可以超过70%。参考数据\[3]

![图片](/images/wechat/infra/011-06e2f21e.png)

2、序列变化。序列长度随着训练进行会波动变化。 当生成序列保持相对平稳时，如下图(I)所示对推理rollout的资源要求也是平稳的，但可能会出现(II)所示的情况\[4] ，rollout所需资源也得相应地提升。

![图片](/images/wechat/infra/012-dec06897.png)

3

**&#xA0;性能提速方案**

**_3.1 负载调整_**

根据Rollout的负载情况调整负载加载顺序，算法依然是On policy，无精度损失。

负载均衡器。通过一个动态负载均衡器实现DPLB(Data Parallel Load Balance)，这个方法发挥出优势的条件：请求数量比较多（>>generator数量），没有极端长度。

![图片](/images/wechat/infra/013-e32cd27e.png)

DPLB给Generator1调度请求

我们团队尝试了该方法，在一定条件下提速收益可达\~30%。

Tail Batching\[5]。根据生成长度组batch，当batch里面出现了过长的数据，打断后放入队列，寻找相同长度数据组batch重新调度。

![图片](/images/wechat/infra/014-65ee364c.png)

Tail Batching (参看论文4)

步骤融合。将rollout与后续的模型操作(例如PPO算法的Reward/Reference/Critic)进行融合。如RLHFuse提到的方案\[6]：rollout阶段将零散的长尾数据打断后集中到部分GPU资源上，把Ref.Infer、Crit.Infer、Rew\.Infer运算调度到rollout阶段释放出的资源中。使得Generate与Inference从串行，变为有部分重叠，从而提升效率。

![图片](/images/wechat/infra/015-d06149a5.png)

_3.2 异步训练_

异步强化学习(Asynchronous Reinforcement)，off-policy方式，本轮训练的数据可能来自上一轮/上几轮的权重的生成。 需要调整算法保证精度。

KV cache保持。在Magistral\[7]中提到了KV cache保持的方式，其原理：当遇到过长的生成时，打断生成，但不释放被打断数据的KV cache。下一轮权重更新后，继续上一轮的生成。单个序列数据可能混合了几个不同权重推理出来的结果。

![图片](/images/wechat/infra/016-0de9b5bc.png)

KV cache重算。rollout持续生成，当需要更新权重时才停止，并打断当前数据生成，未完成的数据下一轮重新推理，已生成的KV cache重新计算，如 AREAL\[8]的异步流方法。

![图片](/images/wechat/infra/017-a67d624e.png)

长序列分段处理。在Kimi K1.5\[9]中采用的策略是Partial Rollout，长序列会分段。大概原理：设置推理的最长tokens阈值，在一轮rollout中当一个序列计算tokens数超过该阈值时，该序列会被打断，并存入一个“replay buffer”中。在后面的rollout过程中，当有序列推理结束，出现空闲资源时，从replay buffer中拿出未计算完的数据继续推理。这个方式不涉及KV cache保存/重算。 为什么是“partial”？因为短序列不受影响。

![图片](/images/wechat/infra/018-6c2f5f88.png)

_3.3 权重同步_

**3.3.1 分离方案的权重同步**

异步权重更新。允许policy model训、推之间权重异步更新，如AsyncFlow\[10]中提到方式，权重从actor的train到rollout传递更新，支持分实例异步操作。rollout实例与实例之间运行的权重可以不一致。

![图片](/images/wechat/infra/019-7d912eee.png)

AsyncFlow

统一的训推权重转换。面对训、推框架不同的并行策略、精度、格式，构建统一的转换层。如asystem-awex\[11]中提到的，在训练中配置权重写入器(WeightWriter)、推理中配置权重读取器(WeightReader)，实现权重分片存储、传输和转换。

![图片](/images/wechat/infra/020-1685bc8a.png)

不同网络用不同传输模式。训练和推理的实例可能位于相同节点、也可能位于不同节点，数据传输的网络不一样需要采用不同的传输模式。比如awex提到的训、推之间先用CUDA IPC的zero-copy操作，然后推理实例之间进行P2P同步如下图所示。

![图片](/images/wechat/infra/021-d63b171d.png)

_3.3.2 共卡方案的权重同步：_

共享内存传递权重。如K1.5的megatron与vLLM的混合框架中，用一个通用模块Checkpoint Engine完成权重传输(解决不同ProcessGroup之间互通的问题)。权重的分片注册到一个公共位置，训练或者推理步骤结束时释放掉一些显存。训推引擎在一个pod中，共享GPU的资源。两个引擎中的TP切分策略存在差异，在加载时完成转换。

![图片](/images/wechat/infra/022-11dc8fdd.png)

相比分离方案，共享权重方案的H2D会降低传输速度，需要做一定优化处理\[12]。

**_3.4 Rollout提速_**

Rollout性能取决于推理的运算速度。要降低E2EL，其中ITL是大头。推理框架的下发效率、特性，算子等对速度都能构成影响。目前主流的两款框架vLLM、SGLang各有优劣，所以在RL架构中两种框架都能见到。

![图片](/images/wechat/infra/023-c85a0478.png)

提速特性：如投机推理（Speculative Decoding）参看\[13]，可以一次生成多个数据，对于一些短输入长输出的场景(prefill 2k -> decode 20k)，能带来较大性能提升。

![图片](/images/wechat/infra/024-291f3b11.png)

量化：在rollout优化的相关讨论\[14]中，提到了一种量化的方式"FlashRL: 8Bit Rollouts"\[15].

![图片](/images/wechat/infra/025-d9e289f0.png)

这种可能对精度产生影响的方案，一般不会由Infra单方发起，需要与算法共同讨论。

框架下发速度：是指框架的调度器(scheduler)将任务从CPU侧下发到GPU侧的速度。，这个是一个通用优化点，既能用于rollout，也能用于推理部署。比如在vLLM中的多进程协同（scheduler与worker在不同进程），multi-step\[16]等特性。

![图片](/images/wechat/infra/026-5ae9af37.png)

muti step

小结：与预训练动辄千卡/万卡规模相比，RL的资源使用量比较少，目前RL性能问题还没有成为AI算法迭代的关键瓶颈。在Infra的性能优化工作项中，RL性能优化工作一般排在预训练性能、推理性能之后。随着大家开始尝试扩大(scale up)RL规模，寻找更好的训练效果，后续RL的性能关注度应该会逐步提升。

本文作者：kaiyuan 主页：<https://www.zhihu.com/people/xky7>

想深耕AI Infra领域？欢迎访问InfraTech库！内容涵盖大模型基础、PyTorch/vLLM/SGLang框架入门、性能加速等核心方向，配套50+知识干货及适合初学者的notebook练习：**<https://github.com/CalvinXKY/InfraTech>**

---

参考:

- \[1]<https://arxiv.org/pdf/2402.03300>
- \[2]HybridFlow <https://arxiv.org/pdf/2409.19256v2>
- \[3]<https://arxiv.org/pdf/2509.21009>
- \[4]<https://arxiv.org/pdf/2506.10910>
- \[5]TailBatching <https://arxiv.org/pdf/2509.21009>
- \[6]<https://arxiv.org/html/2409.13221v1>
- \[7]<https://arxiv.org/pdf/2506.10910>
- \[8]<https://arxiv.org/pdf/2505.24298>
- \[9]<https://arxiv.org/pdf/2501.12599>
- \[10]<https://arxiv.org/abs/2507.01663>
- \[11]<https://github.com/inclusionAI/asystem-awex/tree/main/docs>
- \[12]<https://zhuanlan.zhihu.com/p/1949882680167621566>
- \[13]<https://zhuanlan.zhihu.com/p/1978037808544370747>
- \[14]<https://www.zhihu.com/question/1957780654780551831/answer/1959936433407098998>
- \[15]<https://fengyao.notion.site/flash-rl>
- \[16]<https://blog.vllm.ai/2024/09/05/perf-update.html>

扫码关注我们，了解更多AI Infra基础知识。

![图片](/images/wechat/infra/027-e7f95258.jpg)
