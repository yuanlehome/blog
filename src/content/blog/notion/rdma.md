---
title: RDMA 在大模型推理框架中的应用
slug: rdma
date: '2025-12-25'
tags: []
status: published
cover: ''
notionId: 2d022dca-4210-8002-8300-ff0fc62fb73a
lastEditedTime: '2025-12-25T18:13:00.000Z'
---

## [极简] RDMA 在大模型推理框架中的应用

---


在现代大规模 LLM 推理系统中，**多 GPU / 多节点推理已成为必然**。无论是模型本身的体积（70B/405B/1T+）还是长上下文（128k~1M tokens），都远超单节点能力。


为保证吞吐和延迟，推理框架需要极为高效的跨 GPU / 跨节点通信，而 **RDMA（Remote Direct Memory Access）** 由于具备：

- **GPU → GPU 的零拷贝（GPUDirect RDMA）**
- **绕过 CPU 和内核协议栈（Kernel Bypass）**
- **低延迟（μs）**
- **高带宽（200–400+ Gbps）**

因此成为分布式推理通信的核心。


---


## **1. 张量并行（Tensor Parallel, TP）**


### **场景描述**


大模型的 Attention、MLP 权重沿隐藏维拆分到多个 GPU 上。计算每层时，需要跨 GPU 对局部结果求和或拼接，执行：

- **All-Reduce**
- **All-Gather**

这些操作在 **每层的每个 forward（prefill/decode）** 都会触发。


### **为什么依赖 RDMA？**

- **极高的调用频率**：一个 80 层模型、1 token decode → 150~200+ 次 All-Reduce。
- **对延迟极敏感**：任何一次通信阻塞都会显著提高 token latency。
- **显存到显存直连（GPUDirect RDMA）** 让数据无需经过 CPU 内存，延迟从 ms → μs。

### **框架内部实现**


通过 PyTorch / NCCL 自动选择 RDMA 通道。


如果是多节点，NCCL 会自动切换到 InfiniBand/RoCEv2 通道。


---


## **2. Prefill / Decode 解耦（Disaggregated Serving）中的 KV Cache 传输**


### **场景描述**


Prefill（prompt 计算）与 Decode（逐 token 生成）具有不同的计算特性，因此分配到不同 GPU 组甚至不同节点：

- Prefill 节点：计算 Attention、生成 KV Cache（可能上 GB）
- Decode 节点：需要拉取这些 KV Cache，开始流式生成

### **通信内容**

- KV Cache（头数 × 层数 × 序列长度 × head dim）

    128k context → **数 GB 到几十 GB**。


### **为何必须 RDMA？**

- **大量突发数据**：TCP/IP 会因 CPU copy + 内核开销而变成瓶颈。
- **RDMA 带宽可达 200–400+ Gbps**，能保证 KV Cache 在毫秒级传输。
- **Decode 延迟（TTFT, First Token Latency）直接受其影响**。

### **框架内部实现**

- vLLM: 仍主要 DP/TP，不强调 KV 迁移
- **SGLang: 提供了高效 KV Cache Migration，RDMA 是关键技术基础**

---


## **3. Mixture-of-Experts (MoE) 模型中的 All-to-All 通信**


### **场景描述**


每个 token 会被路由到 k 个不同的 Expert（通常 top-k=2~4）。


专家分布在不同 GPU / 不同节点上，就会发生：

- **token → expert 的路由**
- **expert → token 的聚合**

### **通信模式**

- **All-to-All（DeepEP）**，并且是高度稀疏、动态、分布不均的通信。

### **为什么 RDMA 必需？**

- MoE 的通信量巨大（特别是大 batch）
- 动态负载导致流量不均匀，对带宽和延迟要求极高
- RDMA 能处理不连续显存地址和高速大流量并发传输

---


## **4. 流水线并行（Pipeline Parallel, PP）中的跨节点中间激活传输**


### **场景描述**


模型按层切分：

- Node A 的 GPU0：Layer 0~14
- Node B 的 GPU4：Layer 15~30

    每层之间需要传递 activations。


### **通信原语**

- **Send / Recv（点对点通信）**

### **为何 RDMA 必须？**

- 激活大小通常是 MB 级
- 推理是串行执行，通信延迟会直接增加 token 延迟
- RDMA Send/Recv 显著降低 bottleneck，避免 pipeline bubble

---


## **5. 模型加载（带宽敏感但不高频）**


### **模型权重加载（数百 GB）**


在冷启动、滚动升级、auto-scaling 时：

- 使用 RDMA 加速权重分发
- 或配合 NVMe-oF 进行远程 SSD 访问

---


## **场景总览表**


| 场景                    | 通信模式                    | 数据对象               | 带宽需求             | 延迟敏感度  | RDMA 使用程度 | 备注                   |
| --------------------- | ----------------------- | ------------------ | ---------------- | ------ | --------- | -------------------- |
| **Tensor Parallel**   | All-Reduce / All-Gather | Attention/MLP 局部结果 | 中–高              | **极高** | ★★★★★     | 推理中最核心               |
| **Prefill–Decode 解耦** | P2P                     | KV Cache           | **极高（GB~几十 GB）** | 高      | ★★★★★     | SGLang 重度依赖          |
| **MoE 路由**            | All-to-All              | Token→Expert 数据    | 高                | 高      | ★★★★★     | DeepSeek/Mixtral 等模型 |
| **Pipeline Parallel** | Send/Recv               | Activations        | 中                | 中–高    | ★★★★☆     | 超大模型必要               |
| **模型权重加载**            | RDMA/NVMe-oF            | Weights            | 极高               | 低      | ★★★★☆     | 加速冷启动                |


---


## 一、RDMA 技术概述


**RDMA（Remote Direct Memory Access）**是一种在网络通信中绕过操作系统内核、使一台计算机直接访问另一台计算机内存的技术。通过硬件支持，RDMA 将网络数据传输的大部分工作从 CPU 卸载给 NIC（网络接口卡/ HCA），实现**低延迟**、**高吞吐**和**低 CPU 占用**的数据交换。在传统 TCP/IP 收发过程中，数据需多次在用户态和内核态之间拷贝，CPU 参与度高；而使用 RDMA 时，发送/接收操作由 NIC 直接把数据从本地内存拷贝到远端内存，不需要远端 CPU 处理，极大减少了拷贝和上下文切换开销。


### 1.1 RDMA通信原语


分为“双边 (two-sided)”和“单边 (one-sided)”两类：

- **双边 RDMA**：需要通信双方协同。典型操作是 Send/Receive ——发送端通过 QP 的发送队列发出 Send 请求，对端必须事先在接收队列 Post 一个 Receive 请求来准备缓冲区，否则发送无法完成。这类似传统 socket 的收发模式，但由 NIC 硬件执行传输和通知。在 RDMA 中 Send/Recv 提供了消息语义，适合 RPC 等需要显式握手的场景。其优点是控制灵活，应用可以在 Receive 端知道每条消息边界；但需要远端参与，增加了一次 CPU 处理和同步。
- **单边 RDMA**：又称“one-sided”操作，包括 RDMA Write（写）和 RDMA Read（读），由一端直接对另一端内存执行读写，**不需要远端 CPU 参与**。发起单边操作的一方必须知道远端目标内存地址（virtual address）和远端内存注册后得到的访问密钥（rkey），这样 NIC 才能定位并读写远端内存。例如，RDMA Write 由客户端直接将本地缓冲数据推送到远端机器指定的内存地址；RDMA Read 则让客户端从远端内存拉取数据到本地缓冲区。整个过程远端应用无感知，数据在两端 NIC 和内存之间零拷贝传输。单边操作无需配对的接收动作，**非常适合批量数据传输**，能够最大化吞吐和隐藏通信延迟。不过需要预先通过其他渠道交换远端地址和 rkey 等元数据（通常可以用一次 Send/Recv 握手完成）。

### 1.2 内存注册（Memory Registration, MR）


为了使 NIC 直接访问内存，需要将用户空间的缓冲区“注册”给 RDMA 驱动锁定在物理内存中，得到本地密钥 lkey 和远端密钥 rkey。只有注册后的内存（称为 MR）才能被 RDMA 操作访问。注册时可指定访问权限，比如本地写、远程读/写、原子操作等。内存注册有一定开销，通常应该在初始化时完成、复用已注册的区域，以避免运行中频繁注册/注销导致延迟。


### 1.3 队列对（Queue Pair, QP）


QP 是一对收发队列（发送队列 SQ 和接收队列 RQ）的抽象。应用在 QP 的 SQ 中贴工作请求（Work Request）执行 Send、Write 等操作，在 RQ 中贴 Recv 请求用于接收远端的 Send 数据。每个 QP 可配置不同的**传输类型**：最常用的是 **RC**（Reliable Connection，可靠连接），提供有序可靠传输，要求通信双方建立连接并维护消息顺序；另一类是 **UD**（Unreliable Datagram，不可靠数据报），无需逐一连接，可以根据地址自由发送，但不保证顺序且可能丢包。RC 适合需要可靠有序的大数据传输（如参数、KV 缓存），UD/SRD 适合需要灵活组网、可容忍乱序的场景（如弹性多播、动态集群通信）。QP 创建后需交换 QP 号、LID/IP 等信息并变更状态到 RTS（Ready to Send）才能通信。大量并发连接时 QP 资源管理也很重要，每个 QP 消耗 NIC 内存，工程上可能使用 QP 池或 QP 共享等优化。


### 1.4 完成队列（Completion Queue, CQ）


RDMA 操作完成后，NIC 会将完成信息（CQE）写入关联的 CQ。应用通过轮询CQ或者异步事件通知来获取操作完成情况（例如判断 RDMA Write 是否传输完毕）。为了追求极致性能，RDMA 应用常采用**用户态轮询** CQ 而非中断，以避免内核中断延迟和抖动。


### 1.5 保护域（Protection Domain, PD）


PD 是一种资源隔离机制，所有的 QP、MR、CQ 等都隶属于某个 PD。只有属于同一 PD 的 QP 才能访问同一 PD 下注册的 MR。这相当于给应用划分了隔离区域，防止未授权的内存访问。一般一个进程创建一个 PD 即可。需要注意的是，本节所述 PD 是 RDMA 的保护域概念，不要与后文推理框架中的“Prefill/Decode 分离 (PD)”混淆。


### 1.6 GPUDirect RDMA


这是 NVIDIA 提供的一项技术，使支持 GPUDirect 的 NIC 可以直接读写 GPU 显存而无需经过 CPU 内存中转。RDMA 协议本身不关心内存是在 CPU 还是 GPU，只要内存注册提供了物理地址，NIC 即可 DMA 访问。但普通 NIC 不识别 GPU 地址空间，需要 GPUDirect RDMA 支持才能将 GPU 内存注册为 RDMA MR。GPUDirect RDMA 极大加速了 GPU 间跨节点通信：NIC 直接和 GPU 内存交换数据，避免宝贵的 GPU 数据先拷贝到 CPU 内存再发网卡。现代 NVIDIA ConnectX 系 NIC 普遍支持 GPUDirect RDMA，AMD 的 ROCEnet 卡也有类似功能。值得注意的是，**GPUDirect RDMA** 并非新的网络协议，而是 PCIe 层面的数据路径优化，与 InfiniBand 或 RoCE 协议配合使用。此外，NVIDIA 还提出了 GPUDirect Async/GPUNetIO 等技术，允许 GPU 发起网络操作，将通信进一步从 CPU 卸载。例如 ConnectX 卡支持所谓 IB GPU Direct Async（IBGDA），GPU 可直接触发 RDMA 操作，但云厂商自研的 NIC（如 AWS EFA）通常不支持这类特性，需要依赖 CPU 代理发起，增加了一些延迟。工程上，如果 NIC 缺乏 GPU 直接发起功能，可以采用 **Host-Proxy** 模式：在 GPU 和 CPU 共享的内存区域放置标志，由 GPU 内核写入标志请求发送，CPU 轮询这些标志并代替 GPU 调用 RDMA 发送，实现 GPU 在 CUDA Graph 中间接触发网络 IO。这个技巧在 KV 缓存传输和 MoE 通信中已被证明可行，对一次通信数百微秒的场景来说，额外十几微秒 CPU 介入开销是可以接受的。


### 1.7 InfiniBand vs RoCE vs 以太网TCP


RDMA 所依赖的底层网络可以有不同实现。**InfiniBand (IB)** 是原生支持 RDMA 的高性能网络架构，采用专用交换机和 HCA 硬件，提供极高带宽和低端到端延迟，一般用于 HPC 和 AI 集群。**RoCE (RDMA over Converged Ethernet)** 则是在以太网设备上实现 RDMA 协议，目前常用的是 RoCEv2，通过 UDP/IP 封装 RDMA 包，使其能在标准以太网交换机上传输。RoCE 的优点是利用现有以太网基础设施，但要求数据中心网络配置成“无损”或近似无损（如启用 PFC 优先级流控、ECN 拥塞通知等），否则丢包会导致 RDMA 连接超时或性能剧降。IB 由于封闭环境，链路层自带信用流控和重传，往往能保障不丢包，因此 RDMA RC 在 IB 上运行很稳定；在 RoCE 上如果出现拥塞丢包，需要依赖 RoCE 自身的拥塞控制算法（如 DCQCN）来避免大量超时。**iWARP** 则是另一种老的方案（RDMA over TCP/IP），将 RDMA 协议映射到 TCP，优点是完全兼容现有 TCP 网络且无需无损网络，但实际实现中 iWARP 通常通过 NIC 芯片做 TCP Offload 才能减少 CPU 参与，否则纯软件 iWARP 几乎失去 RDMA 优势。总体而言，主流 AI 集群更青睐 IB 和 RoCE 方案：InfiniBand 性能卓越但成本高、硬件要求专门；RoCE 利用以太网设备更普遍，但需要精心配置数据中心网络来逼近 IB 效果。云厂商由于多租户和成本考虑，多采用 RoCE 风格的定制 RDMA 网（如 AWS EFA）。无论哪种，部署 RDMA 都需安装合适驱动/库（如Mellanox OFED）并配置交换机队列、PFC 等。在推理系统实践中，如果硬件具备 RDMA 能力，那么充分利用其**零拷贝、高带宽**特性来加速跨节点通信是性能优化的关键。


---


## 二、推理系统中的 RDMA 应用场景


大模型推理框架由于模型规模大、数据依赖重，常需要将计算和数据分布在多 GPU、多机器上协同完成，从而引入大量分布式通信需求。RDMA 技术在以下典型场景中被用来突破网络瓶颈、提升推理性能：


### 2.1 模型并行通信（张量并行 TP / 流水并行 PP / 数据并行 DP）


**模型并行**是指将一个模型的不同部分拆分到多卡执行，包括**张量并行（Tensor Parallelism）**和**流水线并行（Pipeline Parallelism）**等。推理时，如果模型层或权重被切分在多张 GPU 上，必须在每次推理前后通过网络交换中间结果。例如，张量并行需要跨节点对张量进行 All-Reduce 或 All-Gather 合并局部计算结果；流水并行则在各阶段间传递每个输入的激活值。这些通信频繁且对延迟敏感，直接关系到吞吐和延迟。RDMA 在此类场景的作用在于提供**高带宽、低延迟**的 GPU 到 GPU 数据通道，使并行开销降到最低。实际上，大多数训练/推理框架都会调用 NCCL 这类通信库实现 GPU 集群的 all-reduce、广播等操作。NCCL 在底层如果检测到 InfiniBand 网络，会使用 RDMA Verbs 传输以获取最佳性能，否则退化为 TCP 套接字传输。Facebook (Meta) 报道中提到，在 8 卡的 LLM 推理中，光是张量并行的 all-reduce 通信就可能贡献**高达 30% 的端到端延迟**。为此他们开发了“直接数据访问 (Direct Data Access, DDA)”算法：让每块 GPU 直接通过 RDMA 读取其他 GPU 的数据再汇总，避免常规环形算法那样串行等待，多张卡并行读取将 all-reduce 延迟从 O(N) 缩减为常数级。这种利用 **RDMA 一侧直接读取远端显存**的思路，使小消息 all-reduce 延迟显著降低，在 AMD MI300X 集群上甚至让 8 卡推理性能追平了 NVIDIA H100 集群。可见，即使已有高度优化的 NCCL 库，RDMA 直通访问仍有潜力进一步优化模型并行通信（特别是小批量、跨节点场景下）。


**数据并行**在推理中通常指多机复制相同模型，各自处理不同请求以扩展吞吐。这种模式下不同节点间很少需要实时通信（因为各自独立生成答案）。然而在某些情况下，仍可能需要跨节点同步数据：例如 **Ensemble** 推理需要汇总多模型输出，或在需要收集所有节点推理结果进行排序、投票时。此类汇总通常数据量不大，可以用一次 all-gather 或参数服务器收集。RDMA 的作用是在这些同步步骤中提供更低的延迟和 CPU 开销，使数据并行的规模扩展更平滑。值得一提的是，在**训练**场景数据并行会频繁 all-reduce 梯度，RDMA/InfiniBand 几乎是标配；但在**推理**场景数据并行通信量小得多，因此 RDMA 更多用于模型并行和其他复杂并行模式。


### 2.2 Expert Parallel（混合专家 MoE）通信


在 **Mixture-of-Experts (MoE)** 大模型中，有许多“专家网络”仅对部分 token 激活。这种模型的推理需要一个路由机制：每个 token 由门控路由到某几个专家执行前向计算。多个专家模块往往分布在不同的 GPU 甚至不同机器上，所以一次推理过程中，各节点需要进行**大规模的 all-to-all 通信**：将属于自己专家的 token 数据从各其他节点收集过来，同时把本节点不属于自己专家的 token 发送出去。此通信模式具有**强非均匀和动态**特点——不同 token 批次，不同专家之间的数据量差别很大，无法预先确定通信模式。传统做法可以用 MPI/NCCL 的 AlltoAll 来一次交换所有节点的数据，但 AlltoAll 接口通常要求各节点提供相同大小的缓冲（必须按最大可能量对齐），这在 MoE 场景下会导致大量空数据传输和低效。例如某实例中使用 DP=64、EP=64（64 路数据并行 ×64 个专家）时，如果用 collective AlltoAll 通信，需要为最坏情况预留 **64 倍**于平均的消息大小，几乎都是无效填充。因此，高效的做法是**点对点按需传输**：每个节点只向那些实际有 token 要给的专家节点发送数据，且发送的字节量正好等于这些 token 的表示大小。RDMA 非常适合实现这种**稀疏高效的数据路由**：通过单边 Write 或 Send/Recv，可以直接将 token 小批数据写入目标专家节点缓冲区，而无需按最大上限填充。DeepSeek 团队的 DeepEP 是一套专门优化 Expert Parallel 通信的开源库。据报道，在NVIDIA H800 GPU + ConnectX-7 400 Gb/s InfiniBand 网络环境下，DeepEP 的“normal 模式”专家通信可以达到每节点 **50~60 GB/s** 的跨机带宽，端到端延迟 100~200 微秒量级（以 8 到 32 个专家并行时测得）。而 DeepEP 的“低延迟模式”通过纯 RDMA 管道进一步降低了延迟，在 8 个专家时单次调度仅 77µs、带宽达 98 GB/s（不过随着专家数增加，每专家数据变少导致带宽利用率下降）。这些结果表明，针对 MoE 的通信模式定制 RDMA 方案能充分发挥网络带宽，同时将微秒级的调度和数据传输延迟控制在可接受范围。


MoE 通信的工程挑战还包括：**通信与计算重叠**（例如 DeepEP 引入了 Hook 机制，使 RDMA 传输在 GPU 计算同时异步进行，不占用 SM 资源），以及**容错和弹性**（当某个专家节点故障时如何快速重路由 token）。Mooncake 平台为此提供了**弹性专家并行**支持：自动检测故障 GPU 并与其搭配的负载均衡模块（EPLB）协作，将 tokens 动态转发到健康专家上。这套机制背后依赖 RDMA 网络的快速通知和重连能力，使在推理过程中替换专家节点成为可能，从而提高服务稳定性。


### 2.3 KV 缓存跨机传输


**KV 缓存（Key-Value Cache）**是指 Transformer 模型在生成过程中的注意力键和值张量缓存。每当模型生成新 token 时，需要将该 token 的 Key/Value 添加到缓存中，并在后续解码时用于计算注意力。对于长上下文或多轮对话，KV 缓存会变得非常大（与生成的 token 数线性相关）。例如 LLaMA3-70B 模型若支持 128k 长序列，其 KV 缓存总大小可达约 40 GB。当推理框架需要在**多机器之间共享或移动 KV 缓存**时，如何快速传输这几十 GB 数据成为瓶颈。例如在 Prefill/Decode 分离架构下，一个“Prefill 节点”算完大段 Prompt 的 KV 后，需要把这些 KV 发送给负责续写的“Decode 节点”；又如在**上下文并行**的实现中，不同节点各自保存一部分 KV，需要在注意力计算时交换 KV 片段。RDMA 为此提供了关键支持——利用**零拷贝远程内存写入**能力，将大容量 KV 从一台机器的显存直接搬运到另一台机器的显存/内存。相比传统 TCP 传输，RDMA 避免了多重拷贝和上下文切换，能充分利用硬件带宽，使如此大的 KV 数据在毫秒级内完成传输。Mooncake 项目的 Transfer Engine 对这一场景进行了高度优化：通过**多 NIC 并行**和**批量 Write** 等技术，在4×200 Gbps RoCE 网络下实测传输 40 GB 数据的速度达到 **87 GB/s**，8×400 Gbps 网络下更提升至 **190 GB/s**（比传统 TCP 提升了 2.4× 到 4.6×）。这意味着一份 40GB 的 KV 缓存在后者环境下约 0.21 秒即可传完，满足长上下文推理的实时性需求。


在 KV 缓存传输中，有几个工程难点：其一，KV 通常由分页内存管理（如 PagedAttention）产生，物理上往往分散在不连续的小块中。NCCL 等通信库倾向于处理**连续缓冲区**，面对碎片化内存会触发大量小消息传输和频繁的 CUDA kernel 启动，进而增加延迟。为此，一些优化策略包括：在发送前将分散的 KV 块做**内存整理**（例如 FlowKV 框架将每层的多个小张量合并为一个大张量，以成倍减少 NCCL 调用次数）；或者利用 RDMA Verbs 支持 **SG list**（Scatter-Gather 列表）的特性，一次性 post 包含多个非连续段的发送请求，由 NIC 自行聚合。Mooncake 的传输库提供了高级接口，允许上层直接请求“发送一个由多个不连续内存块组成的数据对象（如 KV pages）”，库内部会缓存元数据并优化传输过程，比让应用逐块发更加高效。其二，为降低对计算的干扰，常会将 KV 发送放在 CUDA 流以外的异步线程上，并利用 GPUDirect RDMA 使数据直接从 GPU 发出而不消耗 CPU 拷贝。这样 Prefill 节点在计算完 KV 的一部分时即可立刻发起 RDMA Write，把数据流式推送到 Decode 节点；而 Decode 节点可一边接收 KV 一边准备生成任务，实现传输与计算重叠。FlowKV 等研究还提出了**减少传输频度**的思路：通过全局调度尽量让相同前缀的请求在同一节点 decode，避免重复的 KV 搬运。但在需要传输时，RDMA 仍然是效率最高的方案。有实验表明，在没有原生 RDMA 网卡的环境中（如仅有 100 GbE 的云服务器），通过对 NCCL 的优化也许能缩小差距，但仍远不及 RDMA 专用方案。例如 FlowKV 论文称他们针对 AWS EFA 场景优化的 NCCL 管道在多机下平均延迟比 Mooncake RDMA 方案降低了 96.3%（主要因为 Mooncake 在无序网络下需额外同步，性能受限）。然而在拥有 ConnectX 等硬件支持时，RDMA 直接传输 KV 无疑是最快且最省 CPU 的。综合来看，RDMA 让跨机 KV 缓存传输成为**可行的实时操作**，为长上下文、请求迁移等能力铺平了道路。


### 2.4 上下文并行（Context Parallel）


“上下文并行”是一种**面向超长文本**推理的并行技术，用于将一个请求的计算分布到多张 GPU 上加速。随着 LLM 的上下文长度迅速增长到十万甚至百万级 token，把这么长的序列完全交给单卡处理会导致预填充阶段延迟巨大、且显存放不下全部 KV 缓存。上下文并行通过切分序列，将不同段落分配给不同 GPU，各 GPU 各自计算自己段落的 Q/K/V 张量，并通过通信使每个 GPU 都能获取全序列范围的注意力信息。例如，Meta 提出了两种 Ring Attention 实现：

- **Pass-KV 模式**：将输入序列按 token 位置等分到 N 个GPU，每个 GPU 计算自己那段的新 token 的 Q、K、V，然后各 GPU 之间**交换各自算得的 K/V** 张量，使每个 GPU 最终拥有完整序列长度的 K/V（虽然 Q 仅有自己那段）。然后每个 GPU 用完整 K/V 对自己段落的 Q 做注意力计算，得到对应输出。这种模式下通信量主要是广播 K/V 张量，总规模 ≈(N-1)/N 倍的全序列 K/V 大小。
- **Pass-Q 模式**：类似地，各 GPU 先计算各自段落的 Q/K/V，然后**交换 Q 张量**，使每个GPU获得全序列所有 Q；接着每个 GPU 用自己的那份 K/V 对所有 Q 做局部 attention 计算，最后再汇总各 GPU 的 partial 结果得到完整输出[[69]](https://engineering.fb.com/2025/10/17/ai-research/scaling-llm-inference-innovations-tensor-parallelism-context-parallelism-expert-parallelism/#:~:text=%2A%20Pass,tensors%20are%20exchanged%20between%20ranks)。Pass-Q 通信需要广播 Q，量约为全序列 Q 大小的 (N-1)/N 倍。

无论哪种，上下文并行都涉及**多轮的 GPU 互联通信**，而且序列越长通信开销越大。为将长文本推理提速到可用范围，必须依赖 RDMA 级别的网络性能：只有当跨节点通信延迟和带宽足够好时，多机并行才能线性扩展长序列处理。例如 Meta 报告中提到，他们结合高效 attention 内核和两种 CP 模式，实现了**接近线性扩展**的惊人效果：单机 H100 上处理 100 万 token 预填充需不到 1 分钟；在 Llama 3 405B 模型上，16 节点完成 128k token 预填充仅 3.8 秒。这背后需要强大的通信支持——实际上文中也指出“**通信延迟会随着跨多主机并行而增加**”，是长上下文扩展的一大挑战。


上下文并行还有**解码阶段**的版本：当 KV 缓存极大以至单 GPU 放不下时，可按 **token 维度**对 KV 缓存进行分片，分散存储到多 GPU。例如某 70B 模型每层注意力头数 H 有限，把 KV 按 head 平分到几张卡（这相当于先用张量并行分了 H 维度），如仍不足则再按序列长度 T 方向切分给更多卡。解码时，每步生成需要用当前所有 KV 与新 query 做 attention。若 KV 分片在多卡，则每步都需要在这些卡间进行**一次全量通信**：可能是各卡先对自己持有的 KV 计算部分 attention，再把结果汇总（类似 All-Reduce），或者把新 query 广播到所有卡，各卡算出对应输出分量后拼接。这种 fine-grained 通信频率更高：**每生成 1 个token就要通信 1 次**（对比 Prefill 阶段可能只在若干层或段落边界通信）。因此，解码的上下文并行对网络延迟的要求更严苛，每一次 attention 通信都必须极低开销才能不致使 OTPT 过大。Nvidia 等厂家在这方面也有探索，例如 Snowflake 提出的 **Ulysses** 方法，用 NVSwitch 加速长上下文解码，以及 vLLM 实现中通过简单参数 -dcp 就可启用KV T 向分片并据情况增加通信并行度。总的来说，上下文并行是对网络和协调系统极大的考验，而 RDMA 是让这种跨节点长序列推理成为可能的基石。没有 RDMA 的情况下，传输延迟很可能会抵消并行带来的计算加速，得不偿失。


### 2.5 Prefill/Decode 分离（异构解耦架构）


在大模型推理中，**Prefill 阶段**和 **Decode 阶段**的计算特性差异显著：前者需要将长文本输入喂过 Transformer 模型，计算出每层的 KV 缓存，其计算量（FLOPs）随序列长度近似二次增长，非常**计算密集**；后者则是基于已有 KV 逐步生成下文，每步只算一点增量但要频繁访问大 KV，整体呈现**内存带宽受限**特征。这种性质上的不同启发了**异构解耦部署**：将 Prefill 和 Decode 交由不同资源类型的节点来执行，各司其职。例如，使用算力强大的 GPU 或加速器集中跑 Prefill，以最快速度产出 KV；然后把 KV 交给内存容量更大、带宽高的节点负责 Decode，从而同时优化吞吐和成本。Prefill/Decode 分离架构可以**弹性伸缩**：根据实时请求模式，动态调整 Prefill 节点和 Decode 节点的数量，使两种资源都尽量被饱和利用（例如当大批长文本进来时，多加 Prefill 算力；当生成请求积压时，多加 Decode 实例）。


实现 Prefill-Decode 分离的核心挑战在于**如何快速、可靠地传送 KV 缓存**，以及**如何调度异步的 Prefill/Decode 工作流**。RDMA 在这两方面都起到关键作用：

- **高速 KV 传输**：正如前文所述，单次 Prefill 可能产生数 GB 至数十 GB 的 KV 数据，需要尽快发送到 Decode 节点。如果走传统 TCP/HTTP，无论带宽还是延迟都难以令人满意。而使用 RDMA，Prefill 节点可以在计算过程中就将完成的部分 KV 直接一块块写入 Decode 节点内存（零拷贝），计算一结束 KV 也基本传输完毕。Decode 节点收到最后一块 KV 数据时，可以立即投入生成，无需额外等待。为了做到这一点，通常 Prefill 服务和 Decode 服务进程在启动时就通过 RDMA 建立好连接并交换彼此内存信息（如预留接收 KV 的缓冲区地址和 rkey 等）。Prefill 侧采用**单边 RDMAWrite with Immediate** 的方式发送 KV：这样每发完一块，远端 NIC 自动生成一个带 immediate 值的完成通知，Decode 侧累加这些通知即可判断何时收齐所有块。这种“立即数计数”的同步机制避免了依赖传输顺序（因为在 EFA 等无序网络下，最后发送的不一定最后到）。以 Mooncake 的 KV 传输为例，假设 KV 被分页为若干 4MB 页，Prefill 节点可以 post 并行的 RDMA 写请求将这些页送往 Decode 节点，并为最后一页附加一个特殊标记 Immediate 值。Decode 端通信库在 CQ 中看到附带该 Immediate 的写完成后，即可确认本次 KV 全部到达。相较传统做法需要 Prefill 显式发一个“KV 发送完毕”的消息，这种依赖 NIC 完成通知的方案更高效且无竞争。RDMA 的可靠传输也保证了 KV 传输的完整性，一旦有包丢失 NIC 硬件会自动重传，应用层无需处理。
- **动态组网和调度**：Prefill/Decode 分离面临的另一难题是**弹性和容错**。在传统 MPI/NCCL 通信模型中，一组进程的通信拓扑是预先固定的，成员不可增删。但在实际推理服务中，可能随时有 Prefill 或 Decode 实例上线下线（扩缩容或故障重启）。如果按照静态世界来通信，将完全无法适应这种动态性。RDMA 的面向连接/地址模型在这里提供了便利：每当有新 Decode 节点加入，Prefill 节点只需与之建立新的 QP 连接并注册内存，就能开始点对点发送 KV，而不用暂停整个集群。类似地，如果某 Decode 节点掉线，Prefill 服务可快速感知 QP 断开并通知调度层，将相应请求重派到其他 Decode 节点继续处理（前提是 KV 有备份或能重算）。为了协调这种弹性，通常会有一个**全局控制器**或协调组件。FlowKV 的设计就是一个例子：它引入**全局控制器**监控所有 Prefill(P) 和 Decode (D) 节点负载，并进行**负载感知调度**。当部分节点过载时，控制器可以动态增加节点或将请求路由到闲置节点，并调整 KV 传输策略（比如尽量让重复前缀的请求在同一个 D 节点上，以减少 KV 搬运）。这些操作要求通信层能够灵活连接任意 P 和 D 节点对。RDMA 的 QP 连接模型恰恰允许任意两节点直连通信，不需要像 NCCL 那样所有进程都在同一个 comm world。此外，RDMA 的 Send/Recv 可以用来传输一些小的控制消息（如通知 Prefill 任务、下发 KV 元数据等）；而大块数据用 Write，这种**双边+单边结合**方式比纯 RPC 或纯一侧写更灵活高效。总之，RDMA 让 Prefill-Decode 架构既拥有**接近本地的内存传输速度**，又具备**分布式服务的灵活性**。

### 2.6 模型权重远程加载与交换


RDMA 还可以用来优化**模型权重（参数）加载**相关的场景。例如，部署一个数百 GB 的大模型时，权重通常存放在网络存储（如分布式文件系统、对象存储或 NVMe SSD 阵列）上，传统做法通过 CPU 将模型文件读入内存再拷贝到 GPU。这一过程受磁盘和网络吞吐限制，可能耗时数分钟甚至更久。借助 RDMA，我们可以通过 **NVMe-oF (NVMe over Fabrics)** 等方案，实现存储节点直接把数据 DMA 到计算节点内存甚至 GPU。例如 Mellanox OFED 提供的驱动可使 SSD 数据经由 RoCE 直接传输到远端服务器内存，绕过双方 CPU。Mooncake 的 Transfer Engine 就支持 **DRAM 到远端 NVMe** 的数据传输接口。这意味着推理框架可以通过 RDMA 从一个集中式参数服务器或高速存储读取模型权重，无需等待常规 TCP 读取整型参数文件，然后在应用层再搬运到 GPU。这种“零拷贝拉取”显著减少了模型启动延迟。在实际工程中，一些大模型服务会将冷门的大模型权重放在远程节点，在有请求进来时再**按需加载**必要的权重到本地。这类似于参数服务器+缓存的理念，但要求极低的网络开销才能不影响响应时间。RDMA 非常适合在后台执行这种权重页的调度：远程内存中的参数页面可以预先注册 MR，本地需要时直接 RDMA Read 过来，延迟仅几十微秒加网络 RTT，比起通过 gRPC 调用服务再memcpy显著提高效率。当然，这种方案也有权衡——如果频繁跨节点读权重，可能网络成为瓶颈。因此更实际的用法是利用 RDMA 加速**初始权重加载**和**多机权重同步**（例如广播权重给新上线的节点）。Mooncake P2P Store 模块就是用于节点间快速共享大文件（如 checkpoint 模型文件），它依靠 RDMA 的直连带宽，使多副本权重加载不会集中压垮单节点带宽。实验表明，在大规模集群中通过 P2P Store，可以将节点上线时模型数据同步开销从线性降低为对数级别，因为新节点所需的数据块大多直接由邻居节点提供，不走中心瓶颈。简而言之，RDMA 在模型权重的搬移上提供了**几乎内存总线级**的速度，让动态扩容、弹性调度模型成为可能。未来随着 CXL 等新型介质互连发展，或许参数能常驻远程内存，通过 RDMA/CXL 按需访问，从而进一步突破单机显存限制——在这之前，InfiniBand/RoCE 已经为这样的架构探索打下基础。


### 2.7 远程 KV 缓存存储与调用


随着 LLM 服务对**多用户多对话**并发的需求增长，如何在集群层面高效管理 KV 缓存成为一个新课题。理想情况下，不同推理实例之间应该可以**共享和复用**彼此的 KV 缓存（前提是上下文相同），或者将暂时不用的 KV 移出 GPU 腾出空间，然后需要时再取回。传统做法中，每个推理进程都只能访问自己内存里的 KV，形成了“缓存孤岛”：某节点上算出的 KV 别的节点无法拿来用，只能重复计算相同前缀的KV；同时如果节点 A 满负荷而 B 空闲，也很难把 A 上的请求（连同KV）迁移到 B 执行。造成这些限制的很大原因在于**缺乏一个高性能的共享存储**来托管 KV 缓存——由于 KV 数据量大且读写频繁，用普通网络存储（比如 Redis、HTTP 缓存）会过慢，抵消 Prefill/Decode 解耦和重复计算消除的收益。RDMA 为这个问题提供了破解之道。近年来涌现的如 **LMCache**、**Mooncake Store**、**PrisKV** 等系统，都是构建一个**集群级别的 KV 缓存池**，底层通过 RDMA 网络把各节点的 KV 连接起来。比如 Mooncake Store 将 KV 缓存抽象为可分片存放、可多副本容错的分布式存储，支持写策略（直写/回写等）和智能预取。SGLang 的层次化缓存（HiCache）利用 Mooncake Store 实现了**设备 Memory→主机 DRAM→远程 RDMA Store** 三层缓存，冷数据自动下沉，热数据保持在高带宽层。当某请求要访问已卸载到远端的 KV 页时，通过 RDMA 可以做到**零拷贝提取**：远程节点把该页直接 DMA 到本节点 GPU 内存，无需上下文切换。PrisKV 的设计也是类似，客户端 GPU 向服务端存储发送 RDMA Read/Write 实现 KV 的远程读写，性能远超基于 TCP 的键值存储。官方数据显示，将 KV 缓存从每进程私有改为 RDMA 集群共享后，多轮对话的**复用率**和**吞吐**都有显著提升，同时 GPU 因缓存不足导致的空闲时间减少了。更重要的是，这种架构消除了 Prefill/Decode 分离最后一块障碍：即使Prefill 和 Decode 不在同一个进程，KV 也可以由 Decode 侧从远程 Store 中**主动获取**或 Prefill 侧推送，无缝衔接。这在传统 HTTP 服务框架下几乎无法实现，因为搬运几十 GB 数据的延迟太高。而有了 RDMA 共存的 KV 池，各节点之间通过高速互联共享上下文，“**内存墙**”被打破。可预见的是，未来 LLM 推理集群会越来越依赖这种 **KV 缓存即服务**模式，RDMA 网络也将成为这类系统不可或缺的基础设施。


---


## 三、Mooncake：KV 缓存解耦服务与多协议传输引擎


Mooncake 是 Moonshot AI 推出的开源 LLM 推理加速平台。它以 **KVCache-centric** 架构著称，核心组件是通用传输引擎（Transfer Engine, TE）和分布式 KV 缓存存储（Mooncake Store）。Mooncake 在 RDMA 应用上可谓“大成者”，其 RDMA 支持的深度和广度在业内少有：

- **多协议统一通信**：MooncakeTE 抽象出一个批量数据传输接口，可以在不同介质/协议上实现，包括 **TCP**、**RDMA** (InfiniBand/RoCEv2/eRDMA/GPUDirect) 和 **NVMe-oF**。对于上层来说，只需调用一次“传输数据块”操作，TE 会根据源和目的位置选择最佳路径：比如 GPU -> GPU 优先走 RoCE RDMA 直连，GPU ->远端 SSD 则走 NVMe-of，GPU -> 本地 CPU 内存甚至可以用 CXL 直连等。这种架构的优势在于屏蔽了底层复杂性并充分利用 RDMA 优势。当使用 RDMA 时，TE 相比传统 Gloo/TCP 能提供**显著更低的 IO 时延**。Mooncake 官方数据表明，在 Prefill-Decode 解耦场景下，引入 RDMA 使端到端推理延迟分位数显著降低：例如在某测试中，使用 TE (RDMA) 时 99 百分位 TTFT 降低约 **33%**，中位数也有明显改善。
- **极致性能优化**：MooncakeTE 为榨取 RDMA 性能做了多项工程优化：1）支持**多网卡并发**传输，将大数据条带化分配到多块 NIC 上，从而线性汇聚带宽；2）**拓扑感知**，优先选择与源数据同 NUMA 节点或 PCIe 拓扑亲和的 NIC，以减少跨 Socket 访问开销；3）容错：一旦某条传输通道发生错误（如链路闪断），TE会自动切换到备用路径或网卡重试，增加鲁棒性。在 Mooncake 提供的基准中，使用 4×200G RoCE 网络传输 40GB KV Cache 达到 87 GB/s，而标准 TCP 只能约 36 GB/s。这背后除了 RDMA 本身的效率，也归功于上述优化让带宽利用率接近理论最大。另有一项值得注意的是 Mooncake 对**大页内存管理和内核 Bypass** 的利用：为了减轻注册开销，他们让应用长驻分配大块内存给 RDMA 用，并使用用户态驱动直接操作 mlx5 设备进行数据面工作，使 CPU 开销最低。Mooncake 的作者曾提到，DeepEP 虽然标称基于 NVSHMEM，但实则直接用 mlx5dv（Verbs 的 Direct Verbs 扩展）和 NIC 对话。Mooncake 在这方面也是类似的极致路线，通过紧贴硬件获得性能。
- **KV 存储服务化**：Mooncake Store 将分布式 KV 缓存作为一个可复用的服务提供。它支持**多副本容灾**、**一致性哈希**等功能，使同一 KV 对象可存多份缓解热点，并支持集群水平扩展。对于 LLM 推理来说，Mooncake Store 最大的意义是实现了前述 KV 缓存跨节点共享和持久化。vLLM 和 SGLang 等项目已经集成 Mooncake Store 作为它们 Prefill 服务或缓存后端。工程上，这个 Store 本质是一个利用 RDMA 的数据平面服务：各推理节点在需要读写 KV 时，不通过常规 socket RPC，而是使用 Mooncake 提供的客户端库直接执行 RDMA 读/写操作到远端 Store 服务器内存上。这种设计减轻了服务器上的 CPU 负载，可以支持更高的 QPS。PrisKV 等系统与 Mooncake 思路类似，都选择 RDMA 是因为**传统网络堆栈无法满足 LLM 推理的毫秒级数据服务需求**。Mooncake 官方强调，对于 KV 服务，如果没有 RDMA，“在 fast token 生成循环中，常规 TCP 会太慢并导致延迟抖动”。因此 Mooncake 完全基于 RDMA zero-copy 实现了 KV 的获取和释放操作。这也解释了为什么 Mooncake 如此强调部署 RDMA 驱动（如 Mellanox OFED）和配置无损网络的重要性。

---


## 四、RDMA 优势总结及替代方案权衡


经过以上讨论可以看到，RDMA 在大模型推理框架中扮演了不可替代的角色，其**核心优势**可总结如下：

- **极致低延迟、高吞吐**

    RDMA 绕过了内核协议栈，采用硬件直接搬运内存的方式，能以微秒级延迟传送消息、接近链路极限的速率传输大数据。对于 LLM 推理这样需要密集通信的应用，这意味着可以大幅降低分布式并行带来的开销，实现接近单机的延迟和性能。例如，使用 400Gb/s InfiniBand 时 GPU 间通信延迟仅几微秒，带宽每卡可达 50 GB/s 以上。这种能力使诸如百万 token 长文、上千亿参数多机模型的推理成为现实——没有 RDMA，要么根本无法完成（因为数据传不过来），要么延迟高到不可用。可以说，RDMA 打破了网络 IO 对 LLM 推理的瓶颈，提供了传统 TCP/IP 难以企及的性能保障。

- **低 CPU 占用与零拷贝**

    RDMA 把网络数据路径上的 CPU 开销降到最低，数据直接在 NIC 和用户缓冲间搬运，无需 CPU 介入。在 AI 推理场景中，这非常关键：GPU 实例往往已经占满 CPU 核做推理调度和预处理，如果网络传输还耗费大量 CPU，会明显影响可扩展性。而 RDMA 能做到 **CPU 利用率极小**（典型每 Gb 数据传输 <1% 单核占用）。特别地，GPUDirect RDMA 进一步免除了 GPU->CPU 的内存拷贝，使 GPU 到 GPU 完全零拷贝。这一点在 KV 缓存、模型参数这样超大数据传输中意义重大——无论多大都只是在 PCIe 和网络上流动，CPU 几乎不感知，不会拖慢其它任务。


当然，RDMA 并非唯一选择，也存在一些替代方案或互补手段可考虑：

- **传统 TCP/IP(Socket)**

    如果硬件或运维条件限制，无法使用 RDMA，那么走 TCP/IP 协议栈是最通用的方案（例如很多公有云环境就只能使用 TCP）。优点是**兼容性好**、部署简单，所有网络都支持。但缺点也明显：延迟和抖动高一个数量级（微秒变毫秒），CPU 开销大且随并发增加线性上升。在小规模或低并发场景下，优化良好的 TCP 栈配合高速以太网也能凑合，例如Mooncake 测试中 12 tok/s 的输出下，TCP 的 P99 延迟约 6 秒，RDMA 约 4 秒。但随着规模扩大，TCP 会成为瓶颈。因此，如果不得不用 TCP，一般需要进行诸如**内核 Bypass**（DPDK 等）或**批量 RPC** 等改造，并接受性能上的损失。

- **NVLink/NVSwitch（GPU 直连）**

    对于单机多 GPU，NVLink 和 NVSwitch 提供了超高速的互联（带宽高达每 GPU 数百 GB/s），远超 RDMA 网络。因此尽量在单机内部完成并行计算，跨节点通信最少，这是优化的一条思路。在实际部署中，如果模型和上下文能塞进一台 8 卡或 16 卡服务器，那应优先利用 NVSwitch 内部通信而不是拆到多节点。但 NVLink 的局限是**只能在单机或少数直连机器内**使用，无法大规模横向扩展。而 RDMA 可以将成百上千台机器连成一个扁平高效的通信域。因此 NVLink 和 RDMA 并不冲突：前者管机内，后者管机间。例如 Megatron-LM 训练时就是机内 NCCL 用 NVLink，机间 NCCL 用 InfiniBand。对于推理系统，同样建议**充分利用 NVLink 等本地带宽**（如将能在一机内完成的并行不拆到跨机），在此基础上，通过 RDMA 连接多机协作完成超出单机能力的任务。

- **ZeroMQ 等高层消息库**

    ZeroMQ、gRPC、OneRPC 等抽象出了简洁的分布式通信接口，可以简化开发。如果推理框架更注重快速迭代、而对极致性能要求不高，可以使用这些封装好的库。它们通常跑在 TCP 之上（也有的支持 Infiniband 透传消息），在小消息 RPC 方面提供了便利和一定优化。但对于 LLM 场景，动辄 GB 级的数据块传输，用高层库往往隐藏不了成本，反而可能因为序列化和线程切换增加额外开销。因此许多团队选择**关键路径自研 RDMA，非关键路径用 RPC 库**。比如 Prefill 和 Decode 服务的控制信号可以走 gRPC，但 KV 数据还是走 RDMA 直接拉。这种分层方式兼顾了开发效率和性能。

- **自定义网络协议**

    一些公司在定制网络协议，如 Facebook 在优化 TensorParallel 通信时设计的 DDA 算法、还有像微秒级 RPC、无序传输协议等。如果拥有网络硬件研发能力，也可以开发专门针对 LLM 通信模式的协议。例如把 LLM 推理常见的 all-to-all、广播等在 NIC 固化实现，甚至支持 GPU 直接触发。不过短期来看，这种自定义需要投入巨大且难以通用。大部分优化可以通过在 RDMA Verbs 之上实现新协议（如前述 URD 库）来完成，无需重新造轮子到底层电路。因此所谓自定义协议，大多也是跑在 RDMA 承载之上，而非完全抛开 RDMA。

- **非 RDMA 硬件互联**：除了 NVLink/NVSwitch，本地还有 PCIeP2P 直连（同一 Root 下的设备可直接 DMA 对方内存）也可在双机直连场景发挥作用。另外围绕 Intel OPA（Omni-Path）、Cray Slingshot 等 HPC 网络，以及即将兴起的 CXL0 Fabric，都可能为跨节点数据共享提供新手段。比如 CXL Memory Pooling 可以让多节点共享一块大内存，可以想象用它来放 KV 缓存，然后各 GPU 通过 CXL 访问。不过这些技术目前在 LLM 推理上应用还不成熟，生态和软硬件协同也不完善，短期内难以替代 RDMA 的地位。

**总结而言**，RDMA 之所以在大模型推理中难以替代，正是因为它抓住了“**高速传输**”这个软肋，提供了当前计算-存储-网络解耦架构下最有效的胶水。在参数亿亿级、上下文十万级的新一代模型面前，计算再强大，若没有相应的数据流能力配合，也是英雄无用武之地。RDMA 将原本需数百毫秒的网络通信压缩到几十微秒，与 GPU Kernel 的量级相当，这使得各种并行、分离策略得以落地成为正收益方案。当然，我们也应看到 RDMA 部署的门槛和成本：需要专门硬件和网络配置，因此并非所有场景都能立即采用。在无法使用 RDMA 时，只能退而求其次，用更大的模型切分、更保守的并行度，甚至牺牲部分精度（如压缩 KV 传输的数据量）来减轻网络负担。这些都是权衡取舍。但随着AI基础设施的发展，RDMA 或类似的高速网络将会越来越普及。可以预见，在“AI 即基础设施”的未来，大规模模型推理系统将天然建立在 RDMA 级网络之上，其它替代方案只能作为辅助或过渡。正如本文引用的一系列研究和实践所示：**RDMA 已成为现代大模型推理中解决性能瓶颈的关键路径，其不可替代性源于它对硬件潜力的最大释放**。面向未来，我们有理由相信，通过RDMA和不断演进的网络技术，大模型推理的性能和规模还将继续突破，为更复杂智能应用的落地提供坚实支撑。


---


## 参考链接

- 大语言模型系统中RDMA通信的一些探索

    [https://abcdabcd987.com/2025/11/09/rdma-p2p-for-llm/](https://abcdabcd987.com/2025/11/09/rdma-p2p-for-llm/)

- Virtual RDMA Device Driver Implementation (Part II): Building a Kernel-Recognizable RDMA Device from Scratch | by DatenLord | Medium

    [https://medium.com/@datenlord/virtual-rdma-device-driver-implementation-part-ii-building-a-kernel-recognizable-rdma-device-07fed0b9d2ec](https://medium.com/@datenlord/virtual-rdma-device-driver-implementation-part-ii-building-a-kernel-recognizable-rdma-device-07fed0b9d2ec)

- 使用 NVIDIA DOCA GPUNetIO 解锁 GPU 加速的 RDMA - NVIDIA 技术博客

    [https://developer.nvidia.cn/blog/unlocking-gpu-accelerated-rdma-with-nvidia-doca-gpunetio/](https://developer.nvidia.cn/blog/unlocking-gpu-accelerated-rdma-with-nvidia-doca-gpunetio/)

- Scalable Inference with RDMA and Tiered KV Caching | by Nadeem Khan(NK) | LearnWithNK | Nov, 2025 | Medium

    [https://medium.com/learnwithnk/scalable-inference-with-rdma-and-tiered-kv-caching-9d7e494a863b](https://medium.com/learnwithnk/scalable-inference-with-rdma-and-tiered-kv-caching-9d7e494a863b)

- RDMA over Converged Ethernet (RoCE) - NVIDIA Docs

    [https://docs.nvidia.com/networking/display/MLNXOFEDv497100LTS/RDMA+over+Converged+Ethernet+(RoCE)](https://docs.nvidia.com/networking/display/MLNXOFEDv497100LTS/RDMA+over+Converged+Ethernet+(RoCE))

- Scaling LLM Inference: Innovations in Tensor Parallelism, Context Parallelism, and Expert Parallelism - Engineering at Meta

    [https://engineering.fb.com/2025/10/17/ai-research/scaling-llm-inference-innovations-tensor-parallelism-context-parallelism-expert-parallelism/](https://engineering.fb.com/2025/10/17/ai-research/scaling-llm-inference-innovations-tensor-parallelism-context-parallelism-expert-parallelism/)

- FlowKV: A Disaggregated Inference Framework with Low-Latency KV Cache Transfer and Load-Aware Scheduling

    [https://arxiv.org/html/2504.03775v1](https://arxiv.org/html/2504.03775v1)

- Ulysses: Unlocking Low-Latency, High-Throughput Inference for ...

    [https://www.snowflake.com/en/engineering-blog/ulysses-low-latency-llm-inference/](https://www.snowflake.com/en/engineering-blog/ulysses-low-latency-llm-inference/)

- [RFC]: Implement disaggregated prefilling using Mooncake · Issue #10727 · vllm-project/vllm · GitHub

    [https://github.com/vllm-project/vllm/issues/10727](https://github.com/vllm-project/vllm/issues/10727)

- [PDF] Efficient Memory Management for Large Language Model Serving ...

    [https://arxiv.org/pdf/2309.06180](https://arxiv.org/pdf/2309.06180)

- [PDF] An Efficient KV Cache Layer for Enterprise-Scale LLM Inference

    [https://lmcache.ai/tech_report.pdf](https://lmcache.ai/tech_report.pdf)

- Prefill/Decode Disaggregation - llm-d

    [http://llm-d.ai/docs/guide/Installation/pd-disaggregation](http://llm-d.ai/docs/guide/Installation/pd-disaggregation)

- Explorations of RDMA in LLM Systems

    [https://le.qun.ch/en/blog/2025/11/09/rdma-p2p-for-llm/](https://le.qun.ch/en/blog/2025/11/09/rdma-p2p-for-llm/)

