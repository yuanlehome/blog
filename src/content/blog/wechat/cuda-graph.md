---
title: CUDA Graph 学习笔记
slug: cuda-graph
date: '2025-12-27'
tags: ['CUDA Graphs']
status: published
source_url: 'https://mp.weixin.qq.com/s/9SAW3kgI_G1CzvNM9_aLTg'
source_author: 平六层
imported_at: '2025-12-27T22:32:41.224Z'
---

**目录**：为什么需要 - 核心抽象 - 生命周期与执行语义 - 静态性 - 性能收益 - torch.compile - 推理系统

---

# 为什么需要 CUDA Graph

一个 GPU 程序的执行可以分成两个层面来看：

**数据面**是我们通常关注的那部分，也就是 kernel 在 GPU 上的实际计算。矩阵乘、卷积、attention 等等，GPU 的 SM 在跑真正的计算逻辑；这部分的性能主要由算力、显存/缓存带宽、访存模式、并行度与占用率等因素共同决定。

**控制面**则是另一个问题。每次我们调用一个 kernel，Host 端（CPU）需要做一系列事情：准备 kernel 参数、把 launch 请求发给 driver、driver 再把这个请求排到 GPU 的执行队列里。这个过程涉及 CPU-GPU 之间的交互，会引入非零的提交延迟底噪（通常在微秒级，且与 GPU/driver/runtime/系统负载等因素有关）。

控制面开销在很多场景下可以被忽略。如果我们的 kernel 跑得足够久（比如一个大矩阵乘要跑几毫秒），那么 launch 的几微秒开销基本可以忽略不计。但如果我们的 kernel 非常短、数量又非常多，问题就来了。

## 短 kernel 场景下的控制面放大效应

来看一个具体的场景。

假设有一个计算流程，包含大量的短 kernel，每个 kernel 的实际执行时间非常短。NVIDIA 的开发者博客给出过一个微基准测试的数据：在其测试环境中，单个短 kernel 的 GPU 执行时间大约在 2.9 微秒左右。

但如果我们用最保守的方式来跑这些 kernel（例如在每个 kernel launch 之后都做一次同步等待，以构造最糟糕的控制面占比基线），实际测量下来，每个 kernel 折算的总耗时会飙升到 9.6 微秒左右。这意味着什么？执行时间只有不到三分之一是在做真正的计算，剩下的时间都被 launch overhead 和同步等待吃掉了。

这是一个非常典型的控制面主导场景。在这种同步基线下，GPU 会频繁等待 CPU 把下一个 kernel 发过来，而 CPU 这边在忙着处理上一个 kernel 的同步。

## 一个自然的优化方向：把同步挪到循环末尾

在引入 CUDA Graph 之前，一个更低成本的优化方式是调整同步策略。

同样是上面那个微基准，如果把每个 kernel 后面的同步去掉，改成在整个 timestep 末尾才做一次同步，每个 kernel 的折算耗时会降到 3.8 微秒左右。

这个改进的机制是：当我们不在每个 kernel 之后同步时，CPU 可以在 GPU 还在执行 kernel 的时候就开始准备下一个 kernel 的 launch。launch overhead 被 kernel 执行时间隐藏了一部分。这在 CUDA 的 stream 语义下是合法的，因为同一个 stream 内的操作本来就是保序的。

从 9.6 微秒到 3.8 微秒，仅仅通过调整同步位置就拿到了两倍多的提升。这个优化几乎没有代码侵入性，值得在考虑更复杂方案之前先做掉。

## CUDA Graph 的思路：把提交序列固化

调整同步策略能帮上很多忙，但在某些场景下仍然不够。

如果我们的 kernel 实在太短、数量实在太多，即使 launch overhead 被部分隐藏，CPU 这边还是会成为瓶颈。每次循环迭代，CPU 都要重复做同样的提交动作：准备本轮 kernel 参数、重复调用同一组 launch 接口、让 driver 重复做排队与提交。这些重复的控制面工作本身就是浪费。

CUDA Graph 的思路是：既然每次循环要做的事情是一样的，那就不要每次都重新做一遍。把整个提交序列录制下来，固化成一个可复用的结构（Graph）。之后每次循环只需要回放这个 Graph，而不需要重新构造 launch 调用链。

用稍微正式一点的语言来说：CUDA Graph 把一系列 GPU 工作（kernel、memcpy、memset、host work 等）描述成一个 DAG（有向无环图），每个节点是一个操作，边表示依赖关系。这个 DAG 会被实例化成一个可执行对象（GraphExec），之后可以反复 launch。

回到那个微基准，使用 CUDA Graph 后，每个 kernel 的折算耗时进一步降到 3.4 微秒左右。相比调整同步后的 3.8 微秒又有提升，虽然增量不算特别大，但这个收益往往是低侵入的：一旦 Graph 录好，后续每次循环只需要一次 graph launch，就能减少大量逐 kernel 的提交开销。

## 收益来源：减少重复的 Host 侧工作

把 CUDA Graph 的收益拆开来看，核心来自三件事：

**消除重复构造**。原本每次 launch kernel 都需要 CPU 做参数准备和 driver 调用。用了 Graph 之后，这些工作在录制阶段只做一次，后续的 replay 只需要一次轻量的 graph launch。

**前置准备**。Graph 在实例化（instantiate）阶段会做一系列准备工作。这个准备可以通过 upload 接口提前做好，把一次性成本移出关键路径。这对追求低延迟抖动的场景比较有用。

**参数热更新**。在拓扑不变、并且节点类型/更新规则允许的前提下，可以更新节点参数或对 GraphExec 做有限度的更新。这样就能复用 GraphExec 内部已经准备好的资源，而不需要每次都重新 instantiate（但更新能力并不是无限制的，工程上通常需要准备更新失败时的回退路径）。

需要特别强调的一点是：**CUDA Graph 优化的是控制面，不是数据面**。它不会让我们的 kernel 本身跑得更快，矩阵乘还是那个矩阵乘。它节省的是 Host 侧反复做同样工作的开销。

## 一次性成本：Capture 与 Instantiate

CUDA Graph 不是没有代价的。

把一段执行序列录制成 Graph，然后把 Graph 实例化成 GraphExec，这两步本身有成本。NVIDIA 博客里的示例数据是，capture 加 instantiate 的一次性开销大约在 400 微秒的量级（该数字来自特定测试环境，仅用于说明数量级；实际取决于 Graph 的规模、节点类型与系统软件栈）。

另外，第一次 launch GraphExec 往往比后续的 launch 更慢一些。这是因为第一次 launch 可能还需要完成一些初始化工作。按照博客的描述，第一次 launch 可能比后续慢 33% 左右。

这意味着：CUDA Graph 的收益需要足够多的重复 launch 来摊薄一次性成本。如果我们的 Graph 只跑几次就不用了，那 capture 和 instantiate 的成本可能根本收不回来。

从性能评估的角度，测量时需要区分 warmup 阶段和 steady state 阶段。instantiate 只做一次，第一次 launch 更慢，这些因素需要在测量方法里考虑进去。

## 适用场景：

基于上面的分析，CUDA Graph 不是在所有场景下都能带来收益。它更适合具备以下特征的工作负载：

**短 kernel 密集**。如果单个 kernel 的执行时间很长（比如几毫秒），launch overhead 的占比本来就很低，Graph 的边际收益会比较小。相反，如果有大量微秒级的短 kernel，控制面开销占比高，Graph 的效果会更明显。

**高度重复**。Graph 的收益模型是一次录制、多次回放。如果执行序列不重复，或者每次迭代的 kernel 组合都不一样，就没有可复用的结构，Graph 帮不上忙。

**拓扑稳定**。CUDA Graph 是一个静态 DAG，节点集合和依赖关系在录制时就确定了。如果程序有动态控制流（每次迭代走的分支不一样），这种结构天然与 Graph 的静态性冲突。

实际上，深度学习推理（尤其是 decode 阶段）是一个非常典型的 CUDA Graph 适用场景：每个 token step 的计算拓扑高度一致，kernel 数量多、单个 kernel 相对较短，整个序列高度重复。

## 不是银弹

虽然原理听起来很好，但在框架自动化部署的场景下，CUDA Graph 不是一个可以无脑打开的全局开关。它的收益和代价需要按场景评估，这涉及到静态性约束、参数按值捕获的语义、以及框架为满足这些约束而引入的额外开销。后续会详细讨论这些问题。

# CUDA Graph 的核心抽象

CUDA Graph 的设计把定义和执行分成了两个独立的层次。这种分离是 CUDA Graph 能够实现复用的基础，也是理解后续所有机制的起点。

## 四个核心对象

CUDA Graph 的对象模型由四个核心概念组成：Graph、Node、Edge、GraphExec。

**Graph** 是一个 DAG（有向无环图）的定义，描述了一组 GPU/CPU 操作以及它们之间的依赖关系。我们可以把它理解成一个计划书或者蓝图，它本身并不执行任何操作。Graph 的生命周期由创建（cuGraphCreate 或 cudaGraphCreate）到销毁（cuGraphDestroy 或 cudaGraphDestroy）。

**Node** 是 Graph 中的节点，代表一个具体的操作。每个 Node 都有自己的类型，不同类型的节点执行不同的工作。常见的节点类型包括：kernel 执行、memcpy 数据拷贝、memset 内存填充、host 回调（在 CPU 上执行的函数）等等。每个 Node 只属于一个 Graph。

**Edge** 是 Node 之间的依赖边，表示执行顺序。如果存在一条从 Node A 到 Node B 的边，意味着 B 必须等 A 完成之后才能开始。这些边共同定义了整个 Graph 的偏序关系。

**GraphExec** 是 Graph 的可执行实例。通过 instantiate（实例化）操作，一个 Graph 定义会被验证并转换成 GraphExec。GraphExec 是真正可以被 launch（启动）的对象。一个 Graph 可以被实例化多次，得到多个独立的 GraphExec。

用更简洁的方式来说：Graph 是模板，GraphExec 是实例；Graph 描述要做什么，GraphExec 负责真正去做。

## 为什么要分离 Graph 和 GraphExec

这个分离设计不是为了增加复杂性，而是为了支持两个重要的使用模式：

**复用**。同一个 Graph 可以被实例化多次，得到多个 GraphExec。这对于需要并发执行同一拓扑结构的场景非常重要。CUDA 规定同一个 GraphExec 在任一时刻只能有一个实例在执行。如果我们想要并发执行同一个计算流程，必须创建多个 GraphExec。

**更新**。GraphExec 在创建之后，可以在一定条件下更新节点参数，而不需要重新 instantiate。这使得热更新成为可能：保持 GraphExec 的内部状态和资源不变，只更新需要变化的参数。

从系统设计的角度看，instantiate 是一个相对昂贵的操作，它会做结构验证、资源准备等工作。把定义和执行实例分开，可以让这个昂贵操作只做一次，后续通过轻量的参数更新来适应变化。

## 节点类型

CUDA Graph 支持多种节点类型，覆盖了 GPU 程序中常见的操作类别。来看一个完整的清单：

**Kernel Node**

执行一个 GPU kernel。这是最核心的节点类型。Kernel 的参数在添加节点时被拷贝，支持两种参数传递方式：通过 kernelParams 数组传递每个参数的指针，或者通过 extra 把参数打包到一个 buffer（两者互斥，不能同时使用）。

有一个重要的限制：通过 Graph 启动的 kernel 不能使用 texture 和 surface references。通过 reference 读写是未定义行为。但 texture 和 surface objects 不受此限制。

**Memcpy Node**

执行内存拷贝操作。描述方式与 cudaMemcpy3D 类似。在某些设备上，如果涉及 managed memory，会有额外的限制。

**Memset Node**

执行内存填充操作。element size 必须是 1、2 或 4 字节。

**Host Node**

在 CPU 上执行一段回调函数。这个节点类型有一个非常重要的约束：回调函数中不能调用 CUDA API。这一点在 CUDA 文档中反复强调。如果违反这个约束，可能会收到 cudaErrorNotPermitted 错误。Host Node 适合做轻量的 CPU 侧逻辑，比如状态更新、信号通知，但不能用来嵌套调用 CUDA 操作。

**Event Record / Event Wait Node**

用于在 Graph 内部记录和等待 event。Event Record Node 在每次图执行时记录一个 event 来标记依赖节点的完成；Event Wait Node 等待一个 event。它们最常见的用途是在同一 context 内做跨 stream 的依赖衔接；如果需要跨进程或跨 API 的同步，通常需要使用 IPC event 或 external semaphore 等机制来表达。

但有一个限制：Event Record 和 Event Wait 节点不能用于 loops 或 conditionals（条件控制结构）。

**Empty Node**

执行时什么都不做。看起来没什么用，但实际上在依赖边管理中很有价值。

考虑这样一个场景：我们有两个阶段的计算，每个阶段有 n 个节点。如果第二阶段的每个节点都依赖第一阶段的所有节点，需要 n² 条边。但如果在两个阶段之间插入一个 Empty Node 作为 barrier，第一阶段的所有节点连到这个 Empty Node，这个 Empty Node 再连到第二阶段的所有节点，只需要 2n 条边。这是一种常见的边压缩技巧。

**Child Graph Node**

把另一个 Graph 嵌入到当前 Graph 中作为一个节点。在添加 Child Graph Node 时，子图会被克隆到节点内部。这提供了一种模块化组织计算的方式。

但有一个重要的限制：如果子图中包含 allocation node、free node 或 conditional node，添加 Child Graph Node 会失败并返回错误。

**External Semaphore Signal / Wait Node**

用于与外部同步机制（如图形 API、媒体 API）的互操作。在更新 GraphExec 时，signal/wait 的操作数量不能改变。

**Memory Allocation / Free Node**

在 Graph 执行过程中进行内存分配和释放。这是较新的能力，允许把内存生命周期纳入图的控制范围。

Allocation Node 在添加节点时就会返回一个地址（通过 nodeParams.dptr），并且这个地址跨 instantiation 和多次 launch 保持固定。这个地址固定的特性是图内内存管理的核心语义：后续节点可以引用这个稳定地址，但该地址所代表的 allocation 只有在 alloc node 执行之后才可被访问。

Free Node 用于释放 Allocation Node 分配的内存。释放可以发生在同一个 Graph 内（通过 free node），也可以选择不在 owning graph 内 free，而在图执行完成后通过 cudaMemFreeAsync/cudaMemFree、在另一张图中用 free node，或通过 instantiate flag 的自动回收机制完成。

需要注意的是：同一个 allocation 的释放路径必须是唯一且一致的，不能既在 owning graph 内 free，又在外部或其它图里 free，否则会触发错误或未定义行为。

但 alloc/free 节点会带来结构性限制，这部分内容会在后续展开。

**Conditional 相关节点**

用于在 Graph 中实现条件控制流。条件值通常由 device 侧 kernel 写入；同时也可以配置一个默认值，使得每次图执行开始时条件句柄先被设置为该默认值。这是比较新的能力，版本要求较高，使用约束也比较多。

**Batch Memory Operation Node**

执行一组内存操作。有一个重要的警告：通过这个节点建立的同步顺序对 CUDA 调度器不可见。不当使用可能导致死锁。如果使用这类节点，建议同时用 CUDA 可见的依赖机制（比如 event）来确保正确的执行顺序。

## 依赖边的语义

Edge 在 CUDA Graph 中表示必须先于的关系。如果存在从 A 到 B 的边，B 的执行必须等 A 完成。

添加依赖边的 API（cuGraphAddDependencies / cudaGraphAddDependencies）接受 from 数组和 to 数组，按索引配对：from\[i] → to\[i] 表示一条依赖边。每个被引用的节点必须属于同一个 Graph。添加已存在的依赖边会返回错误。

Edge 可以携带额外信息（edgeData）。大多数场景下 edgeData 是默认值（全零），但如果我们使用了非默认的 edgeData，在查询边的时候需要注意：如果 edgeData 参数传 NULL，可能会触发 cudaErrorLossyQuery 或 CUDA_ERROR_LOSSY_QUERY 错误，表示查询结果不完整。这个设计是为了防止调用方在不知情的情况下丢失边的属性信息。

删除依赖边的 API 也存在，但有一个限制：如果 Graph 包含 memory allocation 或 memory free 节点，不能删除依赖边，调用会返回错误。

## GraphExec 的执行语义

GraphExec 是真正可以被 launch 的对象。关于它的执行，有几个关键的语义需要明确：

**并发限制**

同一个 GraphExec 在任一时刻只能有一个实例在执行。这是一个硬性限制。如果我们想要多个执行实例并发运行同一个计算流程，必须从同一个 Graph 实例化出多个 GraphExec。

每次 launch 会与两个方向上的工作形成顺序：排在 launch stream 中之前工作的后面，也排在同一个 GraphExec 之前 launches 的后面。

**Upload 与 Launch**

Launch 是提交 GraphExec 执行的操作。Upload 是一个可选的预热操作：把 GraphExec 上传到 device 但不执行。

同一个 GraphExec 的 uploads 之间会串行化。每次 upload 排在 upload stream 之前工作的后面，也排在该 GraphExec 之前 launches 的后面。

Upload 的价值在于：可以把一些准备工作提前做好，减少 launch 时的抖动。对于追求稳定低延迟的场景，这个机制比较有用。

**Alloc/Free 相关的执行条件**

如果 GraphExec 上一次 launch 产生的 allocations 还没有被释放，并且该 GraphExec 在 instantiate 时没有设置 AUTO_FREE_ON_LAUNCH flag，下一次 launch 会失败，返回 cudaErrorInvalidValue。

这意味着使用图内 alloc/free 的时候，需要明确管理每轮 launch 后的 allocation 状态，否则容易出现跑几轮后突然失败的问题。

## 两套 API：Runtime 与 Driver

CUDA Graph 的 API 有两套：Runtime API（cudaGraph\* 前缀）和 Driver API（cuGraph\* 前缀）。

这两套 API 在对象模型上是同构的：

| Runtime API     | Driver API  | 含义         |
| --------------- | ----------- | ------------ |
| cudaGraph_t     | CUgraph     | Graph 定义   |
| cudaGraphNode_t | CUgraphNode | 节点句柄     |
| cudaGraphExec_t | CUgraphExec | 可执行图实例 |

核心语义是一致的，差异主要在接口风格和一些细节参数上。Runtime API 通常更简洁一些，Driver API 提供更底层的控制。在实际使用中，选择哪套 API 取决于项目的整体技术栈和对控制粒度的需求。

## Graph 的线程安全边界

CUDA 文档在大量 API 的说明中反复强调：**Graph objects are not threadsafe**。

这意味着什么？对同一个 cudaGraph_t、cudaGraphExec_t 或 cudaGraphNode_t 的并发访问（查询、修改、launch、upload、update）需要调用方自己做同步保护。CUDA 内部不会为我们处理这个问题。

如果多个线程要操作同一个 Graph 相关对象，需要外部互斥锁来保护，或者每个线程使用独立的对象。

这个约束在系统设计时非常重要。如果我们的应用是多线程的，并且多个线程可能同时操作 Graph，需要在架构层面考虑如何避免冲突。

---

## 异步错误回传

很多 Graph 管理 API 的文档都提到：**this function may also return error codes from previous, asynchronous launches**。

这意味着：当我们在某个 query/update/launch 调用点看到错误时，错误不一定是这个调用点本身的问题，也可能是之前某个异步执行（kernel/memcpy/graph launch）产生的错误被延迟到这里才回传出来。

工程上的含义是：排障不能只盯当前 API 的参数，要结合最近的同步点（event 或 stream sync）以及最近一次图执行的上下文来定位真正的源头。在复杂系统中，建议在关键阶段（capture 后、首次执行后、更新后）设置明确的同步点，以便更容易归因。

## 调试与可观测性

CUDA Graph 提供了一些调试与可观测能力，用来把“图到底长什么样、错在哪个节点”变得可追踪：

**DOT 输出**

cuGraphDebugDotPrint / cudaGraphDebugDotPrint 可以把 Graph 的结构输出为 DOT 格式文件，便于用 Graphviz 可视化审计。DOT 输出通常会包含拓扑结构、节点类型、节点 ID、kernel 名称、memcpy 方向等；具体字段取决于 CUDA 版本与输出 flags。

**ID 机制**

CUDA Graph 提供了多种 ID 用于定位和关联：

- Graph ID（cuGraphGetId / cudaGraphGetId）
- GraphExec ID（cuGraphExecGetId / cudaGraphExecGetId）
- Node Local ID（cuGraphNodeGetLocalId / cudaGraphNodeGetLocalId）
- Tools Node ID（cuGraphNodeGetToolsId / cudaGraphNodeGetToolsId）

实践中的常见用法是：当 instantiate 或 update 返回 error node handle 时，结合 DOT 输出与 nodeId，把错误精确定位到图中的具体节点。

# 生命周期与执行语义

一张 CUDA Graph 从创建到销毁，会经历几个明确的阶段。理解这些阶段的边界和语义，是正确使用 CUDA Graph 的前提。

## 典型的生命周期阶段

CUDA Graph 的生命周期可以分为五个阶段：

**Build Graph（构建图）**

这个阶段的目标是得到一个 cudaGraph_t 对象。有两条路径可以走：显式构图或者 Stream Capture。无论哪条路径，最终产物都是一个描述了节点和依赖关系的 DAG。

**Instantiate（实例化）**

把 cudaGraph_t 实例化成 cudaGraphExec_t。这一步会对图结构做验证，检查各种约束条件是否满足，并进行必要的准备工作。实例化是一个相对昂贵的操作。

**Upload（上传，可选）**

把 GraphExec 预先上传到 device，但不执行。这是一个可选步骤，目的是把部分准备成本从首次 launch 移出去。

**Launch / Replay（执行 / 回放）**

真正在 GPU 上执行图中描述的操作。Launch 可以反复进行，这是 CUDA Graph 能够摊薄一次性成本的关键。

**Destroy（销毁）**

销毁 GraphExec 和 Graph 对象，释放相关资源。

来看一个典型的使用流程：

```
cudaGraph_t graph;
cudaGraphExec_t exec;

// 1. Build（以 capture 为例）
cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
// ... 提交一系列 kernel/memcpy 等操作 ...
cudaStreamEndCapture(stream, &graph);

// 2. Instantiate
cudaGraphInstantiate(&exec, graph, NULL, NULL, 0);

// 3. Upload（可选）
cudaGraphUpload(exec, stream);

// 4. Launch（可反复执行）
for (int i = 0; i < iterations; i++) {
    cudaGraphLaunch(exec, stream);
    cudaStreamSynchronize(stream);
}

// 5. Destroy
cudaGraphExecDestroy(exec);
cudaGraphDestroy(graph);
```

这个流程体现了 CUDA Graph 的核心价值：Build 和 Instantiate 只做一次，Launch 反复进行。这里在每次 launch 后做同步只是为了让示例更直观（以及方便测量/归因）；在真实系统里，同步策略通常需要单独设计。

## 两种构图方式

### 显式构图

显式构图是通过 Graph API 直接创建节点和依赖边来构建图。

基本流程是：先创建一个空的 Graph，然后逐个添加节点（cudaGraphAddKernelNode、cudaGraphAddMemcpyNode 等），同时指定每个节点的依赖关系。

```
cudaGraph_t graph;
cudaGraphCreate(&graph, 0);  // flags 必须为 0

// 添加节点，指定依赖
cudaGraphNode_t kernelNode;
cudaGraphAddKernelNode(&kernelNode, graph, dependencies, numDependencies, &kernelParams);
```

显式构图的优点是控制力最强：我们可以精确地定义每个节点的类型、参数、依赖关系。对于需要插入 host node、memory alloc/free node 等特殊节点的场景，显式构图是更自然的选择。

缺点是工程复杂度更高。我们需要显式维护节点句柄和依赖关系，代码更像是在写 IR。

### Stream Capture

Stream Capture 是另一种构图方式：把一段提交到 stream 的 GPU 工作录制下来，自动生成对应的 Graph。

基本流程是：调用 cudaStreamBeginCapture 开始录制，然后正常向 stream 提交 kernel、memcpy 等操作，最后调用 cudaStreamEndCapture 结束录制并得到 Graph。

```
cudaGraph_t graph;
cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

// 正常提交 kernel/memcpy 等
kernel1<<<grid, block, 0, stream>>>(...);
cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToDevice, stream);
kernel2<<<grid, block, 0, stream>>>(...);

cudaStreamEndCapture(stream, &graph);
```

Stream Capture 的优点是对已有代码的侵入性小。如果我们已经有一套基于 stream 的 pipeline，可以用最小的改动把它固化成 Graph。另外，在相关库调用本身支持 stream capture 的前提下（并且调用满足 capture 的约束），库内部的 kernel launch 也可以被自动录制进来。

缺点是 capture 过程有自己的限制和复杂性，调试起来可能更困难一些。

### 多 Stream Capture

一个更进阶的用法是在 capture 过程中涉及多个 stream。

基本模式是：主 stream 开始 capture（Global 模式是一种常见选择），然后在同一个 capture 期间通过 event 的 record/wait 把其他 stream 的工作与主 stream 建立依赖关系，使其被纳入同一个 capture graph。

```
cudaStreamBeginCapture(stream1, cudaStreamCaptureModeGlobal);

// 在主 stream 上 record 一个 event
cudaEventRecord(forkEvent, stream1);

// 其他 stream 等待这个 event，然后做自己的工作
cudaStreamWaitEvent(stream2, forkEvent);
kernel_on_stream2<<<grid, block, 0, stream2>>>(...);
cudaEventRecord(joinEvent, stream2);

// 主 stream 等待其他 stream 完成
cudaStreamWaitEvent(stream1, joinEvent);

// 继续主 stream 的后续工作
kernel_on_stream1<<<grid, block, 0, stream1>>>(...);

cudaStreamEndCapture(stream1, &graph);
```

这种方式可以把跨 stream 的并行工作录制成一个 Graph，保留原有的依赖关系。

### 对比

两种构图方式的选择取决于具体场景：

| 场景                                     | 推荐方式                   |
| ---------------------------------------- | -------------------------- |
| 已有成熟的 stream pipeline，想固化复用   | Stream Capture             |
| 需要捕获库调用序列（CUBLAS/CUSPARSE 等） | Stream Capture             |
| 需要精确控制依赖、插入特殊节点           | 显式构图                   |
| 需要 memory alloc/free node              | 显式构图                   |
| 需要 host node                           | 两者都可以，显式构图更可控 |

无论哪种方式构建的 Graph，后续的 Instantiate、Launch、Update、Destroy 流程是一样的。

## Instantiate：从定义到可执行实例

Instantiate 是把 cudaGraph_t 转换成 cudaGraphExec_t 的过程。

这一步会做几件事情：

- 验证图结构的合法性（节点类型、依赖关系、各种约束条件）
- 验证节点内部的参数是否合法
- 进行必要的资源准备

Instantiate 的基本 API 是 cudaGraphInstantiate。还有一个带更多控制选项的版本 cudaGraphInstantiateWithParams / cudaGraphInstantiateWithFlags。

### Instantiate Flags

Instantiate 可以通过 flags 控制一些行为：

**cudaGraphInstantiateFlagAutoFreeOnLaunch**

对于包含 alloc node 的图，设置这个 flag 后，每次 launch 之前会自动释放上次 launch 产生的未释放 allocations。这可以避免因为 allocation 没有被释放而导致的 launch 失败。

**cudaGraphInstantiateFlagDeviceLaunch**

配置这个 GraphExec 可以从 device 侧启动（同时也可以从 host 侧启动）。这是一个比较高级的能力，有额外的约束条件。

注意：AutoFreeOnLaunch 和 DeviceLaunch 不能同时使用。

**cudaGraphInstantiateFlagUseNodePriority**

执行时使用每个节点的 priority 属性，而不是 launch stream 的 priority。在 stream capture 场景下，节点的 priority 是从 capture 时的 stream priority 复制过来的。

**cudaGraphInstantiateFlagUpload**

Instantiate 完成后，立刻在指定的 stream 上做 upload。这是 cudaGraphInstantiateWithParams 才有的选项。

## Upload：把准备成本移出关键路径

Upload 是一个可选但有时很重要的步骤。

API 是 cudaGraphUpload。它的作用是把 GraphExec 上传到 device，完成一些准备工作，但不实际执行。

为什么需要这个步骤？

问题在于：第一次 launch GraphExec 往往比后续的 launch 更慢。根据 NVIDIA 博客描述，示例场景下第一次 launch 可能比后续慢 33% 左右。这通常与首次执行需要完成额外的初始化、准备与上传相关工作有关。

如果我们的应用对延迟抖动敏感（比如追求稳定的 P99 延迟），这种首轮抖动是不可接受的。Upload 提供了一种解决方案：把上传工作提前做好，从关键路径移出去。

```
// 在关键路径之外预先 upload
cudaGraphUpload(exec, prepStream);
cudaStreamSynchronize(prepStream);

// 关键路径中直接 launch，可显著降低首轮抖动
for (int i = 0; i < iterations; i++) {
    cudaGraphLaunch(exec, workStream);
}
```

Upload 的排序语义：

- 同一个 GraphExec 的 uploads 会被串行化
- 每次 upload 排在 upload stream 之前工作的后面
- 每次 upload 也排在该 GraphExec 之前 launches 的后面

CUDA Samples 中的 cudaGraphsPerfScaling 样例专门量化了 instantiate、first launch、repeat launch、upload 等阶段的成本，是理解这些阶段性能特征的一个很好的参考。

## Launch：真正的执行

Launch 是把 GraphExec 提交到 GPU 执行的操作。

API 是 cudaGraphLaunch。每次调用会把图中描述的所有操作按照依赖顺序提交执行。

### 排序语义

每次 launch 会与两个方向上的工作形成顺序：

- 排在 launch stream 中之前工作的后面
- 排在同一个 GraphExec 之前 launches 的后面

这意味着：如果我们在同一个 stream 上连续 launch 同一个 GraphExec 两次，第二次 launch 必须等第一次完成。

### 并发限制

这是一个非常重要的语义：**同一个 GraphExec 在任一时刻只能有一个实例在执行**。

换句话说，即使我们把同一个 GraphExec 同时 launch 到两个不同的 stream 上，它也不会并发执行。如果想要并发执行同一个计算流程，必须从同一个 cudaGraph_t 实例化出多个 cudaGraphExec_t，每个 GraphExec 可以独立 launch。

这个限制在设计系统时非常重要。如果我们的应用需要多路并发执行同样的图结构，需要维护多个 GraphExec 实例。

### Alloc/Free 相关的执行条件

如果 GraphExec 包含 alloc node，并且上次 launch 产生的 allocations 还没有被释放，会影响下次 launch：

- 如果 instantiate 时设置了 AutoFreeOnLaunch flag，下次 launch 之前会自动释放这些 allocations
- 如果没有设置这个 flag，下次 launch 会失败，返回 cudaErrorInvalidValue

这意味着使用图内 alloc/free 的时候，需要明确管理 allocation 的释放时机。

## Destroy：资源释放

当 GraphExec 和 Graph 不再需要时，应该显式销毁它们。

```
cudaGraphExecDestroy(exec);
cudaGraphDestroy(graph);
```

一个常见的工程问题是：如果在循环中反复 capture/instantiate 而不销毁，会导致 graph 对象堆积，产生资源泄漏或 footprint 不可控的问题。

有些 CUDA Samples 的样例更偏向展示某个机制（例如更新策略对比），资源释放与对象生命周期管理未必覆盖生产环境所需的完整闭环。如果直接迁移到长期运行的系统里，通常需要补齐统一的 destroy 逻辑。

---

## Device Launch：从 GPU 侧启动图

Device Launch 是一个高级能力：允许从 device 侧的 kernel 中启动 GraphExec。

要使用这个能力，需要在 instantiate 时设置 cudaGraphInstantiateFlagDeviceLaunch flag。设置后得到的 GraphExec 可以从 host 和 device 两侧启动。

Device Launch 有一系列额外的约束：

**节点类型限制**

图中只能包含以下类型的节点：

- kernel node
- memcpy node
- memset node
- child graph node

其他类型的节点（例如 host node、event node、alloc/free node 等）不能出现在 device-launch 的图中。

**单 device 限制**

图中的所有节点必须位于同一个 device/context。

**非空限制**

图不能为空，必须至少包含一个 kernel、memcpy 或 memset 节点。

**Kernel 限制**

- 不允许 CUDA Dynamic Parallelism（CDP）
- cooperative launch 在某些环境下存在额外限制（例如与 MPS 的交互限制），需要按 CUDA 文档约束使用

**Memcpy 限制**

- 只允许 device memory 和 pinned device-mapped host memory 之间的拷贝
- 不允许涉及 CUDA arrays 的拷贝
- 源和目标必须可被当前 device 访问，且与图中其他节点的 device 一致

另外，DeviceLaunch 和 AutoFreeOnLaunch 不能同时使用。

---

## Warmup 与 Steady State 的区分

在性能测量和系统设计中，需要区分 warmup 阶段和 steady state 阶段。

Warmup 阶段通常包括：

- Capture（如果使用 stream capture）
- Instantiate
- 第一次 Launch（往往包含 upload/准备成本，除非提前调用了 cudaGraphUpload）

Steady State 阶段是：

- 后续的重复 Launch

这两个阶段的成本差异可能很大。在做性能评估时，需要把 warmup 成本与 steady state 成本分开测量，否则很容易得出误导性的结论。

工程上的常见处理方式包括：

- 测量时丢弃前几次迭代
- 分别报告 warmup 成本与 steady state 成本
- 使用 cudaGraphUpload 把 upload/准备成本从 first launch 移出

---

## 错误处理与同步

### 实例化时的错误

Instantiate 可能因为各种原因失败：图结构不合法、节点参数不合法、违反约束条件等。

cudaGraphInstantiateWithParams 提供了更详细的错误诊断信息，通过 result_out 和 errNode_out 可以定位：

- 错误类型（结构错误、参数错误、不支持的节点类型等）
- 哪个节点出了问题

### Launch 时的错误

Launch 的失败原因不止一种。对于工程系统来说，最需要重点关注的是：

- 含 alloc node 的图：存在未释放的 allocations 且没有设置 AutoFreeOnLaunch 时，后续 launch 会失败并返回 cudaErrorInvalidValue
- 其它来自参数/资源/上下文的错误，以及之前异步执行产生的错误回传（见前述“异步错误回传”）

### 同步点与错误归因

由于异步错误回传的存在，建议在关键阶段（capture 后、首次执行后、更新后）设置明确同步点，以便把错误更稳定地归因到更接近根因的位置。

# 静态性的三层含义

CUDA Graph 的很多设计决策和使用限制，都源自同一个根本特征：**静态性**。

理解静态性，就理解了为什么 CUDA Graph 对变化这么敏感，也理解了为什么框架在落地 CUDA Graph 时需要做那么多额外的工作。

## 什么是静态性

CUDA Graph 从一诞生就是一个静态的抽象。所谓静态，指的是对一个**已实例化的可执行图**（GraphExec）而言：它在实例化时就确定了拓扑与关键执行参数，后续执行主要是按照这个固定蓝图反复回放。

这种静态性是 CUDA Graph 能够优化控制面开销的前提：正因为图是固定的，GPU 侧才能预先规划好执行顺序、预先准备好资源，从而减少每次 launch 时的决策和准备工作。

但静态性也带来了约束：当现实世界需要变化时，静态的图无法直接适应。这就是 CUDA Graph 复杂性的主要来源。

静态性可以拆成三个层次来理解：拓扑静态、参数形态静态、地址静态。

## 第一层：拓扑静态

拓扑静态是最直观的一层约束：对一个**已实例化的 GraphExec** 来说，节点集合与依赖关系在实例化时就确定了，后续不能在每次 replay 时随意改变。

一个 cudaGraph_t 包含若干个节点，节点之间通过依赖边连接形成 DAG。需要注意的是：

- cudaGraph_t（图定义）在实例化前通常可以通过 API 增删节点与依赖关系；但某些节点类型（例如图内 alloc/free）会带来更强的结构性限制，使得删除节点/边等操作被禁止。
- 一旦实例化得到 GraphExec，单次 replay 执行的拓扑就固定了；如果要改变拓扑，通常需要重新实例化，或尝试 exec update（失败则回退到重建）。

因此，在以 GraphExec 为单位讨论时，可以把它理解为拓扑固定：

- 不能在 replay 时动态增删节点
- 不能在 replay 时改变依赖关系
- 每次 launch 复用的是同一组节点与依赖约束（DAG 不变；并行调度与具体执行次序不应被当作可依赖的固定序列）

这意味着：如果我们的程序有动态控制流（比如根据某个条件决定执行哪些 kernel），这种动态性与 CUDA Graph 的拓扑静态性天然冲突。

### 框架如何应对

面对动态控制流，常见的应对策略包括：

**Graph Break / Partition**

把计算流程切分成多个片段。每个片段内部结构稳定，可以做成图；片段之间通过普通的 stream 执行连接。遇到无法确定的控制流点就断开，回到 eager 执行。

torch.compile 的 graph partition 机制就属于这类策略：把 CPU 相关的操作、不确定的控制流等拆分出去，剩余的 GPU 子图做 cudagraphify。

**多图缓存**

为不同的执行路径分别创建图，运行时根据条件选择对应的图来 launch。

这需要一个静态键（static key）来区分不同的路径。比如根据 batch size 或 input shape 来选图。如果可能的路径太多，图的数量会爆炸。

**Conditional Nodes**

conditional nodes 提供了一种在图内部表达控制流的能力。条件值由 device 侧写入，图在执行过程中会根据该条件值选择性执行某些子图路径（具体可表达的控制流形态取决于 CUDA 版本与节点能力边界）。

但 conditional nodes 有自己的约束：版本要求较高、与其他节点类型有组合限制（比如不能放在 child graph 里）、条件值必须由 device 侧写入等。

## 第二层：参数形态静态（By-Value Capture）

**By-value capture** 是 CUDA Graph 静态性问题中最容易被低估、也最容易导致问题的一层。

当我们通过 stream capture 或显式构图创建节点时，节点的参数（kernel 参数、memcpy 的源地址和目标地址、大小等）会被按值捕获。也就是说，capture 时这些参数是什么值，就被固化到了图里。

后续 replay 时，图执行的是这些被固化的值，而不是当前这些变量的值。

### 一个典型的问题场景

考虑这样一个场景：

```
size_t bytes = 1024;
dim3 grid(1);
dim3 block(1);

float* buffer = nullptr;
cudaMalloc(&buffer, bytes);

cudaGraph_t graph = nullptr;
cudaGraphExec_t exec = nullptr;

cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
kernel<<<grid, block, 0, stream>>>(buffer /*, ... */);
cudaStreamEndCapture(stream, &graph);

cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0);

cudaFree(buffer);
cudaMalloc(&buffer, bytes);

cudaGraphLaunch(exec, stream);
```

这就是 by-value capture 的风险：capture/构图时记录的**指针值**，在 replay 时可能已经失效。如果系统没有做额外处理，可能导致 silent corruption（结果错误但不报错）或 crash。

同时也要注意一个常见误解：对指针参数而言，CUDA Graph 记录的是指针值本身；指针指向的内存内容并不会在 capture 时被冻结。因此，只要地址稳定，我们仍然可以在每次 replay 前写入不同的数据内容来改变计算结果。

### 更隐蔽的场景：CPU scalar

PyGraph 论文中提到的一个典型案例是 CPU scalar 混入 GPU 计算。

如果一个 Python/NumPy scalar 以 by-value 的形式参与图内计算（例如作为 kernel 参数，或被编译系统内联/特化为常量），它在图里的取值可能会变成被固化的值。如果语义上这个 scalar 应该在每次迭代变化（比如 learning rate 衰减），但图里用的始终是 capture 时的值，就会产生逻辑错误。

在 torch.compile 的实际工程路径中，这类情形也常常表现为：由于存在 CPU 侧输入/不安全依赖，整段计算直接 skip cudagraph，或者需要重新编译/重录来反映新值。

这类问题往往很难追踪，因为程序不会 crash，只是结果不对。

### 框架如何应对

面对 by-value capture 的问题，框架通常采用 **placeholder + copy** 的策略：

1. 在 capture 时，把可变参数替换成指向固定地址的 placeholder
2. 这样图里记录的是 placeholder 的地址，而不是真实数据的地址
3. 每次 replay 之前，把真实数据拷贝到 placeholder 指向的地址
4. 图执行时读取的是 placeholder 上的新值

这个策略解决了 by-value capture 的问题，但引入了新的成本：每次 replay 都需要做数据拷贝。

PyGraph 论文的统计数据显示：在其评估的 PyTorch2 CUDA Graph 实现与工作负载中，graph replay 的额外开销里有相当一部分来自参数数据拷贝（文中给出约 60% 的量级）。这是一笔很容易吞噬收益的成本。

## 第三层：地址静态

地址静态是 placeholder 策略能够工作的前提条件，也是框架落地 CUDA Graph 时需要重点管理的问题。

地址静态的意思是：参与 capture 的 CUDA 内存地址需要保持稳定。更准确地说，是地址固定、值可变。

placeholder 策略之所以能工作，是因为 placeholder 的地址是固定的。图里记录了 placeholder 的地址，每次 replay 前只需要把新值写到这个固定地址上。

但如果 placeholder 本身的地址也变了（比如 allocator 的行为导致地址变化），问题就又出现了。

### 地址稳定的来源

地址稳定可以有几种实现方式：

**调用方保证 Static Inputs**

某些输入在一个较长的运行窗口内可以保持稳定，比如模型的权重 tensor。如果调用方能保证这些 tensor 的 data_ptr 在运行期间保持不变，它们就可以直接作为静态输入参与 capture，不需要 placeholder。

torch.compile 有 static_input_idxs 机制来标记哪些输入是静态的。

**系统分配固定地址 Buffer**

对于可变的输入，系统分配一块固定地址的 buffer 作为 placeholder，每次 replay 前把真实数据 copy 到这个 buffer。

这是最常见的策略，代价是 copy 开销。

**Mempool / Graph Pool**

把 allocator 的行为收敛到一个可控的内存池（或在捕获时显式指定 pool），使得 capture 期间相关的分配在 replay 路径中具备可复用、可预测的地址语义。

PyTorch 的 cudagraph 实现使用了私有的 graph pool；vLLM、SGLang 等推理系统也通常会复用/共享一个 graph pool（或等价的全局 pool 概念）来提升地址稳定性与显存复用。

**图内 Alloc/Free Nodes**

CUDA Graph 原生的 alloc node 返回的地址跨 instantiation 和 launches 保持固定。这是 CUDA 提供的一种图内地址稳定机制。

但使用 alloc/free nodes 会带来结构刚性（不能删节点、不能 clone、只能单 instantiation 等），很多框架反而选择用外部 pool + 固定 buffer 的方式。

## 三层静态性的交互

这三层静态性不是独立的，它们相互交织：

**拓扑变化会触发参数问题**

如果我们的工作负载需要频繁改变节点与依赖关系，节点的参数结构也会随之变化，by-value capture 的假设就更难维护。

**参数形态变化影响地址需求**

如果参数的数量或类型变化（比如从 3 个参数变成 4 个），即使地址稳定也没用，因为整个参数结构变了。

**地址不稳定导致 capture 失效**

如果 capture 期间记录的地址在 replay 时失效，图就无法正确执行，需要重新 capture 或使用其他更新机制。

框架在设计 CUDA Graph 支持时，通常需要同时处理这三层约束：

- 通过静态键和多图缓存处理拓扑变化
- 通过 placeholder + copy 处理参数值变化
- 通过固定 buffer 和 pool 管理处理地址稳定

## 静态性如何驱动系统设计

理解了静态性的三层含义，就更容易理解为什么 torch.compile、vLLM、SGLang 这些系统会采用类似的设计模式：

**静态键**。用一个可哈希的 key 来区分可以复用同一张图的调用类别。key 太细则图数量爆炸，太粗则需要更多 padding。

**固定地址 Buffer**。各个系统都有replay 前把真实数据投影到静态 buffer的步骤，这是 placeholder + copy 策略的具体实现。

**状态机**。把 CUDA Graph 的使用组织成 warmup → record → replay → fallback 的状态机，在满足条件时享受收益，不满足时有降级路径。

## 动态形状的挑战

静态性约束在动态形状（dynamic shape）场景下尤其棘手。

深度学习中，batch size、sequence length 等维度经常变化。每种形状组合如果都独立 capture 一张图，图的数量可能非常大。

常见的应对策略包括：

**Padding 到固定形状**

把不同大小的输入 pad 到几个固定的形状类别上。比如 batch size 1-8 都 pad 到 8，9-16 都 pad 到 16。

这减少了图的数量，但引入了 padding 开销和无效计算。vLLM 的 cudagraph_metrics 记录 padded/unpadded tokens，就是为了量化这个 overhead。

**多图缓存 + 规模限制**

缓存若干个常见形状的图。第一次遇到某个形状时 capture，后续命中缓存。

但需要限制缓存的图数量，否则资源占用会失控。torch.compile 有动态 shape 录图的 warn limit 和 skip 策略。

**只对稳定路径 Graph 化**

对变化频繁的部分不使用 CUDA Graph，只对结构稳定的部分使用。

这需要有能力识别哪些部分是稳定的。torch.compile 的 graph partition 和选择性部署就属于这类策略。

## 生命周期与 Liveness 问题

静态性的另一个延伸是输出的生命周期管理。

当 graph 的输出 buffer 被复用时，如果上一轮的输出还在被外部持有，下一轮 replay 可能会覆盖它。

这种问题在共享 pool 的场景下尤其突出。torch.compile 的 Trees 会追踪 outputs 是否仍然 live，并在某些条件下阻止 replay。它还提供了 torch.compiler.cudagraph_mark_step_begin() 来显式标记 iteration 边界。

Megatron 采用的策略是：对最后一层的输出做 clone + detach，把可复用的内部 buffer与用户可持有的输出分离。

vLLM 使用 weak reference 来减少对图输出的长期持有，但也记录了某些场景可能出现输出过早释放需要 workaround 的问题。

## 与 Allocator 行为的交互

静态性约束还会与 allocator 的状态交互。

capture 期间，allocator 的 bookkeeping 会被正确记账。但 replay 期间，只有 GPU 侧的 kernel 执行，CPU 侧的 allocator bookkeeping 不会随 replay 自动重放/回滚到某个一致的检查点状态。

这意味着：如果系统希望在 replay 之后继续进行新的 capture，或在共享同一 pool 的前提下继续进行复杂的分配/复用，allocator 的逻辑状态就必须被显式管理，否则可能出现不安全的复用风险。

torch.compile 的 Trees 通过 checkpoint allocator state 的机制来处理这个问题，使得 replay 后仍能继续录图，且在共享 pool 情况下避免不安全的内存复用。

# 性能收益与对冲代价

CUDA Graph 的性能影响可以用一个简单公式理解：

```
净收益 = 省下的控制面开销 - 引入的额外成本
```

只有当省下的比引入的多，Graph 才是正收益。前面介绍过微基准数据：从调整同步后的 3.8μs 到使用 Graph 的 3.4μs，增量收益约 10%。收益显著的条件是短 kernel 密集、高重复度、控制面是瓶颈。

## 代价有哪些

代价分两类。

**一次性成本**包括 capture、instantiate、首次 launch。NVIDIA 博客的示例数据是 capture + instantiate 约 400μs 量级，首次 launch 比后续慢约 33%。这些需要足够多的重复 launch 来摊薄。

**每次 replay 成本**包括 graphLaunch 本身的开销，以及框架层面为满足静态性约束而做的额外工作：把可变参数拷贝到 placeholder、输入投影和 padding、状态检查等。这些开销会直接对冲 Graph 带来的收益。

## 框架场景：并非总是正收益

PyGraph 论文（2025 年 arXiv 预印本）对 PyTorch 2 中 CUDA Graph 的实际效果做了统计，样本是 183 个应用、416 个 Graph：

| 效果          | 图数量 | 占比 |
| ------------- | ------ | ---- |
| 带来 >2% 提升 | 191    | 46%  |
| 几乎无收益    | 82     | 20%  |
| 变慢          | 143    | 34%  |

超过三分之一的图实际上让性能变差了。

变慢的核心原因是参数拷贝开销。论文指出，在其评估的实现与工作负载中，graph overhead 的很大一部分来自参数数据拷贝（约 60% 量级）。为了让图能正确 replay，框架需要在每次 replay 前把可变数据拷贝到 placeholder，这个拷贝开销在某些情况下会超过 Graph 省下的控制面开销。

几个容易亏的场景：图太小（省下的 launch overhead 不够）、参数拷贝量大、复用次数不够（一次性成本摊不掉）、kernel 本身很重（控制面占比本来就低）、闭源 kernel 多（无法应用 parameter indirection 优化）。

## 要点

**区分 warmup 和 steady state**。测量时把 capture、instantiate、首次 launch 的成本单独计算，不要混入后续 replay 的统计。

**用 Upload 治理尾延迟**。首次 launch 比后续慢，如果我们的应用对 P99 敏感，可以在关键路径之外提前调用 cudaGraphUpload，把准备成本移出去：

```
cudaGraphUpload(exec, prepStream);
cudaStreamSynchronize(prepStream);
// 关键路径中直接 launch
cudaGraphLaunch(exec, workStream);
```

**量化 padding 开销**。在处理动态形状时，padding 到固定 bucket 会引入无效计算。vLLM 的 cudagraph_metrics 记录 padded/unpadded tokens 比例，可以用来评估这笔交易是否划算。

**选择性部署**。不要把 CUDA Graph 当全局开关。对每张图独立评估，有些图值得用、有些图不用更好。torch.compile 提供了 skip reason 日志和计数器，vLLM 提供了 cudagraph_metrics，这些可观测性工具是做出正确决策的基础。

# 主流框架的使用模式：torch.compile

torch.compile 与 CUDA Graph 的集成不是简单的把整个模型录成一张图。实际的机制要复杂得多。

## 在哪里录图

torch.compile 的编译流程是：Python 代码 → Dynamo trace 成 FX Graph → Inductor 生成 Triton/C++ kernel → 产出一个 callable。

CUDA Graph 的 capture 不是发生在编译阶段，而是发生在 **post_compile 阶段**：Inductor 产出的 callable 被一个叫 cudagraphify 的 wrapper 包起来。这个 wrapper 负责在运行时决定是 warmup、record 还是 replay。

这意味着：编译产物本身不是 CUDA Graph，而是**可以被 capture 成 CUDA Graph 的素材**。capture 是延迟到首次真正执行时才发生的。

## CUDAGraph Trees：解决单图方案的局限

默认的实现叫 CUDAGraph Trees。它解决了三个问题：

- **graph breaks**：Dynamo trace 时产生的多个片段可以各自 capture
- **forward-backward 协调**：训练时 forward/backward 分别录图，Trees 管理它们的 pool 共享和 liveness
- **replay 后继续录图**：通过 checkpoint allocator state 机制使得 replay 后仍能安全地录新图

Trees 采用前面描述的状态机模式（warmup → record → replay → skip），关键决策点是 invariants 检查：地址是否稳定、liveness pattern 是否满足。

对于 static inputs（weights/buffers），直接使用地址不需要 copy；对于 non-static inputs，使用 placeholder + copy 策略。static input 越多，copy 开销越小，但地址稳定要求越强。

## 动态 shape 的处理

CUDA Graph 需要固定的参数形态，不同 shape 意味着不同的 kernel 参数。Trees 的策略是：**为每个 distinct shape key 录一张 Graph**。

shape key 通常来自运行时 inputs 中的 int 值（例如 SymInt/动态尺寸在运行期落地后的整数值），而不是对任意 tensor shape 做一个泛化的 hash。如果 shape 变化太多，会产生大量图，资源消耗增加。Trees 会在图数量超过阈值时发出警告。

如果我们知道只有几个常见 batch size 值得 cudagraph 化，可以通过配置限制只 capture 这些 size。

## Graph Partition：处理不兼容 op

CUDA Graph 不支持 CPU op、device 拷贝、控制流等。如果图中混有这些 op：

- 不开 partition：整张图 skip cudagraph
- 开 partition：图被切分成多个片段，GPU-only 的片段各自 cudagraphify，CPU/unsafe 部分正常执行

这样即使图里混了 CPU op，GPU 子图仍然能享受 cudagraph 收益。

## 为什么会 skip

常见的 skip 原因：

- 输入被原地修改（mutation）且来自 eager 的普通输入
- 存在不兼容 op（在 FORBIDDEN_CUDAGRAPH_OPS 列表中，或带 cudagraph_unsafe 标签）
- 多 device 场景（不开 partition）
- 动态 shape 且配置了跳过动态 shape
- 输入地址不稳定导致 invariants 检查失败
- 图数量超过重录上限

排查 skip 的方法：设置环境变量 TORCHINDUCTOR_CUDAGRAPH_OR_ERROR=1，skip 就会变成报错，而不是 silent fallback。

# 主流框架的使用模式：推理系统（vLLM / SGLang）

推理系统对 CUDA Graph 的使用与 torch.compile 有显著不同。推理场景的特点是：batch size 变化频繁、延迟敏感、同一模型反复调用。这些特点决定了一套不同的设计模式。

## 核心机制：启动期 Capture，运行期 Replay

以 vLLM v1 与 SGLang 的主线路径为代表，推理系统往往采用**启动期预先 capture，运行期只做 replay** 的策略。好处是运行期可以避免 capture 抖动，代价是启动时间更长、启动期显存压力更大。

两个系统都采用**从大到小 capture** 的顺序，让大图先 capture 以复用 memory pool，降低后续 capture 的额外分配压力。

同样采用前面描述的 placeholder + copy 策略：预分配固定地址的 buffer，每次 replay 前把真实数据 copy 进去。SGLang 的典型实现是 GraphInputBuffers；vLLM 在部分模式下会由编译/runner 维护静态输入 buffer，并提供可选的 copy_inputs 策略来满足地址稳定性。和 torch.compile 不同的是，推理系统通常对 buffer 布局与 capture sizes 有更强的工程控制权。

## Padding 与图命中率

实际请求的 batch size 不可能总是和预先 capture 的 size 精确匹配。两种策略：

**启用 padding**（常见默认）：如果实际 bs=5 但最接近的图是 bs=8，就把输入 pad 到 8，replay bs=8 的图，最后裁剪输出。图命中率高，但有无效计算。

**禁用 padding**：只有精确匹配才走图，否则 fallback 到 eager。命中率低，但没有 padding 开销。

vLLM 提供了 cudagraph_metrics 来量化这个 trade-off：padded tokens / unpadded tokens 比例反映了 padding 带来的无效计算占比。

如何选择 capture 哪些 size 是一个工程问题。太少则命中率低或 padding 重，太多则启动慢且显存占用高。两个系统都允许用户指定 capture 的 batch size 列表。

## Full 图 vs Piecewise 图

推理系统面临一个特殊问题：不同的 attention backend 对 CUDA Graph 的支持程度不同。

vLLM 的设计是区分 **FULL**（整图 capture，包含 attention）和 **PIECEWISE**（分段 capture，把 attention 留在 eager）两种模式。

为什么需要 piecewise？有些 attention 实现内部有动态分配或数据依赖的控制流，无法被 capture。piecewise 模式把这些不兼容的部分拆出去，其余 GPU 子图仍然可以享受 cudagraph 收益。

vLLM 会根据当前使用的 attention backend 自动选择合适的模式，或者在 backend 不支持时自动降级。这种设计让系统能在不同硬件和配置下自适应。

## Decode vs Prefill

LLM 推理有两个阶段：prefill（处理 prompt，一次计算多个 token）和 decode（自回归生成，每次一个 token）。

**Decode 阶段**是 CUDA Graph 的主战场：每次 step 的计算拓扑高度一致，非常适合 replay。

**Prefill 阶段**更复杂：不同 prompt 长度不同，形状变化大。多数系统对 prefill 不走 cudagraph 或只走 piecewise。

SGLang 的 decode cudagraph 是主线、且在很多部署配置中会作为默认启用的路径；prefill 的 piecewise cudagraph 则更偏实验性与特定场景。

## 分布式场景的特殊处理

在 tensor parallel 或 pipeline parallel 场景下，capture 需要对通信操作做特殊处理。

问题是：某些通信实现（比如 torch.distributed 的部分操作）在 capture 下不兼容或有性能问题。

两个系统的解决方案类似：在 capture 期间切换到支持 graph 的通信实现（如 pynccl），并在 replay 阶段保持通信语义一致（通常也会继续使用同一实现），以避免 capture/replay 行为不一致。这就是为什么会有 graph_capture 上下文管理器的存在。

## 容易踩的坑

**GPU→CPU 同步**。capture 期间调用 float(tensor) 或 tensor.item() 往往会触发同步，从而破坏 capture 的假设（在一些实现里会直接导致 capture 失败或被迫回退到图外路径）。修复思路：把相关计算与控制逻辑留在 GPU 上，避免在 capture 期间引入 Host 同步点。

**Capture 阶段动态分配**。在 capture 期间创建新 tensor 往往会引入额外分配与地址不稳定问题，影响图的可复用性（在一些实现里也可能导致 capture 失败或引入难以控制的抖动）。修复思路：预分配 buffer，并在 replay 前用 copy/slice/view 等方式填充与映射数据。

**分支不一致**。Capture 时走 A 分支，replay 时走 B 分支。修复：在 capture 模式下固定分支选择，不依赖运行时数据决定分支。

## 与 torch.compile 的关键差异

| 维度         | torch.compile             | 推理系统                   |
| ------------ | ------------------------- | -------------------------- |
| Capture 时机 | 运行时首次遇到            | 启动期预先 capture         |
| 形状缓存     | 运行时按 shape key 动态录 | 预先决定 capture 哪些 size |
| Buffer 管理  | 框架自动分配 placeholder  | 手工预分配，完全可控       |
| 模式选择     | 自动 skip 不兼容 op       | 显式区分 FULL/PIECEWISE    |

推理系统的设计更手工，但换来的是更可预测的行为和更低的运行时抖动。
