---
title: vLLM Model Runner V2 设计文档：从 Persistent Batch、Async-First 到 Triton Native Sampler
slug: model-runner-v2-design-document
date: '2026-03-25'
tags: ['vLLM']
status: published
comments: true
source_author: Woosuk Kwon, Nick Hill
---

这篇文章整理并翻译自 [Model Runner V2 Design Docs](https://docs.google.com/document/d/1gFqtDkcoqhy9j-X0ndshzbhapX1uNey1-wBENwGPI80/edit?usp=sharing)。原文聚焦于 vLLM 新一代 `Model Runner V2` 的设计动机、核心改造点与仍在演进中的一些方向。

为了便于在博客中阅读，我保留了原文的章节结构、代码示例和核心术语，并将部分表达做了中文技术语境下的顺滑处理。

## 引言

自从 vLLM V1 首次实现以来，我们逐渐发现其中存在一些根本性的设计失误，也积累了相当多的技术债。很多后续功能都是在原始设计没有充分考虑的情况下不断“补丁式”加上去的。与此同时，我们也获得了许多新的经验，包括：

- 采样技术，例如 Gumbel-max sampling
- 工具链，例如 Triton
- CUDA 特性，例如 UVA

基于这些经验，我们从第一性原理出发重新实现了 Model Runner V2（MRV2），目标是让它更干净、更高效，也更模块化。

事后看，V1 里很多设计选择其实都不理想。虽然 MRV2 目前还没有完全补齐所有功能，也没有经过足够严格的测试，并且仍有一些开放性的设计问题，但我们相信它相比 V1 已经是一次明显的进步。

本文描述 MRV2 的整体设计。也欢迎大家提出建议、问题和评论。

## 1. Persistent Batch

V1 中一个非常明显的摩擦点，是它的 persistent batch 实现。

### 背景

V1 引入 persistent batch，主要是为了降低输入准备阶段的 CPU 开销。当请求在某一步被调度执行时，model runner 需要构造一组连续的输入张量，例如：

- block tables
- 每个请求对应的 temperature 值

这些张量会被送入模型执行。若每一步都从头在 Python 里构造它们，通常会很慢，尤其是像 block table 这种大张量。虽然在今天这件事因为 async scheduling 的存在没有以前那么敏感，但它仍然是一个实际问题。

persistent batch 的优化思路是：相邻两个 step 之间，请求批次通常几乎相同，每一步往往只有极少数请求加入或结束。于是我们可以维护一组“持久状态张量”，每步只应用增量 diff，而不是完全重建整份输入，从而显著降低 CPU 开销。

### V1 方案的问题

虽然这个优化在效率上成立，但 V1 的 persistent batch 设计由于一个根本性失误而引入了额外复杂度：它把 persistent state 和 input tensor 耦合在了一起。

V1 直接把 persistent state tensor 作为模型与 sampler 的输入，这会带来很强的约束。例如：

- paged attention kernel 要求 block table 的每一行在内存中连续存放
- 某些场景下还要求请求顺序满足特定排列，例如 decode request 排在 prefill request 前面

为了满足这些约束，persistent state 的管理就变得非常复杂。一个请求结束或新请求加入时，常常不是简单地增删一行，而是需要对整个 state tensor 做复杂重排。

更糟糕的是，这种设计迫使 V1 维护 `CachedRequestState`，也就是在 persistent state tensor 之外，再维护一份冗余的请求状态副本。之所以必须这样做，是因为 persistent tensor 中的某些行可能在对应请求尚未完成时就被覆盖。

例如：

- step N 中 scheduler 处理请求 A 和 B
- step N+1 中调度的是 A 和 C，而 B 实际上还没有结束
- 在重排过程中，B 对应的行就可能被覆盖

`CachedRequestState` 就成了这种场景下的兜底备份，因此每次请求状态更新时都需要做重复工作。

最终结果就是，V1 的 persistent batch 在 bookkeeping 上变得极其复杂。而一旦和 async scheduling 结合，这种复杂度几乎会失控。

### MRV2 的解决方案

MRV2 的处理方式很直接：把 persistent state tensor 和 input tensor 解耦。

在每个 step 中，给定一个特定的请求顺序（通常由 attention backend 决定），我们从 persistent state 中 gather 出新的 input tensor。这样一来，persistent state 的管理会简单很多：

- 预分配一个固定大小的 tensor，行数为 `max_num_reqs`，在大多数平台默认是 1024
- 为请求分配“永久行号”。新请求加入时，只需放进任意一个空行
- 请求会一直待在这行里，直到结束或被抢占，不再需要复杂重排
- 把 preemption 当成 completion 处理。请求一旦被抢占，就彻底移除其行；恢复执行时，由 scheduler 重新发送完整请求数据，再像新请求一样加入 persistent state tensor

这就完全消除了对 `CachedRequestState` 的需求，bookkeeping 因而大幅简化。

另外，正如后文会讲到的那样，MRV2 将大多数较大的 state tensor 存在 GPU 内存中，因此 gather 操作也是由 GPU 并行完成，额外开销很小。

## 2. Async-First

在 V1 写出来之后的这段时间里，我们已经彻底转向异步调度。

scheduler 与 worker 在 GPU 执行 step N 的同时，就会准备 step N+1 的输入。通过把 CPU 的准备工作和 GPU 的执行阶段重叠起来，异步执行可以隐藏 CPU 开销，并让 GPU 保持尽可能高的利用率。

但 V1 并不是围绕 async scheduling 设计的。虽然它现在已经默认启用，但我们为了完整支持它，付出了很多“不自然”的 hack，本质上是在勉强适配既有实现。

MRV2 从一开始就是按 async-first 的方式写的。它的核心假设是：模型执行主循环本质上只是一条 CUDA stream，中间不应该和 CPU 有同步点。CPU 侧暴露给 model runner 的入口，只负责把操作排队到这条 stream 上。像 speculative decoding 这样的特性，也都是基于这个前提实现的。

## 3. 移除 Async Barrier

### 异步执行中的竞态条件

要实现真正的异步执行，一个关键要求是：所有 CPU 侧操作都必须是 non-blocking 的，也就是说 CPU 绝不能等待 GPU。

我们既要避免显式同步，例如 `torch.cuda.synchronize()`，也要避免隐式同步，例如 `unpinned_cpu_tensor.to("cuda")`。否则，CPU 和 GPU 的重叠执行就无法真正发生。

但异步执行也会引入一个新问题：竞态条件。更准确地说，当 CPU 和 GPU 同时读写同一段内存地址时，就可能出现 race condition。

下面是一个最小例子：

```python
class ModelRunner:
    def __init__(self, ...):
        # 在 CPU 上分配一个 pinned memory buffer
        self.states = torch.zeros(
            max_num_reqs,
            dtype=torch.int32,
            device="cpu",
            pin_memory=True,
        )

    def execute_step(self, ...):
        ...
        # 写入新请求的数据
        self.states[req_idx] = new_req.data
        ...
        # 异步拷贝到 GPU（触发 cudaMemcpyAsync）
        states = self.states.to("cuda", non_blocking=True)
```

这里我们在 CPU 上分配了一个静态 `states` buffer，用来跟踪每个请求的值。由于使用了 pinned memory，拷贝到 GPU 时可以做到 non-blocking。

乍看之下，这段代码没有问题；但如果把执行循环展开，就能发现一个 race condition：CPU 可能在 GPU 仍通过 `cudaMemcpyAsync` 读取 `self.states` 时，又写入了这块 buffer。也就是说，这段代码其实并不安全。

### V1 的 async barrier 方案

V1 用一个“barrier”来处理这个问题：

```python
with async_barrier(self.event):
    # 保护输入准备阶段，防止 race condition
    self.states[req_idx] = new_req.data
    ...
    states = self.states.to("cuda", non_blocking=True)
```

这个机制会在 critical section 末尾记录一个 CUDA event，并在下一次进入时调用 `event.synchronize()`。这样就能保证，在 GPU 完成上一步拷贝之前，CPU 不会去修改 `self.states`。

但这种做法有几个明显问题：

- 容易出错：很难准确识别哪些 buffer 需要保护，V1 曾经出现过多次静默 race bug，就是因为开发者漏掉了某些 buffer
- 不够灵活：所有 CPU 相关工作都必须塞进 `async_barrier` 语句里，代码组织受限
- 会影响性能：同步会压缩 CPU-GPU overlap，理论上可能损害性能，虽然在实践中通常不是最严重的问题

### MRV2 的方案：从根上消除 race condition

MRV2 采用了更简单的方式：直接消除这类竞态，而不是靠 barrier 保护。

做法是把状态张量与真正拷贝到 GPU 的张量分开：

```python
class ModelRunner:
    def __init__(self, ...):
        # 在 CPU 上分配 buffer，但不使用 pinned memory
        self.states = torch.zeros(
            max_num_reqs,
            dtype=torch.int32,
            device="cpu",
            pin_memory=False,
        )

    def execute_step(self, ...):
        ...
        # 写入新请求的数据
        self.states[req_idx] = new_req.data
        ...
        # 先生成一个临时 pinned 副本，再异步拷贝到 GPU
        tmp_states = self.states.pin_memory()
        states = tmp_states.to("cuda", non_blocking=True)
```

现在，CPU 修改的是 `self.states`，GPU 读取的是 `tmp_states`，两者已经不是同一块 buffer，因此 race condition 从根本上消失了，也不再需要显式同步。

## 4. StagedWriteTensor

对于 block table 这样的大张量，我们希望避免每一步都把整张 tensor 从 CPU 完整拷贝到 GPU。为此，MRV2 引入了 `StagedWriteTensor`。

它的核心思路是：

- “原始张量”放在 GPU 上，而不是 CPU
- 所有 diff 先在 CPU 上暂存
- 把 diff 打包成连续张量
- 将打包后的 diff 拷贝到 GPU
- 启动一个 kernel，把这些 diff 应用到原始张量上

下面是示例：

```python
# 初始化状态，默认全 0
state = StagedWriteTensor(size=(1024, 1000), dtype=torch.int32, device="cuda")

# 在第 2 行，从索引 3 开始写入 [3, 1, 2]
state.stage_write(row=2, start=3, values=[3, 1, 2])

# 在第 0 行，从索引 1 开始写入 [-1, -2, -5]
state.stage_write(row=0, start=1, values=[-1, -2, -5])

# 通过单次 kernel launch 应用所有 staged changes
state.apply_write()

# GPU 上最终得到：
# [[0, -1, -2, -5,  0,  0, ...],
#  [0,  0,  0,  0,  0,  0, ...],
#  [0,  0,  0,  3,  1,  2, ...],
#  ...]
```

`StagedWriteTensor` 让我们能够把不规则的修改应用到一个固定 GPU buffer 上，并且做到：

- 不需要任何 CPU-GPU 同步
- 只需一次 kernel launch

这对 block table 一类状态管理尤其有用。

此外，`StagedWriteTensor` 也适用于那些既可能被 CPU 写入、也可能被 GPU 写入的状态，例如 `num_computed_tokens`。

对这个状态来说：

- CPU 会在新请求创建时初始化它
- GPU 之后每一步按 `num_sampled` 递增更新它

因为 CPU 并不总能提前知道 `num_sampled` 的确切值，例如在 speculative decoding 中就是如此。`StagedWriteTensor` 很优雅地解决了这个 write-write conflict：CPU 的初始化最终也是通过 GPU kernel 完成，所以所有写操作都在 GPU 上按顺序执行。

## 5. GPU 原生的输入元数据准备与输出处理

MRV2 更激进地使用 Triton kernel 来做输入与输出处理。例如，它会用 Triton kernel 来准备以下输入张量：

- `input_ids`
- `positions`
- `query_start_loc`
- `seq_lens`

这样做有两个主要好处。

### 让 async scheduling 更自然

在使用 async scheduling 且启用 speculative decoding 时，CPU 往往并不知道 `seq_lens` 和 `positions` 的精确值，因为它此时还不知道前一步中到底有多少 draft token 会被拒绝。

在 MRV2 中，这不再是问题，因为这些输入张量由 GPU 构造，而 GPU 正好掌握这些信息。

### 降低 CPU 开销

输入准备在 GPU 上通常极其便宜，往往小于 10 微秒；但如果放在 CPU 上做，Python 开销和缺乏并行性会让它变得不那么划算。

因此，把这些工作迁到 GPU 上，可以保证在未来即便面对像 B300、Rubin 这样的更强硬件时，输入准备也不会变成瓶颈。

### Universal Virtual Addressing（UVA）

为了简化 GPU kernel 的输入处理，我们有时会使用 Universal Virtual Addressing（UVA）。

例如，我们会把 `prefill_token_ids` 存在一个形状为 `[max_num_reqs, max_model_len]` 的二维 tensor 中，这个 tensor 可能大到数 GB。如果 GPU kernel 想基于它构造 `input_ids`，通常意味着要把 `prefill_token_ids` 也放进 GPU 内存，这会浪费大量显存。

MRV2 的做法是：把这个 tensor 存在 CPU 内存里，再通过 UVA 让 GPU kernel 可以直接访问它。这样就无需在 GPU 上再复制一份超大的 tensor，同时仍然可以让 GPU kernel 完成数据处理与输入构造。

## 6. Triton 原生 Sampler

MRV2 使用 Triton 重新实现了 sampler。几乎所有操作都直接在 Triton 中完成，这让我们可以对数值行为和内存使用做更明确的控制，同时也为更多性能优化留出了空间。

### Gumbel Sampling Kernel

一个重要变化是引入了 Gumbel sampling kernel。

虽然 V1 的 sampler 也使用了另一种形式的 Gumbel sampling，并且是通过 PyTorch 实现的，但 Triton 版本效率更高，因为它跳过了 softmax 计算。

另外，Triton kernel 版本更易管理，因为它是无状态的：它不依赖 PyTorch 的随机数生成器，而是在 kernel 内部基于给定 seed 生成随机噪声。

### 更高效的 Top-K Logprobs

另一个重要变化是 top-k logprobs 的计算方式。

V1 的 sampler 会先为整个 vocabulary materialize 一整份 logprobs，然后再应用 top-k。MRV2 则是直接从 logits 中找出 top-k token，只对被选中的 token 计算 logprobs。

这样做避免了构造 vocab 规模的 logprob tensor，进而降低了 GPU 峰值显存占用。

### 更省内存的 Prompt Logprobs

MRV2 通过更细粒度的 chunking 实现了更节省内存的 prompt logprobs。

V1 会在 batch 内逐条 prompt 处理，这在长 prompt 情况下依然可能造成显存尖峰。MRV2 更进一步，允许在单个 prompt 内部继续 chunk，也就是把一条超长 prompt 切成更小片段，分块计算 logits 和 logprobs。

结合前面提到的 top-k logprob 优化，MRV2 对长 prompt 场景的处理明显更从容。

### 与 Speculative Decoding 的兼容性更好

在 V1 里，多个 sampling 参数和 speculative decoding 并不兼容，根源在于 shape mismatch：

- sampling state tensor 往往是一请求一值，或者一请求一行
- 但在 speculative decoding 中，一个请求可能会对应多个 logits vector，因为它可能有多个 draft token

V1 的做法是把 sampling state “展开”，使其形状匹配 logits tensor。对于 temperature、top-p 这类标量参数，这还算容易；但对 penalties、logit bias 这类复杂参数就很麻烦，因为它们可能是 2D 甚至更高维状态，很难扩展。

MRV2 的 sampler 用了一种更优雅的方式解决这个问题：间接寻址。

它不再从 state tensor gather 出一个连续输入张量，而是让 sampling kernel 直接接收原始 state tensor，并通过 `idx_mapping` 为每个 logits vector 找到正确的状态值。

这样一来，在 speculative decoding 场景中，无论一个请求有多少 draft token，我们只需要调整 `idx_mapping`，不需要变换其他输入或状态。当然，penalties 仍然是一个特例，需要额外处理。

这使得后续要为 speculative decoding 添加新的 sampling 参数或 logits processor，都会容易很多。

## 7. 模块化

我们花了很大力气让 MRV2 保持模块化。

V1 的 model runner 之所以显得非常复杂，一个重要原因是太多功能都堆在同一个文件 `gpu_model_runner.py` 里，不同特性的 buffer、方法和逻辑彼此纠缠。

在 MRV2 中，我们把不同功能相关的代码拆到单独文件里。例如：

- `mrope_utils.py`：维护 MRoPE 相关状态与 kernel
- `penalties.py`：封装 repetition、frequency、presence penalty 的状态与 kernel
- 以及另外 20 多个类似文件

做个对比：

- V1 用大约 5 个文件实现大致等价的功能，其中最长的文件超过 6000 行
- MRV2 则把实现拆到了 27 个文件里，最长文件不到 1000 行

在 model runner 内部，我们还把所有输入统一收敛到一个 `InputBatch` 类中，并尽量减少对 model runner 自身属性的直接访问。这样，模型输入与内部状态之间的边界就更清晰了。

## 8. 不再滥用 dummy_run

在 V1 中，`dummy_run` 承担了太多职责，因此变得异常复杂。它被用于：

- 初始内存 profiling 与 `torch.compile`
- CUDA graph capture
- warmup run，例如 FlashInfer backend 的预热
- EP+DP 场景下的空 DP forward pass

其中 DP 这个用例尤其麻烦：`dummy_run` 必须和 `execute_model` 保持同步，否则模型执行可能会卡死。

MRV2 的处理方式是：

- 让 `execute_model` 本身支持 dummy run，也就是在不影响任何状态（包括 KV cache）的情况下执行模型
- 把初始内存 profiling、warmup、空 DP forward pass 全都重定向到 `execute_model`
- CUDA graph capture 则走一条单独路径，因为这部分逻辑本质上不同

这样做大幅简化了代码，也消除了 DP 场景下 `execute_model` 和 `dummy_run` 不一致导致的 bug。

## 9. 显式的 CUDA Graph 管理

在 V1 中，CUDA graph 的管理复杂、隐式，而且藏得很深。这会带来很高的认知负担：你很难弄清楚：

- graph 是何时 capture 的
- graph object 和 memory pool 是怎么管理的
- graph execution 与 eager mode 是如何切换的
- 更复杂用例下 CUDA graph 该怎么扩展

MRV2 重新审视了这部分设计，引入了 `CUDAGraphManager`，用标准 PyTorch API 显式地捕获并启动完整 CUDA graph。这让核心逻辑重新回到 model runner 的显式层面，更容易理解。

这种显式方式也更利于扩展。例如，MRV2 就利用它把 draft model 的多次 forward pass 一次性 capture 到同一个 CUDA graph 中，这在 V1 的基础设施下很难甚至无法实现。

## 开发哲学

我们希望对 MRV2 后续的改动和新增功能维持更高的代码质量标准。

尤其是在逐步补齐 V1 尚有而 MRV2 仍未完成的功能时，我们会坚持一个原则：必须放回新设计的上下文里，从第一性原理重新思考，而不是为了尽快“有东西可用”就直接把 V1 的实现搬过来。

理想情况下，我们宁可在前期多花一些时间，把方案统一得更干净、更优雅，也愿意通过 PR 反复打磨，而不是匆忙把功能合进去。前面提到的模块化，将会是必须守住的底线。

一个典型例子是 pooling model。尤其是它是否应该像 V1 那样和 generative model 做紧耦合统一，还是应该进一步分离，甚至这种分离是否还应该延伸到 scheduler 层，这些都值得重新思考。
