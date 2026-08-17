---
title: Decode 阶段 Batch 从 1 增大到 64 时，TPOT 会怎样变化？
slug: decode-stage-batch-tpot
date: '2026-08-17'
tags: ['Performance']
cover: /images/others/decode-stage-batch-tpot.webp
status: published
source: original
---

## 1. 问题

在 LLM 的 decode 阶段，把 batch 从 1 增大到 32，进一步增大到 64；考虑 GEMM 的 $M$ 维效率后，单请求 TPOT 会怎样变化？

这里的 batch 指某个 decode iteration 中实际参加计算的活跃 token 数，而不是服务配置中的 `max_batch_size`。

## 2. 核心结论

在固定模型、上下文长度分布、精度、并行拓扑和调度策略下，且每条活跃序列每轮只生成 1 个 token 时，线性层的 $M$ 约等于本轮活跃 decode token 数 $B$。如果每个请求连续参加每轮调度，则单请求 TPOT 近似该轮 iteration latency：

$$
\operatorname{TPOT}(B) \approx T_{\mathrm{iter}}(B)
$$

每轮产生约 $B$ 个 token，因此聚合吞吐近似为：

$$
\operatorname{Throughput}(B) \approx \frac{B}{T_{\mathrm{iter}}(B)}
$$

Batch 从 1 增大到 64 时，TPOT 通常不会增长 64 倍，也不保证严格单调上升：

- 对短上下文 Dense 模型，权重流量常占主导。增大 $B$ 能复用同一组权重，并改善 Small-M kernel 的 tile 利用率与固定开销摊薄，因此 TPOT 常表现为前段近似平台、kernel 切换点可能局部下降、后段缓慢上升。
- 对长上下文 Dense 模型，不同请求的 KV Cache 通常不能共享，KV 扫描量随总上下文长度增加，TPOT 更容易明显上升。
- 对 MoE，Dense 层的全局 $M=B$ 不再等价于单专家 GEMM 的 $M$。每个专家收到的 token 可能仍很少，同时还有 EP all-to-all、路由偏斜和 padding，曲线更依赖具体实现。
- TP/EP 通信的消息量随活跃 token 数增长；小 batch 常由通信启动延迟主导，大 batch 时带宽项更明显。
- 吞吐 $B/T_{\mathrm{iter}}(B)$ 通常提高且边际收益递减，但这不是数学保证。显存不足、调度抖动、通信拥塞或不合适的 kernel 都可能让吞吐平台甚至回退。

因此，可靠的回答不是一个固定倍率，而是一条由 Dense GEMM、KV Cache、通信、路由和调度共同决定的曲线。

![Decode 阶段随 Batch 从 1 增至 64 时，Dense 权重复用、KV Cache 扫描、TP/EP 通信与 MoE 路由共同塑造 TPOT 和吞吐曲线的机制示意图](/images/others/decode-stage-batch-tpot.webp)

## 3. TPOT、单步时延与吞吐的关系

### 3.1 静态纯 decode

若 batch 内每条序列已经完成 prefill，每轮都只生成 1 个 token，并且没有抢占、chunked prefill、speculative verification 或 padding token 混入，则一轮大致生成 $B$ 个 token：

$$
\operatorname{tokens\ per\ iteration} \approx B
$$

这时 `server-side decode step latency` 与单请求的理想 TPOT 很接近。

### 3.2 Continuous batching

生产服务中，活跃 batch 会随请求到达、完成、抢占和调度而变化。令第 $t$ 轮实际活跃 token 数为 $B_t$，更稳妥的聚合吞吐定义是：

$$
\operatorname{Throughput} \approx \frac{\sum_t B_t}{\sum_t T_{\mathrm{iter},t}}
$$

用户可见的 inter-token latency 还可能包含调度间隙、排队和流式传输，因此不一定严格等于 GPU iteration latency。做实验时必须同时记录两者。

## 4. Dense Small-M GEMM：为什么 TPOT 不会随 Batch 线性增长

考虑一个线性层：

$$
[M,K]\times[K,N]\rightarrow[M,N], \qquad M\approx B
$$

乘加计算量为：

$$
F=2MKN
$$

若输入、权重和输出元素字节数分别为 $s_A$、$s_W$、$s_C$，只计算不可避免的数据流量下界，可写为：

$$
Q_{\min}\approx s_A MK+s_W KN+s_C MN
$$

于是算术强度为：

$$
AI=\frac{2MKN}{s_A MK+s_W KN+s_C MN}
$$

当 $K,N$ 很大、权重项 $s_WKN$ 主导时：

$$
AI\approx\frac{2M}{s_W}
$$

BF16 权重有 $s_W=2$ Byte，因此 $AI\approx M$ FLOP/Byte。其工程含义不是“计算量消失”，而是同一组权重在一轮中服务更多 token：$M$ 增长了，权重的最低流量却没有按 $M$ 成比例增长。

一个简化的 Roofline 下界是：

$$
\begin{aligned}
T_{\mathrm{GEMM}}
&\gtrsim \max\!\left(
\frac{F}{P_{\mathrm{eff}}(M,N,K)},
\frac{Q}{BW_{\mathrm{eff}}(M,N,K)}
\right) \\
&\quad + T_{\mathrm{overhead}}
\end{aligned}
$$

这里必须使用随形状变化的有效算力和有效带宽，而不能把芯片峰值直接当成实测值。$M=1$ 在数学上退化为 GEMV，但后端未必真的 dispatch 一个名为 GEMV 的 kernel；具体选择还受 $N/K$、tile、split-K、persistent scheduling、epilogue 和框架版本影响。

### 4.1 H100 的 Roofline 只提供理论锚点

NVIDIA 的 H100 SXM 页面列出 BF16 Tensor Core 峰值 1,979 TFLOP/s，并注明该数值使用稀疏性。对应的 dense 峰值约为 989.5 TFLOP/s；HBM 带宽为 3.35 TB/s。由此得到理论 dense BF16 ridge point：

$$
\begin{aligned}
I^*
&\approx\frac{989.5\ \mathrm{TFLOP/s}}{3.35\ \mathrm{TB/s}} \\
&\approx295\ \mathrm{FLOP/Byte}
\end{aligned}
$$

在权重主导近似下，$M=64$ 的 $AI$ 约为 64 FLOP/Byte，仍低于这个芯片峰值分界。但这不能单独证明实际 kernel 一定受 HBM 限制，因为 Small-M 的实际计算上限可能远低于芯片峰值，而且 L2 命中、重复读取、融合与调度都会改变 $Q$、$P_{\mathrm{eff}}$ 和 $BW_{\mathrm{eff}}$。

更准确的结论是：Batch 增大让 Dense GEMM 的权重复用和 $M$ 维效率变好，所以每轮计算量虽然增加，iteration latency 往往增加得慢得多。

## 5. WGMMA 的 m64 意味着什么

Hopper PTX 的 `wgmma.mma_async` 由一个 warpgroup，也就是 4 个连续 warp、128 个线程协作。BF16/F16 指令族使用 `.m64nNk16` 一类形状，指令级 $M$ 固定为 64。

这带来一个有条件的判断：如果所选 kernel 确实用 WGMMA 的 m64 tile 覆盖 GEMM 的 $M$ 维，那么逻辑 $M=64$ 可以消除该维的尾块或无效行，$M<64$ 则可能存在 predication、padding 或较低的有效工作占比。

但 `WGMMA m64` 不等于 `Batch 64 必然最快`：

- 逻辑 GEMM 是否走 WGMMA，由库和 kernel dispatch 决定。
- $M<64$ 可能走专门的 GEMV、MMA 或 persistent kernel，而不是简单把 64 行全部算一遍。
- $N/K$ 对齐、grid 并行度、流水深度、寄存器压力、shared memory 与 epilogue 仍会决定最终性能。
- kernel 切换可能让某些 batch 点的 step latency 局部下降，也可能产生阶梯和抖动；具体拐点只能实测。

所以 $B=64$ 是值得测量的形状，不是硬件无关的性能定律。

## 6. KV Cache：为什么上下文越长，曲线越容易变陡

Decode attention 的 query length 通常为 1，但每条序列要读取自己的历史 K/V。对异长请求，忽略当前 token 写入、页表和其他小项时，每层最低 KV 读取量近似为：

$$
Q_{\mathrm{KV}}\approx
2s_{\mathrm{KV}}h_{\mathrm{KV}}d_h\sum_{i=1}^{B}L_i
$$

其中 2 表示 K 和 V，$h_{\mathrm{KV}}$ 是 KV head 数，$d_h$ 是 head dimension，$L_i$ 是第 $i$ 条序列当前上下文长度。若各序列长度近似为 $L$，则 $Q_{\mathrm{KV}}\propto B\cdot L$。

这与 Dense 权重不同：不同请求的 KV 通常不同，不能自然地跨 batch 复用。因此上下文越长，KV 扫描越容易成为 TPOT 的主导项。需要同时注意：

- GQA/MQA 通过减少 KV head 数降低流量系数。
- KV 量化通过减小 $s_{\mathrm{KV}}$ 降低字节数。
- PagedAttention 的块表、碎片和访存局部性会影响有效带宽。
- prefix sharing 或缓存命中可能带来例外，但不能当作一般假设。
- TP 只有在 KV heads 能被有效切分时才近似按 TP 降低单卡 KV 流量；当 TP 大于 KV head 数或实现复制 KV 时，该近似失效。

因此“短上下文”和“长上下文”不是固定 token 阈值。短表示 Dense/固定开销主导；长表示 attention/KV 时间已经与它们相当或占主导，边界应由 profiler 判定。

## 7. TP、EP 与 MoE 如何改变曲线

### 7.1 Tensor Parallel

经典 Megatron 风格的 Dense TP 通常在 attention output 和 MLP output 附近各有一次 activation reduction，但现代实现可能使用 reduce-scatter、all-gather、融合或重叠。若只用 ring all-reduce 做示意，消息大小 $S\approx s_a M d_{\mathrm{model}}$，通信时间可写成：

$$
\begin{aligned}
T_{\mathrm{AR}}
&\approx 2(P-1)\alpha \\
&\quad + \frac{2(P-1)}{P}\frac{S}{\beta}
\end{aligned}
$$

$P$ 是 TP degree，$\alpha$ 是每个通信阶段的启动延迟，$\beta$ 是有效链路带宽。小 $M$ 时常由 $\alpha$ 主导；$M$ 增大后，随消息量增长的带宽项更明显。

TP 同时让每张卡的权重与主要计算大致减少，但也会让本地 $K$ 或 $N$ 变窄，可能削弱 GEMM 效率。NCCL 还会根据拓扑和消息大小选择 ring、tree 或 NVLS 等算法，所以通信与计算不能无条件直接相加。

### 7.2 Expert Parallel 与 MoE

MoE 中，单专家 GEMM 的有效 $M$ 是该专家收到的 token assignment 数 $m_e$。若本轮有 $T$ 个 routed token、top-k 为 $k$、routed experts 数为 $E$，在均匀独立路由近似下：

$$
\lambda=\mathbb{E}[m_e]=\frac{Tk}{E}
$$

以 $T=64$、$k=2$、$E=160$ 为例，$\lambda=0.8$。这不表示每个专家真的收到 0.8 个 token，而表示跨全部专家的平均 assignment 数很小。进一步地，近似非空专家数为：

$$
\mathbb{E}[E_{\mathrm{active}}]\approx E(1-e^{-\lambda})
$$

代入上例约为 88 个非空专家；非空专家的条件均值约为：

$$
\mathbb{E}[m_e\mid m_e>0]\approx
\frac{\lambda}{1-e^{-\lambda}}\approx1.45
$$

所以全局 batch 已经是 64，单专家 GEMM 仍可能非常瘦。与此同时，Batch 增大也会激活更多专家，使一轮中需要流过 HBM 的专家权重集合扩大；它既可能改善 grouped GEMM 效率，也可能增加权重流量。

EP 还需要 dispatch/combine all-to-all，其字节量大致随 $T\cdot k\cdot d_{\mathrm{model}}\cdot s_a$ 增长。跨节点链路、小消息启动延迟、路由偏斜、最慢专家、token packing 与可选 padding 共同决定尾延迟。因此不能无条件断言 MoE 的 TPOT 一定比 Dense 涨得更快；方向取决于专家数、路由分布、EP 拓扑与 grouped GEMM 实现。

## 8. 怎样理解 Batch 1 到 64 的典型区间

下面只描述常见机制，不把区间边界当作通用阈值：

| Batch 区间 | Dense 线性层的常见变化                                      | 其他可能开始显现的因素                         | TPOT 机制示意                      |
| ---------: | ----------------------------------------------------------- | ---------------------------------------------- | ---------------------------------- |
|     1 到 8 | 从 GEMV/极瘦 GEMM 走向 Small-M GEMM；权重复用和固定开销摊薄 | kernel launch、TP 小消息延迟                   | 近似平台或小幅上升；切换点可能下降 |
|    8 到 32 | tile 利用率与有效并行度继续变化                             | KV 流量、activation traffic、collective 带宽项 | 缓慢上升或出现阶梯                 |
|   32 到 64 | 每轮工作量继续增加，权重复用收益边际减弱                    | 长上下文 KV、TP/EP 带宽、容量和调度约束        | 更容易明显上扬，吞吐边际收益变小   |

按模型场景看：

| 场景                          | Batch 增大时更可能出现的曲线 | 主要原因                                               |
| ----------------------------- | ---------------------------- | ------------------------------------------------------ |
| 短上下文 Dense                | 前段平台或缓升，后段上扬     | Dense 权重流量主导，$M$ 维效率改善                     |
| 长上下文 Dense                | 更早、更明显地上升           | KV 扫描随 $\sum_i L_i$ 增长                            |
| 专家多、每专家 token 少的 MoE | 阶梯与抖动更明显，方向依实现 | expert $M$ 小、激活专家数、routing skew、EP all-to-all |
| 高 TP/跨节点 EP               | 通信拐点更明显               | 小消息延迟转向链路带宽与拓扑限制                       |

一个用于定位主导项的端到端近似是：

$$
\begin{aligned}
T_{\mathrm{iter}}
&\approx T_{\mathrm{fixed}}+T_{\mathrm{dense}}+T_{\mathrm{attention}} \\
&\quad +T_{\mathrm{TP/EP}}+T_{\mathrm{routing/misc}}
\end{aligned}
$$

这只是分解框架。真实系统中，计算、HBM 访问和通信可能重叠，应根据时间线判断使用求和、取最大值还是更复杂的 critical path。

## 9. 怎样做可复现的 profiling

1. 固定硬件与软件：GPU SKU、频率和功耗上限、驱动、CUDA、框架、kernel 版本、模型、dtype/量化、KV dtype、TP/EP、CUDA Graph 与调度策略。
2. 隔离纯 decode：先完成 prefill；关闭或单独报告 chunked prefill、抢占、prefix cache、speculative decoding 和请求流式传输。
3. 使用实际 batch：扫 $B=1,2,4,8,16,32,64$，记录每轮真实 $B_t$，不要只记录配置的 max batch。
4. 固定上下文桶：分别测短、中、长上下文；混合长度时同时记录 $\sum_i L_i$、均值和分位数。
5. 预热后重复：报告 $T_{\mathrm{iter}}$、用户侧 ITL/TPOT 的 p50、p95、p99、总吞吐和误差条。
6. 分解 GPU 时间：Dense GEMM、decode attention、NCCL collective、router、dispatch/combine、grouped GEMM、sampling 与 scheduler。
7. 同看硬件计数器：HBM 吞吐、L2 hit rate、Tensor Core/SM 利用率、kernel 名称、tile 形状、wave/tile quantization 与 padding 比例。
8. 对 MoE 额外记录：每专家 token 的 mean、p95、max、变异系数、非空专家数、padding 比例与 all-to-all 的消息分布。

只有当这些条件固定后，才能把某个 Batch 点的下降或拐点归因于 GEMM $M$ 维效率，而不是调度或通信噪声。

## 10. 常见误区

- **“Batch 增大 64 倍，所以 TPOT 增大 64 倍。”** 忽略了 Dense 权重复用、固定开销摊薄和 kernel 效率变化。
- **“权重只读一次。”** 更准确的说法是：理想 compulsory-traffic 下界中，每个 GEMM/step 的权重接近流过一次；实际可能有 L2 命中，也可能因分块和并行策略重复读取。
- **“线性层的 $M$ 永远等于 batch。”** 只对常规单 token Dense decode 近似成立。MoE、speculative verification、padding、packing 与混合 prefill 都会改变有效 $M$。
- **“WGMMA m64 说明 Batch 64 必然是甜点位。”** 指令形状不是端到端性能结论。
- **“短上下文和长上下文有统一 token 阈值。”** 阈值取决于模型结构、GQA/MQA、KV dtype、硬件、kernel 与并行拓扑。
- **“平均 TPOT 就是用户体验。”** 平均值会掩盖调度、routing skew、collective 和抢占造成的长尾。
- **“配置 batch=64 就是在测 64。”** Continuous batching 中应该按实际活跃 $B_t$ 分桶。
- **“对话里的 10/14/30 ms 是 benchmark。”** 它们只是示意；没有原始测量条件就不能用作任何硬件或模型的性能结论。

## 11. 最终回答

Batch 不是单纯“让每步多做 $B$ 倍计算”，而是同时改变 Dense 权重的复用率、GEMM 的 $M$ 维效率、独立 KV 的读取量，以及 TP/EP/MoE 的通信与负载均衡。

对短上下文 Dense 模型，Batch 从 1 增大到 64 时，常见结果是吞吐大幅提高，而 TPOT 只小幅变化：前段可能近似持平，kernel 切换点可能局部下降，后段再缓慢上升。对长上下文、细粒度 MoE 或通信占比较高的配置，KV 扫描、专家路由和 collective 会更早把 TPOT 推高。

所以：**TPOT 不会按 Batch 倍数线性增长，也不存在通用的 32 或 64 拐点。唯一可靠的拐点来自固定配置下逐 Batch、逐上下文桶的 profiling。**

## 12. 参考资料

1. [NVIDIA: Matrix Multiplication Background User's Guide](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html)
2. [NVIDIA H100 Tensor Core GPU Specifications](https://www.nvidia.com/en-us/data-center/h100/)
3. [NVIDIA PTX ISA: Asynchronous Warpgroup Level Matrix Multiply-Accumulate](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-multiply-accumulate-instructions)
4. [NVIDIA NCCL: Collective Operations](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html)
5. [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053)
6. [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245)
7. [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)
8. [MegaBlocks: Efficient Sparse Training with Mixture-of-Experts](https://arxiv.org/abs/2211.15841)
