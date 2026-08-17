---
title: "如何对 TPU 程序进行性能剖析"
description: "到目前为止，本系列一直是纯理论的：基于硬件 Roofline 进行粗略估算。这种理解能带你走得很远，但大量优化最终取决于实践细节：XLA 编译器如何工作，以及当它未能得到理想结果时，如何使用 JAX/TensorBoard Profiler 等性能剖析工具判断应该做什么。本章将讨论这些内容。"
chapter: 9
order: 9
part: 3
partTitle: "实践教程"
sourcePath: "profiling.md"
sourceCommit: "44109cacac9c5a9809a81c68ae4d45d7d2632ea6"
---

<span id="how-to-profile-tpu-programs"></span>

# 如何对 TPU 程序进行性能剖析

<span id="a-thousand-foot-view-of-the-tpu-software-stack"></span>

## 鸟瞰 TPU 软件栈

Google 提供了大量用于 TPU 编程的 API，从高层 JAX 代码到低层 Pallas 或 HLO。大多数程序员只编写 JAX 代码；这让你可以编写抽象的 NumPy 风格线性代数程序，再自动将其编译为能在 TPU 上高效运行的形式。

下面是一个简单示例：用 JAX 程序把两个矩阵相乘。

```py
import jax
import jax.numpy as jnp

def multiply(x, y):
  return jnp.einsum('bf,fd->db', x, y)

y = jax.jit(multiply)(jnp.ones((128, 256)), jnp.ones((256, 16), dtype=jnp.bfloat16))
```

调用 `jax.jit` 后，我们会让 JAX 跟踪这个函数，并生成一种名为 [StableHLO](https://openxla.org/stablehlo) 的较低层 IR；它是一种与平台无关的机器学习计算 IR，随后由 XLA 编译器进一步降低为 HLO。编译器会执行许多遍处理，以确定融合、布局及其他因素，最终得到可在 JAX 性能剖析中观察的 HLO。这种 HLO 以类似 LLVM 的图视图，表示 JAX 代码中所有核心线性代数运算（矩阵乘法、逐点运算、卷积等）。例如，下面是上述程序对应 HLO 的删节版本[^ch9-1]：

```c
ENTRY %main.5 (Arg_0.1: f32[128,256], Arg_1.2: bf16[256,16]) -> f32[16,128] {
  %Arg_1.2 = bf16[256,16]{1,0} parameter(1), metadata={op_name="y"}
  %convert.3 = f32[256,16]{1,0} convert(bf16[256,16]{1,0} %Arg_1.2),
  %Arg_0.1 = f32[128,256]{1,0} parameter(0), metadata={op_name="x"}
  ROOT %dot.4 = f32[16,128]{1,0} dot(f32[256,16]{1,0} %convert.3, f32[128,256]{1,0} %Arg_0.1), lhs_contracting_dims={0}, rhs_contracting_dims={1},
}
```

稍后马上会解释 HLO 的语法；现在只需注意，它其实与上面的 JAX 代码非常吻合。例如，

```c
ROOT %dot.4 = f32[16,128]{1,0} dot(f32[256,16]{1,0} %convert.3, f32[128,256]{1,0} %Arg_0.1), lhs_contracting_dims={0}, rhs_contracting_dims={1}
```

就是上面真正的矩阵乘法：分别沿第 0 维和第 1 维，把两个 f32 矩阵相乘。

**为了把这段 HLO 转换为能在 TPU 上执行的代码，XLA 编译器首先会将其降低为 LLO**（低层优化器）IR。LLO 直接对 TPU 编程：调度不同内存之间的复制、把数组送入脉动阵列，等等。LLO 代码包含这样的原语：将缓冲区送入脉动阵列、取回结果，以及调度在 TPU 不同内存组成部分之间通信的 DMA。降低为 LLO 后，它会进一步编译为机器码，载入 TPU IMEM 并执行。

当程序运行得比预期慢时，我们主要在 JAX 层面改进性能。不过，这往往要求我们理解 HLO 的一些语义，以及代码实际上如何在 TPU 上运行。如果较低层出现问题，我们就会再拉开一道“逃生舱门”，用 [Pallas](https://jax.readthedocs.io/en/latest/pallas/tpu/details.html) 编写自定义内核。要查看程序的 HLO 及其运行时统计信息，我们会使用 JAX profiler。

<span id="the-jax-profiler-a-multi-purpose-tpu-profiler"></span>

## JAX Profiler：多用途 TPU 性能剖析器

JAX 提供了一个多用途 TPU 性能剖析器，其中包含大量实用工具，可帮助理解程序运行时 TPU 上究竟发生了什么。可以使用 `jax.profiler` 模块跟踪正在运行的程序，记录每个子组件的持续时间、每个程序的 HLO、内存用量等所有信息。例如，下面的代码会把跟踪结果写入 `/tmp/tensorboard` 中的文件，再用 TensorBoard 查看（[这里](https://docs.jax.dev/en/latest/profiling.html#tensorboard-profiling)有一份分步指南）。

```py
import jax
with jax.profiler.trace("/tmp/tensorboard"):
  key = jax.random.key(0)
  x = jax.random.normal(key, (1024, 1024))
  y = x @ x
  y.block_until_ready()

# Now you can load TensorBoard in a Google Colab with
#
# !pip install -U xprof
# !pip install -U protobuf
# %load_ext tensorboard
# %tensorboard --logdir=/tmp/tensorboard
#
# or externally with
#
# > tensorboard --logdir=/tmp/tensorboard
#
```

下面概览性能剖析器能够完成的工作：

![](/images/scaling-book/img/xprof-overview.png)

进入 TensorBoard 后，性能剖析器中有几个关键标签页，可以帮助理解程序：

1. **Trace Viewer** 会显示 TPU 上实际发生事件的详细时间线。
2. **Graph Viewer** 会显示 HLO 图，让你看到程序各部分如何相互馈入，以及各部分如何分片。
3. **Memory Profile 和 Memory Viewer：** 二者会显示程序使用了多少内存。

虽然分享完整的性能剖析结果有些困难，但[这里](https://ui.perfetto.dev/#!/?s=fa9f13b487bde622707c1a503f9227c34594760a)有一个 Perfetto 链接，至少包含一个简单 Transformer 的 Trace Viewer 组件。[这个 Colab](https://colab.research.google.com/drive/1_6krERgtolH7hbUIo7ewAMLlbA4fqEF8?usp=sharing)则可以生成完整的 JAX/TensorBoard 跟踪结果，供你动手探索。

### Trace Viewer

**Trace Viewer 可能是性能剖析器中最有用的部分。** 下面的示例展示了一个简单 Transformer，并对其各部分做了标注。这些名称来自代码中提供的标签。

![](/images/scaling-book/img/trace-viewer.png)

Trace Viewer 会显示每个 TPU 核心上所有操作的时间顺序。这里我们只查看 TPU:0，因为通常所有 TPU 都执行相同的指令。几个关键注意事项如下：

1. 最上面一行（XLA Ops）显示真正的 TPU 操作（这些名称是 HLO 名称）。其他所有内容都是根据 `jax.named_scope`、`jax.named_call` 和 Python 堆栈跟踪得到的近似跟踪结果。
2. 留意重复出现的块，就可以在这里分离出单独一层。还可以通过查看代码/理解 Transformer 的工作方式，看出哪些部分是注意力，哪些部分是 MLP。
3. 点击某个 XLA op，就可以查看它来自代码中的什么位置（有助于理解跟踪结果），并看到指向 Graph Viewer 的链接。

**提示：** 可以像操控“电子游戏”一样浏览 Trace Viewer：按 A/D 向左或向右平移，按 W/S 放大或缩小。这些控制方式让浏览轻松许多。

<span id="how-to-read-an-xla-op"></span>

### 如何阅读 XLA op

HLO 其实并不难读，而且非常有助于理解上面跟踪结果中的某个部分对应什么。下面是一个名为 fusion.3 的 op 示例。

```c
%fusion.3 = bf16[32,32,4096]{2,1,0:T(8,128)(2,1)S(1)} fusion(bf16[32,32,8192]{2,1,0:T(8,128)(2,1)S(1)} %fusion.32), kind=kCustom, calls=%all-reduce-scatter.3
```

把它拆成几个组成部分。

* **Op 名称**：fusion.3
  * dot 或 fusion op 是一组操作，其中最多包含 1 次矩阵乘法，也可能包含一系列相关的逐点 VPU op。
* **形状**：`bf16[32,32,4096]`
  * 这是 op 的输出形状。可以看到 dtype 是 bf16（每个元素 2 字节），而 `[32,32,4096]` 是其形状。
* **布局：** `{2,1,0:T(8,128)(2,1)}`
  * `{2,1,0:T(8,128)(2,1)}` 告诉我们各轴在内存中的顺序（列优先、行优先等）以及数组的填充。下文会进一步说明。
* **内存位置：** S(1)
  * S(1) 表示这个数组位于 VMEM。S(0)（有时省略）表示 HBM。S(2) 和 S(3) 是其他内存空间。
* **参数**：`bf16[32,32,8192]{2,1,0:T(8,128)(2,1)S(1)} %fusion.32`
  * 这个 op 有一个输入，即名为 fusion.32、具有特定形状的 bf16 数组。由此可以知道哪个函数向它馈入数据。

再深入理解一下这套记法。以下面这个简单示例为例：

`f32[3,5]{1,0:T(2,2)}`

同样，它告诉我们这个 Op 返回一个形状为 `[3, 5]`、具有特定分块 `{1,0:T(2,2)}` 的 float32 数组。虽然分块并没有*那么*重要，但简要来说，它描述了一个 N 维数组如何依次排列在内存中。下图展示了该数组的布局方式：

![](/images/scaling-book/img/tiling.png)

在 `{1,0:T(2,2)}` 中，`1,0` 部分表示数组各维在物理内存中的次序，从最次维到最主维。可以从右向左阅读这一部分，再从 `f32[3,5]` 中找出对应维度，以确定数组的物理布局。在本例中，物理布局为 `[3,5]`，与逻辑形状相同。
接着，`T(2,2)` 表示数组会分成 `(2, 2)` 的块；每个块内部先排列行（**行优先**），再排列列，也就是 `(0, 0)` 之后依次为 `(0, 1)`、`(1, 0)` 和 `(1, 1)`。由于采用 `T(2, 2)` 分块，数组会填充到 `[4, 6]`，使内存用量增加约 1.6 倍。对于上面给出的大型 bf16 数组 `bf16[32,32,8192]{2,1,0:T(8,128)(2,1)S(1)}`，采用的是 `T(8,128)(2,1)`；这表示数组有两级分块：外层是 `(8, 128)` 分块，该单元内部还有 `(2, 1)` 分块（用于 bf16，使加载大小始终为 4 字节的倍数）。例如，下面是 `bf16[4,8]{1,0:T(2,4)(2,1)}`（颜色表示 (2,4) 分块，红框表示 (2,1) 分块）：

![](/images/scaling-book/img/tiling2.png)

分块会影响张量分块载入 VMEM 的效率。XLA 有时会在程序内部引入副本，对张量“重新分块”或“重新布局”，有时会产生不可忽视的开销。[^ch9-2]

### Graph Viewer

虽然上面有些融合可能显得复杂，但 XLA Graph Viewer 能让它们更易于解析。例如，下面是一个相当复杂的融合视图：

![](/images/scaling-book/img/graph-viewer.png)

仔细观察许多 HLO 图，并尝试把 HLO op 映射到正在剖析的代码，非常有帮助。将鼠标悬停在某个方框上时，通常会看到定义该函数的代码行。

<span id="looking-at-a-realish-example-profile"></span>

### 查看一个接近真实的性能剖析示例

[这个 Colab](https://colab.research.google.com/drive/1_6krERgtolH7hbUIo7ewAMLlbA4fqEF8?usp=sharing)包含一个虚构 Transformer 的性能剖析示例。如果时间紧张，[这里](https://ui.perfetto.dev/#!/?s=fa9f13b487bde622707c1a503f9227c34594760a)有一个 Perfetto 链接，至少可以查看 Trace Viewer。我比平时花了更多功夫用 `jax.named_scope` 调用标注跟踪结果，便于你识别正在发生什么。

![](/images/scaling-book/img/transformer-xprof.png)

请查看性能剖析结果，并尝试真正理解每一部分在做什么。先从 FFW 块开始，稍作拆解：

![](/images/scaling-book/img/transformer-ffw.png)

这里已经放大到 FFW 块。可以看到，上投影 Op 是一个 fusion（矩阵乘法），输入为 `bf16[8, 1024, 8192]` 和 `bf16[8192, 16384]`，输出为 `bf16[8, 1024, 16384]`。我知道（因为这段代码是我写的），这是一次四路 DP、两路 MP 分片矩阵乘法的局部视图，因此实际执行的是

**X：** `bf16[32, 1024, 8192]` \* **W<sub>in</sub>**：`bf16[8192, 32768]` -> **Tmp**：`bf16[32, 1024, 32768]`

**预计需要多长时间？** 首先，每个数据并行分片的批大小为 `8 * 1024 = 8192`，因此显然会达到计算受限。这里使用 8 个 TPU v2 核心，所以预计耗时约为 `2 * 32 * 1024 * 8192 * 32768 / (23e12 * 8) = 95.6ms`，这几乎恰好就是实际耗时（96ms）。太棒了！这意味着 FLOPs 利用率极高！

请注意，Google Colab 已不再提供 TPU v2-8 切片。要获得真实的 8 核切片并跟随示例操作，可以使用仍然免费提供它们的 [Kaggle](https://www.kaggle.com/)，或者在 GCP 上预配一个 8 核切片。[^ch9-3]

**通信呢？** 你会注意到第二次矩阵乘法末尾隐藏着一个小 fusion。点击它，会看到

```c
%fusion.1 = bf16[8,1024,4096]{2,1,0:T(8,128)(2,1)} fusion(bf16[8,1024,8192]{2,1,0:T(8,128)(2,1)} %fusion.31), kind=kCustom, calls=%all-reduce-scatter.1
```

它基本就是一个小型 ReduceScatter（下面是 Graph Viewer）：

![](/images/scaling-book/img/reduce-scatter-xprof.png)

预计这需要多长时间？这里是在 TPU v2 4x2 上执行 ReduceScatter，使用 1.2e11 的双向带宽，应该只需要一跳。数组大小为 `2*32*1024*8192`，且批维度分为 4 路，因此每个分片为 `2*8*1024*8192=128MB`。所以预计耗时约为 1.1ms。**实际需要多长时间？** 性能剖析报告为 1.13ms。因此，我们非常接近 Roofline！

**也看看注意力！** 下面是注意力组件的性能剖析：

![](/images/scaling-book/img/attn-xprof.png)

我点击了 Q 投影 op，它使用形状为 [d<sub>model</sub> = 8192, n<sub>heads</sub> = 32, d<sub>qkv</sub> = 256] 的矩阵 $W_Q$。我们正沿头维度进行 Megatron 分片。请尝试完成同样的练习，计算这些操作预计需要多长时间。

### Memory Profile

Memory Profile 可以轻松地查看程序内存如何随时间变化，这有助于调试 OOM。这里可以看到，分配给模型参数的内存约为 7.5GB，另有约 8.5GB 空闲。因此，内存中还能容纳更多内容。

![](/images/scaling-book/img/memory-viewer.png)

<span id="worked-problems"></span>

## 练习题

**问题 1**：请查看[这个](https://colab.research.google.com/drive/1LfLO3OTr-_MWFPxUN36KJ3cqH0BcAoli?usp=sharing) Colab/性能剖析结果，找出哪些地方看起来可疑，以及这里究竟发生了什么。你能准确说出正在进行哪些计算、每个操作在做什么吗？其中每个矩阵的真实形状是什么，又如何分片？*请先尝试只看性能剖析结果，不要阅读代码。*

![](/images/scaling-book/img/all-reduce-profile.png)

<details>
<summary>点击此处查看答案。 </summary>


这里是两次矩阵乘法，具体来说就是：

```py
def matmul(w1, w2, x):
  return jnp.einsum('wf,bf->bw', w2, jnp.einsum('fw,bw->bf', w1, x))
```

可以看到一次 reduce、两个大型 fusion 和一次 all-reduce。第一个大型 fusion 是：

```%fusion.1 = bf16[4096]{0:T(1024)(128)(2,1)} fusion(bf16[4096,8192]{1,0:T(8,128)(2,1)} %param.1, bf16[8192]{0:T(1024)(128)(2,1)} %reduce.6), kind=kLoop, calls=%fused_computation.1```

它告诉我们，每个分片的形状为 `bf16[8192] * bf16[4096, 8192] -> bf16[4096]`（沿 8192 维度）。观察最后的 AllReduce 及其 `replica_groups={{0,16,32,48,64,80,96,112}, ...}`，可以判断这里采用了八路模型并行，因此真实形状为 `bf16[8, 8192] * bf16[32768, 8192] -> bf16[8, 32768]`。

</details>

**问题 2：** [前面使用的 Transformer Colab](https://colab.research.google.com/drive/1_6krERgtolH7hbUIo7ewAMLlbA4fqEF8?usp=sharing)实现了一个简单的模拟 Transformer。由于 Colab 已不再提供 TPU v2-8 切片，如果想跟随示例操作，需要在 [Kaggle](https://www.kaggle.com/) 或 GCP 的 8 核切片上运行。按照 Colab 中的说明，对使用 GSPMD 分区的朴素 Transformer 进行基准测试。各部分分别需要多长时间？理论上应该需要多长时间？采用了什么分片？请尝试修正分片！*提示：使用 `jax.lax.with_sharding_constraint` 约束其行为。完成这项修正后，能够达到的最佳 MFU 是多少？*

作为参考，初始版本约为 184ms / 层，优化后的性能剖析约为 67ms / 层。完成后，请尝试仔细观察性能剖析结果，看看能否只根据它回答以下问题：

- 这里使用了什么分片策略？
- 批大小、$d_\text{model}$、$d_\text{ff}$ 分别是多少？
- 注意力与 MLP 块分别占用多少比例的时间？
- 在 Roofline 上，各 op 理应分别占用多少比例的时间？

**注意：** 自这道题写成以来，XLA 编译器已经有所改进。初始版本现在约为 90ms / 层，而优化后的性能剖析只快约 10ms / 层（80ms / 层）。尽管如此，仍然值得动手尝试，看看能否做得更好。

<span id="thats-all-for-part-9-for-part-10-with-a-deep-dive-into-jax-parallelism-click-here"></span>

### 第 9 部分到此结束。第 10 部分将深入探讨 JAX 并行，请点击[这里](../10-jax/#programming-tpus-in-jax)。

[^ch9-1]: 要得到这段 HLO，可以运行 `jax.jit(f).lower(*args, **kwargs).compile().as_text()`。
[^ch9-2]: JAX 提供了一项[实验性功能](https://docs.jax.dev/en/latest/notebooks/layout.html)来绕过这个问题：允许 XLA 计算程序输入的“首选”布局。使用 `jax.jit` 对程序进行“即时”编译时，通常会传入“模拟”输入，告诉 JAX 应期待的形状和 dtype。这些输入通常还携带可能并非最优的分块信息。作为替代，可以把输入布局指定为 AUTO，`jax.jit` 就会返回即时编译程序偏好的布局。随后，可以明确地按该布局载入张量，避免在程序内部引入复制。
[^ch9-3]: 如果只想在虚构问题上尝试分片，也可以用 `import jax; jax.config.update("jax_num_cpu_devices", 8)` 在 CPU 上模拟 8 个设备（需要 jax >= 0.4.27 左右），再运行 `print(jax.devices())`。这种方法只适用于玩具问题，并不能反映真实性能。
