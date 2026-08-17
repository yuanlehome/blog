---
title: "使用 JAX 为 TPU 编程"
description: "如何使用 JAX 高效地为 TPU 编程！本章的大量内容取自 JAX 的 shard_map JEP 文档（https://jax.readthedocs.io/en/latest/jep/14273-shard-map.html）。你可以在 Google Colab（https://colab.sandbox.google.com/）上使用免费的 TPU 运行本章代码示例。"
chapter: 10
order: 10
part: 3
partTitle: "实践教程"
sourcePath: "jax-stuff.md"
sourceCommit: "44109cacac9c5a9809a81c68ae4d45d7d2632ea6"
---

<span id="programming-tpus-in-jax"></span>

# 使用 JAX 为 TPU 编程

<span id="how-does-parallelism-work-in-jax"></span>

## JAX 中的并行机制如何工作？

JAX 支持三种多设备编程思路：

1. **编译器，交给你了！** 让 XLA 编译器自动对数组分区，并决定要添加哪些通信来支持给定程序。这样，一个能在单设备上运行的程序无需任何修改，就能自动扩展到数千台设备上运行。
2. **JAX，交给你了！** 自动并行很棒，但编译器有时会做出疯狂的事情。显式分片允许你像往常一样编写单设备代码，同时让 JAX（而不是编译器）处理分片传播。这意味着，当 JAX 不清楚你想要什么时，可以要求你进一步说明。
3. **拜托，让我直接写明自己的意思！** 编译器固然很好，但有时会做错事，添加你并不想要的通信。有时候，我们希望明确写出究竟打算运行哪些通信。

| 模式 | 视图？ | 显式分片？ | 显式集合通信？ |
|:---:|:---:|:---:|:---:|
| Auto | 全局 | ❌ | ❌ |
| Explicit | 全局 | ✅ | ❌ |
| Manual | 每设备 | ✅ | ✅ |

与此对应，JAX 为每种模式都提供了 API：

1. `jax.jit`（配合 `Auto` 网格轴）允许你接收任意现有 JAX 函数，并用分片输入调用它。随后，JAX 会使用 XLA 的 [Shardy](https://openxla.org/shardy) 编译器自动并行化程序。为支持现有操作，XLA 会在需要时替你添加通信（AllGather、ReduceScatter、AllReduce 等）。虽然它并不完美，但通常能在不修改代码的情况下，相当不错地把程序自动扩展到任意数量的芯片。
2. `jax.jit` 配合 `Explicit` 网格轴时与第（1）种模式很相似，但由 JAX 而不是 XLA 处理分片传播。这意味着数组的分片实际上成为 JAX 类型系统的一部分；JAX 检测到通信存在歧义时可以报错，让用户自行解决。
3. `jax.shard_map` 是更加手动的对应方案。你会获得程序的设备局部视图，并且必须明确写出所有需要的通信。如果有一个分片数组，并希望每台设备都获得完整数组，就添加 `jax.lax.all_gather`。如果希望跨设备对数组求和，就添加 `jax.lax.psum`（一次 AllReduce）。这种编程方式更困难，但做出你不想要之事的可能性要低得多。

<span id="auto-sharding-mode"></span>

### 自动分片模式

`jax.jit` 在 JAX 内部扮演两个角色。顾名思义，它会“即时”把函数从 Python 编译为字节码（经由 XLA/HLO/LLO），使其运行得更快。但如果输入已分片，或者用户指定了 `in_sharding` 或 `out_sharding`，它还会允许 XLA 将计算分布到多台设备，并按需添加通信。例如，下面展示了如何用 `jax.jit` 编写分片矩阵乘法：

```py
import jax
import jax.numpy as jnp

Auto = jax.sharding.AxisType.Auto

# This creates a fake set of 8 CPU devices so you can run this on a CPU without TPUs.
jax.config.update("jax_num_cpu_devices", 8)

# This creates a 2D 4x2 mesh with axis names X and Y that JAX uses by default.
# We explicitly tell JAX to let the XLA compiler infer sharding along these axes.
mesh = jax.make_mesh(axis_shapes=(4, 2), axis_names=('X', 'Y'), axis_types=(Auto, Auto))
jax.set_mesh(mesh)

# We create a matrix W and input activations In sharded across our devices.
In = jnp.zeros((8, 2048), dtype=jnp.bfloat16, device=jax.NamedSharding(mesh, jax.P('X', 'Y')))
W = jnp.zeros((2048, 8192), dtype=jnp.bfloat16, device=jax.NamedSharding(mesh, jax.P('Y', None)))

def matmul_square(In, W):
  return jnp.einsum('bd,df->bf', jnp.square(In), W)

# We can explicitly compile the sharded matmul function here. This adds all the
# necessary comms (e.g. an AllReduce after the matmul).
jit_matmul = jax.jit(matmul_square, out_shardings=jax.P('X', None)).lower(In, W).compile()

out = jit_matmul(In, W)
```

无论采用什么分片，这段代码都会自动运行，并把计算分区到各台设备上。**但硬件层面实际发生了什么？**

1. 首先，在各台设备上创建已分片的 In 和 W[^ch10-1]。W 沿收缩维度分为两路，In 则分为八路：沿输入维度分为四路，沿收缩维度分为两路。这对应于 W[D<sub>Y</sub>, F] 和 In[B<sub>X</sub>, D<sub>Y</sub>] 分片，也就是一种模型并行与数据并行。
2. 如果在本地运行（即在单台设备上运行），`matmul_square` 只会对输入求平方，再执行一次简单的矩阵乘法。但由于把 `out_shardings` 指定为 `P('X', None)`，输出将沿批维度分片、在模型维度上复制，因此需要一次 AllReduce 才能计算出来。

沿用前几节的记法，它很可能会执行类似以下操作：

1. Out[B<sub>X</sub>, F] { U<sub>Y</sub> } = In[B<sub>X</sub>, D<sub>Y</sub>] \*<sub>D</sub> W[D<sub>Y</sub>, F]
2. Out[B<sub>X</sub>, F] = **AllReduce**(Out[B<sub>X</sub>, F] { U<sub>Y</sub> })

`jax.jit` 会自动为我们添加这些操作！实际上，可以用 `jit_matmul.as_text()` 打印 HLO，并看到以下 HLO（已大幅删节）：

```py
# This fusion is the actual matmul of the sharded inputs and matrix
%fusion = bf16[2,8192]{1,0:T(4,128)(2,1)S(1)} fusion(bf16[2,1024]{1,0:T(4,128)(2,1)} %param, bf16[8192,1024]{1,0:T(8,128)(2,1)S(1)} %copy-done)

# We reduce the partially summed results across devices
ROOT %AllReduce = bf16[2,8192]{1,0:T(4,128)(2,1)} AllReduce(bf16[2,8192]{1,0:T(4,128)(2,1)S(1)} %fusion)
```

可以在上面看到矩阵乘法（fusion）和 AllReduce。请特别注意这些形状。`bf16[2, 1024]` 是激活值的局部视图，因为 `batch_size=8` 被分到 4 台设备上，而 `d_model=2048` 同样被分为两路。

**这相当神奇！** 无论程序多么复杂，[Shardy](https://openxla.org/shardy) 和 jit 都会尝试为所有中间激活值找到分片方式，并按需添加通信。不过，Shardy 也有缺陷，可能会犯错。有时查看性能剖析结果时，你会注意到某些地方出了问题：一个本无必要的巨型 AllGather 占据了 80% 的剖析时间。发生这种情况时，可以尝试通过 `jax.lax.with_sharding_constraint` 明确标注中间张量，纠正编译器。例如，对于两次矩阵乘法，可以用下面的代码强制中间激活值沿 `y` 维度分片（尽管这并不是个好主意）：

```py
import jax
import jax.numpy as jnp

Auto = jax.sharding.AxisType.Auto

mesh = jax.make_mesh((4, 2), ('X', 'Y'), (Auto, Auto))
jax.set_mesh(mesh)

def matmul(x, W_in, W_out):
  hidden = jnp.einsum('bd,df->bf', x, W_in)
  hidden = jax.lax.with_sharding_constraint(hidden, jax.P('X', 'Y'))
  return jnp.einsum('bf,df->bd', hidden, W_out)
```

在自动分区的世界中，JAX 并行编程约有 60% 都是在做这件事：通过 `jax.lax.with_sharding_constraint` 控制中间分片。不过，“哄编译器”是出了名地令人不快的编程模型。即使标注每个中间变量，仍然不知道最终能否得到正确结果。那么，是否可以改由 JAX 自身处理并控制分片传播？

<span id="explicit-sharding-mode"></span>

### 显式分片模式

显式分片（也称“类型中的分片”）看起来与自动分片很相似，但分片传播发生在 JAX 层面！每个 JAX 操作都有一条分片规则：接收该 op 各参数的分片，并为 op 的结果生成分片。可以使用 `jax.typeof` 查看得到的分片：

```py
import jax
import jax.numpy as jnp
import numpy as np

Explicit = jax.sharding.AxisType.Explicit

# Running on a TPU v5e 2x2. This assigns names to the two physical axes of the hardware.
mesh = jax.make_mesh(axis_shapes=(2, 2), axis_names=('X', 'Y'), axis_types=(Explicit, Explicit))

# This tells JAX to use this mesh for all operations, so you can just specify the PartitionSpec P.
jax.set_mesh(mesh)

x = jax.device_put(np.arange(16, dtype=np.float32).reshape(8, 2), jax.P('X', 'Y'))

@jax.jit
def f(x):
  print(jax.typeof(x))  # float32[8@X,2@Y]
  out = x * 2
  print(jax.typeof(out))  # float32[8@X,2@Y]
  return out

f(x)
```

可以看到，JAX 把分片从输入（`x`）传播到了输出（`out`），并且在跟踪时可通过 `jax.typeof` 检查它们。对大多数操作而言，这些规则既简单又显然，因为合理选择只有一个（例如逐元素 op 会保留相同分片）。但对某些操作而言，结果应如何分片存在歧义；在这种情况下，JAX 会在跟踪时报错，要求程序员明确提供 `out_sharding` 参数（例如 jnp.einsum、jnp.reshape 等）。再来看一个存在冲突的示例：

```py
# We create a matrix W and input activations In sharded across our devices.
In = jnp.zeros((8, 2048), dtype=jnp.bfloat16, out_sharding=jax.P('X', 'Y'))
W = jnp.zeros((2048, 8192), dtype=jnp.bfloat16, out_sharding=jax.P('Y', None))

@jax.jit
def matmul_square(In, W):
  print(jax.typeof(In))  # bfloat16[8@X, 2048@Y]
  print(jax.typeof(W))  # bfloat16[2048@Y, 8192]
  return jnp.einsum('bd,df->bf', jnp.square(In), W)

matmul_square(In, W)  # This will error
```

这段代码会报错：

```
Contracting dimensions are sharded and it is ambiguous how the output should be sharded.
Please specify the output sharding via the `out_sharding` parameter.
Got lhs_contracting_spec=('Y',) and rhs_contracting_spec=('Y',)
```

这很棒，因为 einsum 的输出究竟应如何分片确实存在歧义。输出分片可以是：
* P('X', 'Y')，这会引入一次 ReduceScatter；或者
* P('X', None)，这会引入一次 AllReduce。

与 Auto 模式不同，Explicit 模式检测到存在歧义的通信时会报错，并要求用户解决。因此，这里可以这样写：

```py
@jax.jit
def matmul_square(In, W):
  return jnp.einsum('bd,df->bf', jnp.square(In), W, out_sharding=jax.P('X', 'Y'))

out = matmul_square(In, W)
print(jax.typeof(out))  # bfloat16[8@X,8192@Y]
```

Auto 模式和 Explicit 模式可以通过 `jax.sharding.auto_axes` 与 `jax.sharding.explicit_axes` API 组合使用。想了解更多信息，可以阅读[这份优秀文档](https://docs.jax.dev/en/latest/notebooks/explicit-sharding.html)。

<span id="manual-sharding-mode-via-shard_map"></span>

### 通过 `shard_map` 进行手动分片

如果说 Shardy 是“编译器，交给你了”模式，那么 JAX [shard_map](https://jax.readthedocs.io/en/latest/jep/14273-shard-map.html) 就把一切交到你手中。你像在 jax.jit 中一样指定输入的分片，但随后需要明确写出所有通信。`jax.jit` 提供的是程序跨设备的全局视图，而 `shard_map` 提供的是每台设备的局部视图。

下面是一个示例。请尝试推断这个函数做了什么：[^ch10-2]

```py
import jax
import jax.numpy as jnp

Explicit = jax.sharding.AxisType.Explicit

mesh = jax.make_mesh((2, 4), ('x', 'y'), (Explicit, Explicit))
jax.set_mesh(mesh)

x = jnp.arange(0, 512, dtype=jnp.int32, out_sharding=jax.P(('x', 'y')))

# This function will operate on 1/8th of the array.
@jax.shard_map(in_specs=jax.P(('x', 'y')), out_specs=jax.P())
def slice_and_average(x):
  assert x.shape == (512 // 8,)
  return jax.lax.pmean(x[:4], axis_name=('x', 'y'))

out = slice_and_average(x)
assert out.shape == (4,)
```

**这会做什么？** `slice_and_average` 会在每颗 TPU 上运行，各自接收数组的 1/8；我们从中切出前 4 个元素，再跨整个网格对它们求平均。这意味着实际执行的是 `mean(x[:4], x[64:68], x[128:132], …)`。这相当酷，因为用其他方式很难在 JAX 中表达这种操作。

**为什么不使用 jax.jit？** 如果使用 `jax.jit`，`slice_and_average` 看到的将是数组的全局视图（完整的 `[512,]` 数组）。我们必须切出这种非均匀切片，再执行一次平均，而且 XLA 必须正确解释它。XLA 可能会添加错误的通信，也可能感到困惑。这里则可以看到局部视图，并且只写出真正需要的通信。

**示例［集合矩阵乘法］：** 再看一个更现实的示例。假设要实现模型并行，其中激活值最初按模型分片，即 A[B<sub>X</sub>, D<sub>Y</sub>] \*<sub>D</sub> W[D, F<sub>Y</sub>] -> Out[B<sub>X</sub>, F<sub>Y</sub>]。朴素做法是先对 A 执行 AllGather，再执行局部矩阵乘法：

1. A[B<sub>X</sub>, D] = **AllGather**<sub>Y</sub>(A[B<sub>X</sub>, D<sub>Y</sub>])
2. Out[B<sub>X</sub>, F<sub>Y</sub>] = A[B<sub>X</sub>, D] *<sub>D</sub> W[D, F<sub>Y</sub>]

遗憾的是，这种做法很糟，因为它不允许重叠通信与计算。可以使用“集合矩阵乘法”来重叠二者，如 [Wang et al. 2023](https://dl.acm.org/doi/pdf/10.1145/3567955.3567959) 所述。算法基本如下：

* 对每个 Y 分片，将 A 的局部分块与 W 的局部分块相乘，生成形状为 `[B / X, F / Y]` 的结果。同时，对 A 执行置换，从而在本地获得下一个分块，执行矩阵乘法，再把结果相加。

使用 `jax.shard_map` 可以相当轻松地实现它：

```py
import functools

import jax
import jax.numpy as jnp
import numpy as np

Explicit = jax.sharding.AxisType.Explicit

# This is intended to run on a TPU v5e-8 runtime. If you can't get this,
# try setting jax.config.update('jax_num_cpu_devices', 8).
#
mesh = jax.make_mesh(axis_shapes=(2, 4), axis_names=('X', 'Y'), axis_types=(Explicit, Explicit))
jax.set_mesh(mesh)

B, D, F = 1024, 2048, 8192
A = jnp.arange(np.prod((B, D))).reshape((B, D))
W = jnp.arange(np.prod((D, F))).reshape((D, F))

A = jax.device_put(A, jax.P('X', 'Y'))
W = jax.device_put(W, jax.P(None, 'Y'))

@functools.partial(jax.jit, out_shardings=jax.P('X', 'Y'))
def matmul(lhs, rhs):
  return lhs @ rhs

def collective_matmul_allgather_lhs_contracting(lhs, rhs):
  # lhs is the looped operand; rhs is the local operand
  axis_size = jax.lax.axis_size('Y')  # axis_size = 4 for this example
  idx = jax.lax.axis_index('Y')

  chunk_size = lhs.shape[1]
  assert rhs.shape[0] % chunk_size == 0

  def f(i, carrys):
    accum, lhs = carrys
    rhs_chunk = jax.lax.dynamic_slice_in_dim(rhs, (idx + i) % axis_size * chunk_size, chunk_size)
    # Matmul for a chunk
    update = lhs @ rhs_chunk
    # Circular shift to the left
    lhs = jax.lax.ppermute(
        lhs,
        axis_name='Y',
        perm=[(j, (j - 1) % axis_size) for j in range(axis_size)]
    )
    return accum + update, lhs

  accum = jnp.zeros((lhs.shape[0], rhs.shape[1]), dtype=lhs.dtype)
  accum = jax.lax.pcast(accum, ('X', 'Y'), to='varying')
  accum, lhs = jax.lax.fori_loop(0, axis_size - 1, f, (accum, lhs), unroll=True)

  # Compute the last chunk after the final permute to leave lhs in the state we found it
  i = axis_size - 1
  rhs_chunk = jax.lax.dynamic_slice_in_dim(rhs, (idx + i) % axis_size * chunk_size, chunk_size)
  update = lhs @ rhs_chunk
  return accum + update

jit_sharded_f = jax.jit(jax.shard_map(
  collective_matmul_allgather_lhs_contracting,
  in_specs=(jax.P('X', 'Y'), jax.P(None, 'Y')), out_specs=jax.P('X', 'Y')))

shmapped_out = jit_sharded_f(A, W)
expected_out = matmul(A, W)

np.testing.assert_array_equal(shmapped_out, expected_out)
```

这很漂亮！进行基准测试后，可以看到它也快得多！[这里](https://imgur.com/a/e9I6SrM)是默认 jit 矩阵乘法的性能剖析；它耗时 311us，并且开头有一个大型阻塞式 AllGather：

![](/images/scaling-book/img/not-overlapped.png)

[这里](https://imgur.com/a/21iy0Sv)则是上面耗时 244us 的版本。可以看到，性能剖析中没有 AllGather，所有时间都在做有用工作！FLOPs 利用率也高得多。

![](/images/scaling-book/img/overlapped.png)

还值得注意的是，收缩维度上不分片时的矩阵乘法耗时为 [224us](https://imgur.com/a/i3gNKfq)，所以这里已经非常接近未分片的基线。这很好地展示了，为提高 TPU 利用率，最终可能需要进行怎样的性能工程。想查看更多 `shard_map` 示例，[这份笔记非常好](https://jax.readthedocs.io/en/latest/notebooks/shard_map.html#example-1-all-gather-on-one-side)。

下面给出几道实用练习题，请尝试使用 `jax.jit` 或 `shard_map` 实现！

<span id="worked-problems"></span>

## 练习题

下面是一些随机的 JAX 相关问题，之后还会再补充一些。所有问题都需要若干颗 TPU。Colab 已不再提供 TPU v2-8 切片，因此请使用 [Kaggle](https://www.kaggle.com/)（仍可免费使用这些切片）或 GCP 的 8 核切片。[^ch10-3] 从现在起，假设有 N 台可用设备。

**问题 1：** 设 **A** 为一个形状为 float32[S<sub>X</sub>, D<sub>Y</sub>] 的激活值数组，其中 `X * Y = N`。完成以下任务：

1. 用 JAX 编写一个函数，计算每个 `(X, Y)` 分片内部的平均值；也就是说，返回一个大小为 [X, Y] 的数组，其中 `arr[i, j]` 是分片 `(i, j)` 的平均值。分别使用 `jax.jit` 和 `shard_map` 实现。对两种实现进行性能剖析，看看各自耗时多久。是否添加了任何通信？*提示：本不应该有，但 XLA 有时还是会添加。*

2. 用 JAX 编写一个函数，对某个**位于沿 X 的每个分片内部**的 shift，返回 `roll(x, shift, axis=0) - x`。我还没那么自虐，不会要求你用 jax.jit 实现，所以只需使用 `shard_map`。

<details>
<summary>点击此处查看答案。 </summary>


第 1 部分：下面是第 1 部分的一种解法。请注意，`jax.jit` 解法需要进行相当复杂的 reshape。

```py
import numpy as np

import jax
import jax.numpy as jnp

Auto = jax.sharding.AxisType.Auto

mesh = jax.make_mesh((4, 2), ('X','Y'), (Auto, Auto))

average_shmap = jax.shard_map(
    lambda x: x.mean(keepdims=True),
    mesh=mesh,
    in_specs=jax.P('X','Y'), out_specs=jax.P('X','Y')
)

def average(x):
  X, Y = mesh.axis_sizes
  return x.reshape(X, x.shape[0] // X, Y, x.shape[1] // Y).mean(axis=(1, 3))

average_jit = jax.jit(average, out_shardings=jax.NamedSharding(mesh, jax.P('X','Y')))

x = jnp.arange(8 * 64 * 8, dtype=jnp.float32).reshape(8 * 64, 8)
x = jax.device_put(x, jax.NamedSharding(mesh, jax.P('X','Y')))

y1 = average_shmap(x)
y2 = average_jit(x)

np.testing.assert_array_equal(y1, y2)
```

第 2 部分：下面是第 2 部分的一个类似解法。

```py
import numpy as np

import jax
import jax.numpy as jnp

import functools

Auto = jax.sharding.AxisType.Auto

mesh = jax.make_mesh((4, 2), ('X','Y'), (Auto, Auto))

def shift_shmap(x, shift: int):
  shmapped = jax.shard_map(
      lambda x: jnp.roll(x, shift, axis=0),
      mesh=mesh,
      in_specs=jax.P('X','Y'), out_specs=jax.P('X','Y')
  )
  return shmapped(x)

@functools.partial(jax.jit, static_argnames=['shift'], out_shardings=jax.NamedSharding(mesh, jax.P('X','Y')))
def shift_jit(x, shift: int):
  X, Y = mesh.axis_sizes
  reshaped = x.reshape(X, x.shape[0] // X, -1)
  return jnp.roll(reshaped, shift, axis=1).reshape(x.shape[0], x.shape[1])

x = jnp.arange(8 * 64 * 8, dtype=jnp.float32).reshape(8 * 64, 8)
x = jax.device_put(x, jax.NamedSharding(mesh, jax.P('X','Y')))

y1 = shift_shmap(x, 5)
y2 = shift_jit(x, 5)

np.testing.assert_array_equal(y1, y2)
```

</details>

**问题 2：** 这里将一起构建一个基本的“混合专家”模型。设 **W**: float32[E<sub>X</sub>, D, F] 为一组 E 个“专家”矩阵。设 **A**: float32[S<sub>X</sub>, D] 为激活值，并设 **B**: int32[S<sub>X</sub>] 为一组“路由分配”，其中 B[i] 是范围 `[0, E)` 内的整数，表示希望用哪个矩阵处理对应激活值。我们要用 JAX 编写一个返回 `Out[i] = A[i] @ W[B[i]]` 的函数。

1. 首先完全忽略分片。把所有张量设得足够小，使其能够装入单台设备。编写这个函数的局部实现。*务必不要具体化一个形状为 `[S, D, F]` 的数组！提示：尝试把 token 排序到一个形状为 `[E, S, D]` 的新缓冲区中，同时注意掩码（为什么第二个维度必须为 S？）。*

2. 如果直接对上述方法执行 `jax.jit`，某些事情会发生。对它进行性能剖析，看看系统决定执行什么通信。耗时多久？

3. 你会注意到，上述方法的一个问题是，它很可能会在本地收集完整的激活值集合 **A**，即 AllGather<sub>X</sub>([S<sub>X</sub>, D])。如果无法在本地容纳完整激活值集合，这不仅通信成本极高，内存成本也高得惊人。请使用 `shard_map` 和显式通信实现上述操作。

      1. 第一版实现最简单的方式可能是使用 `jax.lax.all_gather`，再像第 1 步那样重新排序。

      2. 第二版请尝试避免具体化任何大小为 `[E, S, D]` 的数组；也就是说，尝试使用 `jax.lax.all_to_all`，并将其置于 `jax.lax.while_loop` 内，以不规则方式执行计算。这样既能避免具体化完整激活值，也不会把计算浪费在填充上。它比原始实现快多少？

4. 大多数 MoE 会把每个 token 路由到多个（k 个）专家，再对结果取平均。重构上述代码来实现这一点。在这种情况下，令 **B**: int32[S<sub>X</sub>, k] 表示要路由到的 k 个专家。

<details>
<summary>点击此处查看（部分）答案。 </summary>


第 1/2 部分。第（1）部分有许多选择。下面是一种只使用掩码遍历所有专家的方案。

```py
def moe_local(W: jnp.ndarray, A: jnp.ndarray, B: jnp.ndarray) -> jnp.ndarray:
    S, _ = A.shape
    E, _, F = W.shape

    def expert_forward(carry, e):
        output = carry  # [S, F]
        mask = (B == e)[:, None]  # [S, 1]
        expert_result = A @ W[e]  # [S, F] - this expert's transform of ALL tokens
        output = output + expert_result * mask  # Only keep results for assigned tokens
        return output, None

    output = jnp.zeros((S, F))
    output, _ = jax.lax.scan(expert_forward, output, jnp.arange(E))

    return output
```

也可以使用 `jax.lax.ragged_dot`，它会以更高效率完成类似操作。

3. 这里只画出第 3 部分伪代码的轮廓（如果你有简洁的解法，欢迎添加）：

```py
chunk_size = 128
def matmul(W, x, B):
  i = 0
  x = # sort x according to assignments
  while (chunk := x[i:i+chunk_size]).any():
     chunk = all_to_all(chunk)
     out = matmul_local(W, chunk)
     i += chunk_size
  return concat(out)
```

基本思想是遍历数组的各个分块，对它们排序并执行 all_to_all，然后执行局部 FLOPs。

</details>

**问题 3：** 上面的集合矩阵乘法示例其实与真实 LLM 高度相关。下面稍微调整该示例，实现完整的 Transformer 堆栈。

1. 作为练习，先实现一个 AllReduce 集合矩阵乘法，即 A[B<sub>X</sub>, D<sub>Y</sub>] \*<sub>D</sub> W[D<sub>Y</sub>, F] -> Out[B<sub>X</sub>, F]。请注意，输出并未复制。朴素算法已在上文讨论：基本上就是先执行局部矩阵乘法，再执行 AllReduce。请尝试实现这个操作中通信重叠的“集合”版本。*提示：沿输出维度分块，并可以自由使用 `jax.lax.psum`（也就是 AllReduce）。* *注意：由于 XLA 处理这种操作的方式，它实际上可能并不比基线更快。*

2. 上述 AllReduce 集合矩阵乘法的互补操作是 ReduceScatter 集合矩阵乘法，如 Tmp[B<sub>X</sub>, F<sub>Y</sub>] \*<sub>F</sub> W2[F<sub>Y</sub>, D] -> Out[B<sub>X</sub>, D<sub>Y</sub>]。它出现在 Transformer 的下投影矩阵中。请用 JAX 实现该操作中通信重叠的集合版本。注意只传递所需的最少数据。*提示：尝试在累积结果的同时对其进行置换。*

3. 把二者结合成一个端到端 Transformer 块，以重叠通信的方式执行 In[B<sub>X</sub>, D<sub>Y</sub>] \*<sub>D</sub> W<sub>in</sub>[D, F<sub>Y</sub>] \*<sub>F</sub> W<sub>out</sub>[F<sub>Y</sub>, D] -> Out[B<sub>X</sub>, D<sub>Y</sub>]。[^ch10-4] 它比 `jax.jit` 实现快多少？

**问题 4：** 上面实现的所有集合矩阵乘法都是单向的：只沿一个方向进行置换。重写集合 AllReduce 矩阵乘法和集合 ReduceScatter 矩阵乘法，使其使用双向通信。它们会快多少？

<span id="thats-all-for-part-10-thats-basically-it-for-final-conclusions-and-further-reading-click-here"></span>

### 第 10 部分到此结束。基本就是这些！关于最终结论和延伸阅读，请点击[这里](../11-conclusion/#conclusions-and-further-reading)。

[^ch10-1]: 请注意这里的实现方式。这是创建具有特定分片的数组的一种方法（即向创建函数添加 device 参数）。另一种方法是照常用 `jnp.array(....)` 创建数组，再执行例如 `jax.device_put(..., jax.P('X', 'Y'))`。还可以编写一个函数来创建所需数组，再对其执行 jit 编译，把 `out_shardings` 设为所需值。
[^ch10-2]: 如果想在 Colab 中模拟网格并亲自尝试，可以使用下面这个单元格：`import jax; jax.config.update('jax_num_cpu_devices', 8)`。
[^ch10-3]: 如果只想在虚构问题上模拟网格，也可以用 `import jax; jax.config.update('jax_num_cpu_devices', 8)` 在 CPU 上模拟 8 台设备（需要 jax >= 0.4.27 左右），不过这并不能反映真实性能。
[^ch10-4]: 与之前一样，不能先计算 $W_{in} \cdot W_{out}$，因为这里省略了一个非线性操作。
