---
title: A guide on good usage of non_blocking and pin_memory() in PyTorch#
slug: a-guide-on-good-usage-of-nonblocking-and-pinmemory-in-pytorch
date: '2026-02-07'
tags: []
status: published
source_url: 'https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html'
source_author: docs.pytorch.org
imported_at: '2026-02-07T16:52:29.653Z'
source:
  title: docs.pytorch.org
  url: 'https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html'
updated: '2024-07-31'
lang: zh
translatedFrom: en
---

评价此页面

★ ★ ★ ★ ★

<!-- Hidden breadcrumb schema for SEO only -->

non_blocking 和 `pin_memory()` 在 PyTorch 中">

intermediate/pinmem_nonblock

[![](../_static/img/pytorch-colab.svg)](https://colab.research.google.com/github/pytorch/tutorials/blob/gh-pages/_downloads/6a760a243fcbf87fb3368be3d4d860ee/pinmem_nonblock.ipynb)

[在 Google Colab 中运行](https://colab.research.google.com/github/pytorch/tutorials/blob/gh-pages/_downloads/6a760a243fcbf87fb3368be3d4d860ee/pinmem_nonblock.ipynb)

[Colab](https://colab.research.google.com/github/pytorch/tutorials/blob/gh-pages/_downloads/6a760a243fcbf87fb3368be3d4d860ee/pinmem_nonblock.ipynb)

[![](../_static/img/pytorch-download.svg)](https://docs.pytorch.org/tutorials/_downloads/6a760a243fcbf87fb3368be3d4d860ee/pinmem_nonblock.ipynb)

[下载 Notebook](https://docs.pytorch.org/tutorials/_downloads/6a760a243fcbf87fb3368be3d4d860ee/pinmem_nonblock.ipynb)

[Notebook](https://docs.pytorch.org/tutorials/_downloads/6a760a243fcbf87fb3368be3d4d860ee/pinmem_nonblock.ipynb)

[![](../_static/img/pytorch-github.svg)](https://github.com/pytorch/tutorials/blob/main/intermediate_source/pinmem_nonblock.py)

[在 GitHub 上查看](https://github.com/pytorch/tutorials/blob/main/intermediate_source/pinmem_nonblock.py)

[GitHub](https://github.com/pytorch/tutorials/blob/main/intermediate_source/pinmem_nonblock.py)

注意

[前往末尾](#sphx-glr-download-intermediate-pinmem-nonblock-py) 以下载完整示例代码。

# 关于在 PyTorch 中良好使用 和 的指南\#

创建于：2024年7月31日 | 最后更新：2025年3月18日 | 最后验证：2024年11月5日

**作者**：[Vincent Moens](https://github.com/vmoens)

## 引言\#

将数据从 CPU 传输到 GPU 是许多 PyTorch 应用中的基础操作。用户理解可用于设备间数据传输的最有效工具和选项至关重要。本教程探讨了 PyTorch 中设备到设备数据传输的两种关键方法：[`pin_memory()`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.pin_memory.html#torch.Tensor.pin_memory '(in PyTorch v2.10)') 和 [`to()`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.to.html#torch.Tensor.to '(in PyTorch v2.10)') 与 `non_blocking=True` 选项。

### 你将学到什么\#

通过异步传输和内存固定可以优化从 CPU 到 GPU 的张量传输。然而，有一些重要的注意事项：

- 使用 `tensor.pin_memory().to(device, non_blocking=True)` 可能比直接的 `tensor.to(device)` 慢两倍。

- 通常，`tensor.to(device, non_blocking=True)` 是提高传输速度的有效选择。

- 虽然 `cpu_tensor.to("cuda", non_blocking=True).mean()` 执行正确，但尝试 `cuda_tensor.to("cpu", non_blocking=True).mean()` 将导致错误输出。

### 前言\#

本教程中报告的性能取决于构建教程所使用的系统。尽管结论适用于不同系统，但具体观察结果可能会因可用硬件（尤其是较旧的硬件）而略有不同。本教程的主要目标是提供一个理论框架，用于理解 CPU 到 GPU 的数据传输。然而，任何设计决策都应根据具体情况定制，并参考基准吞吐量测量以及任务的具体要求。

```python
import torch

assert torch.cuda.is_available(), "A cuda device is required to run this tutorial"
```

本教程需要安装 tensordict。如果你的环境中还没有 tensordict，请在单独的单元格中运行以下命令进行安装：

```text
# Install tensordict with the following command
!pip3 install tensordict
```

我们首先概述这些概念的理论，然后转向具体测试示例。

## 背景\#

>

### 内存管理基础\#

>

在 PyTorch 中创建 CPU 张量时，该张量的内容需要放置在内存中。这里讨论的内存是一个相当复杂的概念，值得仔细研究。我们区分由内存管理单元处理的两种类型的内存：RAM（为简化起见）和磁盘上的交换空间（可能是硬盘也可能不是）。磁盘和 RAM（物理内存）中的可用空间共同构成虚拟内存，这是可用总资源的抽象。简而言之，虚拟内存使得可用空间大于单独 RAM 中的空间，并创造出主内存比实际更大的错觉。

在正常情况下，常规的 CPU 张量是可分页的，这意味着它被分成称为页的块，这些块可以存在于虚拟内存中的任何位置（无论是在 RAM 中还是在磁盘上）。如前所述，这具有内存看起来比主内存实际更大的优势。

通常，当程序访问不在 RAM 中的页时，会发生“页面错误”，然后操作系统（OS）将该页带回 RAM（“换入”或“页入”）。反过来，OS 可能需要换出（或“页出”）另一页以为新页腾出空间。

与可分页内存相反，固定（或页锁定或不可分页）内存是一种不能被交换到磁盘的内存类型。它允许更快和更可预测的访问时间，但缺点是其比可分页内存（即主内存）更有限。

![](../_images/pinmem.png)

### CUDA 与（非）可分页内存\#

>

为了理解 CUDA 如何将张量从 CPU 复制到 CUDA，让我们考虑上述两种场景：

- 如果内存是页锁定的，设备可以直接在主内存中访问内存。内存地址定义明确，需要读取这些数据的函数可以显著加速。

- 如果内存是可分页的，所有页在发送到 GPU 之前都必须被带到主内存中。此操作可能需要时间，并且比在页锁定张量上执行时更不可预测。

更准确地说，当 CUDA 从 CPU 向 GPU 发送可分页数据时，它必须首先创建该数据的页锁定副本，然后再进行传输。

### 异步与同步操作（CUDA ）\#

>

在执行从主机（例如 CPU）到设备（例如 GPU）的复制时，CUDA 工具包提供了相对于主机同步或异步执行这些操作的模式。

实际上，当调用 [`to()`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.to.html#torch.Tensor.to '(in PyTorch v2.10)') 时，PyTorch 总是调用 [cudaMemcpyAsync](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1g85073372f776b4c4d5f89f7124b7bf79)。如果 `non_blocking=False`（默认），则在每个 `cudaStreamSynchronize` 之后都会调用 `cudaMemcpyAsync`，使得对 [`to()`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.to.html#torch.Tensor.to '(in PyTorch v2.10)') 的调用在主线程中阻塞。如果 `non_blocking=True`，则不会触发同步，主机上的主线程不会被阻塞。因此，从主机的角度来看，多个张量可以同时发送到设备，因为线程不需要等待一个传输完成再启动另一个。

注意

一般来说，传输在设备端是阻塞的（即使它在主机端不是）：设备上的复制不能在另一个操作执行时发生。然而，在一些高级场景中，复制和内核执行可以在 GPU 端同时进行。如下例所示，启用此功能必须满足三个要求：

1. 设备必须至少有一个空闲的 DMA（直接内存访问）引擎。现代 GPU 架构如 Volterra、Tesla 或 H100 设备有多个 DMA 引擎。

1. 传输必须在单独的、非默认的 cuda 流上完成。在 PyTorch 中，可以使用 [`Stream`](https://docs.pytorch.org/docs/stable/generated/torch.cuda.Stream_class.html#torch.cuda.Stream '(in PyTorch v2.10)') 处理 cuda 流。

1. 源数据必须位于固定内存中。

我们通过运行以下脚本的配置文件来演示这一点。

```python
import contextlib

from torch.cuda import Stream

s = Stream()

torch.manual_seed(42)
t1_cpu_pinned = torch.randn(1024**2 * 5, pin_memory=True)
t2_cpu_paged = torch.randn(1024**2 * 5, pin_memory=False)
t3_cuda = torch.randn(1024**2 * 5, device="cuda:0")

assert torch.cuda.is_available()
device = torch.device("cuda", torch.cuda.current_device())

# The function we want to profile
def inner(pinned: bool, streamed: bool):
    with torch.cuda.stream(s) if streamed else contextlib.nullcontext():
        if pinned:
            t1_cuda = t1_cpu_pinned.to(device, non_blocking=True)
        else:
            t2_cuda = t2_cpu_paged.to(device, non_blocking=True)
        t_star_cuda_h2d_event = s.record_event()
    # This operation can be executed during the CPU to GPU copy if and only if the tensor is pinned and the copy is
    #  done in the other stream
    t3_cuda_mul = t3_cuda * t3_cuda * t3_cuda
    t3_cuda_h2d_event = torch.cuda.current_stream().record_event()
    t_star_cuda_h2d_event.synchronize()
    t3_cuda_h2d_event.synchronize()

# Our profiler: profiles the `inner` function and stores the results in a .json file
def benchmark_with_profiler(
    pinned,
    streamed,
) -> None:
    torch._C._profiler._set_cuda_sync_enabled_val(True)
    wait, warmup, active = 1, 1, 2
    num_steps = wait + warmup + active
    rank = 0
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(
            wait=wait, warmup=warmup, active=active, repeat=1, skip_first=1
        ),
    ) as prof:
        for step_idx in range(1, num_steps + 1):
            inner(streamed=streamed, pinned=pinned)
            if rank is None or rank == 0:
                prof.step()
    prof.export_chrome_trace(f"trace_streamed{int(streamed)}_pinned{int(pinned)}.json")
```

在 chrome 中加载这些配置文件跟踪（`chrome://tracing`）显示以下结果：首先，让我们看看如果两个算术运算在`t3_cuda`在可分页张量被发送到主流的GPU之后执行会发生什么：

```text
benchmark_with_profiler(streamed=False, pinned=False)
```

```text
/usr/local/lib/python3.10/dist-packages/torch/profiler/profiler.py:217: UserWarning:

Warning: Profiler clears events at the end of each cycle.Only events from the current cycle will be reported.To keep events across cycles, set acc_events=True.
```

![](../_images/trace_streamed0_pinned0.png)

使用固定（pinned）张量不会显著改变跟踪，两个操作仍然连续执行：

```text
benchmark_with_profiler(streamed=False, pinned=True)
```

![](../_images/trace_streamed0_pinned1.png)

在单独流上将可分页张量发送到GPU也是一个阻塞操作：

```text
benchmark_with_profiler(streamed=True, pinned=False)
```

![](../_images/trace_streamed1_pinned0.png)

只有固定张量在单独流上复制到GPU时，才能与在主流上执行的另一个CUDA内核重叠：

```text
benchmark_with_profiler(streamed=True, pinned=True)
```

![](../_images/trace_streamed1_pinned1.png)

## PyTorch视角\#

>

### \#

>

PyTorch提供了通过[`pin_memory()`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.pin_memory.html#torch.Tensor.pin_memory '(in PyTorch v2.10)')方法和构造函数参数创建并将张量发送到页锁定内存的可能性。在已初始化CUDA的机器上，CPU张量可以通过[`pin_memory()`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.pin_memory.html#torch.Tensor.pin_memory '(in PyTorch v2.10)')方法转换为固定内存。重要的是，`pin_memory`在主机的主线程上是阻塞的：它会等待张量被复制到页锁定内存后再执行下一个操作。新张量可以直接通过诸如[`zeros()`](https://docs.pytorch.org/docs/stable/generated/torch.zeros.html#torch.zeros '(in PyTorch v2.10)')、[`ones()`](https://docs.pytorch.org/docs/stable/generated/torch.ones.html#torch.ones '(in PyTorch v2.10)')和其他构造函数等函数在固定内存中创建。

让我们检查固定内存和将张量发送到CUDA的速度：

```python
import torch
import gc
from torch.utils.benchmark import Timer
import matplotlib.pyplot as plt

def timer(cmd):
    median = (
        Timer(cmd, globals=globals())
        .adaptive_autorange(min_run_time=1.0, max_run_time=20.0)
        .median
        * 1000
    )
    print(f"{cmd}: {median: 4.4f} ms")
    return median

# A tensor in pageable memory
pageable_tensor = torch.randn(1_000_000)

# A tensor in page-locked (pinned) memory
pinned_tensor = torch.randn(1_000_000, pin_memory=True)

# Runtimes:
pageable_to_device = timer("pageable_tensor.to('cuda:0')")
pinned_to_device = timer("pinned_tensor.to('cuda:0')")
pin_mem = timer("pageable_tensor.pin_memory()")
pin_mem_to_device = timer("pageable_tensor.pin_memory().to('cuda:0')")

# Ratios:
r1 = pinned_to_device / pageable_to_device
r2 = pin_mem_to_device / pageable_to_device

# Create a figure with the results
fig, ax = plt.subplots()

xlabels = [0, 1, 2]
bar_labels = [
    "pageable_tensor.to(device) (1x)",
    f"pinned_tensor.to(device) ({r1:4.2f}x)",
    f"pageable_tensor.pin_memory().to(device) ({r2:4.2f}x)"
    f"\npin_memory()={100*pin_mem/pin_mem_to_device:.2f}% of runtime.",
]
values = [pageable_to_device, pinned_to_device, pin_mem_to_device]
colors = ["tab:blue", "tab:red", "tab:orange"]
ax.bar(xlabels, values, label=bar_labels, color=colors)

ax.set_ylabel("Runtime (ms)")
ax.set_title("Device casting runtime (pin-memory)")
ax.set_xticks([])
ax.legend()

plt.show()

# Clear tensors
del pageable_tensor, pinned_tensor
_ = gc.collect()
```

![设备转换运行时（固定内存）](../_images/sphx_glr_pinmem_nonblock_001.png)

```text
pageable_tensor.to('cuda:0'):  0.3691 ms
pinned_tensor.to('cuda:0'):  0.3151 ms
pageable_tensor.pin_memory():  0.1124 ms
pageable_tensor.pin_memory().to('cuda:0'):  0.4325 ms
```

我们可以观察到，将固定内存张量转换为GPU确实比可分页张量快得多，因为在底层，可分页张量必须先复制到固定内存，然后才能发送到GPU。

然而，与一些常见看法相反，在将可分页张量转换为GPU之前调用[`pin_memory()`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.pin_memory.html#torch.Tensor.pin_memory '(in PyTorch v2.10)')不应带来任何显著的速度提升，相反，这个调用通常比直接执行传输更慢。这是合理的，因为我们实际上是在要求Python执行一个CUDA在将数据从主机复制到设备之前无论如何都会执行的操作。

注意

PyTorch的[pin_memory](https://github.com/pytorch/pytorch/blob/5298acb5c76855bc5a99ae10016efc86b27949bd/aten/src/ATen/native/Memory.cpp#L58)实现依赖于通过[cudaHostAlloc](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1gb65da58f444e7230d3322b6126bb4902)创建一个全新的固定内存存储，在极少数情况下，这可能比像`cudaMemcpy`那样分块传输数据更快。同样，观察结果可能因可用硬件、发送的张量大小或可用RAM量而异。

### \#

>

如前所述，许多PyTorch操作具有通过`non_blocking`参数相对于主机异步执行的选项。

这里，为了准确评估使用`non_blocking`的好处，我们将设计一个稍微更复杂的实验，因为我们想要评估在调用和不调用`non_blocking`的情况下将多个张量发送到GPU的速度。

```python
# A simple loop that copies all tensors to cuda
def copy_to_device(*tensors):
    result = []
    for tensor in tensors:
        result.append(tensor.to("cuda:0"))
    return result

# A loop that copies all tensors to cuda asynchronously
def copy_to_device_nonblocking(*tensors):
    result = []
    for tensor in tensors:
        result.append(tensor.to("cuda:0", non_blocking=True))
    # We need to synchronize
    torch.cuda.synchronize()
    return result

# Create a list of tensors
tensors = [torch.randn(1000) for _ in range(1000)]
to_device = timer("copy_to_device(*tensors)")
to_device_nonblocking = timer("copy_to_device_nonblocking(*tensors)")

# Ratio
r1 = to_device_nonblocking / to_device

# Plot the results
fig, ax = plt.subplots()

xlabels = [0, 1]
bar_labels = [f"to(device) (1x)", f"to(device, non_blocking=True) ({r1:4.2f}x)"]
colors = ["tab:blue", "tab:red"]
values = [to_device, to_device_nonblocking]

ax.bar(xlabels, values, label=bar_labels, color=colors)

ax.set_ylabel("Runtime (ms)")
ax.set_title("Device casting runtime (non-blocking)")
ax.set_xticks([])
ax.legend()

plt.show()
```

![设备转换运行时（非阻塞）](../_images/sphx_glr_pinmem_nonblock_002.png)

```text
copy_to_device(*tensors):  16.3513 ms
copy_to_device_nonblocking(*tensors):  12.0479 ms
```

为了更好地理解这里发生的情况，让我们分析这两个函数：

```python
from torch.profiler import profile, ProfilerActivity

def profile_mem(cmd):
    with profile(activities=[ProfilerActivity.CPU]) as prof:
        exec(cmd)
    print(cmd)
    print(prof.key_averages().table(row_limit=10))
```

首先，让我们看看常规`to(device)`的调用栈：

```text
print("Call to `to(device)`", profile_mem("copy_to_device(*tensors)"))
```

```cpp
copy_to_device(*tensors)
-------------------------  ------------  ------------  ------------  ------------  ------------  ------------
                     Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg    # of Calls
-------------------------  ------------  ------------  ------------  ------------  ------------  ------------
                 aten::to         3.89%     788.540us       100.00%      20.269ms      20.269us          1000
           aten::_to_copy        12.84%       2.603ms        96.11%      19.481ms      19.481us          1000
      aten::empty_strided        21.23%       4.304ms        21.23%       4.304ms       4.304us          1000
              aten::copy_        22.90%       4.641ms        62.03%      12.574ms      12.574us          1000
          cudaMemcpyAsync        16.75%       3.394ms        16.75%       3.394ms       3.394us          1000
    cudaStreamSynchronize        22.39%       4.539ms        22.39%       4.539ms       4.539us          1000
-------------------------  ------------  ------------  ------------  ------------  ------------  ------------
Self CPU time total: 20.269ms

Call to `to(device)` None
```

现在是`non_blocking`版本：

```text
print(
    "Call to `to(device, non_blocking=True)`",
    profile_mem("copy_to_device_nonblocking(*tensors)"),
)
```

```cpp
copy_to_device_nonblocking(*tensors)
-------------------------  ------------  ------------  ------------  ------------  ------------  ------------
                     Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg    # of Calls
-------------------------  ------------  ------------  ------------  ------------  ------------  ------------
                 aten::to         5.02%     809.004us        99.85%      16.088ms      16.088us          1000
           aten::_to_copy        16.07%       2.589ms        94.83%      15.279ms      15.279us          1000
      aten::empty_strided        25.91%       4.175ms        25.91%       4.175ms       4.175us          1000
              aten::copy_        31.39%       5.058ms        52.85%       8.515ms       8.515us          1000
          cudaMemcpyAsync        21.46%       3.457ms        21.46%       3.457ms       3.457us          1000
    cudaDeviceSynchronize         0.15%      24.521us         0.15%      24.521us      24.521us             1
-------------------------  ------------  ------------  ------------  ------------  ------------  ------------
Self CPU time total: 16.112ms

Call to `to(device, non_blocking=True)` None
```

毫无疑问，使用`non_blocking=True`的结果更好，因为所有传输在主机端同时启动，并且只进行一次同步。

好处将因张量的数量和大小以及所使用的硬件而异。

注意

有趣的是，阻塞的`to("cuda")`实际上执行了与`cudaMemcpyAsync`相同的异步设备转换操作（`non_blocking=True`），但每次复制后都有一个同步点。

### 协同效应\#

>

既然我们已经指出，将已在固定内存中的张量数据传输到GPU比从可分页内存传输更快，并且我们知道异步执行这些传输也比同步更快，我们可以对这些方法的组合进行基准测试。首先，让我们编写几个新函数，在每个张量上调用`pin_memory`和`to(device)`：

```python
def pin_copy_to_device(*tensors):
    result = []
    for tensor in tensors:
        result.append(tensor.pin_memory().to("cuda:0"))
    return result

def pin_copy_to_device_nonblocking(*tensors):
    result = []
    for tensor in tensors:
        result.append(tensor.pin_memory().to("cuda:0", non_blocking=True))
    # We need to synchronize
    torch.cuda.synchronize()
    return result
```

使用[`pin_memory()`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.pin_memory.html#torch.Tensor.pin_memory '(in PyTorch v2.10)')的好处对于较大批次的大型张量更为明显：

```json
tensors = [torch.randn(1_000_000) for _ in range(1000)]
page_copy = timer("copy_to_device(*tensors)")
page_copy_nb = timer("copy_to_device_nonblocking(*tensors)")

tensors_pinned = [torch.randn(1_000_000, pin_memory=True) for _ in range(1000)]
pinned_copy = timer("copy_to_device(*tensors_pinned)")
pinned_copy_nb = timer("copy_to_device_nonblocking(*tensors_pinned)")

pin_and_copy = timer("pin_copy_to_device(*tensors)")
pin_and_copy_nb = timer("pin_copy_to_device_nonblocking(*tensors)")

# Plot
strategies = ("pageable copy", "pinned copy", "pin and copy")
blocking = {
    "blocking": [page_copy, pinned_copy, pin_and_copy],
    "non-blocking": [page_copy_nb, pinned_copy_nb, pin_and_copy_nb],
}

x = torch.arange(3)
width = 0.25
multiplier = 0

fig, ax = plt.subplots(layout="constrained")

for attribute, runtimes in blocking.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, runtimes, width, label=attribute)
    ax.bar_label(rects, padding=3, fmt="%.2f")
    multiplier += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel("Runtime (ms)")
ax.set_title("Runtime (pin-mem and non-blocking)")
ax.set_xticks([0, 1, 2])
ax.set_xticklabels(strategies)
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
ax.legend(loc="upper left", ncols=3)

plt.show()

del tensors, tensors_pinned
_ = gc.collect()
```

![运行时（固定内存和非阻塞）](../_images/sphx_glr_pinmem_nonblock_003.png)

```text
copy_to_device(*tensors):  394.2794 ms
copy_to_device_nonblocking(*tensors):  316.1562 ms
copy_to_device(*tensors_pinned):  315.1386 ms
copy_to_device_nonblocking(*tensors_pinned):  299.9565 ms
pin_copy_to_device(*tensors):  573.3167 ms
pin_copy_to_device_nonblocking(*tensors):  324.1541 ms
```

### 其他复制方向（GPU -> CPU, CPU -> MPS）\#

>

到目前为止，我们一直假设从CPU到GPU的异步复制是安全的。这通常是正确的，因为CUDA会自动处理同步，以确保在读取时访问的数据有效\_\_每当张量在可分页[内存\_\_](#id2)中时。

然而，在其他情况下，我们不能做出相同的假设：当张量放置在固定内存中时，在调用主机到设备传输后修改原始副本可能会损坏GPU上接收的数据。类似地，当传输在相反方向进行时，从GPU到CPU，或从任何非CPU或GPU的设备到任何非CUDA处理的GPU（例如MPS）的设备时，没有显式同步，无法保证在GPU上读取的数据有效。

在这些场景中，这些传输不能保证复制在数据访问时完成。因此，主机上的数据可能不完整或不正确，实际上使其成为垃圾。

首先，让我们用一个固定内存张量来演示这一点：

```yaml
DELAY = 100000000
try:
    i = -1
    for i in range(100):
        # Create a tensor in pin-memory
        cpu_tensor = torch.ones(1024, 1024, pin_memory=True)
        torch.cuda.synchronize()
        # Send the tensor to CUDA
        cuda_tensor = cpu_tensor.to("cuda", non_blocking=True)
        torch.cuda._sleep(DELAY)
        # Corrupt the original tensor
        cpu_tensor.zero_()
        assert (cuda_tensor == 1).all()
    print("No test failed with non_blocking and pinned tensor")
except AssertionError:
    print(f"{i}th test failed with non_blocking and pinned tensor. Skipping remaining tests")
```

```text
1th test failed with non_blocking and pinned tensor. Skipping remaining tests
```

使用可分页张量总是有效：

```text
i = -1
for i in range(100):
    # Create a tensor in pageable memory
    cpu_tensor = torch.ones(1024, 1024)
    torch.cuda.synchronize()
    # Send the tensor to CUDA
    cuda_tensor = cpu_tensor.to("cuda", non_blocking=True)
    torch.cuda._sleep(DELAY)
    # Corrupt the original tensor
    cpu_tensor.zero_()
    assert (cuda_tensor == 1).all()
print("No test failed with non_blocking and pageable tensor")
```

```text
No test failed with non_blocking and pageable tensor
```

现在让我们演示，没有同步时，CUDA到CPU的复制也无法产生可靠的输出：

```yaml
tensor = (
    torch.arange(1, 1_000_000, dtype=torch.double, device="cuda")
    .expand(100, 999999)
    .clone()
)
torch.testing.assert_close(
    tensor.mean(), torch.tensor(500_000, dtype=torch.double, device="cuda")
), tensor.mean()
try:
    i = -1
    for i in range(100):
        cpu_tensor = tensor.to("cpu", non_blocking=True)
        torch.testing.assert_close(
            cpu_tensor.mean(), torch.tensor(500_000, dtype=torch.double)
        )
    print("No test failed with non_blocking")
except AssertionError:
    print(f"{i}th test failed with non_blocking. Skipping remaining tests")
try:
    i = -1
    for i in range(100):
        cpu_tensor = tensor.to("cpu", non_blocking=True)
        torch.cuda.synchronize()
        torch.testing.assert_close(
            cpu_tensor.mean(), torch.tensor(500_000, dtype=torch.double)
        )
    print("No test failed with synchronize")
except AssertionError:
    print(f"One test failed with synchronize: {i}th assertion!")
```

```text
0th test failed with non_blocking. Skipping remaining tests
No test failed with synchronize
```

通常，只有在目标是CUDA启用的设备且原始张量在可分页内存中时，异步复制到设备才无需显式同步是安全的。

总之，使用`non_blocking=True`时，从CPU复制数据到GPU是安全的，但对于任何其他方向，`non_blocking=True`仍然可以使用，但用户必须确保在访问数据之前执行设备同步。

## 实用建议\#

>

现在，我们可以根据观察总结一些早期建议：

一般来说，`non_blocking=True`将提供良好的吞吐量，无论原始张量是否在固定内存中。如果张量已经在固定内存中，传输可以加速，但从Python主线程手动将其发送到固定内存是主机上的阻塞操作，因此会抵消使用`non_blocking=True`的大部分好处（因为CUDA无论如何都会执行pin_memory传输）。

现在，人们可能会合理地问[`pin_memory()`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.pin_memory.html#torch.Tensor.pin_memory '(in PyTorch v2.10)')方法有什么用。在下一节中，我们将进一步探讨如何利用它来进一步加速数据传输。

## 其他考虑因素\#

>

PyTorch众所周知提供了一个[`DataLoader`](https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader '(in PyTorch v2.10)')类，其构造函数接受一个`pin_memory`论点。考虑到我们之前关于`pin_memory`的讨论，您可能想知道`DataLoader`如何能够加速数据传输，如果内存固定本质上是阻塞的。

关键在于DataLoader使用一个单独的线程来处理从可分页内存到固定内存的数据传输，从而防止主线程中的任何阻塞。

为了说明这一点，我们将使用同名库中的TensorDict原语。当调用[`to()`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDict.html#tensordict.TensorDict.to '(in tensordict v0.11)')时，默认行为是异步将张量发送到设备，随后调用一次`torch.device.synchronize()`。

此外，`TensorDict.to()`包含一个`non_blocking_pin`选项，该选项启动多个线程来执行`pin_memory()`，然后再进行到`to(device)`。这种方法可以进一步加速数据传输，如下例所示。

```python
from tensordict import TensorDict
import torch
from torch.utils.benchmark import Timer
import matplotlib.pyplot as plt

# Create the dataset
td = TensorDict({str(i): torch.randn(1_000_000) for i in range(1000)})

# Runtimes
copy_blocking = timer("td.to('cuda:0', non_blocking=False)")
copy_non_blocking = timer("td.to('cuda:0')")
copy_pin_nb = timer("td.to('cuda:0', non_blocking_pin=True, num_threads=0)")
copy_pin_multithread_nb = timer("td.to('cuda:0', non_blocking_pin=True, num_threads=4)")

# Rations
r1 = copy_non_blocking / copy_blocking
r2 = copy_pin_nb / copy_blocking
r3 = copy_pin_multithread_nb / copy_blocking

# Figure
fig, ax = plt.subplots()

xlabels = [0, 1, 2, 3]
bar_labels = [
    "Blocking copy (1x)",
    f"Non-blocking copy ({r1:4.2f}x)",
    f"Blocking pin, non-blocking copy ({r2:4.2f}x)",
    f"Non-blocking pin, non-blocking copy ({r3:4.2f}x)",
]
values = [copy_blocking, copy_non_blocking, copy_pin_nb, copy_pin_multithread_nb]
colors = ["tab:blue", "tab:red", "tab:orange", "tab:green"]

ax.bar(xlabels, values, label=bar_labels, color=colors)

ax.set_ylabel("Runtime (ms)")
ax.set_title("Device casting runtime")
ax.set_xticks([])
ax.legend()

plt.show()
```

![设备转换运行时](../_images/sphx_glr_pinmem_nonblock_004.png)

```text
td.to('cuda:0', non_blocking=False):  396.8207 ms
td.to('cuda:0'):  317.3885 ms
td.to('cuda:0', non_blocking_pin=True, num_threads=0):  320.5085 ms
td.to('cuda:0', non_blocking_pin=True, num_threads=4):  301.8519 ms
```

在此示例中，我们正在将许多大型张量从CPU传输到GPU。这种情况非常适合利用多线程`pin_memory()`，这可以显著提升性能。然而，如果张量较小，多线程的开销可能超过其带来的好处。同样，如果只有少数张量，在单独线程上固定张量的优势就变得有限。

作为额外说明，虽然创建永久缓冲区在固定内存中以在将张量传输到GPU之前从可分页内存中移动它们似乎是有利的，但这种策略不一定能加速计算。将数据复制到固定内存中固有的瓶颈仍然是一个限制因素。

此外，将驻留在磁盘上的数据（无论是在共享内存还是文件中）传输到GPU通常需要一个中间步骤，即将数据复制到固定内存（位于RAM中）。在此上下文中对大型数据传输使用non_blocking可以显著增加RAM消耗，可能导致不利影响。

在实践中，没有一刀切的解决方案。使用多线程`pin_memory`结合`non_blocking`传输的有效性取决于多种因素，包括具体系统、操作系统、硬件以及正在执行的任务的性质。以下是在尝试加速CPU和GPU之间的数据传输或比较不同场景下的吞吐量时需要检查的因素列表：

- **可用核心数量**

  有多少CPU核心可用？系统是否与其他可能竞争资源的用户或进程共享？

- **核心利用率**

  CPU核心是否被其他进程大量使用？应用程序是否在数据传输的同时执行其他CPU密集型任务？

- **内存利用率**

  当前使用了多少可分页和页面锁定内存？是否有足够的空闲内存来分配额外的固定内存而不影响系统性能？请记住，没有什么是免费的，例如`pin_memory`将消耗RAM并可能影响其他任务。

- **CUDA设备能力**

  GPU是否支持多个DMA引擎以进行并发数据传输？所使用的CUDA设备的具体能力和限制是什么？

- **要发送的张量数量**

  在典型操作中传输了多少张量？

- **要发送的张量大小**

  正在传输的张量大小是多少？少数大型张量或许多小型张量可能不会从相同的传输程序中受益。

- **系统架构**

  系统架构如何影响数据传输速度（例如，总线速度、网络延迟）？

此外，在固定内存中分配大量张量或大型张量可以占用RAM的很大一部分。这减少了其他关键操作（如分页）的可用内存，可能对算法的整体性能产生负面影响。

## 结论\#

>

在本教程中，我们探讨了将张量从主机发送到设备时影响传输速度和内存管理的几个关键因素。我们了解到，使用`non_blocking=True`通常可以加速数据传输，并且[`pin_memory()`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.pin_memory.html#torch.Tensor.pin_memory '(in PyTorch v2.10)')如果正确实施，也可以提升性能。然而，这些技术需要仔细设计和校准才能有效。

请记住，分析您的代码并关注内存消耗对于优化资源使用和实现最佳性能至关重要。

## 额外资源\#

>

如果您在使用CUDA设备时遇到内存复制问题，或想了解更多关于本教程中讨论的内容，请查看以下参考资料：

- [CUDA工具包内存管理文档](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html)；

- [CUDA固定内存说明](https://forums.developer.nvidia.com/t/pinned-memory/268474)；

- [如何在CUDA C/C++中优化数据传输](https://developer.nvidia.com/blog/how-optimize-data-transfers-cuda-cc/)；

- [tensordict文档](https://pytorch.org/tensordict/stable/index.html)和[仓库](https://github.com/pytorch/tensordict)。

**脚本的总运行时间：**（1分钟2.322秒）

[`Download Jupyter notebook: pinmem_nonblock.ipynb`](../_downloads/6a760a243fcbf87fb3368be3d4d860ee/pinmem_nonblock.ipynb)

[`Download Python source code: pinmem_nonblock.py`](../_downloads/562d6bd0e2a429f010fcf8007f6a7cac/pinmem_nonblock.py)

[`Download zipped: pinmem_nonblock.zip`](../_downloads/54407d14cdf41a1a53e1378e68df1aa4/pinmem_nonblock.zip)

<!-- <i class="fa-solid fa-list"></i> Font Awesome fontawesome.com -->

在此页面上

PyTorch库

- [torchao](https://docs.pytorch.org/ao)
- [torchrec](https://docs.pytorch.org/torchrec)
- [torchft](https://docs.pytorch.org/torchft)
- [TorchCodec](https://docs.pytorch.org/torchcodec)
- [torchvision](https://docs.pytorch.org/vision)
- [ExecuTorch](https://docs.pytorch.org/executorch)
- [在XLA设备上的PyTorch](https://docs.pytorch.org/xla)
