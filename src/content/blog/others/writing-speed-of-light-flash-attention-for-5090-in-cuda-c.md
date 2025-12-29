---
title: Writing Speed-of-Light Flash Attention for 5090 in CUDA C++
slug: writing-speed-of-light-flash-attention-for-5090-in-cuda-c
date: '2025-08-23'
tags: []
status: published
source_url: 'https://gau-nernst.github.io/fa-5090/'
source_author: Thien Tran
imported_at: '2025-12-29T16:16:03.283Z'
source:
  title: gau-nernst.github.io
  url: 'https://gau-nernst.github.io/fa-5090/'
updated: '2025-08-23'
cover: /images/others/others-fa-5090/001-d8f57fc4.svg
lang: zh
translatedFrom: en
---

# 为5090编写光速Flash Attention（Flash Attention）的CUDA C++实现

2025年8月23日

在这篇文章中，我将逐步介绍如何学习在CUDA C++中为5090实现Flash Attention（Flash Attention）。主要目标是学习在CUDA C++中编写注意力机制，因为许多功能在[Triton](https://triton-lang.org/main/index.html)中不可用，例如sm120的MXFP8/NVFP4 MMA。我也觉得这是学习矩阵乘法内核后的自然下一步。最后，有[许多](https://alexarmbr.github.io/2024/08/10/How-To-Write-A-Fast-Matrix-Multiplication-From-Scratch-With-Tensor-Cores.html) [优秀](https://www.spatters.ca/mma-matmul) [博客文章](https://cudaforfun.substack.com/p/outperforming-cublas-on-h100-a-worklog)关于编写快速矩阵乘法内核，但没有关于注意力的。所以我想借此机会好好写点东西。

强烈建议读者熟悉CUDA C++以及如何在NVIDIA GPU上使用Tensor核心。当然，您仍然可以阅读，并在过程中用您喜欢的LLM澄清。或者，您可以查看GPU-MODE系列（[幻灯片](https://github.com/gpu-mode/lectures)，[YouTube](https://www.youtube.com/@GPUMODE)）获取基本CUDA C++知识，以及上面提到的优秀矩阵乘法博客文章，以快速上手。

您可以在此处找到本文讨论的完整实现：<https://github.com/gau-nernst/learn-cuda/tree/e83c256/07_attention>。对于`bs=1, num_heads=8, len_query=4096, len_kv = 8192`，5090 @ 400W，使用CUDA 12.9编译，我获得了以下基准测试结果（5090的BF16理论极限为209.5 TFLOPS）

| 内核                          | TFLOPS | SOL百分比 |
| ----------------------------- | ------ | --------- |
| `F.sdpa()`（Flash Attention） | 186.73 | 89.13%    |
| `F.sdpa()`（CuDNN）           | 203.61 | 97.19%    |
| `flash-attn`                  | 190.58 | 90.97%    |
| v1（基础版）                  | 142.87 | 68.20%    |
| v2（共享内存重排）            | 181.11 | 86.45%    |
| v3（两阶段流水线）            | 189.84 | 90.62%    |
| v4（`ldmatrix.x4`用于K和V）   | 194.33 | 92.76%    |
| v5（更好的流水线）            | 197.74 | 94.39%    |

请注意，虽然我在这些实现中仅使用Ampere功能（sm120支持`cp.async.bulk`即TMA，但我这里不使用），我的实现可能无法在早期GPU上高效运行。由于新硬件的改进，您可能需要使用更多技巧在旧GPU上达到光速，例如流水线共享内存到寄存器内存的数据移动。

## Flash Attention算法

让我们从注意力的参考实现开始。

```python
from torch import Tensor

def sdpa(q: Tensor, k: Tensor, v: Tensor):
    # q: [B, Lq, DIM]
    # k: [B, Lk, DIM]
    # v: [B, Lk, DIM]
    D = q.shape[-1]
    scale = D ** -0.5
    attn = (q @ k.transpose(-1, -2)) * scale  # [B, Lq, Lk]
    attn = attn.softmax(dim=-1)
    out = attn @ v  # [B, Lq, DIM]
    return out
```

技术上，如果输入是BF16，某些计算应保持在FP32，尤其是softmax。但为简洁起见，我们省略它们。

我们正在实现[Flash Attention 2论文](https://arxiv.org/abs/2307.08691)中概述的算法。每个线程块负责Q的一个块，我们将沿着KV的序列长度迭代。算法的Python式大纲如下（S和P遵循Flash Attention表示法）。

```python
scale = DIM ** -0.5
for b_idx in range(B):
    for tile_Q_idx in range(Lq // BLOCK_Q):
        ### start of each threadblock's kernel
        tile_O = torch.zeros(BLOCK_Q, DIM)
        tile_Q = load_Q(b_idx, tile_Q_idx)  # [BLOCK_Q, DIM]

        for tile_KV_idx in range(Lk // BLOCK_KV):
            # first MMA: S = Q @ K.T
            # (BLOCK_Q, DIM) x (BLOCK_KV, DIM).T -> (BLOCK_Q, BLOCK_KV)
            tile_Q                               # (BLOCK_Q, DIM)
            tile_K = load_K(b_idx, tile_KV_idx)  # (BLOCK_KV, DIM)
            tile_S = tile_Q @ tile_K.T           # (BLOCK_Q, BLOCK_KV)
            tile_S = tile_S * scale

            # online softmax and rescale tile_O
            ...

            # second MMA: O = P @ V
            # (BLOCK_Q, BLOCK_KV) x (BLOCK_KV, DIM) -> (BLOCK_Q, DIM)
            tile_P                               # (BLOCK_Q, BLOCK_KV)
            tile_V = load_V(b_idx, tile_KV_idx)  # (BLOCK_KV, DIM)
            tile_O += tile_P @ tile_V            # (BLOCK_Q, DIM)

        # normalize output and write results
        store_O(b_idx, tile_Q_idx)
        ### end of each threadblock's kernel
```

隐含`DIM`很小，因此我们可以在内核的整个持续时间内将`tile_Q`保存在寄存器内存中。这就是为什么几乎所有现代模型都使用`head_dim=128`。当然也有例外，例如[MLA](https://arxiv.org/abs/2405.04434)，它使用`head_dim=576`用于Q和K，以及`head_dim=512`用于V。谈到这个，我应该有一天研究[FlashMLA](https://github.com/deepseek-ai/FlashMLA)。

在线softmax相当棘手，难以解释，所以让我们延迟那部分的解释。在高层，您只需要知道在线softmax会将`tile_S`转换为`tile_P`，并重新缩放`tile_O`。

## 版本1 - 基础实现

我们将遵循典型的MMA流程

- 使用[cp.async](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async)从全局内存加载2D数据块到共享内存。这需要Ampere（sm80及更新版本）。
- 使用[ldmatrix](https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions-ldmatrix)从共享内存加载数据到寄存器内存。
- 调用[mma.m16n8k16](https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-fragment-mma-16816-float)进行BF16矩阵乘法（和累加）。

我想首先专注于正确实现算法，因此省略了更复杂的技巧，如共享内存重排和流水线。这减少了错误的可能性，我们稍后将重新审视它们以进行性能优化。

### 全局内存到共享内存数据传输

以下模板化函数执行从全局内存到共享内存的2D块复制。

- 2D块的形状通过`HEIGHT`和`WIDTH`指定。
- `dst`是共享内存地址，`src`是全局内存地址。
- 全局内存`src`是行主序，所以`src_stride`指定移动到下一行的量。
- 共享内存`dst`也是行主序，并将存储为连续块 ->`dst_stride = WIDTH`。

```cpp
#include <cuda_bf16.h>

template <int HEIGHT, int WIDTH, int TB_SIZE>
__device__ inline
void global_to_shared(uint32_t dst, const nv_bfloat16 *src, int src_stride, int tid) {
  constexpr int num_elems = 16 / sizeof(nv_bfloat16);
  constexpr int num_iters = HEIGHT * WIDTH / (TB_SIZE * num_elems);

  for (int iter = 0; iter < num_iters; iter++) {
    const int idx = (iter * TB_SIZE + tid) * num_elems;
    const int row = idx / WIDTH;
    const int col = idx % WIDTH;

    const uint32_t dst_addr = dst + (row * WIDTH + col) * sizeof(nv_bfloat16);
    const nv_bfloat16 *src_addr = src + (row * src_stride + col);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(dst_addr), "l"(src_addr));
  }
}
```

![全局到共享数据传输](/images/others/others-fa-5090/001-d8f57fc4.svg)

从全局内存到共享内存的2D块复制。

我们将使用内联汇编编写`cp.async.cg.shared.global`。此PTX执行16字节传输，或每个CUDA线程8个BF16元素（`num_elems = 16 / sizeof(nv_bfloat16)`）。为确保合并内存访问，连续线程将负责连续的8xBF16组。

![合并内存访问](/images/others/others-fa-5090/002-b5fea70a.svg)

连续线程负责连续的8xBF16组。

注意：

- 循环`for (int iter = 0; iter < num_iters; iter++)`这样编写，以便编译器（`nvcc`）可以完全展开循环。`num_iters`在编译时已知（由`constexpr`保证）。如果我们在循环中混合`tid`，这对编译器是“动态”变量，循环无法展开，即使我们知道变量的某些约束，即`tid < TB_SIZE`。
- 共享内存指针`dst`的数据类型是`uint32_t`。这是故意的。几乎所有PTX指令都期望共享内存地址在[共享状态空间](https://docs.nvidia.com/cuda/parallel-thread-execution/#state-spaces)中。我们可以使用`static_cast<uint32_t>(__cvta_generic_to_shared(ptr))`将C++指针（通用地址）转换为共享状态空间地址。这在`global_to_shared()`外部完成。

要完成使用`cp.async`，我们还需要添加以下内容：

- `cp.async.commit_group`（PTX）：提交所有先前发出的`cp.async`指令到一&#x4E2A;**`cp.async`组**。此组将是同步的单位。
- `cp.async.wait_all`（PTX）：等待所有提交的组完成。
- `__syncthreads()`：确保所有线程（在线程块中）在读取共享内存中加载的数据之前到达此处（因为一个线程可能读取另一个线程加载的数据）。更重要的是，这将新数据的**可见性**广播到线程块中的所有线程。没有`__syncthreads()`，编译器可以自由优化掉内存访问。

一如既往，请参考[PTX文档](https://docs.nvidia.com/cuda/parallel-thread-execution/)获取有关指令的更多信息。基本上，我们发出多个`cp.async`并立即等待它们完成。`commit_group`和`wait_group`为我们提供了一种机制，以便稍后实现流水线。但现在，只需要知道我们必须这样写才能使用`cp.async`。

我们的代码片段将看起来像这样。

```cpp
// nv_bfloat16 *Q;
// uint32_t Q_smem;
// const int tid = blockIdx.x;
// constexpr int TB_SIZE = 32 * 4;
// constexpr int DIM = 128;

global_to_shared<BLOCK_Q, DIM, TB_SIZE>(Q_smem, Q, DIM, tid);
asm volatile("cp.async.commit_group;");
asm volatile("cp.async.wait_all;");
__syncthreads();
```

### 共享内存到寄存器内存数据传输

在进行全局到共享内存的数据传输时，我们以线程块瓦片和单个CUDA线程为单位思考。对于共享内存到寄存器的数据传输，由于这是为后续MMA指令服务的，我们以warp瓦片/MMA瓦片和warp为单位思考。遵循Flash Attention 2（第3.3节），我们让线程块中的每个warp处理一部分`tile_Q`，沿Q序列长度维度进行分割。这意味着不同的warp将索引到`tile_Q`的不同块，但它们都索引到相同的`tile_K`和`tile_V`块在KV序列长度循环中。

![Flash Attention warp partition](/images/others/others-fa-5090/003-ba293e9f.svg)

Flash Attention 2中的warp分区。

由于我们使用`mma.m16n8k16`指令，每个MMA 16x8输出瓦片（`m16n8`）需要16x16 A瓦片（`m16k16`）和8x16 B瓦片（`n8k16`）。`ldmatrix`可以加载一个、两个或四个8x8瓦片的16位元素。因此，

- A瓦片`m16k16`需要四个8x8瓦片 ->`ldmatrix.x4`
- B瓦片`n8k16`需要两个8x8瓦片 ->`ldmatrix.x2`

只有Q在MMA中充当A。K和V都在其MMA中充当B，尽管K需要转置的`ldmatrix`以获取正确布局（所有张量在全局和共享内存中使用行优先布局）。

要使用`ldmatrix`，每个线程提供一行的地址。线程0-7选择第1个8x8瓦片，线程8-15选择第2个8x8瓦片，依此类推。官方PTX文档中A的[布局](https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-fragment-mma-16816-float)可能看起来令人困惑。但更容易（至少对我来说）专注于MMA瓦片内8x8瓦片的顺序。

![ldmatrix for MMA layout](/images/others/others-fa-5090/004-182a0351.svg)

在`ldmatrix`中`mma.m16n8k16`瓦片的顺序。

通过上面的可视化，我希望以下代码片段能讲得通

```cpp
constexpr int MMA_M = 16;
constexpr int MMA_N = 8;
constexpr int MMA_K = 16;

uint32_t Q_smem;
uint32_t Q_rmem[WARP_Q / MMA_M][DIM / MMA_K][4];

for (int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++)
  for (int mma_id_d = 0; mma_id_d < DIM / MMA_K; mma_id_d++) {
    const int row = (warp_id * WARP_Q) + (mma_id_q * MMA_M) + (lane_id % 16);
    const int col = (mma_id_d * MMA_K) + (lane_id / 16 * 8);
    const uint32_t addr = Q_smem + (row * DIM + col) * sizeof(nv_bfloat16);
    ldmatrix_x4(Q_rmem[mma_id_q][mma_id_d], addr);
  }
```

- 两个嵌套循环将`[MMA_M, MMA_K]`（即`[16, 16]`）在共享内存中瓦片化。`[WARP_Q, DIM]`选择warp瓦片。我们不需要为K和V这样做。
- `(warp_id * WARP_Q)`在
- `(mma_id_q * MMA_M)`和`row`在`(mma_id_d * MMA_K)`中选择MMA瓦片。`col`在
- `(lane_id % 16)`和`row`在`(lane_id / 16 * 8)`中为每个线程选择正确的行地址，遵循所需的被乘数A布局（见上图）。`col`是

`ldmatrix_x4()`的一个小包装，为了方便。你可以参考`ldmatrix.sync.aligned.m8n8.x4.b16`common.h[获取更多细节。](https://github.com/gau-nernst/learn-cuda/blob/e83c256/07_attention/common.h)K和V可以类似地从共享内存加载到寄存器内存。需要注意的一点是关于使用

时的行优先/列优先布局。无论是否使用`ldmatrix`修饰符，每个线程仍然提供8x8瓦片中每行的行地址。`.trans`只改变`.trans`结果的**寄存器布局**。`ldmatrix` results.

![ldmatrix for K and V](/images/others/others-fa-5090/005-7d5699b1.svg)

。`ldmatrix`一个判断是否使用转置版本

的技巧是查看K维度或归约维度。第一个MMA的K维度沿`ldmatrix`维度，而第二个MMA的K维度沿`DIM`维度。`BLOCK_KV` dimension.

### 我们有了高级的基于瓦片的设计，并知道如何为MMA加载数据。调用MMA很简单——只需在我们的代码中插入

PTX。我们的草稿版本看起来像这样。`mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`现在，让我们处理在线softmax。

```cpp
constexpr int BLOCK_Q = 128;
constexpr int BLOCK_KV = 64;
constexpr int DIM = 128;
constexpr int NUM_WARPS = 4;
constexpr int TB_SIZE = NUM_WARPS * 32;

// mma.m16n8k16
constexpr int MMA_M = 16;
constexpr int MMA_N = 8;
constexpr int MMA_K = 16;

__global__
void attention_v1_kernel(
  const nv_bfloat16 *Q,  // [bs, len_q, DIM]
  const nv_bfloat16 *K,  // [bs, len_kv, DIM]
  const nv_bfloat16 *V,  // [bs, len_kv, DIM]
  nv_bfloat16 *O,        // [bs, len_q, DIM]
  int bs,
  int len_q,
  int len_kv) {

  // basic setup
  const int tid = threadIdx.x;
  const int warp_id = tid / 32;
  const int lane_id = tid % 32;

  // increment Q, K, V, O based on blockIdx.x
  ...

  // set up shared memory
  // Q_smem is overlapped with (K_smem + V_smem), since we only use Q_smem once
  extern __shared__ uint8_t smem[];
  const uint32_t Q_smem = __cvta_generic_to_shared(smem);
  const uint32_t K_smem = Q_smem;
  const uint32_t V_smem = K_smem + BLOCK_KV * DIM * sizeof(nv_bfloat16);

  // FA2: shard BLOCK_Q among warps
  constexpr int WARP_Q = BLOCK_Q / NUM_WARPS;

  // set up register memory
  uint32_t Q_rmem[WARP_Q / MMA_M][DIM / MMA_K][4];       // act as A in MMA
  uint32_t K_rmem[BLOCK_KV / MMA_N][DIM / MMA_K][2];     // act as B in MMA
  uint32_t P_rmem[WARP_Q / MMA_M][BLOCK_KV / MMA_K][4];  // act as A in MMA
  uint32_t V_rmem[BLOCK_KV / MMA_K][DIM / MMA_N][2];     // act as B in MMA
  float O_rmem[WARP_Q / MMA_M][DIM / MMA_N][4];          // act as C/D in MMA

  // Q global->shared [BLOCK_Q, DIM]
  global_to_shared<BLOCK_Q, DIM, TB_SIZE>(Q_smem, Q, DIM, tid);
  asm volatile("cp.async.commit_group;");
  asm volatile("cp.async.wait_all;");
  __syncthreads();

  // Q shared->register. select the correct warp tile
  // Q stays in registers throughout the kernel's lifetime
  for (int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++)
    for (int mma_id_d = 0; mma_id_d < DIM / MMA_K; mma_id_d++) {
      const int row = warp_id * WARP_Q + mma_id_q * MMA_M + (lane_id % 16);
      const int col = mma_id_d * MMA_K + (lane_id / 16 * 8);
      const uint32_t addr = Q_smem + (row * DIM + col) * sizeof(nv_bfloat16);
      ldmatrix_x4(Q_rmem[mma_id_q][mma_id_d], addr);
    }
  __syncthreads();

  // main loop
  const int num_kv_iters = len_kv / BLOCK_KV;
  for (int kv_idx = 0; kv_idx < num_kv_iters; kv_idx++) {
    // accumulator for the 1st MMA. reset to zeros
    float S_rmem[WARP_Q / MMA_M][BLOCK_KV / MMA_N][4] = {};  // act as C/D in MMA

    // load K global->shared->registers [BLOCK_KV, DIM]
    // similar to loading Q, except we use ldmatrix_x2()
    ...

    // 1st MMA: S = Q @ K.T
    for (int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++)
      for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_N; mma_id_kv++)
        for (int mma_id_d = 0; mma_id_d < DIM / MMA_K; mma_id_d++)
          mma_m16n8k16(Q_rmem[mma_id_q][mma_id_d],
                       K_rmem[mma_id_kv][mma_id_d],
                       S_rmem[mma_id_q][mma_id_kv]);

    // online softmax. we will touch on this later
    // also pack S_rmem to P_rmem for the 2nd MMA
    ...

    // load V global->shared->registers [BLOCK_KV, DIM]
    // similar to loading K, except we use ldmatrix_x2_trans()
    ...

    // 2nd MMA: O += P @ V
    // similar to the 1st MMA
    ...

    // increment pointer to the next KV block
    K += BLOCK_KV * DIM;
    V += BLOCK_KV * DIM;
  }

  // write output
  for (int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++)
    for (int mma_id_d = 0; mma_id_d < DIM / MMA_N; mma_id_d++) {
      const int row = warp_id * WARP_Q + mma_id_q * MMA_M + (lane_id / 4);
      const int col = mma_id_d * MMA_N + (lane_id % 4) * 2;

      float *regs = O_rmem[mma_id_q][mma_id_d];
      reinterpret_cast<nv_bfloat162 *>(O + (row + 0) * DIM + col)[0] = __float22bfloat162_rn({regs[0], regs[1]});
      reinterpret_cast<nv_bfloat162 *>(O + (row + 8) * DIM + col)[0] = __float22bfloat162_rn({regs[2], regs[3]});
    }
}

// kernel launcher
void attention_v1(
  const nv_bfloat16 *Q,  // [bs, len_q, DIM]
  const nv_bfloat16 *K,  // [bs, len_kv, DIM]
  const nv_bfloat16 *V,  // [bs, len_kv, DIM]
  nv_bfloat16 *O,        // [bs, len_q, DIM]
  int bs,
  int len_q,
  int len_kv) {

  // 1 threadblock for each BLOCK_Q
  const int num_blocks = bs * cdiv(len_q, BLOCK_Q);

  // Q overlap with K+V.
  const int smem_size = max(BLOCK_Q, BLOCK_KV * 2) * DIM * sizeof(nv_bfloat16);

  // use dynamic shared memory so we can allocate more than 48kb if needed.
  if (smem_size > 48'000)
    CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

  attention_v1_kernel<<<num_blocks, TB_SIZE, smem_size>>>(Q, K, V, O, bs, len_q, len_kv);
  CUDA_CHECK(cudaGetLastError());
}
```

Now, let’s tackle online softmax.

### 对于原始解释，你可以参考

Online normalizer calculation for softmax[和Flash Attention 2论文。](https://arxiv.org/abs/1805.02867)我们有以下softmax的数学定义。对于每个长度为$L\_{kv}$的行

$p\_l = \frac{\exp(s\_l-m)}{\exp(s\_0-m) + \exp(s\_1-m) + \dots + \exp(s\_{L\_{kv}-1}-m)}$

$$
l\in\[0,L\_{kv})
$$

$m=\max(s\_0,s\_1,\dots,s\_{L\_{kv}-1})$

$-m$是最大值减法，以提高数值稳定性（如果输入很大，$\exp(\cdot)$很容易爆炸）。让我们提取分母归一化器，并将整行写为向量。

$$
\vec P = \begin{bmatrix} p\_0 \\\ \vdots \\\ p\_{L\_{kv}-1} \end{bmatrix} = \frac{1}{\sum\_{l\in\[0,L\_{kv})}\exp(s\_l-m)} \begin{bmatrix} \exp(s\_0-m) \\\ \vdots \\\ \exp(s\_{L\_{kv}-1}-m) \end{bmatrix}
$$

在我们的第二个矩阵乘法

中，P（softmax输出）的每一行与V的相应列进行点积。`O += P @ V`

$$
o=\vec P \cdot \vec V = \frac{1}{\sum\_{l\in\[0,L\_{kv})}\exp(s\_l-m)} \sum\_{l\in\[0,L\_{kv})}\exp(s\_l-m) \cdot v\_l
$$

额外的点积是塞翁失马——我们不再需要行中的单个元素来获取最终结果。这使得Flash Attention能够一次性计算注意力。为了更清楚地看到这一点，让我们考虑在线计算期间添加新元素的迭代过程。

$$
o\_{\[0,L)} = \frac{1}{\sum\_{l\in\[0,L)}\exp(s\_l-m\_{\[0,L)})} \sum\_{l\in\[0,L)}\exp(s\_l-m\_{\[0,L)}) \cdot v\_l
$$

$$
m\_{\[0,L)}=\max(s\_0,s\_1,\dots,s\_{L-1})
$$

我在这里滥用了符号，但我希望我能传达这个想法。当我们添加一个新元素$s\_{L+1}$

$$
o\_{\[0,L+1)} = \frac{1}{\sum\_{l\in\[0,L+1)}\exp(s\_l-m\_{\[0,L+1)})} \sum\_{l\in\[0,L+1)}\exp(s\_l-m\_{\[0,L+1)}) \cdot v\_l
$$

看归一化器（分母）

$$
\sum\_{l\in\[0,L+1)}\exp(s\_l-m\_{\[0,L+1)}) = \colorbox{red}{
$$

\displaystyle\exp(m\_{\[0,L)}-m\_{\[0,L+1)})$}\colorbox{orange}{$\displaystyle\sum\_{l\in\[0,L)}\exp(s_l-m\_{\[0,L)})$} + \colorbox{lime}{$\displaystyle\exp(s_L-m\_{\[0,L+1)})$}$

这个方程意味着我们只需要在添加

新项**之前**重新缩放

先前的归一化&#x5668;**。同样的逻辑可以应用于与V的点积（未归一化输出）。**

这是在线softmax和Flash Attention的关键思想

。**定义**注意力状态

$$
\begin{bmatrix} m \\\ \tilde{o} \\\ \mathrm{sumexp} \end{bmatrix}
$$

**其中$m$是到目前为止看到的元素的最大值，$\tilde{o}$是**未归一化

的输出，而$\mathrm{sumexp}$是归一化器。我们需要$m$来计算如上所示的重新缩放因子。

你可以确信更新注意力状态是一个[结合](https://pytorch.org/blog/flash-decoding/)操作——元素用于更新注意力状态的顺序无关紧要。

### 这种结合性质使得像

Flash Decoding

```python
# attention state
m = torch.zeros(BLOCK_Q)
tile_O = torch.zeros(BLOCK_Q, DIM)
sumexp = torch.zeros(BLOCK_Q)

for _ in range(Lk // BLOCK_KV):
  # 1st MMA
  tile_S = tile_Q @ tile_K.T  # [BLOCK_Q, BLOCK_KV]
  tile_S = tile_S * scale

  # online softmax
  tile_max = tile_S.amax(dim=-1)  # [BLOCK_Q]
  new_m = torch.maximum(m, tile_max)
  tile_P = torch.exp(tile_S - new_m.unsqueeze(-1))

  # rescale
  scale = torch.exp(m - new_m)
  tile_O *= scale.unsqueeze(-1)
  sumexp = sumexp * scale + tile_P.sum(dim=-1)
  m = new_m  # save new max

  # 2nd MMA
  tile_O += tile_P @ tile_V  # [BLOCK_Q, DIM]

# apply normalization
tile_O /= sumexp.unsqueeze(-1)
```

#### Row max

将其翻译为 CUDA C++ 时，最棘手的部分是理解 MMA 布局。让我们从`tile_S`开始。

![MMA m16n8k16 输出布局](/images/others/others-fa-5090/006-c90e5d1a.png)

MMA m16n8k16 输出的线程和寄存器布局。来源：[NVIDIA PTX 文档](https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-fragment-mma-16816-float)。

Softmax 缩放对所有元素应用相同的缩放，因此这很简单。接下来，我们需要计算当前图块的行最大值。请记住，我们为`tile_S`分配寄存器的方式。

```cpp
float S_rmem[WARP_Q / MMA_M][BLOCK_KV / MMA_N][4];
```

`4`意味着`c0,c1,c2,c3`在上图中，即每个线程持有来自 2 行的 2 个连续元素。要在行内（MMA 输出图块）进行归约，我们先对线程持有的 2 个连续元素进行归约，然后在 4 个线程的组内进行归约，即`T0-T3`、`T4-T7`等。然而，行归约实际上是在整个`tile_S`内进行的，因此我们还需要循环遍历`BLOCK_KV / MMA_N`的`S_rmem`。这可以在 4 线程归约之前与线程级归约结合进行。

![行归约](/images/others/others-fa-5090/007-9e2fcb2f.svg)

在 MMA 输出上执行行归约。

```cpp
// initial attention state
float rowmax[WARP_Q / MMA_M][2];
float rowsumexp[WARP_Q / MMA_M][2] = {};
for (int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++) {
  rowmax[mma_id_q][0] = -FLT_MAX;
  rowmax[mma_id_q][1] = -FLT_MAX;
}

// main loop
const int num_kv_iters = len_kv / BLOCK_KV;
for (int kv_idx = 0; kv_idx < num_kv_iters; kv_idx++) {
  // tile_S = tile_Q @ tile_K.T
  S_rmem[][] = ...

  // loop over rows
  for (int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++) {
    // apply softmax scale
    for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_N; mma_id_kv++)
      for (int reg_id = 0; reg_id < 4; reg_id++)
        S_rmem[mma_id_q][mma_id_kv][reg_id] *= softmax_scale;

    // rowmax
    float this_rowmax[2] = {-FLT_MAX, -FLT_MAX};
    for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_N; mma_id_kv++) {
      float *regs = S_rmem[mma_id_q][mma_id_kv];
      this_rowmax[0] = max(this_rowmax[0], max(regs[0], regs[1]));  // c0 and c1
      this_rowmax[1] = max(this_rowmax[1], max(regs[2], regs[3]));  // c2 and c3
    }

    // butterfly reduction within 4 threads
    this_rowmax[0] = max(this_rowmax[0], __shfl_xor_sync(0xFFFF'FFFF, this_rowmax[0], 1));
    this_rowmax[0] = max(this_rowmax[0], __shfl_xor_sync(0xFFFF'FFFF, this_rowmax[0], 2));
    this_rowmax[1] = max(this_rowmax[1], __shfl_xor_sync(0xFFFF'FFFF, this_rowmax[1], 1));
    this_rowmax[1] = max(this_rowmax[1], __shfl_xor_sync(0xFFFF'FFFF, this_rowmax[1], 2));
  }

  ...
}
```

在典型的归约内核中，当只剩下 32 个活动线程时，我们可以使用 warp shuffle[\_\_shfl_down_sync()](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#warp-shuffle-functions)将数据从较高通道复制到较低通道，最终结果存储在线程 0 中。在这种情况下，由于我们需要在组内的 4 个线程之间共享最大值（以便后续进行最大值减法），我们可以使用`__shfl_xor_sync()`来避免额外的广播步骤。

![蝶形归约](/images/others/others-fa-5090/008-f1f2f566.svg)

使用 \_\_shfl_xor_sync() 在 4 个线程内进行蝶形归约。

#### 重新缩放

有了新图块的行最大值，我们可以计算（未归一化的）输出的重新缩放因子以及归一化器（每行的 sumexp）。

```cpp
// new rowmax
this_rowmax[0] = max(this_rowmax[0], rowmax[mma_id_q][0]);
this_rowmax[1] = max(this_rowmax[1], rowmax[mma_id_q][1]);

// rescale for previous O
float rescale[2];
rescale[0] = __expf(rowmax[mma_id_q][0] - this_rowmax[0]);
rescale[1] = __expf(rowmax[mma_id_q][1] - this_rowmax[1]);
for (int mma_id_d = 0; mma_id_d < DIM / MMA_N; mma_id_d++) {
  O_rmem[mma_id_q][mma_id_d][0] *= rescale[0];
  O_rmem[mma_id_q][mma_id_d][1] *= rescale[0];
  O_rmem[mma_id_q][mma_id_d][2] *= rescale[1];
  O_rmem[mma_id_q][mma_id_d][3] *= rescale[1];
}

// save new rowmax
rowmax[mma_id_q][0] = this_rowmax[0];
rowmax[mma_id_q][1] = this_rowmax[1];
```

我们不在这里重新缩放`rowsumexp`，因为我们希望稍后将其与新 sumexp 项的加法融合，即 FMA - 融合乘加。我们无法将乘法与 MMA 融合，因此需要为`O_rmem`进行单独的乘法。

#### 将 打包到 （并计算行 sum exp）

对于下一部分，我们将再次循环遍历行维度（`BLOCK_KV / MMA_N`），以计算并从`tile_P`打包`tile_S`，同时进行 sumexp 的归约。回想一下，我们为`S`和`P`声明寄存器的方式如下。

```cpp
float S_rmem[WARP_Q / MMA_M][BLOCK_KV / MMA_N][4]      // m16n8
uint32_t P_rmem[WARP_Q / MMA_M][BLOCK_KV / MMA_K][4];  // m16k16
```

再次在 PTX 文档中查找 MMA 乘数 A 和输出 C/D 的线程/寄存器布局。幸运的是，布局完全相同——在 8x8 图块内，元素的排列是相同的。

![MMA m16n8k16 的寄存器布局](/images/others/others-fa-5090/009-fe1cdc5e.svg)

乘数 A 的左半部分与累加器具有相同的布局。来源：[NVIDIA PTX 文档](https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-fragment-mma-16816-float)。

这意味着对于所有线程，`S_rmem`中的每 2 个浮点数可以打包为 BF16x2 存储在`P_rmem`的单个 32 位寄存器中，这正是`mma.m16n8k16`对第 2 个 MMA 的期望方式。没有跨线程的数据移动。请注意，这并不总是成立：如果我们对第 1 个和/或第 2 个 MMA 使用 INT8 或 FP8 MMA，则需要跨线程置换数据以将`tile_S`打包到`tile_P`。

我们在线性 softmax 最后部分的代码如下。

```cpp
// rowsumexp
float this_rowsumexp[2] = {};
for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_N; mma_id_kv++) {
  float *regs = S_rmem[mma_id_q][mma_id_kv];
  regs[0] = __expf(regs[0] - rowmax[mma_id_q][0]);  // c0
  regs[1] = __expf(regs[1] - rowmax[mma_id_q][0]);  // c1
  regs[2] = __expf(regs[2] - rowmax[mma_id_q][1]);  // c2
  regs[3] = __expf(regs[3] - rowmax[mma_id_q][1]);  // c3

  this_rowsumexp[0] += regs[0] + regs[1];
  this_rowsumexp[1] += regs[2] + regs[3];

  // pack to P registers for next MMA
  // we need to change from m16n8 to m16k16
  // each iteration of this loop packs half of m16k16
  nv_bfloat162 *this_P_rmem = reinterpret_cast<nv_bfloat162 *>(P_rmem[mma_id_q][mma_id_kv / 2]);
  this_P_rmem[(mma_id_kv % 2) * 2]     = __float22bfloat162_rn({regs[0], regs[1]});  // top row
  this_P_rmem[(mma_id_kv % 2) * 2 + 1] = __float22bfloat162_rn({regs[2], regs[3]});  // bottom row
}

// butterfly reduction on this_rowsumexp[2]
...

// accumulate to total rowsumexp using FMA
rowsumexp[mma_id_q][0] = rowsumexp[mma_id_q][0] * rescale[0] + this_rowsumexp[0];
rowsumexp[mma_id_q][1] = rowsumexp[mma_id_q][1] * rescale[1] + this_rowsumexp[1];
```

之后是第 2 个 MMA：加载 V，然后计算`tile_O += tile_P @ tile_V`。这完成了我们 Flash Attention 的第 1 个版本。实际上，我们还需要在将`O_rmem`写入全局内存之前对输出进行归一化，但该逻辑应该相当直接。

您可以在[attention_v1.cu](https://github.com/gau-nernst/learn-cuda/blob/e83c256/07_attention/attention_v1.cu)找到版本 1 的完整代码。

### 基准测试设置

哇，对于第 1 个版本来说内容很丰富。确实，我在版本 1 上花费了最多时间，试图正确实现 Flash Attention。我花了 2 天才意识到[\_\_shfl_xor_sync() 的掩码应为 2（0b10）而不是 0x10 以进行蝶形归约](https://github.com/gau-nernst/learn-cuda/commit/8fdb3e6a)。

无论如何，现在我们需要一个脚本来进行正确性检查以及速度基准测试。我更喜欢在 Python PyTorch 中做这些事情，因为它很容易操作，并且可以轻松地与其他具有 PyTorch 绑定的注意力内核进行比较。为此，我创建了：

1. [attention.cpp](https://github.com/gau-nernst/learn-cuda/blob/e83c256/07_attention/attention.cpp)：为我的注意力内核提供 PyTorch 绑定。
1. [main.py](https://github.com/gau-nernst/learn-cuda/blob/e83c256/07_attention/main.py)：正确性检查和速度基准测试。

对于正确性检查，我对比`F.sdpa()`，它默认应调度 Flash Attention 2（至少在我的 GPU 和当前 PyTorch 版本上）。我还特意为随机输入添加了一个小偏置，这些输入从标准正态分布中采样，以便输出具有正均值。这是为了避免由零均值引起的大相对误差。

```python
def generate_input(*shape):
    return torch.randn(shape).add(0.5).bfloat16().cuda()
```

对于速度基准测试，通常最好与（1）硬件的理论极限即光速，以及（2）已知的良好实现进行比较。我更关注注意力的计算受限区域，因此我将使用 FLOPS（每秒浮点运算次数，大写 S）作为比较指标。

要计算给定内核的 FLOPS，我们统计所需的浮点运算次数（FLOPs，小写 s），然后除以延迟。仅从 MMA 统计 FLOPs 应该足够好，结果是`4 * bsize * num_heads * len_q * len_kv * head_dim`。

“已知的良好实现”是`F.sdpa()`的 FA2 和 CuDNN 后端，以及来自[flash-attn](https://github.com/Dao-AILab/flash-attention)包的 FA2。对于我的内核，我确实对`BLOCK_Q`和`BLOCK_KV`进行了一些调优，并获得了以下结果。

| 内核                          | TFLOPS | SOL 百分比 |
| ----------------------------- | ------ | ---------- |
| `F.sdpa()`（Flash Attention） | 186.73 | 89.13%     |
| `F.sdpa()`（CuDNN）           | 203.61 | 97.19%     |
| `flash-attn`                  | 190.58 | 90.97%     |
| v1（基础版）                  | 142.87 | 68.20%     |

对于第一个版本来说，看起来还不错，但我们仍有提升空间。这没关系，因为我们在下一个版本中还有一些技巧可用。事实上，这些技巧与优化矩阵乘法内核时使用的技巧完全相同。

#### 性能分析

在进入下一个版本之前，我想谈谈性能分析工具。我认为使用性能分析作为优化指南总是一个好主意。以前我只知道如何非常基础地使用[ncu](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html)。看到这么多人使用[Nsight Compute](https://developer.nvidia.com/nsight-compute)并带有酷炫的图表，我决定学习如何使用它，实际上它相当容易上手。

Nsight Compute 可以在 macOS 上运行，通过 SSH 访问另一台配备 NVIDIA GPU 的机器，这正是我目前使用的设置（是的，我专门在我的 Macbook 上写代码）。如果您不熟悉 Nsight Compute，我建议观看一两个教程来熟悉它。

要启用源代码检查功能，请记住向 NVCC 传递`-lineinfo`（参见[此处](https://github.com/gau-nernst/learn-cuda/blob/e83c256/07_attention/main.py#L22)），并在 Nsight Compute 中启用“导入源代码”。

## 版本 2 - 共享内存置换

让我们用 Nsight Compute 进行一次性能分析，并查看**Warp 状态统计（Warp State Statistics）** 部分。

![Warp state statistics of v1](/images/others/others-fa-5090/010-0e73a50b.png)

内核 v1 的 Warp 状态统计（Warp State Statistics）。

**Stall Math Pipe Throttle** 最高是好的——这意味着 Warp 忙于数学操作，即 Tensor Cores。第二高的是 **Stall Short Scoreboard**。这通常意味着等待对共享内存的访问。您可以查看 [Nsight Compute 文档（Nsight Compute doc）](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html) 并搜索 `stalled_short_scoreboard`。

我们可以通过查看 **内存工作负载分析（Memory Workload Analysis）** 来再次确认这一点，它揭示了几个问题。

![Memory analsysis of v1](/images/others/others-fa-5090/011-2b867178.png)

内核 v1 的内存分析（Memory Analysis）。

- **L1TEX Global Store Access Pattern** 来自存储输出，因为这是我们唯一的全局写入。这不重要，因为当 `len_kv` 很大时，遍历 KV 序列长度的运行时间应占主导。
- **L1TEX Local Load/Store Access Pattern** 是由于寄存器溢出（Register Spilling）。由于是寄存器溢出，每次只溢出和重载 1 个元素是正常的。减少 `BLOCK_Q`（这样我们使用更少的寄存器来保存累加器）将解决此问题，但我的手动调优显示，一些溢出实际上更快。
- **Shared Load Bank Conflicts** 正是我们正在寻找的——导致“Stall Short Scoreboard”的存储体冲突（Bank Conflicts）。

NVIDIA GPU 的共享内存由 32 个存储体（Memory Banks）支持。连续的 4 字节内存地址分配给连续的存储体。当我们使用 `ldmatrix` 从共享内存加载数据到寄存器内存时，这带来了问题。尽管任何文档中都没有明确说明，但 `ldmatrix.x2` 和 `ldmatrix.x4` 每次操作一个 8x8 图块（Tile）。这是好的，因为它使我们的分析更简单：我们只需要考虑加载一个 8x8 图块的情况。

考虑共享内存中一个形状为 8x64、BF16 数据类型的 2D 图块。

![Bank conflicts](/images/others/others-fa-5090/012-a7a0793c.svg)

共享内存中 8x64 BF16 图块的内存存储体分布（Memory Bank Distribution）。

从上图可以看出，当我们加载 8x8 `ldmatrix` 图块时，相同的 4 个存储体 0-3 服务所有 32 个线程，导致 8 路存储体冲突（8-way Bank Conflict）。我不确定为什么 Nsight Compute 报告如上所示的 16 路存储体冲突。我尝试查找 [使用 swizzling 的矩阵乘法博客文章（matmul blogposts with swizzling）](https://alexarmbr.github.io/2024/08/10/How-To-Write-A-Fast-Matrix-Multiplication-From-Scratch-With-Tensor-Cores.html) 和 [NVIDIA 论坛帖子（NVIDIA forum threads）](https://forums.developer.nvidia.com/t/ncu-detects-bank-conflicts-in-matrix-transposition-after-padding/239100/6)，并发现另一种检查存储体冲突的方法是转到 Nsight Compute 的 **Source** 选项卡并检查 **L1 Wavefronts Shared** 和 **L1 Wavefronts Shared Ideal**（我必须手动启用这两列，因为它们默认不显示）。

![Bank conflicts in ldmatrix](/images/others/others-fa-5090/013-f80df431.png)

内核 v1 中 `ldmatrix` 的实际和理想 L1 Wavefronts Shared（Actual and Ideal L1 Wavefronts Shared）。

&#x20;**Actual / Ideal** 的比率为 8，符合我们 8 路存储体冲突的假设。我仍然不确定为什么这个值与 **Details** 选项卡中的值存在差异。

无论如何，这个问题有两种标准解决方案：

1. **填充共享内存（Pad Shared Memory）**。由于 `ldmatrix` 的对齐要求，我们只能将宽度填充 16 字节，相当于 4 个存储体。这意味着当我们转到下一行时，内存存储体偏移 4，避免了存储体冲突。在许多情况下，这已经足够好。然而，这通常相当浪费，因为我们没有利用填充的存储空间。
1. **Swizzle 共享内存地址（Swizzle Shared Memory Address）**。这是黑魔法：您将共享内存地址与一些魔术数字进行 XOR 运算，然后存储体冲突突然消失了！

让我们详细说明第二种方法。我不够聪明发明这个技巧，但至少我希望我能给出一些关于为什么它合理的提示。我们使用 XOR，因为这个操作很好地置换数据——给定固定的第二个输入，输入和输出之间存在一一映射。我们遇到存储体冲突是因为当我们移动到下一行时，我们再次命中相同的内存存储体 -> 我们可以使用这个行索引来置换地址。

具体来说，如果我们查看原始行地址：

- **Bits 0-3** 由于 16 字节对齐约束始终为零。
- **Bits 2-6** 决定存储体索引（Bank Index）。我们只需要关心 bits 4-6，因为低位始终为零（由于对齐）。
- 行步长（Row Stride）决定了当我们移动到下一行时哪些位递增（这是定义上的）。如果我们的 2D 图块宽度是 64 个 BF16 元素，行步长是 128 字节。转到下一行将递增 bit 7，留下 **bits 0-6 不变**（但我们不关心 bits 0-3）。
- 因此，我们可以将行地址的 **bits 4-6** 与行索引的 **bits 0-2** 进行 XOR 运算，这保证每行都会改变。

如果图块宽度不同，例如 32 BF16，我们可以进行相同的推理。还要注意，行索引编码在行地址内，因此我们只需要行地址和行步长来进行 swizzling。

```cpp
// NOTE: stride in bytes
template <int STRIDE>
__device__
uint32_t swizzle(uint32_t index) {
  // no need swizzling
  if constexpr (STRIDE == 16)
    return index;

  uint32_t row_idx = (index / STRIDE) % 8;
  uint32_t bits_to_xor = row_idx / max(64 / STRIDE, 1);
  return index ^ (bits_to_xor << 4);
}
```

要启用此 swizzling，我们需要将其添加到 `cp.async`（写入共享内存）和 `ldmatrix`（从共享内存读取）调用中。

```diff
// for cp.async
- const uint32_t dst_addr = dst + (row * WIDTH + col) * sizeof(nv_bfloat16);
+ const uint32_t dst_addr = swizzle<WIDTH * sizeof(nv_bfloat16)>(dst + (row * WIDTH + col) * sizeof(nv_bfloat16));
asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(dst_addr), "l"(src_addr));

// for ldmatrix
- ldmatrix_x2(K_rmem[mma_id_kv][mma_id_d], addr);
+ ldmatrix_x2(K_rmem[mma_id_kv][mma_id_d], swizzle<DIM * sizeof(nv_bfloat16)>(addr));
```

由于这是矩阵乘法内核中的标准优化，我还为 `ldmatrix` 添加了一个小优化。我在主循环外预计算行地址和 swizzling，这样在热循环中工作更少。当我们在 Warp 图块内迭代 MMA 图块时，我们需要递增地址。然而，swizzling 是一个 XOR 操作，我们不能简单地将 XOR 与加法交换，即 `(a + b) ^ c != (a ^ c) + b`。注意，如果基地址 `a` 有一些对齐，加法就变成了 XOR！即 `100 + 001 == 100 ^ 001`。因此，当递增 `ldmatrix` 的输入地址时，我们将其与列偏移进行 XOR 运算，而不是做加法。行偏移将影响高于 swizzled 位的位，因此我们可以对其保持加法。

```cpp
// K shared->registers
for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_N; mma_id_kv++)
  for (int mma_id_d = 0; mma_id_d < DIM / MMA_K; mma_id_d++) {
    // swizzle(addr + offset) = swizzle(addr) XOR offset
    uint32_t addr = K_smem_thread;
    addr += mma_id_kv * MMA_N * DIM * sizeof(nv_bfloat16);  // row
    addr ^= mma_id_d * MMA_K * sizeof(nv_bfloat16);  // col
    ldmatrix_x2(K_rmem[mma_id_kv][mma_id_d], addr);
  }
```

版本 2：[attention_v2.cu](https://github.com/gau-nernst/learn-cuda/blob/e83c256/07_attention/attention_v2.cu)。

我们可以用 Nsight Compute 验证不再有存储体冲突。基准测试结果显示了一个令人印象深刻的提升（我总是为新版本的内核重新调优 `BLOCK_Q` 和 `BLOCK_KV`）。

| Kernel                       | TFLOPS | % of SOL |
| ---------------------------- | ------ | -------- |
| v1 (basic)                   | 142.87 | 68.20%   |
| v2 (shared memory swizzling) | 181.11 | 86.45%   |

## 版本 3 - 2 级流水线（Version 3 - 2-stage pipelining）

![Warp state statistics of v2](/images/others/others-fa-5090/014-e3f53ee7.png)

内核 v2 的 Warp 状态统计（Warp State Statistics）。

**Stall Short Scoreboard** 不再是问题，因为我们已经用 swizzling 处理了它。现在的问题是：

- **Stall Wait**（在 Nsight Compute 文档中为 `stalled_wait`）：“等待固定延迟执行依赖”，似乎不是大问题。
- **Stall Long Scoreboard**（在 Nsight Compute 文档中为 `stalled_long_scoreboard`）：通常意味着等待全局内存访问。

到目前为止，我们还没有将全局内存操作与计算操作（MMA）重叠。这意味着张量核心在等待全局->共享传输完成时处于空闲状态。这似乎是引入**流水线（pipelining）**&#x6216;**双缓冲（double-buffering）**&#x7684;好时机：分配比所需更多的共享内存，以便我们可以在处理当前迭代时预取下一个迭代的数据。

- 技术上，我们也可以流水线化共享->寄存器的数据传输。这实际上在CUTLASS的[Efficient GEMM文档（Efficient GEMM doc）](https://github.com/NVIDIA/cutlass/blob/v4.1.0/media/docs/cpp/efficient_gemm.md)中提到过。然而，我在我的5090上从未成功实现过。检查我当前代码生成的SASS，我看到在`LDSM`（`ldmatrix`在PTX中）和`HMMA`（半精度`mma`在PTX中）之间存在交错，可能是编译器为了达到类似的内存-计算重叠效果而做的。

让我们讨论**N阶段流水线（N-stage pipelining）**&#x7684;更通用实现。这篇[NVIDIA博客文章（NVIDIA blogpost）](https://developer.nvidia.com/blog/controlling-data-movement-to-boost-performance-on-ampere-architecture/)很好地解释了这个想法，但通常我不太喜欢使用CUDA C++ API（考虑到CUTLASS也不使用，我认为直接使用PTX更有趣）。N阶段意味着在任何时间点都有N个进行中的阶段。这将是我们在内循环中想要保持的**不变性（invariance）**。

- 这与`num_stages`在[triton.Config](https://triton-lang.org/main/python-api/generated/triton.Config.html)中提到的用于自动调优的概念相同。
- 双缓冲是N=2的特殊情况。

```python
num_stages = 4

# set up num_stages buffers
tile_K_buffers = torch.empty(num_stages, BLOCK_KV, DIM)
tile_V_buffers = torch.empty(num_stages, BLOCK_KV, DIM)

# initiate with (num_stages-1) prefetches
# this is async: the code continues before data loading finishes.
for stage_idx in range(num_stages-1):
    tile_K_buffers[stage_idx] = load_K(stage_idx)
    tile_V_buffers[stage_idx] = load_V(stage_idx)

for tile_KV_idx in range(Lk // BLOCK_KV):
    # prefetch tile (num_stages-1) ahead
    # now we have num_stages global->shared inflight.
    # in practice, we need to guard against out of bounds memory access.
    prefetch_idx = tile_KV_idx + num_stages - 1
    tile_K_buffers[prefetch_idx % num_stages] = load_K(prefetch_idx)
    tile_V_buffers[prefetch_idx % num_stages] = load_V(prefetch_idx)

    # select the current tile
    # we need a synchronization mechanism to make sure data loading
    # for this tile has finished.
    # this "consumes" the oldest global->shared inflight, and
    # replaces it with a compute stage.
    tile_K = tile_K_buffers[tile_KV_idx % num_stages]
    tile_V = tile_V_buffers[tile_KV_idx % num_stages]

    # compute attention as normal
    ...
```

NVIDIA工程师/架构师为我们提供了`cp.async.commit_group`和`cp.async.wait_group`来优雅地实现这一点。

- `cp.async.commit_group`：一个`cp.async`组自然地映射到流水线中的一个预取阶段。
- `cp.async.wait_group N`：意味着等待直到最多剩下N个进行中的组。如果我们做`cp.async.wait_group num_stages-1`，意味着我们等待直到最早的预取完成（记住，我们总是有`num_stages`个进行中的预取作为循环不变性）。

在我们实现注意力机制的情况下，有两个小变化。

1. 由于我们已经为K和V消耗了大量共享内存，并且[消费级GPU的共享内存大小通常适中（consumer GPUs typically have modest shared memory size）](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#features-and-technical-specifications-technical-specifications-per-compute-capability)，与服务器级相比，我决定保持为2阶段流水线，这也使代码稍微简单一些。
1. 我们可以拆分K和V的预取，因为发出V的预取可以延迟到第一个MMA之后。第二个变化需要一些小的调整：每个K和V的预取是一个独立的`cp.async`组（以便我们可以独立等待它们）。

我从PyTorch CPU后端维护者[Mingfei Ma](https://github.com/mingfeima)那里学到的一个简洁的编码风格是使用[lambda表达式（lambda expression）](https://github.com/pytorch/pytorch/blob/v2.8.0/aten/src/ATen/native/cpu/int8mm_kernel.cpp#L63)来编写预取代码。它实现了两个好处：（1）保持相关代码靠近调用点，（2）使多次调用同一代码块非常清晰。

```cpp
const int num_kv_iter = cdiv(len_kv, BLOCK_KV);

auto load_K = [&](int kv_id) {
  // guard against out-of-bounds global read
  if (kv_id < num_kv_iter) {
    // select the shared buffer destination
    const uint32_t dst = K_smem + (kv_id % 2) * (2 * BLOCK_KV * DIM * sizeof(nv_bfloat16));
    global_to_shared_swizzle<BLOCK_KV, DIM, TB_SIZE>(dst, K, DIM, tid);

    // load_K() will be in charge of incrementing global memory address
    K += BLOCK_KV * DIM;
  }

  // we always commit a cp-async group regardless of whether there is a cp.async
  // to maintain loop invariance.
  asm volatile("cp.async.commit_group;");
};
auto load_V = ...;

// prefetch K and V
load_K(0);
load_V(0);

for (int kv_id = 0; kv_id < num_kv_iter; kv_id++) {
  // prefetch K for the next iteration
  // now we have 3 prefetches in flight: K-V-K
  load_K(kv_id + 1);

  // wait for prefetch of current K to finish and load K shared->registers
  // now we have 2 prefetches in flight: V-K
  asm volatile("cp.async.wait_group 2;");
  __syncthreads();
  ...

  // 1st MMA
  ...

  // prefetch V for the next iteration
  // now we have 3 prefetches in flight: V-K-V
  load_V(kv_id + 1);

  // online softmax
  ...

  // wait for prefetch of current V to finish and load V shared->registers
  // now we have 2 prefetches in flight: K-V
  asm volatile("cp.async.wait_group 2;");
  __syncthreads();
  ...

  // 2nd MMA
  ...
}
```

我实验了一下在循环中放置`load_K/V`和`cp.async.wait_group`的位置，发现上述放置产生了最佳性能。虽然最终这取决于编译器如何重新排列和交错不同指令，但上述放置是有道理的：将`load_V()`放在第一个MMA之后，以便当K数据在寄存器中时，张量核心可以立即开始工作（而不是等待发出V的`cp.async`），即保持张量核心忙碌；`load_V()`放在在线softmax之前，以保持内存引擎忙碌（而CUDA核心正在处理在线softmax）。同样，最佳放置也可能很大程度上取决于硬件，例如内存和计算的相对速度，不同的内存和计算单元是否可以同时工作……

版本3：[attention_v3.cu](https://github.com/gau-nernst/learn-cuda/blob/e83c256/07_attention/attention_v3.cu)。

![Warp state statistics of v3](/images/others/others-fa-5090/015-f8b68c26.png)

内核v3的Warp状态统计。

Stall Long Scoreboard现在从Warp状态统计中消失了。我还必须将`BLOCK_KV`从64减少到32，因为我们现在为K和V使用两个缓冲区，以便共享内存使用总量保持不变。

| 内核               | TFLOPS | % of SOL |
| ------------------ | ------ | -------- |
| v2（共享内存重排） | 181.11 | 86.45%   |
| v3（2阶段流水线）  | 189.84 | 90.62%   |

## 版本4 - 对K和V使用ldmatrix.x4

对于最后两个版本，我无法从性能分析数据中识别出任何优化机会（可能只是技能问题）。想法主要来自阅读随机资料和盯着内核看。

之前，我们对K和V使用`ldmatrix.x2`，因为它自然地适合`n8k16`MMA瓦片。然而，既然我们无论如何都在处理更大的瓦片，我们可以直接使用`ldmatrix.x4`来发出更少的指令。有两个选项：加载`n16k16`瓦片，或`n8k32`瓦片。

![ldmatrix.x4 for B](/images/others/others-fa-5090/016-78fcfc89.svg)

对乘数B使用ldmatrix.x4的可能选项。

一个选项比另一个更好吗？我们可以尝试从算术强度的角度做一些分析。乍一看，`n16k16`看起来是更好的选项：2个`ldmatrix.x4`（1个用于A和1个用于B）来做2个`mma.m16n8k16`；而`n8k32`选项需要3个`ldmatrix.x4`（2个用于A和1个用于B）来做2个`mma.m16n8k16`。如果我们要为矩阵乘法内核实现这个想法，这个分析是有意义的。然而，在我们的情况下，乘数A（查询）已经在寄存器中，因此我们只需要考虑乘数B（键和值）的加载成本。这个认识表明两个选项应该是相同的。

你绝对可以选择不同的模式来加载K和V，但我希望至少这里提供的两个选项更有条理。要实现这个想法，关键是选择正确的8x8`ldmatrix`瓦片的行地址。

```cpp
{
  // pre-compute ldmatrix address for K, using n8k32 option
  // [8x8][8x8][8x8][8x8]
  const int row_off = lane_id % 8;
  const int col_off = lane_id / 8 * 8;
  K_smem_thread = swizzle<DIM * sizeof(nv_bfloat16)>(K_smem + (row_off * DIM + col_off) * sizeof(nv_bfloat16));
}

for (int kv_id = 0; kv_id < num_kv_iter; kv_id++) {
  ...

  // K shared->registers
  // notice mma_id_d is incremented by 2
  for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_N; mma_id_kv++)
    for (int mma_id_d = 0; mma_id_d < DIM / MMA_K; mma_id_d += 2) {
      uint32_t addr = K_smem_thread + (kv_id % 2) * (2 * BLOCK_KV * DIM * sizeof(nv_bfloat16));
      addr += mma_id_kv * MMA_N * DIM * sizeof(nv_bfloat16);  // row
      addr ^= mma_id_d * MMA_K * sizeof(nv_bfloat16);  // col
      ldmatrix_x4(K_rmem[mma_id_kv][mma_id_d], addr);
    }

  ...
}
```

版本4：[attention_v4.cu](https://github.com/gau-nernst/learn-cuda/blob/e83c256/07_attention/attention_v4.cu)。

| 内核                        | TFLOPS | % of SOL |
| --------------------------- | ------ | -------- |
| v3（2阶段流水线）           | 189.84 | 90.62%   |
| v4（`ldmatrix.x4`用于K和V） | 194.33 | 92.76%   |

我对速度提升感到相当惊讶。这个版本唯一的区别是我们在主循环中使用了2倍更少的`ldmatrix`指令。然而，我们获得了显著的改进，接近SOL。我猜由于新GPU中张量核心和内存引擎非常快，调度和发出指令可能成为瓶颈！

## 版本5 - 更好的流水线

在版本3中，我们对K和V都使用双缓冲区。然而，这是冗余的：在执行第一个MMA时，我们可以预取当前迭代的V；在执行第二个MMA时，我们可以预取下一个迭代的K。换句话说，我们只需要为K使用双缓冲区。

```cpp
// prefetch K
load_K(0);

for (int kv_id = 0; kv_id < num_kv_iter; kv_id++) {
  // prefetch V for current iteration
  // now we have 2 prefetches in flight: K-V
  // __syncthreads() here is required to make sure we finish using V_smem
  // from the previous iteration, since there is only 1 shared buffer for V.
  __syncthreads();
  load_V(kv_id);

  // wait for prefetch of current K and load K shared->registers
  // now we have 1 prefetch in flight: V
  ...

  // 1st MMA
  ...

  // prefetch K for the next iteration
  // now we have 2 prefetches in flight: V-K
  load_K(kv_id + 1);

  // online softmax
  ...

  // wait for prefetch of current V and load V shared->registers
  // now we have 1 prefetch in flight: K
  ...

  // 2nd MMA
  ...
}
```

版本5：[attention_v5.cu](https://github.com/gau-nernst/learn-cuda/blob/e83c256/07_attention/attention_v5.cu)。

更有效地使用共享内存意味着我们可以增加一些瓦片大小。我将`BLOCK_KV`从32增加回64。增加`BLOCK_Q`很困难，因为它会使保存累加器的寄存器数量翻倍。改进是适度但明显的。

| 内核                            | TFLOPS | SOL 百分比 |
| ------------------------------- | ------ | ---------- |
| v4（`ldmatrix.x4` 用于 K 和 V） | 194.33 | 92.76%     |
| v5（更好的流水线）              | 197.74 | 94.39%     |

## 下一步是什么？

| 内核（Kernel）                  | TFLOPS | SOL 百分比 |
| ------------------------------- | ------ | ---------- |
| `F.sdpa()`（Flash Attention）   | 186.73 | 89.13%     |
| `F.sdpa()`（CuDNN）             | 203.61 | 97.19%     |
| `flash-attn`                    | 190.58 | 90.97%     |
| v1（基础版）                    | 142.87 | 68.20%     |
| v2（共享内存交换）              | 181.11 | 86.45%     |
| v3（两阶段流水线）              | 189.84 | 90.62%     |
| v4（`ldmatrix.x4` 用于 K 和 V） | 194.33 | 92.76%     |
| v5（更好的流水线）              | 197.74 | 94.39%     |

回顾一下，我们的内核（Kernel）v3 已经超越了官方的 Flash Attention 内核（Kernel），这是一个不错的惊喜。感觉与之前的几代相比，在 5090 上获得良好性能相对容易。然而，我们最好的内核（Kernel）落后于 CuDNN 的，这意味着仍有提升空间。我尝试检查了 CuDNN 注意力内核（Kernel）的性能分析数据，并获得了以下细节

- 内核（Kernel）名称：`cudnn_generated_fort_native_sdpa_sm80_flash_fprop_wmma_f16_knob_3_64x64x128_4x1x1_kernel0_0` -> 我猜这意味着使用 sm80 特性，`BLOCK_Q=BLOCK_KV=64`，`DIM=128`，以及 4 个 warp（与我们的内核（Kernel）v5 相同）。
- 共享内存：40.96 Kb -> 那是`40960 / (64 * 128 * 2) = 2.5` 倍`(BLOCK_KV, DIM)`。缓冲区的分数数量相当奇怪。或者他们的内核（Kernel）更像是`BLOCK_KV=32` 和 5 个缓冲区？我不知道。

无论如何，这里有一些有趣的想法可以在此基础上构建（除了尝试超越 CuDNN）：

1. 实现反向传播（我听说这比前向传播难得多）
1. 量化/低比特注意力，尤其是在 5090 上使用 NVFP4。我相信[SageAttention](https://github.com/thu-ml/SageAttention) 是这方面的开源前沿。
1. 使用 TMA（即`cp.async.bulk`）与 warp 专业化设计。[Pranjal](https://x.com/pranjalssh) 写了一篇[不错的博客文章](https://cudaforfun.substack.com/p/outperforming-cublas-on-h100-a-worklog) 关于 H100 矩阵乘法的这个主题。
1. [PagedAttention](https://arxiv.org/abs/2309.06180)（即 vLLM 和 SGLang），然后构建一个高性能的无依赖服务引擎。

我希望这篇博客文章对许多人有用。祝编写内核（Kernel）愉快！
