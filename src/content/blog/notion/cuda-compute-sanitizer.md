---
title: CUDA 非法内存访问的“隐式报错”机制与 compute-sanitizer 实战
slug: cuda-compute-sanitizer
date: '2026-02-26'
tags:
  - CUDA
status: published
cover: /images/notion/cuda-compute-sanitizer/31322dca-4210-80c8-a8d6-c31c1db62d6f.png
updated: '2026-02-26T09:47:00.000Z'
source: notion
notion:
  id: 31322dca-4210-80e9-aee8-e2f24856e7b3
---

很多 CUDA 内存 bug 的真实形态不是“当场报错 CUDA error 700”，而是：

- kernel 执行完看起来没问题
- 过一会儿在某个无关算子 / `device_synchronize()` 才报错
- 或者更糟：不报错，但结果被污染（silent corruption）

原因是：**GPU 硬件判定非法访存的条件是“页表无映射/权限不匹配”，不是“是否越过对象逻辑边界”**。

---

## 一、三段内存：合法 / Padding / 非法

以常见 allocator（例如 256B 对齐/分桶/arena）为例，一次分配在物理层面可理解为三段：

- **合法内存（绿色）**：cudaMalloc/allocator 返回的、逻辑可读写范围
- **Padding 内存（黄色）**：对齐/内部碎片导致的额外空间，逻辑不可用但通常仍可访问
- **非法内存（红色）**：未分配区域（或已释放后无映射区域），理论上访问应触发非法地址

![](/images/notion/cuda-compute-sanitizer/31322dca-4210-80c8-a8d6-c31c1db62d6f.png)

---

## 二、为什么“越界不一定报 error 700”

CUDA error 700（illegal address）在硬件层面的必要条件是：

> 当前 GPU context 下，该虚拟地址在 GPU 页表中无映射（或权限不匹配）。

因此会出现两个工程上最常见的“反直觉”：

1. 越界不一定报错：越界地址如果仍落在某个“已映射页”里（同一 arena 的其他 chunk、缓存复用出来的别的 tensor），硬件认为地址有效，可能不报 700，但会写坏别人的数据。
1. 访问 Padding 通常不报错：Padding 区往往仍处于已映射页内，读写可能“成功”，但逻辑上就是错的；后果通常是延迟爆炸或结果污染。

结论：**“没报错”只说明“没碰到无映射页”，不代表“没越界”**。

---

## 三、最小复现：越界写可能不当场报错

下面 demo 故意申请 448 Bytes（112 \* 4），然后越界写 `n+8`。不同机器/驱动/allocator 状态下，同步点可能报错，也可能不报（但数据已可能被污染）。

```c++
// oob_demo.cu
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

__global__ void oob_write(int* p, int n, int idx) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    p[0] = 1;
    p[idx] = 2; // OOB
  }
}

static void ck(cudaError_t e, const char* msg) {
  if (e != cudaSuccess) {
    fprintf(stderr, "%s: %s\n", msg, cudaGetErrorString(e));
    std::exit(1);
  }
}

int main() {
  int n = 112;                // 112 * 4 = 448 Bytes
  int* d = nullptr;
  ck(cudaMalloc(&d, n * 4), "cudaMalloc");

  oob_write<<<1, 32>>>(d, n, n + 8);
  // 1<<20(4 MiB) 越界，直接执行 ./oob_demo 就会报错
  // oob_write<<<1, 32>>>(d, n, n + 1<<20);
  ck(cudaGetLastError(), "launch");

  // 可能这里才报错，也可能不报错（不代表正确）
  auto e = cudaDeviceSynchronize();
  if (e != cudaSuccess) fprintf(stderr, "sync: %s\n", cudaGetErrorString(e));
  else fprintf(stderr, "sync ok (may still be corrupt)\n");

  ck(cudaFree(d), "cudaFree");
}
```

编译运行：

```bash
nvcc -O2 -g -lineinfo oob_demo.cu -o oob_demo
./oob_demo
```

正常不报错，输出：

```bash
sync ok (may still be corrupt)
```

---

## 四、解决方案：compute-sanitizer 做确定性定位

执行：

```bash
compute-sanitizer --tool memcheck --show-backtrace yes ./oob_demo
```

输出检测报告：

```bash
========= COMPUTE-SANITIZER
========= Invalid __global__ write of size 4 bytes
=========     at oob_write(int *, int, int)+0xd0 in oob_demo.cu:9
=========     by thread (0,0,0) in block (0,0,0)
=========     Access at 0x7f241b0001e0 is out of bounds
=========     and is 33 bytes after the nearest allocation at 0x7f241b000000 of size 448 bytes
=========     Saved host backtrace up to driver entry point at kernel launch time
=========         Host Frame: main in oob_demo.cu:25 [0x8a81] in oob_demo
=========
========= Program hit cudaErrorLaunchFailure (error 719) due to "unspecified launch failure" on CUDA API call to cudaDeviceSynchronize.
=========     Saved host backtrace up to driver entry point at error
=========         Host Frame: main in oob_demo.cu:29 [0x8a03] in oob_demo
=========
sync: unspecified launch failure
========= Program hit cudaErrorLaunchFailure (error 719) due to "unspecified launch failure" on CUDA API call to cudaFree.
=========     Saved host backtrace up to driver entry point at error
=========         Host Frame: main in oob_demo.cu:33 [0x8a2d] in oob_demo
=========
cudaFree: unspecified launch failure
========= Target application returned an error
========= ERROR SUMMARY: 3 errors
```

三条最有效的建议：

- 在怀疑点附近插同步（如 `paddle.device.synchronize()`）：把爆点往前推，减少误导
- 使用 `compute-sanitizer`：sanitizer 很慢，但输出确定
- 编译带 `g -lineinfo`：才能将信息落到源码具体行

---

## 五、compute-sanitizer 常见用法

### 5.1 memcheck：越界 / 非法地址 / 未对齐 / 泄漏

适用： `CUDA 700/719`、silent corruption、随机炸、越界读写、misaligned load/store、device malloc/free 错误、CUDA API 错误等。

```bash
compute-sanitizer --tool memcheck --show-backtrace yes ./your_bin
```

常用增强选项：

- `--show-backtrace yes`：打印主机回溯。
- `--force-blocking-launches yes`：强制 kernel 同步执行（更好复现/更早报）。代价：更慢；并且可能只报“第一个触发错误的线程”。
- `--leak-check full`：检查 cudaMalloc 泄漏（memcheck 支持 leak checking）。

### 5.2 针对 cudaMallocAsync 的坑：stream-ordered 生命周期竞态

如果你用 `cudaMallocAsync/cudaFreeAsync` 或框架的 async 分配，建议打开：

```bash
compute-sanitizer --tool memcheck --track-stream-ordered-races all ./your_bin
```

用于抓 use-before-alloc / use-after-free 这类“生命周期竞态”。

### 5.3 racecheck：共享内存（shared memory）数据竞争/危险

适用：shared memory 上的 RAW/WAR/WAW 等 hazard，典型是“结果偶发不稳定”。

```bash
compute-sanitizer --tool racecheck --show-backtrace yes ./your_bin
```

> 注意：racecheck 主要针对 shared memory（以及相关 hazard），不是全局内存的通用 data race 检测。

### 5.4 initcheck：读取未初始化的 device global memory

适用：device global memory 未写就读、路径覆盖不全导致的“随机值污染”。

```bash
compute-sanitizer --tool initcheck --show-backtrace yes ./your_bin
```

### 5.5 synccheck：同步原语使用错误（\_\_syncthreads 等）

**适用：** 分支导致部分线程没到同步点、错误的 barrier 用法、warp-level / block-level 同步误用等。

```bash
compute-sanitizer --tool synccheck --show-backtrace yes ./your_bin
```

---

### 5.6 控制输出与“只检查一部分 launch”（大工程必备）

当你的程序 kernel 很多，常用做法是“只看前 N 次 launch / 跳过前 N 次”。compute-sanitizer 支持按 launch 计数做筛选。

```bash
# 只检查第 N 次之后的一段（跳过前 100 次 launch）
compute-sanitizer --tool memcheck --launch-skip 100 --launch-count 50 ./your_bin
```

---

## 总结：实战推荐的最短排障流程

1. 先 memcheck：抓 OOB / misaligned / illegal address。
1. 如果怀疑是 shared memory hazard → racecheck。
1. 如果怀疑是未初始化读 → initcheck。
1. 如果怀疑是同步使用错误 → synccheck。
1. 大工程：加 `-force-blocking-launches yes` 或用 `-launch-skip/--launch-count` 收敛范围。
