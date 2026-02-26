---
title: CUDA 性能知识点：#pragma unroll 可能让你更慢
slug: cuda-pragma-unroll
date: '2026-02-26'
tags: []
status: published
cover: ''
updated: '2026-02-26T13:06:00.000Z'
source: notion
notion:
  id: 31322dca-4210-8083-b16a-d1c3bba72c1b
---

很多 CUDA kernel 不是“算不动”，而是被资源卡死。最常见的罪魁祸首之一就是：循环展开把寄存器用量顶到临界点，然后性能直接塌。

---

## 一、`#pragma unroll` 的隐性成本：regs/thread 上升

循环展开的本质是把 loop body 复制多份。对于 GEMM/FlashAttention/pipeline 类 kernel，loop body 里往往有大量临时变量（fragment、pointer、mask、pipeline state），展开后会导致：

- live range 变长（变量活得更久）
- 同时活跃变量变多
- regs/thread 增加

regs/thread 一旦跨过某个阈值，后果就不是“慢一点”，而是“形态变了”。

---

## 二、寄存器超限的两种典型“塌方方式”

### 2.1 Spill：寄存器溢出到 local memory

寄存器放不下 → 编译器把部分值 spill 到 local memory（物理上走 global/L2）：

- 指令数上升（额外 load/store）
- 访存延迟变大
- 带宽浪费严重

表现：kernel time 大幅上升。

### 2.2 Occupancy 被寄存器卡死

regs/thread 太高 → 同一个 SM 能驻留的 warps/blocks 变少：

- latency hiding 变差
- pipeline 断断续续

表现：SM 吃不满、stall 增多、吞吐下降。

---

## 三、如何快速判断“是不是 unroll 把你坑了”

### 3.1 看 ptxas 报告

编译时加 `Xptxas=-v`：

```bash
nvcc -O3 -lineinfo -Xptxas=-v your_kernel.cu -o your_bin
```

重点看：

- reg（每线程寄存器数）
- spill stores / spill loads：是否发生 spill（以及数量）

对比 unroll 前后：regs 是否明显下降、spill 是否消失/减少。

### 3.2 用 Nsight Compute 验证

只看最小集合指标：

- regs/thread（或相关编译信息）
- achieved occupancy
- local memory / spill 相关指标
- kernel time

如果 regs↓ 且 spill↓ 或 occupancy↑，并且 time↓，结论基本坐实。

---

## 四、实战处理

1. 把 `#pragma unroll N` 降到 1（或直接去掉）。
1. 缩短 live range：用 `{}` 限制作用域、拆分长 basic block、避免大临时对象跨迭代存活。
1. 资源硬约束（谨慎）：`maxrregcount`、`__launch_bounds__`（可能把问题从 occupancy 换成 spill）。

---

## 总结

当 kernel 是 register-bound 时，`#pragma unroll` 很可能是负优化：它把 regs/thread 推过阈值，触发 spill 或 occupancy 崩塌，于是“展开更多”反而更慢。
