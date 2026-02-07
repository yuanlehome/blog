---
title: MTP 理论加速比分析：从公式到工程决策
slug: mtp
date: '2026-02-07'
tags:
  - Speculative Decoding
status: published
cover: ''
lastEditedTime: '2026-02-07T17:55:00.000Z'
updated: '2026-02-07T17:55:00.000Z'
source: notion
notion:
  id: 2d822dca-4210-8062-81fc-e99d289d0603
---

在大模型推理中，MTP（Multi-Token Prediction）是一类典型的 **Speculative Decoding** 技术：

通过一个较小的 Draft Model 预测多个 token，再由 Target Model 一次性验证，从而**用一次昂贵的主模型计算，摊薄多个 token 的生成成本**。

本文将从**第一性原理**出发，系统梳理 MTP 的理论加速模型，并结合真实工程数据，解释：

- MTP 的加速来源是什么
- 为什么「不是 step 越多越快」
- 如何用理论模型指导工程上的 MTP 深度选择

---

## 一、MTP 的核心问题是什么？

在不使用 MTP 的情况下（baseline）：

- 每生成 1 个 token
- 需要完整执行一次 Target Model 推理
- 成本固定为 $T_{target}$

MTP 的目标很明确：

> 能否用一次 Target Model 的验证，确认多个 token？

如果答案是“能”，那么就有可能提升单位时间内的 token 产出量（throughput）。

---

## 二、接受率（accept ratio）：最基本的概率量

对 MTP 的每一个 step（或 head），定义接受率为：

$\text{accept\_ratio}_{k} = \frac{N_{\text{accepted}}}{N_{\text{draft}}}$

它表示 Draft Model 在第 $k$ 步提出的候选 token 中，有多少比例能被 Target Model 验证通过。

例如：Draft 预测 100 次，有 90 次被 Target 接受，接受率即为 **90%**

---

## 三、平均接受长度（avg_accept_len）：MTP 的“收益核心”

MTP 真正带来的收益不是“接受率高”，而是：

> 一次 Target 验证，平均能连续确认多少个 token

定义平均接受长度：

$\text{avg\_accept\_len} = 1 + p_1 + p_1 p_2 + p_1 p_2 p_3 + \dots$

其中：

- $p_k$ 是第 $k$ 步的接受率
- `1` 表示：无论如何，至少能确认 1 个 token

### 直觉解释

- 第 1 项：一定能确认 1 个 token
- 第 2 项：第 1 个 draft token 被接受的概率
- 第 3 项：前两个 draft token 都被连续接受的概率
- 第 4 项：前三个都被接受的概率
- …

这本质是在计算：

> 一次验证能“走多远”的期望长度

### 示例

假设 3 步 MTP 的接受率分别为：

- $p_1 = 0.8$
- $p_2 = 0.74$
- $p_3 = 0.67$

则：

$\text{avg\_accept\_len} = 1 + 0.8 + 0.8 \times 0.74 + 0.8 \times 0.74 \times 0.67\approx 2.79$

这意味着：

**一次 Target 验证，平均能产出约 2.8 个 token**

---

## 四、将“时间成本”引入模型

仅有 token 数还不够，推理优化必须同时考虑时间。

### 4.1 Baseline（无 MTP）

每生成 1 个 token：

- 成本：$T_{target}$

在固定时间 $T_{total}$ 内：

$\text{Throughput}_{\text{baseline}} = \frac{T_{total}}{T_{target}}$

---

### 4.2 MTP 的一次“验证轮次”

一次 MTP 轮次包含：

1. Draft Model 推理
   - 耗时：($T_{draft}$)
1. Target Model 验证
   - 耗时：($T_{target\_verify}$)
   - 通常比普通 decode 更贵（1.0～1.5×）

但这一轮：

- 平均产出：`avg_accept_len` 个 token

因此：

$\text{Throughput}_{\text{mtp}} = \frac{T_{total} \cdot \text{avg\_accept\_len}}{T_{target\_verify} + T_{draft}}$

---

## 五、理论加速比公式

将 MTP 吞吐量与 baseline 吞吐量相除：

$\text{speed\_up} = \frac{\text{Throughput}_{\text{mtp}}}{\text{Throughput}_{\text{baseline}}} = \frac{T_{target}\cdot \text{avg\_accept\_len}}{T_{target\_verify} + T_{draft}}$

这个公式可以拆成两个直观因子：

$\text{speed\_up} = \underbrace{\text{avg\_accept\_len}}_{\text{一次赚几个 token}}\times\underbrace{\frac{T_{target}}{T_{target\_verify} + T_{draft}}}_{\text{这一轮有多贵}}$

**这就是 MTP 的全部博弈关系。**

---

## 六、用真实工程数据逐步验证 MTP 理论（Step 1 / 2 / 3）

为了验证前文推导的理论模型是否能够指导真实系统决策，我们不只考察最终的三步 MTP，而是**分别对 Step 1 / Step 2 / Step 3 的加速效果进行逐级验证**，观察其**边际收益变化**。

### 6.1 实验环境与基础数据

Baseline（无 MTP）配置下：

- Target Model 单步耗时 $T_{\text{target}} = 76 \text{ ms}$

MTP 模式下，各 step 的统计数据如下：

| MTP 步数 | 单步接受率 | Target Verify(ms) | Draft(ms) |
| -------- | ---------- | ----------------- | --------- |
| Step 1   | 80%        | 90                | 8         |
| Step 2   | 74%        | 104               | 14        |
| Step 3   | 67%        | 117               | 22        |

### 6.2 Step 1：单步 MTP 的理论与实测

平均接受长度：

$\text{avg\_accept\_len}_{(1)} = 1 + 0.8 = 1.8$

理论加速比：

$\text{speed\_up}_{(1)} = \frac{76 \times 1.8}{90 + 8} = \frac{136.8}{98}\approx 1.39$

结论：

- 单步 MTP 已显著提升吞吐（≈ **1.39×**）
- 成本增加有限，收益明显
- **Step 1 是“必选项”**

### 6.3 Step 2：两步 MTP 的边际收益分析

平均接受长度：

$\text{avg\_accept\_len}_{(2)} = 1 + 0.8 + 0.8 \times 0.74\approx 2.39$

理论加速比（VS Baseline）：

$\text{speed\_up}_{(2)} = \frac{76 \times 2.39}{104 + 14} = \frac{181.6}{118}\approx 1.55$

边际加速比（VS Step 1）：

$\frac{\text{speed\_up}_{(2)}}{\text{speed\_up}_{(1)}} = \frac{1.55}{1.39}\approx 1.11$

结论：

- Step 2 仍然带来 **正向边际收益**
- 但收益增幅已明显低于 Step 1
- **属于“可选但需评估”的阶段**

### 6.4 Step 3：三步 MTP 的收益拐点

平均接受长度：

$\text{avg\_accept\_len}_{(3)}=1 + 0.8 + 0.8 \times 0.74 + 0.8 \times 0.74 \times 0.67\approx 2.79$

理论加速比（VS Baseline）：

$\text{speed\_up}_{(3)} = \frac{76 \times 2.79}{117 + 22} = \frac{212.0}{139}\approx 1.53$

边际加速比（VS Step 2）：

$\frac{\text{speed\_up}_{(3)}}{\text{speed\_up}_{(2)}} = \frac{1.53}{1.55}\approx 0.99$

结论：

- 第三步虽然提升了平均接受长度
- 但 **Target Verify 与 Draft 成本增长更快**
- 导致整体吞吐 **不增反降**

👉 **Step 3 已越过最优点，是负边际收益**

### 6.5 汇总对比：为什么“不是 step 越多越好”

| MTP 步数 | avg_accept_len | 总耗时(ms) | 理论加速比 | 边际收益 |
| -------- | -------------- | ---------- | ---------- | -------- |
| Baseline | 1.0            | 76         | 1.00       | —        |
| Step 1   | 1.8            | 98         | 1.39       | ⭐⭐⭐   |
| Step 2   | 2.39           | 118        | 1.55       | ⭐       |
| Step 3   | 2.79           | 139        | 1.53       | ❌       |

### 6.6 工程结论：MTP 是一个“有最优深度”的策略

通过逐步验证可以清晰看到：

- MTP 的收益来自 **接受长度增长快于成本增长**
- 一旦成本增长速度反超收益增长
- **继续增加 step 只会拉低吞吐并恶化延迟**

因此，在真实系统中：

> MTP 不应作为固定深度策略，而应基于接受率与耗时数据动态选择最优 step 数。

---

## 七、为什么 MTP 不是“step 越多越好”？

在工程实践中，常见现象是：

- 第 1 步 MTP：收益明显
- 第 2 步：收益变小
- 第 3 步：收益趋近 0，甚至为负

原因在于：

1. **接受率是条件概率**
   - 一旦前面偏离，后续接受率会快速下降
1. **Target 验证成本随步数上升**
   - Verify 越来越接近完整 decode
1. **Draft 成本不可忽略**

因此应关注 **边际收益**：

> 新增一步 MTP 带来的 token 增量是否大于新增的时间成本

一旦不成立，就应停止增加 MTP 深度。

---

## 总结

> MTP 的加速不是由接受率本身决定的，而是由「平均接受长度 × 成本比」共同决定的。

理解这一点，才能：

- 正确选择 MTP 的 step 数
- 判断什么时候该关 MTP
- 将 MTP 从“经验参数”变成“可计算决策”
