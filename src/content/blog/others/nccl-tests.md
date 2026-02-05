---
title: 'nccl-tests：NCCL 排障与性能定位的“复现机”与标尺'
slug: nccl-tests
status: published
date: '2026-02-05'
tags:
  - NCCL
---

`nccl-tests` 是 NVIDIA 官方维护的一组 NCCL 基准 / 正确性测试程序。它最有价值的地方不在“跑个分”，而在于：把训练框架里一堆复杂因素剥离掉，只剩下 **NCCL 本身 + 你的硬件与网络**，从而让问题变得 **可复现、可对比、可定位**。

我基本把它当成三件事的工具：

- **验证链路有没有走对**：节点内 NVLink/NVSwitch，节点间 IB/RoCE，是否意外回退到 Socket。
- **画出性能 S 曲线**：看带宽是否随着消息变大逐步爬升并贴近“线速”，以及哪里出现拐点/台阶。
- **把偶发 hang/timeout/error 抓现行**：固定 size 无限循环跑，用最小负载重现，再配合 NCCL 日志找卡点。

一句话：训练里出问题，**先用 nccl-tests 把它变成一个谁都能一键复现的命令**，然后你才有资格去谈 `NCCL_PROTO` / `NCCL_ALGO` / 网络调参。

---

## 1. 你会用到哪些二进制（按排障价值排序）

编译出来后通常在 `./build/` 下能看到这些：

- `all_reduce_perf`：排障第一选择（DDP / ZeRO / 绝大多数训练都会碰到 AllReduce）
- `reduce_scatter_perf`、`all_gather_perf`：ZeRO / FSDP / TP 相关通信主力
- `alltoall_perf`：MoE / 专家并行、shuffle 类型负载的“痛点算子”
- `sendrecv_perf`：点对点验证（“IB 真通吗？网卡选对了吗？”特别好用）

---

## 2. 构建：单机 vs 多机（MPI 版本别忘了）

### 2.1 单机最简

```bash
git clone https://github.com/NVIDIA/nccl-tests.git
cd nccl-tests
make -j
```

### 2.2 多机：强烈建议编译 MPI 版本

多节点通常用 `mpirun/srun` 起多进程，建议直接编 MPI 版本二进制：

```bash
make -j MPI=1 MPI_HOME=/path/to/mpi CUDA_HOME=/path/to/cuda NCCL_HOME=/path/to/nccl
```

**常见坑：**

- 多机起得来但行为怪 / 变量没透传 / 日志缺失，回头看发现没编 MPI 版本。
- `mpirun` 没有用 `-x` 透传环境变量（尤其 `LD_LIBRARY_PATH`、`NCCL_*`）。

---

## 3. 运行参数：我真正会频繁改的只有这些

如果你只想记最少的东西：**message sweep + 稳定性 + rank 形态**。

### 3.1 message size sweep（决定你能不能看见“拐点”）

- `-b`：最小 size（如 `8` / `64K` / `1M`）
- `-e`：最大 size（如 `1G` / `8G`）
- `-f`：步进倍率（建议 `2`，扫出标准 S 曲线）

例：8B → 1GiB 翻倍扫：

```bash
./build/all_reduce_perf -b 8 -e 1G -f 2 -g 8
```

### 3.2 rank 形态（尽量贴近你的训练）

- `-g`：每进程使用的 GPU 数
  - 单机单进程多卡：`-g 8`
  - 多机：通常 `-g 1`，每 GPU 一个进程（最贴近 DDP ranks）

### 3.3 稳定性（让结果“能比”，也能复现偶发）

- `-w`：warmup 次数（建议 `>= 5`）
- `-n`：计时迭代次数（建议 `>= 50`）
- `-N`：run cycles，`0` 表示无限循环（抓偶发 hang/error 很好用）

---

## 4. 输出怎么看：`time / algbw / busbw`（三列决定 90% 判断）

`nccl-tests` 每个 size 会打印 `time`、`algbw`、`busbw`。其中最容易被误读的是 `busbw`，下面按官方语义来理解它的用途。

### 4.1 Algorithm BW（algbw）：算子有效吞吐

你可以把它理解为“该 collective 的有效 payload 吞吐”。同操作、同 dtype、同 size、同 ranks 下做 A/B 对比非常直观。

### 4.2 Bus BW（busbw）：对照总线/链路利用率的归一化指标

`busbw` 是从 `algbw` 计算出来的，目的在于得到一个“相对不依赖 ranks 数”的指标，方便你对照硬件能力（总线/链路上限）。

以 AllReduce 为例，常见换算形式是：

```text
busbw = algbw * 2*(n-1)/n
```

其中 `n` 为 ranks 数。

**很实用的直觉：**

- **看 S 曲线**：小消息受延迟主导，带宽爬不起来；消息变大后逐步爬升并在“线速附近”平台化，这就是干净的 S 曲线。
- **出现台阶/拐点/反向掉速**：通常意味着 NCCL 在某些 size 段切了算法 / 协议 / 路径（后面用对照实验逼出来）。

### 4.3 “busbw 怎么可能超过网卡线速？”

这是经典误会：多节点时 NCCL 往往是分层/树形算法，节点内 NVLink/NVSwitch 承担了大量数据搬运，网络未必是唯一瓶颈，所以换算出来的 `busbw` 可能“看起来超过网卡”。这不等价于“真实网卡吞吐超过线速”。

---

## 5. 三条“黄金命令模板”：单机 / 两机 / N 机（含日志落盘）

下面三条命令建议直接收藏。目标不是“调到最好”，而是：**先拿到可信 baseline + 关键证据**。

### 5.1 单机 8 卡：先确认 NVLink/NVSwitch/PCIe 路径

```bash
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,GRAPH,NET
export NCCL_DEBUG_FILE=/tmp/nccl.%h.%p.log   # 每进程一个文件

./build/all_reduce_perf -b 8 -e 1G -f 2 -g 8 -w 5 -n 50
```

### 5.2 两机（每节点 8 卡）：验证 IB/RoCE + GDR 形态

每 GPU 一个进程（最贴近训练 ranks），MPI 透传 NCCL 变量：

```bash
mpirun -np 16 -N 8 \
  -x NCCL_DEBUG=INFO \
  -x NCCL_DEBUG_SUBSYS=INIT,NET,GRAPH \
  -x NCCL_DEBUG_FILE=/tmp/nccl.%h.%p.log \
  ./build/all_reduce_perf -b 8 -e 8G -f 2 -g 1 -w 5 -n 50
```

重点看两件事：

- `NET` 日志里到底走的是 IB plugin 还是 socket fallback；
- `busbw` 的平台值能不能接近你网卡应有水平，以及曲线是否平滑。

### 5.3 N 机扩容：看是否“随规模退化/某节点拖后腿”

```bash
mpirun -np 64 -N 8 \
  -x NCCL_DEBUG=INFO \
  -x NCCL_DEBUG_SUBSYS=INIT,NET,GRAPH \
  -x NCCL_DEBUG_FILE=/tmp/nccl.%h.%p.log \
  ./build/all_reduce_perf -b 64K -e 8G -f 2 -g 1 -w 5 -n 50
```

我的习惯是这样拆分观察：

- **小消息段（64B → 4M）**：主要看 latency / 协议选择；
- **大消息段（4M → 8G）**：主要看带宽上限 / 抖动。

---

## 6. 抓偶发 hang / error 的标准姿势：固定 size + 无限循环

训练里最烦的是：跑俩小时才挂。`nccl-tests` 的办法很朴素：

- 选一个你怀疑能触发的 size
- 固定住
- 无限跑
- 配合每 rank 日志落盘

```bash
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET
export NCCL_DEBUG_FILE=/tmp/nccl.%h.%p.log

# 固定 64MB，持续压测
./build/all_reduce_perf -b 64M -e 64M -g 8 -w 5 -n 50 -N 0
```

**经验技巧：**

- 如果是跨机偶发，先从“两机”开始固定 size 无限跑，别一上来 8/16 节点；
- `-n` 不必太大，关键是 `-N 0` 跑够久；
- 一旦挂住，`/tmp/nccl.<host>.<pid>.log` 里通常能看到“卡在哪个阶段 / 哪个 rank”。

---

## 7. 对照实验：一次只改一个变量（别一把梭把 NCCL 调死）

`nccl-tests` 最厉害的一点就是：你可以做 **控制变量实验**。我推荐的策略是：

> **先排除（最小侵入），再强制（只在调试时用强制策略）**

### 7.1 四个最常用“排除开关”

```bash
# 1) 排除 LL128（常用排障）
export NCCL_PROTO=^LL128

# 2) 禁用 IB（验证是不是 IB/RoCE 链路问题）
export NCCL_IB_DISABLE=1

# 3) 限定网络接口（验证是否选错网卡）
export NCCL_SOCKET_IFNAME=eth0

# 4) 禁用 GPU P2P（隔离 NVLink/PCIe P2P）
export NCCL_P2P_DISABLE=1
```

然后复跑同一条 `nccl-tests` 命令，对比三件事：

- 问题是否消失（hang/error/抖动/掉速）
- 拐点是否移动（说明确实是选择逻辑相关）
- 日志中 `NET/GRAPH` 选择是否变化

> 小提醒：`NCCL_DEBUG_SUBSYS` 支持用 `^` 排除某子系统输出（你只要 NET+GRAPH 时，能把其他都关掉）。

---

## 8. “现象 → 根因假设 → 验证命令”速查（IB + NVLink 专项）

| 现象（训练里看到的）               | 高概率根因（从高到低）                           | nccl-tests 怎么做（建议命令）                                              | 你该看什么证据                                            |
| ---------------------------------- | ------------------------------------------------ | -------------------------------------------------------------------------- | --------------------------------------------------------- |
| 跨机带宽低得离谱（`busbw` 上不去） | 走错接口 / IB 没用上 / 回退 socket / GDR 没生效  | 两机 `-g 1` 跑 `all_reduce_perf`；对照 `NCCL_IB_DISABLE=1` 再跑一遍        | `NET` 日志里是什么 transport；禁 IB 后是否反而更稳        |
| 单机 8 卡 `busbw` 只有 PCIe 水平   | NVLink/NVSwitch 未被识别（驱动/容器/拓扑暴露）   | 单机 `-g 8` sweep；打开 `GRAPH`                                            | `GRAPH` 里拓扑/通道是否合理；曲线是否达到机器上限         |
| 某个 size 段出现“台阶/掉速/抖动”   | algo/proto/transport 在该 size 段切换            | 围绕拐点做小范围 sweep（如 `-b 1M -e 64M -f 2`）；对照 `NCCL_PROTO=^LL128` | 拐点是否移动/消失；日志里是否切了 algo/proto              |
| 偶发 hang/timeout                  | IB/RoCE 丢包/重传、某节点异常、或路径 bug        | 固定 size `-N 0` 无限跑并落盘日志；对照禁 IB                               | “最后一条日志”在哪个阶段/哪个 rank；禁 IB 后是否不再 hang |
| `busbw` 看起来超过网卡线速         | 分层/树算法导致节点内带宽贡献大，换算值≠网卡吞吐 | 对照跑 `reduce_scatter` / `all_gather`；看日志 algo                        | 是否使用 tree/nvlstree 等；注意不要用 `busbw` 当网卡吞吐  |

---

## 9. 排障实验矩阵模板（建议贴到 Issue 里）

目标：把“我感觉是 XX 导致”变成“我有 A/B 证据证明是 XX 导致”。

| 维度           | 取值                     | 目的                 | 预期/判据                                       |
| -------------- | ------------------------ | -------------------- | ----------------------------------------------- |
| baseline       | 无                       | 建立 S 曲线 & 拐点   | 曲线平滑、平台值接近线速；异常段明确            |
| transport      | `NCCL_IB_DISABLE=1`      | 判断是否 IB 专属问题 | 禁 IB 后 hang 消失/更稳 ⇒ 指向 IB/RoCE/网卡配置 |
| iface          | `NCCL_SOCKET_IFNAME=...` | 验证是否选错网卡     | 限定接口后带宽恢复/初始化更快                   |
| p2p            | `NCCL_P2P_DISABLE=1`     | 隔离 NVLink/P2P 路径 | 禁 P2P 后变稳 ⇒ 指向 NVLink/PCIe P2P            |
| proto          | `NCCL_PROTO=^LL128`      | 排除 LL128 路径      | 异常消失/拐点移动 ⇒ 指向协议选择                |
| algo（仅调试） | `NCCL_ALGO=Ring`         | 排除算法切换因素     | 拐点消失 ⇒ 指向算法/拓扑匹配                    |

---

## 10. 推荐的“结果产物”：S 曲线 + 日志落盘 + 一句话结论

跑完 nccl-tests，最好留下三个东西（后续写排障报告会非常省事）：

1. **S 曲线数据**：至少贴出拐点附近几行（size / algbw / busbw）
2. **每 rank 日志文件**：`NCCL_DEBUG_FILE=/tmp/nccl.%h.%p.log`
3. **一句话结论**（强可判定、可复现）：
   - “禁 IB 后 hang 消失 ⇒ 怀疑 IB/RoCE 链路不稳”
   - “排除 LL128 后台阶消失 ⇒ 怀疑协议切换相关”
   - “单机 NVLink 没跑满 ⇒ 优先查拓扑暴露/驱动/容器”

---

## 11. 常用跑法组合（拷贝即用）

### 11.1 只想确认“变量是否被 NCCL 采纳”（看 INIT）

```bash
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT
./build/all_reduce_perf -b 1M -e 1M -g 8 -w 1 -n 5
```

### 11.2 专门抓跨机网络证据（NET 为主）

```bash
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET
export NCCL_DEBUG_FILE=/tmp/nccl.%h.%p.log

mpirun -np 16 -N 8 \
  -x NCCL_DEBUG -x NCCL_DEBUG_SUBSYS -x NCCL_DEBUG_FILE \
  ./build/all_reduce_perf -b 8M -e 8M -g 1 -w 3 -n 20 -N 0
```

### 11.3 专门看拓扑与通道（GRAPH 为主）

```bash
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,GRAPH
./build/all_reduce_perf -b 64K -e 1G -f 2 -g 8 -w 5 -n 50
```

---

## 12. 把“拐点 size”映射回 NCCL 日志：三步读法（效率最高）

你不需要把日志当小说逐行看。按下面流程来：

### Step A：先确认环境变量真的生效（INIT）

每个 rank 日志开头先看 `INIT`，确认你设的禁用项/限定项真的被采纳。  
如果 INIT 里看不出来你设的变量，先别分析性能：先把 **变量透传 / 容器环境 / 启动方式** 搞对。

### Step B：确认路径是不是你以为的路径（NET + GRAPH）

- `NET`：跨节点到底用什么 transport（IB plugin？socket fallback？选错网卡？）
- `GRAPH`：节点内拓扑/通道怎么铺（NVLink/NVSwitch 是否被识别？通道数是否合理？）

### Step C：用对照实验把切换原因逼出来（一次只改一个）

围绕拐点做小范围 sweep，然后逐个对照：

- baseline（不强制）
- `NCCL_PROTO=^LL128`（排除协议路径）
- `NCCL_IB_DISABLE=1`（排除 IB，逼它走 socket）
- `NCCL_P2P_DISABLE=1`（排除 NVLink/P2P）

看拐点是否移动/消失，并对照 NET/GRAPH 输出差异。

---

## 13. 一键脚本：跑曲线、落盘日志、生成 RESULTS.md（可直接用）

下面脚本做了这些事：

- 自动创建目录：`runs/YYYYmmdd_HHMMSS/`
- 同时跑：单机（节点内）+ 多机（跨节点）
- 每次跑都落盘：
  - `raw/*.out`（原始 stdout）
  - `logs/nccl.%h.%p.log`（每 rank NCCL 日志）
- 粗解析输出生成 `RESULTS.md`（够用，适合贴 Issue）

> 默认使用 `mpirun`，也提供 `srun` 分支入口（Slurm 用户可用）。

````bash
#!/usr/bin/env bash
set -euo pipefail

# ========== 用户需要按集群情况改的地方 ==========
NCCL_TESTS_BIN_DIR="${NCCL_TESTS_BIN_DIR:-./build}"
TEST_BIN="${TEST_BIN:-all_reduce_perf}"          # 也可以换成 reduce_scatter_perf/all_gather_perf/alltoall_perf
RUN_TAG="${RUN_TAG:-nccl-tests}"

# 单机（节点内）形态：一个进程吃掉本机所有 GPU
LOCAL_GPUS="${LOCAL_GPUS:-8}"

# 多机（跨节点）形态：每 GPU 一个进程（更贴近 DDP ranks）
NODES="${NODES:-2}"
RANKS_PER_NODE="${RANKS_PER_NODE:-8}"            # 通常=每节点 GPU 数
TOTAL_RANKS=$((NODES * RANKS_PER_NODE))

# 网络接口（IB/RoCE 常见是 ethX / ensX / bondX；仅用于 Socket 选择/对照实验）
IFACE="${IFACE:-eth0}"

# Message sweep：大范围 & 拐点放大镜两套
SWEEP_MIN="${SWEEP_MIN:-8}"
SWEEP_MAX="${SWEEP_MAX:-8G}"
SWEEP_FACTOR="${SWEEP_FACTOR:-2}"

ZOOM_MIN="${ZOOM_MIN:-1M}"
ZOOM_MAX="${ZOOM_MAX:-64M}"
ZOOM_FACTOR="${ZOOM_FACTOR:-2}"

WARMUP="${WARMUP:-5}"
ITERS="${ITERS:-50}"

# 无限循环压测（抓偶发 hang/error），0 = infinite
BURN_SIZE="${BURN_SIZE:-64M}"
BURN_CYCLES="${BURN_CYCLES:-0}"                  # 0=无限循环
BURN_ITERS="${BURN_ITERS:-20}"
BURN_WARMUP="${BURN_WARMUP:-3}"

# MPI 启动器：mpirun / srun 二选一（默认 mpirun）
LAUNCHER="${LAUNCHER:-mpirun}"

# ========== NCCL 日志默认配置（建议先用 INFO） ==========
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS:-INIT,NET,GRAPH}"

# 每 rank 独立落盘（官方支持 %h/%p）
# 注意：这里设置成相对路径，后面会在每次 run 的目录下执行
export NCCL_DEBUG_FILE="${NCCL_DEBUG_FILE:-logs/nccl.%h.%p.log}"

# ========== Run 目录 ==========
TS="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="runs/${TS}_${RUN_TAG}"
mkdir -p "${OUT_DIR}/raw" "${OUT_DIR}/logs" "${OUT_DIR}/meta"

# 记录元信息
{
  echo "timestamp=${TS}"
  echo "bin_dir=${NCCL_TESTS_BIN_DIR}"
  echo "test_bin=${TEST_BIN}"
  echo "local_gpus=${LOCAL_GPUS}"
  echo "nodes=${NODES}"
  echo "ranks_per_node=${RANKS_PER_NODE}"
  echo "total_ranks=${TOTAL_RANKS}"
  echo "iface=${IFACE}"
  echo "sweep=${SWEEP_MIN}..${SWEEP_MAX} factor=${SWEEP_FACTOR}"
  echo "zoom=${ZOOM_MIN}..${ZOOM_MAX} factor=${ZOOM_FACTOR}"
  echo "burn_size=${BURN_SIZE} cycles=${BURN_CYCLES}"
  echo "NCCL_DEBUG=${NCCL_DEBUG}"
  echo "NCCL_DEBUG_SUBSYS=${NCCL_DEBUG_SUBSYS}"
  echo "NCCL_DEBUG_FILE=${NCCL_DEBUG_FILE}"
} | tee "${OUT_DIR}/meta/config.txt" >/dev/null

# 进入 run 目录执行，确保日志都落在当前 run 目录下
pushd "${OUT_DIR}" >/dev/null

BIN="${NCCL_TESTS_BIN_DIR}/${TEST_BIN}"
if [[ ! -x "${BIN}" ]]; then
  echo "ERROR: Cannot find executable: ${BIN}"
  echo "Hint: set NCCL_TESTS_BIN_DIR or TEST_BIN, and ensure you built nccl-tests."
  exit 1
fi

# ========== Helper: 运行并落盘 ==========
run_cmd () {
  local name="$1"; shift
  echo "==> Running: ${name}"
  echo "CMD: $*" | tee "raw/${name}.cmd.txt"
  ( "$@" ) 2>&1 | tee "raw/${name}.out"
  echo "==> Done: ${name}"
  echo
}

# ========== Case 1: 单机（节点内）大范围 sweep ==========
run_cmd "local_sweep" \
  "${BIN}" -b "${SWEEP_MIN}" -e "${SWEEP_MAX}" -f "${SWEEP_FACTOR}" \
  -g "${LOCAL_GPUS}" -w "${WARMUP}" -n "${ITERS}"

# ========== Case 2: 单机（节点内）拐点放大镜 ==========
run_cmd "local_zoom" \
  "${BIN}" -b "${ZOOM_MIN}" -e "${ZOOM_MAX}" -f "${ZOOM_FACTOR}" \
  -g "${LOCAL_GPUS}" -w "${WARMUP}" -n "${ITERS}"

# ========== 多机 launcher ==========
mpi_launch () {
  if [[ "${LAUNCHER}" == "mpirun" ]]; then
    mpirun -np "${TOTAL_RANKS}" -N "${RANKS_PER_NODE}" \
      -x NCCL_DEBUG -x NCCL_DEBUG_SUBSYS -x NCCL_DEBUG_FILE \
      -x NCCL_SOCKET_IFNAME \
      "$@"
  elif [[ "${LAUNCHER}" == "srun" ]]; then
    srun -n "${TOTAL_RANKS}" --ntasks-per-node="${RANKS_PER_NODE}" \
      --export=ALL,NCCL_DEBUG,NCCL_DEBUG_SUBSYS,NCCL_DEBUG_FILE,NCCL_SOCKET_IFNAME \
      "$@"
  else
    echo "ERROR: Unsupported LAUNCHER=${LAUNCHER} (use mpirun or srun)"
    exit 1
  fi
}

# ========== Case 3: 多机 baseline sweep ==========
export NCCL_SOCKET_IFNAME="${IFACE}"
run_cmd "multi_sweep_baseline" \
  mpi_launch "${BIN}" -b "${SWEEP_MIN}" -e "${SWEEP_MAX}" -f "${SWEEP_FACTOR}" \
  -g 1 -w "${WARMUP}" -n "${ITERS}"

# ========== Case 4: 多机 zoom（拐点放大镜） ==========
run_cmd "multi_zoom_baseline" \
  mpi_launch "${BIN}" -b "${ZOOM_MIN}" -e "${ZOOM_MAX}" -f "${ZOOM_FACTOR}" \
  -g 1 -w "${WARMUP}" -n "${ITERS}"

# ========== Case 5: 对照实验（排除 IB / 排除 LL128 / 排除 P2P） ==========
export NCCL_IB_DISABLE=1
run_cmd "multi_zoom_no_ib" \
  mpi_launch "${BIN}" -b "${ZOOM_MIN}" -e "${ZOOM_MAX}" -f "${ZOOM_FACTOR}" \
  -g 1 -w "${WARMUP}" -n "${ITERS}"
unset NCCL_IB_DISABLE

export NCCL_PROTO="^LL128"
run_cmd "multi_zoom_no_ll128" \
  mpi_launch "${BIN}" -b "${ZOOM_MIN}" -e "${ZOOM_MAX}" -f "${ZOOM_FACTOR}" \
  -g 1 -w "${WARMUP}" -n "${ITERS}"
unset NCCL_PROTO

export NCCL_P2P_DISABLE=1
run_cmd "local_zoom_no_p2p" \
  "${BIN}" -b "${ZOOM_MIN}" -e "${ZOOM_MAX}" -f "${ZOOM_FACTOR}" \
  -g "${LOCAL_GPUS}" -w "${WARMUP}" -n "${ITERS}"
unset NCCL_P2P_DISABLE

# ========== Case 6: 固定 size 无限循环（抓偶发） ==========
run_cmd "burn_in_${BURN_SIZE}" \
  "${BIN}" -b "${BURN_SIZE}" -e "${BURN_SIZE}" -g "${LOCAL_GPUS}" \
  -w "${BURN_WARMUP}" -n "${BURN_ITERS}" -N "${BURN_CYCLES}"

# ========== 生成 RESULTS.md（粗解析，够用） ==========
cat > RESULTS.md <<'MD'
# nccl-tests Results Summary

> 本文件由 `run_nccl_tests.sh` 自动生成。
> 建议把 `RESULTS.md` + 拐点附近的 `raw/*.out` + 任意一个 rank 的日志尾部一起贴到 Issue。

## How to read
- `Max busbw`：该 case 下观测到的最大 bus bandwidth（计算值，用于对照平台上限/曲线形态）
- `At size`：达到最大 busbw 的消息 size（常用于找“平台化区间”）
- 若你在某个 size 段看到明显台阶/抖动：优先对照 no_ib / no_ll128 / no_p2p 看拐点是否移动

## Cases
MD

summarize_case () {
  local out="$1"
  local title="$2"
  if [[ ! -f "$out" ]]; then return; fi

  local res
  res=$(awk '
    BEGIN{max=0;size="";line=""}
    /^[[:space:]]*[0-9]+/ {
      bw=$(NF)
      if (bw+0 > max) {max=bw+0; size=$1; line=$0}
    }
    END{
      if (max>0) printf("%s\t%s\t%s\n", max, size, line);
    }
  ' "$out" || true)

  if [[ -n "$res" ]]; then
    local max_bw at_size raw_line
    max_bw=$(echo "$res" | cut -f1)
    at_size=$(echo "$res" | cut -f2)
    raw_line=$(echo "$res" | cut -f3-)
    {
      echo "### ${title}"
      echo ""
      echo "- Max busbw: \`${max_bw}\`"
      echo "- At size: \`${at_size}\` (bytes)"
      echo "- Raw peak row:"
      echo ""
      echo '```'
      echo "$raw_line"
      echo '```'
      echo ""
    } >> RESULTS.md
  else
    {
      echo "### ${title}"
      echo ""
      echo "- (No parsable bandwidth rows found in \`${out}\`. Please open the raw output.)"
      echo ""
    } >> RESULTS.md
  fi
}

summarize_case "raw/local_sweep.out" "local_sweep"
summarize_case "raw/local_zoom.out" "local_zoom"
summarize_case "raw/multi_sweep_baseline.out" "multi_sweep_baseline"
summarize_case "raw/multi_zoom_baseline.out" "multi_zoom_baseline"
summarize_case "raw/multi_zoom_no_ib.out" "multi_zoom_no_ib"
summarize_case "raw/multi_zoom_no_ll128.out" "multi_zoom_no_ll128"
summarize_case "raw/local_zoom_no_p2p.out" "local_zoom_no_p2p"
summarize_case "raw/burn_in_${BURN_SIZE}.out" "burn_in_${BURN_SIZE}"

{
  echo "## Artifacts"
  echo ""
  echo "- Raw outputs: \`raw/*.out\`"
  echo "- NCCL logs per rank: \`logs/nccl.<host>.<pid>.log\`"
  echo "- Run config: \`meta/config.txt\`"
  echo ""
} >> RESULTS.md

popd >/dev/null
echo "Done. Results are in: ${OUT_DIR}/RESULTS.md"
````

---
