---
title: "NCCL\_Debug全栈手段：常用环境变量、日志/拓扑/通信诊断与 Hang/性能/数据异常排查"
slug: nccl-debug-hang
date: '2026-02-05'
tags: []
status: published
source_url: 'https://chatgpt.com/s/dr_69842a5b09b481919b39c496749a5295'
source_author: chatgpt.com
imported_at: '2026-02-05T05:29:57.228Z'
source:
  title: chatgpt.com
  url: 'https://chatgpt.com/s/dr_69842a5b09b481919b39c496749a5295'
---

# NCCL Debug全栈手段：常用环境变量、日志/拓扑/通信诊断与 Hang/性能/数据异常排查

本文面向使用 PyTorch DDP / Megatron / DeepSpeed 或自研分布式训练框架的工程师，系统讲解 NCCL 调试的工具箱和环境变量设置方法，覆盖 **NCCL hang、NCCL error、性能退化、跨机带宽不足、GDR/IB/NVLink 通信异常** 等场景的诊断思路和解决方案。

**目录：**

- A. NCCL Debug 总览：可观测、可控制、可验证的方面

- B. 日志与可观测性环境变量

- C. 拓扑与通信路径诊断

- D. 传输层开关与网络相关环境变量

- E. 算法与协议相关调试手段

- F. 稳定性与容错：Hang/超时/错误处理

- G. 常见故障场景手册（10+案例）

- H. 一页式 NCCL 调优与排障 Cheat Sheet

---

## A. NCCL Debug 总览：可观测、可控制、可验证的方面

NCCL（NVIDIA Collectives Communications Library）提供了丰富的**环境变量**和工具，允许我们从多个层面进行调试：

- \*\*可观测性（Observation）：\*\*通过 NCCL 日志了解内部状态、拓扑检测结果、算法/协议选择、所用网络通道（如 SHM、P2P、Socket、InfiniBand）等信息。例如设置 `NCCL_DEBUG=INFO` 可以打印 NCCL 版本和操作信息，`NCCL_DEBUG_SUBSYS` 允许聚焦特定子系统日志[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=The%20default%20value%20is%20INIT%2CBOOTSTRAP%2CENV)。这些日志有助于找到程序 Hang 的环节或性能瓶颈位置。

- **可控制性（Control）：**NCCL 的众多环境变量可以**强制/禁用**某些行为，从而控制调度决策。例如，可以通过 `NCCL_PROTO` 限制协议（Simple/LL/LL128）选择，通过 `NCCL_ALGO` 限制算法（Ring/Tree/CollNet 等）选择，通过 `NCCL_IB_DISABLE`/`NCCL_SHM_DISABLE` 等开关切换不同传输方式[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=NCCL_IB_DISABLE%EF%83%81)[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=NCCL_P2P_DISABLE%EF%83%81)。这些设置可以帮助我们验证某一机制是否导致了问题——如禁用某模块后问题消失，则该模块可能有关。

- **可验证性（Verification）：**使用**nccl-tests**等基准工具对特定场景进行最小复现和对比实验。例如用 `all_reduce_perf` 测试不同消息大小、不同环境变量组合下的带宽，比较 Algorithm BW（算法带宽）和 Bus BW（总线带宽）[forums.developer.nvidia.com](https://forums.developer.nvidia.com/t/what-is-the-busbw-in-nccl-tests/256858#:~:text=The%20published%20info%20on%20NCCL,p2pBandwidthLatencyTest)来判断硬件通信是否跑满。通过**对照矩阵试验**，我们可以逐步缩小问题范围，并验证修改是否奏效。

总之，NCCL 调试涉及**日志观察**（看现象）、**环境变量调整**（做实验）和**工具对照**（下验证结论）三个环节，形成“**复现→采集信息→缩小变量→定位原因→验证修复**”的闭环流程。在正式进入各部分细节前，建议先收集如下关键信息，作为排障的基础数据：

> **📝 排障信息收集清单：**NCCL 版本、CUDA Driver/Runtime 版本，PyTorch 等框架版本；GPU 型号和拓扑（NVLink/NVSwitch 结构，PCIe 代数），节点间网络类型（InfiniBand/RoCE 还是以太网）、带宽和布线（多 NIC？直连/交换机拓扑？）；当前系统的相关环境变量配置；容器/虚拟化设置（/sys 挂载、`--shm-size`、NUMA 等）；以及**出问题时的具体日志片段、报错信息**。

下面章节将按类别详细介绍 NCCL 的调试手段与参数。

## B. 日志与可观测性环境变量

调试 NCCL 问题的第一步，是启用充分的日志，以**观察 NCCL 内部发生了什么**。NCCL 提供以下环境变量用于控制日志级别和内容：

- **`NCCL_DEBUG` 日志级别：**可取 `WARN`, `INFO`, `TRACE` 等级别[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=NCCL_DEBUG%EF%83%81)。`WARN` 只在发生错误时输出简要信息，`INFO` 会打印调试信息（如各步连接、算法选择），`TRACE` 则会对每次调用输出**可重放的**详细跟踪（大量日志，通常只在小规模测试时使用）。另外，`NCCL_DEBUG=VERSION` 可仅打印 NCCL 版本号用于确认版本[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=The%20,is%20commonly%20used%20for%20debugging)。一般排查从 `INFO` 开始，在问题复杂或需要反馈 NVIDIA 时再用 `TRACE`。注意：过高日志级别可能显著拖慢程序，应在必要时短期使用[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=There%20are%20two%20categories%20of,optimal%20behavior%2C%20crashes%2C%20or%20hangs)。

- \*\*`NCCL_DEBUG_SUBSYS` 日志子系统过滤：\*\*当使用 `INFO`/`TRACE` 级别时，此变量可选定感兴趣的子系统，以减少无关输出[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=NCCL_DEBUG_SUBSYS%EF%83%81)[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=The%20default%20value%20is%20INIT%2CBOOTSTRAP%2CENV)。支持的子系统有 INIT（初始化）、COLL（集合通信算法）、P2P（点对点直连）、SHM（共享内存）、NET（网络传输）、GRAPH（拓扑检测/图搜索）、TUNING（算法/协议调优）、ENV（环境变量设置）、ALLOC（内存分配）、PROXY（Proxy线程）、NVLS（NVLink SHARP）、BOOTSTRAP（进程间引导连接）、REG（注册内存）、PROFILE（粗粒度性能profiling）、RAS（可靠性子系统）等，以及 ALL（全部）。默认的子系统列表是 INIT, BOOTSTRAP, ENV[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=)。例如：
  - `NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=NET,GRAPH` 只看网络连接和拓扑相关日志。

  - 使用前缀 `^` 可排除子模块，如 `NCCL_DEBUG_SUBSYS=ALL,^COLL` 表示记录全部但不含集合算法细节。

- \*\*`NCCL_DEBUG_FILE` 日志重定向：\*\*默认日志输出到 stdout/stderr。设置该变量可将日志写入文件。例如：\
  `NCCL_DEBUG=WARN NCCL_DEBUG_FILE=/tmp/nccl_log.%h.%p`\
  将 WARN 级日志写到文件，文件名中 `%h` 和 `%p` 会分别替换为hostname和进程PID[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=NCCL_DEBUG_FILE%EF%83%81)。这在多进程/多节点场景下很有用，每个进程写自己的日志文件，避免交织。需注意文件名必须唯一，否则多个进程写入同一文件会混乱[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=filename%20can%20also%20be%20set,making%20the%20output%20line%20buffered)。

- **时间戳格式与线程命名：**`NCCL_DEBUG_TIMESTAMP_FORMAT` 可定制日志时间戳格式（例如打印相对时间方便计算耗时）[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=NCCL_DEBUG_TIMESTAMP_FORMAT%EF%83%81)。`NCCL_SET_THREAD_NAME=1` 则让 NCCL 后台线程有易读名称（如 `NCCL I/O Thr`），便于使用 `htop` 等工具观察CPU线程状态[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=NCCL_SET_THREAD_NAME%EF%83%81)。

启用日志后，我们应该**重点关注**：**(1)** 每个进程是否输出了 NCCL 版本（以确认版本一致）[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=)；**(2)** 环境变量设置是否被正确读取。NCCL 在 INIT 阶段通常会打印所用环境变量值（需要 ENV 子系统日志）[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=The%20default%20value%20is%20INIT%2CBOOTSTRAP%2CENV)。例如日志可能包含“`NCCL_SOCKET_IFNAME set by environment to eth0`”等字样，确认调优参数已生效【2†L218-227】【2†L232-236】。

\*\*日志分析技巧：\*\*对于 **Hang 卡住** 的问题，INFO 级日志往往可以看到进程停在哪一步（比如所有日志停在 `... Launch mode Parallel ...` 之后，则可能卡在 kernel launch，或者停在 `Connected all rings` 之前，说明有进程通信连接未完成）。这时可以：

- 将**INFO**细化为**TRACE**重跑短测试，查看详细的通信握手过程，找出最后的操作调用序列。

- 利用进程的 stack trace（如通过 gdb 或 PyTorch 自带的 `TORCH_SHOW_CPP_STACKTRACES=1`）来定位阻塞点函数调用。

而对于**错误立即报错**的情况，`WARN` 日志即可看到 NCCL 返回的错误类型。常见错误类型如：`ncclSystemError`（系统调用失败）、`ncclUnhandledCudaError`（CUDA 调用失败）、`ncclDevMismatch`（GPU设备不一致）等[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting.html#:~:text=NCCL%20calls%20may%20return%20a,and%20returns%20a%20value%20different)。配合 NVIDIA 官方文档“Errors”章节，可以理解错误含义[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting.html#:~:text=Errors%EF%83%81)。在 PyTorch 中，如果开启 `TORCH_DISTRIBUTED_DEBUG=DETAIL`，遇到 NCCL 错误时 PyTorch 也会dump各 rank 的堆栈，辅助定位。

\*\*PyTorch 特有日志和超时Dump：\*\*PyTorch 的 `ProcessGroupNCCL` 实现有一套 Watchdog 机制，可配合 NCCL 日志定位问题：

- 设置 `TORCH_CPP_LOG_LEVEL=INFO`（或 DEBUG）可以看到 PyTorch 内部关于 ProcessGroup 和 Watchdog 的日志。

- \*\*Watchdog超时 Dump：\*\*环境变量 `TORCH_NCCL_DUMP_ON_TIMEOUT=1` 可以让当 NCCL 操作超时/异常时自动转储调试信息[docs.pytorch.org](https://docs.pytorch.org/docs/stable/torch_nccl_environment_variables.html#:~:text=)。需配合 `TORCH_NCCL_TRACE_BUFFER_SIZE` (如设为几百或几千)来开启 NCCL 内部“航迹记录”环形缓冲[docs.pytorch.org](https://docs.pytorch.org/docs/stable/torch_nccl_environment_variables.html#:~:text=)。超时发生时，每个 rank 会将最近的 NCCL 调用事件（开始/结束时间，甚至可选带 C++ 调用栈[docs.pytorch.org](https://docs.pytorch.org/docs/stable/torch_nccl_environment_variables.html#:~:text=)）写入 `TORCH_NCCL_DEBUG_INFO_*` 文件[docs.pytorch.org](https://docs.pytorch.org/docs/stable/torch_nccl_environment_variables.html#:~:text=)。这对排查集体调用失同期（desync）或 Hang 特别有用——我们可以比对各 rank 最后完成的操作，推测是哪一个 rank 停滞[docs.pytorch.org](https://docs.pytorch.org/docs/stable/torch_nccl_environment_variables.html#:~:text=than%200)。此外，`TORCH_NCCL_DESYNC_DEBUG=1` 也可用于打印可能发生不同步的提示信息[docs.pytorch.org](https://docs.pytorch.org/docs/stable/torch_nccl_environment_variables.html#:~:text=than%200)。

**日志级别策略：**在**性能问题**排查时，长时间开启 TRACE 日志不现实，可以先 INFO 粗略看每轮是否进展正常，再用 nccl-tests 短跑 TRACE 查看细节。而**稳定性问题**（Hang/错误）倾向于用 INFO + PyTorch Dump 首先收集线索，然后根据需要放大某子系统日志或使用 TRACE 重现小场景。

总之，**充分且合理过滤的日志**是 NCCL Debug 的基础。下面章节将在此基础上，讨论如何通过拓扑信息和环境变量配置进一步定位问题。

## C. 拓扑与通信路径诊断

NCCL 在初始化时会探测硬件**拓扑结构**，包括 GPU 之间以及 GPU与网络接口之间的连接关系，然后据此决定通信算法（如是否使用 NVLink）和路径选择。因此，排查跨设备通信的问题，往往需要弄清实际**数据流经路径**与 NCCL 认知的拓扑。常用方法如下：

- \*\*拓扑文件与 Dump：\*\*NCCL 提供 `NCCL_TOPO_FILE` 和 `NCCL_TOPO_DUMP_FILE` 环境变量来加载或导出拓扑信息[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=disregarding%20other%20GPUs)[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=NCCL_TOPO_DUMP_FILE%EF%83%81)。
  - `NCCL_TOPO_FILE=<path>`：指定一个 XML 文件，让 NCCL 在硬件探测前先加载此文件中描述的拓扑（如 PCIe 交换机结构、NVLink 布局等）。这常用于**容器或虚拟化**场景下，因为这些环境下 `/sys` 提供的拓扑可能是虚拟的。NCCL 默认会尝试加载 `/var/run/nvidia-topologyd/virtualTopology.xml`（如果存在）[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=NCCL_TOPO_FILE%EF%83%81)，在某些 GPU 分区或 MIG 场景下这个文件由驱动生成，描述了真实拓扑。如果你怀疑 NCCL 读到了错误的拓扑（导致算法选择不佳），可让管理员提供正确拓扑文件并用此变量加载。

  - `NCCL_TOPO_DUMP_FILE=<path>`：让 NCCL 在探测完拓扑后**导出**检测结果为 XML[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=NCCL_TOPO_DUMP_FILE%EF%83%81)。这份文件可以用于进一步分析或者在另一环境重现。当遇到跨节点通信异常时，可收集每台节点的 dump 文件，比对差异。

- \*\*查看日志中的拓扑检测：\*\*启用 `NCCL_DEBUG_SUBSYS=GRAPH`，NCCL 初始化时会打印拓扑相关信息，包括每块 GPU 的 CUDA设备号、所属 PCIe 开关以及网络接口关联等。例如日志可能显示 NVLink 连接对、InfiniBand NIC 和 GPU 的归属关系等。这能帮助确认 NCCL 判断的拓扑是否符合预期。[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting.html#:~:text=Baremetal%20systems)[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting.html#:~:text=NCCL%20relies%20on%20%2Fsys%20to,optimal%20performance)

- \*\*判定通信走哪条通道：\*\*根据 NCCL 日志和系统信息，我们能推断实际使用了 NVLink、PCIe、SHM 还是网络：
  - **NVLink**: 如果两 GPU 同机直连NVLink，NCCL 通常使用 P2P 通道直接传输。日志 `NET/Plugin` 部分不会提及 socket 或 IB 连接。可用 CUDA自带的 `p2pBandwidthLatencyTest` 工具验证GPU对间带宽是否达 NVLink 水平[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting.html#:~:text=GPU)。NVLink 6 (H100) 双向理论带宽可达 50GB/s+，NVSwitch 情况下8卡AllReduce总带宽甚至更高。

  - **PCIe**: 非 NVLink 的同机 GPU 之间，则经 PCIe 或QMPI。NCCL 日志通常会fallback到 SHM 或者 P2P (DMA) 通道，但速率受 PCIe限制。通过 `nvbandwidth` 等工具可测 PCIe 对点带宽[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting.html#:~:text=The%20test%20should%20run%20to,report%20good%20performance%20between%20GPUs)（如 PCIe3 x16 \~12GB/s，PCIe4 x16 \~25GB/s）。

  - **SHM (共享内存)**: 默认启用，用于同一主机跨 NUMA 的 GPU 间通信。当 P2P (直连) 因拓扑原因不可用时（例如不同 CPU 根连接的 GPU），NCCL 会先拷数据到系统内存再让目标GPU读回[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=The%20,GPUs%2C%20using%20NVLink%20or%20PCI)[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=NCCL_SHM_DISABLE%EF%83%81)。如果 `NCCL_SHM_DISABLE=1` 则跳过 SHM 改走网络协议[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=NCCL_SHM_DISABLE%EF%83%81)。可以通过对比开启/关闭 SHM 时性能变化来判断其作用：若关闭后同机不同NUMA GPU带宽骤降甚至类似网络水平，则原本用了 SHM。

  - **InfiniBand/RoCE**: 跨节点主要依赖 IB/RoCE 网络。日志在初始化阶段会打印诸如 “`Using xx:xx:xx (InfiniBand)`” 或者 “`NCCL NET/IB : No device found, fallback to Socket`” 等[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting.html#:~:text=GPU)。若 IB 正常，NCCL 会使用 GPU Direct RDMA (GDR) 直达 NIC；否则可能走 CPU（bounce buffers）。`NCCL_NET_GDR_LEVEL` 环境变量可以控制 GDR 使用距离阈值[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=NCCL_NET_GDR_LEVEL%20)[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=,always%20disabled)（如限制只有 NIC 与 GPU 在同一 PCIe 开关才用 GDR）。如怀疑 GDR 有问题，可尝试 `NCCL_NET_GDR_LEVEL=LOC` 完全禁用直RDMA，观察性能或稳定性是否变化。

  - **Socket (TCP)**: 当 IB 不可用或被禁用时，NCCL 会回退到 TCP/socket。日志会出现 `NCCL Net: Using Socket` 字样。这通常性能较差（几十Gb/s级别），但有助于隔离 IB 问题——如IB硬件有问题，用socket反而不hang，则进一步指向IB配置故障。

- \*\*跨网卡/多通道判断：\*\*在多 NIC 系统（如每台服务器有 dual-port IB）上，NCCL 默认尝试同一环上的节点用相同编号NIC通信[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=NCCL_CROSS_NIC%EF%83%81)[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=)（避免Rail间干扰）。可以通过设置 `NCCL_CROSS_NIC=1` 强制允许环在不同NIC交叉[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=0%3A%20Always%20use%20the%20same,need%20to%20communicate%20across%20NICs)（适合单交换机扇出网络），或 `NCCL_CROSS_NIC=0` 固定不交叉[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=)（适合双网双Rail架构）。若怀疑NCCL没有充分利用多NIC，可调整此值并用日志验证每环使用的接口变化。此外，`NCCL_IB_MERGE_NICS` 控制是否把双端口NIC当作单逻辑设备聚合带宽[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=%28since%202)（默认启用）。如果启用却性能异常波动，尝试设 `NCCL_IB_MERGE_NICS=0` 拆分使用看看区别。

**典型拓扑问题案例：**有时容器中的 `/sys` 只暴露虚拟PCI拓扑，导致 NCCL 误判。例如某 8卡机器实际有 NVSwitch，全机互联120GB/s，但容器里 /sys 不全，NCCL 未检测 NVLink，导致只用 PCIe 带宽（总线带宽仅12GB/s左右）。对此我们看到 Bus BW 明显低于硬件应有水平，日志里 Graph 拓扑只列出 PCI路径而无 NVLink。解决办法是确保挂载正确的 `/sys` 进去或使用 `NCCL_TOPO_FILE` 提供真实拓扑。另外在 VM 中，PCIe ACS 机制可能强制所有 P2P 走 CPU 根复杂交换，从而性能和稳定性降低甚至 Hang[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting.html#:~:text=IO%20virtualization%20,on%20PCI%20bridges%20by%20running)[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting.html#:~:text=redirecting%20all%20PCI%20point,on%20PCI%20bridges%20by%20running)。NCCL 文档建议**裸机禁用 ACS** 或 VM 环境下打开 NIC 的 ATS 支持[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting.html#:~:text=Virtual%20machines)。

总之，**拓扑和路径**决定了 NCCL 算法的基础。通过日志和工具确认实际的数据路径，我们才能有针对性地调整相关环境变量，见下一节。

## D. 传输层开关与网络相关环境变量

NCCL 支持多种通信传输方式，包括：GPU直连（P2P）、共享内存（SHM）、TCP Socket、InfiniBand Verbs 等。其行为可由一系列环境变量控制。下文按类别列出**常用**的网络/传输相关环境变量，以及它们的作用和典型用途（除非特别说明，均参考官方文档[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=System%20configuration%EF%83%81)[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=NCCL_SOCKET_FAMILY%EF%83%81)[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=NCCL_IB_HCA%EF%83%81)[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=NCCL_IB_TIMEOUT%EF%83%81)等）：

### **InfiniBand/RoCE 相关:**

- **设备选择**：`NCCL_IB_HCA` – 指定哪几个 HCA（IB 主机通道适配器）用于 NCCL 通信[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=NCCL_IB_HCA%EF%83%81)。可用格式如：`NCCL_IB_HCA=mlx5_0:1,mlx5_1:1`（精确指定两个卡的1号端口）[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=,mlx5)；或 `^=mlx5_3`（排除特定卡）等[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=,mlx5_1)。默认情况下，NCCL 会自动选择所有可用 IB 设备，优先同名端口。但在多 IB 网卡且某些用于其他用途时，常通过此变量**限制 NCCL 用某些端口**。有上限 32 个 HCA 设备[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=Note%3A%20using%20,to%20ensure%20an%20exact%20match)。

- **连接超时与重试**：`NCCL_IB_TIMEOUT` – 控制 IB Verbs 的**超时时间**，影响QP连接和数据超时。缺省值20，对应 4.096µs \* 2^20 ≈ 4秒的链路层超时[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=The%20,InfiniBand%20Verbs%20Timeout)[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=)。大规模集群上可能需要增大（如NCCL 初始化报 `ibv_poll_cq error 12` 则尝试调大此值[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=The%20timeout%20is%20computed%20as,to%20ibv_poll_cq%20with%20error%2012)）。`NCCL_IB_RETRY_CNT` 控制 IB 层重试次数，默认7次[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=NCCL_IB_RETRY_CNT%EF%83%81)（对应 InfiniBand spec 默认）。一般保留默认，除非特别需要避免过早断开。

- **RoCE 定位**：`NCCL_IB_GID_INDEX` – 指定 RoCE 情况下使用的 GID 表索引[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=NCCL_IB_GID_INDEX%EF%83%81)。RoCE v2 常用 index=3 (对应 IPv4) 或 index=0 (根据配置)，如遇跨网段通信问题可以尝试设置正确的 GID index。`NCCL_IB_ROCE_VERSION_NUM` – 指定 RoCE 版本 (1 或 2)[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=NCCL_IB_ROCE_VERSION_NUM%EF%83%81)，默认 2。`NCCL_IB_SL` 和 `NCCL_IB_TC` – 分别设置 IB Service Level 和 Traffic Class[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=NCCL_IB_SL%EF%83%81)[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=NCCL_IB_TC%EF%83%81)，用于 QoS 优先级，默认都为0[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=)[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=)。在拥塞场景下，可考虑给控制报文和数据报文设不同TC（2.22.3加入 `NCCL_IB_FIFO_TC` 专门为控制信道设TC[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=NCCL_IB_FIFO_TC%EF%83%81)）。

- **IB 上的 GPU Direct 开关**：早期变量 `NCCL_IB_CUDA_SUPPORT`（2.4.0 前）用于强制或禁用 GPU Direct RDMA[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=NCCL_IB_CUDA_SUPPORT%EF%83%81)。2.4.0 后改为 `NCCL_NET_GDR_LEVEL` 等统一控制。[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=NCCL_IB_CUDA_SUPPORT%EF%83%81)。当前：
  - `NCCL_NET_GDR_LEVEL` – **控制 NIC 与 GPU 间直连 RDMA 的拓扑距离阈值**[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=%28since%202,was%20renamed%20to%20NCCL_NET_GDR_LEVEL)[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=,always%20disabled)。可取 `LOC/PIX/PXB/PHB/SYS`（同 P2P_LEVEL 含义但针对NIC-GPU）[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=The%20,the%20topographical%20cutoff%20for%20GpuDirect)[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=,always%20disabled)。默认 NCCL 会自动选。例如在 CPU 直连 NIC (PHB) 的系统上，如不想用GPU直接读写NIC内存，可设 `LOC` 禁用 GDR。反之强制 GDR 则可设 `SYS`（始终开）。**调试场景**：怀疑 GDR DMA-BUF 模式有问题，可暂时降级为 CPU 中转，通过设 `NCCL_NET_GDR_LEVEL=LOC` 来验证性能/稳定性变化。

  - `NCCL_NET_GDR_READ` – 控制发送数据时是否用 GDR **Read**（NIC从GPU内存直接读）[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=NCCL_NET_GDR_READ%EF%83%81)。2.4.2 起对 NVLink 平台默认开启（=1），PCIe 平台默认0，因为某些PCIe上GPU->NIC直读反而略慢[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=Note%3A%20Reading%20directly%20from%20GPU,E)。如果遇到奇怪的性能下降，可尝试切换这个值，看是否GPU->CPU拷贝阶段出了问题[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=The%20,based%20platforms)[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=Note%3A%20Reading%20directly%20from%20GPU,E)。

  - `NCCL_NET_GDR_C2C` – (since 2.26) 针对 CPU 直连 NIC 且 CPU 经 C2C (比如 UPI) 连接GPU的场景，是否仍然启用 GDR[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=,will%20go%20through%20the%20CPU)[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=%28since%202)。默认2.27起=1启用[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=The%20,setting%20for%20this%20particular%20NIC)。若平台不支持可能需设0禁用。

- **PCIe Relaxed Ordering (RO)**：`NCCL_IB_PCI_RELAXED_ORDERING` – 控制 IBverb传输是否启用 PCIe Relaxed Ordering[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=Enable%20the%20use%20of%20Relaxed,InfiniBand%20networks%20in%20virtualized%20environments)。RO 能显著提高虚拟化环境下 IB 带宽[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=%28since%202)。默认=2（自动检测RO支持则用）[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=)。如果在 VMware/Hyper-V 等VM里性能低，检查是否RO生效，可尝试手动设=1强制开启（需要底层支持，不支持会报错）。另一方面，某些平台RO不稳定，可以=0禁用。[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=)

- **Adaptive Routing (AR)**：`NCCL_IB_ADAPTIVE_ROUTING` – 控制是否启用 IB网络的 AR 特性[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=NCCL_IB_ADAPTIVE_ROUTING%EF%83%81)。在大型Clos网络中 AR 可改善拥塞下性能。NCCL 对原生IB默认启用(=1)，RoCE默认关(=0)[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=)。如遇 IB 交换机有 AR bug，可设0禁用以验证。

- **ECE (增强连接建立)**：`NCCL_IB_ECE_ENABLE` – (2.23+) 控制是否使用 IB增强连接建立机制以支持拥塞控制等特性[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=NCCL_IB_ECE_ENABLE%EF%83%81)。默认2.19起=1 开启[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=)。配置不当时ECE可能降低性能[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=Enable%20the%20use%20of%20Enhanced,HCAs%20via%20the%20ECE%20mechanism)[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=Note%3A%20Incorrect%20configuration%20of%20the,enabled%20at%20the%20system%20level)。若怀疑，可设0禁用比较。

以上 IB/RoCE 参数很多是**系统级**调优，不建议轻易改动。但在以下情况下值得关注：**(a)** RoCE 训练出现掉包或者无法通信——检查 GID 和 RoCE v2 设置；**(b)** VM 或直通IB时性能不及裸机——考虑 Relaxed Ordering 是否启用[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=Enable%20the%20use%20of%20Relaxed,InfiniBand%20networks%20in%20virtualized%20environments)；**(c)** IB网络大规模时不稳定——可能试试关掉 AR/ECE 测试稳定性。

### **Socket/TCP 相关:**

- **接口选择**：`NCCL_SOCKET_IFNAME` – 指定 NCCL 使用的网络接口名前缀[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=NCCL_SOCKET_IFNAME%EF%83%81)[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=Examples%3A)。缺省下，NCCL 自动选择具有最高带宽/最低延迟的接口（优先 ib 开头接口）[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=Note%3A%20By%20default%2C%20the%20loopback,interfaces%20matching%20the%20manual%20selection)。但自动选择可能错误，比如多网卡环境或 Docker 虚接口。通过设此变量可以强制使用特定网卡或排除某些网卡：如 `NCCL_SOCKET_IFNAME=eth0` 只用 eth0，`NCCL_SOCKET_IFNAME=^docker,lo` 排除 docker\* 和回环。[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=,%E2%80%A6)[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=Note%3A%20By%20default%2C%20the%20loopback,interfaces%20matching%20the%20manual%20selection)。**应用场景**：多网络环境下防止 NCCL 选错（比如管理网和RDMA网都存在），明确限定接口能避免建立连接超时。

- **协议族**：`NCCL_SOCKET_FAMILY` – 强制使用 IPv4 或 IPv6 接口[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=NCCL_SOCKET_FAMILY%EF%83%81)。可设 `AF_INET` 或 `AF_INET6`。默认情况下，NCCL 会根据接口自动决定。如果遇到 v6 网络问题或名称解析问题，可尝试显式指定。

- **端口重试**：`NCCL_SOCKET_RETRY_CNT` / `NCCL_SOCKET_RETRY_SLEEP_MSEC` – 控制 TCP 连接重试次数和间隔（2.24+）[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=NCCL_SOCKET_RETRY_CNT%EF%83%81)[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=NCCL_SOCKET_RETRY_SLEEP_MSEC%EF%83%81)。默认重试34次，每次等待递增，累计约60秒[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=The%20,60%20seconds)。如果集群初始化时经常因为端口碰撞或连接临时失败，可以增大重试次数或间隔以提高成功率。

- **线程与并发**：NCCL Socket传输采用多线程模型，每条连接可用多个线程和socket并行传输以提升带宽：
  - `NCCL_SOCKET_NTHREADS` – **每个网络连接使用的 CPU 线程数**[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=NCCL_SOCKET_NTHREADS%EF%83%81)。默认云环境AWS=2, GCP gVNIC=4, 其它=1[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=1%20to%2016,the%20default%20value%20is%201)。可调范围1-16，但需注意 `NCCL_SOCKET_NTHREADS * NCCL_NSOCKS_PERTHREAD <= 64`[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=For%20generic%20100G%20networks%2C%20this,NCCL_NSOCKS_PERTHREAD)。在100Gb以上网络，可考虑手动设4线程以提升利用率[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=1%20to%2016,the%20default%20value%20is%201)。**副作用**：线程越多CPU占用越高，甚至抢占训练线程。

  - `NCCL_NSOCKS_PERTHREAD` – **每线程打开的TCP套接字数**[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=NCCL_NSOCKS_PERTHREAD%EF%83%81)。AWS默认8，其它默认1[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=)。如果单连接速度有限（如单TCP流跑不满带宽），可以每线程开多个socket并行发送。同样乘积受限64。[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=On%20AWS%2C%20the%20default%20value,the%20default%20value%20is%201)

  这两个参数对**多节点大带宽AllReduce**性能影响明显[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=the%20default%20value%20is%201)[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=For%20generic%20100G%20networks%2C%20this,NCCL_SOCKET_NTHREADS)。例如在单机4x100Gb网络的DGX A100上，默认配置可能只能到 \~80Gb/s，需要增大线程和sockets并行度才能接近理论带宽。但要小心调优需在确保通信稳定基础上进行。

- **跨 Socket 优化**：`NCCL_NET_SHARED_BUFFERS` – 控制是否启用**共享缓冲**来避免每对连接单独申请内存[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=NCCL_NET_SHARED_BUFFERS%EF%83%81)。默认1启用，通常不需改。`NCCL_NET_SHARED_COMMS` – 控制 PXN场景下是否复用连接[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=NCCL_NET_SHARED_COMMS%EF%83%81)（2.12+，默认1）。除非遇到特殊Bug，否则很少调整。

### **GPU直连 (P2P) 与 SHM 相关:**

- `NCCL_P2P_LEVEL` – **控制 GPU 间直连P2P的最大拓扑距离**[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=NCCL_P2P_LEVEL%EF%83%81)[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=)。可选：\
  `LOC`（同板直连才用P2P），`NVL`（有NVLink则用），`PIX`（同PCIe开关用），`PXB`（跨PCI开关但同CPU用），`PHB`（同NUMA节点用，即跨CPU但不跨QPI），`SYS`（即使跨QPI/UPI的NUMA也用P2P）[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=,always%20disabled)[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=,potentially%20multiple%20hops)。默认为 NCCL 自动判断。**用途**：若某拓扑层次的 P2P 性能不佳甚至出错，可通过降低此级别迫使走其它通道。例如某虚拟化下 NVLink 不可用却错误标识，可设 `PIX` 让远端NVLink不被采用。

- `NCCL_P2P_DISABLE` – 完全禁用 GPU Direct P2P 通信[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=NCCL_P2P_DISABLE%EF%83%81)。设为1后，同机 GPU 间将不走直连（无论 NVLink/PCIe），而统一经 SHM 或网络。[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=NCCL_P2P_DISABLE%EF%83%81)**调试**：如果怀疑某些 P2P 通信导致 hang（如已知NVLink某驱动Bug），可关掉验证。如果禁用后问题消失，则可以进一步细分（例如用 NCCL_P2P_LEVEL 控制不用NVLink但仍允许同PCIe直连）。

- `NCCL_P2P_DIRECT_DISABLE` – 禁用**进程内**的直接P2P访问[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=NCCL_P2P_DIRECT_DISABLE%EF%83%81)。NCCL 对于同一进程内多GPU，本可直接读写彼此显存。如果应用使用了不能共享Peer Memory的allocator，此模式可能失败。设1可强制改用更安全的路径拷贝，避免 hang[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=NCCL_P2P_DIRECT_DISABLE%EF%83%81)。

- `NCCL_SHM_DISABLE` – 禁用共享内存传输[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=NCCL_SHM_DISABLE%EF%83%81)。设1则不同进程间即使在同节点也不使用 /dev/shm 交换，而是退化为网络。**调试用途**：怀疑 /dev/shm 空间不足（初始化报错）或 SHM 通信异常时，可以关掉让 NCCL 走网络，看能否避开问题[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting.html#:~:text=If%20insufficient%20shared%20memory%20is,a%20message%20similar%20to%20this)。但性能会受影响，应尽快恢复SHM并解决根本问题（例如增大Docker的 `--shm-size`、设置 `ulimit -l unlimited` 允许内存锁定等[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting.html#:~:text=In%20particular%2C%20Docker%20containers%20default,the%20docker%20launch%20command%20line)[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting.html#:~:text=When%20running%20jobs%20using%20mpirun,init%20with%20an%20error%20like)）。

### **其他通用配置:**

- **Buffer大小**：`NCCL_BUFFSIZE` – 每个通道使用的 buffer 大小，默认 4MiB[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=NCCL_BUFFSIZE%EF%83%81)[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=The%20default%20is%204194304%20,MiB)。调小可降低内存占用、缓解OOM（代价是可能降速，因为分片变小）；调大在特定网络上可能提升长消息带宽。通常以2的幂为佳。

- **线程数**：`NCCL_NTHREADS` – 每个CUDA区块的线程数[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=NCCL_NTHREADS%EF%83%81)。默认新GPU=512线程。可设 64/128/256/512[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=)。当 GPU 核心频率很低时，多线程可能提高 pipeline 并行度，但也增大每 block 资源占用。一般无需修改，除非定位到 GPU 核心闲置才尝试。

- **通道数**：NCCL 使用多条“通道”（channel）并行通信，对应多个 CUDA block：
  - `NCCL_MIN_NCHANNELS` / `NCCL_MAX_NCHANNELS` – 限制最少/最多通道数[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=%28NCCL_MIN_NRINGS%20since%202,0)[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=%28NCCL_MAX_NRINGS%20since%202,0)。旧版本叫 NRINGS。这影响 GPU 参与通信的 block 数。增加 channels 有助于提升大量小消息的重叠效率，但过多会争夺 GPU 计算资源[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=The%20,uses%20more%20CUDA%20compute%20resources)。NCCL 2.5 起推荐通过更细粒度的 `NCCL_MIN_CTAS`/`NCCL_MAX_CTAS` 控制每SM并发CTA数量[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=The%20old%20,is%20set)。通常除非做性能优化，不建议显式修改这些。

- **Check校验**：`NCCL_CHECKS_DISABLE`（**已废弃**）– 关闭参数合法性检查，可略微降低延迟[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=NCCL_CHECKS_DISABLE%EF%83%81)。2.2.12 后改用 `NCCL_CHECK_POINTERS` 控制是否检查CUDA指针有效性[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=NCCL_CHECK_POINTERS%EF%83%81)。默认关闭检查以提高性能，除非调试内存问题不需要打开。

以上设置很多仅在特定排障或调优场景使用，不宜长期在生产中开启[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=There%20are%20two%20categories%20of,optimal%20behavior%2C%20crashes%2C%20or%20hangs)。**一般原则**：逐步尝试**粒度尽可能小**的干预（如先禁用怀疑模块，再细化），以免引入新的不确定因素。

## E. 算法与协议相关调试手段

NCCL 针对不同规模和拓扑，会在 Ring、Tree、CollNet 等多种**算法**，以及 Simple、LL、LL128 等多种**通信协议**之间自动选择。某些bug或性能问题可能与算法/协议选择有关。因此 NCCL 提供环境变量来**强制或排除**特定算法/协议，从而帮助我们诊断。

- **协议选择 (`NCCL_PROTO`):** 控制允许使用的消息传输协议，包括 **Simple**（分段复制，适用于大消息高带宽）、**LL**（Low Latency，适用于小消息低延迟）、**LL128**（优化长消息的小延迟算法，需要硬件支持）[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=NCCL_PROTO%EF%83%81)。用法为列出协议或以 `^` 列出排除协议[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=)。默认行为：支持 LL128 的平台开启全部三种，否则LL128不用[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=)。**重要提示**：NVIDIA 明确指出，不要随意启用 LL128 在不支持的平台，否则**可能导致数据错误**[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=The%20,will%20be%20allowed%20to%20use)。LL128 一般要求 NVLink 拓扑良好的平台（如DGX），在PCIe集群上NCCL默认已禁用LL128[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=)。调试中，**禁用 LL128** 是常用手段：不少 NCCL 已知Bug（比如 2.8版本 Collnet 算法配合 LL128 在部分拓扑上出错）可以通过 `NCCL_PROTO=^LL128` 规避[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=The%20,will%20be%20allowed%20to%20use)。如果问题消失，可据此怀疑 LL128 实现问题然后查找对应补丁或升级NCCL版本。

- **算法选择 (`NCCL_ALGO`):** 控制集合通信算法，如 Ring、Tree、CollNet 等[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=NCCL_ALGO%EF%83%81)[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=Comma,among)。2.24+版本支持更复杂的配置语法，可按操作类型分别指定算法列表或排除[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=To%20specify%20algorithms%20to%20exclude,start%20the%20list%20with)[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=The%20format%20is%20now%20a,all%20the%20selections%20are%20inverted)。例如：\
  `NCCL_ALGO=Ring` 强制全部用环形算法；\
  `NCCL_ALGO=^Tree` 禁用树算法（如怀疑 Tree 实现有Bug，NCCL 会自动fallback环算法）；\
  `NCCL_ALGO="allreduce:tree,ring"` 仅AllReduce用树或环，其它操作不变。\
  默认NCCL会根据节点拓扑和消息大小自动混用多种算法，避免**盲目固定**导致性能下降[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=The%20accepted%20values%20are%20expanded,Instead%2C%20it%20will%20fail)[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=specified%20as%20a%20valid%20algorithm,Instead%2C%20it%20will%20fail)。然而调试时，当某算法路径怀疑有问题，可以用排除法验证。例如树形算法在跨机时延较大，可以暂禁 Tree 看性能是否提升，从而确认是否需要调整树算法触发阈值（老版本通过 NCCL_TREE_THRESHOLD 控制消息大小阈值[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=NCCL_TREE_THRESHOLD%EF%83%81)）。又如 CollNet 算法（要求特殊网络硬件）在不支持场景下应该自动不用，但如怀疑错误触发，可直接 `^CollNet`。

- **链路聚合算法 (NVLS/Multi-NIC 等)**：新版本 NCCL 针对 NVSwitch 平台引入 NVLS（NVLink SHARP）算法，以及 MNNVL（跨节点NVLink）支持等[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=NCCL_NVLS_ENABLE%EF%83%81)[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=NCCL_MNNVL_ENABLE%EF%83%81)。环境变量如 `NCCL_NVLS_ENABLE` 控制 NVLS 开/关[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=Enable%20the%20use%20of%20NVLink,The%20default%20value%20is%202)（默认2=自动），`NCCL_MNNVL_ENABLE` 控制多节点NVLink[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=%28since%202)。这些一般NCCL默认自动处理。如果遇到 NVLS 资源分配失败引起 hang（2.27版一度出现silent fallback hang问题[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=cannot%20be%20allocated)），可以临时 `NCCL_NVLS_ENABLE=0` 来禁用 NVLS 验证是否问题消失，然后升级新版修复。

- **PXN 机制**（通信基于中间GPU转发）：变量 `NCCL_PXN_DISABLE` (2.12+) 禁用跨节点NVLink转发[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=Default%20is%200%2C%20set%20to,1%20to%20disable%20this%20mechanism)，`NCCL_P2P_PXN_LEVEL` 控制何种情况下使用PXN[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=Control%20in%20which%20cases%20PXN,is%20used%20for%20send%2Freceive%20operations)，以及 `NCCL_PXN_C2C` 控制 C2C 互联时PXN是否可用[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=NCCL_PXN_C2C%EF%83%81)。这些属于**高级优化**，一般无需手调。但在 NVSwitch + IB 的架构中，如果观察到某些GPU流量绕远了，可以看看 PXN 相关配置是否合理。例如默认 `NCCL_P2P_PXN_LEVEL=2` 总是用PXN，有时可能导致不必要的中转占用 NVLink，调为1或0可做比较。

**算法/协议排查思路：**当怀疑 NCCL 内部选择不佳时，可以**依次排除**：先禁 CollNet/NVLS（这些依赖特殊硬件，禁用不影响常规Ring/Tree运行）；再禁 Tree 观察（尤其大批节点场景，tree深度大时易受网络延迟影响）；最后再考虑禁 Ring（一般不需要，因为NCCL总会留至少Ring保证functional）。协议方面则首选**禁LL128**试验，其次 LL vs Simple 切换对比小消息性能和稳定性。需要注意的是，这些变量**仅用于临时诊断**，生产环境遇到相关问题最好升级NCCL或调整代码，让 NCCL 自动策略生效，而非长期强制某算法——正如官方文档所警告的，强制算法会“prevent NCCL from selecting the best setting... cause performance problems or even break functionality”[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=Debugging%EF%83%81)。

## F. 稳定性与容错：Hang/超时/错误处理

大规模分布式训练，除了性能，还必须关注**稳定性**。NCCL 在2.20+版本逐步增强了容错和诊断能力，包括引入 RAS 子系统（Reliability, Availability, Serviceability）和结合框架的 Watchdog 机制。以下是相关工具和环境变量：

### **NCCL 异常处理与 RAS：**

- **异步错误监测**：NCCL 内部如果检测到严重异步错误（如网络掉线、GPU故障）会尝试使通信停止并返回错误。2.23引入 `NCCL_IB_RETURN_ASYNC_EVENTS`（默认1）控制 IB 异步事件处理[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=NCCL_IB_RETURN_ASYNC_EVENTS%EF%83%81)。设为0则忽略IB驱动的异步错误，仅靠超时。这在某些调试下有用（例如允许程序在错误发生后一段时间继续运行，便于收集状态），但一般保持默认即可。

- **NCCL RAS 子系统**：从 NCCL 2.24 起，可以通过 RAS 接口**查询 NCCL communicator 的运行状态**，实现外部监控。相关变量：
  - `NCCL_RAS_ENABLE` – 开启 RAS 功能[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=NCCL_RAS_ENABLE%EF%83%81)（默认1启用[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=)）。如不需要可设0完全关闭。

  - `NCCL_RAS_ADDR` – 指定 RAS 服务监听的 `<ip>:<port>`[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=NCCL_RAS_ADDR%EF%83%81)。默认 `localhost:28028`[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=)。在多用户节点上，每个作业应设不同端口避免冲突[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=%28since%202)。

  - `NCCL_RAS_TIMEOUT_FACTOR` – RAS 内部各种超时的**倍率**[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=NCCL_RAS_TIMEOUT_FACTOR%EF%83%81)。RAS 会周期性检查通信进展，默认有5\~60秒不等的超时[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=%28since%202)。如程序被调试器挂起导致超时，可临时把 factor 设大避免误判。

  开启后，可使用 NCCL 提供的 `ncclras` CLI 工具连接 RAS 端口查询状态（如有哪些Collective在进行，是否卡住）[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=Enable%20NCCL%E2%80%99s%20reliability%2C%20availability%2C%20and,see%20RAS)[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=Specify%20the%20IP%20address%20and,instead%2C%20which%20will%20make%20RAS)。这在**Hang未超时**时特别有价值，可以辅助判断是哪一步停滞。不过 RAS 属新特性，目前主要用于 NVIDIA 内部监控和高级用户。

- **Abort 行为**：NCCL 默认在检测到无法恢复的错误时会调用 `ncclCommAbort` 终止 communicator（而不是安静Hang）。在较新版本，NCCL abort 会打印更详细的上下文信息。用户无须配置此功能，但要确保捕获并处理返回的 ncclResult_t 错误码。

### **PyTorch ProcessGroupNCCL 容错设置：**

PyTorch 自己也提供了**环境变量**来控制 NCCL 后端的错误处理和超时机制[docs.pytorch.org](https://docs.pytorch.org/docs/stable/torch_nccl_environment_variables.html#:~:text=)[docs.pytorch.org](https://docs.pytorch.org/docs/stable/torch_nccl_environment_variables.html#:~:text=)：

- **Watchdog 线程 & 阻塞等待**：默认情况下，PyTorch 每个进程启动一个Watchdog线程监视 NCCL 操作是否卡住。当某GPU卡住时，Watchdog会在一定时间后使所有进程报错退出。可以通过 `torch.distributed.init_process_group(timeout=...)` 设置超时时间（默认一般 30min）。以下环境变量可调整此行为：
  - `TORCH_NCCL_BLOCKING_WAIT` – 设为 `1` 则使得 `dist.all_reduce(...).wait()` 等待调用变为**阻塞模式**[docs.pytorch.org](https://docs.pytorch.org/docs/stable/torch_nccl_environment_variables.html#:~:text=Control%20whether%20to%20use%20high,stream%20for%20the%20NCCL%20communicator)。即发生超时时，会抛出异常而不是静默等待。建议在调试时开启，以便及时捕获Hang而不是无限挂住进程。

  - `TORCH_NCCL_ASYNC_ERROR_HANDLING` – 控制异步错误处理策略[docs.pytorch.org](https://docs.pytorch.org/docs/stable/torch_nccl_environment_variables.html#:~:text=)。默认 `3`，表示一旦超时，**所有进程**一起安全退出（由主进程决定不用先abort communicator，就直接退出）[docs.pytorch.org](https://docs.pytorch.org/docs/stable/torch_nccl_environment_variables.html#:~:text=Control%20how%20we%20perform%20Async,it%20is%20set%20to%203)。选项说明：0=不处理异步错误（可能导致hang住不退出）；1=检测到错误后调用 NCCL Comm.abort 并 kill 进程；2=仅 abort communicator 但不杀进程；3=直接杀进程不做 abort。[docs.pytorch.org](https://docs.pytorch.org/docs/stable/torch_nccl_environment_variables.html#:~:text=Control%20how%20we%20perform%20Async,it%20is%20set%20to%203)调试中推荐用默认3或选1。设0则可能某些rank卡死无法退出。

  - **实用组合**：`TORCH_NCCL_BLOCKING_WAIT=1` + `NCCL_DEBUG=WARN` 是 PyTorch 官方建议用于debug hang的设置，可让在超时发生时抛异常并打印 NCCL 错误日志[forums.developer.nvidia.com](https://forums.developer.nvidia.com/t/what-is-the-busbw-in-nccl-tests/256858#:~:text=The%20published%20info%20on%20NCCL,p2pBandwidthLatencyTest)。

- **超时信息收集**：前述 `TORCH_NCCL_DUMP_ON_TIMEOUT=1` 配合 Trace Buffer，可以在Watchdog认定超时时，自动收集调试信息[docs.pytorch.org](https://docs.pytorch.org/docs/stable/torch_nccl_environment_variables.html#:~:text=)。另外还有：
  - `TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC` – Watchdog心跳检测的周期，默认约 5s。`TORCH_NCCL_ENABLE_MONITORING=1` 时，PyTorch会再启一个监控线程，如果发现 **Watchdog 本身**卡死（可能因为死锁），则在此时间后**强制kill**进程[docs.pytorch.org](https://docs.pytorch.org/docs/stable/torch_nccl_environment_variables.html#:~:text=)[docs.pytorch.org](https://docs.pytorch.org/docs/stable/torch_nccl_environment_variables.html#:~:text=If%20set%20to%20,necessary%20tying%20up%20cluster%20resources)。一般不需改这个值，除非调试环境下希望更快触发监控。

  - `TORCH_NCCL_COORD_CHECK_MS` / `TORCH_NCCL_WAIT_TIMEOUT_DUMP_MS` – 这些控制多个rank协调dump的时序和等待时间[docs.pytorch.org](https://docs.pytorch.org/docs/stable/torch_nccl_environment_variables.html#:~:text=)[docs.pytorch.org](https://docs.pytorch.org/docs/stable/torch_nccl_environment_variables.html#:~:text=)。除非深入分析，否则用默认即可（1000ms间隔，额外等待同样长收集完dump）。

- **数据检查**：`TORCH_NCCL_NAN_CHECK=1` 可在每次collective调用时对张量进行 NaN/Inf 检查[docs.pytorch.org](https://docs.pytorch.org/docs/stable/torch_nccl_environment_variables.html#:~:text=)。发现NaN会报错退出，防止带着坏数据进行 AllReduce。这在怀疑 NCCL 数据腐蚀或上层算子问题时有帮助。但注意性能损耗较大，仅调试暂时开启。

通过以上机制，PyTorch 尽量做到**某进程出错，整体及时退出**，防止集群资源长时间被挂住进程占用[docs.pytorch.org](https://docs.pytorch.org/docs/stable/torch_nccl_environment_variables.html#:~:text=If%20set%20to%20,necessary%20tying%20up%20cluster%20resources)。调试过程中，充分利用这些设置能**缩短排查周期**：与其等待默认30分钟超时，不如设置短超时并开启Dump，快速拿到信息。

**经验**：排查 NCCL hang，应尽量在**出错时刻**就收集信息，而非等作业被迫杀死后再分析。Watchdog+Dump 提供了这样的契机。但另一方面，要防止误触发，比如调优时可能 AllReduce 本身就需要较长时间，此时可暂时调大 `timeout` 以免误判。

---

以上介绍了 NCCL Debug 的各项“武器”。接下来我们将它们应用到具体的**故障场景**中。

## G. 常见故障场景手册（10+案例）

本节按典型现象列举多种 NCCL 故障场景，分析可能原因并给出**优先级渐进**的排查步骤、建议的环境变量设置组合，以及如何用 nccl-tests 等工具复现验证。

**场景1：训练开始时 NCCL 初始化 Hang**

- **现象**：分布式作业启动后打印 NCCL 版本号，但一直卡在 communicator 初始化，既无error也无进展。可能所有进程都挂在 `ncclCommInitRank`。

- **可能原因**：跨节点通信握手不通。常见包括：防火墙未关闭导致 TCP/IB 端口无法建立；节点间网络配置不一致（如一台走 IB 一台却无 IB）；`init_process_group` 参数 world_size 等不匹配；或 IB 的GID配置导致握手包丢弃。

- **排查步骤**：
  1. **基础连通性**：确认各节点间彼此能 ping 通，并且没有防火墙阻挡 NCCL 默认使用的端口 (NCCL默认随机挑选高位端口，可通过 `net.ipv4.ip_local_port_range` 调整范围[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting.html#:~:text=For%20information%20about%20how%20to,environment))。对使用 IB/RoCE 的，检查 `ibstat` 状态、子网管理器（Subnet Manager）正常。

  1. **接口选择**：在环境中显式 `NCCL_DEBUG=INFO` 看日志哪个接口在尝试连接。若看到 fallback 到 Socket 或 `[0] NET/IB: No device found` 则 IB 未被识别。[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting.html#:~:text=GPU)。可以尝试设置 `NCCL_SOCKET_IFNAME` 明确指定正确的网络，例如 `NCCL_SOCKET_IFNAME=^eth,ib0`（排除无关接口）。

  1. **禁用IB验证**：若怀疑 IB 配置问题，临时 `NCCL_IB_DISABLE=1` 强制走 TCP[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=NCCL_IB_DISABLE%EF%83%81)。如果这样就能初始化成功（尽管后续AllReduce慢），说明 IB 通信有问题。接下来重点检查 RoCE 配置（例如 `NCCL_IB_GID_INDEX` 是否一致）以及IB固件/驱动。

  1. **分步缩小**：编写一个最小复现脚本，例如使用 nccl-tests：\
     `mpirun -np 2 -H host1:1,host2:1 ./build/all_reduce_perf -b 8 -e 8M -f 2`\
     尝试在两节点上跑简单 AllReduce，看能否Hang复现。加上 `NCCL_DEBUG=INFO` 捕获在哪一步挂。

- **建议env组合**：
  - _保守调试_：`NCCL_DEBUG=INFO NCCL_SOCKET_IFNAME=<iface>` 用于观察和纠偏。

  - _激进尝试_：`NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=<iface>` 验证是否 IB 专有问题；若确认为IB问题，进一步 `NCCL_IB_GID_INDEX` 等配置比对两端。

- **验证修复**：在确认网络配置无误后（如关闭防火墙或正确设置RoCE PFC等），重新打开 IB 跑 nccl-tests 验证 AllReduce 成功、带宽正常。

**场景2：训练中途某一步挂死（没有显式 error）**

- **现象**：训练运行一段时间后，所有GPU利用率掉为0，进程无响应但未退出。可能日志停在某次collective操作前后，没有错误提示。

- **可能原因**：这通常是**Collective 调用失去同步**（Desynchronization）造成的死锁。可能一个rank跳过或提前退出导致其余rank卡在对应的AllReduce/AllGather。也可能某rank上发生了CUDA错误被吞掉，导致NCCL等待永远不返回。NCCL本身Bug（比如2.7.x曾有LL128算法在特定拓扑卡死的问题）也可能导致所有rank hang。

- **排查步骤**：
  1. **判断哪种Hang**：首先区分是**所有rank都在等**（典型集体不同步），还是**个别rank崩溃**导致others在等。可以通过`dmesg`查看是否有GPU异常日志（如kernel打印 Xid错误表示某rank GPU出问题），也可使用 PyTorch 的 `TORCH_NCCL_BLOCKING_WAIT=1` 让出问题rank抛异常而不是静默挂住[docs.pytorch.org](https://docs.pytorch.org/docs/stable/torch_nccl_environment_variables.html#:~:text=Control%20whether%20to%20use%20high,stream%20for%20the%20NCCL%20communicator)。

  1. **Desync Debug**：设置 `TORCH_NCCL_DUMP_ON_TIMEOUT=1` 并将超时设短（例如5分钟）来触发超时dump[docs.pytorch.org](https://docs.pytorch.org/docs/stable/torch_nccl_environment_variables.html#:~:text=)。同时开 `TORCH_NCCL_DESYNC_DEBUG=1` 以帮助发现不同步信息[docs.pytorch.org](https://docs.pytorch.org/docs/stable/torch_nccl_environment_variables.html#:~:text=than%200)。超时后检查每个rank转储的trace，找出哪个rank在某collective上没有进入或没有退出。比如可能 rank7 停在 allreduce(stream X) 未调用，而其他都完成，则说明rank7代码有分支漏调。

  1. **协议算法角度**：如果所有rank显示都进入了一次AllReduce但出不来，考虑是否NCCL内部死锁。这种情况下可尝试 `NCCL_PROTO=^LL128` 或 `NCCL_ALGO=Ring` 等（逐一改变），看问题是否不再复现。如果禁用LL128后不hang了，则很可能碰到NCCL已知Bug，需要升级NCCL版本[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=The%20,will%20be%20allowed%20to%20use)。

  1. **外部介入**：利用 `gdb` attach到挂住的一个进程，打印堆栈。如果看到某 NCCL kernel 卡在CUDA sync，可能CUDA这端有异常（如非法内存访问未报）。这时设置环境 `CUDA_LAUNCH_BLOCKING=1` 重运行一次，方便让CUDA错误暴露。

- **建议env组合**：
  - _配合监控_：`TORCH_NCCL_BLOCKING_WAIT=1 TORCH_NCCL_ASYNC_ERROR_HANDLING=1` 使任何rank出错立刻中止所有进程，防止部分hang。

  - _Dump信息_：`TORCH_NCCL_DUMP_ON_TIMEOUT=1 TORCH_NCCL_TRACE_BUFFER_SIZE=1000000 TORCH_NCCL_DEBUG_INFO_TEMP_FILE=/tmp/nccl_dump_%h_%p.json` 收集大量调用踪迹。一旦触发，可用工具/脚本汇总对比各rank日志。

  - _隔离NCCL问题_：`NCCL_PROTO=^LL128` 试排除协议因素；`NCCL_ALGO=Ring` 固定算法验证。

- **验证修复**：找到根因后采取相应措施。例如如果是应用代码漏调 collective，要修复逻辑。如果是NCCL bug，则升级到官方修复版本或继续使用工作区（如禁用LL128作为workaround）。最终在修复版本环境下长时间跑验证Hang不再发生。

**场景3：AllReduce 性能严重低于理论带宽**

- **现象**：8卡单机A100，预期 NVSwitch 可达 240 GB/s，但实际 all_reduce_perf 只得到 80 GB/s 算术带宽(algbw)，busbw 约 80 GB/s[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting.html#:~:text=If%20insufficient%20shared%20memory%20is,a%20message%20similar%20to%20this)。或多机时总带宽远低于网络物理速率。例如双40GbE机器AllReduce总吞吐只有 2GB/s (16Gb/s)。

- **可能原因**：**数据路径未充分利用带宽**。单机情况可能 NCCL 未用 NVSwitch 而退化为 PCIe4（约64–80 GB/s，符合观测）。原因如拓扑探测问题、NVSwitch驱动问题等[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting.html#:~:text=If%20insufficient%20shared%20memory%20is,a%20message%20similar%20to%20this)[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting.html#:~:text=Docker)。多机情况，则可能只用了单端口而非Bond、或 GPU Direct RDMA 未启用导致受 CPU 内存复制瓶颈（典型CPU copy速率 \~10-20GB/s），或者线程并行度不够未填满带宽。

- **排查步骤**：
  1. **查看 Bus BW vs Alg BW**：用 `NCCL_DEBUG=INFO` 跑 `all_reduce_perf -g 8 -n 10` 并观察输出。例如 8卡 NVSwitch 理论一来一回BusBW=144 GB/s[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting.html#:~:text=If%20insufficient%20shared%20memory%20is,a%20message%20similar%20to%20this)，而Algbw=120 GB/s时BusBW应达 \~240 GB/s[forums.developer.nvidia.com](https://forums.developer.nvidia.com/t/what-is-the-busbw-in-nccl-tests/256858#:~:text=measured%20for%20the%20operation%2C%20the,p2pBandwidthLatencyTest)。如果BusBW恰好等于当前物理接口峰值，比如 80GB/s \~ PCIe4 x16极限，那么说明NCCL只用了PCIe没有NVSwitch。

  1. **拓扑检测**：检查 NCCL 拓扑日志是否识别 NVSwitch/NVLink（见 C 节内容）。若没有，可考虑驱动或环境问题：确保裸机运行、CUDA driver 正确加载 NVSwitch 控制器。尝试升级驱动或补丁。

  1. **网络瓶颈**：在多机上，对比 `algbw` 和 `busbw`：busbw 代表实际流经网络数据速率[forums.developer.nvidia.com](https://forums.developer.nvidia.com/t/what-is-the-busbw-in-nccl-tests/256858#:~:text=measured%20for%20the%20operation%2C%20the,p2pBandwidthLatencyTest)。如2机100Gbps网络理想busbw≈12.5 GB/s。但若 busbw只有6 GB/s且 algbw更低，则可能 GPU->NIC GDR未用上（需要CPU中转耗时）。验证方法：比较使用 GDR 与否性能，手动 `NCCL_NET_GDR_LEVEL=SYS` 强制GPU直RDMA。如果性能提升，说明之前GPUDirect未启用，可能因为需要加载 `nvidia-peermem` 模块或 NIC 不支持 DMA-BUF[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting.html#:~:text=GPUs%20can%20also%20communicate%20directly,on%20each%20node%20boot%20with)。反之如强制GDR性能下降甚至不稳定，则可能是ROCE PFC没配好造成丢包重传。

  1. **并行调优**：排除以上因素后，如果仍然低于理论，可以尝试**增加并发**：调整 `NCCL_SOCKET_NTHREADS` 和 `NCCL_NSOCKS_PERTHREAD`[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=1%20to%2016,the%20default%20value%20is%201)[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=On%20AWS%2C%20the%20default%20value,the%20default%20value%20is%201)。特别在高速以太网上，默认 (1线程,1 socket) 很可能跑不满 100Gb。尝试值如4和4（总16 socket并行），观察 busbw 是否接近物理线速。注意此调整需在较大 batch 下观察平均性能，并警惕CPU占用上升。

- **建议env组合**：
  - _拓扑修正_：容器中建议 `--cap-add SYS_NICE` 以启用 NUMA 支持，或挂载正确的 /sys。针对 NVSwitch 可用 `NCCL_TOPO_DUMP_FILE` 确认拓扑识别结果。

  - _性能调优_：`NCCL_SOCKET_NTHREADS=4 NCCL_NSOCKS_PERTHREAD=4 NCCL_NET_GDR_LEVEL=PXB`（例如只允许在同PCI域用GDR，跨CPU用bounce缓冲）。[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=For%20generic%20100G%20networks%2C%20this,NCCL_NSOCKS_PERTHREAD)[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=For%20generic%20100G%20networks%2C%20this,NCCL_SOCKET_NTHREADS)这些组合需根据观察逐步调整。并**仅在确认稳定后**用于生产。

- **验证修复**：重新运行 nccl-tests 并比较带宽：Algorithm BW 提升且 busBW 接近硬件峰值（例如 12 GB/s 于100GbE，或NVSwitch下达到120+ GB/s）。还应测试实际训练任务的 step time 是否同步改善，以确保调优有效且无副作用。

**场景4：NCCL 报错 “Unhandled system error” 或 “CUDA Driver error”**

- **现象**：训练中突然终止，并打印 `ncclSystemError: System call (socket, malloc, etc) failed` 或 `ncclUnhandledCudaError` 等。可能还有 IBverbs 层错误信息如 “**failed to register memory**” 或 “**RDMA creation failed**”。

- **可能原因**：**系统资源或调用失败**。典型如：/dev/shm 空间不足导致共享内存segment扩展失败[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting.html#:~:text=If%20insufficient%20shared%20memory%20is,a%20message%20similar%20to%20this)；无限制内存锁定不允许导致 GDR mapping 失败；或CUDA Driver内部错误比如显存访问非法。

- **排查步骤**：
  1. **错误码判断**：`ncclSystemError` 通常表示某个系统API返回错误，可以配合前面的 NCCL WARN 日志找上下文。例如若紧随 “unable to allocate shared memory” 则很明确。`ncclUnhandledCudaError` 则需看是不是之前有 kernel failed 日志。

  1. **共享内存问题**：容器环境下，默认 /dev/shm 仅64MB，远不够多GPU全通信buffer。NCCL初始化时若失败，会 WARN 提示扩展shm失败[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting.html#:~:text=If%20insufficient%20shared%20memory%20is,a%20message%20similar%20to%20this)。解决：Docker跑容器加 `--shm-size=1g --ulimit memlock=-1`[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting.html#:~:text=In%20particular%2C%20Docker%20containers%20default,the%20docker%20launch%20command%20line)[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting.html#:~:text=When%20running%20jobs%20using%20mpirun,init%20with%20an%20error%20like)。另外检查 systemd 是否移除了用户IPC[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting.html#:~:text=When%20running%20jobs%20using%20mpirun,init%20with%20an%20error%20like)（需要 /etc/systemd/logind.conf 设置 RemoveIPC=no）。

  1. **IB 内存注册失败**：如果错误出现在首次 AllReduce 前后，并包含 ibv_reg_mr 失败，可能是进程的内存锁定 (memlock) ulimit 太低。GPUDirect RDMA 需要注册显存映射到HCA，一张 32GB 卡需要注册同等大小内存。将 `ulimit -l` 调为足够（如无限）并确保 `NCCL_MEM_AFFINITY` 环境正确。

  1. **CUDA 异常**：NCCL 使用CUDA流，如果用户前面发生了CUDA illegal memory access，可能在ncclGroupWait时抛出 unhandled cuda error。此类应回溯定位之前的CUDA调用bug，不是NCCL自身问题。可以利用 `cuda-memcheck` 工具运行程序，早期发现非法访问。

- **建议env组合**：
  - 针对shm/内存问题，`NCCL_SHM_DISABLE=0 NCCL_CUMEM_HOST_ENABLE=0` 可尝试不用 cuMem host机制强制用 /dev/shm，以验证是哪种方式问题（2.24+默认用cuMemHost，有时NUMA不支持）[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting.html#:~:text=Starting%20with%20version%202,default%20in%20favor%20of%20%2Fdev%2Fshm)[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting.html#:~:text=%2Fdev%2Fshm%20code,improved%20reliability%20during%20communicator%20aborts)。

  - 对 IB MR 问题，可设 `NCCL_IB_HCA=<specific>` 只用一块HCA测试，或 `NCCL_P2P_DISABLE=1` 绕过GPUDirect RDMA。

  - `CUDA_LAUNCH_BLOCKING=1` 辅助捕获CUDA同步错误。

- **验证修复**：调整系统配置后，重复运行之前出错的位置。如果不再报错且日志中先前的 WARN 提示消失（如共享内存扩展成功或不再需要扩展），则问题解决。需要的话，在调通后可逐步恢复优化选项（如重新打开 `NCCL_CUMEM_HOST_ENABLE` 看是否依旧稳定），以兼顾性能和稳定性。

**场景5：多机通信经常性波动，性能时高时低**

- **现象**：同一任务，不同 step 的AllReduce耗时抖动很大。例如100Gb网络下正常 allreduce 5ms，但偶尔跳到50ms，然后恢复。甚至伴随 NCCL WARN：`NET/IB : Async event: local QP operation err` 之类[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=NCCL_IB_RETURN_ASYNC_EVENTS%EF%83%81)。

- **可能原因**：**网络拥塞或丢包**导致。InfiniBand网络中，当流量大时可能触发拥塞管理或QOS，Adaptive Routing的切换也会导致波动。RoCE 如果PFC配置不完善，可能出现丢包超时重试，使性能断崖式下降。NCCL检测到 IB异步错误时（比如链路波动）默认会Warn然后重连[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=NCCL_IB_RETURN_ASYNC_EVENTS%EF%83%81)。

- **排查步骤**：
  1. **NCCL 日志**：观察NCCL INFO日志中是否频繁出现 `...Disconnecting`、`...Reconnecting`，或 RNR NACK 等IB级别消息。这些表明网络不稳导致重试。

  1. **底层监控**：使用 Infiniband自带工具查看错误计数，如 `ibporterr` 是否增长，`sar -n EDEV` 看各网卡丢包。

  1. **拥塞控制**：如果是RoCEv2网络，确认交换机和网卡配置了 PFC（优先级流控）和 ECN，否则遇到深度缓冲拥塞会丢包导致NCCL重试超时。对于InfiniBand HDR/EDR网络，可检查是否启用了动态拥塞控制（需要 NIC FW 支持）。

  1. **NCCL 调参**：尝试暂时关闭 Adaptive Routing：`NCCL_IB_ADAPTIVE_ROUTING=0`[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=NCCL_IB_ADAPTIVE_ROUTING%EF%83%81)看看波动是否减少。如果有效，可能AR机制不成熟导致reorder，可考虑升级FW或者先禁用。对 RoCE，可以通过降低 `NCCL_IB_TIMEOUT`（比如设18）使超时更敏感，但这治标不治本。

- **建议env组合**：
  - `NCCL_IB_SL=` 设一个高优先级SL用于NCCL，确保交换机QoS优待；配合 `NCCL_IB_FIFO_TC` 把控控制消息TC[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=NCCL_IB_FIFO_TC%EF%83%81)。

  - `NCCL_IB_ADAPTIVE_ROUTING=0` 如上，避免路由波动。

  - 在应用侧，考虑 `torch.backends.cuda.matmul.allow_tf32 = False` 等减少通信量或者梯度压缩以减小网络压力。

- **验证修复**：调整后长时间跑任务，记录AllReduce时间分布，看是否抖动降低。若还存在，则需要进一步比如对每对节点使用 `ib_send_bw` 工具测试裸带宽，锁定是否某特定链路的问题。最终稳定后，应在生产中保留必要的NCCL参数，并将集群网络配置优化（长远方案）。

**场景6：开启混合精度后偶发 NaN/Inf，怀疑通信精度**

- **现象**：训练中偶尔出现梯度为 NaN 或损失暴涨，定位怀疑发生在AllReduce后。怀疑 NCCL 的 sum 精度或LL128压缩算法导致精度损失。

- **可能原因**：NCCL的 float16 AllReduce 默认分两阶段（First reduce in FP16, then finalize in FP32）。精度一般足够。但在极端大规模下，累加顺序可能引入些许不确定。另外 LL128 协议会对数据分块应用低精度 accumulate，存在微小误差。这通常不会导致NaN，NaN更多由于网络错误或算子本身。

- **排查步骤**：
  1. **验证NaN来源**：使用 `TORCH_NCCL_NAN_CHECK=1` 提前检测各步输出NaN[docs.pytorch.org](https://docs.pytorch.org/docs/stable/torch_nccl_environment_variables.html#:~:text=)。看看是否某rank的激活值先成为NaN，而非AllReduce过程注入。

  1. **关闭融合**：禁用GradScaler或将accumulation降低，看看NaN是否还出现。可能是数值本身爆了而非通信。

  1. **协议替换**：试 `NCCL_PROTO=Simple` 强制不用LL/LL128。如果NaN不再出现，可能LL128某bug引发错误sum。也可尝试 `NCCL_ALGO=Tree` 改变累加次序看看。

  1. **Check通信正确性**：用 nccl-tests 自带的验证模式运行几千轮：`all_reduce_perf -c 1 -check` 开启数据正确性检查。如果都有 Pass，则NCCL本身逻辑没问题。

- **建议env组合**：
  - 为安全，可将 `NCCL_ALGO=Ring NCCL_PROTO=Simple` 在要验证精度的实验中使用，确保按最高精度路径汇总。

  - 如果多节点间有可能数据不一致，也可利用 `TORCH_DISTRIBUTED_DEBUG=INFO` PyTorch在不同步时会有提示。

- **验证修复**：确认调整后NaN问题不再出现。若确定是NCCL协议问题，应向NVIDIA反馈或查看release notes已知问题。否则，多半是训练本身需调整（如降低学习率等）。

**场景7：单机多进程模式下 NCCL 初始化缓慢**

- **现象**：例如 PyTorch DDP 模式，8卡单机，调用 `init_process_group` 非常慢（> 30秒），但最终能成功开始训练。

- **可能原因**：在单机多进程场景，NCCL 需要通过 socket 进行 out-of-band 引导（交换ncclUniqueId等）。如果本机开启了很多docker虚接口或 loopback 优先而其他线程还没起来，可能 NCCL 在尝试接口时超时重试。NCCL 默认排除 lo 和 docker\* 除非没其他接口[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=Note%3A%20By%20default%2C%20the%20loopback,interfaces%20matching%20the%20manual%20selection)。另一个原因是生成UniqueId采用全员通信，MPI或文件系统差导致慢。

- **排查步骤**：
  1. **日志观察**：开启 `NCCL_DEBUG=INFO`，看每个rank在初始化阶段的时间戳。如果卡很久，多半在`ncclCommInitRank`内部。INFO日志可能打印 “Trying to bootstrap via x.x.x.x” 之类，可发现如果选错接口。

  1. **指定接口**：设置 `NCCL_SOCKET_IFNAME=<eth_name>`，确保 NCCL 用正确的本地高速接口而非虚拟接口。

  1. **UniqueId交换**：PyTorch中默认使用TCP socket交换uniqueId，如果机器DNS不好或者需翻墙，会拖慢。可以尝试 `init_process_group(..., store=...)` 用本地文件或shared memory作为store，绕过DNS。NCCL 2.23+ 还提供 `NCCL_OOB_NET_ENABLE=1` 可以让引导也走NCCL网络插件而不是系统socket[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=NCCL_OOB_NET_ENABLE%EF%83%81)。但这需要配置，不是默认路径。

- **建议env组合**：
  - `NCCL_SOCKET_IFNAME=eth0 NCCL_IB_DISABLE=1`（单机无IB，也可禁IB插件让其别无选择用 socket）。

  - `NCCL_UID_RUNTIME_BINARY=1` (如果适用，理论上可以缩短uniqueId生成方式，不过这通常不是瓶颈).

- **验证修复**：调整后再次初始化，测量耗时。如果下降到<5秒，则说明确实接口选择或配置改善了。如仍慢，可以在profile中查看是否Python端store阻塞长，定位问题。

**场景8：XLA/TPU 等非常规场景下 NCCL 报错不支持**

- **现象**：使用 PyTorch XLA (GPU+TPU混合) 或 HPC上NVLink+IB混合拓扑时，NCCL 报一些不支持 CollNet/NVLS 之类的错误，或者Hang。

- **可能原因**：NCCL 某算法在当前硬件不适用但被错误启用。如 CollNet 需要服务器有独立网络分层，但混合场景无此条件，如果NCCL版本判断有误可能导致 hang。

- **排查步骤**：
  1. **禁用高级特性**：`NCCL_ALGO=^CollNet`，`NCCL_NVLS_ENABLE=0` 禁用 NVLink SHARP，`NCCL_PXN_DISABLE=1` 禁用PXN。基本回退到经典Ring/Tree。

  1. **查看issue**：搜索NVIDIA NCCL release notes或GitHub issue，有无针对TPU or multi-node NVSwitch的已知问题和补丁。

  1. **版本回退**：有时新特性Bug，可以尝试NCCL降级或升级到最新补丁看是否解决。

- **建议env组合**：保守期间对非典型架构统一加上述禁用的变量，确保NCCL仅用最稳妥路径（虽然可能性能不最高）。

- **验证修复**：让通信能跑通、结果正确，然后再逐一开放看性能提升与稳定性，找到平衡点。

> **注：**以上场景远非穷尽。实际排障中，要结合具体软硬件环境，对症下药。关键是遵循**先易后难、由广到细**的思路：先确保外围配置正确，然后利用 NCCL 提供的调试开关缩小可疑范围，并借助 nccl-tests 做对比实验验证猜想。每个变量改动都应记录效果，最终选择对性能和稳定性最佳的方案。

## H. 一页式 NCCL 调优与排障 Cheat Sheet

最后，将本文介绍的 NCCL 调试“工具箱”汇总成一页速查表，便于在实战中快速复制使用。

### **日志与诊断开关**

- **基础日志**：`NCCL_DEBUG=INFO` – 开启调试日志（版本、初始化细节、错误）[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=)。常用级别：WARN（默认、仅错误）、INFO（推荐）、TRACE（详细追踪，仅短时间使用）。

- **子模块过滤**：`NCCL_DEBUG_SUBSYS=INIT,COLL,...` – 聚焦特定子系统日志[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=The%20default%20value%20is%20INIT%2CBOOTSTRAP%2CENV)。默认 ENV/INIT 等，调网络问题常加 `NET,GRAPH`。

- **日志输出定向**：`NCCL_DEBUG_FILE=nccl_%h_%p.log` – 日志重定向到文件，以 hostname+PID 区分[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=NCCL_DEBUG_FILE%EF%83%81)。避免多进程stdout混杂。

- **时间戳**：`NCCL_DEBUG_TIMESTAMP_FORMAT="%H:%M:%S"` – 修改时间戳格式，或配合 `TZ` 环境变量调整时区[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=NCCL_DEBUG_TIMESTAMP_FORMAT%EF%83%81)。

- **线程命名**：`NCCL_SET_THREAD_NAME=1` – 让 NCCL 后台线程具名，便于 profiling[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=NCCL_SET_THREAD_NAME%EF%83%81)。

- **PyTorch 超时监控**：`TORCH_NCCL_BLOCKING_WAIT=1` – NCCL调用等待改为阻塞，超时抛异常，防止沉默hang[docs.pytorch.org](https://docs.pytorch.org/docs/stable/torch_nccl_environment_variables.html#:~:text=Control%20whether%20to%20use%20high,stream%20for%20the%20NCCL%20communicator)。

- **PyTorch 异常处理**：`TORCH_NCCL_ASYNC_ERROR_HANDLING=1` – 异步错误时自动中止全部进程[docs.pytorch.org](https://docs.pytorch.org/docs/stable/torch_nccl_environment_variables.html#:~:text=Control%20how%20we%20perform%20Async,it%20is%20set%20to%203)。（Pytorch<=1.11 用旧 env `NCCL_ASYNC_ERROR_HANDLING`).

- **PyTorch 超时Dump**：`TORCH_NCCL_DUMP_ON_TIMEOUT=1` + `TORCH_NCCL_TRACE_BUFFER_SIZE=1000000` – Watchdog超时时dump最近操作轨迹[docs.pytorch.org](https://docs.pytorch.org/docs/stable/torch_nccl_environment_variables.html#:~:text=)[docs.pytorch.org](https://docs.pytorch.org/docs/stable/torch_nccl_environment_variables.html#:~:text=)。Dump文件缺省 `/tmp/torch_nccl_<rank>_<pid>.log`，可用 `TORCH_NCCL_DEBUG_INFO_TEMP_FILE` 指定[docs.pytorch.org](https://docs.pytorch.org/docs/stable/torch_nccl_environment_variables.html#:~:text=we%20exit%20and%20throws%20timeout,exception)。

- **PyTorch 额外**：`TORCH_NCCL_DESYNC_DEBUG=1` – 发现collective不同步时提示[docs.pytorch.org](https://docs.pytorch.org/docs/stable/torch_nccl_environment_variables.html#:~:text=than%200)；`TORCH_NCCL_NAN_CHECK=1` – 每次collective后检查Nan[docs.pytorch.org](https://docs.pytorch.org/docs/stable/torch_nccl_environment_variables.html#:~:text=)。调试数据完整性用。

### **传输通道控制**

- **禁用直连P2P**：`NCCL_P2P_DISABLE=1` – 禁 NVLink/PCIe GPU直接通信，改经SHM/网络[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=NCCL_P2P_DISABLE%EF%83%81)。Hang排查用于隔离P2P因素。

- **限制直连级别**：`NCCL_P2P_LEVEL=NVL/PIX/...` – 控制多远的GPU间用直连[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=,always%20disabled)。如只想NVLink用P2P，其它走SHM，则设 `PIX`。

- **禁进程内直访**：`NCCL_P2P_DIRECT_DISABLE=1` – 同一进程内多GPU不直接访存，避免CUDA没有peer access导致hang[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=NCCL_P2P_DIRECT_DISABLE%EF%83%81)。

- **禁共享内存**：`NCCL_SHM_DISABLE=1` – 不用 /dev/shm 传输[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=NCCL_SHM_DISABLE%EF%83%81)。调试 SHM 空间不足或跨NUMA问题，可暂关。

- **禁IB/RoCE**：`NCCL_IB_DISABLE=1` – 禁用 InfiniBand/RoCE 网络，改用 TCP[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=NCCL_IB_DISABLE%EF%83%81)。用于确认IB相关问题（性能骤降则说明tcp接管）。

- **IB 网卡选择**：`NCCL_IB_HCA="^mlx5_2"` – 排除mlx5_2卡不用；`NCCL_IB_HCA=mlx5_0:1` – 只用mlx5_0的1号端口[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=,mlx5)。多HCA环境下调度使用。

- **指定网络接口**：`NCCL_SOCKET_IFNAME=eth0` – 强制用指定前缀接口 (eth0等)[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=,%E2%80%A6)；`^docker` 排除某类接口[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=%60,docker)。避免选错网络。

- **IPv4/v6**：`NCCL_SOCKET_FAMILY=AF_INET` – 强制用 IPv4[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=NCCL_SOCKET_FAMILY%EF%83%81)（有时避免v6解析问题）。

- **GPU直RDMA控制**：`NCCL_NET_GDR_LEVEL=PHB` – 仅NUMA内启用GPU直RDMA[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=The%20,the%20topographical%20cutoff%20for%20GpuDirect)[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=,will%20go%20through%20the%20CPU)。`LOC` 禁GPU直接发NIC，全走CPU内存（可debug GDR问题）[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=The%20,the%20topographical%20cutoff%20for%20GpuDirect)[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=,always%20disabled)。

- **PCIe RO**：`NCCL_IB_PCI_RELAXED_ORDERING=2` – 自动用Relaxed Ordering[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=)；`=0` 强制禁用（debug某些RO问题）。

- **IB自适应路由**：`NCCL_IB_ADAPTIVE_ROUTING=0` – 禁用AR[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=Enable%20the%20use%20of%20Adaptive,NCCL_IB_SL)。调试拥塞波动时可尝试。

- **共享Buffer**：`NCCL_NET_SHARED_BUFFERS=0` – 禁用共享内存池[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=NCCL_NET_SHARED_BUFFERS%EF%83%81)；`NCCL_NET_SHARED_COMMS=0` – 禁用PXN共享连接[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=NCCL_NET_SHARED_COMMS%EF%83%81)。极罕见情况使用（如怀疑内存池问题）。

### **算法与协议调整**

- **禁用LL128**：`NCCL_PROTO=^LL128` – 排除 LL128 协议[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=)。常用于疑似LL128相关bug时（PCIe平台本也默认无LL128）。

- **仅用简单协议**：`NCCL_PROTO=Simple` – 不使用LL/LL128，只用Simple协议。调试小消息性能时可对比LL。

- **算法限定**：`NCCL_ALGO=Ring` – 强制环算法[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=For%20example%2C%20%60NCCL_ALGO%3D,allreduce%20and%20ring%20for%20broadcast)；`NCCL_ALGO=^Tree` – 禁用树算法[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=For%20example%2C%20%60NCCL_ALGO%3D,allreduce%20and%20ring%20for%20broadcast)。定位某算法导致的性能或bug，可以尝试不同组合（Ring vs Tree vs CollNet）。

- **禁用CollNet/NVLS**：`NCCL_ALGO=^CollNet` / `NCCL_NVLS_ENABLE=0` – 关闭高阶聚合算法。防止在不支持配置上误启用导致问题[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=Enable%20the%20use%20of%20NVLink,The%20default%20value%20is%202)。

- **禁用PXN**：`NCCL_PXN_DISABLE=1` – 关闭PxN中继。复杂拓扑中简化调试。

- **限制通道数**：`NCCL_MAX_NCHANNELS=4` – 限制最多4个通道。某些GPU资源紧张场景可试降低并发通信数。

- **调整每线程 socket**：`NCCL_NSOCKS_PERTHREAD=4 NCCL_SOCKET_NTHREADS=4` – 增加并发连接数和线程数[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=For%20generic%20100G%20networks%2C%20this,NCCL_NSOCKS_PERTHREAD)[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=On%20AWS%2C%20the%20default%20value,the%20default%20value%20is%201)。这是**性能调优**选项，在确认稳定后可用于提升大带宽网络利用率（如4×100G NIC）。注意遵守乘积<=64限制。

### **实验排障矩阵模板**

在排障时，可采用以下**实验矩阵**逐项尝试，并记录现象变化：

| 调试手段             | 操作                                              | 预期效果/判断依据                                                                                                                                                                                                                               |
| -------------------- | ------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **禁用IB改TCP**      | `NCCL_IB_DISABLE=1`                               | **若问题消失**：指向IB相关（配置/驱动/FW问题）。                                                                                                                                                                                                |
| **禁用P2P直连**      | `NCCL_P2P_DISABLE=1`                              | **若问题消失**：GPU直连模块异常（NVLink/P2P Bug）。                                                                                                                                                                                             |
| **禁用LL128协议**    | `NCCL_PROTO=^LL128`                               | **若问题消失**：LL128协议bug或数据精度问题[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=The%20,will%20be%20allowed%20to%20use)。                                                                 |
| **改用Tree算法**     | `NCCL_ALGO=Tree` 或 `^Ring`                       | **若性能改善**：环拓扑瓶颈，树算法更优（或反之）。                                                                                                                                                                                              |
| **Socket线程并行**   | `NCCL_SOCKET_NTHREADS=4, NCCL_NSOCKS_PERTHREAD=4` | **若性能改善**：之前单线程未压满网络，可考虑保留。                                                                                                                                                                                              |
| **固定接口**         | `NCCL_SOCKET_IFNAME=<dev>`                        | **若初始化成功**：多网卡下原先选错接口导致握手失败。                                                                                                                                                                                            |
| **GPU直连级别**      | `NCCL_P2P_LEVEL=SYS` / `PIX` 等                   | **性能/稳定性变化**：确认跨CPU直连是否有问题。                                                                                                                                                                                                  |
| **禁用SHM**          | `NCCL_SHM_DISABLE=1`                              | **若初始化通过**：原问题来自 /dev/shm 受限[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting.html#:~:text=In%20particular%2C%20Docker%20containers%20default,the%20docker%20launch%20command%20line)。 |
| **Relaxed Ordering** | `NCCL_IB_PCI_RELAXED_ORDERING=0`                  | **若性能变化**：RO参数影响虚拟化环境中的IB性能。                                                                                                                                                                                                |
| **Adaptive Routing** | `NCCL_IB_ADAPTIVE_ROUTING=0`                      | **若抖动减少**：AR在网络中引发波动。                                                                                                                                                                                                            |

_注：每次仅改动一个变量，观察效果，避免多项变化难以定位原因。_

### **信息收集与版本检查**

- **版本**：确保所有节点 NCCL 版本一致（`NCCL_DEBUG=VERSION` 可打印版本[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=)）。注意 PyTorch 内置NCCL版本，可通过 `torch.cuda.nccl.version()` 获取。已知问题可在 \[NCCL Release Notes] 中查找修复。

- **驱动/CUDA**：CUDA Driver >= NCCL 要求版本，否则可能发生挂起（Release Notes 中通常注明）。尽量使用 NVIDIA 官方稳定的驱动+CUDA组合。

- **拓扑**：使用 `NCCL_TOPO_DUMP_FILE` 保存拓扑，对比实际硬件。检查 NVLink/NVSwitch 节点是否被正确识别；检查 PCI 域和 NIC 归属是否合理[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting.html#:~:text=NCCL%20relies%20on%20%2Fsys%20to,optimal%20performance)。

- **网络设置**：记录 ifconfig/ibstatus，确保所用接口UP状态正常。收集 `sysctl -a | grep mlnx` 等判断RoCE ECN/PFC配置。

- **错误日志**：保存所有 rank 的 NCCL WARN/ERROR 行，包含 error code 和rank信息，便于与NCCL源码/issues对照。

### **安全与性能提示**

- **不要长期保留调试变量**：如 `NCCL_*_LEVEL` 之类在问题解决后应恢复默认[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=There%20are%20two%20categories%20of,optimal%20behavior%2C%20crashes%2C%20or%20hangs)。调优类变量可加入作业配置，但**需有注释**说明理由，防止遗忘。

- **数据正确性**：禁用 `NCCL_CHECK_POINTERS` 可能提升性能，但切勿在开发调试时关闭安全检查[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=%28since%202)。同理，大多数调优选项在 throughput 和 determinism 间权衡，生产环境应充分验证不会引入数值差异。

- **关注官方指南**：NVIDIA 针对新硬件（如 Hopper NVLink4、双rail网络）会发布专门调优指南[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting.html#:~:text=,s%20%24%7BBDF%7D%20ECAP_ACS%2B0x6.w%3D0000%20done)。这些文档提供了**推荐参数**和**已知陷阱**（如 NVLS silent fallback hang 等[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=cannot%20be%20allocated)）。充分利用这些信息可事半功倍。

- **升级与回归**：NCCL 随新版本性能提升也可能带来新bug。建议在关键任务前做小规模A/B测试不同版本 NCCL，观察日志是否有异常warn，性能是否稳健，然后再推广升级。

---

通过以上方法和技巧，我们可以逐步掌握 **NCCL Debug 的“全栈手段”**，从环境变量调优到日志诊断、从协议算法选择到实际案例排查，在遇到 NCCL hang、性能瓶颈或数据异常时做到心中有数、手中有方。现代大规模分布式训练系统复杂多变，但相信凭借扎实的官方资料【1】【2】【3】和工程实践经验，我们能够将 NCCL 的行为透明化、问题可解化，为训练任务保驾护航。

**参考文献：**

- NVIDIA NCCL 官方文档 – _Environment Variables_[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=There%20are%20two%20categories%20of,optimal%20behavior%2C%20crashes%2C%20or%20hangs)[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#:~:text=Debugging%EF%83%81)、_Troubleshooting_[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting.html#:~:text=NCCL%20relies%20on%20%2Fsys%20to,optimal%20performance)[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting.html#:~:text=If%20insufficient%20shared%20memory%20is,a%20message%20similar%20to%20this)等章节

- PyTorch Distributed 官方文档 – _ProcessGroupNCCL Environment Variables_[docs.pytorch.org](https://docs.pytorch.org/docs/stable/torch_nccl_environment_variables.html#:~:text=)[docs.pytorch.org](https://docs.pytorch.org/docs/stable/torch_nccl_environment_variables.html#:~:text=)

- NVIDIA/nccl-tests 项目文档 – _PERFORMANCE.md_[forums.developer.nvidia.com](https://forums.developer.nvidia.com/t/what-is-the-busbw-in-nccl-tests/256858#:~:text=The%20published%20info%20on%20NCCL,p2pBandwidthLatencyTest)（算法带宽与总线带宽解释）

- NVIDIA Developer Forums – NCCL 性能与错误相关讨论[forums.developer.nvidia.com](https://forums.developer.nvidia.com/t/what-is-the-busbw-in-nccl-tests/256858#:~:text=The%20published%20info%20on%20NCCL,p2pBandwidthLatencyTest)[docs.nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting.html#:~:text=Errors%EF%83%81)

- **(经验总结)** 部分未特别标注引用的内容均来自作者实践与常见问题总结。
