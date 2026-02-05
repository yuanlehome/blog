---
title: "NCCL\_Debug 全栈手段：常用环境变量、日志/拓扑/通信诊断与 Hang /性能/数据异常排查"
slug: nccl-debug-hang
date: '2026-02-05'
tags: []
status: published
imported_at: '2026-02-05T05:29:57.228Z'
---

本文面向使用 PyTorch DDP / Megatron / DeepSpeed 或自研分布式训练框架的工程师，系统讲解 NCCL 调试的工具箱和环境变量设置方法，覆盖 NCCL hang、NCCL error、性能退化、跨机带宽不足、GDR/IB/NVLink 通信异常 等场景的诊断思路和解决方案。

## A. NCCL Debug 总览：可观测、可控制、可验证的方面

NCCL（NVIDIA Collectives Communications Library）提供了丰富的环境变量和工具，允许我们从多个层面进行调试：

- 可观测性（Observation）：通过 NCCL 日志了解内部状态、拓扑检测结果、算法/协议选择、所用网络通道（如 SHM、P2P、Socket、InfiniBand）等信息。例如设置 `NCCL_DEBUG=INFO` 可以打印 NCCL 版本和操作信息，`NCCL_DEBUG_SUBSYS` 允许聚焦特定子系统日志。这些日志有助于找到程序 Hang 的环节或性能瓶颈位置。

- 可控制性（Control）：NCCL 的众多环境变量可以强制/禁用某些行为，从而控制调度决策。例如，可以通过 `NCCL_PROTO` 限制协议（Simple/LL/LL128）选择，通过 `NCCL_ALGO` 限制算法（Ring/Tree/CollNet 等）选择，通过 `NCCL_IB_DISABLE`/`NCCL_SHM_DISABLE` 等开关切换不同传输方式。这些设置可以帮助我们验证某一机制是否导致了问题——如禁用某模块后问题消失，则该模块可能有关。

- 可验证性（Verification）：使用 nccl-tests 等基准工具对特定场景进行最小复现和对比实验。例如用 `all_reduce_perf` 测试不同消息大小、不同环境变量组合下的带宽，比较 Algorithm BW（算法带宽）和 Bus BW（总线带宽）来判断硬件通信是否跑满。通过对照矩阵试验，我们可以逐步缩小问题范围，并验证修改是否奏效。

总之，NCCL 调试涉及日志观察（看现象）、环境变量调整（做实验）和工具对照（下验证结论）三个环节，形成“复现→采集信息→缩小变量→定位原因→验证修复”的闭环流程。在正式进入各部分细节前，建议先收集如下关键信息，作为排障的基础数据：

> 📝 排障信息收集清单：NCCL 版本、CUDA Driver/Runtime 版本，PyTorch 等框架版本；GPU 型号和拓扑（NVLink/NVSwitch 结构，PCIe 代数），节点间网络类型（InfiniBand/RoCE 还是以太网）、带宽和布线（多 NIC？直连/交换机拓扑？）；当前系统的相关环境变量配置；容器/虚拟化设置（/sys 挂载、`--shm-size`、NUMA 等）；以及出问题时的具体日志片段、报错信息。

下面章节将按类别详细介绍 NCCL 的调试手段与参数。

## B. 日志与可观测性环境变量

调试 NCCL 问题的第一步，是启用充分的日志，以观察 NCCL 内部发生了什么。NCCL 提供以下环境变量用于控制日志级别和内容：

- `NCCL_DEBUG` 日志级别：可取 `WARN`, `INFO`, `TRACE` 等级别。`WARN` 只在发生错误时输出简要信息，`INFO` 会打印调试信息（如各步连接、算法选择），`TRACE` 则会对每次调用输出可重放的详细跟踪（大量日志，通常只在小规模测试时使用）。另外，`NCCL_DEBUG=VERSION` 可仅打印 NCCL 版本号用于确认版本。一般排查从 `INFO` 开始，在问题复杂或需要反馈 NVIDIA 时再用 `TRACE`。注意：过高日志级别可能显著拖慢程序，应在必要时短期使用。

- `NCCL_DEBUG_SUBSYS` 日志子系统过滤：当使用 `INFO`/`TRACE` 级别时，此变量可选定感兴趣的子系统，以减少无关输出。支持的子系统有 INIT（初始化）、COLL（集合通信算法）、P2P（点对点直连）、SHM（共享内存）、NET（网络传输）、GRAPH（拓扑检测/图搜索）、TUNING（算法/协议调优）、ENV（环境变量设置）、ALLOC（内存分配）、PROXY（Proxy 线程）、NVLS（NVLink SHARP）、BOOTSTRAP（进程间引导连接）、REG（注册内存）、PROFILE（粗粒度性能 profiling）、RAS（可靠性子系统）等，以及 ALL（全部）。默认的子系统列表是 INIT, BOOTSTRAP, ENV。例如：
- `NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=NET,GRAPH` 只看网络连接和拓扑相关日志。

- 使用前缀 `^` 可排除子模块，如 `NCCL_DEBUG_SUBSYS=ALL,^COLL` 表示记录全部但不含集合算法细节。

- `NCCL_DEBUG_FILE` 日志重定向：默认日志输出到 stdout/stderr。设置该变量可将日志写入文件。例如：\
  `NCCL_DEBUG=WARN NCCL_DEBUG_FILE=/tmp/nccl_log.%h.%p`\
  将 WARN 级日志写到文件，文件名中 `%h` 和 `%p` 会分别替换为 hostname 和进程 PID。这在多进程/多节点场景下很有用，每个进程写自己的日志文件，避免交织。需注意文件名必须唯一，否则多个进程写入同一文件会混乱。

- 时间戳格式与线程命名：`NCCL_DEBUG_TIMESTAMP_FORMAT` 可定制日志时间戳格式（例如打印相对时间方便计算耗时）。`NCCL_SET_THREAD_NAME=1` 则让 NCCL 后台线程有易读名称（如 `NCCL I/O Thr`），便于使用 `htop` 等工具观察 CPU 线程状态。

启用日志后，我们应该重点关注：(1) 每个进程是否输出了 NCCL 版本（以确认版本一致）；(2) 环境变量设置是否被正确读取。NCCL 在 INIT 阶段通常会打印所用环境变量值（需要 ENV 子系统日志）。例如日志可能包含“`NCCL_SOCKET_IFNAME set by environment to eth0`”等字样，确认调优参数已生效。

日志分析技巧：对于 Hang 卡住 的问题，INFO 级日志往往可以看到进程停在哪一步（比如所有日志停在 `... Launch mode Parallel...` 之后，则可能卡在 kernel launch，或者停在 `Connected all rings` 之前，说明有进程通信连接未完成）。这时可以：

- 将 INFO 细化为 TRACE 重跑短测试，查看详细的通信握手过程，找出最后的操作调用序列。

- 利用进程的 stack trace（如通过 gdb 或 PyTorch 自带的 `TORCH_SHOW_CPP_STACKTRACES=1`）来定位阻塞点函数调用。

而对于错误立即报错的情况，`WARN` 日志即可看到 NCCL 返回的错误类型。常见错误类型如：`ncclSystemError`（系统调用失败）、`ncclUnhandledCudaError`（CUDA 调用失败）、`ncclDevMismatch`（GPU 设备不一致）等。配合 NVIDIA 官方文档“Errors”章节，可以理解错误含义。在 PyTorch 中，如果开启 `TORCH_DISTRIBUTED_DEBUG=DETAIL`，遇到 NCCL 错误时 PyTorch 也会 dump 各 rank 的堆栈，辅助定位。

PyTorch 特有日志和超时 Dump：PyTorch 的 `ProcessGroupNCCL` 实现有一套 Watchdog 机制，可配合 NCCL 日志定位问题：

- 设置 `TORCH_CPP_LOG_LEVEL=INFO`（或 DEBUG）可以看到 PyTorch 内部关于 ProcessGroup 和 Watchdog 的日志。

- Watchdog 超时 Dump：环境变量 `TORCH_NCCL_DUMP_ON_TIMEOUT=1` 可以让当 NCCL 操作超时/异常时自动转储调试信息。需配合 `TORCH_NCCL_TRACE_BUFFER_SIZE` (如设为几百或几千) 来开启 NCCL 内部“航迹记录”环形缓冲。超时发生时，每个 rank 会将最近的 NCCL 调用事件（开始/结束时间，甚至可选带 C++ 调用栈）写入 `TORCH_NCCL_DEBUG_INFO_*` 文件。这对排查集体调用失同期（desync）或 Hang 特别有用——我们可以比对各 rank 最后完成的操作，推测是哪一个 rank 停滞。此外，`TORCH_NCCL_DESYNC_DEBUG=1` 也可用于打印可能发生不同步的提示信息。

日志级别策略：在性能问题排查时，长时间开启 TRACE 日志不现实，可以先 INFO 粗略看每轮是否进展正常，再用 nccl-tests 短跑 TRACE 查看细节。而稳定性问题（Hang/错误）倾向于用 INFO + PyTorch Dump 首先收集线索，然后根据需要放大某子系统日志或使用 TRACE 重现小场景。

总之，充分且合理过滤的日志是 NCCL Debug 的基础。下面章节将在此基础上，讨论如何通过拓扑信息和环境变量配置进一步定位问题。

## C. 拓扑与通信路径诊断

NCCL 在初始化时会探测硬件拓扑结构，包括 GPU 之间以及 GPU 与网络接口之间的连接关系，然后据此决定通信算法（如是否使用 NVLink）和路径选择。因此，排查跨设备通信的问题，往往需要弄清实际数据流经路径与 NCCL 认知的拓扑。常用方法如下：

- 拓扑文件与 Dump：NCCL 提供 `NCCL_TOPO_FILE` 和 `NCCL_TOPO_DUMP_FILE` 环境变量来加载或导出拓扑信息。
- `NCCL_TOPO_FILE=<path>`：指定一个 XML 文件，让 NCCL 在硬件探测前先加载此文件中描述的拓扑（如 PCIe 交换机结构、NVLink 布局等）。这常用于容器或虚拟化场景下，因为这些环境下 `/sys` 提供的拓扑可能是虚拟的。NCCL 默认会尝试加载 `/var/run/nvidia-topologyd/virtualTopology.xml`（如果存在），在某些 GPU 分区或 MIG 场景下这个文件由驱动生成，描述了真实拓扑。如果你怀疑 NCCL 读到了错误的拓扑（导致算法选择不佳），可让管理员提供正确拓扑文件并用此变量加载。

- `NCCL_TOPO_DUMP_FILE=<path>`：让 NCCL 在探测完拓扑后导出检测结果为 XML。这份文件可以用于进一步分析或者在另一环境重现。当遇到跨节点通信异常时，可收集每台节点的 dump 文件，比对差异。

- 查看日志中的拓扑检测：启用 `NCCL_DEBUG_SUBSYS=GRAPH`，NCCL 初始化时会打印拓扑相关信息，包括每块 GPU 的 CUDA 设备号、所属 PCIe 开关以及网络接口关联等。例如日志可能显示 NVLink 连接对、InfiniBand NIC 和 GPU 的归属关系等。这能帮助确认 NCCL 判断的拓扑是否符合预期。

- 判定通信走哪条通道：根据 NCCL 日志和系统信息，我们能推断实际使用了 NVLink、PCIe、SHM 还是网络：
- NVLink: 如果两 GPU 同机直连 NVLink，NCCL 通常使用 P2P 通道直接传输。日志 `NET/Plugin` 部分不会提及 socket 或 IB 连接。可用 CUDA 自带的 `p2pBandwidthLatencyTest` 工具验证 GPU 对间带宽是否达 NVLink 水平。NVLink 6 (H100) 双向理论带宽可达 50 GB/s+，NVSwitch 情况下 8 卡 AllReduce 总带宽甚至更高。

- PCIe: 非 NVLink 的同机 GPU 之间，则经 PCIe 或 QMPI。NCCL 日志通常会 fallback 到 SHM 或者 P2P (DMA) 通道，但速率受 PCIe 限制。通过 `nvbandwidth` 等工具可测 PCIe 对点带宽（如 PCIe3 x16 ~12 GB/s，PCIe4 x16 ~25 GB/s）。

- SHM (共享内存): 默认启用，用于同一主机跨 NUMA 的 GPU 间通信。当 P2P (直连) 因拓扑原因不可用时（例如不同 CPU 根连接的 GPU），NCCL 会先拷数据到系统内存再让目标 GPU 读回。如果 `NCCL_SHM_DISABLE=1` 则跳过 SHM 改走网络协议。可以通过对比开启/关闭 SHM 时性能变化来判断其作用：若关闭后同机不同 NUMA GPU 带宽骤降甚至类似网络水平，则原本用了 SHM。

- InfiniBand/RoCE: 跨节点主要依赖 IB/RoCE 网络。日志在初始化阶段会打印诸如 “`Using xx:xx:xx (InfiniBand)`” 或者 “`NCCL NET/IB: No device found, fallback to Socket`” 等。若 IB 正常，NCCL 会使用 GPU Direct RDMA (GDR) 直达 NIC；否则可能走 CPU（bounce buffers）。`NCCL_NET_GDR_LEVEL` 环境变量可以控制 GDR 使用距离阈值（如限制只有 NIC 与 GPU 在同一 PCIe 开关才用 GDR）。如怀疑 GDR 有问题，可尝试 `NCCL_NET_GDR_LEVEL=LOC` 完全禁用直 RDMA，观察性能或稳定性是否变化。

- Socket (TCP): 当 IB 不可用或被禁用时，NCCL 会回退到 TCP/socket。日志会出现 `NCCL Net: Using Socket` 字样。这通常性能较差（几十 Gb/s 级别），但有助于隔离 IB 问题——如 IB 硬件有问题，用 socket 反而不 hang，则进一步指向 IB 配置故障。

- 跨网卡/多通道判断：在多 NIC 系统（如每台服务器有 dual-port IB）上，NCCL 默认尝试同一环上的节点用相同编号 NIC 通信（避免 Rail 间干扰）。可以通过设置 `NCCL_CROSS_NIC=1` 强制允许环在不同 NIC 交叉（适合单交换机扇出网络），或 `NCCL_CROSS_NIC=0` 固定不交叉（适合双网双 Rail 架构）。若怀疑 NCCL 没有充分利用多 NIC，可调整此值并用日志验证每环使用的接口变化。此外，`NCCL_IB_MERGE_NICS` 控制是否把双端口 NIC 当作单逻辑设备聚合带宽（默认启用）。如果启用却性能异常波动，尝试设 `NCCL_IB_MERGE_NICS=0` 拆分使用看看区别。

典型拓扑问题案例：有时容器中的 `/sys` 只暴露虚拟 PCI 拓扑，导致 NCCL 误判。例如某 8 卡机器实际有 NVSwitch，全机互联 120 GB/s，但容器里 /sys 不全，NCCL 未检测 NVLink，导致只用 PCIe 带宽（总线带宽仅 12 GB/s 左右）。对此我们看到 Bus BW 明显低于硬件应有水平，日志里 Graph 拓扑只列出 PCI 路径而无 NVLink。解决办法是确保挂载正确的 `/sys` 进去或使用 `NCCL_TOPO_FILE` 提供真实拓扑。另外在 VM 中，PCIe ACS 机制可能强制所有 P2P 走 CPU 根复杂交换，从而性能和稳定性降低甚至 Hang。NCCL 文档建议裸机禁用 ACS 或 VM 环境下打开 NIC 的 ATS 支持。

总之，拓扑和路径决定了 NCCL 算法的基础。通过日志和工具确认实际的数据路径，我们才能有针对性地调整相关环境变量，见下一节。

## D. 传输层开关与网络相关环境变量

NCCL 支持多种通信传输方式，包括：GPU 直连（P2P）、共享内存（SHM）、TCP Socket、InfiniBand Verbs 等。其行为可由一系列环境变量控制。下文按类别列出常用的网络/传输相关环境变量，以及它们的作用和典型用途（除非特别说明，均参考官方文档等）：

### InfiniBand/RoCE 相关:

- 设备选择：`NCCL_IB_HCA` – 指定哪几个 HCA（IB 主机通道适配器）用于 NCCL 通信。可用格式如：`NCCL_IB_HCA=mlx5_0:1,mlx5_1:1`（精确指定两个卡的 1 号端口）；或 `^=mlx5_3`（排除特定卡）等。默认情况下，NCCL 会自动选择所有可用 IB 设备，优先同名端口。但在多 IB 网卡且某些用于其他用途时，常通过此变量限制 NCCL 用某些端口。有上限 32 个 HCA 设备。

- 连接超时与重试：`NCCL_IB_TIMEOUT` – 控制 IB Verbs 的超时时间，影响 QP 连接和数据超时。缺省值 20，对应 4.096 µs \* 2^20 ≈ 4 秒的链路层超时。大规模集群上可能需要增大（如 NCCL 初始化报 `ibv_poll_cq error 12` 则尝试调大此值）。`NCCL_IB_RETRY_CNT` 控制 IB 层重试次数，默认 7 次（对应 InfiniBand spec 默认）。一般保留默认，除非特别需要避免过早断开。

- RoCE 定位：`NCCL_IB_GID_INDEX` – 指定 RoCE 情况下使用的 GID 表索引。RoCE v2 常用 index=3 (对应 IPv4) 或 index=0 (根据配置)，如遇跨网段通信问题可以尝试设置正确的 GID index。`NCCL_IB_ROCE_VERSION_NUM` – 指定 RoCE 版本 (1 或 2)，默认 2。`NCCL_IB_SL` 和 `NCCL_IB_TC` – 分别设置 IB Service Level 和 Traffic Class，用于 QoS 优先级，默认都为 0。在拥塞场景下，可考虑给控制报文和数据报文设不同 TC（2.22.3 加入 `NCCL_IB_FIFO_TC` 专门为控制信道设 TC）。

- IB 上的 GPU Direct 开关：早期变量 `NCCL_IB_CUDA_SUPPORT`（2.4.0 前）用于强制或禁用 GPU Direct RDMA。2.4.0 后改为 `NCCL_NET_GDR_LEVEL` 等统一控制。当前：
- `NCCL_NET_GDR_LEVEL` – 控制 NIC 与 GPU 间直连 RDMA 的拓扑距离阈值。可取 `LOC/PIX/PXB/PHB/SYS`（同 P2P_LEVEL 含义但针对 NIC-GPU）。默认 NCCL 会自动选。例如在 CPU 直连 NIC (PHB) 的系统上，如不想用 GPU 直接读写 NIC 内存，可设 `LOC` 禁用 GDR。反之强制 GDR 则可设 `SYS`（始终开）。调试场景：怀疑 GDR DMA-BUF 模式有问题，可暂时降级为 CPU 中转，通过设 `NCCL_NET_GDR_LEVEL=LOC` 来验证性能/稳定性变化。

- `NCCL_NET_GDR_READ` – 控制发送数据时是否用 GDR Read（NIC 从 GPU 内存直接读）。2.4.2 起对 NVLink 平台默认开启（=1），PCIe 平台默认 0，因为某些 PCIe 上 GPU->NIC 直读反而略慢。如果遇到奇怪的性能下降，可尝试切换这个值，看是否 GPU->CPU 拷贝阶段出了问题。

- `NCCL_NET_GDR_C2C` – (since 2.26) 针对 CPU 直连 NIC 且 CPU 经 C2C (比如 UPI) 连接 GPU 的场景，是否仍然启用 GDR。默认 2.27 起 =1 启用。若平台不支持可能需设 0 禁用。

- PCIe Relaxed Ordering (RO)：`NCCL_IB_PCI_RELAXED_ORDERING` – 控制 IBverb 传输是否启用 PCIe Relaxed Ordering。RO 能显著提高虚拟化环境下 IB 带宽。默认=2（自动检测 RO 支持则用）。如果在 VMware/Hyper-V 等 VM 里性能低，检查是否 RO 生效，可尝试手动设=1 强制开启（需要底层支持，不支持会报错）。另一方面，某些平台 RO 不稳定，可以=0 禁用。

- Adaptive Routing (AR)：`NCCL_IB_ADAPTIVE_ROUTING` – 控制是否启用 IB 网络的 AR 特性。在大型 Clos 网络中 AR 可改善拥塞下性能。NCCL 对原生 IB 默认启用 (=1)，RoCE 默认关 (=0)。如遇 IB 交换机有 AR bug，可设 0 禁用以验证。

- ECE (增强连接建立)：`NCCL_IB_ECE_ENABLE` – (2.23+) 控制是否使用 IB 增强连接建立机制以支持拥塞控制等特性。默认 2.19 起=1 开启。配置不当时 ECE 可能降低性能。若怀疑，可设 0 禁用比较。

以上 IB/RoCE 参数很多是系统级调优，不建议轻易改动。但在以下情况下值得关注：(a) RoCE 训练出现掉包或者无法通信——检查 GID 和 RoCE v2 设置；(b) VM 或直通 IB 时性能不及裸机——考虑 Relaxed Ordering 是否启用；(c) IB 网络大规模时不稳定——可能试试关掉 AR/ECE 测试稳定性。

### Socket/TCP 相关:

- 接口选择：`NCCL_SOCKET_IFNAME` – 指定 NCCL 使用的网络接口名前缀。缺省下，NCCL 自动选择具有最高带宽/最低延迟的接口（优先 ib 开头接口）。但自动选择可能错误，比如多网卡环境或 Docker 虚接口。通过设此变量可以强制使用特定网卡或排除某些网卡：如 `NCCL_SOCKET_IFNAME=eth0` 只用 eth0，`NCCL_SOCKET_IFNAME=^docker,lo` 排除 docker\* 和回环。。应用场景：多网络环境下防止 NCCL 选错（比如管理网和 RDMA 网都存在），明确限定接口能避免建立连接超时。

- 协议族：`NCCL_SOCKET_FAMILY` – 强制使用 IPv4 或 IPv6 接口。可设 `AF_INET` 或 `AF_INET6`。默认情况下，NCCL 会根据接口自动决定。如果遇到 v6 网络问题或名称解析问题，可尝试显式指定。

- 端口重试：`NCCL_SOCKET_RETRY_CNT` / `NCCL_SOCKET_RETRY_SLEEP_MSEC` – 控制 TCP 连接重试次数和间隔（2.24+）。默认重试 34 次，每次等待递增，累计约 60 秒。如果集群初始化时经常因为端口碰撞或连接临时失败，可以增大重试次数或间隔以提高成功率。

- 线程与并发：NCCL Socket 传输采用多线程模型，每条连接可用多个线程和 socket 并行传输以提升带宽：
- `NCCL_SOCKET_NTHREADS` – 每个网络连接使用的 CPU 线程数。默认云环境 AWS=2, GCP gVNIC=4, 其它=1。可调范围 1-16，但需注意 `NCCL_SOCKET_NTHREADS * NCCL_NSOCKS_PERTHREAD <= 64`。在 100Gb 以上网络，可考虑手动设 4 线程以提升利用率。副作用：线程越多 CPU 占用越高，甚至抢占训练线程。

- `NCCL_NSOCKS_PERTHREAD` – 每线程打开的 TCP 套接字数。AWS 默认 8，其它默认 1。如果单连接速度有限（如单 TCP 流跑不满带宽），可以每线程开多个 socket 并行发送。同样乘积受限 64。

这两个参数对多节点大带宽 AllReduce 性能影响明显。例如在单机 4x100 Gb 网络的 DGX A100 上，默认配置可能只能到 ~80 Gb/s，需要增大线程和 sockets 并行度才能接近理论带宽。但要小心调优需在确保通信稳定基础上进行。

- 跨 Socket 优化：`NCCL_NET_SHARED_BUFFERS` – 控制是否启用共享缓冲来避免每对连接单独申请内存。默认 1 启用，通常不需改。`NCCL_NET_SHARED_COMMS` – 控制 PXN 场景下是否复用连接（2.12+，默认 1）。除非遇到特殊 Bug，否则很少调整。

### GPU 直连 (P2P) 与 SHM 相关:

- `NCCL_P2P_LEVEL` – 控制 GPU 间直连 P2P 的最大拓扑距离。可选：\
  `LOC`（同板直连才用 P2P），`NVL`（有 NVLink 则用），`PIX`（同 PCIe 开关用），`PXB`（跨 PCI 开关但同 CPU 用），`PHB`（同 NUMA 节点用，即跨 CPU 但不跨 QPI），`SYS`（即使跨 QPI/UPI 的 NUMA 也用 P2P）。默认为 NCCL 自动判断。用途：若某拓扑层次的 P2P 性能不佳甚至出错，可通过降低此级别迫使走其它通道。例如某虚拟化下 NVLink 不可用却错误标识，可设 `PIX` 让远端 NVLink 不被采用。

- `NCCL_P2P_DISABLE` – 完全禁用 GPU Direct P2P 通信。设为 1 后，同机 GPU 间将不走直连（无论 NVLink/PCIe），而统一经 SHM 或网络。调试：如果怀疑某些 P2P 通信导致 hang（如已知 NVLink 某驱动 Bug），可关掉验证。如果禁用后问题消失，则可以进一步细分（例如用 NCCL_P2P_LEVEL 控制不用 NVLink 但仍允许同 PCIe 直连）。

- `NCCL_P2P_DIRECT_DISABLE` – 禁用进程内的直接 P2P 访问。NCCL 对于同一进程内多 GPU，本可直接读写彼此显存。如果应用使用了不能共享 Peer Memory 的 allocator，此模式可能失败。设 1 可强制改用更安全的路径拷贝，避免 hang。

- `NCCL_SHM_DISABLE` – 禁用共享内存传输。设 1 则不同进程间即使在同节点也不使用 /dev/shm 交换，而是退化为网络。调试用途：怀疑 /dev/shm 空间不足（初始化报错）或 SHM 通信异常时，可以关掉让 NCCL 走网络，看能否避开问题。但性能会受影响，应尽快恢复 SHM 并解决根本问题（例如增大 Docker 的 `--shm-size`、设置 `ulimit -l unlimited` 允许内存锁定等）。

### 其他通用配置:

- Buffer 大小：`NCCL_BUFFSIZE` – 每个通道使用的 buffer 大小，默认 4MiB。调小可降低内存占用、缓解 OOM（代价是可能降速，因为分片变小）；调大在特定网络上可能提升长消息带宽。通常以 2 的幂为佳。

- 线程数：`NCCL_NTHREADS` – 每个 CUDA 区块的线程数。默认新 GPU=512 线程。可设 64/128/256/512。当 GPU 核心频率很低时，多线程可能提高 pipeline 并行度，但也增大每 block 资源占用。一般无需修改，除非定位到 GPU 核心闲置才尝试。

- 通道数：NCCL 使用多条“通道”（channel）并行通信，对应多个 CUDA block：
- `NCCL_MIN_NCHANNELS` / `NCCL_MAX_NCHANNELS` – 限制最少/最多通道数。旧版本叫 NRINGS。这影响 GPU 参与通信的 block 数。增加 channels 有助于提升大量小消息的重叠效率，但过多会争夺 GPU 计算资源。NCCL 2.5 起推荐通过更细粒度的 `NCCL_MIN_CTAS`/`NCCL_MAX_CTAS` 控制每 SM 并发 CTA 数量。通常除非做性能优化，不建议显式修改这些。

- Check 校验：`NCCL_CHECKS_DISABLE`（已废弃）– 关闭参数合法性检查，可略微降低延迟。2.2.12 后改用 `NCCL_CHECK_POINTERS` 控制是否检查 CUDA 指针有效性。默认关闭检查以提高性能，除非调试内存问题不需要打开。

以上设置很多仅在特定排障或调优场景使用，不宜长期在生产中开启。一般原则：逐步尝试粒度尽可能小的干预（如先禁用怀疑模块，再细化），以免引入新的不确定因素。

## E. 算法与协议相关调试手段

NCCL 针对不同规模和拓扑，会在 Ring、Tree、CollNet 等多种算法，以及 Simple、LL、LL128 等多种通信协议之间自动选择。某些 bug 或性能问题可能与算法/协议选择有关。因此 NCCL 提供环境变量来强制或排除特定算法/协议，从而帮助我们诊断。

- 协议选择 (`NCCL_PROTO`): 控制允许使用的消息传输协议，包括 Simple（分段复制，适用于大消息高带宽）、LL（Low Latency，适用于小消息低延迟）、LL128（优化长消息的小延迟算法，需要硬件支持）。用法为列出协议或以 `^` 列出排除协议。默认行为：支持 LL128 的平台开启全部三种，否则 LL128 不用。重要提示：NVIDIA 明确指出，不要随意启用 LL128 在不支持的平台，否则可能导致数据错误。LL128 一般要求 NVLink 拓扑良好的平台（如 DGX），在 PCIe 集群上 NCCL 默认已禁用 LL128。调试中，禁用 LL128 是常用手段：不少 NCCL 已知 Bug（比如 2.8 版本 Collnet 算法配合 LL128 在部分拓扑上出错）可以通过 `NCCL_PROTO=^LL128` 规避。如果问题消失，可据此怀疑 LL128 实现问题然后查找对应补丁或升级 NCCL 版本。

- 算法选择 (`NCCL_ALGO`): 控制集合通信算法，如 Ring、Tree、CollNet 等。2.24+版本支持更复杂的配置语法，可按操作类型分别指定算法列表或排除。例如：\
  `NCCL_ALGO=Ring` 强制全部用环形算法；\
  `NCCL_ALGO=^Tree` 禁用树算法（如怀疑 Tree 实现有 Bug，NCCL 会自动 fallback 环算法）；\
  `NCCL_ALGO="allreduce:tree,ring"` 仅 AllReduce 用树或环，其它操作不变。\
  默认 NCCL 会根据节点拓扑和消息大小自动混用多种算法，避免盲目固定导致性能下降。然而调试时，当某算法路径怀疑有问题，可以用排除法验证。例如树形算法在跨机时延较大，可以暂禁 Tree 看性能是否提升，从而确认是否需要调整树算法触发阈值（老版本通过 NCCL_TREE_THRESHOLD 控制消息大小阈值）。又如 CollNet 算法（要求特殊网络硬件）在不支持场景下应该自动不用，但如怀疑错误触发，可直接 `^CollNet`。

- 链路聚合算法 (NVLS/Multi-NIC 等)：新版本 NCCL 针对 NVSwitch 平台引入 NVLS（NVLink SHARP）算法，以及 MNNVL（跨节点 NVLink）支持等。环境变量如 `NCCL_NVLS_ENABLE` 控制 NVLS 开/关（默认 2=自动），`NCCL_MNNVL_ENABLE` 控制多节点 NVLink。这些一般 NCCL 默认自动处理。如果遇到 NVLS 资源分配失败引起 hang（2.27 版一度出现 silent fallback hang 问题），可以临时 `NCCL_NVLS_ENABLE=0` 来禁用 NVLS 验证是否问题消失，然后升级新版修复。

- PXN 机制（通信基于中间 GPU 转发）：变量 `NCCL_PXN_DISABLE` (2.12+) 禁用跨节点 NVLink 转发，`NCCL_P2P_PXN_LEVEL` 控制何种情况下使用 PXN，以及 `NCCL_PXN_C2C` 控制 C2C 互联时 PXN 是否可用。这些属于高级优化，一般无需手调。但在 NVSwitch + IB 的架构中，如果观察到某些 GPU 流量绕远了，可以看看 PXN 相关配置是否合理。例如默认 `NCCL_P2P_PXN_LEVEL=2` 总是用 PXN，有时可能导致不必要的中转占用 NVLink，调为 1 或 0 可做比较。

算法/协议排查思路：当怀疑 NCCL 内部选择不佳时，可以依次排除：先禁 CollNet/NVLS（这些依赖特殊硬件，禁用不影响常规 Ring/Tree 运行）；再禁 Tree 观察（尤其大批节点场景，tree 深度大时易受网络延迟影响）；最后再考虑禁 Ring（一般不需要，因为 NCCL 总会留至少 Ring 保证 functional）。协议方面则首选禁 LL128 试验，其次 LL vs Simple 切换对比小消息性能和稳定性。需要注意的是，这些变量仅用于临时诊断，生产环境遇到相关问题最好升级 NCCL 或调整代码，让 NCCL 自动策略生效，而非长期强制某算法——正如官方文档所警告的，强制算法会“prevent NCCL from selecting the best setting... cause performance problems or even break functionality”。

## F. 稳定性与容错：Hang/超时/错误处理

大规模分布式训练，除了性能，还必须关注稳定性。NCCL 在 2.20+版本逐步增强了容错和诊断能力，包括引入 RAS 子系统（Reliability, Availability, Serviceability）和结合框架的 Watchdog 机制。以下是相关工具和环境变量：

### NCCL 异常处理与 RAS：

- 异步错误监测：NCCL 内部如果检测到严重异步错误（如网络掉线、GPU 故障）会尝试使通信停止并返回错误。2.23 引入 `NCCL_IB_RETURN_ASYNC_EVENTS`（默认 1）控制 IB 异步事件处理。设为 0 则忽略 IB 驱动的异步错误，仅靠超时。这在某些调试下有用（例如允许程序在错误发生后一段时间继续运行，便于收集状态），但一般保持默认即可。

- NCCL RAS 子系统：从 NCCL 2.24 起，可以通过 RAS 接口查询 NCCL communicator 的运行状态，实现外部监控。相关变量：
- `NCCL_RAS_ENABLE` – 开启 RAS 功能（默认 1 启用）。如不需要可设 0 完全关闭。

- `NCCL_RAS_ADDR` – 指定 RAS 服务监听的 `<ip>:<port>`。默认 `localhost:28028`。在多用户节点上，每个作业应设不同端口避免冲突。

- `NCCL_RAS_TIMEOUT_FACTOR` – RAS 内部各种超时的倍率。RAS 会周期性检查通信进展，默认有 5~60 秒不等的超时。如程序被调试器挂起导致超时，可临时把 factor 设大避免误判。

开启后，可使用 NCCL 提供的 `ncclras` CLI 工具连接 RAS 端口查询状态（如有哪些 Collective 在进行，是否卡住）。这在 Hang 未超时时特别有价值，可以辅助判断是哪一步停滞。不过 RAS 属新特性，目前主要用于 NVIDIA 内部监控和高级用户。

- Abort 行为：NCCL 默认在检测到无法恢复的错误时会调用 `ncclCommAbort` 终止 communicator（而不是安静 Hang）。在较新版本，NCCL abort 会打印更详细的上下文信息。用户无须配置此功能，但要确保捕获并处理返回的 ncclResult_t 错误码。

### PyTorch ProcessGroupNCCL 容错设置：

PyTorch 自己也提供了环境变量来控制 NCCL 后端的错误处理和超时机制：

- Watchdog 线程 & 阻塞等待：默认情况下，PyTorch 每个进程启动一个 Watchdog 线程监视 NCCL 操作是否卡住。当某 GPU 卡住时，Watchdog 会在一定时间后使所有进程报错退出。可以通过 `torch.distributed.init_process_group(timeout=...)` 设置超时时间（默认一般 30 min）。以下环境变量可调整此行为：
- `TORCH_NCCL_BLOCKING_WAIT` – 设为 `1` 则使得 `dist.all_reduce(...).wait()` 等待调用变为阻塞模式。即发生超时时，会抛出异常而不是静默等待。建议在调试时开启，以便及时捕获 Hang 而不是无限挂住进程。

- `TORCH_NCCL_ASYNC_ERROR_HANDLING` – 控制异步错误处理策略。默认 `3`，表示一旦超时，所有进程一起安全退出（由主进程决定不用先 abort communicator，就直接退出）。选项说明：0=不处理异步错误（可能导致 hang 住不退出）；1=检测到错误后调用 NCCL Comm.abort 并 kill 进程；2=仅 abort communicator 但不杀进程；3=直接杀进程不做 abort。调试中推荐用默认 3 或选 1。设 0 则可能某些 rank 卡死无法退出。

- 实用组合：`TORCH_NCCL_BLOCKING_WAIT=1` + `NCCL_DEBUG=WARN` 是 PyTorch 官方建议用于 debug hang 的设置，可让在超时发生时抛异常并打印 NCCL 错误日志。

- 超时信息收集：前述 `TORCH_NCCL_DUMP_ON_TIMEOUT=1` 配合 Trace Buffer，可以在 Watchdog 认定超时时，自动收集调试信息。另外还有：
- `TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC` – Watchdog 心跳检测的周期，默认约 5 s。`TORCH_NCCL_ENABLE_MONITORING=1` 时，PyTorch 会再启一个监控线程，如果发现 Watchdog 本身卡死（可能因为死锁），则在此时间后强制 kill 进程。一般不需改这个值，除非调试环境下希望更快触发监控。

- `TORCH_NCCL_COORD_CHECK_MS` / `TORCH_NCCL_WAIT_TIMEOUT_DUMP_MS` – 这些控制多个 rank 协调 dump 的时序和等待时间。除非深入分析，否则用默认即可（1000 ms 间隔，额外等待同样长收集完 dump）。

- 数据检查：`TORCH_NCCL_NAN_CHECK=1` 可在每次 collective 调用时对张量进行 NaN/Inf 检查。发现 NaN 会报错退出，防止带着坏数据进行 AllReduce。这在怀疑 NCCL 数据腐蚀或上层算子问题时有帮助。但注意性能损耗较大，仅调试暂时开启。

通过以上机制，PyTorch 尽量做到某进程出错，整体及时退出，防止集群资源长时间被挂住进程占用。调试过程中，充分利用这些设置能缩短排查周期：与其等待默认 30 分钟超时，不如设置短超时并开启 Dump，快速拿到信息。

经验：排查 NCCL hang，应尽量在出错时刻就收集信息，而非等作业被迫杀死后再分析。Watchdog+Dump 提供了这样的契机。但另一方面，要防止误触发，比如调优时可能 AllReduce 本身就需要较长时间，此时可暂时调大 `timeout` 以免误判。

---

以上介绍了 NCCL Debug 的各项“武器”。接下来我们将它们应用到具体的故障场景中。

## G. 常见故障场景手册（10+案例）

本节按典型现象列举多种 NCCL 故障场景，分析可能原因并给出优先级渐进的排查步骤、建议的环境变量设置组合，以及如何用 nccl-tests 等工具复现验证。

场景 1：训练开始时 NCCL 初始化 Hang

- 现象：分布式作业启动后打印 NCCL 版本号，但一直卡在 communicator 初始化，既无 error 也无进展。可能所有进程都挂在 `ncclCommInitRank`。

- 可能原因：跨节点通信握手不通。常见包括：防火墙未关闭导致 TCP/IB 端口无法建立；节点间网络配置不一致（如一台走 IB 一台却无 IB）；`init_process_group` 参数 world_size 等不匹配；或 IB 的 GID 配置导致握手包丢弃。

- 排查步骤：

1.  基础连通性：确认各节点间彼此能 ping 通，并且没有防火墙阻挡 NCCL 默认使用的端口 (NCCL 默认随机挑选高位端口，可通过 `net.ipv4.ip_local_port_range` 调整范围)。对使用 IB/RoCE 的，检查 `ibstat` 状态、子网管理器（Subnet Manager）正常。

2.  接口选择：在环境中显式 `NCCL_DEBUG=INFO` 看日志哪个接口在尝试连接。若看到 fallback 到 Socket 或 `[0] NET/IB: No device found` 则 IB 未被识别。可以尝试设置 `NCCL_SOCKET_IFNAME` 明确指定正确的网络，例如 `NCCL_SOCKET_IFNAME=^eth,ib0`（排除无关接口）。

3.  禁用 IB 验证：若怀疑 IB 配置问题，临时 `NCCL_IB_DISABLE=1` 强制走 TCP。如果这样就能初始化成功（尽管后续 AllReduce 慢），说明 IB 通信有问题。接下来重点检查 RoCE 配置（例如 `NCCL_IB_GID_INDEX` 是否一致）以及 IB 固件/驱动。

4.  分步缩小：编写一个最小复现脚本，例如使用 nccl-tests：\
    `mpirun -np 2 -H host1:1,host2:1./build/all_reduce_perf -b 8 -e 8M -f 2`\
    尝试在两节点上跑简单 AllReduce，看能否 Hang 复现。加上 `NCCL_DEBUG=INFO` 捕获在哪一步挂。

- 建议 env 组合：
- _保守调试_：`NCCL_DEBUG=INFO NCCL_SOCKET_IFNAME=<iface>` 用于观察和纠偏。

- _激进尝试_：`NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=<iface>` 验证是否 IB 专有问题；若确认为 IB 问题，进一步 `NCCL_IB_GID_INDEX` 等配置比对两端。

- 验证修复：在确认网络配置无误后（如关闭防火墙或正确设置 RoCE PFC 等），重新打开 IB 跑 nccl-tests 验证 AllReduce 成功、带宽正常。

场景 2：训练中途某一步挂死（没有显式 error）

- 现象：训练运行一段时间后，所有 GPU 利用率掉为 0，进程无响应但未退出。可能日志停在某次 collective 操作前后，没有错误提示。

- 可能原因：这通常是 Collective 调用失去同步（Desynchronization）造成的死锁。可能一个 rank 跳过或提前退出导致其余 rank 卡在对应的 AllReduce/AllGather。也可能某 rank 上发生了 CUDA 错误被吞掉，导致 NCCL 等待永远不返回。NCCL 本身 Bug（比如 2.7.x 曾有 LL128 算法在特定拓扑卡死的问题）也可能导致所有 rank hang。

- 排查步骤：

1.  判断哪种 Hang：首先区分是所有 rank 都在等（典型集体不同步），还是个别 rank 崩溃导致 others 在等。可以通过 `dmesg` 查看是否有 GPU 异常日志（如 kernel 打印 Xid 错误表示某 rank GPU 出问题），也可使用 PyTorch 的 `TORCH_NCCL_BLOCKING_WAIT=1` 让出问题 rank 抛异常而不是静默挂住。

2.  Desync Debug：设置 `TORCH_NCCL_DUMP_ON_TIMEOUT=1` 并将超时设短（例如 5 分钟）来触发超时 dump。同时开 `TORCH_NCCL_DESYNC_DEBUG=1` 以帮助发现不同步信息。超时后检查每个 rank 转储的 trace，找出哪个 rank 在某 collective 上没有进入或没有退出。比如可能 rank7 停在 allreduce(stream X) 未调用，而其他都完成，则说明 rank7 代码有分支漏调。

3.  协议算法角度：如果所有 rank 显示都进入了一次 AllReduce 但出不来，考虑是否 NCCL 内部死锁。这种情况下可尝试 `NCCL_PROTO=^LL128` 或 `NCCL_ALGO=Ring` 等（逐一改变），看问题是否不再复现。如果禁用 LL128 后不 hang 了，则很可能碰到 NCCL 已知 Bug，需要升级 NCCL 版本。

4.  外部介入：利用 `gdb` attach 到挂住的一个进程，打印堆栈。如果看到某 NCCL kernel 卡在 CUDA sync，可能 CUDA 这端有异常（如非法内存访问未报）。这时设置环境 `CUDA_LAUNCH_BLOCKING=1` 重运行一次，方便让 CUDA 错误暴露。

- 建议 env 组合：
- _配合监控_：`TORCH_NCCL_BLOCKING_WAIT=1 TORCH_NCCL_ASYNC_ERROR_HANDLING=1` 使任何 rank 出错立刻中止所有进程，防止部分 hang。

- _Dump 信息_：`TORCH_NCCL_DUMP_ON_TIMEOUT=1 TORCH_NCCL_TRACE_BUFFER_SIZE=1000000 TORCH_NCCL_DEBUG_INFO_TEMP_FILE=/tmp/nccl_dump_%h_%p.json` 收集大量调用踪迹。一旦触发，可用工具/脚本汇总对比各 rank 日志。

- _隔离 NCCL 问题_：`NCCL_PROTO=^LL128` 试排除协议因素；`NCCL_ALGO=Ring` 固定算法验证。

- 验证修复：找到根因后采取相应措施。例如如果是应用代码漏调 collective，要修复逻辑。如果是 NCCL bug，则升级到官方修复版本或继续使用工作区（如禁用 LL128 作为 workaround）。最终在修复版本环境下长时间跑验证 Hang 不再发生。

场景 3：AllReduce 性能严重低于理论带宽

- 现象：8 卡单机 A100，预期 NVSwitch 可达 240 GB/s，但实际 all_reduce_perf 只得到 80 GB/s 算术带宽 (algbw)，busbw 约 80 GB/s。或多机时总带宽远低于网络物理速率。例如双 40GbE 机器 AllReduce 总吞吐只有 2 GB/s (16 Gb/s)。

- 可能原因：数据路径未充分利用带宽。单机情况可能 NCCL 未用 NVSwitch 而退化为 PCIe4（约 64–80 GB/s，符合观测）。原因如拓扑探测问题、NVSwitch 驱动问题等。多机情况，则可能只用了单端口而非 Bond、或 GPU Direct RDMA 未启用导致受 CPU 内存复制瓶颈（典型 CPU copy 速率 ~10-20 GB/s），或者线程并行度不够未填满带宽。

- 排查步骤：

1.  查看 Bus BW vs Alg BW：用 `NCCL_DEBUG=INFO` 跑 `all_reduce_perf -g 8 -n 10` 并观察输出。例如 8 卡 NVSwitch 理论一来一回 BusBW=144 GB/s，而 Algbw=120 GB/s 时 BusBW 应达 ~240 GB/s。如果 BusBW 恰好等于当前物理接口峰值，比如 80 GB/s ~ PCIe4 x16 极限，那么说明 NCCL 只用了 PCIe 没有 NVSwitch。

2.  拓扑检测：检查 NCCL 拓扑日志是否识别 NVSwitch/NVLink（见 C 节内容）。若没有，可考虑驱动或环境问题：确保裸机运行、CUDA driver 正确加载 NVSwitch 控制器。尝试升级驱动或补丁。

3.  网络瓶颈：在多机上，对比 `algbw` 和 `busbw`：busbw 代表实际流经网络数据速率。如 2 机 100Gbps 网络理想 busbw≈12.5 GB/s。但若 busbw 只有 6 GB/s 且 algbw 更低，则可能 GPU->NIC GDR 未用上（需要 CPU 中转耗时）。验证方法：比较使用 GDR 与否性能，手动 `NCCL_NET_GDR_LEVEL=SYS` 强制 GPU 直 RDMA。如果性能提升，说明之前 GPUDirect 未启用，可能因为需要加载 `nvidia-peermem` 模块或 NIC 不支持 DMA-BUF。反之如强制 GDR 性能下降甚至不稳定，则可能是 ROCE PFC 没配好造成丢包重传。

4.  并行调优：排除以上因素后，如果仍然低于理论，可以尝试增加并发：调整 `NCCL_SOCKET_NTHREADS` 和 `NCCL_NSOCKS_PERTHREAD`。特别在高速以太网上，默认 (1 线程, 1 socket) 很可能跑不满 100Gb。尝试值如 4 和 4（总 16 socket 并行），观察 busbw 是否接近物理线速。注意此调整需在较大 batch 下观察平均性能，并警惕 CPU 占用上升。

- 建议 env 组合：
- _拓扑修正_：容器中建议 `--cap-add SYS_NICE` 以启用 NUMA 支持，或挂载正确的 /sys。针对 NVSwitch 可用 `NCCL_TOPO_DUMP_FILE` 确认拓扑识别结果。

- _性能调优_：`NCCL_SOCKET_NTHREADS=4 NCCL_NSOCKS_PERTHREAD=4 NCCL_NET_GDR_LEVEL=PXB`（例如只允许在同 PCI 域用 GDR，跨 CPU 用 bounce 缓冲）。这些组合需根据观察逐步调整。并仅在确认稳定后用于生产。

- 验证修复：重新运行 nccl-tests 并比较带宽：Algorithm BW 提升且 busBW 接近硬件峰值（例如 12 GB/s 于 100GbE，或 NVSwitch 下达到 120+ GB/s）。还应测试实际训练任务的 step time 是否同步改善，以确保调优有效且无副作用。

场景 4：NCCL 报错 “Unhandled system error” 或 “CUDA Driver error”

- 现象：训练中突然终止，并打印 `ncclSystemError: System call (socket, malloc, etc) failed` 或 `ncclUnhandledCudaError` 等。可能还有 IBverbs 层错误信息如 “failed to register memory” 或 “RDMA creation failed”。

- 可能原因：系统资源或调用失败。典型如：/dev/shm 空间不足导致共享内存 segment 扩展失败；无限制内存锁定不允许导致 GDR mapping 失败；或 CUDA Driver 内部错误比如显存访问非法。

- 排查步骤：

1. 错误码判断：`ncclSystemError` 通常表示某个系统 API 返回错误，可以配合前面的 NCCL WARN 日志找上下文。例如若紧随 “unable to allocate shared memory” 则很明确。`ncclUnhandledCudaError` 则需看是不是之前有 kernel failed 日志。

1. 共享内存问题：容器环境下，默认 /dev/shm 仅 64MB，远不够多 GPU 全通信 buffer。NCCL 初始化时若失败，会 WARN 提示扩展 shm 失败。解决：Docker 跑容器加 `--shm-size=1g --ulimit memlock=-1`。另外检查 systemd 是否移除了用户 IPC（需要 /etc/systemd/logind.conf 设置 RemoveIPC=no）。

1. IB 内存注册失败：如果错误出现在首次 AllReduce 前后，并包含 ibv_reg_mr 失败，可能是进程的内存锁定 (memlock) ulimit 太低。GPUDirect RDMA 需要注册显存映射到 HCA，一张 32GB 卡需要注册同等大小内存。将 `ulimit -l` 调为足够（如无限）并确保 `NCCL_MEM_AFFINITY` 环境正确。

1. CUDA 异常：NCCL 使用 CUDA 流，如果用户前面发生了 CUDA illegal memory access，可能在 ncclGroupWait 时抛出 unhandled cuda error。此类应回溯定位之前的 CUDA 调用 bug，不是 NCCL 自身问题。可以利用 `cuda-memcheck` 工具运行程序，早期发现非法访问。

- 建议 env 组合：
- 针对 shm/内存问题，`NCCL_SHM_DISABLE=0 NCCL_CUMEM_HOST_ENABLE=0` 可尝试不用 cuMem host 机制强制用 /dev/shm，以验证是哪种方式问题（2.24+ 默认用 cuMemHost，有时 NUMA 不支持）。

- 对 IB MR 问题，可设 `NCCL_IB_HCA=<specific>` 只用一块 HCA 测试，或 `NCCL_P2P_DISABLE=1` 绕过 GPUDirect RDMA。

- `CUDA_LAUNCH_BLOCKING=1` 辅助捕获 CUDA 同步错误。

- 验证修复：调整系统配置后，重复运行之前出错的位置。如果不再报错且日志中先前的 WARN 提示消失（如共享内存扩展成功或不再需要扩展），则问题解决。需要的话，在调通后可逐步恢复优化选项（如重新打开 `NCCL_CUMEM_HOST_ENABLE` 看是否依旧稳定），以兼顾性能和稳定性。

场景 5：多机通信经常性波动，性能时高时低

- 现象：同一任务，不同 step 的 AllReduce 耗时抖动很大。例如 100Gb 网络下正常 allreduce 5 ms，但偶尔跳到 50 ms，然后恢复。甚至伴随 NCCL WARN：`NET/IB: Async event: local QP operation err` 之类。

- 可能原因：网络拥塞或丢包导致。InfiniBand 网络中，当流量大时可能触发拥塞管理或 QOS，Adaptive Routing 的切换也会导致波动。RoCE 如果 PFC 配置不完善，可能出现丢包超时重试，使性能断崖式下降。NCCL 检测到 IB 异步错误时（比如链路波动）默认会 Warn 然后重连。

- 排查步骤：

1.  NCCL 日志：观察 NCCL INFO 日志中是否频繁出现 `...Disconnecting`、`...Reconnecting`，或 RNR NACK 等 IB 级别消息。这些表明网络不稳导致重试。

2.  底层监控：使用 Infiniband 自带工具查看错误计数，如 `ibporterr` 是否增长，`sar -n EDEV` 看各网卡丢包。

3.  拥塞控制：如果是 RoCEv2 网络，确认交换机和网卡配置了 PFC（优先级流控）和 ECN，否则遇到深度缓冲拥塞会丢包导致 NCCL 重试超时。对于 InfiniBand HDR/EDR 网络，可检查是否启用了动态拥塞控制（需要 NIC FW 支持）。

4.  NCCL 调参：尝试暂时关闭 Adaptive Routing：`NCCL_IB_ADAPTIVE_ROUTING=0` 看看波动是否减少。如果有效，可能 AR 机制不成熟导致 reorder，可考虑升级 FW 或者先禁用。对 RoCE，可以通过降低 `NCCL_IB_TIMEOUT`（比如设 18）使超时更敏感，但这治标不治本。

- 建议 env 组合：
- `NCCL_IB_SL=` 设一个高优先级 SL 用于 NCCL，确保交换机 QoS 优待；配合 `NCCL_IB_FIFO_TC` 把控控制消息 TC。

- `NCCL_IB_ADAPTIVE_ROUTING=0` 如上，避免路由波动。

- 在应用侧，考虑 `torch.backends.cuda.matmul.allow_tf32 = False` 等减少通信量或者梯度压缩以减小网络压力。

- 验证修复：调整后长时间跑任务，记录 AllReduce 时间分布，看是否抖动降低。若还存在，则需要进一步比如对每对节点使用 `ib_send_bw` 工具测试裸带宽，锁定是否某特定链路的问题。最终稳定后，应在生产中保留必要的 NCCL 参数，并将集群网络配置优化（长远方案）。

场景 6：开启混合精度后偶发 NaN/Inf，怀疑通信精度

- 现象：训练中偶尔出现梯度为 NaN 或损失暴涨，定位怀疑发生在 AllReduce 后。怀疑 NCCL 的 sum 精度或 LL128 压缩算法导致精度损失。

- 可能原因：NCCL 的 float16 AllReduce 默认分两阶段（First reduce in FP16, then finalize in FP32）。精度一般足够。但在极端大规模下，累加顺序可能引入些许不确定。另外 LL128 协议会对数据分块应用低精度 accumulate，存在微小误差。这通常不会导致 NaN，NaN 更多由于网络错误或算子本身。

- 排查步骤：

1.  验证 NaN 来源：使用 `TORCH_NCCL_NAN_CHECK=1` 提前检测各步输出 NaN。看看是否某 rank 的激活值先成为 NaN，而非 AllReduce 过程注入。

2.  关闭融合：禁用 GradScaler 或将 accumulation 降低，看看 NaN 是否还出现。可能是数值本身爆了而非通信。

3.  协议替换：试 `NCCL_PROTO=Simple` 强制不用 LL/LL128。如果 NaN 不再出现，可能 LL128 某 bug 引发错误 sum。也可尝试 `NCCL_ALGO=Tree` 改变累加次序看看。

4.  Check 通信正确性：用 nccl-tests 自带的验证模式运行几千轮：`all_reduce_perf -c 1 -check` 开启数据正确性检查。如果都有 Pass，则 NCCL 本身逻辑没问题。

- 建议 env 组合：
- 为安全，可将 `NCCL_ALGO=Ring NCCL_PROTO=Simple` 在要验证精度的实验中使用，确保按最高精度路径汇总。

- 如果多节点间有可能数据不一致，也可利用 `TORCH_DISTRIBUTED_DEBUG=INFO` PyTorch 在不同步时会有提示。

- 验证修复：确认调整后 NaN 问题不再出现。若确定是 NCCL 协议问题，应向 NVIDIA 反馈或查看 release notes 已知问题。否则，多半是训练本身需调整（如降低学习率等）。

场景 7：单机多进程模式下 NCCL 初始化缓慢

- 现象：例如 PyTorch DDP 模式，8 卡单机，调用 `init_process_group` 非常慢（> 30 秒），但最终能成功开始训练。

- 可能原因：在单机多进程场景，NCCL 需要通过 socket 进行 out-of-band 引导（交换 ncclUniqueId 等）。如果本机开启了很多 docker 虚接口或 loopback 优先而其他线程还没起来，可能 NCCL 在尝试接口时超时重试。NCCL 默认排除 lo 和 docker\* 除非没其他接口。另一个原因是生成 UniqueId 采用全员通信，MPI 或文件系统差导致慢。

- 排查步骤：

1. 日志观察：开启 `NCCL_DEBUG=INFO`，看每个 rank 在初始化阶段的时间戳。如果卡很久，多半在`ncclCommInitRank`内部。INFO 日志可能打印 “Trying to bootstrap via x.x.x.x” 之类，可发现如果选错接口。

1. 指定接口：设置 `NCCL_SOCKET_IFNAME=<eth_name>`，确保 NCCL 用正确的本地高速接口而非虚拟接口。

1. UniqueId 交换：PyTorch 中默认使用 TCP socket 交换 uniqueId，如果机器 DNS 不好或者需翻墙，会拖慢。可以尝试 `init_process_group(..., store=...)` 用本地文件或 shared memory 作为 store，绕过 DNS。NCCL 2.23+ 还提供 `NCCL_OOB_NET_ENABLE=1` 可以让引导也走 NCCL 网络插件而不是系统 socket。但这需要配置，不是默认路径。

- 建议 env 组合：
- `NCCL_SOCKET_IFNAME=eth0 NCCL_IB_DISABLE=1`（单机无 IB，也可禁 IB 插件让其别无选择用 socket）。

- `NCCL_UID_RUNTIME_BINARY=1` (如果适用，理论上可以缩短 uniqueId 生成方式，不过这通常不是瓶颈)。

- 验证修复：调整后再次初始化，测量耗时。如果下降到 <5 秒，则说明确实接口选择或配置改善了。如仍慢，可以在 profile 中查看是否 Python 端 store 阻塞长，定位问题。

场景 8：XLA/TPU 等非常规场景下 NCCL 报错不支持

- 现象：使用 PyTorch XLA (GPU+TPU 混合) 或 HPC 上 NVLink+IB 混合拓扑时，NCCL 报一些不支持 CollNet/NVLS 之类的错误，或者 Hang。

- 可能原因：NCCL 某算法在当前硬件不适用但被错误启用。如 CollNet 需要服务器有独立网络分层，但混合场景无此条件，如果 NCCL 版本判断有误可能导致 hang。

- 排查步骤：

1.  禁用高级特性：`NCCL_ALGO=^CollNet`，`NCCL_NVLS_ENABLE=0` 禁用 NVLink SHARP，`NCCL_PXN_DISABLE=1` 禁用 PXN。基本回退到经典 Ring/Tree。

2.  查看 issue：搜索 NVIDIA NCCL release notes 或 GitHub issue，有无针对 TPU or multi-node NVSwitch 的已知问题和补丁。

3.  版本回退：有时新特性 Bug，可以尝试 NCCL 降级或升级到最新补丁看是否解决。

- 建议 env 组合：保守期间对非典型架构统一加上述禁用的变量，确保 NCCL 仅用最稳妥路径（虽然可能性能不最高）。

- 验证修复：让通信能跑通、结果正确，然后再逐一开放看性能提升与稳定性，找到平衡点。

> 注：以上场景远非穷尽。实际排障中，要结合具体软硬件环境，对症下药。关键是遵循先易后难、由广到细的思路：先确保外围配置正确，然后利用 NCCL 提供的调试开关缩小可疑范围，并借助 nccl-tests 做对比实验验证猜想。每个变量改动都应记录效果，最终选择对性能和稳定性最佳的方案。

## H. 一页式 NCCL 调优与排障 Cheat Sheet

最后，将本文介绍的 NCCL 调试“工具箱”汇总成一页速查表，便于在实战中快速复制使用。

### 日志与诊断开关

- 基础日志：`NCCL_DEBUG=INFO` – 开启调试日志（版本、初始化细节、错误）。常用级别：WARN（默认、仅错误）、INFO（推荐）、TRACE（详细追踪，仅短时间使用）。

- 子模块过滤：`NCCL_DEBUG_SUBSYS=INIT,COLL,...` – 聚焦特定子系统日志。默认 ENV/INIT 等，调网络问题常加 `NET,GRAPH`。

- 日志输出定向：`NCCL_DEBUG_FILE=nccl_%h_%p.log` – 日志重定向到文件，以 hostname+PID 区分。避免多进程 stdout 混杂。

- 时间戳：`NCCL_DEBUG_TIMESTAMP_FORMAT="%H:%M:%S"` – 修改时间戳格式，或配合 `TZ` 环境变量调整时区。

- 线程命名：`NCCL_SET_THREAD_NAME=1` – 让 NCCL 后台线程具名，便于 profiling。

- PyTorch 超时监控：`TORCH_NCCL_BLOCKING_WAIT=1` – NCCL 调用等待改为阻塞，超时抛异常，防止沉默 hang。

- PyTorch 异常处理：`TORCH_NCCL_ASYNC_ERROR_HANDLING=1` – 异步错误时自动中止全部进程。（Pytorch<=1.11 用旧 env `NCCL_ASYNC_ERROR_HANDLING`).

- PyTorch 超时 Dump：`TORCH_NCCL_DUMP_ON_TIMEOUT=1` + `TORCH_NCCL_TRACE_BUFFER_SIZE=1000000` – Watchdog 超时时 dump 最近操作轨迹。Dump 文件缺省 `/tmp/torch_nccl_<rank>_<pid>.log`，可用 `TORCH_NCCL_DEBUG_INFO_TEMP_FILE` 指定。

- PyTorch 额外：`TORCH_NCCL_DESYNC_DEBUG=1` – 发现 collective 不同步时提示；`TORCH_NCCL_NAN_CHECK=1` – 每次 collective 后检查 Nan。调试数据完整性用。

### 传输通道控制

- 禁用直连 P2P：`NCCL_P2P_DISABLE=1` – 禁 NVLink/PCIe GPU 直接通信，改经 SHM/网络。Hang 排查用于隔离 P2P 因素。

- 限制直连级别：`NCCL_P2P_LEVEL=NVL/PIX/...` – 控制多远的 GPU 间用直连。如只想 NVLink 用 P2P，其它走 SHM，则设 `PIX`。

- 禁进程内直访：`NCCL_P2P_DIRECT_DISABLE=1` – 同一进程内多 GPU 不直接访存，避免 CUDA 没有 peer access 导致 hang。

- 禁共享内存：`NCCL_SHM_DISABLE=1` – 不用 /dev/shm 传输。调试 SHM 空间不足或跨 NUMA 问题，可暂关。

- 禁 IB/RoCE：`NCCL_IB_DISABLE=1` – 禁用 InfiniBand/RoCE 网络，改用 TCP。用于确认 IB 相关问题（性能骤降则说明 tcp 接管）。

- IB 网卡选择：`NCCL_IB_HCA="^mlx5_2"` – 排除 mlx5_2 卡不用；`NCCL_IB_HCA=mlx5_0:1` – 只用 mlx5_0 的 1 号端口。多 HCA 环境下调度使用。

- 指定网络接口：`NCCL_SOCKET_IFNAME=eth0` – 强制用指定前缀接口 (eth0 等)；`^docker` 排除某类接口。避免选错网络。

- IPv4/v6：`NCCL_SOCKET_FAMILY=AF_INET` – 强制用 IPv4（有时避免 v6 解析问题）。

- GPU 直 RDMA 控制：`NCCL_NET_GDR_LEVEL=PHB` – 仅 NUMA 内启用 GPU 直 RDMA。`LOC` 禁 GPU 直接发 NIC，全走 CPU 内存（可 debug GDR 问题）。

- PCIe RO：`NCCL_IB_PCI_RELAXED_ORDERING=2` – 自动用 Relaxed Ordering；`=0` 强制禁用（debug 某些 RO 问题）。

- IB 自适应路由：`NCCL_IB_ADAPTIVE_ROUTING=0` – 禁用 AR。调试拥塞波动时可尝试。

- 共享 Buffer：`NCCL_NET_SHARED_BUFFERS=0` – 禁用共享内存池；`NCCL_NET_SHARED_COMMS=0` – 禁用 PXN 共享连接。极罕见情况使用（如怀疑内存池问题）。

### 算法与协议调整

- 禁用 LL128：`NCCL_PROTO=^LL128` – 排除 LL128 协议。常用于疑似 LL128 相关 bug 时（PCIe 平台本也默认无 LL128）。

- 仅用简单协议：`NCCL_PROTO=Simple` – 不使用 LL/LL128，只用 Simple 协议。调试小消息性能时可对比 LL。

- 算法限定：`NCCL_ALGO=Ring` – 强制环算法；`NCCL_ALGO=^Tree` – 禁用树算法。定位某算法导致的性能或 bug，可以尝试不同组合（Ring vs Tree vs CollNet）。

- 禁用 CollNet/NVLS：`NCCL_ALGO=^CollNet` / `NCCL_NVLS_ENABLE=0` – 关闭高阶聚合算法。防止在不支持配置上误启用导致问题。

- 禁用 PXN：`NCCL_PXN_DISABLE=1` – 关闭 PxN 中继。复杂拓扑中简化调试。

- 限制通道数：`NCCL_MAX_NCHANNELS=4` – 限制最多 4 个通道。某些 GPU 资源紧张场景可试降低并发通信数。

- 调整每线程 socket：`NCCL_NSOCKS_PERTHREAD=4 NCCL_SOCKET_NTHREADS=4` – 增加并发连接数和线程数。这是性能调优选项，在确认稳定后可用于提升大带宽网络利用率（如 4×100G NIC）。注意遵守乘积<=64 限制。

### 实验排障矩阵模板

在排障时，可采用以下实验矩阵逐项尝试，并记录现象变化：

| 调试手段         | 操作                                              | 预期效果/判断依据                                |
| ---------------- | ------------------------------------------------- | ------------------------------------------------ |
| 禁用 IB 改 TCP   | `NCCL_IB_DISABLE=1`                               | 若问题消失：指向 IB 相关（配置/驱动/FW 问题）。  |
| 禁用 P2P 直连    | `NCCL_P2P_DISABLE=1`                              | 若问题消失：GPU 直连模块异常（NVLink/P2P Bug）。 |
| 禁用 LL128 协议  | `NCCL_PROTO=^LL128`                               | 若问题消失：LL128 协议 bug 或数据精度问题。      |
| 改用 Tree 算法   | `NCCL_ALGO=Tree` 或 `^Ring`                       | 若性能改善：环拓扑瓶颈，树算法更优（或反之）。   |
| Socket 线程并行  | `NCCL_SOCKET_NTHREADS=4, NCCL_NSOCKS_PERTHREAD=4` | 若性能改善：之前单线程未压满网络，可考虑保留。   |
| 固定接口         | `NCCL_SOCKET_IFNAME=<dev>`                        | 若初始化成功：多网卡下原先选错接口导致握手失败。 |
| GPU 直连级别     | `NCCL_P2P_LEVEL=SYS` / `PIX` 等                   | 性能/稳定性变化：确认跨 CPU 直连是否有问题。     |
| 禁用 SHM         | `NCCL_SHM_DISABLE=1`                              | 若初始化通过：原问题来自 /dev/shm 受限。         |
| Relaxed Ordering | `NCCL_IB_PCI_RELAXED_ORDERING=0`                  | 若性能变化：RO 参数影响虚拟化环境中的 IB 性能。  |
| Adaptive Routing | `NCCL_IB_ADAPTIVE_ROUTING=0`                      | 若抖动减少：AR 在网络中引发波动。                |

_注：每次仅改动一个变量，观察效果，避免多项变化难以定位原因。_

### 信息收集与版本检查

- 版本：确保所有节点 NCCL 版本一致（`NCCL_DEBUG=VERSION` 可打印版本）。注意 PyTorch 内置 NCCL 版本，可通过 `torch.cuda.nccl.version()` 获取。已知问题可在 \[NCCL Release Notes] 中查找修复。

- 驱动/CUDA：CUDA Driver >= NCCL 要求版本，否则可能发生挂起（Release Notes 中通常注明）。尽量使用 NVIDIA 官方稳定的驱动+CUDA 组合。

- 拓扑：使用 `NCCL_TOPO_DUMP_FILE` 保存拓扑，对比实际硬件。检查 NVLink/NVSwitch 节点是否被正确识别；检查 PCI 域和 NIC 归属是否合理。

- 网络设置：记录 ifconfig/ibstatus，确保所用接口 UP 状态正常。收集 `sysctl -a | grep mlnx` 等判断 RoCE ECN/PFC 配置。

- 错误日志：保存所有 rank 的 NCCL WARN/ERROR 行，包含 error code 和 rank 信息，便于与 NCCL 源码/issues 对照。

### 安全与性能提示

- 不要长期保留调试变量：如 `NCCL_*_LEVEL` 之类在问题解决后应恢复默认。调优类变量可加入作业配置，但需有注释说明理由，防止遗忘。

- 数据正确性：禁用 `NCCL_CHECK_POINTERS` 可能提升性能，但切勿在开发调试时关闭安全检查。同理，大多数调优选项在 throughput 和 determinism 间权衡，生产环境应充分验证不会引入数值差异。

- 关注官方指南：NVIDIA 针对新硬件（如 Hopper NVLink4、双 rail 网络）会发布专门调优指南。这些文档提供了推荐参数和已知陷阱（如 NVLS silent fallback hang 等）。充分利用这些信息可事半功倍。

- 升级与回归：NCCL 随新版本性能提升也可能带来新 bug。建议在关键任务前做小规模 A/B 测试不同版本 NCCL，观察日志是否有异常 warn，性能是否稳健，然后再推广升级。

---

通过以上方法和技巧，我们可以逐步掌握 NCCL Debug 的“全栈手段”，从环境变量调优到日志诊断、从协议算法选择到实际案例排查，在遇到 NCCL hang、性能瓶颈或数据异常时做到心中有数、手中有方。现代大规模分布式训练系统复杂多变，但相信凭借扎实的官方资料和工程实践经验，我们能够将 NCCL 的行为透明化、问题可解化，为训练任务保驾护航。

参考文献：

- NVIDIA NCCL 官方文档 – _Environment Variables_、*Troubleshooting*等章节

- PyTorch Distributed 官方文档 – _ProcessGroupNCCL Environment Variables_

- NVIDIA/nccl-tests 项目文档 – _PERFORMANCE.md_（算法带宽与总线带宽解释）

- NVIDIA Developer Forums – NCCL 性能与错误相关讨论
