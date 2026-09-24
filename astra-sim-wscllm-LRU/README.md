# astra-sim-wscllm-LRU - 晶圆级芯片（WSC）LLM 推理架构与机制

> 本仓为 Execution-Driven 改造后的**裸仓库终态**：仅保留路径③（strategy 关感知）
> 与路径④（strategy 开感知）两条在线仿真路线；离线静态（①）与 replay（②）
> 已删除。仓库 request-neutral：不带任何 request 队列，正式入口缺失输入
> fail-closed。
>
> 本仓是 ASTRA-sim 2.0 的晶圆级芯片（Wafer-Scale Chip, WSC）推理仿真改造仓，
> 系 `astra-sim-wscllm` 的 **KV 冷热管理改造仓（-LRU）**：在 WSC-LLM 请求→实例
> 映射**逐行保留**（静态 P/D 分区 + 最小排队选点 + 静态路由钉死 + FCFS 队头
> 阻塞，`wsc_llm_scheduler.py` 零改动）的前提下，把原仓的"两态 KV + LRU 零代价
> 删除 + 历史全量重算"整体替换为**三态 KV 冷热管理**——
> LOCAL / PARTIAL（前 ⌈L/2⌉ 层片上 + 后 ⌊L/2⌋ 层远端）/ REMOTE 三态 +
> **去类型化两段式 LRU 逐出**（先扫全部会话后半层、仍不足才整体外迁=唯一回退）+
> 远端共享内存池逐出/回迁物理链路 + PARTIAL 恢复真流水；净额预占/extend 回补、
> 容量纪元与 journal 账本等 wscllm 特有机制**原样保留**并随三态扩账。逐出内核
> 与发射链路承袭 `astra-sim-sh_2.0`（去类型化；恢复取代重算，RECOMPUTE 路径
> 已从历史决策中删除）。六个同源仓
> `astra-sim-face / astra-sim-wscllm / astra-sim-sh_2.0 / astra-sim-sh_3.0 /
> astra-sim-face-LRU / astra-sim-wscllm-LRU`
> 共享同一套硬件架构建模与同一套术语，仅在"请求→实例映射与 KV 管理策略"上分化（见文末对照表）。
> 阅读仓内任何代码、配置或文档前，请先建立以下概念。硬件参数唯一权威来源是
> `sh_test_mesh/hardware/face_case5_config_c.json`，运行时配置
> （system.json / network.yml / remote_memory.json / comm_group.json）由
> `sh_test_mesh/config_resolver.py` 从它派生，禁止手改。

## 核心概念与硬件架构建模

### A. 晶圆级芯片（wafer-scale chip）＝整个计算系统

指被仿真的**整片晶圆这一完整计算系统**：一片二维 mesh 晶圆（行列数由硬件配置的
`mesh.rows/columns` 给出），全部芯粒之间由片上网络（NoC）互连，每颗芯粒自带本地
HBM。NPU 数量禁止单独配置，恒等于行 × 列；芯粒按 row-major 编号，rank = 行 × 列数 + 列。

### B. NoC（片上网络）

晶圆内部芯粒间的互连：非环绕二维 mesh 上相邻芯粒间的有向链路（die-to-die），
链路带宽与每跳时延取硬件配置 `d2d` 的值（多跳逐跳累计），XY（维序）确定性路由，逐跳
store-and-forward，共用同一链路的多条数据流进同一 FIFO 排队（拥塞感知）。C++ 侧由
分析型网络后端实现（`extern/network_backend/analytical/congestion_aware/`），Python
规划层有同一语义的确定性 XY 路由函数。注意：这是分析级模型，不建模路由器微架构
（crossbar / VC / credit），也不做 cycle-accurate 仿真。

### C. 芯粒（die / NPU / rank）＝挂在 NoC 上的每一个单元

挂载在 NoC 上的**每一个单元**即一颗芯粒。代码与配置中统一称 **NPU**（或 rank），
权威映射为"一个 FACE 物理计算 die = 一个 ASTRA-sim NPU"。每颗芯粒包含：

- **计算单元**：峰值算力取硬件配置 `compute` 的值，算子以 Roofline 模型计时执行；
- **本地 HBM**（KV 热层）：带宽 / 时延 / 容量取硬件配置 `local-hbm` 的值（容量按
  profile 档位选择）；多用户带宽竞争模型见 G 节；
- **网络接口（路由端口）**：mesh 上的一个 Device，经有向链路连接其在 mesh 中的物理邻居；
- **远端内存端口**（仅边缘芯粒拥有，且仅在启用远端内存扩展的配置中生效，见 D）：
  与片外统一内存池直连的中转通道，对应物理系统中的 SerDes 等片外互连。

> 术语辨析：本说明中"芯粒"＝mesh 上的完整单元（die＝NPU＝rank）。指标层另有一个
> 仅用于观测展示的"chiplet 投影"（把每 NPU 的本地 HBM 均分为若干份，份数取
> `metrics_config.json` 的 `chiplets_per_npu`，对应论文场景中的 HBM 芯粒划分），
> 不参与仿真与调度决策，与本术语无关。

### D. 边缘芯粒、非边缘芯粒与片外统一内存池

- **片外统一内存池（remote memory pool）**：挂在晶圆之外的远端存储，逻辑名
  `unified-kv-cache-pool`，充当 KV 缓存的冷层。它并不与全部芯粒直连：只有位于 mesh
  **物理边界**（mesh 周界）上的芯粒拥有直连远端池的远端内存端口（数量随 mesh 形状自动推导），
  这些芯粒称为**边缘芯粒（edge NPU）**；其余芯粒称为**非边缘芯粒**。
- **非边缘芯粒借道访问**：非边缘芯粒读写远端池时，数据先经 NoC（XY 路由）送到**最近**
  的边缘芯粒（曼哈顿距离最近、并列取最小 rank），由该边缘芯粒的端口代为完成远端访问，
  再经 NoC 返回。
- **"统一"的含义**：远端池是统一逻辑地址空间——从边缘芯粒 A 写入的数据可以从另一个
  边缘芯粒 B 读出、回到片上任意芯粒，写入端口与读出端口可以不同（manifest 字段
  `logical_pool_shared_across_edges: true`），因此称为统一内存池。
- **端口代价模型**：每个边缘端口为严格 FIFO 事务队列，单次访问
  `耗时 = 端口时延 + 字节数 / 端口带宽`（取硬件配置 `remote-memory` 的值）；
  端口之间完全并行，远端池总容量不设限
  （`extern/remote_memory_backend/analytical/AnalyticalRemoteMemory.cc`）。
- **本仓配置**（2026-09 改造，数据面+发射端全链激活）：远端内存**已启用**——
  硬件配置声明 `PER_NPU_MEMORY_EXPANSION`（mesh 边界 NPU 挂端口，
  `npu-selection: mesh-boundary`、`logical-pool: unified-kv-cache-pool`、
  512 GB/s / 100 ns），resolver 物化 `remote_memory.json`（RC 目录标签
  `edge_remote_memory_pool`），在线 runner 以 `--remote-memory-configuration`
  传入 C++；原仓 loader 侧 `_validate_no_memory_expansion` 守卫已随改造删除。
  C++ 引擎对 KV 语义保持 opaque，仅提供 MEM 节点（mem_store / mem_load /
  local_hbm_kv_restore）的物理计时与 HBM N-way 带宽争用计费（见 G 节）；
  本仓策略层（`kv_cache_policy=session_lru_tiered`）实际发射远端逐出/回迁
  链路（见 §1.1 与《request实例映射与KV冷热管理策略说明.md》）。本仓 KV
  冷热三态：LOCAL 全驻本地 HBM / PARTIAL 前缀片上+后缀远端 / REMOTE 全远端；
  跨实例历史走 NoC 迁移或"远端存取 + 回迁"。

### E. 实例（instance）＝NoC 上紧密相邻芯粒组成的矩形区域，共同承担一个请求的推理

实例是 NoC 上**空间紧密相邻的多颗芯粒组成的连续实心矩形**：形状（行列数）、芯粒数
与张量并行度（TP）由实例布局配置给出，全部实例恰好铺满整片晶圆。布局唯一来源是
`sh_test_mesh/workload/llama2_7b_inference/trace_config.csv` 的各行 `inference_group`
（配置顺序兼作所有调度选择键的最终 tie-break）；
`build_instances()` 对矩形性、不重叠、全覆盖、等尺寸做硬校验，并由物理邻接自动生成
实例邻接图。按所辖芯粒在 mesh 拓扑中的物理位置划分：
- **边缘实例（edge instance）**：包含至少一颗边缘芯粒的实例（即实例所辖 rank 集合与
  mesh 物理边界有交集）；
- **非边缘实例（non-edge instance / edge-free instance）**：完全由内部非边缘芯粒组成、
  不包含任何边缘芯粒的实例（位于晶圆内部核心区域，实例 rank 集合与 mesh 物理边界无交集）。

模型权重与 KV 在实例内按张量并行（TP）做 whole-head 分片（注意力头按整头划分到各
TP rank，不切割单个头）；
**一个 session 的完整 KV 只驻留在一个实例内**，不跨实例切分；跨实例移动表现为 NoC
逐跳迁移（noc_migrate）或"远端存取 + 重新载入"。本仓为 **PD 分离**（disaggregated）
布局：Prefill-only 实例与 Decode-only 实例分区部署（数量与位置由 `trace_config.csv`
各实例的 `phase_role` 给出）；prefill 完成后 KV 按离线静态映射（最近邻、边不相交）
迁往 decode 实例。

### F. 本仓在六仓中的定位

| 仓库 | 实例组织 | 请求→实例映射策略 | KV 驻留与恢复 | 远端内存池 |
|---|---|---|---|---|
| astra-sim-face | 统一实例（P+D 同实例） | FACE 原始映射：prefill 选剩余 chunk 最少；decode 在邻接图加权距离限制（阈值 = D2D 带宽 / 本地 HBM 带宽）内按 per-die Roofline 增量代价 | RESIDENT/EVICTED 两态；LRU 逐出＝零代价删除；恢复＝重算 | 未启用（NO_MEMORY_EXPANSION） |
| astra-sim-wscllm | PD 分离（Prefill-only + Decode-only 分区） | prefill 选排队请求最少；decode 用静态一跳 P→D 映射 | RESIDENT/EVICTED 两态；LRU 逐出；恢复＝重算（跨实例历史走 NoC 迁移） | 未启用（NO_MEMORY_EXPANSION） |
| astra-sim-face-LRU | 统一实例（P+D 同实例） | 同 face（映射逐行保留） | 三态；去类型化两段式 LRU 逐出；恢复＝远端池回迁（PARTIAL 同实例真流水） | 启用（PER_NPU，全部边缘芯粒挂端口） |
| **astra-sim-wscllm-LRU（本仓）** | PD 分离（Prefill-only + Decode-only 分区） | 同 wscllm（映射逐行保留：静态 P/D 分区 + 最小排队 + 静态路由钉死 + FCFS 队头阻塞） | LOCAL/PARTIAL/REMOTE 三态；去类型化两段式 LRU 逐出（先扫全部会话后半层、不足才整体外迁）；恢复＝远端池回迁（跨实例历史走 prefix NoC 迁移 + suffix 远端恢复真流水） | 启用（PER_NPU，mesh 边界 NPU 挂端口） |
| astra-sim-sh_2.0 | 统一实例 | prefill Roofline 剩余负载均衡（历史 KV 全/部分驻留与全逐出统一）；decode 按 per-die Roofline 增量代价 + HBM 剩余 tie-break | 三态（含半驻留 PARTIAL）；两阶段类型感知逐出（human 类先于 tool 类）；流水化部分恢复 + HBM 恢复/推理带宽共享 | 启用（全部边缘芯粒挂端口） |
| astra-sim-sh_3.0 | 统一实例 | 三段式 prefill（首请求避边缘 / HBM 命中 sticky / 远端命中负载均衡）；decode 本地化固定同实例 | 三态；两阶段逐出；流水化部分恢复（机制同 sh_2.0） | 启用（全部边缘芯粒挂端口） |

六仓还共享同一套 Execution-Driven 在线仿真机制层（`astra-sim/workload/execution_driven/`：
RequestIngress / DecisionMailbox / DecisionBridge / GraphBatchCommitter 等）：策略决策由
Python 在线服务层实时给出，计时由 C++ 物理时钟推进；各仓仅保留路径③（strategy 关感知）
与路径④（strategy 开感知）两条在线路线。六仓在线决策均不再使用任何离线 LUT
（残留 LUT 查表已随静态链路一并删除）：face / sh_2.0 的 decode 候选代价由
在线 Roofline 模型即时计算（`per_die_delta_ns`），决策确定可复算复放。

**C1 typed 响应解析（2026-08-29，sh_1.0 参考仓先行、本仓 W3 同步）**：桥响应在
`FileDecisionBridge::deliver_and_receive` 内经 `parse_graph_batch`
（`ParsedGraphBatch.hh/.cc`）**一次性**解析为 typed 批（`GraphBatch` 即其别名），
validate / 强制 liveness preflight / commit 装配 / 锚点注册四处消费同一份
typed 数据——改前"DOM 随批驻留 + 同一棵 nlohmann DOM 被字段级提取 4 遍"的
重复解码消除（sh_1.0 参考仓实测 30s 档 C++ 自身耗时 −16.1%，端到端无回退）。
**协议收紧（fail-closed 强化，fixture 作者可见）**：顶层未知键、节点/边/watch/alarm
键集不符（缺失或多余）、整型域违例（负数、>UINT64 字面量、tag/priority 超 32 位）
在解析层直接经 `bridge_fatal` abort（与既有协议违例同通道同退出码）；错误响应仍
先行于结构解析判定（时序不变）。Python 生产者（graph_batch_builder）本就满足全部
键集，行为不变；三个 verify fixture 服务（same_tick_milestone / wakeup_guard /
lifecycle）原先发射的空 `comm:{}`/`coll:{}` 已补满形缺省。本仓特有的节点级可选键
`is_local_hbm_kv_restore`（仅真值发射）与顶层 HBM 字段（compute 可选
`hbm_access_mode`：1=POOL_READ、2=POOL_WRITE，仅非零发射；comm 可选
`hbm_charge`，仅 False 发射——三键 absent 即默认语义）在解析规则内一并收口
（2026-09 数据面激活扩键集，照 sh_2.0 键集形态）。
数组顺序契约：六个数组与 watch 成员一律按发射序保序，解析层不排序不去重。

### G. 芯粒本地 HBM 带宽模型（`hbm-bandwidth-contention`）

真实硬件中每颗芯粒的本地 HBM 是共享资源。C++ 执行层为每个 rank 建一个
`LocalHbmBandwidthModel` 流体模型（`astra-sim/workload/LocalHbmBandwidthModel.{hh,cc}`，
由 system.json 键 `hbm-bandwidth-contention` 开关，默认开；2026-09 数据面激活起照
sh_2.0 N-way 模型重写），本 rank 的全部 HBM 用户按**六类 JobKind**竞争同一份
`local-mem-bw`（远端池启用后从两类扩为六类；池自身带宽竞争不在此模型内，由
边缘端口 FIFO 单独计时）：

1. **COMP 节点（roofline 计算访存）**：bytes = `tensor_size`（读写合并计费），
   ops 与访存并行排空，单用户时保持 roofline `max(计算, 访存)` 语义。带非 0 校准
   runtime 的 COMP 节点不进流体模型、保持校准时长（本仓 trace 的 COMP 无 runtime）。
2. **NoC p2p 通信的数据端点（COMM_READ / COMM_WRITE）**：发送方 rank 的 HBM 读 +
   接收方 rank 的 HBM 写（bytes = comm 字节数；history_kv、prefill_to_decode_kv
   等迁移两端都计）。**多跳途经的中间芯粒不占 HBM**（路由器直通——网络后端只在
   端点计费）。bytes==0 不建作业，1B ACK 照常计；ET comm 节点属性 `hbm-charge`
   （默认 true）可显式豁免某端点。
3. **KV restore DMA（RESTORE）**：`local_hbm_kv_restore` 节点（节点级
   `is_local_hbm_kv_restore=true`）＝远端回迁目标端的唯一数据计费；每 rank
   `hbm_dma` 单槽（同一 rank 同时至多一个在飞 restore）；system 键
   `hbm-kv-restore-bandwidth-sharing` 决定 restore 是否进入共享模型：`1`
   进入（contention 下随六类作业 N-way 均分；contention 关时与 COMP 两用户
   50/50 A/B 基线）；`0` 时无论 contention 均走单槽闭式时长
   `local_mem_latency + bytes/bw`（不与推理共享带宽；零字节 restore 亦走闭式
   定长，不建 HBM 作业）。
4. **池流量端点（POOL_READ / POOL_WRITE）**：直连池端点的
   `mem_store(hbm_access_mode=1)` 记 POOL_READ（池写时的本地 HBM 读一次）——
   仅当逐出/回迁源 rank 本身是边缘 rank 时发生；借道链路的边缘端
   `mem_store`/`mem_load` 无 mode 键＝本地零计费、仅池端口 FIFO。

仲裁口径：N 个并发用户严格均分（各得 `full_rate/N`），任一作业完成立即释放并给
剩余用户重分配（事件驱动）；带宽直接用配置标量——读写共享同一总线与总带宽，不区分
读写方向，不区分峰值/持续带宽；`local-mem-latency` 在每作业启动时计一次。p2p comm
节点完成 = join(网络侧完成, 本端 HBM 作业完成)，两侧都到齐才触发节点完成（幂等）；
池直连节点的两侧 General 事件共享同一 join 闩锁（同侧二次到达 = critical fail-closed）。
**TP 集合通信（PacketBundle 3x 路径）不在范围**。端点计费总规则：**每字节每链
恰好在正确端点计费一次**（过路 hop 不计、双计有守卫拒收），逐链路计费地图见
§1.1 物理链路表。

`hbm-bandwidth-contention: 0`（或 `local-mem-bw <= 0` 自动回退）＝完全恢复旧行为：
roofline 闭式公式、comm 立即随网络完成、不计 HBM。指标导出（MetricCollector，
旧键不变）：每 rank `hbm_busy_ns`、六类 served bytes（comp / comm_read /
comm_write / restore / pool_read / pool_write）、`hbm_peak_concurrent_jobs`、
`hbm_redistribution_events`、`hbm_shared_ns`（多用户共享窗口时长）、
`local_hbm_restore_bytes_issued`（restore 发射侧带计数）、真实利用率
`hbm_bw_util = busy_ns / 墙钟窗口`。单测：
`astra-sim/workload/execution_driven/tests/local_hbm_bandwidth_model_test.cc` 与
`local_hbm_model_test.cc`（6 作业 N-way 模型权威数值测试，2026-09 数据面激活新增）。
策略细节见《request实例映射与KV冷热管理策略说明.md》。

## 1. 本仓是什么

- **映射语义（保留对象，未改动）**：PD 分离：6P:3D + StaticPdMapping 静态路由
  （`wsc_llm_scheduler.py` 的 `build_instances` / `build_static_pd_mapping` /
  `select_prefill_instance` / 路由钉死 / FCFS 队头阻塞逐行保留，零改动）；
  prefill 实例内严格 FCFS，队头容量不足阻塞整条队列、不改投其他实例。
- **KV 冷热管理（本仓改造核心，2026-09 改造）**：三态（LOCAL / PARTIAL /
  REMOTE）+ 去类型化两段式 LRU 逐出 + 远端共享池逐出/回迁 + 六分支恢复 +
  PARTIAL 恢复与前缀计算真流水（详见 §1.1）；净额预占/extend 回补、容量纪元、
  journal 账本等 wscllm 特有机制保留并随三态扩账；session 驻留状态/字节数/
  驻留层段由运行时 KV 账本（`session_kv_manager.py` 三态内核）动态维护，无 sidecar。
- **执行驱动机制层**（`astra-sim/workload/execution_driven/`）：在线事件驱动
  （RequestIngress/DecisionMailbox/WatchRegistry/GraphBatchCommitter/长连接
  DecisionBridge 等），六仓接口一致。
- **验证证据**：Tier B 等价、感知开/关决策逐字节一致、分层账本对账、
  机制 fixtures、三态不变量审计（`SH_STRICT_KV_INVARIANTS=1` 全绿）、
  journal schema v2 run 末 checksum 五项守恒全绿。

### 1.1 会话 KV 三态冷热管理（`kv_cache_policy = session_lru_tiered`）

本仓 `trace_config.csv` 的 `kv_cache_policy` 已切换为 `session_lru_tiered`
（**交付态值**；旧值 `session_lru_recompute` 仍可读、仅作旧 trace 兼容，行为上
管理器唯一、无档位分支）。值域 fail-closed 登记齐全：loader
（`generate_wsc_llm_trace.py`）+ 调度器构造校验（`wsc_llm_online_scheduler.py`）+
`plan_materializer.py` `_KNOWN_KV_POLICIES` + `metrics_postprocess.py`
`KV_POLICY_LABELS`。逐出内核与发射链路承袭 sh_2.0，按改造策略做**去类型化**裁剪。

**三态与层切分**（`sh_test_mesh/workload/llama2_7b_inference/session_kv_manager.py`）：

| 状态 | 含义 | 片上层数 |
|---|---|---|
| `LOCAL_HBM` | 全部 L 层驻留本实例 | L |
| `PARTIAL_HBM_REMOTE` | 前 ⌈L/2⌉ 层片上 + 后 ⌊L/2⌋ 层远端池 | L − L//2 |
| `REMOTE_MEMORY` | 全部层在远端池，无实例归属 | 0 |

- 切分规则 `partial_resident_prefix_layers = L − L//2`（奇数层较大半驻留），
  **不可配置**；层区间字节按层数正比折算，逐 NPU 按整头（whole-head）分片，
  本地 + 远端逐字节守恒（`kv_cache_shard_bytes_for_layer_range` 守恒断言）。
- 位置值域（决策日志 `history_location_before`）：`local_hbm` /
  `partial_hbm_remote` / `remote_memory`。

**去类型化两段式 LRU 逐出**（触发点与 wscllm 原仓完全一致的**四处收敛点**：
prefill 准入 `prepare_history`、decode 准入 P→D 交接 `move_prefill_to_decode`、
容量增长 `grow_prefill`/`grow_decode`、decode 终态净额预占
`reserve_request_capacity`——净额预占为 wscllm 生产链路真实使用；净额回补
`extend_request_capacity` 仅增量记账、不触发逐出收敛（resident→reserved
1:1 换位保证不超订）；**不新增触发位置，完成路径零逐出**）：

1. 候选池 = 本实例上"已完成（`last_completion_ns` 非空）且非 active"的驻留会话，
   **不按 human/tool 分类**（无 `next_request_type`、请求 CSV 维持 8 列）；
   排序键 `(last_completion_ns, session_id)` 升序 = 纯 LRU；触发请求自身会话受保护。
2. **阶段 1（逐后半层）**：按 LRU 序把 LOCAL 会话的后 ⌊L/2⌋ 层 remote_store
   （`_evict_suffix`）→ PARTIAL；**每逐一笔立即逐 NPU 重查水位，够即停**。
3. 阶段 1 全部候选半层化仍不足 → **阶段 2（整体外迁 = 唯一回退）**：按同一 LRU
   序把驻留前缀 `[0, resident_prefix_layers)` 整体 remote_store（`_evict_session`）
   → REMOTE、清 `instance_index`；逐笔重查。
4. 两阶段耗尽仍不足：记 `deep_gap`（int 计数 + KVCacheEvent）后按 wscllm 语义
   **队头阻塞等待**（容量纪元唤醒后重试，不改投其他实例）——不搬 sh 的
   "耗尽即 raise"。
- **无类型化、无消融档位**（不设 `no_tiered_eviction` 类开关，逐出行为唯一：
  两段式，整体外迁是唯一回退）；**无 R_kv 保留水位**（`kv_reserve_context_tokens`
  维持 manifest-only）；**无历史截断**。
- 守卫：D4-I1 受害者资格复核（逐出后重验受害者确已非 active/非保护）+
  D4-I3 过度逐出守卫（撤销最后一笔必须使至少一个受影响 rank 回到不满足，
  否则不逐）；`SH_STRICT_KV_INVARIANTS=1` 全量审计（三态 location 白名单 /
  层域断言 / REMOTE 无实例无驻留层 / 逐 rank expected_resident+expected_remote
  重算 / 预占一致性）。
- 终局会话 `retire_terminal_session`：层域减账 + 远端账面静默核销（不发传输、
  **不算逐出**）。

**wscllm 特有机制（保留并随三态扩账）**：

- **净额预占 + extend 回补**：prefill 准入在静态 decode 目标预占终态 KV 时，若
  全量预约因本会话旧驻留重复计入而 deep-gap、且旧 KV 仍驻留于该 decode 实例
  （LOCAL 全额 / PARTIAL 取片上前缀账面 `local_shard_bytes`），则按净额
  （终态−旧驻留账面）重试；`prepare_history` 迁移删除旧驻留后经
  `extend_request_capacity` 回补到全量（防抢占语义保持，**resident→reserved
  1:1 换位，任何瞬间不超订**——语义与原仓一致）。
- **容量纪元纪律（capacity_epoch）**：凡产生真实容量变化的 mutation（含三态化
  新增的 suffix/full 逐出与远端回迁落账点）才 bump `_note_capacity_change`，
  多源唤醒**合并覆盖**（净额失败/非净额失败/admission_blocked 释放/成功准入/
  decode 侧五条既有路径逐一接齐；丢中间一次 epoch 会楔死等待重试的准入）；
  extend（仅增预占、可用只减）与无 mutation 路径不 bump。
- **KV delta journal（schema v2）**：`SessionKVCacheManager` 每次公开 mutation
  构成一个事务（`_journal_transaction` 装饰器，**覆盖全部新 mutation**——
  `_evict_suffix`/`_evict_session`/远端回迁/retire 核销均经装饰的公开方法落账）；
  逐 rank delta 流式追加 `results/kv_delta_journal.jsonl`，**v2 行含
  `remote_delta_bytes` 列**、before/after 快照扩 `remote` 账面；run 末 checksum
  门（fail-closed）断言终态守恒 **resident=0 ∧ reserved=0 ∧ physical=weight ∧
  remote=0**（retire 核销后远端池账面与会话账本同归零），产物
  `results/kv_delta_journal_checksum.json` 五项：manager_state_match /
  physical_equals_weight / remote_account_zero / residual_reserved_zero /
  residual_resident_zero。

**远端共享池与物理链路**（远端池**无容量上限、无自身置换**；边缘端口严格 FIFO
单事务 `耗时 = λ_rem + bytes/B_rem`；逐 source/target rank 取**最近边缘端口**
——曼哈顿跳数最小、平局取编号最小；NoC 段走既有 XY 维序确定性路由）：

| 链路 | 发射序与 HBM 计费（每字节恰一次） |
|---|---|
| remote_store 链 A（source≠edge） | [可选 1B trigger] → source `comm_send(bytes)`＝源端 COMM_READ 唯一数据计费 → edge `comm_recv`（`hbm_charge=false` 过路零计费）→ edge `mem_store`（池写本地零计费，仅池端口 FIFO）→ 1B ack 双端；**源端收到 ack 后才物理释放** |
| remote_store 链 B（source==edge 直连） | edge `mem_store(bytes, hbm_access_mode=1)`＝POOL_READ 唯一计费 + 池端口 FIFO 双异步 join |
| remote_load（回迁） | [1B control/arm] → edge `mem_load`（仅池 FIFO）→ edge `comm_send`（`hbm_charge=false`）→ target `comm_recv`（`hbm_charge=false`，目标写由 restore 承担）→ target `local_hbm_kv_restore`（节点级 `is_local_hbm_kv_restore=true`）＝RESTORE 唯一数据计费 |
| noc_migrate（LOCAL 跨实例 / PARTIAL 前缀迁移 / decode P→D 交接） | 既有 1000/3000 类 p2p 配对迁移（相对 TP 编号一一配对），两端正常 COMM_READ/COMM_WRITE 计费 |

本拓扑 9×6 网格的 decode 实例居内部，逐出源多为非边缘 rank → 借道链 A 为主
（直连链 B 仅当逐出源 rank 本身在 mesh 周界时发生；POOL_WRITE 路径 sh 发射端
本就不使用）。KV 传输 tag 由 `TransferTagAllocator` 从 **100,000,000** 起单调分配
（错开既有 `queue_index*10000+{1000,1900,3000}` 段；H8 修正 2026-09-24：原基址
10,000,000 只对 <1000 行队列成立——30s 窗口源 trace 实测 1177 行已段重叠，基址
上移后队列 ≤9999 行两段不相交，且 `_stage_tag` 补了越界 fail-closed 守卫）；新节点命名避开
`first_token` / `batch_train_` C++ 名字锚点子串。

**逐出旁路支链（2026-09-13，KV 逐出与 request 推理并行化）**：三个逐出发射点
（`history_evictions` / `prefill_evictions` / joiner `decode_evictions`，含净额
预占路径经 decode 段的发射）经 `online/graph_batch_builder.py` 的
`_emit_side_branch`（全 rank `chain_checkpoint` fork → 发射 → `restore_chain`，
分支**不 join**）fork 到旁路分支——逐出物理链不再阻塞主链任何节点（回迁/屏障/
列车体），同 rank 上逐出流（COMM_READ）与推理计算（COMPUTE）时间重叠，HBM
争用由 C++ `LocalHbmBandwidthModel` 的 **N-way 均分模型自动在线裁决**（C++ 零
改动，逐流加入/离开即时重分配）。触发门语义不变：history 逐出的到达/interval
门、decode 逐出的 prefill 段末门仍挂在分支首节点（只对齐逐出**开始**时刻，
去掉的是完成阻塞）；`prefill_evictions` 无显式门，支链根 = fork 时各 rank
主链 frontier——fork 点可能合法携带的主链 armed 依赖（turn-0 准入形态的
到达/interval 门 arm，2026-09-13 勘误证实的既有形态）由 helper **暂存清空、
不进分支、恢复后归还原主链消费者**，分支自己的触发门 arming 在分支内完成
并被消费（发射后滞留即泄漏 fail-closed，四仓统一契约）；joiner 迁移
（transfer 3000）与 join 标记保持主链。逐出节点随既有准入/列车批发射（无独立
逐出批、无 watch）。

- **瞬态双占用窗口口径**：并行窗口内被逐旧 KV 尚未物理离开本地 HBM 而新
  request 的 KV 正在写入——容量权威在 Python 台账（决策时刻记账，窗口上界 =
  该 rank 在飞逐出字节数），C++ 无容量强制、指标口径不变；论文引用不得把窗口
  期台账值当物理占用。
- **store→restore 前递依赖（唯一新增正确性边）**：支链化后"同会话逐出池写先于
  其下一轮池读"的传递性失效，发射侧以 `pending_store_tails` 登记表（按会话登记
  在飞 store 支链的边缘 `mem_store` 尾部；两段式 suffix/full 各一条）+ 回迁发射
  统一入口（非 partial 全量 `remote_load` 与 PARTIAL 流水 suffix 恢复）发射前补
  边：同缘直接 arm 依赖（store `mem_store` 完成挂回迁链首）；跨缘 1B p2p 中继
  （store 边缘在池写完成后发 1B、回迁边缘收 1B 后其池读链在其后，tag 走
  `TransferTagAllocator`）。粒度 = 池写落盘（不等源端 ack 全程）；懒处理（store
  早已完成时补边即刻满足零时延）；terminal 会话终结时清登记。

**恢复六分支**（下一轮到达、映射照常选点后——**映射不看 KV 位置**——
`prepare_history` 按态分流；**RECOMPUTE 已从历史路径删除**——恢复取代重算，
`remaining_chunks` 不再有 `ceil(history/p_chunk)` 项）：

| 会话态 × 实例关系 | 动作（决策日志 action） |
|---|---|
| 无历史 | 建片上记录，随 prefill 增长（NO_HISTORY） |
| LOCAL 同实例 | 零开销本地复用（LOCAL_HIT） |
| LOCAL 跨实例 | 整份 KV NoC 迁移（NOC_MIGRATE，既有 1000 类路径原样保留） |
| PARTIAL 同实例 | **只回迁后缀层段 + 与前缀计算真流水重叠**（REMOTE_LOAD）：首 chunk 拆前缀层段/后缀层段，后缀段以远端恢复的逐 NPU 完成门为依赖，恢复流量与前缀计算重叠；拆分发射字节与单次全层发射严格守恒 |
| PARTIAL 跨实例 | 两段链（PARTIAL_MIGRATE）：前缀 NoC 迁移 + 目标实例容量逐出 + 后缀远端恢复；**两段必须序列化在同一条 prefill 决策记录的 `history_transfers` 列表内**（watermark 校验"同请求同 kind 两次即 fail"） |
| REMOTE | 全量回迁到选定实例，逐 target rank 最近边缘端口（REMOTE_RESTORE） |

结构注记：PD 分离下会话经 P→D 交接驻留 D 实例、下一 turn prefill 恒选 P 实例，
PARTIAL 恢复实际走 PARTIAL_MIGRATE（跨实例两段链）；REMOTE_LOAD（同实例后缀
回迁）分支已实现并有专项单测覆盖，真实 trace 无需期待该动作出现。

**KV 指标观测（契约方案 A）**：`repo_variant` 维持 `"astra-sim-wscllm"` 不变；
`kv_hit_state` **五值域不扩**——REMOTE→`full`、PARTIAL→`partial`
（`full_local` / `full_remote` 的区分由 `kv_hit_states.csv` 的 evidence 列承载，
如 `history_location_before=remote_memory`）；命中率分子 `hit_n = full + partial`
公式不变；`slo_tools/kv_cache_adapter.py` **双级解析**——新产物优先
`history_location_before` 三态映射，字段缺失回退 legacy `history_action` 四值表
（旧产物兼容）；逐出/恢复消费契约行（`history_transfers` 逐段对象 +
`history_evictions`/`prefill_evictions`/`decode_evictions`/`completion_evictions`
契约逐出行；legacy 镜像行不再消费防双计；`total_bytes` 标量回退使逐出条目不静默丢）；
`hbm_watermark.py` 接受 journal **schema v1/v2 单文件不混版**，v2 remote 链参与
行级自洽校验且**不计入逐 rank physical 判决口径**（远端池字节不占本地容量）、
certified 层守恒扩五项；检测到 `history_transfers` 在场自动升级 S2 逐段对账重放
（PARTIAL 两段式恢复不再混成标量；逐出条目 report-only + admission 时点
location 对账，决策日志重放为上界/对照列——journal 在场时权威占用=journal 重放）。
`request_metrics.csv` 冻结列不动（kv_hit_state 列维持 postprocess 填 NA，真值在
kv_hit_states.csv）。

## 2. 快速开始

```bash
cd <本仓根>
# ① 构建（首次需先配置；构建树已预置时可跳过 configure）
#    裸仓无 build/：先放拼装 CMakeLists（上游 astra-analytical 仓的 vendored
#    等价物，三行 add_subdirectory 聚合 AstraSim 库/analytical 后端/前端）：
#    mkdir -p build/astra_analytical && cat > build/astra_analytical/CMakeLists.txt <<'EOF'
#    cmake_minimum_required(VERSION 3.22)
#    project(AstraSim_Analytical_Build)
#    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../ astra_sim_build)
#    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../extern/network_backend/analytical backend_build)
#    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../astra-sim/network_frontend/analytical frontend_build)
#    EOF
cmake -S build/astra_analytical -B build/astra_analytical/build_congestion_aware -DBUILDTARGET=congestion_aware -DNETWORK_BACKEND_BUILD_AS_LIBRARY=ON
cmake --build build/astra_analytical/build_congestion_aware -j

# ② 物化输入（唯一允许源 = agent-traces/tracelab/astra_compute_20.csv 前 2 秒，
#    arrival_time < 30e9 ns；物化器：traces/derive_20_first_30_seconds.py（运行 stdout 即权威 provenance 记录））
#    产物放 sh_test_mesh/workload/llama2_7b_inference/traces/，
#    并把 trace_config.csv 第 12 行 request_queue_csv 指向它
#    （裸仓交付态该指针为占位串，用毕还原；kv_cache_policy=session_lru_tiered
#    为本仓交付态值，无需改动——旧值 session_lru_recompute 保留仅为兼容读取）
#    物化器 CLI：[source] [queue] [sidecar] [window_ns] [arrival_scale]；
#    arrival_scale>0 仅缩放 turn-0 session_arrival_time_ns（t0/scale，即
#    负载 ×scale），inter_request_interval_ns（human/tool 外生等待）不动，
#    窗口判定与统计始终用未缩放源时间——scale=1 时 8 列队列与冻结基线
#    逐字节一致；缩放与 request_type 等新信息只进 canonical sidecar 与
#    stdout provenance，不进队列。
#    P0 turn-0 修复（2026-08-30）配套：物化器同时写出 <queue>.provenance.json
#    （FNV-1a 64/字节数/行数/会话数/turn-0 min-max/文件序相邻逆序/块连续）；
#    C++ 侧 WindowedTraceReader 首开即流式复算并与边车逐字段比对，任一不匹配
#    在任何 Submit 发生之前 fail-closed 退出（边车缺省仅记一行、不设门）。

# ③ 生成 plan 目录（runtime_config 四小件 + manifest + metrics_manifest）
cd sh_test_mesh/workload/llama2_7b_inference && python3 plan_materializer.py && cd <仓根>

# ④ 跑③④（GEN_MATCH：generated/ 下须恰一个 llama2_7b_wsc_llm_inference_54npus_* 目录）
bash sh_test_mesh/run_scripts/run_online_strategy.sh <run_dir> <绝对路径 request_csv>
bash sh_test_mesh/run_scripts/run_online_strategy_sensing.sh <run_dir> <绝对路径 request_csv>

# ⑤ 指标后处理 + ④对账
bash sh_test_mesh/run_scripts/run_metrics_postprocess.sh <run_dir>/cpp.log
#    （在线 runner 已自动后处理：raw/normalized_metrics.csv；full 档额外产
#    request_metrics.csv——逐请求冻结 29 列，与 manifest 按 queue_index/
#    request_id fail-closed 连接；summary/off 档不产该文件，原因写
#    postprocess.log）
python3 sh_test_mesh/workload/llama2_7b_inference/online/verify/ledger_reconcile.py --bridge-dir <run>/bridge --manifest <ET_DIR>/manifest.json（期望值缺省 1177/112，异窗口传 --expected-requests/--expected-accepted-sessions；详见 --help）

# ⑤b 自动 SLO 指标提取（P3，2026-08-28；A4/2026-08-29 单遍化）：在线
#    runner（strategy/sensing 两变体）在 postprocess 成功后、归档前自动
#    调用 run_slo_postprocess.sh——内部一次调起 slo_tools/
#    slo_postprocess_driver.py 单进程单遍驱动（9 步产物集/行序/公式/
#    slo_postprocess.log 与逐工具串行逐字节一致；读放大收敛：
#    request_metrics.csv 4 读→1、decision log 4 流→1、[METRIC] init 探测
#    4→1、解释器启动 9→1；七个工具 CLI 保持可独立调用）
#    ——full 档产 9 步最细粒度产物（slo_e2e_stats/backlog/session/
#    load_imbalance/restore_decomposition/hopbytes_total+per_request/
#    hbm_watermark 三件=hbm_intervals（权威 RLE 变点区间）+hbm_plot_
#    series（行预算绘图产物，旧 hbm_watermark_series 已退役）+
#    hbm_watermark_instances、cache_events+kv_hit_states、
#    slo_warmup.json；P1/2026-08-30 起 hbm_watermark 按四层可信度分级：
#    run_dir 含 results/kv_delta_journal.jsonl 时走 journal 权威重放，
#    含 checksum 证书为 per_rank_total_hbm_certified 层——正式逐 rank
#    容量判决、违规 exit 3；缺 journal 的旧 run 为 upper_bound_only
#    上界层，超限只诊断不认证、exit 0），一律不传分桶/聚合参数（粗化
#    留给下游画图脚本）；
#    非 full 档依赖 request_metrics.csv 的子命令按设计跳过并写说明；
#    train_ledger 缺失时跳过 load_imbalance，同样写说明、不算失败。
#    env SH_SLO_POSTPROCESS：1=默认 warn（子命令失败只写
#    slo_postprocess.FAIL 标记，不推翻仿真结果）、0=整步跳过、
#    strict=失败即 runner 非零退出。runner 同时把 per-request manifest
#    （metrics_manifest.json/manifest.json）拷入 run_dir 根（P2）——run_dir
#    自包含，与仓还原状态解耦
# ⑥ 一键清空测试记录 + 编译产物（还原裸仓库 = 无物化输入 + 无 build/）
bash sh_test_mesh/run_scripts/clean_test_records.sh      # 清物化输入/运行产物/缓存（含 trace_config 指针回占位）
bash sh_test_mesh/run_scripts/clean_build_artifacts.sh   # 清编译产物（build/）
#    （或一步到位：clean_test_records.sh --full）
```

## 3. 两条仿真路线

| 路线 | runner | 时钟 | 产物 |
|---|---|---|---|
| ③ strategy 关感知 | run_online_strategy.sh | 真实物理 | 决策日志/metrics（digests 默认关，见 §3.1.1；成功后自动瘦身归档） |
| ④ strategy 开感知 | run_online_strategy_sensing.sh | 真实物理 | ③产物 + ledger.jsonl/感知日志（对账用） |

PASS 判据：completed == 物化请求数、no_decision=0、single_node=0、
delivery == graph_batch 数、③④ 决策日志逐字节一致（感知只开仪表不改判据）。

指标档位（B1/WP0 起）：两个 runner（strategy/sensing）的 `--metrics-detail` 不再硬编码
summary——优先级 env `SH_METRICS_DETAIL` > 本仓
`sh_test_mesh/workload/llama2_7b_inference/metrics_config.json` 的
`detail_level`（off|summary|full）；env 与 json 均非法/缺失时 fail-closed
立即退出。full 档后处理额外产出逐请求 `request_metrics.csv`（列集冻结，
后续工作包只填 NA 占位不加列；request_type/prefix_len 由物化链路从
canonical sidecar 透传进 manifest.json）。

SLO 观测层（B2 起，WP6/WP8/WP9-C++）：`metrics_manifest.json` 新增 `slo_sampling`
节（`watermark_period_ns`/`link_bucket_ns`/`provisional`；plan_materializer 从
`sh_test_mesh/slo_tools/slo_params_manifest.json` 取 value（B4 已填推导值：
watermark 5,000,000 ns / link_bucket 20,000,000 ns），null/缺失时回退文档化
临时锚点 5,000,000 ns 并置 provisional=true；C++ 侧缺节同样回退且
每条相关记录回显实际周期）。新增 `[METRIC]` 记录类型：
- `hbm_watermark` / `hbm_watermark_summary`（WP8，finalize 侧重放 planner 账本，
  零运行时开销）：per-instance 逐桶 resident/committed 峰值、per-rank
  峰值/timeavg/容量违规计数（违规数必须为 0，>0 照实输出并计 consistency
  violation；timeavg 与 capacity_timeavg 交叉差 >1% 计 violation）。桶记录仅发
  "有活动的桶 + 首末桶"；在线合成 manifest 无 planner 账本时输出全零序列并
  注明原因（既有现状，真实水位线由离线脚本从 ledger.jsonl 重放补充）。
- `link_bucket` / `link_total`（WP6，FluidScheduler 只读积分，不改流推进）：
  每桶 max/total bytes 与 mesh 边界链路峰值、每链路 total_bytes/active_ns。
  开关 env `ASTRA_LINK_OBSERVER`（metrics≠off 且值≠0 时启用，设 0 关闭；off 档
  零工作零记录，关时不注册任何回调）。
- request 记录新增 `first_token_ns`（WP9 事件码 8：python 侧首 token 标记节点
  名含 `first_token`，C++ 按 (request_id, rank) 注册 complete 边锚点、跨 rank
  取 min tick；无事件输出 null 并注明；不变量 arrival ≤ first_token ≤ completion、
  decode_length==1 的相等不变量已按 WP9_CONTRACT §6（2026-08-27 主控裁
  决）放宽：code 8 跨 rank 取 min、completion（code 4）取 max，TP 斜台使二者
  天然不等，故仅保留序检查，另发信息性字段
  `first_token_completion_skew_ns`（=completion−first_token，仅 dl==1 且
  两者非空时输出；不再 withhold、不计 violation）。manifest 无
  decode_length 时该字段不可评估并注明 manifest_decode_length_missing）。

SLO B3 集成验证落地（2026-08-27）：

- `slo_tools/load_imbalance.py`：跳过 WP9 首步批台账行
  （`first_step: true` 的发射边界记录非终态 drain；真实 drain 在余量批
  行）——否则 SPLIT=ON 产物触发"每请求恰一次 drain"fail-closed。五仓
  逐字节同文由主控同步。
- `slo_tools/tests/run_golden_live.py`（新）：T2 golden G1-G4 运行版
  ——hand-craft 队列+sidecar（`traces/golden_g{1..4}_request_queue_
  recompute.csv`）→ 真仿真（full+SPLIT=1）→ 手算断言（分解和恒等、
  queue_ns≈占位者 prefill（实测 ratio=1.000）、T_session 恒等、
  request_type 透传、kv_hit_state 手推、restore 三段和=总时长；W 仓
  G4 三轮全部产生锚点）。
- `traces/llama2_7b_wsc_llm_inference_first_60s_canonical_sidecar.csv`
  刷新为当前 derive 输出（15 列含 human/tool/request_type；队列字节
  不变 sha256 78a56480…，B0 基线可比性不受影响；60s 窗 request_type
  透传 60 human/1350 tool/44 unknown=源数据空 human/tool 时间的忠实
  回退）。
- `sh_test_mesh/tests/test_metrics_contract.py`：B3-7 增补（事件码 8
  常量/edge/SERVICE_EVENT_CODES、request_metrics 29 列 frozen、
  terminal_status/first_token_source 枚举、NA 语义用例；五仓逐字节
  相同）。

WP9 退回处置：拆分缺省关 + train-interpolated proxy（B4，2026-08-26）：

- **缺省翻转**：`SH_FIRST_TOKEN_SPLIT` 缺省 `1` → `0`
  （`online/graph_batch_builder.py::first_token_split_enabled`，显式
  `=1` 仍可启用拆分取 exact 首 token，研究/对拍用）。理由：B3_W 60s
  决策等价门-2 失败——拆分物理扰动（首列拆批 → tick 漂移，row 3 起
  +11.9us）在闭环逐轮放大，t=2.736s 同 tick 处理顺序翻转并级联
  （1139/1454 请求实例选择翻转、train_id 多重集不等 OFF 22945 vs ON
  24275 行），ON/OFF 字节等价在 60s 窗不可达（2s 窗保持通过）；调度
  器 tie-break 属禁改区。证据：`/tmp/slo_wps/gates/B3_W.DONE`
  （gate2 FAIL）、`/tmp/slo_wps/b3/W/`（t3_full_off_split vs
  t3_full_split_on 对拍）。拆分语义测试缺省断言同步翻转
  （`online/test_train_machinery.py::SplitEnvironmentTest` 钉缺省关；
  `FirstTokenPlanTest` 维持显式 "1"）。
  **研究专用**：`SH_FIRST_TOKEN_SPLIT=1` 的 exact 口径仅供机制研究（60s 决策等价门未过，证据见仓内 README 所引门文件）；论文指标一律采用默认 proxy（`train_interpolated`）口径，proxy 不得进入任何 SLO 判定路径。
- **proxy 填充**（`metrics_postprocess.py`）：exact（事件码 8）缺失且
  归档 `results/train_ledger.jsonl` 可得时填
  `first_token_proxy = decode_start_ns + (w₁/Σᵢwᵢ)×(first_train_end_ns − decode_start_ns)`，
  `wᵢ = W_bytes + KV_bytes(context+consumed+i)`，`first_token_source
  = train_interpolated`：
  - **Σ 域 = 首列车 iterations N**（i=1..N，与 S3 裁定一致）：debut
    首 token 随列车第 1 迭代完成，权重按列车 N 迭代线性化（对 debut
    自身参与度 P 求和会在 decode_length<iterations 时退化为 share=1
    并越过 completion，S3 实测 7/1454 越界）；debut join 时 consumed=0
    （sticky decode，60s 参考跑 1454/1454 恰一次 joiner 加入）。
  - **first_train_end = 同实例下一列车发射边界（wscllm 实际结构）**：
    PD 分离下台账行全部是 D 实例 decode-only 列车（P 侧整段发射无
    列车概念）；台账行 tick 是发射边界，D 实例 busy 门 = 一列车在飞
    （in_flight_train），故同实例下一行 tick 即 end barrier 后首个
    决策边界；哨兵列车（T_max 截断无 exit 成员，本仓多数行）与普通
    行同等处理；实例末列车 → NA（不可得不编造，60s 重跑 1/1454：
    qi_720 的 debut 恰加入实例 2 的末列车）。
  - **N=1 退化列车钳制**：share=1 → proxy=下一发射边界 > completion
    （边界晚于 end barrier tick），钳到 completion 并在 instructions
    留痕 `clamped_to_completion`（60s 重跑 2/1454，均为 decode_length
    =1——exact 口径语义本就是 first_token==completion）。
  - **换算权威**：W_bytes/KV_bytes 用本仓
    `wsc_llm_scheduler.estimate_model_weight_bytes` /
    `kv_cache_bytes_for_tokens`（trace_config.csv 参数：swiglu →
    13476831232 / 524288 B/token，由 test_wsc_llm_scheduler.py 的
    test_llama2_7b_tp6_partition_is_exact_without_model_padding 冻结；
    与 face_scheduler 同源同构），不另行编造。
  - **运行内 CSV 保持 NA**（ledger 归档前不可得，instructions 记
    `proxy_unavailable:no_train_ledger(results/)`）；对归档 run 离线
    重跑 postprocess 填充——运行产物与离线分析两段语义。
  - **SLO 防线**：proxy 仅展示口径，`first_token_ns/first_token_source`
    被 `slo_common.assert_no_proxy_columns` 从一切判定路径 fail-closed
    拒绝（`slo_tools/tests/test_slo_contract.py::
    test_train_interpolated_proxy_cannot_enter_judgment`：夸张 proxy 值
    不改变 violation verdict，判定输入携带 proxy 列被拒）。
- **离线验证**（`/tmp/slo_wps/b4/W/proxy60/`，源=B3 归档
  t3_full_off_split + 重物化 manifests，plan 摘要 056c8582 与运行时
  一致）：1454 行 = train_interpolated 1453 + NA 1（末列车例）、
  0 序违例（填充值过 fail-closed ordering 检查）、raw/normalized CSV
  与归档逐字节相同；抽 3 请求手算（manifest+ledger+cpp.log 一次产物
  独立重算）全部吻合，含 1 例 N=1 钳制；proxy vs exact
  （t3_full_split_on）分布差 p50 −5.4%/p99 −11.4%（信息性：两时间线
  已因门-2 级联分叉，属预期）。
- 单测：`sh_test_mesh/tests/test_first_token_proxy.py`（新，8 用例）；
  slo_tools 契约 41/41；metrics 契约 35/35；拆分/列车机制 11/11。

WP9-python 首步批拆分（B2wp9py，2026-08-26）：总开关 env `SH_FIRST_TOKEN_SPLIT`
（B4 起缺省 `0` 关——60s 决策等价门-2 失败退回 proxy，见上节；显式 `1`
开启；关闭=与拆分上线前逐字节一致）。开启时含 debut 成员
（`decode_tokens_consumed==0`）且迭代数 ≥2 的 decode 列车两段式发射——首步批 =
joiner 迁移/join 标记 + 各成员第 1 个 span（weight_passes=1）+ debut 成员
first_token 标记 + 批命名空间唤醒标记（`<train_id>_first_step`，哨兵同款通道，
fire 后调度器 no-op 剥离并发射余量批）；余量批 = 剩余迭代（weight_passes=
iterations−1）+ exit/哨兵/end barrier（挂点语义不变）。P 侧 `emit_prefill_batch`
整段路径与 PD 静态映射语义零触碰。拆分新增产物行带 `first_step: true` 标记
（digests/train_ledger，ON/OFF 对拍剥离）；2s A/B：决策内容等价（仅物理漂移
tick 差，相对 ≤6.6e-5）、GraphBatch 增量 = debut 请求数。metrics_schema 事件码
8 常量 `EVENT_FIRST_TOKEN_COMPLETE`（complete 边、service 码集）。prefill 决策
对 history NOC_MIGRATE 迁移附加 `noc_hops` 输出字段（实例图最短路，Hop-Bytes
history 迁移纳入覆盖（原记「70.4%→满覆盖」；2026-09-24 M29 修复起 hopbytes
另将逐出/回迁契约行计入 bytes_without_hops——契约行无 shard 路由，coverage
如实低于 1，满覆盖口径已撤）；基线对拍剥离项）。

### 3.1 在线二进制机制旗标（--online-* 家族）

`AstraSim_Analytical_Congestion_Aware_Online` 显式解析 `--online-*` 家族
（OnlineCli.hh 契约；家族内未知旗标硬错）。机制类旗标一枚：

| 旗标 | 取值 | 缺省 | 语义 |
|---|---|---|---|
| `--online-validate` | `0\|1` | `1`（全量） | **C1（2026-08-28）；B.2 清除（2026-09-05）移除 N 抽检档**：GraphBatch 提交前全量校验开关。`1` = 每批校验（改前行为，裸调用的 fail-closed 缺省）；`0` = 生产快速路径（跳过校验，commit() 的强制活性预检保留）。取值域为严格 `0\|1` 枚举，`N≥2` 或其它任何值启动即硬错误（杜绝陈旧脚本静默改变校验频率）。只影响 validate 计数/诊断（graph_validate_ns 等白名单字段），不影响已提交状态。生产 runner 默认传 0（env `SH_ONLINE_VALIDATE` 可覆盖），冒烟/fixture/verify 脚本显式传 1。 |

`--online-node-gc` 旗标已由 **B.3 清除（2026-09-05）** 整支移除：M2 节点 GC
摊销回收恒开（原 `0` 臂 = pre-M2 永不删除应急回退臂已删，现仅 fixtures 内部
Context 开关可达），任何 `--online-node-gc` 调用按家族未知旗标规则启动即
硬错误（`unknown online-family option`），陈旧脚本无法静默禁用 GC；决策序列
与全部工件不受影响（历史生产调用恒为默认 1）。

runner 脚本（strategy/sensing 两个（装配口径相同；C++ 开关同构））显式传
`--online-validate "${SH_ONLINE_VALIDATE:-0}"`；fixture runner（idle/
wakeup_guard/same_tick_milestone）显式 `--online-validate 1`。运行期证据：
cpp.log 启动行 `[online] node gc: ...` / `[online] graph validate: ...`、
结束行 `[online] node gc: erased=... retained=...`。

停泊看门狗与桥超时（P0-2，2026-08-31；sync-A16 批次4 2026-09-01 以 face
批次P 后交付版为母本同步 FP1 数值/解析加固；**wscllm 范围裁决**见下）：

- `--idle-watchdog-s <秒>`：墙钟停泊看门狗，缺省 `1.0`=武装（2026-09-05：
  静默楔死 ~1s 即 fail-closed abort，不再无声挂死；官方 CSV run 到达全量
  预排为队列事件、健康运行不停车不受影响。显式 `0`=关恢复 `wait_for_work()`
  原无界契约——IDLE fixture 等刻意长停车场景必须显式传 `0`）；开时停泊点
  墙钟超时即带停泊点诊断
  （tick/active/pending_alarm/window_occupancy/mailbox 等 11 字段——2026-09-24
  修复删 pending_fence 输出后由 12 减 1）
  `online_fatal` abort（`--idle-` 前缀同受家族未知旗标硬错保护）。
  **FP1（2026-09-01，sync-A16 批次P）**：数值合同冻结——token 不得含任何
  空白或符号字符（拒 `" +1"`/`" -1"`）；判界唯一顺序为 `==0` 接受（=关）
  → `(0,1e-9)` 拒（"低于时钟分辨率"，常量
  `kMinIdleWatchdogSeconds=1e-9`）→ `>1e9` 拒（平台上限，常量
  `kMaxIdleWatchdogSeconds=1e9`，两端点本身接受）；ERANGE 上下溢均拒。
  运行期改为 `ServiceCoordinator::checked_wait_deadline`（纯函数：tick 域
  判界→转换→加法前判界→单次 deadline）+ `wait_for_work_until`（绝对
  deadline，协调器内不再二次 `now()+timeout`）；主循环单次取 now、单次算
  deadline；`0`=关时必须走原 `wait_for_work()` 阻塞等待。
- **A2 input-open 死端分支不适用（wscllm 裁决，合同 §2.1/P1）**：sh/face 系
  的"input 开 + 窗口有占用 + 无 pending alarm"可证死端形态在本仓 calendar
  reader 下不可达——①泵送与 turn-0 提交次序：reader 先建全量 arrival
  calendar，turn-0 按 arrival 序提交，泵送在 drain 之前/之后的停驻形态与
  calendar 不变量互相闭合；②分类点处 `!csv.empty() && !eof() &&
  pending_alarm==0` 的机器状态互斥（cursor 停驻形态下 pump 与二次 drain
  次序保证任一未触发 turn-0 必持有 pending alarm 或尚未入 calendar，二者
  不可同时为空）；③Error 终止路径（fail-closed）先于停泊发生。**未来重构
  若打破 pump 后置 drain 次序或 calendar 完整性不变量，必须重做可达性
  分析**，在此之前不引入该分支。停泊兜底统一交 `--idle-watchdog-s`
  （含任何未来未知停滞形态）；11 字段 `parking_diagnostics` 报文已随 A3
  移植（`window_occupancy` 在本仓语义=已提交未触发 turn-0 计数）。
- **window advisory 旋钮已删除（2026-09-05 A.3 清除；原 2026-08-30 P0 fix
  延续裁决）**：`--request-window-rows` 死旋钮已从四仓物理删除——calendar
  reader 按 arrival 序提交，窗口值从不约束读取与提交，删除不改变任何行为；
  现传入该选项即 `unknown online-family option` 硬错误（fail-closed），
  runner 不再透传 `SH_REQUEST_WINDOW_ROWS`。
- **FP1 整数解析加固（2026-09-01，sync-A16 批次P）；B.2 清除（2026-09-05）修订**：
  `--request-max-arrival-ns` / `--bridge-timeout-ms`
  统一"纯 ASCII 数字词法（拒
  `" -1"`/`"\t-1"`/`"+1"`）→ `errno=0` + ERANGE 拒 → endptr 到串尾 →
  目标类型上限（前两者 size_t/uint64_t、bridge-timeout-ms
  另加 `<= INT_MAX`）→ 才转换"合同，失败不部分写入
  `out`。`--online-validate` 原同属该整数词法族
  （`--online-validate=4294967296` 曾会回绕为 0 静默关闭图验证，fail-open）；
  B.2 清除后改为严格 `0|1` 枚举解析，任何其它值（含抽检 N≥2 与越界大数）
  启动即硬错误。
- runner 透传 env：`BRIDGE_TIMEOUT_MS` 三态——未设=缺省 120000（须大于
  负载最慢单决策与 Python 侧 FIFO 开启等待）、显式 `0`=永等逃生口、正值=
  该毫秒值；它只武装 C++ 桥 response poll（Python 单次交换停滞族，含
  Python 启动即死形态），管不到停泊族（由 `--idle-watchdog-s` 兜住）。
  **A4 缺省收紧声明（P0-2，唯一行为收紧）**：两个 runner（strategy/
  sensing）未设/空时一律注入 `--bridge-timeout-ms 120000`；逃生口
  `BRIDGE_TIMEOUT_MS=0`；C++ 裸调用缺省仍为 0（永等）不变。

### 3.1.1 运行开销与日志瘦身开关（2026-08-28，A/B/C/D 系列改造）

- **env `SH_ARCHIVE_RUN`**（默认 `1` 开）：成功 run 结束后 runner 自动调
  `sh_test_mesh/run_scripts/archive_run_outputs.sh <run_dir>` 做产物瘦身归档：
  抽 `[METRIC]` 行 → `metrics.log`（常驻）；新运行的桥请求只追加到
  `results/request_journal.jsonl`，不再生成海量 `request_*.json` inode；仅为
  兼容旧运行，若发现旧散装请求才归入 `bridge_requests.tar.gz`。`cpp.log` →
  `cpp.log.gz`（pigz 优先）；results/ 非必需 jsonl →
  `results_extra.tar.gz`。**常驻保留集**：三个 metrics CSV、metrics.log、
  python.log、postprocess.log、`results/request_journal.jsonl`、
  `results/online_decision_log.jsonl`、`results/graph_batch_digests.jsonl`、
  `results/online_stats.jsonl`、`results/profile.jsonl`（后三者存在时）、
  `results/train_ledger.jsonl`、`results/ledger.jsonl`、
  `results/sensing_query_log.jsonl`（仅 sensing 跑产生，对账输入，strategy 跑
  保留集不受影响）、campaign_provenance.json、
  `results/kv_delta_journal.jsonl` + `results/kv_delta_journal_checksum.json`
  （P1/2026-08-30 权威 HBM delta journal 与 run 末 checksum 门产物，常驻
  不压缩；journal 开关 off 时不产生）、
  P2 拷入的 `metrics_manifest.json`/`manifest.json`、
  P3 自动 SLO 提取产物（`slo_*.csv`、`slo_*.json`、`cache_events.csv`、
  `kv_hit_states.csv`、`slo_postprocess.log`、`slo_postprocess.FAIL` 若有）。
  任一阶段失败不归档、
  全量保留供排查（runner 失败路径早已 exit 1）。
- **env `SH_SLO_POSTPROCESS`**（默认 `1` 开，P3/2026-08-28；A4/2026-08-29
  起链内为单遍 driver）：成功 run 在 postprocess 之后、归档之前自动跑
  `sh_test_mesh/run_scripts/run_slo_postprocess.sh`（一次调起
  `slo_tools/slo_postprocess_driver.py` 单进程单遍）提取全套单 run SLO
  指标（9 项最细粒度，清单与失败语义见该脚本头注；归档后的 run_dir 亦可
  幂等重跑——slo_tools 走 cpp.log→metrics.log→cpp.log.gz 回退）。
  `=0` 整步跳过；`=strict` 任一步失败即 runner 非零退出；默认
  warn——失败只写 `slo_postprocess.FAIL` 标记，不推翻仿真结果。
- **env `SH_GRAPH_DIGESTS`**（默认 `0` 关，C2/D2）：`graph_batch_digests.jsonl`
  纯审计产物且每批全量二次序列化，生产默认不写；`=1` 恢复恒写。
- **env `SH_PROFILE_JSONL`**（默认 `0` 关，D2）：`profile.jsonl` 性能剖析
  产物默认不写；`=1` 恢复。decision_log/train_ledger 恒写不动（SLO 指标源）。
- **env `KV_DELTA_JOURNAL`**（默认 `on` 开，P1/2026-08-30）：权威 HBM delta
  journal 开关——strategy 跑在 `SessionKVCacheManager` 装配
  `MemoryActionRecorder`（journal 模式，`online/online_service.py` 在调度器
  构造前 install），manager 每次公开 mutation 构成一个事务（transaction_id
  单调；嵌套公开复用并入外层事务），逐 rank delta 流式追加写
  `<run_dir>/results/kv_delta_journal.jsonl`（append-only、每行完整 JSON+
  flush、行级 crash 一致；`release_request_capacity` 按语义修改①携带真实
  mutation `now_ns`，时序非单调即 fail-closed；`sequence` 为独立单调计数器
  语义修改②，零 delta 记录被跳过时仍连续；每行含 before/after/capacity 三
  类计数快照）。run 末 `verify_run_end → assert_final_state` 链上追加
  checksum 门（fail-closed）：流式重放 journal（sequence 连续/前后快照自
  悽/时序与事务 id 单调/断行报行号）与 manager 终态逐 rank 对账（语义修改
  ③/④：commit 点还有逐事务逐 rank 对账），并断言终态守恒
  resident=0/reserved=0/physical=weight/remote=0（2026-09 三态化起 journal
  **schema v2**：每行含 `remote_delta_bytes` 列，before/after 快照扩 remote
  账面；retire 核销后远端池账面与会话账本同归零），产物
  `results/kv_delta_journal_checksum.json`（行数/sha256/逐 rank 终态表，
  守恒五项 manager_state_match/physical_equals_weight/remote_account_zero/
  residual_reserved_zero/residual_resident_zero）。
  journal 是纯观测旁路——不改任何决策/调度行为（2s 冒烟 journal on/off
  对拍：online_decision_log.jsonl、request_metrics.csv、metrics_manifest.json
  及当时全部 12 项 SLO 产物逐字节一致；P1/2026-08-30 起 hbm_watermark
  产物集扩为三件=intervals+plot_series+instances，SLO 产物共 13 项）；`=off` 完全旁路（不装 recorder、不写
  journal/checksum，行为与改前一致）。这是 journal 的唯一开关，缺省必须 on
  （权威账本与仿真同趟产出，不增加重跑成本）。
- **env `ASTRA_LINK_OBSERVER`**（runner 默认 `0` 关，D3）：在线模式
  link_bucket/link_total 行无消费者（metrics_postprocess 不读、hopbytes 走
  decision log），runner 默认关闭 FluidScheduler link 观测；`=1` 恢复发射。
  同项 D3：在线模式 `hbm_watermark` 逐桶行（全零序列，权威源是
  slo_tools/hbm_watermark.py 的 ledger 重放）不再发射；per-rank
  `hbm_watermark_summary` 保留（含 flat-zero 注记）。
- **C3 桥字节压缩**：Python 侧桥响应 `json.dump` 改紧凑分隔符、C++ 侧
  响应数组改 move——JSON 语义不变，通道字节显著缩小（channel_bytes 等
  桥统计随之变小，属诊断白名单）。
- **C5 [METRIC] 攒批写**：finalize 段记录攒缓冲、1MiB 或 flush 点一次
  `::write`，记录内容与顺序不变。
- **wall_time_ns 真值修复（2026-08-28 收尾）**：`Statistics::wall_time` 原为
  未初始化成员——在线模式 MetricCollector finalize 早于 Workload 的
  `post_processing()` 读取 `get_wall_time()`，rank_compute 记录的
  `wall_time_ns` 一直是逐运行变化的乱数（历史等价口径曾将其列为白名单）。
  现成员零初始化并在 `record_end` 维护运行最大值（与 `post_processing`
  的重算幂等一致），finalize 读到的即真"max operator end_time"；对拍
  口径：除该字段从乱数变为确定性真值外，其余全部字节等价。
- **Python 侧内存/CPU**（B 系列）：committed KV 账本改计数器（B2）、
  sensing_query/online_stats 流式落盘（B3，后者经 `.partial` 两遍合并保
  字节一致）、session KV 事件流增量读取+水位压缩（B1，消除 O(n²) tuple 拷贝；event_index 改单调计数器）、请求完成后 runtime 索引 pop（B4）、
  C++ MetricCollector 锚点按请求回收（A2）与 `start_times` 死重移除；在线
  Statistics 在节点终态后把必要数值折入流式聚合并退役该节点的
  `operator_statistics`，GPU busy 用 active-count 区间积分保持并发并集
  语义，生产内存随在途窗口而非累计节点数增长。显式 microbenchmark 历史
  模式仍保留逐节点记录；在线查询已退役/仍活动节点时 fail-closed，避免用
  不完整历史静默给出错误统计。
- **长跑内存回归（2026-08-29）**：完成 runtime、已确认 response/batch 及
  构图器已收集节点会立即释放；completed ledger 在 sensing 模式逐行流式写出。
  `SCHEDULER_MEMORY_DELIVERIES=1000000 python3 sh_test_mesh/workload/llama2_7b_inference/online/verify/long_run_memory_fixture.py`
  覆盖一百万交付（本地冒烟可降低该环境变量）。完整 `config.request_queue`
  暂不能 cursorize：future alarm 和 GraphBatchBuilder 仍按 `queue_index`
  随机解析未来请求。

## 4. 机制回归 fixtures

`run_online_idle_fixture.sh`（IDLE 五态生命周期）、`run_online_wakeup_guard_fixture.sh`、
`run_online_same_tick_milestone.sh`、`bridge_race_stress_repro.sh`、
`run_bridge_cpp_death_fixture.sh`（C++ 中途死亡 => Python 侧 BrokenPipe
fail-closed 退出 1，纯 Python 无需二进制）——机制层健康自检。

另有 C++ 单测 fixtures（`build/astra_analytical/build_congestion_aware/bin/`，无参数直跑）：
`..._WindowedReaderTest`（P0 turn-0 修复后的日历 reader 单测：索引遍+日历提交/
advisory 窗口/乱序 turn-0/首块超窗/arrival=0 t0 边界/同 tick 按 queue_index 排序/
provenance 篡改与块结构违例 fail-closed/到达审计门自证）、
`..._CalendarOracleTest`（P0 V2 等价 oracle：新日历 reader vs 逐字冻结的修复前
reader 基线拷贝 `tests/LegacyOracleWindowedTraceReader`（window=0 全量臂）在
同一输入上的 (alarm_tick, queue_index) 触发序逐元素相等；可选 argv[1] 传真实
队列 CSV——全量 22,816 行 TraceLab 队列实测两臂各 496 条全等、且新臂
late_static_submit=0），
`..._LocalHbmBandwidthModelTest`（本地 HBM 带宽竞争数值/join 两序/回退，见 G 节）、
`..._LocalHbmModelTest`（6 作业 N-way 模型权威数值测试：均分/回满/双序 join 各恰一次/
hbm-charge=false/中转不计费，2026-09 数据面激活自 sh_2.0 拷入）、
`..._NodeStoreTest`、`..._AlarmCancellationTest`（可取消 alarm 链路：bucket 清空时
outer alarm 从 backend 物理移除、共享 bucket 级联、重复取消幂等、legacy 后端回退
stale guard）、`..._MetricOneShotEraseTest`（MetricCollector one-shot node bucket
擦除 + OnlineNode anchor 快路径标志，双运行 [METRIC] 输出逐字节对拍、sizeof 编译期
锁定）等（后两项 2026-08-29 新增）；另有 2026-09-24 接入的三件套（此前仅有头注
手工 g++ 行、未入任何构建，对齐 face M12② 修法）：`..._CliOnlineTest`（R1–R14
在线 CLI 契约：模式 token/未知项 fail-closed/FP1 无符号整数与 watchdog 词法界域）、
`..._EventQueueDeferredTest`（EventQueue tick-end 收口 + 同 tick deferred 通道 +
FluidScheduler deferred-flush 集成 + 1000 次随机操作对照参考实现）、
`..._IngressIdleTest`（IDLE 五态生命周期/EOF 与 Error 终端/overflow 审计/
close-vs-submit 竞争线性化）。Python 侧
`online/test_propagating_tail.py`（`online_scheduler_base.py` 的在途尾部观测器
PropagatingTailTracker：对到达未完成请求、未 ack 交付、未确认 provisional KV 动作
三类在途工作记 current/peak/按来源计数，超限 fail-closed 报错、绝不截断；8 用例，
pytest 或直跑）。

## 5. 目录导览（关键路径）

- `astra-sim/workload/execution_driven/`：在线机制层（C++）
- `sh_test_mesh/workload/llama2_7b_inference/online/`：在线调度器/构图器/服务层（Python）
- `.../online/verify/`：对账与验证工具
- `sh_test_mesh/run_scripts/`：全部 runner 脚本
- `sh_test_mesh/workload/llama2_7b_inference/traces/`：物化器脚本（数据件由调用方物化，provenance 以物化器 stdout 为准）
- `sh_test_mesh/slo_tools/`：SLO 离线后处理工具集（slo_stats / load_imbalance / restore_decomposition / kv_cache_adapter / hopbytes + `slo_postprocess_driver.py`（A4 单遍合并驱动，run_slo_postprocess.sh 链内使用；工具 CLI 不变）+ `slo_params_manifest.json`（B 类参数唯一来源，B4 已填推导值）+ tests；纯离线只读，详见目录内 README.md）
- `sh_test_mesh/tests/` + workload 根：pytest（2026-09-24 修复后基线：workload
  `llama2_7b_inference/`（含 online/）= 126 passed（另 7 subtests passed）；
  `sh_test_mesh/tests/` =
  49 passed（含 resolver PER_NPU 物化断言，随数据面激活改写）；
  `slo_tools/tests/` = 125 passed + 1 skipped + 1 预存在环境性失败
  （LoadImbalance 手算×1；driver_parity×2 已随 2026-09-24 夹具
  train_ledger exits 键补全修正转为通过）；
  §2 ② 物化后 trace_config 指向真实队列时，`test_wsc_llm_scheduler.py` 的
  request-neutral 占位断言红为已知环境效应，与代码态无关——还原裸仓即绿）

## 6. 边界与纪律

- 仿真输入唯一允许源 = astra_compute_20.csv 前 2 秒（更早的用户指示曾临时
  授权过更大窗口；以当下指示为准）。
- 缺失输入一律 fail-closed（generate 桩/materializer/runner/GEN_MATCH 均实测 exit=1）。
- 映射保留区（`wsc_llm_scheduler.py` 全部映射函数 + 调度器内静态 P→D 路由与
  FCFS 语义）为保留对象，勿改；`session_kv_manager.py` 为三态 KV 内核交付物
  （2026-09 改造），语义以本 README §1.1 与《request实例映射与KV冷热管理策略说明.md》为准。
- 改动机制层后请跑 §4 fixtures + §2 ⑤ 对账再交付。

## 7. Online execution adaptation

在线策略路线（③/④）是实时宿主调度器：调度器进程内逐决策边界做映射决策，
把结果作为 per-rank 图批次发射给执行驱动引擎，决策不预先固化；每 tick 先
处理 completion 批（拼 batch 改造起先核销已完成列车
`_finalize_completed_trains`，再 PREFILL_DRAIN / DECODE_COMPLETION /
REQUEST_COMPLETE 逐请求账本），再处理 arrival 批，最后跑一次准入/发射
pass（含 D 实例迭代列车的冻结与发射；`wsc_llm_online_scheduler.py` 的
`run_variant_policy`）。

- 决策边界与仓内路由：`_on_arrival` 按 prefill 实例最小占用选择并入队，同时
  经静态 P→D 路由固定 decode 目标（不随运行时负载重选）；`_on_prefill_drain`
  复位 P 实例 busy 并登记等待 decode 准入（容量 epoch/dirty 门控下重查，
  固定静态映射不 remap）；`_on_decode_complete` 落 KV 完成账本（拼 batch
  改造后 active_decode 出队与 busy 复位移至列车核销）；`_on_request_complete`
  排下一 turn 的 arrival alarm。准入预占按净额+迁移后回补（2026-09-06；三态化后
  净额口径随层段扩账）：
  prefill 准入在静态 decode 目标预占终态 KV 时，若全量预约因本会话旧驻留
  重复计入而 deep-gap、且旧 KV 仍驻留于该 decode 实例（LOCAL 全额 / PARTIAL
  取片上前缀账面 `local_shard_bytes`），则按净额
  （终态−旧驻留账面）重试；`prepare_history` 迁移删除旧驻留后经
  `extend_request_capacity` 回补到全量（防抢占语义保持，resident→reserved
  1:1 换位，任何瞬间不超订）。
- **上条净额预占的由来（2026-09-06 准入双重计账楔死故障，6×8 拓扑全量首跑
  实证；完整分析见《wscllm准入预占双重计账修复方案.md》）**：全量 22816 请求
  首跑提交至 22740 后楔死——4 个终态 85万–105万 token 超长会话被静态 P→D
  路由各钉死在一个 decode 实例，准入重试永 insufficient 且无事件可唤醒重试，
  C++ 侧事件队列+决策邮箱全空后由统一死端守卫 fail-closed
  （`main_online.cc:1661`，安全网本身正确、未改）。根因是记账错误而非容量
  真不够：准入预占终态全量 KV 时未扣减本会话尚驻留于同一 decode 实例的旧 KV
  （`ensure_physical_fit` 的 remaining 不减自身旧驻留 + `protected_sessions`
  禁逐出自身），tp4 128 KiB/token/rank 下"旧+新"≈260GB ≫ 每 rank 预算
  ≈168.5GB（上下文 ≳64.2万 token 即触发，队内最大 104.8万）；运行中期其它
  会话的准入会触发 LRU 顺手清走旧 KV，故长会话前几百 turn 均可通过，队列
  尾部再无新准入 → `capacity_epoch` 永不变化 → 永等楔死。物理上 KV 原地单调
  增长（旧 ⊂ 新）从不同时容纳旧+新，故只修记账位点＝上条的条件化净额预占 +
  迁移后回补，改动仅 `wsc_llm_online_scheduler.py`、`session_kv_manager.py`
  两个 Python 文件与单测。验证：单测 `online/test_admission_net_credit.py`
  （旧代码永久推迟/新代码准入成功且不变量干净/release 对称归零；旧 KV 在其它
  实例、EVICTED、净额钳 0 边界与旧行为一致）；30s 标准冒烟冻结基线与修复前
  逐字节一致（条件分支不触发）；4instance 全量重跑 22816/22816 全完成、无
  lost-wakeup。
- request_aggregated 折算口径：每个请求相位按 rank 折成 17 类算子节点
  （13 类层内算子 + attention/MLP 两个 All-Reduce + final norm + logits），
  聚合 FLOPs、tensor/HBM 字节、可选远端读与集合通信载荷与 token 展开总量
  恒等，只压缩重复层/chunk/token 与集合通信启动次数
  （`sh_test_mesh/workload/llama2_7b_inference/generate_trace.py` 的
  `transformer_pass_aggregated`）；在线发射仅支持该粒度，其它粒度
  fail-closed（`online/graph_batch_builder.py`）。拼 batch 改造
  （2026-08-22）起聚合函数新增 `weight_passes` 参数（默认 `len(spans)`，
  与历史 batch=1 串行口径逐字节一致；迭代列车传迭代数：一个迭代同时算
  B 个成员各 1 token，**权重每迭代只读一次**，激活/KV/AR 逐 span 精确；
  `online/test_weight_passes.py` A0 夹具钉住 B=1/B=2 同迭代权重字节相等）。
  D 侧 decode 发射为实例迭代列车（跨成员共享体节点，见下），P 侧 prefill
  保持整段聚合（chunk 序列，每 chunk 恰读一遍权重，chunk 之间不互拼）。
- 发射并发骨架（拼 batch 改造形态，2026-08-22；§3.6 PD 分离豁免 = 仅
  decode 互拼、无混合迭代）：busy 门按 phase_role 分裂——prefill-only
  实例保持"一个 prefill 整段在飞"（逐 chunk 不拼，发射骨架与改造前一致）；
  decode-only 实例 = "一个迭代列车在飞"（`in_flight_train` 冻结成员快照 +
  membership_digest）。准入/发射 pass 只服务空闲实例：D 实例把全部
  `active_decode` 成员冻结成一趟列车（迭代数 = max 剩余 token，交付默认
  T_max=8：2026-08-22 §7.4 A2 对拍裁决——无上限 TTFT -67.3%、16 仍
  -19.1%、8 全指标 ≤1.3%，"固定为使位移 ≤5% 的最大值"，原则 1 优先于
  节点数；SH_TRAIN_MAX_ITER 可覆盖，0=不设限；截断且无 exit 标记的
  列车发哨兵标记承载完成信号；退出成员先验、不截断列车），经
  `emit_iteration_train`（`online/graph_batch_
  builder.py`）一次发射：joiner 的 transfer 3000 迁移 + join 标记 →
  17 类聚合体节点（`weight_passes`=迭代数）→ exit 标记（承载
  DECODE_COMPLETION watch 与 completion 指标锚点）→ 共享 end barrier；
  列车内**绝不混入 prefill chunk span**（无混合迭代）。完成核销
  （`_finalize_completed_trains`）逐信号路由 + 幂等：同列车 exit 标记
  watch 跨 tick fire 时首个信号核销（成员 `decode_tokens_consumed` 闭式
  推进恰一次、退出成员移出 active_decode），后续信号清 pending 集合，
  陈旧错配 fail-closed；核销后同 tick 即可冻结下一列车（连续 batching）。
  列车台账 `train_ledger.jsonl`：D 侧行 member_iterations>0，P 侧行
  prefill_chunks>0，两类字段互斥（无混合列车）。
- 接栅栏与段末屏障：列车/段发射不做块末恢复/段内清链，per-rank
  `previous_id` 无条件接续当前 frontier（per-rank 发行序 = 全局发射序，
  2026-08-19 五仓统一），跨请求 P2P 与集合通信参与序不会反转成环；每趟
  D 侧列车以 TP 组 `batch_train_*_end_barrier`（bytes=迭代数）收尾，下一
  turn 的 interval gate `after_node_id` 指向它（completion_gates 账本，
  post-barrier 口径与旧整段发射一致）；KV 准备段后另有 TP 组
  `*_history_tp_ready_barrier`。C++ 机制层适配（2026-08-22，照 sh_1.0
  母本先例）：`GraphBatchCommitter` 的 node/watch coverage 校验从双向
  等式放宽为单向包含（每个 watch 必须有节点覆盖；列车共享体/end barrier
  的批命名空间节点与 joiner 迁移节点不要求配 watch）——计时/事件/网络
  代码与 MetricCollector/WatchRegistry 零改动（批命名空间锚点注册静默
  跳过），机制夹具见 `graph_batch_committer_test.cc` Part I/J。
- 折入输入口径 + 运行时 KV 账本：请求队列为唯一仿真输入，
  turn-0 源前缀折入该行 prefill_length（新会话首轮无历史可复用的输入折算，
  `traces/derive_20_first_30_seconds.py:14-16`）；manifest 仅携带队列派生的
  history_tokens_before 推导值（`plan_materializer.py:61-88`）；会话 KV 的
  规模/位置/驻留层段由运行时 `SessionKVCacheManager` 三态账本动态维护
  （`wsc_llm_online_scheduler.py`；`session_kv_manager.py`）：驻留命中零开销
  复用、跨实例历史经 NoC 迁移或远端池回迁（PARTIAL 两段链/真流水）——
  RECOMPUTE 路径已删除（2026-09 三态化，见 §1.1），恢复取代重算。
- long double 时间精度适配：大 ns 级首达偏移超出 IEEE-754 double 精确整数
  范围（2^53）时，分析网络适配层返回 ASTRA-sim 时间以 `long double` 保持
  64 位事件时间精确，避免完成回调与事件映射键错位 1 ns
  （`astra-sim/network_frontend/analytical/common/CommonNetworkApi.cc:105-110`）。


---

[ASTRA-sim](https://astra-sim.github.io/) is a distributed AI system simulator. It models the end-to-end software and hardware stack of modern AI systems - encompassing workload scheduling, collective communication algorithms, and hardware architectures (compute/memory/network). Through a suite of APIs, it enables plug-and-play of external open/proprietary components for modeling different parts of the AI system. This provides end-to-end multi-fidelity simulation capabilities for aiding in design and deployment of next-generation distributed AI systems. 


### Overview and Documentation
Here is a concise visual summary of ASTRA-sim, showing its layers and APIs:
![alt text](https://github.com/astra-sim/astra-sim/blob/master/docs/images/astrasim_overview_codesign.png)

For a comprehensive understanding of the tool, and to gain insights into its capabilities, please visit our [website](https://astra-sim.github.io/).

For information on how to use ASTRA-sim, please visit our [Wiki](https://astra-sim.github.io/astra-sim-docs/index.html).

ASTRA-sim accepts MLCommons Chakra Execution Traces as workload-layer inputs. For details, please visit [Chakra Github](https://github.com/mlcommons/chakra).


### Releases and Contributions

ASTRA-sim is currently at **version 2.0.**
The previous version, ASTRA-sim 1.0, is available in the `ASTRA-sim-1.0` [branch](https://github.com/astra-sim/astra-sim/tree/ASTRA-sim-1.0).

We encourage community contributions to ASTRA-sim via PRs.


## Contact Us
For any questions about using ASTRA-sim, you can email the ASTRA-sim User Mailing List: astrasim-users@googlegroups.com

To join the mailing list, please fill out the following form: https://forms.gle/18KVS99SG3k9CGXm6


We appreciate your interest and support in ASTRA-sim!
