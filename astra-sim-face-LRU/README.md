# astra-sim-face-LRU - 晶圆级芯片（WSC）LLM 推理架构与机制

> 本仓为 Execution-Driven 改造后的**裸仓库终态**：仅保留路径③（strategy 关感知）
> 与路径④（strategy 开感知）两条在线仿真路线；离线静态（①）与 replay（②）
> 已删除。仓库 request-neutral：不带任何 request 队列，正式入口缺失输入
> fail-closed。
>
> 本仓是 ASTRA-sim 2.0 的晶圆级芯片（Wafer-Scale Chip, WSC）推理仿真改造仓，
> 系 `astra-sim-face` 的 **KV 冷热管理改造仓（-LRU）**：在 FACE 请求→实例映射
> **逐行保留**（`face_scheduler.py` 零改动）的前提下，把原仓的"两态 KV +
> LRU 零代价删除 + 历史全量重算"整体替换为**三态 KV 冷热管理**——
> LOCAL / PARTIAL（前 ⌈L/2⌉ 层片上 + 后 ⌊L/2⌋ 层远端）/ REMOTE 三态 +
> **去类型化两段式 LRU 逐出**（先扫全部会话后半层、仍不足才整体外迁=唯一回退）+
> 远端共享内存池逐出/回迁物理链路 + PARTIAL 同实例真流水恢复。逐出内核与
> 发射链路承袭 `astra-sim-sh_2.0`（去类型化；恢复取代重算，RECOMPUTE 路径
> 已从历史决策中删除）。六个同源仓
> `astra-sim-face / astra-sim-wscllm / astra-sim-sh_2.0 / astra-sim-sh_3.0 /
> astra-sim-face-LRU / astra-sim-wscllm-LRU`
> 共享同一套硬件架构建模与同一套术语，仅在"请求→实例映射与 KV 管理策略"上分化（见文末对照表）。
> 阅读仓内任何代码、配置或文档前，请先建立以下概念。硬件参数唯一权威来源是
> `sh_test_mesh/hardware/face_case5_config_c.json`，运行时配置
> （system.json / network.yml / comm_group.json / remote_memory.json）由
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
  profile 档位选择），作为**多用户共享带宽资源**仿真（见 G 节）；
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
  端口之间完全并行，远端池总容量不设限（sh 系仓实现于各自仓的
  `extern/remote_memory_backend/`）。
- **本仓配置**：远端内存已启用（2026-09-06 改造，自 sh_2.0 同步
  `extern/remote_memory_backend/` 现版 + Sys/CLI 接线 + `config_resolver.py`
  PER_NPU 分支）：硬件配置 `sh_test_mesh/hardware/face_case5_config_c.json` 声明
  `PER_NPU_MEMORY_EXPANSION`（512 GB/s / 100 ns / mesh-boundary /
  unified-kv-cache-pool），resolver 物化 `remote_memory.json`（RC 目录标签
  `edge_remote_memory_pool`），在线 runner 以 `--remote-memory-configuration`
  传入 C++。C++ 引擎对 KV 语义保持 opaque，仅提供 MEM 节点（mem_store /
  mem_load / local_hbm_kv_restore）的物理计时与 HBM N-way 带宽争用计费；
  本仓策略层（`kv_cache_policy=session_lru_tiered`）实际发射远端逐出/回迁
  链路（见 §1.1 与《request实例映射与KV冷热管理策略说明.md》）。

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
逐跳迁移（noc_migrate）或"远端存取 + 重新载入"。本仓为**统一实例**（unified
instance）：每实例同时持有 FCFS prefill 队列与活跃 decode 列表，同一 TP 组时分复用，
prefill 与 decode 不分池。

### F. 本仓在六仓中的定位

| 仓库 | 实例组织 | 请求→实例映射策略 | KV 驻留与恢复 | 远端内存池 |
|---|---|---|---|---|
| **astra-sim-face-LRU（本仓）** | 统一实例（P+D 同实例） | FACE 原始映射（与 astra-sim-face 逐行一致）：prefill 选剩余 chunk 最少；decode 在邻接图加权距离限制（阈值 = D2D 带宽 / 本地 HBM 带宽）内按 per-die Roofline 增量代价 | 三态（LOCAL/PARTIAL/REMOTE）；去类型化两段式 LRU 逐出（先逐后半层、仍不足才整体外迁=唯一回退）；恢复＝远端回迁（PARTIAL 后缀与前缀计算真流水，RECOMPUTE 已删除） | 启用（全部边缘芯粒挂端口） |
| astra-sim-face | 统一实例（P+D 同实例） | FACE 原始映射（同上） | RESIDENT/EVICTED 两态；LRU 逐出＝零代价删除；恢复＝重算 | 不支持（机制已整体移除） |
| astra-sim-wscllm | PD 分离（Prefill-only + Decode-only 分区） | prefill 选排队请求最少；decode 用静态一跳 P→D 映射 | RESIDENT/EVICTED 两态；LRU 逐出；恢复＝重算（跨实例历史走 NoC 迁移） | 未启用（NO_MEMORY_EXPANSION） |
| astra-sim-wscllm-LRU | PD 分离（同 wscllm） | wscllm 原始映射（静态 P/D 分区 + 最小排队 + 静态路由钉死，逐行保留） | 三态 + 去类型化两段式 LRU 逐出 + 远端回迁（与本仓同一内核族） | 启用 |
| astra-sim-sh_2.0 | 统一实例 | prefill Roofline 剩余负载均衡（历史 KV 全/部分驻留与全逐出统一）；decode 按 per-die Roofline 增量代价 + HBM 剩余 tie-break | 三态（含半驻留 PARTIAL）；两阶段类型感知逐出（human 类先于 tool 类）；流水化部分恢复 + HBM 恢复/推理带宽共享 | 启用（全部边缘芯粒挂端口） |
| astra-sim-sh_3.0 | 统一实例 | 三段式 prefill（首请求避边缘 / HBM 命中 sticky / 远端命中负载均衡）；decode 本地化固定同实例 | 三态；两阶段逐出；流水化部分恢复（机制同 sh_2.0） | 启用（全部边缘芯粒挂端口） |

与 sh 系的差异一句话：本仓**映射保留 FACE**（prefill 剩余 chunk 键 + decode
D_limit 邻域 + 单位 NPU 增量时延 + 平局轮转），**KV 管理承袭 sh_2.0 的逐出内核
与发射链路但去类型化**（不按 human/tool 分类、无消融档位、无 R_kv 水位）。

六仓还共享同一套 Execution-Driven 在线仿真机制层（`astra-sim/workload/execution_driven/`：
RequestIngress / DecisionMailbox / DecisionBridge / GraphBatchCommitter 等）：策略决策由
Python 在线服务层实时给出，计时由 C++ 物理时钟推进；各仓仅保留路径③（strategy 关感知）
与路径④（strategy 开感知）两条在线路线。六仓在线决策均不再使用任何离线 LUT
（残留 LUT 查表已随静态链路一并删除）：face / sh_1.0 / sh_2.0 的 decode 候选代价由
在线 Roofline 模型即时计算（`per_die_delta_ns`），决策确定可复算复放。

**C1 typed 响应解析（2026-08-29，随 W3 自 sh_1.0 参考仓同步）**：桥响应在
`FileDecisionBridge::deliver_and_receive` 内经 `parse_graph_batch`
（`ParsedGraphBatch.hh/.cc`）**一次性**解析为 typed 批（`GraphBatch` 即其别名），
validate / 强制 liveness preflight / commit 装配 / 锚点注册四处消费同一份
typed 数据——改前"DOM 随批驻留 + 同一棵 nlohmann DOM 被字段级提取 4 遍"的
重复解码消除（sh_1.0 30s 档 C++ 自身耗时 −16.1%，端到端无回退；本仓 2s 档
逐字节对拍见 W3 批报告）。**协议收紧（fail-closed 强化，fixture 作者可见）**：
顶层未知键、节点/边/watch/alarm 键集不符（缺失或多余）、整型域违例（负数、
>UINT64 字面量、tag/priority 超 32 位）在解析层直接经 `bridge_fatal` abort
（与既有协议违例同通道同退出码）；错误响应仍先行于结构解析判定（时序不变）。
Python 生产者（graph_batch_builder）本就满足全部键集，行为不变；三个 verify
fixture 服务（same_tick_milestone / wakeup_guard / lifecycle）原先发射的空
`comm:{}`/`coll:{}` 已补满形缺省。数组顺序契约：六个数组与 watch 成员一律按
发射序保序，解析层不排序不去重。

### G. 本地 HBM 带宽模型（多用户竞争，`hbm-bandwidth-contention`）

真实硬件中每颗芯粒的本地 HBM 是共享资源：同一时刻的所有访存流量竞争同一份带宽。
本仓用每 rank 一个 `LocalHbmBandwidthModel`（`astra-sim/workload/LocalHbmBandwidthModel.{hh,cc}`，
2026-09-06 改造随远端内存底座自 sh_2.0 同步为 **6 类作业** N-way 版：COMP /
COMM_READ / COMM_WRITE + RESTORE / POOL_READ / POOL_WRITE）建模，
system.json 键 `hbm-bandwidth-contention`
（bool，代码默认 true；置 0 完全回退旧行为——COMP 走 Roofline 闭式公式、comm 不计 HBM；
`local-mem-bw <= 0` 时自动视为 false，防零速率卡死）；另设 legacy 键
`hbm-kv-restore-bandwidth-sharing`（本仓模板值 1）：`hbm-bandwidth-contention` 关闭而
仅该键开启时，模型退化为 sh_2.0 历史 A/B 基线——"1 个 COMP + 1 个 restore" 50/50
均分（无 comm/pool 作业入模）；两键全关则不建模型（COMP 走 Roofline 闭式、restore 走
hbm_dma 单槽闭式计时）。
参与竞争的流量四类：

1. **COMP 节点（Roofline 计算访存）**：bytes = tensor_size（读写在同一标量上合并计费），
   FLOPs 按 peak-perf 并行排空，两者都排空才算完成（单用户时保持 Roofline
   max(计算, 访存) 语义）。若 ET 带**非 0 校准 runtime**（在线 LUT 时钟路径）则不进流体
   模型、保持校准时长——本仓 trace 的 COMP 无 runtime，实际总走流体路径；该分支为在线
   LUT 兼容保留。
2. **NoC p2p 通信的数据端点**：发送方 rank 的 HBM **读**（COMM_READ）+ 接收方 rank 的
   HBM **写**（COMM_WRITE）（bytes = comm 字节数）。多跳路径途经的中间芯粒**不占 HBM**
   （路由器直通——网络后端本来只在端点计费，模型与此对齐）。节点完成 = join(网络侧完成
   回调, 本端 HBM 作业完成)，两事件都到齐才触发，且恰好触发一次。ET comm 节点可用布尔
   属性 `hbm-charge`（默认 true）让该端点退出计费；bytes == 0 不建作业；1B ACK 照常计。
3. **KV restore DMA（RESTORE）**：远端回迁的目标端 `local_hbm_kv_restore` 节点
   （节点级属性 `is_local_hbm_kv_restore: true`）——restore 落 HBM 的唯一数据计费；
   一 rank 同时至多一个在飞 restore（hbm_dma 单槽），带宽分享开关见上。
4. **远端池直连端点（POOL_READ/POOL_WRITE）**：source 恰为边缘 rank 的直连
   `mem_store(hbm_access_mode=1)` 触发 POOL_READ 作业 + 池端口 FIFO 双异步 join
   （非直连链路的数据由 comm_send 承担、边缘侧过路 `hbm-charge:false` 零计费，
   见 §1.1 计费地图）。

仲裁规则：某 rank 同时有 N 个 HBM 用户 → 各得 full_rate/N **严格均分**；任一作业完成
立即释放并把带宽重分配给剩余用户（事件驱动重分配，无固定时间片）。带宽直接用配置值
`local-mem-bw`（读写共享同一总线、共享总带宽的**单一标量**，不区分读写方向，也无
峰值/持续带宽之分）；`local-mem-latency` 按 sh_2.0 惯例在每个作业启动时计一次。
TP 集合通信（PacketBundle 3x 路径）**不在**本模型范围内，行为不变。

指标层新增 `type=local_hbm` 导出（每 rank 一条，`[METRIC]` 行）：HBM busy_ns、按类别
served bytes（compute / comm_read / comm_write，SH2 扩展字段 restore_bytes_served /
pool_read_bytes_served / pool_write_bytes_served、hbm_shared_ns、
local_hbm_restore_bytes_issued）、峰值并发作业数、均分重分配事件数、
真实 HBM 利用率（busy_ns / 墙钟窗口）；旧指标键（rank_compute 的 dram_bw_util 等）语义
不变。数值断言级验证见
`astra-sim/workload/execution_driven/tests/local_hbm_bandwidth_model_test.cc`
（本仓自研 ET 静态版：1/3 → 1/2 → 全速的均分序列、join 两序、单次触发）与
`local_hbm_model_test.cc`（6 作业模型权威数值测试，575 行，自 sh_2.0 随迁）、
`remote_fifo_ledger_test.cc`（远端池端口 FIFO 记账观测件，766 行）。

## 1. 本仓是什么

- **映射语义（保留对象，零改动）**：FACE 原始映射（与 astra-sim-face 逐行一致）：
  prefill 按排序键（剩余 chunk 数, 最近承接时刻, 实例编号）字典序最小选实例；
  decode 以加权实例图邻域（距离阈值 D_limit = D2D 带宽 / 本地 HBM 带宽）内
  per-die Roofline 单位 NPU 增量时延最小者（精确平局轮转）选实例。映射不感知
  KV 位置（位置只决定恢复成本，不进入选择键）。
- **KV 冷热管理（本仓改造核心，2026-09 改造）**：三态（LOCAL / PARTIAL / REMOTE）
  + 去类型化两段式 LRU 逐出 + 远端共享池逐出/回迁 + 六分支恢复（详见 §1.1）；
  session 驻留状态/字节数/驻留层段由运行时 KV 账本（`session_kv_manager.py`
  三态内核）动态维护，无 sidecar。
- **执行驱动机制层**（`astra-sim/workload/execution_driven/`）：在线事件驱动
  （RequestIngress/DecisionMailbox/WatchRegistry/GraphBatchCommitter/长连接
  DecisionBridge 等），六仓接口一致。
- **验证证据**：Tier B 等价、感知开/关决策逐字节一致、分层账本对账、
  机制 fixtures、三态不变量审计（`SH_STRICT_KV_INVARIANTS=1` 全绿）。

### 1.1 会话 KV 三态冷热管理（`kv_cache_policy = session_lru_tiered`）

本仓 `trace_config.csv` 的 `kv_cache_policy` 已切换为 `session_lru_tiered`
（旧值 `session_lru_recompute` 仍可读、仅作旧 trace 兼容；行为上管理器唯一，
无档位分支）。逐出内核与发射链路承袭 sh_2.0，按改造策略做**去类型化**裁剪。

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

**去类型化两段式 LRU 逐出**（触发点与 face 原仓完全一致的三处：prefill 准入
`prepare_history`、decode 准入 `move_prefill_to_decode`、容量增长 `_grow`；
**不新增触发位置，完成路径零逐出**）：

1. 候选池 = 本实例上"已完成（`last_completion_ns` 非空）且非 active"的驻留会话，
   **不按 human/tool 分类**（无 `next_request_type`、请求 CSV 维持 8 列）；
   排序键 `(last_completion_ns, session_id)` 升序 = 纯 LRU；触发请求自身会话受保护。
2. **阶段 1（逐后半层）**：按 LRU 序把 LOCAL 会话的后 ⌊L/2⌋ 层 remote_store
   → PARTIAL；**每逐一笔立即逐 NPU 重查水位，够即停**。
3. 阶段 1 全部候选半层化仍不足 → **阶段 2（整体外迁 = 唯一回退）**：按同一 LRU
   序把驻留前缀 `[0, resident_prefix_layers)` 整体 remote_store → REMOTE、清
   `instance_index`；逐笔重查。
4. 两阶段耗尽仍不足：记 `deep_gap`（int 计数 + KVCacheEvent）后按 face 语义
   `admission_blocked` 推迟 + **容量纪元门重试**（容量变化或纪元重开通道时重试；
   仅空实例装不下才 raise）——不搬 sh 的"耗尽即 raise"。
- **无消融档位**（不设 `no_tiered_eviction` 类开关，逐出行为唯一：两段式，
  整体外迁是唯一回退）；**无 R_kv 保留水位**（`kv_reserve_context_tokens`
  维持 manifest-only）；**无历史截断**。
- 守卫：D4-I1 受害者资格复核（逐出后重验受害者确已非 active/非保护）+
  D4-I3 过度逐出守卫（撤销最后一笔必须使至少一个受影响 rank 回到不满足，
  否则不逐）；`SH_STRICT_KV_INVARIANTS=1` 全量审计（三态白名单 / 层域断言 /
  REMOTE 无实例无驻留层 / 逐 rank expected_kv 重算 / 预占一致性）。
- 终局会话 `retire_terminal_session`：层域减账 + 远端账面静默核销（不发传输、
  **不算逐出**）；构图器同点位清除该会话的 store 支链尾部登记。

**远端共享池与物理链路**（远端池**无容量上限、无自身置换**；边缘端口严格 FIFO
单事务 `耗时 = λ_rem + bytes/B_rem`；逐 source/target rank 取**最近边缘端口**
——曼哈顿跳数最小、平局取编号最小；NoC 段走既有 XY 维序确定性路由）：

| 链路 | 发射序与 HBM 计费（每字节恰一次） |
|---|---|
| remote_store 链 A（source≠edge） | [可选 1B trigger] → source `comm_send(bytes)`＝源端 COMM_READ 唯一数据计费 → edge `comm_recv`（`hbm_charge=false` 过路零计费）→ edge `mem_store`（池写本地零计费，仅池端口 FIFO）→ 1B ack 双端；**源端收到 ack 后才物理释放** |
| remote_store 链 B（source==edge 直连） | edge `mem_store(bytes, hbm_access_mode=1)`＝POOL_READ 唯一计费 + 池端口 FIFO 双异步 join |
| remote_load（回迁） | [1B control/arm] → edge `mem_load`（仅池 FIFO）→ edge `comm_send`（`hbm_charge=false`）→ target `comm_recv`（`hbm_charge=false`，目标写由 restore 承担）→ target `local_hbm_kv_restore`（节点级 `is_local_hbm_kv_restore=true`）＝RESTORE 唯一数据计费 |
| noc_migrate（LOCAL 跨实例 / PARTIAL 前缀迁移 / decode P→D 交接） | face 既有 1000 类 p2p 配对迁移，两端正常 COMM_READ/COMM_WRITE 计费 |

KV 传输 tag 由 `TransferTagAllocator` 从 **10,000,000** 起单调分配（错开既有
`queue_index*10000+{1000,1900,3000}` 段）；新节点命名避开 `first_token` /
`batch_train_` C++ 名字锚点子串。

**逐出与推理并行（旁路支链，2026-09-13）**：四处逐出发射点（准入 history
逐出 / prefill 增长逐出 / 列车头 joiner decode 逐出 / blocked admission
逐出）的 remote_store 物理链经构图器 `_emit_side_branch` fork 到**旁路分支**：
分支根 = fork 时刻该 rank 主链 frontier，触发门（到达 / 间隔 / drain 块末）
原样保留——门只对齐逐出的**开始**时刻；主链从同一 fork 点继续，逐出链尾
（源端 1B ack）不再阻塞其后任何计算（含其他会话的列车体），HBM 争用由 §G
的 N-way 均分模型在线自动裁决（逐出 COMM_READ 流与列车 COMPUTE 流同 rank
时间重叠即均分，无需任何配置；同 rank 第二个受害者 send 仍受 comm 单槽串行
——设备模型既有事实）。分支不 join、不挂 watch、不单独成批（"无 watch 纯
mem 批不驱动决策交付"语义不变）；blocked 逐出的 `(prefill, 0)` 上下文
stamping 与 `_mark`/`_collect` 留在包裹外，C++ in-flight 资格门不受影响；
紧随 joiner decode 逐出其后的 prefill→decode 迁移保持主链（恢复类迁移
语义上必须先于 decode 计算完成）。统一 helper 契约（主方案 §3.2，2026-09-13
勘误后四仓同文）：fork 点可能合法携带属于主链的未消费 pending 门依赖（如
turn-0 准入的到达门——turn-0 也会为给新请求腾容量而逐出其他会话），由
helper 暂存清空、主链恢复后原样归还（图依赖与逐出链在主链上时完全一致）；
分支自己的触发门 arming 必须在包裹的发射闭包内完成并被消费，发射后仍滞留
即泄漏 fail-closed。

**store→restore 前递依赖补偿**：支链化后"同会话逐出池写先于其下一轮池读"
的链序传递性失效——构图器按会话登记在飞 store 支链尾部（`pending_store_tails`
，粒度 = 边缘 `mem_store` 完成），回迁发射统一入口（REMOTE 全量 remote_load
与 PARTIAL 后缀恢复分支）发射前查表补边：同缘直接挂同 rank data_dep、跨缘
经 1B p2p 中继承载时序（桥拒绝跨 rank 直边）；两段式逐出的 suffix 与 full
两笔 store 均被依赖（restore 读全区间须等齐），terminal 会话回收登记。
懒处理：store 早已物理完成时补边即刻满足、零额外时延。

**瞬态双占用窗口口径**：容量权威在 Python 台账、**决策时刻**记账（本仓既有
语义，逐出策略/准入/`capacity_violations` 口径均不变，C++ 无容量强制）；
并行窗口内新 request 的 KV 写入与被逐旧 KV 的搬出在物理上共存——这不是
新引入的误差类别（"决策后、物理传输完成前"的窗口本就存在，并行化只是允许
计算与该窗口重叠），窗口上界 = 该 rank 在飞逐出字节数；论文引用不得把窗口
期台账值当物理占用。

**恢复六分支**（下一轮到达、映射照常选点后，`prepare_history` 按态分流；
**RECOMPUTE 已从历史路径删除**——恢复取代重算，`remaining_chunks` 不再有
`ceil(history/p_chunk)` 项）：

| 会话态 × 实例关系 | 动作（决策日志 action） |
|---|---|
| 无历史 | 建片上记录，随 prefill 增长（NO_HISTORY） |
| LOCAL 同实例 | 零开销本地复用（LOCAL_HIT） |
| LOCAL 跨实例 | 整份 KV NoC 迁移（NOC_MIGRATE，既有 1000 类路径原样保留） |
| PARTIAL 同实例 | **只回迁后缀层段 + 与前缀计算真流水重叠**（REMOTE_SUFFIX_RESTORE）：首 chunk 拆前缀层段/后缀层段，后缀段以远端恢复的逐 NPU 完成门为依赖，恢复流量与前缀计算重叠 |
| PARTIAL 跨实例 | 两段链（PARTIAL_REMOTE_MIGRATE）：前缀 NoC 迁移 + 目标实例容量逐出 + 后缀远端恢复；**两段必须序列化在同一条 prefill 决策记录的 `history_transfers` 列表内**（watermark 校验"同请求同 kind 两次即 fail"） |
| REMOTE | 全量回迁到选定实例，逐 target rank 最近边缘端口（REMOTE_RESTORE） |

**KV 指标观测（方案 A）**：`repo_variant` 维持 `"astra-sim-face"` 不变；
`kv_hit_state` **五值域不扩**——REMOTE→`full`、PARTIAL→`partial`
（`full_local` / `full_remote` 的区分由 `kv_hit_states.csv` 的 evidence 列承载，
如 `history_location_before=remote_memory`）；命中率分子 `hit_n = full + partial`
公式不变；`slo_tools/kv_cache_adapter.py` face 分支**双级解析**——新产物优先
`history_location_before` 三态映射，字段缺失回退 legacy `history_action` 四值表
（旧产物兼容）；逐出/恢复消费契约行（`history_transfers` 逐段对象 +
`history_evictions`/`prefill_evictions`/`decode_evictions`/
`completion_eviction_transfers`，legacy 镜像行不再消费防双计）；
`hbm_watermark.py` 检测到 `history_transfers` 在场自动升级 S2 逐段对账重放
（PARTIAL 两段式恢复不再混成标量）。`request_metrics.csv` 冻结列不动
（kv_hit_state 列维持 postprocess 填 NA，真值在 kv_hit_states.csv）。

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
#    arrival_time < 30e9 ns；物化器：derive_20_first_30_seconds.py）
#    产物放 sh_test_mesh/workload/llama2_7b_inference/traces/，
#    并把 trace_config.csv 第 12 行 request_queue_csv 指向它
#    （裸仓交付态该指针为占位串；kv_cache_policy=session_lru_tiered 为本仓
#    交付态值，无需改动——旧值 session_lru_recompute 保留仅为兼容读取）
#    物化器 CLI：[source] [queue] [sidecar] [window_ns] [arrival_scale]；
#    arrival_scale>0 仅缩放 turn-0 session_arrival_time_ns（t0/scale，即
#    负载 ×scale），inter_request_interval_ns（human/tool 外生等待）不动，
#    窗口判定与统计始终用未缩放源时间——scale=1 时 8 列队列与冻结基线
#    逐字节一致；缩放与 request_type 等新信息只进 canonical sidecar 与
#    stdout provenance，不进队列。

# ③ 生成 plan 目录（runtime_config 四小件 = system.json / network.yml /
#    comm_group.json / remote_memory.json（PER_NPU 边缘端口） + manifest + metrics_manifest）
cd sh_test_mesh/workload/llama2_7b_inference && python3 plan_materializer.py && cd <仓根>

# ④ 跑③④（GEN_MATCH：generated/ 下须恰一个 llama2_7b_inference_54npus_* 目录）
bash sh_test_mesh/run_scripts/run_online_strategy.sh <run_dir> <绝对路径 request_csv>
bash sh_test_mesh/run_scripts/run_online_strategy_sensing.sh <run_dir> <绝对路径 request_csv>

# ⑤ 指标后处理 + ④对账
bash sh_test_mesh/run_scripts/run_metrics_postprocess.sh <run_dir>/cpp.log
#    （在线 runner 已自动后处理：raw/normalized_metrics.csv；full 档额外产
#    request_metrics.csv——逐请求冻结 29 列，与 manifest 按 queue_index/
#    request_id fail-closed 连接；summary/off 档不产该文件，原因写
#    postprocess.log）
# ⑤b 自动 SLO 指标提取（P3，2026-08-28；A4/2026-08-29 单遍化）：在线
#    runner（strategy/sensing 两变体）在 postprocess 成功后、归档前
#    自动调用 run_slo_postprocess.sh——内部一次调起 slo_tools/
#    slo_postprocess_driver.py 单进程单遍驱动（9 步产物集/行序/公式/
#    slo_postprocess.log 与逐工具串行逐字节一致；读放大收敛：
#    request_metrics.csv 4 读→1、decision log 4 流→1、[METRIC] init 探测
#    4→1、解释器启动 9→1；七个工具 CLI 保持可独立调用）
#    ——full 档产 9 项最细粒度产物（slo_e2e_stats/
#    backlog/session/load_imbalance/restore_decomposition/hopbytes_total+
#    per_request/hbm_watermark 三件=hbm_intervals（权威 RLE）+hbm_plot_series（行预算绘图，旧 hbm_watermark_series 已退役）+hbm_watermark_instances、cache_events+kv_hit_states、
#    slo_warmup.json；P1/2026-08-30 起 hbm_watermark 按四层可信度分级：run_dir 含
#    results/kv_delta_journal.jsonl 时走 journal 权威重放，含 checksum 证书为
#    per_rank_total_hbm_certified 层——正式逐 rank 容量判决、违规 exit 3；缺
#    journal 的旧 run 为 upper_bound_only 上界层，超限只诊断不认证、exit 0），
#    一律不传分桶/聚合参数（粗化留给下游画图脚本）；
#    非 full 档依赖 request_metrics.csv 的子命令按设计跳过并写说明；
#    env SH_SLO_POSTPROCESS：1=默认 warn（任一步失败只写
#    slo_postprocess.FAIL 标记，不推翻仿真结果）、0=整步跳过、
#    strict=失败即 runner 非零退出。runner 同时把 per-request manifest
#    （metrics_manifest.json/manifest.json）拷入 run_dir 根（P2）——run_dir
#    自包含，与仓还原状态解耦
python3 sh_test_mesh/workload/llama2_7b_inference/online/verify/ledger_reconcile.py（--bridge-dir/--manifest，详见其 --help）

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

指标档位（B1/WP0 起）：runner 的 `--metrics-detail` 不再硬编码
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

### 3.2 SLO B3 集成验证落地（2026-08-27）

- `metrics_postprocess.py`：request_metrics.csv 的
  `first_token_ns`/`first_token_source` 由 request 记录（事件码 8）填充
  ——有事件 → `exact`，无事件 → `NA`（对齐 wscllm 仓 B2wp9py 同款实现；
  此前 FACE 冻结 NA，60s 全量 first_token 覆盖因此为 0）。
- `slo_tools/hopbytes.py`：face 变体收集器接入 B2wp9py 决策日志只读
  noc 观测字段（prefill=`history_noc_hops`、decode=`kv_noc_hops`，
  per-TP-shard 列表与 shards 一一对齐；hops 全等时聚合 bytes×hops[0]
  与逐 shard 求和严格相等）。60s 窗 coverage 0 → 1.0。
- `slo_tools/load_imbalance.py`：跳过 WP9 首步批台账行
  （`first_step: true` 的发射边界记录非终态 drain；真实 drain 在余量批
  行）——否则 SPLIT=ON 产物触发"每请求恰一次 drain"fail-closed。四仓
  逐字节同文由主控同步。
- `slo_tools/tests/run_golden_live.py`（新）：T2 golden G1-G4 运行版
  ——hand-craft 队列+sidecar（`traces/golden_g{1..4}_request_queue_
  recompute.csv`）→ 真仿真（full+SPLIT=1）→ 手算断言（分解和恒等、
  queue_ns≈占位者 prefill（实测 ratio=1.000）、T_session 恒等、
  request_type 透传、kv_hit_state 手推、restore 三段和=总时长）。
- `traces/llama2_7b_inference_first_60s_canonical_sidecar.csv` 刷新为
  当前 derive 输出（15 列含 human/tool/request_type；队列字节不变
  sha256 78a56480…，B0 基线可比性不受影响）；新增 10s 子窗队列
  （`…_first_10s_request_queue_recompute.csv`，oracle 门-3 用）。
- `sh_test_mesh/tests/test_metrics_contract.py`：B3-7 增补（事件码 8
  常量/edge/SERVICE_EVENT_CODES、request_metrics 29 列 frozen、
  terminal_status/first_token_source 枚举、NA 语义用例；四仓逐字节
  相同）。

### 3.3 WP9 退回处置：拆分缺省关 + train-interpolated proxy（2026-08-26）

- **缺省翻转**：`SH_FIRST_TOKEN_SPLIT` 缺省 "1" → "0"
  （`online/graph_batch_builder.py::first_token_split_enabled`，显式
  `=1` 仍可启用拆分取 exact 首 token，研究/对拍用）。理由：B3_FACE
  60s 决策等价门-2 失败——拆分的物理扰动（首列拆批 → tick 漂移，
  row 3 起 +6.5us）在闭环逐轮放大，t=25.947s 同 tick 处理顺序翻转并
  级联（1184/1454 请求实例选择翻转、1073/1454 decode 候选集翻转、
  train_id 多重集不等），ON/OFF 字节等价在 60s 窗不可达（2s 窗保持
  通过）；调度器 tie-break 属禁改区。证据：
  `/tmp/slo_wps/gates/B3_FACE.DONE`（gate2 FAIL）、
  `/tmp/slo_wps/b3/FACE/`（t3_full_off_split vs t3_full_split_on 对拍）。
  拆分语义测试缺省断言同步翻转
  （`online/test_first_token_split.py::SplitEnvironmentTest` 钉缺省关；
  `DebutPlanTest` setUp 显式置 "1"）。
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
  - **first_train_end = 同实例下一列车发射边界**：台账行 tick 是发射
    边界，face busy 门 = 一列车在飞，故同实例下一行 tick 即 end
    barrier 后首个决策边界（含混拼 prefill chunk 工作）；实例末列车 →
    NA（不可得不编造）。
  - **N=1 退化列车钳制**：share=1 → proxy=下一发射边界 > completion
    （边界晚于 end barrier tick），钳到 completion 并在 instructions
    留痕 `clamped_to_completion`（60s 重跑 1/1454）。
  - **换算权威**：W_bytes/KV_bytes 用本仓
    `face_scheduler.estimate_model_weight_bytes` /
    `kv_cache_bytes_for_tokens`（trace_config.csv 参数：swiglu →
    13476831232 / 524288 B/token，test_face_scheduler.py:301 冻结），
    不另行编造。
  - **运行内 CSV 保持 NA**（ledger 归档前不可得，instructions 记
    `proxy_unavailable:no_train_ledger(results/)`）；对归档 run 离线
    重跑 postprocess 填充——运行产物与离线分析两段语义。
  - **SLO 防线**：proxy 仅展示口径，`first_token_ns/first_token_source`
    被 `slo_common.assert_no_proxy_columns` 从一切判定路径 fail-closed
    拒绝（`slo_tools/tests/test_slo_contract.py::
    test_train_interpolated_proxy_cannot_enter_judgment`：夸张 proxy 值
    不改变 violation verdict，判定输入携带 proxy 列被拒）。
- **离线验证**（`/tmp/slo_wps/b4/FACE/proxy60/`，源=B3 归档
  t3_full_off_split + 重物化 manifests）：1454 行全部
  train_interpolated、0 序违例（填充值过 fail-closed ordering 检查）、
  raw/normalized CSV 与归档逐字节相同；抽 3 请求手算（manifest+ledger
  +cpp.log 一次产物独立重算）全部吻合，含 1 例 N=2 钳制；
  proxy vs exact（t3_full_split_on）分布差 p50 −17.6%/p99 +18.0%
  （信息性：两时间线已因门-2 级联分叉，属预期）。
- 单测：`sh_test_mesh/tests/test_first_token_proxy.py`（新，8 用例）；
  slo_tools 契约 41/41；metrics 契约 35/35；拆分语义 12/12。

### 3.1 在线二进制机制旗标（--online-* 家族）

`AstraSim_Analytical_Congestion_Aware_Online` 显式解析 `--online-*` 家族
（OnlineCli.hh 契约；家族内未知旗标硬错）。机制类旗标一枚：

| 旗标 | 取值 | 缺省 | 语义 |
|---|---|---|---|
| `--online-validate` | `0\|1` | `1`（全量） | **C1（2026-08-28）；B.2 清除（2026-09-05）移除 N 抽检档**：GraphBatch 提交前全量校验开关（占 face 重载墙钟 ~42%）。`1` = 每批校验（改前行为，裸调用的 fail-closed 缺省）；`0` = 生产快速路径（跳过校验，commit() 的强制活性预检保留）。取值域为严格 `0\|1` 枚举，`N≥2` 或其它任何值启动即硬错误（杜绝陈旧脚本静默改变校验频率）。只影响 validate 计数/诊断（graph_validate_ns 等白名单字段），不影响已提交状态。生产 runner 默认传 0（env `SH_ONLINE_VALIDATE` 可覆盖），冒烟/fixture/verify 脚本显式传 1。 |

`--online-node-gc` 旗标已由 **B.3 清除（2026-09-05）** 整支移除：M2 节点 GC
摊销回收恒开（原 `0` 臂 = pre-M2 永不删除应急回退臂已删，现仅 fixtures 内部
Context 开关可达），任何 `--online-node-gc` 调用按家族未知旗标规则启动即
硬错误（`unknown online-family option`），陈旧脚本无法静默禁用 GC；决策序列
与全部工件不受影响（历史生产调用恒为默认 1）。

runner 脚本（strategy/sensing 两变体）显式传
`--online-validate "${SH_ONLINE_VALIDATE:-0}"`；fixture runner（idle/
wakeup_guard/same_tick_milestone）显式 `--online-validate 1`。运行期证据：
cpp.log 启动行 `[online] node gc: ...` / `[online] graph validate: ...`、
结束行 `[online] node gc: erased=... retained=...`。

停泊与输入契约 fail-loud（P0-2，2026-08-31，非 `--online-` 前缀、同属
OnlineCli 在线家族解析）：

- **calendar reader（P0 turn-0 late-discovery fix，2026-08-30，对齐 wscllm）**：
  CSV 队列经一次 64 KiB 分块流式索引遍（fail-closed 结构校验：session 块
  连续、块首 turn-0、块内 turn 严格 +1、turn>0 行 arrival 必空）+ 溯源边车
  门（`<queue>.provenance.json` 存在即逐字段比对，任一不匹配零 Submit 前
  `[Error]` 退出）+ turn-0 到达日历（按 `(arrival, queue_index)` 稳定排序
  提交）读取——turn-0 提交序与文件位置彻底解耦，非单调输入下
  `late_static_submit=0`，run-end 输出 arrival audit 行并以
  `late_static_submit==0 且逐行 ingress_delay==0（t=0 边界行除外）` 为正式
  门禁（fail 则非零退出）。
- **--request-window-rows 死旋钮已清除（2026-09-05，A.4）**：该 advisory
  旋钮（2026-08-30 P0 turn-0 fix 后 calendar reader 按 arrival 序提交，窗口
  值不约束读取与提交，不改变任何行为）已从本仓彻底移除——OnlineCli 解析/
  选项成员、main_online 调用实参、runner 的 `SH_REQUEST_WINDOW_ROWS` 透传
  一并删除；WindowedTraceReader 的 high_water 缺省 128 仅留作 reader 内部
  advisory 记录，无 CLI 暴露口。**本仓不设 C++ 启动 span 预检/拒绝门**。
  plan_materializer 的
  manifest.json 仍持久化 `max_same_session_span`/`span_session_id`/
  `span_row_range`/`span_row_range_convention`，仅作 provenance 审计
  （campaign 复核与根因归档），无任何运行期强制拒绝点。
- **A2 input-open 死端分支不适用（六仓对齐 wscllm 裁决，合同 §2.1/P1）**：
  改造前 face/sh 系的"input 开 + 窗口有占用 + 无 pending alarm"可证死端
  形态（原 41 分钟静默 futex 楔死；P0-2 曾以该分支 fail-loud 兜底）在本仓
  calendar reader 下不可达——①泵送与 turn-0 提交次序：reader 先建全量
  arrival calendar，turn-0 按 arrival 序提交，泵送在 drain 之前/之后的停驻
  形态与 calendar 不变量互相闭合；②分类点处 `!csv.empty() && !eof() &&
  pending_alarm==0` 的机器状态互斥（cursor 停驻形态下 pump 与二次 drain
  次序保证任一未触发 turn-0 必持有 pending alarm 或尚未入 calendar，二者
  不可同时为空）；③Error 终止路径（fail-closed）先于停泊发生。**未来重构
  若打破 pump 后置 drain 次序或 calendar 完整性不变量，必须重做可达性
  分析**，在此之前不引入该分支。停泊兜底统一交 `--idle-watchdog-s`
  （含任何未来未知停滞形态）；12 字段 `parking_diagnostics` 报文保留
  （`window_occupancy` 在本仓语义=已提交未触发 turn-0 计数）。
- `--idle-watchdog-s <秒>`：墙钟停泊看门狗，缺省 `1.0`=武装（2026-09-05：
  静默楔死 ~1s 即 fail-closed abort，不再无声挂死；官方 CSV run 到达全量
  预排为队列事件、健康运行不停车不受影响。显式 `0`=关恢复 `wait_for_work()`
  原无界契约——IDLE fixture 等刻意长停车场景必须显式传 `0`）；开时停泊点
  墙钟超时即同款诊断 fatal
  （`--idle-` 前缀同受家族未知旗标硬错保护）。
  **FP1（2026-09-01，sync-A16 批次P）**：数值合同冻结——token 不得含任何
  空白或符号字符（拒 `" +1"`/`" -1"`）；判界唯一顺序为 `==0` 接受（=关）
  → `(0,1e-9)` 拒（"低于时钟分辨率"，常量
  `kMinIdleWatchdogSeconds=1e-9`）→ `>1e9` 拒（平台上限，常量
  `kMaxIdleWatchdogSeconds=1e9`，两端点本身接受）；ERANGE 上下溢均拒。
  运行期改为 `ServiceCoordinator::checked_wait_deadline`（纯函数：tick 域
  判界→转换→加法前判界→单次 deadline）+ `wait_for_work_until`（绝对
  deadline，协调器内不再二次 `now()+timeout`）；主循环单次取 now、单次算
  deadline；`0`=关时必须走原 `wait_for_work()` 阻塞等待。
- **FP1 整数解析加固（同上批次）；B.2 清除（2026-09-05）修订**：
  `--request-max-arrival-ns` / `--bridge-timeout-ms`
  统一"纯 ASCII 数字词法（拒 `" -1"`/`"\t-1"`/`"+1"`）→ `errno=0` +
  ERANGE 拒 → endptr 到串尾 → 目标类型上限（request-max-arrival-ns uint64_t、
  bridge-timeout-ms 另加 `<= INT_MAX`）→ 才转换"合同，失败不部分写入 `out`。
  `--online-validate` 原同属该整数词法族（`--online-validate=4294967296`
  曾会回绕为 0 静默关闭图验证，fail-open）；B.2 清除后改为严格 `0|1` 枚举
  解析，任何其它值（含抽检 N≥2 与越界大数）启动即硬错误。
- runner 透传 env：`BRIDGE_TIMEOUT_MS` 三态——未设=缺省 120000（须大于
  负载最慢单决策与 Python 侧 FIFO 开启等待）、显式 `0`=永等逃生口、正值=
  该毫秒值；它只武装 C++ 桥 response poll（Python 单次交换停滞族），管不到
  停泊族（由上面看门狗兜住）。

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
  P2 拷入的 `metrics_manifest.json`/`manifest.json`、
  P3 自动 SLO 提取产物（`slo_*.csv`、`slo_*.json`、`cache_events.csv`、
  `kv_hit_states.csv`、`slo_postprocess.log`、`slo_postprocess.FAIL` 若有）。
  任一阶段失败不归档、
  全量保留供排查（runner 失败路径早已 exit 1）。2s 冒烟实测 run 目录
  23MB → 1.8MB。
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
- **Python 侧内存/CPU**（B 系列）：session KV 事件流增量读取+水位压缩
  （B1，消除 O(n²) tuple 拷贝；event_index 改单调计数器）、committed KV
  账本改计数器（B2）、sensing_query/online_stats 流式落盘（B3，后者经
  `.partial` 两遍合并保字节一致）、请求完成后 runtime 索引 pop（B4）、
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
`run_online_same_tick_milestone.sh`、`bridge_race_stress_repro.sh`——机制层健康自检。
另有 C++ 聚焦单测（target 注册于 `astra-sim/network_frontend/analytical/CMakeLists.txt`，
随 §2 ① 构建树编译，可执行文件落在 `build/astra_analytical/build_congestion_aware/bin/`，
无参数直跑；2026-08-29 新增三项，多仓同构；2026-09 -LRU 改造再随迁两项）：
`..._AlarmCancellationTest`（可取消 alarm
链路：bucket 清空时 outer alarm 从 backend 物理移除、共享 bucket 级联、重复取消幂等、
legacy 后端回退 stale guard）、`..._MetricOneShotEraseTest`（MetricCollector one-shot
node bucket 擦除 + OnlineNode anchor 快路径标志，双运行 [METRIC] 输出逐字节对拍、
sizeof 编译期锁定）、`..._RemoteFifoLedgerTest`（RemoteFifoLedger 按 backend 真实端口
记账；自带 PER_NPU/PER_NODE/MEMORY_POOL 三架构 fixture 自证——本仓无 sensing 记账
接线，账本不启用）、`..._LocalHbmTest`（本仓自研 ET 静态版 HBM 模型测试，保留）、
`..._LocalHbmModelTest`（6 作业模型权威数值测试，575 行，2026-09 改造自 sh_2.0 随迁：
RESTORE/POOL_READ/POOL_WRITE 泳道与 N-way 均分序列的逐数值断言）等。
Python 侧 `online/test_propagating_tail.py`
（`online_scheduler_base.py` 的在途尾部观测器 PropagatingTailTracker：对到达未完成
请求、未 ack 交付、未确认 provisional KV 动作三类在途工作记 current/peak/按来源计数，
超限 fail-closed 报错、绝不截断；8 用例，pytest 或直跑）；KV 三态改造配套专项：
`test_face_tiered_eviction_sequence.py`（两段式逐出序：半层×N 逐笔重查停机 →
不足才整体×M、D4-I3 撤销重开、触发会话保护、平局按 session_id、耗尽 admission_blocked
不 raise）、`test_face_kv_incremental_invariants.py`（三态增量账本不变量）、
`online/test_graph_batch_builder.py` 的 `KVTransferBillingNailTest`（remote_store/
remote_load/noc_migrate 逐节点 HBM 计费键钉子）与
`test_partial_first_chunk_two_stage_pipelining`（PARTIAL 真流水首 chunk 层段拆分）。

## 5. 目录导览（关键路径）

- `astra-sim/workload/execution_driven/`：在线机制层（C++）
- `sh_test_mesh/workload/llama2_7b_inference/online/`：在线调度器/构图器/服务层（Python）
- `.../online/verify/`：对账与验证工具
- `sh_test_mesh/run_scripts/`：全部 runner 脚本
- `sh_test_mesh/workload/llama2_7b_inference/traces/`：物化器脚本（数据件由调用方物化，provenance 以物化器 stdout 为准）
- `sh_test_mesh/slo_tools/`：SLO 离线后处理工具集（slo_stats / load_imbalance / restore_decomposition / kv_cache_adapter / hopbytes + `slo_postprocess_driver.py`（A4 单遍合并驱动，run_slo_postprocess.sh 链内使用；工具 CLI 不变）+ `slo_params_manifest.json`（B 类参数唯一来源，B4 已填推导值）+ tests；纯离线只读，详见目录内 README.md）
- `sh_test_mesh/tests/` + workload 根 + online/：pytest（-LRU 改造后基线：
  tests/ 48 + workload 根 34（test_face_scheduler 25 / kv_incremental_invariants 3 /
  tiered_eviction_sequence 6）+ online/ 74（另 7 subtests）= **156 passed**，
  `SH_STRICT_KV_INVARIANTS=1` 下同绿；2026-08-22 拼 batch 改造新增
  online/test_weight_passes.py(4) + online/test_train_machinery.py(9) +
  online/test_graph_batch_builder.py(7)；2026-08-29 内存根治续作新增
  online/test_propagating_tail.py(8)；2026-09 -LRU 改造新增/改写
  test_face_tiered_eviction_sequence.py(6) + 计费钉子/流水专项等；
  2026-09-13 逐出旁路支链改造新增 test_eviction_side_branch_structure.py(7)
  + test_store_restore_ordering.py(5)）
- `sh_test_mesh/slo_tools/tests/`：120 例（B4 后 34→38 watermark / 总 114→120）；
  其中 3 例（driver_parity ×2 + load_imbalance 手算例）为**改前基线既有失败**
  ——原 face 仓与 sh_2.0 仓同样失败（环境性/共享基线问题），与 -LRU 改造无关

## 6. 边界与纪律

- 仿真输入唯一允许源 = astra_compute_20.csv 前 2 秒（更早的用户指示曾临时
  授权过更大窗口；以当下指示为准）。
- 缺失输入一律 fail-closed（generate 桩/materializer/runner/GEN_MATCH 均实测 exit=1）。
- FACE 映射文件 `face_scheduler.py`（Prefill/Decode 选点）为保留对象，**勿改**
  （-LRU 改造全程零改动，diff 可证）；`session_kv_manager.py` 为三态 KV 内核
  （本仓改造核心交付物）——修改它必须连带跑三态不变量测试
  （test_face_kv_incremental_invariants / test_face_tiered_eviction_sequence）
  且 `SH_STRICT_KV_INVARIANTS=1` 全绿。
- 改动机制层后请跑 §4 fixtures + §2 ⑤ 对账再交付。

## 7. Online execution adaptation（拼 batch 迭代列车版，2026-08-22）

在线策略路线（③/④）是实时宿主调度器：调度器进程内逐决策边界做映射决策，
把结果作为 per-rank 图批次发射给执行驱动引擎，决策不预先固化；每 tick 先
核销已完成列车（标记 watch 驱动的 completion 批：PREFILL_DRAIN /
DECODE_COMPLETION / REQUEST_COMPLETE + 列车哨兵信号），再处理 drain/完成，
再处理 arrival 批，最后跑一次准入/发射 pass（冻结并发射各空闲实例的下一趟
迭代列车；`sh_test_mesh/workload/llama2_7b_inference/online/face_online_scheduler.py`
的 `run_variant_policy` → `_finalize_completed_trains` → `_on_prefill_drain` →
`_on_decode_complete` → arrivals → `_admit_pass`）。

- 迭代列车（拼 batch 核心，设计文档《层次 B Continuous Batching 改造》§3.2；
  sh_1.0 定型版为母本）：层次 B 从"请求级大段串行"重构为"实例迭代级列车"——
  decode 互拼（一趟列车 B 个成员各推进 participation 个 token，权重每迭代
  只读一次）、decode 与 prefill chunk 混拼（每迭代 ≤1 chunk，FCFS 队列头，
  chunk 之间不互拼）、批成员只在列车边界变化。每实例状态机：qp（FCFS
  prefill 队列）/ active_decode（批成员表）/ pending_decode_ready（KV 就绪
  待加入）/ in_flight_train（唯一在飞列车，冻结成员快照 + membership_digest；
  busy 门 = "一个列车在飞"）。列车终点 = 队列头 prefill drain（剩余 chunk 数，
  先验）或全部 decode 工作耗尽；交付默认 T_max=8（2026-08-22 §7.4 A2 对拍裁决：无上限 TTFT -67.3%、16 仍 -19.1%、8 全指标 ≤1.3%——"固定为使位移 ≤5% 的最大值"，原则 1 优先于节点数；SH_TRAIN_MAX_ITER 可覆盖，0=不设限；截断且无自然标记的列车发哨兵标记承载完成信号）（SH_TRAIN_MAX_ITER 正整数 =
  上限，截断列车无自然标记时发射哨兵标记承载完成信号）。
- 决策边界与仓内路由：`_on_arrival` 按离线同款选择键选 prefill 实例入 FCFS
  队列；`_on_prefill_drain` 以全局 9 实例快照经加权图候选 + Roofline per-die
  代价动态选择 decode 实例（统一实例，P/D 可同可异；批口径适配 =
  active_tokens 输入用列车核销后闭式推进的 current_decode_token，
  select_decode_instance/estimate_iteration_time_ns 本体不动——保留对象
  红线）；decode 准入（KV 迁移 + 容量增长）在 drain 时点完成后成员进入
  pending_decode_ready，待加入 decode 实例的下一列车（3000 类 prefill→decode
  迁移随加入列车发射）；`_on_decode_complete` 落 KV 完成账本
  （active_decode 出队移至列车核销）；`_on_request_complete` 排下一 turn 的
  arrival alarm。
- request_aggregated 折算口径 + weight_passes：每趟列车按 rank 折成 17 类
  算子节点（13 类层内算子 + attention/MLP 两个 All-Reduce + final norm +
  logits），聚合 FLOPs、tensor/HBM 字节、可选远端读与集合通信载荷与
  成员×迭代 token 展开总量恒等，只压缩重复层/chunk/token 与集合通信启动
  次数；`transformer_pass_aggregated` 的 `weight_passes` 参数（默认 =
  len(spans) 与历史 batch=1 串行口径逐字节一致）由列车传**迭代数**——
  权重字节 ×迭代数、与批成员数无关（激活/KV/AR 逐 span 精确；陷阱 1
  防护，online/test_weight_passes.py 的 A0 夹具）；在线发射仅支持该粒度，
  其它粒度 fail-closed。
- 发射并发骨架（本仓形态：实例单列车在飞门）：实例账本 busy 门 =
  `in_flight_train`（一个列车在飞）；准入/发射 pass 只服务就绪 frontier 的
  非忙实例：qp 头部 prefill 准入在服务时点逐头部尝试（容量纪元门防重试
  风暴，face 策略保留），准入成功发射准入动作（gates/history 迁移/
  readiness 屏障，无 prefill 主体），再冻结列车（头部 chunk × 迭代 +
  active_decode 成员混拼；准入失败退化为纯 decode 列车）；列车核销时闭式
  推进（成员 token / chunk 进度 / current_decode_token，决策边界上与逐
  token 精确值逐点一致，无逐 token 热路径循环）。
- 准入活性与选择期容量过滤（P0-1/P1，2026-08-31）：容量纪元门防重试风暴
  补上"重开通道"——prefill/decode 准入因容量阻塞时把目标实例记入
  `prefill/decode_admission_dirty`（纪元未推进时的唯一重试通道），容量
  不变也逐决策边界重开一次准入尝试（消除阻塞后纪元门不再重试的悬置
  死端）；选择期叠加只读容量可行性过滤（decode/prefill 各一，需求口径
  镜像正式准入，三态：原选可行零扰动 / 可行集内改选 / 全不可行保持
  原选——绝不 remap 已冻结列车），介入时记 `admission_probe` 决策行供
  对拍剥离；探针零副作用（只读快照/纯函数，不触碰任何 mutation）。
- 接栅栏与段末屏障：列车发射不做块末恢复/段内清链，per-rank `previous_id`
  无条件接续当前 frontier（per-rank 发行序 = 全局发射序，2026-08-19 四仓
  统一），跨请求 P2P 与集合通信参与序不会反转成环；每趟列车一个共享 TP 组
  `batch_train_i*_*_end_barrier`（替代每请求一个）+ 列车体后、barrier 前的
  drain/exit/哨兵标记节点（承载 PREFILL_DRAIN / DECODE_COMPLETION watch，
  C++ watch fire 自动同时推 DECODE_COMPLETION + REQUEST_COMPLETE）+ join/
  pstart 起始标记（指标锚点）；准入动作以 TP 组 `*_history_tp_ready_barrier`
  收尾。列车台账 train_ledger.jsonl 每次发射一行（§7.3 不变量断言输入，
  runner 归档至 results/）。
- C++ 机制层批节点适配（拼 batch，2026-08-22）：GraphBatchCommitter 的
  watch 覆盖规则放宽为单向（每个 watch 的 (request_id, stage) 必须被本批
  节点覆盖；批可携带批命名空间共享体/end-barrier 与无 watch 的 joiner/
  准入节点——sh_1.0 同款规则，设计文档 §3.1-2），列车哨兵 watch
  （"batch_train_" 前缀）绕过单请求 in-flight 资格检查；计时/事件/网络
  代码零改动。C++ 侧夹具：graph_batch_committer_test.cc Part I/J（两请求
  列车 / 混合 drain+exit 标记 + 哨兵探针）。
- 折入 turn-0 前缀的输入口径 + 运行时 KV 账本：请求队列为唯一仿真输入，
  turn-0 源前缀折入该行 prefill_length（队列口径不变，
  `traces/derive_20_first_30_seconds.py:14-16`）；manifest 仅携带队列派生的
  history_tokens_before 推导值（`plan_materializer.py:61-88`）；会话 KV 的
  规模/位置/驻留层段由运行时三态账本（`SessionKVCacheManager`）动态维护
  （`face_online_scheduler.py`；`session_kv_manager.py`）：LOCAL 驻留命中直接
  复用、跨实例整份 NoC 迁移、PARTIAL 两段式恢复（同实例后缀回迁真流水 /
  跨实例前缀迁移+后缀回迁）、REMOTE 全量回迁——**RECOMPUTE 已从历史路径
  删除**（被逐出的会话落在远端冷层而非重算，见 §1.1）。
- long double 时间精度适配：大 ns 级首达偏移超出 IEEE-754 double 精确整数
  范围（2^53）时，分析网络适配层返回 ASTRA-sim 时间以 `long double` 保持
  64 位事件时间精确，避免完成回调与事件映射键错位 1 ns
  （`astra-sim/network_frontend/analytical/common/CommonNetworkApi.cc:71-81`）。
- WP9 首 token 首步批拆分（2026-08-26，`SH_FIRST_TOKEN_SPLIT`，B4 起
  缺省 "0" 关——60s 决策等价门-2 失败退回 proxy，见 §3.3；显式 "1"
  开启，关闭后构图/决策/账本产物与上线前逐字节一致）：含 debut 成员
  （decode_tokens_consumed==0 的本交付 joiner）且迭代数 ≥2 的列车拆两批发射
  ——首步批 = 全体成员第 1 个 span/chunk + joiner 迁移/起始标记（weight_
  passes=1）+ 多 token debut 的 first_token 标记（事件码 8，metrics_schema
  `EVENT_FIRST_TOKEN_COMPLETE`，节点名含 `first_token` 子串供 C++ 锚点取
  min tick）+ 批命名空间唤醒标记（`batch_train_<id>_first_step`，哨兵同款
  watch 通道，fire 后调度器无操作吞掉并在下一决策边界冲余量批）；余量批 =
  剩余 span/chunk（weight_passes=iterations-1）+ drain/exit/哨兵标记 + end
  barrier（挂点语义不变；decode_length=1 debut 的 exit 标记改名携带
  first_token 子串，同节点双锚点）。首步批产物带 `"first_step": true` 台账
  标记供 ON/OFF 对拍剥离。另：决策日志新增只读 `history_noc_hops` /
  `kv_noc_hops` 观测字段（逐 TP shard XY 路由跳数，与构图发射同源，不改
  路由），供 Hop-Bytes 核算。

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
