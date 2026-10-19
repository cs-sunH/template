# astra-sim-wscllm - 晶圆级芯片（WSC）LLM 推理架构与机制

> 本仓为 Execution-Driven 改造后的**裸仓库终态**：仅保留路径③（strategy 关感知）
> 与路径④（strategy 开感知）两条在线仿真路线；离线静态（①）与 replay（②）
> 已删除。仓库 request-neutral：不带任何 request 队列，正式入口缺失输入
> fail-closed。
>
> 本仓是 ASTRA-sim 2.0 的晶圆级芯片（Wafer-Scale Chip, WSC）推理仿真改造仓。
> 五个同源仓 `astra-sim-face / astra-sim-wscllm / astra-sim-sh_1.0 / astra-sim-sh_2.0 / astra-sim-sh_3.0`
> 共享同一套硬件架构建模与同一套术语，仅在"请求→实例映射与 KV 管理策略"上分化（见文末对照表）。
> 阅读仓内任何代码、配置或文档前，请先建立以下概念。硬件参数唯一权威来源是
> `sh_test_mesh/hardware/face_case5_config_c.json`，运行时配置
> （system.json / network.yml / comm_group.json）由
> `sh_test_mesh/config_resolver.py` 从它派生，禁止手改。

## 核心概念与硬件架构建模

### A. 晶圆级芯片（wafer-scale chip）＝整个计算系统

指被仿真的**整片晶圆这一完整计算系统**：一片二维 mesh 晶圆（行列数由硬件配置的
`mesh.rows/columns` 给出），全部芯粒之间由片上网络（NoC）互连，每颗芯粒自带本地
HBM。NPU 数量禁止单独配置，恒等于行 × 列；芯粒按 row-major 编号，rank = 行 × 列数 + 列。

### B. NoC（片上网络）

晶圆内部芯粒间的互连：非环绕二维 mesh 上相邻芯粒间的有向链路（die-to-die），
链路带宽与每跳时延取硬件配置 `d2d` 的值（多跳逐跳累计），XY（维序）确定性路由，同一
链路的多条数据流由流体调度器（FluidScheduler，fluid-pipelined）做带宽严格均分并行
推进（拥塞感知，无逐跳 store-and-forward 的 FIFO 串行排队）。C++ 侧由
分析型网络后端实现（`extern/network_backend/analytical/congestion_aware/`），Python
规划层有同一语义的确定性 XY 路由函数。注意：这是分析级模型，不建模路由器微架构
（crossbar / VC / credit），也不做 cycle-accurate 仿真。

### C. 芯粒（die / NPU / rank）＝挂在 NoC 上的每一个单元

挂载在 NoC 上的**每一个单元**即一颗芯粒。代码与配置中统一称 **NPU**（或 rank），
权威映射为"一个 FACE 物理计算 die = 一个 ASTRA-sim NPU"。每颗芯粒包含：

- **计算单元**：die 内成本模型计时（生产 die 后端：硬件 `die` 节 → 共享成本库
  派生 PE/向量峰值并编译 DiePlan，每 COMP 节点一个 DieExecution 状态机驱动执行，
  见 H 节；显式 legacy 后端保留旧 Roofline 闭式）；
- **本地 HBM**（KV 热层）：带宽 / 时延 / 容量取硬件配置 `local-hbm` 的值（容量按
  profile 档位选择）；多用户带宽竞争模型见 G 节；
- **网络接口（路由端口）**：mesh 上的一个 Device，经有向链路连接其在 mesh 中的物理邻居；
- **远端内存端口**（仅边缘芯粒拥有；远端内存扩展已随 2026-09-24 移除批退役，
  现无生效配置，见 D）：与片外统一内存池直连的中转通道，对应物理系统中的
  SerDes 等片外互连。

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
  端口之间完全并行，远端池总容量不设限。
- **本仓配置**：`NO_MEMORY_EXPANSION`（远端池未启用，WSC-LLM 策略场景强制如此）。
  远端内存建模已随 2026-09-24 远端内存清除批退役：`AnalyticalRemoteMemory` /
  `RemoteFifoLedger` 后端与 `extern/remote_memory_backend/` 整目录已物理删除，
  `config_resolver.py` 对任何非 `NO_MEMORY_EXPANSION` 的 `memory-type` 解析期
  fail-closed 报错，仅保留 `NO_MEMORY_EXPANSION` 路径；上文远端池 / 边缘端口
  模型仅余设计记录。本仓 KV 全部驻留片上本地 HBM：LRU 逐出＝零代价删除、
  恢复靠重算，跨实例历史 KV 仅走 NoC 迁移。

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
逐跳迁移（noc_migrate）。本仓为 **PD 分离**（disaggregated）
布局：Prefill-only 实例与 Decode-only 实例分区部署（数量与位置由 `trace_config.csv`
各实例的 `phase_role` 给出）；prefill 完成后 KV 按离线静态映射（最近邻、边不相交）
迁往 decode 实例。

### F. 本仓在五仓中的定位

| 仓库 | 实例组织 | 请求→实例映射策略 | KV 驻留与恢复 | 远端内存池 |
|---|---|---|---|---|
| astra-sim-face | 统一实例（P+D 同实例） | FACE 原始映射：prefill 选剩余 chunk 最少；decode 在邻接图加权距离限制（阈值 = D2D 带宽 / 本地 HBM 带宽）内按 per-die Roofline 增量代价 | RESIDENT/EVICTED 两态；LRU 逐出＝零代价删除；恢复＝重算 | 未启用（NO_MEMORY_EXPANSION） |
| **astra-sim-wscllm（本仓）** | PD 分离（Prefill-only + Decode-only 分区） | prefill 选排队请求最少；decode 用静态一跳 P→D 映射 | RESIDENT/EVICTED 两态；LRU 逐出；恢复＝重算（跨实例历史走 NoC 迁移） | 未启用（NO_MEMORY_EXPANSION） |
| astra-sim-sh_1.0 | 统一实例 | prefill HBM 可行过滤 + 剩余 chunk 最少；decode 按 per-die Roofline 增量代价 | LOCAL_HBM/REMOTE_MEMORY 两态；整 session 粒度逐出；恢复＝远端全量取回 | 启用（全部边缘芯粒挂端口） |
| astra-sim-sh_2.0 | 统一实例 | prefill Roofline 剩余负载均衡（历史 KV 全/部分驻留与全逐出统一）；decode 按 per-die Roofline 增量代价 + HBM 剩余 tie-break | 三态（含半驻留 PARTIAL）；两阶段类型感知逐出（human 类先于 tool 类）；流水化部分恢复 + HBM 恢复/推理带宽共享 | 启用（全部边缘芯粒挂端口） |
| astra-sim-sh_3.0 | 统一实例 | 三段式 prefill（首请求避边缘 / HBM 命中 sticky / 远端命中负载均衡）；decode 本地化固定同实例 | 三态；两阶段逐出；流水化部分恢复（机制同 sh_2.0） | 启用（全部边缘芯粒挂端口） |

**逐出执行口径（删除型；与 -LRU 改造仓对照，2026-09-13 补记）**：本仓 LRU 逐出是**删除型**——
容量压力下 `SessionKVCacheManager._delete()`（`sh_test_mesh/workload/llama2_7b_inference/session_kv_manager.py`，
由 `ensure_physical_fit` 逐出循环调用）只做即时台账扣减（`resident_kv_bytes` 即时减去受害会话分片）
+ 决策日志/事件记录，类 docstring 明示 "an `evict_delete` intentionally has no ET representation"：
**无 ET 图表示、无 DMA 传输、无带宽消耗、零仿真时间占用**，与同 rank 在跑的推理计算不存在物理
资源争用，天然并行。被逐会话的后续到达按全量重算处理，重算段以 `history_recompute` stage 折入
prefill 计算（`online/graph_batch_builder.py` 的 history-recompute 折入段），发射节点与普通 prefill 完全相同（普通
COMP 节点），照常进入本地 HBM 带宽多用户均分模型（本仓带宽模型仅 COMP/COMM_READ/COMM_WRITE
三类作业、无 pool 概念，见 "G." 小节）。本仓唯一物理 KV 流量是 NOC_MIGRATE 活会话跨实例迁移
（history 跨实例迁移与 joiner prefill→decode 迁移两处发射点，恢复类、须先于其后计算）。与 -LRU
改造仓（`template/astra-sim-wscllm-LRU`）的物理逐出/回迁链路形成对照：后者逐出是真实物理传输，
其逐出与推理并行化改造见该仓 README。

五仓还共享同一套 Execution-Driven 在线仿真机制层（`astra-sim/workload/execution_driven/`：
RequestIngress / DecisionMailbox / DecisionBridge / GraphBatchCommitter 等）：策略决策由
Python 在线服务层实时给出，计时由 C++ 物理时钟推进；各仓仅保留路径③（strategy 关感知）
与路径④（strategy 开感知）两条在线路线。五仓在线决策均不再使用任何离线 LUT
（残留 LUT 查表已随静态链路一并删除）：face / sh_1.0 / sh_2.0 的 decode 候选代价由
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
lifecycle）原先发射的空 `comm:{}`/`coll:{}` 已补满形缺省。wscllm 仓特例：本仓
无 MEM 远端池节点，`ComputeAttrs` 无 `hbm_access_mode` 字段，该键不在解析层
compute 键集内（出现即按未知键 fail-closed，与 builder 从不发射该键的现状一致）。
数组顺序契约：六个数组（nodes/parent_edges/watches/assignments/kv_actions/future_alarms）按发射序保序；watches[].members 的成员按 JSON 对象键序（rank 字符串字典序，非数值序）迭代，消费方均序不敏感（2026-10-01 更正：原文宣称 watch 成员按发射序保序，与 ParsedGraphBatch 的实现与头注不符）。解析层不排序不去重。

### G. 芯粒本地 HBM 带宽模型（`hbm-bandwidth-contention`）

真实硬件中每颗芯粒的本地 HBM 是共享资源。C++ 执行层为每个 rank 建一个
`LocalHbmBandwidthModel` 流体模型（`astra-sim/workload/LocalHbmBandwidthModel.{hh,cc}`，
由 system.json 键 `hbm-bandwidth-contention` 开关，默认开），以下三类流量竞争同一份
`local-mem-bw`（本仓无片外内存池）：

1. **COMP 节点（roofline 计算访存）**：bytes = `tensor_size`（读写合并计费），
   ops 与访存并行排空，单用户时保持 roofline `max(计算, 访存)` 语义。带非 0 校准
   runtime 的 COMP 节点不进流体模型、保持校准时长（本仓 trace 的 COMP 无 runtime）。
2. **NoC p2p 通信的数据端点**：发送方 rank 的 HBM 读 + 接收方 rank 的 HBM 写
   （bytes = comm 字节数；history_kv、prefill_to_decode_kv 等迁移两端都计）。
   **多跳途经的中间芯粒不占 HBM**（路由器直通——网络后端只在端点计费）。
   bytes==0 不建作业，1B ACK 照常计；ET comm 节点属性 `hbm_charge`
   （snake_case；默认 true）可显式豁免某端点——kebab 拼写 `hbm-charge` 在
   C1 解析层键集白名单之外，携带即整批 fail-closed 拒绝。
3. **集合通信的本地端点**（ALL_REDUCE 2026-09-28 修复批新增；ALL_GATHER /
   REDUCE_SCATTER / ALL_TO_ALL 2026-09-29 对称补齐，此前三类完全绕过模型、
   端点大块流量系统性低估）：bytes = coll 字节数（读写合并计费，同 COMP
   口径），节点完成 = join(集合通信完成, 本端 HBM 作业完成)，与 p2p 的两
   事件 join 用同一张 `HbmCommJoin` 表，四类集合通信端点口径一致。

4. **DIE_COMPUTE（die 后端内部计算访存阶段，2026-10-03 die 推广新增）**：
   die 后端下每个 DieExecution 的 MemoryRequest 产生一个服务量作业——**排空量 =
   `service_units`（=实际字节/η），实际字节推进 = 服务量×η**（side-band 两列分列，
   绝不回填 tensor_size 或 D2D payload）；**无 FLOPs 腿、无 latency 相**
   （`local-mem-latency` 由 DieExecution 以一次性 TIMER 在首个内存请求前收取，
   每外层节点至多一次）；完成回调直连 DieExecution（内部内存阶段**绝不伪装**
   外层 COMP 完成）。legacy COMP 字节作业（第 1 条）只在 legacy 后端出现。

仲裁口径：N 个并发用户严格均分（各得 `full_rate/N`），任一作业完成立即释放并给
剩余用户重分配（事件驱动）；带宽直接用配置标量——读写共享同一总线与总带宽，不区分
读写方向，不区分峰值/持续带宽；`local-mem-latency` 在每作业启动时计一次
（DIE_COMPUTE 除外，见上第 4 条）。p2p comm
与 ALL_REDUCE 节点完成 = join(通信侧完成, 本端 HBM 作业完成)，两侧都到齐才触发
节点完成（幂等）。集合通信的 PacketBundle 3x 归约路径（隔离时延，不经本模型）
保持原状。

**本仓在线路径的结构性 N=1 登记（D3，2026-09-28）**：本仓在线发射为 per-rank
严格串行链（请求内逐节点 parent 链、请求间完成驱动准入、decode 列车成员共享
聚合节点；无 J 仓 2026-09-25 式"逐笔并行支链"改造），同一 rank 任一时刻至多
一个 HBM 作业在飞——多用户均分仲裁在本路径**结构性不产生计时差**（N 恒 1，
单用户语义 = roofline `max(计算, 访存)` / join(通信, 端点 HBM)，恰为 §G 承诺
的单用户行为）。`hbm-bandwidth-contention` 开关在本仓的真实计时效果来自
**端点计费**（comm/ALL_REDUCE 端点 HBM 字节、COMP 访存排空）而非多用户争用；
N>1 争用仅在未来发射侧引入同 rank 并行支链时才会生效（J 仓形态）。
（2026-10-01 澄清："结构性"指本仓 Python 发射侧的串行链性质，不是引擎保证——
引擎的 GPU comp 槽与 comm 槽互不互斥、COMM_RECV 不占 GPU 槽，第三方 builder
发射同 rank 重叠即出现 N>1；local_hbm_bandwidth_model_test.cc 的 S1 正是这样
构造出 N=3。）开关不是
假开关（=0 回退闭式 roofline、comm 零 HBM 计费，与 =1 有计时差），但"多用户
竞争同一份带宽"的物理在本仓在线 workload 上未被激活——宣称以此为准。

`hbm-bandwidth-contention: 0`（或 `local-mem-bw <= 0` 自动回退）＝完全恢复旧行为：
roofline 闭式公式、comm 立即随网络完成、不计 HBM。die 后端下 =0 的语义（契约 §7.5）：
无仲裁器时 DieExecution 自驱内存腿（满速率排空 service_units 的 TIMER），**双方争用
未模拟**（结果注明），通信侧沿用旧开关语义；峰值/带宽标量在 die 后端下由库派生值
回流填充（§H 配置链）。指标导出（MetricCollector，旧键不变）：每 rank `hbm_busy_ns`、
分类 served bytes（comp / comm_read / comm_write；die 后端下 compute 列只含
marker 微作业的 1/1 字节，真计算走 DIE_COMPUTE 服务量作业）、`peak_concurrent_jobs`、
`redistribution_events`、真实利用率
`hbm_busy_util = busy_ns / 墙钟窗口`（勿与 microbench iteration 的
`hbm_bw_util = bytes/(tp_degree×bw×t)` 混同）。单测：
`astra-sim/workload/execution_driven/tests/local_hbm_bandwidth_model_test.cc`。
策略细节见《request实例映射与KV冷热管理策略说明.md》。

### H. 计算成本后端（die 生产 / roofline / replay legacy）与 die 配置链

COMP 节点的计时由 system.json 键 `compute-cost-model` 选择，共三后端：

| 后端 | 取值 | 语义 |
|---|---|---|
| die（生产） | `die` | 真实计算 COMP 必须携带 `die_op_spec`（GraphBatch compute 子对象；ET 侧为同名 JSON 字符串属性）；`Workload::issue` 分发序：timer op → 旧 timer 路径；cpu op → replay；`op_role=control_marker` → 旧微作业语义（有 HBM 模型则 `issue_compute(1,1)`，否则 1ns 事件）；有 die_spec → DieModelAdapter（编译/缓存 DiePlan + 每节点一个 DieExecution 状态机驱动执行）；真 COMP 缺 die_spec → **fail-closed throw，绝不静默回退 roofline**。`runtime_ns != 0` 与 die_spec 并存被 parser 拒绝 |
| roofline（显式 legacy） | `roofline` | 等价 `roofline-enabled=1`：COMP 走旧 Roofline 闭式/流体路径，保留供回归（§2 配置链整体切换方法：硬件 JSON 换回 `compute` 节 + `validation-160gib` 档 + roofline system 模板 + 删 trace_config 的 `position_encoding` 行） |
| replay（显式 legacy） | `replay` | 旧默认分发（无成本模型） |

冲突矩阵（Sys 与 resolver 双层 fail-closed）：`die` + `roofline-enabled=1` → 报错
（本仓无 `replay-only` 键——Path-2 移除批已删除该 dispatch 路由，矩阵无此项）；
无 `compute-cost-model` 且 `roofline-enabled=1` → 旧 roofline（向后兼容既有模板）；
非 die 后端携带 `die-model-*` 键 → WARN。

**控制 marker 语义**：发射侧把一切 `comp(name,1,1)` 微节点（first_token / exit /
sentinel / join / wakeup 等）显式标 `op_role="control_marker"`（缺省即 compute）；
marker 在两种后端下都保持旧微作业语义原样（本仓 marker 唯一发射点：
`graph_batch_builder.py` 的 `_emit_train_marker`）。

**die 配置链（唯一硬件源 → resolver → 运行时）**：

1. 硬件只使用根目录最新 DOCX 定义的唯一 Core；SRAM 容量与 NoC 位宽决定
   32 行资源表中的安装单元，Core 外形固定为 SRAM 长边。已移除型号选择字段。
   legacy `die` 输入保留 `sram-mb-per-core`（数值单位为 MiB）、`compute-area-mm2`、
   `hbm-per-side`、`hbm-sides`、`noc-width-bits`，并要求用户明确给出
   `mesh-nx`、`mesh-ny`。此兼容后端仅支持相等的正方阵，矩形网格使用
   `coarse_die_v2`；不从面积推导、搜索或修改 Core 数量。
   用户本次已授权按TX8参考算力匹配原则制定一套工程对照输入：默认采用
   10×10 Core、25×32 mm Die（800 mm²）、0.75 MiB/Core、512 bit、legacy HBM4×1。
   其126.6 INT4 TOPS仅在数值上接近TX8的128 BF16 TFLOPS（相差1.09375%），
   不表示不同精度的性能等价或硬件实测校准；Vec12.65728 T与TX8参考1 T未匹配。
   可选 `die-width-mm`/`die-height-mm`必须成对、乘积等于面积且分别容纳Core网格；
   未给宽高的旧输入仅保留面积正方形兼容检查，summary明确标识范围。
   legacy HBM 仍为每 stack 16 GB、410 GB/s 的显式历史参考，不能冒称新 DOCX
   的 HBM3E 配置。PE 峰值以 INT4 TOPS 报告；为旧消费者保留的
   `pe_peak_tflops` 数值是废弃的 INT4 兼容标量，不是 BF16/FP32 实测吞吐。
2. `config_resolver.py` 经 ctypes 调 `libdie_model_cabi.so` 的 `die_hardware_summary`
   取派生值（Python 侧零派生公式），写 `resolved_die_model.json`（入库
   generated/runtime_config/<slug>/，可 round-trip 回灌）；并把派生标量回流既有键
   （`peak-perf`＝废弃的 INT4 峰值兼容标量、`local-mem-bw`＝HBM 总带宽 GB/s；latency 沿用
   hardware `local-hbm.latency-ns`）。
3. system.json 三键（来自 die system 模板 + resolver 注入）：
   `compute-cost-model: "die"`、`die-model-profile: "mesh_die_v1"`（或
   `tx8_reference_v1`）、`die-model-config: <resolved_die_model.json 绝对路径>`
   （resolver managed 注入，模板手写即拒；C++ 侧缺省路径 = system.json 同目录）。
4. 容量对齐硬校验：die 节存在时，所选 capacity-profile 的 bytes 必须等于派生 HBM
   容量（本仓默认档 `validation-die-64g` = 64000000000 B；旧 `validation-160gib`
   在 die 节下会被 resolver 拒绝——容量相关数值变化是 die 点的正常反馈）。
5. manifest 链：plan_materializer 把 `compute-cost-model` / `die-model-config` 写入
   manifest.json 与 metrics_manifest.json → Python 在线服务按 manifest 构造发射语境
   （DieModelContext）注入构图器，"发射走 die"。本仓调度器（wsc_llm_scheduler 的
   build_static_pd_mapping/select_prefill_instance）按拓扑/队列规则工作、无计算成本
   oracle，预测侧不存在半新半旧问题（设计文档 §8.4 的 WSCLLM 条款：仅日志/基线
   用途的 roofline 字段保留 legacy_estimate 身份）。

**发射侧 spec 附着点（本仓两处，均经 `transformer_pass_aggregated(die_ctx=...)`）**：
D 侧 decode 列车体（`_emit_train_body`，phase=decode）与 P 侧 prefill/recompute
整段（`generate_wsc_llm_trace._emit_prefill_stage`，phase=prefill）——span 绝对位置
按本仓 span 恒等式（kv = q_start + q_tokens、kv 自 0 连续）还原；§7.2 纯本地链
生成期融合（qk+scale/mask+softmax+av → attention_region；up+activation+down →
ffn_region），collective 位置/数量/参与者/bytes 与 watch 绑定不变。

**共享库发现序**（ctypes 加载 `libdie_model_cabi.so`，resolver 侧约定）：
env `DIE_MODEL_CABI_PATH` → 仓内
`build/astra_analytical/build_congestion_aware/lib/libdie_model_cabi.so`（再退
`bin/`）→ 失败 **fail-closed 报错（绝不静默回退 roofline）**。库随 §2 ① 全量构建
产出（`die_model_cabi` target，输出目录锚定构建树根的 lib/，恰为上述仓内路径）。

die 后端的利用率口径（`OnlineStatisticsState` 新增 `vector_utilization`）：
compute = FLOP/(PE 峰值×elapsed)、memory = 实际字节/(HBM 带宽×elapsed)、
vector = eq-ops/(向量峰值×elapsed)；legacy 统计路径不含 vector 列（恒 0）。
**指标出口**：每 rank 的 `type=rank_compute` [METRIC] 记录新增
`active_kernel_die_{compute,memory,vector}_util` 三键（门控 =
`compute-cost-model=die`，与 roofline 键的 legacy 门控互不影响；无 comp 节点
的窗口按 null+note 报告）；metrics_postprocess 在后处理侧 fail-closed 断言
"声明 die 后端的 run，每 rank 三键非 null 且 ∈[0,1+ε]"（metrics_manifest.json
由 plan_materializer 同步写入 `compute-cost-model` 声明，归档 run 离线可判）。

**性能注记（随 face 参考实现整体携带，库为同版本字节一致 vendor）**：
- **进程级共享 PlanCache**：`Sys::DieModelRuntime` 的计划缓存按
  `plan_cache_hardware_key` 索引到进程级单例注册表（compile 纯函数、DiePlan 不可变，
  同硬件的 54 个 Sys 共享一份 LRU；单线程仿真无锁）。
- **evaluate_stages 重写 + 稳态快进**（extern/die_model/src/CostModel.cc）：精确路径
  与冻结实现 bitwise 等价（差分模糊钉住），另带带验证回退的稳态直线快进。
- **DieExecution 纯 timer 段合并**：HBM 模型无活跃作业时连续 Timer 动作合并为单个
  Sys 事件（§8.5 外部竞争集合不变守卫；内存请求边界绝不跨越）。
- **候选剪枝下界修正（E3）**：健全下界（每资源槽 busy 和 + 槽前缀+busy + lag 折算
  关键路径）替换 launch 串行和旧界（旧界可超真 makespan 剪掉真最优）；selection_
  fuzz 常驻测试钉住选择等价。本仓 1s die 冒烟 wall ≈2:32（C++ 本体 ≈151s）。

## 1. 本仓是什么

- **策略语义（保留对象，未改动）**：PD 分离：6P:3D + StaticPdMapping 静态路由；唯一在线 KV 策略变体 `session_lru_recompute`（legacy 与 relevant_distributed 两个历史变体已于 2026-09-05 清除——清除方案文档已随改役归档不在仓内，`kv_cache_policy` 值域 fail-closed 收紧为单值）；RESIDENT/EVICTED 两态、session 驻留状态/字节数由运行时 KV 账本（SessionKVCacheManager）动态维护，无 sidecar
- **执行驱动机制层**（`astra-sim/workload/execution_driven/`）：在线事件驱动
  （RequestIngress/DecisionMailbox/WatchRegistry/GraphBatchCommitter/长连接
  DecisionBridge 等），五仓接口一致。
- **验证证据**：Tier B 等价、感知开/关决策逐字节一致、分层账本对账、
  机制 fixtures。

## 2. 快速开始

```bash
cd <本仓根>
# ① 构建（首次需先配置；构建树已预置时可跳过 configure）
#    裸仓无 build/：先放拼装 CMakeLists（上游 astra-analytical 仓的 vendored
#    等价物，三行 add_subdirectory 聚合 AstraSim 库/analytical 后端/前端）：
#    mkdir -p build/astra_analytical && cat > build/astra_analytical/CMakeLists.txt <<'EOF'
#    cmake_minimum_required(VERSION 3.22)
#    project(AstraSim_Analytical_Build)
#    # enable_testing() 必须在聚合顶层：缺了它 CMake 不在聚合根生成
#    # CTestTestfile.cmake，tests/run_all.sh 的 ctest 全量入口会零用例空转
#    enable_testing()
#    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../ astra_sim_build)
#    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../extern/network_backend/analytical backend_build)
#    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../astra-sim/network_frontend/analytical frontend_build)
#    EOF
cmake -S build/astra_analytical -B build/astra_analytical/build_congestion_aware -DBUILDTARGET=congestion_aware -DNETWORK_BACKEND_BUILD_AS_LIBRARY=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build/astra_analytical/build_congestion_aware -j
#    ①b die 后端配套：全量构建含 `die_model`（静态成本库）与 `die_model_cabi`
#    （共享 C ABI，.so 落 build_congestion_aware/lib/ ——即 §H 发现序的仓内
#    路径；Python 侧 ctypes 经它取硬件派生值）。Release 旗标自 die 推广起为
#    缺省口径（无 fast-math，IEEE 浮点行为不变；成本库热路径内存受限，
#    -O0 会显著拖慢 die 计划编译）。

# ② 物化输入（唯一允许源 = agent-traces/tracelab/astra_compute_20.csv 前 30 秒，
#    arrival_time < 30e9 ns——物化器 WINDOW_NS=30e9 缺省即此口径；冒烟实测
#    可显式传更小 window_ns（如 2e9）；验收对账口径以 ⑤ 的
#    ledger_reconcile 缺省值为准（缺省 1177/112 即 30s 窗口验收值，
#    见下文 D3 登记）；
#    物化器：traces/derive_20_first_30_seconds.py（运行 stdout 即权威 provenance 记录））
#    到达链口径（D3 登记，2026-09-28）：物化窗口选择按静态链"turn i 到达 =
#    上一 turn 派生到达 + 上一 turn 行记录的 gap"（2026-08-15 冻结，
#    ledger_reconcile 缺省 1177/112 即该窗口验收值）；运行端回放按因果链
#    "父 turn REQUEST_COMPLETE tick + 本 turn interval"排程（完成时刻在
#    物化期不存在，回放到达恒 >= 静态派生到达，差 = 逐 turn 服务时间累计）。
#    两链回答不同问题（窗口内选哪些行 vs 回放何时发射），刻意不统一；
#    详见物化器 docstring 登记。
#    产物放 sh_test_mesh/workload/llama2_7b_inference/traces/，
#    并把 trace_config.csv 的 request_queue_csv 行指向它
#    物化器 CLI：<source> [queue] [sidecar] [window_ns] [arrival_scale]（source 必填，
#    缺失 exit 2 并打印用法——2026-09-27 对齐 face infra-8 修复批，删除硬编码默认源）；
#    arrival_scale>0 仅缩放 turn-0 session_arrival_time_ns（t0/scale，即
#    负载 ×scale），inter_request_interval_ns（human/tool 外生等待）不动，
#    窗口判定与统计始终用未缩放源时间——scale=1 时 8 列队列与冻结基线
#    逐字节一致；request_type 进 canonical sidecar 并在 stdout 汇总，
#    arrival_scale 只进 stdout provenance 与 <queue>.provenance.json
#    （canonical sidecar 无 scale 列）——两者都不进队列。
#    P0 turn-0 修复（2026-08-30）配套：物化器同时写出 <queue>.provenance.json
#    （FNV-1a 64/字节数/行数/会话数/turn-0 min-max/文件序相邻逆序/块连续）；
#    C++ 侧 WindowedTraceReader 首开即流式复算并与边车逐字段比对，任一不匹配
#    在任何 Submit 发生之前 fail-closed 退出。P1 修复（2026-09-28）起边车
#    必填：缺失边车同样 fail-closed 退出（原"缺省仅记一行、不设门"的口子
#    会让中途夭折的物化留下的截断队列静默缩小仿真输入还全绿通过）；物化器
#    同步改为原子发布——队列/边车/边车 provenance 全部先写 .tmp、全部成功
#    后 rename 落位，失败即清理临时文件且最终路径不出现半成品。
#    P2-9 修复（2026-09-28）起队列表头同样内容校验：首个非空行必须逐字等于
#    canonical 8 列表头（kRequestQueueCsvHeader，WindowedTraceReader.hh；
#    容忍单个行尾 \r 兼容 CRLF），否则在任何 Submit 之前 fail-closed——
#    旧版首行盲跳会让无表头 CSV 吞掉第一数据行、错/乱序列名静默重绑列位，
#    而 expected/total/audit 同源自洽全绿；零字节队列（无表头）同批 fail-closed。
#    C3 修复（2026-09-28）起 request_id 同样在任何 Submit 之前做全行唯一性
#    门：空 id 与跨数据行重复（含跨 session 块复用同一 id——物化器生成的
#    session_<sid>_request_<turn> 天然唯一，只有手写/变形队列可撞）一律
#    fail-closed；旧版 turn-0 臂无任何查重，重复 id 会流入日历并让下游
#    指标账本按 last-wins 错挂身份。
#    输入保留字（多端口哨兵保留门）：id 含 "first_token" 子串（session_id
#    与 request_id 两字段都查）、或 request_id 以 "batch_train_" 前缀开头
#    （本门只查 request_id），一律 fail-closed 拒绝——CSV 读取器
#    （WindowedTraceReader）、future_alarm 信封解析（ParsedGraphBatch）与
#    FIFO Submit 臂三端口同位同守卫；"first_token" 是 WP9 首 token 标记
#    节点名锚点（见 §3），"batch_train_" 是迭代列车 end barrier 与哨兵
#    watch 的机制层保留命名空间（见 §7），真请求携带会与哨兵锚点混叠或绕过
#    fail-closed 资格门。手写携带上述保留字的请求在输入端口即被响亮拒绝，
#    无静默错账路径。

# ③ 生成 plan 目录（runtime_config 三件 system.json/network.yml/comm_group.json
#    + manifest + metrics_manifest；remote_memory.json 已随远端内存机制退役不再物化）
cd sh_test_mesh/workload/llama2_7b_inference && python3 plan_materializer.py && cd <仓根>
#    F2_PY 修复（2026-09-29）起物化器在消费边车前对账 <queue>.provenance.json：
#    记录的 csv_sha256 与当前队列字节不符（derive 三次顺序 rename 的中断窗口
#    留下的"新队列+旧边车+旧 provenance"混合代际）即 fail-closed 报错——旧
#    行为是旧边车行静默绑新队列、透传字段静默省略、两 manifest 合法成形；
#    队列旁无 provenance 记录（如单测合成队列）仍容忍（C++ 在线读取器的
#    provenance 在场门不受影响）。

# ④ 跑③④（GEN_MATCH：generated/ 下须恰一个 llama2_7b_wsc_llm_inference_54npus_* 目录）
bash sh_test_mesh/run_scripts/run_online_strategy.sh <run_dir> <绝对路径 request_csv>
bash sh_test_mesh/run_scripts/run_online_strategy_sensing.sh <run_dir> <绝对路径 request_csv>

# ⑤ 指标后处理 + ④对账
bash sh_test_mesh/run_scripts/run_metrics_postprocess.sh <run_dir>/cpp.log
#    （在线 runner 已自动后处理：raw/normalized_metrics.csv；full 档额外产
#    request_metrics.csv——逐请求冻结 29 列，与 manifest 按 queue_index/
#    request_id fail-closed 连接；summary/off 档不产该文件，原因写
#    postprocess.log）
#    归档交互（2026-10-07 补记）：SH_ARCHIVE_RUN 默认开启时，成功 run 的
#    cpp.log 已被归档步压缩为 cpp.log.gz——上面字面命令里的 <run_dir>/cpp.log
#    随之不存在；工具已支持直接读取 .gz 回退（与 run_slo_postprocess.sh 的
#    cpp.log → metrics.log → cpp.log.gz 同款回退口径），亦可按归档状态解压
#    后传入明文日志。
python3 sh_test_mesh/workload/llama2_7b_inference/online/verify/ledger_reconcile.py --bridge-dir <run>/bridge --manifest <ET_DIR>/manifest.json --cpp-log <run>/cpp.log（--cpp-log 必填——P2-19（2026-09-28）起缺它 R0d/R0e/R2c/R3a/R3c 以 cpp.log 审计计数器为权威数据的对平全部空洞通过（vacuous pass），故缺省即拒绝；归档态透明回退 cpp.log.gz。期望值缺省 1177/112，异窗口传 --expected-requests/--expected-accepted-sessions；详见 --help。前置条件：run 产物须含 `SH_GRAPH_DIGESTS=1` 与 `ASTRA_SIM_ONLINE_COMPLETED_NODES=exact` 的数据源——缺失或为空即 exit 2 fail-closed，不再以空对平"全绿"；train_ledger.jsonl（列车发射账本）同为必填数据源（R7a1/R8b 的列车成员时间线回放依赖，strategy 运行恒产出并归档至 results/，缺失 exit 2 fail-closed）。R5a/R7a1/R8b 为列车感知口径（2026-10-01 对齐）：issued 请求的物理在飞证据 = 请求级未完成组（joiner 迁移/exit 标记）或所在列车的批命名空间组；末边界残差 = 终完成动作的 end-barrier 控制尾（请求级或列车共享，不再按命名空间裁剪）；当前列车含 issued 成员即须在摘要可见——逐请求数值下界在列车聚合世界无定义，故原"在飞节点数 ≥ issued 数"口径退役）

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
#    容量判决、违规 exit 3（2026-09-25 起该层以 checksum 完整性为前置：
#    证书 ranks 块非空且逐 rank 终态对账、missing-rank 反查全过才判
#    certified，ranks 缺失/为空时降级为非 certified、不做 exit 3 判决）；
#    缺 journal 的旧 run 为 upper_bound_only
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
  violation；timeavg 与 capacity_timeavg 交叉差 >1% 计 violation）。采样口径
  （2026-09-28 修复批对齐 Python journal 重放）：逐 ledger delta 采样——同 tick
  多条 delta 的中间瞬态逐条计峰值/违规（不再折叠为组后终值单点）、窗口末端
  锚定的 delta 亦采样（零长度，不改 area）。桶记录仅发
  "有活动的桶 + 首末桶"；`hbm_watermark`/`hbm_watermark_summary` 记录带
  `window_start_ns`/`window_end_ns`（=finalize 的 sim_end_tick；WP6 `link_bucket`
  的窗口右端点是观察者自身 last tick，两族桶记录靠窗口字段对齐而非隐式同窗）。
  账本论域 fail-closed（2026-09-28 C2 批，与 Python journal 重放对齐）：
  `memory_actions` sequence_index 非严格递增、重放中间分量为负——均退出码 1
  拒绝（Python 同流 exit 2，不再"一边出数一边拒绝"）；锚定 tick 倒置需重排、
  ledger 行 rank 不在模拟系统集合（幽灵行）、fallback rank 的 unresolved 计数、
  非 online 模式下 init/summary 的 `input_requests` 同名异值——均计 consistency
  violation（fallback 记录如实上报 per-rank actions_total/unresolved）；
  `requests[].prefill_ranks/decode_ranks` 必须为数组、`request_id` 必须唯一
  （load 时 fatal，防标量静默缩域与 online 锚点 last-wins 挂错 queue_index）。在线合成 manifest 无 planner 账本时输出全零序列并
  注明原因（既有现状；真实水位线由离线工具 hbm_watermark.py 从
  results/online_decision_log.jsonl + 物化 plan manifest 重放重建，阶段 2 起
  另加 results/kv_delta_journal.jsonl 权威逐 rank delta 账本，bridge/ledger.jsonl
  仅 sensing 档落盘且不含 bytes、非占用数据源——见 sh_test_mesh/slo_tools/
  README.md）。
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
  recompute.csv`；该批 golden 队列由本脚本运行期生成、收尾自清陈旧
  golden CSV，属生成式自产自清，非裸仓在场文件——裸仓 `traces/` 仅存
  物化器脚本）→ 真仿真（full+SPLIT=1）→ 手算断言（分解和恒等、
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
    13476831232 B 权重值由 test_wsc_llm_scheduler.py 的冻结断言钉住，
    524288 B/token 为该公式现值；
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
覆盖 70.4%→满覆盖；基线对拍剥离项）。

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
  （tick/active/pending_alarm/window_occupancy/mailbox 等 11 字段）
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
  `sh_test_mesh/run_scripts/archive_run_outputs.sh <run_dir> [metrics_detail]`
  做产物瘦身归档（runner 把解析出的指标档位作第二位置参传入；off 档 cpp.log
  零 `[METRIC]` 行属正常设计、豁免归档并留痕，summary/full 及未传档位的裸调用
  零行仍 fail-closed 拒绝归档）：
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
  P2 同批拷入的 `trace_config.csv`（run-time 配置随 run_dir 冻结，
  2026-09-28：后处理重跑旧 run_dir 经 hbm_watermark 首选候选直接命中，
  不回退仓内现行文件）、
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
- **env `SH_FIFO_WAIT_S`**（两个在线 runner，缺省 60）：C++ 侧桥 FIFO
  就绪等待的墙钟预算（秒）；超时打印 cpp.log 尾部后 fail-closed 退出
  （健康运行 FIFO 秒级就绪；crosscut-F2 2026-10-01 引入）。
- **env `SH_SLO_MEM_LIMIT`**（run_slo_postprocess.sh，缺省 16G）：SLO
  后处理 cgroup v2 内存护栏上限，防 hbm_watermark 重放爆内存。
- **env `SH_SLO_SKIP_HBM`**（run_slo_postprocess.sh，缺省 0）：=1 跳过
  hbm_watermark 重放步骤（排障用；正常验证勿开）。
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
  ③/④：commit 点还有逐事务逐 rank 对账），并断言守恒
  resident=0/reserved=0/physical=weight，产物
  `results/kv_delta_journal_checksum.json`（行数/sha256/逐 rank 终态表）。
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
`run_online_same_tick_milestone.sh`、`bridge_race_stress_repro.sh`——机制层健康自检。

另有 C++ 单测 fixtures（`build/astra_analytical/build_congestion_aware/bin/`，无参数直跑）。
构建约束：检查以裸 `assert()` 表达的测试目标（CliOnlineTest / EventQueueDeferredTest /
AlarmCancellationTest / IngressIdleTest）在 CMake 里已加
`$<$<NOT:$<CONFIG:Debug>>:-UNDEBUG>`（2026-09-27 起由原 `$<$<CONFIG:Release>:-UNDEBUG>`
单档放宽）使断言在缺省 Release 及 RelWithDebInfo/MinSizeRel 构建下常开，
否则 -DNDEBUG 会把断言整体编译掉、二进制恒绿：
`..._WindowedReaderTest`（P0 turn-0 修复后的日历 reader 单测：索引遍+日历提交/
advisory 窗口/乱序 turn-0/首块超窗/arrival=0 t0 边界/同 tick 按 queue_index 排序/
provenance 篡改与块结构违例 fail-closed/到达审计门自证）、
`..._CalendarOracleTest`（P0 V2 等价 oracle：新日历 reader vs 逐字冻结的修复前
reader 基线拷贝 `tests/LegacyOracleWindowedTraceReader`（window=0 全量臂）在
同一输入上的 (alarm_tick, queue_index) 触发序逐元素相等；可选 argv[1] 传真实
队列 CSV——全量 22,816 行 TraceLab 队列实测两臂各 496 条全等、且新臂
late_static_submit=0），
`..._LocalHbmBandwidthModelTest`（本地 HBM 带宽竞争数值/join 两序/回退，见 G 节）、
`..._AlarmCancellationTest`（可取消 alarm 链路：bucket 清空时
outer alarm 从 backend 物理移除、共享 bucket 级联、重复取消幂等、legacy 后端回退
stale guard）、`..._MetricOneShotEraseTest`（MetricCollector one-shot node bucket
擦除 + OnlineNode anchor 快路径标志，双运行 [METRIC] 输出逐字节对拍、sizeof 编译期
锁定）等（后两项 2026-08-29 新增）。
2026-09-24 修复批接入：`..._CliOnlineTest`（R1–R14 在线 CLI 契约，含 FP1 加固
词法与看门狗判界回归）、`..._EventQueueDeferredTest`（EventQueue tick 末收口 +
同 tick deferred 通道 + FluidScheduler deferred-flush 集成回归）、
`..._IngressIdleTest`（IDLE 五态生命周期 + FP1 边界 9 场景）——三者原仅头注
g++ 手工可达（孤儿测试），现均为前端 CMake 正式目标、随主构建落 bin/ 直跑
（现均注册 ctest，随 tests/run_all.sh 全量执行）；同批 6 个既有单测目标的输出目录统一归位
bin/（改前落在前端默认目录，按本文档 bin/ 路径执行找不到文件）。
2026-09-26（tests 同步批）起 `..._NodeStoreTest` 亦无参数直跑：原 part C
（ETFeederGraphSource .et 适配器用例，需 `--fixture-et=` 指向合成 trace）已随
离线静态路径退役删除（合成 trace 生成器 `make_completion_fixture_et.py` 与其
唯一消费者 CompletionFixture 目标已于 2026-09-27 静态臂退役批一并删除）。Python 侧
`online/test_propagating_tail.py`（`online_scheduler_base.py` 的在途尾部观测器
PropagatingTailTracker：对到达未完成请求、未 ack 交付、未确认 provisional KV 动作
三类在途工作记 current/peak/按来源计数，超限 fail-closed 报错、绝不截断；8 用例，
pytest 或直跑）。
2026-09-28（测试与 fixture 批 E1）：`..._BridgeLoopbackTest` 注册 ctest
（本仓首个 analytical `add_test`；argv[1] 绝对路径指向其 Python 对端
`online/verify/bridge_echo.py`——此前该 fixture 仅 add_executable、无任何自动
入口执行它；TIMEOUT 600，场景 C 设计内耗 ~10s 启动宽限），`tests/run_all.sh`
的 ctest 全量入口自此覆盖桥回环契约。同批 verify/ 死入口清理：四个零引用手动
脚本 `bridge_streaming_fixture.py`/`idempotency_fixture.py`/
`profile_scan_audit.py`/`diff_explainability.py` 删除（J 仓 2026-09-27 同款
先例），其可自动执行契约分别移植为 unittest 用例
`online/test_bridge_streaming.py`（tracker 严格单调/松散尾 ≤255/journal 序/
canonical 字节保真/跳号 fail-closed）、`online/test_delivery_idempotency_gate.py`
（§7.2 delivery 幂等门 + ack 流 fail-closed + verify_run_end）、
`online/test_profile_scan_bound.py`（每批 scanned_entries 与总规模无关）；
`bridge_echo.py` 保留（活引用）。`online/` 目录补 `__init__.py`：unittest
discover（pytest 缺失降级 runner）此前不遍历 online/ 子树，降级口径的 root-1
可见用例由 E2 时代的 36 → 151（含 online/ 子树可见化与上列三接替文件的
12 例；pytest 口径不受影响）。

## 5. 目录导览（关键路径）

- `astra-sim/workload/execution_driven/`：在线机制层（C++）
- `sh_test_mesh/workload/llama2_7b_inference/online/`：在线调度器/构图器/服务层（Python）
- `.../online/verify/`：对账与验证工具
- `sh_test_mesh/run_scripts/`：全部 runner 脚本
- `sh_test_mesh/workload/llama2_7b_inference/traces/`：物化器脚本（数据件由调用方物化，provenance 以物化器 stdout 为准）
- `sh_test_mesh/slo_tools/`：SLO 离线后处理工具集（slo_stats / load_imbalance / restore_decomposition / kv_cache_adapter / hopbytes + `slo_postprocess_driver.py`（A4 单遍合并驱动，run_slo_postprocess.sh 链内使用；工具 CLI 不变）+ `slo_params_manifest.json`（B 类参数唯一来源，B4 已填推导值）+ tests；纯离线只读，详见目录内 README.md）
- Python 单测三根（与 `run_python_unit_tests.sh` 的测试根清单一致）：
  `sh_test_mesh/tests/`（66 例）+ workload 根含 `online/`（def 口径 163 例，
  其中 `online/` 74 例）+ `slo_tools/tests/`（181 例，含 2026-10-01 新增的
  transfer 字节门回归钉死 9 例），def 口径合计 410；
  pytest 实跑收集口径为 66+179+181=426（workload 根的
  `test_kv_delta_journal.py` 存在测试类继承，pytest 将基类用例在每个子类
  下重复收集，故实跑数大于 def 数）——两种口径都在 `run_python_unit_tests.sh`
  实跑下全绿，基线计数以该脚本实跑为准（2026-10-01 更正：旧"161/399"为
  历史残留）；
  §2 ② 物化后 trace_config 指向真实队列时，`test_wsc_llm_scheduler.py` 的
  request-neutral 占位断言红为已知环境效应，与代码态无关——还原裸仓即绿
  （2026-09-30 更正：旧"24+33=57"两根口径为历史残留）
- 回归入口分工（2026-09-27，sched-D1/fluid-D5 移植自 wscllm-LRU 同款）：
  `tests/run_all.sh` = C++ ctest 统一入口（canonical 聚合树全量 ctest，
  含根级 OfflineGreedy journal 测试——其 option 缺省已翻 ON；die 推广
  （2026-10-03）起全量 ctest = 45 项（30 项 analytical 既有 + 1 journal +
  11 die_model 库 + 3 die 前端新增：DieModelAdapterTest /
  DieExecutionLifecycleTest / LocalHbmDieMemoryTest）；canonical 树
  在场时瞬开两个 `ANALYTICAL_BUILD_*_TEST` option、只构建两测试目标、ctest
  后经 EXIT trap 还原 OFF，树缺席显式 SKIP——2026-09-28 起 SKIP 路径额外
  打印 `RESULT=SKIP reason=build-tree-missing` 机器可读标记，退出码仍为 0，
  以退出码/文本判断的调用方据此区分 SKIP 与 PASS；聚合顶层 CMakeLists 需含
  `enable_testing()`——见 §2 引导片段，缺失则 ctest 零用例空转）；
  Python 套件批跑由
  `sh_test_mesh/run_scripts/run_python_unit_tests.sh` 承担（逐根 pytest、
  任一根非 0 即整体非 0；pytest 不可导入时——官方镜像为
  standard-library only——自动降级 `python3 -m unittest discover` 逐根
  执行并打 NOTE；套件并非全为 unittest 风格：另含 2 个函数式测试文件
  （`workload/llama2_7b_inference/test_plan_materializer_passthrough.py`
  与 `.../online/test_propagating_tail.py`，裸 `def test_*` + 普通 assert，
  `unittest discover` 从中收集为 0），由脚本 FUNCTIONAL_TESTS 注册清单
  在两种模式下显式直跑兜底、零损失；pytest 仍为权威 runner）。die 推广
  （2026-10-03）的 Python 增量：新 `online/test_die_model_adapter.py`（20 例：
  span 恒等式/13 类映射/融合条件矩阵/spec 形状/from_service 门），
  `test_config_resolver.py` 增 DieHardwareTests（11 例，库可发现性门控制
  skip）、`test_graph_batch_builder.py` 增 DieBackendEmissionTest（4 例）、
  `test_weight_passes.py` 增 die 守恒对拍、`test_metrics_contract.py` 增
  DieUtilizationPipelineContractTests（9 例）；checked-in 配置装载类用例
  （test_wsc_llm_scheduler 4 例）带 .so 可发现性 skip 门（发现序同 §H，
  不用 mock）——裸仓在构建出 .so 前为合法中间态（skip 不红），带库全绿
- `examples/`：上游 astra-sim 示例残留（本仓离线静态管线已退役，目录内无任何
  可执行入口），保留仅作参照（2026-09-27 登记；姊妹仓同态保留，生产链路零引用）。
  `examples/system/custom_collectives/` 已随 2026-09-28 死代码清理批删除：自定义
  集合通信/ET feeder 路径（feeder_v3 + CustomAlgorithm + CustomCollectiveImpl）整体
  退役，system.json 中任何 `*-implementation-custom` / `per-node-custom-implementation`
  键在解析期 fail-closed 报退役（critical + exit 1）。native 四键
  （`all-reduce/all-gather/reduce-scatter/all-to-all-implementation`）仍为
  「未配置的集合类型不建拓扑」——但发射该类型时拓扑查找即 fail-closed
  （`Sys::get_logical_topology` critical + exit 1，报文指名集合类型与缺失/为空的
  system 键；2026-09-29 前：缺键为运行期数字型 runtime_error、键配空数组为
  `operator[]` 空指针裸段错误，均无此诊断）

## 6. 边界与纪律

- 仿真输入唯一允许源 = astra_compute_20.csv 前 30 秒（arrival_time <
  30e9 ns；物化器 WINDOW_NS=30e9 缺省即此口径，2026-08-15 冻结——更早
  "前 2 秒"表述为文档残留。冒烟实测可显式传更小 window_ns（如 2e9），
  验收对账口径 = ledger_reconcile 缺省值（1177/112，见 §2 ② D3 登记））。
- 缺失输入一律 fail-closed（generate 桩/materializer/runner/GEN_MATCH 均实测 exit=1）。
- 策略文件（wsc_llm_scheduler.py / session_kv_manager.py）为保留对象，勿改
  （2026-09-25 深挖二轮唯一例外：两文件各删一处全仓零引用的死方法
  `d2d_to_hbm_bandwidth_ratio` / `final_session_counts`，策略语义零改动）。
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
  复位 P 实例 busy 并登记等待 decode 准入（dirty 门控下重查——等待 decode
  准入的 epoch 门已随 SCHED-1（2026-09-28）以调用图不可达删除，epoch 重试
  现仅存在于 prefill 准入；固定静态映射不 remap）；`_on_decode_complete` 落 KV 完成账本（拼 batch
  改造后 active_decode 出队与 busy 复位移至列车核销）；`_on_request_complete`
  排下一 turn 的 arrival alarm。准入预占按净额+迁移后回补（2026-09-06）：
  prefill 准入在静态 decode 目标预占终态 KV 时，若全量预约因本会话旧驻留
  重复计入而 deep-gap、且旧 KV 仍 RESIDENT 于该 decode 实例，则按净额
  （终态−旧驻留）重试；`prepare_history` 迁移删除旧驻留后经
  `extend_request_capacity` 回补到全量（防抢占语义保持，resident→reserved
  1:1 换位，任何瞬间不超订）。
- **上条净额预占的由来（2026-09-06 准入双重计账楔死故障，6×8 拓扑全量首跑
  实证；分析文档已随修复归档不在仓内，结论摘要见本节）**：全量 22816 请求
  首跑提交至 22740 后楔死——4 个终态 85万–105万 token 超长会话被静态 P→D
  路由各钉死在一个 decode 实例，准入重试永 insufficient 且无事件可唤醒重试，
  C++ 侧事件队列+决策邮箱全空后由统一死端守卫 fail-closed
  （`main_online.cc` 的 dead-end 守卫，安全网本身正确、未改）。根因是记账错误而非容量
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
  fail-closed（`online/graph_batch_builder.py`）。**注意力 QK/AV 按全
  (tokens×kv_length) 矩形计 FLOPs/字节，未扣因果掩码三角（D3 登记，
  2026-09-28，既知建模近似，勿"修"）**：与拆分前展开式逐字节等价（单测
  钉死）并进冻结 30s 冒烟基线与五仓横向对齐；三角折扣只在每 chunk 对角块
  内（p_chunk=512 孤立 chunk 至多高估 2×，带历史/前 chunk 上下文时
  kv_length≫tokens、高估远低于此），对 7B 总 FLOPs 影响 ≤0.1%。改口径 =
  全部黄金值/冻结基线/跨仓对拍作废，须走基线重签，不属缺陷修复。
  拼 batch 改造
  （2026-08-22）起聚合函数新增 `weight_passes` 参数（默认 `len(spans)`，
  与历史 batch=1 串行口径逐字节一致；迭代列车传迭代数：一个迭代同时算
  B 个成员各 1 token，**权重每迭代只读一次**，激活/KV/AR 逐 span 精确；
  `online/test_weight_passes.py` A0 夹具钉住 B=1/B=2 同迭代权重字节相等）。
  D 侧 decode 发射为实例迭代列车（跨成员共享体节点，见下），P 侧 prefill
  保持整段聚合（chunk 序列，每 chunk 恰读一遍权重，chunk 之间不互拼）。
- 发射并发骨架（拼 batch 改造形态，2026-08-22；§3.6 PD 分离豁免 = 仅
  decode 互拼、无混合迭代）：busy 门按 phase_role 分裂——prefill-only
  实例保持"一个 prefill 整段在飞"（逐 chunk 不拼，发射骨架与改造前一致）；
  decode-only 实例 = "一个迭代列车在飞"（`in_flight_train` 冻结成员快照，
  含 exit/signal 集合；2026-10-01更正：原文此处提及的 membership_digest
  摘要字段从未在本仓代码中存在）。准入/发射 pass 只服务空闲实例：D 实例把全部
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
- 折入 recompute 输入口径 + 运行时 KV 账本：请求队列为唯一仿真输入，
  turn-0 源前缀折入该行 prefill_length（整段重算口径，
  `traces/derive_20_first_30_seconds.py:14-16`）；manifest 仅携带队列派生的
  history_tokens_before 推导值（`plan_materializer.py` 的
  `_derive_manifest_requests`）；会话 KV 的
  规模与位置由运行时 `SessionKVCacheManager` 账本动态维护
  （`wsc_llm_online_scheduler.py` 的账本装配处；`session_kv_manager.py` 的
  `SessionKVCacheManager` 类）：驻留
  命中直接复用、跨实例历史先经 NoC 迁移段、被逐出则窗口内重算。
- long double 时间精度适配：大 ns 级首达偏移超出 IEEE-754 double 精确整数
  范围（2^53）时，分析网络适配层返回 ASTRA-sim 时间以 `long double` 保持
  64 位事件时间精确，避免完成回调与事件映射键错位 1 ns
  （`astra-sim/network_frontend/analytical/common/CommonNetworkApi.cc:101-110`）。


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


## 数字来源表（SDK 数字的唯一来源链；2026-10-17 P12-A 落档）

本仓所有 die/coarse 相关数字的来源、身份与失效条件。**发布身份由
`extern/die_model/RELEASE.json` 的 source_digest 承载**（configs/VERSION 的静态
字符串不是发布身份）；当前五仓一致 digest =
`3a021f37b71fb3ea52a02a0c522169cd5a8374ef1f77c2d6a20fe504b274c31c`（33 文件；`die/tools/sync_repositories.py --check` 2026-10-17 实测 passed；canonical25 正式发布声明见 die/docs/RELEASE-canonical25-20261017.md）。

| 数字类别 | 仓内位置 | 唯一来源链 | SDK 身份 | 门槛与失效条件 |
|---|---|---|---|---|
| 单 op 估计时延（estimate_latency_ns 等） | `sh_test_mesh/workload/llama2_7b_inference/online/die_model_adapter.py`（build_spec → C ABI） | emitter → libdie_model_cabi → `extern/die_model/coarse` | digest 见上 | digest 或 plan fingerprint 变更后全部数字失效重算；非整数时延 fail-closed |
| lifecycle 合同（含 R4 重复 region） | `astra-sim/workload/execution_driven/tests/die_execution_lifecycle_test.cc` | 测试 → vendored SDK 编译链接 | digest 见上 | guard fail-closed：prediction_status=not_available 即失败；legacy ×R 聚合保持 not_available |
| 生成式 repeat（R≥2 ffn_region）声明状态 | 同上 adapter（build_spec 条件声明） | emitter → die_op_spec repeat_seams → Plan v5 | digest 见上 | **是**——adapter 对 layer_count×weight_passes≥2 的 ffn_region 条件声明 `repeat_seams={mode:serial_on_drain}`（仅 ffn_region+物化模板；其他 op/策略 fail-closed）；prediction_status=repeat_seam_modeled |
| trace/adapter 合同测试 | sh_test_mesh/workload/llama2_7b_inference/online/test_die_model_adapter.py（本仓无 face 族 D 阶段 oracle 客户端，属合法可选依赖） | adapter 单测 → vendored SDK 符号 | digest 见上 | adapter 合同断言；不依赖外部 oracle 模块 |
| 硬件标量（derive_coarse_die_hardware） | `sh_test_mesh/config_resolver.py` | manifest → C ABI derive → 本仓派生标量 | digest 见上 | manifest 或 SDK digest 变更后重验 |
| et_def Python 绑定（et_def_pb2.py） | `extern/graph_frontend/chakra/schema/protobuf/et_def_pb2.py` | 跟踪的 et_def.proto + **钉版 protoc 3.21.12** 仓根调用再生成 | n/a（仓内跟踪） | `verify_et_def_pb2.py` + `sh_test_mesh/tests/test_et_def_pb2_consistency.py` 防漂移；drift 即失败，protoc 缺席显式 skip |

### 环境与目标名注记（P13/P14）

- **protoc 钉版 3.21.12**：再生成/校验 et_def_pb2.py 必须使用该版本（仓根调用
  `protoc -I . --python_out <tmp> extern/graph_frontend/chakra/schema/protobuf/et_def.proto`）。
- **ctest 目标名以 `ctest -N` 输出为准对账**。已知族间差异：admission 测试目标在本仓族为
  ``AstraSim_Analytical_WorkspaceAdmissionTest`（无 Congestion_Aware 前缀）`；其他仓族命名不同（face 族带 `Congestion_Aware_` 前缀、joint 无此目标），
  按 `ctest -R` 前缀过滤会静默漏测。

## 出厂计算后端：coarse_die_v2 die 内仿真器（2026-10-18 阶段2 翻转批）

本仓出厂默认计算后端为 **coarse_die_v2 单 die 解析仿真器**：system 模板
`die-model-profile: coarse_die_v2` + 硬件 `coarse-die` 节（完整 coarse_die_v2
描述符）+ 容量档 `validation-coarse-144g`（恰等于 coarse 派生每-die HBM 容量
144 GB，resolver fail-closed 对齐）。真实重复区域（layer_count×weight_passes≥2）
以声明式 `repeat_seams={mode: serial_on_drain}` 发射，coarse 编译器按单实例
模板建模、运行时逐实例串行推进并以逐实例区域开销为 seam（引擎 2026-10-18
实例 repeat 通用化批，见 template/die/docs/HANDOFF.md）。旧 `mesh_die_v1` /
`tx8_reference_v1` 保留为显式 legacy 选项（硬件 `die` 节 + `validation-die-64g`
档配套），coarse 与 roofline-enabled=1 互斥 fail-closed。

已知边界：运行时推进路径保留 P8 记载的 O(R²) 活动向量墙（P7-B 遗留项），
大 R 计划的在线排空 wall 随 R² 增长；估计/编译侧已有 P9 闭式快路径。
