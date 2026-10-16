# astra-sim-face - 晶圆级芯片（WSC）LLM 推理架构与机制

> 本仓为 Execution-Driven 改造后的**裸仓库终态**：仅保留路径③（strategy 关感知）
> 与路径④（strategy 开感知）两条在线仿真路线；离线静态（①）与 replay（②）
> 已删除。仓库 request-neutral：不带任何 request 队列，正式入口缺失输入
> fail-closed。
>
> 本仓是 ASTRA-sim 2.0 的晶圆级芯片（Wafer-Scale Chip, WSC）推理仿真改造仓。
> 四个同源仓 `astra-sim-face / astra-sim-wscllm / astra-sim-sh_2.0 / astra-sim-sh_3.0`
> 共享同一套硬件架构建模与同一套术语，仅在"请求→实例映射与 KV 管理策略"上分化（见文末对照表）。
> （2026-09-28 注：sh_2.0/sh_3.0 已于当日工作区整备时移出 template/，此处保留
> 历史谱系叙述；现行 template/ 下为 face/wscllm/joint 及两 LRU 仓。）
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
链路带宽与每跳时延取硬件配置 `d2d` 的值（时延沿多跳逐跳累计），XY（维序）确定性路由，
传输计时为流体模型（FluidScheduler，transmission-mode 仅接受 `fluid-pipelined`）：
共用同一链路的多条数据流按活跃流数严格均分链路带宽（拥塞感知），完成时刻 =
数据量 ÷ 路由瓶颈处的均分速率 + 全路由逐跳时延之和（Σ 每跳时延一次性累计）。
C++ 侧由分析型网络后端实现（`extern/network_backend/analytical/congestion_aware/`），
Python 规划层有同一语义的确定性 XY 路由函数。注意：这是分析级模型，不建模路由器
微架构（crossbar / VC / credit），也不建模逐跳 store-and-forward / FIFO 排队等
排队微结构，也不做 cycle-accurate 仿真。

### C. 芯粒（die / NPU / rank）＝挂在 NoC 上的每一个单元

挂载在 NoC 上的**每一个单元**即一颗芯粒。代码与配置中统一称 **NPU**（或 rank），
权威映射为"一个 FACE 物理计算 die = 一个 ASTRA-sim NPU"。每颗芯粒包含：

- **计算单元**：生产配置（硬件 `die` 节）下每颗芯粒是一个 die 内成本模型
  （`extern/die_model/`：32 行 Core 资源表 + HBM/面积/mesh 派生 + 阶段化资源调度），
  COMP 节点按 `die_op_spec` 编译为不可变 DiePlan 并由阶段状态机计时执行
  （三后端与配置链见 §H）；显式 legacy 后端（`compute` 节 + roofline）保留
  旧 Roofline 闭式模型供回归；
- **本地 HBM**（KV 热层）：带宽 / 时延 / 容量取硬件配置 `local-hbm` 的值（容量按
  profile 档位选择），作为**多用户共享带宽资源**仿真（见 G 节）；
- **网络接口（路由端口）**：mesh 上的一个 Device，经有向链路连接其在 mesh 中的物理邻居；
- **远端内存端口**（仅边缘芯粒拥有，且仅在启用远端内存扩展的配置中生效，见 D）：
  与片外统一内存池直连的中转通道，对应物理系统中的 SerDes 等片外互连。

> 术语辨析：本说明中"芯粒"＝mesh 上的完整单元（die＝NPU＝rank）。指标层另有一个
> 仅用于观测展示的"chiplet 投影"（把每 NPU 的本地 HBM 均分为若干份，份数由
> 跨仓 manifest 的 chiplets 数组携带——2026-09-26 起 metrics_config.json 已裁剪
> 为仅 `schema_version`/`detail_level` 两键，`chiplets_per_npu` 配置键删除，
> metrics_schema 的 `DEFAULT_CHIPLETS_PER_NPU` 常量亦已随死代码清理删除，
> 本仓已无任何内置默认份数，对应论文场景中的 HBM 芯粒划分），
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
  端口之间完全并行，远端池总容量不设限。此段为跨仓模型描述，仅适用于仍实现
  远端内存后端的仓（原方案《SerDes片外链路并发化改造执行方案》所辖其余仓，
  实现于各自仓的 `extern/remote_memory_backend/`）；本仓无该机制（见下条 A.2
  声明），astra-sim-wscllm 亦已于 2026-09-24 整链移除、与本仓对齐。
- **本仓配置**：远端内存机制已整体移除（2026-09-05，A.2 清除）：本仓既不启用也不保留
  远端内存建模代码，硬件配置声明任何 memory expansion 会在
  `config_resolver.py` 解析层 fail-closed 拒绝。因此本仓 KV 只有片上一层——全部驻留本地
  HBM，容量不足时按 LRU 整 session 零代价删除，历史 KV 恢复一律重算（recompute）。

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

### F. 本仓在四仓中的定位

> **并列对照，防误读**：下表为跨仓并列比较——除加粗的"本仓"行外，各行描述的是
> **对应姊妹仓**的机制（"三态（含半驻留 PARTIAL）"、"两阶段类型感知逐出"、
> "远端内存池启用"等表述仅适用于那些仓）。本仓恒为 RESIDENT/EVICTED **两态**、
> 无远端内存，请勿把姊妹仓行为读成本仓机制。

| 仓库 | 实例组织 | 请求→实例映射策略 | KV 驻留与恢复 | 远端内存池 |
|---|---|---|---|---|
| **astra-sim-face（本仓）** | 统一实例（P+D 同实例） | FACE 原始映射：prefill 选剩余 chunk 最少；decode 在邻接图加权距离限制（阈值 = D2D 带宽 / 本地 HBM 带宽）内按 per-die 增量代价（生产走 die 模型 oracle 孤立估值，legacy 后端走旧 Roofline 公式，见 §H） | RESIDENT/EVICTED 两态；LRU 逐出＝零代价删除；恢复＝重算 | 不支持（机制已整体移除） |
| astra-sim-wscllm | PD 分离（Prefill-only + Decode-only 分区） | prefill 选排队请求最少；decode 用静态一跳 P→D 映射 | RESIDENT/EVICTED 两态；LRU 逐出；恢复＝重算（跨实例历史走 NoC 迁移） | 不支持（机制已整体移除，2026-09-24） |
| astra-sim-sh_2.0 | 统一实例 | prefill Roofline 剩余负载均衡（历史 KV 全/部分驻留与全逐出统一）；decode 按 per-die Roofline 增量代价 + HBM 剩余 tie-break | 三态（含半驻留 PARTIAL）；两阶段类型感知逐出（human 类先于 tool 类）；流水化部分恢复 + HBM 恢复/推理带宽共享 | 启用（全部边缘芯粒挂端口） |
| astra-sim-sh_3.0 | 统一实例 | 三段式 prefill（首请求避边缘 / HBM 命中 sticky / 远端命中负载均衡）；decode 本地化固定同实例 | 三态；两阶段逐出；流水化部分恢复（机制同 sh_2.0） | 启用（全部边缘芯粒挂端口） |

**删除型逐出的执行口径（2026-09-13 补记，逐出/推理并行化方案 §4.5 基线仓定位）**：本仓逐出为
**删除型**——容量压力下 `SessionKVCacheManager._delete()`（`session_kv_manager.py`）
只做即时台账扣减（shard 字节出账 + 会话置 EVICTED）与决策日志 `evict_delete` 事件记录，**无 ET
图节点表示、无 DMA 传输、无 HBM 带宽消耗**（类 docstring 明示 `evict_delete` "intentionally
has no ET representation"），因此逐出与同 rank 在跑的推理计算**不存在物理资源争用，天然并行**。
被逐会话的后续到达按**全量重算**处理：`history_recompute_tokens` 直接折进 prefill 计算（决策行
`effective_prefill_tokens = prefill_length + history_recompute_tokens`，
`face_online_scheduler.py`；recompute 段与当前 prefill 段共用同一 chunk 机制进入
迭代列车），重算段是普通 COMP 节点，照常参与本地 HBM 带宽多用户均分模型（§G）。本仓唯一的
物理 KV 流量是活会话跨实例迁移 NOC_MIGRATE（history 驻留迁移与 prefill→decode 迁移，恢复类
语义：迁移节点物理先于使用它的计算）。与姊妹改造仓 astra-sim-face-LRU（下称 -LRU 仓）的
物理逐出/回迁链路形成对照（**物理逐出仅存在于该姊妹仓**）：后者逐出是真实物理传输，其与
推理计算的并行化改造见该仓 README；本仓无可并行化
的逐出物理链，不涉及也未做此类改造。

四仓还共享同一套 Execution-Driven 在线仿真机制层（`astra-sim/workload/execution_driven/`：
RequestIngress / DecisionMailbox / DecisionBridge / GraphBatchCommitter 等）：策略决策由
Python 在线服务层实时给出，计时由 C++ 物理时钟推进；各仓仅保留路径③（strategy 关感知）
与路径④（strategy 开感知）两条在线路线。四仓在线决策均不再使用任何离线 LUT
（残留 LUT 查表已随静态链路一并删除）：face 的 decode 候选代价由在线成本
模型即时计算（`per_die_delta_ns`）——生产（die 后端）走 `die_model_oracle`
（ctypes 直连 `libdie_model_cabi.so` 的批量孤立估值），显式 legacy 后端保留
旧 Roofline 闭式公式回归路径（`estimate_iteration_time_ns` 的 `cost_oracle=None`
默认形，逐字节不变），决策确定可复算复放。

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
**此处"错误响应"仅指交付（delivery）响应路径**——commit-ack 为单向通道：C++
不读取任何 ack 响应，ack 无响应/失败即 Python 非零退出 → resp_notify EOF →
peer-death abort（2026-09-26 契约冻结，禁止新增等待）。
Python 生产者（graph_batch_builder）本就满足全部键集，行为不变；三个 verify
fixture 服务（same_tick_milestone / wakeup_guard / lifecycle）原先发射的空
`comm:{}`/`coll:{}` 已补满形缺省。数组顺序契约：六个数组与 watch 成员一律按
发射序保序，解析层不排序不去重。

### G. 本地 HBM 带宽模型（多用户竞争，`hbm-bandwidth-contention`）

真实硬件中每颗芯粒的本地 HBM 是共享资源：同一时刻的所有访存流量竞争同一份带宽。
本仓用每 rank 一个 `LocalHbmBandwidthModel`（`astra-sim/workload/LocalHbmBandwidthModel.{hh,cc}`，
由 sh_2.0 的两用户版泛化为 N 用户）建模，system.json 键 `hbm-bandwidth-contention`
（bool，代码默认 true；置 0 回退无仲裁行为——legacy 后端下 COMP 走 Roofline 闭式公式、
comm 不计 HBM；die 后端下计算侧内存腿由 DieExecution 按满速率自驱 TIMER（见 §H 与
下方第 1 条）、通信侧沿用旧开关语义；`local-mem-bw <= 0` 时**该旗标**自动视为 false，
防零速率卡死——澄清 2026-10-01：自动关闭仅作用于 hbm-bandwidth-contention 旗标；die
后端下 `local-mem-bw` 由 resolver 从 die 节派生写入（正值），用户未显式写 contention
键时恢复文档缺省（开），不静默落入无仲裁臂。native 集合通信的归约腿（PacketBundle
needs_processing 路径）在任何配置下都要求正值 `local-mem-bw`，缺省 0 时首个归约腿即
fail-closed panic，属预期防误配行为）。参与竞争的流量（本仓无片外内存池，
只有这两类）：

1. **计算访存**（按计算成本后端分派，见 §H）：
   - **die 后端（生产）——`DIE_COMPUTE` 服务量作业**：DieExecution 把 DiePlan
     内存阶段的请求三元组交本模型仲裁（`issue_die_memory`）。作业排空量是
     **service_units**（＝实际字节 / 该请求类别的效率系数 η，按服务量折算的
     HBM 占用时长的统一量纲），模型按均分速率排空 service_units；实际推进的
     物理字节 = service_units × η（**actual_bytes 与 service_units 分列**，
     绝不回填 tensor_size/D2D 载荷——模型内两列 side-band 计数
     `die_service_units_served` / `die_actual_bytes_served` 分列累计，由
     `die_execution_lifecycle_test` / `local_hbm_die_memory_test` 数值断言）。
     **无 FLOPs 腿、无 latency 相**：die 后端的算力计时在 DieExecution 的
     PE/VEC/NOC 阶段状态机内（不进 HBM 模型），`local_mem_latency` 由
     DieExecution 以每外层节点至多一次的一次性 TIMER 在首个内存请求前收取。
   - **roofline 显式 legacy 后端——旧 COMP 字节作业**：bytes = tensor_size
     （读写在同一标量上合并计费），FLOPs 按 peak-perf 并行排空，两者都排空才算
     完成（单用户时保持 Roofline max(计算, 访存) 语义）。若 ET 带**非 0 校准
     runtime**（在线 LUT 时钟路径）则不进流体模型、保持校准时长——本仓 trace
     的 COMP 无 runtime，实际总走流体路径；该分支为在线 LUT 兼容保留。
2. **NoC p2p 通信的数据端点**：发送方 rank 的 HBM **读** + 接收方 rank 的 HBM **写**
   （bytes = comm 字节数）。多跳路径途经的中间芯粒**不占 HBM**（路由器直通——网络后端
   本来只在端点计费，模型与此对齐）。节点完成 = join(网络侧完成回调, 本端 HBM 作业完成)，
   两事件都到齐才触发，且恰好触发一次。ET comm 节点可用布尔属性 `hbm-charge`
   （默认 true）让该端点退出计费；bytes == 0 不建作业；1B ACK 照常计。

仲裁规则：某 rank 同时有 N 个 HBM 用户 → 各得 full_rate/N **严格均分**；任一作业完成
立即释放并把带宽重分配给剩余用户（事件驱动重分配，无固定时间片）。带宽直接用配置值
`local-mem-bw`（读写共享同一总线、共享总带宽的**单一标量**，不区分读写方向，也无
峰值/持续带宽之分）；`local-mem-latency` 按 sh_2.0 惯例在每个作业启动时计一次。
TP 集合通信（PacketBundle 3x 路径）**不在**本模型范围内，行为不变。

指标层新增 `type=local_hbm` 导出（每 rank 一条，`[METRIC]` 行）：HBM busy_ns、按类别
served bytes（compute / comm_read / comm_write）、峰值并发作业数、均分重分配事件数、
真实 HBM 利用率（busy_ns / 墙钟窗口）；旧指标键（rank_compute 的 dram_bw_util 等）语义
不变（die 后端下 compute_bytes_served 只含 marker 微作业的 1/1 字节——真计算访存走
DIE_COMPUTE 服务量作业，不计入该列）。数值断言级验证见
`astra-sim/workload/execution_driven/tests/local_hbm_bandwidth_model_test.cc`
（1/3 → 1/2 → 全速的均分序列、join 两序、单次触发）。

### H. 计算成本后端（die 生产 / roofline / replay legacy）与 die 配置链

COMP 节点的计时由 system.json 键 `compute-cost-model` 选择，共三后端：

| 后端 | 取值 | 语义 |
|---|---|---|
| die（生产） | `die` | 真实计算 COMP 必须携带 `die_op_spec`（GraphBatch compute 子对象；ET 侧为同名 JSON 字符串属性）；`Workload::issue` 分发序：timer op → 旧 timer 路径；cpu op → replay；`op_role=control_marker` → 旧微作业语义（legacy HBM 模型仍为 `issue_compute(1,1)`；coarse_die_v2 有 HBM facade 时以 `owner_kind=die` 提交唯一 1-byte read endpoint，并将 `local-mem-latency` 作为其启动延迟；无 HBM 模型仍为 1ns 事件）；有 die_spec → DieModelAdapter（编译/缓存 DiePlan + 每节点一个 DieExecution 状态机驱动执行）；真 COMP 缺 die_spec → **fail-closed throw，绝不静默回退 roofline**。`runtime_ns != 0` 与 die_spec 并存被 parser 拒绝 |
| roofline（显式 legacy） | `roofline` | 等价 `roofline-enabled=1`：COMP 走旧 Roofline 闭式/流体路径，保留供回归（§2 配置链整体切换方法：硬件 JSON 换回 `compute` 节 + `validation-160gib` 档 + roofline system 模板） |
| replay（显式 legacy） | `replay` | 旧默认分发（无成本模型） |

冲突矩阵（Sys 与 resolver 双层 fail-closed）：`die` + `roofline-enabled=1` → 报错；
`die` + `replay-only=1` → 报错；无 `compute-cost-model` 且 `roofline-enabled=1` → 旧
roofline（向后兼容既有模板）；非 die 后端携带 `die-model-*` 键 → WARN。

**控制 marker 语义**：发射侧把一切 `comp(name,1,1)` 微节点（first_token / drain /
exit / sentinel / join / pstart / wakeup 等）显式标 `op_role="control_marker"`
（缺省即 compute）；legacy HBM 下保留旧 `(1 op, 1 byte)` fluid job 与原 latency，
coarse_die_v2 下由同一 canonical Runtime 计入一字节 `die` endpoint 并延迟
`local-mem-latency`，仍不创建 typed COMP Plan；无 HBM facade 时保留 1ns 事件。
销毁 HBM facade 时只读采样已服务字节前缀并以静默取消回收 marker，不分发 Runtime 中
已就绪的 graph completion callbacks。
`compute_bytes_served` 兼容 getter 继续累计 marker 已服务物理字节。本仓 marker 唯一
发射点：`graph_batch_builder.py` 的 `_emit_train_marker`。

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
   在 die 节下会被 resolver 拒绝）。
5. manifest 链：plan_materializer 把 `compute-cost-model` / `die-model-config` 写入
   manifest.json → Python 在线服务按 manifest 构造发射语境（DieModelContext）与
   调度 oracle（DieModelOracle），"发射走 die、预测也走 die"，禁止半新半旧。

**共享库发现序**（ctypes 加载 `libdie_model_cabi.so`，resolver 与 oracle 两处同约定）：
env `DIE_MODEL_CABI_PATH` → 仓内
`build/astra_analytical/build_congestion_aware/lib/libdie_model_cabi.so`（再退
`bin/`）→ 失败 **fail-closed 报错（绝不静默回退 roofline）**。库随 §2 ① 全量构建
产出（`die_model_cabi` target，输出目录锚定构建树根的 lib/，恰为上述仓内路径）。

die 后端的利用率口径（`OnlineStatisticsState` 新增 `vector_utilization`）：
compute = FLOP/(PE 峰值×elapsed)、memory = 实际字节/(HBM 带宽×elapsed)、
vector = eq-ops/(向量峰值×elapsed)；legacy 统计路径不含 vector 列（恒 0）。
**指标出口（阶段 E2）**：每 rank 的 `type=rank_compute` [METRIC] 记录新增
`active_kernel_die_{compute,memory,vector}_util` 三键（门控 =
`compute-cost-model=die`，与 roofline 键的 legacy 门控互不影响；无 comp 节点
的窗口按 null+note 报告）；metrics_postprocess 在后处理侧 fail-closed 断言
"声明 die 后端的 run，每 rank 三键非 null 且 ∈[0,1+ε]"（metrics_manifest.json
由 plan_materializer 同步写入 `compute-cost-model` 声明，归档 run 离线可判）。

**性能注记（阶段 E2）**：
- **进程级共享 PlanCache**：`Sys::DieModelRuntime` 的计划缓存按
  `plan_cache_hardware_key`（库新接口，=plan_cache_key 的硬件子串）索引到
  进程级单例注册表——compile 是纯函数、DiePlan 不可变，同硬件的 54 个 Sys
  共享一份 LRU（键已含全部区分字段，命中语义不变；单线程仿真无锁）。
- **evaluate_stages 重写**（extern/die_model/src/CostModel.cc）：冻结的
  CSR+堆 Kahn 改为按 stage 头扫描（弹序与逐实例算术逐操作相同，精确路径
  结果与旧实现 bitwise 一致；差分模糊 ≥1000 plan/族钉住等价），另带
  带验证回退的稳态直线快进（隔离调度无外部事件，§8.5 惰性展开的压缩版）。
- **DieExecution 纯 timer 段合并**：连续内部 Timer 动作在
  LocalHbmBandwidthModel 报告"无活跃作业（无非 DIE 用户+无未决 DIE 请求）"
  时合并为单个 Sys 事件（§8.5 外部竞争集合不变的守卫；内存请求边界绝不
  跨越；contention=0 无仲裁器恒可合并）。
- **候选剪枝下界修正（阶段 E3）**：阶段 A2 的编译期物理下界把
  launch（L×P 个 1μs TIMER 实例）按串行前缀计入，而 TIMER 不占资源槽，
  真值中全部 launch 实例 ~1μs 内并发完成——旧下界可超过候选真 makespan
  并剪掉真最优（E2 selection fuzz 实测）。E3 换为健全下界（每资源槽
  busy 和 + 槽前缀+busy + lag 折算关键路径，小实例数候选走含前向 lag
  边的精确关键路径），剪枝不再可能剔除 §5.1 胜者（selection_fuzz 测试
  常驻钉住）；副作用全部为正：大形状 pick 提速 6-37×、1s die 冒烟
  wall 5:01 → 2:34、个别 decode/微型形状选中更短计划（估值下降）。

## 1. 本仓是什么

- **策略语义（保留对象，未改动）**：FACE 原始映射：加权实例图 + per-die Roofline 增量代价选择 decode 实例（平局轮流裁决）；RESIDENT/EVICTED 两态 + 近似 LRU **删除型**逐出 + recompute 恢复（本仓逐出零物理代价，与姊妹仓 astra-sim-face-LRU 的物理化 LRU 改造无关，见上"删除型逐出的执行口径"）；无远端内存；prefix 走 recompute；session 驻留状态/字节数由运行时 KV 账本（SessionKVCacheManager）动态维护，无 sidecar
- **执行驱动机制层**（`astra-sim/workload/execution_driven/`）：在线事件驱动
  （RequestIngress/DecisionMailbox/WatchRegistry/GraphBatchCommitter/长连接
  DecisionBridge 等），四仓接口一致。
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
#    enable_testing()
#    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../ astra_sim_build)
#    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../extern/network_backend/analytical backend_build)
#    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../astra-sim/network_frontend/analytical frontend_build)
#    EOF
cmake -S build/astra_analytical -B build/astra_analytical/build_congestion_aware -DBUILDTARGET=congestion_aware -DNETWORK_BACKEND_BUILD_AS_LIBRARY=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build/astra_analytical/build_congestion_aware -j

#    （`-DCMAKE_BUILD_TYPE=Release` 为阶段 E2 起：缺省空 build type 会以无优化
#    编译，die 后端的计划编译在该形态下慢约一个量级；Release 语义上只有性能
#    差异——无 fast-math，IEEE 浮点行为与数值结果不变。）

#    ①b die 成本库 C ABI（die 后端/Python 侧 ctypes 需要；默认全量构建已含，
#    亦可单独 --target die_model_cabi）。共享库落在
#    build/astra_analytical/build_congestion_aware/lib/libdie_model_cabi.so
#    ——即 Python 侧发现序的仓内路径（见 §H；env DIE_MODEL_CABI_PATH 可显式覆盖）：
cmake --build build/astra_analytical/build_congestion_aware --target die_model_cabi -j

# ② 物化输入（唯一允许源 = agent-traces/tracelab/astra_compute_20.csv；
#    冒烟窗口按工作区根的 Agents.md（本仓内无此文件，相对本仓根为
#    ../../Agents.md）之《冒烟测试与仿真效率规范》= 该 trace 前 1~10 秒、
#    优先 1 秒，即物化时显式传 window_ns=1000000000（按需 ≤10000000000；
#    须传纯十进制整数——物化器按 int() 解析，2e9 这类科学计数法字面量
#    会 ValueError）——物化器的
#    window_ns 缺省 30e9 ns 仅为工具缺省值，不是冒烟口径；
#    物化器：derive_20_first_30_seconds.py）
#    产物放 sh_test_mesh/workload/llama2_7b_inference/traces/，
#    并把 trace_config.csv 的 request_queue_csv 行指向它
#    （路径解析基准，2026-09-29 补记：request_queue_csv 的相对值以
#    sh_test_mesh/workload/ 为基准解析，故写
#    `llama2_7b_inference/traces/<物化产物>.csv`；写绝对路径则按原样使用、
#    不再拼接（解析器显式接受绝对路径，物化器 next-steps 提示打印的绝对
#    路径可直接粘贴）。基准拼错时产物文件找不到，装载器 fail-closed 报错
#    并提示物化入口，可据此自纠）
#    物化器 CLI：<source> [queue] [sidecar] [window_ns] [arrival_scale]
#    （source 必填，缺失 exit 2 并打印用法）；
#    arrival_scale>0 仅缩放 turn-0 session_arrival_time_ns（t0/scale，即
#    负载 ×scale；缩放结果四舍五入到最近 1000 ns 网格点以满足队列
timing%1000 闸门，偏差 ≤500 ns（恰半格点偏差＝500 且向上；2026-10-01 起
四舍五入为显式 half-up：此前 Python round 是银行家舍入，半格点向下与
文档承诺相反），scale=1 为 no-op）
#    inter_request_interval_ns（human/tool 外生等待）不动，
#    窗口判定与统计始终用未缩放源时间——scale=1 时 8 列队列与冻结基线
#    逐字节一致；缩放与 request_type 等新信息只进 canonical sidecar 与
#    stdout provenance，不进队列。

# ③ 生成 plan 目录（runtime_config 三件 + manifest + metrics_manifest）
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
#    归档交互（2026-09-29 补记）：SH_ARCHIVE_RUN 默认开启时，成功 run 的
#    cpp.log 已被归档步压缩为 cpp.log.gz——上面字面命令里的 <run_dir>/cpp.log
#    随之不存在；工具已支持直接读取 .gz 回退（与 run_slo_postprocess.sh 的
#    cpp.log → metrics.log → cpp.log.gz 同款回退口径），亦可按归档状态解压
#    后传入明文日志。
# ⑤b 自动 SLO 指标提取（P3，2026-08-28；A4/2026-08-29 单遍化）：在线
#    runner（strategy/sensing 两变体）在 postprocess 成功后、归档前
#    自动调用 run_slo_postprocess.sh——内部一次调起 slo_tools/
#    slo_postprocess_driver.py 单进程单遍驱动（9 步产物集/行序/公式/
#    slo_postprocess.log 与逐工具串行逐字节一致；读放大收敛：
#    request_metrics.csv 4 读→1、decision log 4 流→1、[METRIC] init 探测
#    4→1、解释器启动 9→1；六个工具 CLI 保持可独立调用）
#    ——full 档产 9 项最细粒度产物（slo_e2e_stats/
#    backlog/session/load_imbalance/restore_decomposition/hopbytes_total+
#    per_request/hbm_watermark 三件=hbm_intervals（权威 RLE）+hbm_plot_series（行预算绘图，旧 hbm_watermark_series 已退役）+hbm_watermark_instances、cache_events+kv_hit_states、
#    slo_warmup.json；P1/2026-08-30 起 hbm_watermark 按四层可信度分级：run_dir 含
#    results/kv_delta_journal.jsonl 时走 journal 权威重放，含 checksum 证书为
#    per_rank_total_hbm_certified 层——正式逐 rank 容量判决、违规 exit 3；缺
#    journal 的旧 run 为 upper_bound_only 上界层，超限只诊断不认证、exit 0；
#    checksum 在场而 journal 缺失=证书孤儿，fail-closed exit 2（B12/
#    2026-09-28 起 runner 搬运与归档常驻集均含 journal+checksum 两件、
#    copy-if-exists——exit-3 判决链在本仓可达，归档后幂等重跑不降级）），
#    一律不传分桶/聚合参数（粗化留给下游画图脚本）；
#    非 full 档依赖 request_metrics.csv 的子命令按设计跳过并写说明；
#    env SH_SLO_POSTPROCESS：1=默认 warn（任一步失败只写
#    slo_postprocess.FAIL 标记，不推翻仿真结果）、0=整步跳过、
#    strict=失败即 runner 非零退出。runner 同时把 per-request manifest
#    （metrics_manifest.json/manifest.json）拷入 run_dir 根（P2）——run_dir
#    自包含，与仓还原状态解耦
python3 sh_test_mesh/workload/llama2_7b_inference/online/verify/ledger_reconcile.py（--bridge-dir/--manifest，详见其 --help）
#    （对账前置，2026-10-01 补记：对账的 R2b 批级摘要对平以
#    graph_batch_digests.jsonl 为数据源，该文件是 env SH_GRAPH_DIGESTS 的
#    开关产物——默认 0 关、生产 run 不写（见 §3.1.1）；缺数据源时 R2b 按
#    设计 fail-closed 恒记 FAIL、对账整体非零退出（报告该项标
#    "FAIL: 数据源缺失"），须以 SH_GRAPH_DIGESTS=1 重跑后再对账）

# ⑥ 一键清空测试记录 + 编译产物（还原裸仓库 = 无物化输入 + 无 build/）
bash sh_test_mesh/run_scripts/clean_test_records.sh      # 清物化输入/运行产物/缓存（含 trace_config 的 request_queue_csv 指针行归一回占位——只改该行，其余未提交修改不触碰、仅 WARN 列出）
bash sh_test_mesh/run_scripts/clean_build_artifacts.sh   # 清编译产物（build/）
#    （或一步到位：clean_test_records.sh --full）
#    注：旧维护脚本 run_scripts/clean_history.sh 已删除（2026-09-26 批次；
#    仅清 generated/ 与 results/ 历史内容且全仓零引用），同等能力由
#    clean_test_records.sh 覆盖。
```

## 3. 两条仿真路线

| 路线 | runner | 时钟 | 产物 |
|---|---|---|---|
| ③ strategy 关感知 | run_online_strategy.sh | 真实物理 | 决策日志/metrics（digests 默认关，见 §3.1.1；成功后自动瘦身归档） |
| ④ strategy 开感知 | run_online_strategy_sensing.sh | 真实物理 | ③产物 + ledger.jsonl/感知日志（对账用） |

PASS 判据：completed == 物化请求数、no_decision=0、single_node=0、
delivery == graph_batch 数、③④ 决策日志逐字节一致（感知只开仪表不改判据）。
§2⑤ 的 ledger_reconcile 对账另有前置（2026-10-01 补记）：R2b 批级摘要
对平的数据源 `graph_batch_digests.jsonl` 随 env `SH_GRAPH_DIGESTS` 产生
（默认 `0` 关，生产 run 不写，见 §3.1.1），缺数据源时 R2b 按设计
fail-closed 恒 FAIL、对账非零退出——批级摘要对平须以 `SH_GRAPH_DIGESTS=1`
重跑后再对账。

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
  **consistency 门禁已接线硬执行（2026-09-26 修复批）**：`metrics_postprocess`
  对任一 type=consistency 记录 ok!=true / violations 非空 / dropped_events>0，
  或已完成 run 缺失 consistency 记录，一律非零退出 fail-closed（旧版 C++ 无
  consistency 记录的日志离线重跑会因此报错，属预期行为）。
- `link_bucket` / `link_total`（WP6，FluidScheduler 只读积分，不改流推进）：
  每桶 max/total bytes 与 mesh 边界链路峰值、每链路 total_bytes/active_ns。
  开关 env `ASTRA_LINK_OBSERVER`（metrics≠off 且 env 值非字符串 `"0"` 时启用
  （main_online 为严格字符串比较，`"00"`/`" 0"`/`"false"` 均视为启用；
  runner 只传 `"0"`/`"1"`，语义澄清 2026-10-01），设 `"0"` 关闭；off 档
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
  ——有事件 → `exact`；无事件时运行内为 `NA`，对已归档 run 离线重跑
  postprocess 且 `results/train_ledger.jsonl` 可得时填 `train_interpolated`
  proxy（§3.3；枚举 exact|train_interpolated|NA，对齐 wscllm 仓 B2wp9py
  同款实现；此前 FACE 冻结 NA，60s 全量 first_token 覆盖因此为 0）。
- `slo_tools/hopbytes.py`：face 变体收集器接入 B2wp9py 决策日志只读
  noc 观测字段（prefill=`history_noc_hops`、decode=`kv_noc_hops`，
  per-TP-shard 列表与 shards 一一对齐；hops 全等时聚合 bytes×hops[0]
  与逐 shard 求和严格相等）。60s 窗 coverage 0 → 1.0。
- `slo_tools/load_imbalance.py`：跳过 WP9 首步批台账行
  （`first_step: true` 的发射边界记录非终态 drain；真实 drain 在余量批
  行）——否则 SPLIT=ON 产物触发"每请求恰一次 drain"fail-closed。四仓
  逐字节同文由主控同步。
- `slo_tools/tests/run_golden_live.py`（新）：T2 golden G1-G4 运行版
  ——hand-craft 队列+sidecar 写在调用方 `--work-dir`（必填，惯用 /tmp，
  全量仓外；2026-09-26 py-slo-tests-4 起不再落仓内 `traces/`，文件名仍以
  `_request_queue_recompute.csv` 结尾以便 sidecar 兄弟 join）→ 真仿真（full+SPLIT=1）→ 手算断言（分解和恒等、
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
    13476831232 / 524288 B/token；权重字节由 test_face_scheduler.py 冻结，
    KV/token 为
2·layers·hidden_size·bytes_per_elem 推导值（hidden_size＝num_heads×head_dim；
2026-10-01 修订：旧文漏乘 layers 因子，按旧式重推导会小 32×；
test_first_token_proxy.py 以 2·32·4096·2 冻结），
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
  slo_tools 契约 41/41；metrics 契约 35/35；拆分语义 12/12（历史计数；现值见 §5
  基线——slo 契约 60（2026-09-30 R2 起新增 manifest 参数消费钉子）、metrics 契约 24）。

### 3.1 在线二进制机制旗标（--online-* 家族）

`AstraSim_Analytical_Congestion_Aware_Online` 显式解析 `--online-*` 家族
（OnlineCli.hh 契约；家族内未知旗标硬错）。机制类旗标一枚：

| 旗标 | 取值 | 缺省 | 语义 |
|---|---|---|---|
| `--online-validate` | `0\|1` | `1`（全量） | **C1（2026-08-28）；B.2 清除（2026-09-05）移除 N 抽检档**：GraphBatch 提交前全量校验开关（占 face 重载墙钟 ~42%）。`1` = 每批校验（改前行为，裸调用的 fail-closed 缺省）；`0` = 生产快速路径（跳过校验，commit() 的强制活性预检保留——该预检自 2026-09-26 修复批起额外强制 `batch_id==delivery_sequence` 恒等与 `batch_train_` 哨兵 watch `stage=prefill`，作为生产路径唯一防线，M2-2/M2-1）。取值域为严格 `0\|1` 枚举，`N≥2` 或其它任何值启动即硬错误（杜绝陈旧脚本静默改变校验频率）。只影响 validate 计数/诊断（graph_validate_ns 等白名单字段），不影响已提交状态。生产 runner 默认传 0（env `SH_ONLINE_VALIDATE` 可覆盖），冒烟/fixture/verify 脚本显式传 1。 |

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
  连续、块首 turn-0、块内 turn 严格 +1、turn>0 行 arrival 必空、
  turn-0 request_id 全局唯一——重复 request_id 会经同 epoch 同身份去重被静默
  丢弃第二个 ARRIVAL，2026-09-26 起在输入边界 fail-closed，M1-2）+ 溯源边车
  门（`<queue>.provenance.json` 必填：缺失或与索引复算字段任一不匹配，零
  Submit 前 `[Error]` 退出——2026-10-07 起缺席从"存在才比对"改为 fail-closed，
  对齐 wscllm 2026-09-28 P1 修复）+ turn-0 到达日历（按 `(arrival, queue_index)` 稳定排序
  提交）读取——turn-0 提交序与文件位置彻底解耦，非单调输入下
  `late_static_submit=0`，run-end 输出 arrival audit 行并以
  `late_static_submit==0 且逐行 ingress_delay==0（t=0 边界行除外）` 为正式
  门禁（fail 则非零退出）。
- **--request-window-rows 死旋钮已清除（2026-09-05，A.4）**：该 advisory
  旋钮（2026-08-30 P0 turn-0 fix 后 calendar reader 按 arrival 序提交，窗口
  值不约束读取与提交，不改变任何行为）已从本仓彻底移除——OnlineCli 解析/
  选项成员、main_online 调用实参、runner 的 `SH_REQUEST_WINDOW_ROWS` 透传
  一并删除；WindowedTraceReader 的 high_water 成员亦随 A.4 整体删除（当前
  reader 沿整个 turn-0 日历提交，无任何行数充填窗，仅受 ingress 容量
  4096 + 背压停驻光标约束）。**本仓不设 C++ 启动 span 预检/拒绝门**。
  plan_materializer 的
  manifest.json 仍持久化 `max_same_session_span`/`span_session_id`/
  `span_row_range`/`span_row_range_convention`，仅作 provenance 审计
  （campaign 复核与根因归档），无任何运行期强制拒绝点。
- **A2 input-open 死端分支不适用（四仓对齐 wscllm 裁决，合同 §2.1/P1）**：
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
  `sh_test_mesh/run_scripts/archive_run_outputs.sh <run_dir> <指标档>` 做产物瘦身归档
  （runner 把解析出的 `--metrics-detail` 档位作参数 2 透传；off 档 C++ 指标收集器
  整体禁用、cpp.log 零 `[METRIC]` 行属正常设计——归档跳过抽行与 gzip 两步并打
  `[archive] SKIP` 留痕，summary/full 档零 `[METRIC]` 行仍拒绝归档）：
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
  保留集不受影响）、`results/kv_delta_journal.jsonl` +
  `results/kv_delta_journal_checksum.json`（P1 权威 HBM delta 账本与 run 末
  守恒证书；本仓为消费侧、无写入方——写入方在 W 仓 KV_DELTA_JOURNAL
  开关与 joint 仓 online_service 侧车链，正常 face run 不存在此二件、
  缺文件不是错误；hbm_watermark certified 层 exit-3 判决与归档后幂等
  重跑的输入，B12/2026-09-28 起入常驻集）、
  campaign_provenance.json、
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
  对账联动（2026-10-01 补记）：ledger_reconcile 的 R2b 批级摘要对平以此
  文件为数据源，缺文件恒 FAIL（fail-closed，见 §2⑤/§3）——须对平批级
  摘要时以 `=1` 重跑后再对账。
- **env `SH_PROFILE_JSONL`**（默认 `0` 关，D2）：`profile.jsonl` 性能剖析
  产物默认不写；`=1` 恢复。decision_log/train_ledger 恒写不动（SLO 指标源）。
- **env `ASTRA_LINK_OBSERVER`**（runner 默认 `0` 关，D3）：在线模式
  link_bucket/link_total 行无消费者（metrics_postprocess 不读、hopbytes 走
  decision log），runner 默认关闭 FluidScheduler link 观测；`=1` 恢复发射。
  同项 D3：在线模式 `hbm_watermark` 逐桶行（全零序列，权威源是
  slo_tools/hbm_watermark.py 的 ledger 重放）不再发射；per-rank
  `hbm_watermark_summary` 保留（含 flat-zero 注记）。
- **env `SH_PROPAGATING_TAIL_LIMIT`**（未设 = 缺省 `8`，2026-09-29 补记）：
  Python 决策侧在途尾部观测器（PropagatingTailTracker）对 delivery 类在途
  工作（`delivery_start` 未 ack 交付、`kv_provisional` 未确认 KV 动作两类
  共用）的 fail-closed 上限。默认 8 = 8× 桥背压合法窗口（当前协议合法未
  确认交付数 = 1，推导见 `online_scheduler_base.py` 模块头）；超限即
  raise（桥层转 error response、非 0 退出），在途条目原样保留、绝不截断；
  `arrivals` 类上限恒为 manifest 请求数，不受本变量影响。值必须为正整数，
  非法值在调度器构造期即报错。离线重放工具的在途深度可超出在线背压窗口
  ——离线重放工具已内置调大逻辑；如需手工覆盖可导出该变量
  （`export SH_PROPAGATING_TAIL_LIMIT=<正整数>`）。
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
  覆盖一百万交付（本地冒烟可降低该环境变量；同脚本的
  `SCHEDULER_MEMORY_CHECK_INTERVAL`（缺省 10000）
控制结构性断言（`_assert_lifecycle`）的抽检间隔，2026-09-30 起登记入册；
RSS 仅运行结束一次性诊断输出（不周期采样）、无阈值门——2026-10-01 措辞修正，旧文误称"RSS 断言"）
  暂不能 cursorize：future alarm 和 GraphBatchBuilder 仍按 `queue_index`
  随机解析未来请求。

## 4. 机制回归 fixtures

`run_online_idle_fixture.sh`（IDLE 五态生命周期）、`run_online_wakeup_guard_fixture.sh`、
`run_online_same_tick_milestone.sh`、`run_bridge_cpp_death_fixture.sh`（2026-09-26 自
wscllm-LRU 母本移植：纯 Python 复刻"假 C++ 死亡"，断言 fixture 退出码==1 且 stderr
含 `C++ side is gone`；缺省 run root `/tmp/face_bridge_cpp_death`）——机制层健康自检。
`bridge_race_stress_repro.sh` **不是自检**：它是旧协议竞态复现演示工具（在 /tmp 复刻
2026-08-16 双向长连接修复前的逐交换握手时序，预算内观察到旧竞态即 PASS/exit 0），
非门禁、不构建不运行本仓代码。
另有 C++ 聚焦单测（target 注册于 `astra-sim/network_frontend/analytical/CMakeLists.txt`，
随 §2 ① 构建树编译；可执行文件落位 2026-10-01 构建树实测复核（取代
2026-09-28 `find` 快照）为三处：该 CMakeLists 的 24 个 add_executable target
（23 个单测 + 1 个在线主程序）中 17 个设 RUNTIME_OUTPUT_DIRECTORY 落在
`build/astra_analytical/build_congestion_aware/bin/`（含在线主程序与
CompletionFixture），其余 7 个单测 target 未设：
StatisticsCompaction /
SystemHistoryLifecycle / MetricAnchorSpool / MetricOneShotErase / LocalMemTracker /
CommunicatorGroupOrder / ServiceVacuum 七个 Test 落在同树 `frontend_build/`（与下文复核注"前端二进制目录
为 `frontend_build/`"一致），根级 journal 落在 `astra_sim_build/`，均无参数直跑）：`..._AlarmCancellationTest`（可取消 alarm 链路：bucket 清空时 outer alarm
从 backend 物理移除、共享 bucket 级联、重复取消幂等、legacy 后端回退 stale guard）、
`..._MetricOneShotEraseTest`（MetricCollector one-shot node bucket 擦除 + OnlineNode
anchor 快路径标志，双运行 [METRIC] 输出逐字节对拍；OnlineNode 布局约束现仅由 pre-R2
mirror 镜像 struct 的 static_assert 承载——2026-09-26 起工具链敏感的
`sizeof(OnlineNode)==384` 数值锚已删除，384 不再是契约）。
**CTest 注册（2026-09-26 修复批；2026-09-30 复核更新：NodeStoreTest 已于 2026-09-29（R4-N1）加入门禁，analytical 目标数 20→21）**：analytical CMakeLists 调用 `enable_testing()`
并 `add_test` 注册 21 个 analytical CTest 目标——`AstraSim_Analytical_ChunkIdGeneratorTest` 与
`AstraSim_Analytical_Congestion_Aware_{WatchRegistryTest, DecisionMailboxTest,
NodeStoreTest, BridgeLoopbackTest, GraphBatchCommitterTest, ParsedBatchTest,
WindowedReaderTest,
CalendarOracleTest, ServiceVacuumTest, StatisticsCompactionTest,
SystemHistoryLifecycleTest, MetricAnchorSpoolTest, MetricOneShotEraseTest,
LocalMemTrackerTest, TerminalTailDrainTest, CommunicatorGroupOrderTest,
AlarmCancellationTest, CliOnlineTest, EventQueueDeferredTest, IngressIdleTest}`；
按 §2 ① 拼装构建树（拼装 CMakeLists 顶层 `enable_testing()`），
`ctest --test-dir build/astra_analytical/build_congestion_aware` 在构建树根目录
即可见并运行全部 36 项（2026-10-03 die 内仿真器批后复核：上述 21 项 analytical +
根级 `AstraSim_OfflineGreedyScheduleJournalTest` + die 成本库 `extern/die_model`
11 项（tx8_reference/hardware/spec_parse/configs_consistency/workload/mapping/
costmodel/tx8_compile/execution/selection_fuzz/capi）+ 3 项 die 前端接入测试
（DieModelAdapterTest / DieExecutionLifecycleTest / LocalHbmDieMemoryTest）；
`selection_fuzz` 为阶段 E3 常驻回归——候选剪枝下界换为健全版（槽位占用和 +
lag 折算关键路径）后，`pick_best_candidate` 带剪枝与无剪枝全枚举的胜者必须在
随机 spec × 随机合法硬件与真实形状上逐例一致（CI 默认 500 例随机 + 全量轻量
真实形状，种子固定；`DIE_SELECTION_FUZZ_N/_SEED/_FULL` 可调，FULL 加入
M512/M4096 重型形状）；该构建树下不存在
`astra-sim/network_frontend/analytical` 子目录路径，analytical 前端二进制目录
为 `frontend_build/`）；BridgeLoopbackTest 的 ctest 实参依赖
python3 在 PATH 与仓内
`sh_test_mesh/workload/llama2_7b_inference/online/verify/bridge_echo.py`。新增覆盖：
decision_mailbox_test Part E（同 tick 完成-到达 epoch 顺序）、windowed_trace_reader_test
Part R（future-alarm 钳位 + 消费腿）。根级 `AstraSim_OfflineGreedyScheduleJournalTest`
同批起**默认构建并注册 CTest**（根 CMakeLists `ASTRA_SIM_BUILD_OFFLINE_GREEDY_SCHEDULE_TEST`
缺省翻 ON——原缺省 OFF 且无任何脚本打开，从不运行）。
2026-09-24
工作树补录三项接入构建的既存单测（analytical `CMakeLists.txt` 中"never
part of the baseline gates"注释块记录接入配方，target 为
`AstraSim_Analytical_Congestion_Aware_CliOnlineTest`/`..._EventQueueDeferredTest`/
`..._IngressIdleTest` 三件——行号从略，免随编辑漂移）：`..._CliOnlineTest`
（在线 CLI 契约测试，R1-R14 含 FP1 加固无符号词法与 watchdog 边界）、
`..._EventQueueDeferredTest`（EventQueue tick 末收口 + 同 tick deferred 通道，含
FluidScheduler deferred-flush 集成）、`..._IngressIdleTest`（IDLE 生命周期 fixture：
IDLE/ACTIVE/DRAINING/FINISHED 迁移、EOF/Error 终态、溢出审计、close-vs-submit 竞态）。
**Release 下恒真的修法（2026-09-25 深挖二轮）**：上三测试的裸 `assert` 全量改为
运行时 expect 制（`expect(` 调用点计数，2026-09-28 实测：cli 103 处 /
event_queue 34 处 / ingress_idle 68 处（cli 2026-10-01 复核为 104：R15 加
test_family_prefix_table 后 +1，2026-09-28 快照 103 未随之更新）；
2026-09-25 改写当时为 105/37/70），缺省
Release（-DNDEBUG）构建不再把断言编译掉；配套地 EventQueue 主队列 strict-increase
不变式由 debug 断言升级为 Release 常开的 fail-closed（违规 fprintf+abort，
`event_queue_deferred_test` case E 即验证该 abort）。两个 build-only 例外不进
ctest、不在"无参数直跑"内（需 `sh_test_mesh/generated/` 物化产物，gitignored
by design，手工配方在各头注释；NodeStoreTest 已于 2026-09-29（R4-N1）
加入 ctest 无参门禁，不再属例外）：`..._CompletionFixture` 需先生成
`sh_test_mesh/generated/completion_fixture/`
（`python3 astra-sim/workload/execution_driven/tests/make_completion_fixture_et.py`，
依赖 §2 ③ 的 runtime_config 已物化）并按其头注传
`--workload-configuration`（指向物化产物 `completion_fixture/fixture`）与
runtime_config 三件（comm-group/system/network）各一参——注意该 fixture
不解析 `--fixture-et`（那是 `..._NodeStoreTest` part C 专用旗标）；
`..._LocalHbmTest` 需先跑 `make_local_hbm_fixture_et.py` 生成
`sh_test_mesh/generated/local_hbm_fixture/`（fast/slow 两套 network.yml）。
Python 侧 `online/test_propagating_tail.py`
（`online_scheduler_base.py` 的在途尾部观测器 PropagatingTailTracker：对到达未完成
请求、未 ack 交付、未确认 provisional KV 动作三类在途工作记 current/peak/按来源计数，
超限 fail-closed 报错、绝不截断；delivery 类默认上限 8，env
`SH_PROPAGATING_TAIL_LIMIT` 可覆盖（见 §3.1.1）；8 用例，pytest 或直跑）。
注意（2026-09-29 补记）：该 8 用例为 pytest 裸函数风格，stdlib unittest
discover 不可见——环境缺 pytest 时先 `pip install pytest`（Docker 镜像内
同理，镜像缺省未预装 pytest）。

`online/verify/` 手动验证工具（python3 直调，非门禁；均为本仓实体）：
`bridge_streaming_fixture.py`（无参：有界 tracker + journal 字节级回归）、
`idempotency_fixture.py --bridge-dir --plan-dir [--config] [--limit N]`（0=全量重放）、
`diff_explainability.py --baseline-run --sensing-run [--report]`（退出码契约：
0=差异全归类 / 1=行数不一致或存在无法归类差异(DEFECT)——2026-09-26 起该退出码
真实可达，报告新增"行数不一致(缺陷)"统计行 / 2=数据源缺失 fail-closed）、
`profile_scan_audit.py --profile [--total-requests N]`（total=0 已零保护；
2026-09-26 起 profile 行不再携带 full_scan_entries，该审计维度已随生产端
死记账删除，旧产物多出的键被忽略）、
`train_a1_eviction_fixture.py`（需先构建在线二进制；既有 generated plan 目录自动
移 /tmp 备份并在结束后迁回，不破坏仓内状态）。

## 5. 目录导览（关键路径）

- `astra-sim/workload/execution_driven/`：在线机制层（C++）
- `sh_test_mesh/workload/llama2_7b_inference/online/`：在线调度器/构图器/服务层（Python）
- `.../online/verify/`：对账与验证工具
- `sh_test_mesh/run_scripts/`：全部 runner 脚本
- `utils/`：上游开发工具链脚本（**手动执行**，仓内无任何自动调用入口，与在线仿真
  路线无关，保留不删）：`build_docker_image.sh`（构建 `astrasim/astra-sim` Docker
  镜像，配套 `start_docker_container.sh` 启动容器）、`install_chakra.sh`（安装
  `extern/graph_frontend/chakra` 前端依赖）、`run_clang_format.sh`（对 `astra-sim/`
  下 `.cc`/`.hh` 跑 clang-format）
- `sh_test_mesh/workload/llama2_7b_inference/traces/`：物化器脚本（裸仓只
  含脚本不含数据件；历史正文中提到的 60s sidecar/10s 子窗队列等数据件
  均为当批运行时物化的产物，数据件由调用方物化，provenance 以物化器
  stdout 为准）
- `sh_test_mesh/slo_tools/`：SLO 离线后处理工具集（slo_stats / load_imbalance / restore_decomposition / kv_cache_adapter / hopbytes + `slo_postprocess_driver.py`（A4 单遍合并驱动，run_slo_postprocess.sh 链内使用；工具 CLI 不变）+ `slo_params_manifest.json`（B 类参数唯一来源，B4 已填推导值）+ tests；纯离线只读，详见目录内 README.md）
- `sh_test_mesh/tests/` + workload 根 + online/：pytest（基线计数 2026-10-03
  die 内仿真器批后实测，口径 = pytest `--collect-only` 静态收集）：
  `sh_test_mesh/tests/` 60 = config_resolver 18（die 硬件解析/派生回流/
  resolved_die_model.json 物化/system 三键注入/容量对齐/发现序 fail-closed 等
  DieHardwareTests 11 例 + legacy 路径回归）+ first_token_proxy 9 +
  metrics_contract 33（阶段 E2 +9：DieUtilizationPipelineContractTests——die
  利用率三元组的管线级断言：合法三元组/无 comp 节点 note 合法 null/未知 note
  null 拒/越界拒/非数值拒/die run 无 rank_compute 记录拒/legacy 不断言/服务
  manifest 声明生效/无声明时记录自推断）；workload 根 44 = test_face_scheduler 30（含 die 估值
  DieModelEstimateTests 8：数值钉子/spec 契约/source 标签/平局轮转/接线）+
  test_die_model_oracle 12（发现序/缺库 fail-closed/批量估值/hardware_summary/
  close 生命周期/config 解析序）+ test_face_kv_incremental_invariants 2；
  online/ 95 = die_model_adapter 20（span 恒等式/13 类映射/融合矩阵/spec
  schema/manifest 门）+ graph_batch_builder 16（含 die 发射 4：spec 附着+
  融合+守恒对拍/绝对位置+TP 切片/collective+marker/调度器注入门）+
  first_token_split 12 + train_machinery 9 + admission_liveness 8 +
  propagating_tail 8 + weight_passes 5（含 die 守恒 1：与 legacy 同 span 的
  COMP 总量/AR bytes 逐字节相等、17→12 体节点）+ graph_batch_rank_ledger 3 +
  verify/test_diff_explainability.py 6 + verify/test_ledger_reconcile.py 8。
  四根（含 slo_tools/tests/ 4 文件 131 例）合计 330（2026-10-03 阶段 E2 实跑：
  **329 passed + 1 skipped**，skip = slo_tools test_hbm_watermark 既有仓外
  phase2 真跑夹具条件跳过）。**die 用例的库可用性门**：带库（.so 经 §H 发现序
  可达——仓内 build 路径或 env `DIE_MODEL_CABI_PATH`）时全量运行；裸仓无 .so
  时 die 组 skip（legacy 回归套件保持全绿），与生产路径 fail-closed 语义的区分
  见 `test_die_model_oracle.py` 模块 docstring。
  另 `sh_test_mesh/slo_tools/tests/` 4 文件 131 例（slo_contract 61（2026-10-01
  R3 轮 +1：warmup 到达跨度≤剔除窗口的 NA 降级锚）+ golden_g1g4 16 +
  driver_parity 13 + hbm_watermark 41；2026-10-03 pytest collect 复核；
  unittest discover 实跑全过，含 1 例仓外夹具条件 skip）。

## 6. 边界与纪律

- 仿真输入唯一允许源 = astra_compute_20.csv，冒烟窗口 = 该 trace 前 1~10 秒
  （优先 1 秒；依据工作区根的 Agents.md（`../../Agents.md`，仓内无此文件）
  《冒烟测试与仿真效率规范》）。
- 缺失输入一律 fail-closed（generate 桩/materializer/runner/GEN_MATCH 均实测 exit=1）。
- 策略文件（face_scheduler.py / session_kv_manager.py）为保留对象，勿改。
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
- 折入 recompute 输入口径 + 运行时 KV 账本：请求队列为唯一仿真输入，
  turn-0 源前缀折入该行 prefill_length（整段重算口径，
  derive_20_first_30_seconds.py 的 recompute 折入口径）；manifest 仅携带队列派生的
  history_tokens_before 推导值（plan_materializer.py 的 _derive_manifest_requests）；会话 KV 的
  规模与位置由运行时 `SessionKVCacheManager` 账本动态维护
  （face_online_scheduler.py 的调度器构造；session_kv_manager.py 的 SessionKVCacheManager）：驻留命中
  直接复用、跨实例历史先经 NoC 迁移段、被逐出则窗口内重算。
- long double 时间精度适配：大 ns 级首达偏移超出 IEEE-754 double 精确整数
  范围（2^53）时，分析网络适配层返回 ASTRA-sim 时间以 `long double` 保持
  64 位事件时间精确，避免完成回调与事件映射键错位 1 ns
  （CommonNetworkApi.cc 的 sim_get_time 转换；平台前提 2026-10-01 补记：该承诺依赖
 long double 尾数 ≥64 位——Linux x86-64 的 80 位扩展与 AArch64 的 quad 满足，
 MSVC 的 long double==double 不满足，本仓构建链仅覆盖 Linux/GCC）。
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
