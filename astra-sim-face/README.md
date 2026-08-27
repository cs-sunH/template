# astra-sim-face - 晶圆级芯片（WSC）LLM 推理架构与机制

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
  端口之间完全并行，远端池总容量不设限
  （`extern/remote_memory_backend/analytical/AnalyticalRemoteMemory.cc`）。
- **本仓配置**：`NO_MEMORY_EXPANSION`（远端池未启用）：远端内存的建模能力保留在代码中
  （`PER_NPU_MEMORY_EXPANSION` 类型与 mesh 边界端口派生逻辑俱在），但本仓场景下 C++
  一旦遇到远端访问节点即 fail-closed 退出。因此本仓 KV 只有片上一层——全部驻留本地
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

### F. 本仓在五仓中的定位

| 仓库 | 实例组织 | 请求→实例映射策略 | KV 驻留与恢复 | 远端内存池 |
|---|---|---|---|---|
| **astra-sim-face（本仓）** | 统一实例（P+D 同实例） | FACE 原始映射：prefill 选剩余 chunk 最少；decode 在邻接图加权距离限制（阈值 = D2D 带宽 / 本地 HBM 带宽）内按 per-die Roofline 增量代价 | RESIDENT/EVICTED 两态；LRU 逐出＝零代价删除；恢复＝重算 | 未启用（NO_MEMORY_EXPANSION） |
| astra-sim-wscllm | PD 分离（Prefill-only + Decode-only 分区） | prefill 选排队请求最少；decode 用静态一跳 P→D 映射 | RESIDENT/EVICTED 两态；LRU 逐出；恢复＝重算（跨实例历史走 NoC 迁移） | 未启用（NO_MEMORY_EXPANSION） |
| astra-sim-sh_1.0 | 统一实例 | prefill HBM 可行过滤 + 剩余 chunk 最少；decode 按 per-die Roofline 增量代价 | LOCAL_HBM/REMOTE_MEMORY 两态；整 session 粒度逐出；恢复＝远端全量取回 | 启用（全部边缘芯粒挂端口） |
| astra-sim-sh_2.0 | 统一实例 | prefill Roofline 剩余负载均衡（历史 KV 全/部分驻留与全逐出统一）；decode 按 per-die Roofline 增量代价 + HBM 剩余 tie-break | 三态（含半驻留 PARTIAL）；两阶段类型感知逐出（human 类先于 tool 类）；流水化部分恢复 + HBM 恢复/推理带宽共享 | 启用（全部边缘芯粒挂端口） |
| astra-sim-sh_3.0 | 统一实例 | 三段式 prefill（首请求避边缘 / HBM 命中 sticky / 远端命中负载均衡）；decode 本地化固定同实例 | 三态；两阶段逐出；流水化部分恢复（机制同 sh_2.0） | 启用（全部边缘芯粒挂端口） |

五仓还共享同一套 Execution-Driven 在线仿真机制层（`astra-sim/workload/execution_driven/`：
RequestIngress / DecisionMailbox / DecisionBridge / GraphBatchCommitter 等）：策略决策由
Python 在线服务层实时给出，计时由 C++ 物理时钟推进；各仓仅保留路径③（strategy 关感知）
与路径④（strategy 开感知）两条在线路线。五仓在线决策均不再使用任何离线 LUT
（残留 LUT 查表已随静态链路一并删除）：face / sh_1.0 / sh_2.0 的 decode 候选代价由
在线 Roofline 模型即时计算（`per_die_delta_ns`），决策确定可复算复放。

### G. 本地 HBM 带宽模型（多用户竞争，`hbm-bandwidth-contention`）

真实硬件中每颗芯粒的本地 HBM 是共享资源：同一时刻的所有访存流量竞争同一份带宽。
本仓用每 rank 一个 `LocalHbmBandwidthModel`（`astra-sim/workload/LocalHbmBandwidthModel.{hh,cc}`，
由 sh_2.0 的两用户版泛化为 N 用户）建模，system.json 键 `hbm-bandwidth-contention`
（bool，代码默认 true；置 0 完全回退旧行为——COMP 走 Roofline 闭式公式、comm 不计 HBM；
`local-mem-bw <= 0` 时自动视为 false，防零速率卡死）。参与竞争的流量（本仓无片外内存池，
只有这两类）：

1. **COMP 节点（Roofline 计算访存）**：bytes = tensor_size（读写在同一标量上合并计费），
   FLOPs 按 peak-perf 并行排空，两者都排空才算完成（单用户时保持 Roofline
   max(计算, 访存) 语义）。若 ET 带**非 0 校准 runtime**（在线 LUT 时钟路径）则不进流体
   模型、保持校准时长——本仓 trace 的 COMP 无 runtime，实际总走流体路径；该分支为在线
   LUT 兼容保留。
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
不变。数值断言级验证见
`astra-sim/workload/execution_driven/tests/local_hbm_bandwidth_model_test.cc`
（1/3 → 1/2 → 全速的均分序列、join 两序、单次触发）。

## 1. 本仓是什么

- **策略语义（保留对象，未改动）**：FACE 原始映射：加权实例图 + per-die Roofline 增量代价选择 decode 实例（平局轮流裁决）；RESIDENT/EVICTED 两态 + LRU recompute；无远端内存；prefix 走 recompute；session 驻留状态/字节数由运行时 KV 账本（SessionKVCacheManager）动态维护，无 sidecar
- **执行驱动机制层**（`astra-sim/workload/execution_driven/`）：在线事件驱动
  （RequestIngress/DecisionMailbox/WatchRegistry/GraphBatchCommitter/长连接
  DecisionBridge 等），五仓接口一致。
- **验证证据**：Tier B 等价、感知开/关决策逐字节一致、分层账本对账、
  机制 fixtures。

## 2. 快速开始

```bash
cd <本仓根>
# ① 构建（首次需先配置；构建树已预置时可跳过 configure）
cmake -S build/astra_analytical -B build/astra_analytical/build_congestion_aware -DBUILDTARGET=congestion_aware
cmake --build build/astra_analytical/build_congestion_aware -j

# ② 物化输入（唯一允许源 = agent-traces/tracelab/astra_compute_20.csv 前 30 秒，
#    arrival_time < 30e9 ns；物化器：derive_20_first_30_seconds.py）
#    产物放 sh_test_mesh/workload/llama2_7b_inference/traces/，
#    并把 trace_config.csv 第 12 行 request_queue_csv 指向它
#    物化器 CLI：[source] [queue] [sidecar] [window_ns] [arrival_scale]；
#    arrival_scale>0 仅缩放 turn-0 session_arrival_time_ns（t0/scale，即
#    负载 ×scale），inter_request_interval_ns（human/tool 外生等待）不动，
#    窗口判定与统计始终用未缩放源时间——scale=1 时 8 列队列与冻结基线
#    逐字节一致；缩放与 request_type 等新信息只进 canonical sidecar 与
#    stdout provenance，不进队列。

# ③ 生成 plan 目录（runtime_config 四小件 + manifest + metrics_manifest）
cd sh_test_mesh/workload/llama2_7b_inference && python3 plan_materializer.py && cd <仓根>

# ④ 跑③④（GEN_MATCH：generated/ 下须恰一个 llama2_7b_inference_54npus_* 目录）
bash sh_test_mesh/run_scripts/run_online_strategy.sh <run_dir> <绝对路径 request_csv>
bash sh_test_mesh/run_scripts/run_online_strategy_sensing.sh <run_dir> <绝对路径 request_csv>
bash sh_test_mesh/run_scripts/run_online_strategy_legacy.sh <run_dir> <request_csv> <legacy_gen>

# ⑤ 指标后处理 + ④对账
bash sh_test_mesh/run_scripts/run_metrics_postprocess.sh <run_dir>/cpp.log
#    （在线 runner 已自动后处理：raw/normalized_metrics.csv；full 档额外产
#    request_metrics.csv——逐请求冻结 29 列，与 manifest 按 queue_index/
#    request_id fail-closed 连接；summary/off 档不产该文件，原因写
#    postprocess.log）
python3 sh_test_mesh/workload/llama2_7b_inference/online/verify/ledger_reconcile.py（--bridge-dir/--manifest，详见其 --help）

# ⑥ 一键清空测试记录 + 编译产物（还原裸仓库 = 无物化输入 + 无 build/）
bash sh_test_mesh/run_scripts/clean_test_records.sh      # 清物化输入/运行产物/缓存（含 trace_config 指针回占位）
bash sh_test_mesh/run_scripts/clean_build_artifacts.sh   # 清编译产物（build/）
#    （或一步到位：clean_test_records.sh --full）
```

## 3. 两条仿真路线

| 路线 | runner | 时钟 | 产物 |
|---|---|---|---|
| ③ strategy 关感知 | run_online_strategy.sh | 真实物理 | 决策日志/digests/metrics |
| ④ strategy 开感知 | run_online_strategy_sensing.sh | 真实物理 | ③产物 + ledger.jsonl/感知日志（对账用） |

PASS 判据：completed == 物化请求数、no_decision=0、single_node=0、
delivery == graph_batch 数、③④ 决策日志逐字节一致（感知只开仪表不改判据）。

指标档位（B1/WP0 起）：三变体 runner 的 `--metrics-detail` 不再硬编码
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
  行）——否则 SPLIT=ON 产物触发"每请求恰一次 drain"fail-closed。五仓
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
  terminal_status/first_token_source 枚举、NA 语义用例；五仓逐字节
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
（OnlineCli.hh 契约；家族内未知旗标硬错）。机制类旗标当前一枚：

| 旗标 | 取值 | 缺省 | 语义 |
|---|---|---|---|
| `--online-node-gc` | `0\|1` | `0`（关） | M2 节点 GC（2026-08-23）：开启时 GraphBatchCommitter 在每次批提交的静止点（issue pass 完全返回后）回收各 rank NodeStore 中"已 finish 且无未完 children"的节点，并按同水位修剪 (rank, json id) → store id 映射，C++ 侧内存维持在途窗口而非全程累计图（30s 冻结输入实测 cpp 峰值 −61%）。被回收节点必已 finished，仍指向它的跨批边按 NodeStore 死父规则无阻塞（validate 仅按 per-rank 稠密前缀水位线放行已修剪 id，从未存在的 id 照旧 fail-closed）。`0` = M2 前永不删除行为。决策序列与全部工件不受该旗标影响（GC 开/关两臂均字节对拍验收）。**默认关的裁决依据（2026-08-23 翻转）**：GC 开存在可复现的轻载墙钟回归（30s 档 +35~55%，机制未明，隔离基准反快 27%，证据见 /tmp/accel_c/DONE open_finding）；按"不为省内存大幅换 CPU"红线默认关，重载/多实验并行等内存受限场景显式 `--online-node-gc 1`（此时 OOM 风险大于墙钟代价）；重档 on/off 对比数据补齐后可一行翻转默认。 |

runner 脚本不传该旗标（走缺省 0 = 墙钟中性）；重载/多实验并行等内存受
限场景在 runner 命令行追加 `--online-node-gc 1`（cpp 峰值 −61%，决策工件
字节不变）。运行期证据：cpp.log 启动行 `[online] node gc: ...`、
结束行 `[online] node gc: erased=... retained=...`。

## 4. 机制回归 fixtures

`run_online_idle_fixture.sh`（IDLE 五态生命周期）、`run_online_wakeup_guard_fixture.sh`、
`run_online_same_tick_milestone.sh`、`bridge_race_stress_repro.sh`——机制层健康自检。

## 5. 目录导览（关键路径）

- `astra-sim/workload/execution_driven/`：在线机制层（C++）
- `sh_test_mesh/workload/llama2_7b_inference/online/`：在线调度器/构图器/服务层（Python）
- `.../online/verify/`：对账与验证工具
- `sh_test_mesh/run_scripts/`：全部 runner 脚本
- `sh_test_mesh/workload/llama2_7b_inference/traces/`：物化器脚本（数据件由调用方物化，provenance 以物化器 stdout 为准）
- `sh_test_mesh/slo_tools/`：SLO 离线后处理工具集（slo_stats / load_imbalance / restore_decomposition / kv_cache_adapter / hopbytes + `slo_params_manifest.json`（B 类参数唯一来源，B4 已填推导值）+ tests；纯离线只读，详见目录内 README.md）
- `sh_test_mesh/tests/` + workload 根 + online/：pytest（基线：17+33+20 = 70 passed，
  无预存失败；2026-08-22 拼 batch 改造新增 online/test_weight_passes.py(4) +
  online/test_train_machinery.py(9) + online/test_graph_batch_builder.py(7)）

## 6. 边界与纪律

- 仿真输入唯一允许源 = astra_compute_20.csv 前 30 秒（更早的用户指示曾临时
  授权过更大窗口；以当下指示为准）。
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
- 接栅栏与段末屏障：列车发射不做块末恢复/段内清链，per-rank `previous_id`
  无条件接续当前 frontier（per-rank 发行序 = 全局发射序，2026-08-19 五仓
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
  `traces/derive_20_first_30_seconds.py:14-16`）；manifest 仅携带队列派生的
  history_tokens_before 推导值（`plan_materializer.py:61-88`）；会话 KV 的
  规模与位置由运行时 `SessionKVCacheManager` 账本动态维护
  （`face_online_scheduler.py:273`；`session_kv_manager.py:363`）：驻留命中
  直接复用、跨实例历史先经 NoC 迁移段、被逐出则窗口内重算。
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
