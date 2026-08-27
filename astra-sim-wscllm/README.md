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
- **本仓配置**：`NO_MEMORY_EXPANSION`（远端池未启用，WSC-LLM 策略场景强制如此）：
  建模能力完整保留（`generate_trace.py` 的 `RemoteMemoryConfig` / `edge_npus` 解析、
  `AnalyticalRemoteMemory` 后端与 mesh 边界端口派生均在，切换到
  `PER_NPU_MEMORY_EXPANSION` 即可启用）。本仓 KV 全部驻留片上本地 HBM：LRU 逐出＝
  零代价删除、恢复靠重算，跨实例历史 KV 仅走 NoC 迁移。

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

### F. 本仓在五仓中的定位

| 仓库 | 实例组织 | 请求→实例映射策略 | KV 驻留与恢复 | 远端内存池 |
|---|---|---|---|---|
| astra-sim-face | 统一实例（P+D 同实例） | FACE 原始映射：prefill 选剩余 chunk 最少；decode 在邻接图加权距离限制（阈值 = D2D 带宽 / 本地 HBM 带宽）内按 per-die Roofline 增量代价 | RESIDENT/EVICTED 两态；LRU 逐出＝零代价删除；恢复＝重算 | 未启用（NO_MEMORY_EXPANSION） |
| **astra-sim-wscllm（本仓）** | PD 分离（Prefill-only + Decode-only 分区） | prefill 选排队请求最少；decode 用静态一跳 P→D 映射 | RESIDENT/EVICTED 两态；LRU 逐出；恢复＝重算（跨实例历史走 NoC 迁移） | 未启用（NO_MEMORY_EXPANSION） |
| astra-sim-sh_1.0 | 统一实例 | prefill HBM 可行过滤 + 剩余 chunk 最少；decode 按 per-die Roofline 增量代价 | LOCAL_HBM/REMOTE_MEMORY 两态；整 session 粒度逐出；恢复＝远端全量取回 | 启用（全部边缘芯粒挂端口） |
| astra-sim-sh_2.0 | 统一实例 | prefill Roofline 剩余负载均衡（历史 KV 全/部分驻留与全逐出统一）；decode 按 per-die Roofline 增量代价 + HBM 剩余 tie-break | 三态（含半驻留 PARTIAL）；两阶段类型感知逐出（human 类先于 tool 类）；流水化部分恢复 + HBM 恢复/推理带宽共享 | 启用（全部边缘芯粒挂端口） |
| astra-sim-sh_3.0 | 统一实例 | 三段式 prefill（首请求避边缘 / HBM 命中 sticky / 远端命中负载均衡）；decode 本地化固定同实例 | 三态；两阶段逐出；流水化部分恢复（机制同 sh_2.0） | 启用（全部边缘芯粒挂端口） |

五仓还共享同一套 Execution-Driven 在线仿真机制层（`astra-sim/workload/execution_driven/`：
RequestIngress / DecisionMailbox / DecisionBridge / GraphBatchCommitter 等）：策略决策由
Python 在线服务层实时给出，计时由 C++ 物理时钟推进；各仓仅保留路径③（strategy 关感知）
与路径④（strategy 开感知）两条在线路线。五仓在线决策均不再使用任何离线 LUT
（残留 LUT 查表已随静态链路一并删除）：face / sh_1.0 / sh_2.0 的 decode 候选代价由
在线 Roofline 模型即时计算（`per_die_delta_ns`），决策确定可复算复放。

### G. 芯粒本地 HBM 带宽模型（`hbm-bandwidth-contention`）

真实硬件中每颗芯粒的本地 HBM 是共享资源。C++ 执行层为每个 rank 建一个
`LocalHbmBandwidthModel` 流体模型（`astra-sim/workload/LocalHbmBandwidthModel.{hh,cc}`，
由 system.json 键 `hbm-bandwidth-contention` 开关，默认开），以下两类流量竞争同一份
`local-mem-bw`（本仓无片外内存池，只有这两类）：

1. **COMP 节点（roofline 计算访存）**：bytes = `tensor_size`（读写合并计费），
   ops 与访存并行排空，单用户时保持 roofline `max(计算, 访存)` 语义。带非 0 校准
   runtime 的 COMP 节点不进流体模型、保持校准时长（本仓 trace 的 COMP 无 runtime）。
2. **NoC p2p 通信的数据端点**：发送方 rank 的 HBM 读 + 接收方 rank 的 HBM 写
   （bytes = comm 字节数；history_kv、prefill_to_decode_kv 等迁移两端都计）。
   **多跳途经的中间芯粒不占 HBM**（路由器直通——网络后端只在端点计费）。
   bytes==0 不建作业，1B ACK 照常计；ET comm 节点属性 `hbm-charge`（默认 true）
   可显式豁免某端点。

仲裁口径：N 个并发用户严格均分（各得 `full_rate/N`），任一作业完成立即释放并给
剩余用户重分配（事件驱动）；带宽直接用配置标量——读写共享同一总线与总带宽，不区分
读写方向，不区分峰值/持续带宽；`local-mem-latency` 在每作业启动时计一次。p2p comm
节点完成 = join(网络侧完成, 本端 HBM 作业完成)，两侧都到齐才触发节点完成（幂等）。
**TP 集合通信（PacketBundle 3x 路径）不在范围**。

`hbm-bandwidth-contention: 0`（或 `local-mem-bw <= 0` 自动回退）＝完全恢复旧行为：
roofline 闭式公式、comm 立即随网络完成、不计 HBM。指标导出（MetricCollector，
旧键不变）：每 rank `hbm_busy_ns`、分类 served bytes（comp / comm_read /
comm_write）、`hbm_peak_concurrent_jobs`、`hbm_redistribution_events`、真实利用率
`hbm_bw_util = busy_ns / 墙钟窗口`。单测：
`astra-sim/workload/execution_driven/tests/local_hbm_bandwidth_model_test.cc`。
策略细节见《request实例映射与KV冷热管理策略说明.md》。

## 1. 本仓是什么

- **策略语义（保留对象，未改动）**：PD 分离：6P:3D + StaticPdMapping 静态路由；session_lru_recompute（默认）与 legacy（FCFS 队头阻塞 + kv_event_payload_legacy.json）双变体；RESIDENT/EVICTED 两态；session 驻留状态/字节数由运行时 KV 账本（SessionKVCacheManager）动态维护，无 sidecar
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
#    arrival_time < 30e9 ns；物化器：traces/derive_20_first_30_seconds.py（运行 stdout 即权威 provenance 记录））
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
python3 sh_test_mesh/workload/llama2_7b_inference/online/verify/ledger_reconcile.py --bridge-dir <run>/bridge --manifest <ET_DIR>/manifest.json（期望值缺省 1177/112，异窗口传 --expected-requests/--expected-accepted-sessions；详见 --help）

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
    13476831232 / 524288 B/token，test_wsc_llm_scheduler.py:646 冻结；
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
（OnlineCli.hh 契约；家族内未知旗标硬错）。机制类旗标当前一枚：

| 旗标 | 取值 | 缺省 | 语义 |
|---|---|---|---|
| `--online-node-gc` | `0\|1` | `0`（关） | M2 节点 GC（2026-08-23）：开启时 GraphBatchCommitter 在每次批提交的静止点（issue pass 完全返回后）回收各 rank NodeStore 中"已 finish 且无未完 children"的节点，并按同水位修剪 (rank, json id) → store id 映射，C++ 侧内存维持在途窗口而非全程累计图（30s 冻结输入实测 cpp 峰值 −60%）。被回收节点必已 finished，仍指向它的跨批边按 NodeStore 死父规则无阻塞（validate 仅按 per-rank 稠密前缀水位线放行已修剪 id，从未存在的 id 照旧 fail-closed）。`0` = M2 前永不删除行为。决策序列与全部工件不受该旗标影响（GC 开/关两臂均字节对拍验收）。**默认关的裁决依据（2026-08-23 翻转）**：GC 开存在可复现的轻载墙钟回归（30s 档 +35~55%，机制未明，隔离基准反快 27%，证据见 /tmp/accel_c/DONE open_finding）；按"不为省内存大幅换 CPU"红线默认关，重载/多实验并行等内存受限场景显式 `--online-node-gc 1`（此时 OOM 风险大于墙钟代价）；重档 on/off 对比数据补齐后可一行翻转默认。 |

runner 脚本不传该旗标（走缺省 0 = 墙钟中性）；重载/多实验并行等内存受
限场景在 runner 命令行追加 `--online-node-gc 1`（cpp 峰值 −60%，决策工件
字节不变）。运行期证据：cpp.log 启动行 `[online] node gc: ...`、
结束行 `[online] node gc: erased=... retained=...`。

## 4. 机制回归 fixtures

`run_online_idle_fixture.sh`（IDLE 五态生命周期）、`run_online_wakeup_guard_fixture.sh`、
`run_online_same_tick_milestone.sh`、`bridge_race_stress_repro.sh`——机制层健康自检。

另有 C++ 单测 fixtures（`build/astra_analytical/build_congestion_aware/bin/`，无参数直跑）：
`..._LocalHbmBandwidthModelTest`（本地 HBM 带宽竞争数值/join 两序/回退，见 G 节）、
`..._NodeStoreTest` 等。

## 5. 目录导览（关键路径）

- `astra-sim/workload/execution_driven/`：在线机制层（C++）
- `sh_test_mesh/workload/llama2_7b_inference/online/`：在线调度器/构图器/服务层（Python）
- `.../online/verify/`：对账与验证工具
- `sh_test_mesh/run_scripts/`：全部 runner 脚本
- `sh_test_mesh/workload/llama2_7b_inference/traces/`：物化器脚本（数据件由调用方物化，provenance 以物化器 stdout 为准）
- `sh_test_mesh/slo_tools/`：SLO 离线后处理工具集（slo_stats / load_imbalance / restore_decomposition / kv_cache_adapter / hopbytes + `slo_params_manifest.json`（B 类参数唯一来源，B4 已填推导值）+ tests；纯离线只读，详见目录内 README.md）
- `sh_test_mesh/tests/` + workload 根：pytest（基线：24+33 = 57 passed，无预存失败；
  §2 ② 物化后 trace_config 指向真实队列时，`test_wsc_llm_scheduler.py` 的
  request-neutral 占位断言红为已知环境效应，与代码态无关——还原裸仓即绿）

## 6. 边界与纪律

- 仿真输入唯一允许源 = astra_compute_20.csv 前 30 秒（更早的用户指示曾临时
  授权过更大窗口；以当下指示为准）。
- 缺失输入一律 fail-closed（generate 桩/materializer/runner/GEN_MATCH 均实测 exit=1）。
- 策略文件（wsc_llm_scheduler.py / session_kv_manager.py）为保留对象，勿改。
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
  排下一 turn 的 arrival alarm。
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
  prefill_chunks>0，两类字段互斥（无混合列车）。legacy 变体
  （kv_cache_policy=legacy）不走列车，保留旧两段式 `emit_decode_batch`。
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
  history_tokens_before 推导值（`plan_materializer.py:61-88`）；会话 KV 的
  规模与位置由运行时 `SessionKVCacheManager` 账本动态维护
  （`wsc_llm_online_scheduler.py:259`；`session_kv_manager.py:367`）：驻留
  命中直接复用、跨实例历史先经 NoC 迁移段、被逐出则窗口内重算。
- long double 时间精度适配：大 ns 级首达偏移超出 IEEE-754 double 精确整数
  范围（2^53）时，分析网络适配层返回 ASTRA-sim 时间以 `long double` 保持
  64 位事件时间精确，避免完成回调与事件映射键错位 1 ns
  （`astra-sim/network_frontend/analytical/common/CommonNetworkApi.cc:71-81`）。


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
