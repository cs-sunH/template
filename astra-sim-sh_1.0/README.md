# astra-sim-sh_1.0 - 晶圆级芯片（WSC）LLM 推理架构与机制

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
  profile 档位选择）；
- **网络接口（路由端口）**：mesh 上的一个 Device，经有向链路连接其在 mesh 中的物理邻居；
- **远端内存端口**（仅边缘芯粒拥有，且仅在启用远端内存扩展的配置中生效，见 D）：
  与片外统一内存池直连的中转通道，对应物理系统中的 SerDes 等片外互连。

> 术语辨析：本说明中"芯粒"＝mesh 上的完整单元（die＝NPU＝rank）。指标层另有一个
> 仅用于观测展示的"chiplet 投影"（把每 NPU 的本地 HBM 均分为若干份，份数取
> `metrics_config.json` 的 `chiplets_per_npu`，对应论文场景中的 HBM 芯粒划分），
> 不参与仿真与调度决策，与本术语无关。

### C.1 本地 HBM 多用户带宽竞争模型（`hbm-bandwidth-contention`）

真实硬件中每颗芯粒的本地 HBM 是共享资源：COMP 访存、NoC 点对点传输的数据端点、
远端池流量的本端端点竞争**同一份带宽**。本仓用每 rank 一个
`astra-sim/workload/LocalHbmBandwidthModel.{hh,cc}` 流体模型建模：

- **带宽口径**：读写共享同一标量 `local-mem-bw`（1640 GB/s），不区分读写方向、
  不区分峰值/持续带宽；`local-mem-latency`（100 ns）在每个作业启动时计一次。
- **三类流量**（同总线竞争）：
  1. **COMP 节点**（roofline 访存）：bytes=tensor_size（读写合并计费），
     ops=num_ops 按峰值算力并行排空，两者都排空才完成（单用户保持 roofline
     max() 语义）；
  2. **NoC p2p 数据端点**：发送方 rank 的 HBM 读 + 接收方 rank 的 HBM 写
     （bytes=comm 字节数，1B ACK 照常计）；NoC 多跳由路由器直通，途经芯粒不占
     其 HBM；
  3. **远端池流量端点**（ET 属性标注，Python 生成器标注、C++ 执行）：直连边缘
     芯粒的 remote_store 记 HBM 读（`hbm-access-mode:1`）、直连 remote_load 记
     HBM 写（`hbm-access-mode:2`）；经 NoC 的 remote_store/remote_load 中，边缘
     芯粒作 NoC↔SerDes 直通（`hbm-charge:false`）不占其 HBM，每条字节流在真实
     数据端点恰好计一次。`noc_migrate` 两端自动计费，无需标注。
- **仲裁**：某 rank 同时有 N 个活跃 HBM 作业→各得 full_rate/N **严格均分**；
  任一完成立即事件驱动重分配。COMM/MEM 节点完成 = join(网络/端口侧完成,
  本端 HBM 作业完成)（`Workload.cc` 内两事件 join，不改网络 API）。
- **范围外**：远端池自身竞争（端口之内严格 FIFO、池内汇聚不建模）、TP 集合通信
  （PacketBundle 3x 路径）不参与本模型。
- **配置与指标**：system.json `hbm-bandwidth-contention`（bool，代码默认 true；
  `local-mem-bw<=0` 自动视为 false 完全回退旧行为；本仓模板已置 1）。指标新增
  `[METRIC]` 每 rank `hbm_busy_ns`、`hbm_served_bytes_{comp,comm_read,comm_write,
  pool_read,pool_write}`、`hbm_peak_concurrent_jobs`、`hbm_reallocation_events`、
  `hbm_real_util`（旧指标键不变）。单测 fixture：
  `AstraSim_Analytical_Congestion_Aware_LocalHbmTest`（均分数值、join 双序、
  不计费豁免、flag 回退）。

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
- **本仓配置**：启用远端池——`PER_NPU_MEMORY_EXPANSION`，全部边缘端口生效。
  本仓 KV 冷热分层为**两态**：一个 session 的全部层 KV 要么整体驻留某实例本地
  HBM（`LOCAL_HBM` 热层），要么整体存入远端池（`REMOTE_MEMORY` 冷层），逐出/恢复以
  整 session 全部层为最小粒度；冷层恢复一律远端全量取回（remote_load，不重算），
  跨实例历史 KV 走 NoC 迁移（noc_migrate）。

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
instance）：每实例同时承担 prefill 与 decode（一个混合 iteration = 队首请求推进
一个 prefill chunk + 全部活跃 decode 各前进一个 token）。

### F. 本仓在五仓中的定位

| 仓库 | 实例组织 | 请求→实例映射策略 | KV 驻留与恢复 | 远端内存池 |
|---|---|---|---|---|
| astra-sim-face | 统一实例（P+D 同实例） | FACE 原始映射：prefill 选剩余 chunk 最少；decode 在邻接图加权距离限制（阈值 = D2D 带宽 / 本地 HBM 带宽）内按 per-die Roofline 增量代价 | RESIDENT/EVICTED 两态；LRU 逐出＝零代价删除；恢复＝重算 | 未启用（NO_MEMORY_EXPANSION） |
| astra-sim-wscllm | PD 分离（Prefill-only + Decode-only 分区） | prefill 选排队请求最少；decode 用静态一跳 P→D 映射 | RESIDENT/EVICTED 两态；LRU 逐出；恢复＝重算（跨实例历史走 NoC 迁移） | 未启用（NO_MEMORY_EXPANSION） |
| **astra-sim-sh_1.0（本仓）** | 统一实例 | prefill HBM 可行过滤 + 剩余 chunk 最少；decode 按 per-die Roofline 增量代价 | LOCAL_HBM/REMOTE_MEMORY 两态；整 session 粒度逐出；恢复＝远端全量取回；本地 HBM 多用户带宽竞争（COMP / p2p 端点 / 池端点，`hbm-bandwidth-contention`） | 启用（全部边缘芯粒挂端口） |
| astra-sim-sh_2.0 | 统一实例 | prefill Roofline 剩余负载均衡（历史 KV 全/部分驻留与全逐出统一）；decode 按 per-die Roofline 增量代价 + HBM 剩余 tie-break | 三态（含半驻留 PARTIAL）；两阶段类型感知逐出（human 类先于 tool 类）；流水化部分恢复 + HBM 恢复/推理带宽共享 | 启用（全部边缘芯粒挂端口） |
| astra-sim-sh_3.0 | 统一实例 | 三段式 prefill（首请求避边缘 / HBM 命中 sticky / 远端命中负载均衡）；decode 本地化固定同实例 | 三态；两阶段逐出；流水化部分恢复（机制同 sh_2.0） | 启用（全部边缘芯粒挂端口） |

五仓还共享同一套 Execution-Driven 在线仿真机制层（`astra-sim/workload/execution_driven/`：
RequestIngress / DecisionMailbox / DecisionBridge / GraphBatchCommitter 等）：策略决策由
Python 在线服务层实时给出，计时由 C++ 物理时钟推进；各仓仅保留路径③（strategy 关感知）
与路径④（strategy 开感知）两条在线路线。五仓在线决策均不再使用任何离线 LUT
（残留 LUT 查表已随静态链路一并删除）：face / sh_1.0 / sh_2.0 的 decode 候选代价由
在线 Roofline 模型即时计算（`per_die_delta_ns`），决策确定可复算复放。

## 1. 本仓是什么

- **策略语义（保留对象，未改动）**：队列深度均衡：PrefillQueueSnapshot + HBM 可行性过滤；edge-rank 远端存取；LOCAL_HBM/REMOTE_MEMORY 两态 KV；prefix 走 recompute；session 历史与 KV 状态由运行时账本动态维护（到达时增量累积，无 sidecar）
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
#    arrival_time < 30e9 ns；物化器：derive_20_first_30_seconds.py
#    [source] [recompute_queue] [canonical_sidecar] [window_ns] [arrival_scale]）
#    产物放 sh_test_mesh/workload/llama2_7b_inference/traces/，
#    并把 trace_config.csv 第 12 行 request_queue_csv 指向它

# ③ 生成 plan 目录（runtime_config 四小件 + manifest + metrics_manifest）
cd sh_test_mesh/workload/llama2_7b_inference && python3 plan_materializer.py && cd <仓根>

# ④ 跑③④（GEN_MATCH：generated/ 下须恰一个 llama2_7b_inference_54npus_* 目录）
bash sh_test_mesh/run_scripts/run_online_strategy.sh <run_dir> <绝对路径 request_csv>
bash sh_test_mesh/run_scripts/run_online_strategy_sensing.sh <run_dir> <绝对路径 request_csv>

# ⑤ 指标后处理 + ④对账
bash sh_test_mesh/run_scripts/run_metrics_postprocess.sh <run_dir>/cpp.log
#    runner 已自动附带 --out-request=<run_dir>/request_metrics.csv（full 档；
#    见 §3.2），此步仅为手工重跑/多 run 合并时使用
python3 sh_test_mesh/workload/llama2_7b_inference/online/verify/ledger_reconcile_sh10.py --run-dir <run_dir> --expected <请求数>

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

### 3.2 指标档位、逐请求 CSV 与物化器参数（SLO B1，2026-08-26）

**--metrics-detail 档位（WP0）**：runner×2 变体（strategy/sensing）不再硬
编码 `summary`，改为优先级 **env `SH_METRICS_DETAIL` > 本仓
`sh_test_mesh/workload/llama2_7b_inference/metrics_config.json` 的
`detail_level`**（python3 解析）。取值恒为 `off|summary|full`；走 json 回退
时文件缺失/非法/取值非法一律 fail-closed 报错退出，env 取值非法同样
fail-closed。runner 启动行打印解析结果与来源
（`metrics detail=<档> (source: env SH_METRICS_DETAIL|json <路径>)`）。

**request_metrics.csv（WP1）**：full 档运行结束后，runner 自动调用
`run_metrics_postprocess.sh --out-request=<run_dir>/request_metrics.csv`，
从 cpp.log 的 `[METRIC] {"type":"request",...}` 逐请求记录 + 队列旁
manifest.json 生成逐请求 CSV（冻结 29 列，列集按 SLO 执行方案 §2 一次定死；
填 timing/terminal_status/instructions 与 prefill/decode/prefix 长度；
first_token 族由 B3/B4 填充——exact 事件 / 归档 ledger 的
train_interpolated proxy / NA 两段语义（§3.3/§3.5）；kv_hit_state 与
restore/hidden 族运行内 CSV 恒 `NA`，由 `slo_tools` 离线适配器
（kv_cache_adapter / restore_decomposition）另行核算输出）。fail-closed：
manifest↔请求行按 queue_index/request_id 连接失败、行数不符、或顺序不变量
违例（arrival≤completion、e2e=completion−arrival、分解之和=e2e）即报错退出。
`off` 档无 [METRIC] 行、`summary` 档无逐请求记录——两种情况不产该文件，
postprocess.log 有一行说明。request_type 规则：human_time 非空→human；
tool_time 非空→tool；皆空→turn 0 记 human、其余记 unknown（计数在
postprocess.log 的 JSON 摘要里上报）。

**物化器参数与到达缩放（WP2）**：
`traces/derive_20_first_30_seconds.py [source] [recompute_queue]
[canonical_sidecar] [window_ns] [arrival_scale]`——`window_ns` 缺省 30e9；
`arrival_scale`（float>0，缺省 1.0）**仅把每个 session 的 turn-0
session_arrival_time_ns 除以 scale**（四舍五入到 ns；scale=1 逐字节不变），
`inter_request_interval_ns` 与窗口成员判定（源时间线）不动，缩放不增删
请求。新审计信息（human_time_ns/tool_time_ns/request_type）只进 canonical
sidecar（列追加在 digest 之后）与 manifest.json——队列 CSV 的 8 列契约
不变。`plan_materializer.py` 把 sidecar 的这三个字段透传进 manifest.json
每请求条目（只加键不删不改既有字段；sidecar 缺失/缺行时按上述 request_type
规则回退并在 stdout/计数上报）。

### 3.3 C++ 观测层：水位线/链路积分/首 token（SLO B2，2026-08-26）

**slo_sampling 注入（B2-A）**：`plan_materializer.py` 组装
metrics_manifest.json 时注入 `"slo_sampling":
{"watermark_period_ns": X, "link_bucket_ns": Y, "provisional": bool}`；值取
本仓 `sh_test_mesh/slo_tools/slo_params_manifest.json` 对应参数的 `value`
（B4 已填推导值：watermark 5,000,000 ns / link_bucket 20,000,000 ns；
null/缺失→临时锚点 **5,000,000 ns**、provisional=true）。C++
`MetricCollector::load_manifest` 解析该节（缺节/非法→锚点
+init 记录 warning，不 fail 运行），init 记录与每条相关记录回显实际周期。

**HBM 水位线（WP8）**：finalize 侧在 capacity_timeavg 的 delta 重放循环内
做 per-bucket 峰值扫描（事件驱动+周期兜底等价于分段常数的逐桶最大值）。
新 `[METRIC]` 记录：`hbm_watermark`（per instance×bucket，full 档；空桶不
发，仅发 occupancy 变化桶+首/末桶）与 `hbm_watermark_summary`（per rank，
summary/full 档；含 bucket 路径 timeavg 与重放 timeavg 的 ≤1% 交叉校验、
capacity_violation_count 恒 0 断言）。**在线模式 memory_actions 为空是既有
现状**：此时输出结构为零的水位线记录并注明原因（权威水位线由离线
ledger.jsonl 重放补充）。

**NoC 链路观测（WP6）**：`FluidScheduler` 内只读 side-band 积分（rate×dt
按 link×bucket 累积，membership/rate 变化点推进；不注册仿真事件、不改流
推进）。门控 = metrics≠off **AND** env `ASTRA_LINK_OBSERVER`≠`"0"`（缺省
开启；off 时零工作）。新 `[METRIC]` 记录：`link_bucket`（summary/full；
空桶不发，仅发有活动桶+首/末桶）与 `link_total`（per link）；edge_* 字段
为 remote-memory 端口 rank（mesh 边界） incident 链路集合。运行结束
`release_link_observer_buckets()` 释放积分数组（RSS 纪律）。

**首 token 事件码 8（WP9-C++）**：`MetricCollector` 枚举加
`FIRST_TOKEN_COMPLETE=8`（complete 边，subject=request→queue_index，多 rank
取 min）；`main_online.cc` 锚点注册对名字含 `first_token` 的节点按
(request_id, rank) 注册（幂等，观测专用——不进 start-anchor 分组）。
request 记录新增 `first_token_ns`（无事件→null+`first_token_note`；序不
变量 arrival≤first_token≤completion 违例计 consistency_violation 并置 null。
decode_length==1 的相等不变量已按 WP9_CONTRACT §6（2026-08-27 主控裁决）
放宽：code 8 跨 rank 取 min、completion 取 max，TP 斜台使二者天然不等，
改为信息性字段 `first_token_completion_skew_ns`（=completion−first_token，
仅 dl==1 且两者非空时输出；不再 withhold、不计 violation；manifest 无
decode_length 时不可评估并注明）。metrics_manifest requests 现透传 `decode_length`；
request_metrics.csv 的 first_token_ns/first_token_source 列在有 exact 事件
时填 exact；exact 缺失且归档 train_ledger 可得时填 train_interpolated
proxy（B4 起默认口径，见 §3.5）；运行内产物恒 NA（两段语义）。

### 3.4 首步批拆分与 first_token 标记（SLO B2 线3，2026-08-26）

**事件码 8（python 侧）**：`metrics_schema.py` 新增
`EVENT_FIRST_TOKEN_COMPLETE = 8`（登记进 `EVENT_EDGE_BY_CODE`→complete 与
`SERVICE_EVENT_CODES`；五仓逐字节相同的 3 行最小插入）。

**首步批/余量批拆分（WP9-python）**：`graph_batch_builder.py` 把含 debut
成员（`decode_tokens_consumed==0`，即本交付加入列车的 joiner）且迭代数
≥2 的列车拆两批发射——**首步批** = joiner 迁移/readiness barrier/起始
标记 + 所有成员第 1 个 span + prefill 队头第 1 个 chunk
（weight_passes=1）+ 各 debut 成员的 first_token 标记节点（1-op COMP，
名字含 `first_token` 子串，携 debut request_id/stage=decode——C++ 按名字
子串锚 code 8 取每 rank min tick）；**余量批** = 剩余迭代/chunk
（weight_passes=iterations-1）+ drain/exit/哨兵标记 + end barrier（挂点
语义不变）。两段激活/KV/AR 逐 span 求和与整列一致，权重 1+(k-1)=k 守恒。
decode_length=1 的 debut 成员不挂独立标记，其 exit 标记名附加
first_token 子串（同节点 code4/code8 双锚点）。总开关 env
`SH_FIRST_TOKEN_SPLIT`（**B4 起缺省 `"0"` 关**——60s 决策等价门-2 失败
退回 proxy，见 §3.5；显式 `"1"`=启用拆分，`"0"`=行为与上线前逐字节
一致）。

**首步唤醒 watch（工程化偏移）**：首步批无请求级 watch，其完成后不存在
决策工作，C++ tick-end 门不会再交付（运行尾部死锁）；故首步批尾部附加
一个批命名空间唤醒标记（`<train_id>_first_step`，batch_train_ 前缀，复用
C++ 哨兵 watch 通道——train 级、无请求 eligibility），fire 经 PREFILL_DRAIN
通道送回，调度器 `_consume_first_step_wakeup` 识别即无操作（不决策/不记
账/不写 decision_log），余量批在 `_plan_and_emit_trains` 的 busy 分支发射
（本交付或任一后续交付；余量节点经依赖边排在首步节点后，早发不改物理
序）。首步批的 train_ledger 行带 `"first_step": true` 标记（ON/OFF 对拍
剥离清单）；commit ack 走基类协议记账，变体零动作。

**decode_length=1 不变量（B2.5 已按 WP9_CONTRACT §6 放宽）**：C++ 侧 code 8
跨 rank 取 min 而 code 4 completion 取 max（MetricCollector.cc 聚合不对称），
TP=6 rank 斜台下同节点双锚点无法严格相等——按 WP9_CONTRACT §6 裁决
（B2.5，五仓统一）不再要求相等：finalize 只保留次序检查
（arrival≤first_token≤completion），dl==1 行附加信息性
`first_token_completion_skew_ns`（completion−first_token），不再
withheld/计数（见 §3.3）。

### 3.5 WP9 退回处置：拆分缺省关 + train-interpolated proxy（SLO B4，2026-08-27）

- **缺省翻转**：`SH_FIRST_TOKEN_SPLIT` 缺省 `"1"` → `"0"`
  （`online/graph_batch_builder.py::first_token_split_enabled`，显式 `=1`
  仍可启用拆分取 exact 首 token，研究/对拍用）。理由：B3_S1 60s 决策等价
  门-2 失败——拆分的物理扰动（首步批 → 每拆分列车多一次交付 → tick
  漂移）在闭环逐轮放大（首分歧 decision row 329 / sim t=16.27s
  +565us，其后 4009/4362 决策分叉、prefill 实例选择翻转 1177 处），
  ON/OFF 字节等价在 60s 窗不可达（2s 窗保持通过；确定性对照
  t3_off==B0 基线逐字节相同，排除运行噪声）；调度器 tie-break 属禁改
  区。证据：`/tmp/slo_wps/gates/B3_S1.FAILED`（gate2 FAIL）、
  `/tmp/slo_wps/b3/S1/`（t3_full_off_split vs t3_full_split_on 对拍）。
  拆分语义测试缺省断言同步翻转
  （`online/test_first_token_split.py::SplitEnvironmentTest` 钉缺省关；
  `DebutPlanTest` setUp 显式置 "1"）。
  **研究专用**：`SH_FIRST_TOKEN_SPLIT=1` 的 exact 口径仅供机制研究（60s 决策等价门未过，证据见仓内 README 所引门文件）；论文指标一律采用默认 proxy（`train_interpolated`）口径，proxy 不得进入任何 SLO 判定路径。
- **proxy 填充**（`metrics_postprocess.py`）：exact（事件码 8）缺失且
  归档 `results/train_ledger.jsonl` 可得时填
  `first_token_proxy = decode_start_ns + (w₁/Σᵢwᵢ)×(first_train_end_ns − decode_start_ns)`，
  `wᵢ = W_bytes + KV_bytes(context+consumed+i)`，`first_token_source
  = train_interpolated`：
  - **Σ 域 = 首列车 iterations N**（i=1..N，与 S3/FACE/W 裁定一致）：
    debut 首 token 随列车第 1 迭代完成，权重按列车 N 迭代线性化（对
    debut 自身参与度 P 求和会在 decode_length<iterations 时退化为
    share=1 并越过 completion）；debut join 时 consumed=0（sticky
    decode，60s 参考跑 1454/1454 恰一次 joiner 加入）。
  - **first_train_end = 同实例下一列车发射边界**：台账行 tick 是发射
    边界，实例 busy 门 = 一列车在飞（in_flight_train），故同实例下一行
    tick 即 end barrier 后首个决策边界；实例末列车 → NA（不可得不
    编造）。
  - **N=1/短列车钳制**：share 接近 1 → proxy=下一发射边界可能 >
    completion（边界晚于 end barrier tick），钳到 completion 并在
    instructions 留痕 `clamped_to_completion`（60s 重跑 2/1454，其一
    N=2）。填充值过 fail-closed ordering 检查
    （arrival≤first_token≤completion）。
  - **换算权威**：W_bytes/KV_bytes 用本仓只读依赖
    `face_scheduler.estimate_model_weight_bytes` /
    `kv_cache_bytes_for_tokens`（trace_config.csv 参数：swiglu →
    13476831232 / 524288 B/token，test_face_scheduler.py:346 冻结），
    不另行编造。
  - **运行内 CSV 保持 NA**（ledger 归档前不可得，instructions 记
    `proxy_unavailable:no_train_ledger(results/)`）；对归档 run 离线
    重跑 postprocess 填充——运行产物与离线分析两段语义。summary 行
    新增 `first_token_source_counts`。
  - **SLO 防线**：proxy 仅展示口径，`first_token_ns/first_token_source`
    被 `slo_common.assert_no_proxy_columns` 从一切判定路径 fail-closed
    拒绝（`slo_tools/tests/test_slo_contract.py::
    test_train_interpolated_proxy_cannot_enter_judgment`：夸张 proxy 值
    不改变 violation verdict，判定输入携带 proxy 列被拒）。
- **离线验证**（`/tmp/slo_wps/b4/S1/proxy60/`，源=B3 归档
  t3_full_off_split + 重物化 manifests（digest bd000c54 复现，计数
  1454/116 与运行时一致））：1454 行全部 train_interpolated、0 序违例
  （填充值过 fail-closed ordering 检查）、raw/normalized CSV 与归档
  逐字节相同（其余列零 diff，仅 first_token_ns/first_token_source/
  instructions 变化）；抽 3 请求手算（manifest+ledger+cpp.log 一次产物
  独立重算）全部吻合，含 1 例 N=2 钳制（qi_1095 raw=252249079189 >
  completion → 252123429690）；proxy vs exact（t3_full_split_on）分布差
  p50 −11.8%/p99 +16.2%/mean −6.1%（信息性：两时间线已因门-2 级联
  分叉，属预期）。
- 2s 冒烟：缺省（不设 env）21 行 first_token 全 NA + source NA
  （运行内两段语义）；`SH_FIRST_TOKEN_SPLIT=1` 恢复 exact 21/21、
  0 序违例。
- 单测：`sh_test_mesh/tests/test_first_token_proxy.py`（新，9 用例）；
  slo_tools 契约 41/41（+proxy 判定防线）；metrics 契约 34 OK；
  拆分语义 39/39（online 全量）。

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
- `sh_test_mesh/tests/` + workload 根：pytest（基线：25 skip5+33 = 58 passed+5 skipped，无预存失败；
  `test_face_scheduler.py` 的 request-neutral 占位断言是裸仓库态检查——按 §2 ②
  物化输入并把 trace_config 指向真实队列后该测试即红，属已知环境效应，
  与代码态无关，恢复占位指针即复绿）

## 6. 边界与纪律

- 仿真输入唯一允许源 = astra_compute_20.csv 前 30 秒（更早的用户指示曾临时
  授权过更大窗口；以当下指示为准）。
- 缺失输入一律 fail-closed（generate 桩/materializer/runner/GEN_MATCH 均实测 exit=1）。
- 策略文件（face_scheduler.py（策略与 KV 语义同文件））为保留对象，勿改。
- 改动机制层后请跑 §4 fixtures + §2 ⑤ 对账再交付。

## 7. Online execution adaptation

在线策略路线（③/④）是实时宿主调度器：调度器进程内逐决策边界做映射决策，
把结果作为 per-rank 图批次发射给执行驱动引擎，决策不预先固化；每 tick 先
处理 completion 批（PREFILL_DRAIN / DECODE_COMPLETION / REQUEST_COMPLETE），
再处理 arrival 批，最后跑一次准入重查
（`sh_test_mesh/workload/llama2_7b_inference/online/sh10_online_scheduler.py:130-157`）。

- 决策边界与仓内路由：ARRIVAL 批先做 HBM 可行性过滤，再按队列深度均衡选择
  prefill 实例并 reserve/prepare（`:161-274`）；PREFILL_DRAIN 以加权图候选 +
  Roofline per-die 增量代价选择 decode 实例（`:286-374`）；
  DECODE_COMPLETION/REQUEST_COMPLETE 按 completion_order 先全部 mark_complete
  再全部 enforce_reserve（`:378-438`）。
- request_aggregated 折算口径：每个请求相位按 rank 折成 17 类算子节点
  （13 类层内算子 + attention/MLP 两个 All-Reduce + final norm + logits），
  聚合 FLOPs、激活/KV/HBM 字节、可选远端读与集合通信载荷与成员-迭代展开
  总量恒等，只压缩重复层/chunk/token 与集合通信启动次数；权重字节按
  `weight_passes` 口径计（默认 = span 数，历史 batch=1 串行口径逐字节
  不变；迭代列车传迭代数——权重每物理前向只读一次，与批成员数无关，
  2026-08-22 拼 batch 改造，`generate_trace.py:781-1055`）；
  在线发射仅支持该粒度，其它粒度 fail-closed
  （`online/graph_batch_builder.py`）。
- 发射并发骨架（本仓形态：迭代列车拼 batch，2026-08-22 改造）：连续
  batching——decode 互拼、decode 与 prefill chunk 混拼（每迭代 ≤1
  chunk）、chunk 之间不拼、批成员只在迭代（列车）边界变化。每实例
  状态机 = FCFS prefill 队列 / active_decode 批成员表 /
  pending_decode_ready（KV 就绪待加入）/ in_flight_train（唯一在飞
  列车 + train_id/membership_digest，busy 门 = 一个列车在飞）；列车
  终点 = 下一个不可预测事件（队列头 prefill drain / 全部工作耗尽），
  默认不设 T_max。发射 = 准入动作（到达 gates/历史迁移/逐出/屏障）→
  迭代列车（joiner 迁移 + 共享 readiness barrier + 折叠列车体
  （weight_passes=迭代数）+ drain/exit 标记节点 + 每列车一个共享 end
  barrier）→ 完成段（completion_evictions + 下一 turn interval
  gates）。列车账本核销幂等（同列车标记 watch 跨 tick fire、事件拆
  交付），推进量全部闭式；`train_ledger.jsonl` 落每列车审计行
  （§7.3 不变量断言输入）。
- 接栅栏与段末屏障：发射不做块末恢复/段内清链，per-rank `previous_id`
  无条件接续当前 frontier（per-rank 发行序 = 全局发射序，2026-08-19 五仓
  统一），列车作为整体接续 frontier，跨请求 P2P 与集合通信参与序不会
  反转成环；列车以 TP 组共享 `*_end_barrier` 收尾，drain/exit 标记
  （列车体后、barrier 前）承载 PREFILL_DRAIN / DECODE_COMPLETION watch
  与指标锚点（R2-2"barrier 前末节点"口径的列车化延续）。
- KV 迁移/远端存取的图语义：会话内 NoC 迁移为配对 comm_send/comm_recv 传输
  节点加完成 ACK；edge 远端存取按 remote_store（comm_send 至 edge + edge
  `mem_store` + ACK）/ remote_load（edge `mem_load` + NoC 投递）表达，恢复
  侧经 timer/依赖门在计算前就绪
  （`sh_test_mesh/workload/llama2_7b_inference/generate_face_trace.py:630-800`；
  `online/graph_batch_builder.py:523`、`:673`）。
- 折入 recompute 输入口径 + 运行时 KV 账本：请求队列为唯一仿真输入，
  turn-0 源前缀折入该行 prefill_length（整段重算口径，
  `traces/derive_20_first_30_seconds.py:14-16`）；manifest 仅携带队列派生的
  history_tokens_before 推导值（`plan_materializer.py:63-88`）；会话 KV 的
  规模与位置由运行时 `KVCacheManager` 账本动态维护
  （`sh10_online_scheduler.py:97`；`face_scheduler.py:991`），会话历史逐到达
  累积（`:172-195`）：驻留命中直接复用、跨实例/远端历史先迁移、被逐出则
  窗口内重算。
- 逐出补偿的 pending 门同步（TOCTOU 修复，2026-08-23）：账本逐出在**决策
  时点**同步翻转会话位置，图发射器 pending 门的位置快照补偿
  （remote_store 逐出 → `_mark_pending_history_remote`）也必须同边界执行——
  调度器 9 个 KV 变更点中返回逐出转移的 7 个
  （`sh10_online_scheduler.py` 改法D 1/9~6/9、9/9）在 bump 行后立即调用
  `graph.sync_pending_history_after_evictions(transfers)`（吃 `KVTransfer`
  对象，过滤条件 `remote_store`）。**这是唯一的标记路径**：三处发射包装器
  （`online/graph_batch_builder.py` 准入/列车/完成段）内的既有发射时补偿
  已于同日修订移除——sh_2.0 seq4689 事故证明，多级逐出发射乱序时迟到的
  旧转移标记会把门回退到过期位置，"双保险"并不幂等。
  `emit_admission_batch` 的 fail-closed 位置一致性检查不变
  （门位置 ≠ 账本快照即中止）。
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
