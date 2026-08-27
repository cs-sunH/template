# astra-sim-sh_2.0 - 晶圆级芯片（WSC）LLM 推理架构与机制

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
  本仓 KV 冷热分层为**三态**：`LOCAL_HBM`（全部层本地）/ `PARTIAL_HBM_REMOTE`
  （前 P 层留本地、后 K 层逐至远端池，P = L - floor(L/2)，L 为模型层数）/
  `REMOTE_MEMORY`（全部层在远端）。逐出为
两阶段类型感知（2026-08-18 起：按 session 下一请求触发类型先 human 后 tool
分两类，类内先只逐后 K 层转半驻留、仍不足才整 session 逐出，四段式
human半→human全→tool半→tool全）；半驻留 session 的后续请求与其他驻留状态
一样按负载均衡映射实例（2026-08-21 起，此前为固定回原实例的硬亲和），映射到
异实例时先经 NoC 迁移已驻留的部分前缀（history_partial_prefix_migrate），
恢复采用"前缀计算与后缀远端加载"流水重叠；本 rank 的全部 HBM
  用户（推理 COMP、KV restore DMA、NoC p2p comm 数据端点读/写、池流量端点读/写）
  按 N-way 流体模型**严格均分**带宽（`full_rate/N`，任一作业完成立即事件驱动重分配；
  读写共享同一总线与总带宽、不区分峰值/持续；`astra-sim/workload/LocalHbmBandwidthModel.cc`，
  system 键 `hbm-bandwidth-contention` 默认开，置 0 完整回退旧两用户 50/50 行为做 A/B）。
- **HBM 端点计费规则（每字节流恰好计一次）**：NoC 多跳路由器直通不占途经芯粒 HBM；
  SerDes 与 NoC 路由器直连、与边缘本地 HBM 两两直连——边缘芯粒 NoC↔SerDes 直通
  （remote_store 经 NoC 的边缘端 comm_recv、remote_load 的边缘端 comm_send）标
  `hbm-charge:false` 不计；只有数据端点计费：p2p 发送端 COMM_READ / 接收端
  COMM_WRITE；remote_store 源即边缘 rank 时其 `MEM_STORE` 标 `hbm-access-mode:1`
  （池流量 HBM 读端点）；remote_load 目标端的 HBM 写由串行跟随的 restore 节点承担
  （其 comm_recv 标 `hbm-charge:false` 防双计）；noc_migrate 两端为真端点自动计费。
  comm/MEM 节点完成 = join(网络/端口事务完成, 本端 HBM 作业完成)。TP 集合通信
  （PacketBundle 路径）与池自身竞争不在计费范围。

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
| astra-sim-sh_1.0 | 统一实例 | prefill HBM 可行过滤 + 剩余 chunk 最少；decode 按 per-die Roofline 增量代价 | LOCAL_HBM/REMOTE_MEMORY 两态；整 session 粒度逐出；恢复＝远端全量取回 | 启用（全部边缘芯粒挂端口） |
| **astra-sim-sh_2.0（本仓）** | 统一实例 | prefill Roofline 剩余负载均衡（历史 KV 全/部分驻留与全逐出统一）；decode 按 per-die Roofline 增量代价 + HBM 剩余 tie-break | 三态（含半驻留 PARTIAL）；两阶段类型感知逐出（human 类先于 tool 类）；流水化部分恢复 + HBM 恢复/推理带宽共享 | 启用（全部边缘芯粒挂端口） |
| astra-sim-sh_3.0 | 统一实例 | 三段式 prefill（首请求避边缘 / HBM 命中 sticky / 远端命中负载均衡）；decode 本地化固定同实例 | 三态；两阶段类型感知逐出（human 类先于 tool 类）；流水化部分恢复（机制同 sh_2.0） | 启用（全部边缘芯粒挂端口） |

五仓还共享同一套 Execution-Driven 在线仿真机制层（`astra-sim/workload/execution_driven/`：
RequestIngress / DecisionMailbox / DecisionBridge / GraphBatchCommitter 等）：策略决策由
Python 在线服务层实时给出，计时由 C++ 物理时钟推进；各仓仅保留路径③（strategy 关感知）
与路径④（strategy 开感知）两条在线路线。五仓在线决策均不再使用任何离线 LUT
（残留 LUT 查表已随静态链路一并删除）：face / sh_1.0 / sh_2.0 的 decode 候选代价由
在线 Roofline 模型即时计算（`per_die_delta_ns`），决策确定可复算复放。

## 本压缩包的精简仿真入口

本仓库保留了运行 `sh_test_mesh` 中 FACE + HBM/KV 仿真所需的 ASTRA-sim
源码、分析网络后端、Chakra ET 最小依赖、硬件/系统/负载配置和测试。构建产物、
历史日志、通用示例、上游测试集与第三方依赖自带的文档/测试目录已经移除。

```bash
# 1. 物化在线运行所需 plan-dir（runtime config、manifest、metrics manifest；
#    不生成 Chakra .et）
cd sh_test_mesh/workload/llama2_7b_inference
python3 plan_materializer.py
cd ../../..

# 2. 以已构建的 online congestion-aware 后端运行两条动态路径
bash sh_test_mesh/run_scripts/run_online_strategy.sh <run_dir> <absolute_request_csv>
bash sh_test_mesh/run_scripts/run_online_strategy_sensing.sh <run_dir> <absolute_request_csv>
```

`yaml-cpp` 已作为最小源码依赖保存在 `extern/helper/yaml-cpp`，CMake 配置时不再
从网络下载。策略细节见《request实例映射与KV冷热管理策略说明.md》。

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

## 1. 本仓是什么

- **策略语义（保留对象，未改动）**：task-load 均衡：InstanceTaskLoadSnapshot（Roofline 剩余负载）；PARTIAL 驻留同走负载均衡映射（2026-08-21 起，异实例时部分前缀 NoC 迁移）+ 后缀恢复并行流水；三态 KV（LOCAL_HBM/PARTIAL_HBM_REMOTE/REMOTE_MEMORY）；折入 recompute 前缀口径（2026-08-21 起：turn-0 前缀折入队列 prefill_length 全量重算，session 历史由运行时 KV 账本动态维护，无 sidecar）
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
#    arrival_time < 30e9 ns；物化器：materialize_first_30s.py（产折入 recompute
#    单队列 + canonical digest；CLI [source] [queue_out] [window_ns]
#    [arrival_scale]，缺省 30e9/1.0 = 冻结行为逐字节不变，见 §3.2）
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
python3 sh_test_mesh/workload/llama2_7b_inference/online/verify/ledger_reconcile_sh20.py --run-dir <run_dir> --expected <请求数>（ledger_reconcile.py 为其薄入口）

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
| `--online-node-gc` | `0\|1` | `0`（关） | M2 节点 GC（2026-08-23）：开启时 GraphBatchCommitter 在每次批提交的静止点（issue pass 完全返回后）回收各 rank NodeStore 中"已 finish 且无未完 children"的节点，并按同水位修剪 (rank, json id) → store id 映射，C++ 侧内存维持在途窗口而非全程累计图（30s 冻结输入实测 cpp 峰值 −61%）。被回收节点必已 finished，仍指向它的跨批边按 NodeStore 死父规则无阻塞（validate 仅按 per-rank 稠密前缀水位线放行已修剪 id，从未存在的 id 照旧 fail-closed）。`0` = M2 前永不删除行为。决策序列与全部工件不受该旗标影响（GC 开/关两臂均字节对拍验收）。**默认关的裁决依据（2026-08-23 翻转）**：GC 开存在可复现的轻载墙钟回归（30s 档 +35~55%，机制未明，隔离基准反快 27%，证据见 /tmp/accel_c/DONE open_finding）；按"不为省内存大幅换 CPU"红线默认关，重载/多实验并行等内存受限场景显式 `--online-node-gc 1`（此时 OOM 风险大于墙钟代价）；重档 on/off 对比数据补齐后可一行翻转默认。 |

runner 脚本不传该旗标（走缺省 0 = 墙钟中性）；重载/多实验并行等内存受
限场景在 runner 命令行追加 `--online-node-gc 1`（cpp 峰值 −61%，决策工件
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
`traces/materialize_first_30s.py [source] [queue_out] [window_ns]
[arrival_scale]`——`window_ns` 缺省 30e9、`queue_out` 缺省写本目录原生
文件名（description 列保持冻结的 first-30-seconds 字符串，同窗产物与 B0
血统逐字节一致；digest sidecar 写 queue_out 旁的 `*_canonical_digest.csv`）；
`arrival_scale`（float>0，缺省 1.0）**仅把每个 session 的 turn-0
session_arrival_time_ns 除以 scale**（四舍五入到 ns 后**量化到 1000ns 网格**
（最近取整，SLO B3-6 2026-08-27）：纯除法会产出半 µs 值，破坏下游整 µs
不变量（graph_batch_builder 的 timer_gate 要求 duration%1000==0；源到达
本身全在 1000ns 网格上）；scale=1 直通逐字节不变），
`inter_request_interval_ns`、`next_trigger_type` 列与窗口成员判定（源时间
线）不动，缩放不增删请求。脚本全量载入源 CSV，运行时对
`/tmp/slo_wps/locks/heavy_mat.lock` 取 flock 与其他重物化器串行。新审计
信息（human_time_ns/tool_time_ns/request_type/decode_length）只进 canonical
digest sidecar（列追加在前六列之后）与 manifest.json——队列 CSV 的 9 列
契约不变。`plan_materializer.py` 把 sidecar 的前三个字段透传进
manifest.json 每请求条目（只加键不删不改既有字段；sidecar 缺失/缺行时按
上述 request_type 规则回退并在 stdout/计数上报）。

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
为 remote-memory 端口 rank（mesh 边界）incident 链路集合。运行结束
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

### 3.4 首步批拆分、first_token 标记与新增诊断字段（SLO B2 线3/线5，2026-08-26）

**事件码 8（python 侧）**：`metrics_schema.py` 新增
`EVENT_FIRST_TOKEN_COMPLETE = 8`（登记进 `EVENT_EDGE_BY_CODE`→complete 与
`SERVICE_EVENT_CODES`；五仓逐字节相同的 3 行最小插入）。

**首步批/余量批拆分（WP9-python）**：`graph_batch_builder.py` 把含 debut
成员（`decode_tokens_consumed==0`，即本交付加入列车的 joiner）且迭代数
≥2 的列车拆两批发射——**首步批** = joiner 迁移/readiness barrier/起始
标记 + 所有成员第 1 个 span + prefill 队头第 1 个 chunk（weight_passes=1）
+ 各 debut 成员的 first_token 标记节点（1-op COMP，名字含 `first_token`
子串，携 debut request_id/stage=decode——C++ 按名字子串锚 code 8 取每
rank min tick）；**余量批** = 剩余迭代/chunk（weight_passes=iterations-1）
+ drain/exit/哨兵标记 + end barrier（挂点语义不变）。sh_2.0 的 partial
前缀两段式迁移在准入批（不拆）；partial 恢复列车的首 chunk 前缀组
（=首 chunk + 各成员第 1 迭代 span）恰为首步组，两段式天然对齐，partial
账本随首步批弹出、余量批是纯聚合段。decode_length=1 的 debut 成员不挂
独立标记，其 exit 标记名附加 first_token 子串（同节点 code4/code8 双
锚点）。总开关 env `SH_FIRST_TOKEN_SPLIT`（**B4 起缺省 `"0"` 关**——60s
决策等价门-2 失败退回 proxy，见 §3.5；显式 `"1"`=启用拆分，`"0"`=行为
与上线前逐字节一致）。

**首步唤醒 watch（工程化偏移）**：首步批无请求级 watch，其完成后不存在
决策工作，C++ tick-end 门不会再交付（运行尾部死锁）；故首步批尾部附加
一个批命名空间唤醒标记（`<train_id>_first_step`，batch_train_ 前缀，复用
C++ 哨兵 watch 通道——train 级、无请求 eligibility），fire 经 PREFILL_DRAIN
通道送回，调度器 `_consume_first_step_wakeup` 识别即无操作（不决策/不记
账/不写 decision_log），余量批在 `_plan_and_emit_trains` 的 busy 分支发射
（发射不改快照输入，实例纪元不 bump）。首步批的 train_ledger 行带
`"first_step": true` 标记（ON/OFF 对拍剥离清单）。

**新增诊断字段（线5，纯输出、不改任何决策/KV 逻辑；进 ON/OFF 对拍剥离
清单）**：admission 决策序列化 `history_location_before`{,`_instance_index`}
与 `history_resident_prefix_layers`（S2 会话历史三态：local_hbm/
partial_hbm_remote/remote_memory）——修复 B2 线4 发现的"内存有值未序列化，
kv_hit_state 只能 not_supported"；`slo_tools/kv_cache_adapter.py` S2 分支
据此映射 full/partial/full（remote 整体恢复非重算；无 recompute 语义，
miss 不出现属预期），旧产物（字段缺失）回退 not_supported。prefill/
decode/completion 决策附加 `transfer_hop_bytes`（逐传输对象
{total_bytes, noc_hop_bytes, shard_count}；KVTransfer shards 本就按
deterministic_xy_route 生成、在线复用）——`slo_tools/hopbytes.py` S2
collector 据此聚合 Hop-Bytes（coverage 0→1；local_hit 无物理搬运不计入；
旧产物回退 coverage=0）。

**逐出 victim/bytes 序列化（SLO B3-6，2026-08-27，纯输出、进剥离清单）**：
prefill/decode/completion 决策附加 `history_evictions`/
`prefill_evictions`/`decode_evictions`/`completion_evictions`（逐出
KVTransfer 逐条落盘：kind/reason/session_id/total_bytes/source_instance_
index/target_instance_index/layer_start/layer_end，字段名与 S1 对齐，
多带分层 layer 区间）与 `history_transfers`（partial 前缀两段式迁移逐段
对象：prefix noc_migrate + suffix remote_load）。此前 decision log 只落
`*_eviction_count`（他人会话逐出不可归因）与聚合 `history_transfer_
bytes`——`slo_tools/hbm_watermark.py` 的 S2 分支只能 count_only 上界重建。
B3-6 起：字段在场（触发键 `history_transfers`，prefill 决策恒在场）即升级
full 口径（逐出可归因、恢复逐段对账、completion 逐出已含自身释放故关闭
kv_location_after_completion 归因避免双重扣减、violation 检查生效）；
旧产物（B3 前基线）无字段则保底回退 count_only。同补丁一并把物化器
arrival_scale 缩放后 t0 量化到 1000ns 网格（见 §3.2）。

**已知偏离（decode_length=1 不变量，B2.5 已放宽）**：C++ 侧 code 8 跨
rank 取 min 而 code 4 completion 取 max（MetricCollector.cc 聚合不对称），
TP=6 rank 斜台下同节点双锚点无法严格相等——按 WP9_CONTRACT §6 裁决
（B2.5，五仓统一）不再要求相等：finalize 只保留次序检查
（arrival≤first_token≤completion），dl==1 行附加信息性
`first_token_completion_skew_ns`（completion−first_token），不再
withheld/计数（2s 窗实测 21/21 非空、skew 9ns）。

### 3.5 WP9 退回处置：拆分缺省关 + train-interpolated proxy（SLO B4，2026-08-27）

- **缺省翻转**：`SH_FIRST_TOKEN_SPLIT` 缺省 `"1"` → `"0"`
  （`online/graph_batch_builder.py::first_token_split_enabled`，显式 `=1`
  仍可启用拆分取 exact 首 token，研究/对拍用）。理由：B3_S2 60s 决策
  等价门-2 失败——拆分的物理扰动（首步批 → 每拆分列车多一次交付 →
  tick 漂移）在闭环逐轮放大（首分歧 decision row 258，其后 4052/4362
  决策分叉、实例选择翻转），ON/OFF 字节等价在 60s 窗不可达（2s 窗保持
  通过；同形态 S1 失败已由 t3_off==B0 逐字节相同的确定性对照排除运行
  噪声）；调度器 tie-break 属禁改区。证据：
  `/tmp/slo_wps/gates/B3_S2.FAILED`（gate2 FAIL）、
  `/tmp/slo_wps/b3/S2/`（t3_full_off_split vs t3_full_split_on 对拍）。
  拆分语义测试缺省断言同步翻转
  （`online/test_first_token_split.py::SplitEnvironmentTest` 钉缺省关；
  `DebutPlanTest` setUp 显式置 "1"）。
  **研究专用**：`SH_FIRST_TOKEN_SPLIT=1` 的 exact 口径仅供机制研究（60s 决策等价门未过，证据见仓内 README 所引门文件）；论文指标一律采用默认 proxy（`train_interpolated`）口径，proxy 不得进入任何 SLO 判定路径。
- **proxy 填充**（`metrics_postprocess.py`）：exact（事件码 8）缺失且
  归档 `results/train_ledger.jsonl` 可得时填
  `first_token_proxy = decode_start_ns + (w₁/Σᵢwᵢ)×(first_train_end_ns − decode_start_ns)`，
  `wᵢ = W_bytes + KV_bytes(context+consumed+i)`，`first_token_source
  = train_interpolated`：
  - **Σ 域 = 首列车 iterations N**（i=1..N，与 S3/FACE/W/S1 裁定一致）：
    debut 首 token 随列车第 1 迭代完成，权重按列车 N 迭代线性化（对
    debut 自身参与度 P 求和会在 decode_length<iterations 时退化为
    share=1 并越过 completion）；debut join 时 consumed=0（sticky
    decode，60s 参考跑 1454/1454 恰一次 joiner 加入）。
  - **first_train_end = 同实例下一列车发射边界**：台账行 tick 是发射
    边界，实例 busy 门 = 一列车在飞（in_flight_train），故同实例下一行
    tick 即 end barrier 后首个决策边界；实例末列车 → NA（不可得不
    编造；S2 60s 重跑 0 例）。
  - **N=1/短列车钳制**：share 接近 1 → proxy=下一发射边界可能 >
    completion（边界晚于 end barrier tick），钳到 completion 并在
    instructions 留痕 `clamped_to_completion`（60s 重跑 5/1454，N=1
    ×3 + N=2 ×2）。填充值过 fail-closed ordering 检查
    （arrival≤first_token≤completion）。
  - **换算权威**：W_bytes/KV_bytes 用本仓只读依赖
    `face_scheduler.estimate_model_weight_bytes` /
    `kv_cache_bytes_for_tokens`（trace_config.csv 参数：swiglu →
    13476831232 / 524288 B/token，test_face_scheduler.py:404 冻结）。
    **sh_2.0 分层 KV**：`kv_cache_shard_bytes_for_layer_range` 按层
    区间切分同一 whole-head 总量（运行时验证 sum(shards)==
    kv_cache_bytes_for_tokens），层区间只改变 bytes 落在哪层、不改
    总量——per-token 权重换算与 S1 一致，不另行编造。
  - **运行内 CSV 保持 NA**（ledger 归档前不可得，instructions 记
    `proxy_unavailable:no_train_ledger(results/)`）；对归档 run 离线
    重跑 postprocess 填充——运行产物与离线分析两段语义。summary 行
    新增 `first_token_source_counts`。
  - **SLO 防线**：proxy 仅展示口径，`first_token_ns/first_token_source`
    被 `slo_common.assert_no_proxy_columns` 从一切判定路径 fail-closed
    拒绝（`slo_tools/tests/test_slo_contract.py::
    test_train_interpolated_proxy_cannot_enter_judgment`：夸张 proxy 值
    不改变 violation verdict，判定输入携带 proxy 列被拒）。
- **离线验证**（`/tmp/slo_wps/b4/S2/proxy60/`，源=B3 归档
  t3_full_off_split + 重物化 manifests（digest 2af3f07d 复现，计数
  1454/116 与运行时一致））：1454 行全部 train_interpolated、0 序违例
  （填充值过 fail-closed ordering 检查）、raw/normalized CSV 与归档
  逐字节相同（其余列零 diff，仅 first_token_ns/first_token_source/
  instructions 变化）；抽 3 请求手算（manifest+ledger+cpp.log 一次产物
  独立重算）全部吻合，含 1 例 N=1 钳制（qi_723 raw=492278469876 >
  completion → 491326310531）；proxy vs exact（t3_full_split_on）分布差
  p50 −13.6%/p99 +0.35%/mean −0.18%（信息性：两时间线已因门-2 级联
  分叉，属预期；S2 的 ON 时间线尾部漂移小于 S1，p99 几乎重合）。
- 2s 冒烟：缺省（不设 env）21 行 first_token 全 NA + source NA
  （运行内两段语义）；`SH_FIRST_TOKEN_SPLIT=1` 恢复 exact 21/21、
  0 序违例。
- 单测：`sh_test_mesh/tests/test_first_token_proxy.py`（新，9 用例）；
  slo_tools 契约 41/41（+proxy 判定防线）；metrics 契约 34 OK；
  拆分语义 52/52（online 全量）。

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
- `sh_test_mesh/tests/` + workload 根：pytest（拼 batch 改造后基线：
  93 passed + 6 skipped，无预存失败；新增 weight_passes / 列车机制 /
  列车发射钉子 / task-load 新折算专项。6 个 skipUnless(\_MATERIALIZED)
  守卫用例在物化输入在位且 trace_config 指向真实队列时自动恢复运行，
  其中 5 个转绿；`test_checked_in_astra_compute_selection` 的
  (3, 53924) 断言仍是 2026-08-18 sidecar 口径旧值（折入口径实际为
  (66, 169395)），属预存潜伏缺陷，与在线机制无关）

## 6. 边界与纪律

- 仿真输入唯一允许源 = astra_compute_20.csv 前 30 秒（更早的用户指示曾临时
  授权过更大窗口；以当下指示为准）。
- 缺失输入一律 fail-closed（generate 桩/materializer/runner/GEN_MATCH 均实测 exit=1）。
- 策略文件（face_scheduler.py（策略与 KV 语义同文件））为保留对象，勿改。
- 改动机制层后请跑 §4 fixtures + §2 ⑤ 对账再交付。

## 7. Online execution adaptation

在线策略路线（③/④）是实时宿主调度器：调度器进程内逐决策边界做映射决策，
把结果作为 per-rank 图批次发射给执行驱动引擎，决策不预先固化；每 tick 先
核销已完成列车（train_id + membership_digest 核验 → 冻结成员闭式推进 →
退出成员移出批 → 推进 prefill chunk），再处理 completion 批
（PREFILL_DRAIN / DECODE_COMPLETION / REQUEST_COMPLETE），再处理 arrival 批
（冻结队列序入 arrival heap），最后跑一次准入 pass 并为各空闲实例冻结发射
下一列车（`sh_test_mesh/workload/llama2_7b_inference/online/sh20_online_scheduler.py`）。

- 决策边界与仓内路由：ARRIVAL 落账 pending_admissions 后由准入 pass 重查
  （HBM 可行掩码过滤 + task-load 三分量快照选择 prefill 实例）；PREFILL_
  DRAIN 以精确 Roofline per-die 增量代价选择 decode 实例，drain 决策完成后
  成员进入 pending_decode_ready（迁移随加入列车发射）；DECODE_COMPLETION /
  REQUEST_COMPLETE 按 completion_order 完成 KV 收尾、completion 段发射与
  下一 turn 排程（active_decode 移除与 token 终值推进移至列车核销）。
  逐出转移的 history-gate 位置补偿为**决策时点同步补偿（唯一标记路径）**
  （2026-08-23 history-gate TOCTOU 修复 + 同日 seq4689 修订）：调度器在每个
  返回逐出转移的 KV 变更点（expand_prefill / move_request_capacity_reservation /
  move_prefill_to_decode / expand_decode / enforce_reserve /
  reserve_request_capacity / prepare_prefill）后立即经
  `GraphBatchBuilder.sync_pending_history_after_evictions` 把 remote_store
  逐出镜像到 pending 门（partial_hbm_remote / remote_memory 三态语义由
  `_mark_pending_history_store` 自身推导）。发射包装器内的既有发射时补偿
  已移除：多级逐出（suffix 半驻留 → full fallback）随列车**乱序发射**时，
  迟到的旧转移标记会把门回退到过期位置（120 档 seq4689 确定性事故：
  门 partial_hbm_remote vs 账本 remote_memory），"双保险"并不幂等。
- request_aggregated 折算口径：每个请求相位按 rank 折成 17 类算子节点
  （13 类层内算子 + attention/MLP 两个 All-Reduce + final norm + logits），
  聚合 FLOPs、tensor/HBM 字节、可选远端读与集合通信载荷与 token 展开总量
  恒等，只压缩重复层/chunk/token 与集合通信启动次数；权重字节按
  `weight_passes` 口径计（默认 = span 数，历史 batch=1 串行口径逐字节
  不变；迭代列车传迭代数——权重每物理前向只读一次，与批成员数无关，
  2026-08-22 拼 batch 改造，`generate_trace.py`）；在线发射仅支持该粒度，
  其它粒度 fail-closed（`online/graph_batch_builder.py`）。
- 发射并发骨架（本仓形态：迭代列车拼 batch，2026-08-22 改造）：连续
  batching——decode 互拼、decode 与 prefill chunk 混拼（每迭代 ≤1
  chunk）、chunk 之间不拼、批成员只在迭代（列车）边界变化。每实例
  状态机 = FCFS prefill 队列 / active_decode 批成员表 /
  pending_decode_ready（KV 就绪待加入）/ in_flight_train（唯一在飞
  列车 + train_id/membership_digest，busy 门 = 一个列车在飞）；列车
  终点 = 下一个不可预测事件（队列头 prefill drain / 全部工作耗尽），
  交付默认 T_max=8（2026-08-22 §7.4 A2 对拍裁决：无上限 TTFT -67.3%、16 仍 -19.1%、8 全指标 ≤1.3%——"固定为使位移 ≤5% 的最大值"，原则 1 优先于节点数；SH_TRAIN_MAX_ITER 可覆盖，0=不设限；截断且无自然标记的列车发哨兵标记承载完成信号）。发射 = 准入动作（到达 gates/历史迁移/逐出/屏障）→
  迭代列车（joiner 迁移 + 共享 readiness barrier + 折叠列车体
  （weight_passes=迭代数）+ drain/exit 标记节点 + 每列车一个共享 end
  barrier）→ 完成段（completion_evictions + 下一 turn interval
  gates）。列车账本核销幂等（同列车标记 watch 跨 tick fire、事件拆
  交付），推进量全部闭式；`train_ledger.jsonl` 落每列车审计行
  （§7.3 不变量断言输入，runner 归档）。
- sh_2.0 特性在列车形态下的保留（策略公式/阈值/KV 语义/映射规则不动）：
  三态 KV 与两阶段类型感知逐出的调用链逐行保留（drain 边界）；
  partial 前缀两段式迁移——admission 发射 prefix noc_migrate →
  prefix ready barrier → suffix remote_load 恢复 → suffix ready
  barrier（chain checkpoint/restore 保持恢复分支与主链并行），首
  chunk 所在列车按层段拆分发射（prefix 层段不等恢复，suffix 层段
  arm 依赖 suffix ready 节点；合成覆盖见
  `online/verify/train_a1_eviction_fixture.py`）；task-load 三分量
  中 active decode 分量按 decode_tokens_consumed 闭式迭代级剩余量
  折算（"整段不可分 fraction=1.0"旧口径作废；打分公式与阈值不动）。
- 接栅栏与段末屏障：发射不做块末恢复/段内清链，per-rank `previous_id`
  无条件接续当前 frontier（per-rank 发行序 = 全局发射序，2026-08-19 五仓
  统一），列车作为整体接续 frontier，跨请求 P2P 与集合通信参与序不会
  反转成环；列车以 TP 组共享 `*_end_barrier` 收尾，drain/exit 标记
  （列车体后、barrier 前）承载 PREFILL_DRAIN / DECODE_COMPLETION watch
  与指标锚点（"barrier 前末节点"口径的列车化延续）。
- 折入 recompute 输入口径 + 运行时 KV 账本：请求队列为唯一仿真输入，
  turn-0 源前缀折入该行 prefill_length（整段重算口径；无第二输入文件，
  会话 KV 自 turn-0 prefill 记账起由运行时账本动态维护，
  `traces/materialize_first_30s.py:3-31`）；manifest 仅携带队列派生的
  history_tokens_before 推导值（`plan_materializer.py:62-89`）；运行时账本为
  `KVCacheManager`（`sh20_online_scheduler.py:188`；`face_scheduler.py:1190`）：
  驻留命中直接复用、跨实例历史先迁移、被逐出则窗口内重算。
- long double 时间精度适配：大 ns 级首达偏移超出 IEEE-754 double 精确整数
  范围（2^53）时，分析网络适配层返回 ASTRA-sim 时间以 `long double` 保持
  64 位事件时间精确，避免完成回调与事件映射键错位 1 ns
  （`astra-sim/network_frontend/analytical/common/CommonNetworkApi.cc:71-81`）。
