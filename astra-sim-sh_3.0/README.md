# astra-sim-sh_3.0 - 晶圆级芯片（WSC）LLM 推理架构与机制

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
  本仓 KV 冷热分层为**三态**（`LOCAL_HBM` / `PARTIAL_HBM_REMOTE` 前缀层留本地、后缀层逐远端 / `REMOTE_MEMORY`）+ 两阶段类型感知逐出（2026-08-18 起：按 session 下一请求触发类型先 human 后 tool 分两类，类内先半逐后整逐，四段式 human半→human全→tool半→tool全）+ 流水化部分恢复 + 本地 HBM 多用户
  带宽争用（见 G 节，2026-08-19 起泛化为 N 用户严格均分并接入 comm/池端点计费）；本仓的分化点在请求→实例分配策略（见对照表）。

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
instance，每实例同时承担 prefill 与 decode）；分配上首请求避开边缘实例（即优先调度至非边缘实例）。

### F. 本仓在五仓中的定位

| 仓库 | 实例组织 | 请求→实例映射策略 | KV 驻留与恢复 | 远端内存池 |
|---|---|---|---|---|
| astra-sim-face | 统一实例（P+D 同实例） | FACE 原始映射：prefill 选剩余 chunk 最少；decode 在邻接图加权距离限制（阈值 = D2D 带宽 / 本地 HBM 带宽）内按 per-die Roofline 增量代价 | RESIDENT/EVICTED 两态；LRU 逐出＝零代价删除；恢复＝重算 | 未启用（NO_MEMORY_EXPANSION） |
| astra-sim-wscllm | PD 分离（Prefill-only + Decode-only 分区） | prefill 选排队请求最少；decode 用静态一跳 P→D 映射 | RESIDENT/EVICTED 两态；LRU 逐出；恢复＝重算（跨实例历史走 NoC 迁移） | 未启用（NO_MEMORY_EXPANSION） |
| astra-sim-sh_1.0 | 统一实例 | prefill HBM 可行过滤 + 剩余 chunk 最少；decode 按 per-die Roofline 增量代价 | LOCAL_HBM/REMOTE_MEMORY 两态；整 session 粒度逐出；恢复＝远端全量取回 | 启用（全部边缘芯粒挂端口） |
| astra-sim-sh_2.0 | 统一实例 | prefill Roofline 剩余负载均衡（历史 KV 全/部分驻留与全逐出统一）；decode 按 per-die Roofline 增量代价 + HBM 剩余 tie-break | 三态（含半驻留 PARTIAL）；两阶段类型感知逐出（human 类先于 tool 类）；流水化部分恢复 + HBM 恢复/推理带宽共享 | 启用（全部边缘芯粒挂端口） |
| **astra-sim-sh_3.0（本仓）** | 统一实例 | 三段式 prefill（首请求避边缘 / HBM 命中 sticky / 远端命中负载均衡）；decode 本地化固定同实例 | 三态；两阶段类型感知逐出（human 类先于 tool 类）；流水化部分恢复（机制同 sh_2.0） | 启用（全部边缘芯粒挂端口） |

五仓还共享同一套 Execution-Driven 在线仿真机制层（`astra-sim/workload/execution_driven/`：
RequestIngress / DecisionMailbox / DecisionBridge / GraphBatchCommitter 等）：策略决策由
Python 在线服务层实时给出，计时由 C++ 物理时钟推进；各仓仅保留路径③（strategy 关感知）
与路径④（strategy 开感知）两条在线路线。五仓在线决策均不再使用任何离线 LUT
（残留 LUT 查表已随静态链路一并删除）：face / sh_1.0 / sh_2.0 的 decode 候选代价由
在线 Roofline 模型即时计算（`per_die_delta_ns`），决策确定可复算复放。

### G. 本地 HBM 带宽模型（N 用户严格均分，2026-08-19 泛化）

每颗芯粒的本地 HBM 由单一标量带宽描述（硬件配置 `local-hbm.bandwidth-gbps`，
如 1640 GB/s）：**读写共享同一条总线、共享同一总带宽，不区分读写方向，也不区分
峰值/持续带宽**（"读写同价"）。`LocalHbmBandwidthModel`
（`astra-sim/workload/LocalHbmBandwidthModel.{hh,cc}`）是该带宽的流体竞争模型：

- **N 用户严格均分**：同一 rank 上所有 HBM 用户进入同一流体模型，任一时刻每个
  正在流字节的作业获得 `full_rate / N`；任一作业完成立即释放并把带宽重分给幸存者
  （事件驱动重分配，无预留、无浪费）。每作业启动时计一次 `local-mem-latency`
  （100 ns）启动时延（既有惯例）。作业类别：
  - `COMPUTE`：推理 COMP 节点（bytes = tensor_size；FLOPs 按峰值算力并行推进，
    保持 Roofline max(计算, 访存) 语义）。带非 0 校准 `runtime_ns` 的 COMP 在模型
    存在时仍由流体模型接管计时（`runtime_ns` 仅在模型不存在时使用——现状保持）；
  - `RESTORE`：KV restore DMA 写（串行图语义不变，HardwareResource 单飞槽保证）；
  - `COMM_READ` / `COMM_WRITE`：NoC p2p 通信数据端点——发送方 HBM 读、接收方
    HBM 写（bytes = comm 字节数）；
  - `POOL_READ` / `POOL_WRITE`：池流量端点（bytes = tensor_size）。
- **端点计费与直通规则**（每字节流恰好计一次）：
  - NoC 多跳路由器直通不占途经芯粒的 HBM（只有端点 rank 发 comm 节点）；
  - SerDes 与 NoC 路由器直连、与边缘本地 HBM 两两直连——边缘芯粒作 NoC↔SerDes
    直通时**不占其 HBM**，只有自己是数据端点才计费；
  - `remote_store` 源即边缘芯粒：边缘 `mem_store` 标注 `hbm-access-mode:1`
    （POOL_READ，节点完成 = 端口事务 ∧ HBM 作业的 join）；
  - `remote_store` 经 NoC：源 rank 的 `comm_send` 自动计 COMM_READ；边缘 rank 的
    `comm_recv` 标注 `hbm-charge:false`（直通），其 `mem_store` 不标注（字节离片）；
  - `remote_load` 目标即边缘芯粒：restore 节点计费（图当前对该情形也发 restore
    节点），`mem_load` 不标（SerDes→NoC 直通）；
  - `remote_load` 经 NoC：边缘 `mem_load` 不标（直通）、边缘 `comm_send` 标
    `hbm-charge:false`；**目标的 `comm_recv` 也标 `hbm-charge:false`**——目标 HBM
    写由后续串行 restore 节点承担，避免双计；
  - `noc_migrate` 与 history_kv/prefill_to_decode 等两端真端点路径：自动计费，不标注。
- **comm / MEM 节点完成语义**：被计费的 p2p comm 节点完成 = join(网络侧完成回调,
  本端 HBM 作业完成)（`Workload` 内两 pending 标志 + 幂等保护实现，不改网络 API
  签名）；被计费的 MEM 节点完成 = join(端口事务, HBM 作业)。`hbm-charge` 缺省
  true；`hbm-access-mode` 0/缺省 = 无。
- **配置开关**：system.json 新键 `"hbm-bandwidth-contention"`（bool，代码默认
  true）：true = N-way 模型接管一切（COMP/restore/comm/池端点）；false = 旧行为
  完全保留（旧键 `hbm-kv-restore-bandwidth-sharing` 的两用户 50/50 语义保留用于
  A/B）。`local-mem-bw <= 0` 自动视为 false。
- **不在范围**：片外池自身的竞争（端口 FIFO 模型不动）；TP 集合通信
  （PacketBundle 3x 路径）不计 HBM 端点。
- **指标**：`local_hbm` 记录（MetricCollector）导出 per-rank busy/shared ns、按类别
  served bytes（comp/comm_read/comm_write/pool_read/pool_write/restore）、峰值并发
  作业数、均分重分配事件数（旧键保持）。


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

- **策略语义**：三段式准入：首请求避边缘 / LOCAL·PARTIAL sticky / REMOTE 边缘实例负载均衡（2026-08-18 起由全集收窄为边缘实例集合）；decode 固定 prefill 同实例；三态 KV；turn-0 前缀折入 recompute（KV 账本动态记账）
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
#    arrival_time < 30e9 ns；物化器：materialize_20_30s.py（产折入 queue +
#    canonical digest 两件，turn-0 前缀已折入 prefill_length））
#    产物放 sh_test_mesh/workload/llama2_7b_inference/traces/，
#    并把 trace_config.csv 第 12 行 request_queue_csv 指向它

# ③ 生成 plan 目录（runtime_config 四小件 + manifest + metrics_manifest）
cd sh_test_mesh/workload/llama2_7b_inference && python3 plan_materializer.py && cd <仓根>

# ④ 跑③④（GEN_MATCH：generated/ 下须恰一个 llama2_7b_inference_54npus_* 目录）
bash sh_test_mesh/run_scripts/run_online_strategy.sh <run_dir> <绝对路径 request_csv>
bash sh_test_mesh/run_scripts/run_online_strategy_sensing.sh <run_dir> <绝对路径 request_csv>

# ⑤ 指标后处理 + ④对账
bash sh_test_mesh/run_scripts/run_metrics_postprocess.sh <run_dir>/cpp.log
python3 sh_test_mesh/workload/llama2_7b_inference/online/verify/sh30_ledger_reconcile.py --run-dir <run_dir> --manifest <plan目录>/manifest.json（请求数期望由 manifest 推导）

# ⑥ 一键清空测试记录（还原裸仓）
bash sh_test_mesh/run_scripts/clean_test_records.sh [--full]
```

## 3. 两条仿真路线

| 路线 | runner | 时钟 | 产物 |
|---|---|---|---|
| ③ strategy 关感知 | run_online_strategy.sh | 真实物理 | 决策日志/digests/metrics |
| ④ strategy 开感知 | run_online_strategy_sensing.sh | 真实物理 | ③产物 + ledger.jsonl/感知日志（对账用） |

PASS 判据：completed == 物化请求数、no_decision=0、single_node=0、
delivery == graph_batch 数、③④ 决策日志逐字节一致（感知只开仪表不改判据）。

## 4. 机制回归 fixtures

`run_online_idle_fixture.sh`（IDLE 五态生命周期）、`run_online_wakeup_guard_fixture.sh`、
`run_online_same_tick_milestone.sh`、`bridge_race_stress_repro.sh`——机制层健康自检。

## 5. 目录导览（关键路径）

- `astra-sim/workload/execution_driven/`：在线机制层（C++）
- `sh_test_mesh/workload/llama2_7b_inference/online/`：在线调度器/构图器/服务层（Python）
- `.../online/verify/`：对账与验证工具
- `sh_test_mesh/run_scripts/`：全部 runner 脚本
- `sh_test_mesh/workload/llama2_7b_inference/traces/`：物化器脚本（数据件由调用方物化，provenance 以物化器 stdout 为准）
- `sh_test_mesh/tests/` + workload 根：pytest（基线：37+33 = 70 passed，无预存失败）

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
再处理 arrival 批，最后跑一次准入/发射 pass
（`sh_test_mesh/workload/llama2_7b_inference/online/sh30_online_scheduler.py:405-429`）。

- 决策边界与仓内路由：ARRIVAL 落账 pending_admissions 后由准入 pass 重查
  （HBM 可行掩码 + task-load 三分量快照 + edge_free 掩码选择 prefill 实例；
  `:455-462`、`:632-680`）；PREFILL_DRAIN 将
  decode 固定为 prefill 同实例（selected = state_index，`:486-488`）；
  DECODE_COMPLETION / REQUEST_COMPLETE 完成 KV 收尾（typed 两类两段逐出）与
  下一 turn 排程（`:524`、`:542`）。
- request_aggregated 折算口径：每个请求相位按 rank 折成 17 类算子节点
  （13 类层内算子 + attention/MLP 两个 All-Reduce + final norm + logits），
  聚合 FLOPs、tensor/HBM 字节、可选远端读与集合通信载荷与 token 展开总量
  恒等，只压缩重复层/chunk/token 与集合通信启动次数
  （`sh_test_mesh/workload/llama2_7b_inference/generate_trace.py:1089-1115`）；
  在线发射仅支持该粒度，其它粒度 fail-closed
  （`online/graph_batch_builder.py:608-611`）。
- 发射并发骨架（本仓形态：批齐发-等完成）：就绪实例一次发射整批——qp 队首
  prefill 整段 + 全部 active_decode 成员各自的 decode 整段，随后置 busy 等
  待物理完成（`sh30_online_scheduler.py:776-793`）；busy 在 PREFILL_DRAIN
  （`:470`）/ DECODE_COMPLETION（`:533`）边界复位，复位后仍有排队工作的
  实例重入就绪 frontier 等下一批；PREFILL_DRAIN 边界新入 active_decode 的
  decode 段即时发射（`:521-522`）。
- 接栅栏与段末屏障：段发射不做块末恢复/段内清链，per-rank `previous_id`
  无条件接续当前 frontier（per-rank 发行序 = 全局发射序，2026-08-19 五仓
  统一；`online/graph_batch_builder.py:717-719`），跨请求 P2P 与集合通信
  参与序不会反转成环；prefill 段以 TP 组
  `*_prefill_chunks_aggregated_end_barrier` 收尾（`:670`），decode 段以 TP 组
  `*_decode_request_end_barrier` 收尾（`:779`）。
- 折入 recompute 输入口径 + 运行时 KV 账本：请求队列为唯一仿真输入，
  turn-0 源前缀折入该行 prefill_length（整段重算口径；无第二输入文件，
  会话 KV 自 turn-0 prefill 记账起由运行时账本动态维护，
  `traces/materialize_20_30s.py:3-37`）；manifest 仅携带队列派生的
  history_tokens_before 推导值（`plan_materializer.py:62-87`）；运行时账本为
  `KVCacheManager`（`sh30_online_scheduler.py:346`；`face_scheduler.py:1226`）：
  驻留命中直接复用、跨实例历史先迁移、被逐出则窗口内重算。
- long double 时间精度适配：大 ns 级首达偏移超出 IEEE-754 double 精确整数
  范围（2^53）时，分析网络适配层返回 ASTRA-sim 时间以 `long double` 保持
  64 位事件时间精确，避免完成回调与事件映射键错位 1 ns
  （`astra-sim/network_frontend/analytical/common/CommonNetworkApi.cc:71-81`）。
