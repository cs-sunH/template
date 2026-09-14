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

**C1 typed 响应解析（2026-08-29，sh_1.0 先行，本仓已随 W3 同步）**：桥响应在
`FileDecisionBridge::deliver_and_receive` 内经 `parse_graph_batch`
（`ParsedGraphBatch.hh/.cc`）**一次性**解析为 typed 批（`GraphBatch` 即其别名），
validate / 强制 liveness preflight / commit 装配 / 锚点注册四处消费同一份
typed 数据——改前"DOM 随批驻留 + 同一棵 nlohmann DOM 被字段级提取 4 遍"的
重复解码消除。**协议收紧（fail-closed 强化，fixture 作者可见）**：顶层未知键、
节点/边/watch/alarm 键集不符（缺失或多余）、整型域违例（负数、>UINT64 字面量、
tag/priority 超 32 位）在解析层直接经 `bridge_fatal` abort（与既有协议违例同
通道同退出码）；错误响应仍先行于结构解析判定（时序不变）。本仓 schema 特例：
节点键集含可选 `mem` 子对象（MemAttrs：tensor_size / is_local_hbm_kv_restore /
hbm_access_mode，逐字段可缺省=旧 value() 缺省语义）、stage 域为三段式
prefill/decode/completion（仅 prefill 为 generation 0）。Python 生产者
（graph_batch_builder）本就满足全部键集，行为不变；三个 verify fixture 服务
（same_tick_milestone / wakeup_guard / lifecycle）原先发射的空 `comm:{}`/`coll:{}`
已补满形缺省。数组顺序契约：六个数组与 watch 成员一律按发射序保序，解析层
不排序不去重。

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

### H. KV 逐出与推理计算的并行执行（2026-09-13，仅 Python 发射层）

KV 逐出的物理传输（`remote_store` 链）与 request 推理计算**并行执行**；C++ 侧零改动，
HBM 带宽仲裁完全由 §G 的 N-way 均分模型在线裁决（逐出流与 COMPUTE 流同 rank 同时
在场时自动 `full_rate / N`，流加入/离开即时重分配——`redistribution_events` 可观测）。

- **旁路支链执行语义**（`WL/online/graph_batch_builder.py` `_emit_side_branch`）：
  三类逐出发射循环（准入 history 逐出 / 准入 prefill 逐出 / joiner decode 逐出）经
  fork 包裹搬离 per-rank 主链——fork 自各 rank 当前 frontier，`chain_checkpoint →
  分支内发射 → restore_chain` 复用既有 PARTIAL 后缀恢复的段内分支机制。触发门
  （到达/间隔 timer gate、drain 列车 barrier）仍在分支内构建并挂**分支首节点**：
  逐出的**开始**时刻语义不变；去掉的是"逐出**完成**（含 1B ack 回环）阻塞其后一切
  计算"的链序串行化。分支不 join、不挂 watch、不发明独立逐出批（逐出节点仍随既有
  准入/列车批发射）。恢复类迁移（`prefill_decode_transfer`、history 回迁、
  noc_migrate）保持主链不动。fork 前既有主链 armed 依赖（turn-0 arrival gate /
  local_hit arm）由 helper stash-and-clear 保真留给主链原消费者；分支内 arming 未被
  消费则 fail-closed。
- **store→restore 前递依赖补偿**（唯一必须新增的正确性边）：支链化后"同会话逐出
  池写先于其下一轮池读"的主链传递性保障失效。构图器维护
  `pending_store_tails` 登记表（session → 在飞 remote_store 支链的边缘 `mem_store`
  尾部，两段式逐出的 suffix 与 full 两笔逐 shard 各登记一条，restore 读全区间须等
  齐），回迁发射统一入口（REMOTE 全量 `remote_load` 段 + PARTIAL 后缀恢复支链入
  口）查表补边：同边缘 rank 直接 `arm_dependency`（同 rank data_dep 跨批次合法）；
  跨边缘 rank（仅 REMOTE 全量跨实例回迁可达）复用 1B p2p 中继模式
  （`..._store_sidelink_s<store>_r<restore>` send/recv 对，桥协议内跨 rank 依赖的
  唯一合法载体）。粒度取 store 的边缘 `mem_store` 完成；store 早已完成时补边即刻
  满足（懒处理）。回迁消费即清；会话终结（`_emit_completion` terminal 分支）清
  登记。`_emit_kv_transfer` 仅扩展返回值暴露尾部节点 id（发射节点/边逐字节不变）。
- **瞬态双占用窗口口径**：并行窗口内新 request 的 KV 正在写入本地 HBM 而被逐出的
  旧 KV 尚未物理离开——这是时间模型的既定建模近似（容量权威在 Python 台账、决策
  时刻记账；C++ 无容量强制，`capacity_violations` 回放的是决策时刻数据）。**窗口
  上界 = 该 rank 在飞逐出字节数**；引用窗口期数值时不得把台账值当物理占用使用。
  若未来要求物理严格，正确修法是 ack 锚定延后释放（独立改造，不在本仓）。
- **测试**：`WL/online/test_eviction_side_branch_structure.py`（fork/支链/门/stash
  结构断言）、`WL/online/test_store_restore_ordering.py`（同缘/跨缘/两段式/PARTIAL
  排序断言）、`WL/online/verify/train_a1_eviction_fixture.py` 场景 2（逐出后立刻
  再到达的端到端 store→restore 覆盖）。


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
#    arrival_time < 30e9 ns；物化器：materialize_20_30s.py（产折入 queue +
#    canonical digest 两件，turn-0 前缀已折入 prefill_length））
#    产物放 sh_test_mesh/workload/llama2_7b_inference/traces/，
#    并把 trace_config.csv 第 12 行 request_queue_csv 指向它
#    B1/WP2 起 CLI：materialize_20_30s.py [source] [queue_out] [window_ns] [arrival_scale]
#    （全可选；默认=30e9/1.0 原生行为逐字节不变；window_ns 换窗口界；
#    arrival_scale 仅把 turn-0 session_arrival_time_ns ÷scale——
#    inter_request_interval_ns 与窗口筛选不动，同一请求集合更密/更疏到达；
#    脚本内部 flock /tmp/slo_wps/locks/heavy_mat.lock 串行重载入）

# ③ 生成 plan 目录（runtime_config 四小件 + manifest + metrics_manifest）
cd sh_test_mesh/workload/llama2_7b_inference && python3 plan_materializer.py && cd <仓根>

# ④ 跑③④（GEN_MATCH：generated/ 下须恰一个 llama2_7b_inference_54npus_* 目录）
bash sh_test_mesh/run_scripts/run_online_strategy.sh <run_dir> <绝对路径 request_csv>
bash sh_test_mesh/run_scripts/run_online_strategy_sensing.sh <run_dir> <绝对路径 request_csv>

# ⑤ 指标后处理 + ④对账
#    指标明细档（B1/WP0 起，两 runner 同规则）：env SH_METRICS_DETAIL=
#    off|summary|full 覆盖；缺省读 workload/llama2_7b_inference/
#    metrics_config.json 的 detail_level（优先级 env > json；非法值
#    fail-closed）。full 档后处理额外产 <run_dir>/request_metrics.csv
#    （逐请求时序：冻结 29 列；填 timing/terminal_status/
#    request_type/instructions，first_token 族由 B3/B4 填充（exact/
#    train_interpolated/NA 两段语义，见下），kv_hit_state 与 restore/
#    hidden 族运行内 CSV 恒 NA（slo_tools 离线适配器另行核算）；
#    manifest.json 侧车按 queue_index/request_id fail-closed
#    连接，行数/不变量违例即报错退出）。summary/off 不产该文件
#    （postprocess.log 有说明）。
#    B2 观测层新增（SLO 指标改造，只读 side-band，不改变仿真行为）：
#    - plan_materializer 向 metrics_manifest.json 注入 "slo_sampling" 节
#      （watermark_period_ns / link_bucket_ns / provisional；值取
#      sh_test_mesh/slo_tools/slo_params_manifest.json 的对应 value（B4
#      已填推导值 5,000,000/20,000,000 ns），为 null/缺失时用文档化临时
#      锚点 5,000,000 ns 并置 provisional=true；
#      manifest 缺节时 C++ 侧同锚点并落 slo_sampling_warning 记录）。
#      每条水位线/链路记录回显实际生效的周期与 provisional 标志。
#    - full 档新增 [METRIC] 记录：hbm_watermark / hbm_watermark_summary
#      （WP8 水位线：finalize 时对 planner memory ledger 的 delta 重放做
#      事件驱动+周期兜底分桶峰值；桶记录仅在被采样占用相对前桶变化时
#      发射、首末桶恒发；capacity_violation_count 应恒 0，
#      resident_timeavg 与 capacity_timeavg 积分交叉差 >1% 计
#      consistency violation），请求记录新增 first_token_ns 字段
#      （WP9 事件码 8；python 侧首 token 标记未上线时为 null+说明；
#      decode_length==1 的相等不变量按 WP9_CONTRACT §6 2026-08-27 裁决放宽
#      为序检查 + 信息性 first_token_completion_skew_ns 字段：code 8 取
#      min、completion 取 max，TP 斜台使二者天然不等，不再 withhold、
#      不计 violation）。
#    - summary/full 档新增 link_bucket / link_total 记录（WP6 NoC 链路
#      观测：FluidScheduler 只读积分，按 link_bucket_ns 分桶累计每链路
#      字节与活跃时长；桶记录仅有活动（任一链路 bytes>0）时发射、首末
#      桶恒发；metrics=off 时零工作零输出）。env
#      ASTRA_LINK_OBSERVER=0 可单独关闭（缺省开启，仅 metrics≠off 生效）。
#    - WP9 首 token 事件码 8（python+B2cpp 联动，2026-08-26）：
#      metrics_schema.py 新增 EVENT_FIRST_TOKEN_COMPLETE=8（complete 边，
#      入 SERVICE_EVENT_CODES）；graph_batch_builder 的迭代列车在含
#      debut 成员（decode_tokens_consumed==0 的本交付 joiner）且迭代数
#      >=2 时拆"首步批+余量批"两段式发射——首步批 = joiner 迁移/起始
#      标记/partial 恢复门 + 各成员第 1 个 span + prefill 队头第 1 chunk
#      （weight_passes=1）+ 每 debut 的 first_token 标记（1-op COMP，名含
#      "first_token" 子串，C++ 锚点按名字子串注册 code 8、每 rank 取
#      min tick）+ 批命名空间唤醒标记（"<train_id>_first_step"，哨兵同
#      款 PREFILL_DRAIN 单事件通道，仅作余量批交付边界——无 watch 的
#      首步批在排空尾部会失去交付边界而死锁，为对"首步批无 watch"的
#      必要工程化偏移，与 sh_1.0/sh_2.0 同构）；余量批 = 剩余 span
#      （weight_passes=iterations-1，与首步 1 次合计守恒）+ 全部
#      drain/exit/哨兵标记 + end barrier（挂点语义不变）。唤醒信号的
#      交付回声在调度器侧无操作（不决策/不记账/不写 decision_log）。
#      decode_length=1 的 debut 无独立标记，其 exit 标记改名携带
#      first_token 子串（同节点 code 4/8 双锚点）。拆分新增批产物带
#      "first_step": true 标记（digests/train_ledger 行）供 ON/OFF
#      对拍剥离。总开关 env SH_FIRST_TOKEN_SPLIT（**B4 起缺省 "0" 关**；
#      显式 "1" 开，研究/对拍用——2026-08-26 B3_S3 60s 决策等价门-2
#      失败：拆分物理扰动经闭环放大（首列 6.5us tick 漂移 → 逐轮放大
#      至 414us → 决策边界穿越、实例选择翻转、tput +13.2%），按主规格
#      §1.6 A 类处置默认退回 proxy 口径；证据 /tmp/slo_wps/b3/S3/
#      t3_full_off_split 与 /tmp/slo_wps/gates/B3_S3.FAILED，2s 门保持）。
#      **研究专用**：`SH_FIRST_TOKEN_SPLIT=1` 的 exact 口径仅供机制研究（60s 决策等价门未过，证据见仓内 README 所引门文件）；论文指标一律采用默认 proxy（`train_interpolated`）口径，proxy 不得进入任何 SLO 判定路径。
#      postprocess 起 first_token_ns/first_token_source 入
#      request_metrics.csv：exact（code-8 事件值，拆分开时）；
#      **train_interpolated（B4 退路 proxy，主规格 §1.6/§1.2：对已归档
#      run（results/train_ledger.jsonl 在场）重跑 metrics_postprocess 时，
#      `first_token = decode_start + (w1/Σwi)·(first_train_end −
#      decode_start)`，wi = W_bytes + KV_bytes(context+consumed+i)（与
#      调度器 pass-span 编码同式，i=1..N 按首列车 iterations——首 token
#      随列车第 1 迭代产出；debut consumed=0；first_train_end = 同实例
#      下一列车发射边界 tick（busy 门保证其晚于 end barrier；首列车为
#      实例末列车时不可得 → NA）；N=1 退化列车 share=1 使 proxy 越过
#      completion 时钳到 completion 并留痕 clamped_to_completion——
#      首物理上 token 不得晚于请求完成，60s 参考跑 7/1454；
#      W_bytes/KV 换算取 face_scheduler.py estimate_model_weight_bytes /
#      kv_cache_bytes_for_tokens × trace_config.csv 模型行——llama2_7b:
#      W=13476831232B、KV/token=524288B）；运行内（runner 自带那遍）
#      postprocess 时 ledger 尚在 bridge/ 未归档，split 关时 first_token
#      保持 NA——proxy 只由对归档产物的离线重跑填充**；NA。proxy 仅为
#      展示口径，禁入 SLO 判定（slo_common assert_no_proxy_columns 拒绝
#      first_token_ns/first_token_source，violation 判定列集自证，
#      slo_tools/tests 有专项单测）。TTFT 展示 = first_token − arrival，
#      禁用 prefill_end − arrival（红线）。
#      在线 decision_log 的 KV 传输摘要补落 shard 级 noc_hops/noc_path
#      （hopbytes 观测，只加字段不改路由）；sh_test_mesh/slo_tools/
#      hopbytes.py B4 起消费该字段（completion_evictions[].shards[].*，
#      count/fallback 语义同 sh_1.0：显式 noc_hops 优先、noc_path 推导
#      兜底；无路由的聚合 bytes 字段不计入本账——60s 参考跑实测
#      coverage 1.0、hop_bytes 2383397232640 == B3 手工复核值）。
#      metrics_manifest.json（合成口径）B4 起每请求附加 human_time_ns/
#      tool_time_ns/request_type 透传（与 manifest.json 同源同值；
#      slo_stats session 的 T_session 输入，附加键不删不改既有键；
#      turn-0/0-gap 双侧 null = 0 贡献而非缺数据，键缺席才 NA——
#      S3 异常③处置）。
bash sh_test_mesh/run_scripts/run_metrics_postprocess.sh <run_dir>/cpp.log
python3 sh_test_mesh/workload/llama2_7b_inference/online/verify/sh30_ledger_reconcile.py --run-dir <run_dir> --manifest <plan目录>/manifest.json（请求数期望由 manifest 推导）

# ⑤b 自动 SLO 指标提取（P3，2026-08-28；A4/2026-08-29 单遍化）：在线
#    runner（strategy/sensing 两变体）在 postprocess 成功后、归档前自动
#    调用 run_slo_postprocess.sh——内部一次调起 slo_tools/
#    slo_postprocess_driver.py 单进程单遍驱动（9 步产物集/行序/公式/
#    slo_postprocess.log 与逐工具串行逐字节一致；读放大收敛：
#    request_metrics.csv 4 读→1、decision log 4 流→1、[METRIC] init 探测
#    4→1、解释器启动 9→1；七个工具 CLI 保持可独立调用）
#    ——full 档产 9 项最细粒度产物（slo_e2e_stats/backlog/session/
#    load_imbalance/restore_decomposition/hopbytes_total+per_request/
#    hbm_watermark 三件=hbm_intervals（权威 RLE）+hbm_plot_series（行预算绘图，旧 hbm_watermark_series 已退役）+hbm_watermark_instances、cache_events+kv_hit_states、
#    slo_warmup.json；P1/2026-08-30 起 hbm_watermark 按四层可信度分级：run_dir 含
#    results/kv_delta_journal.jsonl 时走 journal 权威重放，含 checksum 证书为
#    per_rank_total_hbm_certified 层——正式逐 rank 容量判决、违规 exit 3；缺
#    journal 的旧 run 为 upper_bound_only 上界层，超限只诊断不认证、exit 0），
#    一律不传分桶/聚合参数（粗化留给下游画图脚本）；
#    非 full 档依赖 request_metrics.csv 的子命令按设计跳过并写说明；
#    legacy 变体（face/wscllm）额外跳过 load_imbalance/hbm_watermark
#    （legacy 无列车台账；分配器语义不兼容 session-KV 水位重建），
#    均写说明、不算失败。
#    env SH_SLO_POSTPROCESS：1=默认 warn（子命令失败只写
#    slo_postprocess.FAIL 标记，不推翻仿真结果）、0=整步跳过、
#    strict=失败即 runner 非零退出。runner 同时把 per-request manifest
#    （metrics_manifest.json/manifest.json）拷入 run_dir 根（P2）——run_dir
#    自包含，与仓还原状态解耦
# ⑥ 一键清空测试记录 + 编译产物（还原裸仓库 = 无物化输入 + 无 build/）
bash sh_test_mesh/run_scripts/clean_test_records.sh      # 清物化输入/运行产物/缓存（含 trace_config[_legacy] 指针回占位）
bash sh_test_mesh/run_scripts/clean_build_artifacts.sh   # 清编译产物（build/）
#    （或一步到位：clean_test_records.sh --full）
```

## 3. 两条仿真路线

| 路线 | runner | 时钟 | 产物 |
|---|---|---|---|
| ③ strategy 关感知 | run_online_strategy.sh | 真实物理 | 决策日志/metrics/train_ledger（digests 默认关，见 §3.1.1；成功后自动瘦身归档） |
| ④ strategy 开感知 | run_online_strategy_sensing.sh | 真实物理 | ③产物 + ledger.jsonl/感知日志（对账用） |

PASS 判据：completed == 物化请求数、no_decision=0、single_node=0、
delivery == graph_batch 数、③④ 决策日志逐字节一致（感知只开仪表不改判据）。

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

runner 脚本（strategy/sensing 两变体）显式传
`--online-validate "${SH_ONLINE_VALIDATE:-0}"`；fixture runner（idle/
wakeup_guard/same_tick_milestone）显式 `--online-validate 1`。运行期证据：
cpp.log 启动行 `[online] node gc: ...` / `[online] graph validate: ...`、
结束行 `[online] node gc: erased=... retained=...`。

停泊与输入契约 fail-loud（P0-2，2026-08-31，非 `--online-` 前缀、同属
OnlineCli 在线家族解析；2026-09-01 sync-A16 自 face 母本批次P 同步）：

- **calendar reader（P0 turn-0 late-discovery fix，2026-08-30，对齐 wscllm）**：
  CSV 队列经一次 64 KiB 分块流式索引遍（fail-closed 结构校验：session 块
  连续、块首 turn-0、块内 turn 严格 +1、turn>0 行 arrival 必空）+ 溯源边车
  门（`<queue>.provenance.json` 存在即逐字段比对，任一不匹配零 Submit 前
  `[Error]` 退出）+ turn-0 到达日历（按 `(arrival, queue_index)` 稳定排序
  提交）读取——turn-0 提交序与文件位置彻底解耦，非单调输入下
  `late_static_submit=0`，run-end 输出 arrival audit 行并以
  `late_static_submit==0 且逐行 ingress_delay==0（t=0 边界行除外）` 为正式
  门禁（fail 则非零退出）。
- **window advisory 旋钮已删除（2026-09-05 A.3 清除；原四仓对齐 wscllm
  2026-08-30 P0 fix 延续裁决）**：`--request-window-rows` 死旋钮已从四仓
  物理删除——calendar reader 按 arrival 序提交，窗口值从不约束读取与提交，
  删除不改变任何行为；现传入该选项即 `unknown online-family option`
  硬错误（fail-closed），runner 不再透传 `SH_REQUEST_WINDOW_ROWS`。plan_materializer 的
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
  ERANGE 拒 → endptr 到串尾 → 目标类型上限（前两者 size_t/uint64_t、
  bridge-timeout-ms 另加 `<= INT_MAX`）→ 才转换"合同，失败不部分写入
  `out`。`--online-validate` 原同属该整数词法族
  （`--online-validate=4294967296` 曾会回绕为 0 静默关闭图验证，fail-open）；
  B.2 清除后改为严格 `0|1` 枚举解析，任何其它值（含抽检 N≥2 与越界大数）
  启动即硬错误。
- runner 透传 env：`BRIDGE_TIMEOUT_MS` 三态——未设=缺省 120000（须大于
  负载最慢单决策与 Python 侧 FIFO 开启等待）、显式 `0`=永等逃生口、正值=
  该毫秒值；它只武装 C++ 桥 response poll（Python 单次交换停滞族），管不到
  停泊族（由上面看门狗兜住）。
  sensing 变体 runner 与主 runner 共享同一 CSV reader，同款参数口径。

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
  字节一致）、请求完成后 runtime 索引 pop（B4）、
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
无参数直跑；2026-08-29 新增三项，五仓同构）：`..._AlarmCancellationTest`（可取消 alarm
链路：bucket 清空时 outer alarm 从 backend 物理移除、共享 bucket 级联、重复取消幂等、
legacy 后端回退 stale guard）、`..._MetricOneShotEraseTest`（MetricCollector one-shot
node bucket 擦除 + OnlineNode anchor 快路径标志，双运行 [METRIC] 输出逐字节对拍、
sizeof 编译期锁定）、`..._RemoteFifoLedgerTest`（RemoteFifoLedger 按 backend 真实端口
记账；自带 PER_NPU/PER_NODE/MEMORY_POOL 三架构 fixture 自证——本仓无 sensing 记账
接线，账本不启用）等。Python 侧 `online/test_propagating_tail.py`
（`online_scheduler_base.py` 的在途尾部观测器 PropagatingTailTracker：对到达未完成
请求、未 ack 交付、未确认 provisional KV 动作三类在途工作记 current/peak/按来源计数，
超限 fail-closed 报错、绝不截断；8 用例，pytest 或直跑）。

## 5. 目录导览（关键路径）

- `astra-sim/workload/execution_driven/`：在线机制层（C++）
- `sh_test_mesh/workload/llama2_7b_inference/online/`：在线调度器/构图器/服务层（Python）
- `.../online/verify/`：对账与验证工具
- `sh_test_mesh/run_scripts/`：全部 runner 脚本
- `sh_test_mesh/workload/llama2_7b_inference/traces/`：物化器脚本（数据件由调用方物化，provenance 以物化器 stdout 为准）
- `sh_test_mesh/slo_tools/`：SLO 离线后处理工具集（slo_stats / load_imbalance / restore_decomposition / kv_cache_adapter / hopbytes + `slo_postprocess_driver.py`（A4 单遍合并驱动，run_slo_postprocess.sh 链内使用；工具 CLI 不变）+ `slo_params_manifest.json`（B 类参数唯一来源，B4 已填推导值）+ tests；纯离线只读，详见目录内 README.md）
- `sh_test_mesh/tests/` + workload 根：pytest（基线：2026-08-22 拼 batch 改造后
  97+1 passed——`test_face_scheduler.py` 的 request-neutral 占位测试在正典
  物化件在位而 trace_config 指向占位路径时红为已知环境效应，与代码态无关）

## 6. 边界与纪律

- 仿真输入唯一允许源 = astra_compute_20.csv 前 2 秒（更早的用户指示曾临时
  授权过更大窗口；以当下指示为准）。
- 缺失输入一律 fail-closed（generate 桩/materializer/runner/GEN_MATCH 均实测 exit=1）。
- 策略文件（face_scheduler.py（策略与 KV 语义同文件））为保留对象，勿改。
- 改动机制层后请跑 §4 fixtures + §2 ⑤ 对账再交付。

## 7. Online execution adaptation

在线策略路线（③/④）是实时宿主调度器：调度器进程内逐决策边界做映射决策，
把结果作为 per-rank 图批次发射给执行驱动引擎，决策不预先固化；每 tick 先
核销已完成列车（拼 batch 列车账本），再处理 completion 批（PREFILL_DRAIN /
DECODE_COMPLETION / REQUEST_COMPLETE），再处理 arrival 批，最后跑一次
准入/列车发射 pass
（`sh_test_mesh/workload/llama2_7b_inference/online/sh30_online_scheduler.py`）。

- 决策边界与仓内路由：ARRIVAL 落账 pending_admissions 后由准入 pass 重查
  （HBM 可行掩码 + task-load 三分量快照 + edge_free 掩码选择 prefill 实例；
  三段式 sticky 准入（首请求非边缘 / LOCAL-PARTIAL 驻留 sticky / REMOTE
  边缘负载均衡）判据与亲和规则一行不动，红线）；PREFILL_DRAIN 将
  decode 固定为 prefill 同实例（selected = state_index，红线 #4）；
  DECODE_COMPLETION / REQUEST_COMPLETE 完成 KV 收尾（typed 两类两段逐出）与
  下一 turn 排程。
- 逐出转移的 history-gate 位置补偿为**决策时点同步补偿（唯一标记路径）**
  （2026-08-23 history-gate TOCTOU 修复，与 sh_1.0/sh_2.0 同构 + 同日
  seq4689 修订）：账本逐出在决策时点同步翻转会话位置，调度器在每个
  返回逐出转移的 KV 变更点（`sh30_online_scheduler.py` 改法D 1/9~4/9
  drain 四点 + 7/9 enforce_reserve + 8/9 reserve_request_capacity +
  9/9 prepare_prefill，共 7 处）的 bump 行后立即调用
  `GraphBatchBuilder.sync_pending_history_after_evictions(transfers)`
  （吃 `KVTransfer` 对象，`remote_store` 过滤），把逐出会话立即镜像到
  pending 门（`partial_hbm_remote` / `remote_memory` 三态语义由
  `_mark_pending_history_store` 自身推导）；三处发射包装器
  （`online/graph_batch_builder.py` 准入/列车 joiner/完成段）内的发射时
  补偿已移除——多级逐出发射乱序时迟到的旧转移标记会把门回退到过期
  位置（见 sh_2.0 seq4689 事故），"双保险"并不幂等；准入时位置一致性
  检查（fail-closed）不变。
- request_aggregated 折算口径：每 rank 折成 17 类算子节点
  （13 类层内算子 + attention/MLP 两个 AllReduce + final norm + logits），
  聚合 FLOPs、激活/KV/HBM 字节、可选远端读与集合通信载荷与成员-迭代展开
  总量恒等，只压缩重复层/chunk/token 与集合通信启动次数；权重字节按
  `weight_passes` 口径计（默认 = span 数，历史 batch=1 串行口径逐字节
  不变；迭代列车传迭代数——权重每物理前向只读一次，与批成员数无关，
  2026-08-22 拼 batch 改造，`generate_trace.py`）；在线发射仅支持该粒度，
  其它粒度 fail-closed（`online/graph_batch_builder.py`）。
- 发射并发骨架（本仓形态：迭代列车拼 batch，2026-08-22 改造；原"批齐发-
  等完成"骨架的列车化——一趟列车 = 队列头 chunk × 迭代 + 全部
  active_decode 成员，与旧骨架 qp 队首 prefill 整段 + 全部 decode 整段的
  批齐发口径一致，等待语义由 in_flight_train 替代 busy）：连续
  batching——decode 互拼、decode 与 prefill chunk 混拼（每迭代 ≤1
  chunk）、chunk 之间不拼、批成员只在迭代（列车）边界变化。每实例
  状态机 = FCFS prefill 队列 / active_decode 批成员表 /
  pending_decode_ready（KV 就绪待加入）/ in_flight_train（唯一在飞
  列车 + train_id/membership_digest，busy 门 = 一个列车在飞）；列车
  终点 = 下一个不可预测事件（队列头 prefill drain / 全部工作耗尽），
  交付默认 T_max=8（2026-08-22 §7.4 A2 对拍裁决：无上限 TTFT -67.3%、16 仍 -19.1%、8 全指标 ≤1.3%——"固定为使位移 ≤5% 的最大值"，原则 1 优先于节点数；SH_TRAIN_MAX_ITER 可覆盖，0=不设限；截断且无自然标记的列车发哨兵标记承载完成信号）。发射 = 准入动作（到达 gates/历史迁移/逐出/屏障，含
  partial 前缀两段式恢复分支）→ 迭代列车（joiner 迁移 + 共享 readiness
  barrier + 折叠列车体（weight_passes=迭代数）+ drain/exit 标记节点 +
  每列车一个共享 end barrier）→ 完成批（completion_evictions + 下一
  turn interval gates）。列车账本核销幂等（同列车标记 watch 跨 tick
  fire、事件拆交付），推进量全部闭式；`train_ledger.jsonl` 落每列车
  审计行（§7.3 不变量断言输入，runner 归档至 results/）。
- task-load 快照物理折算（打分公式与阈值不动，红线）：三分量"恰计一次"；
  在飞 chunk 负载 = 冻结列车账本（prefill_chunk_spans，running 分量），
  其余剩余 chunk 自 prefill_tokens_completed 闭式折算（queued 分量，
  running 自 queued 扣除）；active_decode 段的"active 段不可分
  （generated=0/fraction=1.0）"假设作废，改为 decode_tokens_consumed
  闭式迭代级剩余量。
- 接栅栏与段末屏障：发射不做块末恢复/段内清链，per-rank `previous_id`
  无条件接续当前 frontier（per-rank 发行序 = 全局发射序，2026-08-19 五仓
  统一；`online/graph_batch_builder.py`），跨请求 P2P 与集合通信
  参与序不会反转成环；列车以 TP 组共享 `*_end_barrier` 收尾，drain/exit
  标记（列车体后、barrier 前）承载 PREFILL_DRAIN / DECODE_COMPLETION
  watch 与指标锚点（R2-2"barrier 前末节点"口径的列车化延续）；partial
  流水恢复的 suffix 完成门经 `_suffix_body_arms` 账本挂到含其首 chunk 的
  列车体（原"首 chunk 前缀层并行计算"保守化为整列车等恢复，登记于
  builder docstring）。
- 折入 recompute 输入口径 + 运行时 KV 账本：请求队列为唯一仿真输入，
  turn-0 源前缀折入该行 prefill_length（整段重算口径；无第二输入文件，
  会话 KV 自 turn-0 prefill 记账起由运行时账本动态维护，
  `traces/materialize_20_30s.py:3-37`）；manifest 仅携带队列派生的
  history_tokens_before 推导值（`plan_materializer.py:62-87`）；B1/WP2 起
  每请求条目追加 human_time_ns/tool_time_ns/request_type（触发本请求的
  gap：上一队列行 next_trigger_type × 本行 inter_request_interval_ns；
  request_type=human 非空→human / tool 非空→tool / 皆空 turn0→human、
  turn>0→unknown，stdout 计数；既有 9 字段不删不改，供 request_metrics.csv
  连接）；运行时账本为
  `KVCacheManager`（`sh30_online_scheduler.py:346`；`face_scheduler.py:1226`）：
  驻留命中直接复用、跨实例历史先迁移、被逐出则窗口内重算。
- long double 时间精度适配：大 ns 级首达偏移超出 IEEE-754 double 精确整数
  范围（2^53）时，分析网络适配层返回 ASTRA-sim 时间以 `long double` 保持
  64 位事件时间精确，避免完成回调与事件映射键错位 1 ns
  （`astra-sim/network_frontend/analytical/common/CommonNetworkApi.cc:71-81`）。
