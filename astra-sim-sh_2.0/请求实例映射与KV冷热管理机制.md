# 请求 Prefill/Decode 实例映射机制 与 KV 冷热管理机制

> 适用仓库：`astra-sim-sh_2.0`。
> 主要代码位置：`sh_test_mesh/workload/llama2_7b_inference/face_scheduler.py`、`generate_face_trace.py`；
> 权威设计文档另见 `sh_test_mesh/README.md`。

## 0. 总体架构：两段式"规划—执行"

本仿真器把所有调度与 KV 管理决策都放在 **trace 生成期（Python 离散事件规划器）** 中预先确定，固化为 54 个 rank 的静态 Chakra ET；C++ 仿真器（ASTRA-sim）只忠实执行 ET，不做运行期重调度。

| 层 | 职责 | 关键文件 |
|---|---|---|
| Python 规划层 | 请求→实例映射、逐出/恢复/迁移决策、逐 NPU HBM 账本，并把决策编码进 ET | `face_scheduler.py`（`KVCacheManager`、`plan_face_requests()`） |
| C++ 执行层 | 执行 ET：远端内存端口模型、本地 HBM 带宽竞争模型 | `extern/remote_memory_backend/analytical/AnalyticalRemoteMemory.{hh,cc}`、`astra-sim/workload/LocalHbmBandwidthModel.{hh,cc}`、`astra-sim/workload/Workload.cc` |

因此：**改调度/逐出策略只需改 Python 规划层并重新生成 trace**；C++ 侧不涉及 session 级决策。

---

## 1. Prefill / Decode 实例映射机制

### 1.1 统一实例，而非 PD 分离

采用 FACE 风格的 **"统一实例（unified instance）+ 逐请求动态映射"** 方案：

- 9 个实例，每个实例同时承担 Prefill 和 Decode；没有专门的 prefill 池或 decode 池。
- 实例布局唯一来源：`sh_test_mesh/workload/llama2_7b_inference/trace_config.csv:21-29`（9 行 `inference_group`）。每个实例是 3×2 = 6 个 NPU 的连续矩形，TP=6，9 个实例铺满 9×6 = 54 NPU 网格（网格尺寸由 `hardware/face_case5_config_c.json` 派生）。
- 布局顺序为"中心、四邻、四角"（instance 0 为中心 ranks 20,21,26,27,32,33……），该顺序同时是各选择键的最终确定性 tie-break。
- `face_scheduler.py:142-236 build_instances()` 校验：rank 不重叠、实心矩形、恰好覆盖全部 NPU、等尺寸（`require_equal_size=True`，用于 KV shard 按相对 TP rank 一一配对）；由物理邻接自动生成实例邻接图 `adjacency`（`:215-230`）。
- 每个实例绑定一个 ASTRA-sim 通信组：`config_resolver.py:355-398 _prepare_comm_groups()` 生成 `comm_group.json`。
- TP=6 不能整除 32 头，采用 whole-head shard：2 个 rank 各 6 头、4 个 rank 各 5 头（`face_scheduler.py:277` `attention_heads_by_tp_rank`）。**完整 session KV 只放在一个实例内，不跨实例切分**。

注意：生成目录名中的 `aggregated` 指 `trace_granularity=request_aggregated`（算子聚合），不是 PD 聚合；`p1-161734_d1-32000` 是 prefill/decode 长度范围，不是实例配比。

### 1.2 Prefill 实例选择：剩余任务负载最小 + HBM 准入

请求到达事件触发 `try_admit_request()`（`face_scheduler.py:3176-3262`）：

1. **HBM 可行性过滤**（`:3180-3197`）：`KVCacheManager.request_hbm_feasible_instances()`（`:1485-1558`）逐实例检查最终 KV 分片能否在回收非活跃 session 后放下；暂不可行则进 `pending_admissions` 等待重试。
2. **Session 亲和**：若 session 历史 KV 为 `PARTIAL`（前层驻留、后层远端），强制返回驻留实例（`:3207-3216`，`prefill_affinity_reason="resident_prefix_layers"`）。
3. **否则负载均衡**：`select_prefill_instance()`（`:796-817`）在 HBM 可行实例中取 `ordering_key` 最小者（`InstanceTaskLoadSnapshot.ordering_key`，`:787-793`）：

```python
(self.total_task_load_ns,                      # L_instance 升序
 -1 if self.last_arrival_ns is None else self.last_arrival_ns,  # 从未用过优先，其次最久未入队
 self.instance_index)                          # 实例配置顺序
```

`L_instance = L_running_prefill + L_queued_prefill + L_active_decode`（`:779-785`），三项均以 **Roofline 估计服务时间（ns）** 计量：

- running prefill 按当前 iteration 剩余比例折算（`remaining_iteration_fraction`，`:3081-3089`）；
- queued prefill 逐 chunk 累加（`:3091-3117`），chunk 固定 `PREFILL_CHUNK_SIZE = 512`（`face_scheduler.py:22`）；
- active decode 用 `estimate_decode_remaining_task_load_ns()`（`:584-637`），预期生成长度取工作负载均值，达到均值后剩余负载为 0；
- 单 iteration Roofline 估计：`estimate_iteration_time_ns()`（`:486-553`），linear/attention 分别取 max(计算时间, HBM 延迟+字节/带宽)，prefill 与 decode attention 取 max（FACE 重叠假设）。

准入后：`reserve_request_capacity()` 预留容量（可能触发逐出），`prepare_prefill()`（`:2378-2526`）按历史 KV 位置做四种处理——本地命中 / 跨实例 NoC 迁移 / 远端载入 / PARTIAL 后缀流水恢复——然后请求入所选实例 FCFS 队列队尾。

### 1.3 实例内 PD 混合执行

`start_ready_iterations()`（`:3272-3327`）：实例空闲时，取队首请求的 1 个 prefill chunk + 全部 active decode 请求组成一个**混合 iteration**，查 LUT 得 `iteration_time_ns`，排 `iteration_complete` 事件。即同一 TP group 上 PD 以 batch 形式时分复用。

### 1.4 Decode 实例选择：加权 Instance_map + LUT per-die 代价

Prefill 最后一 chunk 完成时（`:3363-3437`）调用 `select_decode_instance()`（`:901-989`）：

1. **候选集**：`WeightedInstanceGraph.schedulable_instances()`（`:862-868`）= 实例邻接图上到 prefill 实例的最短加权距离 ≤ `schedulable_distance_limit`（`:65-66`）= `d2d_bw / hbm_bw`。按当前硬件为 4050/1640 ≈ **2.47** 跳（即 prefill 实例自身 + 一跳邻实例）。
2. **代价**：对每个候选查 LUT（`FaceLut.lookup`，`:706-720`），`cost = (T_prime − T) / instance_size`（`:959-967`）。
3. **选择键**（`:981-988`）：

```python
selected = min(feasible_costs, key=lambda cost: (
    cost.per_die_delta_ns,                 # 先最小 per-die 增量代价
    -cost.remaining_hbm_capacity_bytes,    # 并列时 HBM 剩余大者优先
    cost.instance_index))                  # 再按实例配置顺序
```

HBM 只做硬约束（`decode_hbm_feasible_instances`，`:1711-1736`）和并列 tie-break，不覆盖更低 cost；映射后不再因迁移开销改派。若 decode 实例 ≠ prefill 实例，`move_prefill_to_decode()`（`:2590-2632`）把本轮累积 KV 按相对 TP rank 经最少跳数 NoC 迁到 decode 实例；请求完成后 session 权威位置为 decode 实例。

补充：`WeightedInstanceGraph.increase_path/decrease_path`（`:870-886`）与 `KVAllocator`（`:1009-1098`）实现了 FACE 的边权动态调整与 KV 跨实例分片，但**实际规划路径中没有调用**（仅测试引用）；边权恒为 1，即加权距离 = 实例跳数。

### 1.5 Trace 生成：映射结果如何固化

流程入口：`generate_trace.sh` → `generate_trace.py` → `generate_face_trace.py main()`（`:3080-3118`）：

1. `load_face_trace_config()`（`:747-859`）：读 `trace_config.csv`；加载请求队列 CSV（队列由 `trace_config.csv:12` 指定，规模随其变化；compute_100 三分钟窗口队列对应 180s 窗口、678 session、9179 请求）；`config_resolver.materialize_runtime_configs()` 派生 system/network/remote_memory/comm_group 四件套到 `generated/runtime_config/`。
2. `load_or_build_face_plan()` → `plan_face_requests()`；结果 pickle 缓存于 `generated/planner_cache/`。
3. `write_face_trace()`（`:2048-3006`）：按 KV 因果拓扑 + prefill 开始时间排序逐请求向 **prefill_group.ranks 与 decode_group.ranks 各自的 ET builder** 发射节点——实例映射结果直接体现在"节点写进哪些 rank 的 `.et` 文件、collective 用哪个 pg_name"：
   - 首请求在 prefill 实例各 rank 写 `timer_gate`（到达门控）；
   - 历史 KV 迁移/逐出以 MEM_LOAD/MEM_STORE/NoC 传输节点表示；
   - Prefill 逐 chunk（token_expanded）或聚合（request_aggregated）发射；PARTIAL session 拆 prefix/suffix 两段实现流水恢复；
   - prefill→decode KV 迁移 + decode TP group 就绪屏障；
   - Decode 在 `decode_group.ranks` 上发射，pg 用 `decode_group.pg_name`。
4. 产物：54 个 `llama2_7b_inference.{rank}.et`、`face_lut.csv`、`manifest.json`。manifest 完整记录映射审计信息：`instances[].role="unified_prefill_decode"`、`prefill_assignment_policy`、`decode_policy`、每请求的 prefill/decode 实例、候选代价、KV 迁移、HBM 快照等。

---

## 2. KV 冷热管理机制

### 2.1 冷热分层架构

**热层——每 NPU 本地 HBM**（唯一人工维护来源 `sh_test_mesh/hardware/face_case5_config_c.json`）：

- 带宽 1640 GB/s、延迟 100 ns；
- 容量走 profile：`paper-64gib`（64 GiB）与 `validation-160gib`（160 GiB，当前 trace 选用）；
- `config_resolver.py` 校验后把 `local-mem-bw / local-mem-latency / local-mem-capacity-bytes` 等写入派生 `system.json`；
- Python 侧账本：`KVCacheManager.__init__`（`face_scheduler.py:1254-1329`）为每个 rank 建 `NodeHBMState`（capacity、model_weight、kv_cache、remaining），所有操作后 `_check_invariants()`（`:1863-1934`）校验守恒。

**冷层——mesh 边界挂接的远端内存池**（`face_case5_config_c.json:31-37`）：

- `memory-type: PER_NPU_MEMORY_EXPANSION`、512 GB/s、100 ns、`npu-selection: mesh-boundary`、逻辑池 `unified-kv-cache-pool`；
- 派生的 `remote_memory.json`：26 个边界 rank 各一个端口（9×6 mesh 边界由 `config_resolver.py:342-348` 动态推导）；
- C++ 模型 `AnalyticalRemoteMemory`：每端口一条严格 FIFO 队列，端口间并行；**不限制远端池总容量，也不把 26 个端口汇聚成全局带宽瓶颈**；
- KV 路由：非边界 rank 经最近边界端口存取，端口选择键 `(Manhattan hops, edge rank)`（`physical_edge_ranks` `face_scheduler.py:405-418`、`nearest_edge_rank` `:446-457`）。

### 2.2 三态驻留状态机

`KVCacheManager`（`face_scheduler.py:1247-1252`）为每个 session 维护三态：

| 状态 | 含义 |
|---|---|
| `local_hbm` | 全部层在本地 HBM |
| `partial_hbm_remote` | 前 `P = L − floor(L/2)` 层在本地，后 `K = floor(L/2)` 层在远端 |
| `remote_memory` | 全部层在远端 |

层边界动态计算：`partial_resident_prefix_layers = model.layers − model.layers // 2`（`:1276-1280`），32 层 Llama2-7B 即保留 `[0,16)`、逐出 `[16,32)`，代码无写死常数。

### 2.3 两阶段"降水位"逐出（后 K 层优先）

触发点——`_ensure_capacity()`（`face_scheduler.py:2298-2376`）在以下路径被调用：

- 请求准入预留最终 KV 容量：`reserve_request_capacity`（`:1599-1647`）；
- Decode 预留迁移：`move_request_capacity_reservation`（`:1649` 起，调用点 `:1682`）；
- 历史 KV 跨实例迁移/远端恢复前：`prepare_prefill`（`:2378-2526`）；
- Prefill/Decode KV 增长：`_expand_local_session`、`expand_prefill`、`expand_decode`；
- Prefill→Decode 迁移：`move_prefill_to_decode`（`:2590-2632`）。

**请求完成后的水位线检查**不走 `_ensure_capacity`：`complete_request`（`:2732-2746`）→ `enforce_reserve`（`:2686-2730`）内联实现同样的两阶段逐出（直接调 `_evict_suffix` / `_evict_session`）——要求每 rank 剩余 ≥ `kv_reserve_context_tokens`（默认 1,000,000 token，`trace_config.csv:16`）对应的 KV 分片字节；仍不满足时记录 `reserve_unmet_ranks` 而不死循环。

核心算法（`:2309-2376`）：

- **阶段 1**（`:2310-2332`）：水位不足时，反复取最老的"全本地、已完成、非 active"session（`_completed_full_candidates`，`:2166-2179`），调用 `_evict_suffix`（`:2194-2231`）**只把后 K 层** `remote_store` 到远端，session 转为 `partial_hbm_remote`。
- **阶段 2**（`:2337-2376`）：仅当所有可逐出 session 都已减半仍不足，再按同样顺序调用 `_evict_session`（`:2233-2272`）把剩余前缀层全部逐出，session 转为 `remote_memory`。

排序键为 `(last_completion_ns, session_id)`（`_oldest_completed_first_sort`，`:2156-2164`）——即 **"最后完成时间最旧优先"，效果上等价于 LRU（least-recently-completed）**，代码与元数据均以此命名：每个 rank ET 的 GlobalMetadata 写 `kv_eviction_policy=oldest_completed_first_half_layer_suffix_then_full_fallback`（`generate_face_trace.py:1846-1848`），manifest.json 的 `kv_management.policy` 为 `oldest_completed_first_half_layer_suffix_then_full_session_fallback`（`:2898`）。

保护规则：active session（正在 Prefill/Decode/恢复中）不进候选集；新请求到达先置 `session.active = True` 再做容量检查，保证"请求永远不会成为自己的逐出受害者"。

### 2.4 恢复：远端取回（非重计算）+ 流水重叠

- `partial_hbm_remote` session 的后续请求**固定回到保留前缀的原实例**（`prepare_prefill`，`:2464-2468` 强制亲和），只生成 `[P, L)` 层区间的 `remote_load`（`_remote_load_transfer`，`:2045-2097`）；`remote_memory` session 则全层 `remote_load` 到任意选中实例。恢复后状态归一化为 `local_hbm`。
- **冷层 KV 的恢复一律走远端读取，不重算**。"重计算"只出现在另一种语义：请求声明的源 prefix 超过实际保存的历史 KV 时，超出部分作为 Prefill 重新计算（`_validate_and_expand_requests` `:2932-2989` 中的钳位逻辑 `:2963-2975`）。
- ET 层的流水重叠（`generate_face_trace.py`）：
  - `remote_store`（`:1480-1553`）：源 rank 即边界端口则直接 `MEM_STORE`；否则 `COMM_SEND → edge COMM_RECV → edge MEM_STORE → 1B ACK`，源端释放依赖 ACK；
  - `remote_load`（`:1555-1633`）：`target 发 1B 请求 → edge MEM_LOAD → edge COMM_SEND → target COMM_RECV → target 本地 HBM restore DMA（local_hbm_kv_restore）`；
  - **部分恢复流水**（屏障与并行恢复发起 `:2281-2408`；chunk 拆分 `:2447-2475` token_expanded / `:2519-2544` request_aggregated）：先对前缀做 TP readiness barrier，从 checkpoint 分支并行发起后缀远端加载；第一个 Prefill chunk 拆成 prefix 段（`layer_start=0, layer_end=P`）与 suffix 段，suffix 段算子依赖带 tag 的 P2P readiness barrier（所有 rank 的 HBM DMA 完成）——实现"**前缀计算与后缀恢复并行，首个后层算子才等待**"。

### 2.5 冷层访问的代价建模（C++ 执行层）

远端读取代价分三段计入仿真：

1. **远端端口访问**：`AnalyticalRemoteMemory::get_remote_mem_runtime()`（`AnalyticalRemoteMemory.cc:220-224`）：
   `runtime = remote_mem_latency + floor(tensor_size / remote_mem_bw)`（100 ns + 字节/512 GB/s，带宽项向下截断）。同端口事务 FIFO 串行排队。
2. **NoC 段**：边界端口与目标 rank 间的 `COMM_SEND/RECV` 走分析型网络后端（D2D 4050 GB/s、5 ns/跳）。
3. **目标端本地 HBM DMA 写入**：ET 中 `MEM_LOAD` 节点带 `is_local_hbm_kv_restore` 属性，由 `Workload::issue` 分发（`Workload.cc:191-197`）到 `issue_local_hbm_kv_restore`（`:265-285`）：
   - 开关关闭时：`runtime = max(1ns, ceil(local_mem_latency + tensor_size / local_mem_bw))`（`Workload.cc:278-284`）；
   - 开关打开时：进入 `LocalHbmBandwidthModel`，与同一 NPU 上的推理 COMP 节点做**流体带宽竞争**——双方都有未完成 HBM 字节时各 50%，一方完成后另一方立即回 100%（`LocalHbmBandwidthModel.cc` `advance_to` 53-151、`issue_restore` 224-240 行），并统计 `tics_hbm_dma_ops`。注意开关的代码默认值为 false（`Sys.cc:173`），由当前 system 模板 `hbm-kv-restore-bandwidth-sharing: 1` 启用。

> 补充：system.json 的 `remote-mem-bw/latency` 还被 Roofline 路径用于带 `remote_weight_bytes` 的 COMP 节点远端操作数流水（`Workload.cc:325-347`），但当前 `remote_operand_loads=false` 未启用，且与 KV restore sharing 互斥。

### 2.6 配置参数汇总

| 配置项 | 位置 | 当前值 | 作用 |
|---|---|---|---|
| `kv_reserve_context_tokens` | `trace_config.csv:16` | 1,000,000 | 完成后水位线：每 rank 预留 1M token KV 分片空间 |
| `local_hbm_capacity_profile` | `trace_config.csv:18` | `validation-160gib` | HBM 容量 profile（64/160 GiB） |
| `hardware_config` | `trace_config.csv:17` | `hardware/face_case5_config_c.json` | 热/冷层带宽、延迟、容量、边界端口唯一来源 |
| `hbm-kv-restore-bandwidth-sharing` | `system/llama2_7b_roofline_template.json:13` | 1 | 启用推理/KV 恢复 HBM 50/50 动态共享（`Sys.cc:412-422` 解析） |
| `remote-mem-bw` / `remote-mem-latency` | 派生 `system.json` 与 `remote_memory.json` | 512 GB/s / 100 ns | 远端端口代价（`Sys.cc:400-406`、`AnalyticalRemoteMemory.cc:105-120`） |
| `local-mem-bw` / `local-mem-latency` / `local-mem-capacity-bytes` | 派生 `system.json` | 1640 GB/s / 100 ns / 160 GiB | 热层参数；C++ 只解析 bw/latency（`Sys.cc:393-399`），capacity 写入后 C++ 端不消费（容量约束只在 Python 账本生效） |
| `remote_operand_loads` | `trace_config.csv:20` | false | 不建模远端权重加载 |

策略标识的落盘位置：`kv_eviction_policy=oldest_completed_first_half_layer_suffix_then_full_fallback`、`kv_partial_resident_prefix_layers`、`hbm_kv_restore_bandwidth_sharing=50_50_while_overlapped_then_100_to_survivor` 是每个 rank ET 的 Chakra GlobalMetadata 属性（`generate_face_trace.py:1846-1856`）；manifest.json 中对应记录在 `kv_management` 段（`policy=oldest_completed_first_half_layer_suffix_then_full_session_fallback`、`resident_prefix_layers_after_stage_one`、`hbm_bandwidth_overlap`，`:2897-2927`）。

运行：

- `sh_test_mesh/run_scripts/generate_trace.sh [-j JOBS]`：Python 规划 + 生成 54 份 ET + manifest；
- `sh_test_mesh/run_scripts/run_sh_test_aware.sh`：校验 `REMOTE_MEMORY` 配置后以 `--remote-memory-configuration` 启动 `AstraSim_Analytical_Congestion_Aware`；
- 单测：`sh_test_mesh/workload/llama2_7b_inference/test_face_scheduler.py`（36 个用例，覆盖逐出、亲和、流水恢复）。

---

## 3. 关键文件索引

- `sh_test_mesh/workload/llama2_7b_inference/face_scheduler.py`
  - `build_instances()` `:142-236` — 实例布局/校验/邻接图
  - `InstanceTaskLoadSnapshot` `:749-793` — 负载计量与 prefill 选择键
  - `select_decode_instance()` `:901-989` — decode 候选/代价/选择键
  - `KVCacheManager` `:1247-2746` — HBM 账本、三态机、两阶段逐出、恢复、水位线
  - `plan_face_requests()` `:2992-3624` — 离散事件主循环
- `sh_test_mesh/workload/llama2_7b_inference/generate_face_trace.py:1480-1633, 2281-2408` — KV 迁移/逐出/恢复事件的 ET 编码与流水重叠
- `extern/remote_memory_backend/analytical/AnalyticalRemoteMemory.{hh,cc}` — 远端端口 FIFO 与代价
- `astra-sim/workload/LocalHbmBandwidthModel.{hh,cc}` — 每 NPU HBM 推理/恢复 50/50 流体共享
- `astra-sim/workload/Workload.cc:191-197, 256-285` — 节点分发、远端访问与本地 HBM 恢复发射
- `astra-sim/system/Sys.{hh,cc}` — 内存相关 system 字段解析（`Sys.cc:393-422`）

## 4. 注意点与坑

- 逐出排序为"最久未完成优先"（least-recently-completed，类 LRU 的 recency 语义，非严格 FIFO）；代码与元数据中的命名已统一为 `oldest_completed_first`（旧称 FIFO 已更正）。
- 冷层容量不建模（视为无限），26 个边界端口之间无全局带宽汇聚；读写同价。
- 远端带宽 512 GB/s 在 C++ 中按"每 ns 512 byte"十进制使用，传输时间带宽项向下截断。
- 所有映射/迁移/逐出决策在 trace 生成时确定，仿真运行期不会因实际拥塞重调度（静态 ET 近似）。
- 实例规划时间（manifest，LUT 调度时间）与 ASTRA-sim 执行时间（硬件资源模型结果）是两套口径。
