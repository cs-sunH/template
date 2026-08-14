# 请求 Prefill/Decode 实例映射机制 与 KV 冷热管理机制

> 适用仓库：`astra-sim-sh_3.0`。
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
- `face_scheduler.py:187-281 build_instances()` 校验：rank 不重叠、实心矩形、恰好覆盖全部 NPU、等尺寸（`require_equal_size=True`，用于 KV shard 按相对 TP rank 一一配对）；由物理邻接自动生成实例邻接图 `adjacency`（`:260-275`）。
- 每个实例绑定一个 ASTRA-sim 通信组：`config_resolver.py:355-398 _prepare_comm_groups()` 生成 `comm_group.json`。
- TP=6 不能整除 32 头，采用 whole-head shard：2 个 rank 各 6 头、4 个 rank 各 5 头（`face_scheduler.py:322` `attention_heads_by_tp_rank`）。**完整 session KV 只放在一个实例内，不跨实例切分**。

注意：生成目录名中的 `aggregated` 指 `trace_granularity=request_aggregated`（算子聚合），不是 PD 聚合；`p1-161734_d1-32000` 是 prefill/decode 长度范围，不是实例配比。

### 1.2 Prefill 实例选择：三段式分配（首请求避边缘 / HBM 命中 sticky / 远端命中负载均衡）

请求到达事件触发 `try_admit_request()`（`face_scheduler.py:3697-3823`）：

1. **HBM 可行性过滤**（`:3701-3718`）：`KVCacheManager.request_hbm_feasible_instances()`（`:1574-1648`）逐实例检查最终 KV 分片能否在回收非活跃 session 后放下；全集暂不可行则进 `pending_admissions` 等待重试，永远不可行则 `raise ValueError`。
2. **三段式分配**（`:3728-3782`）：
   - **session 首请求**（判据 `session_id not in kv_manager.session_ids`）：候选集 = **不含边缘 rank 的实例** ∧ HBM 可行。实例级边缘判定由 `edge_free_instance_mask()`（`:466-482`）给出，边缘 rank 即 `physical_edge_ranks()`（`:450-463`）的 mesh 边界 rank；当前 9 实例布局下非边缘集合退化为单例 `{instance 0}`。候选集为空则 `return False` 等待重试（**不回退全集**），并以 `request_hbm_eventually_feasible_instances()` ∧ 非边缘掩码做死等防护（交集为空 `raise RuntimeError`）。记 `prefill_affinity_reason="first_request_non_edge"`。例外：拓扑级不存在任何非边缘实例时（仅 2×2/1×2 等测试玩具拓扑）回退全集负载均衡，记 `"first_request_edge_fallback"`。
   - **历史 KV 为 `LOCAL_HBM` 或 `PARTIAL_HBM_REMOTE`**：**sticky 到 KV 驻留实例**，reason 分别记 `"resident_local_hbm"` / `"resident_prefix_layers"`；该实例暂不可行则等待重试（不回退）。LOCAL_HBM sticky 使 `prepare_prefill` 必然命中本地重用分支；PARTIAL 语义与既往一致。
   - **历史 KV 为 `REMOTE_MEMORY`**：在全部实例 ∧ HBM 可行集上负载均衡（reason 为 `None`，与既往行为一致）。
3. **负载均衡选择键**：`select_prefill_instance()`（`:860-881`）在调用点给定的候选掩码内取 `ordering_key` 最小者（`InstanceTaskLoadSnapshot.ordering_key`，`:851-857`）：

```python
(self.total_task_load_ns,                      # L_instance 升序
 -1 if self.last_arrival_ns is None else self.last_arrival_ns,  # 从未用过优先，其次最久未入队
 self.instance_index)                          # 实例配置顺序
```

`L_instance = L_running_prefill + L_queued_prefill + L_active_decode`（`:844-849`），三项均以 **Roofline 估计服务时间（ns）** 计量：

- running prefill 按当前 iteration 剩余比例折算（`remaining_iteration_fraction`，`:3602-3610`）；
- queued prefill 逐 chunk 累加（`:3612-3638`），chunk 固定 `PREFILL_CHUNK_SIZE = 512`（`face_scheduler.py:22`）；
- active decode 用 `estimate_decode_remaining_task_load_ns()`（`:648-702`），预期生成长度取工作负载均值，达到均值后剩余负载为 0；
- 单 iteration Roofline 估计：`estimate_iteration_time_ns()`（`:550-618`），linear/attention 分别取 max(计算时间, HBM 延迟+字节/带宽)，prefill 与 decode attention 取 max（FACE 重叠假设）。

准入后：`reserve_request_capacity()` 预留容量（可能触发逐出），`prepare_prefill()`（`:2840-3023`）按历史 KV 位置做四种处理——本地命中 / 跨实例 NoC 迁移 / 远端载入 / PARTIAL 后缀流水恢复——然后请求入所选实例 FCFS 队列队尾。注：LOCAL_HBM sticky 与 decode 本地化使"跨实例 NoC 迁移"分支（reason=`history_other_instance`）在规划路径上不可达，代码保留不删。

### 1.3 实例内 PD 混合执行

`start_ready_iterations()`（`:3833-3890`）：实例空闲时，取队首请求的 1 个 prefill chunk + 全部 active decode 请求组成一个**混合 iteration**，查 LUT 得 `iteration_time_ns`，排 `iteration_complete` 事件。即同一 TP group 上 PD 以 batch 形式时分复用。

### 1.4 Decode 实例选择：固定于 Prefill 同实例

Prefill 最后一 chunk 完成时（`:3931-3984`）**不再做候选评估**：decode 固定 `selected = state_index`（`:3947`，即 prefill 实例自身），`decode_candidates` 记录为**空元组**。同实例路径由既有机制天然支持：

- `move_request_capacity_reservation()`（`:1755-1849`）同实例直接 no-op；
- `move_prefill_to_decode()`（`:3099-3151`）同实例走 `local_hit`（reason=`prefill_decode_local_reuse`），**无 NoC 迁移**（`prefill_decode_instance_migrate` 在规划路径上消失）；请求完成后 session 权威位置仍为该实例；
- decode 增长的容量检查仍由 `expand_decode → _expand_local_session → _ensure_capacity` 链承担，不绕过 HBM 账本。

原 FACE decode 选择设施**保留代码、规划路径不再调用**（测试仍直接引用）：`select_decode_instance()`（`:965-1053`）——候选集 `WeightedInstanceGraph.schedulable_instances()`（`:926-932`，实例邻接图最短加权距离 ≤ `schedulable_distance_limit` `:110-112` = `d2d_bw / hbm_bw`，按当前硬件 4050/1640 ≈ **2.47** 跳）、代价 `(T_prime − T) / instance_size`、选择键 `(per_die_delta_ns, −remaining_hbm_capacity_bytes, instance_index)`、HBM 硬约束 `decode_hbm_feasible_instances()`（`:1861-1887`）。`WeightedInstanceGraph` 仍在 `plan_face_requests` 中构造，用于 `final_edge_weights` 审计输出。

补充：`WeightedInstanceGraph.increase_path/decrease_path`（`:934-950`）与 `KVAllocator`（`:1073-1164`）实现了 FACE 的边权动态调整与 KV 跨实例分片，但**实际规划路径中没有调用**（仅测试引用）；边权恒为 1，即加权距离 = 实例跳数。

### 1.5 Trace 生成：映射结果如何固化

流程入口：`generate_trace.sh` → `generate_trace.py` → `generate_face_trace.py main()`（`:3199-3278`）：

1. `load_face_trace_config()`（`:761-875`）：读 `trace_config.csv`；加载请求队列 CSV（队列由 `trace_config.csv:12` 指定，规模随其变化；compute_100 三分钟窗口队列对应 180s 窗口、678 session、9179 请求）；`config_resolver.materialize_runtime_configs()` 派生 system/network/remote_memory/comm_group 四件套到 `generated/runtime_config/`。
2. `load_or_build_face_plan()` → `plan_face_requests()`；结果 pickle 缓存于 `generated/planner_cache/`。
3. `write_face_trace()`（`:2074-3127`）：按 KV 因果拓扑 + prefill 开始时间排序逐请求向 **prefill_group.ranks 与 decode_group.ranks 各自的 ET builder** 发射节点——实例映射结果直接体现在"节点写进哪些 rank 的 `.et` 文件、collective 用哪个 pg_name"（decode 固定同实例后，prefill_group 与 decode_group 逐请求恒相同）：
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
- Python 侧账本：`KVCacheManager.__init__`（`face_scheduler.py:1318-1420`）为每个 rank 建 `NodeHBMState`（capacity、model_weight、kv_cache、remaining），所有操作后 `_check_invariants()`（`:2040-2112`）校验守恒。

**冷层——mesh 边界挂接的远端内存池**（`face_case5_config_c.json:31-37`）：

- `memory-type: PER_NPU_MEMORY_EXPANSION`、512 GB/s、100 ns、`npu-selection: mesh-boundary`、逻辑池 `unified-kv-cache-pool`；
- 派生的 `remote_memory.json`：26 个边界 rank 各一个端口（9×6 mesh 边界由 `config_resolver.py:342-348` 动态推导）；
- C++ 模型 `AnalyticalRemoteMemory`：每端口一条严格 FIFO 队列，端口间并行；**不限制远端池总容量，也不把 26 个端口汇聚成全局带宽瓶颈**；
- KV 路由：非边界 rank 经最近边界端口存取，端口选择键 `(Manhattan hops, edge rank)`（`physical_edge_ranks` `face_scheduler.py:450-463`、`nearest_edge_rank` `:510-523`）。

### 2.2 三态驻留状态机

`KVCacheManager`（`face_scheduler.py:1314-1316`）为每个 session 维护三态：

| 状态 | 含义 |
|---|---|
| `local_hbm` | 全部层在本地 HBM |
| `partial_hbm_remote` | 前 `P = L − floor(L/2)` 层在本地，后 `K = floor(L/2)` 层在远端 |
| `remote_memory` | 全部层在远端 |

层边界动态计算：`partial_resident_prefix_layers = model.layers − model.layers // 2`（`:1342-1344`），32 层 Llama2-7B 即保留 `[0,16)`、逐出 `[16,32)`，代码无写死常数。

### 2.3 两阶段"降水位"逐出（后 K 层优先）

触发点——`_ensure_capacity()`（`face_scheduler.py:2760-2838`）在以下路径被调用：

- 请求准入预留最终 KV 容量：`reserve_request_capacity`（`:1688-1754`）；
- Decode 预留迁移：`move_request_capacity_reservation`（`:1755` 起，规划路径调用点 `:3951`，decode 固定同实例后恒为 no-op）；
- 历史 KV 跨实例迁移/远端恢复前：`prepare_prefill`（`:2840-3023`）；
- Prefill/Decode KV 增长：`_expand_local_session`、`expand_prefill`、`expand_decode`；
- Prefill→Decode 迁移：`move_prefill_to_decode`（`:3099-3151`，decode 固定同实例后走 local_hit，不再触发逐出）。

**请求完成后的水位线检查**不走 `_ensure_capacity`：`complete_request`（`:3251-3267`）→ `enforce_reserve`（`:3205-3250`）内联实现同样的两阶段逐出（直接调 `_evict_suffix` / `_evict_session`）——要求每 rank 剩余 ≥ `kv_reserve_context_tokens`（默认 1,000,000 token，`trace_config.csv:16`）对应的 KV 分片字节；仍不满足时记录 `reserve_unmet_ranks` 而不死循环。

核心算法（`:2771-2838`）：

- **阶段 1**（`:2772-2794`）：水位不足时，反复取最老的"全本地、已完成、非 active"session（`_completed_full_candidates`，`:2612-2626`），调用 `_evict_suffix`（`:2640-2687`）**只把后 K 层** `remote_store` 到远端，session 转为 `partial_hbm_remote`。
- **阶段 2**（`:2796-2838`）：仅当所有可逐出 session 都已减半仍不足，再按同样顺序调用 `_evict_session`（`:2689-2735`）把剩余前缀层全部逐出，session 转为 `remote_memory`。

排序键为 `(last_completion_ns, session_id)`（`_fifo_sort`，`face_scheduler.py:2602-2610`）——即 **"最后完成时间最旧优先"，效果上等价于 LRU（least-recently-completed）**；代码排序函数与元数据取值均以 `fifo` 命名：每个 rank ET 的 GlobalMetadata 写 `kv_eviction_policy=fifo_half_layer_suffix_then_full_fallback`（`generate_face_trace.py:1872-1874`），manifest.json 的 `kv_management.policy` 为 `fifo_half_layer_suffix_then_full_session_fallback`（`:3017`）。

保护规则：active session（正在 Prefill/Decode/恢复中）不进候选集；新请求到达先置 `session.active = True` 再做容量检查，保证"请求永远不会成为自己的逐出受害者"。

### 2.4 恢复：远端取回（非重计算）+ 流水重叠

- `partial_hbm_remote` session 的后续请求**固定回到保留前缀的原实例**（`prepare_prefill`，`:2936-2940` 强制亲和），只生成 `[P, L)` 层区间的 `remote_load`（`_remote_load_transfer`，`:2491-2544`）；`remote_memory` session 则全层 `remote_load` 到任意选中实例。恢复后状态归一化为 `local_hbm`。
- **冷层 KV 的恢复一律走远端读取，不重算**。"重计算"只出现在另一种语义：请求声明的源 prefix 超过实际保存的历史 KV 时，超出部分作为 Prefill 重新计算（`_validate_and_expand_requests` `:3451-3509` 中的钳位逻辑 `:3478-3494`）。
- ET 层的流水重叠（`generate_face_trace.py`）：
  - `remote_store`（`_emit_kv_transfer` store 分支，`:1502-1576`）：源 rank 即边界端口则直接 `MEM_STORE`；否则 `COMM_SEND → edge COMM_RECV → edge MEM_STORE → 1B ACK`，源端释放依赖 ACK；
  - `remote_load`（load 分支，`:1577-1664`）：`target 发 1B 请求 → edge MEM_LOAD → edge COMM_SEND → target COMM_RECV → target 本地 HBM restore DMA（local_hbm_kv_restore）`；
  - **部分恢复流水**（屏障与并行恢复发起 `:2323-2463`；chunk 拆分 `:2497-2531` token_expanded / `:2573-2598` request_aggregated）：先对前缀做 TP readiness barrier，从 checkpoint 分支并行发起后缀远端加载；第一个 Prefill chunk 拆成 prefix 段（`layer_start=0, layer_end=P`）与 suffix 段，suffix 段算子依赖带 tag 的 P2P readiness barrier（所有 rank 的 HBM DMA 完成）——实现"**前缀计算与后缀恢复并行，首个后层算子才等待**"。

### 2.5 冷层访问的代价建模（C++ 执行层）

远端读取代价分三段计入仿真：

1. **远端端口访问**：`AnalyticalRemoteMemory::get_remote_mem_runtime()`（`AnalyticalRemoteMemory.cc:220-224`）：
   `runtime = remote_mem_latency + floor(tensor_size / remote_mem_bw)`（100 ns + 字节/512 GB/s，带宽项向下截断）。同端口事务 FIFO 串行排队。
2. **NoC 段**：边界端口与目标 rank 间的 `COMM_SEND/RECV` 走分析型网络后端（D2D 4050 GB/s、5 ns/跳）。
3. **目标端本地 HBM DMA 写入**：ET 中 `MEM_LOAD` 节点带 `is_local_hbm_kv_restore` 属性，由 `Workload::issue` 分发（`Workload.cc:198-204`）到 `issue_local_hbm_kv_restore`（`:272-299`）：
   - 开关关闭时：`runtime = max(1ns, ceil(local_mem_latency + tensor_size / local_mem_bw))`（`Workload.cc:291-298`）；
   - 开关打开时：进入 `LocalHbmBandwidthModel`，与同一 NPU 上的推理 COMP 节点做**流体带宽竞争**——双方都有未完成 HBM 字节时各 50%，一方完成后另一方立即回 100%（`LocalHbmBandwidthModel.cc` `advance_to` 53 起、`issue_restore` 240 起），并统计 `tics_hbm_dma_ops`。注意开关的代码默认值为 false（`Sys.cc:173`），由当前 system 模板 `hbm-kv-restore-bandwidth-sharing: 1` 启用。

> 补充：system.json 的 `remote-mem-bw/latency` 还被 Roofline 路径用于带 `remote_weight_bytes` 的 COMP 节点远端操作数流水（`Workload.cc:345-363`），但当前 `remote_operand_loads=false` 未启用，且与 KV restore sharing 互斥。

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

策略标识的落盘位置：`kv_eviction_policy=fifo_half_layer_suffix_then_full_fallback`、`kv_partial_resident_prefix_layers`、`hbm_kv_restore_bandwidth_sharing=50_50_while_overlapped_then_100_to_survivor` 是每个 rank ET 的 Chakra GlobalMetadata 属性（`generate_face_trace.py:1872-1882`）；manifest.json 中对应记录在 `kv_management` 段（`policy=fifo_half_layer_suffix_then_full_session_fallback`、`resident_prefix_layers_after_stage_one`、`hbm_bandwidth_overlap`，`:3016-3077`）。

运行：

- `sh_test_mesh/run_scripts/generate_trace.sh [-j JOBS]`：Python 规划 + 生成 54 份 ET + manifest；
- `sh_test_mesh/run_scripts/run_sh_test_aware.sh`：校验 `REMOTE_MEMORY` 配置后以 `--remote-memory-configuration` 启动 `AstraSim_Analytical_Congestion_Aware`；
- 单测：`sh_test_mesh/workload/llama2_7b_inference/test_face_scheduler.py`（42 个用例，覆盖逐出、亲和、流水恢复与 prefill/decode 分配策略）。

---

## 3. 关键文件索引

- `sh_test_mesh/workload/llama2_7b_inference/face_scheduler.py`
  - `build_instances()` `:187-281` — 实例布局/校验/邻接图
  - `InstanceTaskLoadSnapshot` `:813-857` — 负载计量与 prefill 选择键
  - `select_decode_instance()` `:965-1053` — decode 候选/代价/选择键（保留代码，规划路径不再调用）
  - `KVCacheManager` `:1311-3267` — HBM 账本、三态机、两阶段逐出、恢复、水位线
  - `plan_face_requests()` `:3511-4172` — 离散事件主循环
- `sh_test_mesh/workload/llama2_7b_inference/generate_face_trace.py:1502-1664, 2323-2463` — KV 迁移/逐出/恢复事件的 ET 编码与流水重叠
- `extern/remote_memory_backend/analytical/AnalyticalRemoteMemory.{hh,cc}` — 远端端口 FIFO 与代价
- `astra-sim/workload/LocalHbmBandwidthModel.{hh,cc}` — 每 NPU HBM 推理/恢复 50/50 流体共享
- `astra-sim/workload/Workload.cc:198-204, 272-299` — 节点分发、远端访问与本地 HBM 恢复发射
- `astra-sim/system/Sys.{hh,cc}` — 内存相关 system 字段解析（`Sys.cc:393-422`）

## 4. 注意点与坑

- 逐出排序为"最久未完成优先"（least-recently-completed，类 LRU 的 recency 语义，非严格 FIFO）；代码排序函数与元数据取值均以 `fifo` 命名（`_fifo_sort`、`kv_eviction_policy=fifo_half_layer_suffix_then_full_fallback`），指代同一 `(last_completion_ns, session_id)` 排序键。
- 冷层容量不建模（视为无限），26 个边界端口之间无全局带宽汇聚；读写同价。
- 远端带宽 512 GB/s 在 C++ 中按"每 ns 512 byte"十进制使用，传输时间带宽项向下截断。
- 所有映射/迁移/逐出决策在 trace 生成时确定，仿真运行期不会因实际拥塞重调度（静态 ET 近似）。
- 实例规划时间（manifest，LUT 调度时间）与 ASTRA-sim 执行时间（硬件资源模型结果）是两套口径。
