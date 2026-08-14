# 请求 Prefill/Decode 实例映射机制 与 KV 冷热管理机制

> 适用仓库：`astra-sim-sh_1.0`。
> 主要代码位置：`sh_test_mesh/workload/llama2_7b_inference/face_scheduler.py`、`generate_face_trace.py`；
> 权威设计文档另见 `sh_test_mesh/README.md`。

## 0. 总体架构：两段式"规划—执行"

本仿真器把所有调度与 KV 管理决策都放在 **trace 生成期（Python 离散事件规划器）** 中预先确定，固化为 54 个 rank 的静态 Chakra ET；C++ 仿真器（ASTRA-sim）只忠实执行 ET，不做运行期重调度。

| 层 | 职责 | 关键文件 |
|---|---|---|
| Python 规划层 | 请求→实例映射、整 session 逐出/恢复/迁移决策、逐 NPU HBM 账本，并把决策编码进 ET | `face_scheduler.py`（`KVCacheManager`、`plan_face_requests()`） |
| C++ 执行层 | 执行 ET：远端内存端口 FIFO 模型、COMP Roofline；本地内存仅有被动用量记录（默认关闭），**无 HBM 带宽竞争模型** | `extern/remote_memory_backend/analytical/AnalyticalRemoteMemory.{hh,cc}`、`astra-sim/workload/Workload.cc`、`astra-sim/workload/LocalMemUsageTracker.{hh,cc}` |

因此：**改调度/逐出策略只需改 Python 规划层并重新生成 trace**；C++ 侧不涉及 session 级决策。注意本仓库规划结果**没有 pickle 缓存**，每次生成 trace 都重新完整规划。

---

## 1. Prefill / Decode 实例映射机制

### 1.1 统一实例，而非 PD 分离

采用 FACE 风格的 **"统一实例（unified instance）+ 逐请求动态映射"** 方案：

- 9 个实例，每个实例同时承担 Prefill 和 Decode；没有专门的 prefill 池或 decode 池。
- 实例布局唯一来源：`sh_test_mesh/workload/llama2_7b_inference/trace_config.csv:21-29`（9 行 `inference_group`）。每个实例是 3×2 = 6 个 NPU 的连续矩形，TP=6，9 个实例铺满 9×6 = 54 NPU 网格（网格尺寸由 `hardware/face_case5_config_c.json` 派生）。
- 布局顺序为"中心、四邻、四角"（instance 0 为中心 ranks 20,21,26,27,32,33……），该顺序同时是各选择键的最终确定性 tie-break。
- `face_scheduler.py:139-233 build_instances()` 校验：rank 不重叠、实心矩形、恰好覆盖全部 NPU、等尺寸（`require_equal_size=True`，用于 KV shard 按相对 TP rank 一一配对）、邻接图连通；由物理邻接自动生成实例邻接图 `adjacency`（`:212-227`）。
- 每个实例绑定一个 ASTRA-sim 通信组：`config_resolver.py:356-399 _prepare_comm_groups()` 生成 `comm_group.json`。
- TP=6 不能整除 32 头，采用 whole-head shard：2 个 rank 各 6 头、4 个 rank 各 5 头（`face_scheduler.py:274-283 attention_heads_by_tp_rank`）。**完整 session KV 只放在一个实例内，不跨实例切分**。

注意：生成目录名中的 `grequest_aggregated` 指 `trace_granularity=request_aggregated`（算子聚合），不是 PD 聚合；标签中 `p334-464_d34-55` 一类字段是请求队列的 prefill/decode 长度范围，`2sess_4req` 是 session/请求数（`build_trace_label`，`generate_face_trace.py:904-924`），都不是实例配比。

### 1.2 Prefill 实例选择：排队剩余 chunk 数最小 + HBM 准入

请求到达事件触发 `try_admit_request()`（`face_scheduler.py:2375-2439`）：

1. **HBM 可行性过滤**（`:2379-2396`）：`KVCacheManager.request_hbm_feasible_instances()`（`:1232-1271`）逐实例检查最终 KV 分片能否在回收已完成非活跃 session 后放下；暂不可行则进 `pending_admissions` 等待事件驱动重试；若连空实例都永远放不下（`request_hbm_eventually_feasible_instances()`，`:1273-1293`）则直接报错。
2. **负载均衡**：`select_prefill_instance()`（`:627-648`）在 HBM 可行实例中取 `ordering_key` 最小者（`PrefillQueueSnapshot.ordering_key`，`:618-624`）：

```python
(self.remaining_chunks,                                  # 排队中的剩余 prefill chunk 总数升序
 -1 if self.last_arrival_ns is None else self.last_arrival_ns,  # 从未用过优先，其次最久未入队
 self.instance_index)                                    # 实例配置顺序
```

`remaining_chunks` 由 `queue_snapshot()`（`:2368-2373`）对该实例 FCFS 队列中全部请求的剩余 chunk 求和。**本仓库的选择键只计量排队 prefill chunk 数，不含 decode 负载项，也不做 Roofline 服务时间折算**；chunk 固定 `prefill_chunk_size = 512`（`trace_config.csv:13`，经 `build_face_plan()` `:755` 传入；不传时回退为平均 prefill 长度，`:2324-2327`）。本仓库**没有 session 亲和强制**（不存在部分驻留状态，见 §2.2），历史 KV 位置不影响选实例，只影响后续传输。

准入后：`reserve_request_capacity()`（`:1295-1339`）按最终 KV 预留容量（可能触发逐出，`protected_session_id` 保证请求不会成为自己的逐出受害者），`prepare_prefill()`（`:1798-1903`）按历史 KV 位置做三种处理——本地命中 / 跨实例 NoC 迁移 / 远端全量载入——然后请求入所选实例 FCFS 队列队尾并记录 `last_arrival_ns`。

### 1.3 实例内 PD 混合执行

`start_ready_iterations()`（`:2449-2498`）：实例空闲时，取队首请求的 1 个 prefill chunk + 全部 active decode 请求组成一个**混合 iteration**，查 LUT 得 `iteration_time_ns`，排 `iteration_complete` 事件（优先级 0，高于 arrival 的优先级 1，`:2347-2349` 注释、`:2363`、`:2493-2498`）。即同一 TP group 上 PD 以 batch 形式时分复用。实例忙闲用布尔 `busy` 标记（`_InstanceRuntime`，`:2246-2253`），无"当前 iteration 剩余比例"折算。

### 1.4 Decode 实例选择：加权 Instance_map + LUT per-die 代价

Prefill 最后一 chunk 完成时（iteration_complete 处理内，`:2523-2594`）调用 `select_decode_instance()`（`:730-798`）：

1. **候选集**：`WeightedInstanceGraph.schedulable_instances()`（`:693-699`）= 实例邻接图上到 prefill 实例的最短加权距离 ≤ `schedulable_distance_limit`（`:61-63`）= `d2d_bw / hbm_bw`。按当前硬件为 4050/1640 ≈ **2.47**，边权恒为 1 即 **≤ 2 个实例跳**：从中心实例出发全部 9 个实例可达，从角实例（如 5 号）出发可达 6 个（自身 + 1 跳 {1,2} + 2 跳 {0,6,7}）。
2. **代价**：对每个候选查 LUT（`FaceLut.lookup`，`:569-583`；`instance_size/p_chunk/d_batch` 精确匹配、`d_token` 取最近档），`cost = (T_prime − T) / instance_size`（`:778-786`）。
3. **HBM 硬约束**：`decode_hbm_feasible_instances()`（`:1401-1417`）过滤不可行候选（`:789-793`）。
4. **选择键**（`:794-797`）：

```python
selected = min(feasible_costs, key=lambda cost: (
    cost.per_die_delta_ns,      # 最小 per-die 增量代价
    cost.instance_index))       # 实例配置顺序 tie-break
```

HBM 只做硬约束，**没有"HBM 剩余大者优先"的并列 tie-break**；映射后不再因迁移开销改派。随后依次：`move_request_capacity_reservation()`（`:1341-1388`）迁移容量预留、`move_prefill_to_decode()`（`:1967-2009`）把本轮累积 KV 按相对 TP rank 经 XY 最少跳数 NoC 迁到 decode 实例（`deterministic_xy_route`，`:385-401`）、`expand_decode()`（`:2011-2028`）按最终长度扩容、`allocation_for_session()`（`:2030-2053`）生成单件本地分配、释放预留并加入 decode 实例的 `active_decode`（`:2591-2594`）。请求完成后 session 权威位置为 decode 实例。

补充：`WeightedInstanceGraph.increase_path/decrease_path`（`:701-717`）与 `KVAllocator`（`:818-907`）实现了 FACE 的边权动态调整与 KV 跨实例分片，但**实际规划路径中没有调用**（仅 `test_face_scheduler.py:1147,1179,1198` 引用）；边权恒为 1，manifest 的 `final_edge_weights` 全为 1。

### 1.5 Trace 生成：映射结果如何固化

流程入口：`run_scripts/generate_trace.sh` → `generate_trace.py main()`（`:2190-2198`，仅做分发）→ `generate_face_trace.py main()`（`:2364-2401`）：

1. `load_face_trace_config()`（`:584-723`）：读 `trace_config.csv`；加载请求队列 CSV（队列由 `trace_config.csv:12` 指定，见 §4 的缺失说明）；`config_resolver.materialize_runtime_configs()`（`config_resolver.py:416-451`）派生 system/network/remote_memory/comm_group 四件套到 `generated/runtime_config/face_case5_config_c__validation-160gib__edge_remote_memory_pool/`。
2. `build_face_plan()`（`:742-756`）→ `plan_face_requests()`（`face_scheduler.py:2299-2770`）；无缓存，每次重新规划。
3. `write_face_trace()`（`generate_face_trace.py:1608-2299`）：先按 `order_plans_for_static_emission()`（`:787-901`，session 顺序 + KV store 因果拓扑 + `prefill_start_ns` 堆排序）确定整请求发射顺序，再逐请求向 **prefill_group.ranks 与 decode_group.ranks 各自的 ET builder** 发射节点——实例映射结果直接体现在"节点写进哪些 rank 的 `.et` 文件、collective 用哪个 pg_name"：
   - 首请求在 prefill 实例各 rank 写 `timer_gate`（到达门控，时刻 = `admission_time_ns`，`:1695-1706`）；后续请求的间隔门控 = 上一请求 decode 完成节点 + `inter_request_interval_ns + hbm_wait_ns`（`:2019-2046`）；
   - 历史 KV 迁移/逐出以 COMM_SEND/RECV、MEM_LOAD/MEM_STORE 节点表示（`_emit_kv_transfer`，`:1086-1346`）；
   - 每阶段前后各发一次 1B all_reduce 的 TP readiness barrier（`_emit_tp_readiness_barrier`，`:1349-1371`）；每请求的固定阶段序：history_evictions → history_transfer → prefill_evictions → prefill 屏障 → prefill → decode_evictions → prefill→decode 迁移 → decode 屏障 → decode → completion_evictions（`manifest["kv_management"]["transfer_order_per_request"]`，`:2240-2251`）；
   - Prefill/Decode 按 `token_expanded`（逐 chunk/token）或 `request_aggregated`（聚合，当前配置）发射（`:1854-1920`、`:1951-2000`）；
   - Decode 在 `decode_group.ranks` 上发射，pg 用 `decode_group.pg_name`。
4. 产物：54 个 `llama2_7b_inference.{rank}.et`、`face_lut.csv`（`:1618-1619`）、`manifest.json`（`:2297-2298`）。manifest 完整记录映射审计信息：`instances[].role="unified_prefill_decode"`（`:2167`）、`prefill_queue_policy`（`:2172-2177`）、`decode_policy`（`:2178-2184`）、每请求的 `prefill_assignment`/`decode_assignment`（含候选代价，`_request_plan_dict` `:1543-1565`）、KV 迁移路由、HBM 快照等。

---

## 2. KV 冷热管理机制

### 2.1 冷热分层架构

**热层——每 NPU 本地 HBM**（唯一人工维护来源 `sh_test_mesh/hardware/face_case5_config_c.json`）：

- 带宽 1640 GB/s、延迟 100 ns；
- 容量走 profile：`paper-64gib`（64 GiB）与 `validation-160gib`（160 GiB，当前 trace 选用，`trace_config.csv:18`）；
- `config_resolver.py:431-434` 校验后把 `local-mem-bw / local-mem-latency / local-mem-capacity-bytes` 写入派生 `system.json`；
- Python 侧账本：`KVCacheManager.__init__`（`face_scheduler.py:1031-1101`）为每个 rank 建 `NodeHBMState`（`:921-946`：capacity、按 TP rank 精确切分的 model_weight、kv_cache、remaining），所有操作后 `_check_invariants()`（`:1451-1502`）校验守恒。

**冷层——mesh 边界挂接的远端内存池**（`face_case5_config_c.json:31-37`）：

- `memory-type: PER_NPU_MEMORY_EXPANSION`、512 GB/s、100 ns、`npu-selection: mesh-boundary`、逻辑池 `unified-kv-cache-pool`；
- 派生的 `remote_memory.json`：26 个边界 rank 各一个端口（9×6 mesh 边界由 `config_resolver.py:342-349` 动态推导，写入 `npu-ids`）；
- C++ 模型 `AnalyticalRemoteMemory`：每端口一条严格 FIFO 队列（`AnalyticalRemoteMemory.cc:181-186` 入队、`:205-218` 完成回调），端口间并行；**不限制远端池总容量，也不把 26 个端口汇聚成全局带宽瓶颈**；
- KV 路由：非边界 rank 经最近边界端口存取，端口选择键 `(Manhattan hops, edge rank)`（`physical_edge_ranks` `face_scheduler.py:363-376`、`nearest_edge_rank` `:404-415`）。

### 2.2 两态驻留状态机

`KVCacheManager`（`face_scheduler.py:1025-2110`）为每个 session 只维护**两态**（`:1028-1029`）：

| 状态 | 含义 |
|---|---|
| `local_hbm` | 全部层完整 KV 在某实例的本地 HBM |
| `remote_memory` | 全部层完整 KV 在远端池 |

**本仓库没有 `partial_hbm_remote` 部分驻留状态**：KV 逐出与恢复都以"整个 session 全部 32 层"为最小粒度，层区间切分、前缀保留等机制均不存在。

### 2.3 单阶段整 session 逐出（最久完成优先）

触发点——`_ensure_capacity()`（`face_scheduler.py:1745-1796`）在以下路径被调用：

- 请求准入预留最终 KV 容量：`reserve_request_capacity`（`:1323-1330`）；
- Decode 预留迁移：`move_request_capacity_reservation`（`:1372-1379`）；
- 历史 KV 跨实例迁移/远端恢复前：`prepare_prefill`（`:1858-1865`、`:1883-1890`）；
- Prefill/Decode KV 增长：`_expand_local_session`（`:1933-1940`，经 `expand_prefill`/`expand_decode`）；
- Prefill→Decode 迁移：`move_prefill_to_decode`（`:1989-1996`）。

核心算法（`:1756-1796`）：只要仍有 rank 有效剩余不足（`_insufficient_ranks`，`:1721-1743`），就取最老的"全本地、已完成、非 active"session（`_completed_local_candidates`，`:1677-1695`），调用 `_evict_session`（`:1697-1719`）把**整个 session 全部层** `remote_store` 到远端，session 转为 `remote_memory`；无候选可用时直接抛 `ValueError`（`:1769-1787`）。**没有"先逐出后 K 层、再逐出整 session"的两阶段设计**。

排序键为 `(last_completion_ns, session_id)`（`:1689-1694`）——即 **"最后完成时间最旧优先"，效果上等价于 LRU（least-recently-completed）**。注意本仓库的落盘命名仍沿用 **FIFO**：manifest.json 的 `kv_management.policy = node_private_hbm_complete_session_fifo_remote_pool`、`fifo_key = last_request_completion_ns_then_session_id`（`generate_face_trace.py:2226-2239`），`mapping_strategy` 写 "complete-session FIFO remote stores"（`:2096-2100`），trace 目录标签含 `kvfifo_r1000000`（`:918`）。

**请求完成后的水位线检查**不走 `_ensure_capacity`：`complete_request`（`:2096-2110`，规划主循环内联调用 `:2626-2641`）→ `enforce_reserve`（`:2063-2094`）——要求每 rank 剩余 ≥ `kv_reserve_context_tokens`（默认 1,000,000 token，`trace_config.csv:16`）按 whole-head 精确切分的 KV 分片字节（`reserve_bytes_by_tp_rank`，`:1057-1059`）；不足时按同样顺序整 session 逐出（`reason="reserve_threshold"`）；仍不满足时记录 `reserve_unmet_ranks` 返回而不死循环（`:2092`）。

保护规则：active session（正在 Prefill/Decode/恢复中，`active=True`）与从未完成的 session（`last_completion_ns is None`）不进候选集（`:1681-1688`）；准入预留时 `protected_session_id` 把当前 session 排除在候选外（`:1329`、`:1763-1768`），保证"请求永远不会成为自己的逐出受害者"。

### 2.4 恢复：远端全量取回 + 跨实例 NoC 迁移（不重计算、无流水）

- 恢复目标实例**不做亲和强制**：后续请求按 §1.2 正常选实例，`prepare_prefill()` 再按历史位置处理——同实例 `local_hit`（`:1843-1856`）；历史在**其他实例**则整 KV `noc_migrate` 过来（`:1858-1879`）；历史在**远端**则全层 `remote_load` 到所选实例（`:1881-1903`）。恢复后状态一律归一化为 `local_hbm`。
- **冷层 KV 的恢复一律走远端读取，不重算**。本仓库也不存在"声明 prefix 超过已存 KV 需钳位重算"的逻辑：`_validate_and_expand_requests`（`:2256-2296`）按上一 turn 最终上下文严格累加，请求历史与保存的 KV 始终一致（不一致在 `prepare_prefill` `:1834-1841` 直接报错）。
- ET 层的传输编码（`generate_face_trace.py _emit_kv_transfer`，`:1086-1346`）：
  - `remote_store`（`:1202-1275`）：源 rank 即边界端口则直接 `MEM_STORE`；否则 `COMM_SEND → edge COMM_RECV → edge MEM_STORE → 1B ACK`，源端释放依赖 ACK（`source_release_dependency`）；可由触发方控制 rank 先发 1B trigger（`_emit_transfer_trigger`，`:1036-1083`）；
  - `remote_load`（`:1277-1342`）：控制 rank（上一请求 decode 实例的对应 rank）过 `timer_gate` 后，若控制 rank ≠ 边界端口则先发 1B 请求 → `edge MEM_LOAD → edge COMM_SEND → target COMM_RECV`；**没有目标端本地 HBM restore DMA 节点，也没有回 ACK**；
  - **没有部分恢复流水**：由于不存在部分驻留状态，2.0 版"前缀计算与后缀恢复并行"的 prefix/suffix 两段式发射在本仓库不存在，远端载入整体发生在 prefill 屏障之前。

### 2.5 冷层访问的代价建模（C++ 执行层）

远端读取代价分两段计入仿真：

1. **远端端口访问**：`AnalyticalRemoteMemory::get_remote_mem_runtime()`（`AnalyticalRemoteMemory.cc:220-224`）：
   `runtime = remote_mem_latency + floor(tensor_size / remote_mem_bw)`（100 ns + 字节/512 GB/s，带宽项向下截断）。同端口事务 FIFO 串行排队。ET 中所有 `MEM_LOAD/MEM_STORE` 节点都由 `Workload::issue` 分发（`Workload.cc:185-187`）到 `issue_remote_mem`（`:244-251`）进入该模型。
2. **NoC 段**：边界端口与目标 rank 间的 `COMM_SEND/RECV` 走分析型网络后端（D2D 4050 GB/s、5 ns/跳，由派生 `network.yml` 提供，`config_resolver.py:402-413`）。

**目标端没有第三段代价**：数据经 `COMM_RECV` 到达即视为落位，不建模本地 HBM 恢复 DMA，更不与推理 COMP 竞争 HBM 带宽。本仓库**没有** `LocalHbmBandwidthModel`；`astra-sim/workload/LocalMemUsageTracker.{hh,cc}` 只是被动的 tensor 读写用量记录器（recordStart/recordEnd、峰值用量与 timeline 导出，`LocalMemUsageTracker.hh:28-56`），由 system 配置 `track-local-mem` 门控（`Sys.cc:433-438`，代码默认 false，当前模板 `track-local-mem: 0` 亦为关闭），只产出统计、不影响任何节点时长（接入点 `Workload.cc:179-181`、`:620-630`）。

> 补充：system.json 的 `remote-mem-bw/latency` 还被 Roofline 路径用于带 `remote_weight_bytes` 属性的 COMP 节点远端操作数流水（`Workload.cc:289-306`，`pipeline-tile-fraction` 解析于 `Sys.cc:406-409`），但生成器仅在 `remote_operand_loads=true` 时才写该属性（`generate_trace.py:749`），当前 `trace_config.csv:20` 为 false，未启用。COMP 的常规 Roofline 在 `local_mem_latency > 0` 时取 max(纯计算时间, 本地 HBM 延迟+字节/带宽)（`Workload.cc:279-286`）。

### 2.6 配置参数汇总

| 配置项 | 位置 | 当前值 | 作用 |
|---|---|---|---|
| `kv_reserve_context_tokens` | `trace_config.csv:16` | 1,000,000 | 完成后水位线：每 rank 预留 1M token KV 分片空间 |
| `local_hbm_capacity_profile` | `trace_config.csv:18` | `validation-160gib` | HBM 容量 profile（64/160 GiB） |
| `hardware_config` | `trace_config.csv:17` | `hardware/face_case5_config_c.json` | 热/冷层带宽、延迟、容量、边界端口唯一来源 |
| `request_queue_csv` | `trace_config.csv:12` | `compute_20_first_20_minutes` 队列 | 1200s 窗口请求队列；**当前工作区文件缺失**，见 §4 |
| `prefill_chunk_size` | `trace_config.csv:13` | 512 | 每 prefill chunk 固定 token 数 |
| `trace_granularity` | `trace_config.csv:15` | `request_aggregated` | 算子聚合粒度 |
| `request_queue_session_limit` | `trace_config.csv:14` | 0 | 0 = 取队列全部 session |
| `roofline-enabled` | `system/llama2_7b_roofline_template.json:12` | 1 | COMP 走 Roofline（`Sys.cc:411-416`） |
| `track-local-mem` | `system/llama2_7b_roofline_template.json:14` | 0 | 关闭 C++ 本地内存被动记录（`Sys.cc:433-438`） |
| `remote-mem-bw` / `remote-mem-latency` | 派生 `system.json` 与 `remote_memory.json` | 512 GB/s / 100 ns | 远端端口代价（`Sys.cc:399-404`、`AnalyticalRemoteMemory.cc:105-120`） |
| `local-mem-bw` / `local-mem-latency` / `local-mem-capacity-bytes` | 派生 `system.json` | 1640 GB/s / 100 ns / 160 GiB | 热层参数；C++ 只解析 bw/latency（`Sys.cc:392-397`），capacity 写入后 C++ 端不消费（容量约束只在 Python 账本生效） |
| `remote_operand_loads` | `trace_config.csv:20` | false | 不建模远端权重加载 |

策略标识的落盘位置：本仓库**没有** `kv_eviction_policy` 之类的 Chakra GlobalMetadata 策略属性（`_build_metadata`，`generate_face_trace.py:1374-1446`，只写 `kv_reserve_context_tokens` 与 `kv_reserve_total_bytes_all_tp_ranks`，`:1434-1443`）；策略名只出现在 manifest.json 的 `kv_management` 段（`policy=node_private_hbm_complete_session_fifo_remote_pool`、`fifo_key=last_request_completion_ns_then_session_id` 等，`:2226-2264`）与 trace 目录标签（`kvfifo_r1000000`，`:918`）。

运行：

- `sh_test_mesh/run_scripts/generate_trace.sh [-j JOBS]`：Python 规划 + 生成 54 份 ET + manifest；
- `sh_test_mesh/run_scripts/run_sh_test_aware.sh`：校验 `REMOTE_MEMORY` 配置后以 `--remote-memory-configuration` 启动 `AstraSim_Analytical_Congestion_Aware`（`:82-108`）；
- 单测：`sh_test_mesh/workload/llama2_7b_inference/test_face_scheduler.py`（26 个用例，覆盖逐出、水位线、迁移、选择键）；另 `sh_test_mesh/tests/test_config_resolver.py`（4 个用例）。当前运行状态见 §4。

---

## 3. 关键文件索引

- `sh_test_mesh/workload/llama2_7b_inference/face_scheduler.py`
  - `build_instances()` `:139-233` — 实例布局/校验/邻接图
  - `PrefillQueueSnapshot` + `select_prefill_instance()` `:612-648` — prefill 选择键（剩余 chunk 数）
  - `select_decode_instance()` `:730-798` — decode 候选/代价/选择键
  - `KVCacheManager` `:1025-2110` — HBM 账本、两态机、单阶段整 session 逐出、恢复、水位线
  - `plan_face_requests()` `:2299-2770` — 离散事件主循环
- `sh_test_mesh/workload/llama2_7b_inference/generate_face_trace.py:1086-1346, 1608-2299` — KV 迁移/逐出/恢复事件的 ET 编码与请求发射
- `sh_test_mesh/workload/llama2_7b_inference/generate_trace.py:266-399, 410-499` — remote_memory 校验、请求队列加载（缺失时自动生成默认队列）
- `sh_test_mesh/config_resolver.py:327-353, 356-399, 416-451` — 边界端口推导、通信组、四件套派生
- `extern/remote_memory_backend/analytical/AnalyticalRemoteMemory.{hh,cc}` — 远端端口 FIFO 与代价（`.cc:220-224`）
- `astra-sim/workload/Workload.cc:185-187, 244-251, 253-331` — MEM 节点分发、远端访问与 COMP Roofline
- `astra-sim/workload/LocalMemUsageTracker.{hh,cc}` — 被动本地内存用量记录（默认关闭，不计时）
- `astra-sim/system/Sys.cc:388-445` — 内存相关 system 字段解析（无 `local-mem-capacity-bytes` 消费）

## 4. 注意点与坑

- **逐出语义与命名不一致**：manifest/目录标签/README 沿用 "FIFO" 命名，实际排序键是 `(last_completion_ns, session_id)`，即"最久未完成优先"（least-recently-completed，类 LRU 的 recency 语义），不是严格到达 FIFO。
- **配置的请求队列文件缺失**：`trace_config.csv:12` 指向 `agent-traces/TraceLab_ASTRA_WSC_empirical_arrival_compute_20_50_v1/derived/compute_20_first_20_minutes/astra_compute_20_first_20_minutes_request_queue.csv`（1200s 窗口），该文件在当前工作区**不存在**（同目录只有 `compute_20_first_10_minutes` 版本，且其内容为 4 行/2 session 的合成数据）。缺失时 `load_request_queue()` 会在该路径**自动生成** 2 session × 2 请求的合成队列（`generate_trace.py:410-412` → `create_default_request_queue` `:376-399`；`random.Random()` 无种子，每次内容不同，prefill∈[128,512]、decode∈[32,64]、到达时刻全 0），切勿把生成物当作真实 TraceLab 数据。
- **两个单测因此失败**：`test_face_scheduler.py` 共 26 个用例，实测 24 过 2 失败——`test_checked_in_astra_compute_selection_uses_three_minute_window`（`:249`，断言配置指向 3 分钟窗口队列、2091 请求/136 session，与当前 20 分钟窗口配置不符）与 `test_shell_config_does_not_build_full_face_plan`（`:188`，期望 `REQUEST_COUNT=2091`，因队列缺失被自动合成数据顶为 4）。两者均属请求队列配置/数据问题，与本文档的实例映射、KV 管理机制无关；其余覆盖逐出/水位线/迁移/选择键的用例全部通过。`test_config_resolver.py` 4 个用例全过。
- 冷层容量不建模（视为无限），26 个边界端口之间无全局带宽汇聚；读写同价。
- 远端带宽 512 GB/s 在 `AnalyticalRemoteMemory` 中按"每 ns 512 byte"十进制使用，传输时间带宽项向下截断；`Sys.cc:399-401` 的 `remote_mem_bw`（×1e9）只服务 COMP 的 `remote_weight_bytes` 流水路径，两者不要混用。
- 目标端无 HBM restore DMA 与带宽竞争建模（无 `LocalHbmBandwidthModel`）；KV 恢复的代价 = 远端端口 + NoC 两段。
- 所有映射/迁移/逐出决策在 trace 生成时确定，仿真运行期不会因实际拥塞重调度（静态 ET 近似）。
- 每 rank ET 的 GlobalMetadata `hardware_case="A_FACE_CASE3"`（`generate_face_trace.py:1391`）是遗留标签，与实际使用的 `face_case5_config_c` 不符，仅作元数据展示。
- `sh_test_mesh/README.md` 引用的 `sh_test_mesh/remote_memory/edge_remote_memory_pool.json` 路径在仓库中不存在；remote_memory 配置实际由 `config_resolver.py` 派生到 `generated/runtime_config/`。
- 实例规划时间（manifest `planned_timing_ns`，LUT 口径）与 ASTRA-sim 执行时间（硬件资源模型结果）是两套口径。
