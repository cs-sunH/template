# 请求 Prefill/Decode 实例映射机制 与 KV 管理策略（session_lru_recompute）

> 适用仓库：`astra-sim-face`。
> 主要代码位置：`sh_test_mesh/workload/llama2_7b_inference/face_scheduler.py`、`session_kv_manager.py`、`generate_face_trace.py`；
> 权威设计文档另见 `sh_test_mesh/README.md`。

## 0. 总体架构：两段式"规划—执行"，KV 全部驻留片上

本仓库与 `astra-sim-sh_1.0/sh_2.0` 一样采用两段式架构：所有请求→实例映射、KV 驻留/迁移/逐出决策都在 **trace 生成期（Python 离散事件规划器）** 中预先确定，固化为 54 个 rank 的静态 Chakra ET；C++ 仿真器（ASTRA-sim analytical congestion-aware）只忠实执行 ET，不做运行期重调度。

| 层 | 职责 | 关键文件 |
|---|---|---|
| Python 规划层 | 请求→实例映射、KV 二态状态机、LRU 逐出/重算决策、逐 NPU HBM 账本，并把决策编码进 ET | `face_scheduler.py`（`plan_face_requests()`）、`session_kv_manager.py`（`SessionKVCacheManager`） |
| Python ET 编码层 | 把计划转成 54 份 ET、写入 `manifest.json`、`face_lut.csv`、`kv_cache_events.csv`、`metrics_manifest.json` | `generate_face_trace.py`（`_write_face_session_lru_trace()`） |
| C++ 执行层 | 执行 ET：Roofline COMP、分析型网络 COMM、远端内存端口（本仓库实际不使用） | `astra-sim/workload/Workload.cc`、`extern/remote_memory_backend/analytical/AnalyticalRemoteMemory.{hh,cc}` |

与 sh 系列最核心的差异在于 **KV 管理策略**：

- `kv_cache_policy=session_lru_recompute`（`@sh_test_mesh/workload/llama2_7b_inference/trace_config.csv:16`）：一个 session 的完整 KV 要么整体驻留在某一个 TP 实例的本地 HBM（`RESIDENT`），要么被整 session 删除（`EVICTED`）。被删除后仍保留逻辑上下文 token 数，后续请求按该历史长度**重算** KV，而不是从远端取回。
- 硬件配置为 `NO_MEMORY_EXPANSION`（`@sh_test_mesh/hardware/face_case5_config_c.json:31-34`）：**没有远端内存层**。KV 只在片上（本地 HBM + 实例间 NoC 迁移）流动；生成器不发射任何 `MEM_LOAD/MEM_STORE` 节点，C++ 侧 `AnalyticalRemoteMemory` 在 `NO_MEMORY_EXPANSION` 下遇到内存节点会直接报错退出（`@extern/remote_memory_backend/analytical/AnalyticalRemoteMemory.cc:181-185`）。
- 因此没有 sh_2.0 的三态（`local_hbm/partial_hbm_remote/remote_memory`）、两阶段"后 K 层优先"逐出、部分恢复流水；也没有 `LocalHbmBandwidthModel`（face 仓库的 `astra-sim/workload/` 下不存在该文件）。
- 规划不落 pickle 缓存（无 `planner_cache`），每次生成 trace 都重新完整规划。

当前默认配置（`@sh_test_mesh/workload/llama2_7b_inference/trace_config.csv:12-18`）：

- 请求队列：`compute_20_first_3_minutes`，**当前实际是 2 session × 4 请求的合成队列**（prefill 172~507、decode 47~59）。
- `trace_granularity=request_aggregated`、`prefill_chunk_size=512`、`kv_reserve_context_tokens=1000000`、`record_planning_iterations=false`。
- 注意：`sh_test_mesh/README.md` 与 `test_face_scheduler.py` 描述/断言的是 compute_100 大队列（README 写 678 session/9179 请求；单测断言 136 session/2091 请求，路径在 `agent-traces/...compute_80_100_120_v1/...compute_100_trunc1M_first_3_minutes...`，当前工作区不存在）。README、单测与当前配置三者的口径目前不一致，详见 §5。

---

## 1. 请求→实例映射机制

### 1.1 统一实例，而非 PD 分离

采用 FACE 风格的 **"统一实例（unified instance）+ 逐请求动态映射"**：

- 9 个实例，每个实例同时承担 Prefill 与 Decode，没有独立的 prefill/decode 池（`@sh_test_mesh/workload/llama2_7b_inference/trace_config.csv:23-31`，9 行 `inference_group`）。
- 每个实例是 3×2 = 6 个 NPU 的连续矩形，TP=6，9 个实例铺满 9×6 = 54 NPU 网格。布局顺序为 **"中心、四邻、四角"**（instance 0 为中心 ranks 20,21,26,27,32,33；之后北、西、东、南、西北、东北、西南、东南），该顺序同时是各选择键的最终确定性 tie-break。
- `build_instances()`（`@sh_test_mesh/workload/llama2_7b_inference/face_scheduler.py:168`）校验：rank 不重叠、实心轴对齐矩形、恰好覆盖全部 NPU、等尺寸（`require_equal_size=True`，用于 KV shard 按相对 TP rank 一一配对）；由物理邻接自动生成实例邻接图 `adjacency`。
- 每个实例绑定一个 ASTRA-sim 通信组：`config_resolver.py` 的 `_prepare_comm_groups()`（`@sh_test_mesh/config_resolver.py:356`）生成 `comm_group.json`（每个 pg 的 `dimensions=[列数, 行数]=[2,3]`）。
- TP=6 不能整除 32 头，采用 whole-head shard：2 个 rank 各 6 头、4 个 rank 各 5 头（`attention_heads_by_tp_rank()` `@sh_test_mesh/workload/llama2_7b_inference/session_kv_manager.py:69`）。模型权重也按同一归属精确切分（`model_weight_shard_bytes_by_tp_rank()` `:107`）。**完整 session KV 只放在一个实例内，不跨实例切分**。

### 1.2 Prefill 实例选择：剩余 chunk 数最少 + 最久未入队 + 配置序

请求到达事件（首请求的 `session_arrival_time_ns` 或后续请求的 `inter_request_interval_ns`）触发 `select_prefill_instance()`（`@sh_test_mesh/workload/llama2_7b_inference/face_scheduler.py:569`），在 9 个实例的 PrefillQueueSnapshot 中取 `ordering_key` 最小者（`:555-566`）：

```python
(remaining_chunks,                          # 剩余 prefill+history chunk 总数升序
 -1 if last_arrival_ns is None else last_arrival_ns,  # 从未入队优先，其次最久未入队
 instance_index)                            # 实例配置顺序
```

要点：

- `remaining_chunks` 在 admission 后 = `ceil(history_recompute_tokens/p_chunk) + ceil(prefill_length/p_chunk)`（`try_admit_prefill()` `@face_scheduler.py:1382` 内，`:1442-1446`）。即**历史重算也算作 prefill 负载**参与排队，这是与 sh_1.0 "仅当前 prompt chunk 数" 不同的细节。
- 映射**先于**容量准入：到达时直接选实例并入 FCFS 队列，随后 `try_admit_prefill()` 才调用 KV manager 做容量检查。容量不足时请求被 `admission_blocked`/延迟重试（`capacity_epoch` 机制），但**不会重新选择实例**。这与 sh_2.0 的"HBM 可行性过滤后才选实例"相反，是本仓库刻意设计："KV 压力绝不改变 FACE 映射"（README §Mapping implemented by the planner）。
- `try_admit_prefill()` 依次执行 `kv_manager.prepare_history()`（历史命中/迁移/重算决策 + 容量预检）、`grow_prefill()`（把 KV 账本增长到 prefill 结束上下文）；成功后才把 `remaining_chunks` 设为含重算的总 chunk 数。

### 1.3 实例内 PD 混合执行

`start_ready_iterations()`（`@face_scheduler.py:1517`）：实例空闲时，取队首请求的 1 个 prefill chunk（若队首尚未 admission 则跳过），叠加全部 active decode 请求组成一个**混合 iteration**；查 LUT 得 `iteration_time_ns` 后排 `iteration_complete` 事件。Prefill 队列为 FCFS，decode 列表为活跃请求集合，二者在同一个 TP group 上时分复用。

### 1.4 Decode 实例选择：加权 Instance_map + LUT per-die 代价

Prefill 最后一 chunk 完成时调用 `select_decode_instance()`（`@face_scheduler.py:654`）：

1. **候选集**：`WeightedInstanceGraph.schedulable_instances()`（`:575-648`）= 实例邻接图上到 prefill 实例的最短加权距离 ≤ `schedulable_distance_limit`（`@face_scheduler.py:91`）= `d2d_bw / hbm_bw` = 4050/1640 ≈ **2.47 跳**（即 prefill 实例自身 + 一跳邻实例）。`WeightedInstanceGraph.increase_path/decrease_path`（`:630-648`）与 legacy `KVAllocator`（`:726`）保留了 FACE 的边权动态调整能力，但 **session_lru 路径不调用它们**，边权恒为 1，加权距离即实例跳数。
2. **代价**：对每个候选查 LUT（`FaceLut.lookup` `:477`），`cost = (T′ − T) / instance_size`；LUT 匹配规则为 exact `(instance_size, p_chunk, d_batch)` + 最近 `d_token`（`:487`）。LUT 由分析型 Roofline 估计生成（`estimate_iteration_time_ns()` `:315`），face 仓库不虚构论文未公开的算子 tile 数（manifest 明确 `operator_tile_sizes: not available in the paper and not fabricated`）。
3. **选择键**（`:706`）：

```python
selected = min(costs, key=lambda cost: (
    cost.per_die_delta_ns,   # 先最小 per-die 增量代价
    cost.instance_index))    # 再按实例配置顺序
```

与 sh_2.0 不同：**HBM 不参与 decode 代价计算，也没有"剩余 HBM 大者优先"的 tie-break**；容量只是硬约束。若 decode 实例 ≠ prefill 实例，规划器先做 `move_prefill_to_decode()`（`@session_kv_manager.py:1330`）容量预检（可能逐出/defer），然后 ET 层把本轮累积 KV 按相对 TP rank 经 XY 路由 NoC 迁移到 decode 实例（`_paired_transfer()` `@generate_face_trace.py:526`）。

容量不足时的处理：`try_admit_waiting_decodes()`（`@face_scheduler.py:1455`）把未准入的 decode 放进 `waiting_decode_admissions[instance]` 队列，用 `capacity_epoch` 只在容量可能变化时重试；**映射已定，只延迟不重选**。

### 1.5 Trace 生成：映射结果如何固化

入口：`generate_trace.sh` → `generate_trace.py`（main 委托 `generate_face_trace.py`，`@generate_trace.py:2050-2060`）→ `generate_face_trace.py main()`（`:2032`）：

1. `load_face_trace_config()`（`:307`）：读 `trace_config.csv`，加载请求队列；`config_resolver.materialize_runtime_configs()`（`@config_resolver.py:416`）派生 system/network/remote_memory/comm_group 四件套到 `generated/runtime_config/`。
2. `build_face_plan()`（`:463`）→ `plan_face_requests()`（`@face_scheduler.py:1807`）→ `_plan_face_session_lru_recompute()`（`:1294`）离散事件主循环。
3. `_write_face_session_lru_trace()`（`@generate_face_trace.py:1051`）按 `prefill_start_ns` 排序逐请求发射节点到对应实例各 rank 的 ET builder：
   - 首请求在 prefill 实例各 rank 写 `timer_gate`（CPU op 到达门控）；后续请求在上一 decode 实例各 rank 写 `interval_timer_gate`（`TraceBuilder.timer_gate` `@generate_trace.py:689`，duration 必须为整微秒）。
   - 历史 KV 若在别的实例，先 `_paired_transfer` 跨实例 NoC 迁移（或 1B `_emit_control_trigger` 控制消息，`:876`）；若被逐出，先发一段 `history_recompute` prefill stage（`:1164-1176`）。
   - `history_tp_ready_barrier` → `current_prefill` stage（`_emit_prefill_stage` `:911`；`request_aggregated` 模式每 rank 折叠为 17 个算子类别节点，`token_expanded` 则逐 chunk）。
   - prefill→decode KV 迁移（如需）→ decode aggregated stage（逐 token span 折叠）→ `decode_request_end_barrier`。
4. 产物：54 个 `llama2_7b_inference.{rank}.et`、`face_lut.csv`、`kv_cache_events.csv`、`manifest.json`、`metrics_manifest.json`。session_lru 的 manifest 按请求记录 prefill/decode 实例、历史动作、KV 迁移路由、逐出记录、HBM 快照与规划时间（`_session_lru_request_record` `:994`）；decode 候选代价与 LUT 明细（`_candidate_dict` `:652`）只在 legacy trace 的 manifest 中落盘，session_lru 路径为控制内存只保留请求级摘要；`kv_management` 段记录策略标识与统计（`:1355-1404`）。

实测样例（当前 4 请求 trace 的 manifest）：请求 0 prefill 实例 0（中心）→ decode 实例 1（北），prefill→decode 迁移 `116,916,224` B，每个相对 TP shard 走 3 跳 XY 路由（如 rank 20→14→8→2）；`kv_management` 统计为 `local_hit_count=1, noc_migrate_count=5, recompute_count=0, eviction_count=0`。

---

## 2. KV 管理策略：session_lru_recompute

### 2.1 二态驻留模型，无远端内存层

`SessionKVCacheManager`（`@session_kv_manager.py:363`）为每个 session 维护二态：

| 状态 | 含义 |
|---|---|
| `RESIDENT` | 完整 session KV 驻留在某一个 TP 实例的本地 HBM（`instance_index` 非空） |
| `EVICTED` | KV 已被删除；仅保留 `logical_context_tokens`（窗口内逻辑上下文）等元数据，后续请求按该长度重算 |

模块 docstring 明确：**"no remote-memory tier, partial-layer state, or capacity-driven remapping"**（`@session_kv_manager.py:1-7`）。因此：

- 逐出是**零代价删除**（manifest `delete_cost: zero`，`:1364`）：`_delete()` 只改 Python 账本，不产生任何 ET/NoC/内存节点。
- 恢复是**重算**（recompute），不是远端取回：`RECOMPUTE` 分支把历史 KV 当作 prefill 阶段重新计算（`prepare_history()` `:1020-1114`）。
- KV 跨实例移动只有一种形式：`NOC_MIGRATE`（history 迁移或 prefill→decode 迁移），以 COMM_SEND/RECV 建模。

### 2.2 逐 NPU HBM 账本与精确 TP shard

- 热层参数唯一来源 `@sh_test_mesh/hardware/face_case5_config_c.json`：每 NPU 本地 HBM 1640 GB/s、100 ns、`validation-160gib`（160 GiB）容量；`config_resolver` 把 `local-mem-bw/latency/capacity-bytes` 写入派生 `system.json`（C++ 只消费 bw/latency，容量约束只在 Python 账本生效）。
- Python 侧每个 rank 一个 `NodeHBMState`（`@session_kv_manager.py:177`）：`capacity_bytes + model_weight_bytes + resident_kv_bytes + reserved_request_bytes`，`used/remaining` 由属性计算；每次 mutation 后 `_check_invariants()`（`:1534`）校验账本守恒与容量不超。
- KV shard 按 whole-head 精确切分：`kv_cache_shard_bytes_for_tokens()`（`:142`）按 `attention_heads_by_tp_rank()` 把 `2*L*T*H*2` 字节分到 6 个 rank；权重 shard 同理精确切分（`:107`）。初始化即校验 1M token 预留 shard 能放进 160 GiB（`:400-409`）；64 GiB profile 会在构造时 fail-fast（单测 `test_64_gib_fails_fast_for_the_exact_one_million_reserve`）。
- `reserve_request_capacity()/release_request_capacity()`（`:926/:984`）是预留 API，但**当前 session_lru 主路径未调用**：实际准入走 `prepare_history → grow_prefill → move_prefill_to_decode → grow_decode` 的预检/增长，其中每次调用都会先 `enforce_watermark()` 再 `ensure_physical_fit()`。

### 2.3 LRU 整 session 逐出（most-recently-completed 保护）

触发点：任何容量相关 mutation 前的水位检查与物理适配检查。

- `enforce_watermark()`（`:821`）：要求目标实例**每个 rank** 剩余 ≥ `kv_reserve_context_tokens`（1,000,000 token）对应的 KV shard 字节；不足则逐出。
- `ensure_physical_fit()`（`:866`）：要求能放下即将新增的 shard 增量；不足则逐出。
- 候选集 `_candidate_sessions()`（`:506`）：`RESIDENT`、位于目标实例、**非 active**、有 `last_completion_ns`、且不是被保护 session；排序键 `(int(last_completion_ns), session_id)` —— 即 **"最后完成时间最旧优先"（least-recently-completed，LRU 语义）**。
- `_delete()`（`:761`）：从实例各 rank 移除 `shard_bytes`，session 置 `EVICTED`、清空 `instance_index`、记录 `evicted_at_ns/evicted_by_request_id`；事件写入 `evict_delete`。
- 保护规则：当前请求自己的 session 作为 `protected_sessions` 传入（`:1020` 起各调用点），active session 永不进候选集，**"请求永远不会成为自己的逐出受害者"**。
- 无候选时：记录 `admission_blocked`/`watermark_deferred` 事件并返回 `admission_blocked=True`，调用方稍后重试（`capacity_epoch`/`decode_admission_dirty` 去抖，避免每 token 重复刷屏）。
- 请求完成：`mark_complete()`（`:1482`）置 inactive、记录完成时间，并立即执行一次 `enforce_watermark`（完成事件 `retain_complete`，若有逐出则 `evict_delete`）。

与 sh_2.0 的差异：**没有"阶段 1 逐后 K 层、阶段 2 逐前缀"的两阶段降水位**；逐出是整 session 一次性删除。排序命名也直接叫 `last_completion_ns_then_session_id`（manifest `eviction_order`，`:1363`），不存在 FIFO 命名遗留。

### 2.4 恢复：重算（非远端取回）+ 历史跨实例迁移

`prepare_history()`（`:1020`）按 session 历史状态返回四种动作：

| 动作 | 条件 | 语义 |
|---|---|---|
| `NO_HISTORY` | 窗口内首请求 | 无历史，直接开始 prefill |
| `LOCAL_HIT` | RESIDENT 且在目标（prefill）实例 | 历史 KV 原地可用 |
| `NOC_MIGRATE` | RESIDENT 但在其他实例 | 历史 KV 先按相对 TP rank 跨实例 NoC 迁移到 prefill 实例 |
| `RECOMPUTE` | EVICTED | 按 `logical_context_tokens` 长度重新计算历史 KV |

重算的 ET 表示（`@generate_face_trace.py:1164-1176`）：

- 在 prefill 实例各 rank 先发 `history_recompute` stage（`_emit_prefill_stage`，`tokens=history_recompute_tokens`、`initial_context_tokens=0`），随后一个 `history_tp_ready_barrier`（1B All-Reduce），再发 `current_prefill` stage。
- 重算与当前 prompt 的 KV 增长由 `grow_prefill()` 一次性记账到 `prefill_context_tokens`；`effective_prefill_tokens = prefill_length + history_recompute_tokens` 写入 manifest。
- `RECOMPUTE` 只出现在"session 被逐出后再次访问"；"重计算"不是指远端取回失败后的 fallback，也不存在 sh_2.0 的 prefix/suffix 部分恢复流水。

### 2.5 事件流、审计与指标

所有 KV 状态变化以 `KVCacheEvent` 落盘到 `@generated/<trace_dir>/kv_cache_events.csv`（`_write_kv_events_csv` `@generate_face_trace.py:822`），字段包括：`event_type`（`no_history/local_hit/noc_migrate/recompute/retain_complete/evict_delete/admission_blocked/admission_retry/watermark_deferred`）、`phase`（history/prefill/decode/prefill_decode/completion/watermark）、源/目标实例、token 数、shard 字节、事件前后各 rank remaining 快照。

`manifest.json` 的 `kv_management` 段（`:1355-1404`）记录：`policy=session_lru_recompute`、`reserve_context_tokens`、`watermark_scope=per_physical_npu_exact_tp_shard`、`eviction_order=last_completion_ns_then_session_id`、`delete_cost=zero`、`remote_memory_used=false`、各类事件计数、`final_hbm_snapshots`（54 rank）与 `final_session_snapshots`。

仓库还附带 read-only 指标管线（`metrics_schema.py` / `metrics_integration.py` / `metrics_postprocess.py`）：规划器侧 `MemoryMetricsObserver` 镜像 HBM 增删（不影响决策），仿真侧 `MetricCollector` 记录节点事件，最终输出 `metrics_manifest.json`（trace_digest 校验）、服务 summary（mean/p50/p95/p99 E2E、吞吐）与 memory ledger/peaks。实测的 482 请求验证 run（`@sh_test_mesh/results/run_logs/` 中最近日志，对应 40 session 配置，与当前 4 请求合成队列不同）输出：mean E2E ≈ 34.66 s、p99 ≈ 374.89 s、drain 吞吐 ≈ 0.372 rps、`memory_actions_replayed=21258`、`memory_replay_final_state_mismatches=0`。

### 2.6 配置参数汇总

| 配置项 | 位置 | 当前值 | 作用 |
|---|---|---|---|
| `kv_cache_policy` | `trace_config.csv:16` | `session_lru_recompute` | 二态 LRU+重算策略（`legacy` 为旧多实例分片规划器，默认不用） |
| `kv_reserve_context_tokens` | `trace_config.csv:17` | 1,000,000 | 每 rank 水位线：保留 1M token KV shard 空间 |
| `prefill_chunk_size` | `trace_config.csv:15` | 512 | prefill 与历史重算的固定 chunk |
| `local_hbm_capacity_profile` | `trace_config.csv:20` | `validation-160gib` | 160 GiB/NPU；64 GiB 无法容纳 1M 预留，构造期 fail-fast |
| `hardware_config` | `trace_config.csv:19` | `hardware/face_case5_config_c.json` | HBM/D2D/容量唯一来源；`remote-memory.memory-type=NO_MEMORY_EXPANSION` |
| `trace_granularity` | `trace_config.csv:14` | `request_aggregated` | 请求级算子聚合（17 节点/rank/阶段），保持 FLOPs/字节/collective 总量 |
| `record_planning_iterations` | `trace_config.csv:18` | false | 默认不保留逐 iteration 记录（大队列内存有界） |
| `remote_operand_loads` | `trace_config.csv:22` | false | 不建模远端权重加载（权重预置本地） |
| `track-local-mem` | `system/llama2_7b_roofline_template.json:14` | 0 | 关闭 C++ `LocalMemUsageTracker` 本地内存记录 |

---

## 3. C++ 执行层（内存/通信相关）

face 仓库的 C++ 侧不参与任何 session 级决策，只执行 ET 中的三类节点：

1. **COMP（Roofline）**：`Workload::issue_comp()`（`@astra-sim/workload/Workload.cc:260-320`）按 `num_ops/tensor_size` 求操作强度，`max(计算时间, local_mem_latency + tensor_size/local_mem_bw)`（`local-mem-bw=1640 GB/s`、`local-mem-latency=100 ns`，`@astra-sim/system/Sys.cc:392-398`）。`remote_weight_bytes` 流水路径（`Workload.cc:305-319`）存在但 `remote_operand_loads=false` 未启用。
2. **COMM（分析型网络）**：COMM_SEND/RECV/COLL 走 analytical congestion-aware 后端；D2D 4050 GB/s、5 ns/跳（`network.yml`）。`timer_gate` 是带 `is_timer_op` 的 CPU COMP 节点，用 `duration_micros` 实现到达/间隔门控（`@generate_trace.py:689-717`）。
3. **MEM（远端内存）**：`Workload::issue()` 对 MEM_LOAD/STORE 分发到 `issue_remote_mem()`（`@Workload.cc:186-197, 251-257`）→ `AnalyticalRemoteMemory::issue()`。由于配置为 `NO_MEMORY_EXPANSION`，一旦出现内存节点会 `exit(1)`（`@AnalyticalRemoteMemory.cc:181-185`）；**当前 face trace 不发射内存节点**，所以该路径实际不会触发。`remote-mem-bw=512 GB/s` 仍写入 `system.json`/`remote_memory.json`，仅为兼容字段。

另有一个时间精度修复：TraceLab 到达时间在 `1.7e16~2.0e16 ns`，超出 IEEE-754 double 精确整数范围，分析型网络适配器以 `long double` 返回 ASTRA-sim 时间（`@astra-sim/network_frontend/analytical/common/CommonNetworkApi.cc:79-80`）。

---

## 4. 关键文件索引

- `sh_test_mesh/workload/llama2_7b_inference/face_scheduler.py`
  - `FaceHardware.schedulable_distance_limit` `:91` — decode 候选距离上限 `d2d_bw/hbm_bw`
  - `build_instances()` `:168` — 实例布局/校验/邻接图
  - `estimate_iteration_time_ns()` `:315`、`FaceLut` `:374` — 调度 LUT（Roofline 估计）
  - `PrefillQueueSnapshot/select_prefill_instance` `:555/:569` — prefill 选择键
  - `WeightedInstanceGraph` `:575`、`select_decode_instance` `:654` — decode 候选/代价/选择键
  - `KVAllocator` `:726` — legacy 多实例 KV 分片（默认未使用）
  - `_plan_face_session_lru_recompute()` `:1294` — 默认离散事件主循环
- `sh_test_mesh/workload/llama2_7b_inference/session_kv_manager.py`
  - `attention_heads_by_tp_rank/model_weight_shard_bytes_by_tp_rank/kv_cache_shard_bytes_for_tokens` `:69/:107/:142` — 精确 TP shard
  - `NodeHBMState` `:177` — 每 NPU HBM 账本
  - `SessionKVCacheManager` `:363`、`_candidate_sessions` `:506`、`_delete` `:761`、`enforce_watermark` `:821`、`ensure_physical_fit` `:866`、`prepare_history` `:1020`、`move_prefill_to_decode` `:1330`、`mark_complete` `:1482`
- `sh_test_mesh/workload/llama2_7b_inference/generate_face_trace.py`
  - `load_face_trace_config()` `:307`、`build_face_plan()` `:463`
  - `_paired_transfer()` `:526`（NoC KV 迁移）、`_emit_control_trigger()` `:876`、`_emit_prefill_stage()` `:911`
  - `_write_face_session_lru_trace()` `:1051`（ET 编码 + manifest）、`_build_metadata()` `:596`
- `sh_test_mesh/workload/llama2_7b_inference/generate_trace.py`
  - `TraceBuilder` `:634`（timer_gate/comm_send/comm_recv/transformer_pass）、`main()` `:2050`（委托 face 生成器）
- `sh_test_mesh/config_resolver.py` — 硬件解析（`:143`）、远端内存（`:327`）、comm groups（`:356`）、运行时四件套（`:416`）
- `astra-sim/workload/Workload.cc` — 节点分发/Roofline/远端内存分发（`:186-320`）
- `extern/remote_memory_backend/analytical/AnalyticalRemoteMemory.cc` — NO_MEMORY_EXPANSION 行为（`:181-185`）
- `astra-sim/system/Sys.cc:392-407` — local/remote mem 字段解析
- `astra-sim/network_frontend/analytical/common/CommonNetworkApi.cc:79-80` — long double 时间精度

运行方式：

- `bash sh_test_mesh/run_scripts/generate_trace.sh`：规划 + 生成 54 份 ET 与配套文件（staging 目录原子发布，`_atomic_publish_directory` `@generate_face_trace.py:859`）；
- `bash sh_test_mesh/run_scripts/run_sh_test_aware.sh`：校验 `NO_MEMORY_EXPANSION` 配置与 metrics manifest trace_digest 后启动 `AstraSim_Analytical_Congestion_Aware`；
- 单测：`python3 -m unittest sh_test_mesh/workload/llama2_7b_inference/test_face_scheduler.py`（16 个用例，覆盖 shard、LRU 删除保护、decode 代价、确定性规划等）。

---

## 5. 注意点与坑

- **README/单测与当前配置口径不一致**：README 声称默认是 678 session/9179 请求的 compute_100 队列（且引用的文件路径当前不存在）；`test_checked_in_three_minute_workload_configuration` 断言 2091 请求/136 session（当前实际是 4 请求/2 session，该用例当前失败）；而 `trace_config.csv:12` 实际指向 `compute_20_first_3_minutes` 的 4 请求合成队列。文档、测试、配置三者需要统一后再作为实验基线。
- **逐出是 least-recently-completed（LRU 语义）**：排序键为 `(last_completion_ns, session_id)`，manifest 明确 `eviction_order=last_completion_ns_then_session_id`；不是严格 FIFO，也不是最近使用（LRU 的 recency 维度是"完成时间"而非"访问时间"）。
- **删除零代价、恢复付重算代价**：被逐 session 的 KV 在 ET 中无任何痕迹（无远端写、无迁移），下次访问按历史 token 数重新 prefill。因此逐出只影响后续请求的时延/吞吐，不影响 ET 中已固化的其他请求。
- **映射先于容量准入**：Prefill 实例在到达时即选定，容量不足只会 defer/逐出，绝不重映射；这保证 FACE 映射与 KV 策略解耦，但也可能造成"到达时刻与 prefill 实际开始时刻之间出现间隙"。验证 run 的 consistency 检查曾报告 3 条 `arrival > prefill_start` 违例（queue_index 59/350/448），即该间隙被指标检查视为异常。
- **无远端内存层**：`remote_memory.json` 里的 512 GB/s 只是兼容字段；ET 中一旦出现 MEM 节点，C++ 会 abort。做实验时不要混用 sh_2.0 的远端内存 trace。
- **legacy 与 session_lru 两条规划路径并存**：`KVAllocator`（跨实例分片 + 边权动态调整 + offload）只在 `kv_cache_policy=legacy` 时启用；默认 `session_lru_recompute` 下 `WeightedInstanceGraph` 边权恒为 1，KV 也不跨实例切分。
- **规划时间与仿真时间是两套口径**：manifest 中 `planned_timing_ns` 来自 LUT/离散事件规划，仿真端 E2E 来自硬件资源模型；两者不应混算。
- **`record_planning_iterations=false`**：默认 manifest 的 `planning_iterations` 为空数组，避免大队列内存爆炸；需要逐 iteration 审计时再打开。
- **生成目录原子发布**：trace 先写入 staging，全部 rank 成功后才 rename 发布；失败不会留下半成品目录。
