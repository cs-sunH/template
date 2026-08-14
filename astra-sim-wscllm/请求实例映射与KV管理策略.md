# 请求实例映射机制 与 KV 管理策略（WSC-LLM PD 分离版）

> 适用仓库：`astra-sim-wscllm`。
> 主要代码位置：`sh_test_mesh/workload/llama2_7b_inference/wsc_llm_scheduler.py`、`session_kv_manager.py`、`generate_wsc_llm_trace.py`；
> 权威设计说明见 `sh_test_mesh/README.md`。

## 0. 总体架构：两段式"规划—执行"

与 `astra-sim-sh_2.0` 一致，本仿真器把全部调度与 KV 管理决策放在 **trace 生成期（Python 离散事件规划器）** 预先确定，固化为 54 个 rank 的静态 Chakra ET；C++ 仿真器只忠实执行 ET，不做运行期重调度。

| 层 | 职责 | 关键文件 |
|---|---|---|
| Python 规划层 | WSC-LLM 风格的 Prefill/Decode 分离规划：请求→实例映射、离线静态 P→D 路由、会话 KV 驻留/删除/迁移/重计算决策、逐 NPU HBM 账本，并把决策编码进 ET | `wsc_llm_scheduler.py`（`plan_wsc_llm_requests()`、`_plan_wsc_llm_session_lru_recompute()`）、`session_kv_manager.py`（`SessionKVCacheManager`）、`generate_wsc_llm_trace.py`（`_write_wsc_session_lru_trace()`） |
| C++ 执行层 | 执行 ET：Roofline 计算、分析型网络（D2D）、COMM_SEND/RECV 传输；**无远端内存访问、无本地 HBM 带宽竞争模型** | `astra-sim/workload/Workload.cc`、`extern/remote_memory_backend/analytical/AnalyticalRemoteMemory.cc` |

因此：**改调度/逐出策略只需改 Python 规划层并重新生成 trace**；C++ 侧不涉及 session 级决策。

注意：`generate_trace.sh` 调用的 `generate_trace.py` 只是一个分发入口，真正的主流程在 `generate_wsc_llm_trace.py main()`（`generate_trace.py:2050-2058` 直接转调）。本仓库没有 sh_2.0 的 planner pickle 缓存，每次生成都重新完整规划。

---

## 1. Prefill / Decode 实例映射机制

### 1.1 专用实例，PD 分离（6P:3D），而非统一实例

与 FACE 的"统一实例（unified instance）+ 逐请求动态映射"相反，本仓库采用 WSC-LLM 风格的 **"专用实例（dedicated instance）"**：

- 9 个实例，每个实例是 `3×2 = 6` 个 NPU 的连续矩形，TP=6，铺满 9×6 = 54 NPU 网格（网格尺寸由 `hardware/face_case5_config_c.json` 派生）。
- **6 个 Prefill-only 实例**（上、下两行）与 **3 个 Decode-only 实例**（中间行）各司其职；没有任何实例同时承担两个阶段。
- 实例角色与 rank 成员的唯一来源：`trace_config.csv` 的 9 行 `inference_group`，其中 `phase_role` 列显式声明 `prefill` / `decode`。

| 配置序号 | 角色 | 物理位置 | Ranks |
|---:|---|---|---|
| 0 | Decode | 中央（行 3-5，列 2-3） | 20, 21, 26, 27, 32, 33 |
| 1 | Prefill | 北（行 0-2，列 2-3） | 2, 3, 8, 9, 14, 15 |
| 2 | Decode | 西中（行 3-5，列 0-1） | 18, 19, 24, 25, 30, 31 |
| 3 | Decode | 东中（行 3-5，列 4-5） | 22, 23, 28, 29, 34, 35 |
| 4 | Prefill | 南（行 6-8，列 2-3） | 38, 39, 44, 45, 50, 51 |
| 5 | Prefill | 西北（行 0-2，列 0-1） | 0, 1, 6, 7, 12, 13 |
| 6 | Prefill | 东北（行 0-2，列 4-5） | 4, 5, 10, 11, 16, 17 |
| 7 | Prefill | 西南（行 6-8，列 0-1） | 36, 37, 42, 43, 48, 49 |
| 8 | Prefill | 东南（行 6-8，列 4-5） | 40, 41, 46, 47, 52, 53 |

`build_instances()`（`wsc_llm_scheduler.py:188-317`）执行一组强校验：

- rank 不重叠、每个实例是实心轴对齐矩形、恰好覆盖全部 54 个 NPU、等尺寸（`require_equal_size=True`，保证 KV 按相对 TP rank 一一配对）；
- 由物理邻接自动生成实例邻接图 `adjacency`；
- 要求 Prefill 与 Decode 角色都存在，且 **Decode 实例中心到晶圆中心的最大曼哈顿距离 ≤ Prefill 实例中心到晶圆中心的最小距离**（即 decode 中心优先，`wsc_llm_scheduler.py:305-312`）。

每个实例绑定一个 ASTRA-sim 通信组：`config_resolver.py:356-401 _prepare_comm_groups()` 生成 `comm_group.json`，维度为 `[列数, 行数] = [2, 3]`。

TP=6 不能整除 32 头，KV 与权重均采用 **whole-head shard**：`attention_heads_by_tp_rank()`（`session_kv_manager.py:73-86`）把 32 个头分成 `6, 6, 5, 5, 5, 5`，2 个 rank 各 6 头、4 个 rank 各 5 头；**完整 session KV 只放在一个实例内，不跨实例切分**。

> 说明：6P:3D = 2:1 是确定性的仿真假设而非最优声明。README 说明这是依据论文 Fig. 7（NP=8, ND=4）与 Fig. 8（24:12 die）的 2:1 比例推导，同时保留用户要求的 6-NPU 实例尺寸，而不是复现论文的离线实例大小/TP 搜索。

### 1.2 Prefill 实例选择：最少排队请求数 + 配置顺序

请求到达事件（`_plan_wsc_llm_session_lru_recompute` 的 `arrival` 分支）只做一件事：调用 `select_prefill_instance()`（`wsc_llm_scheduler.py:826-831`）在所有 Prefill 实例中取 `ordering_key` 最小者，然后把请求追加到该实例 FCFS 队列队尾：

```python
PrefillQueueSnapshot.ordering_key = (request_count, instance_index)
```

其中 `request_count` 是当前队列长度（**包括正在执行/FCFS 队首的请求**，`wsc_llm_scheduler.py:1739-1744`）。即 WSC-LLM 论文中的 "least occupied queue" 被解释为 **`QP` 中的请求数（含 active FCFS head）**，配置顺序是最终 tie-break。选择时：

- **不看 HBM 容量**（没有 sh_2.0 的 `request_hbm_feasible_instances` 预过滤）；
- **不看历史 KV 位置**（没有 session 亲和强制）；
- **不做负载时间折算**（没有 Roofline 服务时间负载键）。

真正的容量把关发生在 **准入时**，而不是选择时。`start_ready_iterations()`（`:1915`）取队首请求后调用 `try_admit_prefill()`（`:1746-1849`），按顺序：

1. `reserve_request_capacity()`（`session_kv_manager.py:930-987`）：在**静态 decode 目标**上预留本请求最终 KV 分片空间（可能触发逐出）；失败则本次不启动 iteration。
2. `prepare_history()`（`:1024-1257`）：决定历史 KV 是本地命中 / NoC 迁移 / 重计算，并把历史 KV 放到 Prefill 实例上。
3. `grow_prefill()`（`:1314-1332`）：按 `prefill_context_tokens` 增长 Prefill 实例上的 KV 账本。

任一步因容量不足失败时，队首请求保持原位，`capacity_epoch` 去重机制（`:1728-1737`、`prefill_attempt_epoch`）保证只在容量状态变化后重试；**FCFS 队首阻塞会连带阻塞其后所有请求**（WSC-LLM Algorithm 2 的 head-of-line 语义，`wsc_llm_scheduler.py:1916-1948` 中 `if not try_admit_prefill(...): continue`）。

### 1.3 Decode 实例选择：离线静态映射，无运行期重选

Decode 目标**不是在运行期从负载/LUT 增量代价动态选出的**，而是在规划开始前一次性构建：

`build_static_pd_mapping()`（`wsc_llm_scheduler.py:438-540`）：

1. 对每个 Prefill 实例，在实例邻接图上找最近的 Decode 实例，枚举所有等长最短实例路径（`InstanceGraph.all_shortest_paths`，`:357-401`）；
2. 全局枚举组合，按目标函数取全局最小：

```python
objective = (total_hops, shared_edge_occurrences, path_signature)
```

即先最小总跳数，再最小共享边出现次数，最后路径签名确定性排序（`:505-514`）。
`alpha >= 1` 只作为 `adjusted_transfer_cost = total_hops + (alpha-1)*shared_occurrences` 的元数据保留（`:535-539`），不参与当前布局的选择。

当前布局恰好得到 **6 条一跳、边不相交** 的路由：

```text
P1 -> D0    P4 -> D0
P5 -> D2    P7 -> D2
P6 -> D3    P8 -> D3
```

每个请求在 arrival 时即固定 `static_route` 与 `decode_instance_index`（`wsc_llm_scheduler.py:2075-2090`），之后：

- 不再因 decode 负载/队列长度/容量重选目标（manifest 中 `runtime_decode_reselection: false`）；
- 论文中"least occupied queue"只用于 Prefill 选择，decode 端没有实现运行期最短队列调度；
- Prefill 完成只是把请求交给固定 decode 队列（`waiting_decode_admissions` + `try_admit_waiting_decodes`，`:1851-1913`），decode 队列按 FCFS 顺序追加进 continuous batch。

### 1.4 实例内阶段分离执行（无 PD 混合 iteration）

`start_ready_iterations()`（`:1915-2018`）中，每个实例每个规划 iteration 只执行一种阶段：

- **Prefill-only 实例**：空闲时取队首请求的 1 个 chunk（history 重计算 chunk 或当前 prompt chunk，固定 `PREFILL_CHUNK_SIZE = 512`），查 LUT 得 `iteration_time_ns`；
- **Decode-only 实例**：空闲时让 active decode 批（FCFS 顺序）**全体前进 1 个 token**，查 LUT 用 `d_batch = len(active_decode)`、`d_token = max(current_decode_token)`；
- 任何 iteration 要么是 Prefill 要么是 Decode，与 FACE"1 prefill chunk + 全部 active decode 混合"的假设不同（`WscLlmIterationRecord` 校验阶段互斥，`wsc_llm_scheduler.py:1113-1140`）。

时间表 `WscLlmTimingLut`（`:636-810`）是 **纯阶段时间 LUT**：Prefill 只有 `(p_chunk=512, d_batch=0)` 一类条目，Decode 按 2 的幂 token bin 惰性生成；它只估算迭代耗时，**不参与实例选择、不改变静态映射**（与 sh_2.0 用 LUT 做 per-die 代价不同）。`estimate_iteration_time_ns()`（`:577-634`）对 linear 与 attention 分别取 `max(计算时间, HBM 延迟+字节/带宽)`，并把 D2D 单跳延迟 5 ns 加进结果。

### 1.5 Trace 生成：映射结果如何固化

流程入口：`generate_trace.sh` → `generate_trace.py` → `generate_wsc_llm_trace.py main()`（`:2118-2178`）。

1. `load_wsc_llm_trace_config()`（`:398-544`）：读 `trace_config.csv`；按 `request_queue_session_limit` 用 `select_first_session_requests()`（`:368-395`）**整 session 保留**地选取前 N 个 session 的全部请求（当前仓库内派生队列为 2 session / 4 request 的小规模队列；README 描述的三分钟窗口全量是 678 session / 9179 request，运行日志中也出现过 40/482、136/1839 等规模）；`config_resolver.materialize_runtime_configs()`（`config_resolver.py:416-451`）派生 system/network/remote_memory/comm_group 四件套到 `generated/runtime_config/`。
2. `build_wsc_llm_plan()`（`:561-585`）→ `plan_wsc_llm_requests()`（`wsc_llm_scheduler.py:2210-2247`）→ `_plan_wsc_llm_session_lru_recompute()`；结果直接写 ET，**没有 pickle 缓存**。
3. `_write_wsc_session_lru_trace()`（`generate_wsc_llm_trace.py:1075-1375`）按 `prefill_start_ns` 排序逐请求向 `prefill_group.ranks` 与 `decode_group.ranks` 各自的 ET builder 发射节点：
   - 首请求在 Prefill 实例各 rank 写 `arrival_timer_gate`；后续请求在上一轮 decode 实例各 rank 写 `interval_timer_gate`（依赖上一轮 end barrier 节点）；
   - 历史 KV 迁移：`NOC_MIGRATE` 时用 `_paired_transfer()`（`:629-697`）把相对 TP rank 一一配对成 `COMM_SEND/RECV` 对（字节按 whole-head shard，XY 路由）；`RECOMPUTE` 时先发 history 重计算 Prefill 段；
   - 各 rank 先做 `history_tp_ready_barrier`（TP=6 All-Reduce 1B），再发当前 prompt 的 Prefill（`token_expanded` 逐 chunk 或 `request_aggregated` 聚合）；
   - Prefill 完成后发 prefill→decode KV 迁移（同样 `_paired_transfer`，category=3000）；
   - Decode 阶段在 `decode_group.ranks` 发射，pg 用 `decode_group.pg_name`，结尾 `decode_request_end_barrier` 作为下一轮 interval gate 的锚点。
4. 产物：54 个 `llama2_7b_wsc_llm_inference.{rank}.et`、`manifest.json`、`wsc_llm_timing_lut.csv`、`kv_cache_events.csv`，以及默认开启的 `metrics_manifest.json`（指标 sidecar）。

manifest 完整记录映射审计信息：`prefill_assignment.policy="least_request_count_including_active_head,config_order"`、`decode_assignment.policy="offline_static_nearest_decode_no_runtime_reselection"`、`static_pd_mapping.routes`、每请求的 prefill/decode 实例、静态路由、`history_action`、迁移路由、HBM 快照、逐出记录与规划时间轴等（`:1181-1370`）。

运行：

```bash
cd astra-sim-wscllm
bash sh_test_mesh/run_scripts/generate_trace.sh          # Python 规划 + 生成 54 份 ET + manifest
bash sh_test_mesh/run_scripts/run_sh_test_aware.sh       # 校验 NO_MEMORY_EXPANSION 后启动 congestion-aware 仿真
```

单测：`sh_test_mesh/workload/llama2_7b_inference/test_wsc_llm_scheduler.py`（覆盖静态路由、least-queue 选择、LRU 删除、容量压力等）。

---

## 2. KV 管理机制

### 2.1 单层本地 HBM 账本（无远端冷层）

**热层——每 NPU 本地 HBM**（唯一人工维护来源 `hardware/face_case5_config_c.json`）：

- 带宽 1640 GB/s、延迟 100 ns、D2D 4050 GB/s / 5 ns/跳、峰值算力 261.12 TFLOPS；
- 容量走 profile：`paper-64gib`（64 GiB）与 `validation-160gib`（160 GiB，当前选用）；
- `config_resolver.py:416-451` 把 `local-mem-bw / local-mem-latency / local-mem-capacity-bytes / peak-perf` 写入派生 `system.json`；
- 派生 `remote_memory.json` 的 `memory-type` 为 **`NO_MEMORY_EXPANSION`**，只保留 `remote-mem-bw: 512.0` 作为模拟器硬件假设；
- `generate_wsc_llm_trace.py:359-366 _validate_no_memory_expansion()` 强制校验默认配置必须是 `NO_MEMORY_EXPANSION`。

**Python 侧账本**：`SessionKVCacheManager.__init__`（`session_kv_manager.py:367-464`）为每个 rank 建 `NodeHBMState`（`capacity_bytes`、`model_weight_bytes`、`resident_kv_bytes`、`reserved_request_bytes`），所有操作后 `_check_invariants()`（`:1538-1591`）校验守恒与不超容量。

模型权重与 KV 都按 **whole-head 对齐的 TP 分片**记账：`model_weight_shard_bytes_by_tp_rank()`（`:111-145`）与 `kv_cache_shard_bytes_for_tokens()`（`:146-167`）保证 `sum(shards) == 全局字节数`。例如 6 头 rank 上 223 token 的 KV = `2 × 32 层 × 223 × 128 head_dim × 6 头 × 2 B = 21,921,792 B`（`kv_cache_events.csv` 中可见）。

> 容量约束只在 Python 账本生效：`system.json` 的 `local-mem-capacity-bytes` 写好后 C++ 端不解析、不消费（`Sys.cc` 只解析 bw/latency/peak-perf 等）。C++ 端也没有 sh_2.0 的 `LocalHbmBandwidthModel`（本仓库不存在该文件），因此**没有推理/KV 恢复的 HBM 带宽竞争建模**。

### 2.2 两态驻留状态机

`SessionKVCacheManager` 为每个 session 维护 **两态**：

| 状态 | 含义 |
|---|---|
| `RESIDENT` | 完整 KV 驻留在**单一实例**的 6 个 rank 上，按相对 TP rank 一一配对；本仓库中权威位置是请求完成时所在的静态 decode 实例 |
| `EVICTED` | 物理 KV 已被删除（零代价），但 `logical_context_tokens` 保留，后续请求可据此**重计算**历史 |

模块 docstring 明确声明（`session_kv_manager.py:1-10`）：只建模三分钟 TraceLab 工作负载用到的策略；**没有 partial 层驻留、没有远端分层、没有容量驱动的重映射**。与 sh_2.0 的三态（`local_hbm` / `partial_hbm_remote` / `remote_memory`）和 FACE 的跨实例 KV 分片都不同。

`WscRelevantKvAllocator`（`wsc_llm_scheduler.py:852-1077`）仍保留在调度器中，但只是 **legacy 策略的回归测试实现**：默认 `session_lru_recompute` 不调用它，不允许把默认 session 的 KV 跨 P/D 实例 spill。

### 2.3 静态 decode 目标预留与 1M token 水位线

**准入预留**：请求开始 Prefill 前，`try_admit_prefill()` 调用 `reserve_request_capacity()`（`session_kv_manager.py:930-987`）在**固定 decode 目标**上按 `final_context_tokens` 预留最终 KV 分片：

1. `enforce_watermark()`（`:825-868`）：先把该实例压回水位线；
2. `ensure_physical_fit()`（`:870-928`）：必要时逐出非 active session，直到能放下预留；
3. 预留只占 `reserved_request_bytes` 账本，KV 物理上尚未到达 decode 实例。

**完成水位线**：请求 Decode 完成后 `mark_complete()`（`:1486-1519`）把 KV 标记为 inactive、记录 `last_completion_ns`，并再次 `enforce_watermark()`。水位线要求**每个物理 rank** 剩余空间 ≥ `kv_reserve_context_tokens`（默认 1,000,000 token，`trace_config.csv:17`）对应的 KV 分片字节：

- 6 头 rank：`2 × 32 × 1e6 × 128 × 6 × 2 = 98,304,000,000 B`（约 91.6 GiB）；
- 5 头 rank：约 76.3 GiB；
- 加上权重分片（约 2.1-2.2 GiB）后仍可装入 160 GiB profile——因此 64 GiB profile 会在 `SessionKVCacheManager.__init__` 快速失败（`:421-430`）。

水位线不足时按 `_candidate_sessions()`（`:510-526`）顺序删除已完成、非 active 的 RESIDENT session；没有候选时记录 `watermark_deferred` 事件并返回，最终状态检查 `assert_final_state()`（`:1521-1537`）不允许规划结束时仍低于水位。

### 2.4 逐出策略：LRU 整 session 删除（零成本 + 重计算）

触发点——`enforce_watermark()` / `ensure_physical_fit()` 被以下路径调用：

- 准入预留：`reserve_request_capacity()`（`:930`）；
- 历史 KV 准备：`prepare_history()`（`:1024`）中的 watermark + fit；
- Prefill/Decode KV 增长：`_grow()`（`:1259-1312`）；
- Prefill→Decode 迁移：`move_prefill_to_decode()`（`:1334-1464`）；
- 请求完成：`mark_complete()`（`:1486`）。

候选集与排序（`:510-526`）：

- 必须是 `RESIDENT`、位于目标实例、`active == False`、有 `last_completion_ns`，并排除当前请求自己的 session；
- 排序键 `(last_completion_ns, session_id)`——即 **"最后完成时间最旧优先"（least-recently-completed，类 LRU）**，与 sh_2.0 命名一致。

删除动作 `_delete()`（`:765-823`）：从该实例 6 个 rank 直接减去 shard 字节，session 转为 `EVICTED`（`instance_index=None`）。**删除在 ET 中没有任何节点，代价为零，也不产生远端流量**；`kv_cache_events.csv` 记录 `evict_delete` 事件，manifest 中 `kv_management` 段记录 `policy="session_lru_recompute"`、`eviction_order="last_completion_ns_then_session_id"`、`delete_cost="zero"`、`remote_memory_used=false`。

与 sh_2.0 的"两阶段降水位（先逐后 K 层到远端，再整 session）"不同，本仓库是**单阶段整 session 删除**，没有 partial 状态、没有远端 store。

### 2.5 历史 KV 的四种处理动作

`prepare_history()`（`session_kv_manager.py:1024-1257`）在 Prefill 准入时根据 session 历史状态决定 `history_action`：

| 动作 | 条件 | 行为 |
|---|---|---|
| `NO_HISTORY` | 窗口首请求（turn 0，history=0） | 无传输、无重算 |
| `LOCAL_HIT` | 历史 KV 已在当前 Prefill 实例（`source == target`） | 零传输直接复用 |
| `NOC_MIGRATE` | 历史 KV 在其它实例（典型：上次完成时留在 decode 实例，本次 Prefill 选到另一实例） | 按相对 TP rank 从源实例 `COMM_SEND/RECV` 迁到 Prefill 实例（`KVTransferShard` 一一配对） |
| `RECOMPUTE` | 历史 KV 已被逐出（`EVICTED`） | 在 Prefill 实例上以 512 token chunk **重计算** `history_recompute_tokens`，再算当前 prompt |

要点：

- `RECOMPUTE` 只发生在物理 KV 被删除之后，`logical_context_tokens` 保留使重算量精确等于历史长度；
- 重计算与当前 prompt 的 Prefill 都是普通 COMP 节点走 Roofline，`effective_prefill_tokens = prefill_length + history_recompute_tokens` 记录在 manifest；
- 冷层 KV"远端取回（不重算）"的路径在本仓库**不存在**，因为根本没有远端层；
- 一次请求的 KV 移动最多可以出现**两段**：历史 `NOC_MIGRATE`（decode 实例 → Prefill 实例）+ `prefill_decode` 迁移（Prefill 实例 → decode 实例），后者在 `move_prefill_to_decode()`（`:1334`）完成，并把 `grow_decode()` 增长到最终 context。

### 2.6 KV 迁移的代价建模（C++ 执行层）

迁移/重计算在 C++ 侧的代价模型很简单：

1. **NoC 段**：`_paired_transfer()` 为每个相对 TP rank 生成 `COMM_SEND/RECV` 对，走分析型网络后端（D2D 4050 GB/s、5 ns/跳），字节按 whole-head shard（`total_bytes % num_heads == 0` 时按头数均分，否则精确字节切分），路径为 XY 路由；
2. **计算段**：重计算 chunk 是 Prefill COMP 节点，`Workload::issue_comp()`（`Workload.cc:260-346`）按 Roofline 与 local-mem 带宽取最大；
3. **无 MEM 节点**：ET 中不出现 `MEM_LOAD` / `MEM_STORE`；`Workload::issue`（`Workload.cc:167-235`）对 MEM 节点一律走 `issue_remote_mem()`，而 `AnalyticalRemoteMemory::issue`（`AnalyticalRemoteMemory.cc:154-211`）对 `NO_MEMORY_EXPANSION` 直接报错退出——因此 WSC-LLM trace 里出现 MEM 节点即配置/生成错误；
4. **无 HBM restore 竞争**：sh_2.0 的 `is_local_hbm_kv_restore` 分发、`LocalHbmBandwidthModel` 与 `hbm-kv-restore-bandwidth-sharing` 开关在本仓库全部移除（`Workload.cc` 与 `Sys.cc` diff 可证）；
5. **逐出零代价**，只有下一次 turn 的 `RECOMPUTE` 才产生计算代价。

当前 6 条 P→D 路由**边不相交**，配合 congestion-aware 网络后端可避免共享链路带宽瓶颈；`alpha` 元数据为此保留描述空间。

### 2.7 配置参数汇总

| 配置项 | 位置 | 当前值 | 作用 |
|---|---|---|---|
| `kv_cache_policy` | `trace_config.csv:16` | `session_lru_recompute` | 选择"单实例驻留 + LRU 删除 + 重计算"策略 |
| `kv_reserve_context_tokens` | `trace_config.csv:17` | 1,000,000 | 完成后水位线：每 rank 预留 1M token KV 分片空间 |
| `prefill_chunk_size` | `trace_config.csv:15` | 512 | Prefill 与历史重计算的固定 chunk 大小 |
| `local_hbm_capacity_profile` | `trace_config.csv:20` | `validation-160gib` | HBM 容量 profile（64/160 GiB；64 GiB 会快速失败） |
| `hardware_config` | `trace_config.csv:19` | `hardware/face_case5_config_c.json` | 唯一硬件来源：1640 GB/s / 100 ns / 4050 GB/s / 5 ns / 261.12 TFLOPS / `NO_MEMORY_EXPANSION` |
| `trace_granularity` | `trace_config.csv:14` | `request_aggregated` | 算子聚合粒度（`token_expanded` 亦可） |
| `request_queue_csv` | `trace_config.csv:12` | 派生三分钟队列（当前 2 session / 4 request） | 请求源；`request_queue_session_limit=0` 表示全选 |
| `remote_operand_loads` | `trace_config.csv:22` | false | 不建模远端权重加载 |

策略标识落盘位置：

- `manifest.json` → `kv_management.policy="session_lru_recompute"`、`eviction_order="last_completion_ns_then_session_id"`、`watermark_scope="per_physical_npu_exact_tp_shard"`、`delete_cost="zero"`、`remote_memory_used=false`（`:1305-1344`）；
- 每个 rank ET 的 GlobalMetadata 写 `scheduler="WSC_LLM_STATIC_PD"`、`execution_mode="wsc_llm_pd_disaggregated_static_et"`、`instance_name/pg_name/phase_role`、`remote_memory_expansion=false`（`generate_wsc_llm_trace.py:699-746`）；
- `kv_cache_events.csv` 逐事件记录 `no_history` / `local_hit` / `noc_migrate` / `recompute` / `retain_complete` / `evict_delete` / `admission_blocked` / `watermark_deferred` 等，含前后 HBM 剩余与不足 rank 列表。

---

## 3. 与 `astra-sim-sh_2.0`（FACE 风格）的关键差异

| 维度 | `astra-sim-sh_2.0`（参考文档） | `astra-sim-wscllm`（本文档） |
|---|---|---|
| 实例角色 | 9 个统一实例，PD 共址 | 6P:3D 专用实例，PD 分离 |
| Prefill 选择 | HBM 可行过滤 + Roofline 负载 `L_instance` + session 亲和 | 最少排队请求数（含 active head），配置顺序 tie-break |
| Decode 选择 | 运行期候选集（≤2.47 跳）+ LUT per-die 代价 + HBM tie-break | 离线静态最近 decode，全局最小（跳数、共享边、签名），无运行期重选 |
| 实例内执行 | 混合 iteration（1 prefill chunk + 全部 active decode） | 阶段分离 iteration（每个实例每轮只做 Prefill 或 Decode） |
| KV 驻留状态 | 三态：`local_hbm` / `partial_hbm_remote` / `remote_memory` | 两态：`RESIDENT` / `EVICTED` |
| 冷层 | mesh 边界 26 端口远端内存池（无限容量、512 GB/s） | 无远端层（`NO_MEMORY_EXPANSION`），ET 中不允许 MEM 节点 |
| 逐出 | 两阶段：先逐后 K 层到远端，再整 session；`remote_store` 有代价 | 单阶段整 session 删除，零代价；下轮 `RECOMPUTE` 承担代价 |
| 恢复 | 远端取回（不重算）+ 前缀计算/后缀恢复流水重叠 | 无远端恢复；被删 KV 直接重计算，历史跨实例则 NoC 迁移 |
| HBM 带宽竞争 | `LocalHbmBandwidthModel` 50/50 流体共享 | 无（该模型在本仓库不存在） |
| 迁移次数 | 通常 1 段（prefill→decode），历史 PARTIAL 时半层恢复 | 最多 2 段（历史 decode→prefill + prefill→decode） |
| LUT 作用 | 参与 decode 候选代价选择 | 纯阶段时间表，不参与任何选择 |
| 产物 | `face_lut.csv`、manifest（含 `kv_eviction_policy`） | `wsc_llm_timing_lut.csv`、`kv_cache_events.csv`、`metrics_manifest.json`（指标 sidecar） |

---

## 4. 关键文件索引

- `sh_test_mesh/workload/llama2_7b_inference/wsc_llm_scheduler.py`
  - `build_instances()` `:188-317` — 专用实例布局、校验、邻接图
  - `build_static_pd_mapping()` `:438-540` — 离线静态 P→D 路由
  - `select_prefill_instance()` `:826-831` — least-queue 选择键
  - `WscRelevantKvAllocator` `:852-1077` — legacy Relevant(P,D) 分配器（未选中）
  - `_plan_wsc_llm_session_lru_recompute()` `:1667-2208` — 离散事件主循环
  - `try_admit_prefill()` `:1746-1849` — 预留/历史/增长三步准入
  - `try_admit_waiting_decodes()` `:1851-1913` — Prefill 完成后的 decode 移交
  - `start_ready_iterations()` `:1915-2018` — 阶段分离执行
- `sh_test_mesh/workload/llama2_7b_inference/session_kv_manager.py`
  - `SessionKVCacheManager` `:367-1591` — HBM 账本、两态机、LRU 删除、水位线
  - `_candidate_sessions()` `:510-526` — 逐出候选与排序
  - `enforce_watermark()` `:825-868` / `ensure_physical_fit()` `:870-928`
  - `reserve_request_capacity()` `:930-987` — 静态 decode 目标预留
  - `prepare_history()` `:1024-1257` — NO_HISTORY / LOCAL_HIT / NOC_MIGRATE / RECOMPUTE
  - `move_prefill_to_decode()` `:1334-1464` / `mark_complete()` `:1486-1519`
- `sh_test_mesh/workload/llama2_7b_inference/generate_wsc_llm_trace.py`
  - `select_first_session_requests()` `:368-395` / `load_wsc_llm_trace_config()` `:398-544`
  - `build_wsc_llm_plan()` `:561-585` / `_paired_transfer()` `:629-697`
  - `_emit_prefill_stage()` `:942-1006` / `_write_wsc_session_lru_trace()` `:1075-1375`
- `sh_test_mesh/config_resolver.py`
  - `load_hardware_config()` `:143-309` — 硬件解析与容量 profile
  - `_prepare_remote_memory()` `:327-354` / `_prepare_comm_groups()` `:356-401` / `materialize_runtime_configs()` `:416-451`
- C++ 执行层
  - `astra-sim/workload/Workload.cc:167-235` — 节点分发（MEM 一律走 remote_mem，无 local restore 分支）
  - `extern/remote_memory_backend/analytical/AnalyticalRemoteMemory.cc:154-211` — `NO_MEMORY_EXPANSION` 下访问即报错
  - `astra-sim/system/Sys.cc:393-405` — bw/latency/peak 解析（无 capacity、无 hbm sharing 字段）

---

## 5. 注意点与坑

- **Prefill 选择与 KV 状态无关**：least-queue 选择发生在 arrival，容量/历史处理在准入时；因此同一 session 的下一轮可能选到不同 Prefill 实例，历史 KV 需要先 `NOC_MIGRATE` 过去，再在 Prefill 完成后迁回 decode——最坏一轮两段全量迁移。
- **decode 目标是"静态最终权威位置"**：请求完成后 KV 一定留在固定 decode 实例；`terminal_kv_release_at_completion=false`，即使 session 不再有后续请求也不自动释放（除非被水位线逐出）。
- **逐出排序为"最久未完成优先"**（least-recently-completed，类 LRU），排序键 `(last_completion_ns, session_id)`，manifest 已统一为 `last_completion_ns_then_session_id`。
- **删除零代价，重计算是唯一重建路径**：没有远端 store/load；"重计算"指被逐出 KV 的重新 Prefill，而不是论文中可能存在的远端取回。
- **ET 中不能出现 MEM 节点**：`NO_MEMORY_EXPANSION` 下 `AnalyticalRemoteMemory::issue` 直接 `exit(1)`；任何 KV 动作只能表现为 `COMM_SEND/RECV`（迁移）或 COMP（重计算）。
- **无 HBM 带宽竞争**：KV 迁移只走 NoC，不消耗本地 HBM 带宽；`LocalHbmBandwidthModel`、`hbm-kv-restore-bandwidth-sharing` 在 sh_2.0 存在但本仓库没有。
- **容量只在 Python 账本生效**：`local-mem-capacity-bytes` 写进 system.json 但 C++ 不消费；改硬件容量必须重新生成 trace，否则 ET 与账本不一致。
- **容量压力是 head-of-line 阻塞**：`try_admit_prefill` 失败时整个 Prefill FCFS 队列停摆，直到容量事件（完成/逐出/迁移）触发 `capacity_epoch` 变化才重试；`admission_blocked` 事件做了去重，不会每 token 刷屏。
- **LUT 只是时间表**：`wsc_llm_timing_lut.csv` 的行是阶段时间（Prefill 512 chunk ≈ 4,320,413 ns，decode 1 token/batch=1/kv=512 ≈ 1,397,295 ns），不代表调度决策，改 LUT 不会改变映射。
- **请求队列规模可扩展**：`request_queue_session_limit` 控制选取的 session 数（0 = 全部）；README 描述的全量三分钟窗口为 678 session / 9179 request，当前仓库内派生队列是 2 session / 4 request 的小规模冒烟集，`build_trace_label()` 的目录名会随队列摘要变化。
- **指标系统是旁路**：`metrics_manifest.json` 只做只读观测与摘要（含 KV 事件摘要、请求映射摘要），不反向影响调度；`metrics_postprocess.py` 负责把运行日志整理成 CSV。
