# request 映射策略与历史 KV 管理机制

## 0. 体系结构总览

本系统是一条四层通路：**输入物化层 → 调度层 → KV 管理层 → 图发射/模拟器执行层**。

- **输入物化层**：`plan_materializer.py` 将请求队列 CSV（session/turn/prefill/decode/到达时刻）物化为 `manifest.json` 的九个请求事实字段；`generate_face_trace.load_face_trace_config` 将 trace_config.csv 中的 `inference_group` 行物化为实例（TP 推理组）布局，经 `config_resolver` 派生 system/network/comm_group 三件运行时配置。静态拓扑（rank→网格坐标→实例→邻接图）在此层由 `face_scheduler.build_instances` 一次性铸型并 fail-closed 校验。
- **调度层**：`FaceOnlineScheduler`（继承 `online_scheduler_base.OnlineSchedulerBase`）持有实例账本（`qp` FCFS prefill 队列 / `active_decode` 批成员表 / `pending_decode_ready` / `in_flight_train` 唯一在飞列车）。控制流按交付批次（delivery）驱动：C++ 侧 `RequestIngress` 到达 alarm 与 `WatchRegistry` 的 drain/exit 标记 fire 经 `DecisionMailbox` 聚成 StateDelta，经文件桥 `decision_bridge.BridgeServer`（FIFO 通知 + 原子 JSON）送达 Python；调度器在 `run_variant_policy` 内完成 completion 核销 → arrival 入队 → 准入/发射 pass，产出 GraphBatch（节点/边/watch/assignment/kv_actions/future_alarms）回传 C++ `GraphBatchCommitter` 原子提交。
- **KV 管理层**：`SessionKVCacheManager` 是逐 rank HBM 的权威状态机。会话 KV 只有 `RESIDENT`/`EVICTED` 两态（外加无记录的 NO_HISTORY 态），策略代号 `session_lru_recompute`：完成边界保留 KV（retain），被动逐出按 LRU（`last_completion_ns`），被逐历史以 RECOMPUTE 重建，跨实例驻留走 NOC_MIGRATE 成对传输。所有变更即时记账并产生 `KVCacheEvent` 流。
- **图发射/模拟器执行层**：`GraphBatchBuilder`（含 `OnlineTraceBuilder` per-rank 节点构造器）把调度决策翻译为 17 类聚合节点、KV 成对传输（XY 路由）、列车标记（drain/exit/哨兵/first_token）与 end barrier；ASTRA-sim C++ 侧按节点 rank 所有权注入各 NPU 的事件队列执行，物理完成经 watch 回流形成下一个决策边界。`kv_actions` 在 C++ 侧半透明（Python 权威），经 commit-ack 后由 Python 侧 provisional 账本 finalize。

两条主数据流：请求流（manifest → 到达 → qp → prefill 实例 → decode 实例 → 列车成员 → 完成/退休）与 KV 字节流（分片计算 → 准入/迁移/增长 → 驻留 → 被动逐出 → 终态释放），二者在每个决策边界上交汇。离线观测链（`slo_tools/kv_cache_adapter.py`、`hbm_watermark.py`）在 run 结束后从决策日志重放这两条流，做命中率归因与 HBM 水位重建。

以下路径缩写：`W/` = `sh_test_mesh/workload/llama2_7b_inference/`，`O/` = `W/online/`，`S/` = `sh_test_mesh/slo_tools/`。

---

## 1. request 映射策略：全分支枚举

### 1.1 静态拓扑映射（rank → 实例 → 网格 → 邻接图）

映射原语：`rank_coordinates`（`W/face_scheduler.py:128-131`）以 `divmod(rank, mesh_cols)` 将线性 rank 映射为网格坐标；rank 越界即 raise。

`build_instances`（`W/face_scheduler.py:134-222`）把 `inference_group` 规格（name/pg_name/ranks）铸型为 `FaceInstance` 集合，逐条防御分支：

| # | 触发条件 | 行为 | 位置 |
|---|---|---|---|
| T1 | specs 为空 | raise（至少一个实例） | face_scheduler.py:140-141 |
| T2 | name 为空 / pg_name 为空或 "0" / pg_name 重复 | raise | face_scheduler.py:149-155 |
| T3 | ranks 为空 / 实例内 rank 重复 | raise | face_scheduler.py:156-159 |
| T4 | rank 已归属其他实例 | raise（rank 所有权互斥） | face_scheduler.py:163-166 |
| T5 | 实例坐标集 ≠ 填充轴对齐矩形 | raise（矩形性硬校验） | face_scheduler.py:172-184 |
| T6 | 全部实例 rank 并集 ≠ [0, npus_count) | raise（全覆盖恰好一次） | face_scheduler.py:194-195 |
| T7 | `require_equal_size=True` 且实例尺寸不齐 | raise（直接 KV 分片配对要求等 TP 度） | face_scheduler.py:196-199 |
| T8 | 实例数 >1 且邻接图有孤立节点 | raise（铺砖连通性） | face_scheduler.py:201-216 |

该组校验在物化期即前移执行（`W/plan_materializer.py:384-395`），布局非法时 plan 不落盘；实例名唯一性另有 load 期校验（`W/generate_face_trace.py:274-284`，同名实例会使迁移发射器的"同名零节点"判定失真）。邻接图边权恒为 1（`WeightedInstanceGraph.__init__`，face_scheduler.py:332-338）；remote-memory 跨实例放置路径已被整体删除（face_scheduler.py:382-387 注记）。

### 1.2 模型分片映射（权重/KV 字节 → TP 相对秩）

- `partition_values_exact`（`W/session_kv_manager.py:59-66`）：整数精确划分，余数归低相对秩——所有 TP 分片的基底原语。
- `attention_heads_by_tp_rank`（session_kv_manager.py:69-81）：整头（whole-head）归属划分。
- `estimate_model_weight_bytes`（session_kv_manager.py:84-109）：`mlp_variant` 二分支——swiglu（3 个 MLP 矩阵、2·hidden norm、final norm）与 gelu（2 矩阵、4·hidden norm、无 final norm）。
- `model_weight_shard_bytes_by_tp_rank`（session_kv_manager.py:112-144）：矩阵分片后残差（norm 等）再走精确划分回填；残差为负或总和不守恒 → RuntimeError。
- `kv_cache_shard_bytes_for_tokens`（session_kv_manager.py:147-166）：逐头 KV 分片 = `2·layers·tokens·head_dim·bytes_per_elem × head_count`，总和不守恒 → RuntimeError。
- `hidden_size % num_heads != 0` → raise（session_kv_manager.py:115-116, 155-156）。

### 1.3 请求入口映射（队列行 → RequestSpec → manifest 事实 → 到达事件）

- `load_request_queue`（`W/generate_trace.py:146-235`）逐行分支：request_id 以 `#` 开头 → 注释跳过（:167）；turn-0 行必须有 `session_arrival_time_ns` 且 interval 必须为空，turn>0 行反之（:176-205）；timing 必须是 1000 ns 整数倍（:211-214）；空队列 → raise（:233-234）。
- `select_first_session_requests`（`W/generate_face_trace.py:202-229`）：`request_queue_session_limit == 0` → 全选；`>0` → 取前 N 个不同 session 的全部行；N 超过可用 session 数 → raise。
- manifest 九字段派生（`W/plan_materializer.py:265-327`）：
  - turn-0：`history_tokens_before = 0`，`prefill_context_tokens = prefill_length`（recompute 单口径，turn-0 的 prefix 已折入队列）。
  - turn>0：父 turn 的 final 缺失（session 块不连续/turn 不严格 +1）→ RuntimeError（fail-closed，绝不静默回退 history=0）；否则 `history = 上一请求 final`，`context = history + prefill_length`，`final = context + decode_length`。
  - sidecar（WP2 透传）行在场 → 附带 human/tool_time_ns、request_type、prefix_len 四字段；缺席则不写、由消费方回退 unknown（:311-324）。队列命名不合规但同目录存在 sidecar → WARN（不改退出码，plan_materializer.py:243-260）。
- 运行期请求账本构造守卫（`O/face_online_scheduler.py:199-218`）：`prefill_length <= 0` → raise（零工作请求会永久滞留 qp）；`decode_length <= 0` → raise（对称守卫，避免晚期状态机崩）。
- 策略开关收敛：`kv_cache_policy` 仅允许 `session_lru_recompute`（配置层 `W/generate_face_trace.py:161-167`；服务装配层 `O/online_service.py:186-210`；调度器构造层 `O/face_online_scheduler.py:321-325` 三重 fail-closed）；`trace_granularity` 在线仅允许 `request_aggregated`（`O/graph_batch_builder.py:358-361, 415-418, 458-461, 497-500` 四个发射入口同校验）；`prefill_chunk_size <= 0` → 构造期 raise（face_online_scheduler.py:343-346）。
- C++ 到达入口（`astra-sim/workload/execution_driven/RequestIngress.cc`）：命令队列满 → 拒收背压（:40-47）；Submit 的 `arrival <= current` → 钳到 current+1，并按来源分桶计数（StaticCsv 的 t=0 边界态 / 背压延迟态 / 真迟态，ExternalStream 单列，:81-118）；CloseInput / EndOfFile / Error 三种终态命令分流，Error → `std::abort` fail-closed（:145-169）；future arrival 校验（空 id、queue_index 非法、批内重复、已调度未到、queue_index 与登记不符 → 拒绝，:210-254）；到达回调把 envelope 落成 `DecisionReason::ARRIVAL` 事件（:340-374）。

### 1.4 prefill 映射通路（到达 → 实例选择 → qp 入队）

`_on_arrival`（`O/face_online_scheduler.py:1038-1063`）：

1. 全实例队列快照 `_queue_snapshots`（:1616-1628）：`remaining_chunks` = 实例 qp 成员剩余 chunk 之和，`last_arrival_ns` = 实例最近到达。
2. `select_prefill_instance`（`W/face_scheduler.py:326-329`）：取 `ordering_key = (remaining_chunks, last_arrival_ns 缺席取 -1, instance_index)` 最小者——最少积压 → 最早最近到达 → 最小实例号的确定性字典序；快照集为空 → raise。
3. **P1 选择期容量过滤** `_filter_prefill_selection`（face_online_scheduler.py:1731-1784），三分支：
   - 原 selected 幸存（`_probe_instance_capacity` 全 rank `remaining_bytes >= 全上下文保守需求分片`）→ 原样返回，决策零扰动（:1757-1758）；
   - 原 selected 不可行且幸存集非空 → 幸存集内按原 `ordering_key` 取 min 重选，落 `admission_probe` 决策行（outcome=`reselected_infeasible_first`，:1767-1777）；
   - 幸存集空 → 保持原选，阻塞交给 P0-1 重试通路（outcome=`kept_all_infeasible`，:1778-1784）。
   预检为只读红线：仅调 `hbm_snapshots` 与纯函数，不触发逐出/放置（`_probe_instance_capacity` docstring，:1632-1650）；保守口径不含冷会话逐出信用，误判由重试兜底。
4. 入队被选实例 `qp`、更新 `prefill_remaining_chunks` 与 `last_arrival_ns`、置 ready frontier（非忙时）、账本登记 `prefill_qp` 层。

### 1.5 decode 映射通路（prefill drain → 实例选择 → 等待准入）

`_on_prefill_drain`（face_online_scheduler.py:1065-1128）在 PREFILL_DRAIN 边界执行：

1. drain 请求不在其 qp → raise（:1079-1080）；扣除实例聚合 chunk、`qp.remove`（:1081-1083）；仍有排队工作 → frontier add，否则 discard（:1084-1087）。
2. 顺序敏感全局快照：`has_prefill_work`（各实例 qp 非空）与 `decode_token_lengths`（各实例 active_decode 成员的 `current_decode_token`）（:1089-1093）。
3. `select_decode_instance`（`W/face_scheduler.py:410-494`）分支树：
   - 候选集 = `schedulable_instances(prefill_instance, d2d_bw/local_hbm_bw)`——以带宽比为距离上限的可达实例集合（face_scheduler.py:374-380；空集 → raise，:434-435）；
   - 每候选 Roofline 增量成本：`p_chunk = fixed_p_chunk 当该实例有 prefill 工作，否则 0`（:441）；`updated_d_token = max(d_token, 新请求 prefill_context)`（:458）；`per_die_delta_ns = (updated − current 迭代时间)/实例尺寸`（:473-482）；迭代时间估计内部分支：`p_chunk==0 且 d_batch==0` → 0（:259-260），prefill 段仅在 `p_chunk>0` 计入、decode 段仅在 `d_batch>0` 计入、两注意力段取 max 与线性段相加（:287-307），最终 `max(1, ceil(ns)+d2d_latency_ns)`（:308）；
   - 平局裁决：`per_die_delta_ns` 精确并列且候选 ≥2 → 传入 `DecodeTieCounter` 时按 `counter % len(tied)` 轮转并前进计数器（:489-491）；否则取 tied[0]（最小 instance_index 旧行为，:492-493）。
4. **P1 decode 过滤** `_filter_decode_selection`（face_online_scheduler.py:1652-1729）：
   - 需求 = `final_shards − (会话 RESIDENT 且已驻该候选的 shard_bytes)`，source==target 按增量、否则全量（双驻留口径，:1683-1690）；
   - 增量需求任一分量为负（会话驻留超过 final 需求，manifest 自相矛盾）→ 该候选按不可行剔除（fail-closed 对称，R3-I4，:1691-1697）；
   - 原 selected 可行 → 原样返回（:1700-1701）；可行集非空 → 按 `(per_die_delta_ns, instance_index)` 最小重选（绕过 tie_counter，:1710-1722）；全不可行 → 保持原选走阻塞-重试（:1723-1729）。
5. 落定 `decode_instance_index`（"绝不 remap"不变量：一经落定仅可 defer）、登记 `waiting_decode_admissions[selected]` 队列、置 `decode_admission_dirty`、账本先登记 `waiting_decode` 再立即尝试准入（:1112-1128）。

### 1.6 准入映射（KV 预约 → 允许占用实例资源）

**prefill 准入** `_try_admit_prefill`（face_online_scheduler.py:1280-1368）：

| # | 触发条件 | 行为 | 位置 |
|---|---|---|---|
| A1 | `admitted_prefill` 已真 | 直接 True（幂等） | :1285-1286 |
| A2 | `prefill_instance_index` 为 None | raise（固定映射丢失） | :1287-1288 |
| A3 | 纪元门：`prefill_attempt_epoch == capacity_epoch[实例]` 且实例不在 `prefill_admission_dirty` | return False（本边界不重试，防重试风暴） | :1293-1296 |
| A4 | 越过纪元门 | 记录 attempt epoch、清本实例 dirty | :1297-1301 |
| A5 | `prepare_history` 返回 `admission_blocked` | 有逐出则 `_note_capacity_change`（推进容量纪元重开重试门）；**置 prefill_admission_dirty**（P0-1 活性修复：无逐出的阻塞不推纪元，dirty 保证下一决策边界必重试）；return False | :1323-1336 |
| A6 | 准入成功 | 记录 history_action/source/transfer_bytes/recompute_tokens；`grow_prefill` 失败 → raise（预约丢失）；`remaining_chunks` 重算为 `ceil(recompute/p_chunk) + ceil(prefill_length/p_chunk)`；`admitted_prefill=True`；通知容量变更（源实例 + 逐出受害者实例） | :1337-1367 |

跨阻塞 attempt 的逐出记录累积语义：`admission_evictions` 只增不覆盖（:1321-1322），保证 hbm_watermark"逐出全量落盘、重放闭合"契约。

**decode 准入** `_try_admit_waiting_decodes`（face_online_scheduler.py:1370-1508）：

| # | 触发条件 | 行为 | 位置 |
|---|---|---|---|
| B1 | ready_targets 过滤：实例等待队列非空 且（在 dirty 集 或 `decode_admission_epoch != capacity_epoch`） | 进入本轮准入扫描；扫描头同步纪元并清 dirty | :1375-1389 |
| B2 | 队首 runtime 已不处等待态 | 跳过（continue） | :1394-1395 |
| B3 | runtime 的 decode 目标 ≠ 本队列 | raise（队列腐坏） | :1396-1398 |
| B4 | `move_prefill_to_decode` 返回 `admission_blocked` | 有逐出则通知容量变更；**置 decode_admission_dirty**（P0-1：重开下一边界的重试门）；`queue.append` 重排队尾（face 阻塞重排队语义）并 continue | :1409-1428 |
| B5 | 迁移成功 | `grow_decode` 失败 → raise；成员进入 `pending_decode_ready`（KV 就绪栅栏，等下一列车边界加入）；实例空闲则 frontier add；`waiting_decode_admission=False`；写 assignment（双实例索引一次性携带，C++ 校验器要求非负）；落 decode 决策行（decode_candidates 全记录、prefill_decode_transfer、kv_noc_hops、decode_target_evictions 快照后核销置空）；账本升级 `active_decode`；通知容量变更 | :1429-1508 |

容量纪元 `_note_capacity_change`（:1786-1790）：对涉及的实例集合（含逐出受害者所在实例与迁移源实例）`capacity_epoch += 1`，重开两侧纪元门。

### 1.7 批次/列车映射（请求 → 迭代列车成员/chunk/span）

统一实例上 prefill chunk 与 decode 成员混拼为"迭代列车"（每迭代 ≤1 个队头 chunk，批成员只在列车边界变化）。

`_plan_train`（face_online_scheduler.py:630-720）分支：

| # | 触发条件 | 行为 | 位置 |
|---|---|---|---|
| C1 | active_decode 成员剩余 token ≤0 | raise（无剩余 token 的解码成员） | :641-647 |
| C2 | 无队头且无成员 | return None（实例无工作） | :650-652 |
| C3 | 无队头（纯 decode 列车） | 自然迭代数 = 成员最大剩余 token | :653-654 |
| C4 | 有队头 | 自然迭代数 = 队头 `remaining_chunks`；≤0 → raise | :656-660 |
| C5 | `SH_TRAIN_MAX_ITER>0` 且迭代数超限 | 截断到 T_max（默认 8，环境变量构造期校验：非非负十进制整数 → raise，:396-401；0=不设限）并标 capped | :663-665 |
| C6 | 成员 `participation >= remaining` | 列入 exit_members（列车内先验退出） | :676-683 |
| C7 | 未截断且存在队头 | 队头列入 drain_members（列车终于 drain 迭代） | :687-688 |
| C8 | capped 且无 drain 且无 exit | 列车自身 id 作哨兵完成信号（signal_set 加入 train_id） | :693-695 |
| C9 | 队头首 chunk（`prefill_tokens_completed==0`） | 标记 head_first_chunk（prefill_start 指标锚点） | :689-690 |

chunk 两段式序列 `_plan_head_chunks`（:722-748）：`completed < history_recompute_tokens` → recompute 段 chunk（span 上下文自 0 增长）；否则当前 prefill 段（span 在 `history_tokens_before` 上追加）；当前段 token 耗尽仍被要求 → raise（:741-744）。decode 成员 span = `(1, context + consumed + step)` 逐迭代 +1（:678-680）。

发射通路 `_plan_and_emit_trains`（:750-808）分支：

- WP9 余量批优先：任一实例存在 `first_step_remainder` → 先发余量体（:770-772）；
- frontier 按实例号排序遍历（决策确定性）；`in_flight_train` 非空 → 跳过（busy 门 = 一列车在飞，:776-777）；
- 队头选择跳过 `remaining_chunks==0` 的头部（drain 决策在途，后续请求 chunk 物理上可开始，:779-786）；
- 队头未准入：`_try_admit_prefill` 成功 → `_emit_admission` 后并入列车；失败 → `qp_head=None`，本列车退化为纯 decode（face 回退支路，:787-791）；
- `pending_decode_ready` 非空 → 整批转正 active_decode 并记发射/账本（:793-801）；
- plan 为 None 且实例三队列全空 → frontier discard（:803-807）。

WP9 首 token 拆分 `_first_token_plan`（:908-945）：`SH_FIRST_TOKEN_SPLIT != "1"`（缺省关，`O/graph_batch_builder.py:71-92`）→ None；无 debut 成员（joiner 中 `decode_tokens_consumed==0`）→ None；`iterations >= 2` → 两段式（split=True，首步批 + 余量批）；否则仅做不拆车的标记增强（split=False）。首步唤醒回声经 `_consume_first_step_wakeup`（:532-542）从哨兵通道剥离（无操作）。

`_emit_train`（:810-884）：stage 取 `"decode"`（有成员或 joiner）或 `"prefill"`（纯 prefill 列车，:821）；split 计划 → `_emit_train_first_step` 后返回（:843-853）；整列发射则注册 drain/exit/哨兵 watch（statuses 允许 Success/Skipped）、挂起 in_flight_train、frontier discard、写列车台账行。

列车核销 `_finalize_completed_trains`（:544-628）：

- 哨兵 train_id 无路由 → raise（:563-566）；
- 在飞列车首个信号到达：成员按 participation 闭式推进 `decode_tokens_consumed/current_decode_token`；exit 成员移出 active_decode（不在 lookup → raise，:583-588）；chunk 进度推进并递减两侧 remaining；`in_flight_train=None`；残余信号登记 finalized_trains pending（:568-607）；
- 后续信号命中已核销列车 → 幂等清 pending，清空即出列表（:608-615）；
- 信号无法归属任何在飞/近期列车 → raise（陈旧完成错配，:616-622）；
- 核销即 frontier 判定：有排队工作 → add else discard（:624-628）。

### 1.8 节点/rank 发射映射（决策 → 物理图）

准入动作发射 `_emit_admission_actions`（`O/graph_batch_builder.py:736-818`）：

| # | 触发条件 | 行为 | 位置 |
|---|---|---|---|
| E1 | turn_index==0 | 每 prefill rank 发 arrival timer gate（session_arrival 缺失 → raise）；在线语义 duration==0 不发节点 | :750-762 |
| E2 | turn>0 | 取用并弹出 `completion_gates[session]`（缺门或无 interval → raise）；发 interval timer gate（duration==0 仅建 after_node_id 依赖） | :763-781 |
| E3 | `history_action == NOC_MIGRATE` | 源索引缺失 → raise；源组 ≠ 上一 turn 完成门组 → 先发 1 字节 control trigger 跨实例接力并清 timers；随后 `_paired_transfer` 类别 1000 的 history KV 成对传输 | :782-804 |
| E4 | 其他 history_action（NO_HISTORY/LOCAL_HIT/RECOMPUTE） | 仅发 interval control trigger（无 KV 字节） | :805-811 |
| E5 | 汇合 | prefill 组 `history_tp_ready_barrier` all_reduce 屏障 | :812-815 |

`_paired_transfer`（`W/generate_face_trace.py:419-486`）：TP 度不等 → raise（:431-432）；**源组名 == 目标组名 → 零节点迁移**（仅 arm timer gates，:433-437）；`total_bytes % num_heads == 0` → 按整头均分分片，否则按字节精确划分回退（不向上取整，:442-453）；逐 shard 发 comm_send/comm_recv，comm_tag = `queue_index*10000 + category + relative_rank`（越 31 位 → raise，`_stage_tag` :389-403）；路由为 XY 序（先列后行，`_xy_route` :406-416）。

列车头/体/尾发射（graph_batch_builder.py）：`_emit_train_head` joiner 迁移为类别 3000 成对传输（同实例零节点）+ join/pstart 起始标记（:513-557）；`_emit_train_body` 每 rank 调 `transformer_pass_aggregated`，`weight_passes = 迭代数`（权重每迭代只读一次，:559-588）；`_emit_train_tail_markers`（:610-717）：drain/exit 标记（每成员每 rank 一节点，承载 PREFILL_DRAIN/DECODE_COMPLETION watch）、`decode_length==1` 的 debut exit 标记改名携带 `first_token` 子串（同节点双锚点，:648-662）、哨兵标记固定 prefill stage 单事件通道（避免 decode stage 连带 REQUEST_COMPLETE 计账下溢，:664-680）、共享 end barrier all_reduce（bytes 口径 = 迭代数，既定基线，:682-699；block_ends 出现 None → raise，:702-703）、exit 成员的 end barrier 写入 `completion_gates[session]`（下一同 session turn 的 interval gate 依赖源，:708-710）。split 计划误入整列发射口 → raise（:419-423）；first_step/remainder 两段入口各校验计划形态（:462-465, 501-504）。

`OnlineTraceBuilder.timer_gate`（:204-250）：`duration_ns == 0` → 不发节点直接返回 after_node_id；`duration_ns % 1000 != 0` 或负 → raise；否则发 runtime=0 的 timer 节点（等待由 C++ arrival alarm 承担，避免双重等待）。

`_kv_noc_hops`（face_online_scheduler.py:1570-1594）：源/目标为 None 或同实例 → `[]`（无物理传输）；TP 度不等 → raise；否则逐 shard 返回 XY 跳数（只读导出，不进策略判据）。

### 1.9 决策边界与协议分支（Python↔C++ 控制面）

- delivery 序列门 `_gate_delivery_sequence`（`O/online_scheduler_base.py:487-516`）：seq == last → 幂等重放（delta 与缓存不一致 → raise；一致则返回缓存 batch 深拷贝，状态零变更）；seq != last+1 → raise（跳号/旧重复均为协议违例）。
- completion 核销 `_settle_completions`（:862-905）：`batch_train_` 前缀 → 跳过（变体按 train_id 核销）；未知请求 → raise；stage==REQUEST_COMPLETE → 整体核销 + 账本 completed 层；stage 不在 pending 集 → raise（重复完成）；否则从 pending 集移除并同步 issued 层。同一交付内 DECODE_COMPLETION 必须先于 REQUEST_COMPLETE（乱序即 unknown-request 门 abort，docstring :870-877）。
- arrivals 处理 `_process_arrivals`（:907-933）：request_id 撞 `batch_train_` 保留字前缀 → raise（防哨兵特权逃逸）；未知/重复到达 → raise；否则登记 in_flight{prefill,decode} 并做在途尾部观测。
- commit ack `on_commit_ack`（:992-1066）fail-closed 分支：ack 非 dict / schema != v1 / delivery_sequence 非法 / `batch_id != delivery_sequence` / `success is False` / seq 领先已应用交付 / 序列已被水位或乱序洞记录（幂等 return）/ 无 pending emitted 记录 → 各自 raise 或幂等跳过；合法则 finalize provisional KV 账本（转入 committed 计数）并做层转移、ack 水位推进与 reply 缓存释放。
- 文件桥 `O/decision_bridge.py`：request/ack 重复旧序号 → 幂等早退（:320-321, :381-383，双侧冻结契约）；未来序号 → BridgeError fail-closed（`_BoundedSequenceTracker.validate_new`，:101-125）；handler/ack 回调异常 → 写 error response + 非 0 退出（`_fail` :403-429；ack 流失败按 A2-1 口径对 `next_seq` 写 error，:464-497）；resp 通道 BrokenPipe = C++ 已死 → fail-closed（:502-508）。
- 在途尾部 fail-closed 上限（`PropagatingTailTracker`，online_scheduler_base.py:57-126, 316-346）：arrivals 上限 = 请求总数；delivery/kv_provisional 上限默认 8（`SH_PROPAGATING_TAIL_LIMIT` 环境变量 > 类属性覆盖 > 默认，非正整数构造期 raise）；超限 raise，绝不截断真实在途工作。
- C++ 侧 stage→reason 映射（`astra-sim/network_frontend/analytical/congestion_aware/main_online.cc:1332-1364`）：prefill stage fire → 单推 PREFILL_DRAIN；decode stage fire → 同时推 DECODE_COMPLETION + REQUEST_COMPLETE。`GraphBatchCommitter.cc:1399-1406`：assignment 的 request_id 为空或任一实例索引为负 → 校验失败拒绝提交。
- run-end 审计（face_online_scheduler.py:1836-1895）：完成数不足、runtime 未释放、实例非空闲（qp/active_decode/pending/在飞列车任一非空）、prefill 聚合非零、未冲刷首步余量、未消费唤醒/迟到列车信号、等待 decode 准入非空、arrival heap 非空、ready frontier 非空、completion gates 未退休 → 各自 raise；最后 `kv_manager.assert_final_state()`。

---

## 2. 历史 KV 管理机制：全分支枚举

会话状态机：`NO_HISTORY`（无记录）→ `RESIDENT(instance_index, shard_bytes, active)` ⇄ `EVICTED`（保留 logical_context_tokens，物理删除）；唯一终态出口为 `retire_terminal_session` 删除记录（`W/session_kv_manager.py:17-22, 220-241`）。

### 2.1 命中判定（prepare_history 的动作分派）

`prepare_history`（`W/session_kv_manager.py:921-1141`）前置谓词：

| # | 触发条件 | 行为 | 位置 |
|---|---|---|---|
| H1 | `history_tokens < 0` / `required_context_tokens < 0` / `required < history` | raise | :933-938 |
| H2 | 会话不存在且 `history_tokens != 0` | raise（新会话不得有历史 KV） | :940-941 |
| H3 | 会话逻辑上下文 ≠ 请求 history | raise（口径漂移） | :942-946 |
| H4 | 会话 active | raise（重叠请求，同会话串行不变量） | :947-948 |
| H5 | 目标实例增量需求（desired − 目标上已有驻留分片）任一为负 | raise（KV 不允许收缩） | :950-960 |

准入受阻支路（`admission_blocked=True`，:972-995）——动作标签镜像成功路径分类法：无会话 → `NO_HISTORY`；`EVICTED` → `RECOMPUTE`（`recompute_tokens = history_tokens`）；RESIDENT 同实例 → `LOCAL_HIT`；RESIDENT 异实例 → `NOC_MIGRATE`（挂起迁移语义，M5-2 修复：绝不标 LOCAL_HIT）；transfer_shards 恒空，附带 `insufficient_ranks`。

成功支路四动作：

| 动作 | 触发条件 | 数据流 | 位置 |
|---|---|---|---|
| `NO_HISTORY` | 会话不存在 | 新建 `SessionKVState(RESIDENT, target, shard=0, active=True)`，发 `no_history/window_first_request` 事件 | :998-1030 |
| `RECOMPUTE` | 会话 `EVICTED` | 目标实例加 history 分片、状态转 RESIDENT、active=True，发 `recompute/deleted_history_recompute` 事件；返回 `recompute_tokens = history_tokens`（全量重算，无 partial 语义） | :1032-1065 |
| `LOCAL_HIT` | RESIDENT 且 `source == target` | 零字节，仅置 active 与 last_request_id，发 `local_hit/history_local_reuse` 事件（驻留谓词满足，走重用支路） | :1067-1099 |
| `NOC_MIGRATE` | RESIDENT 且 `source != target` | 逐 TP 相对秩构造 `KVTransferShard`；**先目标 `_add_shards` 后源 `_remove_shards`**（先准入后释放的动态传输契约），会话归属迁至目标；发 `noc_migrate/history_other_instance` 事件 | :1101-1141 |
| （防御） | RESIDENT 但 instance_index 为 None | raise | :1067-1068 |

### 2.2 准入（ensure_physical_fit：被动逐出引擎）

`ensure_physical_fit`（session_kv_manager.py:803-916）：

| # | 触发条件 | 行为 | 位置 |
|---|---|---|---|
| F1 | 需求分片向量长度 ≠ TP 度或含负 | raise | :816-817 |
| F2 | 需求 > (capacity − model_weight)（空实例都装不下） | 记 `deep_gap`（reason=`request_exceeds_empty_instance`）后 raise（结构不可行 fail-closed） | :818-843 |
| F3 | 当前不缺（`_insufficient` 为空） | 直接 admitted，无逐出 | :844-853 |
| F4 | 缺口开启 | 候选 = `_candidate_sessions`（一次性快照排序，循环中不再重排） | :846-853 |
| F5 | 候选耗尽仍有缺口 | 记 `deep_gap`（reason=`exhausted_completed_candidates`）+ 去重压力事件 `admission_blocked`，返回 `admitted=False`（优雅推迟） | :856-880 |
| F6 | 受害者状态漂移（非 RESIDENT/非本实例/active/无完成时刻） | raise（I1 镜像守卫，捕获快照与删除间的漂移） | :883-895 |
| F7 | 逐出一个受害者 | `_delete` 释放分片、转 EVICTED、发 `evict_delete` 事件；重算缺口继续循环 | :900-909 |
| F8 | 循环退出但缺口仍在 | raise（I3 防御） | :910-915 |

候选资格谓词 `_candidate_sessions`（:463-479）：RESIDENT ∧ 驻留本实例 ∧ 非 active ∧ `last_completion_ns` 非 None ∧ 不在保护集（当前会话自身）；排序键 `(last_completion_ns, session_id)`——LRU 优先，平局按会话号全序（`_session_key` 入口统一 str 化，:46-56，防混排类型比较崩溃）。

压力事件去重 `_pressure_event`（:522-565）：键 `(event_type, phase, request, target)` 首次出现 → 记原事件；同键再现 → 每请求/相位/目标恰发一条 `admission_retry`；完成时 `_forget_pressure_event_keys`（:573-575, :1380）按请求索引 O(1) 退休全部键。

### 2.3 驻留与增长（grow 通路）

`_grow`（session_kv_manager.py:1143-1178）：非 RESIDENT/非本实例/非 active → raise；目标 − 现有任一为负 → raise（上下文不收缩）；`ensure_physical_fit` 预约增量（含被动逐出）；受阻 → 返回 `admitted=False`（调用侧 prefill 段视为不可达 raise、decode 段已在上游拦截）；成功 → 加分片、更新 `shard_bytes/logical_context_tokens`。

- `grow_prefill`（:1180-1199）：instance 缺失 → raise；长到 `prefill_context_tokens`。
- `grow_decode`（:1317-1336）：instance 缺失 → raise；长到 `final_context_tokens`。
- `move_prefill_to_decode`（:1201-1315）三分支：
  - 前置：非 RESIDENT/instance 缺失/非 active → raise（:1212-1213）；
  - 阻塞：返回 `MoveDecision(action = LOCAL_HIT 当 source==target 否则 NOC_MIGRATE, admission_blocked=True)`（:1244-1252）；
  - 同实例：`LOCAL_HIT`，零字节，发 `prefill_decode_local_reuse` 事件（:1253-1271）；
  - 跨实例：`NOC_MIGRATE`，先加目标后删源，构造逐 shard 迁移描述，发 `prefill_decode_instance_migrate` 事件（:1273-1315）。

### 2.4 完成驻留与被动逐出（保留语义）

- `mark_complete`（session_kv_manager.py:1338-1381）：非 active RESIDENT → raise；置 `active=False`、`last_completion_ns`；**发 `retain_complete/request_completed_keep_kv` 事件——完成边界不逐出**（主动逐出路径已删，返回值恒空元组，A1-16）；同时退休该请求的压力去重键。
- `_delete`（:752-801）：受害者不满足"已完成 + 非活跃 + RESIDENT + 有完成时刻"→ raise（:761-767）；`_remove_shards` 逐 rank 扣减（释放超持有 → raise，:620-628）；状态转 EVICTED、instance 清空；`evict_delete` 事件在 ET 图上无对应节点（纯规划态动作）。
- 终态退休 `retire_terminal_session`（:1383-1428）——仅当调度器证明该完成请求无后继 turn 后调用：未知会话 → KeyError；active 或未完成 → raise；request 不拥有会话 → raise；无 terminal 请求号 → raise（:1397-1415）；RESIDENT → 释放分片并返回实例号（触发容量纪元推进）；EVICTED → 无字节动作；未知状态 → raise（:1417-1423）；删除会话记录。

### 2.5 恢复（RECOMPUTE 重建与在线重放口径）

- 规划层恢复 = `prepare_history` 的 RECOMPUTE 支路（2.1）：逐出历史的逻辑上下文保留在 `logical_context_tokens`，重建以两段式 chunk 序列（recompute 段 + 当前段）在 prefill 列车内执行（`_plan_head_chunks`，face_online_scheduler.py:722-748）。
- 观测层恢复重放 `WatermarkReplay.apply_restore`（`S/hbm_watermark.py:1370-1475`）分支：`RECOMPUTE` 或 `state_before==EVICTED` 且跟踪态仍有驻留 → 判定为未落盘静默逐出，在恢复点对账扣减（silent_evictions_reconciled，:1395-1406）；`LOCAL_HIT` → 零字节，源≠目标时占用随迁（restore_occupancy_relocated，R6-U4，:1411-1422）；0 字节 restore → 同型随迁（:1436-1444）；`source is None` → 远端/回载语义只增目标（:1445-1455）；`source == target` → 分层半层回载只增（:1456-1463）；否则跨实例搬移：源扣目标加，超扣 → raise（:1464-1475）；bytes 与 `f(tokens)=coef×tokens` 比值偏离 1.0/0.5 → restore_bytes_mismatch 异常计数（:1379-1387）。

### 2.6 事件流水、水位与不变量

- 事件分配 `_event`（session_kv_manager.py:481-520）：`event_index` 由单调计数器分配，与列表位置解耦。
- 水位压缩 B1（:411-422）：消费方（在线调度器 kv_actions 流，face_online_scheduler.py:519-528）持 `_kv_events_emitted` 游标走 `events_since` 增量切片；已消费 ≥ `_EVENTS_COMPACT_THRESHOLD`(8192) 且 ≥ 现存一半 → `compact_events` 整段删除前缀并同步回退游标（摊还 O(1)/事件）。
- 深缺口观测 D4（:424-427, 577-611）：两个失败出口（结构不可行 / 候选耗尽）前计数 +1 并落 `deep_gap` 事件；实证负载预期恒 0，>0 即批次安全边界被触及的观测信号（汇入 online_stats 汇总行，`O/online_service.py:260-263`）。
- 增量不变量账本（:630-749）：每次变更后仅刷新受影响会话/秩的贡献——负记账 / 超 HBM 容量 / 会话-秩记账错配 / 未知状态 / RESIDENT 无实例 / 分片向量长度错 / EVICTED 残留实例 → 各自 raise；`SH_STRICT_KV_INVARIANTS` 环境开关（"1/true/yes/on"，:30-38）令每次变更追加全量审计 `_check_invariants`（:1438-1473）。
- 终态断言 `assert_final_state`（:1430-1436）：压力去重键残留 → raise；active 会话残留 → raise。
- 运行期准入联动的容量纪元：`capacity_epoch` 数组 + `decode_admission_epoch`/`decode_admission_dirty`/`prefill_admission_dirty`（face_online_scheduler.py:374-386）构成"映射决策何时允许重试"的控制面（见 1.6）。

### 2.7 观测适配层（映射与重放的输出层分支）

- `S/kv_cache_adapter.py` 命中态映射 `_hit_state_face_wscllm`（:120-133）：`NO_HISTORY → no_history`（不计 miss）、`LOCAL_HIT → full`、`NOC_MIGRATE → full`、`RECOMPUTE → miss`；未知动作 → `not_supported`。决策 kind 闸（:86-98）：`prefill/decode/completion` 为载荷行；`decode_admission_probe/prefill_admission_probe` 为观测行（跳过计数）；未知 kind → fail-closed。canonical 事件提取 `extract_events_face_wscllm`（:313-371）：prefill 行的 `history_transfer_bytes>0` → history_transfer 事件；admission/decode_target/completion 三族逐出列表逐条映射（bytes 回退链 shard_bytes → total_bytes → Σshards，:271-310）；decode 行的 `prefill_decode_transfer.total_bytes>0` → prefill_decode_transfer 事件。`--reconcile` 以 native 日志重放对 canonical 逐 family 对账（:794-937）；Σshards ≠ total_bytes → 默认 warn+非零码、reconcile 下硬 fail（:781-790）。
- `S/hbm_watermark.py` 四层可信度（:148-151, 2021-2178 判定）：`per_rank_total_hbm_certified`（journal+checksum 四道门全过，唯一可出退出码 3 的判决层）/ `resident_kv_exact` / `lifecycle_replay_exact` / `upper_bound_only`（journal 缺失，decision-log 重放为上界，occupancy_valid 恒 false）；证书孤儿（checksum 在而 journal 缺）→ fail-closed 退出码 2。容量三口径 `compute_capacity_calibers`（:656-733）：正式认证（逐 rank physical=weight+resident+reserved ≤ capacity）、resident 硬上限（min_r ⌊(capacity−weight_r)/kv_r⌋ token，0 头 shard 不参与约束）、水位目标（`kv_reserve_context_tokens` 扣减后同式，仅报告不判决）。重放主序（`WatermarkScan.consume` :1580-1663）：记录内固定"逐出 → 恢复/迁移 → 增长"；prefill 行缺整数实例 → fail；请求缺 token manifest → fail；tick 回退 → fail；同请求同 kind 决策两次 → fail；token manifest 全覆盖校验（finish，:1665-1687）。`apply_evict`（:1324-1350）：受害者不在跟踪态/超扣 → fail；bytes < 跟踪 → partial_evictions 计数；清零后解除实例归属。`apply_move`（:1477-1507）：mismatch 检查先于 stay 分支；同实例或 0 字节 → decode_stay；跨实例 → 源扣目标加。`apply_grow`（:1509-1522）：目标 < 跟踪 → fail（负增长）。逐出 tick 早于实例末变点 → 钳制到末变点（保 RLE 单调，:1294-1304）。桶长 `watermark_sample_period_ns` null/错型 → 临时锚点 5 ms + provisional 标注（:965-984）；S2 映射的字段在场升级分支（:749-781）不适用于 face（face 恒 full 口径）。

---

## 3. 两条机制在体系结构中的交汇

映射决策处处受 KV 状态约束：prefill/decode 实例选择在落定前经 P1 只读预检读取 `hbm_snapshots` 与 `session_snapshot`（remaining-only 保守口径），原选不可行时在候选集内重选；落定后受"绝不 remap"不变量保护，容量不足一律转为阻塞-推迟，由容量纪元（`capacity_epoch` + dirty 门）控制在下一个决策边界重试；`prepare_history` 内部同实例驻留折减（增量需求）与迁移候选的"先准入后释放"次序，使实例选择直接决定 KV 字节流的源与汇。反向地，KV 状态被每次映射决策改写：NO_HISTORY 建立驻留、RECOMPUTE 在被逐后重建、NOC_MIGRATE 改变会话归属实例、grow 沿映射实例增长分片、被动逐出（LRU）由新请求的准入压力触发、`retire_terminal_session` 在无后继 turn 时释放终态驻留并经 `_note_capacity_change` 反向重开其它请求的重试门。调度层的"列车成员冻结 + 决策边界原子提交"与 KV 层的"规划期即时记账 + kv_actions 事后经 commit-ack 确认"构成双时态：会话状态机对策略即时可见，而对 C++ 执行层经 provisional 账本最终一致；观测层（adapter/watermark）从决策日志重放这两条流，恰好覆盖该双时态的全部出口（含 deep_gap、admission_retry、silent eviction 对账等防御分支），形成映射—KV 闭环的外部审计面。

---

## 附：分支覆盖自查记录

对 `face_scheduler.select_decode_instance`（4 分支：候选空 raise/p_chunk 折减/平局轮转/默认 tied[0]）、`session_kv_manager.prepare_history`（9 分支：4 前置 raise + 折减负 raise + 4 阻塞动作 + 4 成功动作）、`ensure_physical_fit`（8 分支：F1-F8）、`move_prefill_to_decode`（3 动作 + 前置 raise）、`_try_admit_prefill`（A1-A6）、`_try_admit_waiting_decodes`（B1-B5）、`_plan_train`（C1-C9）、`_emit_admission_actions`（E1-E5）、`_filter_prefill_selection`/`_filter_decode_selection`（各 3 出口 + 负需求剔除）、watermark 重放（restore 6 支路 + move 2 + grow 1 + evict 3）逐函数与本文条目对账；环境开关共 6 个（`SH_STRICT_KV_INVARIANTS`、`SH_TRAIN_MAX_ITER`、`SH_FIRST_TOKEN_SPLIT`、`SH_GRAPH_DIGESTS`、`SH_PROFILE_JSONL`、`SH_PROPAGATING_TAIL_LIMIT`）与配置开关（`kv_cache_policy`、`trace_granularity`、`prefill_chunk_size`、`request_queue_session_limit`、`kv_reserve_context_tokens`、`--sensing`）均已入表。唯一审阅后确认不入正文的：`transformer_pass_aggregated` 内部聚合代数（纯节点字节计量，无控制分支影响映射/KV）、`metrics_postprocess` 的 first_token_source 回退（输出层口径）。
