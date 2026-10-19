# request 映射策略与历史 KV 管理机制

## 一、架构总览

系统为五层结构，自上而下为：**输入物化层 → 在线调度层 → KV 管理层 → 图发射层 → 模拟器执行层**。

- **输入物化层**（`plan_materializer.py` / `generate_trace.py::load_request_queue`）：请求以会话多轮（session/turn）形态驻留于请求队列 CSV，经 loader 的会话块结构校验后，被物化为 `manifest.json` 的九字段请求事实（`request_id/session_id/turn_index/queue_index/prefill_length/decode_length/history_tokens_before/prefill_context_tokens/final_context_tokens`）。历史 token 派生规则：turn-0 恒 0，turn>0 取同会话上一轮的 `final_context_tokens`——"恢复取代重算"的单口径在物化层即冻结。
- **在线调度层**（`online/wsc_llm_online_scheduler.py` + `online/online_scheduler_base.py`）：调度器进程经 `decision_bridge.py` 的文件桥（request/response FIFO 通知协议）与 C++ 模拟器对端耦合。每个 tick 末端收到一份 StateDelta 交付，按固定批序运行策略：completion 批 → arrival 批 → 准入/发射 pass。请求→实例的映射决策（Prefill 实例选择、静态 P→D 路由）、KV 容量仲裁全部发生在此层，策略输入全部来自 Python 侧账本（关感知口径）。
- **KV 管理层**（`session_kv_manager.py`）：会话级二态冷热状态机（`LOCAL_HBM` 全层驻留单实例 / `REMOTE_MEMORY` 全层驻留远端池）叠加单级整会话 LRU 逐出、net-credit 预约账本与逐 rank HBM delta journal。
- **图发射层**（`online/graph_batch_builder.py` + `generate_wsc_llm_trace.py` 发射原语）：把调度决策翻译为 per-rank 节点图（GraphBatch 的 nodes/parent_edges），节点携带 `rank` 字段，请求到 rank 的物理归属在此最终落形。
- **模拟器执行层**（`astra-sim/workload/execution_driven/`：`RequestIngress`、`GraphBatchCommitter`、`WatchRegistry` 等）：`GraphBatchCommitter.cc` 按每个节点的 `node.rank` 把节点登记到对应 Sys 实例（rank 所有权为类型域约束：send 节点 `comm.src == node.rank`、recv 节点 `comm.dst == node.rank`，`GraphBatchCommitter.cc:1225-1245`）；真实完成事件（PREFILL_DRAIN / DECODE_COMPLETION / REQUEST_COMPLETE）由 C++ 物理时钟推进并回推调度器。

数据流走向：请求 → 到达堆 → Prefill 实例选择 → 静态 P→D 路由钉定 decode 目标 → KV 准入仲裁（预约/历史准备/增长）→ per-rank 节点图 → C++ 按 rank 执行 → 完成信号回推。控制流核心环路：`run_variant_policy`（`wsc_llm_online_scheduler.py:550`）内 completion 批（列车核销 `_finalize_completed_trains` → `_on_prefill_drain` → `_on_decode_complete` → `_on_request_complete`）先于 arrival 批（`_push_arrival`/`_drain_arrival_heap` → `_on_arrival`），末段为 `_admit_pass`（decode 准入 `_try_admit_waiting_decodes` 先于逐实例发射）。映射关系上：**一个请求恰映射一个 Prefill 实例 + 一个由静态路由钉定的 Decode 实例；一个会话恰映射 KV 管理器中的一个二态状态对象；一个实例的 KV 分片按 whole-head TP 切分映射到实例内全部 rank**。

---

## 二、request 映射策略：全分支枚举

### 2.1 输入物化通路（请求 → 队列事实/到达形状）

| # | 分支 | 触发条件 | 行为与数据流向 | 代码位置 |
|---|------|----------|----------------|----------|
| M1 | turn-0 行派生 | `spec.turn_index == 0` | `history_tokens_before = 0`；`prefill_context_tokens = prefill_length`（折入口径） | `plan_materializer.py:323-334` |
| M2 | turn>0 行派生 | `turn_index > 0` | `history = last_final_by_session[session_id]`；`context = history + prefill_length`；`final = context + decode_length` | `plan_materializer.py:335-348` |
| M3 | 重复 turn-0 拒绝 | 同 session 出现第二条 turn-0 行 | fail-closed（此前静默覆盖游标） | `plan_materializer.py:326-331` |
| M4 | 轮次断裂拒绝 | turn>0 无紧邻前序轮 | fail-closed，不静默 history=0 | `plan_materializer.py:339-344` |
| M5 | request_id 重复拒绝 | manifest 内重复 id | fail-closed（下游按 id 键控的账本无法消歧） | `plan_materializer.py:379-384` |
| M6 | metrics 到达形状：turn-0 | `turn_index == 0` | `arrival = {kind: "absolute", value_ns: session_arrival_time_ns}` | `plan_materializer.py:413-419` |
| M7 | metrics 到达形状：turn>0 | `turn_index > 0` | `arrival = {kind: "after_request", parent_queue_index, interval_ns}` | `plan_materializer.py:420-429` |
| M8 | (session,turn) 重复拒绝 | metrics 派生遇重复键 | fail-closed | `plan_materializer.py:406-411` |
| M9 | sidecar join | 队列旁存在 canonical sidecar 且行命中 | 追加 `request_type/prefix_len`（manifest）与 `human_time_ns/tool_time_ns`（metrics_manifest）；缺席不写、不猜测 | `plan_materializer.py:366-373, 451-458` |
| M10 | 队列装载校验族 | 会话块不连续 / turn 非严格 +1 / turn-0 缺 `session_arrival_time_ns` / turn-0 携带 interval / turn>0 缺 interval / timing 非 1000ns 整数倍 / BOM / 行宽不符 / 字段含逗号 / 空 request_id / '#' 前缀 / id 含 `kv` 或 `first_token` 保留子串 | 每项独立 fail-closed（C++ 读者同款词法的 Python 侧镜像） | `generate_trace.py:189-327`（各条件逐条在场） |

### 2.2 到达通路（请求 → 调度器运行时）

| # | 分支 | 触发条件 | 行为 | 代码位置 |
|---|------|----------|------|----------|
| M11 | ingress 到达登记 | StateDelta.arrivals 含该 id | `in_flight[id] = {prefill, decode}` 待办集；重复到达/未知 id fail-closed | `online_scheduler_base.py:954-972` |
| M12 | 下一轮到达 alarm | REQUEST_COMPLETE 边界且存在同 session 后继轮 | 写 `future_alarms`（`tick + inter_request_interval_ns`），由 C++ 到期再投递；interval==0 合法（物理到达被钳到 tick+1） | `wsc_llm_online_scheduler.py:1268-1286` |
| M13 | 终末轮 | 无后继轮 | 不建 alarm，转终态退休支路（见 M26） | `wsc_llm_online_scheduler.py:1305-1317` |
| M14 | arrival heap 弹出 | 堆顶 `time <= tick` | 按 `(tick, priority=1, queue_index, sequence)` 冻结序稳定弹出并进 `_on_arrival`；非 arrival kind fail-closed | `wsc_llm_online_scheduler.py:1345-1356` |
| M15 | 幂等重放 | 交付 seq == last_applied_sequence | 深比对后返回缓存 batch 深拷贝，零状态变更；跳号/旧 seq fail-closed | `online_scheduler_base.py:492-522` |

### 2.3 Prefill 实例选择（请求 → Prefill 实例）

| # | 分支 | 触发条件 | 行为 | 代码位置 |
|---|------|----------|------|----------|
| M16 | 快照构建 | 每次 `_on_arrival` | 仅 `phase_role == prefill` 的实例进入快照集（`PrefillQueueSnapshot(instance_index, len(qp))`） | `wsc_llm_online_scheduler.py:1936-1944` |
| M17 | 最短队列选择 | 快照非空 | `min(ordering_key)`，`ordering_key = (request_count, instance_index)`——先比排队深度，平局取配置序小者（确定性 tie-break） | `wsc_llm_scheduler.py:567-575` |
| M18 | 空快照拒绝 | 无 Prefill 实例 | ValueError（构造期已保证双角色齐备，防御） | `wsc_llm_scheduler.py:573-574` |
| M19 | 入队与赋值 | 选择完成 | `runtime.prefill_instance_index = selected`；`prefill_assignment_key = (request_count, instance_index)`；`qp.append`（FCFS deque 尾入）；ready frontier 置位 | `wsc_llm_online_scheduler.py:1134-1150` |

### 2.4 静态 P→D 路由（Prefill 实例 → Decode 实例，一次性构建）

**拓扑构建守卫族**（`build_instances`，`wsc_llm_scheduler.py:151-282`）：

| # | 分支 | 触发条件 | 行为 | 代码位置 |
|---|------|----------|------|----------|
| M20 | 空实例表 / 空 name / 空 pg_name | specs 为空等 | ValueError | `wsc_llm_scheduler.py:155-156, 166-169` |
| M21 | pg_name 非规范正整数 | int 解析失败 / ≤0 / 字面量与 int 规范形不符（如 `"00"`） | ValueError（防异形同 id 绕过查重） | `wsc_llm_scheduler.py:174-183` |
| M22 | pg_name 重复 / phase_role 非法 / 实例无 rank / rank 重复 | 逐项 | ValueError | `wsc_llm_scheduler.py:184-194` |
| M23 | rank 跨实例重叠 | 同 rank 归属两个实例 | ValueError | `wsc_llm_scheduler.py:197-204` |
| M24 | 实例非轴对齐实心矩形 | 坐标集 ≠ 包络矩形 | ValueError | `wsc_llm_scheduler.py:207-219` |
| M25 | 覆盖性 / 等大小 / 连通性 / 双角色 / 向心布局 | ranks ≠ 全 NPU 恰一次；实例大小不一；邻接图不连通；缺 Prefill 或 Decode；`max(decode 中心距) > min(prefill 中心距)` | 全部 ValueError——Decode 必须中心优先（向心 P/D 布局） | `wsc_llm_scheduler.py:234-276` |

**路由选择**（`build_static_pd_mapping`，`wsc_llm_scheduler.py:413-555`）：

| # | 分支 | 触发条件 | 行为 | 代码位置 |
|---|------|----------|------|----------|
| M26 | 空 Prefill 拓扑 | `prefill_indices` 空 | fail-closed（不再静默返回空映射） | `wsc_llm_scheduler.py:427-429` |
| M27 | 最近 Decode 候选枚举 | 每 Prefill 实例 | BFS 全最短路（`InstanceGraph.all_shortest_paths`，`wsc_llm_scheduler.py:335-382`）；等价最短路 > 64 条 fail-closed（`MAX_EQUIVALENT_SHORTEST_PATHS`，L32） | `wsc_llm_scheduler.py:432-446` |
| M28 | 组合爆炸拒绝 | 候选笛卡尔积 > 4096（`MAX_ROUTE_CANDIDATE_COMBINATIONS`，L33） | ValueError（防网格拓扑指数爆炸） | `wsc_llm_scheduler.py:448-457` |
| M29 | 目标函数选优 | 全组合枚举完成 | 按 `(total_hops, shared_occurrences, signature)` 字典序取最小——先最小总跳数，再最小共享链路次数，末以选择签名保证确定性 | `wsc_llm_scheduler.py:459-495` |
| M30 | 一跳契约拒绝 | 被选中路由存在多跳 | ValueError（doc §5.1 向心布局契约升格为构造期不变量） | `wsc_llm_scheduler.py:510-521` |
| M31 | 边不相交契约拒绝 | 任一物理边被两条路由共享 | ValueError | `wsc_llm_scheduler.py:522-530` |
| M32 | 路由查询 | `route_for_prefill(idx)` | 返回该 Prefill 实例的静态路由；缺失即 KeyError（decode 目标由此钉定：`runtime.decode_instance_index = route.decode_instance_index`，`wsc_llm_online_scheduler.py:1144`） | `wsc_llm_scheduler.py:406-410` |

### 2.5 Prefill 准入通路（`_try_admit_prefill`，`wsc_llm_online_scheduler.py:1407-1619`）

| # | 分支 | 触发条件 | 行为与数据流向 | 代码位置 |
|---|------|----------|----------------|----------|
| M33 | 已准入直通 | `runtime.admitted_prefill == True` | 返回 True（幂等） | `:1413-1414` |
| M34 | 映射缺失异常 | prefill_instance/static_route 为 None | RuntimeError | `:1415-1417` |
| M35 | 容量纪元门 | `(capacity_epoch[p], capacity_epoch[d])` 与上次尝试相同 | 返回 False 跳过重试（同纪元内重试无意义）；纪元在每次容量变化点 `_note_capacity_change` 递增 | `:1420-1426, 1946-1950` |
| M36 | 全量预约 | 通过纪元门 | 以 `final_context_tokens` 的终态 KV 分片向量对 **decode 实例**发起 `reserve_request_capacity`（静态 decode 终态预约，先于任何数据移动） | `:1427-1438` |
| M37 | 全量预约成功 | `reservation.admitted == True` | `decode_capacity_reserved = True`，落历史快照（M39） | `:1518-1530` |
| M38 | 全量预约失败 → 净额重试 | 预约失败 且 会话快照为 `LOCAL_HBM` 且驻留在 decode 目标实例 | `credit = snap.local_shard_bytes`；`net = max(0, final − have)` 逐分量取净额二次预约（原因 `static_decode_final_kv_reservation_net_credit`）；成功则 `reservation_credit = credit` 待回补 | `:1449-1474` |
| M39 | 净额重试仍失败 | 净额预约 `admitted == False` | 合并两次尝试逐出的受害实例唤醒容量纪元；`decode_evictions_pending` 无门立即发射（`_flush_decode_evictions(gated=False)`）；返回 False | `:1484-1506` |
| M40 | 全量失败且无本地旧驻留 | 快照非 LOCAL 或不在 decode 目标 | 容量纪元唤醒（若有逐出）；无门发射逐出；返回 False | `:1508-1517` |
| M41 | 历史位置快照 | 预约成功后 | `history_cache_state_before = "ABSENT" 或 location`；`history_location_before / history_resident_prefix_layers` 三态序列化源 | `:1519-1530` |
| M42 | 历史准备 | 总是执行 | 对 **prefill 实例**发起 `prepare_history`（四分支见 §3.1）；其 fit 逐出登入 `history_evictions` 分账 | `:1531-1545` |
| M43 | 历史准备阻塞 | `decision.admission_blocked == True` | 释放刚成功的预约（`release_request_capacity`）、作废 credit、容量纪元唤醒（prefill/decode/source/逐出受害实例全集合）、无门发射 pending 逐出、返回 False——此为生产可达的活阻塞臂 | `:1546-1567` |
| M44 | 净额回补 | `reservation_credit` 非 None 且历史迁移成功 | 校验 `decision.source_instance_index == decode_instance`（不符 RuntimeError）；`extend_request_capacity(credit)` 把净额预约回补到全量（resident→reserved 1:1 换位） | `:1573-1586` |
| M45 | Prefill 增长 | 历史决策放行 | `grow_prefill` 至 `prefill_context_tokens`；`growth.admitted == False` 即 RuntimeError（预检已成功的增长不可能再失败——防御死管道） | `:1595-1602` |
| M46 | 准入完成 | 全部通过 | `admitted_prefill = True`；三段逐出（预约/历史/增长）全量登账并镜像构图器 pending 门；容量纪元唤醒 | `:1603-1618` |

### 2.6 Decode 准入与迭代列车通路

| # | 分支 | 触发条件 | 行为 | 代码位置 |
|---|------|----------|------|----------|
| M47 | 等待队列就绪判定 | 每 decode 实例 | 队列非空 **且**（`decode_admission_dirty` 含该实例 **或** `decode_admission_epoch != capacity_epoch`）才进入本轮准入——纪元/脏位双重门控 | `wsc_llm_online_scheduler.py:1626-1636` |
| M48 | 逐条准入 | 就绪队列弹出 | 校验 `decode_capacity_reserved`（未预约即 RuntimeError）；`release_request_capacity(anchor_kind="decode_start")` 释放静态预约 | `:1644-1657` |
| M49 | P→D 迁移 | 预约释放后 | `move_prefill_to_decode(final_context_tokens)`；其逐出登入 decode 分账并镜像 pending 门 | `:1659-1674` |
| M50 | 迁移阻塞防御臂 | `move.admission_blocked == True` | RuntimeError（预约不变量保证恒放行，结构性不可达的 fail-closed 兜底） | `:1675-1682` |
| M51 | decode 增长 | 迁移后 | `grow_decode(final_context_tokens)`；`growth.admitted == False` 同为防御死管道 RuntimeError | `:1684-1694` |
| M52 | 入批 | 增长成功 | `active_decode.append` + lookup 集同步；`decode_queue_depth_before_enqueue` 记录入批深度；非忙则进 ready frontier | `:1701-1711` |
| M53 | 列车规划 | `_admit_pass` 遇非忙 decode 实例且 active_decode 非空 | 冻结成员快照：`iterations = max(剩余 token)`；`SH_TRAIN_MAX_ITER`（默认 8，0 不设限）截断即 `capped` | `:810-866, 521-538` |
| M54 | 哨兵标记 | 列车被 T_max 截断且无自然退出成员 | 生成 `batch_train_i{inst}_{seq}` 哨兵 id，经 PREFILL_DRAIN 单事件通道回推核销 | `:849-853, 1018-1028` |
| M55 | busy 门分裂 | `_instance_busy` | Prefill 实例 = `state.busy`（一个整段在飞）；Decode 实例 = `in_flight_train is not None`（一列车在飞）；忙实例滞留 frontier 被跳过（合法路径，不抛） | `:1958-1965, 1381-1382` |

### 2.7 发射粒度通路

| # | 分支 | 触发条件 | 行为 | 代码位置 |
|---|------|----------|------|----------|
| M56 | P 侧整段发射 | prefill 准入成功 | `_emit_prefill` 聚合发射整段 prefill（逐 chunk 不拼，`weight_passes = chunk 数`）；watch = PREFILL_DRAIN；assignments 行携带双实例索引 | `wsc_llm_online_scheduler.py:1725-1762` |
| M57 | 粒度拒绝 | `trace_granularity != "request_aggregated"` | RuntimeError（唯一合法粒度） | `graph_batch_builder.py:494-497, 1088-1091` |
| M58 | D 侧列车发射 | `_emit_train` | joiner 迁移（category 3000 配对传输）→ join 标记 → 17 类聚合体 → exit 标记 → 共享 end barrier；`completion_gates[session] = (instance, barrier ids)` 供下一轮 interval gate | `graph_batch_builder.py:1044-1104, 1183-1401` |
| M59 | WP9 首 token 拆分开关 | `SH_FIRST_TOKEN_SPLIT == "1"` 且列车含 debut 成员且 `iterations >= 2` | 两段式发射：首步批（第 1 迭代 + 唤醒标记）→ 唤醒到达后余量批；缺省 "0" 关闭（对拍字节等价） | `graph_batch_builder.py:79-100`；`wsc_llm_online_scheduler.py:997-1006` |
| M60 | 逐出旁路支链（gated） | 预约成功随 prefill 批发射 | `emit_decode_eviction_actions(gated_on_prefill_segment_end=True)`：触发门 = 该请求 prefill 段 post-barrier 块末；支链 fork 自各 rank frontier、不 join | `graph_batch_builder.py:638-707`；`wsc_llm_online_scheduler.py:1743` |
| M61 | 逐出旁路支链（ungated） | 阻塞 attempt 的已放行逐出 | 无触发门，链到各 rank 当前 frontier 立即发射（决策→发射窗口收敛为零） | `wsc_llm_online_scheduler.py:1505, 1516, 1566`；`graph_batch_builder.py:653-656` |
| M62 | KV 传输 tag 分配 | 每条 KV 传输腿/触发/ack | `TransferTagAllocator` 自 100_000_000 单调分配，≥ 500_000_000（C++ NATIVE 段上界）即 OverflowError；`_stage_tag` 越入分配器段同样 OverflowError | `generate_wsc_llm_trace.py:204-229, 437-446` |

### 2.8 完成通路

| # | 分支 | 触发条件 | 行为 | 代码位置 |
|---|------|----------|------|----------|
| M63 | 完成信号四分类 | completed_groups 逐条 | `batch_train_` 前缀 → 哨兵/首步唤醒（`_consume_first_step_wakeup` 区分二者：唤醒无操作，真哨兵进列车核销）；`prefill` → drained；`decode` → decode_done；`""` → request_done；未知 stage ValueError | `wsc_llm_online_scheduler.py:573-605` |
| M64 | 列车核销 | exit 标记 watch fire | 闭式推进各成员 `decode_tokens_consumed`；退出成员移出 active_decode；首信号核销恰一次，残余信号幂等清 pending；无法归属任何列车的信号 fail-closed | `:645-735` |
| M65 | PREFILL_DRAIN | prefill 段物理完成 | FCFS 队首校验（错序 RuntimeError）；`busy=False`、`qp.popleft`；登记 `waiting_decode_admissions[decode_instance]` + dirty 置位 | `:1152-1178` |
| M66 | DECODE_COMPLETION | 退出标记完成 | `mark_complete`（LRU 时间戳唯一写点，见 §3.5）；完成快照落决策日志；M4 胖字段核销即删 | `:1180-1250` |
| M67 | REQUEST_COMPLETE（有后继） | 同 session 下一轮存在 | 建 M12 alarm；`register_pending_history` 为下一轮登记 pending 门（位置取 turn-0 deferred 通道优先，否则管理器即时快照） | `:1293-1304`；`graph_batch_builder.py:529-547` |
| M68 | REQUEST_COMPLETE（终末） | 无后继轮 | `retire_completion_gate`（清 completion_gates/pending/deferred/pending_store_tails）+ `retire_terminal_session`（见 §3.5）+ 容量纪元唤醒 | `:1305-1317`；`graph_batch_builder.py:515-527` |

---

## 三、历史 KV 管理机制：全分支枚举

会话位置状态机仅二态：`LOCAL_HBM`（全 `L` 层驻留单 TP 实例，`resident_prefix_layers == model_layers`）与 `REMOTE_MEMORY`（本地零驻留，`resident_prefix_layers == 0`，远端账面记于 `remote_source_instance_index/remote_shard_bytes`）。`PARTIAL_HBM_REMOTE` 是仅供旧日志解析的拒绝值，运行期不可达（`session_kv_manager.py:52-64, 1298-1302, 3232-3236`）。

### 3.1 命中判定与历史准备（`prepare_history`，`session_kv_manager.py:2461-2783`）

前置校验族（全部 fail-closed）：新会话携带非零 history（`:2488-2489`）、既有会话 logical context 与请求 history 不符（`:2490-2494`）、会话在飞重叠请求（`:2495-2496`）、`required_context_tokens < history_tokens`（`:2485-2486`）、驻留会话缺实例号（`:2562-2563`）、LOCAL 跨实例迁移缺源实例（`:2627-2628`）。

| # | 分支（动作常量） | 触发条件 | 行为与数据流向 | 代码位置 |
|---|------|----------|----------------|----------|
| K1 | `NO_HISTORY` | `state is None`（会话首见） | 对 prefill 实例 `ensure_physical_fit(desired)`（无保护集）；阻塞 → `admission_blocked=True` 的 NO_HISTORY 决策（携带已完成的逐出）；放行 → 创建零字节新会话对象（`LOCAL_HBM`，全层驻留标记，`active=True`），事件 `no_history`，零传输 | `:2513-2560` |
| K2 | `LOCAL_HIT` | `location == LOCAL_HBM` 且 `instance_index == target` | 净需求 = `desired − existing`；fit 保护自身；放行 → 仅置 `active=True`，事件 `local_hit`，零传输零搬移。**PD 分离拓扑生产不可达的防御分支**（准入目标恒为 prefill 实例，历史 LOCAL 驻留只在 decode 实例），保留作单测替身路径 | `:2583-2623` |
| K3 | `NOC_MIGRATE` | `location == LOCAL_HBM` 且跨实例 | fit（保护自身）→ 阻塞即 blocked NOC_MIGRATE；放行 → `_noc_transfer` 构造 XY 路由 shard；**目标先加账再源释放**（`_add_shards(target)` → `_remove_shards(source)`，动态传输契约的目标先行序）；`instance_index = target`；metrics 段搬迁（target add 段先于 source remove 段）；事件 `noc_migrate` | `:2626-2695` |
| K4 | `REMOTE_RESTORE` | `location == REMOTE_MEMORY`（唯一远端路径，整会话全层恢复） | 远端账面缺失 RuntimeError（`:2704-2705`）；fit → 阻塞即 blocked REMOTE_RESTORE；放行 → `_remote_load_transfer`（层域 `[0, L)`）；目标加账 + `_remote_remove` 远端账面借记；状态迁移 `REMOTE_MEMORY → LOCAL_HBM`（`resident_prefix_layers = L`，远端账目清零）；事件 `remote_restore` | `:2697-2783` |
| K5 | 未知位置拒绝 | location 不在二态白名单（含 PARTIAL） | RuntimeError | `:2698-2702` |
| K6 | 驻留萎缩拒绝 | 净需求任一分量 < 0 | RuntimeError（历史准备不允许收缩） | `:2575-2576` |

适配层的命中口径映射（`slo_tools/kv_cache_adapter.py:100-140`）：`history_location_before ∈ {local_hbm→full, partial_hbm_remote→partial(旧日志), remote_memory→full}`；字段缺失回退 `history_action ∈ {NO_HISTORY→no_history, LOCAL_HIT→full, NOC_MIGRATE→full, RECOMPUTE→miss(旧基线)}`；未知值 → `not_supported`。`no_history` 不计入 miss；命中率双分母（有历史请求/全部请求）。

### 3.2 容量收敛通路（`ensure_physical_fit`，`session_kv_manager.py:2114-2284`）

| # | 分支 | 触发条件 | 行为 | 代码位置 |
|---|------|----------|------|----------|
| K7 | 输入向量校验 | 分片向量含非 int/负值/长度不符 | ValueError（`_require_int_vector` 拒绝 `int()` 静默截断） | `:2155-2158, 191-205` |
| K8 | 空实例不可容纳 | 任一 rank 上 `required > capacity − weight` | ValueError "cannot fit an empty instance"——任何逐出都救不了，属契约违例而非容量耗尽 | `:2195-2205` |
| K9 | 非候选驻留边际门 | 当前不足 **且** 逐 rank `needed ≤ available − own − held`（own = 保护/活跃会话驻留，held = 预约额） | 进入逐出循环；边际不满足则直接落入深缺口支路 K12——不执行注定无效的逐出（M1-10 无效逐出循环防御） | `:2210-2215` |
| K10 | 逐出循环 | 候选池非空且仍不足 | 按 LRU 序逐受害者整会话逐出，**每逐一个立即重查缺口**，首次满足即停（允许过量释放，从不追求字节精确边界）；候选耗尽即出循环 | `:2222-2236` |
| K11 | 受害者复核 | 返回前对全部受害者 | 逐个断言：非活跃、非保护、已落 REMOTE——任一不成立 RuntimeError（deep-gap 路径因边际门恒空转，是未来不变量破坏的第一道兜底） | `:2243-2256` |
| K12 | 深缺口（graceful exhaustion） | 循环后仍不足 | `deep_gap_events += 1` + 有界压力事件（`admission_blocked`）+ 返回 `CapacityResult(evictions, admitted=False, insufficient_ranks)`——**优雅推迟而非 raise**（与 sh 蓝图的 raise 语义刻意分叉；capacity_epoch 重试机制保证无死锁） | `:2257-2278` |
| K13 | 成功不变量 | 返回成功但仍不足 | RuntimeError（内部记账 bug） | `:2281-2284` |

### 3.3 LRU 淘汰/逐出链路（本仓与基础 wscllm 的核心差异点）

| # | 分支 | 触发条件 | 行为 | 代码位置 |
|---|------|----------|------|----------|
| K14 | 候选池谓词 | `_completed_local_candidates(instance)` | 四重过滤：`location == LOCAL_HBM` ∧ `instance_index == instance` ∧ `not active` ∧ `last_completion_ns is not None`——**活跃会话、远端会话、未完成会话、保护集会话永不入池** | `session_kv_manager.py:948-962` |
| K15 | LRU 序 | `_fifo_sort` | `(last_completion_ns, session_id)` 升序 = 纯 LRU，`session_id` 为确定性 tie-break；`last_completion_ns` 只经 `mark_complete` 写入（非负 int、会话内非递减），是 LRU 时序的唯一维护点 | `:934-946` |
| K16 | 整会话逐出执行 | 循环选中受害者 | 守卫：仅完成非活跃（`:1770-1771`）、仅全本地（`:1772-1773`）、层数必须为 L（`:1774-1776`）；动作序：构造 `remote_store` 传输（层域 `[0, L)`，`resident_prefix_layers_after=0`）→ 本地分片释放 `_remove_shards` → 状态迁移 `LOCAL_HBM → REMOTE_MEMORY`（instance=None、resident=0、远端账面 += 全量分片）→ `_remote_add` 远端池记账 → 增量不变量审计 → metrics 段整删 → 事件 `evict_session`。**单一逐出路径、无层段拆分、无两段式**——原 PARTIAL 半层化/后缀半逐出已物理删除（模块头 `:18-27`） | `:1757-1834` |
| K17 | 远端池逐 rank 记账 | 每次 `_remote_add/_remote_remove` | 长度/符号门（A10）+ 下溢检测（`remote-pool accounting underflow` RuntimeError）；journal 行带 `remote_delta_bytes` 列 | `:1079-1149` |
| K18 | 发射侧逐出形态 | graph 层消费 KVTransfer | `remote_store` shard 走两条物理链：**链 B（source == edge 直连）**单 `mem_store(hbm_access_mode=1)` = POOL_READ 唯一计费；**链 A（source ≠ edge）**源 `comm_send`（发送端 HBM 读计费）→ 边缘 `comm_recv(hbm_charge=False)` → 边缘 `mem_store`（池写零本地计费）→ 1B ack 对 | `generate_wsc_llm_trace.py:877-972` |
| K19 | 逐出发射时点分账 | 调度器侧 | `#2` 预占逐出：决策时点就近发射（准入成功 gated 于 prefill 段块末 / 阻塞无门立即）；`#4` move / `#5` grow_decode 逐出：运行时经预约不变量恒零，保留随 joiner 列车头发射的原路径；决策日志账本 `decode_target_evictions` 保留全集，`decode_evictions_pending` 为待发射子账本，`_decode_decision_dict` 以 prefill 行快照断言两行之间只增不减（防双计 fail-closed） | `wsc_llm_online_scheduler.py:1439-1448, 1666-1671, 887-897`；`graph_batch_builder.py:1183-1237` |

### 3.4 恢复链路（REMOTE_RESTORE 的物理通路）

| # | 分支 | 触发条件 | 行为 | 代码位置 |
|---|------|----------|------|----------|
| K20 | remote_load 物理链 | 历史恢复发射 | 1B arm/触发（control rank 与 edge 同 rank 直接 arm；跨 rank 1B p2p）→ 边缘 `mem_load`（仅池端口流体计时、零本地 HBM 计费）→ 边缘 `comm_send(hbm_charge=False)` → 目标 `comm_recv(hbm_charge=False)` → 目标 `local_hbm_kv_restore`（RESTORE 唯一数据计费，节点名含 `kv` 子串为 C++ code-7 锚点） | `generate_wsc_llm_trace.py:974-1061` |
| K21 | 就绪屏障 | 每次历史准备段 | 单一全层 `history_tp_ready_barrier`（恢复完成前不得消费历史 KV；整体恢复恒为单段全层传输，无 prefix/suffix 双栅栏） | `graph_batch_builder.py:1016-1023` |
| K22 | store→restore 前递边 | 回迁发射前查 `pending_store_tails` | 同会话在飞逐出的边缘 `mem_store` 完成必须先于其下一轮池读：同边缘 rank 直接 arm 依赖；跨边缘 rank 走 1B p2p 中继；补边完成即整条弹出（一次消费） | `graph_batch_builder.py:762-835` |
| K23 | 空表 + 探针命中 | tails 空 但调度器探针报告该会话仍有已决策未发射的 remote_store | RuntimeError（结构性时序违例，fail-closed） | `graph_batch_builder.py:789-800` |
| K24 | 无传输历史分支 | `history_transfers` 为空（NO_HISTORY/LOCAL_HIT） | 仅 arm 门；turn>0 到达此分支即 RuntimeError（PD 分离下 turn>0 恒恰一段迁移/恢复——防御断言） | `graph_batch_builder.py:965-994` |
| K25 | 传输数上界 | `len(history_transfers) > 1` | RuntimeError（二态下恢复恒单段） | `graph_batch_builder.py:962-964` |

### 3.5 驻留、共享与生命周期终点

| # | 分支 | 触发条件 | 行为 | 代码位置 |
|---|------|----------|------|----------|
| K26 | 完成驻留（retain） | `mark_complete` | 守卫：仅活跃本地会话可完成；完成时刻非负 int 且会话内非递减（回退即 ValueError——LRU 序的时间完整性）；`active=False`、`last_completion_ns=completion`；事件 `retain_complete`；**完成期零逐出**（恒空元组为既定架构） | `session_kv_manager.py:3026-3069` |
| K27 | 终态退休（retire） | REQUEST_COMPLETE 且无后继轮 | 释放该会话拥有的全部预约（两段式：全部逐 rank 校验通过才统一减账）；LOCAL 分支本地分片整删 + metrics 段删；REMOTE 分支**静默核销**（`_remote_remove`，不产传输、不是逐出）；删除会话对象与压力事件键 | `:3071-3208` |
| K28 | 退休守卫族 | 未知会话 KeyError / 活跃或未完成 / request_id 不拥有 / 预约跨会话持有 | 各自 fail-closed | `:3090-3115` |
| K29 | 预约登记/释放/扩张 | `reserve/release/extend_request_capacity` | 重复预约 RuntimeError；释放未知预约 RuntimeError（带上下文）；两段式下溢校验（先全查后减账，失败重试仍是同一笔已知预约——幂等）；extend 为净额回补侧（resident→reserved 1:1 换位，任意瞬间不超订） | `:2287-2455` |
| K30 | 终态断言 | `assert_final_state`（run 末） | 无残留预约、无压力键残留、无活跃会话 | `:3210-3220` |
| K31 | 共享面 | 会话 KV 逻辑上属会话私有 | 无跨会话共享 KV 分页；"共享"只发生在远端池的物理容量维度（逐 rank 账面 `_remote_bytes_by_rank`），逻辑归属始终单一会话（allocation_key = `remote:{session_id}` / `resident:{session}:{segN}`） | `:1109, 1458` |

### 3.6 水位控制（逐 rank HBM 账本）

| # | 分支 | 触发条件 | 行为 | 代码位置 |
|---|------|----------|------|----------|
| K32 | 水位公式 | 每次读取 | `used = weight + resident_kv + reserved_request_bytes`；`remaining = capacity − used`；逐 rank 不变量：负记账 RuntimeError、used > capacity RuntimeError（每次 mutation 后增量审计 + 全量审计两档） | `session_kv_manager.py:447-462, 1356-1366, 3276-3285` |
| K33 | 增量 vs 全量审计 | mutation 后 | 增量审计只触及受影响 rank/会话/预约（会话/rank/预约三账本逐 rank 对账）；`SH_STRICT_KV_INVARIANTS` 环境变量（或构造参数）开全量审计 | `:92-100, 1339-1404, 772-776` |
| K34 | 事件流水水位 | `compact_events(consumed)` | 已消费前缀 ≥ `_EVENTS_COMPACT_THRESHOLD`（8192）才整段删除，均摊 O(1)；`event_index` 由单调计数器分配，压缩不改已发射事件载荷；生产消费走 `events_since` 增量切片 | `:69, 822-842` |
| K35 | journal-off 旁路终态告警 | run 末且 recorder 未装/非 journal 模式 | 对"非活跃会话仍占留本地 HBM"打 stderr 结构化告警（刻意不 fail-closed，旁路态零数值影响）；journal-on 时该口径由守恒门强制 | `wsc_llm_online_scheduler.py:2092-2128, 2129-2139` |
| K36 | 水位重放（离线观测） | `slo_tools/hbm_watermark.py` 重放决策日志 | 逐出条目 → evict 动作（bytes 取 shard_bytes 列表优先、回退 total_bytes；victim 实例/时刻/层域类型谎报 fail-closed）；恢复列表逐段成动作（local_hit 段恒 0 字节）；occupancy 变点合并成水位曲线 | `hbm_watermark.py:240-413` |

### 3.7 delta journal（权威逐 rank 账本）

| # | 分支 | 触发条件 | 行为 | 代码位置 |
|---|------|----------|------|----------|
| K37 | 旁路态 | recorder 未装 或 非 journal 模式 | 装饰器透传：零快照、零回滚、零 journal 行（与未引入 journal 前逐字节一致） | `session_kv_manager.py:159-161` |
| K38 | 嵌套公开复用 | 事务深度 > 0 | 深度 +1 并入外层同一事务，不另起 id（事务粒度 = 调度器视角的一次公开 mutation） | `:162-167` |
| K39 | 最外层事务 | 公开 mutation 入口 | `begin_transaction` 取单调 id + manager 全量可变状态回滚快照；方法体正常返回 = commit 点，随即 `_journal_reconcile_transaction` 逐 rank 对账（manager 实际 (weight, resident, reserved, remote) vs journal before+Σdelta 推出的 after） | `:168-186, 1843-1881` |
| K40 | 失败路径 | 方法体异常 / journal 写失败 / 对账不一致 | manager 整体回滚到入口快照 + `abort_transaction`，原异常上抛——事务对 manager 真原子；已写出的 journal 行不可回收，由本轮 fail-closed 终止兜底 | `:172-186, 1930-1980` |
| K41 | 记录路径 | 每次 `_add_shards/_remove_shards/_remote_add/_remote_remove/预约/权重预载` 之后 | 逐 rank 流式落盘一行（schema v2：含 `remote_delta_bytes` 列与 before/after/capacity 快照）；时序护栏 `planner_time_ns` 非单调即 raise；分量下溢（负 component）即 raise；构造期权重预载为首 record（transaction_id=0, tick_zero anchor） | `metrics_integration.py:216-337`；`session_kv_manager.py:795-814` |
| K42 | anchor 映射 | 每 mutation | `_metrics_anchor_for_phase`：history/prefill/admission/prefill_admission → `prefill_start`；decode → `decode_start`；其余 → `stage_boundary`；release 的 audit anchor 由调用方显式传（decode 准入边界传 `decode_start`） | `session_kv_manager.py:111-127`；`wsc_llm_online_scheduler.py:1656-1657` |
| K43 | run 末 checksum 门 | `verify_journal_checksum`（journal-on） | 流式重放 journal：行数一致、逐 rank 终态一致、未知 rank 拒绝、**守恒断言**（resident==0 ∧ reserved==0 ∧ physical==weight ∧ remote==0——retire 静默核销后远端账面同归零）；产物 `kv_delta_journal_checksum.json`；任何不一致 RuntimeError | `session_kv_manager.py:1982-2110` |
| K44 | journal 总开关 | `KV_DELTA_JOURNAL` 环境变量（缺省 on；off 完全旁路） | 唯一开关；off 时与改前行为逐字节一致 | `online_service.py:69-79` |

### 3.8 事件流与压力事件

| # | 分支 | 触发条件 | 行为 | 代码位置 |
|---|------|----------|------|----------|
| K45 | kv_actions 流 | 每决策批末 | `events_since(cursor)` 增量发射到 GraphBatch.kv_actions；发射后按水位压缩并同步回退游标；先入 provisional 暂存，commit ack 后 finalize（C++ 原子提交确认） | `wsc_llm_online_scheduler.py:632-641`；`online_scheduler_base.py:470-475, 1092-1094` |
| K46 | 压力事件去重 | 同 (event_type, phase, request, target) 重复 | 首次观测记 `admission_blocked`；其后同键仅补一条 `admission_retry`；完成/退休时按 request 产权索引 O(keys) 清键 | `session_kv_manager.py:1005-1058` |
| K47 | 深缺口计数口径 | 每次 fit 尝试失败 | `deep_gap_events` 按"每次尝试"递增不去重（同一请求经纪元重试再撞深缺口重复计数）——语义为阻塞事件数而非阻塞请求数，纯观测列不进策略判据；由 online_service summary 行落盘 | `:2259-2266`；`online_service.py:331-332` |

### 3.9 pending 门位置镜像（决策时点同步）

| # | 分支 | 触发条件 | 行为 | 代码位置 |
|---|------|----------|------|----------|
| K48 | remote_store 镜像 | 每次 KV 变更点返回逐出 | `sync_pending_history_after_evictions` 仅处理 `kind == "remote_store"` 的传输，立即把 pending 门位置改写为 `remote_memory`（决策时点同步是唯一标记路径，发射侧不重复标记） | `graph_batch_builder.py:559-567` |
| K49 | 半层残留拒绝 | `resident_prefix_layers_after != 0` | RuntimeError（整会话逐出必须清零本地驻留） | `:569-573` |
| K50 | 无门延迟通道 | 会话无 pending 请求或无门 | 位置写入 `deferred_session_locations`，延迟到下一 gate 登记点消费（turn-0 防御通道，二态管理器下生产不可达——turn-0 在飞会话恒活跃无完成时刻，LRU 候选池双重排除） | `:574-593` |
| K51 | 位置对账 | 下一轮 prefill 发射 | `pending_gate.location` 必须等于准入时点快照 `history_location_before`，不一致即 RuntimeError（sync 接线缺口的 fail-closed） | `:930-938` |

---

## 四、配置开关与控制点清单

| 控制点 | 值域/缺省 | 位置 |
|--------|-----------|------|
| `kv_cache_policy` | 唯一值 `session_lru_tiered`，其余 fail-closed | `generate_wsc_llm_trace.py:253-264`；`online_service.py:226-260`；`wsc_llm_online_scheduler.py:392-396` |
| `trace_granularity` | 唯一值 `request_aggregated` | `generate_wsc_llm_trace.py:265-274` |
| `SH_STRICT_KV_INVARIANTS` | 1/true/yes/on 开全量审计，缺省关 | `session_kv_manager.py:92-100` |
| `KV_DELTA_JOURNAL` | 缺省 on；off/off 族完全旁路 | `online_service.py:69-79` |
| `SH_TRAIN_MAX_ITER` | 列车长度上限，缺省 8；0 不设限；非法词法/负值构造期 fail-fast | `wsc_llm_online_scheduler.py:521-538` |
| `SH_FIRST_TOKEN_SPLIT` | 缺省 "0" 关；"1" 开两段式首 token 拆分 | `graph_batch_builder.py:79-100` |
| `SH_GRAPH_DIGESTS` / `SH_PROFILE_JSONL` | 缺省关 | `online_service.py:82-88, 200-202` |
| `SH_PROPAGATING_TAIL_LIMIT` | 在途尾部上限，缺省 8（桥背压窗口 8x） | `online_scheduler_base.py:54-55, 331-341` |
| `--sensing` | 感知账本开关（查询/审计输入，不进策略判据） | `online_service.py:159-163` |
| `prefill_chunk_size` / `mlp_variant` | 512 / gelu\|swiglu | `generate_wsc_llm_trace.py:88-100, 275-278` |
| `MAX_EQUIVALENT_SHORTEST_PATHS` / `MAX_ROUTE_CANDIDATE_COMBINATIONS` | 64 / 4096（路由枚举上界，fail-closed） | `wsc_llm_scheduler.py:32-33` |
| `_EVENTS_COMPACT_THRESHOLD` | 8192（事件前缀压缩水位） | `session_kv_manager.py:69` |
| tag 段 | KV 传输分配器 [100_000_000, 500_000_000) 半开段 | `generate_wsc_llm_trace.py:204-229` |

---

## 五、两条机制在体系结构中的交汇点

**映射决策受 KV 状态约束**：请求自入口通路进入映射组件后，Prefill 实例选择（最短 FCFS 队列）与静态 P→D 路由（向心一跳布局）本身不读 KV 状态，但**准入通路立即被 KV 账本门控**——静态 decode 终态预约（`reserve_request_capacity`）在请求获得发射资格之前必须通过 `ensure_physical_fit` 的水位裁决；净额修正分支（M38）读取会话历史位置快照（LOCAL 且驻留 decode 目标）才允许净额重试；`prepare_history` 的四分支完全由会话 KV 位置状态（ABSENT/LOCAL 同实例/LOCAL 跨实例/REMOTE）决定请求走哪条恢复支路。容量纪元（`capacity_epoch`）是约束的反向传播通道：任何 KV 状态变化点（逐出、迁移、恢复、退休）都唤醒受影响实例的纪元，解锁曾在该实例上被拒的准入重试——映射决策的重试时机由 KV 状态变迁驱动。

**KV 状态被映射决策改变**：每一次成功的准入都在三个位置改写 KV 状态机——decode 目标实例的预约账本（resident→reserved 1:1 换位链）、prefill 实例的会话驻留（prepare_history 迁入 + grow_prefill 增长）、以及可能牺牲的旁路会话（整会话 LRU 逐出至远端池）。P→D 交接边界（`_try_admit_waiting_decodes`）完成预约释放与物理迁移的闭环，会话的 `instance_index` 随之从 prefill 实例改写为 decode 实例。终态交汇于 REQUEST_COMPLETE：有后继轮则 KV 驻留被保留并经 pending 门把位置事实传递给下一轮的命中判定，无后继轮则整会话连同远端账面一并核销——delta journal 以逐 rank 守恒断言（resident=0/reserved=0/remote=0）为这一双向耦合的闭合盖印。
