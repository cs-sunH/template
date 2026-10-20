# request 映射策略与历史 KV 管理机制介绍

## 0. 架构总览

整个系统是一条自输入清单通往物理仿真时钟的分层通路,共五层:

- **输入/清单层**:`generate_trace.py:load_request_queue` 装载并校验请求队列 CSV(会话/turn/到达时刻/间隔),`generate_wsc_llm_trace.py:load_wsc_llm_trace_config` 解析 `trace_config.csv`(模型、硬件剖面、inference_group、策略键),`plan_materializer.py` 把队列派生成 `manifest.json` 的九字段请求事实(`history_tokens_before/prefill_context_tokens/final_context_tokens` 等链式推导)。C++ 侧 `WindowedTraceReader` 以同一数据行序编号 `queue_index`,两端同源。
- **调度决策层**(Python):`wsc_llm_scheduler.py` 提供拓扑装配(`build_instances`)、实例邻接图(`InstanceGraph`)、静态 P→D 路由(`build_static_pd_mapping`)与 Prefill 队列选择语义(`select_prefill_instance`);`online/online_scheduler_base.py` 提供批次框架(交付序号门、completion 核销、ack 记账);`online/wsc_llm_online_scheduler.py` 是唯一在役策略变体(`kv_cache_policy == "session_lru_recompute"`),每决策批次按固定批序执行 completion 批 → arrival 批 → 准入/发射 pass。
- **KV 管理层**:`session_kv_manager.py:SessionKVCacheManager` 是会话 KV 的权威状态机(RESIDENT/EVICTED 二态、逐 rank HBM 三账户 weight/reserved/resident、LRU 被动逐出),`metrics_integration.py:MemoryActionRecorder` 在 journal 模式下构成权威 HBM delta 账本旁路。
- **图发射/桥接层**:`online/graph_batch_builder.py` 把策略决策物化为 GraphBatch 节点/边(prefill 整段、decode 迭代列车、KV transfer 1000/3000、控制触发 1900);`online/decision_bridge.py` + `online/online_service.py` 经 FIFO 文件协议与 C++ 交换 request/response/commit_ack。
- **模拟器执行层**(C++,`astra-sim/workload/execution_driven/`):`RequestIngress` 持有到达 alarm 与命令队列,`ServiceCoordinator` 维护 IDLE/ACTIVE/DRAINING/FINISHED 生命周期,`GraphBatchCommitter` 校验并原子提交 GraphBatch(节点按 `rank` 字段注入对应 NPU),`WatchRegistry` 在 watch 节点终态时把完成事实推回决策层。

数据流:请求自清单进入 → 到达 alarm 触发 ARRIVAL 决策事件 → 调度器把请求映射到 Prefill 实例并经静态路由绑定 Decode 实例 → KV 管理层执行历史命中/准入/迁移决策 → 构图器发射节点 → C++ 执行层消费节点、按真实完成事件回推 DECODE_COMPLETION/REQUEST_COMPLETE → 下一 turn 排程。控制流反向:所有策略判据只读 Python 账本(队列深度、KV 账本、capacity_epoch),C++ 只推进完成边界。观测旁路(kv_cache_adapter/hbm_watermark)只重放日志,不回馈决策。

映射关系总纲:**请求 → Prefill 实例**(到达时按队列深度最小选);**Prefill 实例 → Decode 实例**(构造期静态最短路唯一绑定);**请求 → Decode 实例**(继承静态路由);**会话 KV → 实例**(随请求准入在 P/D 实例间迁移,完成后驻留 Decode 实例直至被动逐出或终局回收)。

---

## 1. request 映射策略全分支枚举

### 1.1 拓扑装配通路(`wsc_llm_scheduler.py:build_instances`,149-266)

配置的 inference_group 是放置单元:每个实例保持六 rank TP 通信域,整场运行独占 Prefill 或 Decode 角色。

| # | 分支 | 触发条件 | 行为/数据流向 | 位置 |
|---|------|---------|--------------|------|
| T1 | 空规格拒绝 | `specs` 为空 | 构造期 ValueError(fail-closed) | wsc_llm_scheduler.py:153-154 |
| T2 | 实例名字段拒绝 | `spec.name` 空串 | ValueError | :164-165 |
| T3 | pg_name 非法/重复 | pg_name 为空或 "0",或重复 | ValueError | :166-169 |
| T4 | phase_role 非法 | 非 `prefill`/`decode` | ValueError | :171-174 |
| T5 | 无 rank/重复 rank | `spec.ranks` 空或含重复 | ValueError | :175-178 |
| T6 | rank 归属冲突 | 同一 rank 被两个实例声明 | ValueError | :181-185 |
| T7 | 非矩形填充 | 坐标集不构成轴对齐实心矩形 | ValueError | :191-203 |
| T8 | 覆盖不全 | 实例并集 ≠ 全部 NPU | ValueError("must cover every configured NPU exactly once") | :218-219 |
| T9 | 实例规模不等 | 实例间 rank 数不一致 | ValueError(直接 KV shard 配对要求等大) | :220-223 |
| T10 | 邻接图不连通 | 实例 tiling 未形成连通邻接 | ValueError | :239-240 |
| T11 | 角色缺失 | 无 Prefill 或无 Decode 实例 | ValueError | :242-249 |
| T12 | 向心布局拒绝 | max(Decode 中心距离) > min(Prefill 中心距离) | ValueError("Decode instances must be center-prioritized") | :250-260 |

通过后产出 `WscLlmTopology(instances, adjacency)`,并预计算每实例的 `wafer_center_manhattan_distance`(:204-215)。

### 1.2 静态 P→D 路由通路(`build_static_pd_mapping`,386-529;`InstanceGraph`,294-358)

| # | 分支 | 触发条件 | 行为 | 位置 |
|---|------|---------|------|------|
| R1 | 空 Prefill fail-closed | `prefill_indices` 为空 | ValueError(不再静默返回空映射) | :398-400 |
| R2 | 无路径拒绝 | BFS 不可达 | ValueError("no instance path") | :328-329 |
| R3 | 等价最短路爆炸拒绝 | 某 (source,node) 等价最短路数 > `MAX_EQUIVALENT_SHORTEST_PATHS`(64) | ValueError(拒绝组合爆炸) | :32, 345-351 |
| R4 | 候选组合爆炸拒绝 | Prefill 候选笛卡尔积 > `MAX_ROUTE_CANDIDATE_COMBINATIONS`(4096) | ValueError | :33, 419-428 |
| R5 | 最近 Decode 候选集合 | 距离等于 `nearest_distance` 的全部 (decode_index, path) 进入候选 | 候选按 (decode_index, path) 排序 | :404-417 |
| R6 | 目标函数选择 | 全枚举后按 `(total_hops, shared_occurrences, signature)` 字典序取最小 | 唯一确定性路由表 `StaticPdMapping.routes` | :448-455 |
| R7 | 多跳契约拒绝 | 被选中路由任一 hop_count ≠ 1 | ValueError("centripetal P/D layout contract violated ... one hop") | :484-495 |
| R8 | 共享物理边拒绝 | 任一物理边被两条路由共享 | ValueError("handoff edges must be pairwise disjoint") | :496-504 |

`route_for_prefill`(:379-383)按 Prefill 实例索引查表,查无即 KeyError——运行期路由恒定,不做动态重映射。

### 1.3 到达边界:请求 → Prefill 实例映射(`wsc_llm_online_scheduler.py`)

映射谓词在 `select_prefill_instance`(wsc_llm_scheduler.py:546-549):取 `PrefillQueueSnapshot.ordering_key = (request_count, instance_index)`(:541-543)的最小者——**先按排队深度最浅,再按实例索引最小 tie-break**。快照仅覆盖 `phase_role == PREFILL_ROLE` 的实例(`_prefill_snapshots`,wsc_llm_online_scheduler.py:1614-1622)。

`_on_arrival`(:991-1013)的固定通路:记录 `estimated_arrival_ns` → 取快照(在 append 之前,使 ordering_key 反映选择时刻深度)→ `select_prefill_instance` → `route_for_prefill(selected)` → 写 `runtime.prefill_instance_index / prefill_assignment_key / static_route`,并**当场把 `runtime.decode_instance_index` 绑定为静态路由的 Decode 端**(:1006)→ 入该实例 `qp`(FCFS deque)→ ready frontier 登记 → 感知账本 `prefill_qp` 层登记。

到达次序由 arrival min-heap 决定:`_push_arrival`(:1161-1171)键为 `(tick, priority=1, queue_index, sequence, "arrival", payload)`,同 tick 到期项按冻结队列序稳定弹出;`_drain_arrival_heap`(:1173-1184)只消费 `tick <= current_tick` 的项,堆内出现非 "arrival" 项即 RuntimeError。

同 turn 链:`_on_request_complete`(:1105-1157)经 `next_request[(session_id, turn_index+1)]` 索引查后继 turn;有后继 → 向 ingress 注册 `future_alarms`(到达时刻 = 本完成 tick + `inter_request_interval_ns`,间隔经 `queue_position_by_index` 映射定位,`_interval_ns`,:1667-1677);无后继(terminal turn)→ `retire_completion_gate` + `retire_terminal_session` 回收 KV(见 §2.7)。

### 1.4 Prefill 准入通路(`_try_admit_prefill`,wsc_llm_online_scheduler.py:1236-1400)

| # | 分支 | 触发条件 | 行为/数据流向 | 位置 |
|---|------|---------|--------------|------|
| A1 | 幂等直通 | `runtime.admitted_prefill` 已置位 | 返回 True | :1241-1242 |
| A2 | 映射丢失 fail-closed | `prefill_instance_index` 或 `static_route` 为 None | RuntimeError | :1243-1245 |
| A3 | **capacity_epoch 重试门** | `runtime.prefill_attempt_epoch == (epoch[prefill], epoch[decode])` | 返回 False,本 tick 不重试 | :1248-1253 |
| A4 | 全量预约 | epoch 变化 | `reserve_request_capacity(decode_instance, final_context_tokens 全量 shard)`(终态 KV 落位 Decode 实例的防抢占预占),逐出累积进 `decode_target_evictions` | :1255-1268 |
| A5 | **净额修正重试** | 全量预约 `admitted=False` 且旧会话 KV `RESIDENT` 于 decode 目标 | 以 `max(0, final−旧驻留)` 净额重试预约;成功则 `runtime.reservation_credit = credit` | :1269-1306 |
| A6 | 净额重试仍失败 | 净额 `admitted=False` | 合并两次尝试逐出实例的 epoch 唤醒(`_note_capacity_change`),返回 False 等待重试 | :1294-1305 |
| A7 | 普通失败路径 | 全量失败且旧 KV 不在 decode 目标 | 逐出非空则 epoch 唤醒,返回 False | :1307-1314 |
| A8 | 历史状态快照 | 预约成功 | `history_cache_state_before = "ABSENT"`(快照 None)或快照 state(RESIDENT/EVICTED) | :1315-1319 |
| A9 | `prepare_history` | 无条件 | 三支历史动作(见 §2.1) | :1320-1329 |
| A10 | **历史准入阻塞回滚** | `decision.admission_blocked` | `release_request_capacity` 释放预约、`reservation_credit` 失效、epoch 唤醒覆盖 prefill/decode/历史源/两次逐出实例,返回 False | :1330-1345 |
| A11 | **净额回补** | `reservation_credit` 非 None 且历史源 == decode 实例 | `extend_request_capacity(credit)` 把净额预约回补到全量(resident→reserved 1:1 换位);源不符即 RuntimeError | :1346-1364 |
| A12 | `grow_prefill` | 历史就位后 | 增长到 `prefill_context_tokens`;`admitted=False` 即 RuntimeError("Prefill growth lost its successful capacity preflight")——预检已过,失败只可能是不变量破坏 | :1372-1379 |
| A13 | 成功提交 | 全部通过 | `admitted_prefill=True`,epoch 唤醒,返回 True | :1380-1400 |

`capacity_epoch` 是实例级计数器(`_note_capacity_change`,:1624-1628):任一相关实例(P/D/逐出受害者所在)发生容量事件即 +1,是阻塞请求被重新考虑的唯一唤醒机制。

### 1.5 Decode 等待准入通路(`_try_admit_waiting_decodes`,wsc_llm_online_scheduler.py:1402-1484)

SCHED-1 裁决:decode 侧 capacity_epoch 重试门已删除(等待成员目标容量已在 Prefill 准入时预占,move/grow 不成立即 raise,无失败重试语义),仅保留 dirty 标记作为 append 契约活断言。

| # | 分支 | 触发条件 | 行为 | 位置 |
|---|------|---------|------|------|
| D1 | 目标选择 | 实例队列非空且在 `decode_admission_dirty` | 按实例索引序处理,弹出 dirty | :1416-1424 |
| D2 | 队列腐化 fail-closed | 成员 `decode_instance_index != target_instance` | RuntimeError | :1429-1430 |
| D3 | 未预占 fail-closed | `decode_capacity_reserved` 为 False | RuntimeError("not reserved before Prefill") | :1431-1433 |
| D4 | 先释放后迁移 | 无条件 | `release_request_capacity(anchor_phase="decode")`(§4.4 先释放预留、后 P→D 迁移落位) | :1436-1439 |
| D5 | `move_prefill_to_decode` | 无条件 | 见 §2.4;`admission_blocked` 即 RuntimeError("released static Decode reservation was not physically usable") | :1440-1452 |
| D6 | `grow_decode` | 迁移后 | 增长到 `final_context_tokens`;失败即 RuntimeError | :1454-1464 |
| D7 | 入批 | 全部通过 | 记录 `decode_queue_depth_before_enqueue`,append 进 `active_decode`(双侧同步 lookup 集合),非忙则入 ready frontier,账本转 `active_decode` 层,epoch 唤醒 | :1465-1484 |

### 1.6 准入/发射 pass 与批次组成(`_admit_pass`,wsc_llm_online_scheduler.py:1188-1234)

固定次序:先 `_try_admit_waiting_decodes`,再按 `sorted(self._ready_frontier)` 逐实例服务(实例索引序保证决策确定性)。

| # | 分支 | 触发条件 | 行为 | 位置 |
|---|------|---------|------|------|
| E1 | frontier/busy 失步 fail-closed | frontier 内实例 `_instance_busy` | RuntimeError(frontier/busy desync) | :1207-1211 |
| E2 | **PREFILL_ROLE 支路** | `phase_role == PREFILL_ROLE` | 校验无 decode 混入(:1213-1215);`qp` 空则 skip(:1216-1217);队首 `_try_admit_prefill` 失败则 skip 本 tick(:1219-1220);成功则 `_emit_prefill` 整段发射,`busy=True`(P 侧 busy = 一个 prefill 整段在飞) | :1212-1223 |
| E3 | **DECODE_ROLE 支路** | `phase_role == DECODE_ROLE` | 校验无 prefill 混入(:1225-1227);`active_decode` 空则 skip(:1228-1229);否则 `_emit_train` 冻结并发射迭代列车(D 侧 busy = 一个列车在飞) | :1224-1234 |

busy 语义按角色分裂(`_instance_busy`,:1636-1643):P 实例看 `state.busy`,D 实例看 `in_flight_train is not None`。

**列车组成(`_plan_train`,:704-761)**:

| # | 分支 | 触发条件 | 行为 | 位置 |
|---|------|---------|------|------|
| E4 | 成员余额异常 fail-closed | 某 `active_decode` 成员剩余 token ≤ 0 | RuntimeError | :716-720 |
| E5 | 无工作 | members 空 | 返回 None(不发射) | :723-724 |
| E6 | 迭代数 | `iterations = max(成员剩余 token)` | 列车终点 = 全部 decode 工作耗尽 | :725 |
| E7 | **T_max 截断** | `SH_TRAIN_MAX_ITER`(缺省 8;0=不设限)非零且 iterations > 上限 | 截断到上限,`capped=True` | :410-427, 726-729 |
| E8 | 退出成员判定 | `participation >= remaining` | 该成员在本列车内挂 exit 标记,列车边界不变 | :736-743 |
| E9 | **哨兵标记** | `capped` 且无自然 exit 成员 | signal_set 追加 `batch_train_i<inst>_<seq>` 批命名空间哨兵 | :744-748 |

**首 token 拆分(`_first_token_plan`,:656-702;开关 `first_token_split_enabled`,graph_batch_builder.py:75-96,`SH_FIRST_TOKEN_SPLIT` 缺省 "0" 关)**:

| # | 分支 | 触发条件 | 行为 | 位置 |
|---|------|---------|------|------|
| E10 | 开关关闭 | `SH_FIRST_TOKEN_SPLIT != "1"` | 返回 None,行为与拆分上线前逐字节一致 | graph_batch_builder.py:96 |
| E11 | debut 不变量 fail-closed | 某 joiner `decode_tokens_consumed > 0` | RuntimeError(joiner 恒为零消费) | wsc_llm_online_scheduler.py:674-677 |
| E12 | 无 debut | joiners 空 | 返回 None | :678-680 |
| E13 | debut 分类 | `decode_length == 1` 与否 | 多 token debut 挂独立 first_token 标记;`decode_length==1` 的 debut 其 exit 标记名附加 first_token 子串(同节点 code 4/8 双锚点) | :681-687 |
| E14 | 两段式拆分 | `iterations >= 2` | 物理拆首步批 + 余量批(split=True);否则仅做不拆车的标记增强 | :688-702 |

**发射通路(`_emit_train`,:795-895)**:计划 None 直接返回(:811);joiner 账本(决策日志/issue 登记/`decode_target_evictions` 序列化后置空核销)先于发射(:832-859);split 分支走 `_emit_train_first_step`(:924-951)后提前返回(:862-869);整列发射注册每退出成员一个 watch(statuses `["Success","Skipped"]`)+ 哨兵 watch(固定 prefill 单事件通道,:873-891);挂 `in_flight_train` 并退出 frontier。

**列车核销(`_finalize_completed_trains`,:530-621)**:首个到达信号执行核销(闭式推进每成员 `decode_tokens_consumed += participation`,exit 成员移出 `active_decode`,清 `in_flight_train`);残余信号登记进 `finalized_trains` 的 pending 集合,后续交付幂等清理;信号不属任何在飞/已核销列车 → RuntimeError(stale completion mismatch)。busy 门解除后仍有成员则回 frontier,否则出 frontier(:603-606)。

**首步唤醒剥离(`_consume_first_step_wakeup`,:623-630)**:`batch_train_*_first_step` 信号是无操作交付回声,识别后从哨兵核销通道剥离,并在 digest 行打 first_step 标记(:481-492)。

### 1.7 图发射层的映射分支(`graph_batch_builder.py`)

| # | 分支 | 触发条件 | 行为 | 位置 |
|---|------|---------|------|------|
| G1 | 粒度拒绝 | `trace_granularity != "request_aggregated"` | RuntimeError(在线发射只支持聚合粒度) | :357-360, 526-529, 567-570, 606-609 |
| G2 | **turn-0 到达门** | `turn_index == 0` | 每 prefill rank 挂 arrival timer gate;`session_arrival_time_ns` 缺失即 RuntimeError;duration==0 不发节点(runtime=0 的 gate 由 C++ alarm 替代等待) | :389-401; timer_gate :209-256 |
| G3 | **turn>0 间隔门** | `turn_index > 0` | 从 `completion_gates` 弹出上一完成列车的 post-barrier 节点(取用即删);账本缺失或 interval None → RuntimeError;duration==0 直接以 after_node_id 建依赖 | :402-420 |
| G4 | **NOC_MIGRATE 历史迁移发射** | `history_action == NOC_MIGRATE` | 源缺失即 RuntimeError;`source_group != previous_group`(跨实例唤醒链,本仓映射不变量下不可达,保留为防御)→ 先发 control trigger 再清 timer;然后 `_paired_transfer(category=1000)` 逐 shard send/recv | :421-449 |
| G5 | 非迁移间隔控制 | 其他 history_action | `_emit_control_trigger`(1 字节控制依赖,category 1900) | :450-456 |
| G6 | **RECOMPUTE 重算段** | `history_action == RECOMPUTE` | 在 prefill 组上发射 `history_recompute` 阶段(initial_context_tokens=0) | :457-463 |
| G7 | 当前 prefill 段 | 无条件 | 按 `prefill_chunk_size`(缺省 512)切 chunk span,聚合发射;tokens==0 返回空界 | :464-477; generate_wsc_llm_trace.py:569-609 |
| G8 | **joiner P→D 迁移** | 列车头内每 joiner | `_paired_transfer(category=3000)`,bytes = `kv_cache_bytes_for_tokens(prefill_context_tokens)`(整段上下文迁移);同组则无节点 | :621-644 |
| G9 | 哨兵/exit/end barrier | 列车尾 | exit 标记(或带 first_token 子串)、哨兵标记(仅 sentinel)、共享 end barrier;`completion_gates[session] = (instance, block_ends)` 供下一 turn 间隔门消费 | :707-786 |
| G10 | 完成门释放 | terminal session | `retire_completion_gate` 弹出账本 | :377-379 |

**`_paired_transfer`(generate_wsc_llm_trace.py:471-538)内部**:TP 度不等 → ValueError(:483-484);同组 → 无 send/recv、仅 arm timer gate(:485-489);`total_bytes % num_heads == 0` → 整头归属切分,否则精确字节切分兜底(总量绝不上取整,:494-505);每 shard 伴随 `_xy_route` 维度序路由(X 先 Y 后,:458-468)与 `_stage_tag` 域界守卫(queue_index×10000 块宽 + C++ NATIVE 窗 [0,5e8),越界即 ValueError,:434-455)。

**权重遍数口径**:`transformer_pass_aggregated` 的 `weight_passes`(generate_trace.py:778-819)——缺省 = len(spans)(P 侧每 chunk 读一遍权重,chunk 间不互拼);D 侧列车传迭代数(权重每迭代只读一次);`weight_passes > len(spans)` 即 ValueError。首 token 拆分时首步批 weight_passes=1、余量批 = iterations−1(graph_batch_builder.py:576, 614-616)。

**prefill 整段发射的决策行分支(`_emit_prefill`,wsc_llm_online_scheduler.py:1488-1588)**:`history_transfer_bytes` 非零时附 `noc_hops` 字段(实例图最短路,只读派生,:1548-1555);台账行迭代数 = `prefill_chunks + recompute_chunks`,其中 `recompute_chunks` 仅当 `history_action == RECOMPUTE` 时按重算 token 折算,否则为 0(:1566-1570)。

### 1.7b 完成边界的三段核销(wsc_llm_online_scheduler.py)

| # | 分支 | 触发条件 | 行为 | 位置 |
|---|------|---------|------|------|
| K1 | **PREFILL_DRAIN 边界**(`_on_prefill_drain`,1014-1040) | prefill 整段物理完成 | FCFS 序校验(队首非本请求即 RuntimeError,:1021-1022);`busy=False`、qp 首出;队列仍非空 → 回 frontier,否则出(:1025-1028);Decode 目标丢失 → RuntimeError(:1029-1030);登记 `waiting_decode_admissions[decode_instance]` + `decode_admission_dirty.add`(append→dirty 恒配对,:1031-1035);账本转 `waiting_decode` 层 |
| K2 | **DECODE_COMPLETION 边界**(`_on_decode_complete`,1042-1103) | exit 标记 watch fire | `mark_complete`(§2.5);decode 实例 epoch 递增(:1064);快照缺失 → RuntimeError("completed session disappeared",:1066-1067);completion 决策行(冻结键 `completion_evictions=[]`);M4 核销即删全部 KV 胖字段 |
| K3 | **REQUEST_COMPLETE 边界**(`_on_request_complete`,1105-1157) | 同 tick 随 DECODE_COMPLETION 交付 | decode 账本完整性检查(`decode_tokens_consumed != decode_length` 或未加入列车 → RuntimeError,:1114-1118);有后继 turn → future alarm;terminal → 完成门回收 + 终局回收 + 容量 epoch 发布;runtime/索引/next 链全部释放(:1149-1157) |

### 1.8 交付协议与映射防护(`online_scheduler_base.py` / `decision_bridge.py`)

| # | 分支 | 触发条件 | 行为 | 位置 |
|---|------|---------|------|------|
| P1 | **幂等重放** | `seq == last_applied_sequence` | 深比对 delta(不一致即协议违例),一致则深拷贝返回缓存 GraphBatch——不重新产生任何 assignment/KV action/图节点,状态零变更 | online_scheduler_base.py:473-496 |
| P2 | 序号跳变 fail-closed | `seq != last_applied_sequence + 1` | ValueError(单调 +1;重复旧 seq 亦违例) | :497-501 |
| P3 | 重复到达 fail-closed | request 不在 unseen 集或在 in_flight | ValueError | :882-894 |
| P4 | 未知请求完成 fail-closed | completion 的 request 无 pending | ValueError;列车哨兵(`batch_train_` 前缀)旁路由给变体核销 | :846-880 |
| P5 | **B4 在途尾部上限** | `SH_PROPAGATING_TAIL_LIMIT`(缺省 8)超限 / arrivals 超 expected_request_count | RuntimeError(fail-closed,真实在途工作禁止截断) | :56-57, 60-130, 312-332 |
| P6 | **commit ack 门** | schema/batch_id/success=false/领先应用/无 pending 记录 | 各自 ValueError;幂等 ack(水位或乱序洞内)直接返回 | :965-1038 |
| P7 | provisional KV 账本 | 本批 kv_actions 非空 | 深副本入暂存区,ack 到达才 finalize;run 末暂存区非空即 fail-closed | :453-458, 1099-1107 |
| P8 | 桥协议严格单调 | request/ack 序号旧重复或未来序 | 重复幂等忽略,未来序 BridgeError → error response + 非 0 退出 | decision_bridge.py:106-130, 318-331, 383-396 |
| P9 | **ack 通道失败路由** | `_handle_ack` 异常 | 错误落到下一必读槽位 `response_<ack_seq+1>`(写到已消费槽位永不可达) | :442-486, 519-535 |
| P10 | request 通道失败 | handler 异常 | error response 写当前槽位 + 非 0 退出;BrokenPipe(C++ 已死)单独留痕退出 | :408-440, 536-550 |

### 1.9 配置开关与值域门(映射相关)

| # | 开关/键 | 值域与缺省 | 位置 |
|---|--------|-----------|------|
| C1 | `kv_cache_policy` | 仅 `session_lru_recompute`,其余 fail-closed | generate_wsc_llm_trace.py:190-198;online_service.py:226-260;wsc_llm_online_scheduler.py:292-296 |
| C2 | `kv_eviction_mode` | 残留键直接拒绝(主动驱逐已物理清除,不得静默忽略或复活水位逐出) | test_wsc_llm_scheduler.py:351-368(loader 未知键 fail-closed) |
| C3 | `trace_granularity` | `token_expanded` \| `request_aggregated`(在线发射仅后者) | generate_wsc_llm_trace.py:199-205;graph_batch_builder.py:357-360 |
| C4 | `request_queue_session_limit` | 非负整数,0=不过滤;limit 超过可用会话数 fail-closed;过滤后经 `queue_index_mappings` 保持 queue_index 与 C++ 同源 | generate_wsc_llm_trace.py:186-187, 233-288 |
| C5 | `remote_operand_loads` | 必须为 false(本仓无远端权重池) | :172-185 |
| C6 | `prefill_chunk_size` | 缺省 512 | :188-189 |
| C7 | `kv_reserve_context_tokens` | 遗留观测键,决策管线零消费,仅 hbm_watermark 读 CSV 作水位口径 | :74-81;hbm_watermark.py:680 |
| C8 | `SH_TRAIN_MAX_ITER` | 缺省 8;0=不设限;空串/负数构造期 fail-fast | wsc_llm_online_scheduler.py:410-427 |
| C9 | `SH_FIRST_TOKEN_SPLIT` | 缺省 "0" 关;"1" 启用拆分 | graph_batch_builder.py:75-96 |
| C10 | `KV_DELTA_JOURNAL` | 缺省 on;off 完全旁路(不装 recorder) | online_service.py:70-80 |
| C11 | `SH_STRICT_KV_INVARIANTS` | 缺省关;开则每次 mutation 后跑完整审计 | session_kv_manager.py:47-55 |
| C12 | `SH_GRAPH_DIGESTS` / `SH_PROFILE_JSONL` | 缺省 "0" 关 | online_service.py:83-89, 204-206 |
| C13 | `SH_PROPAGATING_TAIL_LIMIT` | 缺省 8(8x 桥背压窗口) | online_scheduler_base.py:56-57 |

请求队列装载侧的身份门(generate_trace.py:181-383):空 request_id / 重复 request_id / 保留子串(`first_token`) / `batch_train_` 前缀 / turn-0 缺 session_arrival 或带 interval / turn>0 缺 interval 或带 arrival / 时刻非整微秒 / (session_id, turn_index) 重复——各自 ValueError。

### 1.10 C++ 执行层的请求派发分支(`astra-sim/workload/execution_driven/`)

| # | 分支 | 触发条件 | 行为 | 位置 |
|---|------|---------|------|------|
| X1 | 入口闭合拒绝 | `enqueue_command` 时 `closed_` | 返回 false(有界队列背压,producer 重试) | RequestIngress.cc:35-46 |
| X2 | 容量溢出 | 队列满 | `overflow_count_++` 并拒绝 | :40-47 |
| X3 | 终态命令线性化 | Submit 之外的命令 | `closed_=true`,后续入队全拒 | :48-56 |
| X4 | Submit 身份门 | 空 request_id / 进程级重复 id | std::abort(fail-closed,防下游 metric 身份映射 last-wins) | :91-110 |
| X5 | **到达钳位** | `arrival_world_ns <= current` | alarm 钳到 current+1;钳位按来源分桶计数(StaticCsv 的 t=0 边界例豁免 / late_static_submit / late_external_stream / late_future_alarm_rounding 防御桶) | :114-149 |
| X6 | StaticCsv 到达审计 | source==StaticCsv | 记录 (queue_index → arrival, alarm) 供 run-end 审计 | :150-157 |
| X7 | **命令三分支** | CloseInput / EndOfFile / Error | 前两者分别以 ExplicitClose/EndOfFile 关输入;Error → std::abort(三态区分,禁静默降级) | :177-201 |
| X8 | **future alarm 调度** | Python future_alarms 到达 | 校验(空 id/重复/batch 内重复/已排程/queue_index 未注册或不符)任一违例抛 runtime_error;一次性 queue_index 消费;钳位计数 | :288-352 |
| X9 | 到达回调 | alarm fire | 弹出排程集合 → `DecisionEvent(ARRIVAL)` 入 mailbox(同 tick 同身份去重属 mailbox 契约),携带 ingress_seq + 冻结 queue_index | :373-412 |
| X10 | 生命周期迁移 | 计数器事件 | IDLE↔ACTIVE↔DRAINING→FINISHED;重复到达/完成回调下溢即 abort;全排空且输入仍开 → 回 IDLE;FINISHED 评估延迟到 `note_input_drained`(关闭线性化后) | ServiceCoordinator.cc:21-109, 252-281 |
| X11 | GraphBatch 提交 | Python response 到达 | validate+commit 两段(校验失败即 abort,无补偿路径);节点按 `rank` 字段注入对应 NPU 的执行流 | GraphBatchCommitter.hh:113-345 |
| X12 | watch 终态 | 被观测节点 terminal | 推 completed_groups(stage 映射:prefill/decode/request),同 fire 推 DECODE_COMPLETION + REQUEST_COMPLETE 两条 | WatchRegistry.hh:145-263 |

### 1.11 输入清单层的历史链派生分支(`plan_materializer.py`)

历史 KV 的数量口径在清单层即被冻结:

| # | 分支 | 触发条件 | 行为 | 位置 |
|---|------|---------|------|------|
| Q1 | **turn-0 派生** | `turn_index == 0` | `history = 0`;`context = prefill_length`(recompute 单口径:turn-0 队列 prefill 已折入 prefix) | :353-358 |
| Q2 | **turn>0 链式派生** | `turn_index > 0` | `history = 上一 turn 的 final`(按 (session_id, turn−1) 精确父轮查找,递归记忆化 `_final_for`);`context = history + prefill_length` | :359-368, 329-345 |
| Q3 | 父轮缺失 fail-closed | 上一 turn 请求不在队列 | RuntimeError("lost its previous-turn parent")——历史长度宁可抛错不静默取 0 | :360-365, 336-339 |
| Q4 | final 口径 | 无条件 | `final = context + decode_length` | :369 |
| Q5 | request_id 唯一门 | manifest 内重复 id | RuntimeError(下游按 id 键控的账本无法消歧) | :400-405 |
| Q6 | sidecar 透传 | 旁车行/列在场 | request_type/prefix_len(human/tool 时间入 metrics_manifest)逐字段透传;缺席不写、不猜测填充 | :387-394, 471-478 |
| Q7 | 队列 provenance 对账 | `.provenance.json` 在场且 `csv_sha256` 与队列字节不符 | RuntimeError(混代产物 fail-closed;无 provenance 的合成队列容忍) | :256-296 |
| Q8 | metrics 到达派生 | turn-0 vs turn>0 | turn-0 → absolute(session_arrival_time_ns,缺失即 RuntimeError);turn>0 → after_request(parent_queue_index, interval),父轮/间隔缺失即 RuntimeError | :434-451 |
| Q9 | 目录原子发布 | 暂存/回退 | 双 manifest 先落点前缀暂存目录再两步换名;中途崩溃最坏态 = GEN_MATCH 命中零目录;发布后清扫陈旧 cfg8 目录维持"恰一" | :128-188 |

---

## 2. 历史 KV 管理机制全分支枚举

会话 KV 状态机只有两态:`RESIDENT`(驻留于唯一实例,携带逐 TP rank shard 向量)与 `EVICTED`(逻辑上下文保留、物理字节已删、实例指针为空)(session_kv_manager.py:19-23, 828-837)。每个 rank 维持三账户:`model_weight_bytes`(构造期预载)/ `resident_kv_bytes` / `reserved_request_bytes`,`used ≤ capacity` 是硬不变量(:264-270, 2064-2117)。

### 2.1 命中判定(`prepare_history`,session_kv_manager.py:1540-1741)

请求进入 Prefill 准入时,以 `(session_id, prefill_instance_index)` 查历史:

| # | 前置校验分支 | 触发条件 | 行为 | 位置 |
|---|------------|---------|------|------|
| H1 | 新会话携史拒绝 | state None 且 `history_tokens != 0` | ValueError | :1558-1560 |
| H2 | 逻辑长度失配 | 会话逻辑上下文 ≠ 请求的 history_tokens | ValueError | :1561-1565 |
| H3 | 重叠请求拒绝 | state.active | RuntimeError("overlapping request") | :1566-1567 |
| H4 | 容量预检 | 无条件 | `ensure_physical_fit(prefill_instance, required_context_tokens 全量)`(含本会话自保护) | :1575-1585 |

三支主动作:

| # | 动作 | 触发条件 | 行为/数据流向 | 位置 |
|---|------|---------|--------------|------|
| H5 | **NO_HISTORY** | state is None | 创建零字节 RESIDENT 会话(active=True),发 `no_history/window_first_request` 事件;决策 transfer_shards=() | :1607-1639 |
| H6 | **RECOMPUTE** | `state.state == EVICTED` | 在 prefill 目标实例重放 `history_shards`(state→RESIDENT、instance=target、active=True),journal cause=`recompute_history_restore`,发 `recompute/deleted_history_recompute` 事件;`recompute_tokens = history_tokens`(全量重算,无 partial 语义) | :1641-1683 |
| H7 | **NOC_MIGRATE** | `state.state == RESIDENT` | 逐 relative rank 构造 `KVTransferShard(source_rank, target_rank, bytes)`;**目标侧先加、源侧后删**(动态迁移契约:目标 shard 先落位再释放源);state.instance 切到 prefill 实例;journal cause=`history_noc_migrate`(target_add 先于 source_remove);发 `noc_migrate/history_other_instance` 事件 | :1685-1741 |
| H8 | **admission_blocked 支路** | 预检 `fit.admitted=False` | 返回带 `admission_blocked=True` 的 HistoryDecision;action 按同口径标注(NO_HISTORY/RECOMPUTE/NOC_MIGRATE),`recompute_tokens` 仅 EVICTED 时非零;无 transfer,无状态变更 | :1587-1604 |

已删分支登记:同实例 `LOCAL_HIT` 臂与 existing_target 臂已于 2026-09-26 删除——本仓拓扑下 inactive 会话 KV 只驻留 Decode 实例而 `prepare_history` 目标恒为 Prefill 实例,`source == target` 恒假(:1569-1574, 1690-1691 注释)。动作域收窄为三值 {NO_HISTORY, NOC_MIGRATE, RECOMPUTE}(kv_cache_adapter.py:31-34 的登记口径)。

### 2.2 准入与容量预约(`ensure_physical_fit` / `reserve_request_capacity` / `release_request_capacity` / `extend_request_capacity`)

**`ensure_physical_fit`(session_kv_manager.py:1268-1366)——被动逐出的唯一入口**:

| # | 分支 | 触发条件 | 行为 | 位置 |
|---|------|---------|------|------|
| F1 | shard 向量非法 | 长度 ≠ TP 度或含负值 | ValueError | :1280-1283 |
| F2 | **空实例都放不下** | required > capacity − weight(任一 rank) | 不抛异常:deep_gap_events+1,发 `admission_blocked: request_infeasible_even_empty_instance` 压力事件,返回 `CapacityResult(admitted=False, insufficient_ranks)`——请求等待而非 abort(§4.5 契约:任一 NPU 不满足即整体不可行,等待而非重映射) | :1284-1319 |
| F3 | 空闲充足 | `_insufficient` 为空 | 直接 admitted=True,无逐出 | :1321-1328 |
| F4 | **LRU 逐出循环** | 任一 rank remaining < required | 按 `_candidate_sessions` 固定序逐个 `_delete` 受害者,每删一次重算 `_insufficient`,逐到刚好够即停 | :1330-1360 |
| F5 | **候选耗尽深缺口(D4)** | 候选全部删除仍不足 | deep_gap_events+1,发 `admission_blocked` 压力事件,返回 admitted=False(capacity_epoch 重试机制保留,无死锁) | :1331-1348 |
| F6 | I3 不变量断言 | 循环退出仍 insufficient | RuntimeError("physical-fit stage returned success while still insufficient")——多逐或少逐都是缺陷 | :1361-1365 |

**`_candidate_sessions`(:594-610)的谓词五连**:候选必须同时满足 `state == RESIDENT` ∧ `instance_index == 目标实例` ∧ `not active` ∧ `last_completion_ns is not None`(已完成)∧ 不在 protected 集合(正在准入的会话自保护)。排序键 `(last_completion_ns, session_id)`——**最久未完成的完成时刻最先被逐(会话级 LRU),同时刻按 session_id 字典序 tie-break**。

**预约族**:

| # | 操作 | 分支 | 位置 |
|---|------|------|------|
| F7 | `reserve_request_capacity`(1368-1423) | 重复预约 request → RuntimeError(:1382-1383);内部先嵌套 `ensure_physical_fit`(protected 含本会话),失败原样返回;成功则逐 rank 加 `reserved_request_bytes` 并登记 `RequestCapacityReservation`,journal cause=`request_reservation` |
| F8 | `release_request_capacity`(1425-1475) | `anchor_phase` 三边界映射 journal 锚点(prefill_start=准入阻塞回滚 / decode_start=Decode 放行先释放 / completion=终局兜底);账户下溢即 RuntimeError(:1453-1454) |
| F9 | `extend_request_capacity`(1477-1535) | 未知预约 → RuntimeError;向量非法 → ValueError;净额→全量回补,resident→reserved 1:1 换位,任何瞬间不超订 |

### 2.3 驻留与增长(`_grow` / `grow_prefill` / `grow_decode`,session_kv_manager.py:1743-1929)

| # | 分支 | 触发条件 | 行为 | 位置 |
|---|------|---------|------|------|
| W1 | 状态前提拒绝 | 非 RESIDENT / 实例不符 / 非 active | RuntimeError("not active on target instance") | :1755-1756 |
| W2 | **上下文收缩拒绝** | delta 任一分量 < 0 | ValueError("session KV context cannot shrink") | :1758-1760 |
| W3 | 增量预检 | 无条件 | `ensure_physical_fit(delta)`(可能触发逐出);失败返回 admitted=False(上层 P 侧 raise、D 侧 raise) | :1761-1772 |
| W4 | 增长落账 | 预检通过 | `_add_shards(delta)`,shard_bytes/logical_context_tokens/last_request_id 更新,journal 逐段记账 | :1773-1787 |

`grow_prefill`(:1789-1808)增长到 `prefill_context_tokens`(phase="prefill",reason=`prefill_growth_capacity`);`grow_decode`(:1910-1929)增长到 `final_context_tokens`(phase="decode",reason=`decode_growth_capacity`)。

### 2.4 P→D 迁移(`move_prefill_to_decode`,session_kv_manager.py:1810-1908)

| # | 分支 | 触发条件 | 行为 | 位置 |
|---|------|---------|------|------|
| M1 | 前提拒绝 | 非 RESIDENT / 无实例 / 非 active | RuntimeError("Prefill KV must be active and resident before Decode") | :1821-1822 |
| M2 | 终态口径 | `final_context_tokens=None` | 用当前逻辑上下文;显式传值则按 final 切 shard | :1829-1831 |
| M3 | 目标容量预检 | 无条件 | `ensure_physical_fit(decode 目标, required, protected=本会话)` | :1832-1841 |
| M4 | **阻塞决策** | fit 失败 | 返回 `MoveDecision(action=NOC_MIGRATE, transfer=None, admission_blocked=True)`——唯一调用方立即 raise,不构造占位 transfer | :1842-1854 |
| M5 | 迁移落账 | fit 通过 | 逐 shard `KVTransferShard`;**目标先加、源后删**;state.instance 切换;journal cause=`prefill_decode_migrate`(transfer_complete 锚);发 `noc_migrate/prefill_decode_instance_migrate` 事件;返回携带完整 KVTransfer 的 MoveDecision | :1857-1908 |

同实例 LOCAL_HIT 臂同样已删(source==target 恒假,:1824-1828, 1855-1856 注释)。

### 2.5 完成驻留(`mark_complete`,session_kv_manager.py:1931-1967)

单一支路、无分支选择:**completion 恒零逐出(retain_complete 语义)**——active=False、last_completion_ns=completion_ns、发 `retain_complete/request_completed_keep_kv` 事件,字节零变动。SCHED-9(2026-09-28)删除了恒空逐出返回管道。同时 `_forget_pressure_event_keys`(完成请求永不再重试准入,压力去重键退休)。逐出唯一路径 = 准入期 `ensure_physical_fit` 按需逐出(test_wsc_llm_scheduler.py:370-373 的口径)。

调度侧 `_on_decode_complete`(wsc_llm_online_scheduler.py:1042-1103)在此之上:递增 decode 实例 capacity_epoch(完成时点本身是账本变化)、快照 `kv_state_after_completion / kv_instance_after_completion` 进 completion 决策行(`terminal_kv_release_at_completion=False`, `completion_evictions=[]` 冻结键),随后 M4 核销即删全部胖字段。

### 2.6 淘汰/逐出(`_delete`,session_kv_manager.py:1210-1266)

| # | 分支 | 触发条件 | 行为 | 位置 |
|---|------|---------|------|------|
| V1 | 受害者资格 | 非 RESIDENT / 无实例 / active / 无完成时刻 | RuntimeError("only completed inactive resident sessions may be deleted")——**活跃会话与从未完成的会话绝不可逐** | :1219-1225 |
| V2 | 删除落账 | 资格通过 | `_remove_shards` 释放字节;state→EVICTED、instance=None(逻辑上下文 tokens 保留供 RECOMPUTE);journal cause=`evict_delete:<reason>`;发 `evict_delete` 事件(前后 remaining 快照);构造 `EvictionRecord`(victim 会话/实例/完成时刻/shard) | :1226-1266 |

**压力事件有界去重(`_pressure_event`,:653-696)**:同一 (event_type, phase, request, target) 只记首条;重复压力只补一条 `admission_retry/retry_after_previous_capacity_block`;键按 request 建索引,完成时 O(该请求键数) 退休(:698-706)。

逐出的消费端归属约定(SCHED-2 归因):decode 决策行携带 `decode_target_evictions` 全集(#2 预占/#4 move/#5 grow_decode 累积,发射后写端置空核销),prefill 行只携带发射时点子集(另有 admission_evictions/#3);下游工具只消费 decode 行全集,禁止两行求和(wsc_llm_online_scheduler.py:843-859;kv_cache_adapter.py:480-540;hbm_watermark.py:476-493)。

### 2.7 终局回收(`retire_terminal_session`,session_kv_manager.py:1969-2050)

| # | 分支 | 触发条件 | 行为 | 位置 |
|---|------|---------|------|------|
| N1 | 未知会话 | `_sessions` 无此 id | KeyError | :1984-1986 |
| N2 | 资格拒绝 | active 或无完成时刻 | RuntimeError("only inactive completed sessions may be terminally retired") | :1987-1990 |
| N3 | 属主校验 | 显式 request ≠ last_request_id / 预约属他 session | RuntimeError | :1991-2009 |
| N4 | 预约清算 | 该会话名下全部预约 | 嵌套 `release_request_capacity(anchor_phase="completion")`(并入同一 journal 事务) | :2014-2025 |
| N5 | RESIDENT 支路 | state==RESIDENT | 释放驻留 shard,journal cause=`terminal_session_retire`(completion 锚),返回释放的实例索引 | :2028-2039 |
| N6 | EVICTED 支路 | state==EVICTED | 物理字节早已不在,仅删账 | :2040-2041 |
| N7 | 未知状态 | 其他 | RuntimeError | :2040-2041 |
| N8 | 收尾 | 无条件 | 删 metrics 容器、压力键、会话条目,增量不变量校验 | :2043-2050 |

调用时机(`_on_request_complete`,wsc_llm_online_scheduler.py:1136-1148):terminal turn 且全部输出消费完毕后;释放的容量经 `_note_capacity_change` 发布 epoch 唤醒(不读已删会话状态)。`assert_final_state`(:2052-2062)收口:残留预约/压力键/活跃会话任一存在即 RuntimeError。

### 2.8 水位控制

**(a) 事件流水压缩水位(manager 内,B1)**:`compact_events`(session_kv_manager.py:538-549)——已消费游标 ≥ `_EVENTS_COMPACT_THRESHOLD`(8192)且 ≥ 现存一半时整段删除前缀;`event_index` 由单调计数器分配,压缩不改载荷(:28, 464-467)。消费方 `run_variant_policy` 走 `events_since` 增量切片 + 游标同步回退(wsc_llm_online_scheduler.py:513-526)。

**(b) 容量三口径剖面(hbm_watermark.py:compute_capacity_calibers,709-797)**:

| # | 口径 | 语义 | 位置 |
|---|------|------|------|
| B1 | `per_rank_total_hbm` | 正式认证口径:逐 rank physical = weight+resident+reserved ≤ capacity(判决在 journal 重放执行) | :792-797 |
| B2 | `resident_kv_hard_limit` | reservation=0 时任意时刻成立的 resident 硬上限 = min_r ⌊(capacity−weight_r)/kv_r⌋ token × Σkv_r;capacity < 最大权重分片 → ValueError(拒绝产出负上限) | :734-744, 771-782 |
| B3 | `watermark_reserve_target` | `kv_reserve_context_tokens`(缺省 1M)逐 rank 扣减后的水位目标——**非任意时刻上限,仅报告不作判决**(R_kv 保留水位机制未落地) | :745-749, 783-791 |

**(c) 决策日志重放的水位重建(hbm_watermark.py:WatermarkReplay/WatermarkScan)**:按文件顺序单遍重放,记录内固定次序"先逐出→后恢复/迁移→再增长"(WatermarkScan.consume:1669-1695,manager 语义先腾后放);负占用即 fail(:1388-1391);capacity 超限计数属诊断口径(upper_bound_only,不构成物理违规认证,:1396-1403)。恢复动作的分支选择(apply_restore:1441-1538):LOCAL_HIT → 零搬移(restore_none);bytes==0 → 归属切换;source None → 远端回载只增目标;source==target → 同实例半层回载;否则跨实例全量搬移(source 扣、target 加)。**静默逐出对账支路(W 仓)**:kind==RECOMPUTE 或 state_before==EVICTED 且跟踪态仍有驻留 → 说明期间有未落盘逐出,在此扣减并计入 `silent_evictions_reconciled`(真实逐出时刻 ∈ 上次可见事件与本 tick 之间,占用为上界窗口,:1472-1483)。decode 迁移零字节交割且源仍有跟踪字节 → fail(幻影占用,:1555-1559)。

**(d) journal 权威重放与信任层(JournalReplay / replay_journal,1744-2052)**:行级链自洽(before == 前行 after ∧ after == before+delta ∧ 分量非负 ∧ sequence/planner_time_ns 单调 ∧ rank 的 instance/capacity 不得漂移)任一破坏即 fail;逐 rank `physical > capacity_bytes` 计数为**正式判决**(仅 certified 层 exit 3);resident 超硬上限仅报告(:1855-1902, 1887-1902)。tier 判定(:1936-2052):sha256 不符/行数不符/终态矛盾 → 硬失败(exit 2);checksum 完备且四检查(manager_state_match / physical_equals_weight / residual_reserved_zero / residual_resident_zero)全真 → `TRUST_TIER_CERTIFIED`;否则降 `lifecycle_replay_exact`;无 checksum → `resident`(上界)。绘图桶长 `B_eff = max(B_requested, ceil(S·N/(R−N)))`,行预算不足时拒绝稠密输出(resolve_effective_bucket_ns:2055-2069)。

**(e) 深缺口观测**:D4 计数 `deep_gap_events` 由 online_service summary 行 `kv_deep_gap_events` 落盘观测(session_kv_manager.py:479-483;online_service.py:311-314;实证负载预期恒 0)。

### 2.9 KV delta journal(权威 HBM 账本,P1)

| # | 分支 | 触发条件 | 行为 | 位置 |
|---|------|---------|------|------|
| J1 | 旁路态 | recorder 未装或非 journal 模式 | 装饰器透传,零开销零行为差 | session_kv_manager.py:106-110 |
| J2 | **嵌套事务并入** | `_journal_transaction_depth > 0` | 深度 +1,并入外层同一事务,不另起 id(retire 内的 release / 准入前置的 ensure_physical_fit) | :111-116 |
| J3 | **事务提交点** | 最外层公开 mutation 正常返回 | `finish_transaction` 取 (transaction_id, ranks),随即 `_journal_reconcile_transaction` 逐 rank 对账:manager 实际 (weight,resident,reserved) vs journal 运行计数(before+Σdelta),不符即 RuntimeError | :117-127, 1057-1090 |
| J4 | **事务中止** | 方法体抛异常 | 深度清零、`abort_transaction`、不执行对账(避免二次异常掩盖原始错误);已写出部分行保留,本轮 run fail-closed 终止 | :119-124 |
| J5 | 时序护栏 | `planner_time_ns` 回退 | recorder 立即 raise(不钳位) | metrics_integration.py:126-127, 254-263 |
| J6 | 嵌套 begin / 无事务 finish | 协议违例 | RuntimeError | metrics_integration.py:206-236 |
| J7 | **run 末 checksum 门** | `verify_journal_checksum`(verify_run_end 链尾) | 三重断言:重放终态 == manager 快照(逐 rank 三账户 + 差额表);写出条数 == 重放行数;**守恒断言——终态 resident==0 ∧ reserved==0 ∧ physical==weight**(权重驻留、无会话/预约残留);产物 `kv_delta_journal_checksum.json` | session_kv_manager.py:1092-1207, 1719-1725 |
| J8 | journal 行锚点 | 每次 record | `anchor_kind` 经 `_metrics_anchor_for_phase` 映射:history/prefill/admission/prefill_admission→prefill_start,decode→decode_start,completion→completion,其余→stage_boundary;权重预载首 record 取 tick_zero(transaction_id=0) | :66-84, 499-518 |

### 2.10 观测适配层的命中态映射(kv_cache_adapter.py)

| # | 分支 | 触发条件 | 输出 | 位置 |
|---|------|---------|------|------|
| O1 | NO_HISTORY | turn-0 | `no_history`(不计入 miss) | :137-150 |
| O2 | LOCAL_HIT | 旧产物容忍 | `full` | :142-147 |
| O3 | NOC_MIGRATE | 跨实例迁移 | `full`(命中,迁移成本入 hopbytes) | :142-147 |
| O4 | RECOMPUTE | 已删历史 | `miss`(全量重算,无 partial 语义) | :142-147 |
| O5 | 未知动作 | 值域漂移 | `not_supported`(禁止猜测) | :148-149 |
| O6 | 缺 prefill 决策 | manifest 请求无决策行 | `not_supported` + 证据列 | :701-709 |
| O7 | canonical 事件提取 | prefill 行 history_transfer_bytes>0 / decode 行 transfer / 各行逐出列表 | 只映native字段;`--reconcile` 以 native 为准逐 family 对账(bytes/count),不一致 exit≠0 | :480-540, 776-964 |
| O8 | Σshards 复检 | shard_sum_check 变体 | Σshards ≠ total_bytes 计入 violations,run 末 fail | :650-674, 961-963 |

### 2.11 不变量校验通路(session_kv_manager.py)

增量账本与完整审计构成双口径校验网,每条失败路径都 fail-closed:

| # | 分支 | 触发条件 | 行为 | 位置 |
|---|------|---------|------|------|
| I1 | 会话态校验 | state ∉ {RESIDENT, EVICTED} | RuntimeError("unknown KV state") | :828-837, 2079-2093 |
| I2 | RESIDENT 结构 | 无实例 / shard 向量长度 ≠ TP 度 | RuntimeError | :831-835, 2082-2086 |
| I3' | EVICTED 残留 | 逐出会话仍持有实例指针 | RuntimeError("evicted session retained an instance") | :836-837, 2092-2093 |
| I4 | 账户负值 / 超容 | `resident/reserved < 0` 或 `used > capacity` | RuntimeError("negative HBM accounting"/"HBM capacity exceeded") | :862-870, 2065-2072 |
| I5 | 会话/rank 对账 | 实例内 Σsession.shard ≠ 逐 rank resident 账 | RuntimeError("session/rank KV accounting mismatch") | :880-896, 2101-2109 |
| I6 | 预约/rank 对账 | Σreservation ≠ 逐 rank reserved 账 | RuntimeError("reservation/rank accounting mismatch") | :897-908, 2110-2117 |
| I7 | 释放超持有 | `_remove_shards` 释放量 > rank 现存 | RuntimeError("attempted to release more KV than the rank owns") | :721-723 |
| I8 | 严格模式放大 | `SH_STRICT_KV_INVARIANTS` 开 | 每次 mutation 后追加完整审计(缺省仅增量局部) | :910-911, 484-488 |
| I9 | 运行终态断言 | `assert_final_state` | 残留预约 / 压力去重键 / 活跃会话任一存在即 RuntimeError | :2052-2062 |

---

## 3. 两条机制在体系结构中的交汇点

映射决策与 KV 状态在三个控制点上互为因果:

1. **映射受 KV 状态约束**:Prefill 准入通路的每一步(A4→A12)都先问 KV 账本——终态预约先看 Decode 实例剩余空间,净额修正分支只在旧 KV 确实 RESIDENT 于静态 Decode 目标时打开,`prepare_history` 的三支动作由会话状态机的 (存在性, RESIDENT/EVICTED) 二维谓词唯一确定,阻塞时预约回滚、epoch 挂起等待,请求滞留在 qp 而非改绑其他实例(§4.5:等待而非重映射)。映射组件因此是被 KV 容量调制的节流阀。
2. **KV 状态被映射决策改写**:每一次发射决策都驱动状态机迁移——到达映射决定 KV 将落位哪个 Prefill 实例(H5/H6/H7 的 target),静态路由决定 KV 随后迁往哪个 Decode 实例(M5),列车组成决定 joiner 的 transfer-3000 迁移时点(G8),terminal turn 的判定触发终局回收(N5/N6)。状态迁移的全部字节级痕迹经 delta journal 逐 rank 落账,run 末以守恒断言闭环。
3. **容量 epoch 是双向耦合的振荡器**:KV 侧任何字节事件(逐出、迁移、释放、完成)都提升相关实例的 epoch,从而解除映射侧阻塞请求的重试门(A3);映射侧的每次发射又反过来制造下一批 KV 事件。被动 LRU 逐出只在这条耦合回路里被触发(准入压力驱动),完成路径永不主动逐出——整个系统的显存治理因此是"映射压力 → 被动回收 → epoch 唤醒 → 重试"的单向环流,不存在主动水位驱逐机(R_kv 仅存观测口径)。
