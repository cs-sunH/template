# request 映射策略与历史 KV 管理机制

## 架构总览

系统是一条自上而下的五层通路，请求作为控制流载体在其中逐层折返：

- **输入物化层**（`plan_materializer.py` + `generate_trace.load_request_queue` + `generate_face_trace.load_face_trace_config`）：请求队列 CSV 经 `trace_config.csv` 装载为 `FaceTraceConfig`，物化出 manifest 的九个派生字段（`history_tokens_before` / `prefill_context_tokens` / `final_context_tokens` 等）。9 个 `inference_group`（各 6 rank）铺满 9×6 mesh 的全部 54 个 NPU，构成"请求 → 实例 → rank 组"的静态映射骨架（`build_instances`，face_scheduler.py:138-230）。
- **调度层**（`online/face_online_scheduler.py`，纯模型在 `face_scheduler.py`）：请求自 ingress ARRIVAL 事件进入入口通路，在到达边界映射到 prefill 实例（`select_prefill_instance` + 容量预检过滤），在 prefill drain 边界映射到 decode 实例（`select_decode_instance` + 容量预检过滤）；两个准入门由容量纪元（capacity epoch）与 dirty 标志控制。计时由"实例迭代列车"（train）承载——decode 互拼、decode 与 prefill chunk 混拼、批成员只在列车边界变化。
- **KV 管理层**（`session_kv_manager.py` 的 `SessionKVCacheManager`）：会话级两态状态机（`LOCAL_HBM` 全层域驻留单实例 / `REMOTE_MEMORY` 全量远端池），LRU 受害者池、整体逐出、全量恢复、逐 rank HBM 记账全部在此层闭合。
- **构图/发射层**（`online/graph_batch_builder.py` 复用 `generate_face_trace.py` 发射原语）：把 KV 决策（`KVTransfer` 对象）翻译成物理节点链——1000 类 history 迁移、3000 类 prefill→decode 迁移（`_paired_transfer`）、remote_store / remote_load 远端池链（`_emit_kv_transfer`）、控制触发 1B p2p（`_emit_control_trigger`）。
- **模拟器执行层**（C++ `astra-sim/workload/execution_driven/`）：`RequestIngress` 喂入到达 alarm，`ServiceCoordinator` + `DecisionBridge`（文件 FIFO 桥）把 StateDelta 交付给 Python 调度器，`GraphBatchCommitter`/`NodeStore` 原子提交返回的 GraphBatch，`WatchRegistry` 以 drain/exit/哨兵标记节点驱动下一决策边界；assignments 仅做结构校验（两索引非负，ParsedGraphBatch.cc:546-576），映射权威在 Python 侧。
- **分析适配层**（`slo_tools/kv_cache_adapter.py`、`hbm_watermark.py`）：离线重放决策日志，产出 canonical cache 事件、`kv_hit_state` 命中态与 HBM 水位时序。

控制流走向：C++ 到达/完成事件 → Python `on_decision_batch` → `run_variant_policy`（完成批 → 到达批 → 准入/发射 pass）→ GraphBatch 回注 C++ → watch fire 再触发。数据流走向：请求事实（policy-independent）下行进入调度器运行账本；KV 状态变更（逐出/迁移/恢复/增长）经 `kv_actions` 事件流与 `KVTransfer` 双通道下行进入构图层与分析层。映射决策与 KV 状态在"准入门"处交汇：映射选择读取会话驻留位置与实例水位作为谓词，KV 状态机又以目标实例的准入需求作为逐出触发源。

---

## 一、request 映射策略全分支枚举

### 1. 静态骨架映射（配置 → 实例 → rank）

| 分支 | 触发条件 | 行为 / 数据流向 | 代码位置 |
|---|---|---|---|
| rank 坐标派生 | 任一 rank | row-major：`rank = 行 × mesh_cols + 列`（`divmod(rank, mesh_cols)`） | face_scheduler.py:132-135 |
| 实例矩形校验 | 实例 rank 集合的坐标包络非"满填充轴对齐矩形" | 构造期 raise（实例物理形状约束） | face_scheduler.py:180-188 |
| 全覆盖校验 | 实例 rank 并集 ≠ 全部 NPU 恰一次 | raise（每 rank 恰属一实例） | face_scheduler.py:202-203 |
| 等尺寸校验 | 实例 size 不等且 `require_equal_size=True` | raise（直接 KV shard 配对要求等 TP 度） | face_scheduler.py:204-207 |
| 邻接连通校验 | 多实例邻接图有孤岛 | raise（实例平铺须连通） | face_scheduler.py:209-224 |
| 组名唯一校验 | `inference_group` 重名 | 物化期 raise（实例身份按名判定，重名使跨实例迁移在发射层被误判同实例零节点） | generate_face_trace.py:297-308 |
| 邻接图构建 | 实例间物理相邻（rank 网格 (0,1)/(1,0) 步） | 无向邻接表 + 常值边权 1（远程 KV 放置路径已删，边权恒 1） | face_scheduler.py:209-230, 396-401 |

### 2. 到达边界：prefill 实例映射（`_on_arrival`）

请求到达后经两段映射：先纯排序选择，后只读容量过滤。

**2.1 排序选择 `select_prefill_instance`（face_scheduler.py:331-343）**——键为 `ordering_key = (remaining_chunks, last_arrival_ns 或 -1, instance_index)`，取最小：

| 分支 | 触发条件 | 行为 | 代码位置 |
|---|---|---|---|
| 最浅队列分支 | 某实例 `prefill_remaining_chunks`（qp 成员 `remaining_chunks` 之和）最小 | 请求入该实例 qp（FCFS append） | face_online_scheduler.py:2016-2039, 1230-1252 |
| 到达时间平局分支 | 剩余 chunk 数并列 | 取 `last_arrival_ns` 更早者；`None`（从未接收到达，键值 -1）优先于一切有值实例 | face_scheduler.py:331-337 |
| 索引平局分支 | 前两键均并列 | 取最小 `instance_index`（确定性终裁） | face_scheduler.py:336-343 |

**2.2 容量可行性过滤 `_filter_prefill_selection`（face_online_scheduler.py:2156-2229）**——对每个实例快照做只读预检：

| 分支 | 触发条件 | 行为 / 数据流向 | 代码位置 |
|---|---|---|---|
| 增量需求分支 | 会话 `location == LOCAL_HBM` 且已驻留在候选实例上 | 需求分片 = `prefill_context` 全量分片 − 已驻留 `local_shard_bytes`（M12 修正：与 `prepare_history` 的 LOCAL_HIT 增量口径镜像，避免把原位零传输准入误判为不可行而改派他实例） | face_online_scheduler.py:2190-2198 |
| 全量需求分支 | 无会话 / 远端 / 驻留在其他实例 | 需求 = `kv_cache_shard_bytes_for_tokens(prefill_context_tokens)` 全量 | face_online_scheduler.py:2182-2189 |
| 原选可行分支 | 原 selected 在幸存集 | 原样返回，决策日志零扰动 | face_online_scheduler.py:2202-2203 |
| 改选分支 | 原选不可行且幸存集非空 | 幸存集内按同一 `ordering_key` 取 min 改选，写 `prefill_admission_probe` 决策行（outcome=reselected_infeasible_first） | face_online_scheduler.py:2212-2222 |
| 全不可行分支 | 幸存集为空 | 保持原选，落入 P0-1 阻塞-重试通路（探测不 remap、零容量副作用；预检口径保守：仅 remaining，不含冷会话逐出信用） | face_online_scheduler.py:2223-2229, 2043-2062 |

选择落定后：`runtime.prefill_instance_index` 固化，请求 append 入该实例 qp，`last_arrival_ns` 更新，实例加入 ready frontier，感知账本登记 `prefill_qp` 排队层（face_online_scheduler.py:1246-1257）。

### 3. Drain 边界：decode 实例映射（`_on_prefill_drain`）

prefill 主体（列车 drain 标记）物理完成时，请求从 qp 出队并做一次性的 decode 实例选择：

**3.1 候选集圈定（`schedulable_instances`）**

| 分支 | 触发条件 | 行为 | 代码位置 |
|---|---|---|---|
| 可调度域分支 | 加权图最短路距离 ≤ `schedulable_distance_limit`（= `d2d_bandwidth_gbps / local_hbm_bandwidth_gbps`，单位无关比值） | 进入候选集 | face_scheduler.py:67-68, 388-394, 444-447 |
| 同点分支 | source == target | 距离 0，恒在域内 | face_scheduler.py:368-369 |
| 候选集空 | 全部实例超出可调度距离 | raise（decode 可调度实例图为空属结构错误） | face_scheduler.py:448-449 |

**3.2 候选代价计算（Roofline per-die 增量）**

每个候选构造 current / updated 两份 `FaceRooflineEstimate`：`p_chunk` 取 `fixed_p_chunk`（该实例 qp 非空）或 0；`updated_d_token = max(d_token, new_request_token_length)`；代价 = `delta_time_ns / instance.size`（`per_die_delta_ns`）（face_scheduler.py:452-497）。迭代时间估计内部的分支：`p_chunk==0 且 d_batch==0` 返回 0（face_scheduler.py:273-274）；swiglu/gelu 的 MLP 算子倍率分支（face_scheduler.py:285）；prefill / decode 注意力段各按是否为空跳过（face_scheduler.py:300-314）。

**3.3 选择与平局裁决（face_scheduler.py:498-508）**

| 分支 | 触发条件 | 行为 | 代码位置 |
|---|---|---|---|
| 最小代价分支 | `per_die_delta_ns` 唯一最小 | 选该实例 | face_scheduler.py:498-507 |
| 真实平局轮转分支 | ≥2 个候选 `per_die_delta_ns` 精确相等且传入 `DecodeTieCounter` | tied 集按 `instance_index` 升序，取 `counter % len(tied)` 并前进计数器（同决策序列确定可重放） | face_scheduler.py:414-421, 503-505 |
| 无计数器平局分支 | 平局但未传计数器 | 取最小 `instance_index`（旧行为） | face_scheduler.py:506-507 |

**3.4 容量可行性过滤 `_filter_decode_selection`（face_online_scheduler.py:2064-2154）**

| 分支 | 触发条件 | 行为 | 代码位置 |
|---|---|---|---|
| 增量需求分支 | 会话 `LOCAL_HBM` 且驻留在候选 c 上 | 需求 = `final_context` 全量分片 − `local_shard_bytes`（镜像 `move_prefill_to_decode` 的 `existing_target`） | face_online_scheduler.py:2099-2109 |
| 全量需求分支 | 其余情形 | 需求 = final 全量分片（源副本仍占源实例的双驻留口径） | face_online_scheduler.py:2094-2098 |
| 原选可行分支 | 原 selected 可行 | 原样返回（tie_counter 零扰动） | face_online_scheduler.py:2112-2113 |
| 改选分支 | 原选不可行且可行集非空 | 可行集内按 `per_die_delta_ns` 最小重选，精确并列按 `instance_index` 升序用 `_decode_tie_counter` 轮转（M14 修正），写 `decode_admission_probe` 行 | face_online_scheduler.py:2122-2147 |
| 全不可行分支 | 可行集为空 | 保持原选，进入 P0-1 阻塞-重试（"绝不 remap"不变量：选择只发生一次，落定后容量可 defer 但绝不改派） | face_online_scheduler.py:2148-2154 |

选择落定后：`decode_instance_index` 固化，`waiting_decode_admissions[selected]` 登记 + `decode_admission_dirty` 置位 + 立即 `try_admit_waiting_decodes`（face_online_scheduler.py:1316-1330）。

### 4. prefill 准入门（`_try_admit_prefill`，服务时点逐请求尝试）

| 分支 | 触发条件 | 行为 / 数据流向 | 代码位置 |
|---|---|---|---|
| 已准入幂等分支 | `admitted_prefill == True` | 直接返回 True | face_online_scheduler.py:1544-1545 |
| 映射缺失异常分支 | `prefill_instance_index is None` | raise（准入丢失固定映射） | face_online_scheduler.py:1546-1547 |
| 纪元门关闭分支 | `prefill_attempt_epoch == capacity_epoch[实例]` 且实例不在 `prefill_admission_dirty` | 返回 False（本决策边界不重试，防重试风暴） | face_online_scheduler.py:1552-1555 |
| 纪元门放行分支 | 纪元推进或 dirty 置位 | 记录尝试纪元、清 dirty，继续准入链 | face_online_scheduler.py:1556-1560 |
| B4 store→restore 前递推迟分支 | 会话快照 `location == REMOTE_MEMORY` 且 `graph.has_pending_store_decision(session)`（存在已决策未发射的 remote_store） | 置 dirty、返回 False——纯推迟零 mutation，防全量回迁的池读先于池写；store 随 joiner 列车头发射后计数归零，下一边界重试 | face_online_scheduler.py:1570-1575; graph_batch_builder.py:1042-1045 |
| 历史准备阻塞分支 | `prepare_history` 返回 `admission_blocked` | 有逐出则 `_note_capacity_change`（重开纪元门）；阻塞 attempt 的逐出是真实 mutation（不回滚）——`emit_eviction_actions` 随当前批发射 remote_store 物理链，已发射者从 `history_evictions_pending` 剥离（B4 防双发）；置 dirty，返回 False | face_online_scheduler.py:1606-1635 |
| 历史准备成功分支 | `prepare_history` 准入 | 记录 `history_action` / `history_source_instance_index` / `history_transfers` / `history_transfer_bytes`（只计非 local_hit 的物理搬运字节） | face_online_scheduler.py:1636-1647 |
| 增长失败异常分支 | `grow_prefill` 返回未准入 | raise（准入已成功的增长不可能失败——预留语义失守） | face_online_scheduler.py:1654-1656 |
| 准入完成分支 | prepare + grow 均成功 | `remaining_chunks = ceil(prefill_length/p_chunk)` 重算（不含 history 重算段，B2 已删 RECOMPUTE）；`admitted_prefill = True`；容量纪元推进 | face_online_scheduler.py:1666-1684 |

### 5. decode 准入门（`_try_admit_waiting_decodes`）

| 分支 | 触发条件 | 行为 / 数据流向 | 代码位置 |
|---|---|---|---|
| 目标筛选分支 | 实例等待队列非空且（dirty 或 `decode_admission_epoch != capacity_epoch`） | 进入本轮扫描；扫描头同步纪元 + 清 dirty | face_online_scheduler.py:1692-1706 |
| 成员跳过分支 | runtime 已不处于等待态 | continue | face_online_scheduler.py:1711-1712 |
| 队列错位异常分支 | `decode_instance_index != target_instance` | raise（准入目标队列被破坏） | face_online_scheduler.py:1713-1715 |
| 迁移阻塞重排队分支（face 特有语义） | `move_prefill_to_decode` 返回 `admission_blocked` | 有逐出则纪元推进；阻塞逐出即时发射 + pending 剥离；置 dirty；`queue.append(runtime)` 重排尾并 continue——阻塞不弃单 | face_online_scheduler.py:1736-1772 |
| 迁移成功分支 | move 准入 | `prefill_decode_transfer = move.transfer`；`grow_decode` 必须成功否则 raise；逐出账本（`decode_evictions` / `decode_evictions_pending`）累积并 `sync_pending_history_after_evictions` 镜像 pending location | face_online_scheduler.py:1773-1795 |
| KV 就绪栅栏分支 | 迁移 + 增长完成 | runtime 进入 `pending_decode_ready`，待加入 decode 实例的下一列车（3000 迁移随列车头发射，物理先于列车体）；实例空闲则入 ready frontier；assignment 条目一次性携带双索引（prefill 边界不产 assignment）；写 decode 决策行（候选代价全记录 + hopbytes） | face_online_scheduler.py:1796-1873, 1886-1954 |

### 6. 列车规划与发射（实例内"批次"映射）

| 分支 | 触发条件 | 行为 | 代码位置 |
|---|---|---|---|
| 无工作分支 | qp 头为空且 active_decode 为空 | `_plan_train` 返回 None，实例移出 frontier | face_online_scheduler.py:809-811, 953-956 |
| 纯 decode 列车分支 | qp 头为空但有 decode 成员 | 自然迭代数 = max 成员剩余 token | face_online_scheduler.py:812-813 |
| 混拼列车分支 | qp 头在场 | 自然迭代数 = 头部剩余 chunk 数；每迭代至多 1 个 prefill chunk，chunk 间不互拼 | face_online_scheduler.py:814-819, 873-896 |
| T_max 截断分支 | `iterations > SH_TRAIN_MAX_ITER`（缺省 8；0 = 不设限；负值构造期 fail-closed） | 截断并置 `capped`；无自然 drain/exit 标记时列车自 id 作哨兵完成信号 | face_online_scheduler.py:820-824, 852-854, 526-536 |
| 准入失败回退分支（face 回退） | qp 头 `_try_admit_prefill` 失败 | `qp_head = None`，本列车退化为纯 decode | face_online_scheduler.py:937-941 |
| 头部跳过分支 | qp 头 `remaining_chunks == 0`（drain 决策在途） | 不提供 chunk 工作，选下一个有剩余 chunk 的成员 | face_online_scheduler.py:929-936 |
| 成员退出先验分支 | `participation >= remaining` | exit 标记挂列车内（退出不是列车边界） | face_online_scheduler.py:835-842 |
| 首 token 拆分分支 | `SH_FIRST_TOKEN_SPLIT=1` 且有 debut 成员且 `iterations >= 2` | 两段式发射：首步批（首 chunk + 各成员首 span + joiner 迁移 + 唤醒标记）与余量批；缺省关（决策等价门裁决） | face_online_scheduler.py:1083-1120; graph_batch_builder.py:87-108 |
| 首 token 不拆车分支 | 拆分开但 `iterations < 2` | 仅做标记增强（多 token debut 挂 `_first_token_` 标记；decode_length=1 的 debut 其 exit 标记改名同节点双锚点） | face_online_scheduler.py:1116-1120 |
| busy 门分支 | `in_flight_train` 非空 | 跳过发射（一实例至多一列在飞列车） | face_online_scheduler.py:926-927 |

### 7. 完成边界与生命周期分支

| 分支 | 触发条件 | 行为 | 代码位置 |
|---|---|---|---|
| 完成事件分类分支 | completed_groups 的 stage = prefill / decode / REQUEST / batch_train_ 前缀 | 分别路由到 drained / completed_now / 同 tick 配对处理 / 哨兵核销；未知 stage raise | face_online_scheduler.py:603-620 |
| 首步唤醒剥离分支 | train_id 在 `_pending_first_steps` | 无操作剥离（不决策不记账） | face_online_scheduler.py:627-630, 682-692 |
| 陈旧完成错配分支 | 完成信号不匹配任何在飞/已核销列车 | raise（fail-closed） | face_online_scheduler.py:766-772 |
| 迟到残余信号分支 | 已核销列车的 pending 信号 | 幂等清 pending，不改就绪态 | face_online_scheduler.py:758-765, 781-787 |
| 下一 turn 排程分支 | 同 session 存在 turn+1 请求 | future alarm = `max(父请求 code-4 完成 tick + interval, 交付 tick)`（M16 钳位：deferral 下防注册到达早于交付 tick 被 C++ 提交门拒绝） | face_online_scheduler.py:1441-1471 |
| terminal 回收分支 | 无后续 turn | `retire_completion_gate` + `kv_manager.retire_terminal_session`（终局核销，见 KV 侧）+ 容量纪元推进 | face_online_scheduler.py:1472-1484 |
| 运行收尾断言 | run 结束 | arrival heap / ready frontier / 待发射余量 / store 登记 / 等待队列 / KV 终态全部须空，任一非空 raise | face_online_scheduler.py:2301-2376 |

### 8. 物化层的请求派生映射（plan_materializer.py）与队列装载分支

| 分支 | 触发条件 | 行为 | 代码位置 |
|---|---|---|---|
| 队列 turn-0 时序分支 | `turn_index == 0` | 必须携带 `session_arrival_time_ns` 且 interval 列须为空；否则 raise | generate_trace.py:233-247 |
| 队列 turn>0 时序分支 | `turn_index > 0` | 必须携带 `inter_request_interval_ns` 且 arrival 列须为空；否则 raise；时序值须整千 ns | generate_trace.py:248-271 |
| 队列身份唯一分支 | request_id 重复 / (session_id, turn_index) 重复 | 装载期 raise（每个键控消费者静默碰撞的入口防御） | generate_trace.py:216-228 |
| 会话子集截断分支 | `request_queue_session_limit == 0 / >0 / > 可用会话数` | 0 = 全部会话；N = 源行序前 N 个会话的全部行；超界 raise（物化入口对 >0 另行 fail-closed，见下） | generate_face_trace.py:214-241 |
| turn-0 派生分支 | `turn_index == 0` | `history=0`；`context = prefill_length`（recompute 单口径：队列 prefill 已折入） | plan_materializer.py:275-279 |
| turn>0 派生分支 | `turn_index > 0` | 按 `(session_id, turn-1)` 精确键取父 final 作 history（P4 修正：非"文件序上一行"）；`context = history + prefill_length`；父缺失 raise | plan_materializer.py:280-288 |
| final 派生 | 恒 | `final = context + decode_length` | plan_materializer.py:289 |
| 指标到达派生 | turn-0 / turn>0 | absolute（`session_arrival_time_ns`）vs after_request（父 queue_index + interval）；父缺失 raise | plan_materializer.py:332-349 |
| session 截断 fail-closed | `request_queue_session_limit > 0` | 物化拒绝（Python 侧截断与 C++ 全量枚举系统性分歧） | plan_materializer.py:392-401 |
| mesh 钉死分支 | NPU 总数 ≠ 54 | 物化拒绝（plan 目录名钉死 54npus） | plan_materializer.py:409-420 |
| 策略透传断言 | `kv_cache_policy ∉ {session_lru_tiered}` | raise（值域漂移防御） | plan_materializer.py:78-90 |

### 9. 执行层的结构校验分支（C++）

| 分支 | 触发条件 | 行为 | 代码位置 |
|---|---|---|---|
| assignment 结构分支 | 条目缺 `request_id` / `prefill_instance_index` / `decode_instance_index` 或任一索引为负 | 提交前 fail（映射权威在 Python，C++ 只验形状） | ParsedGraphBatch.cc:546-576; GraphBatchCommitter.cc:1525-1531 |
| 节点 rank 域分支 | 节点/边 rank 越界 | fail | ParsedGraphBatch.cc:131-135 |
| 决策交付幂等分支 | `delivery_sequence == last_applied` | 深比对后重放上批 digest（零状态变更）；同 seq 异内容 / 跳号 raise | online_scheduler_base.py:505-534 |
| commit ack 失败分支 | ack `success == false` | raise（provisional KV 账本不得以失败 ack 终化） | online_scheduler_base.py:1032-1035 |
| 重复完成 / 重复到达分支 | 同 request 同 stage 完成两次 / 二次到达 | raise（fail-closed） | online_scheduler_base.py:883-937 |

### 10. 配置开关与值域分支

| 开关 | 值域与缺省 | 语义 | 代码位置 |
|---|---|---|---|
| `kv_cache_policy` | 仅 `session_lru_tiered`，其余 fail-closed | 本仓唯一 KV 行为，无档位分支 | generate_face_trace.py:168-179; face_online_scheduler.py:436-440; online_service.py:192-216 |
| `trace_granularity` | 在线发射仅 `request_aggregated` | 非该值发射入口 raise | graph_batch_builder.py:569-572 等; generate_face_trace.py:180-186 |
| `prefill_chunk_size` | 正整数（缺省 512）；float / ≤0 构造期 fail-closed | chunk 切分粒度 | face_online_scheduler.py:455-468 |
| `request_queue_session_limit` | loader 支持 N，物化入口对 >0 fail-closed | 会话子集旋钮（事实不可用） | generate_face_trace.py:214-241; plan_materializer.py:392-401 |
| `SH_TRAIN_MAX_ITER` | 缺省 8；0 不设限；负 raise | 列车长度上限 | face_online_scheduler.py:526-536 |
| `SH_FIRST_TOKEN_SPLIT` | 缺省 "0" 关 | 首 token 两段式拆分 | graph_batch_builder.py:87-108 |
| `SH_QUEUE_AGG_VERIFY` | "1" 时启用 | 队列聚合影子校验 | face_online_scheduler.py:552-553, 2022-2031 |
| `SH_STRICT_KV_INVARIANTS` | 缺省关 | 每次变更后跑全量审计 | session_kv_manager.py:66-74, 1171-1172 |
| `SH_PROPAGATING_TAIL_LIMIT` | 缺省 8 | 在途尾部 fail-closed 上限 | online_scheduler_base.py:53-54, 349-368 |
| `SH_GRAPH_DIGESTS` / `SH_PROFILE_JSONL` / `BRIDGE_TIMEOUT_MS` | 缺省 0/0/120000ms | 审计产物与桥超时旋钮 | online_service.py:59-65, 172-174; decision_bridge.py:49-66 |

---

## 二、历史 KV 管理机制全分支枚举（按生命周期阶段）

会话是 KV 的唯一归属单位：状态域恒两值——`LOCAL_HBM`（`resident_prefix_layers == model.layers`，全层域驻留单实例）与 `REMOTE_MEMORY`（`resident_prefix_layers == 0`，无本地实例）；legacy 的 `partial_hbm_remote` 仅为旧日志解析常量，运行态白名单对其实施 fail-closed（session_kv_manager.py:46-58, 1078-1082, 2255-2259）。

### 阶段 1：命中判定（`prepare_history` 四分支派发，session_kv_manager.py:1669-1928）

| 分支 | 触发条件 | 行为 / 数据流向 | 代码位置 |
|---|---|---|---|
| 输入守卫分支 | 新会话但 `history_tokens != 0`；会话逻辑上下文 ≠ 请求 history；会话仍 active（重叠请求） | raise / RuntimeError | session_kv_manager.py:1690-1705 |
| **NO_HISTORY 分支** | 会话不存在且 history=0 | `ensure_physical_fit(全量 desired)` 作容量门；阻塞 → `HistoryDecision(NO_HISTORY, admission_blocked=True)`；成功 → 建 `SessionKVState(LOCAL_HBM, instance=target, 0 字节, active=True)`，落 `no_history` 事件 | session_kv_manager.py:1722-1764 |
| **LOCAL_HIT 分支** | `location == LOCAL_HBM` 且 `instance_index == target` | 增量需求 = `desired − 现有分片`（同实例零字节增量）；阻塞 → blocked LOCAL_HIT 决策；成功 → 零开销复用：`_local_hit_transfer`（无网络 shard，`local_hit` 类传输禁携 shard），落 `local_hit` 事件，水位不变 | session_kv_manager.py:1766-1829, 1186-1211 |
| **NOC_MIGRATE 分支** | `location == LOCAL_HBM` 且实例不同 | 增量需求 = 全量 desired（源副本仍占源）；阻塞 → blocked NOC_MIGRATE；成功 → 整份 NoC 迁移：`_noc_transfer`（逐 TP shard XY 路径，要求会话全量驻留），先 `_add_shards(target)` 后 `_remove_shards(source)`（动态传输契约：目标准入先于源释放），`instance_index` 原子切换 | session_kv_manager.py:1831-1870, 1213-1259 |
| **REMOTE_RESTORE 分支**（唯一恢复路径） | `location == REMOTE_MEMORY` | 容量收敛先于恢复：`ensure_physical_fit(全量 desired)`——恢复的容量检查本身可触发对其他会话的整体逐出；阻塞 → blocked 决策；成功 → 单笔 `remote_load` 全量回迁 `[0, model_layers)`，加回全部字节，`location → LOCAL_HBM`、`resident_prefix_layers → layers`、active，落 `remote_load` 事件；恢复完成前会话保持 REMOTE 不可消费 | session_kv_manager.py:1872-1928, 1261-1313 |
| 未知位置 fail-closed 分支 | location ∉ {local_hbm, remote_memory}（含手工置入的 PARTIAL） | RuntimeError | session_kv_manager.py:1872-1873 |
| 收缩守卫分支 | 增量需求任一分片为负 | RuntimeError（会话 KV 不允许在历史准备中收缩） | session_kv_manager.py:1779-1780 |

### 阶段 2：准入（容量收敛 `ensure_physical_fit`，session_kv_manager.py:1461-1602）

| 分支 | 触发条件 | 行为 / 数据流向 | 代码位置 |
|---|---|---|---|
| 参数非法分支 | 分片向量长度 ≠ TP 度或含负值 | ValueError | session_kv_manager.py:1484-1485 |
| **结构不可行分支（fail-closed）** | 任一 rank 上 `required > capacity − model_weight`（空实例都装不下） | 先落 deep_gap 台账（reason=request_exceeds_empty_instance），再 raise ValueError——唯一 raise 出口 | session_kv_manager.py:1486-1511, 911-945 |
| 无缺口直通分支 | 逐 rank `remaining >= required` | 不触发逐出，`admitted=True` | session_kv_manager.py:1514-1519 |
| **LRU 逐出循环分支** | 存在缺口 | 受害者池 = `_completed_local_candidates`：`LOCAL_HBM` ∧ 本实例 ∧ 非 active ∧ `last_completion_ns` 非空的已完成会话，按 `(last_completion_ns, session_id)` 升序（= LRU 序）；剔除 `protected_sessions`（触发请求自身会话）；候选池扫描经 `full_scan_observer` 如实计数；逐 victim 整体 `_evict_session`，**每逐一笔立即重查逐 rank 缺口，够即停** | session_kv_manager.py:1520-1559, 782-813 |
| 候选漂移守卫分支 | 循环内 victim 状态偏离（非 LOCAL / 换实例 / active / 未完成） | RuntimeError（捕获候选快照与执行间的漂移） | session_kv_manager.py:1539-1549 |
| **耗尽优雅推迟分支** | 候选逐光仍不足 | deep_gap 台账（reason=exhausted_completed_candidates）+ `admission_blocked` 压力事件（首观测 + 每请求/相位/目标恰一条 retry，有界去重）→ `CapacityResult(admitted=False, insufficient_ranks)`；不 raise | session_kv_manager.py:1561-1585, 856-899 |
| 逐出后复核分支 | 返回前复查被逐会话均已完成 / inactive / 非保护 | 任一违例 raise（入口守卫外的循环间漂移捕获） | session_kv_manager.py:1591-1602 |

配套的预约通路：`reserve_request_capacity` / `release_request_capacity` 在产线主链零调用（face 准入的容量收敛由 `ensure_physical_fit` 直连承载，`_reservations` 真实运行恒空），仅测试经反射驱动，保留为预约语义扩展位（session_kv_manager.py:1604-1664）。

### 阶段 3：驻留与共享（增长、完成、两态不变量）

| 分支 | 触发条件 | 行为 / 数据流向 | 代码位置 |
|---|---|---|---|
| 增长前置守卫 | 会话非 LOCAL / 非本实例 / 非 active | RuntimeError | session_kv_manager.py:1941-1947 |
| 增长收缩守卫 | 目标分片 < 现有分片 | ValueError（上下文不收缩） | session_kv_manager.py:1949-1951 |
| 增长阻塞分支 | 增量 `ensure_physical_fit` 失败 | 返回未准入（增长同样可触发整体逐出） | session_kv_manager.py:1952-1963 |
| 增长成功分支 | fit 准入 | `_add_shards(delta)`，shard/total/logical_tokens 前移；`grow_prefill`（phase=prefill）/`grow_decode`（phase=decode）入口分别校验实例在场 | session_kv_manager.py:1964-1971, 1973-2109 |
| **完成驻留分支（LRU 池的入口）** | `mark_complete` | 零逐出保留 KV：`active=False`、`last_completion_ns=completion_ns`（LRU 时序键的更新点）、落 `retain_complete` 事件、清该请求压力去重键；返回空逐出元组 | session_kv_manager.py:2111-2149 |
| 迁移交割分支 | `move_prefill_to_decode`：源 == 目标 | `local_hit` 传输（prefill_decode_local_reuse），零字节零水位变化 | session_kv_manager.py:2031-2056 |
| 迁移交割分支 | 源 ≠ 目标 | `noc_migrate` 整份迁移 + 目标容量先收敛（`decode_target_capacity`），阻塞 → `MoveDecision(transfer=None, admission_blocked=True)`（原"伪造 local_hit 占位 transfer"已删，防误读） | session_kv_manager.py:2013-2030, 2058-2089, 510-523 |
| 两态白名单分支 | 增量/全量审计遇非法 location、LOCAL 层数 ≠ 全层、REMOTE 携实例/驻留层、分片数不匹 TP 度、分片和 ≠ 总量 | RuntimeError fail-closed（PARTIAL 手工翻转的不可达证明见 test_face_kv_incremental_invariants.py:231-268） | session_kv_manager.py:1073-1098, 2234-2304 |
| 逐 rank 记账分支 | `used = weight + resident + reserved > capacity` 或负值 | RuntimeError（HBM 守恒） | session_kv_manager.py:1123-1131, 2240-2247 |
| 事件水位压缩分支 | 已消费 ≥ 8192 且 ≥ 现存一半 | 前缀整段删除，游标回退（均摊 O(1)） | session_kv_manager.py:63, 664-675 |
| 严格不变量分支 | `SH_STRICT_KV_INVARIANTS` 开 | 每次变更后追加全量审计 | session_kv_manager.py:1171-1172 |

### 阶段 4：LRU 淘汰 / 逐出（`_evict_session`——唯一逐出路径，session_kv_manager.py:1382-1459）

| 分支 | 触发条件 | 行为 / 数据流向 | 代码位置 |
|---|---|---|---|
| 逐出守卫分支 | victim active / 未完成 / 非 LOCAL_HBM / 无实例 | RuntimeError（仅"已完成 inactive 的完整本地驻留会话"可逐） | session_kv_manager.py:1399-1402 |
| **整体外迁主干** | 守卫通过 | 一笔 `remote_store` 覆盖 `[0, model_layers)` 全层域、全部 TP shard、全部本地驻留字节（`before=layers / after=0`）：逐 source rank 经最近边缘端口（`nearest_edge_rank`：Manhattan 距离最小、平局取最低 rank）走 XY 路径；`_remove_shards` 释放本地字节；状态迁移 `LOCAL_HBM → REMOTE_MEMORY`（instance=None, prefix=0）；落 `evict_full` 事件（reason 形如 `evict_request_physical_fit_full:layers0-L`，携带 before/after 水位） | session_kv_manager.py:1403-1459, 1315-1370, 271-284 |
| 一 victim 一记录分支 | 恒 | 多条 TP shard 传输不折算成多个逻辑 victim；`EvictionRecord` 恒挂单笔 `transfer` | session_kv_manager.py:1431-1442; test_face_tiered_eviction_sequence.py:288-327 |
| **过量释放放行分支（本仓特有）** | 单会话全量字节 > 空间缺口 | 仍整体释放、允许过释放、不 raise（spec §7——旧 D4-I3"恰好够"守卫已随两阶段流程删除，绝不出现部分层逐出） | session_kv_manager.py:1476-1479; test_face_tiered_eviction_sequence.py:137-163 |
| LRU 时序维护分支 | 恒 | 排序键 `(last_completion_ns, session_id)` 升序；不同完成时刻按真实 LRU 序，tie 按 session_id 字典序确定性稳定排序（无 `_eviction_class` / `trigger_type` / `next_request_type` 类型分层——去类型化） | session_kv_manager.py:782-813; test_face_tiered_eviction_sequence.py:165-180 |
| 活跃保护分支 | victim 候选含 active/in-flight 会话 | 永不入池；触发请求自身会话经 `protected_sessions` 过滤 + 返回前双重复核 | session_kv_manager.py:806-813, 1521-1525, 1591-1601; test_face_tiered_eviction_sequence.py:182-239 |
| 完成零逐出分支 | completion 边界 | face 语义：完成不触发逐出（驻留喂养 LRU 池） | session_kv_manager.py:2111-2149 |
| 边缘端口派生分支 | `edge_ranks` 未显式给定 | 缺省 = 完整 mesh 物理边界（行 0 / 末行 / 列 0 / 末列的全部 rank）；显式给定则查重、排序、逐 rank 坐标校验 | session_kv_manager.py:579-595, 243-260 |
| 远端池无界分支 | 恒 | 远端池无容量上限、自身从不逐出（sh 语义）；仅 `retire_terminal_session` 能核销远端账面 | session_kv_manager.py:546-553, 2151-2220 |

### 阶段 5：恢复（REMOTE → LOCAL 的物理链）

| 分支 | 触发条件 | 行为 / 数据流向 | 代码位置 |
|---|---|---|---|
| 全量回迁主干 | REMOTE_RESTORE 决策发射 | `remote_load` 链：控制侧（到达/间隔 gate）→ 边缘 rank `mem_load`（仅池端口流体计时）→ `edge → target` 数据 p2p（双端 `hbm_charge=False`）→ 目标 rank `local_hbm_kv_restore`（RESTORE 唯一数据计费，可与计算重叠）；节点名携带 `_kv_transfer_action` 精确段以注册 C++ code-7 传输锚点 | generate_face_trace.py:977-1065; graph_batch_builder.py:1417-1441 |
| 控制同缘分支 | `control_rank == edge_rank` | 直接 arm timer gate | generate_face_trace.py:991-992 |
| 控制跨缘分支 | 控制侧与边缘不同 rank | 1B request p2p 中继（arm + send/recv） | generate_face_trace.py:993-1009 |
| **store→restore 前递补偿分支（B4）** | 同会话存在在飞 store 支链尾部登记 | 回迁发射前补显式依赖：同缘 `arm_dependency(store mem_store id)`；跨缘 1B p2p 中继；有登记而回迁零边缘分片 → RuntimeError（防御端点） | graph_batch_builder.py:1157-1216 |
| 决策→发射窗口推迟分支 | 会话存在"已决策未发射"的 remote_store 计数 | 调度器推迟该会话下一 turn 的回迁准入（纯推迟零 mutation） | graph_batch_builder.py:462-472, 1042-1045; face_online_scheduler.py:1570-1575 |
| 恢复不可消费分支 | 恢复链在飞 | 会话 planner 态已前移但物理恢复先于列车体（readiness 屏障 + history_tp_ready_barrier all_reduce） | graph_batch_builder.py:1442-1445 |

### 阶段 6：逐出物理链与发射面（graph_batch_builder / generate_face_trace）

| 分支 | 触发条件 | 行为 / 数据流向 | 代码位置 |
|---|---|---|---|
| 逐出旁路支链分支 | history / prefill / decode / 阻塞逐出发射 | `_emit_side_branch` fork：全 rank 暂存 pending 依赖 → 快照 → 分支内自链 → 恢复主链；分支遗留未消费依赖 raise——逐出物理链不阻塞主链任何节点，HBM 争用由 C++ N-way 均分在线裁决 | graph_batch_builder.py:520-553, 1063-1086, 1315-1335, 1406-1415 |
| remote_store 直连分支 | `source_rank == edge_rank` | 单 `mem_store(hbm_access_mode=1)` = POOL_READ 唯一计费 + 池端口双异步 join；无 ack | generate_face_trace.py:901-921 |
| remote_store 中继分支 | source ≠ edge | 源 `comm_send`（源端 HBM 读唯一计费）→ 边缘 `comm_recv(hbm_charge=False)` 直通 → `mem_store` 池写 → 1B ack 双端（源收 ack 后物理释放） | generate_face_trace.py:922-975 |
| 触发门分支 | 逐出有 `trigger_gate` | 控制侧节点门对齐逐出开始时刻：history 逐出门 = 到达/间隔 gate；decode 逐出门 = 该成员 drain 列车 post-barrier 块末（`drain_block_ends`）；控制 rank == 源 rank 同 rank arm，否则 1B trigger p2p | generate_face_trace.py:736-783; graph_batch_builder.py:748-770, 1316-1334 |
| 双发射防护分支（B4） | 阻塞 attempt 已发射的逐出再次进入发射面 | `history_evictions_pending` / `decode_evictions_pending` 双账本：决策日志账本保留全集，待发射账本按 `id(transfer)` 剥离已发射者——最终准入批 / joiner 列车头不重发同一批 remote_store | face_online_scheduler.py:244-249, 1617-1628, 1747-1758 |
| store 尾登记分支 | 每笔 remote_store 物理发射 | `pending_store_tails[session]` 登记边缘 mem_store 尾部（同会话至多一条——整体 store 单笔）；`pending_store_decisions` 计数核销；terminal 会话回收时清双表 | graph_batch_builder.py:1125-1155, 984-996 |
| noc_migrate 契约分支 | `noc_migrate` 传输进入 `_emit_kv_transfer` | fail-closed raise——NoC 迁移恒走 `_paired_transfer`（1000/3000 类）契约 | generate_face_trace.py:851-864 |
| 同实例零节点分支 | `_paired_transfer` 的源组名 == 目标组名 | 只 arm timer gate，零节点零字节 | generate_face_trace.py:451-455 |
| 头对齐切分分支 | `total_bytes % num_heads == 0` / 否则 | 整头所有权分片 / 精确字节配额回退（不取整） | generate_face_trace.py:462-475 |
| pending location 链分支 | REQUEST_COMPLETE 登记 / 后续逐出镜像 / 下一 turn 准入消费 | `deferred_session_locations` 承载完成位置，`sync_pending_history_after_evictions` 是唯一标记路径（remote_store → remote_memory）；准入时 pop 链初值 + 两态值域校验；partial 层域 store fail-closed | graph_batch_builder.py:998-1061, 1284-1292 |
| 零时长门分支 | 到达/间隔 `duration_ns == 0` | 不发射 gate 节点，直接以 `after_node_id` 编码依赖（等待由 C++ arrival alarm 承担，保持时长会双重等待） | graph_batch_builder.py:241-296 |
| 位置一致性分支 | pending gate 位置与计划快照不一致 | RuntimeError（无 partial 折算特例） | graph_batch_builder.py:1304-1313 |
| 终局核销分支 | terminal turn 完成 | `retire_terminal_session`：LOCAL → 释放本地字节；REMOTE → 远端账面静默核销（不发传输、不算逐出）；预约所有权与压力键清理；face 语义不发 KVCacheEvent | session_kv_manager.py:2151-2220 |

### 阶段 7：水位控制与审计（slo_tools 适配层）

**7.1 水位重放（hbm_watermark.py，WatermarkReplay）**

| 分支 | 触发条件 | 行为 | 代码位置 |
|---|---|---|---|
| 逐出重放分支 | 逐出条目 | 扣 victim 会话与实例占用；未跟踪会话 / 超跟踪字节 fail-closed；部分逐出计数（仅旧产物，新日志恒整体） | hbm_watermark.py:1479-1509 |
| restore-LOCAL_HIT 分支 | kind=local_hit | 零字节；source ≠ target 时占用随迁（防幽灵占用滞留源） | hbm_watermark.py:1571-1587 |
| restore-远端回载分支 | `source is None` | 只增目标 | hbm_watermark.py:1615-1625 |
| restore-同实例分支 | `source == target` | 只增不扣源（旧 partial 半层兼容） | hbm_watermark.py:1626-1633 |
| restore-跨实例搬移分支 | 源在目标外 | 扣源加目标；超会话跟踪字节 fail-closed | hbm_watermark.py:1634-1646 |
| restore 静默逐出对账分支 | kind=RECOMPUTE / state_before=EVICTED（旧产物） | 恢复点对账扣减（占用上界窗口） | hbm_watermark.py:1555-1566 |
| move-stay / move 分支 | `source == target 或 nbytes == 0` / 否则 | 归属切换占用不变 / 扣源加目标；0 字节跨实例交割计 `move_bytes_mismatch`（幽灵占用形态） | hbm_watermark.py:1648-1684 |
| grow 分支 | 目标 < 跟踪 | fail-closed（上下文不收缩） | hbm_watermark.py:1686-1699 |
| 自身去向归因分支 | completion 的 `kv_location_after_completion` | remote → 扣自身；partial → 保留上界计数；local → no-op；缺字段 / 未知枚举 → anomaly 留痕 | hbm_watermark.py:1701-1729 |

**7.2 可信度四层与容量三口径**

| 分支 | 触发条件 | 行为 | 代码位置 |
|---|---|---|---|
| certified 层 | journal + checksum 在场且四道门全过 | 唯一正式容量判决层（exit 3）；本仓自产 run 无 journal 生产者（M17 如实披露），恒落 upper_bound_only | hbm_watermark.py:20-56, 2331-2355 |
| 证书孤儿分支 | checksum 在场而 journal 缺失 | fail-closed exit 2（不静默降级为上界） | hbm_watermark.py:2335-2342 |
| resident_exact / lifecycle 层 | journal 在场但缺证书 / 证书有 false | 精确报告不出判决 | hbm_watermark.py:28-35 |
| upper_bound_only 层 | journal 缺失 | decision-log 重放即权威（上界口径），occupancy_valid 恒 false，不出物理违规认证 | hbm_watermark.py:36-46 |
| 容量链缺失分支 | profile / hardware / npus 任一环缺失 | capacity=NA 降级为峰值记录，绝不代拟容量 | hbm_watermark.py:2259-2287 |
| 三口径分支 | 恒 | per-rank physical 认证 / resident 硬上限（诊断）/ `kv_reserve_context_tokens` 水位目标（仅报告不判决）；负容量档（capacity < 权重分片）raise | hbm_watermark.py:762-850 |
| 口径升级分支 | 决策日志在场 `history_transfers` 列表 | face / face-LRU 映射升级 full 口径（契约行 + 逐段 restore，防 legacy 镜像双计与标量聚合击穿下界） | hbm_watermark.py:874-910, 2241-2246 |

**7.3 命中态映射（kv_cache_adapter.py）**

| 分支 | 触发条件 | 行为 | 代码位置 |
|---|---|---|---|
| 新产物位置分支 | `history_location_before` 在场 | `local_hbm → full`；`remote_memory → full`（整体驻留远端并恢复，非重算）；`partial_hbm_remote → partial`（仅旧产物可解析）；未知 → not_supported | kv_cache_adapter.py:143-158 |
| 旧产物回退分支 | 位置字段缺失 | `history_action` 四值表：NO_HISTORY→no_history / LOCAL_HIT→full / NOC_MIGRATE→full / RECOMPUTE→miss（旧 face 产物兼容） | kv_cache_adapter.py:159-170 |
| kind 闸分支 | `decode_admission_probe` / `prefill_admission_probe` | 跳过不消费但计数留痕；未知 kind fail-closed（双侧同盲会让对账恒 pass） | kv_cache_adapter.py:95-108, 727-731 |
| 事件提取双级分支 | 契约行在场（history_transfers / *_evictions / completion_eviction_transfers / prefill_decode_transfer 列表）vs 缺席 | 逐段传输对象产事件 vs legacy 聚合 bytes + 镜像逐出行；local_hit 恒零事件（名义规模非搬移量） | kv_cache_adapter.py:344-448, 451-479 |

---

## 三、两条机制在体系结构中的交汇

映射决策与 KV 状态在三个控制点上互相约束、互相改写：

1. **选择期谓词（映射读 KV）**：prefill / decode 实例选择的容量过滤以会话驻留位置为谓词——会话 `LOCAL_HBM` 且已在候选实例上时需求按增量折算（LOCAL_HIT 支路），否则按全量（NOC_MIGRATE / REMOTE_RESTORE 支路）；`schedulable_distance_limit` 圈定 decode 候选域，使跨实例迁移的物理代价上限在映射期即被编码。B4 的 store→restore 推迟门进一步以"已决策未发射的逐出"作为回迁准入的前置谓词，保证池写先于池读。
2. **准入期改写（映射写 KV）**：目标实例的准入需求是 LRU 逐出的唯一触发源——`ensure_physical_fit` 在目标实例上按 `(last_completion_ns, session_id)` 升序逐个整体外迁已完成会话，逐笔重查缺口；迁移与恢复则改写会话的 `instance_index` / `location`，从而改写下一轮映射的亲和谓词与 LRU 池的成员资格（活跃会话在完成边界进入池，恢复的会话重新活跃并离开池）。
3. **状态回灌（KV 约束映射的下游）**：逐出与迁移经 `kv_actions` 事件流、`KVTransfer` 决策日志与 pending location 链下行到构图层（物理链发射、store 尾登记）与分析层（水位重放、命中态统计），形成"映射决策 → KV 变更 → 水位/位置 → 下一映射决策"的闭环。两态模型使该闭环上不存在中间形态：任一时刻一个会话要么整体可亲和（LOCAL_HIT 支路谓词可满足），要么整体需搬运（MIGRATE/RESTORE 支路），映射组件据此在重用支路与重建支路之间二择。
