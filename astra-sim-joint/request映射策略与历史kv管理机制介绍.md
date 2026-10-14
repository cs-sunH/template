# request 映射策略与历史 KV 管理机制

> 以下所有路径均相对仓库根 `astra-sim-joint/`；位置引用精确到文件级（2026-10-14
> 远读退役三动作化后同步改写，动作宇宙 = stay/recompute/copy）。

## 一、体系结构总览

系统自顶向下分四层。**输入 trace 层**由 `sh_test_mesh/workload/llama2_7b_inference/plan_materializer.py`（manifest/metrics 物化）与 `generate_face_trace.py` / `generate_trace.py`（离线静态入口已删除，仅余只读符号，见 `generate_face_trace.py`）构成：请求以 `(session_id, turn_index, input_tokens, decode_length)` 形态自 CSV 队列进入，`select_first_session_requests`（`generate_face_trace.py`）做会话截断，inference_group 的 `pg_name × ranks` 定义了 rank→实例的静态编队，但**请求→实例的映射并不在此层发生**。**调度层**是 `online/sh30_online_scheduler.py` 的 `Sh30OnlineScheduler`：请求沿 `_push_arrival → pending_admissions → _admit_pass → _try_admit_request`（P 发射点准入）→ `_on_prefill_drain`（P→D 边界二次选点）→ `_complete_requests`（结算）的状态机推进；选择规则本体在 `joint/joint_scheduler.py`，计价在 `joint/joint_cost_model.py`（W3 世代：候选经 `action_work_graph` 构图 + `resource_projection` 事件前推，`cost = predicted_settle_done`）。**KV 管理层**是 `face_scheduler.py` 的 `KVCacheManager`：会话账本（`SessionKVState` 的 location 四态 local_hbm / partial_hbm_remote / remote_memory / none、resident_prefix_layers、base_* 快照）、容量逐出（T 类别序 × E 层数策略）、池恢复、merge 结算，以及唯一权威区间目录 `joint/kv_location_directory.py`（内容键 `(session, model_version, layer_range, token_range, tp_rank)`）。**模拟器执行层**为 `online/graph_batch_builder.py`（把调度决策翻译为逐 rank 的 Chakra 节点：copy 逐块交接支链）与 `astra-sim/astra-sim/system/` 的 C++ 交付核。

控制流：trace 层只提供到达流；每个请求在两个决策点（P、D）各做一次全实例 × 全动作候选遍历，`SelectionRecord` 逐候选落盘；KV 状态（驻留、部分驻留、池 backing、目录条目、容量摘要）经 `SessionKVView` / `PDBoundaryView` 只读快照进入计价，映射决策再经 `reserve_request_capacity / prepare_prefill / commit_stage_reservation_transaction / commit_boundary_decode_transition / merge_back` 五个事务写回 KV 账本——形成"KV 状态约束映射决策、映射决策改变 KV 状态"的闭环。离线侧 `sh_test_mesh/slo_tools/kv_cache_adapter.py`（命中态映射）、`hbm_watermark.py`（水位重放）、`restore_decomposition.py`（恢复三段分解）只消费决策日志，不回流控制。

---

## 二、request 映射策略全分支枚举

### 2.0 配置层控制点（决定分支形态的开关族）

配置解析单点：`joint/joint_config.py`（`parse_joint_config`），运行期只换配置不换代码。八组合预设 `COMBO_PRESETS`（`joint_config.py`）固定映射 (category_mode, scheduler_mode, layer_policy) ∈ {none,T,J,E,TJ,TE,JE,TJE}。

| 开关（env） | 取值 | 生效位置 |
|---|---|---|
| `JOINT_ABLATION_COMBO` | 八组合名 | `joint_config.py`（与显式开关同设即 fail-closed，`joint_config.py`） |
| `JOINT_CATEGORY_MODE`（T） | typed / lru | `joint_config.py` |
| `JOINT_SCHEDULER_MODE`（J） | joint / load-first / affinity-first / face_static | `joint_config.py` |
| `JOINT_LAYER_POLICY`（E） | adaptive / legacy_half / minimal_layer_groups | `joint_config.py` |
| `JOINT_ALLOWED_ACTIONS`（M2 掩码） | all / 逗号分隔三动作子集（候选集上限非强制） | `_parse_allowed_actions`，`joint_config.py` |
| 退役键 `JOINT_REMOTE_ACTIONS` / `JOINT_REMOTE_READ_TILE_BYTES` / `JOINT_REMOTE_READ_PARTIAL` | 任何显式值即拒绝（2026-10-14 远读退役，随 `JOINT_REMOTE_CREDIT_ITERS` 并入 `_RETIRED_ENV_KEYS` fail-closed） | `joint_config.py` |
| `JOINT_QUOTA_MODE` | off / static / aimd | `joint_config.py` |
| `JOINT_BOUNDARY_POLICY` | off / p-continuation（一次映射对照臂；要求掩码放行 stay，否则启动即拒） | `joint_config.py`、裁决 `joint_config.py` |
| `SH_TRAIN_MAX_ITER` | 缺省 8；显式 ≤0 fail-closed（有界性） | `parse_train_max_iter`，`joint_config.py` |
| face_static ⇒ quota-off 强制覆盖 | 非 off 显式覆盖为 off 并注记，不拒启动 | `joint_config.py`（F7 唯一耦合例外） |

预测配置 `JOINT_PREDICTION_CONFIG`（`joint/prediction_config.py`，W0 冻结单入口）正交五轴：`projection_mode`（known_work 正式 / frozen_diagnostic A 对照臂）、`future_demand_mode`（none / recent / reverting）、`history_use`（shadow / validated）、`stage_policy`（two_stage / once_mapping，与 `JOINT_BOUNDARY_POLICY` 同机制两入口，矛盾 fail-closed）、`settlement_policy`（consolidate / distributed）。（原 `action_space`（pure_four / source_subset_hybrid）与 `multi_source_protocol` 两轴已随 2026-10-14 远读退役删除，预设同步升版 predcfg-v4。）预设文件见 `sh_test_mesh/run_scripts/prediction_presets/{default,distributed,arm_a_frozen,arm_c_recent,arm_d_reverting}.json`（原 hybrid.json 已删除）。

### 2.1 P 入口动作适用性（三动作 × 会话 KV 状态的谓词面）

单点：`JointCostModel.applicable_actions`，`joint/joint_cost_model.py`。"某动作不适用不等于删除该实例"——实例候选集恒为全集，只剔除动作。

| 分支 | 触发条件 | 行为 | 位置 |
|---|---|---|---|
| stay 适用 | `history_tokens == 0`（turn-0 新会话，任意实例本地建立）或 `resident_instance == instance ∧ location ∈ {local_hbm, partial_hbm_remote}` | 记 stay 候选 | `joint_cost_model.py` |
| stay 不适用 | 上述不满足 | reason="history not resident at target" | 同上 |
| recompute | 物理上任何实例恒适用（R13 后重算的是缺失区间） | 恒 True | `joint_cost_model.py` |
| copy 适用 | `location != "none" ∧ history_tokens > 0 ∧` 非驻留于目标 ∧ 非（resident=None 且 home==目标 且 local_hbm 的 distributed 分裂形态） | 记 copy 候选 | `joint_cost_model.py` |
| copy 不适用 | 无可搬历史（REMOTE 整驻除外——REMOTE 基 ×copy@home 保持适用，K5 就地转正） | reason="no history to copy" | 同上 |
| M2 掩码尾部过滤 | `allowed_actions is not None` 且动作命中掩码外且原本可用 | 统一改判 `action excluded by JOINT_ALLOWED_ACTIONS mask` | `joint_cost_model.py` |

### 2.2 P 入口实例选择（select_instance_and_action，四模式）

位置：`joint/joint_scheduler.py`。枚举阶段对全部实例 × `ACTION_ORDER(stay, recompute, copy)`（2026-10-14 远读退役后三动作）调 `estimate_P_with_continuation`（只读，W3 事件前推；不适用候选 cost=None，`joint_cost_model.py`）。

| 分支 | 触发条件 | 行为 | 位置 |
|---|---|---|---|
| 非法模式 | `mode not in _SCHEDULER_MODES` | `JointSchedulerError`（fail-closed） | `joint_scheduler.py` |
| 无负载视图 | `cost_model.loads` 为空 | raise | `joint_scheduler.py` |
| **joint 主臂** | `mode == "joint"` | 全候选 argmin，键 `(cost_ns, (instance_index, ACTION_ORDER))`，`decision_stage="P"`；无适用候选（含掩码剔空）→ fail-closed raise（非容量拒绝） | `joint_scheduler.py` |
| **load-first** | `mode == "load-first"`（J-off 参照） | 全实例按 `(total_task_load_ns, index)` 选实例（无任何掩码），再 `_best_at_instance` 选动作 | `joint_scheduler.py` |
| **affinity-first** 亲和支路 | `home_instance is not None ∧ location ∈ {local_hbm, partial_hbm_remote, remote_memory}` | 选 home（偏好非掩码） | `joint_scheduler.py` |
| affinity-first 回退支路 | 无可复用 home | 落入 load-first 同判据 | `joint_scheduler.py` |
| **face_static** 无锚支路 | `resident_instance is None ∧ home_instance is None`（turn-0） | 静态球退化为空约束，全部实例参与，`ball_vacuous=True` 披露，不 fail-closed | `joint_scheduler.py` |
| face_static 掩码支路 | 锚在场（resident 优先，home 兜底） | 球内实例集 = `hops(anchor, i) <= floor(rho_eff)`（rho=B_D2D/B_HBM，`_static_ball_radius` `joint_scheduler.py`；ρ<1 退化半径 0 仅锚实例）；球内逐实例 `_best_at_instance` 后按 `(cost_ns, order_key)` 择优；披露 `masked_out_instances` | `joint_scheduler.py` |
| face_static 空球防御 | 掩码后为空（锚视野外陈旧索引） | fail-closed raise | `joint_scheduler.py` |
| face_static 球内零适用 | M2 掩码把球内全部实例动作剔空 | fail-closed raise（掩码是上限非强制） | `joint_scheduler.py` |
| 顺序模式落点零适用 | 选定实例上无适用动作（M2 掩码可达） | fail-closed raise | `joint_scheduler.py` |
| 平局 | cost 相等 | 固定 `(instance_index, ACTION_ORDER)` 确定序，不按事后输赢调整 | `joint_scheduler.py`（合同）、`:254-258` |

### 2.3 准入事务（_try_admit_request：P 决策点的提交与回退）

位置：`online/sh30_online_scheduler.py`。R2 事务化：预约 + prepare 为一个事务，失败零残留。

| 分支 | 触发条件 | 行为/数据流向 | 位置 |
|---|---|---|---|
| 到达前置违约 | `estimated_arrival_ns is None` | raise | `sh30_online_scheduler.py` |
| **distributed 结算依赖门** | `settlement_policy == distributed ∧ 会话上一轮结算账本在册` | 不消费：真实到达/排队照常，准入留队，落 `joint_admission` 行 `reason="settlement_open_dependency_gate"`，清失败键，return False | `sh30_online_scheduler.py` |
| 候选计价 | — | 构造 `SessionKVView` + `RequestView`（无 oracle：decode_length 结构性排除；n̂_D 来自条件剩余估计器 `estimate_conditional`）+ `select_instance_and_action` | `sh30_online_scheduler.py` |
| **配额判据分支** | `_quota_tracker` 在场 | `_quota_filter_candidates` 逐候选判：stay/recompute 恒适用；copy 链路门 + merge bulk 名额（`_quota_candidate_verdict`；realtime/elastic 流类与双端点平价门已随 2026-10-14 远读退役删除，在册流恒 oneshot） | `sh30_online_scheduler.py` |
| **quota_deferred 回队** | 全部动作不可行 | 落失败日志（`quota_deferred_{wait_reason}`），计 wait_reason ∈ {quota_link, quota_port}，记 `quota_deferred_since_ns`，失败键 = KV 纪元⊕候选集实例纪元⊕配额代数，return False（严禁零候选 fail-closed） | `sh30_online_scheduler.py`；回队语义 `joint/link_quota.py`（`quota_deferred_requeue`，永不 raise） |
| 配额可行集重选 | 有可行候选 | joint = argmin 同序；顺序模式 = 可行集上重跑选点规则（`_quota_reselect_feasible`，`sh30_online_scheduler.py`）；δ_adm=0 与既有比较恒等 | `sh30_online_scheduler.py` |
| **P 入口强制钩子** | `JOINT_FORCE_ENTRY_ACTION/_TARGET` 命中本请求 | 选择器产出后强制最终选择；per-action 形态按 `(cost, order_key)` 取最优实例，TARGET 形态精确匹配 (instance, action)，不适用/缺位 → fail-closed raise；与 boundary 钩子互斥武装 | `_apply_forced_entry_choice`，`sh30_online_scheduler.py` |
| **容量失败回退** | reserve/prepare 抛 `KVCapacityError` | 已提交逐出 bump 纪元 + 图同步 + 旁路支链发射 → `_release_orphan_reservation` → 失败日志 `capacity_deferred` → 失败键冻结（applicable 候选 ∪ selected 实例的纪元）→ return False（三态分类：存在任一组合物理可行 = 暂时不可行；全部不可行 = 结构性 fail-closed，`_classify_physical_feasibility`） | `sh30_online_scheduler.py` |
| 事务成功 | — | 纪元 bump + 图同步；runtime 落账（`prefill_instance_index / joint_action / joint_stage=P_COMMITTED / origin_home_instance / history_transfers / history_evictions`）；recompute 时改写工作描述（`_apply_joint_recompute_work_description`，整份 H+U 重放，`sh30_online_scheduler.py`） | `sh30_online_scheduler.py` |
| 流登记 | — | 主链 owner=rid、逐出支链 owner=rid#evict（merge 预留 owner=rid#merge-reserve；原前缀读流 owner=rid#prefill_read 与 est 承诺账本已随 2026-10-14 远读退役删除） | `sh30_online_scheduler.py` |
| **配额入册失败防御分支** | 主流程入册 False（判据与入册间 tracker 外变） | 回滚全部在途流承诺 + `_release_reserved_admission_or_fail`（未物化释放+回队，已物化 fail-closed——无 un-prepare 逆路径），failure_class="quota_deferred_quota_link"，return False | `sh30_online_scheduler.py` |
| deferred 驻留结算 | `quota_deferred_since_ns` 在场 | 记录首等待→准入 dwell 样本 | `sh30_online_scheduler.py` |
| 成功收尾 | — | 决策日志（逐候选 breakdown/hops/notes/measured=None）→ θ 预测暂存 → P 预测里程碑登记 → 入实例 qp → bump 实例纪元 → `_emit_admission` → return True | `sh30_online_scheduler.py` |

### 2.4 重试门（准入失败后的再评估时机）

位置：`_admit_waiting_requests`，`sh30_online_scheduler.py`。失败键 = `(kv_ledger_epoch, ((instance, ledger_epoch)...), quota_epoch)`，实例集冻结于失败时刻（选中 ∪ applicable 候选，F3 粒度红线——严禁粗化为全实例之或）。

| 分支 | 触发条件 | 行为 | 位置 |
|---|---|---|---|
| 键未变跳过 | last_key == 当前键 | FIFO 位置不变，跳过重试（重试必返同 False） | `sh30_online_scheduler.py` |
| 影子断言 | `SH_ADMIT_GATE_VERIFY` 开且键未变仍重试 | 若准入成功 → 等价性 violation raise | `sh30_online_scheduler.py` |
| 正常重试 | 键已变（容量释放/负载迁移/配额代数 bump） | 重跑 `_try_admit_request`；失败记录新键 | `sh30_online_scheduler.py` |
| 缺省失败键 | `_last_admit_failure_key` 缺席（异常路径） | 回退 `(当前 KV 纪元, ((0, 实例0纪元),))`——语义为"多开一次门" | `sh30_online_scheduler.py` |

### 2.5 D 边界动作适用性（applicable_D_remaining_actions）

位置：`joint/joint_cost_model.py`。判据源 = `BoundaryKVLayout.missing_intervals_for`（`joint/boundary_layout.py`：目标已持有权威副本的区间不算缺失；继承义务中已在向目标传输的区间不重复发送）。边界布局恒完整型单族（2026-10-14 远读退役起——P ∈ stay/copy/recompute，H_B 全量在 i_P；原增量型族（P = remote-read，基础区间在源端）随远读动作消亡，`is_increment_layout` 判别已删除）。

| 分支 | 触发条件 | 行为 | 位置 |
|---|---|---|---|
| stay | j == i_P | 原地继续 | `joint_cost_model.py` |
| recompute | 物理上任意 j 恒有定义（历史基数 0 重建） | 恒 True | `joint_cost_model.py` |
| copy | j 缺失区间非空（j≠i_P 补 H_B；第三目标补 H+U） | reason="no history to copy" 当 j 已持有全部 | `joint_cost_model.py` |
| M2 掩码尾部过滤 | 同 P 入口模板 | 剔除 | `joint_cost_model.py` |

### 2.6 D 边界选择（select_decode_target_and_action，式 1b/1c）

位置：`joint/joint_scheduler.py`。协议级边界规则，不随调度模式变体；原地继续 x0 归并在对应 `(i_P, 延续动作)` 记录上（恒 stay@i_P——2026-10-14 远读退役后完整型单族），不作第五动作。

| 分支 | 触发条件 | 行为 | 位置 |
|---|---|---|---|
| 掩码剔空 | 无适用候选 ∧ 延续动作被掩码剔除 | fail-closed raise（归因掩码，非容量拒绝） | `joint_scheduler.py` |
| 延续记录缺失 | 无适用候选且延续未被掩码 | fail-closed raise（边界视图与候选面不一致的结构性缺陷） | `joint_scheduler.py` |
| 掩码剔延续后记录仍现身 | continuation_masked ∧ 有 continuation 记录 | fail-closed raise（掩码过滤失效） | `joint_scheduler.py` |
| **延续被掩码时的选择** | continuation_masked | 允许面内 argmin，平局键退化为 `(cost, instance, action_order)`，披露 `continuation_masked_out=True` | `joint_scheduler.py` |
| 正常遍历后无延续记录 | 延续未被掩码但记录缺席 | fail-closed raise | `joint_scheduler.py` |
| **正常选择（式 1c）** | 有延续记录 | argmin 键 `(cost, 非继续标记, instance, action_order)`——**成本相同优先原地继续，只有严格更低成本才切换到迁移** | `joint_scheduler.py` |

### 2.7 D 边界提交事务（_on_prefill_drain：真实边界上的第二控制点）

位置：`online/sh30_online_scheduler.py`。

| 分支 | 触发条件 | 行为/数据流向 | 位置 |
|---|---|---|---|
| 幂等门 | `prefill_complete_ns` 已写（重复 P_COMPUTE_DONE） | 落 `boundary_duplicate_ignored` 行，return（D20 阶段 epoch） | `sh30_online_scheduler.py` |
| 边界前置清理 | — | 释放入态流（主链）、配额准入相 settle、expand_prefill 推进工作副本（原 #prefill_read 流与暂存 lease 注销已随 2026-10-14 远读退役删除） | `sh30_online_scheduler.py` |
| **已知终止直结** | `_joint_boundary_terminal`（decode_length==0 等） | `_settle_terminal_at_boundary` 直接结算，不做 D 选点 | `sh30_online_scheduler.py` |
| **一次映射对照臂** | `boundary_policy == p-continuation` | 不做二次全候选重估：decode_instance=state.index、decode_action = 默认延续动作（恒 stay——2026-10-14 远读退役后完整型单族），candidates 恒空、cost NA | `sh30_online_scheduler.py` |
| 两阶段闭环（缺省） | off | `_joint_boundary_cost_model` + `select_decode_target_and_action` + `_apply_forced_boundary_choice`（D 入口强制钩子，同 2.3 的 entry 钩子语义，`sh30_online_scheduler.py`） | `sh30_online_scheduler.py` |
| **提交点版本复查** | 决策→提交间 KV 纪元漂移 | `_VersionDrift` → 与容量失败同款回退结构 | `sh30_online_scheduler.py` |
| **跨阶段预约事务** | — | `commit_stage_reservation_transaction`（幂等 epoch；换址/resize/释放单事务）+ `commit_boundary_decode_transition`（工作副本迁移） | `sh30_online_scheduler.py` |
| **容量/版本回退** | 提交抛 KVCapacityError / KVStageVersionConflict / _VersionDrift | `selected == state.index`（继续态自身失败）→ 上抛 fail-closed；否则 abort 事务（足迹恢复提交前快照）→ 已提交逐出照常入账 → `selected = state.index` 重提交延续方案 → 落 `boundary_commit_fallback` 行（`fallback_rule` 分列 kv_capacity_error / stage_version_conflict） | `sh30_online_scheduler.py` |
| 回退后落账 | boundary_fallback | `joint_decode_is_continuation=True`，动作 = 延续动作；不暂存失败方案 θ 预测 | `sh30_online_scheduler.py` |
| recompute 元数据 | D 动作 recompute 且非回退 | `joint_decode_rebuild_pending=True`（首个 decode 列车前置重建 span）、suppress_sampling/output、stage_work 切 context_rebuild 模板 | `sh30_online_scheduler.py` |
| D_COMMITTED | 第一笔不可撤销新阶段操作（预约换址） | 此后本轮无第三次位置决策；获取全部来源目录引用（exec pin 源保护） | `sh30_online_scheduler.py` |
| 阶段引用闭合 | — | 释放 stage reference；预约注销（原"暂存转执行期 lease"已随 2026-10-14 远读退役删除） | `sh30_online_scheduler.py` |
| 异常安全（except Exception） | 边界窗口内任何异常 | 幂等注销 lease → 未到 D_COMMITTED 且事务在窗 → abort → 移出 pending_decode_ready → 释放 stage reference → 重抛 | `sh30_online_scheduler.py` |

### 2.8 D 边界工作副本事务（commit_boundary_decode_transition 的物理分支）

位置：`face_scheduler.py`。

| 分支 | 触发条件 | 行为 | 位置 |
|---|---|---|---|
| stay / copy | `action==stay` 或 copy | 复用 `move_prefill_to_decode`：stay=local_hit 零字节；copy=H_B 整份迁移（kind="stay_local_hit"/"complete_copy_migration"） | `face_scheduler.py` |
| **copy 落到 base 驻留实例折叠** | copy ∧ target == base_instance ∧ base 有片上前缀 | 退化为"就地并基"：迁移腿照发，落点即 base 侧；journal 闭合或重复基础副本就地释放；kind="complete_copy_base_fold" | `face_scheduler.py` |
| 重复边界事务防御 | `boundary_source_instance_index` 已在场且其间无结算 | fail-closed raise | `face_scheduler.py` |
| copy@第三目标 | 目标非 base 驻留实例 | U 迁移 + H 复制两腿（multi-leg copy-in，元组形态平铺入册）；执行面 `move_prefill_to_decode` 族 | `face_scheduler.py` |
| recompute（j≠i_P） | 重建不迁移 | 旧副本转来源侧记录（base_* 空闲则记 base，否则记 boundary_source——片上位置 ≤3）；j 侧工作副本零化起步，重建由 decode 增长钩子物化 | `face_scheduler.py`、来源记录 `face_scheduler.py` |
| recompute（j==i_P） | 原地重建 | 无迁移事务 | 同上 |
| （已退役） | — | remote-read 折叠/原地继续与 W10 hybrid 分支已随 2026-10-14 远读退役删除 | — |

### 2.9 P 相物化分支（prepare_prefill：动作 × 基础历史形态的真值表）

位置：`face_scheduler.py`。基形态 = 会话 KV 账本的 location/prefix。

| 分支 | 触发条件 | 行为/数据流向 | 位置 |
|---|---|---|---|
| 非法动作 | action ∉ 三动作 | raise | `face_scheduler.py` |
| **新会话建档** | `session_id not in _sessions`（history_tokens 必须 0 否则 raise） | 建立 home（此后异地执行/逐出不改 home），return `(None, (), ())` | `face_scheduler.py` |
| **distributed 目录驱动分支** | `session.distributed_layout` | 走 `_prepare_prefill_distributed`（覆盖断言取代单位置等式；来源/池恢复按目录条目迭代，锚 = 目录 base 锚） | `face_scheduler.py`；`face_scheduler.py` |
| 元数据失配 | context/total/shards 与请求不符 / home None | raise（账本完整性） | `face_scheduler.py` |
| **stay × LOCAL** | 驻留即执行实例 | `_local_hit_transfer`（零字节）；实例不符 raise | `face_scheduler.py` |
| **stay × PARTIAL** | 驻留即执行实例 | 缺失后缀 [p,L) `_ensure_capacity` + `_plan_suffix_restore_transfers` 池恢复 → 转 LOCAL 全层 | `face_scheduler.py` |
| stay × 其他 | 无驻留 | raise（stay 要求驻留历史） | `face_scheduler.py` |
| **copy@home × LOCAL**（退化防御） | copy 且 base 在执行实例且 LOCAL | 等价 stay local_hit（不建第二份工作副本） | `face_scheduler.py` |
| **copy@home × PARTIAL** | copy 且 base 在执行实例且 PARTIAL | 镜像 stay-partial：只恢复缺失后缀，working_kind 保持 None | `face_scheduler.py` |
| **recompute@home 本地事务** | recompute ∧ base 在执行实例 ∧ prefix>0 | 驻留字节不清零（空间复用口径非计算口径），缺失后缀层入账，prefix 推进 L，working_kind=None（与 stay 同构本地提交） | `face_scheduler.py` |
| **REMOTE 基 / 前缀 0** | `base_location == REMOTE_MEMORY or base_prefix == 0` | copy：整份池恢复为工作副本；recompute：工作副本零化起步，home := exec，working_kind=action | `face_scheduler.py` |
| **copy（home 侧驻留）** | 基础驻留 home，目标异侧 | 前缀 [0,p) 逐 chunk 交接流（`plan_copy_handoff_layer_chunks` + `CopyHandoffJournal`：块 0 走准入主链 readiness barrier，块 ≥1 构图器旁挂支链）+ PARTIAL 后缀池恢复 + 工作副本物化入账 | `face_scheduler.py` |
| **recompute（home 侧驻留）** | 同上 | 工作副本零化（不搬历史，重算 prefill 逐 chunk 物化） | `face_scheduler.py` |

### 2.10 decode 增长与停滞（R14：写入前才准备资源）

位置：`_joint_grow_decode`，`sh30_online_scheduler.py`。列车核销推进 `decode_tokens_consumed` 后逐列车因果增长（不按真实最终长度一次预约）。

| 分支 | 触发条件 | 行为 | 位置 |
|---|---|---|---|
| 无增长 | 会话缺席或 context 已达目标 | 直接 return | `sh30_online_scheduler.py` |
| 增长成功 | `expand_decode` 通过 | 纪元 bump；有逐出则旁路支链进图 + 流登记 | `sh30_online_scheduler.py` |
| **容量停滞** | `expand_decode` 抛 KVCapacityError | 已提交逐出入图后 `_enter_decode_stall`（stalled 态：暂缓后续列车参与，wake_key = KV 纪元⊕实例纪元；等待如实计入 E2E） | `sh30_online_scheduler.py`；唤醒 `_wake_stalled_decodes` `sh30_online_scheduler.py` |
| 全停滞死锁 | 全部 decode 停滞且无事件源 | `_check_decode_deadlock` fail-closed（宁要有诊断信息的 abort） | `sh30_online_scheduler.py` |

### 2.11 配额门与 AIMD 控制律（link_quota：动作级准入的资源治理分支）

模块：`joint/link_quota.py`。配额是**动作级准入机制而非实例掩码**：任一判据不满足只把动作标记不适用（inapplicable_reason 注明资源种类与余量），实例候选集不动；资源集合 = 有向链路 ∪ 实例 HBM 端口。

**流类别（C9 冻结；2026-10-14 远读退役后单类，`link_quota.py`）**：原 realtime（decode 远读读流，链路门 + 端口平价门）与 elastic（prefill 首遍读流，端口并发上限）两类已随远读机制删除，在册流恒 oneshot。

| 分支 | 触发条件 | 行为 | 位置 |
|---|---|---|---|
| oneshot（copy/merge/逐出写回/池恢复） | — | 全额计价 + 链路计数入占用与预留、端口在册 | `link_quota.py` |
| 链路 Q 派生 | 逐配置硬件 | `Q_init = max(1, floor(rho_eff))`（仅防结构性死锁；rho<1 域空化由定价选中率表达，非配额关闭） | `derive_q_init`，`link_quota.py` |
| 域出清 | `reachable_domain` | 域(t) = 经由仍有信用的链路与端口可达的实例集合——出清结果非输入，只做分析披露不裁剪候选 | `link_quota.py` |

**merge 义务预留（借还配对）**：链路侧双候选方向路径各 1 槽（owner=`rid#merge-reserve`），service_done 时刻 `adjudicate_merge_direction` 释放败者侧、胜者侧到 `_on_merge_done` 释放（`link_quota.py`）；端口侧胜者各 1 个 bulk-write 名额，封顶 `N_bulk = Q_init`，超限 quota_deferred（wait_reason=quota_port）。

**AIMD 控制律（仅 mode=aimd；static 下 Q 恒 = Q_init）**，`observe_telemetry` 按 `aimd_band_signal`（`link_quota.py`：rate < r_KV → shrink；r_KV ≤ rate < 1.2·r_KV → hold 死区；rate ≥ 1.2·r_KV → comfort，两触发集不相交结构性排除阈值 chatter）分派八动作：

| 动作 | 触发条件 | 行为 | 位置 |
|---|---|---|---|
| shrink（MD） | 收缩信号 | `Q ← max(1, floor(Q/2))`；在册流 grandfathered（余量负期间封新准入） | `link_quota.py`、`:222-230` |
| shrink_floor | Q 已在下限 1 的收缩信号 | 只记事件（streak 清零）不调整 | `link_quota.py` |
| hold | 死区 | streak 清零（打断"连续舒适"） | `link_quota.py` |
| comfort | 舒适 | streak 累计中 | `link_quota.py` |
| comfort_cold_start | 舒适但无流寿命样本 | 扩张抑制（T_expand 不可定义） | `link_quota.py` |
| comfort_frozen | 舒适但 `allow_expansion=False`（O6② r̂ 代表值回退窗） | 舒适判定照常、清空 streak、不评估扩张 | `link_quota.py` |
| expand（AI） | 连续 `T_expand = k(=10) × EWMA(流寿命)`（α=1/8）舒适 | `Q += 1`、streak 清零重计 | `link_quota.py`、`:370` |
| expand_capped | 触上界 `max(1, floor(B_link/r_KV))` | streak 清零不调整（防自毁性抖动） | `link_quota.py`、`:370` |

---

## 三、历史 KV 管理机制全分支枚举

### 3.1 命中判定（决策视图与观测口径）

决策侧单点：`_joint_session_view`，`sh30_online_scheduler.py`——P 计价的会话 KV 只读视图。

| 分支 | 触发条件 | 输出视图 | 位置 |
|---|---|---|---|
| 无会话 | turn-0 | `location="none"`、全零向量 | `sh30_online_scheduler.py` |
| **distributed 目录并集视图** | `distributed_layout` | 目录条目并集：锚 = token_start==0 的最大覆盖片上条目；锚完整覆盖并集 → local_hbm@锚；锚层残缺且残缺段池 backing → partial_hbm_remote@锚；残缺段在其他实例（跨轮留存）→ local_hbm 且 `resident=None`（stay/remote 不适用，copy/recompute 按锚差计） | `sh30_online_scheduler.py` |
| 跨实例执行期（working_kind 非 None） | — | 视图描述**基础历史**（base_* 快照；空串 base_location = 元数据腐坏 raise，None 落 REMOTE） | `sh30_online_scheduler.py` |
| 稳态 | — | 快照直通（local/partial 字节按层区间派生，missing = full − local） | `sh30_online_scheduler.py` |

观测侧（离线）：`sh_test_mesh/slo_tools/kv_cache_adapter.py`（`_hit_state_joint`）按 R12 披露字段映射：`history_tokens_before == 0 → no_history`；`joint_stay/joint_copy → full`（迁移非重算，命中语义看结果驻留）；`joint_recompute → miss`；字段缺席/越出词表 → `not_supported`（不猜测）；类型异常 fail-closed。

### 3.2 准入（预约足迹与可行性）

| 分支 | 触发条件 | 行为 | 位置 |
|---|---|---|---|
| **动作感知预约足迹** | 三动作恒 history+input 整份（2026-10-14 远读退役后；原 `joint_reservation_context_tokens` 单一裁决点与 remote-read/hybrid 的 input 增量口径已随增量族删除） | stay/copy/recompute = history+input 整份（decode 因果增长不预约；PARTIAL 跨实例后缀另按字节预约）——防 F1 幻影预约 | `face_scheduler.py` |
| 不可行实例 | `request_hbm_feasible_instances[target] == False` | KVCapacityError（容量类，可转延迟） | `face_scheduler.py` |
| PARTIAL 去钉扎 | joint R1' | PARTIAL 驻留不构成实例亲和掩码——容量只影响所需字节（final − resident），不做候选集过滤 | `face_scheduler.py` |

### 3.3 驻留与共享（唯一权威区间目录）

状态机（`kv_location_directory.py`，表外转移 fail-closed）：

```
WRITING → READY | RELEASED
READY → EVICTED_TO_POOL | RELEASED
EVICTED_TO_POOL → RESTORING | RELEASED
RESTORING → READY | EVICTED_TO_POOL | RELEASED
RELEASED → （终态墓碑，不进活跃表）
```

| 分支 | 触发条件 | 行为 | 位置 |
|---|---|---|---|
| 同 (内容, 持有者) 重复登记 | `_assert_no_duplicate` 命中 | fail-closed | `kv_location_directory.py` |
| 引用获取/释放 | `acquire_entry_reference`（exec_pin 位）/ `release_entry_reference` | 引用>0 或执行 pin 的条目不可驱逐；pin 在持时裸放引用 / 悬空 release / 越权 unpin → raise | `kv_location_directory.py` |
| **逐出** | `evict_to_pool`：READY → EVICTED_TO_POOL | 引用>0 raise；holder 翻 POOL_HOLDER、role 换 pool-backed、generation+1、owner 记录保留（release_condition 审计原角色） | `kv_location_directory.py` |
| **恢复发射** | `restore_from_pool`：EVICTED_TO_POOL → RESTORING@target | 非池基 raise；目标重复副本断言 | `kv_location_directory.py` |
| 在途 copy 去重 | `begin_inflight_copy` / `end_inflight_copy` | 同 (来源, 目标, 区间) 在途不重发 | `kv_location_directory.py` |
| **缺失并集查询** | `query_missing_intervals(session, target)` | 初等单元格（token×layer）并集去重：目标已持/已在途的格不重复搬运；多副本按冻结序（ROLE_ORDER：base → boundary-source → resident → working → pool-backed）只选读一条权威副本；部分重叠只补差集片段（字节按面积线性精确比例，不整除 raise） | `kv_location_directory.py`；矩形差集 `:646`；片段视图 `:683` |
| **片上来源查询** | `query_sources` | 池条目不进片上来源（池恢复另列 `query_pool_intervals`）；READY/WRITING 且有字节者 | `kv_location_directory.py` |
| 唯一保护来源 | `query_unique_protected_sources` | 某内容区间仅存一条有效条目时受保护（不可驱逐至零副本） | `kv_location_directory.py` |
| 元数据合并 | `merge_adjacent_metadata` | 同会话/holder/role/层区间、token 恰相邻、引用为零才合法（仅元数据，不免费物理搬运） | `kv_location_directory.py` |
| 容量投影 | `capacity_projection(holder)` | 只计片上状态条目（WRITING/READY/RESTORING）逐 rank 直和；池条目不占实例容量 | `kv_location_directory.py` |
| 有效保留量 V | `effective_resident_bytes` | 唯一内容按单元格并集只计一次（重复副本不增加 V） | `kv_location_directory.py` |
| 镜像同步 | `sync_entries` | 按差量原位改写/退休/登记；非法状态迁移 raise；RELEASED 目标 = 以缺席表达（写墓碑 raise） | `kv_location_directory.py` |
| 守恒断言 | `assert_conservation` | 不变量 1-5 全量（唯一性/并集只计一次/占容量/引用守恒/驱逐不清 owner） | `kv_location_directory.py` |

### 3.4 淘汰/逐出（T 类别序 × E 层数策略 × 留存段）

总入口：`_ensure_capacity`，`face_scheduler.py`。逐 rank 缺口 `gap = max(0, required − effective_remaining)` 驱动，满足即停。

**victim 合法集**（`_completed_resident_candidates`，`face_scheduler.py`）：location ∈ {LOCAL, PARTIAL} ∧ 驻留本实例 ∧ 非 active ∧ 已完成（last_completion_ns 非 None）∧ 不持阶段引用（D20：P 末→D 提交窗口的基础 KV 保护）∧ 类别匹配触发遍。类内 FIFO 键 `(last_completion_ns, str(session_id))`（`face_scheduler.py`）。受保护会话（触发请求自身）被剔除（`face_scheduler.py`）。

**T 类别遍历序**（`joint/eviction_priority.py`）：

| 分支 | 触发条件 | 行为 | 位置 |
|---|---|---|---|
| typed（T-on） | category_mode=typed | 两遍：human → tool；human 类内可释放层未耗尽且缺口未满足时不得转 tool | `eviction_priority.py` |
| lru（T-off） | category_mode=lru | 单遍类型无关全集合 | `eviction_priority.py` |
| 类别判定 | `classify_with_fallback` | 仅显式 `tool` 归工具类；human/None/未知按 human 优先级；未知值保留原始值 + fallback 标记披露 | `eviction_priority.py` |

**E 层数策略**（`joint/layer_eviction_policy.py` `plan_release`，只读输出 `EvictionPlan`）：

| 分支 | 触发条件 | 行为 | 位置 |
|---|---|---|---|
| 零缺口早退 | 初始 gap 全 0 | 空计划 satisfied=True（K8：早退也回等长零向量） | `layer_eviction_policy.py` |
| **legacy_half 第一段** | mode=legacy_half | 每**全本地**（prefix==L）victim 恰一步释放固定半层后缀 [L−L//2, L)；partial 已在后缀内直接跳过 | `layer_eviction_policy.py` |
| legacy_half 第二段 | 第一段后仍不足 | 整份外迁 [0, current)（全有或全无，不做步合并） | `layer_eviction_policy.py` |
| **minimal_layer_groups（E-off）** | mode=minimal | 无隐藏目标，victim 序从高向低释放最少完整层组到 0，满足即停 | `layer_eviction_policy.py` |
| **adaptive 第一轮（E-on）** | mode=adaptive | 扫各对象**软目标外**后缀 [k_hide+1, h]（高→低完整层组） | `layer_eviction_policy.py` |
| **adaptive 第二轮** | 第一轮后仍不足 | 扫**目标内**仍合法后缀（目标可被实际缺口突破，`target_breach_sessions` 披露） | `layer_eviction_policy.py` |
| 软目标惰性求解 | O4 | `retention_target_layers_fn` thunk 按需调用、plan 内缓存——缺口满足即停 ⇒ 未消费 victim 不触发 k_hide 递推 | `layer_eviction_policy.py` |
| adaptive 目标来源 | `_adaptive_retention_target_tokens`（C15 事件递推预测器，非 E 模式或无池速率或无输入估计 → 保守保留全部 L，标 `cold_start_unknown_input`） | k_hide = min{k: R̂_ℓ(k) ≤ D̂_ℓ, ∀ℓ>k}；解析锚 `k_hide_deadline` `layer_eviction_policy.py` | `face_scheduler.py` |

**步执行**（`face_scheduler.py`）：`layer_start <= 0` → `_evict_session`（整份 remote_store → REMOTE_MEMORY、instance=None、prefix=0）；否则 `_evict_suffix`（后缀 remote_store → PARTIAL_HBM_REMOTE、prefix=layer_start；home 不变）。两入口均守卫"仅已完成非 active 会话可逐出"（`face_scheduler.py/5796`）。

**distributed 留存段驱逐**（W9 区间级 E）：主位置 victim 耗尽仍不足时，逐 `_completed_retained_segments`（目录 owner 过滤 = 段所在实例即持有者；FIFO 继承会话完成时刻；驱逐经目录 `evict_to_pool`）——`face_scheduler.py`。

**终局与防御**：

| 分支 | 触发条件 | 行为 | 位置 |
|---|---|---|---|
| **深缺口** | 全类别耗尽仍不足 | `KVCapacityError` 携带逐 rank gap_records 与已提交逐出（D7 清账：调用方 bump 纪元+图同步）；落账 deep_gap_events 仅限确认不可恢复点（R14 死锁守卫 / merge 双侧深缺口） | `face_scheduler.py`；`commit_deep_gap_records` `:5854` |
| victim 状态漂移 | 逐出后复核发现 active/未完成 | raise（D4-I1） | `face_scheduler.py` |
| 过度逐出 | 撤销末笔仍全满足 | raise（D4-I3；末笔字节口径） | `face_scheduler.py` |

### 3.5 恢复（池恢复通路）

**池恢复通路**（`_plan_suffix_restore_transfers` → `_remote_load_transfer`，`face_scheduler.py`；RestoreGroupJournal 同文件）：后缀/整份经池边缘端口物化为热 KV，逐组发射 + ack 义务 + drain/prefill_drain/merge 三边界兜底结算（`_settle_restore_groups`，`face_scheduler.py`）。

**remote-read 读流通路（已退役）**：原"有限字节双缓冲 tile 流"（D09）整链——tile 裁决、每 rank 双缓冲暂存预约、prefill 前缀读流扫描次数闭式、多源 credit plan、来源目录化与 est 账本对账——已随 2026-10-14 远读退役整体删除（无开关保留档），登记见仓根 `PROVENANCE.md` 退役批登记节。

### 3.6 结算（compute_done 之后的归并）

策略分派点：`_complete_requests`，`sh30_online_scheduler.py`——`settlement_policy == distributed` → `settle_distributed`；否则（consolidate）→ `merge_back`。new_tokens 口径：恒 input + decode（2026-10-14 远读退役后；原 D 动作 remote-read（增量族）的 `_joint_working_context` 口径已随增量族删除）。

**merge_back（consolidate：少并多物理归并）**，`face_scheduler.py`：

| 分支 | 触发条件 | 行为 | 位置 |
|---|---|---|---|
| 重复结算 | `last_merged_request_id == trigger` | raise（版本键：stay 空 merge 也恰一次） | `face_scheduler.py` |
| **stay 本地提交** | `working_kind is None` | 零流量，direction="stay"，落 kv_delta 行 | `face_scheduler.py` |
| N9 防御（base==exec） | 基础驻留于执行端的 working 组合 | REMOTE 基/前缀 0 豁免（K5 就地转正）；boundary_source+recompute 豁免（D12 重复副本）；空串 base_location 腐坏 raise；其余 raise（R13 已消除合法来源） | `face_scheduler.py` |
| 轮末兜底结算 | — | `_settle_copy_handoffs`（C13 逐块交接）+ `_settle_restore_groups`（C15）在 merge 边界兜底 | `face_scheduler.py` |
| **三位置合并** | `boundary_source_instance_index is not None`（D 边界 recompute 的 i_P 旧副本） | `_merge_back_three_position`：完整并集持有者优先（recompute 重建副本 ⊇ 其他位置——零载荷）；否则保留量最大者为目标（平局 → 逻辑 home ω_B → D 实例 → 编号），一次性全局序，不依赖两两合并顺序 | `face_scheduler.py` |
| **方向裁决** | home_shards 空（REMOTE 基/前缀 0，无片上基础保留量） | direction="in_place"（裁定③：无主就地保留，零传输零池写，工作副本转正，home := exec） | `face_scheduler.py` |
| 零字节翻转 | `working_kind ∈ {copy, recompute}`（exec 恒持并集 ⊇ home；未知 kind fail-closed） | direction="reverse"，零传输，home 侧基础释放 | `face_scheduler.py` |
| **空间准备失败回退** | 三位置合并胜者侧 `_ensure_capacity` 抛容量错 | 按保留量降序回退（至多两次，`fallback_sequence` 披露）；双侧深缺口 → 落账 + fail-closed（热 KV 裁定封死"执行完再踢出去"，R4 自降级已删；两位置 reverse/in_place 均零传输零准备不可能失败） | `face_scheduler.py` |
| （已退役）方向二选一兜底 | — | 原 remote-read 双向比大小（forward/reverse 二选一）与 forward 迁移腿已随 2026-10-14 远读退役删除——两位置合并恒 ∈ {in_place, reverse}，真实搬移只在三位置分支发射（`_shard_vector_noc_transfer`） | `face_scheduler.py` |
| in_place 落账 | — | 无释放无入账无传输 | `face_scheduler.py` |
| 守恒断言 | 胜者侧字节 ≠ 全量 | raise（账本破损） | `face_scheduler.py` |
| 终态 | — | 全层 LOCAL@胜者、`home_instance = winner`（全仓第二个赋值点，I7）、base_* 清空、kv_delta journal 行、`last_merge_outcome` 披露 | `face_scheduler.py` |

**settle_distributed（distributed：目录提交型，无物理归并）**，`face_scheduler.py`：

| 分支 | 触发条件 | 行为 | 位置 |
|---|---|---|---|
| 重复结算 | 版本键 | raise | `face_scheduler.py` |
| 旧义务闭合 | — | copy 交接 journal / restore group 兜底结算 | `face_scheduler.py` |
| 基础段留存 | working 且 base 有片上前缀 | 旧基础段移入跨轮留存通道 E（`RetainedSegment` role=base，owner 记录保留，无字节移动） | `face_scheduler.py` |
| 边界来源段留存 | boundary_source 有片上字节 | 同上（role=boundary-source，token 域接续 base 段之后） | `face_scheduler.py` |
| **家族判定** | working_kind | None → `stay_local`（final=context）；copy/recompute → `zero_byte_flip`（守恒断言 exec==full union）；原 `remote_increment` 家族已随 2026-10-14 远读退役删除（未知 kind fail-closed） | `face_scheduler.py` |
| journal 残量防御 | settle 头部闭合后仍有 home 残量 | fail-closed（原 fail-open 已修） | `face_scheduler.py` |
| 会话收敛 | — | working_kind=None、base_*/boundary_source_* 清空、`distributed_layout=True`、home 不迁移（粘性 §2.1） | `face_scheduler.py` |
| 目录提交 + 披露 | — | 来源面枚举（source_count/interval_count + 跨轮增长断言 `assert_round_growth_within_budget`，`joint/distributed_settlement.py`）；`same_round_merge_saved_bytes` = 留存段片上字节（本轮省下的归并载荷） | `face_scheduler.py` |

结算闭合（两策略共用）：`_joint_close_settlement`，`sh30_online_scheduler.py`——释放来源引用（源保护解除）→ 未闭合继承义务 raise（settle 门）→ SETTLE_DONE + KV 纪元 bump（重开 distributed 依赖门）→ merge 载荷在途而 merge_done 缺席 raise → settle_done 行 → 预测成熟配对 → 账本弹出。

### 3.7 水位（容量摘要 θ 预测与离线水位重放）

**决策侧水位近似**（D11，式 19/20）：`InstanceLoadView.capacity_summary` = KVCacheManager 增量维护的分桶快照（T 类别 × E 后缀类 × 写回端口；逐 rank R/干净/脏/原子向量），候选只查询。

| 分支 | 触发条件 | 行为 | 位置 |
|---|---|---|---|
| θ 份额消费 | 逐桶固定序 | θ_b = min(1, max_{R_b,r>0} d_r/R_b,r)；TP 耦合（θ 由最重缺口 rank 决定，其余 rank 同比例释放）；预计脏写回 = θ·D_dirty（计价式 7 写回腿）；预计事务数 = ⌈θ·原子数⌉ | `consume_gap_theta`，`joint/capacity_summary.py` |
| 桶跳过 | 桶不能覆盖任何仍有缺口的 rank（全部 R_b,r == 0 于缺口 rank）| θ=0 跳过 | `capacity_summary.py` |
| **干净对象零写回费** | clean 分量 | 账面 clean 恒空（容量逐出恒经 remote_store ⇒ 驻留字节无有效 backing）——由池 backing 交叠真实推导而非硬编码 | `joint_cost_model.py` |
| **深缺口近似** | 全桶消费后仍有缺口 | active 工作因果结束估计（running+active 排空时刻）作有限近似，标 `capacity_wait_uncertain`（非可执行保证） | `joint_cost_model.py` |
| 摘要缺席退化 | 直接构造的测试夹具/离线视图 | 退 `reclaimable_bytes_by_tp_rank` 单桶合成锚（同一 θ 公式，分桶退化） | `synthesise_summary_from_reclaimable`，`capacity_summary.py` |
| 摘要与实际对账 | — | 决策半面 `capacity_theta_prediction` 暂存（`_stash_capacity_theta_prediction`）；实际半面由 kv_eviction 决策行按 trigger_request_id 对账 | `sh30_online_scheduler.py` |
| victim 未来恢复不计价 | — | 只承担当前请求真实前驱上的写回与空间等待（§14.4） | `joint_cost_model.py` |

**离线水位重放**：`sh_test_mesh/slo_tools/hbm_watermark.py`——从决策日志逐事件重建逐 rank KV 水位（evict/restore/decode move/base fold 审计行——staging lease 行已随 2026-10-14 远读退役删除），水位目标 `kv_reserve_context_tokens`（1M/rank）扣减后同式，"非任意时刻上限，仅报告不作判决"（`hbm_watermark.py`）；行预算 R 约束（`hbm_watermark.py`）。`restore_decomposition.py` 把 memory_anchor 对（restore_start/complete）与 prefill 窗口做三段分解（pre_prefill / hidden / exposed_restore_stall + hidden_ratio，§3.2 固定公式；无锚点请求四字段全 NA 并计数）。

---

## 四、两条机制在体系结构中的交汇

映射决策与 KV 状态在三个控制点互为因果。其一，**候选面由 KV 谓词裁剪**：`SessionKVView`/`PDBoundaryView` 的 location 四态、resident/home 锚、prefix 层数与目录区间并集直接决定三动作的适用支路（2.1/2.5）——映射组件读到的不是队列，而是 KV 账本投射出的"数据引力场"。其二，**代价由 KV 水位塑形**：HBM 剩余、容量摘要 θ 消费、在途流除数与配额信用共同进入每个候选的 merge_done 终点预测，空间缺口不改 argmin 的比较结构，但通过驱逐等待与配额 wait_reason 改变每个支路的成本与可行性，deferred/停滞请求在纪元键上等待 KV 状态迁移来重开重试门。其三，**映射提交反向改写 KV 状态**：`prepare_prefill` 的工作副本建档、D 边界的事务化迁移与来源转记录、`merge_back`/`settle_distributed` 的方向裁决与 home 迁移（全仓仅两个 home 赋值点：首建与 merge 胜者），每一次选择都推进目录状态机并留下 generation 痕迹；而准入期逐出（写侧）占用的水位，使"为下一个请求腾出的空间"本身成为下一个映射决策的输入（原"读写两侧 lease 互斥"已随 2026-10-14 远读退役收敛为单写侧）。两机制由此构成一个以目录为单一事实源、以两个决策点为控制点、以纪元键为反馈信号的闭环调节结构。
