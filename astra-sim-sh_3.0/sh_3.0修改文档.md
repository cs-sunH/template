# sh_3.0 修改文档：Prefill/Decode 实例分配策略调整

> 适用仓库：`astra-sim-sh_3.0`。
> 本文档自包含：执行者只需本文档 + 仓库代码。文中所有文件路径、类名、函数名、行号均已对照 sh_3.0 实际代码核实（行号以 `astra-sim-sh_3.0` 当前版本为准；注意仓库根目录《请求实例映射与KV冷热管理机制.md》的行号基于旧版本，已整体偏移，不要引用）。
> 核心结论先行：**所有改动都在 Python 规划层（trace 生成期），C++ 执行层（ASTRA-sim）零改动。**
> §3 的全部 6 个待确认问题已于 2026-08-08 逐条书面确认，结论内联在 §3 各条末尾并被 §5/§7/§8/§9 引用，可直接执行。

---

## 1. 总体目标

**一句话概括**：调整 Python 规划层（`face_scheduler.py`）的请求→实例映射——prefill 按"session 首请求避开边缘实例 / HBM 命中 sticky / 远端命中全局负载均衡"三段式分配，decode 固定在 prefill 同实例执行，KV 冷热管理机制本身不变。

**不改动行为清单**（以下行为必须保持现状，执行后需可回归验证）：

1. **KV cache 冷热管理策略不变**：三态驻留状态机（`local_hbm` / `partial_hbm_remote` / `remote_memory`）、两阶段"降水位"逐出（后 K 层优先、`oldest_completed_first` 排序）、远端恢复（含 PARTIAL 流水恢复）、`enforce_reserve` 完成后水位线检查、HBM 账本守恒校验，以上代码路径一行不改。
2. C++ 执行层不变：`extern/remote_memory_backend/analytical/AnalyticalRemoteMemory.{hh,cc}`、`astra-sim/workload/LocalHbmBandwidthModel.{hh,cc}`、`astra-sim/workload/Workload.cc`、`astra-sim/system/Sys.{hh,cc}` 均不涉及。
3. 实例布局不变：`trace_config.csv:21-29` 的 9 实例 × 6 NPU（TP=6）矩形布局、实例邻接图、通信组生成均不变。
4. 实例内 PD 混合执行模型不变：队首 prefill chunk + 全部 active decode 组成混合 iteration、LUT 查表计时、FCFS 队列语义不变。
5. 冷热相关元数据不变：ET GlobalMetadata 的 `kv_eviction_policy`、`kv_partial_resident_prefix_layers`、`hbm_kv_restore_bandwidth_sharing`（`generate_face_trace.py:1872-1881`）及 manifest 的 `kv_management` 段不改。
6. 远端内存端口模型不变：26 个边界端口、FIFO 队列、512 GB/s / 100 ns 代价模型、`nearest_edge_rank` 路由选择不变。
7. 硬件配置不变：`sh_test_mesh/hardware/face_case5_config_c.json` 是唯一硬件来源，不新增、不修改任何硬件字段。

---

## 2. 现状分析

### 2.0 决策架构：规划期静态固化

所有调度与 KV 管理决策在 **trace 生成期**由 Python 离散事件规划器预先确定，固化为 54 个 rank 的静态 Chakra ET；C++ 仿真器只忠实执行 ET。因此本次"分配策略"改动只落在：

- 主文件：`sh_test_mesh/workload/llama2_7b_inference/face_scheduler.py`（4172 行，改动完成后）
- 审计文案：`sh_test_mesh/workload/llama2_7b_inference/generate_face_trace.py` 的 manifest 段
- 测试：`sh_test_mesh/workload/llama2_7b_inference/test_face_scheduler.py`（42 个用例，改动完成后）

规划主循环入口：`plan_face_requests()`（`face_scheduler.py:3511-4172`），产物 `FacePlan`（`:3379-3392`）经 pickle 缓存后由 `write_face_trace()`（`generate_face_trace.py`）按每请求的 `prefill_instance_index` / `decode_instance_index` 写入对应 rank 的 `.et` 文件。

### 2.1 Prefill 现有分配链路

请求到达事件触发 `try_admit_request(request_index, now_ns)`（`face_scheduler.py:3697-3823`，定义在 `plan_face_requests` 内部）：

1. **HBM 可行性过滤**（`:3701-3718`）：`KVCacheManager.request_hbm_feasible_instances()`（`:1574` 起）返回逐实例布尔列表——最终 KV 分片在回收"已完成、非活跃"session 后能否放下。全集都不可行时再用 `request_hbm_eventually_feasible_instances()` 区分"暂不可行（返回 `False`，进 `pending_admissions` 等待重试）"与"永远不可行（`raise ValueError`）"。
2. **Session 亲和（仅 PARTIAL）**（`:3720-3727` 取快照、`:3762-3776` 分支 2）：若 `session_id in kv_manager.session_ids`（`:1425`），取 `session_snapshot()`（`:1888`）；当且仅当 `location == PARTIAL_HBM_REMOTE` 时，强制 `selected = history_snapshot.instance_index`，记 `prefill_affinity_reason = "resident_prefix_layers"`；该实例当前 HBM 不可行则 `return False` 等待。
3. **其余一律负载均衡**（`:3777-3782` 分支 3）：调用 `select_prefill_instance(snapshots, hbm_feasible_instances)`。**注意现状缺口：历史 KV 为 `LOCAL_HBM`（全在片上）时并不 sticky，而是参与全局负载均衡；若选中实例 ≠ KV 所在实例，由 `prepare_prefill` 做跨实例 NoC 迁移。**
4. 准入后（`:3784-3822`）：`reserve_request_capacity()` 预留最终 KV 容量 → `prepare_prefill()`（`:2840` 起）按历史 KV 位置分四类处理：
   - 新 session（`:2857-2873`）：在目标实例新建 `SessionKVState`（`location=LOCAL_HBM`，`context_tokens=0`）；
   - `LOCAL_HBM` 且同实例（`:2890-2899`）：`local_hit`，无传输；
   - `LOCAL_HBM` 且跨实例（`:2901-2934`）：`_ensure_capacity` + `_noc_transfer`（reason=`history_other_instance`），KV 整体经 NoC 迁到目标实例；
   - `PARTIAL_HBM_REMOTE`（`:2936-2985`）：**强制目标 == 驻留实例，否则 `raise ValueError`**（`:2937-2940`）；只远端加载后缀层 `[resident_prefix_layers, L)`（reason=`history_remote_suffix_restore`），恢复后归一化为 `LOCAL_HBM`；
   - `REMOTE_MEMORY`（`:2987-3023`）：全层远端加载到目标实例（reason=`history_remote_restore`），归一化为 `LOCAL_HBM`。
5. 请求入所选实例 FCFS 队列 `qp` 队尾，更新 `last_arrival_ns`。

### 2.2 Decode 现有分配链路

Prefill 最后一 chunk 完成时（`iteration_complete` 分支内，`face_scheduler.py:3931-3984`）：

1. `expand_prefill()` 把本轮 prefill 的 KV 增长记账到 **prefill 实例**（`:3937-3943`）。
2. 构造 `has_prefill`、`active_tokens` 两个辅助列表（仅供下一步使用），调用 `select_decode_instance()`（`:965-1053`）（原调用点已在改动 C 中移除）：
   - 候选集：`WeightedInstanceGraph.schedulable_instances()`（`:926-932`）= 实例邻接图上到 prefill 实例的最短加权距离 ≤ `FaceHardware.schedulable_distance_limit`（`:110`）= `d2d_bw / hbm_bw` ≈ 4050/1640 ≈ 2.47 跳（即 prefill 实例自身 + 一跳邻实例）；
   - 代价：逐候选查 LUT，`cost = (T_prime − T) / instance_size`，选择键 `(per_die_delta_ns, −remaining_hbm_capacity_bytes, instance_index)`；
   - HBM 只做硬约束（`decode_hbm_feasible_instances()`，`:1861` 起）与并列 tie-break。
3. 选定后（`:3948-3984`）：`move_request_capacity_reservation()`（`:1755`，同实例时 `:1766-1767` 直接 no-op）→ `move_prefill_to_decode()`（`:3099-3151`，同实例时 `:3111-3120` 走 `local_hit` 无传输；跨实例时 `_noc_transfer` reason=`prefill_decode_instance_migrate`）→ `expand_decode()`（`:3153`）→ `allocation_for_session()` → 释放预留 → 请求挂入 decode 实例的 `active_decode`。
4. 请求完成后 session 权威位置为 decode 实例（`session.instance_index` 在 `move_prefill_to_decode` 中更新）。

### 2.3 现有负载均衡策略（`select_prefill_instance`）的实现位置与接口

**名称**：`select_prefill_instance()`（`face_scheduler.py:860-881`），策略语义为"**HBM 准入过滤 + 剩余任务负载最小优先**"，manifest 中记为 `prefill_assignment_policy.primary_key = "ascending_total_task_load_ns"`（`generate_face_trace.py:2926-2957`）。

接口：

```python
def select_prefill_instance(
    loads: Sequence[InstanceTaskLoadSnapshot],   # 逐实例负载快照，含全部 9 实例
    hbm_feasible_instances: Sequence[bool],      # 逐实例布尔掩码（候选过滤的唯一入口）
) -> int:                                        # 返回选中 instance_index；无可行实例 raise ValueError
```

- 负载计量：`InstanceTaskLoadSnapshot`（`:813-857`），`total_task_load_ns = running_prefill + queued_prefill + active_decode`（均以 Roofline 估计服务时间 ns 计）。
- 选择键 `ordering_key`（`:851-857`）：`(total_task_load_ns 升序, last_arrival_ns 早优先且从未用过最优先, instance_index 配置顺序)`。
- **关键设计事实：候选集过滤完全由 `hbm_feasible_instances` 布尔掩码表达**。因此"只从不含边缘节点的实例中选择"不需要改 `select_prefill_instance` 本身——在调用点构造"候选集 ∧ HBM 可行"的复合掩码即可，接口零变更。

### 2.4 边缘节点的代码表示

- rank 级：`physical_edge_ranks(hardware)`（`face_scheduler.py:450-463`）——9×6 行主序 mesh 外圈 rank（`row ∈ {0, 8}` 或 `col ∈ {0, 5}`），共 26 个，即硬件配置 `face_case5_config_c.json:35` 的 `npu-selection: mesh-boundary`（远端内存端口挂接 rank）。
- 规划层持有：`KVCacheManager.__init__`（`:1318-1369`）在 `edge_ranks=None` 时自动调 `physical_edge_ranks` 并归一化存为 `self.edge_ranks`（`:1369`）；`plan_face_requests` 返回时透传到 `FacePlan.edge_ranks`（`:4168`）。
- 实例级（**本次需新增的判定**）：当前代码**没有**"实例是否含边缘 rank"的现成接口，需用 `FaceInstance.ranks`（`:149-168`）与 edge rank 集合求交自行计算。
- **当前布局下的具体结论（已逐实例核实）**：`trace_config.csv:21-29` 的 9 个实例中，instance 0（中心，rows 3-5 × cols 2-3，ranks `20-21,26-27,32-33`）是**唯一不含边缘 rank 的实例**；instance 1-8 每个都含有 `row∈{0,8}` 或 `col∈{0,5}` 的 rank。即"不包含边缘节点的 instance 集合"在当前配置下退化为单例 `{instance 0}`（该语义及配套等待行为已经 §3 问题 1 确认）。

### 2.5 片上 HBM / 远端存储 KV 位置的可查询接口

- 三态常量：`KVCacheManager.LOCAL_HBM / PARTIAL_HBM_REMOTE / REMOTE_MEMORY`（`:1314-1316`）。
- 单 session 查询：`KVCacheManager.session_snapshot(session_id)`（`:1888` 起）→ `SessionKVSnapshot`（`:1205-1219`），关键字段：
  - `location: str`——三态之一；
  - `instance_index: Optional[int]`——KV 驻留实例（`REMOTE_MEMORY` 时为 `None`）；
  - `resident_prefix_layers: int`——本地驻留的前缀层数（`L` 表示全本地）；
  - `local_bytes / remote_bytes / shard_bytes` 等账本字段。
- 存在性判定：`session_id in kv_manager.session_ids`（`:1425`）——session 是否已有 KV 状态记录。**该判定已被确认为"session 首请求"的判据（§3 问题 2）。**
- **单 session 的完整 KV 只放在一个实例内，不跨实例切分**（whole-head shard + 等尺寸实例校验保证，`build_instances()` `:187` 起）。"历史 KV 分散于多个 instance"在当前架构下不可能出现（见 §7 边界情况 B0）。

---

## 3. 待确认问题与确认结论（2026-08-08 全部确认）

以下 6 条原待确认问题已全部得到书面确认，结论内联于各条末尾。**执行者不得 reinterpret 这些结论；发现结论与代码现实矛盾时按 §8 停手上报。**

1. **非边缘候选集退化为单实例**。当前 9 实例布局下"不含边缘节点的实例集合" = `{instance 0}`（§2.4），所有 session 首请求的 prefill 都将涌入中心实例。原问题：(a) "边缘节点"是否就是 mesh 边界 rank？(b) instance 0 当前 HBM 不可行时，请求排队等待还是回退全集？
   **结论**：(a) 确认"边缘节点"= mesh 边界 rank（`physical_edge_ranks` 语义），候选集退化是预期行为；(b) **等待**——沿用现有 `return False` → `pending_admissions` 重试机制，不回退到含边缘节点的全集。（适用范围：该"不回退"结论针对**非边缘集合非空**——当前布局 = `{instance 0}`——时的暂不可行；"拓扑级不存在任何非边缘实例"的情形确认时未覆盖，2026-08-08 执行中由执行者上报，补充裁决见 §7-B8。）
2. **"session 首请求"的判定口径**。原问题：用 `session_id not in kv_manager.session_ids` 还是 `request.turn_index == 0`？
   **结论**：采用 `session_id not in kv_manager.session_ids`（与 `prepare_prefill` 的"新 session"判定 `:2857` 天然一致）。注意：启用 `request_queue_context_csv` 窗口外上下文时，带外部历史但规划器未放置 KV 的 session 会被当作首请求（其 `history_tokens != 0` 将触发 `prepare_prefill :2857-2861` 的 `raise ValueError`——现状行为，见 §7-B4）。
3. **LOCAL_HBM sticky 目标实例 HBM 不可行时的行为**。
   **结论**：**等待**（`return False` → `pending_admissions` 重试），与 PARTIAL sticky 现状（`:3770-3771`）一致，不回退负载均衡。
4. **decode 本地化后，现有 decode 选择设施的处置**。
   **结论**：**保留代码不调用**——`select_decode_instance()`（`:965-1053`）、`WeightedInstanceGraph`（`:884-950`）、`FaceHardware.schedulable_distance_limit`（`:110`）、`decode_hbm_feasible_instances()`（`:1861`）全部保留（测试仍引用 `select_decode_instance`；`graph` 仍被 `:3552` 构造、`:4147-4150` 用于 `final_edge_weights` 审计输出）。`FaceRequestPlan.decode_candidates`（`:3358`）字段保留，规划路径填**空元组**。
5. **审计文案同步范围**。
   **结论**：同步两处——manifest 的 `prefill_assignment_policy`（`generate_face_trace.py:2926-2957`）与 `decode_policy`（`:2958-2962`）两段文案，以及仓库根目录《请求实例映射与KV冷热管理机制.md》的 §1.2/§1.4（含修正其过时行号）。ET GlobalMetadata 经核实不含 prefill/decode 分配策略属性（`:1810-1881`），不动。
6. **sticky 与 `prefill_affinity_reason` 的取值**。
   **结论**：新增两个取值——LOCAL_HBM sticky 记 `"resident_local_hbm"`；session 首请求的非边缘负载均衡记 `"first_request_non_edge"`。既有取值 `None`（REMOTE_MEMORY 负载均衡）与 `"resident_prefix_layers"`（PARTIAL sticky）保持不变。（2026-08-08 追加第三个新取值：拓扑级无非边缘实例时首请求回退全集负载均衡记 `"first_request_edge_fallback"`，见 §7-B8；生产 9 实例配置下该取值不可达。）

---

## 4. 改动文件清单

| 文件 | 改动类型 | 影响范围 |
|---|---|---|
| `sh_test_mesh/workload/llama2_7b_inference/face_scheduler.py` | 逻辑修改 + 新增辅助函数 | `try_admit_request` 的 prefill 候选集与 sticky 分支；`iteration_complete` 的 decode 选择；新增实例级边缘判定 |
| `sh_test_mesh/workload/llama2_7b_inference/generate_face_trace.py` | 文案/审计字段更新 | manifest 的 `prefill_assignment_policy`（`:2926-2957`）与 `decode_policy`（`:2958-2962`）两段 |
| `sh_test_mesh/workload/llama2_7b_inference/test_face_scheduler.py` | 用例更新 + 新增 | 依赖 decode ≠ prefill / 跨实例迁移的断言按新语义改写；新增边缘排除、LOCAL_HBM sticky、decode==prefill 用例 |
| `请求实例映射与KV冷热管理机制.md`（仓库根目录） | 文档同步（已确认，§3 问题 5） | §1.2 Prefill 实例选择、§1.4 Decode 实例选择两节的行为描述，及全文过时行号 |

以下文件经核实**不需要改**：`generate_trace.py`、`config_resolver.py`、`metrics_schema.py`、`metrics_integration.py`、`metrics_postprocess.py`（指标层只按 `decode_instance_index` 归属事件，`metrics_integration.py:366-368` / `metrics_schema.py:222` 的字段语义在 decode==prefill 后依然成立）、`trace_config.csv`、全部 C++ 源码与 system/network/remote_memory 配置模板。

---

## 5. 逐文件改动明细

### 5.1 `face_scheduler.py`

#### 改动 A：新增"实例级边缘判定"辅助（新增代码）

- **位置**：建议放在 `physical_edge_ranks()`（`:450-463`）之后，作为同组拓扑工具函数。
- **现状行为**：无此判定。
- **目标行为**：给出逐实例布尔掩码——该实例的 `ranks` 是否与 edge rank 集合有交。
- **建议改法（接口契约）**：

```python
def edge_free_instance_mask(
    topology: FaceTopology,
    edge_ranks: Sequence[int],
) -> tuple[bool, ...]:
    """逐实例返回 True 表示该实例不包含任何边缘 rank。

    edge_ranks 来源：KVCacheManager.edge_ranks（:1369，已归一化）。
    返回长度 == len(topology.instances)，按下标对齐，与
    select_prefill_instance 的 hbm_feasible_instances 掩码同形，
    可直接逐元素 AND 复合。
    """
```

- **数据结构/接口变更**：纯新增，不改任何现有签名。调用点在 `plan_face_requests` 内一次性计算（`kv_manager.edge_ranks` 在规划期不变）。

#### 改动 B：`try_admit_request()` 的 prefill 分配三段式（核心逻辑修改）

- **位置**：`face_scheduler.py:3722-3782`（`plan_face_requests` 内部函数 `try_admit_request`）。
- **现状行为**：仅 PARTIAL sticky；LOCAL_HBM 与新 session 一律全局负载均衡。
- **目标行为**：
  1. `history_snapshot is None`（session 首请求，判据已确认为 `session_id not in kv_manager.session_ids`）：候选集 = 非边缘实例 ∧ HBM 可行实例，在其上调用 `select_prefill_instance`；候选集为空则 `return False` 等待（已确认：不回退全集）。**例外（§7-B8）**：若拓扑级不存在任何非边缘实例（`edge_free_mask` 全 False），回退为全集负载均衡并记 `prefill_affinity_reason = "first_request_edge_fallback"`；
  2. `location in (LOCAL_HBM, PARTIAL_HBM_REMOTE)`：`selected = history_snapshot.instance_index`（None 检查沿用 `:3767-3768` 风格），该实例当前 HBM 不可行则 `return False` 等待（已确认：不回退）；
  3. `location == REMOTE_MEMORY`：候选集 = 全部实例 ∧ HBM 可行实例（即现状行为）。
- **建议改法（伪代码）**：

```python
        # :3720-3727 快照与 history_snapshot 获取保持不变
        if history_snapshot is None:
            if not any(edge_free_mask):
                # 分支 1a（§7-B8）：拓扑级无非边缘实例 —— 回退全集负载均衡
                # （生产 9 实例布局不可达；玩具拓扑测试依赖此路径）
                selected = select_prefill_instance(snapshots, hbm_feasible_instances)
                runtime.prefill_affinity_reason = "first_request_edge_fallback"
            else:
                # 分支 1：session 首请求 —— 排除边缘实例
                candidate_mask = tuple(
                    feasible and edge_free_mask[i]
                    for i, feasible in enumerate(hbm_feasible_instances)
                )
                if not any(candidate_mask):
                    # 死等防护（§7-B2）：eventually 可行 ∧ 非边缘 交集为空 → raise 上报
                    # （request_hbm_eventually_feasible_instances 调用签名沿用 :3705-3718 既有调用点）
                    eventually_mask = request_hbm_eventually_feasible_instances(...)
                    if not any(
                        ev and edge_free_mask[i]
                        for i, ev in enumerate(eventually_mask)
                    ):
                        raise RuntimeError(
                            "edge-free candidate set can never become HBM-feasible"
                        )
                    return False   # 等待重试（已确认：不回退全集）
                selected = select_prefill_instance(snapshots, candidate_mask)
                runtime.prefill_affinity_reason = "first_request_non_edge"
        elif history_snapshot.location in (
            KVCacheManager.LOCAL_HBM,
            KVCacheManager.PARTIAL_HBM_REMOTE,
        ):
            # 分支 2：HBM 命中（全量或部分）—— sticky 到驻留实例
            if history_snapshot.instance_index is None:
                raise RuntimeError("resident history lost its instance")
            selected = history_snapshot.instance_index
            if not hbm_feasible_instances[selected]:
                return False   # 等待（已确认：不回退）
            runtime.prefill_affinity_reason = (
                "resident_prefix_layers"
                if history_snapshot.location == KVCacheManager.PARTIAL_HBM_REMOTE
                else "resident_local_hbm"
            )
        else:
            # 分支 3：REMOTE_MEMORY —— 全集负载均衡（现状不变）
            selected = select_prefill_instance(snapshots, hbm_feasible_instances)
```

- **关键说明**：
  - 分支 2 对 LOCAL_HBM 的 sticky 使 `prepare_prefill` 必然命中 `:2890-2899` 的 `local_hit` 分支（跨实例迁移路径 `:2901-2934` 成为不可达，**但该路径代码保留不删**，见 §6）；对 PARTIAL 行为与现状完全一致，`prepare_prefill :2937-2940` 的强制亲和校验天然兼容。
  - `:3705-3718` 的"全集不可行"预检查**保留在原位置不动**（它决定 raise vs 等待）；分支 1 的 `candidate_mask` 全空检查是新加的、更严格的等待条件（死等防护要求见 §7-B2，必须一并实现；该防护仅适用于非边缘集合非空的情形，拓扑级无非边缘实例时走分支 1a，见 §7-B8）。
  - 下游 `:3784-3822`（预留、`prepare_prefill`、入队）一行不改。
- **数据结构/接口变更**：`select_prefill_instance` 签名不变；`runtime.prefill_affinity_reason` 新增三个取值（已确认/已裁决）：`"first_request_non_edge"`、`"resident_local_hbm"`、`"first_request_edge_fallback"`（§7-B8）；其类型 `Optional[str]` 不变，`FaceRequestPlan.prefill_affinity_reason`（`:3354`）不变。

#### 改动 C：decode 固定到 prefill 实例（核心逻辑修改）

- **位置**：`face_scheduler.py:3944-3949`（`iteration_complete` 分支内，prefill 末 chunk 完成处）。
- **现状行为**：构造 `has_prefill` / `active_tokens`，调 `select_decode_instance()` 在 ≤2.47 跳候选集中按 LUT per-die 代价选择，`decode_candidates` 记录全部候选代价。
- **目标行为**：`selected = state_index`（即 prefill 实例自身），不再做候选评估。
- **建议改法（伪代码）**：

```python
                    # 删除 :3883-3887 的 has_prefill / active_tokens 构造
                    # （这两个局部变量仅供 select_decode_instance 调用使用）
                    # 删除 :3888-3907 的 select_decode_instance(...) 调用
                    selected = state_index   # decode 与 prefill 同实例
                    runtime.decode_instance_index = selected
                    runtime.decode_candidates = ()   # 已确认：字段保留，填空元组
                    # :3950 起下游全部保留：
                    # move_request_capacity_reservation（同实例 :1766-1767 自动 no-op）
                    # move_prefill_to_decode（同实例 :3111-3120 自动 local_hit）
                    # expand_decode / allocation_for_session /
                    # release_request_capacity_reservation / active_decode.append
```

- **关键说明**：同实例路径已被现有代码完整支持（`move_request_capacity_reservation :1766-1767` 与 `move_prefill_to_decode :3111-3120` 的恒等分支），因此 `decode_hbm_feasible_instances()` 的调用可随 `select_decode_instance` 调用点一并移除；decode 增长的容量检查由保留下来的 `expand_decode → _expand_local_session → _ensure_capacity` 链承担，不会绕过 HBM 账本。
- **数据结构/接口变更**：`FaceRequestPlan.decode_instance_index`（`:3357`）不变；`decode_candidates`（`:3358`）字段保留、内容固定为空元组（已确认）；`select_decode_instance` / `WeightedInstanceGraph` / `schedulable_distance_limit` / `decode_hbm_feasible_instances` **全部保留代码不调用**（已确认，理由：`graph` 仍被 `:3552` 构造、`:4147-4150` 用于 `final_edge_weights` 审计，测试仍直接引用 `select_decode_instance`）。

### 5.2 `generate_face_trace.py`

#### 改动 D：manifest 分配策略文案更新

- **位置**：`prefill_assignment_policy`（`:2926-2957`）、`decode_policy`（`:2958-2962`）。
- **现状行为**：`prefill_assignment_policy` 只描述"全局剩余负载最小"；`decode_policy` 描述"weighted_distance ≤ D2D_BW/DRAM_BW 候选 + LUT per-die 代价"。
- **目标行为**：如实描述新策略，例如：
  - `prefill_assignment_policy` 增加三个字段（命名自定）：`first_request_candidate_set: "instances_without_mesh_boundary_ranks"`、`resident_history: "sticky_to_kv_resident_instance_for_local_or_partial_hbm"`、`remote_history: "load_balance_over_all_instances_including_boundary"`；原有 `primary_key`/`tie_break`/`task_load_unit`/`components` 保留（`select_prefill_instance` 语义未变）。
  - `decode_policy` 整体改写为：`{"placement": "same_instance_as_prefill", "candidate_evaluation": "none", "kv_migration": "none_local_hit_only"}`（措辞自定，但必须能据此区分新旧 trace）。
- **数据结构/接口变更**：manifest JSON 两个段的字段增改；不触碰 `kv_management` 段与 ET GlobalMetadata（`:1810-1881`，经核实其中无分配策略属性）。

### 5.3 `test_face_scheduler.py`

#### 改动 E：受影响用例改写 + 新增覆盖

- **位置**：全文件（改动完成后 42 个用例）。
- **现状行为**：以下用例与新语义冲突，需逐一核对改写（行号为当前版本）：
  - `:1012-1039`（原 `:985-1017`，含被截断在区间边界的 `assertNotEqual`，原跨 `:1016-1019`，该断言已按本条改写移除）：用 `select_prefill_instance` 复算期望 prefill 实例、断言 decode 实例可取邻实例——按三段式重算期望（注意该用例使用 2×2 全边缘玩具拓扑，首请求走 §7-B8 fallback，期望 = 全集掩码下的复算结果，affinity reason 为 `"first_request_edge_fallback"`）；decode 断言改为 `== prefill_instance_index`。
  - `:1071-1113` `test_plan_pins_partial_history_to_resident_prefix_instance`：PARTIAL sticky 语义不变，但其中对 `decode_instance_index` 的断言（实际跨 `:1100-1107`）需改为同实例。
  - `:1677` / `:1698` / `:1712`：直接单测 `select_decode_instance`——已确认保留该函数（§3 问题 4），这些用例**原样保留**。
  - `:1872` / `:1884` 本身只是两次 plan 对比的确定性签名字段投影，无需改写；真正与新语义冲突的断言在 `:1899-1903`（`assertTrue(plan.decode_candidates)`、`assertIn(decode_instance_index, candidates)`、weighted_distance 检查）——按 `decode_candidates == ()` 与 `decode_instance_index == prefill_instance_index` 语义改写。
  - 原文档遗漏、执行中核实同样受影响的用例：`:293` `test_exact_prefix_metadata_reuses_truncates_and_recomputes`（2×2）、`:1593` `test_plan_assigns_prefill_by_running_and_queued_roofline_load`（2×2）、`:1759` `test_hbm_blocked_request_waits_until_active_session_completes`（1×2）——三者均使用全边缘玩具拓扑，§7-B8 fallback 保证不 raise，但期望值必须按三段式 + decode==prefill 语义逐一重算核对；`:1806` `test_request_larger_than_empty_instance_does_not_wait_forever` 不受影响（`:3705-3718` 全集预检查先抛 ValueError）。核对中若发现任何用例行为与 §2 现状分析矛盾，按 §8 停手上报。
- **新增用例（至少）**：
  1. 首请求 prefill 不落入含边缘 rank 的实例（当前配置下断言 `prefill_instance_index == 0` 且 `prefill_affinity_reason == "first_request_non_edge"`）——注意 2×2/1×2 玩具拓扑全边缘、不适用本断言，需构造含非边缘实例的拓扑（如带中心 rank 的 3×3 或等效布局）；
  2. LOCAL_HBM 历史 session 的后续请求 sticky 到驻留实例，`prefill_affinity_reason == "resident_local_hbm"`，`history_transfer.reason == "history_local_reuse"`，且无 `history_other_instance` 迁移记录；
  3. REMOTE_MEMORY 历史 session 的后续请求走全集负载均衡（构造边缘实例负载最低的场景，断言可以选中边缘实例）；
  4. 每请求 `decode_instance_index == prefill_instance_index`，`prefill_decode_transfer.reason == "prefill_decode_local_reuse"`，且 `decode_candidates == ()`；
  5. §7-B2 死等防护：非边缘候选集为空（但拓扑级非边缘集合非空）时的行为符合"等待"语义且 eventually 检查存在；
  6. §7-B8 fallback：全边缘拓扑下首请求走全集负载均衡且 `prefill_affinity_reason == "first_request_edge_fallback"`。
- **数据结构/接口变更**：无。

### 5.4 `请求实例映射与KV冷热管理机制.md`（已确认同步，§3 问题 5）

- §1.2 改写为三段式 prefill 分配；§1.4 改写为"decode 固定于 prefill 实例"；同时修正该文档全部过时行号（其行号与 sh_3.0 实际代码整体偏移约 40-50 行）。
- 2026-08-08 执行中核实、补充授权：该文档 §2.3/§2.6/§4 关于逐出排序命名与取值的陈述与 sh_3.0 实际不符——实际排序函数为 `_fifo_sort`（`face_scheduler.py:2602-2610`，排序键 `(last_completion_ns, session_id)`，语义与"oldest completed first"一致、仅名异）；ET `kv_eviction_policy` 实际值为 `"fifo_half_layer_suffix_then_full_fallback"`（`generate_face_trace.py:1872-1874`）；manifest `kv_management.policy` 实际值为 `"fifo_half_layer_suffix_then_full_session_fallback"`（`:3017`）。**同步修正这三处命名/取值陈述为 sh_3.0 实际**（仅文档陈述修正，代码与 ET 输出一行不动；§9-7 的逐字节比对对象为 ET/manifest 产物，不受文档修正影响）。文档头部"适用仓库"一并修正为 `astra-sim-sh_3.0`。除此之外的内容（如 §1.5 的历史请求队列括注）不在同步范围内。

---

## 6. 明确排除项

| 排除项 | 位置 | 不动的理由 |
|---|---|---|
| **KV 冷热管理全路径**（三态机、两阶段逐出 `_ensure_capacity :2760` / `_evict_suffix` / `_evict_session`、`enforce_reserve` 水位线、远端恢复 `_remote_load_transfer`、PARTIAL 流水恢复记账） | `face_scheduler.py` `KVCacheManager`（`:1311-3267`，含 `mark_complete :3197`、`enforce_reserve :3205`、`complete_request :3251`） | 需求第 3 条明确"冷热管理策略保持不变"。本次只改变这些机制的**调用时机/频率**（sticky 减少跨实例迁移、decode 本地化消除 `prefill_decode_instance_migrate`），机制代码一行不改 |
| `prepare_prefill` 的跨实例 NoC 迁移路径（`:2901-2934`，reason=`history_other_instance`） | `face_scheduler.py` | 新策略下不可达，但它是 `prepare_prefill` 的合法分支且被测试直接覆盖；删除属于冷热/迁移机制的行为变更，超出本次范围 |
| C++ 执行层（远端端口模型、HBM 带宽竞争模型、Workload 分发、Sys 解析） | `extern/remote_memory_backend/`、`astra-sim/` | 决策全部在规划层固化；C++ 无 session 级逻辑，本次无 C++ 改动 |
| ET 发射器（`write_face_trace`、远端 store/load 编码、流水重叠） | `generate_face_trace.py:1502-1664` 等 | 发射器只按 `prefill/decode_instance_index` 与 transfer 记录写 ET；分配结果变化不需要改发射逻辑 |
| 实例布局 / 邻接图 / 通信组 | `trace_config.csv:21-29`、`build_instances :187`、`config_resolver.py` | 需求未要求改布局；非边缘集合是布局的派生结果，不是新配置 |
| `WeightedInstanceGraph.increase_path/decrease_path`、`KVAllocator` | `face_scheduler.py:934-950`、`:1073-1164` | 现状已无规划路径调用（仅测试引用），与本次改动无关 |
| 指标层（`metrics_schema.py` / `metrics_integration.py` / `metrics_postprocess.py`） | 同目录 | 只消费 `decode_instance_index` 等结果字段，字段语义不变（§4 已说明） |
| `kv_eviction_policy` 等冷热元数据（ET GlobalMetadata `:1872-1881`、manifest `kv_management` 段） | `generate_face_trace.py` | 属于第 3 条"保持不变"的直接体现，改动会破坏回归比对 |

---

## 7. 边界情况与异常路径

**已澄清事实（不算开放问题）**：

- B0："历史 KV 部分在 HBM、部分在远端且**分散于多个 instance**"在当前架构下不存在——单 session 完整 KV 只驻留一个实例（§2.5）；`PARTIAL_HBM_REMOTE` 的远端部分属于统一逻辑池、无实例归属，sticky 目标唯一（`session_snapshot.instance_index`）。

**需求覆盖、需按本文档执行的边界**：

- B1：**首请求候选集为空**（非边缘实例当前 HBM 均不可行）：`return False` 进 `pending_admissions` 等待重试（已确认：不回退全集，§3 问题 1）。
- B2：**首请求候选集"永远不可行"的死等防护（必须实现）**：现有 `:3705-3718` 的 raise-vs-wait 判定基于全集，覆盖不了非边缘子集。执行时必须在分支 1 的 `not any(candidate_mask)` 处补充：将 `request_hbm_eventually_feasible_instances()` 与非边缘掩码取交，**交集为空时必须停手上报（不得静默死等，也不得自行回退全集——已确认的"等待"语义不覆盖此情形）**。说明：在当前等尺寸实例布局下该情形不可能出现（`KVCacheManager.__init__ :1332-1334` 强制等尺寸，"空实例能否装下"逐实例一致，全集 eventually 可行 ⇒ 非边缘实例也 eventually 可行）；但防护措施必须写出，以防未来布局变为不等尺寸。**适用前提：非边缘集合非空**（`edge_free_mask` 至少有一个 True）；拓扑级不存在任何非边缘实例的情形见 B8，不适用本防护。
- B3：**sticky 目标实例暂不可行**（LOCAL_HBM/PARTIAL）：`return False` 等待，与现状 PARTIAL 行为一致（`:3770-3771`）；请求已被 `reserve_request_capacity` 之前的逻辑保护，不会成为自身逐出受害者（session 在准入前先置 active 的机制保持不变）。
- B4：**带窗口外历史的首请求**（`request_queue_context_csv` 启用时）：首请求判据已确认为 `session_ids` 口径（§3 问题 2）——此类 session 会被当作首请求走非边缘候选集；其 `history_tokens != 0` 将触发 `prepare_prefill :2857-2861` 的 `raise ValueError`，该现状行为不因本次改动改变，也不在本次修复范围内。
- B5：**decode 同实例后的容量挤占**：decode 固定在 prefill 实例后，该实例 HBM 压力上升，`expand_decode → _ensure_capacity` 与 `complete_request → enforce_reserve` 会更频繁触发逐出——这是预期内的冷热机制自然响应，不是 bug；回归时应预期逐出/恢复计数与基线不同，但逐出**排序与两阶段语义**不变。
- B6：**`decode_candidates` 为空元组时的下游消费**：已确认规划路径固定填空元组（§3 问题 4）。manifest 逐请求记录与任何遍历 `decode_candidates` 的代码必须容忍空元组（`write_face_trace` 现有消费点需执行者逐一确认，若发现不允许空值的消费点，按 §8 停手上报）。
- B7：**等待-重试的公平性**：分支 1/分支 2 新增的 `return False` 复用现有 `pending_admissions` FIFO 重试（`admit_waiting_requests :3825-3831`），不引入新的饥饿机制；首请求集中涌向 instance 0 会显著增加其排队长度与等待时间，该行为已经 §3 问题 1 确认知悉。
- B8：**拓扑级非边缘实例集合为空（2026-08-08 执行中上报、当日补充裁决）**：`edge_free_mask` 全 False 时——测试玩具拓扑 2×2（`_tiny_kv_manager`，test_face_scheduler.py `:151-192`）与 1×2（`:1759`/`:1806`）的全部 rank 都在 mesh 边界上，即属此情形——"等待/不回退"语义无从附着（没有可等待的实例），§3 问题 1 的确认未覆盖。**裁决：首请求回退为全集负载均衡（同分支 3 语义），`prefill_affinity_reason` 记 `"first_request_edge_fallback"`**，以便审计与验收区分。当前 9 实例生产布局下该路径不可达（非边缘集合 = `{instance 0}`），不改变任何已确认的生产行为；§3 问题 1 的"不回退"结论与 B2 死等防护仅适用于非边缘集合非空的情形，均不受影响。选择回退而非 raise 的理由：raise 会迫使 5 个玩具拓扑用例重选拓扑并重标定容量/逐出期望值，放大改动范围且这些用例的 HBM 账本语义与分配策略正交；回退保持了"候选集过滤由掩码表达"的既有设计。

---

## 8. 冲突处理规则

1. 执行者发现本文档任何陈述（行号、函数签名、分支结构、字段名）与代码实际不符时，**必须停止修改并报告差异**，不得自行推断"应该是笔误"而继续。
2. 遇到本文档未覆盖的场景（包括但不限于：§3 已确认结论的新变体、B1-B7 之外的异常路径、测试中发现基线行为与本文档"现状分析"矛盾），**必须停止并报告**，不得自行扩展策略语义。
3. §3 问题 1-6 已于 2026-08-08 全部书面确认；执行中若发现确认结论与代码现实矛盾（如 §7-B2 所述情形真实出现、或"保留不调用"的设施被证明无法保留），**必须停手上报**，不得 reinterpret 结论。
4. 任何"顺手优化"（重构 `select_prefill_instance`、清理看似无用的代码、统一命名等）一律禁止；改动范围以 §4 文件清单为限，超出清单的需求必须回报。

---

## 9. 验收清单

每条均可独立判定：

1. **编译/生成通过**：`sh_test_mesh/run_scripts/generate_trace.sh` 完整跑通，产出 54 个 `.et` + `face_lut.csv` + `manifest.json`；C++ 未改动，`build/` 下既有二进制无需重编（若执行者重编，应无新错误）；测试套件通过——当前环境无 `python`/`pytest`（仅 `python3`，无 pytest 模块），用 `python3 -m pytest`（若可用）或 `python3 sh_test_mesh/workload/llama2_7b_inference/test_face_scheduler.py`（unittest 直跑）。**预存失败豁免**：`test_shell_config_does_not_build_full_face_plan`（`:194`）与 `test_checked_in_astra_compute_selection_uses_three_minute_window`（`:417`）在未改动基线上即失败（checked-in 请求队列已被裁剪为 4 请求/2 会话，与断言期望的 `REQUEST_COUNT=2091` / first_10_minutes 数据路径不符），属 checked-in 数据依赖型失败、与本次改动无关；验收标准为其余 40 用例全绿，且该 2 例失败特征与基线完全一致（不得因本次改动新增失败，也不得改动这 2 例去"凑绿"）。
2. **首请求不落入边缘实例**：manifest 逐请求记录中，凡 `prefill_affinity_reason == "first_request_non_edge"` 的请求，其 `prefill_instance_index` 对应的实例 ranks 与 `edge_ranks` 无交集（当前配置下恒为 instance 0）；当前 9 实例生产配置下 manifest 中 `prefill_affinity_reason == "first_request_edge_fallback"` 的出现次数应为 0（该取值仅为 §7-B8 退化拓扑兜底）。
3. **HBM 全量命中 sticky**：凡请求到达前 session 快照 `location == "local_hbm"`，其 `prefill_instance_index == history_source_instance_index`，`prefill_affinity_reason == "resident_local_hbm"`，且 `history_transfer.reason == "history_local_reuse"`、`history_transfer_bytes == 0`。
4. **HBM 部分命中 sticky**：凡 `location == "partial_hbm_remote"`，`prefill_instance_index` 等于驻留实例，`prefill_affinity_reason == "resident_prefix_layers"`，transfer 只含后缀层 `remote_load`。
5. **远端命中走全集负载均衡**：凡 `location == "remote_memory"`，其 prefill 实例选择键（`prefill_assignment_key`）等于在全部 9 实例 ∧ HBM 可行集上按 `ordering_key` 复算的最小者；构造用例证明可以选中含边缘 rank 的实例。
6. **decode 与 prefill 同实例**：manifest 逐请求 `decode_instance_index == prefill_instance_index` 全量成立、`decode_candidates` 全为空元组；全 manifest 无 `prefill_decode_instance_migrate` 与 `history_other_instance` 两种 transfer reason（前者因 decode 本地化消失，后者因 LOCAL_HBM sticky 消失）；每请求 `prefill_decode_transfer.reason == "prefill_decode_local_reuse"`。
7. **冷热管理回归不变**：
   - diff 层面：`KVCacheManager` 的逐出/恢复/水位线方法（`_ensure_capacity`、`_evict_suffix`、`_evict_session`、`enforce_reserve`、`_remote_load_transfer`、`expand_*`、`_check_invariants`）在本次改动 diff 中零改动；
   - 产物层面：ET GlobalMetadata 的 `kv_eviction_policy`、`kv_partial_resident_prefix_layers`、`hbm_kv_restore_bandwidth_sharing` 与基线逐字节一致；manifest `kv_management` 段与基线一致；
   - 行为层面：同一请求队列下，每个 session 的最终 `location`、`resident_prefix_layers`、逐出排序键（`last_completion_ns`）语义不变（计数可因分配变化而不同，但任一逐出记录都必须能被"两阶段、oldest-completed-first"规则复现解释）。
8. **审计字段一致**：manifest 的 `prefill_assignment_policy` / `decode_policy` 段文案与实际代码行为一致，且可据此区分新旧 trace；`prefill_affinity_reason` 的取值集合恰为 `{None, "first_request_non_edge", "first_request_edge_fallback", "resident_local_hbm", "resident_prefix_layers"}`（`"first_request_edge_fallback"` 为 §7-B8 追加，生产 trace 中应为 0 次）。
