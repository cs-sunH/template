# astra-sim-wscllm-LRU 当前 workspace 准入审计

审计时间：2026-10-13。只读；未构建、未运行测试，也未改生产代码。本文实际审计对象是 `astra-sim-wscllm-LRU`，不是同级的 `astra-sim-wscllm`；以下代码行、`SessionKVCacheManager` 回收说明和 bridge 建议均只对应 LRU 仓库。对照历史报告 `die/docs/analysis/handoff-20261013-final/five-repo-workspace-audit.md` 中“Plan workspace never enters capacity admission or release”这一缺口。无 LRU 的 `astra-sim-wscllm` 对应入口、账本和策略差异另见 `work/workspace-admission-wscllm-current.md`。

## 结论

历史报告指出的缺口在当前 `astra-sim-wscllm-LRU` 仍然存在。Coarse Plan 已经计算并输出 `hbm_workspace_peak_bytes`，但运行时没有把它纳入 HBM 容量门、没有对它创建在役 reservation，也没有在终态/取消/提交异常路径释放它。当前 Python `SessionKVCacheManager` 与 C++ 每节点 issue 热路径之间没有同步的 workspace API，因此只给 `DieModelAdapter` 加一份本地总数会与权威 KV 状态脱节。

## 当前调用链和缺口位置

1. Python 调度器先选实例、再发 GraphBatch。`online/wsc_llm_online_scheduler.py:1135-1145` 调 `select_prefill_instance(snapshots)`，然后用原有 `static_mapping.route_for_prefill(selected)` 设 prefill/decode 路由。它管理的是请求/阶段，不是某个 C++ COMP 的 HBM workspace；`online/graph_batch_builder.py:483-515` 的批次字段也没有 capacity/workspace 快照。

2. C++ 在线 GraphBatch 提交后对被触及 rank 调用 `issue_dep_free_nodes()`：`main_online.cc:1606-1612`；完成后还会排每 rank 的延迟 issue pass：`main_online.cc:1478-1513`。`NodeStoreGraphSource::for_each_dep_free()` 按稳定 free-node 快照、升序遍历且不消费节点（`execution_driven/NodeStore.cc:392-404`）；`Workload::issue_dep_free_nodes()` 只检查 `hw_resource->is_available(nv)`（`workload/Workload.cc:398-415`）。因此现有位置可以在不改 selector/顺序的情况下加一个容量 gate：容量不够时跳过该候选，保留它在 free set 中，之后的 issue pass 可重试。

3. 现在一旦进入 `Workload::issue()`，先 `take_node`、再 `HwResource::occupy`、开始 online stats，之后才分派 COMP（`workload/Workload.cc:417-483`）。对 Die COMP，`issue_die_comp()` 最终调用 adapter（`workload/Workload.cc:504-568`）。所以将 admission 放进 adapter 的当前 `issue_die()` 太晚：此时 node 已 issued、GPU COMP slot 已占、stats 已开始。

4. coarse adapter 在 `workload/DieModelAdapter.cc:46-63` 才 compile/reuse Plan 并 `submit_plan`；计划缓存位于 `workload/CoarseDieRuntime.cc:47-80`，提交在 `:93-108`。Coarse Plan 中 `hbm_workspace_peak_bytes` 由 vendored `extern/die_model/coarse/src/Mapping.cc:940` 写入，`Compiler.cc:553-572` 读取它做候选映射排序；当前 wscllm-LRU 运行时侧的 `rg` 只找到这些 compiler/mapping 内部用途，没找到 HBM admission/reservation consumer。workspace 是 Plan 的峰值容量量，不是 `hbm_read_bytes_total` / `hbm_write_bytes_total` 传输量。

5. 正常成功 callback 在 `DieModelAdapter.cc:64-111` 移除 `coarse_active_` 后计算利用率并调用 `Workload::finish_generic_node()`。该终态函数在 `Workload.cc:1290-1345` 单次标记 node terminal、释放 HW、完成 stats/metrics、记录 terminal、最后 `finish_node()` 放开依赖。workspace token 应在成功路径调用 `finish_generic_node` 前归还一次。

6. 取消和异常目前没有 workspace 所有权。adapter teardown 在 `DieModelAdapter.cc:183-194` 用 `cancel(handle, now, false)` 后直接删 active entry；`CoarseDieRuntime::cancel()` 在 `CoarseDieRuntime.cc:127-157` 对 `notify=false` 删除 callback 并 retire。若未来使用 `notify=true`，adapter callback 在 `DieModelAdapter.cc:68-74` 见到 cancelled 就直接 return。workspace 释放必须覆盖这两条；notify callback 应先走统一 `release_once(handle)`，再按状态分支。另需注意 live cancellation 现有路径不会 terminalize/rollback graph node（历史审计也已指出），释放 workspace 并不自动修复该 graph 生命周期。

7. `submit_plan()` 先调用 `runtime_.submit()`，再登记 `completion_`，再排事件（`CoarseDieRuntime.cc:93-108`）；adapter 随后才登记 `coarse_active_`（`DieModelAdapter.cc:112`）。这些步骤任何一步抛异常时都可能没有终态 callback。现有 issue pass 在 `main_online.cc:1496-1510` 按 fail-closed 退出。新的 workspace token 应用 move-only/RAII 临时所有权覆盖 submit；只有 submit 和 adapter active-map 登记完成后才交给 adapter。若 submit 后登记失败，应先 cancel/retire 已建 handle，再由同一个临时 token 归还一次。若希望异常后继续模拟，还要补 graph unissue/rollback；当前 `take_node()` 之后没有对应 unissue API，现行策略是致命失败而不是恢复继续。

## SessionKVCacheManager 现状及边界

`session_kv_manager.py:447-474` 的 `NodeHBMState.used_bytes` 精确等于 `model_weight_bytes + resident_kv_bytes + reserved_request_bytes`；容量来自拓扑硬件配置，按 rank 记录（`:728-748`）。`_remaining()` / `_insufficient()` 在 `:912-931` 按 rank 判断 exact fit；`ensure_physical_fit()` 在 `:2114-2284` 会按既有 LRU 策略逐出非 active、非 protected 的完整本地 session，且把不能逐出的 request reservation 与 active/protected resident 纳入不足判断。`reserve_request_capacity()` / `release_request_capacity()` 在 `:2286-2394` 管 decode KV 预约，并同步 `reserved_kv_delta_bytes` journal；`:2416-2455` 的 extend 保持同一 KV 预约语义。

当前 manager 没有 workspace 字段或 workspace API。`hbm_snapshots()`（`:871-878`）明确标成 test-only；不应直接把它改成热路径 API。也不应把 transient workspace 塞进 `reserved_request_bytes` 或 `reserved_kv_delta_bytes`：那会把它伪装成 KV 预约，污染 KV journal/指标。`GraphBatchCommitter.cc:1578-1588` 也明确把 `kv_actions` 当作 Python 权威、C++ opaque；不能借 `kv_actions` 的现有格式暗中承载 workspace。`DecisionMailbox.hh:22-27` 规定普通 node completion、普通 rank idle 和 metrics-only 事件都不触发 Python 决策 round-trip；现有 bridge 是阶段/请求决策通道（字段约定见 `DecisionBridge.hh:36-55`），不是 C++ issue 时可直接调用的 Session manager RPC。

## 可实施的最小 API / 所有权建议

保留 `SessionKVCacheManager` 为权重/KV/请求 reservation 的唯一权威，不复制一份 KV ledger。新增独立的 per-rank transient workspace 账本，并用版本化的 bridge capacity 视图把两边接起来：

- Python manager 新增生产 API，例如 `capacity_snapshot_by_rank()`，返回每 rank `{capacity_bytes, model_weight_bytes, resident_kv_bytes, reserved_request_bytes}`。不要复用 test-only `hbm_snapshots()`。GraphBatch 响应携带带版本的 `hbm_capacity_snapshot`，C++ GraphBatch commit 后更新各 `Workload`/`Sys` 的 base view。
- C++ 每个 rank 有 `WorkspaceAdmission`/等价 owner，只保存 transient token：`try_reserve(node_id, generation, plan_workspace_bytes) -> move-only token`；活跃项 keyed by `(rank,node_id,generation)`，同时维护 `workspace_reserved_bytes_by_rank`。判定逐 rank 用 checked subtract/add，exact-fit 接受，`workspace > capacity - base_used - live_workspace` 时拒绝；即使调用 serial 也要让 ledger 支持多 token，以覆盖未来/跨调用重叠，避免 check-then-reserve 分离造成超订。
- 在 `Workload::issue_dep_free_nodes()` 中，对真实 coarse-v2 COMP 在 `take_node()` 前通过 adapter `prepare/compile_plan(node)` 取得同一个缓存 Plan，并从 typed summary 读取 `uint64_t hbm_workspace_peak_bytes`。随后原有 `is_available` 和 workspace feasibility 都通过才进入 `issue()`；容量不足时 `continue`，不改 ready order、不取节点、不占硬件槽。`issue_die()` 接收 prepared Plan，避免重复编译/两次产生不同决策。Plan summary 缺字段或类型不对应 fail-closed；workspace=0 可用空 token。
- C++ `StateDelta` 增加每 rank 活跃 `workspace_reserved_bytes`（仅容量状态，不是 KV delta）；Python 在处理下一次既有决策时把它作为不可回收的外部 transient hold 纳入 `SessionKVCacheManager` 的 fit/eviction margin。GraphBatch 响应的 base snapshot只含 manager 的 weights/resident/reserved KV；C++ 本地再加 active workspace，保证不重复计数。此协议允许保持现有“无普通节点逐个往返”的结构：在现有决策轮次上同步活跃 workspace，不为每个 COMP 建 Python RPC。
- token 的唯一所有者从 `Workload` 预留临时对象，成功 `submit_plan` 且 `coarse_active_` 登记后 move 到 adapter 的 active record；正常 callback、cancel notification、teardown suppress-cancel、提交异常统一经过 `release_once(handle or key)`，先从 owner map 中 `extract` 再归还 token。提交异常的临时对象析构归还；若已经产生 handle 则先取消/retire。不得让 callback 与 shutdown 各直接减账，也不得把 workspace 写入 HBM 流量、KV actions/journal 或 `tensor_size`。
- capacity snapshot 更新及 `StateDelta.workspace_reserved_bytes` 应在同一序号/回合内和 GraphBatch 绑定，避免使用不同 tick 的 residency。当前 bridge schema 明称 v1 冻结；这项需要明确的字段版本/兼容策略，不宜默默复用 opaque 字段。

这里 C++ 在役 workspace token 是 runtime transient owner；Python manager仍负责持久 weights/KV/reserved。新 workspace gate 只消费 plan peak 作为一项单独增量，`hbm_read/write` 继续只决定 traffic/利用率。C++ 的 `HwResource::is_available()`/selector 仍保留原意；workspace 不足的节点仅留在 free set 待后续 pass。

## 最小可复用 fixture

1. C++ admission 单测可放新独立文件，或先扩 `astra-sim/workload/execution_driven/tests/die_model_adapter_test.cc`（已有真实 `Sys`、NodeStore、EventQueue、Coarse Runtime/终态 fixture，Case B 在 `:368-428`）。给 workspace owner 一个小的 capacity snapshot 和预编译 Plan：`base_used + workspace == capacity` 必须 take/occupy/submit；加 1 byte 必须保持 free、不能占 GPU slot；两个 reservation 的并发算术不能过订；下次更新 base snapshot/释放 token 后再次 issue 应成功。断言工作期间 workspace ledger 增、成功 callback 后归零，HBM served/read/write bytes 和 KV 账本不变。再分别注入 `submit_plan` 失败、suppress cancel 和 notify cancel，断言每个 token 恰归还一次。注意当前同一 rank 的 GPU COMP 在 `HardwareResource.cc:236-237` 由 `is_available` 序列化，双 token 超订最好另加 ledger 独立并发单测，而不是假设 production issue pass 能同时发两个 COMP。
2. Python capacity fixture 可复用 `online/test_admission_net_credit.py:_real_manager()`（`:423-442`，四 rank TP、`local_hbm_capacity_bytes=1000`、strict invariants）；新增单测独立检查 transient workspace 与 KV reserved 的分列、exact-fit/overcap、两笔 token 叠加、释放后剩余容量恢复、未知/重复 release fail-closed、manager `used_bytes <= capacity`。验证 workspace 不产生 `reserved_kv_delta_bytes` journal 行，也不改变 HBM transfer bytes。现有请求预约 exact-fit / 深缺口与 invariant 测试（`test_tiered_eviction_sequence.py`、`test_wscllm_kv_incremental_invariants.py`、`online/test_admission_net_credit.py`）可继续复用其 manager 构造与断言风格。
3. bridge 集成 fixture 再用 `bridge_loopback_fixture.cc` + Python `online/verify/bridge_echo.py` 模式验证新的 capacity snapshot 和 `workspace_reserved_bytes` 在一轮里严格往返，`GraphBatchCommitter` 对缺失/错版本数据 fail-closed；这是验证跨进程所有权同步所需，单跑 adapter 或 Python manager 单测不能替代。

以上是静态代码审计和 API 建议；没有宣称任何新增实现或测试已通过。
