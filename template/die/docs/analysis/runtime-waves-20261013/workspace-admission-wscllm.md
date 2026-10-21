# `astra-sim-wscllm` 当前 workspace 准入审计

审计时间：2026-10-13。只读；未构建、未运行测试，也未改生产代码。对象是 `/workspace/template/astra-sim-wscllm`，用于父任务计划中的首个无 LRU 仓库实现。历史审计 `die/docs/analysis/handoff-20261013-final/five-repo-workspace-audit.md` 所述“Plan workspace never enters capacity admission or release”在该仓库仍然成立。先前的 `work/workspace-admission-current.md` 实际追踪的是 `astra-sim-wscllm-LRU`，该文件标题与范围已更正；两个仓库的 KV 回收流程不能互相代用。

## 当前入口和未覆盖处

在线 GraphBatch 提交后，`GraphBatchCommitter::commit_after_preflight()` 在 `astra-sim/workload/execution_driven/GraphBatchCommitter.cc:1788-1796` 对 touched ranks 调 `issue_rank`。完成回调触发的 per-rank issue pass 则由 `WatchRegistry.cc:225-269` 排入同 tick deferred 队列。入口最终都到 `Workload::issue_dep_free_nodes()`：它按 free-node 快照顺序遍历候选，在 `Workload.cc:404-435` 只调用 `hw_resource->is_available(nv)`，然后直接 `issue(nv)`。因此最早且语义合适的准入点是这里、`is_available` 通过后、`issue()` 前：workspace 不足时跳过候选，节点仍留在 free set，不改 selector、ready 顺序或节点状态。

现有 `Workload::issue()` 在 `Workload.cc:438-450` 先 `take_node()`，再 `HwResource::occupy()`、启动统计，然后才分派节点。真正的 coarse v2 COMP 分派在 `Workload.cc:523-585`；CPU/timer/control marker 走不同路径，不应套用 Plan workspace gate。对 coarse v2，`DieModelAdapter::issue_die()` 在 `DieModelAdapter.cc:46-56` 才通过 `CoarseDieRuntime::compile()` 获得 Plan；`Mapping.cc:940` 将 `hbm_workspace_peak_bytes` 写入 Plan summary。故不能把检查只加在 adapter 的 `issue_die()`：那时 node 已从 free set 取走并占用了 COMP slot。

首个可实施入口是把 adapter/runtime 拆成“预备同一缓存 Plan”和“提交该 Plan”两步：在 `issue_dep_free_nodes()` 中只针对真实 GPU coarse-v2 COMP 调 `prepare_plan(nv)`，从 typed Plan summary 取得 `uint64_t hbm_workspace_peak_bytes`，通过后才调用原 `issue()`；`issue_die_comp()`/adapter 接收这个 prepared handle，禁止二次编译出不同 Plan。当前 `Plan` 的 `summary` 是 JSON（`extern/die_model/coarse/include/coarse_die/Plan.hh:68-69`），推荐在 coarse API 封装严格整数读取，而不是在热路径直接宽松转换 JSON 数值。字段缺失、负数、非整数或超过 `uint64_t` 必须 fail-closed；`workspace == 0` 可取得空 token。若首版只支持 coarse v2，应将其他 Die Plan 和 legacy 节点排除在此 gate 之外，不能把缺字段默认解释为零。

当前 `HardwareResource::is_available()` 对 `NodeKind::Compute` 要求本 rank `num_in_flight_gpu_comp_ops == 0`（`HardwareResource.cc:115-130`），注释确认线上仍是 0/1（`:37-44`）。所以目前同 rank 的 Compute/Plan 不并发；这不替代容量检查，因为活动 Plan workspace 仍与 Python 后续的 KV 预约同时占用 HBM。独立 workspace owner 仍应以 `(rank,node_id,generation)` 为 key 并做一次不可分割的检查加预留，避免以后硬件并发语义变化或其他提交路径绕过 gate。

## 持有、释放和异常边界

成功生命周期是 `issue_die()` 中 compile/submit，Coarse Runtime callback 移除 active 项并最终调用 `finish_generic_node()`：`DieModelAdapter.cc:54-112`。workspace token 应在 `submit_plan()` 成功、adapter active record 登记成功后转给 active record；正常 callback 先通过统一 `release_once(handle/key)` 归还，再进入可能抛异常的利用率处理和 `finish_generic_node()`。`CoarseDieRuntime::submit_plan()` 先 `runtime_.submit()`，然后登记 completion、`schedule_next()`（`CoarseDieRuntime.cc:93-108`）；adapter 再登记 `coarse_active_`（`DieModelAdapter.cc:112`）。因此提交期间用 move-only 临时 token；若 submit 已有 handle 而后续步骤抛异常，先 cancel/retire handle，再让临时 token 恰好释放一次。

取消时 `DieModelAdapter::shutdown()` 通过 `cancel(handle, now, false)` 删除 callback 并 retire，之后删除 adapter active entry（`DieModelAdapter.cc:183-200`；Runtime 的具体分支在 `CoarseDieRuntime.cc:127-157`）。`notify=true` 的 cancelled callback 目前在 `DieModelAdapter.cc:67-74` 直接 return。两个路径都必须调用同一个 `release_once`。新 token 不应靠 callback 和 shutdown 各自直接减计数；active map 先 `extract` 再归还可以阻止双释放。当前 `take_node()` 之后没有 unissue/rollback，dispatch 抛异常由在线 fatal barrier 中止进程；RAII 可保证进程内账本不泄漏，但不能声称异常后图可恢复。

## Python 端权威、真实策略和 selector

本仓 `SessionKVCacheManager` 是每 rank 权重与 KV/request reservation 的权威。`NodeHBMState.used_bytes` 精确是 `model_weight_bytes + resident_kv_bytes + reserved_request_bytes`（`sh_test_mesh/workload/llama2_7b_inference/session_kv_manager.py:255-282`）；容量按 topology rank 初始化（`:428-460`）。`hbm_snapshots()` 被注释为 test/observability-only（`:561-570`），目前不能当生产 bridge API。workspace 是短时 Plan 占用，不应加进 `reserved_request_bytes` 或 `reserved_kv_delta_bytes`，否则把 runtime token 伪装成 KV 状态/journal。

本仓的实际回收不是可直接照抄 `astra-sim-wscllm-LRU` 的策略。需要精确说清：虽然仓库名不含 LRU，这里的 `_candidate_sessions()` 仍按 `(last_completion_ns, session_id)` 排序（`session_kv_manager.py:594-610`）；不同点在 victim 处置和账本。当前仓 `ensure_physical_fit()`（`:1268-1366`）先按 `capacity - model_weight_bytes` 检查连空实例都放不下的情形，再按候选顺序删除，直到 fit；候选只包括同 instance、`RESIDENT`、inactive、已完成且未受保护的 session。`_delete()` 将 shard 从本地移除、状态置为 `EVICTED` 并清空 instance（`:1210-1266`）；manager docstring 明确 `evict_delete` 没有 ET 表示（`:420-426`），后续历史可走 recompute。候选耗尽返回 `admitted=False` 并记 deep gap（`:1332-1348`）。对照 LRU 仓，`ensure_physical_fit()` 会把完整 session 走 `_evict_session()` / `remote_store` transfer，释放本地 residency 的同时保留 remote-pool 账本（`astra-sim-wscllm-LRU/sh_test_mesh/workload/llama2_7b_inference/session_kv_manager.py:2115-2137,1757-1810`）。首版应复用当前仓的本地 delete/recompute 语义，不能凭 LRU 仓代码自行生成 `remote_store` 迁移 ET，也不能改当前仓的候选资格/顺序。

容量相关的 Python 调用不止一个入口：`reserve_request_capacity()` 通过 `ensure_physical_fit()` 预留 decode KV（`:1368-1423`；在线 prefill 准入处 `online/wsc_llm_online_scheduler.py:1236-1265`）；`prepare_history()`、prefill/decode `_grow()` 与 `move_prefill_to_decode()` 也都调用同一个 fit 入口（manager `:1541-1585`, `:1743-1777`, `:1810-1840`）。因此如果传入 C++ 活动 workspace 占用，应让所有 fit 路径以同一 rank margin 作判断，而不是只改 prefill reservation。

selector 和静态路由也应保持原样。到达时 `_on_arrival()` 根据 `_prefill_snapshots()` 调 `select_prefill_instance()`，之后沿既定 `static_mapping.route_for_prefill(selected)` 固定 prefill/decode 路由（`online/wsc_llm_online_scheduler.py:991-1008`）。`select_prefill_instance()` 的 ordering key 只有 `(request_count, instance_index)`（`wsc_llm_scheduler.py:535-547`）。`_admit_pass()` 遍历 sorted ready frontier（online scheduler `:1188-1234`），容量不足时 `_try_admit_prefill()` 返回 false 并保留队列（`:1236-1265`）。workspace gate 不应在 C++ 重新挑 rank，也不应将 selector 改成 workspace-aware。

最小权责划分建议：Python 继续唯一维护 weights/KV/request reservation；C++ 每个 rank 的 `WorkspaceAdmission` 只维护 transient workspace token。每个桥接输入携带 C++ 当前 `workspace_reserved_by_rank` 及 epoch/version，Python 在本次 admission 的 fit 运算中把它作为不可回收的外部 hold 相减，但不写进 `NodeHBMState.used_bytes`、KV journal 或 C++ 侧 HBM traffic。Python 返回的 rank capacity snapshot 则只包含 Python 的 `{capacity_bytes, model_weight_bytes, resident_kv_bytes, reserved_request_bytes}`。C++ 算术检查 `capacity - base_used - active_workspace`，拒绝 base 使用量本身越界、字段缺失/错 rank/错版本及整数溢出；exact fit 接受。

本仓在线 scheduler 的 `capacity_epoch` 已防止在相同容量状态对同请求重复尝试（`online/wsc_llm_online_scheduler.py:1248-1254`）。若 workspace holds 作为 fit 输入，hold 变化必须参与 epoch/wakeup：否则某次因为活动 workspace 阻塞的请求可能永不重试。workspace 释放后应使相应 rank 的 capacity epoch 前进；C++ 也要在 lease release 或 base snapshot 增容后安排 deferred issue pass。GraphBatchCommitter 目前只 issue 本批 touched ranks（`GraphBatchCommitter.cc:1788-1796`），所以 snapshot 改变了未 touched rank 时也须显式唤醒该 rank 的 free nodes。

## Bridge 快照和过订判断

现有 v1 bridge 没有 workspace/capacity 合约：请求 `StateDelta` 没有 active workspace 字段，v1 的 `snapshot` 明确保持空对象（`DecisionBridge.cc:177-230`，接口文档 `DecisionBridge.hh:36-47`）；响应只有 frozen GraphBatch 字段，`ParsedGraphBatch.hh:35-53` 拒绝未知 top-level key。不能把新容量藏进 `kv_actions`（它是 Python-authoritative KV action 流）或复用目前空的 `snapshot` 语义。

时序上有一个重要限定：当前 `ed_driver_tick_end()` 在一轮决策事件后同步阻塞等待 Python，拿到 GraphBatch 后只排一个 same-tick deferred commit（`main_online.cc:513-518, 590-597`）；`GraphBatchCommitter` 在 commit 内 issue touched ranks（其调用顺序文档 `main_online.cc:33-55`，实际 `GraphBatchCommitter.cc:1788-1796`），然后 C++ 才发 commit ack。当前 single-in-flight 和单线程事件循环下，Python 不会在同一响应到达和该次 issue pass 之间并行改 KV。因此，仅这个极短的 response→issue 窗口不会天然用到另一轮 Python mutation。

但如果响应的 Python capacity snapshot 被 C++ 长期复用，而 C++ 新建的 workspace lease 没有随下一条 `StateDelta` 回送，跨 epoch 会真实超订。具体序列：rank 容量 100，Python base 使用 60；epoch k 发给 C++ 的 snapshot 显示剩余 40。C++ 发射一个 workspace=30 的 Plan 后，实际占用到 90。该 Plan 尚未完成时另一请求到达；C++ tick-end 通过同一 bridge 送达 ARRIVAL。若 Python 仍按旧 KV-only 剩余 40 判断，它可以预留 25，base 变成 85；Plan 的 30 仍在役，合计 115。C++ 后来看到 base=85 后拒绝新 Plan 不能撤销已经超出容量的 active Plan。现有 `HardwareResource` 的 0/1 Compute 门只能避免同 rank 两个 Plan 同时运行，不能避免活跃 Plan workspace 与新 KV reservation 相加超过 HBM。对于未来允许同 rank 并发 Compute 的配置，若两个 25-byte workspace 都只读同一 base=60 snapshot，也会得到 `60 + 25 + 25 = 110`；故本地 check-and-reserve 仍须针对 ledger 原子更新。

首版 bridge 应显式扩展为 v2，而不是悄悄放宽 v1：请求携带 per-rank live workspace totals；响应携带 Python mutation 完成后的 typed base capacity snapshot，且两者都绑定同一 `delivery_sequence`/`batch_id`。C++ 在 GraphBatch parse/preflight 成功后，将该响应 snapshot 和 batch 一起原子应用，并在 touched-rank issue pass 前更新 workspace owner 的 base view；Python 用该次 StateDelta 里的 workspace view 完成本次 fit，再在 `run_variant_policy()` 后从 manager 生成响应 snapshot（handler 管线位于 `online_scheduler_base.py:351-390`）。commit ack 仍只在 C++ commit 后发送，Python 现有 provisional KV action 在 ack 时 finalize（`online_scheduler_base.py:1000-1073`），可沿用这个提交边界；容量快照本身不是 KV action。

为兼容旧接口，保留 v1 的原有字节/字段和行为只用于 workspace admission 关闭的运行；启用 coarse-v2 workspace admission 时必须要求 request/response 两端都声明/支持 v2，缺字段、版本不符、旧 peer 或过期 epoch 都在首次发射前 fail-closed，不能把旧 v1 的空 snapshot 当作“零 KV / 无限容量”。当前 bridge 和 typed parser 都严格锁定 v1（`DecisionBridge.hh:98-100`、`ParsedGraphBatch.hh:35-53`、Python `decision_bridge.py:133-143`），所以需要显式双版本解析/能力门或统一升级两端；目前没有隐式协商能力。

## 安全与活性需要分开

有版本的 snapshot 加本地 ledger 可以保证物理容量安全，但不保证新 Plan 一定能进场。Python 在 GraphBatch 中建出 coarse Plan 之前，并不知道本轮 C++ Plan 的精确 workspace；若 Python 恰好把 KV/请求 reservation 填满，C++ 会把新节点留在 free set。此时需要给 Python 的 fit 保留一个明确的 workspace quota/headroom，或新增能在 KV 准入前交换精确 Plan demand 的两阶段握手。quota 要按 rank 扣减，且只保留本仓 KV manager 的 delete/recompute 策略。若首版不提供 quota/握手，则 workspace 不足节点必须可观测地 deferred，后续 snapshot/lease 变化触发重试；配置上永远放不下的 workspace 必须作为 unschedulable fail-closed，而不是静默选择另一实例或无限等待。

因此准入时有三条同步规则： (1) Python 只基于最新 StateDelta workspace epoch 和自身刚完成的 KV mutation 生成响应 snapshot；(2) C++ 对该 snapshot 只在对应 batch 成功 preflight 后安装，并用同一 owner 在每个候选上原子 check/reserve；(3) quota 不足、snapshot 过期/漏 rank、checked arithmetic overflow、owner/key mismatch 时不发射，并向运行错误/diagnostic 明确暴露。现有 bridge 每次仅一个 in-flight delivery；若将来改为异步或多在途，则必须加按 rank 的 commit reservation/quota 或串行化 admissions，不能让多个 Python 响应共享同一“剩余空间”视图。

## 可复用的最小验证夹具

- C++ workspace ledger 单测可新建独立测试源文件，不改现有 CMake/生产块：容量 100、base 60，依次尝试 workspace 40（exact fit 成功）、1（失败）、两个 token 累加、溢出、重复 release/未知 handle；验证失败不改变 owner map。另用 `die_model_adapter_test.cc` 的真实 `Workload`/`Sys`/NodeStore/Coarse Runtime fixture 覆盖真实 Plan compile→issue→Runtime callback：capacity gate 在 `take_node()` 前拒绝时节点保持 free 且 COMP slot 不占用；success/cancel-notify/shutdown-suppress/submit 失败分别恰好释放一次。现有 coarse adapter 集成 Case 位于 `astra-sim/workload/execution_driven/tests/die_model_adapter_test.cc`；`die_execution_lifecycle_test.cc` 可借其终态与 issue-pass fixture。
- Python 侧以 `online/test_admission_net_credit.py` 的 `_real_manager()`（文件内约 `:397-416`）或本仓 `session_kv_manager.py` 的构造方式建 manager，传入 per-rank external workspace view。覆盖 `used_bytes` 仍仅等于 weights+KV+request reservation；`ensure_physical_fit`、`reserve_request_capacity`、history/grow/move 全路径扣 workspace margin；exact fit、当前 hold 导致的暂时 blocked、候选 exhausted、hold 释放后 retry、sequence stale fail-closed；确认 workspace 不产生 `reserved_kv_delta_bytes` journal 行，KV delete/recompute event 仍符合本仓现有语义。
- Bridge 集成可扩展 `bridge_loopback_fixture.cc` 与 Python `online/verify/bridge_echo.py` 的单交付方式：校验 workspace totals 和 post-mutation per-rank base capacity 同序号往返；v1 feature-off 往返保持不变；v1/缺字段/错 rank/错版本在 feature-on 下失败；snapshot 应在 GraphBatch preflight 后、issue rank 前安装；未 touched 但容量改变的 rank 被唤醒。容量快照和 workspace totals 都要 checked integer/domain 校验。

以上均为静态读取后的实现建议；本文没有执行任何新增构建或测试，不表示新实现已通过。
