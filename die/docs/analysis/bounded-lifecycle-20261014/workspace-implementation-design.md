# 无 LRU `wscllm` workspace 准入：接口落地细化

范围：`/workspace/template/astra-sim-wscllm`（无 LRU 仓）。本稿承接
[`workspace-admission-wscllm.md`](../runtime-waves-20261013/workspace-admission-wscllm.md)，只细化接口、提交顺序和验证点；未改生产代码，未构建或运行测试。没有改用
`astra-sim-wscllm-LRU` 的迁移/逐出策略。

## 建议定案

将容量账本拆成两个明确所有者：Python `SessionKVCacheManager` 是 rank
容量、模型权重、resident KV 和 request KV reservation 的唯一权威；C++
每 rank 的 `WorkspaceAdmission` 只持有临时 Plan workspace lease。任何一侧都不把
workspace 写入 KV reservation、KV journal 或 HBM/网络 traffic。

workspace-enabled 运行使用显式桥接 schema v2；schema v1 的字节字段、空
`snapshot` 语义和 strict parser 保持不变，只供 workspace admission 关闭的旧
运行。当前 bridge 是严格冻结的 JSON 协议，不是 protobuf bridge：`DecisionBridge.hh`
固定 v1，`ParsedGraphBatch.cc` 拒绝未知 response key；仓内 `.proto` 是 Chakra
ET 格式，不能承载这次决策快照。仅把可选字段塞进 v1 不足以建立准入合同。

v2 的每个 response 必须带齐 Python 在处理该 delivery 后的 rank base-capacity
snapshot。C++ 先校验 snapshot 与 delivery/workspace revision，再完成 GraphBatch
preflight/commit，安装该 snapshot，最后才 issue 新节点。issue 前 C++ 以该 Python
base snapshot 和本地 live workspace ledger 原子取得 lease。lease 释放或 base
capacity 变化时，排入 deferred per-rank issue pass 唤醒留在 free set 的节点。

## 1. JSON schema 和版本门

现状入口是 `StateDelta` (`astra-sim/workload/execution_driven/DecisionMailbox.hh:177`)、
`build_request_json()` (`DecisionBridge.cc:177`)、`ParsedGraphBatch` 的 strict
key set (`ParsedGraphBatch.cc:566-582`) 以及 Python
`_validate_schema()` (`sh_test_mesh/workload/llama2_7b_inference/online/online_scheduler_base.py:517-598`)。
Python 目前会读已知必需字段但允许未知 top-level request key；所以旧 Python
可能忽略新 request field 并继续修改 KV 状态。C++ response parser 则会拒绝任何
未知字段。版本门必须覆盖这两个方向。

workspace-enabled v2 的 JSON 字段建议如下；所有 byte/revision 是非负、精确的
uint64 整数，不接受 bool、负数、float、溢出值或不完整 rank 表。rank 数组按数值
升序、每个 rank 恰好一次，包含零 workspace 的 rank。

```json
{
  "schema_version": 2,
  "delivery_sequence": 42,
  "workspace_state": {
    "protocol_version": 1,
    "ranks": [
      {"rank": 0, "workspace_revision": 7, "active_workspace_bytes": 1048576},
      {"rank": 1, "workspace_revision": 3, "active_workspace_bytes": 0}
    ]
  }
}
```

Response 继续有既有 GraphBatch 字段，并另加必需 snapshot：

```json
{
  "schema_version": 2,
  "source_delivery_sequence": 42,
  "workspace_capacity_snapshot": {
    "protocol_version": 1,
    "ranks": [
      {"rank": 0, "workspace_revision": 7,
       "capacity_bytes": 80000000000, "model_weight_bytes": 12000000000,
       "resident_kv_bytes": 2000000000, "reserved_request_bytes": 1000000000},
      {"rank": 1, "workspace_revision": 3,
       "capacity_bytes": 80000000000, "model_weight_bytes": 12000000000,
       "resident_kv_bytes": 0, "reserved_request_bytes": 0}
    ]
  }
}
```

`workspace_revision` 必须逐 rank 回显 request 中的值；外层
`source_delivery_sequence` 绑定这次 Python KV mutation。Python 不得从之前 response
复用 snapshot。C++ 校验所有 rank 覆盖、revision 回显、容量分量和 checked sum；
任何缺 rank、重复 rank、版本/sequence 不符、`base_used > capacity` 或整数错误都在
preflight 前 fail-closed。

兼容结论：plain optional v1 field 若按“缺失即 0”处理不安全。可以用显式
`workspace_admission=required` 模式把 optional transport extension 做成 fail-closed：
required Python 在任何 manager mutation 前拒绝缺 request field；required C++ 在
任何 commit/issue 前拒绝缺 response snapshot。旧 Python 即使忽略请求扩展，也会因
缺 response snapshot 被 C++ 拦住；旧 C++ 若收到新 response 会因 unknown key 拒绝。
这能避免静默误算，但仍改写当前冻结的 v1 合约，且非对称部署会中止运行。建议用
schema v2 + 独立 workspace extension version；v1 仅 `off`，v2 必须 `required`，
不做缺字段降级。v2 的 request、response、commit ack 和 Python ack 校验都要接受同一
版本；否则旧 `on_commit_ack()` 会按 v1 拒绝 v2 ack。

## 2. Python 端：先同步 C++ hold，再做 KV 准入

### 新接口和接入时序

建议在 `session_kv_manager.py` 增加生产接口：

```python
set_external_workspace_view(revision_by_rank, bytes_by_rank) -> changed_ranks
workspace_capacity_snapshot() -> list[dict]
```

第一接口校验完整 rank 集合、整数域及版本单调性，并保存 transient hold；第二接口
返回 capacity/weight/resident-KV/request-reservation 的精确分量，**不**将 workspace
混进 `NodeHBMState.used_bytes` (`session_kv_manager.py:256-282`) 或现有
`hbm_snapshots()` test/observability 接口 (`:565-570`)。每 rank 保存最近 revision；
新 revision 小于已见值拒绝；相同 revision 必须对应相同 byte 总数；revision 或 byte
总数变化必须能解释为新 hold 状态。revision 达 uint64 上界后拒绝继续递增，不回绕。

接入 `online_scheduler_base.py:on_decision_batch()` (`:342-390`) 的固定管线：

1. `_validate_schema()` 依据运行模式要求 v2 `workspace_state`，严格校验后才允许进入
   sequence gate；v2 `required` 不接受 v1。
2. `_gate_delivery_sequence()` 成功判为新 delivery 后，调用 manager 同步 hold，并在
   completion/arrival/policy 之前生效。重复 delivery 仍走既有幂等门，只返回缓存
   response；同 seq 内容不同继续 fail-closed。新 schema 的 hold bytes/revision 也进入
   现有 delta 深比对/缓存范围。
3. 正常调用 `_settle_completions()`、`_process_arrivals()`、`run_variant_policy()`，
   Python 按更新后的 hold 完成当前 delivery 所有 KV capacity fit。
4. `build_graph_batch()` 之后、`_delivery_reply_cache` 之前（当前 `:370` 后）把
   `workspace_capacity_snapshot` 填入 response。此时 snapshot 已包括本次 policy 的
   KV reservation/release/delete 结果，且会被幂等 reply 原样缓存。

capacity authority 在 Python manager 初始化的 topology/capacity 与权重账本
(`session_kv_manager.py:428-460`)。C++ 只缓存该 snapshot，并持有本地 workspace
lease；不能再从另一份 C++ HBM 配置推导 weights/KV。C++ 对 Python 给出的 base 做
cross-check 后用 `capacity - base_used - active_workspace` 作有界准入。

### 如何加入 hold 而不改变 KV 策略

不要只在 `reserve_request_capacity()` 入口加判断。其核心调用
`ensure_physical_fit()` (`session_kv_manager.py:1269-1423`)，相同 fit 还被
`prepare_history()`、prefill/decode `_grow()`、`move_prefill_to_decode()` 调用
(`:1577,1761,1832`)。把所有 fit 统一改为读取同一 external hold；保留当前无 LRU 仓
的候选资格、次序、delete/recompute 语义，不引入 LRU 仓 `remote_store`/transfer。

把 hold 作为 manager 外部 fit margin，而不是把 `NodeHBMState.remaining_bytes` 改成
带 workspace 的 HBM snapshot。capacity check 可抽成 `_remaining_for_fit(rank) =
capacity - (weights + resident_kv + request_reservation) - external_workspace`，所有
相加先检查 uint64 域/容量边界。`ensure_physical_fit()` 现有 `maximum = capacity -
weights` 用于“空实例仍不够”的分支；要保留判断含义：如果要求超过不含 hold 的空实例
容量，是原有不可行；若只因当前 hold 不够，则返回 admission blocked、保留 KV、等待
下一 epoch，不要把可释放的 workspace 当作永久不可行，也不要为 workspace margin
继续删除 session。若在 hold 下通过删掉现有候选 KV 可以满足，则沿用现有候选顺序删除
到刚好够即止。此修改不改变 KV selector、journal 行或 HBM transfer 统计。

workspace hold 变化需要调用 WSC scheduler 现有 `_note_capacity_change()`
(`online/wsc_llm_online_scheduler.py:1624-1628`) 为关联 instance 增加
`capacity_epoch`。`_try_admit_prefill()` 将 prefill/decode epoch 元组记在请求上
(`:1248-1254`)；更新 epoch 让之前因 hold 阻塞的 KV reservation 在下一次 admit pass
重试。该 hook 只刷新重试版本，不修改
`select_prefill_instance()`、静态 route 或 ready frontier 选择。

## 3. C++：同一 immutable Plan 预备、准入、提交

### 接口和准入位置

真实 Plan summary 的 `hbm_workspace_peak_bytes` 当前由 Mapping 写入；这里建议把读取
封装成 coarse typed API，禁止 issue 热路径直接宽松转 `nlohmann::json` 数值。
workspace 必须是精确 uint64，缺失、负数、float、越界都 fail-closed。

在 `DieModelAdapter` 增加类似接口：

```cpp
struct PreparedCoarsePlan {
    std::shared_ptr<const coarse_die::Plan> plan;
    uint64_t workspace_bytes;
    uint64_t node_id;
    uint64_t generation;
};
PreparedCoarsePlan prepare_coarse_plan(const NodeView&);
void issue_prepared_coarse_plan(const NodeView&, PreparedCoarsePlan&&,
                                WorkspaceToken&&);
```

`Workload::issue_dep_free_nodes()` (`Workload.cc:404-435`) 保持现有 free-node 顺序和
`HardwareResource::is_available()` gate。只在真实 GPU coarse-v2 compute COMP 上，
availability 通过后、`issue()`/`GraphSource::take_node()` 前调用 prepare，然后
`WorkspaceAdmission::try_acquire({rank,node_id,generation}, workspace_bytes)`。不覆盖
timer、CPU、control marker、旧 Die spec 或 replay。workspace 不够时候选保持 free、
不占 COMP slot/stat；沿用 availability gate 的候选遍历，不改 selector、不换 rank。
成功时将同一个 `shared_ptr<const Plan>` 和 move-only token 传入 `issue_die_comp()` /
adapter，禁止重新 compile 出另一 Plan。工作入口为 `Workload.hh:136-139,167-171`、
`Workload.cc:472-477,523-585`；adapter 当前每次都在 `DieModelAdapter.cc:46-56`
compile、`:62-112` submit/register。

per-rank `WorkspaceAdmission` 可作为 `Workload` 成员，随 `Workload`/rank 生命周期；
`main_online.cc` 在当前 `build_state_delta_v1()` 调用处 (`:577-582`) 聚合所有
workload totals 放入 `StateDelta`。一个 token key 至少含
`(rank,node_id,generation,lease_id)`；lease 不可复制，active map 保存 token/plan
record，而不再仅保存 `handle -> node_id`。

准入必须用 checked subtract/add：先验证 `base_used = weights + resident_kv +
reserved_request` 无溢出且不大于 capacity，再检查
`workspace_bytes <= capacity - base_used - active_workspace`。等于剩余容量时成功；
大 1 byte 失败。active 总和累计也须检查溢出。超过空实例 capacity 的单个 Plan 返回
独立 `impossible`，在 take/occupy 前报告明确的 unschedulable/fatal，而不能永久静默
阻塞；普通暂时不足留在 free set，等释放/新 snapshot 重试。revision/lease_id 的
uint64 溢出禁止回绕并 fail-closed。零 workspace 可以使用空 token，但仍通过 Plan 字段
校验。

### StateDelta、response 安装和提交顺序

在 `DecisionMailbox.hh` 加类型化 `WorkspaceRankState{rank,revision,active_bytes}` 和
`StateDelta.workspace_state`；不要把它塞进 `snapshot:{}` 或 `ledger_summary`。保留
`build_state_delta_v1()` 字节行为，另提供 v2 构造/serializer 路径。`main_online.cc`
从全部 per-rank workspace owner 读总量，每个 request 都发送全 ranks；本地 active
lease 变动时按 rank revision 增加一次，不因重复读取递增。

为 response 新增 typed `WorkspaceCapacitySnapshot`；`ParsedGraphBatch` v1 key set
保持原样，v2 parser 要求 snapshot 并校验全部 rank、delivery seq、每 rank revision
echo、整数范围和 `base_used <= capacity`。`GraphBatchCommitter::Context` 增加明确的
`apply_workspace_capacity_snapshot` 回调。`commit_after_preflight()` (`:1612+`，
issue 循环 `:1788-1796`) 的次序应为：

1. parse 和完整 GraphBatch preflight；失败时不动 graph、capacity view 或 workspace
   ledger。
2. 提交本批 graph/assignment/KV action 等现有状态；如果此处失败，仍按现有 fatal
   语义终止，不把新 snapshot 当成成功 batch。
3. 在任一 `issue_rank()` 前一次性安装 response snapshot，校验活动 workspace 与新
   Python base 未超容量。记录 capacity changed ranks。
4. 以 `touched_ranks ∪ capacity_changed_ranks` 排序去重后 issue。snapshot 变化但
   graph 未触及的 rank 也能唤醒既有 free nodes。
5. 原有 commit ack 仍在 commit 成功后发出；v2 ack 带 v2 schema/同 delivery sequence。

单在途 bridge 和同步 tick-end 往返意味着同一 response 到 issue callback 之间没有
并发 Python handler；风险窗口在两个 delivery 之间。每次 C++ 新建/释放 workspace 后
更新 rank revision；下一条 StateDelta 把最新**绝对总量**交给 Python。Python 在新
delivery 的 KV mutation 前应用该视图。例如 capacity=100、Python base=60；C++ 发射
30-byte Plan 后，若下一 epoch 未见这 30-byte hold，Python 可再 reserve 25，base=85，
总需求 115。revision + absolute bytes 让该 epoch 的 KV fit 先扣 hold；response
snapshot 再返回 mutation 后的 base。所有 snapshot 都绑定 delivery sequence，过期
response 不应用。

## 4. lease 生命周期、取消、失败和唤醒

- **正常完成：** adapter callback 先从 active map `extract(handle)`，取得唯一
  active record；release token 并更新 workspace revision，然后做利用率记录，最后
  调 `finish_generic_node()`。必须在 generic finish 排的 deferred issue pass 前归还
  token，容量释放能同 tick 唤醒同 rank 等待节点。未知 handle 或第二次 release 是
  owner invariant 失败，不能再次减总量。
- **live cancel：** 将 active record 从 map 中一次性取出并 release；随后必须走一个
  明确的 graph cancellation terminal/finalize 路径，释放节点/COMP slot 并排 deferred
  issue pass。现 callback 对 `state == "cancelled"` 在
  `DieModelAdapter.cc:67-74` 直接 return，所以新增 lease 不能照搬这个 return 后就
  宣称节点/slot 已释放。shutdown 的 `cancel(..., notify=false)` (`:171-182`) 是另一
  分支：抑制完成回调，抽取 active record 后 cancel/retire 并释放 token，不排正常
  issue retry。
- **submit/登记异常：** prepared 节点在 gate 时 token 先由 move-only RAII 临时持有；
  在 active record 成功登记后才转给 active map。`CoarseDieRuntime::submit_plan()`
  (`CoarseDieRuntime.cc:93-108`) 当前顺序是 Runtime submit → completion map 登记 →
  schedule，adapter 又在返回后 `coarse_active_.emplace()`；需要给 submit API 明确强
  异常保证（失败则无 live handle），或在已返回 handle 后 catch 时 cancel/retire。
  异常无论在哪个阶段都只 release 一次。若要把 transient submit error 变成 retry，
  还必须提供 coarse-only dispatch rollback：撤销 `take_node()`、`occupy()` 和已启动
  statistics，然后 defer 一次 issue pass；当前 `take_node()` (`Workload.cc:447`)
  没有 rollback，dispatch exception 目前走 fatal barrier，不能假设节点自动回到 free
  set。不能重试已被 Runtime 接受但 completion 状态不确定的 submit；这种情况须 cancel/
  retire 成功后才可 rollback，否则 fail-closed。capacity blocked 节点的 retry 则由
  workspace release 或 snapshot change 驱动，不做忙循环。

每个释放来源都走唯一的 `release_once(token)`；active-map `extract`/move ownership
保证正常完成、live cancel、shutdown cancel 不会双减。lease revision 在 acquire 和
release 时前进；请求总是发绝对 bytes，所以 Python 对重复 delivery 可幂等处理。

## 5. 必测边界与可复用 fixture

本轮没有运行测试。实现后可复用：

| 层 | 现有入口 | 新断言 |
|---|---|---|
| C++ ledger | 新增独立 `workspace_admission_test.cc`；不必改现有 CMake 的方案需另行安排测试发现方式 | exact fit 成功、差 1 拒绝且 state 不变、多个 lease 总和、capacity/base/active/workspace 加法和 revision/lease id overflow、单 Plan overcap、重复/未知 release 不二次减账、rank 隔离 |
| 真 Plan→Runtime | `astra-sim/workload/execution_driven/tests/die_execution_lifecycle_test.cc` 已有 coarse-v2 Runtime lifecycle；`die_model_adapter_test.cc` 有 Workload fixture | 用真实 `compile -> prepared Plan -> Runtime callback`；不够时 node 仍 free 且 COMP 未 occupy；成功把相同 Plan 提交；正常完成/cancel notify/shutdown suppress/submit throw 均只释放一次；release 后 deferred retry 被排入；对不确定已提交的 runtime 错误 fail-closed |
| protocol/parser | `decision_mailbox_test.cc`、`parsed_graph_batch_test.cc`、`bridge_loopback_fixture.cc` | v1 feature-off request/response/ack 原字段不变；v2 缺 workspace snapshot、缺 rank、重复/错序 rank、stale delivery/revision、echo 不符、bool/float/negative/>uint64、base 超 capacity 均在 commit 前拒绝；v2 正常 loopback + ack |
| commit 顺序 | `graph_batch_committer_test.cc` 的 `issue_rank` callback fixture | snapshot install 发生在任何 issue callback 前；只 capacity-change 的 untouch rank 也 issue；preflight 失败不更新 view；active hold 与新 base 合计超限 fail-closed |
| manager 实际 fit | `online/test_admission_net_credit.py` 的 `_real_manager()`；`test_admission_eviction_accumulation.py`；manager fit 的 history/grow/move 调用路径 | workspace margin 覆盖 `ensure_physical_fit`、reserve、prepare history、prefill/decode grow、move；exact fit；hold 暂阻不误报 empty-instance infeasible、不做多余 KV delete；release 后恢复准入；workspace 不出现在 `used_bytes`、KV journal 或 HBM transfer |
| WSC 重试/桥接 | `online/test_delivery_idempotency_gate.py`、`online/test_bridge_streaming.py` | 更新 active hold bump 对应 capacity epoch；stale revision 拒绝；相同 seq+payload 返回缓存 snapshot、相同 seq+不同 payload 失败；response snapshot 是 policy KV mutation 后；旧 peer 缺字段在 required mode、任何 manager mutation 前拒绝 |

此方案不能声称新代码/测试已经通过；下一步需要主代理确认 schema v2 与取消/dispatch
rollback 的实现边界，再开始改动。
