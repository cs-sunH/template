# WSCLLM-LRU workspace 预准入需求策略审计

审计日期：2026-10-14。只读检查；未改 tracking 源码、未运行测试。共享树含并行 v2 修改，代码行号以本次检查为准。

## 结论与建议

当前 v2 只同步“已运行的 workspace bytes + revision”，没有同步尚未发射 Plan 的 workspace demand。Python 因此可能按 KV 容量成功准入，返回图后 C++ 才编译真实 Plan，发现 workspace 不够并把 dep-free node 留在 GraphSource。该节点不生成容量事件或 retry alarm；若随后既无 lease release、也无容量 snapshot 更新/外部输入，闭合输入流后仍会停住。base 与 LRU 的 C++ admission 时序相同，base 也没有 prospective Plan demand，故同样有这个限制。

推荐优先验证 **编译器给出的 phase-specific workspace 上界**：对所有合法 Prefill/Decode Plan 形状，按每 rank 编译并取精确 peak 的上界；Python 在 KV admission/growth 前把该值作为“fit-only margin”计入 whole-session LRU fit；C++ 仍对真实 Plan 执行精确 lease acquire，并 fail-closed 检查 `actual_plan_bytes <= declared_bound`。上界必须能被完整枚举或有编译器证明，不能用经验常数、INT4 TOPS 或新的 timing 公式推算。这个方案不把 margin 记成 KV reservation、resident KV、active workspace 或 remote-pool debit。

不过，上界本身不足以证明安全：当 margin 通过 LRU 冷会话逐出获得时，当前 GraphBatchBuilder 将这些 remote-store 支链 fork 后与计算并行；C++ 的 RemoteMem 资源槽不阻塞 GPU COMP。必须证明被逐出 KV 在 store 发射时已可供 workspace 使用，或为确实依赖这些释放字节的目标 Plan 增加 store-tail → Plan 的 rank-local 依赖/等效 admission gate。未验证这条物理释放排序前，静态 margin 只能证明 Python 账本不超订，不能证明执行中 Plan 的 workspace 与仍在 store 的数据无冲突。

若无法为 decode 拼批等合法形状得到足够紧且可证明的静态上界，安全替代是 **两阶段 plan-probe / admission / commit**：先在任何 manager 变更和 GraphBatch commit 前得到实际目标 Plan demand，再让 LRU fit 使用该 demand，提交同一任务/计划身份。仅在 Plan 已阻塞后发送 `WORKSPACE_DEMAND`、然后逐出 KV，不构成等价安全方案：这时当前 candidate 的 KV 预约/驻留已经可能是 protected/non-reclaimable；manager 不支持跨整个 scheduler decision 的事务回滚，GraphBatch ack 也不是 rollback 边界。简单 post-block handshake 可用于诊断/唤醒或显式报不可行，不能保证修复所有阻塞。

## 当前时序与失活边界

1. C++ issue 时先检查 GPU 资源可用，再从节点 `die_task_json` 编译 immutable Plan、取得精确 `hbm_workspace_peak_bytes` 并尝试 workspace lease。`WorkspaceAdmission::Blocked` 保留 node 在 GraphSource free set，既不 acquire lease 也不排 timer/容量事件；注释指定只有 lease release 或 changed capacity snapshot 再触发 issue（LRU `Workload.cc:399–450`、`DieModelAdapter.cc:92–125`、`WorkspaceAdmission.hh:151–203`）。GPU COMP 是单槽（`HardwareResource.cc:278` / `available_class`），remote MEM 是独立且不因在途数阻塞的槽（`HardwareResource.hh:40–75`、`HardwareResource.cc:31–60,223–260`）。所以不能靠已阻塞 node 自己重试，也不能假设 remote-store 自动串行化 GPU Plan。
2. C++ 的 v2 delivery 含每 rank 的 absolute `workspace_revision/active_workspace_bytes`（LRU `main_online.cc:693–719`）；Python 基类严格解析 full-rank active-state，而 LRU policy 前安装到 manager，policy/build GraphBatch 后才返回 KV-capacity snapshot（LRU `online_scheduler_base.py:399–449,584–634`、`wsc_llm_online_scheduler.py:554–570`）。这份 state 不含 prospective Plan demand。
3. LRU `_admit_pass` 先遍历 ready frontier，再 `_try_admit_prefill`；它先在 Decode 目标预约终态 KV，再对 Prefill 目标做 `prepare_history`、`grow_prefill`，成功后发射 Prefill GraphBatch（`wsc_llm_online_scheduler.py:1383–1459,1554–1635,1748+`）。当前 reservation 的 bytes 只代表最终 KV，C++ 真实 workspace 在 task 已出现在图内后才编译。
4.  manager fit 已把 active workspace hold 纳入现有 required-KV fit，并保护 active/protected residency 和 reserved KV；冷候选按 whole-session LRU 顺序逐出到 remote（`session_kv_manager.py:2289–2507`）。它没有 future plan margin 参数。reservation、history prep、growth 是多次独立 manager public mutation；journal decorator 的 rollback 只覆盖单个公开 mutation 中的异常，正常 `admitted=False` 是该次调用的正常 commit。后一步 blocked 时先前成功调用产生的真实逐出及已发射的 graph actions 不会因 scheduler 返回 false 而整体撤销（`session_kv_manager.py:131–190,2509+`；`wsc_llm_online_scheduler.py:1453–1590`）。
5. GraphBatch 在 C++ 先做完整 preflight，后执行 Phase B graph/KV commit；workspace capacity snapshot 在图/KV state commit 后安装，随后对 touched/changed ranks 立刻跑 issue pass（`GraphBatchCommitter.cc:363–408,1978–2032`）。Python `on_commit_ack` 仅 finalize provisional ledger；生产失败提交不发 negative ack，不能拿 ack 做申请回滚点（`online_scheduler_base.py:1156–1228`）。
6. LRU eviction transfer 的 store tail 目前用于同一 session 后续 remote restore 的 store→restore 排序。常规 history/prefill eviction branch 被明确设计为 fork 后不 join（`online/graph_batch_builder.py:920–940,1024–1105`）；WorkspaceAdmission 在收到 eviction 后更新的 Python KV snapshot 时看不到这些 store 是否完成。若 workspace margin 是靠这些 victim bytes 腾出的，必须另测/补目标 Plan 的释放排序，不能把同会话 store→restore 前递边误当成通用 HBM 释放 barrier。

## 方案对比

| 方案 | 可安全做什么 | 关键条件与边界 | 判断 |
|---|---|---|---|
| 编译器导出的 Prefill/Decode 上界（推荐先验证） | 在 manager 准入前保留未来 phase 的 workspace slack；超冷候选仍按 whole-session LRU 逐出并产生实际 remote-store；真实 Plan 仍走精确 acquire | 对每种 legal phase/task shape 得到 per-rank 严格上界；确定实际 Plan 并发上限；把上界 identity 与 hardware、model、compiler/task-shape set 绑定；校验 evict-store bytes 物理释放先于依赖它的 Plan | 最小运行时协议变化；若合法形状是有限的且上界不太宽，最容易落地。没有证明上界或 store 排序时不可宣称安全 |
| 精确 blocked-demand 的 post-block event | 把 `(rank,node,generation,actual_bytes,workspace_revision)` 作为单次阻塞事实送回 Python；可诊断、去重、避免 busy polling | 发生在 Python KV admission / GraphBatch commit 后。可能由当前 Plan 自己的 protected/reserved KV 造成不可回收缺口；manager 没有 scheduler 决策级 rollback。若 Python 逐冷会话，store 完成前也不能仅因逻辑 snapshot 变化就无条件重试 Plan | 不能单独解决所有 deadlock；只适合作为诊断、终止原因或有证明可回收时的带 store-tail gate 的 retry。新增 reason/field 需新协议版本，不能静默扩充严格 v2 |
| 两阶段 exact plan-probe → admission → GraphBatch commit | 在 manager 改变之前拿到本次实际 Plan demand；fit 用 exact bytes 做 LRU；不能 fit 时不生成/提交 task，避免当前 request 的 protected/reserved KV 把自己困死 | 必须在 proposal 阶段冻结相同的队列/Decode train membership、phase shapes、task spec/hash 与 capacity epoch；对多阶段 Prefill+未来 Decode 仍需同时知道 future Decode Plan，或额外有 Decode bound。需要新的请求/响应/幂等状态和 abort 生命周期。若 probe 后 LRU victim store 才能腾空间，候选 task 应等待 store-tail release 后再 commit/issue | 正确且精确，但改动面最大。是缺少可证明静态上界时的可靠 fallback；当前 ack/manager transaction 机制不能直接拼出这个流程 |

## 静态上界的具体合同

每 rank `r` 定义 `H_r = capacity_bytes - model_weight_bytes`、`R_r = resident_kv_bytes + reserved_request_bytes`、`A_r = active_workspace_bytes`，以及按目标 phase `p` 的 `E[r,p]`（所有可发射该 phase Plan 的严格 workspace 峰值上界）。拟准入的 KV 增量为 `K_r` 时，满足 `R_r + K_r + A_r + E[r,p] <= H_r` 后才能接受；跨阶段的同一准入要同时满足每个目标实例的相关约束。`E` 只是调用 fit 的额外 margin，永远不进 Python reserved-kv 字段，也不远程池扣账。C++ 编译真实 Plan 后仍按 `A_r + actual_plan_bytes <= H_r - R_r` 精确 acquire，且 `actual_plan_bytes > E[r,p]` 立即 fail-closed，表示 envelope/config 不匹配。

LRU 调用必须显式分相位：

- `reserve_request_capacity` 为 future Decode final KV 预留 Decode margin，否则 Prefill admission 虽成功，最终 Decode reservation 仍可能把计划自身堵住。
- `prepare_history` / `grow_prefill` 为 Prefill 目标用 Prefill margin；任何与 history restore 相关的 remote-pool debit 仍只在原 manager restore 成功路径发生。
- `grow_decode` / reservation extension 保留 Decode margin。Decode final KV reservation 若覆盖整个终态容量，需证明之后 resident↔reserved 换位不会二次加算同一份 KV 或丢失 margin。
- 每个后续准入/增长仍使用 margin；只在首次 reserve 时检查一次会让下一 request 消耗已承诺的 workspace slack。
- 活跃 C++ workspace 保持由 absolute v2 hold 单独计入。当前生产 GPU COMP 单槽，未来 Plan bound 按“当前活动 lease + 下一计划”保守 fit；若硬件允许并发 COMP，必须把 workspace 并发 envelope 相加或改用实际活动集合合同，不能仍取单个 max。

要产出可信 `E[r,p]`，先列尽所有生成器形状：Prefill 的 chunk/span、history context、layer range、TP rank；Decode 的单次/拼批 train、成员参与 span、first-step/remainder 形态、rank-local task。实际任务由 `online/graph_batch_builder.py` 配合 `online/die_model_adapter.py` 创建，当前 C++ 只在 `Workload::issue_dep_free_nodes` 见到 committed NodeView 后调用 compiler。Decode 拼批组合可能很多；只有证明最大合法输入一定覆盖最大 peak，或对全部合法 shape/组合枚举 compile 后取 max，才是安全上界。不能因当前 fixture 覆盖了 MHA 4-head 就宣称所有 GQA/shape 已界定。

建议上界作为独立、带 hash/版本的 immutable startup/config artifact，在 C++ 与 Python service 两端同源加载；不要把它伪装成 v2 的 `active_workspace_bytes`，也不要在原 v2 exact-object parser 中偷偷加字段。若需在线下发，应明确定义新的协议/配置版本和 fail-closed 协商。

## 精确 handshake 必须满足的先决条件

能修复 admitted-but-blocked 的不是“blocked 后多发一个 JSON event”，而是发生在 `_try_admit_prefill` 调用 `reserve_request_capacity` 之前的 proposal 阶段。最低合同应包括：

1. proposal id + 输入 delivery/capacity epoch + rank/phase + task-spec digest + 精确需求 bytes；Decode proposal 冻结 train membership 与首步拆分形态，避免下一次重跑策略生成另一个 task。
2. C++ compile-only 返回 workspace demand，不取 lease、不消费 node、不产生 timer/资源事件；或 Python持有等价的纯 Plan-demand artifact。Plan spec/hash 必须与最终 GraphBatch node 完全一致。
3. Python 仅在 proposal 需求已知时调用 manager fit。无法 admission 且无冷候选可回收时，未有 manager/graph mutation；有候选时 whole-session store 依现有远端池与决策账本实际发出。若需要等待 store 物理释放，proposal 必须保持未提交并仅在真实 store-tail completion 唤醒，不得在同一个 unchanged snapshot 上自旋。
4. C++ 再验证原 capacity/revision epoch、proposal identity、task digest，原子提交同一 Plan node/edges，并在 commit 后正常 acquire exact workspace lease。stale proposal 作废后不能套用旧 demand。
5. 需支持 failed proposal/timeout/EOF 的明确终态；不能复用目前只接成功 GraphBatch 的 commit ack 作为 rollback。必须覆盖 delivery replay 与重复 proposal 的幂等行为。

目前 Python policy 入口一次调用内完成 arrival/completion、manager mutations、policy、build GraphBatch（`online_scheduler_base.py:399–449`），C++ bridge 再同步返回并在同 tick commit；要在 policy 前插入 probe，需纯规划或 proposal/shadow state，而不是先跑当前 `_try_admit_prefill` 再假装可回滚。每个 manager public method journal 化的原子性不等于这一串 scheduler+builder+C++ commit 的跨组件事务。

## 最小安全实施顺序与测试清单

1. 先确认每 rank 的实际 coarse Plan workspace 是否有有限 phase 上界；生成所有 legal Prefill/Decode Shape 清单，利用官方 C++ compiler 批量编译并保存每个 plan 和 peak。检验任务 shape 的 hash、compiler/硬件/model/profile 标识和逐 rank最大值。若无法枚举/证明，决定是否投入 two-phase probe；在决策前不要把 post-block event当修复。
2. 为 manager 增加 fit-only phase margin 参数，复用现有 per-rank workspace view、rank-wise checked arithmetic、whole-session LRU candidate/ordering；保留 protected/active/reserved 的不可回收规则、remote pool/restore原合同。为 future Decode reservation、Prefill history/growth、Decode growth 分别接通。测试证明拒绝/阻塞不创建 fabricated reservation、不错误 debit pool，冷 victim 仍只按既有 LRU 次序整会话逐出。
3. 对被 workspace margin 选择的 victim，确认并补齐 `REMOTE_STORE` 每个 HBM shard 的 tail 到目标 phase Plan 首个 coarse COMP 的排序；Store 同会话→Restore 的既有 dependency 与 pool 写读顺序继续不变。测试 store/Plan 同 rank及多rank、store仍在途、capacity snapshot 同批安装时，Plan 不会提前 acquire；store completion 到达后只 retry 一次。
4. C++ 每个实际编译 Plan 对 phase 上界做 fail-closed assertion。当前 acquire 的 Blocked 不生 synthetic busy retry；外部容量改变/lease release/真实 store completion是唤醒条件。上界超出时必须给 node/rank/phase/task digest诊断，不静默删margin。
5. liveness 测试覆盖：closed input + 无 active lease + 多 ready Plan；足额 margin 使任务由 C++ 通过真实 Plan issue；margin 不足时 Python 在 GraphBatch 前按 KV fit阻塞；protected/reserved情形没有候选伪逐出；lease release/eviction-store completion 触发一轮重试，无定时循环。再测 capacity_epoch/manager reservation与C++ workspace revision同步。
6. 协议回归：v1 off/旧 GraphBatch字节不变；v2 active workspace full-rank revision echo 与 strict schema不变；如新增 envelope 在线字段或 demand/proposal event，显式 bump protocol 并测 unknown/missing/stale/replay/error。不能将 `E` 加到 `active_workspace_bytes`、`resident_kv_bytes` 或 `reserved_request_bytes` 中。
7. 在 base 和 LRU 都跑 deadlock fixture。base 也需要计划上界/精确需求之一；不要因 Python LRU helper已支持 current active-workspace hold，误以为 base 能看到 post-commit actual Plan demand。LRU 的 whole-session ordering/remote pool/restore tests仍需跑，以确保额外 margin 仅改变合适时机与容量结果，不改 victim policy。

## 审计局限

没有执行 fixture、编译 C++、跑 Python tests。已确认 `HardwareResource` 允许 remote MEM 和 GPU COMP 各用独立槽并发，且 GraphBatchBuilder 对普通 eviction branch 明确“不 join”；但本审计没有测量 C++ coarse HBM model 在 remote-store source buffer完成前是否仍把它算作真实物理占用。实施前需由 runtime 负责方以带在途 store 的实际 test/trace 核实此点；在证据出来前按“store 尚未完成则不能依赖其 bytes 给新 workspace”处理。
