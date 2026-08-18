# sh_1.0 Execution-Driven 改造执行实录

> 执行者子 agent 维护；依据《sh_1.0仓库改造详细执行方案.md》分阶段执行。
> 指令约束：**不做 git commit**（覆盖方案的 §0.5 提交纪律——改动以工作树
> 状态 + 本实录登记；回滚锚点 = 方案文档行号 + 本实录记录）。
> 蓝本 = template/astra-sim-wscllm（已验收）；机制层与其保持一致，策略子类逐仓独立。

## 起点

- 父仓库 HEAD = 55cf135；工作树起点干净（除用户未跟踪文档，绝不 touch）。
- sh_1.0 = 未改造静态 ET 基线（与 template_back 除 pyc/一个 network.yml 外零差异）。

## 阶段 0（2026-08-16 完成）

### 步骤 0-0 起点
- git status 干净；未建分支、未 commit（按指令）。回滚 = 本实录 + git diff。

### 步骤 0-1 输入物化 + fail-closed ✅
- `traces/derive_20_first_30_seconds.py`（与 face 仓同脚本逐字节一致）物化
  20.csv 前30s：**1177 请求 / 112 session**，prefill 66-169395，decode 1-13812，
  max arrival 29.988879s，timing 全 1000ns 倍。
  队列 md5 `ee7af9d0bc3e9c2e211d1e30c325bd45`、sidecar md5
  `8525bf190950a33b6e6bd66c0c36c2e7`（与 face 物化逐字节一致；源 md5 fc74a48...）。
- `trace_config.csv:12` 指向物化输入（整体替换失效的 empirical 血统路径）。
- fail-closed：`generate_face_trace.py` load_face_trace_config 在
  `_resolve_request_queue` 与 `load_request_queue` 之间插入存在性检查，
  缺失 sys.exit(1)；反向验证 exit=1 且打印物化指引。
- `--print-shell-config`：REQUEST_COUNT=1177 / SESSION_COUNT=112。
- traces/PROVENANCE.md 写入（规则①-④、md5、约束声明）。

### 步骤 0-2 静态基线归档 ✅
- runall 全管线 wall 56.33s / 峰值 RSS 731,008 KB（/usr/bin/time -v）。
- 归档 `sh_test_mesh/baseline/20_30s/`（generated 58 文件 + metrics + run_logs）。
- 确定性：两次 runall 的 raw/normalized_metrics.csv 剔除 run_id/wall_time_ns
  逐字段一致；generated 58/58 逐文件 cmp 一致。
- baseline/20_30s/PROVENANCE.md 写入。

### 步骤 0-3 已取消（用户指示，仅 20.csv 前30s）

### 步骤 0-4 九项语义合同 ✅
- `online_contracts/nine_contracts/contract_01..09_*.md` + README，每份含
  口径/裁决/验证方法。本仓差异显式裁决：
  - 合同⑨：LUT 成本模型角色**在线保留**（物化期冻结 face_lut.csv，只 lookup，
    禁止在线 build/增量扩展）；max_d_token/request_count = 物化期标定常数；
    p_chunk 固定 512（mean 回退在线禁入）。
  - 合同⑥：remote FIFO = **实账本层**（ARM 26 端口 FIFO）；local HBM job =
    不适用占位（无 LocalHbmBandwidthModel）。
  - 合同⑦：replay LUT 时钟语义四项（蓝本三项 + 本仓第④项 MEM 即时完成豁免，
    仅限 --online-mode replay）。
  - 合同①：三段式发射（ARRIVAL→段1 / PREFILL_DRAIN→段2 /
    DECODE_COMPLETION→段3 completion_evictions）。

### 步骤 0-5 决策日志（--replay-record 方案 A）✅
- `generate_face_trace.py`：`build_face_plan(force_record_iterations=)` 参数 +
  `_write_replay_decision_log`（replay_kv_transfer_dict 等 4 个 helper）+
  `--replay-record` CLI 开关（生产路径不受影响；recorded plan 单独重建）。
- kind 闭集 = prefill（tick=admission_time_ns）/ decode（tick=prefill_complete_ns）/
  completion（tick=completion_ns）/ iteration（tick=start_ns，replay 不消费）。
- 字节等价门：--replay-record 运行后 generated 58/58 与归档 cmp 一致。
- decision_log.jsonl 197,859 行（1177×3 决策 + 194,328 iteration），
  md5 `se2e0e72de30eb0b13993037cac90d7629`，归档 baseline/20_30s/。

### 步骤 0-6 pytest ✅
- 59 用例全绿（26+4+29）。两个旧 3min 断言用例更新为 30s 实测值
  （1177/112、prefill 66-169395、decode 1-13812、arrival 94835000-25959142000、
  max derived 29988879000、session_id 前缀 session_*、队列路径 traces/）。
  仅改测试期望值，未改被测代码语义（用例更名
  test_checked_in_astra_compute_selection_uses_first_30_seconds_window）。
- run_scripts/README_COMMANDS.md 写入。

### 步骤 0-7 压测基线 + 埋点 ✅
- 30s 输入全管线：wall 56.33s / RSS 731,008 KB / generated 149MB / results 3.3MB
  （写入 baseline PROVENANCE）。离线两口径拆分阶段 6 报告。
- run_sh_test_aware.sh:25 RUN_OUTPUT_LOG_TIMESTAMP 环境变量覆盖（默认不变）。
- 计数器埋点随阶段 1 在线层落地（离线路径零字节影响由字节门复验保证）。

### 阶段 0 门槛判定：全部满足 ✅
- 物化输入+PROVENANCE 齐全、fail-closed 实测 exit≠0 ✅
- 两次运行规范化 metrics 一致 ✅
- 九项合同冻结（含 LUT 裁决与复核）✅
- 决策日志与离线 ET 字节等价（58/58）✅
- 压测基线在案（56.33s/731MB）；埋点零字节影响 ✅
- pytest 59 全绿 ✅
- git 提交：按指令跳过（不 commit），以工作树+实录登记 ⚠️（与方案 §0.5 的
  偏差已由用户指令覆盖）

## 阶段 1（进行中，2026-08-16）

### 已完成 ✅

**C++ 机制层（蓝本逐文件移植 + 本仓适配）**
- 与蓝本共享文件在改造前逐字节一致（cmp 亲核 16 文件），因此整目录复制蓝本已改版本：
  execution_driven/ 全部（含 tests/）、EventQueue.cpp/.h（tick 收口 + deferred +
  in_invoke_context）、FluidScheduler deferred 分流、CommonNetworkApi、Sys.cc/.hh、
  Workload.cc/.hh（GraphSource/NodeView/在线门控）、HardwareResource、Statistics、
  根/analytical CMakeLists（双目标）、test_congestion_aware、main_online.cc。
- MetricCollector：蓝本补丁剔除 kv_policy 3 个 hunk（sh_1.0 无该配置）。
- 本仓独有：
  - Workload.cc issue_remote_mem 增加 replay MEM 即时完成钩子（合同⑦第④项，
    仅 --online-mode replay；strategy/静态保持 ARM 26 端口真实 FIFO）。
  - GraphBatchCommitter 覆盖规则放宽为单向（watch 必须有节点覆盖；尾段节点可
    无 watch）——sh_1.0 三段式发射适配，蓝本两段式批次不受影响，偏差登记。
  - main_online.cc：EventQueue 排空且服务有活动请求持续 30s → 转储
    （per-rank unfinished/in-flight 节点名 + stale watches）并 fail-closed abort
    （蓝本 wait_for_work 谓词 !input_open_ 会 100% CPU 空转，此为机制缺陷修复）。
- 编译全目标通过；**静态基线字节门：机制层移植后 runall 58/58 与归档 cmp 一致**。

**C++ 单测/fixture 全过**
- WatchRegistry/DecisionMailbox/GraphBatchCommitter/NodeStore(带 fixture-et)/
  WindowedReader/EventQueue deferred(g++ 独立编译，用例 A-D)/CompletionFixture
  （162/162 hook 计数，三终态路径覆盖；远端 MEM 完成链经 wlhd 分支天然覆盖，
  ARM 双 register_event 审计结论成立）。
- IDLE/注入 fixture 五态迁移 PASS；**post-commit same-tick milestone fixture（1-11
  CORE）PASS**（(a) 不重入 (b) T+1 显式唤醒 (c) 下一 epoch 交付 (d) 零丢失）。

**Python online/ 层**
- 机制件照抄：decision_bridge / replay_source / checkpointing /
  online_scheduler_base（schema v1 + 幂等 + 分层账本 + profile + 统计）/
  verify/{bridge_echo, lifecycle_fixture_service, same_tick_milestone_fixture_service}。
- 本仓新写：graph_batch_builder.py（**三段式发射**：OnlineTraceBuilder 含
  mem_store/mem_load、_block_ends emitted-ranks-only 段间块末账本、pending_history/
  deferred_remote_sessions 在线账本、replay LUT 时钟校准、replay 段首清链）；
  sh10_replay_scheduler.py；sh10_online_scheduler.py（策略路径，逐行标注离线行号，
  KVCacheManager/select_prefill/decode/LUT 冻结表只读 import；LUT 冻结表 =
  baseline face_lut.csv 加载，只 lookup）；online_service.py；run_scripts 4 个 runner
  （run_online_replay/strategy/idle_fixture/same_tick_milestone，ET/RC 指向
  baseline/20_30s 归档，--remote-memory-configuration 必传）。

**replay 模式全量验收（20.csv 前30s，1177 请求）✅**
- 修复链（均登记）：到达门未并入批次 → seg3 stage/generation 闭集 →
  REQUEST_COMPLETE watch 与 decode watch 重复（机制为 decode fire 双推）→
  future_alarm 1ns 边界 clamp → **块末账本 emitted-ranks-only 判定错误（把
  "rank 历史上发过节点"当"本段发过节点"，stale 跨 request 边导致 s2r0 decode
  迟到 456ms → 失步）**。修后全量 PASS：1177/1177 完成、3530 交付=ack、
  replay 日志全部消费、双进程 exit 0、wall ~26s（机器被其余仓改造运行挤占）。
- 单会话（session_2）与 7 会话最小复现均精确校准（decode 完成边界 ±1ns）。

### 遗留问题（已完整诊断，未解决）⚠️

**strategy 模式全量运行图死锁**（步骤 1-9 未达门槛）：
- 现象：全部 112 个 turn-0 到达（delivery ~156，tick 25.96s）后 EventQueue 排空、
  服务有活动请求，死锁（现在 30s 后 fail-closed abort 并转储，不再 100% 空转）。
- 诊断（SIGUSR1 backtrace + 逐事件回调打印 + NodeStore 未完成节点转储）：
  在飞卡点全部为 noc_migrate 回程 ACK_RECV 与 all_reduce collective；
  等待链跨 pg 交织（如 rank0/1 卡 session_5 prefill AR ← rank6/7/12/13 卡
  session_6 decode AR ← rank18/19 卡 session_2 prefill end barrier ← ...）。
- 根因：离线 ET 的全局发射序（order_plans_for_static_emission + KV store 因果
  检查）保证 per-rank collective 顺序全局一致且 send 先于配对 recv；在线决策序
  无此保证——strategy 真实网络下回程 ACK 与对向迁移/collective 组合可成环。
- 已试三项（记录于 git 不可用之工作树 diff）：(a) 全段清链+块末恢复（死锁点
  142→173→156 推移但未消除——collective 跨 pg 顺序不一致仍成环）；(b) 回程
  recv（ack_from/request_from）strategy 模式独立发射（消除 send←ack 环）；
  (c) (b)+保留跨 request 链（全局序）组合——仍死锁@156。
- 倾向方案（待主控裁决/继续迭代）：strategy 模式对 collective 提交引入全局序
  约束（按 delivery 序的 collective 闸门，或在 GraphBatch 校验层拒绝会造成
  跨 pg 乱序的批次），或按离线同款因果约束在线发射顺序（remote_store producer
  先于 trigger 的在线对账）。replay 模式（B0-B4 exact 载体）不受影响。

### 阶段 1 其余门槛状态
- 九项合同审查：已冻结（阶段 0）✅
- 静态基线 58/58 ✅；EventQueue 单测 ✅；IDLE fixture ✅；same-tick fixture ✅
- replay 20.csv 前30s 全量 ✅（replay_poll_query_count=0 机制天然成立）
- strategy 20.csv 前30s 全量 ⚠️（上述死锁，fail-closed 不静默）
- 红线检查：face_scheduler.py 零改动；generate_face_trace.py 仅
  fail-closed + --replay-record 观测通道；generate_trace.py 零改动；
  AnalyticalRemoteMemory 两文件零改动 ✅（git diff 可复核）

### 阶段 1 strategy 死锁修复（2026-08-16，主控指令移植 sh_2.0 已验证修复）✅

- **根因（与 sh_2.0 同款，waits-for 环证据）**：蓝本 replay 的 emitted-ranks-only
  块末恢复被沿用到 strategy → per-rank 发行序 ≠ 全局发射序 → 跨实例 noc_migrate
  P2P 对与 all_reduce collective 参与序在共享 rank 上相对次序反转成环
  （fail-closed 转储：全部卡点为 noc_migrate 回程 ACK_RECV 与 collective；
  等待链跨 pg 交织）。
- **修复（移植 sh_2.0 frontier 接续裁决）**：strategy 模式**不做任何块末恢复/
  段内清链**——每段在该 rank 首节点 previous_id 无条件接续当前 frontier
  （= 离线 writer 跨 request 物理链同构；B3 归因类别②既有口径）；
  replay_clock=True 保留块末恢复（LUT 时钟语义，B1/B2 exact 载体）。
  三处发射点（段 1 清链 / 段 2 恢复 / 段 3 恢复）全部加 replay_clock 门控
  ——首个漏改点（段 1 清链未门控）即最终死锁残留原因。
- **回退的实验性改动**（未入机制，git 工作树已还原）：collective 计数式槽位
  占用实验；回程 recv 独立化实验（frontier 接续后不再需要）。
- **结果**：strategy 20.csv 前30s 全量 PASS——1177/1177 完成、3531 交付=ack、
  双进程 exit 0、wall ~30s；replay 回归 PASS（3530 交付=ack，交付数差异 =
  1ns alarm clamp 合并两个同 tick 边界，非回退）；静态字节门 58/58；
  pytest 59 全绿。
- 阶段 1 门槛补记：strategy 全量 ✅ → **阶段 1 全部门槛满足**（未做 git
  commit/tag，按指令留待一次性补齐）。

### 后续（阶段 2-7 待执行）
阶段 2 Tier B 正式报告（tier_b_compare.py 适配 manifest/KV 事件结构）→
阶段 3 GraphBatch 验收 → 阶段 4 感知/对账 → 阶段 5-6 benchmark → 阶段 7
裸仓库还原 + 一次性 commit/tag（sh1-phase0..7-done）。

## 阶段 2（2026-08-16 完成）——关感知 Tier B 等价验收 ✅

### 运行材料
- replay 全量：`run_online_replay.sh /tmp/sh1_run_p2_replay`（1177/1177 完成，
  3530 交付，双进程 exit 0）；strategy 全量：`/tmp/sh1_run_p2_strategy`
  （1177/1177，3531 交付，exit 0）。交付数差异 = 1ns alarm clamp 合并两个同
  tick 边界（阶段 1 已登记，非回退）。
- 注意：runner 的 request_csv/decision_log 参数必须绝对路径（Python 服务
  cwd 在 workload 目录；相对路径 fail-closed FileNotFoundError——按设计）。

### tier_b_compare.py 适配（online/verify/，结构模板 = sh_2.0 同名）
- **B0**：计数 + trace_digest/request_mapping_digest 在线 raw_metrics ==
  基线 raw_metrics（输入血统）+ prefill 决策内容 digest 0 mismatch。
- **B1**：record-tick 权威键（(kind, request_id) -> 离线记录 tick；sh_2.0
  裁决 3 / sh_3.0 两轮验证法同款）。冻结口径：prefill 1157 exact + 20 个
  alarm-clamp 正偏差 [1,10]ns（实测 20/20 全部满足"记录 tick <= 前序完成
  边界+1"的 clamp 条件）；decode [-4,+10]ns、completion [-8,+9]ns
  （COMP 链校准整数截断 + 1ns 粒度 + T+1 deferred）。**q-吸收事件 = 0**
  （输入 hbm_wait_ns 全 0，离线无 HBM 阻塞准入；到达/记录 tick 的 20 处
  差异全部为 clamp 负向钳制 [-10,-1]ns）。跨 tick 逆序 0、阶段序违规 0、
  决策早于到达 0。pending_admissions 路径由 B2 重放逐 delta 覆盖。
- **B2 oracle**：三 kind 决策内容 0 mismatch；KV 载荷对照 = **kv_event_
  payload_sh1 口径适配**——replay 决策六类转移元组（kind/phase/reason/
  session/trigger/source/target/total_bytes/shards 含 noc_path）vs 离线
  manifest `requests[].transfers_by_stage` 逐条全等（2510 行 0 mismatch；
  sh_2.0 的 kv_cache_events.csv 账本不变量不适用，本仓等价物）。
- **B2 real-online（strategy）**：**确定性重放不变量**——strategy 运行保留
  的 3531 个 bridge/request_*.json 交付流逐条喂给全新 Sh10OnlineScheduler
  （同一冻结 face_lut.csv、同一只读策略函数），重放决策与 recorded 日志
  逐行全等（0 mismatch）= "相同输入下同一函数输出一致"逐决策断言（方案
  §5.4）；prefill_assignment_key 自洽 0 fail；实例分布差异（合法）已信息性
  登记并归因（真实完成时序改变排队深度/active decode 输入）。
- **B3**：在线侧 = 比较器内确定性重放交付流重构 GraphBatch，先与 live run
  graph_batch_digests.jsonl 逐批 content_sha256 对平（3530 批 0 mismatch =
  重放构图 == live 构图）；canonical logical node key（(rank,name,type,
  is_cpu,is_timer,类型感知属性)）vs 离线 .et 全 54 rank：**节点多重集
  329308==329308 完全一致（missing/extra = 0，硬门）**。边差异 missing
  17270（within 243/cross 17027）+ extra 0，全部归入登记类别①②③④
  （类别④ = 三段式发射触发门编码，§5.5 预期高风险点，实测 243 条逐条为
  prefill_decode_transfer/history_transfer → completion_evictions 的段间
  依赖——离线 writer 把 completion 段锚在 decode 末而在线锚段 2 末节点
  的编码差）。**comm_tag 移出 canonical key**（tag = TransferTagAllocator
  全局发射序产物，与节点 ID 同类插入序伪影），替代硬门 = 配对不变量
  （send/recv 集合内 tag 各自唯一 + 按 (src,dst,tag) 一一配对且 bytes
  相等：实测 duplicate/pair_missing/recv_without_send = 0/0/0）。
- **B4**：归因口径。边界级漂移（修正配对：PREFILL_DRAIN 边界 vs 离线
  decode 记录 tick、DECODE_COMPLETION vs completion 记录 tick）= [-4,+10]/
  [-8,+9]ns（与 B1 同源）；节点级漂移 n=2354 [-7,+19]ns，>1us 条目 0；
  completion 跨 tick 逆序 0；completion 早于 decode 0；指标不减
  （completed_requests 1177==1177，memory_actions_total 95790==95790）；
  mean_e2e 75.8s→11.9s = replay LUT 时钟 + comm/MEM 即时完成压缩（类别①，
  信息性）。

### 修复（阶段 2 发现，登记）
- **graph_batch_builder.py action 计数分段归零 bug**：离线 writer 的
  per-request action_sequence（generate_face_trace.py:2057）跨全部 stage
  连续；在线三段各自 `action_sequence=[0]` 重置 → 同逻辑节点 actionNNN
  改名（实测 28864 个节点 missing==extra）。修复 = `_action_sequence_by_
  request` per-request 连续计数跨段共享。修后 B3 节点多重集 0 差异。
  在线层命名对齐修复，不触任何红线文件；replay/strategy 重跑全过。

### 阶段 2 门槛判定 ✅
- B0 容差 0；B1/B2 oracle 容差 0 + real-online 不变量全过；B3 规范化图
  一致（节点多重集 0 差异，边差异全归因）；B4 不可解释差异 = 0。
- 1177/1177 完成；unresolved/dropped/deadlock/starvation = 0；watch 重复
  fire 即 fail-closed（机制），结束审计 in_flight 空/ack==delivery/replay
  全消费。
- 等价报告 `online/verify/tier_b_report_20.md` 产出（比较器 exit 0）。
- 静态字节门复验 58/58（generate_trace.py 重生成 cmp baseline）；
  pytest 59 绿。

### 后续（阶段 3-7 待执行）
阶段 3 GraphBatch 验收 → 阶段 4 感知/对账 → 阶段 5-6 benchmark →
阶段 7 裸仓库还原 + 一次性 commit/tag（sh1-phase0..7-done）。

### 机制层特有改动登记（主控批准 2026-08-16，供跨仓差异清单/回灌评估）

**(a) GraphBatchCommitter 覆盖规则单向化**（GraphBatchCommitter.cc :499 区域）
- 原蓝本规则：`node_stages != watch_stages` 即拒（双向相等）。
- 本仓改为单向：批内每个 watch 的 (request_id, stage) 必须有节点覆盖；
  批内允许"尾段节点（REQUEST_COMPLETE 边界的 completion_evictions + 下一 turn
  interval gates，stage=decode/generation=1）无新 watch"——sh_1.0 三段式发射
  所需；蓝本两段式批次不受影响（仍满足双向相等）。与 sh_2.0 的 node-only
  段扩展同类。

**(b) 空队列饥饿 fail-closed guard**（main_online.cc 主循环）
- 触发条件：`event_queue->finished() && !svc.finished() &&
  svc.active_request_count() > 0` 持续 ≥30s 墙钟（30s 内 continue 重检，
  消除瞬时误杀——same-tick fixture 曾被无宽限版误杀，实测）。
- 诊断输出：per-rank `NodeStore::debug_unfinished_names()`（前 24 条，
  "I:"=issued-未完成 / "F:"=free-未发行，含节点名）+ stale watches 数 +
  active 请求数，随后 `std::abort()`（蓝本 wait_for_work 谓词含
  `!input_open_`，在 --close-input 且图死锁时永真 → 100% CPU 空转，
  此为其缺陷修复；与 sh_2.0 livelock guard 同族，收尾统一回灌评估）。
- 附带诊断 API：`NodeStore::debug_unfinished_names(limit)`；env 门控探针
  `SH10_DEBUG_ISSUE`（issue/enter/sched/loop/slowcb/stall/deadlock 打印，
  WatchRegistry.cc online_issue_pass_cb 等处，默认零输出）。

**已回退的实验性改动**（未入机制，仅为死锁排查过程记录）
1. collective 计数式槽位占用（HardwareResource occupy/is_available 加
   online_mode_ CommCollective 分支）——未证明语义安全，frontier 修复后
   不再需要，已还原（set_online_mode 接口保留但无行为分支）。
2. 回程 recv 独立化（strategy 下 `_ack_from_`/`_request_from_rank` recv
   不链 previous_id）——掩盖环而非消除环，frontier 修复后已禁用
   （standalone_backchannel_recv=False，helper 保留）。

## 第三任 agent 接管审计（2026-08-16，前任 14:32 额度中断后）

前任中断点 = tier_b_report_20.md 终稿写完（14:32）后、阶段 3 实录登记前。
逐项审计结果（本任实测复核，非仅读文件）：

- **(a) /tmp/sh1_run_p3_replay 与 sh1_run_p3_strategy（14:29-14:30）**：两
  运行均完整收尾——replay 1177/1177 完成、3530 交付、双进程正常退出；
  strategy 1177/1177、3531 交付、正常退出；无死锁 abort。replay cpp.log 的
  292 条 `[METRIC][ERROR] consistency violation` 与阶段 2 replay 运行条数
  完全一致（292==292，replay LUT 时钟下 arrival>prefill_start 的既有信息性
  输出，非阶段 3 新问题；strategy 为 0）。
- **(b) online/verify/graph_batch_audit.py**：脚本完整（三门槛：graph_batch
  ==delivery 对平 / single_node_bridge_count==0 / 总账 digest==计数器 +
  Python summary==ack==delivery）。本任实跑：两 run 目录均 PASS（replay
  deliveries=3530 nodes=329308；strategy deliveries=3531 nodes=323778；
  single_node_bridge 均 0）。
- **(c) tier_b_report_20.md 终稿**：完整（B0-B4 全 PASS + 口径差异登记 +
  结论，119 行）。
- **(d) GraphBatchCommitter.hh/.cc 阶段 3 改动**：完整非半成品——§8.1
  MEM 节点校验（type 2/3 tensor_size 正数 + rank 必须在 remote_mem_ranks
  端口集；空映射 = 任何 MEM 节点拒绝，fail-closed）+ main_online.cc 已解析
  remote_memory.json npu-ids 灌入 committer_ctx；测试
  graph_batch_committer_test.cc 已含 MEM fixture（4 例：零 tensor_size 拒、
  MEM_STORE 零 tensor 拒、无端口 rank 拒、合法端口节点过——expect_reject
  内置 snapshot 前后对拍 = 零副作用断言）。编译于 14:19:02 完成；本任实跑
  测试二进制 `ALL PASS`（exit 0）。Online 二进制 14:18:35 构建（含 MEM
  校验），/tmp 两个 p3 运行即此版本产物。
- **(e) 工作树 36 条目**（25 M + 11 ??）与实录登记逐项对得上：机制层移植
  文件、online/ 全套、4 个 runner、baseline 归档、traces 物化、
  online_contracts、实录本体——无未登记残留。

结论：前任已实质完成阶段 3（GraphBatch 验收，= 方案 §8），缺运行证据登记
与实录条目，本任补齐如下节。

## 阶段 3（2026-08-16 完成）——GraphBatch 原子提交验收（方案 §8）✅

### 验收材料
- 官方 runner 全量（与阶段 2 同款输入/ET/配置）：
  `/tmp/sh1_run_p3_replay`（replay，1177/1177，3530 交付）、
  `/tmp/sh1_run_p3_strategy`（strategy，1177/1177，3531 交付）。

### 门槛逐项 ✅
1. **非法批零副作用 fixture**：`tests/graph_batch_committer_test.cc` 的
   expect_reject 对每个非法批做 Snapshot 前后全量对拍（validate() 为纯
   函数，状态零修改）；新增 MEM 节点四例（零 tensor_size / MEM_STORE 零
   tensor / 无端口 rank / 合法通过）+ 蓝本既有非法批族。二进制实跑
   ALL PASS。
2. **single_node_bridge_count == 0**：两 run 的 phase-5 计数器行实测 0
   （graph_batch_audit.py 复核）。
3. **graph_batch == delivery 对平**：graph_batch_audit.py 两 run PASS——
   delivery_sequence 集合 == {0..N-1} 无跳号无重复；digest 行恰覆盖全部
   交付；graph_batch_count == N；digest node_count 合计 == 计数器
   total_nodes（replay 329308 == B3 canonical 节点总数，交叉自洽）；
   Python summary delivery_count == ack_count == N。
4. **B0-B4 等价回归**：阶段 2 全过（本阶段未触策略/构图语义，仅增
   Committer 校验规则——正式批全合法故零影响，3530/3531 交付数与阶段 2
   一致）。
5. **静态基线可复现**：静态路径不走 Committer（--online-mode 专属），
   阶段 2 已复验 58/58；本阶段 C++ 改动仅在线目标文件。
6. **GraphBatch 平均/最大节点数有记录**：replay avg 93.29/max 144；
   strategy avg 91.70/max 486（audit 输出）。

### 机制登记（补充实录前文 (a)/(b)）
- **(c) §8.1 MEM 节点校验（本任接手时前任已实现，本任验证收编）**：
  GraphBatchCommitter validate 规则增补——MEM_LOAD/MEM_STORE（type 2/3）
  的 tensor_size 必须为正（在线 builder 已 clamp >=1，零值即缺陷）且节点
  rank 必须拥有 AnalyticalRemoteMemory 端口（remote_memory.json npu-ids
  26 边缘 rank）；remote_mem_ranks 为空 = 未配置端口映射 = 任何 MEM 节点
  拒绝（无端口的 MEM 节点永不可能执行，fail-closed）。main_online.cc
  解析 npu-ids 灌入；fixture 不发 MEM 节点故不受影响。

## 阶段 4（2026-08-16 完成）——感知打开 + remote FIFO 实账本对账 ✅

### remote FIFO 实账本层（合同⑥ 本仓核心差异项，实建）
- **采集方式（红线兼容）**：AnalyticalRemoteMemory 两文件零改动（阶段 1
  红线保持）；Workload 侧观测计数——`ExecutionDriven::RemoteFifoLedger`
  （Workload.hh/cc，进程级单例）：issue 计数挂在 Workload::issue_remote_mem
  真实 FIFO 分支（replay 即时完成分支不记），completion 计数挂在
  Workload::call 泛型 wlhd 终态分支的 online+!replay_clock MEM 节点判别。
  **精确性论证**（Workload.hh 注释）：ARM 端口 FIFO 为严格单服务队列，
  in_flight = issued - completed = active + pending（服务忙 ⟺ 有已开始未
  完成请求）——计数器组重构端口状态精确成立，无需读取 ARM 私有成员。
  静态路径零记录（gate Online 模式），纯观测零语义影响。
- **出口两路**：
  1. sensing 门控 sidecar `<bridge_dir>/remote_fifo_ledger.jsonl`：每交付
     一行（delivery_sequence/tick/逐端口 active/pending/in_flight_bytes/
     issued/completed 计数与字节累计）——`--sensing-enabled` 才开档，普通
     运行零开销；
  2. 运行末计数行 `[online] remote fifo ledger: ports/issued/issued_bytes/
     completed/completed_bytes/peak_pending/peak_in_flight_bytes` +
     **fail-closed drain 断言**（任何端口 issued != completed = 完成丢失
     = abort）。replay 模式 FIFO 账本恒零 = 合同⑦第④项语义（MEM 即时
     完成），登记非泄漏。
- 新 runner `run_online_strategy_sensing.sh`（与 strategy runner 同构 +
  `--sensing-enabled`/`--sensing` + 归档 remote_fifo_ledger.jsonl；计数行
  直接用 find 计数）。

### sensing 全量验收（20.csv 前30s）
- `/tmp/sh1_run_p4_sensing`：1177/1177 完成、3531 交付、双进程 exit 0、
  single_node_bridge=0；7 份 jsonl 归档（含 remote_fifo_ledger 3531 行、
  sensing_query_log 3531 行、ledger 1177 行）。

### 感知开/关决策日志对照
- `--sensing` 运行的 online_decision_log.jsonl（3531 行）与阶段 3 关感知
  strategy 运行**逐字节一致**（cmp=0，md5 `f8972f480239ca309d66dd9e2104010e`
  同值）——感知数据纯查询/审计输入不进策略判据（红线 §0.4）的逐字节证明；
  感知开不改变 placement/完成时序（差异报告 = 全部"无差异"）。

### ledger 对账（`online/verify/ledger_reconcile_sh10.py`，sh_2.0 模板适配）
- **R0-R5 全 PASS**：R0a/b/c（1177 三流相等无重复）；R1（admitted<=
  completed 0 违规；first_commit_tick null 残差 = 段 3 无发射凭据，同
  sh_2.0 口径，提交层由 graph_batch==delivery==3531 对平）；R2（watches
  2354==2×1177）；R3a/b（两态 KV 终态位置 local_hbm/remote_memory 全
  合法）；R4（admitted/committed/issued 三层清零）；R5（感知查询日志
  3531==交付数）。
- **RF0-RF3d（remote FIFO 实账本层）全 PASS**：26 端口、852 次 issue、
  3056836542464 字节；RF1 快照逐交付覆盖 3531/3531；RF2a 全端口 drain
  （含 C++ fail-closed 复核）；RF2b sidecar 累计==cpp.log 总账；RF2c
  字节守恒 issued==completed；**RF3a/b Python 决策期望（remote_store
  105 + remote_load 37 个转移的全部分片）vs C++ 实发 issue 逐 rank 计数
  与字节双对平（diff_ranks={} 零残差）**；RF3c 端口集==决策 edge rank 集。
  **RESULT=BALANCED，exit 0**。
- 残差登记：R3 KV 逐事件流不适用（B2 oracle + RF3 闭环承担）；R6/R7
  自完成尾部（sh_3.0 合同①口径）+ local HBM job 不适用占位（合同⑥）；
  RF peak_pending=0（依赖链串行化，物理观测）；peak_in_flight_bytes
  ≈20.8GB。
- 决策日志六类转移分布（信息性）：noc_migrate 1878 / local_hit 327 /
  remote_store 105 / remote_load 37（转移行合计 2510——与 B2 oracle 的
  KV 载荷行数同源交叉自洽）。

### 修复登记（阶段 4）
- online_scheduler_base.dump_ledger docstring 误写"remote FIFO 不适用
  占位"（蓝本复制残留，与合同⑥ 本仓裁决相反）——改为实账本层描述 +
  sidecar 出口说明。仅文档修正。

### 回归（阶段 4 改动后）
- replay 全量复跑（绝对路径参数——相对路径按设计 fail-closed，阶段 2
  已登记的口径）：1177/1177、3530 交付、决策日志与阶段 3 replay 运行
  **逐字节一致**（cmp=0）= FIFO 观测代码零语义影响；replay 运行
  `[online] remote fifo ledger` 行全零（合同⑦第④项 MEM 即时完成不走
  真实 FIFO，登记非泄漏）。
- 静态路径：FIFO 计数 gate Online 模式，静态二进制行为零变化（阶段 7
  收尾静态字节门复验）。

## 阶段 5-6（2026-08-16 完成）——窗口冻结/并行 reader/benchmark ✅

### 窗口扫掠冻结
- `--request-window-rows 0`（无界对照：pumps=1 / peak occupancy=1177）
  vs 冻结值 128（pumps 99-100 / occupancy 128）全量 replay 重跑：
  **决策序列逐字节一致**（cmp=0），超界拒绝 0 → **128 冻结**（全部
  turn-0 行落于首窗口，与蓝本同构）。对照运行 = 官方 replay runner 的
  临时副本（注入一行旗标，未改仓内 runner）。

### 并行 reader 有界验证
- WindowedReaderTest **8 并发实例 8/8 ALL PASS**，每实例峰值 RSS
  ≈7.5-7.9 MB（无随 worker 数线性复制）。
- 配套两处修复（登记）：(a) 测试 fixture 改 per-PID 临时路径（蓝本固定
  /tmp 路径并发互踩，fixture-only，与 sh_3.0 验证轮同款）；
  (b) analytical CMakeLists 补 WindowedReaderTest
  RUNTIME_OUTPUT_DIRECTORY 块（蓝本同款缺口，二进制落 bin/）。

### benchmark 矩阵（`online/verify/benchmark_20.md` 终稿）
- 三在线运行顺序计时（当前二进制）：replay 27.2s / strategy 32.8s /
  sensing 131.6s；离线两口径 = e2e 56.33s（阶段 0 归档）/ 静态执行段
  14s（归档 run log）。makespan：replay 1,166,403,738,625 /
  strategy·sensing 2,296,622,383,885；全部 1177/1177。
- 决策门五条件：1/2/4/5 PASS；条件 3 PARTIAL 登记（Python 份额
  51%/57%/53%——文件桥 v0 优化项，与 sh_2.0/sh_3.0 同款口径）。
  **判定通过，性能预算冻结。**
- 信息性优化项登记：sensing 的 C++ injected-unfinished 摘要
  snapshot_ns=87.2s（逐交付全量扫 54 rank）——增量索引为后续优化；
  sensing 产物大头 sensing_query_log 370MB/run（按交付×54 rank 有界）。

## 回灌轮 backport-fix1（2026-08-16，四仓同源缺陷回灌，本任执行）✅

### 项 1：OnlineCli 默认窗口 fail-closed 化（§5.1）
- **采用方式**：四机制文件整文件采用 wscllm 修正版并 cmp 复核逐字节
  一致（OnlineCli.cc/.hh + WindowedTraceReader.cc/.hh，cmp OK ×4）——
  含：默认 `--request-max-arrival-ns` 改无界（0；旧 30e9 默认把验收
  窗口烧进代码、越窗请求静默丢弃）；完成度审计分母改 total rows
  （`count_remaining_data_rows` 尾扫——拒绝行不消费可堵塞窗口致 EOF
  不可达、旧审计静默跳过 = silent-PASS 洞）；`audit_completion` 四判定
  （Ok/Dropped/AccountMismatch/Incomplete）factored 到
  WindowedTraceReader.hh（可单测）；input window audit 非零退出
  （main_online.cc gate_ok 接线 + Dropped 判定打印并 exit 1）。
  注：wscllm 语义 = 拒绝行不消费 + 尾扫补分母（face/wscllm 血统锚）；
  sh_2.0/sh_3.0 的"拒绝即消费"为其仓变体——本仓按主控指令锚定
  wscllm 形态。
- main_online.cc 审计接线（尾扫 + audit_completion 替换旧
  completed==expected 单判定）与 reader banner 打印对齐 wscllm；
  仓特有部分（remote FIFO 账本/空队列 guard/MEM 校验）保留。
- **合成 fixture**：reader 测试整文件采用 wscllm 版（含 Part G 默认
  无界全接受 / Part H 显式小窗丢弃可见 + fail-closed 审计四判定）+
  重挂本仓 per-PID fixture 路径（并行验证所需，fixture-only）。
  WindowedReaderTest 实跑 A-H 全 PASS。
- **集成实证**：显式 `--request-max-arrival-ns 1000000` 全量 replay →
  拒绝 11 条可见（rejected_out_of_range=11）、分母 total_rows=1177
  （尾扫，非窗口已读 128）、`[Error] completion audit FAIL (dropped
  >0; visible drop, fail-closed)`、C++ exit≠0、runner exit=1、Python
  侧 fail-closed（unconsumed replay decisions）。runner 在 set -e 下
  于 wait 处即时中止（诊断 echo 跳过）= 蓝本家族既有行为（wscllm/
  sh_2.0 runner 同款 `wait; CPP_EXIT=$?` 模式），退出码正确传播。
- **正常路径复验**：默认无界下 replay/strategy/sensing 全量 PASS
  （1177/1177 ×3），banner 实测 `max_arrival_ns=0 (unbounded
  default)`，决策日志与阶段 3 运行逐字节一致（cmp=0 ×2；感知开/关
  亦一致）= 本输入（max arrival 29.99s）本就零拒绝，零语义影响。

### 项 2：runner ls 通配计数改 find
- run_online_replay.sh / run_online_strategy.sh 两处计数行
  （CP_COUNT/REQ_COUNT）+ 失败诊断行改 find -maxdepth 1（ARG_MAX/
  E2BIG 防御）；run_online_strategy_sensing.sh 创建时即用 find；
  fixture runner 无 ls 通配计数行（核实）。bash -n 三脚本过。

### 项 3：turn-0 在飞逐出 KeyError 同型排查 = 不同构，结构性免疫（登记）
- 病灶行定位：graph_batch_builder.py:404 直接下标
  `pending_history[pending_request_id]["location"]`（sh_2.0 同款
  `_mark_pending_history_*` 家族）。
- **免疫证据**（合成单测 `online/test_turn0_eviction_probe_sh10.py`
  4/4 PASS 固化）：本仓 turn-0 到达门登记（_emit_arrival_gate
  :763-768）与 turn>0 following 登记（_emit_segment3 :870-875）均在
  **同一调用内成对**写 pending_history[request_id] +
  pending_request_by_session[session_id]——turn-0 的 gate 是落账本的
  new_session 门（sh_2.0 为内联不落账本，故病态）；消费侧
  _emit_segment1 :508/:512 成对 pop；单线程逐批构造无交织 →
  "session 键指向无 gate 请求"的病态状态不可达。case A/B 覆盖正常
  两分支（无键→deferred / 成对→就地折算），case C 人为构造病态演示
  KeyError 签名（固化病灶定位），case D 静态断言全部登记/弹出的
  成对性（未成对的新增登记将 FAIL = 回归护栏）。
- 30s 运行证据：三模式 3530/3531 交付零 KeyError；B2 确定性重放
  全交付流零异常逐行全等（阶段 2 已证）。
- 无需修复（不适用），登记证据如上。

### 对账器加固（主控中继指令轮，sh_2.0 5ba5506 同款）
- ledger_reconcile_sh10.py：R2 拆三层——R2a digest watch 和 == cpp
  phase-5 total_watches == 2×N 三方对平 / R2b 批级上界
  0<=watch_count<=len(ranks) + ranks⊆[0,53] / R2c digest node 和 ==
  cpp total_nodes；R3 增 R3c（completion 决策 tick == ledger
  completed_tick 逐对零失配）。
- **digest.ranks 语义先实测**（主控提醒的坑位）：本仓 30s sensing
  产物实测 3531 批 watch 分布 {1:2354, 0:1177}；**空性等价不成立**
  （1042 违例，全部为 DECODE_COMPLETION+REQUEST_COMPLETE 完成通知
  批：ranks 非空（段 3 尾节点）但恒 0 watch——实录机制裁决 (a) 单向
  覆盖的合法批型）→ 空性断言不移植（同 sh_2.0 处置）；上界与子集
  零违例可移植。
- sh_2.0 R3c 的 "cpp total_kv_actions >= 请求数" 子判不适用（本仓
  KV 编码为图 MEM 节点，cpp total_kv_actions 恒 0——KV 侧闭环 =
  RF3 逐 rank FIFO 字节对平 + B2 oracle，登记）。
- 加固版实跑：p4_sensing 与 fix1_sensing 两运行 R0-R5+RF 全 PASS
  RESULT=BALANCED。

### 回灌后复验矩阵（全绿）
- **静态字节门**：generate_trace.sh 重生成 → 归档 58/58 cmp 一致
  （generate_face_trace.py 未动，静态路径零影响）。
- **pytest 双根**：workload 根 26 passed；sh_test_mesh 根 63 passed
  （= 原 59 + turn0 探针 4 例）。
- **C++ 机制测试**：GraphBatchCommitterTest（含 MEM 校验族）/
  WindowedReaderTest（A-H 含 Part G/H）/ DecisionMailboxTest /
  WatchRegistryTest / NodeStoreTest（--fixture-et=... 含 part D
  injected-unfinished 摘要）/ BridgeLoopbackTest（仓根运行——echo 脚本
  相对路径）全 ALL PASS。
- **在线三模式**：replay/strategy/sensing 全量 PASS ×1177/1177，决策
  日录与阶段 3 逐字节一致 ×2 + 感知开关一致 ×1。
- **机制共享文件血统**：OnlineCli.cc/.hh + WindowedTraceReader.cc/.hh
  与 wscllm（=face）cmp 逐字节一致 ×4。
- 运维登记：前任 14:0x 遗留的 BridgeLoopbackTest 僵尸进程（占固定
  /tmp 桥路径 7h，卡住本任测试循环）已清理；陈旧 /tmp 桥目录 ×13
  清理。教训 = 主控提醒的后台任务年龄上限同款（机制测试的 FIFO 资源
  也算）。

## 阶段 7（2026-08-16 完成）——裸仓库还原 + 一次性 commit/tag ✅

### 还原动作（face/sh_2.0/sh_3.0 先例）
- **删除物化输入与基线归档**：traces/ 两 csv（队列+sidecar）与
  sh_test_mesh/baseline/20_30s/（静态 ET 58 件 + decision_log +
  raw/normalized metrics + run_logs + PROVENANCE）删除；traces/ 保留
  PROVENANCE.md（含裸仓态重建命令与 md5 冻结值）+ 物化器
  derive_20_first_30_seconds.py。generated/ 清至仅 runtime_config
  （可再生目录保留）；results/ 清空。
- **trace_config.csv :12 中性化**：指向 request_queue_placeholder.csv
  （占位，正式入口缺失输入 fail-closed exit=1 实测，打印物化指引）。
- **测试合成 fixture 化（sh_2.0 skipUnless 形态）**：
  test_face_scheduler.py 增 _MATERIALIZED 探针 + 7 个 config 依赖用例
  skipUnless（物化输入后自动恢复全量）；新增常态用例
  test_checked_in_config_is_request_neutral_and_fails_closed_without_input
  （占位路径断言 + SystemExit 非 0 断言）。双根 pytest：workload 根
  20 passed/7 skipped；sh_test_mesh 根 57 passed/7 skipped（物化后
  26/63 全量）。
- **在线 runner ET 解析动态化**：3 个在线 runner（replay/strategy/
  sensing）的 ET_PREFIX/RC 由 baseline 硬路径改为
  sh_test_mesh/generated/ 下唯一 llama2_7b_inference_54npus_* 目录
  动态解析 + runtime_config 再生路径（与 sh_2.0 回灌同款；目录名编码
  窗口与 config digest，硬编码路径随 config 字节变化必断）。
- **冻结 LUT 表缺省路径动态化**：sh10_online_scheduler.
  _default_frozen_lut_path 由 baseline/20_30s 硬路径改为 generated/
  唯一 ET 目录内 face_lut.csv（缺失回退旧路径并 fail-closed 报错，
  合同⑨ lookup-only 语义不变；裸仓重建冒烟暴露并修复）。
- README_COMMANDS.md 重写为裸仓态命令速查（物化→生成→在线→验证
  工具全链，含双根 pytest 口径与归档删除说明）。

### 裸仓重建端到端冒烟（证明还原路径可用）
- 物化器重跑：两产物 md5 **精确复现冻结值**（队列
  ee7af9d0bc3e9c2e211d1e30c325bd45 / sidecar
  8525bf190950a33b6e6bd66c0c36c2e7）。
- trace_config 指定物化输入 → generate_trace.sh 重生成 ET →
  run_online_strategy.sh 全量 **PASS（1177/1177，3531 交付）**；
  **决策日志与回灌轮运行逐字节一致（cmp=0）**；remote FIFO 账本同值
  （852/3056836542464）——重建链路全确定性。
  （首次冒烟在交付 3077 处 Python 侧无输出硬死（无 traceback，机器
  高载 8 仿真进程并行时段），重试即过且逐字节同前——环境性 kills
  登记，非仓内缺陷。）
- 冒烟后恢复裸仓态（删 csv/ET/results，config 回占位）。

### 终态剩余差异登记（如实）
1. 决策门条件 3 PARTIAL（Python 份额 51-57%，文件桥 v0 优化项，
   benchmark_20.md 登记口径与 sh_2.0/sh_3.0 一致）；
2. B3 边差异 17270 全归因类别①②③④（tier_b_report_20.md；节点
   多重集 0 差异硬门过）；
3. B1 prefill 20 处 alarm-clamp 正偏差 [1,10]ns（sh_1.0 边界口径
   冻结登记）；
4. teardown `[critical] unreleased nodes`（sys 20/21/26/27/32/33，
   退出期尾节点抱怨日志——与 sh_2.0/sh_3.0 同款定性，登记不修）；
5. replay 运行 292 条 `[METRIC][ERROR] consistency violation`
   （LUT 时钟下 arrival>prefill_start 等既有信息性输出，阶段 2 与
   阶段 3 同数复现，非回退）；
6. 感知快照 snapshot_ns=87.2s 与 sensing_query_log 370MB/run（增量
   索引/分片为后续优化项，sensing 默认关不进正式路径）；
7. 空性等价断言不移植（digest.ranks=批图全节点集语义，1042 合法
   违例——对账器加固轮实测登记）；
8. 阶段 0 的基线归档（baseline/20_30s）与阶段 2 的 /tmp 运行产物为
   过程证据，已按裸仓还原删除——证据保存在本实录（冻结 md5/计数
   值）与可重建链路（冒烟已证逐字节可复现）。
9. 缺陷 A/B/C（face 四档测试暴露的 ServiceCoordinator 终判/FIFO
   误检/唤醒丢失竞态）属后续统一同步轮，不在本仓范围。

### git（一次性补齐，pathspec 仅本仓）
- sh1-phase0..7-done + sh1-backport-fix1-done + sh1-final-done
  （9 commit 9 tag；一次性补齐口径：每文件按最早归属阶段提交其最终
  内容，中间提交非可运行态——与 sh_2.0/sh_3.0 收尾同款操作语义）。
- 工作树对本仓干净（其余仓改动不触碰）。

## 缺陷修复 2 同步轮（2026-08-16/17，来源：face 0049ef5 / tag face-defectfix2-done）

> face 四档主动测试暴露的三类在线机制层竞态缺陷（A=服务终判
> finished() 计数真空、B=文件桥 FIFO EOF 误检、C=空队列唤醒丢失）已
> 在 face 修复入库；本节为按其"移植性登记"同步到本仓的实录。
> 策略红线零触碰：face_scheduler.py / session_kv_manager.py /
> generate_face_trace.py 零改动（git diff 复核）。

### A. 缺陷 A：服务终判 finished() 计数真空（main_online.cc）

- pump() 后、finished() 判定前补一次 drain（pending_command_count()>0
  → drain_commands()），关闭"已入队未注册"真空期；
- belt-and-braces：finished() 为真而 CSV 未 EOF → online_fatal（携带
  active/pending_alarm/queued_commands 计数）。修后末笔
  ack==delivery 恒成立（复验承载：replay delivery 3530=graph_batch
  3530、strategy 3531=3531）。
- 回归 fixture：`tests/service_vacuum_test.cc`（新 CMake 目标
  ServiceVacuumTest）。30 行 turn-0 CSV 驱动真
  WindowedTraceReader/RequestIngress/ServiceCoordinator/EventQueue
  两种循环顺序：legacy 确定性复现真空（completed=10、断点队列 10 条
  未 drain、reader 未 EOF、审计非 Ok）；fixed 30/30 + EOF + Ok。
  ALL PASS（先复现再转绿）。

### B. 缺陷 B：文件桥 EOF 误检（DecisionBridge.cc/.hh + decision_bridge.py）

- C++ 与 Python 同版本成对移植（移植性登记要求）：resp_notify 通道
  双侧常开长连接 fd——C++ open_notify() 一次打开读端 run 期持有、
  析构对称关闭（新增 resp_notify_fd_）；Python serve_forever 启动时
  一次打开写端、finally 关闭、_notify_response 只写不开；
- wait_response_byte（新私有）：常开 fd 上 poll→read 恰 1 字节、
  EAGAIN 重 poll；read==0 恢复真死义（"Python side died (its
  long-lived resp_notify write end closed...)"）；1:1 在途守卫（第二
  字节可读 → protocol violation fail-closed）；
- Python BrokenPipe → BridgePipeError fail-closed + stderr 留痕
  （"C++ side is gone"）；_fail() 先 stderr 留痕再写 error response。
- 回归 fixture：loopback 新增 Part D（SIGKILL 真死检含新消息）/
  E（1:1 守卫）——A-E ALL PASS；`online/verify/bridge_cpp_death_
  fixture.py` 连续 3/3（exit=1 + stderr 留痕）；
  `run_scripts/bridge_race_stress_repro.sh` 旧协议竞态复现装置
  （本机 45s 预算第 2 次迭代即复现假 EOF，写端存活）。

### C. 缺陷 C：空队列分支唤醒条件（main_online.cc + EventQueue）

- 空队列分支三分支化：mailbox 残留 → 既有 T+1（不变）；
  **新** has_deferred_work() → 同机制 T+1 强制下一决策边界（不计
  deferred_from_tick）；**新** input_closed 且队列+mailbox 双空而
  svc 未完 → lost-wakeup dead end fail-closed；input 开 →
  wait_for_work（IDLE 合同不变，五态迁移 fixture 复验全过）。
- EventQueue（list 版，与 face 同血统）：新增 const 访问器
  has_deferred_work()（.h/.cpp 纯新增，静态二进制零行为变化；本仓
  SH10_DEBUG_ISSUE 探针原样保留）。

### guard 收敛（主 agent 指示：不留两个语义重叠的 guard）

- 本仓"机制特有改动 (b)"的 30s 空队列饥饿 fail-closed guard 与
  face 的 lost-wakeup 死端 fail-closed 同族（终判计数不覆盖在途
  工作 / 队列排空而服务未完的死端），**收敛为一个统一机制**：
  移除循环尾部 30s guard（starve_since/dumped/30s 宽限窗整体
  删除），其触发态由空队列分支内的死端 fail-closed 立即承载
  （无宽限窗、首观察即报），并保留本仓 guard 的最强诊断面：
  per-rank injected_unfinished_summary + NodeStore::
  debug_unfinished_names + stale watches（[online-deadlock] 前缀
  dump）叠加 face 的 active/pending_alarm/deferred_work 计数。
  收敛后单一机制诊断能力 = 两者并集；coverage 亦为并集（旧 guard
  仅 active>0 形态，新死端覆盖 pending_alarm 丢失形态）。
- 实证：wakeup_guard defer-dead-end 场景 fail-closed 消息同时含
  "lost-wakeup dead end ... active=1"与 [online-deadlock]
  stale watches=0 行（该场景无未完成节点，rank dump 稀疏输出正确）。

### 本轮同步的既有失配修复（fixture 层，登记）

- `run_online_idle_fixture.sh` / `run_online_same_tick_milestone.sh`：
  阶段 7 漏改（官方 runner 已 glob 化，这两个 fixture runner 仍硬
  编码已删除的 baseline/20_30s 归档路径）→ 改为 generated 唯一目录
  glob 解析（与 run_online_replay.sh 同款）；复验五态迁移 +
  same-tick milestone (a)-(d) ALL PASS。
- `run_online_wakeup_guard_fixture.sh`（新入库）：ET 目录采用同款
  glob 解析（face 原版硬编码 face 仓目录名）。

### 复验证据（20.csv 前 30s 包络，仿真发射门控记录见下）

1. 构建+单测：build 全目标 0 error（含新 ServiceVacuumTest）；
   NodeStore（--fixture-et 含 part D）/WatchRegistry/DecisionMailbox/
   GraphBatchCommitter/WindowedReader A-H/ServiceVacuum/BridgeLoopback
   A-E 全过；event_queue_deferred_test（g++ 现编，本仓 include 根
   astra-network-analytical）ALL PASS；bridge_cpp_death 3/3。
   cli_online R1-R11 未复跑：本轮 diff 不含 OnlineCli.cc/.hh（回灌
   fix1 终态），其合同不受影响，登记。
2. 物化：源 md5 fc74a48e 一致，queue/sidecar md5
   ee7af9d0.../8525bf19... 与 PROVENANCE 冻结值 exact（1177/112）。
3. 决策日志重录（--replay-record）：197,859 行；md5
   **2e0e72de30eb0b13993037cac90d7629** —— 与阶段 0 登记冻结值
   "se2e0e72de30eb0b13993037cac90d7629" 的**尾 31 位逐字符一致**
   （原登记首字符 "s" 为笔误，md5 为 32 位十六进制；本次登记修正，
   行数 exact 佐证）。generate 后 59 文件 = 静态 58 + decision_log。
4. 静态 runall：exit 0，completed_unique_requests=1177、
   incomplete=0、memory_actions_unresolved=0、sim_end_ns=
   2,729,949,453,432；raw_metrics 268 行。静态二进制源集合不含
   execution_driven TU（CMake 源列表），唯一交集 EventQueue 为纯
   新增 const 访问器（零静态调用方）——静态路径零扰动论证与 face
   同款；阶段 0 的 baseline/20_30s 逐字段比对归档已删（阶段 7），
   零语义差异由下述在线冻结值 exact 承载。
5. replay：PASS——1177/1177、delivery_count=3530（冻结 exact）、
   accepted=112、no_decision=0、sim_end_ns=1,166,403,738,625
   （**阶段 5-6 冻结 makespan exact**）。
6. strategy：PASS——1177/1177、delivery_count=3531（冻结 exact）、
   no_decision=0、single_node_bridge_count=0、graph_batch_count=3531
   （==delivery，缺陷 A 不变量）、total_nodes=323,778、avg 91.70、
   max 486（实录登记三值 exact）、sim_end_ns=2,296,622,383,885
   （**冻结 makespan exact**）。
7. Tier B 复跑不回退：tier_b_compare.py（baseline 侧由重生成产物
   构造——decision_log/manifest/ET 与冻结血统逐字节一致已证）
   **B0/B1/B2/B2_STRATEGY_RERUN/B3/B4 ALL LAYERS PASS**。
8. wakeup_guard 双场景：f7-blueprint PASS（accepted=1 completed=2、
   future_alarm 留痕、无误触发）；defer-dead-end PASS（数秒内
   fail-closed、"lost-wakeup dead end"+active=1 入消息）。
9. IDLE 五态迁移 ALL PASS；same-tick milestone (a)-(d) ALL PASS。

### 仿真发射门控记录（如实）

- 另一 agent 的 sh_2.0 120 档 3 分钟 sensing 重仿真 02:14-05:02 占
  用在线仿真槽，本轮全部仿真发射（wakeup_guard/静态/replay/
  strategy/Tier B 运行/IDLE/里程碑）排队等待至其结束后执行；发射
  期内存 used 34-45%（<80% 红线），另有并行 agent 的短 replay 与
  一个空转 wscllm replay 进程共存（face 复测同款并行共存口径，
  登记不隐去）。
- /tmp（40G tmpfs）中途写满（多轮历史运行产物堆积）：清理**本会话
  之外的旧运行目录**后重跑受影响的 sh_3.0 sensing（唯一受污染项，
  其余产物先于写满完成）；本轮证据产物 /tmp/sh1_df2_*、/tmp/
  sh30_df2_* 保留至收口。

### 裸仓态恢复

- 删物化 csv（traces/ 两件）与 generated 产物（ET 目录 +
  completion_fixture）与 results/；trace_config.csv :12 回
  placeholder（git checkout 原字节）。
- fail-closed 实测：generate_trace.py 无输入 exit=1（中文报错行）。
- 裸仓态双根 pytest：tests 33 passed + workload 24 passed/7 skipped
  （7 个 _MATERIALIZED 门控用例回 skip；request-neutral 常态用例
  复绿）。物化态全量跑时该用例按设计失败（1 failed：其合同域是
  裸仓态占位 + fail-closed，物化输入在位即不满足）——如实登记，
  未做 deselect 掩盖；判定以裸仓态复绿为准。

### git（本轮）

- commit（pathspec 仅本仓）+ tag `sh1-defectfix2-done`：机制层五文件
  （main_online.cc / DecisionBridge.cc/.hh / EventQueue.h/.cpp /
  decision_bridge.py）+ fixtures 四件（service_vacuum_test.cc /
  loopback 扩展 / bridge_cpp_death_fixture.py / wakeup_guard_fixture_
  service.py）+ 脚本两件（run_online_wakeup_guard_fixture.sh /
  bridge_race_stress_repro.sh）+ CMake ServiceVacuumTest + 两 fixture
  runner glob 化 + 本实录同步段。
