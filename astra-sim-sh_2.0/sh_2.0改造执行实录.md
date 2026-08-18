# sh_2.0 改造执行实录

> 执行 agent 实时维护。依据《sh_2.0仓库改造详细执行方案.md》（下称"方案"）。
> 纪律差异登记：主 agent 指示 **不要 git commit**——方案 §0.5 的逐阶段
> commit/tag 纪律改为在本文档记录阶段边界与改动清单；回滚锚点 =
> 父仓 commit `55cf135`（工作树起点，已核实干净、与 template_back 零差异）。

## 阶段 0

### 0-0 工作树核实（2026-08-16）
- `git status` 干净；起点 commit = `55cf135`。
- `diff -rq template/astra-sim-sh_2.0 template_back/astra-sim-sh_2.0`
  （排除 `__pycache__`）零差异 → 未改造静态 ET 基线确认。
- 发现 `template/agent-traces/TraceLab_.../derived/compute_20_first_20_minutes/`
  存在（方案 §1.6-2 已记录的同源 stub 垃圾，系其他仓触发写出；非本次运行
  产生，本仓 fail-closed 已阻断再写出，未删除——待主 agent 处置）。

### 0-1 输入物化 + fail-closed（2026-08-16）✅
- 新增 `sh_test_mesh/workload/llama2_7b_inference/traces/materialize_first_30s.py`
  （单次遍历产出 queue + context sidecar + canonical digest 三份文件）。
- 实测：**112 session / 1177 请求**；timing %1000 全过；最大派生到达
  25,959,142,000 ns；session_0 两行与 3 分钟 sidecar 血统逐字段一致
  （`13045,15659` / `15656,16305`）。
- 标定常数（合同⑨）：average_decode_length=459.4944774851317、
  max_d_token=53924、request_count=1177。
- `trace_config.csv:12` 修正指向
  `llama2_7b_inference/traces/astra_compute_20_first_30_seconds_request_queue.csv`。
  `:13` 保持空（sidecar off，默认 checked-in 语义；文件已物化存档）。
- `generate_face_trace.py` `load_face_trace_config` 内
  `_resolve_request_queue` 与 `load_request_queue` 之间插入 fail-closed
  `sys.exit`（queue 与 sidecar 两处，sidecar 区分"未配置/配置但缺失"打印）。
- 反向验证：指向不存在路径 → exit=1、打印 fail-closed 信息、无仓外文件
  产生。`--print-shell-config`：REQUEST_COUNT=1177 / SESSION_COUNT=112。
- 新增 `traces/PROVENANCE.md`（源 md5、规则①-④、常数、约束声明）。

### 0-2 基线复现与归档（2026-08-16）✅
- `run_sh_test_aware.sh:22` 改 `RUN_OUTPUT_LOG_TIMESTAMP=${RUN_OUTPUT_LOG_TIMESTAMP:-$(date...)}`（可复现 run_id）。
- 首次 runall（含 build）wall 53.28s、峰值 RSS 748,032 KB（time -v，步骤 0-7 压测基线在案）。
- 归档 `sh_test_mesh/baseline/20_30s/`：generated 58 文件 + raw/normalized metrics + run log + PROVENANCE.md。
- 可复现性：固定时间戳重跑 → 剔除 run_id/wall_time 后 raw_metrics 184 行逐字段一致（diff=0）；generated 58 文件逐文件 cmp 字节一致。
- 途中 bug：fail-closed 诊断 print 混入 --print-shell-config stdout 被 eval——改 stderr（实录 bug #1）。

### 0-4 九项合同冻结（2026-08-16）✅
- `online_contracts/nine_contracts/contract_01..09` + README（口径/裁决/验证方法三要素）。
- 关键裁决入合同：③ 1 EventTime=1ns（亲核）；④ map 版收口形态 + reason 闭集（无 TRANSFER_CONFIRMED）+ watch 边界映射；⑥ 八层全实（remote FIFO/local HBM 实账本）；⑦ replay 四裁决（含本仓 MEM/HBM-DMA 即时完成）+ B4 归因类别；⑧ sidecar_restore 禁双计、默认 off；⑨ LUT 在线继续消费、标定常数（459.4944774851317/53924/1177）、Kahn offline-only。

### 0-5 决策日志（2026-08-16）✅
- 方案 A：`generate_trace.py --replay-record` → `_write_replay_decision_log`（3531 行 = 1177×3，prefill/decode/completion 三流各 1177；iteration 流空——request_aggregated 不记录，相位时长用 tick 差）。
- 字节等价门：--replay-record 运行的 54 .et + manifest + face_lut.csv 与基线 cmp 全一致（两轮，含后补 history_location_before/hbm_wait_ns 字段后重跑）。
- 归档 baseline/20_30s/decision_log.jsonl（10.9MB）。

### 0-6 pytest（2026-08-16）✅
- test expectations 更新为 30s 物化值（1177/112/session_N/459.49/3-53924/1-13812/arrival 94835000-25959142000/derived max 29988879000）——仅测试期望值，被测代码零改动。36 passed。
- `run_scripts/README_COMMANDS.md` 建立。

### 0-7 压测基线（2026-08-16）✅
- time -v 数据在案（见 0-2）；计数器埋点：ED 机制层自带 OnlineStatsCounters/FileDecisionBridge::Stats/Python online_stats（离线路径零开销——hook 默认 null，静态字节门验证）。

## 阶段 1（C++ 机制层）

### 1-0/1-1 map 版 EventQueue 收口（2026-08-16）✅
- `EventQueue.h/.cpp`（map 族）：set_tick_end_callback / schedule_event_deferred / in_invoke_context + in_invoke_ + deferred_queue_（std::list<EventList>，与主 map 容器无关）。proceed() = begin/invoke(in_invoke_ 包裹)/erase/回调/deferred drain——回调在 erase 之后（map 特有：若在 erase 前，同 tick try_emplace 合入已 drain 的当前 list 会被 erase 静默丢弃）。
- FluidScheduler.cpp/.h、CommonNetworkApi.cc 与 wscllm 蓝本基线逐字节相同 → 直接复制蓝本改造版（deferred flush 模式 + sim_recv in_invoke_context 分流 + LinkCongestionSnapshot）。
- 单测 event_queue_deferred_test.cc：ReferenceQueue 改 map 语义（try_emplace 合并）；用例 A/B/C/D 全过 + **用例 E（map 特有）**：tick-end 内 schedule_event(current_time) 下一轮 proceed 触发 :31 断言（fork 验证 SIGABRT）；future 事件正常落 map。
- 静态基线字节门：runall 后 raw_metrics 184 行一致（diff=0）。

### 1-0 机制层移植（2026-08-16）✅
- `astra-sim/workload/execution_driven/` 整目录复制自 wscllm（CompletionObserver/WatchRegistry/DecisionMailbox/DecisionBridge/NodeStore/GraphSource/ServiceCoordinator/RequestIngress/GraphBatchCommitter/OnlineCli/OnlineStatsCounters/WindowedTraceReader/ExecutionMode + tests/）。
- sh_2.0 扩展：
  - GraphSource.hh OnlineNode 增 `is_local_hbm_kv_restore` 位；NodeStore ETFeederGraphSource MemLoad/MemStore 读 attr；GraphBatchCommitter commit 解析该字段。
  - Workload.hh/.cc：以 wscllm 改造版为基合并 sh_2.0 delta——LocalHbm 装配（两模式同装，B11）、issue() MEM 分支按 is_local_hbm_kv_restore 分发、issue_local_hbm_kv_restore(NodeView)（metrics hook + model 路径 + fallback）、issue_comp LocalHbm compute 路径（runtime_ns!=0 时走 register_event 校准路径）、结束判定增 hbm_dma/has_active_jobs 条件（仅静态块内）。
  - **合同⑦裁决④落地（replay 作用域）**：issue_remote_mem / issue_local_hbm_kv_restore 在 online+replay_clock 下即时完成（1ns General），strategy/静态真实物理。
  - HardwareResource：NodeView 重载 + hbm_dma 分类（count-based occupy，replay bypass 下并发；strategy 经 is_available 串行）。
  - Sys.hh/.cc：执行模式工厂三参（Static 默认，静态路径字节不变）。
  - Statistics/MetricCollector（含 online_register_* 动态锚点族）：基线相同/纯增量 → 直接套补丁。
  - CMake：根 GLOB 增 execution_driven/*.cc + include 路径；frontend CMakeLists 复制蓝本（8 目标：静态/Online/6 fixture）。
  - main_online.cc：蓝本逐字复制（静态 main 两仓逐字节相同，初始化面一致；含 AnalyticalRemoteMemory 装配）。
- 全部目标编译通过；**静态基线字节门再验证 diff=0**。

### 1-3 CompletionObserver + 两条自驱动链 fixture（2026-08-16）✅
- 三终态路径挂载随 Workload 移植落地（record_node_terminal 静态 request/stage=nullptr）。
- completion fixture 扩展：node3 MEM_LOAD(remote) 仅 26 个 edge rank（端口可达性；非 edge rank 无端口 fail-closed）、node4 MEM_LOAD+is_local_hbm_kv_restore（LocalHbm 链）。54×4+26=242 完成全部恰好记录一次；site 分类 skip/generic(108=comp54+hbm54)/coll54/remote26；ALL PASS。
- 第四条终态入口审计：grep register_event/->call 全仓——LocalHbmBandwidthModel(.cc:285)与 AnalyticalRemoteMemory(.cc:195)均经 workload->call wlhd 分支汇入，无绕过路径（与方案 §4.3.1 预期清单一致）。

### 1-4/1-5/1-6/1-7 机制单测（2026-08-16）✅
- WatchRegistryTest / DecisionMailboxTest / GraphBatchCommitterTest / WindowedReaderTest ALL PASS（蓝本 fixture 直用）。
- BridgeLoopbackTest 未跑（需 python echo 脚本，阶段 1 收尾补）。

## 阶段 1（Python 在线层）

### 1-8 Python 层（2026-08-16）
- `online/`：decision_bridge/replay_source/online_scheduler_base/checkpointing/verify/* 复制蓝本；新写：
  - `graph_batch_builder.py`（本仓核心对齐件）：OnlineTraceBuilder 实现 TraceBuilder 全接口面（含 mem_store/mem_load/local_hbm_kv_restore/arm_dependency/chain_checkpoint/restore_chain/timer_gate 量化语义），直接驱动离线 `_emit_kv_transfer/_emit_tp_readiness_barrier/_emit_tp_point_to_point_readiness_barrier/transformer_pass_aggregated`（只读 import）；三段发射（prefill/decode/completion）；pending_history/pending_request_by_session/deferred_session_locations/mark_pending_history_store 账本与离线 writer 同构；emitted-ranks-only 块末恢复；replay 清 prefill 组跨 request previous_id；相位校准 _calibrate_phase（marker 作用域）。
  - `sh20_replay_scheduler.py`：登记式消费 + 相位校准 + future alarm（replay 权威 tick）+ **发射对齐 alarm**（本仓新增，见下）。
  - `sh20_online_scheduler.py`：plan_face_requests 逐行迁移（注释标离线行号）——PREFILL_DRAIN 收尾链（expand_prefill→select_decode_instance→reservation 迁移→move_prefill_to_decode→expand_decode→allocation→release→active_decode.append）调用顺序逐行一致；completion 收尾（mark_complete→enforce_reserve→快照）；arrival（truncate_history→pending）；try_admit_request 全复刻（HBM 过滤/PARTIAL 亲和/task-load 选择/reserve/prepare_prefill）；LUT 标定常数构建在线消费；task-load 三分量在线复刻（request-aggregated 口径，公式同源）。
  - `online_service.py`：replay manifest 从 decision_log 三流合并（标识字段从 config 队列补齐）；strategy manifest 从 config 队列经离线同源推导。
- runner：run_online_replay.sh / run_online_strategy.sh（蓝本同构；ET_PREFIX/runtime_config 指向本仓）。

### 阶段 1 E2E 调试链（bug 总表，全部已修复）
| # | bug | 根因 | 修复 |
|---|-----|------|------|
| 1 | shell-config eval 语法错 | fail-closed print 进 stdout 被 eval | print → stderr |
| 2 | KeyError prefill_length | 合并 manifest 缺输入事实 | config 队列补齐 |
| 3 | invalid stage: completion | committer stage 闭集 {prefill,decode} | completion 段归 decode |
| 4 | generation 2 != expected 1 | 同上 generation 闭集 | generation=1 |
| 5 | node/watch coverage 不匹配 | 三段发射：completion 段无 watch | committer 扩展：node-only 段允许（decode: prefill_drained 或本 delta REQUEST_COMPLETE；prefill: in_flight）——sh_2.0 多段发射机制扩展，登记合同① |
| 6 | pending_history KeyError | 漏 offline :2258 的 pending_request_by_session.pop | 补 pop |
| 7 | past arrival_world_ns | interval==0 turn 在线时钟数 ns 漂移使计划 tick 落到当前之前 | alarm 钳制 max(tick, now+1)（ingress 迟到钳制同款） |
| 8 | strategy AttributeError ×2 | _RequestRuntime 字段面（request_id/hbm_wait_ns） | 修正访问路径 |
| 9 | assignment null/negative | strategy 在 prefill 发射时 decode 实例未知 | assignment 移到 decode 决策后追加 |
| 10 | B1 序列 174 处反转 | turn-0 准入排队秒级（arrival≠admission tick），发射在 arrival 边界 | **发射对齐 alarm**：首达只登记+注册 future alarm 对齐 record tick，第二次 ARRIVAL 才发射/记日志（蓝本 replay_source 合法重复语义）；配套 committer alarm 校验扩展（in_flight 但未 drained 允许对齐 alarm）+ base 重复到达放行（prefill 仍在待办） |

### E2E 结果
- replay（/tmp/sh20/replay10，对齐机制前）：**PASS**——completed=1177/1177、no_decision=0、watch stale=0、single_node_bridge=0、graph_batch==delivery_count=3521、328,976 节点、峰值 RSS 414,708 KiB、late arrivals 0；B2 决策内容按 identity 比对 0 差异；B1 序列 3010/3531 位序一致（174 处相邻反转→对齐机制修复中，replay16 运行中）。
- strategy：真实物理运行（replay 校准关闭）——strategy6 后台运行中。

## 阶段 1/2 收尾验证（2026-08-16）

### 最终验证矩阵（全部通过）
- 静态基线字节门：清理调试埋点重建后复验，raw_metrics 184 行 diff=0。
- C++ 机制单测：EventQueue(map, A/B/C/D/E)、WatchRegistry、DecisionMailbox、
  GraphBatchCommitter（B/D 用例按 sh_2.0 对齐 alarm 扩展更新语义）、
  WindowedReader、BridgeLoopback（echo 回环）——ALL PASS。
- CompletionObserver fixture：54×4+26=242 完成（含 AnalyticalRemoteMemory
  与 LocalHbmBandwidthModel 两条自驱动链各恰好一次）。
- IDLE/注入 fixture：五态迁移 IDLE→ACTIVE→IDLE→DRAINING→FINISHED 全齐
  （注入 2 request、显式 close、退出 0）。
- pytest：36 passed（两次，含最终状态）。

### replay 最终结果（/tmp/sh20/final_replay，PASS）
- completed=1177/1177、no_decision_python_callback_count=0、
  watch_stale=0、single_node_bridge_count=0、graph_batch_count==delivery_count=3521、
  ingress_overflow=0、late arrivals=0、峰值 RSS 414,708 KiB、
  tick_end_without_decision_count=129,112（允许非零，单独报告）。
- **Tier B**：
  - B0：digest 1177/1177 逐行重算一致（sidecar_restore、无 prefix 双计）；
  - **B1（决策序列 order+tick，权威 record 键）与 B2（决策内容）：3531/3531
    逐行完全一致（容差 0）**；
  - B3（计数层）：在线总节点 328,976 == 离线 manifest total_nodes 328,976
    （逐节点 canonical 比较为后续项；在线构图直接复用离线 emitter 函数，
    节点属性/顺序/rank ownership 与离线 .et 同源）；
  - B4：replay 时钟口径（comm/MEM/HBM-DMA 即时完成 + COMP 校准 + 跨 request
    previous_id 清除）为合同⑦登记的刻意差异（归因类别①③）；
    sim_end 684.8s（LUT 对齐时钟）vs 离线静态 2026s（真实物理）——差异
    全部可归因于已登记类别。

### bug #11（run-end 活锁）与修复
- 现象：run 末尾 C++ 100% CPU 空转（无事件、无交付）。
- 根因：ServiceCoordinator::wait_for_work 的谓词含 `!input_open_`——输入
  关闭后立即返回；队列空 + svc 未 finished（active 卡住）即忙等死循环。
  蓝本潜在缺陷，wscllm 未踩中（其 run 末态 finished 先成立）。
- 修复：main_online 主循环加 livelock guard（input closed + 队列空 +
  mailbox 空 + 未 finished → 带诊断 fail-closed 退出，含 per-rank
  pending/free 状态转储）。触发源（对齐 alarm 双计 active）已随对齐机制
  回退移除；guard 保留为机制层防护。

### 裁决与机制扩展登记（sh_2.0 特有，供 sh_3.0 迁移）
1. GraphBatchCommitter 覆盖规则扩展：node-only 段（completion 段在
   REQUEST_COMPLETE 边界提交，watch 已在此前 epoch fire）——decode 段要求
   prefill_drained 或本 delta REQUEST_COMPLETE；prefill 段要求 in_flight。
2. future_alarm 校验扩展：in-flight 但未 drained 的 request 允许对齐 alarm
   （原 in-flight 即拒）。
3. B1 口径：replay 在线决策日志行携带权威 (record_tick, record_seq)，
   online_service 排序出口按其重排——sh_2.0 秒级 turn-0 准入排队使到达/
   准入边界天然分离（wscllm 30s 输入 q≤17ns 无此问题）；排序后与离线
   decision_log 逐行一致。
4. interval==0 的 turn：在线时钟数 ns 漂移使下一计划 tick 落于当前之前
   → alarm 钳制 max(tick, now+1)（迟到钳制同款）。

### 遗留问题（open）
1. **strategy 模式死锁（阶段 1 步骤 1-9 验收未完）**：真实物理运行至
   tick≈1.55s、completed=172/1177 后图停滞（queue 空、mailbox 空、active=98）。
   诊断（livelock guard 转储）：若干 rank 存在 free 未发射的
   collective/transfer-ack 节点（comm 槽被在飞 collective 占用），另一些
   rank 全部在飞——疑似跨实例组 P2P（noc_migrate transfer 的
   send/recv 对）+ collective 参与序在在线交错发射下的循环等待
   （离线由全局 Kahn 排序规避）。需要下一轮专项修复（候选方向：
   在线发射序对跨组 P2P 对做与离线 writer 等价的触发门/次序约束）。
   replay 路径（Tier B oracle 验收主体）不受影响。
2. B3 逐节点 canonical 比较器与 B4 执行 tick 级对照报告未完成
   （计数层 + 内容/序 exact 已过；材料已具备：graph_batch_digests +
   离线 .et + 决策日志）。
3. 阶段 3 感知运行（--sensing）/ 阶段 6 基准矩阵：因 strategy 死锁暂缓。
4. same-tick milestone fixture（步骤 1-11）脚本未移植（机制在位：
   T+1 唤醒 + deferred_from_tick 记录，main_online 已含全部代码路径；
   专用 fixture 脚本待补）。

### git 状态
- 未提交（执行纪律：主 agent 指示不 commit）；全部改动在工作树。
  红线自检：face_scheduler.py 零改动（git diff 确认）；
  generate_face_trace.py 改动 = fail-closed 插入 + --replay-record 记录
  通道（观测性，生产产物字节门两轮验证）；LocalHbmBandwidthModel /
  AnalyticalRemoteMemory 与基线逐字节相同（修复调试损伤后 diff 确认）。

## strategy 死锁：裁决与修复（2026-08-16，主控指令轮）

### 步骤 1：循环证明（waits-for 证据，sdg1 运行）
livelock guard 增强（槽位持有者 + free 头部）后转储：
- rank0（实例0）：comm 槽被 node41（在飞 collective）持有；free 头 node50
  （session_4 all_reduce）等槽；rank6（实例1）对称：槽持 node50、free 头
  node41（session_2 end_barrier）——同一对 collective 在两实例上相对次序
  相反；
- 两 rank 的 free 头部还含 `prefill_decode_transfer_noc_migrate_*_ack_to_*`
  P2P 节点（被本 rank 槽反压）。
环：instance0.node41 ⇄ (同组 peer 的参与被 noc_migrate ack 链挡住) ⇄
ack 被 rank0 槽（node41）挡住。**结论：真循环等待，非单边缺陷。**

### 根因（与主控机理分析第 3 点一致）
`emit_decode_batch` 的 emitted-ranks-only 块末恢复（蓝本 replay 机制被
误用于 strategy）把 decode 组 rank 的 per-rank frontier 链断开（恢复到
本 request 的 prefill 块末或 None）→ per-rank 发行序 ≠ 全局发射序 →
跨实例 P2P + collective 参与序在两 rank 上反转。离线全局 Kahn 序保证任意
两 plan 在所有 rank 上相对次序一致——这正是缺失的不变量。

### 修复（发射次序类，符合修复约束）
`graph_batch_builder.emit_decode_batch`：仅 replay_clock=True 保留块末恢复
（LUT 时钟语义，B1/B2 exact 依赖）；strategy（真实物理）不恢复——
per-rank previous_id 接续到当前 frontier（= 离线 writer 跨 request 物理
链同构；decode 组首节点的跨 request 边为离线 .et 既有结构，B4 归因
类别②既有口径，非新增差异）。

### 修复后验证
- **strategy 全量 1177/1177 PASS**（/tmp/sh20/sfix1）：sim_end 1,678s
  （真实物理，与离线静态 2,026s 同量级），watch stale=0、
  single_node_bridge=0、no_decision=0、late arrivals=33（钳制，计数可观测）；
- 回归：replay PASS + B1/B2 exact 复验（3531/3531）、静态字节门 diff=0、
  pytest 36/36；
- 结构审计（--dump-nodes + b3_node_compare --frontier-check）：
  **cross_request_breaks=0**（不变量成立）；42 处断链均为同 request 内
  并行恢复分支（离线 restore_chain 同构，合法）。

### 同轮补齐
- **B3 逐节点比较器**（online/verify/b3_node_compare.py）：per-rank
  (name,type) 多重集 + 属性 + P2P 配对一致性 + frontier 审计。
  **PASS**：54 rank 全一致、328,976==328,976、属性 0 差异、配对 0 错。
  途中对齐两处命名（turn-0 短前缀 gate 名；transfer action{seq:03d} 计数
  ——per-request 跨段共享计数器，离线同构）；tag 为协议层令牌按配对
  口径比较（绝对值因发射序不同必然不同，登记口径）。
- **B4/Tier B 报告**：online/verify/tier_b_report_20.md（四归因类别 +
  benchmark 矩阵 + 机制差异登记 6 项）。
- **same-tick milestone fixture**（run_online_same_tick_milestone.sh +
  same_tick_milestone_fixture_service.py 移植）：四断言全过（单 tick 单次
  delivery、显式 T+1 唤醒 deferred_from_tick 记录、下一 epoch 交付、
  零丢失无死锁）。
- **阶段 3 感知运行**（run_online_strategy_sensing.sh 移植）：1177/1177
  PASS，ledger.jsonl/sensing_query_log.jsonl 产出。ledger_reconcile.py
  为 wscllm 工件格式耦合（R2/R4/R6 过；R0c2/R1/R3/R5/R7 需要 sh_2.0 的
  kv_actions 流与完成事实格式适配——sh20 调度器未填充 kv_actions 批字段，
  admitted 层口径差异）——登记遗留，非状态不一致（运行侧全部审计通过）。
- benchmark 矩阵（报告内表）：三模式 callback==delivery、py_sched
  2.7-5.6ms/delivery、峰值 RSS 415MiB。

### 更新后的遗留清单
1. ledger_reconcile 的 sh_2.0 工件适配（kv_actions 批字段发射 + 完成事实
   口径）；
2. B4 执行 tick 级逐节点对照（当前为归因口径 + 计数/集合等价）；
3. 阶段 7 裸仓库还原与 commit/tag 补齐（sh2-phase0..7-done）；
4. 换 IPC 桥接的性能优化（阶段 6 决策留待主控）。

## 收尾轮（2026-08-16，主控三项裁决执行）

### 裁决 1：ledger_reconcile sh_2.0 适配 ✅
- 调度器侧补齐：kv_actions 批字段（_kv_action_rows：history/prefill/
  decode/completion 逐出与迁移，KVTransfer 字段同源）+ strategy 决策日志
  （prefill/decode/completion 三流，sense2 运行 3531 行）。
- 新对账器 `online/verify/ledger_reconcile_sh20.py`（sh_2.0 工件口径）：
  **R0(a/b/c)/R1/R2/R3(a/b)/R4/R6 全 PASS，RESULT=BALANCED**；三类残差
  逐条归因（R1 commit-tick 接线缺口——提交层由 C++ graph_batch==
  delivery==3531 对平；R3 KV 事件流不适用——B2 oracle 承担；R5/R7
  end-barrier 自完成尾部——sh_3.0 合同① 同口径）。

### 裁决 2：B4 逐节点 tick 对照（归因口径）✅
- `online/verify/b4_tick_compare.py` + `b4_tick_report_20.md`：相位终点
  语义配对（PREFILL_DRAIN↔decode 记录、DECODE_COMPLETION↔completion
  记录）；**节点级 2354 条中位漂移 -1ns**（校准精确到舍入），completion
  边界 ±21ns；正向长尾归因 completion 段 node-only 尾（类别④）；四类别
  归因齐备，不可解释差异 = 0；exact 条款按 §9.2 不适用（提交序列不同）。

### 裁决 3：裸仓库还原 + commit/tag ✅
- 删物化输入（traces/*.csv）与基线归档（baseline/）；trace_config.csv
  中性化（PLACEHOLDER + PROVENANCE 指引）；traces/ 保留 PROVENANCE.md +
  materialize_first_30s.py（重建入口）。
- 测试 fixture 化：8 个 config 依赖用例 skipUnless(物化输入存在)——
  **双根 pytest 28 passed / 8 skipped 全绿**；fail-closed 实测
  exit=1（PLACEHOLDER 路径）。
- commit/tag 补齐：`sh2-phase0..7-done` + `sh2-final-done`（9 个 tag，
  9 个 commit，pathspec 提交，本仓工作树 0 残留）。

### 裁决 4：IPC 桥接优化——登记不执行（与 sh_3.0 PARTIAL 口径一致）

### 终态剩余差异登记（如实）
1. frontier 接续修复（strategy 跨 request 物理链=离线同构；replay 保留
   LUT 语义的块末恢复）——已入 B4 归因类别②；
2. 文件桥 v0（bridge_ns≈19.8s/3521 次往返）——IPC 优化项 PARTIAL 登记；
3. wait_for_work 蓝本缺陷（!input_open_ 忙等）——livelock guard 修复，
   待回灌 wscllm/face 统一评估；
4. sh_2.0 特有机制扩展 3 项（committer node-only 段/对齐 alarm/迟到钳制）
   ——sh_3.0 迁移输入；
5. 静态字节门/pytest/replay Tier B 在收尾改动后为裸仓库态（输入删除），
   最后一次全量验证证据：本实录前文（diff=0/36 绿/B1B2 exact/B3 PASS）。

## 回灌修复轮（2026-08-16，四档 3 分钟对比测试 §5 缺陷回灌）

依据：`sh_2.0测试/对比报告.md` §5.1/§5.2/§5.3 + 测试副本
TEST_ADAPTATIONS.md（补丁在 4 份 3 分钟副本验证后回灌本仓）。

### 回灌项 A：OnlineCli 默认窗口 fail-closed 化（§5.1）
- **源缺陷机理（比报告更进一步定位）**：默认
  `--request-max-arrival-ns=30e9`（OnlineCli.cc:57）把 30s 验收输入的
  窗口假设冻进机制层；更长输入的超窗 turn-0 行被 WindowedTraceReader
  拒绝后**永不消费**（拒绝行不再触发到达 alarm）→ 窗口被未消费行塞满
  于 high_water → **EOF 永不可达** → expected_requests 恒 0 →
  completed==expected / single_node_bridge / delivery_count 全部
  expected-scoped 审计被静默跳过 → 实测丢 12.5% 仍 exit 0 PASS。
- 修复（三件套，机制层）：
  1. 默认改**无界**（OnlineCli.cc/.hh `request_max_arrival_ns = 0`）；
     窗口语义保留为显式参数（非 0 = 显式窗口）；
  2. WindowedTraceReader 拒绝行**拒绝即消费**（行序读取下即推进消费
     前缀；窗口不再被拒绝行堵塞，EOF 可达 → 完成度审计分母
     = data_rows = 物化输入总行数）；
  3. main_online run-end 新增 **input window audit**：CSV 驱动 run 若
     `rejected_out_of_range != 0` 或 `!csv_eof` → 打印丢弃明细计数
    （rejected/窗口值/rows_read/data_rows/accepted/completed）→
    gate_ok=false → 非零退出（fail-closed，不再静默）。
- 测试：cli_online_test 默认断言改 0；windowed_trace_reader_test 新增
  **Part G**（合成 fixture，40/60/90s 任意到达窗口：(i) 默认无界全接受
  rejected=0；(ii) 显式 30s 窗口 3 行全拒+计数、EOF 单泵可达、分母=
  总行数）。**集成实测**：合成队列 + `--request-max-arrival-ns 30e9`
  → cpp **exit=1** + 审计错误行含全部丢弃计数（对照：默认无界的主
  replay/strategy run rejected=0 PASS）。

### 回灌项 B：在飞 turn-0 逐出 KeyError（§5.2）
- 补丁回灌 `online/graph_batch_builder.py`
  `_mark_pending_history_store`：pending_request_id 无 gate（turn-0 在
  prefill 发射登记 session 键、gate 为内联 new_session 不入账本）时改挂
  deferred_session_locations（与 turn>0 无登记语义一致，完成时折算），
  与测试副本验证版逐语义一致（仅注释措辞差异）。

### 回灌项 C：runner REQ_COUNT ARG_MAX（§5.3）
- `run_online_replay.sh` / `run_online_strategy.sh` REQ_COUNT 诊断行
  `ls bridge/request_*.json` → `find -maxdepth 1`（>2e4 envelope 时
  glob 展开 E2BIG → set -e exit 126；与测试副本同款）。

### 附加回灌（复验必需，测试副本 TEST_ADAPTATIONS 第 5a 条同款）
- 两 runner 的 ET 目录硬编码（30s 验收期 config digest 目录名）改
  **glob 动态解析唯一 generated 目录**：config digest 随 trace_config
  字节变化（复验轮物化文案与验收期不同 → 目录名 c0441ce42→c793f7a52），
  硬编码必断。runner 层改动，非策略红线。

### 复核 D：teardown `[critical] unreleased nodes` 定性（§4.3/§7 遗留）
- **结论：退出期在飞尾节点的抱怨性日志，非真实泄漏，不修**。证据：
  1. 输出来自 HardwareResource **析构函数**（main `delete systems` 时），
     每个受影响 sys 恰持 **1 个** GPU comm 槽，且相邻 TP 对/多 rank 共享
     同一在飞 transfer（本轮实测：replay sys 2/3（node 14050/13802 对）；
     strategy sys 40/41、46/47、52/53 五 rank 同持 node 6010 + 单 rank
     6037）——run 尾部最后一批 transfer 的 send/recv 腿；
  2. main 循环终局权威 = svc.finished()（请求级，设计语义）：末批
     REQUEST_COMPLETE fire 后循环即断，尾节点完成事件仍在队列未处理，
     槽位未释放 → 析构抱怨；
  3. 全部完成度/watch/bridge 审计通过（stale=0、registry 空、
     graph_batch==delivery）；3 分钟档最大 11,121 请求 100% 完成——
     运行中槽泄漏会阻塞后续发射（livelock），与观测矛盾；
  4. 静态路径（全 ET 排空后退出）无此输出；进程退出 OS 回收，非内存
     泄漏。不修理由：改主循环退出时机会扰动冻结的确定性基线
     （sim_end_ns/tick 计数逐位复现门）。

### 复验（20.csv 前 30s，全序通过）
1. **物化**：materialize_first_30s.py 重跑 → 112/1177、
   average_decode_length=459.4944774851317；产物 md5：queue
   `6aaf28365c783c01630278e2e7ce2c89` / context
   `25b7da357cac249c14bf4a3ae26ddfd4` / digest
   `476a02d0b276bc66abda32663b74a173`（确定性）。
2. **编译全目标 + C++ 单测**：8 目标全过；WindowedReaderTest
   A-G ALL PASS（含新 Part G）；CliOnlineTest R1-R11 PASS。
3. **双根 pytest**：物化态 36+33 全绿；裸仓态复验 28 passed/8 skipped
   +33（登记口径一致）。
4. **runall 静态**：exit 0；raw_metrics 184 行与验收期归档
   **剔除 run_id/wall_time_ns 后逐行一致（0 差异）**；generated 57/59
   文件字节一致（manifest 仅 source_config_digest/路径派生 4 键差异=
   config 文案差异）；sim_end_ns=2,025,985,021,143（≈实录 2026s）、
   1177/1177、incomplete=0。
5. **--replay-record**：decision_log 3531 行与验收期归档**字节一致**；
   ET 复生成字节一致。
6. **replay 全量 PASS**：delivery=3521、completed=1177、stale=0、
   no_decision=0、single_node=0、total_nodes=328,976、
   tick_end_without_decision=129,112、late=0、rejected=0、csv_eof=true、
   sim_end_ns=684,836,133,954（实录 684.8s exact）、峰值 RSS
   414,664 KiB（实录 414,708，负载微差）。
7. **strategy 全量 PASS**：delivery=3531、1177/1177、stale=0、
   total_nodes=317,726（实录 exact）、sim_end_ns=1,678,453,667,807
   （实录 1678s）、late=33（实录 exact）。
8. **Tier B 不回退**：B1+B2 **3531/3531 逐行 exact**（kind+request_id+
   record_tick+decision 内容全等）；B3（b3_node_compare
   --frontier-check）PASS（328,976==328,976、canonical 属性 0 差异、
   P2P 配对 0 错；frontier cross_request_breaks=16446 为 replay LUT
   块末恢复既有口径=归因类别②，strategy 侧 0 不变量未动）。
   注：tier_b_compare.py 含 wscllm 冻结计数（214450/7932/5935），
   本仓 B1/B2 一直以 record 键逐行对照承担（同实录口径）。
9. **合成 fixture**：见回灌项 A（exit=1 + 明细）。
10. **teardown 复核**：见复核 D（本轮 replay/strategy 输出与 3 分钟档
    同款：2 sys/6 sys）。

### git
- commit（pathspec 仅本仓）+ tag `sh2-backport-fix1-done`。

## 补验轮：回灌后 sensing 复跑 + ledger 对账（2026-08-16）

目的：backport-fix1（OnlineCli 默认无界+审计分母、turn-0 逐出补丁、
runner find/glob）落地后，复跑阶段 4 的 sensing 运行与 ledger 对账确认
零回退——turn-0 补丁改动 builder 的 pending-history 记账路径，属对账
检验对象。

### 输入与 ET
- 物化重跑（materialize_first_30s.py，源 md5 fc74a48e874cf7798794d7bb17c426b7）：
  queue `6aaf28365c783c01630278e2e7ce2c89` / context
  `25b7da357cac249c14bf4a3ae26ddfd4` / digest
  `476a02d0b276bc66abda32663b74a173`——冻结值逐字节一致；112/1177、
  average_decode_length=459.4944774851317。
- trace_config :12 指向物化 queue（:13 保持空，sidecar off 语义）→
  config digest cc4a70f62（目录名随 :12 文案变化，与冻结 md5 无关，
  runner glob 解析不受影响）；generate_trace 产出 54 .et + manifest，
  --print-shell-config 1177/112。

### sensing 全量（run_online_strategy_sensing.sh，/tmp/sh20_recheck/sense1）PASS
- completed=1177/1177（incomplete=0）、no_decision=0、
  single_node_bridge=0、watch stale=0、graph_batch==delivery=3531、
  total_nodes=317,726、**sim_end_ns=1,678,453,667,807（实录登记 exact）**、
  late=33（exact）、rejected_out_of_range=0、csv_eof=true、kv_actions=2325、
  future_alarms=1065、tick_end_without_decision=4,097,743（允许非零，
  单独报告）；ledger.jsonl(1177)/sensing_query_log.jsonl(3531)/
  online_decision_log(3531)/digests(3531) 产出；teardown
  `[critical] unreleased nodes` 6 行（复核 D 定性：退出期尾节点抱怨日志）。

### 感知开/关对照（同输入同 ET）
- 关感知 strategy 全量 PASS（/tmp/sh20_recheck/strat1）：全部审计计数与
  sensing 运行一致（sim_end_ns 同为 1,678,453,667,807）；
  **online_decision_log 3531 行与 sensing 运行逐字节一致（cmp=0）**——
  感知只开仪表不改判据的实测（本仓首次登记该对照，与 sh_3.0 阶段 2
  B4 结论同型）。

### ledger 对账（online/verify/ledger_reconcile_sh20.py --run-dir sense1）
- **R0(a/b/c)/R1/R2/R3(a/b)/R4/R6 全 PASS，RESULT=BALANCED**——与回灌前
  （收尾轮裁决 1）登记逐项一致；三类残差类别与条数不劣化（R1-residual
  first_commit_tick=null ×1177、R3 KV 事件流不适用、R5/R7 自完成尾部，
  均同登记口径）。

### turn-0 逐出补丁覆盖查证（graph_batch_builder._mark_pending_history_store）
- **30s 输入实测：补丁分支（在飞 turn-0 会话被逐出）不可达**。方法：
  observation-only monkeypatch 包装（零仓内改动）复跑 sensing——全部
  审计计数与正式运行一致、决策日志逐字节一致（观测无副作用实证）——
  分类计数：83 次 remote_store 全部走 pending 无键的 A 分支
  （deferred_session_locations 原语义）；gate 在账本的 B 分支 0 次；
  **补丁针对的 C 分支 0 次**。
- 覆盖证据登记：该路径在 30s 输入不可达；真实覆盖证据 = 3 分钟档测试
  （`sh_2.0测试/对比报告.md` §5.2，8 run 全过）+ 补丁语义与 turn>0
  无登记分支一致（deferred 折算，完成时消费）。
- **补最小合成单测** `online/test_turn0_eviction_patch.py`（3 用例：
  C 分支 remote_memory/partial_hbm_remote 双位置不抛 KeyError 且挂
  deferred 且账本不破坏；A 分支原语义；B 分支 gate 更新不入 deferred）；
  standalone 与 pytest 双通道 3/3 PASS，裸仓态可运行（合成 KVTransfer，
  无输入依赖）。

### runner 补齐（runner 层，非策略红线）
- run_online_strategy_sensing.sh 回灌轮漏打的两处同款修复：ET 目录
  硬编码（c0441ce42，config digest 变化必断）→ glob 动态解析唯一
  generated 目录；REQ_COUNT `ls` 通配 → `find -maxdepth 1`（ARG_MAX）。
  与 replay/strategy runner 的回灌项 C/附加回灌 5a 同款。

### 复验与收尾（裸仓态）
- 物化态双根 pytest 36+33 全绿；清理（删物化 CSV 与 generated ET、
  trace_config 还原占位）后裸仓态 28 passed/8 skipped + 33 全绿；
  fail-closed 实测 exit=1；traces/ 保留 PROVENANCE.md + 物化器。
- 红线自检：face_scheduler.py / generate_face_trace.py 零 diff；改动面 =
  sensing runner 两处 + 新增合成单测 + 本实录。

### git
- commit（pathspec 仅本仓）+ tag `sh2-sensing-recheck-done`。

## wscllm 同型缺陷排查（054118e 对账工具缺陷，2026-08-16）

wscllm 四档开感知验证发现两个**验收工具缺陷**（均在对账脚本
ledger_reconcile.py，不在仿真/调度本体），已修进 wscllm（054118e）。本仓
同型排查结论与处置：

### 排查结论（逐缺陷）

- **缺陷 1（陈旧契约：读消费即删的 response_*.json）**
  - `ledger_reconcile_sh20.py`（本仓权威对账器）：**不成立**。数据源自
    诞生起即归档产物（ledger/online_decision_log/graph_batch_digests/
    sensing_query_log + cpp.log），从不读 response 文件。30s 补验轮与
    20 档 3min sensing（2091 请求）对账全 PASS 正因如此（runlogs/
    reconcile_20.log，RESULT: BALANCED）。
  - `ledger_reconcile.py`（迁移继承的 wscllm 修复前版本，md5 与 wscllm
    054118e^ 完全一致）：**成立**。load_responses 读 response_*.json；
    本仓 decision_bridge.py ack 后 os.remove（与 wscllm phase-7 §10.3
    同款消费即删），run 结束 bridge 目录只余 request_*.json（30s 与
    20 档 3min 产物实测 response 计数恒 0）→ 跑旧版必得 R2/R3/R5b
    空数据源 FAIL。
- **缺陷 2（错误不变量：watch ∈{0,1}）**
  - `ledger_reconcile_sh20.py`：**不成立**。R2 为总量不变量
    watches==2×N（与 wscllm 修复后语义不变量一致），无逐批 {0,1} 假设。
  - `ledger_reconcile.py` 旧版：**同型成立**（R2a 假设每 (request,stage)
    恰注册一次 watch）。
  - 本仓 digest 语义校准（054118e 修法移植的关键差异）：
    graph_batch_digests 的 ranks=批交付图全部节点 rank 集
    （online_scheduler_base._digest_row），非 wscllm 的批内 committed
    rank 集 → wscllm 修复版的"watch_count==0 ⇔ ranks 空"空性等价在
    本仓【不成立】（DECODE_COMPLETION+REQUEST_COMPLETE 完成通知批
    有节点恒 0 watch；30s/3min-20 档全量实测该判据合法违例 1036/1936
    条），只可移植上界判据 `0<=watch_count<=len(ranks)`（实测零违例）。

### 处置（verify 工具层，仿真/调度本体零改动）

1. `ledger_reconcile_sh20.py` 吸收 054118e 修法加固：R2 拆 R2a（digest
   watch 和 == cpp phase-5 total_watches == 2×N 三方对平）/R2b（批级
   0<=wc<=len(ranks) + ranks⊆[0,53]，**不含**空性等价并注明原因）/R2c
   （digest node 和 == cpp total_nodes）；R3 增 R3c（completion 决策
   tick == ledger completed_tick 逐对零失配 + cpp total_kv_actions >=
   请求数）。cpp.log 缺 phase-5 行时宽限跳过（兼容旧产物）。
2. `ledger_reconcile.py` 整体替换为 ledger_reconcile_sh20 的薄入口
   （数据源=归档 jsonl，即缺陷 1 修法；兼容 --run-dir / --bridge-dir /
   --online 三种调用，吞并旧版 --manifest/--cpp-log/--report），文件头
   完整登记两个同型缺陷的证据链。
3. 同步 4 个 3min sensing 测试副本（sh_2.0测试/astra-sim-sh_2.0_{20,50,
   80,120}_3mins_sensing 同路径两文件，md5 与主仓一致）——矩阵对账
   直接可用。

### 复验

- 30s 补验轮产物（/tmp/sh20_recheck/sense1，1177 请求）：修复版
  R0-R6 全 PASS（新增 R2a/R2b/R2c/R3c 全 PASS），RESULT: BALANCED，
  与补验轮结论一致。
- 20 档 3min sensing 产物（2091 请求）：sh20 直跑 + 薄入口 --bridge-dir
  式 + --online 式三通道全 BALANCED（R2a watches=4182==2×2091==cpp；
  R2b 零违例；R2c 569252==cpp；R3c 零失配）。
- 副本仓内脚本相对路径复跑同 PASS；py_compile 全绿。

### git
- commit（pathspec 仅本仓）+ tag `sh2-reconcile-fix-done`。

## diff_explainability 缺陷 3 排查修复（kv_actions/assignments 真空通过，2026-08-16）

wscllm/face 开感知验证追记立档的缺陷 3（`online/verify/diff_explainability.py`
kv_actions/assignments 真空通过假绿，与缺陷 1 同源：phase-7 §10.3 起
response_*.json 消费即删，`_load_run()` 仍从 response 文件收集
kv_actions/assignments，两侧恒读成空列表，空==空打出 PASS，签名
sha256=4f53cda1… 即空数组 [] 的哈希——比较实际没有发生）。

### 排查结论

- 本仓 `diff_explainability.py` 与 wscllm/face 同名文件逐字节一致
  （md5 08db8c12，427 行同段陈旧代码）：**缺陷成立**。
- 实证复现（缺陷版 + 真实产物）：/tmp/sh20_recheck（本仓 30s 开/关
  对，1177 请求）与 face测试 face_20_3mins（20 档 3min 开/关对，2091
  请求，response 计数实测 0）喂缺陷版均打出 `PASS kv_actions 一致
  sha256=4f53cda18c2baa0c / PASS assignments 一致(同签名)`——假绿
  成立；同时决策行维度 3531/6273 行逐字节一致为真实比较（数据源归档
  jsonl 未受影响），与立档描述一致。

### 处置（verify 工具层，仿真/调度本体零改动）

按 wscllm 054118e 同思路整体重写数据源段：

1. **kv_actions 维度迁** online_decision_log.jsonl 的 KV 决策载荷流
   （每行 decision 字段投影 {kind, request_id, decision}，tick/seq
   无关；载荷字段各仓调度器不同——本仓 *_eviction_count 族——按载荷
   整体做稳定摘要，不耦合字段名）。
2. **assignments 维度迁** graph_batch_digests.jsonl 批级摘要流
   （delivery_sequence/reasons/ranks/node_count/edge_count/
   watch_count/content_sha256；content_sha256 即批内 nodes+
   parent_edges 内容摘要）。
3. **空数据源 fail-closed（禁止空==空通过）**：jsonl 缺失（bridge/
   results 两布局均无）或 0 行、决策行缺 decision 载荷、bridge 无
   request_*.json、cpp.log 缺失或未解析出任何门计数器——一律报错
   退出 2，不产出报告；jsonl 位置解析 bridge/ 优先、results/ 次选
   （官方 runner 运行后把 jsonl 归档移入 results/，两种布局实测内容
   一致）。
4. 同步 8 个测试副本（sh_2.0测试/astra-sim-sh_2.0_{20,50,80,120}_
   {3mins,3mins_sensing} 同路径文件，md5 与主仓一致，5f72c396）。

### 复验

- 缺陷签名消失：30s 对（sh20_recheck）修复版打出 `KV 决策载荷一致
  3531 行 sha256=119f7c84a3d1610e / 批级 assignment 摘要一致 3531 批
  sha256=03fdf7e7d7385202`；face 20 档 3min 对 6273 行/6273 批
  （41deb73f/1c67a6f6）；50 档 13941 行、80 档 22269 行同 PASS——
  维度真实比较出结果，全部一致（与"决策序列逐字节一致"结论互证）。
- 真实差异可检出：篡改一侧 1 个 prefill_instance_index 载荷字段 →
  KV 载荷维度 DIFF（两侧哈希 119f7c84 vs 92ba25fb）——空==空不可能
  产生不同哈希，比较为真。
- fail-closed 5 例实测（缺 bridge/、jsonl 两处皆缺、jsonl 0 行、
  cpp.log 缺失）：全部 exit 2 + 明确报错。

### git
- commit（pathspec 仅本仓）+ tag `sh2-diffexp-fix-done`。

## 在线机制层竞态缺陷修复同步（2026-08-16/17，自 face 0049ef5 / face-defectfix2-done；tag：sh2-defectfix2-done）

> 来源：face 四档 3 分钟主动测试暴露的三类在线机制层竞态缺陷
> （face主动测试错误分析.md；face 修复实录见 face改造执行实录.md
> "在线机制层竞态缺陷修复"节）。本节为 A/B/C 三修复向 sh_2.0 的
> 同步移植与复验实录。本族为"服务终判/等待/死端"机制层缺陷家族，
> face 移植性登记指定 sh_2.0 同步轮做 livelock guard 收敛。
> 授权链：用户 2026-08-16 同步指令（第三组：sh_2.0 单仓）。
> 策略红线零触碰：face_scheduler.py / generate_face_trace.py /
> GraphBatchCommitter（含本仓 node-only 段与对齐 alarm 扩展）零 diff
> （git diff 复核，redline_diff_lines=0）。

### A. 缺陷 A：finished() 计数真空（main_online pump 后补 drain + EOF 审计）

- 移植面：`main_online.cc` 主循环，与 face 0049ef5 同源段落逐段套用
  （含同步出处注释）：
  1. `windowed.pump()` 之后、`svc.finished()` 判定之前，若
     `ingress.pending_command_count() > 0` 则补一次 `drain_commands()`
     ——pump 只把 Submit 入队（下一轮 drain 才 on_alarm_scheduled 注册），
     旧顺序在"上批 alarm 全部触发 + pump 顶部刚补窗"的真空期判终，
     会以 CSV 尾部未交付结束（face F2/F3/F4：末笔 ack!=delivery 恒差 1）；
  2. belt-and-braces：finished() 为真而 CSV 未 EOF → online_fatal
     （携带 active/pending_alarm/queued_commands 计数）——修复后该态
     不可达，防回归保险。
- 回归 fixture（新 CMake 目标 ServiceVacuumTest）：
  `astra-sim/workload/execution_driven/tests/service_vacuum_test.cc`，
  按 sh_2.0 机制层 API 适配（本仓无 face 的 audit_completion/
  total_data_rows——那是 sh_1.0 回灌族；等价断言改用 rows_read/
  data_rows/eof 直接承担"读入未交付"审计证据）。30 行 turn-0 CSV
  （3 批×10、high_water=10、到达即完成）驱动真 WindowedTraceReader/
  RequestIngress/ServiceCoordinator/DecisionMailbox/map EventQueue：
  **legacy 顺序确定性复现真空（断点 completed=10/读入 20、队列 10 条
  未 drain、未 EOF）→ fixed 顺序 30/30 全完成 + EOF + 队列空**，
  ALL PASS。

### B. 缺陷 B：文件桥 resp 通道双侧长连接（EOF 误检/楔死竞态族消除）

- `DecisionBridge.hh/.cc`：**整文件采用 face 修复版（cmp 逐字节一致）**
  ——sh_2.0 两文件与 face 修复前基线本就逐字节相同，移植后仍与 face
  修复版一致（跨仓机制件同源最大化）。语义：open_notify 增设
  `resp_notify_fd_` 常开读端（O_RDONLY|O_NONBLOCK，run 生命周期持有，
  析构对称关闭）；deliver_and_receive 改经新私有 `wait_response_byte`
  在常开 fd 上 poll→read 恰 1 字节（EAGAIN 重 poll）；read()==0 恢复
  真死义（"Python side died (its long-lived resp_notify write end
  closed...)"——旧逐交换握手下同签名也会对活对端误报，face F1）；
  1:1 在途守卫：读到通知字节后若还可读到第二字节 →
  "protocol violation: more than one response byte in flight"
  fail-closed（背压合同防漂移）。
- `online/decision_bridge.py`：按 face 修复段移植并保留本仓调试增强
  （handler 异常 traceback 入 error 响应；与 face 版 diff 仅剩同步出处
  注释与该 traceback 段）。serve_forever 启动一次 `os.open(resp_notify,
  O_WRONLY)` 常开（配对 C++ 常开读端；开序无死锁），_notify_response
  只写不开；写端 BrokenPipe/OSError 包装 `BridgePipeError`（C++ 常开
  读端关闭只能因 C++ 死亡，不可重试）→ serve 循环 fail-closed：stderr
  留痕 "decision_bridge: C++ side is gone (BrokenPipe on resp_notify
  write, seq N)" + sys.exit(1)；_fail 先 stderr 留痕再写 error response
  （修复 F1 现场 python.log 0 字节无可诊断性）。全部 BridgeServer
  消费方（online_service / 同族 fixture 服务）经一处自动继承。
- 回归 fixture：
  - `bridge_loopback_fixture.cc` 新增 Part D/E（整文件采用 face 版）：
    D=Python 完成交换 1 后 SIGKILL 自杀 → C++ 常开读端 abort 且消息含
    "long-lived resp_notify write end closed"；E=一笔响应写 2 字节 →
    1:1 守卫 abort。A-E ALL PASS（须自仓库根运行——Part A 的 echo
    脚本按相对路径 spawn）。
  - `online/verify/bridge_cpp_death_fixture.py`（纯 Python 假 C++，
    第 2 笔交付前死亡）：真实 BridgeServer 撞 BrokenPipe → exit 1 +
    stderr 留痕，连续 **3/3** 确定性。
  - `run_scripts/bridge_race_stress_repro.sh`（旧逐交换模式竞态复现
    装置，"缺陷曾真实存在"的可运行证据）：本机 45s 预算内第 **5** 次
    迭代复现假 EOF（POLLHUP+read==0，写端存活）→ PASS。

### C. 缺陷 C：空队列分支唤醒补全 + 与 bug#11 livelock guard 收敛为统一死端守卫

- `EventQueue.h/.cpp`（map 族）：新增 `has_deferred_work()` const
  访问器（`!deferred_queue_.empty()`）。map 版适配说明：deferred_queue_
  为独立于主 map 的 std::list<EventList>（阶段 1 收口先例：容器无关），
  访问器与 list 族逐字同义；finished() 仍只看主 map——deferred 残留
  对其不可见。静态二进制零行为变化（纯新增 const 查询，静态路径无
  deferred 调度）；event_queue_deferred_test A-E（g++ 对 build 内静态库
  链接）复跑 ALL PASS。
- `main_online.cc` 空队列分支三分支化（与 face 同构）：
  1. mailbox 有残留 → 既有 T+1 显式唤醒（step 1-11 CORE，不变）；
  2. **新**：`has_deferred_work()` 为真 → 同机制调度 T+1 强制下一决策
     边界（proceed() 内排空 deferred 队列；不计 deferred_from_tick）；
  3. input 已关 + 队列/mailbox/deferred 三空而 svc 未完 → **统一死端
     守卫 fail-closed**；
  4. input 仍开 → wait_for_work（IDLE 合同不变，五态迁移 fixture 复证）。
- **收敛（不留语义重叠双 guard）**：原 bug#11 livelock guard（return
  EXIT_FAILURE）与 face lost-wakeup dead end（online_fatal）同位置同条件
  同族，收敛为单一守卫：保留 sh_2.0 的 per-rank waits-for 证据转储
  （strategy 死锁诊断价值，未动）+ "run-end livelock" 计数行，终判改用
  face 同款可归因 `online_fatal("lost-wakeup dead end: ...active=/
  pending_alarm=/deferred_work=...")`（abort，与 face 语义对齐）。
  RED 基线说明：face 的 legacy-stall 模式在本仓结构性不适用——本仓
  bug#11 守卫先于 face 修复已将该死端 fail-closed（这正是收敛的依据）；
  本轮实证的是收敛后语义（fixture 场景 2）。
- 回归 fixture（按仓内命名/路径适配）：
  `run_scripts/run_online_wakeup_guard_fixture.sh` +
  `online/verify/wakeup_guard_fixture_service.py`（ET 目录 glob 动态解析
  + 本仓 runtime_config 路径，同 backport-fix1 5a 款式）：
  - **f7-blueprint**（健康蓝图，防误伤）：未到期 future alarm（r1@T+1000）
    + 同 tick 里程碑 → 两请求全完成 exit 0，accepted=1/completed=2
    （future-alarm 到达不计 accepted）、future_alarm 调度留痕、
    no_decision=0——修复后的空队列分支不误触发；
  - **defer-dead-end**（唤醒真空死端）：空批 defer + CloseInput → 数秒内
    fail-closed abort（exit 134），cpp.log 含 "lost-wakeup dead end"
    （active=1 入消息）+ "run-end livelock" 行 + per-rank 转储——
    替代静默挂死。两场景 ALL PASS。

### 移植适配与既有问题登记（非本轮引入，HEAD 态实证）

- **NodeStoreTest part C 预存失败**：现役 completion_fixture ET 由本仓
  扩展版生成器产出（node3 MEM_LOAD/node4 local-hbm 链，>3 个 dep-free
  节点），蓝本 node_store_test.cc（未随生成器扩展更新）仍断言 3 个。
  stash 复核：HEAD 态（本轮改动前）同失败——与缺陷修复无关，登记
  不修（part A/B 过；修复需同步改测试期望或回退生成器，超出本轮
  移植面）。completion_fixture 本轮由生成器确定性重建（54 文件，
  digest 8ce3a9d0…）。
- **tier_b_compare.py 两个 wscllm 血统残留**（6f0afc1 起未动）：
  ① 顶层 `from generate_wsc_llm_trace import ...`（本仓改名
  generate_face_trace）→ 本次经 PYTHONPATH shim 模块重导出调用，仓内
  零改动；② b0 内嵌 wscllm 冻结计数断言（offline_decision_lines==
  214450、kv_cache_events.csv）在本仓结构性不适用——实录既有口径
  （"本仓 B1/B2 一直以 record 键逐行对照承担"）继续承担，本轮以同款
  record 键逐行对照复跑（见下）。同族 b1 的 wscllm tick-delta 断言
  （prefill 1177 exact 等）同样不适用：本仓秒级 turn-0 准入排队使
  触发 tick 与离线记录 tick 天然分离（阶段 1/2 裁决 3 已登记），
  权威 (record_tick, record_seq) 出口排序才是验收面。
- 运行注记（复现本轮结果用）：replay/strategy/sensing 的
  request_csv 与 decision_log 须传**绝对路径**（online_service 的
  cwd 在 workload 目录）；decision_log 需 `generate_trace.py
  --replay-record` 产出（runall 的 generate 步不记录，runall 后须补跑
  才能起 replay）。

### 复验证据（20.csv 前 30s 包络，全部冻结值 exact）

1. **输入物化**：materialize_first_30s.py 重跑，源 md5 fc74a48e…，
   112 session/1177 请求、average_decode_length=459.4944774851317；
   产物 md5 冻结值一致：queue `6aaf28365c783c01630278e2e7ce2c89` /
   context `25b7da357cac249c14bf4a3ae26ddfd4` / digest
   `476a02d0b276bc66abda32663b74a173`。--replay-record decision_log
   3531 行。
2. **构建与单测**：全目标 0 error（含新 ServiceVacuumTest）；C++ 机制
   单测全绿：EventQueue(map) A-E / WatchRegistry / DecisionMailbox /
   GraphBatchCommitter / WindowedReader A-G / CompletionFixture（242
   完成）/ **ServiceVacuum（新，真空先复现再转绿）** / **BridgeLoopback
   A-E（扩展 D/E）** / cli_online R1-R11；双根 pytest 物化态
   36+3 / 33 全绿。
3. **静态字节门**：runall exit 0（仿真段 10s，全流程 37s），raw_metrics
   **184 行对冻结基线剔 run_id 后逐行 0 差异**——缺陷修复对静态路径
   零扰动。
4. **replay 全量 PASS**：delivery=3521、**ack==delivery==3521（缺陷 A
   不变量，末笔恒等）**、completed=1177/1177、accepted=112、
   no_decision=0、single_node_bridge=0、watch_stale=0、
   total_nodes=328,976、tick_end_without_decision=129,112、late=0、
   rejected=0、csv_eof=true、**sim_end_ns=684,836,133,954（实录
   exact）**、峰值 RSS 414,468 KiB（实录 414,708，负载微差同前）。
5. **strategy 全量 PASS**：delivery==ack==3531、1177/1177、
   total_nodes=317,726（实录 exact）、late=33（实录 exact）、
   tick_end_without_decision=4,097,743、**sim_end_ns=1,678,453,667,807
   （实录 exact）**。
6. **Tier B 不回退**：B1（决策序列 order+tick，权威 record 键）与 B2
   （决策内容）**3531/3531 逐行 exact（容差 0，0 mismatch）**——
   在线 replay 决策日志（record 键排序出口）与离线 decision_log 的
   kind/request_id/record_tick/record_seq/priority/decision 逐行全等。
7. **sensing + ledger 不回退**：sensing 全量 PASS（计数与 strategy 全
   同、sim_end 同 exact、**感知开/关决策日志逐字节一致 cmp=0** 复证）；
   ledger_reconcile_sh20 R0(a/b/c)/R1/R2(a/b/c)/R3(a/b/c)/R4/R6 全 PASS
   **RESULT=BALANCED**，三类残差类别与条数与冻结登记一致
   （R1-residual first_commit_tick=null×1177 / R3 事件流不适用 /
   R5/R7 自完成尾部）。
8. **在线合同回归**：IDLE 五态迁移 fixture ALL PASS（修复后空队列
   分支不破坏 IDLE 等待合同）；same-tick milestone (a)-(d) ALL PASS
   （既有 T+1 唤醒合同未回归）。
9. **发射门控合规**：sh_2.0测试/ 另一 agent 的 120 档 sensing 重仿真
   在跑期间（02:14–04:43）零仿真发射（构建/单测/物化/pytest 之外
   全部等待）；门控复判（mem 34.5%、ps 无在线仿真进程）后才发射
   本轮全部在线运行。
10. **裸仓态恢复与复核**：删物化 3 CSV + generated ET 目录；
    completion_fixture 由生成器重建；trace_config.csv:12 还原占位
    原字节；generate_trace.py fail-closed 实测 exit=1（缺输入明确报错）；
    裸仓态双根 pytest 31 passed+8 skipped / 33 全绿；traces/ 保留
    PROVENANCE.md + 物化器。

### git
- commit（pathspec 仅本仓）+ tag `sh2-defectfix2-done`。

---

## 路径②清理实录（2026-08-18，步骤 1/2：删 replay 路线）

依据：《路径功能代码对应说明.md》。

**删除（②专有）**：`run_online_replay.sh`；`online/replay_source.py`、`online/sh20_replay_scheduler.py`；online_service.py 的 replay 分发/`_manifest_from_decision_log`/权威日志重排出口；generate_face_trace.py 的 `--replay-record` 与 `_write_replay_decision_log`（词法保留显式拒绝）；graph_batch_builder.py 的 replay_clock/清链/emitted-ranks-only 块末恢复与 LUT 校准；C++ replay_clock 全套（Workload.cc :255/:378 remote FIFO 1ns/:405 HBM restore 1ns/:550 comm 1ns）。

**保留（红线）**：face_scheduler/generate_face_trace 被 import 符号（含 `PREFILL_CHUNK_SIZE`、`_validate_and_expand_requests`、`_to_scheduler_requests`、`PendingHistoryGate`）；sh20_online_scheduler 与 frontier 接续死锁修复语义（strategy 无条件接续 frontier——replay 分支删除后语义不变）；tier_b_compare（本仓无 replay 依赖，strategy 对基线口径保留）；turn0 eviction patch 测试。

**验证（2026-08-18）**：重编 PASS（8 binaries）；pytest 64+8skip 全绿；③④冒烟（30s 全量 1177）双 PASS、交付 3531、no_decision=0、③④决策日志逐字节一致；fail-closed exit=1 实测。

---

## 路径①清理实录（2026-08-18，步骤 2/2：删离线静态全管线）

依据与设计：《路径功能代码对应说明.md》（§4-b 替代产出/§7 步骤 2/裁决 A/裁决 i 附三条件）。执行纪律更新（用户 2026-08-18）：全部改动留工作树，不自主 commit/tag。

**替代产出（新增 `plan_materializer.py`，③④唯一输入物化入口）**：
- `load_*_trace_config()` 装载副产 runtime_config 四小件（路径零改动）；
- manifest.json = 队列派生 9 字段（turn0 history=0；turn>0 = min(prefix, 上请求 final)（sidecar 变体）/上请求 final（recompute）；context = input_tokens_total（sidecar）/折入 prefill（recompute t0）/history+prefill（recompute t>0））——**对①产出的真实 manifest 逐字段验证 0 mismatch（1177/1177 或 270/270）**；
- metrics_manifest.json（裁决 i）：schema_version=1 + requests[]，`manifest_source="synthetic-prerun"` 显式标记（条件 a）；prefill/decode instance 恒 0、ranks 恒 instance-0——**静态 rank 归因维度不可信；请求级指标（e2e/完成/sim_end/tput）可信**（条件 b）；对账工具不取本 manifest 决策事实——synthetic manifest 本身不含决策字段（条件 c，结构性满足）；
- 输出目录 `<prefix>_54npus_plan_<cfg8>`（保留 54npus 前缀过 GEN_MATCH）；幂等重跑覆盖。

**删除（①专有）**：
- 入口链：`run_scripts/runall.sh`、`run_sh_test_aware.sh`、`generate_trace.sh`；
- CMake：`AstraSim_Analytical_Congestion_Aware` 静态目标（link/include/properties 段）+ `congestion_aware/main.cc`（裁决 A；上游 examples/run_scripts/analytical/congestion_aware 三脚本随①失效，ET 数据文件保留——登记于本实录）；
- generator ①专有符号（AST 全仓引用面分析驱动，被 import 符号全存活）：write_face_trace/build_face_plan/load_or_build_face_plan/print_shell_config/resolve_output_dir/build_trace_label/ParallelTraceOutputs/StreamingTraceOutputs/_planner_cache_path/_replay_trace_worker 及 _*_dict/_log_dict 快照族（25 符号）；`main()` 改为 fail-closed 拒绝桩（"path-1 removed; use plan_materializer.py"）；
- `generate_trace.py` 的 `main()` 委派段改同款拒绝桩（模块本体整文件保留——Chakra 常量/TraceBuilder/transformer_pass(_aggregated) 等为③④与 microbench 共享符号库）。

**GEN_MATCH 改造**：（本仓 runner 原生 GEN_MATCH，无改造）

**测试处置**：pytest 31+33 通过；8 个失败为**锚点 tag 上的预存缺陷**（test_face_scheduler 期望 checked-in 配置指向已物化 30s 队列，而裸仓库为 request-neutral 占位——锚点态实测同 8 failed，与本次清理无关，登记待后续单独处置）

**验证（2026-08-18，①删除后仅靠 plan_materializer 输入）**：
- 替代产出等价性：合成 manifest vs ①真实 manifest 逐字段 **0 mismatch**；
- 重编 PASS（Online+fixtures，静态目标已不存在）；幸存 pytest 全绿（见各仓数字）；
- ③冒烟 PASS：10s 270 request 交付 810、no_decision=0；④冒烟 PASS 且 online_decision_log 与③**逐字节一致**；
- 分层账本对账：**BALANCED**（10s/270，ledger_reconcile --run-dir --expected 270）；
- fail-closed 实测：generate_trace.py 拒绝桩 exit=1；plan_materializer 空队列 exit=1；runner 缺 request_csv exit=1；GEN_MATCH 零目录/双目录均 exit=1。

**本步工作树改动文件清单（供审阅提交）**：新增 plan_materializer.py；删 runall.sh/run_sh_test_aware.sh/generate_trace.sh/run_online_replay.sh、congestion_aware/main.cc、online/replay_source.py、online/sh20_replay_scheduler.py；
改 CMakeLists.txt、main_online.cc、Sys.cc/hh、Workload.cc/hh、OnlineCli.cc/hh、cli_online_test.cc、generate_face_trace.py、generate_trace.py、online/{online_service,graph_batch_builder}.py、online/verify/idempotency_fixture.py、run_online_strategy.sh、README_COMMANDS.md、本实录。

---

## 路径①②终清与重验实录（2026-08-18，残余清扫轮）

**扫描口径**：同 face 仓。

**本仓改动（终清轮）**：
1. 【功能残留-删】online/online_service.py 的 `--dump-nodes` CLI 参数 + `SH20_DUMP_NODES` 环境门 + online_nodes.jsonl 写出块（~35 行）：其唯一消费方 b3_node_compare.py 已在前轮补清删除，全仓（脚本/测试/文档）零残留引用；属②族验收工具配套转储，删除后经重建+③④冒烟复核零影响；
2. 【活文件陈旧注释-改写】generate_face_trace.py / generate_trace.py 拒绝桩 docstring；online/graph_batch_builder.py（docstring 的 write_face_trace 删除符号引用→generator 模块级函数口径、B3/B4 归因、②清链/LUT 时钟注释、「replay 20 @ delivery 3575」历史事故引用）；online/online_service.py（②删除说明×2）；astra-sim/system/Sys.hh execution_mode_ 注释（replay_clock_ 删除括注）；metrics_integration.py；run_metrics_postprocess.sh 头注释；plan_materializer.py docstring；Workload.cc；execution_driven/tests/ 六文件构建注释；
3. 【活文档陈旧内容-改写】sh_test_mesh/README.md（Static Chakra ET adaptation→在线执行口径；Run and validate→③④工作流，物化器=traces/materialize_first_30s.py）；README_COMMANDS.md 整表重写（对账命令按本仓双工具 ledger_reconcile.py --online / ledger_reconcile_sh20.py --run-dir --expected）；
4. 【边界-保留并登记】test_face_scheduler.py:187 mock 目标 load_or_build_face_plan 为已裁符号——该用例被 _MATERIALIZED 门控，裸仓库态跳过不执行（基线口径维持，重验零新增失败）；ETFeeder 共享边界保留；历史记录载体不动。

**本仓重验数字**：③④冒烟 10s/270 双 PASS：completed=270、no_decision=0、single_node=0、delivery=810==digests 810、③④决策日志 cmp 零差异；④对账 ledger_reconcile_sh20.py --run-dir --expected 270 = **BALANCED**（注：ledger_reconcile.py --online 变体内置 1177/112 验收期望且无 CLI 覆盖，10s 场应使用 sh20 工具）；pytest 双根 31+33（另 8 失败=锚点预存缺陷，与基线同败，零新增）。

**重验（2026-08-18 终清后全量）**：
- clean 重建（rm -rf build_congestion_aware 后 cmake 重配 + 全目标）：exit=0，10 个 add_executable 目标（Unaware/_Online/8 fixtures）全部产出，0 error；
- fail-closed 复测：generate_trace.py 拒绝桩 / 生成器 main 拒绝桩 / plan_materializer 空队列 / runner 缺 request_csv / GEN_MATCH 零目录 / GEN_MATCH 双目录 —— 全部 exit=1；
- pytest 双根：workload 根 + sh_test_mesh/tests 根，与终清前基线逐项一致，零新增失败；
- ③④ 冒烟：run_online_strategy.sh 与 run_online_strategy_sensing.sh 各一场，completed==物化数、no_decision_python_callback_count=0、single_node_bridge_count=0、delivery==graph_batch_digests 行数、③④ online_decision_log.jsonl 逐字节一致（cmp）、④ 分层账本对账 verdict=对平/BALANCED；发射前内存门控实测 ~11-12%（<70%）。
