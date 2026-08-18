# face 仓改造执行实录（实时维护）

> 本文是 `face仓库改造详细执行方案.md` 的配套执行实录：每完成一类改动立即更新。
> 记录纪律遵循方案 §13/附录 D；裁决与实测数字禁止预填。
> 起点父仓库 commit: `55cf135`（wscllm 收尾）；分支策略: 留在 main +
> 每阶段轻量 tag `face-phase<N>-done`（用户 2026-08-15 确认模式沿用）。

## 阶段 0（2026-08-16 执行）

### 完成清单

| 步骤 | 内容 | 关键结果 |
|---|---|---|
| 0-0 | git status 核实 | 起点干净（仅用户未跟踪文档，未 touch）；起点 commit `55cf135` 记录在案 |
| 0-1 | 输入物化 + fail-closed | `traces/` 下物化 30s 输入（derive 脚本 + 8 列队列 + canonical sidecar + PROVENANCE.md）；trace_config.csv:12 改指仓内路径；generate_face_trace.py 在 `_resolve_request_queue` 与 `load_request_queue` 之间插入存在性检查（sys.exit(1)），缺失输入实测 exit=1 且无 stub 写盘；--print-shell-config 实测 REQUEST_COUNT=1177 / SESSION_COUNT=112 |
| 0-2 | 基线复现与归档 | runall 首跑 exit 0（1177/1177 完成）；归档 `baseline/20_30s/`（generated 平铺 + raw/normalized CSV + run 日志）；`run_sh_test_aware.sh:22` 加 `RUN_OUTPUT_LOG_TIMESTAMP` 环境变量覆盖（默认行为不变）；确定性复跑: 规范化（剔 run_id/wall_time_ns）raw_metrics.csv 逐字段一致 + generated 逐文件 cmp 全一致 |
| 0-4 | 九项语义合同 | `online_contracts/nine_contracts/contract_01..09` + README；①-⑧ 自蓝本同构移植并 face 化改写；**⑨ 全文重写为 face 版 LUT 裁决五条**（标定常数 + decode 代价在线保留；grep 实测 lut.lookup 仅 :683/:689/:1087/:1551 四处，复核无反例） |
| 0-5 | 决策日志物化 | `--replay-record` CLI + `_write_replay_decision_log`（prefill/decode/completion/iteration 四类；decode 记录含 `decode_candidates` 全记录——face 独有，B2 oracle 证据）；213,892 行；字节等价门: 开启与否生产产物 cmp 全一致 |
| 0-6 | 单测基线 | `test_checked_in_three_minute_workload_configuration` 期望值更新 2091/136/(3,158929)/(1,32000) → 1177/112/(66,169395)/(1,13812) + session id 形态 `session_N`；pytest 16 passed + unittest OK（仅期望值，被测代码零改动） |
| 0-7 | 压测基线 + 计数器 | /usr/bin/time -v: wall 59.15s / RSS 714MB（全管线含 build）；Python 侧 `Phase0Counters` 9 计数器（离线恒 0，零字节影响——字节门复验通过）；C++ 侧计数器随阶段 1 MetricCollector 机制层移植落地（登记为显式决策，见下） |

### 关键决策与偏差登记

1. **物化产物复用蓝本 commit**：face 与 wscllm 共享同一源 csv（md5 实测一致
   `fc74a48e...`）与同一派生规则，物化产物自蓝本阶段 0 commit `c98d746`
   提取，并以仓内脚本对源 csv **重派生 30s 窗口逐字节 cmp 通过**
   （1177/112，prefill 66–169395，decode 1–13812，max arrival 29.988879s）。
   wscllm 蓝本对应物 = 其 traces/ 目录；sh_1.0/2.0/3.0 对应物 = 同款
   traces/ 物化（各自仓重派生复核）。
2. **C++ 计数器埋点延后并入阶段 1**：MetricCollector 的 ED diff 与阶段 1
   机制层改动在同一文件，分两次改会重复触碰；改为随阶段 1 一起移植，
   以静态基线字节门复验（cmp 全一致）作为零影响证据。此为对步骤 0-7 的
   显式偏差，已在 baseline PROVENANCE 登记。
3. **trace_config 描述列含逗号会破坏 7 列 CSV 解析**（`clean_csv_row`
   AttributeError）——描述列去逗号后正常；实录登记避免三仓复踩。

### 验证证据汇总

- fail-closed: 缺失输入 exit=1（打印 missing request queue 提示）；
  `template/agent-traces/` 无新生成 stub 文件。
- 确定性: 规范化 raw_metrics.csv 两跑一致（232 行）；generated 逐文件
  cmp 一致；字节等价门（--replay-record 开/关）一致。
- pytest 16 passed；unittest OK。

## 阶段 1（2026-08-16 执行）：最小在线闭环（主变体 session_lru_recompute）

### 完成清单与验证证据

| 项 | 内容 | 结果 |
|---|---|---|
| 1-0 机制层移植 | face-pre 与 wscllm-pre 共享 C++ 文件 15 个逐字节核实一致（cmp 全 SAME）→ wscllm-post 版本直接拷贝：EventQueue.h/.cpp（tick-end 收口 + deferred）、FluidScheduler（deferred flush）、CommonNetworkApi（sim_recv 分流）、Sys/Workload/HardwareResource/Statistics（GraphSource/构造工厂/observer 挂载）、CMake×2（在线目标 + execution_driven glob）、test_congestion_aware.cpp；execution_driven/ 25 文件 + tests/ 11 文件、main_online.cc 整目录拷贝；MetricCollector 拷贝后**剥离 4 处 kv_policy 血统差异**（恢复 face 语义） | 双目标 + 6 个测试二进制全部编译通过 |
| 1-1..1-7 C++ 单测 | event_queue_deferred_test（A-D 含 FluidScheduler 集成，g++ 手动链接）/ NodeStoreTest（含 fixture ET part C）/ WatchRegistryTest / DecisionMailboxTest / GraphBatchCommitterTest / WindowedReaderTest / CompletionFixture（hook 计数=完成节点数、三终态路径）/ BridgeLoopbackTest（往返无损 + ack 幂等 + 崩溃/超时 fail-closed） | 全部 ALL PASS |
| 静态字节门 | 机制层移植后 runall 静态重跑：generated 逐文件 cmp 与阶段 0 归档全一致；规范化 raw_metrics.csv 一致；1177/1177 完成 | **全过（零字节影响）** |
| 1-8 Python online 层 | 决策桥/基类/replay 源/检查点/verify 框架自蓝本移植；graph_batch_builder.py 适配 face writer 助手；face_replay_scheduler.py（裁决 12 流序语义）；**face_online_scheduler.py 逐行迁移 `_plan_face_session_lru_recompute`**（统一实例/LUT 标定常数/加权图+per-die 代价 decode 选择/阻塞重排队语义保留，每处 `# offline: face_scheduler.py:XXXX` 标注） | import/运行通过 |
| 1-9/1-10 端到端 | replay：1177/1177 完成，3526 delivery，reasons 1177×4，决策日志 3531 行，metrics 产出；strategy（关感知）：1177/1177 完成，3531 delivery，零死锁；IDLE fixture：IDLE/ACTIVE/IDLE/DRAINING/FINISHED 五态迁移日志齐全 | **全过** |
| 1-11 same-tick fixture | (a) 单 tick 单 delivery (b) seq=1 tick=T+1 deferred_from_tick=T (c) DECODE_COMPLETION+REQUEST_COMPLETE 交付、completed=1、双进程 exit 0 (d) no_decision_python_callback_count=0 | **四断言全过** |
| 红线检查 | `git status` 中 face_scheduler.py / session_kv_manager.py 零改动（只读 import） | 干净 |

### 关键决策与 face 适配点（相对蓝本的差异，实录登记）

1. **两处 face 独有的 strategy 适配**（蓝本无对应物，因 face 的 decode 实例
   在 PREFILL_DRAIN 边界才选定）：
   - graph_batch_builder `_emit_prelim`：decode_instance_index 为 None 时块末
     账本只登记 prefill 组 rank；`_emit_decode` 对 decode 组未登记 rank 显式
     None-restore（recv 无父，与蓝本裁决 9 终态同构）。
   - assignment 条目移至 decode 发射时一次性携带双索引（C++ 共享校验器要求
     两索引非负；共享机制层不改，face 侧适配）。
2. **online 账本快照口径**（real-online 刻意差异，类 docstring 登记）：
   current_decode_token 在飞期间保持 prefill_context_tokens；remaining_chunks
   保持准入时值（含 history chunks）至 drain。
3. **LUT 在线角色**（合同⑨裁决落地）：FaceLut.build 用 manifest 冻结事实
   推导 max_d_token=max(final_context_tokens) / request_count=len(requests)
   （与离线同一推导、同一常数）；仅 select_decode_instance 消费。
4. **计数器**：replay_poll_query_count==0（无轮询接口，结构上为 0）；
   no_decision_python_callback_count==0（same-tick fixture 断言 (d)）；
   tick_end_without_decision_count 允许非零（C++ 计数器单独报告）。
5. **实施顺序偏差**：机制层按"整批移植 + 综合字节门复验"执行（蓝本按步
   逐门）；每步独立字节门改为移植后一次全量门（cmp 全一致已证）+ 全部
   单测一次跑齐。此为执行节奏差异，非语义差异。

### 蓝本对应物映射（记录纪律）
- 机制层 25+11 文件、main_online、runner×5、online/ 框架文件：wscllm 同名
  同路径直接复用（list 族同血统）；sh_1.0 同族可直接复用本仓同款补丁；
  sh_2.0/3.0（map 族）需按各自 proceed 亲核 deferred 通道。
- face_online_scheduler.py：蓝本对应物 wsc_llm_online_scheduler.py
  （逐行重写，策略不同）；sh_1.0/2.0/3.0 对应各自 scheduler 的同位迁移。

## 阶段 2（2026-08-16 执行）：关感知 Tier B 等价验收

### 完成与证据
- `online/verify/tier_b_compare.py` 重写为 face 版比较器（蓝本对应物 =
  wscllm tier_b_compare.py；sh_1.0/2.0/3.0 迁移时按各自策略字段适配）。
  face 版适配点：
  1. **B3 在线图来源**：阶段 7 §10.3 起 response_*.json 消费即删（C++ 侧
     unlink），比较器改用确定性 Python 侧重建（bridge request_*.json 喂
     FaceReplayScheduler；与幂等 fixture 同机制）——不触碰共享机制层。
  2. **B2 face 独有核对**：decode_candidates 全记录逐值一致（合同⑨）；
     real-online 不变量 C1 = argmin(per_die_delta_ns, instance_index) ==
     所选 decode 实例（"同一快照输入 → 同一输出"的可计算投影）；
     C2 = schedulable 候选域；C3 = history 链。
  3. B4 采用蓝本操作性标准（差异归因类别①-④ + 完成序不变量）。
- 实测（20.csv 前30s）：B0-B4 全 PASS。B3 within=1744（全类别④
  transfer3000；蓝本 wscllm 1495——face 实测值，不照抄）；cross=12634
  （类别①③信息性）。B1 tick 差：decode [-11,+1]ns / completion
  [-14,-1]ns；prefill 最小 -994ms（replay 装置到达边界起链语义，登记）。
- 验收报告：`online/tier_b_report_20.md`。

## 阶段 3（2026-08-16 执行）：感知打开最小版

- sensing strategy 运行（`run_online_strategy_sensing.sh`）：1177/1177 完成。
- `ledger_reconcile.py`：**对平（balanced）**——R0-R8 全过。face 适配：response
  消费即删后对账改经确定性 Python 侧重建（request_*.json 喂
  FaceOnlineScheduler；--manifest 必填）。face 的 retain_complete 事件语义
  （每 request 一条、completion 边界）与离线 kv_cache_events.csv 口径一致。
- `diff_explainability.py`（sensing off vs on）：**3531/3531 决策行逐字节
  一致**——感知数据不进策略判据（红线），决策序列不变；全部差异归类
  （无差异 3531 / 时序 0 / 排队 0；ledger_summary 载体与日志行为设计内差异）。
- 感知口径登记（方案 §6.1）：face 策略不读网络拥塞（统一实例 + LUT 代价 +
  排队/KV 账本）；"感知"实质 = 决策账本由 C++ 真实完成事件驱动（阶段 1
  已成立），本阶段补齐两层剩余负载查询（injected-unfinished / admitted-
  not-injected）、分层账本最小子集与结束总账核对；拥塞快照接口本仓不消费。

## 阶段 4/5/6（2026-08-16 执行）

- 阶段 4（StateDelta v1/幂等/索引队列，随机制层移植）：幂等 fixture
  PASS（3531 deliveries 重放零重复产出 + verify_run_end ok）；idempotency
  fixture 适配 response 消费即删后的 request 重放；profile：full_scan=0，
  scanned 平均 2.3/批 max 3（与总 request 数无关）。schema v1 合同文档
  （仓根 online_contracts/state_delta_v1.md）随蓝本拷贝在位。
- 阶段 5（GraphBatch 原子提交）：GraphBatchCommitterTest ALL PASS（两段
  validate/commit + 非法 batch 零副作用）；正式路径 single_node_bridge_
  count == 0（C++ 计数器，仅有 batch 通道）。
- 阶段 6（基准矩阵 + 预算冻结，见 baseline/20_30s/PROVENANCE.md 阶段 6 节）：
  offline-end-to-end ≈27.5s / static-only 10s / replay 26.1s / strategy
  28.6s / sensing ≈28.6s；决策门五条件全过。
- C++ cli_online_test（R1-R11）ALL PASS；checkpointing 单测 8 passed。

## 阶段 7（2026-08-16 执行）：补齐 + legacy 第二变体

### 10.4 窗口扫掠（face 输入实测）
- 旋钮：run_online_replay.sh 的 `WINDOW_ROWS` 环境变量 →
  `--request-window-rows`。3 臂扫掠（64/128/512）：
  - 64：fail-closed（turn-0 行读入晚于 arrival → alarm 过去 tick，C++
    GraphBatch 校验拒绝，delivery 983 abort）——窗口下限效应实测复现；
  - 128（缺省）/512：online_decision_log md5 = `43f32dea...` 逐字节一致
    （与缺省运行一致）——零扰动。
- **冻结 high_water=128**（蓝本同值但按 face 输入独立实测；不得下调）。

### 10.6 legacy 第二变体在线迁移
- 新增：`online/face_legacy_online_scheduler.py`（蓝本对应物
  wsc_llm_legacy_online_scheduler.py；sh_1.0 对应物 = 各仓 legacy 分支同位
  迁移）+ builder `emit_prefill_batch_legacy`（离线 legacy writer 的在线
  复刻：HistoryPieceGate 逐 piece 迁移 + prefill aggregated spans + end
  barrier）+ `trace_config_legacy.csv`（唯一差异 kv_cache_policy=legacy）
  + `run_online_strategy_legacy.sh`。
- face legacy 语义逐项保留（§10.6 差异清单）：无 FCFS 队头阻塞
  （allocate 容量不足 raise = fail-closed，未引入等待机制）；KVAllocator
  跨实例分片 + allocate/release 的 increase_path/decrease_path 边权动态
  调整（共用同一 allocator 实例按离线同序应用）；session 重访 release
  时机在 arrival；p_chunk=4954 标定常数（冻结输入同一推导，manifest 全量
  一次导出，禁止在线增量 mean）。
- 离线 legacy 基线：trace_config_legacy.csv 物化（label ..._pc4954_..._
  ce3b40440，--replay-record 决策日志在案，归档 baseline/20_30s_legacy/）。
- 在线 legacy strategy 运行：**1177/1177 完成**，3531 delivery，run-end
  审计通过；kv_event_payload_legacy（终值边权/剩余容量）产出。
- B-LEGACY 口径（蓝本"顺序敏感字段"裁决）：实例指派为顺序敏感字段——
  在线赋值 vs 离线逐值匹配 155/176（of 1177，信息性；real-online 时序
  差异合法）；硬验收 = 每请求全覆盖 + C1 argmin(per_die_delta_ns) 不变量
  0 违例 + run-end 非空闲实例为 0 + p_chunk 常数一致——**通过**。
  B3 级 legacy 图字节等价未单独验收（legacy 图形制复刻为功能性复刻，
  span/barrier 结构与离线同构；逐字节门未跑）——**登记为剩余差异**。

### 10.3/10.5 有界性与中间产物
- response/ack 消费即删；request_*.json 审计保留（3531 个）；run 目录
  清理后 56-61MB；检查点 1 个/run；checkpointing 单测 8 passed；
  WindowedReaderTest（含 no-CSV request-neutral 路径）ALL PASS。

### 收尾状态
- pytest：test_face_scheduler(16) + test_checkpointing(8) = 24 passed；
  C++ 机制单测全过；静态基线字节门保持全一致；红线 diff（face_scheduler/
  session_kv_manager 自 55cf135 起）为空。
- 中间产物规模：decision_log 213,892 行（~173MB，--replay-record 显式
  产物）；bridge request 3531 个/run；均在允许输入（20.csv 前30s）范围内。

### 剩余差异与未覆盖点（逐项报告）
1. legacy 变体的 B3 级图字节等价验收未做（见上）；
2. C++ MetricCollector 动态 anchor 注册（full metrics 在线）随蓝本移植
   在位但未做 full-metrics 在线验收（本仓全部在线运行用 summary 档）；
3. LiveRequestProducer 的独立压测未单独跑（注入 fixture 已验注入语义；
   背压/迟到策略随机制层移植在位）；
4. 感知开启后的 placement 差异归因报告基于"sensing off/on 决策逐字节
   一致"（感知不进判据）；"真实时序 vs LUT"类差异归因在 Tier B 报告
   类别①-④覆盖；
5. 裸仓库收尾（删除物化输入/基线归档、trace_config 中性化）留待主控
   裁决（方案 §0.5 收尾形态预告；开发期物化与归档未删除）。

## 收尾阶段（2026-08-16，主控裁决 ①②③④）

### ① legacy B3 级单独验收（比照 B-LEGACY 口径，如实记录）
- 装置：确定性 Python 侧重建（legacy run 的 request_*.json 喂
  FaceLegacyOnlineScheduler），与离线 legacy .et（baseline/20_30s_legacy）
  做与主分支 B3 同款的 per-(rank,request) 位置比较（同一比较器函数，
  未放宽）。
- **过程中发现并修复一个真实缺陷**：legacy 构图分块误用
  config.prefill_chunk_size=512，应为 legacy 标定常数 p_chunk=4954
  （graph_batch_builder.emit_prefill_batch_legacy 改从 plan dict 取
  p_chunk；FaceLegacyOnlineScheduler._plan_dict 携带）。修复后 legacy
  在线运行重跑 1177/1177 通过。
- 实测结果（25,806 组比较）：count/name/type/attr diffs =
  25038/7716/2730/7716——**不通过字节等价**；within-request 依赖差异 0；
  cross 768。归因：legacy 离线 writer 的 per-request 块布局（HistoryPieceGate
  链/control 触发序/跨 request 在 decode rank 上的交错发射）未被在线功能
  复刻逐字节重现（在线为两段式发射 + 延迟 decode 选定）。按 B-LEGACY
  口径（顺序敏感字段不逐值比对）登记为**已登记差异**：legacy 硬验收 =
  全请求完成 + C1 argmin 不变量 + run-end 审计 + p_chunk 常数一致
  （此前已全过）；字节等价不宣称。**未放宽比较器**，以上数字如实记录。

### ② full 档在线 metrics（动态 anchor）验收
- 装置：手动 full 档在线 strategy 运行（FIFO 等待后启 Python；
  /tmp/face_full_metrics）。
- 结果：**通过，未发现缺陷**。1177/1177 完成；动态 anchor 全覆盖
  （prefill/decode/completion tick 1177/1177/1177）；memory_anchor
  24,708 行；rank_compute/capacity_timeavg/planner_memory_peaks 各 54；
  consistency 1。与 summary 档对照：completed/incomplete/input/memory
  actions（replayed/unresolved/final_state_mismatches）/sim_end_ns
  **全部一致**（同输入确定性运行的正确表现）。

### ③ LiveRequestProducer 最小 fixture
- `run_online_idle_fixture.sh` 复跑 ALL PASS：scenario 2 = 注入 2 个
  request（tick 3000000000 精确 alarm）→ ACTIVE → 完成 → IDLE →
  CloseInput → DRAINING → FINISHED，双进程 exit 0。
- 背压行为（fixture §10.7 段）：容量 2 的 ingress 提交第 3 个 Submit 被
  拒绝（overflow_count == 1，峰值占用 2）——有界命令队列背压契约带
  可观测拒绝计数，断言在 fixture 内通过。

### ④ 裸仓库收尾（比照 wscllm 55cf135 同款）
- 删除：traces/（物化输入 + PROVENANCE + derive 脚本）、baseline/
  （20_30s 与 20_30s_legacy 归档）、online/tier_b_report_20.md、
  generated/completion_fixture（合成 fixture ET 产物）。
- trace_config.csv:12 中性化 →
  `llama2_7b_inference/request_queue_placeholder.csv`（不存在占位）。
- 测试合成 fixture 化：test_face_scheduler.py 的
  test_checked_in_three_minute_workload_configuration 及其余 3 个读
  checked-in 配置的用例改经 load_checked_in_config()（临时配置副本 +
  手写 8 session/10 request 合成队列；绝不引用真实 trace 数据）。
- 验证：**双根 pytest 57 passed**（workload 根 24 + sh_test_mesh/tests
  33）；**fail-closed 实测** `generate_trace.py --print-shell-config`
  exit=1，打印 missing request queue 提示；README_COMMANDS.md 改
  request-neutral 说明（含唯一允许源与物化规则指引）。
- 裁决依据：主控 2026-08-16 收尾指示；wscllm 收尾先例（55cf135）。

### 最终 git 状态
- 提交体系：face-phase0..7-done（8 个阶段 tag）+ 本收尾 commit
  `face-final-done`；分支 main；face_scheduler.py / session_kv_manager.py
  自 55cf135 起零改动（红线）。
- 裸仓库可复现路径：物化输入（方案 §3 步骤 0-1）→ runall（静态基线）
  → run_online_replay/strategy/legacy（request csv 显式传入）→
  tier_b_compare（--baseline 指向重新归档的基线）。

## 回灌修复（2026-08-16，来源：sh_2.0测试/对比报告.md §5）

> 背景：sh_2.0 四档 3 分钟对比测试发现三处缺陷，其中两处源于 wscllm
> 蓝本机制层（五仓复制传播）。本节为 face 仓回灌修复与复验实录；
> 机制层文件与 wscllm 仓保持逐字节一致（血统一致约束，cmp 复核）。
> tag：`face-backport-fix1-done`。

### A. OnlineCli 默认窗口 fail-closed 化（对比报告 §5.1）

- 缺陷：`--request-max-arrival-ns` 默认 30e9 = 把 30s 验收输入窗口烧进
  代码默认值；超窗 turn-0 请求被 WindowedTraceReader 静默丢弃，且被丢弃
  行永不消费 → 窗口占用钉死在 high_water、pump 在 EOF 前失速 →
  expected_requests 停留在 0、完成度审计被整体跳过（sh_2.0 strategy-20
  实测：输入 2091、完成 1830、261 请求静默缺失仍 exit 0 "PASS"）——
  request-neutral 与 fail-closed 双违反。
- 修复（最小、语义保持）：
  1. 默认改**无界**（0 = 不设上限）：OnlineCli.cc/.hh 默认值、
     WindowedTraceReader 构造默认参数 30e9 → 0；拒绝判定加
     `max_arrival_ns_ > 0 &&` 守卫。窗口保留为显式实验旋钮
     （--request-max-arrival-ns），任何丢弃可见且 fail。
  2. 完成度审计分母改 **total rows**：WindowedTraceReader 新增
     count-only 尾扫（`count_remaining_data_rows()`：不 Submit、不注册
     queue_index/metrics，仅计数；同时区分 turn-0/turn>0）与
     `total_data_rows()`/`turn0_data_rows()`；main_online 主循环结束后
     对 CSV 运行尾扫兜底（正常 EOF 路径 no-op），失速窗口无法再缩小
     审计分母。
  3. 审计算术抽为可单测函数 `audit_completion()`（四判定：
     Dropped=dropped>0 任何显式窗口丢弃即失败（计数可见）；
     AccountMismatch=accepted+dropped != turn0_rows（turn-0 行 neither
     accepted nor rejected）；Incomplete=completed+dropped != total_rows；
     Ok）。计数语义按官方运行实测校准：accepted 仅计 turn-0 提交
     （112/1177），turn>0 走 future-alarm 路径不计入 accepted，由
     completed vs total 覆盖——首版误用 accepted+dropped==total 不变量
     被 face replay 实测（accepted=112）当场证伪后修正。
  4. 启动行打印 `max_arrival_ns=0 (unbounded default; backport fix ...)`；
     窗口阅读器 report 行增列 total_rows/turn0_rows。
- 合成 fixture 测试（不依赖物化 csv）：windowed_trace_reader_test.cc 新增
  Part G（默认无界：构造默认参 + CLI 默认参均 0，180s 到达全接受）与
  Part H（显式小窗 25s：丢弃计数可见、窗口在不可消费拒绝行上失速后
  尾扫恢复 total=7/turn0=5 全文件分母、尾扫零 Submit、audit_completion
  对 sh_2.0 strategy-20 形状 {2091,136,112,1830,261} 判 Dropped 及
  AccountMismatch/Incomplete/Ok 各形态）；cli_online_test.cc R11 默认值
  断言 30e9 → 0。两仓 WindowedReaderTest 全部 A-H PASS。

### B. runner REQ_COUNT ARG_MAX 修复（对比报告 §5.3）

- 缺陷：在线 runner 尾段 `REQ_COUNT=$(ls .../request_*.json | wc -l)` 在
  保留 envelope 数 >约 2e4 时通配展开超 ARG_MAX（E2BIG → bash exit 126，
  `set -euo pipefail` 下终止脚本），仿真本体成功后收尾段异常退出。
- 修复：本仓 run_online_replay.sh / run_online_strategy.sh /
  run_online_strategy_legacy.sh / run_online_strategy_sensing.sh 四脚本
  的**全部两处** request-glob 计数行（FAIL 诊断行 request= 与尾段
  REQ_COUNT=）改 `find "${RUN_DIR}/bridge" -maxdepth 1 -name
  'request_*.json' | wc -l`（与 sh_2.0 测试副本登记补丁同型，且把 FAIL
  行一并修复——同型同风险）。CP_COUNT（checkpoints/*.json，1 个/run）与
  jsonl 计数（固定 5 文件名）为构造有界，排查过不适用，未改。
  语法 bash -n 全过。

### C. turn-0 逐出 KeyError 同型排查（对比报告 §5.2）——不适用，登记

- sh_2.0 缺陷形态：builder `_mark_pending_history_store` 假定
  pending_request_by_session 里的 pending request 必有 pending_history
  gate；turn-0 在 prefill 发射时登记 session 键而无 gate，其他请求
  completion 的逐出命中该会话时直接下标 pending_history → KeyError。
- 本仓排查结论：**不适用**。face 的 online/graph_batch_builder.py（701 行）
  谱系与 sh_2.0（994 行）不同：`pending_request_by_session` /
  `pending_history` / `deferred_session_locations` /
  `_mark_pending_history_store` 全部 0 命中（grep 实测）；本仓 builder
  仅维护 completion_gates 与 _prefill_block_ends 两本账，无逐会话
  pending-历史位置账本、无 builder 侧逐出路径（admission/decode_target/
  completion 逐出为 scheduler 侧 kv_manager 记录，逐 request 当批翻译，
  无跨 request session 键下标访问组合）。face 的离线
  generate_face_trace.py 亦无该机制（sh_2.0 的 generate_face_trace.py
  :2178+ pending_request_by_session 为 sh_2.0 谱系独有演进）。
  无需修复，不加测试。

### D. teardown unreleased nodes 观测（对比报告 §4.3）——已观测，对齐登记

- 本次复验 face 在线运行 cpp.log 退出期实测：replay 2 个 sys
  （sys.id=32/33 各 1 行 `!!!Hardware Resource ... unreleased nodes!!!`）、
  strategy 6 行——与 sh_2.0 §4.3（replay 每档 2 sys、strategy 6 sys）
  同族同量级；且与 wscllm 蓝本实录 B17（方案 §15 bug 表：
  "HardwareResource 析构警告 sys.id=32/33 node 13610 = 最后 request end
  barrier 的 1ns 事件在结束权下未处理，判定良性结束伪影"）**sys.id 与
  根因分析逐字对齐**。运行本体全部通过（完成度审计/bridge 审计/exit 0）。
  本仓登记为良性退出期伪影（观测项），不修改；与 sh_2.0 侧根因分析
  （另一 agent）结论一致：end-barrier 1ns 控制事件在完成权下未释放。

### 复验证据（20.csv 前 30s，1177/112）

1. **重物化**：traces/ 物化器 + PROVENANCE 自 git 历史（01d0a3b）恢复
   重建入口，对源 csv（md5 fc74a48e...）重派生：1177 请求/112 session、
   prefill 66-169395、decode 1-13812、max arrival 29.988879s；queue 与
   sidecar 对冻结血统**逐字节 cmp 一致**。
2. **编译 + 单测**：build_analytical_aware.sh 全目标（含 Online + 7 个
   测试二进制）编译通过；WindowedReaderTest A-H ALL PASS、cli_online_test
   R1-R11 ALL PASS、NodeStore/WatchRegistry/DecisionMailbox/
   GraphBatchCommitter 回归通过（NodeStore part C 需 --fixture-et、
   CompletionFixture 需运行参数，裸仓态既有形态，与本次改动无关）；
   **双根 pytest 57 passed**（workload 24 + tests 33）。
3. **静态字节门**：runall 全管线 exit 0（1177/1177，incomplete=0），
   raw_metrics 232 行对阶段 0 冻结基线剔除 run_id/wall_time_ns 后
   **逐字段 0 差异**——回灌对静态路径零扰动。
4. **在线路径**：replay PASS（delivery 3526、completed 1177/1177、
   accepted=112、turn0_rows=112、total_rows=1177、dropped=0、
   no_decision=0、single_node_bridge=0、决策日志 3531 行）；strategy
   PASS（delivery 3531、1177/1177、single_node=0）——与阶段 1-9 实录
   数字一致；完成度审计（新 fail-closed 口径）Ok。
5. **Tier B 复跑不回退**：基线重新归档（decision_log.jsonl md5
   93e8998f... 与冻结一致）后 tier_b_compare B0-B4 **ALL LAYERS PASS**
   （B3 within=1744 全类别④/cross=12634、B1/B2 零失配、B4 cross-tick
   反转 0——与阶段 2 实录一致）。
6. **fixture**：IDLE fixture 五态迁移 ALL PASS；same-tick milestone
   (a)-(d) ALL PASS。
7. **裸仓态恢复**：物化输入（traces/）、基线归档（baseline/）、运行
   产物（results/）删除；trace_config.csv 回 placeholder（git checkout）；
   generated/ 恢复 pc512 + pc4954（label ce3b40440 与原目录名一致）+
   runtime_config；fail-closed 实测 `--print-shell-config` exit=1；
   双根 pytest 复跑 57 passed。

## 在线机制层竞态缺陷修复（2026-08-16，来源：face主动测试错误分析.md；tag：face-defectfix2-done）

> 触发：用户授权的四档 3 分钟主动测试（face测试/，2091-11121 请求）暴露
> 三类在线机制层竞态缺陷，120 档在线 0/4 全败（F1-F7 详见
> face主动测试错误分析.md §2）。本节为修复与复验实录。
> 授权链：3 分钟主动测试与复测均为用户 2026-08-16 指示；缺陷 B 协议
> 选型（双侧长连接）经主 agent 批准（四点执行要求含 F7 归因如实修正、
> 双向死检 fixture、移植性登记、复测口径不变）。
> 策略红线零触碰：face_scheduler.py / session_kv_manager.py /
> generate_face_trace.py 零改动（git diff 复核）。

### 0. 失败现场法证修正（F7 归因，如实录登记）

- **F7（120 strategy v2 停滞）重新定性为缺陷 B 竞态族的"楔死"形态**，
  而非原分析文档的缺陷 C 唤醒丢失：
  - 末态证据：C++ 主线程 `do_sys_poll` = 文件桥 `wait_readable` 的
    poll（全仓唯一 poll 调用点；wait_for_work 是 futex 等待，不是
    sys_poll）+ `request_26847.json` 已写出（C++ 已进入第 26847 轮
    round-trip）+ **commit_ack_26846.json 未被 Python 消费**（ack 消费
    即删，文件仍在 = Python 从未读到该门铃）+ digest 止于 26846 ——
    Python 楔在 resp 通道握手（wait_for_partner 睡眠），双端互等。
  - 缺陷 C 的唤醒真空作为代码层真实角落仍成立（空队列分支注释自述；
    本节修复保留并加固），但 F5/F6/F7 的现场证据均指向文件桥握手竞态
    族；7 场复测验证各修复的实际承载关系，不预记功劳。
- **缺陷 B 竞态的内核级实证**（本机 WSL2 6.6.87）：
  - 语义实验：FIFO 读端 fd 对"从无写端"或"写端早已关闭"一律阻塞
    （不报 POLLHUP）；仅当本 fd 的 poll_wait 注册期间发生写端 1→0
    关闭转移且缓冲空，才以 POLLHUP 唤醒、read()==0。即旧协议的
    "逐交换开/关"握手存在**应用层不可消除**的竞态窗口。
  - 旧模式压测（复刻 deliver_and_receive 的逐交换
    open(O_RDONLY|O_NONBLOCK)→poll→read→close 与 Python 逐交换
    open(O_WRONLY)→write→close，200 万次预算）：观察到 C++ 侧
    `POLLHUP+read==0` 假"Python 已崩"（两轮分别于第 145/7 次迭代
    触发，写端进程存活）与 Python 侧 `BrokenPipeError` 互补变体
    ——分别对应 F1 的误杀签名与握手楔死形态。复现装置入库：
    `sh_test_mesh/run_scripts/bridge_race_stress_repro.sh`。

### A. 缺陷 A：服务终判 finished() 计数真空（F2/F3/F4）

- 根因（原分析 §3 A + fixture 确证）：主循环顺序
  `drain_commands() → windowed.pump() → svc.finished()`，而 pump()
  只把 Submit **入队**（下一轮 drain 才 on_alarm_scheduled 注册）。
  当上一批 alarm 全部触发、active 归零的那一轮恰逢 pump 顶部补窗，
  finished() 在"已入队未 drain"真空期返回真 → run 提前结束、CSV 尾部
  未交付 → Python `ack_count != delivery_count`（恒差 1）→ 双端连环
  fatal/abort。
- 修复（`main_online.cc` 主循环）：
  1. pump() 之后、finished() 判定之前，若 ingress 有未 drain 命令则
     再 drain 一次（`pending_command_count() > 0 → drain_commands()`），
     关闭真空期——修后"末笔 ack_count==delivery_count 恒成立"
     （3 分钟复测七场全部 delivery==ack 实测，见复验节）；
  2. belt-and-braces：finished() 为真而 CSV 尚未 EOF 时
     online_fatal（携带 active/pending_alarm/queued 计数）——修复后
     该状态不可达，审计为防回归的 fail-closed 保险。
- 决策序列零扰动论证：额外 drain 只把"下一轮首语句的 drain"提前到
  本轮 finished() 判定之前执行，EventQueue 按时间排序、注册先后不
  改变事件序；20.csv 前 30s 包络 replay/strategy/Tier B 全绿 + 3 分钟
  replay 完成数/交付数与冻结基线一致的实证支撑。
- 回归 fixture（先复现再转绿）：
  `astra-sim/workload/execution_driven/tests/service_vacuum_test.cc`
  （新 CMake 目标 ServiceVacuumTest）。30 行 turn-0 CSV（3 批到达 ×
  10 行、high_water=10、到达即完成的 no-node 合同）驱动**真**
  WindowedTraceReader/RequestIngress/ServiceCoordinator/EventQueue
  跑两种循环顺序：legacy 顺序确定性复现真空（completed=10、断点时
  队列尚有 10 条未 drain 命令、reader 未 EOF、审计 AccountMismatch
  非 Ok）；fixed 顺序 30/30 全完成、EOF 到达、审计 Ok。ALL PASS。

### B. 缺陷 B：文件桥 EOF 误检（F1；含 F5/F6/F7 楔死形态）

- 协议选型：**双侧长连接**（主 agent 2026-08-16 批准）。心跳/保活被
  否决：不消除竞态窗口本身、引入调参与新误判面。长连接镜像了 req 通道
  （C++ 持写端整个 run 期）自阶段 1 以来零故障的既有模式；run 期内
  双端无 fd 开/关转移，read()==0 的语义恢复为"对端进程死亡"（真）。
- C++ 侧（`DecisionBridge.hh/.cc`）：
  - 新增 `resp_notify_fd_` 读端，open_notify() 一次打开
    （O_RDONLY|O_NONBLOCK 恒成功）、run 生命周期持有、析构关闭
    （与 req 写端对称）；fixture 懒打开路径保留（deliver_and_receive
    首次调用时）。
  - deliver_and_receive 的响应等待改为新私有 `wait_response_byte`：
    常开 fd 上 poll→read 恰 1 字节，EAGAIN 重 poll；read==0 →
    abort，消息改为"Python side died (its long-lived resp_notify
    write end closed; ...)"——明确的死因语义（不再有"暂无写端"态）。
  - 1:1 守卫：读到通知字节后立刻再 read 一次，若还有字节 →
    "protocol violation: more than one response byte in flight"
    fail-closed（背压合同防漂移；主 agent 执行要求 2c）。
- Python 侧（`online/decision_bridge.py`）：
  - serve_forever 启动时一次 `os.open(resp_notify, O_WRONLY)` 常开
    （配对 C++ 常开读端；打开顺序无死锁：C++ 先 req 写后 resp 读，
    Python 先 req 读后 resp 写），run 期持有、finally 关闭；
  - _notify_response 只写不开；BrokenPipe/OSError 包装为
    `BridgePipeError`（C++ 常开读端关闭只能因 C++ 进程死亡，不可重试）
    → serve 循环 fail-closed：stderr 留痕
    "decision_bridge: C++ side is gone (BrokenPipe on resp_notify
    write, seq N)" + sys.exit(1)——修复 F1 现场 python.log 0 字节
    无可诊断性的问题；
  - _fail() 先 stderr 留痕再写 error response 退出（同上动机）。
  - 全部 6 个 BridgeServer 消费方（online_service / bridge_echo /
    same-tick / lifecycle 等）经此一处自动同步。
- 回归 fixture（先复现再转绿，主 agent 执行要求 2a/2b/2c）：
  - `bridge_loopback_fixture.cc` 新增 Part D/E（原 A/B/C 语义在新协议
    下保持全过）：D=Python 完成交换 1 后 SIGKILL 自杀 → C++ 常开读端
    abort 且消息含"long-lived resp_notify write end closed"；E=
    Python 一笔响应写 2 字节 → "more than one response byte in
    flight" abort（1:1 守卫）。
  - `online/verify/bridge_cpp_death_fixture.py`（纯 Python 假 C++，
    复刻 C++ 侧 fd 语义后于第 2 笔交付前死亡）：真实 BridgeServer 撞
    BrokenPipe → exit 1 + stderr 留痕（连续 3/3 确定性）。
  - `sh_test_mesh/run_scripts/bridge_race_stress_repro.sh`：旧模式
    竞态复现装置（见 §0 实证；预算内观察到假 EOF 即 PASS）。

### C. 缺陷 C：空队列分支唤醒条件（F5/F6/F7 原分析口径）

- 修复（`main_online.cc` 空队列分支三分支化 + `EventQueue` 访问器）：
  1. mailbox 有残留 → 既有 T+1 显式唤醒（不变，step 1-11 CORE）；
  2. **新**：`EventQueue::has_deferred_work()`（新访问器，
     `EventQueue.h/.cpp`）为真 → 同机制调度 T+1 唤醒强制下一决策边界
     ——deferred 队列只在 proceed() 内排空，主队列为空时它没有执行
     路径（原代码会对其盲等）；不计 deferred_from_tick（非交付延后）；
  3. **新**：input 已关（官方 runner 形态）且队列+mailbox 双空而
     svc 未完 → **lost-wakeup dead end fail-closed**：active>0 而无
     任何可触发事件（如策略 defer 后全局静默）是协议死端，无事件会
     再唤醒、官方路径无人 signal_work——旧代码 wait_for_work 永久
     自旋/挂死（实测预修复二进制：CloseInput 后 10s+ 无退出），
     修复后立即 abort 并携带 active/pending_alarm/deferred 全计数
     诊断（唤醒真空从"50 分钟静默停滞"变为可归因失败）；
  4. input 仍开 → wait_for_work（IDLE fixture 合同不变，实测五态
     迁移 fixture 仍 ALL PASS）。
- F7 末态蓝图回归 fixture（先复现再转绿）：
  `sh_test_mesh/run_scripts/run_online_wakeup_guard_fixture.sh` +
  `online/verify/wakeup_guard_fixture_service.py`，两场景：
  - **f7-blueprint**（健康形态）：ARRIVAL 批携带[同步完成控制节点
    （同 tick 里程碑）+ 未到期 future alarm（T+1000）]，里程碑与
    r1 的 ARRIVAL 同 epoch 交付、两请求全部完成（accepted=1
    completed=2——future-alarm 到达不计 accepted）、exit 0 ——证明
    修复后的空队列分支不在健康蓝图形态误触发；修复前后均过
    （预修复二进制同过，该场景为防误伤回归 guard）。
  - **defer-dead-end**（唤醒真空死端）：ARRIVAL 空批 defer +
    CloseInput → EXPECT_MODE=fixed（修复后）：数秒内 fail-closed
    abort、cpp.log 含"lost-wakeup dead end"且 active=1 入消息；
    EXPECT_MODE=legacy-stall（RED 基线，face测试 120 副本预修复
    二进制）：CloseInput 后 10s+ 无退出（挂死复现，runner 断言后
    SIGTERM）。同场景 RED→GREEN 双证。

### 移植性登记（主 agent 同步其余四仓用；本族为"服务终判/等待/死端"机制层缺陷家族）

| 修复 | 文件 | 改动语义 | 同步要点 |
|---|---|---|---|
| A 计数真空 | `main_online.cc`（主循环 pump 后补 drain + eof 审计） | pump 入队命令在 finished() 判定前注册为 pending alarm；finished&&!eof→fatal | 各仓 main_online 同源段落可直接套用；sh_1.0 的 30s 饥饿 guard 与此同族（终判计数不覆盖在途工作），同步轮建议统一收敛为"判终前 drain + 审计"而非各贴各的 guard |
| B 长连接 | `DecisionBridge.hh/.cc` + `online/decision_bridge.py`（+6 消费方自动继承） | resp 通道双侧常开 fd；read==0=对端死亡（真语义）；Python BrokenPipe→BridgePipeError fail-closed+stderr 留痕；C++ 1:1 在途守卫 | **C++ 与 Python 必须同版本成对移植**（旧 Python 逐交换开写端 × 新 C++ 常开读端会在每笔交付后伪 EOF）；loopback Part D/E、cpp_death fixture、stress repro 脚本一并同步 |
| C 唤醒条件 | `main_online.cc`（空队列三分支）+ `EventQueue.h/.cpp`（has_deferred_work 访问器） | deferred 残留→T+1 强制边界；双空+svc 未完+input 关→lost-wakeup dead end fail-closed；input 开→IDLE 合同不变 | EventQueue 属 extern 共享层（纯新增 const 访问器，静态二进制零行为变化）；sh_2.0 的 livelock guard 与死端 fail-closed 同族，同步轮统一收敛 |

- 三处修复均不与 face 特有代码纠缠（无策略/决策语义引用），补丁独立；
  fixture 全部为机制层合成输入（不引用真实 trace 数据）。
- face测试 四副本已同步全部机制层文件（execution_driven/* +
  main_online.cc + EventQueue + decision_bridge.py + fixture/脚本）
  并原地重编译（四仓 build exit 0）；注意副本原为 face-final-done
  血统，本次同步连带补齐了 backport-fix1 的
  WindowedTraceReader/OnlineCli（副本缺该两件，其 runner 显式
  180e9 参数掩盖了默认值差异）。

### 复验证据

1. **三缺陷回归 fixture 全绿**（先复现再转绿）：
   - ServiceVacuumTest：legacy 顺序确定性复现真空（completed=10/30、
     断点队列 10 条未 drain、审计非 Ok）→ fixed 顺序 30/30 + Ok；
   - BridgeLoopbackTest A-E ALL PASS（D=SIGKILL 死检含新消息、
     E=1:1 守卫）；bridge_cpp_death_fixture 3/3（exit 1 + stderr
     "C++ side is gone"）；bridge_race_stress_repro PASS（假 EOF
     于第 7 次迭代复现，写端存活）；
   - wakeup_guard：f7-blueprint PASS（未到期 alarm+同 tick 里程碑
     健康完成 accepted=1/completed=2）+ defer-dead-end PASS
     （fail-closed 诊断 active=1）；RED 基线：预修复二进制
     legacy-stall 复现（10s+ 无退出）。
2. **20.csv 前 30s 包络全量回归**：
   - 重物化：源 md5 fc74a48e... 与 PROVENANCE 一致，queue+sidecar
     与冻结血统（sh_1.0 仓同源副本）逐字节 cmp 一致（1177/112）；
     trace_config.csv:12 恢复 01d0a3b 原字节（ET 目录 label
     cd6682e3b 与 runner 一致）。
   - 编译 + 单测：build 全目标 0 error；WindowedReader A-H /
     cli_online R1-R11（g++ 现编）/ NodeStore / WatchRegistry /
     DecisionMailbox / GraphBatchCommitter / ServiceVacuum（新）/
     BridgeLoopback A-E（扩展）全过；**双根 pytest 57 passed**
     （workload 24 + tests 33）。
   - 静态字节门：runall exit 0（1177/1177，incomplete=0），raw
     _metrics 232 行对冻结基线剔 run_id/wall_time_ns **逐字段 0
     差异**——缺陷修复对静态路径零扰动。
   - 在线路径：replay PASS（delivery 3526、completed 1177/1177、
     accepted=112、no_decision=0、决策日志全量消费）；strategy
     PASS（delivery 3531、1177/1177）——与冻结实录数字一致；
   - **Tier B 复跑不回退**：基线自 git 重建（decision_log.jsonl
     213892 行 md5 93e8998f... 与冻结一致）后 B0-B4 **ALL LAYERS
     PASS**（B3 within=1744 全类别④/cross=12634、B4 cross-tick
     反转 0——与冻结实录一致）。
   - IDLE fixture 五态迁移 ALL PASS；same-tick milestone (a)-(d)
     ALL PASS（既有 T+1 唤醒合同未回归）。
3. **3 分钟主动复测**（原失败 7 场，face测试 四副本重编译后真实负载
   下顺序执行；并行 agent 共存：sh_1.0 在线测试进程同期运行）：

| 场景 | 原失败 | 复测结果（completed/物化） | delivery==ack | 墙钟 |
|---|---|---|---|---|
| F1 50 replay | 13625 交付 EOF 误杀+孤儿 | **4647/4647 PASS** | 13833==13833 | 203s |
| F2 80 strategy | 481 交付 ack 错位 | **7423/7423 PASS** | 22269==22269 | 472s |
| F3 120 replay | 30926 交付提前关桥 | **11121/11121 PASS** | 33069==33069 | 957s |
| F4 120 replay2 | 13135 交付提前关桥 | **11121/11121 PASS** | 33069==33069 | 968s |
| F5 80 strategy2 | 12453 交付死锁停滞 | **7423/7423 PASS** | 22269==22269 | 524s |
| F6 120 strategy | 219s 后停滞 | **11121/11121 PASS** | 33363==33363 | ~1050s |
| F7 120 strategy2 | 26846 交付后楔死 | **11121/11121 PASS** | 33363==33363 | ~1090s |

   - 七场全部 cpp_exit=0 python_exit=0、completed==物化请求数、
     **末笔 ack_count==delivery_count 恒成立**（缺陷 A 不变量）、
     no_decision_python_callback_count=0、无孤儿/无停滞（原 120 档
     0/4 → 4/4）；runner 尾部 artifacts 计数行 exit=126 为副本
     runner 未含 backport-fix1 的 find 修复（ls 通配 ARG_MAX），
     系脚本收尾行非仿真门（cpp/python exit 与审计已在 runner 内
     判过）——已按同款 find 修复补丁四副本全部 16 个在线 runner
     （4 副本 × replay/strategy/sensing/legacy，登记，不重跑）。
   - **20 档三路径复测全过**（2091/2091，机器仍有并行 agent）：
     replay exit=0（93s，delivery==ack==6264）、strategy exit=0
     （99s，delivery==ack==6273）；静态路径 exit=0，raw_metrics
     244 行对对比测试期原始 3 分钟静态运行（run_logs 103724 副本
     后处理重导出）剔 run_id/wall_time_ns 后**逐字段 0 差异**——
     静态基线在缺陷修复后零漂移；副本原有归档产物未受影响（本轮
     静态重跑产物独立后处理核对，未覆写 results/ 归档位）。
4. **非缺陷登记项复核**：--request-max-arrival-ns 默认无界
   （backport-fix1）+ 副本 runner 显式 180e9 护栏语义正常（本轮七场
   重跑输入全部 <180e9，dropped=0）；tier_b_compare.py 期望常量
   硬编码 30s 物化值（1177/112）维持已知限制登记（30s 包络验收
   输入即该值，3 分钟口径已在副本中改写，逻辑未动）。

## 验收工具修复（2026-08-16；tag：face-tooling-fix-done）

> 两件待入库补丁落地：diff_explainability.py 假绿补丁（来源：
> diff_explainability缺陷排查报告.md §六）与 ledger_reconcile.py
> 对账加固+参数化（来源：开感知120报告_3mins.md §5 建议）。
> 均为 verify 工具层改动，仿真/调度本体与策略红线零触碰。

### 1. diff_explainability.py 假绿补丁入库（缺陷 3）

- 应用：以三可写仓修复版（sh_2.0 55f0893 同款 263 行补丁）覆盖本仓
  同路径文件——本仓缺陷版（md5 08db8c12）与三仓修复前版本逐字节
  一致，覆盖即等价于补丁原样应用。入库后 md5 =
  **5f72c396f21ee5fa12a4f2c1f9e87370**，与 sh_2.0/sh_3.0/wscllm
  修复版逐字节一致（cmp 实测），py_compile 通过。
- 内容（与三仓同版）：kv_actions/assignments 维度数据源迁移——
  KV 决策载荷流（online_decision_log.jsonl 每行 decision 投影）+
  批级 assignment 摘要流（graph_batch_digests.jsonl）；空数据源
  全维度 fail-closed（jsonl 缺失/0 行/缺 decision 载荷/无
  request_*.json/cpp.log 零计数器 → exit 2 不产报告）；jsonl 双布局
  解析（bridge/ 优先、results/ 次选）。
- 真实比较复验（30s 既有产物对 /tmp/face_strategy_run2 vs
  /tmp/face_sensing_run1，只读）：
  - exit 0；KV 载荷 **3531 行 sha256=c4c5ecf7b2ad849d**、assignment
    **3531 批 sha256=70d53f1b123bdbbc**——与排查报告 §五 face 布局
    登记值逐字一致，非空哈希（空数组签名 4f53cda1 消失）；
  - mutant 实测（sensing 副本篡改 1 个 decision 数值载荷字段）：
    DIFF 行两哈希相异（baseline=c4c5ecf7… sensing=f0f665f6…），
    比较为真；
  - fail-closed 实测（空 run 目录）：stderr 报错 exit 2，不产报告。

### 2. ledger_reconcile.py 对账加固与参数化

- 加固（wscllm 054118e 同款不变量原样适用，新增 R2b/R2c）：
  - R2b 批级摘要不变量：graph_batch_digests.jsonl 每批
    watch_count ∈ [0, len(ranks)] 且 **watch_count==0 ⇔ ranks 空**
    （空性等价；本仓 30s 实测零违例；watch_count=2 批实存——PREFILL_
    DRAIN 同批 prefill+decode 双 watch，禁 {0,1} 假设，本轮 30s run
    分布 {0:1197, 1:2314, 2:20}）；ranks ⊆ [0,53]；
    sum(node_count) == C++ phase-5 total_nodes、sum(watch_count)
    == phase-5 total_watches（30s 实测 298222/2354 双对平）；
    digests jsonl 缺失/0 行 = 数据源缺失 fail-closed（不静默跳过）；
  - R2c phase-4 end audit 全零（watch 生命周期干净收尾）；
  - 新增 load_cpp_phase_counters()（phase-4/phase-5 正则解析）与
    load_digests()。
- 参数化（开感知120报告_3mins.md §5 建议）：R0a/R0e 的输入期望常量
  硬编码 30s 验收值（1177/112）改命令行参数
  `--expected-requests`（缺省 1177）/`--expected-accepted-sessions`
  （缺省 112），全部 8 处期望字串随之改实际参数值；报告头
  "1177 requests / 112 sessions"陈旧文案改实测值+期望参数（120 报告
  登记的外观性问题一并消除）。
- 复验（30s sensing run1 + 仓内 manifest，只读产物）：
  - 加固+参数化后 R0-R8 + R2b/R2c **全 PASS（对平 balanced）**，
    与加固前基线（balanced）零回退；
  - 显式等值传参路径复跑 balanced；负向验证（传 11121/814 模拟
    120 档常量）：仅 R0a/R0e FAIL（manifest=1177 expected=11121、
    accepted=112 expected=814）——与 120 报告 §4.1 观测签名一致，
    现经 CLI 控制无需改码。
- 复验期临时物化（traces 队列 + trace_config 指向）验后已恢复
  （config 回 placeholder、物化 csv 删除）；恢复后双根 pytest
  **57 passed**（workload 24 + tests 33）复跑全绿。

### git

- commit（pathspec 仅本仓：diff_explainability.py +
  ledger_reconcile.py + 实录补记）+ tag `face-tooling-fix-done`。
