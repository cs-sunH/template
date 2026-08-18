# wscllm 仓改造执行实录补记（回灌修复）

> 本文是 `wscllm仓库改造详细执行方案.md` §15/§16 执行实录的补记载体
> （本仓无独立实录文件；阶段 0-7 实录整合于仓外方案文档 §15）。
> 回灌来源：`/home/sunhao/wsc-simulator/sh_2.0测试/对比报告.md` §5
> （2026-08-16 四档 3 分钟对比测试发现的同源缺陷，两处源于本仓
> 蓝本机制层，五仓复制传播）。本仓为机制层蓝本仓：本次修复即
> **蓝本修正**；机制层文件与 template/astra-sim-face 同步修改并保持
> 逐字节一致（血统一致约束，cmp 复核）。
> tag：`wscllm-backport-fix1-done`。

## A. OnlineCli 默认窗口 fail-closed 化（对比报告 §5.1）

- 缺陷（本仓蓝本层）：`--request-max-arrival-ns` 默认 30e9 = 把 30s
  验收输入窗口烧进代码默认值；超窗 turn-0 请求被 WindowedTraceReader
  静默丢弃，且被丢弃行永不消费 → 窗口占用钉死在 high_water、pump 在
  EOF 前失速 → expected_requests 停留 0、完成度审计（completed ==
  已读行数）被整体跳过（sh_2.0 strategy-20 实测：输入 2091、完成
  1830、261 请求静默缺失仍 exit 0 "PASS"）——request-neutral 与
  fail-closed 双违反。
- 蓝本修正（与 face 同文件同改动）：
  1. 默认改**无界**（0 = 不设上限）：OnlineCli.cc/.hh 默认值、
     WindowedTraceReader 构造默认参数 30e9 → 0；拒绝判定加
     `max_arrival_ns_ > 0 &&` 守卫。窗口保留为显式实验旋钮，任何丢弃
     可见且 fail。
  2. 完成度审计分母改 **total rows**：新增 count-only 尾扫
     `count_remaining_data_rows()`（不 Submit、不注册 queue_index/
     metrics，仅计数并区分 turn-0/turn>0）与 `total_data_rows()`/
     `turn0_data_rows()`；main_online 主循环结束后对 CSV 运行尾扫
     兜底（正常 EOF 路径 no-op），失速窗口无法再缩小审计分母。
  3. 审计算术抽为可单测函数 `audit_completion()`（四判定：Dropped=
     dropped>0 任何显式窗口丢弃即失败；AccountMismatch=accepted+dropped
     != turn0_rows；Incomplete=completed+dropped != total_rows；Ok）。
     计数语义按官方运行实测校准：accepted 仅计 turn-0 提交（112/1177），
     turn>0 走 future-alarm 路径不计入（face replay 实测 accepted=112
     证伪首版 accepted+dropped==total 不变量后修正为 turn-0 口径）。
  4. 启动行与窗口阅读器 report 行增列 unbounded 提示与 total_rows/
     turn0_rows。
- 合成 fixture 测试（不依赖物化 csv）：windowed_trace_reader_test.cc
  新增 Part G（默认无界：构造默认参 + CLI 默认参均 0，180s 到达全接受）
  与 Part H（显式小窗 25s：丢弃计数可见、失速窗口尾扫恢复 total=7/
  turn0=5 分母、尾扫零 Submit、audit_completion 对 sh_2.0 strategy-20
  形状判 Dropped 及 AccountMismatch/Incomplete/Ok 各形态）；
  cli_online_test.cc R11 默认断言 30e9 → 0。本仓 WindowedReaderTest
  A-H ALL PASS、cli R1-R11 ALL PASS。

## B. runner REQ_COUNT ARG_MAX 修复（对比报告 §5.3）

- 缺陷：在线 runner 尾段 `REQ_COUNT=$(ls .../request_*.json | wc -l)`
  在保留 envelope 数 >约 2e4 时通配展开超 ARG_MAX（E2BIG → bash
  exit 126，`set -euo pipefail` 下终止脚本）。sh_2.0 实测阈值介于
  13,864（未触发）与 22,012（触发）之间。
- 修复：本仓 run_online_replay.sh / run_online_strategy.sh /
  run_online_strategy_legacy.sh / run_online_strategy_sensing.sh 四脚本
  的全部两处 request-glob 计数行（FAIL 诊断行 request= 与尾段
  REQ_COUNT=）改 `find "${RUN_DIR}/bridge" -maxdepth 1 -name
  'request_*.json' | wc -l`。CP_COUNT（1 个/run）与 jsonl 计数（固定
  文件名）构造有界，排查过不适用未改。bash -n 语法全过。

## C. turn-0 逐出 KeyError 同型排查（对比报告 §5.2）——不适用，登记

- sh_2.0 缺陷形态：builder `_mark_pending_history_store` 假定
  pending_request_by_session 里的 pending request 必有 pending_history
  gate；turn-0 在 prefill 发射时登记 session 键而无 gate，逐出命中该
  会话时直接下标 pending_history → KeyError。
- 本仓排查（含两个变体的 builder 路径）：**不适用**。本仓
  online/graph_batch_builder.py（574 行）中
  `pending_request_by_session` / `pending_history` /
  `deferred_session_locations` / `_mark_pending_history_store` 全部
  0 命中（grep 实测）；两变体调度器（wsc_llm_online_scheduler.py =
  session_lru_recompute 主变体、wsc_llm_legacy_online_scheduler.py =
  legacy 变体）与 wsc_llm_replay_scheduler.py 共用这唯一 builder，
  三者均无逐会话 pending-历史位置账本。逐出在本仓为 scheduler 侧
  kv_manager 记录（admission/decode_target/completion evictions），
  逐 request 当批翻译成图，无"pending 账本 + 逐出路径下标访问 +
  turn-0 无 gate"的组合。本仓离线 generate_wsc_llm_trace.py 亦无
  pending_request_by_session（该机制为 sh_2.0 谱系在其
  generate_face_trace.py:2178+ 的独有演进，其在线复刻引入了缺陷）。
  无需修复，不加测试。

## D. teardown unreleased nodes 观测（对比报告 §4.3）——已观测，对齐登记

- 本次复验本仓在线运行 cpp.log 退出期实测：replay 2 行
  （sys.id=32/33 `!!!Hardware Resource ... unreleased nodes!!!`）、
  strategy 6 行——与 sh_2.0 §4.3（replay 每档 2 sys、strategy 6 sys）
  同族同量级。与蓝本实录 B17（方案 §15 bug 表：sys.id=32/33 node
  13610 = 最后 request end barrier 的 1ns 事件在结束权下未处理，
  **判定良性结束伪影**，步骤 1-8/1-9 在案）sys.id 与根因逐字对齐——
  即该观测在蓝本 30s 验收阶段已存在且已判定（对比报告"30s 验收阶段
  未见记载"系未在 §15 之外单独登记）。与 sh_2.0 侧根因分析（另一
  agent）结论一致：end-barrier 1ns 控制事件在完成权下未释放，良性
  退出期伪影，运行本体全部通过。登记为观测项，不修改。

## 复验证据（20.csv 前 30s，1177/112）

1. **重物化**：traces/ 物化器 + PROVENANCE 自 git 历史（7801566）
   恢复重建入口，对源 csv（md5 fc74a48e...）重派生：1177/112、
   prefill 66-169395、decode 1-13812、max arrival 29.988879s；queue
   与 sidecar 对冻结血统逐字节 cmp 一致。
2. **编译 + 单测**：build_analytical_aware.sh 全目标编译通过；
   WindowedReaderTest A-H ALL PASS、cli R1-R11 ALL PASS、
   NodeStore/WatchRegistry/DecisionMailbox/GraphBatchCommitter 回归
   通过；IDLE fixture 与 same-tick milestone (a)-(d) ALL PASS。
3. **静态字节门**：runall exit 0（1177/1177，incomplete=0），raw
   metrics 234 行对冻结基线（55cf135^ 归档）剔除 run_id/wall_time_ns
   后逐字段 0 差异——蓝本修正对静态路径零扰动。
4. **在线路径（双变体各跑）**：
   - replay（决策日志 md5 12a10345... 与 §15 登记一致）：PASS，
     delivery 3491、completed 1177/1177、accepted=112、turn0_rows=112、
     total_rows=1177、dropped=0、no_decision=0、single_node_bridge=0、
     avg_nodes_per_batch 105.94/max 460（与 §15 阶段 5 登记一致）；
   - strategy（主变体 session_lru_recompute）：PASS，delivery 3531、
     1177/1177、single_node=0、avg 98.66/max 350、total_nodes 348374
     （与 §15 阶段 5/38 登记一致）；
   - strategy legacy（第二变体，legacy_gen 现物化 pc4954）：PASS，
     delivery 3531、1177/1177、single_node=0。
   - 完成度审计（新 fail-closed 口径）三 run 全部 Ok。
5. **pytest**：双根 68 passed（workload 根 35 =
   test_wsc_llm_scheduler + test_wsc_llm_legacy_online_scheduler +
   test_checkpointing；sh_test_mesh/tests 33）。开发期 trace_config
   指向物化队列时 request-neutral 占位断言用例按设计 fail（临时态），
   恢复裸仓态后全绿。
6. **裸仓态恢复**：物化输入（traces/）、运行产物（results/）、开发期
   生成的 ET 目录（pc512 主 + pc4954 legacy）删除，generated/ 恢复为
   仅 runtime_config；trace_config.csv / trace_config_legacy.csv 回
   placeholder（git checkout）；fail-closed 实测 `--print-shell-config`
   exit=1；pytest 复跑 35+33 全绿。
7. **登记备注**：本次物化 legacy 基线目录 label 为
   ..._pc4954_..._q23936bbc_c4b4b47e7（configuration_digest 现值），
   与 §15.7 历史登记 ..._ce3b40440 不同——硬件/系统模板/配置文件对
   7801566 逐字节一致，差异为历史基线生成于更早的输入状态；runner
   以显式 <legacy_gen> 参数消费，机制无影响，如实登记。

## E. diff_explainability 缺陷 3 修复（kv_actions/assignments 真空通过，2026-08-16）

开感知验证追记立档的缺陷 3（问题总结文档【缺陷 3】）：`online/verify/
diff_explainability.py` 的 `_load_run()` 仍从 response_*.json 收集
kv_actions/assignments（phase-7 §10.3 起消费即删），两侧恒读成空列表，
空==空打出 PASS 假绿，签名 sha256=4f53cda1… 即空数组 [] 的哈希。054118e
修 ledger_reconcile 时未覆盖到本文件，本次补修。

### 处置（verify 工具层，仿真/调度本体零改动）

与 sh_2.0/sh_3.0 同版修复（md5 5f72c396）：

1. kv_actions 维度迁 online_decision_log.jsonl 的 KV 决策载荷流
   （每行 decision 字段投影 {kind, request_id, decision}，tick/seq
   无关；wscllm 载荷字段 history_action/prefill_assignment_key/
   kv_state_after_completion 族——按载荷整体稳定摘要，不耦合字段名）。
2. assignments 维度迁 graph_batch_digests.jsonl 批级摘要流
   （delivery_sequence/reasons/ranks/node_count/edge_count/
   watch_count/content_sha256，content_sha256 即批内 nodes+
   parent_edges 内容摘要）。
3. 空数据源 fail-closed（禁止空==空通过）：jsonl 缺失（bridge/ 优先、
   results/ 次选，两布局均支持）或 0 行、决策行缺 decision 载荷、无
   request_*.json、cpp.log 缺失或零计数器——exit 2 不产报告。

### 复验（真实产物，/tmp 残留 + face测试归档）

- 缺陷复现：face 30s 对（face_strategy_run2 vs face_sensing_run1）与
  20 档 3min 对（face测试 face_20_3mins，response 计数实测 0）缺陷版
  均打出 `PASS kv_actions 一致 sha256=4f53cda18c2baa0c`——假绿实证。
- 修复后同对产物：30s 对 KV 载荷 3531 行 sha256=c4c5ecf7b2ad849d、
  assignment 摘要 3531 批 sha256=70d53f1b123bdbbc；20 档 3min 对
  6273 行/6273 批（41deb73f/1c67a6f6）；50/80 档 13941/22269 行同
  PASS——签名 4f53cda1 消失，维度真实比较出结果（全部一致）。
- mutant 实测：篡改一侧 1 个载荷字段 → KV 载荷维度 DIFF（两哈希相异），
  比较为真；fail-closed 5 例实测 exit 2。

### git
- commit（pathspec 仅本仓）+ tag `wscllm-diffexp-fix-done`。

## F. 在线机制层竞态缺陷同步修复（2026-08-16，来源：face 0049ef5；tag：wscllm-defectfix2-done）

> 本仓为机制层蓝本仓。face 四档 3 分钟主动测试暴露的三类在线机制层
> 竞态缺陷（face主动测试错误分析.md）源出本仓蓝本、五仓同源同病；
> face 已修（0049ef5，双侧长连接方案经主 agent 批准），本节为同步
> 移植与复验实录。**蓝本再修正**：本次修复即对本仓机制层的历史缺陷
> 补修（face 错误分析 §3 已坦白"蓝本潜伏竞态遗传"为主解释——本仓
> 同规模四档双模式当时在更安静环境通过，竞态窗口固有存在）。

### 移植内容（机制层文件与 face 0049ef5 修复版逐字节一致，cmp 复核）

| 修复 | 文件 | 移植语义 |
|---|---|---|
| A 计数真空 | main_online.cc | 主循环 pump() 后、finished() 前补 drain（pending_command_count>0 → drain_commands，关闭"pump 入队未 drain"真空）；belt-and-braces：finished 真而 CSV 未 EOF → online_fatal（active/pending_alarm/queued 计数入消息） |
| B 长连接 | DecisionBridge.hh/.cc + online/decision_bridge.py | resp 通道双侧常开 fd（C++ open_notify 一次开 O_RDONLY\|O_NONBLOCK 持有至析构；Python serve_forever 启动一次 O_WRONLY 常开）；wait_response_byte 单字节 poll→read、EAGAIN 重 poll；read==0 = 对端死亡（真语义）；1:1 在途守卫（第二字节即 protocol violation）；Python BrokenPipe → BridgePipeError fail-closed + stderr 留痕；_fail 先留痕后退出 |
| C 唤醒条件 | main_online.cc 空队列分支三分支化 + EventQueue.h/.cpp | ①mailbox 残留→T+1 显式唤醒（不变）；②新增 EventQueue::has_deferred_work() 访问器，deferred 残留→T+1 强制下一决策边界；③input 已关+队列/mailbox 双空+svc 未完→lost-wakeup dead end fail-closed（全计数诊断）；④input 开→wait_for_work（IDLE 合同不变） |

- EventQueue 属 extern 共享层（纯新增 const 访问器，静态二进制零行为
  变化——静态字节门 0 差异实证）。
- 双变体（wsc_llm_online_scheduler=session_lru_recompute /
  wsc_llm_legacy_online_scheduler=legacy）共用同一 decision_bridge.py
  与 C++ 在线入口，一处移植两变体同步覆盖（legacy runner 实测全过）。

### 与既有蓝本缺陷的收敛关系（登记，主 agent 收敛口径）

1. **sh_2.0 发现的 wait_for_work `!input_open_` 忙等蓝本缺陷**：
   ServiceCoordinator::wait_for_work 的谓词含 `!input_open_`
   （ServiceCoordinator.cc:104）——input 关闭而服务未 finished
   （active>0 无可触发事件）时条件变量立即返回，主循环无进展空转
   （sh_2.0 排查发现的蓝本忙等形态）。**本仓 C 修复的死端 fail-closed
   分支恰好覆盖该路径**：input_closed() && event_queue->finished() &&
   !mailbox.has_decision_work() && !has_deferred_work() && !svc.finished()
   → 立即 abort 并携带 active/pending_alarm/deferred 计数，不再进入
   wait_for_work。wakeup_guard 场景 2（defer 空批 + CloseInput）实测：
   数秒内 fail-closed（cpp_exit=134，cpp.log 含 "lost-wakeup dead end"
   且 active=1 入消息），忙转/永挂路径封死。**收敛**：sh_2.0 形态与
   face F5/F6/F7 同族（服务未终判+无事件+输入关），本仓按 face 移植
   登记的"服务终判/等待/死端"族统一收敛为死端 fail-closed 口径。
2. **sh_1.0 的 30s 饥饿 guard 同族**（终判计数不覆盖在途工作）：本仓
   统一收敛为"判终前 drain + belt-and-braces 审计"（A 修复），不再各贴
   各的 guard。
3. **startup 真空观察（登记，非新缺陷）**：--close-input + 队列 CSV 的
   官方形态下，启动序 pump→mark_input_closed→maybe_finish 会在 t=0 把
   state 置 FINISHED（此时 Submits 尚未 drain，计数真空）——face 健康
   产物（/tmp/face_strategy_run2）同款序列；主循环判终用计数器口径，
   首 drain 后 pending_alarm>0 即继续正常运行，属共享的既有外观性行为，
   A 修复的主循环真空已封，startup state 枚举早置不影响终判权威性。

### fixtures 移植（按本仓 CMake/路径适配）

- ServiceVacuumTest（service_vacuum_test.cc + CMake 目标，与 face 同版
  逐字节一致）：legacy 顺序确定性复现真空（break 时 completed=10/20、
  队列 10 条未 drain、未 EOF、审计非 Ok）→ fixed 顺序 30/30 + Ok；
- bridge_loopback_fixture.cc Part D/E（与 face 同版）：D=SIGKILL 真死检
  （消息含 "long-lived resp_notify write end closed"）、E=1:1 守卫
  （"more than one response byte in flight"）；A-C 语义保持全过；
- bridge_cpp_death_fixture.py（纯 Python 假 C++，与 face 同版）：3/3
  exit 1 + stderr "C++ side is gone"；
- wakeup_guard_fixture_service.py + run_online_wakeup_guard_fixture.sh
  （**唯一适配点**：ET_PREFIX 改本仓
  llama2_7b_wsc_llm_inference_*_pc512_*_c38c794e6 目录、RUN_ROOT 缺省
  /tmp/wscllm_wakeup_guard，其余与 face 逐字节一致）：f7-blueprint
  健康蓝图（accepted=1/completed=2/无死锁/未误触发）+ defer-dead-end
  fail-closed 双场景 PASS；
- bridge_race_stress_repro.sh（与 face 同版）：旧协议竞态复现装置，
  本机实测第 69 次迭代触发假 EOF（POLLHUP+read==0 写端存活）。

### 复验证据（20.csv 前 30s，1177/112）

1. **重物化**：traces/ 物化器 + PROVENANCE 自 git 历史（55cf135^）恢复，
   源 csv（md5 fc74a48e...）重派生 1177/112 与冻结血统逐字节 cmp 一致；
   trace_config.csv 用 7801566 原字节（第 12 行 description 差异会使
   configuration_digest 漂移）生成 ET 目录——pc512 主目录名复现冻结值
   `..._c38c794e6`（runner 硬编码一致）、legacy 现值
   `..._pc4954_..._c4b4b47e7`（与 §复验证据 7 登记同款）；决策日志
   --replay-record 再生 md5 12a10345... 与 §15 登记一致。
2. **编译 + 单测**：build 全目标 0 error；ServiceVacuum / BridgeLoopback
   A-E / NodeStore（--fixture-et）/ WatchRegistry / DecisionMailbox /
   GraphBatchCommitter / WindowedReader A-H / cli R1-R11（g++ 现编）全过。
3. **静态字节门**：runall exit 0（1177/1177，incomplete=0），raw_metrics
   234 行对冻结基线（55cf135^ 归档）剔 run_id/wall_time_ns **逐字段
   0 差异**——A/B/C 对静态路径零扰动。
4. **在线三变体**（门控：等待另一 agent 的 sh_2.0 120 档 sensing 结束
   （04:44）后发射；后续 sh_1.0 agent 的 replay 并存期间跑 30s 包络短
   仿真，与 face 0049ef5 复验的"并行 agent 共存"同款口径，内存口径
   38-46% << 80%）：
   - replay：PASS，delivery 3491、completed 1177/1177、accepted=112、
     no_decision=0、single_node_bridge=0、**末笔 ack_count==delivery_count
     ==3491**（缺陷 A 不变量）；
   - strategy（主变体）：PASS，delivery 3531、1177/1177、
     single_node=0、avg 98.66/max 350、total_nodes 348374、watch_sum
     2354、phase-4 end audit 全零——与 §15/回灌轮登记数字逐项一致；
     末笔 ack==delivery==3531；
   - strategy legacy：PASS，delivery 3531、1177/1177、
     no_decision=0、single_node=0、末笔 ack==delivery==3531。
5. **Tier B 工具**：B0/B1/B4 PASS（对本次 replay/strategy 运行产物）；
   B2/B3 为排查报告 §三登记的陈旧数据源 fail-loud（response_*.json
   消费即删契约，断言崩溃非假绿）——tier_b_compare.py 本次零触碰、
   与 3086613 态逐字节一致，非本次改动引入；face 版已在迁移期适配
   （两仓工具谱系差异登记，不在本次同步范围）。
6. **fixtures**：IDLE 五态迁移 ALL PASS；same-tick milestone (a)-(d)
   ALL PASS（既有 T+1 唤醒合同未回归）。
7. **使用注意（登记）**：runner 的 request_csv/decision_log 参数必须传
   绝对路径（python 侧 cd 至 workload 目录后按相对路径解析；误传相对
   路径会致 python 启动即 fatal，C++ 停在 open_notify 的 ENXIO 重试环
   等待永不出现的 python 读端——timeout=0 无界）。本轮一次误用实测
   该形态并清理后以绝对路径复跑全过。
8. **裸仓态恢复**：物化输入（traces/）、基线归档（baseline/）、运行
   产物（results/）、开发期 ET 目录（pc512+pc4954）与 completion_
   fixture 全部删除，generated/ 恢复仅 runtime_config；trace_config
   两文件回 placeholder（git checkout）；fail-closed 实测
   --print-shell-config exit=1；双根 pytest 复跑 **35+33=68 passed**
   全绿（request-neutral 占位断言用例恢复通过）。

### git

- commit（pathspec 仅本仓：机制层 6 文件 + fixtures 5 件 + CMakeLists
  + 补记）+ tag `wscllm-defectfix2-done`。
