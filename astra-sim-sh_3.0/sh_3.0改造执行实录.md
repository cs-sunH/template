# sh_3.0 改造执行实录

> 执行 agent 维护的实时修改实录（任务指令要求）。对应方案：
> /home/sunhao/wsc-simulator/sh_3.0仓库改造详细执行方案.md（下称"方案"）。
> 记录合一政策的正式载体是方案 §15/§16/附录 C（主控填写）；本文件是执行期
> 工作实录，最终由主控审核整合。
> 起点 commit：55cf135a7cafcc07fc6a01a6730c0e5708066bc7（执行时实测）。
> 注：本任务指令明确"不要 git commit"，故未做阶段提交/tag；git 状态为
> 工作树改动（可由主控审核后统一提交）。

## 阶段 0（全部完成）

### 步骤 0-0 起点
- HEAD = 55cf135（与方案记载一致）；工作树对本仓干净（父仓有若干用户未
  跟踪文档，未触碰）。

### 步骤 0-1 输入物化（sidecar_restore 三件套）
- 新建 `traces/materialize_20_30s.py`（单次遍历同源产出三件套，规则①-④
  与蓝本 30s 物化同窗口同规则）。
- 实测：**1177 请求 / 112 session**（与蓝本一致）；turn-0 prefix>0 行 79；
  max input_tokens_total=169,395；max 窗口内到达 29.988879s；非 1000ns
  倍数行 0。
- md5：queue=8b615b5a12b15143a10bbbf13f31a6bb、context=25b7da35rac...（见
  traces/PROVENANCE.md 权威值）。
- `trace_config.csv` :12/:13 改指仓内 traces/（queue + context sidecar）。
- fail-closed：`generate_face_trace.py` load_face_trace_config 在调用
  load_request_queue 前检查存在性，缺失 sys.exit(1)；反向实测 exit=1 +
  fail-closed 信息，无 stub 生成。
- 预存失败用例期望值更新（:194 与 :417 块）：2091/136 → 1177/112，
  PREFILL_RANGE 80-169395，路径/前缀断言改 traces/ 物化口径，新增
  `input_tokens_total == prefix_tokens + prefill_length` 全量断言与
  turn-0 prefill work == input_tokens_total 断言（sidecar_restore 生效核验）。
  授权依据：主任务指令的完成标准明确"pytest 全绿"，视为蓝本先例（阶段 0
  步骤 0-6 主控授权）的同款授权；test 417 名保留 "three_minute_window"
  未改名（方案允许，登记"名不符实"已知项）。
- pytest：42 passed（test_face_scheduler.py）+ 33 passed（sh_test_mesh/tests/）。

### 步骤 0-2 静态基线归档
- runall 首跑成功：1177/1177 完成、incomplete=0、memory_actions 64914/0、
  C++ 仿真 10s、全管线 55-79s。
- `run_sh_test_aware.sh` 补 RUN_OUTPUT_LOG_TIMESTAMP 环境变量覆盖（默认
  行为不变）。
- 确定性：固定时间戳复跑，2218 行 [METRIC]（剔除 run_id/wall_time_ns）
  逐行相等（equal: True）。
- 归档 `sh_test_mesh/baseline/20_30s/`（generated 54 .et + manifest +
  metrics_manifest + face_lut + raw/normalized_metrics.csv + run log +
  PROVENANCE.md）；二进制 md5 8ead82c5ff596b4ab060d2517945474f。

### 步骤 0-4 九项合同冻结
- `online_contracts/nine_contracts/` 九份（contract_01..09），每份含
  口径/裁决/验证方法三要素；sh_3.0 特有内容：两段式发射边界、HBM/远端
  transition 非 decision reason、不设 TRANSFER_CONFIRMED、八层账本全实建
  + task_load_snapshot 三分量映射、replay 时钟口径第④项（远端 MEM 与
  HBM restore 即时完成）、sidecar_restore canonical、average_decode_length
  标定常数（459.4944774851317）。
- LUT 复核（合同⑨ 附记录）：lut.lookup 仅 :3853（事件时钟）；:1011/:1017
  在 select_decode_instance（保留不调用设施）；策略消费的全量派生量只有
  average_decode_length——**无反例**。

### 步骤 0-5 决策日志
- `--replay-record` CLI（方案 A）：强制 record_iterations=True 的专用规划
  运行；写 decision_log.jsonl（293,801 行 = 1177×3 决策 + 290,270
  iteration 记录），md5 7452f0fa89f83f120fa90d985bfdd60d。
- 字节等价门：--replay-record 运行的 54 .et + manifest + metrics_manifest
  与归档逐文件 cmp 全部一致（replay plan 经 dataclasses.replace 剥离
  iterations 再写 trace，manifest 保持生产形状）。
- bug 修复记录：模块内已有 `_kv_transfer_dict(transfer, shard_records)`
  （metrics 用），新增的 decision-log 序列化改名
  `_replay_kv_transfer_dict` 避免覆盖。

### 步骤 0-7 压测基线
- /usr/bin/time -v runall：wall 55s（C++ 仿真 10s）、峰值 RSS 400.6 MiB、
  generated/ 116 MiB、results/ 1.5 MiB（PROVENANCE 压测段）。

## 阶段 1（进行中）

### 步骤 1-1 EventQueue（map 版）tick 收口 ✅
- EventQueue.h/.cpp：set_tick_end_callback / schedule_event_deferred /
  in_invoke_context + deferred_queue_（std::list<EventList>，与主队列 map
  容器无关）；proceed 顺序 = invoke → erase(it) → tick-end 回调 → deferred
  drain（§4.2 逐行：erase 后不得触碰列表引用；硬规则注释入码）。
- FluidScheduler deferred flush 模式 + CommonNetworkApi sim_recv 分流：
  两文件与蓝本改造前逐字节相同，直接采用蓝本已打补丁版本（附带带入
  蓝本阶段 7 的 LinkCongestionSnapshot 只读访问器——方案 §10.2 本就要求，
  登记）。
- 单元测试（map 版，用例 A-D + 本仓新增 E）：A 收口顺序 / B deferred
  插入序+嵌套同轮 / C 1000 随机与 map 版 reference（begin/erase/
  try_emplace 镜像）逐事件一致 / D FluidScheduler 集成两模式 / E1
  invoke 上下文同刻合批（try_emplace 合并）/ E2 tick-end 回调内
  schedule_event(current_time) 下一轮 proceed 触发 :31 断言（fork 死亡
  测试）。全部 PASS。

### 步骤 1-0 ED 机制层移植 ✅（构建）
- `astra-sim/workload/execution_driven/`（17 文件 + tests/11 文件）与
  `main_online.cc` 从 wscllm 蓝本整体移植（文件名/类名/测试名一致；
  注释去 wscllm 品牌）。
- Workload.hh/cc、Sys.hh/cc、Statistics、HardwareResource、MetricCollector、
  两级 CMakeLists 按 wscllm diff 适配落地（Workload.cc 6 个 rejected hunk
  手工重写：issue_dep_free_nodes/issue/issue_replay/issue_remote_mem/
  issue_local_hbm_kv_restore/issue_comp/call 结束判定）。
- sh_3.0 特有适配：
  - NodeView 新增 MemAttrs{tensor_size, is_local_hbm_kv_restore}
    （GraphSource.hh + GraphBatchCommitter 解析 + ETFeederGraphSource 适配）；
  - HardwareResource NodeView 重载增加 hbm_dma 第四槽位分类（与静态
    ETFeederNode 路径同语义：remote MEM 节点落 comm 槽）；
  - Workload 结束判定在线模式整体跳过（含 has_active_jobs 条件）；
  - HBM 模型创建分支在线路径保留。
- 全部目标编译通过（含 8 个测试目标 + Online 二进制）。
- **静态基线字节门**：ED 构建后 run_sh_test_aware 复跑，54 .et + manifest
  等全部产物与归档 cmp 一致；2218 行 [METRIC] 一致。

### 步骤 1-3 CompletionObserver ✅（含本仓补充维度）
- 三条终态路径挂载（skip_invalid / collective / wlhd 通用分支）。
- 第四终态入口审计：LocalHbmBandwidthModel（workload->call(General, wlhd)
  :285）与 AnalyticalRemoteMemory（start_request 双事件 :195-200 的
  workload 侧）均汇入 wlhd 通用分支——已覆盖；transition/FIFO 推进事件
  不触碰 Workload。
- hook 计数 fixture 扩展：node 3 = 远端 MEM_LOAD（26 边界端口 rank 才
  发射；remote_memory.json npu-ids 实测 26 rank）、node 4 =
  is_local_hbm_kv_restore MEM_LOAD。实测 hook_total=242（54×4+26）、
  site_skip=54/site_generic=54/site_coll=54/site_remote_mem=26/
  site_hbm_restore=54，ALL PASS（exit 0）。
- NodeStore/WatchRegistry/DecisionMailbox/GraphBatchCommitter/CLI/
  BridgeLoopback/IDLE fixture（6 场景含 EOF 三态/溢出审计/Error
  SIGABRT）全部 PASS。

### 步骤 1-8/1-9 Python 在线层（本次新增）
- online/ 移植 + 重写：graph_batch_builder.py（sh_3.0 版：两段式发射 +
  completion 批；复用离线 _emit_kv_transfer/_emit_tp_*_readiness_barrier/
  transformer_pass_aggregated + OnlineTraceBuilder 提供 TraceBuilder 全
  接口面含 chain_checkpoint/restore_chain/mem_store/mem_load/
  local_hbm_kv_restore；pending_history/deferred_session_locations 账本与
  离线 writer 同构）、sh30_replay_scheduler.py（manifest 事实 +
  decision_log 时序；KVTransfer dict→对象重建）、sh30_online_scheduler.py
  （三段式准入/decode 同实例/KVCacheManager 只读复用逐行迁移，注释标
  离线行号）、online_service.py 分发适配（sh_3.0 单变体）。
- runner：run_online_replay/strategy/sensing/idle_fixture/same_tick_
  milestone 五脚本从蓝本移植适配（ET label/RC profile/路径）。

### replay E2E 调试记录
- bug 1：runner 传相对 decision_log 路径，Python 服务 cwd 在 workload/
  下找不到文件 → C++ 桥 EOF 立即 FINISHED。修复：调用侧用绝对路径
  （runner 接口不变）。
- bug 2：`multiple inference jobs entered one NPU HBM model`——replay
  并发校准 COMP 链向 LocalHbmBandwidthModel 注册多 compute job（单槽位）。
  修复（合同⑦ 第④项落地）：Workload::issue_comp 在 replay_clock_ 且
  runtime_ns≠0 时不向 HBM 模型注册 job，直接 register_event(runtime_ns)；
  issue_local_hbm_kv_restore 与 issue_remote_mem 在 replay_clock_ 下 1ns
  即时完成（terminal/释放链完整保留）。strategy/静态路径不动。
- bug 3：decode 批 assignment 缺 prefill_instance_index → C++ 校验
  "negative instance index"。修复：补字段。
- bug 4：completion 批节点 stage 与 watch 覆盖集冲突（蓝本等式规则）。
  sh_3.0 裁决（登记合同①/⑤）：第三发射边界（completion 批）的节点使用
  stage="completion"；GraphBatchCommitter 校验放宽为 watch ⊆ node 且额外
  stage 仅允许 "completion"（node stage 闭集 {prefill, decode, completion}）。
- bug 5：离线 writer 的 pending-gate location 依赖 order_plans_for_static_
  emission 全局 KV 因果预排序；在线按决策边界序发射时跨 request
  remote_store 触发序反转导致 gate/planned 反转（实测 session_8_request_1：
  gate=remote_memory vs planned=local_hbm @ delivery 304）。sh_3.0 replay
  裁决（登记合同⑦/§13）：replay 模式归一化到 manifest 权威
  （reconcile 与 history_transfer 分支两处）；strategy 模式保持严格校验。
  物理时序由图依赖保证（store 节点挂触发 request 链）。
- bug 6：_LocationShim 缺 total_bytes（reconcile 访问）。补字段。
- **replay E2E 全量 PASS（/tmp/sh30_replayA，2026-08-16）**：
  completed_unique_requests=1177/1177、incomplete=0、
  memory_actions_total=64914、unresolved=0（与离线基线完全一致）；
  delivery_count=3525 = ack_count；online_decision_log 3531 行（1177×3）；
  sim_end_ns=995,087,425,606（vs 离线静态 1,714,696,296,631——replay LUT
  时钟口径差异，B4 归因类别①登记）。cpp_exit=0 python_exit=0。


## 待办（按方案）
- 步骤 1-10/1-11：IDLE fixture runner 实测、same-tick milestone fixture、
  replay/strategy 全量完成数与计数器断言。
- 阶段 2：Tier B B0-B4（tier_b_compare.py 按本仓产物字段适配）。
- 阶段 3-7：感知/StateDelta/GraphBatch/benchmark/补齐。


## 阶段 1 完成状态（2026-08-16 收尾）

### 全部门槛项实测
- **replay 模式全量 PASS**（/tmp/sh30_replayA）：1177/1177 完成、
  incomplete=0、memory_actions 64914/0（与离线基线一致）、
  delivery=ack=3525、online_decision_log 3531 行（1177×3）。
- **strategy 模式（关感知）全量 PASS**（/tmp/sh30_strat6）：1177/1177、
  no_decision_python_callback_count=0、delivery=ack=3531、
  gate counters event_count=4708 / coalescing 1.33 /
  tick_end_without_decision_count=5,866,550（允许非零，单独报告）。
- **IDLE/注入 fixture PASS**：五态迁移日志齐全（IDLE→ACTIVE→IDLE→
  DRAINING→FINISHED，含 EOF 三态/溢出审计/Error SIGABRT 六场景）。
- **post-commit same-tick milestone fixture PASS（1-11 CORE）**：
  PREFILL_DRAIN 不重入（单 tick 单次交付）、显式 T+1 唤醒
  （deferred_from_tick 登记）、decode 图下一 epoch 提交、零事件丢失。
  本仓适配：fixture 的 decode 节点改 is_cpu_op=True（本仓 system 模板
  roofline-enabled，GPU COMP 会被 roofline 计时 ~101ns 而非 runtime_ns=1；
  CPU comp 走 issue_replay 确定性 1ns——fixture-only 改动，登记）。
- **静态基线字节门（阶段 1 全部改动后复验）**：全量重建 + run_sh_test_
  aware 复跑，54 .et + manifest + metrics_manifest 与归档逐文件 cmp
  全部一致；pytest 42+33 全绿。
- **红线检查**：face_scheduler.py 零 diff；generate_face_trace.py diff
  仅含授权改动（fail-closed 拦截 + --replay-record 观测通道）。

### strategy 调试链（bug 记录）
- bug 7：_admit_pass 遗漏 ready frontier 标记（_try_admit_request 入队后
  未 add）→ 发射 pass 永远空转，全部 request 卡在 admitted 层（诊断：
  ONLINE_PROGRESS_PROBE 探针显示 store_pending_total=0 + queue_finished=1
  忙转）。修复：准入后置 frontier。
- bug 8：wait_for_work 在 input closed 时谓词恒真 → 空转（上述死锁的
  忙等形态；机制层代码与蓝本一致，蓝本未触发该状态）。修复根因后不再
  触发；探针保留（env 门控，零开销）。
- bug 9：assignment decode_instance_index=None（prefill 边界 decode 未
  选定）→ C++ "type must be number, but is null"。修复：填 prefill 实例
  （与最终 decode 实例恒等，红线 #4）。
- bug 10：_OnlineRequestRuntime 缺 completion_ns slot。补。
- bug 11：_on_request_complete 先发射 completion 批后 mark_complete/
  enforce_reserve → kv_location_after_completion=None 报错。修复：按
  离线 :4016-3042 顺序（先账本后发射）。
- bug 12（裁决，登记合同⑦/§13）：pending-gate location 派生缓存在两
  模式都会滞后于权威账本（strategy 实测 session_16_request_1：
  gate=local_hbm vs planned=remote_memory——同一 delivery 内准入逐出先
  于 gate 消费）。统一裁决：弹出 gate 时归一化到 history_location_before
  （replay 权威=manifest，strategy 权威=kv_manager 快照，两者即该字段
  本身）；物理时序由图依赖保证；结构性错误仍 fail-closed。

### 阶段 2 首轮（Tier B 检查报告）
- 详见 online/verify/tier_b_report_20.md：B0 PASS（容差 0）；B2 oracle
  exact（0/1177 赋值+reason 不匹配；decode 流 exact）；B1 差异全部可
  归因（准入排队 q：中位 25ms/最大 6.5s——q-吸收机制生效的证据 = decode
  流顺序 exact；completion 流同窗抖动重排）；B4 归因类别①（replay LUT
  时钟口径，含本仓第④项豁免）。
- 遗留收尾项：B3 canonical key 对照器适配；B1 两轮决定性验证
  （turn-0 alarm=记录 tick 变体 / q 阈值扫掠）。

### 阶段 3-7 状态
- 机制层（StateDelta v1/GraphBatch 两阶段提交/WindowedTraceReader/
  感知账本/计数器/检查点）已随蓝本整体移植并在上述运行中生效；
  sh_3.0 特有收尾（remote FIFO + local HBM job 层实账对账、
  LocalHbmBandwidthModel/AnalyticalRemoteMemory 只读访问器、感知
  打开运行、基准矩阵、窗口压测）为后续工作。

## 主控裁决确认（2026-08-16，收尾指令）

1. 三项关键裁决（GraphBatch stage 闭集+completion；pending-gate 归一化；
   replay 第④项校准 COMP 不入 HBM 模型）**批准**。
2. pytest 期望值更新授权确认；test 417 命名保持不改（本实录登记即可）。
3. 后续清单：B1 两轮决定性验证出结论性证据 → B3 对照器适配 + 正式
   Tier B 报告 → 阶段 3-7 → face 先例裸仓库还原（删物化输入/基线归档、
   配置中性化、测试合成 fixture 化、双根 pytest 全绿、fail-closed 实测、
   剩余差异登记）→ 最后一次性按阶段补齐 commit+tag（sh3-phase0..7-done
   + sh3-final-done）。
4. 每阶段结束自查红线 diff 并记录。

## 阶段 2-6 定稿记录（2026-08-16，续主控指令）

### 阶段 2 定稿
- B1 两轮决定性验证（证据入 tier_b_report_20.md）：Round 1 = 在线
  prefill 触发序与期望触发序完全一致 + 23,803 到达/准入反转中 0 对
  双方 q==0（100% 准入排队归因）+ completion 偏差 0/6/10/18ns；
  Round 2 = 关闭 q-吸收后 decode 流失序 + completion 偏差膨胀至
  32.6s/89.0s（=q 量级）→ q-吸收必要性证明，缺省 1000ns 冻结。
- B3（b3_canonical_compare.py，类型感知 canonical key）：节点总数
  exact 300,298=300,298；差异收敛三步登记（类型感知键/actionNNN 连续
  计数/turn-0 gate 短前缀命名）；终态差异 10,654 全部 comm_tag-only，
  归因 = TransferTagAllocator 分配序（离线全局因果序 vs 在线决策序），
  补偿不变量 (rank,src,dst,dir,tag) 零碰撞实测。B3 通过（tag 归因豁免）。
- B4：归因完成（类别① + real-online 四源）；感知开/关决策序列
  3531 行逐字节一致（查询/审计不进判据的实测）。

### 阶段 3（GraphBatch 原子提交验收）
- GraphBatchCommitterTest ALL PASS（29 类非法批零副作用 + validate 纯
  函数）；正式路径 single_node_bridge_count=0（两模式）；
  graph_batch_count==delivery_count（3525/3531）；avg 89.03 节点/批。

### 阶段 4（感知 + 对账）
- sensing 全量运行 PASS（1177/1177）；sh30_ledger_reconcile.py：
  R0/R1/R2/R5/R6 全过，verdict 对平（R5 残差 = 自完成 barrier 尾部
  6 条，合同① 口径）。
- wscllm 版 ledger_reconcile.py 不适配本仓产物（kv 事件流/事实抽取
  面不同），sh_3.0 版对照器独立实现（sh30_ledger_reconcile.py，登记）。

### 阶段 5-6（窗口/并行/benchmark，benchmark_20.md）
- 窗口扫掠：128 vs 无界决策序列逐字节一致 → 128 冻结；
- 并行 reader：8 并发 8/8 PASS，RSS ~7.8MB/实例（有界）；fixture 改
  per-PID 路径（原固定 /tmp 路径并发互踩，fixture-only 修复，登记）；
- CMake 修复：WindowedReaderTest 缺 RUNTIME_OUTPUT_DIRECTORY 块（蓝本
  同款缺口），补 bin/ 输出；
- 基准矩阵 + 决策门五条件：1/2/4/5 PASS，条件 3 PARTIAL 登记
  （strategy Python 占比 74%，文件桥优化项）。

### 中间产物有界核验
- bridge request_*.json 按交付数有界（3525/3531，~50MB/run，run 目录
  rm -rf 清理）；results jsonl 2.9-86MB（sensing 查询日志大头，按
  交付×54 rank 有界）；窗口检查点 1 份/run；峰值 RSS 400.6MiB（离线）。

## 阶段 7 收尾：裸仓库还原（2026-08-16，face/wscllm 先例）

- **删除物化输入与归档**：traces/ 三件套 + materialize 脚本 +
  PROVENANCE、baseline/20_30s/ 全部删除（规则与 30s 实测记录保留在本
  实录与方案文档；重建命令见下）；generated/ 与 results/ 清空（保留
  runtime_config 再生目录，测试按需 materialize）。
- **配置中性化**：trace_config.csv :12 → request_queue_placeholder.csv
  （占位，不物化任何默认队列）；:13 context sidecar 槽位置空（与队列
  一并由调用方物化后指定）；fail-closed 文案改为指向方案文档。
- **测试合成 fixture 化**：test_face_scheduler.py 新增 SYNTHETIC_QUEUE/
  CONTEXT_ROWS（10 请求/8 session，sidecar_restore 双件）+
  load_checked_in_fixture_config()；六个直接调用 checked-in 配置的用例
  改经 fixture；417 用例改名
  test_checked_in_config_is_request_neutral_and_fails_closed_without_input
  （含占位路径断言 + SystemExit fail-closed 断言）。
- **双根 pytest 全绿**：workload 根 42 passed + sh_test_mesh 根 33 passed。
- **fail-closed 实测**：checked-in 配置直接运行 --print-shell-config
  exit=1（缺失队列信息打印，无 stub 落盘）。
- **删除 wscllm 产物绑定的对照器**：tier_b_compare.py /
  ledger_reconcile.py（其输入面为 wscllm 产物；本仓对照由
  b3_canonical_compare.py + sh30_ledger_reconcile.py 承担）。
- **剩余差异如实登记**：
  1. 决策门条件 3 PARTIAL（strategy Python 占比 74%——文件桥 v0 的
     优化项，接口已按方案 §4.1 第 2 条保留）；
  2. B3 的 comm_tag-only 差异（分配序归因，补偿不变量零碰撞）；
  3. B1 的准入排队序差（两轮决定性证据 + q-吸收必要）；
  4. wait_for_work 在 input-closed + 图空闲时忙转（本仓
     strategy 调试中发现；修复根因后不再触发，机制层代码与蓝本一致，
     登记为已知坑——若未来再现，谓词应排除"仅 !input_open"分支）；
  5. sensing 查询日志体量（86MB/run，按交付×54 rank 有界；分片/压缩
     为后续优化项）。
- **重建物化输入**（调用方）：按方案 §3 步骤 0-1 从
  agent-traces/tracelab/astra_compute_20.csv 前 30 秒派生三件套（规则
  ①-④；实测 1177/112；AVERAGE_DECODE_LENGTH=459.4944774851317），在
  trace_config.csv :12/:13 指定后即可复现全部离线/在线链路。
- **git**：按阶段补齐 commit + tag（sh3-phase0..7-done + sh3-final-done），
  一次性入仓；face_scheduler.py 全程零 diff（红线）。

## git 收尾（2026-08-16）

- 按阶段补齐 commit + tag：ff6c9fd(阶段0) → 01936eb(阶段1) →
  79a3d36(阶段2) → c202849(阶段3-4) → fab0dee(阶段4) → 056404c(阶段5)
  → ccca37f(阶段6) → 0767dc1(阶段7) → 952a5ea(收尾)；
  tags：sh3-phase0..7-done + sh3-final-done。
- 收尾态复核：工作树对本仓干净；双根 pytest 42+33 全绿；fail-closed
  实测 exit=1；face_scheduler.py 对起点零 diff（红线全程）。

## 回灌修复轮（2026-08-16，四档 3 分钟对比测试 §5 缺陷回灌，sh_2.0 同款）

依据：`sh_2.0测试/对比报告.md` §5.1/§5.2/§5.3 + 测试副本
TEST_ADAPTATIONS.md（sh_2.0 系补丁在 3 分钟副本验证后回灌；本仓为
同构机制层的对应处置）。

### 回灌项 A：OnlineCli 默认窗口 fail-closed 化（§5.1，同 sh_2.0 三件套）
- **源缺陷机理（本仓同构）**：默认 `--request-max-arrival-ns=30e9`
  使超窗 turn-0 行被窗口阅读器拒绝后永不消费 → 窗口堵塞于
  high_water → EOF 不可达 → expected_requests 恒 0 → 全部
  expected-scoped 审计静默跳过（丢 12.5% 仍 PASS 的根因）。
- 修复：默认无界（0）；WindowedTraceReader 拒绝行拒绝即消费（EOF
  可达，完成度审计分母=data_rows=物化总行数）；main_online run-end
  新增 input window audit（rejected!=0 或 !csv_eof → 打印丢弃明细
  计数 → 非零退出）。本仓 main_online 与 sh_2.0 版仅差 progress
  probe 与 livelock guard 块（本仓无 guard：strategy 死锁根因已在
  阶段 1 修复，实录遗留 4 登记在案），审计块落在两仓同位。
- 测试：cli_online_test 默认断言 0；windowed_trace_reader_test 新增
  Part G（与 sh_2.0 同款合成 fixture，本仓 per-PID 路径版）。
  **集成实测**：合成队列（40/50/60s 到达）+ 显式 30e9 窗口 →
  cpp **exit=1** + rejected=3 明细审计行；默认无界的正式 replay/
  strategy run rejected=0 PASS。

### 回灌项 B：turn-0 在飞逐出 KeyError——**排查过，不适用**
- 排查结论：本仓 `graph_batch_builder.py` **不同构得病**。
  `pending_request_by_session` 仅在 `_emit_completion` 有 following
  request 时与 `pending_history[following]` gate **成对登记**（:884-890），
  到达时 gate 与 session 键**成对弹出**（:500-507）；turn-0 在 prefill
  发射**不登记** session 键（sh_2.0 的 :778 无 gate 登记分支本仓不
  存在）→ `_mark_pending_history_store`（:421-434）的
  `pending_history[pending_request_id]` 下标不可能 KeyError（会话键在
  则 gate 必在）。在飞 turn-0 被逐出走 `pending_request_id is None` 的
  deferred_session_locations 既有分支（语义同 sh_2.0 补丁目标态）。
  不移植补丁（避免无病灶处的防御性改动）。

### 回灌项 C：runner REQ_COUNT ARG_MAX（§5.3）
- `run_online_replay.sh` / `run_online_strategy.sh` REQ_COUNT 诊断行
  `ls` 通配 → `find -maxdepth 1`（同 sh_2.0）。

### 附加回灌（复验必需，测试副本 TEST_ADAPTATIONS 第 5a 条同款）
- 两 runner ET 目录硬编码（`..._qf4e4583d_c734f7cb4`）改 glob 动态
  解析唯一 generated 目录（复验轮 config digest 目录名变化：
  c734f7cb4→cf7022398，硬编码必断）。runner 层，非策略红线。

### 复核 D：teardown `[critical] unreleased nodes` 定性（§4.3 遗留）
- **结论：退出期在飞尾节点抱怨日志，非真实泄漏，不修**（与 sh_2.0
  同判）。本轮实测：replay 1 对 sys（同持 node 21723 的 send/recv 对）；
  strategy 6 sys（5 rank 同持 26947 + 1 rank 27947）——每 sys 恰 1 个
  GPU comm 槽、run 尾部 id、相邻 TP 对/多 rank 共享同一在飞 transfer。
  svc.finished() 为请求级终局权威（设计语义），尾节点完成事件仍在
  队列未处理即退出 → HardwareResource 析构审计抱怨。全部完成度/watch/
  bridge 审计通过；3 分钟档最大 11,121 请求 100% 完成（运行中槽泄漏
  会 livelock，矛盾）；静态路径无此输出；进程退出 OS 回收。不修：
  改退出时机会扰动冻结确定性基线。

### 复验（20.csv 前 30s，全序通过）
1. **物化器重建**：阶段 7 删除的 `traces/materialize_20_30s.py` +
   PROVENANCE.md 按冻结 md5 反推重建（queue description 文案
   `compute_20 first-30-seconds window; turn-0 prefix kept as
   remote-resident history KV (see context sidecar)`——无 sh_2.0 的
   "(sidecar_restore)" 后缀，为冻结 md5 唯一命中）。重跑产物与阶段 0
   冻结值**逐字节一致**：queue md5 `8b615b5a12b15143a10bbbf13f31a6bb`、
   context md5 `25b7da357cac249c14bf4a3ae26ddfd4`；112/1177、
   turn-0 prefix>0 行 79、max input_tokens_total=169,395、
   average_decode_length=459.4944774851317 全对上登记。traces/ 的
   重建入口按 sh_2.0 先例**保留入库**（物化 CSV 仍不入库）。
2. **编译全目标 + C++ 单测**：全目标过；WindowedReaderTest A-G ALL
   PASS（bin/ 新版二进制；AstraSim_Analytical/ 下旧位置二进制为
   RUNTIME_OUTPUT_DIRECTORY 修复前的陈旧残留，勿用）；CliOnlineTest
   R1-R11 PASS。
3. **双根 pytest**：物化态 42+33 全绿；裸仓态复验 42+33 全绿 +
   fail-closed exit=1（见下：417 用例改状态感知）。
4. **runall 静态**：exit 0；[METRIC] 2218 行（登记确定性口径）；
   sim_end_ns=**1,714,696,296,631（实录登记值 exact）**、
   incomplete=0、memory_actions 64914/0（登记 exact）。
5. **--replay-record**：decision_log 293,801 行、md5
   **7452f0fa89f83f120fa90d985bfdd60d（实录冻结值 exact）**。
6. **replay 全量 PASS**：delivery=ack=3525、completed=1177、stale=0、
   no_decision=0、single_node=0、total_nodes=300,298（登记 exact）、
   rejected=0、csv_eof=true、sim_end_ns=**995,087,425,606（实录登记
   exact）**；online_decision_log 3531 行；两轮 replay 决策日志逐字节
   一致（确定性）。
7. **strategy 全量 PASS**：delivery=ack=3531、1177/1177、stale=0、
   avg 89.03 节点/批（登记 exact）、late=33、
   sim_end_ns=1,324,294,856,383（真实物理）。
8. **Tier B 不回退**：B3（b3_canonical_compare，SH30_B3_DUMP=1 重跑）
   节点总数 exact 300,298==300,298；canonical 差异 **10,654/10,654
   （与登记终态同数）经键剥 tag 复核 = 0 残差**——全部 comm_tag-only
   （name/rank/type/bytes/src/dst 逐项相同，TransferTagAllocator 分配
   序既有归因），登记结论不变（对照器脚本 exit=1 + multiset equal
   False 为登记终态的固有输出，判定按归因豁免口径）。B1/B2 权威
   对照（record 键逐行）随 replay 决策日志 3531 行与冻结
   decision_log（md5 exact）由 6 复验承载；B4 归因类别①不变
   （sim_end exact 对上）。ledger 对账需 sensing run 产物
   （ledger.jsonl），本轮未跑 sensing，登记为未复跑项（阶段 4 已验收，
   本轮改动不触及账本路径）。
9. **合成 fixture**：见回灌项 A（exit=1 + 明细）。
10. **teardown 复核**：见复核 D。

### 测试改动登记（测试层，非策略红线）
- `test_face_scheduler.py` 417 用例（checked-in 配置双态断言）改
  **状态感知**：traces/ 物化输入在位时断言 :12 指向物化 queue 且
  正式入口解析 1177/112；不在位时保持原占位断言 + fail-closed
  SystemExit 断言（与 sh_2.0 skipUnless fixture 化同款语义，两态全绿；
  原版在物化态必失败）。

### git
- commit（pathspec 仅本仓）+ tag `sh3-backport-fix1-done`。

## 补验轮：回灌后 sensing 复跑 + ledger 对账（2026-08-16）

目的：backport-fix1（OnlineCli 默认无界+审计分母、runner find/glob；
本仓 turn-0 补丁排查为不适用）落地后，补齐回灌轮登记的"未复跑项"
（ledger 对账需 sensing run 产物），复跑阶段 4 的 sensing 运行与对账
确认零回退。

### 输入与 ET
- 物化重跑（traces/materialize_20_30s.py）：queue
  `8b615b5a12b15143a10bbbf13f31a6bb` / context
  `25b7da357cac249c14bf4a3ae26ddfd4`——冻结值逐字节一致（物化器自带
  frozen 比对 PASS）；112/1177、turn-0 prefix>0 行 79、max
  input_tokens_total=169,395、average_decode_length=459.4944774851317。
- trace_config :12/:13 指向物化 queue + context sidecar
  （sidecar_restore 语义）→ config digest c67924a9b（目录名随 :12 文案
  变化，runner glob 解析不受影响）；generate_trace 产出 54 .et +
  manifest（total_nodes=300,298，登记 exact）。

### sensing 全量（run_online_strategy_sensing.sh，/tmp/sh30_recheck/sense1）PASS
- completed=1177/1177（incomplete=0）、no_decision=0、
  single_node_bridge=0、watch stale=0、graph_batch==delivery=3531
  （本轮日志无独立 ack 计数行；ack 等价性由 graph_batch==delivery 与
  phase-4/5 审计承担）、avg 89.03 节点/批（登记 exact）、
  **sim_end_ns=1,324,294,856,383
  （实录登记 exact）**、late=33（exact）、rejected_out_of_range=0、
  csv_eof=true、tick_end_without_decision=5,866,550（登记值 exact）；
  ledger.jsonl(1177)/sensing_query_log.jsonl(3531)/online_decision_log
  (3531)/digests(3531) 产出；teardown `[critical] unreleased nodes`
  6 行（复核 D 定性：退出期尾节点抱怨日志）。

### 感知开/关对照（同输入同 ET）
- 关感知 strategy 全量 PASS（/tmp/sh30_recheck/strat1）：全部审计计数
  与 sensing 运行一致（sim_end_ns 同为 1,324,294,856,383）；
  **online_decision_log 3531 行与 sensing 运行逐字节一致（cmp=0）**——
  与阶段 2 B4 登记结论（感知开/关决策序列逐字节一致）同款复现，
  回灌修复轮改动不影响该不变量。

### ledger 对账（online/verify/sh30_ledger_reconcile.py）
- R0（1177×3 决策行）/R1（序 + decode==prefill 实例）/R2（issued 核销 +
  completed 覆盖 1177）/R5 全 PASS，**verdict: 对平 (balanced)**——与
  阶段 4 登记逐项一致；R5 残差 = 自完成 barrier 尾部 **6 条**（登记值
  exact，completing=['session_60_request_47']，合同① 口径）；R6 观测项
  peak injected node_count=1。

### runner 补齐（runner 层，非策略红线）
- run_online_strategy_sensing.sh 回灌轮漏打的两处同款修复：ET 目录
  硬编码（qf4e4583d_c734f7cb4，config digest 变化必断）→ glob 动态
  解析唯一 generated 目录；REQ_COUNT `ls` 通配 → `find -maxdepth 1`
  （ARG_MAX）。与 replay/strategy runner 的回灌项 C/附加回灌 5a 同款。

### 复验与收尾（裸仓态）
- 物化态双根 pytest 42+33 全绿；清理（删物化 CSV 与 generated ET、
  trace_config 还原占位）后裸仓态 42+33 全绿（417 用例状态感知）+
  fail-closed 实测 exit=1；traces/ 保留 PROVENANCE.md + 物化器。
- 红线自检：face_scheduler.py / generate_face_trace.py 零 diff；改动面 =
  sensing runner 两处 + 本实录。
- turn-0 逐出 KeyError：本仓排查结论不变（不同构得病，不移植补丁，
  见回灌项 B）；builder 记账路径本轮 sensing 全量运行零异常。

### git
- commit（pathspec 仅本仓）+ tag `sh3-sensing-recheck-done`。

## diff_explainability 缺陷 3 排查修复（kv_actions/assignments 真空通过，2026-08-16）

wscllm/face 开感知验证追记立档的缺陷 3（`online/verify/diff_explainability.py`
kv_actions/assignments 真空通过假绿）：`_load_run()` 从 response_*.json
收集 kv_actions/assignments，phase-7 §10.3 起 response 消费即删（本仓
DecisionBridge.cc 同款 unlink），两侧恒读成空列表，空==空打出 PASS，
签名 sha256=4f53cda1… 即空数组 [] 的哈希——比较实际没有发生。

### 排查结论

- 本仓 `diff_explainability.py` 与 wscllm/face/sh_2.0 同名文件逐字节
  一致（md5 08db8c12，427 行同段陈旧代码）：**缺陷成立**（本仓尚无
  直接调用该脚本的既有验证流程，缺陷为潜伏态，修复消除）。
- 实证复现（缺陷版 + face测试真实开/关产物对）：face_20_3mins 20 档
  3min 对（2091 请求，bridge response 计数实测 0）打出 `PASS
  kv_actions 一致 sha256=4f53cda18c2baa0c / PASS assignments 一致
  (同签名)`——假绿成立。

### 处置（verify 工具层，仿真/调度本体零改动）

与 sh_2.0/wscllm 同版修复（md5 5f72c396，逐字节一致）：

1. kv_actions 维度迁 online_decision_log.jsonl 的 KV 决策载荷流
   （{kind, request_id, decision} 投影；本仓载荷字段 kv_instance_
   after_completion/completion_evictions 族——按载荷整体做稳定摘要，
   不耦合字段名，跨仓同版）。
2. assignments 维度迁 graph_batch_digests.jsonl 批级摘要流
   （delivery_sequence/reasons/ranks/node_count/edge_count/
   watch_count/content_sha256）。
3. 空数据源 fail-closed：jsonl 缺失（bridge/ 优先、results/ 次选，
   两布局均支持）或 0 行、决策行缺 decision 载荷、无 request_*.json、
   cpp.log 缺失或零计数器——一律 exit 2 不产报告。

### 复验

- 修复版在 face测试 20/50/80 档 3min 开/关对（6273/13941/22269 决策行）
  全 PASS，KV 载荷与 assignment 摘要均为真实非空哈希（41deb73f/
  1c67a6f6 等），4f53cda1 签名消失；mutant 篡改实测 DIFF 可检出；
  fail-closed 5 例实测 exit 2。
- 本仓 jsonl 行格式与 sh_2.0 同源（online_scheduler_base._digest_row
   /log_decision 同构），30s 级产物（sh20_recheck 对）同版脚本复跑
  PASS（119f7c84/03fdf7e7）。

### git
- commit（pathspec 仅本仓）+ tag `sh3-diffexp-fix-done`。

## 缺陷修复 2 同步轮（2026-08-16/17，来源：face 0049ef5 / tag face-defectfix2-done）

> face 四档主动测试暴露的三类在线机制层竞态缺陷（A=服务终判
> finished() 计数真空、B=文件桥 FIFO EOF 误检、C=空队列唤醒丢失）
> 已在 face 修复入库；本节为按其"移植性登记"同步到本仓的实录。
> 策略红线零触碰：三段式准入/decode 同实例/三态 KV/sidecar_restore
> 相关策略文件零改动（git diff 复核）。B17 族 unreleased nodes
> 观测维持既有登记不动（本轮不触及 Workload 侧）。

### A. 缺陷 A：服务终判 finished() 计数真空（main_online.cc）

- pump() 后、finished() 判定前补一次 drain（pending_command_count()>0
  → drain_commands()）；finished() 为真而 CSV 未 EOF → online_fatal
  （携带 active/pending_alarm/queued_commands 计数）。
- 回归 fixture：`tests/service_vacuum_test.cc`（新 CMake 目标
  ServiceVacuumTest）。**本仓适配**：WindowedTraceReader 为分叉版
  （回灌 fix1 走 main_online 侧 run-end 审计，无 wscllm/sh_1.0 的
  audit_completion API），fixture 的审计断言改为本仓语义——
  completed == data_rows（main_online 的 EXIT_FAILURE 判定）：
  Part1 legacy 复现真空（completed=10、断点队列 10 条未 drain、
  reader 未 EOF、audit FAIL）；Part2 fixed 30/30 + EOF + audit Ok。

### B. 缺陷 B：文件桥 EOF 误检（DecisionBridge.cc/.hh + decision_bridge.py）

- C++ 与 Python 同版本成对移植：resp_notify 双侧常开长连接 fd、
  wait_response_byte（read==0 真死义 + 1:1 在途守卫 + EAGAIN 重
  poll）、Python BridgePipeError fail-closed + stderr 留痕、_fail()
  留痕先行——与 face/sh_1.0 逐行同源（仅文件头署名行保留本仓口径）。
- fixtures：loopback Part D/E（A-E ALL PASS）、bridge_cpp_death_
  fixture.py 3/3（exit=1 + "C++ side is gone"）、bridge_race_
  stress_repro.sh（本机 30s 预算第 273 次迭代复现假 EOF，写端存活）。

### C. 缺陷 C：空队列分支唤醒条件（main_online.cc + EventQueue map 版适配）

- 空队列分支三分支化（mailbox 残留→T+1 不变；has_deferred_work→
  T+1 强制边界；input_closed+双空+未完→lost-wakeup dead end
  fail-closed；input 开→wait_for_work，IDLE 合同不变）。
- **EventQueue map 版适配**（移植性登记要求参考本仓阶段 1 先例）：
  `has_deferred_work()` 访问器直接适配 std::map<EventTime,
  EventList> 实现——deferred_queue_（std::list<EventList>）与主队列
  map 容器无关（阶段 1 步骤 1-1 既有结构），函数体与 list 版逐字
  相同（`return !deferred_queue_.empty();`），.h 注释按 map 版措辞
  （主 map 为空而 deferred 仍挂起的盲区语义）。纯新增 const 访问
  器，静态二进制零行为变化；event_queue_deferred_test（map 版
  A-E 含 E2 death test）复跑 ALL PASS。
- wakeup_guard 双场景复验：f7-blueprint PASS（不误触发）；
  defer-dead-end PASS（fail-closed "lost-wakeup dead end"+active=1）。
  ONLINE_PROGRESS_PROBE 调试探针原样保留。

### 本轮同步的既有失配修复（fixture 层，登记）

- `run_online_idle_fixture.sh` / `run_online_same_tick_milestone.sh`：
  曾硬编码旧 label 后缀目录名（qf4e4583d_c734f7cb4；trace_config
  :12/:13 物化路径变化即目录名漂移）→ 改为官方 runner 同款
  generated 唯一目录 glob 解析。复验五态迁移 + milestone (a)-(d)
  ALL PASS。
- `run_online_wakeup_guard_fixture.sh`（新入库）：同款 glob 解析。

### 复验证据（20.csv 前 30s 包络；仿真门控记录见下）

1. 构建+单测：build 全目标 0 error（含新 ServiceVacuumTest）；
   WatchRegistry/DecisionMailbox/GraphBatchCommitter/WindowedReader/
   ServiceVacuum/BridgeLoopback A-E/event_queue_deferred（map 版）
   全过；bridge_cpp_death 3/3。
   **NodeStoreTest part C 预存在失配（如实登记，与本轮无关）**：
   本仓 make_completion_fixture_et.py 为 5-node 版（阶段 1 扩展
   MEM 节点，port rank 5 / 非 port 4），而 node_store_test.cc part C
   期望 3 dependency-free 节点（wscllm 蓝本 3-node 版期望未随生成
   器升级）——两文件均自 01936eb 未再改动，HEAD 状态即 part C
   FAIL（parts A/B/D 过）。非本轮引入、不属缺陷修复范围，登记
   待后续修复轮处理；CLI/OnlineCli 未改动（cli_online R1-R11 合同
   不受影响，登记跳过）。
2. 物化：materialize_20_30s.py 自带 frozen md5 比对全 exact
   （queue 8b615b5a / context 25b7da35 / digest 476a02d0；1177/112）。
3. 决策日志重录：293,801 行、md5 **7452f0fa89f83f120fa90d985bfdd60d**
   （实录冻结值 exact）。
4. 静态 runall：exit 0，completed_unique_requests=1177、
   incomplete=0、memory_actions_unresolved=0、sim_end_ns=
   **1,714,696,296,631（实录登记值 exact）**、raw_metrics 164 行。
   静态二进制不含 execution_driven TU、EventQueue 交集为纯新增
   访问器（零静态调用方）——静态零扰动论证与 face/sh_1.0 同款。
5. replay：PASS——1177/1177、delivery_count=3525（登记 exact）、
   no_decision=0、single_node_bridge_count=0、total_nodes=300,298
   （登记 exact）、sim_end_ns=**995,087,425,606（登记 exact）**。
6. strategy：PASS——1177/1177、delivery_count=3531、no_decision=0、
   single_node=0、graph_batch_count=3531（==delivery，缺陷 A 不变
   量）、avg_nodes_per_batch=**89.03（登记 exact）**、
   tick_end_without_decision_count=**5,866,550（回灌实录登记
   exact）**、sim_end_ns=**1,324,294,856,383（登记 exact）**。
7. **Tier B 不回退（回灌轮口径复现）**：B3（SH30_B3_DUMP=1 replay
   重跑 + b3_canonical_compare.py）：节点总数 300,298==300,298
   exact；canonical 差异 **10,654/10,654（与登记终态同数）**，按
   既有归因豁免口径判定（comm_tag-only，TransferTagAllocator 分配
   序既有归因；对照器 exit 码非判定门）。
8. **sensing + ledger 对账（本轮补齐回灌轮的未复跑项）**：
   strategy_sensing PASS（3531 retained）+ 感知开/关决策日志与
   strategy 运行**逐字节一致**（cmp=0，阶段 2 B4 结论复现）；
   sh30_ledger_reconcile R0/R1（两断言）/R2/R5 全 PASS、verdict
   对平——R5 自完成尾部 **6 条 exact（completing=
   session_60_request_47，与 90d9a4c 登记完全一致）**。
9. wakeup_guard 双场景 + IDLE 五态 + same-tick milestone (a)-(d)
   ALL PASS。

### 仿真发射门控记录（如实）

- 另一 agent 的 sh_2.0 120 档 sensing 重仿真占用在线仿真槽
  （02:14-05:02），本轮仿真发射排队至其结束后；发射期内存
  used 34-45%（<80% 红线），与并行 agent 的短 replay 进程共存
  （face 复测同款并行共存口径）。
- /tmp（40G tmpfs）中途写满：首次 sensing 运行 python_exit=1
  （产物写失败，无仿真语义问题），清理本会话之外的旧运行目录后
  重跑 PASS——受污染运行已删除，未进入任何判定。

### 裸仓态恢复

- 删物化三件（traces/*.csv）与 generated 产物与 results/；
  trace_config.csv :12/:13 回占位（git checkout 原字节）。
- fail-closed 实测：generate_trace.py 无输入 exit=1。
- 裸仓态双根 pytest：tests 33 passed + workload 42 passed（417 用例
  状态感知设计：物化输入不在位时占位断言 + fail-closed 断言通过）。

### git（本轮）

- commit（pathspec 仅本仓）+ tag `sh3-defectfix2-done`：机制层五文件
  （main_online.cc / DecisionBridge.cc/.hh / EventQueue.h/.cpp map 版
  适配 / decision_bridge.py）+ fixtures 四件 + 脚本两件 + CMake
  ServiceVacuumTest + 两 fixture runner glob 化 + 本实录同步段。

---

## 路径②清理实录（2026-08-18，步骤 1/2：删 replay 路线）

依据：《路径功能代码对应说明.md》。

**删除（②专有）**：`run_online_replay.sh`；`online/replay_source.py`、`online/sh30_replay_scheduler.py`；online_service.py 的 replay 分发与 consumed 校验；generate_face_trace.py 的 `--replay-record`（词法保留显式拒绝）、`build_face_plan(replay_record)` 形参、`_write_replay_decision_log`/`_replay_kv_transfer_dict`；graph_batch_builder.py 的 replay_clock/清链与 LUT 校准；C++ replay_clock 全套（Workload.cc :249 dep-free、:375 remote MEM 1ns、:399 HBM restore 1ns、:502 校准 COMP 不入 HBM 模型、:545 comm 1ns）。

**共享符号迁移（登记）**：`build_plan_dict`（manifest 记录 → plan dict 转换器，含 _LocationShim/_shard_from_dict/_transfer_from_dict/_transfers）原位于 sh30_replay_scheduler.py 但被 sh30_online_scheduler import——已整体迁入 sh30_online_scheduler.py（函数体零改动，仅迁移位置）。

**保留（红线）**：face_scheduler/generate_face_trace 被 import 符号；sh30_online_scheduler（三段式准入/三态 KV）；tier_b_compare（无 replay 依赖）；idempotency_fixture（kwarg 适配）。

**验证（2026-08-18）**：重编 PASS（9 binaries）；pytest 75 全绿；③④冒烟（30s 全量 1177）双 PASS、no_decision=0、③④决策日志逐字节一致；fail-closed 实测。

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
- generator ①专有符号（AST 全仓引用面分析驱动，被 import 符号全存活）：write_face_trace/build_face_plan/load_or_build_face_plan/print_shell_config/resolve_output_dir/build_trace_label/ParallelTraceOutputs/StreamingTraceOutputs/_planner_cache_path/_replay_trace_worker 及 _*_dict 快照族（22 符号；_write_replay_* 已于步骤1删）；`main()` 改为 fail-closed 拒绝桩（"path-1 removed; use plan_materializer.py"）；
- `generate_trace.py` 的 `main()` 委派段改同款拒绝桩（模块本体整文件保留——Chakra 常量/TraceBuilder/transformer_pass(_aggregated) 等为③④与 microbench 共享符号库）。

**GEN_MATCH 改造**：（本仓 runner 原生 GEN_MATCH，无改造）

**测试处置**：pytest 40+33 通过；删 1 个①耦合用例 test_shell_config_does_not_build_full_face_plan（print_shell_config 已删）；余 1 失败为锚点预存缺陷（request-neutral 断言期望物化队列名，锚点态同败，登记）

**验证（2026-08-18，①删除后仅靠 plan_materializer 输入）**：
- 替代产出等价性：合成 manifest vs ①真实 manifest 逐字段 **0 mismatch**；
- 重编 PASS（Online+fixtures，静态目标已不存在）；幸存 pytest 全绿（见各仓数字）；
- ③冒烟 PASS：10s 270 request 交付 810、no_decision=0；④冒烟 PASS 且 online_decision_log 与③**逐字节一致**；
- 分层账本对账：**对平 (balanced)**（10s/270，sh30_ledger_reconcile --run-dir --manifest=合成 manifest——其 manifest 仅队列派生字段，决策事实不外取，裁决 i 条件 c 结构性满足）；
- fail-closed 实测：generate_trace.py 拒绝桩 exit=1；plan_materializer 空队列 exit=1；runner 缺 request_csv exit=1；GEN_MATCH 零目录/双目录均 exit=1。

**本步工作树改动文件清单（供审阅提交）**：新增 plan_materializer.py；删 runall.sh/run_sh_test_aware.sh/generate_trace.sh/run_online_replay.sh、congestion_aware/main.cc、online/replay_source.py、online/sh30_replay_scheduler.py；
改 CMakeLists.txt、main_online.cc、Sys.cc/hh、Workload.cc/hh、OnlineCli.cc/hh、cli_online_test.cc、generate_face_trace.py、generate_trace.py、online/{online_service,graph_batch_builder,sh30_online_scheduler}.py（build_plan_dict 族自已删 replay 调度器迁入）、online/verify/idempotency_fixture.py、test_face_scheduler.py、run_online_strategy.sh、README_COMMANDS.md、本实录。

---

## 路径①②终清与重验实录（2026-08-18，残余清扫轮）

**扫描口径**：同 face 仓。

**本仓改动（终清轮）**：
1. 【功能残留-删】online/online_service.py 的 `_CanonicalSink` 类 + `SH30_B3_DUMP` 环境门 + canonical_nodes.jsonl 实例化与挂接块 + online_scheduler_base.py 的 `batch_sink` 桥接钩子（getattr 守护 4 行）：唯一消费方 b3_canonical_compare.py 已在前轮补清删除，全仓零引用；经重建+③④冒烟+逐字节一致复核零影响；
2. 【活文件陈旧注释-改写】generate_face_trace.py / generate_trace.py 拒绝桩 docstring；online/graph_batch_builder.py（docstring 的 write_face_trace 删除符号引用、B3 canonical key、②清链/LUT 时钟注释、「replay 自 decision_log」口径、「replay 权威=manifest/planner」句）；online/online_service.py（模块 docstring ②删除说明、根因#5 replay 括注、legacy 终值注释中的 tier_b_compare 引用、B3 dump hook 注释随钩子删除）；online/sh30_online_scheduler.py build_plan_dict 族迁移来源注释；metrics_integration.py；run_metrics_postprocess.sh 头注释；run_online_wakeup_guard_fixture.sh 报错指引；plan_materializer.py docstring；Workload.cc；execution_driven/tests/ 六文件构建注释；
3. 【活文档陈旧内容-改写】sh_test_mesh/README.md（同型改写，物化器=traces/materialize_20_30s.py）；README_COMMANDS.md 整表重写（原表含「--layers B0..B4」孤儿行与②清理实录尾注、已删 ledger_reconcile.py 引用——改为 sh30_ledger_reconcile.py --run-dir --manifest 实际口径）；
4. 【边界-保留并登记】同 face（ETFeeder 共享边界、issue_replay 机制语义、metrics microbenchmark 库级支持、历史记录载体）。

**本仓重验数字**：③④冒烟 10s/270 双 PASS：completed=270、no_decision=0、single_node=0、delivery=810==digests 810、③④决策日志 cmp 零差异；④对账 sh30_ledger_reconcile.py --run-dir --manifest = **对平 (balanced)**；pytest 双根 40+33（另 1 失败=锚点预存缺陷，与基线同败，零新增）。

**重验（2026-08-18 终清后全量）**：
- clean 重建（rm -rf build_congestion_aware 后 cmake 重配 + 全目标）：exit=0，10 个 add_executable 目标（Unaware/_Online/8 fixtures）全部产出，0 error；
- fail-closed 复测：generate_trace.py 拒绝桩 / 生成器 main 拒绝桩 / plan_materializer 空队列 / runner 缺 request_csv / GEN_MATCH 零目录 / GEN_MATCH 双目录 —— 全部 exit=1；
- pytest 双根：workload 根 + sh_test_mesh/tests 根，与终清前基线逐项一致，零新增失败；
- ③④ 冒烟：run_online_strategy.sh 与 run_online_strategy_sensing.sh 各一场，completed==物化数、no_decision_python_callback_count=0、single_node_bridge_count=0、delivery==graph_batch_digests 行数、③④ online_decision_log.jsonl 逐字节一致（cmp）、④ 分层账本对账 verdict=对平/BALANCED；发射前内存门控实测 ~11-12%（<70%）。
