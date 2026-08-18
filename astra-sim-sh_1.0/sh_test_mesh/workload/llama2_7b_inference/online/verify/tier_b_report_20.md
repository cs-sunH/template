# sh_1.0 Tier B 等价验收报告(20.csv 前30s,1177 请求 / 112 session)

> 阶段 2 交付物;比较器 = `online/verify/tier_b_compare.py`(exit 0 全过)。
> 材料:replay/strategy 官方 runner 全量运行(各 1177/1177 完成,双进程
> exit 0)+ 阶段 0 归档基线 `sh_test_mesh/baseline/20_30s`
> (decision_log md5 `se2e0e72de30eb0b13993037cac90d7629`;输入队列 md5
> `ee7af9d0bc3e9c2e211d1e30c325bd45`,源 csv md5 `fc74a48e...`,
> 1177/112,prefill 66-169395,decode 1-13812)。

运行命令(相对仓库根):
```bash
bash sh_test_mesh/run_scripts/run_online_replay.sh <run_dir> \
  sh_test_mesh/workload/llama2_7b_inference/traces/astra_compute_20_first_30_seconds_request_queue_recompute.csv \
  sh_test_mesh/baseline/20_30s/decision_log.jsonl
bash sh_test_mesh/run_scripts/run_online_strategy.sh <run_dir> \
  sh_test_mesh/workload/llama2_7b_inference/traces/astra_compute_20_first_30_seconds_request_queue_recompute.csv
python3 sh_test_mesh/workload/llama2_7b_inference/online/verify/tier_b_compare.py \
  --replay-run <replay_dir> --strategy-run <strategy_dir> \
  --baseline sh_test_mesh/baseline/20_30s \
  --config sh_test_mesh/workload/llama2_7b_inference/trace_config.csv
```

## 结论总表

| 层 | 判定 | 关键数值 |
|---|---|---|
| B0 输入等价 | PASS(容差 0) | 1177/112;离线决策行 3531+194328 iteration;在线 replay/strategy 各 3531;trace_digest 与 request_mapping_digest 在线==基线;prefill 决策 digest 0 mismatch |
| B1 生命周期/决策(oracle) | PASS | prefill exact 1157 +20 clamp;decode [-4,10]ns;completion [-8,9]ns;跨 tick 逆序 0;阶段序违规 0;提前决策 0;q-吸收事件 0 |
| B2 策略等价(oracle) | PASS(容差 0) | 三 kind 决策内容 0 mismatch;KV 载荷 2510 行 0 mismatch(kv_event_payload_sh1 口径)|
| B2 real-online(strategy) | PASS | 确定性重放 3531 交付逐行全等;assignment_key 自洽 0 fail |
| B3 图结构等价(replay) | PASS(容差 0) | 节点 canonical 多重集一致(offline 329308 == online 329308);digest 对平 live run 0 mismatch;边差异 within/cross = 243/17027(missing)+0/0(extra),全部归入登记类别①②③④ |
| B4 执行等价(归因口径) | PASS | 边界级/节点级漂移全部纳秒级;completion 跨 tick 逆序 0;completion 早于 decode 0;completed_requests/memory_actions_total 基线==在线(1177/95790) |

全部请求完成(1177/1177);unresolved/dropped/deadlock/starvation = 0;
watch/fence 无重复 fire(重复完成/到达即 fail-closed,基类
`_settle_completions`/`_process_arrivals`);结束审计无 stale 泄漏
(`verify_run_end`:in_flight 空、ack==delivery、replay 全消费)。

## B0 输入等价

- 请求/session:1177/112(manifest selected_*;与物化 PROVENANCE 一致)。
- 决策日志:离线 197859 行(3531 决策 + 194328 iteration;iteration 为 LUT 计时计划,replay 不消费);在线 replay 3531 行、strategy 3531 行(各 3x1177)。
- 输入血统:trace_digest `e4686943a6938c90...`、request_mapping_digest `20cbc93938ecd232...` 在线(replay/strategy)与基线 raw_metrics 同值。
- prefill 决策内容 digest:replay vs 离线 0 mismatch(容差 0)。

## B1 生命周期/决策等价(oracle/replay)

权威键 = (kind, request_id) -> 离线记录 tick(record-tick 权威键,sh_2.0 裁决 3/sh_3.0 两轮验证法同款);决策内容容差 0(B2 复核)。

| kind | exact | negative | positive | delta 范围(ns) |
|---|---|---|---|---|
| prefill | 1157 | 0 | 20 | [0, 10] |
| decode | 286 | 525 | 366 | [-4, 10] |
| completion | 207 | 638 | 332 | [-8, 9] |

登记口径(冻结;超出即未登记差异 -> 比较器 fail):

- prefill:1157 exact + 20 个正偏差全部为 alarm-clamp 边界(下一 turn 记录 tick <= 前序完成边界+1 的 session,alarm 钳制到 tick+1,`sh10_replay_scheduler.py` 显式登记的 sh_1.0 边界口径;实测 20 个全部满足该条件,偏差 [1,10]ns)。
- decode/completion 纳秒级双向偏差 = 类别①(COMP 链校准整数截断 + 1ns 事件粒度)+ 类别③(tick-end/deferred T+1)。
- 跨 tick 逆序 0(硬门);同 tick 组内逆序对 12(信息性,③)。
- 阶段序:tick 空间违规 0,交付 seq 空间违规 0;决策早于到达 0。
- 本仓特有核对点(准入排队):输入 hbm_wait_ns 全 0(离线无 HBM 阻塞准入)→ q-吸收事件 0;到达边界与记录 tick 的 20 处差异全部为 alarm-clamp 负向钳制([-10,-1]ns)。pending_admissions 事件驱动重试路径(§0.4 #17)由 B2 确定性重放逐 delta 复核(admit_waiting_requests 每 delta 尾部执行,重放逐行全等即覆盖)。

## B2 策略等价

**oracle(replay)**:三 kind 决策内容逐 request 全等(0 mismatch,容差 0)——包括 prefill 实例/assignment_key/历史迁移/逐出序列、decode 实例/candidates/迁移/逐出、completion 逐出与终态位置。

KV 事件载荷对照(kv_event_payload_sh1 口径,六类转移 × 1177 request,2510 行):replay 决策载荷 vs 离线 manifest `requests[].transfers_by_stage` 逐条全等(kind/phase/reason/session/trigger/source/target/total_bytes/shards 含 noc_path)——0 mismatch。

**real-online(strategy,关感知)**:

- 确定性重放不变量:把 strategy 运行保留的 3531 个交付(bridge/request_*.json)逐条喂给全新 `Sh10OnlineScheduler`(同一冻结 LUT 表、同一只读策略函数 KVCacheManager/select_prefill_instance/select_decode_instance),重放产出决策与 recorded 日志逐行全等(0 mismatch)——即"相同输入下同一函数输出一致"的逐决策断言。
- prefill_assignment_key 尾元素 == 所选实例(排序键 (remaining_chunks, last_arrival, config order) 自洽):0 fail。
- 信息性(差异可解释性证据,合法不同):strategy prefill 实例分布 {0: 122, 1: 120, 2: 133, 3: 119, 4: 138, 5: 137, 6: 145, 7: 139, 8: 124};decode 实例分布 {0: 342, 1: 120, 2: 167, 3: 139, 4: 92, 5: 94, 6: 97, 7: 55, 8: 71};与离线/replay 的差异来源 = 真实完成时序改变排队深度快照与 active decode 成员(队列深度均衡 + LUT 代价模型的输入变化),逐决策由确定性重放证明不变量。

## B3 图结构等价(replay)

在线侧材料 = 比较器内确定性重放交付流重构的 GraphBatch 节点/边;先与 live run 的 `graph_batch_digests.jsonl` 逐批对平(3530 批 content_sha256 0 mismatch = 重放构图 == live 构图)。

canonical logical node key(合同⑦;插入顺序/节点 ID 不作比较项):(rank, name, type, is_cpu_op, is_timer_op, 类型感知属性四元组——comp(num_ops,tensor_size,remote_weight_bytes)/comm(bytes,src,dst)/coll(comm_type,bytes,priority,pg_name,involved_dim)/mem(tensor_size)/timer 无载荷)。comm_tag 不入 key:tag 由 TransferTagAllocator 按全局发射序顺序分配(离线静态预写序 vs 在线决策交错序),与节点 ID 同类的插入顺序伪影;其配对语义由替代硬门证明(send/recv 集合内 tag 各自唯一+ 按 (src,dst,tag) 一一配对且 bytes 相等:实测 duplicate=0, pair_missing=0, recv_without_send=0)+ 在线运行执行配对成功(1177/1177 完成,tag 错配即网络死锁)。

- 节点多重集:offline 329308 == online 329308;missing 0,extra 0(容差 0,硬门)。
- 边:offline 335721,online 318451;missing 17270(within 243 / cross 17027),extra 0(within 0 / cross 0)。
- 边差异归因(登记类别,方案 §5.5 ①②③④):
  - cross-request 差异 = 类别①(replay LUT 时钟:段 1 清 prefill 组 previous_id + 段间 own-block-end 恢复;离线 writer 保留物理跨 request 链)+ 类别②③(块末恢复 emitted-ranks-only / 同 tick 发射序)。
  - within-request 差异 = 类别④(三段式发射的段间边界与触发门编码:段 2 noc_migrate 触发门锚段 1 末、段 3 completion_evictions 锚 decode 末——本仓 §5.5 预期高风险点,实测逐条归入)。
  - within missing 样例(rank/from/to/count):[[1, 18, 'q0117_session_9_turn3_session_9_request_3_prefill_decode_tra', 'q0117_session_9_turn3_session_9_request_3_completion_evictio'], [1, 18, 'q0109_session_8_turn6_session_8_request_6_prefill_decode_tra', 'q0109_session_8_turn6_session_8_request_6_completion_evictio'], [1, 18, 'q0102_session_7_turn7_session_7_request_7_prefill_decode_tra', 'q0102_session_7_turn7_session_7_request_7_completion_evictio'], [1, 18, 'q0356_session_28_turn12_session_28_request_12_history_transf', 'q0356_session_28_turn12_session_28_request_12_completion_evi'], [1, 18, 'q0731_session_60_turn1_session_60_request_1_prefill_decode_t', 'q0731_session_60_turn1_session_60_request_1_completion_evict'], [1, 18, 'q0204_session_14_turn15_session_14_request_15_history_transf', 'q0204_session_14_turn15_session_14_request_15_completion_evi']]
  - cross missing 样例:[[1, 0, 'q0049_session_4_turn0_session_4_request_0_prefill_decode_tra', 'q0064_session_6_turn2_session_6_request_2_history_transfer_a'], [1, 0, 'q0064_session_6_turn2_session_6_request_2_prefill_decode_tra', 'q0124_session_10_turn0_session_10_request_0_prefill_kv_ready'], [1, 0, 'q0124_session_10_turn0_session_10_request_0_decode_request_e', 'q0041_session_3_turn4_session_3_request_4_prefill_decode_tra']]

## B4 执行等价(归因口径)

同图前提不成立(登记差异:跨 request 串行化边/三段式发射/timer gate 时长/MEM 即时完成),exact 条款不适用(总体方案 §9.2);操作性标准 = 差异全部可归因并记录:

- 边界级漂移(delivery tick - 离线记录 tick):
  - prefill_drain_vs_decode_record: n=1177 min=-4 max=10 median=0 ns
  - decode_completion_vs_completion_record: n=1177 min=-8 max=9 median=-1 ns
- 节点级漂移((request,stage) 最大完成 tick - 对应记录 tick):n=2354 min=-7 max=19 median=0 ns;>1us 条目 0。
- completion 跨 tick 逆序 0(硬门);同 tick 组内逆序对 7(信息性)。
- completion 早于同 request decode 决策:0(硬门)。
- 指标不减:completed_requests 基线/在线 = (1177, 1177);incomplete = (0, 0);memory_actions_total = (95790, 95790)(unresolved 基线/在线 = (0, 0));mean_e2e_ns 基线/在线 = 75803356905.9/11917843845.7(replay LUT 时钟 + comm/MEM 即时完成的压缩,类别①,信息性)。

归因类别(不可解释差异 = 0):

- **1_replay_rig_LUT_clock**:COMP chains calibrated to LUT phase durations with integer truncation (runtime_ns = dur*ops//total); comm instant (RECV side, contract 7 rule 2); MEM instant (contract 7 rule 4, sh_1.0-specific); -> ns-scale boundary drift and large e2e compression vs static ET physics
- **2_cross_request_serialization**:replay clears cross-request previous_id edges (segment-1 clearing + own-block-end restore); offline .et keeps physical chains -> cross-request edge diffs (category 1)
- **3_three_segment_emission**:three-segment emission (arrival/prefill-drain/completion boundaries) + block-end restore differences vs offline single-pass writer (category 2/3)
- **4_trigger_gate_dependencies**:segment-2 noc_migrate trigger on segment-1 end, segment-3 completion_evictions/interval gates on decode end — explicit trigger-gate edges encode the same ordering as offline writer anchors (category 4)

## 与 sh_2.0/sh_3.0 的口径差异登记(本仓特有)

- 三段式发射(ARRIVAL/PREFILL_DRAIN/DECODE_COMPLETION→段 1/2/3)对照蓝本两段式;Committer 覆盖规则单向化(watch 必须有节点覆盖,尾段节点可无 watch)为本仓已批机制差异(实录登记 (a))。
- replay 装置 MEM 即时完成(合同⑦第④项,sh_1.0 独有)参与 B4 类别①。
- 两态 KV(LOCAL_HBM/REMOTE_MEMORY)+ edge-rank remote:B2 KV 载荷按六类转移元组对照,无 shard instance_remaining 字段(sh_2.0 的 kv_cache_events.csv 账本不变量不适用;本仓等价物 = transfers_by_stage 全等 + 终态位置)。
- list 版 EventQueue;决策 tick 以 EventQueue EventTime 为准(合同③)。

## 结论

B0/B1/B2 oracle 容差 0 全过;B3 canonical 节点多重集一致(边差异全部归入登记类别);B4 归因口径下不可解释差异 = 0。阶段 2 门槛满足。
