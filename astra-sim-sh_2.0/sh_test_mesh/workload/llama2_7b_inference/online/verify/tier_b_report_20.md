# Tier B 等价验收报告（sh_2.0，20.csv 前 30s 输入，2026-08-16）

输入：`traces/astra_compute_20_first_30_seconds_request_queue.csv`（112 session /
1177 请求，sidecar_restore，PROVENANCE 冻结）。基线：`sh_test_mesh/baseline/20_30s`
（静态 ET 运行 + decision_log.jsonl）。

## 分层结果

| 层 | 内容 | 结果 |
|---|---|---|
| B0 输入等价 | request/session 数、8 列转换、arrival/gap、prefix canonical（queue + sidecar + digest 三件套，prefix_mode=sidecar_restore） | **PASS**：digest 1177/1177 逐行重算一致，容差 0，无 prefix 双计 |
| B1 生命周期/决策等价（oracle/replay） | arrival/决策顺序与 tick、IDLE/ACTIVE 转换、输入关闭/排空、决策与 milestone 顺序 | **PASS**：3531/3531 行 order+tick 逐行一致（权威 record 键口径，容差 0） |
| B2 策略等价（oracle/replay） | request→instance 赋值序列、KV action 序列/状态转换、阶段推进 | **PASS**：决策内容 3531/3531 完全一致 |
| B2 real-online（strategy） | 策略不变量 | **PASS**：1177/1177 完成（真实物理 sim_end 1,678s vs 离线静态 2,026s），watch 零 stale、mailbox 零丢失、无死锁（frontier 修复后） |
| B3 图结构等价 | canonical logical node key（per-rank (name,type) 多重集 + 属性 + P2P 配对） | **PASS**：54 rank 全一致；328,976 == 328,976 节点；P2P send/recv/ack 配对一致（tag 为协议层令牌按配对口径比较）；属性 0 差异 |
| B4 执行等价 | C++ 执行 tick（replay）vs 静态 ET | **按归因口径 PASS**：差异全部归入登记类别（见下） |

## B4 归因类别（合同⑦登记）

① replay 时钟口径：COMP 链 LUT 校准（runtime_ns）、comm/MEM/HBM-DMA
   即时完成、timer gate runtime=0（alarm 替代）——replay sim_end 684.8s 为
   LUT 对齐时钟，与离线静态 2026s（真实物理）之差全部源于此类；
② 跨 request previous_id 串行化边：replay 清除 prefill 组跨 request 链
   （LUT 时钟语义，wscllm 蓝本同款）；strategy 保持物理链（与离线 .et
   同构，frontier 接续修复后 cross_request_breaks=0）；
③ tick-end/deferred 顺序合同：post-commit same-tick milestone 经显式 T+1
   唤醒交付（deferred_from_tick 记录），fixture 四断言通过；
④ 发射时序差异：在线按决策边界交错发射 vs 离线全局 Kahn 序——节点集合/
   属性/配对逐字节一致（B3），仅同 rank 内插入顺序不同（canonical 口径
   不作比较项）；strategy 的 KV/逐出决策随真实完成时序变化
   （real-online 语义，节点数 317,726 vs 328,976，归因于完成时序驱动的
   逐出/迁移决策差异，非机制差异）。

## 计数器与 benchmark（阶段 6 材料，20.csv 前30s）

| 运行 | deliveries | callbacks | py_sched_ns（avg/delivery） | gil_wait_ns | 结果 |
|---|---|---|---|---|---|
| offline 静态（仅 C++ 执行） | — | — | — | — | wall 12s（C++），全管线 ~25s |
| 在线 replay | 3521 | 3521（=deliveries，无重入） | 9.44s（2.68ms） | 5.35s | PASS |
| 在线 strategy | 3531 | 3531 | 17.64s（5.00ms） | 7.72s | PASS |
| 在线 strategy+sensing | 3531 | 3531 | 19.81s（5.61ms） | 74.1s | PASS |

- `no_decision_python_callback_count == 0`（三模式）；callback 总数 =
  delivery 数（单 tick 单次交付）；`single_node_bridge_count == 0`；
  `graph_batch_count == delivery_count`；峰值 RSS 415 MiB（在线），
  中间产物有界（request_*.json 3531 + digests；online_nodes.jsonl 133MB
  仅 --dump-nodes 审计运行产出，默认关）。
- 判定门（§9.3）：① 在线总 wall 未出现数量级增长（replay 全程 ~4.7min，
  含 3521 次文件桥往返 bridge_ns≈19.8s；换 IPC 的决策留给后续优化）；
  ② callback 数与边界同量级 ✓；③-⑤ 见计数器行。

## 机制差异登记（显式，非静默）

1. GraphBatchCommitter 覆盖规则扩展（node-only completion 段）；
2. future_alarm 对 in-flight-未-drained 请求的对齐许可；
3. B1 权威 record 键排序出口（sh_2.0 秒级准入排队特性）；
4. interval==0 迟到钳制（max(tick, now+1)，33 次观察计数）；
5. frontier 接续修复（strategy 跨 request 物理链 = 离线 .et 同构；
   审计：intra_request_branches=42 为离线同构并行恢复分支，
   cross_request_breaks=0）；
6. 蓝本缺陷修复：wait_for_work 含 !input_open_ 的 run-end 忙等 → livelock
   guard fail-closed。

## 材料路径

- 比较器：`online/verify/b3_node_compare.py`（B3 + frontier 审计）；
- replay 节点材料：`/tmp/sh20/b3mat3/bridge/online_nodes.jsonl`（--dump-nodes）；
- 各运行产物：`/tmp/sh20/{final_replay,sfix1,sense1,stm1,idle_fixture}`。
