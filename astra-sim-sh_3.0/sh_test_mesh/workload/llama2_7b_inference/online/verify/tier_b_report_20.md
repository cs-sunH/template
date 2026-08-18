# sh_3.0 在线 Tier B 等价性检查报告（阶段 2 首轮，20.csv 前30s）

日期：2026-08-16。材料：
- 基线侧：`sh_test_mesh/baseline/20_30s/`（静态 ET 运行 + decision_log.jsonl
  293,801 行 + manifest）。
- 在线侧：replay 全量运行 `/tmp/sh30_replayA`（PASS，1177/1177，3525 交付，
  memory_actions 64914/0 与离线一致）；strategy 全量运行 `/tmp/sh30_strat6`
  （PASS，1177/1177，3531 交付，no_decision_python_callback_count=0）。

## B0 输入等价：PASS（容差 0）

- 三件套 digest 互校（物化脚本内置 sha256，1177 行全一致）；
- `input_tokens_total == prefix_tokens + prefill_length` 全量成立
  （loader :740-744 + FaceRequest :3304-3310 双防线 + 测试断言）；
- C++ `[METRIC] init` input_requests=1177 与物化一致；
- replay/strategy 两模式 input_requests 均为 1177。

## B1 生命周期/决策等价：部分达成，差异全部可归因（登记）

- 生命周期不变量：IDLE/ACTIVE/DRAINING/FINISHED 五态迁移日志齐全
  （IDLE fixture + 两模式全量运行）；输入关闭排空正常；无死锁
  （全部 request 完成，unresolved=0）。
- 尚未到达的 request 不提前决策：ingress alarm 语义保证（交付 tick =
  到达 tick；decision_log 消费 head-advance fail-closed）。
- 决策序对照（replay online_decision_log vs 离线 decision_log）：
  - decode 流：**顺序完全一致**（1177/1177）；
  - prefill 流：1177/1177 计数一致，顺序反转（首个 @6）；
  - completion 流：计数一致，顺序反转（首个 @774）；
  - tick 差异中位数 25.2ms、最大 6.5s。
- **归因（合同⑦ B4 归因类别①扩展登记）**：sh_3.0 的三段式准入含
  HBM 等待排队（hbm_wait，离线 :4112-4115），turn-0 的 CSV 到达 tick 与
  准入 tick（=prefill 记录 tick）存在真实排队 q（中位 25ms/最大 6.5s）。
  C++ 侧 turn-0 到达 alarm 由 CSV 到达时间排定（机制层固定），蓝本对
  wscllm 同款问题的既定解 = q-吸收（`_DEFER_THRESHOLD_NS`，prefill 相位
  时长吸收 q，链终点 = decode 记录 tick）——本仓已实现且生效（decode 流
  顺序 exact 是其直接证据）。prefill 流反转 = 触发序（到达序）vs 记录序
  （准入序）的成对反转，与蓝本 §10.8 的 3min 案例同性质；completion 流
  反转 = 真实完成时刻对同窗 completion 的纳秒~毫秒级抖动重排。
- **两轮决定性验证（2026-08-16 定稿，结论性证据）**：
  - **Round 1（解析归因）**：在线 prefill 触发序与期望触发序（turn-0 =
    CSV 到达、turn>0 = prefill 记录 tick）**完全一致**（1177/1177，机制
    确定性成立）；到达序 vs 准入序的成对反转共 **23,803 对，其中
    "双方 q==0"的反转 0 对**——prefill 流差异 100% 由准入排队产生，
    无机制缺陷成分；completion tick 偏差 min/median/p95/max =
    0/6/10/18 ns（同窗抖动级别）。
  - **Round 2（q 阈值对照实验）**：`SH30_DEFER_THRESHOLD_NS` 关闭吸收
    后全量重跑——运行可完成（3527 交付）但 **decode 流顺序不再
    exact**、completion tick 偏差 p95/max 膨胀到 32.6s/89.0s（= q 的
    量级）；生产缺省（1µs）下 decode 流 exact 且偏差 ≤18ns。结论：
    **q-吸收是保持决策序等价的必要机制**，生产缺省冻结为 1000ns。
  - B1 终态判定：生命周期/到达/消费不变量 exact；决策流的机制序
    exact（Round 1）；prefill 全局记录序差异 100% 归因准入排队
    （类别①扩展：sh_3.0 三段式准入的 hbm_wait），completion 重排
    归因 ≤18ns 同窗抖动。**B1 以"不变量 exact + 差异 100% 可归因"
    通过（oracle 逐 tick exact 不适用于含准入排队的本仓输入，裁决
    依据 = 蓝本 §10.8 同性质先例 + Round 1/2 决定性证据）。**

## B2 策略等价（oracle/replay）：PASS（容差 0）

- prefill 实例赋值：**0/1177 不匹配**（online vs manifest
  prefill_assignment.instance_index）；
- prefill_affinity_reason：**0/1177 不匹配**（含 first_request_non_edge /
  resident_* / edge_fallback 全部取值）；
- decode 实例恒等于 prefill 实例（红线 #4 不变量，离线 0 违例；
  在线 assignment 同构）；
- KV 动作对照：replay 的 memory_actions_total=64914、unresolved=0、
  kv_event_digest 与离线基线逐字节一致（[METRIC] init 行）。

## B3 图结构等价：PASS（canonical key 对照，2026-08-16 定稿）

对照器 `b3_canonical_compare.py`（类型感知 canonical key = rank + name +
type + is_cpu/is_timer + 该类型实际消费的属性四元组；离线 .et protobuf
直读 vs 在线 SH30_B3_DUMP=1 canonical dump）：

- **节点总数 exact**：offline 300,298 == online 300,298（replay 模式）；
- canonical 多重集差异收敛过程（登记）：① 类型无关属性键的缺席-默认值
  双边不一致（collective 的 comm_src/dst 等）→ 改类型感知键；②
  actionNNN 命名：离线 per-request 跨阶段连续计数 vs 在线三批各自重启
  → 改 per-request 连续计数器（builder `_action_seq`）；③ turn-0 到达
  gate 命名：离线短前缀（q{queue}_{request_id}）vs 在线长前缀 → 对齐；
- **终态**：剩余差异 10,654/300,298（3.5%）**全部为 comm_tag-only**
  （name/rank/type/bytes/src/dst 逐项相同，仅 tag 数值不同）；
- **tag 差异归因（登记类别）**：tag 由 TransferTagAllocator 按发射序
  全局分配——离线全局 KV 因果预排序 vs 在线决策边界序，分配序必然不同；
  tag 是运行期 send/recv 配对标识（非语义键）。**补偿不变量实测**：
  在线 (rank,src,dst,方向,tag) 零碰撞（10,654 个传输节点全部可无歧义
  配对；运行完成本身即配对一致性的端到端证据）。
- **B3 判定：节点 count/name/type/attrs 容差 0（tag 归因类别豁免），
  通过。** 边对照：插入序与 id 空间天然不同（合同⑦ 不作比较项），
  within-request 依赖由同构发射函数保证（同一 _emit_kv_transfer/
  transformer_pass_aggregated 驱动）。

## B4 执行等价：归因路径

- replay sim_end=995,087,425,606 vs 离线静态 1,714,696,296,631：归因类别
  ①（replay 装置 LUT 时钟口径：无网络/远端/HBM 传输时长 + 并发校准 COMP
  + RECV 侧 comm-0 + 远端 MEM/HBM restore 即时完成——本仓第④项豁免）；
- strategy sim_end=1,324,294,856,383（真实物理；与离线差异属 real-online
  可解释性范围，阶段 3 完成归因报告）。

## B4 执行等价：归因完成

- replay sim_end=995,087,425,606 vs 离线静态 1,714,696,296,631：归因
  类别①（replay 装置 LUT 时钟口径四项：无网络/远端/HBM 传输时长、
  并发校准 COMP、RECV 侧 comm-0、远端 MEM/HBM restore 即时完成）；
- strategy sim_end=1,324,294,856,383（真实物理）：与离线差异属
  real-online 可解释性范围——归因：真实完成事件时序（无 LUT 时钟）、
  排队物理化（per-rank 单槽位串行）、HBM 50/50 竞争、远端 FIFO 排队；
  感知开关（--sensing）开/关决策序列 **3531 行逐字节一致**（查询/审计
  输入不进策略判据的实测证据）。

## 阶段 4 附：结束总账核对（sensing 运行 /tmp/sh30_sense1）

`sh30_ledger_reconcile.py`：R0（1177×3 决策行）PASS；R1（tick 序 +
decode==prefill 实例）PASS；R2（issued 层核销 + completed 覆盖 1177）
PASS；R5（末次交付残差 = 自完成 barrier 尾部 6 条，合同①"物理收尾不
阻塞核销"）PASS；R6 峰值 injected node_count/rank = 1（观测）。
**verdict：对平（balanced）。**

## 总结论（阶段 2 定稿）

- B0 输入等价：PASS（容差 0）；
- B1 生命周期/决策：不变量 exact + 差异 100% 可归因（两轮决定性证据）；
- B2 策略等价：oracle exact（0/1177）；
- B3 图结构等价：节点 canonical 多重集容差 0（tag 归因类别豁免，
  补偿不变量实测零碰撞）；
- B4 执行等价：全部差异归因（类别① + real-online 四源）；
- 对账：分层账本对平；感知开关决策序列逐字节一致。

**Tier B（关感知在线 vs 离线 ET 基线）分层验收：通过（含登记归因
类别）。**
