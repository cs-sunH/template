# 合同⑦ Tier B 合同（B0-B4 分层验收）

冻结时间: 2026-08-16（阶段 0 步骤 0-4）。全部验收仅使用 20.csv 前30s
输入（用户指示 2026-08-15，方案 §0.3）。

## 口径
- oracle/replay 模式：回放阶段 0 decision_log（离线决策时序冻结），B1
  （含 tick）/B2/B3 exact 仅在此模式；B4 走归因路径。
- real-online（strategy 关感知）：验策略不变量（决策输入与离线同一函数
  同输入同输出）、账本正确性与差异可解释性，不要求 exact。
- 分层：B0 输入等价（含 sidecar_restore prefix canonical，容差 0）；
  B1 生命周期/决策；B2 策略（prefill 实例+assignment_key、decode 实例与
  candidates 摘要、KV action 序列/状态转换）；B3 图结构（canonical
  logical node key，request/stage 粒度）；B4 执行等价（同图前提不满足，
  操作性标准 = 差异全部可归因并记录）。
- 任何层级失败不得通过放宽比较器掩盖。

## 裁决
- replay 装置时钟口径（蓝本裁决继承 + 本仓扩展，仅限 --online-mode replay）：
  ① COMP 按 LUT 校准窗口并发执行（放行 HardwareResource 单槽位门，**本仓
  含 hbm_dma 槽位**）；
  ② comm 节点即时完成，**边界 = RECV 侧**（五仓红线，不得再尝试 SEND 侧
  豁免）；
  ③ prefill 段发射清除 prefill 组各 rank 跨 request previous_id（保留
  within-request 串行化 + own-prefill-end 恢复 + 同 session interval
  gate；emitted-ranks-only 精确语义）；
  ④ **本仓新增裁决（登记）**：MEM_LOAD/MEM_STORE（远端 FIFO）与 HBM
  restore DMA（LocalHbmBandwidthModel）节点在 replay 模式**即时完成**（与
  ② 同理：planner LUT 时钟不含这些节点的物理时长对应物；依赖/watch/
  terminal/释放链完整保留）；strategy/静态路径真实物理不动。
- B4 归因类别至少覆盖：① replay LUT 时钟口径（含本④）；② 跨 request
  previous_id 串行化边差异；③ tick-end/deferred 顺序合同。
- 本仓预期新增对照面：eviction store 段与 restore 流水段 within-request
  边、trigger gate 的在线替代（决策边界即触发）。

## 验证方法
- 比较器 online/verify/tier_b_compare.py：基线侧 = 阶段 0 归档静态 ET 运行
  + decision_log.jsonl；在线侧 = online_decision_log.jsonl + 在线 metrics
  + graph_batch_digests.jsonl。报告 online/verify/tier_b_report_20.md。
