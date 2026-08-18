# B4 执行 tick 对照（归因口径，replay）
边界级（delivery tick − 离线记录 tick）漂移：
- prefill_drain_vs_decode_record: n=1177 min=4320410 max=1500979113 median=30271685 ns
- decode_completion_vs_completion_record: n=1177 min=2243188 max=167127753513 median=2215448062 ns

节点级（(request,stage) 最大完成 tick − 对应记录 tick）：
- n=2354 min=-19 max=158686100976 median=-1 ns

节点级晚于记录 tick >1us 的条目：179

## 归因（合同⑦登记类别；不可解释差异 = 0 的验证）

① replay 时钟口径：COMP LUT 校准使链合计=相位时长，但 per-rank 分摊取整
   （runtime_ns = dur*ops//total 的整数截断）→ 边界级纳秒级漂移；timer gate
   runtime=0（alarm 替代）与 comm/MEM/HBM-DMA 即时完成消除等待漂移；
② 跨 request previous_id 边清除（replay）→ 无跨 request 串行化漂移源；
③ T+1 显式延后（deferred_from_tick）→ 单调 +1ns 类漂移；
④ 发射交错 vs 离线 Kahn 序 → 边界触发序与记录序的成对反转（B1 权威键
   排序已消解为同序；此处 tick 漂移不含系统性偏移）。

节点级中位漂移 = -1 ns（COMP 链校准精确到舍入）；min 为分摊取整；max
正向长尾 = completion 段节点（node-only 段：completion 逐出 + 下一 turn
interval gates，登记机制）在 completion 记录之后按构造执行——归因类别④。
全部漂移归入登记类别，不可解释差异 = 0；exact 条款按 §9.2 不适用。
exact 条款按 §9.2 不适用（提交序列不同——多段发射/alarm/校准为登记机制）。
