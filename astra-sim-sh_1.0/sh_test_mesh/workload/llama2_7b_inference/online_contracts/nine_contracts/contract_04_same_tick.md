# 合同④ 同 tick 合同(物理事件 → 决策 → commit → deferred)

冻结时间: 2026-08-16(阶段 0,步骤 0-4);审查通过是阶段 1 开工硬门槛。

## 口径
- 单 tick 内固定顺序:物理事件 invoke → pop 之后 tick-end 收口回调 → Python
  决策交付(单 tick 单次 delivery)→ GraphBatch validate/prepare/commit →
  post-commit deferred drain(commit 产生的节点经 `schedule_event_deferred` 执行)。
- **单 tick 单次 delivery**:commit 后同 tick 形成的新 milestone 不重入 Python;
  deferred drain 结束后 mailbox 非空 → 必须显式排下一 delivery 边界
  (schedule_event(T+1, delivery_cb) 或等价,StateDelta 记录 T→T+1 延后)。
- 同 tick 排序(离线语义保留,face_scheduler.py:2970【亲核】):batch 按
  `(priority, sequence)` 排序;iteration_complete priority 0(:2957-2962)
  先于 arrival priority 1(:2823/:3084);在线事件索引队列必须同序。
- 硬规则:tick-end 回调之后禁止向主队列插 current_time 事件(下一轮 proceed
  :33 严格递增断言必失败);同 tick 事件走 `schedule_event_deferred`。

## 裁决
- tick-end 回调与 deferred 通道默认不启用;静态基线逐字节等价是回归门。
- 决策交付门控:无决策事件不进 Python(`no_decision_python_callback_count`=0;
  `tick_end_without_decision_count` 允许非零单独报告)。
- DecisionMailbox reason 闭集(阶段 1 最小集):`ARRIVAL`、`PREFILL_DRAIN`、
  `DECODE_COMPLETION`、`REQUEST_COMPLETE`;保留 `RESOURCE_READY` 枚举位
  (阶段 1 不产生;本仓 HBM 准入重试由容量释放事实在 completion 边界驱动,
  无需新增 C++ reason)。普通节点完成/rank idle/metrics-only 事件不置 decision 标志。
  - PREFILL_DRAIN = 该 request prefill 段(history_evictions/history_transfer/
    prefill_evictions/prefill 屏障/prefill)全部节点完成;
  - DECODE_COMPLETION = decode 段(decode_evictions/prefill→decode 迁移/
    decode 屏障/decode)全部节点完成;
  - REQUEST_COMPLETE = 全部阶段(含 completion_evictions)完成。

## 验证方法
- EventQueue 单元测试(用例 A/B/C/D,照蓝本);post-commit same-tick milestone
  fixture(步骤 1-11 硬验收);静态基线 runall cmp 一致。
