# 合同④ 同 tick 合同(物理事件 → 决策 → commit → deferred)

冻结时间: 2026-08-15(阶段 0,步骤 0-4);审查通过是阶段 1 开工硬门槛。

## 口径

- 单 tick 内固定顺序:
  1. 物理事件(EventQueue 当前 EventList 全部 invoke);
  2. pop 之后 tick-end 收口回调(回调未设置时与改造前行为完全一致);
  3. Python 决策交付(DecisionMailbox 非空才进 Python,单 tick 单次 delivery);
  4. GraphBatch validate/prepare/commit(四段见合同⑤);
  5. post-commit deferred drain(commit 产生的节点/事件经
     `schedule_event_deferred` 执行)。
- **单 tick 单次 delivery**:commit 后同 tick 形成的新 milestone 不重入
  Python(计数器断言);deferred drain 结束后 mailbox 非空 → 必须显式排
  下一 delivery 边界(schedule_event(T+1, delivery_cb) 或等价机制,并在
  StateDelta 记录 T→T+1 延后)——否则主队列空时 wait_for_work 永久阻塞、
  milestone 永不交付(阶段 1 步骤 1-11 fixture 硬验收)。
- 同 tick 排序(离线语义保留):completion(priority 0)先于 arrival
  (priority 1),同 priority 按 sequence 稳定排序
  (`wsc_llm_scheduler.py:1475-1477` legacy、`:2002-2004` lru【亲核】);
  在线事件索引队列必须同序。
- 硬规则:禁止任何代码在 tick-end 回调之后向主队列插入 current_time 事件
  (下一轮 proceed 的严格递增断言必失败);同 tick 事件必须走
  `schedule_event_deferred`,未来事件才走 `schedule_event`。

## 裁决

- tick-end 回调与 deferred 通道默认不启用(set 回调前 = 零行为变化);
  静态基线逐字节等价是每次机制改动的回归门。
- 决策交付门控:无决策事件不进 Python(`has_decision_work()` 为 false 时仅
  递增 `tick_end_without_decision_count`,允许非零单独报告;
  `no_decision_python_callback_count` 必须为 0)。
- DecisionMailbox reason 闭集(阶段 1 最小集):`ARRIVAL`、`PREFILL_DRAIN`、
  `DECODE_COMPLETION`、`REQUEST_COMPLETE`;保留 `RESOURCE_READY` 枚举位
  (legacy 迁移用,阶段 1 不产生)。普通节点完成/rank idle/metrics-only 事件
  不置 decision 标志。
  - PREFILL_DRAIN = 该 request prefill 阶段全部节点完成;
  - DECODE_COMPLETION = 该 request decode 阶段(request-aggregated 构图下
    为其全部 decode 节点)完成;
  - REQUEST_COMPLETE = 该 request 全部阶段完成(收尾/核销/下一次 session
    arrival 排程的边界)。

## 验证方法

- EventQueue 单元测试(list 版;tick-end 回调在物理事件后、deferred 前恰好
  一次;deferred 按插入顺序同 tick 执行,deferred 内再 deferred 同轮 drain;
  未设置回调时 1000 次随机 schedule/proceed 与改造前 reference 实现逐事件
  一致)。
- post-commit same-tick milestone fixture(阶段 1 步骤 1-11):PREFILL_DRAIN
  不重入、下一 delivery epoch 交付、decode 图正常提交、零事件丢失、
  无死锁。
- 静态基线 runall cmp 一致(每次机制改动后)。
