# 合同④ 同 tick 合同(物理事件 → 决策 → commit → deferred)——sh_3.0

冻结时间: 2026-08-16(阶段 0,步骤 0-4);审查通过是阶段 1 开工硬门槛。

## 口径

- 单 tick 内固定顺序:
  1. 物理事件(EventQueue 当前 EventList 全部 invoke;map 版
     `EventQueue::proceed` `invoke_events` 后按迭代器 `erase`);
  2. erase 之后 tick-end 收口回调(回调未设置时与改造前行为完全一致;
     **严禁把回调放在 erase 之前**——同 tick 事件会被 try_emplace 合入
     已清空的当前列表并随 erase 静默丢弃);
  3. Python 决策交付(DecisionMailbox 非空才进 Python,单 tick 单次 delivery);
  4. GraphBatch validate/prepare/commit(四段见合同⑤);
  5. post-commit deferred drain(commit 产生的节点/事件经
     `schedule_event_deferred` 执行)。
- **单 tick 单次 delivery**:commit 后同 tick 形成的新 milestone 不重入
  Python(计数器断言);deferred drain 结束后 mailbox 非空 → 必须显式排
  下一 delivery 边界(schedule_event(T+1, delivery_cb) 或等价机制,并在
  StateDelta 记录 T→T+1 延后)。
- 同 tick 排序(离线语义保留,`face_scheduler.py:3903`):completion
  (priority 0)先于 arrival(priority 1),同 priority 按 sequence 稳定排序。
- 硬规则:禁止任何代码在 tick-end 回调之后向主队列插入 current_time 事件
  (map 版 `EventQueue.cpp:31` 严格递增断言必失败);同 tick 事件必须走
  `schedule_event_deferred`,未来事件才走 `schedule_event`。

## 裁决

- tick-end 回调与 deferred 通道默认不启用(set 回调前 = 零行为变化);
  静态基线逐字节等价是每次机制改动的回归门。
- 决策交付门控:无决策事件不进 Python(`has_decision_work()` 为 false 时仅
  递增 `tick_end_without_decision_count`,允许非零单独报告;
  `no_decision_python_callback_count` 必须为 0)。
- DecisionMailbox reason 闭集(阶段 1 最小集):`ARRIVAL`、`PREFILL_DRAIN`、
  `DECODE_COMPLETION`、`REQUEST_COMPLETE`;保留 `RESOURCE_READY` 枚举位
  (本仓不产生:准入重试全部由同 tick completion/arrival 批内复查触发,
  对应离线 `face_scheduler.py:4060-4061` 的 retry_admissions 机制)。
  普通节点完成/rank idle/metrics-only 事件不置 decision 标志;
  **本仓补充:HBM 模型 transition 事件(`LocalHbmBandwidthModel.cc:214-218`)
  与远端 FIFO 推进事件(`AnalyticalRemoteMemory.cc:196-200`)是普通物理
  事件,不产生 decision reason**。
  - PREFILL_DRAIN = 该 request prefill 段全部节点完成(含 prefill end
    barrier all_reduce 与段内 transfer/HBM restore 节点,成员 = 段发射时
    登记的全部节点,精确到 CompletionKey);
  - DECODE_COMPLETION = 该 request decode 段全部节点完成(含 decode
    end barrier);
  - REQUEST_COMPLETE = 该 request 全部阶段完成(completion_evictions 传输
    节点属于物理收尾,其完成不阻塞 REQUEST_COMPLETE 的核销——容量释放
    在 Python 账本决策时已生效)。
- **不设 TRANSFER_CONFIRMED reason(本仓裁决)**:离线规划器的 KV 传输在
  决策时即时生效(账本先行),传输节点只承担物理时序;没有任何策略决策
  以"传输物理完成"为触发条件。若执行中发现反例,立即停下报告。
- sticky 等待的在线触发映射:分支 1 候选为空与分支 2 驻留实例暂不可行的
  `return False` → 请求留在 pending_admissions,不产生任何 C++ 事件;重试
  由后续 completion/arrival 批的 admit pass 触发;等待中的请求不占图、
  不占 C++ 状态,只占 Python 账本。

## 验证方法

- EventQueue 单元测试(**map 版**,含 §4.2 用例 A-D:tick-end 回调时机/
  deferred 顺序与嵌套/1000 次随机对照 reference 实现/invoke 上下文合批
  与 tick-end 回插死亡测试)。
- post-commit same-tick milestone fixture(阶段 1 步骤 1-11):PREFILL_DRAIN
  不重入、下一 delivery epoch 交付、decode 图正常提交、零事件丢失、
  无死锁。
- 静态基线 runall cmp 一致(每次机制改动后)。
