# 合同④ 同 tick 合同（物理事件 → 决策 → commit → deferred）

冻结时间: 2026-08-16（阶段 0 步骤 0-4）；审查通过是阶段 1 开工硬门槛。

## 口径
- 单 tick 固定顺序：①物理事件（当前 EventList 全部 invoke）→ ②
  erase(begin()) 之后 tick-end 收口回调 → ③ Python 决策交付（单 tick 单次
  delivery）→ ④ GraphBatch validate/prepare/commit → ⑤ post-commit
  deferred drain（经 schedule_event_deferred）。
- **map 版收口实现形态（§4.2.3）**：`proceed()` = begin/invoke（in_invoke_
  =true 包裹）→ erase → tick-end 回调 → deferred drain。erase 之后不得
  再触碰 current_event_list 引用；回调放 erase 之后（若放之前，同 tick
  try_emplace 合入已清空 list 会被 erase 静默丢弃——严禁）。
- 硬规则：tick-end/deferred 期间 `schedule_event(current_time)` 下一轮
  proceed 的 :31 严格递增断言必失败（fail-fast）；同 tick 事件必须走
  schedule_event_deferred，未来事件才走 schedule_event。
- 同 tick 排序（离线语义保留）：completion(priority 0) < arrival(priority 1)，
  同 priority 按 sequence（face_scheduler.py:3542-3558/:3842）。
- 单 tick 单次 delivery；deferred drain 后 mailbox 非空必须显式排下一
  delivery 边界（T+1 并记录延后——步骤 1-11）。
- current-time 自调度审计（§4.2.4）：FluidScheduler flush（:82-85）与
  CommonNetworkApi::sim_recv（:119-127）两处按 in_invoke_context 分流到
  deferred；LocalHbmBandwidthModel/AnalyticalRemoteMemory/Sys 全部
  register_event 调用点 delta 恒 >0，安全。

## 裁决
- reason 闭集（阶段 1 最小集）：ARRIVAL、PREFILL_DRAIN、DECODE_COMPLETION、
  REQUEST_COMPLETE；保留 ADMISSION_RETRY/RESOURCE_READY 枚举位（阶段 1
  不产生）。无 TRANSFER_CONFIRMED（本仓离线无 transfer 确认决策点，§4.1
  第 9 条裁决）。
- PREFILL_DRAIN = prefill 阶段全部节点完成（成员含 history 段、readiness
  barrier、prefill 算子段与 PARTIAL 恢复流水 suffix 尾段——transfer tail
  晚于 compute tail 时不提前 fire）；DECODE_COMPLETION = decode 阶段
  （含 prefill→decode 迁移 transfer 尾与 decode 就绪 barrier）；本仓
  decode 即末阶段，DECODE_COMPLETION 与 REQUEST_COMPLETE 同 tick 同
  delivery。离线 iteration_complete 不映射任何 reason。

## 验证方法
- event_queue_deferred_test 用例 A/B/C/D（D 为 map 版特有 fail-fast 证明）。
- post-commit same-tick milestone fixture（步骤 1-11，CORE）。
- 静态基线 runall cmp 一致。
