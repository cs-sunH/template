# 合同③ 时间合同（EventTime ↔ ns 换算与迟到规则）

冻结时间: 2026-08-16（阶段 0 步骤 0-4）；换算值亲核同日。

## 口径
- StateDelta 的 tick 以 EventQueue 全局 EventTime 为唯一决策时间。
- 迟到 request 规则：到达 tick 前不得进入策略决策/账本准入/图提交；
  超界（>= 30e9）拒绝（输入约束）。
- 物化侧：`session_arrival_time_ns` = 源绝对时间（未归一化），
  `inter_request_interval_ns` = 逐行 human_time||tool_time gap；均 1000ns
  整数倍（loader 校验，实测 30s 输入 0 违例）。

## 裁决（换算值，亲核 2026-08-16）
- **1 EventTime = 1 ns**：`CommonNetworkApi.cc:71-81` sim_get_time 直取
  EventQueue 时间（NS 单位，long double 传递保 >2^53 精度，无舍入）；
  `Sys.cc:476` `boostedTick()` = time / CLOCK_PERIOD，CLOCK_PERIOD=1
  （`astra-sim/common/Common.hh:16`）。双向 1:1，无系数、无舍入。
- sim_schedule：相对 delta(ns) → 绝对 EventTime → schedule_event；
  `assert(event_time >= current_time)`（EventQueue.cpp:45），同 tick 严格
  递增由 proceed 的 :31 断言强制。
- 注入侧 arrival_world_ns 直接作为到达 alarm 的 EventTime；迟到钳制到
  current+1。

## 验证方法
- 步骤 1-2 IDLE fixture（注入 tick T 断言 alarm 在 T 触发）；
  阶段 2 B4 oracle 模式 C++ 执行 tick 对照。
