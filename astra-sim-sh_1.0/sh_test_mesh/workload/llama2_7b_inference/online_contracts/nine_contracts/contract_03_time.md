# 合同③ 时间合同(EventTime ↔ ns 换算与迟到规则)

冻结时间: 2026-08-16(阶段 0,步骤 0-4);审查通过是阶段 1 开工硬门槛。

## 口径
- StateDelta 的 tick 以 EventQueue 全局 `EventTime` 为唯一决策时间。
- 迟到 request 规则:到达 tick 前不得进入策略决策/账本准入/图提交。
- 物化侧时间口径(仅输入侧):`session_arrival_time_ns` 为源绝对时间
  (未归一化),`inter_request_interval_ns` = 逐行 `human_time || tool_time`
  gap(空→0);均须 1000 ns 整数倍(实测 0 违例)。

## 裁决(换算值——阶段 1 步骤 1-2 填写)
填写时间: 2026-08-16(步骤 1-2;本仓与蓝本同文件同逻辑,复核后照填):
- `sim_get_time` 返回值即 EventQueue 全局 `EventTime`,单位 ns,1:1 无缩放,
  无舍入(long double 传递仅为 2^53 以上保持整数 ns 精确)。
- `sim_schedule` 相对 delta(ns),绝对时刻 = current + delta,含端点
  (≥1 ns;`assert(event_time_ns >= current)`)。
- `boostedTick`:`CLOCK_PERIOD = 1`(Common.hh:16),`Tick == EventTime`(ns),
  整数除无舍入。
- **1 ns = 1 EventTime,双向 1:1,无系数、无舍入**;注入侧 arrival_world_ns
  直接作为到达 alarm 的 EventTime;迟到请求钳制到 current+1。

## 验证方法
- 步骤 1-2 读三处源码复核换算值并跑 IDLE fixture(注入 tick T 断言 alarm 在 T 触发)。
- 阶段 2 B4(oracle/replay)执行 tick 对照;迟到规则 B1 检查。
