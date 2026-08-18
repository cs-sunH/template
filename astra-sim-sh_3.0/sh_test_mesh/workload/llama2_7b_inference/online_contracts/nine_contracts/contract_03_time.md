# 合同③ 时间合同(EventTime ↔ ns 换算与迟到规则)

冻结时间: 2026-08-16(阶段 0,步骤 0-4);审查通过是阶段 1 开工硬门槛。

## 口径

- EventQueue `EventTime` 与纳秒(ns)的换算、舍入规则:先冻结口径与验证
  方法,**具体换算值由执行者在阶段 1 步骤 1-2 读
  `astra-sim/network_frontend/analytical/common/CommonNetworkApi.cc` 的
  `sim_get_time`(:71【侦察】)/ `sim_schedule`(:83-97【侦察】)与
  `astra-sim/system/Sys.cc` 的 `boostedTick`(:453【侦察】)后填入本合同的
  "换算值"小节,并在该步骤验证命令中复核**。
- StateDelta 的 tick 以 EventQueue 全局 `EventTime` 为唯一决策时间。
- 迟到 request 规则:到达 tick 前不得进入策略决策/账本准入/图提交
  (窗口内可预取未来请求原始元数据,但决策边界由 alarm/到达事件驱动)。
- 物化侧时间口径(与在线运行无换算关系,仅输入侧):8 列执行队列的
  `session_arrival_time_ns` 为源绝对时间(未归一化),`inter_request_interval_ns`
  = 逐行 `human_time || tool_time` gap(空→0);两者均须为 1000 ns 整数倍
  (见合同⑨与方案文档 §3 步骤 0-1 物化规则)。

## 裁决(换算值——阶段 1 步骤 1-2 填写)

填写时间: 2026-08-16(阶段 0 预核;源码亲核: CommonNetworkApi.cc:71-81、Sys.cc:465-476、Common.hh CLOCK_PERIOD=1。本仓为 sh_2.0 同源 map 版,换算与蓝本一致;源码侦察:
`CommonNetworkApi.cc` `sim_get_time`(:71)/`sim_schedule`(:83-97)、
`Sys.cc` `boostedTick`(:453 实测 :468-479)、`Common.hh` `CLOCK_PERIOD`(:16))。

- `sim_get_time` 返回值单位与 EventTime 关系:
  `timespec_t{time_res=NS, time_val=static_cast<long double>(event_queue->get_current_time())}`,
 即返回值就是 EventQueue 的全局 `EventTime`,单位 ns,1:1 无缩放。
 注释明示:经 long double 传递是为在 2^53 以上仍保持整数 ns 精确
 (EventTime=uint64 ns),**无舍入**。
- `sim_schedule` 的 tick 语义(相对/绝对、含端点):
  相对 delta,`assert(delta.time_res == NS)`;绝对时刻 =
  `sim_get_time().time_val + delta.time_val`,再
  `static_cast<EventTime>` 直接 `schedule_event`。即:相对 delta(ns),
  换算成绝对 EventTime 时刻排事件;**含端点**(下一事件可在当前时刻之后任意
  ≥1 ns;`assert(event_time_ns >= current)` 保证不早于当前,严格递增由
  EventQueue::proceed 的 :33 断言强制)。
- `boostedTick` 与 EventQueue current_time 的关系:
  `Tick = sim_get_time().time_val / CLOCK_PERIOD`,`CLOCK_PERIOD = 1`
  (Common.hh:16,注释 "1ns"),故 `Tick == EventTime == current_time`(ns),
  1ns = 1 tick,整数除无舍入。
- ns ↔ EventTime 换算系数与舍入方向:
  **1 ns = 1 EventTime,双向 1:1,无系数、无舍入**;StateDelta 的 tick 即
  该 EventTime 全局决策时间(见"口径")。注入侧 arrival_world_ns 直接作为
  到达 alarm 的 EventTime(未来 tick 才合法;迟到请求钳制到 current+1,
  即"当前时刻到达",见 RequestIngress::drain_commands)。

## 验证方法

- 阶段 1 步骤 1-2:读上述三处源码,把换算值填入本合同并跑 IDLE fixture
  (注入 tick T 断言 alarm 在 T 触发)。
- 阶段 2 B4(oracle/replay):C++ 执行 tick(node issue/complete)与静态 ET
  运行一致(同图、同后端、同资源参数、同到达序列、同提交序列前提)。
- 迟到规则:阶段 2 B1 检查"尚未到达的 request 不得提前进入决策"。
