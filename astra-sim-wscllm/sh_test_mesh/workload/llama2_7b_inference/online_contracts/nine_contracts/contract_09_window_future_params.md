# 合同⑨ 窗口与未来参数合同(含 LUT 关键裁决复核记录)

冻结时间: 2026-08-15(阶段 0,步骤 0-4);审查通过是阶段 1 开工硬门槛。

## 口径(窗口派生口径,与方案文档 §3 步骤 0-1 物化规则一致)

- 窗口:session 首行 `arrival_time < 30,000,000,000 ns` 才纳入;turn 派生
  到达 = 上一到达 + 上一 gap,派生到达 `>= 30e9` 即截断该 session 后续
  (即"只有派生到达 < 30s 的 request 属于前 30 秒输入")。
- 窗口内 turn 重编号:纳入的 session 从 turn 0 连续编号;窗口外 turn 丢弃,
  不做到达时间归一化。
- 边界 arrival/gap 推导:`session_arrival_time_ns` = 源绝对时间;
  `inter_request_interval_ns` = 逐行 `human_time || tool_time` gap(空→0);
  两者均须 1000 ns 整数倍(实测 0 违例)。
- 8 列转换:session_id,turn_index,request_id,prefill_length,decode_length,
  session_arrival_time_ns,inter_request_interval_ns,description。
- interval 口径选择:"human_time||tool_time 逐行 gap"(总体方案 §5.7 合同⑨
  二选一之一),已登记于方案文档 §3 步骤 0-1 物化规则。
- 实测:1177 请求 / 112 session(主控 2026-08-15 裁决接受;原预估"约 1830"
  为口径笔误,已在方案文档更正)。

## 裁决

### 未来参数裁决表(方案 §12 冻结版)

| 参数 | 位置(【亲核】) | 裁决 | 在线实现要求 |
|---|---|---|---|
| session_lru `p_chunk` | trace_config.csv:15(=512,经 generate_wsc_llm_trace.py:576-580 传入) | 固定实验配置 | 在线 runner 显式传同一配置值,不改动 |
| legacy `p_chunk` | wsc_llm_scheduler.py:1306(全量 mean) | offline-only;legacy 在线迁移须改固定配置 | 禁止在线计算全量 mean |
| `max_d_token` | wsc_llm_scheduler.py:1308/:1685 | offline-only 产物(LUT 惰性 token bin 上界) | 在线热路径不消费 |
| `request_count` | wsc_llm_scheduler.py:1314/:1691 | offline-only 产物(LUT d_batch 上界) | 在线热路径不消费 |
| LUT(`WscLlmTimingLut` :636-809) | 依赖上述三者 | offline-only 产物 | 在线迭代推进由 C++ 真实完成事件驱动,不读 LUT |
| 同 tick 排序 | wsc_llm_scheduler.py:1475-1477/:2002-2004 | 保留:completion(0) < arrival(1),同 priority 按 sequence | 在线事件索引队列同序 |
| 决策粒度口径 | — | 按 request/stage 对照 | B3/B4 不要求在线逐迭代构图 |

### LUT 关键裁决复核记录(执行者 2026-08-15 逐行复核,无反例)

核心裁决:wscllm session_lru 分支的策略决策输入 = 排队深度
(`PrefillQueueSnapshot.ordering_key`)+ KV 容量(`SessionKVCacheManager`)
+ 静态路由;LUT 只用于离线事件循环 `start_ready_iterations` 推进模拟时钟
(估计 iteration 时长并推 iteration_complete 事件),不参与准入/选路/KV
决策。在线模式由 C++ 真实完成事件推进,LUT 及 max_d_token/request_count
在线热路径全部不消费。

四个函数逐行复核证据(`wsc_llm_scheduler.py`,行号以 2026-08-15 工作树为准):

1. `try_admit_prefill`(:1746-1849)——**不读 LUT**。
   读取面:runtimes(request 运行时)、capacity_epoch/:1730 prefill_attempt_epoch
   (epoch 防重入)、kv_manager.reserve_request_capacity(:1764)/session_snapshot
   (:1784)/prepare_history(:1789)/grow_prefill(:1826)/release_request_capacity
   (:1799)、model、topology、p_chunk;准入判据 = KV 容量裁决(reservation/
   prepare_history/growth 的 admitted 标志),无任何 timing_lut 引用。
2. `try_admit_waiting_decodes`(:1851-1913)——**不读 LUT**。
   读取面:waiting_decode_admissions、decode_admission_dirty/epoch(:1852-1864)、
   kv_manager.release_request_capacity(:1876)/move_prefill_to_decode(:1878)/
   grow_decode(:1891)、instances;全部为排队/KV 容量逻辑,无 timing_lut 引用。
3. `select_prefill_instance`(:826-829)——**不读 LUT**。
   仅 `min(queues, key=lambda queue: queue.ordering_key)`(:829),
   ordering_key=(request_count, instance_index)(:821-823),最少排队优先、
   平局取 instance_index 最小;无 timing_lut 引用。
4. `start_ready_iterations`(:1915-1996)——**唯一读 LUT 的函数,仅用于
   计时**。先调 try_admit_waiting_decodes(:1917)与 try_admit_prefill(:1928)
   (决策先行),然后 `timing_lut.lookup(...)`(:1943 prefill、:1962 decode)
   仅用于 `end_ns = now_ns + entry.iteration_time_ns`(:1974)并
   push_event(end_ns, 0, "iteration_complete", ...)(:1996)——即 LUT 只
   估计模拟时钟推进,不参与任何准入/选路/KV 决策。

结论:**无反例**。LUT 仅离线事件循环计时用;在线模式由 C++ 真实完成事件
推进,本裁决成立。若未来在策略决策路径发现任何 LUT 读取,按 §0.4 红线
立即停止上报。

## 验证方法

- B0:窗口派生与 sidecar digest 一致性(见合同⑧)。
- 阶段 1-9:在线策略路径以离线 `_plan_wsc_llm_session_lru_recompute`
  (:1667-2207)为蓝本逐行对应迁移,删除 LUT 计时部分、保留排队/配对语义;
  每处迁移注释标注离线对应行号。
- 阶段 2:B2 oracle 模式 KV action 与 kv_cache_events.csv 对照;
  阶段 3:对账与差异归因。
