# 合同⑨ 窗口与未来参数合同

冻结时间: 2026-08-16（阶段 0 步骤 0-4，物化实测值已填）。

## 口径
- 窗口派生：session 首行 arrival_time < 30e9 才纳入；turn 派生到达 =
  上一到达 + 上一 gap（human_time||tool_time 逐行，空→0）；>= 30e9 截断。
  窗口内 turn 重编号（0 起）；8 列转换（session_<id>/turn/request_id 命名）。
- interval 口径 = human_time||tool_time 逐行 gap。
- 决策粒度：request/stage 对照（离线决策记录粒度为逐 chunk 迭代
  FaceIterationRecord，离线 ET 按 request 整段构图；B3/B4 不要求在线
  逐迭代构图）。

## 裁决（未来参数表，值 = 物化实测 2026-08-16）
| 参数 | 裁决 | 值/说明 |
|---|---|---|
| p_chunk | 固定实验配置 | 512 |
| average_decode_length | 标定常数（物化期 sum/len 导出，离线/在线共用） | 459.4944774851317；禁止在线增量 mean |
| max_d_token | 标定常数（LUT token bin 覆盖上界） | 53924 |
| request_count | 标定常数（LUT d_batch 覆盖上界） | 1177 |
| FaceLut | **在线继续消费**（与蓝本相反）：select_decode_instance 代价查询（:992-1003）是策略输入；迭代计时角色（start_ready_iterations :3792-3797）offline-only | 固定常数构建与离线同值 |
| Roofline 估算器三件套 | 在线继续消费（task-load 策略输入，无全量依赖） | 只读 import |
| order_plans_for_static_emission | offline-only 构图机制（全局 Kahn 依赖全量 plan） | 在线按事件触发发 batch，禁止调用 |
| planner pickle 缓存 | offline-only | 在线路径不读不写 |
| 同 tick 排序 | 保留 completion(0) < arrival(1)，同 priority 按 sequence | 在线事件索引队列同序 |
| kv_reserve_context_tokens | 固定实验配置 | 1,000,000 |
| request_queue_context_csv | 默认 off（合同⑧） | 开启须用户裁决 |

关键复核结论（步骤 0-4 复核记录）：LUT 只在 decode 代价与离线迭代计时
两处消费；估算器不读全量集合（average_decode_length 来自标定常数）；
未发现策略决策额外读取全量派生量。**蓝本"LUT offline-only"结论对本仓
不成立，不得误移植。**

## 验证方法
- B0 输入等价（容差 0）；阶段 2 B2 oracle 模式决策序列一致即参数口径
  无泄漏的运行时证据。
