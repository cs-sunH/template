# 合同⑨ 窗口与未来参数合同（含 face 版 LUT 关键裁决五条）

冻结时间: 2026-08-16（阶段 0，步骤 0-4）；审查通过是阶段 1 开工硬门槛。
**face 版与蓝本（face 蓝本 = wscllm）合同⑨的关键差异：蓝本 LUT 全部
offline-only；face 的 LUT 是 decode 策略决策的代价来源（`select_decode_instance`
消费），裁决为"标定常数 + 在线保留"（face 方案 §12）。**

## 口径（窗口派生口径，与方案文档 §3 步骤 0-1 物化规则一致）

- 窗口: session 首行 `arrival_time < 30,000,000,000 ns` 才纳入；turn 派生
  到达 = 上一到达 + 上一 gap，派生到达 `>= 30e9` 即截断该 session 后续
  （即"只有派生到达 < 30s 的 request 属于前 30 秒输入"）。
- 窗口内 turn 重编号: 纳入的 session 从 turn 0 连续编号；窗口外 turn 丢弃，
  不做到达时间归一化。
- 边界 arrival/gap 推导: `session_arrival_time_ns` = 源绝对时间；
  `inter_request_interval_ns` = 逐行 `human_time || tool_time` gap（空→0）；
  两者均须 1000 ns 整数倍（实测 0 违例）。
- 8 列转换: session_id,turn_index,request_id,prefill_length,decode_length,
  session_arrival_time_ns,inter_request_interval_ns,description。
- interval 口径选择: "human_time||tool_time 逐行 gap"（总体方案 §5.7 合同⑨
  二选一之一），已登记于 traces/PROVENANCE.md。
- 实测: 1177 请求 / 112 session（face 仓 2026-08-16 重派生逐字节一致；
  与蓝本 wscllm 同源同规则同值）。

## 裁决

### 未来参数裁决表（face 方案 §12 冻结版）

| 参数 | 位置（【亲核】2026-08-16） | 裁决 | 在线实现要求 |
|---|---|---|---|
| session_lru `p_chunk` | trace_config.csv:15（=512，经 generate_face_trace.py build_face_plan 传入，仅 session_lru 分支） | 固定实验配置 | 在线 runner 显式传同一配置值，不改动 |
| legacy `p_chunk` | face_scheduler.py:1021（全量 mean，`ceil(mean(prefill_length))`） | **目标 workload 标定常数**（用户裁决 2026-08-15 框架）：物化时从冻结输入按离线同一推导预先导出，离线/在线共用同一常数；在线用固定值不构成策略变化 | 禁止在线运行时从已到达请求算增量 mean |
| `max_d_token` | face_scheduler.py:1316（lru）/ :1023（legacy） | **标定常数（face 独有）**：LUT 惰性 token bin 上界（`_power_of_two_token_bins(max_d_token)` :299-313），影响 lookup 匹配结果 → 影响 decode 代价 → 影响 decode 选择；取冻结输入的 `max(final_context_tokens)` 同一推导值，登记 provenance | 在线 LUT build 用同一常数；禁止在线增量扩展 token bin；超范围查询 fail-closed（KeyError :521-525 已有行为） |
| `request_count` | face_scheduler.py:1322（lru）/ :1029（legacy） | **标定常数（face 独有）**：LUT d_batch 上界检查（lookup :494-497 区段）；取冻结输入请求数（1177） | 在线 LUT build 用同一常数；禁止放宽上界 |
| LUT（`FaceLut` :374-530；`estimate_iteration_time_ns` :315-373） | 依赖上述三者 | **在线保留为静态代价函数（face 独有）**：估计公式与匹配规则逐行不动；离线时钟推进角色删除 | 在线只被 `select_decode_instance` 消费（:683-694 同款两次 lookup）；`start_ready_iterations` 计时 lookup（:1551-1559）与 legacy :1087 不迁移 |
| decode 选择快照（`has_prefill_work`/`decode_token_lengths`） | face_scheduler.py:1622-1627（构造）、:654-709（消费） | 在线账本口径（真实完成事件驱动）；prefill 完成边界才读取（时机不变） | 顺序敏感决策点：B2 oracle exact + strategy 不变量（方案 §5.4） |
| 同 tick 排序 | face_scheduler.py:1126（legacy）/ :1596（lru） | 保留：completion(0) < arrival(1)，同 priority 按 sequence | 在线事件索引队列同序 |
| 决策粒度口径 | — | 按 request/stage 对照 | B3/B4 不要求在线逐迭代构图 |

### face 版 LUT 关键裁决五条（2026-08-16 逐行复核记录，无反例）

核心判定：蓝本裁决"LUT 全部 offline-only"依据是其 LUT 只推进离线时钟；
**face 不同**——face 的 `select_decode_instance`（face_scheduler.py:654-709）
在 prefill 完成时刻对候选实例做两次 `lut.lookup`（:683-694）计算
`per_die_delta_ns` 并作为选择键（:706）——LUT 是 face decode 策略决策的
组成部分，不能离线化。裁决如下：

1. **LUT 的离线时钟推进角色删除**：离线 `start_ready_iterations`
   （:1517-1590）用 `lut.lookup`（:1551-1559）估计 iteration 时长并推
   `iteration_complete` 事件；在线由 C++ 真实完成事件推进，该角色不迁移
   （legacy 分支 :1087 的计时 lookup 同理不迁移）。
2. **LUT 的策略代价角色在线原样保留**：`estimate_iteration_time_ns`
   （:315-373）是纯函数（hardware/model/instance_size/p_chunk/d_batch/
   d_token → ns），不依赖 request 集合；`FaceLut` 惰性条目生成只依赖 build
   时固定的 `_lazy_p_chunk`/`_lazy_request_count`/`_lazy_token_bins`。在线
   按与离线同一 build 参数（标定常数）构造 LUT，lookup 结果与离线逐值
   相同——不构成策略变化（用户裁决 2026-08-15 框架）。
3. **max_d_token 与 request_count 以标定常数处理**：物化时从冻结 30s 输入
   按离线同一推导导出（`max(final_context_tokens)`；请求数 1177），登记
   provenance；在线任何超范围查询 fail-closed，禁止在线扩展 bin 或放宽
   上界——扩展会改变匹配结果，等同策略变化。
4. **decode 选择的全局快照输入**：`has_prefill_work`（各实例 qp 非空）与
   `decode_token_lengths`（各实例 active_decode 的 current_decode_token
   列表）按离线 :1622-1627 同一构造；在线值由真实完成事件驱动的账本提供，
   prefill 完成边界才读取（时机不变）。顺序敏感决策点，B2 双层验收
   （oracle exact + strategy 不变量），不得逐值强等与放宽比较器并举。
5. **逐行复核证据（2026-08-16 grep 实测）**：`face_scheduler.py` 全部
   `lut.lookup` 调用点 = :683/:689（select_decode_instance decode 代价）、
   :1087（legacy 计时）、:1551（lru 计时）；`FaceLut.build` = :1024（legacy）/
   :1317（lru）。除上述外无任何策略路径读取 LUT 或全量参数；
   `try_admit_prefill`（:1382-1453）/`try_admit_waiting_decodes`
   （:1455-1516）/`select_prefill_instance`（:569-572）/KV 准入链
   （SessionKVCacheManager）均不读 LUT。**无反例。**

## 验证方法

- B0: 窗口派生与 sidecar digest 一致性（见合同⑧）。
- 阶段 1-9: 在线策略路径以离线 `_plan_face_session_lru_recompute`
  （:1294-1806）为蓝本逐行对应迁移，删除 LUT 计时部分、保留排队/配对
  语义与 decode 代价 lookup；每处迁移注释标注离线对应行号。
- 阶段 2: B2 oracle 模式 decode_candidates 全记录逐值对照（含 LUT 值与
  tie-break）；KV action 与 kv_cache_events.csv 对照。
- 阶段 3: 对账与差异归因。
