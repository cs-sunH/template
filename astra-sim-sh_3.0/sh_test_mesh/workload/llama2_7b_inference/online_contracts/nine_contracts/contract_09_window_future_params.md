# 合同⑨ 窗口与未来参数合同(含 LUT 关键裁决复核记录)——sh_3.0

冻结时间: 2026-08-16(阶段 0,步骤 0-4);审查通过是阶段 1 开工硬门槛。

## 口径(窗口派生口径,与 traces/PROVENANCE.md 一致)

- 窗口:session 首行 `arrival_time < 30,000,000,000 ns` 才纳入;turn 派生
  到达 = 上一到达 + 上一 gap,派生到达 `>= 30e9` 即截断该 session 后续。
- 窗口内 turn 重编号:纳入的 session 从 turn 0 连续编号;不做到达归一化。
- 边界 arrival/gap 推导:`session_arrival_time_ns` = 源绝对时间;
  `inter_request_interval_ns` = 逐行 `human_time || tool_time` gap(空→0);
  两者均须 1000 ns 整数倍(实测 0 违例)。
- 8 列转换 + sidecar_restore 三件套:见合同⑧。
- interval 口径选择:"human_time||tool_time 逐行 gap"。
- 实测:1177 请求 / 112 session(与 wscllm 蓝本 30s 物化同数)。

## 裁决

### 未来参数裁决表(方案 §12 冻结版)

| 参数 | 位置(【亲核】) | 裁决 | 在线实现要求 |
|---|---|---|---|
| `p_chunk` | `face_scheduler.py:22`(`PREFILL_CHUNK_SIZE=512`;消费点 :3527/:3625/:3842-3846) | 固定实验配置 | 在线 runner 显式传同一配置值,不改动 |
| `average_decode_length` | 导出 `generate_face_trace.py:788-790`(全量均值)→传入 `face_scheduler.py:3530-3533`;消费 `estimate_decode_remaining_task_load_ns` :648-702(经 `task_load_snapshot` :3672-3684) | **目标 workload 标定常数**(用户裁决 2026-08-15 模式):物化期从冻结输入按离线同一推导预先导出(= 459.4944774851317,登记 traces/PROVENANCE.md),离线/在线共用;在线使用固定值不构成策略变化 | **禁止在线运行时从已到达请求算增量 mean**;在线侧为模块常量/显式配置注入 |
| `max_d_token` | `face_scheduler.py:3543` | offline-only 产物(LUT 惰性 token bin 上界) | 在线热路径不消费 |
| `request_count` | `face_scheduler.py:3547` | offline-only 产物(LUT d_batch 上界) | 在线热路径不消费 |
| LUT(`FaceLut` :704-812) | 依赖上述两者;唯一计时消费点 `start_ready_iterations` :3853-3858 | offline-only 产物 | 在线迭代推进由 C++ 真实完成事件驱动,不读 LUT;离线 LUT 产物保留供对照 |
| Roofline 估算函数组 | `estimate_iteration_time_ns` :550-618、`estimate_prefill_task_load_ns` :620-647、`estimate_decode_remaining_task_load_ns` :648-702 | 在线保留(策略判据本身,红线 #2);除标定常数外无全量依赖 | 输入状态来自在线账本(合同⑥口径),函数逐行复用 |
| `order_plans_for_static_emission` | `generate_face_trace.py:983-1106` | offline-only(构图层全局 Kahn 排序) | 在线按事件触发顺序提交 GraphBatch;B3 用 canonical key 比较,与该顺序无关 |
| 同 tick 排序 | `face_scheduler.py:3903`((priority, sequence));completion 0(:3885-3890)< arrival 1(:3579/:4006) | 保留 | 在线事件索引队列同序 |
| 决策粒度口径 | — | 按 request/stage 对照 | B3/B4 不要求在线逐迭代构图 |
| `remaining_iteration_fraction` / `current_decode_token` 进度 | `face_scheduler.py:3602-3610`、:3665-3684 | **执行事实驱动化**(本仓特有裁决,非未来参数):planner LUT 模拟进度 → 在线账本真实进度;折算公式见合同⑥ | 禁止离线 LUT 时钟渗入在线策略输入 |

### LUT 关键裁决复核记录(执行者 2026-08-16 逐行复核,无反例)

核心裁决:sh_3.0 策略决策输入 = `InstanceTaskLoadSnapshot` 三分量
(Roofline 估计服务时间,`ordering_key` :851-857)+ HBM 可行掩码
(`request_hbm_feasible_instances`)+ edge_free 掩码(:466-483)+ KV 快照
(三态/驻留实例);LUT 只用于离线事件循环 `start_ready_iterations` 推进
模拟时钟,不参与三段式准入/sticky/decode 同实例/KV 决策。

复核证据(grep + 逐行,行号以 2026-08-16 工作树为准):

1. LUT 全部消费点:`grep -n "lut.lookup\|lut\." face_scheduler.py` →
   `:3853`(start_ready_iterations 计时,唯一规划路径调用)、`:1011/:1017`
   (`select_decode_instance` 内——红线 #5"保留不调用"设施,规划路径无调用,
   测试仍直接引用)、`:3544`(FaceLut.build 构造)。策略路径无 LUT 读取。
2. `try_admit_request`(:3697-3823):读取面 = kv_manager
   (request_hbm_feasible_instances :3701-3718 / session_ids :3722-3727 /
   reserve_request_capacity :3784)、`task_load_snapshot` :3739-3742/:3777、
   edge_free_mask :3739、select_prefill_instance、prepare_prefill——无 LUT。
3. `task_load_snapshot`(:3640-3695):三分量 = queued(纯账本 :3612-3638)、
   running(remaining_iteration_fraction :3602-3610 的 LUT 模拟进度折算——
   **进度输入而非 LUT 查表**,在线按合同⑥改真实进度)、active
   (estimate_decode_remaining_task_load_ns + 标定常数)。
4. `start_ready_iterations`(:3833-3890)::3853-3858 的 `lut.lookup` 仅用于
   `iteration_complete` 事件时钟推进;:3885-3890 推 priority 0 事件。

结论:**无反例**(策略消费的全量派生量只有 `average_decode_length` 一项,
且按用户裁决以固定常数形态在线使用)。

## 验证方法

- B0:三件套 digest 一致性(合同⑧)。
- 阶段 1 步骤 1-9:在线策略路径以离线 `plan_face_requests` 为蓝本逐行
  对应迁移(注释标注离线行号),删除 LUT 计时部分(:3853-3858/:3885-3890)、
  保留排队/配对语义。
- 阶段 2:B2 oracle 模式 KV action 与 metrics manifest KV 事件载荷对照。
