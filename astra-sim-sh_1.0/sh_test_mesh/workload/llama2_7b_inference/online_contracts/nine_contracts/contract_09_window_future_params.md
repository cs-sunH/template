# 合同⑨ 窗口与未来参数合同(含 LUT 关键裁决)

冻结时间: 2026-08-16(阶段 0,步骤 0-4);审查通过是阶段 1 开工硬门槛。

## 口径(窗口派生,与 traces/PROVENANCE.md 一致)
- 窗口:session 首行 arrival_time < 30e9 ns 才纳入;turn 派生到达 = 上一到达
  + 上一 gap,≥ 30e9 即截断。窗口内 turn 从 0 连续编号;不做到达归一化。
- interval 口径 = 逐行 `human_time || tool_time` gap(空→0),1000 ns 整数倍。
- 8 列转换列序固定。实测 1177 请求 / 112 session(与蓝本一致)。

## 裁决(未来参数裁决表,方案 §12 冻结版)

| 参数 | 位置【亲核】 | 裁决 | 在线实现要求 |
|---|---|---|---|
| `p_chunk` | trace_config.csv:13(=512);build_face_plan :769 恒传 | 固定实验配置 | 在线 runner 显式传同一配置值;**全量 mean 回退(face_scheduler.py:2784-2787)在线禁入**,缺失 fail-closed |
| `max_d_token` | face_scheduler.py:2789(全量 max) | **物化期标定常数**(总方案 §5.2 用户裁决 2026-08-15) | 物化阶段按离线同一推导导出并登记 provenance;在线不做全量 max |
| `request_count` | face_scheduler.py:2795(LUT d_batch 上界) | **物化期标定常数** | 同上 |
| LUT 计时角色 | `estimate_iteration_time_ns` :488-546;lookup 于 start_ready_iterations :2930-2935 | offline-only 产物 | 在线由 C++ 真实完成事件推进,不读 LUT 计时 |
| LUT 成本模型角色 | lookup 于 select_decode_instance :810-821 | **在线保留,物化期冻结表**(sh_1.0 独有裁决,与蓝本"LUT 全部 offline-only"不同):物化期 FaceLut.build(:571-610) 按冻结输入同一推导构建,export_csv(:629) 落盘 face_lut.csv;在线只 lookup(:613-628) | 在线调度器构造时加载冻结 LUT(face_lut.csv);禁止在线 build/增量扩展 token bin(增量扩展会中途改变 decode 决策=策略变化);加载失败 fail-closed |
| per-session 上下文累积 | `_validate_and_expand_requests` :2716-2757 | 在线按到达增量累积(turn k 只依赖已到达 turn 0..k-1) | 每 session 维护增量累积状态机;turn 连续性逐条校验 |
| FacePlan.iterations 记录 | record_iterations=False(生产,:768) | offline-only 产物 | 在线不消费;replay 决策日志走 --replay-record 独立通道 |
| 同 tick 排序 | face_scheduler.py:2970((priority, sequence);completion 0 < arrival 1) | 保留 | 在线事件索引队列同序 |
| 决策粒度 | — | 按 request/stage 对照 | B3/B4 不要求在线逐迭代构图 |
| edge_ranks / reserve_context_tokens | trace_config.csv:16;:407-421;KVCacheManager.__init__ :1075-1178 | 固定实验配置 | 在线从同一 config 读取,不改动 |

### 复核纪律(阶段 0 已做)
策略决策读取的全量集合依赖仅上表四处(p_chunk 回退 / max_d_token /
request_count / LUT);LUT 在 select_decode_instance 内读取仅 :810-821 两处
lookup;start_ready_iterations 内 LUT 仅用于计时(:2930-2936)。若未来发现
反例,按红线立即停止上报。

## 验证方法
- B0:窗口派生与 sidecar digest 一致;阶段 2 B2 oracle 比对 decode 实例/
  代价一致(核对冻结 LUT 与离线同表同值,face_lut.csv digest 入 provenance)。
