# 合同⑦ Tier B 合同(B0-B4 分层验收)——sh_3.0

冻结时间: 2026-08-16(阶段 0,步骤 0-4);审查通过是阶段 1 开工硬门槛。
全部验收仅使用 20.csv 前30s 输入(用户指示 2026-08-15,方案 §0.3)。

## 口径

验收模式拆分(总体方案 §5.7 第 7 条:估算器时间与真实 C++ completion tick
**不无条件强行相等**):

- **oracle/replay 模式**:回放阶段 0 的 decision_log(离线决策时序原样
  冻结),验证驱动改造本身的等价性——B1(含 tick)、B2、B3 的 exact 只在
  此模式成立;B4 走归因路径(见下)。
- **real-online 模式**(strategy 关感知):由 C++ 真实完成事件推进,验收
  对象是策略不变量(每次决策的输入——HBM 可行掩码、edge_free 掩码、逐实例
  `InstanceTaskLoadSnapshot` 三分量、KV 快照三态/驻留实例——与离线同一
  函数在相同输入下的输出一致)、账本正确性与差异可解释性;不得要求与
  离线决策序列 exact。

分层验收:

| 层 | 比较内容 | 判定 |
|---|---|---|
| B0 输入等价 | request/session 数量、8 列转换结果、arrival/gap、窗口 turn、prefix 口径(**本仓 sidecar_restore**:queue prefill_length=新增 token + context sidecar prefix_tokens/input_tokens_total + canonical sidecar 四字段,三者 digest 互校;manifest 逐请求 `source_prefix_tokens`/`source_input_tokens_total` 与 canonical 一致;`input_tokens_total == prefix_tokens + prefill_length` 全量成立;prefix 只计一次) | 逐字段一致;容差 0 |
| B1 生命周期/决策等价 | oracle:arrival/retry 顺序与 tick、IDLE/ACTIVE、输入关闭/排空、decision 与 milestone 顺序及 tick(以离线 decision_log 为准);尚未到达的 request 不得提前决策。real-online:只验生命周期不变量 | oracle 容差 0;real-online 不变量成立 |
| B2 策略等价 | oracle:request→instance 赋值序列(prefill 实例 + `prefill_affinity_reason`;decode 实例恒等于 prefill 实例)、KV action 顺序/数量/状态转换(与 metrics manifest 的 KV 事件载荷 `kv_event_payload_sh2` 对照,对照材料以阶段 0 归档为准)、prefill/decode 阶段推进序列。real-online:策略不变量逐决策断言 | oracle 容差 0;real-online 不变量全过;oracle 下 strategy 与 replay 必须相同决策序列 |
| B3 图结构等价 | 每个 GraphBatch 节点集合、关键属性(compute num_ops/tensor_size、通信 bytes/src/dst/tag、MEM 节点 `is_local_hbm_kv_restore`)、父依赖、rank ownership vs 离线 .et 对应段落(request/stage 粒度);比较基准 = canonical logical node key | 规范化图一致;容差 0(父依赖按归因类别豁免,见裁决) |
| B4 执行等价 | oracle/replay:C++ 执行 tick vs 静态 ET 运行 metrics tick——只有同图、同后端、同资源参数、同到达序列且同提交序列时要求完全一致;"同图"前提在 replay 模式不可满足,exact 条款不适用 | **操作性标准 = 差异全部可归因并记录**;归因类别①-④见裁决 |

## 裁决

- 任何层级失败不得通过放宽比较器掩盖;比较器不得以 ID/插入顺序为比较项。
- 决策粒度:按 request/stage 对照(离线决策记录是逐 chunk 迭代
  FaceIterationRecord,离线 ET 构图 request_aggregated)。
- **replay 装置 LUT 时钟口径(蓝本裁决 4 三项 + 本仓第 4 项,全部仅限
  `--online-mode replay`;strategy 保持真实物理;静态路径不动)**:
  ① 并发校准 COMP 放行单槽位门(count 式);② comm 节点即时完成
  (**边界 = RECV 侧,五仓红线**,SEND 侧豁免 desync 复现即红线);
  ③ prefill 发射清除 prefill 组各 rank previous_id;④ **本仓新增:远端
  MEM_LOAD/MEM_STORE 与本地 HBM restore 节点即时完成**——离线 planner 的
  LUT 时钟不含任何传输时长(KV 传输决策时即时生效),replay 逐请求对齐
  记录 tick 必须豁免这些节点的物理时长;豁免只改完成时刻,依赖/watch/
  terminal/释放链完整保留;HBM 模型 50/50 竞争在 replay 模式随豁免旁路
  ("replay 模式不向 HBM 模型注册 job,直接 1ns register_event")。
- **B3 父依赖归因类别**:within-request 父依赖差异若两轮决定性验证仍同点
  失步,回退并登记为类别④(decode 组 rank 链 None-restore,replay 装置
  LUT 时钟语义);B3 最终判定 = count/name/type/attr 容差 0 + 父依赖除
  类别④集合外逐边一致(④ 集合必须有完整证据)。本仓额外候选差异源
  (登记用):partial 流水恢复的 chain_checkpoint/restore_chain 分支边、
  trigger gate 依赖边、interval gate 边。
- **B4 归因类别闭集**:① replay 装置 LUT 时钟口径(无网络/远端/HBM 传输
  时间、并发校准 COMP、RECV 侧 comm-0、远端/HBM 即时完成);② 跨 request
  previous_id 串行化边差异;③ tick-end/deferred 顺序合同;④ decode 组
  rank 链 None-restore(若登记)。任一差异无法归入已登记类别,阶段 2
  不通过。

## 验证方法

- 比较器 `online/verify/tier_b_compare.py`(蓝本同名参照,按本仓产物字段
  适配):基线侧 = 阶段 0 归档静态 ET 运行(raw_metrics.csv + run_logs +
  manifest + metrics_manifest)+ decision_log.jsonl;在线侧 =
  online_decision_log.jsonl + 在线 metrics + graph_batch_digests.jsonl。
- 等价报告 `online/verify/tier_b_report_20.md` 提交 git。
