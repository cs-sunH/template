# 合同⑦ Tier B 合同(B0-B4 分层验收)

冻结时间: 2026-08-16(阶段 0,步骤 0-4);审查通过是阶段 1 开工硬门槛。
全部验收仅使用 20.csv 前30s 输入(用户指示 2026-08-15,方案 §0.3)。

## 口径
验收模式拆分(估算器时间与真实 C++ completion tick 不无条件强行相等):
- **oracle/replay 模式**:回放阶段 0 decision_log(离线决策时序原样冻结),
  B1(含 tick)、B2、B3、B4 的 exact 只在此模式成立。
- **real-online 模式**(strategy 关感知):C++ 真实完成事件推进,验收策略
  不变量、账本正确性与差异可解释性;不要求与离线决策序列 exact。

分层验收:
| 层 | 比较内容 | 判定 |
|---|---|---|
| B0 输入等价 | request/session 数量、8 列转换、arrival/gap、窗口 turn、prefix 口径(recompute)、provenance | 逐字段一致;容差 0 |
| B1 生命周期/决策等价 | oracle:arrival/retry 顺序与 tick(含 pending_admissions 挂起/重试逐条对齐——本仓特有核对点)、IDLE/ACTIVE、decision/milestone 顺序及 tick。real-online:只验生命周期不变量 | oracle 容差 0;real-online 不变量成立 |
| B2 策略等价 | oracle:prefill+decode 实例赋值序列(decode candidates)、KV action 顺序/数量/状态转换(对照 manifest requests[] 段与 KV 事件载荷)、阶段推进序列。real-online:策略不变量(排队深度快照、HBM 可行集、LUT 标定表值与离线同一函数同输入同输出) | oracle 容差 0;real-online 不变量全过 |
| B3 图结构等价 | GraphBatch 节点集合/关键属性(compute 时长/通信 bytes/MEM tensor_size)/父依赖/rank ownership vs 离线 .et 对应段落(request/stage 粒度) | 规范化图一致;容差 0 |
| B4 执行等价 | oracle/replay C++ 执行 tick vs 静态 ET metrics tick——同图前提在 replay 不满足(已知差异),操作性标准 = 差异全部可归因并记录 | 差异全部有合同内解释 |

## 裁决
- 任何层级失败不得通过放宽比较器掩盖。
- 比较基准 = canonical logical node key(与插入顺序无关);静态 writer 按
  `order_plans_for_static_emission`(:801)一次预写,在线三段式交错提交,
  节点 ID/插入顺序不作比较项。
- 决策粒度:按 request/stage 对照(离线记录逐 chunk 迭代,ET 构图按 request 整段)。
- replay 装置 LUT 时钟语义(**本仓四项,前三项照搬蓝本裁决 4,全部仅限
  `--online-mode replay`;strategy/静态保持真实物理**):
  ① decode/prefill COMP 按各自 LUT 校准窗口并发执行(放行单槽位门);
  ② comm 节点即时完成(**RECV 侧**,五仓红线:不得再尝试 SEND 侧豁免);
  ③ 段 1 发射清除 prefill 组各 rank previous_id(保留 within-request 串行化
     + 段间 own-block-end 恢复 + 同 session interval gate);
  ④ **本仓新增(sh_1.0 独有)**:MEM_LOAD/MEM_STORE 节点即时完成——离线
     planner 的 LUT 时钟不含远端存取时间;若致依赖失步按蓝本裁决 8/9 方法
     处理并登记归因类别,不得反复试机制变更。
- B3 预期差异类别(执行时按实测定案):① replay LUT 时钟口径(无网络/远端
  时间、并发 COMP、MEM 即时);② 跨 request previous_id 串行化边差异;
  ③ tick-end/deferred 顺序合同;④ 段间 block-end 恢复 None-restore 集合
  (三段式发射预期高风险点:段 2 迁移节点对段 1 末节点依赖、段 3 remote_store
  触发节点对 decode 段末 barrier 依赖,generate_face_trace.py:2156-2167)。

## 验证方法
- 比较器 `online/verify/tier_b_compare.py`;基线材料 = 阶段 0 归档
  (raw_metrics.csv + run_logs + manifest + face_lut.csv)+ decision_log.jsonl;
  在线侧 = online_decision_log.jsonl + 在线 metrics + graph_batch_digests.jsonl。
- 等价报告 `online/verify/tier_b_report_20.md`。
