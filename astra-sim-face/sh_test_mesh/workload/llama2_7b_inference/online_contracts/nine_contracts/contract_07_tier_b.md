# 合同⑦ Tier B 合同(B0-B4 分层验收)

冻结时间: 2026-08-15(阶段 0,步骤 0-4);审查通过是阶段 1 开工硬门槛。
全部验收仅使用 20.csv 前30s 输入(用户指示 2026-08-15,方案 §0.3)。

## 口径

先定验收模式拆分(总体方案 §5.7 第 7 条原文:估算器时间与真实 C++
completion tick **不无条件强行相等**):

- **oracle/replay 模式**:回放阶段 0 的 decision_log(离线决策时序原样
  冻结),验证驱动改造本身的等价性——B1(含 tick)、B2、B3、B4 的 exact
  只在此模式成立。
- **real-online 模式**(strategy 关感知):由 C++ 真实完成事件推进,验收
  对象是策略不变量(每次决策的输入与离线同一函数输出一致)、账本正确性
  与差异可解释性;不得要求与离线决策序列 exact;只有先证明两者拥有同一
  事件与提交序列时才允许声明 B1/B2 exact。

分层验收:

| 层 | 比较内容 | 判定 |
|---|---|---|
| B0 输入等价 | request/session 数量、8 列转换结果、arrival/gap、窗口 turn、prefix 口径(prefix_mode=recompute,turn-0 prefix 折入 prefill)、输入 provenance | 逐字段一致;容差 0 |
| B1 生命周期/决策等价 | oracle:arrival/retry 顺序与 tick、IDLE/ACTIVE 转换、输入关闭/排空、decision 与 milestone 顺序及 tick(以离线 decision_log 为准);尚未到达的 request 不得提前决策。real-online:只验生命周期不变量 | oracle 容差 0;real-online 不变量成立 |
| B2 策略等价 | oracle:request→instance 赋值序列(prefill 实例 + decode 静态路由)、KV action 顺序/数量/状态转换(与 kv_cache_events.csv 对照)、prefill/decode 阶段推进序列。real-online:策略不变量(排队深度、KV 容量、路由表与离线同一函数同输入同输出) | oracle 容差 0;real-online 不变量全过;oracle 下 strategy 与 replay 必须相同决策序列 |
| B3 图结构等价 | 每个 GraphBatch 节点集合、关键属性(compute 时长/通信字节等)、父依赖、rank ownership vs 离线 .et 对应段落(按 request/stage 粒度) | 规范化图全部一致;容差 0 |
| B4 执行等价 | oracle/replay:C++ 执行 tick(node issue/complete)vs 静态 ET 运行 metrics tick——只有同图、同后端、同资源参数、同到达序列且同提交序列时要求完全一致;离线 planner 内部 LUT 估计时间戳不参与 B4;real-online 不参与 exact,差异归入阶段 3 可解释性报告 | oracle tick 一致;差异必须有合同内解释并显式记录 |

## 裁决

- 任何层级失败不得通过放宽比较器掩盖(仿真加速分析.md §12.5)。
- 比较基准 = 与插入顺序无关的 canonical logical node key(request/stage/层/
  类型/rank 等语义键 + 规范化哈希);节点 ID 与插入顺序天然可能不同,
  不作为比较项(静态 writer 按估计 prefill_start 排序一次预写完整 request
  图 :1100,在线按真实阶段完成交错提交)。
- 决策粒度口径:按 **request/stage** 对照(离线决策记录是逐 chunk 迭代
  WscLlmIterationRecord,离线 ET 构图按 request 整段;B3/B4 按 request/stage
  对照,不要求在线逐迭代构图——仿真加速分析.md §12.5)。
- 主控裁决 2026-08-15(步骤 1-8,三项 replay 作用域机制变更,全部仅限
  `--online-mode replay`;strategy 模式保持真实物理;静态路径不动):
  - ① 并发校准 COMP 放行单槽位门(计数式):LUT 时钟把同实例 decode 建
    模为批量处理器(同 tick 并发),物理单槽位串行化与 decision_log 完成
    序不一致;replay 模式校准 COMP 链并发执行,未校准路径仍走资源门。
  - ② comm 0 时长:离线 LUT 时钟不含网络时间,replay 时钟保留真实网络
    时长会破坏顺序;comm 按 1ns 即时完成,依赖/watch/terminal/释放链
    完整保留。
  - ③ prefill 发射时清除 prefill 组各 rank previous_id:per-rank 物理跨
    request 串行化边在 LUT 时钟下无对应物(离线 .et 同类边存在而 LUT
    不串行),replay 权威 = decision_log,故清除;保留 within-request
    串行化、decode 段 own-prefill-end 恢复、同 session interval gate。
  - **归因口径(取代初版"跨 request 边不可消除、保留"表述)**:
    replay 模式跨 request previous_id 边数 = 0 为**刻意差异**(LUT 时钟
    语义),归入 **B4 归因类别①**(刻意、有合同记录的差异);strategy
    模式保持物理跨 request 链。

## 补记(阶段 2,主控裁决 2026-08-15 round 2)——SEND 侧豁免

- 裁决②"comm 0 时长"的豁免边界经两轮决定性验证确认为 **RECV 侧**:
  - round 1(方案 A 精确补丁 emitted-ranks-only):replay runner 在
    delivery 55 / tick=5057359000 抛 ReplayDesyncError。
  - round 2(主控裁决批准备选:COMM_SEND 即时完成 + emitted-ranks-only
    重验):HardwareResource NodeView 三方法 CommSend 豁免(与 CommRecv
    同款,replay 作用域,strategy/静态不动)+ Workload.cc 接线,补丁
    重应用后**同一失步点复现**(delivery 55 / tick=5057359000 /
    session_14_request_0)。
  - 机制结论:send 本已经 issue_comm 的 replay_clock_ 1ns 即时路径完成,
    CommSend 豁免只改变 slot 记账,不改变 send 的完成时间与 issue 门控
    (其依赖链含上一 decode 的 END BARRIER——stale cross-request 边,
    离线 .et 结构真实但与 LUT 时间线不一致:PD 分离实例并发 decode 时
    PREFILL_DRAIN 触发仍执行)。恢复 own-request 边(无论 send 是否即时)
    必失步——两轮同点复现即决定性证据。
  - **终态协议**:五仓通用——comm 0 时长豁免 = RECV 侧;decode 组 rank
    链 None-restore(5abaf23)为 LUT 时钟稳定性必要机制,不得再尝试
    SEND 侧豁免(desync 复现即红线)。1495 条 within-request 差异登记
    B4 归因类别④。详细两轮验证记录见 CHANGE_STATUS/phase2_status.md。

## 验证方法

- 比较器 `online/verify/tier_b_compare.py`(阶段 2 新写):逐层比较输出差异
  报告;基线侧材料 = 阶段 0 归档静态 ET 运行(raw_metrics.csv + run_logs +
  manifest + kv_cache_events.csv)+ 阶段 0 decision_log.jsonl;在线侧 =
  online_decision_log.jsonl + 在线 metrics + graph_batch_digests.jsonl。
- 产出等价报告 `online/verify/tier_b_report_20.md` 提交 git。
