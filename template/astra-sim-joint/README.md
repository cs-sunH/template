# astra-sim-joint - 三机制联合策略仓（T＋J＋E 消融统一实现）


## 2026-10-12 v3 加速轮交付（当前最新）

两主线源码交付与验收详见根《joint加速改造结果v3.md》与仓外证据根 `/home/sunhao/joint_accel_v3_20261012.k7f3q`；PROVENANCE.md 2026-10-12 节为权威摘要。要点：**F 线**（当前物理 anchor 候选集合：NodeStore free 集 role 分区 + RRE per-rank 有序当前集合 + Workload 三路归并，A→B→C 同轮/已走过/新 ready 三语义经真实全栈场景钉死）已上生产——完整 first-1s 实测 anchor 查询 49.94M→1.66M（约 30×）、坐标拒绝 48.48M→0、候选枚举 50.09M→1.84M（约 27×）；**L 线**融合流式 apply 语义逐位等价但真帧 0.87×，按方案 §4.3 否决回退（保留共享重构与事务 count+last，0.996× 平价）；账本 apply 生产流程仍是 `validate_envelope`+`prepare_envelope` 两段式（单一 decoder 路径）。诊断计数语义变化与无开关增删见 PROVENANCE 同节。21/21 CTest、微型闭环业务字段对账 V4 证据、canonical 回归全绿；完整 first-1s 仍未自然完成（180s guard / 32 deliveries / RSS 2.96GB），`speed_goal_met=false`。

## 2026-10-11 target-slot V4 语义链路验收完成（历史段，已被上文 v3 加速轮取代）

strict V4 target-slot delta 协议（schema/protocol 4、`bridge-v4-20261011`、`workledger-v4-20261011`、snapshot handle `work_ledger_v4`）已通过完整最小语义链路验收并保护回写 live：真实 P2/D3 磁带五帧与 V3 参照物理段精确相等（仅 3 处核实的 binary identity 差异）；strict V3 归档与 strict V4 双 harness 对全部 5 个前缀重建的公开双账本状态逐位相等（watermark 1032、版本、tick、槽行）；R_current raw-V2 规范前缀对照、T2 同状态 3 边界×8 候选、坏输入原子性与新进程 journal 往返全部 PASS。真实 Online 微型闭环自然完成（5 delivery/5 ACK/1 arrived/1 completed、36 fill+12 release 物理 ACK、232 parent 边、328 流、settle tick 1221064、watermark 1144），生产 journal 全部 handle 为 `work_ledger_v4` 且 delivery/base/result/token 链连续，新进程 strict-4 重放对账 V3 同工作量参照。补齐三个坏输入缺口实证：data_recv 就绪证明缺失时 `cleared=true` 拒绝（"cleared before its ready/terminal proof"）、reclaim 后旧 owner 回调拒绝且新槽不动、非零 delivery/base/二次 full 均显式拒绝（相同 token 相同内容的重复 full 为规范幂等重放）。21 项注册 CTest 与 RemoteReadExecutor/GraphBatchCommitter/ParsedBatch/WorkLedgerBridge/BridgeLoopback 直跑全 PASS；344-tile 历史 descriptor 类型校验与 D20 当前构造 PASS。最终非 trace 回归 2316 PASS/3 预存已知失败（inf/nan ETA、open_restore journal mock）/2 skip，SLO unittest OK(1 skip)；本轮另修复 c7a/W8 计价零漂移比较面未剥离 `exact_reprice_wall_ns`/`first_projection.wall_ns` 宿主墙时字段的测试侧漂移（该两测试在 V3 live 上即失败，非 V4 回归）。V4 组件成本证据：生产者 S=T=8/32/128 的 claim 一次 O(S)/consume 线性 O(T)/initialize 近常数与真实字节，消费者固定 T 改 S、固定 S 改 T 两轴无乘法项。以上均为语义/复杂度/组件证据：完整 first-1s 性能验收仍未通过（见下文性能段），`speed_goal_met=false` 不因本段改变。

## 2026-10-11后续局部交付：ready成员证明与目标成员二分（历史段，已被上文 V4 验收取代）

以下4source修订晚于最近正式53-binding冻结；没有新完整first-1s计时，不能追认其T3倍率或T4进度。Workload仅在exact NodeStore source、成功snapshot、资格有效且非MAX的同版本内复用成员证明；保存强owner，源identity/资格/version任一变化即在本pass粘滞回退当前source的权威查询。lookup、anchor动态复核、物理/virtual排序与回调guard保持。相同最终probe object分别链接保留旧/new libAstraSim.a，N8/32/128各3次held-GPU实际扫描，候选24/96/384全部检查，membership queries24/96/384→0/0/0，ready vector/顺序/tick不变。probe的payload摘要仅是ID/kind/name FNV64及ID/name长度，不能称完整NodeView字节证明；完整物理行为由下面真实夹具另证。early issue后的suffix仍查询，不宣称全部60M queries消失或墙钟收益。

目标slot已验证sorted uint tuple成员查询改为C bisect_left，0/1成员直接路径，状态与wire保持。S8/32/128的旧独立list oracle比较36/528/8256、新实际输入索引读34/194/1026，指标分别标明；46相关方法0.378284s PASS，真实12prefix current-only回放4.296546s/峰474816512B，对复用的同hash旧参考所有state/canonical/末event replay精确相等，watermark103390、62404 typed events、11末树仍处理。

全target17.987516s/峰1006534656B PASS；首次21CTest5.083930s为20PASS/1新增N6fixtureFAIL，原先在issue前复制的replacement缺live-start统计。仅修test为真实terminal hook内复制已started/未completed的current source，生产统计校验与Workload优化未改；该target重编10.607706s，只有失败leaf重跑0.025475s PASS，不称首轮21全绿。预审也纠正了N5稳定query与MAX−1首次proof的旧计数预期，未伪造这些为运行失败。C++实际P2/D3微型producer0.116927s/当前双账本audit0.159753s，五帧与原physical/callback/certificate/wire完全同值，仅3处核实binary identity差异；watermark1032。旧/new纯count probe0.006649/0.006789s仅功能/工作量，不签发倍率。

目标成员数组的复杂度漏洞尚未关闭：相同合法direct-marker构造S8/32/128、T=S，真实target成员uint/geometry读取120/2016/32640，consumed累计成员读取56/992/16256。source touched-row稀疏不代表行内数组稀疏，P2当前为部分通过。正在准备专用target-slot声明/消费增量及exact predecessor/cause证明；不是新增通用checkpoint或更改模型。现有v3物理语义仍保持。累计仿真/投影1449.333684672s，364guard快照；上述离线/算法/build不计仿真。最新正式完整first1s仍是这些4source之前的180.016744s TIMEOUT/33deliveries，speed_goal_met=false、implementation_complete=false，不新增整段或旧长臂。

源码/实际检查与首次失败详见证据根logs/resume-20261010/remaining-ready-membership/validation-summary.json；不新增生产开关。--membership-cost-probe仅现有测试程序的构造计数入口，不进入生产runner。

> 本仓为 Execution-Driven 改造后的**裸仓库终态**（沿用基底两条在线仿真
> 路线；仓库 request-neutral，正式入口缺失输入 fail-closed）。由
> `template/astra-sim-sh_3.0` 工作树复制建立（基底快照：commit `4e9e2ca`
> + 工作树实际状态，含 2026-09-13"KV 逐出与推理计算并行执行"改造）。设计
> 规格：`paper_write/本论文撰写内部材料/三机制联合策略_template仓库设计
> 方案.md`（下称《设计方案》）；实验总纲：`paper_write/本论文撰写内部
> 材料/chatgpt建议实验思路.md`。
>
> 本仓是 ASTRA-sim 2.0 的晶圆级芯片（Wafer-Scale Chip, WSC）推理仿真
> 改造仓。七个同源仓共享同一套硬件架构建模与术语，仅在"请求→实例映射
> 与 KV 管理策略"上分化；本仓的分化点 = **T/J/E 三机制联合策略与八组合
> 消融能力**。硬件参数唯一权威来源仍是
> `sh_test_mesh/hardware/face_case5_config_c.json`（+ C18 D2D 带宽孪生
> 集，见 §7-I），运行时配置由 `sh_test_mesh/config_resolver.py` 派生，
> 禁止手改。

## 1. 本仓定位：同一套共同实现内的 T/J/E 独立开关

本仓承载三机制（《设计方案》§1）：

| 机制 | 开关 | on 语义 | off 共同替代（§7.1） |
| --- | --- | --- | --- |
| **T** 工具/人类区分 | `category_mode=typed` | SH 严格类别逐出：合法 victim 先 human 类后 tool 类，类内 `(last_completion_ns, session_id)` FIFO；human 层未耗尽且缺口未满足不得动 tool | `lru`：同一合法 victim 集合上的类型无关 LRU（单遍），其余流程不变 |
| **J** 联合决策 | `scheduler_mode=joint` | 全 instance 候选（无容量/边缘/距离掩码），发射点全候选联合 argmin 选 instance × stay/recompute/copy（2026-10-14 起三动作——第四动作 remote-read 已随远读机制整体退役，登记见仓根 `PROVENANCE.md` 退役批登记节）；**跨实例平局（最优 cost_ns 整数相等、并列实例 ≥ 2）自 v9（2026-10-20）起按实例编号全局环序轮转分摊**（`joint_scheduler.joint_tie_rotation_pick`；单实例平局/唯一最优不读不推游标、行为与旧固定序恒等——见 §3 J 行与 PROVENANCE 2026-10-20 节） | `load-first`：先按在线负载（ns 服务台账）在全部 instance 选点，再在该位置选 KV 动作；`affinity-first`（home 优先、缺省回退 load-first）为 J 边际比较第二参照，不进八组合；`face_static`（C17）= 静态距离对照臂（policy variant，不进八组合、不冒称完整 FACE，强制 quota-off——F7 唯一耦合例外） |
| **E** 自适应层数逐出 | `layer_policy=adaptive` | 按层消费期限公式计算最小热前缀 `k_hide` 软目标（§5.2），释放时目标外后缀→目标内后缀两次扫描 | `minimal_layer_groups`：同一合法对象排序下按逐 rank 缺口释放最少完整层组；`legacy_half` 为原半层逻辑回归对照，不是 E-off |

**remote-read（远读）机制已整体退役（2026-10-14）**：联合决策动作空间
由 stay/recompute/copy/remote-read 四动作收缩为 **stay/recompute/copy
三动作**，远读执行链（P 准入前缀读流、`rid#readplan` 承诺账本、远读暂存
lease、D 边界 decode credit 腿、C++ `RemoteReadExecutor`/
`TileTransportAdapter` 与 pipeline descriptor tile 流水协议）、W10 hybrid
动作空间扩展、增量型 P→D 边界布局族及 D09/D07 全部协议设施随之下线；
旧开关 `JOINT_REMOTE_ACTIONS`/`JOINT_REMOTE_READ_TILE_BYTES`/
`JOINT_REMOTE_READ_PARTIAL`（及更早的 `JOINT_REMOTE_CREDIT_ITERS`）并入
`_RETIRED_ENV_KEYS` fail-closed——任何显式值即启动失败。删除范围、版本
升版与被取代章节清单见仓根 `PROVENANCE.md` 顶部退役批登记节；历史验收
记录（§3 状态表日期化行、§2.9/§7.1 历史沿革注）按原文保留。

### 1.1 八组合固定映射（§7.1 硬性要求，2026-09-13 裁定）

`JOINT_ABLATION_COMBO` 一键预设（无 J 组合一律 `load-first`）：

| 组合 | category_mode | scheduler_mode | layer_policy |
| --- | --- | --- | --- |
| 无 T/J/E（"三无"基础组） | lru | load-first | minimal_layer_groups |
| T | typed | load-first | minimal_layer_groups |
| J | lru | joint | minimal_layer_groups |
| E | lru | load-first | adaptive |
| T＋J | typed | joint | minimal_layer_groups |
| T＋E | typed | load-first | adaptive |
| J＋E | lru | joint | adaptive |
| T＋J＋E（完整方法） | typed | joint | adaptive |

八组合共用同一套生命周期、KV 账本、资源计费、自动合并与事件模型；每个
开关只改变其指定分支（KV 物理布局、新 KV 写回/合并语义、硬件配置、初始
状态、在线估计与更新规则、动作计费在八组合间严格一致）。开关经环境变量
一次读取、fail-closed；预设与显式开关互斥（同值重复指定同样拒绝）：

```bash
# 八组合预设（任选其一）
JOINT_ABLATION_COMBO=TJE   # 或 none/T/J/E/TJ/TE/JE
# 或显式开关（与预设互斥）
JOINT_CATEGORY_MODE=typed|lru
JOINT_SCHEDULER_MODE=joint|load-first|affinity-first|face_static
JOINT_LAYER_POLICY=adaptive|legacy_half|minimal_layer_groups
JOINT_ALLOWED_ACTIONS=all|stay,recompute,copy 的逗号子集
                                     # 动作掩码（M2，缺省 all）——候选集
                                     # 上限不是强制：掩码外动作从适用集
                                     # 剔除；臂预设恒含 recompute；剔除
                                     # recompute 致零适用维持 fail-closed
# 与三机制/combo 正交的治理与遥测开关（C6–C11；缺省 off/未设）
JOINT_QUOTA_MODE=off|static|aimd     # 通用流量治理（aimd ⇒ 自动注入
                                      # --link-telemetry，F7 耦合）
SH_LINK_TELEMETRY=1                  # C++ 链路遥测旗标（显式直跑遥测臂）
```

补充披露（2026-10-01 审查轮补记，此前代码在读而 README 零披露的两个
治理键）：

* `SH_TRAIN_MAX_ITER`：列车步数上限 T_max（D16 有界性）。缺省 8；
  显式值必须为正整数字符串，≤0/非法词法 fail-closed 拒绝（唯一裁决点
  `joint.parse_train_max_iter`；`joint_runner.py` 白名单放行）。
* `SH_METRICS_DETAIL`：`--metrics-detail` 的 runner 侧档位注入
  （summary|full）；与 `JOINT_QUOTA_MODE=aimd` 存在耦合门——
  aimd ∧ `SH_METRICS_DETAIL=off` 组合 fail-closed 拒启（O11/L6：AIMD
  判读依赖 metrics 产物），经 `run_online_strategy.sh` 的 aimd 双守卫
  与 `joint_runner.py` 同构拦截。

其余代码读取的 `SH_*`/`ASTRA_*` 内部与调试档（`SH_ONLINE_VALIDATE`、
`BRIDGE_TIMEOUT_MS`、`SH_EDGE_AUDIT`、`SH_PROFILE_JSONL`、
`SH_PROPAGATING_TAIL_LIMIT`、`SH_STRICT_KV_INVARIANTS`、
`SH_FIRST_TOKEN_SPLIT`、`SH_ADMIT_GATE_VERIFY`、`SH_SNAPSHOT_VERIFY`、
`SH_SLO_HAVE_RM/SH_SLO_HAVE_TL/SH_SLO_SKIP_HBM/SH_SLO_SORT_CHUNK`、
`ASTRA_SIM_ONLINE_COMPLETED_NODES`、`ASTRA_LINK_OBSERVER`、
`SH_GRAPH_DIGESTS` 等）不属策略开关面：白名单运维变量以
`joint_runner.py` 的 `_ENV_ALLOWLIST` 披露为准，其余为内部行为档（多数
缺省关闭，逐键语义见各自读取点注释）。

开关入口裁定（2026-09-14 用户裁定，经 kimi 审查上呈）：**接受 JOINT_*
环境变量族作为策略开关唯一入口**（原执行方案 D7 的 joint_config.json
路径不实施；偏差登记见仓根 `PROVENANCE.md`）；**默认 = 完整三机制
TJE**（原执行方案 D9 的 affinity-first/legacy_half 回归对拍档经显式
env 获得，即 §8.3 #9 哨兵配置）。推荐经 `joint_runner.py` 起跑（D14）：

```bash
python3 sh_test_mesh/run_scripts/joint_runner.py <run_dir> <request_csv> \
    --combo TJE            # 或 --category/--scheduler/--layer 显式开关
    [--quota off|static|aimd]   # 通用流量治理（C11，缺省 off；aimd 自动
                                # 注入 --link-telemetry——F7 耦合）
```

runner 叠加：仓内仿真锁（`sh_test_mesh/runs/.single_simulation.lock`，
`joint_runner.py`、`run_online_strategy.sh`、sensing 演示入口、矩阵与容量压力夹具共享继承的
FD，先锁定再切换 trace/物化输入，嵌套入口复用同一锁）、二进制 sha256
前置校验、`<run_dir>/invocation.json`
（argv/开关/哈希/exit/UTC）、陈旧策略 env 清洗（SH30_*/SH3_CAUSAL_*/
SH_JOINT_* 全弹出；白名单运维变量取值披露）。

运行侧 provenance：每次运行把 `joint_config.manifest_dict()`（开关族 +
combo + off 替代语义披露 + 来源 env）写入 `<bridge_dir>/joint_mechanism_
manifest.json`；每次准入决策在 decision log 落 `joint_mode/joint_action/
joint_cost_ns/origin_home_instance/horizon_source` 与全部候选的成本表；
completion 行落 `merge_transfers`（merge v2 少并多的真实传输摘要）与
`merge_direction`（stay/forward/reverse/in_place）/`home_flipped_to`/
`merge_transferred_bytes`（合并方向与 home 迁移披露，2026-09-17）。
基底的 `SH30_ABLATION`（no_lb/no_affinity）已退役：**任何显式值（含
`none`）即启动失败**（fail-closed，防陈旧脚本静默假装旧消融生效）。

## 2. 策略架构（《设计方案》§7 模块边界）

```
sh_test_mesh/workload/llama2_7b_inference/
├── joint/                        # 三机制策略包（本仓新增）
│   ├── joint_config.py           #   开关解析/八组合预设/manifest
│   │                             #   （+ quota/face_static 耦合解析）
│   ├── eviction_priority.py      #   T：typed/lru victim 类别序（纯函数）
│   ├── layer_eviction_policy.py  #   E：k_hide 期限公式 + 三模式释放计划
│   │                             #      + 在线估计器（输入长度/EWMA）
│   ├── joint_cost_model.py       #   J：无 oracle (instance×action) 代价
│   │                             #      + 链路流登记表 + 因果时域估计
│   │                             #      + 遥测 divisor_effective/C7 三件
│   │                             #      （W3/v4-projection 起两入口内部 =
│   │                             #       事件前推，见下机制表 W3 行）
│   ├── action_work_graph.py      #   W3：三动作预测工作图构造器（P 参考
│   │                             #      接续/结算/控制延迟/继承义务）
│   │                             #      + WorldSource 背景世界源
│   ├── resource_projection.py    #   W2：公共资源事件前推内核（事件队列
│   │                             #      + 分段恒速积分 + N-way 等分 +
│   │                             #      buffer/barrier 门 + 守恒校验）
│   ├── runtime_snapshot.py       #   W1：因果一致性冻结快照（版本化资源
│   │                             #      索引 + 输入来源审计；白名单投影
│   │                             #      剥离 decode_length 等禁止字段）
│   ├── work_ledger.py            #   W1：因果运行账本（义务身份/生命周期/
│   │                             #      剩余量/依赖；C++ bridge 侧车对账
│   │                             #      + journal 前缀重放保真）
│   ├── kv_location_directory.py  #   W4：唯一权威 KV 区间目录（内容键 =
│   │                             #      session×层×token×rank；owner/
│   │                             #      replica 与逻辑 home 分立；在途/
│   │                             #      多副本/引用守恒事务）
│   ├── distributed_settlement.py #   W9：distributed 目录提交型结算纯逻辑
│   │                             #      层（留存段/来源面枚举/单元格级
│   │                             #      可定位审计/终局无悬挂/开销计量）
│   ├── history_estimator.py      #   W6：因果历史统计与未知工作量（条件
│   │                             #      剩余长度/删失/纯服务样本；先
│   │                             #      shadow，W5 起正式 n̂ 消费）
│   ├── background_forecast.py    #   W7：新增需求预测 → 背景工作模板
│   │                             #      确定性注入（互斥类别/时间桶衰减/
│   │                             #      忙时注入/闭环返回约束）
│   ├── forecast_validation.py    #   W7：预测成熟恰一次差分回填、按距离
│   │                             #      准入、固定回退链与开销账本
│   ├── prediction_config.py      #   W0：版本化预测配置单入口
│   │                             #      （JOINT_PREDICTION_CONFIG JSON
│   │                             #      preset；正交研究轴与来源登记）
│   ├── schema_registry.py        #   W0：各 schema/协议独立版本登记
│   │                             #      （模型语义/快照/bridge/目录/
│   │                             #      预测配置/日志——fail-closed）
│   ├── boundary_layout.py        #   D19：D 候选专用边界视图——区间布局
│   │                             #      （copy 只数目标缺少部分）
│   │                             #      + StageWork 阶段工作 + 等待接口
│   │                             #      拆分（原地接续/新接纳）
│   ├── capacity_summary.py       #   D11：分阶段足迹 PhaseFootprint + 容量
│   │                             #      驱逐摘要 CapacitySummary（θ 份额）
│   ├── predictor_cost_accounting.py  # ACC：预测器自身开销恒开账本
│   │                             #      （P50/P95/P99 + run 末快照）
│   ├── joint_scheduler.py        #   J：joint/load-first/affinity-first/
│   │                             #      face_static 选择
│   ├── hbm_port_flow_registry.py #   实例 HBM 端口流注册表（u_port 除数）
│   │                             #      + D06 候选阶段份额只读叠加视图
│   ├── link_quota.py             #   通用流量治理：链路∪端口动作级配额
│   │                             #   + AIMD 控制律 + ρ_eff/Q_init 派生
│   └── event_recursion_predictor.py  # E：运行期事件递推预测器（adaptive
│                                 #   正式在线实现，写回×恢复双方向）
├── face_scheduler.py             # KVCacheManager：T/E 注入 + home/工作
│                                 #   副本账本 + merge_back 合并事务
│                                 #   + kv_delta_journal 披露 + 逐组恢复
└── online/
    ├── sh30_online_scheduler.py  # joint 准入（替换三段式）+ 因果 decode
    │                             #   增长 + 完成合并 + 配额准入
    │                             #   + 遥测 ingest/AIMD + 到达合同
    ├── graph_batch_builder.py    # 多传输准入（copy 前缀+后缀）/merge 发射
    │                             #   + copy 块级交接 + 逐层段就绪门控
    └── online_service.py         # 服务入口 + joint manifest 落盘
                                  #   + 遥测完备性侧车回填
```

共同生命周期（§2/§3）：轮次 `QUEUED → PREPARING → EXECUTING →
COMPUTE_DONE → MERGE_WAIT/MERGING → COMMITTED → SERVICE_DONE` 的账本
映射——

* **home（§2.1，2026-09-17 修订）**：每 session 持 `home_instance`
  （首轮发射建立）；逐出不改变 home；**合并事务可迁移**——merge v2
  少并多的胜者若为执行实例则 `home := exec`（copy/recompute 零字节
  翻转与 REMOTE 无主回暖均迁移；REMOTE 态下标签冻结为无主语义，下次
  服务落地即改写）。**逻辑 home ≠ 基础物理所在**（2026-09-26 D-BI）：
  P→D 迁移（`move_prefill_to_decode`）＋ merge stay 提交会让会话终驻
  偏离 home，跨实例工作副本事务另持 `base_instance_index`（准备相快照
  的物理基础驻留实例）——守恒审计、merge 的基础侧台账/传输端点、以及
  P→D 边界视图（`PDBoundaryView.home_instance`＝源端；D18/D19 的原地
  继续适用性与合并计价端点）一律按物理基础
  所在核账，home 仅作 §2.1 标签与平局语义。
* **执行与增长（§3.1）**：新 KV（prefill 输入 + 实际 decode）在执行
  instance 产生；准入预约**动作感知足迹**（R1'：三动作 stay/copy/
  recompute 一律按 history+input 整份计——2026-10-14 远读退役后增量
  预约口径随增量布局族一并删除，PARTIAL 基跨实例动作的后缀池恢复
  预约仍逐 rank 记账并在物化后抵扣；可行性/预约/物理可行性三处同口径；
  无 PARTIAL 驻留钉扎：容量只影响所需字节与驱逐等待计价，不做候选
  掩码），decode 按列车核销的实际消费**因果增长**（每次增长经同一
  T+E 释放机制准备空间；容量不足时进入**停滞/唤醒**（R14）：会话
  暂缓后续列车参与，容量释放（KV 纪元 ⊕ 实例纪元重试键）后唤醒，
  等待如实计入 E2E，全停滞死锁由守卫显式 fail-closed；不提前按真实
  最终 decode 长度预约——`final_context_tokens` 不进任何决策输入）。
* **合并（§2.2/§3.2，merge v2 少并多——2026-09-17《部分层逐出kv管理
  改造分析方案》需求②）**：`merge_back` 在完成处理时比较**两侧保留量**
  （基础侧 = 物理基础驻留实例（`base_instance_index`，2026-09-26 D-BI
  起可与逻辑 home 不同）的基础驻留前缀 H；exec 侧 = 工作副本 W——
  copy/recompute 为
  并集），**小的整份 NoC 搬给大的**（省 D2D 流量；时机不变）；合并
  **零池写**（I6：恢复的 KV 按热 KV 处理、不执行完再踢出去——用户
  裁定），结果恒为**全层 LOCAL@胜者**、`home := 胜者`；copy/recompute
  执行端恒持并集 → **反向零字节翻转**（零传输、home 侧基础释放）；
  REMOTE 基 = 无主 session（裁定③）：就地保留（零传输零池写）、
  工作副本转正、home := exec（"远端服务一次后顺势回暖"）。空间准备只
  经统一 T+E 逐出（`_ensure_capacity`，胜者侧；零字节翻转/in_place
  无需准备），**双侧深缺口才 fail-closed**
  （R4 自降级与 k=0 池归并兜底退役——"merge 永不失败"合同变更，热 KV
  裁定的必然推论；`merge_degrade_events` 台账冻结、新 run 恒空）；恰好
  归并一次由 `last_merged_request_id` 版本键断言保证（`mark_complete`
  拒绝未合并的工作副本——C3b 新合同下 service_done 先于 merge_done
  分报：完成标记钉在响应交付，merge_done 为资源结算边界独立披露行，
  两者分别可观测）；merge
  结算闭合 = `kv_delta_journal` 行在案（C14：watch 交付 ⇔ journal 行，
  缺行 fail-closed）。
* **到达合同（2026-09-22 C3b 迁移，全方法/对照统一）**：下一轮到达
  alarm 锚定**响应完成 service_done**（`a(s,k+1) = f(s,k) + z(s,k)`，
  f = 上一轮向外完成响应时刻、z = 工具/用户等待；有/无 merge 流同锚
  单一路径），thinktime 自该时刻起算；**merge_done 为资源结算边界独立
  分报**（决策日志 `kind=merge_done` 披露行：merge_done_ns /
  next_arrival_world_ns / next_turn_arrived_before_merge_done——F11
  分报配套，两事件分别可观测）；到达 ≠ 就绪——**到达后数据依赖门控**
  （保守口径）：图侧 interval gate 前递锚 merge 尾标记 + local_hit 同
  rank 串行链接续 + 池在途块 store 尾前递边；thinktime < merge 时长
  时（N2）merge 成本以"下一轮数据等待"留在会话闭环内（不逃出 E2E、
  等待窗口由 merge watch 交付解除）。
* **动作语义（§2.2 表；三动作 = stay/recompute/copy（2026-10-14 起——
  第四动作 remote-read 已随远读机制退役），全
  候选逐 (instance, action) 对联合 argmin，实例永不掩码；跨实例平局
  （最优 cost_ns 整数相等、并列实例 ≥ 2）自 v9（2026-10-20）起按实例
  编号全局环序轮转分摊（单实例平局/唯一最优不读不推游标、行为与旧
  固定序恒等，见 §3 J 行）；实例内同价并列仍 `ACTION_ORDER` 终裁）**：
  stay=本地命中/部分恢复；copy=基础历史复制
  为工作副本（前缀 NoC + 缺失后缀池恢复）——**实现/测试锚点（R16，
  2026-09-15）**：物化 = `face_scheduler.py` `_noc_transfer`（层区间化，
  `[0, base_prefix)`）+ `_remote_load_transfer`（`[base_prefix, L)`）、
  计价 = `joint_cost_model.py` `estimate_P_with_continuation`（D18 起入口
  改名，原 `estimate_action`）ACTION_COPY 两腿
  （`noc_prefix+pool_suffix_restore`；腿内自 2026-09-22 计价改造批起
  为逐 shard 三腿 min——NoC 腿 + home 读腿 + exec 写腿（C1/C2），
  u_port 端点争用经 HbmPortFlowRegistry）；单测 =
  `joint/test_joint_review3_fixes.py`（复合两笔物化/计价闭式/图发射/
  水印重放/merge 回归 22 用例，含 8b 图发射快路径）+ 容量压力夹具（PARTIAL×copy 端到端
  命中）；**copy 块级交接（C13，2026-09-22）**：前缀 NoC 迁移按
  K 对齐块化（`_cb{k}` 唯一相位键）、源端释放点=**prefill drain 边界
  批量结算**（P2-6 承诺降级登记：计划"同一交接完成事件立即释放"实现
  为 drain 边界——保守方向=高估驻留窗；逐块立即释放需图侧逐 chunk
  watch 新通道，机构件级未立项），守恒科目 `#handoff`（交接重复量）
  与 `#copy-stream`（一遍历史）入决策日志审计；recompute=**完整 H+U 重放**
  （D10，v2.0 §6.2，2026-09-26 取代 R13"只重算缺失区间"折算口径：全形态
  LOCAL/PARTIAL/REMOTE × 驻留/异地 span 基 = 0、重建基数 = 整份已保存上下文
  H——缺失层比例折算 ceil(H×(L−prefix)/L) 取消（KV 非任意层激活检查点）；
  驻留旧分配仅作重建**空间**复用（prepare 账本字节照旧物化缺失后缀、
  recompute@home 仍 stay 同构本地提交、merge 零流量），**不免 FLOP**——重算
  量按整份 H 折入列车 span；不把旧 U 再算成新 prefill（token 序列 = H 重建 +
  本轮 U 各恰一次）；阶段字段/token-RNG 元数据（`joint_rebuild_context_tokens`
  / `joint_prefill_stage=prefill_replay` / `joint_suppress_sampling=output=
  False`——重建段 logits 丢弃、不重复输出历史 token，末 chunk 正常采样本轮首
  token 恰一次；D 边界静默重建 `context_rebuild` + suppress=True 语义
  已在视图/计价层定义（`StageWork`/`BoundaryKVLayout`，test_d19 钉测）；
  D 边界静默重建执行面已接线（2026-09-27 修正轮 C4——rebuild span/
  首 token 恰一次/核销物化+预约注销全链路，掩码臂真实执行验证，见
  D18 行登记），两语义不混用）三方同源：runtime、决策日志、plan_dict
  （构图器））。（第四动作 remote-read 及其混合形态/前缀读流语义已随
  2026-10-14 远读退役整段删除，历史描述见 PROVENANCE 退役批登记节。）
* **链路遥测（C6–C8，2026-09-22；D13 修订 2026-09-27）**：C++ 在线旗
  标 `--link-telemetry` 每决策 epoch 在桥请求顶层携带 `link_telemetry[]`
  逐链路窗口差分；SH 侧 `_ingest_link_telemetry` 解析（整型 LinkId →
  端点 (src,dst) 键换算）喂 JCM `divisor_effective`（max(注册在册流,
  遥测时间加权流数) 合并；D13 起速率样本只作观测/披露通道，速率-only
  链路 = 流数显式未知——除数退在册流近似，不再以 capacity/速率冒充
  流数）与 AIMD（`rid#readplan` 读流承诺账本与 `#prefill_read` 前缀
  读流登记已随 2026-10-14 远读退役删除）；collective_coverage 翻转
  条件（全程开启 ∧
  逐 epoch 窗口无空洞）入 manifest 与决策行。
* **配额机制（C9–C11，2026-09-22，缺省 off）**：`JOINT_QUOTA_MODE`
  = off|static|aimd（三开关之一，与三机制/combo 正交）。链路 ∪ 实例
  HBM 端口的**动作级准入**（逐候选判据、实例永不掩码；全动作不可行 ⇒
  `quota_deferred` 回队，等待行 `wait_reason ∈ {capacity, quota_link,
  quota_port}` 三分）；static = 固定预算（链路 `Q_init =
  max(1,⌊ρ_eff⌋)` 逐配置派生、merge bulk 名额 N_bulk=Q_init——
  realtime/elastic 流类与端口平价门已随 2026-10-14 远读退役删除，
  在册流类恒 FLOW_ONESHOT）；aimd = static 基础上遥测驱动的 AIMD
  动态调整（F7：aimd ⇒ 发射层自动注入 `--link-telemetry`，不存在
  "aimd+无遥测"合法启动路径）；流生命周期借还配对审计（merge
  预留 `#merge-reserve`，service_done 裁决释放败者侧 /
  merge_done 闭合门后释放胜者侧）；δ_adm=0（A3' 终冻，与 argmin 合取
  恒等）。off 臂决策序列与 off 前逐字节同（F7 零漂移实证）。
* **逐出（§4/§5.6）**：T 定类别/对象序（`eviction_class_order`），E
  定层数（`LayerEvictionPolicy.plan_release`，只读计划、提交前复核）；
  逐 rank 缺口检查（不用实例汇总掩盖单 rank 超容）；deep-gap 台账
  fail-closed 沿用基底 D4（准入/merge/decode 语境按 N3' 类型化
  `KVCapacityError` 分别转延迟/自降级/停滞，合同类异常原样上抛）。

### 2.9 紧凑管线与稀疏账本协议（2026-10-09 v2 施工；版本表随 2026-10-14 远读退役更新）

各接口独立版本化（下表为 2026-10-20 v9 平局轮转终态）。未知版本及旧
envelope 在生产入口拒绝。文件中的 `WorkLedgerV2*` 类名和
`raw_v2_frame` 测试审计标签沿用原名称，不表示保留生产 v2 解码通路。

| 层 | 版本 |
|---|---|
| 外层 decision bridge schema | 4；`bridge-v4-20261011`，`snapshot_handle.kind=work_ledger_v4` |
| work-ledger envelope schema / protocol | 4 / 4；`workledger-v4-20261011`（D11 最小联动档：pipeline 行恒 0 行、0 行帧合法） |
| GraphBatch schema | 3（必需 `kernel_contract`；`pipeline_descriptors` 键已随 2026-10-14 远读退役删除——恒不再出现，出现即 fail-closed） |
| descriptor | **已退役**（2026-10-14：`PipelineDescriptor` typed wire 整体删除；其原尾部的 `PipelineKernelContract` 迁出为独立头 `astra-sim/workload/execution_driven/PipelineKernelContract.hh` 保留，die 计算后端 window 契约） |
| native ABI | 1（`jproj_*` stateful C ABI） |
| prediction config | `predcfg-v4-20261014`（action_space / multi_source_protocol 两轴删除） |
| 模型语义 | `v9-tie-roundrobin-20261020`（经 `v8-two-actions-20261015` 二动作终态升版：joint 主臂 P 入口跨实例平局按实例编号全局环序轮转分摊、成本函数本体未动；与 v8 及以前的选择分布类指标不可比，登记见 PROVENANCE 2026-10-20 节） |
| planner-state | `joint-accel-planner-state-v2-20261014` |
| python runtime asset | `joint-accel-python-runtime-v2-20261014` |
| 决策日志 / 快照 schema | `declog-v3-20261014` / `snapshot-v3-20261014` |

流水 transition 的 `slot_patches`、`credit_patches`、`obligation_patches`
仅携带实际触及行的 `{before, after}`；null 明确表示创建/删除。不可变
描述只声明一次。`cursor_patches` 仅更新改变的 run，退休使用显式
run tombstone。活动资源在每次 delivery 的 `active_resources` 中提供
一份当前 tick 的权威剩余量，不嵌入每条 transition。

`request_close` 显式记录既有行政关闭行为；每个 run 的
`cancelled_endpoint_count` 区分未发射尾部取消与实际完成。已发射
 data/ACK/HBM 继续真实排空，取消不增加完成计数，不归零在途资源。
节点/流水共享连续逻辑序号；两账本校验完成后原子提交。只承诺上一
 delivery 的精确幂等重放，内容比较保留整数与浮点类型区别。
journal 保留生产写入及实际重放路径，按序流式读取账本侧车。

发布精度参数继续为 `window_tile_limit=2 / window_local_error_ppm=0 /
window_refine_margin_ppm=20000`；K=1 使用同内核的精确配置。K 是精度
参数，本轮没有新增旧新实现开关，也没有调整物理 tile 大小。

历史 F 链的 137 案例、13 档 tune、123 案例 holdout 属于当时源码下的
冻结精度证据；本轮不重跑，也不把它们当作当前源码的新增验收。
历史旧臂约 1459 秒、新臂未完成，只提供诊断背景；历史报告的 86 倍
归因不作为正式速度证据。当前速度结论须分别验证工程、近似额外收益
和总收益，整体 20× 必须由完整 first-1s 工作量支持。

本轮针对验证入口是现有 ledger/remote-read/GraphBatch 测试，以及
`joint/test_sparse_pipeline_integration.py` 的单案例桥接检查。
构建/运行应放在仓外副本，按根《joint改造执行文档v2.md》的 T0—T4
预算和进程组内存护栏执行；禁止默认启动旧 capture/tune/holdout 银行。
实际运行结果和未完成项随本轮仓外 evidence root 的
`manifest.json`、`checks.json`、`performance.json` 记录，不预支达标。

本轮最终代码增加 NodeStore 的 ready 集合计数并移除逐节点修改排序向量的
后缀搬移：实际 ready 集合仍按 ID 有序、发射前捕获快照、回调后重新验证
成员资格，资源阻塞的计算节点仍遵守原发射规则。delta 使用有序集合，
仅在既有诊断读面请求时物化一次有序向量。原有 compact descriptor、
依赖、必要 ACK、KV/mapping 及物理模型保持。

精确仲裁后的 `projection_stats` 保持最终有效精度披露，同时保留首次
`build_wall_ns`，分别记录 `first_projection` 与 `exact_reprice_wall_ns`。
P/D 决策行增加 DIE 现有累计/增量及投影累计计数，避免异常退出后丢失
已完成决策的归因。DIE、构图、投影、策略与桥的父子桶存在包含关系，
不能相加。

2026-10-10 阶段结果：单转移的同域义务从 16 增至 64、256，实际访问均为
11 次，prepare/commit 临时峰约 28 KiB；P2tile/D3tile 真实执行、五帧接收、
同 K1 前态全候选对照和新进程 journal 重放通过。该结果与 T3/full first-1s
性能分开登记。

前一 source freeze 的三对正式微任务中位数为工程 0.507×、近似额外 1.022×、
总计 0.515×；后续 attention-only freeze 的三对更新为 **0.5223×、1.0586×、
0.5380×**，近似额外倍率范围 0.9959–1.0706，跨 1。工程及总墙钟在这组小
任务仍慢。两组 T3 都绑定各自冻结源码，不代表现有 24 文件 final-cpp freeze。

full first-1s K2 的 105.195 s 及后续 30.021774 s 运行都未自然完成。进展解析
审计证明旧 guard 漏读了有效 MainWall tick/decision 记录；撤回“确认连续 30 秒
无业务进展”的结论，两次停止时间仍作为已消耗资源保留。当前冻结源码尚未
完成 full first-1s 运行，因此没有完整任务加速比，`speed_goal_met=false`。
交付范围、前阶段证据与身份见根《joint改造执行结果v2.md》和仓外证据目录
`/home/sunhao/joint_accel_v2_20261009.gt48r455`。不能用微任务倍率替代整体 20×。

### 2.9.1 2026-10-10 后续局部复核

Attention 编译对每个可行的 `(q_tile, k_tile, k_partition)` 惰性计算一次
layout 无关工作量，并在后续布局候选间复用；原 span/q/partition 遍历和
浮点加法顺序保留。两个真实输入的全部有序候选字节完全相同：4404 与
4410 条候选，均含阶段、依赖、汇总、最终选择与 schedule double bits。
新路径各建 112 个 aggregate，实际 unit walk 分别为 380160 与 1453530；
旧路径按原调用结构推得每候选一次（该旧计数非仪器实测）。局部枚举耗时
分别为 187.118 ms→8.781 ms、2694.169 ms→61.707 ms。这些是两个纯构造
诊断值，不代表完整 DIE 编译、handler 或模拟加速。

同一 first-arrival 输入的严格诊断中，102-node GraphBatch response 逐字节
相同，36 个候选的适用性/选择以及全部适用成本和 native quality 对齐；
6 个 cold DIE batches、48 specs、264 memo hits、6 misses、36 projections、
18 次 exact E0 repricing 和 7812 projection events 均未改变。report 的 scheduler `handler_ns` 为
8.4540 s→0.5096 s，cProfile `_handle_request` 累计范围约 8.4547 s→0.5102 s；
两项计时边界略有不同。DIE FFI 为 8.2423 s→0.2987 s。该结果不含 C++
事件循环，不是完整 first-1s trace，也不证明 20×。18 个受影响 CTest
通过（5.21 s）。

新增 DIE issue 计时只用于归因：`total_issue_ns` 包含嵌套的 key/cache/
compile/put/setup/start 子计时；MainWall 的 `issue_dep_free_nodes` 覆盖两
个在线 issue 调用点，`issue_minus_die_ns` 是包含式边界的剩余差值，不能把
父子桶相加。计数器是单事件线程上的进程累计值，不在 heartbeat 间重置。
本轮未改变模型、物理、调度、精度、协议或 native ABI。


### 2.9.2 历史下标：final-cpp source freeze 的局部复核

`logs/resume-20261010/final-cpp-source-freeze.json` 记录该 executable/source
snapshot 自续工基线后的 24 个改动文件及 850 个来源 hash；构建范围为 all
CMake targets。以下构建核验绑定当时的冻结版本，本节是之后的文档续记。构建用时
662.871087 s、进程组峰 RSS 1,810,722,816 B；`final-cpp-build-source-audit.json`
的冻结来源核对无差异（对账时 source bindings 未变）。此后只修改本 README/PROVENANCE
续记，未改源码、header、tests、config、build artifact 或 sidecar。注册 CTest
为 21/21 PASS，5.091343 s、进程组峰 RSS 45,723,648 B。
六个 direct tests 为 PASS，6.252883 s、进程组峰 RSS 39,145,472 B。首次
GraphBatchCommitterTest 因测试 fixture 缺 physical anchor 抛 ParseError；保留
该输入错误记录，单独重建 test leaf（8.051097 s）并修正 fixture 后六项通过，
不是生产代码失败。使用既有 449 B ET fixture 的 NodeStore 全 parts 回归无 skip；
私有 work 三尺度断言及 lazy-key/pending getter/AP-sweep/CandidateIdPreview
针对测试均通过。

真实 seq9 metadata-only probe 在 54-rank、前缀 seq0..8、3822-node seq9 输入下
完成，用时 0.306718 s、进程组峰 RSS 119,222,272 B；不触发 physical issue，tick
不变、active transfer 为 0。seq9 的阶段计时为 JSON parse 50.686243 ms、typed
parse 123.189758 ms、full validation 1.574980 ms、mandatory preflight repeat
0.167937 ms、prepare 44.228798 ms、Phase B 10.093235 ms。对照 before 版
10.029838 s timeout 只用于定位局部 metadata 瓶颈；负载和外部 CPU 环境不同，
不报告正式 speedup。

`micro-physical-final-audit.json` 对同一真实 P2tile/D3tile fixture 逐帧核对：五个
delivery frame 完全相等，物理 measurement/certificate/audit 除三处核实过的
producer binary identity 记录外相等；当前两账本接受五帧，watermark=1032。最终
micro 实跑 0.121939 s、进程组峰 RSS 29,761,536 B；严格 audit/apply 0.166785 s、
峰 RSS 40,226,816 B。AP sweep 特定形状计数为 1,869,540 个 P² pair/envelope
运算降至 3,528 个 AP candidate/envelope 运算；这是公式与原谓词的形状计数，
不是旧循环的仪器实测。旧 CRT 在同输入也给出 3,528 个候选，不能宣称 CRT
530×；generic worst case 仍为 P²。UInt 末端原子拒绝验证亦通过。

该 source 当时尚无正式 T3 或 full first-1s 计时；当时已测 T3 是此前 attention-only
freeze 的 0.5223× / 1.0586× / 0.5380×，不得追认到当前 24 文件源码。full
first-1s 当前版本仍未验证；新正式资源配对待外部四仓暂停安排明确后再执行。
当前 `implementation_complete=false`、`speed_goal_met=false`。生产 K2 参数、模型
精度、物理语义和 schema 未变，本轮没有新增用户功能开关。

### 2.9.3 历史下标：closure-final 时的跨组槽位坐标修复

完整 first-1s 新臂越过原 seq9 prepare 阻塞后，在 seq11 的严格槽位校验
失败（20.020876 s，采样组 RSS 1,066,356,736 B）。owner 的 tile 编号为
组内编号，而物理两槽 parity、release lag/watermark 用全局 tile 轴。
消费者现通过 immutable descriptor 的 `global_tile_first` 换算；data_ready
槽键、跨组释放和复用判断同步，owner 的组内身份、必要 ACK 和物理规则保持。
两组各3tile 的 source/target before/after 及错误输入原子拒绝回归通过。
80项相关测试/15个子测试通过，原失败行的槽1接受/槽0拒绝通过；生产
journal 0..10 加完整失败 delta11 的真实双账本回放通过（4.831338 s，
峰组RSS 662,237,184 B，watermark103390）。这些是语义修复证据，完整
8请求/1646decode 自然闭环和20×仍待验收。

MainWall/FinalWall 另披露已有 Workload snapshot/issue 累计桶和 NodeStore
ready复制计数，用于定位剩余工作；同步回调可嵌套，不能与外层/DIE桶相加。
该冻结版本的九臂 T3 中位工程0.5423×、近似额外0.9919×、总计0.5352×；
该小任务仍慢，不能追认为本次 Python 修复后版本的性能。

### 2.9.4 2026-10-10 历史测量与随后局部修复

本节 `closure-history-index` 是当前结果；2.9.2–2.9.3 的 build、T3/T4 与消费端记录仍绑定各自历史 source freeze。当前正式 T3 为 CPU0 串行独占资源的三对九臂自然完成小任务：工程 `S_engineering` 中位0.5515840519×（0.5312025661–0.5920320307），近似额外 `S_approx_extra` 中位1.01506212585×（0.9161532170–1.0266011234，收益未证明），总计 `S_total` 中位0.54533315106×（0.5423920495–0.5598920803）。小任务工程及总耗时变慢，不支持全链加速。

当前完整 first-1s K2 T4 使用 8 请求/3 sessions/61,550 prefill/1,646 decode，生产 ppm=0、margin=20,000、event budget=262,144，保留全量 journal、ACK 与 validation。180.018345174 s 达 TIMEOUT，进程组峰 RSS 3,500,515,328 B（约3.26 GiB）；末次 heartbeat 为23 deliveries/tick 6,021,916,728，仍显示业务进展。请求和 settle 未全部完成，故没有全链倍率，`speed_goal_met=false`。

最新心跳的 outer `issue_dep_free_nodes` 为59.091820304 s/214,200 calls；其中 DIE adapter 嵌套5.519954883 s，含式差值53.571865421 s待定位。Workload snapshot/issue分别为6.391442574/52.672786866 s，34,204,744 candidates、184,308 issued、46,710 NodeView materializations、34,067,146 ready ID copies。Bridge round-trip 71.798419381 s/23 deliveries 是包含式桶，不与嵌套阶段相加。seq11 profile 的9.628852653 s、645,169,152 B在guard处 GUARD_ABORT，双账本apply不完整，只能作为诊断记录。

R_work 完成的七个 NodeStore/Workload C++ source/test 文件包含于当前47-file measurement freeze；其 SHA 与 `history-materialization-final-source-hashes.json` 七项一致，因此正式 T3/T4 测量绑定的代码已经包含这些修复。测量后 root 才将七项 hash 保护回写 live；`protected-history-index-integration.json` 记录850个来源文件核验、842项未变，归档副本为 `delivered-history-index-source`。其后9个RRE dependency-index/Python Mapping source/test文件已验证并保护回写live，但晚于该freeze，不能追认进正式T3/T4性能；`protected-dependency-mapping-integration.json`记录850个预期文件核验、10项改变（9 source/test + inventory）、840 untouched及inventory65，归档副本为 `delivered-dependency-mapping-source`。更晚的4个canonical source/test也已通过检查并保护回写live；见 `protected-canonical-integration.json`（850 expected、845 untouched、inventory65）及 `delivered-canonical-source`。NodeStore 用非空 append-only reservation blocks 和每 block interval tree 替换 flat history 查询，不展开未来 tile；历史目录二分后再由相关 block/tree 剪枝并走原精确 CRT。point lookup 可按 O(log B + log P + C) 描述；AP range query 还要计相关 block/tree 访问，通用 C 可到 P，不能称所有查询最坏对数。Workload 在 availability 判定之后才为真正发射构造普通 NodeView；blocked candidate 不 copy，N=8/32/128 时真实微型计数为3N候选、0 blocked copy、N issue copy。生产配置、语义、精度和物理行为未改，无新开关。

RRE dependency index 在 `(rank, group-local group, leg, shard)` 上建立不可变 tile interval buckets 与 max-end tree，替换三处全表扫描；命中顺序仍来自原 dependency 声明，dynamic terminal checks、checked affine ID、排序/重复项 fail-closed、atomic prepare 与 retirement 保持。索引不展开未来 tile或缓存 ready bool。点/范围查询按相关 bucket/tree 与实际候选计成本；generic query worst case可达 O(R)，C 个命中排序为 O(C log C)，不声称所有查询对数。单次建索引 O(M log M)，storage O(M+K)。三尺度 unrelated=8/32/128 的tree访问为9/13/17，旧谓词测试oracle为14/38/134 rows，固定匹配各2，memberships=16/40/136，tile expansion=0；这是计数复杂度证据，不是旧生产wall-time倍率。该微型协议案例真实贯通 C++ execution→bridge→Python 双账本→consumer并自然settle；仍不替代full T4，也没有新增生产开关或改变物理/精度语义。

Python runtime Mapping 检查改用私有 `issubclass(type(obj), collections.abc.Mapping)` 路径，保留 API/annotations 和 dict、UserDict、MappingProxy、自定义实现的接受行为，也保留对 Mock/伪造 `__class__` 的拒绝。85项测试覆盖中首轮84 PASS、1项新 descriptor 错误消息regex不符；更新既有断言后只复跑该项并PASS，不称一轮85全绿。12个实际 journal prefix 的 accepted state/canonical bytes SHA 与最终 replay tuple before/after 全等、watermark103390；该纯replay诊断的墙钟改善不代表独占正式倍率或全链收益。

原47-file冻结源码的 all-target build PASS（658.6168134 s，峰RSS1,819,267,072 B；最终增量 build closure 0.639557 s），CTest 21项 PASS（5.045493 s），五项相关 direct tests PASS（5.89389956 s）。后续9-file dependency/mapping源码 all-target build PASS（236.899159105 s，峰组RSS1,858,707,456 B），CTest 21项 PASS（5.08387993 s）。新增 direct checks 中前四项先PASS；Bridge初次因错误cwd导致peer不存在并在10 s timeout失败（14.842969 s），保留该记录；只对Bridge用绝对peer和正确cwd复核后PASS（1.134862568 s），没有重复前四项。Python 18项runtimechecks通过；descriptor 85项首轮84 PASS/1测试断言失败，修正既有错误消息断言后仅该项PASS。NodeStore 449 B ET完整成本检查另PASS（0.005280 s，早先单条成本断言失败保留并已修正测试）；P2/D3真实 micro五帧为0.110254528 s、严格audit/apply 0.179410663 s，watermark1032，物理状态/回调/证书/wire精确相等（3处已核实binary identity例外）。原47-file的seq9 no-issue十批metadata prepare 0.291727720 s仍只表示prepare、不含physical issue。

末尾 event canonical-tree 消除后，77项相关测试PASS（0.312025750 s）；覆盖两node→一末尾tree、10种坏位置在beginTxn前拒绝、int/float bytes分异、v3/legacy幂等与篡改拒绝、缺末canonical在publish前拒绝。public/legacy parser 的 `build_canonical_content` 缺省仍true；v3仅物化末尾node event，即使尾部还有pipeline rows，完整frame validation/canonical bytes/journal/replay均保留。producer未改，当前consumer复核原真实五帧磁带PASS（0.183188126 s，不重跑物理producer），目标依赖闭包PASS（0.597408393 s）。12个真实journal prefix中，每个 canonical frame SHA、accepted双账本状态和末尾ordinary replay均相同，watermark103390。全部typed parses=62,404、canon trees=11（空首帧0）；seq11的48,348 node transitions/typed parses及最后一棵tree保留，canonical bytes 58,641,488 B。guard wall为4.359588688 s、峰组RSS524,906,496 B；seq11 apply 2.510265463 s、sum apply 3.184235362 s。这是单次组件诊断，不是formal速度比。`validate_envelope` 私有DTO nested payload浅拷贝，`_parsed`快路只支持同一调用栈validate→prepare；不宣称通用immutable checkpoint。

`closure-global-coordinate` 的180.0109 s/19 deliveries与 `closure-final` 的20.0209 s/seq11 parity failure是较早历史运行；consumer坐标问题之后已修复。更早的105.195 s/30.021774 s guard stop也不证明stall。当前manifest快照为225条guard checks、653.914876979 s仿真/投影guard累计；纯数据诊断不增加该simulation seconds。RRE/Mapping/canonical三组局部修复均已验证并回写live；formal T3/T4仍绑定47-file freeze，工程/近似额外/总倍率及20×结论不变。代码余项已清；剩独占formal T3与一次完整first-1s/180 s cap T4闭环。四仓目前只读恢复，root将取得新的外部暂停窗口后再测。`closure-dependency-canonical`是已准备但未运行的测量包，不得记PASS。`implementation_complete=false`、`minimal_integration_completed=true`（仅微型链路/journal）、`speed_goal_met=false`（完整first-1s scope）、`research_conclusions_validated=false`。详见根《joint改造执行结果v2.md》及仓外 `logs/resume-20261010/`。

### 2.9.5 精确工程改造与未通过的性能验收

本次续工完成两处进一步复杂度修订：`Execution::preview_consecutive_internal_timers` 原先每次复制全Impl，包括未访问的plan后缀和dependency arrays；现只在该函数内部复制四个live frontier，借用dense数组并记录触及写，返回及异常时逆序恢复所有dense值、完整scalar cursor和原vector buffers/capacity。没有新增public checkpoint API、开关或coalescing范围，HBM idle边界、FIFO、Memory交接、近似精度及物理量保持。一个Timer后接Memory再接N8/32/128无关后缀的真实原生调用，旧请求分配669/2013/7405B、各17次，新196/196/196B、各6次，实际都只消费1个Timer/1953125ns；full progress/token/cache及后续到Done轨迹精确一致。新实现逐个注入1Timer与4Timer前缀的6+9个allocation failure点全部rollback并保留未来轨迹；overflow与连续preview检查同样通过。scratch成本随live frontier F与实际写U增长O(F+U)，不随无关后缀T/E复制；若预览覆盖长pureTimer前缀，undo可能比旧dense copy大，未主张所有输入内存更低。

`LedgerWorldSource.build_context` 原route key逐项any(resources)成员扫描，N不同键的独立旧列表oracle实际比较28/496/8128；现单个local ledger_keys set实际contains/add各8/32/128，保留首个ResourceSpec、资源/work/approx顺序、pipeline覆盖、fallback及同一JCM共享context，无跨snapshot动态cache。集合查询为expected hash complexity，不保证对抗碰撞的最坏常数。混合多hop/shared/fallback/overlay context及8个合成P/D action additions的投影输入逐字段相同；此证据不冒充实际完整候选pricing性能。

obligation正常parsed冻结行不再重复比较overlay中同一引用；before、首次new/expected、seen集、aggregate仍全校验，非builtin字符串/key/行走旧postcheck。真实12-prefix中typed comparison节点1049461→877297，同时新增guard50064次/294264个key访问，额外工作如实计入。所有accepted state、canonical bytes、末事件replay精确一致，watermark103390，62404 typed node events仍处理；两个单次guard4.231725/4.225347s、apply总3.027902/3.022581s相近，未证明墙钟收益。

全目标build 6.398614s/采样峰411267072B PASS，228相关C++来源绑定未变；注册21CTest 5.067288s PASS，含真实DIE/HBM mutation/生命周期。相关Python首轮56methods中55通过、1新增scale方法3个非法HBM resource key子错误；仅修该fixture key后失败方法单独PASS(0.121262s)，生产没有为错误放宽校验。root在发build前删重复end/done声明，不伪造编译失败。P2/D3真实微型producer 0.116421s/物理及当前双账本audit 0.183054s PASS，全physical/certificate/callback/wire与原磁带相同，仅3处已核实binary identity例外，水位1032。

上述6source增量晚于最近正式`closure-slim-ready-native`冻结，不能追认旧倍率；尚无新的完整first-1s结果。原正式全8req/61550P/1646D仍180.019101s TIMEOUT、29delivery/RSS2993188864B，speed_goal_met=false、implementation_complete=false。累计仿真/投影1232.094314430s（20.53min）不重置，无新增整段/旧长臂；算法、离线replay、编译另列。源码与各真实检查见evidence root `logs/resume-20261010/remaining-preview-context/validation-summary.json`。

以下记录按各自当时source冻结；其中“当前”不覆盖本次新增6source。

当前已验证的最新局部增量是 HBM 临时分配消除：`advance_to` 的用户指针数组仅在本次调用内复用，每个 substep 仍在更新 latency 前按原序重新捕获用户；`schedule_next_transition` 用计数和按原 jobs 序的只读扫描取代临时数组，除法、最小值及 callback/observer/版本顺序保持。没有改物理服务、调度、KV/mapping、必要 ACK、journal、validation、K2/ppm0/margin20000，没有新增开关。

| N | issue 分配次数（旧→新） | drain 回调分配次数（旧→新） | drain 请求分配字节（旧→新） |
| --- | --- | --- | --- |
| 8 | 87→58 | 36→15 | 3432→2928 |
| 32 | 371→204 | 174→19 | 21544→11376 |
| 128 | 1687→782 | 912→23 | 216872→45168 |

上述计数来自实际模型、同一 probe object 的旧/新链接，不是算法副本。完成数、顺序、tick=1、version=N+3、以17位 roundtrip 文本记录的服务/实际字节均逐项相同；同时含一个尚在 latency 的 transfer，故是局部前缀测试，不能签发完整任务倍率。计数是请求分配字节，不是峰 RSS。积分扫描仍是每 substep O(J)，调用内 scratch 的容量 O(J)；预测第二遍由遍历 U 个用户变成检查 J 个作业，仍 O(J)，U≪J 时常数可能增加，未主张普遍墙钟收益。

共享源码全目标构建 PASS（8.080965s，采样峰组 RSS 575,021,056B），228项CPP来源未变；21项 CTest PASS（5.051170s），含标准分配器下 fractional HBM 及真实 DIE lifecycle。P2/D3真实微型五帧生成 0.110097s、物理/wire/当前双账本审核 0.150040s PASS，物理字段与原磁带精确相同，仅3个已核实二进制身份字段不同，水位1032。新夹具最初把外层旧alarm当模型callback导致3个边界断言失败；只修等待真实membership version发布的测试假设，原tick/服务/版本断言保留，失败日志仍在。另一次审核guard绑定了不存在的文件，在child启动前失败，改为实际work_ledger.py后审核通过，不计模拟或伪装为测试PASS。

最近正式测量仍是本HBM增量之前的 `closure-slim-ready-native` 52项冻结：小任务工程/近似额外/总中位0.6927878269×/0.9165469529×/0.6223067358×，均未证明加速；完整8请求/3sessions/61550P/1646decode保留全部候选、ACK、校验、journal及naturalsettle，180.019100863s TIMEOUT、峰组RSS2,993,188,864B、29 deliveries，未自然闭环。新HBM源码不追认这些倍率，也没有新完整任务性能测量。`speed_goal_met=false`、`implementation_complete=false`。累计仿真/投影1226.910604767s（20.45分钟，含失败/中止），停止新增wholetrace；编译/算法/replay另列。资源窗口已经结束并告知用户恢复四仓，未发送外部信号。细项见证据根 `logs/resume-20261010/native-hbm-scratch/validation-summary.json`、`allocation-comparison.json` 及 manifest/checks/performance。

下列段落是此前各冻结源码的记录；其中“当前”“未实测”和累计时长只指该段当时版本，不覆盖上述已交付增量。

#### 历史正式冻结：closure-frontier-interval

最新正式证据为仓外 `logs/resume-20261010/closure-frontier-interval/` 的
`campaign-result.json`：52项输入、源码、二进制和工具绑定保持不变。绑定所用
exact code 已包含并已保护回写 live 的 node factory/selected-shard、C++
completed-count、Python OOO interval 和 anchor prefilter；结果记录
`source_bindings_unchanged=true`。此前各轮数据继续保留各自 source scope，
不把局部修复追认为旧性能。

T3三对九臂均自然完成。可观察工作守恒审核状态为
`PASS_WITH_VERSIONED_EVIDENCE_LIMITATION`：旧v1参考不暴露物理ACK类别和
parent edges，因此此两项没有跨版本可观察性补证。小任务正式倍率如下，
工程与总墙钟较慢，近似额外收益范围跨1、仍未证明：

| 指标 | 中位数 | 三对范围 |
|---|---:|---:|
| `S_engineering` | 0.6093206156430171× | 0.5123558082302367–0.6278346176654013× |
| `S_approx_extra` | 0.9666092555845509× | 0.9566970138184215–1.0834975620356697× |
| `S_total` | 0.6068707524117647× | 0.49016927174639136–0.6601974015472823× |

完整first-1s T4仍使用8请求/3 sessions/61,550 prefill/1,646 decode、生产
K2/ppm0/margin20000/eventbudget262144，并保留full validation、journal、必要
ACK和自然settle。180.014851448秒以`TIMEOUT`结束，峰值进程组RSS为
3,088,588,800 B（2.88 GiB）；末次记录28 deliveries、tick 6,940,513,414，
业务持续进展但请求和settle未自然完成。因此没有完整任务倍率，
`speed_goal_met=false`；没有延长timeout、tick外推或不满足前置条件的旧长臂
复跑，运行配置已恢复。

该T4末次记录中的outer `issue_dep_free_nodes` 为22.286471119秒，嵌套DIE为
6.965576110秒，含式残差为15.320895009秒。anchor资源查询47,022,024次中，
45,655,056次坐标预筛拒绝，1,366,968次进入动态预检。28个已完成delivery
的Python累计计时为handler 73.524561353秒、ledger apply 49.020812594秒
（typed parse 26.379335322秒、prepare 21.882650714秒、commit 0.246578429秒）、
policy 21.962450287秒。父子与包含式计时有重叠，不能相加成墙钟拆分，也不能
跨不同进展状态相除解释倍率。

累计仿真/投影时间为1,028.989339378秒，包含失败和中止；纯算法、回放、
profile与编译另列。状态为`implementation_complete=false`、
`minimal_integration_completed=true`（仅微型链路/journal）、
`speed_goal_met=false`、`research_conclusions_validated=false`。没有新增生产
开关；K2、ppm0、margin20000及物理/策略语义保持。历史137银行、完整精度与
SLO证据只对各自冻结源码有效，不能据此宣称当前整体全绿或整体完成。具体
来源、记录和计时范围见证据根manifest/checks/performance及本节campaign结果。

正式测量之后的精确修订现已通过核验并保护写回（8源文件，`protected-ready-selected-integration.json`），不能追认上表倍率。ready ID vector只在确切NodeStoreGraphSource、同一source身份、未饱和且可复用的ready版本下复用；全部逐项membership/lookup/硬件/anchor检查和virtual snapshot仍执行。N=8/32/128的固定3次blocked pass仍检查3N候选/成员，ID复制由独立旧对照3N减为N，实际fullfill各1、reuse各2。复制/移动/赋值的store回到完整snapshot路径，保留成员数据复制语义，覆盖同source同version不同ID反例；MAX饱和不复用、派生source override不缓存。

Python cursor active-vs-completed检查改为已排序half-open区间二分：A固定8、H=8/32/128，实际读取32/48/64，对照旧list谓词64/256/1024。target/cursor/issue只构造被消费shard；S=8/32/128固定相关项时target构造1/1/1、cursor4/4/4、issue4/4/4，对照旧fulltile算法分别8/32/128、32/128/512、24/96/384。run初始化仍随实际相关K增长，不宣称全部初始化常量。保留Mapping ABC fallback和数值类型区别；仅避免双外层dict复制和同一rank重复验证。

最终全target构建559.074331秒、峰组RSS1,906,380,800B，228项相关CPP source审计未变；21CTest全部PASS（5.068702秒），已注册Ready不再直跑。NodeStore传真实449B ET完整验证0.004761秒，Graph direct4.624419秒。40个相关Python方法首轮39PASS/1新virtualMapping测试误假设原生__eq__失败；改为严格typed结构比较后只重跑失败项PASS（0.115231秒），生产代码未改，不称一轮40全绿。

真实P2/D3五帧重新生成0.112785秒、严格物理/wire/双账本审计0.155700秒，只有3处核实过的binary SHA身份例外、水位1032。当前12个大帧prefix完整回放4.554776秒、峰521,478,144B，状态/canonical bytes/末ordinary replay逐帧精确一致、水位103390；执行62,404 typed事件、11棵末事件tree，40,920个concrete selector及其他cursor/slot初始化总计59,598次selected geometry/实际factory，整tile构造为0。当前真实Online+Bridge单完整请求0.296554秒自然结束，all8 P/D candidates、journal、5 bridge ACK、328物理flows、36fill/12release物理ACK、232parents和service/settle对照已审核PASS。这些运行用于功能和复杂度，未在独占窗口正式计时，不能作为工程/近似/总倍率。

最近正式52-binding源与后续38份源码接口已归档；最新8-file修订没有完整first-1s计时，`speed_goal_met=false`、整体原任务未完成。累计仿真/投影1,039.595242307秒，失败/中止计入，编译/算法/回放另列。接下来只做剩余稀疏行重复工作和策略调用链的小范围定位；不默认重跑整段或延时。

后续内部DTO与计时可见性修订已通过核验并保护写回（`protected-slim-heartbeat-integration.json`）：pipeline transition的24个wire字段仍全部校验，解析后只保存prepare消费的16项，取代原30项内部字典；已验证store ID只转换一次。T=8/32/128实际保留键240/960/3840→128/512/2048；独立旧retention oracle的outer/legacy/two-quantity额外dict copy实测4→0，selector的必要detached copy仍1。44方法首轮43PASS/1新计数探针把必要keys比较误计为copy；root改为真实key迭代计数后仅该失败项PASS，生产代码不变，完整wire/canonical/replay/坏输入原子性检查保持。

当前真实12prefix回放4.086243秒、采样峰组RSS478,949,376B（约457MiB），仍62,404 typed事件/11末事件tree、59,598 selected geometry/factory、0整tile构造，accepted state/canonical bytes/末事件replay逐帧精确一致、水位103390。已存P2/D3五帧以当前消费者审计0.149671秒PASS；已存真实Online+Bridge完整请求日志的before/after五个prefix全状态/canonical/last replay SHA相同（0.110880/0.112026秒，水位1144），不重跑物理producer。这些为局部功能/内存诊断，不能作为独占性能倍率。

MainWall现输出既有fluid五桶及EventQueue.proceed的累计已完成调用，未新增clock或仿真事件，原5秒节奏与业务progress键保持。仅main source改动后的全target构建8.589137秒、峰901,341,184B PASS；短用例没有触发5秒打印分支，运行时心跳输出尚未实测。标记`native_wall_scope=completed_calls_inclusive`：eq_proceed包含callbacks/bridge/commit，当前打开scope及中止的bridge/commit还未计入；不能用180秒减这些桶宣称native独占时间。

最新完整first-1s性能仍绑定上述增量之前的52项`closure-frontier-interval`冻结源码；本次代码尚无新完整任务计时，`speed_goal_met=false`，工程/近似额外/总20×均未证明。仿真/投影累计1039.595242308秒（约17.33分钟）未增加，以上均为build或T0；真实请求、decode、物理语义、候选/精度、必要ACK和journal不变。

#### 历史正式冻结（52-binding）：closure-slim-ready-native；已由 53-binding closure-preview-context-prepared 取代

该历史正式冻结为`closure-slim-ready-native`（52项绑定，包含ready cache/copy-move fallback、selected cursor/OOO、16-field DTO与native心跳都在冻结源中）。三对九臂自然完成、同工作量守恒审核`PASS_WITH_VERSIONED_EVIDENCE_LIMITATION`；旧v1不具physical ACK/parent fields可观察性。小任务工程中位0.6927878269×（0.6249446585–0.7094891561）、近似额外0.9165469529×（0.8527735484–0.9957789498）、总0.6223067358×（0.6050335851–0.6349725718），该任务均未证明加速，K2在此小任务比K1慢。不能外推整体。

完整first-1s仍保留8requests/3sessions/61550P/1646decode、K2/ppm0/margin20000/eventbudget262144，全候选/fullvalidation/journal/必要ACK/naturalsettle。T4在180.019100863秒TIMEOUT，采样峰组RSS2,993,188,864B（约2.79GiB），最后29deliveries、tick7,051,643,069，仍有业务进展但未自然闭环。配置byte-exact恢复，source bindings测量期间未变；`speed_goal_met=false`，没有完整任务倍率、没有延时/tick外推/旧25分钟臂。已通知用户恢复外部任务。开始前PID-only核查漏了同组新wave actor，gate两次拒绝且无仿真；后续完整PGID核查与用户授权确认，测量前外部组及所有Astra均不存在，没有发送外部信号。preflight日志单列，不计模拟或假装测量。

新native心跳已在真实T4执行。最后已完成EQ.proceed计174.074579195秒/5,033,274calls，是包含callbacks/bridge/commit的外层桶；fluid flush/dirty/service/tail/completion分别0.143664537/0.020981029/0.079879240/0.945085185/0.928935048秒，含式重叠不相加。Bridge96.094982995秒、outerissue21.382138378秒/298,106calls（nestedDIE6.830065812秒，差14.552072566秒），commit1.597950308秒。最后心跳不包含打开中的eq/bridge/commit span，不能把180减closed桶当native独占。WLT fullfill116618/reuse181488、candidate47,727,432、IDcopy36,739,740/membership47,535,960，证明实际快照复用并保留复核；不将callcounts当墙钟倍率。

29个完成Python delivery的handler72.795073277秒、ledger50.181415299秒（typed26.409144602/prepare23.112074294/commit0.257759495）、policy20.198753431秒、readparse5.485849949秒、journal1.814854688秒，均有包含关系，不能相加或跨不同工作进展签发倍率。累计仿真/投影1221.723843094秒（约20.36分钟），309次guard是本轮结束快照，失败/中止包括，build/alg/replay另列。暂停新增整段仿真并回到代码/小夹具：native5百万事件来源及callback重复工作尚在只读核查；不能据次数认定全是DIE或空poll。整体原任务与20×仍未完成。

## 3. 状态披露（《设计方案》§5.6/§8：已设计/已实现/已验证分开陈述）

| 项 | 状态 | 说明 |
| --- | --- | --- |
| T 开关（typed/lru） | 已实现＋单测验证 | 类别严格序、满足即停、未知类别 fallback 披露 |
| E：minimal_layer_groups / legacy_half | 已实现＋单测验证 | legacy 与基底逐字节等价（既有测试回归通过） |
| E：adaptive（k_hide） | **已实现＋单测验证（2026-09-22 C15：运行期事件递推预测器 = adaptive 正式在线实现，F12 三态闭合）** | 解析例 1/17/25 两径一致；递推预测器（`joint/event_recursion_predictor.py`）：统一 N-way 均分时间线（与 C1 每路径份额 / C2 u_port 同源口径；D13 起 C1 并集 divisor_multi 退役）、写回×恢复两方向同推演仲裁、恢复写腿×计算 memory 腿同端口仲裁、候选自致减速单列不回移期限、剪枝两条钉死（逐 rank 逐腿独享速率串行累计下界；D̂ 钉在无候选恢复流量的消费开始时刻，prune vs 全枚举 50 例一致）、未知 ETA/解析不可用保守全 L + 披露；η/γ 因果更新入口 `observe_valid_service_sample`（混合等待默认不可观测；§8.1 单账本 2026-09-30 修复：sh30 joint 装配经 `service_factor_source` 注入后 γ/η 改读 W6 统一服务样本估计器的只读发布值 `PublishedServiceFactorGroup`，`observe_valid_service` 在该装配 fail-loud、M4 时间衰减桥接退役；standalone 装配保持本更新通道——生产无喂入点恒冷启动 1.0）；adaptive_decisions 披露侧车（source/statuses/divisor/stall/η/γ；M4 2026-09-26 起行携可选 `at_ns`——SH 在线面经 `_ensure_capacity` 链路传 tick，缺省 None 保确定性测试） |
| J：选择与代价模型 | 已实现＋单测验证 | 解析近似代价（关键路径合成、因果时域估计）；**在线反馈通道已接线（R15，2026-09-14）**：链路流登记表按逐 shard 全路径登记/完成事件注销（F-B 登记足迹；**D13 起计价经 `CandidatePathNocShare` 每资源/每路径服务视图消费**——旧并集聚合除数退役）、池端口仲裁份额（含 E 内核 r_j，P1）、ServiceFactors EWMA（D14 改 v2.0 §9.2 冻结固定系数 1/16——首样本同规则自 1.0 出发，0<η_sample≤1 窗口外样本记异常跳过；2026-10-03 起 transfer 时间因子经 WorldContext 转为 NoC 链路 η=1/transfer_factor，正式 P/D 前推统一消费；流数竞争仍由资源仲裁承担；§8.1 单账本 2026-09-30 修复：纯列车服务样本唯一更新点 = W6 `ServiceSampleRecorder`，prefill/decode 因子生产值 = `ServiceFactors.unified_source` 发布视图在 flush（JCM 构造点）刷新的只读发布值，`observe_prefill`/`observe_decode` 在发布模式 fail-loud（对拍期双账本并行退役）；样本纯度排除 joiner 迁移/partial 恢复/copy 门控传输（remote-read 读流列车排除项已随 2026-10-14 远读退役删除）——复审 M5）；decode 负载标定在线化（N12：session 均值 → run 均值 → 冷启动 32 前向步（v2.0 式(4) 冻结先验，D14；每会话 (count,sum) 常数空间存储），全 trace 均值常数不再进决策输入；估计器版本入快照缓存失效键——复审 M3）；merge 段增量计价（复审 K4：input + 因果 decode 增长，不随基础历史膨胀；REMOTE 基走池端口口径——自查 C）；准入重试键 = KV 纪元 ⊕ 失败候选集纪元（复审 M2，R2.5 口径）；无 oracle（`RequestView` 结构性不接受 decode_length/final_context_tokens）；**D01 选择器保留与阶段扩展（2026-09-26）**：全实例×二动作（stay/copy）argmin（v8 two-actions 2026-10-15 起，动作空间沿革四→三→二；v2.0 C07 现状即无组合搜索，不重写新优化器）；初始平局序沿革＝v8 及以前固定序 `(cost, (instance_index, ACTION_ORDER))`（跨实例平局恒坍缩编号最小实例——空闲期放置确定性堆积实证见 PROVENANCE 2026-10-20 节），**v9（2026-10-20）起 joint 主臂 P 发射点跨实例平局改为按实例编号全局环序轮转分摊**：最优 cost_ns 整数相等且并列实例 ≥ 2 时，自游标起沿实例全集升序环序（越最大值回绕）取首个在集实例（`joint_scheduler.joint_tie_rotation_pick` 纯函数）；单实例平局/唯一最优不读不推游标、行为与旧固定序恒等；实例内 stay/copy 同价并列仍由 `ACTION_ORDER` 终裁；顺序参照（load-first/affinity-first）与 face_static 对照臂固定序不变；D 边界式(1c) 继续优先键不变——P 放置分摊与 D 继续优先为两层正交平局语义（`select_decode_target_and_action` 永不读不推游标）；游标 = `Sh30OnlineScheduler._joint_tie_cursor`（`__init__` 置 0、run 级生命周期无中途复位），经 `select_instance_and_action` kw-only 纯函数参数 `tie_cursor` 注入（选择入口不读写调度器状态——L2 批内缓存与 VERIFY 影子双跑前提）；推进点 = P 准入"最终有效选择"处（sh30 `_consume_joint_tie_rotation`，强制钩子之后、容量事务之前；判据 = joint 模式 ∧ 非 probe 强制 ∧ 最终候选表上 chosen 为最优价且并列实例 ≥ 2）；两处镜像同构（`joint_scheduler.select_instance_and_action` 主臂 ＋ sh30 `_quota_reselect_feasible` joint 分支含替身缺省臂，共用同一纯函数）；决策行披露三键 `joint_tie_cursor`（本行消费游标值）/`joint_tie_instances`（最优价并列实例升序）/`joint_tie_rotated`（joint_admission 成功行/joint_admission_failed 失败首行，wait 折叠行无）；离线 `slo_tools/domain_metrics.py` 重放轮转感知（三键在场且 rotated 时比较基准 actual_ref = 本行游标＋候选表现算的轮转期望；键缺席 = 旧行为逐位不变，旧 run 前向兼容）；`MODEL_SEMANTICS_VERSION` 升 `v9-tie-roundrobin-20261020`（成本函数本体/copy-stay 判据/动作适用性面零改动）；估价器"阶段参数与当前布局参数"落点 = 两阶段入口 `estimate_P_with_continuation`（P 阶段，布局 = 决策时点 SessionKVView）与 `estimate_D_remaining`（D 边界，布局 = PDBoundaryView→BoundaryKVLayout，D18/D19 已落）；两阶段候选 notes 以同键 `decision_stage=P`/`decision_stage=D` 自描述决策阶段（D18 的 `boundary_stage=D` 披露位保留）、两阶段 SelectionRecord `diagnostics["decision_stage"]`（"P"/"D"）随记录落盘——两个时刻的全部实例×动作审计项逐候选可辨阶段；每阶段恰 2N 候选记录（v8 二动作 2026-10-15 起；沿革 4N→3N→2N）、每请求两次线性遍历（初始候选零 D 估价，无 9N² 组合枚举）、继续标记恒归并在 (i_P, action) 记录不作第五动作（式 1c 平局键见 D18 行）。**零新增开关**。单测：`joint/test_d01_selector_stage.py`（2026-10-14 远读退役后 9 用例：初始平局序/3N×2 遍历计数/继续归并/边界平局键键形/阶段披露） |
| 逐层数据/空间就绪递推（D02，2026-09-26） | **已实现＋单测验证** | `estimate_P_with_continuation` 的"历史准备—计算—关键路径合成"三段改为 v2.0 式(12)/(13) 逐层就绪递推（计价修改；不声称全局 P/D 组合最优、不声称未测低开销）：一次性传输按 式(12) 供给链只写它**真实消费的层**的 `ready_data` 门——copy handoff chunk 通道内按层消费序串行（逐 chunk 逐 rank 链 `R_{k,r} = max(R_{k-1,r}, A_dest, S_source) + X_{k,r}`，`copy_handoff_chunk_walls` notes 披露链壁）、池恢复只门控缺失后缀层 [p,L)（stay-PARTIAL/copy-PARTIAL/remote-PARTIAL 同口径；与 NoC 前缀不同通道并行，§11.1）、remote tile 到达与 `_double_buffer_recurrence_ns` 的通道串行 T 序列逐位同源（`remote_read_tile_arrival_ns` 闭式，多遍扫描取末遍=保守）；聚合 prefill 计算按固定层数均分（余数前置、总和守恒）为固定数量算子段，式(13) 递推 = 新增纯算术 **`estimate_first_forward`**（前/后段=0——层数据门覆盖整段份额，投影前段重叠属未建模保守项登记；O(L) 不随 token/tile/步数增长）；repeat 段统一入口 = 新增纯算术 **`estimate_repeat_segment`**（零供给退化为纯计算链；`_remote_double_buffer_leg_ns` 委托之，值逐位不变——D09 手算锚不漂移）。**验收（v2.0 §11.3）**：110 ns 例（第二层晚到 100、计算 10 ⇒ max(1,100)+10）通过；prefill 结束前所有需要层就绪；旧整体 max 合成三种形态（`first_credit + max(remaining_stream, compute)`、`max(尾 chunk 流, 计算)`、`max(prefill 腿闭式, suffix_restore)`）全部废除、无开关——数据到达后的计算不再被吸收（旧口径吸收量为登记的预测低估）。零门控动作（stay-LOCAL/recompute）递推 = 整段计算（守恒锚，逐位不变）；target_wait/eviction 串行前置口径已随 **D03 统一时间轴（2026-09-26）**废除（见下行）；`estimate_D_remaining` 组合式已随 D03 改为 `max(Q, A) + 准备腿`（迁移/credit 腿自加入列车发射）。审计披露：C5 冻结字段不动；notes 增 `prefill_first_forward_ns`（递推结果），`copy_handoff_first_chunk_ns` 语义保留、`copy_handoff_remaining_stream_ns` 由 `copy_handoff_chunk_walls`（逐 chunk 链壁列表）替代。**零新增开关**。单测：`joint/test_d02_layer_readiness_recurrence.py`（20 用例：110 ns 例/整体 max 对照/式 13 三段门/零门控守恒/逐 rank max/到达闭式 vs T 递推网格/repeat 段锚/copy 末 chunk 到达后计算保留/TP 偏斜传播/stay 池后缀仅门控消费层/池速率单路径穿透）+ 重推导锚 `joint/test_joint_credit_pricing.py`（D02 递推手算，D03 时间轴后 3308/3466/407）、`joint/test_joint_remote_read_staged_path.py`（2776/2420/2754/4406）、`joint/test_joint_shard_pricing.py`（copy 204） |
| Q/驱逐/准备统一时间轴（D03，2026-09-26） | **已实现＋单测验证** | `target_wait_ns` 降级为**观测字段**（C5 冻结 breakdown 位保留；不声称全局 P/D 组合最优、不声称未测低开销），不再被所有动作机械前置相加——Q 只门控计算（式(13) 首层 `F_0 = Q`，经 `estimate_first_forward(start_ns=Q)` 折入递推）；空间可用时刻 **A** 成为相对本次决策的接口（`_space_available_ns_estimate`，取代旧 `_eviction_wait_estimate` 时长串行前置与 K5 `covered_wait_ns` 抵扣；**D11 起该接口升级为 `_space_ready_ns_estimate` 分阶段足迹形态——见 D11 行**——`A = max(写回完成, 深缺口活跃排空 running+active)` 绝对量，无串行加法即无被覆盖分量双计）：式(12) `A_dest` 门控全部目标写腿（copy chunk 链/池恢复/tile 暂存流自 A 起发射）、式(13) `A^work` 门控计算投影。**发射 frontier 核对（验收红线——不能仅改预测器制造不存在的重叠）**：P 准入三腿在真实执行侧均**决策时刻**发射——`reserve_request_capacity`/`prepare_prefill`/prefill 前缀读流规划以 `at_ns=now` 提交、流表准入相登记、请求其后才入队（sh30_online_scheduler 准入链），构图中 chunk 0 在准入主链、尾块旁挂支链流式到达（graph_batch_builder，无"先整份搬运后计算"串行段）——故 P 准备腿自 0 起估、与排队重叠是真实执行顺序；D 边界**迁移/credit 腿随加入列车发射**（pending_decode_ready → 下一列车，"迁移腿在加入列车才发射，此刻零字节移动"）——自 `max(Q, A)` 起估、不与 D 等待重叠（不虚构）；**边界逐出**（阶段预约事务＋换址半边）决策时刻旁路支链即时——A 自决策时刻起、与 D 等待重叠（真实）。旧 `Q + max(history_prep, eviction_wait)`（D 入口）与 `frontier = Q + eviction_wait` 前置（P 入口）废除、无开关——旧 max 把依赖驱逐后才能写目标的腿提前到尚无空间时刻、串行前置把与排队/写回的真实重叠双重计费。**验收（v2.0 §22-D03）**：Q=100、空间和准备在计算前已可用的例子不再重复支付准备时间（restore 500/Q 100 ⇒ cost 544，旧串行 644）；依赖驱逐后才能写目标时不早于 A（restore ready = A+500 = 802，非 max 形态 500）。merge 腿容量准备在 merge 控制点才发射（无更早真实 frontier）——串行保留不重叠。notes 披露：`space_ready_ns`（A）、`prep_issue_frontier`（各腿发射 frontier 说明）。**零新增开关**。单测：`joint/test_d03_unified_timeline.py`（10 用例：Q 观测字段守恒锚/发射 frontier 披露/Q×恢复重叠/copy 链排队吸收/A_dest 门控/深缺口绝对排空/D 写回×等待重叠/D 深缺口无双计/D 迁移腿不虚构重叠/D 迁移写等空间）＋改锚 `joint/test_joint_credit_pricing.py`（3308/3466/407）、`joint/test_joint_fixes.py`（A 接口组件锚）、`online/test_joint_k_batch.py`（深缺口绝对排空）、`joint/test_joint_remote_read_staged_path.py`（4406 退化锚口径） |
| merge 目标 HBM 写腿（D05，2026-09-27） | **已实现＋单测验证** | `_shard_leg_ns` 的目标写腿条件自"仅 read/copy/staging"扩为四 kind 全量在——kind="merge" 同付**源读＋正确方向 NoC＋目标 HBM 写**三腿 max（v2.0 C02/§15.2 式 22"每次 payload 仍包含源 HBM 读、正确方向 NoC、目标 HBM 写"；计价修改：不声称全局 P/D 组合最优、不声称未测低开销）。与执行侧同腿型：sh30 `_register_transfer_flows` 对 noc_migrate 恒登记双 HBM 端点（目标端口 exec 写腿对 merge 落点写恒有）、`hbm_port_flow_registry` 模块注记同面——旧"merge 无 exec 写腿"计价（C1 期落字）与执行腿型失配废除，**无开关可选项**（直接替换规范）。落点空间准备是另一前驱：仍由调用方空间接口单列计价（home/exec_merge 前缀；D03 期 `_space_available_ns_estimate`，**D11 起升级为 `_space_ready_ns_estimate` 分阶段足迹形态——见 D11 行**），容量充足不令写腿消失——写腿是带宽服务、空间等待是容量前驱，两者正交。目标端口除数 = 各自方向路径末端 rank（正反向独立路线集合不变，K5 口径）。**控制确认（metadata/commit 段，式 22 T^commit）**：本协议真实存在的确认 = 图侧 merge 尾标记节点（每 decode rank 1 op/1 B COMP，链在本 rank merge 传输链后）——零载荷字节、无 NoC 流，服务时间低于整数 ns 粒度且无独立服务率来源 ⇒ 不虚构控制字节时间；其结构位置（提交在载荷之后、整轮串行殿后）已由 merge 段合成式如实保留，真实控制载荷协议若引入须在该段补计。link_quota δ_adm 冻结值 0 注记同步（合并腿改三腿 max 后"I/有效带宽" ≤ 已计价合并腿，正 δ_adm 仍属重复计费，C10 决议不变）。**验收（joint修改文档 D05）**：目标 HBM 慢 NoC 快时 merge 反映目标写瓶颈（原语层 600→2400、估计层 merge_ns=3600 且 reverse 披露 15000）；容量充足不令该项消失（双侧 10^9 空闲、无 home/exec_merge_eviction_writeback_est 注记而写腿在场）。**零新增开关**。单测：`joint/test_joint_shard_pricing.py::MergeTargetWriteLegTest`（3 用例）；既有金样全部未漂——成因：既有夹具 hbm=1e9 快于 noc 或 port_registry=None（u=0 时写腿与源读腿同值不改 max）。
| 候选计算与候选传输共用 HBM 阶段份额（D06，2026-09-27） | **已实现＋单测验证** | 本地算子求值器接受候选阶段 HBM 份额（v2.0 §9.1 式(6)/§22-D06；计价修改：不声称全局 P/D 组合最优、不声称未测低开销）：face_scheduler `estimate_iteration_time_ns`/`estimate_prefill_task_load_ns`/`estimate_decode_remaining_task_load_ns` 新增 `hbm_share_divisor`（int ≥ 1 fail-closed）——roofline 的 HBM 带宽项按 1/份额缩放（式(8) 的 M/μ_HBM 项），峰值 FLOP 项与时延项不动；缺省 1 = 独占带宽（台账/排队负载等非候选计价路径零漂移）。JCM P 入口（`estimate_P_with_continuation`）按动作建立候选阶段并发流集合 k_exec = 计算读 1（同一计算流先后读权重和 KV 不拆成多条流）＋ copy 留存写 ＋ pool 恢复写 ＋ remote 暂存写——k 按通道/缓冲协议并发界计数（copy 通道内按层串行链 = 1 条流、remote 双缓冲同时 1 块传入 = 1 条流、池恢复逐 rank 1 条流；串行通道上排队的层组不重复计），经 **`CandidateStageHbmShare` 只读叠加视图**（`joint/hbm_port_flow_registry.py`：`divisor` = m+(k−1) 保持消费方 +1 口径、`stage_denominator` = m+k 供计算求值、`snapshot()` 审计）传入该候选全部计价腿（copy 前缀/chunk 串行链/池恢复/remote 读流族含首 tile/余量流送/供给基元）与计算求值——同一端口上计算读、copy 写、pool 恢复写、remote 暂存读写**同一分母 μ = B/(m+k)**，不各自独占整端口。`HbmPortFlowRegistry` 继续描述在册工作（m_e = 活跃 decode + 在册传输/远读），候选临时占用另加、**不污染真实登记**（视图从不 register/unregister，阶段结束只有预测视图失效；"预留份额近似"——阶段内暂时空闲的流不退出分母）。`prefill_task_load_ns_fn` 契约升级为第三关键字参数 `hbm_share_divisor`（JCM 恒传；SH `_joint_prefill_total_load_ns`/`_prefill_chunk_task_load_ns` 前传且入 memo key；线性回退路径无带宽/FLOP 分解可分摊，照旧独占口径——夹具/离线登记项）。阶段裁定：P 初始准备与 P 计算（决策时刻发射、逐层递推下真实并发）合成一个冻结占用集；P→D 转换（迁移腿）与最终 merge 为独立冻结阶段且各端口单流（k=1）⇒ 旧 m+1 口径不变（D 入口迁移腿/merge 腿计价不动，`estimate_D_remaining` 锚钉测）；D 本地/远读服务阶段的 decode 计算份额随 D08 上下文模板接线（decode 计算当前为 per-token 标量，无逐调用份额接口——登记项）。**D 边界 credit 腿同分母收口（2026-09-28，v3.2）**：D 边界 remote 候选的 decode credit 暂存腿——双缓冲闭式供给基元/tile x（单源 `_remote_double_buffer_leg_ns` 与多源 tile x 求值同面；闭式 `start_ns=move_done_ns` 即暂存供给与 decode 计算同端口并发）改走 decode 阶段冻结份额（k_e=2，与同阶段 decode 计算读同 m_e+2 分母；修复前裸注册表 m_e+1/离线独占，暂存腿偏快 ⇒ remote D 边界候选偏乐观）；离线无注册表口径同面 k=2（与 P 入口离线腿同族）；P→D 迁移腿（copy 三腿）仍独立冻结阶段 k=1（m+1）不动、`StageScopeAnchorTest` 锚不动。notes 新增 `hbm_stage_share=exec_flows=…(compute+copyX/poolY/stagingZ),compute_denom=…` 披露键。**验收（v2.0 §22-D06）**：①单端口两条长期满载候选流（计算读+copy 写）各得 B/(m+k)，隐含带宽和 = B ≤ 峰值（旧各自独占口径虚报 2B）；②copy 尾流隐藏在计算后（交接墙钟 ≪ 计算段）仍能看到共享资源减速（计算段随 k=2 放大，首前向递推可辨）。**零新增开关**。单测：`joint/test_d06_hbm_stage_share.py`（现 28 用例，含 2026-09-28 收口钉测 `DboundaryCreditLegStageShareTest` 5 用例：离线/m_e=2/计算-腿同分母/多源四形态）；受锚值更新（旧"传输腿各自独占 m+1"按式(6) 修正为 m+k）：`test_hbm_port_flow_registry.py` A4' 写腿（400→800/1600→2000）、`test_joint_shard_pricing.py`（per-step 平价锚改共享份额锚、端点腿绑定 10+500→10+1000）。 |
| decode 服务按上下文增长＋固定 8 区段折叠（D08，2026-09-27） | **已实现＋单测验证** | decode 不再是固定单 token 时间乘步数（v2.0 §13.2 式(16)/(18b)；计价修改：不声称全局 P/D 组合最优、不声称未测低开销）：`decode_ns_per_token × 步数` 接口扩为**按当前上下文求值的固定成本模板** `decode_task_load_ns_fn(z, hbm_share_divisor)`（z = 该前向 KV 可见上下文，d_token=z；式(6) 阶段份额分母由 JCM 按候选阶段传入——D06 预登记的"D 本地/远读服务阶段 decode 计算份额随 D08 上下文模板接线"登记项随本卡落地：P 延续/D 边界各自独立冻结占用集，k_e = 计算读 1 ＋ remote credit 暂存 1）。两入口现场取上下文/时域：P 默认延续 H_z = H+U × n̂_D、D 边界 H_B × n̂_B（式 18b 的 H_z 口径；SH `_joint_cost_model` 构造期不再传 `decode_context_tokens`/`decode_average_length`，D 边界同）；剩余 n̂ 步总量 = `fold_repeat_service_ns` 的 K=min(8,n̂) 连续尽量等长区段**中点折叠**（τ((l_j+r_j)/2)，区段端点 ⌊(j−1)m/K⌋+1/⌊jm/K⌋、Σw_j=m 守恒；K=8 为固定数值精度预算非调优权重；m ≤ 8 逐次求值对该服务模型精确；中点非整时 τ 在 ⌊z⌋/⌈z⌉ 线性插值——仅解析估值）——估计器内不循环 n̂ 次，τ 求值次数 ≤ 2K 与 n̂ 无关（n̂ 增大数千倍调用次数不变，钉测）。**字节总量与空间峰值不经折叠**：远读 = 式(18a) 基数×遍数、增长 = 每 token KV 字节×n̂、累计逻辑上下文 = 式(18b) `decode_context_weighted_tokens` 整数闭式（notes 披露位 `decode_ctx_weighted_tokens`；KV 读量守恒钉测——模板在场与否 remote_read_ns/驱逐账/merge 计价逐位不变）。prefill 长 chunk 序列同规则：SH `_joint_prefill_total_load_ns` 首 chunk 逐值（恢复启动与稳态分开，§13.1）＋其余整 chunk K=min(8,m) 折叠＋尾 chunk 单独计（执行侧列车 span 冻结/核销仍逐 chunk 精确——两账分离，式(16) 只约束候选估值）。旧 C3 "horizon 逐步求和折回 per-token 均值、构造期传上下文"机制直接退役（其 O(n̂) 循环挂构造处、per-token 均值无法携带候选份额；无开关可选项）；`decode_ns_per_token` 标量位 = context=1 单步名义率（模板缺省的夹具/离线回退锚，模板在场时不消费——N4 prefill 名义率同族，直接构造的测试夹具零漂移）。notes 新增 `decode_ctx_fold=base=…,stride=1,K=…,share=…,ns=…` 披露键。**零新增开关**（`decode_task_load_ns_fn` 为构造注入位，非 env/CLI/配置键）。单测：`joint/test_d08_context_growth_fold.py`（17 用例：式 16 区段守恒/线性精确/式 17 诊断区间/调用界/式 18b 闭式/两入口长短上下文差异/标量回退零漂移/KV 读量守恒）＋ `online/test_joint_decode_context.py`（重写为 D08 语义，12 用例：模板注入与单前向上下文敏感/m≤8 折叠==逐步总量/冷启动 32 的式 17 区间/调用界与 memo 有界/prefill chunk 折叠四钉/式 18b）；键形同步：decode memo key 扩 `hbm_share_divisor`（缺省 1，`test_sh30_task_load_snapshot.py` 键形随附）。 |
| **四动作 J 正式计价切换为事件前推（W3，v4-projection-20260929，2026-09-29）** | **已实现＋单测验证** | `estimate_P_with_continuation`/`estimate_D_remaining` 内部 = `joint/action_work_graph.py` 构图（四动作节点集 §6.2 + P 参考接续 + consolidate 结算腿 + 控制延迟节点 + §7.2 继承义务解析）→ `joint/resource_projection.py` 事件前推（W2 内核）→ `cost = predicted_settle_done`（service/settle 分列；决策相对轴 t=0）。选择器面/签名/loads 键集/breakdown schema 不变；审计披露（notes/remote_read_ns/merge notes）仍走保留原语。世代退役（无开关、旧代码删除）：Q 冻结（target_wait 只作计算门 = 首层计算腿 ready_at + 观测字段）、K=8 折叠（decode 服务 = 逐步精确 Στ(z_k)；`fold_repeat_service_ns` 保留为登记的局部近似算子——SH N4 prefill 求值器消费）、阶段冻结份额（`CandidateStageHbmShare` 只作静态审计视图；端点腿实际份额 = HBM 资源活跃集动态计数；2026-10-02 v5 起计算腿也以纯计算/HBM 工作量动态前推，缺显式分解的夹具标 `aggregate_service_equivalent_hbm_profile`）；`estimate_first_forward`/`estimate_repeat_segment`/双缓冲闭式降为审计/参照算子。`MODEL_SEMANTICS_VERSION` 升 `v4-projection-20260929`。WorldSource：`LoadsWorldSource`（注册表计数 → unknown-ETA 背景腿，inferred）/`LedgerWorldSource`（W3 交付时为占位，W5 起正式消费账本剩余量）。`distributed` 结算零载荷 settle 腿（W9 目录提交型已实现——本层只改 settle 腿形态）；frozen_diagnostic A 臂 = 本层预冻结（资源峰值 ÷ 决策 tick 活跃集、背景移出——不调旧实现）；预算超限 fail-closed raise（不返回截断时刻）。sh30 `_joint_cost_model` 注入 `prediction_config`（`JOINT_PREDICTION_CONFIG`，W0 面）。单测：`joint/test_action_work_graph.py`（21 用例）+ `joint/test_joint_projection_pricing.py`（16 用例）+ 重锚 d02/d03×2/d06/d08/d17_schema（义务依赖语义升级）/credit_pricing 1 恒等式。 |
| 分阶段足迹＋容量驱逐摘要（D11，2026-09-27） | **已实现＋单测验证** | 空间预测从"一个 KV 向量"升级为**分阶段峰值 × 逐出摘要**（v2.0 §14 式(19)/(20)；计价修改：不声称全局 P/D 组合最优、不声称未测低开销）。新增 `joint/capacity_summary.py`：`PhaseFootprint`（准备/prefill/decode/merge 四阶段逐 rank 新增峰值，式(20) 逐阶段对照 F 检查 Δb=max(0,peak−F)、不机械相加成"全程同时驻留"；P 入口分解 = 池恢复瞬态+remote 暂存（准备）→ final−growth（prefill）→ final（decode/merge），D 入口 = 迁移到达→+growth；merge 腿容量准备语境 `PhaseFootprint.single` 单阶段声明）与 `CapacitySummary`（按 **T 类别 × E 易释放后缀/其余层 × 写回端口/路径归属** 分桶，逐 rank R/干净/脏/原子数/最大原子向量；同桶释放原子同一逐 rank 分片比例——KV 字节对 token 精确线性 ⇒ gcd 归一签名，比例不同必须分桶，不把耦合 TP 分片拆成各 rank 独立 victim）。`KVCacheManager` 在**完成/激活/引用变化/真实释放**事件增量维护摘要（挂 `_check_invariants_after_mutation` 咽喉点 + 阶段引用两入口；最大原子经逐值多重集精确回收；strict 全量对账兜底），候选**只查询**冻结快照（O(桶数×rank 数)，不随已完成 session 累计数增长——哨兵钉测）；`_ensure_capacity` 的真实 T/E/victim 决策保留原位。JCM `_space_ready_ns_estimate`（取代 D03 期 `_space_available_ns_estimate` 单向量接口）按固定桶序以 **θ 份额近似消费**缺口（θ_b=min(1,max_r d_r/R_b,r)，不能覆盖任何缺口 rank 的桶 θ=0；预计脏写回 θ·D 计式(7) 写回腿，预计事务数 = θ·原子数作启动费用近似且同端口并发流按事务数共享——池热点按实际共享计价）；**干净对象零写回费**（本仓执行协议容量逐出恒经 remote_store、账面忽略池侧陈旧副本 ⇒ 驻留字节无有效 backing、clean 恒空——由池 backing 交叠真实推导非硬编码，分支以摘要构造 API 钉测）。深缺口：active 工作因果结束估计（running+active 排空时刻）有限近似＋`capacity_wait_uncertain` 注记（取代旧 `deep_gap_unresolved`；非可执行保证）。**不将未来 victim 恢复费用加到当前请求**（§14.4）。`InstanceLoadView` 新增 `capacity_summary`/`capacity_ledger_by_tp_rank` 可选字段：生产 SH `_joint_cost_model` 恒注入真实快照与逐 rank 台账 (C,W,O,Z)（F=C−W−O−Z 失配 fail-closed；Z→O 转账守恒钉测）；缺席 = 夹具退化锚（reclaimable 单桶合成，同一 θ 公式）。**零新增开关**。单测：`joint/test_d11_capacity_summary.py`（25 用例）＋改锚 `joint/test_d03_unified_timeline.py`、`joint/test_joint_fixes.py`、`online/test_joint_k_batch.py`、`joint/test_joint_fix1_pricing.py`、`online/test_joint_preadmit_visibility.py`、`joint/test_joint_review2_fixes.py` |
| 至多三位置确定性合并（D12，2026-09-27） | **已实现＋单测验证** | `project_end_state`（v2.0 §15.1 式(21) 轮末**有限位置 ≤3** 摘要投影：权威区间唯一内容去重——交接重复副本不增加 V、两个权威区间 token 交叠 fail-closed；候选先行准备迁入 D 目标的区间权威随之转移——增量型切换 U、完整型 copy 的 context；基础区间 role=base 权威留在源端照旧远读；decode 增长为新内容区间）＋ `JointCostModel.merge_from_layout`（返回**固定目标**、各来源缺失区间、通道/引用义务、目标空间（service_done 投影占用的净新增，`_space_ready_ns_estimate` merge 控制点单列计价）——完整并集位置直接作为结果位置，否则保留量最大者，平局**边界逻辑 home ω_B → D 实例 → 实例编号**；**不搜索目标速度最优**；容量回退按同一保留量降序＋固定平局次序至多两次（`merge_capacity_fallback_order`）、首笔载荷提交后锁定（`merge_capacity_lock`），决策时点按主目标计价、运行时容量兜底翻转不可计价）。P 入口与 D 边界结算两处 home/exec 双标量 merge 块**整体替换**为该内核（单一实现、**零新增开关**）；两位置退化与既定 merge v2 少并多逐位一致（byte_le_tie_forward 平局含等号、`merge_v2_forward_ns`/`merge_v2_reverse_ns`/`merge_to_home` 双腿审计面保留、D05 kind=merge 三腿与 K5 反向自身方向路线不变）；**P 阶段 action 名不再决定合并方向**（§15.1：合并依据 D 结束的实际/预测区间 owner——copy/recompute 完整并集在执行端即零载荷，home 更大不回搬）。`PDBoundaryView` 新增 `logic_home_instance`（ω_B 式(21) 平局首位，不等于数据所在地；生产 `_joint_boundary_view` 注入逻辑 home，与物理基础驻留实例 D-BI 分立；缺省退回 base holder）。运行时 `merge_back` 曾不在 D12 计价批改动范围，**2026-09-27 修正轮 C4/C5 已落地**：`face_scheduler._merge_back_three_position` 为式(21) 运行时实体实现（并集优先零载荷/保留量最大/平局 home→D 实例→编号/容量按序回退；含 C5 补缺——回退候选覆盖全部持有位置、recompute@基础驻留端重合豁免与重复副本零载荷释放），钉测 `online/test_c4_boundary_full_surface.py`（三位置合并两用例）＋ `joint/test_d12_merge_three_position.py`。不声称全局 P/D 组合最优、不声称未测低开销。单测：`joint/test_d12_merge_three_position.py`（21 用例：两位置退化/三位置顺序无关/重复副本去重/权威迁移/平局次序 home→D→编号/容量回退与不按速度选点/notes 审计面/ω_B 端到端穿透） |
| 每路径资源份额，去掉吞吐倒推流数（D13，2026-09-27） | **已实现＋单测验证** | `CandidatePathNocShare`（joint/joint_cost_model.py，小型只读**每资源/每路径** NoC 服务视图，与 D06 `CandidateStageHbmShare` 同族）替换单个全局并集瓶颈除数 `priced_divisor`：每条 shard 路径独立求值其实际经过链路的份额 = max(注册在册流, 遥测时间加权流数) + 候选自身经该链路份数（include_self 可剥离）——**单路径拥塞不错误减速无关路径**；零字节 shard 不进自身份额族、共链自身重叠计费保留；链路/路径两级 memo 使同路径/阶段的查询共用结果（E4 去重的每路径形态）。消费点全覆盖：`_transfer_ns_shards`/`_shard_stream_ns_shards`/`_pool_transfer_ns_shards`（`noc_divisor` 参数位改 `noc_share`）、P 入口 copy/remote 主计价腿、D 边界迁移/credit 腿、chunk 链与逐 rank 供给基元；`_union_noc_divisor` 删除；breakdown 冻结字段 `contention_divisor` 改披露视图最坏路径除数（ceil，保守侧）、stay/recompute 无计价腿维持单次注册表调用。**删除 capacity/rate 冒充流数回退**（v2.0 §9.2 不做清单）：`TelemetryLinkFlowView` 遥测腿收紧为流数口径——速率-only 链路 = 流数**显式未知**，除数退在册流近似；`measured_effective_rate` 保留观测/披露通道；披露键 `legacy_rate_divisor_links` → `unknown_flow_count_links`（SH 镜像 `telemetry_degradation.legacy_rate_divisor_entries` → `unknown_flow_count_entries`）；流数优先与旧流 max 去重保留（不重计）。**复用在线逐 shard 路由身份**（C11）：路径族 = `route_paths_fn`（与 `_register_transfer_flows` 的 shard.noc_path 同一 `deterministic_xy_route` 源），链路键与注册表同空间。**零新增开关**；不声称全局 P/D 组合最优、不声称未测低开销。单测：`joint/test_d13_path_share.py`（12 用例：路径解耦手算锚/原语内部构造/共链自身重叠/零字节过滤/遥测流数只抬覆盖路径/旧流与候选不重计/速率-only 显式未知/register_path 路由身份复用/只读快照/memo 共用/披露形态）+ `joint/test_joint_telemetry_divisor.py`（D13 语义重钉：流数口径除数/未知披露/整型 LinkId fail-closed/速率-only 不抬计价）+ `joint/test_hbm_port_flow_registry.py` E4（remote/copy divisor_multi 调用 1→0 严格下降、stay/recompute 单次保留） |
| 决策 schema、传输取整与语义版本（D17，2026-09-27） | **已实现＋单测验证** | 受控验证批的 schema/取整/版本身份（v2.0 §18/D17；零新增开关）。**版本身份**：常量 `MODEL_SEMANTICS_VERSION`（joint_config，现值 `v3.2-2phase-20260928`——2026-09-28 D06 式(6) D 边界 credit 腿同分母收口升版，前代 `v3.1-2phase-20260927` = 2026-09-27 修复轮二、`v3-2phase-20260927` = 两阶段边界首版；本修订含 D09 tile 固定化/D16 有界模板/D10 静默重建执行图/D03 frontier 对齐/D17 遥测基准取整等，remote-read 族成本与执行时序相对前代有系统性变化）随 run manifest（`manifest_dict` 键 `model_semantics_version`）与逐条决策日志（`joint_admission`/`decode` 行及 `boundary_decision` 记录）落盘——**语义改变必须升级该版本号，旧实验不自动代表新模型，旧结果文件与旧 SHA 原样保留、不重命名为新协议结果**。**schema 扩展**：`ActionCostBreakdown` 由 C5 冻结 11 字段扩至 28 字段（公式版本、决策阶段/纪元、两阶段实例/动作配对、继续标记、剩余前向、预测里程碑 `predicted_next_token_done`/`predicted_service_done`/`predicted_merge_done`、当前区间布局、继承义务 ID、分阶段峰值空间、暂存峰值、近似状态；全部带缺省=直接构造夹具零漂移）；`_JOINT_BREAKDOWN_LOG_FIELDS` 同步扩 28、嵌套元组递归转 list（JSON 可重建）；决策行新增公式版本/阶段/纪元/两阶段配对/剩余前向键与逐候选 `measured` 真值槽位（未执行候选恒 NA——§19.2，真值只来自受控同状态执行，不用预测值冒充实测）；`estimate_D_remaining`/`select_decode_target_and_action` 新增 kw-only 纯披露参数 `decision_epoch`/`inherited_obligation_ids`（不计价、不改选择）。**取整口径**：传输函数带宽服务项由 `int(bytes/rate)` 向下截断改为**向上取整**（`_transfer_ns`/`_shard_leg_ns`/`_pool_transfer_ns_shards` 及 `_transfer_ns_shards`/`_shard_stream_ns_shards`；恰整除零漂移），零字节无事务严格为 0（F1 启动固定成本契约不变）。验收：可从日志重建一次候选计算的公式版本、输入来源、字节/空间与关键等待。单测：`joint/test_d17_schema_version_rounding.py`（20 用例）+ `online/test_joint_decision_schema.py`（冻结锚 28 字段重钉与决策行 D17 键）+ 改锚 `joint/test_d03_unified_timeline.py`/`test_d06_hbm_stage_share.py`/`test_d08_context_growth_fold.py`/`test_d18_boundary_decision.py`/`test_joint_shard_pricing.py`/`test_joint_credit_pricing.py`/`test_hbm_port_flow_registry.py`/`online/test_joint_m_batch.py`（旧截断数值锚按 D17 取整口径重钉，不按旧数值逐位一致阻止语义修正） |
| home/merge 账本事务 | **merge v2 已实现＋单测验证（2026-09-17 需求②）** | 少并多方向裁决（两侧保留量比大小、小侧整份搬大侧——min 双向计价与物化同判据）；零池写（I6：merge 事务不产生本会话 remote_store）；结果恒全层 LOCAL@胜者、home 迁移胜者（全仓第二个 home 赋值点）；copy/recompute 反向零字节翻转（零传输、无空间准备）；REMOTE 基就地保留（裁定③）；空间准备只走统一 T+E `_ensure_capacity`（R4 自降级/k=0 池归并兜底**退役**——双侧深缺口才 fail-closed，合同变更登记 PROVENANCE；`merge_degrade_events` 冻结、新 run 恒空）；版本键恰好一次（R4/N9 沿用）+ 到达合同 = service_done 锚 + 依赖门控（C3b，见 §2）；容量缺口台账落账边界 = 确认终态（K6 沿用；run 末导 `joint_kv_ledgers.json` 侧车含 deep_gap 台账）；decode 停滞死锁守卫与逐列车增长逐出三路进图不变（自查 A/D） |
| remote-read 执行流（D09/D07） | **已退役（2026-10-14 远读退役）** | 整条执行链（有限字节双缓冲 tile 流、prefill 前缀读流分阶段、`rid#readplan` 承诺账本、远读暂存 lease、decode credit 腿、D07 模板身份闭式）随 remote-read 动作整体删除，无开关保留档；删除范围与被取代章节见仓根 `PROVENANCE.md` 顶部退役批登记节，历史验收原文见该节登记条目。 |
| merge 物理流 | 已实现 | 完成批发射真实回传/池写回流 + merge 尾标记节点（watch 送达 = merge_done 事件侧，决策日志独立披露行——到达锚在 service_done，尾标记转为到达后依赖门控，C3b）；下一轮数据准备经 store-tail 前递补偿等待（合并成本进入下一轮数据等待；全部动作/八组合一致处理） |
| 逐层恢复与 prefill 重叠 | **已实现＋单测验证＋真实路径对拍（2026-09-22 C15）** | 列车级保守门退役为**逐层段就绪门控**（`emit_layer_segmented`：热前缀段 + 各恢复组层段，段 i 只等组 i——跨段字节与整段单次发射逐位一致测试钉死）；restore_group 标记的后缀恢复腿组间并行发射（2026-09-25 P2-R2：组间无边互并发，每组只 arm 与本组层区间交集的 store 条目、每跨缘对恰一次共享 1B 中继、并集条目同批发射内消费——组 0 区间无匹配的 fail-closed raise 面不变；同实例 partial 分支 + 跨实例旁挂支链）、逐组逐 rank 目标 HBM 门；`rid#restore` 区间账本（issue/complete 守恒，drain/merge/mark_complete 三结算边界，双结算 fail-closed；M4 2026-09-26：issue/complete 事件携带可选 `at_ns`——SH 在线面传 tick，配对 = 恢复窗口下界；缺省 None 保确定性测试）；RESTORE_GROUP_LAYERS=8（≤8 层单组 = 旧单笔锚）；completion 残留门 fail-closed。真实路径对拍：stress 夹具 10s/270 请求 partial_copy_hits=78、restore 分解 270 行 hidden_ratio p50=0.9999（旧单笔口径保留为回归锚，非开关） |
| 链路遥测三件（C6–C8） | **已实现＋单测验证＋真 run 对拍** | C++ `--link-telemetry` 每 epoch 导出链路合计字节、活跃时间及时间加权活跃流数 `active_flows`；JCM 计价与 AIMD 分母走**流数口径**（D13：合计速率只作 transfer_factor 观测/披露通道，速率-only 链路 = 流数显式未知——除数退在册流近似，不做 capacity/速率换算；混合缺失拒绝）。JCM `divisor_effective` 将时域 `#readplan` credit 账本与实际并发代表流分开：共享链上先按真正可同时在场的流登记，再和遥测流数合并；SH 侧 LinkId→端点键换算 + `#readplan` 预登记对账核销。既有 2s 窗流数与注册数相等，分叉形态由单测钉死；旧对拍覆盖率 100%、字节覆盖 99.99987%、漏计方向单向微欠（0 过计）、divisor 最大相对偏差 6e-8 仅描述原对拍窗。 |
| 配额（通用流量治理） | **已实现＋单测验证＋冒烟四臂（2026-09-22 C9–C11）** | `JOINT_QUOTA_MODE=off\|static\|aimd`（缺省 off——F7：off 臂决策序列零漂移实证）；动作级准入（链路∪实例 HBM 端口、实例永不掩码、quota_deferred 回队 + wait_reason 三分）；AIMD 收缩/扩张（遥测驱动，膨胀阈值带 + 流寿命 EWMA）；流生命周期借还配对（admits=releases 全配对）；port_snapshot 实测值接通（off 态 NA 保持）；决策时延入档（30s 臂：纯决策段 avg 3.510ms/max 9.031ms；判据层 ≈7% 决策段）；矩阵 `TJE_quota_static`/`TJE_quota_aimd` 两臂常驻 |
| copy 块级交接（C13） | **已实现＋单测验证（23 用例）** | 前缀 NoC 迁移 K 对齐块化（`_cb{k}` 唯一相位键）+ 源端交接即释放（home 释放点前移）；尾块旁挂支链 2026-09-25 P1 起逐笔并行（块间无边、每块 fork 头块后主链 frontier，块内 ack 新形态 ack_recv_c←本块 send_c／ack_send_c←本块 recv_c，gate 逐块重 arm——旧 send 直连链/exec recv 顺序链/ack 链尾随的整段串行删除）；守恒科目 `#handoff`（交接重复量）/`#copy-stream`（一遍历史）入决策日志；`copy_handoff_events` 一行入 run 末 ledger 侧车（C11 落点） |
| remote 完整结算与 kv_delta_journal（C14） | **已实现＋单测验证（14 用例）** | 完成检查清单七项逐项"机制在案⇔测试钉死"（源端结构保护双通道/保护解除时机 = merge+mark_complete 同 tick/少并多按两侧实际保留量/失败分支闭合）；`kv_delta_journal` merge_back 两出口恰一行（方向互斥/零字节翻转/传输字节/两侧保留量/home 迁移轨迹）+ `_on_merge_done` 闭合审计门（watch 交付 ⇔ journal 行，缺行 fail-closed）；C4b-FIX 补 remote-read auto-K 多块 credit 命名缺陷（`_rcb{b}` 后缀，5 用例 + 两族臂 exit 0） |
| face_static 静态对照臂（C17） | **已实现＋单测验证（18 用例）** | `JOINT_SCHEDULER_MODE=face_static`（policy variant，不进八组合）：实例按 `hop ≤ floor(ρ)` 静态距离球掩码、球内逐实例择优；身份纪律（manifest `face_static_identity` 键：本文内部的静态距离变体、不冒称完整 FACE）；F7 唯一耦合例外——强制 `JOINT_QUOTA_MODE=off`（manifest 注记 `quota_forced_off`）；矩阵 `TJE_face_static` 臂常驻 |
| P→D 边界二次决策（D18，2026-09-26） | **已实现＋单测验证** | **强制 P/D 同址废除**（原红线 #4 / `selected = state_index` 钉扎退役）：初始决策 = 式(1a) `estimate_P_with_continuation`——每候选估 P＋默认 D 延续 π0 的整轮、**只提交 P**（初始候选不做全 D 搜索）；真实 P 完成事件触发状态机 `P_COMPUTE_DONE -> D_DECISION_PENDING -> D_PLAN_SELECTED -> D_COMMITTED -> D_RUNNING`（单请求全阶段见 `joint_stage`，v2.0 §16.4）：以真实边界快照 `PDBoundaryView`（恒完整型单族布局——2026-10-14 远读退役起，增量型族随 P=remote-read 消亡）＋共同 n̂_B（式 4a，`CausalHorizonEstimator.estimate_boundary`）做**恰好一次全候选遍历** `estimate_D_remaining`（式 1b），原地继续 x0 恒为候选（归并在 (i_P, action) 记录，`is_continuation=true`，不作第五动作）且**平局优先**——平局键 `(cost, 非继续标记, instance_index, action_order)`（式 1c：只有严格更低成本才切换）；已完成 P 不重复收费、切换迁移腿是 t_B 后新工作不回填 P 期重叠；**已知终止直接结算不选 D**（`decode_length==0` 即首 token 满足停止条件的回放协议形态，`P_COMPUTE_DONE -> MERGE_PENDING`；无退出列车 → completion 触发门锚 drain 列车 seg1 块末，`emit_completion_batch block_end_stage="seg1"`——P 完成即本轮全部计算完成，语义精确）；D 提交（第一笔不可撤销操作=预约换址）后无第三次选点。执行侧配套：跨实例预约事务（C15 重设计；**2026-09-26 D20 起该两段式 `move_request_capacity_reservation`/`revert_...` 调用口整体退役替换为幂等 epoch 的 `commit_stage_reservation_transaction`/`abort_stage_reservation_transaction` 单一事务面——见 D20 行**；§16.4 提交失败回退继续态，`boundary_commit_fallback` 披露行）；`decode_hbm_feasible_instances` 死代码随钉扎删除；切换候选 = copy 迁移（`move_prefill_to_decode` 真实 NoC 迁移；增量型 remote-read 换执行实例形态已随 2026-10-14 远读退役删除）——**边界可选面曾为协议裁定的两族子集（2026-09-26 修缺登记），2026-09-27 修正轮 C4 已翻转为完整动作面；2026-10-14 远读退役后恒完整型**：边界 recompute（静默重建）/边界基材料化 copy（@i_P 基材料化、@第三目标两腿）的执行钩子全部接线（多源 remote-read 钩子已随远读退役删除）（计价 `estimate_D_remaining` 补 recompute/copy/纯多源闭式、KV 账本 `commit_boundary_decode_transition`＋`_merge_back_three_position`、在线调度 drain 按动作分派＋rebuild span/首 token 恰一次、构图器多腿槽位），旧协议裁定串 `BOUNDARY_RULED_OUT_REBUILD/_BASE_MATERIALIZATION/_MULTI_SOURCE` 随之删除（适用面重写为区间缺失物理语义），掩码臂（`mask_recompute`/`mask_copy_recompute`；`mask_remote_recompute` 已随 2026-10-14 远读退役删除）冒烟真实执行验证全 exit=0，钉测 `online/test_c4_boundary_full_surface.py`（13 用例，含 recompute+load-first 全链路）；同状态对照（验收纪律2）经 `run_scripts/joint_same_state_probe.py` 受控执行实测（不再真值 NA）；`select_decode_target_and_action` 边界规则为协议级（不随调度模式变体——J-off 臂差异只在初始选择）；决策行披露 `decision_stage`/`joint_decode_action`/`is_continuation`/`boundary_decision`（含 n̂_B 与全部边界候选代价）。单测：`joint/test_d18_boundary_decision.py`（2026-09-27 修正轮改写后 19 用例：沉没纪律/迁移新工作/候选面/不适用理由=物理语义钉测＋掩码×边界钉测（C1）/式 1c 平局与切换/恰好一次遍历/终止直结/继续提交）＋ `joint/test_joint_fixes.py` 预约事务钉测（换址/容量失败保留继续态）＋ `online/test_decision_route_serialization.py` noc_migrate 全链序列化 |
| D 候选专用边界视图（D19，2026-09-26） | **已实现＋单测验证** | `joint/boundary_layout.py`（不反向 import JCM，单向依赖）：`BoundaryKVLayout`——从 `PDBoundaryView` 派生的区间级因果布局（恒完整型单 context 区间——2026-10-14 远读退役起，增量型 base+increment 两区间族随 P=remote-read 消亡；权威归属、片上位置 ≤3 界、继承义务表、首输出 pending token 与终止状态），逐目标派生 **D copy 只补目标缺少部分**（D 远读只数远端基础区间的口径已随 2026-10-14 远读退役删除）；`StageWork`——v2.0 §13.1 两决策点工作表接口（D 边界零新 U-prefill；D recompute = H_B **静默重建**不输出/不采样/不推进 RNG；长度入口归一为剩余前向 n̂_B；首 token 只消费一次——二次消费 fail-closed）。`estimate_D_remaining` 字节口径改由布局逐目标推导（远读基数/增量区间 U/残余远读三项口径已随 2026-10-14 远读退役删除，现恒完整型）；第三目标 copy 补 H+U 与已承诺接收端排除（义务表 `committed_transfer` 命中即不再发送）在布局接口给全并钉测——多源 remote-read/增量型 copy@第三目标两形态曾于 2026-09-27 修正轮 C4 接入运行时可选面，**已随 2026-10-14 远读退役删除**（`boundary_two_source_remote_bytes` 披露键随协议裁定串一并退役清理）。**等待接口拆分**（§19.1 P→D 专项）：原地接续（j==i_P）等待基 = 在服流 running+active——**不重排队**，排队积压不再重压已有请求、也不再能翻转边界 argmin；新接纳（j≠i_P）等待基 = 目标全部在账负载——**不免费插队**（执行侧经 pending_decode_ready KV 就绪栅栏与busy 门进入，不进新请求队尾）。披露：候选 notes 增 `boundary_wait_kind`/`boundary_on_chip_locations`/`boundary_move_legs`/`boundary_copy_legs`（`boundary_two_source_remote_bytes` 纯两源披露键已随远读退役删除）；边界决策记录增 `stage_work`/`inherited_obligations`/`boundary_wait_kind`（decode 决策行透传，schema 前向兼容）。D19 不新增任何功能开关。受 D19 等待拆分影响的两处旧测**按新协议重写**（argmin 翻转机制自排队积压改在服负载）：`joint/test_d18_boundary_decision.py` 回退用例与 `online/test_decision_route_serialization.py` noc_migrate 用例。单测：`joint/test_d19_boundary_layout.py`（27 用例：§19.1 P→D 专项四态、D recompute 静默重建＋首 token 一次、等待接口拆分、区间纪律（≤3 位置/账本字节真值/已承诺接收端排除）、执行侧边界决策披露与成员入队面） |
| 跨阶段预约事务、引用交接与边界构图（D20，2026-09-26） | **已实现＋单测验证** | v2.0 §14.5/§16.4 阶段事务面（第二批执行协议统一收口；**零新增开关**；D18 两段式 `move_request_capacity_reservation`/`revert_request_capacity_reservation_move` 调用口**整体退役替换**，无并存档）。**幂等 epoch**：runtime `joint_boundary_epoch`（每进入一次未提交边界 +1）= 阶段 idempotency key，同 (request, epoch) 重复提交原子重放回执（重复 P 完成通知/在途 ack 不产生第二份换址或逐出；同 epoch 冲突目标/动作与 epoch 回退 fail-closed；容量类失败 epoch 不在案——换状态后同 epoch 重评合法，§16.4）；重复 P_COMPUTE_DONE 通知在 drain 入口幂等忽略（`boundary_duplicate_ignored` 披露行）。**resize 半边（式 20a）**：`stage_reservation_footprint` 单源（与准入同套助手：动作感知 token 基数＋目标侧 PARTIAL 后缀；D09 暂存分量已随 2026-10-14 远读退役删除）重推 D 阶段承诺，`required = max(0, 承诺 − own 驻留)` 真实 T+E；缩小释放未被物理工作占用的旧承诺分量（回执 `released_shard_bytes` 逐 rank 披露；D09 暂存随动作切换清账）；缩到自有驻留之下 fail-closed。**引用交接**：会话阶段引用 `acquire/release_stage_reference`——P 末尾到 D 提交期间本请求基础 KV **不进冷 victim 集合**（`_completed_resident_candidates` 与可回收视图同面排除；其他请求不停顿、无全系统栅栏）；D 目标预留与来源读计划建立后 P 阶段引用先于预约注销释放；引用未释放时注销预约 fail-closed（引用与预约不得双活/丢失）；release 幂等空放。**abort**：提交失败且无不可逆工作时保留继续方案——换址回退源实例＋足迹恢复提交前快照，epoch 结局记 aborted（重复 abort 重放；已提交逐出照常携带在回执，不丢账）。**运行态与完成归属**：runtime 新增 `completion_instance_index`（边界提交点锁定，完成信号按归属实例核销，缺归属 fail-closed）、`joint_stage_reference_id`、`joint_boundary_epoch`（plan_dict/决策行披露；边界决策记录增 `stage_epoch`/`stage_transaction`，schema 前向兼容）。**构图器守卫**：`_plan_and_emit_trains` 加入列车前先验全体成员阶段（仅 D_COMMITTED 可入——未定位置/动作的 D 永不入列车；拒绝原子性=不做任何账面变更），qp 队列头带 chunk 工作而处于 P 后阶段同样 fail-closed；选定成员迁移腿＋共享 readiness barrier 即目标第一步门（既有机制）。**边界可选面协议裁定（2026-09-26 修缺登记，2026-09-27 修正轮 C4 翻转）**：曾裁定为两族子集（完整型 {stay@i_P, copy@j≠i_P}、增量型 {remote-read@i_P, remote-read@j≠i_P}——增量型族已随 2026-10-14 远读退役消亡，边界布局恒完整型）——边界 recompute（静默重建）/边界基材料化候选曾保持运行时排除；修正轮 C4 已把上述形态的执行图钩子（目标第一步门/边界基材料化传输）接入运行时协议（多源引用交接钩子已随远读退役删除），适用面重写为区间缺失物理语义、协议裁定串 `BOUNDARY_RULED_OUT_*` 删除（`test_boundary_surface_ruling_reasons` 钉测随面改写）；批次4 同状态对照（验收纪律2）经 `run_scripts/joint_same_state_probe.py` 受控执行（同状态强制钩子 `JOINT_FORCE_BOUNDARY_REQUEST/ACTION`），不再限于可选面子集；口径同面登记于 README 本四行（D10/D18/D19/D20）与实验总纲 §7.1。**验收清单"取消"幂等项的承载映射（同日登记）**：本仓回放协议无请求取消机制（无 EOS/取消事件侧，停止条件由 workload 脚本定义），"取消"语义由 (a) **已知终止直结**（`_joint_boundary_terminal`→`_settle_terminal_at_boundary`：`P_COMPUTE_DONE→MERGE_PENDING`，不选 D、边界当场闭合完成事务——首 token 已满足停止条件即无 D 可取消）与 (b) **幂等 release/abort 重放**（`release_stage_reference` 重复释放幂等空放＝"取消与提交竞合"安全面；`abort_stage_reservation_transaction` 同 epoch 重放 aborted 回执；重复 P_COMPUTE_DONE 通知 drain 入口幂等忽略）承载；钉测 = `test_terminal_settles_directly_without_d_selection`（test_d18）＋ `test_reference_release_is_idempotent`/`test_abort_reverses_flip_and_records_epoch_outcome`/`test_abort_without_committed_flip_is_idempotent_noop`/`test_duplicate_prefill_done_notification_is_idempotent`（test_d20）。单测：`joint/test_d20_stage_transaction.py`（19 用例）＋ `joint/test_joint_fixes.py` 预约事务钉测（换新 API）＋ `test_sh30_kv_incremental_invariants.py`（同实例事务轮换） |
| 预测器自身开销统计接口＋验收清单补钉（ACC·验收纪律第 1/3 类，2026-09-27） | **已实现＋单测验证** | v2.0 §19.3 代码侧统计接口（**恒开、零新增开关**——§19.3 是验收必需项非可选机制）。**账本**：`joint/predictor_cost_accounting.py`——P 准入/D 边界两阶段决策墙时逐次样本（最近秩法 P50/P95/P99，整数纳秒确定性；`candidates_max` 候选面宽度供"3N→6N 不推定总仿真只慢固定倍数"的逐决策证据（2026-10-14 三动作终态；原 4N→8N））、D11 容量驱逐摘要快照查询费样本（摘要维护费）、后端事件量（图批节点/边/watch/未来 alarm 四计数，经基类 `_note_backend_batch` no-op 钩子由 sh30 覆盖喂入——幂等重放在 sequence 门早退不重复计数）、run 末快照 `predictor_cost_snapshot`：预测器合计、宿主总占比（整数 ppm，宿主锚 = 调度器构造点）、峰值 RSS（`ru_maxrss`，Linux KiB→字节口径如实披露；无 resource 平台报 unavailable）、决策日志大小（`online_service` 在 `online_decision_log.jsonl` 落定后以文件字节回填，未产出 = None 不猜）——快照以 `predictor_cost` 键入 `online_stats.jsonl` 汇总行。宿主 Python 墙时 ≠ 真实片上决策时延（§19.3 末段），快照只供验收证据；首轮工程目标（两次决策额外宿主时间 ≲ 固定回放 10%）是待验证预算，本接口只供数不下结论。**验收清单逐项核对补钉（第 1 类）**：通用清单补 TP 不均匀/KV 头复制（式 2 整头不整除精确＋TP>头数 padding rank 零 KV 记账＋逐 rank 真值进边界空间等待）、实际增长超预测（管理器级：Z→O 转账不双计、超预约天花板 fail-closed、超材料化足迹真实 T+E 逐出）；P→D 专项补第 6 条（控制费用使小收益迁移不再合算——切换-only 迁移腿 2,570 ns 为控制费，毛收益 2,000<2,570 继续胜出/2,700>2,570 才切换，正控制对称钉）、第 10 条（第一笔新物理工作后不再次选点——提交后环境变化使 argmin 翻转，迟到重复 P_COMPUTE_DONE 仍幂等忽略、决策零改写；新请求同环境真实切换为正控制）、第 12 条（P 末尾不预发 D——初始预测含 D 而真实边界前零 decode 决策行/零就绪队列成员/归属未锁定）。其余清单项由既有钉测看护（逐项映射见 `joint/test_acc_acceptance_formula_conservation.py` 模块头）。单测：`joint/test_acc_acceptance_formula_conservation.py`（11 用例）＋ `joint/test_acc_predictor_cost_accounting.py`（10 用例：分位数学/空集/fail-closed 校验/快照字段完备/JSON 可序列化/sh30 接线冒烟） |
| 修复轮二（2026-09-27，独立审查 12 项"部分完成"判定的修复） | **已实现＋钉测**（集中门禁由并行车道执行，见 `~/joint_fix_round2_20260927/gates/`，本行不预填门禁结论） | **D09 固定字节双缓冲**：`remote_read_exec_tile_tokens` 放大废除（旧 max(tile,⌈base/2⌉)），tile = 配置值固定，每 pass/窗 tile 数 = ⌈base/tile⌉ 可 >2、逐 tile lag-2 流水消费，暂存 = 2×配置 tile×kv_token_bytes 恒定（§1/§3 remote-read 行已同面改写）。**D16 有界模板**：`plan_prefill_remote_read_transfers` 返回 O(窗数) 模板（不整轮物化窗×tile 列表）、逐窗 `prefill_read_window_tiles` 消费点物化、builder 复检改模板级；P 决策行新增 `prefill_remote_read_summary` 有界聚合披露键。**D10 静默重建执行图**：构图器消费 `joint_suppress_sampling/output`——重建 span 不进 final norm/logits 聚合输出位、不产生 debut first_token 观测、算子时间照常，首 token 只消费一次（`online/test_d10_silent_rebuild_execution.py`）。**D03 frontier 对齐**：P 准备腿起点 = 目标实例在飞列车剩余工作（sh30 `_prep_frontier_by_instance` 冻结快照，roofline 服务估计），不虚构"决策时刻发射"重叠；notes 键 `P_prep_at_target_frontier(+Nns)`/空闲保持 decision_epoch（`joint/test_d03_frontier_alignment.py`）。**D14 审计落定**：runtime.decode_length = D−1 前向步（首 token 由 P 末前向产出，§5.2），`observe_completed` 直收已是正确单位——换算需求经证据链审计取消（span 铁证＋dl=0 结算＋KV 增长对账，`joint/test_d14_forward_step_semantics.py` 钉死）。**D17 取整**：`observe_transfer_from_link_window` base_ns 向下取整改 ceil（基准时长不为少报方向截断；`joint/test_d17_rounding.py`）。**D19 边界视图填充**：`BoundaryKVLayout.terminated/first_token/obligations` 经 `BoundaryRuntimeStateEnvelope` 真实接线（在线侧物化），区间 layer_start/layer_end 填真实层覆盖（PARTIAL 基 = 驻留前缀层；`joint/test_d19_boundary_layout_population.py`）。**D20 预约回滚**：drain 边界异常路径显式 `abort_stage_reservation_transaction`（幂等 epoch；post-commit 物理落地窗口不伪造回退，留账随 fail-closed 终止收敛并披露；`online/test_d20_reservation_rollback.py`）。**D11/D17 D 边界容量审计**：D 边界决策行新增 `capacity_summary_view`（与 P 行同构造）＋ `capacity_ledger_by_tp_rank`（C/W/O/Z＋F 逐实例逐 rank 序列化）——仅凭日志可重算 D 阶段容量判断。**D12 平价与回退披露**：`joint/test_d12_merge_projection_parity.py` 预测-执行逐字段平价（三位置/退化/去重/容量回退 ≤2）；`last_merge_outcome` 新增 `fallback_sequence` 完整回退序列。**D05 披露**：merge 尾部 commit = 零时 watch 锚（核实无真实时长），计价不加幽灵段，notes 键 `merge_commit_anchor=zero_time_watch`（`joint/test_d05_merge_commit_disclosure.py`）。**D04 端点登记**：零距离池路径（noc_path<2）的 remote_load/remote_store HBM 端点补登记（与计价端点同覆盖；执行侧 `_register_transfer_flows` 的旧 `len(noc_path)≥2` 门漏登部分由修复轮三补全为无条件登记——见 §6 修复轮三注）。**测量钩子**：`JOINT_FORCE_ENTRY_ACTION`（与 `JOINT_FORCE_BOUNDARY_REQUEST` 成对武装 P 入口单点强制、与 `JOINT_FORCE_BOUNDARY_ACTION` 互斥、缺省 off＝行为逐字节不变，测量仪器——`online/test_c7a_force_entry_probe_hook.py`）；probe 工具 `joint_same_state_probe.py` 扩 entry/boundary/replay 三族臂＋式(24) 边界损失＋宿主开销 ≤10% 预算如实披露。**版本串升版 `v3.1-2phase-20260927`（前代 v3-2phase-20260927）：旧实验记录持旧版本串时不得重命名为新协议结果、不得自动充当新模型证据**（§3 D17 行同面） |
| 域三口径度量（C16） | **已实现＋单测验证（现有 43 用例）** | `slo_tools/domain_metrics.py`：D_feed/D_econ/实际选择三口径分列（elected/forced 分列、measured=NA 如实标注）+ 派生指标（hop 分位/方向分布/瓶颈资源/内外等值线逐 (请求,实例) 导出）+ A3' δ 敏感性重定价（δ∈{σ̂,2σ̂,4σ̂} 离线诊断，σ̂ 三级链）+ home 迁移轨迹消费面；driver 五 sink 扇出、非 joint run 完全静默 |
| 负载区间与运行输入快照（2026-09-24） | 已实现＋回归验证 | `load_imbalance.py` 对 joint 请求以决策日志 `completion.tick` 为实际完成端点；`train_ledger.exits` 只核对成员与实例，其 tick 为列车发射时刻。历史非 joint run 继续输出 `legacy_train_ledger_emit` 代理端点并显式标注。运行入口把生效的 `trace_config.csv` 与 hardware JSON 归档到 run_dir；`domain_metrics.py` 优先显式参数，其次用 run 本地快照或旧命名副本推导 ρ，缺 run 本地证据时给 NA 与告警，不读取当前 checkout 猜测历史硬件。 |
| SLO 水印对 joint 的口径 | 已登记（joint 词表已扩，R12；**merge v2 重放已扩，2026-09-17**） | slo_tools 三工具已登记 astra-sim-joint；水印重放词表已扩 joint 专属事件（跨实例工作副本双驻留记账——SessionState base/working；决策行携带 joint_action/history_transfers/重算口径字段；完成行显式披露 `joint_working_copy` 真值——复审 K3）；**merge v2**：completion 行携带 `merge_direction` 走 v2 重放（forward=exec 工作副本搬 home／reverse=home 基础释放搬 exec 或零字节翻转／in_place=无主回暖就地转正；零池写断言——本会话 remote_store 即 fail；noc 源端与方向交叉校验；`joint_merge_forward/reverse/in_place/zero_byte_flip/home_migration` 计数；混合形态准入的池恢复后缀进期望工作副本基数），无该字段的旧日志走 legacy 分支（金样对照兼容）；**非账本披露行词表（2026-09-22 C19）**：`kv_eviction` 账本行之外，joint_admission 族 + `readplan_reconcile`/`readplan_settle`/`quota_oneshot_overflow`/`merge_done`/`link_telemetry_coverage`/`joint_decision_metrics` 披露行并入跳过词表（M4 2026-09-26 增补 `kv_eviction_done`——逐出尾标记完成披露行，跳过 = 零重放语义差）（run 级行 request_id 为空不误触 fail-closed；13 臂矩阵 postprocess 全净实证）；**2026-09-27 修正轮 W1–W5 词表扩展**：P→D 迁移重放（W1，`decode_instance_index != prefill_instance_index` 先按披露口径源扣/目标加再做 D 增长——修前 combo_T 重放负占用）、三位置 merge 形态（W2，`_joint_completion_three_position` 含 REMOTE 基 forward 胜者=source 与 rebuild×base==exec 重合豁免——W5）、staging lease 两阶段生命周期（W3，acquire/release 在册表＋run 末悬挂 fail-closed）、P→D 折叠清账（W4，迁移腿落点==base 槽按 W vs f(P) 恒等式二分：完整型 −base_bytes 释放/增量型并基零占用差）；16 臂水印后处理统一收口 ALL-STEPS-PASSED、b4 五臂复算绿（证据 ~/joint_fix_round_20260927/gates/g4_*.log）；决策日志重放仍为上界口径（逐 rank 判决需 kv_delta_journal 权威层——journal 已随 run 末 sidecar 序列化为第四键、`domain_metrics` 四层可信度分级已消费（settlement_full_join>partial>empty>decision_log_only；F3 修复批 2026-09-22），同 S3） |
| 容量逐出的决策日志披露（R17，2026-09-17） | 已实现＋单测验证 | 旁路逐出咽喉点（`_emit_eviction_only_nodes`）在图发射后同 tick 落 `kind=kv_eviction` 决策行（decision.evictions 携带 `_transfer_summary` 条目）——覆盖 decode 增长逐出（成功/停滞/唤醒三路）与准入失败已提交逐出（潜伏位点：当批零触发——reserve 的 feasible 预检拦截在先 + prepare 受 R1' 预约不变量覆盖，R17-7 探针实证）；`prefill_evictions` 为结构性恒空死通道（准入预约覆盖全动作足迹，drain expand gap≡0），注释+单测钉死不新增空字段载运；转移摘要补序列化 `resident_prefix_layers_before/after` 与 `source_instance_index`（连锁不变量免重建 + victim 归位，R17-1d）；tick 守护 = 断言 `_batch` 在场（N4）。M4（2026-09-26）：行 decision 增 `watch_id` 配对键，逐出尾标记真实完成处（`_on_eviction_watch`）同键落 `kind=kv_eviction_done` 披露行——kv_eviction→kv_eviction_done tick 差 = 实际池写回时长（驱逐/恢复时间线测量执行侧真值）。工具侧配套：hbm_watermark 主循环 kind 门前拦截 kv_eviction（自担 tick 单调 + 逐条 `apply_evict` + 非空率哨兵）+ `_joint_prefill` fail-early 对称校验（prefill_shrink/discarded>0 fail-closed）+ (primary, base) 二元组前缀连锁不变量（R17-2/3/4）；hopbytes `collect_joint` 消费 kv_eviction 池写流（此前通道 2/3 系统性漏计）并移除 prefill_evictions 死读。历史 run 决策日志定格缺失、离线不可重建——水印三件套与可信 hopbytes 需带本修复重仿真（第三次错误处置，方案 v4） |
| 远端端口并发后端（2026-09-24 SerDes 片外链路并发化） | 已实现＋C++ 夹具验证＋固定输入 A/B 恒等（**零池负载平凡一致性**，不证争用正确性） | 串行 FIFO 端口整体替换为并发流体端口（§7-D'；`AnalyticalRemoteMemory.{hh,cc}`）＋发射门控改造（远端 MEM 独立无上限计数槽、与 comm 槽解耦——comm/hbm_dma 槽 2026-09-25 起同为计数制无上限（PROVENANCE §45）、static 结束门计入远端 MEM、`NDEBUG` 无关的 occupy/release fatal 校验、析构未释放诊断）。夹具锚：`RemotePortNwayTest` S1–S10（300｜90/120/140/150ns｜1B 流体 1/6ns）＋16 端口×128 流压力（2048/2048、peak_streaming=128、redistribution=2032=16×127）；`RemotePortOnlineGateTest`/`RemotePortStaticGateTest`（同 rank 双 MEM 发射后 in_flight=2、peak_streaming=2、shared_busy_ns>0；MEM 与 COMM_SEND 门互不占用；static finish 等全部终结）；`HbmNwayTest` 重锚（contention ON/OFF）。sensing 开启时逐事务明细 `remote_memory_transactions.jsonl` 惰性落 `bridge_dir/`（首个完成事务才建文件；成功后 runner 搬 `results/` 并入归档常驻白名单；写失败 fail-closed FATAL）。固定输入 A/B（快照基线树 vs 本树）：决策/发射序列与全系统指标除宿主计时字段外逐字节恒等——窗口内池后端完成事务数为 0，属零池负载平凡一致性；Python 计价面与 C++ 零耦合，无重标定触发。证据 `/tmp/serdes-joint-work/`（gate_rerun.log 49 步全 PASS、precision_rerun.log、r4_*.log、stress_results.txt、pricing-audit.md） |
| 三机制消融/外部对照实验 | 未运行（工具面已具备） | 仿真规模正式 campaign 仍属另行登记的实验任务（§8 第 6 步）。2026-09-27 修正轮已交付其工具面：冒烟矩阵（交付时 16 臂含三掩码臂；2026-10-14 远读退役删除 `TJE_remote_off` 与 `mask_remote_recompute` 两臂后为 14 臂）＋同状态 probe 工具（`run_scripts/joint_same_state_probe.py`，式(24) 选择损失受控实测，首轮 2 请求 6 候选 6/6 PASS）；**2026-09-27 修复轮二 probe 扩展**：entry/boundary/replay 三族臂（P 入口 `JOINT_FORCE_ENTRY_ACTION` 枚举/边界同状态对照/固定动作回放）＋式(24) 边界损失＋宿主开销 ≤10% 预算达成/未达成/数据不足如实披露——正式规模同状态 campaign 与四外部 baseline 仓对齐仍属修正轮之外 |
| P→D 边界策略仪器开关 `JOINT_BOUNDARY_POLICY`（一次映射对照臂；验收纪律第 2 类，2026-09-28 审计整改批） | **已实现＋单测验证（`online/test_boundary_policy_one_shot.py` 23 用例）** | 取值域 {`off`,`p-continuation`}（定义源 `joint/joint_config.py` `BOUNDARY_POLICIES`），缺省 `off` = 两阶段闭环（D18 主语义）**行为零漂移**（off 时 D 决策行无 `boundary_policy` 披露键 = 行与既有钉测逐位一致）。`p-continuation` = **一次映射对照臂**（测量仪器，验收纪律第 2 类"一次映射 vs 两阶段闭环对照"专用；生产 run 不设）——D 边界不做二次全候选重估（跳过 `select_decode_target_and_action`，候选恒空列表、`cost_ns` 真值 NA 如实披露，ACC 照常记样本 candidates=0 供控制开销对照），直接按 P 准入 `estimate_P_with_continuation` 披露的默认延续形态提交：decode_instance = prefill_instance = i_P、延续恒 stay@i_P（2026-10-14 远读退役起完整型单族；原增量型（P=remote-read）= remote-read@i_P 延续分支已随远读退役删除）；延续 = 原地继续无迁移腿，D 提交后的完成归属/构图器/merge 链路与两阶段延续分支完全同路（同一 commit 路径）；已知终止仍直结结算（`_joint_boundary_terminal`→`_settle_terminal_at_boundary`，不进一次映射分支）。**fail-closed**：非法值启动失败（`joint_config.__post_init__` 单一裁决点，env 与直接构造同面）；**与 probe 强制钩子 `JOINT_FORCE_BOUNDARY_ACTION`/`JOINT_FORCE_ENTRY_ACTION` 互斥武装**（同设 = 仪器二义，调度器启动失败——sh30 `_check_boundary_policy_instrument_conflict`；p-continuation 下边界钩子无候选面可强制）；p-continuation 要求 `JOINT_ALLOWED_ACTIONS` 掩码放行 stay（完整型 P 动作的默认延续恒 stay@i_P，掩码剔除 stay 即一次映射在边界提交掩码外动作、违反"掩码是候选集上限"合同，启动即拒）。生效 run 落 manifest 键 `boundary_policy`（`joint_mechanism_manifest.json`）＋ D 决策行专用披露键 `boundary_policy`（独立键，与 `forced_reason` 词表无冲突）。probe 配套：`run_scripts/joint_same_state_probe.py` one_shot_pair 成对臂（两阶段参照臂 run 目录 twophase_reference × 一次映射臂 run 目录 oneshot_p_continuation，同 trace 同参数仅差 env）＋一次映射 vs 两阶段闭环对照汇总段（D 改点率/同址改动作率/交接字节（P→D 迁移腿 shards 与 C13 copy_handoff 两口径）/E2E 均值/完成率/控制开销（predictor_cost decision_wall 与 D 阶段分列））。消费面：sh30 `BOUNDARY_POLICY_ENV`/`__init__` 裁决/D 边界单一裁决两形态/披露键/日志读取。单测：`online/test_boundary_policy_one_shot.py`（23 用例，含 off 零漂移/fail-closed 三裁决/披露键缺席面） |
| **W5：阶段事务扩展、统一结算接线与账本正式计价（2026-09-29）** | **已实现＋单测验证** | ① 统一结算状态机（§7.4）：旧 `MERGE_PENDING→MERGE_COMMITTED→DONE` 抽象替换为 `SETTLE_PENDING→SETTLE_COMMITTED→SETTLE_DONE`（旧常量已删）；`SERVICE_DONE` 单独记录（completion 行 `service_done_ns` 键 + 闭环后继 alarm 仍锚 service_done+interval，future_alarms 语义不变）；三类结算事件 = service（completion 行）/物理 merge（merge_done 行+载荷披露）/settle（新 `kind="settle_done"` 行——零载荷同刻闭合、merge watch 交付闭合，事件序=物理 merge→settle）；`_joint_settlements` 结算账本（闭合即弹出）+ `verify_run_end` 残留 fail-closed；`distributed` 执行侧当时显式拒绝（与 W3 定价侧一致；W9 起两侧均切换为目录提交型实现）。② 事务扩展（§7.2）：提交点版本校验（`expected_versions`——目录事务版本+阶段纪元表；sh30 侧 KV 纪元复查；漂移=同 tick 竞争 → `KVStageVersionConflict`/回退继续方案+`failure_class` 分列披露）；**所有来源引用**经目录 `acquire_entry_reference`（执行 pin，D 提交点 acquire/settle 子协议提交前释放——源保护窗口=D 执行期）；A22 分记（`retained_obligations`：成功/aborted 回执均携带已提交 T+E 逐出的身份/逐 rank 字节，不取消不漏计）+ attempt 维度；consolidate 回退 m_end−1 推广（`MERGE_CAPACITY_FALLBACK_MAX=2` 删除，预测侧与执行侧 len(ordered) 循环同序）。③ W1 账本正式计价接线：sh30 `__init__` 显式转发 `work_ledger_sink`；`LedgerWorldSource` 实装（资源腿精确剩余量 in-service 腿 / issued-not-streaming 非 in-service 腿 / committed-not-issued 计数披露；**逐键降级**——账本无权威腿的键回退注册表计数，不静默空背景）；决策快照（`build_decision_snapshot`，versions 七键含 ledger）+ `world_ledger_world_source` 终态 manifest 披露；journal 重放保真（剥离字段改由 `request_journal_work_ledger.jsonl.gz` 侧车承载，`iter_request_records(attach_work_ledger=True)` 前缀重放账本状态与原跑一致）；预测 manifest 并入 joint_mechanism_manifest.json 的 prediction 节。④ W6 消费切换：P 准入与 D 边界正式 n̂ 切 `estimate_conditional`（同刻同实例同一规则/版本；horizon 保留诊断/先验），到达/完成/删失/服务样本四挂点接线（shadow）。⑤ §7.2 义务执行侧：`inherited_obligation_ids` 义务表 → JCM `obligation_resolver`（settle 门控语义）+ settle 闭合门（活账本核验）。控制延迟：提交点三重复查（版本/容量/源保护）+ `control_latency_ns`/`control_commit_recheck` 披露（真实时延推进需 C++ 定时事件——登记缺口，0=assumed-zero 不变）。单测：`online/test_settlement_state_machine.py`（7）+ `joint/test_ledger_world_source.py`（8，含 journal 重放保真）+ `joint/test_d20_stage_transaction.py` W5 扩展（5）+ slo_tools settle_done 消费（3）+ 重锚 d12×2/d18/d19/remote_settlement/k_batch/N9。⑥ G-B 冒烟性能接线（首次真实 trace 投影暴露）：`resource_projection` 的活跃集计数（`_active_members`）与 `_finishes`/`_integrate`/`_settle_completions`/`_assert_conservation` 的逐事件全腿扫描在真实 D 边界（~1.5K 腿 × ~49K 事件）为 O(步×腿²)——单边界决策实测 ~407s；改为**增量成员表**（`_active_by_resource` + `_active_legs`，在与旧扫描完全相同的资格变化点维护，读值逐位相同）后 ~9.3s（44x；纯性能变更，广覆盖 1973 全绿零漂移）。 |
| **W7：新增需求预测、验证与回退（future_demand recent/reverting 注入，2026-09-29）** | **已实现＋单测验证** | 新增 `joint/background_forecast.py`（互斥类别 (new_session_first\|session_return × revealed_type) → 资源工作模板：prefill 逐层 fixed 腿＋逐层 TP barrier＋decode 聚合 compute 腿（HBM KV 增长字节流）＋decode TP barrier＋session_return 池恢复 stream 腿；参数从 W6 类别均值派生、缺样本回退冷启动 preset 且不确定标记携带）＋`joint/forecast_validation.py`（成熟恰一次差分回填/按预测距离准入 (mode,band)/固定回退链 no_samples→anomaly→expired→template_uncovered→budget_exceeded→distribution_shift→unvalidated_distance/开销账本）。确定性注入（§8.4 第一版）：序号阈值 θ=1.0 累计跨阈生成完整代表请求、桶内等距相位（无 RNG）、注入时刻经 W2 `forecast_arrival_placeholders_ns` 重仲裁、忙时照常注入（不推迟到空闲）；闭环返回约束（处理中 session 下一轮 ≥ service_done 下界）；候选共享同一情景（`WorldContext.forecast_works/placeholders/barriers` 新增缺省空字段＋`scenario_key` 快照缓存键含模式+统计版本；`none` 档逐字节零漂移）。接线（最小面）：sh30 `_joint_world_source` 包裹 `ForecastAugmentedWorldSource`（future_demand_mode≠none 且 J-on）＋成熟驱动挂点＋闭环下界；preset 新增 `forecast_*` 参数（§9.2 表落地，来源登记 EVIDENCE）。附带 W2 内核修复：`_integrate` dust 钳位对称化（roofline 腿字节侧正残尘曾致零进展死循环）。单测：`joint/test_future_demand_forecast.py`（21，V8 六组受控用例/互斥闭环/B-C-D 单变量/none 零漂移）＋`joint/test_forecast_validation.py`（16，成熟恰一次/距离准入/回退链/开销）＋`online/test_forecast_future_isolation.py`（5，2026-09-30 修复批补 §8.3 预测器与隐藏真值隔离端到端钉测：改写后继请求 manifest 隐藏 prefill_length/decode_length/inter_request_interval_ns、保持已揭示类型与交付时刻线不变 ⇒ ArrivalStatistics/estimate_conditional/build_plan 注入集逐字段不变，future_alarms envelope 反证相异）。 |
| **W8：同状态探针、阶段对照与闭环实验工具（instance×action 精确强制＋状态一致硬证书＋service/settle 分采，2026-09-29）** | **已实现＋单测验证** | 方案 §11.3/§12.1 W8/§3.2 末段两项 probe 迁移点落地。**(1) instance×action 强制钩子**：新 env 对 `JOINT_FORCE_BOUNDARY_TARGET`/`JOINT_FORCE_ENTRY_TARGET`（值 `<instance_index>:<action>`，与 `JOINT_FORCE_BOUNDARY_REQUEST` 成对武装）——强制**恰好的 (instance, action) 候选**（指定组合不在候选面/不适用 ⇒ 显式 fail-closed 含同动作实例清单，**不静默重选实例**）；旧 per-action `*_ACTION` 对保留向后兼容但为弃用方向（同族 TARGET+ACTION 同设 fail-closed）；armed 决策行 `forced_by_probe` 携完整身份 {stage, hook, action, instance_index, is_continuation}；D 边界决策行新增 additive `state_versions` 披露（kv_ledger_epoch/directory_version/stage_epochs——`_joint_boundary_versions` 单一事实源）。单测 `online/test_w8_force_target_hook.py`（env 语法/五类互斥/精确匹配含非最低成本实例反证/越界面显式失败/state_versions 披露/D 入口全链）。**(2) probe 双维重构**：`aggregate_by_instance_action` 逐 (instance,action) 候选全量采集（旧 `aggregate_by_action` 每动作取最低成本实例的口径退役，`rollup_actions` 仅作展示兼容层）；`generate --granularity action|instance-action`（instance-action = 逐适用 (instance,action) 候选各一臂，TARGET env，run 目录 `<rid>__<stage>__i<inst>__<action>`）；summarize 双维面（predictions/actuals/rank 键 `i<inst>:<action>`）。**(3) 状态一致硬证书**（§11.3 一致性是硬门——原 predicted-drift note 软降级退役）：重放前缀在目标边界逐字段核验 tick/seq/冻结候选面（适用性/代价/继续标记/理由）/stage_epoch/state_versions/义务/stage_work/boundary_wait_kind/model_semantics_version（P 面另含 joint_instance/action、service_factors 摘要；rng_basis=deterministic-replay-no-host-rng 披露）；任一不匹配 ⇒ 该分支 `status=state_mismatch` **不纳入同状态损失样本**；`status=ok` 仅发给全字段匹配+强制生效+实测齐全的臂；`same_state_sample_coverage` 如实披露已测子集参照。**(4) service/settle 分采**：`service_end_tick` 的 max(completion,merge_done) 混称口径退役——`service_settle_ticks` 分列 service_done_ns/settle_done_ns（W5 settle 事件），consolidate 零载荷 settle==service 显式记录（zero_payload_settle），旧世代日志回落 legacy 口径如实披露，distributed 缺 settle 事件不得回退 completion=settle（`settle_missing_policy_distributed` 预留 W9 分支）；双指标 actual_service_done−common_decision_tick 与 actual_settle_done−common_decision_tick，**排序/式(24)损失以 settle 为准**。**(5) missing/fixed_policy_arm/P 参考接续分离**：未执行（exit_code 缺）/未完成（无 completion/settle）候选标 missing 不以预测值填补；replay 臂 `arm_class=fixed_policy_whole_run`、one-shot 臂族 `oneshot_policy_whole_run`（whole-run 消融独立标记，不与同状态数据混算）；P 参考接续（P 行 decode_instance/decode_action 参考 D）与实际边界 D 分列披露（接续近似误差与资源误差分开归因）。单测＝probe selftest 扩至 5 场景 206 断言（新增场景 5：instance-action 臂+TARGET env+settle 零载荷/载荷形态+证书全匹配 ok/单字段漂移 state_mismatch 不入样本+missing 标记+P 参考接续派生+arm_class 标记）。 |
| **W9：distributed 跨轮留存与区间级 E（G-DIST，2026-09-29）** | **已实现＋单测验证** | 方案 §7.4/§12.1 W9/§11.1 V9。**(1) 目录提交型结算**（`KVCacheManager.settle_distributed`）：settle = 提交完整有效目录（区间条目权威化、目录事务经 `_check_invariants_after_mutation` 收口）+ 执行 pin/引用释放（不永久 pin）+ 必要旧义务闭合（copy handoff/restore group 于 merge 边界兜底）+ **零物理归并**（无 merge 载荷，W5 零载荷 settle_done 同刻闭合）；版本键 `last_settled_request_id`（一请求恰一次结算）；按族收敛（copy/recompute=零字节翻转族：primary=并集@exec；remote-read=增量族：primary 保增量真值，混合形态 `distributed_shard_truth` 标记审计取 shard 分布）；home 粘性不变（不暗中重置，A13）。旧位置移入**跨轮留存通道 E**（`_distributed_retained`：旧基础段/边界来源段/迁移后的旧 primary——目录条目集合无三位置上限，§7.4），三消费点=目录镜像（`_kv_directory_retained_entries`）/守恒期望表（`_incremental_retained_contributions`）/容量摘要（`_capacity_retained_contribution_bundle`——θ 公式与增量维护框架复用，按所在实例/实际区间分桶）；consolidate 缺省恒空（零漂移）。**(2) 调度器分派**（sh30）：`_joint_settlement_policy` 单一裁决点替换 W5 拒绝点（`_complete_requests` 按 policy 分派 merge_back/settle_distributed；settlement 账本登记 `settlement_policy` 实值 + `directory_commit` 披露块）；**早到下一轮依赖门**：distributed 下会话结算未闭合 ⇒ 准入延迟留队（到达排队照常，消费受数据/目录依赖限制）；`_joint_session_view` 目录化（distributed_layout 会话 history=目录并集、base 锚=片上自 0 起最大覆盖段）；准入决策行 additive `distributed_source_surface`（位置数/区间数分别统计 + 跨轮增长 ≤ +2 断言 fail-closed）。**(3) 目录驱动 prepare**（`_prepare_prefill_distributed`）：元数据校验 = 单元格级覆盖断言（`assert_full_content_locatable`——跨持有者部分重叠区间的真并集，取代单位置等式）；四动作：stay=并集@exec local_hit（否则显式不适用）、copy=目录多源逐区间 NoC 腿（来源段全保留，chunk 释放=consolidate 优化不沿用）+目标既有留存段并入工作副本描述、recompute=0 起步重建、remote-read 连续轮**不清零** primary（exec 已持上轮增量继续增长）+ base 锚留存段检出入 base_*（下游口径兼容，守恒单计）；`resident_token_offset` 单一事实源（增量族真实 token 域 [base_H, base_H+context)——通道 A/留存段投影；merge_back/三位置归并终态归零）。**(4) 区间级 E**：`_completed_retained_segments`（目录 owner 过滤 victim——会话单实例假设推广）+ `_evict_retained_segment`（经目录 `evict_to_pool`：owner 记录保留、唯一内容在池）+ `restore_retained_segment`/`complete_retained_restore`（目录 RESTORING 状态驱动）；`_ensure_capacity` 每类 pass 主位置 victim 后接留存段驱逐；`_evict_suffix/_evict_session/_remote_store_transfer` 层段字节统一经 `_distributed_resident_layer_bytes`（缺省与旧公式逐位一致）。**(5) 会话终局**：`retire_terminal_session` 先审计后释放（悬挂引用/未完 ack/在途 copy fail-closed——`assert_no_dangling_session_state`）+ 留存段字节随终局释放。**(6) 开销计量**（§10）：`DirectoryOverheadMeter`（操作计数/累计墙时/条目峰值/逐会话逐轮来源面时间线）——settle_done 行 `directory_commit` 块 + run 末 `distributed_directory_stats` 披露行（slo_tools 词表已纳）；`same_round_merge_saved_bytes` 显式同轮口径（同轮省 merge 不报告为长期收益，E2E 对照按完整会话分列）。**(7) 预测侧**：`_reject_distributed` 拒绝点退役 → distributed settle 腿零载荷（P/D 图 notes `settlement=distributed:directory-commit-no-merge-payload`，predicted settle_done=service_done）；consolidate 路径零漂移。单测：`online/test_distributed_settlement.py`（12：4 轮多源>3/区间变化/逐出恢复可定位/容量压力留存段 victim/终局清理无悬挂+悬挂负例/consolidate 零漂移/双策略对照面/调度器分派+早到门）+ `joint/test_distributed_settlement_protocol.py`（16：来源面/增长预算/单元格覆盖/终局审计/计量）+ 重锚（action_work_graph distributed 定价、settlement_state_machine 分派、kv_directory_projection_parity 通道 A 真实区间）。 |
| **2026-09-30 修复批（§10.3 误差成熟与归因 cfg1／§5.6+§10.1 计算量组 cfg2／§7.1 控制延迟 cfg3）** | **已实现＋单测验证** | **cfg1 完成预测成熟通道**：新 `joint/forecast_validation.py` `CompletionMaturityLedger`——实际执行候选在决策行发射点登记（P 准入/D 边界各恰一次，重复登记 fail-closed；未执行候选不登记保持 missing，`measured` 恒 None 口径不变），settle_done 闭合点按 (request, stage) 配对实测（误差=实际−预测，决策相对轴；归一化分母=max(1,\|predicted\|) 规则随披露；幂等恰一次；service_done 缺席/pending 未决单列不丢）；run 末新 `kind="forecast_status"` 收尾行（有引擎或有配对样本才落——off/短夹具形态逐字节不变）承载引擎 `disclosure()`（到达桶成熟账/开销账/回退链——此前 run 结束即丢的唯一生产消费出口）+ 完成预测成熟账（P/D 分列绝对/带方向/分位/归一化）；离线 `slo_tools/domain_metrics.py` 新增 `derived.completion_error_report`（消费 decode 行 boundary_decision + settle/completion + manifest：stage×endpoint 主块 + 按实际预测距离档/动作/瓶颈段/负载阶段（P 行承载、D 行 NA）/已知承诺代理（P=在途已提交通信、D=继承义务）/历史模式（manifest future_demand_mode）分组 + 七分归因关键词映射（快照陈旧缺账/物理服务近似/容量摘要偏差/未知长度/新到达/布局协议/取整；启发式规则随披露，无命中单列 unattributed））。**cfg2 计算量组**：`JointCostModel` 新增前推计算量账本（世界快照缓存命中/重建+快照规模（背景工作/资源/预测注入）、逐候选前推事件总量/峰值、闭包资源/工作峰值、背景排空截断、预算超限计数（fail-closed raise 前记账））；sh30 `_joint_accumulate_projection_cost` 逐决策汇入（P/D 两入口）+ D 边界回退额外遍历计数（§5.6"版本失效、失败重评等额外遍历须累计报告"）；`joint_decision_metrics` 收尾行新增 `projection_cost_accounting` 键（含预算超限触发率=超限数/max(1,前推数)）；逐候选面 = breakdown `projection_stats`/`background_delays`/`capacity_theta_prediction` 三键并入 `_JOINT_BREAKDOWN_LOG_FIELDS`（28→31，与同日 §5.1/§5.5 交付面修复批共键）。**cfg3 控制延迟（2026-10-02 纠正）**：此前允许登记后只计价、不兑现延迟，违反 §7.1；现已删除 `control_latency_execution_deferred`，`control_latency_ns>0` 一律拒绝。当前仅支持明确披露的 `0=assumed-zero`；预测配置版本升为 `predcfg-v2-20261002`，六个随仓 preset 同步升级，旧版配置不自动迁移。单测：`joint/test_forecast_validation.py`（16→23，+CompletionMaturityLedger 7 例：恰一次幂等/重复登记 fail-closed/未知阶段 fail-closed/pending 与 service 缺席单列/settle 缺席不成熟/未知请求空放/空账形态）+ `joint/test_prediction_config.py`（非零控制延迟拒绝门）+ `joint/test_action_work_graph.py` 零控制腿与非法配置门回归 + `online/test_joint_decision_schema.py` 字段冻结锚 28→31。 |

### 3.1 W0–W11 交付总览（2026-09-29 收口索引：工作包→模块→门→证据）

依据《joint仓库当下+历史修改方案》§12.1 依赖图逐包施工；证据根
`/home/sunhao/joint_evidence_v30/`（仓外持久目录）。**w0–w6 证据原件毁于
2026-09-29 15:2x 的 /tmp 清除事件**（外部行为者整体删除 /tmp/joint_*），
关键结论以 `GB_GATE.md`（G-B 判定重建版）与根 `EVIDENCE.md`（W5 收尾
重建版）为据，代码面以仓内各 W 机制表行 + 广覆盖全绿为证；w7 起证据在
各包子目录。规范口径广覆盖（§13 命令）：基线 1703（W0 施工前）→ 终态
2098 passed / 1 skipped / 0 failed（W10 交付态，W11 收口复跑确认）。

| 包 | 交付模块/面 | 门/验收 | 证据位置 |
| --- | --- | --- | --- |
| W0 契约冻结 | `joint/schema_registry.py`（六类 schema 独立版本）＋`joint/prediction_config.py`（`JOINT_PREDICTION_CONFIG` 单入口 preset，`run_scripts/prediction_presets/*.json`） | 无独立门（冻结契约，后续包消费） | GB_GATE §6 毁损史注记；机制表 W3 行（v4 世代） |
| W1 快照/账本/桥 | `joint/runtime_snapshot.py`＋`joint/work_ledger.py`＋C++ `DecisionBridge.cc`/`NodeStore.cc` 资源快照 + `tests/work_ledger_bridge_test.cc` | V1 因果锚（test_runtime_snapshot/test_work_ledger） | GB_GATE §1 V1 行；W5 行③（正式计价接线） |
| W2 资源前推内核 | `joint/resource_projection.py`（事件队列/分段恒速积分/N-way 等分/buffer 门/守恒校验） | V2（test_resource_projection 36 用例；W7 附带 dust 钳位修复） | GB_GATE §1 V2 行 |
| W3 四动作工作图 | `joint/action_work_graph.py`＋两计价入口内部切换事件前推（`MODEL_SEMANTICS_VERSION` v4-projection） | V2/V3/V6 组合；G-B 前置 | 机制表 W3 行；GB_GATE §1 |
| W4 权威区间目录 | `joint/kv_location_directory.py`（缺失/来源查询 = 初等单元格并集去重——目标已持/在途不重搬重读、同内容多副本冻结序只选读一副本、部分重叠只补差集片段）＋face_scheduler 读入口部分迁移（distributed 准入视图/decode credit 计划/留存段驱逐走目录；写路径仍以 SessionKVState 旧字段＋`_distributed_retained` 账本为真值、目录经咽喉点同事务镜像重投影——§6.1"旧字段写入口摘除、目录单一真值"终态未达成，登记缺口非能力声明） | V6 投影等价（test_kv_directory_projection_parity/test_kv_location_directory/test_distributed_settlement 并集钉测） | GB_GATE §1 V6 行；机制表 W9 行（distributed 消费） |
| W5 事务/结算/账本计价 | SETTLE 状态机＋事务扩展（版本校验/引用/A22）＋`LedgerWorldSource` 正式接线＋性能修复（407s→9.3s） | **G-B 达成**（V1–V7 全过＋1s TJE 闭环＋16 臂矩阵 16/16） | `GB_GATE.md`（完整判定）＋根 `EVIDENCE.md`（W5 重建版）；矩阵重跑 `smoke_matrix_rerun/` |
| W6 因果历史统计 | `joint/history_estimator.py`（条件剩余长度/删失/纯服务样本） | 先 shadow；W5 起正式 n̂ 消费（V8 前置）；N14（2026-09-30，§8.2/A08）：任务负载视图 active_decode 分量同切条件剩余估计器（E[N−d|N>d] 下限 1——已消费超过会话均值的活跃请求保留正剩余负载，不再"均值减进度截零"估为零步；P/D/负载视图同刻同规则；horizon 保留诊断/先验，估计器版本入快照缓存键） | 机制表 W5 行④；GB_GATE §1 V1 行 |
| W7 新增需求预测 | `joint/background_forecast.py`＋`joint/forecast_validation.py`（成熟/距离准入/回退链/开销；2026-09-30 §8.7 语义修正：`history_use=shadow` 档**零注入**——全 crossing 只产桶预测与成熟误差，不影响正式选择；validated 档转正唯一依据 = 显式验证登记（`validation_registry`），run 内在线成熟误差只作 shadow 披露、不同 run 误差自评转正） | **G-H 达成**（V8 受控用例全过；validated 档无预置登记恒降级 shadow——不得声称生产获益） | `w7/EVIDENCE.md`（含 preset 数值来源登记表） |
| W8 同状态探针 | `run_scripts/joint_same_state_probe.py` 重构＋`JOINT_FORCE_*_TARGET` env＋`state_versions` 披露 | 工具验收（selftest 206 断言＋1s 受控 run 2 臂证书全匹配） | `w8/EVIDENCE.md` |
| W9 distributed 留存 | `joint/distributed_settlement.py`＋`settlement_policy=distributed` 双侧＋区间级 E＋开销计量 | **G-DIST 达成**（V9 全判据 PASS＋1s 双臂冒烟） | `w9/EVIDENCE.md` |
| W10 hybrid 扩展 | source_subset_hybrid（子集枚举/端点去重/预算/执行形态 i_P∈L） | **V10 达成**（全判据 PASS＋1s 双臂冒烟；pure_four 零漂移） | `w10/EVIDENCE.md` |
| W11 收口 | 裸仓还原/文档终态/旧假设残留排查/性能登记/§14 验收判定 | **G-FINAL**（§14 19 项逐项判定） | `w11/FINAL_ACCEPTANCE.md` |

### 3.2 修复轮（2026-09-30，§4 账本/投影/披露一致性批）

独立核实批次修复（文档依据 =《joint仓库当下+历史修改方案》§4/§5.1/§5.5/§5.6），
全部**零新增开关**（行为/披露面级变更，无 env/CLI/配置键增删；被替换旧行为直接
替换、无回退档）：

* **账本权威剩余量（§4.3）**：`work_ledger` delta commit 的资源腿
  接收面新增 `remaining_ops/remaining_bytes/remaining_latency_ns` 平铺字段与
  C++ 生命周期字符串状态映射（`Streaming→in_service`/`LatencyWaiting→issued`，
  未知名 fail-closed）——携带权威剩余量的中途首见腿 `progress_source=
  authoritative`、`served = total − remaining`（守恒线自首见点保持）；**未携带时
  `remaining` = 总量上界占位（stale），`conservation_report.stale_amount_works`
  计数披露**；`LedgerWorldSource` 对非 authoritative 腿**逐键**登记
  `ledger_leg_remaining_unverified:{key}={n}`、无量纲腿逐键注册表回退照旧披露
  （2026-10-02：C++ 桥 `DecisionBridge.cc`/`main_online.cc` 的 commit/progress
  delta 已携带 `remaining_*`；HBM/池/NoC 快照在当前交付 tick 同步，Python
  拒绝倒退进度及有冲突的重复事实）。
* **快照资源/容量面（§4.1）**：`WorkLedger.export_state()` 导出 C++
  `work_ledger.state` 同构形态；sh30 `_joint_decision_snapshot` 自账本累积状态
  传 `ledger_state` + kv_manager 容量账/摘要版本投影 `capacity_view`——delta 帧
  不再静默空骨架；`build_decision_snapshot` 对"delta 帧无 ledger_state"与
  "capacity_view 缺席"两条降级路径显式登记 projection_log。
* **背景延迟/复杂度交付面（§5.1/§5.6）**：`ActionCostBreakdown` 扩至 31 字段
  （+`projection_stats`/`background_delays`/`capacity_theta_prediction`）；三估价
  入口（P/D/hybrid）计时并透传 `ProjectionResult` 计数与逐背景工作候选反作用
  对照（仅受影响工作）；`_JOINT_BREAKDOWN_LOG_FIELDS` 同步 31（run 级聚合见
  决策指标行 `projection_cost_accounting` 键）。
* **NoC 除数不确定性（§4.4/A03）**：`TelemetryLinkFlowView` 构造登记遥测时间窗
  与来源、构造期急切记录注册表↔遥测流数逐链路冲突（两侧值/谁覆盖谁/来源），
  `disclosure()` 新增 `divisor_uncertainty`/`registry_telemetry_conflicts`/
  `telemetry_window_ns`/`telemetry_source`；`CandidatePathNocShare.disclosure()`
  标 `divisor_basis`；决策行审计块新增 `noc_telemetry_divisor`；
  `LoadsWorldSource` 遥测 max 口径背景腿改标
  `link_registry_telemetry_max_window_mean_inferred_unknown_eta`。
* **容量摘要版本与 θ 对账（§5.5）**：`InstanceCapacitySummary`/`CapacitySummary`
  携带 `snapshot_version`（KVCacheManager 每次增量维护 +1）/`maintenance_event_
  count`/`error_sources`（θ 份额连续近似 + 增量维护滞后）；`_space_ready_ns_
  estimate` 经 `prediction_out` 输出 θ 结构化预测（随 breakdown 落决策行）；
  kv_eviction 决策行新增 `capacity_theta_reconciliation` 对账块（预测 released
  向量 + 实际逐出累计 + 差值；按 request 绑定、无预测请求键缺席）。
* **同 tick 可见性钉子（§4.2/W1）**：test_work_ledger 新增账本通道交付级
  可见性用例（delta commit 在同交付 run_variant_policy 内对账本活跃集与
  DecisionSnapshot work_index 立即可见）与"中途首见腿 full/delta 等价"分支。

### 3.3 深审修复（2026-10-02 至 2026-10-03）

模型语义版本为 `v5-projection-20261002`，预测配置版本为
`predcfg-v2-20261002`；旧配置不自动迁移，旧结果不能作为新语义的证据。
混合远读工作副本分别保存 S 的 suffix 矩形和 I 的 resident prefix，目录、
迁移、逐出、恢复及容量摘要统一消费；PARTIAL 远读补齐 S 的实际恢复腿。
多源 credit 按真实 layer/token 片段生成，不能把 suffix 当全层 prefix。
边界携带 TP 各 rank 的完整层 token 字节基准，支持整头不均匀分片；
P=copy/recompute 的完整副本在 D 远读切换后仍登记完整 token 域，后续
copy 不重复物化基础历史。distributed 结算的完整上下文按驻留 token
偏移与本地增量共同计算，包含边界前已有的输入。

在线计算 profile 同源拆出纯 FLOP 时间、每 rank 的完整 HBM 字节量以及
HBM 启动/控制器固定时延。效率因子只修正纯计算时间，字节量守恒；聚合
HBM 启动放在首个对应计算腿，控制器时延由独立固定腿计入，并披露
`aggregate_phase_latency_folded`。NoC 链路遥测的时间倍数通过
`η=1/transfer_factor` 修正正式前推的链路服务率，不套用到 HBM 或池端口。
计算与 HBM 两侧按积分前的完成时刻归零并结算，保留尚未到期的浮点余量，
避免接近同时完成时丢失事件、误报不可推进；亚时间戳精度的边界也会释放带宽。

历史到达预测按各桶 `[start,end)` 的实际到达计数成熟；左右边界由模拟时间
推进捕获，迟交付不纳入桶外请求。模板预算在展开请求前检查，非法强度进入
明确回退。预测配置拒绝分数计数、布尔冒充整数及非有限数值；验证登记须达到
冻结样本量、平均绝对归一化误差与选择损失门槛，并匹配完整参数 preset、
模型语义版本及独立验证前缀来源。`service_efficiency_init=η` 发布时间倍数
`1/η`；离散长度桶严格按 `N>d` 排除已结束质量，非法样本不半写历史。

最小短 trace 不能提供剔除 15 秒 warmup 后的统计；当前三槽水印重放器
也不支持 distributed 跨轮多位置布局，这两项后处理按既有合同拒绝出值。
distributed 仿真正确性由执行期 KV 目录守恒与 rank 台账对账裁决，不能
把主链通过解释成已经获得 certified HBM 水印。

生产新增需求模板尚无候选目录/容量/驱逐内物化及可按需延长的共同到达生成器。
`ForecastAugmentedWorldSource` 因此恒隔离为 shadow，披露
`template_uncovered` 和 `missing_candidate_state_and_extensible_arrivals`，
有效模式见 `forecast_status` 的 `priced_mode/last_plan`。验证登记也不能越过
该能力门；桶预测继续成熟，已知工作 B 模式照常完整前推。受控模板现在使用
纯计算/HBM profile，仍不能宣称生产 C/D 计价已完成。W4 的目录单一写入真值
迁移也仍是结构性缺口；本轮没有把镜像目录改写成已经满足 §6.1 的能力声明。

同状态探针只有证书匹配、强制选择生效且 P 的动作和实例身份匹配的分支才进入
排序及选择损失；证书必须包含完整强制前队列、工作、目录、容量、仲裁和历史
事实及各视图的必需结构，不能以空对象代替。普通 baseline 未采集完整证书时，
探针使用与其实际选中动作/实例一致的强制分支作为采证参照，同时核对普通
baseline 已有的边界事实，并披露 `baseline_source=forced_selected_reference`。
现代 consolidate/distributed 日志缺有效结算事件都保留 settle 真值
缺失。离线指标读 `bridge/joint_mechanism_manifest.json` 的 `prediction` 节。

完整 request 指标覆盖预定请求集合。观察窗内尚未释放到达的请求标
`not_arrived`，已到达而未完成的请求标 `in_flight`；不凭缺少 completion
推断失败。SLO 后处理保留两类在分母中，并报告完成率和两类数量。

## 4. 构建、物化与运行（最小命令链）

（2026-10-01 审查补标题：本命令块此前缺失章节号，extern/network_backend
的 CMake 报错文案与各处"README 构建节"引用悬空。）

```bash
cd <本仓根>
# ① 构建（与 sh_3.0 同款；裸仓无 build/：先放拼装 CMakeLists）
#    mkdir -p build/astra_analytical && cat > build/astra_analytical/CMakeLists.txt <<'EOF'
#    cmake_minimum_required(VERSION 3.22)
#    project(AstraSim_Analytical_Build)
#    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../ astra_sim_build)
#    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../extern/network_backend/analytical backend_build)
#    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../astra-sim/network_frontend/analytical frontend_build)
#    EOF
cmake -S build/astra_analytical -B build/astra_analytical/build_congestion_aware -DBUILDTARGET=congestion_aware -DNETWORK_BACKEND_BUILD_AS_LIBRARY=ON
cmake --build build/astra_analytical/build_congestion_aware -j
# ①b native 投影库：常规构建产出共享库 libjoint_projection.so（固定
#     输出 ${CMAKE_BINARY_DIR}/lib/）与 libdie_model_cabi.so。Python
#     生产端（joint/native_projection.py）只从常规 build 位置查找；
#     缺失或 ABI 不符即报错，**无旧 Python 投影回退**。验收/测试注入
#     入口（library_path 形参 / JOINT_NATIVE_PROJECTION_LIBRARY /
#     DIE_MODEL_CABI_PATH）只接受同一 ABI 的库，不构成实现选择开关。
# ①b-2 native 双账本接收器（joint 加速改造 v4，2026-10-12）：同位置产
#     出 libjoint_ledger_cabi.so（extern/joint_ledger；协议 v4 双账本
#     的解析/校验/应用/提交/查询全部驻留原生侧）。Python 生产端
#     （joint/native_ledger.py）同样只从常规 build 位置查找，缺失即
#     fail-closed，无 Python 账本回退；work_ledger.py/pipeline_ledger.py
#     保留离线档参考解码器与等价性对照测试用途。wire 协议不变
#     （bridge 2 / workledger-v4-20261011），无新增开关。
# ①b-3 v5 投影共享上下文与紧凑 run span（joint 加速改造 v5，2026-10-12）：
#     libjoint_projection.so 的 jproj_project 载荷改为对象形态
#     {"candidates":[...], "private_overlay":{...}}（决策级共享上下文 +
#     逐调用候选私有 overlay，reference 运行按 overlay 内容记忆化；裸数组
#     形态退役）；die_op_spec.spans 允许混合 [span|run] 记录（run 在原生
#     解析边界无损展开），DIE 计划缓存键升 v2（内容寻址，剔除合成
#     request_id 与 tensor_parallel_rank）。均为内部 FFI/内存缓存演进，
#     无新增用户可见开关；Python 生产端（joint/native_projection.py、
#     online/die_model_adapter.py）已同步，缺库/形态不符即 fail-closed。
# ①b-5 v7 事件长尾归因诊断 + HBM 执行器状态索引重构（joint 加速改造 v7，
#     2026-10-12）：新增构建期诊断选项 JOINT_EVENT_PROFILE（CMake，缺省
#     OFF；启用后 EventProfile.h 在后端队列三相位（主事件/tick-end/
#     deferred）与 Sys 子分发层做回调分类计时+owner 直方图，经 [MainWall]
#     心跳转储；生产构建零插桩指令、仿真语义零变化）。生产重构
#     （LocalHbmBandwidthModel，直接替换）：DieExecution 完成即撤 HBM
#     mutation observer（撤除仅跳过可证 no-op 回调）；观察者分发改稳定
#     deque+墓碑+通行序列号（快照语义与旧逐通知拷贝逐点一致，免
#     O(N) 拷贝）；jobs 稳定槽位 + latency/stream/compute 相位索引 +
#     完成候选集 + drain 计数（消除每事件全量扫描；服务算术/浮点顺序/
#     observer 边界/资源版本逐操作保持）。新增诊断访问器
#     transition_event_count()/resource_observer_count() 与闭环组件回归
#     AstraSim_Analytical_Congestion_Aware_HbmTailComponentTest（已注册
#     CTest；512 节点串行 die 链闭式 oracle + canonical A/B 字节等价）。
#     归因结论与性能口径见《joint改造结果v7.md》：尾段 97.6% 在
#     hbm_transition 类，其中 99.8% 在完成回调链（本轮已排除 HBM 模型
#     自身算术/扫描与观察者分发），组件 A/B 1.19×、完整任务门未过。
# ①b-6 v8 完成回调链归因收口 + observer 墓碑回收修复（joint 改造 v8，
#     2026-10-12）：修复 v7 观察者墓碑压缩的不可达分支（guard 存活期判
#     depth==0 恒假）——压缩改由 ObserverPassGuard 析构在最外层通行结束
#     触发（含异常退栈，原地 noexcept 搬移），新增
#     resource_observer_storage_size()/resource_observer_tombstone_count()
#     真实计数访问器；hbm_tail_component_test 扩展回收/嵌套/异常用例与
#     K2 生产合同臂（[CANONICAL_K2]）。die_model Execution 新增 preview
#     E/T/U/F 常开诊断计数器（active_id_* 同口径）。
#     【2026-10-13 v9 纠错】本条原文"98.7% 在 machine_.next_action()"的
#     归因系 v8 PROFILE 头文件 nxt/preview 标签互换所致，已撤回——真实
#     热点是 preview_consecutive_internal_timers（105.66µs/次，四前沿
#     全量复制），next_action 为 0.029µs。见《joint改造结果v9.md》§2。
#     v7 的 retire/watch/transfer 候选排除结论不受影响。详见
#     《joint改造结果v8.md》（顶部纠错指引）与仓外
#     /home/sunhao/joint_accel_v8_20261012。运行期用户开关零增删。
# ①b-7 v9 精确预演事务：消除 preview 四前沿全量复制（joint 改造 v9，
#     2026-10-13）：Execution::preview_consecutive_internal_timers 的
#     PreviewBorrowGuard 每调用整体复制 heap/events/future_starts/
#     active_ids 四容器（生产尾段 avgF≈35 万条目/次、105.66µs/次——v8
#     归因纠错后的真热点）退役，替换为 PreviewTxn 日志式精确事务：heap/
#     events 的 push/pop/逐槽写在 sift 前预留最坏路径容量并写前记录
#     （HeapJournal，逆序回放、回滚 noexcept 不分配）；future_starts/
#     active_ids 经调用链证明为预演调度与返回值不可读的派生结构，事务
#     内不维护（真实路径完整保留）；真实推进与预演共用 sift_up/sift_pop
#     单一实现与同一条 run() 控制流。E/T/U/F 计数器升级：U 全出口统一
#     收口、F 结构性归零并与四容器 S 分离，新增 J/S 峰值/退出分类/容量
#     增长（Execution::preview_counters()，只读）。internal_service_
#     progress_test 新增计数标定/大前沿孪生等价/规模阶梯（S=131,072 时
#     BASE 1.80ms vs WORK 0.52µs≈3,445×）/分配失败扫频（测试进程
#     operator new 倒计时，49 档真实 bad_alloc 全恢复）/真实 spec 完整
#     环（qk prefill q=61550，BASE/WORK 1.372×）五臂；hbm_tail_component
#     _test 新增 V9 大前沿生产臂（真实 DieExecution/HBM/K2，canonical
#     BASE/WORK 逐字节等价）。完整 first-1s 配对因预算未完成（BASE 172s
#     超时于 delivery 36 尾段，WORK 未发射），整体倍率如实 null。详见
#     《joint改造结果v9.md》与仓外 /home/sunhao/joint_accel_v9_20261013。
#     运行期用户开关零增删；异常注入仅在测试二进制。
#     【2026-10-13 v10 纠错】本注记两处修正：①"BASE 超时于 delivery 36
#     尾段"的"tick 冻结"推断撤回——MainWall 原件显示 tick 持续增长（同
#     tick 事件洪峰下推进极慢非卡死）；②v9 验证段的"HbmNway…全绿"当时
#     为缺参假绿（断言未执行），v10 已补全参实跑双绿。见 v9 结果顶部
#     v10 纠错指引。
# ①b-8 v10 虚拟端点成员资格索引（joint 改造 v10，2026-10-13）：v10
#     PROFILE（唯一 first-1s 生产取证，[V10P] 具名分桶+witness+策略
#     cProfile）实测 die 引擎侧 A1/A2/R/T 候选独占份额全部 ≤3.5% 不可
#     防御，真热点为 Python 策略侧 is_virtual_endpoint 对
#     virtual_endpoint_ranges（等差数列 (first,count,stride)，跨交付单调
#     增长至数千条）的每查询线性全量扫描（两重型 handler 内占 policy
#     45%，单次 235→427µs 随运行上涨）。生产直接替换（无开关）：
#     _VirtualEndpointLookup 精确索引——按 first 排序副本 + last 的 max
#     线段树，查询 bisect 截断 + 双向整枝剪除，仅对区间覆盖候选做与旧
#     扫描逐字相同的模测试（any-match 与顺序无关；正确性不依赖区间
#     不相交或注册有序，乱序走防御性重建仅慢不错）；注册侧 sync 从
#     规范列表推导。test_virtual_endpoint_index.py 7 项合同测试（全 id
#     域等价/三合并分支/重叠/乱序/生产量级）。正式配对（双臂 172s、
#     二进制逐字节相同、决策日志剥宿主字段后等价、36 帧字节等价仅差
#     生产者身份串）：两臂均未自然完成整体倍率 null；共同逻辑尾段窗
#     1.847×、全窗 1.352×、policy handler 49.63s→15.75s（−68%）；机制
#     阶梯 115.7×/309.0×（合成形态单列）。详见《joint改造结果v10.md》
#     与仓外 /home/sunhao/joint_accel_v10_20261013。运行期用户开关零增删。
# ①b-4 v6 紧凑账本 wire 与 typed 接收核心（joint 加速改造 v6，2026-10-12）：
#     work_ledger 段升级为二进制紧凑编码 workledger-compact-v1（外层
#     bridge JSON 不变；C++ 生产端回调直接产出 typed 记录——义务行只携
#     带 (segment, run_index) 坐标与动态标量，身份/世代/游标可推导字段
#     由接收端重建；紧凑帧内字符串走帧内字典引用）。原生接收
#     （extern/joint_ledger）对热行（pipeline runs / slot / credit /
#     obligation / cursor / resource）全部 typed 化：解析即结构体、
#     staging/commit 直接消费结构字段，仅查询边界（pricing/snapshot/
#     export）序列化小 JSON；旧 JSON 帧经版本 reader 进同一 typed 核心
#     （归档可读，无双状态机）。journal 侧车升级
#     request_journal_work_ledger.compact.bin.gz（长度帧记录，截断按
#     完整记录前缀恢复），旧 jsonl.gz 侧车 reader 保留。外层
#     ledger 语义版本仍为 4；无新增用户可见开关。
# ①b-9 v11 §8 保底：决策桥帧载体文件→shm 替换（2026-10-14）：C++↔
#     Python 决策桥的 request/response/commit_ack 传输从散装 JSON 文件
#     （tmp+rename 原子写 + 文件存在性轮询 + unlink，每交付 ~0.7s 文件
#     开销）替换为 POSIX-shm 单槽帧通道（ShmFrameChannel.{hh,cc} +
#     online/bridge_shm.py；FileDecisionBridge→ShmDecisionBridge 直接
#     更名替换，无运行期开关）。帧格式 [u32 magic][u64 seq][u32 len]
#     [payload][u32 crc32]（小端，IEEE crc32 覆盖头+负载；槽容量 256MiB，
#     超限/损坏/crc 不符 fail-closed），帧负载与旧文件内容逐字节相同
#     （work_ledger 二进制拼接仍在 C++ 侧完成后入帧）；FIFO 门铃字节、
#     1:1 背压、poll 超时、双字节协议违例检测、启动顺序（先 C++ 后
#     Python、FIFO 就绪轮询）全部照旧。段名含 bridge_dir 哈希 + 启动
#     世代，指针文件 bridge/bridge_shm.json 由 C++ 原子写、Python 启动
#     校验世代（跨 run 陈旧段拒绝）；ensure_bridge_dir 仍建目录+FIFO+
#     清陈旧（含按前缀清 shm 段）。可见变化：bridge/ 不再有散装
#     request_/response_/commit_ack_*.json；request_journal.jsonl 与
#     work_ledger gz 侧车照旧为审计字节流；崩溃现场改为 abort 残留的
#     /dev/shm 段（析构 unlink，abort 不跑析构）。统计口径不变
#     （channel_bytes=帧负载字节=旧文件字节；forced_flush_count=每次
#     发布一计；roundtrip 计时不变）。单测：C++ 进程内
#     ShmDecisionBridgeLoopbackTest（字节往返/超限/crc/陈旧世代/三向
#     逐字节/统计/清理，已注册 CTest）+ 跨语言 BridgeLoopbackTest
#     （真实 python3 对端，journal 字节断言）+ pytest
#     online/test_decision_bridge_shm.py（帧编解码/世代/journal 字节
#     保持）。
# ①b-10 v11 Phase 4：issue pass 增量拼接 + K2 窗口查询纪元 memo
#     （2026-10-14）。① ready 快照增量拼接：NodeStore 的 ordinary 自由
#     分区快照（原 fill_free_ordinary_id_snapshot 每次全量遍历拷贝
#     std::set）改为"上一代快照向量 − 擦除增量 ∪ 插入增量∩当前活集"的
#     拼接（splice_free_ordinary_id_snapshot；NodeStore 维护插入/擦除
#     双增量集与版本窗口守卫，窗口失效回退全量 fill）；Workload 的
#     issue_dep_free_nodes 在 reuse 与 full fill 之间接入 splice 分支。
#     等价性：拼接结果与全量 fill 逐元素相同（对拍测试
#     node_store_test 的 F2 差分 + ready_issue_order_test 三场景钉死；
#     被资源闸门跳过仍留在集合的 id 由"留在向量"覆盖重查）。同
#     (tick,rank) 多次排队型触发的合并在既有 per-rank pending 旗标已
#     存在，同 tick 后续 pass 是"同轮新 ready"钉死语义，不再合并（登记
#     未做）。② die_model 的 query_internal_timer_window 只读证明查询
#     加执行状态纪元 memo：Impl 全部 7 个变更入口自增 state_epoch，
#     按 (纪元, max_count) 记忆查询结果（含拒绝理由）；纪元变即失效
#     ——同状态同参必同结果（read-only 契约 + 变更面审计注释），
#     PreviewTxn 经入口自增纪元只过度失效不供旧值。孪生 oracle 单测
#     extern/die_model/tests/k2_timer_window_memo_test.cc。两项均无
#     新增用户可见开关。
# ①b-11 v12 Phase 4 K2 否定缓存 v12 契约（joint 改造 v12，2026-10-13）：
#     query_internal_timer_window 在 v11 纪元 memo（同层、随状态失效，
#     完整保留）之外新增进程级结构否定缓存（extern/die_model 的
#     DieTypes.hh 与 Execution.cc；方案 §6.2）：双键 =（die 计划内容
#     寻址缓存键 + 窗口近似配置指纹），不含任何运行状态，可跨
#     Execution 实例与运行状态存续——因此只允许缓存"在计划的任何可达
#     状态下、任何驱动历史下，完整返回值（含 reason 串）逐字节恒同"
#     的拒绝。Sound 性由构造期一次计算的结构谓词 P_struct = P_A∨P_B
#     保证（P_A：total==0 构造即 finished；P_B：全部实例为零时长内部
#     stage 且无 MEM/MEM_NOC 实例，构造期 run() 依零时长规则同步排干
#     整个 DAG）——该类计划可达状态集为单例 {finished}，查询必返常量
#     拒绝 {supported=false, reason="execution-finished"}；谓词同时把守
#     探测与写入两侧，supported=true 永不入缓存。近似配置指纹在生产
#     侧为 die_model 内编译期注册表（tile_limit=2 / local_error_ppm=0
#     / refine_margin_ppm=20000），不同指纹条目互不可见，精确/近似
#     模式永不共享结论。审计结论：生产计划必携正时长 launch/compute/
#     memory，P_struct 恒假——生产点燃面为空，交付价值 = 已审计的
#     指纹键控 Sound 接线与测试钩子（negative_cache_hits_for_test()
#     等只读访问器）；C++ 侧 set_enabled 主开关缺省启用、仅测试二进
#     制使用，非用户可见开关。双生 oracle 测试
#     extern/die_model/tests/k2_plan_negative_cache_test.cc（计划族×
#     全速/2x/4x 内存驱动的启/停缓存逐字段等价、P_A/P_B 计划实际命中
#     而真实计划零写入零命中、指纹隔离、主开关、参数校验；已注册
#     CTest）。
#     v12 窗口资格取证结论（Phase 5 门禁输入）：生产 PROFILE0 全部
#     4,136 个 die 计划 × stretch{1,2,4} 内存速度、每档约 1,030 万
#     边界（10,305,050）穷尽实测，k≥2 可平均窗恒零（0.00%）；四桶
#     根因逐桶判定唯"确无合格片段（负载结构）"成立（其余三桶
#     否定）——timer 片段被 MemoryRequest 边界切成单片段，窗口退化
#     K=1，§6.1 局部平均算子在本负载无平均对象。Phase 5 资格
#     NO-GO：近似线不落地，无近似参数、无新开关。详见
#     《joint改造结果v12.md》与仓外证据根
#     /home/sunhao/joint_accel_v12_20261013。运行期用户开关零增删
#     零取值变化。
# ①c （已退役 2026-10-14）紧凑描述符 + 专用执行器：远读生产路径的
#     紧凑 PipelineDescriptor（schema 1 typed wire）、online/
#     graph_batch_builder.py 的原发射点构造与 finalize_pipeline_
#     identities 身份提交、执行端 RemoteReadExecutor +
#     TileTransportAdapter 及 finish() 级仿射 run 合并，已随
#     remote-read 机制整体删除（无开关保留档）；原 PipelineDescriptor.hh
#     尾部的 PipelineKernelContract 迁出为独立头
#     astra-sim/workload/execution_driven/PipelineKernelContract.hh
#     保留（die 计算后端 window 契约，每批无条件发射），登记见仓根
#     PROVENANCE.md 退役批登记节。
# ② 物化输入（唯一允许源 = agent-traces/tracelab/astra_compute_20.csv；
#    窗口 ≤ 前 30 秒为物化器冻结契约，缺省整 30s——冒烟矩阵用 2 秒窗
#    `WINDOW_NS=2000000000`，见 joint_smoke_matrix.sh）：
python3 sh_test_mesh/workload/llama2_7b_inference/traces/materialize_20_30s.py     <绝对路径 astra_compute_20.csv> <绝对路径 …_request_queue.csv> 30000000000
# ③ plan 物化（runtime_config 四件 + manifest；不产队列 CSV）+ ④ 运行。
#    注意：trace_config.csv 的 request_queue_csv 行缺省指向占位文件（裸仓
#    request-neutral），③ 前须把它改指 ② 物化出的队列 CSV（物化器 stdout
#    的 [next-steps] 行即此提示），否则 ③ 在队列装载处 fail-closed 退出。
cd sh_test_mesh/workload/llama2_7b_inference && python3 plan_materializer.py && cd <仓根>
python3 sh_test_mesh/run_scripts/joint_runner.py <run_dir> <绝对路径 request_csv> --combo TJE
# 等价直跑（不经 runner 的锁/哈希/清洗层）：
# JOINT_ABLATION_COMBO=TJE bash sh_test_mesh/run_scripts/run_online_strategy.sh <run_dir> <绝对路径 request_csv>
# ⑤ 指标后处理/对账/SLO 提取/⑥ 裸仓还原：与基底同款
bash sh_test_mesh/run_scripts/run_metrics_postprocess.sh <run_dir>/cpp.log
bash sh_test_mesh/run_scripts/clean_test_records.sh && bash sh_test_mesh/run_scripts/clean_build_artifacts.sh
```

## 5. 与基底 sh_3.0 的语义差异（迁移清单）

1. **准入**：三段式准入（首请求避边缘/LOCAL·PARTIAL sticky/REMOTE 边缘
   负载均衡）与 PARTIAL 钉扎**整体移除**，替换为 joint 框架三模式
   （§7.1）；HBM 容量不再作为准入过滤（只影响动作计价中的驱逐等待）。
2. **decode 选点**：恒同执行 instance（§1.1）；基底 no_affinity 的
   drain 均衡迁移随 SH30_ABLATION 退役。
3. **预约口径**：准入预约只覆盖已知 prefill 上下文；decode 逐列车因果
   增长（§3.1；基底在 drain 一次按真实最终长度扩容的 oracle 用法移除）。
4. **跨实例历史**：noc 迁移（move 语义）→ copy 工作副本 + 完成时
   merge_back **v2 少并多**（2026-09-17：两侧保留量比大小、小侧整份
   搬大侧、零池写、全层 LOCAL@胜者、home 迁移——§2.2；基底
   copy/recompute 顺便改写 session 位置的行为已修正）。
5. **逐出**：类别序经 T 模块、层数经 E 策略；`legacy_half` 缺省保持与
   基底逐字节等价（供回归对照）。
6. **完成路径**：完成批新增 merge 回传流发射；响应交付
   `service_done` 与资源结算 `merge_done` 分别记时。下一轮到达以
   `service_done` 为锚，后续数据就绪由 merge 尾门控。
7. **远端内存端口（2026-09-24 SerDes 片外链路并发化改造）**：基底
   串行 FIFO 端口（一事务完成后再发下一笔）→ 并发在途流体端口
   （§7-D'）；发射门控远端 MEM 与 comm 槽解耦（comm/hbm_dma 槽 2026-09-25
   起同为计数制无上限）、static 仿真结束门计入远端 MEM 在途（§3 状态表
   "远端端口并发后端"行）。
8. **发射门槽位与图形态（2026-09-25 PARTIAL 跨实例 copy 流水化）**：
   comm（COMM_SEND/COLL）与本地 KV restore（hbm_dma）发射门由单槽改
   计数制无上限槽（带宽仲裁归既有 HBM N-way／NoC／池端口模型，§7-G；
   COMP/CPU 单槽门保留）；copy 交接尾块、恢复组由逐笔串行链改逐笔并行
   支链／组间并行（首块准入与栅栏、逐层就绪
   门、账本语义不变——§3 状态表 C13/C15 行，PROVENANCE
   §45；remote-read credit 尾块形态已随 2026-10-14 远读退役删除）。
9. **（已退役 2026-10-14）prefill remote-read 分阶段（2026-09-25，
   零新增开关）**：remote-read 轮 prefill 前缀读流与后缀池恢复腿的
   分阶段关键路径（prefill_stage + decode_stage）已随 remote-read
   机制整体删除（无开关保留档）；历史描述见 PROVENANCE §46/§47 与
   仓根 PROVENANCE.md 退役批登记节。

## 6. die 内成本模型（运行时 + 预测两层接口）

本仓已接入 `extern/die_model/` 共享成本库（vendoring 与 face 仓字节一致）。
`system.json` 键 `compute-cost-model`（`die`/`roofline`/`replay`）选择 COMP
计时后端；`die-model-config` 指向 resolver 物化的 `resolved_die_model.json`
（配置链与冲突矩阵见仓外契约 §3；legacy roofline 后端显式保留供回归）。

**运行时（一期）**：die 后端下真实 COMP 节点携带 `die_op_spec`（生成侧
`online/die_model_adapter.py` 按契约 §2 构造、13 类聚合类别经 §7.2 region
编译），C++ `DieModelAdapter`/`DieExecution` 按 A3 执行协议推进（阶段状态
机；内存阶段请求三元组交 `LocalHbmBandwidthModel` 的 `DIE_COMPUTE` 作业
等分仲裁——`actual_bytes`/`service_units` 双量分列）；marker 微节点走旧
微作业语义，缺 spec 的真实 COMP fail-closed（不静默回退 roofline）。

**预测层（二期，设计文档 §9.6 两层接口）**：joint 的策略计价
（`JointCostModel`）与运行时同源——

1. **名义 service 查询**：`_joint_cost_model` 在 die 后端注入
   `_JointDiePlanProvider`（`online/die_model_adapter` 构造 DieOpSpec +
   `online/die_model_oracle` ctypes 直连 `libdie_model_cabi.so` 的
   `die_estimate_batch`/执行原语；Python 不写第二套公式）。
   `prefill/decode_compute_profile` 返回 `(非内存资源时间 ns,
   service_units)` 双量；既有 `hbm_share_divisor=s` 语义改为**仅预测
   内存服务率 B/s** 推进该计划（service_units ÷ (B/s)），不整段 ×s、
   NoC/PE/vec 不降速。
2. **投影 compute leg**：`TaskSpec.die_plan`（`joint/die_plan_projection.py`
   的 `DiePlanDescriptor`）经 C ABI 执行原语由 `resource_projection.
   ProjectionWorld` 推进——timer 阶段 = 内部时长（不占 HBM 分享集合）、
   memory 阶段 = service_units 等分排空（分享集合/除数仍由 ProjectionWorld
   裁决；actual/service 双量传递）。P 入口逐层段腿与 D 边界静默重建腿
   按层段 DieOpSpec 协议驱动；decode 地平线因逐 step 计划的 tile 级展开
   （~3.6e5 协议动作/链）不可行，以同源聚合双量走 compute 腿（notes 披露
   `die_decode_leg=aggregate_service_estimate`）。
3. **退役回退**：`(pure_compute_ns, hbm_bytes)` 聚合与
   `service_time×HBM_bandwidth` 等效字节构造仅 legacy roofline 后端保留；
   die 后端的 `prefill/decode_compute_profile` 不再走该回退（TaskSpec 层
   fail-closed 拒绝聚合面），逐层段/块工作从其 DieOpSpec 编译。
4. **service factor**：只缩放预测 service_units（描述符内存阶段经
   `ScaledDieExecution` 包装），物理 actual_bytes 与硬件校准系数不动。
5. **策略面零改动**：候选枚举、quota、剩余步估计、telemetry、
   copy/remote/recompute 选择规则与在线 service factor 更新机制保持原样。

验收（§10.2 预测一致性）：`joint/test_die_plan_projection.py`——同 spec
ctypes 估值/执行自洽（C++ 同值由库与 A3 ctypes_parity 保证，抽例登记）、
投影 parity（同一计划 + 同一服务率轨迹 == 手算等分仲裁，容差 =
预约次数×1ns 级）、不把 service 时间伪造成 bytes（聚合面 fail-closed +
双量分列钉测）。

## 7. 测试

```bash
# ① 全量 Python 回归（窄命令，含 joint/ 与 online/）
cd sh_test_mesh
python3 -m pytest tests/ slo_tools/tests/ workload/llama2_7b_inference \
    --ignore=slo_tools/tests/test_driver_parity.py \
    --ignore=slo_tools/tests/test_golden_g1g4.py \
    --ignore=slo_tools/tests/test_slo_contract.py -q

# ② 注册 CTest（新 native/fixture targets，全量、无 -R/-E）
ctest --test-dir build/astra_analytical/build_congestion_aware \
    --output-on-failure
# 登记包括 NativeProjectionCApiTest / JointPipelineWindowTest /
# DieExecutionLifecycleTest / JointReadyIssueOrderTest /
# JointPipelineMutationTest 及 die_model 单测；先用 ctest -N 核对。
# ParsedBatchTest / GraphBatchCommitterTest 等需直跑。
# （JointPipelineDescriptorTest / RemoteReadExecutorTest 与
# JointPipelineExecutionFixture 已随 2026-10-14 远读退役删除。）

# ③ 现有 C++ fixtures（不经 CTest 的直跑目标；BridgeLoopback 传 echo 脚本）
build/astra_analytical/build_congestion_aware/bin/AstraSim_Analytical_\
Congestion_Aware_{ParsedBatchTest,GraphBatchCommitterTest,BridgeLoopbackTest}

# ④ 冷夹具两段链（§19.1A）：Stage A raw → Stage B canonicalize
#    Stage B: joint/acceptance_projection.py::canonicalize_raw_frames
#    （原 Stage A 驱动器 JointPipelineExecutionFixture 已随 2026-10-14
#    远读退役删除——descriptor 执行链不存在，冷夹具链仅保留
#    canonicalize 面）

# ⑤ L3 批量评估（E0/F0 输入）：l3-batch/evaluate_l3_boundaries.py

# ⑥ 前 1 秒必要冒烟（唯一许可 trace 窗口）：
#    astra_compute_20.csv 前 1s 8 请求，入选请求完整自然 decode
```

**已退役入口**（2026-10-07 C5/F2）：`joint/test_resource_projection.py`
与 `joint/test_joint_projection_pricing.py`（仓外冻结脚本执行，证据根
`packages/python-impl-20261006/c5-prep/frozen-tests/`）；Python 投影
事件循环（`ProjectionWorld` 前推路径）已由 native 投影唯一替代。
**远读退役批（2026-10-14）另整删测试**：Python 侧 `joint/` 8 个
（test_d07_remote_scan_derivation / test_d09_remote_read_double_buffer /
test_joint_remote_read_staged_path / test_joint_credit_pricing /
test_source_subset_hybrid / test_sparse_pipeline_integration /
test_pipeline_descriptor / test_compact_pipeline_range）＋`online/` 14 个
（prefill_remote_read 五件、remote_credit 三件、
test_prefill_read_summary_compat、test_r4_mixed_remote_read_train_pin、
test_w10_hybrid_execution、test_growth_reservation_hold、
test_pipeline_descriptor_submission、test_compact_credit_gate_ranges）；
C++ 侧 remote_read_executor 双件、pipeline_descriptor 测试双件与
pipeline_execution_fixture_main.cc（其 4 组 PipelineKernelContract 断言
并入 pipeline_window_test.cc）；slo_tools 侧
test_hbm_watermark_staging_lease.py（词表 3 用例迁入
test_hbm_watermark.py）。joint/online 其余受波及测试已按三动作改写
（joint 1021 绿 / online 636 绿 / 脚本与 slo_tools 284 绿）。

### 7.1 历史沿革口径注（各批次计数基线，保持可比）

```text
以下 # 注释为历代验收口径注原文保留（W0–W11 → C/F/G/R 批 → D0–D1
epoch → E0/F 链），供批次计数溯源；最新 F 链终态见 §2.9。
python3 -m pytest tests/ slo_tools/tests/ workload/llama2_7b_inference \
    --ignore=slo_tools/tests/test_driver_parity.py \
    --ignore=slo_tools/tests/test_golden_g1g4.py \
    --ignore=slo_tools/tests/test_slo_contract.py -q
# 口径注（M7，2026-09-23 验收审计）：本命令对 slo_tools 三文件
# （driver_parity/golden_g1g4/slo_contract）显式 ignore——三文件在
# 曾在 HEAD 为 3 红（driver_parity×2 + slo_contract×1 陈旧 fixture）；
# O13 已修夹具，独立的 `cd slo_tools && python3 -m unittest discover
# -s tests` 当前也全绿。本命令保留历史 ignore 集以维持各批 canonical
# 计数可比，SLO 目录测试另行独立验收。
# W0–W11 世代（《joint仓库当下+历史修改方案》2026-09-29）：W0 施工前
# 1703 → W5/G-B 1973 → W8 2033 → W10 交付态 **2098 passed + 1 skipped
# + 33 subtests 全绿**；W11 收口终跑复验同数零回退（证据
# /home/sunhao/joint_evidence_v30/w11/final_pytest.txt；逐包计数沿革与
# 判定见该树 README 与 w11/FINAL_ACCEPTANCE.md）。
# 交付基线：366 passed + 1 skipped + 11 subtests（部分层逐出 KV 改造批
# ＋kimi 收尾批 2026-09-17 起 = R17 批 321 基线 + credit 交错流批/部分层
# 逐出批/收尾批新增用例；目标窄命令 workload/llama2_7b_inference 实收
# 279 passed + 9 subtests）
# 2026-09-22 三机制改造收口（C1–C19）终态：全量 724 passed + 1 skipped
# + 29 subtests 全绿（366 基线 → +358：窄命令增量 +335 = C1+22/C3+5/
# C3b+11/C5+11/C9+66/C10+20/C13+18/C14+14/C2+24/C17+18/C15+42/C4b-FIX+5/
# C7+34/C8+19/C11+26；+ C16 slo_tools 19 + C18 孪生 round-trip 4
# （tests/test_hardware_twins_roundtrip.py，+18 subtests）；窄命令
# 实收 614 passed + 9 subtests）。slo_tools 目录内 unittest discover
# = 138 ran（driver_parity 2 失败 + slo_contract 1 error 为 HEAD 先在，
# 与本仓改动无关——C16/C19 两次复跑与基线逐签名相同）。
# 2026-09-22 复审修复批（F1–F6，补遗 A7'–A10' + 独立复审缺陷清零）终态：
# 全量 761 passed + 1 skipped + 29 subtests 全绿（724 → 净增 +37：F1+22/
# F2+5/F3+9/F6+1；skip 仍为 phase2 journal fixture 缺失，C0 既有；
# 窄命令实收 646 = 614+32：F1+22/F2+5/F3+4/F6+1——该数当批漏登，
# 由下行 G 批注释补锚）；slo_tools 目录内 unittest discover = 143
# ran，预存 3 红（driver_parity ×2 + slo_contract ×1，HEAD 先在）
# 签名与 C16/C19 基线相同；13 臂冒烟矩阵重跑 13/13 exit 0（smoke_
# input queue md5 与 C19 证据根逐位一致，证据根 /home/sunhao/
# joint_smoke_evidence_fix，PROVENANCE §35）。
# 2026-09-22 第二轮深挖复审修复批（G1–G4，补遗 A11'/A12'）终态：
# 全量 775 passed + 1 skipped + 29 subtests 全绿（761 → 净增 +14：
# 窄命令 +11 = fix1+5（钳位/零速率同窗失效/A10b 真实预约处置三例
# ——另有早退改钉 1 例为净 0 改钉，H 批口径澄清）/predictor+4（守
# 恒注入×2/大基座/激活切割）/face+1
# （dump 哨兵）/quota+1（压力态安全网）；slo_tools +3 = tier 哨兵/
# 非 dict 层/乒乓零假阳性）；窄命令 657 passed + 9 subtests；
# slo_tools unittest discover -s tests = 146 ran，预存 3 红签名同
# 基线；13 臂矩阵重跑 13/13 exit 0（queue md5 62a14f16 与锚一致；
# 剥离 host 计时与本批新增披露键 service_factors.clamped 后，
# 13/13 臂决策日志/统计/请求账/列车账与 F 批证据根逐位一致——激活
# 切割与零速率同窗失效 2s 窗内零决策漂移；bridge sidecar 差异 =
# stay 行 home_after 语义修正逐行核对零其他差异。证据根
# /home/sunhao/joint_smoke_evidence_fix2，PROVENANCE §36）。
# 2026-09-22 第三轮深挖复审修复批（H1–H10，补遗 A13'/A14'）终态：
# 全量 784 passed + 1 skipped + 29 subtests 全绿（775 → 净增 +9：
# 窄命令 664 = 657+7：fix1+2（病态样本 fail-closed/LOCAL 基 ceil
# 对齐）/predictor+4（η·γ·重复 id·start 加固/大基座激活切割回归）/
# quota+1（夹具哨兵门禁接线钉）；另有净 0 三处——守恒通道①②隔离
# 重钉（A13'：原②零钉住——step() 已结算移册 post=0 使①先炸）、
# A10b failure_class 改钉（H6 对齐后新契约 quota_deferred_quota_
# link）、face dump 哨兵测试 importlib/sys.path finally 回收；slo_
# tools +2 = journal 值非列表层/全 None 会话乒乓不可见）；slo_tools
# discover -s tests = 148 ran，预存 3 红签名同基线。矩阵处置：本批
# 全部修复对合法输入零行为变更（注释订正/不可达分支一致性/非法输入
# fail-closed 加固/消费端值级层判/夹具脚本/测试文档），13 臂冒烟
# 不重跑（A14' 落字理由：决策与统计输出面不变）。修复面：stress
# 夹具逐键哨兵门禁收紧（G3 遗留假 GREEN 封死）/predictor 直连输入
# 加固四件（含幸存流负尘埃零化——大基座时刻回退与 drained-below-
# zero 误炸双封）/sh30 病态样本封死（served>0∧active==0）+ 防御分
# 支 wait_reason/重试门口径对齐/domain_metrics 值级层判贯彻/JCM
# 驻留折算防御性对齐；PROVENANCE §37。
# 编号消歧（2026-09-22 登记）：本文与执行计划中的 F1–F15（如 F7
# 耦合、F10 四动作、F11 预测终点）= 执行计划 §4 冻结项编号空间；
# "修复卡 F1–F6"（PROVENANCE §29–§34）、"G1–G4"（§36）与
# "H1–H10"（§37）与"K1–K8"（§38）与"L1–L8"（§39）与"M1–M8"（§40）
# = 复审/外部审计修复批卡号——四编号空间独立同形，按语境判别。
# 2026-09-23 外部六路审计修复批（K1–K8，补遗 A15'，PROVENANCE §38）
# 终态：全量 815 passed + 1 skipped + 29 subtests 全绿（784 → 净增
# （online/test_joint_k_batch.py 新 31 用例））；窄
# 命令 695（K 批原登 693 系笔误——L 批排除法复核 + kimi 复核亲测同
# 证 664 基线+31 精确吻合，§39.6 勘误）；slo_tools discover = 148 ran
# 预存 3 红同基线。**矩阵处置
# 与 H 批相反：本批对合法输入有行为变更**（K5 计价判据镜像物化/K6
# copy 层段门控移除时序乐观/K7-② AIMD 缺测不进 streak/K7-④ aimd
# 遥测真信号/K1 r̂ 成员平均/K2 终轮 merge 闭合），13 臂冒烟矩阵重跑
# （证据根 joint_smoke_evidence_k）。**run 级归因勘误（L 批 §39.4）**：
# 矩阵全 13 臂 remote-read 选中数恒 0、merge_done 零发生——quota 臂
# 未进入 merge 通道，P1-② 修复的 run 级闭合未达成（由终轮交叉单测 +
# 接线文本钉承载）；quota-on 强制压力由 /home/sunhao/joint_stress_
# quota_k（10s × stress-28gib ×quota-static：partial_copy_hits=49、
# 守恒零 raise）承载，"quota-on + remote-read 选中 + 终轮"组合现旋钮
# 结构性造不出（C20 前需 action-pin 新旋钮，未闭合项登记）。
# 修复面：P1-① r̂_KV active_decode
# 成员类型/P1-② 终轮 merge watch 注册（胜者侧预留滞留→verify_run_end
# abort 封死）/P1-③ copy 首（层消费）体块逐层段门控（C13(4) 层粒度
# 兑现，M≥3 必现乐观）/P1-④ merge 计价方向 = 字节判据镜像物化（min
# 双源废除）/P1-⑤ 端口 bulk 需求聚合/P1-⑥ 零字节腿 rank 串行链；
# P2：反向腿自身路径计价/空 reclaimable 保守化/深缺口 covered 拆叠
# 双计/copy@home×REMOTE = 物化侧 N9 豁免 pool-only 基（裁定③就地
# 转正兑现）/quota_deferred 分位枚举（C5 规格变更扩一值）/AIMD
# streak 连续性/set_link_quota 同值零 bump/ASTRA_LINK_OBSERVER 数据
# 源注入（aimd 臂真闭环，C20-③ 前置）/离线平局序镜像在线/quota_
# admissible 列派生；P3：predictor NaN 拒收/EvictionPlan 等长零向
# 量/cold_start 接受后翻/弱断言修拓五处/注释订正三件/§38 勘误
# （§20.2 getattr 旧文、journal 守恒自检名目、P2-6 逐块释放承诺
# 降级为 drain 边界实现语义）。
# 2026-09-23 K 批复核审计修复批（L1–L8，补遗 A16'，PROVENANCE §39）
# 终态：全量 834 passed + 1 skipped + 29 subtests（815 → 净增 19 =
# online/test_joint_l_batch.py 18 + quota_admissible CSV 0 分支 1）；
# 窄命令 713（695+18）；slo_tools discover = 149 ran 预存 3 红同基
# 线。**零合法输入行为变更 ⇒ 矩阵不重跑**（不可达路径防御/入口
# isfinite/账本不变式 raise/脚本断言/纯文档）。修复面：L1 forced
# 计数键扩 quota_deferred（§38.3"计数键同源"登记不实勘误 + 4 处测试
# 镜像同步）/L2 predictor NaN 同族三处（now_ns 死循环/release_eta_ns
# 乐观/first_block_wait_ns ValueError 逃降级通道）+ 跨腿 rank 等长/
# L5 copy 层段账本内部缺口 fail-closed（块间 + copy↔restore 间；
# covered==0 热前缀合法）/L6 sensing 旁路与 metrics-off 双
# fail-closed（aimd 组合矛盾即拒启动）/L7 文档订正四处（forced 三成
# 因枚举×2、开关清单 observer 例外 + metrics 门）/L8 卫生（bak×5
# 删、generated 复核零）+ 补钉两枚（前向抬升对偶/quota_admissible
# "0"）；落字披露：L3 run 级归因勘误 + stress 产物补引 + 终轮闭合
# 未闭合项、L4 r̂ 双口径窗、_telemetry_seq 未来形态约束（per-link seq
# 前置）。
# 2026-09-23 chatgpt 验收级审查修复批（M1–M8，补遗 A17'，PROVENANCE §40）
# 终态：全量 850 passed + 1 skipped + 29 subtests（834 → 净增 16 =
# online/test_joint_m_batch.py 14 + domain_metrics 2）；窄命令 727
# （713+14）；slo_tools discover = 151 ran 预存 3 红同基线。**有合法
# 输入行为变更 ⇒ C++ 全量重建 + 13 臂矩阵重跑**（证据根
# joint_smoke_evidence_m，全 exit=0；aimd 遥测覆盖与 K 批逐位一致零
# 回归）。修复面：M1 遥测物理口径统一（C++ 观测器增时间加权活跃流数
# flow_active_ns 积分 + 桥 sample.active_flows；JCM 遥测除数流数口径
# ——修前 capacity/合计速率口径满载退 1/下游瓶颈误判争用双症状；
# AIMD 分母 max(在册,流数)——修前 collective 份额归因 KV 流产生
# comfort 反向信号）/M2 配额前经济域重建（配额拒候选保留 cost_ns +
# D_econ 纳入 quota_ 候选 + quota_admissible_remote 单列 + 摘要
# status 派生）/M3 σ̂ 同终点配对（merge_done 行索引 + service 延迟
# 单列，异终点 −M 偏差不进误差）/M4 γ_prefill 生产桥接（纯 prefill
# 列车样本喂 face ServiceFactorGroup——修前零调用方恒冷启动 1.0；
# η_pool 无纯源落字）/M5 零字节 shard 免逐跳时延（startup 保留——
# F1 冻结契约锚优先于审计建议，中途改道记录在案）/M6 C18 真冒烟
# 补登记（stress 臂即 SH_RUNTIME_RC_DIR 路径真运行）/M7 文档订正
# （README service_done 先后矛盾 + 交接即释放超实现 + §6 口径注 +
# 尾随空格）/M8 维持项确认（④C13 drain 边界降级/⑧C4b C20 授权面/
# ⑨§39.4 已勘误）。M1 流数口径 run 级分叉证据受 2s 轻载窗限制（流数
# 恒等于注册数两口径等值），分叉形态由对拍单测承载（§40.6 如实登记）。
# 2026-09-23 chatgpt 对 M 批交付的复核修复批（N1–N13，补遗 A18'，
# PROVENANCE §41）终态：全量 868 passed + 1 skipped + 29 subtests
# （850 → 净增 18）；窄命令 743（727+16 = test_joint_n_batch.py）；
# slo_tools discover 153 ran 预存 3 红同基线。**有合法输入行为变更
# ⇒ C++ 全量重建 + 13 臂矩阵重跑**（证据根 joint_smoke_evidence_n 全
# exit=0；aimd 遥测覆盖与 M 批逐位一致零回归）。修复面：N1 配额同
# 事务槽位复用（读流与 merge 预留 per-link max + tracker 借槽/转移/
# 消解——修前空链 Q=2 需 3 槽结构性排除远读、违反"ρ<1 域空化由定
# 价涌现"；run 级实证 quota 臂 quota_link 拒 128→96、applicable
# 0→32，chosen_remote=0 = 配额不关定价不选的正确形态）/N2 流数除
# 数叠加候选（max(注册旧流,遥测旧流)+候选份额——修前两未登记流+
# 候选实 3 算 2）/N3 仅流数遥测保留（零速率窗不丢流数 + A9' 剪除并
# 集 + JCM 披露 flow_count_links/legacy_rate_divisor_links）/N4
# prefill 基数生产同形（JCM prefill_task_load_ns_fn 注入 + SH 同形
# + face E 递推；chatgpt 复算例数字未复现如实登记，形状差异实证
# 0.13–0.28）/N5 merge 尾门纯度排除（R11(ii) 门等待不入 γ 样本）/
# N6 双域导出（remote_cost_pre_quota_ns 列 + replay 实际可行集重放
# + quota_counterfactual_flips + D_feed 订正）/N7 C4b 归属勘误
# （G2 未闭合、不借 C20 名目移转——重做须专门一轮）/N8 C18 孪生档
# 真冒烟（fixture SH_STRESS_JSON 覆盖 + x05/2025 GB/s 档 GREEN，
# /home/sunhao/joint_c18_twin_x05_evidence——修前认领错硬件 4050
# stress 档）/N9 merge_done 缺行 NA+告警/N10 M5 测试 1B 钉/N11
# C++ 观察器流数断言（上游继承测试首次构建运行暴露 record 级等价
# 环境差异——金值直钉+总量守恒处置，与 M1 无关实证在案）/N12 M4
# 生产路径测试/N13 文档残留订正。
# 2026-09-23 kimi 终轮深挖处置批（O1–O14，补遗 A19'，PROVENANCE §42）
# 终态：canonical 936 passed + 1 skipped + 29 subtests（868 → 净增 68）；
# 窄命令 807 全绿（763+1 红 → +43 新（online/test_joint_o_batch.py）+1
# 红修复）；slo_tools discover 160 ran OK（O13 清零 HEAD 先在 3 红——
# discover 首次全绿）。**有合法输入行为变更 ⇒ C++ 全量重建 + 13 臂矩阵
# 重跑**（证据根 joint_smoke_evidence_o 全 exit=0 + 10s stress PASS；
# 二进制 sha256 与 N 批逐位一致 = C++ 零改动；run 级行为变更 = recompute
# 选中全域清零转 stay——O1 @驻留基计价基订正的预期形态；chosen_remote
# 仍全 0，applicable 128 在案维持 §41.6 登记）。修复面：O1 recompute
# history 驻留二分（_resident_here）/O2 配额入册序不变量（判据时刻逐出
# 足迹不可知——主流程恒先于支链、支链失败按 quota_oneshot_overflow
# 披露降级不 abort，:4505"正常不可达"注释写实）/O3 merge_tail_gated
# 扩域本实例粒度（任意 session watch ∪ frontier pending_store_tails，
# decode 分支同款排除）/O4 victim 惰性求值/O5 kv_delta_find O(1) 索引/
# O6 AIMD 断供推进 seq + r̂ 回退窗冻结扩张（comfort_frozen 不计进度）/
# O7 遥测 ingest 四件（混合缺席 raise/双零同步 pop/降级三键入覆盖裁决/
# 口径 docstring）/O8 域度量五件（跨族全键序平局/hops 过滤/σ̂ NA/note
# 订正/分位单排序）/O9 jsonl parse_constant 拒 NaN/O10 守恒四件（run 尾
# tracker 零账断言/borrow O(1) 校验/三注册表 leaked_owners/credit_arms
# fail-closed）/O11 runner aimd∧off⇒exit1/O12 分类器 N1(a) 同步 +
# oneshot_overflow 入 run 指标/O13 3 红夹具补 exits/O14 N9 空串显式化
# 等择要。改钉三处推导在案（§42.4：δ=10 跨族平局/PARTAL 基 remote-read
# 反转/victim 夹具补字段）；A19'(b) 实施订正（O2 实走不可知分支）。
# 未决后续项登记：run_golden_live.py drains 键/read_cpp_metric_records
# 裸 loads 两件下一批（§42.6）。
# 2026-09-24 全仓深挖修复批（105 项：严重错误修复＋死代码删除）注销上项：
# 两件均已闭合——tests/run_golden_live.py 已删除（死依赖清除：硬依赖仓外
# /tmp/slo_wps/set_trace_pointer.py 不可运行；golden 语义由 test_golden_
# g1g4.py 离线承担，slo_tools/README.md 已同步）；read_cpp_metric_records
# 已接 parse_constant/parse_float 双拒 NaN/Infinity（slo_common.py，O9
# 同口径）。同批清除面（OfflineGreedy 整链/RemoteFifoLedger 整链/3 个
# 孤儿测试/4 组死配置键/死函数死字段等）与本 README 其余章节无交集，
# 中央验证全绿（构建＋20 C++ 测试＋192 python 测试＋TJE 2s 冒烟）。
# 2026-09-24 逐出尾 watch 修复验证轮（watchfix，PROVENANCE §43）终态：
# canonical 972 passed + 1 skipped + 33 subtests（957 暂停基线 → 净增 15 =
# 新文件 online/test_eviction_tail_watch_real_paths.py 逐出尾 watch 四路径
# 真实生产链钉测 15 用例，零后端）；窄命令 834 passed + 13 subtests；
# slo_tools unittest discover -s tests = 171 ran OK（skipped=1，与暂停
# 基线持平——watch 用例不入 slo_tools 目录）。生产改动仅为潜伏缺陷加固
# （decode_eviction_watch_id slot：现行 run 结构性不触发，§43.2）；本轮
# 三证据根：C4b 八臂 /home/sunhao/joint_c4b_evidence（全 PASS；MAE=
# 661602934.5ns、MAPE=39.56%（零值实测排除数 0）、排序错误 0、选择损失
# 0ns/0%；"强制 ≠ 自主选中"三层披露在案）、sensing 入口
# /home/sunhao/joint_sensing_entry_evidence（SH_RUNTIME_RC_DIR 消费完整
# run exit 0）、13 臂矩阵 P 批 /home/sunhao/joint_smoke_evidence_p（13/13
# exit=0；准入级 chosen 分布与 O 批 §42.5 逐数一致 = 该粒度无漂移）。
# 2026-09-25 PARTIAL 跨实例 copy 流水化批（PROVENANCE §45，零新增开
# 关）测试增量：T2 test_joint_copy_handoff.py 18→23、T3 test_joint_
# layer_restore.py 17→20、T8 test_store_restore_ordering.py 9→13（三
# 文件本批复跑 56 passed）；窄命令 workload/llama2_7b_inference 实收
# 846 passed + 13 subtests（watchfix 基线 834 → 净增 12，本批复跑）。
# C++ 增量与复跑见下方"跨实例 copy 流水化 C++ 夹具"块。canonical 全量
# pytest 与 10s 压力档不在开发自测面内，由交付终门禁统一执行（门禁结
# 果：回归门禁全绿、压力夹具通过）。
# 2026-09-25 prefill remote-read 分阶段批（规格书批次 + 评审修复
# PROVENANCE §46/§47，零新增开关）测试增量：新增五文件 66 用例——
# online/test_prefill_remote_read_{manager,graph,lifecycle,scheduler}.py
# 16/25/9/7 + joint/test_joint_remote_read_staged_path.py 9；另 test_
# joint_preadmit_visibility.py 净 +1（3 个 readplan 命名用例改写为
# prefill_read 变体 + 1 个新增，diff 在案）。
# 文档同步轮复跑实收：窄命令 workload/llama2_7b_inference 916 passed +
# 13 subtests（§45 批 846+13 → 净增 70 = 新五文件 66 + preadmit +1 +
# 其余 3 未逐文件归因）、online 441 passed + 11 subtests、joint 419
# passed（两套自查命令全绿）、canonical 全量 1054 passed + 1 skipped +
# 33 subtests（本轮文档同步员亲跑，门禁外补充证据）。
# 口径订正（外部审查处置批 2026-09-15；三轮深审再订正分解式）：此前
# 此栏把全树计数配给了 workload/llama2_7b_inference 四目标窄命令
# （该命令实收 229 = 本仓目标面 joint/+根两 test 文件+online/）；
# 全树实收 = 目标面 + slo_tools 可收集 + sh_test_mesh/
# tests → R17 批 320 passed + 1 skipped（skip = slo_tools test_hbm_
# watermark.py phase2 journal fixture 缺失）。slo_tools 三文件
# （driver_parity/golden_g1g4/slo_contract）从 sh_test_mesh 根收集
# 即 ImportError（伴生模块 synthetic.py 在其目录内、不在 sys.path；
# 目录内可收集 80 项）——基底固有的调用目录依赖，与本仓改动无关，
# --ignore 排除（目录内直跑时 driver_parity 2 失败 + slo_contract 1
# 失败为 HEAD 先在，git stash 还原基线复跑坐实，与 R17 无关）。
# 2026-09-26 D14 长度估计器与在线因子批（v2.0 阶段 1 等价瘦身 + 冻结
# 值变更，零新增开关）测试增量：新增 joint/test_d14_horizon_service_
# factors.py 13 用例（sum/count 等价逐次一致 + 百万轮常数空间 + 冷启
# 动 32 前向步生产钉 + 式(4a) 边界 max(1,n̂) + 固定 1/16 EWMA 闭式轨
# 迹；transfer_factor 原仅披露的身份在 2026-10-03 改为正式 NoC 链路效率）；窄命令
# workload/llama2_7b_inference 实收 982 passed + 31 subtests（969+31
# 基线 → 净增 13）。行为变更登记：冷启动先验 1 token → 32 decode 前
# 向步（式(4)）；ServiceFactors 时间衰减 α EWMA → 固定 1/16（首样本
# 同规则自 1.0 出发）；亚名义/亚 roofline 样本由钳位 1.0（clamped 计
# 数）改为记录异常跳过（anomalous 计数）；决策日志 service_factors 披
# 露键 clamped → anomalous。所有新实验从同一定义的冷启动初态重开。
# 2026-09-27 验收纪律批（ACC，joint修改文档「验收纪律（v2.0 §19 三类，
# 全部纳入）」代码侧第 1/3 类，零新增开关）测试增量：新增两文件 21 用
# 例——joint/test_acc_acceptance_formula_conservation.py 11（通用清单补
# TP 不均匀/KV 头复制 padding/实际增长超预测 6 ＋ P→D 专项补第 6/10/12
# 条 5）＋ joint/test_acc_predictor_cost_accounting.py 10（分位数学/
# fail-closed/快照完备/sh30 接线冒烟）；另 7 个替身夹具对齐 __init__
# 新账本初值（F6 销账口径，净 0 用例）。终态：全量 tests/ workload/
# llama2_7b_inference = 1332 passed + 31 subtests 全绿（ACC 前 1311+31
# → 净增 21）；窄命令 workload/llama2_7b_inference 实收 1279 passed +
# 13 subtests。行为面登记：sh30/base/online_service 新增恒开预测器开
# 销账本接线（决策墙时样本/摘要维护费/后端事件量/online_stats 汇总行
# predictor_cost 键）——决策序列与既有披露键零漂移（账本只读新增，
# 不进任何计价与选择路径）。
# 2026-09-26/27 v2.0 联合代价模型施工批（joint修改文档 v2.0：第二批执行
# 协议统一 D09/D10/D18/D19/D20＋第三批新估价内核 D01–D08/D11–D13＋第四批
# 受控验证 D15/D16/D17＋消融前置 M2/M3/M4；**零新增功能开关**——唯一开关
# 级变更 = M2 新增 `JOINT_ALLOWED_ACTIONS`（动作掩码消融，§1 env 块）与
# D09 以 `JOINT_REMOTE_READ_TILE_BYTES` 取代退役的
# `JOINT_REMOTE_CREDIT_ITERS`（env＋`--remote-credit-iters` CLI 同批退役，
# 旧键出现即启动失败）；其余全部为协议级/计价级行为变更，逐卡登记见
# §3 状态表对应行与《experiment/仿真各功能开关清单.md》§7.0 批注记）。
# 测试增量（各卡自报，累计口径）：D14 +13（独立记录见上）、M3 ±0、
# D15 +4、M4 ±0（替身夹具补三字段净 0）、M2 +13、D09 +18、D10 ±0（验收
# 四组落既有文件）、D18 +16、D19 +27、D20 +19、D16 +12、D01 +12、D02 +20、
# D03 +12、D04 +19、D05 +3、D06 +21、D07 +14、D08 +17（＋decode 上下文
# 12 用例改宗重写）、D11 +25、D12 +21、D13 +14、D17 +20（8 文件旧截断
# 锚按取整口径重钉，非删测）、ACC +21（独立记录见上）。
# 文档同步轮复跑实收（2026-09-27，本批收尾终态）：canonical 全量
# tests/ slo_tools/tests/ workload/llama2_7b_inference（三 ignore 沿用）
# = 1418 passed + 1 skipped + 33 subtests 全绿；题给门禁命令
# tests/ workload/llama2_7b_inference = 1333 passed + 31 subtests 全绿
# （ACC 记录 1332 → +1 = test_d20_stage_transaction.py 尾改新增 1 用例，
# 06:25 落盘晚于 ACC 终计数，本同步轮实测确认）；窄命令
# workload/llama2_7b_inference 实收 1280 passed + 13 subtests。
# 本批如实登记的未尽/受阻项（2026-09-27 修正轮后逐项复核，①②④⑤已
# 翻转、③维持，均非缺陷）：
# ① 边界 recompute 静默重建/边界基材料化（copy-in）/完整型多源 remote-read
#   的执行图钩子**已接线（修正轮 C4）**——边界可选面自协议裁定两族子集
#   翻转为区间缺失物理语义的完整动作面（D18/D19/D20 行同面更新；三掩码
#   臂冒烟真实执行验证全 exit=0）；三位置 merge_back 运行时实体同步落地
#   （C4/C5，`_merge_back_three_position` 式(21)）；
# ② §6.5 纯两源 remote 形态**已可选（修正轮 C4）**——完整型多源
#   remote-read 与增量型 copy@第三目标进运行时候选面（旧"运行时不可选"
#   表述废除，协议裁定串删除）；
# ③ v2.0 §22-D01 一句「公共工作量与静态路由同决策共享」未实现（等价
#   瘦身性质、数学结果不变；候选内逐候选重算公共量为纯开销问题，留后续）；
# ④ prefill 前缀读流**已按消费窗×tile 拆分（修正轮 C3）**——逐消费窗
#   （chunk）×tile 消费驱动发射＋双缓冲释放链＋C2 lease 暂存持有，JCM
#   P 腿 tile 口径按执行窗基数对齐（旧"每次 scan 独立事务不拆分"披露
#   废除；钉测 online/test_prefill_remote_read_tile_protocol.py）；
# ⑤ 同状态 probe 工具与掩码臂冒烟**已具备（修正轮 C7）**——
#   `run_scripts/joint_same_state_probe.py`（generate/summarize/selftest，
#   强制钩子 JOINT_FORCE_BOUNDARY_REQUEST/ACTION）＋冒烟矩阵三掩码臂；
#   正式规模同状态 campaign 仍属实验任务（候选 `measured` 真值槽位在
#   正式 run 中恒 NA）；宿主预测开销 10% 首轮预算为待验证目标，账本只
#   供数不下结论。
# 全程未运行任何仿真/构建（pytest 门禁除外）；13 臂冒烟矩阵与 10s 压力
# 档由交付终门禁统一执行（本批含合法输入行为变更，与 D14/D17 批同面）。
# 2026-09-27 审查修正轮（C1–C7a workload 主链七环＋W1–W5 slo_tools 水印
# ＋C7b run_scripts＋两轮 G3 修缺；基线 HEAD=0ef6b19，详见
# joint仓v2.0施工进度.md 修正轮段）终态实测：canonical 全量
# tests/ slo_tools/tests/ workload/llama2_7b_inference（三 ignore 沿用）
# = 1552 passed + 1 skipped + 33 subtests 全绿（修正轮前基线 1418+1+33，
# ~/joint_fix_round_20260927/gates/g0_canonical_baseline.log /
# g1_final_canonical.log）；C++ 真实构建＋5 gate 测试（BridgeLoopback/
# GraphBatchCommitter/ParsedBatch/WindowedReader/WatchRegistry）＋
# tests/run_all.sh 全过（gates/g2_build_ctest.log）；16 臂×2s 冒烟
# （13 既有＋3 掩码臂）全 exit=0（gates/g3c_smoke_final.log，证据根
# ~/joint_fix_round_20260927/smoke_final/）；16 臂水印后处理统一收口
# ALL-STEPS-PASSED（gates/g4_final_watermark16.log）＋ b4 五臂水印复算
# 绿（gates/g4_watermark_b4.log）；同状态 probe 6/6 PASS（gates/
# g5_probe_runs3.log，联表 probe/probe_20260927T080742Z/probe_summary.md：
# request_0 损失 0ns、request_1 损失 195812108ns——remote-read 预测
# 低估样本）。
# 2026-09-27 修复轮二（独立审查 12 项"部分完成"判定的修复；详见
# joint仓v2.0施工进度.md 修复轮二段与 §3 状态表"修复轮二"行）：D09
# 固定 tile（放大废除）/D16 有界模板与 prefill_remote_read_summary 键/
# D10 静默重建执行图/D03 P 准备腿 frontier 对齐/D14 前向步语义审计落定
# （无需换算）/D17 遥测基准 ceil/D19 边界视图真实填充/D20 预约回滚/
# D 边界容量审计（capacity_summary_view＋capacity_ledger_by_tp_rank）/
# D12 平价钉测与 fallback_sequence 披露/D05 零时 commit 锚/D04 零距离
# 池路径端点登记＋测量钩子 JOINT_FORCE_ENTRY_ACTION 与 probe 三族臂。
# 新增/重钉钉测文件（collect-only 69 用例）：joint/test_d03_frontier_
# alignment.py 6、test_d05_merge_commit_disclosure.py 4、
# test_d12_merge_projection_parity.py 4、test_d14_forward_step_
# semantics.py 7、test_d17_rounding.py 5、test_d19_boundary_layout_
# population.py 9、online/test_c7a_force_entry_probe_hook.py 15、
# test_d10_silent_rebuild_execution.py 9、test_d20_reservation_
# rollback.py 6、test_prefill_read_summary_compat.py 4；另有
# test_d09_remote_read_double_buffer.py / test_d18_boundary_decision.py /
# test_d16_runtime_linkage_bounded_backend.py / test_prefill_remote_read_
# tile_protocol.py 等既有文件按新语义改锚。**版本串升版
# v3.1-2phase-20260927（前代 v3-2phase-20260927）——旧实验记录持旧
# 版本串时不得重命名为新协议结果**。集中门禁（canonical/构建/冒烟/
# 水印/probe）由并行车道执行，证据落 ~/joint_fix_round2_20260927/gates/
# ——本文档收官时不预填门禁结论，以门禁日志为准。
# 2026-09-27/28 修复轮三（审查修正轮收敛循环——对照两文档查缺＋深挖
# 严重错误/死代码/冗余注释，查→核实→改循环到收敛；9 项明细见
# joint仓v2.0施工进度.md 修复轮三段）：确认并修复 9 项严重错误——D04
# 零距离池事务执行侧 HBM 端点登记补全（无条件登记）/probe 臂 env 与调度
# 器互斥合同对齐（boundary 臂不设 ENTRY，P 面同状态由基线 P 选择承载为
# entry_pin）/多源 decode credit tiles_per_pass 改逐源和（TJE_quota_aimd
# 冒烟红根因，规划期 1:1 守卫）/P 入口双缓冲腿补 D06 阶段份额/JCM 多源
# 块数·逐 pass 重切·PARTIAL 基率三处与执行切片对齐/R5 三位置 merge
# base 双计（boundary_source_complete_union）/R6 折叠清账 base 字节滞留
# /R8 BOUNDARY 钩子仅 D 边界接线（P 准入不受影响，P 面强制一律走
# JOINT_FORCE_ENTRY_ACTION）/R9 copy handoff journal 来源转记清账；另
# 清理死代码（KVAllocator 族/base_shard_bytes/FaceInstance 几何字段/
# verify 死 fixture 7 件/约 50 未用 import 等）与失实行号族注释约 20 处；
# 工具加固（probe --strict-version 同代版本校验、水印跳过词表补 3 个
# boundary kind）。新增钉测约 25 用例（R5/R6/R9 由 probe 强制臂踩红、
# 先红后绿）；版本串不变 v3.1-2phase-20260927。集中门禁（probe 重跑/
# 冒烟矩阵/水位线）执行中，证据落 ~/joint_fix_round3_20260927/gates/
# ——本段不预填门禁结论。
# 2026-09-28 D06 式(6) D 边界 credit 腿同分母收口（审计裁定 NEED_FIX）：
# D 边界 remote 候选的 decode credit 暂存腿（双缓冲闭式供给基元/tile x
# ——单源 `_remote_double_buffer_leg_ns` 不传 hbm_port_registry 走裸注
# 册表 m_e+1、多源 tile x 同；闭式 start_ns=move_done_ns 即与 decode 计
# 算同端口并发）改走 decode 阶段冻结份额 k_e=2（与同阶段 decode 计算
# 读同 m_e+2 分母；离线无注册表口径同面 k=2）——修复前暂存腿偏快 ⇒
# remote D 边界候选偏乐观；P→D 迁移腿（copy 三腿）仍独立冻结阶段 k=1
# 不动。`_remote_double_buffer_leg_ns` docstring 同步订正（删除对
# StageScopeAnchorTest 的错误归据——该锚钉的是迁移腿非 credit 腿）。
# **版本串升版 v3.2-2phase-20260928（前代 v3.1-2phase-20260927）——本
# 修订改变 D 边界 remote 候选腿时延，旧实验记录持旧版本串时不得重命名
# 为新协议结果、不得自动充当新模型证据**。新增钉测
# test_d06_hbm_stage_share.py DboundaryCreditLegStageShareTest 5 用例；
# 全量回归 1488 passed + 13 subtests（基线 1483 + 新增 5）。
```

定向验收覆盖（`joint/test_joint_mechanisms.py` 对应《设计方案》§8 表）：
八组合固定映射与 fail-closed；T 严格序/满足即停/未知类别披露；E k_hide
解析例（L=32, c=1ms, q=0, r=0.5/2/4ms ⇒ 1/17/25）与三模式计划；J 三模式
选择与三动作候选面（remote-off 消融用例与退役 env fail-closed 钉测随
2026-10-14 远读退役改写）；home/merge（增量恰好一次、无双份
驻留、recompute/copy 工作副本口径）；八组合同实现配置切换。
2026-09-22 改造批新增机制测试文件（全部零后端）：
`joint/test_joint_shard_pricing.py`（C1 三腿 min/平价锚/读放大）、
`joint/test_hbm_port_flow_registry.py`（C2 u_port/rider）、
`online/test_joint_decode_context.py`（C3 上下文依赖；2026-09-27 D08
改宗重写为增长服务折叠语义，见 §3 状态表 D08 行）＋
`joint/test_d08_context_growth_fold.py`（D08 增长服务折叠 17 用例）、
`online/test_joint_arrival_contract.py`（C3b 到达合同 11 用例）、
`online/test_joint_decision_schema.py`（C5 冻结 schema 11 用例）、
`joint/test_joint_telemetry_divisor.py`（C7 遥测三件现有 39 用例）、
`online/test_joint_preadmit_visibility.py`（C8 #readplan 现有 20 用例；
2026-09-22 收口时为 19，§6 历史总数仍按当日实收登记）、
`joint/test_link_quota.py`（C9 现有 77 用例）、
`joint/test_link_quota_stability.py`（C10 稳定性 5 剧本 20 用例）、
`online/test_joint_quota_integration.py`（C11 SH 集成 29 用例）、
`online/test_joint_copy_handoff.py`（C13 块级交接 23 用例）、
`online/test_joint_remote_settlement.py`（C14 结算闭合 14 用例）、
`joint/test_event_recursion_predictor.py` + `online/test_joint_layer_
restore.py`（C15 25+20 用例）、`slo_tools/tests/test_domain_metrics.py`
（C16 现有 43 用例）、`joint/test_face_static_mode.py`（C17 18 用例）、
`online/test_remote_credit_multiblock.py`（C4b-FIX 5 用例）、
`tests/test_hardware_twins_roundtrip.py`（C18 三孪生×2 容量档
round-trip 4 用例 + 18 subtests）、
`online/test_eviction_tail_watch_real_paths.py`（2026-09-24 watchfix 批
逐出尾 watch 四路径真实生产链钉测 15 用例，PROVENANCE §43）；
`online/test_forecast_future_isolation.py`（2026-09-30 修复批 §8.3
预测器与隐藏真值隔离端到端钉测 5 用例——改后继请求 manifest 隐藏
interval/长度、保持已揭示类型与交付时刻线不变 ⇒ 预测侧逐字段不变）。

端到端冒烟（astra_compute_20.csv 前 2 秒，21 请求）：执行方案 §8.3 的
**13 配置矩阵全部 PASS**（八组合 #1–#8 + 辅助臂 #9 typed/affinity-first/
legacy_half、#10 typed/affinity-first/adaptive、#11 typed/affinity-first/
minimal_layer_groups、#12/#13 remote-off 正交臂）；行为矩阵：joint 组
选点 10/4/6/1 vs load-first 组 12/6/2/1（J 联合选择真实改变选点），
load-first/none 系真实触发 7 次跨实例 copy + 7 次增量 merge 回传流
（R15/K4 计价修复后决策序列快照，修复前为 6/6）；
affinity-first 恒 stay（home 优先与联合选择的最优重合）。**两处口径
注记（2026-09-22 复审补，2026-09-27 修正轮更新，2026-10-14 远读退役
再更新）**：上文执行方案 §8.3 的
13 配置矩阵（含 #9–#11 调度器变体辅助臂）与下文 `joint_smoke_matrix.sh`
的冒烟矩阵是**两个不同集合**（历史交集 = 八组合 + remote-off 臂族；
remote-off 臂族已随 2026-10-14 远读退役删除）——前者
服务调度器行为对照、后者为交付一键回归矩阵，引用时勿混用。**冒烟矩阵已
脚本化**：`sh_test_mesh/run_scripts/joint_smoke_matrix.sh`（2s 窗
**14 臂**；C19 起 13 臂一键全绿、2026-09-27 修正轮 C7b 扩 3 掩码臂 =
16 配置后全绿、2026-10-14 远读退役删 `TJE_remote_off` 与
`mask_remote_recompute` 两臂 = 14 配置）——八组合 `combo_*` +
`TJE_shadow_verify`（影子验证）+ C19 三新臂：`TJE_quota_static`
（JOINT_QUOTA_MODE=static）/ `TJE_quota_aimd`（=aimd，F7 自动注入链
全程生效——runner 置位 SH_LINK_TELEMETRY → .sh 追加 --link-telemetry →
C++ 遥测开）/ `TJE_face_static`（--scheduler face_static 对照臂，quota
强制 off 入 manifest 注记 quota_forced_off）；后三臂经 `joint_runner.py`
CLI 面起跑。C7b 掩码臂（纯 env 面）：`mask_recompute`
（JOINT_ALLOWED_ACTIONS=recompute）/ `mask_copy_recompute`
（=copy,recompute），
一律配 `JOINT_SCHEDULER_MODE=load-first`、**不设** combo（预设与显式开关
互斥 fail-closed）；两臂依赖 C4 掩码×边界修复（修复前为记录性失败），
修正轮终态全 exit=0（~/joint_fix_round_20260927/smoke_final/）。
（`TJE_remote_off`/`mask_remote_recompute` 两臂依赖已退役的
remote-read 开关与动作值域，随远读退役删除——旧 16/16 记录为历史证据。）
产物落仓外持久目录 `/home/sunhao/joint_smoke_evidence/
<label>/`（或显式证据根）全套留存——run.log/决策日志/env 快照/退出码/
invocation.json，复审可独立复验；PROVENANCE §7-14/§9/§28）。

**容量压力夹具（R16-4-7，2026-09-15）**：
`sh_test_mesh/run_scripts/joint_capacity_stress_fixture.sh [window_ns]
[capacity_profile] [evidence_root] [combo]`（缺省 10s × stress-28gib ×
TJE，缺省档即可全判据 PASS）——小 HBM 压力档
（`hardware/face_case5_config_c_stress.json` 孪生源，备份→切
trace_config→还原）使 E 逐出真实发生、PARTIAL 形成、跨实例
复合 copy 真实命中；判据 = 退出码 0 + PARTIAL×copy 计数 > 0（noc 腿
layer_end < L）+ deep_gap 台账空（硬门禁，台账侧车缺失 = fail-closed）；
merge_degrade 台账已随 merge v2 冻结（新 run 恒空，R4 自降级退役
2026-09-17——历史 run 侧车兼容读取，判据不再涉足）中量程门禁档 =
150s × stress-128gib（两台账皆 0）/stress-96gib（动态工况最全，见
PROVENANCE §11）。runner 经 `SH_RUNTIME_RC_DIR` 覆盖指向
stress 档 runtime_config（正式跑保持缺省 160gib）。证据落
`/home/sunhao/joint_r16_stress_evidence/`；PROVENANCE §11。

**远端端口并发改造 C++ 夹具（2026-09-24，SerDes 方案阶段 5）**：三个测试目标随
`astra-sim/network_frontend/analytical/CMakeLists.txt` 注册——
`AstraSim_Analytical_Congestion_Aware_RemotePortNwayTest`（端口模型精度 S1–S10
+ stress 模式）、`..._RemotePortOnlineGateTest`（online NodeView 双 MEM 与
MEM+COMM_SEND 门控）、`..._RemotePortStaticGateTest`（static ET 双 MEM 门控 +
仿真结束门，配 `make_remote_port_static_fixture_et.py --out-dir <临时目录>`
生成器，测试二进制 `--fixture-dir <同一临时目录>` 消费）；二进制均在
`build/astra_analytical/build_congestion_aware/bin/`。2026-09-24 终门禁 49 步
（configure + 相关目标构建/运行 + golden + stress）全 PASS
（/tmp/serdes-joint-work/gate_rerun.log；四夹具复跑 r4_nway/r4_online/r4_static/
r4_hbm_on/r4_hbm_off 全 ALL PASS）。口径注：本仓缺省 configure 无可用 ctest
（实测 `ctest -N` = 0 项；原唯一 add_test 的 OfflineGreedyScheduleJournalTest
已随 2026-09-24 死代码清除批删除），回归一律以上述显式命令与退出码为准，
ctest 绿不构成执行证据。

**跨实例 copy 流水化 C++ 夹具（2026-09-25，PROVENANCE §45）**：两个 test-only
目标随同一 `astra-sim/network_frontend/analytical/CMakeLists.txt` 注册——
`..._PipelineConcurrentGateTest`（场景 P：同 rank COMP＋KV-restore＋两笔
charged send 一次 pass 齐发，free 集合空、sends 先于 restore 终结；场景 Q：
两笔并发 RESTORE 与 COMP/comm 同池、动态重分 peak=4/redistribution=5）、
`..._NocSharedLinkNwayTest`（4 NPU 线拓扑共享链路 F2=12/F1=8/F3=6——共享期
严格均分与流完成后动态重分的实证钉）；`HbmNwayTest` 重锚 2（uncharged send
由 ≥700 floor 改精确钉 6——tick 0 齐发、网络侧到达；其余数值锚与 OFF 臂结构
性不变）。本会话复跑：PipelineConcurrentGate／NocSharedLinkNway／HbmNway
ON＋OFF 全 ALL PASS。

## 8. 硬件概念（沿用 sh_3.0 口径；**§7-D' 远端端口并发模型为本仓
2026-09-24 起的显式偏离**）

**A. 晶圆级芯片（wafer-scale chip）＝整个计算系统**：一片二维 mesh 晶圆
（行列数由 `mesh.rows/columns` 给出），芯粒间由片上网络互连，每颗芯粒自
带本地 HBM；NPU 数恒等于行×列，row-major 编号。

**B. NoC（片上网络）**：非环绕二维 mesh 有向链路（die-to-die），XY 确
定性路由、逐跳 store-and-forward、共享链路 FIFO 排队（拥塞感知）；分析
级模型（C++ 侧 `extern/network_backend/analytical/congestion_aware/`，
Python 侧同语义确定性 XY 路由函数）。

**C. 芯粒（die/NPU/rank）**：计算单元（Roofline 计时）+ 本地 HBM（KV
热层）+ 路由端口 +（边缘芯粒）远端内存端口。

**D. 边缘芯粒与片外统一内存池**：`unified-kv-cache-pool` 为 KV 冷层；
仅 mesh 物理边界芯粒直连（远端内存端口），非边缘芯粒借道最近边缘芯粒；
统一逻辑地址空间（跨缘写读）；池容量不设限。本仓启用远端池（全部边缘
端口生效），KV 冷热分层为三态（LOCAL/PARTIAL/REMOTE）。

**D'. 远端端口并发模型（2026-09-24 SerDes 片外链路并发化改造；本节为本
仓相对 sh_3.0 口径的显式偏离，实现
`extern/remote_memory_backend/analytical/AnalyticalRemoteMemory.{hh,cc}`）**
旧"端口严格 FIFO（`耗时 = 端口时延 + 字节/端口带宽`）、一事务完成后再发
下一笔"的串行模型已整体替换，新语义：

- **流体并发**：issue 后各事务固定 latency 独立计时、可重叠，latency 不占
  传输带宽；latency 到期的正字节事务进入该端口传输流集合，N 条活跃流
  均分端口带宽 `remote_mem_bw / N`；任一流字节耗尽立即退出带宽分母，
  幸存流即刻重分。端口内无 FIFO 等待队列、无人为 outstanding 上限。
- **端口共享域（`memory-type`，runtime 派生件 `remote_memory.json`）**：
  `PER_NPU_MEMORY_EXPANSION` = 每 rank 独立端口（本仓生效口径——
  `npu-ids` = mesh 边界 rank，face_case5_config_c 9×6 下 26 个边界端口）；
  `PER_NODE_MEMORY_EXPANSION` = 每 node 共享一端口（`sys_id /
  num-npus-per-node` 映射，缺键启动即拒）；`MEMORY_POOL` = 全部事务压入
  单一逻辑端口。发射门控侧远端 MEM 节点走
  `HardwareResource` 独立计数制无上限槽（`num_in_flight_remote_mem_ops`），
  与 comm/hbm_dma 槽互不占用（两类槽 2026-09-25 起同为计数制无上限，
  COMP/CPU 单槽门保留）；该计数是**节点占用数而非端口流数**，不作任何带宽
  分母。
- **事件精度**：端口内部按连续 ns 子步推进，物理完成时刻
  `fluid_finish_ns` 可为整数 Tick 内的子步值；Workload 可观察回调唯一落在
  整数 Tick `ceil(fluid_finish_ns)`——`callback_tick − issue_tick` 不等于端口
  服务时长（至多约 1 Tick 量化差，小于 1ns 的正传输内部服务可小于 1ns
  而回调仍跨到下一 Tick）。bytes=0 且 latency=0 的事务不再同步完成，改为
  独立一次性定时作业精确 +1ns 异步回调（不进带宽集合）。全部端口与
  双零定时作业共享挂在首次 `set_sys` Sys 上的**单个**全局最早 deadline
  可取消变迁事件；同一 Tick 到期的完成跨端口收集后按
  `(port_index, issue_seq)` 升序整批交付，交付前先结算端口统计。
- **限制与口径声明（模型假设，非硬件事实）**：① `remote-mem-bw` 沿用
  历史口径：JSON 数值（名义 GB/s，当前 512）直接按 **B/ns** 消费，单位
  解释未经物理资料核实；② 远端端口与 NoC/D2D 是相互独立的资源，本改造
  不建立封装带宽耦合；③ Python 决策侧池计价（D04 2026-09-26 起
  `_pool_transfer_ns_shards` 逐 shard 式(7)：λ = 池时延 + 逐跳时延，
  带宽项 = 池端口 ∪ 边界↔端点 NoC ∪ 端点 HBM 取 max；池端口除数按流
  查询实际 edge，remote_load/remote_store 端点入 HBM 端口表）仍是发射
  时点快照除数下的解析估计，与 C++ 流体服务不同口径，不能当作端口并
  发服务时间的预言；④ 固定输入 2s 冒烟
  窗内池后端完成事务数为 0，基线/改造两树全系统输出恒等属零池负载下的
  平凡一致性，不构成争用条件下并发模型的正确性或收益证据（争用行为
  证据目前仅有后端级确定性夹具与压力夹具，见 §6）；⑤ 本仓不设
  serial/parallel 行为开关，旧串行路径已删除、无开关可回退。

**E. 实例（instance）**：紧密相邻芯粒的连续实心矩形，TP whole-head 分
片；全部实例铺满晶圆；布局唯一来源 `trace_config.csv` 的 `inference_
group` 行。P 与 D 自 2026-09-26 D18 起可在不同实例执行（P→D 边界二次
决策，原"统一实例"钉扎废除）；一 session 的权威 KV 基础历史驻留单一
home 实例（执行副本另计）。

**F. 七仓对照表中本仓行**：三机制联合（T 逐出序 + J 联合选择 + E 自适
应层数）；三态 KV；home 保持 + 完成合并；八组合消融。其余六仓见
`template/astra-sim-sh_3.0/README.md` 对照表。

**G. 本地 HBM 带宽模型**：N 用户严格均分流体模型（读写同价、事件驱动
重分配、端点计费与直通规则）——沿用基底 `LocalHbmBandwidthModel`。

**H. KV 逐出与推理计算并行执行**（2026-09-13，基底改造随工作树继承）：
逐出物理传输旁路支链化 + store→restore 前递依赖补偿 + 瞬态双占用窗口
口径——本仓 merge 回传流同样经 `_register_store_tails` 登记，下一轮池读
经前递补边等待写入完成。

**I. D2D 带宽孪生集与 ρ 档位**（2026-09-22 C18，WP7/DSE 输入合同）：
`hardware/` 在 base 源 `face_case5_config_c.json`（D2D 4050 GB/s）之外
交付三个 D2D 带宽孪生——`face_case5_config_c_d2d_x05.json`（2025 GB/s，
x0.5）、`_d2d_x2.json`（8100 GB/s，x2）、`_d2d_sub.json`（1200 GB/s，
**ρ<1 档**）。孪生对 base 的差异严格限于
`{slug, label, d2d.bandwidth-gbps, notes}`（其余逐字段 identical——
round-trip 测试钉死：`tests/test_hardware_twins_roundtrip.py`，三孪生 ×
2 容量档 = 6 组合的 exact-key schema/ρ 派生/canonical 写回零漂移/白名单
diff 四断言）。派生 **ρ = B_D2D/B_HBM**（简化解析锚点，仅历史扫描项
平价——设计文档 §1.2）：base 4050/1640 ≈ 2.4695、x05 ≈ 1.2348、
x2 ≈ 4.9390、sub ≈ 0.7317 < 1（带宽平价条件不成立——单条流亦不满足
相对无争用峰值 HBM 的平价；**不排除**远端负载更低/容量压力更小带来的
系统收益，收益方向不预设，选中率口径的域形态是开放研究点）；配额
Q_init = max(1,⌊ρ⌋) 同源派生（x2 档 → 4，其余 → 1）。孪生经
trace_config 行切换 + `SH_RUNTIME_RC_DIR`（`joint_capacity_stress_
fixture.sh` 同款范式）选用。

执行驱动机制层（RequestIngress/DecisionMailbox/GraphBatchCommitter 等）、
typed 响应解析（C1）、在线二进制旗标（`--online-*` 家族）、运行开销与
日志瘦身开关（SH_ARCHIVE_RUN/SH_SLO_POSTPROCESS/SH_GRAPH_DIGESTS 等）、
机制回归 fixtures、SLO 后处理链（slo_tools 九项产物）、冒烟输入纪律
（astra_compute_20.csv 前 2 秒）——均沿用基底实现与口径（详见基底
README 同名章节；本仓未改动其语义）。

## 9. 边界与纪律

- 仿真输入唯一允许源与窗口遵循 Agents.md 冒烟规范。
- 缺失输入 fail-closed；deep-gap 逐无可逐仍不满足按 rank 记账后
  fail-closed（沿用 D4）。
- 决策输入无 oracle：CSV 未来行/未来输出长度/真实返回时间不进任何
  决策；完成时已观测长度仅用于实际结算与在线估计器更新。
- E 的 adaptive 公式与因果规则已冻结（§5）；不做逐 workload 拟合。
- 本仓不运行三机制消融/外部对照实验（§8 第 6 步另行登记）。

## 10. 2026-09-30 修复追记（§8.1 服务样本单账本 / §8.5 epoch 接线 / §8.4 名实订正 / A12 注释残留清理；零新增开关）

- **§8.1 单账本落地（hist4）**：纯列车服务样本（decode/prefill 纯段核销 span）唯一更新点 = W6 `ServiceSampleRecorder`（`sh30_online_scheduler._observe_unified_service_sample`）。J 计价 `ServiceFactors` 构造注入 `unified_source` 转为只读发布视图——prefill/decode 因子在 `flush()`（JCM 构造点）刷新自统一估计器 `efficiency_factor`，`observe_prefill`/`observe_decode` 在发布模式 fail-loud（`JointCostError`）；face γ/η 消费方（k_hide 递推/adaptive 披露）经 KVCacheManager `service_factor_source` 注入的 `PublishedServiceFactorGroup`（event_recursion_predictor 新类）读同一发布值，`observe_valid_service` fail-loud。**M4 时间衰减桥接退役**（同一 prefill 样本二次更新 face `ServiceFactorGroup` 的 `α=1−exp(−Δt/τ)` 通道——正是 D14 已退役公式，且旧通道连亚 roofline 异常样本都照收初始化）；"对拍期双账本并行"表述随之作废。transfer 遥测通道不变（链路窗口样本不与 T/E 共享，仍由 `ServiceFactors` 自身承担）。决策日志 `service_factors` 披露键集不变；`updates`/`rejected`/`anomalous` 键集生产收窄为 transfer 通道，prefill/decode 因子值 = 统一估计器发布值（冷启动 = `params.service_efficiency_init` 的倒数方向，缺省 1.0 零漂移）。
- **§8.5 epoch 接线（hist6）**：sh30 装配新增共享 `_joint_epoch_log`（EpochLog，W6 装配块第四件）——注入 `BackgroundForecastEngine`（distribution_shift 回退分支自生产可达，不再只靠测试手工注入复现）与 `ServiceSampleRecorder(epoch_log=...)`。运行期唯一可分叉源 = 服务样本硬件版本：`ServiceSampleRecorder` 纯度裁决前登记版本，混分叉恰一次 `mark_epoch_change(kind="hardware")`（no_reset_quarantine 规则不变）；配置/策略/硬件摘要均在装配期一次读取 fail-closed、run 内不变（策略轴运行时校验点 fail-closed raise 而非继续，无"变化可标记"面——如实登记，不硬造挂点）。`joint_prediction_manifest` 新增 `epoch_log` 键（as_dict：version/reset_rule/quarantine_ns/epochs——空 epochs = 本运行无 epoch 触发源的显式登记，非缺省缺席）。
- **§8.4/A18 名实订正（hist5，注释级）**：`action_work_graph.projection_input` docstring 不再引用 §8.4"模板读该候选预测世界内的状态"为现行接线背书——现接线仅为共享资源竞争耦合（候选腿/预测腿互相拖慢 + FORECAST_ARRIVAL 重仲裁）；候选级目录/容量/驱逐重派放置与恢复量（共享冻结计划的候选内物化）**未实现**，依赖 KV 目录投影与容量摘要的候选级只读面，列为专项施工缓修项；uncertainty 披露（RESTORE_PRESET/KV_LOCATION_UNKNOWN/POOL_PORT_PROXY）保持。
- **A12 注释残留清理（legacy2，注释级）**：`sh30_online_scheduler` 结算策略常量注释更新为 W9 后事实（consolidate/distributed 双值均合法分派，`_joint_settlement_policy` 单点裁决未知策略 fail-closed；`_reject_distributed` 已退役、引用作废）。纯注释勘误，无行为变化。
- 测试面：`online/test_joint_m_batch.py`（M4 钉改发布视图钉）、`online/test_joint_n_batch.py`、`online/test_joint_o_batch.py`、`online/test_joint_c6_sample_integrity.py`、`joint/test_joint_review2_fixes.py`（断言迁到统一 recorder/发布视图）；`joint/test_history_estimator.py` 增硬件版本分叉→epoch 挂点钉。

## 最新正式验收（2026-10-11）：closure-preview-context-prepared

当前正式性能冻结绑定 53 项输入、源码、二进制和工具，测量期 source_bindings_unchanged=true；包含 HBM scratch、native preview、resource context 与 obligation 修订。52-binding closure-slim-ready-native 及更早的 frontier 数据均保留为历史 source scope。

三对九臂 T3 全部自然完成。工程、近似额外、总倍率中位数分别为 0.6790036702×、0.9140707531×、0.6276408495×；范围分别为 0.6126955669–0.7039552280×、0.9116539733–1.0243926731×、0.6206573962–0.6417635807×。这是固定小任务，三项中位均低于1×，近似额外范围跨1；没有证明整体加速。旧 v1 对 physical ACK 类别与 parent edges 缺少可观察字段，不把它们记作跨版本验证通过。

修正入口 T4 保留 8 requests/3 sessions/61,550 prefill/1,646 decode、K2/ppm0/margin20000/eventbudget262144、全候选/full validation/journal/必要 ACK/natural settle。180.016744250 s TIMEOUT、峰组 RSS 3,038,920,704 B，末次 33 deliveries/tick 7,974,622,822，仍有业务进展但没有自然闭环；speed_goal_met=false、implementation_complete=false，无完整任务倍率或20×通过。首次 30.010625 s 尝试因未重绑 Python trace_config.request_queue_csv 而没有业务执行，标为无效启动，不作性能证据；实际 loader T0 0.0682787 s 确认相同 workload 后才运行修正入口，配置已恢复。

心跳 eq_proceed 173.075345692 s/6,214,533 calls 是包含 callback/bridge/commit 的外层桶；bridge 126.853137860 s 和 outer issue 26.287039746 s（DIE 7.638294696 s、差额 18.648745050 s）有嵌套关系，当前打开的 scope 未计入，不能相加或用180 s减桶归因。T4 前缀记录 9,544 次 DIE adapter calls，但未完成完整 workload。局部验证见 remaining-preview-context/validation-summary.json：all-target build 6.398614 s PASS、21项 CTest 5.067288 s PASS；Python 56-method 首轮55 PASS、1 fixture error，修正 fixture 后只重跑该方法通过，不称首轮全绿。累计仿真/投影 1,444.107352131 s、350次 guard；本次授权窗口有效仿真 212.013037701 s/240 s，资源窗口已结束，旧长臂未运行。

### 2026-10-11 target-slot V4 续工（语义与构建验收已完成；性能验收见上）

当前 target-slot 增量的目标生产合同为 strict V4：schema/protocol 4、`bridge-v4-20261011`、`workledger-v4-20261011`，snapshot handle kind 为 `work_ledger_v4`。target row 使用精确稳定键和有序 delta operations；claim 一次声明 owner 的完整 target membership，consume 携带实际新插入成员、indexed cause 与精确 predecessor lineage。只在同一 logical transition 内合并同键操作，公开 snapshot 仍重建原完整 row 与字段形状。source slot rows 保持现有 before/after 全行表示；ordinary FIELD 1、outer bridge schema 2、node/pipeline logical sequencing、ACK/dependency/credit/obligation、物理 callback 数和 K2 精度合同不变。物理资源键仍为 `slot:<domain>:target:<index>`。在线入口严格选 V4；保留的 V3 journal 使用明确 archive decoder 按原始 child bytes 离线重放，不能混版或回退到 V3。journal child bytes 作为 opaque payload 保存。`test_sparse_pipeline_integration` 的 `--raw-v4`/`--archive-v3` 和 cost-test CLI 都是离线测试入口，不是生产开关。

该 V4 变更已完成完整最小语义与构建验收并保护回写 live（证据摘要见本文档文首 2026-10-11 段）：真实磁带五帧、V3/V4 双 harness 逐前缀重建等价、真实 Online 闭环与生产 journal 身份/重放、三个坏输入缺口负例、21 CTest + 五个直跑 target、D20 344-tile 构造与历史 descriptor 校验、最终非 trace 回归 2316/3 预存、SLO unittest。consumer 59-method 与调用方 21 项的分批覆盖见 `logs/target-v4-*.log`；seeded consumer 成本夹具在固定 T=4、S=8/32/128 下测得 consume canonical 4,685 B、indexed cause lookup、0 次 consume-run 遍历、每个触及 owner 一次 COW 与一次 freeze；claim 只编码 S 个成员和 S+2 个几何值。固定 S=128、T=8/32/128 时 canonical 分别 8,851/33,904/134,277 B，实际 additions 8/32/128，cause reads 16/64/384（最后一档含 128 项 clear proof），每个触及 owner 一次 COW/freeze。这些是 seeded consumer 算法成本，不代表完整 workload wall time；物理 producer 的 S=T 字节/计数证据为 `logs/target-v4-producer-wire-bytes-and-atomicity.log`。

CPP 初次 all-target 构建 7.043183 s 因 optional shard JSON 类型失败；根因已用 target `nullptr` 的局部修正处理并核对反向补丁 SHA。修正后的 228-binding all-target 构建 PASS（`target-v4-final-affected-caller-closure`，49.208369 s）。direct C++ fixture 已覆盖 whole-commit→close→settle→retire→reclaim 链（见 stale-owner 负例与 erase/reinitialize 正例）。Publisher append 只承诺隔离回滚 metadata sequence/lineage；若 append 在 RRE 已改物理状态并移出 patch buffer 后失败，main 会中止，本实现不支持同进程重试，也不增加 generic checkpoint。

因此最新正式性能结果仍是上文 `closure-preview-context-prepared` 的 53-binding 冻结：完整 T4 为 180.016744250 s TIMEOUT、33 deliveries，`speed_goal_met=false`；V4 代码不追认旧 T3/T4，也没有新的完整测量。仿真/投影累计见 manifest；whole-trace 预算按 v2 纪律管理。V4 语义、物理 producer 和交付保护已由 root 串行验证收口。

[ASTRA-sim](https://astra-sim.github.io/) is a distributed AI system simulator.
It models the end-to-end software and hardware stack of modern AI systems.
For our website and documentation, see [astra-sim.github.io](https://astra-sim.github.io/).


## 数字来源表（SDK 数字的唯一来源链；2026-10-17 P12-A 落档）

本仓所有 die/coarse 相关数字的来源、身份与失效条件。**发布身份由
`extern/die_model/RELEASE.json` 的 source_digest 承载**（configs/VERSION 的静态
字符串不是发布身份）；当前五仓一致 digest =
`3a021f37b71fb3ea52a02a0c522169cd5a8374ef1f77c2d6a20fe504b274c31c`（33 文件；`die/tools/sync_repositories.py --check` 2026-10-17 实测 passed；canonical25 正式发布声明见 die/docs/RELEASE-canonical25-20261017.md）。

| 数字类别 | 仓内位置 | 唯一来源链 | SDK 身份 | 门槛与失效条件 |
|---|---|---|---|---|
| 单 op 估计时延（estimate_latency_ns 等） | `sh_test_mesh/workload/llama2_7b_inference/online/die_model_adapter.py`（build_spec → C ABI） | emitter → libdie_model_cabi → `extern/die_model/coarse` | digest 见上 | digest 或 plan fingerprint 变更后全部数字失效重算；非整数时延 fail-closed |
| lifecycle 合同（含 R4 重复 region） | `astra-sim/workload/execution_driven/tests/die_execution_lifecycle_test.cc` | 测试 → vendored SDK 编译链接 | digest 见上 | guard fail-closed：prediction_status=not_available 即失败；legacy ×R 聚合保持 not_available |
| 生成式 repeat（R≥2 ffn_region）声明状态 | 同上 adapter（build_spec 条件声明） | emitter → die_op_spec repeat_seams → Plan v5 | digest 见上 | **是（2026-10-17 Wave 3 补声明，共识④）**——adapter 对 layer_count≥2 的 ffn_region 条件声明 `repeat_seams={mode:serial_on_drain}`（与四仓同款；test_die_model_adapter 的 R24 用例显式断言声明存在且其他 op 不携带） |
| coarse 投影/adapter 合同测试 | sh_test_mesh/workload/llama2_7b_inference/online/test_coarse_die_adapter.py | adapter/投影单测 → vendored SDK | digest 见上 | 合同断言；die 侧 not_available 状态按 fail-closed 处理 |
| 硬件标量（derive_coarse_die_hardware） | `sh_test_mesh/config_resolver.py` | manifest → C ABI derive → 本仓派生标量 | digest 见上 | manifest 或 SDK digest 变更后重验 |
| et_def Python 绑定（et_def_pb2.py） | `extern/graph_frontend/chakra/schema/protobuf/et_def_pb2.py` | 跟踪的 et_def.proto + **钉版 protoc 3.21.12** 仓根调用再生成 | n/a（仓内跟踪） | `verify_et_def_pb2.py` + `sh_test_mesh/tests/test_et_def_pb2_consistency.py` 防漂移；drift 即失败，protoc 缺席显式 skip |

### 环境与目标名注记（P13/P14）

- **protoc 钉版 3.21.12**：再生成/校验 et_def_pb2.py 必须使用该版本（仓根调用
  `protoc -I . --python_out <tmp> extern/graph_frontend/chakra/schema/protobuf/et_def.proto`）。
- **ctest 目标名以 `ctest -N` 输出为准对账**。已知族间差异：admission 测试目标在本仓族为
  `（本仓无 admission 测试目标）`；其他仓族命名不同（face 族带 `Congestion_Aware_` 前缀、joint 无此目标），
  按 `ctest -R` 前缀过滤会静默漏测。

### repeat（R≥2 ffn_region）声明状态（2026-10-17 **已补声明**，共识④执行完毕）

本仓 workload 产出 R≥2 的 ffn_region（ffn 8 层 × 3 遍 = R24；`_spec_record` 默认链
R96）。按 2026-10-17 共识决策④，Wave 3 已在 emitter `build_spec` 补条件声明
`repeat_seams={mode:serial_on_drain}`（照四仓先例；仅 ffn_region 且
layer_count≥2，其他 op/策略不变）；`test_die_model_adapter.py` 的 R24
用例新增显式断言（ffn 声明存在、其余 op 不携带），全套 30 passed。R≥2 预测今日走
`repeat_seam_modeled`（不再是 not_available 能力缺口）。历史设计/验收见 die 仓
`die/docs/analysis/repeat-generative-v1/`。

## 出厂计算后端：coarse_die_v2 die 内仿真器（2026-10-18 阶段2 翻转批）

本仓出厂默认计算后端为 **coarse_die_v2 单 die 解析仿真器**：system 模板
`die-model-profile: coarse_die_v2` + 硬件 `coarse-die` 节（完整 coarse_die_v2
描述符）+ 容量档 `validation-coarse-144g`（恰等于 coarse 派生每-die HBM 容量
144 GB，resolver fail-closed 对齐）。真实重复区域（layer_count≥2；2026-10-19 起 coarse spec 的 weight_passes 恒为 1——spans 已逐迭代枚举，repeat 计数只由 layer 实例数决定）
以声明式 `repeat_seams={mode: serial_on_drain}` 发射，coarse 编译器按单实例
模板建模、运行时逐实例串行推进并以逐实例区域开销为 seam（引擎 2026-10-18
实例 repeat 通用化批，见 template/die/docs/HANDOFF.md）。旧 `mesh_die_v1` /
`tx8_reference_v1` 保留为显式 legacy 选项（硬件 `die` 节 + `validation-die-64g`
档配套），coarse 与 roofline-enabled=1 互斥 fail-closed。

已知边界：运行时推进路径保留 P8 记载的 O(R²) 活动向量墙（P7-B 遗留项），
大 R 计划的在线排空 wall 随 R² 增长；估计/编译侧已有 P9 闭式快路径。
