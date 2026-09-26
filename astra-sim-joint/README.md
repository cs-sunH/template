# astra-sim-joint - 三机制联合策略仓（T＋J＋E 消融统一实现）

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
| **J** 联合决策 | `scheduler_mode=joint` | 全 instance 候选（无容量/边缘/距离掩码），发射点联合选择 instance × stay/recompute/copy/remote-read | `load-first`：先按在线负载（ns 服务台账）在全部 instance 选点，再在该位置选 KV 动作；`affinity-first`（home 优先、缺省回退 load-first）为 J 边际比较第二参照，不进八组合；`face_static`（C17）= 静态距离对照臂（policy variant，不进八组合、不冒称完整 FACE，强制 quota-off——F7 唯一耦合例外） |
| **E** 自适应层数逐出 | `layer_policy=adaptive` | 按层消费期限公式计算最小热前缀 `k_hide` 软目标（§5.2），释放时目标外后缀→目标内后缀两次扫描 | `minimal_layer_groups`：同一合法对象排序下按逐 rank 缺口释放最少完整层组；`legacy_half` 为原半层逻辑回归对照，不是 E-off |

**remote on/off**（`remote_actions=on|off`）是与三机制正交的能力消融
开关：off 仅从候选动作集中移除 remote-read，全部 instance、其余动作、
驱逐、合并与生命周期不变；单独记录，不与 T/J/E 合并。remote-read 的
执行口径为 **credit 交错流唯一机制**（2026-09-17 用户裁定，无执行口径
开关）；块大小 K 经 `JOINT_REMOTE_CREDIT_ITERS`（`auto`=自适应
max(1,⌈S_j/8⌉) | 显式正整数，K=1 仅验证配置）调节，见 §3"remote-read
执行流"行。**PARTIAL 基适用面消融**（2026-09-17《部分层逐出kv管理改造
分析方案》需求①）：`JOINT_REMOTE_READ_PARTIAL=on|off`（缺省 on）——
on 时 PARTIAL 基（前缀驻留 home＋后缀在池）进 remote-read 候选（混合
形态：准入相后缀池恢复物化＋decode 相前缀层 [0,p) credit 读流）；off
只把 PARTIAL 基的 remote-read 移出候选集（copy/recompute 照常参与比较，
与 `remote_actions` 同一消融模式，非旧机制回归档）。

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
JOINT_REMOTE_ACTIONS=on|off          # 缺省 on
JOINT_REMOTE_READ_PARTIAL=on|off     # PARTIAL 基 remote-read 消融（缺省 on）
# 与三机制/combo 正交的治理与遥测开关（C6–C11；缺省 off/未设）
JOINT_QUOTA_MODE=off|static|aimd     # 通用流量治理（aimd ⇒ 自动注入
                                      # --link-telemetry，F7 耦合）
SH_LINK_TELEMETRY=1                  # C++ 链路遥测旗标（显式直跑遥测臂）
```

开关入口裁定（2026-09-14 用户裁定，经 kimi 审查上呈）：**接受 JOINT_*
环境变量族作为策略开关唯一入口**（原执行方案 D7 的 joint_config.json
路径不实施；偏差登记见仓根 `PROVENANCE.md`）；**默认 = 完整三机制
TJE**（原执行方案 D9 的 affinity-first/legacy_half 回归对拍档经显式
env 获得，即 §8.3 #9 哨兵配置）。推荐经 `joint_runner.py` 起跑（D14）：

```bash
python3 sh_test_mesh/run_scripts/joint_runner.py <run_dir> <request_csv> \
    --combo TJE            # 或 --category/--scheduler/--layer/--remote 显式开关
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
│   ├── joint_scheduler.py        #   J：joint/load-first/affinity-first/
│   │                             #      face_static 选择
│   ├── hbm_port_flow_registry.py #   实例 HBM 端口流注册表（u_port 除数）
│   ├── link_quota.py             #   通用流量治理：链路∪端口动作级配额
│   │                             #   + AIMD 控制律 + ρ_eff/Q_init 派生
│   └── event_recursion_predictor.py  # E：运行期事件递推预测器（adaptive
│                                 #   正式在线实现，写回×恢复双方向）
├── face_scheduler.py             # KVCacheManager：T/E 注入 + home/工作
│                                 #   副本账本 + merge_back 合并事务
│                                 #   + kv_delta_journal 披露 + 逐组恢复
└── online/
    ├── sh30_online_scheduler.py  # joint 准入（替换三段式）+ 因果 decode
    │                             #   增长 + 完成合并 + remote 读流
    │                             #   + #readplan 预登记核销 + 配额准入
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
  服务落地即改写）。
* **执行与增长（§3.1）**：新 KV（prefill 输入 + 实际 decode）在执行
  instance 产生；准入预约**动作感知足迹**（R1'：stay/copy/recompute
  按 history+input 整份、LOCAL 基 remote-read 仅 input 增量；PARTIAL
  基跨实例 remote-read 还须逐 rank 预约池恢复的历史后缀——输入 token
  基数由 `joint_reservation_context_tokens` 统一，后缀单独记账并在物化
  后抵扣；可行性/预约/物理可行性三处同口径；
  无 PARTIAL 驻留钉扎：容量只影响所需字节与驱逐等待计价，不做候选
  掩码），decode 按列车核销的实际消费**因果增长**（每次增长经同一
  T+E 释放机制准备空间；容量不足时进入**停滞/唤醒**（R14）：会话
  暂缓后续列车参与，容量释放（KV 纪元 ⊕ 实例纪元重试键）后唤醒，
  等待如实计入 E2E，全停滞死锁由守卫显式 fail-closed；不提前按真实
  最终 decode 长度预约——`final_context_tokens` 不进任何决策输入）。
* **合并（§2.2/§3.2，merge v2 少并多——2026-09-17《部分层逐出kv管理
  改造分析方案》需求②）**：`merge_back` 在完成处理时比较**两侧保留量**
  （home 侧 = 基础驻留前缀 H；exec 侧 = 工作副本 W——remote-read LOCAL
  基为增量 I、PARTIAL 混合形态为池恢复后缀 S＋增量 I、copy/recompute 为
  并集），**小的整份 NoC 搬给大的**（省 D2D 流量；时机不变）；合并
  **零池写**（I6：恢复的 KV 按热 KV 处理、不执行完再踢出去——用户
  裁定），结果恒为**全层 LOCAL@胜者**、`home := 胜者`；copy/recompute
  执行端恒持并集 → **反向零字节翻转**（零传输、home 侧基础释放）；
  REMOTE 基 = 无主 session（裁定③）：就地保留（零传输零池写）、
  工作副本转正、home := exec（"远端服务一次后顺势回暖"）。空间准备只
  经统一 T+E 逐出（`_ensure_capacity`，胜者侧；零字节翻转/in_place
  无需准备），remote-read 双向二选一兜底，**双侧深缺口才 fail-closed**
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
* **动作语义（§2.2 表；四动作 = stay/recompute/copy/remote-read，全
  候选逐 (instance, action) 对联合 argmin，实例永不掩码）**：
  stay=本地命中/部分恢复；copy=基础历史复制
  为工作副本（前缀 NoC + 缺失后缀池恢复）——**实现/测试锚点（R16，
  2026-09-15）**：物化 = `face_scheduler.py` `_noc_transfer`（层区间化，
  `[0, base_prefix)`）+ `_remote_load_transfer`（`[base_prefix, L)`）、
  计价 = `joint_cost_model.py` `estimate_action` ACTION_COPY 两腿
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
  与 `#copy-stream`（一遍历史）入决策日志审计；recompute=**只重算缺失区间**（R13：@驻留目标复用权威前缀、
  仅物化缺失后缀层（span 基 = H，recompute@home 与 stay 同构本地提交、
  merge 零流量）；@异地/REMOTE 基础整份重算（span 基 = 0））；
  remote-read=基础历史
  留 home，执行端驻留新增量 KV——**混合形态（N1(a) 解除，2026-09-17
  需求①）＋ 前缀读流分两段接力（2026-09-25 prefill remote-read 分阶
  段）**：LOCAL 基 = 基础全层留 home、执行端仅增量驻留（prefill 期前
  缀读流覆盖全层 [0,L)，history_transfers 恒空）；PARTIAL 基 = 后缀层
  [p,L) 准入相从池**恢复物化**到执行实例（热 KV，复用 copy 池腿原语/
  发射/前递补边/空间准备）＋ 前缀层 [0,p) 读流直接用（不先复制再计
  算）——准入相 prefill 前缀读流（`plan_prefill_remote_read_transfers`
  纯规划、图侧旁挂分支逐层段门控；瞬时 staging 不入容量账本）与
  decode 相 credit 读流两段接力（同一条前缀读流不得同时登记为 prefill
  流与 decode 流，§二.5）；REMOTE 基仍拒（无主，走池
  恢复/重算就地转正）；适用面消融 `JOINT_REMOTE_READ_PARTIAL=off` 时
  PARTIAL 基照旧不进候选。
* **链路遥测（C6–C8，2026-09-22）**：C++ 在线旗标 `--link-telemetry`
  每决策 epoch 在桥请求顶层携带 `link_telemetry[]` 逐链路窗口差分；
  SH 侧 `_ingest_link_telemetry` 解析（整型 LinkId → 端点 (src,dst)
  键换算）喂 JCM `divisor_effective`（max(注册表瓶颈, 并集遥测下界)
  合并）与 AIMD；remote-read 准入以 `rid#readplan` 预登记读流承诺
  （est 账本；**2026-09-25 起准入相注册表零登记**——prefill 期间注册
  表上的远读流 = `rid#prefill_read` 前缀读流实流本身：准入登记 NoC
  路径 + home 读/exec 写双端点 HBM 端口、prefill drain 释放，逐 shard
  单代表流折叠；同一条前缀读流不得同时登记为 prefill 流与 decode
  流），完整 credit 块数只用于时域对账，`#readplan` 注册表半边自
  drain 对账（`_reconcile_readplan_at_drain`）起按每条 TP shard 路径
  登记一条未来代表流；当前列车发射时由该列车的实际代表流接管，
  列车完成后按剩余工作恢复未来流，不把串行块误计为并发流；
  drain 真计划冻结处对账核销（`readplan_reconcile` 差值行）+ 完成边界
  残差清结（`readplan_settle`）+ run 末泄漏审计；同 tick 读流承诺对
  第二笔决策可见（互见性）；collective_coverage 翻转条件（全程开启 ∧
  逐 epoch 窗口无空洞）入 manifest 与决策行。
* **配额机制（C9–C11，2026-09-22，缺省 off）**：`JOINT_QUOTA_MODE`
  = off|static|aimd（三开关之一，与三机制/combo 正交）。链路 ∪ 实例
  HBM 端口的**动作级准入**（逐候选判据、实例永不掩码；全动作不可行 ⇒
  `quota_deferred` 回队，等待行 `wait_reason ∈ {capacity, quota_link,
  quota_port}` 三分）；static = 固定预算（链路 `Q_init =
  max(1,⌊ρ_eff⌋)` 逐配置派生、端口平价门 `B_HBM/(u_port+1) ≥ r̂_KV`、
  merge bulk 名额 N_bulk=Q_init）；aimd = static 基础上遥测驱动的 AIMD
  动态调整（F7：aimd ⇒ 发射层自动注入 `--link-telemetry`，不存在
  "aimd+无遥测"合法启动路径）；流生命周期借还配对审计（remote-read
  读流 + merge 预留 `#merge-reserve`，service_done 裁决释放败者侧 /
  merge_done 闭合门后释放胜者侧）；δ_adm=0（A3' 终冻，与 argmin 合取
  恒等）。off 臂决策序列与 off 前逐字节同（F7 零漂移实证）。
* **逐出（§4/§5.6）**：T 定类别/对象序（`eviction_class_order`），E
  定层数（`LayerEvictionPolicy.plan_release`，只读计划、提交前复核）；
  逐 rank 缺口检查（不用实例汇总掩盖单 rank 超容）；deep-gap 台账
  fail-closed 沿用基底 D4（准入/merge/decode 语境按 N3' 类型化
  `KVCapacityError` 分别转延迟/自降级/停滞，合同类异常原样上抛）。

## 3. 状态披露（《设计方案》§5.6/§8：已设计/已实现/已验证分开陈述）

| 项 | 状态 | 说明 |
| --- | --- | --- |
| T 开关（typed/lru） | 已实现＋单测验证 | 类别严格序、满足即停、未知类别 fallback 披露 |
| E：minimal_layer_groups / legacy_half | 已实现＋单测验证 | legacy 与基底逐字节等价（既有测试回归通过） |
| E：adaptive（k_hide） | **已实现＋单测验证（2026-09-22 C15：运行期事件递推预测器 = adaptive 正式在线实现，F12 三态闭合）** | 解析例 1/17/25 两径一致；递推预测器（`joint/event_recursion_predictor.py`）：统一 N-way 均分时间线（与 C1 divisor_multi / C2 u_port 同源口径）、写回×恢复两方向同推演仲裁、恢复写腿×计算 memory 腿同端口仲裁、候选自致减速单列不回移期限、剪枝两条钉死（逐 rank 逐腿独享速率串行累计下界；D̂ 钉在无候选恢复流量的消费开始时刻，prune vs 全枚举 50 例一致）、未知 ETA/解析不可用保守全 L + 披露；η/γ 因果更新入口 `observe_valid_service_sample`（混合等待默认不可观测）；adaptive_decisions 披露侧车（source/statuses/divisor/stall/η/γ） |
| J：选择与代价模型 | 已实现＋单测验证 | 解析近似代价（关键路径合成、因果时域估计）；**在线反馈通道已接线（R15，2026-09-14）**：链路流登记表按逐 shard 全路径登记/完成事件注销（F-B 并集瓶颈除数）、池端口仲裁份额（含 E 内核 r_j，P1）、ServiceFactors EWMA（P3 α=1−exp(−Δt/τ) 时间衰减；transfer 因子保留接口位——节点级传输完成遥测未交付，传输争用在线修正由除数通道承担；样本纯度排除 joiner 迁移/partial 恢复/copy 门控传输/remote-read 读流列车——复审 M5）；decode 负载标定在线化（N12：session 均值 → run 均值 → 冷启动 1，全 trace 均值常数不再进决策输入；估计器版本入快照缓存失效键——复审 M3）；merge 段增量计价（复审 K4：input + 因果 decode 增长，不随基础历史膨胀；REMOTE 基走池端口口径——自查 C）；准入重试键 = KV 纪元 ⊕ 失败候选集纪元（复审 M2，R2.5 口径）；无 oracle（`RequestView` 结构性不接受 decode_length/final_context_tokens） |
| home/merge 账本事务 | **merge v2 已实现＋单测验证（2026-09-17 需求②）** | 少并多方向裁决（两侧保留量比大小、小侧整份搬大侧——min 双向计价与物化同判据）；零池写（I6：merge 事务不产生本会话 remote_store）；结果恒全层 LOCAL@胜者、home 迁移胜者（全仓第二个 home 赋值点）；copy/recompute 反向零字节翻转（零传输、无空间准备）；REMOTE 基就地保留（裁定③）；空间准备只走统一 T+E `_ensure_capacity`（R4 自降级/k=0 池归并兜底**退役**——双侧深缺口才 fail-closed，合同变更登记 PROVENANCE；`merge_degrade_events` 冻结、新 run 恒空）；版本键恰好一次（R4/N9 沿用）+ 到达合同 = service_done 锚 + 依赖门控（C3b，见 §2）；容量缺口台账落账边界 = 确认终态（K6 沿用；run 末导 `joint_kv_ledgers.json` 侧车含 deep_gap 台账）；decode 停滞死锁守卫与逐列车增长逐出三路进图不变（自查 A/D） |
| remote-read 执行流 | **已实现（credit 交错流＋2026-09-25 起 prefill 前缀读流分阶段，唯一执行口径）＋单测验证** | 读流按列车切片、切片内按 K 迭代分块（K=`auto` 自适应 max(1,⌈S_j/8⌉)——每成员每列车块数 ≤8、节点膨胀与 S 解耦；或 `JOINT_REMOTE_CREDIT_ITERS` 显式正整数，K≥S_j 时单块、块 1 走原 pd_transfer 发射路径=逐字节等价锚），credit 块只栅栏对应计算体块——读与 decode 计算重叠（D2 拓扑，2026-09-25 P4 起尾块逐笔并行支链：块间无边、每块 fork 主链 frontier，块内 ack_recv_b←本块 send_b／ack_send_b←本块 recv_b、跨 rank 因果由 tag 配对在运行时承载；每 recv_b 完成门=体块 b 的 arm 门）。首列车块 1 走 pd_transfer 主链（barrier 前，barrier 语义自动降级），尾块旁挂支链（发射序 checkpoint→尾块支链→restore→块 1，D2）；续列车（续坐成员，D7 新槽位）块 1 上主链（无 barrier，per-rank 链序先行——**T1/T2+ 块 1 门控强度不对称为已登记披露**：T1 经全 rank barrier 同步，T2+ 仅 per-rank 链序）；多 remote 成员同列车时体块门取覆盖区间并集（I2）、K 取列车统一值 max(成员 K_j)；R15 逐切片键 `rid#decode#{j}` 登记于切片创建点（列车规划期——drain 边界 S_j 未知）、Tj 核销边界注销（无陈旧多计，R-6）；R14 停滞成员跳车即跳过本列车切片顺延；字节口径沿用 v1 均匀终态上下文（I1 逐 shard 守恒：Σcredits ≡ S×f(终态)）；WP9 first_token 拆分与体块化正交（首步批=迭代 1、余量批沿 K 对齐）。**v1"批量读流+readiness barrier 硬栅栏"串行口径已删除**（2026-09-17 用户裁定：旧机制不作为开关可选项保留——`_joint_remote_read_stream` 方法删除、无 `JOINT_REMOTE_EXEC` 开关；回归锚改为 `JOINT_REMOTE_ACTIONS=off` 字节等价 + K≥S_j 单块结构等价）；计价（2026-09-25 起**分阶段关键路径**，规格书 §5）：`prefill_stage = 前缀首 credit + max(前缀余量流, 后缀池恢复, prefill 计算)`、`decode_stage = decode 首 credit + max(decode 余量流, decode 计算)`、`cost = target_wait + eviction_wait + prefill_stage + decode_stage + merge`——前缀读流与后缀池恢复自同一准入 frontier 并行分叉、prefill 计算按层段随数据到达推进，旧 `max(history_prep, eviction_wait) + first_credit + max(remaining_stream_ns, compute_ns)`（后缀恢复全量串在 prefill 计算之前）合成废除、无开关可选项；C5 冻结 breakdown 字段口径不动（`remote_read_first_credit_ns`/`remote_read_stream_ns` 转为合并流日志兼容披露、不进关键路径），阶段量经 notes 四键披露 `remote_read_prefill_ns`/`remote_read_decode_ns`/`suffix_restore_ns`/`prefill_pipeline_overlap_ns`（两腿与执行同 K 源、腿内仍 `首 credit + max(余量流, 计算)` 形态；**decode 相多列车重付首块流水填充的系统性低估偏差**沿用——n 决策时刻因果不可知、上界 ≈(n−1)×(链路时延×hops+切片/(M×B))，已登记 PROVENANCE §13）；计价基数含 input + 在线 decode 增长（N11，因果可见；执行侧终态上下文为后端真值不进决策）；**适用性 = LOCAL 基＋ PARTIAL 基混合形态（需求①，2026-09-17：读流层区间化 [0,p)——per_step 字节按前缀层精确派生、后缀池恢复计入 history_prep 段与空间足迹；`JOINT_REMOTE_READ_PARTIAL` 消融）；REMOTE 基仍拒**（见上"动作语义"；decode 相不变锚维持 `JOINT_REMOTE_ACTIONS=off` 字节等价 + K≥S_j 单块结构等价，prefill 相旧 pass 分支已按下文分阶段废除）；**prefill 期前缀读流分阶段（2026-09-25 规格书§一/§二/§三）**：准入相独立规划 `plan_prefill_remote_read_transfers`（纯规划零副作用，置于 reserve 之前；home 驻留前缀 [0,p) 按 RESTORE_GROUP_LAYERS `plan_layer_groups` 组切分、逐组 `noc_migrate` 瞬时流 `stream_only=True`——不 _add_local_shards、不改驻留指针/shard_bytes、不进 merge 工作副本账与容量账本），与后缀池恢复腿自同一准入 frontier 并行分叉（无全局 barrier）；图侧 `_emit_prefill_remote_read_branch` 旁挂分支：interval/arrival gate 结构性重建到本轮 prefill（exec）实例（R5 同语义）后经 `TransferTriggerGate(control=exec)` 走 `_emit_transfer_trigger` 同款 1B relay 触发链**恰消费一次**（exec timer→exec trigger send→home trigger recv→home KV send→exec KV recv），后续组经 home send 链/exec recv 链序连接、不逐组重复消费 gate；逐组 recv 完成门＋层区间入 `_prefill_remote_read_arms/_layers` 双账本（成对登记/成对消费，单边在场即 raise），列车体 `emit_layer_segmented` 按层段门控消费——[0,p) 前缀段等前缀 recv 门、[p,L) 后缀段等恢复写门，首步批恰一次消费/余量批不重复/列车尾与 completion 残留 fail-closed；LOCAL 基 = 全层 [0,L) 读流、history_transfers 恒空（旧"gate 从未被消费＋前缀层计算无数据门"的 pass 分支废除）；流生命周期 owner=`rid#prefill_read`（准入登记、prefill drain 释放、serial 单代表流折叠，与 #readplan/#decode#j 恰一 owner 在册）；单测=`online/test_remote_credit_stream.py`（含 PARTIAL 混合读流 3 用例：read_prefix 派生/层区间 KVTransfer/总量 I1）+ `joint/test_joint_credit_pricing.py`（8 用例：手算锚/退化锚/自适应 K/动作隔离）+ prefill 分阶段五件（2026-09-25）：`online/test_prefill_remote_read_manager.py`（16 用例：两腿分离/瞬时流纪律/LOCAL·REMOTE 基回归）+ `online/test_prefill_remote_read_graph.py`（25 用例：触发链/双账本/层段铺满/并行分叉偏序）+ `online/test_prefill_remote_read_lifecycle.py`（9 用例：decode 复用/#prefill_read·#readplan 生命周期恰一 owner）+ `online/test_prefill_remote_read_scheduler.py`（7 用例：runtime 字段/登记释放/回滚零残留）+ `joint/test_joint_remote_read_staged_path.py`（9 用例：分阶段关键路径手算锚） |
| merge 物理流 | 已实现 | 完成批发射真实回传/池写回流 + merge 尾标记节点（watch 送达 = merge_done 事件侧，决策日志独立披露行——到达锚在 service_done，尾标记转为到达后依赖门控，C3b）；下一轮数据准备经 store-tail 前递补偿等待（合并成本进入下一轮数据等待；全部动作/八组合一致处理） |
| 逐层恢复与 prefill 重叠 | **已实现＋单测验证＋真实路径对拍（2026-09-22 C15）** | 列车级保守门退役为**逐层段就绪门控**（`emit_layer_segmented`：热前缀段 + 各恢复组层段，段 i 只等组 i——跨段字节与整段单次发射逐位一致测试钉死）；restore_group 标记的后缀恢复腿组间并行发射（2026-09-25 P2-R2：组间无边互并发，每组只 arm 与本组层区间交集的 store 条目、每跨缘对恰一次共享 1B 中继、并集条目同批发射内消费——组 0 区间无匹配的 fail-closed raise 面不变；同实例 partial 分支 + 跨实例旁挂支链）、逐组逐 rank 目标 HBM 门；`rid#restore` 区间账本（issue/complete 守恒，drain/merge/mark_complete 三结算边界，双结算 fail-closed）；RESTORE_GROUP_LAYERS=8（≤8 层单组 = 旧单笔锚）；completion 残留门 fail-closed。真实路径对拍：stress 夹具 10s/270 请求 partial_copy_hits=78、restore 分解 270 行 hidden_ratio p50=0.9999（旧单笔口径保留为回归锚，非开关） |
| 链路遥测三件（C6–C8） | **已实现＋单测验证＋真 run 对拍** | C++ `--link-telemetry` 每 epoch 导出链路合计字节、活跃时间及时间加权活跃流数 `active_flows`；合计速率按活跃流数换算单流份额后供 JCM 计价、瓶颈与 AIMD 使用（字段全缺时兼容旧二进制口径，混合缺失拒绝）。JCM `divisor_effective` 将时域 `#readplan` credit 账本与实际并发代表流分开：共享链上先按真正可同时在场的流登记，再和遥测流数合并；SH 侧 LinkId→端点键换算 + `#readplan` 预登记对账核销。既有 2s 窗流数与注册数相等，分叉形态由单测钉死；旧对拍覆盖率 100%、字节覆盖 99.99987%、漏计方向单向微欠（0 过计）、divisor 最大相对偏差 6e-8 仅描述原对拍窗。 |
| 配额（通用流量治理） | **已实现＋单测验证＋冒烟四臂（2026-09-22 C9–C11）** | `JOINT_QUOTA_MODE=off\|static\|aimd`（缺省 off——F7：off 臂决策序列零漂移实证）；动作级准入（链路∪实例 HBM 端口、实例永不掩码、quota_deferred 回队 + wait_reason 三分）；AIMD 收缩/扩张（遥测驱动，膨胀阈值带 + 流寿命 EWMA）；流生命周期借还配对（admits=releases 全配对）；port_snapshot 实测值接通（off 态 NA 保持）；决策时延入档（30s 臂：纯决策段 avg 3.510ms/max 9.031ms；判据层 ≈7% 决策段）；矩阵 `TJE_quota_static`/`TJE_quota_aimd` 两臂常驻 |
| copy 块级交接（C13） | **已实现＋单测验证（23 用例）** | 前缀 NoC 迁移 K 对齐块化（`_cb{k}` 唯一相位键）+ 源端交接即释放（home 释放点前移）；尾块旁挂支链 2026-09-25 P1 起逐笔并行（块间无边、每块 fork 头块后主链 frontier，块内 ack 新形态 ack_recv_c←本块 send_c／ack_send_c←本块 recv_c，gate 逐块重 arm——旧 send 直连链/exec recv 顺序链/ack 链尾随的整段串行删除）；守恒科目 `#handoff`（交接重复量）/`#copy-stream`（一遍历史）入决策日志；`copy_handoff_events` 一行入 run 末 ledger 侧车（C11 落点） |
| remote 完整结算与 kv_delta_journal（C14） | **已实现＋单测验证（14 用例）** | 完成检查清单七项逐项"机制在案⇔测试钉死"（源端结构保护双通道/保护解除时机 = merge+mark_complete 同 tick/少并多按两侧实际保留量/失败分支闭合）；`kv_delta_journal` merge_back 两出口恰一行（方向互斥/零字节翻转/传输字节/两侧保留量/home 迁移轨迹）+ `_on_merge_done` 闭合审计门（watch 交付 ⇔ journal 行，缺行 fail-closed）；C4b-FIX 补 remote-read auto-K 多块 credit 命名缺陷（`_rcb{b}` 后缀，5 用例 + 两族臂 exit 0） |
| face_static 静态对照臂（C17） | **已实现＋单测验证（18 用例）** | `JOINT_SCHEDULER_MODE=face_static`（policy variant，不进八组合）：实例按 `hop ≤ floor(ρ)` 静态距离球掩码、球内逐实例择优；身份纪律（manifest `face_static_identity` 键：本文内部的静态距离变体、不冒称完整 FACE）；F7 唯一耦合例外——强制 `JOINT_QUOTA_MODE=off`（manifest 注记 `quota_forced_off`）；矩阵 `TJE_face_static` 臂常驻 |
| 域三口径度量（C16） | **已实现＋单测验证（现有 43 用例）** | `slo_tools/domain_metrics.py`：D_feed/D_econ/实际选择三口径分列（elected/forced 分列、measured=NA 如实标注）+ 派生指标（hop 分位/方向分布/瓶颈资源/内外等值线逐 (请求,实例) 导出）+ A3' δ 敏感性重定价（δ∈{σ̂,2σ̂,4σ̂} 离线诊断，σ̂ 三级链）+ home 迁移轨迹消费面；driver 五 sink 扇出、非 joint run 完全静默 |
| 负载区间与运行输入快照（2026-09-24） | 已实现＋回归验证 | `load_imbalance.py` 对 joint 请求以决策日志 `completion.tick` 为实际完成端点；`train_ledger.exits` 只核对成员与实例，其 tick 为列车发射时刻。历史非 joint run 继续输出 `legacy_train_ledger_emit` 代理端点并显式标注。运行入口把生效的 `trace_config.csv` 与 hardware JSON 归档到 run_dir；`domain_metrics.py` 优先显式参数，其次用 run 本地快照或旧命名副本推导 ρ，缺 run 本地证据时给 NA 与告警，不读取当前 checkout 猜测历史硬件。 |
| SLO 水印对 joint 的口径 | 已登记（joint 词表已扩，R12；**merge v2 重放已扩，2026-09-17**） | slo_tools 三工具已登记 astra-sim-joint；水印重放词表已扩 joint 专属事件（跨实例工作副本双驻留记账——SessionState base/working；决策行携带 joint_action/history_transfers/重算口径字段；完成行显式披露 `joint_working_copy` 真值——复审 K3）；**merge v2**：completion 行携带 `merge_direction` 走 v2 重放（forward=exec 工作副本搬 home／reverse=home 基础释放搬 exec 或零字节翻转／in_place=无主回暖就地转正；零池写断言——本会话 remote_store 即 fail；noc 源端与方向交叉校验；`joint_merge_forward/reverse/in_place/zero_byte_flip/home_migration` 计数；混合形态准入的池恢复后缀进期望工作副本基数），无该字段的旧日志走 legacy 分支（金样对照兼容）；**非账本披露行词表（2026-09-22 C19）**：`kv_eviction` 账本行之外，joint_admission 族 + `readplan_reconcile`/`readplan_settle`/`quota_oneshot_overflow`/`merge_done`/`link_telemetry_coverage`/`joint_decision_metrics` 披露行并入跳过词表（run 级行 request_id 为空不误触 fail-closed；13 臂矩阵 postprocess 全净实证）；决策日志重放仍为上界口径（逐 rank 判决需 kv_delta_journal 权威层——journal 已随 run 末 sidecar 序列化为第四键、`domain_metrics` 四层可信度分级已消费（settlement_full_join>partial>empty>decision_log_only；F3 修复批 2026-09-22），同 S3） |
| 容量逐出的决策日志披露（R17，2026-09-17） | 已实现＋单测验证 | 旁路逐出咽喉点（`_emit_eviction_only_nodes`）在图发射后同 tick 落 `kind=kv_eviction` 决策行（decision.evictions 携带 `_transfer_summary` 条目）——覆盖 decode 增长逐出（成功/停滞/唤醒三路）与准入失败已提交逐出（潜伏位点：当批零触发——reserve 的 feasible 预检拦截在先 + prepare 受 R1' 预约不变量覆盖，R17-7 探针实证）；`prefill_evictions` 为结构性恒空死通道（准入预约覆盖全动作足迹，drain expand gap≡0），注释+单测钉死不新增空字段载运；转移摘要补序列化 `resident_prefix_layers_before/after` 与 `source_instance_index`（连锁不变量免重建 + victim 归位，R17-1d）；tick 守护 = 断言 `_batch` 在场（N4）。工具侧配套：hbm_watermark 主循环 kind 门前拦截 kv_eviction（自担 tick 单调 + 逐条 `apply_evict` + 非空率哨兵）+ `_joint_prefill` fail-early 对称校验（prefill_shrink/discarded>0 fail-closed）+ (primary, base) 二元组前缀连锁不变量（R17-2/3/4）；hopbytes `collect_joint` 消费 kv_eviction 池写流（此前通道 2/3 系统性漏计）并移除 prefill_evictions 死读。历史 run 决策日志定格缺失、离线不可重建——水印三件套与可信 hopbytes 需带本修复重仿真（第三次错误处置，方案 v4） |
| 远端端口并发后端（2026-09-24 SerDes 片外链路并发化） | 已实现＋C++ 夹具验证＋固定输入 A/B 恒等（**零池负载平凡一致性**，不证争用正确性） | 串行 FIFO 端口整体替换为并发流体端口（§7-D'；`AnalyticalRemoteMemory.{hh,cc}`）＋发射门控改造（远端 MEM 独立无上限计数槽、与 comm 槽解耦——comm/hbm_dma 槽 2026-09-25 起同为计数制无上限（PROVENANCE §45）、static 结束门计入远端 MEM、`NDEBUG` 无关的 occupy/release fatal 校验、析构未释放诊断）。夹具锚：`RemotePortNwayTest` S1–S10（300｜90/120/140/150ns｜1B 流体 1/6ns）＋16 端口×128 流压力（2048/2048、peak_streaming=128、redistribution=2032=16×127）；`RemotePortOnlineGateTest`/`RemotePortStaticGateTest`（同 rank 双 MEM 发射后 in_flight=2、peak_streaming=2、shared_busy_ns>0；MEM 与 COMM_SEND 门互不占用；static finish 等全部终结）；`HbmNwayTest` 重锚（contention ON/OFF）。sensing 开启时逐事务明细 `remote_memory_transactions.jsonl` 惰性落 `bridge_dir/`（首个完成事务才建文件；成功后 runner 搬 `results/` 并入归档常驻白名单；写失败 fail-closed FATAL）。固定输入 A/B（快照基线树 vs 本树）：决策/发射序列与全系统指标除宿主计时字段外逐字节恒等——窗口内池后端完成事务数为 0，属零池负载平凡一致性；Python 计价面与 C++ 零耦合，无重标定触发。证据 `/tmp/serdes-joint-work/`（gate_rerun.log 49 步全 PASS、precision_rerun.log、r4_*.log、stress_results.txt、pricing-audit.md） |
| 三机制消融/外部对照实验 | 未运行 | 本次交付为仓库构建＋单元级验证；仿真规模实验按另行登记的有限计划进行（§8 第 6 步） |

## 4. 快速开始（构建与运行入口沿用基底）

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
# ② 物化输入（唯一允许源 = agent-traces/tracelab/astra_compute_20.csv
#    前 2 秒）+ ③ plan 物化 + ④ 运行（与基底同款）
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
   COMP/CPU 单槽门保留）；copy 交接尾块、恢复组与 remote-read credit
   尾块由逐笔串行链改逐笔并行支链／组间并行（首块准入与栅栏、逐层就绪
   门、账本语义不变——§3 状态表 C13/C15/remote-read 行，PROVENANCE
   §45）。
9. **prefill remote-read 分阶段（2026-09-25，零新增开关）**：remote-read
   轮 prefill 前缀读流（home→exec 逐层组瞬时流，准入相独立规划 +
   图侧旁挂分支 + 列车体层段门控）与后缀池恢复腿自同一准入 frontier
   并行分叉、无全局 barrier；decode 相 credit 读流只覆盖前缀 [0,p)
   （后缀准入相已恢复、就地复用）；准入计价改分阶段关键路径
   （prefill_stage + decode_stage，旧"后缀恢复全量前置"合成废除、无
   开关可选项）——§3 状态表 remote-read 行，PROVENANCE §46/§47。

## 6. 测试

```bash
cd sh_test_mesh
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
```

定向验收覆盖（`joint/test_joint_mechanisms.py` 对应《设计方案》§8 表）：
八组合固定映射与 fail-closed；T 严格序/满足即停/未知类别披露；E k_hide
解析例（L=32, c=1ms, q=0, r=0.5/2/4ms ⇒ 1/17/25）与三模式计划；J 三模式
选择与 remote-off 仅移除 remote 候选；home/merge（增量恰好一次、无双份
驻留、recompute/remote-read 工作副本口径）；八组合同实现配置切换。
2026-09-22 改造批新增机制测试文件（全部零后端）：
`joint/test_joint_shard_pricing.py`（C1 三腿 min/平价锚/读放大）、
`joint/test_hbm_port_flow_registry.py`（C2 u_port/rider）、
`online/test_joint_decode_context.py`（C3 上下文依赖）、
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
逐出尾 watch 四路径真实生产链钉测 15 用例，PROVENANCE §43）。

端到端冒烟（astra_compute_20.csv 前 2 秒，21 请求）：执行方案 §8.3 的
**13 配置矩阵全部 PASS**（八组合 #1–#8 + 辅助臂 #9 typed/affinity-first/
legacy_half、#10 typed/affinity-first/adaptive、#11 typed/affinity-first/
minimal_layer_groups、#12/#13 remote-off 正交臂）；行为矩阵：joint 组
选点 10/4/6/1 vs load-first 组 12/6/2/1（J 联合选择真实改变选点），
load-first/none 系真实触发 7 次跨实例 copy + 7 次增量 merge 回传流
（R15/K4 计价修复后决策序列快照，修复前为 6/6）；
affinity-first 恒 stay（home 优先与联合选择的最优重合）。**两处"13"口径
注记（2026-09-22 复审补）**：上文执行方案 §8.3 的 13 配置矩阵（含 #9–#11
调度器变体辅助臂）与下文 `joint_smoke_matrix.sh` 的 13 臂是**两个不同集合**
（交集 = 八组合 + remote-off 臂族）——前者服务调度器行为对照、后者为交付
一键回归矩阵，引用时勿混用。**冒烟矩阵已
脚本化**：`sh_test_mesh/run_scripts/joint_smoke_matrix.sh`（2s 窗
**13 臂**，C19 起一键全绿）——八组合 `combo_*` + `TJE_remote_off` +
`TJE_shadow_verify`（影子验证）+ C19 三新臂：`TJE_quota_static`
（JOINT_QUOTA_MODE=static）/ `TJE_quota_aimd`（=aimd，F7 自动注入链
全程生效——runner 置位 SH_LINK_TELEMETRY → .sh 追加 --link-telemetry →
C++ 遥测开）/ `TJE_face_static`（--scheduler face_static 对照臂，quota
强制 off 入 manifest 注记 quota_forced_off）；后三臂经 `joint_runner.py`
CLI 面起跑。产物落仓外持久目录 `/home/sunhao/joint_smoke_evidence/
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

## 7. 硬件概念（沿用 sh_3.0 口径；**§7-D' 远端端口并发模型为本仓
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
  不建立封装带宽耦合；③ Python 决策侧池计价 `_pool_transfer_ns` 仍是
  发射时点快照除数下的"latency + 字节/有效带宽"串行公式估计，与 C++
  流体服务不同口径，不能当作端口并发服务时间的预言；④ 固定输入 2s 冒烟
  窗内池后端完成事务数为 0，基线/改造两树全系统输出恒等属零池负载下的
  平凡一致性，不构成争用条件下并发模型的正确性或收益证据（争用行为
  证据目前仅有后端级确定性夹具与压力夹具，见 §6）；⑤ 本仓不设
  serial/parallel 行为开关，旧串行路径已删除、无开关可回退。

**E. 实例（instance）**：紧密相邻芯粒的连续实心矩形，TP whole-head 分
片；全部实例铺满晶圆；布局唯一来源 `trace_config.csv` 的 `inference_
group` 行。本仓为统一实例（P+D 同实例）；一 session 的权威 KV 基础历史
驻留单一 home 实例（执行副本另计）。

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

## 8. 边界与纪律

- 仿真输入唯一允许源与窗口遵循 Agents.md 冒烟规范。
- 缺失输入 fail-closed；deep-gap 逐无可逐仍不满足按 rank 记账后
  fail-closed（沿用 D4）。
- 决策输入无 oracle：CSV 未来行/未来输出长度/真实返回时间不进任何
  决策；完成时已观测长度仅用于实际结算与在线估计器更新。
- E 的 adaptive 公式与因果规则已冻结（§5）；不做逐 workload 拟合。
- 本仓不运行三机制消融/外部对照实验（§8 第 6 步另行登记）。

[ASTRA-sim](https://astra-sim.github.io/) is a distributed AI system simulator.
It models the end-to-end software and hardware stack of modern AI systems.
For our website and documentation, see [astra-sim.github.io](https://astra-sim.github.io/).
