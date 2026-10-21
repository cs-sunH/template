# request 映射和历史 KV 管理（大白话版）

> 下面写的路径都相对 `sh_test_mesh/workload/llama2_7b_inference/`，位置只精确到文件。姊妹篇《request映射策略与历史kv管理机制介绍.md》是同一套分支的详细版，那份偏全偏细，这份讲人话。

## 一、系统怎么搭起来的

这套代码是在 ASTRA-sim 上改出来的 LLM 推理模拟器，分四层，活一层层往下交。

最上面是 trace 层（`plan_materializer.py`、`generate_face_trace.py`）。它从 CSV 读请求，每条请求就是"哪个会话、第几轮、输入几个 token、要生成几个 token"。这一层只管按时把请求吐出来，再定好实例编队（一个实例占哪些 rank）。请求该去哪个实例，这层不管。

第二层是调度层（`online/sh30_online_scheduler.py` 里的 Sh30OnlineScheduler）。每个请求要过它两道关。第一道是准入，代码在 `_try_admit_request`：挑一个实例、挑一个动作，把显存空间和要搬的数据都安排好。第二道在 prefill 算完的边界上，代码在 `_on_prefill_drain`：决定 decode 阶段是留在原地还是换地方，这一步提交完位置就定死，后面不再换。怎么挑的规则在 `joint/joint_scheduler.py`，每个选项值多少钱在 `joint/joint_cost_model.py` 里算：给每个"实例 × 动作"组合排一张工作图，把事件往前推，算出照这个方案干下去大概什么时候能全部干完（连收尾都算上），谁快挑谁。

第三层是 KV 管理层（`face_scheduler.py` 里的 KVCacheManager）。它记着每个会话的历史缓存现在在哪、有多大：全在某个实例的显存里？一部分层被挪到片外池子了？还是全在池子里？空间不够它负责赶人，历史要搬它负责发传输，请求算完它负责把散在各处的副本收成一份。旁边还有一个位置目录（`joint/kv_location_directory.py`）：每段 KV——哪个会话、第几层到第几层、哪些 token、在哪个 rank——都登记在案，状态只能按规定的方式变。

最底下是执行层（`online/graph_batch_builder.py` 加 `astra-sim/astra-sim/system/` 的 C++ 核）。调度层的决定在这里被翻译成每个 rank 上的计算图节点：拷贝怎么一块一块传，都变成具体的图，交给模拟器跑。

四层串起来就是：trace 吐请求 → 调度层看着 KV 的状态挑实例和动作 → KV 管理层照着决定挪数据、腾空间 → 执行层算 → 算完 KV 管理层收尾合并，KV 状态变了，又反过来影响下一个请求怎么挑。

## 二、几个先说清的词

- **实例（instance）**：一组卡。请求被派到一个实例上算。
- **会话的历史 KV**：这个会话以前几轮算出来的上下文缓存。下一轮来了能接着用。
- **KV 的位置四态**：`local_hbm`（全在某个实例显存）、`partial_hbm_remote`（前若干层在显存、后面的层被赶到片外池）、`remote_memory`（全在池里）、`none`（新会话，啥都没有）。
- **两个动作**：stay（历史就在这，原地用）、copy（把历史搬过来）。
- **两个决策点**：P（请求进来准入时选一次）、D（prefill 做完换 decode 前再选一次）。D 提交后不再选第三次。
- **版本号（epoch）**：一堆计数器。KV 每变一次加一，实例队列每变一次加一，配额每动一次加一。调度器拿"上次失败时的版本号"跟现在的比，没变就不浪费时间重试。
- **报错停掉**：代码里叫 fail-closed。意思是发现状态不对劲就直接抛异常终止整个 run，不猜、不硬跑。区别于"暂时进不去、排队等下次"。

---

## 三、request 怎么映射到实例

### 3.1 开关们（决定有哪些分支可走）

配置都从一个口进：`parse_joint_config`（`joint/joint_config.py`）。改实验只改环境变量，不改代码。八组合预设 `COMBO_PRESETS`（`joint_config.py`）把 (类别模式, 调度模式, 层数策略) 绑成八档。

| 开关 | 取值 | 干什么 | 位置 |
|---|---|---|---|
| `JOINT_ABLATION_COMBO` | 八组合名 | 一键设三件套；和单独开关同时设了就直接报错 | `joint_config.py`、冲突检查 `:668` |
| `JOINT_CATEGORY_MODE`（T） | typed / lru | 赶人时先赶哪类会话（见 4.4） | `joint_config.py` |
| `JOINT_SCHEDULER_MODE`（J） | joint / load-first / affinity-first / face_static | 选实例的规则（见 3.3） | `joint_config.py` |
| `JOINT_LAYER_POLICY`（E） | adaptive / legacy_half / minimal_layer_groups | 一次赶多少层（见 4.4） | `joint_config.py` |
| `JOINT_ALLOWED_ACTIONS` | all / 逗号分隔的二动作子集 | 动作白名单，是个上限不是强制要求 | `_parse_allowed_actions`，`joint_config.py` |
| `JOINT_QUOTA_MODE` | off / static / aimd | 链路配额开关和调法（见 3.12） | `joint_config.py` |
| `JOINT_BOUNDARY_POLICY` | off / p-continuation | off 是正常两阶段；p-continuation 是"只选一次"的对照组（见 3.8） | `joint_config.py`、检查 `:402` |
| `SH_TRAIN_MAX_ITER` | 缺省 8 | 训练类会话的迭代上限；显式给 ≤0 直接报错 | `parse_train_max_iter`，`joint_config.py` |
| face_static 强制配额关 | — | 选了 face_static 就把配额覆盖成 off（这是唯一的强制覆盖，会记一笔） | `joint_config.py` |

预测侧还有一组独立起作用的配置 `JOINT_PREDICTION_CONFIG`（`joint/prediction_config.py`），五个轴：projection_mode（正式 / 冻结诊断对照组）、future_demand_mode（none / recent / reverting）、history_use（shadow / validated）、stage_policy（two_stage / once_mapping，跟 `JOINT_BOUNDARY_POLICY` 是同一个东西的两个入口，互相矛盾就报错）、settlement_policy（consolidate / distributed）。预设文件在 `sh_test_mesh/run_scripts/prediction_presets/` 下。

### 3.2 P 入口：两个动作各自能不能用

判断全在 `JointCostModel.applicable_actions`（`joint/joint_cost_model.py`）。注意：某个动作不能用，不等于这个实例被踢出局——实例永远全量参与，只是个别动作被划掉。

| 分支 | 条件 | 结果 | 位置 |
|---|---|---|---|
| stay 能用 | 新会话（history=0），或者历史就驻留在目标实例上（local 或 partial 都行） | 记为候选 | `joint_cost_model.py` |
| stay 不能用 | 上面条件不满足 | 理由记 "history not resident at target" | 同上 |
| copy 能用 | 有历史可搬（location 不是 none、history>0），而且历史没驻留在目标实例上；另外 distributed 分裂形态下 resident=None 且 home=目标且 local 的那种不算 | 记为候选 | `joint_cost_model.py` |
| copy 不能用 | 没历史可搬。例外：REMOTE 基（全在池里）× copy@home 反而算可用——就地转正，不用真搬 | 理由记 "no history to copy" | 同上 |
| 白名单尾部过滤 | 设了 `JOINT_ALLOWED_ACTIONS` 且动作不在名单里且本来可用 | 统一改判 "action excluded by mask" | `joint_cost_model.py` |

### 3.3 P 入口：四种模式怎么选实例

入口 `select_instance_and_action`（`joint/joint_scheduler.py`）。先把所有实例 × 两个动作全算一遍价（只读，不动任何状态），然后按模式挑。

| 分支 | 条件 | 干什么 | 位置 |
|---|---|---|---|
| 模式名不认识 | mode 不在四种里 | 直接报错 | `joint_scheduler.py` |
| 没有负载视图 | loads 是空的 | 直接报错 | `joint_scheduler.py` |
| **joint 主模式** | mode=joint | 在全部可用候选里挑成本最小的；成本一样按（实例编号, 动作顺序）分先后。一个可用候选都没有（包括被白名单划光的）就报错——这是协议出问题了，不是空间不够 | `joint_scheduler.py` |
| **load-first** | mode=load-first | 不管别的，先按（任务负载, 编号）挑最闲的实例，再在这个实例里挑最便宜的动作。没有任何屏蔽 | `joint_scheduler.py` |
| **affinity-first 有家可回** | home 在，且位置是 local / partial / remote | 直接选 home（是偏好，不是硬性屏蔽） | `joint_scheduler.py` |
| affinity-first 没家可回 | 没历史或 home 不明 | 退回 load-first 的挑法 | `joint_scheduler.py` |
| **face_static 没锚点** | 新会话，resident 和 home 都是 None | "球"没中心，等于没有距离限制，所有实例都参与，记一笔 `ball_vacuous=True`，不报错 | `joint_scheduler.py` |
| face_static 有锚点 | 锚 = resident 优先、home 兜底 | 算半径：ρ = 片间带宽/片内带宽（`_static_ball_radius`，`joint_scheduler.py`），半径 = floor(ρ)，基础配置 4050/1640≈2.47 → 半径 2。只留跳数 ≤ 半径的实例；球里每个实例先挑自己的最优动作，再按（成本, 顺序键）挑实例；被球挡在外面的实例会记进 `masked_out_instances` | `joint_scheduler.py` |
| face_static ρ<1 | 半径算出来是 0 | 球里只剩锚点自己这一个实例，如实记录，不保底 | `joint_scheduler.py` |
| face_static 球是空的 | 屏蔽后一个实例都不剩（锚点索引过期才会发生） | 直接报错 | `joint_scheduler.py` |
| face_static 球里全被划光 | 白名单把球内所有实例的动作全划掉了 | 直接报错（白名单是上限不是强制，这里不能静默换选择） | `joint_scheduler.py` |
| 顺序模式挑完实例没动作可用 | load-first / affinity-first 选定的实例上白名单把动作划光了 | 直接报错 | `joint_scheduler.py` |
| 平局 | 成本相等 | 固定按（实例编号, 动作顺序），事先定死，不因结果调整 | `joint_scheduler.py`、`:254-258` |

### 3.4 选完之后：准入这一整套（`_try_admit_request`）

位置 `online/sh30_online_scheduler.py`。预约和准备历史是一个事务：要么全成，要么失败后什么都不留。

| 分支 | 条件 | 干什么 | 位置 |
|---|---|---|---|
| 到达时间缺失 | estimated_arrival_ns 是 None | 直接报错 | `sh30_online_scheduler.py` |
| **distributed 上一轮没收完尾** | 结算策略是 distributed，且这个会话上一轮的结算记录还在 | 请求照常到达照常排队，但不消费：留队等结算闭合（闭合时会加 KV 版本号，重开准入）。记日志 reason="settlement_open_dependency_gate"，清掉旧失败键，返回 False | `sh30_online_scheduler.py` |
| 算候选的价 | — | 拼会话 KV 只读视图 + 请求视图（不含任何"未卜先知"字段；decode 还要几步用条件剩余估计器估），跑 `select_instance_and_action` | `sh30_online_scheduler.py` |
| **配额过滤** | 配额开了（`_quota_filter_candidates`） | 逐候选过一遍：stay 永远行；copy 查链路 + merge 大块写名额（判在一个候选一个候选地做，`_quota_candidate_verdict`） | `sh30_online_scheduler.py` |
| **配额全不过、排队** | 所有动作都被配额卡住 | 记失败日志（quota_deferred，wait_reason 分链路/端口两种），记下开始等的时间，失败键 = KV 版本 ⊕ 候选实例版本 ⊕ 配额代数，返回 False。注意这里**不许**报错——就是排队 | `sh30_online_scheduler.py`；排队语义 `joint/link_quota.py`（永不抛异常） |
| 配额过了、重新挑 | 有活下来的候选 | joint 模式按原顺序挑最小；顺序模式在活下来的集合里重跑选点规则（`_quota_reselect_feasible`，`sh30_online_scheduler.py`） | `sh30_online_scheduler.py` |
| **强制探针** | `JOINT_FORCE_ENTRY_ACTION/_TARGET` 命中这个请求 | 选择器算完后强制指定最终选择；强制出来的动作不在可用面里就直接报错；和边界探针互斥 | `_apply_forced_entry_choice`，`sh30_online_scheduler.py` |
| **空间不够、回退** | reserve/prepare 抛容量错 | 已赶的人照常入账 → 释放孤儿预约 → 记 capacity_deferred → 冻结失败键（候选 ∪ 选中实例的版本号）→ 返回 False。另做三态判断（`_classify_physical_feasibility`）：还有组合物理上放得下 = 暂时不行；全都放不下 = 结构性问题，报错 | `sh30_online_scheduler.py` |
| 事务成功 | — | 加版本号、同步图；runtime 记下 prefill 实例/动作/阶段/出发地/传输/赶人清单 | `sh30_online_scheduler.py` |
| 登记传输流 | — | 主链 owner=请求号，赶人支链 owner=请求号#evict，merge 预留 owner=请求号#merge-reserve | `sh30_online_scheduler.py` |
| **配额登记失败的防御** | 主流程登记配额返回 False（判断和登记之间外部变了） | 回滚所有在途流承诺 + 释放预约（还没物化就释放并回队；已物化直接报错——没有反向 prepare）→ 返回 False | `sh30_online_scheduler.py` |
| 排队时间采样 | 这个请求之前排过队 | 记一条"从开始等到进来的时间"样本 | `sh30_online_scheduler.py` |
| 成功收尾 | — | 落决策日志（逐候选明细）→ 存水位预测 → 入实例队列 → 加实例版本号 → 发准入事件 → 返回 True | `sh30_online_scheduler.py` |

### 3.5 没进去的请求什么时候再试（`_admit_waiting_requests`）

位置 `sh30_online_scheduler.py`。失败键 = （KV 版本, 一串(实例, 实例版本), 配额代数）。实例集合冻结在失败那一刻，是"选中的 ∪ 当时可用的候选"，不许扩大成全部实例（版本号是高频信号，全挂会让重试判断永远在翻转）。

| 分支 | 条件 | 干什么 | 位置 |
|---|---|---|---|
| 键没变、跳过 | 上次失败键 == 现在的键 | 重试必然还是同样的失败，不白跑；排队位置不动 | `sh30_online_scheduler.py` |
| 影子校验 | `SH_ADMIT_GATE_VERIFY` 开着且键没变还硬跑了一次 | 如果居然成功了，说明跳过逻辑有 bug，报错 | `sh30_online_scheduler.py` |
| 键变了、正常重试 | 容量释放了/负载挪了/配额代数动了 | 重跑 `_try_admit_request`，失败就记新键 | `sh30_online_scheduler.py` |
| 失败键缺失 | 异常路径没留下键 | 兜底用（当前 KV 版本, 实例 0 的版本），等于"多试一次" | `sh30_online_scheduler.py` |

### 3.6 D 边界：两个动作各自能不能用

入口 `applicable_D_remaining_actions`（`joint/joint_cost_model.py`）。判断的依据是"目标实例还缺哪些区间"，单一来源 `BoundaryKVLayout.missing_intervals_for`（`joint/boundary_layout.py`：目标已经有的不算缺；正在往目标搬的不重复算）。边界布局恒是完整型这一族：完整历史全在 P 实例上。

| 分支 | 条件 | 结果 | 位置 |
|---|---|---|---|
| stay | 目标就是 P 实例，原地继续 | 可用 | `joint_cost_model.py` |
| copy | 目标缺的区间非空 | 可用。缺什么补什么：别的实例补整份基础。目标啥都不缺就划掉（"no history to copy"） | `joint_cost_model.py` |
| 白名单过滤 | 同 P 入口 | 划掉 | `joint_cost_model.py` |

### 3.7 D 边界：怎么选（`select_decode_target_and_action`）

位置 `joint/joint_scheduler.py`。这一步的规则是协议级的，四种调度模式都一样。"原地继续"不是第四个动作，它就记在 (P 实例, 默认延续动作) 那条候选上——延续动作恒是 stay。

| 分支 | 条件 | 干什么 | 位置 |
|---|---|---|---|
| 一个可用候选都没有 | 且延续动作被白名单划掉了 | 直接报错，归因是白名单 | `joint_scheduler.py` |
| 一个可用候选都没有 | 且延续动作没被划 | 说明快照和候选对不上，结构性问题，直接报错 | `joint_scheduler.py` |
| 延续被划掉但它还在候选里 | 白名单过滤失效 | 直接报错 | `joint_scheduler.py` |
| **延续被划掉时的正常选择** | continuation_masked | 在允许的动作里挑最小，平局键退化成（成本, 实例, 动作顺序），日志记 `continuation_masked_out=True` | `joint_scheduler.py` |
| 遍历完没有延续记录 | 延续没被划但记录缺席 | 直接报错 | `joint_scheduler.py` |
| **正常选择** | 有延续记录 | 挑（成本, 是否非延续, 实例, 动作顺序）最小的——**成本相同优先原地继续，只有别的方案严格更便宜才搬家** | `joint_scheduler.py` |

### 3.8 D 边界提交（`_on_prefill_drain`）

位置 `online/sh30_online_scheduler.py`。这是第二个也是最后一个决策点的提交现场。

| 分支 | 条件 | 干什么 | 位置 |
|---|---|---|---|
| 重复完成通知 | prefill_complete_ns 已经写过（重复的 P_COMPUTE_DONE） | 落一行 boundary_duplicate_ignored 直接返回，什么账都不动 | `sh30_online_scheduler.py` |
| 边界前清理 | — | 释放入态的流（主链）、配额结算准入段、把 prefill 工作副本推进 | `sh30_online_scheduler.py` |
| **已知要终止** | decode_length==0 之类 | 不做 D 选点，直接结算走人 | `sh30_online_scheduler.py` |
| **一次映射对照组** | boundary_policy=p-continuation | 不做第二次全候选重估：decode 实例 = prefill 实例，动作 = 默认延续动作，候选列表空、成本记 NA | `sh30_online_scheduler.py` |
| 两阶段（缺省） | boundary_policy=off | 边界成本模型 + `select_decode_target_and_action` + 强制探针（`_apply_forced_boundary_choice`，`sh30_online_scheduler.py`，语义同 P 入口的探针） | `sh30_online_scheduler.py` |
| **提交前版本复查** | 决策到提交之间 KV 版本变了 | 抛漂移异常，走和容量失败一样的回退结构 | `sh30_online_scheduler.py` |
| 跨阶段预约事务 | — | `commit_stage_reservation_transaction`（换地址/改大小/释放，一个事务办完）+ `commit_boundary_decode_transition`（搬工作副本，见 3.9） | `sh30_online_scheduler.py` |
| **提交失败回退** | 抛容量错 / 阶段版本冲突 / 漂移 | 要是选的就是原地继续（继续方案自己失败），直接上抛报错；否则撤掉换址那半边 → 已提交的赶人照常入账 → 改成"原地继续"重新提交 → 落 boundary_commit_fallback 行（failure_class 分 kv_capacity_error / stage_version_conflict） | `sh30_online_scheduler.py` |
| 回退后的记账 | boundary_fallback | 动作记成延续动作，is_continuation=True；失败方案的水位预测不存（不参与对账） | `sh30_online_scheduler.py` |
| **D_COMMITTED** | 第一笔不可撤销的新阶段操作（预约换址）发生 | 从这以后本轮不再选点；把所有来源在目录里加引用（执行期钉住来源不许动） | `sh30_online_scheduler.py` |
| 阶段引用闭合 | — | 释放阶段引用；预约注销 | `sh30_online_scheduler.py` |
| 异常兜底 | 边界窗口内任何异常 | 还没到 D_COMMITTED 且事务在窗口内就撤销 → 从就绪队列移出 → 释放阶段引用 → 把原异常抛出去 | `sh30_online_scheduler.py` |

### 3.9 D 边界上数据怎么搬（`commit_boundary_decode_transition`）

位置 `face_scheduler.py`。选完之后真正动数据的地方。

| 分支 | 条件 | 干什么 | 位置 |
|---|---|---|---|
| stay / copy | 动作是 stay 或 copy | 走 `move_prefill_to_decode`：stay 零字节（kind="stay_local_hit"）；copy 把整份基础搬过去（kind="complete_copy_migration"） | `face_scheduler.py` |
| **copy 落到基所在实例、就地并基** | copy 且目标 == 基实例且基有片上前缀 | 退化成就地合并：迁移照发但落点就是基那侧；恢复账已清或重复的基础副本就地释放（kind="complete_copy_base_fold"） | `face_scheduler.py` |
| copy 去第三目标 | 目标不是基所在实例 | 增量搬过去 + 基础再复制一份，几条腿都记进账 | `face_scheduler.py` |

### 3.10 P 阶段准备历史（`prepare_prefill`）

位置 `face_scheduler.py`。准入后按选的动作把历史数据摆好。下面"基形态"指会话记录里历史现在的样子。

| 分支 | 条件 | 干什么 | 位置 |
|---|---|---|---|
| 动作名不认识 | 不是两个之一 | 直接报错 | `face_scheduler.py` |
| **新会话建档** | 会话不在记录里（此时 history 必须是 0，否则报错） | 建会话、定 home（以后换地方执行、被赶走都不改 home），直接返回，没有传输 | `face_scheduler.py` |
| **distributed 布局** | 会话是 distributed_layout | 走 `_prepare_prefill_distributed`：按目录条目找来源和做池恢复，锚点用目录里的基锚（`face_scheduler.py`） | `face_scheduler.py` |
| 元数据对不上 | 上下文/总量/分片和请求不符，或 home 丢了 | 报错（记录完整性问题） | `face_scheduler.py` |
| **stay × LOCAL** | 驻留实例就是执行实例 | 记一条零字节的本地命中；实例不符报错 | `face_scheduler.py` |
| **stay × PARTIAL** | 驻留实例就是执行实例 | 缺的后面几层 [p,L) 先腾地方再从池恢复 → 变成 LOCAL 全层 | `face_scheduler.py` |
| stay × 没驻留 | 位置是 remote 或 none | 报错（stay 要求历史驻留） | `face_scheduler.py` |
| **copy@home × LOCAL**（防退化） | copy 但基就在执行实例且是 LOCAL | 等价 stay 本地命中，不建第二份副本 | `face_scheduler.py` |
| **copy@home × PARTIAL** | copy 但基就在执行实例且是 PARTIAL | 照 stay-PARTIAL 办：只恢复缺的后缀，working_kind 保持空 | `face_scheduler.py` |
| **REMOTE 基 / 前缀为 0** | 基全在池里或前缀层是 0 | copy：整份从池恢复成工作副本，home 改成执行实例，working_kind=copy | `face_scheduler.py` |
| **copy（基在 home 侧）** | 基驻留在 home，目标在另一边 | 前缀 [0,p) 切成层块逐块交接（块 0 走准入主链、后面的块挂在旁路上流水发）+ PARTIAL 缺的后缀从池恢复 + 工作副本入账 | `face_scheduler.py` |

### 3.11 decode 期间长个子（`_joint_grow_decode`）

位置 `sh30_online_scheduler.py`。decode 不按最终长度一次占满空间，而是列车消费到哪、写到哪，才给哪准备资源。

| 分支 | 条件 | 干什么 | 位置 |
|---|---|---|---|
| 不用长 | 会话不在或上下文已到头 | 直接返回 | `sh30_online_scheduler.py` |
| 长成功了 | expand_decode 通过 | 加 KV 版本号；有赶人就进图、登记流 | `sh30_online_scheduler.py` |
| **空间不够、停住** | expand_decode 抛容量错 | 已赶的人先入账，然后这个 decode 进入停滞状态：后面的列车先不带它玩，等版本号变了再叫醒（`_wake_stalled_decodes`，`sh30_online_scheduler.py`）。等待时间照实算进端到端 | `sh30_online_scheduler.py` |
| 全停了死锁 | 所有 decode 都停了且没有别的事件 | 直接报错（宁可带着诊断信息死） | `sh30_online_scheduler.py` |

### 3.12 配额和 AIMD（`joint/link_quota.py`）

配额管的是动作，不是实例：哪个动作差资源就把它标成不可用（写明缺什么、差多少），实例本身不出局。资源就两样：有向链路、实例的 HBM 端口。

流只有一类（`link_quota.py`）：在册流全是 oneshot。

| 类 | 是什么流 | 怎么管 | 位置 |
|---|---|---|---|
| oneshot | copy / merge / 赶人写回 / 池恢复这种一次性的 | 全额计价，链路占用和预留都算，端口也计数 | `link_quota.py` |

另外两条：链路初始配额 `Q_init = max(1, floor(ρ))`（`derive_q_init`，`link_quota.py`），max(1,·) 只是防死锁；`reachable_domain`（`link_quota.py`）算"顺着还有余量的链路能到哪些实例"，只拿来分析和记日志，不裁候选。

merge 的借还配对：链路侧给两个候选方向各留 1 个槽（owner=`rid#merge-reserve`），算完的时候放掉输的那边（`adjudicate_merge_direction`），赢的那边等 merge 真做完再放；端口侧赢家各占 1 个大块写名额，上限 N_bulk=Q_init，超了就排队（wait_reason=quota_port）（`link_quota.py`）。

AIMD（只在 mode=aimd 下动链路 Q；static 下 Q 永远等于 Q_init）：拿实测速率跟每流需求速率 r_KV 比，小于就收缩、在 [r_KV, 1.2·r_KV) 之间是死区不动、≥1.2·r_KV 算舒适（`aimd_band_signal`，`link_quota.py`；两个触发区间不重叠，避免在阈值边上抖）。八个动作：

| 动作 | 条件 | 干什么 | 位置 |
|---|---|---|---|
| shrink | 速率 < r_KV | Q 减半（下限 1）；在册的流不动（余量为负期间只是不放新的） | `link_quota.py`、`:222-230` |
| shrink_floor | 已经在下限 1 又来收缩信号 | 只记个事件，不动 | `link_quota.py` |
| hold | 死区 | 连续舒适计数清零 | `link_quota.py` |
| comfort | 舒适 | 计数累计中 | `link_quota.py` |
| comfort_cold_start | 舒适但还没见过流寿命样本 | 不敢扩张（等多久才扩张算不出来） | `link_quota.py` |
| comfort_frozen | 舒适但调用方说这段时间别扩（r̂ 退回窗口） | 照常判舒适，计数清空，不评估扩张 | `link_quota.py` |
| expand | 连续舒适满 `10 × 流寿命滑动平均`（α=1/8） | Q 加 1，计数重新开始 | `link_quota.py`、`:370` |
| expand_capped | 到上界 `max(1, floor(链路带宽/r_KV))` | 计数清零不动（再扩必然马上又收缩，来回抖） | `link_quota.py`、`:370` |

---

## 四、历史 KV 怎么管

### 4.1 调度器看到的 KV 是什么样（`_joint_session_view`）

位置 `sh30_online_scheduler.py`。P 计价用的会话视图，只读。

| 分支 | 条件 | 看到什么 | 位置 |
|---|---|---|---|
| 没这个会话 | 新会话 | 位置 none，字节数全零 | `sh30_online_scheduler.py` |
| **distributed 布局：看目录的并集** | distributed_layout | 锚 = 从 0 开始覆盖最大的片上条目。锚把并集全盖住了 → local@锚；锚缺的层在池里 → partial@锚；锚缺的部分在其他实例上（跨轮留存）→ 位置记 local 但 resident=None（stay 不可用，copy 按锚的差值算） | `sh30_online_scheduler.py` |
| 正在跨实例执行中 | working_kind 不是空 | 视图描述的是**基础历史**（base_* 那组快照；base 位置是空字符串 = 记录坏了，报错；None 按池处理） | `sh30_online_scheduler.py` |
| 平常状态 | — | 快照直接用：local/partial 字节按层区间推，缺的 = 总量 − 本地 | `sh30_online_scheduler.py` |

离线观测口径（`sh_test_mesh/slo_tools/kv_cache_adapter.py` 的 `_hit_state_joint`）：上一轮 history=0 → no_history；stay/copy → full（搬过去了就算命中）；字段缺失或值不认识 → not_supported（不瞎猜）；类型不对直接报错。

### 4.2 留空间（预约）

| 分支 | 条件 | 干什么 | 位置 |
|---|---|---|---|
| **按动作定预约量** | — | 两个动作都按"历史+输入"整份留（decode 的增长是写到才要，不预约；PARTIAL 跨实例的后缀另按字节留）。免得留出根本用不上的"幻影空间" | `face_scheduler.py` |
| 目标放不下 | 该实例被判定不可行 | 抛容量错（这种可以转成"排队等下次"） | `face_scheduler.py` |
| PARTIAL 不构成偏袒 | joint 规则 | PARTIAL 驻留不影响实例能不能进候选，只影响要多少字节（要的 = 总量 − 已驻留） | `face_scheduler.py` |

### 4.3 目录：每段 KV 的登记（`joint/kv_location_directory.py`）

每条目录记录一段 KV（键 = 会话 + 模型版本 + 层区间 + token 区间 + rank），有五种状态，只能按这个表变，表外的变化直接报错（`kv_location_directory.py`）：

```
WRITING（正在写入）→ READY（好了能用）或 RELEASED（已释放）
READY → EVICTED_TO_POOL（赶到池里）或 RELEASED
EVICTED_TO_POOL → RESTORING（正在从池往回搬）或 RELEASED
RESTORING → READY / EVICTED_TO_POOL / RELEASED
RELEASED → 终态，不再出现在活跃表里
```

| 分支 | 条件 | 干什么 | 位置 |
|---|---|---|---|
| 重复登记 | 同（内容, 持有者）已经有一条 | 直接报错 | `kv_location_directory.py` |
| 加/放引用 | 执行要用某段数据时 | 加引用或加执行钉住（pin）。引用 >0 或被钉住的条目不许赶；钉住期间裸放引用、放多了、越权解钉，都报错 | `kv_location_directory.py` |
| **赶进池** | READY 的条目要腾 | 变 EVICTED_TO_POOL：持有者翻成池、角色换成池背书、代数加 1、原来是谁的记录留着（审计用）。引用 >0 时赶，报错 | `kv_location_directory.py` |
| 从池往回发 | 池里的条目要用 | 变 RESTORING@目标。不是池里的报错；目标已有同内容副本也报错 | `kv_location_directory.py` |
| 在途拷贝去重 | 同（源, 目标, 区间）已经在搬 | 不重发 | `kv_location_directory.py` |
| **缺什么查询** | query_missing_intervals(会话, 目标) | 按 token×层 的小格算并集去重：目标已有或已在途的格不再搬；同一内容有多个副本时按固定顺序（base → boundary-source → resident → working → pool）只读一份；部分重叠只补差的那截（字节按面积线性折算，除不尽报错） | `kv_location_directory.py`、差集 `:646` |
| 片上来源查询 | query_sources | 只列片上有字节的（池里的不算，池恢复另查）；READY/WRITING 都算 | `kv_location_directory.py` |
| 最后一份保护 | 某内容只剩一条有效记录 | 受保护，不许赶到零副本 | `kv_location_directory.py` |
| 相邻条目合并 | 同会话/持有者/角色/层区间、token 正好挨着、引用为零 | 只合并记录，不白送物理搬运 | `kv_location_directory.py` |
| 容量投影 | capacity_projection(实例) | 只数片上状态（WRITING/READY/RESTORING）逐 rank 直加；池里的不占实例容量 | `kv_location_directory.py` |
| 有效保留量 V | effective_resident_bytes | 同一内容按格子并集只算一次，重复副本不给 V 添数 | `kv_location_directory.py` |
| 镜像同步 | sync_entries | 按差量改写/退休/登记；非法状态跳变报错；目标是 RELEASED 的用缺席表达（写墓碑报错） | `kv_location_directory.py` |
| 守恒检查 | assert_conservation | 五条不变量全查：唯一性、并集只计一次、占容量的算容量、引用进出相等、赶走不清 owner | `kv_location_directory.py` |

### 4.4 赶人腾地方（eviction）

总入口 `_ensure_capacity`（`face_scheduler.py`）。每个 rank 的缺口 = max(0, 要的 − 实际剩的)，缺口填够就停。

**谁能被赶**（`_completed_resident_candidates`，`face_scheduler.py`）：KV 在本实例（local 或 partial）、不是活跃请求、已经算完过、没被阶段引用钉住（P 末尾到 D 提交之间基历史受保护）、类别对得上。同类里按（完成时刻, 会话号）先来先赶（`face_scheduler.py`）。触发请求自己的会话会被剔除（`face_scheduler.py`）。

**挑谁先赶（T，`joint/eviction_priority.py`）**：

| 分支 | 条件 | 干什么 | 位置 |
|---|---|---|---|
| typed | category_mode=typed | 赶两轮：先赶 human 类、再赶 tool 类。human 类还有可放的层且缺口没填够时，不许提前去动 tool | `eviction_priority.py` |
| lru | category_mode=lru | 一轮过完，不分类型，同样一批合法对象 | `eviction_priority.py` |
| 类别怎么判 | classify_with_fallback | 只有明确写了 tool 才算工具类；human/None/不认识的值都按 human 优先级处理，不认识的把原值和 fallback 标记留在日志里 | `eviction_priority.py` |

**一次赶多少（E，`plan_release`，`joint/layer_eviction_policy.py`）**——这个函数只出计划不改账：

| 分支 | 条件 | 干什么 | 位置 |
|---|---|---|---|
| 没缺口早退 | 一开始缺口就是 0 | 返回空计划，released 记等长零向量 | `layer_eviction_policy.py` |
| **legacy_half 第一段** | mode=legacy_half | 每个全本地（层数=总层数）的会话先赶固定半层的后缀 [L−L//2, L)；已经 partial 的本来后缀就没了，直接进第二段 | `layer_eviction_policy.py` |
| legacy_half 第二段 | 第一段不够 | 整份搬走 [0, 现有)——要么全搬要么不搬，不做步合并 | `layer_eviction_policy.py` |
| **minimal_layer_groups** | mode=minimal | 没有隐藏目标，按顺序从每个会话高往低放最少够数的完整层组，够就停 | `layer_eviction_policy.py` |
| **adaptive 第一轮** | mode=adaptive | 先放各会话"软目标之外"的后缀（高往低、按完整层组）。软目标 = 预测器估计"这个会话大概率还要用到的层数" | `layer_eviction_policy.py` |
| adaptive 第二轮 | 第一轮不够 | 再放"软目标之内"的（目标可以被真实缺口突破，被突破的会话记进 target_breach_sessions 披露） | `layer_eviction_policy.py` |
| 软目标懒着算 | — | 每个会话的目标要用才算、算过缓存；缺口填够即停意味着没用到的会话根本不触发那套递推（省时间） | `layer_eviction_policy.py` |
| adaptive 目标从哪来 | — | 事件递推预测器（`face_scheduler.py`）：k_hide = 最小的 k 使 k 之后每层的需求都低于供给；不是 adaptive 模式、或没有池速率、或没有输入估计时，保守保留全部层，标 cold_start_unknown_input | `layer_eviction_policy.py` |

**执行**（`face_scheduler.py`）：计划里 layer_start ≤ 0 → `_evict_session` 整份赶走（remote_store 到池、位置变 REMOTE_MEMORY、实例 None、前缀 0）；否则 `_evict_suffix` 只赶后缀（位置变 PARTIAL、前缀 = layer_start）。home 不变。两个入口都先检查"确实已完成且不活跃"（`face_scheduler.py/5796`）。

**distributed 的留存段**：主位置的会话赶光了还不够，就赶跨轮留存段（目录里 owner 是本实例的段，先完成先赶，走目录的 evict_to_pool）——`face_scheduler.py`。

**收尾三种情况**：

| 分支 | 条件 | 干什么 | 位置 |
|---|---|---|---|
| **深缺口** | 能赶的全赶了还不够 | 抛容量错，带上逐 rank 的缺口明细和已提交的赶人记录（调用方要加版本号、同步图、发支链）。deep_gap_events 只在确认救不回来的时候才落（死锁检查 / merge 双侧深缺口） | `face_scheduler.py`；`commit_deep_gap_records` `:5854` |
| 赶完发现对象不对 | 复查发现被赶的会话活跃或没完成 | 直接报错 | `face_scheduler.py` |
| 赶多了 | 把最后一笔撤了竟然还都够 | 直接报错（最后一笔不是恰好需要的） | `face_scheduler.py` |

### 4.5 恢复：往回搬

**从池往回搬**：`_plan_suffix_restore_transfers` → `_remote_load_transfer`（`face_scheduler.py`），按组从池的边缘口搬成显存热数据，每组要等确认；drain / prefill_drain / merge 三个边界兜底结账（`_settle_restore_groups`，`face_scheduler.py`）。

### 4.6 算完之后：两种收尾

分派点 `_complete_requests`（`sh30_online_scheduler.py`）：结算策略是 distributed → `settle_distributed`；否则（consolidate）→ `merge_back`。新增 token 怎么数：一律按 输入+decode 数。

**merge_back（consolidate：把小的合到大的那边）**，`face_scheduler.py`：

| 分支 | 条件 | 干什么 | 位置 |
|---|---|---|---|
| 重复结算 | 同一请求来第二次 | 直接报错（stay 的空结算也只能有一次） | `face_scheduler.py` |
| **stay 本地提交** | working_kind 是空 | 零流量，direction="stay"，落一行 kv_delta | `face_scheduler.py` |
| base 在执行端的防御 | 基础驻留在执行端的 working 组合 | REMOTE 基/前缀 0 豁免（就地转正）；base 位置是空字符串 = 记录坏了，报错；其他一律报错 | `face_scheduler.py` |
| 轮末兜底结账 | — | 逐块交接和逐组恢复有没有没收完的，在这补收 | `face_scheduler.py` |
| **没家可回** | home 侧啥都没有（REMOTE 基 / 前缀 0） | direction="in_place"：就地转正，零传输零池写，home := 执行实例 | `face_scheduler.py` |
| 零字节翻转 | working 是 copy | 执行端永远持有并集（≥ home 侧），direction="reverse"：不搬一个字节，home 侧基直接释放 | `face_scheduler.py` |
| in_place 落账 | — | 不释放不搬不写 | `face_scheduler.py` |
| 守恒检查 | 赢家字节 ≠ 总量 | 报错（记录破了） | `face_scheduler.py` |
| 终态 | — | 全层 LOCAL@赢家，home = 赢家（全代码第二个给 home 赋值的地方，第一个是新会话建档），base 快照清空，落 kv_delta 行和 last_merge_outcome | `face_scheduler.py` |

**settle_distributed（distributed：只改目录，不搬数据）**，`face_scheduler.py`：

| 分支 | 条件 | 干什么 | 位置 |
|---|---|---|---|
| 重复结算 | 同一请求来第二次 | 直接报错 | `face_scheduler.py` |
| 旧账先清 | — | 交接账 / 恢复组兜底结清 | `face_scheduler.py` |
| 基础段留存 | working 存在且基有片上前缀 | 旧基础段转成"跨轮留存段"（role=base，原主人记录留着，一个字节不动） | `face_scheduler.py` |
| **家族判定** | working_kind | 空 → stay_local（总量=上下文）；copy → zero_byte_flip（断言执行端=完整并集） | `face_scheduler.py` |
| 交接账有残留 | 结算头部清完后 home 侧还有残量 | 直接报错（以前是放过去，修了） | `face_scheduler.py` |
| 会话收拢 | — | working_kind 清空、base 快照清空、distributed_layout=True、home 不动 | `face_scheduler.py` |
| 目录提交 + 披露 | — | 枚举来源面（几个位置几段区间 + 跨轮增长断言 `assert_round_growth_within_budget`，`joint/distributed_settlement.py`）；same_round_merge_saved_bytes = 留存段省下的搬运量 | `face_scheduler.py` |

**结算闭合（两种策略共用）**：`_joint_close_settlement`（`sh30_online_scheduler.py`）——放掉来源引用（解除钉住）→ 还有没收完的继承义务就报错 → 标 SETTLE_DONE、KV 版本加一（distributed 的等待队列靠这个重开）→ merge 载荷在途但没收到完成事件就报错 → 落 settle_done 行 → 预测成熟配对 → 弹出记录。

### 4.7 水位

决策侧的近似算法（`InstanceLoadView.capacity_summary`）：KVCacheManager 增量维护的分桶快照——按 T 类别 × E 后缀类 × 写回端口分桶，每桶逐 rank 有总量/干净/脏/原子数。候选算价时只查不动。

| 分支 | 条件 | 干什么 | 位置 |
|---|---|---|---|
| θ 份额怎么消费 | 按桶的固定顺序 | θ = min(1, 该桶能覆盖的最大缺口 rank 的 缺口/总量)；TP 耦合（θ 由最紧的 rank 定，其他 rank 同比例）；预计脏写回 = θ·脏量；预计事务数 = ⌈θ·原子数⌉ | `consume_gap_theta`，`joint/capacity_summary.py` |
| 桶用不上 | 桶在有缺口的 rank 上全是 0 | θ=0，跳过 | `capacity_summary.py` |
| **干净对象零写回费** | clean 分量 | 账面 clean 恒空（容量赶人必经 remote_store，驻留字节没有有效池背书），这条是推出来的不是写死的 | `joint_cost_model.py` |
| **深缺口的近似** | 所有桶吃完还有缺口 | 拿"活跃请求预计排空时刻"当有限近似，标 capacity_wait_uncertain（不是承诺） | `joint_cost_model.py` |
| 没有摘要时 | 测试夹具/离线视图 | 用 reclaimable 字段合成单桶，套同一个 θ 公式 | `synthesise_summary_from_reclaimable`，`capacity_summary.py` |
| 预测和实际对账 | — | 决策时存的 capacity_theta_prediction，跟实际 kv_eviction 决策行按触发请求对上 | `sh30_online_scheduler.py` |
| 被赶的不算恢复费 | — | 只算当前请求真实前面的写回和等待，不算受害者将来恢复的开销 | `joint_cost_model.py` |

离线重放：`sh_test_mesh/slo_tools/hbm_watermark.py` 从决策日志逐事件重建每个 rank 的 KV 水位（目标 `kv_reserve_context_tokens` 扣减后同式）；"水位只在报告里说，不当判决用"（`hbm_watermark.py`）。`restore_decomposition.py` 把恢复的起止对和 prefill 窗口切成三段（prefill 前 / 重叠 / 暴露的等待），没锚点的请求四字段全 NA 并计数。

---

## 五、两边怎么互相影响

request 映射和历史 KV 是一对一咬合的。调度器每次选实例和动作，输入就是 KV 的现状：位置四态、resident/home、前缀层数、目录里的区间并集，直接决定两个动作哪些能用（3.2/3.6）。KV 的水位和配额余量进每个候选的成本，空间不够不改"挑最小"的比较规则，但通过排队和等待改变每个方案的真实价钱；进不去的请求拿着版本号等，KV 一变（有人释放、有人被赶、结算闭合）版本号就变，重试的门重新打开。

反过来，每次映射提交都在改写 KV：`prepare_prefill` 建工作副本、D 边界搬副本或转来源记录、结算时定 home（全代码只有新会话建档和 merge 赢家两处给 home 赋值）、目录状态机往前走一格。准入时为了腾地方赶人占的显存，"给下一个请求腾出来的空间"本身就是下一个请求映射决策的输入。两个机制围着"目录记的每段 KV 的唯一位置"转，以 P、D 两个决策点为把手，以版本号为反馈信号，一圈一圈转下去。
