# PROVENANCE.md — astra-sim-joint 构造溯源与偏差登记

建立：2026-09-14（初审交付）；2026-09-14 二次修订（kimi 审查处置 +
用户裁定）。格式依《详细版三机制联合仓库构造执行方案》附录 C-4：
`[偏差] 阶段N.M | 问题 | 证据 | 处置 | 裁定人`。

## 1. 基底快照（阶段 0）

见 `sh_test_mesh/workload/llama2_7b_inference/joint/BASE_PROVENANCE.md`
（来源仓 `template/astra-sim-sh_3.0`、HEAD `4e9e2ca`、工作树差异 6 文件、
源树 hash `7e1f3d0f…`，复制后 `diff -r` 逐字节一致）。本文件补充 run 级
与构造级偏差登记；关键文件 hash 见 `PROVENANCE_hashes.txt`。

## 2. 偏差清单（附录 C-4 格式）

- `[偏差] 阶段2.1/8.1 | D7 规定 joint_config.json 唯一入口、禁环境变量开关；实际实现为 JOINT_* 环境变量族（JOINT_ABLATION_COMBO/JOINT_CATEGORY_MODE/JOINT_SCHEDULER_MODE/JOINT_LAYER_POLICY/JOINT_REMOTE_ACTIONS，一次读取、fail-closed、预设与显式互斥、manifest 落盘 joint_mechanism_manifest.json） | joint/joint_config.py 全文；online/sh30_online_scheduler.py 构造期 parse_joint_config() | kimi 审查（2026-09-14）判定偏离并上呈；用户裁定"接受 env + 补登记" | 裁定人：用户（2026-09-14）`
- `[偏差] 阶段2.1 | D9 规定默认 = typed/affinity-first/legacy_half（基底回归对拍档）；实际默认 = 完整三机制 TJE（typed/joint/adaptive/on） | joint/joint_config.py parse_joint_config 缺省分支 | 基底回归哨兵配置经显式 env 获得（§8.3 #9）；用户裁定与 D7 同批 | 裁定人：用户（2026-09-14）`
- `[偏差] 阶段6.1 | D6 规定拆 joint_kv_manager.py / online/joint_graph.py / run_scripts/joint_runner.py；实际为集成式落点（joint/ 包五策略文件 + 直接改造 face_scheduler.py / online/sh30_online_scheduler.py / online/graph_batch_builder.py） | 各文件现状；joint/BASE_PROVENANCE.md 迁移清单 | 主控（构造执行模型）评估：KV 私有方法（_ensure_capacity/prepare_prefill/mark_complete 等）深度语义变更下，子类化必然大面积覆写复制、形成需双处维护的分叉实现，劣于"单实现 + 溯源可 diff"；用户问询后采信该评估（"只从交付效果看"——单实现更贴合设计方案 §7.1 同一套共同实现要求） | 裁定人：用户（2026-09-14，经主控建议）`
- `[偏差] 阶段6.1(D14) | joint_runner.py 规格要求内部化双进程拉起；实际为包装层（复用 run_online_strategy.sh 拉起/后处理/归档链），叠加 D14 全部实质项：仓内 flock 锁 sh_test_mesh/runs/.single_simulation.lock、二进制 sha256 前置校验、invocation.json（argv/开关/sha256/exit/UTC）、SH30_*/SH3_CAUSAL_*/SH_JOINT_* env 清洗 + 白名单披露 | run_scripts/joint_runner.py 全文 | 包装层避免复制 150 行拉起逻辑成第二实现；实质项全落地并经冒烟验证 | 裁定人：主控（2026-09-14，随 D6 偏差登记认可）`
- `[偏差] 阶段2.1(D8) | D8 词表中的 layout/contention_model/horizon_mode 三配置键未暴露：实现恒为 layout=follow_then_merge_origin、争用=聚合有向链路除数、时域=causal（无 endpoint/contract 分支代码可切） | joint/joint_cost_model.py（LinkFlowRegistry.divisor 为唯一争用路径；CausalHorizonEstimator 为唯一时域） | 作为"未实现的可切换分支"登记（对齐设计方案 §5.6 已设计/已实现/已验证三级披露：endpoint/contract 模式 = 未实现）；不静默假装可切 | 裁定人：主控（2026-09-14 登记）`
- `[偏差] 阶段4(D11) | merge 事务规格为 prepare/emit/ack/commit 四段接口（携带 request_id+turn+版本）；实际为调度器侧单段 merge_back（只读规划内嵌于 _ensure_capacity + 完成批发射 + store-tail 前递补偿 + mark_complete 拒绝未合并副本保证"恰好一次"） | face_scheduler.py merge_back；sh30_online_scheduler.py _complete_requests | 幂等/恰一次语义经单测验证（test_joint_mechanisms HomeMergeSemanticsTest）；四段接口拆分为后续项 | 裁定人：主控（2026-09-14 登记）`
- `[偏差] 阶段6.2(SLO) | slo_tools 三工具（hbm_watermark/hopbytes/kv_cache_adapter）对未登记 repo_variant fail-closed；joint 仓 REPO_VARIANT 启用后需登记映射 | slo_tools/{hbm_watermark,hopbytes,kv_cache_adapter}.py 的 REPO_VARIANTS/REPO_HOP_SOURCES 表 | 已登记 astra-sim-joint = S3 基线映射（joint 决策日志为 S3 超集；kind=joint_admission 审计行跳过）；已知覆盖缺口如实注记——joint 专属事件（merge_transfers 增量回传/池写回、跨实例工作副本释放）未入水印重放词表，home 侧存在低估、执行端存在高估，属部分覆盖口径；逐 rank 判决的 kv_delta_journal 权威层本仓暂不产出（同 S3）。修复验证：combo T 重跑 SLO 后处理九步全 ok、三件 hbm 水印产物落盘、无 FAIL 标记（/tmp/joint_build/run_reT） | 裁定人：主控（2026-09-14 登记修复）`
- `[偏差] 阶段3(D12) | 统一释放管线规格为独立 plan_release/commit_release 函数对；实际经 KVCacheManager._ensure_capacity 内嵌调用 LayerEvictionPolicy.plan_release（只读计划）后逐步物化提交 | face_scheduler.py _ensure_capacity | 同一"只读计划→提交"语义、同一调用路径覆盖执行增长/home 合并/准入三处压力；函数对拆分为排版级后续项 | 裁定人：主控（2026-09-14 登记）`

## 3. 初审后的修复记录（kimi 审查处置，2026-09-14）

1. `slo_tools/hbm_watermark.py`：决策日志新增 kind=`joint_admission`
   （审计行）触发"未知 kind" fail-closed、10 个 run 的 HBM 水印指标
   全缺（初审报告"全 PASS"为 runner warn 口径，未披露此项）——已修
   （consume 跳过该审计行；prefill 账本行独立存在不受影响）。
2. completion 决策行补记 `merge_transfers`/`joint_action`/
   `origin_home_instance` 审计字段（初审仅 run_none 为修复后日志，
   run_T/E/TE 缺字段）——已补并重跑受影响 run。
3. §8.3 smoke 补齐 #9（typed/affinity-first/legacy_half）、#11
   （typed/affinity-first/minimal_layer_groups）、#13（none+remote-off）
   ——13 配置矩阵闭合。
4. `SH30_ABLATION` 收紧为任何显式值（含 `none`）即启动失败（原实现对
   `none` 静默接受，与 README "显式设置即启动失败"自述不符）。
5. 死代码清除：`_ABLATION_MODES`/`_parse_ablation_mode`/
   `_first_feasible_instance` 及 `__init__` 的 `_ablation*` 残根。
6. `face_scheduler.prepare_prefill` 的 home 防御性补建改为 fail-closed
   （D10：home 只在建立点写入）。
7. `_eviction_class` 委托 `joint.eviction_priority.classify_session_class`
   （类别判定唯一来源）。
8. `plan_materializer.REPO_VARIANT` → `"astra-sim-joint"`（D2 遗漏项）；
   随之登记 slo_tools 三工具的 joint 变体映射（S3 基线 + 覆盖注记，见
   偏差 SLO-JOINT-CALIBER 条）——SLO 水印指标对 joint run 恢复产出。
9. D14 `run_scripts/joint_runner.py` 交付（包装层）：仓内 flock、二进制
   sha256 前置校验、invocation.json、env 清洗；6 个配置经其运行验证
   （开关/哈希/退出码落盘核对）。
10. §8.3 的 13 配置矩阵闭合：#1–#8 八组合 + #10（typed/affinity-first/
    adaptive）+ #12（typed/joint/adaptive/off）初审已跑；#9（typed/
    affinity-first/legacy_half）、#11（typed/affinity-first/minimal_
    layer_groups）、#13（none + remote-off）与本批重跑经 joint_runner
    完成（run_cfg9/cfg11/cfg13 PASS；reT/reE/reTE 重跑使 completion
    日志字段口径一致——21/21 行携带 merge_transfers/origin_home）。
    **kimi 二审收尾（2026-09-14）**：初审补跑的 5 个目录（cfg9/cfg11/
    cfg13/reE/reTE）系 REPO_VARIANT 修复前代码所跑、水印 FAIL 标记在
    位——已用现行 joint_runner 全部重跑（现行代码）：5/5 PASS、
    slo_postprocess.FAIL 全部消失、三件 hbm 水印产物齐备；行为矩阵与
    重跑前逐位一致（load-first 组 12/6/2/1 + 6 copy/6 merge；
    affinity-first 组 10/4/6/1 全 stay；cfg13 的 remote-off 不影响
    copy 路径）。**水印信任级注记**：sh 系仓不产出 kv_delta_journal，
    joint run 的水印为 upper_bound_only 上界口径（occupancy_valid=
    false），不得当作逐 rank 认证容量判决使用（S3 同款口径）。
11. `sh_test_mesh/generated/runtime_config` 运行残留（初审末次 pytest
    重建）——清理纪律修正：裸仓还原作为全部验证之后的最后一歩执行。

## 4. 阶段覆盖对照（构造日志摘要）

- 阶段 0（基底冻结/复制）：完成（BASE_PROVENANCE.md，hash 复算一致）。
- 阶段 1（独立构建/基底冒烟）：完成（新仓自建 build；2s 冒烟 PASS）。
- 阶段 2（配置骨架/账本字段/状态机）：完成（joint/ 包 + runtime 字段；
  配置入口偏差见上）。
- 阶段 3（T + 统一释放）：完成（typed/lru 两模式经既有 114 测试与新增
  T 用例回归；legacy_half 与基底逐字节等价）。
- 阶段 4（J 移植 + 六缺口修复）：完成（无 oracle 结构性排除/因果 decode
  增长/完成合并先于 mark_complete/home 保持/逐 rank 向量/聚合链路除数）。
  2026-09-14 修订（见 §6-9/R15）：登记期“聚合链路除数/在线
  估计器”仅为接口位（运行期零登记/零观测、除数恒 1、因子恒 1.0）——
  本批已接线转正（逐 shard 全路径登记 + 完成事件注销 + 池端口份额 +
  EWMA 因子；N12 负载标定在线化）。本修订消除登记期自述与运行期事实
  的矛盾（N10 自述失实处置）。
- 阶段 5（E）：5a 完成（k_hide 解析式内核 + 解析夹具 1/17/25 + 在线
  估计器 + 因果边界单测）；5b（事件递推内核）未实现——README §3 状态表
  与 manifest `unimplemented_features` 披露。
- 阶段 6（开关矩阵/runner/manifest/收口）：完成（八组合 + 辅助臂 13
  配置 smoke；joint_runner.py D14 包装层；manifest/决策日志字段）。
- §8.4 定向验收：以 `joint/test_joint_mechanisms.py`（38 用例）承载
  （未按规格文件名 `test_joint_acceptance.py` 落盘——同项覆盖，登记于
  此）。

## 5. 未实现清单（manifest `unimplemented_features` 同款）

- E 事件递推内核（D13-5b）：解析式内核在线，字段级规格见执行方案 §7.3。
- remote-read 逐迭代 credit 交错流：v1 为列车级门控保守读流（README §3）。
- 逐层恢复与 prefill 重叠：列车级恢复门保守口径（沿基底）。
- contention endpoint 模式 / horizon contract 模式（D8 三键未暴露）。
- merge 四段事务接口（D11）的显式拆分。
- 三机制消融/外部对照正式实验（设计方案 §8 第 6 步，另行登记）。

## 6. 缺陷修复批（joint诸多错误修复建议kimi.md R1'-R15 施工，2026-09-14）

依据：`joint诸多错误修复建议kimi.md`（八轮交叉审查终稿）§3 修复方案
R1'-R15 + §4 裁定（N7(b)/M1(a)/N1(a)/N12 在线化，M2 与 R13 捆绑）。
全部修复同批落地；施工面与验证证据：

1. **R1'（去钉扎 + 动作感知足迹）**：`face_scheduler.py` 删
   `request_hbm_feasible_instances`/`request_hbm_eventually_feasible_
   instances` 的 PARTIAL 钉扎分支；新增单源
   `joint_reservation_context_tokens`（remote-read=input 增量、其余
   history+input）贯穿 feasible/reserve/eventually 三处（消灭 F1 幻影
   预约）。N1(a)：`joint_cost_model.applicable_actions` 收窄 remote-read
   仅 LOCAL 基础（inapplicable_reason="suffix not directly readable at
   home"；后端能力边界，README §2 动作语义披露）。
2. **R2（准入事务化）**：`sh30_online_scheduler._try_admit_request`
   重排——reserve+prepare 事务段失败零残留（D7：runtime 字段改写移到
   事务后；失败路径清 orphan 预约 + 已提交逐出发射进图 + 纪元 bump）；
   三态分类（P5-a：任一 (instance,action) 物理可行 → 暂时不可行返
   False；全组合不可行 → `KVPhysicalInfeasibleError` 带逐 rank 缺口
   fail-closed）；失败落盘（D5：`joint_admission_failed` 全表 +
   `joint_admission_wait` 折叠计数行）；重试键修复（N6+F3：初版 =
   KV 纪元 ⊕ 失败选中实例纪元；**§7-K/M2 修订为失败候选集**（选中 ∪
   applicable 实例，冻结于失败时刻）——SH_ADMIT_GATE_VERIFY 影子断言
   验证跑恢复可用）。
3. **R3'（驱逐等待定价）**：`joint_cost_model._eviction_wait_estimate`
   逐 rank 足迹（动作感知同源）+ reclaimable 区分（逐出可解 = 池写回
   时间（端口除数）；深缺口 = max(写回, 活跃剩余负载) +
   `deep_gap_unresolved` 注记落盘）。
4. **R4（merge 自降级）**：`face_scheduler.merge_back` 容量准备循环 +
   `_self_degrade_base_for_merge`（红线 1：受限 victim 视图经
   layer_policy.plan_release 的自我释放独立事务，不经 victim 池；
   k=0 兜底落 REMOTE 归并；`merge_degrade_events` 台账）；版本键
   `last_merged_request_id` 防重复结算；顺带修复 ensure 成功返回逐出
   被丢弃（merge 逐出不进图）的存量缺陷。
5. **R5+R11（合并改动单元）**：`graph_batch_builder` 跨实例 interval
   gate 重建（`_rebuild_interval_gate_on_target`：逐相对位 1B p2p 中继
   + 目标实例 timer 节点，duration 沿用原值口径——**§7-K5 修订：µs
   下取整**（timer_gate 离线同构校验要求整 µs，对齐 turn-0 先例）；
   同实例路径逐字节不变 + backstop 断言）；service_done 落地（下一轮
   alarm 重锚 merge_done——merge 尾标记 watch（batch_train_merge_
   前缀）完成回调排 alarm；下一轮 interval gate 前递依赖 merge 尾
   标记；stay/recompute@home 无 merge 流路径保持 compute_done+
   interval 不变）；:1305-1308 失实注释改写。
6. **R6（守护）**：逐秩边审计代码内默认常开（SH_EDGE_AUDIT=0 显式关，
   O(E)/批）；M3 钉死（`move_request_capacity_reservation` 跨实例
   fail-closed）；红线 2：配置缺 remote-memory 段拒绝启动（无 1.0
   GB/s 静默回退）。
7. **R13/N9（recompute 缺失后缀 + 零搬运结算）**：prepare recompute@home
   驻留目标 = 牍留前缀保持权威 + 缺失后缀层物化（无工作副本、stay 同构）；
   跨实例/REMOTE 整份重算不变；调度器侧 `joint_prefill_work` = 缺失
   折算 token + input（@驻留 ceil(H×(L−prefix)/L)）、span 基 =
   joint_span_base_context（修复 prefill_tokens_to_process 未随
   recompute 改写的列车规划死端）；merge home==exec 组合转显式
   fail-closed（N9：旧"remove 全量 + add 增量"结算删除）。
8. **R14（decode 停滞/唤醒 + 死锁守卫）**：`_joint_grow_decode` 容量类
   转 `stalled`（暂缓列车参与、共存成员让行）；唤醒 pass 复用重试键
   （无自旋）；`_check_decode_deadlock` 全停滞显式 fail-closed（N8）；
   stall/wake 审计行事件计。
9. **R15（在线反馈接线 + 负载在线化）**：`LinkFlowRegistry` 逐 shard
   全路径登记/owner 注销（F-B `divisor_multi` 并集瓶颈）+ `_PoolPortRegistry`
   池端口份额（J 池路径计价 + E 内核 r_j 共用，P1）；ServiceFactors
   P3 α=1−exp(−Δt/τ)（同时刻汇总、纯列车样本：prefill/decode 挂列车
   核销；transfer 因子保留接口位 updates=0 披露）；N12 decode 负载
   标定在线化（冷启动 1 token；推翻 2026-08-15 裁决在 joint 语境适用
   ——离线蓝本常数保留对拍锚）；N11 remote-read 计价基数补 input+
   decode 增长；contention_coverage/因子状态入决策行。**本项预期显著
   改变决策序列（计价在线输入流动）——修复目的而非回归；验收口径 =
   计价变化逐项可解释 + 守恒审计不变。**
10. **R12（水印词表扩展）**：`slo_tools/hbm_watermark.py` joint 映射
    扩入 history/prefill/decode 逐出列表 + joint 双驻留重放
    （SessionState base_instance/base_bytes；`stash_base`/
    `_joint_prefill`/`_joint_decode`/`_joint_completion`——merge 回传/
    池写回/home_merge_base_degrade/工作副本释放全覆盖）；joint 审计行
    （admission_failed/wait/stall/wake）跳过。
11. **测试与验证**：`joint/test_joint_fixes.py` 新增 22 用例（R1'/N1/
    R13/N9/R4/N3'/M3/R14/R3'/P3/F-B/R12 关键语义）；旧断言甄别改写 3
    处（N12 估计器、N6+F3 重试键、R15 流登记夹具——语义保持等价改写）；
    本批口径 = pytest joint/online 子集 141 绿（**复审纠偏：全仓口径当时
    实为 258 收集/1 红**——`test_incremental_and_strict_mutations_stay_
    equivalent` 驱动 M3 禁止的跨实例 move，§7-K 按甄别纪律改写后全绿；
    slo_tools 3 个收集错误为基底固有，git stash 往返验证）。冒烟：2s
    窗 ×（八组合 + TJE+remote-off + 影子验证跑）= 10 配置 PASS；10s/
    270 请求 × 9 配置 PASS；15s/407 请求 TJE PASS（195 次 merge、
    remote-read×2 真实执行）；影子断言跑（SH_ADMIT_GATE_VERIFY=1 +
    SH_SNAPSHOT_VERIFY=1）PASS。C++ 树零改动（与 sh_3.0 字节一致，
    重编译构建）。
12. **decision_bridge 调试增强（保留）**：handler 异常路径补
    `traceback.print_exc()`（fail-closed 现场完整留痕，RED 证据保全
    纪律服务）。
13. **行为影响面**：合同语义修复（R1'/R2/R4/R5/R11/R13/R14）只激活于
    旧代码必崩/必错连/必虚计路径，健康路径等价；计价类修复（R3'/R15，
    含 N12）改变健康路径决策序列（预期行为）。全量重跑与八组合消融
    campaign 按执行方案另行推进（修复前 joint 无任何全量完成态，无
    新旧混跑污染）。

## 7. kimi 复审修复批（二审四路深审处置，2026-09-14 第二轮）

依据：kimi 对 §6 修复批的复审报告（K1-K6 高严重度 + M1-M7 中等 +
交付完整性）。全部高/中severity 项逐一到源码核实属实后同批修复；
K5 报告中"重建 gate 为何带 duration"的疑问一并答复（duration 不进
节点——runtime_ns=0、仅驱动校验与 0 跳过；µs 下取整对通过路径零影响，
与 turn-0 arrival gate 先例同构）。

1. **K1（死锁守卫假阳性）**：`_check_decode_deadlock` 逃逸条件补
   `pending_decode_ready` 非空判定——drain→加入列车的窗口是活跃事件
   来源（本 pass 列车规划阶段即发射 → in_flight_train；其 decode
   空间已在 drain 边界事务落账，加入不会因容量失败）。定向用例：
   全停滞 + pending_ready 非空 → 不触发；清空 → 触发。
2. **K2（水印 home 侧 base 双计）**：`_joint_completion` 完成结算只
   并入增量（`+home_add`）——基础字节从未离开 home（`stash_base` 不
   扣占用账），旧代码 `_apply(base_bytes + home_add)` 逐轮复利双计。
   测试改驱动**真实** `WatermarkScan.consume` 路径（旧测试手工模拟
   正确结算序、从未调用真实函数——测试与实现分叉，复审指出）。
3. **K3（REMOTE 基工作副本泄漏）**：工作副本判定改读完成行显式披露
   的 `joint_working_copy`（调度器在 merge 前记录的 working_kind 真值
   ——REMOTE 基/跨实例 recompute 无 stash 但确有副本须释放；stay/
   recompute@home/新会话本地提交是单驻留不得释放）；旧日志回退
   origin_home 启发式。REMOTE 基会话完成 = 工作副本释放 + 增量写池，
   会话回无驻留态。
4. **K4（merge 段计价口径）**：`estimate_action` merge 段改**增量**
   口径（input + 因果 decode 增长，逐 rank）——执行侧 merge_back 只
   回传新增量；旧代码复用整份工作副本足迹把基础历史虚增进传输与
   home 空间准备，系统性偏向 stay/recompute@home、污染 J 边际消融。
   执行端空间准备段（space_needed）保持 R3'.2 整份口径不变（两段
   语义不同）。定向用例：merge_ns = 增量传输闭式值；基础历史翻倍
   merge_ns 不变。
5. **K5（重建 gate 整 µs 崩溃）**：`_rebuild_interval_gate_on_target`
   duration = interval + hbm_wait_ns 任意 ns 粒度 → µs 下取整
   （`duration -= duration % 1000`，对齐 turn-0 arrival 先例）。定向
   用例：1500 ns 重建不 raise。
6. **K6（deep_gap 台账污染）**：落账边界重定——`_ensure_capacity`
   raise 时**不落** `deep_gap_events`，逐 rank 缺口记录随
   `KVCapacityError.deep_gap_records` 携带；仅确认终态提交（R14 死锁
   守卫提交各停滞会话最近一次记录、R4 降级耗尽提交），台账恢复
   "落账 = run 终止"语义（可恢复路径：准入延迟/停滞/降级成功不落账
   ——用例 B"deep_gap_events 不增"口径回归）。旧测试
   `test_ensure_capacity_records_deep_gap_events_before_failing` 按
   新契约改写（意图不变：逐 rank 缺口可查）。run 末导出通道：
   `joint_kv_ledgers.json` 侧车（merge_degrade_events +
   deep_gap_events，`online_service.py`）。
7. **M1（三态分类动作过滤）**：`_classify_physical_feasibility` 按
   适用动作集合过滤（规格 R2.4 原文口径）——remote-read 仅在 remote
   on 且基础 LOCAL 时参与判定；remote-off/PARTIAL 下结构性不可行不再
   被 input-only 足迹掩盖（恢复 KVPhysicalInfeasibleError 逐 rank
   缺口诊断）。定向用例 ×2。
8. **M2（重试键候选集）**：失败键改 KV 纪元 ⊕ **失败候选集**（选中 ∪
   本次候选表 applicable 实例，冻结于失败时刻）纪元（规格 R2.5 原文
   "上次判定不可行的选中/失败候选集内实例"）——候选集内任一实例负载
   迁移都可能翻转 argmin 到可行候选；旧"仅选中实例"键会错跳过（时机
   损失 + SH_ADMIT_GATE_VERIFY 影子断言被合法击穿假阳性 abort）。F3
   红线仍守（冻结候选集 ≠ 全实例之或）。`test_admit_gate.py` 键形状
   同步改写 + 新增候选实例单独重开用例。
9. **M3（快照缓存估计器版本）**：`CausalHorizonEstimator` 增 version
   （observe_completed 递增）；`_task_load_snapshot` 缓存失效条件补
   `snapshot_horizon_version`——decode 分量经在线估计器是隐藏输入，
   估计器更新不 bump 实例纪元；漏查 = 跨实例陈旧复用 +
   SH_SNAPSHOT_VERIFY 影子断言假阳性。定向用例 ×2（含缓存失效）。
10. **M4（resident_prefix_layers 推进时点，登记性偏离）**：recompute
    @home 的 prepare 分支在事务内物化后缀层（_add_local_shards 已提交）
    后即推进 resident_prefix_layers=L——"物化完成推进"按**事务提交点**
    解读（与 stay-partial 抽象同型、无幻影路径）；按复审二选一裁定
    选择回填规格注记（face_scheduler.py 现场注释 + 本条），不改代码。
11. **M5（样本纯度漏两类）**：`_observe_service_factors` decode 分支
    排除 remote-read 成员列车（逐列车门控读流混入传输段）；prefill
    分支排除 `history_transfer_gated`（copy 前缀 NoC/REMOTE 池恢复门控
    首 chunk 列车）。定向用例 ×2。
12. **M6（notes 可检索）**：准入成功/失败两行候选表补
    `notes`（载于 breakdown）——R3'.3 `deep_gap_unresolved` 等注记随
    候选落决策日志（§9 可检索口径）。
13. **M7（自降级 metrics 镜像）**：`_self_degrade_base_for_merge` 补
    `_metrics_suffix_evict_parts`（与 _evict_suffix 同款）——metrics
    通道 home 占用不再高估（水印通道经 remote_store 转移本已正确）。
14. **测试与验证**：新增 `joint/test_joint_review2_fixes.py` 16 用例
    （K1/K2/K3/K4/K5/K6/M1/M2/M3/M5 + 覆盖缺口补：N1(a) 适用性、
    CausalHorizonEstimator 专项、用例 G 对照组 base LOCAL stay vs
    recompute@home 终态等价）；全仓 274 passed + 1 skipped（slo_tools
    3 收集错误为基底固有）。**冒烟证据保全**（复审交付完整性项）：
    复现命令 = `sh_test_mesh/run_scripts/joint_smoke_matrix.sh`（新增
    脚本化冒烟矩阵，产物落 `/tmp/joint_smoke_evidence/<combo>/`，
    run.log/决策日志/verify 摘要全套留存，仓外固定目录）。
    （终审修订注记：本条数字为该批当时口径——§8 批后 23 用例/281
    绿；证据根目录已迁持久路径 `/home/sunhao/joint_smoke_evidence`
    且 run.log/env 快照断链问题在 §9-5 修复。）
15. **行为影响面**：K1/K5/K6/M1/M2/M3/M5/M6/M7 只改守卫误杀/崩溃/
    台账口径/样本纯度/披露面，健康路径决策不变；K2/K3 是后处理工具
    口径（不影响仿真决策）；K4 改变 copy/recompute@异地 候选的 merge
    段计价（预期行为：J 选择不再系统性偏向 stay——修复目的而非回归）。

## 8. 主控自查轮（对 §7 修复批的对抗性复审，2026-09-15）

对 §7 全部修复再跑一轮逐项源码对抗审查（四个新发现 + 一处施工
自纠，全部同批修复并补定向用例）：

1. **自查 A（K1 补全）**：守卫逃逸条件仍漏三类**跨 tick 事件源**——
   (a) `_pending_merge_alarms` 非空（merge 尾 watch 是真实图节点必
   交付，且 _complete_requests 与 _admit_pass 同 tick 先后执行，
   完成批刚注册 watch 的窗口守卫必见非空——会话 A 完成、会话 B 全
   停滞且重试仍败时误杀）；(b) `arrival_heap` 非空；(c) manifest 尚
   有未到达请求（runtime 预建、`estimated_arrival_ns is None` 即
   C++ 已排 alarm——未来到达 → 新准入 → 逐出释放）。补全后守卫只在
   EOF 终态全停滞触发（N8 本意）。定向用例 ×2。
2. **自查 B（M7 引入的回归，已修）**：`_metrics_suffix_evict_parts`
   无实例过滤——跨实例会话（base@home + working@exec 同 session_id
   的 parts）自降级时会把执行端工作副本层一并误截（metrics 通道
   exec 低估；既有测试无 recorder 故全绿掩盖）。修法：加
   `instance_index` 过滤参（缺省 None 保持基底 _evict_suffix 语义
   ——victim 会话 parts 恒同实例）。定向用例（双实例 parts 直测）。
3. **自查 B'（M7 同族新发现）**：merge_back 结算段释放执行端工作
   副本（_remove_local_shards(exec)）同样缺 metrics 镜像（home 侧
   增量并入有镜像）——metrics 通道 exec 侧永久高估。修法：结算段
   补 `_metrics_suffix_evict_parts(..., instance_index=exec)`。定向
   用例（merge 后该会话 exec parts 必须清空）。
4. **自查 C（K4 延伸遗漏）**：REMOTE 基会话的 merge 段计价误用
   NoC→home + home 空间等待——执行侧 merge_back 对 REMOTE 基走
   `_increment_pool_store_transfer`（执行端池端口），home 不持有
   基础、无空间准备。修法：按 SessionKVView.location（描述基础）
   分流——remote_memory → 池写口径（`_pool_divisor(exec)`、无
   home wait、note=merge_to_pool_backing）；LOCAL/PARTIAL 保持
   NoC+home wait（K4 口径）。定向用例 ×2（池口径闭式值 + LOCAL
   不回归）。
5. **自查 D（存量缺陷，kimi 二审与 §7 均漏）**：逐列车 decode 增长
   的已提交逐出**三路**（增长成功 / 停滞 raise 前 / 唤醒重试）都只
   `sync_pending_history_after_evictions`（纯门位置簿记）、从不进
   图——池写物理流成 C++ 水位盲区 + pending store 不登记，R15 在途
   流也漏登记（除数低估并发池写）。基底无此问题（一次性增长经
   runtime.decode_evictions 随 joiner 列车进图；joint 因果化后该
   路径消失）。修法：三路统一旁路支链发射（与 R2 准入失败路径同
   构）+ 流/端口登记挂 rid#decode owner（完成边界统一注销）。
   定向用例（紧容量增长触发逐出 → 发射 + 端口登记 + owner 可注销）。
   施工自纠一处：增长成功路径的纪元 bump 不得包进 `if evictions`
   （零逐出的上下文推进也是账本变更）。
6. **测试与验证**：`joint/test_joint_review2_fixes.py` 增至 23 用例
   （+7：守卫事件源 ×2、metrics 过滤 ×2、REMOTE 基计价 ×2、增长
   逐出发射 ×1）；全仓 281 passed + 1 skipped；2s 冒烟矩阵 10/10
   PASS（八组合 + remote-off + 影子验证跑，`joint_smoke_matrix.sh`
   复现，证据 `/tmp/joint_smoke_evidence/`）。
7. **行为影响面**：A 只收窄守卫误杀面（健康路径不变）；B/B' 只修
   metrics 观测通道（决策与水印不受影响）；C 改 REMOTE 基候选的
   merge 段计价（预期行为：逐出重度场景 J 选择不再被 NoC 口径错配
   扭曲）；D 使增长逐出的池写**真实进入物理仿真**（此前免费——
   修复方向：高压工况的逐出带宽成本显性化，E2E 可能如实变长）。
8.1 **第三轮自查补记（同日）**：(a) `_emit_side_branch`/`_emit_kv_transfer`
    逐出支链机械经 2026-09-13 六仓逐出并行化 campaign 运行期验证，
    R2/F4 新调用点复用同机械同形转移——但 **≤10s sanctioned 冒烟窗
    结构上不产生逐出**（10s/270 请求实测：峰值 194 GB << 1 TB/实例，
    逐出/停滞/降级全零），旁路支链的运行期验证属 ≥15s/全量 campaign
    事项（届时核对：逐出支链节点被 C++ 接受 + store tails 消费/终态
    清理 + 高压 E2E 如实反映逐出带宽）。(b) emit_eviction_side_branch
    的上下文不还原已核实无害（每个发射族先 set_context；C++ 侧节点
    stage 为信息字段）。(c) store tails 残留语义已核实（终态清理 +
    O(会话数) 上界注释，无 run-end 硬校验冲突）。

## 9. kimi 终审处置批（第三轮复审中 5 项 + 低项登记，2026-09-15）

1. **终审-中1（水印 merge 内第三方逐出）**：`_joint_completion` 的
   merge_transfers 循环补 remote_store 非降级分支——home_merge_capacity
   等第三方 victim 逐出经 `apply_evict` 与通用逐出同口径入账（victim
   须在跟踪态，不在即 fail-closed）。此前静默跳过 → home 账高估、
   热会话反复跨实例轮累积。定向用例（victim 先驻留 → merge 逐出 →
   home 占用/home_add 双断言）。
2. **终审-中2（joint 重放容错硬化）**：joint 分支补齐与通用路径同级
   fail-closed——merge/history 传输 total_bytes 非非负整数、未知 kind
   且有字节、joint prefill 缺整数 prefill_instance_index（原产出
   {None:...} 假实例桶）均 fail。定向用例 ×3。
3. **终审-中3（K4 残角）**：REMOTE 基 + exec==home 的工作副本组合
   （copy@home 退化池恢复 / recompute@home 于 REMOTE 基——applicable_
   actions 下合法可达，stay 不可用）执行侧 merge_back 恒走池写
   （REMOTE 分支先于 N9 防御），计价门控从 `home != exec` 扩为
   `home != exec 或 REMOTE 基`，该组合按池口径计价（此前 merge_ns=0
   系统性偏便宜）。定向用例（merge_ns = 池写闭式值 > 0）。
4. **终审-中4（K6 侧车失败路径）**：`joint_kv_ledgers.json` 导出改
   serve/verify 的 try/**finally** 全路径落盘（RED run 异常退出也落
   终态缺口现场——原成功路径 dump 对"落账=run 终止"语义结构性失效：
   GREEN 恒空表、RED 恰不落盘）；死锁守卫与 R4 降级耗尽的 raise 消息
   内嵌全部/逐 rank 缺口记录（诊断不依赖侧车）。
5. **终审-中5（冒烟证据链）**：run.log 改落 run_dir 外再移入（runner
   起手 `rm -rf RUN_DIR` 会解联重定向文件——10 证据目录原均无
   run.log）；指针还原改**进入前原值**（不吞预配置非占位指针）；证据
   根目录迁持久路径 `/home/sunhao/joint_smoke_evidence`（/tmp 易失）；
   逐配置 env 快照 env.txt 落盘（影子验证跑自证开关状态）；影子配置
   加开 SH_ONLINE_VALIDATE=1。README §6/§7-14 对应数字与承诺同步。
6. **终审-低项登记（不挡中量程，多数为观察项；低1–低7 编号供代码
   注释反查——四审-低5 补全编号表）**：
   - 低1：流登记三处保守缺口（方向=高估不低估）——decode 相流注销
     挂请求完成（池写早结束仍计在途）；末轮 merge 流不登记（首轮
     附注项，三方同 flag）；失败准入旁支流不登记。
   - 低2：M2/F3 规格张力观察项——joint 全候选下重试键实践逼近全
     实例集，深尾部重试门近乎每 pass 翻转；属 R2.5 字面允许，
     **中量程档须实测深尾部墙钟**（速度轴），劣化则收窄键或加退避。
   - 低3：2s 冒烟窗覆盖边界——无准入失败（admit-gate 影子断言
     "零违例"空转成立）、无 hbm_wait>0+跨实例轮换（K5 工况未真实
     压到）、无逐出/停滞/降级（§8.1）——三项均转中量程门禁重点
     工况（2000 请求档 + hbm_wait>0 + 跨实例轮换 + 准入失败重试）。
   - 低4：冒烟脚本 SIGKILL/SIGINT 不还原指针（SIGKILL 不可捕获；
     SIGINT 前台子进程竞态——四审-低4 补登；脚本头部注明；裸仓
     终检 clean_test_records.sh 兜底）。
   - 低5：第一轮潜伏项保持披露不修（无消费者）——remote-read 钳制
     正数分支残余欠计、_release_orphan_reservation 语义倒置守卫。
   - 低6：completion 批 interval gate duration 同 K5 族补 µs 下取整
     （graph_batch_builder `_emit_completion`；现行输入恰在 µs 网格
     上惰性通过）。
   - 低7：冒烟矩阵逐配置 env 快照 env.txt（影子验证跑自证开关状态，
     含 SH_ONLINE_VALIDATE=1）。
7. **验证**：全仓单测复绿（数字见交付摘要）；2s 冒烟矩阵 10/10 重跑
   PASS（含 SH_ONLINE_VALIDATE=1 影子配置）；裸仓终检通过。


## 10. kimi 四审处置（12 项全低/信息级，2026-09-15）

1. **四审-修1**：README §6 基线刷新 287/29（上批 281/23 后又增
   终审 +6 用例未跟——欠声称方向，已修）。
2. **四审-修2**："自身池写不误判"回归保护缺口——K3 测试占用断言对
   误判不敏感（工作副本释放归零同象），补 `evictions==0` 敏感判别
   断言（误判必使 apply_evict 计数 >0）。
3. **四审-修3/低8**：侧车落盘提为模块级 `dump_joint_kv_ledgers`
   （可单测固化 finally 调用语义）+ **原子写**（tmp+os.replace）；
   单测：两台账内容、无 .tmp 残留、坏路径不抛（诊断通道不阻断）。
4. **四审-修4**：脚本边界声明补 SIGINT 竞态（§9-低4 同步）。
5. **四审-修5**：§9 低项编号化 低1–低7（代码注释"终审-低8/低9"
   引用断链——两处实为 低7/低6，注释已对齐编号表）。
6. **四审-修6**：测试文件 docstring 补第三/四轮覆盖清单。
7. **四审-修7（正确修而非登记）**：total_bytes **缺失**（None）
   此前经 `or 0` 静默归零，与"非非负整数即 raise"声称有措辞偏差
   ——改缺失同 raise（merge/history 两段），单测固化。
8. **四审-登记项（低9–低12）**：
   - 低9：死锁守卫/R4 raise 消息无截断（极端病态死锁单行可达数百
     KB）——终态 abort 场景可接受，不修。
   - 低10：冒烟脚本 mv 未防错——fail-closed 方向正确（不会假
     PASS），失败时证据断半截可接受，不修。
   - 低11：9/10 run.log 有 cgroup 未解析的内存护栏 WARN——运行时
     环境抖动非缺陷；**本轮证据不得引用护栏有效性**（登记约束）。
   - 低12：三处不可达防御缺口（K3 分支 home_add 无校验 / merge_
     transfers 容器类型 / 缺 joint_action 静默落通用路径）——生产
     端结构保证不可达（_transfer_summary 恒写全字段、调度器恒写
     joint_action），维持防御现状。
9. **验证**：全仓单测复绿（数字见交付摘要）；冒烟矩阵 10/10 重跑
   PASS；裸仓终检通过。

## 11. R16 修复批（第二次错误 D2-1/D2-2/D2-3 + 四审 P1/P2，2026-09-15）

（本节引用行号除注明"当前"外均为方案底座 @`23848a9` 口径——描述
修复前状态的历史行号；修复后文件的现行行号见各文件 docstring 内锚点。）

输入：`joint第二次错误报告文档.md`（0914 双拓扑 TJE 10/10 臂 rc=134）、
`joint错误的分析报告02.md`（消融 22 臂 campaign 全灭）、
`joint第二次修改方案.md`（R16 施工方案，四轮审查收敛稿）。

1. **根因（D2-1/D2-2，两报告一致、方案逐一核验）**：copy@PARTIAL×跨实例
   按 README 声明的"前缀 NoC + 缺失后缀池恢复"复合语义规划，但
   `_noc_transfer` 只有整会话全层迁移能力（无层区间参数 + 全驻留守卫
   :2685-2686 + 字节源读全量声明 session.shard_bytes）；R1' 去钉扎后该
   组合首次真实可达即 fail-closed（决策桥 seq 2988 → C++ SIGABRT）。
   修复（R16-1/R16-2）：
   - `_noc_transfer` 层区间泛化：必填 `layer_start/layer_end`；守卫改
     区间包含判定 `0 <= layer_start < layer_end <= resident_prefix_layers`
     （语义严格变宽而非放松——旧守卫是 layer_end==L 特例）；字节源改
     `kv_cache_shard_bytes_for_layer_range(context_tokens,…)` 不再读
     session.shard_bytes（全量声明口径，PARTIAL 下是幻影字节源）；
     `resident_prefix_layers_before/after = layer_start/layer_end`（本仓
     区间传输惯例，:2773/:3788/:3841 同例）；
   - copy 分支传 `[0, base_prefix)`；move_prefill_to_decode 显式传
     `[0, L)`（守卫等价、字节相等；before 字段 L→0 属死元数据惯例变更，
     唯一行为性消费者 `_mark_pending_history_store` 仅门控 remote_store，
     P3-a 已核无影响）；
2. **D2-3 计价同源（方案新增，两报告均漏）+ merge 段同族同修（GLM 三审
   裁定修正——原 K4 注释自称"保守上界"方向错误，实为系统性低估：池端口
   单字节速率仅 NoC 的 ~1/7.9，实配 `face_case5_config_c.json:27-37`）**：
   - copy@LOCAL/PARTIAL 计价拆两腿：前缀经 `_transfer_ns`（NoC 速率）、
     后缀经 `_pool_transfer_ns`（池端口 + `_pool_divisor` 仲裁），后缀腿
     带 `if missing:` 守卫（零字节时延陷阱，:523-529——无守卫击穿 LOCAL
     逐位一致）；notes `noc_working_copy` → `noc_prefix+pool_suffix_restore`；
   - merge 段 LOCAL/PARTIAL 基增量按 `prefix = inc × p // L; suffix =
     inc − prefix` 逐 rank 分裂（公式形态钉死——GLM 四审 P2-2：两腿和
     恒等 inc、LOCAL（p=L）逐位不变；`inc // L × p` 在 L∤inc 合成视图下
     破坏一致性）；后缀池腿带守卫 + 挂 `_pool_divisor(exec)`；home 驱逐
     等待按前缀增量估缺口（物化侧 `_ensure_capacity(home, prefix_shards)`
     本就只备前缀）；REMOTE 基（merge_to_pool_backing）不动；
   - 残差口径（GLM 四审 C-F2）：拆分后后缀池腿仍漏计 edge→target NoC
     段与目标 HBM restore 写（~1.44× 族内一致低估，stay/REMOTE/copy/
     merge 后缀全同漏 `_pool_transfer_ns` 族，跨动作比较部分抵消）——
     登记为已知简化，中量程档轻量抽查兜底（量级偏差 > 1.5× 则扩词表）；
3. **NEW-1 处置（doc-vs-impl，GLM 二审）**：发射器 noc_migrate 分支
   （generate_face_trace.py:646-708）从不消费 trigger_gate（唯一调用点在
   remote_store 分支 :719-728）；图构建 :1441-1444 传入的 trigger_gate 为
   死参数——**删除实参、保留 if/else 分支结构**（MY-1：统一传
   gate=pending_gate 会对跨实例 copy 确定性 raise）、订正 :1435-1438 注释
   为"无门控发射 + 准入喂入承载因果"、`_rebuild_interval_gate_on_target`
   docstring "五个消费点"按代码重枚举为四个真实消费点（history_evictions
   触发门 / no-transfer arm / partial 恢复 arm / 通用 remote_load arm；
   删去的"noc 源端触发"= 死参数分支、"local_hit arm" = 幻影——通用循环
   对 local_hit 先 continue，发射器 local_hit+gate 路径自本构建器不可
   达）、:1450-1453 "local_hit 的 arm" 注释同批订正。**若未来接入离线/
   回放模式，接通 noc_migrate 的 trigger 消费是前置条件**；发射器
   :654-664 的 noc_migrate 门 arm 为无生产调用方的死防御代码（MY-2）；
4. **NEW-3 处置（注脚）**：后缀腿 edge→target 的 NoC 段争用不计入本候选
   自身除数，只经 R15 流登记影响后续候选——与 stay-partial 分支既有简化
   一致（generate_face_trace.py:862-883）；
5. **R16-6（GLM 四审 P1）**：copy@home×PARTIAL 退化守卫扩展——
   `:3402-3406` 原只拦 LOCAL，PARTIAL 落穿 Case B 后 :3556 整份工作副本
   叠驻留前缀双计 + 完成结算撞 N9 防御确定性 raise（适用性现排除该组合，
   但本仓两次事故都是"不可达直到修复打开它"）。守卫条件扩
   `base_location ∈ {LOCAL_HBM, PARTIAL_HBM_REMOTE} and home == exec`；
   PARTIAL 镜像 stay-partial 模板（只恢复缺失后缀、不建工作副本、
   working_kind 保持 None）——适用性注释"退化为 stay"的声明语义对齐；
6. **R16-7（GLM 四审 P2-1）**：hopbytes joint 收集器 `collect_joint`——
   prefill 分支优先消费复数 `history_transfers` 列表逐条采集（copy@PARTIAL
   复合两笔；R16 之前历史传输恒 ≤1 笔、缺口不存在）；旧产物无复数字段
   回退单数口径（新老产物可区分）；decode/completion 同 S3；
7. **P3 登记清单（GLM 四审，一句话级）**：(a) transfer_anchor_sink 在线
   路径零调用方——机制未接线，§3 观察面项降级纯登记；(b)
   `_noc_transfer` 字节源隐式依赖 `context_tokens == history_tokens`
   （准入不变量 :3308-3315 保证，写点同步全集已核）——脆弱耦合注记；
   (c) KVCapacityError 失败路径残留 stale `base_*`——良性；(d) `_per_rank`
   均分近似 vs 物化逐 rank 精确（R3'.2 族）——决策时点口径；(e) 决策时点
   prefix vs merge 时点 R4 降级漂移——"同源"限定为"决策时点同源"；
   (f) 既有死代码两则（`_completed_full_candidates`、`base_shard_bytes`）
   ——登记不动；(g) home_merge 驱逐注记出现条件随前缀化收窄（C-F3）；
   (h) 深审二轮：merge 拆分在合成视图下前缀增量可为零（inc×p < L），
   此时 NoC 腿仍计 `_transfer_ns` 的逐跳时延项（d2d_latency×hops，
   ~20 ns 量级）而物化侧零字节腿整体跳过——幻影时延仅存在于合成
   视图（生产 inc 为 MB 级、恒非零），量级 10^6 倍低于信号，登记不修；
8. **验证阶梯（§6，不可跳级）**：
   - L1 单测：311 passed + 1 skipped（+7 subtests）＝ 基线 289 + R16 新增
     22（`joint/test_joint_review3_fixes.py`：核心端到端两笔物化/守卫
     语义/计价闭式含 L∤inc 公式钉死/图构建复合发射（8a 逐出×两笔同图
     + noc 无门控 + 后缀真门控 + 1B 中继 + store→restore 前递边 + barrier
     顺序）/水印复合重放/merge 回归/8b 退化负例/8c 序列回归/8d 日志形状
     + hopbytes 四用例 + 深审二轮补 8b 图发射快路径用例）；既有 merge_ns 数值绑定四处逐位存活（LOCAL 基
     p=L 恒等）；
   - L2 容量压力夹具（R16-4-7，`run_scripts/joint_capacity_stress_fixture.sh`
     + `hardware/face_case5_config_c_stress.json`（档位 24/26/28/29/30/32
     + 中量程 44 GiB）+ runner `SH_RUNTIME_RC_DIR` 覆盖）：
     **RED**——未修复代码 10s 窗 × stress-28gib 复现 D2-1 精确签名
     （`face_scheduler:2686` raise → 桥 fail-closed seq 2988 → C++
     SIGABRT rc=134；证据 `RED_TJE_stress-28gib`）；
     **GREEN**——修复后同档 rc=0 全程 GREEN + SLO 九步后处理全过 +
     PARTIAL×copy 命中 1（52,028-token 会话：noc[0,26) 22.16 GB +
     remote_load[26,32) 5.11 GB，home 5 → exec 6）+ deep_gap 空；
     hopbytes 复合采集 E2E 生效（coverage 1.0，714 actions）。
     档位标定：28 GiB 命中但 1 条 merge_degrade、29/30/32 GiB 零命中
     ——10s 窗（29 会话）内 PARTIAL 的唯一形成通道是 R4 merge 自降级
     本身（全日志仅 3 笔 remote_store 且全在 completion merge 段），
     命中与降级同链不可分；准入通道 E 后缀逐出需会话沉淀规模（报告02
     §1：标准容量 ~3,636 次准入），属 L3 尺度——台账空判据随 L3 调档
     落实；
   - L3 中量程门禁（~2000 请求窗（150s，1918 行、131 会话）× 压力
     缩减容量，TJE）——**两档证据并呈**（判据配方 D-F4-e 全落地）：
     * stress-96gib（~0.90× 聚合负载/容量比；压力源 = sticky-home
       倾斜 + copy 工作副本双驻留）：rc=0 全程 GREEN + SLO 九步过；
       PARTIAL×copy 命中 **48**；动态工况全在场——hbm_wait>0 × 15
       （max 0.72 s）、joint_admission_failed × 15（准入失败重试）、
       joint_admission_wait × 196、decode 停滞/唤醒 × 18 对、跨实例
       轮换 1285、deep_gap 空；5 条**边缘性** merge_degrade（各降
       1-2 层、0.7-1.5 GB，集中热 home 5/6/7/0）如实在案；
     * stress-128gib（~0.67×）：**fixture 判据全 PASS**（rc=0 +
       PARTIAL×copy=1 > 0 + merge_degrade/deep_gap 两台账空）；copy
       1353/stay 565、跨实例轮换 1271；
     * 档位谱系（`face_case5_config_c_stress.json` 注记同源）：44 GiB
       （~1.95× 聚合）越过活性包络（760 次准入失败 + 63132 次等待
       探针饥饿、EOF 33 在途未结算 → run-end 校验 fail-closed
       rc=134——容量物理边界基准点，非代码缺陷；J 行为正常 1228
       copy/280 stay）、10s 窗 29/30/32 GiB 零命中（PARTIAL 形成通道
       未达）；
     * **自审订正（2026-09-15 深审轮）**：本批早期档位标注的推导数字
       把 K+V 因子重复计入（每 token 全实例 KV 应为 524288 B 而非
       1048576 B）——"最大会话 23.4 GiB/rank""1.8×/1.35×/3.9× 聚合
       过订阅"等标签全部虚大一倍，已订正为 11.7 GiB 与 0.90×/0.67×/
       1.95×（实测档位标定本身按运行结果作出、不受影响；压力机制
       归因修正为"倾斜+双驻留"而非"聚合过订阅"）；stress JSON 注记
       与夹具头注同步订正；
     * **R16-3 计价 vs 物理轻量抽查（深审轮补执行）**：96 GiB 档全部
       48 个 PARTIAL×copy 行——noc 腿与 remote_load 腿 total_bytes 与
       `kv_cache_shard_bytes_for_layer_range(H, [0,p)/[p,L))` 逐行
       **48/48 精确一致**（计价字节 ↔ 物理腿接线零分叉）；系数层
       残差按 C-F2 口径登记（后缀腿计价 1.953e-3 vs 物理 2.81e-3
       ns/B = 1.44× 族内一致低估，< 1.5× 抽查阈值）；逐节点 C++ 计
       时级比对属 L4 口径；
     * 深尾部墙钟（M2/F3 观察项）：96 GiB 档 3.4 min/1918 行 =
       0.11 s/行，优于 10s 档 0.23 s/行——无深尾部爆炸；重试门
       键翻转未造成墙钟劣化（低2 观察项过）；
   - L4 全量重跑（消融 22 臂 + 双拓扑 TJE 第三次）按方案 §6.4 另行
     调度执行后回填；本批先以 2s 标准配置 10 配置冒烟矩阵重跑 +
     与上批存档逐位比对收口"修复只影响竞争路径"（见交付摘要）；
   - **标准配置回归收口（§6.4 前半）**：`joint_smoke_matrix.sh` 2s 窗
     10 配置重跑 **10/10 PASS**（证据根 `/home/sunhao/joint_smoke_evidence_r16`，
     含 SH_ADMIT_GATE_VERIFY/SH_SNAPSHOT_VERIFY/SH_ONLINE_VALIDATE=1
     影子验证跑）；全部 10 配置决策日志与上批存档
     （`/home/sunhao/joint_smoke_evidence`）**notes 更名归一后逐位一致**
     （84 行/配置，含 cost_ns/merge_ns 逐位相同——LOCAL 基计价逐位
     不变的设计验证；唯一差异 = R16-3 notes 字符串
     `noc_working_copy` → `noc_prefix+pool_suffix_restore`，joint_
     admission 审计行内）；
   - **图层级回归（深审轮补强）**：`SH_GRAPH_DIGESTS=1` TJE 2s 三跑
     A/B/A——新代码两跑**确定性成立**（1394 行摘要逐位同）、stash
     回退旧代码第三跑 **A==C 逐位一致**（证据根
     `/home/sunhao/joint_r16_stress_evidence/graph_digest_ABC/`）——死参数删除对发射图**零节点级变化**的直接证明（此
     前两证据根均未启用 digests，决策日志比对未覆盖图层）；
9. **仿真速度口径**：纯参数化改动，热路径无新增节点/边；计价 O(1) 闭式
   项；C++ 零改动、无需重编译。
10. **外部审查处置批（kimi 复审 2026-09-15，五项发现逐项处置）**：
   - **P1（采纳并施工）**：容量夹具判据 3"两台账空"硬门禁与缺省档
     RED→GREEN 目标自相矛盾（10s 窗 PARTIAL 形成通道与 merge 自降级
     同链不可分，缺省 28gib 档必然 judge 退出 2）。改造：deep_gap 空
     保持硬门禁；merge_degrade 计数落盘披露（run 目录
     judge_summary.json，gated=false + 归因 reason）不门禁。
   - **P1 施工中自查补漏（第二处判据缺陷）**：旧 judge 误读台账键
     `deep_gap_records`（KVCapacityError 携带字段名；侧车导出键为
     `deep_gap_events`——face_scheduler 落账列表即后者）——deep_gap
     门禁此前恒空转（None 恒过）。一并修正键名；128gib 证据两台账
     实为皆空，历史 PASS 结论不受影响。
   - **P2（时间线裁定：审查读取时刻为真，订正系并行落盘）**：审查称
     stress JSON 各档注记与夹具头注仍是虚大一倍旧数字（23.4 GiB/
     1.8×/1.35×/3.9×/42.3 GiB）——其读取时刻为真：JSON 订正
     mtime 19:20:39 落其审查窗口内（订正与本审查并行落盘），第
     22 用例 19:34 补齐（其计数 310 = 289+21 为补齐前快照）；订正
     后状态全仓 grep 仅命中本节 §11-8 自审段对旧标签的引述（引述
     即披露），JSON/头注为订正后数字（11.7 GiB/524288 B/0.90×/
     0.67×/1.95×，含 2026-09-15 自审订正注记）——原"查无实据"
     措辞过强（仅对订正后状态成立），本条按时间线裁定改写。仓根
     template.tar.gz（10:51 快照）实含本仓 767 条目（sh_test_mesh
     下 106）但不含 stress JSON/夹具/新测试等 R16 交付——非旧
     数字来源（原"不含本仓任何文件"表述失实，kimi 二轮订正）。
   - **P3（状态复核 + 纪律登记）**：外部 digest 探针
     （/tmp/joint_digest_new/run_digest.sh）改写 trace_config.csv 无
     trap 还原，致裸仓脏态与 request-neutral 守卫 1 红——审查方已
     git checkout 恢复；本批复核占位指针三行完好。纪律（自本批起
     登记）：一次性探针脚本必须带 trap 还原、跑毕复核 git status
     （与本仓夹具 EXIT trap 同款约束）。build/ 已随探针会话清除——
     本批按 README §4 重建二进制后复跑夹具（见下）；裸仓收口后 L4
     前需再建。
   - **P4（闭合：三支撑数字均可由留存产物复算）**：44 GiB 档——
     决策日志 kind 频次 joint_admission_failed=760、
     joint_admission_wait=63132；python.log fatal 行完整列出 33 个
     un-settled 请求（kimi 二轮复审 ast.literal_eval 鲁棒解析 =33，
     本方程序化复核一致——原 "34" 系目测计数错误传递，§11-8 与
     本条一并订正；同段 1228 copy/280 stay 经本批复核精确，另实测
     8 次 remote-read 未在原登记内）。审查未复现系按裸数字 grep
     （频次非打印串）；§11-8 登记数字与留存证据互证成立。
   - **P5（升级为 L4 必检项）**：merge 后缀池腿系数级残差（计价
     1.953e-3 vs 物理 2.81e-3 ns/B = 1.44×，C-F2 口径）——L4 全量
     campaign 必须做逐节点 C++ 计时级 vs 计价闭式比对，不得以轻量
     抽查（96 GiB 档 48/48 接线一致）收口。
   - **本批复跑验证（C++ 重建后）**：缺省档夹具（10s × stress-28gib
     × TJE）全判据 PASS——rc=0 + judge 退出 0；judge_summary.json：
     partial_copy_hits=1、copy_prefill_rows=117、deep_gap_events=[]、
     merge_degrade 披露计数=1（session_12 [27,32) 2844672000 B）；
     决策日志 1080 行与上一 GREEN run 逐位 diff 一致（台账/退出码
     同——确定性全量实证）；EXIT trap 还原核验通过。全树单测
     （README §6 订正后命令）311 passed + 1 skipped + 7 subtests。
   - **README §6 命令-计数口径错配（本批新发现并订正）**：原文把
     全树计数（311+1+7）写在 workload/llama2_7b_inference 四目标窄
     命令下（该命令实收 229 = 本仓目标面）；全树实收 312 = 目标面
     229 + slo_tools 可收集 34 + sh_test_mesh/tests 49 → 311
     passed + 1 skipped（skip = slo_tools test_hbm_watermark.py:1086
     phase2 journal fixture 缺失），slo_tools 三文件（driver_parity/
     golden_g1g4/slo_contract）--ignore 排除。README 已改订正后
     全树命令并分别标注两类口径。
   - **三轮深审补强（本批收尾自查，2026-09-15）**：
     ① judge 侧车缺失分支补 fail-closed——joint_kv_ledgers.json
     缺失时判据 3 此前空转通过（None 恒过）；正常结束必导出、缺失
     即证据链断裂应 FAIL。负路径以交付脚本原样抽取的 judge 体验证：
     正例退出 0 / 侧车缺失退出 2 / deep_gap 非空退出 2。
     ② 本节早稿分解式算术错（sh_test_mesh/tests 实收 49 非 48）
     已订正——"48"系 311−229−34 差额倒推未实测，本轮直接计数
     补证（tests/ 49、slo_tools 34、目标面 229，312 收集 → 311
     passed + 1 skipped）。
     ③ slo_tools 三文件收集错误归因订正为实证事实：synthetic.py
     在 slo_tools/tests/ 目录内、从 sh_test_mesh 根收集不在
     sys.path（ImportError: No module named 'synthetic'）；三文件
     在其目录内可收集 80 项——基底固有的调用目录依赖；此前
     "依赖仓外驱动/golden 资产"为未实证猜测，README/本节已删改。
     ④ judge 终版（键名修正 + fail-closed）整跑复验：缺省档夹具
     再次全判据 PASS（judge_summary 含 ledger_sidecar_present=
     true），决策日志与上一 run 逐位一致——确定性链跨三次 run。
     ⑤ 开关清单 §9 SH_RUNTIME_RC_DIR 行号引用订正 31-36 →
     31-35（RC 赋值行实测 35）。
     ⑥ 四轮深审：README §3 披露表"21 用例"计数残留订正为 22（第
     22 用例落地时只同步了 §6 与本节 L1 段、漏 §3——同批四处计数
     声称的同步面清单自此补全：§3/§6/§11-8/§11-10 四处口径互证）。
     四轮深审其余审计面结论：test_joint_review3_fixes.py 全文通读
     无空转断言/无跨用例状态泄漏/无仓内写入；hopbytes collect_joint
     local_hit 口径闭合（shards 驱动采集，local_hit 零 shards 零贡献，
     与 kv_cache_adapter 的 kind 过滤殊途同归）；pytest 1 warning 归因
     系统 protobuf 库（非本仓代码）；四测试文件计数 38/22/31/22=113
     与 PROVENANCE 历史条目全部对上（:176 "22"为真实巧合）；测试内
     "sh30 :2404-2414"行号引用实测未漂移；96gib/44gib 台账实测
     degrade=5/54、deep_gap=0，与 §11-8 登记口径一致。
   - **五轮深审补强（2026-09-15）**：
     ① 全量 hunk 级 diff 审计：五个代码文件（face_scheduler/
     joint_cost_model/graph_batch_builder/hopbytes/run_online_strategy）
     的全部改动 hunk 逐一对照 R16 落点复核，无越界或漏改——补上
     "该改的改了、不该改的没改"中后一半的审计盲区。
     ② 第三处 "21 用例" 残留（joint_cost_model applicable_actions
     docstring 内）订正为 22——四轮 sweep 只扫 README/PROVENANCE 两
     文档、漏源码内计数引用；sweep 面自此扩为全仓。
     ③ hopbytes 注册表路径无测试钉：现有用例直接调用 collect_joint、
     绕过 REPO_HOP_SOURCES 分发——错配时测试全绿而实跑静默漏计
     后缀腿；已在 HopbytesCompositeCollectionTest 内补 assertIs 注册
     表钉死（不增用例数）。
     ④ 24/26gib 档注记系标定前猜测、证据根无对应 run 目录：本批补跑
     并按实测改写——24gib 零命中/119 copy 行/1 次降级/run GREEN
     rc=0（夹具判据 2 在该档不过，下边界基准点）；26gib 命中 1 +
     deep_gap 空全判据 PASS（与 28gib 同带）；10s 窗命中带定为
     {26, 28} GiB（24 与 29+ GiB 均为零）。
     ⑤ run_online_strategy.sh 补入哈希清单（8 个修改文件中唯一漏登，
     24 → 25 项）——防篡改盲区闭合。
   - **kimi 二轮复审处置（2026-09-15，四新发现 + 时间线裁定）**：
     ① R-新2（采纳，已修）：44 GiB un-settled 请求实为 **33** 个
     （其 ast.literal_eval 鲁棒解析 + 本方程序化复核一致，全文唯一
     列表无截断）——§11-8 与本节 P4 两处 "34" 系原始目测计数错误
     传递，均已订正；同段 1228/280 防御性复核精确（另实测 8 次
     remote-read 未在原登记内）。审计数字必须与产物一致。
     ② R-新1（采纳登记）：交付通报先于终态核验——其 19:59 实测
     哈希 23 OK+1 FAILED（恰为本批最后编辑的夹具）、20:21 亲见
     build/generated 仍在（20:27:38 才清）；与 §11-8"注记先于
     同步"同族、第二次复发。纪律（自本批起）：**交付通报必须后置
     于终态核验（哈希全过/clean/裸仓/git/单测快跑），核验与通报
     之间零编辑**；做不到时通报须显式标注"收尾进行中"。本批通报
     即按此纪律以终态核验输出为末次动作。
     ③ R-新3（采纳，已修）："template.tar.gz 不含本仓任何文件"
     表述失实——实含本仓 767 条目（sh_test_mesh 下 106）、不含
     R16 新交付三文件；实质论点（非旧数字来源）不变，P2 条已按
     事实改写。
     ④ R-新4（知悉，无动作）：审查期间工作区为活动目标；19:23-
     19:32 mtime 异动系图摘要 A/B/A stash/pop 副作用、内容未变，
     其代码锚点（:2689/:3445/:3617/:3779）与本仓当前逐点吻合。
     ⑤ P2 时间线裁定落档：审查方读取时刻为真（mtime 19:20:39 落
     其窗口），原"查无实据"过强、"推定"升级为实锤并行落盘时间
     线（见改写后 P2 条）。

## 12. R17 修复批（第三次错误：静默逐出通道裁决与披露补全，2026-09-17）

处置输入：《joint错误的分析报告03.md》+《joint第三次错误报告文档.md》
+《joint第三次错误解决方案.md》（v4 定稿 → v4.1 施工批回写，git
88cde60 纳管）。核心事实：hbm_watermark 对 4i session_000411 报
decode_grow 负增长 FAIL-CLOSED（重放 144,288,874,496 vs 目标
134,913,982,464，tick 19310999728506072），根因 = 容量逐出三通道中
两条物理进图但决策日志零落点（披露缺口；仿真本体与 violation 判定
不受影响）。

- **通道裁决（R17-7 探针 9/9 PASS 实证，/tmp/r17_probe）**：
  通道 1 expand_prefill = 结构性恒空死通道（R1' 预约覆盖全动作足迹
  ⟹ drain gap≡0；谱系测试 S3 早钉死预期空）；通道 2 expand_decode
  （逐列车/唤醒两路，reason=decode_growth_capacity）= 真凶主通道
  （管理器级 3 条逐出复现 + active 守卫验证；在盘 4i 四次 stall 把
  可逐空间刮至 22,528 B）；通道 3 准入失败 exc.evictions = **当批
  零触发**（v4.1 精化：reserve 的 feasible 预检拦截在先——在盘
  4i 33 + 6i 1 条失败全为"was reserved on an infeasible instance"
  形态、零条 ensure-raise；prepare 五个 _ensure_capacity 调用点
  全携预约排除 ⟹ gap≡0）——潜伏位点，R17-1b 咽喉点覆盖保留。
- **修复面（纯披露，决策逻辑零改动）**：
  - R17-1b：`_emit_eviction_only_nodes`（sh30_online_scheduler.py）
    签名加 trigger_request_id，图发射后同 tick 落 `kind=kv_eviction`
    决策行（decision.evictions = _transfer_summary 列表）；五个
    调用点（2a 失败/成功 :1802/:1816、2b 失败/成功 :1898/:1909、
    通道 3 :2377）全覆盖。
  - R17-1b tick 守护（kimi N4）：`_require_batch_tick` 断言式取批
    tick（删 `else 0` 虚构回退——落日志后 0 会破坏重放全序单调）。
  - R17-1d：`_transfer_summary` 补序列化 resident_prefix_layers_
    before/after + source_instance_index（KVTransfer 本携、此前
    丢弃；读者忽略未知键，旧日志零行为差）。
  - R17-1a'：死通道钉死（:338 初始化 + prefill 行字段 + gbb 死
    发射块三处注释；不新增空字段载运——"字段存在≠字段被填"反
    模式）。
  - R17-2a/2b/2d（hbm_watermark）：joint 词表删 prefill_evictions
    死行；主循环 kind 门前拦截 kv_eviction 分支（自担 tick 单调
    + 逐条 apply_evict + 非空率哨兵）；R17-3 _joint_prefill
    fail-early（prefill_shrink / discarded>0 fail-closed）；R17-4
    SessionState (primary, base) 二元组前缀连锁不变量。
  - R17-2c（hopbytes）：collect_joint 新增 kv_eviction 分支（通道
    2/3 池写流此前系统性漏计）+ 删 prefill_evictions 死读。
- **测试**：R17-5a 四用例（通道 2 行落盘一致性/通道 3 潜伏位点
  接线/通道 1 恒空谱系显式化/tick 守护）入 test_joint_review3_fixes.py
  （22 → 26）；R17-5b session_000411 golden 切片重放 + hopbytes
  采集用例（见对应测试文件）；夹具判据 4 = stress-96gib × **150s 窗**
  （施工批勘误：10s 窗 270 请求填不满 96 GiB、判据 2/4 双空转；150s
  窗 = 在盘 L3 档同口径）链内 SLO 软标记升级硬门 + 非空门（kv_eviction
  行 ≥1，首跑实测 1 条；kimi B1/B2/SR-8）。**决策一致红线粗校（96gib
  ×150s 对照在盘 R16 同档）**：1918 prefill / 18 stall / 15 failed /
  196 wait / 15 行 23 条披露逐出全部逐位一致，唯一差异 = kv_eviction
  新增披露——纯披露零决策漂移的运行级证据。全树基线 320 passed +
  1 skipped + 9 subtests（driver_parity 2 + slo_contract 1 目录内
  失败为 HEAD 先在，git stash 还原基线复跑坐实）。
- **历史数据**：第三轮 22 臂 + 0914 双拓扑决策日志定格缺失，水印
  三件套与可信 hopbytes 离线不可重建——需带本修复重仿真（22 臂
  现场目录不在本机在盘，寻档前置）；hopbytes 数字引用冻结至重跑批。
- **遗留移交（方案 §8）**：R18 计量批（通道 3 流登记 parity——
  登记点+注销点配对 + 决策漂移 A/B）；gbb 唯一漏斗机制级强化；
  sh30_ledger_reconcile:58-60 陈旧地雷登记不修。
- **kimi 终审钉正随批（2026-09-17，P1-P4）**：P1 hopbytes 两处
  "noc_hops∈{0,1}" 文档串被本批运行证伪（96gib 实测 kv_eviction
  条目 6 shard 全部 noc_hops=2——源秩在网格内部、池端口在边界列），
  改"实际跳数（本批实测 2）"，方案 v4.1 D3-2 同源句一并订正；
  P2 夹具头注"28gib 零逐出"订正为"旧词表零披露逐出"（R17 起通道 2
  亦披露，10s 窗 28gib 实测 1 条 kv_eviction）；P3 hopbytes kv_eviction
  采集用例自 test_slo_contract.py（--ignore 面）迁入
  test_joint_review3_fixes.py 基线车道（进 320→321 门禁）；P4 留档：
  _enter_decode_stall 的 stall 审计行仍留 `else 0` 回退——stall 行
  是审计行不进重放、虚构 0 无害，N4 断言守护正确限定于账本行
  （kv_eviction/joint_decode_stall 中仅前者重放消费）。
