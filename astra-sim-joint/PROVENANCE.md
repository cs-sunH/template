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
