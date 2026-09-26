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

## 13. remote-read credit 交错流交付批（v1→v2 执行语义，2026-09-17）

依据：工作区根《remote-read改造分析方案.md》v3（路线 B，两轮 kimi 审查
后"可按图施工"）+ 用户 2026-09-17 实施中裁定（见下 P4）。改造对象 =
joint 动作 remote-read 的执行口径与决策计价。

**P4 用户裁定（实施中，2026-09-17）**：v1"批量读流 + readiness
barrier 硬栅栏"串行口径**删除**，不作为开关可选项保留——原方案 D5
（v1 保留为 `JOINT_REMOTE_EXEC=serial` 回归锚）与 P0"休眠交付"口径随之
废止；`JOINT_REMOTE_EXEC` 开关撤销（从未对外交付过），执行口径 =
credit 交错流唯一机制。回归锚改为：`JOINT_REMOTE_ACTIONS=off` 字节
等价（off 下 remote-read 不进候选、credit 代码全不触发）+ K≥S_j 单块
结构等价（`_joint_remote_read_slice` 的块 1 走原 pd_transfer 发射路径，
`online/test_remote_credit_stream.py::SingleBlockDegenerateEquivalenceTest`
以同构对照列车逐字节钉住）。

**执行语义（v2）**：

- drain 边界建立持久读计划（每步读量 = 均匀终态上下文、逐 shard XY
  路由一次冻结——**字节口径与 v1 逐位同源，字节语义零变更**；I1 守恒
  基数 = S×f(终态)）；逐列车切片在列车规划期生成（participation/S_j
  彼时冻结；`_plan_train` 的 R14 停滞跳过先于成员表 ⇒ 停滞即跳本列车
  切片顺延）。
- 块大小 K = 列车统一值 max(成员 K_j)；K_j = `remote_credit_block_
  size`（auto：max(1,⌈S_j/8⌉)——块数 ≤8、节点膨胀与 S 解耦；显式
  K 钳到 min(K,S_j)）。配置键 `JOINT_REMOTE_CREDIT_ITERS`
  （joint_config 同源，计价与执行同一裁决点）。
- 发射（D2 拓扑 + D2 发射序）：首列车（joiner）块 1 走 pd_transfer
  主链（barrier 前，barrier 语义自动降级）；尾块（2..M）旁挂支链，
  代码序 checkpoint→尾块支链→restore→块 1（restore_chain 回滚语义，
  块 1 先发射会被回滚抹掉）；home 侧 send_b 直连链不等 ack、ack_recv
  尾随；exec 侧 recv_b 顺序链（=流式到达）、ack_send 尾随；recv_b
  完成门经 `_credit_arms` 账本 arm 到体块 b 首节点。续列车（续坐成员）
  新发射槽位：块 1 上主链（无 barrier，per-rank 链序先行）+ 尾块同构
  旁挂。**T1/T2+ 块 1 门控强度不对称**（T1 经全 rank barrier、T2+ 仅
  链序）为已登记披露（README §3）。多 remote 成员同列车：体块门 =
  覆盖迭代区间成员的切片块完成门**并集**（I2 泛化）；WP9 first_token
  拆分与体块化正交（首步批=迭代 1；余量批沿 K 对齐切块，门按区间
  重叠取并集）。
- C++ 侧零改动（引擎只认图；三笔争用账——D2D 链路/home COMM_READ/
  exec COMM_WRITE——全程真实节点在线裁决）。

**决策计价（同形态）**：remote-read 候选合成 = `first_credit_ns +
max(remaining_stream_ns, compute_ns)`（Workload.cc:558-566 闭式同构；
与执行同 K 源）；K≥read_passes 时退化为旧加法形态数值（单 credit 恒等
锚，`joint/test_joint_credit_pricing.py` 手算钉住）。**计价偏差披露
（方案 §4.3.1）**：单流水线公式只付一次"首块流水填充"，多列车执行下
每个 Tj 重付一次（跨列车预取不可能——单在飞列车句柄
`state.in_flight_train` + per-rank `previous_id` 串行链双依据）⇒
**系统性低估 remote-read 成本**；量级上界 ≈ (n−1)×(d2d_latency×hops +
切片/(M_j×B_eff))，自适应 K 下 M_j ≤ 8 故填充份额 ≤ 切片/8；n
（跨列车数）决策时刻因果不可知（取决于未来 prefill 到达），公式不做
n 修正。`ActionCostBreakdown` 新增审计字段 `remote_read_first_credit_ns/
remote_read_stream_ns`。

**R15 切片键口径（方案 §4.3.2 v3）**：owner = `rid#decode#{j}` 逐切片
一键，登记钩子 = 切片创建点（列车规划期——drain 边界 S_j 未知），注销
= Tj 核销边界（`_finalize_completed_trains`，先例 = 准入相流 drain
边界注销"消费栅栏已物理通过"）；v1 的 `rid#decode` 单键继续承载 decode
增长逐出流（完成边界注销，不变）。决策日志：decode 行新增
`remote_read_credit_plan`（计划摘要）、completion 行新增
`remote_read_slices`（逐列车切片摘要，Σtotal_bytes ≡ 计划总量可日志侧
审计 I1）。

**测试**：`online/test_remote_credit_stream.py` 7 用例（I1 单/多列车
字节守恒、I2 门并集与门序、单块退化逐字节等价、I3b 续列车结构正则、
R15 逐切片键登记/核销、D2 fork 序与支链不 join、v1 删除钉）；
`joint/test_joint_credit_pricing.py` 8 用例（流水式手算锚、compute-
bound 分支、单 credit 退化锚、自适应 K 边界、非 remote 动作隔离）；
既有全套回归通过（unittest 可收集口径 joint 126 + online 75，含本批
两个新文件及交付后复核新增的 arm 账本缺失 fail-closed 钉子用例；
交付后复核补跑 pytest 函数式 7 例（test_weight_passes×4 +
test_graph_batch_rank_ledger×3——unittest 不收集，交付时漏跑）与
test_propagating_tail 自有 runner 8 例，全绿）。冒烟
（astra_compute_20.csv）：2s/10s/15s credit 模式 PASS（15s 例外窗沿
§6-11 先例，为触发 remote-read）；**off 等价门通过**（新代码 vs HEAD
worktree 基线同 10s 输入双跑：决策日志 1080 行/列车台账/请求日志/三项
metrics CSV 逐字节相同，唯一差异 = 新增空 schema 字段
remote_read_credit_plan/remote_read_slices 与宿主计时字段）。**边界
披露**：≤10s 全部窗口（含 stress-28gib/24gib）remote-read 零选中
（最小成本差 3.9ms；HEAD 同输入同零命中——历史两例来自不同 407 行
窗口边界，轨迹敏感非回归）；credit 图的引擎级通路由机制等价流量
（noc_migrate×copy / 旁挂支链×逐出 / arm 门×partial 恢复）+ R6-2 边
审计 + 结构单测覆盖；确定性 remote-read 夹具为后续项（分析方案
§4.6.2 P2）。

**交付后复核处置（kimi 现场复核 3 项，2026-09-17）**：
1. 裸仓库残留：复核发现 `sh_test_mesh/generated/` 残留 4 个物化 runtime
   配置文件（交付时"四要素核验通过"声明失实——核验跑在最后一次测试
   复跑之前、终态声明未落在最后一次状态变更之后；受控复现证实
   unittest discover 不重建该目录，残留引入路径未最终定位）。已重新
   执行双 clean 脚本并复核四要素；纪律修正 = 裸仓终检必须是仓内最后
   一个动作。
2. 测试口径洞：test_weight_passes / test_graph_batch_rank_ledger 为
   pytest 函数式（python3 直跑静默 0 例、退出 0）——交付计数"online 74"
   仅覆盖 unittest 可收集口径，上述 7 例当时漏跑；已补跑全绿（见上）。
3. 硬化采纳：builder 体块门消费处"门节点缺失静默跳过"改为 fail-closed
   ——块号 ≥2 的 arm 账本整块缺失（尾块未发射/未登记/块号错位）即
   raise（`MissingArmFailClosedTest` 钉死）；保留两类合法跳过：块号 1
   （主链先行性承担）与零字节 rank（无 shard 即无到达可等，v1 同口径）。

## 14. 部分层逐出 KV 管理改造批（PARTIAL 基 remote-read 混合形态 + merge v2 少并多，2026-09-17）

依据：《部分层逐出kv管理改造分析方案.md》（工作区根，四轮 kimi 评审闭环
＋用户逐轮裁定）需求①②。裁定④执行：旧合并机制（增量拆分归并入
home/池、R4 自降级、k=0 池归并兜底、REMOTE 整份写池归并）**直接删除、
单实现**，不设 `JOINT_MERGE_DIRECTION` 类回归档开关。

1. **N1(a) 解除（需求①）**：remote-read 适用面从 LOCAL 基扩至 PARTIAL
   基——混合形态 = 准入相后缀 [p,L) 池恢复物化（热 KV，复用 copy 池腿
   `_remote_load_transfer`/admission 发射/前递补边/`_ensure_capacity`）
   ＋ decode 相前缀 [0,p) credit 读流（`_joint_remote_read_credit_plan`
   层区间化，per_step 字节按前缀层精确派生＝I1）＋增量驻留执行端。
   LOCAL 基（p==L）逐字节不变（回归锚：单块结构等价/计价前向腿锚）。
   REMOTE 基仍拒（无主，裁定③）。适用面消融 `JOINT_REMOTE_READ_PARTIAL`
   （on|off 缺省 on；与 `JOINT_REMOTE_ACTIONS` 同模式——copy/recompute
   常驻候选，off 只移出 PARTIAL 基的 remote-read）。计价三段式：
   history_prep=池恢复（missing，与 ACTION_COPY 池腿同源）＋读流基数
   层区间化（前缀驻留＋input/增长的前缀份额，守恒拆分 value×p//L——
   自旧 merge 公式迁移）＋空间足迹=missing+input。
2. **merge v2 少并多（需求②）**：`merge_back` 重写——两侧保留量比
   大小（home=基础驻留前缀 H；exec=工作副本 W 账本真值）、小的整份
   NoC 搬给大的；**零池写**（I6：本会话 remote_store 禁止）；结果恒
   全层 LOCAL@胜者；`home_instance := 胜者`（**home 可迁移**——全仓
   第二个赋值点，§2.1 语义修订）；copy/recompute 执行端恒持并集 →
   反向**零字节翻转**（零传输、无空间准备、home 侧基础释放）；REMOTE
   基无主**就地保留**（裁定③：零传输零池写、工作副本转正、home :==
   exec 回暖）。空间准备只经统一 T+E `_ensure_capacity`（I8）；remote-
   read 双向二选一兜底；双侧深缺口才 fail-closed——**"merge 永不失败"
   合同退役**（热 KV 裁定封死自外迁兜底的必然推论，知悉项）。删除：
   `_self_degrade_base_for_merge`、`_increment_pool_store_transfer`、
   `_increment_noc_transfer`（v1 增量腿专用，无调用点）、R4 循环、
   REMOTE/k=0 池写归并分支；`merge_degrade_events` 冻结（历史侧车兼容
   读取，新 run 恒空）。
3. **工作副本模型（D1 裁决）**：混合形态 `context_tokens`=增量口径、
   `shard_bytes`=后缀 S＋增量物理真值（两口径分离）；forward merge 的
   传输字节直接用 shard_bytes（新 helper `_working_copy_noc_transfer`，
   不从 context 派生）；不变量审计四个消费点＋`_expand_local_session`
   对混合形态改用真值口径（`_working_copy_uses_shard_truth`）；全量
   审计补齐 home 侧基础驻留贡献（旧代码 strict+working 不可达掩盖的
   缺口）。
4. **计价同步（需求②/①）**：merge 段 = min(前向, 反向)（前向 = noc
   (exec 保留量)＋home 等待；反向 = noc(home 保留量)＋exec 等待；
   copy/recompute 反向=0；REMOTE in_place=0）——与物化同判据同公式
   （物化按实际 I、计价按因果估计，K4 同源披露）。旧增量拆分公式
   （prefix = inc×p//L 两腿）整体退役。
5. **前递补边硬化（需求①配套）**：`pending_store_tails` 条目扩层区间
   5 元组；消费改按区间交集选择性消费（未匹配保留）；无交集条目
   fail-closed（原 fail-open 静默直放）。行为保持注记：现行池写恒后缀
   形 [k,L) 与消费者恒交集 ⇒ 与 pop-all 等价（防御性硬化，零时间线
   扰动）。
6. **词表先行＋重放 v2（R-1 前置）**：hbm_watermark——准入侧混合形态
   期望工作副本 = 池恢复后缀（日志真值）＋coef(增量)（修掉
   prefill_shrink 误报）；完成行 `merge_direction` 分发 v2 重放
   （forward/reverse/in_place 占用语义、零池写断言、noc 源端方向交叉
   校验、joint_merge_* / joint_home_migration 计数）；无字段旧日志走
   legacy（金样对照兼容）。
7. **决策日志契约**：completion 行新增 `merge_direction`/
   `home_flipped_to`/`merge_transferred_bytes`（`last_merge_outcome`
   七键快照：merge_back 每次调用含 stay 设置）；runner 白名单＋
   `--remote-read-partial` CLI＋invocation.json。
8. **验证**：单元 276 passed（workload 全量：joint/＋face_scheduler＋
   sh30 不变量＋online/；新增 MergeV2Direction/MergeV2Capacity/
   PrepareRemoteReadHybrid/PartialHybridReadStream/补边交集与交错等
   套件；旧语义用例改写为 v2 真值表＋金样留档注释——git 9a95e06 可
   复算旧期望值）；slo_tools 38 passed＋3 既有失败（与改造前快照
   逐字节一致）；系统级冒烟 = 改造前快照树（/tmp 完整拷贝＋共用二
   进制，§4.6.4 双载体）与改造后各跑 TJE×2s——前 PASS/后 PASS，差异
   局限于 merge 经济性引起的决策翻转（copy 12→4；零字节翻转 1 次＋
   home 迁移 1 次、合并传输 0 字节、零池写）；10s 窗复核（88 次零字节
   翻转＋88 次 home 迁移、全 local 终态）；容量压力夹具复核见下。
   默认容量窗内 PARTIAL 不形成 ⇒ PARTIAL×remote-read 系统级命中由
   压力档承担（机制级已由 PrepareRemoteReadHybrid/PartialHybridRead
   Stream 端到端钉死）。
9. **C++ 侧零改动**；README §1/§2/§3/夹具注记、
   `experiment/仿真各功能开关清单.md` 同步（本案）。

**§14 收尾批（kimi 评审"达标（附条件）"六项处置，2026-09-17）**：
① 方案文档真值源闭合——范围边界补注 credit 交错流独立裁定批先行落地
（授权链 =《remote-read改造分析方案.md》＋本档 §13），本案前缀读流搭载
该机制；② I10 显式用例补齐（`I10ActiveProtectionTest`：混合形态在飞
工作副本直接逐出 → "only completed inactive sessions may be evicted"
raise；经 `_ensure_capacity` → KVCapacityError 且副本原样在账）；
③ manifest 增 `merge_semantics: "v2"` 机读键（恒单值披露、非开关——
裁定④无候选档位；`test_manifest_carries_merge_semantics_v2`）＋新开关
env→parse→属性→manifest→来源全链路单测（含非法值 fail-closed 三例）；
④ 注释/docstring 扫除四处（sh30 完成批旧 merge 描述、joint_config 旧
拒绝理由串引用、README §5-4 迁移清单与 §6 基线计数 363 passed）；
⑤ I6 措辞实施口径回写方案文档（本会话零 remote_store；第三方 victim
写池随 merge 返回为唯一自洽口径）；PARTIAL@home 适用性拒绝串改
"base resident at target instance"（location 感知链）。两批混在同一
未提交工作树（credit 批＋本案）——commit 授权时留意切分或整批进
（用户裁量）。


## 15. C12 P0 语义冻结记录补遗（四动作/块生命周期/预测终点/结算规则/暂存规格，2026-09-22）

依据：设计文档《joint机制改造方案_局部统一内存域》§0/§2.1–§2.4/§3.1/§6.2/§8
＋ 仓库设计方案《三机制联合策略》§2.2/§2.3/§7 ＋ 执行计划 F10/F11/F14/F15、D6。
性质：只读核对后落字（先落字后运行）；工作稿 `/tmp/joint_exec/C12/`（因果输入
表 / 状态机 / 可执行-失败语义清单 / remote 通路审计 / GAPS）。本轮全部冻结项
相互一致、无矛盾；逐项实现一致性核对结论如下。

### 15.1 四动作与候选完整性（冻结）

`S_exec` = 全部 instance 联合决策（joint_scheduler.py:136-149 全实例 × 四动作
枚举；load/affinity 参照同在全实例上选——"no mask" 实测一致）。R_direct 仅约
束 direct-remote 本身（JCM:642-659）：remote-read 不适用只标 inapplicable +
理由串，**不删实例**，copy/recompute/stay 继续比较。容量缺口 → 合法驱逐/等待/
恢复计划入计价（驱逐等待估计 + 池恢复腿），不以剩余 HBM/距离/边缘/PARTIAL
形成实例掩码。无合法完成计划 → 记录不可执行原因，不以无限缓存/无限等待/
静默丢请求伪造完成。**零候选 fail-closed 与 C9 deferred 重试的衔接复核成立**
（按类不相交）：joint_scheduler.py:156-159 零 applicable 候选 = 执行协议缺口
（明文 "not a capacity rejection"），异常直通不进重试队列；容量类失败
（KVCapacityError 三态，P5-a）→ `_try_admit_request` return False →
pending_admissions 纪元门重试（SH:1775-1806，重试键 = KV 纪元 ⊕ 失败候选集
实例纪元，有界、无成本爆炸）。recompute 恒适用 ⇒ 零候选结构性不可达，两
通道无重叠语义。

### 15.2 块生命周期状态机（冻结）

`QUEUED → PREPARING → EXECUTING → SERVICE_DONE → MERGE_WAIT/MERGING →
MERGE_DONE`（仓库设计方案 §2.3）。**09-21 新合同序：service_done（响应完
成）在前、merge_done（结算完成）在后，分别报告**；本地提交可跳过网络传输
腿、不可跳过合法资源准备；human/tool 是等待属性不是物理事务状态；merge 事
务携带 request/turn 版本键、恰好一次。**状态机字段可观测核对 ✓**：
service_done = `runtime.completion_ns`（REQUEST_COMPLETE 交付 tick，completion
决策行同 tick 落盘）；merge_done = merge 尾 watch（真实图节点
`batch_train_merge_<rid>`，EOF 前必须交付断言 SH:3432）交付 tick；结算事实 =
completion 行 `merge_direction`/`home_flipped_to`/`merge_transferred_bytes`
（`last_merge_outcome` 七键快照，stay 亦照实落账）。**披露缺口（归 C3b）**：
merge_done 物理时刻暂无独立事件行（代理 = 下一轮到达时刻；C3b 迁移锚后消
失，建议随其 service_done 时间戳发射通路同排 merge_done watch 交付时刻披露
——其验收原文"事件侧 service_done 与 merge_done 分别可观测"即含此项）。
现行到达锚 = merge_done + interval（旧版合同，R11/N2），迁移全部归 C3b
（到达锚与预测终点为独立条款，锚必迁）。

### 15.3 预测终点冻结（F11）＋命名映射勘误

**终点 = merge_done，四动作同终点、不按动作改变。** 现行代码语义即此
（JCM:851 关键路径 `cost = target_wait + prep + effective_compute + merge_ns`
穿过合并；计价与物化同判据同公式）。**命名映射勘误**：JCM:6-8 docstring 与
:672、joint_scheduler.py:9 的"service_done 边界"字样为旧版合同术语残留（旧
序 service_done 在 merge_done 之后、含合并；新序两者对调），文档/manifest
侧即日起统一用 **merge_done** 指称该预测终点；**JCM 实际 docstring 修改由主
agent 移交 C4 执行**（本波次 JCM 被 C1 并行占用，C12 禁改——本条即移交凭
据）；joint_scheduler.py:9 随同批术语对齐（C4 顺带或 Lane-TOOL 后继，主
agent 裁量）；SH:601/650/1470-1473/2100-2101 旧到达合同注释归 C3b 改写。
service_done（响应完成）作为服务指标从事件侧分别报告。**FS:4466-4468 注释
已按窄域例外当场修正**（新序术语 + mark_complete 定位 = merge_done 之后的
收尾点/可逐出转折，非响应完成锚；"merge 事务先于完成标记"守恒不变量本身
保留）。

### 15.4 轮末结算规则冻结（含 F15）

少并多（两侧**唯一有效块集合**实际保留量比大小、小侧整份搬大侧、**合并传
输字节 ≡ 败者侧保留量**）；零池写；home := 胜者；copy/recompute 零字节结算
且不重复释放已交接源块；无主基就地回暖（零传输零池写，home := 执行实例）；
部分驻留/多源按同一规则作用于聚合保留量（混合形态 exec 侧 = 池恢复后缀 S
＋增量 I 账本真值 `shard_bytes`，home 侧 = 驻留前缀真值）；首选方向容量不
足改试另一方向、双侧深缺口逐 rank 落账后 fail-closed。**F15 = 相等保留量
前向合并、home 标签不变：一致性核对通过**——FS:4117-4120 `sum(working)
<= sum(home)` 判定使严格相等落入 forward（执行侧并入 home），forward 胜者
= home（:4212）、`home_flipped = False`（:4326）；整数字节账本下严格相等可
达（C14 合成平局用例将按此构造验证）。计价侧 `merge_ns = min(前向, 反向)`
（JCM:903-928，含胜者侧空间准备等待估计）取两方向估计最小值、不预设方向，
与物化的账本真值裁决互补不矛盾。实现一致性（FS merge_back :3999-4329 落账
段 :4280-4330，A6 勘误锚）：守恒断言逐 rank 精确相等 fail-closed ✓、版本键
恰一次 ✓、零池写（无 remote_store）✓、统一 T+E 空间准备 ✓、双侧深缺口
fail-closed ✓。

### 15.5 部分/多源历史共同动作处理（冻结，不设独立开关）

读流基数与恢复腿按唯一有效块集合确定：LOCAL 基全层单源读流；PARTIAL 混合
形态 = 准入相后缀 [p,L) 池恢复物化（热 KV）＋前缀 [0,p) credit 读流；REMOTE
无主基 = copy 池恢复整份或 recompute（remote-read 被适用性排除、到达即
fail-closed）。**片上远读、片外恢复与本地读取分列计价**（池恢复不冒充片上
remote-read）：NoC 腿 `_transfer_ns` 与池腿 `_pool_transfer_ns` 在
stay/copy/remote-read 三分支各自分列（notes=noc_prefix+pool_suffix_restore /
pool_suffix_restore_hybrid 等），实测一致。**不设独立开关**：现行无处理机制
开关；既有 `JOINT_REMOTE_READ_PARTIAL`（缺省 on）是**适用面消融**（off 仅把
PARTIAL 基的 remote-read 移出候选集、copy/recompute 仍比较，与
JOINT_REMOTE_ACTIONS 同模式，joint_config.py:44-50 裁定④边界澄清），非机制
变体开关，不违规——口径登记供 C20/开关清单读者参考。GB 恢复门现为后缀级
保守门、逐层门控归 C15（E17 既有登记）。

### 15.6 remote-read 暂存规格冻结（F14）

**无留存型暂存、暂存复用范围 = 0**：执行端不保留基础历史副本（账本只含增
量 [+池恢复后缀]，FS:4100-4108 对 remote-read×REMOTE 基 fail-closed 为该口
径账本级钉死）；读流直达消费（credit 逐块到达即门控计算，first_credit +
max(remaining_stream, compute)）；重复读按实际消费遍数计价（read_passes =
因果估计步数；prefill 重扫遍数口径核对归 C1/C3）；执行端 HBM 仅驻留本轮增
量（R1' 预约足迹，计入容量账）。两条披露随规格生效：(a) 复用范围 = 0 ⇒
prefill 分块重扫按实际遍数**全额走 NoC**，不得暗示暂存复用；(b) **建模假设**
——credit 在途字节按链路在途建模、不占执行端 HBM 容量、**计价侧不产生执行
端 HBM 写腿**。

### 15.7 copy 共享物理块引用处理（冻结；实现归 C13）

有其他合法使用者 → 先处理引用关系或明确动作不适用，不得一边宣称立即释放
一边保留整轮源端保护；不引入新的共享前缀复制机制。逐块四步交接协议（安全
前提 → 源读+传输+目标写后更新位置/所有权/路由 → 同一交接完成事件立即释放
home 侧对应块不等轮末 → 后续消费用目标副本、未到达块按就绪事件等待）与守恒
式 `H_home(t)+H_exec(t)=H+D_handoff(t)` 归 C13 补实现（E16 证未实现；本卡
只冻结规则，不重复登记差距）。

### 15.8 remote 通路前置审计与 STAGING_WRITE 条件项裁定（C14 前置输入）

Workload.cc 五分发点按 E6 清单逐一实测吻合（RESTORE :374-377 / POOL
:451-461 / COMPUTE :587 / COMM_READ :772-774 / COMM_WRITE :810-812）；
LocalHbmBandwidthModel 六类 JobKind 全 N-way 严格均分。源端保护 = 轮内
session.active 位（victim 候选过滤 inactive+completed only，入口再守卫）；
merge watch 链路真实存在（GB `merge_done_members` → SH watch →
`_on_merge_done`，EOF 前必须交付）✓。

**STAGING_WRITE 条件项裁定：不触发（S1 波次一落字）。** 审计事实：remote-read
credit 读流全部以 kind=noc_migrate 发射（SH `_joint_remote_read_slice`），
其 exec 侧 `comm_recv` 缺省 hbm_charge=True（noc_migrate 分支与 GB credit
尾块支链两处），C++ `issue_recv_comm` 在 `hbm_endpoint_charge_active()`（
`hbm-bandwidth-contention` 缺省 true、两 hardware JSON 均未覆盖 ⇒ joint 运行
恒开）下对每块到达实计 **COMM_WRITE HBM 写作业**——即"remote 暂存写节点"
存在。但该写节点**已有服务类别**（COMM_WRITE，既有 N-way 严格均分仲裁，
即 STAGING_WRITE 预登记模式所述仲裁形态），D6 触发条件"暂存写节点**无服务
类别**"不满足 ⇒ 不新增类别、不动 C++、+≈2h/+2 构建的条件项预算不启用、
Lane-CPP 不因本条件项回归。**但计价-执行不对称登记并路由 C14**（GAPS G1，
高优先级）：F14 披露 (b) 计价侧无执行端 HBM 写腿，而执行侧每块实计
COMM_WRITE 真实争用（与 COMPUTE/增量写竞争）——预测方向性偏乐观
（remote-read 偏快），正是 C14 步骤 2"不得一边产生 HBM 写节点、一边按绕过
HBM 的通路估算"的既定复核对象；处置（计价补端点写腿 / 修订披露 (b) 措辞并
对拍披露 / recv 改 hbm_charge=False 移除写节点）由 C14 按偏差流程先补遗后
动工，若选移除节点则 Lane-CPP 回归。

### 15.9 本卡变更与测试

- 修改：`face_scheduler.py` mark_complete 不变量注释（FS:4466-4472，窄域例
  外——仅注释文本对齐 09-21 新合同术语，守恒不变量与 raise 行为逐字节不
  变）；`PROVENANCE.md`（本节，append-only + flock）。
- 新增：无仓内新增；工作稿五件落 `/tmp/joint_exec/C12/`。
- 测试（零后端）：merge/mark_complete 相关 19 项 passed（含 subtests）；
  test_face_scheduler + joint/test_joint_mechanisms + joint/test_joint_fixes +
  test_sh30_kv_incremental_invariants 全量 111 passed + 1 failed——失败项
  `test_checked_in_config_is_request_neutral_and_fails_closed_without_input`
  为并行卡 C6 物化 trace_config 指针（/tmp/joint_exec/C6/queue_2s.csv）所致
  环境性失败，**经还原注释复跑证实与本卡改动无关**（新旧注释下同样失败，
  断言对象 = trace_config 占位指针，与 mark_complete 无因果链）。

## 16. C6 链路遥测三件接线批（WP2-C++，--link-telemetry，2026-09-22）

### 16.1 变更与接口

- 修改五文件（Lane-CPP）：`main_online.cc`（OnlineDriverContext 新增
  `fluid_scheduler` 字段 + 遥测差分簿记；匿名 ns 新增
  `compute_link_telemetry()`；main() 注入与 provider 注册 + 启动证据行）；
  `DecisionBridge.hh/.cc`（新增 `LinkTelemetryProvider` 钩子 +
  `set_link_telemetry_provider()`（首交付后安装 fail-closed）；
  `deliver_and_receive` 在 `build_request_json` 产物上条件挂接顶层
  `link_telemetry[]` 数组——与 `ledger_summary` 并列、不占 snapshot 保留位）；
  `OnlineCli.hh/.cc`（新旗标 `--link-telemetry`，默认关（F7 fail-closed），
  `--link-` 前缀纳入 online family 拼写防御）。
- **实现形态偏差登记**：任务卡步骤 3 原文写"`build_request_json` 新增数组"，
  但 `StateDelta` 定义于 `DecisionMailbox.hh`——该文件不在本卡文件清单内。
  为同时满足"文件清单纪律"与"旗标关字节等价"，采用 provider 钩子形态：
  `deliver_and_receive` 内 `build_request_json(delta)` 完成后条件挂接
  `request["link_telemetry"]`。线上结果与卡文意图一致（顶层数组、与
  ledger_summary 并列）；旗标关时字段整体缺席，请求字节与改造前逐字节相同
  （A/B 冒烟的 request_journal.jsonl 逐字节相等为证）。
- 窗口语义（F5）：窗口 = 一个决策 epoch 的物理时长（上一交付 tick → 本交付
  tick，自适应、无新常数）；`served_bytes/active_ns` = fluid 链路观测器累计
  量（const `link_observer_totals()`，FluidScheduler.cpp:880）的窗口差分；
  首窗口 [0, 首交付 tick)（pre-loop 锚点），空 epoch 推进锚点不产生样本——
  窗口铺满无洞（32908 样本 0 违例实测）。观测器桶序列 spool 机制与 run 末
  `emit_link_observer_records`/`link_observer_release`（main_online.cc 发射
  侧）不动。
- 观测器归属边界（继承自观测器语义，披露）：观测器仅在调度器事件点积分，
  窗口末（=epoch tick）之后未积分字节不计——`window_end_ns` 可严格晚于最后
  归属时刻；同一边界适用于 C7 对拍所用的 spool 记录。

### 16.2 F6 覆盖边界披露（冻结项落字）

- **遥测只盖 NoC 腿（含集合通信）**：核对确认本仓一切集合通信路径汇聚于
  `CongestionAwareNetworkApi::sim_send` → `fluid_scheduler->start_flow`
  （CongestionAwareNetworkApi.cc:77-80）：Workload 发送节点（Workload.cc:776
  `front_end_sim_send`）、astraccl native/custom 集合算法（Ring/
  HalvingDoubling/DoubleBinaryTreeAllReduce/CustomAlgorithm 均
  `stream->owner->front_end_sim_send`）、SimSendCaller（Sys.cc:1544 /
  SimSendCaller.cc:34 `comm_NI->sim_send`）。设计文档 §4.1"集合通信覆盖"
  缺项在此闭合——集合通信流量自动入遥测，无需单独插桩。
- **池端口不经 FluidScheduler**：`Workload::issue_remote_mem`
  （Workload.cc:437-464）→ `sys->remote_mem->issue` → AnalyticalRemoteMemory
  端口 FIFO，全程不进 fluid 链路表；其争用保持注册表模型（C2 的
  HbmPortFlowRegistry / u_port），遥测不覆盖、不替代。
- 旗标开而观测器关（metrics off 或 ASTRA_LINK_OBSERVER=0）时累计量恒零 →
  数组恒空、不伪造数据（启动行披露该形态）；C7 的 collective_coverage 翻
  True 判据应以"全程旗标开 + 窗口无空洞 + 观测器开"联合判定。

### 16.3 测试与法证

- 构建：README §4 拼装（build/astra_analytical/CMakeLists.txt + congestion_aware
  目标），改造前后各一次全量构建成功；C++ 既有测试二进制
  BridgeLoopbackTest / DecisionMailboxTest ALL PASS（前者含 unknown
  top-level key 容忍用例）。构建目录保留（后续卡复用，C19 收口清理）。
- A/B 字节等价（旗标关，2s 窗 21 请求 TJE，launch 2 次）：run_A=改造前基线
  二进制（sha1 41118116…），run_B=新二进制旗标关。cpp.log 剥离 host 计时
  字段（logger 墙钟前缀 / wall_ms / total wall time / phase-6 ns 计数器 /
  peak rss / FluidProgress wall_seconds / io_read_ns / 读吞吐）后**字节等价**；
  raw/normalized/request metrics、cache_events、kv_hit_states、slo_* 全家、
  manifest、request_journal.jsonl（=C++ 请求原始字节）、online_decision_log、
  train_ledger **逐字节相等**；online_stats.jsonl 仅 host 计时字段
  （processing_ns / gil_wait_ns / scheduler_self_ns*）不同。仿真域计数器
  （event_count=1413 / delivery_count=1392 / tick_end_without_decision=
  1781407 / completed=21）A/B 相同。
- 五主文件 sha1 窗口：pre/post 快照 face_scheduler.py 不一致——mtime 法证
  （15:19:31 +0800）晚于 run_B 进程退出（15:19:08）23s，两 run 均加载旧字节
  （=pre sha），窗口未污染，不耗重做额度。
- 旗标开冒烟（launch 2 次）：第一次 abort（exit 134）= 并行 C1 卡编辑中的
  joint_cost_model.py 中间态（`_shard_paths` 引用先于 def 落盘，Python 侧
  AttributeError @ seq 58；C++ 侧遥测已正常工作 57 个交付）——窗口污染，
  非本卡缺陷；重做（第 4 次也是最后一次 launch，五文件 sha1 前后一致）
  PASS。验证：1392/1392 请求带 link_telemetry[] 字段、32908 样本、字段集
  精确、served_bytes/active_ns 无负值（逐链路累计单调）、窗口铺满无洞；
  与 spool 物理真值（ASTRA_LINK_OBSERVER=1 同跑发射：1335 link_bucket +
  186 link_total [METRIC] 记录，cpp.log.gz/metrics.log 留档
  /tmp/joint_exec/C6/run_flag2 供 C7 对拍复用）逐链路对账：62 观测链路
  0 违例、字节覆盖 =1.0000（遥测累计 == spool total_bytes）；旗标开不扰动
  物理仿真（gate/service 计数器与 run_A 逐字段相同）。
- launch 预算：4/4 用尽（A、B、旗标开-污染、旗标开-重做）。
- 仓还原：clean_test_records 过（trace_config 占位、traces/ 仅 *.py、无
  generated/）；run_online_strategy.sh 临时 guard（SH_LINK_TELEMETRY）跑毕
  立即还原，sha1 对 C0 快照核验一致（5b397e8c…）；该脚本持久改造归 C11。

### 16.4 开关登记移交

- `--link-telemetry` 已实现（默认 off）。开关清单
  `experiment/仿真各功能开关清单.md` §7.0 的正式登记与本旗标与
  JOINT_QUOTA_MODE=aimd 的耦合（aimd ⇒ 自动注入遥测，F7）同归 **C11 步骤
  2/3**（计划明文：C11 持有 env 注入落点与开关清单同步）；本卡文件清单不
  含开关清单文件，故不越界代笔。

## 17. C3b 到达合同迁移批（下一轮到达锚 merge_done → 响应完成 service_done + 依赖门控，2026-09-22）

合同依据：实验思路 §三.4（a(s,k+1) = f(s,k) + z(s,k)，f = 上一轮向外完成响应
时刻，z = 工具/用户等待）；仓库设计方案 §2.3 / 设计文档 §2.4 同义
（service_done = 服务边界、merge_done = 资源结算边界，分别报告；到达后仍须
满足真实块就绪与合并依赖）。F11 到达锚独立条款：预测终点（merge_done）
不动，只迁锚。

### 17.1 代码迁移（SH：sh_test_mesh/workload/llama2_7b_inference/online/sh30_online_scheduler.py）

- 下一轮到达 alarm 自 `_on_merge_done`（原 R11 重排点）迁移到**响应完成事件**
  （`_complete_requests` 的 REQUEST_COMPLETE 交付 tick = service_done 事件侧）；
  thinktime（inter_request_interval_ns）自该时刻起算；**有/无 merge 流同锚点**
  （原 no-merge 分支 compute_done + interval 与新 service_done 锚同 tick 同值，
  统一为单一路径），全部方法与对照统一适用（FACE/WSC 适配版同口径——到达
  机制在 `_complete_requests` 无模式分叉，合同自动全臂生效）。
- `_on_merge_done` 语义收窄：仅 ① merge 在途流注销（R15 职责保留）＋ ② 新增
  **merge_done 事件侧披露行**（决策日志 kind="merge_done"，decision 携
  merge_done_ns / session_id / next_turn_request_id / next_arrival_world_ns /
  next_turn_arrived_before_merge_done——C12 G5 缺口路由承接：物理 merge_done
  此前无独立事件侧披露行，锚迁移后"下一轮到达时刻"代理消失）。披露行落
  online_decision_log.jsonl（决策日志事件文件，与 completion 行同文件同
  schema 形态；若 C16/C20 需机器读 merge_done 时刻，经该文件按 kind 取行）。
- 到达后数据依赖门控（**保守口径披露**）：图侧 interval gate 前递锚 merge
  尾标记（R11(ii) 结构保留，语义自"到达锚点"改为"到达后门控"）＋ local_hit
  路径经同 rank 串行链接续（merge 传输节点先于准入发射提交于同 rank 链）＋
  池在途块经 store 尾前递边（_register_store_tails 既有机制）。**初版 =
  merge_done 整体门控（保守），块级门控留后续，不冒称已块级**。
  graph_batch_builder.py 零改动——OnlineTraceBuilder.timer_gate 的 duration
  本就仅结构语义（在线等待时长由 C++ arrival calendar/future_alarms 承载），
  锚迁移无需动构图器。
- 属性名 `_pending_merge_alarms` 沿用（R11 历史名；条目瘦身为流注销 + 披露
  账目，不再含 interval/envelope）——避免联动改名 WL/joint/test_joint_fixes.py
  与 test_joint_review2_fixes.py（Lane-JCM 侧文件，不在本卡清单；改名会打红
  C4 将跑的基线，实测 5 用例 AttributeError）。
- 死锁守卫逃逸项 (a) 与 EOF 残留检查语义保留（"图侧必有 merge watch 交付"
  事件源；"未来到达释放容量"推链由逃逸项 (c) 承载），注释按新合同改写；
  SH:601/650/1470-1473 一带旧合同注释（C12 G4 路由的实测锚点）已全部改为
  新合同表述。

### 17.2 服务指标锚点核查（步骤 6，只读；预登记条件项不触发）

核查结论：主 E2E/SLO 的 completion 时刻**不取自 FS mark_complete**。完整
通路 = decode 退出标记 watch 成员完成（REQUEST_COMPLETE 事件侧 =
service_done）→ C++ MetricCollector completion 锚（main_online.cc
register_online_metrics_anchors：非 prefill watch 成员注册 kind=
"completion"；merge watch 因注册为 STAGE_PREFILL 且 request_id 为
batch_train_merge_ 前缀（未知请求）不产生指标锚）→ [METRIC] request 记录 →
metrics_postprocess（request_metrics.csv，e2e = completion − arrival 不变式）
→ slo_tools（e2e-stats / SLO 判定 / session 级 t_session = last_completion −
first_arrival 同源于该 CSV）。FS mark_complete 的 last_completion_ns 仅有
typed-eviction 排序键（joint/eviction_priority.py）与不变量检查读者，不进
任何指标通路。因此预登记条件项（"若指标取自 mark_complete"）**不触发**：
发射半（service_done 时间戳落事件流）已天然成立（completion 决策行 tick ＋
MetricCollector completion_ns 两条既有通路），消费半（slo_tools 改读）无需
C16 改造；merge_done 侧由 17.1 披露行补齐（F11 分报配套：service_done 行
tick = REQUEST_COMPLETE 交付、merge_done 行 tick = merge watch 交付，两事件
分别可观测且单测钉死 tick 不同）。**到达锚迁移后合同完整，非半份合同**。
守恒不变量（未结算工作副本不得标记完成）保留：逻辑 merge_back 同 tick
先于 mark_complete，属结算边界而非服务指标锚；FS:4466-4468 注释勘误归
C12（计划 §2 A3），本卡未动 FS。

### 17.3 N2 承接、测试与可比性边界

- N2（2026-09-14 修复关切在新合同下的承接，必测项已绿）：thinktime < merge
  时长时到达早于 merge_done，merge 成本以"下一轮数据等待"形式留在闭环内
  ——到达 = service_done + z 不被 merge 推迟（**不逃出会话 E2E**），等待窗口
  [arrival, merge_done] = M − z > 0 且由 merge watch 交付事件解除（**不
  消失**）；z ≥ M 时合并与思考合法重叠、无残余等待（对照半边用例钉死
  next_turn_arrived_before_merge_done=False）。测试 =
  online/test_joint_arrival_contract.py（11 用例：有/无 merge 同锚统一、终轮
  无 alarm、到达≠就绪（到达被接受时 watch 仍在册）、门控事件驱动解除且不
  排 alarm、重复/未知 watch 交付 fail-closed、N2 双向、双事件分别可观测、
  图侧 interval gate 前递锚 = merge 尾标记（真实构图器只读驱动）/ 无 merge
  时 = seg2 块末）。
- **可比性边界**：v2/v3 历史结果（2026-09-22 前的一切 run）系旧到达合同
  （R11：到达 = merge_done + interval）产物，与本卡后（到达 = service_done +
  interval，到达后保守门控）不在同一到达合同下；跨界对比必须按本条说明。
  无 merge 流场景两合同逐字节同口径；thinktime ≥ merge 时长场景数值重合；
  thinktime < merge 时长场景到达提前 z、等待移入下一轮 E2E，E2E 分布必然
  移动（预期行为，非回归）。C20-⑥ 增列到达合同一致性检查项（对全部方法
  与对照统一生效）。
- 冒烟（2s 窗 / 21 请求 / TJE / 1 次 launch）：PASS（exit 0，21/21
  completed，e2e_p50 = 529.681 ms）；实测逐会话"下一轮到达 − 上一轮完成"
  = 队列 interval 精确成立（如 session_1 turn1：completion 1,160,113,734 →
  arrival 1,360,113,734 = +200,000,000 ns；session_3 turn1：+253,000,000
  ns）。窗口内无 merge 流发生（0 merge_done 行，全部 stay 本地提交——2s
  窗联合决策未选中异地动作），merge 分支行为由单测覆盖，如实披露。
- 基线回归（法证 /tmp/joint_exec/C3b/，SH 改前 sha1 2dbd7506… = C3 交付
  态）：改前 online/ 101 passed + 7 subtests、kv 2 passed；改后 online/
  114 passed + 7 subtests（+11 新增）、kv 2 passed、face_scheduler 48
  passed + 2 subtests、joint/ 216 passed（余 5 红均为
  test_joint_credit_pricing 金值 = C1 预期红，非本卡回归；本卡中途曾因属性
  改名打红 test_joint_fixes/test_joint_review2_fixes 共 5 用例，已按上文
  沿用旧名处置归零）。收尾 clean_test_records 过（trace_config 占位、
  traces/ 仅 *.py、无 generated/；build/ 未触碰——Lane-CPP/C6 独占）。

## 18. C13 copy 块级交接与源端立即释放批（四步协议补实现 + 守恒科目 #handoff/#copy-stream，2026-09-22）

依据：设计文档 §2.3（四步交接协议 + 守恒式）/§6.1；仓库设计方案 §2.2/§2.3；
C12 冻结规格 §15（§15.7 copy 共享物理块引用处理；executable_failure_
semantics §5）。E16 代码事实（现行整轮双驻留）为本卡替换对象——本批
以补实现为主、核验为辅。

### 18.1 FS 账本侧（face_scheduler.py）

* **逐 chunk 交接账本**：新增 `CopyHandoffJournal`/`CopyHandoffChunk` +
  `plan_copy_handoff_layer_chunks`（确定性切分：块跨度目标
  `COPY_HANDOFF_CHUNK_LAYERS=8` 层，n = ceil(p/8) 块均衡；p ≤ 8 时单块 =
  旧单笔口径回归锚——test_face_scheduler 两腿/单腿结构断言零改动通过）。
  `SessionKVState.copy_handoff` 为轮内 home 侧残量的唯一事实源；
  `KVTransfer.handoff_chunk`（0 基消费序）标记交接块传输。
* **四步协议落点**：(1) `prepare_prefill` copy 分支——`_assert_copy_
  handoff_source_safety`（在途读取 = 上轮工作副本/账本未闭合即拒绝；
  共享引用 = 本账本 KV 块会话私有、无共享前缀机制，结构性断言——
  C12 §5 两分支均落守卫）+ `_ensure_capacity` 目标空间准备；(2) 前缀
  改发 M 块 NoC 交接流（块 0 主链 + 尾块 `history_prefix_handoff_tail`）
  ——物理到达/所有权/路由更新由图侧逐块门控承载；(3) `_apply_copy_
  handoff_event` 在**同一事件**内完成目标侧权威化 + `_release_copy_
  handoff_chunk`（home 侧 `_remove_local_shards` + 占用记账撤销 +
  metrics 前缀段精确释放 `_metrics_prefix_release_parts`——suffix 版
  的前缀镜像，消费顺序自层 0 向上）；(4) 后续消费读目标副本（图侧
  recv 完成门 arm 进列车体，未到达块由就绪门等待）。
* **结算边界（不等轮末）**：`_settle_copy_handoffs` 边界白名单 =
  `prefill_drain`（expand_prefill 唯一在线调用面 = _on_prefill_drain，
  全部 prefill 列车核销后——逐块就绪门控 guarantee 已兑现）与 `merge`
  （防御兜底）；decode 相中段 expand_decode **不是**安全边界（多列车
  prefill 下尾块可在途），越界即拒绝（交接前释放的边界形式防护）。
  `_incremental_base_contribution` 随逐块释放收缩（全释放后贡献为零）。
* **守恒式与科目**：`assert_conservation` 逐 rank 线性精确断言
  `H_home(t)+H_exec(t)=H+D_handoff(t)` + 独立计数器/块状态交叉核验。
  口径披露：**工作副本（前缀+后缀）在准入相物化入执行端容量账**
  （容量保守上界，与旧口径同刻——test_face_scheduler 准入态双持有
  断言零改动通过；分配关系经既有 `_reservation_extra_shards` 核账，
  物理↔预约分离），故 D_handoff = 准入物化∧未释放的重复量（launch 时
  = H、随逐块释放递减、轮末归零）。科目事件入 `copy_handoff_events`
  （owner 串 `rid#handoff`/`rid#copy-stream`，与 R15 流登记同命名纪律；
  launch/handoff/stream-once/close 四类；#copy-stream payload = 被
  迁移原有历史恰好一遍，逐跳传输/池恢复后缀/增量/受害者写回/协议
  ack 在 plan 事件 `separate_ledgers` 另列分计不入 payload）。
* **merge 闭合**：merge_back 头部兜底结算 + 零字节翻转分支对
  `copy_handoff` 非空会话**不再重复释放** home 基础（C12 §2 规则 4），
  改为断言闭合（未全结算/残留 home 字节均 fail-closed——"基础历史
  最终完整落到执行端是 copy 完成条件"）；`last_merge_outcome` 保持
  七键不变（test_face_scheduler 逐字典相等断言兼容）。recompute/
  REMOTE 基 copy 无 journal，merge 释放路径原样。

### 18.2 GB 图侧（online/graph_batch_builder.py）

* **准入发射**：交接块 0 走既有主链 history_transfer 路径（readiness
  barrier 只等它——**不设"先整份搬运后计算"串行段**）；尾块（块
  ≥1）经 `_emit_copy_handoff_tail` 旁挂支链（`_emit_side_branch` 包裹，
  fork 点 = 头块发射后的主链 frontier）：home 侧 send 链全速泵出、
  exec 侧 recv 链流式到达（与 `_emit_credit_stream_tail` 同款拓扑），
  逐 rank recv 完成门入 `_copy_handoff_arms`（逐块就绪门控锚）、
  ack_recv 入 `_copy_handoff_release_anchors`（源端释放物理挂点，
  source release dependency = noc_migration_ack_recv 语义的逐块化）。
* **列车体逐块门控**：`_copy_handoff_body_blocks`（首 chunk 列车）把
  体切 B = min(尾块数+1, 迭代数) 块，尾块 c → 体块 min(c,B) 1:1 流水
  （体块 c 等尾块 c；列车体完成 ⇒ 全部尾块到达 ⇒ prefill drain 结算
  因果成立）；span 划分按布局契约（头部 span = 前 iterations 条精确
  入块；成员 span 按块迭代数比例确定性分配——GB 侧 train_plan 不携
  带 members/prefill_chunk_tokens，成员级精确迭代归属不可重建，块
  字节总量与权重总量守恒）。`_emit_train_body` 的门查找泛化为
  `_credit_arms` ∨ `_copy_handoff_arms`。**工程发现**：同列车多体块
  的聚合/集体节点不可重名（C++ commit 预检按名字对集体做跨 rank
  签名一致性校验，跨块字节不同即 fail）——copy 体块带唯一 phase
  后缀 `{train_id}_cb{k}`；remote-credit 多块在现行 auto-K 下恒单块
  未触发过该约束，此处为首个真实多块路径。WP9 首步拆分列车遇待
  消费交接门时 fail-closed（SH_FIRST_TOKEN_SPLIT 缺省关）。
* **完成边界**：completion 批弹出释放挂点账本；交接门残留（请求完成
  而尾块从未被体块消费）fail-closed。

### 18.3 本卡变更与测试

- 修改：`face_scheduler.py`（交接账本/prepare 逐块化/结算钩子/merge
  闭合/前缀段 metrics 释放）；`online/graph_batch_builder.py`（尾块
  支链/体门控/挂点账本）；`PROVENANCE.md`（本节，append-only + flock）。
- 新增：`online/test_joint_copy_handoff.py`（零后端 18 用例）。
- 测试（零后端）：本文件 18 passed（块切分确定性/prepare 逐块化+守恒/
  drain 释放/merge 不双释放+直接兜底/审计通道收缩/边界拒绝/源安全
  拒绝/**失败注入四件**：双释放、交接前释放、重复交接、乱序交接/
  GB 图结构五件：挂点+释放锚、fork 顺序、barrier 只等头块、体块
  就绪门控、完成弹出+残留 fail-closed）；回归面全绿：online/ 全量
  141 + test_face_scheduler 48 + kv_incremental 2 + joint/test_joint_
  mechanisms+fixes 62 = **253 passed**（占位态核跑；C1 金值红与 C3b
  SH 面不在本卡回归面，未出现）。
- 冒烟（2s 窗 = astra_compute_20.csv 前 21 请求，TJE）：**窗口内
  copy 自然选中 1 次**（session_1_request_1 → 实例 5，selected_action
  计数 stay 20 / copy 1；该请求 terminal_status=completed），run PASS
  （/tmp/joint_exec/C13/run_1）——自然选中，无需强制夹具，无"强制≠
  自主选中"披露项。开发期同 run_dir 3 次 launch：前 2 次为本卡新路径
  bug 的 fail-closed 崩溃（span 布局重建误依赖被 SH 丢弃的 train_plan
  键；多体块集体重名触发 C++ commit 预检），修复后第 3 次绿。跑毕
  clean_test_records 已过并核验（指针=占位、traces/ 仅 *.py；C5 并行
  卡的 generated plan 目录已按原样恢复）。
- 后续登记：(a) `copy_handoff_events` 的 run 级 sidecar 序列化需在
  `online_service.dump_joint_kv_ledgers` 加一行（该文件不在本卡清单，
  建议 C19 收口或 C16 顺带）；(b) 成员 span 的块级迭代精确归属（GB
  侧键缺失）现为比例分配，若后续需要逐迭代精确可在 SH 侧 train_plan
  增补 assignments（登记不实施）；(c) remote-credit 多块路径若未来
  启用（K<S_j），需同样的体块 phase 后缀处理（本卡已证约束）。


## 19. C5 决策日志 schema 冻结批（WP6a，joint_admission 审计流扩展，2026-09-22）

**对象**：SH `joint_admission` 决策行（`_try_admit_request` 日志点）；schema **先行冻结**（F8，先于 WP3），字段存在性与可解析性为本卡判据——breakdown 数值由 C2 物化填充、值的完整落库由 C4 冒烟复核；配额类字段（port_snapshot 族）本卡记 `"NA"`，由 C2 注册表/C11 接通后替换（F8：不缺席）。测试：`online/test_joint_decision_schema.py`（11 项零后端单测）。

### 19.1 冻结字段清单（joint_admission 决策行）

决策级（decision 顶层新增六键）：

| 字段 | 语义 |
| --- | --- |
| `selected_action` | 选中动作 ∈ {stay, recompute, remote-read, copy}（v2.8 用户裁定；与既有 `joint_action` 同值，显式冗余断言位） |
| `applicable_actions` | 适用动作集（ACTION_ORDER 序，可由 candidates 导出——显式冗余断言位） |
| `recompute_selection` | 选中 recompute 时 `{"tier": elected\|forced, "forced_reason": null\|no_history\|evicted_permanent}`；非 recompute 决策记 null |
| `load_view` | 逐决策负载视图：逐实例 `instance_index` + 五个负载量字段（queued/running/active_decode_task_load_ns、hbm_remaining/reclaimable_bytes_by_tp_rank）——InstanceLoadView 构造点数据（决策输入同刻快照） |
| `flow_snapshot` | 流表快照摘要：每链路登记流数（`"src->dst": count`，LinkFlowRegistry.snapshot()；在本请求准入相登记**之前**捕获 = estimate_action 计价所见争用环境） |
| `port_snapshot` | 端口快照：逐实例 `u_port_active_decode_streams / u_port_registered_transfer_flows / u_port_total / bulk_slots_used / bulk_slots_cap / parity_gate_headroom`——**WP3 前恒 "NA"**（C2/C11 接通后替换） |

候选级（candidates[] 每项新增两键）：

| 字段 | 语义 |
| --- | --- |
| `hops` | 逐候选 hop（route_fn 可算，与 JCM `_route` 同语义：source = resident ?? home、同实例 0 跳）；applicable 与 inapplicable 候选均落；applicable 候选与 `breakdown.hops` 同值（冗余断言位，单测钉死） |
| `breakdown` | 逐候选 ActionCostBreakdown 全 11 字段（`target_wait_ns / history_prep_ns / eviction_wait_ns / compute_ns / remote_read_ns / merge_ns / contention_divisor / hops / notes / remote_read_first_credit_ns / remote_read_stream_ns`；SH 冻结字段序常量 `_JOINT_BREAKDOWN_LOG_FIELDS` 与 JCM 声明逐一同名，单测锚定防漂移）——**仅 applicable 候选**（体量控制），inapplicable 候选记 null；值可为 null（C2 物化前） |

### 19.2 recompute elected/forced 分位与选中计数口径（设计文档 §5.2，v2.8 用户裁定）

- **elected（真选中）** ⇔ 至少一个其余适用动作存在（`len(applicable_actions) > 1`）且联合比较仍选中 recompute；
- **forced（必经重算）** ⇔ 适用动作集 = {recompute} 单元素；原因 ∈ {`no_history`（首轮，无任何历史 KV：`history_tokens == 0`）, `evicted_permanent`（历史已被永久驱逐、无有效后备副本）}。现行三态模型 REMOTE 基可经池恢复、turn-0 恒有 stay 适用——**forced 两态当前均不可达**，schema 先行冻结枚举（未来永久驱逐态/退化拓扑的断言位）；
- **计数语义**：recompute 选中数仅含 elected 口径；forced 单列计数披露（按 forced_reason 分计）、不并入选中数、不被静默丢弃出分母（elected + forced = recompute 全行数，分母可重建）；两口径由 `tier` 字段分列，下游计数（C11 守卫指标埋点 / C16 四动作选中计数 / C20-① 弃赛守卫）不得混报。

### 19.3 披露边界（冻结）

本 schema 只含**决策时刻**记录；结算时刻的逐请求合并披露（合并方向/零字节结算/home 迁移）走 FS 侧 kv_delta_journal（接口声明见 C14 步骤 5）——C5 先行冻结、不受 C14 字段需求牵连。C3b 交付的 `kind="merge_done"` 事件披露行与本扩展共存不冲突（schema 扩展仅作用于 joint_admission 决策行）。

### 19.4 冒烟证据（2s 窗 / 21 请求 / --combo TJE）

- run：`/tmp/joint_exec/C5/run`（joint_runner --combo TJE，exit 0 PASS，21/21 completed；窗口含 C13 交付后的自然 copy 选中 1 次：session_1_request_1，四动作适用集全齐，copy 候选 breakdown 全 11 字段实测值在场（hops=3、contention_divisor=3）——字段存在性/可解析性判据过）；
- 决策日志解析重建（`/tmp/joint_exec/C5/verify_decision_log.py`）：21 条 joint_admission 行全部可重建"每决策 × 每候选（9 实例 × 4 动作 = 36 候选）× 每字段"；selected_action 计数 {stay: 20, copy: 1}；recompute elected=0 / forced=0（窗口内无 recompute 选中，分母语义断言在场）；flow_snapshot 本窗决策时刻均为空（无背景在途流，如实披露；非空分支由单测覆盖）；port_snapshot 六字段逐实例 NA（WP3 前占位）；
- 测试：online/test_joint_decision_schema.py 11 项零后端单测全绿（全路径 turn-0 重建 / 有历史四动作适用集 / 冻结字段序 vs JCM 声明锚 / selected_action 枚举 / elected+两 forced 原因合成覆盖 / §5.2 计数口径不得混报分列 / load_view·flow·port 结构）；SH 侧回归 online/（含 C13 新 18 项后 191 passed + 9 subtests，其中本卡新增 11）+ kv 2 + face 48+2 全绿；joint/ 余 7 红中 5 为 C1 预期金值红（C4 重推导）、2（test_joint_review2_fixes MetricsInstanceFilter / test_joint_review3_fixes CompositeCopyMerge）经归因系 C13 批 FS 改动的跨车道效应（两用例直接驱动 face_scheduler KVCacheManager、不涉 SH；C3b 交付时点该两用例为绿）——非本卡回归，登记供 C4/C13 收口复核。

### 19.5 归档

冒烟 run 产物与验收脚本归 `/tmp/joint_exec/C5/`（run/、verify_decision_log.py、smoke_run.log、plan_materialize.log、queue_2s.csv.*）；仓内收尾已 `clean_test_records.sh` 还原占位并核验（traces/ 仅 *.py、无 generated/、trace_config git 干净；build/ 未触碰 = C6 交付在场）。


## 20. C14 remote 完整结算与暂存计费一致性批（结算闭合审计 + F14 收口 + C12-G1 处置 + kv_delta_journal，2026-09-22）

依据：设计文档 §6.2（remote 完成检查清单）/§2.4/§4.2/§3.2；仓库设计方案
§2.2 remote 行/§3.2/§8；C12 冻结 §15（重点 §15.6 暂存规格、§15.8
STAGING_WRITE 裁定）；C12 GAPS G1 预登记处置授权（补价/改披露/移除节点
三选一，先补遗后动工）。

### 20.1 C12-G1 处置补遗（先补遗后动工；裁定 = (a) 补价，实现移交 C2 rider）

**事实复核（2026-09-22 工作树实测，与 C12 §15.8 / GAPS G1 记录逐点一致）**：

- 执行侧：remote-read credit 读流每块以 kind=noc_migrate 发射（SH
  `_joint_remote_read_slice`），exec 侧 `comm_recv` 走**缺省
  `hbm_charge=True`**（GB `comm_recv` 缺省 GB:328-329；generate_face_trace
  noc_migrate 分支 gft:673-679 未覆盖该参）；C++ Workload.cc:809-812 在
  `hbm_endpoint_charge_active()`（Workload.cc:820
  `sys->hbm_bandwidth_contention`，两 hardware JSON 均无覆盖 ⇒ joint 运行
  恒开）下对**每 credit 块到达实计 COMM_WRITE HBM 写作业**
  （LocalHbmBandwidthModel 六类 JobKind N-way 严格均分，与 COMPUTE/
  增量写同仲裁类）。
- 计价侧：JCM remote-read 读流三处调用（全流 JCM:960-967、首 credit 腿
  JCM:982-989、其余流送段 JCM:990-997）均 kind="read" **两腿**（
  `_shard_leg_ns` JCM:601-607 = NoC 流送腿 + 源端读腿，无执行端写腿；
  注释 JCM:953-954 自认"F14 口径下不产生执行端写腿"）。
- 结论：两侧不对称实证成立——执行侧真实产生执行端 HBM 写服务需求、
  计价侧按绕过该写腿的通路估计 ⇒ remote-read 预测方向性偏乐观（偏快）。
  STAGING_WRITE 条件项维持 **C12 §15.8 裁定不触发**（写节点已有服务类别
  COMM_WRITE；本批不复活、不新增类别、不动 C++）。

**处置 = (a) 补价**（Python 侧为 remote-read 增执行端 HBM 写腿）：写物理
真实发生（每到达块 DMA 入执行端 HBM、后被消费读取），计价忠实于执行属
仿真正确性第一原则（设计文档 §3.2 端点 HBM 行、§6.2"不能一方面产生
HBM 写节点，另一方面按绕过 HBM 的通路估算"）；JCM/F1 归 Lane-JCM
（C1/C2）冻结占用，**实现移交 C2 rider，本卡零 JCM 改动**。

**否决项与理由**：

- (c) 移除节点（recv 改 hbm_charge=False 不再计写）——**否决**：物理上
  写真实发生，移除 = 执行侧漏计真实服务需求，把"计价偏乐观"反向换成
  "执行偏乐观"，不对称只是换了方向而非消除。如后续被推翻须附物理论证
  并回归 Lane-CPP（预登记路径，本批未启用）。
- (b) 改披露（维持 F14(b) 冻结建模假设、仅登记乐观偏差）——**不选为
  终态**：忠实计价可行且不违反冻结纪律（F1 公式修订经本补遗类流程即可
  生效），保留可消除的已知偏差无正当理由。(b) 的登记内容**降级为 rider
  落地前的过渡态披露**（下条，本批即生效）。

**过渡态披露（rider 落地前生效）**：remote-read 计价维持两腿，作为已
登记的**已知乐观偏差**——方向 = remote-read 候选 cost 只低不高；量级 =
漏计的执行端写腿值 `shard_bytes × read_passes / (B_HBM_exec/(u_exec+1))`
（与既有 copy/staging 第三腿同式，JCM:608-612）。无争用（u_exec=0 且
u_home=0）时该腿与源端读腿同值、通常不改变 max 腿 ⇒ 偏差趋零；执行端
端口争用（u_exec>0，与 COMPUTE/增量写 N-way 均分）时按 (u_exec+1) 因子
放大并可能成为瓶颈腿。C4 对拍与 C20 消融读数在 rider 落地前须按本条
口径解读。

**C2 rider 接口说明（实现规约 = 移交凭据）**：

1. 落点：JCM remote-read 读流计价三处调用（全流 :960-967、首 credit 腿
   :982-989、其余流送段 :990-997，现均 kind="read" 两腿）；
2. 增腿公式：`shard_bytes × read_passes / (local_hbm_bytes_per_ns /
   (u_exec + 1))`——与 copy/staging 第三腿（`_shard_leg_ns` :608-612）
   同式同源；u_exec 自 hbm_port_registry（C2 同批接通真实端口争用，现行
   恒 0，JCM:50 注释）；read_passes 倍乘与读腿同口径（F14 无留存暂存 ⇒
   每遍重扫的块都真实再到达、再写一次）；首 credit 腿按 credit_steps
   倍乘、其余流送段按 (read_passes − credit_steps) 倍乘，与现有字节
   基数同切法；
3. 语义边界：仅增**带宽/服务争用腿**，不改 F14 无留存暂存口径的容量半
   边（credit 在途字节仍不占执行端 HBM 容量、无驻留账本变动、暂存归还
   仍为无操作）；镜像的是既有 COMM_WRITE 服务类，不新增服务类别、
   不复活 STAGING_WRITE；
4. 实现形态二选一（rider 决定）：(i) kind="read" 增可选执行端写腿参；
   (ii) 复用 kind="staging" 三腿形并修订其 docstring（注意既有命名语义
   为"目标端留存写"、remote-read 为瞬态到达写，采用 (ii) 须如实改写
   措辞避免误读为留存暂存复活）；
5. 连带：F1 冻结公式与金值（test_joint_credit_pricing 3286 等）随修订
   移动 ⇒ C4 金值重推导须排在 rider 落地之后；rider 生效时本节过渡态
   披露作废、由 rider 批次落字取代。

### 20.2 remote 生命周期审计结论与补齐（步骤 1，设计文档 §6.2 完成检查清单）

审计基线 = C3b+C5+C13 交付后的工作树；逐项机制核对与补齐如下（全部
"机制在案 ⇔ 测试钉死"配对，测试见 §20.5）：

| 检查项 | 机制（实测锚点） | 结论 |
| --- | --- | --- |
| 基础历史远读有效期内源端有效+必要保护 | 结构保护双通道：prepare 后 primary `instance_index` 指向执行端 ⇒ home 实例逐出候选集（`_completed_resident_candidates` 的 instance 匹配）结构性不含本会话；直接逐出入口 active 位守卫（`_evict_suffix`/`_evict_session`）；守恒审计经 `_incremental_base_contribution` 恒含 home 侧基础（不误判泄漏） | ✓ 在案；失败注入 #1 钉死（两通道均 fail-closed、源端字节原样在账） |
| 保护解除时机 | 解除 = merge_back 败者侧释放 + mark_complete 同 tick（执行端 active 清除、可复为 victim）；forward 时 home 基础保留为胜者并集；物理消费序由同 rank 链顺序发射保持（结算 tick 后到达的 merge 传输先于后续任何同 rank 逐出/读节点——builder 顺序 append 语义） | ✓ 在案（审计落字 SH `_on_merge_done`/`_complete_requests` 注释） |
| 本轮增量执行端生成并保留 | remote-read 工作副本 LOCAL 基零化起算/PARTIAL 混合基 S+I 账本真值，expand 逐 chunk 增长 | ✓ 在案（§15.6 口径不动） |
| 暂存归还（F14 无留存型口径） | 无操作：无驻留账本变动；披露位 = journal `staging_return_bytes` 恒 0 + forward 后执行端实例账本字节 ≡ 0（无残留） | ✓ 补齐披露位（原仅口径字面） |
| 增量结算（少并多，按两侧实际保留量） | merge_back 以账本真值（home 前缀推导 / working `shard_bytes`）裁决方向、传输字节 ≡ 败者侧保留量；F15 相等保留量 = 前向、home 不变（`<=` 判据） | ✓ 在案；F15 合成平局用例构造验证（严格相等先断言，非肉眼比对） |
| 合并目标容量（胜者侧真实空间准备） | `_prepare_winner_capacity` → `_ensure_capacity` 取**完成时刻** `_effective_remaining_by_tp_rank`，与发射时预测空闲无关联；可逐则统一 T+E 逐出、活跃占满则双向二选一兜底、双侧深缺口逐 rank 落账 fail-closed | ✓ 在案；"发射时空闲/结算时被占"夹具钉死（可逐 victim 与活跃 blocker 两变体） |
| 失败分支闭合（取消/重复回调不双释放、不丢权威副本） | 重复回调 = `last_merged_request_id` 版本键 raise（不双释放/不重复计 token）；取消路径 = 本仓无轮中取消（fail-closed 哲学：run 级失败即终止，无半结算清理面）；watch 重复/未知交付 raise（C3b）；**本卡补齐**：watch 交付 ⇔ 结算事实在案的闭合门（下条） | ✓ 在案 + 补齐闭合门 |

**补齐交付**：SH `_on_merge_done` 新增结算闭合审计门——merge watch 交付
时 `kv_delta_find(request_id)` 必须命中 FS 结算行，缺行 = watch 通道与
结算账本脱钩（fail-closed）；替身缺接口时软跳过（getattr 软门，与完成行
`last_merge_outcome` 的并行兼容口径同款——C3b 测试替身零改动兼容）。

### 20.3 kv_delta_journal 披露接口（步骤 5；C16 消费面）

* **产出通路**：FS `KVCacheManager.merge_back` 两个出口（主路径 + stay
  早退本地提交）各恰追加一行——行存在 ⇔ 结算完成；失败结算（双侧深
  缺口 fail-closed）零行（失败分类走 `deep_gap_events`，两科目互斥）。
* **字段面**（`_append_kv_delta_row`）：seq（单调）/session_id/
  trigger_request_id/working_kind（结算时刻类别快照，终态清零前捕获）/
  direction（stay|forward|reverse|in_place）/zero_byte_flip/winner_
  instance/loser_instance/**home_before/home_after/home_migration**（C16
  home 迁移轨迹所需）/transferred_bytes/**home_side_retained_bytes 与
  exec_side_retained_bytes**（少并多裁决输入 = 两侧实际保留量；copy 的
  home 侧取交接 journal 残量口径——已释放字节不计入，与 #handoff/
  #copy-stream 科目不冲突）/new_tokens/staging_return_bytes（F14 无操作
  披露位，恒 0）。
* **访问器**：`kv_delta_find(trigger_request_id)`（最新命中/None）——SH
  闭合门与 C16 消费共用入口。
* **披露边界**：不进 C5 冻结的 decision-log schema（决策时刻记录）；
  C3b 的 kind="merge_done" 事件行与完成行既有 merge 三键零改动。**run 级
  sidecar 序列化按 C13 (a) 同款登记归 `online_service.dump_joint_
  kv_ledgers` 属主**（C16/C19 顺带加一行；该文件不在本卡清单）。
* **README 口径注记**：README:194"逐 rank 判决需 kv_delta_journal 权威
  层，本仓暂不产出"的指称对象 = wscllm 式 checksum 归档权威层
  （`results/kv_delta_journal.jsonl`+证书，开关清单 §KV_DELTA_JOURNAL
  仅 W 仓），与本接口（仓内 per-request 结算事实，无开关恒开、不落
  results/）不同物——该陈述在本卡后仍然成立，README 不动；C16 接通
  序列化/消费时同步文档。

### 20.4 G1 落地状态与 STAGING_WRITE（步骤 2）

本卡零 JCM/C++ 改动（处置 (a) 的实现归 C2 rider，接口规约 = §20.1）；
过渡态披露（已知乐观偏差，方向/量级见 §20.1）自本批生效至 rider 落地。
两侧一致性复核结论：执行侧写节点（COMM_WRITE）与计价侧无写腿的不对称
**实证成立并已处置**；STAGING_WRITE 条件项维持 C12 §15.8 裁定**不触发**
（写节点已有服务类别；本卡未复活、未新增类别、未动 C++、Lane-CPP 零
回归）。

### 20.5 测试与冒烟证据

* **新增**：`online/test_joint_remote_settlement.py`（零后端 14 用例）——
  A 组 journal 行 5 件（前向全字段/翻转 home 迁移轨迹/stay 最小行/copy
  零字节残量口径/find 访问器）；B 组失败注入三件（源端保护泄漏双通道/
  结算重复回调/合并目标容量未准备两变体）；C 组 F15 合成平局（严格相等
  构造验证 + 前向 + home 不变 + 传输字节 ≡ 任一侧保留量）；D 组 SH
  watch 闭合门三态（行在案放行+流注销+披露行/缺行 fail-closed 于流注销
  之前/缺接口软跳过并行兼容锚）；E 组覆盖率与失败分类互斥闭合（4 成功
  行恰一方向互斥 + 深缺口零行 + deep_gap 台账在案）。
* **回归**：改前基线 online/ 141+7subtests、face 48+2、copy_handoff 18
  全绿；改后 online/ **155+7subtests**（新增 14）、face 48+2、copy_
  handoff 18、kv_incremental 2、joint mechanisms+fixes 62 = 267 passed
  全绿。joint/ 余 7 红 = 5 金值（C1 预期）+ review2/review3 两例
  （C13 跨车道效应，C5 §19.4 已登记归 C4）——**与本卡改动零因果**
  （失败形态为 metrics 过滤/merge 字节断言，不涉 journal 通路；C5 交付
  时点即红）。
* **冒烟**（2s 窗/21 请求/--combo TJE/1 次 launch，预算 ≤2 用 1）：
  run `/tmp/joint_exec/C14/run_1` exit 0 PASS，21/21 completed，e2e_p50
  = 529.681 ms（与 C3b 基线逐位一致——本卡为披露/审计侧改动，零行为
  漂移实证）；deep_gap/merge_degrade 台账均空；selected_action 计数
  {stay: 20, copy: 1}——**窗口内 remote-read 自然选中 = 0（合法结果，
  步骤 4）**；copy 结算行（session_1_request_1）merge_direction=
  reverse/home_flipped_to=5/transferred=0 照实落账（经本卡 journal
  append 通路）。**强制≠自主选中披露**：remote 通路验证 = 强制 remote
  夹具（零后端 14 用例直接驱动 remote-read 全生命周期：准入→读流计划→
  结算→watch 闭合），非自主选中证据、不构成性能收益证据；2s 窗内自然
  选中率为零的原因 = 轻载窗口 stay 恒优（跨实例动作中仅 copy 因负载均衡
  选中 1 次）；stress 档 2s 窗按仓内夹具自注无逐出压力、预期同为零，
  未燃烧第二次冒烟预算。收尾 `clean_test_records.sh` 过并核验
  （trace_config=占位、traces/ 仅物化器、无 generated/、build/ 未触碰）。
* **法证**：/tmp/joint_exec/C14/（baseline_sha1.txt→post_sha1.txt：
  本卡仅动 SH/FS/PROVENANCE 三文件 + 新增测试文件，GB/JCM 与基线
  sha1 逐位一致；queue_2s.csv.*、plan_materialize.log、smoke_run_1.log、
  run_1/ 冒烟产物、trace_config.orig.csv）。

### 20.6 移交与登记

1. **C11 输入（步骤 3）**：原"#merge-reserve 释放时机核对"随配额建成
   后联合断言——核对对象 = `#merge-reserve` 预留释放与 remote 结算路径
   （service_done 方向裁决后败者/胜者侧到 `_on_merge_done` 的释放序）
   的联合断言；本卡交付的 kv_delta_journal 结算行与 `_on_merge_done`
   闭合门为该核对的现成事实源。
2. **C16/C19**：`kv_delta_journal`（与 C13 (a) 的 `copy_handoff_events`
   同批）的 run 级 sidecar 序列化行归 `online_service.dump_joint_
   kv_ledgers` 属主；README 口径同步（§20.3 注记）随该批。
3. **C2 rider**（§20.1 接口规约即移交凭据）：remote-read 计价增执行端
   HBM 写腿；落地前 C4 对拍/C20 消融按过渡态披露口径解读。

## 21. 主 agent 事故登记（/tmp 证据区误删与恢复处置，2026-09-22）

- 事故：C16 子 agent 在验证 domain_metrics 的"非 joint run 静默跳过"分支时误执行
  shutil.rmtree(tempfile 根的父目录)（rd.parent == /tmp），/tmp 大部被删。
  受损 = 过程证据区 /tmp/joint_exec（各已完成卡 DONE 摘要、sha1 法证、冒烟证据、
  C0 基线三件、C10 终冻报告原件）。**仓内交付物零受影响**；事故详情 =
  /tmp/joint_exec/INCIDENT-C16-20260922.md（C16 落）。
- 恢复处置（主 agent，同日）：重建 /tmp/joint_exec/RECOVERY.md——含 C0 仓内
  git status 20 条逐字重建（C19 残留判定比对面）、关键 sha1 抢救登记（金值两
  文件/run_online_strategy.sh 等）、各已完成卡交付物清单登记（替代已失 DONE 原件）。
  C10 终冻报告要义已固化于执行计划 §4.3 补遗 A3'（先于事故落字，未受损）。
- 义务变更：C19 的"/tmp DONE 汇总并入 PROVENANCE 索引"对已失文件以 RECOVERY.md
  登记为准履行；法证归因自此以仓内 diff + RECOVERY.md 为准。
- 教训登记（对齐 Agents.md 子 Agent 编排规范的过程纪律）：子 agent 的删除类操作
  必须先解析目标绝对路径并断言其位于 /tmp/joint_exec/<本卡号>/ 之下方可执行；
  本事故即"未断言父子目录关系"的直接后果。
## 22. C16 域三口径度量工具批（WP6b domain_metrics.py + A3' δ 敏感性重定价，2026-09-22）

**对象**：`sh_test_mesh/slo_tools/domain_metrics.py`（新增）+ driver Sink 扇出接入
（`slo_postprocess_driver.py`，kv → load → hop → **domain** → watermark，watermark
仍最后）+ `run_scripts/run_slo_postprocess.sh` 步骤注释 + `slo_tools/README.md` 工具
表行与专节 + `slo_params_manifest.json` 两参数（B 类唯一来源）。测试 =
`slo_tools/tests/test_domain_metrics.py`（19 例零后端，合成 C5-schema fixture）。

### 22.1 三口径冻结落字（设计文档 §5.2，F13；消费 C5 §19 schema）

- **D_feed(q,p,s)** = 决策日志中 remote-read 候选 applicable 的实例集合（准入
  计价时已声明供给/启动/缓冲条件判过可行）；逐成员标注本地参考（min-cost
  stay）与性能容忍度（C_remote−C_stay_ref，绝对 ns + 比值）。
- **D_econ(q,s)**：`C_alt* = min C(e,a)` 遍历**全部实例的适用非 remote 动作
  （必含 copy）**；`D_econ = {e: C_remote(e) ≤ C_alt*+δ}`。**同位置动作比较**
  （remote vs 同实例最优非 remote）另列单列，口径不与全实例 C_alt* 混。
- **实际选择与 measured**：在线仅最终 (e,a)；四动作选中计数 recompute 仅
  elected、forced（no_history/evicted_permanent）单列（§19.2 口径，下游不混
  报）；聚合窗口 = 决策日志首准入 tick 至末事件 tick，随 summary 落字、不
  解释为瞬时能力。**measured 集合 = NA**（独立同状态执行不在本产物——C4b
  受控测量三表），预测/实测差仅作诊断分列。
- **双域同图**：D_econ 等值线侧 vs 配额准入域侧（两域之差即配额作用量）；
  配额域数据源 = C5 port_snapshot/flow_snapshot，**WP3 前恒 NA——工具兼容
  NA 并如实标注，判据定义归 WP3 后续（本工具不发明）**。
- **叙事纪律内置（输出注释落字）**：饱和下域由配额关闭而非 argmin 涌现，
  "涌现域"表述限定配额界内；域是解释与可选准入分析对象，不裁剪任何动作/
  实例候选。

### 22.2 ε/δ 登记（F13：含义、单位、来源实验前固定，不据结果扩大平局带）

- **δ 主臂 = 0（单位 ns，与 cost_ns 同域）**：来源 = C10 δ_adm 终冻（维持 0，
  被否决来源 I/有效带宽=重复计费已登记于 C10 报告）；离线 D_econ 与在线决策
  同 δ 同值（manifest `domain_delta_adm_ns`，B 类 fail-closed）。**双口径分
  列**：集合口径 ≤ C_alt*+δ；决策路径口径（C10 challenger_flips 同构）
  C_remote+δ ≤ C_alt*（等号成立即翻转——"胜 ≥ δ"）。
- **δ 敏感性臂 = {σ̂, 2σ̂, 4σ̂}（A3' 强制修订）**：原"对拍臂 δ_adm=0"撤销
  （C10 终冻=0 后与主臂同构空转）；σ̂ = 在线预测噪声尺度，三级链——A 显式
  误差披露字段（C5/C15 schema 现无，键名清单留位）→ B 观测 |pred−measured|
  p50（pred=joint_cost_ns（merge_done 终点预测，F11）；measured=completion 行
  tick−准入 tick；**上偏警示落字**：含排队/状态演化共同项，非候选间差分噪
  声，差分量级归 C4b）→ C 冷启动锚点 = breakdown.merge_ns p50（I_merge_leg/
  B_eff 前向合并腿量级族，JCM:903-911，C10 指定替身身份披露）。三档仅离线
  诊断：不进任何在线决策路径、不据其修改 δ_adm 终冻、不据结果扩大平局带
  （manifest `domain_delta_sensitivity_multipliers` = [1,2,4]）。
- **ε = 0.01（ratio）= SLO violation 口径**（manifest 既有 `epsilon`，QoServe
  容量门槛族）：本工具计算不使用——F13"ε/δ 含义、单位、来源实验前固定"的
  登记口径，非域计算参数。

### 22.3 边界对拍口径（诊断不是验收）

- remote/local 分界（观测 = max{hops(e): C_remote(e) ≤ C_stay_ref}，内等值线
  半径的 hop 投影）vs d ≤ ρ 解析锚点（ρ = B_D2D/B_HBM 逐配置硬件派生，
  trace_config hardware_config → hardware json）——**仅历史扫描项平价、不保
  证完整请求平价**；偏差本身是模型诊断结果，不要求逐点重合。
- remote/copy 分界**不设单变量理论线**；容量静态参考 `V_copy≈[H+I+b_c−F]+`/
  `V_remote≈[I+b_r−F]+`（设计文档 §3.3）——**b_c/b_r 未随 schema 披露 → 锚点
  整体 NA**，组件 H/I/F 逐请求分列（coef=2·layers·hidden·bytes_per_elem；
  F=选中实例 load_view 逐 rank 最小值）。
- **无 n* = d′/(d−ρ) 式**（v2.2 已删除）。

### 22.4 home 迁移轨迹（kv_delta_journal 消费面）

- completion 行（C3b 决策时刻披露）：merge 方向/字节/home 翻转/session 轨迹
  （是否减少乒乓由结果判断，§5.3）。
- kv_delta_journal（C14 §20.3 结算时刻披露，字段含 home_before/home_after/
  home_migration/zero_byte_flip/两侧实际保留量）：**读取器已就绪**（sidecar
  键位 = bridge/joint_kv_ledgers.json 的 `kv_delta_journal` 列表；驻留时间以
  trigger_request_id 联 completion tick 近似并披露）；**run 级序列化一行未落**
  ——`online_service.dump_joint_kv_ledgers` 不在本卡文件清单，按 C13(a)/
  C14 §20.6 登记归 C19 收口（或偏差批准后顺带），产物出现前该块 status=TODO。

### 22.5 非 joint run 静默合同（字节对拍契约保持）

domain 步骤仅当 decision log 含 joint_admission 行时执行并落 run/ok/FAIL 行
与 slo_domain_* 三产物；非 joint run（四仓共链 fixture）**完全静默**（0 行、
0 产物）——test_driver_parity 的 sh_1.0 fixture 复跑实证：log 域行 0、run 行
9、domain 产物空（该测试的 2 个预存红 = 旧链复刻内 load_imbalance exit 2，
断言在旧链侧即失败、与本卡无因果；改前改后同签名）。离线重定价平局裁决 =
SH ACTION_ORDER（stay,recompute,copy,remote-read）偏好，δ=0 谓词重放与在线
argmin 一致（不一致计数单列披露）。

### 22.6 验收记录（含 2026-09-22 /tmp 事故影响，详见 §21/RECOVERY.md）

- **真 run 验证（事故前完成，对象 = C5 冒烟产物 /tmp/joint_exec/C5/run）**：
  21 条 joint_admission 全指标跑通——四动作计数 {stay:20, copy:1}（与 C5
  §19.4 核验一致）、recompute elected/forced 均 0；D_feed size p50=8/max=8/
  mean≈6.1（128 成员，turn-0 请求 0）、hop 分位 p50=5/p90=8/max=10、方向分布
  8 向（SE 31/S 27/SW 23/E 18/W 14/N·NE·NW 各 5）；D_econ(δ=0) 全空（remote
  无严格胜出）、敏感档成员 128；σ̂=9.34e9 ns（B 级观测，|err| p90=3.86e10，
  merge 腿冷启动锚点 7.9e4 对照——5 个量级差本身即"观测误差被排队/状态演化
  主导"的诊断结论）；ρ=2.469512、观测分界 n=1@hop8（deviation +6，远超锚点
  ——重载 home 下 remote 相对 stay 变廉的个例）；瓶颈 128/128=compute_ns；
  约束原因 stay"history not resident"128/remote"no resident"45+"base
  resident"16/copy"no history"61；home 迁移 1 次（session_1 home 1→5，
  reverse 合并）；replay_mismatch@δ0=0。**该 run 原件已随事故丢失**（§21），
  上列数字为事故前实测记录。
- **替代验收（主 agent 裁定）**：C4 卡 2s 冒烟产物（/tmp/joint_exec/C4/run）
  为权威替代——本卡收尾时 C4 状态见 DONE 注记（若仍未 DONE，run 级复验留
  TODO，G5 收口时以最新冒烟产物补验，验收责任不丢）；合成 C5-schema
  fixture 19 例单测全绿 + 非 joint 静默实证承担机制面验证。
  joint_smoke_evidence/TJE* 系 C5 前旧 schema，仅作解析兼容对照（其
  joint_admission 行无 C5 扩展字段，域指标对其输出空集+说明，不误读）。
- **回归**：slo_tools/tests 138 ran（基线 119 + 本卡 19），失败集与基线逐签
  名相同（2 parity 预存红 + 1 contract 预存 error + 1 skip），零新增红。

### 22.7 移交

1. **C19**：kv_delta_journal/copy_handoff_events 的 dump_joint_kv_ledgers 序列
   化一行（§22.4）；本卡无新增运行开关（两 manifest 参数非运行开关，开关
   清单无需增补）。
2. **WP3 后（C11）**：port_snapshot/flow_snapshot 接通后 quota_admissible 列
   自动携带实测值，配额域判据定义归 WP3 后续卡。
3. **C4b**：σ̂ 差分噪声量级的受控测量；measured 口径三表落地后
   `calibers.measured` 从 NA 升级。


## 23. C2 HbmPortFlowRegistry + u_port 接线 + breakdown 物化 + A4' 补价 rider 批（WP1b，2026-09-22）

（章节号注：派发时预定"接 §22"，C16 已按 flock 顺序先落 §22，本批实际
序号 = §23。）

**交付物**：新增 `WL/joint/hbm_port_flow_registry.py`（HbmPortFlowRegistry
＋ HbmPortFlowError）；M `WL/joint/joint_cost_model.py`（消费 u_port +
ActionCostBreakdown 全 11 字段物化 + E4 去重 + A4' 补价 rider）；M
`WL/online/sh30_online_scheduler.py`（登记钩子 + notes 通道）；新增
`WL/joint/test_hbm_port_flow_registry.py`（零后端 24 用例）。

### 23.1 注册表语义与登记覆盖（F4）

- `u_port(port) = 该端口所属实例活跃 decode 的 KV 消费流数（因果负载
  视图派生——SH provider 读 `len(state.active_decode)` 经 rank→instance
  映射，闲置为 0，不假设恒 1）＋ 在册指向该端口的传输/远读流数`。API 与
  LinkFlowRegistry 同构：`register(port_id, owner)/unregister(port_id,
  owner)/divisor(port_id)/snapshot()`＋`release_owner(owner)`（幂等空放）
  与 `leaked_owners()`（漏释放审计位）；owner 沿用 SH 既有字符串约定。
  `divisor` 返回在册他流数（不含候选自身 +1——JCM `_shard_leg_ns` 显式
  加，与 LinkFlowRegistry 的 self_overlap 分离口径一致）。
- SH 登记点：`_register_transfer_flows`/`_release_transfer_flows` 统一
  扩展（准入在途流、逐切片 `rid#decode#{j}`、drain 边界 `rid#decode`、
  merge watch `rid#merge`、停滞旁路逐出五类调用点全部覆盖；释放复用既有
  四点，时序不变——C14 `_on_merge_done` 的 watch⇔journal 闭合门仍在
  注销之前）。noc_migrate 逐 shard 登记**双端点端口**：源端口（home 读
  腿）恒有；目标端口（exec 写腿）对 copy 留存写 / merge 落点写 /
  remote-read credit 到达写恒有（rider 后计价与执行同腿型）；中间跳
  rank 不登记。池路径（remote_load/remote_store）不进本表（F3：池端口
  口径由 _PoolPortRegistry 独占）。
- `__new__` 测试替身兼容：`_hbm_ports` 经 getattr 软兼容（C14
  kv_delta_find 软门同款先例；真 `__init__` 恒在场——C3b/C5/C14 等
  12 个既有替身零改动全绿）。

### 23.2 JCM 侧

- **端点腿消费**：`_shard_endpoint_divisors` 自本批取真实 u_port（未注入
  注册表时 u=0 = C1 离线口径不变）；`_joint_cost_model` 构造注入
  `hbm_port_registry=self._hbm_ports`。
- **A4' 补价 rider（§20.1 规约逐条落地；本批生效后 §20.1 过渡态披露
  作废）**：`kind="read"` 增执行端 HBM 写腿——公式与 copy/staging 第三
  腿同式同源（`shard_bytes(已含 read_passes 倍乘) / (B_HBM_exec/
  (u_exec+1))`，u_exec 自 hbm_port_registry）；首 credit 腿按
  credit_steps 倍乘、其余流送段按 (read_passes − credit_steps) 倍乘
  （与既有字节基数同切法，调用点字节向量不变）；仅增带宽/服务争用腿，
  F14 容量半边不动（credit 在途字节仍不占执行端 HBM 容量、暂存归还仍
  无操作、eviction_wait 不受 u_port 影响）；镜像既有 COMM_WRITE 服务
  类、未新增类别、未复活 STAGING_WRITE、零 C++ 改动。**实现形态 =
  §20.1 之 (i) 的直接替换形态**：kind="read" 无条件三腿（无开关无旗标，
  Agents.md 直接替换规范）；kind="merge" 维持 C1 冻结两腿（落点空间
  准备在 _eviction_wait_estimate 单列计价）。
- **E4 去重（A1 勘误）**：删除 breakdown 披露处对 candidate_paths 的
  第二次 `divisor_multi` 调用——新增 `_union_noc_divisor` 助手，copy
  前缀腿与 remote 读流族（全流/首 credit/其余流送三处腿，同路径集同
  非零过滤）共享一次并集除数计算，`contention_divisor` 复用之；
  stay/recompute 无计价腿可复用时维持单次调用（原本也恰一次）。单测
  锚定：remote 估计 divisor_multi 调用 4→1、copy 2→1、stay/recompute
  恒 1。
- **breakdown 物化**：estimate_action 对全部 applicable 候选返回全 11
  字段非 None（值可为 0）；C5 冻结的 SH 序列化（字段序常量
  `_JOINT_BREAKDOWN_LOG_FIELDS`）零改动即得实测值。
- **u_port 分解披露**：`u_port_home=r{rank}:{active}act+{flows}fl` /
  `u_port_exec=...` 两条 note 走候选 breakdown.notes（既有 notes 通道，
  C5 schema 不动）；正式字段位 port_snapshot 的接通归 C11 步骤 6（本批
  port_snapshot 仍 NA）。

### 23.3 保真边界披露（F4，冻结口径落字）

u_port 不含权重/激活读——短上下文权重受限区间端口模型**系统性乐观**
（执行端权重驻留下逐层前向的 HBM 读流未入除数）；KV-bound 区间（remote
相关区间）近似正确。端口无遥测，注册表即执行器（登记/负载视图派生即
真值；遥测校正登记为可选后续）。池路径端点不经本表（F3 池端口口径，
池↔实例链路段未计价的既有披露不变）。

### 23.4 测试与冒烟证据

- **新增**：`joint/test_hbm_port_flow_registry.py` 零后端 24 用例——
  登记/注销配对 + 失败注入三态（双释放 fail-closed/漏释放 leaked_
  owners 可见/预登记泄漏检出）+ u_port 数值（2 活跃 decode + 1 在册流
  → u_port=3、计价除数 +1=4；闲置 0）+ snapshot 字段名与 C5 port_
  snapshot 逐实例分解同名 + breakdown 11 字段全非 None + rider 四例
  （u_exec=0 腿值在场/u_exec=3 按 4 因子放大且成 max 腿/首 credit 与
  其余流送倍乘切法/容量半边不动）+ E4 去重四例 + SH 接线四例
  （noc_migrate 双端点登记/池路径不进表/释放幂等/provider rank 派生）。
- **回归**：joint/test_joint_shard_pricing.py 22 绿（C1 锚不漂——u=0
  时 exec 写腿与 home 读腿同值，max 不变）；joint/ 全目录 7 红 + 276
  绿（7 红 = C1 金值 5 + C13 跨车道 2，与本批零因果、数量与改前一致；
  rider 对金值的影响并入既有 C4 重推导范围——本批使 remote-read 金值
  再增执行端写腿分量，C4 重推导排在 rider 之后的要求由此满足）；
  online/ 155+7subtests、face 48+2、kv 2、copy_handoff 18、settlement
  14 全绿（与基线逐数一致）。
- **冒烟**（2s 窗/21 请求/--combo TJE/1 次 launch）：run
  `/tmp/joint_exec/C2/run` exit 0 PASS，21/21 completed，e2e_p50 =
  529.681 ms；决策分布 {stay: 20, copy: 1} 与 C14 基线同形。
  **u_port>0 样本**（notes 通道）：`session_3_request_1` 的 copy/remote
  候选 `u_port_exec=r20:2act+0fl`（决策时刻实例 0 端口 2 条活跃 decode
  消费流）等 8+ 条；本窗决策时刻在册传输流恒 0（轻载窗口，flows 分量
  未上场——活跃 decode 分量即活证据）。**决策漂移如实记录**：
  session_1_request_1 的 copy 选中实例 0→5 翻转——实例 0 端口 1 条活跃
  decode（u_port_exec=r20:1act）使 copy@0 端点腿放大 (1+1)、代价
  3.08e9 反超 copy@5 的 2.70e9——u_port 首次真实参与 argmin 的实证；
  e2e_p50 逐位不变（p50 由 20 个 stay 主导）。跑毕 clean_test_records
  过并核验（指针=占位、traces/ 仅物化器、无 generated/、build/ 未触碰）。
- **登记（后续卡）**：`test_joint_shard_pricing.py::ThreeLegMaxTest::
  test_read_kind_has_no_exec_write_leg` 的用例名/注释自本批起语义过时
  （read 已有 exec 写腿；断言值仍绿——u=0 时两腿同值），该文件不在 C2
  清单内未动，留给 C4 金值重推导批顺带刷新注释。

### 23.5 法证

/tmp/joint_exec/C2/：baseline_sha1.txt（改前三文件 sha1：
JCM=1da49f5a…、SH=800b4dc3…、PROVENANCE=bd0251fe…）、queue_2s.csv.*、
trace_config.orig.csv、plan_materialize.log、smoke_run.log、run/（含
results/online_decision_log.jsonl——u_port 证据源）、prov_append.md。
## 24. C4 金值重推导 + 基线回归 + 2s 受控对拍批（WP1d，Lane-JCM 收口，2026-09-22）

（S1 收官卡；前驱 C1/C2/C3/C3b/C13/C14 全 DONE。改前 sha1 自落
/tmp/joint_exec/C4/baseline_sha1.txt：credit_pricing=c71c1df1…（= C0
金值锚未动）、review3=54549689…（同）、review2=8aea7625…、JCM=
45734a7b…、shard_pricing=3c0c805d…、SH=a9748561…（改后一致——零
SH 改动）、PROVENANCE=20bc5393…。）

**交付物**：M `WL/joint/test_joint_credit_pricing.py`（8 测试金值重推
导）；M `WL/joint/test_joint_review3_fixes.py`（CopyCompositePricingTest
3 测试手算注释按现行公式重写 + CompositeCopyMergeRegressionTest C13
triage）；M `WL/joint/test_joint_review2_fixes.py`（MetricsInstanceFilter
Test C13 triage）；M `WL/joint/joint_cost_model.py`（**仅 G2 术语
rider**：模块 docstring :6-9 与 estimate_action docstring 两处
"service_done" 旧术语 → merge_done 终点表述，语义零变更——§15.3 移交
凭据落地，joint_scheduler.py:9 随批术语归主 agent 裁量后继）；M
`WL/joint/test_joint_shard_pricing.py`（§23.4 登记：`test_read_kind_
has_no_exec_write_leg` 更名 `test_read_kind_three_legs_exec_write_leg_
at_u0` + 注释刷新——A4' 后 read 恒三腿，u=0 时 exec 写腿与 home 读腿
同值故断言锚不漂，增 read==copy 同值显式断言）。

### 24.1 金值重推导（不静默改期望；公式变更来源逐项列明）

改前 joint/ 7 红 = C1 金值 5（credit_pricing）+ C13 跨车道 2，全部收口。
手算过程逐字写进测试注释；四/五项公式变更来源：

| 来源 | 对金值的作用 |
| --- | --- |
| C1 三腿 min（F1） | 读流/前缀腿 wall = startup + 跳数×时延 + max(noc, home_read, exec_write)，逐 shard 取 max；旧聚合串行口径废除 |
| A1' 并集瓶颈除数（§4.3 补遗） | 2 TP shard 共链 → divisor_multi 并集除数 = 2（noc 有效率减半）；字节均衡 + 空链路下与旧聚合同 wall（桥接锚——review3 CopyComposite 闭式期望因此逐位保留） |
| D1 读放大因果（C1 步骤 2b） | 远读基数 = 仅远端基础历史（input/增量本地读）；read_passes = prefill 遍数 + decode 步数 = 1+10 = 11（A2' 缺省单遍计入基数，§24.4）——旧基数 = 终态上下文 2060 B × 10 遍 |
| C3 上下文依赖 | 夹具参数下 compute 同值（70/550 两锚均不变），注释列明 |
| A4' 补价（C2 rider，§20.1） | kind="read" 增执行端 HBM 写腿（u=0 时与源端读腿同值、不触 max；离线夹具不改变腿值，真实参与 argmin 见 §23.4 冒烟） |

- credit_pricing 新金值（手算 = 模型逐位一致）：ANCHOR_REMOTE_READ_NS
  2070→**2210**、FIRST_CREDIT 422→**410**、STREAM 1648→**1800**、
  auto-K 全链 cost 3286→**3426**、单块退化 cost 3356→**3496**（K=
  read_passes 显式档 "10"→"11"）、compute-bound 628→**616**（first 62→
  50、stream 208→180）、adaptive K 披露 (1,1,2,13)→**(1,2,2,13)**
  （read_passes 前移 +1）。8/8 绿。
- review3 CopyCompositePricingTest 3 测试：期望值**逐位不变**（1110/
  610/1700）——A1' 桥接锚（逐 shard 3000/(10/2)=600 ≡ 聚合 6000/10=
  600；端点腿 30 < noc 腿不触 max），手算按现行三腿+并集除数公式重写
  进注释，类 docstring 落桥接说明。
- **C13 跨车道 2 红 triage（判定 = 预期语义变更，非缺陷）**：
  - review3 `CompositeCopyMergeRegressionTest::test_merge_after_composite_
    copy`：失败形态 = merge 时刻 home 剩余回升 (0,0) ≠ 期望 base_prefix
    (640,640)。依据 = C13 §18.1 源端立即释放——home 基础前缀释放点自
    merge 前移至交接块到达（expand_prefill 的 prefill_drain 结算
    `_settle_copy_handoffs`→`_metrics_prefix_release...`/`_remove_local_
    shards` 物理对）。断言对象刷新：expand 内回升 = base_prefix、
    merge 时刻增量 = (0,0)（不重复释放已交接源块）、全程总释放量守恒
    = base_prefix（探针实测留 /tmp/joint_exec/C4/probe_c13_review3.py
    输出）；零传输/LOCAL@exec/home 迁移/版本键断言原样保留。
  - review2 `MetricsInstanceFilterTest::test_suffix_evict_parts_filters_
    by_instance`：失败形态 = expand 后 parts 实例集 {1} ≠ 期望 {0,1}。
    依据同上（C13 metrics 前缀镜像释放前移）。断言时点刷新：两实例并
    存探测移至 prepare 后（{0,1} 成立）、实例过滤语义（exec parts 逐
    字节不动/home 截断）原断言保留，增 C13 语义披露断言（expand 后
    home parts 已被逐块精确释放 → {1}）。

### 24.2 基线全量回归（计数增量归因）

- 窄命令（`cd sh_test_mesh && python3 -m pytest workload/llama2_7b_
  inference -q`）：**488 passed + 9 subtests 全绿**（C0 基线 279+9 →
  +209 = C1 shard_pricing 22 + C3 decode_context 5 + C3b arrival_contract
  11 + C5 decision_schema 11 + C9 link_quota 66 + C10 quota_stability 20
  + C13 copy_handoff 18 + C14 remote_settlement 14 + C2 hbm_port_flow_
  registry 24 + C17 face_static 18；逐卡闭合无残差）。
- 全量命令（README §6 原文口径，含 slo_tools 三 --ignore）：**594
  passed + 1 skipped + 11 subtests 全绿**（C0/README 基线 366+1+11 →
  +228 = 上述 209 + C16 domain_metrics 19（slo_tools/tests，§22.6））。
- joint/ 目录 **283 passed 0 红**（改前 276+7）。

### 24.3 冒烟与 2s 受控对拍（保真度测量；报告 =
/tmp/joint_exec/C4/parity_report.md，结构化 parity_report.json/_rows.json）

- 冒烟（2s 窗/21 请求/--combo TJE/1 次 launch）：run
  `/tmp/joint_exec/C4/run` exit 0 PASS，21/21 completed，e2e_p50 =
  529.681 ms（与 C2/C14/C3b 逐位一致——本卡测试/注释/术语改动零行为
  漂移实证）；决策分布 {stay: 20, copy: 1} 与 C2 同形；C2 记录的
  copy@0→copy@5 u_port 翻转保持（copy@0 = 3.079e9 被 r20:1act 端点腿
  放大 vs copy@5 = 2.700e9），**无新增漂移**（不调参）。
- **受控对拍**（预测 = 决策日志 C5 schema 落库值，实测 = completion
  tick − admission tick；本窗 0 merge watch 流 → merge_done 与
  service_done 同 tick，终点不对称 = 0）：状态重建经现行 JCM 原代码
  重放 506 项 applicable 候选（stay 61/recompute 189/copy 128/
  remote-read 128）七传输段逐位零失配（闸门）；旧口径 = C1 前聚合
  字节 ÷ 单链路率闭式（脚本内实现）并排。**逐动作 MAE（ns）**：
  stay n=20 new=old=18,563,496,079（无传输腿恒等）；copy n=1
  new=1,993,429,964 < old=1,994,200,647（新口径优 770,683——六 shard
  双列并流并集除数 D=3 → 逐 shard 1350 B/ns，旧聚合 675 B/ns 串行
  高估）；合计 new=17,774,445,311 < old=17,774,482,011。**排序错误率
  （新旧 argmin 对拍）= 0/21（0% 翻转）**；真值排序错误率不可测
  （在线 run 未选动作无同状态实测——归 C4b 证据等级 2）。**判定 =
  不劣于现状（成立）**：每个可测样本 MAE_new ≤ MAE_old。**强制夹具
  披露（强制 ≠ 自主选中）**：remote-read（128 评估，new 均值比 old
  低 99.7e6 ns/候选——D1 基数收缩主导）与 recompute（189 评估，
  new ≡ old）窗口内零自然选中，预测侧并排披露、**实测 NA**（SH 无
  强制动作开关、增开关超本卡清单；同状态四动作受控测量归 C4b）。
- 绝对偏差如实归因：stay 主导的系统性高估（measured/prediction
  p50=0.085）= roofline 串行台账（target_wait + 单请求串行 compute）
  vs 批处理/流水线执行器，**新旧共有**（对拍隔离传输腿族后恒等）；
  最欠预测样本 session_3_request_4（5.19×）= 因果时域估计低估真实
  decode 5951（估计器状态依赖，新旧同值）——均非本批公式族引入。

### 24.4 A2' rider 结论（prefill 分块重扫存在性验证）

证据链（详 /tmp/joint_exec/C4/a2p_prefill_scan_verification.md）：
128 项 remote-read 候选披露 prefill_scan_passes=1；SH 读计划
steps = max(1, decode_length)（decode 相独占，prepare 相基础前缀留
home 零传输）；GB credit 拓扑块 1 主链 + 尾块支链各恰一次发射、无
重扫节点族；SH_FIRST_TOKEN_SPLIT 缺省 "0" 关；prefill_chunk_size=512
系计算节点分块、不产生 KV 传输。**结论 = 执行器不存在 prefill 分块
重扫，缺省单遍即真值，A2' 闭合**——SH 条件授权（分块真值注入）不
触发，本卡零 SH 改动。边界披露：执行侧读计划字节基数 = S × f(终态)
（I1 冻结）与 D1 计价基数 (1+S) × 前缀的**字节**差异系 v1 既披露的
估计-计划边界，遍数两侧一致、不在 A2' 问题域。

### 24.5 法证与移交

- /tmp/joint_exec/C4/：baseline_sha1.txt、queue_2s.csv.*、trace_
  config.orig.csv、plan_materialize.log、smoke_run.log、clean_records.log、
  run/（**冒烟产物保留不删**——C16 §22.6 替代验收的权威对象）、
  parity_check.py、parity_report.{md,json}、parity_report_rows.json、
  a2p_prefill_scan_verification.md、probe_*.py（对拍/复现探针）、
  baseline_narrow.txt、baseline_full.txt、prov_append.md。
- 移交：①G1 门检查（本卡绿 = G1 全项就绪——C4 绿 ∧ C6 绿（§16.3）
  ∧ C5 schema 冻结落字（§19）∧ C12 P0 落字（§15）∧ C13 守恒绿
  （§18.3）∧ C14 结算闭合落字（§20.5）∧ C3b 到达合同绿（§17.3））；
  ②C4b：σ̂ 差分噪声受控测量 + 同状态四动作实测真值（本卡强制夹具
  的 measured NA 由其补齐）；③C16：run 级替代验收
  （/tmp/joint_exec/C4/run，本 DONE 即其 §22.6 待办注记）；④C19：
  joint_scheduler.py:9 "service_done" 字样随批术语对齐仍未做（主
  agent 裁量归 Lane-TOOL 后继——§15.3 原文，非本卡清单）。

## 25. C15 E 运行期事件递推预测器 + 逐层恢复与计算重叠批（Lane-FS，2026-09-22）

依据：设计文档《joint机制改造方案_局部统一内存域》§9（E 的运行期事件
递推为实现缺项）；仓库设计方案《三机制联合策略》§5.1/§5.2/§5.4/
§5.5/§5.6/§5.7；实验思路 §三.3；执行计划 F12/F9；C12 §15.5 登记
（GB 恢复门为后缀级保守门、逐层门控归 C15——GAPS G7 路由闭合）。
前驱 C13✓/C14✓。章节号勘误：卡文写"接 §23"系派发时快照——本批落笔
时 PROVENANCE 末节已为 §24（C2/C4 先后追加），按 append-only 取下一
空节号 §25。

### 25.1 交付物一：事件递推预测器（adaptive 正式在线实现，F12）

新增 `WL/joint/event_recursion_predictor.py`（纯函数 + 状态，
eviction_priority 范式；无 I/O、无环境读取、不改账本/事件队列）：

- **统一时间线**（`UnifiedTimeline`）：等权 N-way 均分流体事件递推
  （份额口径与 C1 `divisor_multi` / C2 `u_port` 同源：分母 = 已提交
  在册 + 活跃候选；流启动/完成/已提交 ETA 到期重算份额；分段恒速
  精确积分——推进时按各流当前速率扣减剩余字节，曾暴露并修复"推进
  不扣减"积分缺陷）。已提交流已知 ETA 占用至 ETA、未知 ETA 永久占用
  （状态标注 `unknown_release_eta`，不当零代价）。
- **召回方向**（`LayerRecursionPredictor`）：恢复腿逐组消费顺序、
  rank 内串行链 / rank 间并行（与 GB 逐组恢复支链同构）；恢复写腿与
  计算 memory 腿同端口仲裁（§5.4"恢复写入造成的 HBM 竞争不能遗漏"）
  ——候选自致计算减速单列 `compute_slowdown_ns`、期限不回移。
- **写回方向**（`predict_space_available`）：对象顺序逐 victim 决策、
  已定写回流登记进共享快照（同批 victim 不得各自按独享带宽估算——
  三 victim 共享池端口实测 1000/2000/3000 递增）；合流入口
  `predict_release_and_recall` 以 q̂ = max(首块基础等待, 空间可用时
  刻) 关键路径串联两方向。
- **k̂_hide 枚举与钉死剪枝**（§5.2 剪枝段逐条）：(i) 下界 =
  max_r [q̂ + Σ b_{j,r}/B_{j,r}^excl]，**逐腿**取路径资源 η·peak 最
  小值（rank 路径逐层不同时按全腿全局取 min 会在快腿上高估、破坏
  剪枝健全性——一致性抽检 50 例实测抓出并修复）；明确**不是**
  Σ_j max_r r̂_j 逐层近似（反例断言钉死：2 rank 交替大字节层，钉死
  下界 16.16 vs 逐层近似 32）。(ii) D̂_ℓ 钉在无候选恢复流量
  （reference 递推）的消费**开始**时刻（= 前层完成；修复过"取本层完
  成"的锚点错误——该错使 k=0 在 r=0.5ms 解析例被误判可行）。
  (iii) 解析不可行 ⇒ 递推必不可行（无漏剪，枚举自解析最小可行 k 起
  向上）；解析不可用（调用方显式声明 `analytic_min_k=None`）自 0 全
  枚举 + `analytic_unavailable` 状态 + fallback 来源。**不二分、不
  假设共享资源下单调**（prune vs 全枚举一致性抽检 50/50 一致）。
- **在线效率状态**（`ServiceFactorGroup`，§5.4 因果规则）：θ/α/τ/首
  样本初始化/无效样本拒绝全部复用 LEP `EwmaServiceFactor` 同规则；
  缺可分离样本 `mark_unobservable` 保持原估计并标记（不把受混合等
  待污染的墙钟当纯服务样本）；逐扫描点重置 = 新实例。
- **重试复核断言口径**（`revalidate_committed_plan`，§5.5 末段）：
  重交同一拆分 + 仅保护/合法性复核（层转在用/在途、区间有效）；
  快照新鲜度复核不在已提交计划面（函数不重推演、不读流量观测）。

### 25.2 交付物二：GB 逐层就绪门控（列车级保守门退役，G7 闭合）

`graph_batch_builder.py`：

- 新账本 `_suffix_restore_arms`（request_id → ((组号, 层区间, 逐
  rank 目标 HBM 写完成门), ...)）；准入批发射带 `restore_group` 标记
  的后缀恢复腿：同实例 partial 快速路径（驻留前缀屏障 + checkpoint
  分支保留）与跨实例（copy/remote-read 工作副本后缀）主链后旁挂支
  链两型，均逐组顺序发射（rank 内消费顺序串行链）、逐组逐 rank 门
  登记；**无 suffix p2p readiness barrier**——逐 rank 门控替换整段
  栅栏。旧单笔口径（无标记 remote_load）仍走 `_suffix_body_arms`
  整段门（回归锚，不构成生产路径开关——FS 全部 5 站点已逐组化）。
- 列车体层段门控（`_emit_train_body` + `emit_layer_segmented`）：
  含首 chunk 计算体按层段切分（热前缀段 + 各恢复组层段），每段一次
  `transformer_pass_aggregated`（layer_start/layer_end，跨段求和与
  整段单次发射**逐字节一致**——激活/KV/AR 按 Σ 段层跨度 × spans、
  权重按 passes × Σ 层跨度、final norm/logits 仅末段一次，测试钉
  死）；段 i 只等组 i 的恢复（逐 rank arm）——"全部冷层恢复完才开
  始 prefill"串行段不设。credit/copy 体块组合形态：首块层段化、块
  门由首段首节点消费、块 ≥ 1 原样（传递覆盖论证见代码注释）。
- completion 批残留恢复组门 fail-closed（C13 copy 同款纪律）。
- **跨车道修复披露（C13 先在缺陷，非 C15 引入）**：
  `_copy_handoff_body_blocks` 的 per_block 上取整可产生空尾块
  （end_iter < start_iter 直接 raise），crash 形 (it, tails) ∈
  {(4,2),(5,3),(6,3),(9,3),(6,4),(7,4),(8,4),(11,4),(12,4),(16,4)}
  ——stress 10s 窗实测 seq=1516 崩溃。C13 交付（2026-09-22，未提交
  工作树）起即存在（C13/C14 的 2s 冒烟窗无此列车形态故未暴露）；
  本卡以解析法+零后端复现钉死后修复：block_count 收紧至全覆盖所需
  最小块数，多出尾块经 min(tail_index, block_count) 并入末块（与
  C13"迭代数不足时尾块并入最后体块"披露语义一致；"体块完成 ⇒ 全部
  尾块到达"不变量保持）。回归测试 12 crash 形全绿。

### 25.3 交付物三：FS 恢复钩子与区间账本 + 预测器接线 + 披露侧车

`face_scheduler.py`：

- `KVTransfer.restore_group` 字段 + `RESTORE_GROUP_LAYERS=8` +
  `plan_suffix_restore_groups`（C13 同款确定性规划器；后缀 ≤ 8 层单
  组 = 旧单笔口径回归锚：transfers 数量/区间/字节不变仅加标记）。
- 5 个后缀恢复站点全部逐组化（stay@PARTIAL ×2 / copy 跨实例后缀 /
  copy REMOTE 基全量 / remote-read@PARTIAL 后缀），统一走
  `_plan_suffix_restore_transfers`。
- **区间账本**（`RestoreGroupJournal`，守恒科目 `rid#restore`
  issue/complete）：Σ issued = Σ in_flight + Σ consumed 逐 rank 精确
  fail-closed；结算边界 = prefill drain（`_expand_local_session`
  prefill 相，C13 同界）/ merge 兜底 / mark_complete 兜底（直连 API
  面不经 drain 的合法序列——boundary=mark_complete 落账披露，不静默
  抹账）；完成时未结算先兜底再校验闭合。
- **预测器接线（F12 身份）**：`_adaptive_retention_target_tokens`
  由解析在线模型直连替换为 `LayerRecursionPredictor`（同公式同因果
  边界：解析例 L=32/c=1ms/q=0/r=0.5|2|4ms → 1/17/25 两径一致已测；
  腿粒度 = 逐层，与 §5.2 公式同粒度——执行侧按 ≤8 层组门控，组内保
  守方向披露）。快照 = 池端口（pool_bandwidth + divisor 注入的在册
  他流，P1 口径 divisor−1）+ 目标 HBM 端口（local_hbm 带宽）峰值表；
  腿路径 = 池边缘 → 目标端口（nearest_edge 实拓扑）；计算段 memory
  腿 = 逐层 KV 读字节（与恢复写同端口仲裁）。无样本/缺池速率保守
  保留全 L（cold_start_unknown_input）不变；递推异常 fail-closed 保
  留全 L + `recursion_error` 披露（不当零代价、不删候选）。
- **η/γ 接入**：`ServiceFactorGroup` 实例挂 manager（逐扫描点重置 =
  新 manager）；`observe_valid_service_sample` 公开因果喂入通道；
  mark_complete 的墙钟含混合等待——默认按不可观测处理（保持原估计
  + 计数），不污染样本。**边界披露**：FS 可见的完成时刻不可分离排
  队/冷 KV 等待，故在线路径的默认姿态是"标记不可观测"（§5.4 明文合
  法态）；可分离样本的自动喂入通道（如逐列车服务时长侧车）留给
  C16/C19 序列化面（与 kv_delta_journal 同款登记）。
- **披露侧车（不改 SH 的通道）**：`adaptive_decisions` 逐决策落
  {k_target, source(recursion/fallback), statuses(coverage_* 等),
  analytic_min_k, pool_divisor, exposed_stall_ns, compute_slowdown_ns,
  η/γ 现值}；`adaptive_decision_find` 访问器（C14 kv_delta_journal
  同款范式）。run 级 sidecar 序列化归 C16/C19（既有登记）。

### 25.4 LEP 条件授权修改披露（逐条）

`layer_eviction_policy.py` 仅模块 docstring 状态段更新（零行为改动，
sha1 变更仅此）：登记"运行期事件递推预测器已交付（C15/F12）——
adaptive 正式在线实现直连 face_scheduler；k_hide_deadline 保留为
§5.2 闭式串行公式的单机实现（零后端解析例回归与对照锚共用），在线
路径不再直接调用；三模式身份不变"。`legacy_half` /
`minimal_layer_groups` 身份与行为零改动（tests 全绿佐证）。

### 25.5 adaptive 身份升级登记（F12 三态）

- **已设计**：§5（2026-09-21 版）——公式/递推双方向/剪枝口径/因果
  参数/决策落定与迟到不补救两项用户裁定。
- **已实现**：本批——递推预测器（召回×写回×合流 + 钉死剪枝 + η/γ
  因果更新 + 重试复核口径）+ GB 逐层就绪门控 + FS 逐组发射/区间账
  本/接线/披露侧车；无新增开关、无静默回退（旧解析在线模型被直连
  替换而非旗标隐藏；`k_hide_deadline` 保留为解析例回归锚非在线回退
  路径）。
- **已验证**：§5.7 清单逐项落测（25.6）+ 受控真实路径（25.7）；未
  验证项 = 在线 η/γ 可分离样本自动喂入（通道归 C16/C19，25.3 披露）。

### 25.6 零后端验证（§5.7 清单全落测）

新增 `joint/test_event_recursion_predictor.py`（25 用例）+
`online/test_joint_layer_restore.py`（17 用例）：

- 解析例回归：L=32/c=1ms/q=0/r=0.5/2/4ms → 1/17/25（两径一致，
  analytic_min_k == 闭式 k）；r=2ms 保留 16 层末层暴露 1ms（Ŝ=1000、
  binding=32）；k=0 首层阻塞、k=L 无缺失层（binding=None）、r=0 →
  k=0；首层赶上后层断供（迟层 2000ns > D̂_3=200ns ⇒ k=3、k=2 trace
  binding=层 3 如实入账）。
- **剪枝一致性抽检**：50 例随机竞争场景 prune vs 全枚举选层全一致
  （曾抓出逐腿速率缺陷并修复）。
- **未来信息扰动隔离**：未来行（到达/输出长度/返回间隔）不进任何
  FS 因果通道 ⇒ E 输出与估计器状态不变；已完成样本 ⇒ 输入估计/服务
  因子/保留目标允许更新（限保留目标估计与估计器）。
- **决策落定**：同缺口同 victim 下注入池端口他流（divisor 4）⇒ 已
  提交层数拆分不变（800/800 缺口两次 plan 同拆分）；保留目标可随观
  测更新（新决策）。
- 逐层 R/D 方向用例：长历史短输入 / 带宽高低（同字节峰值减半 ⇒ k
  增）/ 竞争升降（在册他流 ⇒ k 增 + unknown_eta 状态）各 ≥1 例；
  期限钉死（候选字节增大 ⇒ D̂ 不变、减速单列增大、无候选减速=0）。
- 层级重叠/写回后释放/接收缓冲与恢复增长守恒：GB 层段发射 vs 整段
  单次发射五类字节总量逐位相等；三 victim 写回 1000/2000/3000（同
  批不按独享带宽）；journal Σ issued = Σ in_flight + Σ consumed +
  issue 字节 = kv_cache_shard_bytes_for_layer_range 真值。
- **迟到不补救**：迟组门（链上更晚节点）⇒ 组层段父依赖含迟门（等待
  如实）、零 noc_migrate 路径切换节点、图侧零估计器状态。
- **重试复核**：仅背景流变 ⇒ RESUBMIT_SAME；层转在用/在途 ⇒
  INVALIDATE_LAYER_IN_USE；区间失效 ⇒ INVALIDATE_INTERVAL_RELEASED。
- 失败注入：腿乱序/未知资源/有字节无路径（缺模型事实 fail-closed，
  不当零代价）/残留恢复组门 completion fail-closed/双结算 fail-closed。
- 回归：test_face_scheduler 48+2sub、online 173+7sub（含本卡 17）、
  joint/ 308（含本卡 25）、kv_incremental 2 全绿——合计 581 passed
  + 9 subtests。joint/ 基线 7 红（C1 金值 5 + C13 跨车道 2）经并行
  C4 批转绿（非本卡所为，sha1 时间线可证）。

### 25.7 受控真实路径对拍 + 冒烟（4 launch 全记录）

1. **run_1（冒烟 #1）**：2s 窗 21 请求 / --combo TJE / exit 0 /
   21/21 completed / e2e_p50=529.681ms 与 C14 冒烟基线**逐位一致**
   ——轻载窗零 PARTIAL 恢复时 C15 改动行为不变的实证（旧单笔路径
   未触发的决策分布不变）。
2. **stress 第一次（废弃，跨车道竞态非本卡所致）**：与 C8 在飞 SH
   编辑竞态（Python 进程导入半写状态 SH——`_ingest_link_telemetry`
   AttributeError；SH mtime 与运行时刻交错可证），exit 134。
3. **stress 第二次（暴露 C13 先在缺陷，见 25.2 修复披露）**：exit
   134 @ seq1516 "copy handoff body block covers no iterations"。
4. **stress 第三次（对拍例，PASS）**：10s 窗 270 请求 / stress-28gib
   / TJE / run exit 0 + judge exit 0 / **partial_copy_hits=78**（真实
   后端驱动逐组后缀恢复路径 78 次：copy@PARTIAL 的恢复组支链发射 +
   层段门控消费全 GREEN，残留门零 fail-closed）/ 270/270 completed /
   deep_gap 空 / merge_degrade 空；slo_restore_decomposition（既有
   工具面）270 行：hidden_ratio p50=0.9999（逐层门控下恢复绝大部分
   隐藏）、exposed_stall p50=8872ns（首块等待如实入账）、max
   334.9ms（个别请求如实暴露——负结果保留）。
   **跨车道观察（非本卡缺陷，登记供 C8×C16 裁置）**：
   `hbm_watermark.py` FAIL-CLOSED——决策日志 seq1089 `link_
   telemetry_coverage` 行 request_id 为空（C8 在飞交付的决策行发射
   形态；hbm 工具按缺 request_id fail-closed）。fixture 缺省 warn 档
   不门禁，run 判定不受影响。

### 25.8 边界与移交

- 本卡未改 SH/JCM（C2 在飞占用）——决策日志正式字段的预测器来源披
  露经 FS 侧 `adaptive_decisions` 侧车通道实现（卡文授权路径）；
  若后续需进 decision-log 正式字段，由 SH 车道后继卡接线。
- η/γ 可分离服务样本的自动喂入通道（逐列车/逐层段服务时长侧车）
  归 C16/C19 序列化面；本批提供 `observe_valid_service_sample` 因果
  入口 + 不可观测默认姿态。
- 预测器 R̂/D̂ 与后端实测的逐请求数值级对拍（预测 vs
   slo_restore_decomposition 实测的 MAE 报告）超出本卡零后端+1 对拍
  授权面（需逐请求状态回放夹具），登记为 C4b 回归重跑的扩展输入。
- 冒烟/对拍毕 `clean_test_records.sh` 已还原并核验（占位指针 /
  traces 仅 *.py / generated 空 / build 未触碰）。
## 26. C4b-FIX remote-read auto-K 多块 credit 执行协议缺陷修复（Lane-FS，2026-09-22）

依据：执行计划 §4.3 补遗 A6'（偏差流程登记）；/tmp/joint_exec/C4b/
{BLOCKED.md,DONE} 缺陷全文与两族复现签名；C13 §"关键工程发现"（copy
体块 `_cb{k}` 唯一 phase 方案——本修复的同款手法）。前驱 C4b（缺陷
发现与资产）✓/C15（GB 现状树）✓。

### 26.1 根因（fail-closed abort，非机制缺口）

- 签名：`commit preflight: mandatory liveness preflight: collective
  operation batch_train_i2_<j>_all_layers_mlp_all_reduce of pg_name 3
  has 8 participants on rank 30 (expected exactly one)`（C++ 拒批）→
  次生 `ack_count != delivery_count`。
- C++ 语义（astra-sim/workload/execution_driven/GraphBatchCommitter.cc
  :634-659）：集体按 (pg_name, 节点名) 计参与者，同批同名集体在单
  rank 出现 N 次 = N participants，要求恰 1。
- 构造点：GB `_emit_train_body` 的 credit 体块发射——SH 规划侧
  `_remote_credit_body_blocks` 产出的体块**不带 phase 键**，发射时
  `phase=block.get("phase")` 缺省回落 train_id；auto-K（K =
  max(1,⌈S_j/8⌉)，S_j=列车参与步数）下 M=⌈S_j/K⌉>1 个体块各发一次
  `transformer_pass_aggregated`（generate_trace.py:1198/1207），产生
  M 份同名 `{train_id}_all_layers_{attention,mlp}_all_reduce`——
  C4b 两族臂 S_j=8/K=1/M=8，即"8 participants on rank 30"（rank 30
  ∈实例 2 组）。K≥S_j 单块（JOINT_REMOTE_CREDIT_ITERS=209）不触发
  ——v1 等价锚路径完整执行通过，与 C4b 观测一致。
- 历史为何未暴露：C13 落 copy 体块后缀时注记"remote-credit 多块在
  现行 auto-K 下恒单块"——该判断不成立（auto-K 的 K 恰为产生 ≤8 块
  的除数，S_j>1 即多块）；remote-read 在本仓此前零自然选中（轻载
  stay 支配），C4b 首次后端受控执行即暴露。零后端单测
  （test_remote_credit_stream 多块用例）只断言结构/门控，未镜像 C++
  同名计数谓词，故未拦截。

### 26.2 修复（GB 单文件，与 C13 copy 同款）

`WL/online/graph_batch_builder.py` `_emit_train_body`：credit 体块
发射循环内计算 `block_phase = block.get("phase")`；无显式 phase 且
`len(credit_blocks) > 1` 时逐块挂唯一后缀
`{train_id}_rcb{position+1}`（rcb = remote credit block），层段发射
（C15 restore 组合形态）与常规块发射两分支统一消费该值。规则：
- M>1：块 b 的聚合/集体节点名携带 `_rcb{b}` ——同名集体每 rank 恰
  1 参与者，预检合规；
- M=1：不加后缀——K≥S 单块 = v1 等价锚逐字节不变（I3a）；
- copy 体块自带 `_cb{k}` phase（C13）原样透传，零影响。

拓扑语义零改动：门控（I2 尾块 recv 完成门）、块 1 主链先行（T1
pd_transfer / T2+ per-rank 链序）、尾块旁挂支链不 join barrier、
字节计费（spans/weight_passes）均不变——仅节点名相位区分。

### 26.3 变更文件与测试

- 修改：`WL/online/graph_batch_builder.py`（上述；本卡唯一仓内代码
  修改面。FS 无需配套——结算/账本不解析集体节点名；SH 未触碰
  ——C11 在飞占用，修复面落 GB 即闭合，无 BLOCKED 事项）。
- 新增：`WL/online/test_remote_credit_multiblock.py`（5 用例：T1
  多块预检谓词+命名、I2 门控映射到块相位、T2+ 续坐多块、单块无
  后缀 v1 命名锚、C15 restore+多块组合命名唯一性。预检谓词 =
  C++ 参与者计数规则的零后端镜像 `(rank, pg_name, name)→count==1`）。
- 反向验证：临时还原修复→4/5 用例红（单块锚用例应绿）→恢复修复
  →全绿（/tmp/joint_exec/C4b-FIX/graph_batch_builder.fixed.py 为
  修复态备份）。
- 回归：online/ 全部测试 + joint/ 全部测试 + test_face_scheduler +
  test_sh30_kv_incremental_invariants 全绿；唯一红 =
  test_joint_quota_integration（SH `_quota_filter_candidates` 的
  `_dataclass_replace` 类型错——C11 在飞文件的既有红，与本卡无关、
  该测试零 GB 依赖，未修）。

### 26.4 后端受控复跑（C4b 资产复用：强制钩子 + 两族队列 + run_arm
编排改道 /tmp/joint_exec/C4b-FIX/；launch 账目 launch_ledger.tsv）

1. A-remote-read auto-K（2 请求夹具，S_j=8/K=1/M=8 形态）：**exit 0**
   （修复前同臂 exit 134）；2/2 completed、incomplete=0、phase-4 审计
   watch_registry_size=0/terminal 19890=19890、决策日志含
   readplan_reconcile+readplan_settle 正常收尾；目标请求全生命周期
   （prefill_drain→decode 完成）经 journal 核验，失败位点 rank 30
   所在实例 2 的 batch_train_i2_2（iterations=8）完整执行并结算。
2. B-remote-read auto-K（2s 全窗 21 请求）：**exit 0**（修复前 exit
   134）；21/21 completed、merge_done 行在案、readplan_settle 收尾。
3. B-remote-read-k209（v1 等价锚）：exit 0；对 C4b 修复前同臂产物
   逐位对照——request_metrics/normalized/raw_metrics/cache_events/
   kv_hit_states/全部 SLO csv/json（15 文件）+ request_journal +
   train_ledger **逐位相同**；决策日志剥离 C11 并行卡漂移（新增
   quota_mode 字段 + joint_decision_metrics 行类型）后 87 行逐位
   相同（0 mismatch）；online_stats 仅 host 计时字段差
   （scheduler_self_ns*/gil_wait_ns/processing_ns——C4b 已甄别口径）；
   invocation.json 仅 host 元数据差（路径/UTC/runner sha——并行卡
   改 run_scripts 所致）。
- launch 预算：3/3（另 1 次 plan 物化失败重试未起仿真、不计 launch）；
  每跑毕 clean_test_records.sh + 核验（3/3 verify=ok：指针=占位、
  traces 仅 *.py；本卡运行毕时点 generated/ 为空——其后并行卡新建
  generated/runtime_config 非本卡产物，未触碰）。

### 26.5 边界

- 本卡未改 SH/JCM/FS；多块形态的既有语义（读与 decode 重叠、首块
  主链、尾块旁挂、T2+ 链序）经零后端门控断言 + 两族后端完整结算
  双重核验保持。
- C4b 三表的 remote-read auto-K 真值补录（NA→实测）归 C4b 报告侧
  收口（c4b_report.md 属 /tmp 资产，本卡不改动其历史结论；复跑产物
  在 /tmp/joint_exec/C4b-FIX/runs/ 可直接取数）。


## 27. C11 SH 配额集成 + 开关落点 + 决策时延批（WP3c，Lane-SH 收口，2026-09-22）

依据：设计文档 §4.2/§4.4（配额与反馈控制语义权威）；执行计划 C11 卡
+ 冻结项 F2/F7/F8/D3/D4/D5；§4.3 补遗 A3'（δ_adm = 0 终冻）与 A5'（B3
键换算 rider 落 C11）；前驱 C8（遥测接入）/C9+C10（link_quota 模块与
AIMD 控制律）/C14（kv_delta_journal + 结算闭合门）/C13（copy_handoff
events 移交）/C17（face_static 交接）。

### 27.1 准入序列（步骤 1）

`_try_admit_request` 在 select_instance_and_action 之后逐候选执行配额
判据（`_quota_candidate_verdict`——link_quota 公共谓词的只读镜像，无试
探性借还、配额代数不受候选评估扰动）：stay/recompute 恒适用（无跨实例
主流程；准入相逐出支链按 oneshot 事后入册收紧后续判据，D4）；copy =
oneshot 链路门；remote-read = realtime 链路门 + 双端点端口平价门
（B_HBM/(u_port+1) ≥ r̂_KV）+ 双候选胜者侧 merge bulk 余量 + merge 预留
链路需求（dedup 1 槽/方向，与入册半同构——杜绝"判据过而入册拒"成为
常规路径）。判据不过只把该动作标记不适用（`inapplicable_reason` 携带
quota_link/quota_port 前缀与余量；cost_ns/breakdown 清空与 estimate_
action 不适用形态同构，C5 schema 零扩展）；配额可行集内重选 = argmin
同序——**δ_adm = 0 与既有比较合取恒等**（challenger_flips margin ≥ 0
同真域，单测钉死；不改值）。全动作不可行 ⇒ quota_deferred_requeue 回
队（永不 JointSchedulerError），重试键扩展配额代数（`_compose_admit_
failure_key` 单一裁决点：KV 纪元 ⊕ 失败候选集实例纪元 ⊕ 配额代数；容
量路径同键形——off 模式退化为既有纯容量键，门语义零漂移）。

### 27.2 端口/链路粒度裁定与流生命周期（C11 裁定披露）

- **端口粒度 = 实例级**（port_id = instance_index）：TP 全 rank 同账下
  逐 rank 端口 u_port ≡ 实例级流计数（每 TP 并行流对实例各 rank 端口
  各占 1 条），与 C5 port_snapshot 逐实例 schema 同构；C2 注册表
  （rank 粒度）在 port_snapshot 披露时逐实例取最忙 rank 口径。
- **流生命周期入册**：准入事务成功后 `_quota_enroll_admission`——
  evictions（全部动作，owner rid#evict）/ copy 主流程（rid + rid#src
  双端点 oneshot）/ remote-read 读流（rid#readplan + #src 双端点
  realtime，r̂_KV = 负载视图派生：活跃 decode 平均上下文的每 rank 每步
  KV 字节 ÷ roofline 单步时长，因果）+ merge 预留（reserve_merge 双向
  各 1 槽 + 双候选 bulk）。释放点与 C8/R15 流登记边界成对：drain =
  准入相（evict/copy）；完成 = decode 相增量（#decode#q{seq}，逐 grow
  事件独立 oneshot）+ 读流（流寿命样本双时标进 AIMD EWMA）；merge 预
  留见 27.3。收尾守恒审计 fail-closed（_quota_enrolled/_quota_merge_
  reserves 必空）。
- **实际合并流占用 = 胜者侧预留槽连续覆盖**（预留→实占转换裁定）：
  预留自准入持续持有至 merge_done，1 槽覆盖实际传输，无双重计数、无
  reserve→flow 换册间隙。
- **elastic 类无消费者披露**：现行 remote-read 执行口径（credit 交错
  流）无独立 prefill 首遍读流（prefill 输入增量本地生成），C9 冻结的
  elastic 流类别在 SH 侧暂无入册点——模块位保留，如实披露。

### 27.3 merge-reserve 释放时机（步骤 7，C14 移交联合断言）

service_done（REQUEST_COMPLETE 交付 tick，merge_back 落账后）
`_quota_on_service_done`：按 last_merge_outcome 方向裁决——winner ==
exec ⇒ forward，败者侧链路槽 + bulk 名额释放（adjudicate_merge_
direction）；零传输分支（stay/in_place/零字节翻转 ⇒ merge_transfers
空 ⇒ 无 merge watch）整体释放（release_merge 未裁决双侧撤销语义）。
`_on_merge_done` 中胜者侧释放排在 **C14 kv_delta_journal 结算闭合门
之后**（watch 交付 ⇔ journal 行在案 ⇒ 胜者侧释放必晚于结算落账——
联合断言落地）；到达而未裁决 = watch 通道与预留账目脱钩，fail-closed。

### 27.4 F7 耦合规则与开关落点（步骤 2/3）

`JOINT_QUOTA_MODE` 缺省 **off**（F7；off = tracker 不构造、全部配额调
用点短路——2s off 臂 e2e_p50 与 C14 基线逐位一致，决策行仅增
quota_mode 行键与收尾指标行）。aimd ⇒ 发射层自动注入 `--link-telemetry`：
joint_runner（--quota aimd 置位 SH_LINK_TELEMETRY=1 + invocation.json 记
link_telemetry_injected）→ run_online_strategy.sh（消费 env 追加 C++ 旗
标 + 防御性断言：aimd 而遥测未置位 fail-closed——不存在"aimd + 无遥
测"的合法启动路径，含直跑 .sh 绕过 runner 的路径）。`--scheduler`
choices 扩 face_static（C17 交接；joint_config 强制 quota-off 的唯一耦
合例外在解析侧已落地）。开关清单 §7.0 同步（三开关行 + face_static
取值域扩展 + fail-closed 补则）。

### 27.5 port_snapshot 接通（步骤 6）与指标埋点（步骤 4/5）

port_snapshot：配额 on = 实测值替换 NA（u_port 分解自 C2 注册表快照
逐实例最忙 rank 口径 + tracker bulk/平价门余量）；off 保持 C5 冻结 NA
占位（F8 不缺席语义 + 冻结测试兼容）。埋点：决策行 selected_action/
recompute_selection 同源的选中计数（recompute 仅 elected、forced 按
no_history/evicted_permanent 单列）+ quota 等待分列计数 + deferred 驻
留时长分布（min/p50/p90/max）+ 决策时延（纯决策段墙时 total/avg/max
+ 配额判据层墙时 + candidate_checks 计数）+ AIMD 动作计数 + 借还事件
计数，收尾行 kind=joint_decision_metrics（为 C20 预置，只埋点不判读；
remote-read 选中率按负载档由决策行 load_view 侧导出）。

### 27.6 AIMD 闭环（步骤 8）与 A5'/B3 rider

C8 `_ingest_link_telemetry` 产出的有效速率喂 tracker.observe_telemetry
（仅 aimd）：每流速率 = 链路实测速率 ÷ 在册流数，**无在册流的链路不
出现在字典**（C9/C10 冻结契约"无测量即无信号"）；实测含 collective
等 non-KV 流量（F6 披露），每流归因按在册流均摊（保守向披露）。收缩/
扩张经 set_link_quota 落地、bump 配额代数 ⇒ deferred 重试门重开。
δ_adm = 0 合取确认见 27.1（不改值）。**A5'/B3 rider**：`_ingest_link_
telemetry` 喂入前 LinkId → (src,dst) 端点键换算（C++ MultiDimTopology::
connect_dimension 确定性枚举；dims = network.yml npus-count =
[mesh_cols, mesh_rows]，C++ 平层 rank = col + mesh_cols×row ≡ Python
rank 空间——端点键与 LinkFlowRegistry/quota 链路键空间一致）——JCM
divisor_effective 的 max 合并自此在生产路径真实生效；未知 id（生产）
fail-closed，__new__ 替身（未装配换算表属性）按 C7 期退化路径整型透
传（C8 测试兼容）。C7 测试 34 例 + C8 测试 19 例双绿。

### 27.7 riders：manifest 侧车（C8-BLOCKED① 收口）与 copy_handoff（C13 移交）

online_service 窄域两行级增量：joint_mechanism_manifest.json 增
`link_telemetry_injected`（发射层注入事实，F7）+ run 末回填遥测完备性
五键（epoch/sample 计数、键缺席/窗口破损、collective_coverage 终值
——G2 门"翻转条件落 manifest"侧车半；决策日志收尾行为另一半，C8 已
交付）；joint_kv_ledgers.json 增 copy_handoff_events（C13 逐 chunk 四
步交接事件级计量，getattr 缺省兼容）。

### 27.8 测试与冒烟（含 30s 特批一次）

零后端：新增 online/test_joint_quota_integration.py 26 例（候选级判据/
quota_deferred 闭合与重试门/借还生命周期与守恒/AIMD 闭环/port_snapshot
两态/A5' 逐 id 键换算/F7 runner 注入 + .sh 防御断言）；回归 online/
222 passed（基线 191 + 本卡 26 + 并行卡增量）、joint/ 342 passed、
workload 根 50 passed 全绿（C7/C8 测试双绿）。

冒烟 4 launch（预算 ≤4 用 4；每跑毕 clean_test_records + 裸仓核验）：
- off-2s（21 请求）：exit 0；e2e_p50 529.681ms 与 C14 基线逐位一致
  （off 模式零漂移实证）；port_snapshot 全 NA 保持。
- static-2s：exit 0；决策行 quota_mode=static、port_snapshot 实测值
  （bulk cap=2/headroom 12856）；506 候选检查/4.27ms 判据层；quota
  admits=releases=2 配对；守恒审计过。
- aimd-10s（270 请求）：exit 0；F7 自动注入链全通（invocation
  injected=true → C++ "link telemetry: enabled, links=186"）；observer
  缺省关 ⇒ link_telemetry[] 空 ⇒ AIMD 无信号不动（合法 no-signal 路
  径；观测器开关 ASTRA_LINK_OBSERVER 与 F7 旗标耦合正交——数据源开关
  属 §5.3，未擅自扩耦合面，30s 臂以 observer=on 补通路验证）。
- aimd-30s（1177 请求，**用户 2026-09-22 特批仅此一次**：通路验证 +
  墙时入档、不取性能结论）：exit 0；整跑墙钟 117s；F7 注入 + observer
  on ⇒ 987,480 遥测样本/30,564 epoch/collective_coverage=true（A5'
  键换算生产路径零未知 id）；AIMD 闭环真实数据驱动：comfort 10,468 +
  expand 186（无 shrink——本窗速率恒 ≥ r̂_KV，合法）；752 借还全配对；
  **决策时延入档**：纯决策段 1177 次/总 4.131s/均值 3.510ms/最大
  9.031ms；配额判据层 29,706 检查/总 302.6ms（≈10.2µs/检查、≈0.26ms/
  决策，占决策段 ≈7%）；quota_deferred 本窗 0 次（轻载合法——判据通
  路由单测覆盖）；selected = stay 801 + copy 376（remote-read 自然选中
  0，合法结果）。

### 27.9 移交与披露

- 冒烟矩阵新臂（quota static/aimd 三臂）归 C19；README 开关族段落同步
  归 C19/G5（现落后 C6/C9/C17 多卡，开关清单 §7.0 为权威登记面）。
- off 臂 port_snapshot 保持 NA（配额族字段与开关共生态）——若未来需
  off 态也披露 u_port 分解，按偏差流程补遗（涉及 C5 冻结测试联动）。
- JCM（divisor_effective 消费端点形键）归 C7 已交付双形消费；C4 金值
  与本卡零耦合（判据不改计价）。

## 28. C19 总收口批（G5 门：矩阵 13 臂 + 全量终跑 + 文档同步 + 裸仓还原 + 残留判定，2026-09-22）

C19 = 全计划最后一张卡（19 张计划卡 + 1 张偏差修复卡 C4b-FIX 交付后的总收口；
G1–G4 门均已过）。本节为总收口登记 + §15–§27 批次索引 + 编号勘误说明。

### 28.1 总收口范围与 G5 判据对照

- 冒烟矩阵 3 新臂绿 → 28.2（13 臂一键全绿）；
- 三孪生 round-trip 过 → 28.4（C18 测试入仓重写，主 agent 已批准）；
- README/开关清单/PROVENANCE 同步 → 28.5 + 本节存在性；
- 静态臂可用（身份纪律落 manifest）→ 28.2（TJE_face_static 臂
  face_static_identity + quota_forced_off 注记实证）；
- D 三口径可导出 → §22（C16 已交付，收口无追加动作）；
- 裸仓还原核验过 → 28.6（clean 两脚本 + 四要素 + 残留判定）；
- 【条件项】指标锚迁移闭合（C3b 步骤 6）→ §17.2 已裁不触发（核查结论：
  指标不取自 FS mark_complete），无收口动作。

### 28.2 冒烟矩阵扩 3 臂（10 → 13，一键全绿）

`joint_smoke_matrix.sh` 新增 `run_one_runner`（与 run_one 同款证据链，经
`joint_runner.py` D14 入口起跑——开关证据 = invocation.json 的
joint_switches/quota_mode_env/link_telemetry_injected）与三臂：

- `TJE_quota_static`（--combo TJE --quota static）：exit 0；
  joint_admission 行 port_snapshot 实测值物化（C11 §27.5 接通的冒烟侧
  首证）；
- `TJE_quota_aimd`（--combo TJE --quota aimd）：exit 0；**F7 自动注入链
  全程生效实证**——invocation.json `link_telemetry_injected=true`（runner
  置位 SH_LINK_TELEMETRY=1，臂环境未显式设遥测变量）→ C++
  "link telemetry: enabled (--link-telemetry; links=186)"→ 决策日志收尾
  kind=link_telemetry_coverage 行在案；observer 缺省关 ⇒ 数组空 ⇒ AIMD
  无信号不动（合法 no-signal 路径，C11 aimd-10s 臂同款，30s 臂已补全链）；
- `TJE_face_static`（--scheduler face_static --quota static）：exit 0；
  manifest `face_static_identity` 键落字 + `quota_forced_off=true` /
  `quota_forced_off_note="face_static forces quota-off"`（F7 唯一耦合
  例外在对照臂的实证——env 配 static 被显式覆盖为 off，不拒绝启动）。

13 臂（八组合 + TJE_remote_off + TJE_shadow_verify + 三新臂）全部
exit=0，证据根 `/home/sunhao/joint_smoke_evidence_c19/`（2s 窗 21 请求，
astra_compute_20.csv 物化；每次跑毕矩阵自带指针还原，收尾统一
clean_test_records 核验）。**全臂无 slo_postprocess.FAIL**——同时是
28.3 裁置的 live 验证（三新臂决策日志均含 request_id 为空的
link_telemetry_coverage + joint_decision_metrics run 级行）。

### 28.3 C15 登记跨车道观察项裁置（hbm_watermark 对空 request_id 决策行）

- **诊断**：C15 stress 对拍实证（§25.7）——决策日志 run 级披露行
  （C8 `link_telemetry_coverage`、C11 `joint_decision_metrics`，
  request_id 恒空）在 `WatermarkScan.consume` 的 kind 门序（audit 跳过
  词表 → request_id 校验 → 未知 kind fail-closed）下先触"决策记录缺
  request_id"；同类还有带 request_id 但不在 `("prefill","decode",
  "completion")` 白名单的 C8/C11/C14 披露行（readplan_reconcile/
  readplan_settle/quota_oneshot_overflow/merge_done）触"未知 kind"。
  行为本身是缺陷（水印对 C8 后的一切新 run 破损），非正确行为。
- **修法**（约束履行：仅 slo_tools 侧 kind 门扩容，参照 C16 domain_
  metrics 非账本行静默跳过模式；C5 冻结 schema 与 SH 零改动）：
  `REPO_VARIANTS["astra-sim-joint"].joint_audit_kinds` 并入上述六种
  非账本披露行（均不载运逐出/恢复/增长等 KV 账本事件，跳过 = 零重放
  语义差；词表仅挂 joint 映射，非 joint 仓零影响）。
- **验证**：test_hbm_watermark 38 passed + 1 skipped + 2 subtests 全绿；
  slo_tools unittest discover 与 C16 基线逐签名相同（failures=2 +
  errors=1 + skipped=1，均 HEAD 先在）；13 臂矩阵 postprocess 全净
  （watermark 产物 slo_hbm_intervals.csv/slo_hbm_watermark_instances.csv
  在案、零 FAIL 标记）。

### 28.4 C18 round-trip 测试入仓（重写）

新增 `sh_test_mesh/tests/test_hardware_twins_roundtrip.py`（原件失于
§21 /tmp 事故，按 RECOVERY.md C18 登记与计划 §5 C18 验收条款重写，
主 agent 已批准；范式 = tests/test_config_resolver.py）：4 用例 +
18 subtests——三孪生（_d2d_x05=2025 / _d2d_x2=8100 / _d2d_sub=1200
GB/s）× 2 容量档（paper-64gib / validation-160gib）= 6 组合：exact-key
schema 过（load_hardware_config 全链）／ρ 派生与
`joint.link_quota.derive_rho_eff` 显式量纲换算（GB/s→B/ns，与
JointHardwareRates.from_gbps 冻结恒等同源）后逐组一致 + Q_init 派生
（1/4/1）+ notes 披露 ρ 四位小数交叉核对／canonical 写回（_serialize_
json）→ resolver 重载逐字段零漂移（含 metadata 深比较）／对 base 源
递归 diff 严格限于白名单 {slug, label, d2d.bandwidth-gbps, notes}
（且四键全部实际发生差异，notes 增量恰 3 行）；另断言同容量档下孪生与
base 的 ResolvedHardware 仅身份/d2d/ metadata 不同。

### 28.5 文档同步（README + 开关清单）

- `README.md` 全面同步：§2 架构树（link_quota/hbm_port_flow_registry/
  event_recursion_predictor 新模块 + online 侧职责）与四动作范围、
  copy 块级交接（#handoff/#copy-stream、源端交接即释放）、E 递推预测器
  状态（adaptive = 递推正式在线实现 F12）、**到达合同披露**（C3b：
  service_done 锚 + 依赖门控 + merge_done 独立披露行——原 R11"重锚
  merge_done"旧表述全文清除）、链路遥测与配额机制两新 bullet（三开关
  + F7 耦合）；§3 状态披露表按 19 卡交付刷新（E adaptive/逐层恢复/
  遥测/配额/copy 交接/remote 结算/face_static/域度量八行改写或新增，
  home-merge 与 merge 物理流行的旧到达合同表述订正，SLO 水印行补非
  账本披露行词表注记）；§6 计数刷新（窄 614+9 / 全量 724+1skip+29sub
  + 增量归因）与新测试文件枚举 + 矩阵 13 臂；§7 新增 I 条（三孪生与
  ρ 档位）；§1 开关代码块补 JOINT_QUOTA_MODE/SH_LINK_TELEMETRY/
  face_static 取值域与 runner --quota。C2 移交"两腿"表述复核：判定
  维持无矛盾（指 NoC 前缀+池后缀的块级复合，JCM 代码同款表述），
  补"腿内为逐 shard 三腿 min"精度注记。
- `experiment/仿真各功能开关清单.md`：核对 C11 已落 §7.0 三开关
  （JOINT_QUOTA_MODE / SH_LINK_TELEMETRY / JOINT_SCHEDULER_MODE 扩
  face_static + fail-closed 补则）——登记完整无残差；唯一残差 =
  §10.1 硬件源登记行补 C18 D2D 带宽孪生集（三档 ρ + round-trip 测试
  指针 + 选用路径），本批补录。

### 28.6 全量测试终跑与裸仓还原

- 窄命令（workload/llama2_7b_inference）：**614 passed + 9 subtests**
  全绿（C0 279+9 → +335：C1+22/C3+5/C3b+11/C5+11/C9+66/C10+20/C13+18/
  C14+14/C2+24/C17+18/C15+42/C4b-FIX+5/C7+34/C8+19/C11+26，闭合零残差）。
- 全量（README §6 命令）：**724 passed + 1 skipped + 29 subtests** 全绿
  （C0 366+1skip+11sub → +358 = 窄增量 335 + C16 slo_tools 19 +
  C18 round-trip 4；subtests +18 = round-trip）。slo_tools 目录内
  unittest discover = 138 ran（2f+1e+1skip 与基线逐签名同）。
- 裸仓还原：`clean_test_records.sh` + `clean_build_artifacts.sh` 后
  四要素核验（trace_config 指针 = 占位、traces/ 仅 *.py 物化器、无
  generated/、无 build/）与残留判定（git status 条目 ⊆ C0 重建 20 条
  ∪ 各卡交付清单并集）结论记录于 /tmp/joint_exec/C19/DONE（/tmp 清理
  前的权威载体；本节登记判定方法）。

### 28.7 批次索引（§15–§27 一句话索引）与编号勘误说明

| 节 | 卡 | 一句话索引 |
| --- | --- | --- |
| §15 | C12 | P0 语义冻结记录（四动作/块生命周期/预测终点 F11/轮末结算含 F15/部分多源/remote 暂存 F14/copy 共享引用；G2 术语 rider 源） |
| §16 | C6 | 链路遥测三件接线批（C++ `--link-telemetry` 桥遥测；F6 覆盖边界冻结落字；开关登记移交 C11） |
| §17 | C3b | 到达合同迁移批（到达锚 merge_done → service_done + 依赖门控；merge_done 披露行；N2 承接；指标锚核查 = 条件项不触发） |
| §18 | C13 | copy 块级交接与源端立即释放批（四步协议补实现；#handoff/#copy-stream 守恒科目；18 用例） |
| §19 | C5 | 决策日志 schema 冻结批（joint_admission 审计流冻结字段；recompute elected/forced 分列；披露边界） |
| §20 | C14 | remote 完整结算与暂存计价一致性批（生命周期审计七项；C12-G1 裁定 (a) 补价 → C2 rider 规约 §20.1；kv_delta_journal 接口 + watch 闭合门） |
| §21 | （主 agent） | /tmp 证据区误删事故登记与恢复处置（RECOVERY.md 义务变更，见下勘误 ③） |
| §22 | C16 | 域三口径度量工具批（domain_metrics.py：D_feed/D_econ/实际选择 + A3' δ 敏感性重定价 + driver 五 sink + 非 joint 静默；19 用例） |
| §23 | C2 | HbmPortFlowRegistry + u_port 接线 + breakdown 全 11 字段物化 + A4' 补价 rider 批（24 用例；F4 保真边界） |
| §24 | C4 | 金值重推导 + 基线回归 + 2s 受控对拍批（joint/ 7 红 → 0；金值来源逐项；A2' 结论 = 缺省单遍即真值） |
| §25 | C15 | E 运行期事件递推预测器 + 逐层恢复与计算重叠批（F12 adaptive 正式在线实现；GB 逐层段就绪门控；rid#restore 区间账本；η/γ 入口；25+17 用例；含 C13 先在空尾块缺陷跨车道修复披露） |
| §26 | C4b-FIX | remote-read auto-K 多块 credit 执行协议缺陷修复（GB 体块 `_rcb{b}` 唯一相位键；5 用例；两族臂 exit 0；K209 等价锚逐位不变） |
| §27 | C11 | SH 配额集成 + 开关落点 + 决策时延批（动作级配额判据/quota_deferred 回队/F7 自动注入 + 防御断言/A5'-B3 键换算 rider/port_snapshot/时延入档；26 用例；四臂冒烟含 30s 特批一次） |

**编号勘误说明**（本文件 §15 起为多车道并行 append-only 落笔，实际节序
与计划 §1.4 写者顺序（C12→C3b→C13→C5→C4→C11→C14→C15→C16→C19）不同，
逐条登记）：

1. **C13 §16→§18**：计划卡文原编 §16（写者序第三位），主 agent 重编号
   ——C6（Lane-CPP 串行车道先完）先占 §16、C3b 占 §17，C13 落 §18；
   C13 派发 prompt 快照的节号与实际不一致属预期（append-only 以实际
   落笔节号为准）。
2. **撞号自纠两例**：C15 落笔时末节为 §24（卡文快照"接 §23"过期），
   自取下一空节号 §25 并在段首登记勘误；C4b-FIX 与 C11 并行落笔撞
   §26，C11 在自追加块内改序为 §27（append-only 语义保持，两节内容
   无覆盖）。C2 预定"接 §22"被 C16 先占，实落 §23（段首已注明）。
3. **/tmp 事故与义务变更（§21）**：C16 执行中误删 /tmp 大部，各已完成
   卡 DONE 原件与 sha1 法证丢失；重建登记 = /tmp/joint_exec/
   RECOVERY.md（C0 基线 20 条重建 + 关键 sha1 存档 + 各卡交付物登记）。
   原"DONE 汇总并入 PROVENANCE 索引后删除 /tmp"义务对已失文件不可
   履行——**以 RECOVERY.md 登记为准并入本节索引**（28.7 表 + 各卡
   引用）；C19 残留判定比对面同步换为 RECOVERY.md 重建集（见 28.6）。

### 28.8 总收口引用集（跨卡冻结/变更物汇总）

- **公式变更链（补遗编号 → 落地节）**：C1 逐 shard 三腿 min（F1，金值
  重推导源之一 → §24.1）；A1' 并集除数披露口径（→ §24.1 来源列）；
  A2' prefill 分块重扫 = 缺省单遍即真值（→ §24.4）；A3' δ_adm=0 终冻
  + C16 离线敏感性重定价 δ∈{σ̂,2σ̂,4σ̂}（→ §22.2；执行计划 §4.3）；
  A4' remote-read 执行端写腿补价（C12-G1 裁定 (a)，规约 §20.1、落地
  §23，过渡态披露作废）；A5'/B3 遥测整型 LinkId → 端点键换算
  （→ §27.6）；A6' C4b-FIX 多块 credit 命名缺陷（→ §26）。
- **新审计科目（守恒框架扩展）**：#handoff/#copy-stream（C13 → §18）；
  #readplan 预登记核销/残差清结/泄漏审计（C8 → C8 DONE 载体，无
  PROVENANCE 条目——本索引补位）；#merge-reserve 预留释放时机联合
  断言（C11 → §27.3）；rid#restore issue/complete 区间账本（C15 →
  §25.3）；配额借/还配对（C11 → §27.2）；kv_delta_journal 结算闭合
  门（C14 → §20.3）。
- **金值重推导**：C4 §24.1（credit_pricing 新金值 rr 2210/first 410/
  stream 1800/cost 3426(auto-K)·3496(单块退化)/compute-bound 616/
  adaptive K (1,2,2,13)；手算过程入测试注释，不静默改）。
- **P0 冻结记录**：C12 §15（四动作/终点/结算/部分多源/暂存规格；
  15.1–15.8 八小节）。

## 32. F4 复审零效/弱断言修复批（独立复审清单六组，2026-09-22）

依据：独立复审发现的零效/弱断言清单。本批**全部为测试文件改动**，零
生产文件变更（F1 并行卡的生产改动与本批无交集）。复跑绿：六文件
169 passed / 0 skipped（修复前 168 passed / 1 skipped——唯一 skip 即
F7 runner 用例）。

### 32.1 逐项登记（原位置 → 处置）

1. **online/test_joint_quota_integration.py:719-763**（F7 runner 注入
   用例）：skipUnless 仿真二进制存在 ⇒ 本仓（无 build）恒 skip，注入
   断言从未执行。→ 删 skip 门；测试内 tmp（tempfile.mkdtemp）造可执行
   占位二进制，monkeypatch `joint_runner` 的 `_REPO_ROOT`（二进制路径
   常量 `_BINARY_RELPATH` 的查找根随之指向占位）与 `_RUNS_DIR`/
   `_LOCK_PATH`（锁文件同指 tmp，不在仓内落 .single_simulation.lock）。
   内层 `subprocess.run` 本就被假函数替换（二进制从不真跑），占位只需
   通过 exists/X_OK/sha256 三道启动前检查。改后本仓真实执行注入断言。
2. **online/test_joint_quota_integration.py:745 附近**（同用例）：
   `subprocess.run = original_run` 全局补丁恢复位于 try 体内——断言或
   runner.main 抛异常时泄漏。→ 恢复移入 finally（模块常量转存/恢复
   同入 finally）。
3. **online/test_joint_quota_integration.py:318-324**（δ_adm=0 谓词
   用例）：期望式 `incumbent - challenger >= 0` 与 challenger_flips
   实现裁决式**同源恒等**（零效）。→ 写死手工推导查表：11 对代表性
   (challenger, incumbent)（含 c==i 平局边界 ×4、c<i、c>i），期望布尔
   逐对注释推导，assertIs 严格布尔。
4. **online/test_joint_quota_integration.py:798-805**（shell 旗标
   用例）：纯文本 grep（只查脚本源含两子串）零效。→ 提取执行式（同
   `test_shell_defensive_assertion` 的提取范式）：注入段后拼
   `printf "%s\0"` 数组展开探针，三场景（aimd+遥测 / 未置 quota+遥测 /
   static+未置遥测）核对 `--link-telemetry` 真占独立参数位 / 空参数
   向量。
5. **joint/test_link_quota_stability.py:525-526**（阈值带互斥用例）：
   `assertNotEqual(rate < r and rate >= 1.2r, True)` 恒真式。→ 删除，
   注释登记"原恒真式已删（复审发现）"；同循环的分支分域断言
   （shrink⟹rate<r / hold⟹[r,1.2r) / comfort⟹≥1.2r）保留，互斥性由
   分域逐点钉死。
6. **joint/test_link_quota.py:555-558**（δ=0 谓词用例）：同 3 的同源
   恒等。→ 同 3 改法（6 对查表 + 推导注释，assertIs）。
7. **joint/test_link_quota.py:611**（`_exhausted_verdicts` helper）：
   裸 `assert not verdicts[action].admitted`（python -O 下被剥除，且
   非 unittest 断言）。→ `self.assertFalse` + 失败消息（helper 本就是
   TestCase 成员）。
8. **online/test_joint_remote_settlement.py:336-341**（源端保护通道
   1）：`assertRaises(Exception)` + 仅 `assertNotIn("RuntimeError")`
   过宽——几乎任何异常都绿。→ 探针实测（/tmp/joint_fix/F4/
   probe_exception.py）该路径实际抛 `face_scheduler.KVCapacityError`
   （消息 "insufficient target HBM"）；改
   `assertRaisesRegex(KVCapacityError, "insufficient target HBM")`
   （与同文件通道 2 的 assertRaisesRegex 风格对齐），import 列表补
   KVCapacityError。
9. **online/test_joint_preadmit_visibility.py:559-567**（冻结接口
   用例）：软门自切换（字段在场→强断言；缺席→assertNotIn 自证
   缺席）恒绿。→ C7 已交付 `link_telemetry_rates` 字段
   （joint/joint_cost_model.py dataclass），删 else 分支改硬断言：
   `assertIn(..., JointCostModel.__dataclass_fields__)` + 字典逐位
   assertEqual；用例头部的软门注释同步改写。
10. **joint/test_hbm_port_flow_registry.py:210**（u_port 除数用例）：
    装饰性 `assertEqual(registry.divisor(5) + 1, 4)`（对上一断言
    结果 +1 的重复算术，不约束生产行为）。→ 删除；注释登记（+1 属
    JCM `_shard_leg_ns` 调用侧语义，非 registry 契约）。

### 32.2 复跑与并行态

- 复跑（pytest，六文件逐个）：online/test_joint_quota_integration.py
  26 passed；joint/test_link_quota_stability.py 20 passed；
  joint/test_link_quota.py 66 passed；online/test_joint_remote_settlement.py
  14 passed；online/test_joint_preadmit_visibility.py 19 passed；
  joint/test_hbm_port_flow_registry.py 24 passed——合计 169/169 绿、
  0 skipped。复跑未遇 F1 并行中间态 ImportError（活性重试协议未触发）。
- 复跑时 F1 状态：/tmp/joint_fix/F1.done 不存在（**F1 未完成**）；其
  生产文件改动与本卡六测试文件无路径交集。

## 30. F2 ETA 截断积分缺陷修复批（UnifiedTimeline.step 分段恒速积分钉正 + 守恒断言，2026-09-22）

依据：独立复审实锤的 ETA 截断分支积分缺陷。本卡只改两独占文件
（joint/event_recursion_predictor.py、joint/
test_event_recursion_predictor.py）；写回半面（WritebackVictim/
predict_space_available/with_committed）原样保留——预建未接线是已登
记状态，本卡只修积分器。

### 30.1 缺陷

`UnifiedTimeline.step()` ETA 截断分支（原 :392-395）：快照含已知
ETA（release_eta_ns 非 None）的在册流与活跃候选流共享资源、且 ETA
早于最早完成时刻时，分支推进 now_ns = expiry 后直接 return True，
未按当前速率扣减活跃流 remaining——[now, expiry) 区间字节凭空消失，
完成时刻系统性偏晚。审计复现：1000 B/ns 资源、2000B 候选流、共享流
ETA=1ns → 修复前模型完成 3ns、物理应为 2.5ns（偏晚 20%）。违反预
测器冻结的"分段恒速积分"不变量。当前生产路径 ETA=None 未命中
（face_scheduler.py 唯一注入点 :5442 release_eta_ns=None，
_next_committed_expiry 恒 None）；写回方向（predict_space_available/
with_committed 注入已知 ETA 流）一接通即成活性缺陷。

### 30.2 修法（event_recursion_predictor.py）

- **:464-478 截断分支**：推进到 expiry 的同时按 elapsed × rate 对全
  部活跃流扣减 remaining（与正常完成分支同一积分式；速率复用 finishes
  计算时已求的份额速率，不二次重算）；随后 _settle_completions 先到
  先结（浮点触底护栏——精确算术下 expiry < min(finishes) 不会有流在
  此完成，判定式/确定序与正常分支同口径）。
- **:378-395 _settle_completions**：从原正常分支提取的完成结算 +
  穿透零复核（两分支共用，行为不变）。
- **:397-422 _assert_conservation + :83-88 模块开关**：见 30.3。

### 30.3 守恒断言

step() 步末 assert 级复核（_CONSERVATION_CHECK 默认开、可关，热路
径 O(|active|) 求和；docstring 说明）两条：① 字节账目——本步
Σremaining 减少量 == Σ区间服务字节；② 时刻-服务联动——Σ区间服务
字节 == Σ活跃流份额速率 × 推进时长。单靠①抓不住旧缺陷（丢服务时两
边同为零），②才是牙齿：注入旧缺陷分支复跑审计场景，断言即触发
"served=0.0 expected=500.0 elapsed=1.0"（/tmp/joint_fix/F2/
assert_teeth.py 验证）；修复版同场景通过。

### 30.4 测试与零扰动验证

- 审计场景回归（test_event_recursion_predictor.py:502）：完成时刻
  2.5ns（assertAlmostEqual，修复前 3ns）；完成账本 int 量化
  ceil(2.5)=3 另断言。
- 新增 EtaTruncationTests 5 例（:493-580）：审计回归 + 三变体（ETA
  恰等于完成时刻→走正常分支 1.0ns / ETA 极小 1ns 对 2e9B@1e6B/ns →
  2000.5ns / 多流混合 ETA 双资源双到期 → 双流同刻 13/3ns）+ 守恒开
  关关闭数值不变（开关仅断言级）。
- 既有期望修正 1 处：test_sequential_victims_share_bandwidth
  [1000,2000,3000]→[1000,1500,1917]——原值即缺陷值（v2/v3 在 ETA
  截断区间的服务量被丢弃；修正后按分段积分 [0,1000) 均分、之后独享，
  用例本意"同批不得各自按独享带宽估算"仍成立且更严）。
- 复跑：预测器文件 30 passed（原 25 + 新 5，unittest 入口同 30 OK）；
  joint/ 全套 347 passed（基线 342 + 新 5）；face 侧零扰动——
  test_face_scheduler.py 48 passed + 2 subtests、
  joint/test_face_static_mode.py 18 passed、
  online/test_joint_layer_restore.py（_adaptive_retention_target 直
  达路径）17 passed；整个 workload 目录 619 passed + 9 subtests，
  0 failed / 0 skipped（F2 范围）。
- 零 git 操作（无 commit/checkout/restore/stash）。

## 29. F1 修复卡：§4.3 补遗 A7'/A8'/A9'/A10' 四件 + quota_deferred 安全网钉测 + _transfer_ns_shards 空 shard 契约（2026-09-22，用户授权修复批）

依据：主对话冻结补遗 §4.3:476-479（2026-09-22 交付后独立复审，用户
授权）。本卡文件面（独占）：joint/joint_cost_model.py、
online/sh30_online_scheduler.py；测试 = 新文件
joint/test_joint_fix1_pricing.py（22 例）+ online/
test_joint_preadmit_visibility.py 夹具补 A8'/A10'(a) 装配属性。

### 29.1 A7' PARTIAL 驻留 recompute 计价修正

- joint_cost_model.py `_recompute_missing_tokens`：`resident_here`
  判据 `location == "local_hbm"` 单态 → LOCAL/PARTIAL 双态（对齐执行侧
  SH recompute 分支与同文件 `_space_footprint_by_tp_rank` 的既有双态口
  径）。HEAD 既有笔误：双态分支写入时单态判据使其成为死分支——PARTIAL
  驻留场景恒按整份 history 计费，recompute 工作量系统性高估约 2 倍、
  动作选择偏离（SH 侧重算工作量用含 partial 的正确公式）。
- 专项测试（joint/test_joint_fix1_pricing.py）：审计复现锚
  L=32/prefix=16/H=1000 ⇒ **500**；取整边界 ceil 在缺失侧（H=1001 ⇒
  501）；PARTIAL 异地 / REMOTE 基整份、LOCAL 驻留 0 既有行为不动；
  estimate_action 级 compute_ns 分层锚（驻留 (50+50)+20 < 异地
  (50+100)+20，LOCAL (50+0)+20）。既有非 PARTIAL 金值全绿
  （shard_pricing 22 / credit_pricing 8 / telemetry_divisor 34）。

### 29.2 A8' transfer_factor EWMA 接线

- sh30_online_scheduler.py `_ingest_link_telemetry`：逐窗口样本喂
  `self._joint_factors.observe_transfer_from_link_window`（名义速率 =
  `_joint_rates.noc_link_bytes_per_ns`；端点键换算复用
  `_telemetry_endpoint_link_key`；仅 served>0∧active>0 样本喂入）。
  getattr 软门：`__new__` 测试替身未装配 `_joint_factors` 时跳过
  （与 `_hbm_ports`/kv_delta_find 软门同款；真 `__init__` 恒在场）。
- 名实统一（消除"三件已交付 vs observe 全仓零调用"自相矛盾）：JCM
  模块头"状态"节"transfer 因子保留接口位——updates=0 披露"改"样本源
  = SH 链路遥测窗口（A8' 接线；列车核销通道无纯传输段可因果分离故不
  经该通道）"；C7 节末补 A8' 落地注记；SH `_observe_service_factors`
  docstring 同步改写。
- 测试：拥胀窗（rate 2 B/ns、名义 200）⇒ updates>0 且 factor==100.0；
  决策构造（JCM `__post_init__` flush）后因子经共享对象到达代价模型；
  同 tick 双链路 Σactual/Σbase 单条更新（base_ns 逐链路 int 截断 ⇒
  1000/7）；实测=名义 ⇒ 恰 1.0；软门替身 ingest 不炸、速率照写。

### 29.3 A9' 遥测速率过期

- `_ingest_link_telemetry`：present_keys 在场集（本包**全部样本**的
  端点键——含零活跃样本占位；未知 id fail-closed 提前于 active 判定）
  → 包尾删除缺席既有条目。语义锚：C++ 契约"空闲链路省略"即空闲信号，
  删除后 divisor_effective 退回注册表除数（measured=None 退化分支），
  消除"忙转闲后陈旧实测速率永久驻留、长 run 单调抬升 NoC 计价除数"。
- 测试：忙（遥测除数 100.0 抬过注册表 1 流）→ 转闲（包缺席）⇒ 条目
  删除、divisor_effective 回落注册值 1.0；缺席一拍后重现 ⇒ 键重建
  （新值 6.0，非陈旧 2.0）；空数组包 = 全闲 epoch ⇒ 全部过期；键缺席
  包（旗标关）不做在场核对；全零样本（active=0）在包内 = 链路在场、
  旧速率保留；整型键透传形态（未装配换算表替身）同语义。

### 29.4 A10' 两件

- (a) ingest 处 `served_bytes==0 ∧ active_ns>0` 整数截断伪影样本
  **丢弃 + 披露计数**（`_telemetry_zero_rate_dropped`）：不写 rates
  （写入会让 JointCostError 崩在远离成因的 TelemetryLinkFlowView 构造
  处）、不喂因子（不进 updates/rejected 双通道）、样本证据计数照涨
  （覆盖证据不受丢弃影响）。决策日志遥测块重构为
  `_telemetry_coverage_decision` 单一构造点（verify_run_end 复用，
  行为零变更）并新增 `telemetry_zero_rate_samples_dropped` 披露位。
  真零速率持续场景由 A9' 缺席失效兜底。
- (b) `_quota_enroll_admission` 失败防御分支补回滚
  `_rollback_admission_registrations`：`_release_transfer_flows(rid)`
  ＋ `_release_transfer_flows(rid#readplan)` ＋ 清
  `runtime.remote_read_preplan`（幂等空放；消除"重试改选动作成功前
  注册表躺着幽灵承诺（除数虚高）"窗口）。失败注入测试：
  `_try_admit_request` 全链驱动（事务段/流登记/预登记全真实，仅
  select_instance_and_action 场景桩 + enroll 注入 False）——入册失败
  时刻注册表非空锚（fake enroll 内捕获 flows/preplan/HBM 泄漏）⇒ 链路
  流 / 池端口 / HBM 端口 / preplan 账目全归零、`_assert_no_readplan_
  leaks` 通过、quota_deferred 失败日志落地。场景注记：remote-read@3
  路由每链需求恰 Q=2 不先触链路门（quota 夹具同款事实），配额过滤
  放行后才能走到入册防御分支。

### 29.5 quota_deferred 安全网钉测（不改生产逻辑）

- C9 冻结语义直接单测：生产中 recompute 恒可行（免链路动作不被配额
  裁）属设计结果，deferred 机器是安全网——以**合成全拒 verdicts**
  （QuotaVerdict(admitted=False, quota_link)）驱动真实
  `_quota_filter_candidates`（deferred 记录形态：requeue=True /
  wait_reason / 重试键尾位 = 配额代数、候选表全不适用化）+ 真实
  `_admit_waiting_requests` 闭环：pass1 回队**单份**（无丢请求、无重复
  回队、quota_deferred_since_ns 落锚）→ pass2 键未变 ⇒ 重试门跳过
  （verdict 调用计数零增、仍单份）→ tracker 信用入册+释放 bump 配额
  代数 ⇒ `_current_retry_key` 与失败键分离（唤醒门重开）→ pass3 唤醒
  再评估（verdict 计数前进、二次 defer 折叠 joint_admission_wait
  attempt_count=2、仍单份）。

### 29.6 小项：_transfer_ns_shards 空 shard 契约钉字

- docstring 登记（行为零变更、判断 = 维持现状）：`paths_by_rank` 空
  序列 ⇒ **返回 0 且 startup_ns 不计入**——无 shard 即无传输事务，
  启动时延属"事务发生"的固定成本，不随不存在的流产生（与
  `_shard_stream_ns_shards` 的 default=0 同语义族）。测试两锚：空序列
  ⇒ 0（startup 12345 丢弃）；在场 shard（零字节、零跳路径）⇒ startup
  计入（12345）——边界分界锚。

### 29.7 测试与冒烟

- 零后端：新增 joint/test_joint_fix1_pricing.py **22 例**全绿；全量
  复跑 joint/ **369 passed**、online/ **222 passed**（+7 subtests）、
  workload 根 test_face_scheduler + test_sh30_kv_incremental_
  invariants **54 passed**。online/test_joint_preadmit_visibility.py
  夹具补 `_joint_rates`/`_joint_factors`/`_telemetry_zero_rate_dropped`
  （19 例全绿，其遥测族顺带覆盖 A8' 接线路径）。
- 冒烟（互斥协议全程：/tmp/joint_fix/sim.lock fcntl 排他锁 → 备份
  trace_config.csv → 改指针 → 跑 → 还原 → sha1 对比 → 放锁；输入 =
  agent-traces/tracelab/astra_compute_20.csv 前 2s；joint_runner.py
  D14 入口；产物 /tmp/joint_fix/F1/{baseline_TJE_2s,fixed_TJE_2s,
  fixed_aimd_2s}/）：
  - **修复前基线 TJE-2s**：exit 0；e2e_p50 529.681ms / p99
    17797.756ms / n=21；admissions 21（stay 20 + copy 1）；service_
    factors decode 1.10072…（1297 updates）/ prefill 4.02981…（31）。
  - **修复后缺省 joint 臂 TJE-2s**：exit 0；sidecar 关键指标与基线
    **逐位一致**（e2e 全分位/n/动作分布/service_factors 全字段同值；
    transfer_factor 1.0、telemetry 关闭态零漂移——A8' 缺省臂零扰动
    实证）。A7' 修正面 = PARTIAL 驻留 recompute 计价，本 2s 窗口无该
    类决策（stay 20 + copy 1），决策序列零变化属预期登记。唯二差异：
    决策日志遥测块新增 telemetry_zero_rate_samples_dropped: 0 披露位
    （A10'(a) 新键，预期内）；调度处理墙时 1070.41→1061.715ms（非决策
    量噪声）。
  - **修复后 aimd 臂**（--combo TJE --quota aimd，F7 自动注入；另置
    ASTRA_LINK_OBSERVER=1——§27.8 30s 臂同款"observer=on 补通路"，
    样本数据源开启）：exit 0；manifest link_telemetry_injected=true、
    cpp "link telemetry: enabled, links=186, observer=on"；
    **A8' 端到端走量**：transfer_factor updates **1376**（本 workload
    无拥胀窗、样本比恒 1.0——与 §28.2 aimd-30s"无 shrink"同象；非 1
    数值路径由 29.2 单测钉死）；**A9' 端到端**：1392 epoch / 32908
    样本 / rate_entries 14（在场核对逐包剪枝空闲链路，无单调累积）；
    伪影丢弃 0（本窗口无整数截断样本，伪影路径单测覆盖）；
    collective_coverage=true；e2e 与缺省臂逐位一致（同决策序列）。
  - 裸仓还原：trace_config 指针 sha1=9a539b11… 与备份一致；
    sh_test_mesh/generated/ 清回会话前态（仅 runtime_config）。

### 29.8 移交

- A7' 金值影响面仅 PARTIAL 驻留 recompute 场景（新增专项锚钉 500）；
  既有非 PARTIAL 金值零改动。A9'/A10'(a) 披露键
  （telemetry_zero_rate_samples_dropped）为决策日志遥测块新增字段，
  重放/收集侧前向兼容（新键缺席即旧值语义）。
- 完成标记：/tmp/joint_fix/F1.done 已落（下游卡冒烟门依赖）。

## 31. F3 kv_delta_journal 序列化收口批（dump 第四键 + domain_metrics 四层可信度分级读取，2026-09-22）

**对象**（本卡独占文件，零越界）：`workload/llama2_7b_inference/face_scheduler.py`
（仅加 journal 序列化导出，不动其他）、`workload/llama2_7b_inference/online/
online_service.py`、`slo_tools/domain_metrics.py`、`slo_tools/tests/test_
domain_metrics.py`、`workload/llama2_7b_inference/test_face_scheduler.py`（加
用例）；PROVENANCE §31 登记。零 git 操作（无 commit/checkout/restore/stash）。

### 31.1 移交义务链与履行（§20.3/§20.6-2 → §22.4/§22.7-1 → 本卡）

C14 §20.3 披露边界落字："run 级 sidecar 序列化按 C13 (a) 同款登记归
`online_service.dump_joint_kv_ledgers` 属主（C16/C19 顺带）"（§20.6-2 移交
C16/C19）；C16 §22.4 如实登记"读取器已绪、run 级序列化一行未落、产物出现
前该块 status=TODO"（§22.7-1 再移交 C19）。两次移交在收口卡被静默丢弃
（复审实锤：dump 恒 merge_degrade/deep_gap/copy_handoff 三键、
domain_metrics.py:815-824 停 status="TODO"）。本卡（F3）履行该义务：
dump 加 `kv_delta_journal` 第四键 + domain 消费面解除 TODO（四层可信度
分级）。

### 31.2 落地（文件:行）

1. **face_scheduler.py:1606** `KV_DELTA_JOURNAL_FIELDS`——行字段冻结面常量
   （构造序 = `_append_kv_delta_row` 的 append 序）；**:5286**
   `KVCacheManager.kv_delta_journal_rows()`——序列化导出接口（与
   copy_handoff_events/restore_events 同披露纪律：GREEN run 亦保留、非失败
   台账；逐行 dict 浅拷贝 = 冻结事实不回写；字段集漂移/seq 断链
   fail-closed——构造面唯一入口保证 seq 恒为追加序号，到达即破损）。
2. **online_service.py:173-175** `dump_joint_kv_ledgers` 第四键
   `"kv_delta_journal"`（与三键同风格 JSON 序列化；原子写 tmp+os.replace
   与 try/finally 全路径落盘语义不变；行源 = `kv_delta_journal_rows()`，
   两文件同卡落地无版本偏斜，直连访问器）。
3. **domain_metrics.py:135-146** 四层可信度常量（`KV_DELTA_TIER_*` +
   `KV_DELTA_TIER_NOTE`）+ **:837** `_kv_delta_trajectory` 重写：status 由
   TODO/ok 二值改 tier 名；模块 docstring 数据源条目同步（bridge 侧车键位，
   wscllm 式 results/kv_delta_journal.jsonl 权威层本仓不产出）。

### 31.3 字段口径（从 C14 实际记账结构取，不发明新口径）

逐事件行 = C14 `_append_kv_delta_row` 的 16 字段冻结面：seq（单调）/
session_id/trigger_request_id（请求标识）/working_kind（结算时刻类别快照）/
direction（方向/科目：stay|forward|reverse|in_place）/zero_byte_flip/
winner_instance/loser_instance/home_before/home_after/home_migration（C16
home 迁移轨迹）/transferred_bytes（delta 字节）/home_side_retained_bytes
与 exec_side_retained_bytes（两侧实际保留量；copy 的 home 侧取交接 journal
残量口径——已释放字节不计入）/new_tokens/staging_return_bytes（F14 恒 0
披露位）。**rank/层区间不在 C14 记账结构**——kv_delta_journal 为逐请求
结算行；copy 块层区间在 copy_handoff_events、后缀恢复组层区间在
restore_events，各有其科目——序列化不代拟逐 rank/逐层拆分（水印
certified 层所需的逐 rank before/after 链 + checksum 证书属 wscllm 归档
权威层，本仓不产出，README:194 口径维持成立）。

### 31.4 四层可信度分级（domain 消费面；语义镜像 hbm_watermark 四层纪律）

tier 由 run_dir 内容自动判定、status 如实标注（证据强度降序）：

1. `settlement_full_join`——sidecar 键在场非空、行链自洽（seq int 严格
   增）、全部行联上 completion tick（驻留时序完备）；
2. `settlement_partial_join`——行在案且链自洽，但有行联不上 tick
   （rows_unjoined/dwell_unjoined 如实计数；结算事实完整——行内字段自证，
   仅驻留时序近似降级）；
3. `settlement_empty`——sidecar 在场但键缺席/空列表（零结算 run——行
   存在 ⇔ 结算完成，或 F3 前旧 sidecar schema；sidecar_keys 键清单披露）；
4. `decision_log_only`——sidecar 文件缺席/不可解析（home 迁移轨迹仅剩
   决策时刻 completion 披露；无法区分"零结算"与"F3 前未序列化"）。

**分级缺口的降级如实标注**：水印 certified 层（per_rank_total_hbm_
certified）判定信号 = results/kv_delta_journal.jsonl + kv_delta_journal_
checksum.json（run 末守恒证书）——本仓不产出（C14 §20.3 README 注记），
顶层按可得信号（sidecar 行 + 行链自洽 + completion 联 join）降级实现，
tier_note 落字"不冒认 certified"。行链断裂（非对象行/seq 非严格增）
fail-closed 退出码 2（水印同款纪律：账本损坏不得静默降级）。

### 31.5 测试

- **test_face_scheduler.py:2609** `KVDeltaJournalExportTests` 4 例：前向
  合并导出字段完整（键集+构造序 = 冻结面）与字节守恒（transferred ≡ merge
  返回 KVTransfer total_bytes ≡ last_merge_outcome 披露快照；两侧 retained
  = 结算时刻账本真值 kv(10)/kv(5)）；stay 行 + seq 链 0/1 + 冻结拷贝语义
  （改写导出行不回写账本、重复导出逐位一致）；字段集漂移/seq 断链
  fail-closed；零结算导出空元组。全文件 52 passed（基线 48 + 新 4）。
- **test_domain_metrics.py:621** `KVDeltaTierTests` 5 例：partial_join
  （联不上 tick 的行 → rows_unjoined=1 且事实照常全量披露）/ 键空列表 →
  settlement_empty + sidecar_keys 键清单 / 三键旧 schema → empty（区别于
  文件缺席的 decision_log_only）/ 半截 JSON → decision_log_only / seq 断链
  fail-closed（SloToolError）。既有两断言随 tier 化更新：无 sidecar fixture
  → decision_log_only；双行全联 → full_join + tier_note"不冒认 certified"
  在场。文件 24 passed（19 + 新 5）。
- **回归**：online/ 全目录 22 测试文件全绿（含 test_joint_remote_
  settlement 14 例 C14 journal 面、test_joint_copy_handoff 18）；joint/
  mechanisms 40 + fixes 22 + review2 34（dump 固化用例兼容第四键）+
  review3 29、test_sh30_kv_incremental 2、tests/test_metrics_contract 35
  全绿；slo_tools 143 ran（C16 §22.6 基线 138 + 新 5），失败集与基线逐
  签名相同（2 parity 预存红 + 1 contract 预存 error + 1 skip），零新增红。

### 31.6 冒烟（2s 缺省臂 TJE，互斥锁协议）

前置等待 /tmp/joint_fix/F1.done（12:28:17Z 出现）→ /tmp/joint_fix/sim.lock
fcntl 排他锁（fd 9 持锁）→ trace_config.csv 备份至 /tmp/joint_fix/F3/ →
指针临时指向 2s 物化队列（astra_compute_20.csv 前 2s，materialize_20_30s.py
window=2e9）→ plan 物化 → `joint_runner.py --combo TJE` → 还原 sha1 对比
（9a539b11cf03129fa9812525f70d19e49d292466 verified）→ generated/ 清理 →
放锁。产物 = /tmp/joint_fix/F3/F3_TJE_2s/（exit 0；invocation.json/run.log
在案）。脚本 = /tmp/joint_fix/F3/run_smoke.sh（F1 smoke_lib 同款协议）。

- **sidecar 实测键清单（四键）**：`copy_handoff_events` / `deep_gap_events`
  / `kv_delta_journal` / `merge_degrade_events`——第四键在场且非空。
- **kv_delta_journal 21 行**（21/21 请求全部结算，行存在 ⇔ 结算完成闭合）：
  seq 链 0-20 无断裂；direction = {stay: 20, reverse: 1}；transferred_
  bytes_total = 0。reverse 行 = session_1_request_1（working_kind=copy、
  zero_byte_flip=true、home 1→5、两侧 retained 6364332032/0、new_tokens
  1052）——与 C14 §20.5 冒烟记录的 copy 结算事实（session_1_request_1
  reverse/home_flipped_to=5/transferred=0）逐字段同源，journal 通路照实
  落账。deep_gap/merge_degrade 台账均空、copy_handoff_events 11 行。
- **零行为漂移实证**：delivery/ack = 1392/1392，与 F1 修复前基线
  （baseline_TJE_2s，同 2s 输入）逐位一致——本卡为纯披露/序列化侧改动。
- **真实 run 消费面终验**：domain_metrics 直跑本 run（产物落
  /tmp/joint_fix/F3/domain_check/）——kv_delta_journal 块 status =
  `settlement_full_join`（21 行全部联上 completion tick；rows_unjoined=0、
  dwell 样本 n=16、sessions=5、multi-home=1 即 session_1 的 1→5 迁移），
  σ̂=9.34e9 ns 与 C16 §22.6 事故前实测同值（同源 run 特征复现）。

### 31.7 边界与残留

- **README 同步不在本卡文件清单**（只读约束不越界）：slo_tools/README.md
  :78/:106-107 的 "kv_delta_journal TODO(C14)" 描述已过时（本卡后 = 四层
  tier 读取 + sidecar 序列化已落）；README:194/:258 的 wscllm 式权威层
  口径陈述仍成立（本仓仍不产出 checksum 归档层）——行级同步登记归后续
  文档批。
- dwell 口径维持 C16 原设计（journal 行无 tick → 联 completion tick 近似
  并披露）；full_join 层即该近似的完备形态，不新造时序源。
- 冒烟空 journal 时序的如实登记机制保留（键在场断言恒执行，行数照实披露）。
## 33. F5 真 C++ commit 路径多块 remote-read credit 验证批（joint_fix F5，2026-09-22）

使命：C4b-FIX（§26）的验证止于 Python 镜像命名计数，真 C++
GraphBatchCommitter 从未在多块形态下运行——本卡让真二进制的
commit 路径真实执行多块 credit 形态并确认通过。**零仓内源文件
修改**；临时产物全部落 /tmp/joint_fix/F5/；零 git 写操作
（无 commit/checkout/restore/stash；git show 只读对比）。

### 33.1 构建与输入

- 构建：README §4 配方（拼装 build/astra_analytical/CMakeLists.txt +
  congestion_aware 目标），当前树（F1 落笔后）全量构建成功。
  `AstraSim_Analytical_Congestion_Aware_Online` sha1=
  3578a584e2aa1fdcf8ef2d7cfd5863c6171c132d（sha256=
  11c817050febb16b9ec518a5f54c5f69a6c738d956ed2cd4be3c96de5ca1fa16，
  invocation.json 法证）；同构建的 GraphBatchCommitterTest ALL PASS。
- 输入：固定冒烟源 agent-traces/tracelab/astra_compute_20.csv 前 2 秒
  （5 会话/21 请求；queue md5 62a14f16c976c16f8259256fae505950），
  materialize_20_30s.py 物化到 /tmp/joint_fix/F5/smoke_input/。
- 互斥与指针协议：/tmp/joint_fix/sim.lock fcntl 排他（等门 F1.done
  20:28:30 后起跑）+ trace_config.csv 备份→改指→跑→还原，还原后
  sha1=9a539b11…（== 备份 == git HEAD blob，逐字节核验）。

### 33.2 强制钩子（C4b"强制钩子"资产失于 §21 /tmp 事故后的重建）

remote-read 在 2s 轻载窗零自然选中（stay 支配，§24.3 同口径；本卡
对 F1 baseline 决策日志复算确认全部 target_wait=0、RR 差距 = 首
credit+merge）。因仓内无强制开关（§24.3 已披露"SH 无强制动作开关"）
且本卡禁改源文件，采用 **sitecustomize 运行时钩子**（/tmp/joint_fix/
F5/hooks/，PYTHONPATH 注入，F5_FORCE_REMOTE_HOOK=1 门控）：包装
`joint.joint_scheduler.select_instance_and_action`，存在适用 remote-read
候选时改选其中成本最低者。诚实性合同（对齐 §24.3"强制 ≠ 自主选中"）：
强制事实**写入决策日志本体**（instance_rule_note 追加 "F5 forced
remote-read (verification hook)" + 选中候选 breakdown note
f5_forced_remote_read_selection 含自然胜者/强制者成本对照）。第二包装
`OnlineSchedulerBase._digest_row`：逐批捕获 `_rcb{n}` 集体节点名 + 仓内
代码自算的 content_sha256 + 逐名 rank 参与者集（C++ 预检谓词
"同名集体每 rank 恰 1" 的零后端镜像），侧车 /tmp/joint_fix/F5/
node_capture.*.jsonl。钩子离线自测 PASS（含 F1 落笔后复测）；失败
不吞异常（fail-loud）。

### 33.3 双臂结果（joint_runner.py 起跑，2s 窗 21 请求）

1. **mb3 多块臂**（--combo TJE --remote-credit-iters auto + 钩子）：
   **exit 0**；21/21 completed；16/21 准入 remote-read（5 个 turn-0
   无异地历史自然 stay）；决策日志 M>1 切片摘要 1240/1243；捕获
   **1240 批含 `_rcb` 集体、167,892 个去重 `_rcb` 节点名、块号分布
   _rcb1.._rcb8（max M=8 = 原崩溃形态 S_j=8/K=1/M=8）**；同名集体
   单 rank 重复违规 **0**（预检谓词镜像全绿）；捕获行与本 run
   graph_batch_digests.jsonl 的 content_sha256 **逐批绑定 1240/1240**
   （多块批次即交付提交批次的密码学证据）；cpp.log 无
   "expected exactly one"/"participants on rank"/abort 签名；
   readplan_reconcile 16 + readplan_settle 14 + merge_done 13 正常
   收尾；delivery_count=1405。样例：
   `batch_train_i5_2_rcb1_all_layers_attention_all_reduce` @
   ranks [0,1,6,7,12,13]。
2. **sb3 单块锚**（同配置 + --remote-credit-iters 8000 ≥ 窗内最大
   read_passes 5952 ⇒ M=1）：**exit 0**；21/21；16/21 remote-read
   同位选中；**零 `_rcb` 批**（捕获文件不产生）；1243/1243 切片摘要
   全 M=1——K≥S 单块 = v1 等价命名路径（I3a 锚）在后端真实执行且
   逐位不破。
- launch 账目：4 次（mb_autok/mb/sb_k8000 三次为编排脚本参传缺陷
  期的弃跑——invocation.json joint_switches={} 复核发现 heredoc 未
  透传 "$@"，实际以缺省 TJE+auto 跑；修复后 mb3/sb3 为登记臂），
  全部 exit 0，无污染窗口（缺陷只影响开关记录不影响树状态）。

### 33.4 自清与边界

- 自清：plan 物化目录 generated/llama2_7b_inference_54npus_plan_48a3294f
  删除（路径断言后）；clean_build_artifacts.sh 清 build/；trace_config
  还原 sha1 == HEAD；traces/ 仅 *.py；无 results/run_logs/log 残留。
  留存（非本卡产物，未触碰）：generated/runtime_config/（19:18 并行卡
  先建）、sh_test_mesh/runs/.single_simulation.lock（20:08 F1 baseline
  经 D14 runner 创建的共享锁文件，flock 语义常驻）。
- 边界披露：选择强制为验证夹具而非策略行为（决策日志内自披露）；
  多块语义本体（门控/主链/旁挂/字节守恒）为未触碰的仓内代码；C++
  侧零改动（sha 见上）。证据资产：/tmp/joint_fix/F5/（runs/mb3、
  runs/sb3、node_capture.*.jsonl、hooks/、trace_config.csv.bak）。

## 34. F6 `__new__` 测试替身软门面销账批（joint_fix F6，2026-09-22）

复审定性：生产代码里为 `__new__` 测试替身开的"软门面"（`getattr(self,
"_x", None)` 式防御 + 类级软缺省块）在替身漏设属性时静默跳过（如 HBM
端口登记静默丢失），无销账清单；真 `__init__` 恒设这些属性，软门 = 纯
防御残留。本卡销账：删软门改直接属性访问（AttributeError = 替身漏设，
fail-loud），并系统性对齐全部测试替身。判定标准：属性在真 `__init__`
无条件设置 ⇒ 软门，删；属性确有条件语义（跨模块 schema 可选 / 生命周期
后置赋值）⇒ 保留并在行侧注释钉明生产条件。以 grep 全面定位为准（不限于
复审清单；F1 修复批新增的 observe_transfer_from_link_window 喂入点软门
一并命中）。

### 34.1 销账清单（sh30_online_scheduler.py；face_scheduler.py 零软门）

删除（17 处 getattr/hasattr + 1 个类级块，属性均在真 `__init__` 无条件
赋值）：

| 位置（改后行号域） | 软门 | 处置 |
|---|---|---|
| :471-475（原 C11 类级软缺省块，16 个类属性） | `_quota_tracker/_quota_enrolled/_quota_merge_reserves(+_created)/_quota_decode_owner_seq/_joint_action_selection_counts/_quota_deferred_wait_counts/_quota_deferred_dwell_ns/_quota_aimd_action_counts/_admission_decision_wall_ns_total/_max/_count/_quota_verdict_wall_ns_total/_candidate_checks/_quota_admit_events/_quota_release_events` 类级缺省 | 整块删（留 F6 墓碑注释）；替身自行补 __init__ 初值 |
| :1725-1729 | `getattr(self.kv_manager, "last_merge_outcome", None)` | 直接 `self.kv_manager.last_merge_outcome`（真 KVCacheManager `__init__` 恒设，face :1747） |
| :1894-1898 | `getattr(runtime, "joint_working_copy", False)` | 直接访问——同方法首循环（completion_facts）无条件先行赋值，读恒晚于写 |
| :2378-2382 | `getattr(self.kv_manager, "kv_delta_find", None)` 闭合门软跳 | 直接调用（真 manager 恒定义，face :5277）；替身缺接口 = AttributeError |
| :2412-2436 | `_register_transfer_flows` 的 `_hbm_ports` None 门 | 直接 `self._hbm_ports.register(...)`（noc_migrate 端点腿） |
| :2440-2444 | `_release_transfer_flows` 的 `_hbm_ports` None 门 | 直接 `self._hbm_ports.release_owner(owner)`——端口注销静默丢失通道闭合 |
| :3244-3248 | `_assert_no_readplan_leaks` 的 `_hbm_ports` None 门 | 直接 `self._hbm_ports.leaked_owners()` |
| :3329-3336 | `_ingest_link_telemetry` 的 `_joint_factors` None 门（F1 新增喂入点）+ :3329 `if factors is not None` | 直接 `self._joint_factors.observe_transfer_from_link_window(...)`（A8'） |
| :3594-3599 | `_joint_cost_model` 遥测 kwargs 的 `getattr(self, "_link_telemetry_rates", None) or {}` | `dict(self._link_telemetry_rates)` |
| :3613-3615 | `hbm_port_registry=getattr(self, "_hbm_ports", None)` | 直传 `self._hbm_ports`（不再静默退回端点腿独占口径） |
| :3938-3944 | `_note_action_selection` 计数表 None 短路 | 直接 `self._joint_action_selection_counts` |
| :4040-4040 | port_snapshot 的 `_hbm_ports` None 门 | 直接 `self._hbm_ports.snapshot()` |
| :5068-5070 | `verify_run_end` 计数表 `is not None` 包裹 + `_quota_deferred_dwell_ns or ()` | 解包裹（块整体 dedent）；dwell 直接 `sorted(self._quota_deferred_dwell_ns)` |
| :5115-5119 | `verify_run_end` 的 `getattr(self, "_quota_tracker", None)` | 直接 `self._quota_tracker`（**is not None 判据保留**——off/static/aimd 是真生产条件语义） |
| :2931-2940 | `_telemetry_endpoint_link_key` 的 `hasattr(self, "_telemetry_link_id_map")` 整型键透传分支 | 删分支——`__init__` 恒设 None、首样本惰性构建；替身裁定见 34.3 |
| :4768-4772 / :4791-4795 | memo 包装 `capacity=getattr(self, "_task_load_cache_capacity", 常量)` ×2 | 直接 `self._task_load_cache_capacity`（`__init__` :697 恒设） |

保留（合法缺省，行侧注释钉明，不扩面）：

| 位置 | getattr | 保留理由（生产条件） |
|---|---|---|
| :534 | `getattr(config, "remote_memory", None)` | 外部 config schema 可选段；缺席 ⇒ 紧随 fail-closed ValueError（红线 2），非静默 |
| :2024-2028 | `getattr(self, "_last_admit_failure_key", None)` | 生命周期后置：仅 `_try_admit_request` 失败分支赋值（:4185/:4247/:4349——失败分支三处），首请求成功前结构性缺席；回退键 = "多开一次门" 语义 |
| :3595 | `getattr(JointCostModel, "__dataclass_fields__", {})` | 跨模块 schema 探测（joint_cost_model.py 他卡交付面）；C7 已交付该字段、分支恒走通 |
| :3921 | `getattr(breakdown, name, None)` | `_JOINT_BREAKDOWN_LOG_FIELDS` 序列化超集 vs JCM dataclass 字段集——跨模块字段可选，非替身兼容 |
| face_scheduler.py :176/:770 | `getattr(self, name)`（2 参） | dataclass `__post_init__` 校验循环的动态属性访问，缺即 AttributeError，本就 fail-loud |

### 34.2 `_telemetry_link_id_map` 替身语义裁定（二选一，如实实现）

hasattr 透传删除后，替身走 `_telemetry_endpoint_link_key` 必须**显式**
装配该属性，两条如实路径 + 一条钉死路径（不静默）：
1. 补 `None`（配合真 hardware 网格 ⇒ 映射真实构建，生产同款）；
2. 预置整型恒等映射 `{i: i}`（等价旧 C7 期退化透传、语义自担）——
   `test_joint_fix1_pricing.test_int_key_passthrough_form_expires_too`、
   `test_joint_quota_integration.test_stub_identity_map_passthrough`、
   `test_joint_preadmit_visibility` 两窗测例按此改写（主题是窗链/过期/
   冻结接口语义，端点换算语义由 endpoint 键测例单独钉死）；
3. fail-loud 钉死：`test_stub_missing_map_attribute_fails_loud`
   （quota_integration）显式断言属性缺席 = AttributeError。
未知 id 的 KeyError→ValueError fail-closed 路径不变（生产路径未动）。

### 34.3 测试替身对齐（AttributeError 指哪补哪；值取真 `__init__` 初值）

14 个测试文件、46 处属性/接口对齐（每处带"对齐 __init__ 初值（F6 销账）"
注释）：`_task_load_cache_capacity`（9 文件）、`_quota_tracker=None`
（off 档初值，8 处）、`_hbm_ports=HbmPortFlowRegistry()/_RecordingFlows()`
（6 处）、`_link_telemetry_rates={}`（3 处）、`_telemetry_link_id_map`
（3 处）、`_joint_factors=ServiceFactors()`、`_prefill_task_cache={}`、
`_decode_task_load_cache={}`、C11 墙时埋点五键 + 计数表（decision_schema
全配）、kv_manager 替身补 `last_merge_outcome=None`（arrival）与
`kv_delta_find` 合成结算行（arrival；SettledKvManager 原有）。
**软门自钉测试改写 5 例**（不改弱断言，改钉新契约）：
`test_stub_without_factors_fails_loud`（fix1_pricing，原"软门跳过"）、
`test_watch_delivery_missing_interface_stub_fails_loud`（remote_
settlement，原"软跳过锚"）、`test_stub_identity_map_passthrough` +
`test_stub_missing_map_attribute_fails_loud`（quota_integration，原
"属性缺席透传"）、`test_int_key_passthrough_form_expires_too`（恒等映射
化）。`test_remote_credit_multiblock` 经共享 `_scheduler`（remote_
credit_stream）间接修复。

### 34.4 测试（全量，分列）

- joint/：**369 passed**；
- online/：**223 passed + 7 subtests**；
- face（workload 根 test_face_scheduler + test_sh30_kv_incremental_
  invariants）：**54 passed + 2 subtests**；
- workload 全树（joint+online+face）：**646 passed + 9 subtests**；
- sh_test_mesh/tests/：**53 passed + 18 subtests**；
- slo_tools/（tests/ 目录内跑，synthetic 导入路径约束）：**139 passed +
  1 skipped + 3 failed**——3 失败（driver_parity 2 + slo_contract 1）为
  **HEAD 先在**（§29.2 已登记"git stash 还原基线复跑坐实"；本卡 diff
  不触及 slo_tools 任何 import 面：kv_cache_adapter/hbm_watermark 对
  scheduler 仅注释引用）。与本卡无关，未修（他卡交付面纪律）。

### 34.5 冒烟（2s 缺省臂 TJE，互斥锁协议，行为不变证明）

- 前置：run 二进制曾被 clean_build_artifacts.sh 清除（§33.4 自清遗留），
  按 README §4 拼装 CMakeLists 重建（build/ 为 gitignore 区，产物留场
  供后续卡复用；cmake 配置/编译日志在 /tmp/joint_fix/F6/cmake_*.log）。
- 协议：fcntl 取 /tmp/joint_fix/sim.lock 排他锁 → 备份 trace_config 到
  /tmp/joint_fix/F6/ → 改指针 → 物化（astra_compute_20.csv 前 2s）→
  跑 → 还原（sha1=9a539b11cf03129fa9812525f70d19e49d292466 对比通过）→
  放锁。改前基线 pre_TJE_2s 与改后 post_TJE_2s 均 exit 0。
- 对比（/tmp/joint_fix/F6/smoke_compare_F6.log）：剥离已知 host 计时
  字段（online_stats 行级 processing_ns/scheduler_self_ns + 汇总行
  gil_wait_ns/scheduler_self_ns_total/scheduler_self_ns_avg_per_delivery；
  decision 行 admission_decision_wall_ns + quota_verdict_layer.
  wall_ns_total）后，四件 sidecar jsonl **逐位一致**：
  online_stats=5d45a8a2…、online_decision_log=d0bc3c23…、
  request_journal=5f10da05…（原文即同）、train_ledger=3214f2bc…（原文
  即同）；衍生产物 request_metrics.csv / slo_session.csv / slo_e2e_
  stats.csv / slo_domain_summary.json / slo_backlog.csv 原始字节一致。
- 交叉参考（非判据）：F1 fixed_TJE_2s（F1 期产物）与 post_TJE_2s 剥离
  后同样逐位一致——F1→F6 该臂零漂移。决策日志仅末行 joint_decision_
  metrics 的墙时三键随宿主噪声浮动（1352→1204 µs 级），85/86 行原文
  byte-identical。

### 34.6 边界与自清

- 零 git 操作（无 commit/checkout/restore/stash；git diff/status 仅读）。
- 本卡触碰面：sh30_online_scheduler.py（生产，唯一）+ 14 个测试文件；
  face_scheduler.py **零改动**（复审线索"face 的 kv_delta_find 相关"
  实际软门在 sh30 消费侧，face 侧为接口定义方）；joint_cost_model/
  event_recursion_predictor/graph_batch_builder/online_service/
  domain_metrics/link_quota 零触碰。
- 临时产物全部落 /tmp/joint_fix/F6/（smoke_lib.sh/run_baseline.sh/
  run_postfix.sh/pre_TJE_2s/post_TJE_2s/smoke_compare_F6.log/cmake_*.log/
  prov34.md/trace_config.csv.bak）；删除动作仅限该目录内（rm -rf 前断言
  路径）。generated/ plan 目录冒烟后清除（clean_generated）；build/ 留场
  （gitignore，供后续卡）。

## 35. 复审修复批收口（独立复审缺陷清零 F1–F6 + 九卡索引补全 + 终局核验，2026-09-22，用户授权"缺陷都要改掉"）

### 35.1 授权与流程

源 = 交付后深挖自查（主对话现场核验 + 三个只读审计 agent 分审测试质量/代码缺陷/文档账目，问题清单已呈报用户）；用户 2026-09-22 指令"缺陷都要改掉，不需要我提醒"。偏差流程履行：执行计划 §4.3 先落字补遗 **A7'（PARTIAL 驻留 recompute 计价口径）/A8'（transfer_factor EWMA 接线）/A9'（遥测速率样本缺席即失效）/A10'（零速率样本设防+配额入册失败防御回滚）**，后动工。六卡并行（文件互斥矩阵 + sim.lock 仿真互斥协议 + F1.done 门）：F1→§29、F2→§30、F3→§31、F4→§32、F5→§33、F6→§34（flock 并发追加致物理顺序≠编号顺序，编号唯一合规）。

### 35.2 修复总账（详情见 §29–§34）

- §29 F1：A7'-A10' 四件落地（PARTIAL 死分支激活审计锚 500 / transfer_factor updates=1376 / 过期剪枝 1392 epoch / 零速率丢弃+回滚）+ quota_deferred 安全网钉测（C9 冻结语义裁定=免链路动作不被配额裁、deferred 为安全网非漏实现）+ 空 shard 契约钉字。缺省臂 sidecar 逐位一致（2s 窗无 PARTIAL 驻留 recompute 决策，零漂移如实登记）。
- §30 F2：UnifiedTimeline.step ETA 截断积分丢字节修复（审计场景 3.0→2.5ns）+ 时刻-服务联动守恒断言（注入缺陷即触发，有牙齿）+ 既有缺陷期望值修正 1 处（test_sequential_victims_share_bandwidth 原期望即缺陷值）。
- §31 F3：kv_delta_journal 序列化收口（dump 第四键 21 行实测 / domain_metrics 四层分级 settlement_full_join>partial>empty>decision_log_only，certified 不冒认）——C14 §20.3→C16 §22.7-1→C19 被丢弃的移交链闭合。
- §32 F4：六组零效/弱断言修复（F7 恒 skip→占位二进制真跑 169 passed 0 skipped / 恒真式删 / 同源恒等改查表 / assertRaises 精确到 KVCapacityError / 软门自切换改硬断言 / grep 断言升提取执行式 / 猴补丁恢复入 finally）。
- §33 F5：真 C++ commit 路径多块验证——M=8（原崩溃形态）exit 0、1240 多块批次全过预检、_rcb1.._rcb8 与 graph digests content_sha256 逐批绑定、单块锚 K=8000 零 _rcb 逐位不破。C4b-FIX 验证链从"Python 镜像"升级到"真后端"。
- §34 F6：__new__ 替身软门面销账（删 17 处真软门含 C11 类级软缺省块与 F1 新增软门、留 5 处合法条件语义钉注释、14 测试文件 46 处替身对齐、5 个软门自钉测试改钉新契约）；F1→F6 冒烟逐位一致零漂移。

### 35.3 九卡索引补全（勘误登记——履行 §28.7 未尽部分）

§28.7 索引表原仅 §15-§27 十四行，缺 C0/C1/C3/C4b/C7/C8/C9/C10/C17 九行（RECOVERY.md 失守的连带缺口）。补全（交付物锚以仓内文件+git 终态 63 条为准）：

| 卡 | 交付物锚 | PROVENANCE 挂靠 | 状态 |
| --- | --- | --- | --- |
| C0 基线快照 | git HEAD 9a95e06 + C0 仓内 20 条基线（已由终态对账覆盖） | §1/§2/§28.1 | 闭合 |
| C1 JCM shard 计价+D1 | joint_cost_model.py（_transfer_ns_shards/A1' 并集除数/A2' 单遍）+ test_joint_shard_pricing.py | §23（A4' rider）/§24（C4 金值重推导覆盖公式） | 闭合（无专节，挂靠成立） |
| C3 decode 上下文 | SH decode_context_tokens/decode_average_length kwargs + test_joint_decode_context.py | §24 计数行（C3+5）/§28.6 | 闭合（同上） |
| C4b 受控测量 | σ̂ 差分受控测量三表 | §24/§26 引用 | **不可复核**（三表随 §21 /tmp 事故失守、无仓内副本——见 35.4） |
| C7 遥测 divisor | joint_cost_model.py TelemetryLinkFlowView/divisor_effective/collective_coverage + test_joint_telemetry_divisor.py | §16（C6 接线）/§28.6 计数（C7+34）/§29（A8' transfer_factor 接线收口） | 闭合（同上） |
| C8 readplan 预登记 | SH _preregister_readplan_flows 四环闭合 + test_joint_preadmit_visibility.py | §27（C11 集成报告）/§28.6 计数（C8+19） | 闭合（同上） |
| C9 link_quota | joint/link_quota.py（三类流判据/借还配对/回队）+ test_link_quota.py | §27/§29（安全网钉测）/A3' | 闭合（同上） |
| C10 AIMD+δ_adm 终冻 | link_quota.py AIMD（k=10/[r_KV,1.2r_KV]）+ test_link_quota_stability.py + DELTA_ADM_INITIAL_NS=0 | 执行计划 §4.3 A3'（终冻报告要义固化处，/tmp 原件已失） | 闭合（同上） |
| C17 face_static | joint_scheduler.py face_static 分支 + test_face_static_mode.py | §28.2（矩阵臂）/开关清单 §7.0 | 闭合（同上） |

### 35.4 C4b 三表失守确认

C4b 的 σ̂ 差分受控测量三表（证据等级 2 同状态四动作实测）随 §21 /tmp 事故失守且无仓内副本，是九卡中唯一无可复核替代的产物缺口；次级留存 = §24/§26 中的结论性引用（两族臂 exit 0、K209 单块锚）与 A6'/§33 的真后端复验。后续如需 σ̂ 真值，按执行计划 §4.3 A3' 的 σ̂ 三级链估计或重跑受控测量（需另行登记）。

### 35.5 终局核验（2026-09-22）

- **13 臂矩阵重跑**：joint_smoke_matrix.sh 2000000000ns（2s）窗口，证据根 /home/sunhao/joint_smoke_evidence_fix（C19 旧证据根保留对照）——13/13 exit 0；smoke_input queue md5 62a14f16c976c16f8259256fae505950 与 C19 证据根逐位一致（输入等价锚）。
- **全量测试（README §6 同口径命令）**：761 passed + 1 skipped + 29 subtests（C19 终态 724 → 净增 +37：F1+22/F2+5/F3+9/F6+1）；skip = phase2 journal fixture 缺失（C0 既有）；slo_tools 目录内 unittest discover = 143 ran，预存 3 红（driver_parity×2+slo_contract×1，HEAD 先在）签名与 C16/C19 基线相同。
- **裸仓还原（文件系统级核验，非 git status）**：①trace_config.csv 指针=占位 request_queue_placeholder.csv 且相对 HEAD 零 diff；②traces/ 仅 materialize_20_30s.py；③sh_test_mesh/generated/ 不存在（clean_test_records.sh 实删）；④build/ 不存在（clean_build_artifacts.sh 实删）。另清运行碎屑 sh_test_mesh/runs/.single_simulation.lock（0 字节 flock 载体）。父目录 PROVENANCE.md 误写残留核验=不存在（§33.4 的外科摘除确认干净）。
- **终态 git 对账**：joint 范围 63 条（28 untracked + 35 modified）= C0 后交付并集（C1-C19 ∪ F1-F6），无过程碎屑/BLOCKED/临时文件混入；C19 基线 58 条 → +5（test_joint_fix1_pricing.py 新文件 + 4 个首次进入 M 的既有文件，全部属修复批合法交付面）。
- 零 git commit 维持（全程 HEAD=9a95e06）。

### 35.6 方法修正固化（G5 教训）

裸仓四要素核验必须**文件系统级实测**（ls/test 实体检查），git status 对 .gitignore 覆盖目录（generated/、build/）结构性盲区——本批 §28.6"无 generated/"声称被清理后测试活动推翻的根因即此；且"清理 → 跑测试 → 未再清理"的时序错误须固化为：**任何清理后的仓内活动（含测试）之后，必须重新清理并重新核验**，核验时点=交付时点。

---

## 36. 第二轮深挖复审修复批收口（G1–G4 独立缺陷清零 + 账目勘误 + 终局核验，2026-09-22，用户授权"改啊"）

### 36.1 授权与偏差流程

源 = F 批交付后的第二轮深挖自查（主对话现场核验 + 四只读审计 agent
分审：F1 生产代码正确性 / F2+F3 正确性 / 修复批测试质量 / 文档账目
交叉；问题清单 P1×2 + P2×8 + P3×12 已呈报用户，含"已查证排除项"）。
用户 2026-09-22 指令"改啊"授权全量修复。偏差流程履行：执行计划 §4.3
先落字补遗 **A11'（P1 双件：A10'(b) 预约处置语义修正 + F2 守恒断言
通道统一与量化容差、pending 激活切割）/ A12'（P2 批：dump 逐键哨兵 +
stay 行 home_after 语义 + A10'(a) 零速率同窗失效 + transfer_factor
≥1 钳位与名实订正 + 旗标关早退失效）**，各含被否决项；后动工。四卡
文件互斥：G1 = sh30 + JCM + fix1 测试；G2 = predictor + 其测试；
G3 = face + online_service + domain_metrics + 对应测试；G4 = 纯测试
补强（quota_integration/remote_settlement）。本批零 git 操作（无
commit/checkout/restore/stash）。

### 36.2 G1 卡（A11'(a) 预约处置 + A12' 遥测/JCM 件）

- **A11'(a) 预约处置两态**（审计 P1-1：安全网自崩）：新助手
  `_release_reserved_admission_or_fail`——未物化（extra 全零）释放 +
  纪元 bump + 照常 quota_deferred 回队；**已物化**（prepare 已推进会
  话账目，输入>0 恒此态）fail-closed RuntimeError 带全上下文（
  `release_request_capacity_reservation` 对部分物化恒 raise、无
  un-prepare 逆路径；保留预约回队则重试必崩在 `reserve_request_
  capacity` 重复预约 ValueError——假回队 = 腐坏转嫁下游）。A10'(b)
  分支重排：`_rollback_admission_registrations` 先行（可逆半边恒回
  滚）；旧 `_release_orphan_reservation` 容量失败路径语义/守卫不动。
- **A12' 遥测件**：零速率窗（served=0∧active>0）丢弃时同窗
  `_link_telemetry_rates.pop`（A9' 在场核对不剪除在场键，不失效则
  陈旧高速率驻留、NoC 除数欠计拥塞）；旗标关/桥断供早退分支 clear
  缓存速率（与 A9' 缺席失效同语义）；因子喂入先于 rates 落账
  （fail-loud 不留半笔状态——原注释承诺与代码顺序不符，对调使成立）。
- **JCM**：transfer_factor 样本 ≥1 钳位 + `clamped` 计数（as_dict
  披露；prefill/decode 不钳——历史口径允许 <1）；A7' docstring 公式
  （missing = history − floor(H×prefix/L)，等价 ceil 在缺失侧）与
  行号残留订正；observe_transfer_from_link_window 钳位注记。
- **名实订正**（审计 P2-4）：sh30 `_observe_service_factors`
  docstring 与开关清单——transfer_factor 为披露/观测通道，传输争用
  的计价修正实效由链路/端口除数通道唯一承担（divisor_effective 的
  max(registered, B/measured) 合并；因子进 estimate_action 将与除数
  通道双计同一遥测信息，被否决）。
- 测试：fix1 27（22 → +5：亚名义钳位/零速率同窗失效（含重建）/
  早退改钉（旧"键缺席不过期"钉测改钉 A12' 新契约）/真实
  KVCacheManager 预约处置两态（未物化释放归零、已物化 RuntimeError
  且预约在册）+空放）；`test_stub_without_factors_fails_loud` 升级
  assertRaisesRegex 锚定属性名 + rates 空断言。

### 36.3 G2 卡（A11'(b) 守恒通道 + 量化容差 + 激活切割）

- `ConservationViolationError(EventRecursionError)`：守恒校验自裸
  assert 改 raise 级——python -O 免疫；face 调用方既有降级通道
  （保守保留全层 + recursion_error 披露，face_scheduler.py:5533）
  直接接住，"同一数值事件两通道分裂"消除（审计 P1-2）。
- 量化松弛：两条校验容差 + `Σrate×ulp(now_ns)`（finishes =
  now + rem/rate 的 catastrophic cancellation——1e12 ns 基座实证
  误杀修正；C20 长跑 ns 量级前置雷拆除）。
- pending 激活切割：激活时刻入 step() 区间边界候选（min(完成,
  ETA 到期, 下一条激活)），截断/完成两分支合一为同一分段恒速积分
  式——兑现类 docstring"任何事件重算速率"承诺，消除在册流"提前
  完成"的可行性乐观偏差（审计 P3-5）。
- 测试：predictor 34（30 → +4：注入×2（字节账目/时刻联动，且证
  ConservationViolationError isinstance EventRecursionError——"有牙
  齿"落地为测试，审计 P3-3）/ 大基座（now=1e12）无误杀 / 激活切割
  金值 [1900,2000]（原 [1000,2000] = 缺陷值——probe4 场景））。既有
  [1000,1500,1917] 金值不受影响（全部 start=0，实测通过）。

### 36.4 G3 卡（A12'(a)(b) dump 哨兵 + home_after 语义）

- `dump_joint_kv_ledgers` 逐键独立导出：单键失败写
  `<key>_export_error` 哨兵字符串（其余键不受连坐；kv_delta_journal
  的 seq 断链 fail-closed 经哨兵显式落盘——审计 P2-1：旧行为被兜底
  except 吞成"四键尽失 + 消费端误判零结算 + deep_gap/copy_handoff
  连坐"）。诊断通道"不得阻断主异常路径"纪律保留（外层兜底仍只管
  落盘自身失败）。
- kv_delta 行 `home_after`：winner 缺席（stay 等无胜者出口）取
  home_before（审计 P2-2：原 None 被消费端当独立 home 计入集合，
  sessions_with_multiple_homes 乒乓指标假阳性）；消费端 homes 集合
  过滤 None（对齐 _completion_trajectory 口径，旧 sidecar 防御）；
  哨兵键/顶层非 dict JSON → decision_log_only 层 + 明示注记（证据
  等级不虚标 settlement_empty，审计 P3-8）。
- 测试：face KV 12（+1 dump 哨兵：注入字段漂移 ⇒ 哨兵落盘 + 其余
  三键在场）；domain_metrics 27（24 → +3：哨兵层/非 dict 层/乒乓零
  假阳性双形态）；两处旧 home_after=None 钉测改钉新契约
  （test_face_scheduler.py stay 行、test_joint_remote_settlement.py
  stay 行——均补 home_migration=False 断言）。

### 36.5 G4 卡（测试补强）

- 压力态安全网（⇒ 方向，审计 P3-11）：链路门（Q_init=2 占满）+
  端口平价门（2 条 r̂=50 realtime）双占满下 copy 确被拒而
  stay/recompute 恒适用——空 tracker 用例无法区分"恒适用"与"恰好
  配额空闲"的缺口补上。
- 占位二进制 sha 锚定（审计 P2-1/P3-12 降级处置）：
  invocation.json 的 binary_sha256 == 占位内容 sha256（runner 半
  绑定 sanity；C++ 接口漂移的真防护 = F5 真 commit 路径验证 + 13 臂
  矩阵真跑，单测不冒充）。
- assertRaisesRegex 锚定属性/接口名 ×3（fix1 `_joint_factors`、
  quota_integration `_telemetry_link_id_map`、remote_settlement
  `kv_delta_find`——别的属性先炸不许冒名通过，审计 P3-3）。

### 36.6 勘误与账目补登记（append-only，不改写前文）

1. **§29.4(b) 表述失实勘误**：F1 卡"事务段/流登记/预登记全真实"——
   当时 reserve/prepare 为 lambda 打桩（`_reservations` 恒空、守卫走
   空放），"回滚归零"在真空里通过（审计 P1-1 的测试半边）。本轮
   G1 真实 KVCacheManager 两态测试补齐 + 生产语义修正（36.2）。
2. **§33 F5 launch 账目算术勘误**："4 次（mb_autok/mb/sb_k8000 三次
   弃跑 + mb3/sb3 登记臂）"枚举 3+2=5≠4——正确口径 = 弃跑 3 + 登记
   臂 2 = 5 次 C++ launch；另补登记矩阵第一次窗口参数误传 `2`
   （=2ns，物化即 ZeroDivisionError、sessions=0）弃跑 1 次（矩阵
   调用不占 C++ launch 预算口径，但属弃跑应入账）：矩阵弃跑 1 +
   成功 1。
3. **§35.5"C19 基线 58 条"构成不可重构**：权威载体
   /tmp/joint_exec/C19/DONE 已随过程清理失档，C 卡章节验收清单广泛
   提及测试文件名致逐名重构不可判定（本轮以 basename grep §1–§28
   法实测零候选输出）——如实登记为不可复核锚（终值 63 两次实测
   为真；G 批全部编辑既有文件、无新条目）。
4. **/tmp/joint_fix 计划内清理补登记**：§35 收口时按过程文件纪律
   删除（结论已固化 §29–§34），但未在 §35 落字——六卡引用的
   /tmp/joint_fix/... 路径自此为无生命周期注记的死链（对照 §21
   第一次 /tmp 误删事故有 RECOVERY 账）；/tmp/joint_exec 空目录
   （19:17）残留同批清理。本轮 /tmp 工作面 = /tmp/cmake_config.log、
   /tmp/cmake_build.log、/tmp/prov_s36.md、/tmp/strip_compare.py
   （结论固化本节后清理）。
5. **C 系列卡历史 git stash 用法注记**（§16 前：A/B/A stash/pop、
   "git stash 还原基线复跑"）：字面触及 Agents.md"禁 stash 目标文
   件"红线——当时已登记、已还原（"19:32 mtime 异动系 stash/pop
   副作用、内容未变"）、无损失；F/G 批零 git 操作纪律维持。方法
   承诺：后续基线对比一律用工作树副本，不再用 stash 往返。
6. **计数勘误**：F1 分项 22 = 4/5/6/**3**/2/2（"A10' 2 例"漏计
   (b) 配额回滚 1 例）；F6"5 个软门测试改钉 AttributeError"实为
   3 处 AttributeError（另 2 处钉恒等透传）——3 处本轮已补
   assertRaisesRegex 精确锚定；F6 销账计数 17/18 三处不同构维持
   原文（§34.1 头"17 处 getattr/hasattr + 1 类级块"为最细口径）。
7. **F 编号空间撞名登记**：执行计划冻结项 F1–F15 与修复卡 F1–F6
   同仓并用（README 内 F7=冻结项 8 处 vs F1–F6=修复卡 7 处，§3 表
   内同排）——README 已加消歧注记（本批）；后续复审修复批改用
   G/H 序列避免续撞（本批 G1–G4 已启用）。

### 36.7 终局核验（核验时点 = 交付时点，§35.6 方法修正固化）

- **全量**（README §6 命令）：775 passed + 1 skipped + 29 subtests
  全绿（761 → +14：窄命令 +11 = fix1+5/predictor+4/face+1/
  quota+1；slo_tools +3 = tier 哨兵/非 dict/乒乓）。**窄命令**
  （workload/llama2_7b_inference）：657 passed + 9 subtests
  （646+11；F 批窄命令 646 = 614+32 当批漏登，README 本批补锚）。
  **slo_tools 目录内** unittest discover -s tests = 146 ran
  （143+3），2f+1e+1skip 预存签名与 C16/C19/F 批基线逐签名相同。
- **13 臂矩阵**（window=2000000000 ns，证据根
  /home/sunhao/joint_smoke_evidence_fix2；_fix 保留为 F 批证据）：
  13/13 exit 0；queue md5 62a14f16 与 C19/F 批锚一致。**漂移判定**：
  剥离 host 计时（decision 行 admission_decision_wall_ns +
  quota_verdict_layer.wall_ns_total（含收尾汇总行）+ online_stats
  行级 processing_ns/scheduler_self_ns + 汇总 gil_wait_ns/
  scheduler_self_ns_total/scheduler_self_ns_avg_per_delivery）与本
  批新增披露键 service_factors.clamped（as_dict 增量，空 dict）后，
  13/13 臂 online_decision_log/online_stats/request_journal/
  train_ledger 与 F 批证据根**逐位一致**——激活切割与零速率同窗失
  效在 2s 窗内零决策漂移（当前拓扑 start≤now，实证暴露面小）；仅
  quota 两臂多出收尾行 wall_ns_total 差异（wall-clock，剥离后同）。
  bridge sidecar 差异 = stay 行 home_after 语义修正（16-20 行/臂，
  逐行核对零其他差异）。
- **裸仓还原与卫生**：build/（本批按 README §4 配方重建用于矩阵）+
  generated/（矩阵 plan 物化）+ 仓内 __pycache__（测试物化）清理后
  四要素文件系统级核验；仓根 `9`（0 字节身份不明文件，20:29 战役
  窗内产物）删除；/tmp/joint_exec 删除。
- **git 对账**：joint 范围 63 条（28 untracked + 35 modified，与
  §35.5 终态同数——G 批全部编辑既有文件、零新增条目、零过程碎屑）；
  零 commit（HEAD=9a95e06 不变）。
  （补记：矩阵三 runner 臂经 joint_runner 落 `sh_test_mesh/runs/
  .single_simulation.lock` 锁目录——clean 脚本清单未覆盖，交付核验时
  发现为第 64 条未跟踪条目，断言仅含锁文件后删除，git 复测 63 条；
  硬件目录 d2d_sub/x05/x2 三配置为用户 15:47 并行创建的 D2D 消融
  准备件，在 §35.5 的 28?? 基数内、非本批产物，保留。）

## 37. 第三轮深挖复审修复批收口（H1–H10，2026-09-22，用户授权"深挖+自己修改"）

### 37.1 授权与偏差流程
- 用户指令："你再进行一轮深入的自我检查，把这次改造存在的问题都
  深挖出来，然后自己修改"——第三轮深挖，审计对象 = G 批（§36）
  交付面自身。
- 偏差流程履行：先落字（执行计划 §4.3 补遗表追加 A13'/A14' 两行，
  含被否决项与五项不修处置）后动代码。
- 审计方式：本人基线核验（git 63 条/HEAD=9a95e06/裸仓四要素/全量
  775 基线复现）+ 四只读审计 agent 并行交叉（调度器面/predictor
  数学/成本模型+G3 三件/测试+文档账目），P2×3 证据链逐一亲验后
  采信（通道②零钉住经 _assert_conservation 签名与 post 派生式推
  导证实；夹具假 GREEN 经 :192/:236 与 HEAD 旧实现对读证实；注释
  残留三处逐行读证）。
- 审计结果：P1=0、P2=3、P3=22；三项 P2 全部修复，P3 修复 10 项、
  不修处置 5 项（理由落字 A14'）、其余为审计过程中已排除的疑点
  （如开关清单 :254"消费三面"疑漏订正——实测 A12' 标注与③订正
  均在场，疑点不成立）。

### 37.2 H1 卡（A13'(a)：stress 夹具逐键哨兵门禁收紧）
- 缺陷（审计 C-P2）：G3 逐键哨兵把旧 fail-closed 拆掉——deep_gap
  导出失败 ⇒ 键缺席 + `deep_gap_events_export_error` 哨兵在场 ⇒
  夹具 `deep = ledgers.get(...)` = None ⇒ 硬门禁 `not deep` 恒真
  ⇒ 假 GREEN（对照 HEAD 旧实现：单 try 包整体 json.dump，任一生
  产器失败 ⇒ 侧车整体缺失 ⇒ sidecar_present=False ⇒ FAIL）。
- 修复：`export_errors` 扫描（`key.endswith("_export_error")`）入
  硬门禁（`not export_errors` 与 sidecar_present/not deep 同列）+
  judge_summary 披露清单；恢复 G3 之前 fail-closed 强度且不连坐
  逐键导出其余收益。
- 测试：quota 文件接线钉 test_stress_fixture_gate_fails_on_
  export_sentinels（哨兵扫描/门禁项/键名拼接与生产端一致；执行级
  验证需完整 run 目录三工件夹具，成本与收益不成比例，文本级接线
  钉 + 理由注释——同 test_shell_defensive_assertion 的提取执行
  范式为邻证）。

### 37.3 H2/H4 卡（A13'(b) 守恒通道②真钉住 + A14'(a) predictor 加固）
- 缺陷（审计 D-P2）：原 test_time_linkage_violation_raises 中
  step() 已把 10000B 流结算移册（post=0），①式 |pre−0−0| 先炸——
  ②从未被评估，删②测试仍绿（零钉住）；byte-theft 测试注释
  "post=100 流仍在册"与实际 post=0 不符（①仍真钉住，注释失实）。
- 重写：两测试各成隔离钉——①钉（served=500=pre−post 使②式
  |500−1000×0.5|=0 恒过、①式 |1000−0−500| 炸）+ ②钉（served=
  1000=pre−post 使①恒过、rewind now=0.5 使②式 |1000−500| 炸）
  ——assertRaisesRegex 钉通道名（"byte conservation"/"time-
  service linkage"），删任一通道对应测试即红。
- H4 加固四件：(i) EfficiencyFactors.__post_init__ 验证 η/γ 有限
  正（η=0 在 finishes 计算除零且逃出 EventRecursionError 族击穿
  face 降级通道、η=inf 零代价瞬时完成且 quant_slack=∞ 掩蔽两条
  守恒——与 ResourceSnapshot 对 peak 的验证同教义；生产路径 EWMA
  恒正有限，仅直连误用可触发）；(ii) add_flow 重复 id 检查补在册
  active（原只查 pending/completions——同 id 二次添加双份份额且
  next() 只结第一份）；(iii) start_ns 有限性（NaN 永不激活且空活
  跃分支 max(now,min([nan]))=now ⇒ run() 死循环；inf 推时刻到无
  穷）；(iv) 幸存流负尘埃零化（−(1e-9+rate×ulp(next_time)), 0 内
  钳 0——不零化则下一步 finishes<now 时刻回退、大基座下尘埃可达
  rate×ulp(now)≫1e-9 误炸 drained-below-zero 且根因误标；真破损
  仍由阈值拦截；完成流在 post 求和前已移除，零化只影响幸存流且
  ①式残差只会更小）。
- 测试：InputHardeningTests 四例（η/γ 四坏值×双通道/重复 id 在册
  态/非有限 start 三形态/大基座激活切割回归——1e12 基座 + 333B@
  t 与 777B@t+5 共享 7B/ns，金值 {t+91, t+159} 手算复核 + 全程
  时刻单调断言）；predictor 文件 38 = 34+4。

### 37.4 H3/H5/H6 卡（A13'(c) + A14'(b)(c)：sh30 面）
- H3：三处"整数截断伪影"残留订正（__init__ 计数注释/_ingest_
  link_telemetry docstring 第三条/_telemetry_coverage_decision
  docstring 与行内注释；另测试内联"伪影样本/伪影键"简写随钉）——
  A12'(c) 定性订正只改了执行点（:3345-3346）与外部文档，同文件
  只读面漏改、新旧定性并存自相矛盾。
- H5：served>0 ∧ active==0 样本 fail-closed（ValueError，消息含
  "differential contract"）——uint64 差分契约（有载荷必有活跃时
  间）下的不可能形态；原为本入口唯一病态样本静默通道（不落速率/
  不计数/不喂因子，却以在场键豁免 A9' 剪除、钉驻陈旧速率），与
  未知 link_id 的 fail-closed 教义对齐封死；双零样本维持良性在场
  语义（既有钉测不动）。测试：test_positive_bytes_zero_active_
  fails_closed。
- H6：防御分支（quota 判决翻转，_drive 注入可达）两处口径对齐——
  failure_class 裸 "quota_deferred" 改 "quota_deferred_quota_link"
  （决策日志 wait_reason 与 _quota_deferred_wait_counts 计数键同
  源，消"日志判 capacity/指标计 quota_link"矛盾；防御分支不掌握
  翻转侧别，沿指标侧既有归类）；重试门键 {selected, chosen.
  instance_index}（恒单元素——chosen.instance_index 即 selected）
  改"applicable 候选 ∪ selected"并集（对齐容量/配额两路径口径：
  argmin 翻向可行候选时重试门随负载迁移重开）。A10b 旧钉测改钉
  新 failure_class + 补 wait_reason=="quota_link" 同源断言。

### 37.5 H7/H8/H9/H10 卡（A14'(d)-(g)：消费端/JCM/测试卫生/文档）
- H7：domain_metrics 值级层判贯彻——kv_delta_journal 键在而值非
  列表（schema 损坏）→ decision_log_only（原与空列表/键缺席合并
  判 settlement_empty 且 note 断言"零结算 run"，顶层 A12' 原则在
  值级未贯彻）；键缺席仍走 EMPTY（F3 前旧 schema 兼容契约，既有
  钉测不动）；顺带清除判 dict 后的 `else None` 死分支。测试 +2：
  值非列表层（note 含"键值非列表"）、单会话全 None home（计入
  sessions、homes 集合空 ⇒ 乒乓指标不可见——不崩溃不假阳性）。
- H8：JCM _recompute_missing_tokens 的 ceil 分支去掉 location==
  "partial_hbm_remote" 额外门——resident_here ∧ prefix<L 恒 ceil
  （LOCAL 基且 prefix<L 被"转 LOCAL 恒置 prefix=layers"不变量排
  除、生产不可达；原该形态落穿 return history 整份计费，与 docstring
  "和执行侧逐字节一致"宣称不符）。测试：LOCAL 基防御形态钉测
  （H=1001/L=32/p=16 ⇒ 501，修正前恒整份 1001）。
- H9：face dump 哨兵测试的 importlib 动态导入/sys.path 插入改
  try/finally 回收（sys.modules 残留 + 路径序滞留会让跨目录收集
  的 pytest 会话命中缓存、错载兄弟仓同名 online_service 模块——
  全仓 50 份同名文件的实测量级）。
- H10：JCM clamped 计数口径注记（计"同 tick 聚合更新钳位条数"
  而非"亚名义链路窗口数"——Σactual/Σbase 聚合稀释 + base_ns 向
  下取整偏乐观；因子恒 ≥1 不变量不受影响）；predictor 测试
  "assert 级守恒复核"过时表述订正（A11' 后为 raise 级）。

### 37.6 勘误三件与不修处置五项
- 勘误 1：§36.4 "测试：face KV 12" 计数基不可复核——
  KVDeltaJournalExportTests 类实为 5 例（F3 期 4 + G3 期 +1），
  以"+1 dump 哨兵"增量为准；底数 12 无口径来源（第三轮审计
  D-P3-2），不以底数引用。
- 勘误 2：§29（:2891"整数截断伪影样本"、:2967"本窗口无整数截断
  样本"）与 §30（:2805"步末 assert 级复核"）为历史节旧定性——
  append-only 不回改，本行为回指锚：读者沿 §29/§30 须以 §36（A11'/
  A12' 订正）与 §37 为准。
- 勘误 3：README G 批块 fix1"+5"枚举把净增 0 的早退改钉计入六项
  列举——已就地澄清（README 非 append-only：净增 5 = 钳位/零速
  率/A10b 三例，早退改钉净 0）。
- 不修处置（A14' 落字，理由见执行计划）：EWMA 断供不复位（历史
  统计量语义、非"当前值"缓存）；clamped 不入 run 末 coverage 行
  （该行语义=遥测覆盖证据非因子状态）；fail-closed raise 无决策
  日志行（桥 error response+stderr 为本仓 raise 类披露制度位）；
  防御分支回队路径 runtime 字段驻留（消费者以 admitted/队列为门，
  重试全量重算）；base_ns=int() 截断不改 round（动因子数值 ⇒ 决
  策面漂移，无冻结依据）。
- 审计已核验无恙项择要（不再改动）：_telemetry_zero_rate_dropped
  初始化→自增→_telemetry_coverage_decision→verify_run_end 落链
  完整；早退 clear 可达且幂等；RuntimeError fail-closed 全链无吞
  点（桥 except Exception = 转译 error response + sys.exit(1) 非
  吞）；全零 pony 预约释放无内部断言风险；钳位样本级作用 + EWMA
  归纳恒 ≥1；A7' 恒等式（整数 H 平移不变性）与执行侧逐字节比对；
  守恒式紧但充分的量化松弛推导；ConservationViolationError 唯一
  捕获者丢弃对象、无腐蚀复用；开关清单 :254 三面消费与实现逐条
  对应。

### 37.7 终局核验
- 测试三口径：全量 784 passed + 1 skipped + 29 subtests（775 →
  净增 +9：窄命令 +7 = fix1+2/predictor+4/quota+1；slo_tools +2
  = 值非列表层/全 None 会话）；窄命令 664 passed + 9 subtests
  （657+7）；slo_tools unittest discover -s tests = 148 ran（146+2），
  预存 3 红签名与基线逐名一致（driver_parity ×2 + slo_contract
  ×1，HEAD 先在）。另有净 0 三处：守恒①②隔离重钉、A10b 改钉、
  face importlib 回收。
- 矩阵处置：13 臂冒烟矩阵不重跑——本批全部修复对合法输入零行为
  变更（注释订正/不可达分支一致性/非法输入 fail-closed 加固/消费
  端值级层判/夹具脚本/测试文档），决策日志与统计输出面不变（A14'
  落字；F/G 批证据根 _fix/_fix2 仍为最近一次行为面锚）。
- 裸仓/基线：git joint 范围 64 条 = 28 untracked + 36 M（G 批收口
  63 + 1 = 夹具脚本本批由 clean 转 M，全部增量可解释）；全仓 92
  条（+1 同源）；HEAD=9a95e06 零 commit；traces/ 仅
  materialize_20_30s.py；trace_config.csv 指针 request_queue_
  placeholder.csv（占位语义 = 调用方物化，裸仓缺席即正确态）；
  __pycache__/.pytest_cache/build/generated 清零（本会话测试运行
  产物已清，extern/graph_frontend protobuf 缓存一并）；runs/ 锁目
  录零残留。
- /tmp：本批过程文件零落盘（审计与修复全程无临时脚本）。

## 38. K 批（K1–K8）：外部六路审计（kimi 2026-09-23）P1×6 + P2×11 + P3 择要处置

### 38.1 授权与偏差流程

- 授权：用户转交 kimi 六路审计报告并指示"你觉得对的就要改，你觉得不
  对的需要深入分析最后给出依据"——逐条读码亲验后裁定（P1×6 全属实；
  P2×11 全属实/成立，其中 P2-6 定性为承诺降级处置；P3 择要属实部分
  修、其余落字），处置落字执行计划 §4.3 补遗 A15'（先落字后动代码）。
- 偏差登记：A15' 初版漏登 P2-5（裁定属实后补登记⑤）；P2-5 处置后由
  "JCM 门排除"改为"物化侧 N9 豁免"（发现 2026-09-17 裁定③已把
  REMOTE 基 copy@home 定为 in_place 免单形态、JCM 计价面有既有钉测
  RemoteBaseAtHomeMergePricingTest——物化侧豁免才是与裁定闭合的最小
  修复，A15' ⑤文字同步改写）。
- **矩阵处置（与 H 批相反）：本批对合法输入有行为变更**（K5 计价/
  K6 门控时序/K7-② AIMD 动态/K7-④ 遥测数据源/K1 r̂/K2 终轮闭合），
  13 臂冒烟矩阵重跑 + 证据根换新（_k）。

### 38.2 K1/K2 卡（sh30 调度器）

- **K1（P1-①）**：`_quota_r_hat_kv_bytes_per_ns` 把 `state.active_decode`
  成员（_OnlineRequestRuntime 对象）当 request_id 字符串查
  `runtime_by_request_id.get()` 恒 None ⇒ contexts 恒空 ⇒ "活跃 decode
  平均上下文"口径从未生效（平价门/remote-read 入册/AIMD
  observe/port_snapshot 四处 r̂ 全错，审计实证差 2.2 倍）。修 = 直接
  迭代对象（`member.prefill_context_tokens + member.decode_tokens_
  consumed`）。测试：成员平均 vs 代表值自洽等值钉（非空 active_decode
  下 777 代表值不再泄漏）。
- **K2（P1-②）**：`_quota_on_service_done` 在 `following is None:
  continue` 之前执行（:1753），胜者侧 merge 预留保留至 _on_merge_done；
  而 watch 仅 `following is not None` 注册（:1825）——终轮等不到
  merge_done ⇒ 预留滞留 ⇒ verify_run_end :5202 fail-closed abort；
  次生：终轮 merge 流不进 R15 登记（:1855 同分支内）。修 = 注册条件
  收窄为 `runtime.merge_transfers` 单条件，pending 记录 following 三
  字段终轮落 None、披露行 `next_turn_arrived_before_merge_done` None
  条件化，`_register_transfer_flows` 随同出分支。被否决："终轮
  service_done 即整体释放胜者侧"（绕过 C9 冻结"胜者侧到 merge_done"
  的结算事实联合断言语义）。测试：`_on_merge_done` 终轮 pending 记录
  行为测（预留清零 + None 字段披露）+ 接线文本钉 + 既有
  test_terminal_turn_schedules_no_alarm 按新契约改钉（无到达 alarm
  不变、watch 现照常注册）。

### 38.3 K3/K4/K7 卡（link_quota / predictor）

- **K3（P1-⑤）**：reserve_merge 端口 bulk 门逐方向独立验 `remaining <
  1` 不聚合（链路侧 :1045-1049 有聚合）——port_forward ==
  port_reverse 时需 2 槽只验 1（N_bulk 可破）。修 = 端口需求先聚合
  再统一裁决（与链路侧同构，verdict reason 披露 aggregated need）。
  生产调用恒异端口（remote-read target≠source），缺陷为原语层
  （docstring"双向各 1 槽"承诺在同端口下被破坏）。测试：N_bulk=2
  同端口双候选第二预留拒收 + 双槽空闲时 admitted。
- **K4（P1-⑥）**：零字节恢复腿无条件 `done = base_now +
  first_block_wait` 无视同 rank 前驱（:746-748）⇒ 后继组提前解锁
  （审计实证 g0=3e6/g1=0/g2=1e6 @1000B/ns 低估 50%）。修 = 零字节腿
  等 `(leg_index-1, rank)` 完成事件（done = max(前驱 done, now+q̂)；
  首腿无前驱仍 q̂ 即时就绪）。测试：审计同形三腿串行链金值（层 1/2/3
  = 3000/3000/4000）+ 首腿即时性。
- **K7-①（P2-4）**：quota 压塌 {recompute} 被标 evicted_permanent
  （C5 冻结枚举无 quota 诱因可落）——**规格变更**：forced_reason 枚举
  扩 `quota_deferred`（检测 = 非 recompute 候选 inapplicable_reason
  前缀 "quota_"；优先序 no_history（物理成因）> quota_deferred（策略
  成因）> evicted_permanent；决策行 schema docstring 与
  `_note_action_selection` 计数键 `recompute_forced_quota_deferred`
  同源；domain_metrics 读决策行不重推导，新值自动流经）。测试：四态
  分位（quota 塌/结构性塌/首轮/多适用 elected）。
- **K7-②（P2-7）**：AIMD 缺测段被 comfort streak 吸收——重观测 dt 含
  整段缺测时长，违反冻结原文"streak 冻结不进不出"（模块 docstring
  :96-98 自证）。修 = 逐调用序号连续性门（`_telemetry_seq` +
  `_AimdLinkState.last_seq`：衔接才计 dt；缺测不进不重置），per-link
  披露加 `contiguous` 键，aimd_link_state 快照加 last_seq。测试：缺测
  中间调用后重观测 dt 不进 streak（quiet=0、无扩张）+ 连续采样恢复
  计 dt + 连续 comfort 仍扩张（回归锚）。
- **K7-③（P2-8）**：set_link_quota 同值调用 bump 配额代数（deferred
  虚假唤醒，与"非释放侧事件零 bump"纪律不对齐）。修 = 同值零 bump
  （内部 AIMD 调用本就带 target≠current 守卫，外部调用面无同值场景，
  原语层纪律对齐）。测试：两次同值不 bump、一次真变化 bump 1。
- **K7-④（P2-9）**：F7 耦合只注入旗标不注入数据源——ASTRA_LINK_
  OBSERVER 缺省 0 ⇒ C++ 观测门关 ⇒ link_telemetry[] 每 epoch 恒空
  （main_online 注释自证"observer off the totals are all zero"）⇒
  官方 TJE_quota_aimd 臂空转全绿。修 = run_online_strategy.sh：
  SH_LINK_TELEMETRY=1 且未显式置 observer 时缺省置 1（显式 =0 仍被
  尊重 = 刻意 no-signal 对照臂）；冒烟矩阵注释同步（aimd 臂从"合法
  no-signal 路径"改真闭环）。**C20-③ 前置条件**。
- **K7-⑤（P2-10）**：domain_metrics 离线平局序 (cost, action_rank,
  instance) vs 在线 (cost, instance, action_priority)——注释却称
  "replay_mismatch 应为 0"（三向平局证伪）。修 = `_min_cost` 键镜像
  在线全键序 (cost, instance, _ACTION_RANK, action)。测试：双实例
  平局取 instance 0（旧序取 stay@1）。
- **K7-⑥（P2-11）**：quota_admissible 列恒 NA（C11 后 inapplicable_
  reason 可导出而 §22.7-2"自动携带实测值"未兑现）。修 = 提取
  `_quota_admissible_flag` 助手（候选 applicable ∨ quota_ 前缀理由 ⇒
  1；纯结构性 ⇒ 0）接入行构造——"双域同图、两域之差即配额作用量"
  对 quota 臂可算。测试：助手三态 + 既有 NA 钉测改钉派生值（CSV 字符
  串形态 "1"）。

### 38.4 K5 卡（JCM 计价 + FS N9 豁免）

- **K5-①（P1-④）**：merge 计价取 min(forward, reverse) 而物化侧 =
  字节比大小 + F15 平局前向（face :4977-4980）——判据双源，"保留量接
  近 + home 侧空间紧张"场景系统性低估 remote-read（argmin 偏乐观；
  test_home_wait_in_forward_formula 把不一致钉成期望）。修 = 计价复用
  物化判据（`Σ exec_retained ≤ Σ home_retained` → forward，notes 落
  merge_v2_direction/rule 两键；KVCapacityError 容量兜底翻转不可计价
  落 note 披露）。测试：平局含等号取前向钉 + 既有 home-wait 测试重钉
  （merge_ns = 前向 1090，反向 610 虽廉不取）。
- **K5-②（P2-1）**：反向腿沿前向路径计价（自身重叠份额/已登记他流/
  端口除数全按错误方向；注释"方向差异由流登记表的调用方覆盖"无对应
  实现）。修 = 反向腿用 `_route_paths_pair(home, exec)` 自身方向路径。
  测试：前向边背景流抬 forward 311 而 reverse 不动 610（方向特异实
  证）。
- **K5-③（P2-2）**：`reclaimable_bytes_by_tp_rank` 缺省空元组静默关闭
  深缺口升级（`and reclaimable` 守卫把"无信息"当"全部可回收"，方向
  反了；生产 SH 恒显式供给、缺省臂不可达）。修 = 去守卫（空 = 零可
  回收 ⇒ 保守侧）。被否决：改 fail-closed raise（击碎既有合法视图构
  造面）。测试：空 reclaimable + 活跃负载 ⇒ 深缺口 max(写回, 活跃)。
- **K5-④（P2-3）**：深缺口 running+active 与 target_wait 双计
  （target_wait = queued+running+active 已含活跃排空视界，加法合成
  重复计、高估——R3' 遗留）。修 = `_eviction_wait_estimate` 增
  covered_wait_ns 参数（主调用传 target_wait：深缺口分量 = max(0,
  active_remaining − covered) ⇒ 主位分量恒归零；home/exec-merge 侧传
  0 语义不变；note 语义 = 深缺口条件成立，与"额外等待"两事实分列）。
  测试：covered = target_wait ⇒ 等待 = 写回 + 零 covered ⇒ 分量完整
  保留。
- **K5-⑤（P2-5）**：copy@home × REMOTE 基地雷——JCM 认定适用且裁定③
  按 in_place 免单计价（RemoteBaseAtHomeMergePricingTest 钉住在案），
  物化侧 merge_back N9 对 home==exec 一刀切 raise ⇒ 确定性 abort
  （裁定③从未被 N9 放行，交叉象限零测试）。修 = **物化侧 N9 豁免
  pool-only 基**（REMOTE 基/前缀 0 不 raise——方向裁决既有
  `not home_shards → in_place` 分支即裁定③就地转正；R13/N9 消除的是
  "基础在 home 的 copy@home"，无主基不在其列；N9 docstring 更新；
  JCM 计价与裁定③测试原样保持）。测试：FS 物化级
  （_seed → 整份逐出 → prepare_prefill copy@home → merge_back：零
  传输、终态 LOCAL@home、direction=in_place）。

### 38.5 K6 卡（GB copy 层段门控）

- **K6（P1-③）**：copy 交接体块门只挂尾块 c（块 c 等尾块 c）而
  emit_pass 是全层聚合 pass——体块 1 计算尾块 2..M-1 层不等其到达
  （时序乐观，违反 C13(4)"未到达块按就绪事件等待"冻结原文；生产
  llama2_7b 32 层 = 4 chunk 必现，16 层夹具 = 2 chunk 退化正确恰好测
  不出；对照同文件 C15 restore 的 emit_layer_segmented 逐层段门控是
  正确处理，同一约束两处不一致）。修 = (a) 尾块发射登记层区间账本
  `_copy_handoff_layers[rid][chunk] = (layer_start, layer_end)`（生命
  周期与 arms 同步：体消费弹出/完成残留 fail-closed）；(b)
  emit_layer_segmented 泛化——段集 = 热前缀 [0, 首段起点)（copy 块 0
  走准入主链）+ copy 尾块层段（逐 rank recv 门）+ restore 组层段（逐
  rank 写完成门），连续铺满 [0, L] fail-closed（尾部零 KV 区间无门
  段补齐）；(c) 首体块层段化时跳过块粒度 arming（段门精确接管；块
  ≥1 维持整段发射——per-rank 链序在首块末段之后，既有 per-block 门
  成为冗余无害保险）；(d) M=1 单块路径（旧单笔锚）不动、C15
  restore-only 路径逐字节不变（copy_layers 空走原分支）。测试：M=3
  （24 层）结构钉——尾块 2 recv 门 arm 进体节点（修前不进）+ 层区
  间账本消费弹出 + 尾块 1/2 双门分段独立；16 层既有 copy 测试族全绿
  （M=2 首块单门段化，门仍进体节点——既有断言兼容）。
- **矩阵影响**：copy 被选中的臂结果面变化（块 1+ 计算不再早于尾块
  到达——乐观偏置移除）；行为变更为本卡目的。

### 38.6 K8 卡（P3 批 + 勘误）

- 输入加固：predictor 数值字段 NaN/inf 拒收（RestoreGroupLeg.
  bytes_by_rank/startup_ns、ComputeLayerSegment.base_ns_by_rank/
  memory_bytes_by_rank、WritebackVictim.bytes_by_rank/startup_ns 补
  isfinite——NaN 经 `<0` 检查存活到 int() 以裸 ValueError 逃出
  except EventRecursionError 降级通道）；EvictionPlan satisfied 早退
  返回等长零向量（空元组被消费端 zip 静默截断）；
  ServiceFactorGroup.observe_valid_service 的 cold_start 只在接受样本
  后翻转（组内拒绝不 raise 而早退——以样本计数判接受）。
- 弱断言修拓：test_future_info_isolation 恒真 assertTrue 改实钉
  （未来行字段值不进输入统计/服务因子两可见通道）；
  test_recursion_matches_closed_form 名实订正（断言面 = ≥，等号仅
  异缘形态）；test_link_quota 平局三态去重复行；
  test_read_prefix_share_conservation_pinned 截断敏感重钉（3009/rank
  使 6018/6024 分桶——原 6000/6006 两桶同为 600 钉不住任何口径，
  D1 账本真值语义入 docstring）；test_joint_fixes 空集恒真 all() 删
  除。
- 注释订正：JCM 合成注释"保守串行"与 max() 矛盾（改"max 重叠合
  成"）；_shard_paths"保守方向"标反（(f+n)/(n·(f+1)) ≤ 1 = 广播口径
  偏乐观，生产 route_paths_fn 恒注入不可达）；quota_deferred_requeue
  "永不 raise"与调用方 bug fail-closed 矛盾（改"对合法输入不
  raise"）。
- **勘误（append-only，本节回指）**：§20.2/§1585 的"getattr 软门/
  替身缺接口软跳过"描述已被 F6（§35）推翻——现行实现直接调用
  （AttributeError fail-closed），旧文按历史时序保留不作改写；
  §20.5/:2864 同款。CopyHandoffJournal/RestoreGroupJournal 的
  assert_conservation 为 **journal 内部恒等式自检**（各项同源于
  released_indices/chunk 状态；物理账交叉核验在
  _check_invariants_after_mutation 通道）——此前"守恒审计"措辞名目
  偏强，本节校准（审计实证：绕过 journal 物理释放一块后 journal 自
  检仍通过、物理通道捕获）。**P2-6 承诺降级**：C13(3)"同一交接完成
  事件中立即释放 home 侧对应 HBM"的实现语义 = prefill drain 边界批量
  结算（_settle_copy_handoffs docstring 如实描述；因果序安全、不等
  轮末、方向保守 = 高估驻留窗）——真逐块立即释放需图侧逐 chunk
  watch 新通道（机构件级），登记为未兑现偏差而非宣称已实现
  （A15' 不修处置）。credit K 计价/执行实参口径差（joint_config.py:
  511 已披露、保守方向）维持；run_online_strategy_sensing.sh 为
  sensing 演示旁路（无 F7 断言承载）注记在案；slo_tools 3 红
  （driver_parity×2 + slo_contract×1）HEAD 先在维持既有处置。

### 38.7 终局核验

- 三口径：canonical 全仓 815 passed + 1 skipped + 29 subtests
  （slo_tools 三 ignore 口径）；窄命令（workload 四目标）693 passed；
  slo_tools `unittest discover` 148 ran、3 failures（driver_parity×2
  + slo_contract×1，HEAD 先在同基线）。K 批新增
  online/test_joint_k_batch.py（31 用例）+ 既有改钉四处
  （terminal_turn / home_wait / read_prefix / quota_domain 列）。
- 矩阵：13 臂冒烟矩阵重跑（本批行为变更——证据根
  /home/sunhao/joint_smoke_evidence_k；TJE_quota_aimd 臂经 K7-④ 首次
  真信号闭环）。quota 臂端到端（P1-② 修复的 run 级实证）以矩阵
  quota 两臂退出码 0 + verify_run_end 守恒审计零 raise 为准。
- 裸仓/基线：git 零 commit（HEAD=9a95e06）；build/ 为本节矩阵运行
  的临时构件（README §4 流程，交付时 clean_build_artifacts 还原裸
  仓）；__pycache__/.pytest_cache/generated 清零；/tmp 过程文件
  （k38_append.md 等）交付前清理。

## 39. L 批（L1–L8）：K 批复核审计修复（2026-09-23，用户转交 kimi 复核报告并沿用"对的就改、不对的给依据"授权）

### 39.1 授权、偏差流程与矩阵处置

源 = 用户转交 kimi 对 K 批的复核报告（P2×4 + P3 择要 + 处置建议三
档），指令同 K 批："仔细核对分析每一条"。本轮全部逐条读码亲验（不信
审计结论）：13 臂决策日志逐臂统计 remote-read 选中数、AIMD 判据方向
读码、copy/restore 双侧 gap 分支对照、bak 残留全仓扫描（kimi 点名 1
个实为 5 个同族）。裁定：P2×4 全属实；P3 属实大半修；与审计不同处置
三项（EfficiencyFactors 值检查 A14' 已罩而键回退缺省 1.0 是冻结契约/
零字节腿忽略 startup_ns 属语义正确/bytes int() 截断尘埃级）落字给依
据——执行计划 §4.3 补遗 A16' 先落字后动代码（含被否决项）。

矩阵处置：本批对合法输入零行为变更（全部为不可达路径防御/入口
isfinite 加固/账本不变式破坏改 raise/脚本层新断言/纯文档），13 臂冒
烟矩阵不重跑（A14' 同款理由）；全量单测三口径复跑。

### 39.2 L1（P2-1）计数键 + L2（P2-2）NaN 同族三处 + 跨腿等长

- L1：`_joint_action_selection_counts` 初始化扩
  `recompute_forced_quota_deferred` 键（§38.3 K7-① 声称"计数键同源"
  而键没加——登记不实经本节勘误；动态 format 键在 quota 几何变更后
  必 KeyError，当前几何约束不可达属潜伏）。同步 4 处测试镜像
  （decision_schema :183 期望表 + :555 assertIn 硬枚举扩三值、
  quota_integration :221、fix1_pricing :273）+ sh30 :589 注释与
  _note_action_selection docstring 两处过时订正（"forced 按
  no_history/quota_deferred/evicted_permanent 三成因单列"）。
- L2（K8 同教义补齐）：① ResourceSnapshot.__post_init__ 增 now_ns
  isfinite（NaN 进 step() 流永不激活 `start<=NaN` 恒 False、
  `max(NaN,min(starts))` 保持 NaN ⇒ run() 死循环）；② committed 循环
  增 release_eta_ns 非 None 时 isfinite（NaN `eta>now` 恒 False ⇒ 在册
  流被当已过期、竞争份额静默忽略——K8 自称封死的乐观同族；负值合法
  =已过期语义）；③ first_block_wait_ns 构造器与
  predict_release_and_recall 合流入口先 isfinite+非负再 int()（int
  (NaN) 裸 ValueError 逃出 face_scheduler :5549 只 catch
  EventRecursionError 的降级通道）；④ predictor 构造器增跨腿 rank 数
  等长 fail-closed（P3 小项：短腿 bytes_by_rank[rank_index] 裸
  IndexError，生产腿恒同 TP 组不可达）。

### 39.3 L5（P3）copy 层段账本缺口 fail-closed + L6 双脚本 fail-closed

- L5：copy 层段 `layer_start > covered` 原静默插无门段（restore 侧同
  情形 raise，双侧不对称；账本不变式破坏时静默降级 = 缺口层不等到达
  即计算）。修 = covered>0 的缺口 raise（copy 块间与 copy↔restore 间
  两形态；covered==0 热前缀保持合法无门段——copy 块 0 走准入主链）；
  尾部 (covered,L) 零 KV 无门段补齐维持冻结设计（PARTIAL 基零后缀字
  节形态，docstring 明说）。生产不可达（账本同源恒连续），纯防御。
- L6：① run_online_strategy_sensing.sh 增 fail-closed
  （JOINT_QUOTA_MODE=aimd ⇒ exit 1 指引主路径——旁路无遥测注入链，
  注记只劝不拦）；② run_online_strategy.sh F7 断言块扩一条（aimd ∧
  SH_METRICS_DETAIL=off ⇒ exit 1——C++ 观测门压在
  MetricCollector::enabled（main_online.cc:1145）下，off 档即便
  observer=1 遥测仍恒空）。

### 39.4 L3（P2-3）§38.7 勘误 + L4（P2-4）r̂ 双窗披露

- **§38.7 勘误（本节为 append-only 纪律下的订正位）**：原文"quota 臂
  端到端（P1-② 修复的 run 级实证）以矩阵 quota 两臂退出码 0 + 守恒审
  计零 raise 为准"归因夸大——本批复核亲测证据根 joint_smoke_evidence_k
  全 13 臂决策日志 remote-read 选中数恒 0（各臂 stay×20+copy×1 等）、
  merge_done 事件零发生，quota 臂根本未进入 merge 通道；P1-② 的 run
  级闭合**未达成**，由终轮交叉单测 + 接线文本钉承载。同批补引 K 批
  stress 附加臂产物 /home/sunhao/joint_stress_quota_k（TJE_stress-28gib
  10s 容量压力 ×quota-static：partial_copy_hits=49、深缺口空、
  verify_run_end 守恒零 raise——原 §38.7/README 均漏引）。**登记未闭
  合项**：终轮闭合 run 级实证需"quota-on + remote-read 选中 + 终轮"
  组合，现有旋钮结构性造不出（stay 恒过配额、copy@home 恒自动准入、
  配额只会额外堵 remote-read）——C20 前需新增测试专用 action-pin 旋
  钮或 trace 手术（属 C20 决定性实验集授权面）。
- L4（披露级不改数值）：r̂ 双口径窗——① AIMD observe（:2932）/
  port_snapshot（:4097）裸调用无代表值，active_decode 瞬空而读流在册
  窗口回退 ctx=1 ⇒ 判据恒 comfort 累积扩张压力（K1 已把窗口从"永远"
  缩到"瞬空"；窗口=准入到入批间隙，实害有界）；② decode_tokens_
  consumed 仅列车核销推进 ⇒ 长列车在飞期间 r̂ 系统性低估（乐观方向，
  既有账粒结构）。已注记于 _quota_r_hat_kv_bytes_per_ns docstring。

### 39.5 L7 文档订正 + L8 卫生/补钉 + 未来约束登记

- L7：domain_metrics.py 头注与 slo_tools/README.md 的 forced 枚举补
  quota_deferred；experiment/仿真各功能开关清单.md §5.3"官方 runner
  总是显式传 0"订正（K7-④ 后 SH_LINK_TELEMETRY=1 ⇒ 缺省置 observer=1
  例外）+ L6 metrics 门注记；SH_LINK_TELEMETRY 行补记 K7-④ 缺省耦合
  与 L6 断言。
- L8：① bak 残留 5 个（slo_common.py/slo_params_manifest.json/
  README.md/slo_stats.py/test_slo_contract.py，均 .bak_caliberfix_
  20260905 后缀——09-05 校准批备份，断言为源文件旧版快照后删除；
  kimi 点名 1 个实为 5 个）；② kimi 代清 sh_test_mesh/generated/ 复
  核确认零残留（K 批"全清零"声明不实经 kimi 代清 + 本批复核闭环）；
  A15' 落点 = joint仓库改造执行计划.md §4.3（kimi 已核对内容齐全，
  落字确认）；③ 测试补钉两枚：ForwardLegContendedByBackgroundTest
  （K5-② 反向腿方向特异的对偶面——背景流确实抬前向腿）、
  test_quota_admissible_zero_branch_at_csv_level（全候选结构性不可行
  ⇒ CSV "0" 三态 0 分支）。**未来约束登记**：_telemetry_seq 当前"每
  epoch 单字典全链路"形态安全；任何未来采样子集/逐链轮替观测形态会使
  AIMD 扩张永久冻结（kimi 探针实证）⇒ 该形态变更前必须先改 per-link
  seq（强制前置约束，非本批改动）。

### 39.6 终局核验

- 三口径：canonical 全仓 834 passed + 1 skipped + 29 subtests
  （815 → 净增 19 = online/test_joint_l_batch.py 18 + slo_tools
  quota_admissible CSV 0 分支 1）；窄命令（pytest workload/
  llama2_7b_inference）713 passed + 9 subtests；slo_tools unittest
  discover 149 ran、3 failures（driver_parity×2 + slo_contract×1）
  HEAD 先在同基线。
- **窄命令登记勘误（§38.7/README K 批块）**：K 批登记"窄命令 693
  passed"实为 695（kimi 复核亲测同数：664 基线 + 31 新增精确吻合；
  本批排除法复核证实 695——K 批登记差 2 系笔误）；L 批后 = 695+18 =
  713 精确吻合。
- 测试反事实：L2 四类 NaN/等长拒收与 L5 两形态缺口 raise 在修前代码
  上必红（检查代码为本批新增）；L1 文本钉（__init__ 计数表三成因键）
  同理。
- 裸仓/基线：git 零 commit（HEAD=9a95e06）；__pycache__/.pytest_cache
  交付时清理；generated/ 零残留（kimi 代清后本批复核）；
  /tmp 过程文件（k39_append.md 等）交付前清理。

## 40. M 批（M1–M8）：chatgpt 验收级审查修复（2026-09-23，用户转交 C1–C19 验收视角 11 条并沿用逐条亲验授权）

### 40.1 授权、亲验方法与矩阵处置

源 = 用户转交 chatgpt 审查报告（11 条：高优 3/中优 4/证据缺口 4），
指令同前"仔细核对分析每一条"。亲验不限于读码：C++ FluidScheduler
观测器积分循环与 active_ns 头注语义（link 级"至少一条流时间"确证）、
stress 夹具 SH_RUNTIME_RC_DIR 机制与 joint_stress_quota_k 的
trace_config.backup.csv 在案事实、face ServiceFactorGroup 全仓调用方
扫描（零生产调用点确证）、README 新旧合同表述对照。裁定：①②③⑤⑥⑦
六条属实需修；④⑧⑨属实但已有登记处置（维持）；⑩属实但性质=证据未
认领（补登记）；⑪文档/卫生订正。**本批对合法输入有行为变更**（M1
遥测除数口径/AIMD 分母、M2 quota 臂决策日志保留配额前成本、M4 γ
因子离开冷启动、M5 零字节 shard 计价）⇒ C++ 全量重建（裸仓无 build/
按 README §4 拼装）+ 13 臂冒烟矩阵重跑（证据根
/home/sunhao/joint_smoke_evidence_m）。

### 40.2 M1（①+⑥）遥测物理口径统一——C++ 时间加权活跃流数

C++ `link_observer_totals().active_ns` 头注自证为 link 级"至少一条
活跃流的时间"（非 Σ流×时间）⇒ served_bytes/active_ns = 链路**合计**
吞吐（collective 含入）；JCM `capacity/rate` 按单流速率口径算
divisor_effective（test_joint_telemetry_divisor 注释自证注入假定）。
双症状实证：满载多流合计≈B ⇒ 除数≈1（注册表漏计的 collective 无法
补足）；流受下游瓶颈合计<B ⇒ 除数=B/合计 把"流慢"误判为"流多"（
单流 0.1B ⇒ 除数 10）。⑥AIMD 同根：per_flow=合计/在册流数，分子含
collective 分母不含 ⇒ KV 流+collective 均分 B 时输入 B ≥ 1.2·r̂ 恒
comfort（真实份额 B/2 应收缩——反向信号；sh30 :2923 docstring"保守
向披露"用词据此订正为乐观向）。

修 = C++ 报**时间加权活跃流数**（物理正解——合计吞吐无法反推流数）：
LinkObserverTotals/State 增 flow_active_ns；积分循环
`active_ns[link]+=dt` 处同步 `+=dt×link_states[link].active_flows.
size()`（流加入/离开是事件、segment 内流数恒定——直接读 link_states
免 membership 路径镜像）；main_online compute_link_telemetry 差分 +
sample 增 "active_flows"（窗口流数积分差分/active_ns 差分）。SH
ingest 解析落 _link_telemetry_flow_counts（缺席容忍=旧二进制兼容，
负值/非有限 fail-closed；与 rates 同生命周期三处失效镜像）；JCM
TelemetryLinkFlowView.effective_divisor/divisor_effective/_union_floor
三消费点流数在场取流数口径（缺席退 capacity/速率旧口径——既有单流
语义注入测试零改钉）；AIMD per_flow 分母 = max(在册流数, 遥测流数）。
对拍测试：3 流均分满载 ⇒ 除数 3（非 1）；单流受 cap 0.1B ⇒ 除数 1
（非 10）；KV+collective 均分 ⇒ shrink（非 comfort）。弃 SH 侧
rate/occupancy 折中（满载退注册数与现状同值、瓶颈误判不减——流数
口径唯一物理自洽）与 C++ per-flow 速率上报（逐流归因标识合同侵入
过大）；被否决 min(注册,流数)（丢注册多计侧保守性，C7 冻结 max）。

### 40.3 M2（②）配额前经济域重建 + M3（③）σ̂ 双终点分列

- M2 四子项：sh30 配额拒候选不再清 cost_ns（applicable=False 已挡
  feasible/argmin，决策行 candidates 自动携带配额前成本）；domain_
  metrics D_feed/D_econ/_min_cost 纳入条件扩 `applicable ∨ quota_前缀`
  （配额拒候选的经济域重建——quota-on run 不再被配额自身裁域）；
  实例 CSV 增 quota_admissible_remote 单列（四动作 any 列对 remote
  被配额裁不敏感的修复——stay 可用即 1 的反例由 remote 特异列分载，
  K7-⑥ 既有列语义不动）；摘要 quota_domain_status 从数据派生
  （derived_from_candidates/structural_only——固定 NA 与实例 CSV 0/1
  的自相矛盾订正，含打印块第三处 NA 残留）。测试：quota-on 夹具
  D_econ 重建 + remote 列三态 + status 派生。
- M3：pred=merge_done 终点（F11 冻结）vs measured=completion 行 tick
  （service_done 终点，C3b 新合同在 merge 前）混用 ⇒ 有 merge 流时
  系统性 −M 偏差污染 σ̂/δ 敏感性。修：DomainScan 增 kind="merge_done"
  行索引；measured_service 恒报（新 CSV 列）、merge_done 行在场时
  measured/err 用 merge 终点（同终点配对）、异终点不进 σ̂。
  测试：merge_done 夹具 err=+900（混终点修前 +300=−M 形态）+
  service 延迟 1000 单列。

### 40.4 M4（⑤）γ 生产桥接 + M5（⑦）零字节 shard 计价 + η 落字

- M4：face KVCacheManager.service_factors（递推预测器 η/γ）零生产
  调用方（mark_complete 恒 mark_unobservable、γ 恒冷启动 1.0——计划
  "同一 run 持续因果更新"未落地；chatgpt 与 JCM _joint_factors 的
  区分正确）。修：sh30 纯 prefill 列车核销分支（纯度约束已在——无
  joiner/无恢复门/无传输门）桥接 observe_valid_service（ratio=实际/
  roofline 闭式）。**η_pool 不桥接落字**：无可因果分离的纯池传输样本
  源——含争用样本 ratio>1 恒 ⇒ η 退化争用指标、与 UnifiedTimeline
  份额仲裁双计（A12' 对 JCM transfer 因子同款名实订正的镜像论证）；
  冷启动 1.0+mark_unobservable 是纯度约束的正确行为。被否决：JCM
  样本直接复喂 face 组（两套因子语义不同=口径混装）。
- M5：_transfer_ns_shards wall-time 循环零字节 shard 不计逐跳时延与
  流腿（与除数并集过滤同语义族；有数据一跳 10ns + 零字节十跳修前
  ≈100ns）；**startup_ns 保留计入**——F1 冻结契约"shard 在场即事务"
  （对照锚 test_present_shard_keeps_startup 重新对绿，chatgpt 指控
  本聚焦 hop latency 不含 startup）。

### 40.5 M6（⑩）C18 真冒烟补登记 + M7（⑪）文档订正 + M8（④⑧⑨）维持

- M6：C18"切换 trace_config + SH_RUNTIME_RC_DIR 至少冒烟一档"的真
  路径 = joint_capacity_stress_fixture.sh（:147 SH_RUNTIME_RC_DIR=
  RC_DIR + trace_config 行切换备份→切→还原）——K 批 stress 附加臂
  /home/sunhao/joint_stress_quota_k 即该路径真实后端运行（10s、exit
  0、judge_summary partial_copy_hits=49、trace_config.backup.csv 在
  案）。C18 验收证据=该产物（本节认领登记），不重跑。
- M7：README :172 括号注"service_done 位于 merge_done 之后"（C3b
  前旧合同残留）订正；:201"源端交接即释放"超实现订正为 drain 边界
  批量结算（P2-6 降级登记同步）；§6 命令下补注三 ignore 口径与
  slo discover 3 红基线关系；joint_capacity_stress_fixture.sh:242
  尾随空格清（git diff --check 归零）；PROVENANCE :4016 尾随空格
  append-only 不动已落盘字节、本节登记。
- M8 维持既有登记：④ C13 逐块释放（P2-6 降级：drain 边界=实现语义，
  逐块需图侧逐 chunk watch=机构件级，chatgpt 未提供新实现路径）；
  ⑧ C4b 三表（/tmp 事故无副本已登记，受控测量属 C20 授权面——21
  决策窄样本不能替代 G1/G2 完整验收=登记确认）；⑨ 13 臂终轮闭环
  （§39.4 已勘误+未闭合项登记，chatgpt 引证即该登记）。

### 40.6 终局核验

- 三口径：canonical 全仓 850 passed + 1 skipped + 29 subtests
  （834 → 净增 16 = online/test_joint_m_batch.py 14 + slo_tools
  domain_metrics 2）；窄命令（pytest workload/llama2_7b_inference）
  727 passed + 9 subtests（713+14）；slo_tools unittest discover
  151 ran、3 failures（driver_parity×2 + slo_contract×1）HEAD 先在
  同基线。改钉四处（quota_integration 配额拒成本保留 100=镜像 M2、
  review2 替身补 kv_manager、m 批替身补 _link_telemetry_flow_counts
  ×5 文件、domain_metrics status 派生）。
- M5 修复中途撞 F1 冻结契约锚（零字节 shard startup 计入）——改道
  精准解（免 hop/stream 腿、保 startup），对照锚零改钉重新对绿：
  冻结契约优先于审计建议的教训入档。
- 矩阵：13 臂重跑全 exit=0（证据根 /home/sunhao/joint_smoke_evidence_m，
  C++ 含 M1 观测器流数积分的全量重建——裸仓无 build/ 按 README §4
  拼装；verify_run_end 守恒零 raise）。aimd 臂遥测覆盖逐位一致
  （epoch_count=1392/sample_count=32898/rate_entries=14，与 K 批证据
  根同值——确定性输入下 M1 ingest 扩展零回归）；AIMD 动作计数与
  copy 选中成本 K/M 逐位一致（comfort_cold_start×3 / 2700471133）——
  2s 轻载窗内遥测流数恒等于注册流数（无 collective 竞争时刻），两
  口径等值不分叉属预期。**M1 流数口径的 run 级分叉证据受窗口限制
  （与 K 批终轮闭合同款如实登记）：分叉形态（3 流均分除数 3 非 1/
  单流受 cap 除数 1 非 10/AIMD 混链 shrink 非 comfort）由
  test_joint_m_batch.py 对拍单测承载**。
- 裸仓/基线：git 零 commit（HEAD=9a95e06）；build/generated/
  __pycache__/.pytest_cache 交付时清理（clean_build_artifacts 流程）；
  /tmp 过程文件交付前清理。


## 41. N 批（N1–N13）：chatgpt 对 M 批交付的复核修复（2026-09-23，用户转交 6 生产问题 + 2 证据认领 + 4 回归覆盖并沿用逐条亲验授权）

### 41.1 授权、亲验方法与矩阵处置

源与授权同前批（"对的就改、不对的给依据"）。本轮证据实测不限于读码：quota 两臂 decision_log 逐候选统计（remote-read 189 候选 0 适用；quota_link 拒 128、结构不可行 61 = 189−128 精确吻合 chatgpt 数字）；拒因样本逐条提取（`remaining=2 of Q=2, need=3, occupancy=0, reserved=0` —— 空链路 need=3 铁证，读流 2 shard + merge 预留叠加）；stress 臂 2430 候选 1928 拒复核吻合；stress JSON d2d=4050 实测（C18 认领错硬件坐实）；三孪生 JSON 在仓事实；执行计划 v2.2"归 C20 的指认落空属实、新增 C4b"原文复核（M 批把 C4b 归 C20 授权面系重复 v2.2 已否定的归因）。裁定：主项 1–6 全属实修；证据项 7（C4b 归属勘误登记不重做——强制动作机制全仓 grep 零命中，重做须专门一轮立项呈报）与 8（C18 补孪生档真冒烟）；回归组 M3 缺行/M5 测试钉不住/C++ 流数断言缺/M4 未走生产路径四项属实全修；C13/η_pool 维持既有登记（chatgpt 亦认可）。**有合法输入行为变更（N1 配额准入域/N2 除数/N4 prefill 基数/N5 样本纯度/N6 工具输出）⇒ 13 臂矩阵重跑**（证据根 joint_smoke_evidence_n）。落字：执行计划 §4.3 A18'（处置、被否决与不修依据全量）。

### 41.2 N1（生产1）配额同事务槽位复用——空链路 TP=2 需 3 槽结构性排除远读

判据端 `_quota_candidate_verdict` 把读流逐 shard demand 与 merge 双向预留 demand **相加**（TP=2 共链 ⇒ 2+1=3），入册端 admit_flow（occupancy）与 reserve_merge（reserved）两笔独立记账同链叠加 ⇒ 空链路 remaining=Q=2 < need=3 恒拒；ρ<1（Q_init=1）更恒拒——违反详细计划 C9 WP3a "ρ<1 配置的域空化判据 = 定价选中率趋 0（argmin 涌现），不是配额关闭"。物理正解：读流与 merge bulk 写是**同一事务的时序先后阶段**（读流 settle → 计算 → service_done 裁决 → merge 写），同事务同链峰值 = max(读阶段, 写阶段) 而非 sum；跨事务叠加不变（他事务预留与本事务读流可真并发）。修（判据 + tracker 同构双修，保"杜绝判据过而入册拒"防御一致性）：① 判据端 per-link demand = max(read_demand, merge_demand)；② tracker `reserve_merge` 增同事务借槽（`_merge_borrow` 账本：rid 名下流已占链路的预留不重复入 reserved）；③ `release_flow`（读流 settle）借槽**转移**（occ→res，总计数不变——满借场景无新准入窗口；需求 1 借 1 的部分转移后余量归还，同样物理正确）；④ `_release_merge_side` 预留先撤时借记就地消解；⑤ snapshot 增 merge_borrowed 披露。被否决：判据端单修（重引"判据过而入册拒"成为常规路径）；预留延至 service_done 才占槽（改变 C9 冻结语义且裁决时刻链满 ⇒ merge 排队语义未定义）。验收：online/test_joint_n_batch.py 组 1 五测（空链 Q=2 TP=2 准入+跨事务仍拒 / settle 转移守恒 / 预留先撤消解 / ρ=1 单 TP 准入恢复 / 判据-入册同构文本钉）。

### 41.3 N2+N3（生产2+3）遥测除数叠加候选 + 仅流数遥测保留

N2：view 层合并式 max(注册含候选, 遥测旧流) 漏"候选 + 未登记旧流"叠加——遥测不含候选（决策前旧流）、注册不含未登记流（collective），两未登记旧流 + 一候选时实际并发 3 模型只算 2。修：divisor_effective/divisor/divisor_multi 三方法逐链路改 **max(注册旧流, 遥测旧流) + 候选份额（include_self 时恒叠加）**——C7 冻结的 max 防漏计保守性保留（注册/遥测两口径取大者为旧流并发估计）；registry 增公共 registered_flows 访问器。窗口均值 ≠ 决策瞬时：落字披露（C++ 周期采样无决策时刻事件，改瞬时 = 合同侵入）。既有五处旧口径钉值测试改钉（test_joint_telemetry_divisor：include_self_registered_leg 4.5/6.5、union_floor_bottleneck 6.0、single_path_merges 8.0、fractional_ceil 4/1400、lifts_remote_pricing 3/三倍——逐个新值物理复核后钉）。N3：C++ 整字节进位（carry→uint64）合法短窗 served=0 ∧ active>0 ∧ 流在场；SH 零速率窗把流数与速率一并 pop（丢 collective 在场唯一证据）。修：零速率窗只 pop 速率保留流数（JCM 已支持仅流数分支）；A9' 剪除循环改遍历 rates ∪ flow_counts 键并集（修后纯流数条目不漏剪陈旧驻留）；JCM disclosure 增 flow_count_links / legacy_rate_divisor_links（旧二进制退 capacity/合计口径的降级可辨认）。验收：n_batch 组 2/3 七测。

### 41.4 N4（生产4）prefill 基数生产同形 + N5（生产5）merge 尾门纯度排除

N4：生产列车核销逐 chunk 按 min(p_chunk, 剩余) 切分、累计 context（span 基 + 前序 + 本 chunk）求 roofline 求和；预测三处（SH 构造 context=1/chunk=1 单 token 率、JCM ×token 线性外推、face E 递推同款）全丢形状。**chatgpt 复算例（1800 vs 4464）未复现**：按生产 JSON 参数（4050/1640/261TF）复算 chunk=10/ctx=100 得 330 vs 线性 3100——方向与其例相反（口径未披露不可复核），但形状差异实质确凿（实测比值 0.13–0.28 即线性外推系统性偏差 3.5–7×），修复不受影响（A18'(d) 如实登记）。修：JCM 增可选 `prefill_task_load_ns_fn`（(input_tokens, history_tokens)→ns；None 退旧线性口径——既有测试零改）；SH 注入 `_joint_prefill_total_load_ns`（p_chunk 切分 + 累计 context 逐 chunk roofline 求和，与 _plan_train/:963 核销同值同 memo）；recompute 调用点 history 恒 0（R13 副本 0 基）；face E 递推 `_adaptive_retention_target` 同形替换（_prefill_chunk_load_ns memo）。N5：R11(ii) 门（graph_batch_builder 下一轮 interval gate 锚上一轮 merge_done 标记）hold 本列车物理计算 ⇒ span 混入门等待且样本初始化 γ（双因子通道污染）。修（排除法）：train 增 `merge_tail_gated`（发射时同 session 的 `_pending_merge_alarms` 有未交付 watch），纯 prefill 纯度门排除；扣除法被弃（需复刻 merge_done + interval µs 对齐 + hbm_wait 三层合成门语义，SH 侧观测面不完备，强复刻引入新形状误差——A18'(e)）。验收：n_batch 组 4/5 五测（同形对拍逐位相等 / 形状差异实证 / JCM fn 消费与 recompute 0 基 / 无门列车双通道入样（兼 N12 生产路径）/ 门控列车排除）。

### 41.5 N6+N9 工具双域导出与缺行告警；N7+N8 证据认领勘误；N10–N13 回归组

N6：① quota_admissible_remote 语义消歧为**配额前结构域**（docstring/模块 docstring/README；列名冻结不动——实际域已有 remote_applicable 列承载，K7/M2 测试钉不破坏）；② instances CSV 增 `remote_cost_pre_quota_ns`（配额拒候选保留成本——quota run 实例层配额前成本等值线可画，修前 :739 applicable-only 置 NA）；③ replay_mismatch 重放集改**实际可行集**（applicable-only remote_min_actual/alt_star_actual——纯谓词一致性），新增 `quota_counterfactual_flips`（配额前集重放与在线之差 = 配额作用量 run 级，计划 :574 "两域之差"落地）——修前被拒远读更便宜时配额效应混入 replay_mismatch；④ D_feed definition 文本订正为现实现谓词。N9（M3 缺行分支）：completion 行 merge_transferred_bytes>0 ∧ merge_done 行缺失 ⇒ measured_ns 置 NA + warnings 告警（修前静默退 service 终点，−M 系统偏差不报）；derive_entries 增 warnings 透传（两遍调用查重去重）；基础夹具 R4 补 merge_done 行（同刻 2000，断言语义保持）。N7（C4b 归属勘误）：M 批 §40.5 M8 把 C4b 三表登记为"C20 决定性实验集授权面"与执行计划 v2.2"归 C20 的指认落空属实、新增 C4b（G2 门槛）"直接冲突——本节勘误：C4b 状态 = **G2 未闭合**（三表随 §21 /tmp 事故丢失无副本），重做属 C4b 卡自身义务须专门一轮（强制动作机制不在仓：全仓 grep force_action 零命中，夹具须新建），不借 C20 名目移转验收归属；C20 前置表述仅指 action-pin 类共享工具。N8（C18 认领错硬件）：§40.5 M6 认领的 stress 臂固定 stress JSON（d2d=4050 实测）——非 C18 三孪生档位（2025/8100/1200）。补真冒烟：fixture 增 `SH_STRESS_JSON` 覆盖（缺省 stress 语义零漂移；RC_DIR 从 JSON 基名派生）+ `hardware/face_case5_config_c_d2d_x05.json`（2025 GB/s）2s 真冒烟 GREEN（证据 /home/sunhao/joint_c18_twin_x05_evidence：runtime_config 自 x05 物化确认、trace_config 行切换 + SH_RUNTIME_RC_DIR 路径、judge PASS、exit 0、trace_config 已还原）——C18 验收条款（计划 :605 "冒烟一档过"）自此以孪生档位闭合。N10：M5 测试数据 shard 改 1B（修前 610ns 主导新旧同值钉不住；修后旧实现 100 / 新实现 10 区分度恢复）。N11：C++ 观察器测试增流数积分断言（链 0 双流重叠 ⇒ flow_active_ns > active_ns 且时间加权均值 ∈(1,2)；链 1 恒单流 ⇒ 相等且均值 = 1）；**首次在本仓构建运行该上游继承测试暴露 record 级 streaming/legacy 等价破裂**（legacy 手排分段点 11/19 与 scheduler 事件钟分段不符；总量守恒 14/14、10/10，分桶进位不同；注释掉 M1 积分块复跑同红 ⇒ 与 M1 无关、系上游环境继承差异且本仓从未运行过——PROVENANCE 此前零运行记录）——处置：streaming 金值直钉（本仓事件钟实测）+ legacy 总量守恒断言 + active time/window 金值（20/20、20）+ 上述注释如实落测试；上游手排等价不可恢复的原因登记在案。N12：M4 测试补走 `_observe_service_factors` 生产核销路径（__new__ 替身，与 N5 同载体）。N13：slo_tools/README "port_snapshot/flow_snapshot 仍为 C5 冻结 NA 占位"过时订正（quota-on 臂实测值 C11 已接通）；domain_metrics 模块 docstring 同款；D_feed 订正并入 N6④。

### 41.6 矩阵重跑结果与远读准入域实证

13 臂全 exit=0（证据根 /home/sunhao/joint_smoke_evidence_n，含 TJE_quota_static/aimd/face_static 三 C19 扩臂 + 影子验证臂）。**N1 生效 run 级实证**：quota 两臂 remote-read 189 候选 quota_link 拒 128 → 96、applicable 0 → 32（空链结构性排除解除——修前 remaining=2 of Q=2, need=3, occupancy=0, reserved=0 恒拒形态消失）；chosen_remote 仍 0 = 配额不再关闭准入域、选择由定价涌现（stay/copy/recompute 胜出）——正是计划 C9"ρ<1 域空化判据 = 定价选中率趋 0（argmin 涌现），不是配额关闭"要求的形态。剩余 96 拒因样本复核为 max 口径下真实路径形态（如 link=(14,20) need=3 = 3 条 shard 读流共链并发 > Q=2——生产拓扑 TP shard 路径共链，非旧"读 2 + 预留 1 叠加"伪 need）。**决策分布行为变更（N1+N4 联动，预期内）**：quota 臂 stay 20+copy 1 → stay 5+copy 1+recompute 15（N4 chunk 化基数下 recompute 0 基相对更优）；combo_none = stay 5+copy 5+recompute 11；全臂 exit=0 无 raise。aimd 臂遥测覆盖与 M 批逐位一致（telemetry_epoch_count=1392、sample_count=32898、rate_entries=14、collective_coverage=true）——零回归。chosen_remote=0 / merge_done=0 维持：2s 窗定价未选远读，远读选中与 merge 通道的 run 级实证仍受窗口限制（C20 action-pin 前置维持 §39.4 登记；N1 打开的准入域使该通道在更长窗口/负载下可达——32 个 applicable 候选在案）。

### 41.7 测试计数与勘误登记

canonical（README §6 命令）= 868 passed + 1 skipped + 29 subtests（850 → 净增 18 = online/test_joint_n_batch.py 16 + domain_metrics 2）；窄命令（workload/llama2_7b_inference 全目录）= 743（727 → 净增 16 = n_batch）；slo_tools discover = 153 ran（151+2），红签名逐名同基线：test_driver_parity×2 + test_slo_contract×1（红签名逐名同基线：driver_parity×2 + slo_contract×1）。勘误：§40.5 M6 的 C18 认领（本节 41.5 N8）与 M8 的 C4b 归属（41.5 N7）两处由本节订正；A17'(d) M4 的"纯度约束已在"表述由 N5 增补（上一轮 merge 尾门纯度缺口本轮补齐）。M 批 §40.6 的"2s 轻载窗流数恒等于注册数"结论不受 N2 影响（N2 修的是候选叠加侧，遥测侧窗口等值性维持）。

## 42. O 批（O1–O14）：kimi 终轮深挖（13 路审计）处置收口（2026-09-23，用户授权"派子agent改，你自己负责思考规划调度"）

### 42.1 授权、执行方式与总览

源：用户转交 kimi 终轮深挖报告（13 路只读审计 + 探针亲验：P1×6 + P2×27 + P3 择要；审计方法与探针清单见执行计划 §4.3 A19' 行）。执行方式：主控负责规划/裁决/集成核验，五路执行代理分波落地——波次 1 四路并行（QUOTA=link_quota / JCM=joint_cost_model / SLO=domain_metrics+slo_common / RD=runner+文档），波次 2 SH 路十项（sh30_online_scheduler + graph_batch_builder）；主控亲办 O4/O5 与 face 侧 N9 空串三件。落字：执行计划 §4.3 A19'（处置/不修/被否决全量）；A19'(b) 与被否决项的**实施订正**（O2 实际走"不可知"分支，见 §42.4-6）。**本批对合法输入有行为变更**（O1 计价 / O2 配额入册序 / O3 纯度门 / O6 AIMD 断供与回退窗 / O7 遥测 ingest / O12 口径与计数）⇒ C++ 全量重建 + 13 臂矩阵重跑（§42.5；本批 C++ 零改动，重建为拼装复原）。

### 42.2 P1 六件全修

**O1（r̂ 之外的计价基数——recompute @驻留基 history 恒 0）**：JCM 增 `_resident_here` 静态助手（joint_cost_model.py :1958）——recompute 的 history 按驻留二分：驻留 ⇒ session.history_tokens，异地/REMOTE ⇒ 0（joint_cost_model.py :1663）；信息在 JCM 作用域内自足，SH 零调用点改动、既有钉测全未破。SH/JCM 两侧"recompute 传 history=0"的过时 docstring 同步订正（sh30 :5001 区、online/test_joint_n_batch.py 头注）。
**O2（配额判据不含本请求逐出足迹 ⇒ 入册被自身 #evict 击穿整 run abort）**：落地读码判定**判据时刻逐出足迹不可知**（history_evictions 由准入事务产出、事务成功才落账 :4449、判据在事务前 :2705 执行且对全候选×全实例评估——前瞻 = 整套 KV 账本的投机性复刻），走任务单预留的不可知分支：`_quota_enroll_admission`（sh30 :2794）入册序不变量——主流程恒先于逐出支链（单线程内判据与入册间 tracker 零变更 + 主流程需求判据已前瞻 ⇒ 主流程失败只剩真异常，保持 raise→回滚→False）；支链失败 = 判据不可前瞻的固有残差，按 quota_oneshot_overflow 披露**降级**不 abort（与 decode 相 oneshot 溢出同姿态）；:4505 区"正常不可达"注释、防御分支决策日志标注、`_rollback_admission_registrations` docstring 按事故链写实订正。
**O3（跨 session 完成批尾段混入下一纯列车 span——纯度门系统性缺口）**：`merge_tail_gated`（sh30 :1269）从"同 session 未交付 merge watch"扩为**本实例粒度**两源并集：① 本实例任意 session 未交付 merge watch（R11(ii) interval gate 按实例 frontier 锚定而非 session——alarm 登记增 `instance_index = runtime.decode_instance_index` 字段，:1903）；② 本实例 frontier 未交付完成批尾段（`graph.pending_store_tails` 任一条目 edge_rank ∈ 本实例 ranks，topology 直查不依赖替身）；decode 纯度分支（`_observe_service_factors` member_parts 分支 :3730）补同款排除。测试全走 `_plan_train`→`_emit_train` 真 graph 生产路径。
**O4（adaptive 逐出 victim 急切求值 0.6s/次墙钟）**：`VictimView` 增可选 `retention_target_layers_fn` thunk（layer_eviction_policy.py :354），`_adaptive_soft_target` 惰性求解 + plan 内按 session 缓存（两轮扫描共享）；face :3673 调用侧传 `retention_target_layers=0` 占位 + thunk。
**O5（kv_delta_find 反扫 O(n²)）**：face_scheduler 增 `_kv_delta_index`（trigger_request_id→行 dict，`_append_kv_delta_row` 唯一入口维护 = 最新命中语义不变），`kv_delta_find` 改 O(1)。
**O6（AIMD 两件）**：① 断供门——`_ingest_link_telemetry` 缺席早退空调用 `_quota_ingest_telemetry`（sh30 :3523），且后者删空 `per_flow` 早退、恒调 `observe_telemetry`（空字典 = 仅序号/遥测钟簿记）——断供期 dt 不计入缺测后首样本；② r̂ 回退窗冻结扩张——`_quota_ingest_telemetry` 传 `allow_expansion=has_active_decode`（sh30 :3075，与 `_quota_r_hat_kv_bytes_per_ns` 内部回退分支同条件同源）；link_quota 侧 `observe_telemetry(..., *, allow_expansion: bool = True)`（keyword-only 非 bool fail-closed）+ 新 action `comfort_frozen`（"不计进度"语义：冻结期 quiet 不累计、解冻后重新累计 T_expand——防解冻瀑布）。

### 42.3 P2 组处置

**O7（遥测 ingest 四件）**：① active_flows 同 epoch 混合缺席 = 不可能出自同一版本二进制 ⇒ fail-closed raise，全缺席才按旧二进制容忍（sh30 :3590 区）；② 双零样本（served=0∧active=0）**带 active_flows 字段**时速率+流数同步 pop（:3640——修陈旧速率钉驻 + max(1.0,0)=1 幽灵流）；字段缺席（旧二进制）双零维持 A9' "包内样本为链路在场作保"语义（既有钉测 test_zero_active_sample_keeps_link_present 未动，陈旧速率存活 ≤1 epoch 有界——收窄依据落字 :3641 区注释）；③ `_union_floor` 混合形态仅流数链路（N3 已修，本批 disclosure 侧收口）；④ `_telemetry_coverage_decision` 增 `telemetry_degradation` 三键块（sh30 :5549：legacy_rate_divisor_entries / flow_count_only_entries / zero_rate_samples_dropped，全派生自现存状态零新增账本）+ ingest docstring "窗口均值≠决策瞬时" 口径警告（A18'(b) 登记补履行）。
**O8（domain_metrics 五件）**：① `_selection_under_delta` 跨族平局改在线全键序 `(cost+δ, instance, ACTION_ORDER 秩)`（改钉一处见 §42.4-1）；② build_summary 聚合 hops≥0 过滤（-1 哨兵不进分位/方向分布）；③ `_sigma_tier_available`——σ̂=None 时 instances CSV in_d_econ_* 报 NA 非 0；④ quota_domain_note 过时表述订正 + slo_tools README 同款；⑤ `_quantiles_or_na`→`nearest_rank_percentile_many` 单排序（3000 组随机对拍逐值一致）。
**O9（slo_common）**：iter_jsonl 增 `parse_constant` 拒 NaN/Infinity 字面量 fail-closed + fmt_ratio isfinite 守卫 + 单测。
**O10（守恒加固四件）**：① `verify_run_end` 增 `_assert_quota_tracker_ledgers_clean`（sh30 :3416——snapshot 的 occ/res/enroll/bulk/merge_borrowed 全零，quota=off 条件化跳过）；② link_quota `_assert_borrow_within_occupancy`（borrow ≤ rid 名下 occ 的 O(1) 增量校验，创建/转移/sweep 三落点）；③ `_PoolPortRegistry.leaked_owners`（sh30 :201，镜像 LinkFlowRegistry 语义）+ `_assert_no_flow_registry_leaks`（:3447——链路/池端口/HBM 三注册表 run 尾全 owner 断言）；④ GB `_credit_arms` 三侧 fail-closed（登记侧前序残留 raise :1320 区——检查在块循环**之前**；列车尾标记点对"发射过尾块却无账本"raise :1270 区——单块切片合法缺席不误报；`emit_completion_batch` 完成残留 raise :1660 区，与 copy/restore arms 同款纪律）。
**O11（runner 运行期守卫）**：joint_runner.py `_resolve_metrics_detail()`——aimd ∧ SH_METRICS_DETAIL=off ⇒ exit 1（无逃生；L6 shell 断言的运行期双保险，/tmp 实测拦截/放行两向正确）。
**O12（口径两件）**：① `_classify_physical_feasibility`（sh30 :4959）remote-read 适用面与 N1(a) 解除同步——PARTAL 基（partial_hbm_remote）参与判定（与 JCM action_applicability :1410-1416 逐条件同构，消融开关 JOINT_REMOTE_READ_PARTIAL=off 时照旧排除，REMOTE 基仍拒）；修前"仅 LOCAL 基参与"是 N1(a) 前旧口径残留，PARTIAL 基上 remote-read 唯一可行时误报 structural_infeasible（改钉见 §42.4-2）；② quota_oneshot_overflow（decode 相 + O2 准入支链两发射点）计入 run 级配额指标 `joint_decision_metrics.quota_events.oneshot_overflows`（sh30 :632/:2909/:2945）。
**O13（3 红夹具清零）**：test_driver_parity×2 + test_slo_contract×1 夹具 train_ledger 行补 exits 数组（2026-09-04 口径修正漏改夹具的 HEAD 先在红项——slo discover 自此全绿；工具侧不加 drains 兜底 = 不削弱 schema fail-closed）。
**O14（P3 择要）**：N9 豁免 `base_location or REMOTE` 空串借道改显式 fail-closed（face_scheduler.py :4936/:4967 与 sh30 :3816 镜像——`""` raise RuntimeError，None 才落 REMOTE_MEMORY；全仓 grep 确认无第三处）；link_quota `_release_merge_side` desync raise 补注入测试（pragma 移除）；文档计数订正（README 19→34、26→29 等）与开关清单/run_online_strategy.sh 行号订正；test_joint_telemetry_divisor 文件头覆盖清单同步 N2/M1/N3 + "窗口均值≠决策瞬时" docstring；N11 C++ 测试注释精确化（断金零改）。

### 42.4 改钉与裁决记录（逐条推导）

1. **domain_metrics δ=10 跨族平局改钉**：K7-⑤ 只镜像个族内序；跨族平局修前按枚举序（remote-read 先于 stay），在线全键序 (cost+δ, instance, ACTION_ORDER 秩) 下 δ=10 等号情形的期望从 ("remote-read",1) 改钉为 ("stay",0)——新值与在线 argmin 逐键推导一致（推导过程落测试注释）。
2. **test_remote_on_partial_base_excludes_remote_read 反转改钉**（joint/test_joint_review2_fixes.py :365）：原断言"PARTIAL 基即使 remote on 也不参与判定"钉的是 N1(a) 解除（2026-09-17）之前旧语义；解除后 JCM 与分类器都把 PARTIAL 基 remote-read 列为合法适用动作，M1 自己的设计前提"按适用动作集合过滤"不变、变的是适用集合。处置 = 改钉不删除：夹具补 `remote_read_partial_enabled=True`（F6 直读无软门），断言反转为 `assertFalse(structural)` + detail 含 remote-read（本场景 remote-read input-only 足迹可行 ⇒ 不再结构性不可行）；docstring 写明改钉依据与日期。
3. **test_committed_eviction_split_finality 夹具补字段**（online/test_joint_layer_restore.py :660 区）：O4 给 VictimView 增字段后 SimpleNamespace 替身吃不到 dataclass 默认值（AttributeError）——本批自伤基线红，补 `retention_target_layers_fn=None`（静态值路径，原测试语义不变）。
4. **m_batch fake_observe 签名**：keyword-only `allow_expansion` 兼容（任务单授权内；真实行为断言未动）。
5. **O7② 收窄**：见 §42.3 O7②——字段缺席形态维持 A9' 语义系既有钉测 + 物理（旧二进制无流数通道）双重约束，非放松。
6. **A19'(b)/被否决项实施订正**：原登记选"判据端并入足迹"并否决"入册序分派"；落地读码证明判据时刻足迹不可知（§42.2 O2），最终采用的恰是被否决项的入册序分派（主流程失败仍 raise、仅支链降级，逐出 oneshot 记账时点语义未变）——执行计划 §4.3 A19' 行内已同日订正两处文本。

### 42.5 矩阵重跑结果

C++ 全量重建（README §4 拼装，-j 6 限并行——满载 `-j` 曾致 29Gi WSL 内存耗尽会话中断，本批起构建限并行入操作纪律）；二进制 sha256 与 N 批 invocation.json 记录**逐位一致**（O 批 C++ 零改动实证）。13 臂全 exit=0（证据根 `/home/sunhao/joint_smoke_evidence_o`，输入 requests.csv md5 与 N 批逐位一致）；10s quota-static 容量压力附加臂 PASS（run/judge 双 exit=0，partial_copy_hits=48，K 批 49——少 1 与决策偏移同向）。**行为变更面（预期内，O1 计价基订正的 run 级实证）**：recompute 选中全域清零——非 J 族 N {stay5, copy5, recompute11} → O {stay16, copy5}；J 族 N {stay5, copy1, recompute15} → O {stay20, copy1}；@驻留基 recompute 不再被 0 基误压价后，stay 全面胜出（merge/copy/journal/quota 计数逐臂不变）。**O 批新接线 run 级实证**：aimd 臂 invocation.json 含 `"metrics_detail":"full"`（O11 新门——N 批无此字段）；`quota_events.oneshot_overflows` 键全 13 臂在案（值 0——本工作负载结构性无 oneshot 超额，非零形态由 o_batch 单测承载）；verify_run_end 零账断言 13 臂零触发；`comfort_frozen` 未出现（回退窗需遥测断供/空载才触发，本窗无此形态，冻结语义由 QUOTA/o_batch 单测承载）；守恒审计零 raise（deep_gap/merge_degrade/ledger_export_errors 全空）。**chosen_remote 仍全 0**：applicable 候选 128 在案（与 N 批同）、不满足原因 {base resident at target:16, no resident remote history:45}——远读选中与 merge 终轮的 run 级实证维持 §39.4/§41.6 登记（受 2s 窗动作组合限制，C20 action-pin 前置不变）。

### 42.6 测试计数与勘误登记

窄命令（workload/llama2_7b_inference 全目录）= **807 passed + 9 subtests 全绿**（N 批终态 743+1 红 → 本批 763 基线含 1 红（O4 自伤，§42.4-3）+ 43 新用例（online/test_joint_o_batch.py）+ 红修复 = 807 零红）；canonical（README §6 命令）= **936 passed + 1 skipped + 29 subtests**（868 → 净增 68 = O 批各件合计）；slo_tools discover = **160 ran OK**（O13 清零 HEAD 先在 3 红——discover 首次全绿）。勘误：§41.7"slo_tools 3 红同基线"的既有红项陈述由 O13 清零（历史陈述定格不追改）；§41.7 重复括号与"canoncial"拼写两处按 A19'(n) 登记（append-only 不改原文）。**登记未决后续项**（agent 移交，本批不扩面）：tests/run_golden_live.py :170-194 仍读 `drains` 键、read_cpp_metric_records 仍裸 json.loads（未接 O9 的 parse_constant 拒 NaN）——两件为工具侧健壮性跟进，下一批处置。不修/降级项与 C4b/C18/C20 实验面登记维持 A19' 与 §41.5 N7/N8 不变。

## 43. 逐出尾 watch 修复验证轮（watchfix，2026-09-24，用户授权按交接文档执行）

### 43.1 修复回顾（交接轮交付，本收口轮零修改、只读只跑）

**旁支流/配额改由 eviction_done 尾 watch 释放**：四条逐出旁支路径（admission history、admission prefill、decode joiner、失败准入、decode 增长——admission 侧两条共用一 watch 的半边交付语义）的流登记与配额入册不再由主链 `_release_transfer_flows`/`_quota_release_admission_phase` 顺带释放，改在尾 watch 交付时按 `watch_id#flow`/`watch_id#quota` owner 精确释放。图侧仅对旁支发射原语返回非空 members 才追加 watch：graph_batch_builder.py:2189-2212（history_evictions）、:2352-2369（prefill_evictions）、:2003-2056 `emit_eviction_side_branch`（缺省 watch_id :2021-2023、members 空返 None :2049-2050）；调度器发射咽喉点为 `_emit_admission`（sh30_online_scheduler.py:5323）与 `_emit_eviction_only_nodes`（:5134，空集早退 :5152-5153、watch None 不登记流/配额）。owner 族核对：旁支 owner 恒 `batch_train_evict_*` 前缀 watch_id 派生，与主链族（rid/rid#decode/rid#merge/rid#readplan/rid#evict/rid#evict#prefill）互不相交；LinkFlowRegistry.release_owner（joint_cost_model.py:465）与 HbmPortFlowRegistry.release_owner（hbm_port_flow_registry.py:90）对未知 owner 幂等返 0——回滚先放 + 尾 watch 后放的配对安全。

**测试替身对齐**：钉测全链走真实 GraphBatchBuilder/`_emit_admission`/`_emit_eviction_only_nodes`/`_register_train_eviction_watches` 生产原语；唯一镜像面为 decode joiner drain 半边——sh30_online_scheduler.py:1685-1703 系 `_on_prefill_drain` 内联块、非独立方法不可单独调用，钉测以同款真实原语镜像组装（`_next_eviction_watch_id`/`_register_transfer_flows`/`_quota_enroll_eviction_branch` + pending dict 逐字段同形，末行含 :1703 逐字生产语句，测试文件模块头如实披露），另由 `DrainRegistrationSlotsFixPinTest` 在真实 `_OnlineRequestRuntime` 上逐字钉住——登记的限制而非缺陷。

**sensing 入口消费 SH_RUNTIME_RC_DIR**：run_online_strategy_sensing.sh:74 读该环境变量（缺省回退同目录）并在 :211-214 将 runtime_config 四件装配进 C++ argv——R16 覆盖语义对 sensing runner 同样生效，run 级证据见 §43.6。

### 43.2 本轮追加修复登记：decode_eviction_watch_id 潜伏 AttributeError 加固

`_OnlineRequestRuntime.__slots__` 末尾补声明 `decode_eviction_watch_id`（sh30_online_scheduler.py:333-334）+ `__init__` 紧随 `decode_evictions=()` 无条件置 None（:430-433）——修复 :1685-1703 drain 登记块（门 :1685-1687、pending 注册 :1696-1702、`runtime.decode_eviction_watch_id = watch_id` 赋值 :1703）在 slot 缺失时的潜伏 AttributeError。**结构性不可达刻画（如实登记）**：decode_evictions 三来源现行恒空——`decode_growth_evictions` 字面 `()`（sh30_online_scheduler.py:1681）、`move_request_capacity_reservation` 同实例恒 `()`（face_scheduler.py:2377-2378；未知预约 fail-closed raise :2374-2375）、`move_prefill_to_decode` 同实例返 `(local_hit, ())` 恒空逐出（face_scheduler.py:4628-4637）——**decode joiner watch 链路现行 run 不 engage**，本修复属潜伏缺陷加固；一旦任一来源产生 decode_evictions 即按钉测形态运转。全文件恰 5 处生产引用无遗漏（:334/:433/:1242-1244 消费面 getattr 容缺省、对 `__new__` 替身兼容/:1703/builder graph_batch_builder.py:776 get 消费）；转正链 `_register_train_eviction_watches`（:2581 区起）对无 drain 登记 fail-closed raise，生产调用点 :1359/:1490。

### 43.3 四路径钉测（新测试文件 15 用例全绿）

新文件 `workload/llama2_7b_inference/online/test_eviction_tail_watch_real_paths.py`（15 用例 5 类，零后端）：AdmissionEvictionWatchRealPathTest（真实 GraphBatchBuilder+真实 `_emit_admission` 全链：主链释放后旁支仍在三注册表、两 watch 逐个交付仅释放各自半边且对照 owner 不受扰动、`_assert_run_tail_clean`；空 shard 逐出零 watch）、DecodeJoinerAndGrowthWatchRealPathTest（drain 镜像登记→真实 emit_iteration_train 构图（watch id 取自 getattr 生产消费面）→真实转正→尾 watch 释放两旁支；无 drain 登记 fail-closed 拒绝；同用例增长腿以 API 合法 noc_migrate 转移钉住 HBM 端口登记与交付后三表配对清空）、FailedAdmissionWatchRealPathTest（真实 `_emit_eviction_only_nodes` 全链：kv_eviction 披露行、缺省 failed_q001 口径、回滚边界旁支在场、交付仅释放本旁支；空集零 watch）、EvictionWatchGuardBranchesTest（8 分支守卫逐支对齐生产：malformed/outside-batch/scheduled-twice/duplicate-owners/fired-before-scheduled 条目存活/unknown+重复交付/错阶段拒绝且不销账/changed-owner）、DrainRegistrationSlotsFixPinTest（:1703 赋值在真实 runtime 上合法）。聚焦四文件复跑（交接轮实收）88 passed、新文件单跑 15 passed；本收口轮三口径计数见 §43.8。

### 43.4 C4b 夹具修复登记：overlay extern 符号链接

首次 8 臂尝试在第一臂 plan 物化阶段失败、**0 launch、协调者授权 ≤8 次首轮后端 launch 预算未耗**：根因 = generate_trace.find_project_root 的 extern 标记在 overlay 布局中缺失（overlay 内无 extern 符号链接，物化器报 "Cannot find ASTRA-sim project root from generator path"）；修复 = prepare_overlay 增补 overlay/extern 符号链接后重试成功（终版 stay.plan_materializer.json 合法、该臂 rc=0）。失败日志存档 `/home/sunhao/joint_c4b_evidence/long_real_background/stay.plan_materializer.attempt1-failed.json`（945 字节 ASCII traceback，登记不删除）；注意该重试为物化器重试而非第二次后端 launch、不体现在 launch_ledger（账本 8 launch 口径不受影响）。

### 43.5 C4b 八臂 PASS 与三表数字

8/8 臂 PASS（交接复核逐臂核验在案：目标请求恰一次准入、四动作均适用且不适用原因在案、同 case 跨动作 preadmit 状态哈希一致、long_real_background 真实外部在飞背景非空、action_pin==forced_action==臂动作、6 非 stay 臂 chosen_instance 为 applicable 内 (cost_ns, instance) argmin 独立复算一致、终点 final_settlement 同口径、2s 冒烟输入与源 trace 逐请求 token 映射零违例）。三表：`c4b_prediction_errors.csv` 8 行——**MAE=661602934.5 ns、MAPE=39.55863242789121%（zero_measured_excluded_from_mape=0，无零值实测排除）**；`c4b_action_ranking.csv` 12 对——**ranking_error=0（action_ordering_errors=0/ranking_error_rate=0.0），strict comparable 10 + 显式并列 2（stay vs recompute 预测与实测均并列）**；`c4b_selection_loss.csv` 2 行——**选择损失 0 ns / 0.0%**（两 case 自然选择=stay@inst0=实测最优）。证据根 `/home/sunhao/joint_c4b_evidence`。冻结注记：freeze v2 watchfix-2026-09-24，source_tree_sha256 前 16 位 `f8a12038f3e77a6c`、binary sha256 前 16 位 `a0425b9656b7f0eb`（HEAD=9a95e06a78f004ca），lock 同 inode、`assert_c4b --self-test` 零 launch 自检 PASS、ledger_starts=0。**披露"强制 ≠ 自主选中"三层在案**：章程原句（joint仓库改造执行详细计划.md:341/:326/:351）、数据层（8/8 臂 bridge capture 同时记录 forced_action/action_pin 与 natural_chosen_action 全 stay，selection_loss.csv 含 natural_predicted_action 列）、机制层（overlay c4b_hooks/sitecustomize.py:255-257 预注册注释 + assert_c4b.py:160-166 argmin 强制校验）。

### 43.6 sensing 入口完整成功 run

exit 0；证据根 `/home/sunhao/joint_sensing_entry_evidence`（run_dir=sensing_run）。快照核对通过（trace_config.csv.snapshot 与 hardware_config.json.snapshot 在案，后者与 face_case5_config_c 权威源逐字节一致，含 validation-160gib capacity-profile）；SH_RUNTIME_RC_DIR 显式设值的 run 级消费为间接证据链（cpp.log custom ring id 20/21/26/27 与 RC 目录 comm_group.json ranks 精确对应 + RC 四件本次 run 启动时刻 mtime + run_online_strategy_sensing.sh:74/:211-214 代码行）——run.log/cpp.log 不回显 RC 目录路径且显式值恰为缺省档，行为上与缺省回退不可区分（限制如实登记）；账本 ledger.jsonl 21 行与请求一一对应、sensing_query_log.jsonl 1392 行；后端仿真全程持仓级锁 `.single_simulation.lock`（flock OFD 范式），事后 `flock -n` 复验 LOCK-FREE 无残留持有者。plan 物化步对 request-neutral 占位指针 fail-closed（plan_materializer.py:322-328），故按 joint_smoke_matrix.sh 范式在同一锁内先把 trace_config.csv 的 request_queue_csv 指针切至 requests_2s.csv、run 结束后已还原为占位值（对任务单命令清单的必要补充而非替换）；仓内 `generated/` 新增本次物化目录且 runtime_config 四件被重物化——按指示未清理，留收尾统一处理。

### 43.7 13 臂矩阵（P 批）全绿与行为注记

13/13 臂 exit=0，证据根 `/home/sunhao/joint_smoke_evidence_p`，输入为授权 2s 窗（smoke_input provenance 记 generator_version="materialize_20_30s.py window_ns=2000000000"，21 请求 5 session）。**行为注记（如实登记）**：准入级 chosen 分布与 O 批（§42.5）逐数一致——非 J 族 {stay:16, copy:5}、J 族 {stay:20, copy:1}、recompute 与 remote-read 选中全域 0（chosen_remote 全 0 维持 §39.4/§41.6/§42.5 登记）——任务单预告的逐出生命周期修复致分布漂移在准入级 chosen 粒度未显形，如需呈现应在 per-request candidates/passes 粒度另查；quota 两臂 remote-read applicable=32（128 中 96 条新增 quota_link 不适用拒判定，quota 机制生效 run 级实证；quota_events={admits:2, releases:2}、aimd 臂另含 aimd_action_counts={comfort_cold_start:3}——O 批无对照基线，仅登记）；TJE_quota_static 臂 link_telemetry_injected=False、本窗零遥测样本，与 aimd 臂（注入且 collective_coverage=true/1392 epoch）不对称（仅登记，不判定是否本批预期配置）；verify_run_end 检查由 exit_code=0（fail-closed 断言，online_scheduler_base.py:1041 协议）与决策日志收尾行（link_telemetry_coverage→joint_decision_metrics，verify_run_end 体内落字）承载，run.log 无字面 verify_run_end 摘要行——若后续审计要求字面结果行则当前证据面不满足（口径限制登记）。

### 43.8 测试计数（三口径，本收口轮实收复跑）

canonical（README §6 命令）= **972 passed + 1 skipped + 33 subtests**（上轮暂停基线 957+1 skipped+33 subtests → 净增 15 = §43.3 新测试文件）；窄命令（workload/llama2_7b_inference 全目录）= **834 passed + 13 subtests**；slo_tools `python3 -m unittest discover -s tests` = **171 ran OK（skipped=1）**，与上轮暂停基线 171 OK+1 skip 持平（watch 新用例不入 slo_tools 目录，符合预期）。三命令完整输出存 /tmp/joint_exec/docs_close/{canonical,narrow,slo}.out。

### 43.9 限制与未验证项（承接登记）

C20 决定性实验集未启动（需单独授权，本轮明确禁止）；C13 维持既有登记；2s 轻载窗 M1 流数分叉仍由单测承载（§40.6 口径不变）；C15 无改动，故 C4b 回归重跑登记 N/A；sensing 决策序列与 strategy 档的逐字节一致性未做全量 diff。另有 C4b 复核三项登记不修复残留：物化 CSV description 列文本陈旧（写 "first-30-seconds window" 与实际 2s 窗矛盾，数据与正式溯源均确证 2s）、long 案例 pre_admit_state.external_active_members 成员条目重复两见（不影响哈希一致性，身份语义待机制侧确认）、"强制 ≠ 自主选中"披露文句无证据根内成文报告文件（三层披露在案见 §43.5，正式报告若需保留该文句属证据根之外交付物）——均在 `/home/sunhao/joint_c4b_evidence` 在案。

## 44. SerDes 片外链路并发化改造（远端端口流体后端 + 发射门控解耦，2026-09-24，《SerDes片外链路并发化改造执行方案.md》V5.3 joint 仓实施批）

> 本节由阶段 7 文档同步员落字；所有断言仅覆盖 /tmp/serdes-joint-work/ 在案证据与本仓工作树亲读，未做的检查不冒认（见 §44.5）。

### 44.1 落点与范围（仅本仓；其余仓由各自批次登记）

- **后端整链替换** `extern/remote_memory_backend/analytical/AnalyticalRemoteMemory.{hh,cc}`（hh 338 行/cc 1150 行）：旧串行 FIFO（`PendingMemoryRequest`/`pending_requests`/`ongoing_transaction`/`start_request` 与串行 runtime helper）删除，替换为每端口作业集 + 连续 ns 子步流体推进 + 挂首次 `set_sys` Sys 的单一全局最早 deadline 可取消变迁事件 + 双零一次性定时作业 + 只读 `PortStats` 快照 + 惰性逐事务明细写者（hh:35-63 顶注；事件 payload 只带 kind+seq，hh:226-231）。
- **发射门控** `astra-sim/workload/HardwareResource.{hh,cc}`：单一代语义分类 `classify_hw_resource()`（timer no-op / local HBM KV restore→hbm_dma 单槽 / 其余 MEM_LOAD/MEM_STORE→remote_mem 独立计数制无上限槽 / cpu / comp / comm）由 ETFeederNode 与 NodeView 两套 overload 逐字共享（cc:27-90 区注释）；远端 MEM 不再落 comm 单槽（旧隐藏门两路径均除）；release 递减前 NDEBUG 无关 fatal 校验、occupy 重复 id abort、析构未释放诊断（hh:26-52）；`tics` 不为 remote MEM 新增（端口服务指标唯一来自后端流体/账本，hh:114-119）。
- **Workload** `astra-sim/workload/Workload.{hh,cc}`：static 仿真结束门补 `num_in_flight_remote_mem_ops==0`（cc:1143-1148）；新增 `hbm_join_pending_count()` 正常结束审计入口（hh:78-85）。
- **online main** `network_frontend/analytical/congestion_aware/main_online.cc`：`memory_api` 去 const（:1211-1215）；sensing 开启时 `enable_transaction_telemetry`（run_id 取 metrics manifest，空回退规范化 bridge_dir，:1216-1235）；正常收尾三段 = 逐 rank `hbm_join_pending` 审计（>0 报错退出，:2186-2205）→ `is_drained()` fail-closed 门 + `shutdown()/reset()`（Sys 全存活期执行，之后才删 Sys，:2207-2244）。
- **CMake/runner**：三个 test-only 目标注册（`..._RemotePortNwayTest`/`..._RemotePortOnlineGateTest`/`..._RemotePortStaticGateTest`，不入生产门）；`run_online_strategy{,_sensing}.sh` 搬运循环加 `remote_memory_transactions`、`archive_run_outputs.sh` 常驻白名单加同名并更新头注。

### 44.2 新语义（与方案 §3 逐条对应，行号为亲读锚点）

- **流体**：issue→固定 latency 可重叠不占带宽→正字节入流集合 N 路均分→任一流耗尽即出分母、幸存流即刻重分；Workload 回调唯一落在 `ceil(fluid_finish_ns)`；耗尽流在速率计算前翻转（`flip_due_jobs`，cc:277-283）。
- **双零（0B/0ns）**：独立一次性定时作业精确 +1ns 异步回调、不进带宽集合（cc:817-846）——旧同 Tick 同步完成语义废止（方案 §3.2 语义变更）。
- **事件**：issue 恰逢已入队事件 Tick 时保留原事件（keep rule，cc:532-536/:861-862，`transition_keeps` 计数）；派发先全端口推进、收集**全部**到期作业、按 `(port_index, issue_seq)` 排序（cc:601）、端口统计结算严格先于回调、不递归派发；stale generation 无操作（payload 经 `register_event_cancellable` deleter 释放）。
- **数值 fail-closed**：`remote-mem-bw` 正 finite（double 直读不再 uint64 截断、JSON 1e999/NaN/负值构造即拒，cc:183-206）；`remote-mem-latency` finite 非负且 Tick 可表示（cc:163-181）；bytes ≤2^53（cc:763-770）；ceil 前范围检查（cc:255-266）；残余 clamp 显式容差 1e-6 B / 1e-9 ns（cc:30-35）；`NO_MEMORY_EXPANSION` 下 issue 即 exit(1)（cc:752-756）。
- **端口映射**：PER_NPU（`npu-ids` 显式表去重校验或 set_sys 增量）、PER_NODE（`sys_id/num-npus-per-node`，缺键/越界 fatal，cc:694-723）、MEMORY_POOL 单端口 0（cc:734-736）。
- **PortStats**：issued/completed count+bytes、in_flight/peak（issue→callback 含 latency）、streaming/peak（事件边界重建非采样）、latency_waiting/completion_waiting 活计数（sensing 关也保留以支撑无条件审计）、`redistribution_events`（按连续完成时刻、幸存流>0 才计）与 `stream_join_events` 分列（cc:335-347）、port_busy_ns/shared_busy_ns/bytes_served 区间积分。
- **遥测**：`enable_transaction_telemetry` 复用既有 `--sensing-enabled`（**无新配置键**）；惰性建文件（首个完成事务才建）；观测键 issue 时拷贝、完成结算时交付前写出（永不从已交付 cookie 回读）；写失败/误用 fail-closed fatal（cc:1071-1103 区；`cannot open the transaction-detail stream` FATAL 在案）。
- **收尾合同**：`is_drained()` 无条件有效（sensing 无关）；`shutdown()` 幂等——取消事件与双零 timer（deleter 释放 payload）、销毁未交付 wlhd、结算审计查 count+bytes+活计数+容器重数+（遥测开时）行数=完成数。

### 44.3 验证（证据根 `/tmp/serdes-joint-work/`，2026-09-24；均当批亲跑）

- **终门禁** `gate_rerun.log`：49 步（configure + 全相关 C++ 目标构建/运行 + golden + stress）全 PASS（树=主工作树）。
- **端口模型** `RemotePortNwayTest` S1..S10 全 PASS（`precision_rerun.log` 尾行：anchors 300｜90/120/140/150｜1B fluid 1/6 ns，三 memory-type 全覆盖）；S10 fail-closed 反例 14 项（`portnway_s10_fail_*`：bw 缺/0/负/inf/字符串、latency 负/inf/字符串、未知 memory-type、per-node 缺 shape/零 nodes/未绑定 rank、huge tensor、no-issue）。
- **压力** `stress_results.txt`：16 端口×128 事务全并发——2048/2048 收发、peak_streaming=128、redistribution=2032=16×127（每完成时刻幸存>0 计 1、末流不计）、join=0、CPU 49.6ms、wall 54.7ms、peak RSS 13.5MiB。
- **系统门控**：online（`r4_online.log`）同 rank 双 MEM issue pass 后 in_flight=2、peak_streaming=2、shared_busy_ns=200；MEM+COMM_SEND 独立门；static（`r4_static.log`）双 MEM 同发 300/300、恰一次 occupy/release、结束门等全部终结；`hbm_nway_test` contention ON/OFF 重锚全 PASS（`r4_hbm_on/off.log`）。HardwareResource 行为 16 项检查（`hr_online_counted.log`/`hr_static_counted.log`，/tmp 一次性探针非仓内交付）。
- **事务明细后端级**：/tmp 编译探针 `remote_port_stats_test`（`build-joint/run11.log`，53 检查全过）——双流 2 行 JSONL 且行与统计结算严格先于每个回调（回调入口探针）、多时刻分批、join/完成两口径分列、零字节正 latency 不入分母、双零 +1 Tick、sensing 关聚合仍审计、早 shutdown 不伪造完成且零完成零文件、sensing 开零事务零残留、不可写路径 FATAL。
- **生产 2s A/B**（`pricing-audit.md` A2）：基线 = `/tmp/serdes-joint-baseline` 隔离快照树（14:32 双树 721 文件散列 manifest 恒等；基线构建 PASS、二进制 sha256 `a0425b96…`；树差仅 C++——基线串行符号 15 命中/流体符号 0，主树反之）；validation-160gib n=21 固定输入下 e2e 分位与全部指标文件逐字节恒等、决策 86 行仅宿主墙钟差、21 条 admission 全候选 cost_ns 逐字节同；sensing-on run 明细写者已武装（cpp.log `[online] remote-memory transaction detail: enabled`）而 JSONL 合法缺席 ⇒ **窗口池后端完成事务=0，恒等属零池负载平凡一致性，不构成争用正确性/收益证据**。
- **快照溯源补录（终审必改项，2026-09-24 晚落盘）**：任务指定的 `/tmp/serdes-joint-work/manifest-joint.txt` 此前缺失，现以可复算重建版入库——快照 tracked 基座 ≡ 工作区整备提交 `b2bfd9e420d6dee563ad13ef38eac20abf4a68cb` 的 template/astra-sim-joint 子树（678/678 文件 sha256 对 `git cat-file blob` 逐一相等、0 失配；含 2 个非 ASCII 路径单独验证）＋ 43 个 untracked（.zcode 15 / protobuf 派生 4 / 其余 .pyc 24）＋ ignored 输入散列（generated runtime_config 四件、plan 目录两件、trace_config 占位指针、traces/ 仅物化器）＋ 构建命令与双树二进制锚（基线 a0425b96… = §43.5 watchfix 冻结位逐位一致、candidate 09668a84…）＋ 五 run 输入 requests.csv 同散列 a09cef99…；自验证 V1-V6 记录在册（721/721 条对盘复核零失配）。同批入库 `/tmp/serdes-joint-work/joint-deep-audit-reconciliation.md`＝深挖条目↔快照现状对账表：方案 §2.2 指名条目（H1-H5/M3/M13/M17/M20/L7/L8/L10/L11/M18-M19/M1-M2/OfflineGreedy 整链/孤儿测试三件/P11 四组死键/run_golden_live）逐项对 b2bfd9e blob 亲验全部吻合"已修/已处置"，唯一仍在 = main_online detached command-FIFO 线程（:1333@b2bfd9e、:1357@工作树）与 §2.2 登记一致；顺带登记 M4 （CustomAlgorithm 基类成员未初始化）快照内仍未修——方案 §2.2 本未声称其已修，无冲突；其余未指名条目如实标注"未逐项复核"，不假装修复状态。
- **计价/标定**（`pricing-audit.md` A1/A4/A5）：divisor 四口径为 Python 登记表决策面、机制上不受并发化影响；remote-read 读流本体无池端口腿（不漏计）、混合基后缀池腿已计价；**重标定零值改动**（无 SLO/LUT/因子移动依据）；离线 golden 锚 `test_golden_g1g4.py` 门禁 step48 PASS（joint 的 `run_golden_live.py` 已随死代码清除批删除，本仓 live 锚不适用——与开关清单退役登记一致，未复活）。

### 44.4 文档同步（本轮交付面）

- `README.md`：§7-D' 重写（FIFO→流体并发/端口共享域/事件精度/限制五条，§7 标题加偏离注记）＋ §5 迁移清单第 7 条 ＋ §3 状态表"远端端口并发后端"行 ＋ §6 远端端口 C++ 夹具块（含 ctest=0 实测口径）。
- `experiment/仿真各功能开关清单.md`：新增 §18（零新增开关；`--sensing-enabled` J 仓行为扩展登记；`remote_memory.json` 键行为语义改写；§17"串行基线=改造前 commit 直接对拍"说法废止、改精确工作树快照口径；已退役死键 logical-pool/boost-mode/run_golden_live 不复活）；§2 路线④产物行与 §3 `--sensing-enabled` 行加 J 仓指针；§10.1 remote-memory 行补 J 仓取值。
- 本节（§44）。

### 44.5 限制与未验证项（如实登记）

- **争用条件生产级行为未验证**：现有全部 run 零池事务，逐事务明细复算、Python/C++ 传输对账、完成分布无生产样本；需编排层后续含池负载（如 PARTIAL 基或驱逐压力）sensing-on run。
- **明细写者仅后端级验证**：S7 + run11 53 检查；生产 run 零行（文件合法缺席）。
- **裸仓还原未执行**：本轮任务边界=文档同步；joint 工作树现存 `build/` 与 `generated/` 产物，待收尾按 Agents.md 执行 `clean_test_records.sh`/`clean_build_artifacts.sh`（不得对原始工作区盲跑，须按阶段 0 manifest 保护用户未提交/未跟踪文件）。
  - **【2026-09-24 收尾闭合】**上条已按其自记口径执行完毕：先做保护预检（`git status --porcelain` 亲核——4 个未跟踪交付件 `tests/remote_port_*_test.cc`/`make_remote_port_static_fixture_et.py` 均不在清理脚本 rm 目标内；`git diff` 亲核 `trace_config.csv` 相对 HEAD 零改动），随后亲跑 `clean_test_records.sh`（移除 `sh_test_mesh/generated/` 全部〔含本批 smoke 物化的 `llama2_7b_inference_54npus_plan_9b94e196`〕、全树 `__pycache__`/`.pytest_cache`；traces/ 仅 *.py；trace_config 占位指针未动）与 `clean_build_artifacts.sh`（移除 `build/`）。复核（均亲跑）：`diff -rq` vs `/tmp/serdes-joint-baseline` 仅余本批有意交付面（C++/CMake/runner 三脚本/README/PROVENANCE 修改 + 4 个新测试文件）与基线侧产物条目（基线自带 build/、generated/、7 处 __pycache__——工作树较基线更严格，按裸仓定义清除，基线在案 ignored 输入不回填）；裸仓四要素逐项过（指针=占位、traces 仅 .py、无 generated/、无 build*/）；`find . -name __pycache__ -o -name .pytest_cache` 零命中。
- **detached command-FIFO 线程观察项维持不处置**（原 joint 深挖清单 §1.3）：本轮 main 收尾新增的审计/结算步均在 `svc`/`ingress` 栈作用域（main_online.cc:1272-1275）仍存活处执行、FIFO 线程 detach 语义与 :1357-1359 未动——未扩大其销毁后访问窗口；该观察项本体仍开放。

## 45. PARTIAL 跨实例 copy 流水化（发射门槽位放宽 + 图侧并行化，2026-09-25，仅本仓，零新增开关）

> 本节由文档同步员落字；断言覆盖任务单开发自测记录（`/tmp/joint-pipeline-work/` 在案）与本同步轮亲读亲跑复核，未做的检查不冒认（见 45.4）。

### 45.1 落点与范围

- **发射门 C1** `astra-sim/workload/HardwareResource.{hh,cc}`：`hbm_dma`（本地 KV restore）与 `gpu_comm`（COMM_SEND/COMM_COLL）两类单槽改**计数制无上限槽**（occupy/release 删单槽 assert、available_class 恒放行；与 remote_mem 同款合同——计数仅供 static 结束门/release 前置校验/析构诊断，永不作带宽分母；COMP/CPU 单槽门按方案保留）。`LocalHbmBandwidthModel.hh` 注释同步（并发 RESTORE 为受支持状态）。带宽仲裁零改动归既有模型：HBM 六类作业 N-way 严格均分（加入/完成动态重分）、NoC 链路按 active flows 均分（FluidScheduler）、池端口按 active streams 均分（`AnalyticalRemoteMemory`，§44 交付不回退）——单槽压住的"每 rank 至多 1 笔并发 RESTORE"结构保证删除，NoC 共享链均分模型已在位、图侧并行化后首次真生效。
- **图侧** `graph_batch_builder.py`（发射依赖面）：P1 `_emit_copy_handoff_tail` 尾块逐笔并行支链——每块对涉及 rank `chain_checkpoint`/`restore_chain` 段内分支、全部块首节点 fork 头块后主链 frontier、块间无边；块内 ack 新形态 ack_recv_c←本块 send_c／ack_send_c←本块 recv_c；跨 rank send/recv 无图边、(src,dst,tag) 配对运行时承载；同实例 gate 逐块重 arm（等价旧"全部尾块等 gate"）；旧「home send 直连链/exec recv 顺序链/ack 链尾随」三段串行删除。P2-R2 `_arm_pending_store_edges` 公开签名/行为逐字节不变地分解为 `_match_store_entries`（纯查询+历史文案 fail-closed）／`_emit_store_relays_once`（每跨缘对恰一次 1B 中继+可选去重缓存）／`_consume_store_entries`（恰移除 matched）三助手；同实例（stay partial）与跨实例（copy/remote-read 支链）两个恢复组发射点改**组间并行**：①组 0 区间无匹配 raise 面②并集区间中继先发（中继 recv=各组共同链序父）③每组 checkpoint＋只 arm 与本组层区间交集的条目＋发射＋回滚④并集条目同批发射内消费（span 外保留；pending_store_tails 窗口有界，sh30 merge_tail_gated 判据输入不放大，sh30 零改动）；组 i 首节点依赖由"组 i-1 全部节点"弱化为"驻留前缀屏障/到达 gate＋本组交集 store 直接 arm＋本缘共享中继（链序）"，前递保障不减。P4 `_emit_credit_stream_tail` 同款并行化（home/exec 两腿图形态对称）。新增 `_restore_group_involved_ranks`。
- **零改动面**：首块（chunk 0）准入主链、readiness barrier、逐块/逐层就绪门、FS 四步协议、merge_back v2、home/merge 账本一行未动。

### 45.2 测试与验证（C++/Python 为本同步轮亲跑复验，其余为任务单在案）

- **C++**：新增 `PipelineConcurrentGateTest`（场景 P：同 rank COMP＋KV-restore＋两笔 charged send 一次 pass 齐发、free 集合空、sends=650<restore=700 先于 restore 终结、COMP=500、peak=4、redistribution=5、served bytes 精确钉；场景 Q：RESTORE_A=500/RESTORE_B=650 动态重分、peak=4、redistribution=5）ALL PASS；新增 `NocSharedLinkNwayTest`（4 NPU 线拓扑：F2=12/F1=8/F3=6——共享期 2000 B/ns 严格均分、F2 完成后链路动态重分、排序 6<8<12）ALL PASS；两目标注册 analytical CMake（:192/:196，test-only 照 RemotePort* 模式）。`HbmNwayTest` **重锚 2**：rank0 node3 uncharged send 由 ≥700 floor（钉 comm 单槽串行）改精确钉 6（tick 0 齐发、网络侧到达）＋HBM 池不受泄漏复核断言——ON/OFF 双跑 ALL PASS，ON terminals r0#0=400/r0#1=600/r0#2=700/**r0#3=6**/r0#4=5400/r1#0=500/r1#1=800/r1#2=1000/r1#3=1100 全命中（其余数值锚与 OFF 臂结构性不变）。开发批整链 `/tmp/joint-pipeline-work/run_joint_cpp_regress.sh` ALL PASS（任务单在案，本节未重跑）。
- **Python**：T2 `test_joint_copy_handoff.py` 18→23（+CopyHandoffParallelTailTests 5）、T3 `test_joint_layer_restore.py` 17→20（+RestoreGroupParallelTests 3）、T8 `test_store_restore_ordering.py` 9→13（+StoreForwardingHelperTests 4）——三文件本同步轮复跑 56 passed；窄命令 `workload/llama2_7b_inference` 实收 **846 passed + 13 subtests**（§43.8 watchfix 基线 834 → 净增 12，本同步轮复跑）。
- **冒烟与终门禁**：2s 冒烟矩阵 13/13 臂 exit=0（证据根 `/tmp/joint-pipeline-work/smoke_evidence`，含 combo 8 臂+remote_off+shadow_verify+quota_static/quota_aimd/face_static；2s 窗 21 条 admission 全 copy 不适用——冒烟验证通用发射链路无回归，PARTIAL copy/恢复组图形态正确性由 T2/T3 结构断言与 C++ 夹具承担）；交付终门禁结果 = **回归门禁全绿＋10s 压力夹具通过**（命令面 regressionCommands，本节未逐一重跑）。

### 45.3 文档同步

- `README.md`：§3 状态表 copy 块级交接（18→23 用例＋P1 并行形态）/逐层恢复（逐组顺序→组间并行）/remote-read 执行流（D2 拓扑 P4 并行形态）/远端端口并发后端（comm 槽措辞）四行＋§5 迁移清单第 7 条措辞与新第 8 条＋§6 测试计数注记（T2/T3/T8 增量、窄命令 846+13）与"跨实例 copy 流水化 C++ 夹具"块＋§7-D' 槽位措辞（comm/hbm_dma 同为计数制无上限）。
- `experiment/仿真各功能开关清单.md`：§18 已有 2026-09-25 追记（开发批落字，本核对轮逐条复核与代码一致）；§14（F/W）/§16（FL/WL）差异表无本批（仅 J 仓）涉改条目，零订正零新增开关。
- 本节（§45）＋`PROVENANCE_hashes.txt` 的 PROVENANCE.md 行重算。

### 45.4 限制与未验证项（如实登记）

- canonical 全量 pytest 与 10s 压力档由终门禁统一执行，本同步轮未重跑（门禁结果以任务单为准，未复核门禁日志本体）。
- 端到端 PARTIAL copy/恢复组时序未入 2s 冒烟窗（窗内零 copy 命中），由终门禁压力档与 T2/T3 结构断言、C++ 夹具分层承担——压力档逐请求数字本节未复核。
- §44 已落地的 AnalyticalRemoteMemory 并发端口与 RemotePort* 绿灯未回退（本批零改动其后端；RemotePort* 三目标本同步轮未重跑，开发批整链复跑在案）。

## 46. prefill remote-read 分阶段评审修复（#prefill_read 串行折叠 + 偏差注销 + 同日共存登记，2026-09-25，仅本仓）

> 本节由修复工程师落字；四处评审结论逐条处置（①medium/②③④low），全部改动亲改亲跑（46.2），未做的检查不冒认。

### 46.1 改动与裁定

- **评审①（medium，#prefill_read 除数随组数虚涨）**：`sh30_online_scheduler.py` 准入登记 `#prefill_read` 改传 `serial_credit_stream=True`——图发射侧前缀层组是 stop-and-wait 串行链（组 g+1 的 home send/exec recv 经 builder 链序后随组 g 的 ack，GB `_emit_prefill_remote_read_branch` ②注释在案），同 shard 路径同刻至多一组在途；此前逐组按并发流登记使同 tick 后续决策看到的争用除数随组数（⌈p/8⌉）虚涨。与 `#readplan`/`#decode#j` 同纪律；`_register_transfer_flows` docstring 的适用面注释同步扩展。释放/回滚/泄漏审计均按 owner 注销（`_release_transfer_flows`/`_rollback_admission_registrations`/`_assert_no_flow_registry_leaks`），在册流条数变化零影响。`test_joint_preadmit_visibility.py` 的 `_register_prefill_read` 夹具为单笔 transfer（2 shard 异键不折叠），断言不受影响、未改。
- **评审②（low，偏差清单过时）**：`face_scheduler.py` `KVTransfer.stream_only` 注释原载「decode credit 读流同口径应置 True……本仓任务面未动，见 ImplHandoff 偏差」已过时——规格书§一.7 已在调度器 `_joint_remote_read_slice` 补齐（`stream_only=True`，其 docstring 亦载「2026-09-25 补前序 KV 管理器卡登记的偏差项」），并被 test_prefill_remote_read_scheduler / test_prefill_remote_read_lifecycle 的 `block.stream_only` 断言钉死。注释订正为闭环口径，ImplHandoff 偏差注销；以代码为准。
- **测试钉死值随①更新**：`test_prefill_remote_read_lifecycle.py` 准入 `#prefill_read` HBM 条目 8→4（`(0,2,1,3,0,2,1,3)`→`(0,2,1,3)`，单代表流口径，与 drain 点 `#readplan` 同形）；另两处注释（「不做 serial 折叠」「8=2 组并行层段流」）同步订正。断言强度不减（仍精确 tuple 相等）。
- **评审④（low，无需改码）**：`test_prefill_remote_read_graph.py::test_branches_overlap_not_serial_under_asymmetric_bytes` docstring 已载时延等价性说明（图侧时延由字节×拓扑经 C++ 代价模型派生、无独立时延参数；双向 `_ancestors` 不可达偏序 = 任意时延指派下两支重叠），维持原样。

### 46.2 验证（本轮亲跑，均带 `-p no:cacheprovider`）

- 窄集（①直接触及面）：`python3 -m pytest template/astra-sim-joint/sh_test_mesh/workload/llama2_7b_inference/online/test_prefill_remote_read_lifecycle.py template/astra-sim-joint/sh_test_mesh/workload/llama2_7b_inference/online/test_prefill_remote_read_scheduler.py template/astra-sim-joint/sh_test_mesh/workload/llama2_7b_inference/online/test_joint_preadmit_visibility.py -q` → **37 passed**。
- 全量自查：`python3 -m pytest template/astra-sim-joint/sh_test_mesh/workload/llama2_7b_inference/online -q` → **441 passed + 11 subtests**；`python3 -m pytest template/astra-sim-joint/sh_test_mesh/workload/llama2_7b_inference/joint -q` → **419 passed**（与修复前基线逐数一致，零回归）。

### 46.3 与 §45（PARTIAL 跨实例 copy 流水化，同日另一股工作）共存的文件级清单

全仓零 commit，两股未提交改动在同一批文件中交织，git 无法分离归属——提交时建议按本节/规格书批次与 §45 分拆 commit。文件级触及面：

| 文件 | prefill remote-read 分阶段（规格书批次 + 本节） | §45 copy 流水化 |
|---|---|---|
| `online/graph_batch_builder.py` | `_prefill_remote_read_arms/_layers` 双账本、`_emit_prefill_remote_read_branch`、`emit_layer_segmented` 前缀组段、首步消费/列车尾与完成残留 fail-closed | P1 `_emit_copy_handoff_tail` 尾块并行支链、P2-R2 `_arm_pending_store_edges` 三助手分解＋恢复组组间并行、P4 `_emit_credit_stream_tail` 并行、`_restore_group_involved_ranks` |
| `online/sh30_online_scheduler.py` | runtime 三字段、准入规划/两腿分离/`#prefill_read` 登记（含本节 serial 折叠）、`#readplan` 对账链、`_joint_remote_read_slice` stream_only | 按 §45.1 自述零改动（P2-R2 特意保持 sh30 零改动） |
| `face_scheduler.py` | `plan_prefill_remote_read_transfers`、`KVTransfer.stream_only` 字段（含本节注释订正） | 零改动 |
| `joint/joint_cost_model.py` | remote-read 分阶段关键路径 + notes 四键披露 | copy_streaming 关键路径、`_final_footprint` 逐 rank 校验 |
| 测试（新增/增量） | 新增 `online/test_prefill_remote_read_{manager,graph,lifecycle,scheduler}.py`、`joint/test_joint_remote_read_staged_path.py` | 增量 `test_joint_copy_handoff.py`（+5）、`test_joint_layer_restore.py`（+3）、`test_store_restore_ordering.py`（+4） |

其中 P2-R2 恢复组组间并行顺带改变了 remote-read 后缀恢复腿的组间拓扑（组间串行→并行，`graph_batch_builder.py` C15 发射点注释在案）——超出本规格书字面但经独立评审核与本任务语义无冲突（定向测试全过、`_arm_pending_store_edges` 公开行为不变），归属切分由仓库所有者在提交时确认。

### 46.4 限制与未验证项（如实登记）

- canonical 全量 pytest 与 10s 压力档不在本轮任务面（任务约束为两套自查命令），未执行。
- `PROVENANCE_hashes.txt` 仅重算本轮实际改动且在钉清单内的三行（PROVENANCE.md / sh30_online_scheduler.py / face_scheduler.py）；graph_batch_builder.py 等行的既存陈旧（§45 与规格书批次遗留）未代更。
- joint_cost_model.py 未在本轮改动；其工作区状态系规格书批次与 §45 交织的原样。

## 47. prefill remote-read 分阶段改造（前缀读流 prefill/decode 两段接力 + 分阶段关键路径计价，2026-09-25 规格书批次主体；2026-09-26 文档同步收尾登记，仅本仓，零新增开关）

> 本节由收尾文档同步员落字；机制断言全部亲读工作树代码锚定（行号为 2026-09-26 收尾时点），验证仅覆盖亲跑命令与 /home/sunhao/joint_smoke_evidence/ 在案证据的亲核项，未做的检查不冒认（见 47.4）。评审修复轮已由 §46 登记，本节登记规格书批次主体与文档同步面，两节互补。

### 47.1 落点与范围（规格书批次 = §一 规划 / §二 准入与流生命周期 / §三 图发射 / §5 计价）

- **KV 管理器** `face_scheduler.py`：`plan_layer_groups(start, end, group_layers)` 公共规划器抽取（:1255 起；C15 后缀组 / C13 交接块 / prefill 前缀组三特化同源，确定性、无运行期状态输入）；`KVTransfer.stream_only` 瞬时流标记（:1582，True = 只产 send/recv/HBM 写服务节点与完成门、不物化入任何持久账本；prefill 前缀组与 decode credit 切片同置 True——§一.7 偏差闭环已由 §46.1 登记注释订正）；`plan_prefill_remote_read_transfers()`（:4525-4596）纯规划零副作用：home 驻留前缀 [0,p) 按 RESTORE_GROUP_LAYERS 组切分为逐组 noc_migrate 读流（phase="prefill"、reason="remote_read_prefill_prefix"、stream_only=True、驻留指针 before/after 恒 p），后缀 [p,L) 仍由 prepare_prefill 规划池恢复、两腿并行分叉；适用性合同 = LOCAL/PARTIAL 基 ∧ exec≠驻留实例 ∧ history_tokens 相符，REMOTE 基 fail-closed raise。
- **在线调度器** `online/sh30_online_scheduler.py`：runtime 三字段 `prefill_remote_read_{transfers,plan,bytes}`（:396-398；plan_dict 披露键 :329-330）与 history_transfers 严格分离（history_transfer_bytes 不含前缀读流）；准入序 = 规划置于 reserve 之前（:4843-4858，fail-closed raise 不留预约残留）、事务成功后落账（:4943-4967，无条件覆盖赋值防 quota 回队重试残留）；流生命周期 = 准入登记 owner ``rid#prefill_read``（:5008-5022，serial_credit_stream 单代表流折叠——同 shard 路径同刻至多一组在途、争用除数不随组数虚涨，§46.1）、prefill drain 释放（:1624）、`_assert_no_flow_registry_leaks` 覆盖（:3650-3665）；**#readplan 准入相注册表零登记**（:3393-3416/:5023-5035，规格书§二.5）——prefill 期间注册表上的远读流 = rid#prefill_read 实流本身，同一条前缀读流不得同时登记为 prefill 流与 decode 流，#readplan 注册表半边自 `_reconcile_readplan_at_drain`（:3479 起）承接 decode credit；decode 侧 credit 计划层界 = 前缀 [0,p)。
- **构图器** `online/graph_batch_builder.py`：`_prefill_remote_read_arms/_layers` 双账本（:448-454，成对登记/成对消费、单边在场即 raise）；准入发射摘出前缀读流独立旁挂分支（:2497-2633）——remote-read 无 history_transfers 时旧"gate 从未被消费＋前缀层计算无数据门"pass 分支废除、缺前缀读流即 fail-closed；`_emit_prefill_remote_read_branch`（:2936-3129）：组区间复检 [0,p) 自 0 连续铺满（缺口/重叠/基形态/相对位 rank 对齐即 raise）、`TransferTriggerGate(control=exec)`（:3054）经 `_emit_transfer_trigger` 同款 1B relay 触发链恰消费一次 interval/arrival gate（:3060-3081）、组间 home send 链/exec recv 链序连接不逐组重复消费 gate、逐组 recv 完成门＋层区间入双账本（:3123-3127）；列车体首块恰一次消费双账本（:938-966）按层段门控（[0,p) 前缀段等前缀 recv 门、[p,L) 后缀段等恢复写门），首步批恰一次/余量批不重复，列车尾（:1638-1663）与准入发射残留（:1845-1847）fail-closed。
- **代价模型** `joint/joint_cost_model.py`：remote-read 候选合成改**分阶段关键路径**（模块注释 :100-115、:1517-1522）——prefill 腿 = 前缀 [0,p) 读流（prefill_scans 遍）、decode 腿 = 前缀 credit 读（decode_steps 遍；后缀 exec HBM 就地复用不再恢复），两腿同 K 源（remote_credit_block_size 单一裁决点）各自拆首 credit/余量流（:1785-1849）；合成 `prefill_stage = 前缀首credit + max(前缀余量流, 后缀池恢复, prefill计算)`、`decode_stage = decode首credit + max(decode余量流, decode计算)`、`cost = target_wait + eviction_wait + prefill_stage + decode_stage + merge`（:2008-2050）；旧 `max(history_prep, eviction_wait) + first_credit + max(remaining_stream, compute)`（后缀恢复全量串在 prefill 计算之前）**废除、无开关可选项**；C5 冻结 breakdown 字段口径不动——`remote_read_first_credit_ns`/`remote_read_stream_ns` 转合并流日志兼容披露（:2097-2103），阶段量经 notes 四键 `remote_read_prefill_ns`/`remote_read_decode_ns`/`suffix_restore_ns`/`prefill_pipeline_overlap_ns`（:2044-2048）。
- **规格书符号对接（任务规则 6 项）**：本批规格书符号 `_emit_transfer_trigger()` / `TransferTriggerGate` 与本仓代码实名一致（graph_batch_builder.py:86/:3054/:3072 亲读在案），无新增命名偏差；任务提示中"_emit_store_relays_once/_arm_pending_store_edges 对应"一例属 §45 P2-R2 分解的命名对接、已在 §45 登记，与本批无涉。

### 47.2 验证（命令与证据均本轮亲跑/亲核，2026-09-26）

- **自查命令（任务指定两套，均带 `-p no:cacheprovider`，全绿、零 --ignore）**：`python3 -m pytest template/astra-sim-joint/sh_test_mesh/workload/llama2_7b_inference/online -q -p no:cacheprovider` → **441 passed + 11 subtests**；`python3 -m pytest template/astra-sim-joint/sh_test_mesh/workload/llama2_7b_inference/joint -q -p no:cacheprovider` → **419 passed**（与 §46.2 逐数一致、零回归；两套合计 860 = 任务单基线 793 + 新五文件 66 + preadmit 净 1，精确闭合）。
- **补充计数（本轮亲跑）**：新五文件逐一单跑 = manager 16 / graph 25 / lifecycle 9 / scheduler 7 / staged_path 9（合计 66）；窄命令 `workload/llama2_7b_inference` 916 passed + 13 subtests（§45 批 846+13 → +70 = 66+1 + 3 未逐文件归因，与 47.4 差额条目同源，如实登记不强解）；canonical 全量（sh_test_mesh 下 `pytest tests/ slo_tools/tests/ workload/llama2_7b_inference --ignore=slo_tools/tests/test_{driver_parity,golden_g1g4,slo_contract}.py -q`）**1054 passed + 1 skipped + 33 subtests**。
- **冒烟证据（/home/sunhao/joint_smoke_evidence/，文件时戳 2026-09-26 00:05–00:57 +0800 亲核；任务指定的 10 个指针文件全部在场）**：
  - `pin_overlay/sitecustomize.py`（v2 迭代 3）：仓外 action-pin overlay——生产配置 remote-read 自然选中恒 0（README §6 L 批勘误"矩阵全 13 臂 remote-read 选中数恒 0"、§41.6/§42.5 在案；定价涌现结果），overlay 强制策略 (a) PARTIAL 基有适用 remote-read 候选即强制选中 / (b) 自然 copy ∧ 同实例有适用候选改选 remote-read；决策管线其余（quota 过滤/reserve/prepare/两腿规划/图发射）零改动全走生产路径；逐决策审计行落 `JOINT_PIN_LOG`。
  - `pin_stress64_150s/`（迭代 2，**exit 1**）：python.log 在案 fail-closed RuntimeError"train carries both remote-credit body blocks and copy handoff arms"（graph_batch_builder.py:616 emit_iteration_train 混编列车守卫；强制 remote-read 后才可达的构图组合，自然选中下不可达）——守卫发现如实上报，overlay 策略 (b) 即为绕开该组合取得全绿 run 的 overlay 层处置。
  - `pin_stress64_150s_r3/`（迭代 3，**exit 0**，16:42:24–16:56:46 UTC）：`results/online_decision_log.jsonl`（89.7 MB）亲扫 = joint_action remote-read 5384 行 / stay 2288 行；`decision.prefill_remote_read_bytes` 披露行 3836 行（非零 2692 / 零 1144；首个非零样本 session_1_request_1 = 5,812,781,056 B、32 层全前缀）——prefill 前缀读流分阶段路径端到端真实走通；`slo_restore_decomposition.rerun.csv`（155,231 B）在场。
  - `pin_overlay/pin_log_64gib150s.jsonl`（734 行）/ `pin_log_64gib150s_r3.jsonl`（1991 行）：决策侧 pin 审计行。
  - `pin_stress96_150s/invocation.json`（**exit 0**，16:26:52–16:29:46 UTC，combo=TJE，binary sha256 e05ac5a2…）。
  - `combo_TJE_2s/`（**exit 0**，16:05:06–16:05:10 UTC）：2s 标准冒烟臂无回归。
  - `stress_10s_28gib/TJE_stress-28gib/judge_summary.json`：partial_copy_hits=80、copy_prefill_rows=80、deep_gap_events=[]、merge_degrade 恒空——容量压力夹具判据 PASS 形态。
- **混编列车守卫（本轮冒烟暴露的新面，如实上报）**：同列车同时携带 remote-credit 体块与 copy 交接 arm 的构图组合（remote-read decode 续读成员 × copy 头列车）触发 fail-closed abort；本轮以 overlay 层避免该组合，机制级处置（守卫语义细化或列车隔离强化）未立项，留待后续轮。

### 47.3 文档同步（本轮交付面）

- `README.md`：§2"动作语义"remote-read 条（前缀读流分两段接力、LOCAL 基全层读流、瞬时 staging 不入容量账本）；§2"链路遥测"条（#readplan 准入相注册表零登记 + rid#prefill_read 实流占位 + drain 承接）；§3 状态表 remote-read 行（标题口径、分阶段执行口径、分阶段关键路径计价与 notes 四键、触发链/双账本/层段门/生命周期、五测试文件指针）；§5 完成路径第 9 条；§6 测试台账新批行（66 用例、窄命令 916+13、两套自查全绿、canonical 1054+1 skipped+33）。
- `experiment/仿真各功能开关清单.md`（仓外、.gitignore:49 忽略不入 git）：§7.0 `JOINT_REMOTE_ACTIONS`（两段接力）/`JOINT_REMOTE_CREDIT_ITERS`（分阶段计价 + 单测补 staged_path）/`JOINT_REMOTE_READ_PARTIAL`（池恢复改 prefill_stage 并行 max 项）三行 + §7.0.1 状态表 remote-read 执行流行；**零新增开关、零键值变更（旧直接改新）**。该文件本轮存在并行会话同步写入，本节四处改动均经写入后回读逐一验证在场。
- `PROVENANCE_hashes.txt`：README.md / PROVENANCE.md 两行随本节重算；sh30/face 两行 §46 已按当批重算；graph_batch_builder/joint_cost_model 等行维持 §46.4"既存陈旧、未代更"口径（本批零改码，不代前批钉清单行越权重算）。
- 本节（§47）。

### 47.4 限制与未验证项（如实登记）

- **冒烟 = 强制选中下的端到端通路，非自然选中**：pin 证据仅证明"选中后全链（规划→图发射→层段门控→decode 复用→生命周期恰一 owner）无 raise 无泄漏"；r3 决策日志 1144 行 `prefill_remote_read_bytes=0`（强制到无异地历史/零前缀场景）未逐行归因。
- **性能/收益未评估**：本轮不主张分阶段改造的时延或吞吐收益——无 A/B 对照；judge_summary 等数字仅作夹具判据 PASS 形态证据。
- 压力 run 的 SLO 全产物（slo_* csv 族）未逐件复核；150s 窗 cpp.log 未亲读。
- canonical 全量与窄命令为本轮文档同步员亲跑的补充证据；10s/150s 压力判据脚本未重跑（judge_summary 为在案文件读取）。
- 窄命令 +70 与可归因 +67 的差 3（见 47.2）：属 §45 批 846 记数与今日实收之间的历史口径差，未强行调和。
- `experiment/仿真各功能开关清单.md` 与本 PROVENANCE 同期存在并行会话写入；本节落字前对其余小节内容未做一致性复核（不在本批职责面）。
- **PROVENANCE_hashes.txt 全表复核（本轮亲跑 `sha256sum -c`）**：29 行中 13 行 OK（含本节重算的 README.md/PROVENANCE.md 两行与 §46 重算的 sh30/face 两行）、16 行 FAILED——均为 R17 定版后历批（§44/§45/规格书批次）改码遗留陈旧，维持 §46.4"不代更"口径，由仓库所有者在提交前决定是否整表重算。
