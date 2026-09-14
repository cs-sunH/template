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
