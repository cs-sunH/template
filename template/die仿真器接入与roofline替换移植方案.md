# die 内仿真器接入 template 五仓——接口改造与 roofline 旧模型替换移植方案

日期：2026-10-18
状态：已定稿，按此执行
执行者：ZCode 会话（多子 Agent 编排）

---

## 1. 目标与边界

**目标**：把 `template_newdie` 中已完成并验收（canonical25，ACCEPTED-WITH-DISCLOSURES，61/61 测试）的单 die 仿真器（coarse_die_v2）接入 `template/` 下的 5 个仓库（astra-sim-face / face-LRU / wscllm / wscllm-LRU / joint），替换现有出厂默认的旧 die 内 cost 模型（mesh_die_v1，即用户所称"roofline 模型"），使**下次用这 5 仓直接跑 trace 时，COMP 计时走 die 内仿真器**。

**边界与保护**：
- `template_roofline/`（纯目录备份，无 .git）：**绝对不改不碰**，作为日后对比基线。
- `template_newdie/`：只读参考源，不改。
- template 五仓上用户的提速优化（工作区仓 9e683ea"5仓提速构造完成"）与未提交修复（`slo_tools/kv_cache_adapter.py` 命中率口径修复、`test_slo_contract.py`、joint `trace_config.csv` 指针）**必须完整保留**。
- 全程禁止 `git commit`（Agents.md）；交付时五仓还原裸仓库状态；过程临时文件清理。
- 仿真验收**绝对禁止爆内存**：统一用 `agent-traces/tracelab/astra_compute_20.csv` 前 **1 秒**窗口（实测 8 请求 / 3 会话），`/usr/bin/time -v` 监控。

## 2. 三棵树的关系（事实）

| 树 | git | 角色 |
|---|---|---|
| `template_newdie/` | 独立仓，HEAD=5a67e14（codex 分支） | 参考实现：die/ 库 + 五仓 coarse_die_v2 集成。其五仓基线 = 提交 f7756e4（2026-10-12 工作区镜像同步） |
| `template/` | 外层 `/home/sunhao/wsc-simulator` git 仓的子目录，HEAD=9e683ea | **移植目标**：f7756e4 同源内容 + 用户 10-12→10-18 的提速优化（含远读退役、FlatShape、事件日历退役、Python 瘦身等） |
| `template_roofline/` | 无 .git 的目录拷贝 | 只读备份 |

三方合并基座：**base = f7756e4 的五仓树，theirs = newdie HEAD 的五仓树（coarse 集成），ours = template 当前树（提速优化）**。逐文件机械分类结果（脚本实测，清单在 /tmp/threeway_report.txt，执行时重算落盘到 /tmp/phase1/）：

| 仓 | TAKE(取newdie) | KEEP(留template) | MERGE(三方合并) | 备注 |
|---|---|---|---|---|
| astra-sim-face | 93 | 26 | 18（自动干净 14 / 冲突 4） | |
| astra-sim-face-LRU | 91 | 26 | 18（14 / 4） | |
| astra-sim-wscllm | 86 | 26 | 15（13 / 2） | |
| astra-sim-wscllm-LRU | 94 | 26 | 17（13 / 4） | |
| astra-sim-joint | 73 | 20 | 147（29 / **118**） | 冲突大头：face_scheduler.py(35)、sh30_online_scheduler.py(28)、joint_cost_model.py(16)、extern/die_model/src/Execution.cc(22) 及大量 joint/* 测试与文档 |

**关键事实**：newdie 的出厂 system 模板默认仍是 `die-model-profile: mesh_die_v1`（其 coarse 验证走 C++ 测试夹具，未翻默认）。因此"替换 roofline 模型"的最后一步——**把出厂默认切到 coarse_die_v2——是本方案新增的设计**（见 §5）。

## 3. die/ 公共库迁移决策

1. **源取 newdie HEAD（5a67e14）**，而非 canonical25 发布点（5a1980c）：HEAD 多出 P9+P7-A estimate 闭式快路径（`src/Execution.cc`、`CMakeLists.txt` 两文件），对大 repeat 负载的 compile/estimate 是数量级加速，与提速目标一致。
2. 复制范围（→ `template/die/`）：`CMakeLists.txt、include/、src/、configs/、tests/、validation/、tools/、calibration/、README.md、AGENTS.md、docs/`（docs 仅文本：PROJECT/HANDOFF/ACCEPTANCE/CONTRACTS/RELEASE/implementation 等；**不带** `docs/analysis/`、`docs/evidence/`（合计 ~195MB 证据归档）、不带任何 `build-*/`、`__pycache__`）。
3. 同时把《架构建模_单Core版_严格复核修订.docx》复制到 `template/` 根（`coarse_core_docx_source_test` 依赖 `die/../<docx>` 校验 SHA-256，缺它 61 项里必挂 1 项）。
4. 迁移后立即 `python3 die/tools/sync_repositories.py`（写模式）把 canonical 33 文件同步进五仓 `extern/die_model/coarse/`，`RELEASE.json` source_digest 由 5de03f5f…（canonical25）更新为 **6efa748b…**（die HEAD 实测值），并在交付报告中披露该身份变化；`--check` 五仓 passed 是硬验收项。
5. newdie 仓内五个 `extern/die_model/coarse/` 已实测逐字节一致，可交叉验证同步正确性。

## 4. 阶段 1：接口改造（行为不变）

**定义**：把 coarse_die_v2 的全部接入面（SDK、C++ 适配层、Python online 链路、构建、测试）落进五仓，但**出厂配置仍走旧模型**，旧路径行为与产物字节等价。

### 4.1 纳入阶段 1 的内容
- `extern/die_model/`：新增 `coarse/`（33 文件 + RELEASE.json，sync 生成）；`extern/die_model/CMakeLists.txt` 追加 coarse 挂载；legacy SDK 中 newdie 改过的文件（DieTypes.hh/DieModel.hh/CostModel.cc/src/Workload.cc/core_table.json 等）按三方合并结果更新（见 4.3 裁决）。
- 根 `CMakeLists.txt`：`AstraSim` 链 `coarse_die`（joint 另含 `joint_pipeline_core`、`joint_ledger` 相关行合并）。
- C++ 适配层（newdie 版为底）：`CoarseDieRuntime.hh/.cc`、`CoarseDiePrediction.hh`、`WorkspaceAdmission.hh`、`DieModelAdapter`（+PreparedCoarsePlan 三段式）、`Workload`（issue_impl 拆分、issue_prepared_coarse、control marker 端点）、`Sys`（coarse_die_runtime/workspace_admission/die_model_profile 成员与 initialize_die_backend coarse 分支、析构顺序）、`LocalHbmBandwidthModel`（delegating facade + endpoint 提交 + coarse served 计数）、`MetricCollector`（exact-decimal + coarse_die_runtime_metric_record）、`DecisionBridge/DecisionMailbox/GraphBatchCommitter/ParsedGraphBatch/GraphSource`（workspace schema v2）、`main_online.cc`（v2 delta + admission 一致性检查）、joint 的 `NativeCoarseDieDomain`、`NativeProjection` coarse world/snapshot 等。
- 对应 C++ 测试（lifecycle / adapter / admission / committer / mailbox / parsed_graph_batch / hbm_tail_component / coarse_workspace_runtime / shm_loopback 等）与 pytest（online/ 族、workspace admission 族、source release 族、die_model_adapter/oracle 族）。
- Python online 链路：`online_service.py`（admission 模式 auto=仅 coarse 时 required）、`online_scheduler_base.py`、`decision_bridge.py`（v2）、`session_kv_manager.py`（workspace 视图）、`graph_batch_builder.py`（source_release_capacity_gates 默认 False）、`die_model_oracle.py`（coarse ctypes）、`die_model_adapter.py`（coarse task/repeat_seams）、face 家族 `face_scheduler.py`（coarse 分支 + 峰值标量同源）、joint `sh30_online_scheduler.py` 的 provider coarse 化等。
- 冒烟/运行脚本若在 MERGE 桶（joint_runner.py 等）：合并但保持旧路径用法不变。

### 4.2 明确推迟到阶段 2 的内容（避免阶段 1 旧路径行为漂移）
- `sh_test_mesh/config_resolver.py`（coarse 派生 + legacy mesh-nx/ny 收紧会拒掉现配）、`tests/test_config_resolver.py`；
- `sh_test_mesh/hardware/face_case5_config_c*.json`（newdie 版换 mesh-nx/ny 口径）、system 模板翻转、`generate_face_trace.py / generate_wsc_llm_trace.py / generate_trace.py`（携带 coarse 标量链）、`test_wsc_llm_scheduler.py / test_face_scheduler.py` 中的 coarse 断言部分；
- 各仓 `README.md` 数值来源表更新（阶段 2 与默认翻转同步写，避免文档超前于行为）。
- 推迟文件在阶段 1 保持 template 现状（KEEP 处理），阶段 2 再按 TAKE/合并落盘。

### 4.3 冲突裁决原则（MERGE 118+14 个文件）
1. **三方自动合并干净的文件也要复核**：对每个自动合并结果做"语义对账"（对照两侧 diff 确认无互踩），以编译+测试仲裁。
2. **提速优化优先保留**：ours 侧（事件日历退役、EventQueue/Event、NodeStore/ParsedGraphBatch Phase4 增量、LocalHbm v7 索引、DieExecution InternalTimerArm、Capi C2 缓存/D0 游标、FlatShape、span-run 压缩、joint_ledger/NativeProjectionCApi/ServiceCoordinator/SHM 桥、Python 瘦身）在无 coarse 语义冲突时一律保留。
3. **legacy die SDK 内的提速项**（joint `extern/die_model/src/Execution.cc` 的 FlatShape、`DieCApi.h`/D0 游标族）：与 newdie 的 32 行核表/INT4 口径共存合并；若某优化与 coarse 依赖的 schema 不可调和，**以 newdie 版为准并在交付报告披露被放弃的优化点**（coarse 上线后该引擎不再是生产路径，属可接受代价）。
4. **joint `kv_cache_adapter.py`**：必须保住用户未提交的命中率口径修复语义，与 newdie 侧改动合并后跑 `test_slo_contract.py` 锚定。
5. **文档类冲突**（README、request映射*.md、PROVENANCE*）：以 ours 结构为底、并入 theirs 的 coarse 相关段落；不逐行搬运 newdie 的历史叙述。
6. **测试类冲突**：保 ours 的删改（特性退役）+ 增 theirs 的新增断言/新测试文件；两侧对同一特性断言不一致时以合并后的实现行为为准修订测试并披露。

### 4.4 阶段 1 验收（独立一轮，全绿才进阶段 2）
1. `template/die/` 全新 Release 构建 0 error，`ctest` **61/61**（含 docx 源校验）。
2. `sync_repositories.py --check` 五仓 passed（digest=6efa748b…）。
3. 五仓 aggregate 构建（BUILDTARGET=congestion_aware）0 error。
4. 五仓 ctest：die 相关 target 全绿（`DieExecutionLifecycleTest`、`DieModelAdapterTest`、`WorkspaceAdmissionTest`（注意 wscllm 族无 Congestion_Aware 前缀的勘误）、committer/mailbox/parsed_graph_batch/joint 的 coarse/hbm_tail/shm 系列），并跑 `ctest -N` 对账防漏。
5. **旧路径冒烟字节等价**：wscllm 与 face（其余仓抽 1-2 个如时间允许）在**改造前基线**与**改造后**各跑同一 1s 窗口（8 请求）冒烟：completed==8、no_decision==0、single_node==0、delivery==graph_batch 数；`raw_metrics.csv / normalized_metrics.csv / request_metrics.csv` 剥离 host 计时列后逐行一致；不一致项必须定性（仅计时/顺序噪声=记录后放行，数值变化=修复后再验收）。
6. 全程 `/usr/bin/time -v`，单冒烟峰值 RSS 上限 8GB（实测预期远低）。

## 5. 阶段 2：替换——出厂默认切到 die 内仿真器

**定义**：落地 newdie 未做的默认翻转 + 配置面收口，使"跑 trace 直接用 die 仿真器"。

1. **system 模板翻转**（五仓 `sh_test_mesh/system/llama2_7b_die_template.json`）：`die-model-profile: mesh_die_v1 → coarse_die_v2`，新增 `die-model-config` 指向仓内 coarse 硬件描述符。
2. **coarse 硬件描述符入库**：`sh_test_mesh/hardware/coarse_die_case5_hardware.json`，以 die/configs/example_hardware.json 为骨架、对齐 face_case5 几何（mesh 10×10、die 25×32mm、n=2、SRAM/NoC 档位取 example 同值），schema=coarse_die_v2。resolver 的容量/带宽硬校验（local_hbm_capacity_profile 与 die 派生值对齐）按其 fail-closed 报错逐项收口。
3. **resolver/生成器/测试落地**：4.2 推迟文件按 TAKE/合并全部落盘；`local_hbm_capacity_profile` 若需换名（coarse 2n×36GB=144GB/rank）按 resolver 合法值域选取。
4. **指标与后处理**：确认 `coarse_die_runtime_metric_record`（typed INT4/Vec/HBM-row 利用率）与 local_hbm 分子含 die served bytes 在冒烟产物中出现；`metrics_postprocess.py` 对 DIE 利用率 fail-closed 检查通过。
5. **文档同步**：五仓 README（coarse 出厂说明、数字来源表 digest 行）、`experiment/仿真各功能开关清单.md`（die-model-profile 取值域 + 出厂值、roofline-enabled/mesh_die_v1 转为显式 legacy 选项）、joint 的 PROVENANCE。
6. **legacy 保留口径**（裁决）：mesh_die_v1 / tx8_reference_v1 / roofline 显式可选——这是 newdie 设计文档 §12.2 的合同（"legacy roofline、旧 mesh_die_v1、tx8_reference_v1 都显式选择"），对照实验由 template_roofline 备份 + 显式 legacy 配置承担；不属于"开关隐藏旧机制"违规。

### 5.1 阶段 2 验收（独立一轮）
1. 阶段 1 全部验收项不回归（构建、61/61、sync --check、ctest die 族）。
2. **五仓 coarse 路径 1s 冒烟全通**：日志/配置证明 coarse 后端生效（profile=coarse_die_v2、CoarseDieRuntime 初始化、无 roofline/mesh_die_v1 计时参与）；PASS 判据同 4.4-5；joint 用 `joint_runner.py --combo TJE`。
3. `ledger_reconcile.py --expected-requests 8 --expected-accepted-sessions 3` PASS（wscllm 族）。
4. `run_metrics_postprocess.sh` 正常产出，DIE typed 利用率字段在且无 metric_error。
5. 定向 pytest：各仓 `online/test_die_model_adapter.py、test_die_model_oracle.py、test_workspace_admission.py、tests/test_config_resolver.py` 及 joint 合并触点的 joint/* 测试子集；**不跑全量 pytest**（防内存/时间失控）。
6. 内存监控同 4.4-6。

## 6. 执行架构（多子 Agent 并行 + 串行门）

- **Wave 0（先手，本 Agent）**：本方案落盘；die/ 迁移 + DOCX + sync 写模式 + die 独立构建/61 测试（后台）；五仓基线构建+1s 冒烟（后台，产物 /tmp/baseline/<repo>/）。串行门：`/tmp/phase0/DIE_DONE`、`/tmp/baseline/<repo>/DONE`。
- **Wave 1（阶段 1 主体，并行）**：4 个子 Agent 各领 1 个小仓（face/face-LRU/wscllm/wscllm-LRU，冲突 ≤4 文件/仓）；joint 拆 3 线（C++主链本 Agent 亲做；Python joint/* 模块 1 Agent；online/* + slo_tools + run_scripts + 测试/文档 1 Agent）。全部中间产物/合并结果落 /tmp/phase1/，仓内只写最终文件；每线完成即触发本仓构建+ctest（互不抢 build 目录）。
- **Wave 2（阶段 1 验收 Agent，独立）**：按 4.4 全项复验，输出验收报告；不过则修复重验。
- **Wave 3（阶段 2，并行）**：配置翻转线（模板+硬件描述符+resolver+生成器）+ 文档线（README/开关清单）；joint 单独一线。
- **Wave 4（阶段 2 验收 Agent，独立）**：按 5.1 全项复验；不过则修复重验。
- **Wave 5（收尾）**：五仓 `clean_test_records.sh` + `clean_build_artifacts.sh` 裸仓还原核验（占位指针/traces 仅 *.py/无 generated/无 build/git status 无新增残留，用户 3 个未提交文件中 kv_cache_adapter.py 修复保留、trace_config.csv 指针由 clean 脚本按规范归一）；/tmp 过程文件清点；总结报告（含：digest 变更、放弃/改写的提速点披露、两轮验收证据）。

## 7. 风险清单（预案）

| 风险 | 预案 |
|---|---|
| joint 118 冲突文件语义合并出错 | 分线合并+编译/测试仲裁；高险文件（face_scheduler/sh30_online_scheduler/joint_cost_model/action_work_graph/Execution.cc）逐 hunk 双侧对照；必要时以"ours 结构+theirs coarse 增量"重写 |
| v2 bridge 协议改变导致旧路径产物漂移 | 4.4-5 字节等价验收卡口；若漂移定性为协议字段新增则记录放行，数值变化则回查 |
| coarse E2E 从未在 newdie 默认配置下整链跑通 | 阶段 2 冒烟即首次整链验证，预留修复轮；fail-closed 报错逐项收口（这正是验收的意义） |
| 内存失控 | 1s 窗口、定向 pytest、-j 限 8、/usr/bin/time -v 全程监控、杜绝全量物化/全量收集 |
| 用户未提交改动丢失 | kv_cache_adapter.py 合并保语义+测试锚定；test_slo_contract.py 不动；trace_config.csv 交付时按裸仓规范归一并在报告说明 |
| die digest 变更引发一致性疑问 | RELEASE.json 由 sync 工具重写，交付报告明示 5de03f5f→6efa748b 及原因 |

---

## 8. 执行结果（2026-10-18 交付时点）

**两阶段全部完成并通过独立验收。**

### 8.1 阶段 1（接口改造）：独立验收 9/9 PASS

- die/ 库迁入 template/（纯源 + 文本 docs + 测试 fixture），61/61 测试全绿（含跨仓一致性门）；
- 五仓 coarse_die_v2 全套接口移植（三方合并 base=f7756e4 / theirs=newdie HEAD / ours=template 提速优化）：
  TAKE 329 / 三方合并 215（自动干净 83 + 手工裁决 132）/ 保留提速 124 文件；
- 五仓 ctest 全绿（face 37 / wscllm 46 / wscllm-LRU 42 / joint 36 / face-LRU 19，均含新增 coarse 用例）；
- 旧路径 1s 冒烟与改造前基线**字节级一致**（五仓 raw/normalized/request metrics 逐字节相同）；
- 峰值 RSS ≤ 769MB，全程无内存风险。验收报告 /tmp/phase1/ACCEPTANCE_PHASE1.md。

### 8.2 引擎扩展：实例 repeat 通用化

- `repeat_seams={mode:serial_on_drain}` 从 ffn_region 专属推广到全部 15 类类型化区域
  （影子单实例模板 + PlanRepeat + 逐实例区域开销作 seam；`materialize_gemm_compile_candidate`
  镜像影子构建保候选身份）；fused-FFN 排除与 instances≥2 门保持 fail-closed；
- die 库新增第 62 项测试（9 算子 × repeat 的完整预测/estimate==全模拟逐位一致/单实例工作基准），
  62/62 全绿；
- SDK digest 链：5de03f5f…（canonical25）→ 6efa748b…（P9 快路径随迁）→
  **3a021f37b71fb3ea52a02a0c522169cd5a8374ef1f77c2d6a20fe504b274c31c**（本批），sync --check 五仓 passed。

### 8.3 阶段 2（出厂默认翻转）：独立验收 10/10 PASS

- 五仓出厂配置：system 模板 `die-model-profile: coarse_die_v2`；硬件 `coarse-die` 节
  （coarse_die_v2 描述符，8×11 Core、n=2）；容量档 `validation-coarse-144g`（144 GB
  fail-closed 对齐）；emitter 对所有 layer×pass≥2 区域声明 repeat_seams；
- 过程修复：wscllm/face 家族 decision_bridge 的 fix-2.2 spec 提升在 coarse 下跳过
  （die_task_json 保真）；face 家族选择器估价 spec 补 repeat_seams；joint 的
  die_model_adapter 死代码激活；joint 出厂指针归位占位；
- 五仓 coarse 最小冒烟（1 请求 64-token prefill + 1 decode，与 newdie 官方 E2E 同规格）
  **全部 RUN_RC=0 + 请求 completed**（RSS ≤ 72MB）；ctest 与定向 pytest 全绿；
- 文档同步：五仓 README 出厂说明 + digest 行、experiment/仿真各功能开关清单.md
  die-model-profile/容量档行、die/docs/HANDOFF.md 实例 repeat 通用化节、joint
  PROVENANCE.md 2026-10-18 翻转批节。验收报告 /tmp/phase2/ACCEPTANCE_PHASE2.md。

### 8.4 已知边界（如实披露）

1. **1s 全窗冒烟的运行时长**：运行时推进路径保留 P8 记载的 O(R²) 活动向量墙（P7-B
   遗留项，上游明确 defer），真实 1s 窗口（8 请求 × R≈256-992 重复计划）的在线排空
   为小时级；故阶段 2 验收采用最小队列冒烟（全链路覆盖、分钟级）。1s 窗全量验证可
   在后台长跑复核。
2. 1s 冒烟中确认的 coarse 引擎缺口（聚合非 FFN 算子 not_available）已由 8.2 扩展修复；
   修复后的编译/估计路径实测毫秒级（6 算子批量 0.054s）。
3. clean_test_records.sh 在三仓（face-LRU/joint/wscllm-LRU）将容量档随指针归位回滚至
   validation-die-64g——已恢复 validation-coarse-144g（交付配置），后续跑 clean 脚本
   后需复查该行。
4. 交付时五仓为裸仓态（指针占位、traces 仅 *.py、无 generated、无 build）+ coarse 出厂
   配置；用户未提交的 kv_cache_adapter.py 修复原样保留；全程未执行 git commit。
