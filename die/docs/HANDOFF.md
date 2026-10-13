# Codex Cloud 开发检查点与接力状态

记录日期：2026-10-13（Asia/Shanghai）。用户已明确要求停止新任务、完成 GitHub 交接；本 Session 在保存当前最小实现与验证后停止。

## Current Status

**代码交接检查点齐全；工程精度/全量五仓验收未完成。** 最终公共库默认完整构建与 **17/17 CTest 通过**；当前最小任务的 GEMM 尾块预算与 typed task 校验已实现并测试，部分研究也已保存。独立 compiled K-wave oracle 仍 **failed**，没有可以用于已验证架构排序的粒度证书；启动/重复实例、向量驻留/转换、workspace 与部分五仓路径未验收。

公共库 `ee8a48d5234862efbdffd64465648816091a30e3` 与五仓集成 `ac2ea4188359fa7760e9929c78c2df7cb92c5703` 已进入 main。最终源码提交 `37b13ac928d43b055d0b31ae28caf45d11763659`、测试、审计与本文件保存到交接 PR [#3](https://github.com/cs-sunH/template/pull/3)。**只有该 PR 合入 main 且远端 SHA 经核对后，另一个账号才可从最新 main 安全接续开发**；这不等于工程验收通过。GitHub 实际 PR/SHA 见 Handoff Notes 和 Git 元数据。

## Integration Branch

共享集成分支：`main`；当前工作分支：`codex/die-continue-20261013`。不 force-push，不覆盖他人提交。GitHub 同步的实际 commit/PR/合并结果见 Handoff Notes。

## Last Verified Base

- 本轮开始 fetch 并切换至 `81a11752c940690427b6f3b85d42572464929a9a`；它包含上一轮已合并代码检查点和移除过期停工指令的文档修改。
- 上轮实现检查点 `ac6fc4cd4b9e9577297ead16fb74e61b1d691b0b` 经 PR [#1](https://github.com/cs-sunH/template/pull/1) 合入 `6206491620ea841cb976f09585248d3e84223510`。上轮源码保存/审计证据保留在 [evidence/handoff-20261013/](evidence/handoff-20261013/)，不冒充本轮回归。
- 架构资料源版本 `9fdcad1f8a1640d3e91558e3b2d11c55c66b53c8`，Core64 表与根设计文件未改。
- 本轮已推送的小修复 `7e7d8a4` 是 benchmark target 编译定义；完整当前改动的共享状态须核对 Handoff Notes 和远端 Git 元数据。

## Completed

本轮公共源的具体改动与验证范围：

- benchmark 与常规测试使用一致的源码目录宏；SRAM 字节派生恢复精确整数类型，source/Core64 与几何约束没有放宽。
- 编译 credit_delta 允许有符号有限值；物理工作仍要求非负。Runtime LP 以相对公平变量缩放，修复 TB/s 与信用系数尺度差异导致可执行活动被压成零速率的问题；不放宽数值容差。
- 首重叠 epoch 保留首载入信用，末写回消耗末输出信用；这只是修复宏框架局部错误，尚未证明 K-wave 生命周期。
- GEMM 区分原始输入、转换缓存、尺度、PE 操作数读取、INT32 partial、FP32 accumulator、输出转换及 L/C/S SRAM；WR 复用、分块 INT4 奇数打包和层/权重 pass 倍数纳入独立计数测试。
- QKV 后的 RoPE 成为实际串行 Vec primitive，`none` 不收费，V 不旋转；materialized SwiGLU/GELU 使用真实中间 dtype，去掉重复写回；融合 SwiGLU 重算、down 累加、原始/转换缓存与独立 pending outputs 有显式账本。GELU 显式 fused 尚不支持。
- Attention 以实际 query tile/head cohort 映射，区分 causal source pairs 与执行 slots，full materialized 使用完整矩形；compact KV progression、Q 重读、GQA、独立 QK/AV、score/probability HBM、INT4 奇数块和 SRAM 均由小离散计数测试核对。最终测试状态见 Tests。
- compound region 的 `logical_tensor_bytes` 计数学边界输入、参数/权重和最终输出；内部 materialization/recompute 分别进入物理流量与 workspace。QKV/RoPE 仅增加 cos/sin 逻辑参数，FFN 融合策略不改变源边界逻辑量。
- 保守容量账本预留双输入/预取权重和双 pending output，累加器不与同时存储的 FP32 输出别名；fused 每核 64 B metadata 是显式工程预留，没有由 TX8 拟合。
- endpoint Plan 使用完整 canonical payload hash，ABI 生命周期测试覆盖部分服务、取消、复用、fork/import 与失败导入原子性。默认两排奇数字节由 row0 多一个，标 `allocation_evidence=uniform_default`；显式 fraction 保留原 floor/remainder 约定；无 evidence 的旧 Plan 保留冻结旧分配。
- 候选排名复用一次只读 Hardware，使用同一个真实 Runtime 的 typed terminal-finish 驱动，免去候选进度/报告 JSON；不改公开 estimate 报告与校验语义。单活动有经过等价分拆和阻塞信用测试的封闭分配路径。
- 累计 typed served/category counters 已补齐并测试：COMP、comm_send/recv、restore、pool、die 与 unknown 历史分别归账；active/cancel/retire 后的有效服务不会丢失。版本迁移与验证见 Tests/Important Decisions。

### 最终最小任务：尾块预算与 typed task

- `src/GemmTailBudget.hh` / `src/Mapping.cc`：OS 按输出 tile 的 Core cohort、WR 按列组与 M 循环，闭式累加每 cohort 最慢核的 PE+Vec 串行计算预算，至多四类 full/tail 成本，使用 checked uint64 索引/128 位乘积；不创建 Core/tile/wave 数组。layers×passes 只乘一次；活动 work 已是总量，repeat_count 不重复乘。
- `tests/gemm_tail_budget_test.cc`：对 144 个真正 `enumerate_mappings` 候选独立逐 tile/cohort 核对，覆盖 1/2/4 Core、OS/WR、M/N/K tail、重复次数及 strict FP32；计算 gate、Activity 与 summary PE/Vec 账本一致。helper scratch 的 518,400 例与 sanitizer 不冒充生产库 sanitizer。
- `src/Compiler.cc` / `tests/task_schema_test.cc`：直接 v2 task 各算子要求真实维度；shape aliases 统一、冲突拒绝；局部 heads/FFN/vocab 与 TP 元数据一致，不再次除 TP；显式 rank=0 正确接受；未知字段/算子、非法维度/variant/profile alias、冲突 span alias 明确拒绝。standalone row 算子可从所有 span 的 q_tokens×count 推导 M，与显式 M 必须一致。legacy v1 转换路径保持既有约定，example_tasks 全部仍编译。
- **范围限制**：只修累计 slowest compute budget；append_epoch 宏仍将一轮全部 K 聚合，PRIVATE SRAM/LOCAL tail、WR 列组围栏和真实有限缓冲生命周期仍未闭环。κ 仍截断，strict FP32→INT4 output 仍收费 1；预算故意与当前 ledger 一致，未仅修改时间掩盖计数问题。
- 五仓重新发布最新 **19 文件**（新增私有预算头），digest `d9beb83c59d5cbeec90644f07b773787100facf427d9e4801508abe239700a0b`；最终接入复核见 Tests。前一 18 文件 digest 保留在 integration-20261013 作为历史证据，不与最终版本混淆。

### TX8 超参数校准

根目录工作簿 SHA-256：`eba181c80a2d51bd5a604bad6928610ea5039824b4ec4d5d9bd023ee4bda52ee`。原始九条测量已移除，没有可审计日志、预热/重复分布或独立留出集。

- `计时组13行!B2/D2` 的 246 μs 是 `reported_anchor`，按说明解释为完整 RMSNorm；第二次 Norm 是锚点复用。
- `B13/D13` 的 3363 μs 是 `conditional` 的整个 FFN 重解释，不是独立 SiLU/PE/Vec 测量。
- 其他计时组是派生/工程重建值，实测拟合权重为 0；19 个数学节点不是 19 个独立样本。

| reference 参数/结果 | 数值 | 限定含义 |
|---|---:|---|
| `eta_compute_reference` | `0.9079551923877489` | 接受 3363 μs FFN 解释、固定 128T/1T 峰值、矩阵/向量效率绑定的 TX8 BF16/FP32 参考。 |
| `eta_vector_memory_reference` | `0.3411668292682927` | 固定 Norm byte 假设下由 246 μs 反推向量 DDR 服务。 |
| 向量 DDR 有效吞吐 | 约 `68.23336585365854e9 B/s` | 名义 DDR 200 GB/s。 |
| 矩阵 DDR 有效吞吐 | `140e9 B/s` | 名义峰值的 70%，工程设定而非独立拟合。 |
| 13 组参考复算总计 | `11897.602619368332 μs` | B16 原 XML cache 为 `11897.602619368336 μs`。 |
| 最大单组参考误差 | 约 `4.55e-13 μs` | 源公式复现误差，不是预测准确率。 |

工作簿唯一公式为 `B16=SUM(B2:B14)`，B15 是硬编码总计。AV→`av_pv` 只映射一次；Attention、FFN、BLOCK 小计不重复相加。

目标参数仍是工程先验：`eta_pe=eta_vec=0.90`，`eta_hbm_matrix=eta_hbm_vector=0.70`，`eta_sram=1.0`，`eta_local=eta_mesh=eta_inject=0.70`，`rho_sram=0.90`，启动 1000 ns，setup 0 ns，quant 4 eq-op/元素，acc convert 2 eq-op/结果。**没有把 TX8 reference 效率迁成目标 INT4/HBM 校准。** 有数值不等于对应消费者已闭环，例如 `new_plan().launch_ns` 当前置 0。

### 五仓集成

五仓为 `astra-sim-{wscllm,wscllm-LRU,face,face-LRU,joint}`。上轮 Sys/coarse facade、adapter、HBM endpoint、typed graph、config resolver 原型全部保留。

本轮已恢复 JOINT adapter 被原移植删除的 timing/window/pipeline/readiness 旧接口，保留 legacy die_model，并发布五仓完全一致的 18 文件公共源，digest `f3e07793a2a6ee7a68d751c242287a0616470402d10284cba97b36c98449f9b7`；`sync_repositories.py --check` 通过。跨目录修改来自五仓集成及原功能兼容要求。

- 五仓当前 legacy/coarse C ABI targets 构建成功，配置诊断兼容旧消息，JOINT 不可达库测试的环境隔离已修正。五仓根 CMake 直接链接 coarse；Chakra `.proto` 自动生成至 build tree，clean checkout 不依赖未跟踪的生成源码。五仓实际 AstraSim 核心库与 Online target 均构建成功。
- 修正 FACE/JOINT HBM 清理语法、FACE 不存在的 restore/pool enum 移植，以及 FACE-LRU endpoint 分支被放错函数的问题；缺失 `<map>` 头文件已补齐。
- coarse facade 批完成回调按 handle 顺序全部排空，即使某回调抛异常也清理其余终态、恢复 dispatch 状态，再重抛首个异常；只读 `counters_at(now)` 在私有 Runtime 副本推进，不发外部回调或改变实跑。
- 修正五仓 coarse 显式 contention=1 被旧零带宽 guard 误关闭的问题，显式 contention=0 仍拒绝。四仓 Sys 直到 Workload 取消端点任务后才从全局时钟表移除，避免销毁中的时钟 panic。
- 五仓完成利用率读取实际 typed served INT4/Vec/HBM；wscllm MetricCollector 输出 active/cancelled/retired 的 rank 生命周期分类账本，DIE 与 COMM 的 HBM 分子只计一次，分母使用硬件原始双排峰值。旧统计 key 保留；旧队列 owner peak/redistribution 不能冒充 coarse 历史峰值，覆盖边界写入记录。
- wscllm 真正 typed COMP+COMM fixture 验证部分服务取消、共享 HBM、无取消终态、过期事件 drain、完成 join/一次完成/GPU 槽释放、退休计数结算。取消前缀使用原 microbenchmark 节点历史策略；正常完成路径另用 service 策略，未放宽 service 禁止 live GPU 结算的保护。
- JOINT 适配器 fixture 覆盖 callback-throw 清理、只读 partial counter 采样与取消；coarse+冻结 legacy pipeline contract 仍 fail-closed，不冒充已支持 coarse live projection。

选定接入 CTest 共 17/17 通过（wscllm、wscllm-LRU、FACE、FACE-LRU 各 3，JOINT 5）；FACE-LRU 先发现未注册测试，补齐注册后真实跑到 3/3，不能采用空测试 exit=0。证据见 [integration-20261013/](evidence/integration-20261013/)。这些不是全量五仓业务验收。

Portable Protobuf 3.21.12 已安装到忽略的 `work/deps/protobuf/`；源码固定 protobuf v21.12 tar SHA-256 `22fdaf641b31655d4b2297f9981fa5203b2866f8332d3c6333f6b0107bb320de`。wscllm aggregate 配置已成功，真正 AstraSim 编译仍待 vendor。接手环境须重新准备依赖，不假定 `work/` 已同步。

## Next Step

按优先级接续，先复用本轮源码与证据，不重复通过的 source/byte/ledger 回归或没有新测量的 TX8 拟合。

1. **P0：真实编译宏的 K-wave readiness/retire 与 busy union。** 入口 `Mapping.cc::append_epoch/add_gemm_epoch`、`Execution.cc`、`validation/compiled_gemm_oracle.py`。从已失败的 1-Core OS、65×65×4097、R=10/Tk=65 fixture 开始，修每个 K 输入完成才可消费、非末 K 不产出、末 K 产出到写完才释放输出槽、WR 列组切换围栏。需要有界类别/周期表示，不能把生产变成逐 wave 数组/事件；保留真实 bytes/ops 与正速率 busy 定义。先 standalone，再零/50ms offset 并发，再 multi-Core/tail/WR。按原独占 3%、并发及 busy 5% 门槛选最小有效粒度；E8/16 实现之前保持 unavailable/not_run，不改门槛。
2. **P1：补齐成本与输入语义，依赖第一项的真实生命周期。** (a) fractional κ 与 strict FP32→INT4 同时修 ledger、gate、Plan/Runtime/ABI，见已保存的 fractional audit；不能 ceil 或只修时间。默认整数 exact/hash、旧 Plan/snapshot 和 C ABI v2 layout 必须保留，分数的 fixed-layout 接口需版本化/明确拒绝。(b) launch=1000ns/setup=0 对真实区域/层/pass 实例收费，不能只在最前面 lump 总启动。(c) 按 consumer-Core 首次读取、完整参数切片与其余 live buffers 同时 fit 的规则实现 Norm/RoPE 参数复用、长行两遍输入与真实 storage conversion；source math/special-function/conversion 口径分开。(d) task residency/readiness/math 尚未兑现；直接 v2 compact run 的 batch>1 与递增 KV 序列当前会把 count×batch flatten，须先独立复现再改保持序列与 batch 重复的解析语义；TP 局部区间尚无明确实现，不把已修维度一致性当作区间支持。
3. **P1：五仓业务准入与专属路径，依赖可审核的 Plan workspace/lifecycle。** 先 wscllm 把 workspace 增量 reservation 放到 NodeStore take / GPU occupy 前，补取消、异常、完成时释放，再逐仓 KV/LRU/restore/pool/readiness/网络 join 与 owner 历史统计；用 `five-repo-workspace-audit.md`，保留原 scheduler/KV/权重容量，不双收费。当前 endpoint owner/request identity 与历史 coarse owner peak 仍需细化。JOINT 继续真实 live import/fork、外部网络/pool join 与 overlay 隔离，实现并测试之前保留 coarse+冻结旧 pipeline 的 fail-closed guard，不借迁移改 selector 策略。
4. **P2：性能和发布验收。** 模型正确后测 cold compile、真正的 cache hit、J≤32 LP、事件/CPU/RSS、legacy Roofline/mesh/coarse 三方与五仓同图/完整在线 workload 开销。30 样本探针只是方向数据。每次公共源变化统一发布五仓 digest，复核实际受影响目标；通过有意义阶段再更新文档/GitHub。没有新的可审计硬件测量时继续冻结工程先验，目标硬件留出保持 not_available。

上述优先级不是自动停工阶段上限。下一账号收到“继续”后可据此自主推进；本 Session 按用户交接指令停止。

## Important Decisions

1. 根《架构建模新.md》控制硬件输入/派生，根性能设计控制执行假设；旧模型仅用于明确 legacy profile。`die/` 是唯一公共源，五仓只使用内容校验的固定发布。
2. 13 个物理域不变；INT4/Vec 不合并峰值，逻辑 bytes、physical payload、D/eta 与私有 theta 分开。L/C/S work 已为整个 epoch 总量，repeat_count 是审计 metadata，不再乘一次。
3. signed credit 按确定性 LP/词典序 max-min；公平 owner 为 `(node_id,execution_id,generation)`，request_id 只追踪。取消只记录已经服务的量。
4. 缓冲峰值是保守容量检查；双槽预留不能证明流体 macro 的 tile-ready/retire 正确，也不能替代最慢核、精确路由及独立粒度误差证明。
5. Runtime 快照新增累计账本需要版本化。新导出计划采用 snapshot v3，C ABI 保持 v2；旧 v2 不能恢复已销毁 execution 的 typed/category 历史，迁移必须标 history incomplete，将旧全局 HBM 保存为 unclassified，不静默把未知视为零。新 v3 需要完整字段、守恒校验及原子导入。
6. 手工 grouped-wave Runtime 驱动只能证明其合成 fixture 的语义/误差；`discrete_oracle.py` 已将结果归入 `runtime_grouped_*`，编译模板 `granularity_validation` 保持 not_run。真实 compiled oracle 的结果另报。
7. 缓存 4096 条/64MiB 是 retained-allocation 上限，不是 host RSS。known_local_world 不自动复制外部 network/pool；JOINT guard 只在真正的 live import/fork/join 实现并测试后移除。
8. TX8 条件复现、目标校准、模型粒度、五仓工程、目标硬件留出是独立证据。无新可审计测量，不升级两个 reference 参数的解释。

当前 §7.2 明确使用 active_cores 分母的保守 COMPUTE_GATE 占用，§17 把跨 kernel 独立 PE/Vec 并发列为扩展；保持该指定 profile，不把实验 `/Ncore` 解释直接替换进去。GEMM 按 §6.1/§13.1 保留最多 128 个候选；降序首可行 tk 只适用于 §6.5 融合 FFN。tail slowest-cohort 与向量参数驻留仍待修，scratch 推导不是编译器证书。

## Files to Continue

- 公共库：`src/{Compiler,Mapping,Execution}.cc`、`src/GemmTailBudget.hh`、`include/coarse_die/{Plan,Execution}.hh`、`CMakeLists.txt`。
- 独立计数：`tests/{gemm,operator,fused_ffn,attention}_accounting_test.cc`；Runtime：credit_scale/counters/endpoint_allocation/estimate_driver/capi_lifecycle。
- 真编译 oracle：新增 `validation/compiled_gemm_oracle.py`、`tests/compiled_gemm_oracle_driver.cc`（已保存并实际运行）；现合成 oracle `validation/discrete_oracle.py`。
- 集成：五仓 CMake、coarse facade/adapter、MetricCollector/HBM、workspace/Sys teardown tests；JOINT NativeProjection/Python scheduler guard。
- 校准：已有 `calibration/report.md`、reference configs 与 `tools/calibrate_tx8.py`；不把生成文件重新改写当作新证据。
- 最终新增测试：`tests/gemm_tail_budget_test.cc`、`tests/task_schema_test.cc`；待实现研究/公式/性能原始样本：`docs/analysis/handoff-20261013-final/`，先读其中 README 对历史审计与已解决项的区分。

## Tests

最终统一 configure、默认全部 targets 构建与 **17/17 CTest 通过**；含新 tail budget/schema、GEMM/Attention 独立计数、累计 counters、endpoint 分配和旧快照迁移。最终证据：[handoff-20261013-final/](evidence/handoff-20261013-final/)；先前证据 [continue-20261013/](evidence/continue-20261013/) 保留。

- source_golden、hardware_search、compile_execution、signed credit、operator/fused accounting、endpoint allocation（含旧默认奇字节冻结导入）、Runtime credit_scale、estimate driver、runtime_semantics、C ABI lifecycle 已运行通过；新增 counters 的 partial/cancel/retire、v2→v3 迁移、失败导入和分类守恒也已通过。
- 合成 grouped Runtime oracle 独立对照真实运行：E=5/8/16 均未满足全部门槛，最大完成误差约 9.72%/9.46%/6.97%，最大 HBM busy 相对误差约 3.39。对应 CTest 是 measurement driver，退出 0 不代表误差验收通过。
- 真实编译 GEMM oracle 已构建并运行：singleCore OS/strictFP32，M=N=65/K=4097，tm16/tn64/tk64，R=10、Tk=65、650 个真实消费 wave；独立工作账本全部守恒，但 E5 独占完成误差 3.827%、HBM busy 124.951%，50ms phase-offset 并发第一任务完成误差约 16.67%、HBM busy 158.87%、shared busy 478.94%，因此 **failed**。生产 E8/16 不可用，保持 not_run；不拟造 theta、不改门槛。完整报告见 evidence。
- 先前 30 样本 128³探针 cold/hot P95 约 2.23/2.11 ms；另保存 J=32 30 样本报告 cold 2.499ms/hot 3.552ms，均不是最终源码的正式全域验收，2ms 目标尚未满足。报告 passed 只代表运行成功；原始样本/二进制 hash 见 analysis/handoff-20261013-final。
- 五仓实际核心库/Online 构建与选定 17/17 CTest 已运行；wscllm 新夹具的逐项输出保存。配置 unittest discovery 初次共 74：49 通过、25 跳过；随后构建五仓当前 legacy/coarse C ABI 并显式选择其路径，补跑共 **83/83 通过、无跳过**（JOINT class-level skip 原来隐藏了多个方法）。构建不是全部 targets/全量业务回归；上轮 JOINT 28/28 CTest 与 72 tests+12 subtests 仍是旧二进制历史结果。
- 集成工具实际版本以 evidence/verification.json 为准（CMake 4.4.4、GCC 14.2.0、Protobuf 3.21.12）。父 aggregate mirror README 的三 subdirectory 布局，Release、congestion_aware、NETWORK_BACKEND_BUILD_AS_LIBRARY=ON；依赖/产物位于忽略的 work/，不依赖它们作为源码交付。
- 最终 19 文件发布后，五仓实际核心库/Online/coarse C ABI 已重新构建，选定 **17/17 CTest** 与显式选择当前 ABI 的 **83/83 配置测试**再通过；`sync_repositories.py --check` 通过。最终 source digest、逐仓计数与日志保存到 handoff-20261013-final，不只引用前一 18 文件二进制。
- TX8 source/cache 审计复用上轮已通过且工作簿未变的证据；目标硬件留出仍 not_available。

本环境可复用命令（工具路径须在新环境重新核实）：

```bash
work/tooling/cmake/data/bin/cmake -S die -B work/benchmark-fix -DCMAKE_BUILD_TYPE=Release
work/tooling/cmake/data/bin/cmake --build work/benchmark-fix -j2
work/tooling/cmake/data/bin/ctest --test-dir work/benchmark-fix --output-on-failure
python3 die/validation/compiled_gemm_oracle.py --driver work/benchmark-fix/coarse_compiled_gemm_oracle_driver --output work/benchmark-fix/compiled-gemm-oracle-report.json
python3 die/validation/discrete_oracle.py --runtime-driver work/benchmark-fix/coarse_oracle_granularity_test --output work/benchmark-fix/synthetic-oracle-report.json
```

完整源码/测试与阶段证据保存在三个本轮 evidence 目录；关键 scratch、数学测试、性能样本与未实施审计已保存到 `docs/analysis/handoff-20261013-final/`，不会仅留 work/。本轮生产库未执行 sanitizer；只有独立 scratch 数学 helper 做过 ASan/UBSan，不混淆范围。GCC __int128 pedantic 警告保留。

## Blocking Issues

- 编译宏的 input/output rounds 不能代表 R*Tk 真实有限缓冲生命周期；累计 tail 最慢核 compute budget 已修，但各宏内 tail/PRIVATE memory、WR 围栏、精确 full/tail routing 未闭环，真实编译 E5 fixture 已失败，E8/16 尚不可用，没有可用证书。
- launch/setup 尚未收费；layers/passes 工作量倍数已修，真实重复实例时序未修；fractional kappa 仍有整数截断，巨大有限 κ 的直接整数 cast 还须在转换前做 overflow 检查；向量缓存/转换与 partial/tail 分布须进一步验证。
- 五仓固定源/真实构建、wscllm typed MetricCollector/DIE HBM 总分子及首个 COMP+COMM/GPU 槽夹具已验证；其他仓 coarse 业务、workspace 增量准入/释放、KV/LRU/network/pool 与 JOINT live projection 尚未验收。
- TX8 不能辨识目标 INT4/Vec/HBM/NoC/SRAM/启动/转换/竞争，数据不足不是可通过修改拟合权重消除的问题。
- cache/LP/J<=32/事件/RSS/Roofline 正式开销证据尚缺；单小 shape 探针不能覆盖全参数域。

## Handoff Notes

用户已明确要求交接并停止。有效生产代码、测试、校准脚本、固定 vendor、失败 oracle 证据及关键未实施研究均保存；本 Session 不再启动新开发任务。`.github/workflows/die-handoff-check.yml` 包含真正的公共库默认 build/CTest；未获得 GitHub Actions 成功状态，不把本地测试冒充 CI。

本轮已在 main 保存 `7e7d8a4` benchmark 修复、`ee8a48d5234862efbdffd64465648816091a30e3` 公共库、`ac2ea4188359fa7760e9929c78c2df7cb92c5703` 五仓集成。最终最小任务、所有保存的审计与本文件通过交接 PR [#3](https://github.com/cs-sunH/template/pull/3) 合入流程，源码提交为 `37b13ac928d43b055d0b31ae28caf45d11763659`，后续仅补 PR 元数据。创建时 PR 为 open，采用 expected-head SHA 的 merge 防止误合其他更新；实际 merged 状态与 merge SHA 以 PR 元数据和 Git fetch/ancestry 为准，不 force-push、不覆盖远端提交。本文件所属完整 SHA 从 Git 元数据读取，避免编造自身 SHA。

**代码交接判定**：PR #3 显示 merged、远端 main 包含最终源码提交与本文件、19 文件 digest 一致时，本检查点 READY FOR CODE HANDOFF；工程精度/全量业务仍 NOT ACCEPTED。

接手流程：从 GitHub 最新 main 创建新分支；读取 AGENTS/PROJECT/本文件与两份根设计；核对工作区干净、main 包含本交接 PR 和最终 19 文件 digest。复用通过的证据，直接按 Next Step 的 P0 开始；公共构建能复核，真实 oracle 失败是待解决的模型问题。代码可安全继续开发的条件与“模型精度验收通过”分开，未通过的工程不能用于已验证架构排序。

本环境工具/依赖/产物在 work/，不是跨账号交付：新环境按标准方式安装 CMake/C++17/Protobuf；本轮 Protobuf 3.21.12、源码 v21.12 固定 SHA256 `22fdaf641b31655d4b2297f9981fa5203b2866f8332d3c6333f6b0107bb320de`。五仓 CONFIG 模式 configure **和 build** 均需保留 PATH、CMAKE_PREFIX_PATH 与 PROTOBUF_FROM_SOURCE=True；若曾切到缺少依赖的 MODULE 模式导致缓存 NOTFOUND，用正确环境重新 configure `-UProtobuf_*` 后重建，不改源码绕过依赖。所有生成 Chakra bindings 在 build tree，勿提交生成文件。
