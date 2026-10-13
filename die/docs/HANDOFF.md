# Codex Cloud 开发检查点与接力状态

记录日期：2026-10-13（Asia/Shanghai）。持续开发中；本文件记录可复核的实现、验证边界和 GitHub 状态，不表示用户已要求停工或交接。

## Current Status

**NOT READY FOR HANDOFF / 工程验收未完成**。默认构建阻塞已解除，公共库完整构建与 15/15 CTest 已通过，算子归账、运行时信用和快照形成新的验证检查点；真实编译模板粒度、启动/重复实例、五仓业务兼容仍未验收。

公共库检查点与本文件一起保存；五仓正在实施的适配修改需下一集成检查点继续保存。下面 Completed 与 Tests 只对明确标明的范围有效，不能当作完整性能模型通过证明。接手前仍须核对远端 main 和本文件所属 Git 提交。

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

本轮已恢复 JOINT adapter 被原移植删除的 timing/window/pipeline/readiness 旧接口，并补齐各仓根 CMake 的 coarse 链接，保留 legacy die_model。跨目录修改来自五仓集成及原功能兼容要求。公共源尚待固定版本 vendor；这些改动尚未通过五仓真实构建/业务回归，不能宣称 legacy 或 coarse 模式正常。

Portable Protobuf 3.21.12 已安装到忽略的 `work/deps/protobuf/`；源码固定 protobuf v21.12 tar SHA-256 `22fdaf641b31655d4b2297f9981fa5203b2866f8332d3c6333f6b0107bb320de`。wscllm aggregate 配置已成功，真正 AstraSim 编译仍待 vendor。接手环境须重新准备依赖，不假定 `work/` 已同步。

## Next Step

持续推进，不因此检查点结束停工：

1. 当前公共库统一 build/15 个 CTest 已通过；同步此可复核检查点至 main，再继续下面的模型与集成工作，不重复已通过的账本测试。
2. 运行真正 compile/enumerate 输出的 GEMM 与独立 K-wave 离散 oracle 对照，先覆盖 singleCore OS、tail、多个 K 与 output round、phase-offset 并发；当前 E=8/16 不可用时必须报告 unavailable，不能模拟一个未实现的升级。修 ready/retire 曲线、慢核 tail、WR 列组围栏后再申请粒度证据。
3. 补齐 launch/setup 对真实区域实例的延迟、重复实例时序、fractional kappa、向量参数驻留及转换；保持候选数/epoch 上限，并用独立 ledger 验证。
4. 用 `tools/sync_repositories.py` 发布全部五仓固定 digest，验证 source 一致，再以 wscllm COMP+COMM cancellation/join/typed metrics 为首个真实业务 fixture，随后逐仓验证 KV/LRU/pool/readiness/workspace 与 JOINT 专属路径。
5. 检查 cold compile、cache 查询、J<=32 LP、事件/CPU/RSS/Roofline 开销；有数据不足的目标参数继续冻结工程先验，已有 TX8 条件审计不重复无意义重跑。

## Important Decisions

1. 根《架构建模新.md》控制硬件输入/派生，根性能设计控制执行假设；旧模型仅用于明确 legacy profile。`die/` 是唯一公共源，五仓只使用内容校验的固定发布。
2. 13 个物理域不变；INT4/Vec 不合并峰值，逻辑 bytes、physical payload、D/eta 与私有 theta 分开。L/C/S work 已为整个 epoch 总量，repeat_count 是审计 metadata，不再乘一次。
3. signed credit 按确定性 LP/词典序 max-min；公平 owner 为 `(node_id,execution_id,generation)`，request_id 只追踪。取消只记录已经服务的量。
4. 缓冲峰值是保守容量检查；双槽预留不能证明流体 macro 的 tile-ready/retire 正确，也不能替代最慢核、精确路由及独立粒度误差证明。
5. Runtime 快照新增累计账本需要版本化。新导出计划采用 snapshot v3，C ABI 保持 v2；旧 v2 不能恢复已销毁 execution 的 typed/category 历史，迁移必须标 history incomplete，将旧全局 HBM 保存为 unclassified，不静默把未知视为零。新 v3 需要完整字段、守恒校验及原子导入。
6. 手工 grouped-wave Runtime 驱动只能证明其合成 fixture 的语义/误差；`discrete_oracle.py` 已将结果归入 `runtime_grouped_*`，编译模板 `granularity_validation` 保持 not_run。真实 compiled oracle 的结果另报。
7. 缓存 4096 条/64MiB 是 retained-allocation 上限，不是 host RSS。known_local_world 不自动复制外部 network/pool；JOINT guard 只在真正的 live import/fork/join 实现并测试后移除。
8. TX8 条件复现、目标校准、模型粒度、五仓工程、目标硬件留出是独立证据。无新可审计测量，不升级两个 reference 参数的解释。

## Files to Continue

- 公共库：`src/{Compiler,Mapping,Execution}.cc`、`include/coarse_die/{Plan,Execution}.hh`、`CMakeLists.txt`。
- 独立计数：`tests/{gemm,operator,fused_ffn,attention}_accounting_test.cc`；Runtime：credit_scale/counters/endpoint_allocation/estimate_driver/capi_lifecycle。
- 真编译 oracle：新增 `validation/compiled_gemm_oracle.py`、`tests/compiled_gemm_oracle_driver.cc`（已保存并实际运行）；现合成 oracle `validation/discrete_oracle.py`。
- 集成：五仓 CMake、coarse facade/adapter、MetricCollector/HBM、workspace/Sys teardown tests；JOINT NativeProjection/Python scheduler guard。
- 校准：已有 `calibration/report.md`、reference configs 与 `tools/calibrate_tx8.py`；不把生成文件重新改写当作新证据。

## Tests

当前统一 configure、默认全部 targets 构建与 **15/15 CTest 通过**；含 GEMM/Attention 独立计数、累计 counters、endpoint 分配和旧快照迁移。证据：[continue-20261013/](evidence/continue-20261013/)。

- source_golden、hardware_search、compile_execution、signed credit、operator/fused accounting、endpoint allocation（含旧默认奇字节冻结导入）、Runtime credit_scale、estimate driver、runtime_semantics、C ABI lifecycle 已运行通过；新增 counters 的 partial/cancel/retire、v2→v3 迁移、失败导入和分类守恒也已通过。
- 合成 grouped Runtime oracle 独立对照真实运行：E=5/8/16 均未满足全部门槛，最大完成误差约 9.72%/9.46%/6.97%，最大 HBM busy 相对误差约 3.39。对应 CTest 是 measurement driver，退出 0 不代表误差验收通过。
- 真实编译 GEMM oracle 已构建并运行：singleCore OS/strictFP32，M=N=65/K=4097，tm16/tn64/tk64，R=10、Tk=65、650 个真实消费 wave；独立工作账本全部守恒，但 E5 独占完成误差 3.827%、HBM busy 124.951%，50ms phase-offset 并发第一任务完成误差约 16.67%、HBM busy 158.87%、shared busy 478.94%，因此 **failed**。生产 E8/16 不可用，保持 not_run；不拟造 theta、不改门槛。完整报告见 evidence。
- 30 样本 128³ GEMM 探针在 shared-Hardware/typed-finish 改动后 cold/hot P95 约 2.23/2.11 ms；cold 2ms 目标尚未达到。这是单 shape 探针，不是正式全域开销验收。
- 五仓配置仅 wscllm 完成；AstraSim 真实构建、业务/指标/cancellation 回归尚未运行。上轮 JOINT 28/28 CTest 与 72 tests+12 subtests 是旧二进制历史结果，不是当前回归。
- TX8 source/cache 审计复用上轮已通过且工作簿未变的证据；目标硬件留出仍 not_available。

本环境可复用命令（工具路径须在新环境重新核实）：

```bash
work/tooling/cmake/data/bin/cmake -S die -B work/benchmark-fix -DCMAKE_BUILD_TYPE=Release
work/tooling/cmake/data/bin/cmake --build work/benchmark-fix -j2
work/tooling/cmake/data/bin/ctest --test-dir work/benchmark-fix --output-on-failure
python3 die/validation/compiled_gemm_oracle.py --driver work/benchmark-fix/coarse_compiled_gemm_oracle_driver --output work/benchmark-fix/compiled-gemm-oracle-report.json
python3 die/validation/discrete_oracle.py --runtime-driver work/benchmark-fix/coarse_oracle_granularity_test --output work/benchmark-fix/synthetic-oracle-report.json
```

完整阶段证据已保存到 `docs/evidence/continue-20261013/`；scratch/build/probe 留在忽略的 `work/`，不能是跨账号唯一证据。GCC __int128 pedantic 警告保留；未执行 sanitizer 不标 passed。

## Blocking Issues

- 编译宏的 input/output rounds 不能代表 R*Tk 真实有限缓冲生命周期；tail 最慢核、WR 围栏、精确 full/tail routing 未闭环，真实编译 E5 fixture 已失败，E8/16 尚不可用，没有可用证书。
- launch/setup 尚未收费；layers/passes 工作量倍数已修，真实重复实例时序未修；fractional kappa 仍有整数截断，向量缓存/转换与 partial/tail 分布须进一步验证。
- 五仓固定源发布、typed MetricCollector、DIE HBM 总分子、workspace 增量准入/释放、GPU 槽一次完成、KV/LRU/network/pool 回归与 JOINT live projection 尚未验收。
- TX8 不能辨识目标 INT4/Vec/HBM/NoC/SRAM/启动/转换/竞争，数据不足不是可通过修改拟合权重消除的问题。
- cache/LP/J<=32/事件/RSS/Roofline 正式开销证据尚缺；单小 shape 探针不能覆盖全参数域。

## Handoff Notes

这是持续开发中的阶段文档，不是用户要求的完整交接。公共库检查点已经完成完整构建和 15/15 CTest 验证，真实编译 oracle 的失败结果一并保存；`.github/workflows/die-handoff-check.yml` 增加真正的默认构建和 CTest job，旧文档检查仍保留。

本轮先以 `7e7d8a4` 推送 benchmark 小修复，再将本文件与完整公共库检查点一起提交。同步 main 使用非 force fast-forward；本文件所属提交的实际 SHA 从 Git 元数据读取，不能编造自身 SHA。五仓固定源发布和业务改动按下一有意义检查点保存，不以公共库测试代替五仓结果。旧 PR #1 是上轮历史事实，本轮不重复创建无必要 PR。

收到用户明确交接指令后，须完整更新本文件、保存所有有效成果、同步 GitHub，并核对工作区与远端；没有该指令时继续实施 Next Step。当前工程精度/五仓验收不足仍为 NOT READY，不能直接用于已验证的架构性能排序。
