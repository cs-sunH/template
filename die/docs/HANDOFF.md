# Codex Cloud 开发检查点与接力状态

记录日期：2026-10-14（Asia/Shanghai）。本轮已从最新 main 恢复并持续开发；没有收到本轮停止或交接指令。检查点同步后继续推进，不按阶段数量或额度停工。

## Current Status

**公共源及所选接入回归通过，模型精度和全量工程验收仍未完成。** compact batch、fractional κ 与 strict INT4 成本成果已通过 PR #4 合入 main。本检查点缓存 compile-batch 的 escaped hash/summary 响应，保留新句柄、逐项错误、JSON 顺序与版本标记；公共默认 targets 构建成功，**22/22 CTest** 通过。五仓同一最新 20 文件发布，本阶段两 ABI 增量构建、所选现有 CTest **17/17**、显式 ABI 配置测试 **83/83**，均无跳过；Online/AstraSim executable 本阶段未重建。

真实 compiled K-wave 比较仍 **failed**。新独立参考已纠正为 owner Q/α：该 GEMM 的 C 活动即使 LOCAL 容量不绑定，Q 仍为 LOCAL byte，而不是纯 compute 的 1；历史 equal-progress 结果并列保留。默认生产尚无真实逐 K 输入完成、K-final 产出、写完才退槽的有界生命周期；E8/E16 不可用。周期研究和独立 C++ scratch 不给生产模型颁发证书。

相同三 shape、每路径 30 样本复测，真正 same-model C ABI hit P95 从 **34/36/42 μs** 降至 **10.426/14.292/18.788 μs**，本夹具集达到 20 μs P95 目标；Attention P99 为约 36 μs。128³ GEMM C++ cold P95 **3.788 ms**、C ABI miss **3.861 ms**，仍超过 2 ms。C++ hot 是重编译，核心未改，其差异只作对照。J=32 未测，不宣称全参数性能验收。

PR [#4](https://github.com/cs-sunH/template/pull/4) 已确认 merged，Actions #21/#23 success，fetch/ancestry 核实 main `47079d88fcf3a8408743a282804cd2098dee8557` 包含其两个提交。当前检查点基于该 main，GitHub 同步结果在 Handoff Notes。**当前检查点合入 main 并经 fetch/ancestry 核对前，NOT READY FOR CODE HANDOFF**；持续开发不因此停止。

## Integration Branch

共享集成分支：`main`；当前检查点分支：`codex/die-bounded-lifecycle-20261013`；上一已合并分支为 `codex/die-runtime-waves-20261013`。不 force-push，不覆盖他人提交。后续开发继续以已合并检查点为基线，不假定不同账号共享 work/ 或原 Session。

## Last Verified Base

- 开始时 fetch、切换 main 并 fast-forward；核实 PR #3 merged、工作区干净，基线 `f3b53eb906df788eecf613642106aa974d5d24a7` 包含最终源码 `37b13ac928d43b055d0b31ae28caf45d11763659`。
- PR #4 合并后 fetch/main fast-forward 至 `47079d88fcf3a8408743a282804cd2098dee8557`，核实包含 `c0f559a...` 与 `8b2ab3a...`。当前检查点前再次 fetch，远端 main 仍为此 SHA，没有发现他人并行改动。此前源码与五仓接入成果复用。
- 根硬件/设计资料与 TX8 工作簿未改。前两组证据保留于 `evidence/handoff-20261013-final/` 与 `evidence/runtime-waves-20261013/`；当前单列 `evidence/cache-lifecycle-20261014/` 及 `analysis/bounded-lifecycle-20261014/`。

## Completed

此前已通过的硬件黄金、几何、signed credit/LP scaling、GEMM/Attention/FFN/RoPE 独立账本、typed task 校验、慢核 tail 预算、累计分类服务计数、endpoint 分配、C ABI 生命周期/快照迁移、五仓 facade/Online/COMP+COMM 夹具全部保留；详细历史在前一 evidence 与 Git 中，不重复实施或无新数据重拟合。

### compact progression × batch

- `Compiler.cc` 保留 progression 的 `count`，另保存 `batch>1`；不再把 count×batch 扁平化为更长 KV progression。q-token 总行数计 count×batch，最终 KV/query 位置只沿原 progression 推进，检查乘积/位置溢出。
- `Mapping.cc` 的 Attention、scale/softmax/row consumer 按 batch 重复实际序列工作量。materialized score/probability 同时 workspace 乘 batch；fused per-tile SRAM 峰值保留每 tile 的容量语义。
- 新 `compact_batch_semantics_test` 将实际 compile/estimate 与独立显式 spans 比较，覆盖 causal/noncausal、fused/materialized、row 算子、workspace、非法溢出与 roundtrip。batch=1 的冻结输入/hash 不变。

### 非整数 κ 与兼容边界

- 私有 `EquivalentWork.hh::Work` 保存 exact u128 whole + long double dyadic fraction；按输入 binary64 κ 的真实表示加权，在 cast 前检查整数/分数溢出，不 ceil、不截断总工作。宏分摊将未分配的分数保留在 exact remainder，基础块、放大块和重组块都能精确保留该份额。
- GEMM gate 与账本同时使用真实 κ；strict FP32→INT4 output 按 κ_quant。Attention/FFN 的 strict INT4 输入解码及 INT4 输出同样按 κ；修复 Attention FP32 输出未转换时多算一次 cast-write 的 physical ledger。
- Activity 保留 `uint64_t vec_eqops` whole，新增内部 C++ fraction；只有实际非零分数导出 **Plan v3**，精确 dyadic wire 为 `{significand,exponent}`。整数 Plan v2 原 JSON/key/hash 保留；Plan v2/v3 导入、hash/canonical 校验与 snapshot v2/v3 兼容。
- fraction 无法在 Runtime scalar 与 whole 精确合并时，导入/校验报 overflow：既拒绝完全消失，也拒绝向上舍入，不静默改计数。当前支持 binary64/x87 等最多 64 significand-bit 的 long double；更宽格式明确不支持。
- summary 的近似数值保留给普通展示，fractional 情况增加 authoritative whole/dyadic sidecar 与 schema marker；C ABI compile-batch 标明 Plan v3。固定 `DieActivityV2/DieCompletionV2` layout、ABI version 2 未改，固定整数字段不承载分数；所有五仓 C++ 消费者已按新增内部布局重建。
- Runtime 服务、终态修正、progress 和 snapshot 使用分数工作；极小非零 decimal counter 在固定 18 位小数不能往返时改用 max_digits10 科学计数，防止 snapshot 把它写成零。
- `fractional_work_test` 覆盖 1.25/.75、0、0.1/.2、巨大有限 κ、strict INT4 GEMM/Attention/FFN、C ABI、partial/snapshot/terminal、denorm counter 与非法 wire 精度边界；`equivalent_work_test` 检查 exact whole/fraction 及分摊子集守恒。真实 GEMM tail oracle 扩至 **288 candidates**。
- 当前七个默认 example task hash **及 summary 7/7 不变**；不能将这个冻结输入集的兼容性推广成所有旧错误成本场景 hash 不变。

### K 生命周期研究与工程边界

- 手工 synthetic Plan 在真实最终 Runtime 中显示：8 至 10^6 waves 均 4 次 event advances、finish 守恒，但连续正速率 load 的 HBM busy 大幅膨胀；不能通过改 busy 定义、阈值或 theta 消除这个问题。
- exact rational grouped Q/alpha 研究的 262 个固定双 owner、非末 K-full offset 样本均与逐事件积分一致，最多 9 probes 回返；cap=1 验证 fail-closed 前缀。它没有覆盖真实 K-final/store/retire/取消/容量变化及生产 13 域 LP，也不是所有 offset 的证明。
- C++17/GMP 独立生命周期 scratch 已在有限 650-wave fixture 核对 load-ready、K-final、store-retire；当前 v1 只跳过同一输出中的非末 K-full，因此整体仍 O(R)，不是生产所需对 R 有界的表示。含完整输出周期的下一原型和真正生产接入继续开发，GMP 不成为公共库依赖。
- workspace 只读审计分别核对无 LRU wscllm 与 wscllm-LRU：准入应在 `take_node/occupy` 前，KV manager 保持容量权威，C++ transient workspace 必须随下一 bridge epoch 回送，否则新 KV reservation 可与 active workspace 超订。单 Compute slot 不能代替此容量同步。报告不等于实现。

### 缓存响应与更完整的周期研究（2026-10-14）

- 不可变 cached ResponseTemplate 保留 escaped hash/summary JSON，每个成功 occurrence 仍返回 fresh Plan handle。新字符串 capacities/NUL、对象和 allocator allowance 均纳入 4096/64MiB retained 上限。批内后续淘汰不会破坏早先行的 response/Plan；返回 buffer 分配成功后才登记句柄，登记异常回滚已登记句柄。malloc-failure 未故障注入，不将静态异常安全审阅写成该测试已通过。
- 新 `capi_cache_response_test` 覆盖 estimate 预填、混合失败/重复、fractional marker、canonical field order、批内淘汰、句柄 submit/cancel/release；真实 unique GEMM 在 4096 条前触发 64MiB byte cap，count cap 未独立触发。默认 7 个 hash/summary 不变，实际 accounted bytes 从 141094 增至 164810。
- 新 `grouped_gemm_oracle.py` 按 owner active Q 总和分层公平，冻结后不重新归一化；disabled-domain 零速活动排除 owner 分母，private bounds 保留。8 项数学测试及 12 个真实 Runtime rate/α 合成夹具独立对照通过。严格 1-Core FP32 的 per-wave LOCAL 合计 9636640 byte；历史 equal-progress 与 Q=1 toy 不再被当作真实 GEMM grouped 参考。
- whole-output exact recurrence 原型保存固定 2 输入/2 输出槽、每 owner 至多 3 活动，实际 K-final→store→retire。root 重编并跑 7 arrival scenarios×3 policy，独立 equal-progress/grouped LOCAL-Q 的完成/busy 对照各 7/7 通过。R=10⁶、Ntiles=2 的 solo 仅 72 个实际 interval、跳过 999994 次输出退休，closed-form byte/ops/LOCAL/retire 守恒；没有展开其 6500 万 wave oracle。
- 双 owner offset 1/20、R=1000 的 exact-key 探测仍在 outer256 fail-closed，progress 有约 16k-bit 有理数；不能换成固定 128/256-bit rational。长逐事件对照被中断，未通过。Ntiles=1000 的绝对列号键也阻碍同类 full-column 合并，不能靠删 tail guard 强行跳过。
- 独立 520-event symbolic checker 对每步 grouped LP、521 个队列/槽边界、严格 event-order、identity ties 与 progress 域逐项复验；猜测的 1D 状态线经合成证明闭合，得到 `x'=a*x+b`、`0<a<1`，x0 与 fixed point 均落在同一合法 trace 区间。497 周期的状态/时间误差界使用 exact Fraction 断言。只证明固定 offset/full-MN 区段，不证明 flat256 可发现、不跨 tail/取消/容量变化；生产没有接入 GMP 或此 scratch。
- C++ scratch row-prefix 含 `TN*TM`/`Ntiles*TM` typo；当前 FP32 payload/prefix 均为 4 倍数、分行恒精确一半，结果不受影响。checker 使用正确前缀，旧 source/trace 冻结并明列限制，不推广任意 dtype。
- wscllm 已细化显式 bridge v2、版本化 workspace view、Python base-capacity snapshot、prepare同一Plan→pre-issue lease→一次释放/唤醒方案。现 bridge 是 JSON，不是 Chakra protobuf；v1 optional/missing=0 不安全。该稿仍未实施/测试。

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

## Next Step

复用当前通过的实现和证据，持续推进；这些优先级不是自动停工阶段上限。

1. **P0：真实编译 K lifecycle 与 busy union。** 保留正确的 LOCAL-Q grouped oracle、独立分配/生命周期及 affine proof。后续表示需同时处理可精确重复和严格收缩轨道，推导固定复杂度的 piecewise affine state/service/busy 积分及安全 event-order 区域；不能以 epsilon cycle equality 或 GMP 大整数进入生产。520-step 证明不是256-budget发现算法，要研究嵌套 K/output 合成和解析类别段。先在真实 1-Core OS fixture 接入 shared Runtime，再 offset/tail/multi-Core/WR fence/外部容量取消；不跨外部事件，字节/ops与正速率busy union守恒。保留3%/5%和E≤8/16上限，未实现前 unavailable/not_run。
2. **P1：剩余成本/输入语义。** fractional κ、strict INT4 conversion、compact batch 已完成，不重做。launch=1000ns/setup=0 尚未按真实区域/层/pass 实例收费；参数按 consumer-Core 首次读取与完整切片/live-buffer fit 复用、Norm/RoPE 长行两遍输入与 storage conversion 未完整；task residency/readiness/math/TP 局部区间仍需兑现。依赖有界生命周期，不把所有 launch lump 在开头。
3. **P1：五仓业务容量与专属路径。** 先无 LRU wscllm 做 prepare Plan→发射前 workspace token→一次释放，连接版本化 KV capacity/StateDelta workspace 视图，保证同步 commit 与跨 epoch 不超订；不改变 selector/order、weights/KV/旧 journal、HBM traffic。随后逐仓 LRU/restore/pool/readiness/network join/owner history 与 JOINT live import/fork/overlay；真实实现测试前保留 frozen legacy pipeline guard。
4. **P2：性能与发布。** cached response 已解决本三shape的hit P95，复用新旧n30证据。128³ cold仍未达标，正在独立work目录profiling候选生成/seal/Runtime/LP，不先减候选或换模型公式。再测 J≤32/LP/CPU/RSS、三 profile 与五仓完整在线业务；不把一组 host P95 当全域验收。新增cache retained内存继续受双上限约束；公共源变化统一发布五仓digest，达到有意义检查点再同步GitHub。

没有新可审计硬件测量时，继续冻结两个 TX8 reference 与目标工程先验，target holdout 保持 not_available。

## Important Decisions

1. 根《架构建模新.md》控制硬件输入/派生，根性能设计控制执行假设；13 物理域不变，INT4/Vec 峰值、logical bytes、physical payload、D/eta 与 private theta 分开。`die/` 是唯一公共源，五仓只使用固定内容校验发布。
2. L/C/S work 已为整个 epoch 总量，repeat_count 是审计 metadata，不再乘；tail compute 保留 slowest cohort、§7.2 的 active-Core 保守 COMPUTE_GATE 与最多 128 候选，不能无说明替成 /Ncore 或缩减候选。
3. 分组公平 owner 为 `(node_id,execution_id,generation)`，request_id 追踪；signed credit、词典序 max-min 与取消仅计已服务量保留。连续宏信用不是逐 K readiness/retire 的证明。
4. Plan v3 只用于非零 fractional work，整数 v2 格式冻结；whole/fraction sidecar 是精确账本，近似 JSON number 不能被 get<uint64_t>() 当权威。固定 C ABI v2 layout 与 JSON 版本扩展分开。
5. Runtime snapshot v3 的累计分类账本、v2 history-incomplete 迁移和原子失败导入保留；旧历史不能静默当零。分数计数的 decimal 需精确往返当前 scalar。
6. 手工 Runtime、Python grouped rational 和 C++ scratch 仅证明其明确 fixture；真实 compiler oracle 单列并继续 failed。周期键不能只看类别或近似相位，skip 不得跨外部事件；256 cap 不能因逐输出无限重置而冒充整体有界。
7. cache 4096/64MiB 是 retained allocation 上限，不是 host RSS。C++ hot compile 不是 cache hit；报告 passed 可以仅指测量完成，预算判断另报。
8. known_local_world 不自动包含 network/pool；JOINT guard 只在真实 live import/fork/join 测试后移除。workspace 是 capacity 增量，不是传输费用或 KV reservation，不双收费。
9. source/cache 复现、TX8 条件 reference、粒度精度、目标硬件、五仓业务与 host 性能都是独立证据；无新测量不升级参数解释。

## Files to Continue

- 公共编译与运行：`src/{Compiler,Mapping,Execution,Plan,Capi}.cc`、`src/{GemmTailBudget,EquivalentWork}.hh`、`include/coarse_die/{Plan,Execution,Capi}.h*`。
- 新计数/兼容测试：`tests/{compact_batch_semantics,equivalent_work,fractional_work,gemm_tail_budget,capi_cache_response}_test.cc`、`tests/grouped_gemm_oracle_test.py`；已有 operator/GEMM/Attention/FFN、counters/endpoint/credit/C ABI 测试全部保留。
- 真 compiler oracle：`validation/{compiled_gemm_oracle,grouped_gemm_oracle}.py`、`tests/compiled_gemm_oracle_driver.cc`；measurement driver 成功不是误差验收通过。
- 当前证据/研究：`docs/evidence/cache-lifecycle-20261014/`、`docs/analysis/bounded-lifecycle-20261014/`；上一检查点 `runtime-waves-20261013/` 原始证据保留，历史 audit 先读各目录 README 的已解决/未实现区分。
- 五仓：Workload 发射前 gate、DieModelAdapter、CoarseDieRuntime、KV manager/DecisionBridge/GraphBatch、MetricCollector/HBM endpoint、JOINT NativeProjection；各仓业务策略不得互换。
- TX8：`calibration/report.md`、reference configs、`tools/calibrate_tx8.py`；生成文件不重新改写冒充新测量。

## Tests

当前逐项结果、真实日志、工具/二进制 SHA、固定源码 manifest 在 [cache-lifecycle-20261014/](evidence/cache-lifecycle-20261014/)，原 PR #4 在 [runtime-waves-20261013/](evidence/runtime-waves-20261013/)。

- 公共默认全部 targets 增量构建，**22/22 CTest**；保留 PR #4 的20项，新增真实cache响应/淘汰测试及grouped oracle数学测试。最终日志 `public-ctest.txt`。
- 默认七个 examples 用基线和最终 C ABI 实际 compile，hash 与 summary **7/7 unchanged**。修复过的 INT4 成本场景另有新正确结果。
- 真实compiled E5对新的grouped参考：独占finish3.827%、HBMbusy124.951%；50ms offset第一任务finish约10.782%、HBMbusy158.87%、sharedbusy约994.515%，**failed**，命令exit1。复用上次真实driver raw与未改变的Runtime binary；legacy equal-progress第一任务16.67%/shared478.94%并列留存。生产E8/E16 unavailable/not_run。
- 五仓source digest `05c3f91c59e4ac3d22b5147a12566d93902814aa77bbf3384a9b7a5468a9e755`，20文件；sync check passed。本阶段请求legacy+coarse两个ABI targets，CMake实际也重编core依赖；运行现有选定CTest 3+3+3+3+5=**17/17**，当前ABI config unittest20+15+18+17+13=**83/83**，无跳过。Online/AstraSim executable本阶段未重建，其此前e877构建日志保留；不宣称完整五仓业务。
- 三shape各30个C++ cold/hot和30组true CABI miss/hit，P95为第29排序值。hit P95 **10.426/14.292/18.788μs**，该夹具集达20μs；Attention P99约36μs。128³ C++cold **3.788ms**/CABI miss **3.861ms**仍失败。此次RSS9344–13056KiB、load/CPU/二进制hash与原始samples全保存；原baseline未覆盖。J=32 not_run。
- root重编small C++ prototype，21个policy/arrival配对event/cycle一致，独立正确grouped参考7/7；12个真实Runtime分配rate/α通过；root重跑520-event affine proof checker通过。规模solo的closed ledger与并发fail-closed分别报告，不是生产精度/复杂度证书。
- TX8 source/cache 复用已验证且未变化的历史证据，target hardware 留出 not_available。本轮生产库未跑 sanitizer；历史独立 Work helper ASan/UBSan 不代表本轮生产 sanitizer。

工具为 CMake 4.4.4、GCC 14.2.0、Protobuf 3.21.12。aggregate mirror 三 subdirectory，Release/congestion_aware/library backend。CMake 位于本地忽略的 `work/tooling/cmake/data/bin/`，Protobuf 在 `work/deps/protobuf/`；接手环境要重建依赖。protobuf v21.12 tar SHA `22fdaf641b31655d4b2297f9981fa5203b2866f8332d3c6333f6b0107bb320de`，configure **和 build** 保留 PATH/CMAKE_PREFIX_PATH/PROTOBUF_FROM_SOURCE=True，所有 Chakra bindings 位于 build tree，勿提交。

## Blocking Issues

- 真实有界 K lifecycle 尚未生产实现。whole-output exact cycle 已在solo规模样本跳过R，但双owner严格收缩轨道不会精确重现，探测/位宽及更一般affine区域需闭环；有限证明不能消除生产busy误差。E8/E16无证书。
- launch/setup/重复实例时序、向量参数驻留/完整切片 fit/长行转换、task residency/readiness/TP 区间未完整；已完成 fractional/batch 不在阻塞列表中。
- 五仓选定构建/回归通过，但 workspace admission/释放与权威容量桥接、其余粗粒度业务/KV/LRU/pool/network、JOINT live projection 未验收。
- TX8 不能辨识目标 INT4/HBM/NoC/SRAM/启动/转换/竞争；不能改拟合权重制造独立数据。
- 128³ host cold P95仍超预算；三shape cache-hit P95已达标，不能外推全参数。J≤32/LP/完整workload开销未验证。

## Handoff Notes

本轮未接到停止/交接指令；完成这个代码检查点后继续自主开发。不要继承上一轮文档里的“已停止”指令，不检测额度或自动切换账号。

上一检查点 `c0f559a071505c7ea750530e1adb3c47238ffcf0`、metadata `8b2ab3a893cab31e2c15cb28e868f20e0b259837` 已由 PR #4 合并，Actions #21/#23 success，merge/main `47079d88fcf3a8408743a282804cd2098dee8557` 经fetch/ancestry确认。当前源码/证据在 `codex/die-bounded-lifecycle-20261013`，将按expected-head同步检查点；实际commit/PR/CI从GitHub元数据读取，不编造自身SHA。当前PR合入且远端确认前NOT READY FOR CODE HANDOFF；误差与全量业务验收独立保持NOT ACCEPTED。没有收到停止或交接指令，检查点后继续P0/workspace/冷编译热点。

下一账号仅在用户真正发出交接指令、代码与最新 HANDOFF 进入 main、远端 ancestry/digest 确认后接力。`.github/workflows/die-handoff-check.yml` 跑真实公共默认 build/CTest 与必需文档检查；本地成功不能代替尚未读取的 GitHub Actions 结果。工作目录中的依赖/二进制不跨账号共享，依据保存的源码/命令/真实证据直接接续 P0。
