# Codex Cloud 开发交接状态

记录日期：2026-10-14。用户已明确要求停止启动新开发任务，当前最小任务收尾后保存并完成 GitHub 交接；合并与远端核实后立即停止。工程整体 **NOT ACCEPTED**。本文替代上轮当前状态；历史源码、测试和停止指令保留在 Git 历史及各 evidence/analysis 目录，[上轮完整记录](https://github.com/cs-sunH/template/blob/912b748/die/docs/HANDOFF.md) 不作为本轮指令。

## Current Status

本 Session 从最新 main `ad865f11b6466dcc1d07a75be442740b8db20d01` 恢复，已核实包含上轮 PR #10 源码 `2e3e0294eecdbe54cd6c645adefa1d078fe085fc`。复用了公共 Compiler/Mapping/Runtime/C ABI、bounded OS facts、LRU workspace v2 桥接和前轮研究，没有重新拟合硬件常量或重做已通过成果。

本轮实现 compiler-derived capacity-only bound、LRU finite-manifest future margin/实际 Plan 超界检查、源 store/migrate 物理释放依赖；公共、LRU focused 与两组实际 Online 已验证。用户新增公式优先原则已入 AGENTS、PROJECT、README 和根设计 §1.4。新活动 `astra-sim-joint` 完成 additive typed adapter/runtime/workspace seams 的定向检查点，但业务 sh30 provider/native projection 尚未迁移，不能称完整 JOINT 接入。`astra-sim-joint_old` 本轮未改，只作历史档案。

生产 E5 finish/busy 原 3%/5% 门槛仍失败，E8/E16 不可用；TX8 reference/target priors 冻结，目标 held-out 数据不可用，host 开销预算没有新通过证书。两组 Online 不是容量极限压力或芯片测量。

**GitHub 源码交接已完成，可基于最新 main 安全接棒。** 源码与初版完整交接文档提交 `60d18dc26173f265f050e7f71fa3ad12196cbc5e` 经 [PR #11](https://github.com/cs-sunH/template/pull/11) 合并，merge `721ab833c3e17f856e7bf18b7175b7d5f3980901` 已 fetch/ff；source/principle ancestor 检查均 exit0。[源码 CI 38590775233](https://github.com/cs-sunH/template/actions/runs/38590775233) 的公共构建/CTest和文档检查均success。本文件最终元数据与 [GitHub事实](evidence/capacity-joint-handoff-20261014/github/) 在该 merge 后补入 main；工程整体仍 NOT ACCEPTED。

## Integration Branch

共享分支 `main`；本轮交付分支 `codex/die-capacity-joint-20261014`。用户原则文档提交 `912b748`、源码/交接文档 `60d18dc` 均已通过 PR #11 合入 main。随后仅补合并/CI/远端核实元数据，不再开发或新测试；不 force-push，不覆盖外部 main，不丢弃有效修改。下一账号 fetch 最新 main，核实包含源码60d18dc、merge721ab833和本HANDOFF，再创建工作分支。

## Last Verified Base

- 起始 main 基线 `ad865f11b6466dcc1d07a75be442740b8db20d01`，上轮 PR #10 已 merged；交接源码 merge 后实核 main `721ab833c3e17f856e7bf18b7175b7d5f3980901`，最终元数据提交不预编自身SHA。
- 实际 LRU Online 的 checked-out context 为 `912b748` dirty tree；每文件 source SHA、diff、实际二进制 SHA 和 manifest provenance 单列，不能把 context SHA 冒充运行源码身份。
- 公共 coarse C ABI SHA `c678d268a7f9e624d5389d5a11a830258f5b2a6bdbc4d6157c2a647cf231b0f1`，public static SHA `089ee1a432fb77491e57d5b613310b6aee59b131af121d9622969d9dbe7f8c81`。
- canonical23 发布 digest `9d025e25699e44b708997da5e4bc9c45e60c75b02166be931282f20921c0c4f4`，活动五仓同步；`_old` 未同步/未改。

## Completed

### 公共编译器容量接口

`include/coarse_die/{Plan.hh,Capi.h}`、`src/{Compiler,Mapping,Capi}.cc`、`tests/workspace_envelope_test.cc`：新增 C++ `WorkspaceCapacityBound` 和 `die_coarse_workspace_capacity_bound_v2`，同 normalization/identity，输出 exact u64、`workspace_capacity_bound_schema_version=1`、`capacity_model_revision=workspace_lifetimes_v1`、`capacity_bound_kind=all_template_lifetimes_without_sram_filter`。共享 checked-u128 attention rectangle/query 与 FFN lifetime 公式，省略 SRAM 候选筛选，不 enumerate/estimate，也不证明 task 可执行；不改 cache/Plan handles。

旧候选 envelope/Plan JSON、hash 和 summary 不变。加入 candidate max 不超过新 bound 的内部断言。反例：父任务可因 SRAM 筛选只余零 workspace mapping，子任务却有物化 workspace 8,704 B；筛选前父 bound 98,304 B 能覆盖。token 总和单独不能证明 attention span 矩形覆盖。INT4 奇数/重复/overflow/零shape/head/dtype/metadata 边界有正式测试。旧62任务 identity 完整全等，旧240 envelope response/error 完整全等。

### LRU future margin 与物理释放

`online/{die_workspace_envelope,manifest_workspace_envelope,wsc_llm_online_scheduler,online_service}.py`、`session_kv_manager.py`：required+coarse finite `request_aggregated` manifest 初始化前查询 C++ bound，为每rank安装不可变 E。P 每请求 fresh production emitter；D 取各请求完整 compact decode progression 的并集和最大 passes，对所有14 typed op/all group ranks 查询。没有 Python 延迟/workspace 公式。证书一次保存 `bridge/workspace_envelope_certificate.json` 后丢弃大 proof，保留 frozen margin/identity。

`WorkspaceAdmission.hh`、`ParsedGraphBatch.cc`：optional full-rank `prospective_workspace_bytes` strict u64，显式0与缺省区分，后续不允许新增/移除/改变。fit 为 `base + max(A,E) <= capacity`，E 不进 active/revision/KV/reservation/remote账。实际 Plan `W>E` 在 acquire/mutation 前拒绝。正常每rank串行 Compute gate 是 max 而非 sum 的前提；不覆盖绕过 gate 的 direct 并发调用。

`generate_wsc_llm_trace.py`、`online/graph_batch_builder.py`：仅 required 路径保存 per-source-rank release ACK ledger。remote_store source=edge 用 mem_store completion，否则 ACK recv；noc_migrate 用 source ACK recv。后续同rank ordinary COMP、remote_load 目的 HBM 写及迁移目的 recv_kv 在容量复用前依赖该终端，保留 pool store tails/marker/timer/旧 OFF emission。OFF JSON baseline 字节全等，SHA `64945939abc1ef398486bf8823f87e1770a8ddae29fc4ad41e63790e46a099d0`。

`execution_driven/tests/graph_batch_committer_test.cc` Part Q：real parser/committer/NodeStore 的 live ACK、terminal、跨batch与GC语义；test-local parent+p2p Kahn检查及负cycle witness。**生产 preflight 仍是 perrank parent cycle 检查，不是完整 parent+p2p cycle 验收。**

两组真实8rank Online：两session各两turn，MHA4/H512/F1024/vocab512/BF16/1layer；P64/700、P96/900，D2/5、D3/3；合法4×2 mesh、P0–3/D4–7 Ring2×2，CABI capacity 144,000,000,000 B/rank，无假容量/需求或policy patch。默认/iteration cap2 均 C++0/Python0，16/19 batches，480/560 typed Plans，48/56 正 workspace Plans，764/880 successful terminals，128/152 capacity rows fit，8条migration ACK容量消费者依赖，GC retained0。所有 Plan W<=E，最终active0；revision推导480/560 balanced lease pairs。bridge边界没有采到正live workspace，不称实际采样占用时间线。两run同证书 SHA `6dcb74dc62af91711c8f2161aca9a36e7678ecee32b89cee8d38c4cf57603abb`；P E=12,128,704 B/rank，D E=324,200；actual最大P921,600/D4096（cap2 D2048）。未观测 remote-store/restore 容量压力，仅 focused fixtures 覆盖其排序。

### 新 JOINT additive 检查点

活动新仓中的 Python `online/die_model_adapter.py`/oracle/builder、`generate_trace.py`/face scheduler 保留 legacy `die_op_spec` 和旧 A3 路径，新增 explicit `coarse_die_task_v2` typed构造/真实ABI校验，缺必要输入 fail-closed，无另写Python时序。GraphSource/parser/NodeStore保存 typed JSON；`DieModelAdapter`、`CoarseDieRuntime`、Sys/Workload 在occupy前以同一Plan完成 workspace lease→submit→一次release，LocalHbm端点接同Runtime。CMake protobuf生成放build tree，链接canonical coarse源。

schema3 legacy请求字节保持原样，coarse可加workspace_state；typed response须有完整snapshot/revision，commit原子install后reissue。四项新/扩充C++binary（CoarseWorkspaceRuntime、GraphBatchCommitter、ParsedBatch、WorkLedgerBridge）及 legacy Adapter/Lifecycle binary通过；Python legacy31+coarse14共45/45，无skip，全部14typed op实际CABI估值0error。**这些是seam/fixture证据，不是新JOINT production Online/native projection/world parity。** `_JointDiePlanProvider` 的 native frozen guard 保留，sh30真实业务typed emitter尚不可达。有效部分全部保存，不把旧JOINT已验结果套用到新仓。

### OS研究恢复（不接生产）

`analysis/os-kernel-recovery-20261014/` 保存恢复后的fixedwidth scratch及rebuild入口，evidence保存17场景full differential、2247 Fraction primitive、正确overflow输入的ASan/UBSan smoke通过。精确gcd=1快捷路径减少long division计数，256-bit normalized/768-bit scratch/u128域未扩展；没有walltime speedup结论。n30与并发build重叠，结果污染，P95约151–194ms不能认证设计预算；perf不可用。public-ns探索3场景仍超原2ns比较界，未放宽。single-owner日志没有可达“两个输出槽全满挡住final-C”竞争witness。

只读 `analysis/os-kernel-e5-granularity-audit-20261014.md` 指出E5 `R10,Tk65` 的650 K消费被聚合成R10宏轮，repeat8仅metadata；compute账守恒但HBM busy/finish不准。建议最小full/tail M/N和K类/有限输出槽及稳定联合owner周期的解析表示，简单公式能表达的辅助计算不深化。该建议未实现，研究不替代生产E5证书。

## Next Step

本 Session **不执行**以下工作；下一账号从交付 main 继续，不重做上面的已通过范围。

1. 先核共享main源码/本HANDOFF和工具，读根两设计文档及对应代码；使用新JOINT，old仅查历史并把有用记录迁到新仓。
2. 完成新JOINT业务入口：从 `_JointDiePlanProvider`/sh30 frozen native guard、native world projection、JointKVCacheManager/decision bridge开始，明确原业务策略/virtual pipeline/pool/ledger和cancel回放映射。复用已落地typed/runtime seams；先真实legacy兼容和单请求production Online，随后known-world/snapshot/并发/取消，不能仅移除guard绕过未支持语义。
3. LRU收紧formal coverage与真正容量压力：已有四请求55合法D cohort/offset、cap与first/remainder tests和两run不是全manifest定理或容量极限证据。证明每typed OP的shape/span domination；准备源capacity真实可达的store/restore/迁移压力和closed-input路径。base WSCLLM尚缺prospective margin，FACE/FACE-LRU workspace桥接须按各策略迁移；完整 parent+p2p cycle preflight 若需要按bounded实际图实现。
4. 生产精度优先修必要K/final-output readiness：复用 bounded geometry/per-K facts、现有grouped-Q/alpha oracle、full-tail/MN类和简单周期公式；先同选tile K深度/shape轴反例，再单/同步/offset owner。只在联合state/LP区域证明稳定时跳过，外部capacity/cancel边界切段，不能逐tile扩张或reset lifetime256预算。完整outputblocked witness/整数publicclock语义未证明。不要直接把重型fixed-rational scratch接Runtime；其成本和复杂度不符合当前证据。
5. 若另改attention重复workspace峰值（目前与FFN不一致，可能把顺序layer/pass计入峰值），独立修复并明确版本/identity变化，不能偷偷改变本检查点旧Plan。完整GQA producer的q_heads/kv_heads输入及各KV/TP消费者仍待接齐。
6. 达到结构精度后按设计测试host query/cold/update/J32/Online开销，再逐仓真实旧record/key/gate/window、COMP+COMM/restore/pool/cancel/snapshot。只扩展与新失败/改源相关测试，不重复已验证工作。TX8仅有cached源重建，等独立目标raw/heldout/probe及可辨识性再拟合；不得拟合源常量或把reference效率迁到目标。

## Important Decisions

- 用户2026-10-14新增原则：与硬件设计参数、软件任务输入参数不直接相关的建模细节，能用简单数学公式直接计算就采用公式，不深入建模。该要求高于研究方案偏好；必要硬件参数覆盖、任务shape/精度/TP/span和真实readiness/竞争仍保留。复杂机制必须给简单公式不足的输入/行为证据及成本。
- 一份C++ canonical与同Runtime；Python仅数据/协议/调度，不复制延迟或容量公式。capacity-only bound和candidate envelope是两种证据，不混用。
- Future E是fit margin，与active取max，不是实际占用；immutable fullrank/strict u64、actual Plan超界fail-before-mutation。default OFF字节兼容。
- 不扩大3%独占/5%并发busy/finish、2ns exploratory fixture、active/event/LP预算，不用延迟修正系数掩盖结构错误。research256 events不是Runtime256 LP pivots证书。
- 只编辑活动joint；各仓业务差异/旧策略保留，vendor发布不等于业务接入。
- TX8两conditional reference参数：`eta_compute_reference=0.9079551923877489`（同PE/vector），`eta_vector_memory_reference=0.3411668292682927`；源Norm246μs、FFN3363μs后者是条件解释，13group不重复加BLOCK。源workbook SHA `eba181c80a2d51bd5a604bad6928610ea5039824b4ec4d5d9bd023ee4bda52ee`本轮核实，未再拟合/重跑audit。
- target assumed：eta_pe/eta_vec0.90，eta_hbm_matrix/vector0.70，eta_sram1，eta_local/mesh/inject0.70，rho_sram0.90，launch1000ns/macro，memory_setup0，kappa_quant4、kappa_acc_convert2。本轮未改。cache/formula/13group条件reference历史passed不是目标测量；无raw/heldout，不可辨识目标INT4/HBM/NoC/launch。

## Files to Continue

- 公共容量：`include/coarse_die/{Plan,Capi}`、`src/{Compiler,Mapping,Capi}`、`tests/workspace_envelope_test.cc`；证明/重放 `analysis/workspace-capacity-proof-20261014/` 与对应evidence。
- LRU：`WorkspaceAdmission.hh`、execution_driven parser/committer/tests；Python manifest/capacity client/scheduler/service/manager/builder；新portable `test_manifest_workspace_envelope.py` 只需tracked seed+实际CABI，在TemporaryDirectory重建，不依赖旧work。
- 实际Online重建脚本 `analysis/lru-future-workspace-20261014/`，结果/身份 `evidence/lru-future-workspace-20261014/`；source-release focused `evidence/new/lru-source-release-20261014.md`。raw strace、大PlanJSON/依赖/二进制不提交，脚本可再生成；重建必须生成新source/binary身份，勿冒用历史package。
- JOINT：`astra-sim-joint/astra-sim/{system,workload}/`及execution_driven；`sh_test_mesh/workload/llama2_7b_inference/{generate_trace,face_scheduler,online/*}`，优先 provider/sh30/native projection/business bridges；`evidence/joint-additive-20261014/` 记录实测范围；JOINT full Online target 本轮未构建/实跑。
- 生产OS：Mapping append_epoch / bounded counters，Execution/Runtime，`validation/compiled_gemm_oracle.py`，历史 `evidence/cache-lifecycle-20261014/compiled-grouped-oracle-report.json`；新只读audit与os-kernel-recovery/PUBLIC_CLOCK；历史affine/solo proofs保留，按各自边界复用。
- TX8：`configs/{calibration_tx8,engineering_prior}.json`、`calibration/`、`tools/calibrate_tx8.py`。根两设计文档是最终需求，不只依赖本摘要。

## Tests

忠实复用现有真实运行，停止指令后只做提交/证据/远端完整性检查，不扩展开发测试。

| 本轮范围 | 真实结果与边界 |
|---|---|
| 公共最终default Release/CTest | 30/30，约16.39s；public-ctest.log；capacity additions没有改生产timing |
| 旧Plan/旧envelope | 62/62 hash+完整summary全等；224success+16expectederror完整response全等 |
| LRU实际CABI client | 13pass，显式提供incompatible legacy ABI测试缺符号；fixture历史archived库缺失不冒充通过 |
| LRU online Python discovery | 182pass/2skip/15subtests；skip为可选oracle和缺archived pre-envelope库，专门client命令补covered incompatible路径；2 protobuf deprecation warnings |
| LRU focused C++ | CTest5/5；GraphBatchCommitter单独binary ALL PASS，含新增PartQ ACK/GC |
| LRU source release Python | 7/10/13/18/6/10 suite各pass，OFF字节全等；manifest+source+workspace组合18pass；不是unique加总 |
| 两组真实 LRU Online | default与cap2 C++0/Python0/audit通过；具体batch/Plan/terminal范围见Completed及原始日志 |
| 五仓canonical发布 | 23files同digest；base/FACE/FACE-LRU两ABI构建与config20+18+17=55无skip；LRU最终Online/两ABI构建/config15无skip。新JOINT ABI和定向seams另列；本轮JOINT config13未重跑，不能称全83/83 |
| 新JOINT | 四项coarse/parser/committer/bridge及legacy adapter/lifecycle direct binary通过，Python45/45无skip，typed14估值0error；不外推真实Online和native projection |
| OS research | recovered17/fullsuite、2247primitive、ASanUBSan smoke通过；n30污染/publicclock3failure保存，生产timing不变 |

工具为GCC14.2/Python3.12.14，CMake3.31.10（work/tooling/cmake/data/bin）、protobuf C++21.12/protoc3.21.12 prefix work/deps/protobuf，Python protobuf4.25.8 prefix work/deps/python；`source work/activate.sh`设PATH/CMAKE_PREFIX_PATH/PYTHONPATH/PROTOBUF_FROM_SOURCE=True。这些ignored依赖不可跨账号假设。公共命令 `cmake -S die -B work/build-die -DCMAKE_BUILD_TYPE=Release`、build、ctest。variant使用enable_testing+repo/backend/frontend aggregate、Release/congestion_aware/NETWORK_BACKEND_BUILD_AS_LIBRARY=ON/ASTRA_SIM_ROOT。Online脚本需strace与真实生成Chakra binding；JOINT生成binding在其ignored work目录。

失败保留：早期错误PATH/环境/namespace/零workspace fixture/legacyJSON expected排序、stale six-field auditor和cap-run inspector路径、历史sanitizer2^200错误输入均标范围后修正；不把harness没执行测试当pass，不重新归一化provenance CSV CRLF。既存GCC __int128/potential dangling等warnings未全部审查。提交源码与文档通过 `git -c core.whitespace=cr-at-eol diff --cached --check`；原样捕获的diff证据中的空白context行及C++原始日志末尾空行单独排除，不修改已记录SHA的原始证据。

## Blocking Issues

- 工程：生产E5 finish/busy失败，E8/E16未提供；完整K/readiness/尾块/offset/multiCore/WR/cancel通用证明未完成。publicclock/outputblocked及cold/J32全域/Online10%预算未通过。
- 集成：新JOINT sh30/nativeprojection业务仍guarded；base prospective margin、FACE/FACE-LRU workspace、全GQA producer/consumer、真正KV容量压力/remote_store Online和各仓全业务兼容未验收。
- 校准：TX8缺raw/独立目标heldout，不能继续辨识目标参数。保持reference和target分离，不随软件测试结果拟合。
- 交接：PR未merged、push失败或远端main未核源码/文档会阻止安全接棒；上述预期工程失败如实保存，不阻止保存可继续开发的代码交接。

## Handoff Notes

本 Session 用户停止指令生效，所有代理扩展任务已冻结。已完成和部分完成的有效源码、测试、解析研究、校准状态与失败证据一起保存；没有丢弃工作区、改 `_old` 或覆盖外部提交。停止时 JOINT 只完成 additive seams，下一账号按 Next Step 对其业务路径继续，不误判已完成。

GitHub代码交接与工程验收分开：PR #11 merged、源码CI success、fetch/ff和ancestor均已实际确认，API结果已入本检查点GitHub evidence。本文件不预编自身commit；最终main HEAD在推送后核实。下一账号可从含本文件的最新main接棒，继续明确未验收项。本Session完成最后元数据同步及远端核实后立即停止，不执行Next Step。
