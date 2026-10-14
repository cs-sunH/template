# Codex Cloud 开发交接状态

记录日期：2026-10-14（Asia/Shanghai）。用户已明确要求停止开发并完成 GitHub 交接。本 Session 已停止启动新开发任务；仅进行保存、必要验证及提交/合并收尾。交接完成后停止。下一账号收到新的开发指令后从本文件 Next Step 继续，不能继承本 Session 的停止指令为永久项目限制。

## Current Status

**公共源和 wscllm 定向回归已验证；工程整体与模型精度仍 NOT ACCEPTED。** 本轮从已合并 PR #5 的 main c47ae85f479edd0953bbb432bc05139ef1c63541 开始，保留既有 compact batch、fractional κ、Plan v2/v3、C ABI v2、缓存响应与所有旧计数成果。

最终公共默认 targets 构建和 25/25 CTest 通过。新增精确 workspace 读取、提交失败事务回滚、完全相同 owner 块的 LP 对称压缩；GEMM 编译仅延后落选候选的 canonical seal，所有原结构/物理校验和同源 Runtime 保留。五仓已统一发布 20 文件，digest 为 6c2340b9d409f863d7a373d2f3c7ffe3d8d38db3361ab287adb779c15dc12468。

无 LRU wscllm 实现发射前 workspace lease、v2 request/response/ack、Python KV authority snapshot、跨 epoch fit hold 与取消一次释放。最终源重建 Online 和所选 targets，7/7 CTest；最终 Python 在线 discovery 106/106，typed hardware+train 定向 35/35（显式 ABI，无跳过）。最终五仓两 ABI 均构建成功，显式配置测试 83/83（20+15+18+17+13，无跳过）；完整构建统计见 evidence/handoff-runtime-20261014/verification.json；其他四仓本轮未推广 workspace 协议，也没有完整业务/Online E2E 证书。

真实 compiled E5 oracle 重跑仍 failed，独占 finish 误差约 3.827%、HBM busy 约 124.951%；生产 E8/E16 unavailable/not_run。新 affine 嵌套及固定精度原型仅为独立研究，未接入生产。

本交接是否可以直接接棒以 GitHub 为准：[PR #6](https://github.com/cs-sunH/template/pull/6) 必须 merged 到 main，且 fetch/ancestry 核实包含源码交付提交和本交接元数据。未合并、推送失败、必要测试阻塞或远端未确认时为 NOT READY FOR HANDOFF。模型验收失败、未跑的 E2E 和后续研究边界另行保留，不冒充项目完成。源码 SHA、PR 与已实跑源码 CI 元数据写在本目录 evidence/handoff-runtime-20261014/github.json；未来 merge SHA 与该文件自身提交 SHA 无法提前写入，必须从所链接 PR/fetch 获取，不能编造。

## Integration Branch

- 共享集成分支：main。
- 本轮工作分支：codex/die-workspace-runtime-20261014。
- 上轮 PR #5 已 merged，merge/main SHA 为 c47ae85f479edd0953bbb432bc05139ef1c63541。
- 本轮源码/成果交付提交：45cf4d23d56bfe37c19af1f62163bb821d65c55f；[PR #6](https://github.com/cs-sunH/template/pull/6) 目标 main。该源码提交 GitHub Actions [run 38516660520](https://github.com/cs-sunH/template/actions/runs/38516660520) 的公共构建/测试与交接检查均 success，结果保存为 source-ci.json。其后的提交只补交接元数据。最终 head/merge SHA 与 merged 状态取链接 PR；没有 force-push 或覆盖他人提交。
- 下一账号先 fetch 最新 main，确认包含交付源码；忽略目录 work/、依赖与二进制不跨账号共享，按保存的源码/命令重建。

## Last Verified Base

启动工作区为旧干净 work 分支 9fdcad1f8a1640d3e91558e3b2d11c55c66b53c8；fetch 后切换 main 至 c47ae85，创建本轮分支。GitHub API 确认 PR #5 merged，head fc6460f2713448a75404b87280ef78512ad9bf07，merge c47ae85。交接前再次 fetch 远端 main，仍为此基线，没有发现外部并行提交；推送/合并前将再次检查。

根两份硬件/设计文档和 TX8 工作簿未修改。历史证据保留于 handoff-20261013-final、runtime-waves-20261013、cache-lifecycle-20261014；不覆盖旧样本。当前新增证据在 evidence/handoff-runtime-20261014，研究和未完成实验准备在 analysis/handoff-runtime-20261014。

## Completed

### 公共 Runtime 正确性与容量读取

- Plan.hh 新增 hbm_workspace_peak_bytes(Plan)：只接受精确非负 uint64 JSON 整数，缺失、bool、float、负值、越界 fail closed；workspace 是增量容量，不是 HBM 流量或权重/KV。真实 materialized FFN 与 Plan roundtrip 测试通过。
- Runtime::submit_impl 在 tentative execution 插入后的 settle/allocate 异常中删除新执行，恢复旧 rates/alpha、累计分类服务量及 event/allocation/generation；只有成功返回才递增 handle。旧世界先 advance_to(now) 的真实服务保留。成功后才登记 facade callback；登记/预约异常清理已接受 handle。
- submit_endpoint 在接受工作前分配 endpoint/transaction registry，失败清除本次 reservation。真实 solver precision failure、完整 snapshot 相等、未来 tick 服务、ID 重用均测试。没有 bad_alloc/malloc 故障注入；不宣称该项已实测。
- 对完整且精确相同的 owner 活动块取 LP 对称商：work/Q 全局行乘 multiplicity，PRIVATE/credit 不乘，每 owner alpha 保留；同 fair owner 跨多个活跃 execution 不压缩。等价比较不使用 epsilon phase/cycle equality；progress 不进入瞬时 LP，之后依旧逐真实活动终结、积分和记 busy。完全不同的 owner 仍走一般 solver。
- 平坦 block range 避免逐 owner map/vector 分配；J=1 不做对称签名。96 活动及 256-pivot 上限未提高。32-owner 独立闭式、空/中/满 credit、混合边界、取消和 snapshot 重建测试通过。静态独立审查支持商 LP 与回滚；准备的额外 Fraction oracle 未执行，不是新证书。

### 候选编译效率与兼容边界

- Mapping 私有 enumerate_mappings_for_compile 仅对顶层 GEMM 延后 seal；公开 enumerate_mappings 仍逐候选返回真实 hash。Compiler 私有候选调用 detail::estimate_candidate_latency_ns，最终 winner 完整 seal。
- 新候选 API 先调用 Plan.cc 原 validate_plan_structure（不构造 JSON/hash），再经同一 Runtime::submit_impl 的硬件/物理/credit 校验；只略过内部 trial 的 fingerprint-required 检查，只返回 typed latency，没有可逃逸 Runtime、句柄或 snapshot。公共 submit 仍拒绝未 seal Plan；没有 sentinel 假 hash。
- 28 个 128³ 候选及 odd INT4、strict FP32+fractional κ、低 SRAM OS/WR、2 layers×3 passes 逐项比较 sealed/trial latency、完整 tie tuple、winner summary/hash。128³ canonical hash 为 1bf9ea72a8fcad7149ed2ad50c6dba5e0a1abade8b4cf10366c3cbcddbfcd096。
- 历史默认七样本 hash/summary 证据保留；本轮没有另跑完整七样本冻结对拍，不将上述新案例推广为所有输入的 hash 证书。候选数量/固定档位、模型公式和公开版本未减少或替换。

### wscllm workspace v2 与业务容量

- Sys 持有每 rank WorkspaceAdmission：checked base/active sum、精确 fit、Blocked/Impossible、move-only RAII lease、rank/node/generation/lease identity、revision 为所有活跃 lease release 预留空间。
- Workload 先原 availability gate，再 prepare 同一 immutable Plan、acquire、随后 take/occupy/issue；不足时节点保留 free、不占 slot/stat，不忙循环。零 workspace 仍校验字段并使用唯一 lease。adapter 完成先 release，再统计/terminal/deferred issue；live cancel 使用已服务工作利用率、既有 Skipped watch 状态和新增 die_cancelled_comp_count；shutdown suppress callback，不排正常重试。dispatch 异常是 fatal cleanup，不宣称提供 graph rollback/retry。
- SH_WORKSPACE_ADMISSION=auto|required|off；默认 auto：coarse required、legacy off；off+coarse、required+legacy、混合 rank profile fail closed。legacy v1 wire 保留；required v2 不允许缺省容量字段。
- 唯一序列字段：request.delivery_sequence、response.source_delivery_sequence、ack.delivery_sequence+batch_id。v2 request.workspace_state={protocol_version:1,ranks:[{rank,workspace_revision,active_workspace_bytes}]}；response.workspace_capacity_snapshot={protocol_version:1,ranks:[{rank,workspace_revision,capacity_bytes,model_weight_bytes,resident_kv_bytes,reserved_request_bytes}]}。全 rank 升序，exact uint64，revision echo 与 seq 绑定。
- 完整 GraphBatch preflight（包括 online-validate=0）先纯校验；commit 后、任何 issue 前安装全部 rank snapshot，再 issue touched ∪ capacity_changed ranks。坏 batch 不改 graph/capacity view。仅 capacity 改变的 rank 也唤醒。
- Python manager 为容量权威；workspace 单独 fit margin，不进入 NodeHBMState.used_bytes、KV journal、传输或 pool。下一 delivery 在 policy/KV mutation 前安装绝对 hold，返回 mutation 后 base snapshot，重复 delivery 返回原缓存 snapshot。仅 hold 导致 deficit 也沿原冷会话 eligibility/order 回收最小前缀；当前 hold 下空实例也不够则保留 KV 等释放，不误报永久不可行。所有 reserve/history/grow/move fit 路径沿同一入口。
- workspace revision 更新 WSC capacity_epoch。跨 epoch 100/base60/hold30 再 reserve25 超订反例、release 后重试、严格 ack true、stale/坏类型/全 rank、冷会话最小前缀已覆盖。
- 实际 Online 输入准备暴露 loader 旧 generic peak 必填问题；修复 WscLlmHardware/_to_wsc_llm_hardware，coarse 保留独立 INT4 ops/s 与 Vec FP32-equivalent ops/s，generic peak 可 None；legacy 保留正数 scalar，旧 roofline getter 对 coarse 明确拒绝。没有把 INT4 TOPS 冒充 TFLOPS。
- 2 rank P/D、64-token prefill+1 decode 的官方 loader/plan_materializer 输入已物化；实际 Python↔C++ request/response/ack Online E2E 未启动，正 workspace COMP 与终止输出尚未证实。准备脚本/输入在 analysis/online-preparation，重生成写入 work/。不能仅凭描述中的 positive-workspace 字样声称该场景经过正 workspace 准入。

### 有界 K 生命周期研究（未生产接入）

- 旧 520-event exact proof 上新增嵌套 K/output 表示：每 output 8 类、32 个 representative event steps；58 个 middle-K 宏是精确状态恒等，队列/cursor 平移，保留 K-final→store→retire。
- half map 为 x'=(849475/2156112)x+548564055014789/8288669491200000；平方精确复原旧 whole-cycle map/time。逐类别独立 grouped LOCAL-Q LP、严格 event-order/tie、slot、byte/ops/LOCAL 与正率 busy union 守恒。
- Q128 向外区间及标准 C++17 long-double outward evaluator，固定 256-bit decimal input limbs，无 GMP/公共第三方新增。root 实跑独立 Fraction 1/10/100/995 与 Q128至10¹²对照通过；995 段 time enclosure 3.747e−16 s（<1ns），整数 stop 123可跳/124不可跳，257-bit/零分母 fail closed。超过995只证明 numeric powering，不跨 N-tail 声称生命周期正确。
- 只覆盖 offset=1/20、单 Core OS strict FP32、full M/N、特定三域容量。类别取自旧 trace；未实现通用 bounded event-order/LP-cell discovery、多 Core/WR/tail/容量变化/取消。不得给生产 E5/E8/E16 颁证。完整源码和边界报告已保存，生产库未加入 GMP 或该原型。
- 额外 owner review oracle/probe 保存为 unexecuted；member_change_runtime.cc 的旧 7s 预期错误已在 README 明列应为6s/shared4s，未编译/实跑，不作为 pass。

### TX8 超参数校准

无新可审计硬件测量，本轮未重新拟合；复用已验证的 source/cache 和条件 reference。工作簿 SHA256 为 eba181c80a2d51bd5a604bad6928610ea5039824b4ec4d5d9bd023ee4bda52ee；原九条测量已移除，没有原始重复/预热/独立留出。

| 参数/结果 | 数值 | 证据边界 |
|---|---:|---|
| eta_compute_reference | 0.9079551923877489 | 条件3363μs整个FFN解释，固定128T/1T、共享矩阵/向量效率 |
| eta_vector_memory_reference | 0.3411668292682927 | reported_anchor 246μs完整Norm、固定byte假设 |
| 向量DDR有效吞吐 | 68.23336585365854e9 B/s | reference DDR，不是目标HBM |
| 矩阵DDR有效吞吐 | 140e9 B/s | 200GB/s×0.70工程设定，未独立拟合 |
| 13组reference合计 | 11897.602619368332μs | 源公式复现，XML B16 cache约11897.602619368336 |
| 最大单组reference误差 | 约4.55e−13μs | 复算误差，不是预测精度 |

Norm 第二次为锚点复用；其它组派生/工程重建，拟合权重0；19数学节点不是19独立样本。唯一Excel公式为B16=SUM(B2:B14)，B15硬总计；AV→PV映射一次，Attention/FFN/BLOCK不重复相加。

目标 engineering priors 冻结：eta_pe=eta_vec=.90，eta_hbm_matrix/vector=.70，eta_sram=1，eta_local/mesh/inject=.70，rho_sram=.90，launch=1000ns/setup=0，quant=4 eq-op/元素、acc convert=2。Plan.new_plan().launch_ns 仍为0，launch consumer尚未兑现。target calibration/heldout 仍 not_available，不迁移 TX8 BF16/DDR 参数到 INT4/HBM。

## Next Step

以下是明确优先级及依赖；不是任意阶段/额度上限。接手后先核 main/source digest，复用25项和workspace成果，不重复 compact/fractional/cache 开发或无数据重拟合。

1. P0：真实编译有界 K lifecycle/busy union。读当前 affine/general-offset-analysis 与旧 grouped oracle；需要从真实 Plan geometry/lifecycle/LP active sets 合成有上限的 piecewise affine descriptor，不能手贴 fixture map。先单 Core OS→offset/tail→多 Core/WR fence→外部容量/取消；严格止于外部事件，byte/ops/slots与正率busy守恒。通用 state≤3J，deadline/LP guard cells可能爆炸，256 cap不能按每 output无限重置。保留独占3%/并发与busy5%、E≤8/16；生产未通过前 failed/not_run。
2. P1：wscllm真实Online E2E与兼容输出。准备源码已保存，typed loader已修；重建当前Online/两ABI，运行2-rank实际桥接，核schema2全rank/revision/base+workspace、原request first-token/完成/key/window、正workspace Plan、COMPs release/terminal与进程终止。必要时选合法显式fusion实验，不能为了证明强改生产策略或伪造workspace。然后完整workload业务/CPU/RSS；此前7/7与Python discovery不替代E2E。
3. P1：在基础E2E明确后逐仓 workspace rollout。wscllm-LRU先读已保存审计：保留 whole-session LRU/remote_store/pool、永久overflow异常、history四路径与事务rollback，hold阻塞restore不得改remote debit/发transfer，release后一次restore；独立测试store-before-restore/容量epoch。FACE/FACE-LRU保留selector composition/score、p_context/ready/history/pool，再JOINT live import/fork/overlay/network join；未验证前不移除frozen projection guard。其他四仓本轮只是同源发布/ABI配置验证。
4. P1：剩余输入/成本。launch/setup按真实区域/层/pass时序收费依赖有界生命周期；向量参数按consumer-Core首次读取与完整切片fit复用、Norm/RoPE长行两遍读/存储转换、residency/readiness/TP局部区间仍未完整。不要把启动lump在开头，不重做已完成 fractional κ、strict INT4转换、compact batch。
5. P2：性能。最终flat owner probe J=32中/空P95约19.048/19.779μs，满credit55.694μs仍超50；只是完全对称 synthetic 13-domain fixture，不证明异构J32。一般LP仍可能256-pivot fail closed；完善表示/active-set，不抬上限。pre-flat N30 cold C++P95仍4.055ms失败、CABI miss1.902ms/hit12.119μs仅该fixture；新的flat source未重测cold N30。保留源码/二进制身份和caller计时，cold includes创建/解析/resolve，不能与hot compile或cache hit混列。以后验证三profiles/大span/FFN/J32/全workload≤10%开销，别用一组均值外推全域。
6. 无新增独立测量时TX8 reference和target priors保持冻结。目标数据缺失按not_available记录；source复现、reference、粒度、硬件精度、五仓业务、host性能分别升级证据。

## Important Decisions

1. 根架构源文决定硬件；设计文档决定执行/ABI；13物理域、PRIVATE、保守COMPUTE_GATE、最多128 GEMM候选与原tie规则保留。die/是唯一公共源，五仓固定摘要。
2. 工作量为整个epoch总量，repeat_count仅审计；数学/physical payload/D÷eta与时间预算区分。owner为node/execution/generation，request只追踪，不因拆分增加份额。
3. 整数Plan v2、非零fraction Plan v3、snapshot迁移和fixed C ABI v2布局保留。新bridge JSON v2与Plan/Chakra/C ABI版本分开。canonical candidate seal延迟只在内部、原校验不省。
4. LP精确同构压缩不等于cycle skipping；积分仍逐真实活动、busy仍正率并集。极端work/Q×multiplicity可能数值overflow fail closed，未证实旧solver可处理该极端；不能宣称任意参数J32精度/预算通过。
5. workspace是容量增量，不是KV reservation或HBM访问；Python容量权威+全rank绝对revision视图，发射前lease，释放一次。原KV/LRU/网络/selector/journal未互换；只在wscllm实施。
6. v2 ack需要字面success=true；legacy Success/Skipped watch枚举不扩展，cancel typed指标另列。未知submit是否接受不能静默retry；graph dispatch失败仍fatal。
7. generic TFLOPS不能由INT4 TOPS替代；typedrate可有限正浮点，精确byte仍uint64。coarse旧roofline getter fail closed。
8. 4096/64MiB retained cache上限不是RSS。profile gprof粗采样只是热点定位；性能identity按前后源隔离，不把pre-flat样本当最终flat证明。
9. 独立 affine/prototype与未执行review分目录；no epsilon repeat、no GMP生产、no逐tile fallback。当前源码交接可完成，不代表模型或五仓整体工程验收。

## Files to Continue

- 公共：src/Execution.cc、Compiler.cc、Mapping.cc、Plan.cc；include/coarse_die/Plan.hh、Execution.hh；新tests/runtime_submit_transaction_test.cc、runtime_owner_symmetry_test.cc、workspace_metadata_test.cc、扩展estimate_driver_test.cc。
- wscllm C++：WorkspaceAdmission.hh、Sys、Workload、DieModelAdapter、CoarseDieRuntime、DecisionMailbox/Bridge、ParsedGraphBatch、GraphBatchCommitter、main_online、MetricCollector；对应新增/扩展测试。其他四仓coarse vendoring已同步，不误以为workspace也已同步。
- wscllm Python：session_kv_manager.py、online/{online_scheduler_base,wsc_llm_online_scheduler,decision_bridge,online_service,test_workspace_admission}.py、wsc_llm_scheduler.py、generate_wsc_llm_trace.py及typed硬件/train测试。
- 真compiler oracle：validation/compiled_gemm_oracle.py、grouped_gemm_oracle.py、compiled_gemm_oracle_driver.cc。新失败输出保存，不以measurement成功当精度pass。
- 研究：analysis/handoff-runtime-20261014/affine；旧bounded-lifecycle-20261014保留；新online-preparation和LRU audit。unexecuted-review目录先读README再运行。
- TX8：calibration/report.md、reference configs、tools/calibrate_tx8.py；本轮未重生成伪造新测量。
- 真实日志/源码/binary identity：evidence/handoff-runtime-20261014，旧evidence保持原样。

## Tests

- 公共最终source默认targets build，25/25 CTest，无跳过；保留已通过22项，新增workspace metadata、submit transaction、owner symmetry，estimate driver增加sealed/trial语义验证。最终日志ctest-die-cohort-flat.log。
- wscllm最终digest Online及所选targets重建，7/7 CTest，无跳过。真实FFN workspace blocked/free/前置lease、正服务取消、served util、Skipped、slot/lease一次释放、dependent issue与die_cancelled_comp_count已断言。
- 最终Python online discovery106/106；typed scheduler/train35/35且显式ABI无跳过。交接前更新后的五仓两ABI/config执行范围、每仓数量/日志在verification.json；没有继承旧17/17为本轮五仓业务通过。
- compiled实际C++ driver对正确grouped oracle仍failed，账本checks全部通过；E8/E16 not_run/unavailable，阈值没变。
- 128³/28候选/pre-flat各30样本：C++ cold P50/P95=2.225458/4.055090ms，hot compile1.729552/2.260951ms，estimate-alone241.523/439.331μs；CABI miss1.694489/1.902401ms，真实same-model hit10.325/12.119μs。baseline c47 C++cold3.496709/4.250943ms、miss3.405632/4.333397ms。冷cpp失败，CABI预算仅此夹具通过；RSS分别13056/9344KiB。metadata给精确测量SHA，不套用之后flat Runtime。
- 最终flat n30 synthetic J=1/8/32 probe：最大J32满credit55.694μs失败；中/空19.048/19.779μs；数值/复杂度/覆盖单列，非全域工程证书。旧LP J8边界曾256 fail、J32中约38.5ms；保留原始对照。
- root实跑nested/固定精度独立checker通过上述限定范围。额外review oracle/probe未运行，已保存不算pass。
- 本轮未跑production sanitizer、malloc failure注入、目标硬件留出或实际Online E2E。TX8 historical source/cache复用，不作为本轮重新测量。

依赖需重建：CMake4.4.4（本地pip安装于work/tooling/cmake/cmake/data/bin）、GCC14.2.0、Protobuf3.21.12（work/deps/protobuf），Python3.12.14/protobuf4.25.8（work/deps/python-protobuf）。此次protobuf-all-21.12.tar.gz SHA256为2c6a36c7b5a55accae063667ef3c55f2642e67476d96d355ff0acb13dbb47f09，与上轮不同tar包，不混身份。configure和build都设PROTOBUF_FROM_SOURCE=True、PATH含protobuf/bin、CMAKE_PREFIX_PATH指该prefix。aggregate原repo/backend/frontend三个subdirectory，Release/congestion_aware/library backend；generated Chakra在build tree，不提交。

## Blocking Issues

- 生产真K readiness/K-final store/retire有界曲线未闭环，busy/并发误差失败；prototype不覆盖通用offset/tail/多Core/WR/外部事件。
- launch/setup/instance时序、向量参数/长行转换和完整residency/readiness/TP语义仍缺。
- workspace只有wscllm定向验证；真实Online E2E未启动，四仓workspace/LRU/pool/FACE/JOINT及全旧key/window业务未验收。
- C++ cold P95、J32 full-credit/异构仲裁与完整online开销仍未达标或未测。
- TX8无新原始日志/独立目标数据，不能辨识目标INT4/HBM/NoC/启动超参。
- 合并/远端确认失败会阻止代码交接；未执行项和expected failed精度独立如实记录。

## Handoff Notes

本轮收到明确停止/交接指令，已经停止新开发任务；保存全部有效源码、测试、精确样本和有边界的研究。未运行的Online准备、未执行审查与已知错误scratch都标注，不留聊天独占的关键后续判断。临时依赖/二进制、gmon缓存和大型生成产物不提交；保存可重建的源码与精确身份。

请以evidence/handoff-runtime-20261014/github.json读取真实源码SHA/PR/CI/merge信息，再fetch/ancestry核对main。只有代码和本HANDOFF进入main并确认后才可以直接新建另一账号任务接棒；否则NOT READY FOR HANDOFF。合并后的代码交接不使工程整体状态从NOT ACCEPTED变成通过。

完成GitHub交接后本Session立即停止。下一账号先读AGENTS/PROJECT/本HANDOFF，再按Next Step推进。不要重复已完成模块，也不要把原Session忽略目录或其它账号隐藏聊天当依赖。
