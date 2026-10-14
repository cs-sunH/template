# Codex Cloud 开发交接状态

记录日期：2026-10-14（UTC）。用户最新指令为停止启动新任务，在当前最小任务完成后保存、合入 main 并停止。本 Session 已完成在途 marker teardown 收尾，未继续启动功能开发。下一账号收到新的开发指令后按 Next Step 接续；停工指令不是永久项目限制。

## Current Status

**公共/五仓定向回归通过，代码交接待远端合并核实；工程整体仍 NOT ACCEPTED。** 本轮从已合并 PR #6 的 main `037f87fa6c2e9a6a0e0c665598f7bcad0e41b87f` 开始，复用既有 compact batch、fractional κ、Plan v2/v3、C ABI v2、缓存、事务回滚、owner LP 对称商及 wscllm workspace v2，不重复这些成果。

本轮公共默认构建/27 CTest 通过；新增 exact Fraction LP 对照、每 K OS access counter 基础（尚未接入生产）、端点可选 ready delay。五仓修复 coarse ControlMarker 的 scalar COMP 错误路径，保留 legacy 时序与各仓 carrier/pool/join，处理 live prefix/取消/teardown。五仓 Online 及所选 targets 构建成功，最终定向 CTest 共 16/16（3+4+3+3+3）；五仓两 ABI 最终构建成功，显式配置测试 83/83（20+15+18+17+13），无跳过。21 个 canonical 文件发布 digest 为 `f5e49a82f534db77623eada68657d44c63830afd076f152a8262f5ba864cbde3`。

真实 wscllm Python↔C++ 两 rank Online E2E 本轮已运行并成功：20 typed COMP、正 workspace FFN、3 个 schema2 ack、31 成功 terminals、REQUEST_COMPLETE、进程均 exit 0。该 E2E 早于最后 teardown 修正，实际二进制身份保留，不冒充最终源码 E2E。其他四仓 workspace v2 尚未移植，完整业务 E2E 未跑。

实际 compiled E5 oracle 新重跑仍 failed：独占 finish 约 3.827%、HBM busy 约 124.951%，原 3%/5% 阈值不变；生产 E8/E16 unavailable/not_run。新 geometry-generated affine return/固定精度 lowering 与 solo shared-HBM 完整研究有明确证明边界，尚未接入生产；双 owner 完整发现仍无法在全局 256 cap 闭环。TX8 没有新测量或重新拟合，host performance 没有新最终源证书。

本轮 [PR #7](https://github.com/cs-sunH/template/pull/7) 的源码提交为 `77af39781485af527339051b580f5250c93f004f`；[源码 CI run 38525345962](https://github.com/cs-sunH/template/actions/runs/38525345962) 公共构建/测试和交接检查均 success。GitHub 入口及真实源码交付 SHA/已完成 CI 保存在 `evidence/handoff-lifecycle-online-20261014/github.json`。只有所链接 PR merged 到 main，fetch/ancestry 确认包含源码与交接文档后才能安全代码接棒；此前为 NOT READY FOR HANDOFF。合并不等于模型/工程验收。

## Integration Branch

- 共享集成分支：`main`；本轮工作分支：`codex/die-lifecycle-online-20261014`。
- 上轮 [PR #6](https://github.com/cs-sunH/template/pull/6) 已 merged，merge/main SHA 为 `037f87fa6c2e9a6a0e0c665598f7bcad0e41b87f`。上轮工作与失败证据原样保留。
- 本轮源码、测试、研究、Online 证据与本 HANDOFF 一起提交为 `77af39781485af527339051b580f5250c93f004f`，目标 main 的 [PR #7](https://github.com/cs-sunH/template/pull/7)。源码 CI run 38525345962 两 jobs 均 success；真实 API 结果见本轮 `github.json`、`source-ci.json`。后续仅补交接元数据，不 force-push。最终 merge SHA/本元数据文件自身 commit 不提前猜测，从 PR/fetch 获取。
- 下一账号 fetch 最新 main，核源码 ancestry、21 文件 source digest 和证据索引，再建独立工作分支。忽略目录 work/、工具/依赖/二进制不跨账号共享，保存的源码和命令可重建。

## Last Verified Base

本 Session 接续开发前同步并核实 main `037f87f`，读取 AGENTS、PROJECT、HANDOFF、根架构/设计文档及相关代码；新分支以该已合并基线创建。交接收尾再次 `git fetch origin main`，远端仍为 `037f87fa6c2e9a6a0e0c665598f7bcad0e41b87f`，未发现外部并行提交或无法解释的修改；合并前继续核实，不覆盖他人提交。

根两份设计文档/TX8 工作簿未修改。之前的 `evidence/handoff-runtime-20261014`、`analysis/handoff-runtime-20261014` 及更早证据保持原样。本轮新增证据在 `evidence/handoff-lifecycle-online-20261014`，可重建研究/实验源码在 `analysis/generated-lifecycle-20261014`；两树 ARTIFACTS.json 是保存文件的 SHA256 索引。

## Completed

### 公共端点与 LP 验证

- `src/Execution.cc`：endpoint JSON 可选 `initial_delay_ns` 使用现有 exact uint64 parser（非负整数或严格十进制字符串），省略默认0；写入 endpoint Plan.launch_ns，复用 Runtime ready gate，delay 内不服务/不记 HBM busy。只有显式字段才加入 summary，保留旧省略字段的 canonical summary/fingerprint 行为；snapshot import 校验 launch 与 endpoint delay 一致。不是给所有真实 COMP/COMM 或每 region 实现 launch/setup。
- `tests/endpoint_allocation_test.cc`：pre-ready/ready tick、snapshot/replay/fork、部分取消、registry 清理及非法 float/negative/bool/overflow；旧省略字段 summary 无新 key。端点 delay 加法沿原 checked 路径，未改变 ABI layout 或旧 Plan 版本。
- `tests/owner_lp_oracle_driver.cc` + `validation/owner_lp_oracle.py`：真正 public Runtime 输出对照独立 stdlib Fraction vertex LP，5 static cases 和13实际快照，完整/商 LP、PRIVATE、不同 Q、同 fair owner 跨 execution、顺序、arrival/completion、capacity/cancel/snapshot。规模最多5 service variables 加 fairness floor、dimension≤6；不是96活动、pivot极值/异构J32性能证书。
- `tests/runtime_owner_symmetry_test.cc` 正式加入 phase-offset arrival/member-change，真实 finishes=5s/6s、HBM busy=6s/shared=4s。上轮 scratch 的7s误预期不再作为依据；上轮回滚/对称商实现未重做。CMake 注册新的独立 LP CTest。

### 每 K OS access counter（完成基础，未生产接入）

- `src/GemmOSAccessCounter.hh` 是内部纯 O(1) 计数器：调用方提供实际 m/n/k tail、first/final K、存储 dtype 和 matrix path；OS only、WR 拒绝。没有 mapping 选择、L/C/S readiness、buffer ownership、time 或 cycle skip。
- 支持 FP32/BF16/INT4/INT32、target INT4 PE 与 strict FP32 Vec；checked u128 中间积/和、u64 byte/ops 输出，Vec 用既有 EquivalentWork whole-u128/exact dyadic fraction，不能把大 whole 与 tiny fraction 合成损失精度的浮点。
- 每 K input HBM/load SRAMwrite、native/converted execution inputs及4B scale、target INT32 partial、非 first accumulator read/每 K write、仅 final cast/store/HBM output；LOCAL 只为 Compute SRAM R+W，与当前 Mapping action ledger 对应。INT32 storage4B、strict cast1eq/元素、target quant κ、final INT32 cast1eq/元素。
- `tests/gemm_os_access_counter_test.cc` 手算 odd INT4 packing、first/middle/final、转换、fractional κ、非法/overflow；独立逐 tile 小和对拍两 profiles 所有可行 OS 固定候选（含INT32）；whole=2^54/fraction=2^-20 组合拒绝损失。正式 CMake 单项及最终27套通过。
- **Mapping.cc/Runtime 没有调用此 header**。下一步从此构造真实 consumption 类 descriptor，不能宣称当前 compiler 已有真实 K lifecycle 或精度通过。

### 五仓 ControlMarker、指标及拆除

- 本轮真实 Online 失败暴露 `ControlMarker -> issue_compute(1,1)` 进入 coarse scalar guard。五仓 `Workload.cc` 使用专用 `issue_control_marker()`；LocalHbmBandwidthModel coarse 分支发唯一 canonical `owner_kind=die`、1B read、route=false、initial_delay=local_mem_latency。无 typed math/vector/int4 ops、无 workspace lease。普通 scalar COMP/DIE_COMPUTE 仍拒绝；legacy 回落仍原 `issue_compute(1,1)`、100ns+1 drain timing，原无 HBM 路径保留。
- 五仓保留各自 JobKind ordinal（新增 CONTROL_MARKER 在 KIND_COUNT 前）、carrier、restore/pool分类、旧 join/abandon/Workload callback。JOINT 的 virtual pipeline marker 也改用该入口，原 pipeline_sink cookie 恰好移交一次，未解除 coarse projection frozen guard。
- `CoarseDieRuntime::{hh,cc}` 新 read-only `query_progress_at(handle,now)`：copy public Runtime、advance copy、读 per-handle progress；不调用 export_snapshot，不改变 live clock/service/counters/slots，不发布 graph callbacks；stale/reverse fail closed。`compute_bytes_served()` 原 COMP 兼容视图累计已关闭 marker prefix并采样仍活跃 marker，COMM/RESTORE/POOL 不挪入。
- 正常 marker explicit cancel 回调记录已服务 prefix一次；**teardown** 先 query 精确 prefix、从本地 map退休对应记录、`cancel(handle,now,false)` 并回收原 cookie/join；不能用 notify=true 获取前缀，因为它会分发同 tick已完成 marker或兄弟 terminal，造成析构时错误成功。LRU/JOINT map删除顺序按原取消无通知合同保持；所有最终测试覆盖 partial prefix与fully served/callback pending sibling suppress。
- MetricCollector 未重写；coarse Runtime 全局 payload为权威总量，COMP兼容 getter/`die_compute_*` 的 legacy aliases可能重叠，不能相加当第二份流量/分母。只修 marker view，不冒称所有 legacy指标已覆盖真实coarse全部类别。
- lifecycle测试包含延迟/一次完成/零 math/分类隔离/live采样无副作用/取消/teardown，JOINT追加 zero-network-byte TileTransportAdapter pipeline sink。base CaseI直接 Workload dispatch验证 shared Runtime，不证明生产单Compute gate允许 COMP+marker自然并发；CaseL正常 availability 路径验证 marker+COMM、slots0及分类分离。
- 所有五仓 Online+定向 targets成功构建；3+4+3+3+3 CTest通过，最终21文件发布与双 ABI/config验证通过。marker专属adapter snapshot恢复、四仓完整业务/RemoteReadExecutor E2E未跑。四仓 workspace v2仍未实施；原 LRU/pool/FACE/JOINT策略不互换。

### 真实 wscllm Online E2E

- 生产 Online C++ + Python `online_service.py`、官方 config loader/plan materializer、`derive_20_first_30_seconds.py` provenance gate；2 ranks P/D，1 request=64 prefill/1decode，tiny1layer hidden128/FFN256/heads4/vocab512，144GB/rank。合法显式 `SH_FIRST_TOKEN_SPLIT=1`、workspace auto、online-validate0仍保留必需 workspace preflight；没有 monkeypatch manager/policy或伪造 wire。
- 3 exchanges/成功 schema2 ack，全rank/seq/batch/revision echo和base+workspace≤capacity；final revisions [20,20]、active [0,0]，每stage10次Plan acquire/release。captured20 COMP specs以同 CABI重编全通过；prefill/decode FFN workspace分别65536/1024B，attention/FFN两region都存在。
- first_token_ns=completion_ns=36185，prefill25294/decode10890；31成功terminal，0Skipped/other，accepted1/completed1/active0/pending_alarm0，metrics一致性通过。rank1Die2B来自join+first-token markers，Die math=0，两rank typed COMP服务正数。
- wire只在phase-drain观察 active0；不声称它直接捕获正live hold，相关真实live lease测试在lifecycle夹具。strace加开销，不是CPU/RSS/吞吐证书。
- E2E Online SHA `9bc7d51869c821b7748b11955bbc1988a1a839cb1bb4b79377568dfa77b72ec4`，Python/C++coarse CABI同 SHA `49e596bc661b410ac3ef7499fc7f68af4065e3946be2dbd2b36d224f859ef3cd`。provenance中的gitHEAD037是当时dirty tree基线，不是声称已commit该运行源。正常完成路径成功发生在最后teardown修改前；最终focused验证覆盖新修复，没有重跑整次E2E。
- scalar marker cpp134/python1失败与更早missing provenance setup失败均保存；完整decodedwire、日志/指标、原始strace SHA/bytes与重生成脚本保留。大型strace不提交，work中仍保留；详见证据README与analysis/online-reproduction。

### 生命周期研究（非生产证书）

- `analysis/generated-lifecycle-20261014/discovery/` 从 geometry、shared HBM grouped LOCAL-Q LP和离散2input/2output queues生成 exact symbolic initial-offset propagation/guard；不从旧trace抽类，不用epsilon相等或三点拟合。全局256 cap不按output重置，只跳守卫内精确 middle-K identity；外部arrival前停止。
- solo/同步双owner65³K4097：120 probes、跳595/1190waves，对独立design-aligned grouped discrete oracle实际C++selectedfixture误差<1.2e-15s。offset1/20小场景完整需要1950probes（研究诊断cap4000），normal256 failedclosed，独立误差<9.55e-15s；不能抬生产上限冒称闭环。
- M15985/N63937/K4097在256内生成26 guarded one-output returns、middle跳842waves，但完整仍failedclosed；K64000001仍failedclosed，middle跳约14M也不能替代output powering。
- 保存首return：1state、11ledgers、36代表steps、297guards（202ge/95gt/0eq），a=849475/2156112，b=6183347594952049573/18255628780978176000。strict domain=[65101124099499917/130771586319360000,73426828337688317/130771586319360000]；完整初/末queue/cursor/guard/账本在first-return-certificate.json。初态全full-MN可996次，end后追加995，997跨firstnonfull64935。与旧fixture坐标不同，不能混用旧b。
- root三exactorbit points bounded replay，每次260expanded events、queue/cursor及11ledgers精确；复用新geometry/ratehelper，独立于旧trace但不是独立LP算法或全256证书。
- `interval/` standardC++17 64significand LD、每primitive nextafter outward、固定256bit input limbs，无GMP；flags -frounding-math/-fno-fast-math/-ffp-contract=off。Fraction独立重建全部297guards/严格边界并验证1/10/100/996state与11ledgers包含；996宽3.9551695e-16s=3.95517e-7ns，integer stop13099179518ns可123/不可124，20compositions搜索。10^6/10^12仅numeric-only，不延长tail/cancel覆盖。
- `solo-shared-hbm/` exact shared L/S HBM+OS完整solo生命周期：65/65/4097为72probes vs1310expanded，617waves/4outputs跳过，全部账本/time exact；K64000001/2000010output cases均72，大N6400001为37/100001outputs；240 small full/tail、K1/2outputslots/store-retire通过。只固定strictFP32单Core、无外部事件；Fraction大整数offline不证明production可用/性能。`selfcheck` synthetic small geometry重构与equal-progress范围已标记，grouped accelerated vs自身expanded不冒充hardware验证。
- independent-server-maxplus原型没有sharedHBM，保留有效数学历史，不能作为生产默认。新研究不变更当前Mapping/Runtime，不新增公共第三方依赖；未完成部分完整保存，不丢失。

### TX8 校准（冻结，未新拟合）

工作簿 SHA `eba181c80a2d51bd5a604bad6928610ea5039824b4ec4d5d9bd023ee4bda52ee`。原九条日志已移除，无独立重复/预热/heldout；本轮没有新目标硬件数据。

| 参数/结果 | 值 | 证据边界 |
|---|---:|---|
| eta_compute_reference | 0.9079551923877489 | 条件整个FFN3363μs，固定128T/1T |
| eta_vector_memory_reference | 0.3411668292682927 | Norm246μs reported anchor、固定bytes |
| vector DDR有效吞吐 | 68.23336585365854e9 B/s | reference DDR，非目标HBM |
| matrix DDR | 140e9 B/s | 工程200GB/s×.7，非新拟合 |
| 13组合计 | 11897.602619368332μs | XML B16约11897.602619368336 |
| 最大单组复算误差 | 约4.55e-13μs | 源公式复现，非预测精度 |

Norm复用锚点；其余派生/工程组weight0；19math节点不是19独立测量。B16=SUM(B2:B14)是唯一公式，B15hard total；AV→PV一次，Attention/FFN/BLOCK不重复相加。

target priors继续冻结：eta_pe=eta_vec=.90，eta_hbm_matrix/vector=.70，eta_sram=1，eta_local/mesh/inject=.70，rho_sram=.90，launch1000ns/setup0，κquant4/κacc2。真实region Plan.launch仍0，consumer未兑现；endpoint delay为独立兼容语义，不算目标启动校准完成。target calibration/heldout not_available，不能迁移BF16/DDR参数到INT4/HBM。

## Next Step

按依赖推进，以下不是人为数量/阶段限额。先核最新main/source digest和证据；复用27套、marker与base workspace/E2E，不重做cache/fractional/compact/LP对称商或无数据重新拟合。

1. **P0真实有界K生命周期**：以新GemmOSAccessCounter→actual Plan geometry与storage/native/conversion ledger→first/middle/final/full/tail/cohort类（预算≤32）→generated LP/deadline guard+queue/output-slot retire→固定区间lowering/外部stop→公共Runtime落地。先单Core OS solo/sharedHBM完整闭环，继而offset双owner+tail，再multiCore/WR。新generator的大双owner不能仅靠已生成首return交付：还要守卫表示/外层return powering/边界切换，且整次256不能每output重置。保存 exact byte/ops/LOCAL及positive-rate busy union，不做逐tile生产fallback/epsilon phase equality，不提高96活动/256pivot上限。
2. **P0验收随实现走**：生产一旦接入，对真实compiled driver重跑E5/可用E8/E16、原独占3%/并发及busy5%，包含capacity/arrival/cancel/snapshot/fork/namespace及止于外部事件。研究checker各自pass不等价于生产证书；未接入层级明确not_run。OS counter是工作量基础，不能把timing简化成其合计。
3. **P1 workspace五仓推广**：base tiny Online已成功，下一步其多request/memory压力/取消窗口/旧record key及全business验证，然后WSCLLM-LRU。读旧analysis/handoff-runtime-20261014/LRU audit与本轮marker审计：保留whole-session LRU/remote_store/pool、overflow/history路径/事务rollback；workspace仅fit hold，blocked restore不debit remote/不发transfer，release后一次restore，测试capacity epoch/store-before-restore。再FACE/FACE-LRU selector composition/score/p_context/ready/pool，最后JOINT liveimport/fork/overlay/networkjoin，保持frozen projection guard直到真正验证。marker完成不等于workspace协议已推广。
4. **P1输入与成本余项**：真实region/layer/pass launch/setup时序依赖生命周期；endpoint delay仅marker使用，其它endpoint延迟需独立审计，不能宣称generalCOMM setup已补。vector consumer-Core首次参数读/完整切片fit、Norm/RoPE长行两遍读与存储转换、residency/readiness/TP局部区间仍缺。兼容指标alias不能加总，未来采样优化保留readonly语义。
5. **P2准确身份的性能证据**：无新当前源CPU/RSS。旧finalflat J32 full-credit P95=55.694μs仍fail50；中/空19.048/19.779μs仅完全对称fixture，异构未证。旧preflat n30 C++cold4.055ms fail2ms；CABI miss1.902ms/hit12.119μs仅该fixture，不套用当前源。待算法落地后测三profiles/大span/FFN/J32/真实Online≤10%开销，caller timing含cold创建/解析/resolve，cache hit/miss/hostRSS分别列身份。
6. 无新独立测量时TX8reference/targetpriors保持冻结。遇新数据先验证粒度、dtype和测量来源/heldout，再谈可识别超参；架构源自变量/13物理域/候选覆盖继续按原文，不能借校准抵消缺失生命周期。

## Important Decisions

- 根架构源文决定硬件，设计文决定执行与ABI。die唯一公共源，五仓固定digest；13物理域、PRIVATE、保守COMPUTE_GATE、最多128GEMM候选及原tie规则保留。
- 工作量是epoch总量，repeat_count仅审计；source math/physical payload/D÷eta/time budget分开。owner=node/execution/generation，request只追踪；不因拆分增份额。
- 整数Plan v2/nonzero fraction v3、snapshot迁移、C ABI v2固定布局、cache/retained limits与candidate内部delayed seal保持；endpoint optional JSON不改变旧省略字段，bridge v2与Plan/Chakra/ABI不是同一版本。
- LP精确同构商不等于cycle skipping；busy按positive-rate并集。bounded guard discovery、固定精度区间与实际生产接入/精度验收分层，no epsilon repeat/no GMP生产/no逐tile fallback；不以无限重置cap隐藏线性开销。
- workspace是增量容量，非KV reservation/HBM访问；Python权威全rankabsolute revision，发射前lease/释放一次，目前只base wscllm。未知submit接受不能静默retry，graph dispatch异常仍fatal cleanup，非graph rollback证书。
- ControlMarker保留1B物理占用与本地ready delay，不记typed math/no workspace；legacy+101保持。正常cancel与teardown区分，teardown针对handle无通知，不触发全体finished callback。只用copy Runtime采样。
- genericTFLOPS不能由INT4TOPS替代，typed rate可正浮点，精确bytes为uint64；旧roofline coarse getter failclosed。metrics alias不二次汇总。
- 代码安全接棒与模型/工程验收分别判定；保存所有有效部分成果和失败证据，未跑不写pass。用户stop后只保存/验证/合并，不继续新开发。

## Files to Continue

- 公共：`src/{Execution,Mapping,Compiler,Plan}.cc`、`src/GemmOSAccessCounter.hh`、`src/EquivalentWork.hh`；`tests/{gemm_os_access_counter,endpoint_allocation,runtime_owner_symmetry}_test.cc`、`tests/owner_lp_oracle_driver.cc`；`validation/owner_lp_oracle.py`、compiled/grouped oracle与真实driver。旧workspace/rollback/candidate tests继续保留。
- 五仓：各 `astra-sim/workload/{Workload,LocalHbmBandwidthModel,CoarseDieRuntime}.{cc,hh}` 与 `execution_driven/tests/die_execution_lifecycle_test.cc`；JOINT `issue_pipeline_control_marker`/TileTransportAdapter/RemoteReadExecutor；原variant joins/pool callbacks必须逐仓读。FACE两README更新，其它原legacy说明未统一重写。
- base workspace：WorkspaceAdmission、Sys、DieModelAdapter、DecisionMailbox/Bridge、ParsedGraphBatch/Committer、main_online、MetricCollector；Python session_kv_manager、online schedulers/service/bridge、config loader/trace materializer与现有 tests。此前v2完整设计见旧HANDOFF提交/PROJECT记录，本轮没推倒重做。
- 本轮研究/重建：`analysis/generated-lifecycle-20261014/README.md`及discovery/interval/solo-shared-hbm/online-reproduction；旧affine和unexecuted-review按其边界复用，不把旧trace拟合当新generated证明。
- 校准：`calibration/report.md`、reference configs、tools/calibrate_tx8.py；本轮无新生成数据。最新真实证据及GitHub身份入口：`evidence/handoff-lifecycle-online-20261014/README.md`、verification/github/source-ci JSON。

## Tests

- 本轮最终公共configure/default all targets build exit0，27/27 CTest，无skip，17.11s；`public/final-{configure,build,ctest}.log`。owner LP包含5 static+13 Runtime snapshot，counter为新正式单项。此前26及新counter1日志保留，不累加当第二轮覆盖。
- 五仓最终 Online/selected targets build exit0；WSCLLM、FACE、FACE-LRU各3/3（Lifecycle/Adapter/LocalHbmDieMemory），WSCLLM-LRU4/4（Lifecycle/Adapter/LocalHbmBandwidthModel/LocalHbmModel），JOINT3/3（Lifecycle/Adapter/LocalHbmDieMemory）；合计16/16无skip。partial/full pending teardown与legacy timing均通过。日志、final binary/sourceSHA在integration/verification索引；不声称whole业务覆盖。
- 21文件sync/check pass；每仓coarse_die_cabi+die_model_cabi最终增量构建exit0，显式ABI config83/83无skip，真实cmd/binarySHA在five-abi-config.json。JOINTlegacyABI与其它不同为原仓实现，不要求相同binary。
- 本轮wscllm production Online E2E exit0/0，protocol/lifecycle/positiveworkspace/metrics通过，source identity与时间先后如Completed；failed尝试保留。没有重跑final teardown后wholeE2E或其它四仓业务。
- 新compiled实际driver oracle exit1 expected精度失败，所有服务账本checks通过，E8/E16not_run；raw/source identity保存。root generated return replay/interval checker、solo研究selfcheck按限定范围实跑通过，完整offset discovery仍failedclosed，不作为CTest pass替代。
- PR #6 public25/wscllm7/Python106/typed35为已合并历史，本轮未重跑Python整套，不冒称新测试。没有新sanitizer、malloc fault、硬件heldout、hostCPU/RSS/performance测试。
- 官方 queue/sidecar CSV 的 CRLF 为真实 provenance bytes，原样保存；默认 diff whitespace 检查会报 CR，采用 `git -c core.whitespace=cr-at-eol diff --check` 核实无其它空白错误，不归一化已记录 SHA 的证据。
- 编译仍有既存 `__int128` pedantic/Plan.cc temporary possibly-dangling等GCC warnings，build exit0；本轮没有独立确认所有warning或新增warning修复任务，未将warnings当成通过的内存安全证书。

重建命令：`cmake -S die -B work/build-die -DCMAKE_BUILD_TYPE=Release`，`cmake --build work/build-die --parallel 2`，`ctest --test-dir work/build-die --output-on-failure`。五仓aggregate CMake enable_testing后 add_subdirectory(repo)、analytical backend、analytical frontend，`ASTRA_SIM_ROOT`指定variant；Release/congestion_aware/NETWORK_BACKEND_BUILD_AS_LIBRARY=ON。`analysis/.../integration-reproduction/` 保存aggregate与本轮验证脚本供重建，工具prefix按新环境调整。

本环境CMake4.4.4在work/tooling/cmake/cmake/data/bin，GCC14.2、Python3.12.14；Protobuf3.21.12 prefix work/deps/protobuf，Python protobuf4.25.8 work/deps/python-protobuf。configure/build同时设置PROTOBUF_FROM_SOURCE=True，PATH含protobuf/bin，CMAKE_PREFIX_PATH指prefix。tarSHA `2c6a36c7b5a55accae063667ef3c55f2642e67476d96d355ff0acb13dbb47f09`；generatedChakra在build tree不提交。Online runner依赖strace且保存旧attempt。

## Blocking Issues

- 生产真实K readiness/final store/outputretire未闭环，finish/busy精度failed，双owner大场景256发现仍failedclosed；counter和研究尚未接入，不覆盖generaloffset/tail/multiCore/WR/外部cancel。
- launch/setup/region instance、vector参数/长行转换/residency/readiness/TP仍缺；delay字段只兑现marker兼容，不把其当所有真实任务成本。
- base tiny Online成功但多request/压力/旧业务全域未证；其它四仓workspace/restore/pool/FACE/JOINT及完整Online未验收，marker专属snapshot未单独测。
- 当前源C++cold/J32异构及全Online开销未达标或未测；研究Fraction性能不代表production；GCC既存warning尚未全部审查。
- TX8无新raw/独立目标heldout，不可辨识目标INT4/HBM/NoC/launch超参。
- PR未merged、push失败、必要测试阻塞或远端未确认将阻止代码交接；上述expected工程失败独立如实保留，不以失败隐瞒或以merge误报验收。

## Handoff Notes

本轮收到明确停止/交接指令，所有在途子任务已收尾冻结，保存有效代码/测试/计数基础/研究/校准状态/Online失败与成功证据。重要判断和下一步依赖均入仓，不依赖其它账号的隐藏聊天。没有丢弃有效修改或覆盖外部提交；二进制、依赖缓存、大型可重生成strace/重复descriptor不提交，保留来源身份与源码。

读取本轮github.json链接PR及实跑source CI，然后fetch核main源码与元数据ancestry。只有main包含代码和此HANDOFF且远端确认后才能直接创建另一账号任务接棒；否则NOT READY FOR HANDOFF。最终head/merge的API事实以后写入最终报告，不预编未来SHA。本轮用户要求交接后立即停止，合并核实完成后不继续开发；下一账号先读AGENTS/PROJECT/HANDOFF，再按Next Step推进。
