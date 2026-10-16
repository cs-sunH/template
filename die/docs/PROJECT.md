# die 子项目：项目说明与验收标准

2026-10-17最新数值检查点：正小曲线事件/精确ready/严格target与真ceil、微步前逐角色及credit预检，最终一致Release58/58；5个新测试对fresh A+B真实red。严格矩阵仍78/84、API13个busy微差明确保留；两配置本次各一轮正常30速度全过，历史失败不抹去。原A+B剩余native三仓各3/4保留R4 guard失败；当前源码五仓待新验证。Norm/busy/R>1及target实测未验收，正式SDK未发布、main未激活，工程NOT ACCEPTED。持续研发及下一步以[HANDOFF最新章节](HANDOFF.md)为准。

2026-10-17 用户恢复持续自主开发：已从真实最新 `codex/die-runtime-continue-20261016@9c464ca` 接续并保留main交接指针。A+B公共源码保持；C2实际混合ready回归失败，不采用。当前C3隔离研究及剩余三仓native推进中，以[HANDOFF顶部](HANDOFF.md)与新身份/原始证据为准；下方停止状态均历史，正式SDK未发布、工程NOT ACCEPTED。

## 当前持续开发（2026-10-16）

用户已恢复持续自主开发。`codex/die-runtime-continue-20261016` 从恢复分支2750e24安全接续并普通合并main94d1978，A+B全新Release完整53/53通过，fresh main两新测试实际red；公开API/8真实轨迹除修复的fork外逐字段相同。严格矩阵78/84保留六舍入失败，两套正常30性能均failed；新组合五仓验证和当前Core七epoch Norm数值修复仍进行中。有效状态以[HANDOFF顶部](HANDOFF.md)为准，下方停止状态均为历史。正式SDK仍canonical24，工程NOT ACCEPTED。


## 最新停止交接指针（2026-10-16）

用户已要求立即停止。未验收 Runtime A+B 源码已远端保存于 `codex/die-core-resume-20261016@6f01a6af1dc8976a8124d754e40446edbcd0b443`，后续纯归档提交保存全部部分成果；组合源码 fresh构建/CTest/API/预算均not_run。main已激活唯一Core，未激活此次Runtime组合；正式SDK未发布，工程NOT ACCEPTED。恢复入口为[HANDOFF最新停止章节](HANDOFF.md)。下方持续研发描述均为停止前历史状态。

## 2026-10-16 唯一 Core 来源与当前状态

硬件权威为根 `架构建模_单Core版_严格复核修订.docx`，只有 Core，32行 `core32_v1`、`L0=a`。完整迁移从远端 `codex/die-single-core-20261016@b8ddbcfd62e03738b0c3996444355cbbe2e99b80` 恢复，已保留 main 独有交接归档；当前分支 `codex/die-core-resume-20261016`。最终公共51/51与192资源点、五仓隔离10库/98配置已验证，实际 native C++22通过/4个原R4失败，JOINT另6项native Python通过。两套正常30样本性能failed。已安全快进激活main至 `6b83624600cdd13c7a8ebe72d36c48cf29873a98` 并实核远端，后续状态以 [HANDOFF](HANDOFF.md) 顶部为准；正式五仓SDK未发布，工程 **NOT ACCEPTED**。

用户授权的10×10、25×32mm、0.75MiB/512bit工程对照点为126.6 INT4 TOPS，只数值接近128 BF16 TFLOPS参考，不等效实测，Vec/存储未匹配，不改变先验或校准身份。此前“停止/未激活”段落是旧检查点；完整恢复归档仍保留，不覆盖当前持续开发授权。

## 仓库与范围

- GitHub：`cs-sunH/template`；公共仿真源 `die/`；共享集成基线 `main`。
- 两个独立 Codex Cloud 账号按 AGENTS 接力，以已合并代码与 HANDOFF 为共享状态，不共享原始 Session。
- 原用户需求包含五仓薄适配（wscllm、wscllm-LRU、face、face-LRU、joint），接入按公共源验证、固定版本发布、逐仓构建及业务回归推进；跨目录修改须说明必要性并验证各仓业务差异。

## 当前恢复状态（2026-10-16）

用户已恢复持续开发。公共生产源码仍与保存迁移一致；五仓最终实际验证范围见[总表](evidence/core-resume-20261016/SDK-VALIDATION-SCOPE.md)。R4不可用guard、WSLRU原optional skip和性能失败均保留。SDK正式canonical24未发布；两项Runtime数值/快照修复和四项性能优化仅在隔离研究中，尚未采用。继续Runtime/Norm/R>1/busy及五仓业务依赖。最新操作以HANDOFF顶部为准。

## 项目目标

**2026-10-16硬件更新**：硬件依据改为根[《架构建模_单Core版_严格复核修订.docx》](../../架构建模_单Core版_严格复核修订.docx)，source_revision=`41f119135fd2d4cc87292f59b4b00adc918eece0`。只有Core这一种设计，资源表为`configs/core32.json`、id=`core32_v1`，4档NoC×8档SRAM共32行；Core边长恒为SRAM长边`L0=a`。公共输入不再有类型选择，出现`architecture.core_type`任意值均拒绝。旧源码/测试/SDK及下方历史检查点只按原身份引用，不能当作本次硬件迁移验收；实际新源码与测试范围以HANDOFF为准。

**2026-10-16明确授权的工程对照点**：用户要求按TX8算力参考选择一套硬件。保留0.75 MiB/512bit/800mm²，明确10×10 Core、25×32mm；DOCX该行每Core633 PE/32 Vec，对应126.6 INT4 TOPS与12.65728T FP32等效Vec。仅矩阵操作率数值距128 BF16 TFLOPS参考1.09375%，跨精度不构成实测等效；Vec/存储不匹配，不拟合target先验。公共完整输入为configs/tx8_compute_matched_hardware.json；五仓legacy保留原16GB/410GB/s每栈HBM兼容配置并补显式网格/尺寸。此为用户委托的一次离线设计，不新增生产硬件搜索。

**2026-10-15最新用户决定**：本项目是给定硬件预测延迟的研究工具。Die宽高/面积、Core行列数及SRAM/NoC/HBM等由用户输入，仿真器不做面积/Core排列/Pareto或Wafer硬件搜索。当前范围到单Die，Wafer铺排不是必填前提；保留输入合法性、资源派生及给定硬件下的软件映射。精度目标10%；多次有记录的合理测试/校准仍难达到时，20%以内也可接受，必须说明10%未达及覆盖范围；>20%失败。此要求替代旧3%/5%硬门槛，原busy定义、守恒/依赖正确性及速度预算保留。基于xlsx正确分解算子做少量有意义超参的粗校准，满足设计机制即可，不追逐噪声逐点拟合。先同步本设计到GitHub，再实施改造。完整条款见根设计§1.6。

实现保留源架构自变量响应的最粗有效单 Die 仿真器，输入每核SRAM/NoC位宽、矩形网格、HBM数量与摆放、Die外形与可选系统诊断，以及具体本地LLM工作负载；预测计算、存储与通信延迟，并保留原Roofline消费者依赖的其他统计、容量和业务机制。

硬件依据为上述单Core版DOCX，算法/校准/ABI/接入/验证依据为《die内仿真器设计文档_最粗粒度参数响应版.md》。硬件几何/资源冲突以DOCX为准；DOCX不含仿真建模§1.5/§1.6/§8.2.1/§13，不能以缺少这些章节为由撤销已确认的路线、精度和速度合同。采用统一C++17公共库、C ABI与薄适配，Python不另写运行时间公式。

TX8 当前工作簿只能提供条件参考校准和工作量回归，不能作为目标 INT4/HBM3E 的独立实测训练集。source/cache、reference、粒度、目标硬件、五仓工程证据分别报告。

建模复杂度遵守用户 2026-10-14 新增要求：与硬件设计参数、软件任务输入参数不直接相关的细节，能用简单数学公式直接计算时，采用公式，不深入建模。优先闭式/解析计数和固定数量分段表达；内部状态只为必要的参数响应、容量/工作量守恒和外部可观察的就绪/竞争保留。新增复杂机制须指出简单公式不足的具体证据和运行成本；硬件自变量与任务输入的正确覆盖继续是硬约束。

开发以有意义的验证检查点同步 GitHub。本轮继续获得源码开发授权；先按最新用户要求同步硬件输入/10%—20%设计，再自主实施。检查点同步不等于模型精度或五仓工程验收。

2026-10-16停止交接补充：生产源码保持已验证5caede8。J32融合遍历无可靠速度收益，不采用，源码patch和49项功能/失败预算证据在j32-update归档。Norm长行两遍首版（R1且每Core一行）作为partial源码/测试/oracle/SHA manifest保存于analysis/norm-long-row-20261016，未激活或发布；定向1/1通过，12synthetic任务/36场景完成及首输出60样本分别≤10%，原busy使36联合全部failed。完整50回归、新scope预算及五仓not_run。归档后work/build-die-continue仍可能含部分版本ELF，接手须fresh build；不得把这些ELF当main已测版本。本次用户要求交接后立即停止，最新具体状态与恢复入口见HANDOFF。

2026-10-16当前源码检查点：内部GEMM仅延后非排序JSON存储，保留所有候选overflow/EqWork/结构/资源/credit/status/预算及Runtime估值，胜者经同一builder物化一次、公开/非GEMM路径不变。最终Release49/49（21.52s），1092完整候选materialization及实际旧/新公开wire、36仲裁轨迹、11完整接口对象和Norm32/96 raw全等。初版routing字符串capacity致cache bytes少10B的比较失败已保留并修复。诊断128³完整分配16498→14447；正常30样本cold P95 1.508469ms、本次适用热/常规冷通过，但J32 77.745μs仍failed，所有历史慢点保留、不选最好样本。总表compiler-ranking/checkpoint-summary明确scope。Norm完成96/首输出160目标10%、原busy54failed，长行两遍及R>1窗口未实现。正式canonical24仍23160c…7e9b未publisher，新公共模型五仓业务/CPU not_run；旧隔离83配置/五R3拒绝与旧SDK10项沿原scope有效。继续J32真实热路径与公共模型依赖，TX8/目标先验冻结、独立target数据缺失，工程NOT ACCEPTED，持续自主开发。

2026-10-16当前接续（覆盖上方历史停止状态）：已从远端d16b545安全恢复，工作分支codex/die-continue-20261016，用户授权持续自主开发。新增零开销重复region可用性修复：通用早退曾让12实际非Norm R>1探针误报ok_assumed，现明确not_available；无新活动/PRIVATE/前置总等待，body数值和21探针完整epoch不变，5R1对象及11默认完整API对象相同。真实red失败后公共50/50通过18.68s。正常30样本预算仍failed（128³cold6.128634ms、OS J8 73.397μs；J32本次47.911μs不抹历史失败），全部raw及identity保存于repeat-availability-20261016。下一步Norm长行BF16/FP32子域/mixed dtype/多Core/容量验证；INT4 scale账本缺口、R>1真正屏障与36busy失败不伪称已解。正式五仓SDK24未publisher，target raw缺失，工程NOT ACCEPTED。

## 当前强制建模路线（2026-10-15，用户确认）

**解析工作量 + 固定数量的分段累计服务曲线 + 稳定区间直接积分**。完整合同见根设计§1.5、§8.2.1、§13及AGENTS。以下替代历史“恢复Program→逐K生命周期→周期证明”的接续建议：

1. 复用原硬件派生、Geometry/Counter/容量/公共Runtime与五仓接口；循环及full/tail只做解析计数，重复次数进入公式，不逐Core/tile/K/wave推进。
2. 输入以统一K消费单位、输出以统一输出块单位分别生成累计曲线；首输出门槛定义在工作进度，端点对应真实总量，不拟合参考时间。SRAM静态字节峰值与流体credit分别负责；边界切向速率约束连续推进，不做逐块启停轮换。
3. 每primitive优先最多5段，必要时8/16段，每owner最多三活动；曲线折点/隐藏切换统一计入预算。固定外部事件序列及活跃规模，放大Core/tile/K/repeat时，内部事件与状态不线性增长。
4. 保留owner Q/alpha、13域、PRIVATE及真实外部依赖/指标；解析/水填充仅在与原公平定义等价时使用。轻量查询、规范化缓存和速度基线同步推进，不把性能后置。
5. 从第一版共同检查10%误差目标/有记录的20%容许范围，以及命中P95≤20μs、常规冷编译P95≤2ms、J≤32单次仲裁P95≤50μs和五仓端到端预算。busy定义不变，另报绝对误差、参考共享时长和适用范围；failed/tradeoff_unresolved如实保留，不超出当前用户授权口径。
6. **周期搜索、精确周期/相位返回证明、大整数周期求解、逐K回放不进入生产方案。** 归档Program只作历史参考，离线小规模逐wave oracle仅用于误差验证，不恢复为默认实现主线。

当前已有部分解析服务曲线源码与新测试：OS v3首K/首cohort/余量/末排空、Plan v4曲线、Runtime直接积分与水填充、规范缓存/轻量cost和memo。最终验证、已知失败、停止状态见HANDOFF和analysis/analytical-curves-partial-20261015；WR/其他模板、精度/性能及五仓新发布仍未完成；工程仍 **NOT ACCEPTED**。活动JOINT是 `astra-sim-joint`；`astra-sim-joint_old` 已由main的 `5e2fb0a` 删除，不恢复。

## 当前代码与运行方式

- 最新实现基于设计先行提交35f4580及实现89c2499：显式硬件输入/取消搜索和10%目标/有据20%规则已落地。五仓隔离Release双ABI+配置83/83通过；FACE重复cost结果修复，消费者133/133及152/152通过，0skip。allocator最终42/42通过（22.34s），新行省略速度待测；先前cold/OSbulk J32及五shape busy仍failed。证据explicit-input-integration-20261015；正式vendor尚未publisher，完整业务/五仓CPU预算未验收，工程NOT ACCEPTED，继续自主开发。 本Session已按用户要求停止并交接；vector升级尚未启动，新JOINT Online因缺protobuf链接路径失败，后续测试not_run，详见HANDOFF最终状态。

- 2026-10-15当前恢复检查点从main `ec07f90`继续：修复曲线终态credit/物理舍入快照、旧计划整数tick、memo/空查询回归，公共Release及41/41 CTest通过；六shape v3 axis仅1过、五shape busy失败（tail独占误差6.9163%），30样本速度仍failed。新route helper仅partial一阶矩未接Mapping；WR/其他模板、二阶/odd-row及五仓新发布仍待完成。证据 `analytical-runtime-recovery-20261015`；工程NOT ACCEPTED，当前持续开发，历史停止记录不适用。

- 2026-10-15最新源码交接基于main `1f7a242`、分支`codex/die-analytical-curves-20261015`。本轮不是仅文档任务；新累计曲线源码直接入库，五仓vendor仍旧canonical24，未publisher。请从HANDOFF最新Next Step接续，不以本文下方历史完成状态认定新源已验收。

- 2026-10-15本次文档更新基于最新main `5e2fb0a10e34563e08e798217670592606eaae8d`，保留c13e9a7、b22e065、0f56921及后续归档/删除。仅更新Markdown，canonical24 digest、生产代码、ABI/配置和历史测试证据不变；不重新声称运行代码测试。新任务按上方解析路线和HANDOFF Next Step继续。

- 2026-10-15最终交接：已完成K参考/Cursor c13e9a7、clock b22e065、新JOINT取消0f56921均在main并实核。未验收单Core OS Program全部源码patch/新文件/测试/SHA/真实构建记录保存在 `analysis/gemm-os-program-partial-20261015/`，默认公共源保留已验证clock版，五仓canonical24不混入半成品。该历史接续建议已由解析累计服务曲线路线替代；下一账号不恢复Program/周期路线，归档仅供参考；TX8与目标先验不变，整体NOT ACCEPTED。最终归档/文档提交main实核后本Session停止。

- 2026-10-15新JOINT known cancel：冻结可选coarse_cancel_events复用Domain/Runtime，same-tick终态优先、pending取消不造handle、partial served保留、重复事件幂等；不传播success/依赖，不猜测gate/barrier lease释放。snapshot消费前缀、双向终态绑定和原子非法拒绝通过；真实clock版native/window/Domain4/4+Python6/6无skip，portable重放通过。旧wire/完整liveworld与sh30 guard保留，生产Sys采集/graph/lease回放仍待接；公共Program开发不混入该检查点。证据native-world-cancel-20261015。

- 2026-10-15时钟检查点：真实middle-K两活动发现大绝对起点使ceil漂移1ns；内部改为checked整数原点+[0,1)小数，next_event整数相加，服务/LP/credit/校准不变，旧Plan及snapshot v2/v3 wire保留。72项跨起点/partial import/cancel/capacity严格比较、legacy小数/u64边界、完整33/33 CTest、full/tail九诊断及五仓双ABI/config83/83（0skip）通过。canonical24 digest `23160c512203596611d5c534104f5dd2015be2ef75a2665c88ffe1cfddcc7e9b`。尚无周期skip或生产E5新证书；此处历史opt-in后续计划已由上方解析主线替代，新JOINT known cancel后续已完成。详见runtime-clock-20261015分析/证据；保持持续开发。

- 2026-10-15本轮恢复持续开发：main d115117→`codex/die-k-readiness-20261015`。已修oracle/诊断final-only输出门，独立Python/C++可达反例精确15s；新六计数器Cursor与真实13域Runtime显式槽位逐步一致，三组full/tail九场景grouped通过，定向CTest3/3（含Python13项）。新内部基础尚未进入生产Plan/Runtime生命周期，O(events)诊断不作E5/E8/效率认证。canonical24 digest dd62e6787557ff2f764fc5be8f38865a2c15916c20f1a9fc95a28ebcd4fb7005；五仓check/独立configure5/5，原生产timing/TX8未改。下一步生产有界生命周期与新JOINT剩余接线；旧停止/交接段落是历史，当前不按阶段停工。证据见HANDOFF及k-readiness-corrected-20261015。

- 2026-10-15最终用户交接：停止新任务，归档单Core K就绪部分诊断/原始结果/旧设计草案/周期成果盘点和replay入口。只有M80/N128/K4096三场景grouped输出passed；诊断与参考共用过严的非final输出槽门，尚未修正/量化影响，不能当生产E5/效率证书。生产源码e73a378不变、整体NOT ACCEPTED。下一账号先修验证门控反例，再K tail/有界生产路径，随后JOINT/五仓剩余项；详情以HANDOFF最新最终交接为准。完成main同步后本Session立即停止。

- 2026-10-15 Runtime数值/原压力检查点：owner共享progress scale与非递减lexicographic minimum修复真实TP logits credit越界，不放宽模型/容差。public30、五仓配置83、相关生命周期/native、FACE Python132/151全过；原模型/队列/144GB SKU压力两仓均0/0、各24完成（FACE7944/LRU5998成功节点），真实source reuse顺序通过。canonical23=`80637c3ed93c58570c2ea05aedeb38f7db0e9911b05d246d4e13d64d56e69646`；重复publish保留mtime。新E5 axis仍failed原3%/5%，继续K readiness。范围/原始失败与成功证据见runtime-credit-20261015，整体NOT ACCEPTED。
- 2026-10-15 future/source-release接线：FACE两仓启动finite C++容量证书、固定E protocol2和真实ACK/源释放依赖；C++各8/8、Python132/151无skip、各900实际Plan受界、六次最终Online通过（186/378/270终态，含overlap与cap2+split）。该历史检查点压力失败，单logits原ABI复现status11；后续修复和同输入成功见最新Runtime记录；不作压力或模型精度验收。证据/入口face-future-workspace-20261015，canonical23与TX8未变。
- 2026-10-15最小KV检查点：FACE两仓manager支持opt-in不可变E与max(A,E) fit、结构不可行提前拒绝，相关Python39/39与34/34无skip；默认active-only协议不变。生产启用仍待C++容量protocol2/manifest证书/source-release接线与实跑，证据见face-future-manager-20261015。
- 2026-10-15持续开发检查点：FACE/FACE-LRU coarse活动workspace admission/v2 handshake已独立C++各8/8、Python120/118全过、六个真实Online通过，含tick150000正lease采样98304B；legacy OFF兼容、原KV/远端策略及单COMP gate保持。该历史检查点E=0；后续future/source-release结果见最新记录，容量极限压力仍待补，不作全业务/精度验收。新JOINT冻结known-capacity事件经Domain/Runtime同源接入，native/window4/4；known cancel/生产live import/sh30仍待接。分析/证据见face-workspace-20261015及native-world-capacity-20261015，TX8与canonical源不变。
- 硬件 `die/src/Hardware.cc`；计划/任务/映射 `Plan.cc`、`Compiler.cc`、`Mapping.cc`；执行 `Execution.cc`；ABI `Capi.cc`。
- 接口在 `die/include/coarse_die/`；配置、工具、测试分别在 `configs/`、`tools/`、`tests/`。
- 依赖 C++17、CMake、Python3 和 vendored `astra-sim-wscllm/extern/helper/json/json.hpp`；TX8 工具仅 Python 标准库。
- 下一环境应核实真实工具安装。本 Session 使用 `/workspace/cloud-setup/activate.sh`、CMake3.31.10、GCC14.2、protoc3.21.12/Python protobuf6.33.5，`PROTOBUF_FROM_SOURCE` 未设置，使用系统 Protobuf module。环境依赖不是跨账号交付；重建入口随证据保存。
- `cmake -S die -B work/build-die -DCMAKE_BUILD_TYPE=Release` 是公共库入口；benchmark 宏缺失已修复。实际最新构建/CTest 状态以 HANDOFF 为准。
- 独立 Runtime、TX8 审计及已移除硬件搜索功能的历史验证命令、日志与结果在 HANDOFF 与 `docs/evidence/`；历史搜索证据不代表当前生产接口仍支持硬件搜索。

## 验收标准

1. 全部源变量正确校验/派生，core32_v1的32行SRAM/B资源、L0=a及几何黄金与DOCX一致；类型字段任意值拒绝，不拟合源硬件常量。
2. 实际局部 shape/精度/TP/span/causal/驻留输入到有界确定候选与容量/byte/ops 守恒；不能仅靠旧 num_ops/tensor_size 补缺失数学图。
3. 估值/实跑/已知 world 投影共用 Runtime；独立 L/C/S、公平 credit、取消/drain/一次完成、完整快照与原子导入。
4. 实际C++对独立离散oracle及有来源的分解算子分别报告：目标误差10%，多次合理测试/校准仍难达到时20%可带记录接受（根设计§1.6）；选择满足精度/速度的最小描述。历史3%/5%结果保留，不能拿Python自对拍或oracle拟合冒充硬件校准。
5. TX8 cache/13组/Attention/FFN/BLOCK 不重复归账；reference 参数与目标先验隔离；真实目标数据缺失保持 not_available。
6. 五仓同一公共源版本，调度/network/KV/LRU/pool/readiness/workspace、取消回放、旧 record/key/gate/window 与新增 typed 指标逐仓实跑验证。
7. 保存真实构建/测试/CPU/RSS/cache/事件/LP 开销证据；未执行不是 pass。GitHub 文档 workflow 不代替工程测试。

详细规范以设计文档 §14–15 为准，当前完成/失败/未运行状态以 HANDOFF 为准。

## 分支及交接方式

每轮以已合并 main 为起点，记录真实 SHA、PR 和测试结果。推送/合并失败、测试阻塞或远端未确认时标 **NOT READY FOR HANDOFF**；代码检查点同步与正式工程验收分开，不让两个独立 Session 并行推冲突内容，不 force-push 或丢弃有效修改。

前次代码分支 `codex/die-k-readiness-20261015` 的成果已在main；本次文档分支 `codex/die-analytical-model-docs-20261015` 从 `5e2fb0a` 创建。新开发任务同步最新main另建分支，按最新解析建模合同和HANDOFF接续。仅使用新活动 JOINT；已有 LRU finite-manifest/source-release 接线不重做。已修 attention sequential layer/pass workspace peak（v3 template、workspace revision v2）、native fresh local-domain bridge 和同冻结输入 native session wire 恢复，并形成真实 E5 shape-axis 失败证据；生产 Sys/ledger/network/pool live 输入与 sh30 guard 仍是后续依赖。GitHub同步状态、精确实跑及限制见 HANDOFF，不将 fixture parity 或发布 digest 冒充五仓工程验收。

## 重要决定及变更记录

- 2026-10-15用户确认解析粗粒度路线：固定分段服务曲线/切向credit/稳定直接积分，排除周期精确与逐K生产推进，速度精度同时验收。本次仅文档，不改用户最新代码、不恢复已删除old仓；下一账号提示词见RELAY_PROMPT。

- 2026-10-15 最小任务：native Domain新增已知物理容量变更接口，复用canonical Runtime校验/更新，保持旧速率结算与同tick terminal优先顺序；共享COMP+endpoint减速、暂停、恢复和import/fork序列、invalid原子拒绝通过，最终4/4 native/window。生产world容量事件、sh30和FACE闭环仍待接，不改变公共模型或校准。按用户要求提交此最小任务后继续开发。

- 2026-10-14 实际sample桥接检查点：显式rank/handle→leg ID将真实只读Sys sample导入NativeCoarseDieDomain，canonical Runtime负责导入与terminal清理，pending completion恰好一次；实际COMP/transfer 2408ns/303318ns和served/整数/registry精确匹配。四项累计busy字符串有±1e-16ns级尾差，完整wire严格全等false，原始失败/Decimal差值保留，未放宽门槛或改变timing。最终native/window4/4+Python4/4无skip；WSCLLM两仓lease保护独立生命周期各1/1通过，保留blocked/retry/cancel。FACE admission/v2 bridge后续独立验证见本页最新记录，future margin/source-release仍open；局部桥接不等于production graph/ledger/network/pool/sh30完成。详见local-sample-domain与ws-lease-hardening证据。

- 2026-10-14 真实 Sys 采样检查点：新增 Runtime-copy 只读 envelope 及实际 COMP/endpoint/lease catalogs，Workload 验证 handle 和权威 workspace ledger 双射；terminal-before-callback 副本可采样，live Runtime/队列/回调不变。提交前拒绝 foreign ledger token、错误 canonical bytes 和 stale Prepared task。当前源8/8 CTest、重链 tiny legacy Online 0/0且192/192成功；真实 sample tick1、原 Sys COMP/transfer 完成2408ns/303318ns。下一步用显式 handle→leg 绑定做实际 sample 的 Domain 继续对拍，完整 graph/ledger/network/pool/sh30 guard仍未接通；另四仓不外推通过。公共 mapping/timing、TX8和canonical23不变，详见sys-local-sample-20261014证据。

- 2026-10-14 native wire 检查点：新 JOINT domain wrapper 与同冻结输入 native session 完整 snapshot/import，复用公共 Runtime v3；恢复后的 step/drain 与未导入 source 完整 inspect 全等。补齐 forecast 同 tick 前缀、fixed pending terminal、跨 gate 唤醒、zero-service child 和 Python barrier facade；4/4 native/window、独立 fresh build4/4+Python4/4、最终 ABI18/18无skip、tiny legacy Online再次0/0且192/192成功。这里只覆盖 native session，生产 Sys/ledger/network/pool/lease 与 sh30 guard未接通；现有 Sys export_snapshot有推进/回调副作用，下一步先加 Runtime-copy只读采样。公共 mapping/timing、TX8/target先验和canonical23 digest不变，E5精度仍failed。详见HANDOFF和native-world-snapshot-20261014证据。

- 2026-10-14 接续检查点：同步 main d9cc024；materialized attention/QK workspace 只取单次 lifecycle，物理 work/byte 仍重复计费，受影响 template/dataflow v3、capacity revision v2，LRU strict-v2 certificate；canonical23 五仓 digest45b7bf3d…c3264。新 JOINT native projection 使用共享公共 Runtime，支持 fresh frozen domains/Plan chains/显式本地endpoint和native gate/join，拒绝scalar活跃恢复及prediction-only倍率，sh30 guard未开放。实际 tiny legacy Online成功，coarse-intended在guard明确失败。新E5单因素可比OS场景继续failed，累计long-double尘埃与计划整数守恒分开报告；TX8不变。真实命令/范围见HANDOFF和新evidence。

- 2026-10-14 用户停止/交接：本轮源码和完整HANDOFF `60d18dc` 已经 PR #11 合入 main，merge `721ab833` 已fetch/ancestry实核，源码CI38590775233 success。仅补最终GitHub元数据后停止；下一账号基于最新main继续，不把工程 NOT ACCEPTED 当作交接失败，也不把PR merge当作模型验收。

- 2026-10-14 容量/源释放检查点：新增独立 `die_coarse_workspace_capacity_bound_v2`，以共享 checked-u128 生命周期公式覆盖 SRAM 筛选前模板，不证明 task 可执行，不改变旧 envelope/Plan identity。LRU finite request-aggregated manifest 为各 rank 安装不可变 E，fit 使用 max(A,E)，真实 Plan 超界在 acquire 前拒绝。源 store/migrate ACK 在复用容量的 COMP/目的 KV 写前形成依赖。公共30/30、62 identity全等、旧240 envelope查询全等；LRU Python182pass/2skip、实际 Online 默认/迭代cap2均0/0，480/560 typed Plans全部受界。完整 source/binary 与局限见新 evidence。生产 E5 精度、全业务压力、完整 GQA 与 host预算仍未验收。

- 2026-10-14 用户新增复杂度原则已写入 AGENTS、本项目说明和根设计 §1.4；本 Session 已从已合并 main `ad865f11` 恢复持续开发，上轮停止指令属于历史交接。后续研究与实现按上述公式优先原则收紧，保存已有有效成果。

- 2026-10-14 workspace上界基础：一个task的全部Mapping候选由C++输出max exact u64，不改旧Plan identities；public30/30+62hash/summary全等、真实legacy240queries和生产ABI全等重放。LRU optional E默认0，fit取串行max(A,E)，永久E不可行fail、临时A可retry，不进KV/active/remote账；manager57与online154pass1skip/44通过。五仓ABI10build/config83、四仓Online/focused18通过；新main JOINT替换后仅纯库发布，旧业务结果不套用。scheduler E尚未激活，source-release门须早于KV容量复用。

- 2026-10-14 LRU v2检查点：strict fullrank容量桥接与exact Plan lease落地，保持whole-session/remote/pool/history；typed硬件单位隔离；coarse指标/cancel与旧LRU字段兼容。最终CPP9/9、Python143pass1skip+44/44、实际8rankMHA官方Online0/0/0、124terminals、八个正workspace FFN Plan；bridge只采阶段末active0。LRU Ring2×2约束不放宽，GQA publicproducer尚缺。base/LRU prospectivePlan压力失活与store物理释放排序成为下一明确实现依赖。
- 同检查点：actual13域facts exact solo powering/226外部stop前缀与resume/capacity对拍，standardC++256slot/768scratch primitive2247差分与sanitizer self-test通过，研究尚未进生产。naive256scratch与三个public-ns偏差诚实保留，原3%/5%门限不变；TX8无新fit/heldout/host证书。
- 2026-10-13：建立 AGENTS/PROJECT/HANDOFF/RELAY_PROMPT 与文档 workflow；保留原流程初始化事实。
- 2026-10-13：本 Session 保存 coarse v2 公共源、五仓原型、两项 TX8 条件参数；默认构建阻塞、校准边界和下一最小任务已记录，未宣称工程完成。
- 2026-10-13 持续开发：恢复默认构建、signed credit 和端点快照，扩充 GEMM/FFN/Attention 独立计数测试，优化候选 Runtime 驱动；边界逻辑量、物理流量、保守槽位容量与粒度证据仍分别报告。手工 grouped-wave Runtime 的 oracle 结果不能给实际编译模板颁发证书。

- 2026-10-13 集成检查点：五仓统一发布 18 文件固定源，实际核心库/Online 构建和选定 17/17 CTest 通过；wscllm 新增 COMP+COMM cancellation/join/typed metrics 真实夹具。构建当前 ABI 后配置 83/83 pass、无跳过；全量业务、workspace/JOINT coarse projection 与模型精度仍未验收。

- 2026-10-13 最终交接：用户明确要求停止，保存 GEMM slowest-cohort 预算和 typed task 校验，公共库 17/17 CTest；五仓最新 19 文件固定源。完整计划/失败 oracle/工程边界及关键 scratch 已写 HANDOFF 与 analysis，最终 GitHub 状态须核对交接 PR。

- 2026-10-13 接续开发：PR #3 已合并并核对 ancestry。实现 compact progression 独立 batch、fractional κ exact whole/dyadic 账本、strict INT4 转换成本、Plan v3/旧 v2 兼容和极小计数快照；公共 20/20、五仓最新 20 文件同 digest，所选 17/17 与配置 83/83 通过。默认 7 个 task hash/summary 保持。
- 本轮真实 K-wave oracle 仍 failed；固定双 owner exact 周期研究不能替代生产证书，有限 C++ 原型仍 O(R)，后续继续真实有界生命周期。30 样本 true cache-hit 探针已保存，128³ cold P95 4.012ms、hit P95 34–42μs，预算仍未满足；TX8 reference/目标先验未重新拟合。该项为历史持续开发检查点；当前停止/交接状态以本页最上方及HANDOFF为准。

- 2026-10-14持续开发：PR #4已merged，缓存immutable escaped hash/summary响应，保持JSON合同/fresh handles，新增内存计入原双上限；公共22/22、五仓20文件digest `05c3f91c…e755`，本阶段两ABI增量构建与选定17/17/配置83/83通过，Online executable未重建。n30 same-model hit P95降至10.426/14.292/18.788μs，128³ cold仍约3.8ms。原样保留旧样本及新尾延迟，不外推全域。
- 独立compiled参考改为设计owner Q/α及真实GEMM LOCAL-Q，legacy equal-progress并列保留；生产E5 finish/busy仍failed，门槛不变。whole-output exact scratch已在solo R=10⁶跳过输出周期；双owneroffset仍fail-closed且大整数增长。520-event严格affine contraction proof只覆盖固定full-MN区段，未接入生产、不引入GMP、不证明flat256发现。下一步合成有界piecewise affine表示与workspace显式v2容量桥接；TX8reference和target priors继续冻结。

- 2026-10-14 最终交接：实现公共候选延迟seal/原结构校验、提交失败回滚、完全同构owner块LP压缩、精确workspace accessor；无LRU wscllm bridge v2/发射前lease/权威KV容量epoch/取消typed统计闭环及typed硬件loader。公共25/25、wscllm最终源7/7、Python在线106/106、硬件/train35/35通过；五仓20文件同digest，ABI/config真实范围以HANDOFF证据为准。生产K-wave oracle仍failed、E8/E16不可用，真实Online E2E未启动，四仓workspace待推广，cold/J32全域预算未通过。新nested affine/固定精度研究只限已证明fixture，未进生产；TX8冻结。本轮用户明确要求交接后立即停止。

- 2026-10-14 接续及交接：公共 endpoint 可选 ready delay 保持旧省略字段摘要；独立 Fraction LP 实际 Runtime 对照与精确 OS 每 K access counter（未接入生产）加入正式测试。五仓 ControlMarker 专用 1B DIE endpoint/本地延迟、read-only live prefix、无通知 teardown，保留 legacy/carrier/pool/join，JOINT virtual pipeline sink 恰好一次。最终公共27/27、五仓定向16/16、21文件同digest、双ABI/config83/83通过；base生产 tiny Online E2E 成功但早于最后teardown源，四仓workspace仍待推广。新 geometry-generated symbolic return/标准C++ outward lowering和solo sharedHBM研究保存，完整双owner256发现及生产E5精度仍failed、E8/E16不可用，counter尚未落地timing；TX8冻结，无新host预算证书。详细依赖与验收边界在当前HANDOFF，用户要求合并交接后立即停止。

- 2026-10-14 接续开发：核PR #7 merge与main ancestry；生产OS总账采用<=32 bounded geometry/per-K counter（实际<=12类），新增wave/cohort/full-tail/q-core facts与single-Core13域活动事实。公共29/29、62任务hash/summary不变，canonical23文件五仓两ABI/config83/83通过；离线实际Runtime+Fraction六场景不替代仍failed的生产E5。LRU manager45项含whole-session/remote/pool/transaction回归通过，协议/adapter继续推进。补跑最终teardown main基线tiny Online成功，按旧binary/source身份单列；TX8保持冻结，无新host预算证书。
