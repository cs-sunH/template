# Codex Cloud 开发交接状态

记录日期：2026-10-15（本 Session UTC）。本轮实际进行了源码开发，最新用户现要求迅速交接并停止。基于 main `1f7a2424b7f49da2b1bbc4c18090cd43add129fe`，分支 `codex/die-analytical-curves-20261015`；保留全部上轮提交、归档和旧JOINT删除。以下最新检查点优先于历史“只更新文档”和旧停止记录。工程 **NOT ACCEPTED**，新源尚未发布五仓。

## Current Status

### 最新：解析累计服务曲线源码检查点及停止交接

本轮先核实干净工作区，0f56921 fast-forward 到最新main 1f7a242（包含3c8face归档、5e2fb0a删除、最新解析建模条款），未覆盖/丢弃外部提交。10月15日再次fetch，远端main仍为1f7a242。未恢复任何归档Program或旧JOINT。用户在开发途中要求迅速交接；本轮在现有最小边界修复/缓存任务收束后停止，不启动五仓发布/大规模验证。

**当前实现为部分完成的新生产源码，不能沿用旧版本的33/33、83/83或精度证书。** Plan新增可选累计credit曲线（v4）；空曲线继续旧v2/v3。OS采用v3解析前缀：真实首K装载→首cohort整K工作（其余首cohort Load与Compute连续重叠）→其余L/C/S流体段→末输出排空；单cohort省略余量段。实际描述预算按各活动独立终止/曲线折点计数，3/4个epoch、通常7个描述归入8段预算，小形状≤5；先前v1/v2错误少算活动端点的口径已修。首输出在首cohort真实compute终态释放；K消费单位和输出块单位分离。所有字节/ops来自原有有界Geometry/Counter，未用参考时间拟合曲线。WR及其他模板仍是既有append_epoch，**尚未完成全模板累计服务升级**。

Runtime以最多固定曲线点推进，边界用原signed LP连续切向积分；long double曲线进度、显式exact快照字段、signed tangent下取整防有限credit长期漂移；增加非负packing水填充快路及typed仲裁统计。最新收束修复对接近终态的曲线活动先检查整epoch信用可行才闭合，修复“进度投影1但credit先清零导致零速率”的问题。曲线snapshot保存已发布速率，在重建公平解、物理/PRIVATE及严格credit切向验证后恢复，避免1ulp差导致百万repeat续算漂移；旧Plan-only snapshot不加此标记。**这些最终修复的具体通过/失败以本节Tests最终日志为准，未完成广泛回归。**

C++语义规范化成为缓存唯一Plan键；新增cost-only ABI复用选中Plan孤立估值，不再次Runtime或完整报告；compile仍每次新句柄，完整estimate保留。收束时加入单槽已验证raw-batch memo，跳过重复JSON/normalize/dump，map插入前失效、命中更新原LRU，所有保留容量计入原4096/64MiB预算；仅全成功、≤256行、≤64KiB才memo。最后测试调整明确区分Plan字节和raw memo字节；**缓存失效/内存与性能广泛验证尚待下一账号**。

证据/可移植入口：`docs/analysis/analytical-curves-partial-20261015/` 与 `docs/evidence/analytical-curves-partial-20261015/`。保存中间编译失败、死锁原state、历史oracle raw、旧ABI构建/回退身份、数学笔记及未运行axis/五仓runner。work不跨账号，ELF/build不提交；源码本体与新测试直接保存在仓库，未把部分实现藏在work。

TX8/reference与目标先验均未重拟：eta_compute_reference=0.9079551923877489，eta_vector_memory_reference=0.3411668292682927；Norm246μs、FFN3363μs仍只属条件参考。目标eta_pe/vec=.90、HBM=.70、SRAM=1、local/mesh/inject=.70、rho=.90、launch1000ns、memory_setup0、kappa_quant4、acc_convert2保持。无独立目标测量，target raw=not_available，不能称重新校准完成。

五仓canonical仍为旧24文件digest `23160c512203596611d5c534104f5dd2015be2ef75a2665c88ffe1cfddcc7e9b`，本轮未publisher、未重建五仓新模型，当前`sync_repositories.py --check`预计发现源差异，**是待发布状态，不是同步通过**。仅两FACE的Python成本调用薄适配已改，新符号存在时用cost-only，缺失时仍完整estimate；真实旧vendor双ABI独立构建成功，两仓旧符号fallback均实测361ns且scheduler走旧full路径。新符号真实consumer及83配置/132+151 Python/Online/CPU预算均未跑。

以下为历史记录，冲突时以上方最新检查点为准。

### 最新有效指令：解析粗粒度路线与本次文档范围（2026-10-15）

**生产必须采用解析工作量 + 固定数量分段累计服务曲线 + 稳定区间直接积分。** 这落实原设计§1/§8.2/§13；当前完整合同为根设计§1.5/§8.2.1/§13与AGENTS。本段和下方最新Next Step替代所有历史“先恢复Program、逐K推进、再精确周期证明/跳周期”的建议；归档保留，不作为生产待办。

本次从GitHub最新main `5e2fb0a10e34563e08e798217670592606eaae8d` 创建独立文档分支 `codex/die-analytical-model-docs-20261015`。保留用户已提交的K参考/Cursor c13e9a7、clock b22e065、新JOINT known cancel 0f56921、Program归档3c8face及old目录删除5e2fb0a。本次只改Markdown，不修改/恢复任何源代码、测试、配置、构建文件、canonical release、raw证据或归档patch；未运行新代码测试，也未实现新模型。历史33/33、83/83等仅按其原源码/证据范围复用，canonical24 `23160c512203596611d5c534104f5dd2015be2ef75a2665c88ffe1cfddcc7e9b`保持。工程 **NOT ACCEPTED**。

强制边界：服务曲线从循环/真实输入/工作量解析生成；输入K消费单位与输出块单位分别统一，首输出为工作进度门槛；静态SRAM字节峰值与动态credit分别负责。缓冲触边后改变可行速率连续积分，不逐块满空切换。每primitive最多5段起，必要时8/16，所有曲线折点/描述符切换计入同一预算；固定外部事件与活跃规模，重复放大不增加线性内部事件。owner Q/alpha、13域、PRIVATE和每owner最多三个活动不变。周期搜索、精确周期/相位证明、大整数周期求解、逐K回放排除出生产路线；小规模离线wave oracle仅作误差工具。

速度和精度从第一版同时验收：轻量查询/规范化缓存与曲线修改同步；命中P95≤20μs、常规冷编译P95≤2ms、J≤32单次仲裁更新P95≤50μs及原五仓端到端预算保持。原3%/5%和busy定义不变，另报绝对误差/参考共享时长/覆盖域；未通过标failed或tradeoff_unresolved，任何口径调整需用户另行明确决定。不保证固定16段必能覆盖所有离散脉冲情形，不以精确时间线作为默认目标。

本次文档更新的交付SHA以包含本段的Git提交为准，须正常push/fetch实核main。新任务可复制 `RELAY_PROMPT.md` 的提示词；该新任务授权持续自主开发，本次“只改文档”是当前任务范围，不要求接手任务也停留在文档修改。


### 前次代码交接（2026-10-15；历史事实，接续路线以上方为准）

本轮已验证成果按顺序合入并实核main：K就绪参考/紧凑Cursor `c13e9a7`，绝对时钟修复 `b22e0654ed5894322f1474b20a42f85f63631196`，新JOINT冻结已知取消 `0f56921b9830961101d241899fd1ebda1ed3d43d`。工作分支 `codex/die-k-readiness-20261015`。最终归档/文档提交以包含本段的Git提交为准，须再次push/fetch/远端exact SHA确认；无新PR/CI可核实，采用普通git fast-forward main，不force-push。

新Program正在实现时收到停止指令。为不将未经完整验证的Runtime/wire更改发布五仓，全部七个tracked文件diff和四个新源码/测试已逐字节归档到 `analysis/gemm-os-program-partial-20261015/`（patch、new-files、SHA、实际构建/检查日志）；默认公共源恢复已验证clock版，五仓canonical24仍 `23160c512203596611d5c534104f5dd2015be2ef75a2665c88ffe1cfddcc7e9b`。这不是删除或丢弃部分成果。恢复入口和精确实现/API/依赖/未完成项在该README与verification.json；不得将研究源码头部的production字样当作验收声明。

部分实现已写：显式小预算单Core OS task/Plan v4、共享Descriptor工厂及capacity/ledger核验、Runtime三个动态stage槽/六游标/终态服务修正、固定类前缀累计、snapshot v4/cancel/estimate、两项新测试与实际v4 driver。它仍O(events)，没有周期skip；完整编译测试和独立精度验收以归档verification为准，未通过项保留。初次编译发现factory shared_ptr赋值及driver命名空间问题，已修当前源码并保存初始日志；不为交接扩展新功能或重复全量测试。独立grouped oracle的新Program来源识别尚未落盘，下一账号先补此项，不能让通用报告冒称E5/最小E证书。

归档前最小检查的明确结果：新公共库/Program driver/schema test编译成功；runtime test因嵌套driver main宏冲突编译失败。schema与actual full-shape driver均exit1，合法Plan在factory重建summary的 `mapping_assumptions` 比较被拒绝。先统一重建Metrics的assumptions与原候选，再处理测试helper入口宏；不能为求通过去掉capacity/ledger防伪。未跑新Program独立oracle或全量回归，也没有五仓Program发布。这些是部分代码的未完成问题，已验证main默认源未受影响。

已完成部分的真实结果：公共Release all build/33项CTest通过；clock72种平移/取消/容量/import及旧库负回归通过（旧实现确实失败）；full/tail诊断九场景通过；五仓两ABI重建+配置83/83无skip；新JOINT native/window/Domain4/4及Python6/6无skip，portable重放通过。没有新全Online/压力、生产E5/E8/E16、host预算或目标heldout结果。TX8没有重新拟合，conditional reference与目标先验见下方Important Decisions，目标raw仍not_available。

前次提出的“恢复Program→独立oracle→有界周期”优先级现已撤出生产主线；新的解析累计曲线/有限宏段路线见上方及Next Step。历史待办仅说明归档当时的状态；再推进新JOINT生产Sys known事件采集/graph/ledger/lease回放及其他五仓业务压力。先前通过未修改部分直接复用，不重做。完整live world前继续保留sh30和`live_world_import_supported=false`；无独立目标数据不重新拟合TX8或目标效率。完成本次main同步实核即停止。

### 新JOINT已知取消检查点（2026-10-15）

活动新JOINT的native冻结输入新增可选 `coarse_cancel_events`，按(time,input order)消费。复用Domain.cancel和公共Runtime，不写第二套timing；同tick物理终态/credit先结算，再cancel，再ready/arrival/capacity。未发出的Plan/endpoint只记outer cancelled且不造Domain handle；已发出执行保留真实partial served与Domain取消终态，重复事件幂等。取消不冒充success，不减少成功work计数、不释放依赖；依赖自然blocked。首片只支持唯一冻结Domain内、无acquire/release或barrier signal副作用的leg，其余明确拒绝，不能猜测lease归属后释放。

snapshot仅显式非空日历增加取消状态/精确消费前缀，旧无日历wire和priority保留。导入同时检查pending事件双射、Domain binding与outer terminal，以及每个首个已消费取消事件的逆向约束：目标只能在该tick前/同tick自然done，或恰在该tick cancelled；不能前缀已消费而target仍active/pending、或更晚才done。原子非法拒绝、same-tick terminal wins、actual partial-byte direct Domain对照和source/import/fork继续均通过。首轮restore漏验取消标志被测试发现并修复；最初输出未重定向，仅保留明确标注的转录，不伪称raw。

在canonical24 clock digest `23160c…7e9b`的真实新JOINT构建上，native/window/Domain CTest4/4、Python真实projection+coarse ABI6/6，0skip；portable入口实际重放通过。依赖恢复使用隔离work目录中的真实Debian protobuf3.21.12开发包/protoc，无mock。精确源码/库/命令、失败及成功日志在 `analysis/native-world-cancel-20261015/` /同名evidence。没有重跑全Online，也未接生产Sys已知事件采集/graph/lease恢复；`live_world_import_supported=false`与sh30 provider guard继续保留。公共die新Program代码仍在本地开发，不混入此检查点；本项正常同步main后继续。

### 绝对时钟平移修复（2026-10-15）

在准备窄 middle-K 周期验证时，实际Runtime暴露相同两活动从不同绝对时刻开始会相差1ns的问题：`16×1×64`周期从43521变43520ns，`1×64×64`从166400变166401ns。原因是将短小数deadline直接加到很大的absolute long double，丢失小数后改变ceil。现将内部时钟分为checked uint64整数原点与[0,1)小数，所有ready/target比较先做整数差；next_event也用checked整数相加。13域物理工作量、LP、Q/alpha、credit、原epsilon/预算和校准不变。公共调用仍结束在整数tick，旧Plan与snapshot v2/v3 wire不变；旧可表示的小数快照仍可恢复。

四种真实per-K facts、六种起点（包括2^53-1、2^63和UINT64尾部）、三种模式（32周期、partial import+cancel、partial import+capacity）共72项严格比较通过；相对终态、busy/served精确字符串和计数均与0起点一致，source/import完整快照继续全等。新测试在ABI一致的旧header+旧库上失败，已保留日志。另测小数legacy wire和uint64溢出只读拒绝/末tick取消。完整Release all build与33/33 CTest、修复后九项full/tail诊断、五仓双ABI及配置83/83（0skip）通过。canonical24 digest `23160c512203596611d5c534104f5dd2015be2ef75a2665c88ffe1cfddcc7e9b`，五份coarse SO SHA `bf56b1c1b7afd3c91a8cace974db19249d3e82567c3c770eda0e623e741cfddc`一致。仅活动新JOINT参与，旧仓未改。

实际middle-pair study四类×五起点×32周期，修复后同类跨起点的完整相对记录严格一致；同一运行内不同周期的busy差分仍有累计浮点减法尾差，未宣称bulk累加或周期跳跃已解决。测试平台为GCC14.2/x87 long double（64位有效二进制位），不外推其他long-double格式。证据及可移植重放在 `analysis/runtime-clock-20261015/` 与同名evidence。尚无生产K生命周期/跳周期、E5新认证、目标校准或完整Online重跑。正在实现显式opt-in有界单Core OS程序，以及新JOINT known cancel；未完成代码不计本检查点成果。普通git同步main后继续开发。

### 本轮持续开发：K readiness 修正检查点（2026-10-15）

已从干净旧工作区 fast-forward 最新 main `d1151179d2a3c3246e0b7a896db9e3dacef52885`，创建 `codex/die-k-readiness-20261015`。Python/C++ 诊断仅 final-K Compute 预约输出槽；两套独立从空状态的反例均得 nonfinal@5s、final wait@6s、Store release/final start@7s、finish15s。原错误同域小夹具没有达到output满，失败记录保留。旧raw逐字节不动；旧K4096重评 comparisons 原有字段全等（仅新增revision元数据）。参考语义 `final_k_output_reservation_v2`，grouped oracle v3，诊断不再冒用E5/minimum-E标签。

新增内部 `GemmOSReadiness.hh`：复用bounded geometry/per-K 13域facts，六个单调计数器表达两个FIFO与槽预约，严格顺序、final-only输出、原子restore、多Core/overflow拒绝已有测试。显式槽位driver在真实Runtime的每次准入/完成/恢复与Cursor逐项相等。M80/N128/K4096、K4097、M65/N65/K4097均OS16×64×64，solo/sync/50ms-offset九项grouped比较passed；最大完成相对误差2.367e-6、shared-HBM约5.060e-4，原3%/5%门限不变。三shape未观察到output满非final（均0），独立反例另报。定向CTest3/3含Python13项通过；metadata/cursor不是生产Plan生命周期，driver仍O(events)，不计生产E5/效率证书。

canonical发布24文件，digest `dd62e6787557ff2f764fc5be8f38865a2c15916c20f1a9fc95a28ebcd4fb7005`；五仓check/独立tests-OFF CMake configure5/5。新发布只新增内部头文件/CMake目标，原Runtime/Mapping/Plan/Compiler、TX8和目标先验未改；先前public30/five83/压力成果复用，未重跑或重新认证Online。仅新活动JOINT参与发布。证据/源码SHA/原始报告/复现见 `analysis/k-readiness-corrected-20261015/` 与同名evidence。本检查点按普通git快进同步共享main；精确SHA由包含本文的Git提交确定，最终以远端HEAD/祖先关系实核，不以本地提交冒充共享。继续下一依赖，不停止开发。


### 上轮最终交接（2026-10-15；历史记录）

用户已要求迅速交接并停止。生产源码最后检查点 `e73a378511f34c50261d6074e437e382e8ae68a2` 已核实同时在工作分支和 main；本次交接在其上保存 K-readiness **部分诊断**、只读复用盘点和完整接力说明，没有新增生产模型/校准参数。当前交接提交 SHA 以包含本文的 Git commit 为准；共享分支仍是 main，最终同步必须以 fetch/远端 exact SHA 实核，不以本地提交或 PR 创建代替合入。GitHub API Forbidden，不能创建/核实新 PR/CI；可用普通 git fast-forward 推送 main，不 force-push。

新增成果完整保存在 `analysis/k-readiness-20261015/` 与 `evidence/k-readiness-20261015/`：原执行源码 `raw/probe.cc`、完整原始 JSON/对照报告/空错误日志、原旧方案草案、周期/有限尾类/异步 LP 研究盘点、源码和二进制身份、可移植 replay.py。未提交 ELF/build/cache。只完成单Core OS M80/N128/K4096、solo/同步/50ms-offset 的 O(events) 诊断：复用真实 per-K 13域 facts 和未改公共 Runtime，同一 owner 的最多三个单阶段执行以完整终态传递 ready token；没有把原编译 E5 Plan 的执行结果冒充此诊断。grouped checker 三场景原输出 passed，solo finish相对误差1.118012e-6、offset shared-HBM相对误差0.000324556；solo1281 advances/1290阶段完成/3fact类。原 equal-progress 旁证的失败仍保留。**这些结果不是生产 E5/最小E/效率证书。**

收尾只读审计发现：诊断 `probe.cc` 和 `validation/compiled_gemm_oracle.py:455-472` 都在 output occupancy==2 时阻止所有 Compute；设计§6.2/6.3/8.5、`tests/os_lifecycle_runtime_driver.cc:588-603`、Solo13参考只对 final-K Compute 预约 output slot。静态SRAM含两份pending output并不要求非final Compute也有空output。共同门过严已确认，是否影响当前三场景数值 **未验证**，未修正、未改变参考门槛。原 checker 报告还继承了“无手工Plan”及 E5/minimum-E 标签，这些基线说明不适用于手工封装 facts 的诊断；raw字节不改，范围纠正在README/summary。本轮没有运行K4097或M/N tail，也没有生产周期压缩、多Core/WR新实现。

最新生产状态不变：Runtime logits有限credit修复、FACE固定E证书/真实源释放、原129层同输入容量压力已闭环；公共30/30、五仓实际配置83/83、相关native/生命周期与FACE Python132/151均通过（详见下文及runtime-credit证据）。生产 E5 新binary axis仍失败原3%/5%；E8/E16、完整GQA、完整JOINT live world/sh30、全业务/端口拓扑压力与host预算未通过，工程 **NOT ACCEPTED**。新 `astra-sim-joint` 是唯一活动JOINT；`_old` 未改。


2026-10-15 Runtime修复/容量压力闭环检查点：实际logits M512/N21846与21845/K16384/pass8的有限credit越界已修。按r=x/Q、alpha=Q/owner_Q消去重复除乘，以owner共享progress scale组装原physical/private/credit行；保存lexicographic非递减minimum，防止已有simplex epsilon内后层低于前层。最终没有增加LP行/事件/物理系数，也没有放宽credit、256pivot预算或3%/5%门槛。两种实际TP extent各8候选全部完成，原typed CABI任务均无错误；public30/30、五仓dual-ABI配置83/83（coarse SO SHA `0d4aa60101f440f80379715801f3e0b76a53d0b9402e617de29b1e3b97cd626a`）、FACE各3/3/WS各1/1/新JOINT8/8相关C++及FACE Python132/151无skip通过。canonical23新digest `80637c3ed93c58570c2ea05aedeb38f7db0e9911b05d246d4e13d64d56e69646`；publisher重复同步不改120个相同文件mtime，避免无关重编译。

原大模型/原队列/144GB SKU同输入重跑两仓均C++0/Python0，各完成24请求：FACE111ACK/7944成功终态，LRU84ACK/5998成功终态，GC retained0。FACE9重计算/2NoC迁移，12个source ACK中6个有后续容量复用且顺序通过；LRU5恢复，30个真实池写分片=10direct+20ACK release，30个restore写和30个后续source容量复用顺序均通过。E=148659840..155738880B/rank；原始源码/二进制前后稳定。只证明该合成压力输入，不能外推所有端口/拓扑/参数。FACE服务未接KV_DELTA_JOURNAL，未声称相应日志通过。分析/原失败与修复中间探针/成功证据在 `analysis/runtime-credit-20261015/` / `evidence/runtime-credit-20261015/`。

最终新source/binary的E5 axis已复跑，可比OS的solo/sync/50ms-offset仍全部failed，K128选WR不作OS比较；K/final-output readiness下一步仍需修，TX8/target先验未重拟。完整GQA、生产JOINT live world/sh30、E8/E16/目标raw及host预算仍open，整体NOT ACCEPTED。此为历史持续开发检查点；最终停止/交接状态以上方最新记录为准。

2026-10-15 future workspace/source-release检查点：FACE两仓生产coarse startup已启用finite manifest C++容量证书和固定E（capacity protocol2，全rank必填）；旧capacity protocol1/legacy OFF保持。C++各8/8、真实protoc/双ABI Python132/132与151/151无skip、每仓900个实际emitter选中Plan workspace逐一受界。六个最终Online均0/0：各legacy186终态/3ACK、overlap378/6ACK且tick150000采到98304B活动workspace、cap2+首token拆分270/4ACK；overlap E=26880B/rank，tiny cap2 E=3072B/rank。NOC ACK、LRU直连store/边缘ACK到所有后续COMP/目标KV写的跨batch尾账本已接；池store→restore账本独立，长队列ACK tag使用独立空间。证据与可移植入口 `analysis/face-future-workspace-20261015/` / `evidence/face-future-workspace-20261015/`。公共模型/校准源和canonical23未变。

历史future/source-release检查点中的大模型容量压力 **FAILED**（后续同输入修复/成功见本页最新记录）：不改SKU，129层GELU/H16384/F65536/head128/vocab131072+12个双turn session，两仓C++134/Python1，在首请求出现 `credit crossed a finite buffer boundary`。已用单个实际logits typed任务通过原公共ABI独立复现status11，原输入/response/库SHA与最小replay入口已归档；不是workspace握手引入的错误。该失败记录保留；不允许调大credit容差、改物理容量或删失败门槛。source-release真实极限/GC压力尚未验收，整体NOT ACCEPTED；本轮按用户最小任务要求先同步manager bbcacc5后继续，此处继续开发。

2026-10-15最小KV任务：两仓manager新增可选不可变future workspace floor，fit使用max(A,E)，扣除固定E后连空实例也放不下的请求在逐出冷KV前明确拒绝；原KV快照/日志不伪增workspace。默认None仍为原active-only/protocol1，显式E（含0）才输出capacity protocol2。真实protoc+双ABI相关Python FACE39/39、LRU34/34且0skip，交付文件SHA与实测stage逐一核对，证据 `evidence/face-future-manager-20261015/`。生产startup证书、C++ protocol2与物理source-release在途，未包含在此最小提交，整体NOT ACCEPTED；提交后继续。

历史2026-10-15活动桥接检查点：FACE/FACE-LRU 已完成 coarse Plan lease、完整rank v2绝对状态/容量握手、规范batch身份及同版本ACK，保留v1/两仓原KV策略和单GPU COMP约束。独立C++各8/8、真实protoc+双ABI Python120/120与118/118（0skip）；四个真实legacy/coarse tiny均0/0、3delivery/ACK、186成功终态；两仓coarse overlap均0/0、6/6、378成功终态，tick150000真实采得6rank×16384B=98304B。实际源/ABI/二进制/命令与失败探针保存在 `analysis/face-workspace-20261015/` 和对应evidence。LRU mailbox仅测试断言在Online后补全，不影响已测生产二进制。该历史检查点中未来E=0；后续固定E/源释放结果与压力限制见本页最新记录。

新JOINT known-world capacity接线源码提交 `6924c83`：冻结可选coarse_capacity_events全部预检，按(time,input order)复用Domain update_capacity，不改变旧ready/latency/forecast顺序；snapshot非空事件记录精确消费前缀、pending双射及对应真实rates，same-tick预算导入后继续与source/同clock direct Domain全state严格对拍。最终native/window4/4通过，证据 `evidence/native-world-capacity-20261015/`。这是native JSON已知输入能力，生产Sys事件采集、known cancel及完整live graph/ledger/network/pool仍待接，sh30 guard保留。

WS兼容修复 `ca604f5e49dc6df387888a33c75b0134642d42f3` 已正常推送main并实核；最小Domain任务 `8447b76` 也已发布。上述native/FACE成果按各自验证源码SHA提交并正常同步，远端最终SHA以Git历史和本轮remote核验记录为准；没有PR/CI可核实权限，不作相应声称。当前继续开发。

2026-10-15兼容性修复：WSCLLM 两仓的真实旧 typed parser 同时保留 raw task 和 legacy diagnostic，prepared coarse preflight 不再因 diagnostic 非空拒绝 canonical Compiler 支持的任务。真实 parser→prepare/256B lease→issue/同Plan Runtime→finish 完整回归各1/1 CTest通过；owner/task/bytes 防护保留。JOINT 新 `die_task`/旧 `die_op_spec` 的互斥 wire 约束不扩展。证据 `evidence/typed-spec-compat-20261015/`；FACE 在途代码不包含在此小修复检查点。

2026-10-15最小任务检查点：新增 native Domain `update_capacity(resource, absolute_physical_rate, now_ns)`；先以空canonical Runtime预检，再结算旧速率与同tick终态/chain successor，最后调用公共Runtime更改物理速率。invalid输入不推进live域；共享COMP+endpoint减速/暂停/恢复、import/fork继续对拍与5ns完成边界后继15ns完成通过，最终native/window 4/4 CTest。公共数值模型、校准和sh30 guard不变；native world事件输入仍待接。证据在 `evidence/native-domain-capacity-20261015/`。用户最新要求最小任务后提交一次、随后继续，FACE在途改动不混入该提交。

从远端 main `d9cc0240e07064996b5330c5659f7f8c6866998a` 恢复，已核实包含 PR #11 merge `721ab833` 和源码 `60d18dc`。本轮分支 `codex/die-joint-runtime-20261014`，复用此前 Compiler/Mapping/Runtime、LRU finite-manifest margin/source-release、typed adapter 和 canonical23 发布基础；不重复已通过且未改的研究/业务范围。

本检查点包含三项成果：修复 attention 顺序 layer/pass 的 workspace 峰值；为新 JOINT 增加真正使用公共 Runtime 的 native local-domain 投影片段；将 E5 编译后 oracle 工具扩为实际 shape 单因素探针。源码及证据已提交 `bb1677f67074eec02a612ed7b99af2740187772d` 并正常快进推送至工作分支和共享 main；fetch/ancestry 实核 main 包含该源码。代码检查点已安全同步，工程整体仍 NOT ACCEPTED。当前没有停止指令，继续剩余依赖。GitHub事实见 `evidence/native-local-projection-20261014/github/remote-verification.json`。

随后完成新 JOINT domain wrapper 和同冻结输入 native session 的完整 wire snapshot/import；恢复后的下一步及最终完整 inspect 与未导入的 source continuation 全等。补齐同 tick forecast 前缀、fixed pending terminal、多 gate 唤醒和零服务前驱激活边界；Python barrier facade 修正真实字段名。源码检查点 `b83525c213f5fda26a0bc26e9c0b700ebd3e68f7` 已正常快进进入 main，2026-10-14T17:10:50Z fetch/ancestry 实核。源码/二进制身份及证据在 `analysis/native-world-snapshot-20261014/` 与对应 evidence，独立全新 build 和既有 build 的 projection .so SHA 均 `19301229d118c838a73d56e02835435cd0f4201c514b6fa5b632760ea3a3347b`。

当前新增实际 Sys 的 Runtime-copy 只读采样及 COMP/endpoint/lease ownership catalog；采样不推进 live Runtime、不 dispatch callback、不改变队列或容量账。提交前校验同 rank 外 ledger token、canonical workspace bytes 和 stale Prepared task。当前源码真实 Sys/DIE/HBM/native/window 8/8 CTest；重链 tiny legacy Online C++/Python 0/0、192/192 successful terminals。真实采样保留 tick1 的一个 COMP、一个 COMM_READ 和40960 workspace bytes；原 Sys 队列实际完成分别2408ns/303318ns。详见 `analysis/sys-local-sample-20261014/` 与对应 evidence；这仍不是完整生产 world import。

只读采样源码检查点 `385641813de28a32b1c8f01314442e6a510a8b4e` 已正常进入 main，2026-10-14T17:38:43Z fetch/ancestry 实核。随后新增严格 local sample→NativeCoarseDieDomain 导入，显式 expected rank 和 handle→leg map；实际采样继续执行的2408ns/303318ns及served/整数/registry精确相等，四项累计浮点busy字符串有 ±1e-16ns级差值，**完整 Runtime wire 严格全等为 false**，没有容差掩盖。最终4/4 native/window+4/4真实库Python通过。WSCLLM 两仓同一lease保护已独立构建，生命周期CTest各1/1通过。此阶段证据见 local-sample-domain 与 ws-lease-hardening 包；FACE 两仓现已通过本页新增的有限Online桥接验证，未来margin/压力/物理源释放仍待验收。

新 JOINT 的 wire 检查点 tiny legacy Online 再次 C++/Python exit0，192/192 terminals、1 request 完成；该次 C++ Online relink 后字节与前检查点相同，stage Python 源/新 projection 库逐文件 SHA 实核。当前 Sys 采样检查点 Online 已重新编译，二进制身份单列在 sys-local-sample evidence。真实 sh30 coarse-intended Online 的既存 provider guard 失败保留，没有删除 guard 或假装完整接入。新 native session 可从自己的活动 wire 恢复；生产 Sys/ledger、network/pool 所有权和 sh30 provider 尚未接通，`live_world_import_supported=false` 保留。

生产 E5 的 finish/HBM busy 原 3%/5% 门槛仍失败，E8/E16 不可用；目标 raw/held-out 数据不可用，TX8 reference 和目标先验保持冻结；没有新的 host-overhead 通过证书。

## Integration Branch / Last Verified Base

- 共享分支 `main`；本轮工作分支 `codex/die-k-readiness-20261015`，基线 main `d1151179`。下方较早分支/SHA仅为历史实核。
- 本次最终交接前 `git fetch origin main` 和 `git ls-remote` 核实 main/工作分支同为 `e73a378511f34c50261d6074e437e382e8ae68a2`，工作区生产源码干净；本次仅提交新诊断/证据/接力文档。前一检查点远端实核记录在 `evidence/k-readiness-20261015/previous-checkpoint-remote-verification.json`。
- 起始远端/本地 HEAD 均 `d9cc0240e07064996b5330c5659f7f8c6866998a`，工作区起始干净。当前运行使用脏工作树，实际源码按 evidence 中逐文件 SHA 绑定，不能用起始 commit 冒充全部运行代码。
- 历史 canonical23 五仓 digest `80637c3ed93c58570c2ea05aedeb38f7db0e9911b05d246d4e13d64d56e69646`，sync/check 通过；`_old` 不参与发布。
- Git fetch 和 branch push dry-run 成功。当前环境 `gh api` 对 `api.github.com` 返回 Forbidden，不能声称 PR 已创建或 CI 已核实；检查点通过正常 git 推送及远端 ancestry 核实同步，不 force-push。

## Completed

### 本轮新增（部分工程，非验收完成）

- `Plan.hh/Plan.cc`：累计曲线校验、工作进度插值/斜率/边界，v4序列化，旧v2/v3无曲线格式保留，描述数真实计数。
- `Mapping.cc`：OS真实K/输出单位与首输出整cohort物理前缀；SRAM精确峰值计算提前剪枝，静态byte与流体credit独立。
- `Execution.hh/.cc`、`PackingAllocation.hh`：曲线直接积分、数值边界/快照修复、非负水填充和实际仲裁诊断；signed仍原LP，ratio快路仅推导未实现。
- `CompilerInternal.hh`、`Compiler.cc`、`Capi.cc`、`Capi.h`、`Memory.hh`：规范语义缓存、轻量cost、单槽请求memo及其实际保留内存；不缓存执行句柄。
- 新schema/runtime/OS/scaling/cache/packing测试，实际Plan oracle驱动扩展、独立离线分析工具、单次仲裁benchmark；测试状态见下节，不能把文件存在当作运行通过。
- 两FACE薄适配、旧ABI真实构建与fallback；五仓runner/axis runner只准备未执行。有效草稿/推导、raw和失败日志均保存。

以下为历史记录，冲突时以上方最新检查点为准。

- 本轮已验：c13e9a7的K参考修正/六计数器基础、b22e065的绝对时钟修复和五仓dual ABI/config、0f56921的新JOINT冻结known cancel；详细真实结果及边界见Current Status。未验收Program全部源码/测试与失败记录已归档，默认源未启用。

### Attention workspace 生命周期修复

`die/src/Mapping.cc` 在乘 `layer_instances * weight_passes` 之前保存一次 invocation 的 materialized score/query counts。workspace 和 candidate-independent capacity bound 使用该生命周期峰值；真实 slots/ops/读写/转换/计时工作仍按顺序重复次数计量。compact span count/batch 仍属于本次物化的请求集，不被擅自改成峰值 max。

受影响 full-score 和 standalone QK template/dataflow 升 v3，Plan 带 `workspace_model_revision=workspace_lifetimes_v2`；C ABI capacity revision 升 v2。旧 Plan wire import 仍保留其原始 identity/语义，新编译结果不能冒用旧 hash。LRU client/manifest 证书 strict-v2，拒绝 v1，不混用旧 E 证书。详见 `analysis/attention-workspace-20261014/` 与 evidence。

### 新 JOINT native local-domain 垂直接线

`NativeCoarseDieDomain` 以一个共享 `coarse_die::Runtime` 服务一个物理 rank/domain；Plan 链按真实 terminal tick 提交下一项，外节点只完成一次，端点与 COMP 共用同一仲裁/credit/served/busy 域。提交前校验完整链/端点，失败不推进已有工作；cancel 只保留已服务量；终态 Runtime handle 回收，completion 消费后才可复用 leg ID；in-process fork 复制 Runtime、链绑定和完成队列。

`NativeProjection` 新增冻结 `coarse_die_domains` 与显式 `coarse_endpoint`；typed coarse descriptor 直接编译公共 Plan，旧 A3 timer/memory 路径不接管 coarse 工作。baseline/candidate overlay 保留同一冻结硬件、各自私有 Runtime。native gate/依赖仍由原图管理，coarse outer terminal 才释放。Python 仅序列化输入和结果，无延迟公式；legacy wire 省略新字段。

查询 profile `fresh_integer_local_domains_v1`，整数 ns <=2^53-1；拒绝旧 scalar 活跃剩余状态、混合未分配所有权的 stream/pipeline、仅预测侧 eta/gamma/service factor。`export_runtime_snapshot()` 仍仅是原始 Runtime 诊断导出；完整恢复应使用下述 wrapper/world wire。生产 live import 尚未接通，结果明确 `live_world_import_supported=false`。sh30 guard 保留。初次 native 联测发现 overlay 丢失冻结 domains，已修复并保留失败/成功日志。

真实 tiny legacy Online 成功，新 coarse-intended Online 在 provider guard 拒绝；这两项与 native fixture parity 分开记录。说明、实际命令和证据在 `analysis/native-local-projection-20261014/` 及对应 evidence。

### 新 JOINT domain/native session wire 恢复

`native_coarse_die_domain_snapshot_v1` 保存 Runtime v3、全部 immutable Plans/链游标、Owner/handle、endpoint/transaction bindings、保留终态和未消费 completion；复用公共 Runtime import 并验 live handle bijection、Plan/owner、链边界和引用数。取消链旧 inspect 语义不变：completed 数组仅保留成功前缀，取消当前项在 last_progress。

`NativeProjectionWorldSession` 新 `export_snapshot`/`import_snapshot` 使用 `native_coarse_projection_world_snapshot_v1`，严格匹配冻结输入和 kernel contract，保存 graph legs、ordered active timers/events、gate queues/history、barrier、clock/counters/controls 和完整 domains；工作完成账和依赖数由图重算。导入在 disposable candidate 完全校验并成功 inspect 后才发布，失败不变更调用方。一个 processed forecast prefix 数区分同 tick 已消费/仍待处理事件；native 时长/ready 下界采用简单公式，无 acquire 的启动时刻精确相等。Python 仅转发/同步 facade controls，并修 barrier signallers/passed_at_ns 解析。

coarse-only 修复三个实际调度边界：controlled integrate 抵达 fixed terminal 后仍可同 tick 结算；跨 acquire gate 释放重查既有 FIFO 队列；初始扫描不重复激活已被 zero-service predecessor 回调激活的 typed child。通用 gate fit 改用差值比较避免 u64 加法溢出。

独立新目录 portable runner：4/4 native/window CTest +4/4 Python，0skip；最终既有 build 4/4，真实 ABI protocol+facade 18/18，0skip。覆盖原 source step/run 与 import continuation 全 inspect 对拍、部分物理进度/chain/known arrival/gate/barrier/fixed/latency/budget/terminal、same-tick forecast prefix、invalid atomic 和独立 session。既有 tiny legacy Online 最终源再跑 0/0；ledger 仍报告48 stale_amount_works，不冒称全 ledger 验收。复现 runner 两次环境配置失败（已有 cache source mismatch、缺 backend library flag）与修正后的全新成功日志一起保留。

此合同只做结构一致性及合法继续执行校验，不认证任意编辑过的完整历史；不包含未知网络流量、pool、workspace lease、外部 callback 或生产 ledger 身份。Domain cancel 已覆盖；native-world cancel/capacity 事件和完整生产 world 仍待补。生产 `CoarseDieRuntime::export_snapshot(now)` 会 advance/dispatch/schedule，**不是只读采样**，不能直接用于投影而假装无副作用。

### 真实 Sys 只读采样与 workspace preflight

新增 `sample_snapshot_at(now)` 只在当前 Sys decision tick 复制 Runtime 并推进副本；导出 Runtime v3 和真实 completion Owner/handle，包括副本已终态但 live callback 未运行的合法边界。adapter catalog 读取实际 COMP node/generation/raw task/Plan/lease；HBM catalog 保存并读取实际 completion rank/node/sink 和 typed endpoint。Workload 聚合验证 handle 完整双射、唯一 lease 数/字节与权威 workspace ledger 相等，复制 revision/capacity 和 local endpoint join flags，不导出原始 callback 指针，也不恢复外部网络或 pool。

WorkspaceAdmission 验证实际 ledger owner、live node/generation/id 和 exact bytes；Prepared 保留共享 immutable raw task 引用，并以公共 Plan accessor 校验 bytes/HW/identity。Workload 在 graph take/COMP occupy/stats 前执行 const preflight，adapter 在提交前复核，acquire 在容量 mutation 前拒绝不一致 Prepared。实际夹具覆盖所有 public key 碰撞的同 rank foreign token、错误 lease/Prepared bytes 和 stale task；旧 blocked/retry/cancel 流程保留。

真实 tick1 partial sample、drained sample、terminal-before-callback sample 和原 Sys terminal ticks 已保存。8/8 CTest 与该源重链 tiny legacy Online 0/0 通过。另四仓按实际差异推进：WSCLLM 两仓保护移植和独立生命周期验证已完成；FACE 两仓 admission/v2 handshake 另有当前独立验证，不把 JOINT 结果外推给它们。

### 实际 local sample 导入及 WSCLLM lease 保护

`NativeCoarseDieDomain::import_local_sample` 严格校验 schema/rank/clock/HW 和 Runtime/completion Owner/handle/lifecycle 双射，调用方必须完整显式给出唯一 leg ID。活动执行保留原 Plan、handle 和进度；副本已终态但外部 callback 待运行的 handle 先保留原 progress，再调用 canonical Runtime destroy，转换为一个待消费 completion。同 finish tick 的 pending terminal 保留 handle 顺序，不被 leg 名字排序改变。最后复用既有 Domain wire importer，不增加 timing/容量公式，不恢复生产 callback/graph/lease。

真实已保存 Sys 样本对拍：COMP2408ns、endpoint303318ns与未触碰原队列完全一致；terminal sample在1ns只产生一次completion；最终所有served、registry、容量、clock及整数/event/generation字段相等。完整 raw wire 严格比较失败，仅compute_busy_ns +2.23e-16ns、hbm_shared_ns及两row_shared_ns -1.110e-16ns。原 live 从0ns积分，采样副本在1ns切分；记录确切Decimal差值，不把这四项计为严格通过、不修改生产数值或放宽精度门槛。第一严格探针exit2保留；最终portable runner仅对其余字段/实际terminal进行验收，busy四项明确ungraded。4/4 native/window、4/4 Python无skip通过，复现源码/脚本和证据在local-sample-domain-20261014。

WSCLLM/base与LRU各自仅移植WorkspaceAdmission/Prepared/adapter/Workload preflight及原生命周期测试；LRU校验在清理blocked诊断之前。独立clean Release build各1项真实Sys生命周期CTest通过，直接二进制分别63/54个断言，包括原有blocked、capacity retry、live cancel和依赖释放及新增foreign ledger全键碰撞/错误bytes/stale task拒绝。未声称广泛suite、当前Online或其他仓业务已覆盖；身份与实际命令在ws-lease-hardening-20261014。

### E5 单因素诊断工具

实际 C++ driver 新增 M/N/K CLI，默认旧 shape 保留；独立 grouped-Q/alpha K-wave oracle 支持 full/tail K 与 M/N tile classes，并披露逐行/方向累计计数。M80/N128/K128 实际选择 WR `1x64x64`，不与深 K 的 OS `16x64x64` 强做同因素对拍。K4096 对 K4097、full-MN 对 edge-MN 两组实际保持同 OS tile/template。

三个可比场景的 solo/synchronous/50ms-offset 精度均 failed。计划整数工作量守恒；累计 long-double serialized counters 有微小非零尘埃，严格相等与数值 reconciliation 分开，不把容差通过称整数全等。此工具不修改生产 timing、不颁发粒度证书。报告在 `analysis/gemm-axis-20261014/` 与 evidence。

### FACE/FACE-LRU active workspace 与已知容量事件

FACE两仓的原schema1 typed任务经真实parser同时保留raw JSON与legacy diagnostic，coarse预检接受该合法组合；Prepared/真实ledger owner/canonical bytes/HW/task检查在graph take和GPU/stats mutation前进行。Blocked保留真实online free节点，更新capacity split后重试；finish先释放lease再发布Success，cancel保留served并发布Skipped且只释放一次。v2 request全rank absolute revision/bytes，response全rank容量拆分回显revision，v2 batch_id显式且等于delivery，ACK schema与该batch一致；v1缺省batch行为/省略workspace字段保留。

Python原KV snapshot继续KV-only，workspace仅扣physical fit并唤醒既有capacity epoch。合法coarse loader从已经解析的typed摘要转换selector单位，Resolver.peak_perf_tflops仍None，system不注入legacy scalar；FACE-LRU复用FACE现有coarse薄ABI。粗粒度selector仍是原组合，final stats显式披露approximate/代表TP rank/未含collective和queue/fixed D2D，不能称完整known-world predictor。独立tests与真实Online结果见Current Status。

上述历史桥接阶段E=0；后续finite E/source-release接点及原容量压力数值阻塞已完成修复和同输入闭环，当前进度以本页最新Runtime检查点为准。scratch对有限manifest候选上界已使用真实CABI探索，不是生产证书；未在Python计算workspace bytes。没有改变public mapping/timing、canonical23、TX8、E5门槛，整体NOT ACCEPTED。

新JOINT冻结capacity事件与baseline/candidate共享同源物理速率日历。非空日历才增加snapshot前缀字段，旧无日历wire字段保持；受控same-tick暂停/恢复、预算截断和旧native/window行为4/4通过。完整生产world和已知cancel仍未开放。

## Next Step

### 最新唯一接续顺序（覆盖下方历史条目）

1. 首先阅读本检查点README、verification.json与最终handoff构建/测试日志；重新核实main和源。若最终定向仍失败，优先修`service_curve_runtime_test`的published-rate/credit tangent恢复、`os_service_scaling_test`的终态credit闭合或CAPI memo回归；不要先publisher或恢复Program。
2. 做一致源码Release构建与必要公共回归。补memo命中/插入淘汰/失败兄弟/4096与64MiB边界，严格保留fresh handle和legacy wire。benchmark的alias cache字节相等假设也需区分memo，不能改性能门槛。当前未测最终P95。
3. 同时运行`coarse_service_curve_benchmark --samples 30 --output work/...json`及保存的`run_axis.py --final-driver-ready`（先检查--help），留实际source/ELF身份；不得用总时间除更新数。重测v3 solo/sync/50ms、绝对busy误差/参考共享时长/覆盖，历史v1失败不可冒充v3结果。非负水填充已有独立Fraction LP通过记录；signed ratio闭包仅数学候选，未实现/未验证。
4. 依据误差/CPU决定5/8/16最小描述。busy数学笔记给出明确假设下的条件下界（正服务on-run数、有限K预取、恒定消费），**不是所有表示不可能的证明**。原busy/3%/5%不变，不能利用率加权/事后修计数；无共同通过者标tradeoff_unresolved。WR/K及其他模板升级仍未做，不能声明架构全部覆盖已验收。
5. 公共源稳定后`die/tools/sync_repositories.py`发五个活动仓，重建两ABI、83配置及两FACE真实新/旧符号、132/151套件；prepared runner在本检查点integration目录，旧baseline源来自Git的1f7a242 vendor。然后才做五仓完整固定输入开销对照和在线业务、JOINT生产依赖/lease/liveworld。保留sh30/unsupported live import guard，永不恢复旧JOINT。
6. TX8继续独立conditional reference；有可识别新数据才校准。无target raw保持not_available，不用粒度oracle拟合eta。完成有意义检查点再正常push/整合main，保留其他提交。

以下为历史记录，冲突时以上方最新检查点为准。

**按解析分析模型继续，不恢复逐K Program或周期精确路线。** 先同步最新main并核对已完成范围；K参考final-only输出门/独立15s反例、clock、full/tail九诊断与新JOINT冻结known cancel已有验证，不重复。下列步骤服务于依赖顺序，不是任务数量或自主开发额度限制。

1. **复用计数和接口，生成累计服务曲线。** 从Mapping、Geometry/AccessCounter/13域facts及已修参考提取循环和工作量解析关系，输入K消费与输出块分别统一计量。首输出门槛、full/tail、稳态/排空及端点总量由解析推导，不拟合参考时间；静态SRAM峰值含双输入/累加器/待写输出/转换/中间状态。现有Cursor仅作为离线反例/守恒检查参考，不能转成生产逐K调度器。
2. **验证最粗框架，并同步建立速度基线。** 优先最多5段/每owner三活动，用累计曲线与credit切向约束在稳定区间直接积分。固定外部事件/活跃规模，扩大Core/tile/K/repeat，验证实际事件和状态不线性增长，曲线折点纳入预算；用小规模独立wave oracle验依赖、原3%/5%、工作量/容量和busy绝对误差/参考时长。保留原外部完成、cancel、snapshot/import、容量变更与旧Plan/ABI指标兼容；不新增逐wave fallback。
3. **同阶段优化热查询和仲裁；必要时才到8/16段。** 规范化语义缓存、轻量孤立cost查询、重复查询去重避免命中后重新枚举或生成完整报告；活跃world必须使用正确clock/generation/owner身份。解析/水填充快路径须与原Q/alpha/13域/PRIVATE公平分配等价，不支持的耦合保留原小规模求解。分别测命中P95≤20μs、常规冷编译P95≤2ms、J≤32每次更新P95≤50μs。只有证据显示5段不足时才评估8/16，取共同满足速度/精度的最小者；都不满足标failed/tradeoff_unresolved，不转周期精确、不改busy口径或放宽门槛。
4. **五仓完整负载与剩余接线。** 仅活动新JOINT；domain/native wire、Sys只读sample→Domain、known capacity/cancel已完成，继续生产Sys已知事件采集及真实graph/ledger/lease/active-leg回放，不直接删sh30/`live_world_import_supported=false` guard。FACE固定E/protocol2/startup/source-release及原129层压力复用；扩其他端口/拓扑/竞争/GQA，保留KV/weights/TP/restore/pool/ACK及本地发送完成→容量复用顺序，WS typed兼容与lease不回退。固定事件输入和完整在线策略分开测CPU/墙钟/RSS/缓存/事件/LP及原端到端开销预算。
5. **校准继续遵守数据边界。** 无独立目标raw/heldout不重新拟合TX8或目标效率，不以BF16/DDR参考代替INT4/HBM测量；reference/target参数冻结。新模型精度/速度未验收，不能把文档更新、oracle诊断或GitHub合入当作通过。

## Important Decisions

### 本轮解析实现决定

- 服务描述数是每epoch各独立活动的曲线段之和（无活动epoch仍1），不能仅数epoch或合并同progress的独立终止。v1/v2旧计数低估，当前共同7描述选8预算；不假称5段已达标。
- 首个整K输出cohort用有界类表提取真实工作并单独终态释放；末输出在最后Compute完成后排空。prefetch credit为K消费单位，输出credit为块，SRAM另做精确byte峰值；无per-K内部生产回放。
- 曲线使用扩展进度精度与已发布分配续算；物理/PRIVATE验证沿用原LP精度边界，credit切向不允许向外漂移。终态闭合必须检查整个epoch容量，不能恢复旧盲进度epsilon或增大credit容差。
- 默认五仓旧模型暂不升级；当前新源是未验收开发检查点。真实旧fallback通过不代表新轻量符号/新模型整合通过。新memo为收束最小实现，正式性能结论待下一轮。

以下为历史记录，冲突时以上方最新检查点为准。

- 2026-10-15用户明确：数学公式分析模型为生产唯一主线；周期搜索/精确周期/大整数周期/逐K回放不进入方案，历史Program归档不恢复为默认待办。只在离线小规模oracle复用其有效计数/反例，不以状态固定或opt-in绕过事件成本限制。
- 单段解析曲线在工作进度上定义，输入/输出单位各自一致，触边更新速率可行域而非逐块轮换；折点统一预算，固定外部序列的事件放大测试和原速度/精度门槛同时执行。

- 快速交接时保留未完成Program为可重放patch+新文件快照，恢复默认源到已验证clock release；不将未验收wire/Runtime发布五仓。归档逐字节及前向/反向patch检查通过，源码有效成果未丢失。

- 新JOINT known cancel首片是native冻结事件能力：取消为终态阻塞，不作为自然成功传播依赖；无确定lease ownership不释放gate。保持默认旧wire，完整生产live world/sh30仍未开放。

- 时钟只拆整数原点和小数，不改变服务模型或引入精确有理数；绝对时刻无关性已有真实反例和72项回归。旧wire保留，C++ Runtime布局变化要求所有使用方重编译，本检查点五仓双ABI已实做。尚未启用任何周期跳跃。

- 本轮Descriptor/Cursor是内部基础，六计数器只表达严格序的离散就绪；没有新增生产Plan schema、signed-credit服务行或压缩。规范发布现在canonical24，未执行的生产/业务测试不因新摘要升级。最新诊断参考已修final-only输出门；历史raw/报告不重写。

- K-readiness新诊断是部分研究代码，实际 public Runtime 服务13域facts，但facts本来没有signed-credit，硬槽reservation/终态队列提供就绪；这不证明生产signed-credit行或周期路径已接通。只读旧方案包含尚未采纳的credit生命周期构想。此处是早期诊断历史：output满时非final不应阻塞的参考规则现已由c13e9a7修复，后续复用已修参考，不重复该修复。
- 预算口径已核：Runtime `advance_to` 的1e6是单调用内部实际interval guard；256 simplex pivots是一次`allocate()`内各tier/fallback共享预算；standalone estimate另有累计driver event_guard；研究scratch的expanded/discovery cap是独立口径。下一实现应分开逻辑完成/跳过与实际探测/求解计数，不能混称lifetime256事件，也不任意扩大原预算。

- Runtime保存非递减lexicographic minimum是原max-min定义的不变量，不是扩大credit容差；owner共享progress变量与canonical Q/alpha数学等价。共享scale单独修复不足，TP尾片和中间失败必须保留；最终不增加LP行。相同内容的canonical发布不改mtime，仍逐文件校验digest。
- 简单公式优先原则继续有效；复杂度仅服务于源架构/任务输入的可观察响应、容量/工作量守恒和真实依赖/竞争。native wrapper 是现有 Runtime 的生命周期桥接，没有第二套数值模型。
- FACE v2 strict batch/revision/schema不能静默退回v1；旧typed raw+diagnostic共存适用于WS/FACE，JOINT独立新wire互斥边界保留。活动租约桥接与future E/source物理释放分开认证。
- die 是唯一 canonical；5 仓同 digest，Python不重写 timing/capacity 公式。只改活动 JOINT；old已在main移除，历史资料从Git历史查，不恢复目录。
- 顺序 layer/pass 不乘 temporary peak；compact span/batch 的当前物化集合语义保留。版本变化显式，旧证书不可复用。
- future E 仍是 fit margin，`max(A,E)`，不计 active/KV/remote；actual Plan 超界在C++ lease/issue前拒绝，Python provisional KV事务不能由C++回滚，启动证书负责其前置覆盖；正常串行 compute gate 是 max 而非 sum 的前提。source ACK/物理完成释放与容量复用排序独立保留。
- fresh/query 与同冻结 native-session wire parity 均不是生产 Sys live snapshot/native integration 完成，也不证明目标性能。禁用 unsupported legacy active state 时显式错误，不能给 scalar ETA 假恢复；生产 live import guard 保留。
- 实际 sample→Domain 仅恢复局部物理Runtime与pending completion；显式外部leg ID和expected rank，terminal清理复用canonical destroy。积分切分产生的四项busy Decimal差值单列ungraded，不用其余字段的严格相等冒充整份wire严格全等。
- TX8 conditional reference：eta_compute_reference=0.9079551923877489、eta_vector_memory_reference=0.3411668292682927；Norm246us/FFN3363us 的身份与条件解释不变。目标 eta_pe/vec=.90、HBM=.70、SRAM=1、local/mesh/inject=.70、rho=.90、launch1000ns、memory_setup0、kappa_quant4/acc_convert2保持工程先验。

## Files to Continue

### 本轮直接入口

- `src/{Mapping,Execution,Plan,Compiler,Capi}.cc`；`src/{CompilerInternal,PackingAllocation}.hh`；`include/coarse_die/{Plan,Execution,Capi,Memory}`。
- `tests/{service_curve_runtime,service_curve_schema,os_service_curve,os_service_scaling,capi_cost_query,packing_allocator}_test.cc`、`tests/capi_cache_response_test.cc`与`service_curve_benchmark.cc`。
- `validation/analytical_service_oracle.py`、实际`compiled_gemm_oracle_driver.cc`；本检查点analysis目录的run_axis/ratio/busy/integration/README，最终evidence中的verification与原始失败。
- 两FACE `online/die_model_oracle.py`、`face_scheduler.py`；未碰JOINT生产源码或五仓vendor。

以下为历史记录，冲突时以上方最新检查点为准。

- 主线先读根设计§1.5/§8.2.1/§13、AGENTS和PROJECT最新路线；实现位置仍为 `die/src/{Mapping,Compiler,Execution,Capi}.cc` 及公共Plan/Execution接口。归档 `analysis/gemm-os-program-partial-20261015/` 仅供历史核查，不先解压恢复patch；原factory/test/oracle待办均属已撤出主线的未完成Program。

- known cancel：新JOINT `NativeProjection.cc`、对应两项native/C API C++ tests与Python `joint/{resource_projection,native_projection,test_native_coarse_projection}.py`；重放 `analysis/native-world-cancel-20261015/run_checks.sh`。实际使用的vendor clock与公共Program在途版本分开识别。

- 时钟：`die/src/Execution.cc`、`include/coarse_die/Execution.hh`、`tests/runtime_clock_translation_test.cc`；实际重放 `analysis/runtime-clock-20261015/replay_pair.py`。原单Core opt-in Program已归档，不继续该逐K路线；后续Plan/Compiler/Mapping/Runtime按解析服务曲线改造并保持兼容。

- 当前基础：`die/src/GemmOSReadiness.hh`、`tests/gemm_os_readiness_test.cc`、`tests/k_readiness_runtime_driver.cc`、`validation/{compiled,grouped}_gemm_oracle.py`。新复现 `analysis/k-readiness-20261015/run_corrected.py`，说明与证据在 `k-readiness-corrected-20261015`。归档未修probe只作历史。

- 最新部分成果：`die/docs/analysis/k-readiness-20261015/{README.md,replay.py,k-readiness-proposal-original.md,periodic-k-readiness-inventory.md}`，原始源码/日志/JSON与summary/identity/checksums在同名evidence目录。无需依赖旧账号work/路径；replay复制原源码并替换绝对include，固定单形状，交接时仅做语法/帮助检查，未宣称新入口全重放。

- `die/src/Execution.cc`/`die/tests/runtime_credit_scale_test.cc`：已完成数值修复；新证据入口runtime-credit-20261015。下一步主要`die/src/Mapping.cc`与E5 driver/grouped oracle、已归档 `analysis/k-readiness-20261015/k-readiness-proposal-original.md`；不能再把原logits失败当未修。
- 当前桥接：两FACE `WorkspaceAdmission`、Workload/adapter/bridge/committer与Python KV/online；复现脚本 `analysis/face-workspace-20261015/`。新known capacity见NativeProjection及 `evidence/native-world-capacity-20261015/`。下一步future E仅fit margin，保留物理source释放依赖，禁止把ACK/byte状态伪造为已释放。
- 新 native：活动 JOINT `execution_driven/{NativeCoarseDieDomain,NativeProjection}.{hh,cc}`、两新 C++ tests、CMake；Python `joint/{resource_projection,native_projection,die_plan_projection,test_native_coarse_projection}.py`。
- 业务入口：`online/sh30_online_scheduler.py` provider、`face_scheduler.KVCacheManager`（不是假定存在 JointKVCacheManager 类）、`generate_trace.py`、`joint/{joint_cost_model,action_work_graph,pipeline_ledger,work_ledger,native_ledger}.py` 和 C++ PipelineKernel/WorkLedger runtime snapshot。按真实新仓实现迁移。
- Workspace：公共 Mapping/Capi 与 attention/workspace tests；LRU client/manifest/test；之前 finite-manifest/capacity/source-release 证据仍有效，但旧v1容量证书不适用于新v2查询。
- OS：`die/src/{Mapping,Execution}.cc`、GemmOSGeometry/AccessCounter/ResourceFacts，compiled/grouped oracle/driver 和既有 bounded proofs；旧 scratch 仅按其限定覆盖复用。
- 新证据入口：`analysis/{attention-workspace,native-local-projection,gemm-axis}-20261014/` 和对应 evidence；五仓 ABI/config 在 `evidence/five-abi-attention-20261014/`。
- 当前 native wire：`analysis/native-world-snapshot-20261014/`；真实 Sys 只读 ownership：`analysis/sys-local-sample-20261014/`；实际样本桥接：`analysis/local-sample-domain-20261014/`及对应evidence，含portable replay源与脚本。WSCLLM两仓保护证据：`evidence/ws-lease-hardening-20261014/`。

## Tests

### 本轮最终记录（不得外推）

最终一致源码定向Release构建成功（6个指定target）；CTest **5/8 passed、3 failed，6.24s**：packing、OS结构/计量、curve schema、CAPI cost、Python oracle通过。失败原文：

- `coarse_os_service_scaling_test`: `tiles_n65536: live work has no next Runtime event`；K4096原反例已越过，但大N仍阻塞。
- `coarse_service_curve_runtime_test`: `invalid_snapshot: published activity allocation differs from canonical LP`；新快照费率续算尚未解决。
- `coarse_capi_cache_response_test`: 64MiB压力预期失败，实际`entries=709, bytes=17634823`；须先检查4098行的逐行编译错误和实际footprint，不得直接放宽缓存断言。

命令与完整日志在本检查点evidence的`verification.json`及`raw/handoff-{build,tests}.log`。最终新源有上述已知失败，**仅供继续开发，不可声称运行验收或发布五仓完成**。为遵照快速停止指令，不再展开修复或全量测试。

中间证据：v2定向7项4通过/3失败（OS非法4×1网格夹具、曲线snapshot续算、CAPI小形状无正速率），全部保留；合法网格改2×2，v3 OS结构/计量test实际通过。v3 scaling K4096及CAPI仍无next事件，已定位为终态credit舍入，最终closure修复按最后日志验收。snapshot第一次严格物理行拒绝1+2.22e-16的原LP速率，最后改回原LP数值容差、credit仍零容差；未将该中间失败写成通过。

独立Fraction owner-LP 18.62s通过，packing/schema及Python新oracle测试通过（最终Python5项）。v1 M80/N128/K4096与4097，solo/sync/50ms全部shared-busy原阈值failed；solo完成时间接近不等于busy通过。v3新axis未运行。无最终host性能证书：norm hot成本interim ctypes p50约35.4μs、p95约291.8μs，受并行扰动且早于memo，只供失败诊断，不作最终bench。benchmark构建/执行及30样本P95、五仓CPU≤10%仍未验收。

旧FACE vendor真实双ABI构建成功，legacy SHA `7b961c60505ce8e37be3520fe3f363d1cadbe7fb8cf99cced50daa6285690d68`，旧coarse SHA `bf56b1c1b7afd3c91a8cace974db19249d3e82567c3c770eda0e623e741cfddc`，与历史clock版本身份一致；两FACE缺少新符号fallback真实通过。未运行新五仓83、132/151/Online或TX8新校准；旧已通过且未改的参考门/JOINT known cancel无需重复认证。

以下为历史记录，冲突时以上方最新检查点为准。

- 本次仅文档：执行Markdown差异/链接/一致性和非文档Git对象不变检查；不运行编译、生产测试、校准或publisher，不声称历史测试已在本次重跑。最终push/fetch核实main后交付新提示词；下方均为前次源码/归档的真实结果。

- 最终归档：7个tracked diff+4个新源码/测试逐字节保留、gzip/raw SHA和patch前向/反向检查通过；恢复后的五仓canonical24 check通过。归档Program库/driver/schema编译成功，runtime test编译失败；schema与full driver均exit1（mapping_assumptions重建不一致），未跑新Program oracle/全回归。此失败属于未启用片段，不能计入默认源33项通过结果。

- 新JOINT取消：实际canonical clock版native/window/Domain4/4 CTest、Python6/6无skip，portable脚本再跑亦通过；source/import/fork、直接Domain、同tick前缀/terminal wins和非法导入原子拒绝均覆盖。详情见native-world-cancel证据，不能外推完整Online或生产Sys live回放。

- 时钟检查点：Release all build成功，33/33 CTest；72项起点平移、legacy小数和u64边界；旧header+旧库负回归确实失败。full/tail九项诊断重新通过，五仓真实双ABI全部重建，配置20+15+18+17+13=83/83、0skip；库/源码/命令身份见runtime-clock证据。未重跑全部Online、FACE压力、目标held-out或host预算，不以本轮诊断覆盖旧生产E5失败。

- 本轮修正/紧凑基础：定向CTest3/3（含13 Python项）、C++精确15s可达反例、三shape九场景grouped全部passed；对应完整raw/计数序列化严格差值在新evidence。五份canonical24 check/独立tests-OFF configure5/5；未重跑public30/five83/完整Online，不由诊断外推生产E5或效率。

- 当前最小诊断：实际C++构建与 M80/N128/K4096 solo/sync/50ms-offset运行、grouped checker已执行，三场景原输出passed；原始脚本/报告/metadata限制与共同门控问题如上，不计生产验收。K4097、tail、修正门控、压缩/多Core未执行。交接只新增归档/可移植入口/文档；不重复未修改的public30/five83。交接检查含replay语法/--help、raw逐字节/校验和、Git差异检查；实际结果保存在同名evidence packaging-validation.json。

- 最新Runtime source：public30/30、五仓实际配置83/83、FACE相关C++3/3各仓、WS生命周期1/1各仓、新JOINT8/8、FACE Python132/151无skip；两种logits extent各8候选和原CABI任务均通过。原同输入压力两仓0/0，各24完成/GC retained0，逐trace验证source reuse。E5新axis仍failed原门槛。完整身份/中间失败/实际输出见runtime-credit-20261015。
- 最新future证书/源释放：两仓C++各8/8，Python132/132与151/151无skip；各900个实际选中Plan均受界，六个最终Online均C++0/Python0，186/378/270终态分别对应legacy/overlap/cap2+split。大模型压力两仓失败，单实际logits任务原ABI独立status11复现；完整身份、原始错误及修正过的夹具探针在face-future-workspace-20261015。
- 当前FACE两仓选定C++各8/8；真实protoc/双ABI Python120/120与118/118，0skip。最终六个真实Online C++/Python均0/0；四tiny各186成功终态、两overlap各378，实际positive workspace绝对状态见Current Status及独立evidence。16项C++中的最后LRU mailbox测试单独补建后全8项重跑通过。
- 当前known-capacity native/window最终4/4，direct Domain同clock完整snapshot、source continuation/import和invalid atomic均通过；最初两次夹具层级/时钟比较错误保留，未放宽字段/容差。
- 公共 Release 默认 all build 成功；30/30 CTest passed，43.49s。这是 attention/public production 源的真实运行；另一 agent 同时开发的诊断 utility 另做最终独立 build/11项测试，不冒称其未稳定版本已被旧 all build覆盖。
- LRU 最终 portable real-protoc client/manifest suite：19 total，18 passed/1 archived pre-envelope ABI unavailable skip；v1 revision rejection 是实际 ABI response 的回归。早先 marker shim setup 作废，正式 runner 不 mock generator/Path.exists/路由。
- 四个非 JOINT 双 C ABI Release 全成功；显式真实两库 config 测试20+15+18+17=70/70，0skip。coarse库 SHA `14c35c774333814de2d6403696db2280c4f8d4d2eec9024f1fd5c0883307509e`、legacy `7b961c60505ce8e37be3520fe3f363d1cadbe7fb8cf99cced50daa6285690d68`跨四仓一致。
- 新 JOINT actual Online、joint_projection、双Die ABI、joint_ledger C ABI 均构建成功。NativeProjectionCApiTest、NativeCoarseProjectionTest、NativeCoarseDieDomainTest 3/3 pass；后加slot/join fixture增量test pass。真实两Die ABI的Python legacy+new native suite17/17，0skip。
- 新 wire 最终 native/window4/4；独立新 build 重建并执行4/4 CTest+4/4 Python，0skip，两次 projection .so SHA相等。实际 legacy protocol+coarse facade18/18无skip；再次 tiny legacy Online 0/0、192/192成功。详见新 identity-and-results.json；未把 raw Runtime export 或两个同样 import 的 world 自对拍当原 source continuation 验证。
- 新 Sys 只读采样检查点：真实 CoarseWorkspaceRuntime、DieModelAdapter、DieExecutionLifecycle、LocalHbmDieMemory 及四项 native/window 合8/8通过；实际样本 tick1/原队列2408ns/303318ns与无 live mutation 校验通过。当前源码 Online 重链并再次0/0、192/192成功；canonical23 check不变。source/binary SHA 和 ledger stale48 限制见 sys-local-sample evidence。
- 实际sample→Domain最终4/4 native/window+真实库Python4/4无skip；portable actual Sys replay的完成时刻/served/整数/registry精确匹配，四项busy字符串不严格匹配，原始失败和Decimal差值完整保留。WSCLLM两仓独立真实生命周期各1/1通过、直接二进制63/54断言通过；没有在该阶段重跑其Online或广泛suite。FACE后续独立检查点结果如下，不由该阶段外推。
- 新JOINT tiny legacy Online C++0/Python0，1accepted/1completed、3delivery/3batch、192success terminals、0incomplete；coarse-intended Online在provider guard失败如实保留。当前服务只有strategy/Sh30入口，face/load-first也在策略选择前触发同一provider guard，故无合法非sh30 coarse Online旁路。JOINT config13/13无skip，与另四仓合83/83；focused adapter/workspace Python45/45无skip。
- E5 utility 最终11项回归与独立 C++ target build通过；model-to-model precision仍failed，strict cumulative scalar equality/数值误差分别报告，不以Python自对拍当生产证书。
- 未跑新的目标held-out、sanitizer、malloc fault、当前host预算或完整五仓业务。既存GCC __int128/JSON helper dangling-reference warnings保留：已读helper返回persistent JSON引用，未把警告直接断言成UB，也未声称完整内存安全审查。

## Blocking Issues

### 当前新增限制

新源未完成广泛回归/精度/性能验收且未发布五仓；SRAM/工作量及schema测试通过不等于时间模型准确。v1 busy失败、v3范围/最终速度未知；WR/其他模板尚旧聚合；memo失效/内存与snapshot极端输入仍需独立覆盖；无target raw。旧完整业务/JOINT liveworld/sh30/开销缺口仍在。不要把本次GitHub交接完成写成工程ACCEPTED。

以下为历史记录，冲突时以上方最新检查点为准。

归档Program尚未验收：上述factory metadata/test入口错误、完整snapshot/cancel/capacity/前缀累计审查、新来源oracle、周期批量累计/效率及默认兼容还需完成；现有明确恢复顺序见Next Step和归档README。

工程验收缺口：sh30/完整native live world、network/pool/ledger/cancel回放，E5精度/最小有效E，其他容量/拓扑/端口竞争压力、完整GQA/FACE future/source-release全业务验收、成本余项和host预算。目标测量not_available。以上不应由merge、fixture或条件reference数字冒充通过。

生产 e73a378、K诊断修正c13e9a7及先前源码检查点均已进入main并实核。本次时钟修复按普通git同步main并实核；精确SHA由包含本文的Git提交确定。工程NOT ACCEPTED与代码检查点安全分开，不能隐瞒未通过项。

## Handoff Notes

### 本次停止/交接

用户明确要求完成当前最小任务后迅速保存、同步main并停止。未启动新开发、未publisher；本轮源码/新测试直接提交，原始失败、未运行runner、推导及旧ABI证据一起入库，不提交ELF/build/cache。共享分支以最终git fetch相同HEAD实核为准；GitHub API/gh既有Forbidden限制，采用普通git fast-forward main，不能伪造PR/CI。合入后本Session立即停止。精确提交SHA以包含本文的Git提交和最终答复为准。

当前环境实际有`/workspace/cloud-setup/activate.sh`（GCC14.2/CMake3.31/Python3.12/protoc），历史“不存在该路径”只属于之前环境；下一账号重新检查，不依赖work ELF。此处entry scripts显式写work，原始报告中的绝对路径只作来源身份，不假定跨账号可读。

以下为历史记录，冲突时以上方最新检查点为准。

- 最新任务仅更新文档，保留用户5e2fb0a及其全部代码/归档/删除成果；本次不恢复/清理任何源码。新任务按RELAY_PROMPT持续开发，建模路线以上方为准，历史周期建议不再生效。

当前用户明确要求迅速交接并停止。本轮canonical24含未启用内部基础及绝对时钟修复，校准冻结；未验收Program整片归档而不发布。所有有效修改、成功/失败与源码身份随证据保存，没有修改old、覆盖其他人提交或丢弃有效修改。最终main实核后本Session立即停止，不启动后续任务。

本轮新环境的CMake3.31.10在Python venv `work/tool-env`，GCC14.2/Python3.12.14；新JOINT真实protobuf3.21.12开发依赖隔离在work/protobuf-root，精确包/命令见native-world-cancel。旧 `/workspace/cloud-setup/activate.sh` 说明属于之前环境，不能假定新账号存在。work内依赖、build、ELF、stage不跨账号交付；复现入口明确需要重建。`work/build-joint` 的旧CMake源是 `work/joint-reproduction/aggregate`，不要在旧cache换源重配；新账号优先新目录。公共库入口 `cmake -S die -B work/build-die -DCMAKE_BUILD_TYPE=Release`；K/clock/JOINT及归档Program各见对应README。其他未改且通过的结果直接复用。

GitHub API/gh受Forbidden限制，没有新PR链接或可核实CI；本轮采用普通git同步工作分支并fast-forward main，最终以远端相同HEAD/祖先关系验证，不force-push。最终提交SHA由Git历史与交接最终报告确认；如果main有新增外部提交，必须保留并先整合，不能强推。下个账号基于含本交接文件的最新main接棒；工程未验收项按本文继续，不能因代码合入标成通过。
