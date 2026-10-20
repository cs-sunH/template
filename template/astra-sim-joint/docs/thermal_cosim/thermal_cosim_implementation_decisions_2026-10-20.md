# thermal/cosim 内嵌 joint：实施报告问题答复与补齐裁决

日期：2026-10-20（Asia/Shanghai）  
审查版本：`cs-sunH/template@12422091a147982d0044f6dedba1672bba6a9c3e`  
依据：[实施报告](https://github.com/cs-sunH/template/blob/12422091a147982d0044f6dedba1672bba6a9c3e/template/astra-sim-joint/docs/thermal_cosim/thermal_cosim_implementation_report_2026-10-20.md)。本答复裁决接口、实施顺序和验收边界，不代表这些补齐项已经实现。

## 1. 当前完成状态怎么认定

可以认定热核心已内嵌、会话与功率账本已有实现、执行降额路径已有原型和报告中的在线证据；现在不能认定“热反馈已正确进入生产联合决策，完整工程集成完成”。目标器件精度也尚未验收。

需要把报告中 D/E/F/G 的“完成”细分到具体能力：当前缺口同时涉及活动覆盖、cap 的物理目标、同 tick 时序、生产候选估价、存量状态接续和独立分发，并不只是补一项缓存键。报告的 +42.5% 延迟只能说明夹具引起了执行减速，不能证明逐 die 降额正确或联合选择已使用降额。

继续保持原始分工：joint 解析负载、软件/物理映射和硬件，负责温控策略及联合选择；cosim 根据输入活动估计功率并推进温度，不接管 placement、P/D 或 KV 决策。

## 2. Q1—Q12 的明确答复

### Q1：configs 移到 wafercosim/configs，接受

接受，并建议保留当前位置。原方案的根级目录是组织示意，不是必须逐字遵守的接口；原方案已允许 package data 或明确配置路径。当前 wheel 已声明打包 JSON/YAML；本次在临时目录构建 wheel、安装到隔离目录后，确认含 15 个配置资源，thermal、energy、fixture 配置均可读取。

要求是资源定位不受调用者 cwd 影响、正常安装后可用，不要求本轮额外支持 zipapp。包内默认配置只保留一个权威副本；运行配置、审计文件和生成物不要写入安装目录。`package:` 引用应明确走包内资源，避免 cwd 下同名文件遮蔽。

依据：`extern/thermal_cosim/pyproject.toml:19`、`wafercosim/paths.py:11`。这项可以关闭。

### Q2：允许 Python 解析 rank→die，但必须消除身份歧义

接受由 Python 使用 joint 导出的 `MappingSnapshot` 解析，C++ 无需知道冷却网格。C++ 应发送明确的执行源身份，例如 `source_kind=rank_compute, source_id=rank:20`，由 Python 解析为物理 die/热块；不应继续用看似物理 die 的 `die20.comp` 同时表达 rank 20。

当前代码不能原样保留：`_mapping_source()` 在 rank 缺失、甚至其映射后的源不存在时，会尝试把相同编号当 thermal die。必须移除这种隐式回退。closed_loop 初始化要验证所有参与 rank/资源源的映射完整性，不是只验证字典非空；缺失项应列出具体 rank/source 并拒绝启动。

本次实际复现：映射仅有 `rank_to_die={7:0}`，物理源存在 `die0.compute`，输入执行源 `die0.comp` 时，当前函数仍返回 `die0.compute`，尽管 rank 0 没有映射。

一 rank 覆盖多个物理对象时，要扩展执行源/控制域标识，不能用单值 rank→die 强行代替。多 rank 共享同一器件时，动态贡献可以聚合，器件背景功率必须只计一次。映射来源应是 joint 的硬件/拓扑与显式物理配置，不能按取模或参考网格猜测。

依据：[S1]。

### Q3：不接受当前多域“对所有 rank 安装、末条覆盖”的近似

需要双端升级带目标的 cap 协议。建议保留稳定 `control_domain_id`，由 Python 根据映射生成显式 `target_ranks`，并携带 `resource_index`、绝对速率、`effective_from_ns`、资源 epoch、硬件/映射身份。若一个 rank 内有多个独立执行域，再增加对应的本地域 ID。

C++ 只对目标 runtime 安装；一个批次中每个目标资源只能有一个最终有效速率。多个限制应由 policy 按声明规则合成，例如取硬件峰值、其他上限、热上限的最小值，不允许按事件顺序覆盖。先校验全部目标、时间、速率和版本，再安装并确认实际生效 epoch；cap 与执行图在同一回复时须保证先安装 cap、再提交图。

新增最小验收：两个独立控制域，A 热到 0.5、B 保持 1.0；分别检查 live runtime、预测输入和后续活动。再测恢复、共享时钟域、非法目标和重复事件。只有真实硬件就是一个共享控制域时，才允许明确声明一个统一 cap；当前多域最后一条覆盖所有 rank 的行为不属于这一情况。

依据：[S2]。报告的 `cap_events_applied=12528` 是安装调用计数，不是热 epoch 数，也不是逐域正确性的证据。合成配置有 54 ranks、4 policy domains，`12528/(54×4)=58` 个含 cap 的回复；周期回调 27 次之外，普通 bridge delivery 也会触发采样与回复。

### Q4：V12 必须保留

必须证明至少在合适夹具中，生产联合选择确实因热降额而改变；并不要求所有生产工作负载每次都改变选择。仅证明 cap 能进入执行或 native 测试接口，不满足原始目标。

构造同一个决策前快照的两臂：相同候选集合、负载、队列、拓扑、KV/配额状态、seed 和策略，只改变相关域的 cap/epoch。示意目标是原先候选 A/B 的真实估价为 100/120 ns，仅 A 降额后变为 200/120 ns，生产 selector 从 A 选到 B。数值应由真实图和投影得出，不能注入期望成本或强制选择。

对 P、D 各自入口验证所需路径，记录候选成本、资源 epoch、选择和事务提交结果；保留明显差距，避免平局、候选顺序或两条轨迹的不同积压导致伪结论。先测人工 envelope 的因果影响，再补温度→policy→envelope 的端到端夹具。现有 `--combo none` 单请求执行减速证据不能代替该项。

依据：[S3]、[S4]。

### Q5：保留现有单调性测试，另加纯 gate 的 V08 解析夹具

不能用“单调推迟+恢复”替换 V08。原方案已经限定 V08 为单资源、无其他瓶颈的 gate-only 合成工作；多资源、多阶段的 coarse `norm` 夹具本来就不应套用该解析式，也不需要为此改变生产 coarse 模型。

纯 gate 总工作为 100 ns 的全速服务量，40 ns 时降为 0.5：剩余 60 ns 服务量耗时 120 ns，finish 为 **160 ns**。若 80 ns 时恢复为 1.0，40–80 ns 仅完成 20 ns 服务量，此后还剩 40 ns，finish 为 **120 ns**。恢复不能追回已经损失的服务时间。

用这一独立夹具同时验证真实 runtime 与 projection、整数 tick 舍入、旧完成事件失效和不重复 callback。保留当前 coarse 集成测试验证混合资源行为，并补 V09/V10。当前测试中的“恢复回基线”属于该具体 workload 的表现，不是任何已被 gate 限制的工作都应满足的普遍性质。

### Q6：因果同步必须满足，当前架构不必增加 MPI barrier

当前 executable 的所有模拟 rank 使用一个全局 EventQueue，热回调一次遍历全部 runtime，bridge 同步返回后才继续队列。因此可用进程内统一事件阶段实现共同 watermark，不必先增加跨进程协议。

但现实现尚不能直接通过时序验收。周期热回调在同 tick 的普通 FIFO 事件表里执行，完整物理事件表之后才进入 tick-end。若周期事件与 arrival/completion 落在同一 tick，周期回调先提交热窗，tick-end 再发普通决策；provider 再次执行会形成 `[t,t)`，被 Python 拒绝。

推荐让周期回调只标记 `thermal_due`。在 tick-end 完成旧区间物理结算后，统一闭合一次活动窗、更新热状态、确定 cap，再处理该 tick 的普通决策。也可采用等价的独立握手，但同 tick 后续帧只能读取最新 view，不得重复积分或发送零长度活动窗。真实多进程/独立事件引擎以后才需要 collect/install ACK 的跨进程协调；未支持时明确拒绝该拓扑。

补充测试：周期事件与到达、完成、取消及恢复同 tick，颠倒事件插入顺序，仍得到相同账本、cap、选择与 callback 次数。每个参与源必须覆盖到共同截止时间，不能把“没有返回样本”视为已确认零活动。

依据：[S5]。

### Q7：要补资源采集与覆盖诊断，不能把所有问题概括为“缺系数”

最终纳入哪些资源，按 joint 硬件和负载的实际能量范围决定：SRAM、NoC、D2D、HBM、pool/外部 DRAM 等产生的活动应被采集、映射到相应热块或 external，并明确系数支持域。带宽降额可以后续扩展；这些资源的能量采集不能因为第一版只控制 compute gate 就自动省略。

必须分别记录：①动作是否采集，②测得计数是否为零，③能量系数是否可用及其来源，④空间映射是否完整，⑤目标校准是否通过。未采到、未支持、负差/计数损坏、缺系数、实测零是不同状态。

报告“报错而非当 0”的结论仅对已进入 Python 的未知 action/缺系数等情况成立。当前 C++ 只产生 PE、Vec、HBM 四类 action；其他资源根本没被发送。同时 invalid/首个样本/负计数差会省略 source，HBM 行缺失可保留默认零，帧却仍固定标 `measured`。这会让未采集活动伪装成零活动，必须修。

合成工程夹具可以限定资源范围并使用明确标注的合成参数，但不能据此宣称完整真实负载覆盖。真实目标模式缺必需计数、映射或系数应报错；补系数前先把采集及覆盖状态做正确，不能造一个零系数消除错误。

依据：[S6]。完成对象的累计 ledger/tombstone 与跨窗不漏不重仍属 V06 必做项；首个采样 baseline 应在执行前建立，或在历史完整时从零累计量恢复，不能无条件丢弃首段服务。

### Q8：接受未校准的工程交付边界，但有前置条件

接受以“可复现的合成/研究工程闭环，`target_calibrated=false`”作为本轮范围；前提是其余工程正确性和联合决策接线验收完成。这个标记不能豁免错误映射、缺失活动或未接入成本路径。

当前仓内有参考模型和上游来源，没有足以认证目标器件的动作能耗、SRAM 宏、OPP、floorplan/堆叠/TIM、冷却流量及测量数据。RC 候选也明确写了 `empirical_silicon_calibration=false` 和 `scientific_accuracy_gate=failed`。不能通过放宽域检查或清掉 failed 标记把参考参数变成目标校准。

H 阶段单独开展：按目标配置取得可追溯的动作能耗和物理参数，在留出轨迹上评估温度/功耗误差。RC 对 3D-ICE/HotSpot 的一致性不等于对真实器件的准确性。当前无需等齐这些数据才补工程接线。

### Q9：这份小型复现输入可以入库，按测试夹具管理

当前一个请求、约 304 bytes 的 CSV 和 `.provenance.json`、`.canonical_digest.csv` 已在 Git 跟踪，而原始 compute trace 不随当前仓库提供。为保留现有三臂复跑能力，先保留这组三件文件，并明确它是 smoke fixture，不能把它当完整 workload 回归。

建议后续给它稳定的 fixture 名称/目录，由命令显式选择；日常批量物化的队列、run 输出、审计与性能结果不常规入库。不要仅加 `.gitignore` 就以为已跟踪文件会被忽略，也不要在尚无替代输入时直接删除唯一可复跑队列。若迁移路径，须同步修复运行入口。

provenance 应一起记录生成器版本、参数、输入 trace 身份及 hash、输出 hash/摘要。当前 sidecar 主要记录输出摘要，打印过源文件 MD5 不等于 sidecar 保存了源 hash；这部分建议补上。

### Q10：先修物理目标、时序和活动，再闭合生产决策；缓存不是唯一第一优先级

按下表实施。P1 的两项可以并行，但 V11/V12 必须等它们汇合后验收。

| 优先级 | 工作 | 退出条件 |
|---|---|---|
| P0-1 | 完整 rank/source 映射校验、cap 明确控制域/目标、协议版本与真实能力检查 | 热 A/冷 B 的 live 速率分别正确；无隐式回退、无末条覆盖 |
| P0-2 | 同 tick 一次闭合、统一事件阶段、cap/图安装与 ACK 顺序 | 到达/完成与热边界碰撞不崩溃、不重复积分；所有源截止一致 |
| P0-3 | 活动 collector、首窗/完成对象历史、逐资源覆盖诊断 | V06 与必要计数覆盖通过；未采到不会当零 |
| P1-A | 决策前推进热状态；P/D 使用一致只读 view；成本和选择缓存带版本 | 同一决策所有候选读取同一 view；热变化不复用旧派生成本 |
| P1-B | live runtime sample、当前容量、Owner/leg/依赖接续到 production projection | V11；存量 Plan 不以 fresh 整条再提交，生产 action graph 真正传入资源状态/事件 |
| P1-C | 生产 selector 的人工 cap 与温度触发两层夹具 | V12 改变选择，KV/配额等约束保持 |
| P2 | 运行配置物化与路径收口、standalone 测试、完整矩阵及回归/性能 | V01–V16 按项记录证据和退出状态，独立目录可复跑 |
| H | 目标硬件模型校准 | 独立精度证据；未完成时持续标记 false |

**生产接线的具体要求：**

1. 当前 `ThermalCostView` 只有定义与测试引用，生产 `JointCostModel` 构造没有接入它。capacity-event 已有 native 接口和测试，不等于在线候选评分已经使用；生产 action graph 未传入对应 coarse 资源状态/事件。
2. 当前热处理是在 `_process_arrivals()`、`run_variant_policy()` 和 `build_graph_batch()` 之后，才由 `_attach_thermal_cap()` 推进 session。要拆分“先消费活动、生成/冻结 view”与“回复中序列化 cap”：前者必须在候选选择前发生，后者不能再次积分。与 C++ 联合回复时，候选可在待安装 cap 的一致视图中评估，但图提交须以该 cap 校验和安装成功为条件。
3. P/D 各决策读取当时最新一致 view，不要求不同 tick 永远使用同一个 epoch。一次决策内固定硬件/映射/资源版本及 live sample；评估期间版本变化要重新评估。shadow 可以审计拟议 cap，执行与候选成本必须继续使用 off 的有效速率，以维持选择和物理结果一致。
4. 热相关的派生成本、P 的 L1 模型和 L2 选择缓存、native world/session 应按依赖更新。至少带硬件 hash、映射 epoch、资源 epoch；直接依赖温度时再带热状态版本和相应 horizon。纯负载原始量缓存无需无差别清空。
5. `NativeProjection` 当前从 fresh domain 起步并声明 `live_world_import_supported=false`。已有 projection-session 自身 snapshot 不等于生产 `Sys` 的 live 状态。要接续服务进度、当前容量、稳定 Owner/leg 及完成依赖，结合现有图/网络/ledger 因果状态；一个 `remaining_ns` 标量不能替代多资源存量状态。
6. 配置物化跟随真实在线调用链，不以某文件 `thermal` 关键词引用数作验收。`generate_face_trace.py` 的独立静态 CLI 已停用，但其配置加载 helper 仍被 `plan_materializer.py`、`online_service.py` 使用；应在共享解析/物化层接入，不必恢复已停用流水线。最终 run manifest 应记录解析后的配置身份与内容 hash，缺映射在启动执行前拒绝。

依据：[S3]、[S4]、[S7]。

### Q11：要求自动化复制分发测试，当前确有可复现阻塞

要求，属于“直接嵌入 joint、可独立运行”的完成条件。入口 `JOINT_THERMAL_CONFIG` 使用绝对路径本身没有问题；问题是 shadow/closed_loop JSON 内部的 policy、hardware facts、mapping 硬编码 `/home/sunhao/thermal/...`，读取时直接 `Path(...).read_text()`，复制后仍指向旧位置。

内部相对引用应以配置文件所在目录或明确 run root 解析；包内默认资源保持独立的 package 语义，不应碰巧按 cwd 命中。manifest 记录解析后路径、内容 hash 与来源，避免只记一个原始字符串。

测试至少包含：仅复制 joint 子树及声明依赖到临时新目录；隔离 Python 环境，不借用原仓 editable 安装或原目录 PYTHONPATH；从其他 cwd 安装/加载包；在新目录构建或按声明部署 native 依赖；使用仓内小夹具跑 off/shadow/closed_loop，并验证 shadow/off 一致、闭环存在正确目标 cap 与选择证据。测试环境中原兄弟 thermal 和旧绝对路径应不可访问，防止假通过。

当前 acceptance 的 `V01_standalone_package` 只在源码目录运行热核 unittest，不能证明上述能力。wheel 包资源通过也只能证明包安装这一层。

依据：[S8]。

### Q12：允许新增 C++ 测试目标并注册 CMake/CTest

允许，属于本次集成的正常范围，可直接实施。优先复用现有轻量 core 测试目标；需要覆盖独立协议或生产 runtime 时再新增窄目标，不必为名字重复构建一个相同 executable。注册为 CTest 测试，让统一验收入口能执行并关联 V08/V09/V10/V11/V07 证据。

现有 `NativeCoarseDieDomainTest`、`NativeCoarseProjectionTest` 已有注册先例。仅测底层 domain 不足以证明生产 `CoarseDieRuntime` 的完成事件失效和 callback 行为，后者需要对应测试。缺 native 构建产物应记录 not_run；完整验收模式缺必测项应返回非零，而不是以没有 fail 就宣布通过。

## 3. 本次额外发现与报告建议修订

建议在实施报告 §4 新增下面的明确问题，而不只写“待确认”或“关键词引用数 0”：

- cap 对所有 rank 重复安装、最后一个域覆盖：确定的目标定位缺陷。
- rank 缺失时回退为同编号热 die：确定的 fail-closed 缺陷，已用最小调用复现。
- 同 tick 周期热帧和普通帧可能生成零长度窗口：按当前调用链可达的协议缺陷，仍需 native 碰撞测试固化。
- 热推进发生在选择之后：生产决策阶段顺序需要调整。
- 未采集/invalid/负差活动仍可被标记为 measured zero：覆盖与计数审计缺口。
- cap 的 native 单测能力未接入生产候选输入；live continuation 仍不支持。
- 分发配置包含开发者机器绝对路径：确定的重定位阻塞。

矩阵应按 V01–V16 分别列出 pass/fail/not_run/不适用及原因，避免一个聚合测试通过就代表其中所有 V 项通过。当前脚本的 V14 证据主要是同轨迹重放，不足以证明热步减半收敛；后者应预先约定温度/能量/完成时间误差预算，记录选择是否变化及原因。完整验收入口要区分快速子集与完整模式，完整模式不能因缺必测项仍返回成功。

## 4. 本次实际验证范围

本次在固定快照上执行了只读代码核对和以下检查：

| 检查 | 结果/边界 |
|---|---|
| 内嵌热核心现有测试 | unittest 共运行 32 项，其中 1 项因缺 3D-ICE native bridge 跳过；其余通过 |
| joint thermal epoch 现有测试 | 20 项通过 |
| wheel 安装与包配置读取 | 隔离构建/安装；15 个配置资源入包，三类配置读取成功 |
| rank 缺失的 `_mapping_source` 最小调用 | 复现错误回退为同编号物理 die |
| 现有 `thermal_acceptance.py` | 4 pass / 0 fail / 4 not_run；现有 8 个聚合检查键不等于完整 V01–V16 |
| 报告中的真实在线三臂、ABI、native projection | 本环境无对应 C++ build/libs，未独立复现；不据此否定报告已有运行记录 |

本轮 thermal core 的精确口径是“32 项运行，含 1 项 skip”，不将 skip 计为通过；按此统一报告中的 pass/skip 表述。当前缺 `google.protobuf` 的独立在线 Python 导入也需和 native 构建缺失分开诊断，不能都隐藏在同一 native guard 后面。

## 5. 固定版本源码证据

以下链接均固定在审查 commit，避免后续 main 变化导致定位失效。路径均相对于 `template/astra-sim-joint/`。

- [S1：rank/die 回退逻辑](https://github.com/cs-sunH/template/blob/12422091a147982d0044f6dedba1672bba6a9c3e/template/astra-sim-joint/sh_test_mesh/workload/llama2_7b_inference/joint/thermal_epoch.py#L102)：重点 148–155 行；`thermal_bridge.py:400` 只检查 rank map 非空。
- [S2：cap 安装到全部 runtime](https://github.com/cs-sunH/template/blob/12422091a147982d0044f6dedba1672bba6a9c3e/template/astra-sim-joint/astra-sim/network_frontend/analytical/congestion_aware/main_online.cc#L764)：重点 788–800 行；cap 构造见 `thermal_epoch.py:275`。
- [S3：热处理发生在策略之后](https://github.com/cs-sunH/template/blob/12422091a147982d0044f6dedba1672bba6a9c3e/template/astra-sim-joint/sh_test_mesh/workload/llama2_7b_inference/online/online_scheduler_base.py#L499)：到达、策略、图构造之后 521 行才 attach/ingest。
- [S4：生产成本构造和缓存](https://github.com/cs-sunH/template/blob/12422091a147982d0044f6dedba1672bba6a9c3e/template/astra-sim-joint/sh_test_mesh/workload/llama2_7b_inference/online/sh30_online_scheduler.py#L7360)：构造及 L1/L2 见 7360–7559；`joint/thermal_cost.py:19` 当前只有定义和测试引用。
- [S5：周期热回调](https://github.com/cs-sunH/template/blob/12422091a147982d0044f6dedba1672bba6a9c3e/template/astra-sim-joint/astra-sim/network_frontend/analytical/congestion_aware/main_online.cc#L1422)：配合 [EventQueue 的 tick 阶段](https://github.com/cs-sunH/template/blob/12422091a147982d0044f6dedba1672bba6a9c3e/template/astra-sim-joint/extern/network_backend/analytical/common/event-queue/EventQueue.cpp#L63)、`DecisionBridge.cc:1096` 的每帧 provider，以及 `thermal_epoch.py:202` 的时间校验。
- [S6：活动差分与 coverage](https://github.com/cs-sunH/template/blob/12422091a147982d0044f6dedba1672bba6a9c3e/template/astra-sim-joint/astra-sim/network_frontend/analytical/congestion_aware/main_online.cc#L638)：采样、首帧、负差和固定 measured 标记见 638–751。
- [S7：生产投影输入](https://github.com/cs-sunH/template/blob/12422091a147982d0044f6dedba1672bba6a9c3e/template/astra-sim-joint/sh_test_mesh/workload/llama2_7b_inference/joint/action_work_graph.py#L249) 与 [NativeProjection live import 能力声明](https://github.com/cs-sunH/template/blob/12422091a147982d0044f6dedba1672bba6a9c3e/template/astra-sim-joint/astra-sim/workload/execution_driven/NativeProjection.cc#L1480)。
- [S8：分发配置中的绝对路径](https://github.com/cs-sunH/template/blob/12422091a147982d0044f6dedba1672bba6a9c3e/template/astra-sim-joint/sh_test_mesh/workload/llama2_7b_inference/thermal/synthetic_run_config_closed_loop.json#L8)；shadow 文件同样；`joint/thermal_bridge.py:177`、`:567` 直接读取引用。

