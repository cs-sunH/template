# 单 die 仿真器设计文档：最粗粒度参数响应版

版本：coarse_die_v2（审查修订，待实现）  
日期：2026-10-13；解析路线及硬件输入/10%—20%精度要求更新：2026-10-15<br>
推荐实现：C++17 公共库，C ABI，五仓库各保留薄适配层  
建议源码位置：template/die/；交付给各仓库的固定版本位于 extern/die_model/  
资料基线：cs-sunH/template，提交 9fdcad1f8a1640d3e91558e3b2d11c55c66b53c8

本次修订根据外部评价补齐模型假设和实施契约。没有新增硬件实测，也没有把参考公式复算计为新仿真器验收。

| 审查项 | 判定 | 本版修订 |
|---|---|---|
| P0：HBM 入口可能掩盖 n 等变量 | 成立。原单拍边缘入口有标注，但没有结构证据，不能作为无条件默认事实 | 入口/路由 profile 必须显式选择；加入联合结构敏感性、瓶颈转移与排序稳健性检查，未知项保留状态 |
| P0：五阶段可能扭曲动态竞争 | 成立。单一比例将短活动摊长，theta 还会增加虚构的共享占用 | 保留首尾框架；L/C/S 独立推进与释放，以有限缓冲约束稳态；只在验证需要时有上限地细分 |
| P0：Excel 不能证明预测准确、无法识别全部超参 | 成立；原稿已说明证据不足，但完成门槛仍过于简略 | 分开源表核对、参考重建、新库执行、粒度验证及独立硬件验证；禁止将 TX8 共同效率迁成目标 INT4 校准 |
| P1：分块/复用/映射不够确定 | 成立。wave 口径和驻留循环仍有歧义 | 固定候选、循环顺序、Core 归属、计数、容量与平局规则；重复计算也计费 |
| P1：五仓接口未完全闭环 | 部分成立。原稿已有归账和快照设计，尚缺旧键逐项映射与 ABI 生命周期 | 冻结兼容指标语义、状态结构、一次归账、已知 world 回放和逐仓验收清单 |

固定数量资源域、解析分块、不可变计划、公共 C++ 库与薄适配层继续保留。入口物理证据、目标校准和新库运行结果仍是待补证据，不能由文档修订替代。

## 1. 目标、依据与取舍

### 1.1 本文要实现什么

建立一个能嵌入五个 Astra-sim 仓库的单 die 性能后端。输入《架构建模新.md》的全部架构自变量，以及该 die 实际承担的本地工作负载；输出独占估计延迟、运行时完成事件、资源工作量、容量需求和上层已有指标需要的数据。

粒度取“保留这些自变量影响所需的最粗粒度”：

- 不模拟单 PE、向量 lane、SRAM bank、HBM channel、flit 或时钟周期。
- 不逐 Core 建执行器，不逐 tile 建事件，不逐有向边推进包。
- Core 资源聚合，但保留 Nx、Ny、每核 SRAM 和整数安装数。
- 分块只做解析计数与小规模候选选择；重复块用数量表示。
- HBM 均匀条带假设下聚合为两排服务组，不逐栈建事件。
- Core 内互连用一个聚合服务域；Core 间 Mesh 用方向截面和总 hop 服务域。
- 每个外部可观察的计算区域生成少量启动、稳态、收尾阶段。
- 五仓库的原事件时钟继续推进时间，网络、KV、恢复和策略仍由原模块管理。

这是容量约束的解析成本模型与宏阶段事件模型的组合。它使用工作量/服务率计算时间，但会根据 SRAM、分块、复用、整数核数、Mesh 形状和共享资源改变工作量及完成时间。

### 1.2 依据优先级

1. 硬件自变量、Core 面积与安装数、Die/HBM/Wafer 几何及 bump/D2D 预算，以《架构建模新.md》为准。
2. 本文定义源文没有规定的性能执行假设、工作量规则、聚合服务域和接入接口。
3. 五仓库现有代码决定请求、KV 所有权、网络端点、图依赖和调度业务的含义。
4. TX8 工作簿提供参考数据和条件性校准约束，不覆盖目标硬件定义。
5. 两份旧“动态建模版”文档只作历史背景；与以上规则冲突的部分不沿用。

《架构建模新.md》没有给出完整微架构时序。本文新增的带宽效率、输入映射和流水近似必须明确标注为性能模型假设，不能反称源文已提供这些参数。

### 1.3 参数响应不等于所有参数都改变延迟

每一个源文自变量都必须被读取、校验并进入相应派生结果。但是：

- 增加边缘空白、修改 Wafer 可铺排区域，不应在 Core 网格不变时凭空改变计算时间。
- HBM 颗数增加后，若入口或内部互连已饱和，延迟可以不再下降。
- SRAM 增大但没有跨过分块或驻留阈值，某个算子的时间可以相同。
- 用户给定的同一硬件实例与任务必须产生相同时间；不生成或推荐面积/Core排列的硬件候选。
- 源文只给几何等价的两种 HBM 摆放。本文若根据入口位置产生时间差异，依据是显式的聚合入口假设，不是两种外廓大小不同。

验收检查的是“参数是否通过正确机制影响模型”，不要求给所有参数人为添加非零延迟系数。

### 1.4 建模复杂度：简单公式优先

用户新增要求（2026-10-14）：**对于与硬件设计参数、软件任务输入参数不直接相关的建模细节，能用简单数学公式直接计算的，就不进行深入建模。**

实现优先采用闭式计算、解析计数及固定数量的分段公式。几何派生、容量合计、重复工作量和可以直接求解的辅助环节不增加独立模拟对象或内部事件。内部状态仅为保留架构/任务自变量的可观察影响、精确容量与工作量守恒，以及实际外部依赖和共享竞争所需的机制而保留。

需要新增状态或复杂数值机制时，应明确指出其对应的输入参数或可观察行为、简单公式不能满足的具体证据，以及准确性收益与仿真开销。已有复杂研究不能成为继续扩展的理由。源架构变量、真实 shape/精度/TP/span、必要的就绪/容量/竞争约束继续有效；精度目标以§1.6的最新用户决定为准。

### 1.5 已确认的解析粗粒度主线（2026-10-15）

用户确认采用**解析工作量、固定数量的分段累计服务曲线、稳定区间直接积分**。本次落实§1/§8.2/§13的原目标，不另建第二套Python时间模型；本节是后续实现约束，不表示代码已实现或验收通过。

内部循环只用于解析计数、full/tail分类、映射与容量公式，重复次数只作为数值进入公式。生产不逐Core/tile/K/wave推进，不恢复归档Program作为接续主线；周期搜索、精确周期/相位返回证明、大整数周期求解和逐K回放均排除出本生产方案。即使状态量固定，逐K推进仍不符合运行成本要求。已归档源码/失败证据保留，小规模逐wave oracle仅在离线误差验证中使用，不扩展为生产依赖或新的精确周期研究项目。

逐primitive从最多5个服务段开始，必要时评估8/16段；保留最多32个解析工作类别，复合region保持原64段上限。每owner至多三个活动，所有累计曲线的折点、活动边界及描述符切换必须统一计入实际复杂度预算，不能在少量宏段下隐藏逐块展开或为每条曲线另设无约束段数。固定外部事件序列和活跃规模后，放大Core/tile/K/repeat不能导致内部状态/事件数线性增长；真实外部事件和输入span数量的成本单列。

服务曲线、单位及切向约束按§8.2.1生成；总量/容量/外部依赖严格正确，内部细碎交错采用显式聚合近似，不以复现每个离散就绪时刻作为默认目标。保留架构变量响应、owner Q/alpha、13域、PRIVATE及五仓业务/指标兼容。解析或水填充仲裁只在与原公平定义等价时启用，多资源不能简单均分带宽后取最小值。

速度和精度从第一版共同检查，规范化缓存、轻量查询与模型修改同步实施；§13的速度预算及原busy定义保留。精度按§1.6：目标10%，多次合理测试/校准仍难达到时可接受20%，不得静默将20%写成10%通过。另报绝对误差、参考共享时长及适用范围；超过20%、缺少有效对照或速度未通过时，分别如实记录failed/not_available/tradeoff_unresolved，不事后修busy或逐tile fallback。

### 1.6 最新用户决定：给定硬件预测延迟与适度精度（2026-10-15）

本节优先于本文及历史交接中有关硬件搜索、3%/5%门槛的旧要求。仿真器是用户架构研究的工具：**用户输入一套硬件配置和实际本地任务，公共库预测其延迟；不替用户搜索或决定Die面积、Core行列数或Wafer的Die排列。** 当前实现范围到单Die面积/外形及Nx×Ny Core网格即可，Wafer级铺排不作必填项或前置条件。源架构文中的研究搜索表/结论保留作用户资料，不构成仿真器功能。

移除生产硬件候选枚举、Core-first/Wafer-first搜索、Pareto/最优面积推荐及相应CLI/API。给定Core类型、每核SRAM、NoC位宽、Die宽高、Core行列、HBM及性能profile后，只做输入一致性、几何/容量合法性和资源服务率推导，不自动扩大Die、改变网格或选择另一套硬件。给定硬件下的算子tile、驻留/融合等软件映射选择仍保留。

**延迟预测与已有模型近似对照的目标相对误差为10%；多次有记录的合理测试与超参调整仍难达到时，20%以内也可接受。** 这一决定替代原独占3%、并发完成/busy5%的硬门槛。沿用独占、逐请求并发完成和原busy对照口径并逐项报告，busy仍为正服务时间的并集/共享积分；不以平均利用率替换。参考非零时报告相对及绝对误差；参考为零时单报绝对值与覆盖，不能以任意分母制造通过。

误差≤10%记目标达成；10%—20%只有在保留多次测试/调整记录、说明10%未达及适用范围时记为容许范围内接受，不称10%通过；>20%记failed。旧3%/5%的历史raw/报告不重写，按新阈值复评须另存指标版本和原始数值。20%不是跳过测试的默认目标，也不允许超出后自行继续放宽。

校准以满足docx/本文的资源、工作量、容量、依赖及参数响应要求为前提，基于xlsx“实测”正确分解后的算子时间，用少量有含义且可识别的超参使量级和趋势大致吻合；不追求噪声数据逐点精确，不为贴时间增加复杂机制或每算子独立乘数。TX8条件参考、目标工程先验、模型间oracle和独立目标测量继续分别标注；无独立目标数据不宣称目标芯片已实测校准。

工作量/容量/归账、必要依赖和公平策略的正确性不适用10%/20%近似容差。§13的hot20μs、冷编译2ms、J≤32更新50μs及五仓CPU增加≤10%预算未改变。继续固定分段解析模型，不恢复逐K/周期生产方案。实施顺序为先提交并同步本设计决定，再改硬件输入/移除搜索，最后按新指标复评、校准与五仓兼容验证。

## 2. 粒度与公共架构

### 2.1 最小模拟对象

| 对象 | 表示方式 | 为什么保留 |
|---|---|---|
| 一个 die | 一个 Hardware 对象及一个资源仲裁域 | 与 Astra-sim rank 业务映射相容 |
| Core 配置 | 64 行整数资源表中的一行 | 保留大小核、SRAM、B 对安装数的影响 |
| Core 网格 | Nx、Ny 两个整数 | 保留同核数不同矩形的方向与截面差异 |
| SRAM | 每核容量；聚合读、写服务域 | 容量决定复用，带宽决定操作数供应 |
| Core 内互连 | 按活跃核比例折算的聚合服务域 | 同一个 B 必须影响 SRAM↔PE/Vec |
| HBM | 两排，每排 n 栈；排内均匀条带 | n、容量、共享带宽可表达，无须逐栈事件 |
| HBM 接入 | 两个聚合入口域；位置由显式 profile 决定 | 表达条件性的接入上限，不推断未知 PHY |
| Core 间 Mesh | X+/X−/Y+/Y− 中央截面与总 hop 域 | 保留 B、Nx、Ny 的基本影响 |
| 一个计算区域 | 不可变 MacroPlan | 保留外部依赖、数据就绪和融合边界 |
| 重复 tile | 数量及最多八类尾块统计 | 不产生与 tile 数成比例的事件 |
| 运行阶段 | 最多 L/C/S 三个独立活动、有限缓冲状态 | 完成的活动立即释放资源，不摊长访存 |

两排 HBM 不是两条复制出来的免费总线。具体是否共享入口、从哪一侧进入，由显式 timing_topology_profile 决定；源文只规定等外廓摆放，未规定实际端口。所有 profile 仍保留真实 HBM 字节、片上访问和相应互连约束。

### 2.2 计算区域的边界

优先复用五仓库实际 COMP 节点携带的类型化算子或 region。只有不存在外部可观察的网络、KV readiness、恢复、watch、credit、首 token 或业务回调边界时，才可合并相邻本地算子。

可按 Norm、QKV、RoPE、Attention、O projection、Residual、FFN、Logits 编译。完整 Transformer block 的九组仅是报告方式，不强迫所有仓库改成九个节点。

一个跨多个KV block的attention区域，只有所有被消费block就绪才整体提交，否则沿已有readiness边界切区。这由外部依赖产生，不是内部tile事件。单内部region最多四个固定primitive kernel并串行推进；外COMP若携带更长算子图，由PlanBundle组成若干这种region，保留原一个GPU槽/外节点完成join，不改变业务节点边界。额外成本随显式数学图大小，不能随Core/tile重复数增长。

### 2.3 静态编译与动态执行

静态部分生成资源、选择分块、计算字节与工作量、形成宏阶段计划。动态部分只在请求提交、阶段完成、资源成员变化、取消和外部数据就绪时更新剩余进度。

同一个不可变计划用于：

- 独占预测：在空闲资源域执行。
- 候选投影：在已知快照的私有副本执行。
- 真实仿真：在原 Sys 时钟和共享资源域执行。

禁止把一套公式写在 Python、另一套写在 C++。Python 通过 C ABI 获取计划或估值。

## 3. 架构输入：全部源文变量的覆盖

### 3.1 三类配置

配置以 architecture（用户给定的单Die实例）、performance_profile（执行假设/待校准超参）为主体；可选validation_constraints仅检查该实例，不生成候选。旧search_profile仅作已有配置的迁移兼容元数据，不再作为必填项、几何来源或搜索接口。

派生量不能作为第二份独立输入覆盖结果。例如 A_die、Core 数、HBM 总颗数必须由规范输入计算；若客户端附带镜像值，只允许一致性校验。

### 3.2 Core、Die、HBM 输入

| 字段 | 域/源文设定 | 作用 |
|---|---|---|
| core_type | small / large | 读取相应整数资源表 |
| sram_mib_per_core | 0.25、0.50、…、2.00 MiB | 面积、每核容量、分块和复用 |
| noc_width_bits | 64 / 128 / 256 / 512 | 同时影响 Core 内互连、Core 间 Mesh 及 PE/Vec 安装数 |
| core_columns_nx | 正整数 | Core 列数、X/Y 截面和几何 |
| core_rows_ny | 正整数 | Core 行数、X/Y 截面和几何 |
| die_width_mm、die_height_mm | 用户给定的正数宽高 | 面积由W×H派生；不自动选择或扩展外形 |
| die_geometry | explicit；旧标记仅作迁移兼容 | 使用给定W/H，不生成最低面积边界候选 |
| core_pitch_x_mm、core_pitch_y_mm | 源文示例为 0 | Core 间几何余量 |
| die_extra_width_mm、die_extra_height_mm | 源文示例每轴总余量 0.10 mm | 最小 Die 外形；不是两边各 0.10 |
| hbm_stacks_per_row | n；源文主搜索 1…14 | 两排总 2n 栈、容量和理论带宽 |
| hbm_placement_mode | same_side_two_rows / two_sides_one_row_each | 源外廓相同；只有指定接入profile时才产生位置时序差异 |
| system_region_width_mm、system_region_height_mm | 可选历史系统诊断，不是单Die必填输入 | 不决定Die实例、rank数或单Die延迟 |

用户必须给定一套明确的Die外形与Core网格，resolver只解析和校验这一套实例。只给面积不能确定宽高或Nx×Ny；缺少必要尺寸/网格就返回缺字段诊断，不反推“最优”配置或floor(sqrt(area))方阵。若同时附带面积、Core总数等镜像字段，只检查其与W×H、Nx×Ny一致。

### 3.3 实例校验与旧搜索配置迁移

面积/曝光场/长宽比/D2D预算等研究约束可由用户显式启用，用来接受或拒绝当前实例；不调整输入、不返回另一套硬件。Nx/Ny/HBM范围、geometry_search_mode、area_extension_rules、wafer范围、objective及Pareto选择不属于生产仿真接口。旧配置携带这些字段时，只按明确迁移规则兼容读取；不得触发候选枚举，也不得要求新配置提供它们。

Core64源表可用于解析用户指定Core类型/SRAM/B对应的资源数，这不是硬件方案搜索。无Wafer/system字段时正常解析单Die；已有上层提供的系统诊断可保留，但不反向约束或自动改写单Die输入。

### 3.4 版本化常量

以下常量按源文保存。可以在独立敏感性版本中修改，但修改后必须重新生成资源表或几何结果，不能保留原 64 行却更换底层面积常量。

| 项目 | 源文数值 |
|---|---|
| 工艺/频率口径 | 7 nm / 1 GHz |
| SRAM 宏 wordwidth | 1024 bit；这是面积建模口径，不证明端口时序 |
| SRAM 7 nm 尺寸 | a={0.40662,…,3.25296} mm；b=0.13925 mm |
| Vec lane | 0.1897×0.1265 mm；3.9554 FP32 等效 GFLOPS/lane |
| PE 单元 | 0.0000376 mm²；INT4 1 MAC/cycle=2 GOPS |
| PE/Vec 名义比例 | N_pe=round_half_up(19.777 N_vec) |
| NoC 面积 | 基准占 Core 15%；g(B)=0.20+0.80 B/256 |
| HBM SKU | Micron MT65B18G16120A00QH-92:A |
| 单栈 HBM | 36e9 byte；1024 bit×9.2e9 transfer/s；1177.6e9 B/s |
| HBM 外廓 | 10.98×10.98 mm；厚 0.78 mm 不进入二维铺排 |
| HBM/模块间隙 | 0.15 mm |
| bump | pitch 0.055 mm；40% 通信；2.5 Gb/s/bump |
| 模块间 NoC 占位 | 扩展后模块总占位的 10% |

模型输入可携带这些常量的完整镜像和 source_revision，用于追溯；标准 profile 对与源文不一致的值报错。扩展 profile 必须有单独名称和资源表摘要。

## 4. 硬件解析与派生公式

### 4.1 Core 资源表

构建或离线生成时读取源文 §1.3 的 64 行 N_vec、N_pe。运行时 O(1) 查表，不重做整数几何搜索。L0 按源文公式从 a、b 计算；不使用附录中四舍五入的 Die 尺寸做 floor。

~~~text
small: L0 = a
large: L0 = (a + sqrt(a*a - 0.6*a*b)) / 0.3
A_noc = 0.15 * L0*L0 * (0.20 + 0.80*noc_width_bits/256)
N_core = Nx * Ny
P_int4_per_core = 2 * N_pe * 1e9          integer-op/s
P_vec_per_core  = 3.9554 * N_vec * 1e9    FP32-equivalent-op/s
S_core = sram_mib_per_core * 2^20         byte
~~~

PE INT4 和 Vec FP32 峰值分别报告，禁止相加成一个 FLOP/s。

### 4.2 Die、HBM、bump

~~~text
W_min = Nx*L0 + (Nx-1)*pitch_x + extra_width
H_min = Ny*L0 + (Ny-1)*pitch_y + extra_height
require W_die >= W_min and H_die >= H_min
A_die = W_die * H_die

N_HBM = 2*n
HBM_capacity = 2*n*36e9
HBM_peak_Bps = 2*n*1.1776e12

HBM_row_width = n*10.98 + (n-1)*0.15
W_module = max(W_die, HBM_row_width) + 0.30
H_module = H_die + 22.56

B_bump_GBps = A_die/(0.055*0.055) * 0.40 * 2.5/8
B_D2D_bi_GBps = B_bump_GBps - 2*n*1177.6
B_D2D_oneway_GBps = B_D2D_bi_GBps/2
~~~

Core 数只随显式 Nx、Ny 改变，不因增加 Die 留白自动增长。HBM 超宽时设置 hbm_row_overhang=true，扩大模块并记录物理待验证；不按旧文档硬性过滤。

W_die、H_die、Nx、Ny均由用户给定。W_min/H_min只用于检查Core网格能否装入该Die；bump与HBM扣除预算按给定面积解析计算。不得按面积下界自动grow_width/grow_height/isotropic，不要求输入恰好落在旧搜索产生的三个边界点上。可选研究约束拒绝不合法实例时，返回原因并保持输入，不推荐替代方案。

### 4.3 可选历史系统诊断（不属于当前单Die前置条件）

~~~text
W_effective = W_module / sqrt(0.90)
H_effective = H_module / sqrt(0.90)
Mx = floor(system_region_width / W_effective)
My = floor(system_region_height / H_effective)
N_module = Mx*My

system_core_count = N_module*N_core
system_hbm_capacity = N_module*HBM_capacity
system_peak_int4 = N_module*N_core*P_int4_per_core
system_peak_vec_fp32 = N_module*N_core*P_vec_per_core
~~~

面积利用率按源文分开：Core/Die、Core/系统区域、含 HBM 与分摊互连的模块占位率。10% 模块间 NoC 只影响空间占位，不成为内部 NoC 减速系数。

~~~text
eta_core_die = Nx*Ny*L0*L0/A_die
eta_core_region = N_module*Nx*Ny*L0*L0/(region_width*region_height)
eta_module_region = N_module*W_module*H_module/(0.90*region_width*region_height)
~~~

这些字段只在用户显式提供系统区域并请求诊断时派生，不合成一个“面积效率”，也不成为单Die延迟API必填条件。

源文的系统区域不是圆形制造晶圆。N_module 不自动覆盖五仓库 rank_count；创建系统拓扑属于上层显式步骤。

## 5. 工作负载输入与精度执行

### 5.1 DieTaskSpec 必需信息

| 字段 | 含义 |
|---|---|
| op_kind / region_ops | GEMM、Norm、RoPE、Attention、FFN、Residual、Logits 或已支持的本地融合区域 |
| phase | prefill / decode / kv_recompute；不是神经网络训练阶段 |
| tensor_shapes | 真实局部 M/N/K、头数、head_dim、FFN 等维度 |
| spans | 每段 batch、q_len、kv_len、绝对 query/KV 位置；不以最大长度代替每段 |
| tp_slice | tensor_parallel、rank、各局部头/FFN/vocab 区间 |
| layer_instances / weight_passes | 实际执行的层和次数，不是已加载权重的字节量 |
| storage_dtype | 权重、激活、KV、score、累加器各自位宽 |
| execution_profile | 显式选择下述执行路径 |
| fusion_policy / residency | 融合范围、可存活缓冲、输入是否已有本地副本 |
| readiness_scope | 当前区域可以消费的已就绪 token/block 范围 |
| graph_identity | request/session/pass/layer/region/node 身份 |

不能仅把旧 num_ops、tensor_size 传下来：这两个数无法决定 SRAM 分块、GQA 复用、causal 跳过和矩形映射。

### 5.2 默认目标路径：INT4 矩阵、FP32 向量

目标 profile 名称：int4_matrix_vec_fp32_assumed_v1。它明确采用：

- 矩阵乘法的乘数为 INT4，累加为 INT32，进入原架构的 INT4 PE。
- Norm、RoPE、softmax、激活、缩放/转换等在 FP32 Vec 的等效操作模型上运行。
- 外层权重、激活和 KV 的存储精度按 DieTaskSpec 保留；五仓库已有 BF16 存储默认不自动改成 INT4。
- 存储为 BF16/FP32 的操作数，在装入本地矩阵 tile 时计入量化/打包工作；PE 输出进入向量路径时计入转换/反量化工作。
- 已是 INT4 的权重或激活不重复收费量化；同一 tile 的转换结果在声明的生命周期内可复用。
- 量化尺度、格式元数据和转换缓冲均计入容量和真实访问量。

这是显式的量化执行假设，不声称 INT4 PE 原生执行 BF16，也不证明量化后的模型精度。性能仿真不生成数值 token，但结果必须标明这一执行路径。

对要求严格 BF16 矩阵运算的实验，允许选择独立 fp32_vec_matrix_assumed_v1：把 BF16 输入转换为 FP32，矩阵工作全部进入 Vec，PE 不承担这些 GEMM。若用户另提供 BF16 PE 能力，须建立独立扩展硬件 profile，不能修改原 INT4 资源定义。

### 5.3 TX8 参考路径

tx8_reference_v1 使用工作簿自己的矩阵/向量峰值、DRAM 和 SRAM，不伪造其 Core 网格，也不覆盖目标硬件。它用于工作量与条件性校准复核。共享数学图；硬件 profile、流量路径和执行精度分别标明。

### 5.4 三个字节口径

1. logical_tensor_bytes：原图张量的存储量，供兼容指标和容量账本使用。
2. physical_transfer_bytes：按分块、重复加载、写回和融合实际计费的物理 payload。
3. timing_service_units：效率及局部供数折算后的时间服务量；尾波修正计算服务预算，不放大物理byte。

服务量不能写回 tensor_size、KV 大小、网络消息大小或实际服务字节。跨精度统计必须保留单位。

## 6. 解析分块、复用和工作量

### 6.1 确定性的候选与选择

候选集合是编译器策略常量，不是架构自变量或校准自由度。GEMM 的 tm∈{1,16,64,256}，tn、tk∈{64,256,1024,4096}；对各维使用 min(候选值,实际长度)，升序去重，最多 64 个基本候选与两种数据流。Attention 的 tq、tk∈{16,64,256,1024}，最多 16 个基本组合。FFN 见6.5。

每个候选先生成缓冲生命周期、解析访问和活动计划，再按空闲域的同源模型估时。选择键固定为：

~~~text
(ceil(estimated_latency_ns), hbm_payload_bytes, hbm_workspace_peak_bytes,
 sram_peak_bytes_per_core, template_id, tm, tn_or_tf, tk, th)
~~~

逐字段升序选最小者；整数纳秒相同才比较后续字段，不用不可重现的 epsilon 平局，也不查看 Excel time 列。禁用某模板、候选档位、精度或融合规则均进入 template_revision 和计划键。存在可行候选却未完成粒度认证时，可输出假设估计，必须同时标 granularity_validation=not_run/failed。

容量条件为 peak_live_buffer_bytes_per_core≤rho_sram*S_core。别名只计一次；双输入槽、FP32 累加、量化缓存、score/softmax 状态、等待写回的输出槽及元数据均计入。不能把 TX8 的48 MB分给每个目标 Core，不能由带宽参数减少容量。目标 INT4 路径的 scale 使用固定 per-tile 对称标量模板，每个转换缓冲额外4 byte；更换量化方法必须换模板版本。

字节计数使用整数。BF16/FP32 可用元素数×2/4；INT4 块用 ceil(elements/2)，在完整/尾块类别内乘重复数，不能把每次奇数尾块的半字节忽略。总数/乘积用带溢出检查的至少128位中间整数；超出64位输出范围返回 overflow。

### 6.2 GEMM 的固定循环与 Core 归属

对 C[M,N]=A[M,K]W[K,N]：math_matrix_ops=2MNK，Tm=ceil(M/tm)、Tn=ceil(N/tn)、Tk=ceil(K/tk)。先区分两个数量：

- output_round_count：若干 Core 同时处理一组输出块的轮数。
- consumption_wave_count：能由当前有限缓冲装入并消费的 K 子块轮数。OS 为 output_round_count*Tk；不把一轮输出的全部 K 输入当作首波一次装入。

**OS：output_stationary_sram_v2。** 输出块编号 q=m_tile*Tn+n_tile。round=floor(q/N_core)，core=q mod N_core，x=core mod Nx，y=floor(core/Nx)。轮次递增；轮内每个核独立按 k_tile 递增消费，同核 FP32 输出累加块跨 K 保存在 SRAM；最后一个 K 才产生该输出块的外存写回。没有跨核广播，也没有跨输出块的 A/W 缓存。

~~~text
output_round_count = ceil(Tm*Tn/N_core)
active_cores_in_round = min(N_core, remaining_output_tiles)
D_A_read = b_A*M*K*Tn
D_W_read = b_W*K*N*Tm
D_C_write = b_C*M*N
~~~

上述简式适用于整数 byte/element；INT4 用同一循环的块类别求精确 packed byte。末轮空核不消费数学工作或物理字节。计算预算按轮内最慢核的实际块计算，轮间相加；统计始终保留真实2MNK，不能把空槽或补零算成真实 FLOP。

**WR：weight_resident_sram_v2。** n_tile 按递增顺序组成每轮至多 N_core 个列块，core=n_tile mod N_core；每核一次装入完整 K×tn 权重片，然后按 m_tile、k_tile 递增完成该列所有 M 输出块。只允许这个权重驻留，不顺便赠送输入跨列/跨核复用。完整驻留权重、两个A输入槽、累加与输出等同时通过容量检查才合法。列组切换先完成旧计算/写回、释放旧权重，再装下一组完整权重，不能未计双份权重容量就跨列组预取；此串行边界以有限阶段描述和重复数计费。

~~~text
output_round_count = ceil(Tn/N_core)*Tm
consumption_wave_count = output_round_count*Tk
D_W_read = b_W*K*N
D_A_read = b_A*M*K*Tn
D_C_write = b_C*M*N
~~~

WR 可能节省权重读取而减少 M 方向并行，必须重新估时，不能只修改字节。装入过的权重量化一次；OS 的重复装入按实际转换次数收费。返回 output_round_count、consumption_wave_count 与 dataflow_id；旧 wave_count 若保留只能是后者的别名。

M/N/K 形成至多八种尺寸类别；K 首/中/尾、是否最终输出等状态合成至多32个类别。解析计数器用整除、区间交集与 floor_sum 统计上述映射，不创建 Core/tile 数组。内部阶段保留循环的先后关系；类别计数不能把先发生的短访存和后发生的长计算交换顺序。地址未知时，源入口列分布和目标 Core 分布的近似另由路由 profile 声明，不能声称掌握逐张量地址。

### 6.3 统一访问动作账本

AccessCounter 从选定循环和缓冲生命周期生成以下固定动作；所有模板共用它，不能各自补一个带宽惩罚因子。

| 动作 | 计数与资源路径 |
|---|---|
| 外存装入 | HBM读→入口/指定Mesh→SRAM写；实际存储 byte 各域只计一次 |
| 外存写回 | SRAM读→指定Mesh/入口→HBM写；精度转换另列 |
| SRAM→PE/Vec | 当前消费块的读 byte，同时占 SRAM_READ、LOCAL_BUS；在执行单元内允许本块寄存器复用 |
| PE/Vec→SRAM | 结果或状态的写 byte，同时占 SRAM_WRITE、LOCAL_BUS |
| FP32累加 | 每K的INT32 partial转换后更新FP32累加：旧累加共4MN(Tk−1)读、4MN*Tk写，partial/最终转换读写另列 |
| 量化/打包 | 读存储输入、写INT4缓存、读写4-byte scale，Vec等效工作另列；已为INT4的不重复量化 |
| 本核驻留 | 不产生HBM重读，但每次真实消费仍有片上读/总线工作 |
| 融合中间状态 | 按其声明生命周期读写SRAM，必要时按明确模板物化HBM |

默认FP32累加在SRAM。目标PE每K产生INT32 partial，Vec转换成本为kappa_acc_convert*MN*Tk，后续相加另计MN*(Tk−1) eq-op；INT32 partial与旧FP32累加同时存活时占两份缓冲，不能非法别名。最终存储dtype转换另列。严格FP32-Vec与TX8参考按自身模板，不重复加INT32转换。第一版不使用无容量证据的完整累加块永久寄存器变体。所有内部流量均从动作账本推导。计算活动中的片上供应仍是粗活动模型，是否需拆转换/PE/Vec的短链由8.6的误差验证决定。

### 6.4 向量算子

按连续 row/tile 分片，编号递增、core=tile_id mod N_core；参数 gamma/beta、cos/sin 按每个消费 Core 首次读取收费。仅当完整参数片与其它缓冲同时装入 SRAM 时允许区域内复用，跨区域默认无驻留；task 明示的驻留证明可覆盖此默认。

保存 math_vec_ops、vec_equivalent_ops、special_function_counts、conversion_elements。参考 eq-op 模板为 Norm≈6/element、RoPE≈3/参与元素、scale/mask/softmax≈12/实际处理score、FFN激活部分≈16/FFN元素；这些是固定等效计数，不是 exp/rsqrt 的真实周期。没有对应微基准不能把源文 FP32 FMA 峰值当成已测 SFU 吞吐。

目标模板另收费 kappa_quant*实际转换操作数及 kappa_acc_convert*实际转换累加结果。TX8 参考模板不加入目标 INT4 量化成本。修改模板时不能重复计入已经包含的转换。

### 6.5 FFN：明确融合与物化的代价

SwiGLU 图的源数学矩阵量为6MHF。第一版比较两种确定方案：

1. materialized_swiglu_v2：up/gate分别调用6.2的GEMM，Vec执行SiLU/乘法，down调用GEMM；中间张量的写/读及临时容量明确计入，不能假定它们跨三个kernel免费驻留。
2. fused_output_shard_v2：按输出(m_tile,h_output_tile)编号和OS规则映射；循环依次为输出轮→f_tile→H的k_tile。同一H输入片装入/量化一次供该核up/gate复用，片上操作数读仍各收费一次；该核计算本片up/gate，激活后乘本片down权重，把结果累加到tm×th输出片，最后F块才写回。

融合方案没有免费跨核广播：同一个up/gate片若因不同H输出分片被多次计算，真实 executed_matrix_ops 必须加上重算。令 Th=ceil(H/th)、Tf=ceil(F/tf)：

~~~text
source_matrix_ops = 6*M*H*F
executed_matrix_ops = 4*M*H*F*Th + 2*M*H*F
executed_activation_eqops = 16*M*F*Th
D_input_read = b_X*M*H*Tf*Th
D_up_gate_weight_read = 2*b_W*H*F*Tm*Th
D_down_weight_read = b_W*F*H*Tm
D_output_write = b_Y*M*H
~~~

融合消费wave为当前输出cohort的一个H/K子块；L含该子块的输入/up/gate权重，在每F片首K另加载down权重；C为up/gate累加，在末K追加激活与down；S只在末F的末K写最终输出。consumption_wave_count=ceil(Tm*Th/N_core)*Tf*ceil(H/tk)，所有尾块/不活跃核依实际尺寸计费，不能把整个H权重当首wave无容量一次装入。

输入在不同F片之间默认重读；若要持有完整tm×H输入，须作为另一个固定模板加入容量和驻留账本，不能静默改变上述计数。相同up/gate重算也是合法工程取舍，候选选择通常可因此转向物化方案；不能报告融合永远更快。

融合候选：tm四档同GEMM，tf∈{16,64,256,1024}，th∈{64,256}，裁剪去重后最多32组；每组从四档tk按降序找首个可行值，最多128次容量试探。存活峰值包括两份下一输入、up/gate FP32状态、量化/激活片、down累加、待写输出、scale和元数据。选择规则统一按6.1。物化方案的up/gate各写其存储dtype的M×F，Vec各读一次并写M×F激活，down的激活重读依所选GEMM循环计费；中间dtype从task激活角色读取，默认BF16，暂存峰值由四kernel的生命周期确定。它调用固定四个子编译器，不枚举候选笛卡尔积。融合不可行但物化可行时选物化；仅所有方案不可行才unsupported_workspace。普通GELU等按自己的数学图模板，不能套用SwiGLU三矩阵公式。

### 6.6 Attention：头映射、causal 与有限状态

只接受 q_heads%kv_heads=0 的标准GQA；group=q_heads/kv_heads，kv_head=floor(q_head/group)。非标准头映射必须由task显式给出，否则 unsupported_head_mapping。每个span独立保留batch、Sq/Sk、query/key绝对起点、合法KV区间与head_dim；interval先验证无重叠，不能把变长请求压成最大长度。

默认 fused_q_head_shard_v2 的输出任务为(batch,q_head,q_tile)。按该元组字典序编号、core=id mod N_core；同核按k_tile递增执行QK→scale/mask/softmax在线状态更新→PV。Q片与 FP32 running max/sum/output跨KV循环保留；KV不跨Q任务驻留。因GQA不同Q头或q_tile重复读同一KV时照常收费；不存在免费跨核复用。

融合消费wave为当前输出cohort的一个KV块：首L还包括Q，后续L仅相应KV，C为QK→softmax状态→PV，S仅各输出任务最后有效KV后写O。整块跳过/各任务causal结束的时刻由区间规则解析计入，不能把一个任务的全部KV压成首装；q任务顺序与Core归属按上式固定。

causal规则固定为 key_abs≤query_abs。完全在causal区域之外的矩形块跳过；相交块用其真实矩形进行QK/PV，把无效score显式mask，不能既扣无效FLOP又不声明支持三角执行。保存 executed_score_slots 与 valid_score_pairs。对稠密span，有效对数为 sum_q clamp(query_abs(q)−key_abs_start+1,0,Sk)，用等差/分段求和；块级执行量用 floor_sum 和区间交集求得，不做O(Sq*Sk)循环。

| 量 | 默认融合模板规则 |
|---|---|
| QK/PV矩阵量 | 每项2*head_dim*executed_score_slots，尾块真实尺寸 |
| Q读/O写 | 每个输出任务一次，按实际存储精度 |
| K/V读 | 每个输出任务每个未跳过KV块各一次；GQA重复读按上面的头映射计入 |
| 外存score | 0；score片与running状态留SRAM |
| 在线状态工作 | 按score slots的softmax模板，加每个后续KV块对已有output/max/sum的重缩放；合并成本单列，不遗漏反复更新 |
| 缓冲 | Q、当前/下一KV、score片、FP32 output/max/sum、转换与scale，逐生命周期检查 |

在线更新的默认成本为每个后续KV块、每个query行 (2*head_dim+4) eq-op（output缩放与统计合并），这是固定假设，必须进入模板版本。full_score_materialization_v2是另一固定四子kernel方案（QK、mask/softmax、PV及必要的转换按固定图闭合）：每span/head显式处理完整Sq×Sk矩形，mask不扣dense QK/PV工作；QK写4SqSk score、Vec读score并写4SqSk probability、PV读probability依GEMM循环收费，不能只记一次最小张量读取。GQA按上述头映射，score与probability生命周期/别名以softmax模板固定。TX8参考另用自己的完整矩阵口径。两个方案均按6.1选择；task强制指定时只枚举指定方案。默认候选不提供跨Q任务的KV驻留，未来加入时需明确分配及容量，不能用命中率超参减少byte。

### 6.7 本地多层与复现

只有shape、dtype、存储/驻留规则、路径相同且无外部可观察边界的层才能带repeat_count聚合。每层启动、权重访问、业务身份仍按发生次数记账。跨层默认中间张量写回/重读；原KV/权重容量由上层持有。

compile必须返回候选计数、选中模板与尺寸、每类重复数、源/执行操作数、动作账本总数和峰值生命周期。相同hardware/task/template版本在C++、C ABI和五仓应生成相同规范化plan hash；测试覆盖full/tail、INT4奇数、GQA、causal、decode、SRAM阈值与WR减并行，不能只比较最终时间。

## 7. 最粗资源模型：聚合服务域

### 7.1 资源数固定，不随 Core 或 tile 数增长

默认全局服务域最多 13 个：

1. COMPUTE_GATE：串行计算预算，单位 second-of-service；容量 1 second/second。
2. SRAM_READ、SRAM_WRITE：聚合 SRAM 读写。
3. LOCAL_BUS：Core 内 SRAM↔PE/Vec 互连。
4. HBM_ROW0、HBM_ROW1：两排 HBM 等效服务量。
5. INGRESS_0、INGRESS_1：聚合payload入口；edge情景显示为north/south。
6. CUT_X_PLUS、CUT_X_MINUS、CUT_Y_PLUS、CUT_Y_MINUS：方向截面。
7. MESH_HOPS：所有有向边总 hop-byte 服务域。

不存在的方向域直接省去。HBM 排内假定均匀地址条带，因此每排容量/峰值为 n 倍单栈；本版本不模拟单栈偏斜、bank 冲突和逐端口仲裁。

PRIVATE是每活动独立的容量1虚拟行，只约束该活动少量活跃核的速率；不把所有请求的PRIVATE工作相加到全局总线。它参与局部限速与可选离线诊断，不增加全局物理域数量。

B的源文必需响应是NoC面积与N_pe/N_vec变化；本核总线和Mesh服务率按明确的单拍链路假设再重算。入口是否按B变化取决于接入profile，不能把源文面积缩放解释为已知PHY带宽。这几项作用分别报告。

### 7.2 计算预算与活跃核数

一个 wave 的串行计算预算：

~~~text
t_pe = matrix_int4_ops / (active_cores*P_int4_per_core*eta_pe)
t_vec = vec_equivalent_ops / (active_cores*P_vec_per_core*eta_vec)
compute_gate_work_seconds = t_pe + t_vec
~~~

严格 FP32 Vec 矩阵路径把矩阵 FLOP 工作也加到 Vec 预算，t_pe=0。尾波、负载不均和有限活跃核数由编译器解析修正计算预算；不再让调用方除 TP 度或核数。

COMPUTE_GATE 是本版本保守的 die 级计算占用近似，保存一个 wave 内 PE/Vec 依赖。第一版沿用原 Sys 的 GPU 计算槽限制；不会因各计划有私有 PE 计时器而给并发计划复制一份全 die 算力。对 PE/Vec 独立跨 kernel 并发的更细研究应建立扩展版本。

### 7.3 SRAM 与内部互连

源文的 1024-bit wordwidth 只用于面积。性能 profile 额外明确假设每核每周期最多一读一写：

~~~text
SRAM_read_raw_per_core = 1024/8 * 1e9
SRAM_write_raw_per_core = 1024/8 * 1e9
local_bus_raw_per_core = noc_width_bits/8 * 1e9
~~~

乘相应效率后形成聚合容量。LOCAL_BUS 默认读写共用一个额度，SRAM_READ/WRITE 分开；改为共享 SRAM 端口属于独立性能 profile。

少于 N_core 个核参与时，还须设置该阶段的私有速率上限：其局部字节速率不得超过 active_cores 倍单核服务率。全局域只消费真实字节，不用“除活跃核比例后放大的字节”抢占其它任务额度。私有上限和全局约束一起限制阶段进度，无须建逐核资源。

即使数据全部命中本核 SRAM，LOCAL_BUS 与 SRAM 访问仍然有成本。这是保留 B 响应所需的最小约束。

### 7.4 HBM、入口及未知结构

每排容量为 n*36e9 byte、理论率 n*1.1776e12 B/s。HBM访问类的真实D以U=D/eta_hbm_class消耗排服务；入口/Mesh仍消耗D。两排默认均匀条带，或使用task/endpoint的row比例；结果带 allocation_evidence。排内不建逐栈事件。

**撤销“每边缘Core一个B-bit入口”作为隐含默认。** 权威源文没有实际HBM→NoC端口数、位置、并行通路或PHY持续带宽；bump公式也不能推得这些值。timing_topology_profile是必须显式选择的新增结构假设，不是漏补的源文自变量，也不能用Excel效率拟合代替结构证据。

| profile | 聚合入口与路由 | 使用含义 |
|---|---|---|
| distributed_ideal_v2 | 不增加额外入口瓶颈，两个入口域不限速；HBM源端均匀分布网格，Core目的按确定映射解析，仍经过Mesh与本核供数 | 假设分布式接入的敏感性端点，不是整个芯片延迟的严格上下界；没有证据时可显式用于探索 |
| edge_single_word_v2 | 每侧C=Nx*(B/8)*1e9*eta_inject；同侧两排共享north，两侧各排north/south | 保留原假设作为受限接入情景，physical_topology_validation=unverified |
| explicit_ingress_v2 | 每个聚合入口给持续raw B/s、row到入口映射、共享组与位置分布，再乘效率 | 有设计图/端口规格/微基准时采用；字段齐全且具有来源才能升级结构证据 |

无profile返回 missing_topology_profile；完整显式假设可返回有限延迟，但不能自动标为物理已验证。无限入口用enabled=false表示，不用inf参与水填充/JSON。explicit profile仍限两个聚合入口与固定方向域；若真实接入需更多独立组，须先证明可合并，否则另立扩展profile。

同侧/两侧摆放的外廓严格同源文。distributed情景没有源文未提供的位置时序差异，两种摆放时间可完全相同；edge情景的差异来自row→north/south假设，不能说是源架构保证。HBM容量、峰值、bump扣除、模块和系统数量在所有profile中仍按n和几何精确重算。

### 7.5 Mesh 路由与结构敏感性

采用规则矩形Mesh、每条有向链路每周期一个B-bit拍、双向独立的性能假设。L=eta_mesh*(B/8)*1e9 B/s；x中央截面容量Ny*L（Nx>1），y截面Nx*L（Ny>1），总有向边E=2*((Nx−1)*Ny+(Ny−1)*Nx)，hop域容量E*L。这些是性能profile，不能从源文统一面积预算反推为物理签核。

截面byte与hop-byte是同一传输的多重约束，不相加成总延迟。uniform_grid_v2的均匀独立源/目标轴长度d：mean_abs_delta=(d*d−1)/(3*d)，每一方向中央穿越概率floor(d/2)*ceil(d/2)/(d*d)。edge_uniform_v2的入口列与目标列均匀，x同前；north/south到均匀目标行的平均距离均为(Ny−1)/2，north的Y+穿越概率=(Ny−floor(Ny/2))/Ny，south的Y−为floor(Ny/2)/Ny。写回反转方向。不存在的方向不产生非零工作。

上面均匀源/目的公式作为独立校验与uniform_proxy敏感性情景。正常模板采用routing_profile=mapped_uniform_sources_v2：HBM源位置仍是explicit/edge/distributed的条件假设，目的分布必须由6.2/6.5/6.6实际Core归属和payload权重解析得出，少量活跃核不能自动当全网格均匀。MappingCounter求每轴的加权E[u]、E[u²]和中央切面左右份额，不创建Core数组。

对长度d的轴、源位置均匀、目的坐标u∈[0,d−1]：

~~~text
mean_abs_delta = (2*E[u²] - 2*(d-1)*E[u] + d*(d-1))/(2*d)
p_plus  = floor(d/2)/d * P(destination >= floor(d/2))
p_minus = ceil(d/2)/d  * P(destination <  floor(d/2))
~~~

edge的north平均dy=E[y]，south为Ny−1−E[y]；截面用对应目的另一侧份额，写回反转方向。例如仅一个任务固定在core0时目的不是均匀网格；distributed情景的north/south物理标签也不强加给它。OS/WR的满轮、前缀余核与full/tail权重用区间、等差平方和/floor_sum计算；explicit源分布也按矩形/行列余段解析加权。输出routing_evidence=modeled_mapping/assumed_source，并保留源假设。未支持的任意散点分布返回unsupported_mapping，不能偷偷换uniform_proxy。

第一版没有免费广播收益；加入广播须同时更新HBM重复读与树的hop/截面。改变源分布/路由代理是结构敏感性，不能用eta隐藏它。

离线敏感性至少包括：

1. 对同一可行Nx/Ny/B/SRAM配置扫描全部可行n、两种摆放；分别报告HBM峰值、入口容量、Mesh/local约束、每阶段瓶颈与端到端时间。
2. 联合比较distributed、edge与有证据的explicit情景。只把edge入口倍数改大不够：同一north入口后的Nx链路截面可能仍限制供数，必须一起审查入口位置/并行通路与route。
3. 固定结构后对未识别的eta_inject/eta_mesh等做事先登记的区间扫描；区间依据不足时标engineering_range，不能称置信区间。
4. 构造HBM-only端点、SRAM命中、Mesh-bound、本核总线-bound和compute-bound探针。HBM-only必须对n的排容量变化响应；其它探针只在受限域变化时要求时间改变。普通算子平台须能由明确瓶颈解释。
5. 配置排序按相同workload与profile分别求出。未扫结构时ranking_status=not_tested；情景/参数区间内反转时=assumption_sensitive，报告反转点；均保持同序才可标stable_in_tested_scenarios，仍不能当真实芯片证据。

多情景只在离线验证/用户给定配置的响应报告运行；正常五仓热路径使用一个冻结profile，不额外跑三套模型。不为了让n或B改变时间而删除已选情景的合法瓶颈。输出必须列各域容量/工作量、入口/路由身份和物理证据状态，使“无响应”可以审查。

### 7.6 空白几何与原文没有的线长延迟

pitch、外围预留、HBM 间隙、扩展 W/H 和 10% 模块间 NoC 只进入源文的几何链。原文没有距离到传播延迟的公式，第一版不添加每毫米延迟超参。

在相同 Nx/Ny、Core 资源和映射下，单纯增加 Die 留白应保持本地时间；bump/D2D 预算与系统模块数仍重新计算。

## 8. 宏阶段、独立活动与动态竞争

### 8.1 保留首尾框架，取消单一进度耦合

启动为每个真实区域实例的ready延迟，首访setup另列，不按内部tile收费。wave采用6.2定义的consumption单位。L_i为装入，C_i为计算及片上供数，S_i为已生成输出写回；非最后K/非最终输出的S_i可为0，不能捏造写回。

| 框架 | 可就绪的活动 |
|---|---|
| LOAD_FIRST | L1 |
| FIRST_OVERLAP | C1、L2 |
| INTERIOR_REPEAT | C_i、L_(i+1)、S_(i−1)，i=2…W−1，按解析重复表示 |
| LAST_OVERLAP | C_W、S_(W−1) |
| STORE_LAST | S_W |

W=1只有L1→C1→S1；W=2只有首装→(C1,L2)→(C2,S1)→S2。FIRST不提前写S1。宏框架中的每个活动有**独立剩余进度、服务请求和完成状态**；活动完成立即撤销其资源需求，不能等其它活动完成再释放。框架只保留依赖/有限缓冲边界，不再强迫HBM、计算及写回用同一个phi推进。

反例：已就绪重叠阶段C=100s、L=50B，HBM=1B/s；另一40B端点在t=60s到达。旧单phi把首任务HBM的50B摊到100s；公平分配时外部端点到t=120s才完成。独立L在t=50s完成并释放HBM，外部端点在t=100s完成；计算仍在100s结束。两者独占时间相同，并发差20s。因此“独占延迟吻合”不足以认可动态竞争。

### 8.2 重复稳态与有限缓冲

同质重复不逐wave建事件。pipeline_fluid_v2用最多三条活动流表示加载、计算、写回的总工作，并用常数个缓冲credit限制预取/排空。不能一次把所有重复L做完而把整层免费放SRAM。

对具有相同输入/输出单位的重复段，设l,c,s为各流累计完成的等效wave，q_in0/q_out0为段入口已经就绪的输入/输出；B_in/B_out由候选实际缓冲槽数确定：

~~~text
0 <= q_in0 + l - c <= B_in
0 <= q_out0 + c - s <= B_out
~~~

到credit边界后速率必须满足切向约束；例如输入满时dl/dt≤dc/dt，输入空时dc/dt≤dl/dt。无需每周期轮换读写事件。典型双输入/双输出模板B_in=B_out=2，进入中间段时q_in0=q_out0=1；不同模板使用自己的实际槽数，不拿这个示例替代容量检查。

有K累加、延后输出、不同byte/wave的模板，AccessCounter必须提供确定的load_ready/compute_consumed/output_produced/store_retired credit曲线与解析循环类别，不能机械套用l−c=c−s。曲线用固定数量的分段仿射描述；真实离散tile readiness用离线oracle验证，流体近似不声称能看到半个真实tile。原图的外部ready只在对应区域或显式输出写回完整终结后触发，不能由平滑credit提前发布。

最多32个循环类别按原循环次序记录，重复次数可很大。默认合并为少量服务epoch；不等类别不能仅做无顺序的max(sum(work/capacity))。可在首/尾、K首末、输出产生、瓶颈/活跃核变化处解析细分，规则见8.6；不按重复次数展开。

### 8.2.1 累计服务曲线、计量单位和边界推进合同

每条活动进度p_L/p_C/p_S在[0,1]内。AccessCounter根据选中循环、真实shape/精度/TP/span、full/tail及活跃核数，解析生成固定数量的分段仿射函数：

| 曲线 | 含义 | 计量单位 |
|---|---|---|
| L_in(p_L) | 已就绪输入累计量 | 该模板统一的K消费单位 |
| C_in(p_C) | 计算已消费输入累计量 | 与L_in完全相同 |
| C_out(p_C) | 可写回输出累计量 | 该模板统一的输出块单位 |
| S_out(p_S) | 已完成写回累计量 | 与C_out完全相同 |

无K累加的模板使用自身解析输入消费单位，不人为引入K循环。K消费单位与输出块单位分别统一，不能互换或与byte混用；每个单位对应的物理byte/ops由原计数账本明确给出。曲线保留解析得到的首输出进度门槛、稳态和排空，各端点严格对应实际累计总量。竞争通过进度率改变到达门槛的时刻，不把孤立估值的固定时间当门槛。曲线不得根据参考完成时间自由拟合；不要求重现每个内部块的阶梯就绪瞬间。

~~~text
q_in  = q_in0  + L_in(p_L)  - C_in(p_C)
q_out = q_out0 + C_out(p_C) - S_out(p_S)
0 <= q_in <= B_in;  0 <= q_out <= B_out
~~~

SRAM静态检查继续精确计入双输入、累加器、待写输出、转换和中间状态的存活字节峰值；流体credit限制重叠和预取，不能代替静态检查，也不因credit使用抽象单位而改变字节需求。B_in/B_out必须来自同一模板的实际槽位合同。

缓冲满/空时直接约束可行速率（v为相应p的时间导数，曲线折点使用当前有效分段）：

~~~text
输入满：L_in'(p_L)*v_L <= C_in'(p_C)*v_C
输入空：L_in'(p_L)*v_L >= C_in'(p_C)*v_C
输出满：C_out'(p_C)*v_C <= S_out'(p_S)*v_S
输出空：C_out'(p_C)*v_C >= S_out'(p_S)*v_S
~~~

稳定分段内用p(t+Δt)=p(t)+v*Δt直接积分到必要宏段/活动/credit边界或真实外部事件。到边界只更新约束，不通过逐块“满→停→空→启动”轮换制造事件。物理服务仍满足Σ_a work[a,r]*v_a≤capacity[r]及原PRIVATE/owner公平约束；变化的是解析就绪表示，不是物理流量或Q/alpha策略。外部ready只在对应区域或显式输出完整终结后发布，真实collective/KV/restore/migrate/ACK源释放依赖保持。

### 8.3 去掉theta的动态共享预留

旧theta包络满足固定容量下 sum(max(w/cap))≤theta*max(sum(w)/cap)，但不是时变竞争的轨迹或上界。theta*工作拿去仲裁会把并不存在的占用分给其它任务。

本版动态仲裁只使用基本资源工作：真实payload、D/eta、解析计算服务秒及PRIVATE局部约束。theta只允许作为offline_conservative_diagnostic或与旧v1结果比较的字段；**不得乘到共享域工作、实际byte、输出操作数或公平请求上。** 不等循环类的聚合误差用有界活动/类别分段处理，不用拟合theta追平Excel。

每个L/C/S活动仍可同时需要多个路径域，例如一次HBM加载受到排、入口、Mesh、SRAM落地的共同限速；其向量内采用同一进度，避免把同payload逐层串行相加。是否需要将一次活动再拆小，只由8.6证据触发，默认不逐链路、逐PE/Vec拆分。

### 8.4 分组公平与事件算法

活动a的规范基数Q_a固定按HBM payload→LOCAL_BUS→SRAM读写→逻辑NoC payload→纯计算1 byte-equivalent的优先级选择。它只定义仲裁单位，不写入物理字节。活动进度率v_a，canonical服务率y_a=Q_a*v_a，资源系数work[a,r]/Q_a；PRIVATE约束仅限制该活动的活跃核，不占另一条全局物理总线。

fair_owner固定为(rank,outer_node_id,execution_generation)，request/session ID仅追踪业务，不自动聚合成另一个公平策略。一个外节点及其内部拆分/endpoint片段只有总权重1。当前可行活动的权重alpha_a=Q_a/sum_owner(Q)，相同流拆成片段时按Q分权，不增加owner份额。finished、被外部依赖阻塞或没有可行正速率的活动不参与分权；credit在空/满边界可被同段另一流同时供给/消耗时仍是可行活动，不能按零credit机械反复开关。权重只在资源/credit/成员事件变更时重算并存入快照。纯计算仍沿用Sys计算槽约束，不能复制一份die算力。

仲裁定义为在容量、PRIVATE及credit边界速率约束下，对y_a/alpha_a做加权max-min分配；所有活动的总物理服务满足sum_a work[a,r]*v_a≤capacity[r]。带符号credit约束不适合直接套仅支持非负矩阵的旧水填充。实现采用固定规则的active-set/小规模词典序LP；变量≤3J，约束由固定R和每请求常数credit组成，无tile/Core变量。冻结顺序由(resource_id,owner_id,activity_id)决定，退化解按同序选基，数值容差和迭代上限版本化；数值失败返回 numerical_failure，不能继续用旧单phi出结果。

每次先advance_to(t)：用旧速率推进各活动独立进度/计数和credit，积分保持非负；随后按以下同tick顺序处理：

1. 已完成活动、credit边界与框架终结；已完成请求只terminal一次。
2. 取消仍未完成者；撤销其请求/预约并释放原槽。
3. 已知外部ready/到达/容量或profile变更。
4. 重建可行活动集合、分组权重与速率；generation递增，预约最早下一事件。

事件只可能来自活动终结、credit触边、有限epoch边界及原外部事件。稳定速率的同质稳态直接积分到下一边界；不在credit满/空处以每wave发事件反复切换。边界速率约束令流保持可行，内部段切换次数由固定描述符上限决定。正时间间隔ceil到ns且至少1ns；零工作即时闭合，限定一次有限DAG处理，不制造无限零时回调。

物理计数由所属活动自己的进度积分，完成时强制精确整数；多个活动的同类计数相加但不能重复收费。取消只保留已服务部分和整数舍入余量。未完整写回数据不ready；NoC估计计数与HBM真实模型payload分开。详细tile/栈时间线不属于本模型输出。

### 8.5 容量支持与外部路径

双输入缓冲以外仍须计当前累加、待写输出和状态。最小合法候选仍不能装入，返回unsupported_workspace及所需/可用容量；不能漏算中间段或转回Roofline补时。W=1可合法顺序执行而不申请第二份输入。全部64种源Core及支持的常规shape须在工程验收中覆盖。

COMP、COMM、RESTORE、POOL的本地HBM端点共用同一仲裁域。HBM↔Core按profile收费排/入口/Mesh以及实际SRAM落地；HBM↔D2D-edge只在图声明需经过时收费内部互连；remote-pool-only由原池模型处理；NoC-only不增加HBM byte。恢复写与随后kernel读是两次访问；restore、recv和计划不能重复拥有同一写端点。网络由原后端推进。

运行频率按源文固定1GHz。显式update_capacity或共享服务profile更新必须先结算旧速率、再改新速率及generation；不能回写已发生计数或预测侧单独换倍率。

### 8.6 有上限的误差驱动细分

五段框架是候选最低粒度，不是未经验证的正确性结论。离线建立独立的逐消费wave参考oracle，使用相同基本资源容量/数据流但保留离散输入就绪、计算结束、输出产生、有限缓冲；只在小shape/短窗口运行，不进入五仓生产热路径。相同模型假设的oracle用于发现聚合误差，不是硬件精度证据。

从合法最低框架（最多5段，零工作段合并）开始，再检查E≤8、E≤16服务epoch/primitive kernel。细分边界取循环首尾、full/tail、K首末、负载/瓶颈变化和oracle显示的主要活动突变；由解析类边界/等量前缀求位置，不枚举全部tile。选通过证据的最小E，不能总开16段。固定模板的子kernel数显式上限4（物化SwiGLU即四个），且同region子kernel串行，一个region最多64服务epoch；超限由编译器拆成PlanBundle内连续region，不偷偷加活动数或改变外部完成。计数不随内部维度增长。

当前工程精度按§1.6：独占延迟、逐请求并发完成及原共享HBM繁忙积分对照目标误差≤10%；多次合理测试/调整仍难达到时，≤20%可带明确记录与范围接受。上述是当前模型对照预算，不将oracle对拍称为硬件实测校准。工作量、容量和归账必须精确守恒。接近排序的配置若差值小于误差带标undetermined；超过误差带的配置须同序。不同phase-offset、短/长读写、计算长访存短、恢复突入、不同n/B/SRAM/Nx/Ny须分别覆盖，不能仅验总block时间。

运行时默认不启动oracle、不拟合、不按到达频率无限细分。granularity_certificate记录template、E、结构/参数/shape/并发强度覆盖及误差；原参数离开覆盖域标extrapolated。E=16仍失败时返回granularity_validation=failed，假设研究可选择显式接受近似并披露误差，不能把它列入验证性能排序或自动退回逐tile版本。若要覆盖该域，先改聚合表示并重验开销，必要时才另立细粒度扩展。

## 9. 校准超参、初值与可识别性

### 9.1 允许少量自由度，禁止逐架构/逐算子追平

超参存放在独立 calibration.json，包含单位、范围、来源、适用 profile、执行模板、拟合样本和版本。64 行 Core 几何与资源数、HBM 理论规格、bump 及 10% 系统占位均不参与时间拟合。

不为 64 种 Core 各拟合一套效率，也不为每一行 Excel 设独立延迟乘数。一个参数应对应明确资源或执行行为，并在适用域内共享。

### 9.2 超参清单

下表的“初值”仅用于第一版假设仿真；不等于已测量的目标性能。输出默认 calibration_status=assumed，不能因为具有数字就标为 calibrated。

| 超参 | 第一版初值/范围 | 作用 | 现有 XLSX 能否确定 |
|---|---|---|---|
| eta_pe | 0.90；(0,1]，工程先验 | 目标 INT4 矩阵有效率 | 不能确定；TX8共同BF16参考效率不得迁成INT4初始化/校准 |
| eta_vec | 0.90；(0,1]，工程先验 | 固定向量/SFU模板的等效效率 | 不能由现表确定，也不能与目标eta_pe一起拟合 |
| eta_hbm_matrix | 0.70；(0,1] | 矩阵类 HBM 等效服务量 | 0.70 在表中是工程设定，不是独立实测 |
| eta_hbm_vector | 0.70；(0,1]，默认与matrix绑定 | 目标向量类HBM等效服务量 | 不能确定；TX8的0.3411668仅用于reference DDR路径 |
| eta_sram | 1.0；(0,1] | 固定一读一写假设的 SRAM 有效率 | 不能确定，先固定；有本地访存微基准再拟合 |
| eta_local | 0.70；(0,1] | Core 内 B-bit 总线有效率 | 不能确定，固定先验并做敏感性分析 |
| eta_mesh | 0.70；(0,1] | Mesh 有向链路有效率 | 不能确定，固定先验并做敏感性分析 |
| eta_inject | 0.70；(0,1]，仅有入口域时生效 | 聚合入口持续效率 | 不能确定，先固定；无入口限速情景不使用它 |
| timing_topology_profile / ingress_raw_Bps / route | 必须显式选择7.4的情景；explicit值非负且附来源 | 接入位置、原始速率、共享和路由结构 | 无证据；属于结构假设，不能与效率一起拟合追平时间 |
| rho_sram | 0.90；(0,1] | 可用于工作集的每核 SRAM 比例 | 表仅可检查某个工作集能否装入，不足以拟合 |
| launch_ns_per_macro | 1000 ns；非负 | 每个真实区域实例的启动 | 原短计时边界不足，当前不能独立拟合；参考 profile 为0 |
| memory_setup_ns_per_macro | 0 ns；非负 | 每区域首次非零计算侧 HBM 访问的额外延迟 | 当前不能确定；先0并标未覆盖真实首访时序 |
| kappa_quant_eqops | 4 eq-op/转换操作数；非负 | 目标量化/打包、解码等模板成本 | TX8 非 INT4 转换测量，不能确定，先固定 |
| kappa_acc_convert_eqops | 2 eq-op/累加结果；非负 | INT32→FP32 与尺度转换 | 不能确定，先固定 |

初始化时只放开少数参数：

- 参考 profile 的向量 DRAM 有效率。
- 在接受 FFN 计时解释的条件下，共同计算效率 eta_compute，且令 eta_pe_reference=eta_vec_reference。
- 其余维持固定值，并显示 sensitivity_fixed，而非优化器悄悄调整。

后续只有新增独立数据能区分资源影响时才解开绑定。topology/route、rho_sram和流水模板影响结构，默认不与效率共同拟合。目标eta_hbm_matrix/vector先共享同一先验，不将DDR Norm记录迁成HBM向量校准。参考与目标的配置空间、参数名和来源分别保存。

量化的具体尺度算法可能改变工作量。第一版用固定模板计数，kappa_quant 不代表通用准确实现；更换算法需更新 execution_template_revision。

theta仅为旧包络离线诊断，不能用于本版共享仲裁；结构profile也不是优化器的自由效率乘数。基本D/eta、真实D和诊断包络分别记录。

### 9.3 当前 XLSX 的数据身份

当前 TX8_单die建模_精简版.xlsx 的“总览_大小算子数据!A2”注明原始九条记录已移除；当前文件没有可审核的原始测量行/重复日志。保留的13组是硬编码组值与记录解释，很多由工程重建；19个数学节点不是19个独立样本。

| 数据 | 校准处理 |
|---|---|
| RMSNorm 246 μs，计时组13行!B2/D2 | 根据说明解释为完整一次Norm；reported_anchor，无原始日志和重复统计 |
| 3363 μs，计时组13行!B13/D13 | 原SiLU标签重解释为整个FFN；conditional，可开关纳入，不能当独立PE/Vec测量 |
| 其它派生计时组 | 只作工作量/归账回归；实测拟合权重为0 |
| 11897.602619 μs 整层合计 | 参考公式合计；不是独立整层测量 |
| 48 MB SRAM / 200 GB/s /128T /1T | TX8 参考硬件和口径，不进入目标架构资源表 |

导入器逐条保存reported_anchor、conditional、derived及原单元格；measured_raw仅用于以后带可审核原始日志的数据。不能从同一模型生成数据再拟合并报告预测精度。缺少重复/边界元数据不自动补默认并标成测量事实。

### 9.4 TX8 参考复算

参考 shape：B=1，Sq=Sk=1024，H=4096，F=14336，Q/KV 头32/8，head_dim=128，单层、TP=1；BF16 输入/权重，FP32 score/stat/accum，完整 score 外存物化。

~~~text
M=1024
J=Hkv*head_dim=1024
A=B*Hq*Sq*Sk=33554432
eta_compute_reference
 = (6*M*H*F/128e12 + 16*M*F/1e12)/0.003363
 = 0.9079551923877489

BW_vector_reference
 = (4*M*H+2*H)/0.000246
 = 68.23336585365854e9 B/s
eta_vector_memory_reference = BW_vector_reference/200e9
                           = 0.3411668292682927
BW_matrix_reference = 200e9*0.70
~~~

参考组时间采用max(Fop/P+Eop/V,D/BW_class)。这些是本文重建的参考公式；工作簿本身实际公式只有计时组13行!B16=SUM(B2:B14)，B15为硬编码总数。参考复现与新pipeline的填充、credit和动态竞争不是同一模型，不能要求目标粗模型逐组输出等于参考值。

| 组 | Fop | Eop | 参考物理 byte |
|---|---|---|---|
| Norm | 0 | 6*M*H | 4*M*H+2*H |
| Q/O，各一次 | 2*M*H*H | 0 | 2*(M*H+H*H+M*H) |
| K/V，各一次 | 2*M*H*J | 0 | 2*(M*H+H*J+M*J) |
| RoPE QK | 0 | 3*M*(H+J) | 4*M*(H+J)+4*Sq*head_dim |
| QK | 2*A*head_dim | 0 | 2*B*Hq*Sq*head_dim+2*B*Hkv*Sk*head_dim+4*A |
| scale/mask/softmax | 0 | 12*A | 8*A |
| AV（PV） | 2*A*head_dim | A | 4*A+2*B*Hkv*Sk*head_dim+2*B*Hq*Sq*head_dim |
| Residual | 0 | M*H | 6*M*H |
| fused FFN | 6*M*H*F | 16*M*F | 6*H*F+4*M*H |

Norm 两次、Residual 两次；Q/K/V/O 分列；RoPE 和 softmax 组分别合并其数学子项，构成13组。总计应复算为11897.602619368336 μs。

### 9.5 新模型校准流程与证据等级

当前两项反推仅属于tx8_reference_v1：一项是固定byte假设下的向量DDR有效吞吐，另一项是接受3363μs解释、固定峰值和共享效率假设下的共同计算效率。没有噪声估计、剩余自由度或独立验证，不能拆为目标INT4、Vec/SFU、HBM、NoC或流水参数。现有目标参数均是engineering_prior，目标芯片若只是设想，不能宣称已硬件校准。

真正校准时必须调用新公共库的compile/estimate/execute，不能拟合9.4参考公式后直接把系数贴给coarse_die。步骤为：

1. 以设备/clock、dtype/accum、完整shape与布局、TP本地分片、KV位置、融合/缓存、预热/重复/统计量、同步/传输计时边界生成独立CaseManifest；保存原始记录和内容hash。
2. 固定可解释的topology、量化和访问模板；先比对工作量/缓冲/实际byte，再拟合效率。结构有错时改结构，不能由eta补偿。
3. 用独立资源探针建立log-time对参数的灵敏度矩阵。p个自由参数至少需p个独立设计条件且局部灵敏度满列秩；这是必要条件，不是足够的统计验证。始终不敏感/共线的参数固定或绑定，并报identifiability=unresolved。
4. 当前最多放开两个reference有效量；目标参数无数据时全部冻结。补数据后按INT4矩阵、Vec/SFU、HBM读写/排与入口、本核总线/SRAM/Mesh、启动/转换的资源路径分组逐步解开，避免一次拟合全部。
5. 在拟合集上使用事先固定的相对/log误差目标和有界优化；派生组拟合权重0，conditional单独报告。保存每次真正的新库输出、plan hash、参数、残差与模型身份。
6. 留出未参与拟合的shape/配置/资源受限类别，预先确定用途所需的误差门槛，再测偏差、p50/p95、分层误差与排序；重复测量估计噪声。近于测量/模型误差的配置差别标undetermined，不能强排名。
7. 只有通过的精度、设备、shape、配置区间升级为calibrated_subset/validated_subset。TX8一个profile通过不能代表全部64种Core或假想HBM3E硬件。改变模板/topology后旧校准证书不自动有效。

| 验证层 | 证明什么 | 当前证据或实施状态 |
|---|---|---|
| source_integrity / xlsx_cache_reconcile | 权威资源公式、表与Excel缓存一致 | 可复核；不等于性能准确 |
| tx8_reference_reproduction | 条件性参考公式与保留组值一致 | 已有资料可做；仅reference域 |
| coarse_library_execution | 实际编译的新库遵守自身模型、归账与生命周期 | 新库待实现，not_run |
| granularity_oracle_validation | 聚合与离散同假设oracle的差异在预算内 | 待实现/运行，not_run |
| target_component_calibration | 独立目标资源数据能识别选定超参 | 数据not_available |
| heldout_hardware_validation | 新库对独立目标观测的预测准确度 | 数据not_available |
| five_repo_integration / overhead | 旧业务、兼容指标、同源投影和开销符合要求 | 新适配待实现，not_run |

各层互不替代。拟合得到一个参数并不等于独立验证通过，软件fixture通过也不等于测量准确度通过。

### 9.6 最少补测需求

优先恢复原始计时日志/实际边界；补同算子的多M/K/N、prefill/decode、KV长度与SRAM驻留阈值，保留warmup和每次重复值。资源层补INT4及所用转换、Vec/SFU、HBM-only读写与共享、入口/NoC位置与截面、本核SRAM/LOCAL命中、启动及首访。动态层补计算长而访存短、恢复突入、独立排与共享入口、多任务phase-offset的完成与byte/busy轨迹。

不能要求虚构目标配置产生真实测量；目标为假想芯片时，可用有完整来源的RTL/更细模拟器作为model_to_model_validation，保留假设与外推状态，不改名hardware_validated。当前研究用途的预测误差目标为10%、多次测试困难时容许20%（§1.6）；模型oracle、TX8条件参考和独立硬件数据的来源仍必须分开，不能互相冒充。

## 10. 输出契约

### 10.1 静态与独占结果

| 输出 | 单位/含义 |
|---|---|
| status / reason | ok_assumed、ok_calibrated_subset、unsupported、invalid_hardware 等 |
| estimated_latency_ns | 指定执行模板、独占资源下的模型延迟 |
| macro_breakdown | 框架/epoch及各L/C/S独立活动的时间、credit、首尾与setup |
| bottleneck_resource | 单阶段受限域；可含多个并列瓶颈 |
| source_math_ops / executed_ops | 各精度源数学量与含重算/转换的执行量；不得混用 |
| timing_compute_work | η、尾波、活跃核折算后的计算服务秒；不是新的物理 FLOP |
| vec_equivalent_ops / conversion_elements | 版本化等效工作量及转换量 |
| logical_tensor_bytes | 原图兼容字节和张量容量 |
| hbm_read/write_bytes_total | 解析数据流的精确计划payload，含重复访问；不是实测硬件事务 |
| modeled_hbm_read/write_bytes_by_row | 均匀条带或显式row信息分配的模型payload；附allocation_evidence |
| hbm_equivalent_units_by_row | 基本D/η，独立于真实byte及离线theta诊断 |
| resource_service_work | 每活动基本时间/byte/hop-byte工作与PRIVATE约束；不含theta虚构预留 |
| sram_read/write_bytes | 解析得到的片上访问量 |
| local_bus_bytes | 内部总线 payload，计数规则有模板版本 |
| estimated_noc_hop_bytes / cut_bytes | 聚合路由近似量，不是逐边实际包日志 |
| active_cores / output_round_count / consumption_wave_count / tile_choice | 分块、利用率、重算与容量阈值解释 |
| sram_peak_bytes_per_core | 当前区域工作集峰值 |
| hbm_workspace_peak_bytes | 临时外存需求；只返回增量，不重复申请原权重/KV |
| derived_hardware | 64行资源、几何、bump/D2D、模块和系统资源 |
| hardware_fingerprint / calibration_revision | 完整配置与执行假设身份 |
| evidence_flags / validation_statuses | source_model、engineering_assumption、TX8_conditional、derived；按9.5逐层报not_run/不可用/通过 |
| topology / routing / granularity_identity | 入口/路由/epoch/证书身份与coverage，不能只报总延迟 |
| ranking_status / scenario_report_ref | 未测试、对假设敏感、在已测情景稳定或误差内无法确定 |

不支持的操作或未定义的执行精度返回错误；完整假设 profile 允许给出有限延迟并用于五仓库的假设架构实验。空任务可返回0。缺参不能静默切回旧 mesh_die_v1 的数字。

### 10.2 运行结果与既有统计

运行时额外输出：

- issue/start/finish/cancel 时间与 elapsed_ns。
- 各L/C/S活动累计模型已服务byte/操作数、等效服务量、独立剩余进度、credit与终结身份；“已服务”指本仿真模型，不等于硬件测量。
- 模型HBM正服务率繁忙区间及同排共享区间；与setup、等待和GPU槽占用分开，不称硬件测量时间线。
- 计算槽繁忙区间；它不等于 PE 实际利用率。
- 已服务INT4/Vec工作；分别按相容精度峰值和明确观察窗口定义typed利用率。
- 资源域使用率、瓶颈、竞争对象数和重分配次数。
- 区域取消时的已服务量；未传数据不计入物理流量。
- 仍由原仓库产生的 request/session/turn、prefill、decode、首 token、完成边界。

利用率分子必须说明来源。旧 num_ops/tensor_size 的兼容统计、新计划物理工作量和实际已服务量分别保留。NoC 聚合量带 estimated 前缀；不制造不存在的逐 Core/逐栈测量日志。

运行时先保留聚合计数；详细宏阶段日志默认关闭。需要功耗工具时，可以导出所选统计窗口的操作/访存活动计数；本工具不凭这些计数自动宣称已实现能耗、功率或温度模型。

### 10.3 容量与上层账本

原 KV/权重管理器继续负责权重、KV 驻留、预留、逐出和恢复。resolver把2n*36e9 byte作为per-rank/per-die本地容量传入其配置；不能传system total，也不能再除TP度。

新模型必须返回每核 SRAM 和本地 HBM 临时工作区峰值；上层准入应把 workspace 加到已存权重/KV账本中。不能同时由编译器和 KV manager 为同一原张量各记一份驻留容量。

若保留 BF16 外层存储，原 KV 字节定义保持不变；只有显式改变 storage_dtype 的实验才联动修改字节、容量与通信量。内部 INT4 tile 不自动把上层 KV容量缩成四分之一。

### 10.4 旧输出的逐项兼容与新字段

coarse分支保留现有record/key、基本类型及窗口，并声明metric_schema_version=2、cost_model/profile、metric_source和分母来源；需要迁移的计算含义在下表明确。旧后处理同步支持v2，不能依靠同名字段偷换语义。legacy模式按原公式/gate继续；DIE新字段采用分精度资源，不能把INT4 PE峰值塞回generic FP32标量并称同一指标。

| 既有输出/消费者 | v2合同 |
|---|---|
| COMP num_ops / tensor_size，rank_compute.total_num_ops / total_local_bytes | 原旧键继续输出，compat_num_ops/compat_tensor_bytes仅为新增别名/内部侧带；u128累计以decimal string无损输出。不是source/exec ops或解析HBM byte，也不能再收费 |
| Sys peak_flops_per_second / local_hbm_bw_bytes_per_second / latency | 原legacy配置标量及单位保留：TFLOPS×1e12、GB/s×1e9、ns。coarse动态只读typed descriptor；新增die_peak_int4_ops_s、die_peak_vec_eqops_s、die_hbm_raw_Bps_by_row。derived_hardware.hbm_raw_Bps_by_row固定为每排n*1.1776e12，二者同源只读。不得把INT4/Vec相加或把GB/s当Gbit/s |
| Python WscLlmHardware容量/带宽/D2D/peak字段 | KV管理器取per-die的2n*36e9容量；新descriptor提供分精度峰值、两排率、单向/双向D2D及来源。旧scalar估时getter仅用于legacy，不在coarse selector偷偷执行旧roofline |
| rank_compute.achieved_compute_util / dram_bw_util | legacy分别是compat总ops/(legacy峰值×全窗)、compat总byte/(legacy带宽×全窗)，沿用原gate/null/note。coarse不能改用served量覆盖；typed全窗工作利用率另报 |
| active_kernel_roofline_compute/memory_util | 保持legacy逐COMP时间加权和gate；coarse不冒充roofline字段 |
| active_kernel_die_compute/memory/vector_util | node-work/time统计，使用v2对应typed执行/服务工作、相容PE/HBM/Vec峰值×该节点elapsed，再按有效COMP时间加权；不是资源忙碌率。完整完成时served=plan，取消使用served并带cancelled标识。严格Vec矩阵路径PE项为0，不能分错峰值 |
| die_compute_busy_util | 原GPU_COMP区间并集/全窗，可覆盖计算等待期；另报实际COMPUTE_GATE服务区间，不混称PE occupancy |
| local_hbm.dram_bw_util | 所有唯一本地端点模型served payload（含die）/(每rank本地die两排raw B/s之和×全窗)，来源derived_hardware.hbm_raw_Bps_by_row[0]+[1]。v2输出denominator_source；不是rank_compute的requested-byte指标，也不是eta折算服务量 |
| local_hbm.compute_bytes_served | 保留legacy COMP队列类别；新增die_compute_read/write_bytes_served，不把die无声塞进旧类别。总分子仅将不同类别相加一次 |
| comm_read/write、restore、pool_read/write_bytes_served | EndpointResolver按唯一owner/path映射稳定类别；新row/方向分量另报，D2D直通无本地byte |
| local_hbm_restore_bytes / ..._issued | 原issued/requested口径保留，基础仓仍0；不得拿cancel后的served覆盖它 |
| hbm_busy_ns / hbm_busy_util | 至少一个HBM活动有正服务率的区间并集；两排同时busy不加两次。row_busy、等待/setup另报 |
| hbm_shared_ns / hbm_shared_fraction_of_window | v2为至少两个不同owner在同一HBM排域传输的区间并集，字段带shared_definition；不同排独立活动不假称同HBM争用。入口/Mesh的shared另报 |
| peak_concurrent_jobs / redistribution_events | 兼容facade按活跃HBM逻辑owner计job与owner集合变更；新增arbiter_active_activities / allocation_updates，不用L/C/S数量替代旧job数 |
| capacity_timeavg / planner_memory_peaks / hbm_watermark及summary | 保留原账本/窗口/锚点；新增operation级workspace增量计入准入与占用积分，不重记原KV/权重 |
| request / summary / consistency / memory_anchor | 身份、边界、exact/proxy来源、吞吐/审计和容量锚点保留；完成时间随实际模型改变，不由预测总时替换 |
| full-detail iteration | 保留point、TP/ranks_seen、num_ops_total、local_tensor_bytes_total与legacy util gate；iteration_time随真实完成改变，新typed per-iteration利用率用新键 |
| WS microbench汇总 | run总量、iteration分母和有效点选择沿原脚本；新模型量单列，不能用计划重复数再除一次迭代数 |

metrics_postprocess.py目前对DIE三种active_kernel_die利用率做fail-closed检查，必须随v2明确精度/取消/0时间语义更新，不能删检查或靠clamp隐藏>1/负值。正常服务量应满足相容峰值的范围约束；非有限数返回metric_error及来源。全窗、COMP窗、节点elapsed、HBM活跃窗逐字段携带window_kind/start/end。

五仓共同修正die实际byte未进入local_hbm带宽分子的路径；保留die/legacy/comm/restore/pool独立类别。每仓full-detail全部既有record/key的presence、类型、单位、gate与窗口逐项对拍，新增键不破坏旧读取。

## 11. C++ 公共库、ABI 和目录

### 11.1 代码组织

| 路径 | 职责 |
|---|---|
| die/include/Hardware.hh、Task.hh | 全源文变量、派生硬件、类型化task及来源 |
| die/include/Plan.hh、Execution.hh | 不可变epoch/活动/credit描述；少量可变进度和代号 |
| die/include/ResourceArbiter.hh、Capi.h | 固定域仲裁与版本化ABI |
| die/src/Hardware.cc、Mapping.cc、AccessCounter.cc | 源资源、确定候选/循环、动作与生命周期账本 |
| die/src/Compiler.cc、Arbiter.cc | 最小已验证计划；独立活动、credit与实跑/投影共同算法 |
| die/src/Tx8Reference.cc | 分离的TX8参考重建 |
| die/configs/ | Core64、schema、结构情景、工程先验与校准证书 |
| die/tools/、die/tests/ | 离线导入/拟合、离散oracle、变量/事件/兼容/开销报告 |

C++17与现有项目统一；离线XLSX/拟合可用Python，运行时不启动Python、不读Excel、不联网。JSON复用现有依赖，仅配置/批量compile/导出使用，热路径为固定数组和结构体。

### 11.2 同源接口与所有权

~~~cpp
Hardware resolve_hardware(const ResolvedArchitectureInstance&,
                          const PerformanceProfile&);
Plan compile(const Hardware&, const DieTaskSpec&);
CostResult estimate_alone(const Hardware&, const Plan&);
ProjectionResult estimate_on_snapshot(const Plan&, const FrozenSnapshot&);
ExecutionHandle submit(const Plan&, ResourceArbiter&, OwnerToken,
                       CompletionCallback);
void advance_to(ResourceArbiter&, uint64_t absolute_tick_ns);
void cancel(ExecutionHandle, uint64_t absolute_tick_ns);
Progress query_progress(ExecutionHandle);
FrozenSnapshot export_snapshot(ResourceArbiter&, uint64_t now_ns);
PrivateRuntime import_snapshot(const FrozenSnapshot&);
~~~

以上是待实现接口。每rank仅一个物理本地域；所有COMP/外部HBM端点进入同一实例。Plan不可变、可共享；各候选Execution/credit/预约/统计独立。estimate_alone也驱动同一个推进器，不能另外写一个只返回公式max的估时实现。

### 11.3 C ABI v2的字段、错误与生命周期

C ABI以die_coarse_abi_version()==2为合同，所有结构首部含struct_size/schema_version，固定宽度整数，UTF-8文本，资源顺序由ResourceId固定。ID是带generation的opaque handle，不能序列化主机指针；C++异常不得越边界。

~~~c
typedef uint64_t DieHandle;
typedef struct {
  uint32_t struct_size, schema_version;
  uint64_t request_id, node_id, execution_id, generation;
} DieOwnerV2;
typedef struct {
  uint32_t struct_size, schema_version, role, enabled_resource_mask;
  uint64_t activity_id;
  DieOwnerV2 owner;
  double service_work[13]; /* 单位由ResourceId: s/B/hop-B */
  uint64_t hbm_payload_bytes[2][2]; /* row × read/write */
  uint64_t executed_int4_ops, vec_eqops;
  double canonical_Q, owner_weight, private_time_bound_s;
} DieActivityV2;
typedef struct {
  uint32_t struct_size, schema_version, terminal_kind, status_code;
  DieOwnerV2 owner;
  uint64_t issue_ns, finish_ns, elapsed_ns;
  uint64_t hbm_served_bytes[2][2];
  uint64_t int4_ops_served, vec_eqops_served;
} DieCompletionV2;
~~~

三个活动的progress、credit、epoch/tail描述及路径子计数通过query_progress_v2/snapshot_v2完整返回；上面不是用汇总结构替代这些状态。resource mask禁用不存在/无限入口域；所有提交量有限、非负、单位明确；有效路径有非零工作但容量0时拒绝invalid_hardware，不能给inf或把路径偷偷禁用。证据状态由引擎核验加载的certificate/manifest后生成，输入声称calibrated不构成验证。u128累计兼容计数在JSON/wire用精确decimal string，禁止经double转一遍；每次请求的u64量超界返回overflow。

| ABI操作 | 合同 |
|---|---|
| create_model / compile_batch | 返回逐项status、field_path/reason、plan handle/hash和summary；部分失败不伪造延迟，成功项可以保留 |
| estimate_batch | 返回每rank独占ns、工作/容量/身份/证据；不默认返回TP组或selector分数 |
| create_runtime / submit_plan / submit_endpoint | runtime持有Hardware/Plan只读引用；返回Execution及活动token，最终外节点join一次 |
| advance_to / next_event / cancel_execution | 使用绝对ns；时间倒退报invalid_time；先结算旧服务再改变集合 |
| export_snapshot / import_snapshot / fork_snapshot | 导出冻结状态；导入先全量校验再原子创建新world，失败不修改活跃runtime |
| query_progress / query_counters | 区分plan量、issued量、模型served量，携带窗口和generation |
| release_plan / destroy_execution / destroy_runtime / destroy_model | release_plan仅释放调用者引用；活动runtime继续持有plan。活动Execution不得直接destroy，先cancel并终结。runtime destroy需drained；model有live runtime/plan返回busy |

status_code固定含invalid_input/hardware、missing_topology_profile、unsupported_op/dtype/head_mapping/workspace/mapping/projection_state、overflow、incompatible_schema/fingerprint、unsupported_legacy_active_state、invalid_time、busy、stale_handle、numerical_failure；错误诊断含字段路径/ID，不能只给-1。成功的假设估计另附calibration/granularity状态，status=ok不暗示校准通过。

callback只在外层DONE/CANCELLED终结各发一次；内部活动完成不释放GPU槽。适配器管理网络/pool/restore与本地域的join。callback上下文保持到终结或显式取消确认；终结后作废所有内部token/generation，不得访问已释放的host数据。禁止在同一runtime回调内重入推进；后续提交排入同tick待处理队列。C ABI不得销毁尚被预约使用的execution。

### 11.4 身份、缓存与重现

hardware_fingerprint含完整源参数/派生资源、source revision、Core64、topology/route及服务profile；plan_fingerprint再含本地shape、dtype、spans/TP、数据流、驻留、转换、循环/credit、细分级别/证书和校准版本。snapshot额外记录clock/event-order/fairness/rounding版本。任何结构/模板改变不得复用旧时间缓存。

旧search_profile只作迁移元数据另记；预测由同一给定硬件实例与任务决定，不执行硬件搜索。缓存仅放不可变Plan，默认4096条或64MiB双上限按实际字节淘汰；不缓存执行句柄。新schema拒绝把mesh_die_v1/scalar state或coarse_die_v1单phi快照作为v2恢复。

### 11.5 完整快照与私有world

snapshot_schema=coarse_die_snapshot_v2，语义对C++对象和wire相同：

| 组 | 必需状态 |
|---|---|
| 身份/时钟 | ABI/schema、硬件/plan/template/topology/calibration指纹、时刻、同tick排序/舍入版本 |
| 容量/profile | 各域当前容量、启停、单位、shared_service_profile版本及已知变更 |
| pipeline | kernel/epoch/解析循环类、未消费重复量、首尾状态与每个L/C/S active/blocked/done、WR列组切换等宏段边界；不能只存一个phi |
| 活动工作 | 各活动剩余资源向量/独立progress、PRIVATE、Q、alpha、owner/group与fairness epoch |
| 有限缓冲 | 实际槽容量、占用/可用credit、累计ready/consume/produce/retire曲线位置、缓冲owner及依赖 |
| 所有权/事件 | request/node/execution/activity/endpoint/transaction ID、generation、GPU槽、ready/cancel/terminal、下一预约 |
| 计数 | 各活动物理byte/typed操作、D/eta计数与整数/浮点舍入余量，issued和served分开 |
| 外部HBM | COMM/restore/pool端点剩余量、row/route/path、setup状态、owner与join状态 |
| 原网络/池 | 原模型可复制的进度、队列/port、事件或版本化只读句柄及其clone生命周期 |
| 已知未来 | 完整有序到达/ready/取消/释放/profile变更；未知未来显式为空 |

export先advance_to(now)，原Sys适配层补齐外部模型状态。只含ETA/scalar余量的旧快照不能严格恢复v2；当前JOINT reset从immutable输入重新建world也不等于中途恢复，必须增加live-export→validated-import路径。

import/fork复制可变活动/credit/计数/预约/依赖，保留当前进度；Plan可共享，执行器不能共享。原网络/池若以句柄存储，必须由其模型clone出私有可变状态，保持原只读数据引用直至所有world释放；无法clone时明确unsupported_projection_state，不能复用实跑可变对象。每个私有world有独立token空间，原Sys过期回调在其中无效。

导入全量校验schema/指纹、唯一ID、generation、有限非负工作、已服务+剩余守恒、credit与缓冲容量、ready/依赖/terminal一致、事件时刻不倒退。任一失败原子返回对应错误。rate为可重算诊断，不能单靠旧rate恢复集合；相同冻结状态应重算出相同fairness结果。

known_overlay只含已知事件。baseline和每候选都从相同base快照新建world；候选不修改base。外部网络/池由原模型推进，die库只推进本地并join，不能复制收费。

## 12. 五仓库嵌入方案

### 12.1 当前代码基线与复用范围

五仓库已有 compute-cost-model=die、类型化算子、DieModelAdapter、DieExecution 和 extern/die_model。但目前是旧 mesh_die_v1：方形网格、16GB/栈、410GB/s/栈、聚合本地 HBM。不能把旧库改一个常量后称为本设计。

复用现有图输入、调度/依赖、C ABI 装配、完成回调和候选隔离骨架；替换硬件解析、解析工作量、粗计划和共享资源推进规则。

template/die/ 是唯一规范源。五仓库用相同 release/tag/内容摘要 vendoring 到 extern/die_model，构建不依赖兄弟目录或联网。同步工具校验公共源一致，仓库适配器不要求逐字相同。

### 12.2 共同接入链

~~~text
原模型/请求/TP分片
    → Python graph builder 的 typed DieTaskSpec
    → GraphBatch/ParsedGraphBatch/NodeStore/NodeView
    → Workload::issue_die_comp
    → DieModelAdapter 编译或命中不可变 Plan
    → DieExecution 提交宏阶段
    → 本 rank 的共享 ResourceArbiter
    → 原 Sys 事件/依赖完成
    → MetricCollector 与原 request/KV 日志
~~~

配置推荐保持 compute-cost-model=die，新增 die-model-profile=coarse_die_v2；roofline-enabled=0。现有 profile 白名单和 schema 必须同步升级。legacy roofline、旧 mesh_die_v1、tx8_reference_v1 都显式选择，coarse 模式缺参不自动回退。

本地内存单位统一B/s和byte，历史GB/s只在resolver→Sys转换一次；SRAM MiB为2^20，HBM GB为10^9。外端点原本的local-mem-latency仅由其EndpointResolver按原合同收费一次，COMP侧用memory_setup_ns_per_macro，不能在两处叠加旧latency。setup计时不计入HBM传输busy，同源投影保存其剩余量。

### 12.3 必改公共位置

| 位置/模块 | 修改 |
|---|---|
| Sys.cc / Sys.hh | 读取同一 resolved descriptor；切换 profile；给执行器和统计提供资源及校准版本 |
| Workload.cc | typed COMP 走新计划；timer、CPU replay、control_marker 保留原语义 |
| DieModelAdapter / DieExecution | 创建计划/执行句柄；提交宏阶段；共享计费、取消、drain与一次完成 |
| LocalHbmBandwidthModel | coarse 模式改为共享仲裁器的兼容入口；所有本地COMP/COMM/RESTORE/POOL服务归到同一域 |
| GraphBatch builder/parser/NodeStore/NodeView | shape、dtype、spans、TP、region、identity全程不丢 |
| MetricCollector | 区分compat/plan/served量；纳入die实际服务byte并输出正确带宽利用率 |
| Python die_model_oracle | ctypes批量编译/估值；不重新写时间公式 |
| CMake | 固定版本公共库、原依赖、ABI与适配回归 |

仅修改 Roofline.cc 不覆盖 die 分支；往 runtime_ns 填新总时间又会绕过共享访存。新模式真实 COMP 要求 runtime_ns=0，由事件执行器决定完成时间。

不能让新 ResourceArbiter 和旧 scalar HBM model 分别给同一请求计时：coarse 模式旧入口只是新仲裁器的桥接，不再独立 drain。

LocalHbmBandwidthModel在coarse模式是delegating facade：它可以保留旧函数名供调用方使用，但内部只能提交一次CanonicalEndpointRequest、查询新仲裁器计数；禁止在转发后继续旧队列计时。

### 12.3.1 节点收费与完成归属

下表按每次真实物理读/写定义owner。hbm_charge/hbm_access_mode由原图生产器声明；适配器不得仅凭节点名字再追加收费。

| 节点/端点 | 网络或远端端口owner | 本地HBM请求 | 完成与指标 |
|---|---|---|---|
| COMP内部加载/写回 | 无外部网络owner | MacroPlan拥有计算侧读/写；按原row映射 | 全部服务阶段DONE后仅一次完成COMP；仲裁器记served量 |
| 普通COMM_SEND | 原Network send | 只有hbm_charge声明源读时提交一次 | 原网络与本地读完成join；保留send身份 |
| 普通COMM_RECV | 原Network recv | 只有声明落入本地HBM时提交一次目标写 | 网络到达与本地写join；写完整才ready |
| D2D直通/转发 | 原Network转发 | 未实际落地本地HBM则不收费 | 原网络完成；不产生虚构HBM指标 |
| MEM_STORE到pool | 原RemoteMemory/pool写 | 按hbm_access_mode，直接端点可同时拥有一次源HBM读 | pool写与本地读join；迁移ack按原图释放 |
| MEM_LOAD从pool | 原RemoteMemory/pool读 | 若图随后有专门restore节点，此处不再拥有目标HBM写 | pool/网络完成先释放原恢复依赖 |
| local_hbm_kv_restore | 原恢复gate；网络/pool在其前驱 | 唯一拥有相应目标HBM写；不再重复pool读 | 写完成后历史KV ready；仲裁器记restore写 |
| POOL_READ/WRITE组合端点 | 原pool端口 | 只对图声明的本地源读/目标写建一次请求 | pool与HBM完成join，两个域各记自己量 |
| 计算消费已恢复KV | 无额外恢复owner | 当前kernel的实际HBM读；与恢复写不同 | 随MacroPlan完成，不把之前恢复再次收费 |

每个端点带transaction_id、endpoint_id、owner_kind、direction、physical_bytes、path_class和join_token。由图生产器或适配器把一次物理访问分配给唯一owner；同transaction可合法有源读与目标写，两者是不同endpoint，不能当重复删除。

旧端点无row信息时统一按row0/row1各1/2分配；奇数字节使用固定整数余数规则由row0多一个byte，标allocation_evidence=uniform_default。有显式row信息则使用该输入并校验总量；实跑和投影调用同一个EndpointResolver。排内仍均匀，不声称已知单栈地址。

HBM↔D2D-edge默认只收费HBM两排；是否经过内部入口/Mesh由版本化path_class决定，不自动收费Core SRAM。不能同时让原hbm_charge桥和MacroPlan拥有同一端点。

### 12.4 astra-sim-wscllm

- 在 sh_test_mesh/workload/llama2_7b_inference/online/graph_batch_builder.py 聚合前传真实算子/region描述。
- astra-sim/workload/Workload.cc、DieModelAdapter、DieExecution接新计划。
- 基础仓没有远端池业务，保留其已有 COMP、COMM_READ、COMM_WRITE接口。
- wsc_llm_scheduler.py、session_kv_manager.py从同一 descriptor取得本地 HBM容量；额外纳入计划 workspace。
- 当前主要是在线GraphBatch。不要仅为旧设计文档的静态ET愿景新增无关后端；有实际静态ET入口时再传播相同schema。

先用这个基础仓验收单 die 与网络端点并发，减少首次接入需要同时调试的业务。

### 12.5 astra-sim-wscllm-LRU

基础接入相同；额外处理 MEM_LOAD/STORE、RESTORE、POOL_READ/WRITE、远端池容量与完成ack。

按原语义逐条标记收费所有者：

- store：源本地 HBM读取只收一次；远端池写仍走原pool port。
- restore：远端池读取与目标本地 HBM写各归其所有者；随后 kernel读取另行计费。
- D2D直通或边缘转发：不因经过一个rank就给该rank补一份不存在的HBM读写。
- hbm-kv-restore-bandwidth-sharing=false等绕开共享的配置，在coarse物理共享模式下必须拒绝或显式选择独立兼容实验profile；不能静默给restore第二条总线。
- remote_weight_bytes如被启用，要在typed spec和原远端图中明确来源/就绪；未支持时显式报错，不能因转到die分支而漏掉读取。

### 12.6 astra-sim-face

FACE的真实执行用C++新计划；face_scheduler.py的候选代价通过同源oracle查询相同局部shape。

候选成本仍是用于原放置策略的孤立/指定快照估计，不宣称自动预测未知未来负载。保留当前候选集合、拓扑距离、tie-break和队列快照语义。

输出冻结三种量：per_rank_latency_ns是已经TP切分的本rank计划；tp_group_wall_latency_ns是各rank与原collective依赖图的critical path（不是简单把所有rank时间相加）；selector_score=policy_delta/instance.size，policy_delta固定为同composition_profile下candidate相对current的TP组wall-time增量，且仅归一化一次。不能再把真实本rank延迟除TP度。当前等大TP实例的归一化为共同除数，本版不擅改策略。

当前FACE候选仍在Python组合“五个linear求和＋max(prefill attention,decode attention)＋d2d_latency”，这与runtime GraphBatch/COMM并发图不完全相同。局部Plan同源不等于整个selector iteration和实跑必相等：保留旧策略组合时记录composition_profile=face_legacy_selector及approximate_predictor；严格已知world比较另提供用原task graph与同一推进器的oracle。替换selector的组合策略是单独实验，不能借模型迁移隐式改策略。候选假设、是否包含collective/排队、rank0是否代表最慢分片均记录；异构分片需逐rank验证，不能无条件只估rank0。

### 12.7 astra-sim-face-LRU

保留FACE全部接口，同时：

- 使用真实p_context_tokens与绝对query位置，不能退化成只有本chunk的attention。
- 保留全session/local-vs-remote容量状态、逐出、恢复、历史KV就绪及原迁移图。
- 把restore、pool endpoint和计算侧HBM阶段送进同一共享域。
- 修正local_hbm报表：当前代码分开记录die实际服务byte，却未计入该报表的带宽分子；新输出必须包含die served bytes。
- 继续保留首token exact/proxy来源，不能用粗模型估值替换真实事件边界。
- 当前默认TP实例等大；以后支持异构时另验证分片慢者和score归一化，不能沿用含糊的“组总量/单rank时间”。

### 12.8 astra-sim-joint

当前die分支已有_JointDiePlanProvider、DiePlanDescriptor、NativeProjectionContext/World和独立候选执行状态，不另建第二套Python模拟器。

具体升级：

1. online/sh30_online_scheduler.py的provider用新公共计划及估值。
2. joint/joint_cost_model.py读取新资源工作量；不从runtime_ns反推内存byte。
3. joint/action_work_graph.py继续建立候选的网络、restore、recompute和计算依赖。
4. astra-sim/workload/execution_driven/NativeProjection.cc改用相同独立活动ResourceArbiter；现有RuntimeLeg的单timer/memory动作不足，需升级活动/credit状态。
5. 实跑DieExecution和投影都使用相同计划、资源域、公平算法、首访和取消规则。
6. 候选world从冻结输入加私有overlay创建；不可变Plan可共享，活动Execution不可共享。
7. 增加live runtime export→v2 snapshot→validated import/fork路径，保存各L/C/S进度、credit、fairness、依赖/共享端点/计数/舍入。现有NativeProjectionContext immutable输入和reset重建不能当中途快照。
8. 已知未来到达/释放事件按原投影窗口回放；未知未来负载不虚构。慢动作在已知后续事件到来时仍参与资源竞争。
9. provider旧hbm_share_divisor只在独立legacy兼容profile使用。新域从实际活跃集合分配服务，不再把整段PE/Vec/NoC乘一个HBM共享除数。
10. 原在线修正因子如保留，必须说明作用的具体服务率、单位及适用域；不改真实byte，不给copy/remote-read人为追加动作专属倍率。

本设计统一性选项固定为：coarse默认在线修正因子=1；如启用，必须冻结到shared_service_profile并由实跑与投影共同使用，进入快照和指纹。当前JOINT仅在预测侧缩放service_units的包装器不得继续挂到coarse计划上。更新共同profile时先结算旧服务，后续使用同版本；已发生的物理计数不回写。

若保留prediction-only修正，只能进入明确的legacy/approximate实验，并另报factor与disclosure；不适用本设计“相同已知world下投影=实跑”的严格验收，不与coarse一致性结果混列。

JOINT的recompute是前向历史token重放以重建KV；代码中的batch_train/TrainPlan是调度批次，不是反向训练。本设计覆盖现有推理与KV重建，不因为保留这些名称就宣称支持backward/optimizer。

## 13. 运行复杂度与计算资源预算

### 13.1 冷编译与固定上限

Core资源查表与几何O(1)。GEMM最多128个模板候选，Attention最多16基本组合×两种固定方案；融合FFN最多128次容量试探，物化FFN调用固定四个子kernel而不枚举笛卡尔积。每候选至多32个类别、16个epoch/primitive及3条独立活动；复合region最多64个epoch。

总工作约O(候选×固定类别/epoch×kernel/输入span)。causal和Core归属使用闭式/floor_sum（可O(log dimension)），不允许O(token²)、O(Core×tile)、逐wave数组。巨大repeat_count只增整数/浮点运算成本，不增事件/状态；不得用逐K展开、运行时周期搜索或精确周期证明作为实现该上限的生产路线。独立span数是实际输入成本，同shape/位置规则可聚合repeat；不能强行改成最大shape。

### 13.2 热路径

R≤13，J定义为当前活跃primitive/外部端点逻辑owner数，每owner同时最多一个primitive、各最多3个活动，故A≤3J；复合region/PlanBundle子kernel串行，不另增公平份额。credit每owner常数个，状态量O(A*R+J)，与Core/PE/tile数无关。advance_to只推进活跃集。仲裁只在外部集合/容量、活动结束、epoch或credit边界更新；稳定流体稳态通过切向约束连续积分，不做每wave开关。

v2引入带符号credit约束，不能沿用v1的O(R²J+J log J)承诺。采用稀疏active-set，矩阵维度仅取决于J/R和常数credit；固定数值迭代预算初值256，超限报numerical_failure，不以逐tile fallback补结果。实现后报告实际迭代分布、最坏J与耗时；若常态触及上限，修求解器/表示，不调精度证书追平。

每epoch内部活动/credit/描述符切换有固定上限，须在维度×repeat扩大测试中证明event_count不跟tile数线性增长。只预约最早本地事件，与原外部事件合并；取消及时回收代号/句柄，不积无限墓碑。稳定资源下事件量O固定epoch；动态外部到达/取消本来会增加事件，按其数量单独报告。

### 13.3 与原Roofline比较的开销验收

速度与精度从第一版同时测，不等待全部结构实现后再补性能。除固定事件/活跃规模下的Core/tile/K/repeat放大测试，还记录实际描述符/曲线折点、内部事件、候选估值、LP更新及JSON字节和缓存命中/未命中次数。计时区分单次仲裁更新、整次执行、孤立查询与完整world预测；不得用总耗时除以更新数冒充单次更新P95。

规范化缓存键统一由C++语义生成，保留shape、精度、TP/span、驻留/就绪等有意义的输入，只剔除已证明不影响计算的追踪标签。命中查询不重新枚举候选或构造完整诊断报告；轻量孤立估值可复用同一公共Runtime在相同HW/profile/revision下已得结果。活跃world查询必须保留clock/generation/owner/overlay身份，不能误用孤立缓存。完整旧接口、指标及版本化兼容继续保留。

以下为初始工程目标，尚非测量结果。指定CPU/编译器、Release、日志关闭和缓存状态，重复测量p50/p95及内存，保存命令/代码身份。

| 项目 | 初始目标 |
|---|---|
| 常规缓存命中成本查询 | P95≤20μs，不枚举候选/解析大JSON |
| 常规冷编译 | P95≤2ms；复合kernel与大span单列 |
| J≤32、R≤13仲裁更新 | P95≤50μs；同时报LP迭代/credit切换 |
| 稳定profile的内部事件 | 有限描述符上限，不随Core/tile/repeat增大 |
| 计划缓存 | ≤4096条且≤64MiB按实际字节淘汰 |
| 五仓同workload端到端运行 | 相对旧Roofline CPU墙钟增加≤10%作为初始项目预算；各仓/候选强度分别报告 |

开销报告同时列legacy Roofline、旧mesh_die_v1及coarse_v2：cold/hot ns与倍率、总host CPU time/wall time、峰值RSS、缓存命中、事件/仲裁次数、候选world数。测量固定输入图/调度事件序列以隔离后端开销，再另跑完整在线策略看延迟变化造成的图数量差异；不能把两种实验混成一个倍率。真实硬件模型输出的延迟与host仿真耗时分开。

精度目标为10%，有多次测试/合理校准记录仍难达到时20%以内可按§1.6接受；同时报告busy绝对误差、参考共享时长和覆盖范围，不以平均占用率乘积或事后修正统计替代原busy定义。速度预算不因精度目标变化而放宽。

最低验证粒度5→8→16的精度/事件/CPU曲线同时报告，选满足误差与开销门槛的最小者。若没有共同满足者，标tradeoff_unresolved并改聚合表示/批量缓存；不能删除SRAM或互连约束，也不能宣称“最粗粒度已验证”。多个入口情景、离散oracle和拟合均离线，不增加默认热路径成本。

公共库不自动多线程；原事件机制决定调度。可并行独立候选world/用户提供的配置批次，单rank同一事件序列确定。

## 14. 验收：覆盖、数学和五仓库语义

### 14.1 原文资源黄金检查

1. 64行N_pe/N_vec与源文逐项一致，INT4/FP32分别核算。
2. B=256、0.25MiB小核：L0=0.40662mm、N_vec=2、N_pe=40；每核PE=80GOPS、Vec=7.9108等效GFLOPS。
3. 同一Core四档B不重新求L0；g(B)=0.4、0.6、1.0、1.8。
4. Nx×Ny矩形不被改成floor(sqrt(C))方阵。
5. n=1/2/3的HBM颗数=2/4/6，容量=72/144/216GB，峰值=2355.2/4710.4/7065.6GB/s。
6. 每轴0.10mm是总外围预留，不能变成0.20mm。
7. HBM超宽采用max外廓；源文两个摆放的模块外廓相同。
8. bump扣理论HBM后得D2D双向/单向预算，不用实际被限速吞吐替换扣除项。
9. 10%模块间互连按除sqrt(0.9)扩外廓；不再从Core面积扣一次。
10. 所有floor前使用未四舍五入值；Wafer系统模块数不自动改rank_count。

### 14.2 自变量响应矩阵

| 变化 | 必须观察到的响应 | 不要求的响应 |
|---|---|---|
| small↔large | 源表资源/几何、每核有效容量、并行与局部供数重新核算 | 每个算子都单调加速 |
| SRAM档位 | 资源表、tile/驻留阈值、重读/中间访问可能改变 | 未跨阈值也强制改变时间 |
| B | source层N_pe/N_vec与面积必重算；timing层在固定单拍/route profile下重算local/Mesh，入口按所选profile | “B越大一定快”或任意PHY随B变化 |
| Nx、Ny | 核数、尾波、方向截面/映射、最小外形变 | 同核数只换形状必然总时间不同 |
| n | 容量、HBM排峰值、模块/bump/系统预算变 | 入口饱和后仍线性加速 |
| HBM摆放 | 外廓保持源文等价；入口及方向归属按假设变 | 任意固定优劣倍率 |
| W/H留白、pitch、ex/ey | 原文几何与系统预算变 | 相同资源/网格下凭空提升算力 |
| 用户给定W/H与可选曝光/长宽比约束 | 当前实例合法性及派生预算；非法时明确拒绝 | 自动改变面积/网格或生成替代硬件 |
| 无search_profile/Wafer字段 | 正常解析完整的给定单Die配置 | 自动搜索或必须先铺排Wafer |
| 系统区域/Mx/My/10%占位 | 模块数、整片容量/峰值与面积指标变 | 单die执行器生成整片事件 |

分别验证source-required静态响应与profile-conditional时间响应。SRAM命中向量/GEMM固定计算率、构造LOCAL瓶颈再扫B；HBM-only端点扫n；薄长网格及转置网格构造Mesh瓶颈。服务率应按profile变化，端到端时间只在对应域受限时必须改变。普通算子不响应时报告容量/工作/瓶颈，不能强加惩罚因子。

### 14.3 执行、动态竞争与粒度验收

必须实际调用编译后的C++库/C ABI/事件路径，保存输出；设计手算仅用于预期值。离散oracle覆盖同样基本资源假设，但独立实现就绪/credit更新。

- 两个100B owner共享100B/s各2s；1B与100B小者先结束；同owner拆同路由片段不增加总权重。
- eta=.5、D=100B：HBM基本服务200，入口仍100；不存在方向不能有非零工作。
- W=1/2/大repeat的操作/byte守恒；C1=5,C2=1,L2=1,S1=10其它0的两波耗时15s，不能提前S1得到11s。
- C=100s、L=50B、HBM1B/s，t60到达40B端点：外端点t100完成，不残留单phi的t120。L完成后资源/计数停止；busy/共享区间按实际活动率积分。
- 一类纯计算、一类纯HBM且先后依赖：阻塞HBM不能阻塞已经就绪的纯计算；runtime不得收费theta合成工作。
- 长repeat的load不能超预取credit、store不能读尚未生成的输出；输入满/空及输出满/空的速率可行，同质稳态没有每wave事件。
- OS/WR的循环、消费wave、active cores、FP32/INT32累加与转换量；WR省读取但减并行；相同输入规范化plan hash一致。
- FFN融合重算/无中间HBM与物化四kernel的中间读写/workspace分别精确；候选不可行才报unsupported，不能与合法物化方案冲突。
- Attention一个q任务多KV块仅最终输出store；GQA重复读取；causal完整跳块/部分矩形/绝对位置与score slots守恒。
- 奇数INT4尾块、最小SRAM候选、row余数、零工作、overflow；取消不收费未服务部分，写未完整不ready。
- 同tick完成/credit/取消/到达、正间隔ceil、generation、一次terminal/GPU释放；credit阶段结束与外部restore到达同tick。

按8.6多phase-offset和全部架构轴扫描，并报告最小通过E、每请求时间/busy轨迹误差、反转/undetermined与开销；全部工作量守恒是硬约束。合成oracle通过不等于真实设备准确。

### 14.4 校准四层验收

| 检查 | 真正运行/比对内容 | 合格的结论 |
|---|---|---|
| XLSX cache | B2:B14、B15硬总计、B16公式/缓存逐单元格与来源 | xlsx_cache_reconcile；数值误差<1e−6μs，非预测精度 |
| TX8 reference | 独立算13项及attention/FFN/整组；稳定源key AV/PV映射一次，BLOCK_TOTAL不重复相加 | tx8_reference_reproduction；公式误差<1e−6μs |
| 新粗库实际执行 | compile→estimate/execute，检查plan、活动/credit、byte、接口和14.3闭式 | coarse_library_execution；整数计数精确、纯闭式tick差≤1ns。对oracle按8.6误差预算另判 |
| 独立目标留出 | 真实元数据完整的未拟合测量与新库输出 | heldout_hardware_validation；当前not_available，按预先登记精度域/阈值报告 |

246μs与条件3363μs各自报告身份/解释；用这两个点反推的reference参数没有独立验证集。不能要求不同执行/融合/量化/流水的目标coarse逐组等于Excel硬编码值。超shape/精度/设备/结构证书域均extrapolated；改变B/SRAM/Nx/Ny/n后必须重新编译目标而非查参考延迟。

### 14.5 五仓兼容、回放与成本语义

每仓都需以下真实运行证据，not_run不能写pass：

1. 原业务/KV/LRU/network/pool/first-token exact-proxy回归；端点唯一收费、workspace准入、cancel/drain/在线回收与一次GPU释放。
2. 同descriptor/task下C++与C ABI的canonical plan hash/summary、每rank独占成本、工作/容量一致；full-detail全部旧record/key类型、gate、精确计数和窗口按10.4对拍，DIE served进入正确HBM分子。
3. 输出per-rank物理模型成本、TP组图critical path、selector score与实际完成四项，不做二次TP除法。FACE旧Python组合带approximate composition，局部同源与整图严格oracle各自验证。
4. 在t导出分数进度快照→导入/独立fork两份，回放同一(event_tick,priority,producer,sequence,kind,owner/generation)流。比较每事件的L/C/S progress、credit、fairness、真实/等效计数、舍入余量、callback顺序和terminal时间；同模型内部对拍允许数值容差，纳秒终结必须一致。
5. 包含中途共享rate变化、同tick完成/取消/到达、stale token、fractional byte/credit、外部网络/pool join和未知未来。冻结排序版本，原JOINT现有time/priority/producer/sequence/kind/task排序与新的本地域顺序由适配器统一，不各写一套优先级。
6. 导入错误原子拒绝：旧scalar活跃状态、schema/指纹不符、ID重复、非有限/负工作、credit/容量/依赖不一致；不能半导入后继续给结果。
7. 新真实竞争只影响经过的活动域，已完成L不持续占HBM；纯计算不乘hbm_share_divisor，实跑与投影共享同一frozen服务profile。
8. 空闲快照与独占执行一致；相同已知world的投影与真实推进一致。这个一致性不等于对未知未来或真实芯片的准确预测。

逐仓报告schema/公共源hash/ABI/template/calibration/topology/granularity版本与运行命令；五仓共用同一库，不接受只通过某一仓的总block时间。

## 15. 实施顺序与完成标准

| 步骤 | 交付 | 进入下一步的条件 |
|---|---|---|
| 1 | 源变量schema、Core64、resolver与显式结构profile | 全部源变量有消费者；几何/峰值黄金一致；未知入口不伪称事实 |
| 2 | 确定候选/循环、动作与缓冲编译器 | dtype/重算/causal/GQA/尾块/驻留/byte精确与hash稳定 |
| 3 | 独立L/C/S、credit、仲裁与v2 ABI | 动态反例、公平、依赖/取消/释放、数值失败语义通过 |
| 4 | 独立离散oracle、最小E证书、入口联合敏感性 | 粒度误差/排序/事件数量通过，结构未验证状态保留 |
| 5 | TX8参考fixture导入/公式重建 | 原数据身份、唯一Excel公式、两条件参数及reference误差分别报告 |
| 6 | wscllm实跑/指标，再接WS-LRU/FACE/FACE-LRU | 原业务与10.4全部指标、restore/pool/历史KV/候选语义通过 |
| 7 | JOINT live snapshot/import/native回放 | 每活动/credit/fairness/事件/计数/取消与真实world一致 |
| 8 | 五仓开销/全变量/排序报告 | CPU/RSS/事件/缓存与Roofline对照；最小粒度满足工程预算 |
| 9 | 有独立数据时新库拟合与留出验证 | 可识别参数与真实验证适用域升级；数据缺失保持not_available |

工程版本完成需1–8真实运行通过：源变量全部进入正确机制、有限明确假设估计、最低已验证聚合粒度、五仓兼容和同源共享、开销可复核。仅能复算Excel不算工程版本完成。第9的数据缺失不靠合成数据填上；工程完成可用于**明确假设的架构实验**，不能宣称目标性能已实测校准或物理排序已证实。

本次交付是设计修订。已独立核对新增的路由矩、GEMM/融合FFN计数、causal有效对数与数值示例（4200项小尺寸枚举/算术比较），以及JSON/章节/fence一致性；这些仅为文档数学核验。新公共库/适配/离散oracle尚未实现并运行；入口/目标测量证据仍不足。因此软件验收状态not_run，独立目标硬件验证not_available，配置排序稳健性not_tested；这些结论不会因为写进验收表而变成pass。

## 16. 输入示例

以下是用户给定单Die配置的示例，不声称最优；无需search_profile或Wafer字段。explicit输入迁移尚待代码实现，数据类型以更新后的schema定义为准。

~~~json
{
  "schema": "coarse_die_v2",
  "source_revision": "9fdcad1f8a1640d3e91558e3b2d11c55c66b53c8",
  "architecture": {
    "core_type": "large",
    "sram_mib_per_core": 0.25,
    "noc_width_bits": 256,
    "core_columns_nx": 8,
    "core_rows_ny": 11,
    "core_pitch_x_mm": 0,
    "core_pitch_y_mm": 0,
    "die_extra_width_mm": 0.10,
    "die_extra_height_mm": 0.10,
    "die_geometry": "explicit",
    "die_width_mm": 21,
    "die_height_mm": 29,
    "hbm_stacks_per_row": 2,
    "hbm_placement_mode": "two_sides_one_row_each"
  },
  "performance_profile": {
    "execution_profile": "int4_matrix_vec_fp32_assumed_v1",
    "calibration_status": "assumed",
    "timing_topology_profile": "distributed_ideal_v2",
    "routing_profile": "mapped_uniform_sources_v2",
    "pipeline_profile": "pipeline_fluid_v2",
    "granularity_validation": "not_run",
    "physical_topology_validation": "unverified",
    "eta_pe": 0.90,
    "eta_vec": 0.90,
    "eta_hbm_matrix": 0.70,
    "eta_hbm_vector": 0.70,
    "eta_sram": 1.0,
    "eta_local": 0.70,
    "eta_mesh": 0.70,
    "eta_inject": 0.70,
    "rho_sram": 0.90,
    "launch_ns_per_macro": 1000,
    "memory_setup_ns_per_macro": 0,
    "kappa_quant_eqops": 4,
    "kappa_acc_convert_eqops": 2
  }
}
~~~

resolver按源公式求L0及最小容纳宽高，只检查给定21×29mm能否容纳8×11网格；不改输入，不查找更优尺寸。面积为609mm²，得到88核、4栈、144GB及4710.4GB/s理论峰值；HBM行宽22.11mm、模块诊断外廓约22.41×51.56mm。没有系统区域输入就不派生Wafer模块铺排。L0和中间计算保留未舍入精度，不能用显示的2.56355mm近似值执行floor。

示例显式选distributed_ideal_v2探索情景，不增加无证据的入口限速，仍受HBM、Mesh、SRAM与LOCAL约束；最终时间不能只按4710.4GB/s计算。若改选edge_single_word_v2，北/南各179.2GB/s、合计358.4GB/s，比HBM理论峰值低约13.14倍。这只证明原入口假设很可能主导该情景，不能证明实际芯片只有358.4GB/s；入口/路由联合敏感性必须另跑并报告配置排序是否反转。

## 17. 最小粒度的边界与可选扩展

默认表示保留全部源文资源/几何机制；动态/排序准确度通过本版的oracle、敏感性和独立数据分别确认。有限credit与独立活动不是逐tile仿真。以下只有现有聚合表示无法在预算内覆盖研究问题时才增加：

- 指定地址偏斜/单栈热点：从两排均匀条带升级为稀疏栈/控制器组。
- 指定位置热点：从中央截面/总hop升级为少量关键边资源，而非立即逐flit。
- 同die多kernel的PE/Vec独立并发：从COMPUTE_GATE升级为分离服务域。
- 精确tile交错或router credit：不属于当前解析生产方案；只能在用户明确另立研究目标后讨论独立扩展，不能作为本项目默认接续路线或失败fallback。
- 线长/PHY、DVFS、热反馈：补证据与输入机制后建立扩展profile。

扩展需另记granularity_profile、误差证书与开销；不能静默增加默认调用成本或共用未区分缓存。最多16段仍失败先检视表示/结构，不能为了通过就盲目提高tile粒度。

## 18. 依据和代码定位

所有硬件公式以[架构建模新.md](https://github.com/cs-sunH/template/blob/9fdcad1f8a1640d3e91558e3b2d11c55c66b53c8/架构建模新.md)为准：§1为Core64与分精度峰值，§2为动态Die/HBM/bump，§3为系统模块占位。该源文的性能缺项由本文显式假设补齐。

校准数据为[TX8_单die建模_精简版.xlsx](https://github.com/cs-sunH/template/blob/9fdcad1f8a1640d3e91558e3b2d11c55c66b53c8/TX8_单die建模_精简版.xlsx)，参考其总览、硬件配置、维度定义、计时组13行及FFN工作集。

历史背景：[旧die设计](https://github.com/cs-sunH/template/blob/9fdcad1f8a1640d3e91558e3b2d11c55c66b53c8/die内仿真器设计文档_动态建模版.md)、[旧架构详解](https://github.com/cs-sunH/template/blob/9fdcad1f8a1640d3e91558e3b2d11c55c66b53c8/晶圆级芯片die内及die间架构与建模详解_动态建模版.md)。不沿用两文档冲突的逐流/阶段公平与同tick规则。

代码基线的关键位置：

| 仓库/路径 | 核查意义 |
|---|---|
| 五仓extern/die_model/src/Hardware.cc | 旧硬件schema、方阵及16GB/410GB/s常量 |
| 五仓astra-sim/workload/Workload.cc | 真正COMP分发、die/legacy区分和外部MEM/COMM端点 |
| 五仓astra-sim/workload/LocalHbmBandwidthModel.* | 旧聚合仲裁与已服务计数 |
| 五仓astra-sim/workload/MetricCollector.cc及metrics_postprocess.py | full-detail record、兼容计数/窗口、DIE利用率与HBM分子 |
| FACE两仓sh_test_mesh/workload/llama2_7b_inference/face_scheduler.py | 候选估值、TP分片和score归一化 |
| JOINT同路径online/sh30_online_scheduler.py | _JointDiePlanProvider装配 |
| JOINT同路径joint/joint_cost_model.py、action_work_graph.py | 候选依赖、成本和recompute |
| JOINTastra-sim/workload/execution_driven/NativeProjection.cc | 已存在的独立native投影world |

路径用于当前基线实施定位，不把旧函数行号当作新设计的接口定义。本文是待实现规范；验收项和性能预算尚未在新公共库中执行。




