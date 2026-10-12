# die 内仿真器设计与五仓库实现规范（HBM3E / 动态 Die 几何）

本规范定义 `mesh_die_hbm3e_dynamic_v4` 动态矩形 Die 目标硬件的资源模型、die 内计算与访存执行、五仓库适配接口、配置派生、动态事件约束及验收标准。默认 `die_geometry_mode=dynamic_core_grid`，由上层传入实际二维 Core 网格及 Die W/H；`tx8_reference_v1` 是独立参考 profile。文档内的接口与算法为仓库实现所应满足的规范，不代表未经测试的实现已具备这些接口。

**硬件与参考输入**：《架构建模新.docx》§1.1–1.3 提供 SRAM/Vec/PE/NoC/Core 资源表，§2.1–2.4 提供动态矩形计算 Die、Micron HBM3E 双排模块及 Core-first/Wafer-first 几何，§3 提供 Wafer 与 D2D 一阶预算；`TX8最终查看版(1).xlsx` 用于独立的 TX8 数值参考。HBM3E 动态主模型允许 HBM 排宽大于逻辑 Die 宽度并扩大**模块**外廓，同时标注该版图尚未物理验证。200–800 mm² 是上层 DSE 的工程搜索窗口，不是 die 内仿真器固定输入清单。

**模型适用边界**：HBM 的 1177.6 GB/s/颗为根据 1024 bit × 9.2 GT/s 推导的十进制理论峰值，不是实际持续带宽；Core/NoC、间距、bump、效率及二维 Wafer 模型均含工程假设。HBM→NoC 的有效端口/链路参数与 BF16/SFU 有效服务率缺失时，允许输出可验证的静态资源和工作量，但必须拒绝生成可信目标时长、在线策略成本或目标数值执行结果。

总体执行路径为：以真实本地张量形状生成算子工作量，按 Core SRAM 容量进行分块与映射，形成 PE、Vec、片内 NoC、本地 HBM 阶段，并交给 Astra-sim 事件时钟与共享本地内存仲裁器运行。既有 Graph/watch 依赖、KV 所有权、LRU/pool/restore 语义、跨 rank D2D 通信图及调度策略保持独立。

## 1. 总体架构与设计边界

### 1.1 两种硬件执行配置，共用计算工作量定义

| 配置 | 用途 | 时间如何产生 |
|---|---|---|
| `tx8_reference_v1` | 独立复现 `TX8最终查看版(1).xlsx` 的 13 个计时组 | 用表中形状、完整 score 外存路径和已选系数计算资源预算；采用聚合流式近似 |
| `mesh_die_hbm3e_dynamic_v4` | 64 行 Core、四档 B，搭配 Micron HBM3E 36 GB×2n、由上层生成的任意合法矩形 `Nx×Ny` Core 网格和 W/H 模块；面向五仓库及扫参 | 先验证精度执行能力；参数充分时生成 tile/资源计划并进入共享本地 HBM 仲裁，否则只输出资源/工作量及明确的拒绝状态 |

目标硬件架构资料只提供 **PE INT4 GOPS** 和 **Vec FP32 理想等效 GFLOPS**，没有给出 BF16 PE 峰值及 SFU 有效吞吐。对 BF16 LLM 图，在缺少与精度相容的有效服务率时，**不能按目标硬件输出可信目标时长或作为五仓库策略的 compute oracle**；先按 §3.3 的显式错误契约返回资源检查/待校准状态。

TX8 不是所设计 core mesh 的一个已知实例；不能给它编造核数和 NoC 拓扑。目标硬件的执行时间由其资源、实际工作量及校准参数计算，不能直接引用 TX8 计时。

两种配置共用算子语义、形状检查和工作量定义，**但硬件峰值与效率来源必须隔离**；TX8 的 BF16 抽象算力及拟合效率只用于参考 profile。参考配置使用 Excel 对应的理想流式时间包络；目标配置包括首块加载、尾块写回、有限缓冲和 NoC。**不设置逐算子延迟查表，也不对目标配置按算子乘修正系数强行追平 Excel。**

Excel 的合理拆分值在本项目中是固定的工程参考标准。其中 RMSNorm 的 246 μs 来自原记录；3363 μs 的 FFN 解释是采用的假设；其余修正值是派生数据。复现这个标准可以验收参考实现，但不能将这种一致性写成独立硬件验证误差。实现不等待更多资料。

### 1.2 组件职责与集成边界

| 部分 | 决策 |
|---|---|
| die 内算力与物理字节预算 | 统一由 `DieCostModel` 和 `DieExecution` 生成 |
| Python 中影响决策的本地计算成本预测 | 调用同一具备 dtype 能力检查的库；没有相容性能参数时拒绝目标 compute oracle 数值调用，不允许用不同精度的峰值代算 |
| die 内 core mesh | 使用 Core 表的固定 L₀、每核 PE/Vec 资源与四档 NoC，只覆盖一个 rank 内部 |
| rank 之间二维 mesh、collective、P2P、路由和拥塞 | 继续使用现有 Astra-sim 网络后端 |
| 调度器、准入、PD、TP 分组、rank 选择、KV 所有权、LRU 和迁移策略 | 按已有策略规则执行；时序与容量对运行时决策的影响由事件模拟产生 |
| GraphSource 依赖释放、watch、generation、commit、节点回收 | 遵守既定所有权和生命周期 |
| 本地 HBM/DDR 带宽竞争 | 由共享本地内存仲裁器处理计算侧与通信侧阶段，保持各仓库的端点语义 |
| 请求、模型、TP 分片和 KV 长度 | 取实际图生成输入，不统一改成 TX8 的模型尺寸 |

仿真器遵循仓库的策略规则、通信行为、外部依赖与状态生命周期；完成时刻、排队顺序和动态决策由完整资源模型及事件状态共同确定。

## 2. TX8 参考标准：可以直接实现的公式

### 2.1 固定参考输入与系数

```text
B=1, Sq=Sk=1024, H=4096, F=14336
Hq=32, Hkv=8, d=128, J=Hkv*d=1024, M=B*Sq=1024
layers=1, TP=1, BF16 activations/weights, FP32 score/statistics/accumulators
full_score_materialization=true, skip_fully_masked_tiles=false
matrix_peak=128e12 FLOP/s, vector_peak=1e12 eq-op/s
memory_peak=200e9 byte/s, memory_capacity=64e9 byte, SRAM=48e6 byte
extra_launch=0, extra_memory_latency=0, inferred_NoC_cost=0
```

此处 causal mask 可以存在，但仍计算完整 score 矩形，不能用三角形工作量去比对 Excel 的稠密矩形预算。FMA 计 2 FLOP；MB、GB 采用十进制。

系数仅由以下公式生成一次，保留 double 精度：

```text
eta_compute = (6*M*H*F / 128e12 + 16*M*F / 1e12) / 0.003363
            = 0.9079551923877489
P = 128e12 * eta_compute = 116.2182646256319e12 FLOP/s
V = 1e12   * eta_compute = 0.9079551923877489e12 eq-op/s
BW_matrix = 200e9 * 0.70 = 140e9 byte/s
BW_vector = (4*M*H + 2*H) / 0.000246
          = 68.23336585365854e9 byte/s
eta_memory_vector = BW_vector / 200e9 = 0.3411668292682927
```

`4*M*H+2*H=16785408 bytes` 包括一次 BF16 输入、输出和 gamma，不漏 gamma，也不另行重复加一份。

### 2.2 用工作量生成时间，不读取 Excel 的 time 列执行

对参考组定义矩阵量 `Fop`、向量量 `Eop`、实际内存流量 `D`：

```text
C = Fop/P + Eop/V
L = D/BW_class
T_group = max(C,L)
```

计算路径中的 PE/vec 工作按既定先后次序记预算；外存采用聚合流式重叠。此公式没有逐 tile 的填充/排空项，是参考配置明确保留的近似，不能将它描述为 TX8 实际微架构执行重建。

参考执行器将每组编译为两个同时开始的分支：`PE(Fop)→vec(Eop)` 计算预算链，以及一份内存流；二者完成后组才完成。它复现上述包络公式。目标 profile 禁止用这种整组包络替代有加载、依赖和缓冲约束的实际阶段。

令 `A=B*Hq*Sq*Sk`。下表足以复算整个参考 block：

| 组 | Fop | Eop | D：实际 byte | 内存类别 |
|---|---:|---:|---|---|
| RMSNorm | 0 | `6*M*H` | `4*M*H+2*H` | vector |
| Q / O projection，各一次 | `2*M*H*H` | 0 | `2*(M*H+H*H+M*H)` | matrix |
| K / V projection，各一次 | `2*M*H*J` | 0 | `2*(M*H+H*J+M*J)` | matrix |
| RoPE_QK | 0 | `3*M*(H+J)` | `4*M*(H+J)+4*Sq*d` | vector |
| QKᵀ | `2*A*d` | 0 | `2*B*Hq*Sq*d+2*B*Hkv*Sk*d+4*A` | matrix |
| scale/mask/softmax | 0 | `12*A` | `8*A` | vector |
| PV | `2*A*d` | `A` | `4*A+2*B*Hkv*Sk*d+2*B*Hq*Sq*d` | matrix |
| residual add | 0 | `M*H` | `6*M*H` | vector |
| fused FFN | `6*M*H*F` | `16*M*F` | `6*H*F+4*M*H` | matrix |

RoPE 的 cos、sin 各读一次；表中的式子适用于参考样本相同的 token 位置。PV 的 `Eop=A` 是按 tile 将 FP32 概率转 BF16 的固定工程成本；它在参考样本上不改变内存主导的结果。FFN 的中间激活在 SRAM 中，因此无需额外计入独立 SiLU/Hadamard 的 DRAM 流量。

### 2.3 必须复现的结果

| 顺序 | 唯一计时组 | 参考 μs |
|---:|---|---:|
| 1 | RMSNorm_1 | 246.000000 |
| 2 | Q_proj | 359.511771 |
| 3 | K_proj | 134.816914 |
| 4 | V_proj | 134.816914 |
| 5 | RoPE_QK | 315.033675 |
| 6 | QKT | 1033.596343 |
| 7 | score_mul_softmax | 3934.079063 |
| 8 | AV | 1033.596343 |
| 9 | O_proj | 359.511771 |
| 10 | res_add_1 | 368.819912 |
| 11 | RMSNorm_2 | 246.000000 |
| 12 | FFN | 3363.000000 |
| 13 | res_add_2 | 368.819912 |
| — | attention 核心，2–9 | 7304.962795 |
| — | block，1–13 | **11897.602619** |

FFN 参考流量为 `369098752 bytes`，外存服务预算 `2636.419657 μs`，小于计算预算 `3363 μs`。按计算资源分配得到 gate/up/down 各 `1034.769230769 μs`、SiLU `242.524038462 μs`、gate_mul `16.168269231 μs`。这些是**组内归账**，不是五个独立 kernel 的额外延迟。

参考容量检查采用 Excel 的 `F_tile=512`：输入、FP32 输出累加器、三块权重和三份激活合计 `40894464 bytes`，小于 `0.9*48e6=43200000 bytes`。这个单工作集检查不证明有完整双缓冲；参考时间中的理想流式重叠仍是一项模型假设。

验收读取工作簿 `计时组13行`，使用**原始机器键**（尤其第 8 组严格为 `AV`；`PV` 只是算法解释别名），按名称而不是行号匹配，不把 `BLOCK_TOTAL` 再加入求和。测试比较来自公式计算的 **double 未舍入结果**与工作簿原始单元格数值；上述表仅为六位小数的**展示**，不能用表中显示值重新累加作为 `1e-6 μs` 的验收输入。参考原始 `BLOCK_TOTAL≈11897.60261936833 μs`，成本核每组和合计的绝对误差都要求 **< 1e-6 μs**。事件时钟再单独检查每次向 ns 上取整的误差，不能以百分比容忍计数错误。

## 3. 目标 die 的硬件输入与资源生成

### 3.1 Core64 资源与动态 Die 网格：实例主键与验证

**数据来源与实例输入**：《架构建模新.docx》§1.1–1.3 提供 64 行 SRAM/Vec/PE/NoC/Core 资源表，§2–3 提供动态 Die、HBM 模块和 Wafer 几何。每核配置由 `8 SRAM × 2 核型 × 4 NoC=64` 行确定。DSE 生成 `Nx,Ny,W_die,H_die,n` 等参数，并通过 `resolved_die_model.json` 向仿真器交付一个经过选择的硬件实例。执行器负责字段校验、资源计算与本地事件调度；候选搜索由上层完成。

**硬件主键分层**：

- **Core 表行**：`(core_table_revision,core_table_sha256,core_type,sram_mib_per_core,noc_width_bits)`；包含 64 条记录和对应精度定义。`Core L₀` 对相同 `(core_type,SRAM)` 的四档 B 不变。
- **Die 实体网格**：`(core_columns_nx,core_rows_ny,die_width_mm,die_height_mm,core_pitch_x_mm,core_pitch_y_mm,die_extra_width_mm,die_extra_height_mm)`，允许 `Nx≠Ny`、W≠H。`compute_area_mm2=W_die×H_die` 是数值派生输出，不得作为固定枚举输入。
- **HBM 模块**：`(hbm_stacks_per_row=n,hbm_placement_mode,hbm_geometry_revision,row_fit_policy)`，正整数 n，`N_HBM=2n`；动态主模型允许 `row_width>W_die`，自动扩大**模块**外廓，标识 `hbm_row_overhang/physical_layout_unverified`。`row_fit_policy` 默认 `allow_module_overhang`；`required` 只针对明确指定的限制性实验。
- **完整硬件身份**：`(hardware_schema_version,Core64 hash,Die 网格+精确 W/H,HBM 双排物理参数,端口/共享域/路由 ID,模型/校准修订)` 的规范 JSON SHA-256。改变 W/H 或 Nx/Ny，即使面积相等，也必须得到不同的硬件指纹。

**工程 DSE 默认过滤器**（不属于 die 内性能核的硬编码取值集合）：`200≤A_die≤800mm²`、150–200 作为下界敏感性、26×33 mm 曝光场可旋转、长宽比≤2、轴向**总**预留 ex=ey=0.10 mm、Core pitch=0、HBM 每排 n=1..14、Bump-D2D 非负。架构搜索按 Core-first/Wafer-first 生候选，再把单条 `resolved` descriptor 交给仿真器。若在其它明确搜索 profile 下使用另一尺寸，die 内仿真器依据明确 W/H 和合法网格判定，不由面积三档白名单直接拒绝。

**Core 尺寸检查**：`W_min=Nx×L₀+(Nx−1)×px+ex`、`H_min=Ny×L₀+(Ny−1)×py+ey`，输入 `W_die≥W_min,H_die≥H_min` 且 Nx/Ny≥1；`N_core=Nx×Ny`。多余外缘留白不自动长出 Core；若不足返回 `invalid_hardware/core_grid_exceeds_die`。整片峰值等于该实际网格核数乘每核精度相容的名义峰值；只有新增 Nx/Ny 才改变 Core 数。每核 PE/Vec 数以表中整数为准，不以 Die 面积直接除以估算 Core 面积。

**配置验证与缓存**：`mesh_die_hbm3e_dynamic_v4` 要求显式 `Nx,Ny,W_die,H_die` 及 HBM 双排物理配置；只给面积数值无法唯一确定整数网格和逻辑 Die 外形，返回 `migration_required` 或对应的 Schema 校验状态。每份资源、映射和性能配置采用完整硬件指纹，不与其他硬件或参考 profile 共用缓存。源 Word 的几何、PE/Vec 峰值为工程估计，BF16 性能须另行校准。

**部件级参数与工艺映射（非新增扫参维度）**：SRAM 来自 NVSim 固定 wordwidth=1024 bits 的 **22 nm** 尺寸，并按 SRAM 位单元面积比、保持利用率的近似假设映射到 **7 nm**。完整的来源尺寸如下；容量为二进制 MiB，与 HBM 的十进制 GB 区分。

| SRAM 容量 | 22 nm 长 (mm) | 22 nm 宽 (mm) | 7 nm 长 a (mm) | 7 nm 宽 b (mm) |
|---|---:|---:|---:|---:|
| 0.25 MiB | 0.65782 | 0.22528 | 0.40662 | 0.13925 |
| 0.50 MiB | 1.31564 | 0.22528 | 0.81324 | 0.13925 |
| 0.75 MiB | 1.97345 | 0.22528 | 1.21986 | 0.13925 |
| 1.00 MiB | 2.63127 | 0.22528 | 1.62648 | 0.13925 |
| 1.25 MiB | 3.28909 | 0.22528 | 2.03310 | 0.13925 |
| 1.50 MiB | 3.94691 | 0.22528 | 2.43972 | 0.13925 |
| 1.75 MiB | 4.60472 | 0.22528 | 2.84634 | 0.13925 |
| 2.00 MiB | 5.26254 | 0.22528 | 3.25296 | 0.13925 |

- **Vec**：参考 22 nm AraXL **16 lanes 的 `2.4×1.6 mm`** 平面图和 1.4 GHz 下 **44.3 FP64 GFLOPS**；按 1 GHz 与每 64-bit lane 理想并行两路 FP32 FMA 的假设，取 7 nm **每 lane `0.1897×0.1265 mm`**、`3.9554 FP32 等效 GFLOPS`，单 lane 面积 `0.02399705 mm²`。这是等效计算峰值与面积假设，不是 FP32 或特殊函数的实测吞吐。
- **PE**：参考 45 nm NVDLA 的 **16×4 INT4 CMAC**，将 `0.0361 mm²` 均分为 64 个等效位置并假设面积再缩至 `1/15`，得到 **每 PE `A_pe=0.0000376 mm²`**、正方形边长 `a_pe=√A_pe≈0.006131884 mm`。在 7 nm、1 GHz 建模口径下取 **每 PE 每周期 1 次 INT4 MAC=2 INT4 GOPS**，不是 BF16 浮点计算单元的证据。
- **名义计算资源比例**：源 Word 以 **PE INT4 GOPS : Vec FP32 等效 GFLOPS ≈ 10:1** 作为混合精度峰值安装目标，使用 `N_pe=round_half_up(19.777*N_vec)`、0.5 向上取整。下表 `PE/Vec 名义峰值比` 是**源 Word 的印刷比值**；它不表示同一精度的 10:1 加速，也不能用来校准 BF16 或 SFU。



候选集合及无核排除主键见本节开头。**核数只依赖计算 Die 面积和固定 Core L₀，不依赖 HBM 每排颗数和排布面数**；core表数据在该模型中固定。

每核 `L₀` 对同一 `(core_type, SRAM)` 在四档 B 下**固定**；位宽增加只会重新分配该固定 Core 内的 NoC/Vec/PE 预算，**不能**根据位宽放大 Core 边长、减少 die 核数。**执行时以本文下列 64 行（包括最后一列逐区安装信息）为唯一输入资源记录**；`新版架构Core配置64行.csv` 如存在只是同值机器可读副本，不是额外权威源。部署时应从下表按规定列顺序规范化生成新 `core_table_sha256`，校验 64 个主键与布局。若副本冲突则拒绝启动、回查 §3.1 的规范表与来源。§1.2 几何用于重算、校验与测试，不允许简化安装约束。

**Core 64 行资源与逐区整数排布（由来源 §1.3 给出，运行时按此表加载和校验）：**

| B (bit) | 核型 | SRAM (MiB) | Vec lanes | INT4 PE 数 | Vec FP32 等效 GFLOPS/core | PE INT4 GOPS/core | L₀ (μm) | Vec 排布 K×R / PE 列数 | PE/Vec 名义峰值比 |
|---:|---|---:|---:|---:|---:|---:|---:|---|---:|
| 64 | small | 0.25 | 2 | 40 | 7.911 | 80 | 406.62 | 竖2×1/1 | 10.113:1 |
| 64 | small | 0.50 | 15 | 297 | 59.331 | 594 | 813.24 | 竖5×3/3 | 10.012:1 |
| 64 | small | 0.75 | 40 | 791 | 158.216 | 1582 | 1219.86 | 竖8×5/5 | 9.999:1 |
| 64 | small | 1.00 | 77 | 1523 | 304.566 | 3046 | 1626.48 | 竖11×7/7 | 10.001:1 |
| 64 | small | 1.25 | 126 | 2492 | 498.380 | 4984 | 2033.10 | 竖14×9/9 | 10.000:1 |
| 64 | small | 1.50 | 204 | 4035 | 806.902 | 8070 | 2439.72 | 竖17×12/11 | 10.001:1 |
| 64 | small | 1.75 | 280 | 5538 | 1107.512 | 11076 | 2846.34 | 竖20×14/13 | 10.001:1 |
| 64 | small | 2.00 | 368 | 7278 | 1455.587 | 14556 | 3252.96 | 竖23×16/15 | 10.000:1 |
| 64 | large | 0.25 | 243 | 4806 | 961.162 | 9612 | 2563.55 | 下横13×17/14；上竖11×2/4 | 10.000:1 |
| 64 | large | 0.50 | 1037 | 20509 | 4101.750 | 41018 | 5278.58 | 下横27×35/25；上竖23×4/23 | 10.000:1 |
| 64 | large | 0.75 | 2382 | 47109 | 9421.763 | 94218 | 7990.68 | 下横42×53/3；上竖26×6/227 | 10.000:1 |
| 64 | large | 1.00 | 4276 | 84566 | 16913.290 | 169132 | 10702.11 | 下横56×71/12；上横25×12/258 | 10.000:1 |
| 64 | large | 1.25 | 6749 | 133475 | 26694.995 | 266950 | 13413.29 | 下横69×89/52；上横38×16/118 | 10.000:1 |
| 64 | large | 1.50 | 9832 | 194447 | 38889.493 | 388894 | 16124.34 | 下竖126×72/30；上横40×19/326 | 10.000:1 |
| 64 | large | 1.75 | 13422 | 265447 | 53089.379 | 530894 | 18835.31 | 下横97×126/70；上竖80×15/179 | 10.000:1 |
| 64 | large | 2.00 | 17534 | 346770 | 69353.984 | 693540 | 21546.24 | 下竖166×96/89；上竖94×17/155 | 10.000:1 |
| 128 | small | 0.25 | 2 | 40 | 7.911 | 80 | 406.62 | 竖2×1/1 | 10.113:1 |
| 128 | small | 0.50 | 15 | 297 | 59.331 | 594 | 813.24 | 竖5×3/3 | 10.012:1 |
| 128 | small | 0.75 | 40 | 791 | 158.216 | 1582 | 1219.86 | 竖8×5/5 | 9.999:1 |
| 128 | small | 1.00 | 77 | 1523 | 304.566 | 3046 | 1626.48 | 竖11×7/7 | 10.001:1 |
| 128 | small | 1.25 | 126 | 2492 | 498.380 | 4984 | 2033.10 | 竖14×9/9 | 10.000:1 |
| 128 | small | 1.50 | 198 | 3916 | 783.169 | 7832 | 2439.72 | 横11×18/11 | 10.000:1 |
| 128 | small | 1.75 | 273 | 5399 | 1079.824 | 10798 | 2846.34 | 横13×21/13 | 10.000:1 |
| 128 | small | 2.00 | 360 | 7120 | 1423.944 | 14240 | 3252.96 | 横15×24/15 | 10.000:1 |
| 128 | large | 0.25 | 236 | 4667 | 933.474 | 9334 | 2563.55 | 下横13×17/14；上横5×3/2 | 9.999:1 |
| 128 | large | 0.50 | 1005 | 19876 | 3975.177 | 39752 | 5278.58 | 下横27×35/25；上竖15×4/19 | 10.000:1 |
| 128 | large | 0.75 | 2310 | 45685 | 9136.974 | 91370 | 7990.68 | 下横42×53/3；上竖14×6/220 | 10.000:1 |
| 128 | large | 1.00 | 4145 | 81976 | 16395.133 | 163952 | 10702.11 | 下横55×71/43；上竖30×8/70 | 10.000:1 |
| 128 | large | 1.25 | 6534 | 129223 | 25844.584 | 258446 | 13413.29 | 下横70×89/21；上横19×16/278 | 10.000:1 |
| 128 | large | 1.50 | 9524 | 188356 | 37671.230 | 376712 | 16124.34 | 下竖127×72/9；上横20×19/424 | 10.000:1 |
| 128 | large | 1.75 | 12990 | 256903 | 51380.646 | 513806 | 18835.31 | 下竖145×84/80；上竖54×15/105 | 10.000:1 |
| 128 | large | 2.00 | 16978 | 335774 | 67154.781 | 671548 | 21546.24 | 下横112×144/48；上竖50×17/364 | 10.000:1 |
| 256 | small | 0.25 | 2 | 40 | 7.911 | 80 | 406.62 | 竖2×1/1 | 10.113:1 |
| 256 | small | 0.50 | 15 | 297 | 59.331 | 594 | 813.24 | 竖5×3/3 | 10.012:1 |
| 256 | small | 0.75 | 40 | 791 | 158.216 | 1582 | 1219.86 | 横5×8/5 | 9.999:1 |
| 256 | small | 1.00 | 70 | 1384 | 276.878 | 2768 | 1626.48 | 竖10×7/6 | 9.997:1 |
| 256 | small | 1.25 | 117 | 2314 | 462.782 | 4628 | 2033.10 | 竖13×9/8 | 10.000:1 |
| 256 | small | 1.50 | 180 | 3560 | 711.972 | 7120 | 2439.72 | 竖15×12/10 | 10.000:1 |
| 256 | small | 1.75 | 252 | 4984 | 996.761 | 9968 | 2846.34 | 竖18×14/12 | 10.000:1 |
| 256 | small | 2.00 | 336 | 6645 | 1329.014 | 13290 | 3252.96 | 竖21×16/14 | 10.000:1 |
| 256 | large | 0.25 | 221 | 4371 | 874.143 | 8742 | 2563.55 | 横13×17/13 | 10.001:1 |
| 256 | large | 0.50 | 920 | 18195 | 3638.968 | 36390 | 5278.58 | 竖40×23/25 | 10.000:1 |
| 256 | large | 0.75 | 2135 | 42224 | 8444.779 | 84448 | 7990.68 | 竖61×35/39 | 10.000:1 |
| 256 | large | 1.00 | 3854 | 76221 | 15244.112 | 152442 | 10702.11 | 竖82×47/52 | 10.000:1 |
| 256 | large | 1.25 | 6052 | 119690 | 23938.081 | 239380 | 13413.29 | 横68×89/65 | 10.000:1 |
| 256 | large | 1.50 | 8856 | 175145 | 35029.022 | 350290 | 16124.34 | 竖123×72/79 | 10.000:1 |
| 256 | large | 1.75 | 12096 | 239223 | 47844.518 | 478446 | 18835.31 | 竖144×84/92 | 10.000:1 |
| 256 | large | 2.00 | 15840 | 313268 | 62653.536 | 626536 | 21546.24 | 竖165×96/106 | 10.000:1 |
| 512 | small | 0.25 | 2 | 40 | 7.911 | 80 | 406.62 | 横1×2/1 | 10.113:1 |
| 512 | small | 0.50 | 12 | 237 | 47.465 | 474 | 813.24 | 竖4×3/3 | 9.986:1 |
| 512 | small | 0.75 | 32 | 633 | 126.573 | 1266 | 1219.86 | 横4×8/4 | 10.002:1 |
| 512 | small | 1.00 | 56 | 1108 | 221.502 | 2216 | 1626.48 | 竖8×7/5 | 10.004:1 |
| 512 | small | 1.25 | 99 | 1958 | 391.585 | 3916 | 2033.10 | 竖11×9/7 | 10.000:1 |
| 512 | small | 1.50 | 156 | 3085 | 617.042 | 6170 | 2439.72 | 竖13×12/9 | 9.999:1 |
| 512 | small | 1.75 | 210 | 4153 | 830.634 | 8306 | 2846.34 | 竖15×14/10 | 10.000:1 |
| 512 | small | 2.00 | 272 | 5379 | 1075.869 | 10758 | 3252.96 | 竖17×16/11 | 9.999:1 |
| 512 | large | 0.25 | 187 | 3698 | 739.660 | 7396 | 2563.55 | 横11×17/11 | 9.999:1 |
| 512 | large | 0.50 | 805 | 15920 | 3184.097 | 31840 | 5278.58 | 横23×35/22 | 10.000:1 |
| 512 | large | 0.75 | 1855 | 36686 | 7337.267 | 73372 | 7990.68 | 横35×53/34 | 10.000:1 |
| 512 | large | 1.00 | 3290 | 65066 | 13013.266 | 130132 | 10702.11 | 横47×70/44 | 10.000:1 |
| 512 | large | 1.25 | 5192 | 102682 | 20536.437 | 205364 | 13413.29 | 竖88×59/56 | 10.000:1 |
| 512 | large | 1.50 | 7632 | 150938 | 30187.613 | 301876 | 16124.34 | 竖106×72/68 | 10.000:1 |
| 512 | large | 1.75 | 10332 | 204336 | 40867.193 | 408672 | 18835.31 | 竖123×84/79 | 10.000:1 |
| 512 | large | 2.00 | 13536 | 267701 | 53540.294 | 535402 | 21546.24 | 竖141×96/90 | 10.000:1 |

```text
# 新 Word 统一 B0=256 bit，两个核型相同
B ∈ {64,128,256,512};  g(B)=0.20+0.80*(B/256)
A_noc_0=0.15*L0_mm^2
A_noc(B)=A_noc_0*g(B)             # 6%、9%、15%、27% 的固定 Core 面积
L_core(B)=L0_mm                   # 位宽改变不改变 Core 外形
S_compute(B)=L0_mm^2-a_mm*b_mm-A_noc(B)
W_min=Nx*L0_mm+(Nx-1)*core_pitch_x_mm+die_extra_width_mm
H_min=Ny*L0_mm+(Ny-1)*core_pitch_y_mm+die_extra_height_mm
assert Nx>=1 and Ny>=1 and W_die_mm>=W_min and H_die_mm>=H_min
N_core=Nx*Ny
compute_area_mm2=W_die_mm*H_die_mm  # 数值派生，而非面积列表
if N_core==0: invalid_hardware("no_core_fits")
PE_peak_die_int4_gops=N_core*(2*N_pe)
vec_peak_die_fp32_equiv_gflops=N_core*(3.9554*N_vec)
SRAM_installed_bytes_per_core=round(sram_mib_per_core*2^20)   # 档位均精确为整数 byte
SRAM_usable_core_bytes=floor(0.90*SRAM_installed_bytes_per_core)
link_peak_bytes_per_second=B/8*1e9  # 软件近似：1 word/周期/有向链路
```

排布字段格式：`横/竖 K×R/C` 中 `K×R` 为该计算区 Vec 的整数列×行数量，`C` 为该区 PE 整数列数；`下…；上…` 分别是大核的两个计算区。表格是理想几何上界，不包含单元间额外走线/间距，个别原表排布横向余量仅为约 0.0000288 mm（`large/1.75 MiB/B64` 的某计算区）；**不得宣称满足版图制造、布线或时序**。

`a×b` 取 7 nm SRAM 的八档长短边：`a(mm)={0.40662,0.81324,1.21986,1.62648,2.03310,2.43972,2.84634,3.25296}`，`b=0.13925 mm`。小核固定 `L₀=a`，下方计算带 `H=L₀-b, W=L₀-A_noc(B)/H`；大核旋转 SRAM，256-bit 基准 `L₀=[a+sqrt(a²−0.6ab)]/0.3`，`Δw₀=L₀-b, A_noc,0=aΔw₀`：当 `g≤1`，下区 `(L₀,L₀−a)`、上区 `((1−g)Δw₀,a)`；当 `g>1`，下区 `(L₀−(g−1)A_noc_0/(L₀−a), L₀−a)`、无上区。Vec 依区选择横/竖放置并按整数列/行装入，`N_pe=round_half_up(19.777*N_vec)`，全 Core 只取整一次；在全部容量/矩形/PE 列约束成立时最大化 `N_vec`，平局取最小横向剩余面积。执行期优先使用源表，不以简化面积比替代列级约束。

`N_core=Nx*Ny` 是输入已解析的整数二维安装数量；外缘预留及 Core pitch 通过 W_min/H_min 明确验算，不能当作额外 Core 面积。`A_noc(B)` 已把 Core 内互连与 Die 内 Core mesh 的等效面积统一分摊，不能按两个独立 NoC 再扣一次。`0.90` SRAM 系数、链路持续速率和 XY 路由属于仿真器工程假设，不是新 Word 实测参数。

**Core 64 行规范校验值**：仅取下方原表 64 条数据行（不含表头/分隔行），各行每个 Markdown 单元格首尾去空格，以 `|` 联结 10 列，并以 UTF-8 `\n` 串接每行、末尾再加 `\n`；该规范化数据的 SHA-256 为 `371330f3382014215853b81fcbf29a599d66645717169970e8527d46cb8680f8`。生成候选/缓存时统一采用这个表字段身份，并额外绑定动态 v4 的全部 HBM 参数。

**动态 HBM 候选**由实际 `Nx,Ny,W/H,n,placement_mode` 定义。举例：`large/0.25MiB`、`Nx=8,Ny=11`、`n=2` 时即使 `row_width=22.11 mm` 大于逻辑 Die 宽度约 20.6084 mm，也应形成增宽的 module；Core mesh 是 `8×11`，不是面积 floor 的正方网格。NoC B 增加不改变该 Core L₀ 和 `8×11` 安装数量，但每核 PE/Vec 可变。目标 HBM3E 与 TX8 的 200 GB/s、64 GB 参考资源完全隔离。

### 3.2 HBM3E 产品参数、动态模块、容量、注入和矩形片内 NoC

**HBM SKU 参数**：Micron HBM3E **36 GB 12-Hi** `MT65B18G16120A00QH-92:A`，接口 `1024 bit × 9.2 GT/s`、理论带宽 `1177.6 GB/s/颗`、容量 `36_000_000_000 byte/颗`，HBM 封装 `10.98×10.98×0.78 mm`（厚度不进二维占位），HBM/Die/边沿间隙取 `0.15 mm`。所列峰值非持续有效带宽。`n=hbm_stacks_per_row≥1`；两排共 `2n` 颗，同侧两排或两侧各一排。两种排布几何同值而端口/路径竞争未必同值。

**动态资源派生（从 resolved 单 Die W/H 出发）**：

```text
Nx,Ny=core_columns_nx,core_rows_ny
W_min=Nx*L0_mm+(Nx-1)*core_pitch_x_mm+die_extra_width_mm
H_min=Ny*L0_mm+(Ny-1)*core_pitch_y_mm+die_extra_height_mm
require Nx>=1 and Ny>=1 and W_die_mm>=W_min and H_die_mm>=H_min
A_die_mm2=W_die_mm*H_die_mm
n=hbm_stacks_per_row                       # DSE 默认枚举 1..14
N_HBM=2*n
row_width_mm=10.98*n+0.15*(n-1)
W_module_mm=max(W_die_mm,row_width_mm)+0.30
H_module_mm=H_die_mm+22.56
hbm_row_overhang=(row_width_mm>W_die_mm)
local_mem_capacity_bytes=N_HBM*36_000_000_000
hbm_peak_Bps_theoretical=N_HBM*1_177_600_000_000
```

`hbm_row_overhang=true` 时记录 `physical_layout_unverified`；**该标记用于保留未经物理验证的可计算布局候选**。物理设计要求 HBM PHY、供电、边沿互连、时序及热路径单独确认。若针对特定封装要求排宽必须落在 Die 内，可设置独立 `row_fit_policy=required` 且相应记录为限制性筛选，不能冒充动态模型默认条件。实际候选数由 DSE 的整数网格、面积边界扩展、HBM 数量和筛选器共同决定。

**具体资源示例（不是仿真器固定档）**：`large/0.25MiB,L₀=2.56355 mm,Nx=8,Ny=11,px=py=0,ex=ey=0.10mm` 时 W≈20.6084,H≈28.29905,A≈583.2mm²；n=2 得排宽 22.11mm、模块约 `22.41×50.85905mm`、HBM 144GB、理论总峰值 4710.4GB/s。DSE 的另一解如 `Nx=7,Ny=11` 可能产生另一面积及系统模块数；**两者均只作为输入具体硬件描述符交给一个 die 内仿真器**。

**容量/条带/驻留契约**：rank 对应一个计算 Die 和其 `2n` 颗 HBM。上层 KV/权重管理器继续负责实际驻留、保留、LRU、remote/pool/restore、释放；独立 die 测试使用当前实际驻留 `resident_weights_bytes + resident_KV_bytes + temporary_workspace_bytes <= N_HBM*36e9`，每一项是**真实物理字节**，不是 `service_units`。本地 KV 单层表达式仍为 `4*B_batch*Hkv*Sk*d` bytes（BF16 K 和 V），完整 BF16 block 权重近似仍为 `2*(2H²+2HJ+3HF)` bytes；TP、压缩格式、层数、局部 KV 分片、跨 rank 副本必须按真实实例修正。对每颗 HBM 的容量/地址映射可用显式 `stripe_policy`：`uniform_round_robin_bytes_v1` 在同一可条带化区段按颗粒编号分配真实 byte 的整数商/余数；也允许 `explicit_regions_v1` 保存每颗物理区段/所有者。实例化时验证 `Σ_j bytes_j=actual_bytes`、`occupied_bytes_j<=36e9`，并保证恢复/写入和随后读为**不同动作**，不能改原协议或偷偷将 remote 改成 local。均匀默认只是无物理地址资料时的工程示例，不是厂商内存控制器映射。

**HBM→Core Mesh 注入不是免费边沿**。定义并要求显式实现如下资源表及绑定：

| 字段 | 约束/定义 | 未给实测参数时的处理 |
|---|---|---|
| `hbm_injection_map_id` | 精确标识 port 节点/边、mapping、速率、读写方向/争用域的不可变版本（建议 hash） | `TO_BE_DEFINED`，**不准假装已验证** |
| `hbm_injection_ports` | 每端口唯一 `port_id`、边沿 `north/south`、mesh 合法坐标、方向/共享域、`ingress_Bps_per_port` | 未提供时为 `null`，不赋无限带宽 |
| `stack_to_ports` / `port_to_core_mapping` | 每颗 HBM `[0,2n)` 到**确实存在的**端口和进入的 Core 路径映射，权重和为 1 | 未绑定则拒绝定量 mesh/HBM 时长 |
| `hbm_peak_Bps_per_stack` | 每颗 `1_177_600_000_000`（芯片理论上限，十进制） | 可用于上界报告，**不能作为片内链路速率** |
| `noc_width_bits` / `noc_link_eta` | B∈{64,128,256,512}、1 GHz、`eta_noc=0.70` | 都是片内 NoC 软件近似，非 Micron 数据 |
| `ingress_Bps_per_port` / `effective_mem_bw` | 端口的总吞吐与每颗/每端口/每有向链路约束相交；以给定访问图计算，而非仅单一产品峰值 | 若缺端口速率/路由，输出 `missing_ingress_calibration` 及理论上界，不能给精确服务时间 |

对每个合法 `(n, mode)`，若有**明确提供的**端口配置，建议生成 `stack_id`、`physical_row`（0/1）、`row_index`（0..n−1）三元组：`same_side_two_rows` 的两排都沿 `north` 边注入，第二排必须与第一排**竞争可用 north 端口**而非生成额外 `south` 链接；`two_sides_one_row_each` 的两排分别从 `north/south` 边注入。边沿 port 须在 `0..Nx−1` (x) / `0..Ny−1` (y) 的实际 Core 坐标内；若给定候选 port 列，则选距目的 Core group 最近的合法端口，同距离取较小列号，多个 HBM 映射相同端口则共享该端口预算。不能默认每颗有独占可达 1177.6 GB/s 的入口、不能因二维面积相同而复制相同的北南路由。 `mesh_columns=Nx,mesh_rows=Ny`；Nx/Ny 须均≥1，且端口坐标处于 `0≤x<Nx,0≤y<Ny`。映射是经配置的设计/校准对象，非现有 Word 已验证的真实端口实物图。

每条片内**有向链路**一拍传送 B bit 的近似原始速率为 `B/8*1e9 B/s`；乘 `eta_noc=0.70` 后依 B64/128/256/512 得 **5.6/11.2/22.4/44.8 GB/s**。**单颗** HBM 要传完理论峰值，需要并行相同有效链路至少 `ceil(1177.6/link_effective_GBps)=211/106/53/27` 条（只是不考虑其他拥塞的理论必要条件，**不是已存在的端口/链路数量**）。每个阶段按实际物理字节 `D_edge` 统计 XY 单播、共享广播有向边并集、部分和/attention 状态归约/布局转换；链接下界 `t_NoC=max_edge(D_edge/(0.70*B/8*1e9))`。广播每个边只记一次但重复 wave 重新加载必须重新收费。HBM 入口总带宽 `≤min(Σ_j 1177.6e9,Σ_port ingress_Bps_per_port,可达的片内 min-cut/拥塞上界)`；有偏斜时不能用全体芯片聚合峰值掩盖某一颗/某一端口过载。HBM 到 Core 的真实传输需同时占用对应 HBM、入口与片内路径；阶段完成须等待其关联资源释放，禁止和 NoC 重复计虚假的独立数据副本。片内 NoC 与 Astra-sim **跨 rank D2D** 是不同层，绝不在 die 内发一个同量 D2D 消息。

### 3.3 精度能力与 Die/Wafer/D2D 层级边界

`PE_peak_int4_gops` 是 `INT4 MAC×2` 操作/周期的**整数名义**峰值，`vec_peak_fp32_equiv_gflops` 是 AraXL FP64 基准经 1 GHz/理想双 FP32 SIMD 换算的**FP32 等效名义**峰值；二者不同精度，不可相加。BF16×BF16→FP32 的 GEMM/QK/PV/FFN 不能除以 INT4 GOPS，RMSNorm/softmax/SiLU 的 `eq-op` 不能任意除以向量 FP32 FMA 峰值。硬件来源未提供目标 PE BF16、Vec SFU 或 12-Hi HBM 持续吞吐/真实首访延迟测量。`tx8_reference_v1` 的 0.907955.../0.70/0.341... 不能用来补出这些能力。

**统一有效服务率字段及语义**：目标算子包含 `operation_dtype, accumulator_dtype, op_count_unit, supported_precisions, calibration_revision, performance_model_provenance`；对于需要的计算资源只接受 `pe_effective_bf16_flops_per_second_per_core`（每 Core 的**校准后** BF16 FLOP/s）和 `vec_effective_eq_ops_per_second_per_core`（每 Core、指定 SFU/op 域的**校准后**等效操作/s）。均为正且有限的 B/s 之外的**计算**服务率，已含相应效率，不得再乘 `eta_bf16` 或另一层 PE/Vec 折减。若资料是原始峰值加效率，须在版本化校准导入器先计算 `effective=raw_peak×eta`，运行时只存/消费 effective；原始峰值/η 仅作 provenance，禁止同时拿 effective 和 η 再相乘。不同 dtype、累加精度、特殊函数 mix 需要独立能力及适用域；没有合法证据不得由 FP32 等效峰值或 INT4 GOPS 推造。

校验计算单元/累计精度/转换与 SFU 算法的匹配及速率来源：不支持则 `unsupported_precision`；需要的 BF16/Vec 有效服务率、校准版本或适用域缺失则 `missing_performance_calibration`；入口/端口/路径定量计算缺参数则 `missing_ingress_calibration`。上述阻断影响**可信目标时长、C++ 实际 target 执行、Python 策略 oracle**，但不妨碍输出有据可查的 Core/Die/芯片理论资源和字节预算。目标 Schema 仅接受定义明确的有效服务率字段；`pe_peak_bf16_flops_per_core`、`effective_vec_eq_ops_per_second`、`vec_eq_ops_per_second_per_core` 不属于可接受字段。外部输入转换须显式标注 raw/effective 含义和校准来源。

**统一动态几何、Wafer 与名义 D2D 预算**。逻辑 Die 的 `W_die,H_die` 由显式整数 Core 网格及外围预留得到，也可在 DSE 选择边界扩展方案后增大；实际核数依然是 `Nx*Ny`。模块把 HBM 实体、间隙和 Die 间等效 NoC 与逻辑 Die **分层**统计。《架构建模新.docx》§2–3 的参数化公式如下：

```text
A_die_mm2=W_die_mm*H_die_mm
n=hbm_stacks_per_row; N_HBM=2*n
row_width_mm=10.98*n+(n-1)*0.15
W_module_mm=max(W_die_mm,row_width_mm)+0.30
H_module_mm=H_die_mm+22.56
A_module_mm2=W_module_mm*H_module_mm
A_eff_mm2=A_module_mm2/0.9                  # Die 间 NoC 占扩展后模块总占位面积10%
A_inter_die_noc_mm2=A_eff_mm2-A_module_mm2
W_eff_mm=W_module_mm/sqrt(0.9)
H_eff_mm=H_module_mm/sqrt(0.9)
wafer_cols=floor(215/W_eff_mm); wafer_rows=floor(215/H_eff_mm)
N_wafer_modules=wafer_cols*wafer_rows
B_bump_GBps=(A_die_mm2/(0.055*0.055))*0.4*2.5/8
B_D2D_bidirectional_GBps=B_bump_GBps-1177.6*N_HBM
require B_D2D_bidirectional_GBps>=0
B_D2D_oneway_GBps=B_D2D_bidirectional_GBps/2
```

`215×215mm` 是假设的**方形可铺排系统区域**，不是圆形制造晶圆的 Die 良率模型；bump 通信 `55 μm/40%/2.5 Gbps` 是工程预算，不是实测 PHY。单 Die 模型输出 Wafer/D2D 为上层资源报告，不会把模块数自动实例化为 `rank_count`，也不把 D2D 消息算进 die 内 NoC。上层 DSE 独立执行 Core-first/Wafer-first 的搜索窗口、曝光场/长宽比筛选。

**跨目标的源 Word 校验示例（不是输入清单）**：`large/0.25MiB` 计算优先代表点 `7×11 Core/Die,约18.04×28.30mm`，`11×4=44 模块`，全区域 **3388 Core / 3168GB HBM**；同核型容量优先候选 `8×4 Core/Die,约20.61×10.35mm`、`9×6=54 模块`，全区域 **1728 Core / 7776GB HBM**。二者体现不同目标下的资源取舍，均用于给定 DSE 工程约束的数值复核。

示例完整可解析目标硬件描述符（端口图尚待工程补齐，不可据此直接输出 HBM→NoC 的时间）：

```yaml
# 已解析的一个独立 Die 实例（resource_only）；数值仅作源 Word 动态几何示例
hardware_schema_version: mesh_die_hbm3e_dynamic_v4
compute_cost_model: die
execution_intent: resource_only
die_geometry_mode: dynamic_core_grid
core_type: large
sram_mib_per_core: '0.25'
noc_width_bits: 256
core_columns_nx: 8
core_rows_ny: 11
core_pitch_x_mm: '0'
core_pitch_y_mm: '0'
die_extra_width_mm: '0.10'
die_extra_height_mm: '0.10'
die_width_mm: '20.6084'
die_height_mm: '28.29905'
# 派生: compute_area_mm2=583.19814202; N_core=88
hbm_sku: MT65B18G16120A00QH-92:A
hbm_generation: HBM3E
hbm_capacity_bytes_per_stack: 36000000000
hbm_io_data_rate_gts: '9.2'
hbm_data_width_bits: 1024
hbm_peak_bytes_per_second_per_stack: 1177600000000
hbm_package_width_mm: '10.98'
hbm_package_length_mm: '10.98'
hbm_package_height_mm: '0.78'
hbm_stacks_per_row: 2
hbm_placement_mode: two_sides_one_row_each
hbm_stack_count: 4
row_fit_policy: allow_module_overhang
hbm_geometry_revision: physical_two_rows_dynamic_v1
# 派生: hbm_row_overhang=true; W_module_mm=22.41; H_module_mm=50.85905
core_table_revision: arch_model_8_core64_printed_v1
core_table_sha256: 371330f3382014215853b81fcbf29a599d66645717169970e8527d46cb8680f8
model_algorithm_revision: die_cost_hbm3e_dynamic_v1
calibration_revision: UNCALIBRATED
hbm_injection_map_id: TO_BE_DEFINED
hbm_injection_ports: null
stack_to_ports: null
port_to_core_mapping: null
shared_domains: null
pe_effective_bf16_flops_per_second_per_core: null
vec_effective_eq_ops_per_second_per_core: null
memory_service_model: physical_multiresource_v1
injection_map_validation: unverified
performance_model_provenance: null
ingress_map_provenance: null
supported_precisions: []
accumulator_dtype: null
op_domain: null
eta_hbm_matrix: null
eta_hbm_vector: null
local_mem_latency_ns: null
```

**参数校验时**不接受字段模糊别名：字段 `hbm_stacks_per_row` 与 `hbm_placement_mode` 须显式给出；不识别未声明的 `hbm_per_side/hbm_sides` 别名。输入必须满足 `N_HBM=2n` 以及每排物理坐标和模块几何校验。target profile 未给 inlet 路由/能力时仍可做配置、面积、容量和名义峰值静态分析；调用目标 BF16 数值估计/执行时按上述缺参错误阻断。

## 4. 算子输入：必须在形状尚未丢失时传下来

### 4.1 类型化算子描述与输入字段

`num_ops` 与 `tensor_size` 两个汇总量无法区分 GEMM/GEMV、GQA/MHA、两个短序列与一个长序列、融合中间态以及多个层的权重。算子维度必须由类型化输入提供，不依赖节点名称或汇总操作量反推完整 shape。

版本化 `DieOpSpec` 作为 ComputeAttrs 的类型化内容：

| 字段 | 内容 |
|---|---|
| `schema_version`, `hardware_schema_version`, `op_kind` | spec 及目标硬件版本；norm/qkv_rope/attention/o_proj/ffn/residual/logits/local_region 等 |
| `phase`, `model_dims`, `operation_dtype`, `accumulator_dtype`, `op_count_unit` | prefill/decode；H/F/头/d/vocab/MLP；矩阵 BF16/INT4、累加 FP32、工作量单位各自显式记录 |
| `spans` | 每个实际请求/片段的 q_tokens、kv_tokens、query 绝对位置、有效 KV 范围 |
| `pass_groups`, `layer_ids` / `layer_count` | 哪些 spans 属于同一次物理 batch，哪些独立 pass/层重复执行 |
| `weight_passes` | 沿用图生成器已有的真实权重加载批次，不能默认每个 span 重读一次 |
| `tp_slice` | 当前 rank 的 Q/KV 头索引、FFN/vocab 分片范围、输入输出布局、部分和身份 |
| `kv_access` | 当前操作可读取的 KV 范围、local/staged/remote_ready 身份，以及前驱准备范围 |
| `region_ops`, `fusion_policy` | 本地数学节点和依赖；允许融合的边界；实际计算量来自这里 |
| `provenance`, `config_hash`, `capability_id` | 形状来源、64 行表指纹、硬件/精度吞吐能力/算法版本；用于报表和缓存 |

`rank/request_id/stage/generation` 使用上层的外部字段，由计算资源模型只读消费。`num_ops/tensor_size` 可供非 die 计时路径与统计使用，die backend 的物理工作量以 `DieOpSpec` 计算且只计一次。

### 4.2 spans、TP 与工作量的确定规则

1. 同一个物理 batch 的线性层可以合并 M，复用一套权重；不同 pass、不同层的权重和执行次数不得因此消失。重复完全同形工作可用 repeat 压缩。
2. Attention 按每个 span 的 `q_i*k_i` 求和，不用 `sum(q_i)*sum(k_i)`。例如两个独立 `q=1,k=1024` 请求是两份 attention，不是一个长度 2048 的序列。
3. QKV 的实际三个 GEMM 是 `[M,H]×[H,Hq_local*d]`、`[M,H]×[H,Hkv_local*d]` 两次。GQA 的 KV 按真实头索引复用/复制，不能统一除以 TP。
4. O projection 为 `[M,Hq_local*d]×[Hq_local*d,H]`；FFN 为本地 F 分片上的 gate/up/down。跨 die 的部分和由上层 TP collective 合并；本模型只计算 die 内部分和。
5. 非整除 TP 使用现有 `shard_extent()` 分片和实际 rank 范围。空分片是合法的零工作；有实际访存的零 FLOP 操作不是零工作。
6. MHA、GELU、LayerNorm 等操作依据工作负载所提供的真实图与形状执行；参考样本的 GQA/SwiGLU 参数不应用于覆盖工作负载。GQA 元数据取自模型配置；只定义一个头数的模型按 MHA 语义处理。
7. Prefill chunk 的 causal 判断必须带绝对 query/KV 位置；只有完全落在未来区的 tile 才跳过。Decode 不把 `M=1` 填充成 1024 来收计算费。
8. 聚合图可能将 spans×多层按算子类别合并。模型遵从输入图的聚合口径和 collective 数量，不生成图外的逐层分布式执行节点；报告中记录 `graph_aggregation`，说明系统层仍有这一近似。

Norm/residual 输入按 `tp_slice` 和真实图提供的 H 布局处理。若 hidden 维跨 Die 分片且 norm 需要统计归约，上层图必须提供相应通信节点；不能把跨 Die 统计操作计为纯本地 Norm。

## 5. 目标 die：分块、工作量与驻留算法

### 5.1 固定候选与选择规则

```text
token/query tiles:       {1,8,16,32,64,128}
output/FFN tiles:        {32,64,128,256,512}
reduction/KV tiles:      {32,64,128,256,512}
core group rows/cols:    {1,2,4,8,Ny} / {1,2,4,8,Nx}，各自裁剪为 <=Ny / <=Nx
buffer contexts:        {1,2}
K-parallel partitions:  {1,2,4,8}，限制为本组容量与实际工作允许
```

候选裁剪到实际维度，补入小于最小候选的尺寸并去重；尾块按真实尺寸计算。`Nx,Ny` 分别是 Core mesh 列数/行数，**不是** §3.2 `n=hbm_stacks_per_row`；如网格 `8×11`，应保留 8 列与 11 行的非方形整边候选。core group 在 mesh 中按行优先放置，不相交，边缘放不下的核可空闲。候选必须检查每核活跃缓冲，不能以全 die SRAM 代替每核约束。

**分离资源枚举与有时长的编译**：`enumerate_resources(spec,hardware)` 不依赖 BF16/SFU/ingress 性能校准，确定性返回所有合法候选、实际 shape、SRAM/流量/NoC 资源和 `timing_status=unavailable`；它**不得选出声称最快的 timed plan**。`compile_timed(spec,hardware,capability)` 必须先通过 dtype 和物理入口校验，用 §6 唯一的无外部竞争执行内核给全部合法候选估时，按 `(time_seconds, sram_peak_bytes, physical_bytes, canonical_tile_tuple)` 字典序选取唯一最优映射。此顺序的第一项保持未舍入 double，余项为精确整数/元组；含 NaN/∞ 的候选无效。缺能力时 `compile_timed` 返回结构化阻断状态（无 plan、无 `estimated_seconds`），**不能**以最低 SRAM/字节的候选偷偷冒充最短时间。执行器只能接收 `TimedDiePlan`；资源报告使用独立 `ResourceEnumeration`。

实际运行固定选定映射，遇到 DMA 竞争只更新阶段推进，不在中途改变算法或 KV 策略。

相同 tile 的并行执行汇总成 wave，core 上实际分配量与尾波次必须保留。重复 wave 使用 repeat 描述和惰性生成；不为每个元素、PE、core 或 flit 建事件。计划可以缓存，正在执行的资源状态不能缓存重用。

### 5.2 GEMM

BF16 tile `[m,k]×[k,n]`，FP32 accumulator，双缓冲输入和权重。以下首先是**工作量/容量规则**；生成数值 PE 事件前必须通过 §3.3 的 BF16 物理峰值检查，INT4 名义操作率不可用于此 GEMM：

`S_gemm=4mk+4kn+4mn bytes`。

放不下时尝试单缓冲 `2mk+2kn+4mn`；再放不下则缩小 tile。每 tile 矩阵量 `2mkn`，最终 BF16 输出为 `2mn`。未给出 PE 阵列内部形状，不额外假设 128×128 阵列并收取未知 padding。

实现两种明确的循环：

- **输出驻留**：固定输出 tile/wave，依次遍历 K；累加器留 SRAM，到最后才写 BF16 输出。输入和权重依实际重载次数计费，有共同消费者时使用片内广播。
- **权重驻留**：固定 N/K 权重 tile，遍历 M wave 复用权重。如果全部输出累加器放不下，跨 K 时需要读写 FP32 partial C；首个 K tile 不读取先前部分和，末 K 写 BF16，其余按实际 4-byte partial 计入外存。不能同时免费获得全部权重复用与全部输出驻留。

K 跨 core 分片时，每核生成 FP32 partial C；通过片内归约合并，计入 `4mn` 级消息及每个额外 partial 的 `mn` 次向量加。它与跨 rank all-reduce 是不同层次，后者继续留给 Astra-sim。

### 5.3 向量操作

沿行或连续元素分块。Norm 优先整行 H 驻留，输入读一次、gamma 按驻留规则加载、输出写一次；跨行复用 gamma 需有实际缓冲/广播。无法容纳整行时用两遍流式算法，多算一次输入读取和必要归约。

下列是 **SFU/向量算法的 eq-op 权重**，不等同于源 Word 的 FP32 FMA 峰值；新目标缺少 `vec_eq_ops/s` 时不得直接把 `3.9554*N_vec` GFLOPS 当作分母。

固定向量等效工作权重：RMSNorm 6/元素、LayerNorm 8/元素、RoPE 3/旋转元素、residual/gate_mul 1/元素、SiLU 15/元素、GELU 15/元素、scale/mask/softmax 12/score、类型转换 1/元素。LayerNorm/GELU 是为现有图保留的工程近似，不是新增实测锚点。

RoPE 只旋转新 Q 和新 K，历史 KV 不重复旋转；参考 profile 的整段 prefill 是二者长度相同的特例。cos/sin 按实际访问的位置集合加载，并计片内广播，不能在不同执行上下文间无条件共享。

### 5.4 融合 FFN

沿 token m 和 FFN f 分块，执行 gate/up → SiLU×up → down，并保留本 core 的完整 H 输出部分和。一个上下文的保守预算为：

`S_ffn=6mH+12mf+4*max(kf,fn) bytes`。

其中输入 `2mH`、FP32 输出部分和 `4mH`；`12mf` 覆盖 gate/up 累加及可复用的激活缓冲；最后一项是当前权重 tile 的双缓冲。k 是 gate/up 的 H reduction 块，n 是 down 的 H 输出块。

矩阵总量 `6MHF_local` **BF16 FLOP（按浮点 FMA=2 工作量）**，向量量 `16MF_local` **eq-op**。只有具备与两种单位相容的服务率后才能形成 FFN 数值时间。F 分布到多个 core 时归约各 token 的 H 维 FP32 部分和；分布到多个 die 时只输出本 die partial，由已有 collective 处理。

上述数量针对 SwiGLU。GELU FFN 使用 up→GELU→down，两次 GEMM 的矩阵量 `4MHF_local`、激活量 `15MF_local`、权重字节 `4HF_local`；不额外生成 gate 分支和门控乘。其分块仍使用相同容量检查，按实际活跃缓冲分配。

双上下文必须满足两份缓冲容量，不能凭总 SRAM 充足就假设可流水。无可行融合映射则退回分离算子，激活按图实际物化，增加外存流量；回退结果不应被强制修正为 3363 μs。

### 5.5 片上 attention

目标默认采用 tile score/probability 留 SRAM 的执行。以一个 query 头的 q 行、k 个 KV token 为基本单元，K/V 保持紧凑头数，通过 GQA head-group 的广播复用。

为使缓冲规则可直接实现，采用比原说明更保守的显式双输出状态预算：

`S_attn=2qd+8kd+4qk+8qd+16q bytes`。

分别为 Q、双缓冲 K/V、FP32 score、两份 FP32 输出部分和、两份 `(m,l)` 状态。两份累加器支持 softmax 状态合并，避免依赖未经定义的原位 GEMM 累加能力。两个上下文需 `2*S_attn`，均受每核 `floor(0.90*MiB*2^20)` 字节预算约束。这里的 `4qkd` 是 BF16 矩阵工作量，而不是 INT4 整数操作。

每 tile 顺序为加载 → QK → scale/mask/局部 softmax 状态 → probability 转 BF16 → PV → 状态合并。局部概率使用未归一化的 `exp(score-m)`，PV 生成未归一化输出部分和；不能先对每 tile 除以 l 再直接按下式合并。矩阵量为 `4qkd`，向量基础量为 `13qk`；每次非首块状态合并另计 `q*(3d+22)` 等效操作，最后归一化计 `qd`。22 是两次 exp、标量计算的固定工程权重；它与 SiLU 的权重一样作为固定假设。

合并使用稳定的未归一化状态：

```text
m=max(ma,mb)
l=exp(ma-m)*la + exp(mb-m)*lb
Otilde=exp(ma-m)*Oa + exp(mb-m)*Ob
O=Otilde/l             # 仅最终归一化
```

跨 K 分区/跨 core 合并消息为每 q 行 `4q(d+2) bytes`，使用固定二叉树，每一级的消息和向量合并都收费。不能把各分区独立 softmax 的已归一化输出直接相加。

因果 mask 导致某个局部行完全无有效元素时，用空状态参与合并并跳过其指数缩放，避免 `-inf-(-inf)`；完全被 mask 的 tile 本来就不执行。这里只推算成本，不计算实际张量，但状态和工作量定义必须对应合法的数值算法。

K/V 按实际加载次数收费：同波次、同 KV tile 的共享数据可以广播一次；跨 wave 没有保留缓冲就需重读。所有矩阵量按实际执行矩形计入；跨 causal 对角线的 tile 仍计算整个矩形。两个上下文允许相互流水，同一个 tile 的 QK、softmax、PV 仍严格依赖。

## 6. 时间与事件：共享本地内存仲裁

### 6.1 资源阶段和有效吞吐定义

`tx8_reference_v1` 的 PE/Vec 参考效率和 RMSNorm/DDR 拟合只在 §2 运行；HBM3E dynamic v4 的 `2*N_pe INT4 GOPS`、`3.9554*N_vec FP32 等效 GFLOPS` 是不同精度的**名义**峰值，不能直接代入 BF16/SFU 或等效 eq-op 时间。

目标配置中的 `eta_HBM_matrix=0.70`、`eta_HBM_vector=0.3411668292682927`、`eta_noc=0.70`、90% SRAM 及每逻辑宏组 1 μs 启动均为**工程先验/敏感性参数**；前两项不能冒充 Micron HBM 实测。如果用上述 HBM η，需要逐子流规定独立的 `eta_f`（0<η_f≤1），不能再把 `service_units` 对同一 η 反复膨胀。

合法校准仅允许**已含效率**的计算率：

```text
t_PE  = max_core(F_bf16_core / pe_effective_bf16_flops_per_second_per_core)
t_vec = max_core(E_eq_core / vec_effective_eq_ops_per_second_per_core)
t_mem_chip_peak_lower_bound = D_total/(N_HBM*1.1776e12)
t_mem_stacks_lower_bound = max_s(Σ_{f:s(f)=s} D_f/eta_f / R_s)
t_ingress_lower_bound = max_p(Σ_{f:port(f)=p} D_f / C_p)
t_mesh_lower_bound = max_e(Σ_f m[f,e]*D_f / C_e)
C_e = 0.70*(noc_width_bits/8)*1e9 B/s
```

这里 `R_s` 为每颗 HBM 栈的**服务单位/s**（默认最多理论 `1.1776e12`，不是保证实测），`C_p`/`C_e` 是端口及有向链路的**实际 B/s**。公式只是单阶段下界，不与执行中的其他 stage 简单相加、也不把若干下界的 `max` 当作精确事件时间。`m[f,e]` 是路由编译出的每单位 payload 对该有向边的传输次数；单播 0/1、广播每边只收一次，复制到两个独立分支的边流量另计，但 HBM 源字节只计一次。

每个 tile/wave 维护加载、PE、vec、归约、写回及 buffer 依赖；die 级 PE/Vec/NoC 资源槽按既定资源时序保守串行同类阶段、最多两个上下文。HBM↔core 子流必须**同时**占用其 HBM 栈、入口和路由有向边，在同一个事件求解器推进真实 payload；跨核纯 NoC 流仅占 NoC，但必须与 HBM 流**在相同有向链路资源域内联合限速**，不能另开一份免费片内 NoC。端口映射缺失返回 `missing_ingress_calibration`，计算有效率或适用域缺失返回 `unsupported_precision/missing_performance_calibration`。在缺校准的 target 模式，不产生真实 BF16 数值时间。

每次真实 `(pass_instance,layer_instance,macro_kind)` 产生一个宏组身份，九种宏组名称 Norm1/QKV/RoPE/Attention/O/res1/Norm2/FFN/res2 保留。独立 pass/layer 各付一次 1 μs 工程启动先验；不能因为聚合了多个 layer 就只收一次，也不能按 tile 重收。`tx8_reference_v1` 的启动成本设为零。

### 6.2 内存服务模型、共享限制和确定性事件仲裁

**接口实现要求**：`LocalHbmBandwidthModel` 在共享本地内存仲裁域承接 `issue_compute/COMM/RESTORE/POOL` 和逐流计算阶段 API。必须支持计算侧子流、共享资源、内部回调与取消，不允许建立第二条可免费使用的 HBM 总线，不能把 `D/η` 写回 `tensor_size` 或 KV 字节。

**模式边界（互斥）**：规范字段 `memory_service_model` 只能为 `physical_multiresource_v1`（HBM3E target 所需）或 `aggregate_fallback_sensitivity_v1`（**只允许单独 sensitivity/测试，不准驱动 target compute oracle/在线策略/真实目标时延**）。聚合 fallback 独占时 `U_agg=max(Σ_j U_j,N_HBM*max_j U_j)`、`U_j=D_j/eta_j`、时间 `U_agg/(N_HBM*R_stack)`，属于保守近似；不得在正式逐流计算上**再套**该放大系数、`N_active` 聚合均分或一次整组 `service_units` 速率。两个模式的 `hardware_fingerprint`、统计标签和实验目录必须分开。

**不可变请求与资源表**（概念 C++17，不能理解为当前仓库已存在）：

```cpp
using FlowId = uint64_t;
enum class DieMemoryDirection { READ, WRITE };
struct MemorySubflow {
    FlowId flow_id;                // 整个 DieExecution 唯一，不重用
    uint32_t stack_id;            // [0,N_HBM)
    uint32_t port_id;             // 必须存在于 map
    DieMemoryDirection direction;
    uint64_t actual_bytes;       // 子流从 HBM 读取/写入的一次真实字节
    double eta_hbm;              // (0,1]；时间消耗 u=D/eta
    std::vector<DirectedEdgeUse> route; // (edge_id, multiplicity>=1)
};
// 纯 Core 间归约/广播使用同一边资源域，不含 HBM stack/port；
// 其 eta=1，实际 B/s 与服务单位/s 相同。
struct NoCOnlySubflow {
    FlowId flow_id;
    uint64_t actual_noc_bytes;
    std::vector<DirectedEdgeUse> route;
};
struct DieMemoryStageRequest {
    uint64_t die_execution_id;
    uint64_t owner_macro_id;      // 唯一对应 pass_instance/layer_instance/macro_kind
    uint64_t completion_id;
    uint64_t generation;         // 已预约事件的代号，过期回调不得生效
    bool charge_first_access_latency;
    DieMemoryAccessKind access_kind;
    std::vector<MemorySubflow> flows; // 真实 HBM bytes 唯一归属
    std::vector<NoCOnlySubflow> noc_only_flows; // 仅收费每条 edge，不增加 HBM D
    std::string hardware_fingerprint;
    std::string hbm_injection_map_id;
};
// 回调 (execution_id,completion_id,generation) 仅唤醒内部阶段；
// 不得自行调用 GraphSource::finish_node。
```

每个 `flow_id` 唯一拥有一次 HBM payload；同一 stack 进入两个 port 时拆成 `D_{s,p}` 子流，满足 `Σ_p D_{s,p}=D_s`；所有 stack 的总和 `Σ_f D_f=D_stage`。同一地址从 HBM 读取一次并多播到多个 Core 时，`D_f` 只算一次、广播每条有向 edge 收一次链路份额。真实读写各自计数，不将带宽效率带来的 `U` 视为真实数据。所有 `D_f` 为 uint64 非负、对求和进行 uint128 或溢出守卫；`eta_hbm` 正且有限，所有容量和坐标索引必须校验。`flow_id` 不得跨地址重用或被多个 stage 当作免费同一事务。

**每个调度时刻的严格物理限制**：设传输期活跃子流集合 `F`，每流 HBM 服务率 `u_f`（等效服务单位/s），**真实 payload 速率** `v_f=eta_f*u_f`（B/s），`R_s=1.1776e12` 为理论栈服务上限，`C_p` 端口实际 B/s，`C_e=eta_noc*(B/8)*1e9` 有向链路有效 B/s，`m[f,e]` 是路径 multiplicity。**HBM 子流和纯 NoC 子流使用同一个 allocator**：令 `F=F_HBM∪F_NoC`，对纯 NoC 流定义 `eta_f=1, u_f=v_f`，其 stack/port/PHY 消耗系数为 0、只消耗路径上的有向边；HBM 的 `D_total=Σ_{f∈F_HBM}D_f`，**不计入**纯 NoC payload。每一个资源均在**同一 die 的全部 COMP/COMM_READ/COMM_WRITE/RESTORE/POOL 以及 Core 归约子流**上聚合：

```text
∀s: Σ_{f∈F_HBM,stack(f)=s} u_f                     <= R_s
∀p: Σ_{f∈F_HBM,port(f)=p} v_f                      <= C_p
∀e: Σ_{f∈F} m[f,e]*v_f                              <= C_e
∀q: Σ_{f∈F_HBM, shared_domain(f)=q} v_f              <= C_q  # PHY/读写共用域有声明才启用
u_f>=0; v_f=eta_f*u_f; 0<eta_f<=1
# 若模型声明全 die 原总线聚合域另有 cap：Σ_f u_f<=R_shared;
# 它是显式独立硬件资源，不默认拿聚合峰值再施加 N-way 惩罚。
```

`READ` 与 `WRITE` **默认竞争同一 HBM stack 的 R_s**。同 port 或共享 PHY 的双方向业务也默认共享 `C_p/C_q`，除非真实映射显式提供独立方向物理域和各自上限及来源。现存 `COMM/RESTORE/POOL` 若还不知道 stack/port，必须先扩展兼容入口并绑定显式地址/子流，否则 HBM3E target 运行返回 `missing_ingress_calibration`；无完整资源归属的 job 不得占用虚构的免费或无限端口。若需要纯 NoC 的计算间归约，另建仅边资源子流（无 HBM `actual_bytes`），不能伪装成一颗 HBM 访问。

**可复演的默认公平规则：逐子流等权 max-min progressive filling**。默认所有活跃子流权重 `w_f=1`，可变权重只允许新算法版本。此公平度量是 HBM **服务单位速率 `u_f`**，不等于按 job 个数平分总峰值。按 `FlowId` 升序建立资源消耗矩阵 `a_{r,f}`：HBM 流的 stack 为 1、port 为 `eta_f`、有向边为 `m[f,e]*eta_f`、共享域为 `eta_f`；纯 NoC 流只在有向边上取 `m[f,e]`，其余系数为零。从所有 `u_f=0` 开始、所有流为未冻结；在每一轮，对未冻结流共同增加 `Δ`（单位服务率），取所有资源中 `(capacity_r−Σ_f a_{r,f}u_f) / Σ_{f未冻结}a_{r,f}` 的最小非负值，推进所有未冻结流，然后将触及任一饱和资源的相关流冻结。未冻结且不使用饱和资源的流继续分配剩余容量，直至均冻结。资源剩余负小量按统一容差钳零；非有限、负参数或零速率不可进时间路径。资源和 flow 的平局按资源 `(stack,port,edge,domain,id)`、`FlowId` 的稳定顺序处理；以 IEEE754 double 进行数值过程，日志采用确定的序列化/容差；实现必须输出逐流 `u_f/v_f` 与限制它的资源。对于两条流各使用不同 stack/port/edge，**不应因属于同一 die 而平分对方的独立栈**；同栈的两条流必须共同满足 `Σu_f≤R_s`。这个求解器是建议的软件确定性公平模型，**不是** Micron 物理调度器的实测证明。

**时间推进与取消归账**：每流保存 `D_initial`（整数）、`D_remaining`（double）、`U_remaining=D_remaining/eta_f`（double，eta 在该 flow 生命周期内不变）。只在其真正处于有效传输期时，先按上次求得的 `v_f` 推进 `Δt`：`D_done_f=min(D_remaining_f,v_f*Δt)`，据此减少 `D_remaining_f` 与 `U_remaining_f`；任何 flow 的实际字节**不以其他 flow 的完成比例折算**。日志累计整数物理字节以 `min(D_initial,floor(D_initial−D_remaining+ε))` 的单调规则输出，**全部完成时强制精确 D_initial**，取消时只归账已经实传的部分；中间小数仅为 fluid model 的数值状态，不能产生 `ceil(U)` 的物理字节。统计按 `flow_id` 唯一归账，读写单独记、NoC 仅以 `m[f,e]*实际D_done_f` 计路径传输；不重记已完成阶段。

**完成时间与活跃流的事件求解**：对每个活跃且 `v_f>0` 的子流，在当前资源分配之后取 `t_f=D_remaining_f/v_f`，下一次服务事件只能预约这些正时间的最小值（再与已经存在的 latency、PE、Vec、取消、到达事件比较），到该时刻统一推进所有活跃子流后重新分配；绝不能把一个阶段各层下界分别收费或预先固定所有流的完成时间。对 `v_f=0,D_remaining_f>0` 且无未来解除约束事件的情况，返回 `stalled_resource/no_progress`，不得创建无限的零时回调，也不得把待传的真实字节记为完成。NoC-only 流也参与相同下一事件的计算、路由链路共享和完成计数，但不得计入 `HBM actual_bytes`。动态参考实现与测试如果只检查速率（无 event queue），其结论仅为数学资源分配，不等于此时序接口已经实现。

**同 Tick、回调、预约顺序冻结**：资源变化、到期、提交、取消、延迟→传输切换必须对该 die **统一**执行 `advance_to(tick)`（先用上一份速率推进到 tick，不跨 Tick 重放），再按阶段优先级 `已完成的 flow/stage 终结 → 取消尚未完成对象 → 延迟到期/新请求进入 → 重算份额 → generation++ 并重新预约`，同优先级 `FlowId` 升序。精确在该 tick 已完成的 flow，即使同时收到 cancel，也按完成处理；尚未完成者 cancel 胜。过期 generation 预约只丢弃，不触发业务完成。正剩余完成时间按 `max(1,ceil(seconds*1e9)) ns` 预约；若模型连续推进产生终点舍入误差，裁剪剩余至 0，不能超记真实 D。零字节请求同步完成且不收费首次非零访存延迟。`drain/is_finished` 必须包括 pending/active/delayed stage、所有执行器及预约；故障或取消仅单次向 GraphSource 终结、单次释放原 GPU 槽。

**宏组身份与首访计费**：定义 `owner_macro_id` 为对 `(die_execution_id,pass_instance_id,layer_instance_id,macro_kind)` 的无碰撞局部 ID（不能只 hash macro_kind），由 region 编译时冻结。每实例最多一次工程 launch（目标先验 1 μs）与首个 `D>0` HBM stage 的一次 `local_mem_latency`；同一实例的多个 tile/wave 或独立 flow 不能重复；GraphBatch 聚合后仍保留每个 pass/layer 的 macro 实例清单。macro 生命周期、重试新 execution id、取消回滚与已完成状态均有明确枚举；对应首次付费发生在 stage 真正开始等待 HBM 服务时，不能在纯静态资源枚举中消耗。恢复/POOL 自身的老通信 endpoint 延迟继续按仓库既有合同，不能把 target 的“每个 compute macro 一次”误套给 COMM。

**必备动态反例**：同栈上限 100 U/s 的两条 η=1 流同时各需 100B，等权各 50B/s → 两者 2s 完成；分处不同 stack、端口/edge 均独立的两条各需 100B，必须各 1s 完成（不能聚合 N-way 强行变 2s）；一栈按两个 100B/s port 各分 50B 则入口下界 0.5s，但仍要检查栈总上限；η=0.5 与 η=1 的不同 flow 在相同服务推进 50U 时真实完成分别 25B 与 50B。复杂流的绝对物理时间依赖实际端口/边限速，不能因为这些教学夹具而宣称已得到产品测量。

`tx8_reference_v1` 只在独立的 **200 GB/s DDR 参考仲裁域**复现参考 Norm 独占 246 μs；若参考配置内部另有本地 DDR 使用者与之竞争，才按相同 DDR 模型改变带宽份额。**不同 profile 的 TX8 与 HBM3E target 不在同一条真实 HBM 总线上竞争**。

### 6.3 生命周期与完成回调

```text
外部 COMP 满足依赖且取得指定 GPU-comp 槽
  -> 取得/生成 DiePlan
  -> 内部阶段按依赖和缓冲进入 ready 队列
  -> PE/vec/NoC 通过 Sys 时钟预约完成
  -> 内存阶段交给现有 LocalHbmBandwidthModel
  -> 阶段完成只唤醒内部后继
  -> 整个 DieExecution 完成
  -> 原 Workload 完成路径调用一次 GraphSource::finish_node
  -> 原 watch、统计、资源释放、commit/retire 流程继续
```

内部阶段通过专用回调接口；内部 HBM job 不能伪装成外层 COMP 完成，否则会提前释放依赖/GPU 槽。每个外层节点只计一次 GPU 占用区间和一次 terminal。内部状态持有自己的 ID/owner，不保留已回收 NodeStore 记录的裸引用。

共享内存资源状态变化时，执行推进到当前 Tick、撤销过期预约、generation 防护和重排下一事件。成功完成、异常退出和终止均需回收阶段事件。增加新活动状态后，`is_finished`/drain 检查必须包含尚未结束的 DieExecution，避免服务过早退出。

### 6.4 时间、延迟及开关语义

- 成本核使用 double 秒，工作量和字节用检查溢出的整数；接 Astra-sim 时转 ns，严格使用现有 Tick 口径，不拿 1 GHz 核频率再次缩放事件时钟。
- 对正延迟预约 `max(1,ceil(seconds*1e9))`；真零工作在内部同步推进，不生成无穷零时事件链。累计取整误差上限按实际预约次数×1 ns 核查。
- `local_mem_latency` 对新计算侧一次对外宏组最多收一次首次内存接入延迟，不对每个 tile 重收；通信用户按各仓库声明的计费约定。启动成本与内存延迟分别记账，TX8 参考均为零。
- `hbm-bandwidth-contention=1` 是正常生产模式。在关闭竞争的独立实验中，flag=0 时计算按 DiePlan 消耗自己的内存时间，通信按该测试配置的开关语义；结果注明未模拟双方争用。它不属于硬件 DSE 扫描维度。
- die backend 由 `DieExecution` 提交计算及 HBM 流，不能在同一节点重复调用 `issue_compute(num_ops,tensor_size)` 累加独立 roofline 计时。

## 7. 图适配：保留外部机制，修正本地计算表达

### 7.1 当前五仓库实际生成的内容

在线生产路径主要使用 `transformer_pass_aggregated()`，将多个 `(tokens,kv_length)` span 和多个层按算子类别汇总。Transformer 本体通常为 13 个 COMP 类别与 2 个 all-reduce；加最终 norm、logits 后是测试中常见的 17 节点。**这不是 Excel 的 13 个计时组**，两种“13”不能按行对齐。

| COMP 节点类型 | die 本地执行语义 |
|---|---|
| attention_input_rmsnorm / layernorm | 按现有 norm 类型计算 |
| attention_qkv_projection | 三个实际投影；模型启用 RoPE 时在末尾追加新 Q/K 的旋转 |
| attention_qk_matmul | QK；可与接下来的三个本地节点构成 attention region |
| attention_scale_mask、attention_softmax、attention_av_matmul | 同一 attention region 的内部阶段 |
| attention_output_projection | O projection 的本 rank 部分和 |
| attention_all_reduce | 原通信节点，不进入 DieCostModel |
| attention_residual_add | 在上述 collective 之后计算，不能提前融合到 O |
| mlp_input_rmsnorm / layernorm | 现有第二个 norm |
| mlp_gate_up_projection / mlp_up_projection | SwiGLU 两个投影或 GELU 单投影 |
| mlp_swiglu / mlp_gelu、mlp_down_projection | 与前一节点构成 FFN region，类型由模型决定 |
| mlp_all_reduce | 原通信节点 |
| mlp_residual_add | 在 MLP collective 后执行 |
| final_norm、logits | 按图真实存在计算；不加入 TX8 单 block 的参考总数 |

原图已有两个 norm 和两个 residual，不重复添加。当前生成器没有独立 RoPE 描述；为使用旋转位置编码的模型补上 `position_encoding=rope` 和明确子操作，归入 QKV 壳的末段。该壳完成前输出的是旋转后的新 K；不能把 RoPE 塞进 timer 或在历史 KV 上重复做。

### 7.2 本地融合采用生成期 region 编译，不伪造子节点时间

仅在新 backend 下，公共图适配器将可融合的纯本地链编译成一个 COMP region：

```text
QK -> scale/mask -> softmax -> PV        => ATTENTION_REGION
gate/up -> SwiGLU -> down               => FFN_REGION
up -> GELU -> down                      => FFN_REGION(type=gelu)
```

合并需同时满足：同一 rank、相同 span/layer/TP 范围；内部节点无外部分支消费者；中间没有通信、KV 到达 gate、提交边界或观察锚点；输入 ready 条件相容。保留入口的全部外部依赖，出口承接末节点的全部外部后继。所有原数学子项与工作量保存在 region 的 breakdown 中。

这一步只调整本地算子实现，**不更改 collective 的位置、数量、参与者、bytes/tag，也不删除或改写调度器/watch引用的节点身份**。观察节点留在外面；如某条链不能满足条件，则按分离执行编译，真实计入外存中间结果，不强制融合。

测试须覆盖输入图的数学操作、外部依赖、通信与观察锚点。针对 17 节点在线图的测试使用对应固定夹具，节点计数不得取代算子依赖和字节归账验证。

这是采用片上 attention/FFN 所需的本地编译步骤。仅逐个 COMP 更换一个时间函数、同时把中间结果假定驻留，会遗漏跨节点缓冲和依赖；给第一个节点收全部融合时间、其余节点填 0 也不采用。

### 7.3 普通计算、控制和 replay 分开分发

执行类型由生成器填入，parser 验证：

| 节点 | 行为 |
|---|---|
| `die_op` 普通计算 | DiePlan；`runtime_ns` 必须为 0，否则报配置冲突，避免悄悄绕过仿真 |
| timer / CPU 控制节点 | 沿现有计时/replay 路径，不占 die PE/vec 资源 |
| `control_marker` | 遵从微作业完成语义；不加 die 宏组启动，不识别成 GEMM/向量算子 |
| 空计算节点 | 根据类型与工作量校验执行或 skip 语义 |
| 纯访存/纯片上计算 | 依据实际工作执行；不能仅根据 num_ops 或 tensor_size 是否为 0 判断为零工作 |
| 缺 DieOpSpec 的真实 COMP | die backend 下明确报错；不在缺参状态下输出数值执行时间 |

列车尾部、first-token、drain、exit 等 marker 目前常写成 `COMP(1,1)`。应增加显式 marker 身份；不能靠“很小的数”识别，因为真实小算子也可能很小。其 watch/时间戳身份必须保留。

`remote_weight_bytes`、MEM_LOAD/STORE 等操作遵守仓库定义的远端/本地访问支持范围。目标 LLM 默认权重驻留本地 HBM；远端权重访问须由图和内存接口显式表示，不能隐式计入本地 GEMM。

### 7.4 KV 与通信的收费归属

| 动作 | 计费者 |
|---|---|
| 决定 KV 放在哪个 rank、何时逐出/恢复、删掉还是写远端 | 原策略与 KV 管理器 |
| D2D 路径、传输、竞争、collective | 原 Astra-sim 网络及算法 |
| 通信发送端内存读、接收端内存写、restore/pool 服务 | 原端点/内存模型；沿用 hbm-charge、join、门槛等语义 |
| KV 已在本地/已准备范围内，计算核心读取它 | DiePlan 的本地内存及片内 NoC 阶段 |
| 跨中间 die 路由转发 | 原网络模型；不额外访问中间 die 的 HBM |
| 新生成的 K/V 写入本地计算输出/缓存 | 按语义张量写入一次；若原图已有明确内存写作业，适配器标记对应 writer，避免双计 |

接收端把远端 KV 写进 HBM，与计算随后从 HBM 读取是两个真实动作，可以分别计费；它们不是同一读写重复计数。相反，同一 restore 已经由现有模型收取的写入，不能在 DiePlan 中重复计“KV 恢复时间”。

对有流式 KV credit/block 的仓库，每个图中计算块只接收它已经承诺就绪的 KV 范围，按 KV gate 与 credit 条件发射。不把“预计以后会到”的全部 KV 在 COMP 发射时算作已到达，也不要求本地融合等待整个请求全部恢复，并保持图中声明的流水依赖。

## 8. 代码组织与配置接入

### 8.1 共用成本库，仓库只写适配层

采用一个版本化的 C++17 纯成本库，五仓库以相同版本 vendoring 到 `extern/die_model/`。构建不依赖运行时联网，也不要求仓库之外的兄弟目录存在。源版本、配置表和系数文件摘要必须相同；适配器可以因仓库内存接口不同而不同。

建议的源码目录及模块职责：

```text
extern/die_model/
  include/DieTypes.hh           # Hardware、DieOpSpec、DiePlan、Result
  src/Hardware.cc              # Core64、Micron HBM3E dynamic v4、矩形Die、模块/Wafer/D2D、入口验证
  src/Workload.cc              # spans/TP/算子数学图与类型检查
  src/Mapping.cc               # tile、循环、2n颗条带、XY、广播、入口/有向链路、驻留和容量
  src/CostModel.cc             # 资源工作量、孤立执行计划选择
  src/Tx8Reference.cc          # 第2节公式；没有time查表
  src/Capi.cc                  # 给Python调用的同源估值接口
  configs/                    # Core 表、HBM 配置、校准及 JSON Schema

astra-sim/workload/DieExecution.{hh,cc}  # 使用Sys的异步阶段执行器
astra-sim/workload/DieModelAdapter.*    # NodeView/COMP/HBM/统计桥接
sh_test_mesh/.../die_model_adapter.py   # 配置、形状和region编译；不实现第二套公式
```

成本库不依赖 Sys、KV 管理器或策略类，只生成资源计划。C++ runtime 使用该计划参与真实事件竞争；Python 通过 ctypes 调用同一库的 C ABI，获取**孤立运行**估值及资源量。

```text
make_hardware(config) -> Hardware | HardwareError
validate_hardware(config) -> DerivedResources | InvalidHardware
# 不依赖性能能力，返回完整 ResourceEnumeration，永不隐含最快时间：
enumerate_resources(spec,hardware) -> ResourceEnumeration | ResourceError
# 只有全部 dtype/eta/port/route 校准可用才产生唯一 TimedDiePlan：
compile_timed(spec,hardware,calibration) -> TimedDiePlan | CapabilityError
estimate_alone(timed_plan) -> TimedCostResult | CapabilityError
execute(timed_plan,sys,existing_memory_port,on_complete) -> ExecutionHandle

C ABI (version=die_model_c_v4):
  create_model(config_json) -> model_handle | error
  enumerate_resources_batch(model_handle,specs_json) -> UTF8 JSON results
  compile_timed_batch(model_handle,specs_json) -> UTF8 JSON timed results/errors
  estimate_timed_batch(model_handle,timed_plans_json) -> UTF8 JSON timing results/errors
  free_result(buffer); destroy_model(handle)
```

C ABI 不跨边界抛出 C++ 异常，返回结构化 `status,error_code,hardware_fingerprint,resource_enumeration,timed_plan,estimated_seconds`（不可用字段为**缺失/null**，不是 0、Inf 或一个退化假最快值）。静态接口返回 `resource_only`；动态接口必须返回 `timed_valid` 或 `unsupported_precision/missing_performance_calibration/missing_ingress_calibration`。`TimedDiePlan` 和纯资源候选不能共享一个可误用的 `DiePlan` 返回类型，Python/FACE/JOINT 不允许使用 `resource_only` 对候选做时间排名。调用者只能通过 C API 获取校准后的同源时间，禁止 Python 二次计算独立公式。定量 C++ 运行与 Python 估值需使用同一子流/端口/edge 约束和相同算法版本。

JOINT 事件投影的 C ABI 包括 `create_execution(timed_plan)`、`next_action(handle)`、`advance_time(handle,ns)`、`complete_memory(handle,flow_id,generation)`、`cancel_stage(handle,completion_id,generation)`、`destroy_execution(handle)`；`next_action` 输出计时器或具有完整 `flow_id,stack_id,port_id,direction,actual_bytes,eta_hbm,route,owner_macro_id,generation` 的内存请求，原 Sys 负责唯一时钟和资源共享。`create_execution` **不得接收资源枚举结果**。JOINT 适配应以实际存在的 `event_recursion_predictor.ComputeLayerSegment` 为接入点；不得依赖未经代码确认的 `ProjectionWorld` 类。不得在库内另建网络或 KV 状态。

### 8.2 版本化单一 hardware JSON：Core/HBM/Wafer/NoC/系统全部从此派生

五仓的 `sh_test_mesh/config_resolver.py` 是已有的配置分发切入点；应统一读取 §3.3 描述符（JSON 与 YAML 等值），拒绝与声明 Schema 不一致的字段，并以一份**冻结的、可 hash 的** `resolved_die_model.json` 派生 C++、Python、system 和上层 Wafer/D2D 资源。`mesh_die_hbm3e_dynamic_v4` 用于表示显式 `Nx×Ny` Core 网格、Die 宽高及双排 HBM 几何；不同 Schema 使用不同的硬件指纹和缓存命名空间。不要手动修改 `sh_test_mesh/generated/` 中由 resolver 生成的文件。

```json
{
  "compute-cost-model": "die",
  "die-model-profile": "mesh_die_hbm3e_dynamic_v4",
  "hardware-schema-version": "mesh_die_hbm3e_dynamic_v4",
  "die-model-config": "resolved_die_model.json",
  "hbm-bandwidth-contention": 1
}
```

**推荐单源配置（§3.2–§3.3 的动态字段）**：`hardware_schema_version, die_geometry_mode, core_columns_nx, core_rows_ny, die_width_mm, die_height_mm, core_pitch_x_mm, core_pitch_y_mm, die_extra_width_mm, die_extra_height_mm, compute_area_mm2(派生), hbm_sku, hbm_generation, hbm_capacity_bytes_per_stack, hbm_io_data_rate_gts, hbm_data_width_bits, hbm_peak_bytes_per_second_per_stack, hbm_package_width_mm, hbm_package_length_mm, hbm_package_height_mm, hbm_stacks_per_row, hbm_placement_mode, hbm_stack_count, row_fit_policy, hbm_row_overhang(派生), hbm_geometry_revision, core_type, sram_mib_per_core, noc_width_bits, core_table_revision, core_table_sha256, model_algorithm_revision, calibration_revision, hbm_injection_map_id, hbm_injection_ports, stack_to_ports`。目标有效性能能力、访问效率及工程延迟亦须带独立的 `provenance`/适用域，不能凭默认 null 静默返回数字。`hbm_stack_count`、总容量和峰值为**验证派生字段**而非可随意输入：要求 `hbm_stack_count==2*hbm_stacks_per_row`。

| 衍生输出 | 约束、单位及使用者 |
|---|---|
| `local-mem-capacity-bytes` | `36_000_000_000*(2n)`；进入本地 KV/权重/工作区管理器以及 C++/Python；单颗 capacity 另守卫 |
| `local-mem-bw` / `local_mem_peak_Bps` | 从 `(2n)*1_177_600_000_000 B/s` 计算**理论聚合上限**；如果 system 配置要求 GB/s 则写 `1177.6*(2n)` 且标记 GB/s，并在 C++ 入口转换**一次**。实际阶段限速另受 stacks/ports/NoC；不能把总峰值当保证吞吐 |
| `N_core`, SRAM, PE/Vec 名义峰值 | 固定 64 行和显式 `Nx*Ny` 的合法网格验证；`N_core=0` 拒绝；不能把 INT4 GOPS 填入 BF16 的 `peak-perf` |
| `HBM layout`, `hbm_injection_map_id` | 每排 n、两排摆放、模块超宽标记/可选排宽策略、每颗映射/端口负载与工程参数版本；缺实接口拒绝目标定量时延 |
| `D2D budget`, `W₀/H₀`, `N_wafer` | 由 §3.3 统一公式；`B_D2D_oneway_GBps` 给上层 **network** 能力配置单独换算；不能把 `noc_width_bits` 写入 D2D network.yml 的宽度字段，不能根据 `N_wafer` 自动改变 `rank_count` |
| `hardware_fingerprint` | 规范化 JSON + Core表 SHA、精确 Nx/Ny/W/H/pitch/余量、固定 SKU/速率/封装、注入 map 全内容和版本、动态几何算法、校准/eta/首访工程参数；C++、Python、所有仓库日志一致 |

规范 JSON Schema 须与 §3.3 的 `resource_only` 示例相容，定义跨语言输入白名单和字段类型；JSON Schema 不能替代 `N_HBM=2n`、行宽、端口全图/读写域和校准适用性的**语义校验器**。`execution_intent=target_timed` 必须配置真实能力/端口映射；`execution_intent=sensitivity` 只报告不可用于在线策略的敏感性结果。

**Schema 与硬件指纹规范**：`resolved_die_model.json` 只能用规范 snake_case 硬件域字段；`system` 的带连字符字段是 resolver 的**输出**，不能作为规范输入的别名。枚举 `memory_service_model=physical_multiresource_v1` 对 target 强制；`hbm_injection_ports` 每项包含 `port_id:nonnegative_int,edge_side:north|south,mesh_x:int,mesh_y:int,shared_domain_id:string,capacity_Bps:positive_decimal_string,read_write_shared:bool`；`stack_to_ports` 项包含 `stack_id,port_id,weight_integer_numerator,weight_integer_denominator`，每颗归一化权重和精确为 1，分配实际字节用定序整数商/余数。`shared_domains` 定义 `shared_domain_id` 和 `capacity_Bps`；`port_to_core_mapping` 至少明确每端口的合法入网 Core 与路由边；有向边 `edge_id=(src_x,src_y,dst_x,dst_y)`，须是 mesh 内相邻的有向跳。禁止端口图只用文本路径作实现来源。

**目标定时模式的必填字段**。`execution_intent=target_timed` 须满足 dtype/入口/PE/Vec 能力约束，并提供 `eta_hbm_matrix`、`eta_hbm_vector`（均为 `(0,1]` 的正十进制字符串）、`local_mem_latency_ns`（非负整数）、`performance_model_provenance`、`ingress_map_provenance`（非占位证据/模型版本标识）及非空 `op_domain`。`supported_precisions` 必含此 BF16 目标图所需 `BF16`、`accumulator_dtype=FP32`。`UNCALIBRATED`、`TO_BE_DEFINED`、`synthetic_test_only` 或 `unverified` 不能伪装成 `target_timed`。以上仅构成配置完整性门禁：**输入一个文本来源字段不能证明参数真实或准确**，还必须由集成验收及外部工程证据证明适用域。`resource_only` 可以全部为 null 并只返回静态资源。TX8 DDR 的 0.70/0.341 仅适用于参考 profile，不构成 HBM3E 的已标定效率。

**注入映射跨字段不变量**。`stack_to_ports` 中 `(stack_id,port_id)` 必须全局唯一；对每个存在的 stack，所有正权重之和严格为一。每个被 stack 引用的 port 至少具有一条合法的 `port_to_core_mapping` 路由；目标 workload 使用的每个 `(port_id,dst_core)` 都应能生成已校验的明确路径，否则在编译该 stage 时返回 `missing_ingress_calibration`，**仅有一条对某个 Core 的样例路由，不表示整张 Core mesh 均可达**。两侧各一排时，`stack_id<n` 仅能映射 north，`stack_id>=n` 仅能映射 south；同侧两排全部映射 north（固定约定的 `stack_id` 顺序）。每颗栈按地址条带与分端口权重决定完整整数 payload，禁止重复计算。供源文件的两种摆放其几何虽然相等，不能共用未区分的 port map。

**资源份额与端点覆盖**。一个子流可以同时遇到 stack / port / edge / shared-domain 瓶颈。`shared_domains` 的 B/s 上限须纳入与 stack/port/edge **同一个** `progressive filling` 系数矩阵；任何共享 PHY 在同 Tick 都不得给多个请求重复分配总额度。`COMM_READ/WRITE`、RESTORE、POOL 及 NoC-only 的覆盖集合以真实活动端点为准；真实仓库实现需动态证明这些任务进入共享域。合成 fixture 只验证规范算法，不是五仓已实现证明。

**数字的规范等价编码**。`sram_mib_per_core` 是固定的十进制档位：JSON 的 `1` 与 `1.0` 必须在规范化后得到相同的十进制字符串 `"1"` 和相同 SHA-256；不能因数值词法差别令缓存分裂。非整数硬件物理量按无指数规范十进制字符串归一化，整数标识仍用整数 JSON。端口路径数组保持语义顺序，重复 `(stack_id,port_id)` 明确报错，不通过两项权重相加偷渡。前述规则须配置正反例单元测试；结构性校验不能替代外部性能校准。

**数组与字符串正规化细则**：端口数组按 `port_id`、`stack_to_ports` 按 `(stack_id,port_id)`、`shared_domains` 按 `shared_domain_id`、`port_to_core_mapping` 按 `(port_id,dst_x,dst_y)` 排序；路由 hop 顺序是有意义的，不准重新排序。验证所有枚举大小写、非重复 ID、mesh 合法相邻坐标、首跳确从 port 注入、每条 flow 可达合法目的 Core、每个被引用的 domain 具有可追溯的正数 cap、读写共用与分离源数据；任一缺失则阻断目标时间。所有字符串使用 Unicode NFC；非整数输入必须在接收层转为无指数的精确有限十进制字符串：去掉无意义的前导加号、整数前导零和小数末尾零，小数点后全零则去小数点，正负零统一 `"0"`；禁止 float round-trip 影响 hash。配置数组按上述规则正规化后再排序/哈希，C++/Python 需同一黄金 fixture。

**canonical JSON + SHA-256**：把 `hardware_schema_version, SKU, Nx/Ny/W_die/H_die/pitch/余量, 2n,模块尺寸,速率,Core64 SHA,row_fit_policy/overhang,完整 map/ports/weights/shared_domains,效率/延迟/calibration provenance,model_algorithm_revision,memory_service_model` 的**解析后**配置转换为 UTF-8 JSON，按 Unicode 字符串键的字典序递归排序，数组保留语义顺序；整数不能以浮点承载，所有非整数数字以约定十进制字符串正规化（不允许 NaN、Infinity、-0、科学记号别名），`null` 明确保留；序列化无空白、`ensure_ascii=False`，SHA-256 小写 hex。派生/缓存/日志统一使用该 hash，时间变化的 `runtime_stats`、`rank_count`、当前 KV 字节、当前 tick 不进入硬件 hash；算子级缓存再附独立 shape/KV-ready key。该规则是硬件指纹规范，需要以 C++ 与 Python 完全相同的黄金 JSON→hash 测试签收，不能声称现有仓库已完成。

**单独的校准身份**： `pe_effective_bf16_flops_per_second_per_core` 与 `vec_effective_eq_ops_per_second_per_core` 必须带 `supported_precisions,op_domain,accumulator_dtype,calibration_revision,performance_model_provenance,units`；前者 BF16 FLOP/s、后者特殊函数 mix 的 eq-op/s，二者已包含计算效率、不可再次乘 η。两者均为 `null` 时硬件对象合法、数值 backend 不可运行。仿真 HBM 每颗理论服务上界不冒充持续有效标定。`memory_service_model` 模式决定能否用于策略评分；aggregate fallback 只作敏感性结果。

输入字段规则：配置必须包含 `Nx,Ny,W_die,H_die,n,placement_mode`，并满足二维 Core 网格、HBM 双排布局、实际模块外廓和精度/端口校验。仅有面积字段时返回 `missing_die_geometry`；未声明的 `hbm_per_side/hbm_sides` 返回 `invalid_schema/unsupported_layout_field`。`die_geometry_mode=dynamic_core_grid`；默认 `row_fit_policy=allow_module_overhang`，对超宽 HBM 生成扩宽模块并标记物理待验证，`row_fit_policy=required` 仅用于有该封装限制的实验。不同硬件及参考 profile 按隔离的能力、容量和带宽配置解析，不共享隐含常量。

一份 `hardware_fingerprint` 必须贯穿 `make_hardware→enumerate_resources/compile_timed→estimate_alone→execute` 及 Python C ABI；runtime 与 FACE/JOINT 的策略 oracle 必须使用相同的 Micron HBM3E 容量、峰值、端口和硬件指纹。生产启用 die backend 以 BF16/SFU/入口能力满足为前提，缺校准时返回明确的缺参错误；`hbm-bandwidth-contention=0` 仅保留测试对照且必须标识为无共享竞争模拟。

### 8.3 输入链路必须完整贯通

按以下顺序实现字段传递：

```text
模型配置 + 当前pass_spans + 当前TP/rank + KV已就绪范围
 -> transformer_pass[_aggregated]，保留形状与权重pass分组
 -> 本地region编译，保留所有外部通信/观察边界
 -> OnlineTraceBuilder / GraphBatch JSON
 -> ParsedGraphBatch严格schema、GraphBatchCommitter预检查
 -> ComputeAttrs / NodeStore
 -> Workload::issue_comp
 -> DieModelAdapter -> DieExecution
```

静态 ET 兼容使用同一个 JSON 描述作为命名属性，经 `ETFeederGraphSource::view_of()` 解析；没有形状的历史 ET 只在 legacy backend 下运行。online 输入应由类型化算子图提供。

严格 parser 的允许字段、整数范围、`hardware_schema_version=mesh_die_hbm3e_dynamic_v4`、`hardware_fingerprint`、`hbm_injection_map_id`、`hbm_stacks_per_row`、`hbm_placement_mode`、`row_fit_policy`、`core_columns_nx`、`core_rows_ny`、`die_width_mm`、`die_height_mm`、operation/accumulator dtype、真实 BF16 能力 ID、span/layer/TP 关系都要一并更新；任一层丢掉 DieOpSpec 都应被检测到。NodeStore 的尺寸/内存压缩测试若有布局断言，需使用共享不可变 descriptor 或侧表句柄，避免给每个短命 marker 复制大 JSON。

### 8.4 Python 代价预测和 C++ 执行使用同一 HBM3E dynamic v4 能力、上限与字节定义

FACE/JOINT 中参与放置评分的 Python compute oracle 与 Astra-sim C++ `DieExecution` 必须从**同一不可变 `hardware_fingerprint`**产生、共享目标精度能力验证和实际 HBM 的 `2n` 颗容量/理论上界、栈/入口/NoC/D2D 名义预算；C++ runtime 额外读取即时竞争状态，不得产生另一套硬件常量。`enumerate_resources` 输出无时间资源候选；`estimate_alone` 只能对合法 `TimedDiePlan` 用同源成本库计算无外部竞争下的工作量和时间边界，返回 `assumptions + bottleneck`；它不会预言未来实际拥塞。Python 只组合其结果与既存策略 D2D/restore 成本，不改变候选、tie-break、KV、TP 和网络依赖的规则。若 `pe_effective_bf16_flops_per_second_per_core`、`vec_effective_eq_ops_per_second_per_core`、端口 map/速率未知，统一返回 `unsupported_precision`、`missing_performance_calibration` 或 `missing_ingress_calibration`，**拒绝**用抽象 roofline、TX8 效率/峰值或 `service_time×bandwidth` 生成目标 Die 的时延。

验收时相同 DieOpSpec + fingerprint 在 C++/ctypes 输出完全一致的 `N_HBM, physical_bytes, service_units, peak_Bps, capacity_bytes, D2D budget, layout, resolved_port_map, time_status`；如果 runtime 的有效 HBM 服务率较独占估计低，必须给出并发集合/端口/链路瓶颈证据。FACE 的 `FaceRooflineEstimate` 与 JOINT 的 `_joint_cost_model`、`event_recursion_predictor.ComputeLayerSegment` 均要切换到 version-aware 成本与流量边界；`legacy_estimate` 属于独立参考统计，**不可参与目标模式调度评分**。

### 8.5 计划缓存隔离、性能统计、运行规模

缓存至少绑定 `hardware_schema_version=mesh_die_hbm3e_dynamic_v4`、`hbm_sku`、`hbm_capacity_bytes_per_stack`、`hbm_peak_bytes_per_second_per_stack`、封装物理尺寸/速率、`N_HBM=2n`、`hbm_stacks_per_row`、`hbm_placement_mode`、`row_fit_policy`、`hbm_row_overhang`、`hbm_geometry_revision`、`core_columns_nx,core_rows_ny,die_width_mm,die_height_mm,core_pitch_x_mm,core_pitch_y_mm,die_extra_width_mm,die_extra_height_mm`、`hbm_injection_map_id + full mapping/port rates`、Core 64 行规范 SHA、Core 配置 `(type,MiB,B,L₀)`、算法版本、dtype 能力与**校准/效率/延迟**版本，另加 profile、算子/region 形状、span、pass/layer/TP、KV 就绪范围与融合标志。不能让 v2/v3/v4、不同二维网格/外形、row_fit_policy、不同 ingress map、TX8/target 跨命中；计划不可变且不缓存活动执行器，模型实例可先用 4096 条上界、LRU 淘汰。

每个 die、每次宏组和每个 stack/port/有向 NoC link 分别报告：`hbm_theoretical_peak_Bps`、`hbm_effective_ingress_peak_Bps`（若未知为 null）、`hbm_effective_serviced_Bps`（仅实际事件有数据时）、`memory_bottleneck_type`、`noc_link_effective_Bps`、`d2d_oneway_budget_GBps`、`hbm_capacity_bytes`、`N_HBM`、`actual_hbm_read_bytes/write_bytes`、`service_units`、`actual_noc_edge_bytes`、`KV/weight/workspace_reserved_bytes`、`dynamic/optional_row_fit`、schema/SKU、`hardware_fingerprint`、时间/能力 status、工程先验/外推标记。**物理利用率分子只能是真实 D**： `hbm_actual_bytes/(理论聚合峰值*elapsed)` 是对芯片总峰值的实际比，不用效率放大的 `U`；独立提供栈/端口/链路用量，不以一个数字掩盖偏斜。不计数未执行的 stage 或已取消未传输的数据；INT4/FP32/BF16 的计算利用率只在同 dtype 单位内计算。

重复 wave 惰性展开，同一波次不能按 Core 数生成独立冗余事件；动态份额变化时不能用整段 `repeat×独占时间` 快进。GraphSource/watch/MetricCollector 的生命周期与统计字段按其接口契约生成；统计 schema 升级不能导致常驻保存全部内部 tile。原 `tx8_reference_v1` 仍按第 2 节输出 200 GB/s DDR 参考，并在每条记录标 `reference_not_hbm3e`。

## 9. 五仓库接入位置与职责

下列路径以 `cs-sunH/template` 的参考提交 `1491e0ebb69ce25a7d2c37a53947af1d82b8d3de` 为基线定位。实施时需要在实际目标分支重新确认对应文件、符号和 ABI；如果分支结构不同，以代码中的等效接口为准。各仓遵守各自的 KV、通信图及调度语义，但均必须消费相同的 HBM3E 规范硬件描述符。

### 9.1 共同必改点

**五仓统一硬件契约**：五仓均通过同一 resolver 读取 dynamic v4 规范输入，并且 `Sys/Workload::issue_comp/LocalHbmBandwidthModel/Python oracle` 获取**同一** `(schema,SKU,Nx,Ny,W_die,H_die,2n,动态模块外廓,端口映射,Core SHA,D2D,校准版本)`。以下文件和符号仅用于实现定位；各仓既有 job 类型及策略差异分别维护。一次完整目标执行的真正起点是 resolver 的冻结配置和 BF16/入口能力，而不是复制一个新常量到 `Roofline.cc`。

| 文件/函数 | 修改内容 |
|---|---|
| `astra-sim/system/Sys.{hh,cc}` | backend 分发、64 行 Core/四档 B/HBM3E dynamic v4 的矩形网格 Nx×Ny 与 HBM 2n 与 SKU/端口图/profile/config、精度白名单与失败状态、成本库生命周期 |
| `astra-sim/workload/execution_driven/GraphSource.hh::ComputeAttrs` | 增加 DieOpSpec 的不可变描述/句柄及明确控制类型 |
| `ParsedGraphBatch.cc`、`GraphBatchCommitter` | 校验新 schema 和描述，commit 前拒绝不完整真实算子 |
| `NodeStore`、`ETFeederGraphSource::view_of` | 完整保留元数据并遵循回收；兼容静态测试入口 |
| `W/generate_trace.py::transformer_pass[_aggregated]` | 在形状丢失前生成 typed spec，保留 spans/layers/weight_passes |
| `W/online/graph_batch_builder.py` | 传递 spec，标记控制节点，调用纯本地 region 编译 |
| `Workload.cc::issue_comp` | 分发到 DiePlan 计算路径；保证节点完成语义唯一；保持普通完成出口 |
| `LocalHbmBandwidthModel.{hh,cc}` | 支持 dynamic v4 计算内存阶段的 per-stack/port/NoC 限速、`actual_bytes/service_units` 双计量、回调与 COMM/RESTORE/POOL 公平竞争，所有访问进入共享仲裁域 |
| `Workload/HardwareResource/Statistics/MetricCollector` | 一次占槽、一次完成；资源/统计新口径；活动执行器 drain |
| `sh_test_mesh/config_resolver.py`、system 模板 | MiB 2^20、dynamic v4 SKU/36GB/1177.6GB/s/2n/Nx×Ny/W/H及模块超宽标记、Die/Wafer/D2D、HBM ingress map、Core64 指纹/固定 L₀/矩形网格边界拒绝、INT4/FP32 字段及兼容 BF16 峰值检查 |
| 根 CMake 与 analytical 测试 CMake | 显式引入公共成本库/共享 ABI/回归 target；不得对目标 die 的缺参自动采用其他性能模型 |

仅修改 `Roofline.cc` 不会改变默认主路径的计时：现有 HBM fluid model 会自己排空 num_ops 与 tensor_size。仅填 `runtime_ns` 则会绕过 HBM 竞争，这两种方式均不满足 die backend 的资源竞争约束。

### 9.2 astra-sim-wscllm

- **接入要求**：resolver 必须将 2n 颗与动态模块摆放、`local-mem-capacity-bytes/local-mem-bw` 和端口预算传给 Sys；`issue_comp` 的新内部 stage 与 COMM_READ/WRITE 共享原模型，保证 graph/watch 一次完成；WSCLLM 的 KV prefill 重计算规则不改。

- [Workload.cc](https://github.com/cs-sunH/template/blob/1491e0ebb69ce25a7d2c37a53947af1d82b8d3de/astra-sim-wscllm/astra-sim/workload/Workload.cc)：`issue_comp` 在 **395** 行，`finish_generic_node` 在 **951** 行，`on_local_hbm_job_complete` 在 **1039** 行。当前 `issue_comp` 的 fluid 路径仍调用 `local_hbm_bandwidth_model->issue_compute(node_num_ops,node_tensor_size,wlhd)`；`DieExecution` 应在 COMP 分发点取得符合 DieOpSpec 的节点，全部内部阶段完成后经唯一外层完成出口提交。
- [LocalHbmBandwidthModel.hh](https://github.com/cs-sunH/template/blob/1491e0ebb69ce25a7d2c37a53947af1d82b8d3de/astra-sim-wscllm/astra-sim/workload/LocalHbmBandwidthModel.hh)：`JobKind={COMP,COMM_READ,COMM_WRITE}`；新增目标 compute 阶段时不应擅自搬进 LRU 的 RESTORE/POOL 业务。
- [generate_trace.py](https://github.com/cs-sunH/template/blob/1491e0ebb69ce25a7d2c37a53947af1d82b8d3de/astra-sim-wscllm/sh_test_mesh/workload/llama2_7b_inference/generate_trace.py)：`transformer_pass` **401** 行，`transformer_pass_aggregated` **627** 行。应在聚合前留下 span/TP/权重遍数与 dtype 信息，而非从 `num_ops/tensor_size` 反猜。
- [graph_batch_builder.py](https://github.com/cs-sunH/template/blob/1491e0ebb69ce25a7d2c37a53947af1d82b8d3de/astra-sim-wscllm/sh_test_mesh/workload/llama2_7b_inference/online/graph_batch_builder.py)：`comp` **255** 行、`_emit_train_body` **650** 行；marker 与普通计算要类型化隔离。
- [wsc_llm_scheduler.py](https://github.com/cs-sunH/template/blob/1491e0ebb69ce25a7d2c37a53947af1d82b8d3de/astra-sim-wscllm/sh_test_mesh/workload/llama2_7b_inference/wsc_llm_scheduler.py)：`build_static_pd_mapping` **397** 行、`select_prefill_instance` **785** 行；保留队列/拓扑策略与 KV 重新计算语义，不修改 rank 级选实例规则。

### 9.3 astra-sim-wscllm-LRU

- **接入要求**：同一动态 Nx/Ny/W/H、Core64 与端口 fingerprint 延伸到 LRU 的 RESTORE/POOL 账本；本地总量与每颗容量以 36e9×2n 校验，restore/pool/COMM/COMP 同一仲裁域；不得更改 LRU inactive-session 状态机。

- [Workload.cc](https://github.com/cs-sunH/template/blob/1491e0ebb69ce25a7d2c37a53947af1d82b8d3de/astra-sim-wscllm-LRU/astra-sim/workload/Workload.cc)：`issue_remote_mem` **438** 行、`issue_local_hbm_kv_restore` **477** 行、`issue_comp` **529** 行、`finish_generic_node` **1192** 行。接入点需保留远端权重、恢复及其他异步完成门槛。
- [LocalHbmBandwidthModel.hh](https://github.com/cs-sunH/template/blob/1491e0ebb69ce25a7d2c37a53947af1d82b8d3de/astra-sim-wscllm-LRU/astra-sim/workload/LocalHbmBandwidthModel.hh)：真实 `JobKind` 为 `COMPUTE,RESTORE,COMM_READ,COMM_WRITE,POOL_READ,POOL_WRITE`，同一个 N-way equal-share 端口竞争；新增计算内部访存阶段需通过原端口公平仲裁。
- [generate_trace.py](https://github.com/cs-sunH/template/blob/1491e0ebb69ce25a7d2c37a53947af1d82b8d3de/astra-sim-wscllm-LRU/sh_test_mesh/workload/llama2_7b_inference/generate_trace.py)：`transformer_pass_aggregated` **646** 行；与基础 WSCLLM 使用同一 schema/精度能力验证。不得将已有 inactive-session LRU restore 误替换为 base 的 prefill 重算语义。

### 9.4 astra-sim-face

- **接入要求**：FACE 放置评分 `FaceRooflineEstimate` 改消费与 C++ 一致的 `estimate_alone` 和硬件指纹，KV 容量、端口/NoC/D2D 分别限速，缺 BF16 或 ingress 能力阻断 target 排名；COMM hbm-charge/join 依赖不变。

- [Workload.cc](https://github.com/cs-sunH/template/blob/1491e0ebb69ce25a7d2c37a53947af1d82b8d3de/astra-sim-face/astra-sim/workload/Workload.cc)：`issue_comp` **426** 行、`on_local_hbm_job_complete` **1048** 行；`issue_send_comm` 的网络发送和本地 HBM 端点竞争由 join 协同完成，不得因 DieExecution 更换而丢失 `hbm_charge` 控制和唯一完成。
- [generate_trace.py](https://github.com/cs-sunH/template/blob/1491e0ebb69ce25a7d2c37a53947af1d82b8d3de/astra-sim-face/sh_test_mesh/workload/llama2_7b_inference/generate_trace.py)：`transformer_pass_aggregated` **646** 行；DieOpSpec 须携带真实 shape/pass 信息，不根据聚合操作量反推维度。
- [graph_batch_builder.py](https://github.com/cs-sunH/template/blob/1491e0ebb69ce25a7d2c37a53947af1d82b8d3de/astra-sim-face/sh_test_mesh/workload/llama2_7b_inference/online/graph_batch_builder.py)：`comp` **254** 行、`_emit_train_body` **561** 行；first-token/train marker 仍维持独立类型和原 watch 身份。
- [face_scheduler.py](https://github.com/cs-sunH/template/blob/1491e0ebb69ce25a7d2c37a53947af1d82b8d3de/astra-sim-face/sh_test_mesh/workload/llama2_7b_inference/face_scheduler.py)：`FaceRooflineEstimate` **260** 行、`estimate_iteration_time_ns` **269** 行起；当前计算使用 `instance_size*hardware.peak_perf_tflops*1e12` 和 HBM 带宽。本地计算成本应调用与 C++ 相同且精度合法的接口；**不能**把 INT4 GOPS 填入 BF16 `peak_perf_tflops`，候选、评分及 tie-break 保持不变。

### 9.5 astra-sim-face-LRU

- **接入要求**：FACE-LRU 的独立 Python oracle 与 FACE base 同版；`RESTORE/POOL/COMM` 按该仓原状态机的读写占用共享 HBM3E 栈与端口预算，保留 `hbm-charge=false` 和远端 KV 实际 byte。

- [Workload.cc](https://github.com/cs-sunH/template/blob/1491e0ebb69ce25a7d2c37a53947af1d82b8d3de/astra-sim-face-LRU/astra-sim/workload/Workload.cc)：`issue_remote_mem` **480** 行、`issue_local_hbm_kv_restore` **508** 行、`issue_comp` **564** 行。die backend 执行必须遵循 remote/pool/restore 与通信端点语义。
- [generate_trace.py](https://github.com/cs-sunH/template/blob/1491e0ebb69ce25a7d2c37a53947af1d82b8d3de/astra-sim-face-LRU/sh_test_mesh/workload/llama2_7b_inference/generate_trace.py)：`transformer_pass_aggregated` **646** 行；[face_scheduler.py](https://github.com/cs-sunH/template/blob/1491e0ebb69ce25a7d2c37a53947af1d82b8d3de/astra-sim-face-LRU/sh_test_mesh/workload/llama2_7b_inference/face_scheduler.py) 的独立 `FaceRooflineEstimate`/`estimate_iteration_time_ns` 实现仍需与 FACE base 一致替换。
- `hbm-charge=false`、pool、restore、远端 KV 字节归属与 LOCAL/REMOTE LRU 语义保留，不把其状态机改成 base 的逻辑删除。

### 9.6 astra-sim-joint：计算预测与事件递归接口

- **接入要求**：JOINT 的 `Workload::issue_comp`、`JointCostModel`、`event_recursion_predictor.ComputeLayerSegment` 在同一 fingerprint/Nx/Ny/W/H/2n/ingress 口径下处理 compute 与 memory；`runtime_ns` 直接覆盖路径**不得**冒充有公平 HBM 竞争的新 backend，`hbm_share_divisor` 只能调节共享 HBM 活跃阶段。

- [Workload.cc](https://github.com/cs-sunH/template/blob/1491e0ebb69ce25a7d2c37a53947af1d82b8d3de/astra-sim-joint/astra-sim/workload/Workload.cc)：`issue_remote_mem` **415** 行、`issue_local_hbm_kv_restore` **444** 行、`issue_comp` **490** 行。当前 `issue_comp` 使用 `roofline->get_perf()`；当 `compute.runtime_ns!=0` 时直接覆盖解析时间（可绕过 fluid HBM），其余满足条件的任务进入 `issue_compute`。因此不能靠往 `runtime_ns` 填一个新总时间冒充完成 HBM 阶段竞争。
- [generate_trace.py](https://github.com/cs-sunH/template/blob/1491e0ebb69ce25a7d2c37a53947af1d82b8d3de/astra-sim-joint/sh_test_mesh/workload/llama2_7b_inference/generate_trace.py)：`transformer_pass_aggregated` **903** 行；[graph_batch_builder.py](https://github.com/cs-sunH/template/blob/1491e0ebb69ce25a7d2c37a53947af1d82b8d3de/astra-sim-joint/sh_test_mesh/workload/llama2_7b_inference/online/graph_batch_builder.py) 的 `comp` **307** 行、`_emit_train_body` **890** 行、内层 `emit_pass` **995** 行。保持 readiness、credit、TP collective 与计算可用 KV 范围的原图契约。
- [sh30_online_scheduler.py](https://github.com/cs-sunH/template/blob/1491e0ebb69ce25a7d2c37a53947af1d82b8d3de/astra-sim-joint/sh_test_mesh/workload/llama2_7b_inference/online/sh30_online_scheduler.py)：`_joint_cost_model` 从参考提交的 **4114** 行开始；从 **4180** 行附近按 `estimate_prefill_task_load_ns` / `_decode_task_load_ns_cached` 构造 roofline 工作量，并向下游注入 `prefill_task_load_ns_fn`。新接口必须保留真实 chunk/context 与决策时只读快照。
- [joint_cost_model.py](https://github.com/cs-sunH/template/blob/1491e0ebb69ce25a7d2c37a53947af1d82b8d3de/astra-sim-joint/sh_test_mesh/workload/llama2_7b_inference/joint/joint_cost_model.py)：参考提交中 `JointCostModel` 位于该文件；`JointCostModel` 的字段含 `prefill_ns_per_token`、`decode_ns_per_token`、`prefill_task_load_ns_fn`，在 **1853** 行附近按原 roofline 基础乘在线因子计 compute leg。使用 `JointCostModel` 的现有成员及估值入口，不依赖无实现的占位符。
- [event_recursion_predictor.py](https://github.com/cs-sunH/template/blob/1491e0ebb69ce25a7d2c37a53947af1d82b8d3de/astra-sim-joint/sh_test_mesh/workload/llama2_7b_inference/joint/event_recursion_predictor.py)：当前存在 `ComputeLayerSegment`（约 **288** 行）数据类，包含 `base_ns_by_rank`、`memory_bytes_by_rank`、`port_by_rank` 与 `group`，这是需要审查计算与 HBM 端口时序的实际消费点之一。事件预测接口以该文件所定义的 `ComputeLayerSegment` 和现存调用链为准。

**JOINT 适配要求：**

1. `JointCostModel` / `sh30_online_scheduler` 的 compute oracle 用同一能力校验后的局部 DiePlan 产生阶段预算。精度未知时明确拒绝 die backend 的数值预测，不以抽象 TFLOP/s、`prefill_ns_per_token` 或虚构 HBM byte 静默代填。
2. 事件侧首先适配**现存** `ComputeLayerSegment(base_ns_by_rank,memory_bytes_by_rank,port_by_rank)` 和 `event_recursion_predictor` 的共享端口/route 流；再评估需否引入新的可暂停 `DiePlan` 状态机。若引入事件状态机模块，需明确新增源码、ABI 与生命周期，不依赖未经确认的类名。
3. 现有 `JointCostModel` 的 action/credit/route/TP-shard/hbm-port/telemetry/service-factors 和残余工作规则保留。若将来使用一个 `hbm_share_divisor` 或端口流数，在内存活跃阶段缩放带宽，不将无关的 PE/vec/NoC 时间整体乘除数；物理 byte 与预算 service units 分开。
4. 按目标分支实际存在的类与函数增加 dtype-schema、实际 byte 与 estimate/execute 一致性回归，然后才把新成本结果用于 copy/remote-read/recompute 排名。roofline 参数可供单独的参考执行 profile 使用，不进入 die backend 的资源执行。

JOINT 必须同时完成 `Workload.cc`、代价预测与事件递归接口的适配；仅替换 COMP 总时长不满足本规范。 

## 10. 实施顺序、测试与构建验收

### 10.1 实施顺序与依赖关系

| 次序 | 实现任务与验收里程碑 | 能进入下一步的证据 |
|---:|---|---|
| 1 | 冻结 v4 动态 schema、Micron SKU、Core64原表+hash、Nx/Ny/W/H、HBM 双排及动态 Wafer/D2D 公式 | 64 Core 行与来源表一致；合法矩形网格、overhang 模块、DSE 约束、源 Word 510.7/583.2/213.4mm² 示例通过 |
| 2 | 同一 `resolved_die_model.json` 分发 resolver、Sys 和 C ABI；Schema 与硬件指纹完整校验 | 五仓、Python/C++ 同指纹，`local-mem` 容量/单位换算、D2D 与动态 Nx/Ny/W/H 测试一致 |
| 3 | 设计并实现可查来源的 HBM→NoC 端口、路由/带宽/共享域、HBM stack/port/edge 的流体服务限速 | 不存在越界 port 或免费第二排；缺接口参数明确 `missing_ingress_calibration` |
| 4 | 精度能力 API、Core/SRAM tile/真实 tensor shape、NoC 路由与 FFN/attention 阶段 | 无 BF16/SFU 时阻断可信 target 时间；有合法校准才做数值与有限缓冲回归 |
| 5 | WSCLLM base 连接 `DieMemoryStageRequest` 到共享 HBM 仲裁器，使用 Sys Tick 和内部回调 | 多阶段一次 COMP terminal、first access 一次付费、COMP/COMM N-way 动态共享、取消与 drain 守恒 |
| 6 | 在现有图边界实现 RoPE/region/streaming KV，再接 FACE base 及 Python oracle | Graph/watch/gate/TP/dtype/spec 元数据不丢；`estimate_alone` 与运行时在无争用同结果 |
| 7 | 接 WSCLLM-LRU、FACE-LRU 的 RESTORE/POOL/LRU 状态机，再接 JOINT 的实际 `JointCostModel` 与 event recursion | KV/消息真实 byte/owner/credit遵循仓库协议；C++/Python 共用指纹/预测且异常透明 |
| 8 | 用架构 DSE 实际生成的候选集合与已校准的 subset 做五仓比较，保存统一版本的性能与资源结果 | 不设候选固定总数；策略输出和外部D2D行为回归；只有目标精度/入口参数齐备才出可比较 target 时延 |

缺 BF16 性能支持时不得把 `mesh_die_hbm3e_dynamic_v4` 直接作为可生成真实 BF16 在线执行时间的 production backend。

### 10.2 必须具备的数值与动态验收

| 编号 | 用例/输入 | 预期断言及不得越过的边界 |
|---|---|---|
| T-01 | 任意合法整数 `Nx,Ny` 与矩形 `W/H` 的 resolved 实例 | `N_core=Nx×Ny`，满足每轴 Core 长度、pitch 与总外围余量；以配置字段与几何约束判定合法性 |
| T-02 | `n=1,2,3` 及两种摆放模式 | HBM颗数 `2n`、容量 `72n GB`、理论总峰值 `2355.2n GB/s`；同 n 两模式二维几何同值但 port/routing 不默认相同 |
| T-03 | 215×215 mm 系统区域、10%等效模块间NoC、矩形 Die | `W_module=max(W_die,row_width)+0.30,H_module=H_die+22.56`，`W_eff=W_module/√0.9,H_eff=H_module/√0.9`；分别 floor 每轴；源 Word `7×11` 例得 11×4 模块，`8×11/n2` 例得 9×4 |
| T-04 | 实际 `A_die=W×H` 与 n 输入 bump/D2D | `B_bump=(A_die/0.055²)×0.4×2.5/8`，`B_D2D_oneway=(B_bump−2355.2n)/2`，负值判为不可行；必须使用输入的实际 Die 宽高和面积；rank_count不自动变化 |
| T-05 | B64/128/256/512、1GHz、η=0.70 | 每条有向链路 **5.6/11.2/22.4/44.8 GB/s**；单颗若要满芯片峰值至少 **211/106/53/27** 条这样的并行有效链路；端口/链路有限则限制内存服务率 |
| T-06 | Core64 资源表，固定 `Nx,Ny`，四档 B | 四档 B 采用同一核型/SRAM 对应的 `L₀` 与 `Nx*Ny`；PE/Vec 数量分别按各位宽的 Core 表记录计算，非方形网格合法，INT4 峰值不得充当 BF16 算力 |
| T-07 | Excel `计时组13行`，原 200 GB/s/64GB/48MB TX8 参考 | 各项名称匹配、合计 **11897.602619 μs**（double 合计 11897.602619368336 μs），第2节13组及权重/软最大化FFN路径保持不变；独立于 HBM3E 目标硬件校准 |
| T-08 | COMP + COMM_READ/WRITE + LRU RESTORE/POOL，取消、generation、同Tick、0字节/尾字节 | 共享仲裁器公平动态分配；`advance_to` 在重新预约之前；真实 `actual_bytes` 守恒、service_units 单独记时间、COMP仅一次 terminal、不泄露 callback |
| T-09 | 同一硬件输入喂资源生成器及五仓 resolver/Python C API | `schema/N_core/N_HBM/capacity/峰值/D2D/W0/H0/Wafer/layout/port_map/fingerprint` 全一致，runtime 额外拥塞要列瓶颈证据 |
| T-10 | 不给真实 PHY/ingress/访存延迟/BF16 PE/SFU | 可输出名义资源/上下界，但对无支持的目标数值时间按 `unsupported_precision/missing_performance_calibration/missing_ingress_calibration` 结构化阻断；不得编造热/PHY/真实延迟 |
| C-01 | `N_HBM=2` 和 `4`，真实 byte 等分/全部落单颗 | 均匀与极端偏斜服务率不同；每颗上限、端口/NoC 争用、`ΣD_j=D`、`D=0` 不产生 job；一个宏组首个非零 HBM 访问只付一次首访延迟 |
| C-02 | `B=64,128,256,512` 且同一端口数、同时两个及N路stage | B较窄时有向链接瓶颈；同侧第二排不得新增免费入口；多个 stage 不能各获完整聚合上限；先完用户结束后剩余用户速率可更新 |
| C-03 | TX8 参考 profile 与 HBM3E 目标 profile 在相同算子 shape 下运行 | TX8 按独立参考时间计算；目标 profile 缺少有效性能参数时不得生成可信执行时间 |
| C-04 | `row_width>W_die` 的动态 overhang 与可选 row_fit 筛选 | 默认生成增宽 `W_module`、标 `physical_layout_unverified`；只有显式 `row_fit_policy=required` 才过滤，缓存标签不同；布局候选数量从实际搜索结果统计 |
| C-05 | 相同外部图、KV gate/credit、D2D send/recv | 不改 payload/tag/TP/collective/watch/generation/LRU规则；因容量变化导致的实际准入/迁移结果允许变，但不能把 Wafer 模块数自动变 ranks |
| C-06 | C++ 与 FACE/JOINT Python oracle 同 DieOpSpec 与共享轨迹 | 孤立估值同源；拥塞轨迹一致时等值；差异必须能溯到端口、share 或事件；runtime 与策略必须使用同一 HBM3E 硬件指纹与性能口径 |

**多资源仲裁与配置的确定性验收场景**：

| 编号 | 最简物理测试（合成 B/s，不代表 Micron 数值） | 期望 |
|---|---|---|
| M-01 | 一 stack 上限100U/s、同 port/edge 容量≥100B/s；两条 η=1 的 100B 子流 | 每流50B/s，各2s；所有同时刻栈总 `Σu=100` |
| M-02 | 两个互不共享 stack/port/edge，各100U/s、两个100B子流 | 两者各100B/s、1s；不得按 N_active=2 强行各50 |
| M-03 | 一 stack R=200U/s，一笔100B 1:1 拆到两个各100B/s端口且互不共享 edge | 两子流各50B，各0.5s 完成；端口 `D_{s,p}` 汇总为100B；不是每端口各100B |
| M-04 | 同一限制为100U/s，两条流η=0.5和η=1，各在服务域推进50U | 真正已传分别25B与50B；取消按 flow 归账，统计总75B，不可按全局比例平摊 |
| M-05 | 两个读写混合请求同 stack，port/edge 很宽 | 读+写 `Σu≤R_s`，不因不同方向获得两份 stack 上限 |
| M-06 | NoC B64 链路有效5.6e9B/s，端口/栈无限宽（合成测试） | 路由经过同一 directed edge 的 `Σm v≤5.6e9`；广播出边按边一次收费 |
| M-07 | 同 tick 某 stage 达到终点且收到 cancel；另一个尚未完成并取消 | 完成者一次 terminal/全 D，未完成者只付已传 D，过期 generation 无回调副作用 |
| M-08 | 缺 calibration/map 时调用 `enumerate_resources` 与 `compile_timed` | 前者有候选但无时长；后者返回结构化 missing，不能出现假最快 plan |
| M-09 | 宏组聚合两层 Norm，与逐层两次 Norm 生成相同实例 ID 集合 | 两次 launch/每层一个首访延迟；不因 GraphBatch 合并而漏计 |
| M-10 | 校准导入器 raw=100GF/s, η=0.7 与已有效=70GF/s | 统一 effective=70GF/s；相同 BF16 工作时长，运行时不二次乘 η |
| M-11 | Core 网格 `Nx=8,Ny=11`、HBM n=1/2 | Core group 必须保留 8 列和 11 行的各向异性整轴候选；n 变化不得更改 Nx/Ny 或可用 group 尺寸 |
| M-12 | Excel raw `AV`、原始 double 总数 | 严格键匹配，未舍入时每组/总计误差小于1e−6μs |

**字段、映射和配置测试门禁 F-01～F-12**：F-01 规范指纹 `1` vs `1.0` 等价；F-02 重复 `(stack,port)` 拒绝；F-03 被引用端口无路由拒绝；F-04 物理摆放侧/入口侧不符拒绝；F-05 缺效率和首访延迟拒绝 `target_timed`；F-06 η 超出 `(0,1]` 拒绝；F-07 两个独立栈经共享 PHY 争用；F-08 所有缺校准配置均返回正确的结构化错误；F-09 无映射的 `resource_only` 仍可返回静态资源；F-10 一条流完成后将剩余资源速率重新分配；F-11 禁止 YAML 示例重复键；F-12 YAML/JSON 共有硬件字段一致。数学/配置测试与五仓集成回归应分开执行、分别记录结果。

合成 fixture 使用明确给定的 B/s 参数进行数学和事件验收，不得将其误作目标料号的实测吞吐、PHY 或首访延迟。

须继续保留既有 `test_face_scheduler.py`、`test_weight_passes.py`、`test_train_machinery.py`、`local_hbm_bandwidth_model_test.cc`、LRU `local_hbm_model_test.cc`、parser/NodeStore、completion/watch/alarm cancellation/online compaction 测试。参考 profile 的时间快照作为独立测试夹具，不为通过测试而改动策略规则；目标 BF16 能力补齐后才增加真实数值时间与策略排序回归。

### 10.3 动态扫参结果、报表与实验标识

建议架构 DSE 先单独保存一组**实际生成**、通过各自工程条件筛选的 resolved 设计，再由单 Die 仿真器按实例运行：

```text
results/
  dynamic_core_grid/mesh_die_hbm3e_dynamic_v4/<dse_run_id>/
    search_profile.json                     # 200..800等搜索条件，仅此处声明
    candidate_designs.jsonl                 # 每行一个Nx/Ny/W/H/n，数量由真实枚举决定
    pareto_compute_vs_capacity.csv
    core_first_vs_wafer_first_validation.json
    cases/<hardware_fingerprint>/
      resolved_die_model.json
      derived_die_module_wafer_resources.json
      resource_report.json
      runtime_stats.jsonl                   # 仅在真正运行时生成
      validation_results.json               # not_run/passed/failed
  sensitivity/<changed_search_assumption>/<dse_run_id>/
    candidate_designs.jsonl
    pareto_delta.csv
```

候选必须输出 `schema,hardware_fingerprint,Core64 SHA,core_type,SRAM,noc_width_bits,Nx,Ny,W_die_mm,H_die_mm,A_die_mm2,px,py,ex,ey,n,placement_mode,N_HBM,row_width_mm,hbm_row_overhang,W_module,H_module,wafer_cols,wafer_rows,N_module,bump,D2D,HBM容量和理论带宽,precision/ingress status,provenance`。PE INT4/Vec FP32 理论峰值与任何已校准 BF16 执行结果严格分列。

`200–800` 主窗口、`150` 下界、长宽比、外缘预留和各类效率敏感性实验都须在 `search_profile.json` 版本化；动态搜索候选数量由枚举边界和物理约束决定。不同实际形状、端口映射、算法或校准版本必须有不同硬件指纹。NoC、PHY 与 HBM 服务率缺校准时只输出资源/必要下界，不能生成真实 BF16 延迟或在线策略最佳排名。

### 10.4 构建与运行验收准入

完成五仓库集成的每个目标分支均须满足下列要求：

1. **构建与 ABI**：`extern/die_model/` 可离线构建并链接；C++ 和 Python C ABI 版本一致，拒绝不兼容的硬件 Schema、配置字段和计划类型；所有仓库使用相同 `hardware_fingerprint` 和 Core 表哈希。
2. **配置和资源**：动态 Nx/Ny/W/H 的 Core 网格、HBM overhang、Wafer/D2D 参数化公式和 §3.3 的代表示例必须复算一致；输入 Schema 及 HBM 字段必须明确，且单位只换算一次。
3. **执行与事件**：在合成栈、端口、链路和共享 PHY 夹具上执行 §10.2 的 M/F 系列回归；必须验证同一 Sys Tick、动态速率重排、读写竞争、过期 generation、取消、单次 COMP 完成与实际字节守恒。
4. **五仓行为**：WSCLLM、WSCLLM-LRU、FACE、FACE-LRU、JOINT 均应执行本地计算、KV/restore/pool/remote、watch/generation、P2P/collective 及各执行 profile 的独立回归；不得在未显式实例化时改变 `rank_count` 或通信图消息语义。
5. **性能能力**：缺少已校准的 BF16/SFU、HBM 有效访问效率、访问延迟或 HBM→NoC 注入映射时，`resource_only` 可用，`target_timed` 必须结构化拒绝；接入校准后才允许用相同硬件描述符和相同事件规则比较 Python oracle 与 C++ 执行。
6. **验收证据**：每个仓库记录构建所用代码 commit、输入配置及哈希、C ABI 版本、实际执行的测试及通过/失败状态、模拟输出和工程性能参数来源。`not_run` 不得标为 `passed`；合成数学测试通过不代表真实芯片或五仓事件集成已通过。

只有全部适用验收条件完成并有对应证据，才能把该目标分支标记为已实现且已通过集成验收。
