# 单 Die 性能模拟器设计规格：HBM3E 动态 Die v4 架构、模型与接口

本规格定义 `mesh_die_hbm3e_dynamic_v4` 单 Die 硬件资源模型、整数 Core 网格驱动的动态长方形 Die 几何、HBM3E 双排模块、计算/访存成本、资源级事件调度，以及 Die/Wafer/D2D 预算。目标器件为 Micron HBM3E 36 GB 12-Hi `MT65B18G16120A00QH-92:A`。架构搜索生成 Die 宽高和整数 Core 阵列；单 Die 仿真器解析并校验一个具体硬件实例。HBM 排宽大于逻辑 Die 宽度时，按模块外廓公式生成占位，并记录物理待验证状态。

模型输入为已切分到单个 Die 的本地任务，输出 Core、HBM、NoC 和 Wafer 的几何、容量、运算资源、算子分块、实际访存字节量及带宽约束。只有目标精度计算单元的服务率、HBM→NoC 接入/路由、必要首访延迟和资源调度参数均可求值时，才计算带证据等级的模型时间；否则返回静态预算、必要时间下界和结构化缺参原因，不生成未经依据支持的时延。

采用 WSC-LLM 的局部分块思想和 FACE 的片上 attention 复用及流水思想。模型采用容量约束的分块成本估计与资源级事件调度，不包含逐周期 PE/PHY、完整 ASTRA-sim、跨晶圆任务调度或封装布线仿真。INT4 PE 峰值与理想 FP32 Vec 峰值按各自运算精度分别报告；BF16 GEMM 和特殊函数的时间须经过独立能力校准。`tx8_reference_v1` 为隔离的参考校准路径，不覆盖目标硬件参数。

## 1. 仿真范围与数据口径

### 1.1 仿真到哪里

一次调用处理一个 die 上的一份本地算子图，所有输入已在本地 HBM，所有前置跨 die 通信已完成。输出硬件/映射可行性、已知字节量、SRAM 需求和 NoC 负载；**仅在所需执行精度的性能参数齐备时**输出本地执行时间和瓶颈。

这里的 die 单元包含一个 compute die 及其配套 HBM。compute die 内部为二维 core mesh；每个 core 包含矩阵单元、向量单元和私有可管理 SRAM。跨 core 数据传输经过 NoC。

| 建模范围 | 不包含的微观或系统功能 |
|---|---|
| 每核 SRAM 容量、核数、矩阵/向量算力、HBM、NoC | PE 内部流水级、寄存器分配、指令集、逐周期执行 |
| 分块后实际计算量、数据复用、尾块、核并行度 | SRAM bank 冲突、DRAM bank/row、控制器内部状态 |
| 算子依赖、必要的归约、片上融合、有限缓冲流水 | 报文/flit、路由器队列、信用流控、缓存一致性 |
| 本地权重/KV 容量检查、prefill/decode 工作量 | 请求到达、排队、PD 服务调度、KV 迁移和远程 HBM |
| 单 die 的延迟和吞吐，供上层模型调用的接口 | D2D/W2W 时延、collective、wafer 铺排、instance 搜索 |

未显式建模的微观机制由声明的工程效率或能力模型抽象，不作为用户输入。不同精度的性能参数须独立确认，不能通过效率系数推导缺失的算力。固定 1 GHz 频率参与 NoC 位宽到带宽的转换。

### 1.2 TX8 参考数据的适用范围

TX8 参考路径采用 `TX8最终查看版(1).xlsx` 的历史记录、解释和工程估计。相关数据仅用于 `tx8_reference_v1` 的校准或参考核对，不作为 HBM3E 目标硬件的实测数据：

| 数据 | 本项目中的身份及用法 |
|---|---|
| TX8 记录 RMSNorm 246 μs | 提供的计时记录；按完整单次 RMSNorm 解读，作为当前最可用的拟合锚点，计时边界仍是采用的解释 |
| TX8 记录标为 SiLU 的 3363 μs | 时间数值来自原始记录；将其解释成整个 FFN 是假设，只作为条件性约束 |
| GEMM、RoPE、softmax、Hadamard、res add 的 8–15 μs | 计时记录，但无法作为表列完整设备算子的可靠执行时间；保留来源，不用于吞吐拟合 |
| Excel 重建的 13 个计时组、FFN 子项分配 | 推断/模型派生值；用于参考方案核对，不作为新增实测样本 |
| block 11897.60 μs、attention 7304.96 μs | 推断方案合计，没有独立 block/attention 实测支持 |

**不得把模型派生的参考值用作同一模型的独立实测验证集**。拟合训练点、条件性解释和派生值必须在报告中分列。

## 2. 输入与扫描空间

### 2.1 动态几何、硬件输入和上层搜索边界

**系统分工**：上层架构 DSE 探索 `core_type × SRAM × noc_width_bits × (N_x,N_y) × n_hbm_per_row × placement_mode`、必要的 Die 边界扩展及 Wafer 网格；单 Die 仿真器接收、校验并执行某个已解析的 `(W_die_mm,H_die_mm,N_x,N_y,...)` 实例。`compute_area_mm2=W_die_mm×H_die_mm` 为连续派生量。输入必须提供能够确定 Core 阵列与 Die 宽高的几何字段，仅有面积数值不足以唯一确定物理布局。

**必需几何主键**为 `(hardware_schema_version,core_table_revision/core_table_sha256,core_type,sram_mib_per_core,noc_width_bits,core_columns_nx,core_rows_ny,die_width_mm,die_height_mm,core_pitch_x_mm,core_pitch_y_mm,die_extra_width_mm,die_extra_height_mm,hbm_stacks_per_row,hbm_placement_mode,hbm_geometry_revision,hbm_injection_map_id,model_algorithm_revision,calibration_revision)`，与 SKU、NoC/端口图、全部固定假设一起规范化 hash。`compute_area_mm2` 是派生的**连续计算面积数值**，不允许覆盖 `die_width_mm×die_height_mm`。

| 字段 | 动态建模规则 | 用途 |
|---|---|---|
| `hardware_schema_version` | `mesh_die_hbm3e_dynamic_v4` | 目标硬件配置的版本化标识；不同硬件与参考 profile 具有各自的缓存身份 |
| `core_type` / `sram_mib_per_core` / `noc_width_bits` | `small/large`；8 档 MiB；`64/128/256/512` bit | **仅 Core 资源保留 64 行固定表**，不依赖 Die 总面积 |
| `core_columns_nx`, `core_rows_ny` | 正整数 `N_x,N_y ≥1` | 实际二维 `N_x×N_y` Core mesh；不强制正方阵列 |
| `die_width_mm`, `die_height_mm` | 有限、正的 W/H，允许 `W≠H` | 独立输入的已解析逻辑 Die 外形；`compute_area_mm2=W×H` 派生 |
| `core_pitch_x_mm`, `core_pitch_y_mm` | DSE 示例均为 0 mm | 可扩展物理中心距；不可默认为实测布线余量 |
| `die_extra_width_mm`, `die_extra_height_mm` | DSE 示例均为 0.10 mm/轴 | 轴向**总预留**；例子不等同于两侧分别各 0.10 mm |
| `hbm_stacks_per_row` | 整数 `n≥1`，DSE 默认枚举 `1..14` | 每模块 HBM `N_HBM=2n`；14 是 DSE 枚举上界，不是仿真器固定颗数 |
| `hbm_placement_mode` | `same_side_two_rows` / `two_sides_one_row_each` | 不同 ingress/走线映射；等外廓不意味着等延迟 |
| `hbm_injection_map_id` | 端口图缺失时 `TO_BE_DEFINED` | 无校准不能计算目标 HBM→NoC 可靠时长 |

**上层默认 DSE 筛选条件**来自《架构建模新.docx》§2，而非单 Die 计算引擎的硬编码支持域：主搜索 `200≤A_die≤800 mm²`、下界敏感性 `150≤A_die<200`、曝光场 `(W≤26,H≤33)` 或旋转 `(W≤33,H≤26)`、长宽比≤2、每轴总外围预留 0.10 mm、Core pitch=0、每排 `n∈1..14`、D2D 双向余额非负。搜索器用 `Core-first` 与 `Wafer-first` 两条生成/复核路径；输入外部已定型的 Die 可在其他明确 profile 下验证，不由仿真器替 DSE 强制执行特定优化面积窗口。面积、曝光场和模块/端口可行性必须分别返回状态，不能把工程搜索窗口说成物理普适限制。

Core 资源表由 8 档 SRAM、2 种 Core 类型和 4 档 NoC 位宽构成，共 64 条；Die/HBM 候选数量由上层 DSE 的枚举范围及几何可行性决定。`N_x,N_y` 与 W/H 须相容；若逻辑 Die 不能容纳指定 Core 阵列，返回 `invalid_hardware/core_grid_exceeds_die`。HBM 采用规范字段 `hbm_stacks_per_row`、`hbm_placement_mode` 并依据实际二维几何验证，不从面积或别名字段推断布局。

### 2.2 工作负载输入

| 输入 | 定义 |
|---|---|
| `batch`、`q_len`、`kv_len` | 当前本地 batch、query 长度、有效 KV 长度；decode 通常 `q_len=1` |
| `hidden`、`ffn_hidden`、`q_heads`、`kv_heads`、`head_dim` | 直接给出算子维度，不根据模型名称覆盖 |
| `layers_local` | 该 die 运行时执行的层数；单 block 时为 1 |
| `causal` | 是否应用因果 mask；决定可跳过的完整 attention tile |
| `local_graph` | 可选的、上层已切分好的本地算子图；它优先于完整 block 模板 |
| `operation_dtype`、`accumulator_dtype`、`op_count_unit` | 分别声明 BF16/INT4/FP32 和操作量单位；禁止隐式 PE 跨精度换算 |

默认参考形状为 `B=1, S_q=S_k=1024, H=4096, F=14336, H_q=32, H_kv=8, d=128`。 **全篇实现变量命名约定**：将工作负载 batch 记为 `batch`，将 NoC 位宽记为 `noc_width_bits`，将 HBM 每排颗数记为 `n_hbm_per_row`；凡用于 `B×S_q`、`B×H_q` 的 `B` 均指 `batch`，凡用于 B64/128/256/512 的 `B` 均指 `noc_width_bits`。生成代码必须用长字段名，禁止在同一函数用含义不同的局部 `B`。本地 TP 分片可能不满足完整模型的维度关系，此时按 `local_graph` 的实际张量和依赖计算，不能只把完整 block 时间除以 TP 数。

分块大小、core group 和流水缓冲由内部映射器选择，不要求用户手填。INT4 峰值不意味着现有 BF16 输入自动获得有效 GEMM 延迟。prefill chunk、decode batch 和 KV 长度可作为工作负载扫描项；不实现联合 PD 服务调度。

## 3. 从输入生成硬件资源

### 3.1 核配置表

**Core 资源定义**：采用《架构建模新.docx》§1.1～§1.3 的 64 行 Core 参数。SRAM 以 **MiB** 计，HBM 以十进制 **GB/GB/s** 计，Core 基准频率为 1 GHz。7 nm SRAM 尺寸来自 22 nm NVSim 位单元面积比例估计；Vec 峰值来自 AraXL FP64 基准下的**理想 FP32 SIMD 等效**；PE 面积及 INT4 MAC 吞吐来自 NVDLA CMAC 架构估计。以上参数不是 7 nm 实际流片测量值。

**部件级参数与工艺映射（固定参数）**：SRAM 来自 NVSim 固定 wordwidth=1024 bits 的 **22 nm** 尺寸，并按 SRAM 位单元面积比、保持利用率的近似假设映射到 **7 nm**。完整的来源尺寸如下；容量使用二进制 MiB。

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
- **名义计算资源比例**：硬件模型以 **PE INT4 GOPS : Vec FP32 等效 GFLOPS ≈ 10:1** 作为混合精度峰值安装目标，使用 `N_pe=round_half_up(19.777*N_vec)`、0.5 向上取整。下表 `PE/Vec 名义峰值比` 是**架构参数表给出的名义比值**；它不表示同一精度的 10:1 加速，也不能用来校准 BF16 或 SFU。


Core 资源的局部主键 `(core_table_revision, core_type, sram_mib_per_core, noc_width_bits)`，**共 64 行**。完整 Die 硬件主键另绑定 §2.1 的动态二维网格 `(N_x,N_y)`、W/H、预留、HBM 双排几何和 ingress map；Die 候选数量由 DSE 搜索空间和几何筛选结果决定。不同 NoC 位宽对同一 `(core_type,SRAM)` 保持相同 `L₀`，而 PE/Vec 数量按 64 行表变化。

每核 `L₀` 对同一 `(core_type, SRAM)` 在四档 B 下固定；位宽增大只调整 Core 内 NoC/Vec/PE 的占用，不放大 Core 外形，也不因此改变 Die 核数。以下 64 行为运行时 Core 资源表。若导出机器可读 CSV，须验证 64 个唯一主键、全部字段以及内容 SHA-256，并与 `core_table_revision` 一起纳入硬件指纹。表值不一致时拒绝加载。几何生成公式可用于重建和检验，不得以简单面积比近似代替整数排布。

**完整 64 行 Core 配置表（含 Vec/PE 排布）：**

排布写法 `横/竖 K×R/C` 指相应矩形的 Vec 整数列×行数量和 PE 列数；大核可有 `下/上` 两区。`large/1.75 MiB/B64` 某区基于印刷精度的剩余横宽仅约 0.0000288 mm，此为理想整数铺排可行，不等于已经验证物理单元间距、布线、制造和时序余量。

| B (bit) | 核型 | SRAM (MiB) | Vec lanes | INT4 PE 数 | Vec FP32 等效 GFLOPS/core | PE INT4 GOPS/core | 固定 L₀ (μm) | Vec/PE 排布 | PE/Vec 名义峰值比 |
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
# 两种核型均以 B0=256 bit 为基准
B ∈ {64,128,256,512};  g(B)=0.20+0.80*(B/256)
A_noc_0=0.15*L0_mm^2
A_noc(B)=A_noc_0*g(B)             # 6%、9%、15%、27% 的固定 Core 面积
L_core(B)=L0_mm                   # 位宽改变不改变 Core 外形
S_compute(B)=L0_mm^2-a_mm*b_mm-A_noc(B)
W_min_mm=N_x*L0_mm+(N_x-1)*core_pitch_x_mm+die_extra_width_mm
H_min_mm=N_y*L0_mm+(N_y-1)*core_pitch_y_mm+die_extra_height_mm
require die_width_mm>=W_min_mm and die_height_mm>=H_min_mm
N_core=N_x*N_y
compute_area_mm2=die_width_mm*die_height_mm   # 连续派生值，不是输入档位
if N_core==0: invalid_hardware("no_core_fits")
PE_peak_die_int4_gops=N_core*(2*N_pe)
vec_peak_die_fp32_equiv_gflops=N_core*(3.9554*N_vec)
SRAM_installed_bytes_per_core=round(sram_mib_per_core*2^20)   # 档位均精确为整数 byte
SRAM_usable_core_bytes=floor(0.90*SRAM_installed_bytes_per_core)
link_peak_bytes_per_second=B/8*1e9  # 软件近似：1 word/周期/有向链路
```

`a×b` 取 7 nm SRAM 的八档长短边：`a(mm)={0.40662,0.81324,1.21986,1.62648,2.03310,2.43972,2.84634,3.25296}`，`b=0.13925 mm`。小核固定 `L₀=a`，下方计算带 `H=L₀-b, W=L₀-A_noc(B)/H`；大核旋转 SRAM，256-bit 基准 `L₀=[a+sqrt(a²−0.6ab)]/0.3`，`Δw₀=L₀-b, A_noc,0=aΔw₀`：当 `g≤1`，下区 `(L₀,L₀−a)`、上区 `((1−g)Δw₀,a)`；当 `g>1`，下区 `(L₀−(g−1)A_noc_0/(L₀−a), L₀−a)`、无上区。Vec 依区选择横/竖放置并按整数列/行装入，`N_pe=round_half_up(19.777*N_vec)`，全 Core 只取整一次；在全部容量/矩形/PE 列约束成立时最大化 `N_vec`，平局取最小横向剩余面积。执行期优先使用源表，不以简化面积比替代列级约束。

`N_core=N_x×N_y` 是**给定合法整数网格的实际名义安装数**；最小 W/H 包含 Core pitch 和每轴外围总预留。`A_noc(B)` 已把 Core 内互连与 Die 内 Core mesh 的等效面积统一分摊，不能按两个独立 NoC 再扣一次。`0.90` SRAM 系数、链路持续速率和 XY 路由属于仿真器工程假设，不是已验证的物理测量参数。

**交叉校验**：`small/0.25MiB/B256` 是 `L0=406.62 μm, N_vec=2,N_pe=40,PE=80 INT4 GOPS`；`large/0.75MiB/B256` 是 `L0=7990.68 μm,N_vec=2135,N_pe=42224`；若给定 `large/0.75MiB`、`N_x=3,N_y=2`，则名义 6 核且必须验证 W/H 能容纳 `3×2` 个边长 `L₀=7.99068 mm` 的 Core（再加预留）。NoC 位宽 B 不改变此网格核数，但可减少每核 PE/Vec。

### 3.2 Micron HBM3E SKU、动态两排布局、容量与端口定义

**HBM 单颗部件参数**：目标 SKU `MT65B18G16120A00QH-92:A`（Micron HBM3E 36 GB、12-Hi），单颗 `36_000_000_000` byte、接口 `1024 bit × 9.2 GT/s`，理论峰值 `1_177_600_000_000 B/s = 1177.6 GB/s`，二维封装 `10.98×10.98 mm`，厚度 `0.78 mm` 仅记录不计入二维高度。HBM gap、Die-to-HBM 及边框 gap 暂取 `0.15 mm`（工程假设）。实际持续吞吐、PHY 与热特性尚未校准。

两排实物布局中，`n=hbm_stacks_per_row≥1`，每排 n 颗，总 HBM `2n` 颗；`same_side_two_rows` 同侧两排、`two_sides_one_row_each` 对侧各一排。两种模式共用二维**矩形外廓近似**，但必须有不同版本化入口/路径映射（不能将同侧第二排当成另一条免费相对侧通路）。由**实际输入的矩形计算 Die W/H**派生模块：

```text
Nx,Ny = core_columns_nx,core_rows_ny
W_min = Nx*L0+(Nx-1)*core_pitch_x_mm+die_extra_width_mm
H_min = Ny*L0+(Ny-1)*core_pitch_y_mm+die_extra_height_mm
require Nx>=1 and Ny>=1 and W_die>=W_min and H_die>=H_min
A_die = W_die*H_die  # mm²，连续值
n = hbm_stacks_per_row                 # 正整数；DSE 默认探索 1..14
N_HBM = 2*n
W_HBM_row = n*10.98+(n-1)*0.15
W_module = max(W_die,W_HBM_row)+2*0.15
H_module = H_die+2*10.98+4*0.15      # H_die+22.56
A_module = W_module*H_module
HBM_capacity_bytes = N_HBM*36_000_000_000
HBM_theoretical_peak_Bps = N_HBM*1_177_600_000_000
```

**HBM 排宽与模块外廓**：默认 `row_fit_policy=allow_module_overhang`。当 `W_HBM_row>W_die` 时，设 `hbm_row_overhang=true`，由 `W_module=max(W_die,W_HBM_row)+0.30` 生成外廓，并标记 `physical_layout_unverified`；该标记表示封装、PHY、功率及布线尚需工程验证。针对特定封装约束可使用独立 `row_fit_policy=required`，将排宽限制为 `W_HBM_row≤W_die`。这两类实验的规则与搜索配置须显式记录。

**参数化示例（不是固定模板或最优性证明）**：`large/0.25MiB`、`L_c=2.56355 mm`、`N_x=8,N_y=11`、`p_x=p_y=0`、`e_x=e_y=0.10 mm`，得到约 `W_die=20.6084 mm,H_die=28.29905 mm,A_die≈583.2 mm²`。若每排 n=2，HBM 排宽为 `22.11 mm > W_die`，模块宽约 `22.41 mm`、高约 `50.85905 mm`；**该方案的模块外廓依据 HBM 排宽扩展，并标记物理待验证**。W/H 扩展至满足面积、bump 或系统目标所需的更多边界时，Core 数不因额外留白自动增加；必须显式改变 `N_x/N_y` 才改变 Core 安装数。

**上层 DSE 搜索协议**：先枚举 64 行 Core、整数 `N_x,N_y`，得到最小 Die W/H；当低于配置面积下界或 Bump-D2D 最小面积时，源 Word 的复算示例只生成“保持宽度扩高、保持高度扩宽、两轴同比扩大”三类边界解（不宣称穷举所有连续外形）。再枚举 `n∈1..14` 与两种摆放模式，依次验证曝光场旋转、长宽比、面积窗口、D2D、HBM 模块宽高和 Wafer 网格。Wafer-first 另从模块 `M_x×M_y` 反推 Die 可用 W/H，作为相同可行集合上的核对路径；**不改变单 Die 运行器输入/执行语义**。

**默认驻留、物化与跨层计量规则（适用于所有映射）**：权重默认存放本地 HBM，执行到相应 layer 时按实际消费和缓存/重复加载策略发出 HBM→SRAM 加载，不假定多层权重免费预加载或可永久驻留 SRAM。每个未显式声明并通过缓冲容量、生命周期和依赖检查的**跨宏组边界**，生产者默认将该中间张量写回 HBM，消费者默认从 HBM 重新读取；对应一写一读分别计入 `actual_bytes`，即使张量容量已在 HBM 记过账也不能漏掉传输字节。**只有明确的 fused_group_id、producer/consumer 映射、存活 SRAM 缓冲以及执行先后条件同时成立时**才消去相应的真实写回/重读；宏组内部 tile 驻留可依 §5 的显式融合/广播规则省搬运，不得把融合能力扩展到未指定的跨宏组边。每层权重的首次 HBM 加载、重读和缓存命中均单独记账，既不把每次权重永久重载，也不默认为零重载。只有本地所有层在**实际 shape、dtype、映射、缓存/驻留策略、HBM 访问、执行路径、首访/启动费用条件均一致**时，才允许 `T_local_iteration=layers_local*T_single_layer`；否则逐层建立依赖图与读写量求和。此规则修改访问次数，**不修改**物理权重/KV 元素的 byte 定义或原通信消息大小。

**容量可行性（必须同时检查模块总量和每颗物理落点）**：`weights + KV_cache + peak_live_workspace + explicitly_reserved_bytes <= HBM_capacity_bytes` 只是必要的**聚合条件**，不是足够条件。各项先按真实驻留时刻和物理 byte 计费；模型权重常驻层数、本地 TP/PP 分片以及多个同时驻留任务须在同一时间轴合并。**当地址范围、条带或固定颗粒分配已指定时**，进一步对每颗 `stack_id` 检查 `resident_bytes[stack_id,t] <= 36_000_000_000`，并覆盖每颗的保留区域、权重、KV、工作区和未释放中间结果；不能把 A 颗粒的空余空间用于容纳 B 颗粒上已固定映射的地址。若仅聚合容量通过，但缺少可求值的 `address_range/stripe_rule`，报告 `capacity_status=aggregate_upper_bound_only`、`missing_residency_mapping`，不得将其标为**已验证的逐颗驻留可行**。例如两颗各 36 GB，固定驻留量为 `40 GB + 20 GB`，即使聚合仅 `60 GB < 72 GB`，仍因第 0 颗超限而返回 `infeasible/hbm_stack_capacity_exceeded`；若确实允许重排，必须先给出**可实现的重新映射**再复算，不能暗中平均分摊。容量映射与 NoC 注入映射是两种独立检查，二者都有效才可把该映射计入完整执行可行性。示例 BF16 block 公式（H、F、H_kv、d、B、S_k 随 local graph 改变）：

```text
weights = 2*(2*H*H + 2*H*(H_kv*d) + 3*H*F) bytes
kv_cache = 2*2*B*H_kv*S_k*d bytes
```

Norm gamma/偏置、额外层、活跃中间张量、工作区峰值另按真实驻留图计费；`KV` 的重复 query head 不重复计算 K/V 存储。聚合空间不足返回 `infeasible/hbm_capacity_exceeded`，单颗映射超限返回 `infeasible/hbm_stack_capacity_exceeded`，尚未给出逐颗驻留映射返回 `missing_residency_mapping`（不等于数学上证明容量不可行）；不虚构远端容量、压缩率或自动换入换出。**更换 HBM 容量会改变可行性，但不自行改写 KV/通信 payload 的 byte 定义与原调度策略**。

**地址条带与注入是分开的映射**：每个实际 HBM `stack_id ∈ [0,2n)` 有 `physical_row`, `column`, `placement_edge`, `address_range/stripe_rule`, `controller_id`, `injection_port_ids`；地址映射须可将**驻留分配和每次传输**一致地落到具体颗粒，支持逐颗容量和访问流两种独立记账。均匀条带不能仅给 `actual_bytes_per_stack` 而忽略各颗实际驻留容量；也不代表获得了足量 NoC 入口。两侧模式须区别上/下等物理边沿的注入；同侧两排须定义两排的控制器共享关系、接入同一侧的具体 ports、带宽竞争和可能的多跳代价，**第二排不产生免费额外边沿**。端口只能映射到 `0<=x<N_x,0<=y<N_y` 的合法矩形 mesh Core、给定有向边/边缘入口；具体物理映射来源仍缺失时置 `TO_BE_DEFINED`，仅能报告理论外存资源与注入必要条件，而不能输出声称有效的 HBM 满带宽执行时间。

### 3.3 NoC 与 HBM 注入的**三个不同瓶颈**

Core **面积**模型采用 §3.1：`g(B)=0.20+0.80(B/256)`、`A_noc(B)=0.15*L₀²*g(B)`，`L₀` 固定；该 NoC 预算已合并 Core 内通信与分摊的 Die 内 Core mesh。运行时另外采用双向二维 mesh、`1 GHz`、每条**有向**链路每周期传 `B` bit word 的软件近似，`noc_link_raw_Bps=noc_width_bits*1e9/8`，`noc_link_effective_Bps=eta_noc*noc_link_raw_Bps`，`eta_noc=0.70`；XY 路由和广播共享有向边同样是软件近似，并非 Micron 或工艺 PHY 测量。

**必须分别受限**：① `HBM_chip_peak=N_HBM*1177.6e9 B/s`（芯片聚合理论天花板）；② 真实/假设端口总数、端口带宽与每颗 HBM→端口的映射（`hbm_injection_ports`, `ingress_Bps_per_port`, `port_to_core_mapping`, 读写复用与仲裁）所形成的**有效入口上界**；③ 按路径、方向及共享负载合并后的有向 NoC 链路流量 `D_e` 和容量 `noc_link_effective_Bps`。任何单一路径也不得大于沿途最紧的入口或链路。软件需按实际 `stack_id`、`port_id`、`edge_id` 汇总流量，报告瓶颈和偏斜，不能从芯片峰值自动导出 mesh 可接纳的带宽。

| B bit | 单条有向链路原始 GB/s | `eta_noc=0.70` 有效 GB/s | **一颗** 1177.6 GB/s HBM 满峰值所需的最少等效并行链路数（仅必要条件） |
|---:|---:|---:|---:|
| 64 | 8 | 5.6 | 211 |
| 128 | 16 | 11.2 | 106 |
| 256 | 32 | 22.4 | 53 |
| 512 | 64 | 44.8 | 27 |

这些数是 `ceil(1177.6/(0.70*(B/8)))`，只说明**单颗峰值不能直接进入一条 B64/B128/B256/B512 Core 链路**；实际可同时使用多少合法边缘端口、链路与控制器、布线拥塞及 PHY 效率均未给出，不可当作已具备的端口数量。若某 mapping 假定一条外存入口接入一条 B256 有向链路，其名义入口即受 `22.4 GB/s` 约束，不是 `1177.6 GB/s`。

对于已给定端口图、数据流与路由，阶段的 NoC 资源下界为 `t_NoC_lower_bound = max_e(D_directed_edge_bytes[e]/noc_link_effective_Bps)`；`D_e` 在同一执行阶段按有向边合并所有流量。入口的必要下界 `t_ingress_lower_bound >= max_p(D_ingress_port_bytes[p]/ingress_effective_Bps_p)`，其中端口分子严格是物理 byte，**不得**使用 `D/eta_HBM` 等效服务量，同时还受 HBM 芯片、各颗访问模式与共享控制器限制；这只是同一数据传输的一组互相约束，不能在内存与 NoC 各造一份独立可用总线。对缺少 `hbm_injection_map_id` 或端口速率证据的配置输出 `missing_ingress_mapping`/`missing_bandwidth_calibration`，允许输出乐观上界/必要时间下界（标 `hypothetical`），**不允许声称经过验证的 `effective_mem_bw` 或精确目标时延**。

单 Die 时间不包含 D2D/W2W 延迟；Die 间 NoC 的额外 10% 是 §10 的模块**面积**预算，不能被转化成 `eta_noc`、免费 D2D 通道或 Core 计算减速。`noc_width_bits` 只描述 Die 内 NoC，不能代写上层跨 Die `network.yml` 的链路位宽。

### 3.4 计算精度与目标时间是否可报告

`PE_peak_int4_gops` 只表示 **INT4 MAC×2 次整数运算/周期**；`vec_peak_fp32_equiv_gflops` 只表示理想 FP32 SIMD 等效。LLM 工作量中的 `BF16×BF16→FP32` GEMM、QK、PV、FFN 的 `2mkn FLOP` **不能除以 INT4 GOPS**，向量 `6/12/15 eq-op/element` 也不能未经转换直接除以 FP32 FMA GFLOPS。当前硬件资源参数未定义 BF16 PE 吞吐、BF16 指令映射/转换成本或 SFU 实测服务率。TX8 参考的效率 `0.907955...` 不能填补这个缺口。

目标配置引入强制字段 `operation_dtype`、`accumulator_dtype`、`op_count_unit`、`physical_peak_unit`、`supported_precisions`、`pe_peak_bf16_flops`（源默认 `null`）、`effective_vec_eq_ops_per_second`（目标未校准时 `null`）、`performance_model_provenance`。资源库存查询可始终输出 INT4/FP32 名义峰值；数值时间必须按 dtype 查找**相容且有来源的**每核性能参数。缺少 `BF16` 吞吐或必要的向量吞吐/转换模型时，返回结构化 `unsupported_precision` 或 `missing_performance_calibration`，允许输出形状/工作量/容量/NoC/HBM 的已知预算，但**不得给出伪精确 target 总时长、性能排序或系统策略 compute service time**，亦不得无声明地切换为其他未经校准的 Roofline 模型。

未来若确立实际 BF16 服务率，必须单列其单位、测试/假设证据、硬件配置适用域、是否包括降精度/累加/布局变换、对应效率；用 `t_PE=F_bf16/(eta_bf16*P_bf16_flops_per_core)`。INT4 工作负载同样需要有效量化/解量化路径和对应整数算子定义后才能使用名义 `2*N_pe` GOPS。Vec 特殊函数须有单独 `eq-op/s` 服务率或明确标为敏感性外推；**效率系数不能充当精度转换因子**。

为了不混淆目标与参考，`tx8_reference_v1` 使用 TX8 自己的 BF16/FP32 抽象峰值、DDR/DRAM 流量模型和TX8 参考拟合系数；`mesh_die_hbm3e_dynamic_v4` 仅由64 行目标硬件峰值及**另外提供的目标 dtype 校准**生成数值计算阶段。未校准或 ingress 未定义时 `estimate()` 必须返回“资源可算、时间不可算”的类型化结果，不能给出 `T_block=0` 或 `∞` 冒充正常时间。

**完整时延的机器可判定准入状态**（只对目标路径，不改变 TX8 reference 的历史公式）：

1. `dtype_supported` 且本图所有出现的运算有相容的 `P_PE_dtype`、Vec/SFU 有效服务率和足够的效率/换算依据；否则 `unsupported_precision`（完全不支持）或 `missing_performance_calibration`（硬件可能支持但未定量）。
2. 对每个实际非零 HBM `flow`，须有通过验证的逻辑地址→`stack/controller/port/route` 映射，并有正的 `controller_Bps`、`port_Bps`、读写 duplex/共享规则；否则分别 `missing_ingress_mapping` 或 `missing_bandwidth_calibration`。无 HBM 访问的工作图不用这些 HBM 特有字段，跨 Core NoC 仍须路由可解。
3. 若至少一个首次非零 HBM 访问触发延迟，`local_mem_latency_s` 必须是非负有限数且带 `measured` 或 `engineering_assumption` 来源；为 `null` 时 `invalid_reason=missing_memory_latency`、所有依赖该执行路径的 `latency_s=null`；零 HBM 字节且没有首次访问时跳过此项。
4. 所有活跃资源服务容量必须有限、严格正、单位为 B/s；每条非零 flow 有有效 `eta_hbm`、访问分类与来源 ID，方向与请求一致，stage 的 `service_units` 可从 flows 逐项复算；共享调度与缓冲依赖有确定算法版本。若任务产生 HBM 传输，还必须验证每颗的真实驻留映射及容量上限、所用读写端口/路径；缺上述能力不得计算完整时间。多项同时缺失时在 `missing_requirements[]` 全量列出，以稳定优先级 `unsupported_precision > missing_performance_calibration > missing_residency_mapping > missing_ingress_mapping > missing_bandwidth_calibration > missing_memory_latency` 选择主 `invalid_reason`；静态不合法原因优先于全部性能原因。
5. 通过以上仅表示**模型可求值**；若有 `engineering_assumption`、历史 TX8 效率外推或未验证 HBM PHY 参数，报告 `timing_evidence_level=engineering_assumption`、`measured_performance_supported=false`。只有对应配置有独立校准/测量证据才可称 `calibrated_model`；不得因为用户填写了数字就声称是 HBM 厂商实测。

## 4. 算子图与两条执行路径

### 4.1 面向用户的输出粒度

输出九个宏组即可。宏组内部的计算与传输由统一成本模型生成，不需要给每个数学小算子伪造独立实测时间。

| 顺序 | 宏组 | 内部工作 |
|---:|---|---|
| 1 | RMSNorm_1 | attention 前归一化 |
| 2 | QKV_proj | Q、K、V 三个投影，分别计工作量 |
| 3 | RoPE_QK | Q 的 32 头和紧凑 K 的 8 头旋转；尺寸随输入变化 |
| 4 | Attention | QKᵀ、scale/mask、softmax、PV |
| 5 | O_proj | attention 输出投影 |
| 6 | Residual_1 | attention 残差相加 |
| 7 | RMSNorm_2 | FFN 前归一化 |
| 8 | FFN | gate/up、SiLU、gate_mul、down |
| 9 | Residual_2 | FFN 残差相加 |

宏组串行执行；宏组内部只在依赖和缓冲允许时重叠。QKV 默认按三个投影顺序执行，避免依赖未知的 fused QKV 实现。block 时间为九个宏组之和，父项与子项不重复相加。

### 4.2 目标架构路径：默认使用片上分块 attention

目标 die 采用 FACE 的主要执行机制：按 head/head-group、query 行和 KV 序列分块；Q、K/V 通过 core mesh 广播复用；score 与概率块留在 SRAM，不把完整 `B×H_q×S_q×S_k` FP32 矩阵写入 HBM。

GQA 保留紧凑 K/V，通过组内复用服务多个 Q 头。**LLM 工作量仍默认 BF16**，softmax 统计量和累加器为 FP32；这是工作负载定义，不构成 INT4 PE 对 BF16 的硬件支持证明。PV 的概率按 tile 转为 BF16供矩阵单元使用；转换计入向量工作量，模型采用 tile 内概率表示。

沿 K 维分块时保留稳定 softmax 的 `(m, l, O_tilde)` 状态。两个分块状态合并时：

```text
m = max(m_a, m_b)
l = exp(m_a-m)*l_a + exp(m_b-m)*l_b
O_tilde = exp(m_a-m)*O_a + exp(m_b-m)*O_b
O = O_tilde / l
```

不计算实际张量数值，但计算这些状态的存储、合并操作和通信量；不能把独立 softmax 输出直接相加。跨核 O_proj 或 FFN down 的 K 维部分积同样是求和归约，不是拼接。

### 4.3 TX8 参考执行路径

提供独立的 `tx8_reference` 配置，使用 128 TFLOP/s、1 T 等效向量操作/s、48 MB SRAM、64 GB DRAM、200 GB/s。此配置不是第 3 节核阵列的一个候选；TX8 的实际核数和 NoC 拓扑未给出，不为它编造拓扑。

参考路径采用 Excel 中的完整 FP32 score 外存方案：QKᵀ写出 score，scale/softmax 组读写一次，PV 读取概率。不得把目标 die 的片上融合 attention 与这条路径混用后比较延迟误差。

TX8 参考组时间如下，仅用于核对 `tx8_reference_v1` 的公式和算子图计数：

| 组 | 每次 μs | 次数 |
|---|---:|---:|
| RMSNorm | 246.000 | 2 |
| Q_proj / O_proj | 359.512 | 各 1 |
| K_proj / V_proj | 134.817 | 各 1 |
| RoPE_QK | 315.034 | 1 |
| QKᵀ | 1033.596 | 1 |
| score_scale_softmax | 3934.079 | 1 |
| PV | 1033.596 | 1 |
| residual add | 368.820 | 2 |
| FFN | 3363.000 | 1 |
| block 合计 | 11897.603 | — |

以上显示值为三位小数汇总；按原 Excel 13 组双精度公式逐组重算，完整 block 总计 **11897.602619368334 μs**，其中 attention 对应组的汇总为 **7304.962795063794 μs**。此总计不是 BF16 新 Core 的真实运行时间。

重现这些数值只叫“参考公式一致性检查”。目标硬件的资源调度模型即使采用相同先验，也可能因缓冲、搬移顺序和流水填充得到不同结果，不能逐节点乘一个修正系数硬拉回该表。

## 5. 用少量分块保留真正影响扫参的机制

### 5.1 搜索规模与容量

只搜索有限的规则分块。初始候选为：token/query 块 `{1,8,16,32,64,128}`，输出通道/FFN 块 `{32,64,128,256,512}`，reduction/KV 块 `{32,64,128,256,512}`。裁剪到实际维度，补入小于最小候选的实际尺寸并去重；尾块按实际大小计费。

core group 的列数从 `{1,2,4,8,N_x}`、行数从 `{1,2,4,8,N_y}` 中分别选取并各自裁剪到合法整数上限，允许多个不重叠矩形组并行，剩余边缘 core 可以空闲。只给有任务的 core 分配算力；不能对一个很小的任务始终使用全 die 峰值。

每核预算为 `floor(0.90*sram_mib_per_core*2^20)` **byte**，10% 是软件预留而非物理硬件约束；总 SRAM 不可充当共享片上缓存。预留比例是工程常量，不是扫描维度。

### 5.2 GEMM 与向量算子

对于 BF16 `A[M,K] @ W[K,N]`，工作量与容量统计仍使用相同的张量定义；**数值执行时间先通过第 3.4 节的 BF16 能力检查**。基本块为 `[m,k] @ [k,n]`，每核保留 FP32 输出累加器。输入与权重双缓冲的工作集为：

`S_gemm = 4mk + 4kn + 4mn bytes`。

遍历 K 块时保留累加器；不同 M/N 块映射到不同 core，A/W 在有共同消费者时通过 NoC 广播。容量不足时尝试单缓冲，仍不满足则缩小 tile。单缓冲不能获得双缓冲的搬移重叠收益。

映射器至少比较 output-stationary 和 weight-stationary 两种循环顺序。沿 K 拆到多个 core 是低并行度任务的备选，必须计入 FP32 部分积归约。decode 的 `M=1` 按 GEMV 工作量和局部并行度建模，不人为补成完整 prefill tile。

Norm、RoPE、残差和独立激活沿行/连续元素分块。Norm 优先让完整 hidden 行驻留本核；默认目标维度均可满足，超出容量时使用两遍流式读取并增加相应字节。向量算子的张量元素数必须随形状更新。

### 5.3 融合 FFN

FFN 沿 token 和 F 分块，片上执行：

```text
gate tile → SiLU ─┐
                 × → down 部分积 → FP32 输出累加器
up tile ─────────┘
```

每个 core 处理一个 token 块及一部分 F，保留该 token 块的完整 H 输出部分和。gate/up 的 H reduction、down 的 H 输出可继续分块加载权重。一个保守的单执行上下文预算为：

`S_ffn = 6*m*H + 12*m*f + 4*max(k*f, f*n_out_tile) bytes`。

其中 `6mH` 是 BF16 输入及 FP32 输出累加器，`12mf` 为 gate/up/中间块预留，最后一项为当前权重 tile 的双缓冲；`n_out_tile` 是 down 的 H 输出块宽，**不是** §3.2 中 HBM 每排颗数 `n_hbm_per_row`。若多个 core 分担 F，最后按 token 块归约完整 H 的部分和，计入向量计算和 NoC 流量。

容量不够则减小 token/F 块；仍不能融合时回退为 gate/up、激活乘法、down 的分离执行，并对物化的中间张量增加 HBM 读写。**SRAM 增大带来的收益由减少流量、支持融合或增加可并行上下文自然产生，不通过额外奖励系数产生。**

Excel 中“512 通道×28 块、40.89 MB 工作集”只适用于其 TX8 容量级示例。目标 die 使用每核预算重新选块，不能让一个 core 借用全 die SRAM 去容纳它。

### 5.4 Attention 与有限流水

单个 query/KV tile 的一个保守预算为：

`S_attn = 2qd + 8kd + 4qk + 8qd + 16q bytes`。

分别包含 BF16 Q、双缓冲 BF16 K/V、FP32 score 工作块、**两份 FP32 未归一化输出部分和**和**两份 `(m,l)` 行状态**；该预算用于本模拟器的容量约束。score 空间复用于概率转换。两个并行上下文按 `2*S_attn` 检查（并考虑共享缓冲的真实生命周期），不能在单上下文预算里默认包含两份上下文。

映射按 GQA head-group 组织；同一 K/V tile 通过组内广播供多个 query 块或 Q 头使用。HBM 字节按真实加载次数统计，NoC 按广播路径统计。没有同时驻留或有效广播路径时，K/V 必须重复加载；不能无条件假设整层 K/V 只读一次。

对不同 K 分区的局部结果，归约消息至少含每行 `d+2` 个 FP32 数，即 `4q(d+2)` bytes，外加合并所需向量工作。矩阵部分的稠密计算量为 `4*B*H_q*S_q*S_k*d` FLOP；如果启用 causal，只跳过完全位于 mask 区域的 tile，跨对角线 tile 仍按执行的矩形工作量计费。

默认尝试两个独立 head/query 执行上下文，使一个上下文的矩阵计算与另一个的向量阶段重叠。保守地要求两份上下文预算都放入同一 core；放不下则退回一个。仅有一个上下文时，不把同一 tile 的 QKᵀ、softmax 和 PV 并行执行。

## 6. 时间模型：资源服务时间与依赖

### 6.1 统一计量

内部单位显式区分 `byte`、`BF16 FLOP`、`INT4 integer op`、`FP32 ideal FLOP`、`vector eq-op`、秒；**SRAM MiB 按 2^20**，HBM GB/GB/s 按来源十进制，FMA 工作量计两次相应精度操作。最终输出 μs。矩阵与向量操作分别计量，不能把特殊函数的等效成本称为真实 FP32 FLOP 吞吐。

固定的向量工作量近似如下：Norm 每元素 6 次，RoPE 每旋转元素 3 次，residual/gate_mul 每元素 1 次，SiLU 每元素 15 次，scale/mask/softmax 每 score 元素 12 次，格式转换每元素 1 次。跨块 softmax 合并和跨核归约按第 4、5 节实际状态另外计费。这些是工程成本权重，包含特殊函数的近似，不是 TX8 指令计数。

### 6.2 以并行波次计成本：HBM 芯片、injection、NoC 的共同限制

将同一宏组中可并行的同形 tile 合并为波次；映射实际 Core/group，重复波次用计数压缩。矩阵和向量时间在 dtype 能力校验通过后按**最忙 Core**而非整个 Die 名义峰值计算：

```text
t_PE  = max_core(F_bf16_core/(eta_PE_bf16*P_PE_bf16_core))  # 必须相容 dtype 校准
t_vec = max_core(E_eq_core/V_eq_effective_core)             # 必须有 eq-op/s 来源
noc_link_raw_Bps       = noc_width_bits * 1_000_000_000 / 8
noc_link_effective_Bps = eta_noc * noc_link_raw_Bps
t_NoC_lower_bound = max_edge(D_directed_edge_bytes[edge]/noc_link_effective_Bps)
# 这些只是单独资源必要下界，不自动等于重叠后的 stage 结束时刻。
```

未提供相容精度峰值返回 `unsupported_precision/missing_performance_calibration`，不允许用 INT4 GOPS 或 FP32 FMA 等效峰值替代；尾波次、闲置 Core 与真实利用率分别保存。**即使所有计算服务率已给出，ingress 未指定时也不能把 HBM 理论上限冒充可执行系统速率**。

HBM 读/写是**共享**的。对访问类别 matrix/vector，原无外部竞争的单流**芯片峰值乐观时间**可保持：

```text
B_memory_chip_peak = N_HBM*1_177_600_000_000  # byte/s, 十进制峰值
U_stage = sum_over_stacks_and_access_classes(actual_bytes_i/eta_access_i)
t_HBM_chip_optimistic = U_stage/B_memory_chip_peak
# 若分时执行两类独立阶段，等价 D_matrix/(eta_matrix*B_memory_chip_peak)
#                         + D_vector/(eta_vector*B_memory_chip_peak)
# 但它不自动包含每颗 HBM 的偏斜、端口、NoC 共享与首访延迟。
```

若把同一数据请求拆到各颗 HBM，必须逐颗检查：`t_HBM_lower_bound >= max_stack(U_stack_work[stack]/hbm_peak_Bps_per_stack)`，其中 `U_stack_work[stack]=Σ_flow(D_stack_physical_bytes[flow]/eta_HBM[flow])`，且 `t_HBM_lower_bound >= Σ_stack(U_stack_work)/(N_HBM*hbm_peak_Bps_per_stack)`。两项均属于必要下界，只有部分颗被访问时不能自动获用全部颗峰值。**入口/控制器/NoC 的分子必须各自取真实经过该物理资源的 byte，而不是 HBM 等效 `service_units`**：`t_ingress_lower_bound >= max_port(D_ingress_port_bytes/ingress_effective_Bps_per_port)`；`t_controller_lower_bound >= max_controller(D_controller_bytes/controller_effective_Bps)`；`t_NoC_lower_bound >= max_directed_edge(D_edge_bytes/noc_link_effective_Bps)`。`ingress_effective_Bps` 若已包含控制器/PHY 效率，禁止再次除以同一个效率；HBM 访问先验 `eta_HBM` 只在 `U_stack_work` 中作用一次。端口速率的证据身份可为 `measured` 或明确标注的 `engineering_assumption`，不允许自动取 HBM 芯片峰值。只有 §6.2.2 共享资源进度算法满足容量/依赖条件且所有必需速率已指定时，才能从必要下界进一步计算**假设模型执行时间**。`max(t_HBM,t_ingress,t_NoC)` 本身**不是**已验证的结束时刻或可替代仲裁器的公式。

`eta_HBM_matrix≈0.70`、`eta_HBM_vector≈0.3411668293` 和 `eta_noc=0.70` 都是历史 DDR/软件**敏感性先验**，非该 SKU 实测效率；只能在显式声明假设的场景下计算数值，且服务时间模型的等效 `service_units` 不得被当作多传了的物理字节。有效时延还必须包含必要的首访/控制器访问延迟（§7.1），不能认为带宽越大延迟必按比例减小。

### 6.2.1 分层字节、请求所有权与唯一物理流（设计接口）

当 COMP、COMM_READ/WRITE、RESTORE、POOL 同时访问 HBM，使用**同一个 Die 内共享资源域**和每颗 HBM、控制器、端口以及 NoC 有向边的容量约束；不是一条“COMP 总线”和一条“通信总线”。以下字段是执行器和共享访存仲裁器的接口契约。单 Die 分析器不模拟在线服务调度，但其离线重叠推算必须遵守相同守恒规则。

```text
DieMemoryStageRequest:
  die_execution_id          : uint64                   # 调用所属执行域
  requester_kind            : COMP|COMM|RESTORE|POOL   # 无默认身份
  requester_id              : uint64                   # 同一域内唯一操作/事务 ID
  owner_macro_id            : optional<uint64>         # COMP 必填；非 COMP 不得伪造0号宏组
  completion_id             : uint64                   # 全执行域中唯一阶段回调 ID
  generation                : uint64                   # 每次重排/取消递增
  access_kind               : matrix|vector|comm|restore|pool
  io_direction              : HBM_TO_CORE|CORE_TO_HBM  # 本stage内所有flow方向必须一致
  op_dtype                  : explicit
  first_access_scope_key    : tuple                   # 下文定义，不接受全局常数0
  actual_bytes              : uint64                  # 一次真正的HBM payload，不是服务量
  service_units             : double                  # Σflow(D/eta_HBM)，仅HBM服务工作
  flows                     : list<HBMPhysicalFlow>   # 必须足以唯一生成各资源边际

HBMPhysicalFlow:
  flow_id                   : uint64                  # stage 内唯一
  physical_payload_bytes    : uint64                  # 本 flow 的 HBM 读/写 payload
  access_class              : matrix|vector|comm|restore|pool # 决定HBM效率来源
  eta_hbm                   : finite_double_in_(0,1] # 同一次访问的唯一HBM效率，未校准时标先验
  eta_provenance_id         : nonempty_versioned_id    # 使用工程先验时必须显式标注
  io_direction              : HBM_TO_CORE|CORE_TO_HBM # 与stage方向严格相同
  stack_id                  : uint                     # 0..N_HBM-1
  controller_id             : validated_controller_id
  port_id                   : validated_port_id
  src_core_or_edge          : legal_coordinate_or_boundary
  dst_cores_or_boundary     : nonempty_legal_destinations
  route_directed_edges     : ordered_dag_or_route_tree # 记录(edge_id, forwarded_bytes)及顺序
  multicast_copy_policy    : none|explicit_tree
```

**请求拆分与字段校验**：一个 `DieMemoryStageRequest` 只能包含**同一物理方向**的 HBM 流。若一个计算宏组既读取权重/KV 又写回结果，必须拆成至少两个方向明确、`completion_id` 不同的 stage，按真实依赖连边或在允许的独立缓冲下并行；外层宏组仍只完成一次。允许同一宏组关联多个 stage，但不得把写回隐式合并成读取的免费流。`access_kind` 和 `op_dtype` 是请求标签，`flow.access_class`、`flow.eta_hbm` 与显式效率来源共同决定该流在颗粒资源的 `D/eta` 工作量。每个非零 `flow` 的 `eta_hbm` 必须为有限 `(0,1]`，并有 `eta_provenance_id`（已测量配置或注明适用条件的历史工程先验）；缺失时只能统计真实物理量与必要的**乐观峰值**下界，不能求有依据的完整时间。`service_units` 是由 flows **重新计算并严格核验**的派生字段，不能信任请求方任意传入的等效字节。

对每个非零请求，`Σ_flow physical_payload_bytes = actual_bytes`、`service_units = Σ_flow physical_payload_bytes/flow.eta_hbm`；`flow.io_direction=stage.io_direction` 且 `flow.access_class` 必须与访问语义兼容。逐 `stack_id` 汇总 `D_stack_physical_bytes`，逐 `port_id + io_direction` 汇总 `D_ingress_port_bytes`，逐 `controller_id` 汇总 `D_controller_bytes`，逐 `directed_edge_id` 汇总 `D_directed_edge_bytes`。**所有边际都从同一 `flows` 构造，不允许只给各栈、各口两张边际表反推路由**。端口图的合法性检查覆盖边沿、矩形网格坐标 `0<=x<N_x,0<=y<N_y`、读写方向、相应 controller 所属关系与 XY/广播树。单播路径必须是一条合法的有向路，广播复制必须有明确分叉节点的合法有向树/DAG；逐边保存 `(directed_edge_id, forwarded_bytes)`，同一 HBM payload 在每条实际经过的边上按**传输次数**计数，不能用每条边都等于 `actual_bytes` 替代多条 flow 的真实分布，也不能因共享树枝重复虚造 HBM payload。禁止无路由依据的循环、越界坐标、负字节、非法跨边或方向不符的路径；若支持重传，需另列重传事件和 NoC 字节而不暗增 HBM 已成功读取的 payload。一个 HBM 物理读取可在 NoC 内复制，因而 `Σ_edge D_edge` 可以大于 `actual_bytes`；这是转发而不是额外 HBM 读取。若广播需要额外 HBM 读取，必须另外形成 `flow` 并显式增加 `actual_bytes`。任何 HBM 写回不得因为其目标是 HBM 而被算成读。

首访延迟责任键：COMP 为 `(die_execution_id, COMP, owner_macro_id)`；COMM/RESTORE/POOL 为 `(die_execution_id, requester_kind, requester_id)`。一组宏组仅首次**非零 HBM**访问支付一次 `local_mem_latency`；非 COMP 的同一请求事务按一次首访计费，**不同 requester_id 独立计费**。后一条是本规格选定的软件归属规则，不是厂家已验证的 DRAM 首字延迟机制。`first_access_latency` 为状态机**派生结果**，禁止请求方凭空传入矛盾布尔值。`D=0` 同步完成且不消耗首访配额；同一事务重复提交相同 `completion_id` 须拒绝。

`actual_bytes` 是 HBM/KV/日志/通信 payload 的唯一物理字节依据。`service_units` 是**同一 payload 的 HBM 时间等效工作量**，不得填入容量、NoC payload、通信图或 HBM 实际利用率的真实 byte 计数。端口、控制器、NoC 按真实资源流量计；`bytes_by_stack` 和 `bytes_by_ingress_port` 可在报告中作为派生汇总，绝不是可替代 flows 的输入。

### 6.2.2 多资源公平仲裁、完成事件和可求值范围

以有向独立资源集合 `R` 构造共享服务日历。资源包括每颗 HBM（每颗名义上限 `1_177_600_000_000 B/s`）、控制器读/写端口、物理 ingress/egress port 与 NoC **有向**边：每个资源 `r` 有 `capacity_Bps[r] > 0`，其 duplex 政策必须显式列出；半双工读写共用同一容量资源，全双工则读/写各建独立资源。若控制器/端口带宽或 duplex 政策未知，不能填 0、无穷或假定有免费带宽，应阻断完整时间。NoC 模型仍采用 §3.3 的固定 `noc_link_effective_Bps`。

对激活 stage `i`，构造单阶段各资源所需工作量 `work[i,r]`（单位 `byte` 或 **byte 等效 HBM 工作单位**）：在 HBM 颗粒资源取 `Σ D_flow/eta_HBM(flow)`，在 controller/port/NoC 有向边资源取真实转发 byte；`work[i,r]=0` 表示未使用该资源。完成比例 `progress_i∈[0,1]`、所有非零资源同时按同一比例推进，避免对一个 payload 在多个资源队列分别“完成”一次。每个流按 stage 完成比例积分服务进度；物理计数采用**单调累计整数记账**：在每个事件边界令 `served_bytes_flow=floor(physical_payload_bytes*progress_i)`（完成时强制等于完整整数 payload），本次实际增加量为 `served_bytes_flow-previous_served_bytes_flow`。未凑足一个完整 byte 的分数仅留在计算进度中，不能提前写入通信/KV/实际 byte 账本；取消时保留已累计的完整物理 byte 并丢弃未完成的分数，不撤销已发生的物理负载或补收未传字节。**已写出部分字节不意味着输出张量已经有效**：只有对应的写回/加载 stage 完成、所有依赖得到满足时，才将目标数据标记为 `materialized/ready`；取消或代际失效时，保留物理流量日志但不得将不完整中间张量/权重/KV 误标为有效驻留副本。NoC 每有向边同样从其已规划的转发 payload 按进度导出累计整数流量，因此 multicast 可产生多边流量、但不增加 HBM payload。

调度策略固定为**等权最大最小公平的 stage 完成比例速率**，不是依赖实现遍历顺序的模糊“平均共享”：求 `x_i=d(progress_i)/dt>=0`，并满足所有资源的 `Σ_active(work[i,r]*x_i) <= capacity_Bps[r]`。从所有 stage 的 `x_i=0` 等速增长，遇到首个饱和资源就冻结使用该资源的 stage，其余继续增长，迭代到所有 stage 都被冻结；并列饱和时按资源稳定 ID 同时处理，结果对应唯一的渐进填充规则。随后取 `Δt=min_i((1-progress_i)/x_i)`，或取下一外部到达/取消时刻中更早者，推进 `progress_i+=x_i*Δt`。若需要整 tick，统一以向上取整预约但在 `advance_to` 用未舍入服务进度避免重复/丢失；不可把舍入放大的 `service_units` 当物理 byte。不同颗粒、不同入口且未共享瓶颈的两个 stage 允许并行完成；若共享端口，则按该端口容量竞争。一个计算侧至多激活一个 HBM stage，其余 tile 依赖 FIFO/波次合并；COMM/POOL/RESTORE 可以并发参与同一日历，不改变原图字节语义。

事件顺序为 `advance_to(now_tick) → 结算已经发生的连续服务 → 处理该 tick 的完成/取消/新请求（稳定排序：取消、已到期完成、到达；同类按 completion_id）→ 更新首访/资源活跃集 → generation++ 使过期预约失效 → 重算 x_i → 预约下一完成时刻`。处在首访延迟阶段的请求**尚不占**传输资源；完成该延迟后再进入 streaming 竞争集。取消只回收未来未完成份额，不撤销已经服务的物理流量；已完成和取消竞态按上面固定顺序处理，`(die_execution_id,completion_id,generation)` 过期回调不得再次改变进度。只有流经过全部参与资源且 `progress_i=1` 才触发**一次**内部完成，所有内部阶段结束才使外层 COMP **一次** terminal；HBM 与 NoC 不能各触发一次 COMP 完成。

**近似身份与结果界限**：这是选定的抽象软件公平调度策略，不是 HBM3E 真实 PHY 的时序/控制器实现证明。单个 stage 的 `max_r(work[i,r]/capacity[r])` 及所有活跃请求的资源约束给出理论必要下界；在该调度策略、完整流图、校准与延迟均给定时产生可复算的**模型完成时间**，不可称硬件精确最短时间。单标量近似 `U_effective=max(Σ_jD_j/η_j, N_HBM*max_j(D_j/η_j))` 只能用于单阶段的保守预算，不作为多阶段共享仲裁完成时间算法，也不得增加真实 payload。

**动态回归夹具**：① 同宏组 `D1=100,η=.5`、`D2=60,η=.75` → `actual_bytes=160`、`service_units=280`、首次延迟一次；② `2` 独立栈/端口，各 `100 B`、`100 B/s` → 同时完成 `1 s`，不是错误单队列 `2 s`；③ 两栈共用一个 `100 B/s` 半双工口，两请求各 `100 B` → 无其他瓶颈时最后完成 `2 s`；④ `100 B,eta_HBM=.5,HBM raw=100 B/s,ingress=50 B/s` → 两个资源各需 `2 s`，**不得把 `200 service_units` 再除 50 得 4 s**；⑤ 两栈/两口各 `100 B` 的直连和交叉连接具有相同边际但路线不同，必须导出不同 `flow`；⑥ 同 payload 读写分别占用相反有向边；⑦ 两个 COMM 不共享默认 `owner_macro_id=0`；⑧ 零字节、尾字节、同 tick、取消、`generation`、延迟转流化、RESTORE/POOL、读写竞争、`COMP` 恰好一次完成。以上仅是**规格验证用例**；未在现有五仓代码中执行。

### 6.3 最小调度器

每个波次只生成必要阶段：加载、PE、vec、可能的再次 PE、归约和写回。例如 attention 必须保留 `QK → softmax → PV`，FFN 必须保留激活之后才能 down 的依赖。

模拟器维护 die 级 PE、vec、HBM、NoC 的可用时间及最多两个执行上下文的缓冲占用。一个波次已汇总其全部并行 core；不同波次的同类资源保守串行，逐核抢占调度不属于实现范围。

阶段起点取“前驱完成、所需资源可用、缓冲可用”三者最大值。HBM 经 NoC 流式传输的阶段须受已解析的 `HBM chip/stack → ingress port → NoC` 共同约束；仅在端口/路径/竞争映射齐备时以各 `work/capacity` 形成必要下界，并按 §6.2.2 的共享公平推进规则计算有假设身份的阶段结束时刻；不得把 `max(t_HBM,t_ingress,t_NoC)` 直接当作多用户真实完成时间；跨 core 传输仅占 NoC，计算仅占相应 PE/vec。这样能表示稳态搬移/计算重叠，并自动保留首块加载和末块写回。

这是一种资源级近似：忽略 flit 传播和细粒度链路并发，可能高估一些可充分重叠的流量；但没有把各组链路带宽任意相加，也不会让无缓冲的依赖阶段无条件重叠。

每个对外宏组固定加一次 1 μs 启动成本，作为目标架构默认先验，不对每个内部 tile 收费。TX8 参考公式使用其0 μs 附加启动成本，以保留参考口径。8–15 μs 的原表短时间不能直接认定为该启动成本的测量。

对每个可行映射按以上过程评估，选择完成时间最小者。搜索与核算均不读取 Excel 的逐节点时间来决定赢家。

## 7. 默认参数、校准与敏感性分析

### 7.1 TX8 参考系数与目标模型的工程先验（非目标 BF16 校准）

| 系数 | 默认值 | 依据与限制 |
|---|---:|---|
| `eta_PE_tx8_reference` | 0.907955 | **只用于 TX8**：FFN 条件性解释，不代表目标 INT4/BF16 转换或 PE 效率 |
| `eta_vec_tx8_reference` | 0.907955 | **只用于 TX8**；目标 Vec FP32 SIMD 理想峰值不能按此默认得到 SFU eq-op/s |
| `eta_HBM_matrix` | 0.70 | 可配置的目标 HBM3E **未校准敏感性先验**，不是 Micron 实测 |
| `eta_HBM_vector` | 0.3411668293 | 源于 TX8 DDR/RMSNorm 的参考拟合；用于目标 HBM3E 仅作**显式未校准敏感性先验** |
| `eta_noc` | 0.70 | 无独立 NoC 测量，采用固定持续带宽先验 |
| `pe_peak_bf16_flops`, `vec_eq_ops_per_second` | `null` | 硬件资源表未定义；缺少相容服务率时不得报告目标 BF16 时间 |
| SRAM 可用比例 | 0.90 | 为未显式展开的小型状态预留空间；MiB 转 byte 后应用 |
| 启动成本 | 1 μs/宏组 | 目标架构工程默认值；不是 HBM 首访访问延迟 |
| `local_mem_latency` | `null`，由实现显式配置 | 有非零 HBM 访问且该值仍为 null 时完整 `latency=null`、`invalid_reason=missing_memory_latency`；显式工程值只能标为敏感性模型，非厂家实测 |
| `ingress/PHY bandwidth` | `TO_BE_DEFINED` | 真实 HBM→NoC 接入口数量、速率、路由和争用未验证，不能提升为可信 target 时间 |

上述 Norm 速率约为 68.23 GB/s，其中输入输出为 16777216 bytes，gamma 向量为 8192 bytes，按每次调用从外存读取一次计费。目标图也按实际驻留和加载规则统计 gamma，不另加重复流量。

**表中前两项效率不能自动用于目标版 PE/Vec 的数值计算**；仅对另行确认了硬件兼容精度峰值的候选，才允许开展有清楚“未校准”标记的算力敏感性实验。两种 TX8 计算效率来自选定的等效工作量分配：在串行计算预算解释下，以 `eta = (F_FFN/128T + E_FFN/1T) / 3363μs` 得到约 0.908。该关系并不能证明 3363 μs 一定是 FFN，更不能独立识别矩阵、向量、DMA 和流水各自的效率。

目标 die 的向量表值是 FP32 理想等效 FMA 吞吐，**仅作为 FP32 等效 FMA 峰值使用**；未证明它与 TX8 的精度和特殊函数路径相同。因此跨架构沿用这些系数属于工程外推，不是对所有候选 die 的硬件验证。

### 7.2 校准数据与验证数据的区分

1. **记录拟合**：以 RMSNorm 246 μs 约束 TX8 的外存向量类有效速率。它是训练点，拟合误差不作为泛化精度。
2. **条件性约束**：接受“3363 μs 为 FFN 总时”的工作假设，采用上表效率先验；新调度结果与该值的偏差要保留，不能重新给每个阶段强制分配时间。
3. **参考核对**：核对 Excel 图、维度、组次数和派生总时间。此处一致说明参考实现无明显记账差错。
4. **独立实测验证**：当前资料没有明确边界、独立于上述拟合的 block/attention 验证点。不得将派生值上的 MAPE 报告为独立实测预测误差。

若引入新的 TX8 计时数据，必须同时提供测量边界、工作负载形状、设备配置及数据来源；缺少这些信息的样本只能作为条件性参考，不参与独立准确率统计。

### 7.3 参数敏感性分析

在 BF16/vec 目标服务率尚缺失时，硬件扫参**只能报告资源、容量、几何与工作量及可能的必要带宽下界，不能排名 BF16 时延**。一旦显式提供所需的目标 dtype 服务率与效率来源，再做固定的模型敏感性检查：对已定义精度的计算效率假设取 0.70、0.907955（仅作为敏感性对照）、0.95；矩阵 HBM 和 NoC 效率取 0.50、0.70、0.90；向量 HBM 效率取默认值的 0.75、1、1.25 倍；启动成本取 0、1、5 μs。效率不超过 1。

先单项变化，再检查各项同时偏低/偏高的两组情景，避免额外做巨大的笛卡尔积。报告最佳配置和主要排序是否改变。这个范围是模型敏感性范围，不是统计置信区间，不当作新的硬件扫参维度。

硬件定义要求同一 `(SRAM,core_type)` 下固定 `L₀`：位宽扩展挤占 PE/Vec 单元。应将这个硬件约束与**固定 N_vec/N_pe、仅加宽链路**的理想消融对照（后者不是本硬件模型的合法候选），并检查不同 PE/Vec 可用服务率的敏感性。报告这些模型假设对预测排序的影响。

## 8. 模块接口、输出和测试要求

### 8.1 单配置动态解析与上层枚举器的边界

```text
# 仅上层 DSE 扫描：core64 × integer Nx/Ny × n × two placement modes × geometry boundaries
explore_dynamic_design_space(search_profile):
  return list[resolved_die_candidate]  # Core-first 与 Wafer-first 相互复核

# 每次 die 内仿真器调用只接受一个已解析实例，不内部枚举 Die 面积
make_hardware(resolved_config):
  validate(schema == mesh_die_hbm3e_dynamic_v4, sku, units, geometry_revision)
  core = load_exact_64_core_rows(core_table_revision, core_table_sha256)
  validate_integer_mesh(Nx>=1, Ny>=1)
  validate(W_die>=Nx*L0+(Nx-1)*px+ex, H_die>=Ny*L0+(Ny-1)*py+ey)
  validate(derive_area(W_die,H_die), n>=1, N_HBM==2*n)
  derive(module_outer_shape, HBM bytes/theoretical peak, bump, wafer, D2D)
  flag_overhang_without_automatic_rejection(W_HBM_row>W_die)
  resolve_optional_ingress_map_or_mark_unknown(map_id)
  return immutable_hardware_descriptor_and_fingerprint
make_local_graph(workload): return explicit_shapes_dtypes_bytes_dependencies
enumerate_mappings(graph,hardware): return legal_tile_and_rectangular_core_group_mappings
estimate(mapping,hardware): return typed_resource_report_with_optional_timing
```

64 行 Core 表的修订标识与内容 SHA-256、几何算法版本与 v4 Schema 共同构成硬件身份。硬件指纹必须覆盖 `N_x,N_y,W_die,H_die,px,py,ex,ey`、连续派生面积、`n,placement_mode,W_HBM_row,W_module,H_module`、SKU、NoC、HBM 注入图及性能校准来源；必须对实际 W/H、Nx/Ny 与模块几何等物理字段计算 hash。不同 W/H 即便面积相等也不是同一物理硬件：曝光场、Core 矩形网格、Wafer 行列和端口边沿坐标均可能不同。性能/映射缓存键另含真实工作图、dtype、local tile、端口路由及校准版本；C++/Python 使用同一个规范 JSON 指纹。

### 8.2 必须输出的字段与单位

| 类别 | 必须输出（不得混合列） |
|---|---|
| 配置身份 | `hardware_schema_version`, `hbm_sku`, `core_table_revision/hash`, `hardware_fingerprint`, `model_version`, `calibration_version`, `die_width_mm`, `die_height_mm`, `compute_area_mm2`（连续派生）, `core_columns_nx`, `core_rows_ny`, `noc_width_bits`, `core_type`, `sram_mib_per_core` |
| HBM 实体 | `hbm_stacks_per_row=n`, `hbm_placement_mode`, `row_fit_policy`（可选独立筛选）, `hbm_row_overhang`, `hbm_geometry_revision`, `hbm_stack_count=2*n`, `hbm_capacity_bytes_total`, `hbm_capacity_GB`, `hbm_peak_Bps_theoretical`, `hbm_peak_GBps_theoretical`, `row_width_mm` |
| HBM 注入 | `hbm_injection_map_id`, `per_stack_actual_bytes`, `per_stack_peak`, `controller/port_id`, `per_port_ingress_Bps`, `effective_ingress_peak_Bps`（有配置才能数值化）, `ingress_bottleneck`, `bandwidth_assumption_status` |
| Die/NoC | `N_x`, `N_y`, `N_core=N_x*N_y`, PE INT4 GOPS, Vec FP32 等效 GFLOPS, `eta_noc`, 有向链路 peak/实际负载、最忙链路字节与速率、NoC bottleneck |
| Wafer/D2D | `W_module_mm,H_module_mm,A_module_mm2,A_inter_die_noc_mm2,W_eff_mm,H_eff_mm,wafer_cols,wafer_rows,N_wafer,B_bump_GBps,B_D2D_oneway_GBps`；此表仅上层资源，不扩写 rank 数 |
| 工作量/映射 | 本地形状、BF16/INT4/FP32 的操作量单位、prefill/decode、层数、权重/KV/活跃工作区 byte、**逐颗 HBM 驻留峰值与地址条带验证状态**、core group、tile、尾波次、融合与回退、SRAM 峰值 |
| 时间与事件 | 当 capability/校准/ingress 齐备才输出 9 宏组/block/iteration time；否则 `latency=null` + 明确缺失项 `unsupported_precision`/`missing_performance_calibration`/`missing_residency_mapping`/`missing_ingress_mapping`/`missing_bandwidth_calibration`/`missing_memory_latency`；`actual_bytes`、`service_units`、各 flow 的 `eta_hbm/eta_provenance_id`、`pe_busy_time_s`、`vec_busy_time_s` 分列；输出流级读写路径、每栈/每控制器/每口/每有向边字节与饱和度，以及 generation/COMP 唯一结束 |
| 状态与证据 | `dtype_supported`, `performance_calibrated`, `ingress_verified`, `invalid_reason`, `parameter_provenance`, `timing_evidence_level`, `measured_performance_supported`, `missing_requirements` |

`block_complete && timing_capability_complete` 时即可输出 **单个完整 block** 的 `block_token_per_s=batch*q_len/T_block`；**不要求**本地所有层结束。`all_local_layers_complete && timing_capability_complete` 且本地迭代计时窗口完整时才输出 `local_iteration_token_per_s=batch*q_len/T_local_iteration`。两者的完成布尔量独立记录；只完成第 1/4 个 block 的情况下允许前者，拒绝后者。decode 完整一步产生 B 个 token 时才可用 `batch/T_iteration`。这**不是**多 Die 全系统、Wafer 服务或网络吞吐；Wafer 完整模块数从不自动更改当前在线 `rank_count` 或 collective 数据图。

### 8.3 规范一致性与运行时测试

1. **Core 表与动态几何**：对大小核/8 SRAM/4 NoC 的 64 行校验 L₀、整数 Vec/PE 和哈希；测试 `N_x=3,N_y=2`（6 核）及 `N_x=2,N_y=3`（同数量但外形和入口不同），`N_core=N_x×N_y`，任何 W/H 不足须拒绝；候选数量从 DSE 输出统计。
2. **动态边界与反例**：200–800 mm²、150–200 下界、26×33 旋转曝光场、长宽比≤2、总外缘 `e_x=e_y=0.10` 和 pitch=0 **只在选择该 DSE profile 时筛选**。检验按源码约 `8×11 large/0.25MiB`、`W≈20.6084,H≈28.29905,n=2` 可生成扩宽后的模块；HBM 排宽超过 Die 不自动拒绝，必须标物理待验证。仅有面积、缺少 W/H 和 Nx/Ny 的配置应返回几何字段缺失错误。
3. **HBM 通用公式与单位**：每颗 `36e9` bytes 与 `1.1776e12 B/s`；任意 n 时总 `2n` 颗，容量 `72n GB`，理论峰值 `2355.2n GB/s`；单向 NoC B64/128/256/512 有效 `5.6/11.2/22.4/44.8GB/s`；实际内存受颗粒、控制器、端口与拥塞共同限制。
4. **动态 Wafer 与 D2D**：对每个已解析的 W/H/n 独立计算 `W_module=max(W_die,row_width)+0.30,H_module=H_die+22.56`，`W_eff=W_module/√0.9,H_eff=H_module/√0.9`，`N_wafer=floor(215/W_eff)*floor(215/H_eff)`；`B_bump=41.322314…×A_die`、`B_D2D_oneway=(B_bump−2355.2n)/2≥0`。不同 Nx/Ny 和 n 都不能调用固定三档查表。
5. **端口/矩形网络**：Core 地址合法域 `0≤x<N_x,0≤y<N_y`，矩形 NoC XY 邻接路由与分组不越界；同侧第二排仍竞争 north 边，双侧分别 north/south，不能因相同模块外廓复用未经校验的入口图。
6. **容量、精度和校准**：已驻留权重 + KV + workspace 的聚合与逐颗检查均通过才允许标记已验证；仅有 INT4/FP32 理想值不能计算 BF16 GEMM/SFU 的可信时间；缺 `performance_calibration` / `ingress` / 访存延迟时结构化返回 `latency=null`。
7. **事件与字节守恒**：COMP、COMM、RESTORE、POOL 使用同一 HBM 多资源仲裁；覆盖 T-xx / M-xx 读写归属、取消、generation、首访、共享带宽、尾块及 `actual_bytes`/`service_units` 分离规则。
8. **C++/Python 和两条搜索路径**：DSE `Core-first` 与 `Wafer-first` 对同一限定可行集合输出一致的可行性/最优和 Pareto；对一个固定 resolved descriptor，五仓 resolver/C++/Python 输出相同 W/H/Nx/Ny/HBM/D2D/Wafer/指纹。上层模块数**不自动改变**运行中的 `rank_count`。
9. **TX8 参考一致性**：验证 `tx8_reference_v1` 的 13 个计时组及来源语义；目标硬件资源、性能校准和指纹单独验收。
10. **来源分级**：工程宽高/NoC/PHY 与效率先验须标 `engineering_assumption` 或 `unverified_physical_layout`，缺实物证据不得宣称已完成流片验证。

**算子、访存与调度回归用例：**

| 测试输入/动作 | 必须满足的预期或判定条件 |
|---|---|
| 两个未融合宏组依次传递 `X` byte 张量 | 第一个跨边写回 HBM `X`，第二个重新从 HBM 读 `X`，HBM 真实搬运多 `2X`；显式融合且 SRAM 驻留合法时这 `2X` 才可消失 |
| 两层相同/不同 shape、相同/不同映射 | 仅同 shape、dtype、映射、缓存策略、依赖与启动/首访费用前提均一致时可将单层时间乘 2；否则逐层计量；权重加载按各层真实命中计 |
| 父宏组 + 子项、残差输入 `A`,`B`、输出 `C` | 对外总时间不重复加子项；非融合残差默认读取 `bytes(A)+bytes(B)`、写 `bytes(C)`，没有免费中间值 |
| 精度能力有/无、矩阵/向量资源峰值有/无 | 只有 dtype 相容峰值才做操作数/峰值必要下界；缺失须类型化阻断而不能通过 INT4 代 BF16 |
| 固定计算资源、实际工作量与候选映射，只把可用 SRAM 从小调大 | 大 SRAM 搜索仍包含全部较小容量下的可行映射，最优时间不得因新增可选映射变差；**不保证**改变 SRAM 档位并联动计算资源后的整体硬件性能单调 |
| 固定 local_graph，将 KV 长度从 `S_k` 缩短 | 相关 GQA KV 容量及实际 attention 工作量/访问量不得增加；query 头数变化不得把共享 KV 容量重复放大 |
| decode `M=1`、小任务、单缓冲、双上下文 | 只使用活跃 Core 服务率；单缓冲不得获得双上下文流水；不能把 GEMV 补成完整 prefill tile |
| 固定 `N_x,N_y,W/H`、HBM/端口和本地 graph，仅改变上层 Wafer 网格及 Die 间 D2D 策略 | 本地 `actual_bytes`、SRAM 需求、算力和单 Die 执行时间保持不变；Wafer 资源报告可以不同 |
| `layers_local=4` 只完成独立一个 block | 允许合法的 `block_token_per_s`，不允许 `local_iteration_token_per_s` |
| 首访非零 HBM 访问且 `local_mem_latency=null`、或入口 duplex/速率未知 | `latency=null` 且缺失原因确定；仅零 byte 或完全无 HBM 流阶段不受 `missing_memory_latency` 阻断 |

动态测试须记录输入 fixture、硬件 schema/fingerprint、预期结果、实际执行器输出、实现版本、断言和通过状态。数学反例可用作参考期望值，但不得替代执行器的自动化测试。

## 9. 技术依据与参数可信边界

- **Core 资源与 Die/Wafer 几何**：《架构建模新.docx》§1.1～§3 给出 SRAM、Vec、PE、NoC、Die/HBM、bump 以及 Die 间 NoC 的建模参数。本文以其中的整数 Core 网格反推**动态 W/H 与模块外廓**作为主模型；排宽约束 `row_fit_policy=required` 为可选封装筛选策略。
- **HBM 产品信息**：[Micron HBM3E SKU `MT65B18G16120A00QH-92:A`](https://www.micron.com/products/memory/hbm/hbm3e/part-catalog/part-detail/mt65b18g16120a00qh-92-a)；[Micron HBM3E 系列](https://www.micron.com/products/memory/hbm/hbm3e)。单颗 `1177.6 GB/s` 是 `1024 × 9.2 GT/s ÷ 8` 的理论峰值，不是持续吞吐测量。
- **参考校准数据**：`TX8最终查看版(1).xlsx`，用于 `tx8_reference_v1` 的历史计时、DDR 参考带宽、条件性解释和模型派生数据，不能作为 HBM3E 目标性能证据。
- **方法基础**：WSC-LLM 的局部分块与 FACE 的片上 attention 数据复用和流水思想，不代表对其未公开实现或实验结果的复现。

可直接由设计输入计算的是 Core 表、HBM 颗数、理论容量/带宽、几何、整数装配与 bump/D2D 名义预算。NoC `eta_noc`、HBM 效率、SRAM 预留、Wafer 互连面积比例与启动延迟属于明确配置的软件或架构假设。HBM→NoC 的真实 PHY/控制器端口数、持续带宽、布线与热特性，以及 BF16 PE/SFU 的有效服务率没有定义为可直接使用的实测值；缺少所需性能参数时应阻断目标时间输出。

## 10. 上层 Die/Wafer 动态几何与双向探索（不进入单 Die 时延）

### 10.1 输入、边界和架构搜索与执行解耦

《架构建模新.docx》§2—§3 将逻辑计算 Die 与包含两排 HBM/间距/Die 间 NoC 分摊的**模块**区别统计。Die 的候选硬件身份为 `(core_type,SRAM,B,N_x,N_y,W_die,H_die,n,placement_mode)`。`N_x,N_y` 生成最小可装入网格；W/H 可被面积下界或 D2D 需求向外扩展，**Core 数始终是已经明确选定的 `N_x*N_y`**，不能因外形变大自动填上更多 Core。逻辑 Die 允许矩形与方形，面积为实际 W×H。

源 Word 默认**架构 DSE**约束为 `200≤A_die≤800 mm²`、另做 `150≤A_die<200` 下界敏感性、单曝光场 `(W≤26,H≤33) or (W≤33,H≤26)`、长宽比≤2、每轴总外缘预留 0.10 mm、Core pitch=0、每排 `n=1..14`。这些是**源研究的搜索配置**而非 die 内仿真器能接受哪些面积的固定名单。`target_timed` 仍以显式精度/端口/访问延迟校准为前提；DSE 资源 Pareto 不代表 BF16 执行时间排名。

### 10.2 动态几何、带宽、Wafer 的唯一计算链

```text
L_c = lookup_core64(core_type,sram_mib_per_core,noc_width_bits).L0_mm
N_x>=1; N_y>=1
p_x=p_y=0; e_x=e_y=0.10             # 源 Word 可复算例子，单位 mm；均可配置
W_min=N_x*L_c+(N_x-1)*p_x+e_x
H_min=N_y*L_c+(N_y-1)*p_y+e_y
W_die>=W_min; H_die>=H_min
A_die=W_die*H_die                   # mm²，不是面积档标签
N_core=N_x*N_y
n=hbm_stacks_per_row                 # DSE 默认枚举 1..14
N_HBM=2*n
row_width=10.98*n+0.15*(n-1)
W_module=max(W_die,row_width)+0.30  # HBM 可超出 Die 宽度并扩大模块
H_module=H_die+22.56                # 真实两排高 + Die + 间距
A_module=W_module*H_module
A_effective=A_module/0.9            # 最终扩展面积的 10% 给 Die 间 NoC
A_inter_die_noc=A_effective-A_module
W_eff=W_module/sqrt(0.9); H_eff=H_module/sqrt(0.9)
Mx=floor(215/W_eff); My=floor(215/H_eff)
N_module=Mx*My                       # floor 必须逐轴对未舍入值求取
B_bump_GBps=(A_die/(0.055*0.055))*0.4*2.5/8
B_HBM_GBps=1177.6*N_HBM
B_D2D_bidirectional_GBps=B_bump_GBps-B_HBM_GBps
require B_D2D_bidirectional_GBps>=0
B_D2D_oneway_GBps=B_D2D_bidirectional_GBps/2
HBM_capacity_GB_per_module=36*N_HBM
HBM_peak_GBps_per_module=1177.6*N_HBM
HBM_capacity_GB_system=N_module*36*N_HBM
HBM_peak_GBps_system=N_module*1177.6*N_HBM
PE_INT4_GOPS_system=N_module*N_core*PE_INT4_GOPS_per_core
Vec_FP32_eq_GFLOPS_system=N_module*N_core*Vec_FP32_eq_GFLOPS_per_core
eta_core_die=N_core*L_c**2/A_die
eta_core_region=N_module*N_core*L_c**2/215**2
eta_module_region=N_module*A_module/(0.9*215**2)
```

不把 `A_effective` 误认为逻辑 Die 的制造硅面积；不将 Core 内已分摊的 NoC 再向 Core 减一遍；不将 bump/D2D 名义值当已验证 PHY。系统 215×215 mm 是**抽象可铺排方形区域**，不是圆形制造 Wafer 的良率/切割结果。HBM 同侧双排与两侧各一排在该一阶外廓式中 W/H 相同，但端口与可达路径不同。

### 10.3 DSE 的 Core-first 与 Wafer-first 协同

- **Core-first**：枚举 64 行 Core→`N_x,N_y`→最小 W/H；若达不到面积下界或 HBM/D2D 通信必要面积，按源 Word 的三种边界方案（仅增 H、仅增 W、同比增长）构建有限候选；枚举 n/摆放；筛曝光场、面积、长宽比、D2D、模块和 Wafer 整数网格；再计算资源与 Pareto。对 D2D 余量本身作为优化目标的搜索，应扩充边界候选，不能宣称该有限规则覆盖全部连续 W/H。
- **Wafer-first**：给定正整数 `M_x,M_y` 与 `S=215√0.9`，模块允许 `W_module≤S/M_x,H_module≤S/M_y`，推出 `W_die≤S/M_x−0.30,H_die≤S/M_y−22.56`，还须验证 `row_width+0.30≤S/M_x`；在该范围反推合法 `N_x,N_y` 和 W/H 几何断点。两条方法在**同一枚举规则/约束**下交叉验算，不是两个不同目标函数。
- **目标 A**：整片 PE INT4 名义峰值优先，依次最大 Core 总数、HBM 容量与单向 D2D 余量；FP32 Vec 峰值单列且不能与 INT4 相加。**目标 B**：整片 HBM 容量优先，并跟踪 Core 数、带宽及未验证的真实布线风险。输出两目标非支配前沿，并按搜索目标确定非支配设计。

### 10.4 代表性动态几何结果与复算

源 Word §2.6 的示例均为 `large/0.25 MiB`，`L_c≈2.56355 mm`、面积窗口 `200–800`、单曝光场、长宽比≤2、总外围余量 0.10 mm、`215×215` 模块区域。下表是**源示例的舍入展示**，实现以未舍入几何重新计算：

| HBM/模块 | Die W×H（约 mm） | A_die（约 mm²） | Core 网格 | 模块阵列 | Core/区域 | HBM GB/区域 |
|---:|---|---:|---|---|---:|---:|
| 2 | 18.04×28.30 | 510.7 | 7×11 | 11×4 | 3388 | 3168 |
| 4 | 20.61×28.30 | 583.2 | 8×11 | 9×4 | 3168 | 5184 |
| 2 | 12.92×18.04 | 233.1 | 5×7 | 15×5 | 2625 | 5400 |
| 4 | 20.61×18.04 | 371.9 | 8×7 | 9×5 | 2520 | 6480 |
| 4 | 20.61×10.35 | 213.4 | 8×4 | 9×6 | 1728 | 7776 |

对**计算优先**的代表方案，四档 B 的相同 `7×11 Core/Die,11×4` 模块阵列是因 Core `L₀` 固定；整片 INT4 PE 理论峰值依 B64/128/256/512 分别约 **32.565/31.624/29.618/25.058 POPS**，Vec FP32 等效峰值约 **3.256/3.163/2.962/2.506 PFLOPS**。这些不是目标 BF16 的有效吞吐。对**容量优先**可到 7776 GB、216 颗 HBM（该工程窗口），但 Core 总数降至 1728；不同目标没有唯一最优面积。源 Word 的 150 mm² 下界与 3:1 比例单项敏感性并未改变代表核最优值，而去掉 0.10 mm 外围示例余量改变了计算峰值方案的 Core 数：这体现整数网格对假设的敏感性。

### 10.5 上下层接口、输入完整性和实现条件

`Core-first/Wafer-first` 的输出传给单 Die 仿真器时，须提供 `W_die,H_die,N_x,N_y,Core64_key,n,placement,ingress_map` 以及对应的工程约束身份；仿真器对本地任务生成 SRAM/NoC/HBM/PE/Vec 资源预算，并在校准完备时求得执行时间。上层网络可使用 D2D 预算；Wafer 模块数属于容量/装配分析，不隐式设置运行时 `rank_count`。`noc_width_bits` 仅定义 Die 内 NoC 链路，不作为 Die 间 D2D 物理带宽。缺少 W/H/Nx/Ny 的输入返回几何信息缺失状态；HBM 排宽超过计算 Die 宽度时输出 `physical_layout_unverified`。实际 reticle、HBM PHY、功率/热、良率、封装结构和 NoC 拥塞须按独立工程校验结果报告。
