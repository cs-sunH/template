# OS GEMM E5 粒度审计（只读）

日期：2026-10-14。此文只审计当前 `append_epoch`、保存的 E5 比较和既有周期压缩证据；没有重建、运行新数值模型或改生产代码。当前工作区 `Mapping.cc` 的未提交改动位于 workspace/attention 辅助路径；下述 `append_epoch` 函数体未出现在当前 diff 中。E5 数字来自保存的 `compiled-grouped-oracle-report.json`，该报告使用 `compile()` 和实际 Runtime，但硬件为合成单核模型，不是硅片测量；整棵当前脏工作区没有重建。

## 失败是什么

E5 夹具为 FP32 OS、单核，`M=65,N=65,K=4097`，选中 `tm=16,tn=64,tk=64`。因此 `Tm=5,Tn=2,Tk=65`，有 `R=10` 个输出轮次和 `R*Tk=650` 个 K 消费。`append_epoch` 收到的是 `frames=R=10`，不是 650。计划固定为 5 个宏阶段：load-first、first-overlap、interior-repeat、last-overlap、store-last。中间段 `repeat_count=8`，但 [Compiler.cc](../../src/Compiler.cc#L632) 明确规定它只是审计元数据；work 和 `credit_delta` 是整个 epoch 在 `progress∈[0,1]` 上的总量。

保存的 grouped-Q/alpha 对照结果如下。表中误差是实际 Runtime 相对独立离散 K-wave oracle 的相对误差。

| 场景 | 完成误差 | HBM busy 误差 | shared-HBM busy 误差 |
|---|---:|---:|---:|
| 单请求 | 3.827% | 124.951% | 不适用 |
| 两请求，同步 | 两者各 3.827% | 124.951% | 124.951% |
| 两请求，第二个晚 50 ms | 10.782%、0.535% | 158.869% | 994.515% |

单请求的实际/期望 finish 是 `0.226286/0.235291 s`，HBM busy 是 `0.168116/0.074734 s`。同步双请求的 finish 是 `0.452572/0.470582 s`，HBM 和 shared-HBM busy 均为 `0.336231/0.149469 s`。账本检查通过：compute work 完全守恒，compute busy 误差约为 `10^-15`。偏差主因不是漏算 ops/bytes，也不是一般的舍入误差，而是 E5 时间拓扑没有保留消费波边界。

[Mapping.cc 的 `append_epoch`](../../src/Mapping.cc#L800) 用 `W=frames` 平分 load/compute/store 总流量。对 `W>2`，它把 `W-2` 轮合并成一个 interior epoch：三个活动的 work 和 credit 都乘 `W-2`，并在单个 epoch 中由各活动自己的进度坐标改变整批 credit。以本夹具为例，一个 credit/frame 代表 65 个 K 消费；两输入/两输出容量约束因此作用在整批 8 个输出轮次的归一化进度上，而非每个 K 波完成后复位。独立 oracle 则只在相应 K-wave Load 完成后释放 Compute；只有 final-K Compute 才产生输出，Store 在其后运行。[compiled_gemm_oracle.py](../../validation/compiled_gemm_oracle.py#L160) 明列了这些离散依赖，并将 K 波数校验为 `R*Tk`。

形状尾块也是可分离的结构轴。每个 K 消费的 FP32 工作由

`LoadBytes = 4*k_i*(m_i+n_j)`，`ComputeOps = 2*m_i*n_j*k_i`，`OutputBytes = 4*m_i*n_j`（仅 `final_k`）

决定。该夹具的 `m_i∈{16,1}`、`n_j∈{64,1}`、`k_i∈{64,1}`，因此有四种 M/N tile 类和 full/tail K 两类。E5 只把总流量均分到 10 个输出轮次，会抹平这些不同 tile 的成本和 final-K 边界。现有失败把 K 聚合、边缘 tile 及竞争相位混在一起，尚无单因素运行可精确分摊它们各自的百分比。

辅助差异要单独处理：账本与 Compute busy 的守恒说明浮点数值误差不是这里的大项；其量级远小于 3%/5%。另一方面，旧 equal-progress 参照在相位偏移夹具上会再改变 finish/shared-busy 误差；当前 grouped oracle 才是报告选定的 owner-Q/alpha 参照。不要把错误的公平性参照与 E5 粒度差混作一个“epsilon”。单因素 K/shape 反例和原始数值见[保留的 grouped oracle 报告](../evidence/cache-lifecycle-20261014/compiled-grouped-oracle-report.json)。

## 最小表示建议

增加更多静态 E8/E16 阶段本身不够：若 interior 仍是一次性聚合 progress，`Tk` 继续隐藏在一个 epoch 内。可实现的最小方向是让 E8/E16 表示固定的**事件类别模板**，并给中间模板加有语义的重复描述：

- 将输出 tile 按 `(m_i,n_j)` 边界类做有序 run-length 编码（至多四类）；每类的 K 序列只需 full-K body 和一个 K-tail/final 边界。
- 每个非 final-K body 表示一波 Load/Compute，只有输入 credit 变化；final-K body 在 Compute start 预留输出槽，Compute terminal 转为 Ready，再启动 Store 并于 terminal 释放。保持两个输入槽、两个输出槽和现有 pump 优先级。
- body 保存一波的整数 work/credit/ledger 增量及重复数。重复不应继续沿用目前“总量乘数 + 一个 progress”语义，而应在每波边界保持队列/credit 语义。
- 只有当活动 owner 集合、shape/K phase、槽位、credit、LP 速率类别和事件次序构成的联合状态返回，且所选 deadline/LP 区域在整个重复域内已证明稳定，才可按周期跳过：`time += N*Δt`，各 busy/work ledger `+= N*Δledger`。一般仿射返回用有界幂合成；未证明闭合时按原预算逐事件推进并 fail-closed。repeat 不得重置 lifetime event cap。

为守住预算，活动事件上界应取决于固定 shape 类、prologue/tail 模板、owner/外部到达或 capacity-change 边界数，而不取决于 `R*Tk`。周期幂合成可花 `O(log N)` 次固定维数算术，但不能生成 N 个 Runtime event；没有可证明周期时，不能宣称该上界。事件记账和 cap 仍是全生命周期的。

既有 [nested-K/output affine 结果](handoff-runtime-20261014/affine/nested-k-output.md) 已证明一个单核、全 M/N tile、相位 `1/20` 的窄场景可把重复 K/output 段压成固定事件模板和 affine ledger map；它没有覆盖本 E5 夹具的 M/N 尾块，也没有证明任意 phase/LP cell。可复用该边界证明思路，不需要重做固定宽有理数 primitive。

这种解析表示有望使**该固定夹具**达到 3%/5%，因为当前失败由批量化的就绪/credit 节拍主导、且波级 oracle 与工作账本闭合；但尚未验证，不能给出 pass 承诺。单请求可用固定 rates 下的 `T=T_prologue+N*T_period+T_tail`、`Busy=Busy_prologue+N*Busy_period+Busy_tail`。并发时 `T_period` 只能在联合活动集及 grouped-Q/alpha LP 区域保持不变的周期内复用；单一 roofline `max(total_work/capacity)` 无法保留输出背压、共享 busy union 和相位差。

## 可检验反例和验收

用同一合成硬件、同一选中 tile 尺寸（每次确认计划仍为 `16×64×64`），比较 compiled plan 与现有独立 grouped K-wave oracle，不调容差：

1. K 聚合轴：`M=80,N=128` 下比较 `K=128`、`K=4096`、`K=4097`（R=10，Tk 分别为 2、64、65）；区分 K 深度与一个短 K tail。
2. M/N shape 轴：固定 `K=4097`，比较 `M=80,N=128` 与 `M=65,N=65`（同为 R=10、相同 tile、Tk=65）；前者全 tile，后者含 M/N 尾块。
3. 对上述 pair 分别跑单请求、同步双请求和 50 ms phase offset；检查每个 finish、HBM union/shared busy 和 final-K 输出时序是否达到原 3%/5% 门槛。
4. 把同一稳定 body 的 K repeat 扩大数个数量级；计划描述长度和展开 Runtime event 数须保持固定，ledger/time 采用周期增量，lifetime cap 跨 wake/repeat 保持不变。另给一个 deadline 或 LP 区域跨越的反例，确认周期条件失败时 fail-closed，而不是越过边界跳过。

若 1/2 仍失败，剩余结构轴清楚可定位；若 3 失败而单请求通过，问题在联合 owner/phase map；若 4 的周期区域证明不闭合，则不能用固定 E 数量声称 repeat-independent。E5 当前仅有 5 宏阶段，E8/E16 在生产计划中仍不可用；这里没有把 scratch/affine pass 当成生产精度证书。

审计快照 SHA-256：`Mapping.cc` `5dc4a7ccea972d15e30371bc0c8534b3ffbdf5921bf759d918fd096737c0cd90`；`Compiler.cc` `364f99da34222f52ab2ed2a20d8b7e2db6fb415767cb7774bf1b0d7c238abd98`；`compiled_gemm_oracle.py` `212d75c8521fad3da87b9dca1f6e5a6807c9aa86cbb4f74067169da21ce272c4`；grouped report `e18e1c67f67988a7d43c770fe2256660457e30f3732a686eee45a354c7003c7c`。这些是工作区审计快照，不是重建二进制身份。
