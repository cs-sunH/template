# relevant_distributed 溢出负载 A/B 对照 + R2 初裁 + H28 ratio 扫描实验报告

- 日期：2026-09-02（T1 批次，总文档 §9-R2 与 §4 裁决 #11/#22/#28 的正式验收实验）；
  **R2 复裁批次 2026-09-03**（T_max 扫描实证 + 等价性论证，§4 重写为"不触发降级"，
  新增 §2.1 A/B 差分解式与 `tmax_scan.csv`；run 产物 `/tmp/wscllm_r2re/`）
- 对象：`template/astra-sim-wscllm`，`kv_cache_policy=relevant_distributed`（第三变体，B1–B4b 交付态）
- 运行：4 次真仿真（physical / ideal_masked 双臂 + H28 r=1、r=4 两点），全部
  `cpp_exit=0 python_exit=0`，run 末守恒门（allocator assert_final_state + journal
  重放 resident=0/reserved=0/physical=weight）随 exit 0 通过；`KV_DELTA_JOURNAL=on`、
  `SH_METRICS_DETAIL=full`、`SH_ARCHIVE_RUN=0`；C++ 零改动、红线三文件零触碰、
  legacy/session_lru 行为与测试未动、全程未 commit。
- 原始产物：`/tmp/wscllm_t1/runs/{physical,ideal_masked,h28_r1,h28_r4}`；
  指标中间件：`/tmp/wscllm_t1/metrics/`；本目录 CSV 为正式数据表。

## 0. 结论摘要

| 项 | 结论 |
| --- | --- |
| 负载触发 | 3 个溢出成员（B0/C/B1，prefill 段 D-piece 吃满后 stay 溢出到 P）+ 真实 1000 族拉回（B1，d0→P4，23,640 tok / 12.39 GB）；3100/3300 守恒逐字节精确 |
| R1（max() 汇合真的并行） | **PASS，未触发停止条件**：观测窗口比"串行相加"模型低 20%，仅比 max() 模型高 6.2%（残差=读边投递/exit 汇合尾 ~10.7 ms）；且第 2 列车在 +26.8 ms 发射（体完成即续车），而第 1 列车读边 +81~97 ms 才完成——读边不阻塞体 |
| A/B（physical vs ideal_masked） | 非溢出请求双臂 e2e **逐字节相等**（0 差异，开关隔离干净）；3 个溢出成员 physical 反而**快 49.4~49.7 ms**（decode 窗口 171.4/171.5/174.3 ms vs ideal 221.0/221.0/223.9 ms）：ideal 臂按现状口径把全量 KV 记为 D 端 HBM 读（16 迭代 × 12 ms），physical 裁剪本地分量 + 3300 源端读与体重叠 |
| R2 初裁 → **复裁修正** | 初裁按公式触发（口径 a：读边拖尾 p95 / TPOT p95 = 102%；口径 b ≈ 1101%），登记"建议降为迭代级"。**复裁（§4）：不触发降级——原度量为误归因；聚合口径与逐迭代口径在本架构下计时等价（论证 + T_max∈{4,8,16} 扫描实证，差异 ≤0.12% ≪ 5%）；102% 读尾属 H28 远程读瓶颈信号**（与 r*≈3 > 2.47 交叉自洽） |
| H28 拐点 | 单流口径 r=1 即 HBM-bound（§7 算例 A 的 NoC 25 µs < HBM 63 µs 形状复现）；**本拓扑实测拐点 ≈ r=3**（6 对 XY 读边在列合流链 3 流公平分享，有效 NoC=d2d/3）：r=2.47 仍残留 ~13% NoC 分量（eff 1425.6 GB/s = 86.9% HBM），r=4 达 HBM 界 98.3%（1613 GB/s），再增 ratio 收益 <2% |
| §7 算例 A 交叉核对 | r=2.47 单流每迭代最大 shard 读：NoC 4.28 ms < HBM 10.56 ms，比值=2.47=d2d/hbm ✓；实测边有效带宽 1425.6 GB/s 落在 [3 流 NoC 1350, HBM 1640] 带宽界内 ✓ 自洽 |

## 1. 负载设计与触发核验（步骤 2）

容量事实（B4b 校准）：validation-160gib 每实例空域 1,723,921 tok（绑定约束
rank0/1 整头 shard 98,304 B/tok）。hand-craft 队列 9 行（t=2 ms 同刻突发 8 行 +
闭环 turn-1 1 行），构造器 `/tmp/wscllm_t1/make_overflow_queue.py`（provenance
边车经 `derive_20_first_30_seconds.queue_provenance` 现算，C++ 首开比对通过：
`session_blocks_contiguous=true`、`turn0_adjacent_inversions=0`）：

| queue | request | prefill/decode | 落点 | 角色 |
| --- | --- | --- | --- | --- |
| q0 | t1s0_r0 (A) | 1,700,000/16 | P1→d0 | 占位者：d0 空域吃满（stay=0，零 3300） |
| q1–q5 | t1s1..t1s5_r0 | 200/8 | P4..P8 | 填充：铺满 6 个 P，迫使 q6/q7 落回 P1/P4 |
| q6 | t1s7_r0 (C) | 200,000/16 | P1→d0 | 溢出成员①：FCFS 阻塞至 A drain（活体背压→释放→重准入），晚期窗（~2.596e12 ns）读边 |
| q7 | t1s6_r0 (B0) | 200,000/16 | P4→d0 | 溢出成员②：f1 快速 drain 后 ~4.4 ms 即准入，干净窗（~37.6e9 ns，无争用） |
| q8 | t1s6_r1 (B1) | 3,000/16（history 200,016） | P4→d0 | 多轮会话：turn-1 释放 B0 旧 placement + 真实 1000 拉回（d0→P4）；A 仍驻留使其自身也溢出（成员③） |

决策日志核验（进入双臂前）：`kv_placement` 三成员 prefill 段 pieces 均为
`[d0 scatter_remote → P prefill_stay]`（全序 D=0<P=1）；C 的 stay=176,168 tok 与
golden R1 的 B **逐字节一致**（同容量下确定性复现）；`kv_scatter`(3100) 54 行、
`kv_remote_reads`(3300) 36 行（3 成员 × 2 列车 × 6 shard）、`history_pull_routes`
(1000) 6 行（源 d0，非 P 源真实边）——与设计一致后才开双臂。

## 2. A/B 双臂差分（步骤 3/4a）

数据表：`ab_diff_per_request.csv`；per-request 明细：`per_request_{physical,
ideal_masked}.csv`（TTFT 为 train_interpolated 代理，SPLIT=off 交付缺省；后 4 个
filler 为"实例末列车"代理不可得=NA，属已知口径）。

| request | role | e2e physical (ns) | e2e ideal (ns) | Δe2e | TPOT physical | TPOT ideal |
| --- | --- | --- | --- | --- | --- | --- |
| t1s0_r0 | A 占位者 | 2,560,076,078,044 | 同左 | **0** | 105.06 ms | 105.06 ms |
| t1s1..t1s5_r0 | 填充×5 | — | — | **0**（逐字节） | — | — |
| t1s7_r0 | 溢出 C(P1) | 2,596,105,227,331 | 2,596,154,833,258 | **−49,605,927** | 10.4997 ms | 12.9415 ms |
| t1s6_r0 | 溢出 B0(P4) | 37,729,241,544 | 37,778,646,851 | **−49,405,307** | 10.5123 ms | 12.9408 ms |
| t1s6_r1 | 溢出 B1(P4,t1) | 1,275,192,718 | 1,324,861,321 | **−49,668,603** | 10.6811 ms | 13.1155 ms |

读法：
- **开关隔离**：无 3300 的请求（A/填充）双臂产物逐字节相同（A/B 仅差在 3300
  发射与 local_kv_bytes 裁剪，二者对全本地成员互相抵消——与 B4b 2 s 窗零差结论
  同机理，本负载给出了溢出成员上的非零差分）。
- **溢出成员 physical 更快**：ideal 臂 = 现状归因口径（全量 KV 记 D 端本地 HBM
  读：每迭代 rank0 19.7 GB/1640 ≈ 12 ms × 16 迭代 ≈ 192 ms 体）；physical 臂体只
  读 D 本地 23.8k tok 分量（~1.4 ms/迭代）+ 远程分量改由 3300 在源端计费并与体
  重叠。Δ ≈ −49.5 ms（三成员一致到 ±0.15 ms，确定性）。
- TPOT（(completion−TTFT)/decode_len）：physical 10.50/10.51/10.68 ms，ideal
  12.94/12.94/13.12 ms；physical = ideal × 0.811~0.814。

### 2.1 A/B 差分解式（复裁批次补充：physical − ideal = + 真实远程读代价 − D 端归因虚增）

以 t1s6_r0（B0）逐项展开（rank0 口径；三成员同构，±0.15 ms 内一致）：

| 分量 | ideal 臂（现状归因口径） | physical 臂 |
| --- | --- | --- |
| 列车体（17 类聚合节点） | 16 迭代 × (全量 KV 19.66 GB 记 D 端读 12.0 ms + 计算 ≈1.8 ms) ≈ **221 ms 串行**，exit = 体尾 | 16 迭代 × (D 本地 piece 2.34 GB 1.43 ms + 计算 ≈1.9 ms) ≈ **54 ms**（`local_kv_bytes` 裁剪） |
| 3300 读链 | 不发射（Σ3300 = 0，按设计） | 源端串行 277.4 GB / 1640 GB/s ≈ **169 ms**，侧插与体重叠 → exit = max(54, 169) + 投递尾 ≈ **171.5 ms** |
| 差分 | — | Δdecode = 171.5 − 221.0 = **−49.4 ms** |

- **"溢出成员 physical 反快 49.5 ms"不是"远程读免费"**：physical 臂的远程读 169 ms
  恰是关键路径（exit 被读链锚定，§3 的 max 模型 +6.2% 残差即读边投递尾）。physical
  更快的机理是**归因口径差**：ideal 臂把远程 277 GB 也记成 D 端本地 HBM 读、并让计算
  串行排在其后（体 = 221 ms）；physical 臂把远程分量从体中拿掉（体 = 54 ms）、改由
  3300 在源端计费并与计算重叠（读链 169 ms < 221 ms）。
- 恒等式：Δ ≈ +169（真实远程读成为关键路径）− 192（D 端全量 KV 记账虚增，16 × 12 ms）
  − 29（ideal 中计算串行不与任何流量重叠）+ 尾差 ≈ −49.6 ms。即 ideal 的窗口高估
  来自"全量 D 读 + 计算串行"双重保守，而非 physical 少计了远程代价——physical 对
  远程字节的计费（源端 COMM_READ）一分未少（hopbytes 差 = Σ3300 × 3 hops 精确，§6）。

## 3. R1：列车 exit vs 体尾（步骤 4b，未触发停止条件）

数据表：`exit_body_drag_physical.csv`。口径：
- exit = request `completion_ns`（C++ 精确，末列车 exit 标记）；
- 体尾代理 = `body_gate`（第 2 列车发射 tick − decode_start，= 第 1 列车体完成
  + 决策滞后；第 2 列车体时长按同构 ≈ body_gate），`body_end ≈ decode_start + 2×body_gate`；
- 读边完成 = 3300 send 节点 memory_anchor tick（§7 口径说明）。

| 成员 | 窗口 (ms) | body_gate (ms) | exit−体尾 (ms) | 串行模型 (ms) | 串行残差 | max 模型 (ms) | max 残差 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| t1s7_r0 | 171.36 | 26.90 | 117.56 | 214.45 | **−20.1%** | 160.65 | **+6.2%** |
| t1s6_r0 | 171.55 | 26.79 | 117.97 | 214.40 | **−20.0%** | 160.82 | **+6.2%** |
| t1s6_r1 | 174.27 | 26.94 | 120.39 | 217.25 | **−19.8%** | 163.37 | **+6.2%** |

- 串行模型 = 2×体 + Σ各列车最长边（若 recv 链体尾，读全长暴露）；max 模型 =
  max(2×体, 读链)。观测窗口显著低于串行上界（−20%）、贴近 max 模型（+6.2%，
  残差即读边投递/exit 汇合尾，见下）。
- **决定性时序证据**：第 2 列车发射于 +26.8~26.9 ms（体完成即续车，in-flight
  门放开），而第 1 列车读边 +81.0（5 头）/ +97.2~98.8 ms（6 头）才完成——若读边
  链体尾（串行），第 2 列车不可能早于 +81 ms 发射。
- exit − 最后观测读边完成 = 10.71/10.72/10.89 ms（近似常数 = 末列车 6 头边的
  投递/汇合尾；该边 send 锚有已知覆盖缺口，见 §7）。
- **R1 判定：PASS（max() 并行语义实测成立），不写 R1_FAILURE，扫描继续。**

## 4. R2 复裁：**不触发降级**（初裁"建议降为迭代级"修正，2026-09-03）

### 4.0 初裁记录（保留作历史，已被本节推翻）

- 口径 a（汇合尾）：读边拖尾 = exit − 最后读边完成 = {10.71, 10.72, 10.89} ms，
  p95（nearest-rank, n=3）= 10.89 ms；TPOT p95 = 10.68 ms → 比值 **102% > 5%**。
- 口径 b（读链超出体）：拖尾 = 窗口 − 2×体 ≈ {117.6, 118.0, 120.4} ms → /
  TPOT p95 ≈ **1101% > 5%**。
- 初裁登记：R2 触发，建议降为迭代级（并附注：建议降级实现前先做迭代级对照复核——
  本节即该复核的执行与结论）。

### 4.1 等价性论证（初裁度量为误归因的三点论证，均附代码/配置事实）

**(a) 列车体是折叠聚合节点，体内不存在逐迭代节点——per-iteration 读边无论如何只能
汇合于 exit 标记。**

- `generate_trace.py:770-1097` `transformer_pass_aggregated`：docstring 明示
  "**Fold repeated Transformer passes into 17 aggregate Chakra nodes**"；13 类 layer
  类目（`:897-911`）按层聚合 + 2 条 all_reduce + final_layernorm + logits = 17 节点；
  权重分量按 `weight_passes`（`:785`，`:1062-1068` 计入）、KV 分量按 `local_kv_bytes`
  （`:786`，`:944-948` 逐 span 覆盖）——**迭代维度被折叠，体内没有任何可被逐迭代
  依赖的节点**。
- `online/graph_batch_builder.py:32-34`：列车发射结构 = [join 标记] → 17 类聚合体
  → [exit 标记] → 共享 end barrier；DECODE_COMPLETION watch 成员 = exit 标记节点。
- `relevant_kv_emission.py:580-583, 728-745`：3300 recv 侧插依赖列车 join 标记，
  recv 节点 id 注入**该成员自己的 exit 标记**（成员 exit = max(体尾, 其全部读边)，
  总文档 §3.2/§5.5）。
- 推论：即便把 3300 改为逐迭代发射，每条 per-iteration recv 在构图上仍只能依赖
  join 标记（体内无逐迭代锚可用）并汇合于同一 exit 标记——**逐迭代粒度改变的只是
  边数与单边字节，不改变任何汇合点的依赖集合**。唯一能真正逐迭代门控读的路径是把
  列车体展开成 N×17 个逐迭代节点，与 house-style 聚合哲学正面冲突（总文档 §6 让步 3
  已接受同类折叠；`transformer_pass_aggregated` 的 17 类折叠与 `weight_passes` 先例
  均是该哲学的既有实现）。

**(b) 流体后端公平分享下，同总字节、同链路的边集合之"最后字节完成时间"对切分粒度
不敏感（线性性）。**

- 公平分享模型（一手核验）：`extern/network_backend/analytical/congestion_aware/
  fluid/FluidScheduler.cpp:256-258`——每条活动流速率 = min over 路由链路
  `capacity_Bpns / active_flows.size()`；流体推进线性：`:196-197`
  （bytes += elapsed × rate）、`:225-233`（完成 = now + ceil(remaining/rate)）。
- 逐跳时延常数：`topology/Topology.cpp:47,58` 路由各跳 latency 求和取 ceil 为
  `propagation_latency_ns`，`FluidScheduler.cpp:469` 在流完成时一次性附加——
  **与流大小/切分粒度无关**（本拓扑 P/D 直连 1 跳、5 ns/跳、4050 GB/s；
  `hardware/face_case5_config_c.json`）。
- 等价前提（四条，本架构全部满足）：① **同总字节**——decode 段钉 D ⇒ 每迭代远程
  字节为常量（Σ3300 = decode_len × Σ_s R_m 精确式），T_max 只改切分不改总量（三档
  守恒门逐字节精确过，§4.2）；② **同链路集**——3300 路由 = 源 rank→D rank 静态 XY，
  与 T_max 无关；③ **边背靠背无空隙**——侧插读边 join 后连续流动，同成员相邻列车
  的边在源端 HBM 与链路上背靠背/并发衔接（in-flight 门体完成即续车，§3 的 +26.9 ms
  证据），无协议空转期；④ **源端 HBM 串行计费**——send `hbm_charge=true`
  （COMM_READ，与本机负载严格均分 local-mem-bw=1640 GB/s，
  `MetricCollector.cc:1184-1185`），总源端 HBM 时长 = 总字节/1640 是粒度不变量。
- 线性性推论：1 条 277 GB 边在分享率 r 下最后字节完成 = 释放 + 277/r；N 条合计
  277 GB 的边背靠背（各 277/(N·r)）合计同为 277/r，N 条并发（各得 1/N 分享率）同时
  完成于同点——两种切分下"最后字节完成时刻"相同；残余差异只有竞争流集合的分享
  重分配二阶项，本负载下被源端 HBM 界吸收（§4.2 实测：三档 exit 互差 ≤201 ns）。

**(c) 结论：逐迭代发射不改变 exit 时刻（§4.2 实证），R2 降级被否决。**

### 4.2 T_max 扫描实证（SH_TRAIN_MAX_ITER ∈ {4, 8, 16}，同负载同种子，判据 <5%）

方法：复用 T1 溢出队列（构造器 `/tmp/wscllm_t1/make_overflow_queue.py`，provenance
逐字节一致 fnv1a64=5524234870160433653）与 run 管线，仅改 `SH_TRAIN_MAX_ITER` 重跑
physical 臂三档。每档 exit 0、9/9 completed、KV_DELTA_JOURNAL=on、3100/3300 守恒
逐字节精确。**确定性自检（T8 重跑逐字节一致）**：T8 档为与 T1 physical 臂**同配置
（SH_TRAIN_MAX_ITER=8）的重跑**，比较对象范围 = T1 physical 臂的产物；其全部仿真
语义产物（决策日志/列车账本/kv journal/request journal/request_metrics）与 T1
physical 臂逐字节相同，cpp.log 仅差 host 计时字段（wall_ms/wall_seconds 时间戳）。
注意口径区分：本条"T8 重跑一致"是**同配置重跑的确定性自检**（T8 vs T1，不涉及
T4/T16）；下文"三档互差 ≤201 ns"是 **T4/T8/T16 三档之间**的跨档差异——二者比较
对象不同，不互斥。数据表 `tmax_scan.csv`；run 产物
`/tmp/wscllm_r2re/runs/{tmax4,tmax8,tmax16}`，指标中间件 `/tmp/wscllm_r2re/metrics/`。

| T_max | 列车×迭代 | TPOT p50/p95（proxy）| 每 token decode 期（=窗口/16，精确量）| exit（t1s7_r0, ns）| exit−最后读锚 |
| --- | --- | --- | --- | --- | --- |
| 4 | 4×4 | 10.4997 / 10.6684 ms（−0.120%）| 10.72157 / 10.89158 ms（−0.00003%）| 2596107227256 | 10.710 ms |
| 8（基线）| 2×8 | 10.5123 / 10.6811 ms | 10.72157 / 10.89159 ms | 2596107227331 | 10.710 ms |
| 16 | 1×16 | proxy 退化（见注）| 10.72158 / 10.89159 ms（+0.00003%）| 2596107227403 | 0.000101 ms |

- **判据全过（<5%）**：completion/decode 窗口三档互差 ≤201 ns（171 ms 尺度，
  ≤0.00012%）；每 token decode 期（proxy 无关的精确量 = 窗口/decode_len）互差
  ≤0.00003%；TPOT proxy 差 −0.12%（纯插值结构差：首车覆盖 4 vs 8 迭代，插值点
  位移）；读链末端（exit−ε）三档同锚定于 ~171.36 ms——**计时对切分粒度不变**。
- 单列车边时长按字节线性缩放并精确落在 HBM 界：T16 单边 277.09 GB / 171.36 ms =
  **1617 GB/s = 98.6% HBM**（无同对流竞争的干净单流测量）；T8 每边 138.5 GB，
  首末边同 (src,dst) 对重叠（+26.9 ms 续车）→ 97.2 ms / 138.5 GB = 1425 GB/s =
  86.9% HBM（同对流 2 流分享税，与 §5 的 3 流 NoC 列合流分享同族现象）。
- **初裁口径 a 度量本身是列车结构量**：exit−最后读锚在 T4 与 T8 **完全相同**
  （10.710/10.721/10.891 ms）而在 T16 塌缩为 101 ns——它度量的是"末列车 recv 投递
  尾"（末边从 8 迭代扩到 16 迭代时被读链自身吸收），**不是逐迭代 pacing 失真**；
  初裁把它当 p95 失真证据属误归因。读链整体超出体的口径 b（~117 ms）在三档同样
  不变——那是字节量与源端 HBM 带宽之比（277 GB/1640 ≈ 169 ms ≫ 体 54 ms），
  与粒度无关。
- 注（观测层伪影，非计时缺陷）：T_max=16 时成员单列车，TTFT train_interpolated
  代理需要实例上的后继列车 tick 作体末近似而不可得或失真（NA / 插值到无关列车 /
  clamp 到 completion）；该 proxy 是显示指标（first_token_source 被一切 SLO 判定
  路径拒绝，`metrics_postprocess.py:532-534`），proxy 无关的精确量不受影响。

### 4.3 102% 读尾重新归因：H28 远程读瓶颈信号（与 §5 互证）

- 交叉自洽：§5 实测掩盖失效拐点 **r\* ≈ 3 > 主硬件比值 2.47**——在 2.47 处读边仍
  残留 ~13% NoC 分量（eff 1425.6 GB/s = 86.9% HBM），读链 ~169 ms ≫ 体 54 ms，
  读尾自然构成关键路径尾部（§3 max 模型的 +6.2% 残差即其投递尾）。102% 的读尾/
  TPOT 比值正是"远程读在 2.47 配置下处于关键路径"的量化表达，与"2.47 未到拐点"
  同一事实的两种读法。
- 方向验证：ratio↑ 读链相对体缩短、尾部形态随之改变（§5 表：r=1→4 时窗口
  423→171→124 ms、exit−末锚 0→10.7→38.1 ms）——读尾是硬件比值与字节量的函数，
  不是聚合粒度的函数（§4.2 三档不变已直接证明后者）。
- **裁决：R2 不触发降级——原度量为误归因；聚合口径与逐迭代口径在本架构下计时
  等价（论证 + T_max 扫描实证）；102% 读尾属 H28 远程读瓶颈信号。总文档 §6 让步 3
  （列车级读聚合）维持，§9-R2 关闭。**

## 5. H28 三点扫描（步骤 6）

数据表：`h28_three_point.csv`（每点 3 成员 × {6 头, 5 头} 列车①隔离边；边时长 =
send 节点锚完成 − send 就绪（=decode_start）；D2D=r×1640 GB/s，HBM=1640 GB/s
恒定；派生 hardware json 在独立目录 `/tmp/wscllm_t1/h28_hw/`，主 json 未动，
runtime_config 按 slug 独立派生目录，runner 经 `SH_RUNTIME_CONFIG_DIR` 覆盖）。

| 点 | d2d (GB/s) | 6 头边时长 (ms，3 成员) | 有效带宽 (GB/s) | HBM 界 (ms) | 单流 NoC (ms) | 3 流 NoC (ms) | decode 窗口 (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| r=1 | 1640 | 240.03 / 240.27 / 244.07 | 577.2~577.4 | 84.5~85.9 | 84.5~85.9 | 253.4~257.7 | 423.2 / 423.6 / 430.3 |
| r=2.47 | 4050 | 97.20 / 97.30 / 98.83 | 1425.4~1425.8 | 84.5~85.9 | 34.2~34.8 | 102.6~104.4 | 171.4 / 171.5 / 174.3 |
| r=4 | 6560 | 85.91 / 86.00 / 87.36 | 1612.6~1613.1 | 84.5~85.9 | 21.1~21.5 | 63.4~64.6 | 124.1 / 124.2 / 126.2 |

判读（口径：边时长随 ratio 是否继续下降 + 实测有效带宽对 HBM/NoC 界的贴近度；
日志无 NoC/HBM 分量直接分解，故用对比法）：
- r=1→2.47：边时长 −59.3%（NoC 是共同瓶颈：6 对读边的 XY 路径在列合流链
  （14→20 等）3 流公平分享，有效 NoC=d2d/3=546.7 GB/s 主导，观测 577 GB/s）；
- r=2.47→4：仅 −11.6%，有效带宽 1613 GB/s = **98.3% HBM 界**（此点 3 流 NoC
  2187 > HBM 1640，HBM 接管），再增 ratio 边时长趋平于 ~84.5 ms+ε（收益 <2%）；
- r=2.47 点：有效带宽 1425.6 GB/s = 86.9% HBM，落在 [3 流 NoC 1350, HBM 1640]
  之间——HBM 为主、残留 ~13% NoC 争用分量。
- **掩盖失效拐点初值：r\* ≈ 3**（d2d/3 = 1640 处两界交叉）。即：本拓扑 6 对读边
  并发的真实口径下，ratio ≥ 3 读边才完全 HBM-bound；2.47（主硬件实值）已掩盖
  大部分但未到界。单流口径（§7 算例 A 的口径）拐点在 r=1：r=2.47 时单流
  NoC 4.28 ms ≪ HBM 10.56 ms（每迭代最大 shard 17.32 GB），**与 §7 的
  NoC≈25 µs / HBM≈63 µs 同形状（比值恒为 2.47）**——理论值按带宽定义精确复现，
  实测差异全部来自多流链路分享，非模型失真。
- 窗口随 ratio 423→171→124 ms 单调下降且尾部（exit−最后观测读边）0→10.7→38.1 ms：
  r 越大读边越早完成、exit 越贴近体+汇合尾，与"HBM 界内读边被体/重叠掩盖"的
  方向一致。
- **与 §4.3 互证**：R2 复裁把 T1 初裁的 102% 读尾归因为本节 r\*≈3 > 2.47 的远程读
  瓶颈信号（读链主导关键路径），而非列车聚合伪影（T_max 扫描三档计时不变已直接
  排除粒度因素）；R2 与 H28 两节结论互为对方的存在性证据。

## 6. 守恒与 hopbytes（步骤 4c）

- 逐请求 3100 守恒（`conservation_physical.csv`，9/9 全过）：Σ3100 + P-piece =
  kv(prefill_ctx) **逐字节精确**（如 C：12,494,831,616 + 92,362,768,384 =
  104,857,600,000 = kv(200k)）。
- 3300 精确式（3 溢出成员）：Σ3300 = decode_len × Σ_s R_m 逐字节精确（如 C：
  1,477,804,294,144 = 16 × kv(176,168)）；participation 合计 = decode_length
  （8+8=16，T_max=8 拆两列车）。ideal 臂按设计不发射 3300（Σ3300=0，非守恒破坏）。
- 1000 拉回：B1 Σ1000 = 12,394,168,320 B（源 d0，6 shard，3 hops）。
- hopbytes CLI：physical 16,206,123,761,664 hop-bytes（96 actions，coverage
  1.0，bytes_without_hops=0）；ideal_masked 2,824,750,497,792（60 actions）——
  差 13,381,373,263,872 = Σ3300 4,460,457,754,624 × 3 hops **精确相等**。

## 7. 口径说明

- **TTFT**：SPLIT=off（交付缺省）→ `first_token_ns` = train_interpolated 代理
  （离线复跑 `run_metrics_postprocess.sh` 从归档 train_ledger 填充；显示指标，
  不进 SLO 判定——仓内既定语义）。TPOT = (completion − TTFT)/decode_len。
- **3300 边完成时刻** = C++ `[METRIC] memory_anchor`（type=memory_anchor）中该
  (member,train,source_rank) send 节点完成 tick。锚匹配按成员 decode 窗口内源
  rank 锚的 tick 升序逐列车对号（同 (src,dst,tag) 跨列车 FIFO 保序）。**已知覆盖
  缺口**：末列车 6 头 rank 对的 send 锚在 3 成员 × 3 点共 9 例中 6 例未注册
  （golden R1 同样复现；非计时缺失，属锚注册的批内覆盖），这些行标
  `missing`，不借锚回退，防止误归。
- **3100 recv**（owner 侧 HBM 写界）与 **1000 send**（源侧 HBM 读界）同样有锚：
  分别出现在 decode_start/arrival 后 ~1.3 ms，量级与 shard 字节/1640 一致。
- **体尾**：无直接锚；用 body_gate（下列车发射 tick）作体完成代理，模型比较
  （§3）对其 ±决策滞后不敏感（两模型差 43 ms ≫ 滞后 ~1 ms 量级）。
- **H28 边时长**：首列车隔离边（send 就绪 = decode_start，无前驱流）为主口径；
  末列车边与首列车在同链路 FIFO 重叠，时长含排队/分享，不用作带宽推断。
- **确定性**：C 成员全链路与 golden R1 的 B 逐字节一致（同容量、同队列参数部
  分）；A/B/H28 各臂同队列同派生配置，差异仅来自 KV_REMOTE_READ 与 hardware
  json 变体。

## 8. 产物与复跑

- 数据表（本目录）：`ab_diff_per_request.csv`、`per_request_{physical,ideal_masked}.csv`、
  `conservation_physical.csv`、`read_edges_physical.csv`、`exit_body_drag_physical.csv`、
  `h28_three_point.csv`、`tmax_scan.csv`（R2 复裁批次，T_max∈{4,8,16} 扫描）。
- run 目录：`/tmp/wscllm_t1/runs/{physical,ideal_masked,h28_r1,h28_r4}`（原始
  cpp.log/决策日志/train ledger/journal/hopbytes 输入）；指标中间件
  `/tmp/wscllm_t1/metrics/`；队列构造器 `/tmp/wscllm_t1/make_overflow_queue.py`；
  单点 run 封装 `/tmp/wscllm_t1/run_one.sh`；提取器 `/tmp/wscllm_t1/metrics/extract_metrics.py`。
- R2 复裁批次：`/tmp/wscllm_r2re/runs/{tmax4,tmax8,tmax16}`（同队列同种子，仅改
  SH_TRAIN_MAX_ITER）+ `/tmp/wscllm_r2re/metrics/`（三档提取产物与汇总 JSON）。
- 复跑：`bash /tmp/wscllm_t1/run_one.sh <run_dir> <physical|ideal_masked> [variant_hw_json]`
  （H28 点传 `/tmp/wscllm_t1/h28_hw/face_case5_config_c_h28{r1,r4}.json`；runner
  新增 `SH_RUNTIME_CONFIG_DIR` env 覆盖，缺省行为与既有 runner 逐字节一致）；
  T_max 扫描点在调用前 `export SH_TRAIN_MAX_ITER=<4|8|16>`（调度器
  `wsc_llm_online_scheduler.py:413-414` 直读 env，缺省 8）。

## 9. 偏差与遗留

1. 队列规模克制（9 行 / 3 溢出成员 / n=3 分布）：p95 用 nearest-rank 小样本口
   径，R2/H28 结论应视作初裁/初值；扩样本只需在构造器加行（B0 式干净窗成员可
   水平复制）。（R2 复裁注：n=3 的 p95 口径本身不影响力证方向——判据比较的是
   同三成员跨 T_max 档的相对差，非绝对分位数。）
2. 末列车 6 头 send 锚覆盖缺口（§7）：影响个别边时长观测，不影响守恒/窗口/
   模型比较（借道 5 头锚 + max 模型）；根因在 C++ 锚注册的批内 last-kv 选择，
   属只读范围未动，建议后续在观测层补（非本批授权）。复裁批 T4/T8 同样复现
   （每成员末列车 6 头锚 missing，已在 tmax_scan.csv 以锚前观测值标注）。
3. filler t1s2–t1s5 的 TTFT 代理不可得（实例末列车，无下列车 tick）= 已知口径
   （代理定义如此），不影响溢出成员指标。
4. metrics_postprocess 独立复跑依赖 cpp.log init 指向的 generated plan 目录
   存在（run_dir 拷贝只供 slo 工具族）——H28 换点会清前一 plan 目录，本批以
   "换点前先取该点代理"的顺序规避（r1 的代理经重新物化 plan 后补齐）。
5. slo_postprocess 的 hbm_watermark 对 relevant 变体按 B4b 裁决跳过（WARN 行
   正常）；slo_stats warmup 在本小场景同 B4b 记录（场景规模伪影，非缺陷）。
6. （R2 复裁批）T_max=16 时 TTFT train_interpolated 代理退化（单列车无后继列车
   tick：t1s7_r0 NA / t1s6_r0 插值到无关列车 / t1s6_r1 clamp 到 completion）：
   显示指标伪影（SLO 判定路径拒绝 proxy），proxy 无关精确量（窗口/16）不受影响；
   若后续需要 T16 档的 TTFT 显示值，可开 SH_FIRST_TOKEN_SPLIT=1 取 exact（研究
   口径，B4 起缺省关）。
