# WSC-LLM-LRU 策略说明：请求实例映射与 KV 冷热管理

> 本文从体系结构、算法—软件映射与仿真三个视角，给出 WSC-LLM-LRU 策略的通用说明。
> 全文与具体硬件配置、模型规格无关：拓扑规模、算力/带宽/容量、模型层数与维度等
> 均以参数符号表达，任何满足前提的晶圆级系统都可套用。

## 1. 策略定位

WSC-LLM-LRU 是 `astra-sim-wscllm` 的 **KV 冷热管理改造仓**，由两条承袭线构成：

- **映射保留 WSC-LLM 静态 P/D**：请求→实例映射与原仓**逐行一致、零改动**
  ——静态物理 P/D 分区（Decode 实例居中、Prefill 实例环绕）+ 离线一次性
  P→D 路由规划 + 在线 prefill 最小排队选点 + 实例内严格 FCFS（队头阻塞）+
  decode 连续批处理。映射决策不感知 KV 位置、长度或内存状态的本性不变；
- **冷热管理承袭 SH-2.0 的逐出内核与发射链路（去类型化）**：会话 KV 从
  "片上驻留 / 删除重算"两态升级为 **LOCAL / PARTIAL / REMOTE 三态**；空间
  不足时按**去类型化两段式 LRU**逐出（先扫全部冷会话的后半层、仍不足才整体
  外迁）；被逐出的 KV 落入片外统一内存池，下一轮到来时按态分流**恢复**
  （PARTIAL 恢复与前缀计算真流水重叠）——**恢复取代重算**，RECOMPUTE 路径
  已从历史决策中删除。

与 sh 系的关系：逐出算法本体承袭 sh_2.0 与 sh_3.0 逐行同码的共有内核，但
**去类型化**（删除 sh 的 human/tool 类型化循环与 `next_request_type` 机制）；
跨实例 PARTIAL 恢复链取 sh_2.0 的"前缀 NoC 迁移 + 后缀远端恢复"两段式；
发射链路（remote_store / remote_load 的边缘端口借道、HBM 计费、真流水编排）
照 sh_2.0 移植。**不承袭**的：sh 的"耗尽即 raise"失败语义（本仓维持 wscllm
队头阻塞）、sh 的类型化两阶段逐出、sh 的准入预占架构与全局纪元变量。

## 2. 系统模型与记号

### 2.1 体系结构前提

- **晶圆级芯片**：R 行 × C 列二维 Mesh，N = R·C 个 NPU，片上网络互联；
- **每 NPU**：峰值算力 F；本地 HBM 带宽 B_HBM、访问时延 λ_HBM、容量 C_HBM；
- **片间链路**：相邻 NPU 间带宽 B_NOC、逐跳时延 λ_NOC；
- **远端内存池（本策略启用）**：网格边界 NPU 为**边缘 NPU**，各挂一个远端
  端口（带宽 B_rem、时延 λ_rem），全体端口构成逻辑统一的 KV 冷层池
  （写入端口与读出端口可以不同）；远端池**无容量上限、无自身置换**。

### 2.2 实例划分

芯片被划分为 m 个**互不重叠、铺满整片、等大的轴对齐矩形实例**，每实例 t 个 NPU
（张量并行度 TP = t），共同承载一个模型副本。实例间共享物理边即邻接，构成实例级
网格；实例邻接图必须连通。本策略在实例之上再做**静态 P/D 角色分配**（见 3.1）。

### 2.3 模型与 KV 容量记号

设模型 L 层、隐藏维 d、注意力头数 H、元素字节 b。则：

- **每 token 的 KV 字节数**：B_KV = 2·L·d·b（K 与 V 两份）；长度 h 的上下文
  对应 KV 总量 B_KV(h) = 2·L·h·d·b；层区间 [l₀, l₁) 的 KV 与层数成正比；
- **逐 NPU 分片**：注意力头按**整头**划分到 t 个 NPU（H 不能整除时，前若干 NPU
  各多承担一个头），因此同一会话在各 NPU 上的 KV 分片字节可以不等——**容量检查
  必须逐 NPU 进行**，不能只看实例总量；
- **逐 NPU 账本**：本地剩余 = C_HBM − 模型权重分片 − 驻留 KV 分片（含层区间
  折算）− 已承诺请求的预留分片；远端账面单列（不占本地容量）。

### 2.4 负载记号

- 会话由多轮请求构成：首轮带绝对到达时刻；后续轮在**上一轮 Decode 完成 + 思考
  间隔**后到达。会话内串行、会话间并行；
- 每轮派生量：历史 token 数 h_hist = 上轮最终上下文；预填充上下文 h_pre =
  h_hist + 本轮输入长度；最终上下文 h_fin = h_pre + 生成长度；
- 预填充按 **p_chunk**（参数）分块推进。

## 3. 请求实例映射策略（保留自 WSC-LLM 原仓，零改动）

### 3.1 静态角色分配（离线一次性）

运行前完成两项决策，运行期间**永不改变**：

1. **P/D 角色分配**：n_D 个实例为 Decode、n_P 个实例为 Prefill（n_P + n_D = m）。
   布局遵循**向心约束**：所有 Decode 实例的中心到晶圆中心的距离，不大于任何
   Prefill 实例的中心距离——即"D 居中、P 环绕"，使每个 Prefill 实例的近邻中
   总有 Decode 实例。
2. **P→D 路由规划**：在实例邻接图上为每个 Prefill 实例确定**唯一** Decode 目的地，
   目标为字典序最小化：
   1. 全部路由的总跳数最小（每个 P 连到跳数最近的 D）；
   2. 其次最小化路由间共享链路的重复次数（避免流量热点）；
   3. 最后以确定性顺序破除平局，保证规划结果可复现。

### 3.2 在线映射（Prefill 选择）

新请求到达时，唯一的在线决策是**进入哪个 Prefill 实例**：

- **选择键**：（该实例当前排队的请求数， 实例编号），取字典序最小；
- **平局规则**：排队请求数相同选实例编号最小；
- 实例内部为**严格 FCFS** 队列，同一时刻只服务队首请求的一个 p_chunk 块；
- 队首请求若因 KV 容量不足无法准入，**阻塞整条队列**（队头阻塞），等待容量释放，
  不改投其他实例。

该指标计**请求数**而非 token 数、剩余服务时间或 Decode 压力——排一个长请求与排
一个短请求的实例在此指标下等价。映射**不看会话 KV 位置**：历史 KV 驻留在哪个
实例、处于三态中的哪一态，均不改变选点结果。

### 3.3 Decode 归属与批处理

- **Decode 归属**：由静态路由唯一确定——请求进入哪个 Prefill 实例，其 Decode
  实例在到达时刻即已确定，不随任何运行时状态改变；
- **连续批处理**：Decode 实例每轮迭代为**全部活跃请求**各生成 1 个 token；迭代
  时长按（批大小 = 活跃请求数，上下文 = 批内最长上下文）评估；新完成预填充的请求
  动态加入批，生成完毕的请求动态退出；
- **阶段交接**：Prefill 与 Decode 不同实例时，发生一次**整份 KV 的 P→D 迁移**
  （NoC 逐跳、确定性维度序最短路由、相对 TP 编号一一配对）。

## 4. KV 冷热管理策略（本仓改造核心）

### 4.1 三态模型

| 状态 | 含义 | 片上层数 |
| --- | --- | --- |
| 完整片上（LOCAL） | 全部 L 层驻留本实例 | L |
| 部分驻留（PARTIAL） | 前 ⌈L/2⌉ 层片上 + 后 ⌊L/2⌋ 层远端 | L − ⌊L/2⌋ |
| 完整远端（REMOTE） | 全部层在远端池，无实例归属 | 0 |

- 层划分按实际模型层数计算：`partial_resident_prefix_layers = L − L//2`
  （奇数层保留较大半），**不可配置**；
- 层区间分片逐 NPU 按整头规则计算，与全量严格守恒；字节与层数成正比，
  半层恰好减半（`kv_cache_shard_bytes_for_layer_range` 构造期守恒断言）。

### 4.2 去类型化两段式 LRU 逐出

**候选与排序**：候选池 = 本实例上"已完成（`last_completion_ns` 非空）且非活跃"
的驻留会话；排序键 **（最后完成时刻， 会话标识）** 升序 = 纯 LRU（同时刻平局按
会话标识字典序）；活跃会话永不被逐出，触发请求自身会话受保护。**不按
human/tool 类型分类**——无 `next_request_type` 字段、无类型化外层循环，请求
CSV 维持 8 列。

**两阶段**（每次逐出后立即逐 NPU 重查水位，够即停）：

- **阶段 1（逐后半层）**：按 LRU 序把完整片上会话的**后 ⌊L/2⌋ 层** remote_store
  至远端池 → 状态 LOCAL→PARTIAL；
- **阶段 2（整体外迁 = 唯一回退）**：阶段 1 把全部候选半层化后仍不足，才按同一
  LRU 序把驻留**前缀 [0, resident_prefix_layers) 整体** remote_store →
  状态→REMOTE、清实例归属。

**触发时机（四处收敛点，与原仓完全一致，不新增触发位置）**：prefill 准入
（`prepare_history`）、decode 准入 P→D 交接（`move_prefill_to_decode`）、
KV 容量增长（prefill/decode 增长路径）、decode 终态净额预占
（`reserve_request_capacity`，净额口径）——净额预占为 wscllm 生产链路
真实使用。净额回补（`extend_request_capacity`）仅增量记账、**不触发逐出
收敛**：resident→reserved 1:1 换位保证任何瞬间不超订（§4.5），容量已由
净额预占处的收敛 + 迁移删除旧驻留共同保证。

**失败语义**：两阶段耗尽仍不足 → 记 deep_gap（延迟事件计数）后**队头阻塞等待**，
容量纪元（capacity_epoch）唤醒后重试，绝不改投其他实例；不搬 sh 的"耗尽即 raise"。

**显式负面清单**（设计裁决，均有测试锁定）：

- **无类型化**：不引入 next_request_type / CSV next_trigger_type 列；
- **无消融档位**：不设 `no_tiered_eviction` 类开关，逐出行为唯一（两段式，
  整体外迁是唯一回退）；
- **无 R_kv 保留水位**：`kv_reserve_context_tokens` 维持 manifest-only，不驱动
  任何逐出；
- **完成路径零逐出**：请求完成不触发逐出；
- **无历史截断**：会话上下文单调增长，不做到达时截断回收；
- 终局会话 `retire_terminal_session`：层域减账 + 远端账面静默核销（不发传输，
  **不算逐出**）。

**守卫与审计**（随内核承袭 sh_2.0）：

- **D4-I1 受害者资格复核**：每笔逐出后重验受害者确已非活跃、非保护会话；
- **D4-I3 过度逐出守卫**：撤销最后一笔逐出必须使至少一个受影响 rank 回到
  不满足，否则该笔不逐（防过度逐出）；
- `SH_STRICT_KV_INVARIANTS=1` 全量审计：三态 location 白名单、层域断言、
  REMOTE 无实例无驻留层、逐 rank expected_resident + expected_remote 重算、
  预占一致性。

### 4.3 命中复用与恢复路径（六分支）

下一轮到达、映射照常选点后（映射不看 KV 位置），按会话态 × 实例关系分流：

| 终态 × 实例关系 | 动作 |
| --- | --- |
| 新会话 | 建片上记录，随预填充增长（NO_HISTORY） |
| LOCAL 同实例 | 零开销本地复用（LOCAL_HIT） |
| LOCAL 跨实例 | 整份 KV NoC 迁移至目标实例（NOC_MIGRATE，1000 类配对迁移路径原样保留） |
| PARTIAL 同实例 | **只回迁后缀层段**（REMOTE_LOAD）：首 chunk 拆前缀层段 / 后缀层段，后缀段以远端恢复的逐 NPU 完成门为依赖——**恢复流量与前缀计算真流水重叠**；拆分发射的字节与单次全层发射严格守恒 |
| PARTIAL 跨实例 | **两段链**（PARTIAL_MIGRATE）：驻留前缀 NoC 迁移 → 目标实例容量逐出 → 后缀层段远端恢复；两段承载在**同一条 prefill 决策记录**内（对账工具按"同请求同 kind 不重复"校验） |
| REMOTE | 全量回迁到选定实例（REMOTE_RESTORE），逐 target rank 经最近边缘端口 |

- **最近边缘 NPU**：对每个源/目标 NPU 分片取曼哈顿跳数最小的边缘端口，等距取
  编号最小；
- **路由**：NoC 段走确定性维度序（XY）最短路由，逐分片记录路径与跳数；
- **按需恢复（非预取）**：恢复仅在下一请求准入时触发；实例空闲时不主动拉回
  已外迁的 KV；
- **恢复取代重算**：重算段已从调度规划中删除（remaining_chunks 不再有
  `ceil(history/p_chunk)` 加项）；恢复前的容量检查本身仍可触发对其他会话的
  分层逐出（先收敛容量、再执行恢复）；
- 结构注记：PD 分离下会话经 P→D 交接驻留 Decode 实例、下一轮 prefill 恒选
  Prefill 实例，PARTIAL 恢复实际走跨实例两段链；同实例后缀回迁分支已实现并由
  专项单测覆盖，真实 trace 无需期待其出现。

### 4.4 逐出 / 回迁物理链路与 HBM 计费

远端端口为严格 FIFO 单事务队列，单次访问 `耗时 = λ_rem + bytes/B_rem`；端口间
完全并行。逐链路的发射序与计费（**每字节每链恰好在正确端点计费一次**，过路
hop 零计费、双计有守卫拒收）：

- **remote_store 链 A（源≠边缘）**：[可选 1B trigger] → 源 `comm_send(bytes)`
  （源端 HBM 读 = 唯一数据计费）→ 边缘 `comm_recv`（`hbm-charge=false` 过路
  零计费）→ 边缘 `mem_store`（池写，本地零计费，仅池端口 FIFO）→ 1B ack 双端；
  **源端收到 ack 后才物理释放空间**；
- **remote_store 链 B（源==边缘，直连）**：`mem_store(hbm_access_mode=1)` =
  池读端点唯一计费 + 池端口 FIFO 双异步 join（仅当逐出源 rank 本身在 mesh
  周界时发生）；
- **remote_load（回迁）**：[1B control/arm] → 边缘 `mem_load`（仅池 FIFO）→
  边缘 `comm_send`（`hbm-charge=false`）→ 目标 `comm_recv`（`hbm-charge=false`，
  目标写由 restore 承担）→ 目标 `local_hbm_kv_restore` = RESTORE DMA 唯一
  数据计费（每 rank hbm_dma 单槽：同一 rank 同时至多一个在飞 restore）；
- **noc_migrate**：两端为真端点，正常 HBM 读/写计费。

本地 HBM 为 N 用户流体模型：推理 COMP、KV restore DMA、NoC p2p 数据端点读/写、
池流量端点读/写**六类作业严格均分**同一份 B_HBM（任一作业完成立即事件驱动
重分配）；restore 与推理的带宽共享由 `hbm-kv-restore-bandwidth-sharing` 开关。
KV 传输 tag 由专用分配器自 100,000,000 起单调分配，与既有请求内 tag 段
（queue_index×10000 + {1000,1900,3000}）错开，上限 2³²−1。（H8 修正
2026-09-24：原基址 10,000,000 的错开只对 <1000 行队列成立，30s 窗口源
trace 实测 1177 行已段重叠——基址上移至 100,000,000，队列 ≤9999 行两段
保持不相交，`_stage_tag` 越界 fail-closed 守卫随基址同步补上。）

### 4.5 守恒与不变量

- **逐 NPU 账本自洽**：本地驻留分片 + 远端账面 = 会话 KV 总额（按层区间折算），
  逐 rank 守恒；容量判决口径只计本地（远端池字节不占本地容量）；
- **净额预占 + extend 回补（wscllm 特有，保留）**：prefill 准入在静态 decode
  目标预占终态 KV 时，若全量预约因本会话旧驻留重复计入而 deep-gap、且旧 KV
  仍驻留于该 decode 实例（LOCAL 全额 / PARTIAL 取片上前缀账面），按净额
  （终态−旧驻留账面）重试；迁移删除旧驻留后经 `extend_request_capacity`
  回补到全量——**resident→reserved 1:1 换位，任何瞬间不超订**；
- **容量纪元纪律**：凡产生真实容量变化的 mutation（含三态化新增的逐出与回迁
  落账点）才 bump `capacity_epoch`，多源唤醒**合并覆盖**（净额失败 / 非净额
  失败 / 准入受阻释放 / 成功准入 / decode 侧五条路径逐一接齐）；仅增预占的
  extend 与无 mutation 路径不 bump——丢一次中间 epoch 会楔死等待重试的准入；
- **KV delta journal（schema v2）**：管理器每次公开 mutation 构成一个事务
  （`_journal_transaction` 装饰器覆盖全部新 mutation），逐 rank delta 流式追加
  journal；v2 行含 `remote_delta_bytes` 列、before/after 快照含 remote 账面；
  run 末 checksum 门（fail-closed）断言终态守恒 **resident=0 ∧ reserved=0 ∧
  physical=weight ∧ remote=0**（retire 核销后远端池账面与会话账本同归零），
  五项检查 manager_state_match / physical_equals_weight / remote_account_zero /
  residual_reserved_zero / residual_resident_zero 全绿方可 PASS。

### 4.6 指标观测（契约方案 A）

- `repo_variant` 维持 `"astra-sim-wscllm"` 不变；`kv_hit_state` **五值域不扩**：
  REMOTE→`full`、PARTIAL→`partial`；full_local / full_remote 由 evidence 列
  区分（如 `history_location_before=remote_memory`）；命中率分子
  hit_n = full + partial；
- 后处理适配器**双级解析**：优先决策日志 `history_location_before` 三态映射，
  字段缺失回退旧 `history_action` 四值表（旧产物兼容）；逐出条目带
  `total_bytes` 标量回退（不静默丢）；水位工具接受 journal schema v1/v2
  （单文件不混版），v2 remote 链参与行级自洽校验且不计入本地容量判决，
  检测到逐段传输字段在场自动升级逐段对账重放。

## 5. 算法—体系结构映射分析

1. **向心 P/D 布局 ↔ 固定短路由**：Decode 居中、Prefill 环绕，使每个 P 的最近
   D 都在邻接位置，P→D 交接只需一跳链路；离线规划同时最小化总跳数与共享链路
   数。在线调度器完全不需要理解网络拓扑与拥塞——代价是映射对 KV 位置、队列
   压力与内存状态全盲。
2. **三态冷热 ↔ 容量压力的分流出口**：原两态基线把 KV 压力全部转化为计算压力
   （删除-重算，预填充块数随历史线性增长）；三态化后容量压力首先转化为**带宽
   压力**（半层外迁/整体外迁的 remote_store 与恢复的 remote_load，均经边缘
   端口与 NoC），仅当远端往返也换不回空间时才队头阻塞。冷数据的典型生命周期：
   片上增长 → 半层化（腾空间给热会话）→ 整体外迁 → 下一轮经边缘端口回迁。
3. **先半层后整体 ↔ 逐出的边际成本递增序**：阶段 1 每笔只搬后 ⌊L/2⌋ 层
   （约半份 KV）就把受害者留在 PARTIAL——下一轮只需恢复后缀（且可流水）；
   阶段 2 整体外迁是无可奈何的回退（恢复代价升为全量）。这使"腾出单位空间
   的恢复代价"近似单调，冷会话越冷、被搬得越彻底。
4. **队头阻塞 + 纪元唤醒 ↔ 静态映射的活性保障**：容量不足不改映射、不扩候选
   集（保静态 P/D 的可复现性）；活性由容量纪元承担——任何真实容量变化
   （含逐出、回迁、释放）都唤醒阻塞的队头重试，多源变化合并覆盖防丢唤醒。
5. **净额预占 ↔ PD 分离下的记账正确性**：P→D 静态路由把长会话钉死在唯一
   decode 实例，准入预占若按全量计（不扣本会话旧驻留）会出现"旧+新"双计
   深缺口并楔死重试；净额（终态−旧驻留账面）+ 迁移后 extend 回补在不超订
   前提下消除双计，三态化后账面按片上前缀（PARTIAL）精确折算。
6. **决策可复现性**：排序键、平局规则、层切分、边缘端口选择、路由、tag 分配
   全部确定性；`SH_STRICT_KV_INVARIANTS` 与 journal checksum 提供逐 mutation
   的账本级复算验证。

## 6. 仿真视角

- **双层架构**：Python 策略进程 + C++ 离散事件引擎在线闭环，文件桥协议、单在途
  交付背压；决策边界为到达 / 预填充完成 / 解码完成 / 请求完成，完成时解码完成与
  请求完成成对触发。C++ 引擎对 KV 语义 opaque，只提供 MEM 节点（mem_store /
  mem_load / local_hbm_kv_restore）的物理计时与 HBM N-way 带宽争用计费。
- **决策边界与仓内路由**：到达批按 prefill 最小排队选点并入队，同时经静态 P→D
  路由钉死 decode 目标；预填充完成复位 P 实例 busy 并登记 decode 准入（容量
  纪元门控下重查，固定静态映射不 remap）；请求完成落 KV 完成账本并排下一轮
  到达闹钟。
- **发射编排**（每 prefill 批内序）：到达门 → 历史逐出（挂到达门触发）→ 历史
  恢复分流（PARTIAL 走前缀迁移/后缀恢复真流水）→ prefill 增长逐出 → 就绪
  屏障 → prefill 主体（partial 时首 chunk 层段拆分）。decode 侧逐出挂列车头
  joiner 段（post-barrier 触发门）；D 侧 decode 为实例迭代列车（跨成员共享体
  节点 + joiner P→D 迁移 + 共享 end barrier），P 侧 prefill 整段聚合。逐出动
  作批始终伴随宿主请求事件同批发射（桥协议要求，防死锁）。
- **物理模型**：Roofline 计算 + 逐 NPU HBM 流体均分（六类作业）；远端端口
  FIFO（λ_rem + bytes/B_rem）；拥塞感知 NoC + 维度序路由；远端写回在源端收到
  ack 后才物理释放；每链路每字节恰一次端点计费（`hbm-charge=false` 过路豁免、
  `hbm_access_mode=1` 直连池读、restore DMA 目标端唯一写计费）。
- **可观测指标**：决策日志逐请求带 `history_location_before`（三态）、
  `history_resident_prefix_layers`、`history_transfers`（逐段传输对象：kind/
  reason/session/bytes/源目实例/层段/边缘端口路由）与逐出条目（victim/bytes/
  层域/stage）；journal 逐 rank delta（含 remote 列）+ run 末 checksum 五项
  守恒；`cache_events.csv`（remote_store/remote_load/noc_migrate 次数与字节）、
  `kv_hit_states.csv`（五值域 + evidence 区分 full_local/full_remote）、逐
  NPU HBM 水位与逐类 served bytes、边缘端口 FIFO 记账、逐请求时延与吞吐。

## 7. 策略小结

| 维度 | 做法 |
| --- | --- |
| P/D 组织 | 物理静态分区：n_P 个 Prefill 实例环绕、n_D 个 Decode 实例居中（保留原仓） |
| P→D 路由 | 离线一次性规划（总跳数 → 共享链路 → 确定性破平），运行期固定（保留原仓） |
| 在线映射 | Prefill 按（排队请求数， 实例编号）均衡；Decode 由路由唯一确定（保留原仓，不看 KV 位置） |
| KV 层次 | LOCAL / PARTIAL（前 ⌈L/2⌉ 层片上）/ REMOTE，三级（承袭 sh_2.0，去类型化） |
| 层划分 | 前 ⌈L/2⌉ 层驻留，后 ⌊L/2⌋ 层首迁，不可配置 |
| 淘汰 | 去类型化两段式 LRU：（完成时刻， 会话标识）升序；先全部后半层、不足才整体外迁；逐 NPU 收敛、逐笔重查、活跃保护 |
| 触发点 | 四处收敛点（prefill 准入 / decode 准入 / 容量增长 / decode 终态净额预占），不新增位置；净额回补 extend 仅增量记账、不触发逐出收敛 |
| 恢复 | 按需（非预取）六分支：PARTIAL 同实例后缀回迁与前缀计算真流水；PARTIAL 跨实例前缀迁移+后缀恢复两段链；REMOTE 全量回迁经最近边缘端口；RECOMPUTE 已删除 |
| 失败语义 | 队头阻塞 + 容量纪元唤醒重试（不改映射；不搬 sh 的 raise） |
| 预占 | decode 目的地净额预占（终态−旧驻留账面）+ 迁移后 extend 回补，resident→reserved 1:1 不超订 |
| 水位 | 无 R_kv（kv_reserve_context_tokens 仅 manifest 记录）；完成路径零逐出；无历史截断 |
| 守恒 | 逐 NPU 本地+远端账本守恒；journal schema v2 remote 列 + run 末 checksum 五项终态守恒 |
| 历史感知 | 无（映射零改动）——KV 位置只影响恢复动作，不影响选点 |

一句话概括：**保留 WSC-LLM 的静态 P/D 映射与队头阻塞骨架，把"删除-重算"的
容量出口整体替换为承袭 sh_2.0（去类型化）的三态冷热管理：两段式 LRU 先扫后半层、
不足才整体外迁，恢复按态分流并经边缘端口真流水回迁——容量压力从计算侧转移到
带宽侧，而映射决策保持零历史感知。**
