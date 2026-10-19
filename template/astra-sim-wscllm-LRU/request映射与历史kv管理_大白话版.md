# request 怎么派活、历史 KV 怎么管：大白话版

分支编号沿用详细版的 M1–M68（request 映射）和 K1–K51（KV 管理），两条一条不漏，行号都对着现在的代码查过一遍。文里说的"直接报错"，都是指代码抛异常、这次运行停下来，不带猜测往下走。

## 一、这套系统分几层、怎么交接

这个模拟器分五层，一层把活干完交给下一层：

**1. 输入物化层**（`plan_materializer.py`、`generate_trace.py`）。请求排队 CSV 在这里读进来、逐条查格式，写成 `manifest.json`。每个请求记九个字段：request_id、session_id、第几轮、prefill 多少 token、decode 多少 token、历史多少 token、prefill 完多大、算完一共多大等。历史 token 数的算法在这里就定死：第一轮是 0，后面每轮等于同会话上一轮算完的总 token 数。"下一轮直接接着用上一轮的 KV、不重算"这件事，从输入文件就定了。

**2. 在线调度层**（`online/wsc_llm_online_scheduler.py`、`online/online_scheduler_base.py`）。Python 写的调度器，靠 `decision_bridge.py` 的文件队列跟 C++ 模拟器一来一回。每个 tick 收一份状态增量，按固定顺序干三件事：先处理完成信号，再处理新到的请求，最后跑一轮准入加发射。请求派给哪个 Prefill 实例、哪个 Decode 实例、显存够不够放，全是这一层拍板。它只看自己 Python 侧的账，不读 C++ 内部状态。

**3. KV 管理层**（`session_kv_manager.py`）。按会话管 KV。每个会话的 KV 只有两个状态：整份在某实例显存里，或整份在远端内存池里。显存不够就按 LRU 把整个会话搬到远端池，下轮要用再整份搬回来。每张卡记一笔账：权重 + 驻留 KV + 预约，加起来不许超过容量。

**4. 图发射层**（`online/graph_batch_builder.py`、`generate_wsc_llm_trace.py`）。把调度决定翻译成每张卡的节点图。节点带 rank 字段，请求最后算到哪张卡，这一步定死。KV 的搬移（逐出、恢复、Prefill→Decode 迁移）也在这里变成实际的传输节点。

**5. 模拟器执行层**（`astra-sim/workload/execution_driven/`）。C++ 侧。`GraphBatchCommitter.cc` 按节点的 rank 把节点登记到对应实例上（send 节点要求 `comm.src == node.rank`，recv 节点要求 `comm.dst == node.rank`，`GraphBatchCommitter.cc:1225-1245`），用物理时钟真跑，跑完把 PREFILL_DRAIN、DECODE_COMPLETION、REQUEST_COMPLETE 三类完成信号推回调度器。

一个请求的一生：进到达堆 → 挑一个 Prefill 实例排队 → 按开局定好的对应关系知道自己的 Decode 实例是谁 → 准入时先在 Decode 实例上把终态 KV 的位置预约下来、再把历史 KV 准备好 → 发射 prefill 整段 → prefill 完成后把 KV 从 Prefill 实例搬到 Decode 实例、进 decode 批 → 跑完 → 有下一轮就把 KV 留着等下一轮，没有就把会话整个清掉。

两种映射方式必须分开记：

- **Prefill 实例是来了现挑的**：请求到达那一刻，看所有 Prefill 实例谁队短就派谁，每个请求单独挑一次。
- **Decode 实例是开局一次钉死的**：系统启动时就把"每个 Prefill 实例配哪个 Decode 实例"算好了（挑最近的，还强制一跳直达、链路互不共用）。请求挑了 Prefill 实例，Decode 实例跟着就定了，运行期不再变。

## 二、request 映射：M1–M68 全分支

文件都在 `sh_test_mesh/workload/llama2_7b_inference/` 下，online 子目录的标了 `online/`。

### 2.1 输入变成请求清单（M1–M10）

| # | 什么时候走这条 | 走了之后发生什么 | 位置 |
|---|---|---|---|
| M1 | 某会话的第一轮（turn_index==0） | 历史记 0；prefill 后的大小直接等于 prefill_length | plan_materializer.py:321-334 |
| M2 | turn>0 的行 | 历史取同会话上一轮的 final_context_tokens；prefill 后 = 历史 + prefill_length；终态 = prefill 后 + decode_length | plan_materializer.py:335-348 |
| M3 | 同一会话出现第二条 turn-0 行 | 直接报错，这次不跑了（以前会悄悄覆盖游标） | plan_materializer.py:326-331 |
| M4 | turn>0 但队列里没有紧挨着的上一轮 | 直接报错，不悄悄把历史当 0 | plan_materializer.py:339-344 |
| M5 | manifest 里 request_id 重复 | 直接报错。下游全按 request_id 记账，重了分不开 | plan_materializer.py:379-384 |
| M6 | metrics 清单里 turn-0 的到达 | 记绝对时刻（会话到达时间） | plan_materializer.py:413-419 |
| M7 | metrics 清单里 turn>0 的到达 | 记"上一轮完成后隔 interval_ns 再到" | plan_materializer.py:420-429 |
| M8 | metrics 派生时 (session, turn) 键重复 | 直接报错 | plan_materializer.py:406-411 |
| M9 | 队列旁有 sidecar 文件且 request_id 对得上 | 把 request_type、prefix_len 抄进 manifest，把 human_time_ns、tool_time_ns 抄进 metrics_manifest；对不上就不写，不猜 | plan_materializer.py:366-373、451-458 |
| M10 | 装载 CSV 时逐条格式检查 | 会话块被拆开、turn 不是严格 +1、turn-0 缺到达时刻、turn-0 多写了 interval、turn>0 缺 interval、时间不是 1000ns 整数倍、文件带 BOM、行宽不是 8 列、字段里带逗号、request_id 为空、request_id 以 # 开头、id 里含 kv 或 first_token 保留字——每条不合格都直接报错（跟 C++ 读者同口径的 Python 侧镜像） | generate_trace.py:189-327 |

### 2.2 请求到调度器手上（M11–M15）

| # | 条件 | 发生什么 | 位置 |
|---|---|---|---|
| M11 | C++ 通知某请求到了 | 记进 in_flight，标两段活（prefill、decode）没干完；重复到达或没听过的 id 直接报错 | online/online_scheduler_base.py:954-972 |
| M12 | 请求完成且同会话还有下一轮 | 排一个"现在 + 间隔"的闹钟交给 C++，到点再投递下一轮；间隔为 0 也合法（物理到达会被压到下一 tick） | online/wsc_llm_online_scheduler.py:1268-1286 |
| M13 | 没有下一轮 | 不排闹钟，走终轮清理（见 M68） | online/wsc_llm_online_scheduler.py:1305-1317 |
| M14 | 到达堆堆顶时间到点（≤ 当前 tick） | 按 (tick, 优先级, 队列号, 序号) 稳定弹出，交给到达处理；堆里混进别的东西直接报错 | online/wsc_llm_online_scheduler.py:1345-1356 |
| M15 | 同一份交付重复送来（序号相同） | 深比较一致就把上次缓存的应答原样拷一份回去，什么都不改；跳号或旧序号直接报错 | online/online_scheduler_base.py:492-522 |

### 2.3 Prefill 实例怎么挑（M16–M19）

| # | 条件 | 发生什么 | 位置 |
|---|---|---|---|
| M16 | 每次请求到达 | 只给 prefill 角色的实例拍快照，记 (实例号, 队列长度)；decode 实例不进快照 | online/wsc_llm_online_scheduler.py:1936-1944 |
| M17 | 快照非空 | `select_prefill_instance` 取 (排队请求数, 实例号) 最小的那个：先比谁队短，打平取配置序靠前的，每次结果唯一 | wsc_llm_scheduler.py:567-575 |
| M18 | 一个 Prefill 实例都没有 | ValueError。构造期已保证两类角色齐备，这是防御 | wsc_llm_scheduler.py:573-574 |
| M19 | 选定之后 | 请求记下 prefill_instance_index，排进该实例的 FCFS 队尾，实例标成"有活可干" | online/wsc_llm_online_scheduler.py:1134-1150 |

### 2.4 Decode 实例怎么钉死（M20–M32）

`build_instances` 的一排构造期检查（wsc_llm_scheduler.py:151-282），哪条不过都 ValueError：

| # | 条件 | 位置 |
|---|---|---|
| M20 | 实例表为空 / 实例名空 / pg_name 空 | wsc_llm_scheduler.py:155-156、166-169 |
| M21 | pg_name 不是规范正整数（解析失败、≤0、"00" 这种异形写法）——防绕过查重 | wsc_llm_scheduler.py:174-183 |
| M22 | pg_name 重复 / phase_role 不是 prefill 或 decode / 实例没配 rank / rank 在实例内重复 | wsc_llm_scheduler.py:184-194 |
| M23 | 同一个 rank 配给了两个实例 | wsc_llm_scheduler.py:197-204 |
| M24 | 实例占的坐标不是实心矩形（有洞或斜角） | wsc_llm_scheduler.py:207-219 |
| M25 | 没正好覆盖每张卡各一次 / 实例大小不一 / 邻接图不连通 / 缺 Prefill 或 Decode 角色 / Decode 离晶圆中心最远的反而比 Prefill 最近的还远——逼着 Decode 摆在靠中心的位置 | wsc_llm_scheduler.py:234-276 |

`build_static_pd_mapping` 的路由选择（wsc_llm_scheduler.py:413-555）：

| # | 条件 | 发生什么 | 位置 |
|---|---|---|---|
| M26 | 一个 Prefill 实例都没有就来建映射 | 直接报错（以前会悄悄返回空映射，现在构造期拦） | wsc_llm_scheduler.py:427-429 |
| M27 | 给每个 Prefill 实例找最近 Decode | BFS 列出全部等长最短路当候选；等长路超过 64 条直接报错 | wsc_llm_scheduler.py:432-446、335-382 |
| M28 | 各实例候选组合乘起来超过 4096 | ValueError，防网格拓扑把枚举数量引爆 | wsc_llm_scheduler.py:448-457 |
| M29 | 全组合枚举完 | 按 (总跳数, 共用链路次数, 固定选择顺序) 依次取最小：先跳数最少，再共用最少，最后按固定顺序保证结果唯一 | wsc_llm_scheduler.py:459-495 |
| M30 | 选出的路由有一条不是一跳 | ValueError。布局承诺"每个 P 旁边就是它配的 D"，做不到就不让跑 | wsc_llm_scheduler.py:510-521 |
| M31 | 两条路由共用同一条物理边 | ValueError | wsc_llm_scheduler.py:522-530 |
| M32 | 查路由 `route_for_prefill(idx)` | 返回该实例钉死的 (Decode 实例, 路径)；查不到就 KeyError。请求的 decode_instance_index 就是从这里拿的 | wsc_llm_scheduler.py:406-410；online/wsc_llm_online_scheduler.py:1144 |

### 2.5 Prefill 准入（M33–M46，`_try_admit_prefill`，online/wsc_llm_online_scheduler.py:1407-1619）

| # | 条件 | 发生什么 | 位置 |
|---|---|---|---|
| M33 | 已经准入过 | 直接返回 True（幂等） | :1413-1414 |
| M34 | prefill 实例或静态路由丢了 | RuntimeError | :1415-1417 |
| M35 | 容量纪元没变（prefill、decode 两实例的纪元都和上次尝试一样） | 返回 False，这轮不重试。任何容量变化都会把纪元 +1，纪元不变重试也没意义 | :1420-1426；纪元递增在 :1946-1950 |
| M36 | 纪元变了 | 按 final_context_tokens 算出终态 KV 分片向量，调 `reserve_request_capacity` 在 **decode 实例**上发全量预约——先把终态的位置占住，再动数据 | :1427-1438 |
| M37 | 全量预约成功 | decode_capacity_reserved=True，接着拍历史位置快照（M41） | :1518-1530 |
| M38 | 全量预约失败，但本会话旧 KV 恰好整份在 decode 目标实例上 | 全量预约把自己还没搬走的旧 KV 也重复算了一遍。改按差额重试：逐分量取 max(0, 终态 − 已有)，只预约还缺的那部分；成功后把已有部分记成 credit，等 KV 搬走再补 | :1449-1474 |
| M39 | 差额预约还是失败 | 把两次尝试赶出来的受害实例全部唤醒纪元（丢掉第一次会把等重试的请求楔死）；已决定的逐出立刻发射（不等门）；返回 False | :1484-1506 |
| M40 | 全量失败且旧 KV 不在 decode 目标上（在别的实例或在远端池） | 有逐出就唤醒纪元；逐出立刻发射；返回 False | :1508-1517 |
| M41 | 预约成功后 | 给会话历史拍快照：没会话记 ABSENT，有会话记位置和驻留层数——发射侧对账要用 | :1519-1530 |
| M42 | 接着做历史准备 | 对 **prefill 实例**调 prepare_history（四个分支见 K1–K4）；它为腾地方赶出去的会话记到 history_evictions | :1531-1545 |
| M43 | 历史准备被卡（显存实在不够） | 把刚预约成功的容量退掉、credit 作废、相关实例纪元全唤醒、已决定的逐出立刻发射、返回 False。这是生产里真实会走的阻塞分支：请求先排队，等容量变了再试，不报错 | :1546-1567 |
| M44 | 走了差额预约（M38）且历史迁移成功 | 先核对迁移确实是从 decode 实例迁走的（不是就 RuntimeError），再调 `extend_request_capacity` 把预约扩回全量——旧 KV 从"驻留"换成"预约"，两边字节数相等，任何时刻都不超订 | :1573-1586 |
| M45 | 历史放行后 | grow_prefill 把会话 KV 长到 prefill_context_tokens；失败就 RuntimeError——预检都过了这里不该失败，真失败说明内部有 bug | :1595-1602 |
| M46 | 全部通过 | admitted_prefill=True；三段逐出（预约/历史/增长）全部记账并同步到发射侧；唤醒纪元 | :1603-1618 |

### 2.6 Decode 准入和迭代列车（M47–M55，`_try_admit_waiting_decodes` + `_plan_train`）

| # | 条件 | 发生什么 | 位置 |
|---|---|---|---|
| M47 | 每个 decode 实例的等待队列 | 非空且（刚标脏 或 该实例纪元变了）才这轮处理，避免白跑 | online/wsc_llm_online_scheduler.py:1626-1636 |
| M48 | 就绪队列逐条弹出 | 没预约过就 RuntimeError；然后调 `release_request_capacity` 把静态预约释放（审计标签记 decode_start） | :1644-1657 |
| M49 | 预约释放后 | move_prefill_to_decode 按 final_context_tokens 把会话 KV 从 prefill 实例搬到 decode 实例；赶出去的会话记到 decode 侧账上 | :1659-1674 |
| M50 | move 被卡 | RuntimeError。预约不变量保证这里必然放行，真走到说明结构坏了 | :1675-1682 |
| M51 | 迁移后 | grow_decode 把 KV 长到 final_context_tokens；失败同样是防御性 RuntimeError | :1684-1694 |
| M52 | 增长成功 | 进 active_decode 批，记进批时的队列深度，实例不忙就标就绪 | :1701-1711 |
| M53 | 准入 pass 碰到不忙的 decode 实例且有批成员 | 冻结一列车的成员：迭代数 = 成员里剩余 token 的最大值；SH_TRAIN_MAX_ITER（默认 8，0 不限）会截断 | :810-866、521-538 |
| M54 | 列车被截断且这趟没人自然跑完 | 造一个哨兵 id，借 PREFILL_DRAIN 单事件通道把完成信号送回来核销 | :849-853、1018-1028 |
| M55 | 判断实例忙不忙 | 按角色分开：Prefill 实例 = busy 标志（一段 prefill 在飞）；Decode 实例 = 有一列车在飞。忙的实例跳过这轮不发射，不报错 | :1958-1965、1381-1382 |

### 2.7 发射成图（M56–M62）

| # | 条件 | 发生什么 | 位置 |
|---|---|---|---|
| M56 | prefill 准入成功 | 整段 prefill 一次发射（chunk 之间不合并，权重每个 chunk 读一遍）；watch 挂 PREFILL_DRAIN；assignments 行带上 prefill/decode 两个实例号 | online/wsc_llm_online_scheduler.py:1725-1762 |
| M57 | trace_granularity 不是 request_aggregated | RuntimeError，只有这一种合法粒度 | online/graph_batch_builder.py:494-497、1088-1091 |
| M58 | decode 发射 `_emit_train` | 列车头先发新成员的 KV 迁移（3000 号配对传输）和 join 标记，接着 17 类聚合算子节点，尾部 exit 标记加共享 end barrier；每个退出成员的会话记下 barrier 位置，给下一轮的 interval gate 用 | online/graph_batch_builder.py:1044-1104、1183-1401 |
| M59 | SH_FIRST_TOKEN_SPLIT=1 且列车有首次成员且迭代 ≥ 2 | 拆两批发：第 1 迭代单独一批带唤醒标记，唤醒到达后再发余量批。默认关 | online/graph_batch_builder.py:79-100；online/wsc_llm_online_scheduler.py:997-1006 |
| M60 | 预约成功，随 prefill 批发逐出 | `emit_decode_eviction_actions`：触发门 = 该请求 prefill 段块末；逐出链从各 rank 当前链分叉出去、不汇合，和计算并行跑 | online/graph_batch_builder.py:638-707；online/wsc_llm_online_scheduler.py:1743 |
| M61 | 阻塞尝试时已放行的逐出 | 没有触发门（prefill 段还不存在），直接挂到各 rank 当前链尾立刻发射——决策到发射零延迟，不能跨纪元重试攒着 | online/wsc_llm_online_scheduler.py:1505、1516、1566；online/graph_batch_builder.py:653-656 |
| M62 | 每条 KV 传输要领 tag | 从 100_000_000 起单调发号，到 500_000_000（C++ NATIVE 段上限）就 OverflowError；_stage_tag 算出的 stage tag 越进这段同样报错 | generate_wsc_llm_trace.py:204-229、437-446 |

### 2.8 完成处理（M63–M68）

| # | 条件 | 发生什么 | 位置 |
|---|---|---|---|
| M63 | 完成信号四分类 | batch_train_ 前缀 → 哨兵或首步唤醒（唤醒不做事，真哨兵进列车核销）；stage=prefill → drained；decode → decode_done；request → request_done；别的 stage 直接报错 | online/wsc_llm_online_scheduler.py:573-605 |
| M64 | 列车核销 `_finalize_completed_trains` | 每个成员按本趟参与度加 token 数；跑完的成员移出批；第一个信号核销恰一次，后面的信号只清尾巴；信号找不到所属列车直接报错 | :645-735 |
| M65 | PREFILL_DRAIN | 先查它确实是 FCFS 队头（不是就 RuntimeError）；busy 清零、队头出队；把请求登记到 decode 实例的等待准入队列并标脏 | :1152-1178 |
| M66 | DECODE_COMPLETION | 调 mark_complete 写 LRU 时间戳（唯一写点，见 K26）；完成快照写决策日志；请求身上那些大字段立刻清掉省内存 | :1180-1250 |
| M67 | REQUEST_COMPLETE 且有下一轮 | 排 M12 的闹钟；调 `register_pending_history` 给下一轮登记 pending 门，位置优先用延迟通道里存的，没有就问管理器要当前快照 | :1293-1304；online/graph_batch_builder.py:529-547 |
| M68 | REQUEST_COMPLETE 且是终轮 | 构图器 `retire_completion_gate` 清掉该会话的完成门、延迟位置、store 尾巴登记；KV 管理器把会话整个退休（见 K27）；唤醒纪元 | :1305-1317；online/graph_batch_builder.py:515-527 |

## 三、历史 KV 管理：K1–K51 全分支

先说状态。会话 KV 只有两个状态：`local_hbm` = 全部 L 层都在一个实例的显存里；`remote_memory` = 一层都不在本地，整份在远端内存池。没有第三种。代码里留了个 `partial_hbm_remote` 常量，只用来拒绝旧日志，运行期不可能出现（session_kv_manager.py:52-64、1298-1302、3232-3236）。

### 3.1 新请求来了，历史 KV 怎么准备（K1–K6，`prepare_history`，session_kv_manager.py:2461-2783）

进来先过一排硬检查，不合格直接报错：新会话却带历史 token（:2488-2489）、会话记账的 token 数和请求说的对不上（:2490-2494）、会话上还有请求在跑（:2495-2496）、要求的上下文比历史还小（:2485-2486）、local 会话没记实例号（:2562-2563）、要跨实例迁却不知道源实例（:2627-2628）。

| # | 条件 | 发生什么 | 位置 |
|---|---|---|---|
| K1 | 没有这个会话（第一轮请求） | 先在 prefill 实例上确认终态 KV 放得下（不保护任何会话）；放不下 → 返回 blocked 的 NO_HISTORY 决策（带着已完成的逐出）；放得下 → 建一个零字节的新会话对象（local 态、全层、active），发 no_history 事件，一个字节都不搬 | :2513-2560 |
| K2 | 历史就在目标实例上 | 净需求 = 要的减已有的；腾地方时保护自己；放行就只把 active 置 True，发 local_hit 事件，零传输零搬动。P/D 分开的布局下这条生产不可达（历史 local 只会留在 decode 实例，而准入目标恒为 prefill 实例），留着给单测用 | :2583-2623 |
| K3 | 历史 local 但在别的实例 | 腾地方（保护自己）→ 放行就构造 XY 路由的分片传输；账先加目标再减源（传输契约：目标先有地方，源才放）；会话 instance_index 改成目标；发 noc_migrate 事件 | :2626-2695 |
| K4 | 历史 remote | 唯一的从远端回来的路，整会话全层恢复。远端账缺失就 RuntimeError；腾地方 → 放行就构造 remote_load 传输（层域 [0, L)）；目标加账、远端池减账；状态改成 local、驻留层数 = L、远端账清零；发 remote_restore 事件 | :2697-2783 |
| K5 | 位置既不是 local 也不是 remote（含 partial） | RuntimeError | :2698-2702 |
| K6 | 净需求出现负数（历史准备不许缩水） | RuntimeError | :2575-2576 |

顺带说指标口径（slo_tools/kv_cache_adapter.py:100-140）：history_location_before 是 local_hbm 或 remote_memory 都算 full 命中，partial 只可能出自旧日志；字段缺失就退回按 history_action 查表（NO_HISTORY→no_history，LOCAL_HIT/NOC_MIGRATE→full，RECOMPUTE→miss 是旧基线的事）；都不认识 → not_supported。no_history 不算 miss；命中率有两个分母（有历史的请求 / 全部请求）。

### 3.2 显存不够怎么腾（K7–K13，`ensure_physical_fit`，session_kv_manager.py:2114-2284）

| # | 条件 | 发生什么 | 位置 |
|---|---|---|---|
| K7 | 输入的分片向量含非整数、负数或长度不对 | ValueError，不接受 int() 静默截断（1.9 变 1 会少预约） | :2155-2158、191-205 |
| K8 | 需求大到空实例都装不下（扣掉权重后仍不够） | ValueError "cannot fit an empty instance"。这不是内存紧张，是配置矛盾，赶谁都救不了 | :2195-2205 |
| K9 | 当前不够，但把活跃会话、被保护会话、已预约的字节全扣掉后，剩余空间其实够 | 进入逐出循环。连这个条件都不满足就直接跳 K12——逐出注定没用，不白赶 | :2210-2215 |
| K10 | 循环里 | 按 LRU 顺序一个一个赶，每赶一个立刻重查够没够，够了就停（多赶出来的不收回）；候选赶光还不够就出循环 | :2222-2236 |
| K11 | 返回前复查全部受害者 | 挨个查：确实不活跃、不在保护名单、确实已落到远端。有一条不满足就 RuntimeError | :2243-2256 |
| K12 | 赶完了还不够（深缺口） | deep_gap_events 计数 +1，记一条受限的压力事件，返回 admitted=False。不抛异常——请求先排队等着，等容量变了（纪元 +1）再试，不会死锁 | :2257-2278 |
| K13 | 返回成功但实际还不够 | RuntimeError，内部记账出了 bug | :2281-2284 |

### 3.3 赶谁、怎么赶（K14–K19）

| # | 条件 | 发生什么 | 位置 |
|---|---|---|---|
| K14 | 谁能进候选 | 四个条件全满足：local 态、就在这个实例上、不活跃（当前没请求在用它）、完成过至少一次。活跃会话、已在远端的、没完成过的、被保护的，永远进不来 | session_kv_manager.py:948-962 |
| K15 | 怎么排序 | 按 (最后完成时刻, session_id) 升序，先赶最久没用的；session_id 只用来保证并列时结果唯一。最后完成时刻只在 mark_complete 那一刻写（非负、同一会话内不倒退），是 LRU 时间的唯一来源 | :934-946 |
| K16 | 赶的动作 `_evict_session` | 先复查三条：完成了且不活跃、local 且有实例号、层数正好 L。然后构造 remote_store 传输（层域 [0, L)，赶完本地驻留为 0）→ 释放本地全部分片 → 状态改成 remote（实例号清空、驻留 0、远端账加上全量）→ 远端池记账 → 增量不变量审计 → metrics 段删光 → 发 evict_session 事件。赶就是整个会话一起赶，没有拆半层的做法，旧的部分逐出代码已物理删除（模块头 :18-27） | :1757-1834 |
| K17 | 远端池按 rank 记账 | _remote_add/_remote_remove 每次先查长度和符号；减的时候账不够 → RuntimeError "remote-pool accounting underflow"。journal 行带 remote_delta_bytes 列 | :1079-1149 |
| K18 | 逐出传输发成什么物理动作 | 两种：源 rank 就在 mesh 边上 → 一个 mem_store（hbm_access_mode=1）搞定，读 HBM 的账只记这一次；不在边上 → 源 rank comm_send（发送端 HBM 读记账）→ 边缘 rank comm_recv（不记账）→ 边缘 mem_store（纯池写零记账）→ 回一个 1 字节 ack | generate_wsc_llm_trace.py:877-972 |
| K19 | 逐出什么时候发射、怎么记账 | 预约赶的人（#2）：准入成功随 prefill 批发射（门=段末）、被卡就当场发射；move/grow_decode 赶的人（#4/#5）按预约不变量运行时恒为零，保留随列车头发射的老路。决策日志里 decode_target_evictions 记全集，decode_evictions_pending 只记没发的；有一条断言保证 prefill 行和 decode 行之间这个账只增不减，防止两边加起来双计 | online/wsc_llm_online_scheduler.py:1439-1448、1666-1671、887-897；online/graph_batch_builder.py:1183-1237 |

### 3.4 从远端池捞回来（K20–K25）

| # | 条件 | 发生什么 | 位置 |
|---|---|---|---|
| K20 | remote_load 发成什么物理动作 | 1 字节请求（控制 rank 和边缘同卡就直接挂依赖，跨卡走 1B 点对点）→ 边缘 mem_load（只按池端口计时，不算本地 HBM）→ 边缘 comm_send（不记账）→ 目标 comm_recv（不记账）→ 目标 local_hbm_kv_restore（恢复唯一的 HBM 写记账，节点名带 kv 子串给 C++ 认） | generate_wsc_llm_trace.py:974-1061 |
| K21 | 每次历史准备段之后 | 放一个全实例的就绪 barrier（history_tp_ready_barrier）：恢复没完成前不许消费历史 KV。整体恢复就一段传输，用不着前后缀两道 barrier | online/graph_batch_builder.py:1016-1023 |
| K22 | 恢复发射前查 pending_store_tails（`_arm_store_tail_dependencies`） | 这个会话如果还有在飞的逐出（正在往池里写），必须先写完才能读：同边缘 rank 直接挂依赖边，跨边缘 rank 用 1B 点对点转一手；补完边把整条登记弹掉（只用一次） | online/graph_batch_builder.py:762-835 |
| K23 | 表是空的，但调度器探针说该会话还有"已决定没发射"的 remote_store | RuntimeError，时序坏了 | :789-800 |
| K24 | 没有传输的历史（NO_HISTORY/LOCAL_HIT） | 只挂到达门。turn>0 走到这条就 RuntimeError：P/D 分开时 turn>0 必然恰有一段迁移或恢复 | :965-994 |
| K25 | 历史传输数超过 1 | RuntimeError，两个状态下恢复就是一段 | :962-964 |

### 3.5 跑完之后：留着、退休（K26–K31）

| # | 条件 | 发生什么 | 位置 |
|---|---|---|---|
| K26 | 请求 decode 完成时 mark_complete | 只有活跃 local 会话能完成；完成时刻必须是非负整数且不比上次早（倒退就 ValueError，这是 LRU 顺序的时间基础）；然后 active=False、写下最后完成时刻、发 retain_complete 事件、清这个请求的压力事件去重键。完成时一个会话都不赶，恒返回空 | session_kv_manager.py:3026-3069 |
| K27 | 终轮退休 retire_terminal_session | 先把这个会话名下所有预约按"先全查再统一减"释放；local 就把本地分片整删、metrics 段删掉；remote 就把远端账静默划掉（不产传输，不算逐出）；最后把会话对象删了 | :3071-3208 |
| K28 | 退休前的检查，各条独立报错 | 会话不存在（KeyError）、还活跃或没完成过、request_id 不是它的最后一笔、预约却挂在别的会话名下 | :3090-3115 |
| K29 | 预约三件套 | 重复预约 RuntimeError；释放一笔不存在的预约 RuntimeError（带上下文）；释放先逐 rank 查够不够再统一减，失败重试还是同一笔；extend 只给差额回补用（M44），把被迁走的驻留字节一比一换成预约字节 | :2287-2455 |
| K30 | 跑完 assert_final_state | 不许剩预约、不许剩压力事件键、不许还有活跃会话 | :3210-3220 |
| K31 | KV 共不共享 | 不共享。每份 KV 逻辑上只属于一个会话（记账键 remote:{session} / resident:{session}:{segN}）；"共享"只发生在远端池的物理容量上，而且也是按 rank 各记各的 | :1109、1458 |

### 3.6 每张卡的显存账（K32–K36）

| # | 条件 | 发生什么 | 位置 |
|---|---|---|---|
| K32 | 每张卡的账怎么算 | used = 权重 + 驻留 KV + 预约，remaining = 容量 − used。每次改动后查：出现负数 RuntimeError、超容量 RuntimeError | session_kv_manager.py:447-462、1356-1366、3276-3285 |
| K33 | 审计分两档 | 常规只查这次改动碰到的 rank、会话、预约；SH_STRICT_KV_INVARIANTS 开了就每次全量查 | :92-100、1339-1404、772-776 |
| K34 | 事件流水压缩 compact_events | 已消费超过 8192 条就把前缀整段删掉，均摊每条 O(1)；event_index 由独立计数器发号，压缩不影响已发出去的事件 | :69、822-842 |
| K35 | 没装 journal 的旁路模式，跑完发现不活跃会话还占着显存 | 只往 stderr 打一条告警，不报错（旁路模式本来就没数值影响）；journal 开着时这条由跑完的总账把关 | online/wsc_llm_online_scheduler.py:2092-2139 |
| K36 | 离线看水位 | slo_tools/hbm_watermark.py 重放决策日志：逐出条目变成动作（字节优先取 shard_bytes 列、回退 total_bytes；受害实例号/时刻谎报直接报错），恢复列表逐段成动作，占用变化点合并成水位曲线 | slo_tools/hbm_watermark.py:240-413 |

### 3.7 delta journal：逐 rank 流水账（K37–K44）

| # | 条件 | 发生什么 | 位置 |
|---|---|---|---|
| K37 | recorder 没装或不是 journal 模式 | 装饰器直接透传：零快照、零回滚、零 journal 行，行为和没有这功能时逐字节一致 | session_kv_manager.py:159-161 |
| K38 | 嵌套的公开调用（事务已在跑） | 深度 +1 并进外层同一笔事务，不另开 id。一笔事务 = 调度器视角的一次修改 | :162-167 |
| K39 | 最外层入口 | 领一个单调 id，给管理器全部可变状态拍快照；方法正常返回就是提交点，紧接着逐 rank 对账：管理器实际的 (权重, 驻留, 预约, 远端) 对上 journal 按各行增减推出来的终值 | :168-186、1843-1881 |
| K40 | 失败（方法抛异常 / journal 写失败 / 对不上） | 管理器整体回滚到入口快照 + 放弃这笔事务，原来的异常照抛——对管理器是真原子。已写出去的 journal 行收不回来，靠这一轮报错终止兜底 | :172-186、1930-1980 |
| K41 | 每次记账的时机 | 每次加/减分片、远端增/减、预约、权重预载之后，逐 rank 写一行（schema v2，含 remote 列和前后快照）；时刻倒退 raise，分量变负 raise；构造期的权重预载是第一行（transaction_id=0） | metrics_integration.py:216-337；session_kv_manager.py:795-814 |
| K42 | 每行标属于哪个阶段 | history/prefill/admission 都标 prefill_start，decode 标 decode_start，其余 stage_boundary；释放时的标签由调用方指定（decode 准入边界传 decode_start） | session_kv_manager.py:111-127；online/wsc_llm_online_scheduler.py:1656-1657 |
| K43 | 跑完 verify_journal_checksum（journal 开着才跑） | 流式重放 journal：行数要一致、逐 rank 终值要一致、不认识的 rank 拒绝、总账必须对上（驻留=0、预约=0、物理=权重、远端=0——退休把远端账也划到 0）。产物 kv_delta_journal_checksum.json；任何不一致 RuntimeError | session_kv_manager.py:1982-2110 |
| K44 | 总开关 KV_DELTA_JOURNAL | 缺省开；off 就完全旁路，跟没有这功能一样 | online/online_service.py:69-79 |

### 3.8 事件流和压力事件（K45–K47）

| # | 条件 | 发生什么 | 位置 |
|---|---|---|---|
| K45 | 每个决策批末尾 | 把管理器新产出的事件（按游标增量读）塞进 GraphBatch.kv_actions 发给 C++；先放暂存区，C++ 确认提交后才算定稿；发完按水位压缩并回退游标 | online/wsc_llm_online_scheduler.py:632-641；online/online_scheduler_base.py:470-475、1092-1094 |
| K46 | 压力事件去重 | 同一个 (类型, 阶段, 请求, 目标实例) 第一次记 admission_blocked；之后再撞到只补一条 admission_retry。请求完成或会话退休时按键清掉 | session_kv_manager.py:1005-1058 |
| K47 | 深缺口计数 deep_gap_events | 每次尝试失败都 +1，不去重——同一请求重试再撞也照加。这个数是"阻塞事件数"不是"被阻塞请求数"，纯观测，不进任何决策；由 online_service 的 summary 行落盘 | session_kv_manager.py:2259-2266；online/online_service.py:331-332 |

### 3.9 位置信息怎么传给下一轮（K48–K51）

| # | 条件 | 发生什么 | 位置 |
|---|---|---|---|
| K48 | 每次改动返回逐出时 | `sync_pending_history_after_evictions` 只认 remote_store：立刻把这个会话 pending 门上的位置改成 remote_memory。决策时点同步是唯一改法，发射侧不再重复改 | online/graph_batch_builder.py:559-567 |
| K49 | 逐出传输里"赶后驻留层数"不是 0 | RuntimeError：整会话逐出必须把本地清空 | :569-573 |
| K50 | 会话暂时没有 pending 门 | 位置先写进 deferred_session_locations，等下次登记门的时候再消费。这是 turn-0 的防御通道，现在的两态管理器下生产不可达（turn-0 在飞会话恒活跃、没完成时刻，候选池双重排除），留着防御 | :574-593 |
| K51 | 下一轮 prefill 发射时对账 | pending 门上的位置必须等于准入时拍的快照 history_location_before，不一致就 RuntimeError——说明哪次逐出没同步上 | :930-938 |

## 四、映射和 KV 状态怎么互相影响

映射决策受 KV 状态管。挑 Prefill 实例、查静态路由这两步不看 KV；但紧接着的准入全看 KV 的账——decode 目标上的全量预约得过容量裁决；旧 KV 恰好整份在 decode 目标上，才允许只预约差额；prepare_history 走哪条分支，完全由会话 KV 现在在哪决定（没有过 / 在别的实例 / 在远端池）。容量纪元把 KV 的变化传回映射这一侧：任何逐出、迁移、恢复、退休都会把相关实例的纪元 +1，之前被拒的请求看到纪元变了才会重试——请求什么时候能进来，由 KV 状态什么时候变化决定。

反过来，每次准入成功都改三处 KV 状态：decode 实例的预约账（可能挤掉别的会话，被挤的整个会话按 LRU 搬去远端池）、prefill 实例的会话驻留（历史迁入加 prefill 增长）、差额预约在迁移完成后补成全量。P→D 交接那一刻预约释放、KV 落到 decode 实例，会话的实例号从 prefill 改成 decode。最后在 REQUEST_COMPLETE 分开处理：有下一轮，KV 留在原地或留在池里，位置信息经 pending 门交给下一轮的判断；没有下一轮，会话连同远端账一起清掉。跑完的 journal 总账（驻留=0、预约=0、远端=0）就是给这套一来一回收尾的核对。
