# request 映射与历史 KV 管理（大白话版）

Python 侧代码都在 `sh_test_mesh/workload/llama2_7b_inference/` 下面，文中只写文件名。

## 0. 这套系统怎么转

这个模拟器分五层，一层把活交给下一层：

1. **准备输入**。`generate_trace.py` 读请求队列 CSV（哪个会话、第几轮、什么时候到、两轮隔多久），`plan_materializer.py` 把队列变成 `manifest.json`，顺手算好每个请求的三个数：带多少历史 token（history）、prefill 算完上下文多长（context）、decode 算完多长（final）。C++ 侧按同样的行号给每个请求编 `queue_index`，两边对得上。
2. **调度决策**（Python，`wsc_llm_scheduler.py` + `online/`）。请求到点后先挑一个 Prefill 实例排队；prefill 算完，再送到一个 Decode 实例去逐个出 token。两种挑法完全不同：**Prefill 实例是请求来了现挑的**——看哪个 Prefill 实例队最短就去哪个；**Decode 实例是搭好布局时就一对一钉死的**——每个 Prefill 实例对应哪个 Decode 实例，开局建一张表，整场不变。
3. **KV 管理**（`session_kv_manager.py`）。一份全局账：每个会话的 KV 现在在哪个实例、占多少字节，每张卡上还剩多少地方。搬家、赶人、记账、对账全归它。
4. **构图和桥接**（`online/graph_batch_builder.py` + `online/decision_bridge.py`）。把调度决定翻译成一串计算节点（GraphBatch），经 FIFO 文件发给 C++。
5. **C++ 执行**（`astra-sim/workload/execution_driven/`）。节点按自己的 rank 字段塞进对应 NPU 的执行流里跑；哪个节点跑完了，就把"XX 完成"推回 Python，Python 再决定下一步。

一句话：Python 决定"去哪、搬什么 KV"，C++ 负责"真实什么时候跑完"。所有调度判断只看 Python 自己的账本（队长、KV 账、容量计数），C++ 只把完成时刻报回来。观测工具（kv_cache_adapter、hbm_watermark）只重放日志，不影响决策。

---

## 1. request 映射：请求去哪个实例

### 1.1 搭布局时的十二道检查（`build_instances`，wsc_llm_scheduler.py:149-266）

先交代一句：一个实例 = 六张卡（六个 rank）的通信域，整场只干 Prefill 或只干 Decode，中途不换。搭拓扑时下面任何一条不满足，直接 ValueError，程序起不来：

| # | 什么情况不行 | 行号 |
|---|---|---|
| T1 | 一个实例都没配 | 153-154 |
| T2 | 实例名字是空串 | 164-165 |
| T3 | pg_name 是空、是 "0"，或者跟别的实例重了 | 166-169 |
| T4 | 角色既不是 prefill 也不是 decode | 171-174 |
| T5 | 没配 rank，或 rank 列表里有重复 | 175-178 |
| T6 | 同一张卡被两个实例都要 | 181-185 |
| T7 | 实例的卡在网格上拼不出实心长方形 | 191-203 |
| T8 | 所有实例加起来没正好盖住全部 NPU（多一张少一张都不行） | 218-219 |
| T9 | 实例大小不一（KV 分片要求等大） | 220-223 |
| T10 | 实例之间的相邻关系连不成一片 | 239-240 |
| T11 | 没有 Prefill 实例或没有 Decode 实例 | 242-249 |
| T12 | Decode 不贴中心：最远的 Decode 离晶圆中心比最近的 Prefill 还远 | 250-260 |

全过才返回拓扑，同时把每个实例离晶圆中心的曼哈顿距离算好存着（204-215），后面建路由表用。

### 1.2 建静态 P→D 路由表（`build_static_pd_mapping`，386-529）

布局定了就建这张表，整场运行不变。给每个 Prefill 实例找一个最近的 Decode 实例和一条路：

| # | 什么情况 | 结果 | 行号 |
|---|---|---|---|
| R1 | 一个 Prefill 实例都没有 | ValueError（以前会静默返回空表，现在开局就报错） | 398-400 |
| R2 | BFS 找不到路 | ValueError | 328-329 |
| R3 | 两点之间等长最短路超过 64 条（`MAX_EQUIVALENT_SHORTEST_PATHS`） | ValueError，怕组合数爆掉 | 32, 345-351 |
| R4 | 各 Prefill 候选组合乘起来超过 4096 | ValueError，同样怕爆 | 33, 419-428 |
| R5 | 候选怎么定 | 距离等于最近距离的所有（Decode 实例，路径）对都算候选，按序排好 | 404-417 |
| R6 | 最终怎么挑 | 所有组合全枚举一遍，按（总跳数， 共享边次数， 签名）从小到大取第一名——结果唯一确定 | 448-455 |
| R7 | 挑出的路由有一条不是一跳 | ValueError：向心布局要求每个 Prefill 旁边就是它配对的 Decode，交接只许一跳 | 484-495 |
| R8 | 两条路由用了同一条物理边 | ValueError：交接边不许共享 | 496-504 |

运行时查表用 `route_for_prefill`（379-383）：拿 Prefill 实例编号查，查不到就 KeyError。运行期从不重新映射。

### 1.3 请求到达：现挑一个 Prefill 实例（`_on_arrival`，online/wsc_llm_online_scheduler.py:991-1013）

挑法很简单（`select_prefill_instance`，wsc_llm_scheduler.py:546-549）：看所有 Prefill 实例谁队里请求最少；打平就挑实例编号最小的。快照只统计 prefill 角色的实例（`_prefill_snapshots`，1614-1622），而且是在把本请求入队**之前**取的——这样排序键反映的是挑选那一刻的队长。

挑完连做五件事：记到达时刻 → 查静态路由 → 把 prefill 实例号、路由、**decode 实例号**（就是路由表那头的 Decode 实例，此刻定死，后面不改）写进 runtime → 请求进这个实例的先进先出队 → 实例登记为"有活可干"。

到达顺序由小顶堆决定（`_push_arrival`，1161-1171）：堆键是（tick, 优先级, queue_index, 序号），同一 tick 到期的按队列原始顺序稳定弹出。`_drain_arrival_heap`（1173-1184）只弹 tick 不超过当前的项；堆里弹出不是 arrival 的东西就 RuntimeError。

同一会话下一轮怎么接上：请求完成时（`_on_request_complete`，1105-1157）查 next_request 表。有下一轮，就往 C++ 注册一个未来闹钟，到点时刻 = 本次完成时刻 + 这轮配置的间隔（间隔查 `_interval_ns`，1667-1677）。没有下一轮（最后一轮），就回收完成门、把会话 KV 终局清掉（见 2.9）。

### 1.4 Prefill 准入：十三步（`_try_admit_prefill`，1236-1400）

请求排到队首要过这一关，一步一步说：

| # | 什么时候走这步 | 走了之后发生什么 | 行号 |
|---|---|---|---|
| A1 | 之前已经准入过 | 直接返回 True | 1241-1242 |
| A2 | prefill 实例号或路由丢了 | RuntimeError：内部状态坏了 | 1243-1245 |
| A3 | 上次尝试之后，prefill 实例和 decode 实例的容量计数（capacity_epoch）都没变过 | 返回 False，这轮不重试——什么都没变，重试结果也一样 | 1248-1253 |
| A4 | epoch 变了（有容量动静了） | 先给 decode 目标实例**按 final_context_tokens 全量预约** KV——终态 KV 要落在 decode 实例上，先把坑占住防别人抢。预约过程要赶人，赶人记录先攒在 decode_target_evictions 里 | 1255-1268 |
| A5 | 全量预约没成，但这个会话的旧 KV 恰好就在 decode 目标实例上 | 改成"还缺多少约多少"重试：预约 max(0, 全量 − 旧 KV 字节)。成功了把旧 KV 的字节数记成 credit | 1269-1306 |
| A6 | 缺额重试还是没成 | 把两次尝试赶过人的实例都记容量变化（叫醒等着的请求），返回 False 等下次 | 1294-1305 |
| A7 | 全量没成，旧 KV 也不在 decode 目标上 | 有赶人就记容量变化，返回 False 等下次 | 1307-1314 |
| A8 | 预约成了 | 给历史 KV 拍快照：没有会话记 "ABSENT"，有就记状态（RESIDENT/EVICTED） | 1315-1319 |
| A9 | 无条件 | 调 `prepare_history` 处理历史 KV，结果就三种（见 2.2） | 1320-1329 |
| A10 | 历史处理说"放不下" | 把刚预约的退掉、credit 作废、相关实例（P/D/历史来源/两次赶人涉及的）全记容量变化，返回 False | 1330-1345 |
| A11 | 有 credit，且历史 KV 的来源实例确实是 decode 实例 | 历史搬家已经把 decode 上的旧 KV 挪走了，**立刻把预约补回全量**（`extend_request_capacity`）——从准入到 P→D 搬家完成这段时间 decode 侧要留足全量空间。来源对不上就 RuntimeError | 1346-1364 |
| A12 | 历史 KV 就位后 | 调 `grow_prefill` 把 KV 长到 prefill_context_tokens。这时候失败直接 RuntimeError——预检刚过了还失败，只能是内部不变量坏了 | 1372-1379 |
| A13 | 全过 | admitted_prefill 置 True，记容量变化，返回 True | 1380-1400 |

capacity_epoch 是每个实例一个计数器（`_note_capacity_change`，1624-1628）：实例上发生任何容量变化就 +1。堵着的请求靠它知道"该重试了"，这是唯一的叫醒信号。

### 1.5 Decode 准入：七步（`_try_admit_waiting_decodes`，1402-1484）

prefill 物理跑完后，请求进这一步。前提都铺垫好了：decode 目标的空间在 Prefill 准入时（A4/A11）已经预约下并补成全量，所以这里没有"失败等下次"，出问题一律 RuntimeError：

| # | 什么时候 | 干什么 | 行号 |
|---|---|---|---|
| D1 | 某实例的等待队列非空且打了 dirty 标 | 按实例编号从小到大处理，摘掉 dirty | 1416-1424 |
| D2 | 队里成员的 decode 目标不是本实例 | RuntimeError：队列坏了 | 1429-1430 |
| D3 | 这请求居然没预约过 decode 容量 | RuntimeError | 1431-1433 |
| D4 | 无条件 | 先把预约放掉（journal 行记到 decode_start 名下） | 1436-1439 |
| D5 | 无条件 | 调 `move_prefill_to_decode` 把 KV 从 prefill 实例搬到 decode 实例（见 2.6）。放不下直接 RuntimeError——预约刚放掉却放不下，说明内部坏了 | 1440-1452 |
| D6 | 搬完 | `grow_decode` 长到 final_context_tokens；失败 RuntimeError | 1454-1464 |
| D7 | 全过 | 记下入队前深度，塞进 active_decode 名单；实例不忙就登记"有活干"；账本更新；记容量变化 | 1465-1484 |

### 1.6 每个决策批次的顺序（`_admit_pass`，1188-1234）

固定次序：先跑上面的 Decode 等待准入，再按实例编号从小到大逐个实例服务。

| # | 分支 | 行号 |
|---|---|---|
| E1 | 就绪名单里有个实例、busy 判断却说它在忙 | RuntimeError：两边对不上只能是内部坏了 | 1207-1211 |
| E2 | Prefill 实例 | 队里混了 decode 活 → RuntimeError；队空跳过；队首准入失败跳过这轮；成功就发射整个 prefill 段，实例置忙 | 1212-1223 |
| E3 | Decode 实例 | 队里混了 prefill 活 → RuntimeError；active_decode 空 → 跳过；有活就冻结当前成员，发射一趟"迭代列车" | 1224-1234 |

"忙"的定义分角色（`_instance_busy`，1636-1643）：P 实例 = 一个 prefill 整段在飞；D 实例 = 一趟列车在飞。

### 1.7 列车怎么组（`_plan_train`，704-761）

| # | 分支 | 行号 |
|---|---|---|
| E4 | 有成员剩余 token ≤ 0 | RuntimeError | 716-720 |
| E5 | 一个成员都没有 | 返回 None，不发射 | 723-724 |
| E6 | 迭代数怎么定 | 全体成员剩余 token 的最大值——这趟车跑到所有人都干完 | 725 |
| E7 | 上限 `SH_TRAIN_MAX_ITER`（默认 8，0 = 不限）非零且迭代数超了 | 截到上限 | 410-427, 726-729 |
| E8 | 成员的参与数 ≥ 它的剩余数 | 这趟车跑完它就该退出了，给它挂 exit 标记；车长不变 | 736-743 |
| E9 | 车被截断（E7）且没人自然退出（E8） | 加一个以车名命名的哨兵标记——拿车名当"虚拟成员"盯着整趟车跑完 | 744-748 |

发射（`_emit_train`，795-895）：计划是 None 直接返回；先把新成员的账记完（决策行、发射登记、赶人记录序列化后清空）；拆分车走 `emit_train_first_step` 后提前返回；正常车给每个要退出的成员挂 watch，截断车另挂哨兵 watch；挂上 in_flight_train、移出就绪名单。

核销（`_finalize_completed_trains`，530-621）：一趟车的第一个信号到达就核销——每个成员的消费数加上实际参与数，退出的移出 active_decode，清 in_flight_train；同车后面的信号幂等清理；来的信号不属于任何在飞/已核销的车 → RuntimeError。车核销后实例还有成员就回就绪名单，没有就移出（603-606）。

首步唤醒（`_consume_first_step_wakeup`，623-630）：`first_step` 信号是个无操作回声，认出来直接吞掉，不触发任何核销。

### 1.8 首 token 拆分（`_first_token_plan`，656-702；开关 `SH_FIRST_TOKEN_SPLIT` 默认关，graph_batch_builder.py:75-96）

| # | 分支 | 行号 |
|---|---|---|
| E10 | 开关没开 | 返回 None，行为和没这功能时一模一样 | graph_batch_builder.py:75-96 |
| E11 | 新上车成员已经消费过 decode token | RuntimeError：新成员必须零消费 | 674-677 |
| E12 | 没有新成员 | 返回 None | 678-680 |
| E13 | 新成员按 decode_length 分 | 等于 1 的把 exit 标记改名带上 first_token 字样（同一个节点两个锚）；大于 1 的单独挂 first_token 标记 | 681-687 |
| E14 | 迭代 ≥ 2 | 物理拆成首步批 + 余量批；否则不拆车，只加标记 | 688-702 |

### 1.9 构图发射层的十个分支（graph_batch_builder.py）

| # | 分支 | 干什么 | 行号 |
|---|---|---|---|
| G1 | 粒度不是 request_aggregated | RuntimeError：在线发射只支持按请求聚合 | 357-360, 526-529, 567-570, 606-609 |
| G2 | turn 0 | 给每个 prefill rank 挂一个到点门（timer gate）；到达时刻缺失 → RuntimeError；时长为 0 不发节点（等待由 C++ 闹钟干） | 389-401 |
| G3 | turn > 0 | 从账上弹出上一轮完成车的收尾节点（弹出即删，是这轮间隔门的唯一正常消费者）；账上没有或这轮没配间隔 → RuntimeError；间隔为 0 直接拿上一节点当依赖，不发节点 | 402-420 |
| G4 | 历史动作 = NOC_MIGRATE | 源缺失 → RuntimeError；源的实例组跟上一轮完成车的组不是同一个（正常碰不到，留着防将来映射语义变了）→ 先发一个控制触发再清 timer；然后逐 shard 发 1000 类收发节点把历史 KV 搬到 prefill 组 | 421-449 |
| G5 | 其他历史动作 | 只发一个 1900 类控制触发节点（1 字节依赖，保证先后顺序） | 450-456 |
| G6 | 历史动作 = RECOMPUTE | 在 prefill 组上先发一段 history_recompute 计算节点——历史从头算 | 457-463 |
| G7 | 当前 prefill 段 | 按 `prefill_chunk_size`（默认 512）切块聚合发射；0 token 返回空界 | 464-477 |
| G8 | 列车头：每个新成员 | 从 prefill 实例把整段上下文 KV 搬到 decode 实例（3000 类节点，字节按 prefill_context_tokens 算）；本来就在同组就没节点 | 621-644 |
| G9 | 列车尾 | exit 标记、哨兵标记、共享 end barrier；把收尾节点记到 completion_gates 账上，给下一轮的间隔门用 | 707-786 |
| G10 | 会话最后一轮完成 | `retire_completion_gate` 把账上的门清掉 | 377-379 |

`_paired_transfer`（generate_wsc_llm_trace.py:471-538）里面还有几道：两边 TP 度不同 → ValueError；同组 → 不发节点只挂 timer；总字节能按注意力头整分就按头分，不能就精确到字节切，总量绝不多算（494-505）；每个 shard 配 XY 路由（先 X 后 Y），编号越界 ValueError。

权重读几遍的算法（generate_trace.py:778-819）：prefill 默认每个 chunk 读一遍权重，chunk 之间不合并；decode 列车传迭代数，每个迭代读一遍；配了超过 chunk 数的遍数 → ValueError。首 token 拆分时首步批读 1 遍、余量批读 iterations−1 遍（graph_batch_builder.py:576, 614-616）。

发射 prefill 整段时的决策行（`_emit_prefill`，1488-1588）：迁移字节非零就补 noc_hops 字段（实例图最短路的跳数）；行里的迭代数 = prefill 块数 + 重算块数，重算块只有历史动作是 RECOMPUTE 时才折算，否则 0（1566-1570）。

### 1.10 三个完成边界（wsc_llm_online_scheduler.py）

| # | 边界 | 干什么 | 行号 |
|---|---|---|---|
| K1 | PREFILL_DRAIN：prefill 段物理跑完（`_on_prefill_drain`） | 队首不是本请求 → RuntimeError；实例置闲、队首出队；队里还有人就回就绪名单；decode 目标丢了 → RuntimeError；把请求登到 decode 实例的等待名单并打 dirty（入队和打 dirty 永远成对） | 1014-1040 |
| K2 | DECODE_COMPLETION：exit 标记的 watch 触发（`_on_decode_complete`） | 调 `mark_complete`（KV 保留，见 2.7）；decode 实例 epoch +1；快照没了 → RuntimeError；写完成决策行（completion_evictions 恒空列表是冻结字段）；把这请求的 KV 相关大字段全清空——该落盘的都落盘了，留着白占内存 | 1042-1103 |
| K3 | REQUEST_COMPLETE：同 tick 跟在 K2 后面（`_on_request_complete`） | decode 账对不上（消费数 ≠ 应产出、没上过车）→ RuntimeError；有下一轮 → 注册未来闹钟；最后一轮 → 回收完成门 + 终局清 KV + 发布容量变化；runtime、索引、next 链全部释放 | 1105-1157 |

### 1.11 Python 和 C++ 的交付协议（online_scheduler_base.py / decision_bridge.py）

两边按序号一问一答，序号乱一点就报错：

| # | 分支 | 干什么 | 行号 |
|---|---|---|---|
| P1 | 序号等于上次已应用的 | 深比对内容：一致就返回缓存的那份图，一个字不重算、状态零变更；不一致 → ValueError | online_scheduler_base.py:473-496 |
| P2 | 序号不是上次 + 1 | ValueError：跳号和旧号重发都算违例 | 497-501 |
| P3 | 到达的请求不在"没见过"名单、或还在飞 | ValueError | 882-894 |
| P4 | 完成的请求查无此人 | ValueError；列车哨兵（`batch_train_` 前缀）转给列车核销处理 | 846-880 |
| P5 | 在途未确认批次超过 `SH_PROPAGATING_TAIL_LIMIT`（默认 8），或到达数超过预期总数 | RuntimeError：真实在飞的工作不许截断 | 56-57, 60-130, 312-332 |
| P6 | commit ack 五种错：格式不对 / batch_id 不对 / success=false / 序号超前 / 没有对应记录 | 各自 ValueError；重复 ack（水位内或乱序洞里）幂等吞掉 | 965-1038 |
| P7 | 本批有 KV 动作 | 深拷贝进暂存区，ack 回来才正式入账；跑完暂存区还有货 → 报错 | 453-458, 1099-1107 |
| P8 | 桥上序号重复或来自未来 | 重复的幂等忽略；未来的报错回写 + 非 0 退出 | decision_bridge.py:106-130, 318-331, 383-396 |
| P9 | 处理 ack 出异常 | 错误写进下一个必读槽位（写进已消费的槽位对面永远读不到） | 442-486, 519-535 |
| P10 | 处理 request 出异常 | 错误写当前槽位 + 非 0 退出；管道断了（C++ 已死）单独记一笔再退 | 408-440, 536-550 |

### 1.12 配置开关和值域检查

| # | 开关 | 值和缺省 | 位置 |
|---|---|---|---|
| C1 | `kv_cache_policy` | 只许 `session_lru_recompute`，别的直接报错 | generate_wsc_llm_trace.py:190-198; online/online_service.py:226-260; online/wsc_llm_online_scheduler.py:292-296 |
| C2 | `kv_eviction_mode` | 残留键直接拒绝（主动驱逐机制已物理删除，不许静默忽略） | test_wsc_llm_scheduler.py:351-368 |
| C3 | `trace_granularity` | token_expanded 不行，在线只收 request_aggregated | generate_wsc_llm_trace.py:199-205; graph_batch_builder.py:357-360 |
| C4 | `request_queue_session_limit` | 非负整数，0 = 不过滤；limit 超过可用会话数报错；过滤后用 queue_index_mappings 保持两端编号一致 | generate_wsc_llm_trace.py:186-187, 233-288 |
| C5 | `remote_operand_loads` | 必须为 false（本仓没有远端权重池） | generate_wsc_llm_trace.py:172-185 |
| C6 | `prefill_chunk_size` | 默认 512 | generate_wsc_llm_trace.py:188-189 |
| C7 | `kv_reserve_context_tokens` | 遗留观测键，决策管线不读它，只有 hbm_watermark 读 CSV 算水位用 | generate_wsc_llm_trace.py:74-81; hbm_watermark.py:680 |
| C8 | `SH_TRAIN_MAX_ITER` | 默认 8；0 = 不限；空串或负数开局就报错 | online/wsc_llm_online_scheduler.py:410-427 |
| C9 | `SH_FIRST_TOKEN_SPLIT` | 默认 "0" 关；"1" 开拆分 | graph_batch_builder.py:75-96 |
| C10 | `KV_DELTA_JOURNAL` | 默认开；关了就完全不装记账器 | online/online_service.py:70-80 |
| C11 | `SH_STRICT_KV_INVARIANTS` | 默认关；开了每次改动后跑完整自检 | session_kv_manager.py:47-55 |
| C12 | `SH_GRAPH_DIGESTS` / `SH_PROFILE_JSONL` | 默认 "0" 关 | online/online_service.py:83-89, 204-206 |
| C13 | `SH_PROPAGATING_TAIL_LIMIT` | 默认 8（桥背压窗口的 8 倍） | online/online_scheduler_base.py:56-57 |

请求队列装载侧还有一堆身份检查（generate_trace.py:181-383），各自 ValueError：request_id 空 / 重复 / 带保留字 `first_token` / 带保留前缀 `batch_train_` / turn-0 缺到达时刻却带了间隔 / turn>0 缺间隔却带了到达时刻 / 时刻不是整微秒 / (session_id, turn_index) 重复。

### 1.13 C++ 执行层怎么派发（astra-sim/workload/execution_driven/）

| # | 分支 | 干什么 | 位置 |
|---|---|---|---|
| X1 | 入口已经关了还来命令 | 返回 false，生产方重试（有界队列的背压） | RequestIngress.cc:35-46 |
| X2 | 命令队列满 | 溢出计数 +1，拒绝 | RequestIngress.cc:40-47 |
| X3 | 收到 Submit 以外的命令 | 入口置关闭，之后的命令全拒 | RequestIngress.cc:48-56 |
| X4 | Submit 时 request_id 空、或进程内重复 | std::abort：防止下游统计按 id 记账时 last-wins | RequestIngress.cc:91-110 |
| X5 | 到达时刻不晚于当前时刻 | 闹钟钳到当前 +1；钳位按来源分桶计数 | RequestIngress.cc:114-149 |
| X6 | 来源是 StaticCsv 的到达 | 记录 queue_index → 到达/闹钟，跑完审计用 | RequestIngress.cc:150-157 |
| X7 | 命令是 CloseInput / EndOfFile / Error 三种 | 前两种分别关输入；Error 直接 std::abort，不许静默降级 | RequestIngress.cc:177-201 |
| X8 | Python 发来未来闹钟 | 校验（id 空 / 重复 / 批内重复 / 已排过 / queue_index 没注册或对不上）任一违例抛 runtime_error；queue_index 一次性消费；钳位计数 | RequestIngress.cc:288-352 |
| X9 | 闹钟到点 | 弹出排程集合，DecisionEvent(ARRIVAL) 进邮箱，带着 ingress_seq 和冻结的 queue_index | RequestIngress.cc:373-412 |
| X10 | 生命周期状态迁移 | IDLE↔ACTIVE↔DRAINING→FINISHED；重复到达/完成回调把计数减破零 → abort；全排空且输入还开着 → 回 IDLE；FINISHED 的评估推迟到输入关 linearize 之后 | ServiceCoordinator.cc:21-109, 252-281 |
| X11 | Python 的 response 到达 | 先 validate 后 commit 两段（校验失败直接 abort，没有补偿路径）；节点按 rank 字段塞进对应 NPU 的执行流 | GraphBatchCommitter.hh:113-345 |
| X12 | 被观测节点跑到终态 | 把完成事实推回决策层（按 stage 分 prefill/decode/request）；同一次触发推 DECODE_COMPLETION 和 REQUEST_COMPLETE 两条 | WatchRegistry.hh:145-263 |

### 1.14 输入清单层：历史 token 数在源头就定死（plan_materializer.py）

历史 KV 的数量在清单层就冻结了，后面所有层只消费：

| # | 分支 | 干什么 | 行号 |
|---|---|---|---|
| Q1 | turn 0 | history = 0；context = prefill_length（turn-0 的 prefill 已折进前缀，一种算法算到底） | 353-358 |
| Q2 | turn > 0 | history = 上一轮的 final（按（会话, 轮数−1）精确找父轮，递归加记忆化）；context = history + prefill_length | 359-368, 329-345 |
| Q3 | 上一轮的请求不在队列里 | RuntimeError：宁可报错，不静默当 0 | 360-365, 336-339 |
| Q4 | final 怎么算 | final = context + decode_length，无条件 | 369 |
| Q5 | manifest 里 request_id 重复 | RuntimeError：下游全按 id 记账，没法消歧 | 400-405 |
| Q6 | 旁车行/列在场 | request_type、prefix_len 逐字段透传；缺席就不写，不猜 | 387-394, 471-478 |
| Q7 | 有 provenance 文件且 csv 的 sha256 对不上 | RuntimeError：混着旧版本产物的队列不要；没有 provenance 的合成队列容忍 | 256-296 |
| Q8 | metrics 到达怎么派生 | turn-0 → 绝对时刻（缺失 RuntimeError）；turn>0 → 父轮完成 + 间隔（父轮或间隔缺失 RuntimeError） | 434-451 |
| Q9 | 目录发布 | 先写暂存目录再两步换名；中途崩了最坏就是匹配不到目录，不会出半成品；发布后清掉旧目录，保证恰好一个 | 128-188 |

---

## 2. 历史 KV 管理

### 2.1 状态和三本账

每个会话的 KV 只有两个状态：

- **RESIDENT**：在一个实例上，占着字节，记着每个 rank 的分片大小；
- **EVICTED**：字节删了，但"这个会话一共多少 token"这个数留着——给重算用。

每个 rank 记三本账：模型权重（开局预装）、已放的 KV（resident）、预约占坑（reserved）。三本加起来不许超过容量，超了直接 RuntimeError（session_kv_manager.py:264-270, 2064-2117）。

### 2.2 历史命中就三种情况（`prepare_history`，1540-1741）

请求进 Prefill 准入时，按（会话 id, prefill 实例号）查历史。先过四道检查：

| # | 检查 | 违反了怎么办 | 行号 |
|---|---|---|---|
| H1 | 会话没登记过，请求却说带历史 | ValueError | 1558-1560 |
| H2 | 会话记的上下文长度和请求报的历史长度对不上 | ValueError | 1561-1565 |
| H3 | 会话还活跃着又来一个请求 | RuntimeError：同一会话不许并发 | 1566-1567 |
| H4 | 无条件做容量预检 | `ensure_physical_fit`（prefill 实例，按 required_context_tokens 全量，本会话列入受保护） | 1575-1585 |

然后三种情况——**历史命中有且只有这三种**：

| # | 什么情况 | 干什么 | 行号 |
|---|---|---|---|
| H5 | 没有历史（会话头一回） | 在本实例新建一个零字节会话，置活跃，事件记 no_history；没有搬运 | 1607-1639 |
| H6 | 历史被删过（EVICTED） | 在本实例把历史 KV 的字节重新放回来（纯记账），状态改 RESIDENT、实例指过来、置活跃。决策 = RECOMPUTE，重算 token = **全部**历史 token，没有"只重算一半" | 1641-1683 |
| H7 | 历史还在别的实例上（RESIDENT） | 造一份逐 shard 搬运清单（从哪个 rank 搬到哪个 rank、多少字节）。账上**先在目标加、后在源减**，然后会话的所属实例改成 prefill 实例。决策 = NOC_MIGRATE | 1685-1741 |
| H8 | H4 的预检没过 | 返回带 admission_blocked=True 的决策（动作标签照三种口径标好，但没有搬运、不动任何状态）；上层收到就把预约退了等下次 | 1587-1604 |

以前还有第四种"历史就在本实例"的情况（LOCAL_HIT），2026-09-26 删了：这个布局下空闲会话的 KV 只会留在 Decode 实例，而这里的目标恒为 Prefill 实例，永远碰不上（1569-1574 的注释）。所以动作就剩三个：NO_HISTORY、NOC_MIGRATE、RECOMPUTE（kv_cache_adapter.py:31-34 也按这三个登记）。

### 2.3 放不下就腾地方（`ensure_physical_fit`，1268-1366）

所有"放不下就赶人"都走这一个函数，它是**唯一的赶人入口**：

| # | 分支 | 干什么 | 行号 |
|---|---|---|---|
| F1 | 分片向量长度不等于 TP 度、或有负数 | ValueError | 1280-1283 |
| F2 | 把实例倒空（只留权重）都放不下 | 不报错：深缺口计数 +1，发一条"空实例都放不下"的压力事件，返回没过。请求**原地等**——策略定的是任一张卡不够就整体不可行，等，不改绑别的实例 | 1284-1319 |
| F3 | 空间本来就够 | 直接过，一个不赶 | 1321-1328 |
| F4 | 差一点 | 进赶人循环：按排好的候选名单逐个删，每删一个重新算够不够，够就停 | 1330-1360 |
| F5 | 能赶的都赶了还是不够 | 深缺口计数 +1，发压力事件，返回没过（等容量变化再试，不会死锁） | 1331-1348 |
| F6 | 循环退出来了还差 | RuntimeError：多赶了或账错了 | 1361-1365 |

**赶谁？**（`_candidate_sessions`，594-610）五个条件全满足才能被赶：状态是 RESIDENT、就在这个实例上、不活跃（没有在算的请求）、完成过至少一次、不在受保护名单里（正在准入的会话不能被自己赶）。排序按（最后完成时刻, session_id）：**最久没完成的先被赶**，同一时刻按 session_id 字典序。挑到够为止。

### 2.4 预约三件套

| # | 操作 | 分支 | 行号 |
|---|---|---|---|
| F7 | `reserve_request_capacity`（1368-1423） | 同一请求预约两次 → RuntimeError（1382-1383）；内部先嵌套跑 ensure_physical_fit（本会话受保护），没过原样返回；过了逐 rank 加 reserved 账并登记 | |
| F8 | `release_request_capacity`（1425-1475） | 按登记的数值减账；账减成负数 → RuntimeError（1453-1454）。journal 行的阶段标记按调用场景分三种：Prefill 准入阻塞回滚 / Decode 放行先释放 / 终局兜底 | |
| F9 | `extend_request_capacity`（1477-1535） | 查无此预约 → RuntimeError；向量非法 → ValueError。只干一件事：把缺额预约补回全量（旧驻留字节转成预约字节，一比一，任何瞬间不超订） | |

### 2.5 KV 增长（`_grow` 家族，1743-1929）

| # | 分支 | 干什么 | 行号 |
|---|---|---|---|
| W1 | 不是 RESIDENT / 不在目标实例 / 不活跃 | RuntimeError | 1755-1756 |
| W2 | 要长到的数比现在小 | ValueError：KV 上下文只能长不能缩 | 1758-1760 |
| W3 | 无条件对**增量**跑 ensure_physical_fit | 可能赶人；没过返回 False（上层 P 侧、D 侧都是直接 raise） | 1761-1772 |
| W4 | 预检过了 | 加账、更新分片和 token 数，journal 逐段记账 | 1773-1787 |

`grow_prefill`（1789-1808）长到 prefill_context_tokens；`grow_decode`（1910-1929）长到 final_context_tokens。

### 2.6 Prefill → Decode 搬家（`move_prefill_to_decode`，1810-1908）

| # | 分支 | 干什么 | 行号 |
|---|---|---|---|
| M1 | 不是 RESIDENT / 没有实例 / 不活跃 | RuntimeError：搬家前 KV 必须活着且在 prefill 实例上 | 1821-1822 |
| M2 | 没传 final_context_tokens | 按当前上下文算字节数；传了就按 final 切分片 | 1829-1831 |
| M3 | 无条件对 decode 目标跑 ensure_physical_fit | 本会话受保护 | 1832-1841 |
| M4 | 没过 | 返回 admission_blocked=True 的决策，**不造搬运清单**；唯一调用方（D5）马上 RuntimeError | 1842-1854 |
| M5 | 过了 | 造逐 shard 清单，账上**目标先加、源后减**，所属实例改成 decode 实例，journal 记 prefill_decode_migrate，返回完整搬运信息 | 1857-1908 |

同实例搬运的分支也删了（source == target 恒不成立，1824-1828 注释）。所有搬家——历史搬家（H7）和这里（M5）——都是同一个规矩：**目标先把字节放好，源再删**。

### 2.7 完成：KV 留着不删（`mark_complete`，1931-1967）

只有一种行为，没有分支：active 置 False、记完成时刻、发 retain_complete 事件、**一个字节都不动**。会话 KV 留在 Decode 实例上，等下一轮来用（下一轮走 H7 搬去新挑的 prefill 实例，或者被别人赶掉走 H6）。同时把这个请求的压力去重键作废——完成的请求永远不会再重试准入。删 KV 只有一条路：准入时腾地方（2.3 的 F4/F5），test_wsc_llm_scheduler.py:370-373 钉死了这个说法。

调度侧 `_on_decode_complete`（1042-1103）在这之上：decode 实例 epoch +1、把完成后的 KV 状态写进完成决策行、然后清掉这请求的所有 KV 大字段。

### 2.8 被赶走（`_delete`，1210-1266）

| # | 分支 | 干什么 | 行号 |
|---|---|---|---|
| V1 | 不是 RESIDENT / 没有实例 / 还活跃 / 从没完成过 | RuntimeError：在算的和没算完过的绝不能赶 | 1219-1225 |
| V2 | 资格过了 | 减账；状态改 EVICTED、实例指针清空（token 数留着给重算）；journal 记 evict_delete:原因；造 EvictionRecord（受害者是谁、哪个实例、何时完成、多少字节、删前删后余量） | 1226-1266 |

压力事件去重（`_pressure_event`，653-696）：同一种堵（事件类型 + 阶段 + 请求 + 目标实例）只记第一条，重试只补一条 retry 记录；键按请求建索引，请求完成时 O(键数) 全部作废（698-706）。

赶人记录归谁：decode 决策行带 decode_target_evictions **全集**（预约、搬家、增长赶的人全算，发射后清空核销）；prefill 行只带发射时点的子集。下游工具只读 decode 行的全集，不许两行相加（wsc_llm_online_scheduler.py:843-859；kv_cache_adapter.py:480-540；hbm_watermark.py:476-493）。

### 2.9 会话终了清账（`retire_terminal_session`，1969-2050）

会话最后一轮、输出全部交完才调：

| # | 分支 | 干什么 | 行号 |
|---|---|---|---|
| N1 | 没这个会话 | KeyError | 1984-1986 |
| N2 | 还活跃或没完成过 | RuntimeError | 1987-1990 |
| N3 | 传的请求不是最后持有者、或预约挂在别的会话名下 | RuntimeError | 1991-2009 |
| N4 | 这个会话名下还有预约 | 全部释放（journal 行记到 completion 名下，并进同一事务） | 2014-2025 |
| N5 | RESIDENT | 减账，journal 记 terminal_session_retire，返回腾出的实例号 | 2028-2039 |
| N6 | EVICTED | 字节早没了，只删记录 | 2040-2041 |
| N7 | 别的状态 | RuntimeError | 2040-2041 |
| N8 | 收尾 | 删 metrics 容器、压力键、会话条目，跑不变量检查 | 2043-2050 |

腾出的容量立刻发容量变化（调度侧 1148 行）。跑完全场还有一道 `assert_final_state`（2052-2062）：有预约没退、有压力键残留、有活跃会话，任何一条在就 RuntimeError。

### 2.10 HBM 流水账（journal，session_kv_manager.py + metrics_integration.py）

journal 是权威账：KV 每次字节变动都记一行（哪个 rank、变多少、因为谁）。

| # | 分支 | 干什么 | 行号 |
|---|---|---|---|
| J1 | 没装记账器或没开 journal | 装饰器直接透传，零行为差 | session_kv_manager.py:106-110 |
| J2 | 已经在事务里 | 深度 +1，并进外层同一事务，不另起事务号（终局回收里嵌着释放、准入里嵌着容量检查，都靠这个并进去） | 111-116 |
| J3 | 最外层改动正常结束 | 提交事务，**马上逐 rank 对账**：manager 实际的三个数必须和 journal 累加出来的相等，不等 RuntimeError | 117-127, 1057-1090 |
| J4 | 中途抛异常 | 放弃事务、不做对账（免得二次异常把原始错误盖了）；已写的行留着，这轮运行反正是要停的 | 119-124 |
| J5 | 时间戳往回退 | 记账器直接报错，不偷偷修正 | metrics_integration.py:126-127, 254-263 |
| J6 | 嵌套 begin / 没事务却 finish | RuntimeError | metrics_integration.py:206-236 |
| J7 | 跑完对总账 | 三件事：重放 journal 的终态必须和 manager 快照一致；写出的行数必须等于重放行数；**跑完检查一遍账——所有人的 KV 都该清干净了，卡上只剩下模型权重**（resident=0、reserved=0、used=weight）。产物落 kv_delta_journal_checksum.json | session_kv_manager.py:1092-1207, 1719-1725 |
| J8 | 每行的阶段标记 | 按 phase 映射：history/prefill/准入 → prefill_start；decode → decode_start；完成 → completion；其他 → stage_boundary；权重预装那行记 tick_zero | 66-84, 499-518 |

### 2.11 水位报告分几档可信度（hbm_watermark.py）

**容量三种算法**（`compute_capacity_calibers`，709-797）：

| # | 名字 | 是什么 | 行号 |
|---|---|---|---|
| B1 | per_rank_total_hbm | 正式判决口径：逐 rank 算 weight+resident+reserved ≤ capacity，判决在 journal 重放里做 | 792-797 |
| B2 | resident_kv_hard_limit | 假设没人预约，resident 最多能到多少；容量连最大权重分片都装不下 → ValueError（拒绝输出负数） | 734-744, 771-782 |
| B3 | watermark_reserve_target | 按 kv_reserve_context_tokens（默认 1M）逐 rank 扣出来的目标数——**只报告不判罚**，保留水位的机制没做 | 745-749, 783-791 |

**决策日志重放**（WatermarkReplay / WatermarkScan）：按文件顺序一遍过，每条记录按固定次序处理——先逐出、再恢复/搬家、再增长（1669-1695，和 manager 真实语义一致：先腾后放）；算出负占用直接 fail（1388-1391）；超容量只计数，算诊断不算正式判决（1396-1403）。恢复动作分五种（apply_restore，1441-1538）：同实例 → 不搬；0 字节 → 只改所属；没写源 → 只加目标；源 = 目标 → 同实例恢复；其余 → 源减目标加。有一种对不上的情况单独处理：日志说是 RECOMPUTE（或状态本来就是 EVICTED），但重放的账上还有这个会话的字节 → 说明中间有没落盘的逐出，在这扣掉、计入 silent_evictions_reconciled（真实逐出时刻在上次可见事件和本行之间，占用按上界算，1472-1483）。decode 搬家零字节交割但源还有账 → fail（1555-1559）。

**journal 重放的信任三档**（replay_journal，1744-2052）：行内自洽检查（前行的 after = 这行的 before、delta 对得上、序号和时间单调、rank 的实例和容量不许漂移）任一破坏 → 硬失败 exit 2。有 checksum 且四项检查（manager 终态一致 / 只剩权重 / 预约清零 / 驻留清零）全过 → certified 档，这档里超容量算正式判决（exit 3）；checksum 不齐 → 降到 lifecycle_replay_exact 档；没有 checksum → resident 档（只敢说上界）。

（另外两件小事：manager 内存里的事件列表消费游标过 8192 且过半就整段删前缀，编号不变——538-549；深缺口计数 `deep_gap_events` 进 summary 行观测，正常负载预期恒 0——479-483、online_service.py:311-314。）

### 2.12 观测层怎么翻译命中（kv_cache_adapter.py）

| # | 输入 | 输出 | 行号 |
|---|---|---|---|
| O1 | NO_HISTORY | no_history（不算 miss） | 137-150 |
| O2 | LOCAL_HIT（旧产物才有） | full | 142-147 |
| O3 | NOC_MIGRATE | full（算命中，搬运成本另计入 hopbytes） | 142-147 |
| O4 | RECOMPUTE | miss | 142-147 |
| O5 | 未知动作 | not_supported（不猜） | 148-149 |
| O6 | manifest 里的请求没有 prefill 决策行 | not_supported + 证据列 | 701-709 |
| O7 | canonical 事件提取（迁移字节、逐出列表） | 只映 native 字段；--reconcile 以 native 为准逐项对账，不一致非 0 退出 | 480-540, 776-964 |
| O8 | Σshards ≠ total_bytes | 计违规，跑完 fail | 650-674, 961-963 |

### 2.13 不变量自检（session_kv_manager.py）

增量检查跟完整审计两套，失败全都是直接报错：

| # | 检查 | 违反了怎么办 | 行号 |
|---|---|---|---|
| I1 | 会话状态不是 RESIDENT/EVICTED | RuntimeError | 828-837, 2079-2093 |
| I2 | RESIDENT 却没有实例、或分片数不对 | RuntimeError | 831-835, 2082-2086 |
| I3 | EVICTED 还攥着实例指针 | RuntimeError | 836-837, 2092-2093 |
| I4 | 账算成负数、或三本账超过容量 | RuntimeError | 862-870, 2065-2072 |
| I5 | 实例内各会话 KV 之和 ≠ rank 的 resident 账 | RuntimeError | 880-896, 2101-2109 |
| I6 | 各预约之和 ≠ rank 的 reserved 账 | RuntimeError | 897-908, 2110-2117 |
| I7 | 要释放的字节比 rank 现有的还多 | RuntimeError | 721-723 |
| I8 | `SH_STRICT_KV_INVARIANTS` 开 | 每次改动后追加完整审计（默认只做增量局部检查） | 910-911, 484-488 |
| I9 | 跑完 `assert_final_state` | 有残留预约 / 压力键 / 活跃会话，任一在就 RuntimeError | 2052-2062 |

---

## 3. 映射和 KV 怎么互相影响

**映射听 KV 的。** Prefill 准入每一步都在问 KV 账本：先看 decode 实例还剩多少地方（A4），旧 KV 恰好在 decode 目标上才开"缺多少约多少"那条（A5），历史的三种情况完全由会话状态决定（H5/H6/H7），堵了就退预约、原地等，绝不改绑别的实例（F2 写死的规矩）。

**KV 听映射的。** 到达时挑中的 Prefill 实例决定历史 KV 第一站搬到哪（H7 的目标）；静态路由决定 prefill 算完 KV 去哪个 Decode 实例（M5）；列车编组决定成员什么时候做 3000 搬家（G8，跟着车头走）；最后一轮的判定触发终局清账（N5/N6）。每一次字节变动都进 journal，跑完对总账（J7）。

**把两边串起来的是容量计数（capacity_epoch）。** KV 侧任何字节变化——赶人、搬家、释放、完成——都给相关实例的计数 +1，这就是把堵着的准入请求叫醒的唯一信号（A3 的门被打开）；反过来，每次准入和发射又制造新的 KV 变化。整个显存管理就是一个循环：准入放不下 → 被动赶人 → 计数 +1 → 被叫醒重试。没有主动按水位赶 KV 的机制（B3 那个目标数只出报告），完成路径也永远不删 KV——删 KV 只发生在准入要腾地方的时候。
