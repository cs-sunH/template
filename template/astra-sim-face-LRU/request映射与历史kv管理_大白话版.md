# request 映射与历史 KV 管理（大白话版）

代码位置速查：Python 侧主体在 `sh_test_mesh/workload/llama2_7b_inference/`（face_scheduler.py、online/face_online_scheduler.py、session_kv_manager.py、online/graph_batch_builder.py、generate_face_trace.py、plan_materializer.py），统计工具在 `sh_test_mesh/slo_tools/`，C++ 执行层在 `astra-sim/workload/execution_driven/`。下文行号都指这些文件。

## 整体架构：东西怎么一层层往下走

这个模拟器模拟的是很多台 NPU 一起跑 LLM 推理。硬件是一张 9×6 的网格，一共 54 颗 NPU。软件上把它切成 9 个"实例"，每个实例占 6 颗 NPU。实例内部 6 张卡做张量并行，KV 缓存也按 6 份切开、每卡存一份。一个请求从进来到跑完，始终归某个实例管：先在一个实例上做 prefill，prefill 做完再挑一个实例做 decode。会话（多轮对话的一整个 session）的历史 KV 就留在实例的显存里，下轮接着用。

从上到下分五层，加上一个跑完之后才用的统计层：

1. **装载与物化层**（plan_materializer.py + generate_trace.py 的读队列部分 + generate_face_trace.py 的读配置部分）。请求队列 CSV 读进来，给每个请求算出三个数：`history_tokens_before`（这轮之前会话已经攒了多少 token 的 KV）、`prefill_context_tokens`（前面那个数加本轮新 prefill 的 token 数）、`final_context_tokens`（再加 decode 长度）。第一轮 history 是 0；之后每轮的 history 就是上一轮的 final。请求几点到、下一轮隔多久来，也在这层定死。

2. **调度层**（online/face_online_scheduler.py，纯算法在 face_scheduler.py）。请求一到，挑一个 prefill 实例；prefill 做完（代码里叫 drain），再挑一个 decode 实例。挑完还要过准入：目标实例的 KV 显存够不够、要不要先赶走别的老会话。够就放行，不够这轮就先不跑，下个决策点再试。调度层的计时单位叫"列车"（train）：一个实例一次连续跑的那串迭代就是一列车，多个 decode 请求拼在同一列车里，decode 和 prefill chunk 也能混着拼；一批成员只会在列车边界上增减。

3. **KV 管理层**（session_kv_manager.py 的 `SessionKVCacheManager`）。每个会话的 KV 只有两个状态：要么整份在某个实例的本地 HBM 上，要么整份在远端内存池里，没有中间状态。会话做完一轮不删 KV（留给下轮）；本地放不下了，就按 LRU 挑完整会话整个搬到远端池；下轮来了再整个搬回来。每颗 NPU 的 HBM 用了多少字节，这层记数。

4. **构图发射层**（online/graph_batch_builder.py，发节点的原语在 generate_face_trace.py）。调度层做完决定，这层把决定翻译成 C++ 模拟器能执行的节点：1000 类的历史迁移、3000 类的 prefill→decode 迁移、remote_store/remote_load 远端池读写、1 字节的控制信号，都在这层变成具体的节点和边。

5. **C++ 执行层**（astra-sim/workload/execution_driven/）。真正按时间跑。到达闹钟、节点完成这些事件通过一个文件桥送回 Python，Python 做下一轮决策，再把新的节点图交回去，来回循环。C++ 只检查节点图的格式（两个实例索引是不是非负数这类），"请求该去哪个实例"全由 Python 说了算。

统计层（slo_tools/ 的 kv_cache_adapter.py、hbm_watermark.py）跑完之后重放决策日志，算 KV 命中率和 HBM 占用水位曲线，不参与任何决策。

一次完整循环是：C++ 报"到点了/干完了" → Python 处理完成批、到达批，再做准入和发射 → 新节点图交回 C++ → C++ 跑完再报。KV 的每次变动（复用、迁移、搬走、搬回、变大）都记成事件，一份流向构图层变成物理搬运节点，一份流向统计层变成数字。

---

## 一、request 映射：请求被派到哪个实例、哪张卡

### 1. 开工前的静态检查：9 个实例怎么铺满 54 张卡

配置读进来先建实例表（`build_instances`），这一步全是检查，不过就报错退出：

| 条目 | 条件 | 发生什么 | 位置 |
|---|---|---|---|
| rank 坐标 | 每个 rank | 行列号用 `divmod(rank, mesh_cols)` 算出来：第几行、第几列 | face_scheduler.py:132-135 |
| 实例必须是满矩形 | 实例的 rank 坐标外框里有空格 | 构造时报错。实例必须是不缺角的矩形 | face_scheduler.py:180-188 |
| 全覆盖 | 实例的 rank 加起来不是正好 54 颗、每颗一次 | 报错。每颗 NPU 必须恰好属于一个实例 | face_scheduler.py:202-203 |
| 等大 | 实例大小不一 | 报错。实例之间直连搬 KV 要求两边卡数（TP 度）相同 | face_scheduler.py:204-207 |
| 拼起来不许有孤岛 | 实例相邻图有连不到的块 | 报错。所有实例必须连成一片 | face_scheduler.py:223-224 |
| 组名不许重复 | `inference_group` 重名 | 报错。发射层是按组名判断"是不是同一个实例"的，重名会让跨实例迁移被错当成同实例、零字节零时间地凭空消失 | generate_face_trace.py:297-308 |
| 建相邻表 | 物理上挨着的实例（网格里上下左右一步） | 连成无向图，每条边权都是 1 | face_scheduler.py:209-230 |

### 2. 请求到达：挑 prefill 实例（`_on_arrival`）

两步：先纯排序挑一个，再只查容量、不改任何状态地过滤一遍。

**第一步，排序挑（`select_prefill_instance`，比较键是 `(剩余 chunk 数, 最近一次收到请求的时间, 实例编号)`，取最小）：**

| 条目 | 条件 | 发生什么 | 位置 |
|---|---|---|---|
| 挑队列最浅的 | 某实例队列里剩的 prefill chunk 总数最少 | 请求排进这个实例的队列，按先来后到排 | face_scheduler.py:331-343; face_online_scheduler.py:1230-1252 |
| 剩得一样多 | chunk 数打平 | 比谁最近一次收到请求更早；从没收过请求的实例（按 -1 算）最优先 | face_scheduler.py:331-337 |
| 还是打平 | 前两项都一样 | 挑实例编号最小的，保证每次模拟结果可复现 | face_scheduler.py:336-343 |

**第二步，容量过滤（`_filter_prefill_selection`）：**

| 条目 | 条件 | 发生什么 | 位置 |
|---|---|---|---|
| 按增量算需求 | 会话 KV 已经在这个候选实例上 | 只查"还差多少"：本轮全量需求减去已经驻留的字节，不按全量吓自己 | face_online_scheduler.py:2190-2198 |
| 按全量算需求 | 没有会话 / 会话在远端 / 会话在别的实例上 | 需求 = prefill 上下文的全部 KV 字节 | face_online_scheduler.py:2182-2189 |
| 原选放得下 | 排序挑中的实例容量够 | 直接用，日志一个字不多写 | face_online_scheduler.py:2202-2203 |
| 原选放不下、别人放得下 | 幸存集合非空 | 在放得下的实例里按同一个排序键重挑一个，写一条 `prefill_admission_probe` 决策日志 | face_online_scheduler.py:2212-2222 |
| 全放不下 | 一个都放不下 | 还是排进原选实例，准入那一步会被卡住、下轮重试。注意这个预检口径偏保守：只看当前剩余，不把"可以赶走老会话腾出来的地方"算进去；它也绝不改派已定的选择、绝不动容量 | face_online_scheduler.py:2223-2229, 2043-2062 |

挑完落定：`prefill_instance_index` 定死不再变，请求排进该实例队列，实例的"最近到达时间"刷新（face_online_scheduler.py:1246-1257）。

### 3. prefill 做完：挑 decode 实例（`_on_prefill_drain`）

prefill 的物理计算结束（列车上的 drain 标记完成）时，请求出队，一次性挑 decode 实例。

**先圈候选（`schedulable_instances`）：**

| 条目 | 条件 | 发生什么 | 位置 |
|---|---|---|---|
| 圈范围 | 实例离 prefill 实例的跳数 ≤ `schedulable_distance_limit` | 进候选。这个上限 = 卡间带宽 ÷ 本地 HBM 带宽，就是个比值，限制了"搬 KV 最远能搬几跳" | face_scheduler.py:67-68, 388-394, 444-447 |
| 自己算最近的 | 源和目标是同一个实例 | 距离 0，永远在候选里 | face_scheduler.py:368-369 |
| 候选是空的 | 所有实例都超出距离 | 报错，这属于配置就配错了 | face_scheduler.py:448-449 |

**再算代价（face_scheduler.py:452-497）：**每个候选算两份 Roofline 时间估算——"现在跑一步多久"和"加了这个请求后跑一步多久"，差值除以实例卡数，得到"每张卡多花多少纳秒"。估算函数内部还有几个小分支：没有 prefill 也没有 decode 活时直接返回 0（273-274 行）；MLP 是 swiglu 还是 gelu 倍率不同（285 行）；prefill 段、decode 段哪段没活就跳过不算（300-314 行）。

**然后挑（face_scheduler.py:498-508）：**

| 条目 | 条件 | 发生什么 | 位置 |
|---|---|---|---|
| 代价最小 | 某候选每卡增量唯一最小 | 选它 | face_scheduler.py:498-507 |
| 代价精确打平 | 两个以上候选每卡增量完全相等，且传了 `DecodeTieCounter` | 打平的按实例编号排好，用计数器轮流挑（这轮挑第一个、下轮挑第二个……），同一段决策序列可以复现 | face_scheduler.py:414-421, 503-505 |
| 打平但没给计数器 | 没传计数器 | 挑编号最小的（老行为） | face_scheduler.py:506-507 |

**最后容量过滤（`_filter_decode_selection`，face_online_scheduler.py:2064-2154）：**

| 条目 | 条件 | 发生什么 | 位置 |
|---|---|---|---|
| 按增量算需求 | 会话 KV 在本地且就在这个候选上 | 需求 = final 全量减去已驻留字节（跟下面 `move_prefill_to_decode` 的口径一致） | face_online_scheduler.py:2099-2109 |
| 按全量算需求 | 其余情况 | 需求 = final 全量。搬过去之前源实例那份还占着，所以按双份中最多的算 | face_online_scheduler.py:2094-2098 |
| 原选放得下 | 挑中的实例容量够 | 直接用，平局计数器都不动 | face_online_scheduler.py:2112-2113 |
| 原选放不下、别人放得下 | 可行集合非空 | 在可行集合里按每卡增量最小重挑，精确打平按编号用计数器轮流；写 `decode_admission_probe` 日志 | face_online_scheduler.py:2122-2147 |
| 全放不下 | 可行集合空 | 保持原选，进下一轮的阻塞-重试。铁律是"绝不 remap"：选择只做这一次，落定之后容量不够可以等，但绝不再换实例 | face_online_scheduler.py:2148-2154 |

落定：`decode_instance_index` 定死，请求登进目标实例的等待名单，标 dirty，并立刻试一把 decode 准入（face_online_scheduler.py:1316-1330）。

### 4. prefill 准入：真要把 KV 落进显存（`_try_admit_prefill`）

到了服务时点，逐个请求试着正式准入。每个实例有一个"容量纪元"计数器，容量一变就加一——用它判断"上次失败之后容量变没变过，没变就别白试"。

| 条件 | 发生什么 | 位置 |
|---|---|---|
| 已经准入过了 | 直接返回 True | face_online_scheduler.py:1544-1545 |
| `prefill_instance_index` 是 None | 报错（映射丢了） | face_online_scheduler.py:1546-1547 |
| 纪元没变、dirty 也没标 | 返回 False，这轮不试（防止同一个状态反复重试） | face_online_scheduler.py:1552-1555 |
| 纪元推进了或 dirty 置位了 | 记下本次尝试的纪元、清掉 dirty，继续往下走 | face_online_scheduler.py:1556-1560 |
| 会话在远端、且它还有一笔"决定了但还没发出去"的 remote_store | 推迟：标 dirty、返回 False。原因：这个请求进来要从远端池读回 KV，而上一轮往池里写的那笔还没落地，必须先写完再读。store 发射出去后计数归零，下轮重试。推迟期间什么都不改 | face_online_scheduler.py:1570-1575; graph_batch_builder.py:1042-1045 |
| `prepare_history` 返回"被卡住"（容量不够，赶了人还是不够） | 这趟如果赶走了别的会话，纪元推进（重开重试门）；赶人的动作是真的、不回滚——remote_store 物理链随当前批发出去，发过的从待发清单里划掉；标 dirty，返回 False | face_online_scheduler.py:1606-1635 |
| `prepare_history` 成功 | 记下命中类型、KV 来源实例、要搬多少字节（本地命中算 0 字节） | face_online_scheduler.py:1636-1647 |
| `grow_prefill` 居然失败 | 报错。前面都成功了的增长不可能失败，出现就是代码出事了 | face_online_scheduler.py:1654-1656 |
| 全部成功 | 重算 `remaining_chunks`（只数本轮 prefill 的 chunk 数；被逐出的会话回来靠搬数据，不靠重算，重算这条路已经删了）；`admitted_prefill=True`；纪元推进 | face_online_scheduler.py:1666-1684 |

### 5. decode 准入：把 KV 挪到 decode 实例（`_try_admit_waiting_decodes`）

| 条件 | 发生什么 | 位置 |
|---|---|---|
| 实例有等待队列、且（dirty 或纪元没追上） | 进本轮扫描；扫描开始时同步纪元、清 dirty | face_online_scheduler.py:1692-1706 |
| 队员已经不在等待状态 | 跳过 | face_online_scheduler.py:1711-1712 |
| 队员的目标实例和所在队列对不上 | 报错（队列坏了） | face_online_scheduler.py:1713-1715 |
| `move_prefill_to_decode` 被卡住 | 赶了人就推纪元；这趟的逐出立刻发射并从待发清单划掉；标 dirty；队员排回队尾接着等——被卡住不等于丢掉，这是 face 特有的语义 | face_online_scheduler.py:1736-1772 |
| move 成功 | 记下迁移传输；`grow_decode` 必须成功否则报错；decode 逐出的记录往前累，并同步"会话位置"镜像 | face_online_scheduler.py:1773-1795 |
| 迁移和增长都完成 | 队员进 `pending_decode_ready`，等 decode 实例的下一班列车（3000 类迁移随列车头先发，物理上先于列车体，保证列车体开跑时 KV 已就位）；实例空着就进就绪集合；assignment 条目在这时一次带上两个实例索引（prefill 边界不出 assignment）；写 decode 决策行，把每个候选的代价都记上 | face_online_scheduler.py:1796-1873 |

### 6. 实例内部：列车怎么拼、怎么发（`_plan_train` / `_plan_and_emit_trains`）

| 条件 | 发生什么 | 位置 |
|---|---|---|
| 队列没头、也没有 decode 成员 | 这实例没活，列车计划返回 None，移出就绪集合 | face_online_scheduler.py:809-811, 953-956 |
| 没头但有 decode 成员 | 纯 decode 列车：迭代数 = 剩 token 最多的那个成员的剩余数 | face_online_scheduler.py:812-813 |
| 队列有头 | 混拼列车：迭代数 = 头部剩余 chunk 数；每个迭代最多带 1 个 prefill chunk，chunk 之间不互相拼 | face_online_scheduler.py:814-819, 873-896 |
| 迭代数超过 `SH_TRAIN_MAX_ITER` | 截断到上限（默认 8；设 0 表示不限；设负数构造时报错）。截断后既没有 drain 也没有 exit 标记时，拿列车自己的 id 当完成信号（哨兵），免得没人通知列车结束 | face_online_scheduler.py:820-824, 852-854, 526-536 |
| 队头 prefill 准入失败 | 队头这趟不出 chunk 活，本列车退化成纯 decode | face_online_scheduler.py:937-941 |
| 队头 `remaining_chunks == 0` | 它的 drain 决策还在路上，先跳过它，挑下一个有剩余 chunk 的成员当头 | face_online_scheduler.py:929-936 |
| 某成员本次参加的迭代数 ≥ 它剩余的 token 数 | 它的退出标记挂在这列车里（不用等列车边界，列车中途就知道它要退） | face_online_scheduler.py:835-842 |
| 首 token 拆分开着（`SH_FIRST_TOKEN_SPLIT=1`）、有新加入的 decode 成员、迭代数 ≥ 2 | 分两批发：首步批（首 chunk + 各成员第一段 + 新成员的迁移 + 唤醒标记）和余量批。默认关 | face_online_scheduler.py:1083-1120; graph_batch_builder.py:87-108 |
| 拆分开着但迭代数 < 2 | 不拆车，只改标记名（多 token 的新成员挂 first_token 标记；decode 长度为 1 的成员把 exit 标记改名，同一个节点带两个锚点） | face_online_scheduler.py:1116-1120 |
| 实例已有在飞列车 | 跳过发射。一个实例同时只跑一列车 | face_online_scheduler.py:926-927 |

### 7. 请求完成时

| 条件 | 发生什么 | 位置 |
|---|---|---|
| 完成信号按 stage 分流 | prefill → 走 drain 处理；decode → 走完成处理；REQUEST → 同一拍跟 decode 一起处理（安排下一轮到达）；`batch_train_` 前缀 → 核销哨兵列车；不认识的 stage 报错 | face_online_scheduler.py:603-620 |
| 首步唤醒信号 | 自己发出去的首步批的回声，什么都不做，只从哨兵列表里划掉 | face_online_scheduler.py:627-630, 682-692 |
| 完成信号对不上任何在飞/已核销的列车 | 报错，直接终止 | face_online_scheduler.py:766-772 |
| 已核销列车的迟到残余信号 | 清掉 pending 标记就行，不改任何就绪状态 | face_online_scheduler.py:758-787 |
| 同会话还有下一轮（turn+1） | 下一轮请求的到达闹钟 = max(父请求物理完成的 tick + 间隔, 本次交付的 tick)。取 max 是防"排出来的到达时刻比这次交付还早"，那种到达 C++ 会直接拒收 | face_online_scheduler.py:1441-1471 |
| 没有下一轮（terminal） | 回收完成门 + 调 `kv_manager.retire_terminal_session` 做终局清理（见 KV 部分第 6 节最后一条）+ 容量纪元推进 | face_online_scheduler.py:1472-1484 |
| 整个 run 结束 | 收尾大检查：到达堆、就绪集合、没发射的余量、store 登记、等待队列、KV 终态，全部必须空，哪个不空就报错 | face_online_scheduler.py:2301-2376 |

### 8. 更早的入口：装载与物化的检查

| 条件 | 发生什么 | 位置 |
|---|---|---|
| 队列行是 turn-0 | 必须带 `session_arrival_time_ns`，且 interval 列必须空，否则报错 | generate_trace.py:233-247 |
| 队列行是 turn>0 | 必须带 `inter_request_interval_ns`，且 arrival 列必须空，否则报错；时间值必须是整千纳秒 | generate_trace.py:248-271 |
| request_id 重复，或 (session_id, turn_index) 重复 | 装载时报错 | generate_trace.py:216-228 |
| `request_queue_session_limit` 设了值 | 0 = 用全部会话；N = 按源文件顺序取前 N 个会话的全部行；超过可用会话数报错 | generate_face_trace.py:214-241 |
| 派生 turn-0 的数 | history = 0；context = prefill_length（队列里的 prefill 已经算进去了，一个口径） | plan_materializer.py:275-279 |
| 派生 turn>0 的数 | 按 (session_id, turn-1) 精确找父请求，history = 父的 final，context = history + prefill_length；找不到父报错。注意是按键找，不是"文件里上一行" | plan_materializer.py:280-288 |
| 派生 final | 恒等于 context + decode_length | plan_materializer.py:289 |
| 派生到达时间 | turn-0 用绝对时间；turn>0 用父的 queue_index 加间隔；父缺了报错 | plan_materializer.py:332-349 |
| `request_queue_session_limit > 0` | 物化入口直接拒绝：Python 这边截断了，C++ 那边还是全量读物化后的 CSV，两边对不上会先错统计再崩 | plan_materializer.py:392-401 |
| NPU 总数不是 54 | 拒绝。plan 目录名钉死了 54npus | plan_materializer.py:409-420 |
| `kv_cache_policy` 不是 `session_lru_tiered` | 报错（防配置值漂移） | plan_materializer.py:78-90 |

### 9. C++ 那边查什么（只查形状，不查该去哪）

| 条件 | 发生什么 | 位置 |
|---|---|---|
| assignment 条目缺 request_id / 缺某个实例索引 / 索引为负 | 提交前失败 | ParsedGraphBatch.cc:546-576; GraphBatchCommitter.cc:1525-1531 |
| 节点或边的 rank 越界 | 失败 | ParsedGraphBatch.cc:131-135 |
| 同一 delivery_sequence 的重复交付 | 深度比对内容：一致就原样重放上一次的结果（状态零变化）；同号不同内容、或跳号，报错 | online_scheduler_base.py:505-534 |
| commit ack 说失败 | 报错。KV 的临时记账不能拿失败的 ack 定案 | online_scheduler_base.py:1032-1035 |
| 同一请求同一阶段完成两次，或二次到达 | 报错 | online_scheduler_base.py:883-937 |

### 10. 配置开关

| 开关 | 取值 | 什么意思 | 位置 |
|---|---|---|---|
| `kv_cache_policy` | 只认 `session_lru_tiered`，别的直接报错 | 本仓唯一的 KV 行为，没有第二档 | generate_face_trace.py:168-179; face_online_scheduler.py:436-440; online_service.py:192-216 |
| `trace_granularity` | 在线发射只认 `request_aggregated` | 别的值在发射入口报错 | graph_batch_builder.py:569-572; generate_face_trace.py:180-186 |
| `prefill_chunk_size` | 正整数，默认 512；小数或 ≤0 构造时报错 | prefill 切 chunk 的粒度 | face_online_scheduler.py:455-468 |
| `request_queue_session_limit` | loader 认 N，但物化入口对 >0 直接拒绝 | 想截会话子集，实际不可用 | generate_face_trace.py:214-241; plan_materializer.py:392-401 |
| `SH_TRAIN_MAX_ITER` | 默认 8；0 不设限；负数报错 | 列车长度上限 | face_online_scheduler.py:526-536 |
| `SH_FIRST_TOKEN_SPLIT` | 默认 "0" 关 | 首 token 两段式拆分 | graph_batch_builder.py:87-108 |
| `SH_QUEUE_AGG_VERIFY` | "1" 时启用 | 队列聚合数的影子校验 | face_online_scheduler.py:552-553, 2022-2031 |
| `SH_STRICT_KV_INVARIANTS` | 默认关 | 开了以后每次 KV 变动都跑全量审计 | session_kv_manager.py:66-74, 1171-1172 |
| `SH_PROPAGATING_TAIL_LIMIT` | 默认 8 | 在途尾部数量的上限，超了报错 | online_scheduler_base.py:53-54, 349-368 |
| `SH_GRAPH_DIGESTS` / `SH_PROFILE_JSONL` / `BRIDGE_TIMEOUT_MS` | 默认 0/0/120000ms | 审计产物开关和桥超时 | online_service.py:59-65, 172-174; decision_bridge.py:49-66 |

---

## 二、历史 KV 管理：一个会话的 KV 一辈子经过哪些事

先说清楚状态。会话是 KV 的唯一归属单位，状态只有两个值：

- `LOCAL_HBM`：整份 KV——所有层、所有 TP 分片——都在某一个实例的本地 HBM 上；
- `REMOTE_MEMORY`：整份都在远端内存池，本地一点不占。

代码里还留着一个 `partial_hbm_remote` 字符串（一部分层在本地、一部分在远端），但它只用来解析旧日志。运行期你要是手工把会话改成这个状态，代码直接报错——现在的系统里根本到不了这个状态（session_kv_manager.py:46-58, 1078-1082, 2255-2259；不可达的测试证明在 test_face_kv_incremental_invariants.py:231-268）。

### 1. 请求来了先判断：命中判定（`prepare_history`，session_kv_manager.py:1669-1928）

| 条件 | 发生什么 | 位置 |
|---|---|---|
| 输入检查不过（新会话却带历史 token 数 / 会话记的上下文和请求说的对不上 / 会话还有请求在跑） | 报错 | session_kv_manager.py:1690-1705 |
| 会话不存在且 history=0（**NO_HISTORY**） | 先按全量问容量：放不下就返回"被卡住"的决定；放得下就建一条会话记录（LOCAL、0 字节、active），发 `no_history` 事件 | session_kv_manager.py:1722-1764 |
| 会话在本地、就在目标实例上（**LOCAL_HIT**） | 需求按增量算（目标减已有）。放不下返回被卡；放得下就零开销复用：不出任何网络传输，发 `local_hit` 事件，显存水位一个字节不动 | session_kv_manager.py:1766-1829, 1186-1211 |
| 会话在本地、但在别的实例上（**NOC_MIGRATE**） | 需求按全量算（搬过去之前源那边还占着）。放不下返回被卡；放得下就整份 NoC 迁移过去，逐 TP 分片走 XY 路由。顺序是先给目标加字节、再从源减字节——先确认目标进得去，再放掉源。然后会话的实例指针一次性切换 | session_kv_manager.py:1831-1870, 1213-1259 |
| 会话在远端（**REMOTE_RESTORE**，唯一的恢复路） | 先腾容量——注意这一步本身就可能把别的会话赶走。放不下返回被卡；放得下一笔 `remote_load` 把整份 [0, 层数) 搬回来，全部字节加回，状态切回 LOCAL、active，发 `remote_load` 事件。搬完之前这个会话一直保持 REMOTE，别人碰不了它 | session_kv_manager.py:1872-1928, 1261-1313 |
| location 不是这两个值（含手工塞的 PARTIAL） | 报错 | session_kv_manager.py:1872-1873 |
| 算出来的增量需求是负数 | 报错（历史准备不许把 KV 缩小） | session_kv_manager.py:1779-1780 |

### 2. 腾地方：容量检查与 LRU 赶人（`ensure_physical_fit`，session_kv_manager.py:1461-1602）

| 条件 | 发生什么 | 位置 |
|---|---|---|
| 分片向量长度和 TP 度不符，或带负数 | ValueError | session_kv_manager.py:1484-1485 |
| 结构不可行：某张卡上"需求 > 容量 − 模型权重"（把实例清空都装不下这个请求） | 先记一条深缺口记录（reason=request_exceeds_empty_instance），然后 raise——整个函数唯一 raise 的出口，直接报错，这次不跑了 | session_kv_manager.py:1486-1511, 911-945 |
| 每张卡剩余都够 | 不赶任何人，直接通过 | session_kv_manager.py:1514-1519 |
| 有缺口 | 进 LRU 循环：受害者候选 = 在这个实例上、LOCAL、不 active、完成过的会话，按 (完成时刻, 会话id) 从旧到新排；触发请求自己的会话被保护名单挡在外面。每赶走一个完整会话（调 `_evict_session`），立刻重查每张卡的缺口，够了就停。候选池的全程扫描会如实计数到 profile | session_kv_manager.py:1520-1559, 782-813 |
| 循环里发现受害者状态变了（不是 LOCAL 了 / 换了实例 / active 了 / 没完成过） | 报错（快照和执行之间状态漂了） | session_kv_manager.py:1539-1549 |
| 候选赶光了还不够 | 记深缺口（reason=exhausted_completed_candidates）+ 发一条 `admission_blocked` 压力事件（带去重：每个请求/相位/目标只发一条），返回"没准入"。注意不 raise——就是推迟，等下轮再试 | session_kv_manager.py:1561-1585, 856-899 |
| 返回前复查每个被赶的会话确实完成过、不 active、不在保护名单 | 有违例就报错（循环之外再兜一层） | session_kv_manager.py:1591-1602 |

旁边还有一对预约接口 `reserve_request_capacity` / `release_request_capacity`（1604-1664 行）。主链路上没人调用它们——face 的容量检查全走 `ensure_physical_fit`，预约表实际永远是空的，只有测试在用。留着当扩展位。

### 3. 干活期间：增长、完成、迁移

| 条件 | 发生什么 | 位置 |
|---|---|---|
| 增长前置检查不过（会话不是 LOCAL / 不在本实例 / 不 active） | 报错 | session_kv_manager.py:1941-1947 |
| 要长到的数比现有的小 | 报错（上下文不许缩） | session_kv_manager.py:1949-1951 |
| 增长的容量检查失败 | 返回没准入。增长同样会触发赶人 | session_kv_manager.py:1952-1963 |
| 增长成功 | 加上增量字节，token 数前移。`grow_prefill`（prefill 段）和 `grow_decode`（decode 段）入口各自校验实例在场 | session_kv_manager.py:1964-1971, 1973-2109 |
| 请求完成（`mark_complete`）——LRU 池的入口 | 不删 KV：active 置 False，记下完成时刻（这就是 LRU 的排序键），发 `retain_complete` 事件，清掉这个请求的压力去重键，返回空逐出列表。完成这一刻不赶人——KV 留在原地，等下次谁需要地方再说 | session_kv_manager.py:2111-2149 |
| `move_prefill_to_decode`：源实例 == 目标实例 | 出一个 local_hit 传输，零字节、水位不动 | session_kv_manager.py:2031-2056 |
| `move_prefill_to_decode`：源 ≠ 目标 | 先在目标上过容量检查（可能赶人），不行就返回"被卡"（transfer 就是 None；以前塞一个假的 local_hit 占位的写法已经删了，免得误读）；行就整份 noc_migrate 过去 | session_kv_manager.py:2013-2030, 2058-2089, 510-523 |
| 审计遇到：非法 location / LOCAL 但层数不满 / REMOTE 还带着实例 / 分片数和 TP 度不符 / 分片加起来不等于总量 | 全部报错。这是两态模型的硬检查 | session_kv_manager.py:1073-1098, 2234-2304 |
| 每卡记账：占用 = 模型权重 + 驻留 + 预约，超了容量或出现负数 | 报错（HBM 守恒） | session_kv_manager.py:1123-1131, 2240-2247 |
| 事件列表攒太多了（已消费 ≥ 8192 条且过现存一半） | 删掉前面的整段，游标回退，摊下来每条 O(1) | session_kv_manager.py:63, 664-675 |
| `SH_STRICT_KV_INVARIANTS` 开着 | 每次 KV 变动后追加一次全量审计 | session_kv_manager.py:1171-1172 |

### 4. 赶人：整会话一次性 LRU 逐出（`_evict_session`，唯一的逐出路，session_kv_manager.py:1382-1459）

| 条件 | 发生什么 | 位置 |
|---|---|---|
| 受害者是 active / 没完成过 / 不是 LOCAL / 没有实例 | 报错。只有"完成过的、不 active 的、完整在本地的"会话能被赶 | session_kv_manager.py:1399-1402 |
| 检查通过，执行赶人 | 一笔 `remote_store` 盖全部：层范围 [0, 模型层数)、全部 TP 分片、全部本地驻留字节，一个不落。每个源 rank 找最近的边缘口（曼哈顿距离最小，打平取编号小的 rank），XY 路由送出去写进池。本地字节全部释放，状态从 LOCAL 切到 REMOTE（实例指针清空、驻留层数清零），发 `evict_full` 事件（带前后水位） | session_kv_manager.py:1403-1459, 1315-1370, 271-284 |
| 一个受害者一条记录 | 多条 TP 分片传输不折算成多个逻辑受害者，EvictionRecord 恒挂一笔 transfer | session_kv_manager.py:1431-1442; test_face_tiered_eviction_sequence.py:288-327 |
| 一个会话的全部字节比缺口还大 | 照样整份释放。多赶出去的字节就多赶了，不报错、不回滚，绝不出现"只赶一半层" | session_kv_manager.py:1476-1479; test_face_tiered_eviction_sequence.py:137-163 |
| LRU 排序 | (完成时刻, 会话id) 升序：完成时刻不同按真实先后，打平按会话 id 字典序。没有任何按请求类型分层挑人的花活 | session_kv_manager.py:782-813; test_face_tiered_eviction_sequence.py:165-180 |
| active / 在飞的会话 | 永远不进受害者池；触发请求自己的会话被保护名单挡住，返回前还有复查 | session_kv_manager.py:806-813, 1521-1525, 1591-1601; test_face_tiered_eviction_sequence.py:182-239 |
| 完成时刻 | 不触发逐出（上一节说过了）：完成的会话进 LRU 池等着，被谁的需求触发才走 | session_kv_manager.py:2111-2149 |
| 边缘口没显式配 | 默认用 mesh 的物理边界（第一行、最后一行、第一列、最后一列的全部 rank）；显式配了就查重、排序、逐个校验坐标 | session_kv_manager.py:579-595, 243-260 |
| 远端池 | 没有容量上限，池子自己从不往外赶东西；只有 terminal 会话回收能把远端账面划掉 | session_kv_manager.py:546-553, 2151-2220 |

### 5. 搬回来：恢复的物理链（REMOTE → LOCAL）

| 条件 | 发生什么 | 位置 |
|---|---|---|
| REMOTE_RESTORE 决策发射 | `remote_load` 链：控制侧（到达/间隔 gate）→ 边缘 rank 的 mem_load（只有池口这一段计时间）→ 边缘到目标 rank 的数据 p2p（两端都不计 HBM）→ 目标 rank 的 `local_hbm_kv_restore`（整条恢复链里唯一计 HBM 写的点，可以和计算重叠） | generate_face_trace.py:977-1065; graph_batch_builder.py:1417-1441 |
| 控制侧就是边缘 rank | 直接挂 timer gate | generate_face_trace.py:991-992 |
| 控制侧和边缘不是同一个 rank | 发 1 字节 p2p 中继过去 | generate_face_trace.py:993-1009 |
| 同一个会话还有没落地的 store（B4 补依赖） | 回迁发射前补显式依赖：同一个边缘就直接 arm 那笔 mem_store 的依赖；跨边缘用 1B p2p 中继；有登记但回迁链一个边缘分片都没有——报错（防御端点）。这一套保证池写一定排在池读前面 | graph_batch_builder.py:1157-1216 |
| 会话有"决定了但还没发射"的 remote_store | 调度器推迟它下一轮的回迁准入，纯推迟，什么都不改 | graph_batch_builder.py:462-472, 1042-1045; face_online_scheduler.py:1570-1575 |
| 恢复链还在飞 | 规划状态已经前移了，但物理恢复节点排在列车体前面（readiness 屏障 + 一个 all_reduce 等所有卡到位），列车体不会读到没搬完的 KV | graph_batch_builder.py:1442-1445 |

### 6. 逐出的物理链与发射细节

| 条件 | 发生什么 | 位置 |
|---|---|---|
| 发逐出（history / prefill / decode / 被卡的尝试） | 逐出的搬运节点单独成一条链挂在主链旁边（`_emit_side_branch`）：全 rank 暂存 pending 依赖、快照、链内自己接自己、然后恢复主链。链上留下没消费的依赖就报错。这条链不挡主链任何节点；HBM 争用由 C++ 那边 N 路均分当场裁决 | graph_batch_builder.py:520-553, 1063-1086, 1315-1335, 1406-1415 |
| remote_store：源 rank 就是边缘 rank | 一笔 mem_store 搞定——池读的唯一计费点 + 池口双异步 join，没有 ack | generate_face_trace.py:901-921 |
| remote_store：源 ≠ 边缘 | 源 comm_send（源端 HBM 读的唯一计费点）→ 边缘 comm_recv（不计费）直通 → mem_store 写池 → 1 字节 ack 双端（源收到 ack 才算物理释放） | generate_face_trace.py:922-975 |
| 逐出带触发门 | 控制侧节点对齐逐出开始时刻：history 逐出的门 = 到达/间隔 gate；decode 逐出的门 = 成员 drain 列车 barrier 块的末尾。控制 rank == 源 rank 就同 rank arm，否则发 1B trigger p2p | generate_face_trace.py:736-783; graph_batch_builder.py:748-770, 1316-1334 |
| 被卡的准入尝试已经发出去的逐出又到发射口 | 不重发。决策日志记全集；待发清单按对象 id 把已发射的划掉——最终准入批和 joiner 列车头不会把同一批 remote_store 发两遍 | face_online_scheduler.py:244-249, 1617-1628, 1747-1758 |
| 每笔 remote_store 物理发射 | 登记边缘 mem_store 尾节点（一个会话最多一条，因为整份 store 就一笔）；"决定了未发射"的计数核销一笔；terminal 回收时两张表都清 | graph_batch_builder.py:1125-1155, 984-996 |
| noc_migrate 传输进了 `_emit_kv_transfer` | 报错。NoC 迁移永远走 `_paired_transfer`（1000/3000 类）那条合同路径 | generate_face_trace.py:851-864 |
| `_paired_transfer` 源组名 == 目标组名（同一个实例） | 只 arm timer gate，零节点零字节 | generate_face_trace.py:451-455 |
| 迁移字节切分 | 总数能整除注意力头数就整头分；不能就精确字节配额，不取整凑数 | generate_face_trace.py:462-475 |
| 完成时的位置传递 | REQUEST_COMPLETE 时登记会话位置；后续逐出发射时镜像更新（remote_store → remote_memory）；下一轮准入时弹出来当初值并校验值域。partial 层域的 store 直接报错 | graph_batch_builder.py:998-1061, 1284-1292 |
| 到达/间隔的 duration 是 0 | 不发 gate 节点，直接用 after_node_id 把依赖编进去（等待由 C++ 的到达闹钟承担；再发 gate 会等两遍） | graph_batch_builder.py:241-296 |
| pending gate 位置和计划快照对不上 | 报错（没有 partial 折算的特例） | graph_batch_builder.py:1304-1313 |
| terminal 轮完成 | `retire_terminal_session`：LOCAL 的释放本地字节；REMOTE 的远端账面直接划掉（不发传输、不算逐出）；预约和压力键清理；不发 KVCacheEvent | session_kv_manager.py:2151-2220 |

### 7. 跑完之后：统计工具怎么重放

**7.1 HBM 水位重放（slo_tools/hbm_watermark.py）**

| 条件 | 发生什么 | 位置 |
|---|---|---|
| 逐出条目 | 扣掉受害会话在实例上的占用；会话没被跟踪过、或扣的字节超出跟踪值就报错。"部分逐出"只会在旧产物里出现，新日志永远是整份 | hbm_watermark.py:1479-1509 |
| local_hit | 零字节；source ≠ target 时占用跟着挪到目标（防止源上留着幽灵占用） | hbm_watermark.py:1571-1587 |
| 回载且 source 是空（从远端来） | 只加目标 | hbm_watermark.py:1615-1625 |
| 回载且 source == target | 只加不扣源（旧 partial 半层的兼容写法） | hbm_watermark.py:1626-1633 |
| 回载且源在目标之外 | 扣源加目标；超出会话跟踪字节就报错 | hbm_watermark.py:1634-1646 |
| 旧产物的 RECOMPUTE / EVICTED 状态 | 恢复点补扣一次（占用上界口径） | hbm_watermark.py:1555-1566 |
| move：source == target 或 0 字节 / 否则 | 前者只换归属、占用不变；后者扣源加目标；0 字节却跨实例的，记一条 move_bytes_mismatch（幽灵占用的形态） | hbm_watermark.py:1648-1684 |
| grow：目标小于跟踪值 | 报错（上下文不许缩） | hbm_watermark.py:1686-1699 |
| 完成时的去向（`kv_location_after_completion`） | remote → 扣自己；partial → 计入上界计数；local → 不动；缺字段或不认识的值 → 记一条 anomaly | hbm_watermark.py:1701-1729 |

**7.2 可信度分层与容量口径**

| 条件 | 发生什么 | 位置 |
|---|---|---|
| journal + checksum 都在、四道检查全过（certified 层） | 唯一能出正式容量结论的层。本仓自己跑不出 journal（没有生产者），所以实际总是落到最下面那层 | hbm_watermark.py:20-56, 2331-2355 |
| checksum 在、journal 缺 | 以退出码 2 失败，不悄悄降级成上界口径 | hbm_watermark.py:2335-2342 |
| journal 在但没证书 / 证书里有 false | 出精确报告，但不出结论 | hbm_watermark.py:28-35 |
| journal 缺（upper_bound_only 层） | 决策日志重放就是权威（上界口径），occupancy_valid 恒 false，不认证物理违规 | hbm_watermark.py:36-46 |
| 容量链缺环（profile / hardware / npus 任一缺） | capacity 记 NA，只记峰值，绝不自己编一个容量数 | hbm_watermark.py:2259-2287 |
| 三种容量口径 | per-rank physical（认证用）/ resident 硬上限（诊断用）/ `kv_reserve_context_tokens` 水位目标（只报告不判罚）；容量小于权重分片就报错 | hbm_watermark.py:762-850 |
| 决策日志带 `history_transfers` 列表 | face / face-LRU 升级到 full 口径（契约行 + 逐段 restore），防止旧镜像双计和标量聚合把下界击穿 | hbm_watermark.py:874-910, 2241-2246 |

**7.3 命中态映射（slo_tools/kv_cache_adapter.py）**

| 条件 | 发生什么 | 位置 |
|---|---|---|
| 新产物有 `history_location_before` | local_hbm → full；remote_memory → full（整份在远端然后恢复，不是重算）；partial_hbm_remote → partial（只有旧产物能解析）；不认识 → not_supported | kv_cache_adapter.py:143-158 |
| 旧产物没有位置字段 | 按 `history_action` 四值：NO_HISTORY → no_history；LOCAL_HIT / NOC_MIGRATE → full；RECOMPUTE → miss | kv_cache_adapter.py:159-170 |
| 行是 admission probe（decode/prefill_admission_probe） | 跳过不消费但计数留痕；不认识的 kind 直接报错——两边都装看不见的话，对账永远显示通过 | kv_cache_adapter.py:95-108, 727-731 |
| 事件提取 | 契约行在场（history_transfers / *_evictions / completion_eviction_transfers / prefill_decode_transfer 列表）就逐段传输对象出事件；缺席就按旧产物的聚合 bytes + 镜像逐出行。local_hit 恒零事件（那个字节数是名义规模，不是搬动量） | kv_cache_adapter.py:344-448, 451-479 |

---

## 三、映射和 KV 状态怎么互相影响

映射选实例的时候读 KV 状态：容量预检看的就是"这个会话的 KV 在不在、在哪个实例、那个实例还剩多少"。会话在本地且就在候选实例上，按增量算需求，多半直接命中；在远端或在别的实例上，按全量算，走搬回路。decode 候选域的距离上限，等于在选的时候就限制了最多搬几跳。B4 的推迟规则也是先看"有没有决定了还没发射的逐出"，再决定放不放行回迁。

反过来，映射的准入是逐出的唯一触发源：一个实例是因为有新请求要进来、地方不够了，才会按 LRU 赶老的。迁移和恢复会改写会话的实例指针和位置，下一轮映射的判断依据（在不在、在哪）随之改变；会话完成时进 LRU 池，恢复时重新 active 离开池——池里有什么人，完全跟着映射的脚步走。

KV 状态又约束映射的下游：每次逐出和迁移都变成事件流和 KVTransfer 记录，流向构图层变成物理节点，流向统计层变成数字，这些数字和位置又是下一轮映射的输入。因为状态只有两态，这个循环里没有中间形态——任一时刻一个会话要么整体能就地复用，要么整体要搬，映射这边就是在这两条路里二选一。
