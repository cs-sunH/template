# request 怎么找到卡，历史 KV 怎么管（大白话版）

路径缩写：`W/` = `sh_test_mesh/workload/llama2_7b_inference/`，`O/` = `W/online/`，`S/` = `sh_test_mesh/slo_tools/`。

## 整个系统长什么样

这个模拟器仿真的是大模型推理服务：一堆请求进来，每条请求先做一段 prefill（把上下文算一遍），再一个 token 一个 token 做 decode，最后算完走人。调度代码要回答两个问题：每条请求派给哪个实例（一个实例 = 一组拼成 TP 的 NPU 卡）去算？算完留下的 KV 缓存放在哪、什么时候删？

系统分四层，活一层层往下交：

1. **准备层**。`plan_materializer.py`、`generate_trace.py`、`generate_face_trace.py` 把 CSV 变成能跑的输入：请求队列变成 `manifest.json`（每条请求多少 prefill、多少 decode、上一轮聊到哪），`trace_config.csv` 里的 `inference_group` 行变成实例布局（每个实例占哪些卡）。布局检查在这一层就先跑一遍，不合法直接报错，连配置文件都不写盘。

2. **调度层**。`O/face_online_scheduler.py` 里的 `FaceOnlineScheduler` 管所有决定：请求到了进哪个实例的队列、prefill 做完派给哪个实例做 decode、内存不够时让谁先等着。每个决策点它只看两样东西：各实例队列的当前状态，和 KV 管理器报的内存状态。

3. **KV 内存管理层**。`W/session_kv_manager.py` 里的 `SessionKVCacheManager` 记着每张卡上现在压着哪些会话的 KV、各占多少字节。调度层每次要占内存都得先问它；装不下时它去赶别的会话（最久没用的先走），赶出来的空间给新请求用。

4. **执行层**。`O/graph_batch_builder.py` 把调度决定翻译成图节点（算子、点对点通信、定时器、屏障），交给 ASTRA-sim 的 C++ 侧逐卡排事件执行。C++ 跑完一批，把"哪些 watch 触发了"写回文件；Python 读到后核销完成的、收下新到的、再决定下一批活。两边靠 delivery 序号保证一批一批严格按顺序来，一批就是一次交付。

Python 和 C++ 不在一个进程里，消息经 `O/decision_bridge.py`（文件 + 通知）来回送。每收到一批消息，Python 做完"核销完成 → 处理到达 → 试准入 → 发射"这一整套动作，算一个决策边界。

---

## 一、request 映射：每个环节的全部分支

### 1. 开跑之前把卡分好（静态拓扑）

`build_instances`（`W/face_scheduler.py:134-222`）把实例布局定死，八种情况直接报错：

| # | 什么情况 | 结果 | 位置 |
|---|---|---|---|
| T1 | 一个实例都没配 | 报错 | face_scheduler.py:140-141 |
| T2 | 实例名为空 / pg_name 为空或 "0" / pg_name 重复 | 报错 | face_scheduler.py:149-155 |
| T3 | 实例没配 rank / 同一实例里 rank 重复 | 报错 | face_scheduler.py:156-159 |
| T4 | 某张卡被分给了两个实例 | 报错，一张卡只能归一个实例 | face_scheduler.py:163-166 |
| T5 | 实例占的网格格子不是填满的矩形 | 报错 | face_scheduler.py:172-184 |
| T6 | 所有实例的卡拼起来不正好是 0..N-1 各一次 | 报错 | face_scheduler.py:194-195 |
| T7 | 要求等大（默认开）但实例大小不齐 | 报错，跨实例搬 KV 要求 TP 度一样才能一对一配对 | face_scheduler.py:196-199 |
| T8 | 实例多于一个但邻接图上有孤岛 | 报错，网格必须铺得连通 | face_scheduler.py:201-216 |

另外三件事：rank 变网格坐标就是 `divmod(rank, mesh_cols)`，越界报错（face_scheduler.py:128-131）；这组检查在物化期先跑一遍（`W/plan_materializer.py:384-395`），实例名重复在加载期另有检查（`W/generate_face_trace.py:274-284`）；邻接图的边权全是 1（face_scheduler.py:332-338）。

### 2. 模型怎么切开分到卡上（分片计算）

都在 `W/session_kv_manager.py`：

- 整数均分 `partition_values_exact`（:59-66）：除不尽的余数给编号小的秩。后面所有分片都靠它。
- 注意力头分配 `attention_heads_by_tp_rank`（:69-81）：按整个头分，一个头不劈成两半。
- 权重总量 `estimate_model_weight_bytes`（:84-109）：swiglu 是 3 个 MLP 矩阵 + 2·hidden 的 norm + 末尾 norm；gelu 是 2 个矩阵 + 4·hidden 的 norm、没有末尾 norm。就这两种，按 `mlp_variant` 挑。
- 权重分片 `model_weight_shard_bytes_by_tp_rank`（:112-144）：矩阵部分按 TP 切，norm 之类的零头走均分；算出负数或加总对不上 → RuntimeError。
- KV 分片 `kv_cache_shard_bytes_for_tokens`（:147-166）：每张卡的 KV = 2 × 层数 × token 数 × head_dim × 每元素字节 × 这张卡分到的头数；加总对不上 → RuntimeError。
- `hidden_size % num_heads != 0` → 报错（:115-116、:155-156）。

### 3. 请求从 CSV 进来

**读队列 `load_request_queue`（`W/generate_trace.py:146-235`）逐行查四样：**
- request_id 以 `#` 开头 → 注释行，跳过（:167）。
- turn-0 的行必须带 `session_arrival_time_ns`、不写 interval；turn>0 的行反过来（:176-205）。首轮什么时候来、后续隔多久来，两拨字段不能混。
- 时间必须是 1000 ns 的整数倍（:211-214）。
- 一条请求都没有 → 报错（:233-234）。

**挑前 N 个会话 `select_first_session_requests`（`W/generate_face_trace.py:202-229`）三分支：**
- `request_queue_session_limit == 0` → 全要。
- 大于 0 → 取前 N 个不同 session 的所有行。
- N 比现有 session 数还多 → 报错。

**manifest 字段推导（`W/plan_materializer.py:265-327`）：**
- turn-0：history = 0，prefill 上下文 = prefill_length（第一轮没有历史）。
- turn>0：先找同 session 上一 turn 的 final；找不到（session 断了或 turn 不挨着）→ RuntimeError，绝不偷偷当 0 处理——当成 0 整条链的长度全错。找到则 history = 上一条 final，context = history + prefill，final = context + decode。
- 旁边有 sidecar 文件 → 附带四个字段（human/tool 耗时、request_type、prefix_len）；没有就不写，用的人自己按缺省处理（:311-324）。
- 队列文件命名不合规矩但同目录有 sidecar → 只打 WARN，不改退出码（plan_materializer.py:243-260）。

**运行期构造请求对象时查两个数（`O/face_online_scheduler.py:199-218`）：**
- `prefill_length <= 0` → 报错，零活的请求会永远堵在队列里。
- `decode_length <= 0` → 报错，同样的道理。

**策略开关，写错直接报错：**
- `kv_cache_policy` 只认 `session_lru_recompute`，三个地方各查一遍：配置层（`W/generate_face_trace.py:161-167`）、装配层（`O/online_service.py:186-210`）、调度器构造（`O/face_online_scheduler.py:321-325`）。
- `trace_granularity` 在线只认 `request_aggregated`，四个发射入口都查（`O/graph_batch_builder.py:358-361、415-418、458-461、497-500`）。
- `prefill_chunk_size <= 0` → 构造时报错（face_online_scheduler.py:343-346）。

**C++ 侧请求入口（`astra-sim/workload/execution_driven/RequestIngress.cc`）：**
- 命令队列满了 → 拒收，生产方等着重试（:40-47）。
- Submit 的到达时刻 <= 当前时刻 → 拉到 current+1，并按来源分开计数：StaticCsv 的 t=0 边界情况 / 被背压拖过的 / 真迟的，ExternalStream 单独一列（:81-118）。
- 关闭输入、文件读完、出错三种结束命令分开处理；出错 → `std::abort`（:145-169）。
- 预登记的未来到达：id 为空、queue_index 非法、同批重复、已登记未到、queue_index 和登记对不上 → 拒收（:210-254）。
- 到点回调 → 包成 ARRIVAL 事件送进决策邮箱（:340-374）。

### 4. 请求到了，先选哪个实例做 prefill

`_on_arrival`（`O/face_online_scheduler.py:1038-1063`）四步：

1. 给全部实例拍一份队列现状（`_queue_snapshots` :1616-1628）：每个实例队列里还剩多少 chunk、最近一次到达是什么时候。
2. `select_prefill_instance`（`W/face_scheduler.py:326-329`）：按（剩余 chunk 最少 → 最近到达最早 → 实例号最小）挑。三个条件依次比，结果确定。一个实例都没有 → 报错。
3. 挑完做一次容量预检 `_filter_prefill_selection`（face_online_scheduler.py:1731-1784），三种结局：
   - 选中的实例每张卡的剩余字节都够装这条请求的整个上下文 → 维持原选，日志里什么都不多记。
   - 原选不够、别的实例有够的 → 在够的里面按同样的排序键重挑一个，记一条 `prefill_admission_probe` 日志（outcome = `reselected_infeasible_first`）。
   - 全都不够 → 维持原选，堵着的事交给后面准入阶段重试（outcome = `kept_all_infeasible`）。

   预检只读内存数字，不赶人也不占位（`_probe_instance_capacity` :1632-1650）。它算得保守——不把"可以赶走别人腾出的空间"算进来，所以偶尔会把其实能装下的判成装不下；误判没关系，后面准入会真刀真枪再试一次。
4. 请求进选中国实例的队列，更新该实例的剩余 chunk 数和最近到达时刻，实例闲着就放进待发集合。

### 5. prefill 做完，选哪个实例做 decode

`_on_prefill_drain`（face_online_scheduler.py:1065-1128）：

**先做两件杂活：**
- drain 的请求不在它实例的队列里 → 报错（:1079-1080）；正常则出队、扣实例的聚合 chunk 数（:1081-1083）。
- 实例还有排队工作 → 留在待发集合，否则移出（:1084-1087）。

**然后拍全局现状（:1089-1093）**：每个实例有没有 prefill 活、每个实例正在 decode 的成员各自的上下文多长。注意顺序敏感：这些数值反映"这条请求出队之后"的状态。

**`select_decode_instance`（`W/face_scheduler.py:410-494`）四个分支：**
- 候选集 = 从 prefill 实例出发、距离不超过 `d2d带宽/本地HBM带宽` 这个比例的所有实例（`schedulable_instances` :374-380）；候选集空 → 报错（:434-435）。这个比例的含义：跨实例搬 KV 太慢的话，宁可不选那么远的。
- 每个候选算一笔账："把新请求塞进这个实例，每次迭代慢多少纳秒"，再除以实例卡数得到单人成本。其中 prefill 负担只在"该实例现在有 prefill 活"时才算（:441）；decode 的 token 长度取 max(现有最长, 新请求上下文)（:458）。
- 单人成本精确打平的候选有 2 个以上、且传了轮转计数器 → 按计数器轮流挑，挑完计数器 +1（:489-491）。
- 没打平（或没传计数器）→ 挑实例号最小的（:492-493）。

**挑完再做 decode 侧容量预检 `_filter_decode_selection`（face_online_scheduler.py:1652-1729）：**
- 需求怎么算：目标是长到 final 的字节数；会话 KV 已经驻在这个候选上的只算增量（源目标同一个实例），不在的算全量——因为源实例那份还占着源（:1683-1690）。
- 增量算出负数 → 说明会话驻留的字节已经比 final 需求还多，manifest 自己跟自己矛盾才可能；这个候选直接按不可行剔除，不让它进真实路径去崩（:1691-1697）。
- 三种结局和 prefill 侧一样：原选可行 → 维持；可行集非空 → 按（成本, 实例号）挑最小重选；全不可行 → 维持原选、走阻塞重试（:1700-1729）。

**最后落定（:1112-1128）**：decode 实例一旦定下就不换（"绝不 remap"，内存不够宁可等也不改派）。请求进 `waiting_decode_admissions[选中实例]` 队列、打上"要重试"记号、记账，然后立刻试一次准入。

### 6. 准入：真去占内存

**prefill 准入 `_try_admit_prefill`（face_online_scheduler.py:1280-1368）六个分支：**

| # | 什么情况 | 结果 | 位置 |
|---|---|---|---|
| A1 | 已经准入过了 | 直接返回 True | :1285-1286 |
| A2 | prefill 实例号丢了（None） | 报错 | :1287-1288 |
| A3 | 上次尝试时的容量版本号 == 现在的版本号，且实例没打"要重试"记号 | 返回 False，这次不试了——防止每个决策点都白试一遍造成重试风暴 | :1293-1296 |
| A4 | 过了上一条的检查 | 记下本次版本号，清掉重试记号 | :1297-1301 |
| A5 | `prepare_history` 说放不下 | 它顺带赶了人的话，把相关实例的容量版本号 +1（别的请求因此获得重试资格）；再给本实例打"要重试"记号，保证下一个决策点必再试；返回 False | :1323-1336 |
| A6 | 成功 | 记下动作、来源实例、要搬的字节、要重算的 token；接着 `grow_prefill` 把 KV 长到 prefill 上下文大小，grow 失败 → 报错（预约丢了不该发生）；重算 chunk 数 = ceil(重算/chunk) + ceil(prefill/chunk)；标记已准入；相关实例容量版本号 +1（本实例 + 搬出实例 + 被赶人的实例） | :1337-1367 |

被阻塞期间的赶人记录只追加、不覆盖（:1321-1322）：赶人已经真发生了（字节真扣了），重试成功后日志必须还留着这些记录，不然回放工具对不上账。

**decode 准入 `_try_admit_waiting_decodes`（face_online_scheduler.py:1370-1508）五个分支：**

| # | 什么情况 | 结果 | 位置 |
|---|---|---|---|
| B1 | 扫哪些实例：等待队列非空，且（打了重试记号，或上次扫描后容量版本号变了） | 进本轮扫描；扫描开始时同步版本号、清记号 | :1375-1389 |
| B2 | 队首请求已经不在等待状态 | 跳过它 | :1394-1395 |
| B3 | 队首的 decode 目标不是本队列 | 报错，队列坏了 | :1396-1398 |
| B4 | `move_prefill_to_decode` 说放不下 | 赶了人就推容量版本号；打重试记号；请求排回队尾接着等，继续处理队列里的下一个 | :1409-1428 |
| B5 | 搬成功 | `grow_decode` 长到 final 大小，失败 → 报错；请求进 `pending_decode_ready`（等下一趟列车把它收进 decode 批）；实例闲着就放进待发集合；写 assignment（prefill/decode 两个实例号都必须非负，C++ 提交前查）；写 decode 决策日志（候选成本全记、迁移字节、NoC 跳数、赶人快照——写完立刻清空，防止以后被再读一遍造成双计）；记账升级为 active_decode；容量版本号 +1 | :1429-1508 |

容量版本号 `_note_capacity_change`（:1786-1790）：给涉及的每个实例 +1。谁会让版本号变：被赶走人的实例、KV 搬出的源实例、终态释放的实例。变了之后，压在 A3/B1 那两道检查后面的请求就获得重试资格。

### 7. 列车：实例上怎么排活

代码把一个实例连着跑的一段迭代叫一趟 train（列车，这是代码里的名字，`train_id`）。一趟车里每个迭代最多带 1 个队头 prefill chunk，外加全部 decode 成员各出 1 个 token。批成员只在车与车的边界变化，车一旦发出成员就冻结。

**排车 `_plan_train`（face_online_scheduler.py:630-720）九个分支：**

| # | 什么情况 | 结果 | 位置 |
|---|---|---|---|
| C1 | decode 成员剩余 token <= 0 | 报错 | :641-647 |
| C2 | 没队头也没成员 | 返回 None，这实例没活 | :650-652 |
| C3 | 没队头（纯 decode 车） | 迭代数 = 成员里最大的剩余 token | :653-654 |
| C4 | 有队头 | 迭代数 = 队头剩余 chunk 数；<= 0 → 报错 | :656-660 |
| C5 | 迭代数超过上限（`SH_TRAIN_MAX_ITER`，默认 8，0 = 不限） | 截到上限，标记 capped | :663-665 |
| C6 | 某成员这趟车里能跑完剩余 token | 列入 exit（车没停它先下车） | :676-683 |
| C7 | 没截断且有队头 | 队头列入 drain（这趟车跑完它的 prefill 就齐了） | :687-688 |
| C8 | 截断了、又没 drain 又没 exit | 没有自然完成信号，把列车自己的 id 当完成信号（哨兵） | :693-695 |
| C9 | 队头一个 chunk 都没跑过 | 标记 head_first_chunk，统计首 token 用 | :689-690 |

**队头的 chunk 顺序 `_plan_head_chunks`（:722-748）**：先把欠的历史算完（recompute 段，上下文从 0 往上涨），再算本轮新输入（在历史长度上往后接）。排到头还有要求 → 报错（:741-744）。

**发车循环 `_plan_and_emit_trains`（:750-808）六个分支：**
- 有首步余量批没发的实例 → 先把余量批发出去（:770-772）。
- 按实例号从小到大过待发集合；实例还有在飞的列车 → 跳过，一个实例同时只有一趟车（:776-777）。
- 队头剩余 chunk 为 0（它的 drain 决策还在路上）→ 不给它排活，换队列里下一个有活的当队头（:779-786）。
- 队头还没准入：试准入，成功 → 发准入动作、把它的 chunk 并进这趟车；失败 → 这趟车退化成纯 decode（:787-791）。
- `pending_decode_ready` 里有成员 → 全部转正为 active_decode，随这趟车一起发（:793-801）。
- 计划是 None 且三个队列全空 → 从待发集合移除（:803-807）。

**首 token 拆分 `_first_token_plan`（:908-945）四个分支：**
- 开关 `SH_FIRST_TOKEN_SPLIT` 没开（默认）→ None，行为和没有这功能时一模一样。
- 车上没有头一次参加 decode 的成员 → None。
- 迭代数 >= 2 → 物理拆成两批：首步批 + 余量批，首 token 时刻看得更准。
- 迭代数是 1 → 不拆车，只加标记。

**发车 `_emit_train`（:810-884）三个分支：** 车上有 decode 成员或新成员 → stage 记 decode，纯 prefill 车记 prefill（:821）；要拆 → 走首步发射口然后返回（:843-853）；整列发 → 注册 drain/exit/哨兵的 watch（完成或被跳过都算数）、挂起在飞列车、移出待发集合、写列车日志行。

**核销 `_finalize_completed_trains`（:544-628）五个分支：**
- 哨兵列车号查不到归属实例 → 报错（:563-566）。
- 在飞列车收到第一个信号 → 一次核销：每个成员按参与次数推进 token 计数；exit 成员移出 decode 批，不在批里 → 报错（:583-588）；队头 chunk 推进、两边剩余数各减；清掉在飞列车；同车后到的信号挂起待收（:568-607）。
- 后到的信号 → 对着已核销列车的挂起清单销账，清完就删记录（:608-615）。
- 信号对不上任何在飞或刚核销的车 → 报错，陈旧完成错配（:616-622）。
- 核销完看实例：还有活 → 留在待发集合，否则移出（:624-628）。

### 8. 决策变成图节点

**准入动作 `_emit_admission_actions`（`O/graph_batch_builder.py:736-818`）五个分支：**

| # | 什么情况 | 结果 | 位置 |
|---|---|---|---|
| E1 | turn = 0 | 给每张 prefill 卡发一个到达定时器；到达时刻缺失 → 报错；等待本身由 C++ 的 alarm 做，这里不重复等 | :750-762 |
| E2 | turn > 0 | 取走并删掉该会话的完成门（上一轮 exit 留下的 end barrier）；没有门或没配 interval → 报错；发 interval 定时器（间隔为 0 就不建节点，只连依赖边） | :763-781 |
| E3 | 历史动作 = NOC_MIGRATE | 源实例号缺失 → 报错；源实例和上一轮完成门不在一个组 → 先发 1 字节控制消息把接力棒送过去；然后发 1000 类的 KV 成对传输 | :782-804 |
| E4 | 其他历史动作（NO_HISTORY / LOCAL_HIT / RECOMPUTE） | 只发一条 interval 控制消息，不搬字节 | :805-811 |
| E5 | 收尾 | prefill 组做一个 all_reduce 屏障：历史就位，全组可以开算 | :812-815 |

**成对传输 `_paired_transfer`（`W/generate_face_trace.py:419-486`）五个分支：**
- 两边 TP 度不等 → 报错（:431-432）。
- 源组名 == 目标组名 → 一个节点都不发，只挂定时器（:433-437）。同实例"迁移"是零开销的。
- 总字节能整除头数 → 按整头分片；不能 → 按字节精确分，不向上取整（:442-453）。
- 每个分片发一对 comm_send / comm_recv，tag = `queue_index*10000 + 类别 + 相对秩`，超过 31 位 → 报错（`_stage_tag` :389-403）。
- 路由固定 XY：先横着走完再竖着走（`_xy_route` :406-416）。

**定时器节点 `timer_gate`（`O/graph_batch_builder.py:204-250`）两个分支：** 时长 0 → 不发节点，只返回依赖锚点；不是 1000 的倍数或为负 → 报错。

**NoC 跳数统计 `_kv_noc_hops`（face_online_scheduler.py:1570-1594）三个分支：** 没有源/目标或同实例 → 空表；TP 度不等 → 报错；否则逐分片给跳数。纯观测，不进任何决策。

**列车的三段发射（graph_batch_builder.py）：** 车头把新 decode 成员的跨实例迁移发成 3000 类传输（同实例零节点）加起始标记（:513-557）；车身每张卡一个聚合算子节点，权重每迭代只读一次，迭代数就是权重遍数（:559-588）；车尾：drain / exit 标记每成员每卡一个节点（watch 就挂这些节点上）、decode_length == 1 的成员其 exit 节点名带上 first_token 字样（一个节点当两个统计锚点用，:648-662）、哨兵标记只放 prefill 段的单事件通道（放 decode 段会把请求完成记重，:664-680）、全车共用一个 end barrier，exit 成员的 barrier 存进 `completion_gates` 给下一轮同会话的 E2 用（:682-710）。

### 9. Python 和 C++ 之间的规矩

- **交付序号门 `_gate_delivery_sequence`（`O/online_scheduler_base.py:487-516`）三种情况：** 序号 == 上次 → 当重放处理，内容必须和缓存的一模一样（不一样 → 报错），一样就把上次的批次原样回一份，状态零改动；序号 != 上次+1 → 报错，跳号和旧重复都不允许；正好 +1 → 正常往下走。
- **完成核销 `_settle_completions`（:862-905）四种情况：** 请求号 `batch_train_` 开头 → 交给列车核销逻辑；不认识的请求 → 报错；stage 是 REQUEST_COMPLETE → 整条核销并记账；这个 stage 已经核销过 → 报错。同一批里 DECODE_COMPLETION 必须排在 REQUEST_COMPLETE 前面，反着来会被"未知请求"检查拦下。
- **到达处理 `_process_arrivals`（:907-933）三种情况：** 请求号撞 `batch_train_` 保留前缀 → 报错（防止哨兵冒充真请求）；不认识或重复到达 → 报错；正常 → 登记在途。
- **提交回执 `on_commit_ack`（:992-1066）：** 不是 dict、schema 不对、序号非法、批号和序号对不上、success 为假、序号超前、没有在途记录 → 各自报错；序号已被处理过 → 幂等跳过；合法 → 把临时 KV 记账转成正式的、推进已确认水位。
- **文件桥 `O/decision_bridge.py` 四种情况：** 旧序号重复 → 当没看见（两边都这么承诺）；未来序号 → 报错退出（`_BoundedSequenceTracker` :101-125）；回调抛异常 → 写错误回应、非 0 退出（:403-429）；回应管道 BrokenPipe → C++ 进程死了，报错退出（:502-508）。
- **在途数量上限（`PropagatingTailTracker`，online_scheduler_base.py:57-126）：** 在途交付和临时 KV 记账默认最多各 8 个（`SH_PROPAGATING_TAIL_LIMIT` 可改，写非法值构造时报错），在途到达最多等于请求总数；超了 → 报错，绝不默默丢掉真实的在途工作。
- **C++ 侧两件事：** prefill 段的 watch 触发 → 只报 PREFILL_DRAIN；decode 段触发 → 同时报 DECODE_COMPLETION 和 REQUEST_COMPLETE（`astra-sim/network_frontend/analytical/congestion_aware/main_online.cc:1332-1364`）。提交器检查 assignment：request_id 为空或任一实例号为负 → 拒收整批（GraphBatchCommitter.cc）。
- **收尾审计 `verify_run_end`（face_online_scheduler.py:1836-1895）：** 完成数不够、还有没释放的请求、实例没闲下来（队列/decode 批/在飞列车任一非空）、prefill 聚合不是 0、首步余量没发、唤醒没收、等待 decode 准入的队列没清空、待发集合没清空、完成门没退休 → 一律报错；最后让 KV 管理器做终态检查。

---

## 二、历史 KV 的管理：按一辈子的顺序讲

每个 session 在 KV 管理器里有一条记录，三种情况：**没有记录**（第一次来）、**RESIDENT**（KV 实打实占着某个实例的字节，还带一个 active 位表示有没有请求正在用）、**EVICTED**（字节没了，但逻辑长度还记着，下次来得重算）。会话彻底结束时记录整个删掉。代码里只有 `session_lru_recompute` 这一种策略：请求完成时 KV 不删留着；地方不够了按"最久没完成"的顺序赶人；赶过的下次重算；跨实例就搬。

### 1. 建档和认亲：prepare_history

`prepare_history`（`W/session_kv_manager.py:921-1141`）是请求要用历史时第一个调的函数。先过五道检查，任何一道不过直接报错：

| # | 什么情况 | 位置 |
|---|---|---|
| H1 | token 数为负，或需求的上下文比历史还短 | :933-938 |
| H2 | 没见过的会话却带着历史 | :940-941 |
| H3 | 会话记的逻辑长度和请求报的历史对不上 | :942-946 |
| H4 | 会话还 active（上一条请求没算完）——同一个会话必须一条一条来 | :947-948 |
| H5 | 目标实例上"想长到的字节 − 已经在这的字节"出现负数，KV 只能涨不能缩 | :950-960 |

然后 `ensure_physical_fit` 腾地方。**腾不出时（admission_blocked）**，返回的动作标签和成功路径一一对应：没记录 → NO_HISTORY；EVICTED → RECOMPUTE（重算量 = 全部历史）；RESIDENT 且就在目标实例 → LOCAL_HIT；RESIDENT 在别的实例 → NOC_MIGRATE。不搬任何字节，附上哪些卡不够（:972-995）。

**腾得出时，四个动作走其一：**

| 动作 | 什么时候走 | 干了什么 | 位置 |
|---|---|---|---|
| NO_HISTORY | 没有记录 | 新建：RESIDENT、0 字节、active、归属目标实例 | :998-1030 |
| RECOMPUTE | 记录是 EVICTED | 把历史字节加到目标实例，转回 RESIDENT、active；返回重算量 = 全部历史 token（没有只算一半的说法） | :1032-1065 |
| LOCAL_HIT | RESIDENT 且就在目标实例 | 一个字节不动，置 active。这就是"接得上，不用重算" | :1067-1099 |
| NOC_MIGRATE | RESIDENT 但在别的实例 | 先给目标实例加字节、再从源实例扣（先确认装得下再腾地方），会话归属改成目标实例，逐分片给出迁移描述 | :1101-1141 |
| （防御） | RESIDENT 却没有实例号 | 报错 | :1067-1068 |

### 2. 腾地方：ensure_physical_fit

`ensure_physical_fit`（session_kv_manager.py:803-916）是唯一的赶人引擎，八个分支：

| # | 什么情况 | 结果 | 位置 |
|---|---|---|---|
| F1 | 需求向量长度和 TP 度对不上，或有负数 | 报错 | :816-817 |
| F2 | 需求 > 卡容量 − 权重（把实例清空都装不下） | 记一条 deep_gap 观测，然后报错——这种是结构问题，重试也没用 | :818-843 |
| F3 | 现在就装得下 | 直接过，不赶任何人 | :844-853 |
| F4 | 有缺口 | 拍一份候选名单；循环里不再重新排 | :846-853 |
| F5 | 名单用完缺口还在 | 记 deep_gap + 一条去重后的压力事件，返回"没成"——这次先不跑，下个决策点再试 | :856-880 |
| F6 | 轮到某个受害者时它状态变了（不在了 / 搬走了 / active 了 / 没有完成时刻） | 报错——名单是之前拍的，中间状态被改过说明有 bug | :883-895 |
| F7 | 正常赶一个 | 扣字节、转 EVICTED、发事件；重算缺口，够了呢就停，不够继续下一个 | :900-909 |
| F8 | 循环出来了缺口还在 | 报错（防御） | :910-915 |

**谁能被赶（候选资格，`_candidate_sessions` :463-479）：** 必须同时满足 RESIDENT、驻在这个实例上、不 active、有完成时刻、不是当前正要进的会话。排序按（完成时刻最早, 会话号）——最久没用的先走，打平按会话号。会话号入口统一转字符串（:46-56），免得数字和字符串混着排序直接崩。

**压力事件去重 `_pressure_event`（:522-565）：** 同一个（事件类型, 阶段, 请求, 实例）第一次出现 → 记原事件；之后再出现 → 只补一条 admission_retry；请求完成时这些键一次清光（:573-575、:1380）。不然一个堵住的请求每个决策点都刷一条一样的事件。

### 3. 变长：grow

- `_grow`（:1143-1178）：会话必须 active、RESIDENT、就在这个实例上，否则报错；目标长度比现有短 → 报错（KV 不缩）；增量交给 `ensure_physical_fit` 腾；不行 → 返回"没成"（prefill 段调到这里失败会报错——前面刚预约成功不该失败；decode 段在上游就拦住了）；行 → 加字节、更新长度。
- `grow_prefill`（:1180-1199）：长到 prefill 上下文大小；实例号缺失 → 报错。
- `grow_decode`（:1317-1336）：长到 final 上下文大小；同样的检查。

### 4. prefill 搬去 decode 实例：move_prefill_to_decode

`move_prefill_to_decode`（:1201-1315）四个分支：

- 前置：必须 active、RESIDENT、有实例号，否则报错（:1212-1213）。
- 目标实例装不下 → 返回阻塞，动作标签：同实例标 LOCAL_HIT、跨实例标 NOC_MIGRATE（:1244-1252）。
- 同实例 → LOCAL_HIT：零字节，发一条复用事件，decode 接着 prefill 的 KV 继续算（:1253-1271）。
- 跨实例 → NOC_MIGRATE：先给目标加字节、再从源扣，逐分片给出迁移描述（这份描述随后被发成 3000 类传输）（:1273-1315）。

### 5. 请求完成：mark_complete

`mark_complete`（:1338-1381）：必须 active 且 RESIDENT，否则报错（:1356-1357）。置 active = False、记完成时刻（这个时刻就是将来被赶的排队依据）。发 `retain_complete` 事件——**完成时不删 KV**，留给下一轮用（:1363-1377）。顺带把这个请求的压力去重键全清了（:1380）。主动删 KV 的代码路径已经删掉，这个函数永远返回空。

### 6. 被赶走：_delete

`_delete`（:752-801）：受害者必须"已完成 + 不 active + RESIDENT + 有完成时刻"，否则报错（:761-767）。逐卡扣字节，扣的比持有的多 → 报错（:620-628）。状态转 EVICTED、实例号清空、发 `evict_delete` 事件。注意：这是个纯记账动作，图上没有对应节点——字节本来就在本地卡上，删了就删了。

### 7. 会话彻底结束：retire_terminal_session

`retire_terminal_session`（:1383-1428），调度器确认这个会话没有下一轮了才调：没这个会话 → KeyError；还 active 或没完成过 → 报错；传来的请求号和会话最后的主人不一样 → 报错；找不到终结请求号 → 报错（:1397-1415）；RESIDENT → 释放全部字节、返回实例号（调用方拿去推容量版本号，别的请求因此能重试）；EVICTED → 字节早没了，什么都不动；两种之外的状态 → 报错（:1417-1423）；最后删掉整条记录。

### 8. 恢复：被赶走的 KV 怎么办

规划侧的恢复就是 prepare_history 的 RECOMPUTE 分支：逻辑长度一直记在案，重算量 = 全部历史 token，排成"recompute 段 + 当前段"的 chunk 序在 prefill 列车里跑（face_online_scheduler.py:722-748）。

观测侧的恢复是 `S/hbm_watermark.py` 的 `apply_restore`（:1370-1475），重放日志时对账，六个分支：动作是 RECOMPUTE、或恢复前状态是 EVICTED、而跟踪里居然还有字节 → 判定为没记上的静默逐出，对账扣掉（:1395-1406）；LOCAL_HIT → 零字节，源和目标不同就跟着搬（:1411-1422）；0 字节恢复 → 同样跟着搬（:1436-1444）；没有源实例 → 只往目标加（:1445-1455）；源 == 目标 → 只加（:1456-1463）；跨实例 → 源扣目标加，扣超了 → fail（:1464-1475）；字节数和 token 系数算出来的对不上 → 记一条异常计数（:1379-1387）。

### 9. 事件、水位、自检

- 每条事件拿一个只增不减的编号（`_event` :481-520），和它在列表里的位置无关。
- 事件列表会瘦身：消费方拿走了 8192 条以上、且超过现存一半 → 把已消费的前缀整段删掉，成本摊到每次一条（:411-422）。
- deep_gap 计数（:424-427、:577-611）：两个失败出口（空实例装不下 / 候选用完）前各记一条。正常负载应该是 0，出现大于 0 就是有人碰到了安全边界，值得看一眼。
- 增量对账（:630-749）：每次改动后只重算受影响的会话和卡。出现负数、超卡容量、会话和卡的账对不上、RESIDENT 没实例、EVICTED 还挂着实例 → 各自报错。`SH_STRICT_KV_INVARIANTS=1` 时每次改动都做全量检查（:30-38、:1438-1473）。
- 终态检查 `assert_final_state`（:1430-1436）：跑完还有压力去重键没清 → 报错；还有 active 会话 → 报错。

### 10. 跑完之后回看：观测层

- **命中口径（`S/kv_cache_adapter.py:120-133`）：** NO_HISTORY → no_history（不算 miss）；LOCAL_HIT → full 命中；NOC_MIGRATE → 也算 full（字节都在，只是搬了家）；RECOMPUTE → miss；不认识的动作 → not_supported。决策行按 kind 分家（:86-98）：prefill / decode / completion 是正账；两个 admission_probe 只做观测不计数；不认识的 kind → 报错。
- **事件提取 `extract_events_face_wscllm`（:313-371）：** prefill 行的历史迁移字节 > 0 → 记一条 history_transfer；准入、decode 目标、完成三处的赶人名单逐条落事件（字节数按 shard → total → 求和的顺序回退，:271-310）；decode 行的迁移字节 > 0 → 记一条 prefill_decode_transfer。`--reconcile` 模式拿 native 日志对 canonical 逐族对账（:794-937）；分片加总 ≠ 总字节 → 默认警告加非零退出码，reconcile 下直接 fail（:781-790）。
- **HBM 水位的四档可信度（`S/hbm_watermark.py:148-151、2021-2178`）：** journal 和校验和四道检查全过 → 唯一能出正式结论的一档；往下依次是 resident 精确、生命周期重放精确、只有上界（journal 缺了，occupancy 永远标不可信）；校验和在而 journal 缺 → 退出码 2。
- **容量三个口径 `compute_capacity_calibers`（:656-733）：** 正式判决用"权重 + 驻留 + 预留 ≤ 容量"；硬上限按每张卡最多装多少 token 折算（0 头的分片不参与约束）；水位目标再扣一块预留，只报告不判决。
- **重放的主顺序（`WatermarkScan.consume` :1580-1663）：** 每条记录固定先逐出、再恢复/迁移、再增长。prefill 行缺实例号、请求缺 token 清单、时间倒退、同请求同 kind 出现两遍 → 全部 fail；跑完还要检查 token 清单全覆盖（:1665-1687）。
- **逐出 `apply_evict`（:1324-1350）：** 受害者不在跟踪里或扣超 → fail；字节比跟踪的少 → 记 partial_evictions；扣到 0 解除实例归属。
- **迁移 `apply_move`（:1477-1507）：** 同实例或 0 字节 → 记一条 stay；跨实例 → 源扣目标加。
- **增长 `apply_grow`（:1509-1522）：** 目标比跟踪的还小 → fail（负增长不可能）。
- 逐出时刻比实例最后变化点还早 → 拉到那个点再记，保证逐区间的序列单调（:1294-1304）。

---

## 三、两边怎么互相影响

映射的每一步都在看 KV 的状态：选 prefill 实例看的是"现在装不装得下"的保守估计；选 decode 实例算的增量需求，直接取决于"这个会话的 KV 现在驻在哪个实例、占多少字节"——驻在候选本地的只算增量，驻在外地的要按全量算。反过来，每次映射决定都在改写 KV：NO_HISTORY 建档、RECOMPUTE 重建、NOC_MIGRATE 搬家、grow 沿着选定的实例长字节。

装不下的时候映射不会被推翻重选——实例定了就不换，请求卡在准入这一步等。等的是什么：等 `mark_complete` 解除别人的 active（完成不删 KV，但让这个会话变成可被赶的候选）、等 `retire_terminal_session` 释放终态会话的字节、或等 LRU 赶人腾地方。这三件事都会推容量版本号，版本号一变，被 A3 / B1 两道检查压着的请求就拿到重试资格。所以一句话：KV 在哪、还剩多少，决定请求能派去哪、能不能开工；请求怎么派、什么时候完成，又决定哪块内存被占、谁被赶走、谁等到重试的机会。两边咬合在一起转。
