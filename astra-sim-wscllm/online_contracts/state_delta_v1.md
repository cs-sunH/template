# StateDelta schema v1(冻结)

> 状态: **冻结**(阶段 4 §7.1)。本文是 C++ `DecisionMailbox.hh` /
> `DecisionBridge.cc` 序列化与 Python `online_scheduler_base.py` 校验器的
> 唯一权威。v0 的字段与语义全部保留(v0 文档见阶段 1/3 的桥接协议注释),
> v1 只做增补与正式化,不删改任何既有字段的含义。
>
> 生效点: `kDecisionBridgeSchemaVersion`(C++)与 `SCHEMA_VERSION`
> (Python `decision_bridge.py` / `online_scheduler_base.py`)同时从 0 升为 1,
> 两处校验器都在不匹配时 fail-closed(C++ 桥接层 abort / Python 写 error
> response 后非 0 退出)。

## 1. 范围与角色

StateDelta 是 C++ 引擎在**一个决策边界**(tick-end gate 的至多一次 drain)
向 Python 决策服务交付的完整载荷 JSON。Python 的 GraphBatch 响应是反方向
的载荷;commit ack 是第三个方向(§7.2)。三者版本号一致、同批同 seq。

## 2. 冻结字段表

```
{
  "schema_version": 1,            // 冻结值 1
  "delivery_sequence": uint64,    // 单调递增的交付纪元序号,>= 0(0 = 首纪元)
  "delivery_epoch": uint64,       // v1 恒等于 delivery_sequence(见 §3.2)
  "tick": uint64,                 // EventQueue 全局 EventTime(纳秒)
  "deferred_from_tick": uint64,   // v0 字段:显式 T->T+1 延后记录,0=同 tick
  "reasons": ["ARRIVAL"|"PREFILL_DRAIN"|"DECODE_COMPLETION"|"REQUEST_COMPLETE", ...],
  "arrivals": [                   // reasons 中每个 ARRIVAL 一条,与 reasons 对齐
    {
      "request_id", "session_id", "turn_index",
      "prefill_length", "decode_length",
      "inter_request_interval_ns", "arrival_world_ns",
      "ingress_seq": uint64,      // v1 新增:该 request 的入站序号(全局单调)
      "queue_index": int64        // v1 新增:冻结队列序;未知=-1
    }, ...
  ],
  "completed_groups": [           // 每个 PREFILL_DRAIN/DECODE_COMPLETION/
                                  // REQUEST_COMPLETE 一条,与 reasons 对齐
    {"request_id", "stage", "generation", "node_count"}, ...
  ],
  "completed_nodes": [            // v1 新增:本交付纪元内全部节点终态事实
    {"rank", "node_id", "request_id", "stage", "generation",
     "tick", "terminal_status"},  // terminal_status: 0=Success, 1=Skipped
    ...
  ],
  "retry_items": [],              // v1 恒为空数组(占位;legacy 迁移前无生产者)
  "affected_ranks": [int, ...],   // v1 新增:本纪元受影响 rank 集(完成组
                                  // 成员 rank 的并集;v1 无快照驱动来源)
  "snapshot_handle": {            // v1 占位,阶段 7 才接拥塞快照(§6 过期规则)
    "epoch": 0, "tick": 0, "kind": ""   // v1 恒为自一致占位(见 §4)
  },
  "ledger_summary": {"injected_unfinished": [...]}   // 阶段 3 字段,原样保留
}
```

## 3. 时间、序号与去重(时间合同,contract ③)

### 3.1 tick = EventQueue 全局 EventTime

`tick` 是 C++ EventQueue 的全局事件时间(`get_current_time()`,纳秒),是决策
的唯一时间口径,与阶段 1 步骤 1-2 冻结的时间合同一致。一个 tick 至多产生
一次 delivery(tick-end gate 每 tick 至多 drain 一次;**单 tick 单次交付**);
两个不同 delivery 的 tick 严格递增(EventQueue 严格递增规则)。

### 3.2 delivery_sequence 单调 + delivery_epoch

- `delivery_sequence` 从 0 起严格单调递增,每一交付纪元恰好一次(C++ 侧
  `OnlineDriverContext::delivery_seq` 持有),同时是 request/response/
  commit_ack 三类桥接文件的共享文件名序号。0 = 首个 tick-end 交付纪元
  (运行开始即交付,无先行纪元;阶段 3 起存档的桥接序列即为 0 起,本合同
  2026-08-15 修正记录:原稿"1 起"与实现不符,以 0 起为准)。
- `delivery_epoch` 是"交付纪元"的独立记账号。v1 中二者**恒等**:
  `delivery_epoch == delivery_sequence`。字段独立存在是为将来 deferred
  交付纪元(T+1 唤醒在 T 与 T+1 之间交错)准备的显式占位——一旦将来某 epoch
  由多个 drain 片段组成,epoch 仍唯一、sequence 逐个片段递增;v1 禁止二者
  分离,Python 校验器断言相等,不等即 abort。

### 3.3 同 tick 去重 + 确定性排序

- **同 tick 同 identity 去重**(mailbox 内建,阶段 1 冻结):identity =
  (reason, request_id, stage, generation);同一交付纪元内重复 push 是 no-op,
  跨纪元同名身份合法。
- **确定性排序**:`reasons[]`/`arrivals[]`/`completed_groups[]` 三条数组
  按事件在 drain 中的冻结顺序对齐,同一输入(同一 CSV 前 30s)的两次运行
  逐字节一致。
- **arrivals 的冻结队列序**(v1 正式化):ARRIVAL 事件在
  `arrivals[]` 内按 `queue_index` 升序稳定排列(冻结队列序)。
  经验事实(2026-08-15 实测,写入合同):20.csv 前 30s 输入中,离线 LUT
  决策日志的 seq 序 == 队列序,覆盖全部 13 个同 tick 多到达组(strategy
  模式零同 tick 多到达组);因此该排序与阶段 1 冻结的离线同 tick
  `(priority, seq)` 批序逐字节等价,不改变任何既有决策顺序。

## 4. snapshot_handle(占位 + 过期规则,现在写死)

v1 的 `snapshot_handle` 是自一致占位,阶段 7 才接拥塞快照:

- v1 恒为 `{"epoch": <本 delivery_sequence>, "tick": <本 tick>,
  "kind": ""}`,即"指向本次交付所在时刻的空快照",Python 侧只做结构校验。
- **过期规则(冻结,现在写死)**:快照句柄只在**创建它的同一 delivery
  epoch、同一 tick** 内有效;任何跨 tick、跨 epoch 的使用(把某批的
  snapshot_handle 当作另一批的输入,或把过期句柄传给快照查询接口)一律
  fail-closed(abort / 报错退出)。校验器断言
  `snapshot_handle.epoch == delivery_sequence && snapshot_handle.tick == tick`,
  不等即 abort——即使将来阶段 7 引入真实快照,这条过期规则也不变。

## 5. 新字段语义

### 5.1 arrivals[]: ingress_seq 与 queue_index

- `ingress_seq`:该 request 在 RequestIngress 的入站序号(全局单调,从 0 起;
  提交命令与未来到达排程共用同一计数器)。同一 request 的多次到达(多 turn)
  有不同 ingress_seq。幂等/审计可按它识别"同一次到达"。
- `queue_index`:冻结队列序(CSV 数据行序,0 起;manifest.json 中每 request
  的 `queue_index` 同源)。C++ 侧经 `RequestIngress` 的 request->queue_index
  映射填到 ARRIVAL payload;loader 对所有数据行(turn-0 与 turn>0)都登记该
  映射,未来到达排程(`schedule_future_arrival`)同样携带。未知 = -1(防御性,
  正常路径不出现)。离线 `wsc_llm_scheduler.py` 事件循环的同 tick 到达批
  排序(priority=1, seq)与 queue_index 序等价(§3.3 经验事实)。

### 5.2 completed_nodes[]: 逐节点终态事实

每个在**本交付纪元内**完成(或跳过)的图节点一条:

```
{"rank": int, "node_id": uint64, "request_id": str, "stage": str,
 "generation": uint64, "tick": uint64, "terminal_status": 0|1}
```

- 数据源:CompletionObserver 在线 hook 的事实缓冲(每节点终态一条;
  C++ 驱动上下文持有,每次 delivery 清空)。
- 用途:审计/对账输入;本阶段不参与策略判据(策略输入不变,红线 §0.4)。
- `terminal_status` 与 `NodeTerminalStatus` 枚举一致:0=Success, 1=Skipped。
- 与 `completed_groups[]` 的关系:completed_groups 是聚合(一个 watch fire
  一条);completed_nodes 是事实(该 fire 前本纪元的每个节点终态)。
  v1 只保证"本纪元节点事实不丢、不重复",不保证 completed_nodes 与
  completed_groups 的成员级推导一致(推导留给对账脚本)。

### 5.3 affected_ranks[]: 受影响 rank 集

- v1 定义:本纪元所有 completed_groups 的成员 rank 并集(watch fire 时冻结
  的成员 rank 表,`WatchFire::member_ranks`),去重、升序。
- 数据源:C++ WatchRegistry 在 fire 时把 `StageWatch::expected_members` 的
  rank 表记入 WatchFire。
- 用途:为阶段 7 的快照/缓存失效范围预置精确集合;本阶段只做交付与校验
  (审计输入)。与 `snapshot_handle` 配合的规则见 §4。

### 5.4 retry_items[]: 恒空占位

- v1 恒为空数组。它是 admission retry(legacy 迁移)的预留通道;阶段 4 无
  生产者,Python 校验器断言空数组(非空即 abort,防半吊子)。
- Python 侧的对应物是 admission retry ready set(§7.3 事件索引队列),v1
  同样为空占位,无生产者。

## 6. 校验器(v1)契约

Python 侧(`online_scheduler_base.py::_validate_schema` 与
`decision_bridge.py::BridgeServer::_handle_request`)对每个请求强制:

1. `schema_version == 1`;字段闭集齐全(§2 全部字段),类型正确;
2. `delivery_epoch == delivery_sequence`;
3. `delivery_sequence` 严格递增(> 上一批;见 §7.2 的 gap 检查);
4. `reasons` 元素 ∈ 闭集;`arrivals` / `completed_groups` / `completed_nodes`
   与 `reasons` 对齐;`arrivals` 内 `queue_index` 非负且与冻结序一致
   (本批内按 queue_index 升序);
5. `retry_items == []`;
6. `snapshot_handle.epoch == delivery_sequence && snapshot_handle.tick == tick`
   (§4 过期规则的 v1 实例);
7. `affected_ranks` 升序、无重复、且 ⊆ 全图 rank 集。

任何一条不满足:Python 写 error response 并以非 0 退出;C++ 读到 error 即
abort(两侧 fail-closed,绝不静默降级)。

## 7. 配套契约(本阶段其余交付物在此引用,详见各自小节)

- §7.2 delivery sequence / ack / 幂等:Python 记录 `last_applied_sequence`;
  重复 delivery(seq 已应用)直接返回上次 batch 的 digest(幂等重放),不
  重新产生 assignment/KV action/图节点;seq != last+1(gap)即 fail-closed;
  commit ack 的 delivery_sequence 必须 <= last_applied_sequence(ack 校验
  fail-closed)。
- §7.3 事件索引队列:future arrival min-heap(键含 queue_index)、pending
  fence 索引、按 rank ready frontier、admission retry ready set(空占位);
  只消费 `tick <= current_tick` 的到期项;同 tick 冻结队列序稳定排序;
  完成事件直接定位受影响 request/stage;重复事件经 sequence/generation
  幂等;结束审计:mailbox / watch / heap / ready set 全部为空。

## 8. 冻结记录

- 冻结日期:2026-08-15(阶段 4 §7.1)。
- 修改流程:任何字段增删改必须改本文件 + 双侧实现 + 双侧校验器 + B0-B4
  等价回归,并登记 MIGRATION_NOTES(§13 七列记录)。
- 修正记录(同日):§3.2 `delivery_sequence` 起始值由"1 起"修正为"0 起"
  ——实现(C++ `OnlineDriverContext::delivery_seq` 自 0 起)与阶段 3 起
  存档的桥接序列均为 0 起,原稿为书写错误;字段集/类型/其余约束不变,
  不构成 schema 变更。

