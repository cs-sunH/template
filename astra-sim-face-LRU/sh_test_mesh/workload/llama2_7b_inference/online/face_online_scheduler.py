#!/usr/bin/env python3
"""face_online_scheduler.py -- 关感知策略调度器(strategy 模式,步骤 1-9,face 版)。

以 _plan_face_session_lru_recompute(face_scheduler.py)为蓝本迁移,
保持决策顺序逐行对应(每处迁移用 `# offline: face_scheduler.py:XXXX` 注释标注)。
离线事件循环与在线边界的一一对应:

  离线事件循环                                    在线边界
  ----------------                                ----------------
  预置 arrival 堆(:1341-1346)                     ingress ARRIVAL 事件喂入同一
                                                  arrival heap(字段保持
                                                  (time_ns, priority, sequence,
                                                  kind, payload) 形状)
  completion 批(:1598-1682)                        PREFILL_DRAIN / DECODE_COMPLETION
                                                    / REQUEST_COMPLETE 事件处理:
    prefill 完成(:1613-1648)                        _on_prefill_drain:
      qp.popleft + select_decode_instance             qp 出队 + select_decode_
      (:1627-1637,加权图候选 + Roofline per-die 代价,  instance + 等待 decode 准入
      全局 9 实例快照 :1622-1627)                      登记 + try_admit_waiting_decodes
      + waiting_decode_admissions 登记 +
      try_admit_waiting_decodes(:1648)
    decode 完成(:1650-1681)                         _on_decode_complete:
      active_decode 出队 + mark_complete +             mark_complete + 完成快照
      note_capacity_change + 快照                      (active_decode 出队移至列车
      + 下一次 arrival 排程(:1676-1681)                核销;下一次 arrival 排程在
                                                       REQUEST_COMPLETE 边界做)
  arrival 批(:1683-1695)                            _on_arrival(经 arrival heap):
    queue_snapshot -> select_prefill_instance          快照 -> 选择 -> qp.append +
    + qp.append + last_arrival_ns                      last_arrival_ns 更新
  start_ready_iterations(:1517-1590)                _admit_pass(同 tick 末尾):
    try_admit_waiting_decodes(:1519)                   decode 准入(容量 epoch/dirty
                                                       门控,原样保留)
    逐实例 serve(:1520-1588)                           per-instance 列车发射:
      prefill: try_admit_prefill(qp[0])                 try_admit_prefill + 准入动作
      decode:  active_decode 整批                       + 冻结/发射迭代列车(qp 头部
                                                       chunk × 迭代 + active_decode
                                                       成员各 1 token 混拼)

拼 batch 改造(2026-08-22,设计文档《层次 B Continuous Batching 改造》
§3.2"迭代列车聚合发射";sh_1.0 定型版为母本,face 策略结构适配):层次 B
从"请求级大段串行"重构为"实例迭代级列车"——decode 互拼、decode 与
prefill chunk 混拼、chunk 之间不拼、批成员只在迭代(列车)边界变化。每实例
状态机(§3.2):qp(FCFS prefill 队列)/active_decode(批成员表)/
pending_decode_ready(KV 就绪待加入)/in_flight_train(唯一在飞列车,
冻结成员快照 + membership_digest;busy 门 = 一个列车在飞)。列车终点 =
下一个不可预测事件(队列头 prefill drain / 全部工作耗尽),默认不设
T_max(§3.2.8:加入延迟 ≤1 列车,由 §7.4 保真对拍量化治理;SH_TRAIN_MAX_ITER
正整数 = 每列车至多 N 个迭代)。边界原子提交顺序:核验 digest → 推进冻结
成员 token(current_decode_token 闭式)→ 退出成员移出 active_decode →
推进 prefill chunk → 处理 drain/完成 → 合入 arrival → KV 就绪成员入批 →
冻结下一列车成员 → 发射。决策边界仍是四类 reason(ARRIVAL/PREFILL_DRAIN/
DECODE_COMPLETION/REQUEST_COMPLETE),由列车 drain/exit 标记节点的 watch 驱动。

face 独有(与 wscllm 蓝图的差异,逐项保留不抹平):
  - 统一实例:9 实例同时承担 prefill+decode(qp 与 active_decode 可并存);
  - decode 实例动态选择:prefill 完成边界经 WeightedInstanceGraph 候选 +
    Roofline per-die 代价(select_decode_instance,face_scheduler.py)
    选择;每次决策以当前精确工作负载直接计算;拼 batch 后的批口径适配 =
    active_tokens 输入用列车核销后的 current_decode_token(闭式推进,
    §3.4 调用侧适配;select_decode_instance/estimate_iteration_time_ns
    本体不动,红线);
  - decode 选择的顺序敏感全局快照:has_prefill_work(各实例 qp 非空)/
    decode_token_lengths(各实例 active_decode 的 current_decode_token)
    按离线 :1622-1627 同一构造,prefill 完成边界才读取(时机不变);
  - try_admit_waiting_decodes 的阻塞重排队语义(move.admission_blocked ->
    queue.append + continue,face_scheduler.py)原样保留;
  - prefill 准入时点:qp 头部服务时逐请求尝试(容量纪元门防重试风暴),
    与离线 try_admit_prefill(:1382-1453)逐行对应(时机不变)。

real-online 刻意差异(合同⑦ Tier B real-online 验收;不变量 = 同一快照输入
-> 同一输出,不逐值强等):
  - 计时/迭代粒度:离线 Roofline 时钟 + 逐 chunk 迭代 -> 在线真实完成事件 +
    列车聚合构图(weight_passes = 迭代数,权重每迭代只读一次);
  - 实例 busy 语义:离线 busy 覆盖一次混合迭代(1 prefill chunk + 全部
    active decode 各 1 token);在线 busy = "一个列车在飞"(in_flight_
    train,§3.2);
  - current_decode_token 快照口径:离线逐迭代递增(:1652);在线在列车
    核销边界闭式推进(决策边界上与逐 token 精确值逐点一致);
  - remaining_chunks 快照口径:离线逐 chunk 递减(:1610);在线在列车
    核销边界按本列车 chunk 数递减(初始值含 history chunks,:1442-1445)。
"""

import hashlib
import heapq
import json
import math
import os
import sys
from collections import deque

# --------------------------------------------------------------------------
# import 路径:本文件位于 workload/llama2_7b_inference/online/,共享配置、
# 发射与调度模块在上一级。路径只做 import 用途(红线:generate_face_trace.py /
# face_scheduler.py / session_kv_manager.py 只读 import 与注释)。
# --------------------------------------------------------------------------
_ONLINE_DIR = os.path.dirname(os.path.abspath(__file__))
_WORKLOAD_DIR = os.path.dirname(_ONLINE_DIR)
for _path in (_ONLINE_DIR, _WORKLOAD_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from generate_face_trace import (  # noqa: E402
    _candidate_dict,
    _eviction_dict,
    _xy_route,
)
from online.graph_batch_builder import first_token_split_enabled  # noqa: E402
from online.online_scheduler_base import (  # noqa: E402
    BATCH_TRAIN_PREFIX,
    STAGE_DECODE,
    STAGE_PREFILL,
    STAGE_REQUEST,
    OnlineSchedulerBase,
)
from session_kv_manager import (  # noqa: E402
    LOCAL_HBM,
    PARTIAL_HBM_REMOTE,
    SessionKVCacheManager,
    kv_cache_shard_bytes_for_tokens,
)
from face_scheduler import (  # noqa: E402
    NOC_MIGRATE,
    DecodeTieCounter,
    FaceInstanceSpec,
    PrefillQueueSnapshot,
    WeightedInstanceGraph,
    build_instances,
    select_decode_instance,
    select_prefill_instance,
)


class _OnlineInstanceState:
    """在线实例账本(离线 _InstanceRuntime,face_scheduler.py 的在线子集)
    + 拼 batch 列车状态机(§3.2):qp = prefill FCFS 队列(deque),
    active_decode = decode 批成员表,last_arrival_ns = 选择键素材;
    busy 门语义 = "一个列车在飞"(in_flight_train),iteration_count 为
    已完成迭代数(列车核销时闭式推进)。"""

    __slots__ = ("index", "qp", "active_decode", "active_decode_lookup",
                 "last_arrival_ns", "pending_decode_ready", "in_flight_train",
                 "finalized_trains", "iteration_count", "train_seq",
                 "first_step_remainder", "prefill_remaining_chunks")

    def __init__(self, *, index: int) -> None:
        self.index = index
        self.qp = deque()  # FCFS(append 尾入,popleft 首出)
        self.prefill_remaining_chunks = 0
        self.active_decode = []
        self.active_decode_lookup = set()
        self.last_arrival_ns = None
        # ---- 拼 batch 列车账本(2026-08-22) ----
        self.pending_decode_ready = []   # KV 就绪待加入下一列车的成员
        self.in_flight_train = None      # 唯一在飞列车(冻结成员快照)
        self.finalized_trains = []       # 已核销列车(待收后续跨交付信号)
        self.iteration_count = 0         # 已完成迭代数(列车核销时闭式推进)
        self.train_seq = 0               # 列车序号(命名/审计用)
        # ---- WP9 首 token 首步批拆分(2026-08-26) ----
        self.first_step_remainder = None  # 首步批已发、余量批待发射的 train_plan


class _OnlineRequestRuntime:
    """在线请求运行账本(离线 _RequestRuntime 的在线子集)。

    输入事实(request-neutral,来自 manifest,policy-independent):
      history_tokens_before / prefill_context_tokens / final_context_tokens /
      prefill_length / decode_length / queue_index / session_id / turn_index。
    运行期事实(在线决策产出,语义与离线蓝图同名同义)。
    """

    __slots__ = (
        "request_id", "session_id", "turn_index", "queue_index",
        "prefill_length", "decode_length",
        "history_tokens_before", "prefill_context_tokens", "final_context_tokens",
        "estimated_arrival_ns",
        "prefill_instance_index", "prefill_assignment_key",
        "decode_instance_index", "decode_candidates",
        "admitted_prefill", "prefill_attempt_epoch",
        "history_cache_state_before", "hbm_before_request",
        "history_action", "history_source_instance_index",
        "history_transfer_bytes", "history_recompute_tokens",
        "history_transfers",
        "history_location_before",
        "history_evictions", "prefill_evictions", "decode_evictions",
        "drain_block_ends", "kv_pending_location",
        "admission_evictions", "decode_target_evictions",
        "waiting_decode_admission", "prefill_decode_transfer",
        "completion_evictions", "kv_state_after_completion",
        "kv_instance_after_completion", "hbm_after_completion",
        "remaining_chunks", "current_decode_token", "completion_ns",
        "decode_tokens_consumed", "prefill_tokens_completed",
    )

    def __init__(self, record: dict) -> None:
        self.request_id = record["request_id"]
        self.session_id = record["session_id"]
        self.turn_index = record["turn_index"]
        self.queue_index = record["queue_index"]
        self.prefill_length = record["prefill_length"]
        self.decode_length = record["decode_length"]
        self.history_tokens_before = record["history_tokens_before"]
        self.prefill_context_tokens = record["prefill_context_tokens"]
        self.final_context_tokens = record["final_context_tokens"]
        self.estimated_arrival_ns = None
        self.prefill_instance_index = None
        self.prefill_assignment_key = None
        self.decode_instance_index = None
        self.decode_candidates = ()
        self.admitted_prefill = False
        self.prefill_attempt_epoch = None
        self.history_cache_state_before = None
        self.hbm_before_request = None
        self.history_action = None
        self.history_source_instance_index = None
        self.history_transfer_bytes = None
        self.history_recompute_tokens = None
        # B2(2026-09-06):三态恢复路径的 KVTransfer 对象组(契约 §4)。两段
        # 式恢复(前缀 NoC 迁移 + 后缀远端回迁)在同一条 prefill 决策记录
        # 内逐段承载——hbm_watermark 同请求同 kind 两次即 fail,不得拆两条。
        self.history_transfers = ()
        # B3(2026-09-06):准入时点(prepare_history 之前)的会话 KV 快照
        # (SessionKVSnapshot;构图器 partial 流水/位置对账消费)+ 三段
        # 逐出 KVTransfer 账本(history = prepare_history、prefill = grow_
        # prefill、decode = move/grow_decode;跨阻塞 attempt 累积,发射后
        # 核销)+ drain 列车块末(joiner decode_evictions 触发门来源)+
        # 上一 turn 完成时预置的 pending location 链初值。
        self.history_location_before = None
        self.history_evictions = ()
        self.prefill_evictions = ()
        self.decode_evictions = ()
        self.drain_block_ends = None
        self.kv_pending_location = None
        self.admission_evictions = ()
        self.decode_target_evictions = ()
        self.waiting_decode_admission = False
        self.prefill_decode_transfer = None
        self.completion_evictions = ()
        self.kv_state_after_completion = None
        self.kv_instance_after_completion = None
        self.hbm_after_completion = None
        # offline: face_scheduler.py(初始值;准入时 :1442-1445 重算)
        self.remaining_chunks = math.ceil(self.prefill_length / _P_CHUNK_HOLDER[0]) \
            if _P_CHUNK_HOLDER[0] else 0
        # offline: face_scheduler.py(current_decode_token = prefill_context)
        self.current_decode_token = self.prefill_context_tokens
        self.completion_ns = None
        # ---- 拼 batch 列车推进字段(2026-08-22;决策边界上闭式推进,余额
        # 与逐 token 精确值逐点一致,供 select_decode_instance 的
        # active_tokens / queue_snapshot 输入) ----
        self.decode_tokens_consumed = 0  # 已物理完成 decode token 数
        self.prefill_tokens_completed = 0  # 已物理完成 prefill token 数
        # (recompute 段 + 当前段两段式 chunk 序列的合并进度)


# _OnlineRequestRuntime 构造需要 p_chunk(初始 remaining_chunks);调度器
# __init__ 先设置该模块级占位再构造 runtimes(单进程单调度器,无并发)。
_P_CHUNK_HOLDER = [0]


def _kv_event_dict(event) -> dict:
    """Serialize one KVCacheEvent for the online kv_actions stream
    (与 _eviction_dict / _transfer_rows 同构的在线序列化)。"""
    return {
        "event_index": event.event_index,
        "planner_time_ns": event.planner_time_ns,
        "phase": event.phase,
        "event_type": event.event_type,
        "reason": event.reason,
        "trigger_request_id": event.trigger_request_id,
        "session_id": event.session_id,
        "source_instance_index": event.source_instance_index,
        "target_instance_index": event.target_instance_index,
        "context_tokens": event.context_tokens,
        "total_bytes": event.total_bytes,
        "shard_bytes": list(event.shard_bytes),
        "last_completion_ns": event.last_completion_ns,
        "instance_remaining_before_bytes": list(
            event.instance_remaining_before_bytes),
        "instance_remaining_after_bytes": list(
            event.instance_remaining_after_bytes),
        "insufficient_ranks": list(event.insufficient_ranks),
    }


def _transfer_rows(transfers) -> list:
    """B2(2026-09-06):KVTransfer 决策日志行(契约 §3/§4,与 sh_2.0
    ``_transfer_entry_rows`` 行结构逐字对齐:kind/reason/session_id/
    total_bytes/source_instance_index/target_instance_index/layer_start/
    layer_end)。"""
    rows = []
    for transfer in transfers or ():
        if transfer is None:
            continue
        rows.append({
            "kind": transfer.kind,
            "reason": transfer.reason,
            "session_id": transfer.session_id,
            "total_bytes": transfer.total_bytes,
            "source_instance_index": transfer.source_instance_index,
            "target_instance_index": transfer.target_instance_index,
            "layer_start": transfer.layer_start,
            "layer_end": transfer.layer_end,
        })
    return rows


def _eviction_transfer_rows(records) -> list:
    """B3(2026-09-06):逐出条目 → 契约 §3 行(EvictionRecord.transfer 为
    行源;sh 形状逐出条目经 total_bytes 标量回退,不得静默丢)。"""
    rows = []
    for record in records or ():
        transfer = getattr(record, "transfer", None)
        if transfer is not None:
            rows.append({
                "kind": transfer.kind,
                "reason": record.reason,
                "session_id": record.victim_session_id,
                "total_bytes": transfer.total_bytes,
                "source_instance_index": record.victim_instance_index,
                "target_instance_index": transfer.target_instance_index,
                "layer_start": transfer.layer_start,
                "layer_end": transfer.layer_end,
            })
        else:
            rows.append({
                "kind": "remote_store",
                "reason": record.reason,
                "session_id": record.victim_session_id,
                "total_bytes": sum(record.shard_bytes),
                "source_instance_index": record.victim_instance_index,
                "target_instance_index": None,
                "layer_start": None,
                "layer_end": None,
            })
    return rows


def _transfer_hop_rows(transfers) -> list:
    """B3(2026-09-06,sh :148-173):逐传输对象的 noc 路由摘要(纯输出,
    hopbytes 诊断;shards[].noc_path 与构图发射同源)。"""
    rows = []
    for transfer in transfers or ():
        if transfer is None:
            continue
        noc_hop_bytes = 0
        for shard in transfer.shards:
            noc_hop_bytes += int(shard.bytes) * max(
                0, len(shard.noc_path) - 1)
        rows.append({
            "kind": transfer.kind,
            "phase": transfer.phase,
            "reason": transfer.reason,
            "total_bytes": transfer.total_bytes,
            "noc_hop_bytes": noc_hop_bytes,
            "shard_count": len(transfer.shards),
        })
    return rows


def _eviction_hop_rows(records) -> list:
    """B3:逐出条目的 hop 摘要(行源同 _eviction_transfer_rows)。"""
    return _transfer_hop_rows(
        record.transfer for record in records or ()
        if getattr(record, "transfer", None) is not None)


class FaceOnlineScheduler(OnlineSchedulerBase):
    """strategy 变体:face 真实策略(关感知)在在线骨架中运行。

    蓝本: _plan_face_session_lru_recompute(face_scheduler.py),
    kv_cache_policy == "session_lru_recompute"(主变体)。拓扑 / 加权实例图 /
    Roofline 估计 / KV 账本(与离线同一函数、同参数)在运行期按当前候选
    直接计算,策略输入全部来自这些 Python 账本(关感知)。
    """

    def __init__(self, *, manifest, config, graph, digest_sink=None,
                 decision_log_sink=None, train_ledger_sink=None,
                 profile_sink=None, mode: str = "strategy",
                 sensing: bool = False,
                 defensive_reply_cache: bool = False,
                 sensing_query_sink=None, online_stats_sink=None,
                 ledger_sink=None):
        super().__init__(
            manifest=manifest,
            config=config,
            replay=None,  # strategy 无决策日志回放源
            digest_sink=digest_sink,
            mode=mode,
            sensing=sensing,
            decision_log_sink=decision_log_sink,
            profile_sink=profile_sink,
            defensive_reply_cache=defensive_reply_cache,
            sensing_query_sink=sensing_query_sink,
            online_stats_sink=online_stats_sink,
            ledger_sink=ledger_sink,
        )
        if mode != "strategy":
            raise ValueError("FaceOnlineScheduler requires mode == 'strategy'")
        # B2(2026-09-06):kv_cache_policy 值域 fail-closed 同步(契约 §8)。
        # 新值 session_lru_tiered = 三态 KV 内核(本仓唯一行为,无档位
        # 分支);旧值 session_lru_recompute 保留在值域(trace_config 实际
        # 换值归 B3)。
        if config.kv_cache_policy not in (
            "session_lru_recompute",
            "session_lru_tiered",
        ):
            raise ValueError(
                "strategy scheduler supports kv_cache_policy "
                "'session_lru_recompute'/'session_lru_tiered' only, got "
                "{!r}".format(config.kv_cache_policy))
        self.graph = graph  # GraphBatchBuilder(与 replay 路径共用;基类经 self.graph 调 begin_batch)

        # 蓝图 :1314:拓扑(统一实例,require_equal_size)。
        # offline: face_scheduler.py
        specs = tuple(
            FaceInstanceSpec(
                name=group.name,
                pg_name=group.pg_name,
                ranks=group.ranks,
            )
            for group in config.inference_groups
        )
        self.topology = build_instances(
            config.hardware, specs, require_equal_size=True)
        self.p_chunk = int(config.prefill_chunk_size)

        # 蓝图 :1175-1180:加权实例图 + KV 账本。Roofline 估计在候选选择
        # 时直接使用精确 d_token 计算，不持久化工作负载表。
        # offline: face_scheduler.py
        self.instance_graph = WeightedInstanceGraph(self.topology)
        self.kv_manager = SessionKVCacheManager(
            self.topology,
            config.model,
        )
        # 中-1 裁决（2026-08-20）：decode 平局按 instance_index 升序轮流；
        # 在线调度器与离线 plan 各持一个计数器，前进条件相同（仅真实平局）。
        self._decode_tie_counter = DecodeTieCounter()

        # 蓝图 :1331:实例账本(统一实例,无 phase_role)。
        # offline: face_scheduler.py
        self.instances = [
            _OnlineInstanceState(index=instance.index)
            for instance in self.topology.instances
        ]

        # 蓝图 :1333-1334:future arrival min-heap(在线由 ingress ARRIVAL
        # 事件喂入)。键含 queue_index,同 tick 到期项按冻结队列序稳定弹出。
        # offline: face_scheduler.py
        self.arrival_heap = []
        self._sequence = 0

        # 蓝图 :1353-1355:等待 decode 准入登记(按实例)。
        # offline: face_scheduler.py
        self.waiting_decode_admissions = {
            instance.index: deque() for instance in self.topology.instances
        }
        # 蓝图 :1359-1362:容量 epoch / 准入门控。
        # offline: face_scheduler.py
        self.capacity_epoch = [0 for _ in self.topology.instances]
        # Last failed admission epoch lives on the request runtime itself.
        # Keeping a second request_id->epoch dictionary retained one entry for
        # every request ever attempted, even after runtime retirement.
        self.decode_admission_epoch = [-1 for _ in self.topology.instances]
        self.decode_admission_dirty = set()
        # P0-1(2026-08-31):prefill 侧阻塞重排队门(镜像 decode 侧
        # decode_admission_dirty)。原语义下"无逐出的 prepare_history 阻塞"
        # 不推进 capacity_epoch,纪元门(:1262 一带)对本实例永久关闭;
        # 隔离时间线没有邻居完成事件重开它 => 40 个 prefill 卡死形态
        # (总文档 §4 P0-1)。阻塞即置 dirty,下一个决策边界重试。
        self.prefill_admission_dirty = set()
        # T_max 列车长度上限(§3.2.8/§7.4 治理旋钮 + A2 逐迭代 oracle):
        # SH_TRAIN_MAX_ITER 正整数 = 每列车至多 N 个迭代(截断列车无自然
        # drain/exit 标记时发射哨兵标记);缺省/0 = 不设限(交付默认)。
        # 交付默认 = 8(2026-08-22 §7.4 A2 对拍裁决,与 sh_1.0 母本统一:
        # 无上限 TTFT -67.3%,16 仍 -19.1%,8 全指标 ≤1.3%;原则 1 优先
        # 于节点数)。0 = 不设限(oracle/灵敏度复跑用)。
        self._train_max_iter = int(
            os.environ.get("SH_TRAIN_MAX_ITER", "8") or 0)
        # train_id -> instance_index(哨兵事件路由)。
        self._train_instance_index = {}
        # WP9 首 token 首步批拆分(2026-08-26):batch_train_<id>_first_step
        # 唤醒 watch 的 id -> instance_index(首步批交付回声识别;fire 后
        # 从哨兵核销通道剥离,余量批在 _plan_and_emit_trains 的 busy 分支
        # 发射)。
        self._pending_first_steps = {}
        # 拼 batch 列车台账(§7.3 不变量断言输入):每次列车发射一行,
        # 由 online_service 落 bridge 目录 train_ledger.jsonl(审计产物)。
        # M3 流式落盘(2026-08-23):提供 train_ledger_sink 时行即写即
        # 弃,不驻留本列表;缺省 None = 兼容旧路径(行仍缓冲)。
        self.train_ledger_sink = train_ledger_sink
        self.train_ledger_rows = []
        # O(1) queue-depth snapshots.  SH_QUEUE_AGG_VERIFY=1 retains the old
        # full sum as a shadow oracle without putting it on production runs.
        self._queue_aggregate_verify = (
            os.environ.get("SH_QUEUE_AGG_VERIFY") == "1")

        # §7.3 ready frontier:非忙且有排队工作的实例集合(发射时清除,
        # 完成/到达时设置;结束审计必须为空;sorted 保持实例 index 序 =
        # 离线 :1520 的循环序,决策确定性不受影响)。
        self._ready_frontier = set()

        # 请求运行账本(queue_index 序;manifest 事实 policy-independent)。
        _P_CHUNK_HOLDER[0] = self.p_chunk
        self.runtimes = [
            _OnlineRequestRuntime(record)
            for record in sorted(
                manifest["requests"], key=lambda item: item["queue_index"])
        ]
        self.runtime_by_request_id = {
            runtime.request_id: runtime for runtime in self.runtimes
        }
        # §7.3:request_id -> runtimes 下标 + 同 session 下一 turn 索引。
        self._runtime_index = {
            runtime.request_id: index
            for index, runtime in enumerate(self.runtimes)
        }
        by_turn = {}
        for runtime in self.runtimes:
            by_turn[(runtime.session_id, runtime.turn_index)] = runtime
        self.next_request = [None] * len(self.runtimes)
        for index, runtime in enumerate(self.runtimes):
            self.next_request[index] = by_turn.get(
                (runtime.session_id, runtime.turn_index + 1))

        self.completed_requests = 0
        # kv 事件水位:kv_actions 流(阶段 2 对照 kv_cache_events.csv 基线)。
        self._kv_events_emitted = 0

    # ------------------------------------------------------------- 策略 --

    def run_variant_policy(self, delta) -> None:
        """决策顺序逐行对应离线事件循环:completion 批(:1598-1682,先
        prefill 完成后 decode 完成)先于 arrival 批(:1683-1695),最后
        start_ready_iterations(:1697)。拼 batch 改造:completion 批先经
        _finalize_completed_trains 核销列车(账本推进恰一次),再处理 drain/
        完成,最后 _admit_pass 冻结并发射各空闲实例的下一列车。
        """
        tick = delta["tick"]

        # ---- completion 批(离线 priority 0;同 tick 先于 arrival)----
        # offline: face_scheduler.py
        drained = []
        completed_now = []
        sentinel_trains = []
        for group in delta["completed_groups"]:
            request_id = group["request_id"]
            if request_id.startswith(BATCH_TRAIN_PREFIX):
                sentinel_trains.append(request_id)
                continue
            stage = group["stage"]
            if stage == STAGE_PREFILL:
                drained.append(request_id)
            elif stage == STAGE_DECODE:
                completed_now.append(request_id)
            elif stage == STAGE_REQUEST:
                # REQUEST_COMPLETE 与 DECODE_COMPLETION 同 tick 交付;完成
                # 处理在 _on_decode_complete 后一并做(下一次 arrival 排程)。
                continue
            else:
                raise ValueError(
                    "unknown completion stage {!r} for request {!r}".format(
                        stage, request_id))
        # WP9 首 token 拆分(2026-08-26):batch_train_*_first_step 唤醒
        # 信号是"自己发射的首步批"的交付回声——无操作(不决策/不记账/
        # 不写 decision_log),从哨兵核销通道剥离;余量批在
        # _plan_and_emit_trains 的 busy 分支发射(本交付或任一后续交付,
        # 余量节点经依赖边排在首步节点之后,早发不改物理序)。首步批的
        # commit ack 走基类协议记账(ack_count/幂等门),变体侧零动作。
        sentinel_trains = [
            train_id for train_id in sentinel_trains
            if not self._consume_first_step_wakeup(train_id)
        ]
        # 拼 batch 列车账本(§3.2 边界原子提交顺序):先核销已完成列车
        # (核验 train_id + membership_digest → 冻结成员推进 token → 退出
        # 成员移出 active_decode → 推进 prefill chunk),再处理 drain/
        # 完成/到达,最后冻结并发射各空闲实例的下一列车。
        self._finalize_completed_trains(drained, completed_now,
                                        sentinel_trains, tick)
        for request_id in drained:
            self._on_prefill_drain(request_id, tick)
        for request_id in completed_now:
            self._on_decode_complete(request_id, tick)
            # 离线 :1676-1681 的下一次 arrival 排程(REQUEST_COMPLETE 边界,
            # 与 decode 完成同 tick 配对)。
            self._on_request_complete(request_id, tick)

        # ---- arrival 批(离线 priority 1)----
        # offline: face_scheduler.py
        for arrival in delta["arrivals"]:
            self._push_arrival(arrival, tick)
        self._drain_arrival_heap(tick)

        # ---- 准入/发射 pass(离线 start_ready_iterations,计时部分由
        #      迭代列车承载)----
        # offline: face_scheduler.py -> 1517-1590
        self._admit_pass(tick)

        # ---- kv 动作流:本批次 kv_manager 新产出的账本事件 ----
        # B1(2026-08-28):改走 events_since 增量读取(消费方游标本就存在),
        # 消除每批 events property 的全量 tuple() 拷贝;已消费前缀按水位
        # 整段压缩(游标同步回退),事件载荷/event_index 不变。
        new_events = self.kv_manager.events_since(self._kv_events_emitted)
        if new_events:
            self._batch["kv_actions"].extend(
                _kv_event_dict(event)
                for event in new_events)
            self._kv_events_emitted += len(new_events)
            removed = self.kv_manager.compact_events(
                self._kv_events_emitted)
            if removed:
                self._kv_events_emitted -= removed

    # ------------------------------------------------------ 列车账本 --

    def _consume_first_step_wakeup(self, train_id: str) -> bool:
        """识别并吞掉自己发射的首步批唤醒信号(WP9,2026-08-26)。

        首步批的唤醒 watch 用批命名空间 id("<train_id>_first_step",
        batch_train_ 前缀),fire 后与真哨兵同通道送达。返回 True = 这是
        首步唤醒(无操作,仅从哨兵核销列表剥离);False = 非首步 id
        (真哨兵或未知 batch_train_ id,交回哨兵核销逻辑处理)。"""
        if train_id not in self._pending_first_steps:
            return False
        self._pending_first_steps.pop(train_id)
        return True

    def _finalize_completed_trains(self, drained, completed_now,
                                   sentinel_trains, tick: int) -> None:
        """核销本交付中标记 watch 已 fire 的列车(§3.2 原子提交的前半)。
        drain/exit 标记节点是列车体后的最末真实节点,任一标记 watch fire
        即列车物理主体完成。同一列车不同成员的标记 watch 可能跨 tick
        fire、事件拆到不同交付——首个信号执行核销(账本推进恰一次),
        后续信号只清已核销列车的 pending 集合(幂等);信号不属于任何
        在飞/已核销列车即陈旧完成错配 fail-closed。推进量全部闭式
        (每成员 participation 次 token,无逐 token 循环)。"""
        signaled = {}
        for request_id in drained:
            runtime = self.runtime_by_request_id[request_id]
            signaled.setdefault(
                runtime.prefill_instance_index, set()).add(request_id)
        for request_id in completed_now:
            runtime = self.runtime_by_request_id[request_id]
            signaled.setdefault(
                runtime.decode_instance_index, set()).add(request_id)
        for train_id in sentinel_trains:
            instance_index = self._train_instance_index.pop(train_id, None)
            if instance_index is None:
                raise RuntimeError(
                    "sentinel signal for unknown train {}".format(train_id))
            signaled.setdefault(instance_index, set()).add(train_id)
        for instance_index, signals in signaled.items():
            state = self.instances[instance_index]
            pending_signals = set(signals)
            consumed = set()
            train = state.in_flight_train
            if train is not None:
                inflight_hits = pending_signals & train["signal_set"]
                if inflight_hits:
                    # 该列车首个信号到达:核销(账本推进恰一次),残余
                    # 信号(drain/exit 标记跨 tick fire)登记待收。
                    iterations = train["iterations"]
                    for request_id, participation in train["members"]:
                        runtime = self.runtime_by_request_id[request_id]
                        runtime.decode_tokens_consumed += participation
                        runtime.current_decode_token += participation
                    for request_id in train["exit_set"]:
                        runtime = self.runtime_by_request_id[request_id]
                        if runtime not in state.active_decode_lookup:
                            raise RuntimeError(
                                "exiting member {} is not in the decode "
                                "batch".format(request_id))
                        state.active_decode_lookup.discard(runtime)
                        state.active_decode.remove(runtime)
                    for (request_id,
                         chunk_tokens) in train["prefill_chunk_tokens"]:
                        runtime = self.runtime_by_request_id[request_id]
                        runtime.prefill_tokens_completed += chunk_tokens
                        runtime.remaining_chunks -= 1
                        state.prefill_remaining_chunks -= 1
                    state.iteration_count += iterations
                    state.in_flight_train = None
                    # M4 核销即删(2026-08-23):列车核销后其 train_id→
                    # 实例索引条目即死重(哨兵条目已在信号路由处弹出,
                    # 此 pop 对其为幂等 no-op;全仓 grep 证实核销后无读者)。
                    self._train_instance_index.pop(train["train_id"], None)
                    state.finalized_trains.append({
                        "train_id": train["train_id"],
                        "pending": (train["signal_set"] - inflight_hits),
                    })
                    consumed |= inflight_hits
                    pending_signals -= inflight_hits
            # 已核销列车的后续信号对账(幂等清 pending,清空即出列表)。
            for record in state.finalized_trains:
                hits = pending_signals & record["pending"]
                record["pending"] -= hits
                consumed |= hits
            state.finalized_trains = [
                record for record in state.finalized_trains
                if record["pending"]]
            unresolved = pending_signals - consumed
            if unresolved:
                raise RuntimeError(
                    "completion signals {} for instance {} match no "
                    "in-flight or recently finalized train (stale "
                    "completion mismatch)".format(
                        sorted(unresolved), instance_index))
            # §7.3 ready frontier:列车核销 = 实例空闲(离线 :1603 的 busy
            # 复位列车化);仍有排队工作(qp/active_decode/pending)即就绪。
            if state.qp or state.active_decode or state.pending_decode_ready:
                self._ready_frontier.add(instance_index)
            else:
                self._ready_frontier.discard(instance_index)

    def _plan_train(self, state, qp_head):
        """冻结实例的下一列车成员快照(§3.2 构造规则)。

        列车终点 = 下一个不可预测事件之前的最后一个完整迭代:队列头
        prefill 的 drain 迭代(剩余 chunk 数,先验)或全部 decode 工作
        耗尽(无 prefill 工作时 = max 剩余 token;默认不设 T_max,§3.2.8)。
        成员退出不是列车边界(先验):退出成员在列车内挂 exit 标记。
        每迭代至多 1 个 prefill chunk(FCFS 队列头);chunk 之间不互拼。
        返回 None = 实例无工作。"""
        members = []
        for runtime in state.active_decode:
            remaining = (
                runtime.decode_length - runtime.decode_tokens_consumed)
            if remaining <= 0:
                raise RuntimeError(
                    "decode member {} has no remaining tokens".format(
                        runtime.request_id))
            members.append((runtime, runtime.prefill_context_tokens,
                            runtime.decode_tokens_consumed, remaining))
        if qp_head is None:
            if not members:
                return None
            natural_iterations = max(
                remaining for _, _, _, remaining in members)
        else:
            natural_iterations = qp_head.remaining_chunks
            if natural_iterations <= 0:
                raise RuntimeError(
                    "prefill queue head {} has no remaining chunks".format(
                        qp_head.request_id))
        iterations = natural_iterations
        capped = False
        if self._train_max_iter and iterations > self._train_max_iter:
            iterations = self._train_max_iter
            capped = True
        # B3(sh :630-635):partial 恢复列车判定——队列头首 chunk 且历史
        # 位置为 partial(准入时点快照)。该列车 span 平铺改为 [首 chunk]
        # + [各成员第 1 迭代 span] + [其余 chunk + 成员剩余 span],首组
        # 进列车的 prefix/suffix 层段两段发射(真流水,见构图器)。
        partial = (
            qp_head is not None
            and qp_head.prefill_tokens_completed == 0
            and qp_head.history_location_before is not None
            and qp_head.history_location_before.location == PARTIAL_HBM_REMOTE
        )
        member_parts = []
        exit_members = []
        member_first_spans = []
        member_rest_spans = []
        member_spans = []
        pass_spans: list[tuple[int, int]] = []
        chunk_records = []
        if qp_head is not None:
            chunk_records = self._plan_head_chunks(qp_head, iterations)
            pass_spans.extend(span for _, _, span in chunk_records)
        for runtime, context, consumed, remaining in members:
            participation = min(remaining, iterations)
            member_first_spans.append((1, context + consumed + 1))
            member_rest_spans.extend(
                (1, context + consumed + step)
                for step in range(2, participation + 1))
            member_spans.extend(
                (1, context + consumed + step)
                for step in range(1, participation + 1))
            member_parts.append((runtime.request_id, participation))
            if participation >= remaining:
                exit_members.append(runtime.request_id)
        if partial:
            pass_spans = (
                pass_spans[:1] + member_first_spans + pass_spans[1:]
                + member_rest_spans)
        else:
            pass_spans.extend(member_spans)
        # 队列头在列车内完成其全部剩余 chunk(列车长度 = 头部剩余 chunk
        # 数)⇒ 列车终于头部 drain 迭代(drain 是先验已知的列车边界)。
        # T_max 截断时头部未必 drain —— 重算。
        drain_members = [qp_head.request_id] if (
            qp_head is not None and not capped) else []
        head_first_chunk = (
            qp_head is not None and qp_head.prefill_tokens_completed == 0)
        state.train_seq += 1
        train_id = "batch_train_i{}_{}".format(state.index, state.train_seq)
        signal_set = set(exit_members) | set(drain_members)
        if capped and not drain_members and not exit_members:
            signal_set.add(train_id)  # 哨兵:列车自身 id 即完成信号
        snapshot = json.dumps(
            {
                "train_id": train_id,
                "iterations": iterations,
                "members": member_parts,
                "exits": exit_members,
                "drains": drain_members,
                "chunks": [(rid, tokens) for rid, tokens, _ in chunk_records],
                "partial": partial,
                "capped": capped,
            },
            sort_keys=True,
        )
        return {
            "train_id": train_id,
            "iterations": iterations,
            "members": member_parts,
            "exit_members": exit_members,
            "exit_set": set(exit_members),
            "drain_members": drain_members,
            "drain_set": set(drain_members),
            "prefill_chunk_tokens": [
                (rid, tokens) for rid, tokens, _ in chunk_records],
            "head_first_chunk": head_first_chunk,
            "pass_spans": pass_spans,
            "partial": partial,
            "partial_first_chunk_count": (
                1 + len(member_parts) if partial else None),
            "capped": capped,
            "sentinel": bool(capped and not drain_members
                             and not exit_members),
            "signal_set": signal_set,
            "membership_digest": hashlib.sha256(
                snapshot.encode()).hexdigest(),
        }

    def _plan_head_chunks(self, qp_head, count: int):
        """face 队头 prefill 段的接下来 count 个 chunk。

        B2(2026-09-06):RECOMPUTE 从调度器规划中删除(策略 §1.4)——
        被逐出会话的历史经三态恢复路径回迁(local_hit/noc_migrate/
        remote_load),不再生成重算段;remaining_chunks 口径同步去掉
        ceil(R/p_chunk) 项(见 _try_admit_prefill)。当前段在
        history_tokens_before 上追加,span 构造与离线 chunk 序同构。
        返回 [(request_id, chunk_tokens, (tokens, kv) span), ...]。"""
        prefill_tokens = qp_head.prefill_length
        completed = qp_head.prefill_tokens_completed
        history = qp_head.history_tokens_before
        chunks = []
        for _ in range(count):
            tokens = min(self.p_chunk, prefill_tokens - completed)
            if tokens <= 0:
                raise RuntimeError(
                    "prefill queue head {} ran out of work inside the "
                    "planned train".format(qp_head.request_id))
            span = (tokens, history + completed + tokens)
            chunks.append((qp_head.request_id, tokens, span))
            completed += tokens
        return chunks

    def _plan_and_emit_trains(self, tick: int) -> None:
        """为每个空闲且有工作的实例冻结并发射下一列车(§3.2 原子提交
        的后半:KV 就绪成员(pending_decode_ready,3000 迁移随加入列车
        发射,物理先于列车体)进入 active_decode → 冻结成员 → 发射)。
        busy 门 = 一个列车在飞(§3.2):在飞实例跳过,不重复发射。

        face 策略保留(§8.2):qp 头部的 prefill 准入在服务时点逐请求尝试
        (try_admit_prefill,容量纪元门防重试风暴);准入失败时本列车退化
        为纯 decode(离线 :1524-1526"准入失败时服务 decode 队首"的同构
        映射)。§7.3:逐实例 serve 只访问 ready frontier(sorted 保持实例
        index 序 = 离线 :1520 的循环序,决策确定性不受影响)。

        WP9 拆分列车的余量批在 frontier 循环前的实例遍历中发射(首步
        唤醒到达后的首个决策边界,或其后任一交付;余量节点经依赖边排
        在首步节点之后,早发不改物理序)。face 实例 rank 组两两不交,
        余量节点与其它实例本批节点无同 rank 交错;先于 frontier 循环
        发射避免同实例准入动作节点插入首步体与余量体之间。"""
        # WP9:两段式发射的后半(余量体 + drain/exit/哨兵标记 + end
        # barrier;标记 watch 全部在本批注册,in_flight_train 语义恢复
        # 整列口径)。
        for state in self.instances:
            if state.first_step_remainder is not None:
                self._emit_train_remainder(state, tick)
        for instance_index in sorted(self._ready_frontier):  # §7.3 frontier
            self._profile_scan()  # §7.3:frontier 访问条目(就绪实例)
            state = self.instances[instance_index]
            if state.in_flight_train is not None:
                continue  # busy 门:一个列车在飞(防御:frontier 失步)
            qp_head = None
            for runtime in state.qp:
                # drain 决策跨交付未达的头部(remaining_chunks 已在列车
                # 核销时清零,drain 决策事件尚在途中)不提供 chunk 工作;
                # 其后续请求的 chunk 物理上已可开始(头部 prefill 主体已
                # 完成)。
                if runtime.remaining_chunks > 0:
                    qp_head = runtime
                    break
            if qp_head is not None and not qp_head.admitted_prefill:
                if self._try_admit_prefill(qp_head, tick):  # :1524-1525
                    self._emit_admission(qp_head, tick)
                else:
                    qp_head = None  # 准入失败:纯 decode 列车(face 回退)
            joiners = []
            if state.pending_decode_ready:
                joiners = list(state.pending_decode_ready)
                state.pending_decode_ready.clear()
                state.active_decode.extend(joiners)
                for runtime in joiners:
                    state.active_decode_lookup.add(runtime)  # §7.3 双侧同步
                    self._note_emitted(runtime.request_id, STAGE_DECODE)
                    self._ledger_issue(runtime.request_id, tick,
                                       STAGE_DECODE, state.index)
            plan = self._plan_train(state, qp_head)
            if plan is None:
                if not (state.qp or state.active_decode
                        or state.pending_decode_ready):
                    self._ready_frontier.discard(instance_index)
                continue
            self._emit_train(state, plan, joiners, qp_head, tick)

    def _emit_train(self, state, plan, joiners, qp_head, tick: int) -> None:
        """把冻结的列车计划交给构图器发射,注册 drain/exit/哨兵标记
        watch,并挂起 in_flight_train(busy 门 = 一个列车在飞)。

        WP9(2026-08-26):列车含 debut 成员且拆分开启、迭代数 >= 2 时
        两段式发射——本交付只发首步批(迁移/起始标记/首迭代体/first_
        token 标记/唤醒标记),drain/exit/哨兵 watch、completion_gates
        (构图器侧,随余量批 end barrier)与正常 train_ledger 行移至
        余量批(_emit_train_remainder);成员选择/排序/KV 动作/挂点语义
        全部不变。"""
        joiner_plans = []
        for runtime in joiners:
            joiner_plan = self._plan_dict(runtime)
            # B3(sh :856-860):joiner 的 drain 列车块末(decode_evictions
            # 触发门来源)+ decode 逐出账本(列车头发射;发射后核销,
            # 与 drain_block_ends 同点清空——每个 joiner 恰消费一次)。
            joiner_plan["prefill_drain_block_ends"] = dict(
                runtime.drain_block_ends or {})
            joiner_plan["decode_evictions"] = runtime.decode_evictions
            joiner_plans.append(joiner_plan)
        stage = "decode" if (plan["members"] or joiners) else "prefill"
        prefill_start_member = None
        if qp_head is not None and plan.get("head_first_chunk"):
            prefill_start_member = {"request_id": qp_head.request_id}
        train_plan = {
            "train_id": plan["train_id"],
            "instance_index": state.index,
            "stage": stage,
            "joiners": joiner_plans,
            "pass_spans": plan["pass_spans"],
            "iterations": plan["iterations"],
            "prefill_start_member": prefill_start_member,
            "sentinel": plan["sentinel"],
            "drain_members": [
                {"request_id": request_id,
                 "session_id": self.runtime_by_request_id[request_id].session_id}
                for request_id in plan["drain_members"]],
            "exit_members": [
                {"request_id": request_id,
                 "session_id": self.runtime_by_request_id[request_id].session_id}
                for request_id in plan["exit_members"]],
        }
        if plan.get("partial"):
            train_plan["partial_first_chunk_count"] = (
                plan["partial_first_chunk_count"])
        first_token = self._first_token_plan(plan, joiners)
        if first_token is not None:
            train_plan["first_token"] = first_token
            if first_token["split"]:
                # WP9:joiner id 快照随 train_plan 走(余量批的台账行
                # 需要与首步行相同的 joiners 记录;joiner_plans 只在
                # 首步批消费)。
                train_plan["joiner_ids_of_record"] = [
                    runtime.request_id for runtime in joiners]
                self._emit_train_first_step(state, plan, train_plan, tick)
                return
        result = self.graph.emit_iteration_train(train_plan)
        self._train_instance_index[plan["train_id"]] = state.index
        for request_id, members in result["drain_members"].items():
            self._batch["watches"].append({
                "request_id": request_id,
                "stage": STAGE_PREFILL,
                "generation": 0,
                "members": members,
                "statuses": ["Success", "Skipped"],
            })
            # B3(sh :904-906):drain 列车块末随 runtime 走——本请求 decode
            # 准入时 joiner 的 decode_evictions 触发门来源(post-barrier
            # 口径;consumed once at joiner train emission)。
            self.runtime_by_request_id[
                request_id].drain_block_ends = dict(result["block_ends"])
        # B3:joiner 的 decode 逐出账本已随列车头消费,核销防重发。
        for runtime in joiners:
            runtime.decode_evictions = ()
            runtime.drain_block_ends = None
        for request_id, members in result["exit_members"].items():
            self._batch["watches"].append({
                "request_id": request_id,
                "stage": STAGE_DECODE,
                "generation": 1,
                "members": members,
                "statuses": ["Success", "Skipped"],
            })
        if plan["sentinel"]:
            self._batch["watches"].append({
                "request_id": plan["train_id"],
                "stage": STAGE_PREFILL,   # 固定 prefill:单事件通道
                "generation": 0,
                "members": result["sentinel_members"],
                "statuses": ["Success", "Skipped"],
            })
        state.in_flight_train = plan
        self._ready_frontier.discard(state.index)  # §7.3:发射即忙
        self._emit_train_ledger_row(
            state, plan,
            [runtime.request_id for runtime in joiners], tick)

    def _first_token_split_spans(self, plan):
        """WP9 首/余量 span 组切分(机械操作,2026-08-26)。

        首步 = [prefill 队头第 1 个 chunk] + [各 decode 成员第 1 个
        span];余量 = [剩余 chunk] + [各成员剩余 span]。聚合节点对 span
        求和与顺序无关,两组的激活/KV/AR 字节总量与整列一致;权重经
        weight_passes(1 + iterations-1)合计不变。

        B3(sh :779-786):partial 恢复列车的 layout 本就是 [首 chunk]
        + [各成员第 1 迭代 span] + [其余 chunk + 成员剩余 span],前缀组
        恰为首步组(partial_first_chunk_count 切点)。"""
        spans = list(plan["pass_spans"])
        if plan.get("partial"):
            cut = int(plan["partial_first_chunk_count"])
            return list(spans[:cut]), list(spans[cut:])
        chunk_count = len(plan["prefill_chunk_tokens"])
        member_parts = plan["members"]
        if chunk_count + sum(p for _, p in member_parts) != len(spans):
            raise RuntimeError(
                "train span layout does not match the frozen plan")
        first_spans = spans[:1] if chunk_count else []
        rest_spans = list(spans[1:chunk_count]) if chunk_count else []
        offset = chunk_count
        for _, participation in member_parts:
            first_spans.append(spans[offset])
            rest_spans.extend(spans[offset + 1:offset + participation])
            offset += participation
        return first_spans, rest_spans

    def _first_token_plan(self, plan, joiners):
        """WP9 首 token 观测计划(None = 开关关闭/无 debut,行为与拆分
        上线前逐字节一致)。

        debut 成员 = 本交付加入列车的 decode 成员(decode_tokens_consumed
        == 0,计划期可知)。多 token debut 挂独立 first_token 标记;
        decode_length=1 的 debut 其 exit 标记名附加 first_token 子串
        (同节点 code 4/8 双锚点)。iterations >= 2 时物理拆两批(首步批
        + 余量批),否则仅做不拆车的标记增强。"""
        if not first_token_split_enabled():
            return None
        debut = [
            runtime for runtime in joiners
            if runtime.decode_tokens_consumed == 0]
        if not debut:
            return None
        debut_marker_members = [
            {"request_id": runtime.request_id}
            for runtime in debut
            if runtime.decode_length != 1]
        debut_exit_first_token = [
            runtime.request_id for runtime in debut
            if runtime.decode_length == 1]
        if plan["iterations"] >= 2:
            first_spans, rest_spans = self._first_token_split_spans(plan)
            return {
                "split": True,
                "first_spans": first_spans,
                "rest_spans": rest_spans,
                "debut_marker_members": debut_marker_members,
                "debut_exit_first_token": debut_exit_first_token,
                "wakeup_id": "{}_first_step".format(plan["train_id"]),
            }
        return {
            "split": False,
            "debut_marker_members": debut_marker_members,
            "debut_exit_first_token": debut_exit_first_token,
        }

    def _emit_train_ledger_row(self, state, plan, joiner_ids, tick: int,
                               first_step: bool = False) -> None:
        """train_ledger 行(M3 流式落盘)。WP9:拆分列车的首步批先写
        first_step=True 行(ON/OFF 对拍剥离标记),余量批再写正常行
        (与整列发射的行同构)。"""
        ledger_row = {
            "train_id": plan["train_id"],
            "instance_index": state.index,
            "tick": tick,
            "iterations": plan["iterations"],
            "member_count": len(plan["members"]),
            "member_iterations": sum(
                participation for _, participation in plan["members"]),
            "joiners": list(joiner_ids),
            "drains": list(plan["drain_members"]),
            "exits": list(plan["exit_members"]),
            "prefill_chunks": len(plan["prefill_chunk_tokens"]),
            # B3:partial 恢复列车标记(首 chunk 层段拆分发射的审计行)。
            "partial": bool(plan.get("partial")),
            "pass_spans": len(plan["pass_spans"]),
        }
        if first_step:
            ledger_row["first_step"] = True
        # M3 流式落盘:提供 train_ledger_sink 时行即写即弃;缺省缓冲。
        if self.train_ledger_sink is not None:
            self.train_ledger_sink(ledger_row)
        else:
            self.train_ledger_rows.append(ledger_row)

    def _emit_train_first_step(self, state, plan, train_plan,
                               tick: int) -> None:
        """WP9 首步批发射(两段式前半,2026-08-26):构图 + 唤醒 watch
        注册 + busy 门挂起 + first_step 台账行。drain/exit/哨兵 watch
        与正常台账行全部移至余量批。"""
        first_token = train_plan["first_token"]
        result = self.graph.emit_train_first_step(train_plan)
        self._batch["watches"].append({
            # 批命名空间唤醒 watch(哨兵同款单事件通道):首步批完成即
            # 交付余量批;不挂任何请求,fire 事件在 run_variant_policy
            # 的 _consume_first_step_wakeup 处无操作剥离。
            "request_id": first_token["wakeup_id"],
            "stage": STAGE_PREFILL,
            "generation": 0,
            "members": result["wakeup_members"],
            "statuses": ["Success", "Skipped"],
        })
        self._train_instance_index[plan["train_id"]] = state.index
        state.in_flight_train = plan
        state.first_step_remainder = train_plan
        self._pending_first_steps[first_token["wakeup_id"]] = state.index
        self._ready_frontier.discard(state.index)  # §7.3:发射即忙
        # B3:joiner 的 decode 逐出账本已随首步批列车头消费,核销防重发
        # (drain_block_ends 不在此清——余量批的 watch 注册不消费它,
        # joiner 消费点已过)。
        for request_id in train_plan["joiner_ids_of_record"]:
            joiner_runtime = self.runtime_by_request_id[request_id]
            joiner_runtime.decode_evictions = ()
        self._emit_train_ledger_row(
            state, plan, train_plan["joiner_ids_of_record"],
            tick, first_step=True)

    def _emit_train_remainder(self, state, tick: int) -> None:
        """WP9 余量批发射(两段式后半,2026-08-26):余量体 + drain/
        exit/哨兵标记 + end barrier,随后注册 drain/exit/哨兵 watch 与
        正常台账行——与整列发射的后半完全同构(completion_gates 由构图
        器随 end barrier 写入,口径不变)。"""
        train_plan = state.first_step_remainder
        state.first_step_remainder = None
        plan = state.in_flight_train
        result = self.graph.emit_train_remainder(train_plan)
        for request_id, members in result["drain_members"].items():
            self._batch["watches"].append({
                "request_id": request_id,
                "stage": STAGE_PREFILL,
                "generation": 0,
                "members": members,
                "statuses": ["Success", "Skipped"],
            })
            # B3(sh :904-906):drain 列车块末随 runtime 走——与整列发射
            # _emit_train 同款回写(余量批 end barrier 即整列的 end
            # barrier,返回结构相同);漏写会使 SH_FIRST_TOKEN_SPLIT=1
            # 时被拆分列车 drain 的请求其后 arm 全 None,逐出 shard 丢掉
            # 对 drain 列车 end barrier 的时序依赖边且无报错。
            self.runtime_by_request_id[
                request_id].drain_block_ends = dict(result["block_ends"])
        for request_id, members in result["exit_members"].items():
            self._batch["watches"].append({
                "request_id": request_id,
                "stage": STAGE_DECODE,
                "generation": 1,
                "members": members,
                "statuses": ["Success", "Skipped"],
            })
        if plan["sentinel"]:
            self._batch["watches"].append({
                "request_id": plan["train_id"],
                "stage": STAGE_PREFILL,   # 固定 prefill:单事件通道
                "generation": 0,
                "members": result["sentinel_members"],
                "statuses": ["Success", "Skipped"],
            })
        self._emit_train_ledger_row(
            state, plan, train_plan["joiner_ids_of_record"], tick)

    # ------------------------------------------------------------- 边界 --

    def _on_arrival(self, runtime, tick: int) -> None:
        """离线 arrival 批单条(:1683-1695):快照 -> select_prefill_instance
        -> qp 入队 -> last_arrival_ns 更新。快照在 append 之前取(ordering_key
        反映选择时刻的排队深度/最近到达)。

        offline: face_scheduler.py
        """
        runtime.estimated_arrival_ns = tick  # :1688
        snapshots = self._queue_snapshots()  # :1689
        selected = select_prefill_instance(snapshots)  # :1690
        # P1(2026-08-31)选择期容量可行性过滤:select_prefill_instance 的
        # 只读预检包装(全上下文保守口径,详见 _filter_prefill_selection)。
        # 原选可行时选择与原实现逐字节一致(决策零扰动)。
        selected = self._filter_prefill_selection(
            runtime, snapshots, selected, tick)
        selected_snapshot = snapshots[selected]  # :1691
        runtime.prefill_instance_index = selected  # :1692
        runtime.prefill_assignment_key = selected_snapshot.ordering_key  # :1693
        self.instances[selected].qp.append(runtime)  # :1694
        self.instances[selected].prefill_remaining_chunks += (
            runtime.remaining_chunks)
        self.instances[selected].last_arrival_ns = tick  # :1695
        self._note_instance_ready(selected)  # §7.3 ready frontier
        # 阶段 3 感知账本:进入 admitted 层(prefill_qp 排队账本成员;
        # contract ⑥)。查询/审计数据,不进策略判据。
        self._ledger_admit(runtime.request_id, tick,
                           {"type": "prefill_qp", "instance_index": selected})

    def _on_prefill_drain(self, request_id: str, tick: int) -> None:
        """离线 prefill 完成分支(:1476-1512):qp 出队 + 全局快照 ->
        select_decode_instance(加权图候选 + Roofline per-die 代价) ->
        等待 decode 准入登记 + dirty + 立即 try_admit_waiting_decodes。

        offline: face_scheduler.py
        """
        runtime = self.runtime_by_request_id[request_id]
        state = self.instances[runtime.prefill_instance_index]
        # 拼 batch 改造(2026-08-22):drain 事件由列车 drain 标记承载;
        # qp 头部跳过规则(列车核销后 remaining_chunks==0 的头部先于其
        # drain 决策放行后续请求 chunk)允许同实例多请求的 drain 决策
        # 乱序到达——从 qp 移除该已完成成员(排队深度口径 remaining_chunks
        # 求和不变,策略输入语义等价;FCFS 物理序由列车 chunk 串行化
        # 保证)。busy 复位移至 _finalize_completed_trains(列车核销)。
        if runtime not in state.qp:  # :1614-1615 的成员化等价
            raise RuntimeError("draining request is not in its prefill queue")
        state.prefill_remaining_chunks -= runtime.remaining_chunks
        runtime.remaining_chunks = 0
        state.qp.remove(runtime)  # :1620
        if state.qp or state.active_decode or state.pending_decode_ready:
            self._ready_frontier.add(state.index)  # §7.3:仍有排队工作
        else:
            self._ready_frontier.discard(state.index)
        # 全局 9 实例快照(顺序敏感决策点,离线 :1622-1627 同一构造)。
        # offline: face_scheduler.py
        has_prefill = [bool(instance.qp) for instance in self.instances]
        active_tokens = [
            [member.current_decode_token for member in instance.active_decode]
            for instance in self.instances
        ]
        selected, costs = select_decode_instance(  # :1627-1637
            topology=self.topology,
            graph=self.instance_graph,
            hardware=self.config.hardware,
            model=self.config.model,
            fixed_p_chunk=self.p_chunk,
            prefill_instance_index=state.index,
            has_prefill_work=has_prefill,
            decode_token_lengths=active_tokens,
            new_request_token_length=runtime.prefill_context_tokens,
            tie_counter=self._decode_tie_counter,
        )
        # P1(2026-08-31)选择期容量可行性过滤:decode 实例"选定之时"
        # (选择只发生一次,下方 :1637 落定前)对候选做只读预检,内存盲
        # 选择的阻塞几乎不再发生;准入一经落定仍遵守"绝不 remap"不变量
        # (:1086-1087,本过滤不违反)。详见 _filter_decode_selection。
        selected = self._filter_decode_selection(
            runtime, selected, costs, tick)
        runtime.decode_instance_index = selected  # :1637
        runtime.decode_candidates = costs  # :1638
        runtime.waiting_decode_admission = True  # :1639
        self.waiting_decode_admissions[selected].append(runtime)  # :1640
        self.decode_admission_dirty.add(selected)  # :1641
        # :1642-1647:同 tick 的后续 prefill 完成必须看到本 handoff 的排队
        # (离线事件序保留);容量仍可 defer,但绝不 remap。
        self._try_admit_waiting_decodes(tick)  # :1647
        # 阶段 3 感知账本:admitted 层排队类型更新(waiting_decode)。
        self._ledger_admit(
            runtime.request_id, tick,
            {"type": "waiting_decode", "instance_index": selected})

    def _on_decode_complete(self, request_id: str, tick: int) -> None:
        """离线 decode 完成分支(:1650-1675,decode_steps_remaining==0 路径):
        mark_complete + note_capacity_change + 快照。active_decode 出队与
        busy 复位移至 _finalize_completed_trains(拼 batch 改造:退出迭代
        在列车内先验已知,物理完成时刻 = exit 标记节点完成时刻)。下一次
        arrival 排程在 REQUEST_COMPLETE 边界(同 tick,run_variant_policy
        中紧随本方法)。

        offline: face_scheduler.py
        """
        runtime = self.runtime_by_request_id[request_id]
        state = self.instances[runtime.decode_instance_index]
        runtime.completion_ns = tick  # :1659
        self.completed_requests += 1  # :1660
        runtime.completion_evictions = self.kv_manager.mark_complete(  # :1661-1665
            runtime.session_id,
            tick,
            runtime.request_id,
        )
        self._note_capacity_change(  # :1666-1669
            state.index,
            *(record.victim_instance_index
              for record in runtime.completion_evictions),
        )
        completed_snapshot = self.kv_manager.session_snapshot(
            runtime.session_id)  # :1670
        if completed_snapshot is None:  # :1671-1672
            raise RuntimeError("completed session disappeared from KV manager")
        # B2(2026-09-06):快照状态字段换三态 location(local_hbm/
        # partial_hbm_remote/remote_memory)。
        runtime.kv_state_after_completion = completed_snapshot.location  # :1673
        runtime.kv_instance_after_completion = (  # :1674
            completed_snapshot.instance_index)
        runtime.hbm_after_completion = self.kv_manager.hbm_snapshots(
            state.index)  # :1675
        self.log_decision(
            {"kind": "completion", "request_id": runtime.request_id,
             "priority": 0},
            tick,
            decision={
                "kv_state_after_completion": runtime.kv_state_after_completion,
                "kv_instance_after_completion":
                    runtime.kv_instance_after_completion,
                "completion_evictions": [
                    _eviction_dict(record)
                    for record in runtime.completion_evictions
                ],
                # B3(契约 §3,sh :1319-1326):完成段逐出契约行 + hop 摘要
                # (face 完成路径零逐出,行恒空——字段在场性由 adapter/
                # watermark 消费侧要求;命名避开 face 既有
                # completion_evictions 8 字段行)。
                "completion_eviction_transfers": _eviction_transfer_rows(
                    runtime.completion_evictions),
                "transfer_hop_bytes": _eviction_hop_rows(
                    runtime.completion_evictions),
            },
        )
        # M4 核销即删(2026-08-23):请求完成后其 KV 迁移/逐出对象、decode
        # 候选代价与准入 HBM 快照等胖字段再无读者(全仓 grep 证实:
        # admission_evictions 最后读者=准入决策行、completion_evictions=
        # 上方完成决策行、decode_candidates/prefill_decode_transfer/
        # decode_target_evictions=decode 准入链;history_transfer/
        # hbm_before_request/hbm_after_completion 零读者;_on_request_
        # complete 只读 next_request/interval)——置空即删,防随完成
        # 请求数线性常驻。face 差异点:字段集与 sh_3.0 母本不同(face
        # 无三分 evictions/prefill_instance_loads/drain_block_ends,
        # 另有 decode_candidates 候选代价全记录与 hbm 快照)。
        runtime.decode_candidates = ()
        runtime.admission_evictions = ()
        runtime.decode_target_evictions = ()
        runtime.completion_evictions = ()
        runtime.history_transfers = ()
        runtime.prefill_decode_transfer = None
        runtime.hbm_before_request = None
        runtime.hbm_after_completion = None

    def _on_request_complete(self, request_id: str, tick: int) -> None:
        """离线 decode 完成分支的下一次 arrival 排程(:1676-1681):
        now + 该 turn 的 inter_request_interval_ns 注册未来 alarm(向
        ingress,不再 push 到事件堆)。

        offline: face_scheduler.py
        """
        runtime = self.runtime_by_request_id[request_id]
        # §7.3:_runtime_index O(1) 定位。
        following = self.next_request[self._runtime_index[runtime.request_id]]
        # B3(sh ``_emit_completion`` :1341-1355 的 face 落点):REQUEST_
        # COMPLETE 边界登记下一 turn 的 pending location 链 + 下一同
        # session turn 的预置 location(准入时经 plan 消费)。完成路径
        # 零逐出,location 即完成快照;后续逐出经 sync_pending_history_
        # after_evictions 覆盖 deferred 链。
        self.graph.note_request_complete(
            runtime.session_id,
            None if following is None else following.request_id,
            runtime.kv_state_after_completion)
        if following is not None:
            following.kv_pending_location = runtime.kv_state_after_completion
        # B3:逐出块末/action 计数账本条目核销即删(sh M4 同款)。
        self.graph.retire_request_state(runtime.request_id)
        if following is not None:
            interval = self._interval_ns(following)  # :1678-1680
            # :1681 push_event(now + interval, 1, "arrival", following) ->
            # 在线等价:future alarm。
            self._batch["future_alarms"].append({
                "arrival_world_ns": tick + interval,
                "envelope": {
                    "request_id": following.request_id,
                    "session_id": following.session_id,
                    "turn_index": following.turn_index,
                    "prefill_length": following.prefill_length,
                    "decode_length": following.decode_length,
                    "inter_request_interval_ns": interval,
                },
            })
        else:
            # terminal turn 的 completion gate 不会再被下一次 admission
            # 消费；REQUEST_COMPLETE 是所有图边已发射后的安全回收点。
            self.graph.retire_completion_gate(runtime.session_id)
            released_instance_index = self.kv_manager.retire_terminal_session(
                runtime.session_id,
                tick,
                runtime.request_id,
            )
            # Retirement may free local HBM after the completion snapshot and
            # graph batch have been consumed.  Publish that capacity change
            # without consulting the now-deleted session state.
            self._note_capacity_change(released_instance_index)
        # REQUEST_COMPLETE 边界后该 runtime 及其索引再无读者。释放 map、
        # list slot 和已消费的 next link，避免完成请求仍被预构建链持有。
        self.runtime_by_request_id.pop(request_id, None)
        index = self._runtime_index.pop(request_id, None)
        if index is None:
            raise RuntimeError(
                "completed request lost runtime index {!r}".format(request_id))
        self.next_request[index] = None
        self.runtimes[index] = None

    # ------------------------------------------------------- arrival heap --

    def _push_arrival(self, arrival: dict, tick: int) -> None:
        """离线 :1336-1339 push_event(time_ns, priority=1, kind="arrival")。
        字段形状与离线一致;键含 queue_index——同 tick 到期项按冻结队列序
        稳定弹出,与 C++ 序列化的 arrivals 冻结队列序一致。
        """
        runtime = self.runtime_by_request_id[arrival["request_id"]]
        heapq.heappush(
            self.arrival_heap,
            (tick, 1, runtime.queue_index, self._sequence, "arrival",
             runtime))
        self._sequence += 1

    def _drain_arrival_heap(self, tick: int) -> None:
        """离线 :1592-1596 同 tick 批 + 按 (priority, sequence) 排序。只消费
        tick <= current_tick 的到期项(在线单 tick 单次交付下堆内全部为当前
        批 push 的项;未来 alarm 的到期事件由 C++ 在到期 tick 交付)。
        """
        while self.arrival_heap and self.arrival_heap[0][0] <= tick:
            _, _, _, _, kind, payload = heapq.heappop(self.arrival_heap)
            if kind != "arrival":
                raise RuntimeError("arrival heap contains {!r}".format(kind))
            self._profile_scan()  # §7.3:堆弹出条目(到期事件)
            self._on_arrival(payload, tick)

    # ------------------------------------------------------------- 准入 --

    def _admit_pass(self, tick: int) -> None:
        """离线 start_ready_iterations(:1517-1590)的排队/准入部分(拼
        batch 改造,2026-08-22:计时部分由实例迭代列车承载,列车体即
        离线"一次混合迭代(1 prefill chunk + 全部 active decode 各 1
        token)"的连续序列折叠)。先 try_admit_waiting_decodes(:1519),
        再逐就绪实例冻结并发射下一列车(_plan_and_emit_trains:统一实例
        qp 头部 chunk 与 active_decode 成员混拼,每迭代 ≤1 chunk,批成员
        只在列车边界变化;prefill 队首优先,准入失败时本列车退化为纯
        decode——离线 :1520-1588 serve 序的同构映射)。

        offline: face_scheduler.py
        """
        self._try_admit_waiting_decodes(tick)  # :1519
        self._plan_and_emit_trains(tick)

    def _try_admit_prefill(self, runtime, now_ns: int) -> bool:
        """离线 try_admit_prefill(:1382-1453),逐行对应;无逐 chunk 计时
        账本(聚合粒度)。KV 准入链不参与 Roofline 候选成本计算。

        offline: face_scheduler.py
        """
        if runtime.admitted_prefill:  # :1384-1385
            return True
        if runtime.prefill_instance_index is None:  # :1386-1387
            raise RuntimeError("prefill admission lost its fixed mapping")
        target_instance = runtime.prefill_instance_index  # :1388
        # P0-1(2026-08-31):dirty 例外放行——阻塞曾把本实例标记为
        # prefill_admission_dirty 时,即使纪元未推进也在本决策边界重试
        # (镜像 decode 侧 decode_admission_dirty 的重试门语义)。
        if (runtime.prefill_attempt_epoch  # :1389
                == self.capacity_epoch[target_instance]
                and target_instance not in self.prefill_admission_dirty):
            return False
        runtime.prefill_attempt_epoch = (  # :1390-1391
            self.capacity_epoch[target_instance])
        # P0-1:尝试开始即清本实例 dirty 门(阻塞分支会重新置位;
        # 与 decode 侧 ready_targets 扫描头清 dirty :1356-1358 同构)。
        self.prefill_admission_dirty.discard(target_instance)
        before_snapshot = self.kv_manager.session_snapshot(  # :1392
            runtime.session_id)
        runtime.history_cache_state_before = (  # :1393-1395
            # B2:三态 location 值域(local_hbm/partial_hbm_remote/
            # remote_memory);ABSENT 语义不变。
            "ABSENT" if before_snapshot is None else before_snapshot.location
        )
        # B3:准入时点快照对象随 runtime 走(构图器 partial 流水与位置
        # 对账消费;prepare_history 会改会话态,先捕获)。
        runtime.history_location_before = before_snapshot
        runtime.hbm_before_request = self.kv_manager.hbm_snapshots(  # :1396
            runtime.prefill_instance_index)
        decision = self.kv_manager.prepare_history(  # :1397-1404
            runtime.session_id,
            runtime.prefill_instance_index,
            runtime.history_tokens_before,
            now_ns,
            runtime.request_id,
            required_context_tokens=runtime.prefill_context_tokens,
        )
        # B3:逐出 KVTransfer 账本 + pending location 镜像(决策时点同步,
        # 跨阻塞 attempt 累积——每次 attempt 的逐出都是真实 mutation)。
        prepare_transfers = tuple(
            record.transfer for record in decision.evictions
            if record.transfer is not None)
        runtime.history_evictions = tuple(
            (*runtime.history_evictions, *prepare_transfers))
        self.graph.sync_pending_history_after_evictions(prepare_transfers)
        runtime.admission_evictions = tuple(
            (*runtime.admission_evictions, *decision.evictions))  # :1405
        if decision.admission_blocked:  # :1406-1412
            if decision.evictions:
                self._note_capacity_change(
                    target_instance,
                    *(record.victim_instance_index
                      for record in decision.evictions),
                )
            # B3:阻塞 attempt 的逐出是真实 mutation(ensure_physical_fit
            # 失败不回滚)——remote_store 物理链随当前批发射(节点上下文
            # = 本请求 (prefill, 0),in-flight 资格门放行;无 watch 纯 mem
            # 批不驱动决策交付——本批伴随当前决策边界的其余发射)。
            if prepare_transfers:
                self.graph.emit_eviction_actions(
                    self._plan_dict(runtime), prepare_transfers)
            # Liveness fix (P0-1, 2026-08-31): 与 decode 侧同源——"无逐出
            # 的阻塞"不推进 capacity_epoch,纪元门对本实例永久关闭;隔离
            # 时间线没有邻居完成事件重开它(prefill 卡死形态,总文档 §4
            # P0-1)。阻塞即置 dirty,下一个决策边界(任一交付的
            # _admit_pass -> _plan_and_emit_trains -> 本方法)必然重试。
            self.prefill_admission_dirty.add(target_instance)
            return False
        runtime.history_action = decision.action  # :1413
        runtime.history_source_instance_index = decision.source_instance_index  # :1414
        # B3:契约 §3 口径(sh :1466-1472)——传输字节只计物理搬运,
        # local_hit 零字节(值 = Σ transfers.total_bytes excluding local_hit)。
        runtime.history_transfer_bytes = sum(
            transfer.total_bytes for transfer in decision.transfers
            if transfer.kind != "local_hit")
        runtime.history_recompute_tokens = decision.recompute_tokens  # :1416
        # B2:KVTransfer 对象组由管理器构造(kind 载体,契约 §4);两段式
        # 恢复(前缀 noc_migrate + 后缀 remote_load)两段同 holder 承载,
        # 一条 prefill 决策记录序列化(不得拆两条)。
        runtime.history_transfers = decision.transfers
        growth = self.kv_manager.grow_prefill(  # :1430-1435
            runtime.session_id,
            runtime.prefill_context_tokens,
            now_ns,
            runtime.request_id,
        )
        if not growth.admitted:  # :1436-1439
            raise RuntimeError(
                "prefill growth lost its successful admission reservation")
        # B3:prefill 增长逐出账本(grow_prefill 成功路径的逐出)。
        growth_transfers = tuple(
            record.transfer for record in growth.evictions
            if record.transfer is not None)
        runtime.prefill_evictions = tuple(
            (*runtime.prefill_evictions, *growth_transfers))
        self.graph.sync_pending_history_after_evictions(growth_transfers)
        runtime.admission_evictions = tuple(  # :1440
            (*runtime.admission_evictions, *growth.evictions))
        old_remaining_chunks = runtime.remaining_chunks
        runtime.remaining_chunks = (  # :1442-1445(快照素材;在线保持该值
            # 直到 drain,见类 docstring 的刻意差异清单;B2 去掉
            # ceil(history_recompute/p_chunk) 项——RECOMPUTE 已从规划删除)
            math.ceil(runtime.prefill_length / self.p_chunk)
        )
        self.instances[target_instance].prefill_remaining_chunks += (
            runtime.remaining_chunks - old_remaining_chunks)
        runtime.admitted_prefill = True  # :1446
        self._note_capacity_change(  # :1447-1452
            target_instance,
            decision.source_instance_index,
            *(record.victim_instance_index
              for record in decision.evictions),
            *(record.victim_instance_index for record in growth.evictions),
        )
        return True

    def _try_admit_waiting_decodes(self, now_ns: int) -> None:
        """离线 try_admit_waiting_decodes(:1455-1515),逐行对应(含 face 的
        阻塞重排队语义:admission_blocked -> queue.append + continue)。

        offline: face_scheduler.py
        """
        ready_targets = tuple(  # :1456-1465
            instance_index
            for instance_index, queue in sorted(
                self.waiting_decode_admissions.items())
            if queue
            and (
                instance_index in self.decode_admission_dirty
                or self.decode_admission_epoch[instance_index]
                != self.capacity_epoch[instance_index]
            )
        )
        for target_instance in ready_targets:  # :1466
            self.decode_admission_epoch[target_instance] = (  # :1467
                self.capacity_epoch[target_instance])
            self.decode_admission_dirty.discard(target_instance)  # :1468
            queue = self.waiting_decode_admissions[target_instance]  # :1469
            pending_count = len(queue)  # :1470
            for _ in range(pending_count):  # :1471
                runtime = queue.popleft()  # :1472
                if not runtime.waiting_decode_admission:  # :1474-1475
                    continue
                if runtime.decode_instance_index != target_instance:  # :1476-1477
                    raise RuntimeError(
                        "decode admission target queue was corrupted")
                move = self.kv_manager.move_prefill_to_decode(  # :1478-1484
                    runtime.session_id,
                    target_instance,
                    now_ns,
                    runtime.request_id,
                    final_context_tokens=runtime.final_context_tokens,
                )
                runtime.decode_target_evictions = tuple(  # :1485-1487
                    (*runtime.decode_target_evictions, *move.evictions)
                )
                # B3:decode 逐出 KVTransfer 账本(列车 joiner 段发射;跨
                # 阻塞 attempt 累积)+ pending location 镜像。
                move_transfers = tuple(
                    record.transfer for record in move.evictions
                    if record.transfer is not None)
                runtime.decode_evictions = tuple(
                    (*runtime.decode_evictions, *move_transfers))
                self.graph.sync_pending_history_after_evictions(move_transfers)
                if move.admission_blocked:  # :1488-1495
                    if move.evictions:
                        self._note_capacity_change(
                            target_instance,
                            *(record.victim_instance_index
                              for record in move.evictions),
                        )
                    # B3:阻塞 attempt 的逐出是真实 mutation——物理链随
                    # 当前批发射(上下文 = 本请求 (decode, 1);本请求
                    # prefill 已 drain,prefill_drained 资格门放行)。
                    if move_transfers:
                        self.graph.emit_eviction_actions(
                            self._plan_dict(runtime), move_transfers)
                    # Liveness fix (P0-1, 2026-08-31): 阻塞即重开本实例重试门。
                    # 原语义下"无逐出的阻塞"不推进 capacity_epoch,重试门
                    # (上方 ready_targets 的 dirty/epoch 条件)在本轮关闭;
                    # 隔离时间线(计时外包给 C++ 图)没有邻居完成事件来重开
                    # 它,被阻塞会话保持 ACTIVE 钉住 KV,阻塞固化成舰队级
                    # 循环等待。有逐出的阻塞本就经 _note_capacity_change
                    # 重开(幂等,双通道无害)。效果:ready_targets 扫描开始
                    # 时同步纪元+清 dirty(:1356-1358),扫描内再次置 dirty
                    # => 下一个决策边界(任一交付的 _admit_pass)必然重试
                    # 本实例队列;残余环由 P0-2 fail-loud 兜底(总文档 §4)。
                    self.decode_admission_dirty.add(target_instance)
                    queue.append(runtime)  # :1494(阻塞重排队,face 语义)
                    continue  # :1495
                runtime.prefill_decode_transfer = move.transfer  # :1496
                growth = self.kv_manager.grow_decode(  # :1497-1502
                    runtime.session_id,
                    runtime.final_context_tokens,
                    now_ns,
                    runtime.request_id,
                )
                if not growth.admitted:  # :1503-1504
                    raise RuntimeError(
                        "decode growth lost its successful target admission")
                runtime.decode_target_evictions = tuple(  # :1505-1507
                    (*runtime.decode_target_evictions, *growth.evictions)
                )
                # B3:decode 增长逐出账本 + pending location 镜像。
                growth_transfers = tuple(
                    record.transfer for record in growth.evictions
                    if record.transfer is not None)
                runtime.decode_evictions = tuple(
                    (*runtime.decode_evictions, *growth_transfers))
                self.graph.sync_pending_history_after_evictions(
                    growth_transfers)
                # 拼 batch 改造(§3.2 KV 就绪栅栏):decode 准入(KV 迁移
                # 决策 + 容量增长)在此完成,成员进入 pending_decode_ready,
                # 待加入 decode 实例的下一列车(3000 迁移随加入列车发射,
                # 物理先于列车体;迁移列车中途完成的也只能等下列车边界)。
                decode_state = self.instances[target_instance]  # :1508
                decode_state.pending_decode_ready.append(runtime)
                if decode_state.in_flight_train is None:
                    self._ready_frontier.add(target_instance)  # §7.3
                runtime.waiting_decode_admission = False  # :1509
                # decode 决策记录 + assignment(拼 batch 改造:decode 段发射
                # 移至加入列车,即 _plan_and_emit_trains → emit_iteration_
                # train;DECODE_COMPLETION watch 由列车 exit 标记承载)。
                self._batch["assignments"].append({
                    "request_id": runtime.request_id,
                    "prefill_instance_index": runtime.prefill_instance_index,
                    "decode_instance_index": runtime.decode_instance_index,
                })
                self.log_decision(
                    {"kind": "decode", "request_id": runtime.request_id,
                     "priority": 0},
                    now_ns,
                    decision={
                        "decode_instance_index": runtime.decode_instance_index,
                        # face 独有(B2 oracle 证据):decode 候选代价全记录。
                        "decode_candidates": [
                            _candidate_dict(candidate)
                            for candidate in runtime.decode_candidates
                        ],
                        # B2:KVTransfer 换 kind 载体(契约 §4),行结构改
                        # _transfer_rows(sh_2.0 契约 §3 行字段)。
                        "prefill_decode_transfer": _transfer_rows(
                            [runtime.prefill_decode_transfer]),
                        # B3(契约 §3 decode 侧):decode_evictions 同款行
                        # 结构 + transfer_hop_bytes 逐传输对象 noc 路由
                        # 摘要(sh :1233-1240;行源 = decode_evictions 账本,
                        # 含跨阻塞 attempt 累积的全集)。
                        "decode_evictions": _transfer_rows(
                            runtime.decode_evictions),
                        "transfer_hop_bytes": _transfer_hop_rows(
                            list(runtime.decode_evictions)
                            + [runtime.prefill_decode_transfer]),
                        # WP4 hopbytes 观测(WP9 任务 4,2026-08-26):prefill
                        # →decode KV 迁移(列车头 3000 类)的逐 shard 物理
                        # NoC 跳数(与构图发射同源 XY 路由,只读导出;
                        # [] = 同实例零节点迁移)。A/B 剥离清单字段。
                        "kv_noc_hops": self._kv_noc_hops(
                            runtime.prefill_instance_index,
                            runtime.decode_instance_index),
                        # 问题 2A 顺带修复(2026-09-05,与 wscllm 同名同构):
                        # decode 准入逐出快照序列化——发射时点必在本请求
                        # 全部 append(:1406-1408/:1439-1441,含跨阻塞
                        # attempt 累积)之后,快照即全集;发射后立即置空
                        # 核销(见下方),防未来工具重读 decode 行时与
                        # prefill 行的准入预占快照双计。A/B 剥离清单字段。
                        "decode_target_evictions": [
                            _eviction_dict(record)
                            for record in runtime.decode_target_evictions
                        ],
                    },
                )
                # 问题 2A 顺带修复:decode 决策核销——先序列化后置空,
                # 镜像 completion 对 completion_evictions 的语义(:1171
                # 的 M4 置空保留为幂等兜底);每请求恰一条 decode 行,
                # 置空无双计风险。
                runtime.decode_target_evictions = ()
                # 阶段 3 感知账本:admitted 层排队类型更新(active_decode;
                # 类型名保留 drain 时点口径)。
                self._ledger_admit(
                    runtime.request_id, now_ns,
                    {"type": "active_decode", "instance_index": target_instance})
                self._note_capacity_change(  # :1510-1515
                    target_instance,
                    move.source_instance_index,
                    *(record.victim_instance_index
                      for record in move.evictions),
                    *(record.victim_instance_index
                      for record in growth.evictions),
                )

    # ------------------------------------------------------------- 发射 --

    def _emit_admission(self, runtime, tick: int) -> None:
        """准入动作发射 + 决策/账本记录(拼 batch 改造,2026-08-22:
        gates/history 迁移/readiness 屏障经 emit_admission_batch 发射;
        prefill 主体(recompute 段 + 当前段 chunk)移入实例迭代列车,
        PREFILL_DRAIN watch 不再在此注册——移至覆盖其最后 chunk 的列车
        drain 标记)。

        offline: face_scheduler.py(发射对象为整段而非 chunk → 列车)
        """
        plan = self._plan_dict(runtime)
        self.graph.emit_admission_batch(plan)
        self._note_emitted(runtime.request_id, STAGE_PREFILL)
        self._ledger_issue(runtime.request_id, tick, STAGE_PREFILL,
                           runtime.prefill_instance_index)
        # face:统一实例的 decode 在 PREFILL_DRAIN 边界才选定;assignment
        # 在 decode 准入时一次性携带双索引(C++ 校验器要求两索引非负,
        # 共享机制层不改;prefill 边界不产 assignment 条目)。
        self.log_decision(
            {"kind": "prefill", "request_id": runtime.request_id,
             "priority": 0},
            tick,
            decision={
                "prefill_instance_index": runtime.prefill_instance_index,
                "prefill_assignment_key": list(runtime.prefill_assignment_key),
                "estimated_arrival_ns": runtime.estimated_arrival_ns,
                "effective_prefill_tokens": (
                    runtime.prefill_length + runtime.history_recompute_tokens
                ),
                "history_action": runtime.history_action,
                "history_cache_state_before":
                    runtime.history_cache_state_before,
                "history_source_instance_index":
                    runtime.history_source_instance_index,
                "history_transfer_bytes": runtime.history_transfer_bytes,
                "history_recompute_tokens": runtime.history_recompute_tokens,
                # ---- B3(契约 §3,sh :1500-1541 逐字对齐)----
                # 逐出 victim/bytes 逐条序列化 + partial 两段式恢复逐段
                # 对象(一条 prefill 记录承载——hbm_watermark"同请求同
                # kind 两次即 fail");准入时点的会话历史位置三态序列化
                # (kv_cache_adapter S2 分支映射 kv_hit_state);hopbytes
                # 逐传输对象 noc 路由摘要。纯输出字段,进 A/B 对拍剥离
                # 清单。
                "history_eviction_count": len(runtime.history_evictions),
                "history_evictions": _transfer_rows(
                    runtime.history_evictions),
                "prefill_evictions": _transfer_rows(
                    runtime.prefill_evictions),
                "history_transfers": _transfer_rows(
                    runtime.history_transfers),
                "history_location_before": (
                    None if runtime.history_location_before is None
                    else runtime.history_location_before.location),
                "history_location_before_instance_index": (
                    None if runtime.history_location_before is None
                    else runtime.history_location_before.instance_index),
                "history_resident_prefix_layers": (
                    None if runtime.history_location_before is None
                    else runtime.history_location_before.resident_prefix_layers),
                "transfer_hop_bytes": _transfer_hop_rows(
                    list(runtime.history_evictions)
                    + list(runtime.history_transfers)
                    + list(runtime.prefill_evictions)),
                # WP4 hopbytes 观测(WP9 任务 4,2026-08-26):history KV
                # 迁移的逐 shard 物理 NoC 跳数(与构图发射同源 XY 路由,
                # 只读导出;[] = 无物理传输)。A/B 剥离清单字段。
                "history_noc_hops": self._kv_noc_hops(
                    runtime.history_source_instance_index,
                    runtime.prefill_instance_index),
                "admission_evictions": [
                    _eviction_dict(record)
                    for record in runtime.admission_evictions
                ],
                # 问题 2A 顺带修复(2026-09-05):原为硬编码 [](face 准入
                # 链无 reserve_request_capacity 调用,本发射时点
                # decode_target_evictions 恒为初始 (),运行时数值不变);
                # 改为真实序列化,防未来准入链变化后 prefill 行静默失真。
                "decode_target_evictions": [
                    _eviction_dict(record)
                    for record in runtime.decode_target_evictions
                ],
            },
        )

    # ------------------------------------------------------------- 助手 --

    def _kv_noc_hops(self, source_instance_index,
                     target_instance_index) -> list[int]:
        """WP4 hopbytes 观测(WP9 任务 4,2026-08-26):KV 迁移发射的逐
        TP shard 物理 NoC 跳数——与构图发射同源的 XY 路由口径
        (generate_face_trace._xy_route,经 _paired_transfer 内部同函数
        同输入计算;此处只读导出,不改路由/不改发射/不进策略判据)。

        返回 [] = 无物理传输(同实例迁移零节点/无迁移动作/索引缺失);
        非空 = per-shard 跳数(与 _paired_transfer 返回 routes[].noc_hops
        同值)。face 9 实例为同构 3x2 rank 块,逐 shard 跳数恒一致;
        通例保留逐 shard 表(Hop-Bytes = Σ shard_bytes × noc_hops,
        shard_bytes 见 kv_actions 事件流)。"""
        if (source_instance_index is None
                or target_instance_index is None
                or source_instance_index == target_instance_index):
            return []
        source = self.topology.instances[source_instance_index]
        target = self.topology.instances[target_instance_index]
        if len(source.ranks) != len(target.ranks):
            raise RuntimeError(
                "FACE direct KV shard pairing requires equal TP degree")
        return [
            len(_xy_route(self.config.hardware, source_rank, target_rank)) - 1
            for source_rank, target_rank in zip(source.ranks, target.ranks)
        ]

    def _plan_dict(self, runtime) -> dict:
        """graph_batch_builder 消费的 plan 字段(request 事实 + 在线决策)。

        B3:三态 KV 发射面——history_location_before(准入时点快照对象)、
        history_transfers(KVTransfer 组)、history/prefill 逐出账本、
        kv_location_after_completion(上一 turn 完成时预置的 pending
        location 链初值,turn>0 消费)。"""
        return {
            "request_id": runtime.request_id,
            "session_id": runtime.session_id,
            "turn_index": runtime.turn_index,
            "queue_index": runtime.queue_index,
            "prefill_instance_index": runtime.prefill_instance_index,
            "decode_instance_index": runtime.decode_instance_index,
            "history_action": runtime.history_action,
            "history_source_instance_index":
                runtime.history_source_instance_index,
            "history_transfer_bytes": runtime.history_transfer_bytes,
            "history_recompute_tokens": runtime.history_recompute_tokens,
            "history_tokens_before": runtime.history_tokens_before,
            "prefill_context_tokens": runtime.prefill_context_tokens,
            "prefill_length": runtime.prefill_length,
            "decode_length": runtime.decode_length,
            # ---- B3 三态 KV 发射面 ----
            "history_location_before": runtime.history_location_before,
            "history_transfers": runtime.history_transfers,
            "history_evictions": runtime.history_evictions,
            "prefill_evictions": runtime.prefill_evictions,
            "kv_location_after_completion": runtime.kv_pending_location,
        }

    def _queue_snapshots(self):
        """离线 queue_snapshot(:1371-1376):全部 9 统一实例(无角色过滤),
        remaining_chunks = qp 成员之和,last_arrival_ns = 实例最近到达。

        offline: face_scheduler.py / 1689
        """
        if self._queue_aggregate_verify:
            for state in self.instances:
                recomputed = sum(
                    member.remaining_chunks for member in state.qp)
                if recomputed != state.prefill_remaining_chunks:
                    raise RuntimeError(
                        "prefill queue aggregate drift on instance {}: "
                        "ledger={} recomputed={}".format(
                            state.index, state.prefill_remaining_chunks,
                            recomputed))
        return tuple(
            PrefillQueueSnapshot(
                instance_index=state.index,
                remaining_chunks=state.prefill_remaining_chunks,
                last_arrival_ns=state.last_arrival_ns,
            )
            for state in self.instances
        )

    # ------------------------------------------------- P1 选择期容量预检 --

    def _probe_instance_capacity(self, instance_index: int,
                                 required_shards) -> bool:
        """P1(2026-08-31)只读容量预检:候选实例逐 rank 的 remaining_bytes
        (hbm_snapshots,相对 rank 序与分片口径对齐)是否都 >= 需求分片。

        零副作用(红线,总文档 §5 不变量 5):只调用 hbm_snapshots 只读
        快照与纯函数,严禁触碰 ensure_physical_fit / reserve_* / 逐出 /
        放置(ensure_physical_fit 边逐出边检查、失败不回滚,只能进最终
        commit,不进探测循环)。保守剩余容量口径(remaining-only,不含
        冷会话逐出信用——B2 三态化后候选池换私有 _completed_*_
        candidates,口径裁决不变:预检不换 reclaimable,误判不可行时原选
        择保持,由 P0-1 在下一边界重试兜底;策略 §5 裁决 1)。
        """
        return all(
            required <= snapshot.remaining_bytes
            for required, snapshot in zip(
                required_shards,
                self.kv_manager.hbm_snapshots(instance_index))
        )

    def _filter_decode_selection(self, runtime, selected, costs,
                                 tick: int) -> int:
        """P1(2026-08-31)decode 选择期容量可行性过滤:在 decode 实例
        选定之时(select 只发生一次,调用点 :1081 尚未落定)对全部候选
        做只读预检——这不是"准入后改实例",不违反"绝不 remap"不变量
        (:1086-1087:容量仍可 defer,但绝不 remap)。

        需求口径严格镜像 move_prefill_to_decode(session_kv_manager.py,
        只读版):final_shards = kv_cache_shard_bytes_for_tokens(model,
        final_context_tokens, tp);候选 c 的需求 = final_shards −(会话
        本地驻留(location ∈ {local_hbm, partial_hbm_remote})且已在 c 上
        的 local_shard_bytes),即 source==target 按增量、否则全量(双
        驻留:源副本仍占源)。B2:三态口径——增量信用 = 驻留前缀层段
        (local_shard_bytes),PARTIAL 会话只按前缀折减(镜像管理器账本)。

        选择规则(决策确定性,总文档 §4 P1):
          - 原 selected 可行 -> 原样返回:决策日志与原实现零差异,
            tie_counter 语义零扰动(select_decode_instance 本体不动);
          - 原 selected 不可行 -> 可行集内按 per_die_delta_ns 最小重选
            (并列取最小 instance_index,与 select_decode_instance 的
            最小 delta 口径一致);
          - 全部不可行 -> 保持原 selected(进入 P0-1 的阻塞-重试路径,
            探测不 remap、不产生任何容量副作用)。
        仅在过滤实际介入(原选不可行)时记一条决策日志辅助字段
        admission_probe(原选/改选原因),供 A/B 对拍剥离说明。
        """
        final_shards = kv_cache_shard_bytes_for_tokens(
            self.config.model, runtime.final_context_tokens,
            self.kv_manager.tp_degree)
        session_shards = self.kv_manager.session_snapshot(
            runtime.session_id)
        feasible_costs = []
        for cost in costs:
            required = final_shards
            if (session_shards is not None
                    and session_shards.location in (LOCAL_HBM, PARTIAL_HBM_REMOTE)
                    and session_shards.instance_index == cost.instance_index):
                # source==target 增量口径(镜像 move_prefill_to_decode 的
                # existing_target;B2:驻留前缀层段)。
                required = tuple(
                    want - have for want, have in zip(
                        final_shards, session_shards.local_shard_bytes))
            if self._probe_instance_capacity(cost.instance_index, required):
                feasible_costs.append(cost)
        if any(cost.instance_index == selected for cost in feasible_costs):
            return selected  # 原选可行:决策零扰动
        probe = {
            "admission_probe": {
                "original_instance_index": selected,
                "feasible_candidates": [
                    cost.instance_index for cost in feasible_costs],
                "outcome": "",
            }
        }
        if feasible_costs:
            replacement = min(
                feasible_costs,
                key=lambda cost: (cost.per_die_delta_ns,
                                  cost.instance_index))
            probe["admission_probe"]["outcome"] = "reselected_infeasible_first"
            probe["admission_probe"]["selected_instance_index"] = (
                replacement.instance_index)
            self.log_decision(
                {"kind": "decode_admission_probe",
                 "request_id": runtime.request_id, "priority": 0},
                tick, decision=probe)
            return replacement.instance_index
        probe["admission_probe"]["outcome"] = "kept_all_infeasible"
        probe["admission_probe"]["selected_instance_index"] = selected
        self.log_decision(
            {"kind": "decode_admission_probe",
             "request_id": runtime.request_id, "priority": 0},
            tick, decision=probe)
        return selected  # 全不可行:保持原选,阻塞交给 P0-1 重试

    def _filter_prefill_selection(self, runtime, snapshots, selected,
                                  tick: int) -> int:
        """P1(2026-08-31)prefill 选择期容量可行性过滤:包装
        select_prefill_instance(:1690)——到达时刻对每个实例快照对应的
        实例做只读预检。

        需求 = kv_cache_shard_bytes_for_tokens(model,
        prefill_context_tokens, tp) 的全上下文保守口径(不区分会话已在/
        不在目标实例,宁可错杀不可漏杀;与 prepare_history 的
        source==target 折减口径相比偏保守,差异只影响原本会阻塞的路径,
        执行文档 §2.5 偏离记录)。零副作用约束与 decode 侧相同。

        选择规则(决策确定性):原 selected 幸存 -> 原样返回(决策零
        扰动);幸存集非空 -> 按原 ordering_key 取 min(与
        select_prefill_instance 同键);幸存集空 -> 保持原选择(进入
        阻塞-重试路径)。仅在过滤实际介入时记一条 admission_probe
        决策日志辅助字段。
        """
        required_shards = kv_cache_shard_bytes_for_tokens(
            self.config.model, runtime.prefill_context_tokens,
            self.kv_manager.tp_degree)
        survivors = [
            snapshot for snapshot in snapshots
            if self._probe_instance_capacity(
                snapshot.instance_index, required_shards)
        ]
        if any(snapshot.instance_index == selected for snapshot in survivors):
            return selected  # 原选可行:决策零扰动
        probe = {
            "admission_probe": {
                "original_instance_index": selected,
                "feasible_candidates": [
                    snapshot.instance_index for snapshot in survivors],
                "outcome": "",
            }
        }
        if survivors:
            replacement = min(survivors,
                              key=lambda snapshot: snapshot.ordering_key)
            probe["admission_probe"]["outcome"] = "reselected_infeasible_first"
            probe["admission_probe"]["selected_instance_index"] = (
                replacement.instance_index)
            self.log_decision(
                {"kind": "prefill_admission_probe",
                 "request_id": runtime.request_id, "priority": 0},
                tick, decision=probe)
            return replacement.instance_index
        probe["admission_probe"]["outcome"] = "kept_all_infeasible"
        probe["admission_probe"]["selected_instance_index"] = selected
        self.log_decision(
            {"kind": "prefill_admission_probe",
             "request_id": runtime.request_id, "priority": 0},
            tick, decision=probe)
        return selected  # 幸存集空:保持原选,阻塞交给 P0-1 重试

    def _note_capacity_change(self, *instance_indexes) -> None:
        """离线 note_capacity_change(:1364-1369)。"""
        for instance_index in set(instance_indexes):
            if instance_index is not None:
                self.capacity_epoch[instance_index] += 1

    def _note_instance_ready(self, instance_index: int) -> None:
        """§7.3 ready frontier:实例非忙(无在飞列车)且有排队工作 -> 就绪集。"""
        if self.instances[instance_index].in_flight_train is None:
            self._ready_frontier.add(instance_index)

    def _sensing_ready_view(self) -> dict:
        """阶段 7 §10.1 ready 层边界视图:ready frontier 实例队列中等待
        服务的 request(face 统一实例:qp 与 active_decode/pending_decode_
        ready 皆是排队工作)。查询/审计输入,不进策略判据。"""
        detail = []
        for instance_index in sorted(self._ready_frontier):
            state = self.instances[instance_index]
            for runtime in state.qp:
                detail.append({
                    "request_id": runtime.request_id,
                    "stage": STAGE_PREFILL,
                    "instance_index": instance_index,
                    "admitted_tick": self.ledger_admitted.get(
                        runtime.request_id, {}).get("admitted_tick"),
                })
            for runtime in state.active_decode:
                detail.append({
                    "request_id": runtime.request_id,
                    "stage": STAGE_DECODE,
                    "instance_index": instance_index,
                    "admitted_tick": self.ledger_admitted.get(
                        runtime.request_id, {}).get("admitted_tick"),
                })
            for runtime in state.pending_decode_ready:
                detail.append({
                    "request_id": runtime.request_id,
                    "stage": STAGE_DECODE,
                    "instance_index": instance_index,
                    "admitted_tick": self.ledger_admitted.get(
                        runtime.request_id, {}).get("admitted_tick"),
                })
        return {"ready_count": len(detail), "detail": detail}

    def _interval_ns(self, runtime) -> int:
        spec = self.config.request_queue[runtime.queue_index]
        return spec.inter_request_interval_ns

    # --------------------------------------------------------------- 收尾 --

    def verify_run_end(self) -> None:
        """基类协议校验之上,叠加离线 :1699-1705 的收尾断言与 §7.3 结束
        审计(arrival heap / ready frontier 全空)。"""
        super().verify_run_end()
        if self.completed_requests != self.expected_request_count:
            raise RuntimeError(
                "strategy run ended with {}/{} requests complete".format(
                    self.completed_requests, self.expected_request_count))
        if self.runtime_by_request_id or self._runtime_index or \
                any(runtime is not None for runtime in self.runtimes):
            raise RuntimeError("strategy run ended with unreleased runtimes")
        # 拼 batch 列车收尾审计(§3.2 状态机):实例空闲 = qp /
        # active_decode / pending_decode_ready 全空且无在飞列车;已核销
        # 列车的跨交付残余信号必须全部收齐。
        if any(state.qp or state.active_decode or state.pending_decode_ready
               or state.in_flight_train is not None
               for state in self.instances):
            raise RuntimeError("strategy run ended with non-idle instance state")
        if any(state.prefill_remaining_chunks != 0
               for state in self.instances):
            raise RuntimeError(
                "strategy run ended with nonzero prefill queue aggregate")
        # WP9 拆分收尾断言:所有首步批的余量批都已发射、唤醒都已消费。
        if any(state.first_step_remainder is not None
               for state in self.instances):
            raise RuntimeError(
                "strategy run ended with an unflushed first-step remainder")
        if self._pending_first_steps:
            raise RuntimeError(
                "strategy run ended with unconsumed first-step wakeups: "
                "{!r}".format(sorted(self._pending_first_steps)))
        if any(state.finalized_trains for state in self.instances):
            raise RuntimeError(
                "strategy run ended with unconsumed late train signals")
        if any(queue for queue in self.waiting_decode_admissions.values()):
            raise RuntimeError(
                "strategy run ended with pending decode admissions")
        if self.arrival_heap:
            raise RuntimeError(
                "run ended with {} unconsumed arrival events".format(
                    len(self.arrival_heap)))
        if self._ready_frontier:
            raise RuntimeError(
                "run ended with non-empty ready frontier: {!r}".format(
                    sorted(self._ready_frontier)))
        if self.graph.completion_gates:
            raise RuntimeError(
                "strategy run ended with unretired completion gates: {!r}"
                .format(sorted(self.graph.completion_gates)))
        # B3:三态 KV 发射账本收尾审计——pending location 链/partial 流水
        # 登记必须全部被消费(与 §7.3 frontier 同款 fail-closed)。
        if getattr(self.graph, "deferred_session_locations", None):
            raise RuntimeError(
                "strategy run ended with deferred session locations: {!r}"
                .format(sorted(self.graph.deferred_session_locations)))
        if getattr(self.graph, "pending_request_by_session", None):
            raise RuntimeError(
                "strategy run ended with pending request-by-session: {!r}"
                .format(sorted(self.graph.pending_request_by_session)))
        if getattr(self.graph, "_partial_first_chunk", None):
            raise RuntimeError(
                "strategy run ended with unconsumed partial first-chunk "
                "ledgers: {!r}".format(
                    sorted(self.graph._partial_first_chunk)))
        if getattr(self.graph, "_block_ends", None):
            raise RuntimeError(
                "strategy run ended with unretired request block ends: {!r}"
                .format(sorted(self.graph._block_ends)))
        self.kv_manager.assert_final_state()
