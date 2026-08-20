#!/usr/bin/env python3
"""face_online_scheduler.py -- 关感知策略调度器(strategy 模式,步骤 1-9,face 版)。

以 _plan_face_session_lru_recompute(face_scheduler.py:1294-1806)为蓝本迁移,
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
      qp.popleft + select_decode_instance             busy=False + qp.popleft +
      (:1627-1637,加权图候选 + LUT per-die 代价,       has_prefill_work/decode_token_
      全局 9 实例快照 :1622-1627)                      lengths 快照 -> select_decode_
      + waiting_decode_admissions 登记 +               instance -> 等待 decode 准入
      try_admit_waiting_decodes(:1648)                 登记 + try_admit_waiting_decodes
    decode 完成(:1650-1681)                         _on_decode_complete:
      active_decode 出队 + mark_complete +             active_decode 出队 + mark_complete
      note_capacity_change + 快照                      + 快照
      + 下一次 arrival 排程(:1676-1681)                (下一次 arrival 排程在
                                                       REQUEST_COMPLETE 边界做)
  arrival 批(:1683-1695)                            _on_arrival(经 arrival heap):
    queue_snapshot -> select_prefill_instance          快照 -> 选择 -> qp.append +
    + qp.append + last_arrival_ns                      last_arrival_ns 更新
  start_ready_iterations(:1517-1590)                _admit_pass(同 tick 末尾):
    try_admit_waiting_decodes(:1519)                   decode 准入(容量 epoch/dirty
                                                       门控,原样保留)
    逐实例 serve(:1520-1588)                           per-instance 发射:
      prefill: try_admit_prefill(qp[0])                 try_admit_prefill + 发射
      decode:  active_decode 整批                       prefill/decode 整段
    **计时部分删除**(:1551-1588 LUT lookup +
    push iteration_complete):在线由真实完成事件
    (C++ 物理时钟)推进;排队/配对/准入逻辑原样
    保留于账本,不体现在图结构上(构图粒度与离线
    一致:prefill 整段 + 迁移段 + decode 整段)。

face 独有(与 wscllm 蓝图的差异,逐项保留不抹平):
  - 统一实例:9 实例同时承担 prefill+decode(qp 与 active_decode 可并存);
  - decode 实例动态选择:prefill 完成边界经 WeightedInstanceGraph 候选 +
    LUT per-die 代价(select_decode_instance,face_scheduler.py:654-709)
    选择;**LUT 以标定常数在线保留**(合同⑨裁决:estimate 公式不动,
    max_d_token/request_count 从冻结输入导出,禁止在线增量扩展);
  - decode 选择的顺序敏感全局快照:has_prefill_work(各实例 qp 非空)/
    decode_token_lengths(各实例 active_decode 的 current_decode_token)
    按离线 :1622-1627 同一构造,prefill 完成边界才读取(时机不变);
  - try_admit_waiting_decodes 的阻塞重排队语义(move.admission_blocked ->
    queue.append + continue,face_scheduler.py:1488-1495)原样保留。

real-online 刻意差异(合同⑦ Tier B real-online 验收;不变量 = 同一快照输入
-> 同一输出,不逐值强等):
  - 计时/迭代粒度:离线 LUT 时钟 + 逐 chunk 迭代 -> 在线真实完成事件 +
    request-aggregated 构图;
  - 实例 busy 语义:离线 busy 覆盖一次混合迭代(1 prefill chunk + 全部
    active decode 各 1 token);在线 busy 覆盖"一个 prefill/decode 整段
    在飞";
  - decode 发射:离线一次迭代推进整批 active_decode;在线一次发射队首
    整段,完成后再发射下一个(per-rank 物理链天然串行化同实例段);
  - current_decode_token 快照口径:离线逐迭代递增(:1652);在线 decode
    整段在飞期间保持 prefill_context_tokens(0 个已完成聚合迭代);
  - remaining_chunks 快照口径:离线逐 chunk 递减(:1610);在线保持准入时
    的值(含 history chunks,:1442-1445),drain 后随请求离队。
"""

import heapq
import math
import os
import sys
from collections import deque

# --------------------------------------------------------------------------
# import 路径:本文件位于 workload/llama2_7b_inference/online/,离线写出模块在
# 上一级。路径只做 import 用途(红线:generate_face_trace.py /
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
    _transfer_dict,
)
from online.online_scheduler_base import (  # noqa: E402
    STAGE_DECODE,
    STAGE_PREFILL,
    STAGE_REQUEST,
    OnlineSchedulerBase,
)
from session_kv_manager import (  # noqa: E402
    SessionKVCacheManager,
    kv_cache_shard_bytes_for_tokens,
)
from face_scheduler import (  # noqa: E402
    NOC_MIGRATE,
    DecodeTieCounter,
    FaceInstanceSpec,
    FaceLut,
    PrefillQueueSnapshot,
    WeightedInstanceGraph,
    build_instances,
    select_decode_instance,
    select_prefill_instance,
)


class _OnlineInstanceState:
    """在线实例账本(离线 _InstanceRuntime,face_scheduler.py:1004-1009 的
    在线子集):qp = prefill FCFS 队列(deque),active_decode = 已准入 decode
    列表,last_arrival_ns = 选择键素材,busy = 一个整段在飞。"""

    __slots__ = ("index", "qp", "active_decode", "active_decode_lookup",
                 "last_arrival_ns", "busy")

    def __init__(self, *, index: int) -> None:
        self.index = index
        self.qp = deque()  # FCFS(append 尾入,popleft 首出)
        self.active_decode = []
        self.active_decode_lookup = set()
        self.last_arrival_ns = None
        self.busy = False


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
        "history_transfer",
        "admission_evictions", "decode_target_evictions",
        "waiting_decode_admission", "prefill_decode_transfer",
        "completion_evictions", "kv_state_after_completion",
        "kv_instance_after_completion", "hbm_after_completion",
        "remaining_chunks", "current_decode_token", "completion_ns",
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
        self.history_transfer = None
        self.admission_evictions = ()
        self.decode_target_evictions = ()
        self.waiting_decode_admission = False
        self.prefill_decode_transfer = None
        self.completion_evictions = ()
        self.kv_state_after_completion = None
        self.kv_instance_after_completion = None
        self.hbm_after_completion = None
        # offline: face_scheduler.py:997-998(初始值;准入时 :1442-1445 重算)
        self.remaining_chunks = math.ceil(self.prefill_length / _P_CHUNK_HOLDER[0]) \
            if _P_CHUNK_HOLDER[0] else 0
        # offline: face_scheduler.py:999(current_decode_token = prefill_context)
        self.current_decode_token = self.prefill_context_tokens
        self.completion_ns = None


# _OnlineRequestRuntime 构造需要 p_chunk(初始 remaining_chunks);调度器
# __init__ 先设置该模块级占位再构造 runtimes(单进程单调度器,无并发)。
_P_CHUNK_HOLDER = [0]


def _kv_event_dict(event) -> dict:
    """Serialize one KVCacheEvent for the online kv_actions stream
    (与 _eviction_dict / _transfer_dict 同构的在线序列化)。"""
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


class FaceOnlineScheduler(OnlineSchedulerBase):
    """strategy 变体:face 真实策略(关感知)在在线骨架中运行。

    蓝本: _plan_face_session_lru_recompute(face_scheduler.py:1294-1806),
    kv_cache_policy == "session_lru_recompute"(主变体)。拓扑 / 加权实例图 /
    LUT(标定常数)/ KV 账本(与离线同一函数、同参数)在 __init__ 一次性
    构建,运行期策略输入全部来自这些 Python 账本(关感知)。
    """

    def __init__(self, *, manifest, config, graph, digest_sink=None,
                 mode: str = "strategy", sensing: bool = False):
        super().__init__(
            manifest=manifest,
            config=config,
            replay=None,  # strategy 无决策日志回放源
            digest_sink=digest_sink,
            mode=mode,
            sensing=sensing,
        )
        if mode != "strategy":
            raise ValueError("FaceOnlineScheduler requires mode == 'strategy'")
        if config.kv_cache_policy != "session_lru_recompute":
            raise ValueError(
                "strategy scheduler supports kv_cache_policy "
                "'session_lru_recompute' only, got {!r}".format(
                    config.kv_cache_policy))
        self.graph = graph  # GraphBatchBuilder(与 replay 路径共用;基类经 self.graph 调 begin_batch)

        # 蓝图 :1314:拓扑(统一实例,require_equal_size)。
        # offline: face_scheduler.py:1314
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

        # 蓝图 :1316-1324:LUT(合同⑨裁决:标定常数,在线保留为静态代价
        # 函数)。max_d_token / request_count 从冻结输入的 manifest 按离线
        # 同一推导导出(max(final_context_tokens) / 请求数;manifest =
        # 阶段 0 冻结输入的 policy-independent 事实)——**禁止在线运行时
        # 增量扩展 token bin 或放宽 d_batch 上界**(超范围查询 KeyError
        # fail-closed,face_scheduler.py:521-525 已有行为)。
        # offline: face_scheduler.py:1316-1324
        request_count = len(manifest["requests"])
        max_d_token = max(
            record["final_context_tokens"]
            for record in manifest["requests"]
        )
        self.lut = FaceLut.build(
            config.hardware,
            config.model,
            instance_sizes=(instance.size
                            for instance in self.topology.instances),
            p_chunk=self.p_chunk,
            request_count=request_count,
            max_d_token=max_d_token,
        )

        # 蓝图 :1325-1330:加权实例图 + KV 账本。
        # offline: face_scheduler.py:1325-1330
        self.instance_graph = WeightedInstanceGraph(self.topology)
        self.kv_manager = SessionKVCacheManager(
            self.topology,
            config.model,
            reserve_context_tokens=config.kv_reserve_context_tokens,
        )
        # 中-1 裁决（2026-08-20）：decode 平局按 instance_index 升序轮流；
        # 在线调度器与离线 plan 各持一个计数器，前进条件相同（仅真实平局）。
        self._decode_tie_counter = DecodeTieCounter()

        # 蓝图 :1331:实例账本(统一实例,无 phase_role)。
        # offline: face_scheduler.py:1331
        self.instances = [
            _OnlineInstanceState(index=instance.index)
            for instance in self.topology.instances
        ]

        # 蓝图 :1333-1334:future arrival min-heap(在线由 ingress ARRIVAL
        # 事件喂入)。键含 queue_index,同 tick 到期项按冻结队列序稳定弹出。
        # offline: face_scheduler.py:1333-1334
        self.arrival_heap = []
        self._sequence = 0

        # 蓝图 :1353-1355:等待 decode 准入登记(按实例)。
        # offline: face_scheduler.py:1353-1355
        self.waiting_decode_admissions = {
            instance.index: deque() for instance in self.topology.instances
        }
        # 蓝图 :1359-1362:容量 epoch / 准入门控。
        # offline: face_scheduler.py:1359-1362
        self.capacity_epoch = [0 for _ in self.topology.instances]
        self.prefill_attempt_epoch = {}
        self.decode_admission_epoch = [-1 for _ in self.topology.instances]
        self.decode_admission_dirty = set()

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
        """决策顺序逐行对应离线事件循环:completion 批(:1598-1682)先于
        arrival 批(:1683-1695),最后 start_ready_iterations(:1697)。
        """
        tick = delta["tick"]

        # ---- completion 批(离线 priority 0;同 tick 先于 arrival)----
        # offline: face_scheduler.py:1598-1682
        for group in delta["completed_groups"]:
            stage = group["stage"]
            request_id = group["request_id"]
            if stage == STAGE_PREFILL:
                self._on_prefill_drain(request_id, tick)
            elif stage == STAGE_DECODE:
                self._on_decode_complete(request_id, tick)
            elif stage == STAGE_REQUEST:
                self._on_request_complete(request_id, tick)
            else:
                raise ValueError(
                    "unknown completion stage {!r} for request {!r}".format(
                        stage, request_id))

        # ---- arrival 批(离线 priority 1)----
        # offline: face_scheduler.py:1683-1695
        for arrival in delta["arrivals"]:
            self._push_arrival(arrival, tick)
        self._drain_arrival_heap(tick)

        # ---- 准入/发射 pass(离线 start_ready_iterations,计时部分删除)----
        # offline: face_scheduler.py:1697 -> 1517-1590
        self._admit_pass(tick)

        # ---- kv 动作流:本批次 kv_manager 新产出的账本事件 ----
        events = self.kv_manager.events
        if len(events) > self._kv_events_emitted:
            self._batch["kv_actions"].extend(
                _kv_event_dict(event)
                for event in events[self._kv_events_emitted:])
            self._kv_events_emitted = len(events)

    # ------------------------------------------------------------- 边界 --

    def _on_arrival(self, runtime, tick: int) -> None:
        """离线 arrival 批单条(:1683-1695):快照 -> select_prefill_instance
        -> qp 入队 -> last_arrival_ns 更新。快照在 append 之前取(ordering_key
        反映选择时刻的排队深度/最近到达)。

        offline: face_scheduler.py:1683-1695
        """
        runtime.estimated_arrival_ns = tick  # :1688
        snapshots = self._queue_snapshots()  # :1689
        selected = select_prefill_instance(snapshots)  # :1690
        selected_snapshot = snapshots[selected]  # :1691
        runtime.prefill_instance_index = selected  # :1692
        runtime.prefill_assignment_key = selected_snapshot.ordering_key  # :1693
        self.instances[selected].qp.append(runtime)  # :1694
        self.instances[selected].last_arrival_ns = tick  # :1695
        self._note_instance_ready(selected)  # §7.3 ready frontier
        # 阶段 3 感知账本:进入 admitted 层(prefill_qp 排队账本成员;
        # contract ⑥)。查询/审计数据,不进策略判据。
        self._ledger_admit(runtime.request_id, tick,
                           {"type": "prefill_qp", "instance_index": selected})

    def _on_prefill_drain(self, request_id: str, tick: int) -> None:
        """离线 prefill 完成分支(:1613-1648):实例空闲 + qp 出队 + 全局
        快照 -> select_decode_instance(加权图候选 + LUT per-die 代价) ->
        等待 decode 准入登记 + dirty + 立即 try_admit_waiting_decodes。

        offline: face_scheduler.py:1613-1648
        """
        runtime = self.runtime_by_request_id[request_id]
        state = self.instances[runtime.prefill_instance_index]
        if not state.qp or state.qp[0] is not runtime:  # :1614-1615
            raise RuntimeError("prefill FCFS queue order was corrupted")
        state.busy = False  # :1603(iteration_complete 的 busy 复位)
        state.qp.popleft()  # :1620
        if state.qp or state.active_decode:
            self._ready_frontier.add(state.index)  # §7.3:仍有排队工作
        else:
            self._ready_frontier.discard(state.index)
        # 全局 9 实例快照(顺序敏感决策点,离线 :1622-1627 同一构造)。
        # offline: face_scheduler.py:1622-1627
        has_prefill = [bool(instance.qp) for instance in self.instances]
        active_tokens = [
            [member.current_decode_token for member in instance.active_decode]
            for instance in self.instances
        ]
        selected, costs = select_decode_instance(  # :1627-1637
            topology=self.topology,
            graph=self.instance_graph,
            lut=self.lut,
            fixed_p_chunk=self.p_chunk,
            prefill_instance_index=state.index,
            has_prefill_work=has_prefill,
            decode_token_lengths=active_tokens,
            new_request_token_length=runtime.prefill_context_tokens,
            tie_counter=self._decode_tie_counter,
        )
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
        active_decode 出队 + mark_complete + note_capacity_change + 快照。
        下一次 arrival 排程在 REQUEST_COMPLETE 边界(同 tick)。

        offline: face_scheduler.py:1650-1675
        """
        runtime = self.runtime_by_request_id[request_id]
        state = self.instances[runtime.decode_instance_index]
        if runtime not in state.active_decode_lookup:  # :1656-1657
            raise RuntimeError("decode queue membership was corrupted")
        state.active_decode_lookup.discard(runtime)  # §7.3 双侧同步
        state.active_decode.remove(runtime)  # :1658
        state.busy = False  # :1603(iteration_complete 的 busy 复位)
        if state.qp or state.active_decode:
            self._ready_frontier.add(state.index)  # §7.3
        else:
            self._ready_frontier.discard(state.index)
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
        runtime.kv_state_after_completion = completed_snapshot.state  # :1673
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
            },
        )

    def _on_request_complete(self, request_id: str, tick: int) -> None:
        """离线 decode 完成分支的下一次 arrival 排程(:1676-1681):
        now + 该 turn 的 inter_request_interval_ns 注册未来 alarm(向
        ingress,不再 push 到事件堆)。

        offline: face_scheduler.py:1676-1681
        """
        runtime = self.runtime_by_request_id[request_id]
        # §7.3:_runtime_index O(1) 定位。
        following = self.next_request[self._runtime_index[runtime.request_id]]
        if following is None:
            return  # session 最后一 turn:无下一次 arrival
        interval = self._interval_ns(following)  # :1678-1680
        # :1681 push_event(now_ns + interval, 1, "arrival", following) ->
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
        """离线 start_ready_iterations(:1517-1590)的排队/准入部分;计时部分
        (LUT 估计 + push iteration_complete,:1551-1588)删除,由真实完成事件
        推进(合同⑨裁决 1)。先 try_admit_waiting_decodes(:1519),再逐实例
        serve(:1520-1588,统一实例:qp 与 active_decode 可并存;prefill 队首
        优先,准入失败时服务 decode 队首——离线迭代同批混合 prefill chunk +
        decode tokens 的排队语义在聚合粒度下映射为"实例一次一个整段在飞")。

        §7.3:逐实例 serve 只访问 ready frontier(sorted 保持实例 index 序 =
        离线 :1520 的循环序,决策确定性不受影响)。

        offline: face_scheduler.py:1517-1590
        """
        self._try_admit_waiting_decodes(tick)  # :1519
        for instance_index in sorted(self._ready_frontier):  # §7.3 frontier
            self._profile_scan()  # §7.3:frontier 访问条目(就绪实例)
            state = self.instances[instance_index]
            if state.busy:  # 防御:frontier 与 busy 失步即内部错误
                continue
            if not state.qp and not state.active_decode:  # :1521
                continue
            prefill_runtime = state.qp[0] if state.qp else None  # :1523
            if prefill_runtime is not None and not self._try_admit_prefill(
                    prefill_runtime, tick):  # :1524-1525
                prefill_runtime = None
            if prefill_runtime is not None:
                # :1526-1547(prefill chunk 计时/账本删除,聚合为整段)。
                self._emit_prefill(prefill_runtime, tick)
                state.busy = True  # :1582
                self._ready_frontier.discard(state.index)  # §7.3:发射即忙
                continue
            if state.active_decode:  # :1526 decode_indexes 非空
                # 聚合:一次发射 active_decode 队首整段(离线一次迭代推进
                # 整批;在线整段在飞,per-rank 物理链天然串行化同实例段)。
                runtime = state.active_decode[0]
                self._emit_decode(runtime, tick)
                state.busy = True  # :1582
                self._ready_frontier.discard(state.index)  # §7.3:发射即忙

    def _try_admit_prefill(self, runtime, now_ns: int) -> bool:
        """离线 try_admit_prefill(:1382-1453),逐行对应;无逐 chunk 计时
        账本(聚合粒度)。**不读 LUT**(KV 准入链不读 LUT,合同⑨复核结论)。

        offline: face_scheduler.py:1382-1453
        """
        if runtime.admitted_prefill:  # :1384-1385
            return True
        if runtime.prefill_instance_index is None:  # :1386-1387
            raise RuntimeError("prefill admission lost its fixed mapping")
        target_instance = runtime.prefill_instance_index  # :1388
        if (self.prefill_attempt_epoch.get(runtime.request_id)  # :1389
                == self.capacity_epoch[target_instance]):
            return False
        self.prefill_attempt_epoch[runtime.request_id] = (  # :1390-1391
            self.capacity_epoch[target_instance])
        before_snapshot = self.kv_manager.session_snapshot(  # :1392
            runtime.session_id)
        runtime.history_cache_state_before = (  # :1393-1395
            "ABSENT" if before_snapshot is None else before_snapshot.state
        )
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
        runtime.admission_evictions = decision.evictions  # :1405
        if decision.admission_blocked:  # :1406-1412
            if decision.evictions:
                self._note_capacity_change(
                    target_instance,
                    *(record.victim_instance_index
                      for record in decision.evictions),
                )
            return False
        runtime.history_action = decision.action  # :1413
        runtime.history_source_instance_index = decision.source_instance_index  # :1414
        runtime.history_transfer_bytes = decision.transfer_bytes  # :1415
        runtime.history_recompute_tokens = decision.recompute_tokens  # :1416
        if decision.action == NOC_MIGRATE:  # :1417-1429(history_transfer)
            from session_kv_manager import KVTransfer  # noqa: E402
            runtime.history_transfer = KVTransfer(
                action="NOC_MIGRATE",
                phase="history",
                reason="history_other_instance",
                session_id=runtime.session_id,
                trigger_request_id=runtime.request_id,
                source_instance_index=decision.source_instance_index,
                target_instance_index=runtime.prefill_instance_index,
                history_tokens=runtime.history_tokens_before,
                total_bytes=decision.transfer_bytes,
                shards=decision.transfer_shards,
            )
        growth = self.kv_manager.grow_prefill(  # :1430-1435
            runtime.session_id,
            runtime.prefill_context_tokens,
            now_ns,
            runtime.request_id,
        )
        if not growth.admitted:  # :1436-1439
            raise RuntimeError(
                "prefill growth lost its successful admission reservation")
        runtime.admission_evictions = tuple(  # :1440
            (*runtime.admission_evictions, *growth.evictions))
        runtime.remaining_chunks = (  # :1442-1445(快照素材;在线保持该值
            # 直到 drain,见类 docstring 的刻意差异清单)
            math.ceil(runtime.history_recompute_tokens / self.p_chunk)
            + math.ceil(runtime.prefill_length / self.p_chunk)
        )
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

        offline: face_scheduler.py:1455-1515
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
                if move.admission_blocked:  # :1488-1495
                    if move.evictions:
                        self._note_capacity_change(
                            target_instance,
                            *(record.victim_instance_index
                              for record in move.evictions),
                        )
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
                decode_state = self.instances[target_instance]  # :1508
                decode_state.active_decode.append(runtime)  # :1508
                decode_state.active_decode_lookup.add(runtime)  # §7.3 双侧同步
                if not decode_state.busy:
                    self._ready_frontier.add(target_instance)  # §7.3
                runtime.waiting_decode_admission = False  # :1509
                # 阶段 3 感知账本:admitted 层排队类型更新(active_decode)。
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

    def _emit_prefill(self, runtime, tick: int) -> None:
        """聚合粒度发射 prefill 整段(gates/history/recompute/barrier/
        current_prefill;与离线 ET 按 request 整段一致)。watch 注册
        PREFILL_DRAIN。

        offline: face_scheduler.py:1523-1547(发射对象为整段而非 chunk)
        """
        plan = self._plan_dict(runtime)
        members = self.graph.emit_prefill_batch(plan)
        self._note_emitted(runtime.request_id, STAGE_PREFILL)
        self._ledger_issue(runtime.request_id, tick, STAGE_PREFILL,
                           runtime.prefill_instance_index)
        self._batch["watches"].append({
            "request_id": runtime.request_id,
            "stage": STAGE_PREFILL,
            "generation": 0,
            "members": members,
            "statuses": ["Success", "Skipped"],
        })
        # face:统一实例的 decode 在 PREFILL_DRAIN 边界才选定;assignment
        # 在 decode 发射时一次性携带双索引(C++ 校验器要求两索引非负,
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
                "admission_evictions": [
                    _eviction_dict(record)
                    for record in runtime.admission_evictions
                ],
                "decode_target_evictions": [],
            },
        )

    def _emit_decode(self, runtime, tick: int) -> None:
        """聚合粒度发射 decode 整段(transfer 3000 + decode 整段 + end
        barrier)。watch 注册 DECODE_COMPLETION(C++ 同 fire 推
        DECODE_COMPLETION + REQUEST_COMPLETE 两条 completed_groups)。

        offline: face_scheduler.py:1526/1548-1550(decode_indexes 的聚合发射)
        """
        plan = self._plan_dict(runtime)
        members = self.graph.emit_decode_batch(plan)
        self._note_emitted(runtime.request_id, STAGE_DECODE)
        self._ledger_issue(runtime.request_id, tick, STAGE_DECODE,
                           runtime.decode_instance_index)
        self._batch["watches"].append({
            "request_id": runtime.request_id,
            "stage": STAGE_DECODE,
            "generation": 1,
            "members": members,
            "statuses": ["Success", "Skipped"],
        })
        self._batch["assignments"].append({
            "request_id": runtime.request_id,
            "prefill_instance_index": runtime.prefill_instance_index,
            "decode_instance_index": runtime.decode_instance_index,
        })
        self.log_decision(
            {"kind": "decode", "request_id": runtime.request_id,
             "priority": 0},
            tick,
            decision={
                "decode_instance_index": runtime.decode_instance_index,
                # face 独有(B2 oracle 证据):decode 候选代价全记录。
                "decode_candidates": [
                    _candidate_dict(candidate)
                    for candidate in runtime.decode_candidates
                ],
                "prefill_decode_transfer": _transfer_dict(
                    runtime.prefill_decode_transfer),
            },
        )

    # ------------------------------------------------------------- 助手 --

    def _plan_dict(self, runtime) -> dict:
        """graph_batch_builder 消费的 plan 字段(request 事实 + 在线决策)。"""
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
        }

    def _queue_snapshots(self):
        """离线 queue_snapshot(:1371-1376):全部 9 统一实例(无角色过滤),
        remaining_chunks = qp 成员之和,last_arrival_ns = 实例最近到达。

        offline: face_scheduler.py:1371-1376 / 1689
        """
        return tuple(
            PrefillQueueSnapshot(
                instance_index=state.index,
                remaining_chunks=sum(
                    member.remaining_chunks for member in state.qp),
                last_arrival_ns=state.last_arrival_ns,
            )
            for state in self.instances
        )

    def _note_capacity_change(self, *instance_indexes) -> None:
        """离线 note_capacity_change(:1364-1369)。"""
        for instance_index in set(instance_indexes):
            if instance_index is not None:
                self.capacity_epoch[instance_index] += 1

    def _note_instance_ready(self, instance_index: int) -> None:
        """§7.3 ready frontier:实例非忙且有排队工作 -> 就绪集。"""
        if not self.instances[instance_index].busy:
            self._ready_frontier.add(instance_index)

    def _sensing_ready_view(self) -> dict:
        """阶段 7 §10.1 ready 层边界视图:ready frontier 实例队列中等待
        服务的 request(face 统一实例:qp 与 active_decode 皆是排队工作)。
        查询/审计输入,不进策略判据。"""
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
        return {"ready_count": len(detail), "detail": detail}

    def _interval_ns(self, runtime) -> int:
        spec = self.config.request_queue[runtime.queue_index]
        return spec.inter_request_interval_ns

    # --------------------------------------------------------------- 收尾 --

    def verify_run_end(self) -> None:
        """基类协议校验之上,叠加离线 :1699-1705 的收尾断言与 §7.3 结束
        审计(arrival heap / ready frontier 全空)。"""
        super().verify_run_end()
        if self.completed_requests != len(self.runtimes):
            raise RuntimeError(
                "strategy run ended with {}/{} requests complete".format(
                    self.completed_requests, len(self.runtimes)))
        if any(state.busy or state.qp or state.active_decode
               for state in self.instances):
            raise RuntimeError("strategy run ended with non-idle instance state")
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
        self.kv_manager.assert_final_state()
