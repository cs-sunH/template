#!/usr/bin/env python3
"""face_legacy_online_scheduler.py -- legacy 变体在线调度器(阶段 7 §10.6)。

以已移除的离线 _plan_face_requests_legacy(2026-08-21 离线 planner 清除批删去)为蓝本迁移,每处
`# offline: face_scheduler.py:XXXX` 标注。face legacy 与 wscllm legacy 的关键
差异(方案 §10.6 差异清单,不得照搬 wscllm 形态):
  ① 无 FCFS 队头阻塞:KVAllocator.allocate 容量不足直接 raise
    (face_scheduler.py)——在线保持 fail-closed(运行失败),
    不引入等待机制;
  ② KV 跨实例分片会动态改边权(allocate 内 increase_path / release 内
    decrease_path),影响后续 decode 候选的加权距离——在线账本按离线
    同一顺序应用(共用同一 KVAllocator 实例);
  ③ session 重访在 arrival 时 release 上一 allocation(:1200-1214),
    释放时机与 session_lru 的 mark_complete 不同;
  ④ p_chunk = ceil(mean(prefill_length)) 全量推导——目标 workload 标定
    常数(用户裁决 2026-08-15 框架):从冻结输入的 manifest 按离线同一
    推导预先导出(30s 输入 = 4954),离线/在线共用;禁止在线增量 mean。

real-online 刻意差异(与 FaceOnlineScheduler 同款):计时/迭代粒度聚合
(离线 Roofline 时钟 + 逐 chunk 迭代 -> 真实完成事件 + request-aggregated 构图);
busy 覆盖一个整段在飞;decode 一次发射队首整段。
"""

import heapq
import math
import os
import sys
from collections import deque

_ONLINE_DIR = os.path.dirname(os.path.abspath(__file__))
_WORKLOAD_DIR = os.path.dirname(_ONLINE_DIR)
for _path in (_ONLINE_DIR, _WORKLOAD_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from online.online_scheduler_base import (  # noqa: E402
    STAGE_DECODE,
    STAGE_PREFILL,
    STAGE_REQUEST,
    OnlineSchedulerBase,
)
from face_scheduler import (  # noqa: E402  (READ-ONLY, import only)
    DecodeTieCounter,
    FaceInstanceSpec,
    KVAllocator,
    PrefillQueueSnapshot,
    WeightedInstanceGraph,
    build_instances,
    estimate_model_weight_bytes,
    kv_cache_bytes_for_tokens,
    select_decode_instance,
    select_prefill_instance,
)


class _LegacyInstanceState:
    __slots__ = ("index", "qp", "active_decode", "active_decode_lookup",
                 "last_arrival_ns", "busy")

    def __init__(self, *, index: int) -> None:
        self.index = index
        self.qp = deque()
        self.active_decode = []
        self.active_decode_lookup = set()
        self.last_arrival_ns = None
        self.busy = False


class _LegacyRequestRuntime:
    __slots__ = (
        "request_id", "session_id", "turn_index", "queue_index",
        "prefill_length", "decode_length",
        "history_tokens_before", "prefill_context_tokens", "final_context_tokens",
        "estimated_arrival_ns",
        "prefill_instance_index", "prefill_assignment_key",
        "decode_instance_index", "decode_candidates", "kv_allocation",
        "history_source_instance_index", "history_transfer_bytes",
        "completion_ns", "current_decode_token",
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
        self.kv_allocation = None
        self.history_source_instance_index = None
        self.history_transfer_bytes = 0
        self.completion_ns = None
        # offline: face_scheduler.py(current_decode_token = prefill_context)
        self.current_decode_token = self.prefill_context_tokens


class FaceLegacyOnlineScheduler(OnlineSchedulerBase):
    """strategy 模式 legacy 变体:face 原始映射(KVAllocator 跨实例分片 +
    边权动态调整),无 FCFS 队头阻塞(容量不足 fail-closed)。"""

    def __init__(self, *, manifest, config, graph, digest_sink=None,
                 mode: str = "strategy", sensing: bool = False):
        super().__init__(
            manifest=manifest, config=config, replay=None,
            digest_sink=digest_sink, mode=mode, sensing=sensing,
        )
        if mode != "strategy":
            raise ValueError("FaceLegacyOnlineScheduler requires strategy mode")
        if config.kv_cache_policy != "legacy":
            raise ValueError(
                "legacy scheduler requires kv_cache_policy 'legacy', got {!r}"
                .format(config.kv_cache_policy))
        self.graph = graph
        specs = tuple(
            FaceInstanceSpec(name=g.name, pg_name=g.pg_name, ranks=g.ranks)
            for g in config.inference_groups
        )
        self.topology = build_instances(
            config.hardware, specs, require_equal_size=True)
        # offline: face_scheduler.py(标定常数:冻结输入同一推导;
        # 30s 输入 = 4954,禁止在线增量 mean)。
        self.p_chunk = math.ceil(sum(
            record["prefill_length"]
            for record in manifest["requests"]) / len(manifest["requests"]))
        self.instance_graph = WeightedInstanceGraph(self.topology)
        # offline: face_scheduler.py(KVAllocator:分片 + 边权
        # 动态调整,read-only import 共用实例)。
        self.allocator = KVAllocator(
            self.topology, self.instance_graph,
            model_weight_bytes=estimate_model_weight_bytes(config.model),
        )
        # 中-1 裁决（2026-08-20）：decode 平局按 instance_index 升序轮流；
        # 在线调度器与离线 plan 各持一个计数器，前进条件相同（仅真实平局）。
        self._decode_tie_counter = DecodeTieCounter()
        self.instances = [_LegacyInstanceState(index=i)
                          for i in range(len(self.topology.instances))]
        self.session_allocations = {}
        self.arrival_heap = []
        self._sequence = 0
        self._ready_frontier = set()
        self.runtimes = [
            _LegacyRequestRuntime(record)
            for record in sorted(
                manifest["requests"], key=lambda item: item["queue_index"])
        ]
        self.runtime_by_request_id = {
            runtime.request_id: runtime for runtime in self.runtimes}
        self._runtime_index = {
            runtime.request_id: index
            for index, runtime in enumerate(self.runtimes)}
        by_turn = {}
        for runtime in self.runtimes:
            by_turn[(runtime.session_id, runtime.turn_index)] = runtime
        self.next_request = [None] * len(self.runtimes)
        for index, runtime in enumerate(self.runtimes):
            self.next_request[index] = by_turn.get(
                (runtime.session_id, runtime.turn_index + 1))
        self.completed_requests = 0

    # ------------------------------------------------------------- 策略 --

    def run_variant_policy(self, delta) -> None:
        tick = delta["tick"]
        # offline: face_scheduler.py(completion 批)
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
                    "unknown stage {!r} for {!r}".format(stage, request_id))
        # offline: face_scheduler.py(arrival 批)
        for arrival in delta["arrivals"]:
            runtime = self.runtime_by_request_id[arrival["request_id"]]
            heapq.heappush(
                self.arrival_heap,
                (tick, 1, runtime.queue_index, self._sequence, "arrival",
                 runtime))
            self._sequence += 1
        while self.arrival_heap and self.arrival_heap[0][0] <= tick:
            _, _, _, _, kind, payload = heapq.heappop(self.arrival_heap)
            self._profile_scan()
            self._on_arrival(payload, tick)
        # offline: face_scheduler.py(聚合发射 pass)
        self._admit_pass(tick)

    # ------------------------------------------------------------- 边界 --

    def _on_arrival(self, runtime, tick: int) -> None:
        """offline: face_scheduler.py(session 重访 release 上一
        allocation -> 快照 -> select_prefill_instance -> qp 入队)。"""
        runtime.estimated_arrival_ns = tick  # :1200
        previous_allocation = self.session_allocations.pop(
            runtime.session_id, None)  # :1201
        if runtime.turn_index > 0:  # :1202-1213
            if previous_allocation is None:  # :1203-1206
                raise RuntimeError(
                    "session {} has no prior KV allocation".format(
                        runtime.session_id))
            runtime.history_source_instance_index = (  # :1207-1209
                previous_allocation.decode_instance_index)
            runtime.history_transfer_bytes = kv_cache_bytes_for_tokens(  # :1210-1212
                self.config.model, runtime.history_tokens_before)
            self.allocator.release(previous_allocation)  # :1213(边权还原)
        snapshots = self._queue_snapshots()  # :1215
        selected = select_prefill_instance(snapshots)  # :1216
        selected_snapshot = snapshots[selected]  # :1217
        runtime.prefill_instance_index = selected  # :1218
        runtime.prefill_assignment_key = selected_snapshot.ordering_key  # :1219
        self.instances[selected].qp.append(runtime)  # :1220
        self.instances[selected].last_arrival_ns = tick  # :1221
        if not self.instances[selected].busy:
            self._ready_frontier.add(selected)
        self._ledger_admit(runtime.request_id, tick,
                           {"type": "prefill_qp", "instance_index": selected})

    def _on_prefill_drain(self, request_id: str, tick: int) -> None:
        """offline: face_scheduler.py(qp 出队 + 全局快照 ->
        select_decode_instance + KVAllocator.allocate)。"""
        runtime = self.runtime_by_request_id[request_id]
        state = self.instances[runtime.prefill_instance_index]
        if not state.qp or state.qp[0] is not runtime:  # :1142-1143
            raise RuntimeError("prefill FCFS queue order was corrupted")
        state.busy = False  # :1133
        state.qp.popleft()  # :1144
        if state.qp or state.active_decode:
            self._ready_frontier.add(state.index)
        else:
            self._ready_frontier.discard(state.index)
        has_prefill = [bool(instance.qp) for instance in self.instances]  # :1146
        active_tokens = [  # :1147-1150
            [member.current_decode_token
             for member in instance.active_decode]
            for instance in self.instances
        ]
        selected, costs = select_decode_instance(  # :1151-1160
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
        runtime.decode_instance_index = selected  # :1161
        runtime.decode_candidates = costs  # :1162
        candidate_indices = tuple(cost.instance_index for cost in costs)  # :1163
        runtime.kv_allocation = self.allocator.allocate(  # :1164-1171
            request_id=runtime.request_id,
            decode_instance_index=selected,
            candidate_indices=candidate_indices,
            total_bytes=kv_cache_bytes_for_tokens(
                self.config.model, runtime.final_context_tokens),
        )
        # 容量不足在 allocate 内直接 raise(face 无 FCFS 阻塞,
        # :772-776)——fail-closed,本层不捕获。
        self.session_allocations[runtime.session_id] = runtime.kv_allocation  # :1172
        decode_state = self.instances[selected]
        decode_state.active_decode.append(runtime)  # :1173
        decode_state.active_decode_lookup.add(runtime)
        if not decode_state.busy:
            self._ready_frontier.add(selected)
        self._ledger_admit(runtime.request_id, tick,
                           {"type": "active_decode", "instance_index": selected})

    def _on_decode_complete(self, request_id: str, tick: int) -> None:
        """offline: face_scheduler.py(decode 完成路径;
        legacy 完成不触发 KV 动作——release 在下一 turn 的 arrival)。"""
        runtime = self.runtime_by_request_id[request_id]
        state = self.instances[runtime.decode_instance_index]
        if runtime not in state.active_decode_lookup:  # :1182-1183
            raise RuntimeError("decode queue membership was corrupted")
        state.active_decode_lookup.discard(runtime)
        state.active_decode.remove(runtime)  # :1184
        state.busy = False
        if state.qp or state.active_decode:
            self._ready_frontier.add(state.index)
        else:
            self._ready_frontier.discard(state.index)
        runtime.completion_ns = tick  # :1185
        self.completed_requests += 1  # :1186
        self.log_decision(
            {"kind": "completion", "request_id": runtime.request_id,
             "priority": 0},
            tick,
            decision={
                "kv_allocation": self._allocation_dict(runtime),
            },
        )

    def _on_request_complete(self, request_id: str, tick: int) -> None:
        """offline: face_scheduler.py(下一次 arrival 排程)。"""
        runtime = self.runtime_by_request_id[request_id]
        following = self.next_request[self._runtime_index[runtime.request_id]]
        if following is None:
            return
        spec = self.config.request_queue[following.queue_index]
        self._batch["future_alarms"].append({
            "arrival_world_ns": tick + spec.inter_request_interval_ns,
            "envelope": {
                "request_id": following.request_id,
                "session_id": following.session_id,
                "turn_index": following.turn_index,
                "prefill_length": following.prefill_length,
                "decode_length": following.decode_length,
                "inter_request_interval_ns": spec.inter_request_interval_ns,
            },
        })

    # ------------------------------------------------------------- 发射 --

    def _admit_pass(self, tick: int) -> None:
        """offline: face_scheduler.py 的聚合版(计时部分删除)。"""
        for instance_index in sorted(self._ready_frontier):
            self._profile_scan()
            state = self.instances[instance_index]
            if state.busy or (not state.qp and not state.active_decode):
                continue
            if state.qp:
                runtime = state.qp[0]
                self._emit_prefill(runtime, tick)
                state.busy = True
                self._ready_frontier.discard(state.index)
                continue
            runtime = state.active_decode[0]
            self._emit_decode(runtime, tick)
            state.busy = True
            self._ready_frontier.discard(state.index)

    def _plan_dict(self, runtime) -> dict:
        allocation = runtime.kv_allocation
        return {
            "request_id": runtime.request_id,
            "session_id": runtime.session_id,
            "turn_index": runtime.turn_index,
            "queue_index": runtime.queue_index,
            "prefill_instance_index": runtime.prefill_instance_index,
            "decode_instance_index": runtime.decode_instance_index,
            "history_tokens_before": runtime.history_tokens_before,
            "prefill_context_tokens": runtime.prefill_context_tokens,
            "prefill_length": runtime.prefill_length,
            "decode_length": runtime.decode_length,
            "p_chunk": self.p_chunk,
            "kv_allocation_pieces": (
                () if allocation is None else
                tuple({"instance_index": piece.instance_index,
                       "bytes": piece.bytes}
                      for piece in allocation.pieces)),
        }

    def _emit_prefill(self, runtime, tick: int) -> None:
        members = self.graph.emit_prefill_batch_legacy(self._plan_dict(runtime))
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
        self.log_decision(
            {"kind": "prefill", "request_id": runtime.request_id,
             "priority": 0},
            tick,
            decision={
                "prefill_instance_index": runtime.prefill_instance_index,
                "prefill_assignment_key": list(runtime.prefill_assignment_key),
                "estimated_arrival_ns": runtime.estimated_arrival_ns,
                "history_source_instance_index":
                    runtime.history_source_instance_index,
                "history_transfer_bytes": runtime.history_transfer_bytes,
            },
        )

    def _emit_decode(self, runtime, tick: int) -> None:
        members = self.graph.emit_decode_batch(self._plan_dict(runtime))
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
                "decode_candidates": [
                    {"instance_index": c.instance_index,
                     "weighted_distance": c.weighted_distance,
                     "delta_time_ns": c.delta_time_ns,
                     "per_die_delta_ns": c.per_die_delta_ns}
                    for c in runtime.decode_candidates
                ],
                "prefill_decode_transfer": None,
            },
        )

    def _allocation_dict(self, runtime) -> dict:
        allocation = runtime.kv_allocation
        if allocation is None:
            return None
        return {
            "request_id": allocation.request_id,
            "decode_instance_index": allocation.decode_instance_index,
            "total_bytes": allocation.total_bytes,
            "pieces": [
                {"instance_index": piece.instance_index,
                 "bytes": piece.bytes}
                for piece in allocation.pieces
            ],
        }

    def _queue_snapshots(self):
        """offline: face_scheduler.py。"""
        return tuple(
            PrefillQueueSnapshot(
                instance_index=state.index,
                remaining_chunks=sum(
                    math.ceil(member.prefill_length / self.p_chunk)
                    for member in state.qp),
                last_arrival_ns=state.last_arrival_ns,
            )
            for state in self.instances
        )

    def kv_event_payload_legacy(self) -> dict:
        """run-end 终值(与离线 FacePlan 的 legacy 终值口径同构)。"""
        return {
            "final_edge_weights": [
                [source, target, weight]
                for (source, target), weight in
                sorted(self.instance_graph.weights.items())],
            "final_remaining_capacity_bytes": list(
                self.allocator.remaining_capacity),
        }

    def verify_run_end(self) -> None:
        super().verify_run_end()
        if self.completed_requests != len(self.runtimes):
            raise RuntimeError(
                "legacy run ended with {}/{} requests complete".format(
                    self.completed_requests, len(self.runtimes)))
        if any(state.busy or state.qp or state.active_decode
               for state in self.instances):
            raise RuntimeError("legacy run ended with non-idle instance state")
        if self.arrival_heap:
            raise RuntimeError("run ended with unconsumed arrival events")
        if self._ready_frontier:
            raise RuntimeError("run ended with non-empty ready frontier")
