#!/usr/bin/env python3
"""sh10_online_scheduler.py -- Sh10OnlineScheduler(策略路径,关感知)。

方案 §4 步骤 1-9:以已移除的离线 plan_face_requests(2026-08-21 清除)为蓝本
逐行迁移(注释标注离线行号 `# offline: face_scheduler.py:XXXX`),保持决策
顺序与调用链逐行对应;离线迭代计时(start_ready_iterations 推 iteration_
complete 事件)删除,由 C++ 真实完成事件推进;实例内 PD 混合排队语义
(FCFS qp / active_decode / busy / remaining_chunks 记账)原样保留于账本,
不体现在图结构上(request-aggregated 三段式构图,合同④粒度口径)。

红线(§0.4):策略判据全部只读 import 复用——KVCacheManager、
select_prefill_instance、select_decode_instance、PrefillQueueSnapshot、
WeightedInstanceGraph、kv_cache_bytes_for_tokens;Decode 代价由当前硬件、模型
和队列状态直接计算 Roofline 估计，不加载或缓存成本表。

边界映射(方案步骤 1-9 操作 1):
  ARRIVAL(冻结队列序)        -> 离线 arrival 批 :3123-3133 + admit_waiting_
                                 requests + 段 1 发射(准入成功的 request);
  PREFILL_DRAIN               -> 离线 prefill 完成分支 :2981-3062(调用顺序
                                 逐行一致)+ 段 2 发射;
  DECODE_COMPLETION           -> 离线 decode 完成分支 :3064-3084(active_decode
                                 移除 + 下一次 arrival 排程[未来 alarm]);
  REQUEST_COMPLETE            -> 离线完成处理 :3086-3110(completion_order
                                 排序 -> mark_complete -> enforce_reserve)+
                                 段 3 发射;
  每 delta 末尾               -> admit_waiting_requests :2903-2909(容量释放
                                 事件驱动重查,§0.4 #17)。

同 tick 顺序(离线 :2970):completion 批先于 arrival 批;completion 批内
先全部 PREFILL_DRAIN 处理,再按 completion_order(session_id, request_id,
queue_index)处理 DECODE_COMPLETION/REQUEST_COMPLETE 的 mark_complete/
enforce_reserve——与离线 :3086-3110 的"先全部 mark_complete 再全部
enforce_reserve"一致。
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

from face_scheduler import (  # noqa: E402  (红线:只读 import)
    DecodeTieCounter,
    KVCacheManager,
    PrefillQueueSnapshot,
    WeightedInstanceGraph,
    build_instances,
    kv_cache_bytes_for_tokens,
    select_decode_instance,
    select_prefill_instance,
)
from generate_face_trace import FaceInstanceSpec  # noqa: E402
from online.online_scheduler_base import (  # noqa: E402
    STAGE_DECODE,
    STAGE_PREFILL,
    STAGE_REQUEST,
    OnlineSchedulerBase,
)


class Sh10OnlineScheduler(OnlineSchedulerBase):
    """sh_1.0 单策略变体在线调度器(队列深度均衡 prefill + Roofline 代价 decode +
    两态 KV + edge-rank 远端存取;关感知 = 策略输入全是 Python 账本)。"""

    def __init__(self, *, manifest, config, graph, digest_sink=None,
                 mode: str = "strategy", sensing: bool = False):
        super().__init__(
            manifest=manifest,
            config=config,
            digest_sink=digest_sink,
            mode=mode,
            sensing=sensing,
        )
        self.graph = graph
        # ---- 已移除的离线 plan_face_requests 构造段在线复刻 ----
        # offline: face_scheduler.py topology
        specs = tuple(
            FaceInstanceSpec(group.name, group.pg_name, group.ranks)
            for group in config.inference_groups
        )
        self.topology = build_instances(
            config.hardware, specs, require_equal_size=True)
        # offline: :2783 p_chunk —— 固定实验配置(合同⑨:mean 回退在线禁入,
        # 缺失/非法即 fail-closed)。
        p_chunk = int(config.prefill_chunk_size)
        if p_chunk <= 0:
            raise ValueError("online path requires an explicit positive p_chunk")
        self.p_chunk = p_chunk
        # offline: :2798-2804
        self.graph_w = WeightedInstanceGraph(self.topology)
        self.kv_manager = KVCacheManager(
            self.topology,
            config.model,
            edge_ranks=config.remote_memory.edge_npus,
            reserve_context_tokens=config.kv_reserve_context_tokens,
        )
        self.instances = [
            _InstanceState(index=index)
            for index in range(len(self.topology.instances))
        ]
        # 蓝本起点的 decode_tie_counter —— decode 平局
        # 轮流裁决计数器（中-1 裁决 2026-08-20）：调度器生命周期持有一个，
        # 仅真实平局（HBM 可行集内 per_die_delta_ns 精确相等 ≥2）时前进。
        self._decode_tie_counter = DecodeTieCounter()
        # 在线增量累积账本(合同⑨:per-session 上下文按到达增量累积,
        # turn k 只依赖已到达 turn 0..k-1;替代离线全量
        # _validate_and_expand_requests :2716-2757)。
        self._session_history = {}     # session_id -> last final_context_tokens
        self._session_turn = {}        # session_id -> last arrived turn_index
        self._next_by_request = {}     # request_id -> next request spec
        self._runtimes = {}            # request_id -> _OnlineRuntime
        self.pending_admissions = deque()  # request_id(离线 :2826 同构)
        # queue_index -> request spec(到达时登记;索引化,非全量扫描)。
        self._spec_by_queue = {
            index: spec for index, spec in enumerate(config.request_queue)
        }
        self._by_turn = {
            (record["session_id"], record["turn_index"]): record
            for record in manifest["requests"]
        }

    # ------------------------------------------------------------- 策略 --

    def run_variant_policy(self, delta) -> None:
        tick = delta["tick"]
        # 同 tick 顺序(离线 :2970):completion(0)先于 arrival(1)。
        drained = []
        completed_now = []
        for group in delta["completed_groups"]:
            stage = group["stage"]
            request_id = group["request_id"]
            if stage == STAGE_PREFILL:
                drained.append(request_id)
            elif stage == STAGE_DECODE:
                completed_now.append(request_id)
            elif stage == STAGE_REQUEST:
                # REQUEST_COMPLETE 与 DECODE_COMPLETION 同 tick 交付;段 3
                # 发射在 _complete_requests(下)一并处理。
                continue
            else:
                raise ValueError(
                    "unknown completion stage {!r}".format(stage))
        for request_id in drained:
            self._on_prefill_drain(request_id, tick)
        if completed_now:
            self._complete_requests(completed_now, tick)
        for arrival in delta["arrivals"]:
            self._on_arrival(arrival, tick)
        # offline: :3134-3136 retry_admissions -> admit_waiting_requests +
        # start_ready_iterations(在线无 iteration 计时;准入重查保留)。
        self.admit_waiting_requests(tick)

    # ----------------------------------------------------------- ARRIVAL --

    def _on_arrival(self, arrival: dict, tick: int) -> None:
        request_id = arrival["request_id"]
        if request_id in self._runtimes:
            raise RuntimeError("request {!r} arrived twice".format(request_id))
        queue_index = arrival["queue_index"]
        spec = self._spec_by_queue.get(queue_index)
        if spec is None or spec.request_id != request_id:
            raise RuntimeError(
                "arrival {!r} does not match queue index {!r}".format(
                    request_id, queue_index))
        session_id = spec.session_id
        # 增量累积(离线 :2733-2755 的逐到达等价):turn 连续性逐条校验。
        last_turn = self._session_turn.get(session_id, -1)
        if spec.turn_index != last_turn + 1:
            raise ValueError(
                f"session {session_id} turn indexes must be contiguous "
                f"from zero (got turn {spec.turn_index} after {last_turn})")
        history = self._session_history.get(session_id, 0)
        # recompute 单口径(离线 _validate_and_expand_requests 的逐到达
        # 等价):prefill 工作 = 队列 prefill_length(turn-0 已折入 prefix)。
        prefill_tokens_to_process = spec.prefill_length
        prefill_context = history + prefill_tokens_to_process
        final_context = prefill_context + spec.decode_length
        runtime = _OnlineRuntime(
            request=spec,
            queue_index=queue_index,
            history_tokens_before=history,
            prefill_tokens_to_process=prefill_tokens_to_process,
            prefill_context_tokens=prefill_context,
            final_context_tokens=final_context,
            remaining_chunks=math.ceil(prefill_tokens_to_process / self.p_chunk),
        )
        runtime.estimated_arrival_ns = tick  # offline: :3130
        self._session_turn[session_id] = spec.turn_index
        self._session_history[session_id] = final_context
        self._runtimes[request_id] = runtime
        following = self._by_turn.get((session_id, spec.turn_index + 1))
        self._next_by_request[request_id] = following  # manifest 记录或 None
        # offline: :3131-3132 pending_admissions 登记 + retry。
        self.pending_admissions.append(request_id)

    # ------------------------------------------------------ 准入(段 1) --

    def queue_snapshot(self, state):  # offline: :2828-2833
        return PrefillQueueSnapshot(
            instance_index=state.index,
            remaining_chunks=sum(
                self._runtimes[request_id].remaining_chunks
                for request_id in state.qp),
            last_arrival_ns=state.last_arrival_ns,
        )

    def try_admit_request(self, request_id: str, now_ns: int) -> bool:
        """offline: :2835-2901 逐行对应(HBM 可行性过滤 -> 队列深度均衡选择
        -> reserve -> prepare_prefill -> 入队)。"""
        runtime = self._runtimes[request_id]
        if runtime.estimated_arrival_ns is None:
            raise RuntimeError("request cannot be admitted before arrival")
        request = runtime.request
        hbm_feasible_instances = self.kv_manager.request_hbm_feasible_instances(
            session_id=request.session_id,
            final_context_tokens=runtime.final_context_tokens,
        )
        if not any(hbm_feasible_instances):
            eventually_feasible = (
                self.kv_manager.request_hbm_eventually_feasible_instances(
                    session_id=request.session_id,
                    final_context_tokens=runtime.final_context_tokens,
                )
            )
            if not any(eventually_feasible):
                raise ValueError(
                    f"request {request_id} final KV cannot fit on any empty "
                    "instance; final_context_tokens="
                    f"{runtime.final_context_tokens}")
            return False  # 挂起 pending_admissions(§0.4 #17)

        snapshots = tuple(self.queue_snapshot(state) for state in self.instances)
        selected = select_prefill_instance(snapshots, hbm_feasible_instances)
        selected_snapshot = snapshots[selected]
        admission_evictions = self.kv_manager.reserve_request_capacity(
            request_id=request_id,
            session_id=request.session_id,
            instance_index=selected,
            final_context_tokens=runtime.final_context_tokens,
            now_ns=now_ns,
        )
        runtime.prefill_instance_index = selected
        runtime.prefill_assignment_key = selected_snapshot.ordering_key
        runtime.admission_time_ns = now_ns
        (runtime.history_location_before, runtime.history_transfer,
         prepare_evictions) = self.kv_manager.prepare_prefill(
            session_id=request.session_id,
            target_instance_index=selected,
            history_tokens=runtime.history_tokens_before,
            trigger_request_id=request_id,
            reservation_request_id=request_id,
            now_ns=now_ns,
        )
        runtime.history_evictions = admission_evictions + prepare_evictions
        if request.turn_index > 0:
            if runtime.history_location_before is None:
                raise RuntimeError(
                    f"session {request.session_id} has no prior KV state")
            runtime.history_source_instance_index = (
                runtime.history_location_before.instance_index)
            runtime.history_transfer_bytes = kv_cache_bytes_for_tokens(
                self.config.model, runtime.history_tokens_before)
        self.instances[selected].qp.append(request_id)
        self.instances[selected].last_arrival_ns = now_ns
        # 段 1 发射(在线:准入即发射 prefill 整段;物理串行化由图内
        # per-rank previous_id 链承载,strategy 模式保持物理跨 request 链)。
        self._emit_segment1(runtime, now_ns)
        return True

    def admit_waiting_requests(self, now_ns: int) -> None:  # offline: :2903-2909
        blocked = deque()
        while self.pending_admissions:
            request_id = self.pending_admissions.popleft()
            if not self.try_admit_request(request_id, now_ns):
                blocked.append(request_id)
        self.pending_admissions.extend(blocked)

    # ------------------------------------------------- PREFILL_DRAIN(段 2) --

    def _on_prefill_drain(self, request_id: str, tick: int) -> None:
        """offline: :2981-3062 prefill 完成分支,调用顺序逐行一致。"""
        runtime = self._runtimes[request_id]
        state_index = runtime.prefill_instance_index
        state = self.instances[state_index]
        # drain 记实际 prefill 工作量(recompute 单口径 ==
        # request.prefill_length;离线 chunk 累计的整段等价)。
        runtime.prompt_tokens_processed = runtime.prefill_tokens_to_process
        runtime.remaining_chunks = 0
        # offline: :2989-2991 FCFS qp popleft(离线 iteration 串行化保证
        # drain 序 = FCFS 序)。在线 request-aggregated 构图无 iteration
        # 串行化,同实例多 request 的 prefill 段并发执行,drain(物理完成)
        # 可乱序——账本适配 = 从 qp 移除该已完成成员(排队深度口径
        # remaining_chunks 求和不变,策略输入语义等价;登记实录 §15.1)。
        if request_id not in state.qp:
            raise RuntimeError("draining request is not in its prefill queue")
        state.qp.remove(request_id)
        runtime.prefill_complete_ns = tick
        runtime.prefill_evictions = self.kv_manager.expand_prefill(
            session_id=runtime.request.session_id,
            instance_index=state_index,
            context_tokens=runtime.prefill_context_tokens,
            trigger_request_id=request_id,
            reservation_request_id=request_id,
            now_ns=tick,
        )
        has_prefill = [bool(instance.qp) for instance in self.instances]
        active_tokens = [
            [self._runtimes[rid].current_decode_token
             for rid in instance.active_decode]
            for instance in self.instances
        ]
        selected, costs = select_decode_instance(
            hardware=self.config.hardware,
            model=self.config.model,
            topology=self.topology,
            graph=self.graph_w,
            fixed_p_chunk=self.p_chunk,
            prefill_instance_index=state_index,
            has_prefill_work=has_prefill,
            decode_token_lengths=active_tokens,
            new_request_token_length=runtime.prefill_context_tokens,
            hbm_feasible_instances=(
                self.kv_manager.decode_hbm_feasible_instances(
                    session_id=runtime.request.session_id,
                    final_context_tokens=runtime.final_context_tokens,
                    reservation_request_id=request_id,
                )
            ),
            tie_counter=self._decode_tie_counter,
        )
        runtime.decode_instance_index = selected
        runtime.decode_candidates = costs
        reservation_move_evictions = (
            self.kv_manager.move_request_capacity_reservation(
                request_id=request_id,
                target_instance_index=selected,
                now_ns=tick,
            )
        )
        (runtime.prefill_decode_transfer, decode_move_evictions) = (
            self.kv_manager.move_prefill_to_decode(
                session_id=runtime.request.session_id,
                target_instance_index=selected,
                trigger_request_id=request_id,
                reservation_request_id=request_id,
                now_ns=tick,
            )
        )
        decode_growth_evictions = self.kv_manager.expand_decode(
            session_id=runtime.request.session_id,
            instance_index=selected,
            final_context_tokens=runtime.final_context_tokens,
            trigger_request_id=request_id,
            reservation_request_id=request_id,
            now_ns=tick,
        )
        runtime.decode_evictions = (
            reservation_move_evictions + decode_move_evictions
            + decode_growth_evictions)
        runtime.kv_allocation = self.kv_manager.allocation_for_session(
            session_id=runtime.request.session_id,
            request_id=request_id,
        )
        self.kv_manager.release_request_capacity_reservation(request_id)
        self.instances[selected].active_decode.append(request_id)
        # online: start_ready_iterations 的 decode 起始记账(:2927-2928)。
        runtime.decode_start_ns = tick
        self._emit_segment2(runtime, tick)

    # ------------------------------------------ DECODE_COMPLETION(段 3) --

    def _complete_requests(self, completed_now, tick: int):
        """offline: :3064-3110 decode 完成分支 + 完成处理(顺序逐行一致),
        段 3 发射 + 下一次 session arrival 排程(离线 :3084 push_event 在线
        改为 future alarm,时刻 = 完成边界 tick + interval)。"""
        # offline: :3086-3093 completion_order 排序
        completion_order = sorted(
            completed_now,
            key=lambda request_id: (
                self._runtimes[request_id].request.session_id,
                request_id,
                self._runtimes[request_id].queue_index,
            ),
        )
        # offline: :3064-3084 decode 完成分支(active_decode 移除 + 完成时刻
        # + 下一次 arrival 排程)。
        for request_id in completion_order:
            runtime = self._runtimes[request_id]
            state = self.instances[runtime.decode_instance_index]
            if request_id not in state.active_decode:
                raise RuntimeError("decode queue membership was corrupted")
            state.active_decode.remove(request_id)
            runtime.completion_ns = tick
            following = self._next_by_request.get(request_id)
            if following is not None:
                spec = self._spec_by_queue[following["queue_index"]]
                interval = spec.inter_request_interval_ns
                if interval is None:
                    raise RuntimeError(
                        "validated later request lost its interval")
                self._batch["future_alarms"].append({
                    "arrival_world_ns": tick + interval,
                    "envelope": {
                        "request_id": following["request_id"],
                        "session_id": following["session_id"],
                        "turn_index": following["turn_index"],
                        "prefill_length": following["prefill_length"],
                        "decode_length": following["decode_length"],
                        "inter_request_interval_ns": interval,
                    },
                })
        # offline: :3094-3098 先全部 mark_complete
        for request_id in completion_order:
            self.kv_manager.mark_complete(
                self._runtimes[request_id].request.session_id, tick)
        # offline: :3099-3110 再全部 enforce_reserve
        for request_id in completion_order:
            runtime = self._runtimes[request_id]
            (runtime.completion_evictions, runtime.reserve_unmet_ranks
             ) = self.kv_manager.enforce_reserve(
                instance_index=runtime.decode_instance_index,
                trigger_request_id=request_id,
                now_ns=tick,
            )
        # 完成快照(离线 :3111-3121)。
        for request_id in completion_order:
            runtime = self._runtimes[request_id]
            snapshot = self.kv_manager.session_snapshot(
                runtime.request.session_id)
            runtime.kv_location_after_completion = snapshot.location
            runtime.kv_instance_after_completion = snapshot.instance_index
            self._emit_segment3(runtime, tick)

    # ------------------------------------------------------- 段发射封装 --

    def _plan_dict(self, runtime) -> dict:
        request = runtime.request
        return {
            "queue_index": runtime.queue_index,
            "session_id": request.session_id,
            "turn_index": request.turn_index,
            "request_id": request.request_id,
            "prefill_length": request.prefill_length,
            "decode_length": request.decode_length,
            "history_tokens_before": runtime.history_tokens_before,
            "prefill_context_tokens": runtime.prefill_context_tokens,
            "final_context_tokens": runtime.final_context_tokens,
        }

    def _transfer_dicts(self, transfers):
        from online.graph_batch_builder import _restore_kv_transfer  # noqa
        # 在线决策持有 KVTransfer 对象;构图器需要 dict(与决策日志同构)。
        return [
            {
                "kind": t.kind, "phase": t.phase, "reason": t.reason,
                "session_id": t.session_id,
                "trigger_request_id": t.trigger_request_id,
                "source_instance_index": t.source_instance_index,
                "target_instance_index": t.target_instance_index,
                "total_bytes": t.total_bytes,
                "shards": [
                    {
                        "source_rank": s.source_rank,
                        "target_rank": s.target_rank,
                        "edge_rank": s.edge_rank,
                        "bytes": s.bytes,
                        "noc_path": list(s.noc_path),
                    } for s in t.shards
                ],
            }
            for t in (transfers or ())
        ]

    def _emit_segment1(self, runtime, tick: int) -> None:
        plan = self._plan_dict(runtime)
        plan["prefill_instance_index"] = runtime.prefill_instance_index
        plan["decode_instance_index"] = runtime.prefill_instance_index
        plan["admission_time_ns"] = runtime.admission_time_ns
        plan["history_location_before"] = (
            None if runtime.history_location_before is None
            else runtime.history_location_before.location)
        plan["history_transfer"] = self._transfer_dict_or_none(
            runtime.history_transfer)
        plan["history_evictions"] = self._transfer_dicts(
            runtime.history_evictions)
        plan["prefill_evictions"] = self._transfer_dicts(
            runtime.prefill_evictions)
        members = self.graph.emit_prefill_batch(plan)
        self._batch["watches"].append({
            "request_id": runtime.request.request_id,
            "stage": STAGE_PREFILL,
            "generation": 0,
            "members": members,
            "statuses": ["Success", "Skipped"],
        })
        self._batch["assignments"].append({
            "request_id": runtime.request.request_id,
            "prefill_instance_index": runtime.prefill_instance_index,
            # 准入时刻 decode 实例未知(离线同款:decode 在 prefill 完成时
            # 决策);占位 = prefill 实例,段 2 追加完整 assignment。
            "decode_instance_index": runtime.prefill_instance_index,
            "prefill_assignment_key": list(runtime.prefill_assignment_key),
        })
        self.log_decision({
            "kind": "prefill",
            "request_id": runtime.request.request_id,
            "priority": 0,
            "decision": {
                "prefill_instance_index": runtime.prefill_instance_index,
                "prefill_assignment_key": list(
                    runtime.prefill_assignment_key),
                "admission_time_ns": runtime.admission_time_ns,
                "history_transfer": plan["history_transfer"],
                "history_evictions": plan["history_evictions"],
                "prefill_evictions": plan["prefill_evictions"],
            },
        }, tick)
        self._note_emitted(runtime.request.request_id, STAGE_PREFILL)
        self._ledger_admit(runtime.request.request_id, tick, {
            "type": "prefill_qp",
            "instance_index": runtime.prefill_instance_index})
        self._ledger_issue(runtime.request.request_id, tick, STAGE_PREFILL,
                           runtime.prefill_instance_index)

    def _transfer_dict_or_none(self, transfer):
        return None if transfer is None else self._transfer_dicts(
            [transfer])[0]

    def _emit_segment2(self, runtime, tick: int) -> None:
        plan = self._plan_dict(runtime)
        plan["prefill_instance_index"] = runtime.prefill_instance_index
        plan["decode_instance_index"] = runtime.decode_instance_index
        plan["decode_evictions"] = self._transfer_dicts(
            runtime.decode_evictions)
        plan["prefill_decode_transfer"] = self._transfer_dict_or_none(
            runtime.prefill_decode_transfer)
        members = self.graph.emit_decode_batch(plan)
        self._batch["watches"].append({
            "request_id": runtime.request.request_id,
            "stage": STAGE_DECODE,
            "generation": 1,
            "members": members,
            "statuses": ["Success", "Skipped"],
        })
        self._batch["assignments"].append({
            "request_id": runtime.request.request_id,
            "prefill_instance_index": runtime.prefill_instance_index,
            "decode_instance_index": runtime.decode_instance_index,
            "prefill_assignment_key": list(runtime.prefill_assignment_key),
        })
        self.log_decision({
            "kind": "decode",
            "request_id": runtime.request.request_id,
            "priority": 0,
            "decision": {
                "decode_instance_index": runtime.decode_instance_index,
                "decode_candidates": [
                    {
                        "instance_index": c.instance_index,
                        "weighted_distance": c.weighted_distance,
                        "delta_time_ns": c.delta_time_ns,
                        "per_die_delta_ns": c.per_die_delta_ns,
                    } for c in runtime.decode_candidates
                ],
                "prefill_decode_transfer": plan["prefill_decode_transfer"],
                "decode_evictions": plan["decode_evictions"],
            },
        }, tick)
        self._ledger_unissue(runtime.request.request_id, STAGE_PREFILL)
        self._note_emitted(runtime.request.request_id, STAGE_DECODE)
        self._ledger_admit(runtime.request.request_id, tick, {
            "type": "active_decode",
            "instance_index": runtime.decode_instance_index})
        self._ledger_issue(runtime.request.request_id, tick, STAGE_DECODE,
                           runtime.decode_instance_index)

    def _emit_segment3(self, runtime, tick: int) -> None:
        plan = self._plan_dict(runtime)
        plan["decode_instance_index"] = runtime.decode_instance_index
        plan["completion_evictions"] = self._transfer_dicts(
            runtime.completion_evictions)
        plan["kv_location_after_completion"] = (
            runtime.kv_location_after_completion)
        following = self._next_by_request.get(runtime.request.request_id)
        if following is None:
            plan["following"] = None
        else:
            plan["following"] = {
                "queue_index": following["queue_index"],
                "request_id": following["request_id"],
                "hbm_wait_ns": 0,
            }
        members = self.graph.emit_completion_batch(plan)
        if not members:
            members = dict(self.graph._block_ends.get(
                runtime.request.request_id, {}).get("seg2", {}))
        # REQUEST_COMPLETE 不注册独立 watch:decode watch fire 已同时推送
        # DECODE_COMPLETION + REQUEST_COMPLETE(main_online.cc 机制)。
        self.log_decision({
            "kind": "completion",
            "request_id": runtime.request.request_id,
            "priority": 0,
            "decision": {
                "completion_evictions": plan["completion_evictions"],
                "kv_location_after_completion":
                    runtime.kv_location_after_completion,
            },
        }, tick)
        self._ledger_unissue(runtime.request.request_id, STAGE_DECODE)

    # ------------------------------------------------------------- 收尾 --

    def verify_run_end(self) -> None:
        super().verify_run_end()
        # 离线端到端断言(:3138-3152)的在线对账等价物。
        if self.pending_admissions:
            raise RuntimeError(
                "online run ended with blocked HBM admissions: {}".format(
                    list(self.pending_admissions)[:5]))
        incomplete = [
            request_id for request_id, runtime in self._runtimes.items()
            if runtime.completion_ns is None]
        if incomplete:
            raise RuntimeError(
                "online run ended with incomplete requests: {}".format(
                    incomplete[:5]))
        if any(state.qp or state.active_decode for state in self.instances):
            raise RuntimeError("online run ended with non-idle instance state")


class _OnlineRuntime:
    """离线 _RequestRuntime(face_scheduler.py)的在线对应物
    (静态迭代计时字段 decode_steps_remaining/current_decode_token 的增量推进
    删除;current_decode_token 保持在 decode 起始口径,作为
    select_decode_instance 的 active_tokens 账本输入)。"""

    def __init__(self, *, request, queue_index, history_tokens_before,
                 prefill_context_tokens, final_context_tokens,
                 remaining_chunks, prefill_tokens_to_process=None):
        self.request = request
        self.queue_index = queue_index
        self.history_tokens_before = history_tokens_before
        # 实际 prefill 工作量(recompute 单口径 == request.prefill_length,
        # 离线 _RequestRuntime.prefill_tokens_to_process 同款)。
        if prefill_tokens_to_process is None:
            prefill_tokens_to_process = request.prefill_length
        self.prefill_tokens_to_process = prefill_tokens_to_process
        self.prefill_context_tokens = prefill_context_tokens
        self.final_context_tokens = final_context_tokens
        self.remaining_chunks = remaining_chunks
        self.prompt_tokens_processed = 0
        # recompute 单口径下恒 0(源负载无 context-window 丢弃行为;
        # 字段保留以维持输出 schema)。
        self.history_tokens_discarded = 0
        self.current_decode_token = prefill_context_tokens
        self.estimated_arrival_ns = None
        self.admission_time_ns = None
        self.prefill_instance_index = None
        self.prefill_assignment_key = None
        self.prefill_complete_ns = None
        self.decode_instance_index = None
        self.decode_candidates = ()
        self.decode_start_ns = None
        self.completion_ns = None
        self.kv_allocation = None
        self.history_source_instance_index = None
        self.history_transfer_bytes = 0
        self.history_location_before = None
        self.history_transfer = None
        self.history_evictions = ()
        self.prefill_evictions = ()
        self.prefill_decode_transfer = None
        self.decode_evictions = ()
        self.completion_evictions = ()
        self.kv_location_after_completion = None
        self.kv_instance_after_completion = None
        self.reserve_unmet_ranks = ()


class _InstanceState:
    """离线 _InstanceRuntime(:2707-2713);busy/iteration_count 为离线迭代计时
    产物,在线不消费(图内物理串行化替代)。"""

    def __init__(self, index):
        self.index = index
        self.qp = deque()
        self.active_decode = []
        self.last_arrival_ns = None
