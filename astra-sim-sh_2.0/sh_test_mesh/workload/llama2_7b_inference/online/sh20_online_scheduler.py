#!/usr/bin/env python3
"""sh20_online_scheduler.py -- sh_2.0 关感知策略调度器（strategy 模式，步骤 1-9）。

以 plan_face_requests（face_scheduler.py:3492-4132）为蓝本逐行迁移，保持决策
顺序逐行对应（每处迁移用 `# offline: face_scheduler.py:XXXX` 注释标注）。

  离线事件循环                                   在线边界
  ----------------                               ----------------
  预置 arrival 堆（:3553-3558）                   ingress ARRIVAL 事件喂入同一
                                                 arrival heap
  iteration_complete 批（:3847-3966）              PREFILL_DRAIN / DECODE_COMPLETION
                                                 / REQUEST_COMPLETE 事件处理：
    prefill 收尾（:3864-3944）                     _on_prefill_drain（调用顺序
                                                 与离线逐行一致）+ 发射 decode 段
    decode 完成（:3946-3959）                     _on_decode_complete
  completion 收尾（:3968-4002）                    _on_request_complete：
    mark_complete → enforce_reserve → 快照         同款 + completion 段发射 +
                                                 下一 turn arrival 排程（alarm）
  arrival 批（:4004-4018）                         _on_arrival（truncate_history
                                                 → pending_admissions）
  admit_waiting_requests（:3764-3770）             _admit_pass 头部（各决策边界
                                                 触发，与离线 retry_admissions
                                                 门控同集合）
  start_ready_iterations（:3772-3829）              _admit_pass 尾部：实例空闲
                                                 → 发射 qp[0] 的 prefill 整段；
                                                 **LUT 计时删除**（offline-only，
                                                 在线由 C++ 真实完成事件推进；
                                                 PD 混合排队语义保留于账本，
                                                 不体现在图结构上）

关感知口径（§4.1 第 6 条）：策略输入全部来自 Python 账本——task-load 三分量、
KV 容量/位置/可行性（KVCacheManager）、LUT 静态代价表（select_decode_instance
的 lut.lookup——合同⑨：本仓 LUT 在线继续消费，标定常数构建）。不新增任何
C++ 状态读取。

与离线蓝图的刻意差异（real-online 语义，合同⑦ Tier B real-online 验收）：
  - 计时/迭代粒度：离线 LUT 时钟 + 逐 chunk 迭代 → 在线真实完成事件 +
    request-aggregated 构图；
  - task-load 三分量口径：running prefill 的剩余比例与 active decode 的
    剩余步数在 request-aggregated 语义下取整段剩余（fraction=1.0、
    current_decode_token=prefill_context），公式与参数与离线同一估算器
    （estimate_prefill_task_load_ns / estimate_decode_remaining_task_load_ns /
    prefill_chunk_task_load_ns 只读复用）；完成时序由真实物理决定，不要求
    与离线 LUT 时钟 exact；
  - 实例 busy 语义：在线 busy 覆盖"一个 prefill 整段在飞"；decode 段发射
    不受 busy 门（strategy 保持物理跨 request 链，per-rank previous_id
    天然串行化同实例段）。
"""

import heapq
import math
import os
import sys
from collections import deque

# --------------------------------------------------------------------------
# import 路径：本文件位于 workload/llama2_7b_inference/online/，离线规划模块
# 在上一级。路径只做 import 用途（红线：face_scheduler.py /
# generate_face_trace.py 只读 import 与注释）。
# --------------------------------------------------------------------------
_ONLINE_DIR = os.path.dirname(os.path.abspath(__file__))
_WORKLOAD_DIR = os.path.dirname(_ONLINE_DIR)
for _path in (_ONLINE_DIR, _WORKLOAD_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from face_scheduler import (  # noqa: E402 -- 只读 import
    InstanceTaskLoadSnapshot,
    PREFILL_CHUNK_SIZE,
    FaceInstanceSpec,
    FaceLut,
    FaceRequest,
    KVCacheManager,
    WeightedInstanceGraph,
    _validate_and_expand_requests,
    build_instances,
    estimate_decode_remaining_task_load_ns,
    estimate_prefill_task_load_ns,
    select_decode_instance,
    select_prefill_instance,
)
from generate_face_trace import _to_scheduler_requests  # noqa: E402
from online.online_scheduler_base import (  # noqa: E402
    STAGE_DECODE,
    STAGE_PREFILL,
    STAGE_REQUEST,
    OnlineSchedulerBase,
)


def _kv_action_rows(runtime, stage):
    """runtime 决策事实 → kv_actions 批行（sh_2.0 口径：face_scheduler 的
    KVCacheManager 无事件流 API，动作事实取自决策产物——与离线
    FaceRequestPlan 的 KVTransfer 字段同源，B2 oracle 已验证等价）。"""
    rows = []

    def add(event_type, transfer):
        if transfer is None:
            return
        rows.append({
            "event_type": event_type,
            "trigger_request_id": runtime.request.request_id,
            "session_id": transfer.session_id,
            "kind": transfer.kind,
            "total_bytes": transfer.total_bytes,
            "source_instance_index": transfer.source_instance_index,
            "target_instance_index": transfer.target_instance_index,
            "layer_start": transfer.layer_start,
            "layer_end": transfer.layer_end,
        })

    if stage == "prefill":
        for transfer in runtime.history_evictions:
            add("history_eviction", transfer)
        add("history_transfer", runtime.history_transfer)
        for transfer in runtime.prefill_evictions:
            add("prefill_eviction", transfer)
    elif stage == "decode":
        for transfer in runtime.decode_evictions:
            add("decode_eviction", transfer)
        add("prefill_decode_transfer", runtime.prefill_decode_transfer)
    else:
        for transfer in runtime.completion_evictions:
            add("completion_eviction", transfer)
    return rows


class _OnlineInstanceState:
    """在线实例账本（离线 _InstanceRuntime，face_scheduler.py:3417-3429 的
    在线子集；iteration_* 计时字段删除——在线无 LUT 迭代时钟）。"""

    __slots__ = ("index", "qp", "active_decode", "last_arrival_ns", "busy")

    def __init__(self, index: int) -> None:
        self.index = index
        self.qp = deque()          # prefill FCFS 队列（request_index）
        self.active_decode = []    # 已准入 decode 列表（request_index）
        self.last_arrival_ns = None
        self.busy = False


class Sh20OnlineScheduler(OnlineSchedulerBase):
    """strategy 变体：sh_2.0 真实策略（关感知）在在线骨架中运行。"""

    def __init__(self, *, manifest, config, graph, digest_sink=None,
                 mode: str = "strategy", sensing: bool = False,
                 calibrated_constants: dict = None):
        super().__init__(
            manifest=manifest,
            config=config,
            replay=None,
            digest_sink=digest_sink,
            mode=mode,
            sensing=sensing,
        )
        if mode != "strategy":
            raise ValueError("Sh20OnlineScheduler requires mode == 'strategy'")
        self.graph = graph

        # 合同⑨标定常数（缺省取 config 全量均值推导值；禁止在线增量统计）。
        constants = calibrated_constants or {}
        self.average_decode_length = constants.get(
            "average_decode_length", config.source_average_decode_length)
        self.p_chunk = PREFILL_CHUNK_SIZE

        # offline: face_scheduler.py:3499（topology）
        specs = tuple(
            FaceInstanceSpec(group.name, group.pg_name, group.ranks)
            for group in config.inference_groups
        )
        self.topology = build_instances(
            config.hardware, specs, require_equal_size=True)

        # offline: face_scheduler.py:3511-3536（_validate_and_expand_requests +
        # max_d_token + LUT）。在线用同一函数推导 runtime（含 prefix clamp
        # :3459-3475）；request_count/max_d_token 为 LUT 覆盖参数（标定常数，
        # 与离线同值——LUT 值只依赖硬件/模型与 bin 覆盖）。
        requests = _to_scheduler_requests(config.request_queue)
        self.runtimes, self._next_request_map = _validate_and_expand_requests(
            requests, self.p_chunk)
        max_d_token = constants.get(
            "max_d_token",
            max(runtime.final_context_tokens for runtime in self.runtimes))
        self.lut = FaceLut.build(
            config.hardware,
            config.model,
            instance_sizes=(instance.size for instance in self.topology.instances),
            p_chunk=self.p_chunk,
            request_count=constants.get(
                "request_count", len(self.runtimes)),
            max_d_token=max_d_token,
        )
        # offline: face_scheduler.py:3537-3541（graph + KV 账本；Python KV 账本
        # 唯一权威，C++ 不建容量模型）
        self.graph_topology = WeightedInstanceGraph(self.topology)
        self.kv_manager = KVCacheManager(
            self.topology,
            config.model,
            edge_ranks=config.remote_memory.edge_npus,
            reserve_context_tokens=config.kv_reserve_context_tokens,
        )
        self.instances = [
            _OnlineInstanceState(index=i)
            for i in range(len(self.topology.instances))
        ]

        # offline: face_scheduler.py:3542-3561
        self.arrival_heap = []
        self._sequence = 0
        self.pending_admissions = deque()

        self.runtime_by_request_id = {
            runtime.request.request_id: runtime for runtime in self.runtimes}
        self._runtime_index = {
            runtime.request.request_id: index
            for index, runtime in enumerate(self.runtimes)}
        # offline: face_scheduler.py:3559-3560（next_request）
        self.next_request = [
            self._next_request_map.get(index)
            for index in range(len(self.runtimes))]

        self.completed_requests = 0
        self._emitted_prefill = set()

    # ------------------------------------------------------------- 策略 --

    def run_variant_policy(self, delta) -> None:
        """决策顺序逐行对应离线事件循环（:3832-4022）。"""
        tick = delta["tick"]
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
        for arrival in delta["arrivals"]:
            self._push_arrival(arrival, tick)
        self._drain_arrival_heap(tick)
        self._admit_pass(tick)

    # ------------------------------------------------------------- 边界 --

    def _push_arrival(self, arrival: dict, tick: int) -> None:
        # offline: face_scheduler.py:3553-3558（arrival priority 1）
        index = self._runtime_index[arrival["request_id"]]
        heapq.heappush(
            self.arrival_heap, (tick, 1, self._sequence, "arrival", index))
        self._sequence += 1

    def _drain_arrival_heap(self, tick: int) -> None:
        # offline: face_scheduler.py:4004-4018（arrival 批）
        while self.arrival_heap and self.arrival_heap[0][0] <= tick:
            _, _, _, _, index = heapq.heappop(self.arrival_heap)
            runtime = self.runtimes[index]
            if runtime.estimated_arrival_ns is not None:
                raise RuntimeError("request arrival was delivered more than once")
            runtime.estimated_arrival_ns = tick
            runtime.history_tokens_discarded = (
                self.kv_manager.truncate_history(
                    runtime.request.session_id,
                    runtime.history_tokens_before,
                    trigger_request_id=runtime.request.request_id,
                )
            )
            self.pending_admissions.append(index)
            self._ledger_admit(runtime.request.request_id, tick,
                               {"type": "pending_admission"})

    def _on_prefill_drain(self, request_id: str, tick: int) -> None:
        """离线 prefill 收尾分支（:3864-3944）：调用顺序与离线逐行一致。"""
        index = self._runtime_index[request_id]
        runtime = self.runtimes[index]
        state = self.instances[runtime.prefill_instance_index]
        # offline: face_scheduler.py:3851-3862
        state.busy = False
        if not state.qp or state.qp[0] != index:
            raise RuntimeError("prefill FCFS queue order was corrupted")
        state.qp.popleft()
        runtime.prefill_complete_ns = tick
        # offline: face_scheduler.py:3873-3881（expand_prefill）
        runtime.prefill_evictions = self.kv_manager.expand_prefill(
            session_id=runtime.request.session_id,
            instance_index=state.index,
            context_tokens=runtime.prefill_context_tokens,
            trigger_request_id=runtime.request.request_id,
            reservation_request_id=runtime.request.request_id,
        )
        # offline: face_scheduler.py:3882-3907（select_decode_instance；LUT
        # 代价在线继续消费——合同⑨本仓裁决）
        has_prefill_work = [bool(instance.qp) for instance in self.instances]
        active_tokens = [
            [self.runtimes[i].current_decode_token
             for i in instance.active_decode]
            for instance in self.instances
        ]
        selected, costs = select_decode_instance(
            topology=self.topology,
            graph=self.graph_topology,
            lut=self.lut,
            fixed_p_chunk=self.p_chunk,
            prefill_instance_index=state.index,
            has_prefill_work=has_prefill_work,
            decode_token_lengths=active_tokens,
            new_request_token_length=runtime.prefill_context_tokens,
            remaining_hbm_capacity_bytes=(
                self.kv_manager.instance_remaining_capacity_totals()
            ),
            hbm_feasible_instances=(
                self.kv_manager.decode_hbm_feasible_instances(
                    session_id=runtime.request.session_id,
                    final_context_tokens=runtime.final_context_tokens,
                    reservation_request_id=runtime.request.request_id,
                )
            ),
        )
        runtime.decode_instance_index = selected
        runtime.decode_candidates = costs
        # offline: face_scheduler.py:3908-3913
        reservation_move_evictions = (
            self.kv_manager.move_request_capacity_reservation(
                request_id=runtime.request.request_id,
                target_instance_index=selected,
            )
        )
        # offline: face_scheduler.py:3914-3925
        (runtime.prefill_decode_transfer,
         decode_move_evictions) = self.kv_manager.move_prefill_to_decode(
            session_id=runtime.request.session_id,
            target_instance_index=selected,
            trigger_request_id=runtime.request.request_id,
            reservation_request_id=runtime.request.request_id,
        )
        # offline: face_scheduler.py:3926-3935
        decode_growth_evictions = self.kv_manager.expand_decode(
            session_id=runtime.request.session_id,
            instance_index=selected,
            final_context_tokens=runtime.final_context_tokens,
            trigger_request_id=runtime.request.request_id,
            reservation_request_id=runtime.request.request_id,
        )
        runtime.decode_evictions = (
            reservation_move_evictions + decode_move_evictions
            + decode_growth_evictions
        )
        # offline: face_scheduler.py:3936-3942
        runtime.kv_allocation = self.kv_manager.allocation_for_session(
            session_id=runtime.request.session_id,
            request_id=runtime.request.request_id,
        )
        self.kv_manager.release_request_capacity_reservation(
            runtime.request.request_id)
        # offline: face_scheduler.py:3943
        self.instances[selected].active_decode.append(index)
        runtime.decode_start_ns = tick
        # 发射 decode 整段 + DECODE_COMPLETION watch。assignment 在 decode
        # 决策后追加（sh_2.0 的 decode 实例在 prefill 完成时才决策——
        # GraphBatch 校验要求 prefill/decode 实例索引非负齐备）。
        members = self.graph.emit_decode_batch(self._plan_of(runtime))
        self._batch["kv_actions"].extend(_kv_action_rows(runtime, "decode"))
        self.log_decision({
            "kind": "decode", "request_id": request_id, "priority": 0,
            "decision": {
                "decode_instance_index": runtime.decode_instance_index,
                "prefill_instance_index": runtime.prefill_instance_index,
                "decode_eviction_count": len(runtime.decode_evictions),
                "prefill_decode_transfer_bytes": (
                    runtime.prefill_decode_transfer.total_bytes
                    if runtime.prefill_decode_transfer else 0),
            },
        }, tick)
        self._batch["watches"].append({
            "request_id": request_id,
            "stage": STAGE_DECODE,
            "generation": 1,
            "members": members,
            "statuses": ["Success", "Skipped"],
        })
        self._batch["assignments"].append({
            "request_id": request_id,
            "prefill_instance_index": runtime.prefill_instance_index,
            "prefill_assignment_key": list(runtime.prefill_assignment_key),
            "prefill_affinity_reason": runtime.prefill_affinity_reason,
            "decode_instance_index": runtime.decode_instance_index,
        })

    def _on_decode_complete(self, request_id: str, tick: int) -> None:
        """离线 decode 推进分支的完成点（:3946-3959）。"""
        index = self._runtime_index[request_id]
        runtime = self.runtimes[index]
        decode_state = self.instances[runtime.decode_instance_index]
        if index not in decode_state.active_decode:
            raise RuntimeError("decode queue membership was corrupted")
        decode_state.active_decode.remove(index)
        runtime.decode_steps_remaining = 0
        runtime.current_decode_token = runtime.final_context_tokens
        runtime.completion_ns = tick
        self.completed_requests += 1

    def _on_request_complete(self, request_id: str, tick: int) -> None:
        """离线 completion 收尾（:3968-4002）+ 下一 turn arrival 排程
        （:3960-3966）+ completion 段发射。"""
        index = self._runtime_index[request_id]
        runtime = self.runtimes[index]
        # offline: face_scheduler.py:3976-3980（mark_complete）
        self.kv_manager.mark_complete(runtime.request.session_id, tick)
        # offline: face_scheduler.py:3981-3991（enforce_reserve）
        (runtime.completion_evictions,
         runtime.reserve_unmet_ranks) = self.kv_manager.enforce_reserve(
            instance_index=runtime.decode_instance_index,
            trigger_request_id=runtime.request.request_id,
        )
        # offline: face_scheduler.py:3992-4002（快照）
        completion_snapshot = self.kv_manager.session_snapshot(
            runtime.request.session_id)
        runtime.kv_location_after_completion = completion_snapshot.location
        runtime.kv_instance_after_completion = completion_snapshot.instance_index

        following_index = self.next_request[index]
        following_plan = None
        if following_index is not None:
            following_plan = self._plan_of(self.runtimes[following_index])
        self._batch["kv_actions"].extend(_kv_action_rows(runtime, "completion"))
        self.log_decision({
            "kind": "completion", "request_id": request_id, "priority": 0,
            "decision": {
                "completion_eviction_count": len(runtime.completion_evictions),
                "kv_location_after_completion": (
                    runtime.kv_location_after_completion),
                "reserve_unmet_ranks": list(runtime.reserve_unmet_ranks),
            },
        }, tick)
        self.graph.emit_completion_batch(self._plan_of(runtime), following_plan)
        if following_index is not None:
            following = self.runtimes[following_index]
            interval = following.request.inter_request_interval_ns
            if interval is None:
                raise RuntimeError("validated later request lost its interval")
            # offline: face_scheduler.py:3965-3966（push_event(now+interval,1)）
            # 在线：向 ingress 注册未来 alarm（strategy 口径 = 完成时刻 +
            # interval；准入等待由调度器在 ARRIVAL 边界处理）。
            self._batch["future_alarms"].append({
                "arrival_world_ns": tick + interval,
                "envelope": {
                    "request_id": following.request.request_id,
                    "session_id": following.request.session_id,
                    "turn_index": following.request.turn_index,
                    "prefill_length": following.request.prefill_length,
                    "decode_length": following.request.decode_length,
                    "inter_request_interval_ns": interval,
                },
            })

    # ------------------------------------------------------------- 准入 --

    def _try_admit_request(self, request_index: int, now_ns: int) -> bool:
        """离线 try_admit_request（:3676-3762）的逐行复刻。"""
        runtime = self.runtimes[request_index]
        if runtime.estimated_arrival_ns is None:
            raise RuntimeError("request cannot be admitted before arrival")
        hbm_feasible_instances = self.kv_manager.request_hbm_feasible_instances(
            session_id=runtime.request.session_id,
            final_context_tokens=runtime.final_context_tokens,
        )
        if not any(hbm_feasible_instances):
            eventually_feasible = (
                self.kv_manager.request_hbm_eventually_feasible_instances(
                    session_id=runtime.request.session_id,
                    final_context_tokens=runtime.final_context_tokens,
                )
            )
            if not any(eventually_feasible):
                raise ValueError(
                    f"request {runtime.request.request_id} final KV cannot fit "
                    "on any eligible empty instance")
            return False

        snapshots = tuple(
            self._task_load_snapshot(state, now_ns) for state in self.instances)
        runtime.prefill_hbm_feasible_instances = hbm_feasible_instances
        if runtime.request.session_id in self.kv_manager.session_ids:
            history_snapshot = self.kv_manager.session_snapshot(
                runtime.request.session_id)
        else:
            history_snapshot = None
        if (history_snapshot is not None
                and history_snapshot.location
                == KVCacheManager.PARTIAL_HBM_REMOTE):
            if history_snapshot.instance_index is None:
                raise RuntimeError("partial history lost its resident instance")
            selected = history_snapshot.instance_index
            if not hbm_feasible_instances[selected]:
                return False
            runtime.prefill_affinity_reason = "resident_prefix_layers"
        else:
            selected = select_prefill_instance(snapshots, hbm_feasible_instances)

        selected_snapshot = snapshots[selected]
        admission_evictions = self.kv_manager.reserve_request_capacity(
            request_id=runtime.request.request_id,
            session_id=runtime.request.session_id,
            instance_index=selected,
            final_context_tokens=runtime.final_context_tokens,
        )
        runtime.prefill_instance_index = selected
        runtime.prefill_assignment_key = selected_snapshot.ordering_key
        runtime.prefill_instance_loads = snapshots
        runtime.admission_time_ns = now_ns
        (runtime.history_location_before,
         runtime.history_transfer,
         prepare_evictions) = self.kv_manager.prepare_prefill(
            session_id=runtime.request.session_id,
            target_instance_index=selected,
            history_tokens=runtime.history_tokens_before,
            trigger_request_id=runtime.request.request_id,
            reservation_request_id=runtime.request.request_id,
        )
        runtime.history_evictions = admission_evictions + prepare_evictions
        if runtime.request.turn_index > 0:
            if runtime.history_location_before is None:
                raise RuntimeError(
                    f"session {runtime.request.session_id} has no prior KV state")
            runtime.history_source_instance_index = (
                runtime.history_location_before.instance_index)
            runtime.history_transfer_bytes = (
                0 if runtime.history_transfer is None
                or runtime.history_transfer.kind == "local_hit"
                else runtime.history_transfer.total_bytes)
        self.instances[selected].qp.append(request_index)
        self.instances[selected].last_arrival_ns = now_ns
        return True

    def _admit_pass(self, now_ns: int) -> None:
        """offline: face_scheduler.py:3764-3770（admit_waiting_requests）+
        :3772-3829（start_ready_iterations 的在线形态）。"""
        blocked = deque()
        while self.pending_admissions:
            request_index = self.pending_admissions.popleft()
            if not self._try_admit_request(request_index, now_ns):
                blocked.append(request_index)
        self.pending_admissions.extend(blocked)

        for state in self.instances:
            # offline: face_scheduler.py:3773-3775（实例空闲且有工作）
            if state.busy or (not state.qp and not state.active_decode):
                continue
            if state.qp:
                index = state.qp[0]
                runtime = self.runtimes[index]
                # offline: face_scheduler.py:3786-3788（prefill_start）
                if runtime.prefill_start_ns is None:
                    runtime.prefill_start_ns = now_ns
                if runtime.request.request_id in self._emitted_prefill:
                    continue
                self._emitted_prefill.add(runtime.request.request_id)
                state.busy = True
                members = self.graph.emit_prefill_batch(self._plan_of(runtime))
                self._batch["kv_actions"].extend(
                    _kv_action_rows(runtime, "prefill"))
                self.log_decision({
                    "kind": "prefill",
                    "request_id": runtime.request.request_id,
                    "priority": 0,
                    "decision": {
                        "prefill_instance_index": (
                            runtime.prefill_instance_index),
                        "prefill_affinity_reason": (
                            runtime.prefill_affinity_reason),
                        "history_transfer_bytes": (
                            runtime.history_transfer_bytes),
                        "history_eviction_count": (
                            len(runtime.history_evictions)),
                    },
                }, now_ns)
                self._batch["watches"].append({
                    "request_id": runtime.request.request_id,
                    "stage": STAGE_PREFILL,
                    "generation": 0,
                    "members": members,
                    "statuses": ["Success", "Skipped"],
                })
                self._ledger_issue(
                    runtime.request.request_id, now_ns, "prefill",
                    runtime.prefill_instance_index)

    # ------------------------------------------------- task-load 三分量 --

    def _task_load_snapshot(self, state: _OnlineInstanceState,
                            now_ns: int) -> InstanceTaskLoadSnapshot:
        """离线 task_load_snapshot（face_scheduler.py:3619-3674）的在线复刻。

        估算公式与参数同一（prefill_chunk_task_load_ns /
        estimate_decode_remaining_task_load_ns 只读复用）；request-aggregated
        口径差异（real-online 刻意差异，模块 docstring）：running fraction
        取 1.0（整段在飞），current_decode_token = prefill_context（整段
        剩余），prompt_tokens_processed 在段完成前置 0。"""
        instance_size = self.topology.instance(state.index).size

        running_prefill_load_ns = 0
        running_index = (
            state.qp[0] if (state.busy and state.qp) else None)
        if running_index is not None:
            runtime = self.runtimes[running_index]
            remaining_tokens = (
                runtime.prefill_tokens_to_process
                - runtime.prompt_tokens_processed)
            if remaining_tokens > 0:
                context_tokens = (
                    runtime.history_tokens_before
                    + runtime.prompt_tokens_processed + remaining_tokens)
                running_prefill_load_ns = math.ceil(
                    estimate_prefill_task_load_ns(
                        hardware=self.config.hardware,
                        model=self.config.model,
                        instance_size=instance_size,
                        chunk_tokens=remaining_tokens,
                        context_tokens=context_tokens,
                    ) * 1.0)

        active_decode_load_ns = 0
        for request_index in state.active_decode:
            runtime = self.runtimes[request_index]
            generated_tokens = (
                runtime.current_decode_token - runtime.prefill_context_tokens)
            active_decode_load_ns += estimate_decode_remaining_task_load_ns(
                self.config_hardware(),
                self.config_model(),
                instance_size=instance_size,
                current_context_tokens=runtime.current_decode_token,
                generated_tokens=generated_tokens,
                average_decode_length=self.average_decode_length,
                running_step_fraction_remaining=1.0,
            )

        return InstanceTaskLoadSnapshot(
            instance_index=state.index,
            running_prefill_task_load_ns=running_prefill_load_ns,
            queued_prefill_task_load_ns=self._queued_prefill_task_load_ns(
                state, instance_size=instance_size,
                running_index=running_index),
            active_decode_task_load_ns=active_decode_load_ns,
            last_arrival_ns=state.last_arrival_ns,
        )

    def _queued_prefill_task_load_ns(self, state: _OnlineInstanceState, *,
                                     instance_size: int,
                                     running_index) -> int:
        """离线 queued_prefill_task_load_ns（:3591-3617）的在线复刻；
        running request 的当前段剩余计入 running 分量（processed += 剩余）。"""
        total_load_ns = 0
        for request_index in state.qp:
            runtime = self.runtimes[request_index]
            processed_tokens = runtime.prompt_tokens_processed
            if request_index == running_index:
                processed_tokens += (
                    runtime.prefill_tokens_to_process
                    - runtime.prompt_tokens_processed)
            remaining_tokens = (
                runtime.prefill_tokens_to_process - processed_tokens)
            while remaining_tokens > 0:
                chunk_tokens = min(self.p_chunk, remaining_tokens)
                context_tokens = (
                    runtime.history_tokens_before
                    + processed_tokens + chunk_tokens)
                total_load_ns += estimate_prefill_task_load_ns(
                    hardware=self.config.hardware,
                    model=self.config.model,
                    instance_size=instance_size,
                    chunk_tokens=chunk_tokens,
                    context_tokens=context_tokens,
                )
                processed_tokens += chunk_tokens
                remaining_tokens -= chunk_tokens
        return total_load_ns

    def config_hardware(self):
        return self.config.hardware

    def config_model(self):
        return self.config.model

    # ------------------------------------------------------------- 助手 --

    def _plan_of(self, runtime) -> dict:
        """runtime → graph_batch_builder 的 plan dict（strategy 模式直接传
        对象：history_transfer / evictions 已是 KVTransfer，
        history_location_before 已是 SessionKVSnapshot）。"""
        return {
            "queue_index": runtime.request.queue_index,
            "session_id": runtime.request.session_id,
            "turn_index": runtime.request.turn_index,
            "request_id": runtime.request.request_id,
            "prefill_length": runtime.request.prefill_length,
            "decode_length": runtime.request.decode_length,
            "admission_time_ns": runtime.admission_time_ns,
            "prefill_instance_index": runtime.prefill_instance_index,
            "decode_instance_index": runtime.decode_instance_index,
            "history_tokens_before": runtime.history_tokens_before,
            "prefill_context_tokens": runtime.prefill_context_tokens,
            "final_context_tokens": runtime.final_context_tokens,
            "history_transfer": runtime.history_transfer,
            "history_evictions": runtime.history_evictions,
            "prefill_evictions": runtime.prefill_evictions,
            "decode_evictions": runtime.decode_evictions,
            "prefill_decode_transfer": runtime.prefill_decode_transfer,
            "completion_evictions": runtime.completion_evictions,
            "history_location_before": runtime.history_location_before,
            "history_source_instance_index": (
                runtime.history_source_instance_index),
            "kv_location_after_completion": (
                runtime.kv_location_after_completion),
            "hbm_wait_ns": getattr(runtime, "hbm_wait_ns", 0),
        }
