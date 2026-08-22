#!/usr/bin/env python3
"""sh30_online_scheduler.py -- sh_3.0 关感知策略调度器（strategy 模式，步骤 1-9）。

以已移除的离线 plan_face_requests（2026-08-21 离线 planner 清除批删去）为蓝本迁移，保持决策顺序
逐行对应（每处迁移用 `# offline: face_scheduler.py:XXXX` 注释标注）。离线事件
循环与在线边界的一一对应：

  离线事件循环                                   在线边界
  ----------------                               ----------------
  预置 turn-0 arrival 堆（:3574-3579）            ingress ARRIVAL 事件喂入同一
                                                 arrival heap（字段保持
                                                 (time_ns, priority, sequence,
                                                 kind, payload) 形状）
  iteration_complete 批（:3907-4006）             PREFILL_DRAIN /
    prefill 末 chunk（:3931-3984）                  _on_prefill_drain（逐行迁移）
    decode 完成（:3986-3999）                       _on_decode_complete
    下一 turn arrival（:4000-4006）                 _on_request_complete（同 tick）
  completion_order 批（:4008-4042）               _on_request_complete
    mark_complete / enforce_reserve / 快照          同名调用逐行迁移
  arrival 批（:4044-4058）                        _on_arrival（记 arrival →
                                                 pending_admissions → retry；
                                                 truncate_history 已随 sidecar
                                                 机制移除，recompute 单口径下
                                                 恒 no-op）
  admit_waiting_requests（:4060-4061）            _admit_pass 同 tick 末尾复查
  start_ready_iterations（:3833-3890）            _admit_pass 的发射部分：
                                                 **直接 Roofline 计时部分删除**（:3853-3858
                                                 估算与 :3885-3890 推事件），
                                                 排队/配对语义（busy 判定、
                                                 FCFS qp、active_decode 成员）
                                                 原样保留于账本逻辑。

关感知口径（方案 §4.1 第 6 条 + 合同⑥）：策略输入全部来自 Python 账本——
task_load_snapshot 三分量（Roofline 估计服务时间；queued 分量逐行复用
:3612-3638；running/active 分量的进度输入在阶段 1 为"段在飞=全量剩余"，
与离线 busy=False → 1.0（:3603-3604）对齐；阶段 3 感知打开后按 C++ 真实
完成事实折算）、HBM 可行掩码、edge_free 掩码（:466-483）、KV 快照
（KVCacheManager 只读复用，红线 #6-#9）。不新增任何 C++ 状态读取。

与离线蓝图的刻意差异（real-online 语义，合同⑦）：
  - 计时/迭代粒度：离线直接 Roofline 时钟 + 逐 chunk 迭代 -> 在线真实完成事件 +
    request-aggregated 构图（prefill 整段 + decode 整段，与离线 ET 粒度
    一致）；
  - decode 段发射：离线一次 iteration 混合 qp[0] chunk + 全部 active
    decode；在线 prefill 整段发射后，decode 段按 active_decode 成员逐
    request 发射（decode 固定 prefill 同实例，红线 #4：selected =
    state_index，decode_candidates = ()）；
  - 完成顺序：真实完成 tick 决定，不要求与离线决策序列 exact（合同⑦
    real-online 只验不变量与差异可解释性）。
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

from face_scheduler import (  # noqa: E402
    FaceInstanceSpec,
    KVCacheManager,
    edge_free_instance_mask,
    edge_instance_mask,
    build_instances,
    estimate_decode_remaining_task_load_ns,
    estimate_prefill_task_load_ns,
    select_prefill_instance,
)
from online.online_scheduler_base import (  # noqa: E402
    STAGE_DECODE,
    STAGE_PREFILL,
    STAGE_REQUEST,
    OnlineSchedulerBase,
)


class _OnlineInstanceState:
    """在线实例账本（离线 _InstanceRuntime 的在线子集）：qp = 已准入
    prefill FCFS 队列（deque[request_id]），active_decode = 已准入 decode
    列表，busy = 一个整段在飞，last_arrival_ns 供 ordering_key。"""

    __slots__ = ("index", "qp", "active_decode", "active_decode_lookup",
                 "busy", "last_arrival_ns")

    def __init__(self, *, index: int):
        self.index = index
        self.qp = deque()
        self.active_decode = []
        self.active_decode_lookup = set()
        self.busy = False
        self.last_arrival_ns = None


class _OnlineRequestRuntime:
    """在线请求运行账本（离线 _RequestRuntime 的在线子集 + plan dict 字段）。

    输入事实（request-neutral，来自 manifest）：history_tokens_before /
    prefill_context_tokens / final_context_tokens / prefill_length /
    decode_length / queue_index / session_id / turn_index。运行期事实由在线
    决策产出（语义与离线 _RequestRuntime 同名同义）。
    """

    __slots__ = (
        "request_id", "session_id", "turn_index", "queue_index",
        "prefill_length", "decode_length",
        "history_tokens_before", "prefill_tokens_to_process",
        "prefill_context_tokens", "final_context_tokens",
        "estimated_arrival_ns", "admission_time_ns", "hbm_wait_ns",
        "prefill_instance_index", "prefill_assignment_key",
        "prefill_instance_loads", "prefill_hbm_feasible_instances",
        "prefill_affinity_reason",
        "decode_instance_index",
        "history_source_instance_index", "history_transfer_bytes",
        "history_tokens_discarded", "history_location_before",
        "history_transfer", "history_evictions", "prefill_evictions",
        "prefill_decode_transfer", "decode_evictions", "completion_evictions",
        "kv_location_after_completion", "kv_instance_after_completion",
        "reserve_unmet_ranks",
        "admitted", "prefill_emitted", "decode_emitted", "completed",
        "completion_ns",
    )

    def __init__(self, record: dict) -> None:
        self.request_id = record["request_id"]
        self.session_id = record["session_id"]
        self.turn_index = record["turn_index"]
        self.queue_index = record["queue_index"]
        self.prefill_length = record["prefill_length"]
        self.decode_length = record["decode_length"]
        self.history_tokens_before = record["history_tokens_before"]
        self.prefill_tokens_to_process = (
            record["prefill_context_tokens"] - record["history_tokens_before"])
        self.prefill_context_tokens = record["prefill_context_tokens"]
        self.final_context_tokens = record["final_context_tokens"]
        self.estimated_arrival_ns = None
        self.admission_time_ns = None
        self.hbm_wait_ns = 0
        self.prefill_instance_index = None
        self.prefill_assignment_key = None
        self.prefill_instance_loads = ()
        self.prefill_hbm_feasible_instances = ()
        self.prefill_affinity_reason = None
        self.decode_instance_index = None
        self.history_source_instance_index = None
        self.history_transfer_bytes = 0
        # recompute 单口径下恒 0（截断通道已随 sidecar 移除）；字段保留
        # 以稳定 decision_log/manifest 输出 schema。
        self.history_tokens_discarded = 0
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
        self.admitted = False
        self.completion_ns = None
        self.prefill_emitted = False
        self.decode_emitted = False
        self.completed = False

    def plan_dict(self) -> dict:
        """graph_batch_builder 消费的 plan 字段（KV 决策字段在准入/完成时
        已落账本）。"""
        return {
            "request_id": self.request_id,
            "session_id": self.session_id,
            "turn_index": self.turn_index,
            "queue_index": self.queue_index,
            "prefill_length": self.prefill_length,
            "decode_length": self.decode_length,
            "prefill_instance_index": self.prefill_instance_index,
            "decode_instance_index": (
                self.decode_instance_index or self.prefill_instance_index),
            "prefill_affinity_reason": self.prefill_affinity_reason,
            "history_tokens_before": self.history_tokens_before,
            "prefill_context_tokens": self.prefill_context_tokens,
            "final_context_tokens": self.final_context_tokens,
            "admission_time_ns": self.admission_time_ns,
            "hbm_wait_ns": self.hbm_wait_ns,
            "history_source_instance_index": self.history_source_instance_index,
            "history_transfer_bytes": self.history_transfer_bytes,
            "history_tokens_discarded": self.history_tokens_discarded,
            "history_location_before": self.history_location_before,
            "history_transfer": self.history_transfer,
            "history_evictions": self.history_evictions,
            "prefill_evictions": self.prefill_evictions,
            "decode_evictions": self.decode_evictions,
            "prefill_decode_transfer": self.prefill_decode_transfer,
            "completion_evictions": self.completion_evictions,
            "kv_location_after_completion": self.kv_location_after_completion,
            "kv_instance_after_completion": self.kv_instance_after_completion,
        }


class Sh30OnlineScheduler(OnlineSchedulerBase):
    """strategy 变体：sh_3.0 三段式准入 + decode 同实例 + 三态 KV（关感知）。

    蓝本：已移除的离线 plan_face_requests。拓扑 / edge_free
    掩码 / KV 账本（与离线同一函数、同参数）在 __init__ 一次性构建，运行期
    策略输入全部来自这些 Python 账本。
    """

    def __init__(self, *, manifest, config, graph, digest_sink=None,
                 mode: str = "strategy", sensing: bool = False):
        super().__init__(
            manifest=manifest,
            config=config,
            replay=None,
            digest_sink=digest_sink,
            mode=mode,
            sensing=sensing,
        )
        if mode != "strategy":
            raise ValueError("Sh30OnlineScheduler requires mode == 'strategy'")
        self.graph = graph
        p_chunk = int(config.prefill_chunk_size)
        if p_chunk <= 0:
            raise ValueError("online path requires an explicit positive p_chunk")
        self.p_chunk = p_chunk

        # offline: face_scheduler.py 的 build_instances 同参。
        specs = tuple(
            FaceInstanceSpec(group.name, group.pg_name, group.ranks)
            for group in config.inference_groups
        )
        self.topology = build_instances(
            config.hardware, specs, require_equal_size=True)
        # offline: face_scheduler.py 的 edge_free_mask/edge_mask
        # 一次性计算。
        self.kv_manager = KVCacheManager(
            self.topology,
            config.model,
            reserve_context_tokens=config.kv_reserve_context_tokens,
        )
        self.edge_free_mask = edge_free_instance_mask(
            self.topology, self.kv_manager.edge_ranks)
        self.edge_mask = edge_instance_mask(
            self.topology, self.kv_manager.edge_ranks)
        self.instances = [
            _OnlineInstanceState(index=i)
            for i in range(len(self.topology.instances))
        ]
        # 合同⑨：average_decode_length = 物化期标定常数（离线同一推导
        # sum(decode)/len，与 --print-shell-config 交叉核对）；禁止在线
        # 从已到达请求算增量 mean（用户裁决 2026-08-15）。
        self.average_decode_length = config.source_average_decode_length
        self.hardware = config.hardware
        self.model = config.model
        self._prefill_task_cache = {}

        self.runtimes = [
            _OnlineRequestRuntime(record)
            for record in sorted(
                manifest["requests"], key=lambda item: item["queue_index"])
        ]
        self.runtime_by_request_id = {
            runtime.request_id: runtime for runtime in self.runtimes}
        by_turn = {}
        for runtime in self.runtimes:
            by_turn[(runtime.session_id, runtime.turn_index)] = runtime
        self.next_request = {
            runtime.request_id:
                by_turn.get((runtime.session_id, runtime.turn_index + 1))
            for runtime in self.runtimes
        }
        # graph builder 的 interval gate 发射需要下一 turn 的 plan 字段
        # （queue_index/hbm_wait_ns/request_id）；strategy 模式 hbm_wait_ns
        # 在准入时落账本，经 resolver 取实时值。
        self.graph.set_next_plan({
            runtime.request_id:
                (self.next_request[runtime.request_id].request_id
                 if self.next_request[runtime.request_id] else None)
            for runtime in self.runtimes
        })
        self.graph.set_plan_resolver(
            lambda request_id: self.runtime_by_request_id[
                request_id].plan_dict())

        self.arrival_heap = []
        self._sequence = 0
        # offline: face_scheduler.py 的 pending_admissions FIFO。
        self.pending_admissions = deque()
        self.completed_requests = 0
        # §7.3 ready frontier：非忙且有排队工作的实例集合。
        self._ready_frontier = set()

    # ------------------------------------------------------------- 策略 --

    def run_variant_policy(self, delta) -> None:
        """决策顺序逐行对应离线事件循环：completion 批（priority 0，含
        completion_order 处理）先于 arrival 批（priority 1），最后
        admit + start_ready_iterations（:4060-4062）。"""
        tick = delta["tick"]
        # ---- completion 批（offline: face_scheduler.py）----
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
        # ---- arrival 批（offline: face_scheduler.py）----
        for arrival in delta["arrivals"]:
            self._push_arrival(arrival, tick)
        self._drain_arrival_heap(tick)
        # ---- 准入/发射 pass（offline: face_scheduler.py）----
        self._admit_pass(tick)

    # ------------------------------------------------------------- 边界 --

    def _push_arrival(self, arrival: dict, tick: int) -> None:
        """offline: face_scheduler.py push_event(time_ns, 1,
        "arrival", index) 的在线等价（ingress ARRIVAL 事件喂入）。"""
        runtime = self.runtime_by_request_id[arrival["request_id"]]
        heapq.heappush(
            self.arrival_heap,
            (tick, 1, runtime.queue_index, self._sequence, "arrival",
             runtime))
        self._sequence += 1

    def _drain_arrival_heap(self, tick: int) -> None:
        """offline: face_scheduler.py 同 tick 批 + (priority,
        sequence) 排序（:3903）。"""
        due = []
        while self.arrival_heap and self.arrival_heap[0][0] <= tick:
            due.append(heapq.heappop(self.arrival_heap))
        due.sort(key=lambda item: (item[1], item[2]))
        for _, _, _, _, kind, runtime in due:
            if kind != "arrival":
                raise RuntimeError("arrival heap contains {!r}".format(kind))
            self._on_arrival(runtime, tick)

    def _on_arrival(self, runtime, tick: int) -> None:
        """offline: face_scheduler.py arrival 批单条。"""
        if runtime.estimated_arrival_ns is not None:
            raise RuntimeError(
                "request arrival was delivered more than once")
        runtime.estimated_arrival_ns = tick
        self.pending_admissions.append(runtime)
        self._retry = True

    def _on_prefill_drain(self, request_id: str, tick: int) -> None:
        """offline: face_scheduler.py prefill 末 chunk 完成分支
        （decode 固定同实例 :3944-3949），逐行迁移；构图替换为 decode 整段
        发射（聚合粒度）。"""
        runtime = self.runtime_by_request_id[request_id]
        state = self.instances[runtime.prefill_instance_index]
        state.busy = False
        if not state.qp or state.qp[0] is not runtime:
            raise RuntimeError("prefill FCFS queue order was corrupted")
        state.qp.popleft()
        if state.qp or state.active_decode:
            self._ready_frontier.add(state.index)
        else:
            self._ready_frontier.discard(state.index)
        # offline: face_scheduler.py（expand_prefill）
        runtime.prefill_evictions = self.kv_manager.expand_prefill(
            session_id=runtime.session_id,
            instance_index=state.index,
            context_tokens=runtime.prefill_context_tokens,
            trigger_request_id=runtime.request_id,
            reservation_request_id=runtime.request_id,
        )
        # offline: face_scheduler.py（decode 固定 prefill 同实例）
        selected = state.index
        runtime.decode_instance_index = selected
        reservation_move_evictions = (
            self.kv_manager.move_request_capacity_reservation(
                request_id=runtime.request_id,
                target_instance_index=selected,
            )
        )
        (runtime.prefill_decode_transfer,
         decode_move_evictions) = self.kv_manager.move_prefill_to_decode(
            session_id=runtime.session_id,
            target_instance_index=selected,
            trigger_request_id=runtime.request_id,
            reservation_request_id=runtime.request_id,
        )
        decode_growth_evictions = self.kv_manager.expand_decode(
            session_id=runtime.session_id,
            instance_index=selected,
            final_context_tokens=runtime.final_context_tokens,
            trigger_request_id=runtime.request_id,
            reservation_request_id=runtime.request_id,
        )
        runtime.decode_evictions = (
            reservation_move_evictions + decode_move_evictions
            + decode_growth_evictions)
        self.kv_manager.release_request_capacity_reservation(runtime.request_id)
        state.active_decode.append(runtime)
        state.active_decode_lookup.add(runtime)
        if not state.busy:
            self._ready_frontier.add(state.index)
        # ledger（阶段 3 感知账本，查询/审计输入不进判据）。
        self._ledger_admit(
            runtime.request_id, tick,
            {"type": "active_decode", "instance_index": selected})
        # decode 整段发射（聚合粒度；watch 注册 DECODE_COMPLETION）。
        self._emit_decode(runtime, tick)

    def _on_decode_complete(self, request_id: str, tick: int) -> None:
        """offline: face_scheduler.py decode 完成分支（下一次
        arrival 排程移至 _on_request_complete，同 tick 同顺序）。"""
        runtime = self.runtime_by_request_id[request_id]
        state = self.instances[runtime.decode_instance_index]
        if runtime not in state.active_decode_lookup:
            raise RuntimeError("decode queue membership was corrupted")
        state.active_decode_lookup.discard(runtime)
        state.active_decode.remove(runtime)
        state.busy = False
        if state.qp or state.active_decode:
            self._ready_frontier.add(state.index)
        else:
            self._ready_frontier.discard(state.index)
        runtime.completed = True
        runtime.completion_ns = tick  # 基类侧字段由 log_decision 行携带
        self.completed_requests += 1

    def _on_request_complete(self, request_id: str, tick: int) -> None:
        """offline: face_scheduler.py（completion_order 批：下一
        turn arrival 排程 :4000-4006 + mark_complete :4016-4020 +
        enforce_reserve :4021-4031 + 快照 :4032-4042）。"""
        runtime = self.runtime_by_request_id[request_id]
        # offline: face_scheduler.py 先 mark_complete /
        # enforce_reserve / 快照（completion_evictions 与终态 KV location
        # 落账本），再发射 interval gate（离线 writer :3064+ 同序）。
        # typed eviction：在线 runtime 是 manifest 派生账本（无 FaceRequest
        # 字段），完成请求自身的 next_trigger_type 经
        # config.request_queue[queue_index]（FaceTraceConfig 装载的
        # RequestSpec，与 manifest 同序）取回传给 mark_complete。
        self.kv_manager.mark_complete(
            runtime.session_id,
            tick,
            next_request_type=(
                self.config.request_queue[
                    runtime.queue_index].next_trigger_type
            ),
        )
        if runtime.decode_instance_index is None:
            raise RuntimeError("completed request has no Decode instance")
        (runtime.completion_evictions,
         runtime.reserve_unmet_ranks) = self.kv_manager.enforce_reserve(
            instance_index=runtime.decode_instance_index,
            trigger_request_id=runtime.request_id,
        )
        snapshot = self.kv_manager.session_snapshot(runtime.session_id)
        runtime.kv_location_after_completion = snapshot.location
        runtime.kv_instance_after_completion = snapshot.instance_index
        # completion 批（合同①）：completion_evictions + 下一 turn interval
        # gate 依赖登记。
        self.graph.emit_completion_batch(runtime.plan_dict())
        following = self.next_request[request_id]
        if following is not None:
            interval = self.config.request_queue[
                following.queue_index].inter_request_interval_ns
            if interval is None:
                raise RuntimeError(
                    "validated later request lost its interval")
            # offline: face_scheduler.py push_event(now + interval, 1,
            # "arrival", following) -> 在线 future alarm（alarm 时刻语义
            # 不变：完成 tick + interval）。
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
        # 阶段 3 感知账本：completed-unreconciled 核销。
        self.log_decision(
            {"kind": "completion", "request_id": runtime.request_id,
             "priority": 0},
            tick,
            decision={
                "kv_location_after_completion":
                    runtime.kv_location_after_completion,
                "kv_instance_after_completion":
                    runtime.kv_instance_after_completion,
                "completion_evictions": [
                    _transfer_summary(transfer)
                    for transfer in runtime.completion_evictions],
                "reserve_unmet_ranks": list(runtime.reserve_unmet_ranks),
            },
        )

    # ------------------------------------------------------------- 准入 --

    def _admit_pass(self, tick: int) -> None:
        """offline: face_scheduler.py（admit_waiting_requests +
        start_ready_iterations 的排队/发射部分；直接 Roofline 计时删除）。"""
        self._retry = False
        self._admit_waiting_requests(tick)
        self._start_ready_emissions(tick)

    def _admit_waiting_requests(self, now_ns: int) -> None:
        """offline: face_scheduler.py admit_waiting_requests，
        逐行对应（blocked FIFO 重排语义保留）。"""
        blocked = deque()
        while self.pending_admissions:
            runtime = self.pending_admissions.popleft()
            if not self._try_admit_request(runtime, now_ns):
                blocked.append(runtime)
        self.pending_admissions.extend(blocked)

    def _try_admit_request(self, runtime, now_ns: int) -> bool:
        """offline: face_scheduler.py try_admit_request，逐行
        迁移（三段式准入 + HBM 过滤 + reserve/prepare + qp 入队）。"""
        if runtime.estimated_arrival_ns is None:
            raise RuntimeError("request cannot be admitted before arrival")
        hbm_feasible_instances = (
            self.kv_manager.request_hbm_feasible_instances(
                session_id=runtime.session_id,
                final_context_tokens=runtime.final_context_tokens,
            ))
        if not any(hbm_feasible_instances):
            eventually_feasible = (
                self.kv_manager.request_hbm_eventually_feasible_instances(
                    session_id=runtime.session_id,
                    final_context_tokens=runtime.final_context_tokens,
                ))
            if not any(eventually_feasible):
                raise ValueError(
                    "request {} final KV cannot fit on any eligible empty "
                    "instance; final_context_tokens={}".format(
                        runtime.request_id, runtime.final_context_tokens))
            return False

        snapshots = tuple(
            self._task_load_snapshot(state, now_ns)
            for state in self.instances)
        runtime.prefill_hbm_feasible_instances = hbm_feasible_instances
        if runtime.session_id in self.kv_manager.session_ids:
            history_snapshot = self.kv_manager.session_snapshot(
                runtime.session_id)
        else:
            history_snapshot = None
        if history_snapshot is None:
            if not any(self.edge_free_mask):
                # 分支 1a：拓扑级无非边缘实例 → 回退全集负载均衡
                selected = select_prefill_instance(
                    snapshots, hbm_feasible_instances)
                runtime.prefill_affinity_reason = (
                    "first_request_edge_fallback")
            else:
                # 分支 1：session 首请求——排除边缘实例
                candidate_mask = tuple(
                    feasible and self.edge_free_mask[i]
                    for i, feasible in enumerate(hbm_feasible_instances))
                if not any(candidate_mask):
                    eventually_mask = (
                        self.kv_manager
                        .request_hbm_eventually_feasible_instances(
                            session_id=runtime.session_id,
                            final_context_tokens=(
                                runtime.final_context_tokens)))
                    if not any(
                            ev and self.edge_free_mask[i]
                            for i, ev in enumerate(eventually_mask)):
                        raise RuntimeError(
                            "edge-free candidate set can never become "
                            "HBM-feasible")
                    return False
                selected = select_prefill_instance(snapshots, candidate_mask)
                runtime.prefill_affinity_reason = "first_request_non_edge"
        elif history_snapshot.location in (
                KVCacheManager.LOCAL_HBM,
                KVCacheManager.PARTIAL_HBM_REMOTE):
            # 分支 2：LOCAL/PARTIAL sticky 到驻留实例
            if history_snapshot.instance_index is None:
                raise RuntimeError("resident history lost its instance")
            selected = history_snapshot.instance_index
            if not hbm_feasible_instances[selected]:
                return False
            runtime.prefill_affinity_reason = (
                "resident_prefix_layers"
                if history_snapshot.location
                == KVCacheManager.PARTIAL_HBM_REMOTE
                else "resident_local_hbm")
        else:
            # 分支 3：REMOTE_MEMORY —— 边缘实例集合内负载均衡
            candidate_mask = tuple(
                feasible and self.edge_mask[i]
                for i, feasible in enumerate(hbm_feasible_instances))
            if not any(candidate_mask):
                eventually_mask = (
                    self.kv_manager
                    .request_hbm_eventually_feasible_instances(
                        session_id=runtime.session_id,
                        final_context_tokens=(
                            runtime.final_context_tokens)))
                if not any(
                        ev and self.edge_mask[i]
                        for i, ev in enumerate(eventually_mask)):
                    raise RuntimeError(
                        "edge candidate set can never become "
                        "HBM-feasible")
                return False
            selected = select_prefill_instance(snapshots, candidate_mask)
            runtime.prefill_affinity_reason = "remote_edge_load_balance"

        selected_snapshot = snapshots[selected]
        admission_evictions = self.kv_manager.reserve_request_capacity(
            request_id=runtime.request_id,
            session_id=runtime.session_id,
            instance_index=selected,
            final_context_tokens=runtime.final_context_tokens,
        )
        runtime.prefill_instance_index = selected
        runtime.prefill_assignment_key = selected_snapshot.ordering_key
        runtime.prefill_instance_loads = snapshots
        runtime.admission_time_ns = now_ns
        runtime.hbm_wait_ns = now_ns - runtime.estimated_arrival_ns
        (runtime.history_location_before,
         runtime.history_transfer,
         prepare_evictions) = self.kv_manager.prepare_prefill(
            session_id=runtime.session_id,
            target_instance_index=selected,
            history_tokens=runtime.history_tokens_before,
            trigger_request_id=runtime.request_id,
            reservation_request_id=runtime.request_id,
        )
        runtime.history_evictions = admission_evictions + prepare_evictions
        if runtime.turn_index > 0:
            if runtime.history_location_before is None:
                raise RuntimeError(
                    "session {} has no prior KV state".format(
                        runtime.session_id))
            runtime.history_source_instance_index = (
                runtime.history_location_before.instance_index)
            runtime.history_transfer_bytes = (
                0 if runtime.history_transfer is None
                or runtime.history_transfer.kind == "local_hit"
                else runtime.history_transfer.total_bytes)
        self.instances[selected].qp.append(runtime)
        self.instances[selected].last_arrival_ns = now_ns
        runtime.admitted = True
        # §7.3 ready frontier：实例非忙且有排队工作（离线 start_ready_
        # iterations :3835 的循环条件在在线的增量等价——遗漏此标记会使
        # 发射 pass 永远空转，全部 request 卡在 admitted 层）。
        if not self.instances[selected].busy:
            self._ready_frontier.add(selected)
        self._ledger_admit(
            runtime.request_id, now_ns,
            {"type": "prefill_qp", "instance_index": selected})
        return True

    # ------------------------------------------------------------- 发射 --

    def _start_ready_emissions(self, now_ns: int) -> None:
        """offline: face_scheduler.py start_ready_iterations 的
        发射部分（直接 Roofline 计时删除；busy/FCFS/active_decode 配对语义保留）。
        §7.3 ready frontier：只访问非忙且有排队工作的实例（sorted 保持
        实例 index 序 = 离线 :3835 的循环序）。"""
        for instance_index in sorted(self._ready_frontier):
            state = self.instances[instance_index]
            if state.busy or (not state.qp and not state.active_decode):
                continue
            if state.qp:
                runtime = state.qp[0]
                if not runtime.prefill_emitted:
                    self._emit_prefill(runtime, now_ns)
            for runtime in tuple(state.active_decode):
                if not runtime.decode_emitted:
                    self._emit_decode(runtime, now_ns)
            state.busy = True
            self._ready_frontier.discard(state.index)

    def _emit_prefill(self, runtime, tick: int) -> None:
        """聚合粒度发射 prefill 整段（watch 注册 PREFILL_DRAIN；
        offline: face_scheduler.py 的发射对象为整段而非 chunk）。"""
        plan = runtime.plan_dict()
        members = self.graph.emit_prefill_batch(plan)
        runtime.prefill_emitted = True
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
        self._batch["assignments"].append({
            "request_id": runtime.request_id,
            "prefill_instance_index": runtime.prefill_instance_index,
            # prefill 边界 decode 尚未选定（decode 固定 prefill 同实例，
            # 红线 #4：PREFILL_DRAIN 时 selected = state_index）；assignment
            # 结构校验要求非负，填 prefill 实例（与最终 decode 实例恒等）。
            "decode_instance_index": (
                runtime.decode_instance_index
                if runtime.decode_instance_index is not None
                else runtime.prefill_instance_index),
            "prefill_affinity_reason": runtime.prefill_affinity_reason,
        })
        self.log_decision(
            {"kind": "prefill", "request_id": runtime.request_id,
             "priority": 0},
            tick,
            decision={
                "prefill_instance_index": runtime.prefill_instance_index,
                "prefill_assignment_key": list(
                    runtime.prefill_assignment_key),
                "prefill_affinity_reason": runtime.prefill_affinity_reason,
                "estimated_arrival_ns": runtime.estimated_arrival_ns,
                "admission_time_ns": runtime.admission_time_ns,
                "hbm_wait_ns": runtime.hbm_wait_ns,
                "history_tokens_before": runtime.history_tokens_before,
                "prefill_context_tokens": runtime.prefill_context_tokens,
                "history_source_instance_index":
                    runtime.history_source_instance_index,
                "history_transfer_bytes": runtime.history_transfer_bytes,
                "history_tokens_discarded":
                    runtime.history_tokens_discarded,
            },
        )

    def _emit_decode(self, runtime, tick: int) -> None:
        """聚合粒度发射 decode 整段（watch 注册 DECODE_COMPLETION；C++ 同
        fire 推 DECODE_COMPLETION + REQUEST_COMPLETE 两条 completed_groups）。"""
        plan = runtime.plan_dict()
        members = self.graph.emit_decode_batch(plan)
        runtime.decode_emitted = True
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
        self.log_decision(
            {"kind": "decode", "request_id": runtime.request_id,
             "priority": 0},
            tick,
            decision={
                "decode_instance_index": runtime.decode_instance_index,
            },
        )

    # ------------------------------------------------------------- 负载 --

    def _prefill_chunk_task_load_ns(self, *, instance_size, chunk_tokens,
                                    context_tokens) -> int:
        """offline: face_scheduler.py（缓存同款）。"""
        key = (instance_size, chunk_tokens, context_tokens)
        if key not in self._prefill_task_cache:
            self._prefill_task_cache[key] = estimate_prefill_task_load_ns(
                self.hardware, self.model,
                instance_size=instance_size, chunk_tokens=chunk_tokens,
                context_tokens=context_tokens)
        return self._prefill_task_cache[key]

    def _task_load_snapshot(self, state, now_ns: int):
        """offline: face_scheduler.py task_load_snapshot 的在线
        子集。三分量口径（合同⑥），每个请求恰计一次：
          - queued_prefill：逐 chunk Roofline 求和（函数逐行复用
            :3612-3638；仅未发射请求 = 全量剩余 chunk，与离线
            prompt_tokens_processed=0 的排队请求一致；在飞请求只进
            running 分量——恰计一次）；
          - running_prefill：在飞段的全量 chunk 负载（fraction=1.0。离线
            = 当前 chunk×fraction + 剩余 chunk 合计恰一次；在线 phase-1
            无 chunk 级进度事件，以全量×1.0 上界近似，阶段 3 改真实进度）；
          - active_decode：estimate_decode_remaining_task_load_ns 逐请求
            （标定常数 average_decode_length；generated_tokens=0，
            fraction=1.0——阶段 3 改真实完成事件驱动的账本进度）。"""
        instance_size = self.topology.instance(state.index).size
        # queued_prefill（offline: :3612-3638 逐行复用）
        queued_load_ns = 0
        for runtime in state.qp:
            if runtime.prefill_emitted:
                # 在飞段只进 running 分量（恰计一次；离线蓝本将其当前 chunk
                # 折入 running、剩余 chunk 计入 queued，phase-1 无进度事件，
                # 以全量×1.0 上界近似，全部记在 running）。
                continue
            remaining_tokens = runtime.prefill_tokens_to_process
            processed = 0
            while remaining_tokens > 0:
                chunk_tokens = min(self.p_chunk, remaining_tokens)
                context_tokens = (
                    runtime.history_tokens_before + processed + chunk_tokens)
                queued_load_ns += self._prefill_chunk_task_load_ns(
                    instance_size=instance_size, chunk_tokens=chunk_tokens,
                    context_tokens=context_tokens)
                processed += chunk_tokens
                remaining_tokens -= chunk_tokens
        # running_prefill（fraction = 1.0）
        running_prefill_load_ns = 0
        if state.busy and state.qp:
            runtime = state.qp[0]
            if runtime.prefill_emitted:
                processed = 0
                remaining_tokens = runtime.prefill_tokens_to_process
                while remaining_tokens > 0:
                    chunk_tokens = min(self.p_chunk, remaining_tokens)
                    context_tokens = (
                        runtime.history_tokens_before + processed
                        + chunk_tokens)
                    running_prefill_load_ns += (
                        self._prefill_chunk_task_load_ns(
                            instance_size=instance_size,
                            chunk_tokens=chunk_tokens,
                            context_tokens=context_tokens))
                    processed += chunk_tokens
                    remaining_tokens -= chunk_tokens
        # active_decode（offline: :3665-3684；fraction = 1.0）
        active_decode_load_ns = 0
        for runtime in state.active_decode:
            active_decode_load_ns += estimate_decode_remaining_task_load_ns(
                self.hardware, self.model,
                instance_size=instance_size,
                current_context_tokens=runtime.prefill_context_tokens,
                generated_tokens=0,
                average_decode_length=self.average_decode_length,
                running_step_fraction_remaining=1.0,
            )
        from face_scheduler import InstanceTaskLoadSnapshot
        return InstanceTaskLoadSnapshot(
            instance_index=state.index,
            running_prefill_task_load_ns=running_prefill_load_ns,
            queued_prefill_task_load_ns=queued_load_ns,
            active_decode_task_load_ns=active_decode_load_ns,
            last_arrival_ns=state.last_arrival_ns,
        )

    def _sensing_ready_view(self) -> dict:
        """§7.3/§10.1 ready 层边界视图（查询/审计输入，不进判据）。"""
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

    # --------------------------------------------------------------- 收尾 --

    def verify_run_end(self) -> None:
        """基类协议校验之上，叠加离线 :4064-4076 的收尾断言 + §7.3 结束
        审计（arrival heap / ready frontier 全空）。"""
        super().verify_run_end()
        if self.pending_admissions:
            pending_ids = [
                runtime.request_id for runtime in self.pending_admissions]
            raise RuntimeError(
                "strategy run ended with blocked HBM admissions: "
                "{}".format(pending_ids[:5]))
        if self.completed_requests != len(self.runtimes):
            raise RuntimeError(
                "strategy run ended with {}/{} requests complete".format(
                    self.completed_requests, len(self.runtimes)))
        if any(state.busy or state.qp or state.active_decode
               for state in self.instances):
            raise RuntimeError(
                "strategy run ended with non-idle instance state")
        if self.arrival_heap:
            raise RuntimeError(
                "run ended with {} unconsumed arrival events".format(
                    len(self.arrival_heap)))
        if self._ready_frontier:
            raise RuntimeError(
                "run ended with non-empty ready frontier: {!r}".format(
                    sorted(self._ready_frontier)))


def _transfer_summary(transfer):
    """KVTransfer 的日志摘要（online_decision_log 行内嵌）。"""
    if transfer is None:
        return None
    return {
        "kind": transfer.kind,
        "phase": transfer.phase,
        "reason": transfer.reason,
        "session_id": transfer.session_id,
        "total_bytes": transfer.total_bytes,
        "layer_start": transfer.layer_start,
        "layer_end": transfer.layer_end,
    }
