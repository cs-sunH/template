#!/usr/bin/env python3
"""sh20_online_scheduler.py -- sh_2.0 关感知策略调度器（strategy 模式，步骤 1-9）。

以已移除的离线 plan_face_requests（2026-08-21 离线 planner 清除批删去）为蓝本逐行迁移，保持决策
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
  arrival 批（:4004-4018）                         _on_arrival（arrival 落账 →
                                                 pending_admissions；2026-08-21
                                                 起无 truncate 通道，discarded 恒 0）
  admit_waiting_requests（:3764-3770）             _admit_pass 头部（各决策边界
                                                 触发，与离线 retry_admissions
                                                 门控同集合）
  start_ready_iterations（:3772-3829）              _admit_pass 尾部：实例空闲
                                                 → 发射 qp[0] 的 prefill 整段；
                                                 **离线计时表删除**（offline-only，
                                                 在线由 C++ 真实完成事件推进；
                                                 PD 混合排队语义保留于账本，
                                                 不体现在图结构上）

关感知口径（§4.1 第 6 条）：策略输入全部来自 Python 账本——task-load 三分量、
KV 容量/位置/可行性（KVCacheManager）、精确 Roofline 增量代价
（select_decode_instance 直接计算）。不新增任何
C++ 状态读取。

与离线蓝图的刻意差异（real-online 语义，合同⑦ Tier B real-online 验收）：
  - 计时/迭代粒度：离线 Roofline 时钟 + 逐 chunk 迭代 → 在线真实完成事件 +
    request-aggregated 构图；
  - task-load 三分量口径：running prefill 与 queued prefill 均按 512-chunk
    逐块求和（在飞段 fraction=1.0 全量剩余、恰计一次——在线无 chunk 级
    进度事件的上界近似），active decode 剩余步数在 request-aggregated
    语义下取整段剩余（current_decode_token=prefill_context），公式与参数
    与离线同一估算器（estimate_prefill_task_load_ns /
    estimate_decode_remaining_task_load_ns / _prefill_chunk_task_load_ns
    缓存同款）；完成时序由真实物理决定，不要求与离线 Roofline 时钟 exact；
  - 实例 busy 语义：在线 busy 覆盖"一个 prefill 整段在飞"；decode 段发射
    不受 busy 门（strategy 保持物理跨 request 链，per-rank previous_id
    天然串行化同实例段）。
"""

import heapq
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
    FaceInstanceSpec,
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
        add("history_prefix_transfer", runtime.history_prefix_transfer)
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
    """在线实例账本（离线 _InstanceRuntime，face_scheduler.py 的
    在线子集；iteration_* 计时字段删除——在线无预计算迭代时钟）。"""

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
        p_chunk = int(config.prefill_chunk_size)
        if p_chunk <= 0:
            raise ValueError("online path requires an explicit positive p_chunk")
        self.p_chunk = p_chunk
        # task-load 逐 chunk 估计缓存（同款 key 与已移除的离线 plan_face_requests 的
        # prefill_task_cache / sh_3.0 在线版一致）。
        self._prefill_task_cache: dict[tuple[int, int, int], int] = {}
        # 改法A：estimate_decode_remaining_task_load_ns 的全参 key memo
        # （与 _prefill_task_cache 同款；见 _decode_task_load_ns_cached）。
        self._decode_task_load_cache = {}

        # offline: face_scheduler.py（topology）
        specs = tuple(
            FaceInstanceSpec(group.name, group.pg_name, group.ranks)
            for group in config.inference_groups
        )
        self.topology = build_instances(
            config.hardware, specs, require_equal_size=True)

        # offline: face_scheduler.py（_validate_and_expand_requests）。在线用
        # 同一函数推导 runtime（含 prefix clamp）；Decode 候选的 Roofline
        # 代价在决策边界以当前精确 token 直接计算。
        requests = _to_scheduler_requests(config.request_queue)
        self.runtimes, self._next_request_map = _validate_and_expand_requests(
            requests, self.p_chunk)
        # offline: face_scheduler.py（graph + KV 账本；Python KV 账本
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

        # offline: face_scheduler.py
        self.arrival_heap = []
        self._sequence = 0
        self.pending_admissions = deque()

        # 改法D：KV 账本纪元重试门（face capacity_epoch 的调度器全局版——
        # sh 系列 try_admit 的全部 False 判据是"全账本 HBM 可行性纯函数
        # ∧ 静态掩码"，实例账本（qp/busy/active_decode）只影响选哪个、
        # 不影响能不能）。_kv_ledger_epoch 在 9 个 KV 变更点后各 bump 一次；
        # _admit_attempt_epoch 记录各 pending 条目上次失败时的纪元
        # （sh_2.0 的 key = request_index，pending 队列元素即 index），
        # 纪元未变则该条目本批跳过重试（重试必返同样的 False）。
        self._kv_ledger_epoch = 0
        self._admit_attempt_epoch = {}
        # 影子验证开关（SH_ADMIT_GATE_VERIFY=1）：门跳过的条目仍完整评估
        # 并断言必返 False——验证跑零收益、全检查；不设或非"1"则正常运行。
        self._admit_gate_verify = (
            os.environ.get("SH_ADMIT_GATE_VERIFY") == "1")

        self.runtime_by_request_id = {
            runtime.request.request_id: runtime for runtime in self.runtimes}
        self._runtime_index = {
            runtime.request.request_id: index
            for index, runtime in enumerate(self.runtimes)}
        # offline: face_scheduler.py（next_request）
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
        # offline: face_scheduler.py（arrival priority 1）。键含
        # queue_index——同 tick 到期项按冻结队列序稳定弹出，与 face:566 /
        # wscllm:522 / sh_3.0:434 的显式冻结队列序键对齐；基类 schema 门
        # 已保证 arrivals 队列序非降（queue_index 与 sequence 在 push 序中
        # 同向单调），插入后排序结果不变。
        index = self._runtime_index[arrival["request_id"]]
        runtime = self.runtimes[index]
        heapq.heappush(
            self.arrival_heap,
            (tick, 1, runtime.request.queue_index, self._sequence, "arrival",
             index))
        self._sequence += 1

    def _drain_arrival_heap(self, tick: int) -> None:
        # offline: face_scheduler.py（arrival 批）
        while self.arrival_heap and self.arrival_heap[0][0] <= tick:
            _, _, _, _, _, index = heapq.heappop(self.arrival_heap)
            runtime = self.runtimes[index]
            if runtime.estimated_arrival_ns is not None:
                raise RuntimeError("request arrival was delivered more than once")
            runtime.estimated_arrival_ns = tick
            # recompute 单口径（2026-08-21 起）：turn-0 前缀已折入队列
            # prefill_length，session 历史由 KV 账本动态维护，不存在源声明
            # 前缀截断通道；字段保留、恒 0（与离线 face_scheduler 同口径）。
            runtime.history_tokens_discarded = 0
            self.pending_admissions.append(index)
            # 阶段 3 感知账本：进入 admitted 层（排队类型 prefill_qp =
            # 已准入未发射的 prefill 排队，contract ⑥ 三型之一；sh_2.0 的
            # 实例选择推迟到 HBM 准入（_try_admit_request），到达点
            # instance 键缺省）。查询/审计数据，不进策略判据。
            self._ledger_admit(runtime.request.request_id, tick,
                               {"type": "prefill_qp"})

    def _on_prefill_drain(self, request_id: str, tick: int) -> None:
        """离线 prefill 收尾分支（:3864-3944）：调用顺序与离线逐行一致。"""
        index = self._runtime_index[request_id]
        runtime = self.runtimes[index]
        state = self.instances[runtime.prefill_instance_index]
        # offline: face_scheduler.py
        state.busy = False
        if not state.qp or state.qp[0] != index:
            raise RuntimeError("prefill FCFS queue order was corrupted")
        state.qp.popleft()
        runtime.prefill_complete_ns = tick
        # offline: face_scheduler.py（expand_prefill）
        runtime.prefill_evictions = self.kv_manager.expand_prefill(
            session_id=runtime.request.session_id,
            instance_index=state.index,
            context_tokens=runtime.prefill_context_tokens,
            trigger_request_id=runtime.request.request_id,
            reservation_request_id=runtime.request.request_id,
        )
        self._bump_kv_ledger_epoch()  # 改法D：expand_prefill 后 bump
        # offline: face_scheduler.py（select_decode_instance；精确 Roofline
        # 增量代价按当前候选状态直接计算）
        has_prefill_work = [bool(instance.qp) for instance in self.instances]
        active_tokens = [
            [self.runtimes[i].current_decode_token
             for i in instance.active_decode]
            for instance in self.instances
        ]
        selected, costs = select_decode_instance(
            hardware=self.config.hardware,
            model=self.config.model,
            topology=self.topology,
            graph=self.graph_topology,
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
        # offline: face_scheduler.py
        reservation_move_evictions = (
            self.kv_manager.move_request_capacity_reservation(
                request_id=runtime.request.request_id,
                target_instance_index=selected,
            )
        )
        self._bump_kv_ledger_epoch()  # 改法D：move_request_capacity_reservation 后 bump
        # offline: face_scheduler.py
        (runtime.prefill_decode_transfer,
         decode_move_evictions) = self.kv_manager.move_prefill_to_decode(
            session_id=runtime.request.session_id,
            target_instance_index=selected,
            trigger_request_id=runtime.request.request_id,
            reservation_request_id=runtime.request.request_id,
        )
        self._bump_kv_ledger_epoch()  # 改法D：move_prefill_to_decode 后 bump
        # offline: face_scheduler.py
        decode_growth_evictions = self.kv_manager.expand_decode(
            session_id=runtime.request.session_id,
            instance_index=selected,
            final_context_tokens=runtime.final_context_tokens,
            trigger_request_id=runtime.request.request_id,
            reservation_request_id=runtime.request.request_id,
        )
        self._bump_kv_ledger_epoch()  # 改法D：expand_decode 后 bump
        runtime.decode_evictions = (
            reservation_move_evictions + decode_move_evictions
            + decode_growth_evictions
        )
        # offline: face_scheduler.py
        runtime.kv_allocation = self.kv_manager.allocation_for_session(
            session_id=runtime.request.session_id,
            request_id=runtime.request.request_id,
        )
        self.kv_manager.release_request_capacity_reservation(
            runtime.request.request_id)
        self._bump_kv_ledger_epoch()  # 改法D：release_request_capacity_reservation 后 bump
        # offline: face_scheduler.py
        self.instances[selected].active_decode.append(index)
        runtime.decode_start_ns = tick
        # 发射 decode 整段 + DECODE_COMPLETION watch。assignment 在 decode
        # 决策后追加（sh_2.0 的 decode 实例在 prefill 完成时才决策——
        # GraphBatch 校验要求 prefill/decode 实例索引非负齐备）。
        members = self.graph.emit_decode_batch(self._plan_of(runtime))
        # 阶段 3 感知账本：admitted 层排队类型更新（active_decode，face
        # :754 同款；sh_2.0 无 waiting_decode 中间态——decode 在 prefill
        # 完成时即决策即发射，准入更新与发射钩子合位点）+ 发射记录 +
        # issued 层写入。查询/审计数据，不进策略判据。
        self._ledger_admit(request_id, tick,
                           {"type": "active_decode",
                            "instance_index": runtime.decode_instance_index})
        self._note_emitted(request_id, STAGE_DECODE)
        self._ledger_issue(request_id, tick, STAGE_DECODE,
                           runtime.decode_instance_index)
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
        # offline: face_scheduler.py（mark_complete）
        self.kv_manager.mark_complete(
            runtime.request.session_id,
            tick,
            next_request_type=runtime.request.next_trigger_type,
        )
        self._bump_kv_ledger_epoch()  # 改法D：mark_complete 后 bump
        # offline: face_scheduler.py（enforce_reserve）
        (runtime.completion_evictions,
         runtime.reserve_unmet_ranks) = self.kv_manager.enforce_reserve(
            instance_index=runtime.decode_instance_index,
            trigger_request_id=runtime.request.request_id,
        )
        self._bump_kv_ledger_epoch()  # 改法D：enforce_reserve 后 bump
        # offline: face_scheduler.py（快照）
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
            # offline: face_scheduler.py（push_event(now+interval,1)）
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

    def _bump_kv_ledger_epoch(self) -> None:
        """改法D：KV 账本纪元 +1。仅调度器的 9 个 KV 变更点后调用
        （见各调用处注释）；可行性读取与 _check_invariants 只读、不 bump。"""
        self._kv_ledger_epoch += 1

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
        selected = select_prefill_instance(snapshots, hbm_feasible_instances)

        selected_snapshot = snapshots[selected]
        admission_evictions = self.kv_manager.reserve_request_capacity(
            request_id=runtime.request.request_id,
            session_id=runtime.request.session_id,
            instance_index=selected,
            final_context_tokens=runtime.final_context_tokens,
        )
        self._bump_kv_ledger_epoch()  # 改法D：reserve_request_capacity 后 bump
        runtime.prefill_instance_index = selected
        runtime.prefill_assignment_key = selected_snapshot.ordering_key
        runtime.prefill_instance_loads = snapshots
        runtime.admission_time_ns = now_ns
        (runtime.history_location_before,
         runtime.history_prefix_transfer,
         runtime.history_transfer,
         prepare_evictions) = self.kv_manager.prepare_prefill(
            session_id=runtime.request.session_id,
            target_instance_index=selected,
            history_tokens=runtime.history_tokens_before,
            trigger_request_id=runtime.request.request_id,
            reservation_request_id=runtime.request.request_id,
        )
        self._bump_kv_ledger_epoch()  # 改法D：prepare_prefill 后 bump
        runtime.history_evictions = admission_evictions + prepare_evictions
        if runtime.request.turn_index > 0:
            if runtime.history_location_before is None:
                raise RuntimeError(
                    f"session {runtime.request.session_id} has no prior KV state")
            runtime.history_source_instance_index = (
                runtime.history_location_before.instance_index)
            runtime.history_transfer_bytes = sum(
                transfer.total_bytes
                for transfer in (
                    runtime.history_prefix_transfer,
                    runtime.history_transfer,
                )
                if transfer is not None and transfer.kind != "local_hit")
        self.instances[selected].qp.append(request_index)
        self.instances[selected].last_arrival_ns = now_ns
        return True

    def _admit_pass(self, now_ns: int) -> None:
        """offline: face_scheduler.py（admit_waiting_requests）+
        :3772-3829（start_ready_iterations 的在线形态）。"""
        blocked = deque()
        while self.pending_admissions:
            request_index = self.pending_admissions.popleft()
            if (self._admit_attempt_epoch.get(request_index)
                    == self._kv_ledger_epoch):
                if self._admit_gate_verify:
                    # 影子断言：门判跳过 ≡ 重试必返 False。
                    if self._try_admit_request(request_index, now_ns):
                        raise RuntimeError(
                            "admit gate equivalence violated: request {} was "
                            "admitted on a skipped retry (kv epoch {})".format(
                                request_index, self._kv_ledger_epoch))
                    self._admit_attempt_epoch[request_index] = (
                        self._kv_ledger_epoch)
                    blocked.append(request_index)
                    continue
                # 改法D：上次失败以来 KV 账本未变 → False 判据输入未变，
                # 重试必返同样的 False，跳过（FIFO 位置不变）。
                blocked.append(request_index)
                continue
            if not self._try_admit_request(request_index, now_ns):
                self._admit_attempt_epoch[request_index] = (
                    self._kv_ledger_epoch)
                blocked.append(request_index)
            else:
                self._admit_attempt_epoch.pop(
                    request_index, None)  # 成功即清除（有界）
        self.pending_admissions.extend(blocked)

        for state in self.instances:
            # offline: face_scheduler.py（实例空闲且有工作）
            if state.busy or (not state.qp and not state.active_decode):
                continue
            if state.qp:
                index = state.qp[0]
                runtime = self.runtimes[index]
                # offline: face_scheduler.py（prefill_start）
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
                # 阶段 3 感知账本：发射记录 + issued 层写入（face :777-779
                # 同款；查询/审计数据，不进策略判据）。
                self._note_emitted(runtime.request.request_id, STAGE_PREFILL)
                self._ledger_issue(
                    runtime.request.request_id, now_ns, STAGE_PREFILL,
                    runtime.prefill_instance_index)

    # ------------------------------------------------- task-load 三分量 --

    def _prefill_chunk_task_load_ns(self, *, instance_size: int,
                                    chunk_tokens: int,
                                    context_tokens: int) -> int:
        """逐 chunk Roofline 估计（缓存 key 同离线 prefill_task_cache；
        sh_3.0 修复版同款助手）。"""
        key = (instance_size, chunk_tokens, context_tokens)
        if key not in self._prefill_task_cache:
            self._prefill_task_cache[key] = estimate_prefill_task_load_ns(
                hardware=self.config.hardware,
                model=self.config.model,
                instance_size=instance_size,
                chunk_tokens=chunk_tokens,
                context_tokens=context_tokens,
            )
        return self._prefill_task_cache[key]

    def _decode_task_load_ns_cached(self, *, instance_size,
                                    current_context_tokens, generated_tokens,
                                    average_decode_length,
                                    running_step_fraction_remaining):
        """改法A：estimate_decode_remaining_task_load_ns 的全参 key memo
        （与 _prefill_task_cache 同款）。key 含今天恒定的 generated_tokens/
        fraction/ADL——阶段 3 引入真实进度后自然分区，key 结构不变；
        hardware/model 为运行期不可变量，经绑定不入 key。"""
        key = (instance_size, current_context_tokens, generated_tokens,
               average_decode_length, running_step_fraction_remaining)
        cached = self._decode_task_load_cache.get(key)
        if cached is None:
            cached = estimate_decode_remaining_task_load_ns(
                self.config_hardware(), self.config_model(),
                instance_size=instance_size,
                current_context_tokens=current_context_tokens,
                generated_tokens=generated_tokens,
                average_decode_length=average_decode_length,
                running_step_fraction_remaining=running_step_fraction_remaining)
            self._decode_task_load_cache[key] = cached
        return cached

    def _task_load_snapshot(self, state: _OnlineInstanceState,
                            now_ns: int) -> InstanceTaskLoadSnapshot:
        """离线 task_load_snapshot（face_scheduler.py 起）的在线复刻。
        三分量口径（每个请求恰计一次）：
          - running_prefill：在飞段全量剩余**逐 512-chunk 求和，恰计一次**
            （在线无 chunk 级进度事件，fraction=1.0 上界近似；与 sh_3.0
            修复版及离线蓝本的逐 chunk 口径同形，时间折算归阶段 3 事件
            驱动账本）；
          - queued_prefill：未在飞请求逐 chunk 求和（在飞请求折 0，见
            _queued_prefill_task_load_ns）；
          - active_decode：estimate_decode_remaining_task_load_ns 逐请求，
            generated_tokens 在线恒 0（current_decode_token 段内不推进）。"""
        instance_size = self.topology.instance(state.index).size

        running_prefill_load_ns = 0
        running_index = (
            state.qp[0] if (state.busy and state.qp) else None)
        if running_index is not None:
            runtime = self.runtimes[running_index]
            processed_tokens = runtime.prompt_tokens_processed
            remaining_tokens = (
                runtime.prefill_tokens_to_process - processed_tokens)
            while remaining_tokens > 0:
                chunk_tokens = min(self.p_chunk, remaining_tokens)
                context_tokens = (
                    runtime.history_tokens_before
                    + processed_tokens + chunk_tokens)
                running_prefill_load_ns += self._prefill_chunk_task_load_ns(
                    instance_size=instance_size,
                    chunk_tokens=chunk_tokens,
                    context_tokens=context_tokens)
                processed_tokens += chunk_tokens
                remaining_tokens -= chunk_tokens

        active_decode_load_ns = 0
        for request_index in state.active_decode:
            runtime = self.runtimes[request_index]
            generated_tokens = (
                runtime.current_decode_token - runtime.prefill_context_tokens)
            active_decode_load_ns += self._decode_task_load_ns_cached(
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
        """离线 queued_prefill_task_load_ns（:3665-3691）的在线复刻；
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
                total_load_ns += self._prefill_chunk_task_load_ns(
                    instance_size=instance_size,
                    chunk_tokens=chunk_tokens,
                    context_tokens=context_tokens)
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
        对象：history_prefix_transfer / history_transfer / evictions 已是
        KVTransfer，
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
            "history_prefix_transfer": runtime.history_prefix_transfer,
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

    # --------------------------------------------------------------- 收尾 --

    def verify_run_end(self) -> None:
        """基类协议校验之上，对齐 face/sh_1.0/sh_3.0/wscllm 四仓收尾断言集
        （蓝本 sh_3.0 sh30_online_scheduler.py:980-1006；防御深度——正常
        路径零行为变化）。sh_2.0 无 ready frontier 结构（_admit_pass 直接
        遍历全部实例，无 frontier 就绪集），故省略蓝本的 ready frontier
        断言，不为凑断言新增状态结构。"""
        super().verify_run_end()
        if self.pending_admissions:
            pending_ids = [
                self.runtimes[request_index].request.request_id
                for request_index in self.pending_admissions]
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
        if self._admit_attempt_epoch:
            raise RuntimeError(
                "strategy run ended with stale admit attempt epochs: "
                "{}".format(sorted(self._admit_attempt_epoch)[:5]))
