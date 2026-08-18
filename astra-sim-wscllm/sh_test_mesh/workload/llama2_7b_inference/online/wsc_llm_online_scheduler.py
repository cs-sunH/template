#!/usr/bin/env python3
"""wsc_llm_online_scheduler.py -- 关感知策略调度器(strategy 模式,步骤 1-9)。

以 _plan_wsc_llm_session_lru_recompute(wsc_llm_scheduler.py:1667-2207)为蓝本
迁移,保持决策顺序逐行对应(每处迁移用 `# offline: wsc_llm_scheduler.py:XXXX`
注释标注)。离线事件循环与在线边界的一一对应:

  离线事件循环                                    在线边界
  ----------------                                ----------------
  预置 arrival 堆(:1711-1716)                     ingress ARRIVAL 事件喂入同一
                                                  arrival heap(字段保持
                                                  (time_ns, priority, sequence,
                                                  kind, payload) 形状,同 tick
                                                  排序规则逐字节同源)
  completion 批(:1999-2092 中 completion 分支)     PREFILL_DRAIN / DECODE_COMPLETION
                                                  / REQUEST_COMPLETE 事件处理:
    prefill 完成(:2014-2038)                        _on_prefill_drain:
      qp.popleft / waiting_decode_admissions          busy=False + qp.pop +
      append / dirty                                  等待 decode 准入登记
    decode 完成(:2040-2072)                         _on_decode_complete:
      mark_complete + note_capacity_change            mark_complete + snapshots
      + 下一次 arrival 排程(:2067-2072)               (下一次 arrival 排程在
                                                       REQUEST_COMPLETE 边界做,
                                                       与离线同 tick 同顺序)
  arrival 批(:2074-2090)                            _on_arrival(经 arrival heap):
    select_prefill_instance + route_for_prefill      快照 -> 选择 -> 静态路由
    + qp.append                                      -> qp.append(顺序一致)
  start_ready_iterations(:1915-1996)                _admit_pass(同 tick 末尾):
    try_admit_waiting_decodes(:1917)                  decode 准入(容量 epoch/dirty
                                                      门控,原样保留)
    逐实例 serve(:1918-1996)                          per-instance 发射:
      prefill: try_admit_prefill(qp[0]) + 发射         try_admit_prefill + 发射
      decode:  发射 active_decode 整批                 prefill 整段 / decode 整段
    **计时部分删除**:离线用 LUT 估计时长 push
    iteration_complete 事件;在线由真实完成事件
    (C++ 物理时钟)推进,排队/配对逻辑原样保留于
    账本/准入逻辑,不体现在图结构上(构图粒度与
    离线一致:prefill 整段 + decode 整段)。

决策顺序(line-by-line):每 tick 先处理 completion 批,再处理 arrival 批,最后
跑准入/发射 pass——与离线同 tick 批序(priority 0 completion < 1 arrival)
一致。

关感知口径(方案 §4.1 第 6 条):策略输入全部来自 Python 账本——prefill 排队
深度(len(qp))、KV 容量(kv_manager 账本 + capacity_epoch)、静态路由
(build_static_pd_mapping);不新增任何 C++ 状态读取。C++ 物理(真实完成
tick、网络竞争)在阶段 1 即真实生效,只负责推进完成边界。

与离线蓝图的刻意差异(real-online 语义,合同⑦ Tier B real-online 验收):
  - 计时/迭代粒度:离线 LUT 时钟 + 逐 chunk 迭代 -> 在线真实完成事件 +
    request-aggregated 构图(prefill 整段 + decode 整段,与离线 ET 粒度一致,
    方案 §4 步骤 1-9 操作 1);
  - 实例 busy 语义:离线 prefill 实例 busy 覆盖逐 chunk 迭代,decode 实例
    busy 覆盖整批迭代;在线 busy 覆盖"一个 prefill/decode 整段在飞";
  - decode 段发射:离线整批 active_decode 一次迭代;在线一次发射
    active_decode 队首整段,完成后再发射下一个(per-rank 物理链天然
    串行化同实例 decode 段,与离线 .et 的跨 request 物理链同构——strategy
    模式保持物理跨 request 链,不适用 replay 的 LUT 时钟裁决);
  - 完成顺序:真实完成 tick 决定(网络竞争、物理链),不要求与离线决策
    序列 exact(合同⑦:real-online 只验不变量与差异可解释性)。
"""

import heapq
import os
import sys
from collections import deque

# --------------------------------------------------------------------------
# import 路径:本文件位于 workload/llama2_7b_inference/online/,离线写出模块在
# 上一级。路径只做 import 用途(红线:generate_wsc_llm_trace.py / wsc_llm_
# scheduler.py / session_kv_manager.py 只读 import 与注释)。
# --------------------------------------------------------------------------
_ONLINE_DIR = os.path.dirname(os.path.abspath(__file__))
_WORKLOAD_DIR = os.path.dirname(_ONLINE_DIR)
for _path in (_ONLINE_DIR, _WORKLOAD_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from generate_wsc_llm_trace import (  # noqa: E402
    _eviction_record_dict,
    _kv_transfer_dict,
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
from wsc_llm_scheduler import (  # noqa: E402
    DECODE_ROLE,
    PREFILL_ROLE,
    PrefillQueueSnapshot,
    WscLlmInstanceSpec,
    build_instances,
    build_static_pd_mapping,
    select_prefill_instance,
)


class _OnlineInstanceState:
    """在线实例账本(离线 _InstanceRuntime,wsc_llm_scheduler.py:1699-1702 的
    在线子集):qp = prefill FCFS 队列(deque),active_decode = 已准入 decode
    列表,busy = 一个整段在飞。"""

    __slots__ = ("index", "phase_role", "qp", "active_decode",
                 "active_decode_lookup", "busy")

    def __init__(self, *, index: int, phase_role: str) -> None:
        self.index = index
        self.phase_role = phase_role
        # 阶段 4 §7.3:deque 替换 list.pop(0)(O(N) 首出 -> O(1)popleft)。
        self.qp = deque()  # 顺序即 FCFS(append 尾入,popleft 首出)
        self.active_decode = []
        # §7.3:完成事件直接定位(active_decode 成员判定/移除 O(N) -> O(1);
        # 与列表保持同步,append/remove 双侧更新)。
        self.active_decode_lookup = set()
        self.busy = False


class _OnlineRequestRuntime:
    """在线请求运行账本(离线 _RequestRuntime 的在线子集)。

    输入事实(request-neutral,来自 manifest,policy-independent):
      history_tokens_before / prefill_context_tokens / final_context_tokens /
      prefill_length / decode_length / queue_index / session_id / turn_index。
    运行期事实(在线决策产出的字段,语义与离线蓝图同名同义)。
    """

    __slots__ = (
        "request_id", "session_id", "turn_index", "queue_index",
        "prefill_length", "decode_length",
        "history_tokens_before", "prefill_context_tokens", "final_context_tokens",
        "estimated_arrival_ns",
        "prefill_instance_index", "prefill_assignment_key",
        "decode_instance_index", "static_route",
        "admitted_prefill", "prefill_attempt_epoch", "decode_capacity_reserved",
        "history_cache_state_before", "hbm_before_request",
        "history_action", "history_source_instance_index",
        "history_transfer_bytes", "history_recompute_tokens",
        "admission_evictions", "decode_target_evictions",
        "decode_queue_depth_before_enqueue",
        "waiting_decode_admission", "prefill_decode_transfer",
        "completion_evictions", "kv_state_after_completion",
        "kv_instance_after_completion", "hbm_after_completion",
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
        self.prefill_context_tokens = record["prefill_context_tokens"]
        self.final_context_tokens = record["final_context_tokens"]
        self.estimated_arrival_ns = None
        self.prefill_instance_index = None
        self.prefill_assignment_key = None
        self.decode_instance_index = None
        self.static_route = None
        self.admitted_prefill = False
        self.prefill_attempt_epoch = None
        self.decode_capacity_reserved = False
        self.history_cache_state_before = None
        self.hbm_before_request = None
        self.history_action = None
        self.history_source_instance_index = None
        self.history_transfer_bytes = None
        self.history_recompute_tokens = None
        self.admission_evictions = ()
        self.decode_target_evictions = ()
        self.decode_queue_depth_before_enqueue = None
        self.waiting_decode_admission = False
        self.prefill_decode_transfer = None
        self.completion_evictions = ()
        self.kv_state_after_completion = None
        self.kv_instance_after_completion = None
        self.hbm_after_completion = None
        self.completion_ns = None


def _kv_event_dict(event) -> dict:
    """Serialize one KVCacheEvent for the online kv_actions stream
    (与 _eviction_record_dict / _kv_transfer_dict 同构的在线序列化)。"""
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


class WscLlmOnlineScheduler(OnlineSchedulerBase):
    """strategy 变体:真实策略(关感知)在在线骨架中运行。

    蓝本: _plan_wsc_llm_session_lru_recompute(wsc_llm_scheduler.py:1667-2207),
    kv_cache_policy == "session_lru_recompute"(主变体)。拓扑 / 静态路由 /
    KV 账本(与离线同一函数、同参数)在 __init__ 一次性构建,运行期策略输入
    全部来自这些 Python 账本(关感知)。
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
            raise ValueError("WscLlmOnlineScheduler requires mode == 'strategy'")
        if config.kv_cache_policy != "session_lru_recompute":
            raise ValueError(
                "strategy scheduler supports kv_cache_policy "
                "'session_lru_recompute' only, got {!r}".format(
                    config.kv_cache_policy))
        self.graph = graph  # GraphBatchBuilder(与 replay 路径共用)

        # 蓝图 :1682-1683:拓扑 + 静态 PD 路由(alpha 默认 1.0,与离线同参)。
        # offline: wsc_llm_scheduler.py:1682-1683
        specs = tuple(
            WscLlmInstanceSpec(
                name=group.name,
                pg_name=group.pg_name,
                ranks=group.ranks,
                phase_role=group.phase_role,
            )
            for group in config.inference_groups
        )
        self.topology = build_instances(
            config.hardware, specs, require_equal_size=True)
        self.static_mapping = build_static_pd_mapping(
            self.topology, alpha=1.0)

        # 蓝图 :1694-1698:KV 账本(reserve_context_tokens 同参)。
        # offline: wsc_llm_scheduler.py:1694-1698
        self.kv_manager = SessionKVCacheManager(
            self.topology,
            config.model,
            reserve_context_tokens=config.kv_reserve_context_tokens,
        )

        # 蓝图 :1699-1702:实例账本。
        # offline: wsc_llm_scheduler.py:1699-1702
        self.instances = [
            _OnlineInstanceState(index=instance.index,
                                  phase_role=instance.phase_role)
            for instance in self.topology.instances
        ]

        # 蓝图 :1703-1704:future arrival min-heap(在线由 ingress ARRIVAL
        # 事件喂入)。阶段 4 §7.3:键含 queue_index,同 tick 到期项按冻结
        # 队列序稳定弹出(与 C++ 序列化的 arrivals 冻结队列序一致);
        # 消费规则 tick <= current_tick(见 _drain_arrival_heap)。
        # offline: wsc_llm_scheduler.py:1703-1704
        self.arrival_heap = []
        self._sequence = 0

        # 蓝图 :1723-1725:等待 decode 准入登记(按实例)。
        # §7.3:deque 替换 list.pop(0)。
        # offline: wsc_llm_scheduler.py:1723-1725
        self.waiting_decode_admissions = {
            instance.index: deque() for instance in self.topology.instances
        }

        # 阶段 4 §7.3:按 rank 的 ready frontier——非忙且有排队工作的实例
        # 集合(发射时清除,完成/到达时设置;结束审计必须为空)。_admit_pass
        # 只访问该集合(sorted 保持实例 index 序 = 离线循环序,决策确定性
        # 不受影响),不做全量实例扫描。
        self._ready_frontier = set()
        # §7.3:admission retry ready set(legacy 迁移占位;v1 无生产者,
        # 恒为空;结束审计必须为空)。
        self._admission_retry_ready = set()
        # 蓝图 :1729-1732:容量 epoch / 准入门控。
        # offline: wsc_llm_scheduler.py:1729-1732
        self.capacity_epoch = [0 for _ in self.topology.instances]
        self.prefill_attempt_epoch = {}
        self.decode_admission_epoch = [-1 for _ in self.topology.instances]
        self.decode_admission_dirty = set()

        # 请求运行账本(queue_index 序;manifest 事实 policy-independent)。
        # offline: wsc_llm_scheduler.py:1684(runtimes 由
        # _validate_and_expand_requests 构造;在线输入事实直接来自 manifest)。
        self.runtimes = [
            _OnlineRequestRuntime(record)
            for record in sorted(
                manifest["requests"], key=lambda item: item["queue_index"])
        ]
        self.runtime_by_request_id = {
            runtime.request_id: runtime for runtime in self.runtimes
        }
        # 阶段 4 §7.3:request_id -> runtimes 下标(O(1) 定位,替换
        # runtime_by_request_id_index 的 O(N) 全量扫描)。
        self._runtime_index = {
            runtime.request_id: index
            for index, runtime in enumerate(self.runtimes)
        }
        # 蓝图 :1684 的 next_request(同 session 下一 turn 的 runtime 对象;
        # 没有则 None)。§7.3:(session_id, turn_index) 字典索引替换 O(N²)
        # 顺序扫描;turn+1 精确语义与扫描版逐项一致(值 = runtime 对象,
        # 与原扫描版返回对象一致)。
        by_turn = {}
        for runtime in self.runtimes:
            by_turn[(runtime.session_id, runtime.turn_index)] = runtime
        self.next_request = [None] * len(self.runtimes)
        for index, runtime in enumerate(self.runtimes):
            self.next_request[index] = by_turn.get(
                (runtime.session_id, runtime.turn_index + 1))

        self.completed_requests = 0
        # kv 事件水位:kv_actions 流(phase 2 对照 kv_cache_events.csv 基线)。
        self._kv_events_emitted = 0

    # ------------------------------------------------------------- 策略 --

    def run_variant_policy(self, delta) -> None:
        """决策顺序逐行对应离线事件循环:completion 批(:2006-2072)先于
        arrival 批(:2074-2090),最后 start_ready_iterations(:2092)。
        """
        tick = delta["tick"]

        # ---- completion 批(离线 priority 0;同 tick 先于 arrival)----
        # offline: wsc_llm_scheduler.py:2006-2072
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
        # offline: wsc_llm_scheduler.py:2074-2090
        for arrival in delta["arrivals"]:
            self._push_arrival(arrival, tick)
        self._drain_arrival_heap(tick)

        # ---- 准入/发射 pass(离线 start_ready_iterations,计时部分删除)----
        # offline: wsc_llm_scheduler.py:1915-1996
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
        """离线 arrival 批单条(:2074-2090):选择 prefill 实例 + 静态路由 +
        qp 入队。快照在 append 之前取(ordering_key 反映选择时刻的排队深度)。

        offline: wsc_llm_scheduler.py:2074-2090
        """
        runtime.estimated_arrival_ns = tick  # :2079
        snapshots = self._prefill_snapshots()  # :2080
        selected = select_prefill_instance(snapshots)  # :2081
        selected_snapshot = next(
            snapshot for snapshot in snapshots
            if snapshot.instance_index == selected)  # :2082-2084
        route = self.static_mapping.route_for_prefill(selected)  # :2085
        runtime.prefill_instance_index = selected  # :2086
        runtime.prefill_assignment_key = selected_snapshot.ordering_key  # :2087
        runtime.static_route = route  # :2088
        runtime.decode_instance_index = route.decode_instance_index  # :2089
        self.instances[selected].qp.append(runtime)  # :2090
        self._note_instance_ready(selected)  # §7.3 ready frontier
        # 阶段 3 感知账本:进入 admitted 层(prefill_qp 排队账本成员;
        # contract ⑥)。查询/审计数据,不进策略判据。
        self._ledger_admit(runtime.request_id, tick,
                           {"type": "prefill_qp", "instance_index": selected})

    def _on_prefill_drain(self, request_id: str, tick: int) -> None:
        """离线 prefill 完成分支(:2014-2038):实例空闲 + qp 出队 + 等待
        decode 准入登记 + dirty。

        offline: wsc_llm_scheduler.py:2014-2038
        """
        runtime = self.runtime_by_request_id[request_id]
        state = self.instances[runtime.prefill_instance_index]
        if not state.qp or state.qp[0] is not runtime:
            raise RuntimeError("Prefill FCFS queue order was corrupted")
        state.busy = False  # :2011
        state.qp.popleft()  # :2030(§7.3:deque O(1) 首出)
        if state.qp:
            self._ready_frontier.add(state.index)  # §7.3:队列仍有等待
        else:
            self._ready_frontier.discard(state.index)
        runtime.waiting_decode_admission = True  # :2032
        if runtime.decode_instance_index is None:  # :2033-2034
            raise RuntimeError("WSC Prefill completion lost Decode target")
        self.waiting_decode_admissions[runtime.decode_instance_index].append(
            runtime)  # :2035-2037
        self.decode_admission_dirty.add(runtime.decode_instance_index)  # :2038
        # 阶段 3 感知账本:admitted 层排队类型更新(waiting_decode)。
        self._ledger_admit(
            runtime.request_id, tick,
            {"type": "waiting_decode",
             "instance_index": runtime.decode_instance_index})

    def _on_decode_complete(self, request_id: str, tick: int) -> None:
        """离线 decode 完成分支(:2040-2066):active_decode 出队 +
        mark_complete + 快照。下一次 arrival 排程在 REQUEST_COMPLETE 边界
        (同 tick,见 _on_request_complete)。

        offline: wsc_llm_scheduler.py:2040-2066
        """
        runtime = self.runtime_by_request_id[request_id]
        state = self.instances[runtime.decode_instance_index]
        if runtime not in state.active_decode_lookup:  # :2047-2048(§7.3 O(1))
            raise RuntimeError("Decode queue membership was corrupted")
        state.active_decode_lookup.discard(runtime)  # §7.3 双侧同步
        state.active_decode.remove(runtime)
        state.busy = False  # :2011(离线 iteration_complete 的 busy 复位)
        if state.active_decode:
            self._ready_frontier.add(state.index)  # §7.3:队内仍有等待
        else:
            self._ready_frontier.discard(state.index)
        runtime.completion_ns = tick  # :2050
        self.completed_requests += 1  # :2051
        runtime.completion_evictions = self.kv_manager.mark_complete(  # :2052-2056
            runtime.session_id,
            tick,
            runtime.request_id,
        )
        self._note_capacity_change(  # :2057-2060
            state.index,
            *(record.victim_instance_index
              for record in runtime.completion_evictions),
        )
        snapshot = self.kv_manager.session_snapshot(runtime.session_id)  # :2061
        if snapshot is None:  # :2062-2063
            raise RuntimeError("completed session disappeared from KV manager")
        runtime.kv_state_after_completion = snapshot.state  # :2064
        runtime.kv_instance_after_completion = snapshot.instance_index  # :2065
        runtime.hbm_after_completion = self.kv_manager.hbm_snapshots(  # :2066
            state.index)
        self.log_decision(
            {"kind": "completion", "request_id": runtime.request_id,
             "priority": 0},
            tick,
            decision={
                "terminal_kv_release_at_completion": False,
                "kv_state_after_completion": runtime.kv_state_after_completion,
                "kv_instance_after_completion":
                    runtime.kv_instance_after_completion,
                "completion_evictions": [
                    _eviction_record_dict(record)
                    for record in runtime.completion_evictions
                ],
            },
        )

    def _on_request_complete(self, request_id: str, tick: int) -> None:
        """离线 decode 完成分支的下一次 arrival 排程(:2067-2072):
        now + 该 turn 的 inter_request_interval_ns 注册未来 alarm
        (向 ingress,不再 push 到事件堆)。

        offline: wsc_llm_scheduler.py:2067-2072
        """
        runtime = self.runtime_by_request_id[request_id]
        # §7.3:_runtime_index O(1) 定位(替换 O(N) 全量扫描)。
        following = self.next_request[self._runtime_index[runtime.request_id]]
        if following is None:
            return  # session 最后一 turn:无下一次 arrival
        interval = self._interval_ns(following)  # :2069-2071
        # :2072 push_event(now_ns + interval, 1, "arrival", following) ->
        # 在线等价:future alarm(阶段 1 的 alarm 语义,见 C++ Phase 4)。
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
        """离线 :1706-1709 push_event(time_ns, priority=1, kind="arrival")。
        字段形状与离线一致;阶段 4 §7.3 键含 queue_index —— 同 tick 到期项
        按冻结队列序稳定弹出,与 C++ 序列化的 arrivals 冻结队列序一致。
        """
        runtime = self.runtime_by_request_id[arrival["request_id"]]
        heapq.heappush(
            self.arrival_heap,
            (tick, 1, runtime.queue_index, self._sequence, "arrival",
             runtime))
        self._sequence += 1

    def _drain_arrival_heap(self, tick: int) -> None:
        """离线 :2000-2004 同 tick 批 + :2004 按 (priority, sequence) 排序。
        阶段 4 §7.3:只消费 tick <= current_tick 的到期项(在线单 tick 单次
        交付下堆内全部为当前批 push 的项,规则与离线循环同构;未来 alarm
        的到期事件由 C++ 在到期 tick 交付,不会滞留堆中)。
        """
        while self.arrival_heap and self.arrival_heap[0][0] <= tick:
            _, _, _, _, kind, payload = heapq.heappop(self.arrival_heap)
            if kind != "arrival":
                raise RuntimeError("arrival heap contains {!r}".format(kind))
            self._profile_scan()  # §7.3:堆弹出条目(到期事件)
            self._on_arrival(payload, tick)

    # ------------------------------------------------------------- 准入 --

    def _admit_pass(self, tick: int) -> None:
        """离线 start_ready_iterations(:1915-1996)的排队/准入部分;计时部分
        (LUT 估计 + push iteration_complete)删除,由真实完成事件推进。
        先 try_admit_waiting_decodes(:1917),再逐实例 serve(:1918-1996)。

        阶段 4 §7.3:逐实例 serve 改为只访问 ready frontier(非忙且有排队
        工作的实例;sorted 保持实例 index 序 = 离线 :1918 的循环序,决策
        逐字节不变)。frontier 在到达/完成(设)与发射(清)时增量维护,
        单批访问条目数 = 到期/受影响条目数,与总 request 数无关。

        offline: wsc_llm_scheduler.py:1915-1996
        """
        self._try_admit_waiting_decodes(tick)  # :1917
        for instance_index in sorted(self._ready_frontier):  # §7.3 frontier
            self._profile_scan()  # §7.3:frontier 访问条目(就绪实例)
            state = self.instances[instance_index]
            if state.busy:  # 防御:frontier 与 busy 失步即内部错误
                continue
            if state.phase_role == PREFILL_ROLE:  # :1922-1952
                if state.active_decode:  # :1923-1924
                    raise RuntimeError(
                        "Prefill-only instance contains Decode work")
                if not state.qp:  # :1925-1926
                    continue
                runtime = state.qp[0]  # :1927
                if not self._try_admit_prefill(runtime, tick):  # :1928-1929
                    continue
                self._emit_prefill(runtime, tick)  # 聚合:prefill 整段
                state.busy = True  # :1995
                self._ready_frontier.discard(state.index)  # §7.3:发射即忙
            else:  # DECODE_ROLE, :1953-1973
                if state.qp:  # :1954-1955
                    raise RuntimeError(
                        "Decode-only instance contains Prefill work")
                if not state.active_decode:  # :1956-1957
                    continue
                runtime = state.active_decode[0]  # 聚合:一次发射队首整段
                self._emit_decode(runtime, tick)
                state.busy = True  # :1995
                self._ready_frontier.discard(state.index)  # §7.3:发射即忙

    def _try_admit_prefill(self, runtime, now_ns: int) -> bool:
        """离线 try_admit_prefill(:1746-1849),逐行对应;无逐 chunk 计时/
        剩余块账本(聚合粒度)。

        offline: wsc_llm_scheduler.py:1746-1849
        """
        if runtime.admitted_prefill:  # :1748-1749
            return True
        if (runtime.prefill_instance_index is None
                or runtime.static_route is None):  # :1750-1751
            raise RuntimeError("WSC Prefill admission lost its static mapping")
        prefill_instance = runtime.prefill_instance_index  # :1752
        decode_instance = runtime.static_route.decode_instance_index  # :1753
        admission_epoch = (  # :1754-1757
            self.capacity_epoch[prefill_instance],
            self.capacity_epoch[decode_instance],
        )
        if (self.prefill_attempt_epoch.get(runtime.request_id)
                == admission_epoch):  # :1758-1759
            return False
        self.prefill_attempt_epoch[runtime.request_id] = admission_epoch  # :1760
        final_shards = kv_cache_shard_bytes_for_tokens(  # :1761-1763
            self.config.model, runtime.final_context_tokens,
            self.topology.instances[0].size)
        reservation = self.kv_manager.reserve_request_capacity(  # :1764-1772
            runtime.request_id,
            runtime.session_id,
            decode_instance,
            final_shards,
            now_ns,
            phase="prefill_admission",
            reason="static_decode_final_kv_reservation",
        )
        runtime.decode_target_evictions = tuple(  # :1773-1775
            (*runtime.decode_target_evictions, *reservation.evictions))
        if not reservation.admitted:  # :1776-1782
            if reservation.evictions:
                self._note_capacity_change(
                    decode_instance,
                    *(record.victim_instance_index
                      for record in reservation.evictions),
                )
            return False
        runtime.decode_capacity_reserved = True  # :1783
        before_snapshot = self.kv_manager.session_snapshot(  # :1784
            runtime.session_id)
        runtime.history_cache_state_before = (  # :1785-1787
            "ABSENT" if before_snapshot is None else before_snapshot.state)
        runtime.hbm_before_request = self.kv_manager.hbm_snapshots(  # :1788
            runtime.prefill_instance_index)
        decision = self.kv_manager.prepare_history(  # :1789-1796
            runtime.session_id,
            runtime.prefill_instance_index,
            runtime.history_tokens_before,
            now_ns,
            runtime.request_id,
            required_context_tokens=runtime.prefill_context_tokens,
        )
        runtime.admission_evictions = decision.evictions  # :1797
        if decision.admission_blocked:  # :1798-1808
            self.kv_manager.release_request_capacity(runtime.request_id)
            runtime.decode_capacity_reserved = False
            self._note_capacity_change(
                prefill_instance,
                decode_instance,
                decision.source_instance_index,
                *(record.victim_instance_index
                  for record in reservation.evictions),
                *(record.victim_instance_index
                  for record in decision.evictions),
            )
            return False
        runtime.history_action = decision.action  # :1809
        runtime.history_source_instance_index = decision.source_instance_index  # :1810
        runtime.history_transfer_bytes = decision.transfer_bytes  # :1811
        runtime.history_recompute_tokens = decision.recompute_tokens  # :1812
        growth = self.kv_manager.grow_prefill(  # :1826-1831
            runtime.session_id,
            runtime.prefill_context_tokens,
            now_ns,
            runtime.request_id,
        )
        if not growth.admitted:  # :1832-1833
            raise RuntimeError("Prefill growth lost its successful capacity preflight")
        runtime.admission_evictions = tuple(  # :1834
            (*runtime.admission_evictions, *growth.evictions))
        runtime.admitted_prefill = True  # :1840
        self._note_capacity_change(  # :1841-1848
            prefill_instance,
            decode_instance,
            decision.source_instance_index,
            *(record.victim_instance_index for record in reservation.evictions),
            *(record.victim_instance_index for record in decision.evictions),
            *(record.victim_instance_index for record in growth.evictions),
        )
        return True

    def _try_admit_waiting_decodes(self, now_ns: int) -> None:
        """离线 try_admit_waiting_decodes(:1851-1913),逐行对应。

        offline: wsc_llm_scheduler.py:1851-1913
        """
        ready_targets = tuple(
            instance_index
            for instance_index, queue in sorted(
                self.waiting_decode_admissions.items())
            if queue
            and (
                instance_index in self.decode_admission_dirty
                or self.decode_admission_epoch[instance_index]
                != self.capacity_epoch[instance_index]
            )
        )  # :1852-1861
        for target_instance in ready_targets:  # :1862
            self.decode_admission_epoch[target_instance] = (  # :1863
                self.capacity_epoch[target_instance])
            self.decode_admission_dirty.discard(target_instance)  # :1864
            queue = self.waiting_decode_admissions[target_instance]  # :1865
            pending_count = len(queue)  # :1866
            for _ in range(pending_count):  # :1867
                runtime = queue.popleft()  # :1868(§7.3:deque O(1) 首出)
                if not runtime.waiting_decode_admission:  # :1870-1871
                    continue
                if runtime.decode_instance_index != target_instance:  # :1872-1873
                    raise RuntimeError("WSC Decode target queue was corrupted")
                if not runtime.decode_capacity_reserved:  # :1874-1875
                    raise RuntimeError(
                        "WSC Decode target was not reserved before Prefill")
                self.kv_manager.release_request_capacity(  # :1876
                    runtime.request_id)
                runtime.decode_capacity_reserved = False  # :1877
                move = self.kv_manager.move_prefill_to_decode(  # :1878-1884
                    runtime.session_id,
                    target_instance,
                    now_ns,
                    runtime.request_id,
                    final_context_tokens=runtime.final_context_tokens,
                )
                runtime.decode_target_evictions = tuple(  # :1885-1887
                    (*runtime.decode_target_evictions, *move.evictions))
                if move.admission_blocked:  # :1888-1889
                    raise RuntimeError(
                        "released static Decode reservation was not "
                        "physically usable")
                runtime.prefill_decode_transfer = move.transfer  # :1890
                growth = self.kv_manager.grow_decode(  # :1891-1896
                    runtime.session_id,
                    runtime.final_context_tokens,
                    now_ns,
                    runtime.request_id,
                )
                if not growth.admitted:  # :1897-1898
                    raise RuntimeError(
                        "Decode growth lost its successful target admission")
                runtime.decode_target_evictions = tuple(  # :1899-1901
                    (*runtime.decode_target_evictions, *growth.evictions))
                decode_state = self.instances[target_instance]  # :1902
                if decode_state.phase_role != DECODE_ROLE:  # :1903-1904
                    raise RuntimeError(
                        "static route selected a non-Decode instance")
                runtime.decode_queue_depth_before_enqueue = (  # :1905
                    len(decode_state.active_decode))
                decode_state.active_decode.append(runtime)  # :1906
                decode_state.active_decode_lookup.add(runtime)  # §7.3 双侧同步
                if not decode_state.busy:
                    self._ready_frontier.add(target_instance)  # §7.3
                runtime.waiting_decode_admission = False  # :1907
                # 阶段 3 感知账本:admitted 层排队类型更新(active_decode)。
                self._ledger_admit(
                    runtime.request_id, now_ns,
                    {"type": "active_decode", "instance_index": target_instance})
                self._note_capacity_change(  # :1908-1913
                    target_instance,
                    move.source_instance_index,
                    *(record.victim_instance_index for record in move.evictions),
                    *(record.victim_instance_index for record in growth.evictions),
                )

    # ------------------------------------------------------------- 发射 --

    def _emit_prefill(self, runtime, tick: int) -> None:
        """聚合粒度发射 prefill 整段(与离线 ET 按 request 整段一致;
        离线逐 chunk 迭代在在线不存在)。watch 注册 PREFILL_DRAIN。

        offline: wsc_llm_scheduler.py:1927-1952(发射对象为整段而非 chunk)
        """
        plan = self._plan_dict(runtime)
        members = self.graph.emit_prefill_batch(plan, phase_duration_ns=0)
        # 阶段 3 感知账本:本批次发射记录(prefill 阶段;commit ack 到达后
        # 转移入 committed 层)。
        self._note_emitted(runtime.request_id, STAGE_PREFILL)
        # 阶段 7 §10.1:issued 层登记(已发射未完成;completion 核销时由
        # 基类 _settle_completions 移除)。
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
            "decode_instance_index": runtime.decode_instance_index,
        })
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
                    _eviction_record_dict(record)
                    for record in runtime.admission_evictions
                ],
                "decode_target_evictions": [
                    _eviction_record_dict(record)
                    for record in runtime.decode_target_evictions
                ],
            },
        )

    def _emit_decode(self, runtime, tick: int) -> None:
        """聚合粒度发射 decode 整段(transfer 3000 + decode 整段 + end
        barrier)。watch 注册 DECODE_COMPLETION(C++ 同 fire 推
        DECODE_COMPLETION + REQUEST_COMPLETE 两条 completed_groups)。

        offline: wsc_llm_scheduler.py:1953-1973(发射对象为整段而非 chunk)
        """
        plan = self._plan_dict(runtime)
        members = self.graph.emit_decode_batch(plan, phase_duration_ns=0)
        # 阶段 3 感知账本:本批次发射记录(decode 阶段)。
        self._note_emitted(runtime.request_id, STAGE_DECODE)
        # 阶段 7 §10.1:issued 层登记(已发射未完成;decode 段条目,prefill
        # 条目已在其 drain 时移除,单条目语义不变)。
        self._ledger_issue(runtime.request_id, tick, STAGE_DECODE,
                           runtime.decode_instance_index)
        self._batch["watches"].append({
            "request_id": runtime.request_id,
            "stage": STAGE_DECODE,
            "generation": 1,
            "members": members,
            "statuses": ["Success", "Skipped"],
        })
        route = runtime.static_route
        self.log_decision(
            {"kind": "decode", "request_id": runtime.request_id,
             "priority": 0},
            tick,
            decision={
                "decode_instance_index": runtime.decode_instance_index,
                "static_route": {
                    "prefill_instance_index": route.prefill_instance_index,
                    "decode_instance_index": route.decode_instance_index,
                    "path": list(route.path),
                    "hop_count": route.hop_count,
                    "shared_edges": [list(edge) for edge in route.shared_edges],
                },
                "decode_queue_depth_before_enqueue":
                    runtime.decode_queue_depth_before_enqueue,
                "prefill_decode_transfer": _kv_transfer_dict(
                    runtime.prefill_decode_transfer),
            },
        )

    # ------------------------------------------------------------- 助手 --

    def _plan_dict(self, runtime) -> dict:
        """graph_batch_builder 消费的 plan 字段(request 事实 + 在线决策)。
        与 replay 路径同构;history_action 等 KV 决策字段在准入时已落账本。
        """
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

    def _prefill_snapshots(self):
        """离线 prefill_snapshots(:1739-1744):仅 prefill 角色实例,
        在 instances 序(实例 index 序)上构建。"""
        return tuple(
            PrefillQueueSnapshot(instance_index=state.index,
                                 request_count=len(state.qp))
            for state in self.instances
            if state.phase_role == PREFILL_ROLE
        )

    def _note_capacity_change(self, *instance_indexes) -> None:
        """离线 note_capacity_change(:1734-1737)。"""
        for instance_index in set(instance_indexes):
            if instance_index is not None:
                self.capacity_epoch[instance_index] += 1

    def _note_instance_ready(self, instance_index: int) -> None:
        """§7.3 ready frontier:实例非忙且有排队工作 -> 就绪集(发射时由
        _admit_pass 清除;完成/到达时设置)。"""
        if not self.instances[instance_index].busy:
            self._ready_frontier.add(instance_index)

    def _sensing_ready_view(self) -> dict:
        """阶段 7 §10.1 ready 层边界视图:ready frontier 实例队列中等待
        服务的 request(prefill qp / active_decode;frontier = 非忙且有排队
        工作的实例,§7.3)。访问条目 = 就绪实例数(有界,非全量扫描)。
        查询/审计输入,不进策略判据。"""
        detail = []
        for instance_index in sorted(self._ready_frontier):
            state = self.instances[instance_index]
            if state.phase_role == PREFILL_ROLE:
                stage, queue = STAGE_PREFILL, state.qp
            else:
                stage, queue = STAGE_DECODE, state.active_decode
            for runtime in queue:
                detail.append({
                    "request_id": runtime.request_id,
                    "stage": stage,
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
        """基类协议校验之上,叠加离线 :2094-2101 的收尾断言与阶段 4 §7.3
        结束审计(事件索引队列全部为空:arrival heap / ready frontier /
        admission retry ready set)。"""
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
        # 阶段 4 §7.3 结束审计:heap/ready set 全空(与 C++ 侧 mailbox/
        # watch registry 审计同批)。
        if self.arrival_heap:
            raise RuntimeError(
                "run ended with {} unconsumed arrival events".format(
                    len(self.arrival_heap)))
        if self._ready_frontier:
            raise RuntimeError(
                "run ended with non-empty ready frontier: {!r}".format(
                    sorted(self._ready_frontier)))
        if self._admission_retry_ready:
            raise RuntimeError(
                "run ended with non-empty admission retry ready set")
        self.kv_manager.assert_final_state()
