#!/usr/bin/env python3
"""wsc_llm_legacy_online_scheduler.py -- legacy WSC Relevant(P,D) 静态域调度器
(strategy 模式,阶段 7 §10.6 第二变体在线迁移)。

以 _plan_wsc_llm_requests_legacy(wsc_llm_scheduler.py:1295-1664)为蓝本迁移,
与 WscLlmOnlineScheduler 之于 session_lru 同构(每处迁移用
`# offline: wsc_llm_scheduler.py:XXXX` 注释标注)。离线 legacy 事件循环与
在线边界的一一对应:

  离线事件循环                                    在线边界
  ----------------                                ----------------
  预置 arrival 堆(:1336-1341)                     ingress ARRIVAL 事件喂入同一
                                                  arrival heap(字段保持
                                                  (time_ns, priority, sequence,
                                                  kind, payload) 形状,同 tick
                                                  排序规则逐字节同源)
  iteration_complete 批(:1449-1547)               PREFILL_DRAIN / DECODE_COMPLETION
    prefill 完成(:1488-1510)                        _on_prefill_drain:
      busy=False + qp.popleft +                       busy=False + qp.popleft +
      active_decode.append                            active_decode.append(直接,
                                                      无二次 decode 准入——
                                                      legacy 的 KV 在准入时已
                                                      分配在 Relevant(P,D) 域)
    decode 完成(:1511-1547)                         _on_decode_complete:
      active_decode.remove + completed                active_decode.remove + completed
      + terminal 释放(:1534-1547)                     + terminal 释放(同分支逐行)
      + 下一 turn arrival 排程(:1524-1531)            (下一 turn arrival 排程在
                                                       REQUEST_COMPLETE 边界做,
                                                       与离线同 tick 同顺序)
  arrival 批(:1552-1578)                            _on_arrival(经 arrival heap):
    turn>0: 释放前序 session allocation              前序 KV 释放 +
      + history_source_instance_index                 history_source/bytes 记录
      + history_transfer_bytes                         + 快照 -> 选择 -> qp.append
    select_prefill_instance + qp.append              (静态路由在准入时决定,
                                                      与离线 :1368-1376 一致)
  start_ready_iterations(:1357-1469)                _admit_pass(同 tick 末尾):
    prefill: head try_allocate + 发射                 head try_allocate(FCFS 队头
                                                      阻塞)+ 发射 prefill 整段
    decode:  发射 active_decode 整批                 decode 队首整段发射
    **计时部分删除**:离线用 LUT 估计时长 push
    iteration_complete 事件;在线由真实完成事件
    (C++ 物理时钟)推进,排队/配对逻辑原样保留。

决策顺序(line-by-line):每 tick 先处理 completion 批,再处理 arrival 批,最后
跑准入/发射 pass——与离线同 tick 批序(priority 0 completion < 1 arrival)
一致。

§0.4 红线(逐项保留,本变体的存在意义):
  - WscRelevantKvAllocator(只读复用 wsc_llm_scheduler.py:852-1077):Relevant
    (P,D) 静态域分配——decode 实例优先、选中 prefill 次之、同域 sibling
    prefill 最后,永不扩大到其他 Decode 域;
  - FCFS 队头阻塞:head try_allocate 失败 -> continue,队内后续请求一概不
    尝试,直到后续调度事件重查容量(WSC-LLM Algorithm 2 line 6,离线
    :1388-1392 同款);
  - 静态 P->D 路由:build_static_pd_mapping(alpha=1.0),准入时 route_for_
    prefill,解码目标固定;
  - 两分支独立性:本变体与 session_lru 变体结果分开统计(kv_cache_policy
    列);legacy 在线无逐事件 KV 日志,只有 run-end allocator 终值
    (metrics_integration.kv_event_payload_legacy 同构口径)。

p_chunk 标定常数(用户裁决 2026-08-15,见方案文档 §3 步骤 0-1 物化规则与
附录 C 登记):
  = ceil(mean(prefill_length)) = ceil(5830711/1177) = 4954,物化阶段从冻结
  输入(20.csv 前30s,1177 请求)按与离线 legacy 同一过程预先导出,离线/在线
  两路径共用同一常数。在线不做任何"从已到达请求算增量 mean"的统计
  (裁决禁止);p_chunk 在线无决策作用(离线仅用于 LUT 计时与 chunk 切分,
  在线都消失),保留为常量声明与标定登记。

与离线蓝图的刻意差异(real-online 语义,合同⑦ Tier B real-online 验收):
  - 计时/迭代粒度:离线 LUT 时钟 + 逐 chunk 迭代 -> 在线真实完成事件 +
    request-aggregated 构图(prefill 整段 + decode 整段,与离线 ET 粒度一致);
  - 构图 history 粒度:离线 writer 对前序 kv_allocation.pieces 逐 piece 发射
    多个 history_piece transfer(category 1000+piece*100,source = 各
    Relevant(P,D) 存储实例)+ 父 decode 段的 kv_offload(category 5000+);
    在线复用共享构图器 graph_batch_builder 的 NOC_MIGRATE 分支 = 单个
    history_kv transfer(category 1000,source = 前序 KV 主位
    history_source_instance_index,total_bytes = history_transfer_bytes)。
    总字节一致(分配总量 = kv_cache_bytes_for_tokens(final_context) =
    子 request 的 history 需要量),边数为粒度差异(差分归因类别,在线决策
    日志的 history_action 仍按离线口径记录 NO_HISTORY——离线 legacy 不设置
    该字段,决策日志逐字段全等);
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

from session_kv_manager import NOC_MIGRATE, NO_HISTORY  # noqa: E402
from online.online_scheduler_base import (  # noqa: E402
    STAGE_DECODE,
    STAGE_PREFILL,
    STAGE_REQUEST,
    OnlineSchedulerBase,
)
from wsc_llm_scheduler import (  # noqa: E402
    DECODE_ROLE,
    PREFILL_ROLE,
    PrefillQueueSnapshot,
    WscLlmInstanceSpec,
    WscRelevantKvAllocator,
    build_instances,
    build_static_pd_mapping,
    estimate_model_weight_bytes,
    kv_cache_bytes_for_tokens,
    select_prefill_instance,
)


# p_chunk 标定常数(用户裁决 2026-08-15;推导登记于方案文档 §3 步骤 0-1
# 物化规则与附录 C)。
# 物化阶段从冻结输入(20.csv 前30s,1177 请求)按与离线 legacy 同一过程
# ceil(sum(prefill_length)/count) 预先导出;离线/在线共用。在线不使用
# (无 LUT 计时、无 chunk 切分),仅作标定声明;严禁在线从已到达请求计算
# 增量 mean。
P_CHUNK = 4954


class _LegacyInstanceState:
    """在线实例账本(离线 _InstanceRuntime,wsc_llm_scheduler.py:1225-1231 的
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


class _LegacyRequestRuntime:
    """在线请求运行账本(离线 _RequestRuntime,wsc_llm_scheduler.py:1199-1223
    的在线子集)。

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
        "kv_allocation", "terminal_kv_release_at_completion",
        "history_source_instance_index", "history_transfer_bytes",
        "decode_queue_depth_before_enqueue",
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
        self.kv_allocation = None
        self.terminal_kv_release_at_completion = False
        self.history_source_instance_index = None
        self.history_transfer_bytes = 0
        self.decode_queue_depth_before_enqueue = None
        self.completion_ns = None


class WscLlmLegacyOnlineScheduler(OnlineSchedulerBase):
    """strategy 变体:legacy WSC Relevant(P,D) 静态域调度器。

    蓝本: _plan_wsc_llm_requests_legacy(wsc_llm_scheduler.py:1295-1664),
    kv_cache_policy == "legacy"。拓扑 / 静态路由 / WscRelevantKvAllocator
    (与离线同一函数、同参数)在 __init__ 一次性构建,运行期策略输入全部
    来自这些 Python 账本(关感知)。
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
            raise ValueError(
                "WscLlmLegacyOnlineScheduler requires mode == 'strategy'")
        if config.kv_cache_policy != "legacy":
            raise ValueError(
                "legacy strategy scheduler requires kv_cache_policy 'legacy', "
                "got {!r}".format(config.kv_cache_policy))
        if sensing:
            # 阶段 3 感知只接入 session_lru 变体(分层账本 + 两层剩余负载
            # 查询语义);legacy 变体范围未定义,显式 fail-closed。
            raise ValueError(
                "--sensing 仅支持 kv_cache_policy session_lru_recompute "
                "(legacy 变体不挂感知)")
        self.graph = graph  # GraphBatchBuilder(与 replay 路径共用)

        # 蓝图 :1302-1303:拓扑 + 静态 PD 路由(alpha 默认 1.0,与离线同参)。
        # offline: wsc_llm_scheduler.py:1302-1303
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

        # 蓝图 :1310-1313:WSC Relevant(P,D) 分配器(model_weight_bytes 同参)。
        # offline: wsc_llm_scheduler.py:1310-1313
        self.allocator = WscRelevantKvAllocator(
            self.topology,
            self.static_mapping,
            model_weight_bytes=estimate_model_weight_bytes(config.model),
        )

        # 蓝图 :1315-1317:实例账本。
        # offline: wsc_llm_scheduler.py:1315-1317
        self.instances = [
            _LegacyInstanceState(index=instance.index,
                                 phase_role=instance.phase_role)
            for instance in self.topology.instances
        ]

        # 蓝图 :1319:session -> KVAllocation 保留账本(turn>0 arrival 释放
        # 前序,terminal 完成释放终态)。
        # offline: wsc_llm_scheduler.py:1319
        self.session_allocations = {}

        # 蓝图 :1321-1334:future arrival min-heap(在线由 ingress ARRIVAL
        # 事件喂入)。阶段 4 §7.3:键含 queue_index,同 tick 到期项按冻结
        # 队列序稳定弹出(与 C++ 序列化的 arrivals 冻结队列序一致);
        # 消费规则 tick <= current_tick(见 _drain_arrival_heap)。
        # offline: wsc_llm_scheduler.py:1321-1334
        self.arrival_heap = []
        self._sequence = 0

        # 阶段 4 §7.3:按 rank 的 ready frontier——非忙且有排队工作的实例
        # 集合(发射时清除,完成/到达时设置;结束审计必须为空)。_admit_pass
        # 只访问该集合(sorted 保持实例 index 序 = 离线 :1359 的循环序,决策
        # 确定性不受影响),不做全量实例扫描。FCFS 队头阻塞时 head 准入失败
        # 的实例保持 frontier(下次决策批重查容量,与离线 :1390-1392 的
        # "A later scheduler event rechecks capacity"同构)。
        self._ready_frontier = set()

        # 请求运行账本(queue_index 序;manifest 事实 policy-independent)。
        # offline: wsc_llm_scheduler.py:1304(runtimes 由
        # _validate_and_expand_requests 构造;在线输入事实直接来自 manifest,
        # 事实字段同源:history_tokens_before / prefill_context_tokens /
        # final_context_tokens 与离线 expand 的推导一致)。
        self.runtimes = [
            _LegacyRequestRuntime(record)
            for record in sorted(
                manifest["requests"], key=lambda item: item["queue_index"])
        ]
        self.runtime_by_request_id = {
            runtime.request_id: runtime for runtime in self.runtimes
        }
        # 阶段 4 §7.3:request_id -> runtimes 下标(O(1) 定位,替换
        # next_request 顺序扫描的 O(N) 全量扫描)。
        self._runtime_index = {
            runtime.request_id: index
            for index, runtime in enumerate(self.runtimes)
        }
        # 蓝图 :1304 的 next_request(同 session 下一 turn 的 runtime 对象;
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

    # ------------------------------------------------------------- 策略 --

    def run_variant_policy(self, delta) -> None:
        """决策顺序逐行对应离线事件循环:completion 批(:1449-1547)先于
        arrival 批(:1552-1578),最后 start_ready_iterations(:1580)。
        """
        tick = delta["tick"]

        # ---- completion 批(离线 priority 0;同 tick 先于 arrival)----
        # offline: wsc_llm_scheduler.py:1449-1547
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
        # offline: wsc_llm_scheduler.py:1552-1578
        for arrival in delta["arrivals"]:
            self._push_arrival(arrival, tick)
        self._drain_arrival_heap(tick)

        # ---- 准入/发射 pass(离线 start_ready_iterations,计时部分删除)----
        # offline: wsc_llm_scheduler.py:1357-1469
        self._admit_pass(tick)

        # ---- kv 动作流:legacy 无逐事件 KV 日志(metrics_integration.py:251
        # kv_event_payload_legacy 注释:there is no KV event log),kv_actions
        # 恒为空数组;run-end 报告 allocator 终值(见 kv_event_payload_legacy)。

    # ------------------------------------------------------------- 边界 --

    def _on_arrival(self, runtime, tick: int) -> None:
        """离线 arrival 批单条(:1552-1578):turn>0 释放前序 session KV 分配
        + 选择 prefill 实例 + qp 入队。静态路由在准入时决定(与离线
        :1368-1376 一致,arrival 不路由)。快照在 append 之前取(ordering_key
        反映选择时刻的排队深度)。

        offline: wsc_llm_scheduler.py:1552-1578
        """
        runtime.estimated_arrival_ns = tick  # :1554
        # :1555 释放/取走该 session 的前序保留分配(turn-0 时不存在,
        # pop 返回 None;turn>0 时必存在,校验后释放)。
        previous_allocation = self.session_allocations.pop(
            runtime.session_id, None)  # :1555
        if runtime.turn_index > 0:  # :1556
            if previous_allocation is None:  # :1557-1559
                raise RuntimeError(
                    "session {} has no prior KV allocation".format(
                        runtime.session_id))
            runtime.history_source_instance_index = (  # :1561-1562
                previous_allocation.decode_instance_index)
            runtime.history_transfer_bytes = kv_cache_bytes_for_tokens(  # :1563-1565
                self.config.model, runtime.history_tokens_before)
            self.allocator.release(previous_allocation)  # :1566
        snapshots = self._prefill_snapshots()  # :1568
        selected = select_prefill_instance(snapshots)  # :1569
        selected_snapshot = next(  # :1570-1572
            snapshot for snapshot in snapshots
            if snapshot.instance_index == selected)
        runtime.prefill_instance_index = selected  # :1573
        runtime.prefill_assignment_key = selected_snapshot.ordering_key  # :1574
        self.instances[selected].qp.append(runtime)  # :1575
        self._note_instance_ready(selected)  # §7.3 ready frontier
        # 阶段 3 感知账本:进入 admitted 层(prefill_qp 排队账本成员;
        # contract ⑥;感知关闭时同样记账,簿记免费)。查询/审计数据,不进
        # 策略判据。
        self._ledger_admit(runtime.request_id, tick,
                           {"type": "prefill_qp", "instance_index": selected})

    def _on_prefill_drain(self, request_id: str, tick: int) -> None:
        """离线 prefill 完成分支(:1488-1510):实例空闲 + qp 出队 + 直接入
        active_decode(legacy 无二次 decode 准入——KV 在准入时已分配在
        Relevant(P,D) 域,decode 实例参与域分配)。

        offline: wsc_llm_scheduler.py:1488-1510
        """
        runtime = self.runtime_by_request_id[request_id]
        state = self.instances[runtime.prefill_instance_index]
        if not state.qp or state.qp[0] is not runtime:
            raise RuntimeError("Prefill FCFS queue order was corrupted")
        state.busy = False  # :1478
        state.qp.popleft()  # :1497
        if state.qp:
            self._ready_frontier.add(state.index)  # §7.3:队列仍有等待
        else:
            self._ready_frontier.discard(state.index)
        route = runtime.static_route  # :1500
        if route is None or runtime.kv_allocation is None:  # :1500-1502
            raise RuntimeError(
                "Prefill completed without a reserved WSC KV allocation")
        decode_state = self.instances[runtime.decode_instance_index]  # :1503
        if decode_state.phase_role != DECODE_ROLE:  # :1504-1505
            raise RuntimeError("static mapping selected a non-Decode instance")
        runtime.decode_queue_depth_before_enqueue = (  # :1506-1507
            len(decode_state.active_decode))
        decode_state.active_decode.append(runtime)  # :1508
        decode_state.active_decode_lookup.add(runtime)  # §7.3 双侧同步
        if not decode_state.busy:
            self._ready_frontier.add(decode_state.index)  # §7.3
        # 阶段 3 感知账本:admitted 层排队类型更新(active_decode)。
        self._ledger_admit(
            runtime.request_id, tick,
            {"type": "active_decode",
             "instance_index": runtime.decode_instance_index})

    def _on_decode_complete(self, request_id: str, tick: int) -> None:
        """离线 decode 完成分支(:1511-1547):active_decode 出队 + completed
        + terminal 释放(同分支逐行)。下一 turn arrival 排程在
        REQUEST_COMPLETE 边界(同 tick,见 _on_request_complete)。

        offline: wsc_llm_scheduler.py:1511-1547
        """
        runtime = self.runtime_by_request_id[request_id]
        state = self.instances[runtime.decode_instance_index]
        if runtime not in state.active_decode_lookup:  # :1527(§7.3 O(1))
            raise RuntimeError("Decode queue membership was corrupted")
        state.active_decode_lookup.discard(runtime)  # §7.3 双侧同步
        state.active_decode.remove(runtime)  # :1528
        state.busy = False  # :1478(离线 iteration_complete 的 busy 复位)
        if state.active_decode:
            self._ready_frontier.add(state.index)  # §7.3:队内仍有等待
        else:
            self._ready_frontier.discard(state.index)
        runtime.completion_ns = tick  # :1529
        self.completed_requests += 1  # :1530
        following = self.next_request[self._runtime_index[runtime.request_id]]
        if following is None:
            # :1534-1547 terminal 释放:session 最后一 turn 的 decode 完成
            # 即释放其 Relevant(P,D) 分配(离线同分支同序)。
            terminal_allocation = self.session_allocations.pop(
                runtime.session_id, None)  # :1535
            if terminal_allocation is None:  # :1536-1538
                raise RuntimeError(
                    "terminal request has no retained KV allocation")
            if terminal_allocation.request_id != runtime.request_id:  # :1539-1541
                raise RuntimeError(
                    "terminal session KV allocation does not match request")
            self.allocator.release(terminal_allocation)  # :1542
            runtime.terminal_kv_release_at_completion = True  # :1543
        self.log_decision(
            {"kind": "completion", "request_id": runtime.request_id,
             "priority": 0},
            tick,
            decision={
                # 与离线 legacy decision_log 逐字段同构(wsc_llm_scheduler.py
                # :1618-1648 未设置的字段取 WscLlmRequestPlan 默认值)。
                "terminal_kv_release_at_completion":
                    runtime.terminal_kv_release_at_completion,
                "kv_state_after_completion": "UNTRACKED",
                "kv_instance_after_completion": None,
                "completion_evictions": [],
            },
        )

    def _on_request_complete(self, request_id: str, tick: int) -> None:
        """离线 decode 完成分支的下一 turn arrival 排程(:1524-1531):
        now + 该 turn 的 inter_request_interval_ns 注册未来 alarm
        (向 ingress,不再 push 到事件堆)。

        offline: wsc_llm_scheduler.py:1524-1531
        """
        runtime = self.runtime_by_request_id[request_id]
        # §7.3:_runtime_index O(1) 定位(替换 O(N) 全量扫描)。
        following = self.next_request[self._runtime_index[runtime.request_id]]
        if following is None:
            return  # session 最后一 turn:无下一次 arrival
        interval = self._interval_ns(following)  # :1525-1528
        # :1529-1530 push_event(now_ns + interval, 1, "arrival", following) ->
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
        """离线 :1336-1341 push_event(time_ns, priority=1, kind="arrival")。
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
        """离线 :1449-1452 同 tick 批 + :1452 按 (priority, sequence) 排序。
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
        """离线 start_ready_iterations(:1357-1469)的排队/准入部分;计时部分
        (LUT 估计 + push iteration_complete)删除,由真实完成事件推进。

        阶段 4 §7.3:逐实例 serve 改为只访问 ready frontier(非忙且有排队
        工作的实例;sorted 保持实例 index 序 = 离线 :1359 的循环序,决策
        逐字节不变)。frontier 在到达/完成(设)与发射(清)时增量维护,
        单批访问条目数 = 到期/受影响条目数,与总 request 数无关。

        offline: wsc_llm_scheduler.py:1357-1469
        """
        for instance_index in sorted(self._ready_frontier):  # §7.3 frontier
            self._profile_scan()  # §7.3:frontier 访问条目(就绪实例)
            state = self.instances[instance_index]
            if state.busy:  # 防御:frontier 与 busy 失步即内部错误
                continue
            if state.phase_role == PREFILL_ROLE:  # :1364-1395
                if state.active_decode:  # :1365-1366
                    raise RuntimeError(
                        "Prefill-only instance contains Decode work")
                if not state.qp:  # :1367-1368
                    continue
                runtime = state.qp[0]  # :1369
                if runtime.kv_allocation is None:
                    # :1370-1376 准入时静态路由(legacy 的 decode 目标在准入
                    # 才决定,与 session_lru 的 arrival 路由不同)。
                    route = self.static_mapping.route_for_prefill(state.index)
                    decode_state = self.instances[route.decode_instance_index]
                    if decode_state.phase_role != DECODE_ROLE:  # :1373-1375
                        raise RuntimeError(
                            "static mapping selected a non-Decode instance")
                    if runtime.session_id in self.session_allocations:  # :1376-1380
                        raise RuntimeError(
                            "session {} already has a retained KV allocation "
                            "before Prefill admission".format(
                                runtime.session_id))
                    allocation = self.allocator.try_allocate(  # :1381-1386
                        request_id=runtime.request_id,
                        route=route,
                        total_bytes=kv_cache_bytes_for_tokens(
                            self.config.model, runtime.final_context_tokens),
                    )
                    if allocation is None:
                        # :1387-1392 WSC-LLM Algorithm 2 line 6:FCFS 队头
                        # 阻塞——停止本 request 与同队全部后续 request,
                        # 容量由后续调度事件重查(实例保持 ready frontier)。
                        continue
                    runtime.decode_instance_index = (  # :1393
                        route.decode_instance_index)
                    runtime.static_route = route  # :1394
                    runtime.kv_allocation = allocation  # :1395
                    self.session_allocations[runtime.session_id] = allocation  # :1396
                self._emit_prefill(runtime, tick)  # 聚合:prefill 整段
                state.busy = True  # :1456
                self._ready_frontier.discard(state.index)  # §7.3:发射即忙
            else:  # DECODE_ROLE, :1397-1415
                if state.qp:  # :1398-1399
                    raise RuntimeError(
                        "Decode-only instance contains Prefill work")
                if not state.active_decode:  # :1400-1401
                    continue
                runtime = state.active_decode[0]  # 聚合:一次发射队首整段
                self._emit_decode(runtime, tick)
                state.busy = True  # :1456
                self._ready_frontier.discard(state.index)  # §7.3:发射即忙

    # ------------------------------------------------------------- 发射 --

    def _emit_prefill(self, runtime, tick: int) -> None:
        """聚合粒度发射 prefill 整段(与离线 ET 按 request 整段一致;
        离线逐 chunk 迭代在在线不存在)。watch 注册 PREFILL_DRAIN。

        offline: wsc_llm_scheduler.py:1364-1395(发射对象为整段而非 chunk)
        """
        plan = self._plan_dict(runtime)
        members = self.graph.emit_prefill_batch(plan)
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
                # 与离线 legacy decision_log 逐字段同构(离线 legacy 不设置
                # history_action 等字段,取 WscLlmRequestPlan 默认值
                # NO_HISTORY / "UNTRACKED" / 0 / 空;history_source_instance_
                # index 与 history_transfer_bytes 是离线实际记录的字段)。
                "prefill_instance_index": runtime.prefill_instance_index,
                "prefill_assignment_key": list(runtime.prefill_assignment_key),
                "estimated_arrival_ns": runtime.estimated_arrival_ns,
                "effective_prefill_tokens": 0,
                "history_action": NO_HISTORY,
                "history_cache_state_before": "UNTRACKED",
                "history_source_instance_index":
                    runtime.history_source_instance_index,
                "history_transfer_bytes": runtime.history_transfer_bytes,
                "history_recompute_tokens": 0,
                "admission_evictions": [],
                "decode_target_evictions": [],
            },
        )

    def _emit_decode(self, runtime, tick: int) -> None:
        """聚合粒度发射 decode 整段(transfer 3000 + decode 整段 + end
        barrier)。watch 注册 DECODE_COMPLETION(C++ 同 fire 推
        DECODE_COMPLETION + REQUEST_COMPLETE 两条 completed_groups)。

        offline: wsc_llm_scheduler.py:1397-1415(发射对象为整段而非 chunk)
        """
        plan = self._plan_dict(runtime)
        members = self.graph.emit_decode_batch(plan)
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
                # legacy 无会话 KV 迁移对象(离线 WscLlmRequestPlan 默认
                # None,_kv_transfer_dict(None) = None)。
                "prefill_decode_transfer": None,
            },
        )

    # ------------------------------------------------------------- 助手 --

    def _plan_dict(self, runtime) -> dict:
        """graph_batch_builder 消费的 plan 字段(request 事实 + 在线决策)。

        构图 history 语义:turn>0 用 NOC_MIGRATE(共享构图器分支)——单个
        history_kv transfer 从历史 KV 主位(前序 decode 实例)到 prefill
        实例,字节 = kv_cache_bytes_for_tokens(history_tokens_before)。
        与离线 legacy ET 的差异仅粒度(离线逐 kv_allocation.piece 多个
        transfer,category 1000+piece*100;在线单 transfer,category 1000):
        总字节一致,见差分归因类别。决策日志的 history_action 仍按离线
        口径记录 NO_HISTORY(见 _emit_prefill;离线 legacy 不设置该字段)。
        """
        return {
            "request_id": runtime.request_id,
            "session_id": runtime.session_id,
            "turn_index": runtime.turn_index,
            "queue_index": runtime.queue_index,
            "prefill_instance_index": runtime.prefill_instance_index,
            "decode_instance_index": runtime.decode_instance_index,
            "history_action": (
                NOC_MIGRATE if runtime.turn_index > 0 else NO_HISTORY),
            "history_source_instance_index":
                runtime.history_source_instance_index,
            "history_transfer_bytes": runtime.history_transfer_bytes,
            "history_recompute_tokens": 0,
            "history_tokens_before": runtime.history_tokens_before,
            "prefill_context_tokens": runtime.prefill_context_tokens,
            "prefill_length": runtime.prefill_length,
            "decode_length": runtime.decode_length,
        }

    def _prefill_snapshots(self):
        """离线 prefill_snapshots(:1343-1349):仅 prefill 角色实例,
        在 instances 序(实例 index 序)上构建。"""
        return tuple(
            PrefillQueueSnapshot(instance_index=state.index,
                                 request_count=len(state.qp))
            for state in self.instances
            if state.phase_role == PREFILL_ROLE
        )

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

    def kv_event_payload_legacy(self) -> dict:
        """legacy run-end 终值报告(与 metrics_integration.kv_event_payload_
        legacy 同构;legacy 无逐事件 KV 日志,there is no KV event log)。
        在线侧由 online_service 在 verify_run_end 后写
        bridge_dir/kv_event_payload_legacy.json(runner 归档到 results/,
        run-end 终值审计件)。"""
        return {
            "policy": "wsc_relevant_pd_static_decode_domain",
            "final_remaining_capacity_bytes": list(
                self.allocator.remaining_capacity),
        }

    def verify_run_end(self) -> None:
        """基类协议校验之上,叠加离线 :1582-1593 的收尾断言与阶段 4 §7.3
        结束审计(arrival heap / ready frontier 全空)。"""
        super().verify_run_end()
        if self.completed_requests != len(self.runtimes):
            raise RuntimeError(
                "legacy run ended with {}/{} requests complete".format(
                    self.completed_requests, len(self.runtimes)))
        if any(state.busy or state.qp or state.active_decode
               for state in self.instances):
            raise RuntimeError("legacy run ended with non-idle instance state")
        # 离线 :1590-1593:terminal session KV 全释放。
        if self.session_allocations:
            raise RuntimeError(
                "legacy run ended with terminal session KV still retained: "
                "{!r}".format(sorted(self.session_allocations)))
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
