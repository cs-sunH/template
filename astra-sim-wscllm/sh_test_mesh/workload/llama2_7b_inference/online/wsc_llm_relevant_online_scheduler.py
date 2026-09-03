#!/usr/bin/env python3
"""wsc_llm_relevant_online_scheduler.py -- relevant_distributed 第三变体的
混合骨架在线调度器(总文档《wscllm要补充的选择分析方案总文档》§2.1/§2.4/
§3.4/裁决 #19-21/#23-34;执行文档 §5.3,2026-09-02 B3)。

混合骨架(裁决 #20/理由 §5.3,连续批处理是真实推理服务的执行形态):
  - **legacy 准入/边界循环**:arrival heap / completion 批 / FCFS 队头阻
    塞 / ready frontier / 背压全部复用 legacy 模式(online/
    wsc_llm_legacy_online_scheduler.py 的迁移骨架),零新事件类型——
    try_place 返回 None 即 legacy 的 "head try_allocate 失败 -> 整队停,
    容量由后续调度事件重查"(总文档裁决 #19);
  - **主变体列车发射链**:D 侧 decode 互拼列车,_plan_train /
    _finalize_completed_trains / _instance_busy 直接 import 主变体方法
    复用(唯一副作用 train_seq 自增,字段面在 hybrid 状态类上同名提供);
    SH_TRAIN_MAX_ITER 缺省 8 继承(2026-08-22 对拍裁决);SH_FIRST_TOKEN_
    SPLIT 缺省关继承——本变体不拆车(总文档附录"其他实现注意点"),
    行为与 OFF 侧一致。

relevant_distributed 专属生命周期(总文档 §2.1,逐行对应):
  turn t 到达(t>0):释放本会话旧 KVPlacement(allocator.release)→
    plan_history_pull 多源规划(源=P 零边 LOCAL_HIT;纯规划零账本变更)
    → 入 P 的 FCFS 队列(select_prefill_instance 现状不动);
  准入(队头):allocator.try_place 三条件一次性预分配(None → FCFS 队头
    阻塞;ValueError → 配置非法 fail-closed 上抛);
  prefill 发射:turn>0 先发 1000 族多源拉回边(emit_history_pulls,recv
    链新 P frontier,物理先于 prefill 体),再走共享 emit_prefill_batch
    (构图零改动,roofline 本地口径不变);
  prefill drain(decode join):发 3100 散布写边(emit_piece_scatter,send
    侧插显式锚 = PREFILL_DRAIN watch members)+ allocator.release_staging
    (P 的 staging scratch 释放,记账提前一个传输时长,方向保守)→ 直接
    入 decode active_decode(legacy 同款无二次准入);
  decode(列车):joiners=[] 发射列车(裁决 #9:新变体不再发射 transfer
    3000,KV 迁移已由 3100 在 drain 承载——避免 3000/3100 双发);3300 读边
    经 emit_remote_reads 与列车发射同批次调用(join 锚 = 发射前 frontier
    快照,显式锚不依赖 R4 的 frontier 等价假设);physical 模式下列车体
    local_kv_bytes 只计 D 本地 piece 字节(防双计,总文档 §3.3);
  终轮 decode 完成 / 下一轮到达:释放 KVPlacement(allocator.release)。

决策日志(裁决 #23):kv_placement / kv_scatter / kv_remote_reads 三类新
行 + run 头登记 d2d_to_hbm_bandwidth_ratio(python 派生元数据,裁决 #22)
+ "背压持续时长"观测字段(防活锁观察,非机制);KV 事件流三类
(placement/release/history_pull,KV_CACHE_EVENT_COLUMNS 16 列沿用,
kv_event_payload_relevant 与 session_lru 的 16 元事件数组同构)。

run-end 审计(对齐 legacy + 列车账本 + placement 生命周期):全部请求完
成、runtime 全释放、实例全 idle、session_placements 全空、arrival heap /
ready frontier / completion gates 全空;allocator.assert_final_state;
journal 守恒门 = 纯重放函数 verify_relevant_journal_conservation(复用
MemoryActionRecorder.replay_journal,resident=0/reserved=0/physical=
weight;不改 session_kv_manager.py)。

红线(总文档 §2.4"明确不碰"):wsc_llm_scheduler.py / session_kv_manager.
py / generate_wsc_llm_trace.py 只读 import;emit_iteration_train 家族结
构不动(joiners=[] 是入参不是结构改动);legacy/session_lru 行为零变化。
"""

import heapq
import os
import sys
from collections import deque

# --------------------------------------------------------------------------
# import 路径:本文件位于 workload/llama2_7b_inference/online/,共享配置、
# 发射与调度模块在上一级。路径只做 import 用途(红线:generate_wsc_llm_trace.py /
# wsc_llm_scheduler.py / session_kv_manager.py 只读 import 与注释)。
# --------------------------------------------------------------------------
_ONLINE_DIR = os.path.dirname(os.path.abspath(__file__))
_WORKLOAD_DIR = os.path.dirname(_ONLINE_DIR)
for _path in (_ONLINE_DIR, _WORKLOAD_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from generate_wsc_llm_trace import kv_cache_bytes_for_tokens  # noqa: E402
from metrics_integration import kv_event_payload_session_lru  # noqa: E402
from online.graph_batch_builder import _prefix_of  # noqa: E402
from online.online_scheduler_base import (  # noqa: E402
    STAGE_DECODE,
    STAGE_PREFILL,
    STAGE_REQUEST,
    OnlineSchedulerBase,
)
# 主变体复用(裁决 #20"列车混合骨架"):_plan_train 自包含可复用(§5.3);
# _finalize_completed_trains / _instance_busy 的字段面(request_id/
# decode_instance_index/decode_tokens_consumed/decode_length/prefill_
# context_tokens 与实例账本 slots)在本变体的同名状态上提供。_kv_event_dict
# 为 16 列 KV 事件序列化器,直接复用避免 schema 漂移。
from online.wsc_llm_online_scheduler import (  # noqa: E402
    WscLlmOnlineScheduler,
    _kv_event_dict,
)
from relevant_kv_emission import (  # noqa: E402
    emit_history_pulls,
    emit_piece_scatter,
    emit_remote_reads,
    local_kv_override_value,
    remote_read_sources,
    remote_reads_enabled,
)
from session_kv_manager import NO_HISTORY  # noqa: E402
from wsc_llm_scheduler import (  # noqa: E402  (红线:只读 import)
    DECODE_ROLE,
    PREFILL_ROLE,
    PrefillQueueSnapshot,
    WscLlmInstanceSpec,
    build_instances,
    build_static_pd_mapping,
    select_prefill_instance,
)
from wsc_relevant_memory_scheduler import (  # noqa: E402  (B1 产物)
    RelevantKvRequest,
    WscDistributedKvAllocator,
)


def verify_relevant_journal_conservation(recorder) -> dict:
    """journal run 末守恒门(总文档 §2.4 fail-closed 清单;裁决 #23)。

    纯重放函数:复用 MemoryActionRecorder.replay_journal 的输出(逐行校验
    sequence 连续 / before-after 对账 / planner_time 单调 / 事务连续),叠
    加 relevant 变体的终态断言——每 rank resident=0、reserved=0、physical=
    weight(全部 placement 已释放,仅权重常驻)。不装 recorder(None)时由
    调用方旁路(与主变体 verify_journal_checksum 的 no-op 旁路态一致)。
    返回 replay 摘要(行数/sha256/事务数,供调用方登记审计产物)。
    """

    summary = recorder.replay_journal()
    for rank, state in sorted(summary["ranks"].items()):
        if state["resident"] or state["reserved"]:
            raise RuntimeError(
                "relevant kv delta journal run-end conservation violated on "
                f"rank {rank}: resident={state['resident']}, "
                f"reserved={state['reserved']} (every KVPlacement must be "
                "released)")
        if state["physical"] != state["weight"]:
            raise RuntimeError(
                "relevant kv delta journal run-end conservation violated on "
                f"rank {rank}: physical={state['physical']} != "
                f"weight={state['weight']}")
    return summary


def _canonical_hit_state(sources, prefill_instance_index: int) -> str:
    """kv_cache_adapter canonical 映射(总文档 §3.4/附录):全源=P → full;
    否则 partial;本策略永不 miss(无 recompute 枚举路径)。

    sources = plan_history_pull 的 HistoryPullSource 序列(或同字段
    duck-typing);prefill_instance_index = 新选 P。
    """

    for source in sources:
        if (not source.local_hit
                and int(source.source_instance_index)
                != int(prefill_instance_index)):
            return "partial"
    return "full"


class _RelevantInstanceState:
    """在线实例账本(legacy _LegacyInstanceState + 主变体列车字段 hybrids)。

    PREFILL_ROLE:qp = prefill FCFS 队列,busy = 一个 prefill 整段在飞
    (legacy 同款);DECODE_ROLE:active_decode = 已准入 decode 成员,
    busy 门 = 一个迭代列车在飞(in_flight_train,主变体 §3.2 同款)。
    """

    __slots__ = ("index", "phase_role", "qp", "active_decode",
                 "active_decode_lookup", "busy",
                 "in_flight_train", "finalized_trains", "iteration_count",
                 "train_seq")

    def __init__(self, *, index: int, phase_role: str) -> None:
        self.index = index
        self.phase_role = phase_role
        self.qp = deque()  # FCFS(append 尾入,popleft 首出)
        self.active_decode = []
        self.active_decode_lookup = set()
        self.busy = False
        # ---- 列车账本(主变体 _OnlineInstanceState 同名字段面) ----
        self.in_flight_train = None
        self.finalized_trains = []
        self.iteration_count = 0
        self.train_seq = 0


class _RelevantRequestRuntime:
    """在线请求运行账本(legacy 运行期事实 + relevant 专属 placement 面)。

    输入事实(request-neutral,来自 manifest):history_tokens_before /
    prefill_context_tokens / final_context_tokens / prefill_length /
    decode_length / queue_index / session_id / turn_index。运行期事实为
    在线决策产出(语义与 legacy/主变体蓝图同名同义)。
    """

    __slots__ = (
        "request_id", "session_id", "turn_index", "queue_index",
        "prefill_length", "decode_length",
        "history_tokens_before", "prefill_context_tokens", "final_context_tokens",
        "estimated_arrival_ns",
        "prefill_instance_index", "prefill_assignment_key",
        "decode_instance_index", "static_route",
        "kv_placement",
        "history_source_instance_indexes", "history_transfer_bytes",
        "history_remote_transfer_bytes", "history_local_hit_tokens",
        "history_canonical_hit_state", "history_pull_pieces",
        "prefill_end_members", "scatter_recv_ids",
        "decode_local_prefill_tokens",
        "decode_queue_depth_before_enqueue",
        "backpressure_since_ns", "terminal_kv_release_at_completion",
        "completion_ns",
        "decode_tokens_consumed", "decode_train_joined",
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
        self.kv_placement = None
        self.history_source_instance_indexes = None
        self.history_transfer_bytes = 0
        self.history_remote_transfer_bytes = 0
        self.history_local_hit_tokens = 0
        self.history_canonical_hit_state = None
        self.history_pull_pieces = None
        self.prefill_end_members = None      # PREFILL_DRAIN watch members
        self.scatter_recv_ids = None         # 3100 recv 节点 id(3300 锚)
        self.decode_local_prefill_tokens = None
        self.decode_queue_depth_before_enqueue = None
        self.backpressure_since_ns = None    # 背压持续时长观测(裁决 #19)
        self.terminal_kv_release_at_completion = False
        self.completion_ns = None
        # ---- 列车推进字段(主变体同款;核销时闭式推进) ----
        self.decode_tokens_consumed = 0
        self.decode_train_joined = False


class WscLlmRelevantOnlineScheduler(OnlineSchedulerBase):
    """strategy 变体:relevant_distributed(D′ 分布式 KV 存放)混合骨架。

    蓝图:legacy 准入/边界循环(wsc_llm_legacy_online_scheduler.py,逐段
    注释标注)+ 主变体列车发射链(wsc_llm_online_scheduler.py 的
    _plan_train/_finalize_completed_trains/_instance_busy 直接复用)+
    B1 分配器(WscDistributedKvAllocator)+ B2 发射层(relevant_kv_
    emission 三函数)。kv_cache_policy == "relevant_distributed"。
    """

    def __init__(self, *, manifest, config, graph, digest_sink=None,
                 decision_log_sink=None, train_ledger_sink=None,
                 profile_sink=None, mode: str = "strategy",
                 sensing: bool = False, online_stats_sink=None,
                 ledger_sink=None, kv_journal_recorder=None):
        super().__init__(
            manifest=manifest,
            config=config,
            replay=None,  # strategy 无决策日志回放源
            digest_sink=digest_sink,
            mode=mode,
            sensing=sensing,
            decision_log_sink=decision_log_sink,
            profile_sink=profile_sink,
            online_stats_sink=online_stats_sink,
            ledger_sink=ledger_sink,
        )
        if mode != "strategy":
            raise ValueError(
                "WscLlmRelevantOnlineScheduler requires mode == 'strategy'")
        if config.kv_cache_policy != "relevant_distributed":
            raise ValueError(
                "relevant strategy scheduler requires kv_cache_policy "
                "'relevant_distributed', got {!r}".format(
                    config.kv_cache_policy))
        if sensing:
            # 阶段 3 感知只接入 session_lru 变体(legacy :225-230 同款);
            # relevant 变体范围未定义,显式 fail-closed。
            raise ValueError(
                "--sensing 仅支持 kv_cache_policy session_lru_recompute "
                "(relevant_distributed 变体不挂感知)")
        self.graph = graph  # GraphBatchBuilder(与 replay 路径共用)

        # kv_remote_read A/B 开关(裁决 #11):physical(缺省)= 发射 3300 +
        # 列车体裁远程 KV 分量;ideal_masked = 整体跳过 3300 与裁剪(现状
        # 字节口径)。非法值在 remote_reads_enabled 内 fail-closed。
        self._remote_reads = remote_reads_enabled(config.kv_remote_read)

        # 拓扑 + 静态 PD 路由(legacy 蓝图 :1302-1303 同参,alpha=1.0)。
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

        # B1 分配器(逐 NPU 全局账本;纯库,裁决 #18)。注入式 journal
        # recorder:online_service 在构造本调度器之前创建并传入(本变体无
        # SessionKVCacheManager,权重预载行由 allocator 构造期落 journal,
        # run 末守恒门 physical=weight 由此成立)。
        self.kv_journal_recorder = kv_journal_recorder
        self.allocator = WscDistributedKvAllocator(
            self.topology,
            config.model,
            recorder=kv_journal_recorder,
        )

        # 实例账本(legacy :1315-1317 + 列车字段)。
        self.instances = [
            _RelevantInstanceState(index=instance.index,
                                   phase_role=instance.phase_role)
            for instance in self.topology.instances
        ]

        # session -> KVPlacement 保留账本(turn>0 arrival 释放前序,terminal
        # 完成释放终态;legacy session_allocations 同构,裁决 #14)。
        self.session_placements = {}

        # future arrival min-heap(legacy :1321-1334 同款;§7.3 键含
        # queue_index,同 tick 按冻结队列序稳定弹出)。
        self.arrival_heap = []
        self._sequence = 0

        # §7.3 ready frontier(legacy 同款:发射时清除,完成/到达时设置)。
        self._ready_frontier = set()

        # 请求运行账本(queue_index 序;legacy :1287-1315 同款索引面)。
        self.runtimes = [
            _RelevantRequestRuntime(record)
            for record in sorted(
                manifest["requests"], key=lambda item: item["queue_index"])
        ]
        self.runtime_by_request_id = {
            runtime.request_id: runtime for runtime in self.runtimes
        }
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
        # KV 事件流游标(allocator 事件是全量 tuple,只在确有 mutation 的
        # 决策批做一次增量切片,§7.3 等价于 events_since)。
        self._kv_events_emitted = 0
        self._kv_events_dirty = False
        # 列车台账(train_ledger.jsonl;主变体 M3 同款流式/缓冲双模)。
        self.train_ledger_sink = train_ledger_sink
        self.train_ledger_rows = []
        # T_max 列车长度上限(SH_TRAIN_MAX_ITER 缺省 8,主变体 §7.4 A2
        # 对拍裁决继承;0 = 不设限)。
        self._train_max_iter = int(
            os.environ.get("SH_TRAIN_MAX_ITER", "8") or 0)
        # train_id -> instance_index(哨兵事件路由,主变体同款)。
        self._train_instance_index = {}
        # 背压持续时长观测(裁决 #19:防活锁观察,非机制)。
        self.backpressure_episode_count = 0
        self.backpressure_total_ns = 0
        self._run_header_logged = False

    # ------------------------------------------------------------- 策略 --

    def run_variant_policy(self, delta) -> None:
        """决策顺序:completion 批(列车核销先于逐请求账本,主变体 §3.2
        原子提交顺序)→ arrival 批 → 准入/发射 pass(legacy 同序)→
        KV 事件流增量转发。"""
        tick = delta["tick"]
        if not self._run_header_logged:
            # run 头登记(裁决 #22):d2d/hbm 比值保持 python 派生元数据,
            # 掩盖边界走 hardware json 派生 + ratio 扫描(H28)。
            self._log_run_header(tick)
            self._run_header_logged = True

        # ---- completion 批(离线 priority 0;同 tick 先于 arrival)----
        drained = []
        decode_done = []
        request_done = []
        sentinel_trains = []
        for group in delta["completed_groups"]:
            stage = group["stage"]
            request_id = group["request_id"]
            if request_id.startswith("batch_train_"):
                sentinel_trains.append(request_id)
                continue
            if stage == STAGE_PREFILL:
                drained.append(request_id)
            elif stage == STAGE_DECODE:
                decode_done.append(request_id)
            elif stage == STAGE_REQUEST:
                request_done.append(request_id)
            else:
                raise ValueError(
                    "unknown completion stage {!r} for request {!r}".format(
                        stage, request_id))
        # 列车核销(主变体复用):退出成员 active_decode 移除 + token 闭式
        # 推进恰一次;本变体不拆车(无首步唤醒通道)。
        WscLlmOnlineScheduler._finalize_completed_trains(
            self, decode_done, sentinel_trains, tick)
        for request_id in drained:
            self._on_prefill_drain(request_id, tick)
        for request_id in decode_done:
            self._on_decode_complete(request_id, tick)
        for request_id in request_done:
            self._on_request_complete(request_id, tick)

        # ---- arrival 批(离线 priority 1)----
        for arrival in delta["arrivals"]:
            self._push_arrival(arrival, tick)
        self._drain_arrival_heap(tick)

        # ---- 准入/发射 pass(legacy start_ready_iterations 的准入部分)----
        self._admit_pass(tick)

        # ---- KV 动作流:B1 分配器本批次新产出的事件(16 列同构)----
        self._drain_kv_events()

    # ------------------------------------------------------ 列车发射 --

    def _emit_train(self, state, tick: int) -> None:
        """冻结并发射 D 实例的下一列车(主变体 _emit_train 的 relevant 适配)。

        与主变体的差异(逐项注释):
          - joiners=[](裁决 #9:新变体不再发射 transfer 3000——KV 迁移已由
            3100 散布在 drain 边界承载,避免 3000/3100 双发);join 标记随
            joiners 消失,3300 的 join 锚改用**发射前 frontier 快照**(显式
            锚,不依赖 R4 的 frontier 等价假设;跨批节点 id 引用与
            completion_gates 同构先例);
          - physical 模式下 train_plan 附 local_kv_bytes(与 pass_spans 对齐
            的 per-span 逐 rank D 本地 piece 字节;远程分量改由 3300 在源端
            计费,防双计,总文档 §3.3);
          - 列车发射后同批次逐成员发 3300 读边(emit_remote_reads 的 join
            锚名字恢复依赖当前批节点,必须同批调用;per-member 调用使
            scatter_recv_ids 锚按成员各自的 drain 取值)。
        """
        plan = WscLlmOnlineScheduler._plan_train(self, state)
        if plan is None:
            return
        train_plan = {
            "train_id": plan["train_id"],
            "instance_index": state.index,
            "joiners": [],  # 裁决 #9:不发 3000
            "pass_spans": plan["pass_spans"],
            "iterations": plan["iterations"],
            "sentinel": plan["sentinel"],
            "exit_members": [
                {"request_id": runtime.request_id,
                 "session_id": runtime.session_id}
                for runtime in plan["exit_members"]],
        }
        if self._remote_reads:
            train_plan["local_kv_bytes"] = self._train_local_kv_values(
                state, plan)
            # join 锚 = 发射前 frontier 快照(每 decode rank 的链尾 id;
            # joiners=[] 下列车头零节点,快照即列车体首节点的链上前驱,
            # 因果覆盖车头;显式锚,不依赖 R4 的 frontier 等价假设)。
            decode_ranks = self.graph.group_by_index[state.index].ranks
            join_anchors = {
                rank: self.graph.builders[rank].previous_id
                for rank in decode_ranks
            }
            if any(anchor is None for anchor in join_anchors.values()):
                raise RuntimeError(
                    "decode ranks have no prior chain node for train {} "
                    "join anchors".format(plan["train_id"]))
        else:
            join_anchors = None  # ideal_masked:无 3300,无需 join 锚
        # joiner 账本(主变体同款:首个列车加入时点移至此处;无 3000 迁移,
        # prefill_decode_transfer=None)。
        joiners = [runtime for runtime in state.active_decode
                   if not runtime.decode_train_joined]
        joiner_ids = [runtime.request_id for runtime in joiners]
        for runtime in joiners:
            runtime.decode_train_joined = True
            self._note_emitted(runtime.request_id, STAGE_DECODE)
            self._ledger_issue(runtime.request_id, tick, STAGE_DECODE,
                               state.index)
            route = runtime.static_route
            self.log_decision(
                {"kind": "decode", "request_id": runtime.request_id,
                 "priority": 0},
                tick,
                decision={
                    "decode_instance_index": runtime.decode_instance_index,
                    "static_route": {
                        "prefill_instance_index":
                            route.prefill_instance_index,
                        "decode_instance_index": route.decode_instance_index,
                        "path": list(route.path),
                        "hop_count": route.hop_count,
                        "shared_edges": [list(edge)
                                         for edge in route.shared_edges],
                    },
                    "decode_queue_depth_before_enqueue":
                        runtime.decode_queue_depth_before_enqueue,
                    # 裁决 #9:无 3000 迁移对象(散布证据在 kv_scatter 行)。
                    "prefill_decode_transfer": None,
                },
            )
        # SH_FIRST_TOKEN_SPLIT 缺省关继承:本变体不拆车(总文档附录),
        # 行为与 OFF 侧一致,无首步批/唤醒通道。
        result = self.graph.emit_iteration_train(train_plan)
        # exit 标记 watch(每退出成员一个;C++ 同 fire 推 DECODE_COMPLETION
        # + REQUEST_COMPLETE 两条 completed_groups)。
        for runtime in plan["exit_members"]:
            self._batch["watches"].append({
                "request_id": runtime.request_id,
                "stage": STAGE_DECODE,
                "generation": 1,
                "members": result["exit_members"][runtime.request_id],
                "statuses": ["Success", "Skipped"],
            })
        if plan["sentinel"]:
            self._batch["watches"].append({
                "request_id": plan["train_id"],
                "stage": STAGE_PREFILL,
                "generation": 0,
                "members": result["sentinel_members"],
                "statuses": ["Success", "Skipped"],
            })
        self._train_instance_index[plan["train_id"]] = state.index
        state.in_flight_train = plan
        self._ready_frontier.discard(state.index)  # §7.3:发射即忙
        self._emit_train_ledger_row(state, plan, joiner_ids, tick)
        if self._remote_reads:
            self._emit_train_remote_reads(state, plan, result, join_anchors,
                                          tick)

    def _train_local_kv_values(self, state, plan) -> list:
        """列车体 local_kv_bytes 覆盖值(总文档 §3.3;与 plan["pass_spans"]
        逐位对齐的 per-span 逐 rank 序列)。

        每 span 的 D 本地 token 数 = 该成员 D 上 prefill 段 token(d_prefill,
        placement 冻结后为常量)+ 已生成 decode token(consumed + step;
        decode 段恒钉 D,裁决 #7);取值口径 = 单层单份 K 本地字节
        (local_kv_override_value,M2 契约),远程分量由 3300 在源端计费。
        全本地成员的覆盖值与全量 shard 逐字节相等(数值上无差,统一走
        覆盖路径避免分支)。
        """
        tp = len(self.graph.group_by_index[state.index].ranks)
        model = self.config.model
        values = []
        for runtime, participation in plan["members"]:
            if runtime.decode_local_prefill_tokens is None:
                runtime.decode_local_prefill_tokens = sum(
                    piece.token_end - piece.token_start
                    for piece in runtime.kv_placement.pieces
                    if piece.instance_index == state.index
                    and piece.token_end <= runtime.prefill_context_tokens)
            local_prefill = runtime.decode_local_prefill_tokens
            consumed = runtime.decode_tokens_consumed
            for _step in range(participation):
                local_tokens = local_prefill + consumed + _step + 1
                values.append([
                    local_kv_override_value(model, local_tokens, tp, rank)
                    for rank in range(tp)
                ])
        if len(values) != len(plan["pass_spans"]):
            raise RuntimeError(
                "train local_kv_bytes layout mismatch: {} values for {} "
                "spans".format(len(values), len(plan["pass_spans"])))
        return values

    def _emit_train_remote_reads(self, state, plan, result, join_anchors,
                                 tick: int) -> None:
        """3300 读边发射(physical 模式;per (列车, 成员, 源),裁决 #10)。

        逐成员调用 emit_remote_reads(同批次):join 锚用发射前快照(显式);
        send 锚两档在函数内解析(源=P → prefill_end_members;源=中间 die →
        该成员自己的 3100 recv 节点 id);recv 注入该成员自己的 exit 标记
        (成员 exit = max(体尾, 其全部读边));中途成员 exit_anchors=None。
        sources 取 remote_read_sources 的升序规范序(跨列车保持同序,
        R3-② 的同 tag 重复语义)。全本地成员零 3300 边(混合批无需特判)。
        """
        routes_all = []
        read_members = []
        for runtime, participation in plan["members"]:
            sources = remote_read_sources(
                state.index, runtime.kv_placement.pieces)
            if not sources:
                continue
            reads = emit_remote_reads(
                self.graph,
                train_id=plan["train_id"],
                decode_instance_index=state.index,
                member_reads=[{
                    "request_id": runtime.request_id,
                    "queue_index": runtime.queue_index,
                    "participation": participation,
                    "prefill_instance_index": runtime.prefill_instance_index,
                    "sources": [
                        {"source_instance_index": source_index,
                         "piece_tokens": tokens}
                        for source_index, tokens in sources
                    ],
                    "prefill_end_members": runtime.prefill_end_members,
                    "exit_anchors": (
                        result["exit_members"].get(runtime.request_id)
                        if runtime.request_id in plan["exit_set"] else None),
                }],
                scatter_recv_ids=runtime.scatter_recv_ids,
                join_anchors=join_anchors,
            )
            routes_all.extend(reads["routes"])
            read_members.append({
                "request_id": runtime.request_id,
                "participation": participation,
                "sources": [
                    {"source_instance_index": source_index,
                     "piece_tokens": tokens}
                    for source_index, tokens in sources
                ],
            })
        if not routes_all:
            return  # 全本地列车:零 3300 边,无决策行(裁决 #20 混合批)
        # 决策日志(裁决 #23:挂列车发射决策行;routes 含 noc_hops 元数据,
        # hopbytes 消费口径)。
        self.log_decision(
            {"kind": "kv_remote_reads", "request_id": plan["train_id"],
             "priority": 0},
            tick,
            decision={
                "train_id": plan["train_id"],
                "instance_index": state.index,
                "iterations": plan["iterations"],
                "members": read_members,
                "routes": routes_all,
                "total_bytes": sum(route["bytes"] for route in routes_all),
            },
        )

    def _emit_train_ledger_row(self, state, plan, joiner_ids,
                               tick: int) -> None:
        """train_ledger 行(主变体同构;M3 流式/缓冲双模)。"""
        ledger_row = {
            "train_id": plan["train_id"],
            "instance_index": state.index,
            "tick": tick,
            "iterations": plan["iterations"],
            "member_count": len(plan["members"]),
            "member_iterations": sum(
                participation for _, participation in plan["members"]),
            "joiners": list(joiner_ids),
            "drains": [],
            "exits": [runtime.request_id
                      for runtime in plan["exit_members"]],
            "sentinel": plan["sentinel"],
            "prefill_chunks": 0,  # D 侧列车绝无 prefill chunk span
            "pass_spans": len(plan["pass_spans"]),
        }
        if self.train_ledger_sink is not None:
            self.train_ledger_sink(ledger_row)
        else:
            self.train_ledger_rows.append(ledger_row)

    # ------------------------------------------------------------- 边界 --

    def _on_arrival(self, runtime, tick: int) -> None:
        """arrival 批单条(legacy :1552-1578 骨架 + §3.4 多轮拉回规划)。

        turn>0:释放本会话旧 KVPlacement(裁决 #14/§3.4)→ 多源历史拉回
        规划(plan_history_pull:纯规划零账本变更;源=P 零边 LOCAL_HIT;
        总字节 = kv(history_tokens_before))→ 选 P 入队(快照在 append 前
        取,legacy 同款)。拉回的图发射在准入成功时随 prefill 批进行
        (_emit_prefill:recv 链新 P frontier,物理先于 prefill 体)。
        """
        runtime.estimated_arrival_ns = tick
        previous_placement = self.session_placements.pop(
            runtime.session_id, None)
        if runtime.turn_index > 0:
            if previous_placement is None:
                raise RuntimeError(
                    "session {} has no prior KV placement".format(
                        runtime.session_id))
            # 释放点 1/2:下一轮到达释放(总文档 §2.1)。
            self.allocator.release(previous_placement, now_ns=tick)
            self._kv_events_dirty = True
            runtime.history_transfer_bytes = kv_cache_bytes_for_tokens(
                self.config.model, runtime.history_tokens_before)
            snapshots = self._prefill_snapshots()
            selected = select_prefill_instance(snapshots)
            pull_sources = self.allocator.plan_history_pull(
                previous_placement,
                history_tokens=runtime.history_tokens_before,
                target_instance_index=selected,
                now_ns=tick,
            )
            self._kv_events_dirty = True  # history_pull 类 KV 事件
            runtime.history_pull_pieces = previous_placement.pieces
            runtime.history_source_instance_indexes = sorted(
                int(source.source_instance_index)
                for source in pull_sources
                if not source.local_hit)
            # 逐源 token 量从旧 pieces 精确求和(HistoryPullSource 的
            # token_start/token_end 是首末段端点,同实例多段中间可有空洞,
            # 端点差不等于 token 量;字节以 shard_bytes 为准)。
            local_hit_tokens = 0
            remote_tokens = 0
            for piece in previous_placement.pieces:
                low = max(piece.token_start, 0)
                high = min(piece.token_end, runtime.history_tokens_before)
                if high <= low:
                    continue
                if piece.instance_index == selected:
                    local_hit_tokens += high - low
                else:
                    remote_tokens += high - low
            if local_hit_tokens + remote_tokens != runtime.history_tokens_before:
                raise RuntimeError(
                    "history pull token conservation violated for request "
                    f"{runtime.request_id}: {local_hit_tokens} + "
                    f"{remote_tokens} != {runtime.history_tokens_before}")
            if sum(sum(source.shard_bytes) for source in pull_sources
                   if not source.local_hit) != kv_cache_bytes_for_tokens(
                    self.config.model, remote_tokens):
                raise RuntimeError(
                    "history pull byte conservation violated for request "
                    f"{runtime.request_id}")
            runtime.history_local_hit_tokens = local_hit_tokens
            runtime.history_remote_transfer_bytes = kv_cache_bytes_for_tokens(
                self.config.model, remote_tokens)
            # canonical 映射(总文档 §3.4):全源=P → full;否则 partial;
            # 永不 miss。
            runtime.history_canonical_hit_state = _canonical_hit_state(
                pull_sources, selected)
        else:
            snapshots = self._prefill_snapshots()
            selected = select_prefill_instance(snapshots)
        selected_snapshot = next(
            snapshot for snapshot in snapshots
            if snapshot.instance_index == selected)
        runtime.prefill_instance_index = selected
        runtime.prefill_assignment_key = selected_snapshot.ordering_key
        self.instances[selected].qp.append(runtime)
        self._note_instance_ready(selected)
        # 阶段 3 感知账本(legacy 同款;感知关闭时同样记账,簿记免费)。
        self._ledger_admit(runtime.request_id, tick,
                           {"type": "prefill_qp", "instance_index": selected})

    def _on_prefill_drain(self, request_id: str, tick: int) -> None:
        """prefill 完成(legacy :1488-1510 骨架 + §2.1 drain 三件事)。

        ① 3100 散布写边(emit_piece_scatter:send 侧插显式锚 = 本请求
        PREFILL_DRAIN watch members,recv 正常链入 owner 的 D/die frontier,
        P-piece 零边,两端 hbm_charge=true,函数内守恒断言);
        ② 释放 P 的 staging scratch(release_staging:只放 scratch,own
        piece 的 resident 保留到终轮;物理释放点≈3100 完成时,记账提前一
        个传输时长,方向保守);
        ③ 直接入 active_decode(legacy 同款无二次 decode 准入——KV 已在
        准入时按 final 一次性预分配)。
        """
        runtime = self.runtime_by_request_id[request_id]
        state = self.instances[runtime.prefill_instance_index]
        if not state.qp or state.qp[0] is not runtime:
            raise RuntimeError("Prefill FCFS queue order was corrupted")
        state.busy = False
        state.qp.popleft()
        if state.qp:
            self._ready_frontier.add(state.index)
        else:
            self._ready_frontier.discard(state.index)
        placement = runtime.kv_placement
        if placement is None or runtime.static_route is None:
            raise RuntimeError(
                "Prefill completed without a reserved WSC KV placement")
        # ① 3100 散布(总文档 §2.1/裁决 #9/#13)。
        scatter = emit_piece_scatter(
            self.graph,
            queue_index=runtime.queue_index,
            prefix=_prefix_of(self._plan_dict(runtime)),
            request_id=runtime.request_id,
            prefill_instance_index=runtime.prefill_instance_index,
            pieces=placement.pieces,
            prefill_context_tokens=runtime.prefill_context_tokens,
            prefill_end_members=runtime.prefill_end_members,
        )
        runtime.scatter_recv_ids = scatter["recv_ids"]
        # ② staging scratch 释放(总文档 §2.1;记账口径见 docstring)。
        self.allocator.release_staging(placement, now_ns=tick)
        self._kv_events_dirty = True
        # 决策日志 kv_scatter 行(裁决 #23;routes 含 noc_hops)。
        self.log_decision(
            {"kind": "kv_scatter", "request_id": runtime.request_id,
             "priority": 0},
            tick,
            decision={
                "prefill_instance_index": runtime.prefill_instance_index,
                "decode_instance_index": runtime.decode_instance_index,
                "scatter_bytes": scatter["scatter_bytes"],
                "prefill_stay_bytes": scatter["prefill_stay_bytes"],
                "scatter_tokens_by_owner":
                    scatter["scatter_tokens_by_owner"],
                "prefill_stay_tokens": scatter["prefill_stay_tokens"],
                "staging_released_shard_bytes":
                    list(placement.staging_shard_bytes),
                "routes": scatter["routes"],
            },
        )
        # ③ decode 入队(legacy :1500-1508)。
        decode_state = self.instances[runtime.decode_instance_index]
        if decode_state.phase_role != DECODE_ROLE:
            raise RuntimeError("static mapping selected a non-Decode instance")
        runtime.decode_queue_depth_before_enqueue = (
            len(decode_state.active_decode))
        decode_state.active_decode.append(runtime)
        decode_state.active_decode_lookup.add(runtime)
        if not WscLlmOnlineScheduler._instance_busy(decode_state):
            self._ready_frontier.add(decode_state.index)
        self._ledger_admit(
            runtime.request_id, tick,
            {"type": "active_decode",
             "instance_index": runtime.decode_instance_index})

    def _on_decode_complete(self, request_id: str, tick: int) -> None:
        """decode 完成账本(列车核销后执行,主变体 §3.2 同序):completion
        决策行 + 终轮标记(实际释放在同 tick 的 REQUEST_COMPLETE 边界,
        与主变体 retire 位置一致)。"""
        runtime = self.runtime_by_request_id[request_id]
        runtime.completion_ns = tick
        self.completed_requests += 1
        following = self.next_request[self._runtime_index[request_id]]
        runtime.terminal_kv_release_at_completion = following is None
        self.log_decision(
            {"kind": "completion", "request_id": runtime.request_id,
             "priority": 0},
            tick,
            decision={
                "terminal_kv_release_at_completion":
                    runtime.terminal_kv_release_at_completion,
                # relevant 变体的会话 KV 状态口径:非终轮 = 驻留等待下一轮
                # 拉回;终轮 = 本 tick REQUEST_COMPLETE 边界释放。
                "kv_state_after_completion": (
                    "RELEASED_AT_COMPLETION"
                    if runtime.terminal_kv_release_at_completion
                    else "RESIDENT_PENDING_NEXT_TURN"),
                "kv_instance_after_completion": None,
                "completion_evictions": [],
            },
        )

    def _on_request_complete(self, request_id: str, tick: int) -> None:
        """REQUEST_COMPLETE 边界(legacy :1524-1531 排程 + 主变体 retire
        位置):下一 turn future alarm / 终轮 placement 释放 + gate 回收。"""
        runtime = self.runtime_by_request_id[request_id]
        if (runtime.decode_tokens_consumed != runtime.decode_length
                or not runtime.decode_train_joined):
            raise RuntimeError(
                "completed request {!r} has an incomplete decode ledger"
                .format(request_id))
        index = self._runtime_index[request_id]
        following = self.next_request[index]
        self.next_request[index] = None
        if following is not None:
            interval = self._interval_ns(following)
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
            # terminal turn:completion gate 回收 + 释放点 2/2(终轮完成释放,
            # 总文档 §2.1)。
            self.graph.retire_completion_gate(runtime.session_id)
            placement = self.session_placements.pop(runtime.session_id, None)
            if placement is None:
                raise RuntimeError(
                    "terminal request has no retained KV placement")
            if placement.request_id != runtime.request_id:
                raise RuntimeError(
                    "terminal session KV placement does not match request")
            self.allocator.release(placement, now_ns=tick)
            self._kv_events_dirty = True
        # M4 核销即删(主变体同款纪律):完成后 runtime 胖字段再无读者。
        runtime.kv_placement = None
        runtime.prefill_end_members = None
        runtime.scatter_recv_ids = None
        runtime.history_pull_pieces = None
        self.runtime_by_request_id.pop(request_id, None)
        if self._runtime_index.pop(request_id, None) != index:
            raise RuntimeError("relevant runtime index drift for {!r}".format(
                request_id))
        self.runtimes[index] = None

    # ------------------------------------------------------- arrival heap --

    def _push_arrival(self, arrival: dict, tick: int) -> None:
        """legacy :1336-1341 同款(键含 queue_index,同 tick 冻结队列序)。"""
        runtime = self.runtime_by_request_id[arrival["request_id"]]
        heapq.heappush(
            self.arrival_heap,
            (tick, 1, runtime.queue_index, self._sequence, "arrival",
             runtime))
        self._sequence += 1

    def _drain_arrival_heap(self, tick: int) -> None:
        """legacy :1449-1452 同款(只消费 tick <= current_tick 的到期项)。"""
        while self.arrival_heap and self.arrival_heap[0][0] <= tick:
            _, _, _, _, kind, payload = heapq.heappop(self.arrival_heap)
            if kind != "arrival":
                raise RuntimeError("arrival heap contains {!r}".format(kind))
            self._profile_scan()
            self._on_arrival(payload, tick)

    # ------------------------------------------------------------- 准入 --

    def _admit_pass(self, tick: int) -> None:
        """legacy start_ready_iterations 的准入/发射部分(计时部分删除,
        真实完成事件推进):P 侧 head try_place + FCFS 队头阻塞;D 侧列车
        发射(busy 门 = 一个列车在飞)。"""
        for instance_index in sorted(self._ready_frontier):
            self._profile_scan()
            state = self.instances[instance_index]
            if WscLlmOnlineScheduler._instance_busy(state):
                continue  # 防御:frontier 与 busy 失步即内部错误
            if state.phase_role == PREFILL_ROLE:
                if state.active_decode:
                    raise RuntimeError(
                        "Prefill-only instance contains Decode work")
                if not state.qp:
                    continue
                runtime = state.qp[0]
                if runtime.kv_placement is None:
                    # 准入时静态路由(legacy :1370-1376 同款)。
                    route = self.static_mapping.route_for_prefill(state.index)
                    if (self.topology.instance(route.decode_instance_index)
                            .phase_role != DECODE_ROLE):
                        raise RuntimeError(
                            "static mapping selected a non-Decode instance")
                    if runtime.session_id in self.session_placements:
                        raise RuntimeError(
                            "session {} already has a retained KV placement "
                            "before Prefill admission".format(
                                runtime.session_id))
                    if (runtime.prefill_context_tokens
                            + runtime.decode_length
                            != runtime.final_context_tokens):
                        raise RuntimeError(
                            "request {!r} manifest context mismatch".format(
                                runtime.request_id))
                    placement = self.allocator.try_place(
                        RelevantKvRequest(
                            request_id=runtime.request_id,
                            session_id=runtime.session_id,
                            turn_index=runtime.turn_index,
                            prefill_context_tokens=(
                                runtime.prefill_context_tokens),
                            decode_tokens=runtime.decode_length,
                        ),
                        route,
                        now_ns=tick,
                    )
                    if placement is None:
                        # FCFS 队头阻塞(裁决 #19:None → 整队停,容量由
                        # 后续调度事件重查,实例保持 ready frontier);
                        # ValueError(空域不可行)不在此捕获——配置非法
                        # fail-closed 上抛,整轮终止。
                        if runtime.backpressure_since_ns is None:
                            runtime.backpressure_since_ns = tick
                            self.backpressure_episode_count += 1
                        continue
                    # 背压持续时长观测(裁决 #19:决策日志字段,非机制)。
                    if runtime.backpressure_since_ns is not None:
                        duration = tick - runtime.backpressure_since_ns
                        self.backpressure_total_ns += duration
                        runtime.backpressure_since_ns = None
                    else:
                        duration = None
                    runtime.decode_instance_index = route.decode_instance_index
                    runtime.static_route = route
                    runtime.kv_placement = placement
                    self.session_placements[runtime.session_id] = placement
                    self._kv_events_dirty = True
                    self._log_kv_placement(runtime, placement, route,
                                           duration, tick)
                self._emit_prefill(runtime, tick)
                state.busy = True
                self._ready_frontier.discard(state.index)
            else:  # DECODE_ROLE:列车发射(主变体 §3.2)
                if state.qp:
                    raise RuntimeError(
                        "Decode-only instance contains Prefill work")
                if not state.active_decode:
                    continue
                self._emit_train(state, tick)

    # ------------------------------------------------------------- 发射 --

    def _emit_prefill(self, runtime, tick: int) -> None:
        """prefill 整段发射(legacy _emit_prefill 骨架 + 1000 族多源拉回)。

        turn>0:先发 emit_history_pulls(send 链源 frontier、recv 链新 P
        frontier,两端 hbm_charge=true;总字节 = kv(history_tokens_before)),
        再走共享 emit_prefill_batch——构图零改动,prefill 体物理后于拉回
        (per-rank 发行序保证)。共享构图器的 history 分支传 NO_HISTORY
        (多源拉回由本变体自发射,取 else 分支的 interval 控制 1900:新
        prefill 等待前序完成 + interval,legacy 单源分支不重复触发)。
        """
        plan = self._plan_dict(runtime)
        prefix = _prefix_of(plan)
        pull_routes = []
        pull_sources_meta = []
        if runtime.turn_index > 0:
            if not runtime.history_pull_pieces:
                raise RuntimeError(
                    "turn>0 admission lost its history pull plan")
            pull = emit_history_pulls(
                self.graph,
                queue_index=runtime.queue_index,
                prefix=prefix,
                request_id=runtime.request_id,
                old_pieces=runtime.history_pull_pieces,
                new_prefill_instance_index=runtime.prefill_instance_index,
                history_tokens_before=runtime.history_tokens_before,
            )
            pull_routes = pull["routes"]
            pull_sources_meta = pull["sources"]
        members = self.graph.emit_prefill_batch(plan)
        # 3100 send 锚 / 3300 P 源锚 = PREFILL_DRAIN watch members(每 rank
        # 末个真实 prefill 节点 id,显式持有跨批引用)。
        runtime.prefill_end_members = members
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
            "decode_instance_index": runtime.decode_instance_index,
        })
        prefill_decision = {
            "prefill_instance_index": runtime.prefill_instance_index,
            "prefill_assignment_key": list(runtime.prefill_assignment_key),
            "estimated_arrival_ns": runtime.estimated_arrival_ns,
            "effective_prefill_tokens": runtime.prefill_length,
            # relevant 变体多源拉回不走共享构图器的 NOC_MIGRATE 单源分支
            # (1000 族由 emit_history_pulls 多源发射);决策日志按离线口径
            # 记 NO_HISTORY,证据见新字段(legacy 同款字段兼容)。
            "history_action": NO_HISTORY,
            "history_cache_state_before": (
                None if runtime.turn_index == 0
                else runtime.history_canonical_hit_state),
            "history_source_instance_index": (
                (runtime.history_source_instance_indexes or [None])[0]
                if runtime.turn_index > 0 else None),
            "history_transfer_bytes": runtime.history_transfer_bytes,
            "history_recompute_tokens": 0,
            "admission_evictions": [],
            "decode_target_evictions": [],
            # ---- relevant 专属证据字段(§3.4;kv_cache_adapter 映射输入) ----
            "history_canonical_hit_state": runtime.history_canonical_hit_state,
            "history_source_instance_indexes":
                runtime.history_source_instance_indexes,
            "history_remote_transfer_bytes":
                runtime.history_remote_transfer_bytes,
            "history_local_hit_tokens": runtime.history_local_hit_tokens,
            "history_pull_sources": pull_sources_meta,
            "history_pull_routes": pull_routes,  # 含 noc_hops(hopbytes)
        }
        self.log_decision(
            {"kind": "prefill", "request_id": runtime.request_id,
             "priority": 0},
            tick,
            decision=prefill_decision,
        )
        # P 侧退化列车台账行(主变体同款:prefill_chunks>0 与 member_count
        # 互斥)。
        state = self.instances[runtime.prefill_instance_index]
        state.train_seq += 1
        prefill_chunks = -(-max(runtime.prefill_length, 1)
                           // int(self.config.prefill_chunk_size))
        ledger_row = {
            "train_id": "prefill_train_i{}_{}".format(
                state.index, state.train_seq),
            "instance_index": state.index,
            "tick": tick,
            "iterations": prefill_chunks,
            "member_count": 0,
            "member_iterations": 0,
            "joiners": [],
            "drains": [runtime.request_id],
            "exits": [],
            "prefill_chunks": prefill_chunks,
            "pass_spans": prefill_chunks,
        }
        if self.train_ledger_sink is not None:
            self.train_ledger_sink(ledger_row)
        else:
            self.train_ledger_rows.append(ledger_row)

    def _log_kv_placement(self, runtime, placement, route, backpressure_ns,
                          tick: int) -> None:
        """kv_placement 决策行(裁决 #8/#23:位置元数据 = KVPlacement page
        table,准入时冻结,随决策日志序列化;含背压持续时长观测)。"""
        self.log_decision(
            {"kind": "kv_placement", "request_id": runtime.request_id,
             "priority": 0},
            tick,
            decision={
                "prefill_instance_index": placement.prefill_instance_index,
                "decode_instance_index": placement.decode_instance_index,
                "static_route": {
                    "prefill_instance_index": route.prefill_instance_index,
                    "decode_instance_index": route.decode_instance_index,
                    "path": list(route.path),
                    "hop_count": route.hop_count,
                    "shared_edges": [list(edge)
                                     for edge in route.shared_edges],
                },
                "total_tokens": placement.total_tokens,
                "prefill_context_tokens":
                    runtime.prefill_context_tokens,
                "decode_tokens": runtime.decode_length,
                "pieces": [
                    {"instance_index": piece.instance_index,
                     "token_start": piece.token_start,
                     "token_end": piece.token_end,
                     "tier": piece.tier,
                     "distance_to_decode": piece.distance_to_decode}
                    for piece in placement.pieces
                ],
                "staging_shard_bytes": list(placement.staging_shard_bytes),
                "available_shard_bytes_after": list(
                    self.allocator.available_shard_bytes(route)),
                # 背压持续时长观测(裁决 #19:防活锁观察;None = 首次准入
                # 即成功,无阻塞片段)。
                "backpressure_duration_ns": backpressure_ns,
            },
        )

    def _log_run_header(self, tick: int) -> None:
        """run 头登记(决策日志首行):d2d_to_hbm_bandwidth_ratio 派生元数据
        (裁决 #22)+ 变体开关姿态(A/B 与 T_max 可从日志侧复核)。"""
        self.log_decision(
            {"kind": "run_header", "request_id": "", "priority": 0},
            tick,
            decision={
                "kv_cache_policy": "relevant_distributed",
                "kv_remote_read": self.config.kv_remote_read,
                "d2d_to_hbm_bandwidth_ratio":
                    self.config.hardware.d2d_to_hbm_bandwidth_ratio,
                "sh_train_max_iter": self._train_max_iter,
                "sh_first_token_split": "off",
            },
        )

    # ------------------------------------------------------------- 助手 --

    def _plan_dict(self, runtime) -> dict:
        """graph_batch_builder 消费的 plan 字段(legacy 同面;turn>0 的
        history_action 传 NO_HISTORY——多源拉回由本变体自发射,共享构图器
        走 interval 控制 1900 分支,见 _emit_prefill docstring)。"""
        return {
            "request_id": runtime.request_id,
            "session_id": runtime.session_id,
            "turn_index": runtime.turn_index,
            "queue_index": runtime.queue_index,
            "prefill_instance_index": runtime.prefill_instance_index,
            "decode_instance_index": runtime.decode_instance_index,
            "history_action": NO_HISTORY,
            "history_source_instance_index": None,
            "history_transfer_bytes": 0,
            "history_recompute_tokens": 0,
            "history_tokens_before": runtime.history_tokens_before,
            "prefill_context_tokens": runtime.prefill_context_tokens,
            "prefill_length": runtime.prefill_length,
            "decode_length": runtime.decode_length,
        }

    def _prefill_snapshots(self):
        """legacy prefill_snapshots(:1343-1349)同款。"""
        return tuple(
            PrefillQueueSnapshot(instance_index=state.index,
                                 request_count=len(state.qp))
            for state in self.instances
            if state.phase_role == PREFILL_ROLE
        )

    def _note_instance_ready(self, instance_index: int) -> None:
        """§7.3 ready frontier(busy 门按 phase_role 分裂:P=busy,
        D=in_flight_train,主变体 _instance_busy 口径)。"""
        if not WscLlmOnlineScheduler._instance_busy(
                self.instances[instance_index]):
            self._ready_frontier.add(instance_index)

    def _interval_ns(self, runtime) -> int:
        spec = self.config.request_queue[runtime.queue_index]
        return spec.inter_request_interval_ns

    def _drain_kv_events(self) -> None:
        """B1 分配器事件流的增量转发(16 列 KV_CACHE_EVENT_COLUMNS 同构,
        session_lru kv_actions 口径)。只在确有 mutation 的决策批做一次
        全量 tuple 切片(等价 events_since;分配器无压缩接口,游标单调)。"""
        if not self._kv_events_dirty:
            return
        events = self.allocator.events
        if len(events) > self._kv_events_emitted:
            self._batch["kv_actions"].extend(
                _kv_event_dict(event)
                for event in events[self._kv_events_emitted:])
            self._kv_events_emitted = len(events)
        self._kv_events_dirty = False

    # --------------------------------------------------------------- 收尾 --

    def kv_event_payload_relevant(self) -> dict:
        """run-end KV 事件载荷(16 元事件数组风格,session_lru 同构——
        metrics_integration.kv_event_payload_session_lru 直接复用;裁决
        #23)。由 online_service 在 verify_run_end 后写
        bridge_dir/kv_event_payload_relevant.json(runner 归档到 results/)。"""
        return {
            "policy": "relevant_distributed",
            "events": kv_event_payload_session_lru(self.allocator.events),
        }

    def verify_run_end(self) -> None:
        """run-end 审计(总文档 §2.4 fail-closed 清单,对齐 legacy + 列车
        账本 + placement 生命周期 + journal 守恒门)。"""
        super().verify_run_end()
        if self.completed_requests != len(self.runtimes):
            raise RuntimeError(
                "relevant run ended with {}/{} requests complete".format(
                    self.completed_requests, len(self.runtimes)))
        if (self.runtime_by_request_id or self._runtime_index
                or any(runtime is not None for runtime in self.runtimes)
                or any(runtime is not None for runtime in self.next_request)):
            raise RuntimeError(
                "relevant run ended with retained request runtimes")
        if any(state.busy or state.qp or state.active_decode
               or state.in_flight_train is not None or state.finalized_trains
               for state in self.instances):
            raise RuntimeError(
                "relevant run ended with non-idle instance state")
        # terminal session KV 全释放(legacy :1590-1593 同构)。
        if self.session_placements:
            raise RuntimeError(
                "relevant run ended with terminal session KV still "
                "retained: {!r}".format(sorted(self.session_placements)))
        # §7.3 结束审计:heap/ready set 全空。
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
                "relevant run ended with unretired completion gates: {!r}"
                .format(sorted(self.graph.completion_gates)))
        # B1 分配器终态:placements 全空、resident=staging=0;事件游标
        # 与事件流长度一致(全部 placement/release/history_pull 事件均已
        # 转发进 kv_actions)。
        self.allocator.assert_final_state()
        if self._kv_events_emitted != len(self.allocator.events):
            raise RuntimeError(
                "relevant run ended with undrained KV events: forwarded "
                f"{self._kv_events_emitted} of "
                f"{len(self.allocator.events)}")
        # journal 守恒门(纯重放:resident=0/reserved=0/physical=weight;
        # 未装 recorder 时 no-op,与主变体旁路态一致)。
        if self.kv_journal_recorder is not None:
            verify_relevant_journal_conservation(self.kv_journal_recorder)
