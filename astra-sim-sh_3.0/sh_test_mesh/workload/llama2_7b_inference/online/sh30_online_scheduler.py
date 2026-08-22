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
    decode 完成（:3986-3999）                       _complete_requests（同 tick）
    下一 turn arrival（:4000-4006）                 _complete_requests（同 tick）
  completion_order 批（:4008-4042）               _complete_requests
    mark_complete / enforce_reserve / 快照          同名调用逐行迁移
  arrival 批（:4044-4058）                        _on_arrival（记 arrival →
                                                 pending_admissions → retry；
                                                 truncate_history 已随 sidecar
                                                 机制移除，recompute 单口径下
                                                 恒 no-op）
  admit_waiting_requests（:4060-4061）            _admit_pass 同 tick 末尾复查
  start_ready_iterations（:3833-3890）            _plan_and_emit_trains 的列车
                                                 发射：**直接 Roofline 计时部分
                                                 删除**（:3853-3858 估算与
                                                 :3885-3890 推事件），排队/配对
                                                 语义（busy 判定、FCFS qp、
                                                 active_decode 成员）原样保留
                                                 于账本逻辑。

拼 batch 改造（2026-08-22，设计文档《层次 B Continuous Batching 改造》
§3.2"迭代列车聚合发射"；照母本 sh_1.0 定型版机制移植）：层次 B 从
"请求级整段串行"重构为"实例迭代级列车"——decode 互拼、decode 与 prefill
chunk 混拼、chunk 之间不拼、批成员只在列车边界变化。每实例状态机
（§3.2）：qp（FCFS prefill 队列）/active_decode（批成员表）/
pending_decode_ready（KV 就绪待加入）/in_flight_train（唯一在飞列车 +
train_id/membership_digest）；列车终点 = 下一个不可预测事件（队列头
prefill drain / 全部工作耗尽），默认不设 T_max。边界原子提交顺序：核验
digest → 推进冻结成员 token → 退出成员移除 → 推进 prefill chunk → 处理
drain/完成 → 合入 arrival → KV 就绪成员入批 → 冻结下一列车成员 → 发射。
决策边界仍是四类 reason（ARRIVAL/PREFILL_DRAIN/DECODE_COMPLETION/
REQUEST_COMPLETE），由列车 drain/exit 标记节点的 watch 驱动。

sh_3.0 特性保留（红线，策略公式/阈值/KV 语义/映射规则一律不动）：
  - sticky 亲和与批齐发-等完成骨架的列车化重构：原"批齐发-等完成"
    （_start_ready_iterations 的发射对象 = qp[0] prefill 整段 + 全部
    active_decode decode 整段，busy 等待）适配为列车发射（一趟列车 =
    队列头 chunk × 迭代 + 全部 active_decode 成员，等待语义由
    in_flight_train 替代 busy）；三段式准入的 sticky 判据
    （LOCAL/PARTIAL 驻留实例 sticky / REMOTE 边缘负载均衡 / 首请求
    非边缘）与亲和规则本身一行不动；
  - decode 固定 prefill 同实例（红线 #4：selected = state_index，
    decode_candidates = ()）；
  - task_load_snapshot 三分量：打分公式与阈值不动，仅物理折算随列车化
    重订（active 段"段在飞=全量剩余"假设作废 → decode_tokens_consumed
    闭式迭代级剩余量；在飞 chunk 负载 = 冻结列车账本）。

关感知口径（方案 §4.1 第 6 条 + 合同⑥）：策略输入全部来自 Python 账本——
task_load_snapshot 三分量（Roofline 估计服务时间；queued 分量逐行复用
:3612-3638；running/active 分量的进度输入按列车闭式账本折算）、HBM 可行
掩码、edge_free 掩码（:466-483）、KV 快照（KVCacheManager 只读复用，
红线 #6-#9）。不新增任何 C++ 状态读取。

与离线蓝图的刻意差异（real-online 语义，合同⑦）：
  - 计时/迭代粒度：离线直接 Roofline 时钟 + 逐 chunk 迭代 -> 在线真实完成
    事件 + 迭代列车构图（成员×迭代 span，weight_passes=迭代数）；
  - 完成顺序：真实完成 tick 决定，不要求与离线决策序列 exact（合同⑦
    real-online 只验不变量与差异可解释性）。
"""

import hashlib
import heapq
import json
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
    """在线实例账本（离线 _InstanceRuntime 的在线子集 + 拼 batch 列车
    状态机，§3.2）：qp = 已准入 prefill FCFS 队列（deque[runtime]），
    active_decode = decode 批成员表，pending_decode_ready = KV 就绪待加入
    下一列车的成员，in_flight_train = 唯一在飞列车（冻结成员快照）；
    busy 门语义 = "一个列车在飞"（in_flight_train 替代原 busy 的等待语义，
    2026-08-22 列车化重构），last_arrival_ns 供 ordering_key。"""

    __slots__ = ("index", "qp", "active_decode", "active_decode_lookup",
                 "pending_decode_ready", "in_flight_train", "finalized_trains",
                 "iteration_count", "train_seq", "last_arrival_ns")

    def __init__(self, *, index: int):
        self.index = index
        self.qp = deque()
        self.active_decode = []
        self.active_decode_lookup = set()
        # ---- 拼 batch 列车账本（2026-08-22） ----
        self.pending_decode_ready = []   # KV 就绪待加入下一列车的成员
        self.in_flight_train = None      # 唯一在飞列车（冻结成员快照）
        self.finalized_trains = []       # 已核销列车（待收后续跨交付信号）
        self.iteration_count = 0         # 已完成迭代数（列车核销时闭式推进）
        self.train_seq = 0               # 列车序号（命名/审计用）
        self.last_arrival_ns = None


class _OnlineRequestRuntime:
    """在线请求运行账本（离线 _RequestRuntime 的在线子集 + plan dict 字段）。

    输入事实（request-neutral，来自 manifest）：history_tokens_before /
    prefill_context_tokens / final_context_tokens / prefill_length /
    decode_length / queue_index / session_id / turn_index。运行期事实由在线
    决策产出（语义与离线 _RequestRuntime 同名同义）。

    拼 batch 列车推进字段（2026-08-22）：remaining_chunks /
    prefill_tokens_completed / decode_tokens_consumed /
    current_decode_token / drain_block_ends——决策边界上闭式推进，
    余额与逐 token 精确值逐点一致（供 task_load_snapshot 折算与列车规划）。
    """

    __slots__ = (
        "request_id", "session_id", "turn_index", "queue_index",
        "prefill_length", "decode_length",
        "history_tokens_before", "prefill_tokens_to_process",
        "prefill_context_tokens", "final_context_tokens",
        "remaining_chunks", "prefill_tokens_completed",
        "decode_tokens_consumed", "current_decode_token",
        "prompt_tokens_processed", "drain_block_ends",
        "prefill_complete_ns", "decode_start_ns",
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
        "admitted", "completed", "completion_ns",
    )

    def __init__(self, record: dict, p_chunk: int = 512) -> None:
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
        # ---- 拼 batch 列车推进字段（闭式账本） ----
        self.remaining_chunks = math.ceil(
            self.prefill_tokens_to_process / p_chunk)
        self.prefill_tokens_completed = 0
        self.decode_tokens_consumed = 0   # 已物理完成 decode token 数
        self.current_decode_token = record["prefill_context_tokens"]
        self.prompt_tokens_processed = 0
        self.drain_block_ends = None      # drain 列车 barrier（joiner 触发门）
        self.prefill_complete_ns = None
        self.decode_start_ns = None
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
    """strategy 变体：sh_3.0 三段式准入 + decode 同实例 + 三态 KV（关感知）
    + 迭代列车拼 batch（2026-08-22）。

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
        self._decode_task_load_cache = {}

        self.runtimes = [
            _OnlineRequestRuntime(record, p_chunk)
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
        # 改法D：KV 账本纪元重试门（face capacity_epoch 的调度器全局版——
        # sh 系列 try_admit 的全部 False 判据是"全账本 HBM 可行性纯函数
        # ∧ 静态掩码"，实例账本（qp/busy/active_decode）只影响选哪个、
        # 不影响能不能）。_kv_ledger_epoch 在 9 个 KV 变更点后各 bump 一次；
        # _admit_attempt_epoch 记录各 pending 条目上次失败时的纪元，
        # 纪元未变则该条目本批跳过重试（重试必返同样的 False）。
        self._kv_ledger_epoch = 0
        self._admit_attempt_epoch = {}
        # 影子验证开关（SH_ADMIT_GATE_VERIFY=1）：门跳过的条目仍完整评估
        # 并断言必返 False——验证跑零收益、全检查；不设或非"1"则正常运行。
        self._admit_gate_verify = (
            os.environ.get("SH_ADMIT_GATE_VERIFY") == "1")
        self.completed_requests = 0
        # §7.3 ready frontier：非忙（无在飞列车）且有排队工作的实例集合。
        self._ready_frontier = set()
        # 拼 batch 列车台账（§7.3 不变量断言输入）：每次列车发射一行，
        # 由 online_service 落 bridge 目录 train_ledger.jsonl（审计产物）。
        self.train_ledger_rows = []
        # T_max 列车长度上限(§3.2.8/§7.4 治理旋钮 + A2 逐迭代 oracle):
        # 交付默认 = 8(2026-08-22 §7.4 A2 对拍裁决,sh_1.0 母本统一);
        # 0 = 不设限(oracle/灵敏度复跑用)。
        self._train_max_iter = int(
            os.environ.get("SH_TRAIN_MAX_ITER", "8") or 0)
        self._train_instance_index = {}

    # ------------------------------------------------------------- 策略 --

    def run_variant_policy(self, delta) -> None:
        """决策顺序逐行对应离线事件循环：completion 批（priority 0，含
        completion_order 处理）先于 arrival 批（priority 1），最后
        admit + 列车发射（:4060-4062）。

        拼 batch 列车账本（§3.2 边界原子提交顺序）：先核销已完成列车
        （核验 train_id + membership_digest → 冻结成员推进 token → 退出
        成员移出 active_decode → 推进 prefill chunk），再处理 drain/
        完成/到达，最后冻结并发射各空闲实例的下一列车。"""
        tick = delta["tick"]
        # ---- completion 批（offline: face_scheduler.py）----
        drained = []
        completed_now = []
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
                completed_now.append(request_id)
            elif stage == STAGE_REQUEST:
                # REQUEST_COMPLETE 与 DECODE_COMPLETION 同 tick 交付；完成
                # 处理在 _complete_requests（下）一并处理。
                continue
            else:
                raise ValueError(
                    "unknown completion stage {!r} for request {!r}".format(
                        stage, request_id))
        self._finalize_completed_trains(
            drained, completed_now, sentinel_trains, tick)
        for request_id in drained:
            self._on_prefill_drain(request_id, tick)
        if completed_now:
            self._complete_requests(completed_now, tick)
        # ---- arrival 批（offline: face_scheduler.py）----
        for arrival in delta["arrivals"]:
            self._push_arrival(arrival, tick)
        self._drain_arrival_heap(tick)
        # ---- 准入/列车发射 pass（offline: face_scheduler.py）----
        self._admit_pass(tick)

    # ------------------------------------------------------ 列车账本 --

    def _finalize_completed_trains(self, drained, completed_now,
                                   sentinel_trains, tick: int) -> None:
        """核销本交付中标记 watch 已 fire 的列车（§3.2 原子提交的前半；
        照母本 sh_1.0 定型版移植，账本对象换为本仓 runtime）。

        drain/exit 标记节点是列车体后的最末真实节点，任一标记 watch fire
        即列车物理主体完成。同一列车不同成员的标记 watch 可能跨 tick
        fire、事件拆到不同交付——首个信号执行核销（账本推进恰一次），
        后续信号只清已核销列车的 pending 集合（幂等）；信号不属于任何
        在飞/已核销列车即陈旧完成错配 fail-closed。推进量全部闭式
        （每成员 participation 次 token，无逐 token 循环）。"""
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
                    # 该列车首个信号到达：核销（账本推进恰一次），残余
                    # 信号（drain/exit 标记跨 tick fire）登记待收。
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
                    state.iteration_count += iterations
                    state.in_flight_train = None
                    state.finalized_trains.append({
                        "train_id": train["train_id"],
                        "pending": (train["signal_set"] - inflight_hits),
                    })
                    consumed |= inflight_hits
                    pending_signals -= inflight_hits
                    self._refresh_frontier(state)
            # 已核销列车的后续信号对账（幂等清 pending，清空即出列表）。
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

    def _plan_train(self, state):
        """冻结实例的下一列车成员快照（§3.2 构造规则；照母本移植）。

        列车终点 = 下一个不可预测事件之前的最后一个完整迭代：队列头
        prefill 的 drain 迭代（剩余 chunk 数，先验）或全部 decode 工作
        耗尽（无 prefill 工作时 = max 剩余 token；默认不设 T_max，§3.2.8）。
        成员退出不是列车边界（先验）：退出成员在列车内挂 exit 标记。
        每迭代至多 1 个 prefill chunk（FCFS 队列头）；chunk 之间不互拼。
        返回 None = 实例无工作。"""
        qp_head = None
        for runtime in state.qp:
            # drain 事件跨交付未达的头部（remaining_chunks 已在列车核销
            # 时清零，drain 决策事件尚在途中）不提供 chunk 工作；其后续
            # 请求的 chunk 物理上已可开始（头部 prefill 主体已完成）。
            if runtime.remaining_chunks > 0:
                qp_head = runtime
                break
        members = []
        for runtime in state.active_decode:
            remaining = (
                runtime.decode_length - runtime.decode_tokens_consumed)
            if remaining <= 0:
                raise RuntimeError(
                    "decode member {} has no remaining tokens".format(
                        runtime.request_id))
            members.append((runtime.request_id,
                            runtime.prefill_context_tokens,
                            runtime.decode_tokens_consumed, remaining))
        natural_iterations = None
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
        if (self._train_max_iter and iterations > self._train_max_iter):
            iterations = self._train_max_iter
            capped = True
        # span 展开（成员×迭代；KV 逐迭代 +1，退出截断，无 padding）。
        # 聚合对 span 求和与顺序无关，故按 [chunk 序列]+[成员连续段]
        # 平铺（总 span 数与旧 request-aggregated 同级，非新增热路径）。
        pass_spans: list[tuple[int, int]] = []
        chunk_records = []
        chunk_spans = []
        if qp_head is not None:
            work = qp_head.prefill_tokens_to_process
            completed = qp_head.prefill_tokens_completed
            history = qp_head.history_tokens_before
            for _ in range(iterations):
                chunk_tokens = min(self.p_chunk, work - completed)
                if chunk_tokens <= 0:
                    raise RuntimeError(
                        "prefill queue head {} ran out of work inside the "
                        "planned train".format(qp_head.request_id))
                span = (chunk_tokens, history + completed + chunk_tokens)
                pass_spans.append(span)
                chunk_spans.append(span)
                chunk_records.append((qp_head.request_id, chunk_tokens))
                completed += chunk_tokens
        member_parts = []
        exit_members = []
        for request_id, context, consumed, remaining in members:
            participation = min(remaining, iterations)
            pass_spans.extend(
                (1, context + consumed + step)
                for step in range(1, participation + 1))
            member_parts.append((request_id, participation))
            if participation >= remaining:
                exit_members.append(request_id)
        # 队列头在列车内完成其全部剩余 chunk（列车长度 = 头部剩余 chunk
        # 数）⇒ 列车终于头部 drain 迭代（drain 是先验已知的列车边界）。
        # T_max 截断时头部未必 drain —— 重算(先验边界)。
        drain_members = [qp_head.request_id] if (
            qp_head is not None and not capped) else []
        head_first_chunk = (
            qp_head is not None
            and qp_head.prefill_tokens_completed == 0)
        state.train_seq += 1
        train_id = "batch_train_i{}_{}".format(state.index, state.train_seq)
        sentinel = bool(capped and not drain_members and not exit_members)
        signal_set = set(exit_members) | set(drain_members)
        if sentinel:
            signal_set.add(train_id)
        snapshot = json.dumps(
            {
                "train_id": train_id,
                "iterations": iterations,
                "members": member_parts,
                "exits": exit_members,
                "drains": drain_members,
                "chunks": chunk_records,
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
            "signal_set": signal_set,
            "sentinel": sentinel,
            "drain_members": drain_members,
            "drain_set": set(drain_members),
            "prefill_chunk_tokens": chunk_records,
            "prefill_chunk_spans": chunk_spans,
            "head_request_id": qp_head.request_id if qp_head else None,
            "head_first_chunk": head_first_chunk,
            "pass_spans": pass_spans,
            "membership_digest": hashlib.sha256(
                snapshot.encode()).hexdigest(),
        }

    def _plan_and_emit_trains(self, tick: int) -> None:
        """为每个空闲且有工作的实例冻结并发射下一列车（§3.2 原子提交
        的后半：KV 就绪成员（pending_decode_ready，迁移随加入列车发射，
        物理先于列车体）进入 active_decode → 冻结成员 → 发射）。

        原"批齐发-等完成"骨架（_start_ready_iterations 发射部分）的列车
        化形态：一趟列车 = 队列头 chunk × 迭代 + 全部 active_decode 成员
        （与旧骨架 qp[0] prefill 整段 + 全部 decode 整段的批齐发口径一
        致），等待语义由 in_flight_train 替代 busy。busy 门 = 一个列车
        在飞（§3.2）：在飞实例跳过，不重复发射。"""
        for instance_index in sorted(self._ready_frontier):
            state = self.instances[instance_index]
            if state.in_flight_train is not None:
                continue  # busy 门：一个列车在飞
            if not (state.qp or state.active_decode or
                    state.pending_decode_ready):
                continue
            joiners = []
            if state.pending_decode_ready:
                joiners = list(state.pending_decode_ready)
                state.pending_decode_ready.clear()
                state.active_decode.extend(joiners)
                for runtime in joiners:
                    state.active_decode_lookup.add(runtime)
                    self._note_emitted(runtime.request_id, STAGE_DECODE)
                    self._ledger_issue(
                        runtime.request_id, tick, STAGE_DECODE, state.index)
            plan = self._plan_train(state)
            if plan is None:
                continue
            self._emit_train(state, plan, joiners, tick)

    def _emit_train(self, state, plan, joiner_runtimes, tick: int) -> None:
        """把冻结的列车计划交给构图器发射，注册 drain/exit 标记 watch，
        并挂起 in_flight_train（busy 门 = 一个列车在飞）。"""
        joiner_plans = []
        for runtime in joiner_runtimes:
            joiner_plan = runtime.plan_dict()
            joiner_plan["prefill_drain_block_ends"] = dict(
                runtime.drain_block_ends or {})
            joiner_plans.append(joiner_plan)
        stage = "decode" if (plan["members"] or joiner_runtimes) else "prefill"
        prefill_start_member = None
        first_chunk_member = None
        if plan.get("head_first_chunk"):
            # pstart 标记（prefill_start 锚点）与 partial 恢复 suffix 门
            # 挂接（first_chunk_member）同条件：队列头首 chunk 在本列车。
            member = {"request_id": plan["head_request_id"]}
            prefill_start_member = member
            first_chunk_member = member
        result = self.graph.emit_iteration_train({
            "train_id": plan["train_id"],
            "instance_index": state.index,
            "stage": stage,
            "joiners": joiner_plans,
            "pass_spans": plan["pass_spans"],
            "iterations": plan["iterations"],
            "sentinel": plan["sentinel"],
            "prefill_start_member": prefill_start_member,
            "first_chunk_member": first_chunk_member,
            "drain_members": [{"request_id": request_id}
                              for request_id in plan["drain_members"]],
            "exit_members": [{"request_id": request_id}
                             for request_id in plan["exit_members"]],
        })
        for request_id, members in result["drain_members"].items():
            self._batch["watches"].append({
                "request_id": request_id,
                "stage": STAGE_PREFILL,
                "generation": 0,
                "members": members,
                "statuses": ["Success", "Skipped"],
            })
            runtime = self.runtime_by_request_id[request_id]
            runtime.drain_block_ends = dict(result["block_ends"])
        for request_id, members in result["exit_members"].items():
            self._batch["watches"].append({
                "request_id": request_id,
                "stage": STAGE_DECODE,
                "generation": 1,
                "members": members,
                "statuses": ["Success", "Skipped"],
            })
        if plan["sentinel"]:
            # T_max 截断且无自然标记:哨兵 watch(train_id 批命名空间,
            # 固定 prefill 单事件通道——decode 通道会双事件 + 完成计账
            # 下溢)。
            self._batch["watches"].append({
                "request_id": plan["train_id"],
                "stage": STAGE_PREFILL,
                "generation": 0,
                "members": result["sentinel_members"],
                "statuses": ["Success", "Skipped"],
            })
        self._train_instance_index[plan["train_id"]] = state.index
        state.in_flight_train = plan
        self._refresh_frontier(state)
        self.train_ledger_rows.append({
            "train_id": plan["train_id"],
            "instance_index": state.index,
            "tick": tick,
            "iterations": plan["iterations"],
            "member_count": len(plan["members"]),
            "member_iterations": sum(
                participation for _, participation in plan["members"]),
            "joiners": [runtime.request_id for runtime in joiner_runtimes],
            "drains": list(plan["drain_members"]),
            "exits": list(plan["exit_members"]),
            "sentinel": plan["sentinel"],
            "prefill_chunks": len(plan["prefill_chunk_tokens"]),
            "pass_spans": len(plan["pass_spans"]),
        })

    def _refresh_frontier(self, state) -> None:
        """§7.3 ready frontier 增量维护（原 busy 判定的列车化等价）：
        实例非忙（无在飞列车）且有排队工作（qp / active_decode /
        pending_decode_ready 任一非空）即入 frontier，否则出。"""
        if (state.in_flight_train is None
                and (state.qp or state.active_decode
                     or state.pending_decode_ready)):
            self._ready_frontier.add(state.index)
        else:
            self._ready_frontier.discard(state.index)

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
        （decode 固定同实例 :3944-3949），逐行迁移；拼 batch 改造：decode
        段发射移至加入列车（_plan_and_emit_trains → emit_iteration_train），
        成员先进 pending_decode_ready（§3.2 KV 就绪栅栏），DECODE_
        COMPLETION watch 由列车 exit 标记承载。"""
        runtime = self.runtime_by_request_id[request_id]
        state = self.instances[runtime.prefill_instance_index]
        # drain 记实际 prefill 工作量（recompute 单口径 ==
        # request.prefill_length；离线 chunk 累计的整段等价）。
        runtime.prompt_tokens_processed = runtime.prefill_tokens_to_process
        runtime.remaining_chunks = 0
        # offline: :3989-3991 FCFS qp popleft（离线 iteration 串行化保证
        # drain 序 = FCFS 序）。在线列车构图下同实例列车头 chunk 的 drain
        # 物理完成可先于队列中更早请求的 drain 决策事件到达（信号拆交付
        # 时头部跳过规则允许后续请求的 chunk 先行）——账本适配 = 从 qp
        # 移除该已完成成员（母本同款；排队深度口径 remaining_chunks 求和
        # 不变，策略输入语义等价）。
        if runtime not in state.qp:
            raise RuntimeError("draining request is not in its prefill queue")
        state.qp.remove(runtime)
        runtime.prefill_complete_ns = tick
        self._refresh_frontier(state)
        # offline: face_scheduler.py（expand_prefill）
        runtime.prefill_evictions = self.kv_manager.expand_prefill(
            session_id=runtime.session_id,
            instance_index=state.index,
            context_tokens=runtime.prefill_context_tokens,
            trigger_request_id=runtime.request_id,
            reservation_request_id=runtime.request_id,
        )
        self._bump_kv_ledger_epoch()  # 改法D：KV 变更点 1/9（expand_prefill）
        # offline: face_scheduler.py（decode 固定 prefill 同实例）
        selected = state.index
        runtime.decode_instance_index = selected
        reservation_move_evictions = (
            self.kv_manager.move_request_capacity_reservation(
                request_id=runtime.request_id,
                target_instance_index=selected,
            )
        )
        self._bump_kv_ledger_epoch()  # 改法D：KV 变更点 2/9（move_request_capacity_reservation）
        (runtime.prefill_decode_transfer,
         decode_move_evictions) = self.kv_manager.move_prefill_to_decode(
            session_id=runtime.session_id,
            target_instance_index=selected,
            trigger_request_id=runtime.request_id,
            reservation_request_id=runtime.request_id,
        )
        self._bump_kv_ledger_epoch()  # 改法D：KV 变更点 3/9（move_prefill_to_decode）
        decode_growth_evictions = self.kv_manager.expand_decode(
            session_id=runtime.session_id,
            instance_index=selected,
            final_context_tokens=runtime.final_context_tokens,
            trigger_request_id=runtime.request_id,
            reservation_request_id=runtime.request_id,
        )
        self._bump_kv_ledger_epoch()  # 改法D：KV 变更点 4/9（expand_decode）
        runtime.decode_evictions = (
            reservation_move_evictions + decode_move_evictions
            + decode_growth_evictions)
        self.kv_manager.release_request_capacity_reservation(runtime.request_id)
        self._bump_kv_ledger_epoch()  # 改法D：KV 变更点 5/9（release_request_capacity_reservation）
        # 拼 batch 改造（§3.2 KV 就绪栅栏）：drain 决策（decode 实例选择
        # ＝固定同实例/KV 迁移规划）在此完成，成员进入 pending_decode_
        # ready，待加入 decode 实例的下一列车（迁移随加入列车发射，物理
        # 先于列车体；restore/迁移列车中途完成的也只能等下列车边界）。
        state.pending_decode_ready.append(runtime)
        self._refresh_frontier(state)
        # online: start_ready_iterations 的 decode 起始记账（:2927-2928）。
        runtime.decode_start_ns = tick
        # ledger（阶段 3 感知账本，查询/审计输入不进判据）。
        self._ledger_admit(
            runtime.request_id, tick,
            {"type": "active_decode", "instance_index": selected})
        # decode 段发射移至加入列车；drain 边界的 decode 决策记录在此。
        self._emit_join_decision(runtime, tick)

    def _complete_requests(self, completed_now, tick: int):
        """offline: :3986-4042 decode 完成分支 + completion_order 批
        （下一 turn arrival 排程 :4000-4006 + mark_complete :4016-4020 +
        enforce_reserve :4021-4031 + 快照 :4032-4042），段 3 发射 + 下一次
        session arrival 排程（离线 push_event 在线改为 future alarm，
        时刻 = 完成边界 tick + interval）。

        拼 batch 改造：active_decode 移除已移至 _finalize_completed_trains
        （退出迭代在列车内先验已知，物理完成时刻 = exit 标记节点完成时刻）。
        """
        # offline: :4008-4012 completion_order 排序
        completion_order = sorted(
            completed_now,
            key=lambda request_id: (
                self.runtime_by_request_id[request_id].session_id,
                request_id,
                self.runtime_by_request_id[request_id].queue_index,
            ),
        )
        # offline: :3986-3999 decode 完成分支（成员移除 + 完成时刻 +
        # 下一次 arrival 排程）。移除已移至列车核销；此处记完成事实与
        # future alarm。
        for request_id in completion_order:
            runtime = self.runtime_by_request_id[request_id]
            runtime.completed = True
            runtime.completion_ns = tick  # 基类侧字段由 log_decision 行携带
            self.completed_requests += 1
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
        # offline: :4016-4020 先全部 mark_complete
        # typed eviction：在线 runtime 是 manifest 派生账本（无 FaceRequest
        # 字段），完成请求自身的 next_trigger_type 经
        # config.request_queue[queue_index]（FaceTraceConfig 装载的
        # RequestSpec，与 manifest 同序）取回传给 mark_complete。
        for request_id in completion_order:
            runtime = self.runtime_by_request_id[request_id]
            self.kv_manager.mark_complete(
                runtime.session_id,
                tick,
                next_request_type=(
                    self.config.request_queue[
                        runtime.queue_index].next_trigger_type
                ),
            )
            self._bump_kv_ledger_epoch()  # 改法D：KV 变更点 6/9（mark_complete）
        # offline: :4021-4031 再全部 enforce_reserve
        for request_id in completion_order:
            runtime = self.runtime_by_request_id[request_id]
            if runtime.decode_instance_index is None:
                raise RuntimeError("completed request has no Decode instance")
            (runtime.completion_evictions,
             runtime.reserve_unmet_ranks) = self.kv_manager.enforce_reserve(
                instance_index=runtime.decode_instance_index,
                trigger_request_id=runtime.request_id,
            )
            self._bump_kv_ledger_epoch()  # 改法D：KV 变更点 7/9（enforce_reserve）
        # offline: :4032-4042 完成快照 + completion 批（合同①）：
        # completion_evictions + 下一 turn interval gate 依赖登记。
        for request_id in completion_order:
            runtime = self.runtime_by_request_id[request_id]
            snapshot = self.kv_manager.session_snapshot(runtime.session_id)
            runtime.kv_location_after_completion = snapshot.location
            runtime.kv_instance_after_completion = snapshot.instance_index
            self.graph.emit_completion_batch(runtime.plan_dict())
            # 阶段 3 感知账本：completed-unreconciled 核销（基类
            # _settle_completions 已在策略前完成；此处为决策日志）。
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

    def _bump_kv_ledger_epoch(self) -> None:
        """改法D：KV 账本纪元 +1。仅调度器的 9 个 KV 变更点后调用
        （见各调用处注释）；可行性读取与 _check_invariants 只读、不 bump。"""
        self._kv_ledger_epoch += 1

    def _admit_pass(self, tick: int) -> None:
        """offline: face_scheduler.py（admit_waiting_requests +
        start_ready_iterations 的排队/发射部分；直接 Roofline 计时删除）。
        拼 batch 改造：发射部分 = _plan_and_emit_trains（列车化）。"""
        self._retry = False
        self._admit_waiting_requests(tick)
        self._plan_and_emit_trains(tick)

    def _admit_waiting_requests(self, now_ns: int) -> None:
        """offline: face_scheduler.py admit_waiting_requests，
        逐行对应（blocked FIFO 重排语义保留）+ 改法D 纪元重试门。"""
        blocked = deque()
        while self.pending_admissions:
            runtime = self.pending_admissions.popleft()
            rid = runtime.request_id
            if (not self._admit_gate_verify
                    and self._admit_attempt_epoch.get(rid)
                    == self._kv_ledger_epoch):
                # 改法D：上次失败以来 KV 账本未变 → False 判据输入未变，
                # 重试必返同样的 False，跳过（FIFO 位置不变）。
                blocked.append(runtime)
                continue
            if (self._admit_gate_verify
                    and self._admit_attempt_epoch.get(rid)
                    == self._kv_ledger_epoch):
                # 影子断言：门判跳过 ≡ 重试必返 False。
                if self._try_admit_request(runtime, now_ns):
                    raise RuntimeError(
                        "admit gate equivalence violated: request {} was "
                        "admitted on a skipped retry (kv epoch {})".format(
                            rid, self._kv_ledger_epoch))
                self._admit_attempt_epoch[rid] = self._kv_ledger_epoch
                blocked.append(runtime)
                continue
            if not self._try_admit_request(runtime, now_ns):
                self._admit_attempt_epoch[rid] = self._kv_ledger_epoch
                blocked.append(runtime)
            else:
                self._admit_attempt_epoch.pop(rid, None)  # 成功即清除（有界）
        self.pending_admissions.extend(blocked)

    def _try_admit_request(self, runtime, now_ns: int) -> bool:
        """offline: face_scheduler.py try_admit_request，逐行
        迁移（三段式准入 + HBM 过滤 + reserve/prepare + qp 入队）。
        sticky 判据与亲和规则（分支 1/1a/2/3）一行不动（红线）。"""
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
        self._bump_kv_ledger_epoch()  # 改法D：KV 变更点 8/9（reserve_request_capacity）
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
        self._bump_kv_ledger_epoch()  # 改法D：KV 变更点 9/9（prepare_prefill）
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
        # §7.3 ready frontier：实例非忙（无在飞列车）且有排队工作（离线
        # start_ready_iterations :3835 的循环条件在在线的增量等价——遗漏
        # 此标记会使发射 pass 永远空转，全部 request 卡在 admitted 层）。
        self._refresh_frontier(self.instances[selected])
        # 准入动作发射（拼 batch 改造：prefill 主体移入实例迭代列车，在
        # _plan_and_emit_trains 处发射；此处只发到达 gates/历史迁移/逐出/
        # 屏障，物理串行化仍由图内 per-rank previous_id 链承载，strategy
        # 模式保持物理跨 request 链）。
        self._ledger_admit(
            runtime.request_id, now_ns,
            {"type": "prefill_qp", "instance_index": selected})
        self._emit_admission(runtime, now_ns)
        return True

    # ------------------------------------------------------------- 发射 --

    def _emit_admission(self, runtime, tick: int) -> None:
        """准入动作发射 + 决策/账本记录（拼 batch 改造，2026-08-22：
        PREFILL_DRAIN watch 不再在此注册——移至覆盖其最后 chunk 的列车
        drain 标记；原 _emit_prefill 的整段发射与 watch 部分删除）。"""
        self.graph.emit_admission_batch(runtime.plan_dict())
        self._note_emitted(runtime.request_id, STAGE_PREFILL)
        self._ledger_issue(runtime.request_id, tick, STAGE_PREFILL,
                           runtime.prefill_instance_index)
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

    def _emit_join_decision(self, runtime, tick: int) -> None:
        """drain 边界的 decode 决策记录（拼 batch 改造：decode 段发射
        移至加入列车，即 _plan_and_emit_trains → emit_iteration_train；
        DECODE_COMPLETION watch 由列车 exit 标记承载）。"""
        self._batch["assignments"].append({
            "request_id": runtime.request_id,
            "prefill_instance_index": runtime.prefill_instance_index,
            "decode_instance_index": runtime.decode_instance_index,
            "prefill_affinity_reason": runtime.prefill_affinity_reason,
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

    def _decode_task_load_ns_cached(self, *, instance_size,
                                    current_context_tokens, generated_tokens,
                                    average_decode_length,
                                    running_step_fraction_remaining):
        """改法A：estimate_decode_remaining_task_load_ns 的全参 key memo
        （与 _prefill_task_cache 同款）。hardware/model 为运行期不可变量，
        经绑定不入 key。"""
        key = (instance_size, current_context_tokens, generated_tokens,
               average_decode_length, running_step_fraction_remaining)
        cached = self._decode_task_load_cache.get(key)
        if cached is None:
            cached = estimate_decode_remaining_task_load_ns(
                self.hardware, self.model,
                instance_size=instance_size,
                current_context_tokens=current_context_tokens,
                generated_tokens=generated_tokens,
                average_decode_length=average_decode_length,
                running_step_fraction_remaining=running_step_fraction_remaining)
            self._decode_task_load_cache[key] = cached
        return cached

    def _task_load_snapshot(self, state, now_ns: int):
        """offline: face_scheduler.py task_load_snapshot 的在线
        子集。三分量口径（合同⑥），每个请求恰计一次（打分公式与阈值
        不动，红线；拼 batch 改造 2026-08-22 仅重订物理折算）：
          - queued_prefill：逐 chunk Roofline 求和（函数逐行复用
            :3612-3638；自 prefill_tokens_completed 闭式推进的剩余 chunk
            折算——列车核销时推进，决策边界上与逐 chunk 精确值逐点一致）；
          - running_prefill：在飞列车冻结的队列头 chunk 负载（列车账本
            prefill_chunk_spans；原"在飞段全量剩余 × fraction=1.0"的
            不可分假设作废，改为列车级冻结量）；
          - active_decode：estimate_decode_remaining_task_load_ns 逐请求
            （标定常数 average_decode_length；generated_tokens =
            decode_tokens_consumed 闭式迭代级剩余量、current_context_
            tokens = prefill_ctx + consumed——原 generated=0/fraction=1.0
            的"active 段不可分"假设作废；当前在飞迭代仍整计一次
            fraction=1.0，列车粒度下属保守方向）。"""
        instance_size = self.topology.instance(state.index).size
        # queued_prefill（offline: :3612-3638 逐行复用；剩余 chunk 自
        # prefill_tokens_completed 折算）+ 在飞列车 chunk 负载分离。
        queued_load_ns = 0
        for runtime in state.qp:
            remaining_tokens = (
                runtime.prefill_tokens_to_process
                - runtime.prefill_tokens_completed)
            processed = runtime.prefill_tokens_completed
            while remaining_tokens > 0:
                chunk_tokens = min(self.p_chunk, remaining_tokens)
                context_tokens = (
                    runtime.history_tokens_before + processed + chunk_tokens)
                queued_load_ns += self._prefill_chunk_task_load_ns(
                    instance_size=instance_size, chunk_tokens=chunk_tokens,
                    context_tokens=context_tokens)
                processed += chunk_tokens
                remaining_tokens -= chunk_tokens
        # running_prefill：在飞列车的冻结 chunk 负载（含在 qp 头部的
        # 剩余量中，恰计一次——自 queued 扣除）。
        running_prefill_load_ns = 0
        train = state.in_flight_train
        if train is not None:
            for chunk_tokens, context_tokens in train["prefill_chunk_spans"]:
                running_prefill_load_ns += self._prefill_chunk_task_load_ns(
                    instance_size=instance_size,
                    chunk_tokens=chunk_tokens,
                    context_tokens=context_tokens)
            queued_load_ns -= running_prefill_load_ns
        # active_decode（offline: :3665-3684；迭代级闭式剩余量折算）
        active_decode_load_ns = 0
        for runtime in state.active_decode:
            active_decode_load_ns += self._decode_task_load_ns_cached(
                instance_size=instance_size,
                current_context_tokens=(
                    runtime.prefill_context_tokens
                    + runtime.decode_tokens_consumed),
                generated_tokens=runtime.decode_tokens_consumed,
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
        审计（arrival heap / ready frontier / 列车账本全空）。"""
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
        if any(state.qp or state.active_decode
               or state.pending_decode_ready
               or state.in_flight_train is not None
               or state.finalized_trains
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
        if self._admit_attempt_epoch:
            raise RuntimeError(
                "strategy run ended with stale admit attempt epochs: "
                "{}".format(sorted(self._admit_attempt_epoch)[:5]))


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
