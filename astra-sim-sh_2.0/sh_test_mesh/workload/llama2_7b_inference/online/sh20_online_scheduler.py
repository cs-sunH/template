#!/usr/bin/env python3
"""sh20_online_scheduler.py -- sh_2.0 关感知策略调度器（strategy 模式，步骤 1-9）。

以已移除的离线 plan_face_requests（2026-08-21 离线 planner 清除批删去）为蓝本逐行迁移，保持决策
顺序逐行对应（每处迁移用 `# offline: face_scheduler.py:XXXX` 注释标注）。

拼 batch 改造（2026-08-22，设计文档《层次 B Continuous Batching 改造》§3.2
"迭代列车聚合发射"；母本 sh_1.0 定型版同构，策略特性保留）：层次 B 从
"请求级大段串行"重构为"实例迭代级列车"——decode 互拼、decode 与 prefill
chunk 混拼、chunk 之间不拼、批成员只在迭代（列车）边界变化。每实例状态机
（§3.2）：qp（FCFS prefill 队列）/active_decode（批成员表）/
pending_decode_ready（KV 就绪待加入）/in_flight_train（唯一在飞列车 +
train_id/membership_digest）；列车终点 = 下一个不可预测事件（队列头
prefill drain / 全部工作耗尽），默认不设 T_max。边界原子提交顺序：核验
digest → 推进冻结成员 token → 退出成员移除 → 推进 prefill chunk →
处理 drain/完成 → 合入 arrival → KV 就绪成员入批 → 冻结下一列车成员 →
发射。决策边界仍是四类 reason（ARRIVAL/PREFILL_DRAIN/DECODE_COMPLETION/
REQUEST_COMPLETE），由列车 drain/exit 标记节点的 watch 驱动。

sh_2.0 特性保留（拼 batch 改造红线，策略公式/阈值/KV 语义/映射规则不动）：
  - 三态 KV（local_hbm / partial_hbm_remote / remote_memory）与两阶段
    逐出：_on_prefill_drain 的 KV 调用链逐行保留；
  - partial 前缀两段式迁移：admission 发射 prefix 迁移 + suffix 恢复
    分支，首 chunk 所在列车按层段拆分（见构图器 emit_iteration_train）；
  - task-load 三分量：打分公式与阈值不动；active decode 分量废弃
    "整段不可分 fraction=1.0 + generated=0"的旧物理折算，改为按
    decode_tokens_consumed 的闭式迭代级剩余量（决策边界上与逐 token
    精确值逐点一致）；running/queued prefill 分量口径不变。

  离线事件循环                                   在线边界
  ----------------                               ----------------
  预置 arrival 堆（:3553-3558）                   ingress ARRIVAL 事件喂入同一
                                                 arrival heap
  iteration_complete 批（:3847-3966）              列车核销（_finalize_completed_
                                                 trains）→ PREFILL_DRAIN /
                                                 DECODE_COMPLETION / REQUEST_
                                                 COMPLETE 事件处理：
    prefill 收尾（:3864-3944）                     _on_prefill_drain（调用顺序
                                                 与离线逐行一致；drain 决策完成
                                                 后成员进入 pending_decode_ready，
                                                 迁移随加入列车发射）
    decode 完成（:3946-3959）                     _complete_requests 的 decode
                                                 记账段（active_decode 移除已
                                                 移至列车核销）
  completion 收尾（:3968-4002）                    _complete_requests 的完成
                                                 处理段：mark_complete →
                                                 enforce_reserve → 快照 +
                                                 completion 段发射 + 下一 turn
                                                 arrival 排程（alarm）
  arrival 批（:4004-4018）                         _on_arrival（arrival 落账 →
                                                 pending_admissions；2026-08-21
                                                 起无 truncate 通道，discarded 恒 0）
  admit_waiting_requests（:3764-3770）             _admit_pass 头部（各决策边界
                                                 触发，与离线 retry_admissions
                                                 门控同集合）
  start_ready_iterations（:3772-3829）              _plan_and_emit_trains：实例
                                                 空闲（无在飞列车）→ 冻结并发射
                                                 下一列车（prefill chunk 序列 +
                                                 decode 成员折叠为列车体）；
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
    迭代列车构图（拼 batch 改造起）；
  - task-load 三分量口径：running prefill 与 queued prefill 均按 512-chunk
    逐块求和（在飞段 fraction=1.0 全量剩余、恰计一次——上界近似，口径与
    离线同一估算器），active decode 分量按 decode_tokens_consumed 闭式
    折算剩余迭代（见 _task_load_snapshot；打分公式与参数不动）；
    完成时序由真实物理决定，不要求与离线 Roofline 时钟 exact；
  - 实例 busy 语义（拼 batch 改造）：busy 门 = "一个列车在飞"
    （in_flight_train）；列车核销（drain/exit 标记 watch fire）后实例方可
    冻结发射下一列车。
"""

import hashlib
import heapq
import json
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
    在线子集；iteration_* 计时字段删除——在线无预计算迭代时钟）+
    拼 batch 列车状态机（§3.2，2026-08-22）。busy 门语义 =
    "一个列车在飞"（in_flight_train）；iteration_count 为已完成迭代数
    （列车核销时闭式推进）。"""

    __slots__ = ("index", "qp", "active_decode", "last_arrival_ns",
                 "pending_decode_ready", "in_flight_train",
                 "finalized_trains", "iteration_count", "train_seq")

    def __init__(self, index: int) -> None:
        self.index = index
        self.qp = deque()          # prefill FCFS 队列（request_index）
        self.active_decode = []    # 已准入 decode 列表（request_index）
        self.last_arrival_ns = None
        # ---- 拼 batch 列车账本（2026-08-22） ----
        self.pending_decode_ready = []   # KV 就绪待加入下一列车的成员
        self.in_flight_train = None      # 唯一在飞列车（冻结成员快照）
        self.finalized_trains = []       # 已核销列车（待收后续跨交付信号）
        self.iteration_count = 0         # 已完成迭代数
        self.train_seq = 0               # 列车序号（命名/审计用）


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
        # 拼 batch 列车推进字段（2026-08-22；face_scheduler 的 _RequestRuntime
        # 为普通 dataclass，此处仅给实例追加属性——face_scheduler.py 本体零改动）：
        # decode_tokens_consumed/prefill_tokens_completed 在列车核销时闭式推进
        # （决策边界上与逐 token 精确值逐点一致），drain_block_ends = drain
        # 列车 end barrier（joiner 迁移触发门，列车发射时记录）。
        for runtime in self.runtimes:
            runtime.decode_tokens_consumed = 0
            runtime.prefill_tokens_completed = 0
            runtime.drain_block_ends = None
        # 拼 batch 列车台账（§7.3 不变量断言输入）：每次列车发射一行，
        # 由 online_service 落 bridge 目录 train_ledger.jsonl（审计产物）。
        self.train_ledger_rows = []
        # T_max 列车长度上限(§3.2.8/§7.4 治理旋钮 + A2 逐迭代 oracle):
        # 交付默认 = 8(2026-08-22 §7.4 A2 对拍裁决,sh_1.0 母本统一:
        # 无上限 TTFT -67.3%,16 仍 -19.1%,8 全指标 ≤1.3%;原则 1 优先
        # 于节点数)。0 = 不设限(oracle/灵敏度复跑用)。
        self._train_max_iter = int(
            os.environ.get("SH_TRAIN_MAX_ITER", "8") or 0)
        self._train_instance_index = {}

    # ------------------------------------------------------------- 策略 --

    def run_variant_policy(self, delta) -> None:
        """决策顺序逐行对应离线事件循环（:3832-4022）。

        拼 batch 列车账本（§3.2 边界原子提交顺序）：先核销已完成列车
        （核验 train_id + membership_digest → 冻结成员推进 token → 退出
        成员移出 active_decode → 推进 prefill chunk），再处理 drain/
        完成/到达/准入，最后冻结并发射各空闲实例的下一列车。同 tick
        顺序 = 离线批次序：prefill 收尾（:3864-3944）→ decode 完成
        （:3946-3959）→ completion 收尾（:3968-4002）→ arrival 批
        （:4004-4018）→ admit + 列车发射。"""
        tick = delta["tick"]
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
                # 处理在 _complete_requests（下）按 completion_order 一并处理。
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
        for arrival in delta["arrivals"]:
            self._push_arrival(arrival, tick)
        self._drain_arrival_heap(tick)
        self._admit_pass(tick)
        self._plan_and_emit_trains(tick)

    # ------------------------------------------------------ 列车账本 --

    def _finalize_completed_trains(self, drained, completed_now,
                                   sentinel_trains, tick: int) -> None:
        """核销本交付中标记 watch 已 fire 的列车（§3.2 原子提交的前半；
        母本 sh_1.0 定型版同构，成员键 = request_index）。

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
                runtime.prefill_instance_index, set()).add(
                self._runtime_index[request_id])
        for request_id in completed_now:
            runtime = self.runtime_by_request_id[request_id]
            signaled.setdefault(
                runtime.decode_instance_index, set()).add(
                self._runtime_index[request_id])
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
                    for request_index, participation in train["members"]:
                        runtime = self.runtimes[request_index]
                        runtime.decode_tokens_consumed += participation
                        runtime.current_decode_token += participation
                        # decode_steps_remaining 同步折算（face_scheduler
                        # 字段，保持账本自洽；不进策略判据）。
                        runtime.decode_steps_remaining -= participation
                    for request_index in train["exit_set"]:
                        if request_index not in state.active_decode:
                            raise RuntimeError(
                                "exiting member {} is not in the decode "
                                "batch".format(
                                    self.runtimes[request_index]
                                    .request.request_id))
                        state.active_decode.remove(request_index)
                    for (request_index,
                         chunk_tokens) in train["prefill_chunk_tokens"]:
                        runtime = self.runtimes[request_index]
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
                        sorted(self.runtimes[request_index].request.request_id
                               for request_index in unresolved),
                        instance_index))

    def _plan_train(self, state):
        """冻结实例的下一列车成员快照（§3.2 构造规则；母本同构）。

        列车终点 = 下一个不可预测事件之前的最后一个完整迭代：队列头
        prefill 的 drain 迭代（剩余 chunk 数，先验）或全部 decode 工作
        耗尽（无 prefill 工作时 = max 剩余 token；默认不设 T_max）。
        成员退出不是列车边界（先验）：退出成员在列车内挂 exit 标记。
        每迭代至多 1 个 prefill chunk（FCFS 队列头）；chunk 之间不互拼。
        返回 None = 实例无工作。

        sh_2.0 特性保留：partial 前缀两段式迁移的队列头（history_
        location_before.location == "partial_hbm_remote" 且首 chunk 未
        发射）——span 平铺改为 [首 chunk] + [各成员第 1 迭代 span] +
        [其余 chunk + 成员剩余 span]，首组进列车的 prefix/suffix 层段
        两段发射（见构图器）；聚合对 span 求和与顺序无关，总 span 数
        与旧 request-aggregated 同级（非新增热路径）。"""
        qp_head = None
        for request_index in state.qp:
            # drain 事件跨交付未达的头部（remaining_chunks 已在列车核销
            # 时清零，drain 决策事件尚在途中）不提供 chunk 工作；其后续
            # 请求的 chunk 物理上已可开始（头部 prefill 主体已完成）。
            if self.runtimes[request_index].remaining_chunks > 0:
                qp_head = request_index
                break
        members = []
        for request_index in state.active_decode:
            runtime = self.runtimes[request_index]
            remaining = (
                runtime.request.decode_length - runtime.decode_tokens_consumed)
            if remaining <= 0:
                raise RuntimeError(
                    "decode member {} has no remaining tokens".format(
                        runtime.request.request_id))
            members.append((request_index, runtime.prefill_context_tokens,
                            runtime.decode_tokens_consumed, remaining))
        if qp_head is None:
            if not members:
                return None
            iterations = max(remaining for _, _, _, remaining in members)
        else:
            iterations = self.runtimes[qp_head].remaining_chunks
            if iterations <= 0:
                raise RuntimeError(
                    "prefill queue head {} has no remaining chunks".format(
                        qp_head))
        capped = False
        if (self._train_max_iter and iterations > self._train_max_iter):
            iterations = self._train_max_iter
            capped = True
        partial = (
            qp_head is not None
            and self.runtimes[qp_head].prefill_tokens_completed == 0
            and self.runtimes[qp_head].history_location_before is not None
            and self.runtimes[qp_head].history_location_before.location
            == "partial_hbm_remote")
        member_parts = []
        exit_members = []
        member_first_spans = []
        member_rest_spans = []
        member_spans = []
        for request_index, context, consumed, remaining in members:
            participation = min(remaining, iterations)
            member_first_spans.append((1, context + consumed + 1))
            member_rest_spans.extend(
                (1, context + consumed + step)
                for step in range(2, participation + 1))
            member_spans.extend(
                (1, context + consumed + step)
                for step in range(1, participation + 1))
            member_parts.append((request_index, participation))
            if participation >= remaining:
                exit_members.append(request_index)
        pass_spans: list[tuple[int, int]] = []
        chunk_records = []
        if qp_head is not None:
            head_runtime = self.runtimes[qp_head]
            work = head_runtime.prefill_tokens_to_process
            completed = head_runtime.prefill_tokens_completed
            history = head_runtime.history_tokens_before
            for _ in range(iterations):
                chunk_tokens = min(self.p_chunk, work - completed)
                if chunk_tokens <= 0:
                    raise RuntimeError(
                        "prefill queue head {} ran out of work inside the "
                        "planned train".format(
                            head_runtime.request.request_id))
                pass_spans.append(
                    (chunk_tokens, history + completed + chunk_tokens))
                chunk_records.append((qp_head, chunk_tokens))
                completed += chunk_tokens
        if partial:
            pass_spans = (
                pass_spans[:1] + member_first_spans + pass_spans[1:]
                + member_rest_spans)
        else:
            pass_spans.extend(member_spans)
        # 队列头在列车内完成其全部剩余 chunk（列车长度 = 头部剩余 chunk
        # 数）⇒ 列车终于头部 drain 迭代（drain 是先验已知的列车边界）。
        # T_max 截断时头部未必 drain —— 重算(先验边界)。
        drain_members = [qp_head] if (
            qp_head is not None and not capped) else []
        head_first_chunk = (
            qp_head is not None
            and self.runtimes[qp_head].prefill_tokens_completed == 0)
        state.train_seq += 1
        train_id = "batch_train_i{}_{}".format(state.index, state.train_seq)
        snapshot = json.dumps(
            {
                "train_id": train_id,
                "iterations": iterations,
                "members": member_parts,
                "exits": exit_members,
                "drains": drain_members,
                "chunks": chunk_records,
                "partial": partial,
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
            "signal_set": (set(exit_members) | set(drain_members)
                           | ({train_id} if (capped and not exit_members
                                             and not drain_members)
                              else set())),
            "sentinel": bool(capped and not exit_members
                             and not drain_members),
            "drain_members": drain_members,
            "drain_set": set(drain_members),
            "prefill_chunk_tokens": chunk_records,
            "head_first_chunk": head_first_chunk,
            "pass_spans": pass_spans,
            "partial": partial,
            "partial_first_chunk_count": (
                1 + len(member_parts) if partial else None),
            "membership_digest": hashlib.sha256(
                snapshot.encode()).hexdigest(),
        }

    def _plan_and_emit_trains(self, tick: int) -> None:
        """为每个空闲且有工作的实例冻结并发射下一列车（§3.2 原子提交
        的后半：KV 就绪成员（pending_decode_ready，迁移随加入列车发射，
        物理先于列车体）进入 active_decode → 冻结成员 → 发射）。
        busy 门 = 一个列车在飞（§3.2）：在飞实例跳过，不重复发射。"""
        for state in self.instances:
            if state.in_flight_train is not None:
                continue  # busy 门：一个列车在飞
            joiners = []
            if state.pending_decode_ready:
                joiners = list(state.pending_decode_ready)
                state.pending_decode_ready.clear()
                state.active_decode.extend(joiners)
                for request_index in joiners:
                    request_id = (
                        self.runtimes[request_index].request.request_id)
                    self._note_emitted(request_id, STAGE_DECODE)
                    self._ledger_issue(
                        request_id, tick, STAGE_DECODE, state.index)
            plan = self._plan_train(state)
            if plan is None:
                continue
            self._emit_train(state, plan, joiners, tick)

    def _emit_train(self, state, plan, joiner_indexes, tick: int) -> None:
        """把冻结的列车计划交给构图器发射，注册 drain/exit 标记 watch，
        并挂起 in_flight_train（busy 门 = 一个列车在飞）。"""
        joiner_plans = []
        for request_index in joiner_indexes:
            runtime = self.runtimes[request_index]
            joiner_plan = self._plan_of(runtime)
            joiner_plan["prefill_drain_block_ends"] = dict(
                runtime.drain_block_ends or {})
            joiner_plans.append(joiner_plan)
        stage = "decode" if (plan["members"] or joiner_indexes) else "prefill"
        qp_head = state.qp[0] if state.qp else None
        prefill_start_member = None
        if qp_head is not None and plan.get("head_first_chunk"):
            prefill_start_member = {
                "request_id": self.runtimes[qp_head].request.request_id}
        train_plan = {
            "train_id": plan["train_id"],
            "instance_index": state.index,
            "stage": stage,
            "joiners": joiner_plans,
            "pass_spans": plan["pass_spans"],
            "iterations": plan["iterations"],
            "sentinel": plan["sentinel"],
            "prefill_start_member": prefill_start_member,
            "drain_members": [
                {"request_id": self.runtimes[request_index].request.request_id}
                for request_index in plan["drain_members"]],
            "exit_members": [
                {"request_id": self.runtimes[request_index].request.request_id}
                for request_index in plan["exit_members"]],
        }
        if plan.get("partial"):
            train_plan["partial_first_chunk_count"] = (
                plan["partial_first_chunk_count"])
        result = self.graph.emit_iteration_train(train_plan)
        for request_index in plan["drain_members"]:
            request_id = self.runtimes[request_index].request.request_id
            members = result["drain_members"][request_id]
            self._batch["watches"].append({
                "request_id": request_id,
                "stage": STAGE_PREFILL,
                "generation": 0,
                "members": members,
                "statuses": ["Success", "Skipped"],
            })
            self.runtimes[request_index].drain_block_ends = dict(
                result["block_ends"])
        for request_index in plan["exit_members"]:
            request_id = self.runtimes[request_index].request.request_id
            members = result["exit_members"][request_id]
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
        if plan["prefill_chunk_tokens"]:
            # offline:（prefill_start）首 chunk 物理开跑时刻 = 列车发射。
            head_runtime = self.runtimes[plan["prefill_chunk_tokens"][0][0]]
            if head_runtime.prefill_start_ns is None:
                head_runtime.prefill_start_ns = tick
        self._train_instance_index[plan["train_id"]] = state.index
        state.in_flight_train = plan
        self.train_ledger_rows.append({
            "train_id": plan["train_id"],
            "instance_index": state.index,
            "tick": tick,
            "iterations": plan["iterations"],
            "sentinel": plan["sentinel"],
            "member_count": len(plan["members"]),
            "member_iterations": sum(
                participation for _, participation in plan["members"]),
            "joiners": [self.runtimes[request_index].request.request_id
                        for request_index in joiner_indexes],
            "drains": [self.runtimes[request_index].request.request_id
                       for request_index in plan["drain_members"]],
            "exits": [self.runtimes[request_index].request.request_id
                      for request_index in plan["exit_members"]],
            "prefill_chunks": len(plan["prefill_chunk_tokens"]),
            "partial": bool(plan.get("partial")),
            "pass_spans": len(plan["pass_spans"]),
        })

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
        """离线 prefill 收尾分支（:3864-3944）：调用顺序与离线逐行一致。
        拼 batch 改造（2026-08-22）：drain 决策（decode 实例选择/KV 迁移
        规划，三态 KV/两阶段逐出调用链原样保留）在此完成，成员进入
        pending_decode_ready，待加入 decode 实例的下一列车（迁移随加入
        列车发射，物理先于列车体；restore/迁移列车中途完成的也只能等
        下列车边界，§3.2 KV 就绪栅栏）。"""
        index = self._runtime_index[request_id]
        runtime = self.runtimes[index]
        state = self.instances[runtime.prefill_instance_index]
        # drain 记实际 prefill 工作量（recompute 单口径 ==
        # request.prefill_length；列车核销时 prefill_tokens_completed 已
        # 闭式推进到全量，此处对齐 prompt 口径账本）。
        runtime.prompt_tokens_processed = runtime.prefill_tokens_to_process
        runtime.remaining_chunks = 0
        # offline: FCFS qp popleft（离线 iteration 串行化保证 drain 序 =
        # FCFS 序）。在线列车构图无 iteration 串行化，同实例多 request 的
        # chunk 物理上仍按 FCFS 序（列车只取队列头），但 drain（物理完成）
        # 可跨交付乱序——账本适配 = 从 qp 移除该已完成成员（排队深度口径
        # remaining_chunks 求和不变，策略输入语义等价；母本 sh_1.0 同款）。
        if index not in state.qp:
            raise RuntimeError("draining request is not in its prefill queue")
        state.qp.remove(index)
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
        # 拼 batch 改造（§3.2 KV 就绪栅栏）：decode 实例选择与 KV 迁移
        # 规划在 drain 时点完成（上方调用链原样），成员进入
        # pending_decode_ready，加入 decode 实例的下一列车
        # （_plan_and_emit_trains）。
        self.instances[selected].pending_decode_ready.append(index)
        # online: start_ready_iterations 的 decode 起始记账（:3946 前）。
        runtime.decode_start_ns = tick
        self._emit_join_decision(runtime, tick)

    def _emit_join_decision(self, runtime, tick: int) -> None:
        """drain 边界的 decode 决策记录（拼 batch 改造：decode 段发射
        移至加入列车，即 _plan_and_emit_trains → emit_iteration_train；
        DECODE_COMPLETION watch 由列车 exit 标记承载）。

        assignment 在 decode 决策后追加（sh_2.0 的 decode 实例在 prefill
        完成时才决策——GraphBatch 校验要求 prefill/decode 实例索引非负
        齐备）。阶段 3 感知账本：admitted 层排队类型更新（active_decode，
        face :754 同款；sh_2.0 无 waiting_decode 中间态——decode 在
        prefill 完成时即决策，发射随加入列车）+ issued 层写入移至加入
        列车处。查询/审计数据，不进策略判据。"""
        request_id = runtime.request.request_id
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
        self._batch["assignments"].append({
            "request_id": request_id,
            "prefill_instance_index": runtime.prefill_instance_index,
            "prefill_assignment_key": list(runtime.prefill_assignment_key),
            "prefill_affinity_reason": runtime.prefill_affinity_reason,
            "decode_instance_index": runtime.decode_instance_index,
        })
        self._ledger_admit(request_id, tick,
                           {"type": "active_decode",
                            "instance_index": runtime.decode_instance_index})

    def _complete_requests(self, completed_now, tick: int) -> None:
        """离线 decode 完成分支（:3946-3959）+ completion 收尾
        （:3968-4002）+ 下一 turn arrival 排程（:3960-3966）+ completion
        段发射。拼 batch 改造：active_decode 移除与 token 终值推进已移至
        _finalize_completed_trains（退出迭代在列车内先验已知，物理完成
        时刻 = exit 标记节点完成时刻）。"""
        # offline: completion_order 排序（同 tick 完成的确定性处理序）。
        completion_order = sorted(
            completed_now,
            key=lambda request_id: (
                self.runtime_by_request_id[request_id].request.session_id,
                request_id,
                self.runtime_by_request_id[request_id].request.queue_index,
            ),
        )
        # offline: :3946-3959 decode 完成分支（完成时刻 + 计数）。
        for request_id in completion_order:
            runtime = self.runtime_by_request_id[request_id]
            if runtime.decode_tokens_consumed != runtime.request.decode_length:
                raise RuntimeError(
                    "request {} exited its train before consuming its "
                    "decode tokens ({} != {})".format(
                        request_id, runtime.decode_tokens_consumed,
                        runtime.request.decode_length))
            runtime.completion_ns = tick
            self.completed_requests += 1
        # offline: :3968-4002 completion 收尾 + 下一 turn 排程 + 段发射。
        for request_id in completion_order:
            self._on_request_complete(request_id, tick)

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
        # 准入动作发射（拼 batch 改造：到达 gates/历史迁移（含 partial
        # 前缀两段式迁移分支）/逐出/屏障在准入时点发射；prefill 主体移入
        # 实例迭代列车，在 _plan_and_emit_trains 处发射。物理串行化由图内
        # per-rank previous_id 链承载，strategy 模式保持物理跨 request 链）。
        self._emit_admission(runtime, now_ns)
        return True

    def _emit_admission(self, runtime, tick: int) -> None:
        """准入动作发射 + 决策/账本记录（拼 batch 改造：PREFILL_DRAIN
        watch 不再在此注册——移至覆盖其最后 chunk 的列车 drain 标记）。"""
        request_id = runtime.request.request_id
        self.graph.emit_admission_batch(self._plan_of(runtime))
        self._batch["kv_actions"].extend(_kv_action_rows(runtime, "prefill"))
        self.log_decision({
            "kind": "prefill",
            "request_id": request_id,
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
        }, tick)
        # 阶段 3 感知账本：发射记录 + issued 层写入（face :777-779 同款；
        # 查询/审计数据，不进策略判据）。
        self._note_emitted(request_id, STAGE_PREFILL)
        self._ledger_issue(request_id, tick, STAGE_PREFILL,
                           runtime.prefill_instance_index)

    def _admit_pass(self, now_ns: int) -> None:
        """offline: face_scheduler.py（admit_waiting_requests，:3764-3770）。

        拼 batch 改造（2026-08-22）：start_ready_iterations（:3772-3829）的
        发射职责移至 _plan_and_emit_trains——prefill 主体不再是"实例空闲
        即整段发射"，而是折叠进实例迭代列车（busy 门 = 一个列车在飞）。"""
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
        三分量口径（每个请求恰计一次；打分公式与阈值不动）：
          - running_prefill：在飞列车队列头（busy 门 = 一个列车在飞）
            的全量剩余**逐 512-chunk 求和，恰计一次**（在线无 chunk 级
            进度事件，fraction=1.0 上界近似；与 sh_3.0 修复版及离线蓝本
            的逐 chunk 口径同形，时间折算归阶段 3 事件驱动账本）；
          - queued_prefill：未在飞请求逐 chunk 求和（在飞请求折 0，见
            _queued_prefill_task_load_ns）；
          - active_decode：estimate_decode_remaining_task_load_ns 逐请求。
            拼 batch 改造（2026-08-22）物理折算重订：旧口径"整段不可分
            （generated=0 + fraction=1.0，剩余恒为全量 ADL）"作废——
            按 decode_tokens_consumed 的闭式迭代级剩余量（决策边界上
            current_decode_token = prefill_context + consumed 与逐 token
            精确值逐点一致；估算器/公式/参数不动，generated_tokens 与
            current_context_tokens 换为列车闭式账本值）。"""
        instance_size = self.topology.instance(state.index).size

        running_prefill_load_ns = 0
        running_index = None
        if state.in_flight_train is not None:
            # busy 门 = 一个列车在飞；running prefill = 列车 chunk 工作
            # 的队列头（drain 事件跨交付未达的头部不提供 chunk 工作）。
            for request_index in state.qp:
                if self.runtimes[request_index].remaining_chunks > 0:
                    running_index = request_index
                    break
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
            # 闭式迭代级剩余量：generated = 已物理完成 decode token 数
            # （列车核销推进），current_context = prefill_context +
            # generated（KV 随迭代增长的同款口径）。fraction 保持 1.0
            # （决策边界恰在列车边界上，无半个在飞迭代）。
            generated_tokens = runtime.decode_tokens_consumed
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
        断言，不为凑断言新增状态结构。拼 batch 改造：列车状态机收尾
        断言（在飞列车/待加入成员/待收信号必须清空）。"""
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
        if any(state.qp or state.active_decode or state.pending_decode_ready
               or state.in_flight_train is not None
               or state.finalized_trains
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
