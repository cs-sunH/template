#!/usr/bin/env python3
"""sh10_online_scheduler.py -- Sh10OnlineScheduler(策略路径,关感知)。

方案 §4 步骤 1-9:以已移除的离线 plan_face_requests(2026-08-21 清除)为蓝本
逐行迁移(注释标注离线行号 `# offline: face_scheduler.py:XXXX`),保持决策
顺序与调用链逐行对应;离线迭代计时(start_ready_iterations 推 iteration_
complete 事件)删除,由 C++ 真实完成事件推进。

拼 batch 改造(2026-08-22,设计文档《层次 B Continuous Batching 改造》
§3.2"迭代列车聚合发射"):层次 B 从"请求级大段串行"重构为"实例迭代级列车"
——decode 互拼、decode 与 prefill chunk 混拼、chunk 之间不拼、批成员只在
迭代(列车)边界变化。每实例状态机(§3.2):qp(FCFS prefill 队列)/
active_decode(批成员表)/pending_decode_ready(KV 就绪待加入)/
in_flight_train(唯一在飞列车 + train_id/membership_digest);列车终点 =
下一个不可预测事件(队列头 prefill drain / 全部工作耗尽),默认不设
T_max(§3.2.8:加入延迟 ≤1 列车,由 §7.4 保真对拍量化治理)。边界原子
提交顺序:核验 digest → 推进冻结成员 token → 退出成员移除 → 推进
prefill chunk → 处理 drain/完成 → 合入 arrival → KV 就绪成员入批 →
冻结下一列车成员 → 发射。决策边界仍是四类 reason(ARRIVAL/
PREFILL_DRAIN/DECODE_COMPLETION/REQUEST_COMPLETE),由列车 drain/exit
标记节点的 watch 驱动。

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

import hashlib
import heapq
import json
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
from online.graph_batch_builder import (  # noqa: E402
    first_token_split_enabled,
)
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
                 decision_log_sink=None, train_ledger_sink=None,
                 profile_sink=None, mode: str = "strategy",
                 sensing: bool = False,
                 defensive_reply_cache: bool = False):
        super().__init__(
            manifest=manifest,
            config=config,
            digest_sink=digest_sink,
            mode=mode,
            sensing=sensing,
            decision_log_sink=decision_log_sink,
            profile_sink=profile_sink,
            defensive_reply_cache=defensive_reply_cache,
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
        # 改法D：KV 账本纪元重试门（face capacity_epoch 的调度器全局版——
        # sh 系列 try_admit 的全部 False 判据是"全账本 HBM 可行性纯函数
        # ∧ 静态掩码"，实例账本（qp/busy/active_decode）只影响选哪个、
        # 不影响能不能）。_kv_ledger_epoch 在 9 个 KV 变更点后各 bump 一次；
        # _admit_attempt_epoch 记录各 pending 条目上次失败时的纪元，
        # 纪元未变则该条目本批跳过重试（重试必返同样的 False）。
        self._kv_ledger_epoch = 0
        self._admit_attempt_epoch = {}
        # T_max 列车长度上限(§3.2.8/§7.4 治理旋钮 + A2 逐迭代 oracle):
        # SH_TRAIN_MAX_ITER 正整数 = 每列车至多 N 个迭代(截断列车无自然
        # drain/exit 标记时发射哨兵标记);0 = 不设限。
        # 交付默认 = 8(2026-08-22 §7.4 A2 对拍裁决:2s 窗双跑,无上限时
        # TTFT -67.3%/E2E -40.6%(加入延迟通道,§3.2.8 预判),T_max=16
        # 仍 TTFT -19.1%,T_max=8 全指标位移 ≤1.3%——按"固定 T_max 为
        # 使位移 ≤5% 的最大值"取 8;原则 1(保真)优先于节点数(§5 预算
        # 核对见交付报告)。env 覆盖保留(0=无限,oracle/灵敏度复跑用)。
        self._train_max_iter = int(
            os.environ.get("SH_TRAIN_MAX_ITER", "8") or 0)
        # train_id -> instance_index(哨兵事件路由)。
        self._train_instance_index = {}
        # 拼 batch 列车台账(§7.3 不变量断言输入):每次列车发射一行,
        # 由 online_service 落 bridge 目录 train_ledger.jsonl(审计产物)。
        # M3 流式落盘(2026-08-23):提供 train_ledger_sink 时行即写即
        # 弃,不驻留本列表;缺省 None = 兼容旧路径(行仍缓冲)。
        self.train_ledger_sink = train_ledger_sink
        self.train_ledger_rows = []
        # 影子验证开关（SH_ADMIT_GATE_VERIFY=1）：门跳过的条目仍完整评估
        # 并断言必返 False——验证跑零收益、全检查；不设或非"1"则正常运行。
        self._admit_gate_verify = (
            os.environ.get("SH_ADMIT_GATE_VERIFY") == "1")
        # WP9 首 token 首步批拆分（2026-08-26）：batch_train_<id>_first_step
        # 唤醒 id -> 实例索引。首步批（无请求级 watch）完成时其唤醒 watch
        # fire 经 PREFILL_DRAIN 通道送回，本表区分"自己发射的首步唤醒"与
        # 真哨兵信号；唤醒只作余量批的交付边界，不进任何决策/核销路径。
        self._pending_first_steps = {}

    # ------------------------------------------------------------- 策略 --

    def run_variant_policy(self, delta) -> None:
        tick = delta["tick"]
        # 同 tick 顺序(离线 :2970):completion(0)先于 arrival(1)。
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
                # REQUEST_COMPLETE 与 DECODE_COMPLETION 同 tick 交付;完成
                # 处理在 _complete_requests(下)一并处理。
                continue
            else:
                raise ValueError(
                    "unknown completion stage {!r}".format(stage))
        # WP9 首 token 拆分（2026-08-26）：batch_train_*_first_step 唤醒
        # 信号是"自己发射的首步批"的交付回声——无操作（不决策/不记账/
        # 不写 decision_log），从哨兵核销通道剥离；余量批在
        # _plan_and_emit_trains 的 busy 分支发射（本交付或任一后续交付，
        # 余量节点经依赖边排在首步节点之后，早发不改物理序）。首步批的
        # commit ack 走基类协议记账（ack_count/幂等门），变体侧零动作。
        sentinel_trains = [
            train_id for train_id in sentinel_trains
            if not self._consume_first_step_wakeup(train_id)
        ]
        # 拼 batch 列车账本(§3.2 边界原子提交顺序):先核销已完成列车
        # (核验 train_id + membership_digest → 冻结成员推进 token → 退出
        # 成员移出 active_decode → 推进 prefill chunk),再处理 drain/
        # 完成/到达,最后冻结并发射各空闲实例的下一列车。
        self._finalize_completed_trains(
            drained, completed_now, sentinel_trains, tick)
        for request_id in drained:
            self._on_prefill_drain(request_id, tick)
        if completed_now:
            self._complete_requests(completed_now, tick)
        for arrival in delta["arrivals"]:
            self._on_arrival(arrival, tick)
        # offline: :3134-3136 retry_admissions -> admit_waiting_requests +
        # start_ready_iterations(在线无 iteration 计时;准入重查保留)。
        self.admit_waiting_requests(tick)
        self._plan_and_emit_trains(tick)

    # ------------------------------------------------------ 列车账本 --

    def _finalize_completed_trains(self, drained, completed_now,
                                   sentinel_trains, tick: int) -> None:
        """核销本交付中标记 watch 已 fire 的列车(§3.2 原子提交的前半)。
        drain/exit 标记节点是列车体后的最末真实节点,任一标记 watch fire
        即列车物理主体完成。同一列车不同成员的标记 watch 可能跨 tick
        fire、事件拆到不同交付——首个信号执行核销(账本推进恰一次),
        后续信号只清已核销列车的 pending 集合(幂等);信号不属于任何
        在飞/已核销列车即陈旧完成错配 fail-closed。推进量全部闭式
        (每成员 participation 次 token,无逐 token 循环)。"""
        signaled = {}
        for request_id in drained:
            runtime = self._runtimes[request_id]
            signaled.setdefault(
                runtime.prefill_instance_index, set()).add(request_id)
        for request_id in completed_now:
            runtime = self._runtimes[request_id]
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
                    # 该列车首个信号到达:核销(账本推进恰一次),残余
                    # 信号(drain/exit 标记跨 tick fire)登记待收。
                    iterations = train["iterations"]
                    for request_id, participation in train["members"]:
                        runtime = self._runtimes[request_id]
                        runtime.decode_tokens_consumed += participation
                        runtime.current_decode_token += participation
                    for request_id in train["exit_set"]:
                        if request_id not in state.active_decode:
                            raise RuntimeError(
                                "exiting member {} is not in the decode "
                                "batch".format(request_id))
                        state.active_decode.remove(request_id)
                    for (request_id,
                         chunk_tokens) in train["prefill_chunk_tokens"]:
                        runtime = self._runtimes[request_id]
                        runtime.prefill_tokens_completed += chunk_tokens
                        runtime.remaining_chunks -= 1
                    state.iteration_count += iterations
                    state.in_flight_train = None
                    # M4 核销即删(2026-08-23):列车核销后其 train_id→
                    # 实例索引条目即死重(哨兵条目已在信号路由处弹出,
                    # 此 pop 对其为幂等 no-op;全仓 grep 证实核销后无读者)。
                    self._train_instance_index.pop(train["train_id"], None)
                    state.finalized_trains.append({
                        "train_id": train["train_id"],
                        "pending": (train["signal_set"] - inflight_hits),
                    })
                    consumed |= inflight_hits
                    pending_signals -= inflight_hits
            # 已核销列车的后续信号对账(幂等清 pending,清空即出列表)。
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
        """冻结实例的下一列车成员快照(§3.2 构造规则)。

        列车终点 = 下一个不可预测事件之前的最后一个完整迭代:队列头
        prefill 的 drain 迭代(剩余 chunk 数,先验)或全部 decode 工作
        耗尽(无 prefill 工作时 = max 剩余 token;默认不设 T_max,§3.2.8)。
        成员退出不是列车边界(先验):退出成员在列车内挂 exit 标记。
        每迭代至多 1 个 prefill chunk(FCFS 队列头);chunk 之间不互拼。
        返回 None = 实例无工作。"""
        qp_head = None
        for request_id in state.qp:
            # drain 事件跨交付未达的头部(remaining_chunks 已在列车核销
            # 时清零,drain 决策事件尚在途中)不提供 chunk 工作;其后续
            # 请求的 chunk 物理上已可开始(头部 prefill 主体已完成)。
            if self._runtimes[request_id].remaining_chunks > 0:
                qp_head = request_id
                break
        members = []
        for request_id in state.active_decode:
            runtime = self._runtimes[request_id]
            remaining = (
                runtime.request.decode_length - runtime.decode_tokens_consumed)
            if remaining <= 0:
                raise RuntimeError(
                    "decode member {} has no remaining tokens".format(
                        request_id))
            members.append((request_id, runtime.prefill_context_tokens,
                            runtime.decode_tokens_consumed, remaining))
        natural_iterations = None
        if qp_head is None:
            if not members:
                return None
            natural_iterations = max(
                remaining for _, _, _, remaining in members)
        else:
            head_runtime = self._runtimes[qp_head]
            natural_iterations = head_runtime.remaining_chunks
            if natural_iterations <= 0:
                raise RuntimeError(
                    "prefill queue head {} has no remaining chunks".format(
                        qp_head))
        iterations = natural_iterations
        capped = False
        if (self._train_max_iter and iterations > self._train_max_iter):
            iterations = self._train_max_iter
            capped = True
        # span 展开(成员×迭代;KV 逐迭代 +1,退出截断,无 padding)。
        # 聚合对 span 求和与顺序无关,故按 [chunk 序列]+[成员连续段]
        # 平铺(总 span 数与旧 request-aggregated 同级,非新增热路径)。
        pass_spans: list[tuple[int, int]] = []
        chunk_records = []
        if qp_head is not None:
            head_runtime = self._runtimes[qp_head]
            work = head_runtime.prefill_tokens_to_process
            completed = head_runtime.prefill_tokens_completed
            history = head_runtime.history_tokens_before
            for _ in range(iterations):
                chunk_tokens = min(self.p_chunk, work - completed)
                if chunk_tokens <= 0:
                    raise RuntimeError(
                        "prefill queue head {} ran out of work inside the "
                        "planned train".format(qp_head))
                pass_spans.append(
                    (chunk_tokens, history + completed + chunk_tokens))
                chunk_records.append((qp_head, chunk_tokens))
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
        # 队列头在列车内完成其全部剩余 chunk(列车长度 = 头部剩余 chunk
        # 数)⇒ 列车终于头部 drain 迭代(drain 是先验已知的列车边界)。
        # T_max 截断时头部未必 drain —— 重算。
        drain_members = [qp_head] if (
            qp_head is not None and not capped) else []
        head_first_chunk = (
            qp_head is not None
            and self._runtimes[qp_head].prefill_tokens_completed == 0)
        state.train_seq += 1
        train_id = "batch_train_i{}_{}".format(state.index, state.train_seq)
        signal_set = set(exit_members) | set(drain_members)
        if capped and not drain_members and not exit_members:
            signal_set.add(train_id)  # 哨兵:列车自身 id 即完成信号
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
            "drain_members": drain_members,
            "drain_set": set(drain_members),
            "prefill_chunk_tokens": chunk_records,
            "head_first_chunk": head_first_chunk,
            "capped": capped,
            "sentinel": bool(capped and not drain_members
                             and not exit_members),
            "pass_spans": pass_spans,
            "signal_set": signal_set,
            "membership_digest": hashlib.sha256(
                snapshot.encode()).hexdigest(),
        }

    def _consume_first_step_wakeup(self, train_id: str) -> bool:
        """识别并吞掉自己发射的首步批唤醒信号（WP9，2026-08-26）。

        首步批的唤醒 watch 用批命名空间 id（"<train_id>_first_step"，
        batch_train_ 前缀），fire 后与真哨兵同通道送达。返回 True = 这是
        首步唤醒（无操作，仅从哨兵核销列表剥离）；False = 非首步 id
        （真哨兵或未知 batch_train_ id，交回哨兵核销逻辑处理）。"""
        if train_id not in self._pending_first_steps:
            return False
        self._pending_first_steps.pop(train_id)
        return True

    def _plan_and_emit_trains(self, tick: int) -> None:
        """为每个空闲且有工作的实例冻结并发射下一列车(§3.2 原子提交
        的后半:KV 就绪成员(pending_decode_ready,迁移随加入列车发射,
        物理先于列车体)进入 active_decode → 冻结成员 → 发射)。
        busy 门 = 一个列车在飞(§3.2):在飞实例跳过,不重复发射;
        WP9 拆分列车的余量批在 busy 分支发射(首步唤醒到达后的首个
        决策边界)。"""
        for state in self.instances:
            if state.in_flight_train is not None:
                if state.first_step_remainder is not None:
                    # WP9:两段式发射的后半——余量体 + drain/exit/哨兵
                    # 标记 + end barrier。发射后 in_flight_train 语义恢复
                    # 整列口径(标记 watch 全部在本批注册)。
                    self._emit_train_remainder(state, tick)
                continue  # busy 门:一个列车在飞
            joiners = []
            if state.pending_decode_ready:
                joiners = list(state.pending_decode_ready)
                state.pending_decode_ready.clear()
                state.active_decode.extend(joiners)
                for request_id in joiners:
                    self._note_emitted(request_id, STAGE_DECODE)
                    self._ledger_issue(
                        request_id, tick, STAGE_DECODE, state.index)
            plan = self._plan_train(state)
            if plan is None:
                continue
            self._emit_train(state, plan, joiners, tick)

    def _first_token_split_spans(self, plan):
        """WP9 首/余量 span 组切分（机械操作，2026-08-26）。

        首步 = [prefill 队头第 1 个 chunk] + [各 decode 成员第 1 个
        span]；余量 = [剩余 chunk] + [各成员剩余 span]。聚合节点对 span
        求和与顺序无关，两组的激活/KV/AR 字节总量与整列一致；权重经
        weight_passes（1 + iterations-1）合计不变。"""
        spans = list(plan["pass_spans"])
        chunk_count = len(plan["prefill_chunk_tokens"])
        member_parts = plan["members"]
        if chunk_count + sum(p for _, p in member_parts) != len(spans):
            raise RuntimeError(
                "train span layout does not match the frozen plan")
        first_spans = spans[:1] if chunk_count else []
        rest_spans = list(spans[1:chunk_count]) if chunk_count else []
        offset = chunk_count
        for _, participation in member_parts:
            first_spans.append(spans[offset])
            rest_spans.extend(spans[offset + 1:offset + participation])
            offset += participation
        return first_spans, rest_spans

    def _first_token_plan(self, plan, joiner_ids):
        """WP9 首 token 观测计划（None = 开关关闭/无 debut，行为与拆分
        上线前逐字节一致）。

        debut 成员 = 本交付加入列车的 decode 成员（decode_tokens_consumed
        == 0，计划期可知）。多 token debut 挂独立 first_token 标记；
        decode_length=1 的 debut 其 exit 标记名附加 first_token 子串
        （同节点 code 4/8 双锚点，保证 first_token_ns == completion_ns）。
        iterations >= 2 时物理拆两批（首步批 + 余量批），否则仅做不拆车
        的标记增强。"""
        if not first_token_split_enabled():
            return None
        debut = [
            request_id for request_id in joiner_ids
            if self._runtimes[request_id].decode_tokens_consumed == 0
        ]
        if not debut:
            return None
        debut_marker_members = [
            {"request_id": request_id}
            for request_id in debut
            if self._runtimes[request_id].request.decode_length != 1
        ]
        debut_exit_first_token = [
            request_id for request_id in debut
            if self._runtimes[request_id].request.decode_length == 1
        ]
        if plan["iterations"] >= 2:
            first_spans, rest_spans = self._first_token_split_spans(plan)
            return {
                "split": True,
                "first_spans": first_spans,
                "rest_spans": rest_spans,
                "debut_marker_members": debut_marker_members,
                "debut_exit_first_token": debut_exit_first_token,
                "wakeup_id": "{}_first_step".format(plan["train_id"]),
            }
        return {
            "split": False,
            "debut_marker_members": debut_marker_members,
            "debut_exit_first_token": debut_exit_first_token,
        }

    def _emit_train(self, state, plan, joiner_ids, tick: int) -> None:
        """把冻结的列车计划交给构图器发射,注册 drain/exit 标记 watch,
        并挂起 in_flight_train(busy 门 = 一个列车在飞)。

        WP9（2026-08-26）：列车含 debut 成员且拆分开启、迭代数 >= 2 时
        两段式发射——本交付只发首步批（迁移/起始标记/首迭代体/first_
        token 标记/唤醒标记），drain/exit/哨兵 watch、块末账本与正常
        train_ledger 行移至余量批（_emit_train_remainder）；成员选择/
        排序/KV 动作/挂点语义全部不变。"""
        joiner_plans = []
        for request_id in joiner_ids:
            runtime = self._runtimes[request_id]
            joiner_plan = self._plan_dict(runtime)
            joiner_plan["prefill_instance_index"] = (
                runtime.prefill_instance_index)
            joiner_plan["decode_instance_index"] = (
                runtime.decode_instance_index)
            joiner_plan["decode_evictions"] = self._transfer_dicts(
                runtime.decode_evictions)
            joiner_plan["prefill_decode_transfer"] = (
                self._transfer_dict_or_none(runtime.prefill_decode_transfer))
            joiner_plan["prefill_drain_block_ends"] = dict(
                runtime.drain_block_ends or {})
            joiner_plans.append(joiner_plan)
        stage = "decode" if (plan["members"] or joiner_ids) else "prefill"
        qp_head = state.qp[0] if state.qp else None
        prefill_start_member = None
        if qp_head is not None and plan.get("head_first_chunk"):
            prefill_start_member = {"request_id": qp_head}
        train_plan = {
            "train_id": plan["train_id"],
            "instance_index": state.index,
            "stage": stage,
            "joiners": joiner_plans,
            "pass_spans": plan["pass_spans"],
            "iterations": plan["iterations"],
            "prefill_start_member": prefill_start_member,
            "sentinel": plan["sentinel"],
            "drain_members": [{"request_id": request_id}
                              for request_id in plan["drain_members"]],
            "exit_members": [{"request_id": request_id}
                             for request_id in plan["exit_members"]],
        }
        first_token = self._first_token_plan(plan, joiner_ids)
        if first_token is not None:
            train_plan["first_token"] = first_token
            if first_token["split"]:
                # WP9:joiner id 快照随 train_plan 走（余量批的台账行
                # 需要与首步行相同的 joiners 记录；joiner_plans 只在
                # 首步批消费）。
                train_plan["joiner_ids_of_record"] = list(joiner_ids)
                self._emit_train_first_step(state, plan, train_plan, tick)
                return
        result = self.graph.emit_iteration_train(train_plan)
        self._train_instance_index[plan["train_id"]] = state.index
        for request_id, members in result["drain_members"].items():
            self._batch["watches"].append({
                "request_id": request_id,
                "stage": STAGE_PREFILL,
                "generation": 0,
                "members": members,
                "statuses": ["Success", "Skipped"],
            })
            runtime = self._runtimes[request_id]
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
            self._batch["watches"].append({
                "request_id": plan["train_id"],
                "stage": STAGE_PREFILL,   # 固定 prefill:单事件通道
                "generation": 0,
                "members": result["sentinel_members"],
                "statuses": ["Success", "Skipped"],
            })
        state.in_flight_train = plan
        self._emit_train_ledger_row(state, plan, joiner_ids, tick)

    def _emit_train_ledger_row(self, state, plan, joiner_ids, tick: int,
                               first_step: bool = False) -> None:
        """train_ledger 行（M3 流式落盘）。WP9：拆分列车的首步批先写
        first_step=True 行（ON/OFF 对拍剥离标记），余量批再写正常行
        （与整列发射的行同构）。"""
        ledger_row = {
            "train_id": plan["train_id"],
            "instance_index": state.index,
            "tick": tick,
            "iterations": plan["iterations"],
            "member_count": len(plan["members"]),
            "member_iterations": sum(
                participation for _, participation in plan["members"]),
            "joiners": list(joiner_ids),
            "drains": list(plan["drain_members"]),
            "exits": list(plan["exit_members"]),
            "sentinel": plan["sentinel"],
            "prefill_chunks": len(plan["prefill_chunk_tokens"]),
            "pass_spans": len(plan["pass_spans"]),
        }
        if first_step:
            ledger_row["first_step"] = True
        # M3 流式落盘:提供 train_ledger_sink 时行即写即弃;缺省缓冲。
        if self.train_ledger_sink is not None:
            self.train_ledger_sink(ledger_row)
        else:
            self.train_ledger_rows.append(ledger_row)

    def _emit_train_first_step(self, state, plan, train_plan,
                               tick: int) -> None:
        """WP9 首步批发射（两段式前半，2026-08-26）：构图 + 唤醒 watch
        注册 + busy 门挂起 + first_step 台账行。drain/exit/哨兵 watch、
        drain_block_ends 与正常台账行全部移至余量批。"""
        first_token = train_plan["first_token"]
        result = self.graph.emit_train_first_step(train_plan)
        self._batch["watches"].append({
            # 批命名空间唤醒 watch（哨兵同款单事件通道）：首步批完成即
            # 交付余量批；不挂任何请求，fire 事件在 run_variant_policy
            # 的 _consume_first_step_wakeup 处无操作剥离。
            "request_id": first_token["wakeup_id"],
            "stage": STAGE_PREFILL,
            "generation": 0,
            "members": result["wakeup_members"],
            "statuses": ["Success", "Skipped"],
        })
        self._train_instance_index[plan["train_id"]] = state.index
        state.in_flight_train = plan
        state.first_step_remainder = train_plan
        self._pending_first_steps[first_token["wakeup_id"]] = state.index
        self._emit_train_ledger_row(
            state, plan, train_plan["joiner_ids_of_record"],
            tick, first_step=True)

    def _emit_train_remainder(self, state, tick: int) -> None:
        """WP9 余量批发射（两段式后半，2026-08-26）：余量体 + drain/
        exit/哨兵标记 + end barrier，随后注册 drain/exit/哨兵 watch、
        drain_block_ends 账本与正常台账行——与整列发射的后半完全同构。"""
        train_plan = state.first_step_remainder
        state.first_step_remainder = None
        plan = state.in_flight_train
        result = self.graph.emit_train_remainder(train_plan)
        for request_id, members in result["drain_members"].items():
            self._batch["watches"].append({
                "request_id": request_id,
                "stage": STAGE_PREFILL,
                "generation": 0,
                "members": members,
                "statuses": ["Success", "Skipped"],
            })
            runtime = self._runtimes[request_id]
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
            self._batch["watches"].append({
                "request_id": plan["train_id"],
                "stage": STAGE_PREFILL,   # 固定 prefill:单事件通道
                "generation": 0,
                "members": result["sentinel_members"],
                "statuses": ["Success", "Skipped"],
            })
        self._emit_train_ledger_row(
            state, plan, train_plan["joiner_ids_of_record"], tick)

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
        self._bump_kv_ledger_epoch()  # 改法D：KV 变更点 1/9
        # TOCTOU 修复（2026-08-23）：逐出补偿随决策时点同步执行（发射侧
        # 幂等双保险保留）。1/9~2/9 同调用内即发射，防御性同改（风格一致）。
        self.graph.sync_pending_history_after_evictions(admission_evictions)
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
        self._bump_kv_ledger_epoch()  # 改法D：KV 变更点 2/9
        self.graph.sync_pending_history_after_evictions(prepare_evictions)
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
        # 准入动作发射(拼 batch 改造:prefill 主体移入实例迭代列车,在
        # _plan_and_emit_trains 处发射;此处只发到达 gates/历史迁移/逐出/
        # 屏障,物理串行化仍由图内 per-rank previous_id 链承载,strategy
        # 模式保持物理跨 request 链)。
        self._emit_admission(runtime, now_ns)
        return True

    def admit_waiting_requests(self, now_ns: int) -> None:  # offline: :2903-2909
        blocked = deque()
        while self.pending_admissions:
            request_id = self.pending_admissions.popleft()
            if (self._admit_attempt_epoch.get(request_id)
                    == self._kv_ledger_epoch):
                if not self._admit_gate_verify:
                    # 改法D：上次失败以来 KV 账本未变 → False 判据输入未变，
                    # 重试必返同样的 False，跳过（FIFO 位置不变）。
                    blocked.append(request_id)
                    continue
                # 影子断言：门判跳过 ≡ 重试必返 False。
                if self.try_admit_request(request_id, now_ns):
                    raise RuntimeError(
                        "admit gate equivalence violated: request {} was "
                        "admitted on a skipped retry (kv epoch {})".format(
                            request_id, self._kv_ledger_epoch))
                self._admit_attempt_epoch[request_id] = self._kv_ledger_epoch
                blocked.append(request_id)
                continue
            if not self.try_admit_request(request_id, now_ns):
                self._admit_attempt_epoch[request_id] = self._kv_ledger_epoch
                blocked.append(request_id)
            else:
                self._admit_attempt_epoch.pop(request_id, None)  # 成功即清除（有界）
        self.pending_admissions.extend(blocked)

    def _bump_kv_ledger_epoch(self) -> None:
        """改法D：KV 账本纪元 +1。仅调度器的 9 个 KV 变更点后调用
        （见各调用处注释）；可行性读取与 _check_invariants 只读、不 bump。"""
        self._kv_ledger_epoch += 1

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
        self._bump_kv_ledger_epoch()  # 改法D：KV 变更点 3/9
        # TOCTOU 修复（2026-08-23）：3/9~6/9 是 drain 竞争主窗口——账本在
        # 决策时同步逐出，补偿不能再等下一趟列车的物理发射。
        self.graph.sync_pending_history_after_evictions(
            runtime.prefill_evictions)
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
        self._bump_kv_ledger_epoch()  # 改法D：KV 变更点 4/9
        self.graph.sync_pending_history_after_evictions(
            reservation_move_evictions)
        (runtime.prefill_decode_transfer, decode_move_evictions) = (
            self.kv_manager.move_prefill_to_decode(
                session_id=runtime.request.session_id,
                target_instance_index=selected,
                trigger_request_id=request_id,
                reservation_request_id=request_id,
                now_ns=tick,
            )
        )
        self._bump_kv_ledger_epoch()  # 改法D：KV 变更点 5/9
        self.graph.sync_pending_history_after_evictions(decode_move_evictions)
        decode_growth_evictions = self.kv_manager.expand_decode(
            session_id=runtime.request.session_id,
            instance_index=selected,
            final_context_tokens=runtime.final_context_tokens,
            trigger_request_id=request_id,
            reservation_request_id=request_id,
            now_ns=tick,
        )
        self._bump_kv_ledger_epoch()  # 改法D：KV 变更点 6/9
        self.graph.sync_pending_history_after_evictions(
            decode_growth_evictions)
        runtime.decode_evictions = (
            reservation_move_evictions + decode_move_evictions
            + decode_growth_evictions)
        runtime.kv_allocation = self.kv_manager.allocation_for_session(
            session_id=runtime.request.session_id,
            request_id=request_id,
        )
        self.kv_manager.release_request_capacity_reservation(request_id)
        self._bump_kv_ledger_epoch()  # 改法D：KV 变更点 7/9
        # 拼 batch 改造(§3.2 KV 就绪栅栏):drain 决策(decode 实例选择/
        # KV 迁移规划)在此完成,成员进入 pending_decode_ready,待加入
        # decode 实例的下一列车(迁移随加入列车发射,物理先于列车体;
        # restore/迁移列车中途完成的也只能等下列车边界)。
        self.instances[selected].pending_decode_ready.append(request_id)
        # online: start_ready_iterations 的 decode 起始记账(:2927-2928)。
        runtime.decode_start_ns = tick
        self._emit_join_decision(runtime, tick)

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
            # 拼 batch 改造:active_decode 移除已移至 _finalize_completed_
            # trains(退出迭代在列车内先验已知,物理完成时刻 = exit 标记
            # 节点完成时刻)。
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
            self._bump_kv_ledger_epoch()  # 改法D：KV 变更点 8/9
        # offline: :3099-3110 再全部 enforce_reserve
        for request_id in completion_order:
            runtime = self._runtimes[request_id]
            (runtime.completion_evictions, runtime.reserve_unmet_ranks
             ) = self.kv_manager.enforce_reserve(
                instance_index=runtime.decode_instance_index,
                trigger_request_id=request_id,
                now_ns=tick,
            )
            self._bump_kv_ledger_epoch()  # 改法D：KV 变更点 9/9
            # TOCTOU 修复（2026-08-23）：同边界内 seg3 随后发射，防御性同改。
            self.graph.sync_pending_history_after_evictions(
                runtime.completion_evictions)
        # 完成快照(离线 :3111-3121)。
        for request_id in completion_order:
            runtime = self._runtimes[request_id]
            snapshot = self.kv_manager.session_snapshot(
                runtime.request.session_id)
            runtime.kv_location_after_completion = snapshot.location
            runtime.kv_instance_after_completion = snapshot.instance_index
            self._emit_segment3(runtime, tick)
            # M4 核销即删(2026-08-23):请求完成后其 KV 转移对象/准入
            # 负载快照等胖字段再无读者(逐出已随 completion 批发射进图、
            # 审计已随决策行落盘;下一 turn 是独立 runtime;runtimes 表
            # 运行全程存活,不置空会随完成请求数线性常驻)——置空即删。
            # (sh_1.0 无 sh_3.0 的 prefill_instance_loads 字段,余同母本。)
            runtime.history_evictions = ()
            runtime.prefill_evictions = ()
            runtime.decode_evictions = ()
            runtime.completion_evictions = ()
            runtime.history_transfer = None
            runtime.prefill_decode_transfer = None
            runtime.history_location_before = None
            runtime.drain_block_ends = None

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

    def _emit_admission(self, runtime, tick: int) -> None:
        """准入动作发射 + 决策/账本记录(拼 batch 改造:PREFILL_DRAIN
        watch 不再在此注册——移至覆盖其最后 chunk 的列车 drain 标记)。"""
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
        self.graph.emit_admission_batch(plan)
        self._batch["assignments"].append({
            "request_id": runtime.request.request_id,
            "prefill_instance_index": runtime.prefill_instance_index,
            # 准入时刻 decode 实例未知(离线同款:decode 在 prefill 完成时
            # 决策);占位 = prefill 实例,join 决策追加完整 assignment。
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

    def _emit_join_decision(self, runtime, tick: int) -> None:
        """drain 边界的 decode 决策记录(拼 batch 改造:decode 段发射
        移至加入列车,即 _plan_and_emit_trains → emit_iteration_train;
        DECODE_COMPLETION watch 由列车 exit 标记承载)。"""
        plan = self._plan_dict(runtime)
        plan["prefill_instance_index"] = runtime.prefill_instance_index
        plan["decode_instance_index"] = runtime.decode_instance_index
        plan["decode_evictions"] = self._transfer_dicts(
            runtime.decode_evictions)
        plan["prefill_decode_transfer"] = self._transfer_dict_or_none(
            runtime.prefill_decode_transfer)
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
        # decode 发射(note_emitted/issue)移至加入列车处;prefill 层的
        # unissue 已由基类 _settle_completions 在 drain 事件核销时完成。
        self._ledger_admit(runtime.request.request_id, tick, {
            "type": "active_decode",
            "instance_index": runtime.decode_instance_index})

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
        # M4 核销即删(2026-08-23,sh_1.0 策略差异位点):块末账本条目在
        # 上述 seg2 兜底回读(无 completion_evictions 时的 watch 成员来源)
        # 之后即死重——完成请求不会再有任何发射,当场弹出(下一 turn 是
        # 不同 request_id;builder 侧 emit_completion_batch 内弹出会截断
        # 本回读,故置于调度器侧回读之后)。
        self.graph._block_ends.pop(runtime.request.request_id, None)
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
        if any(state.qp or state.active_decode or state.pending_decode_ready
               or state.in_flight_train is not None
               for state in self.instances):
            raise RuntimeError("online run ended with non-idle instance state")
        # WP9:全部首步拆分必须已交付余量批、唤醒信号已核销。
        if self._pending_first_steps:
            raise RuntimeError(
                "online run ended with undelivered first-step wakeups: "
                "{}".format(sorted(self._pending_first_steps)[:5]))
        if any(state.first_step_remainder is not None
               for state in self.instances):
            raise RuntimeError(
                "online run ended with an undelivered train remainder")
        if self._admit_attempt_epoch:
            raise RuntimeError(
                "strategy run ended with stale admit attempt epochs: "
                "{}".format(sorted(self._admit_attempt_epoch)[:5]))


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
        # ---- 拼 batch 列车推进字段(2026-08-22;决策边界上闭式推进,
        # 余额与逐 token 精确值逐点一致,供 select_decode_instance 的
        # active_tokens / queue_snapshot 输入) ----
        self.decode_tokens_consumed = 0  # 已物理完成 decode token 数
        self.prefill_tokens_completed = 0  # 已物理完成 prefill token 数
        self.drain_block_ends = None     # drain 列车 barrier(joiner 触发门)
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
    """离线 _InstanceRuntime(:2707-2713)+ 拼 batch 列车状态机(§3.2)。
    busy 门语义 = "一个列车在飞"(in_flight_train);iteration_count 为
    已完成迭代数(列车核销时闭式推进)。"""

    def __init__(self, index):
        self.index = index
        self.qp = deque()
        self.active_decode = []
        self.last_arrival_ns = None
        # ---- 拼 batch 列车账本(2026-08-22) ----
        self.pending_decode_ready = []   # KV 就绪待加入下一列车的成员
        self.in_flight_train = None      # 唯一在飞列车(冻结成员快照)
        self.finalized_trains = []       # 已核销列车(待收后续跨交付信号)
        self.iteration_count = 0         # 已完成迭代数(列车核销时闭式推进)
        self.train_seq = 0               # 列车序号(命名/审计用)
        # ---- WP9 首步批拆分(2026-08-26):两段式发射的余量批挂起 ----
        self.first_step_remainder = None  # 待发射余量批 train_plan(None=无)
