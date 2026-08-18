#!/usr/bin/env python3
"""online_scheduler_base.py -- OnlineSchedulerBase(批次框架,不做策略判据)。

方案 §4 步骤 1-8 操作 1。固定阶段(子类只能通过 run_variant_policy 挂钩策略):

    on_decision_batch(delta):
        delivery sequence 门(§7.2 幂等 + 单调 fail-closed)
        -> 校验 schema -> 核销 completion -> 处理 arrivals
        -> run_variant_policy() -> build_graph_batch() -> 返回 GraphBatch dict

基类负责 request-neutral 的簿记:
  - delivery sequence 门(阶段 4 §7.2):Python 记录 last_applied_sequence;
    重复 delivery(seq == last_applied_sequence)经深度比对后返回上次 batch
    的 digest(幂等重放,不重新产生任何 assignment/KV action/图节点,状态
    零变更);跳号/重复旧 seq(seq != last_applied_sequence + 1)即
    fail-closed(C++ 单在途背压 + 原子文件发布下不可能出现,出现即协议
    违例);
  - 校验 C++ 请求 schema(delivery_sequence/tick/reasons/arrivals/completed_groups);
  - 核销 completion:completed_groups 里的每个 stage 从 in-flight 请求的
    待办 stage 集合中移除;REQUEST_COMPLETE(stage == "")把请求整体核销;
  - 处理 arrivals:到达的请求登记为 in-flight(prefill+decode 两段待办);
  - 批次累加器:本批次所有决策产出的 nodes/parent_edges/watches/
    assignments/kv_actions/future_alarms 汇集在 self._batch 里,
    build_graph_batch() 组装成 GraphBatch dict 并写 digest 行。
"""

import copy
import hashlib
import json
import time

SCHEMA_VERSION = 1  # 阶段 4 §7.1:StateDelta schema v1(online_contracts/state_delta_v1.md 为唯一权威)

# 完成边界 stage -> 语义(与 C++ build_request_json 的 completed_groups 一致)。
STAGE_PREFILL = "prefill"
STAGE_DECODE = "decode"
STAGE_REQUEST = ""  # REQUEST_COMPLETE: 该 request 全部阶段完成(收尾/核销边界)


class OnlineSchedulerBase:
    """批次框架基类。run_variant_policy 由变体(replay/strategy)实现。"""

    def __init__(
        self,
        *,
        manifest: dict,
        config,
        replay=None,
        digest_sink=None,
        mode: str = "replay",
        sensing: bool = False,
    ):
        self.mode = mode
        self.manifest = manifest
        self.config = config
        self.request_by_id = {
            record["request_id"]: record
            for record in manifest["requests"]
        }
        self.replay = replay
        self.digest_sink = digest_sink  # callable(dict) 或 None(不写 digest)
        # request-neutral 簿记。
        # pending fence 索引(阶段 4 §7.3):request_id -> set[str] 待办
        # stage(已完成 stage 从集合移除;REQUEST_COMPLETE 整体核销)。
        # 完成事件经该索引直接定位受影响 request/stage(O(1),非全量扫描)。
        self.in_flight = {}
        self.completed_request_ids = set()
        self.delivery_count = 0   # 已应用决策批次数(幂等重放不计)
        self.ack_count = 0        # 已收到 commit ack 数(结束时应 == delivery_count)
        # 阶段 4 §7.2:delivery sequence/ack/幂等。
        # 最近一次已应用交付的 delivery_sequence。C++ 从 0 起(0 = 首个
        # tick-end 交付纪元,阶段 3 存档协议,合同 §3.2 修正记录),故 -1 =
        # 尚未应用任何交付(首个合法 seq 为 0)。
        self.last_applied_sequence = -1
        # 最近一次已应用交付的 (seq, delta, batch) 副本(幂等重放的凭据;
        # 重复 delivery 深度比对 delta 后返回缓存 batch 的 digest)。
        self._delivery_reply_cache = None
        # 已处理的 ack delivery_sequence(去重;与协议层文件去重一致,防御性)。
        self._seen_ack_delivery_seqs = set()
        self._batch = None        # 本批次累加器(每次 on_decision_batch 重建)
        self.online_log_rows = []  # online_decision_log.jsonl 行(replay 模式)
        # ---------------------------------------------------------------- 阶段 3
        # 感知(方案 §6.1/§6.2;感知开关经显式 feature flag 进入,阶段 6 前默认关):
        #   --sensing(online_service.py)与 C++ --sensing-enabled 配对;开启后
        #   分层账本最小子集(admitted/committed/completed-unreconciled)与两层
        #   剩余负载查询生效。感知口径:决策账本由 C++ 真实执行事实驱动;感知
        #   数据(C++ injected-unfinished 摘要 / Python admitted-not-injected
        #   排队)是查询/审计输入,不进策略判据(红线 §0.4),故感知开关不改变
        #   决策序列。
        self.sensing_enabled = bool(sensing)
        # 分层 remaining-load 账本(contract ⑥ + 阶段 7 §10.1;remote FIFO /
        # local HBM 为显式"不适用"占位,见合同表,本仓不删除)。八层口径:
        #   admitted: Python 排队账本成员(qp / waiting_decode_admissions /
        #             active_decode),request_id -> {admitted_tick, queue}
        #   committed: GraphBatch 已提交(commit ack 后转移),request_id ->
        #             {first_commit_tick, delivery_seq, stages: [...]}
        #   issued: 已发射(injected)未完成——策略 emit 段时登记、completion
        #             核销时移除(基类 _settle_completions,策略完成处理前,
        #             保证边界查询时刻与 C++ injected-unfinished 摘要对齐),
        #             request_id -> {issued_tick, stage, instance_index}
        #   ready: 边界视图(不常驻,查询时由策略 override 推导):已准入且
        #             所在实例就绪(非忙)可服务的排队成员
        #   network pending/active: C++ 侧 injected-unfinished 摘要
        #             (ledger_summary,审计输入;Python 不建第二套账本)
        #   completed-unreconciled: REQUEST_COMPLETE 边界核销写入,request_id ->
        #             {completed_tick, admitted_tick, first_commit_tick, stages}
        self.ledger_admitted = {}
        self.ledger_committed = {}
        self.ledger_issued = {}
        self.ledger_completed_unreconciled = {}
        # 最近一次交付批次的 C++ injected-unfinished 摘要(ledger_summary;
        # 空列表 = 该批次无未完成注入节点或感知关闭)。
        self.last_injected_unfinished = []
        # 逐批次历史(供结束总账核对与差异报告):{delivery_sequence, tick,
        # injected_unfinished}。
        self.injected_unfinished_history = []
        # 本批次发射记录:delivery_seq -> {"tick": tick, "requests":
        # [(request_id, stage), ...]};commit ack 到达时做层转移的凭据。
        self._emitted_by_delivery = {}
        # 决策边界两层剩余负载查询快照(感知开启时记录,写
        # sensing_query_log.jsonl)。
        self.sensing_query_rows = []
        # 阶段 4 §7.3:每决策批扫描条目数 profile(验收:与总 request 数
        # 无关;full_scan_entries 恒为 0 = 不存在 O(总规模) 全量扫描)。
        # 行: {delivery_sequence, tick, scanned_entries, full_scan_entries}。
        self.profile_rows = []
        self._profile_batch = None
        # 阶段 5 §8.2:Python provisional KV 账本(commit ack 前不入账)。
        # kv_actions 在 build_graph_batch 时先入暂存区(每 delivery 一条,
        # 仅非空),C++ 端 GraphBatch 原子提交成功后发 commit ack,ack 到达
        # 才 finalize 转入 committed 层;提交失败 = C++ abort = Python 以非
        # 0 退出(fail-closed 首版,无补偿路径)。会话 KV 状态机的实际变更
        # 仍由策略链即时可见(session_kv_manager.py 红线只读不动);本账本
        # 是"动作先入暂存、ack 后确认"的记账凭据,结束审计断言暂存区为空
        # (每笔已确认)。
        self._provisional_kv_actions = {}   # delivery_seq -> kv_actions 深副本
        self._committed_kv_actions = {}     # delivery_seq -> 同(ack 后)
        # ---------------------------------------------------------------- 阶段 6
        # §9.1 分项计数器统一采集(Python 侧;写 online_stats.jsonl)。
        #   python_callback_count_by_reason: 按 reason 统计已应用交付所携带
        #     decision 事件数(每个 delta.reasons 条目计 1;合计 = 交付的决策
        #     事件总数)。每次 on_decision_batch 进入 = 一次 Python callback,
        #     其总数 = delivery_count(独立进程架构下无 engine-idle callback)。
        #   scheduler_self_ns_total: on_decision_batch 纯 Python 墙钟累计
        #     (含幂等重放路径;官方路径无重放,二者相等)。
        #   online_stats_rows: 每已应用交付一行 {delivery_sequence, tick,
        #     reasons, scheduler_self_ns, node_count}。
        self.python_callback_count_by_reason = {}
        self.scheduler_self_ns_total = 0
        self.online_stats_rows = []

    # ------------------------------------------------------------- 固定阶段 --

    def on_decision_batch(self, delta: dict) -> dict:
        """固定阶段:delivery sequence 门 -> 校验 schema -> 核销 completion
        -> 处理 arrivals -> run_variant_policy() -> build_graph_batch() ->
        返回 GraphBatch dict。

        幂等重放(§7.2):seq == last_applied_sequence 的重复 delivery 在
        sequence 门处直接返回上次 batch 的 digest,以下任何一步都不执行
        (不重新产生 assignment/KV action/图节点,状态零变更)。

        阶段 6 §9.1:scheduler_self_ns = 本函数 Python 墙钟耗时(两次返回
        路径都累计;官方路径无幂等重放,故累计 = 各已应用交付之和)。
        """
        t0 = time.monotonic_ns()
        self._validate_schema(delta)
        replayed = self._gate_delivery_sequence(delta)
        if replayed is not None:
            self.scheduler_self_ns_total += time.monotonic_ns() - t0
            return replayed
        self._record_ledger_summary(delta)
        self.delivery_count += 1
        self._start_batch(delta)
        self._settle_completions(delta)
        self._process_arrivals(delta)
        if self.sensing_enabled:
            # 决策边界两层剩余负载查询快照(策略运行前 = 边界视图;查询是
            # 审计输入,不进策略判据)。
            self._record_sensing_query(delta)
        self.run_variant_policy(delta)
        batch = self.build_graph_batch()
        # §7.2:缓存本次已应用交付(幂等重放凭据;副本防对端 setdefault
        # 污染缓存)。
        self._delivery_reply_cache = {
            "seq": delta["delivery_sequence"],
            "delta": copy.deepcopy(delta),
            "batch": copy.deepcopy(batch),
        }
        self.last_applied_sequence = delta["delivery_sequence"]
        self._record_online_stats(delta, batch, time.monotonic_ns() - t0)
        return batch

    def _record_online_stats(self, delta: dict, batch: dict,
                             scheduler_self_ns: int) -> None:
        """阶段 6 §9.1:已应用交付的 Python 侧分项统计(每交付一行 +
        reason 计数累计)。"""
        self.scheduler_self_ns_total += scheduler_self_ns
        for reason in delta["reasons"]:
            self.python_callback_count_by_reason[reason] = (
                self.python_callback_count_by_reason.get(reason, 0) + 1)
        self.online_stats_rows.append({
            "delivery_sequence": delta["delivery_sequence"],
            "tick": delta["tick"],
            "reasons": list(delta["reasons"]),
            "scheduler_self_ns": scheduler_self_ns,
            "node_count": len(batch["nodes"]),
        })

    def run_variant_policy(self, delta) -> None:  # noqa: D401 -- 变体挂钩
        raise NotImplementedError

    def build_graph_batch(self) -> dict:
        """组装本批次 GraphBatch dict(节点/边/watch/未来 alarm 来自累加器),
        顺带写 graph_batch_digests.jsonl 一行。

        节点/边由构图器(GraphBatchBuilder)经 begin_batch/_collect 写入
        其自身累加器(self.graph.batch);watch/assignment/kv_action/未来
        alarm 由变体策略写入本类的 self._batch。组装时两边合并。
        """
        delivery_sequence = self._batch["delivery_sequence"]
        graph_batch = self._graph_batch()
        batch = {
            "schema_version": SCHEMA_VERSION,
            "batch_id": delivery_sequence,
            "source_delivery_sequence": delivery_sequence,
            "nodes": graph_batch["nodes"],
            "parent_edges": graph_batch["parent_edges"],
            "watches": self._batch["watches"],
            "assignments": self._batch["assignments"],
            "kv_actions": self._batch["kv_actions"],
            "future_alarms": self._batch["future_alarms"],
            # 阶段 5 §8.2:touched ranks 集合(本批节点 rank 升序去重;零节点
            # 批 = []);C++ GraphBatchCommitter 校验与自身计算结果一致。
            "touched_ranks": sorted({
                int(node["rank"]) for node in graph_batch["nodes"]
            }),
        }
        # 阶段 5 §8.2:本批 kv_actions 先入 provisional 暂存区(非空才记账),
        # commit ack 到达后 finalize(见 on_commit_ack);幂等重放不经过这里
        # (sequence 门直接返回缓存 digest),不会重复入账。
        if self._batch["kv_actions"]:
            self._provisional_kv_actions[delivery_sequence] = copy.deepcopy(
                self._batch["kv_actions"])
        if self.digest_sink is not None:
            self.digest_sink(self._digest_row(batch))
        # 阶段 4 §7.3:本决策批扫描条目数入 profile。
        self.profile_rows.append({
            "delivery_sequence": delivery_sequence,
            "tick": self._batch["tick"],
            "scanned_entries": self._profile_batch["scanned_entries"],
            "full_scan_entries": self._profile_batch["full_scan_entries"],
        })
        return batch

    # ------------------------------------------------------------- 校验/簿记 --

    def _gate_delivery_sequence(self, delta: dict):
        """阶段 4 §7.2 delivery sequence 门。返回 None = 正常新交付,继续
        走完整管线;返回 dict = 幂等重放(上次 batch 的 digest 副本)。

        - 幂等:seq == last_applied_sequence -> 深度比对 delta 与缓存(同
          seq 不同内容 = 协议违例 fail-closed),一致则返回缓存 batch 的
          digest,不重新产生任何 assignment/KV action/图节点,状态零变更;
        - 单调:seq != last_applied_sequence + 1 -> fail-closed(C++ 单在途
          背压 + 原子文件发布下不可能跳号;重复旧 seq 亦属违例——本通道的
          重复只能是对同一交付的重试,即"上一次交付")。
        """
        seq = delta["delivery_sequence"]
        if seq == self.last_applied_sequence:
            cached = self._delivery_reply_cache
            if cached is None or cached["seq"] != seq:
                raise ValueError(
                    "duplicate delivery {} but applied-delivery cache "
                    "missing/inconsistent".format(seq))
            if delta != cached["delta"]:
                raise ValueError(
                    "duplicate delivery {} carries a different delta: "
                    "idempotent replay must byte-match the applied delivery"
                    .format(seq))
            return copy.deepcopy(cached["batch"])
        if seq != self.last_applied_sequence + 1:
            raise ValueError(
                "delivery sequence gap: got {} last_applied={} "
                "(monotonic +1 expected; duplicates replay only the last "
                "delivery)".format(seq, self.last_applied_sequence))
        return None

    # ------------------------------------------------------ §7.3 profile --

    def _profile_scan(self, count: int = 1) -> None:
        """记录本决策批经索引/队列直接访问的条目数(到期事件 + 受影响
        条目;非全量扫描)。验收:每批计数与总 request 数无关。"""
        self._profile_batch["scanned_entries"] += count

    def _profile_full_scan(self, count: int = 1) -> None:
        """记录本决策批的 O(总规模) 全量扫描条目数。§7.3 之后应恒为 0;
        任何 > 0 都意味着索引队列被绕过(profile 审计 fail-closed)。"""
        self._profile_batch["full_scan_entries"] += count

    def dump_profile(self, path: str) -> None:
        """把每决策批扫描条目数写为 profile.jsonl(阶段 4 §7.3 验收输入)。"""
        with open(path, "w", encoding="utf-8") as out:
            for row in self.profile_rows:
                out.write(json.dumps(row, sort_keys=True) + "\n")

    def _validate_schema(self, delta: dict) -> None:
        """schema v1 校验器(阶段 4 §7.1;online_contracts/state_delta_v1.md
        §6 的 7 条强制)。任何不满足即抛异常 -> BridgeServer 写 error
        response 并以非 0 退出(fail-closed,绝不静默降级)。"""
        if not isinstance(delta, dict):
            raise ValueError("delta must be a dict, got {!r}".format(type(delta)))
        if delta.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(
                "delta schema_version {!r} != {}".format(
                    delta.get("schema_version"), SCHEMA_VERSION))
        for field in ("delivery_sequence", "delivery_epoch", "tick",
                      "deferred_from_tick", "reasons", "arrivals",
                      "completed_groups", "completed_nodes", "retry_items",
                      "affected_ranks", "snapshot_handle"):
            if field not in delta:
                raise ValueError("delta missing field {!r}".format(field))
        if not isinstance(delta["reasons"], list):
            raise ValueError("delta reasons must be a list")
        if not isinstance(delta["arrivals"], list):
            raise ValueError("delta arrivals must be a list")
        if not isinstance(delta["completed_groups"], list):
            raise ValueError("delta completed_groups must be a list")
        # v1 新增字段类型 + 语义检查(§6 第 1/4/5/6/7 条)。
        if not isinstance(delta["completed_nodes"], list):
            raise ValueError("delta completed_nodes must be a list")
        if not isinstance(delta["retry_items"], list) or delta["retry_items"]:
            raise ValueError(
                "delta retry_items must be an empty list in v1, got {!r}"
                .format(delta["retry_items"]))
        if not isinstance(delta["affected_ranks"], list):
            raise ValueError("delta affected_ranks must be a list")
        affected = delta["affected_ranks"]
        if any(not isinstance(rank, int) or rank < 0 for rank in affected):
            raise ValueError("delta affected_ranks must be non-negative ints")
        if affected != sorted(set(affected)):
            raise ValueError(
                "delta affected_ranks must be sorted unique, got {!r}"
                .format(affected))
        if not isinstance(delta["snapshot_handle"], dict):
            raise ValueError("delta snapshot_handle must be a dict")
        handle = delta["snapshot_handle"]
        # §4 过期规则(v1 实例):句柄只在创建它的同一 delivery epoch/tick 内
        # 有效;跨 tick/epoch 使用即 fail-closed。
        if (handle.get("epoch") != delta["delivery_sequence"]
                or handle.get("tick") != delta["tick"]):
            raise ValueError(
                "snapshot_handle {} expired or inconsistent with "
                "delivery_sequence={} tick={}".format(
                    handle, delta["delivery_sequence"], delta["tick"]))
        # §3.2:delivery_epoch 与 delivery_sequence v1 恒等。
        if delta["delivery_epoch"] != delta["delivery_sequence"]:
            raise ValueError(
                "delivery_epoch {} != delivery_sequence {} (v1 forbids "
                "divergence)".format(delta["delivery_epoch"],
                                     delta["delivery_sequence"]))
        # §3.3:arrivals 内按 queue_index 冻结队列序升序(C++ 序列化器保证;
        # 校验器断言,防实现漂移)。queue_index 非负(未知 -1 防御性允许,
        # 正常路径不出现)。
        previous_queue_index = -1
        for arrival in delta["arrivals"]:
            if not isinstance(arrival, dict):
                raise ValueError("delta arrivals entries must be dicts")
            queue_index = arrival.get("queue_index", -1)
            if not isinstance(queue_index, int) or queue_index < -1:
                raise ValueError(
                    "arrival {!r} queue_index {!r} invalid".format(
                        arrival.get("request_id"), queue_index))
            if queue_index < previous_queue_index:
                raise ValueError(
                    "arrivals not in frozen queue order (queue_index {} "
                    "after {})".format(queue_index, previous_queue_index))
            if queue_index >= 0:
                previous_queue_index = queue_index
            if not isinstance(arrival.get("ingress_seq"), int):
                raise ValueError(
                    "arrival {!r} missing/odd ingress_seq".format(
                        arrival.get("request_id")))
        for fact in delta["completed_nodes"]:
            if not isinstance(fact, dict):
                raise ValueError("delta completed_nodes entries must be dicts")
            if not isinstance(fact.get("rank"), int):
                raise ValueError("completed_node fact missing rank")
        # 阶段 3 感知:ledger_summary 可选(C++ 总是序列化;缺省视为空)。
        if "ledger_summary" in delta and not isinstance(
                delta["ledger_summary"], dict):
            raise ValueError("delta ledger_summary must be a dict")

    def _record_ledger_summary(self, delta: dict) -> None:
        """阶段 3 感知:记录 C++ 交付的 injected-unfinished 摘要
        (ledger_summary.injected_unfinished)。查询/审计输入,不进策略判据;
        感知关闭时 C++ 序列化空数组,同样记录(供差异报告佐证一致性)。"""
        summary = delta.get("ledger_summary")
        if summary is None:
            summary = {}
        injected = summary.get("injected_unfinished", [])
        if not isinstance(injected, list):
            raise ValueError(
                "delta ledger_summary.injected_unfinished must be a list")
        self.last_injected_unfinished = injected
        self.injected_unfinished_history.append({
            "delivery_sequence": delta["delivery_sequence"],
            "tick": delta["tick"],
            "injected_unfinished": injected,
        })

    # ------------------------------------------------------- 感知账本(阶段 3) --

    def _ledger_admit(self, request_id: str, tick: int, queue: dict) -> None:
        """进入/更新 admitted 层(排队账本成员;contract ⑥:qp /
        waiting_decode_admissions / active_decode)。queue = 描述 dict,如
        {"type": "prefill_qp", "instance_index": 3}。"""
        if request_id in self.ledger_admitted:
            self.ledger_admitted[request_id]["queue"] = queue
            self.ledger_admitted[request_id]["queue_tick"] = tick
            return
        self.ledger_admitted[request_id] = {
            "admitted_tick": tick,
            "queue_tick": tick,
            "queue": queue,
        }

    def _note_emitted(self, request_id: str, stage: str) -> None:
        """本批次发射记录(commit ack 到达时做层转移的凭据)。"""
        self._emitted_by_delivery[self._batch["delivery_sequence"]][
            "requests"].append((request_id, stage))

    def _ledger_commit(self, request_id: str, tick: int,
                       delivery_seq: int, stage: str) -> None:
        """commit ack 后层转移:committed 记录该 request 的已提交阶段。"""
        entry = self.ledger_committed.setdefault(request_id, {
            "first_commit_tick": tick,
            "delivery_seq": delivery_seq,
            "stages": [],
        })
        if not any(item["stage"] == stage for item in entry["stages"]):
            entry["stages"].append({
                "stage": stage,
                "tick": tick,
                "delivery_seq": delivery_seq,
            })

    def _ledger_complete(self, request_id: str, tick: int) -> None:
        """REQUEST_COMPLETE 边界:核销进入 completed-unreconciled,移出
        admitted / committed。"""
        if request_id in self.ledger_completed_unreconciled:
            return  # 幂等(协议层已完成去重,防御性)
        admitted = self.ledger_admitted.get(request_id, {})
        committed = self.ledger_committed.get(request_id, {})
        self.ledger_completed_unreconciled[request_id] = {
            "completed_tick": tick,
            "admitted_tick": admitted.get("admitted_tick"),
            "first_commit_tick": committed.get("first_commit_tick"),
            "stages": list(committed.get("stages", [])),
        }
        self.ledger_admitted.pop(request_id, None)
        self.ledger_committed.pop(request_id, None)

    def _ledger_issue(self, request_id: str, tick: int, stage: str,
                      instance_index=None) -> None:
        """阶段 7 §10.1 issued 层写入:策略发射一个段时登记(已发射未完成)。
        同一 request 同一时刻至多一个段在飞(prefill 段在 drain 时移除、
        decode 段在其后才发射),单条目账本。completion 核销时由
        _ledger_unissue 移除(基类 _settle_completions,策略完成处理前)。"""
        self.ledger_issued[request_id] = {
            "issued_tick": tick,
            "stage": stage,
            "instance_index": instance_index,
        }

    def _ledger_unissue(self, request_id: str, stage: str) -> None:
        """阶段 7 §10.1 issued 层移除:completion 核销时。stage ==
        STAGE_REQUEST(REQUEST_COMPLETE)移除该 request 全部在飞条目
        (decode 段条目已在 decode completion 处理时移除,幂等);普通 stage
        只移除同 stage 条目。"""
        if stage == STAGE_REQUEST:
            self.ledger_issued.pop(request_id, None)
            return
        entry = self.ledger_issued.get(request_id)
        if entry is not None and entry.get("stage") == stage:
            self.ledger_issued.pop(request_id, None)

    def admitted_not_injected_queued(self) -> dict:
        """两层剩余负载查询(Python 侧):admitted-not-injected queued =
        排队账本成员(admitted)中按(队列类型 -> 阶段)尚未提交(committed)
        的部分。输出按队列类型分类并可回溯 request_id(contract ⑥)。"""
        stage_of_queue = {
            "prefill_qp": STAGE_PREFILL,
            "waiting_decode": STAGE_DECODE,
            "active_decode": STAGE_DECODE,
        }
        by_queue = {}
        detail = []
        for request_id, info in self.ledger_admitted.items():
            qtype = info["queue"].get("type", "unknown")
            committed_stages = {
                item["stage"]
                for item in self.ledger_committed.get(
                    request_id, {}).get("stages", [])
            }
            queued_stage = stage_of_queue.get(qtype)
            not_injected = (
                queued_stage is not None and queued_stage not in committed_stages)
            by_queue.setdefault(qtype, {"queued": 0, "not_injected": 0})
            by_queue[qtype]["queued"] += 1
            if not_injected:
                by_queue[qtype]["not_injected"] += 1
                detail.append({
                    "request_id": request_id,
                    "queue": qtype,
                    "stage": queued_stage,
                    "admitted_tick": info["admitted_tick"],
                    "instance_index": info["queue"].get("instance_index"),
                })
        return {
            "admitted_count": len(self.ledger_admitted),
            "committed_count": len(self.ledger_committed),
            "not_injected_count": len(detail),
            "by_queue": by_queue,
            "detail": detail,
        }

    def _sensing_issued_view(self) -> dict:
        """阶段 7 §10.1 issued 层边界视图:已发射未完成 request(常驻账本
        快照;与 C++ 同边界 injected-unfinished per_request 键集对平)。
        查询/审计输入,不进策略判据。"""
        detail = []
        for request_id in sorted(self.ledger_issued):
            entry = self.ledger_issued[request_id]
            detail.append({
                "request_id": request_id,
                "stage": entry["stage"],
                "issued_tick": entry["issued_tick"],
                "instance_index": entry.get("instance_index"),
            })
        return {"issued_count": len(detail), "detail": detail}

    def _sensing_ready_view(self) -> dict:
        """阶段 7 §10.1 ready 层边界视图(默认:无 ready 层;策略变体
        override)。ready = 已准入且所在实例就绪(非忙)可服务的排队成员
        (依赖已满足、资源将可服务)。查询/审计输入,不进策略判据。"""
        return {"ready_count": 0, "detail": []}

    def _record_sensing_query(self, delta: dict) -> None:
        """决策边界分层剩余负载查询快照(策略运行前 = 边界视图;§10.1 后
        为四层视图:admitted / issued / ready + C++ injected-unfinished
        摘要)。"""
        self.sensing_query_rows.append({
            "delivery_sequence": delta["delivery_sequence"],
            "tick": delta["tick"],
            "injected_unfinished": self.last_injected_unfinished,
            "admitted_not_injected_queued": self.admitted_not_injected_queued(),
            # 阶段 7 §10.1:issued(已发射未完成)与 ready(就绪可服务)层
            # 边界视图(与 C++ 摘要同一 delivery 同 tick,逐层对账输入)。
            "issued": self._sensing_issued_view(),
            "ready": self._sensing_ready_view(),
        })

    def dump_ledger(self, path: str) -> None:
        """分层账本导出(阶段 3 + 阶段 7 §10.1,感知开启时):逐 request 一行,
        含 admitted / committed / issued / completed-unreconciled 常驻层信息
        (已核销请求的各层齐备;运行结束 admitted / committed / issued 应已
        清空,由对账脚本复核;ready 层为边界视图不常驻,在
        sensing_query_log.jsonl 逐边界导出)。remote FIFO 为实账本层
        (contract ⑥,sh_1.0 独有:AnalyticalRemoteMemory 26 端口 FIFO 的
        pending/active 计数与字节,由 C++ 侧 Workload 计数采集,经
        remote_fifo_ledger.jsonl 逐交付快照导出,不入本逐 request 视图);
        local HBM job 为显式"不适用"占位(本仓无 LocalHbmBandwidthModel)。"""
        requests = {}
        for request_id, info in self.ledger_admitted.items():
            requests.setdefault(request_id, {})["admitted"] = info
        for request_id, info in self.ledger_committed.items():
            requests.setdefault(request_id, {})["committed"] = info
        for request_id, info in self.ledger_issued.items():
            requests.setdefault(request_id, {})["issued"] = info
        for request_id, info in self.ledger_completed_unreconciled.items():
            requests.setdefault(request_id, {})["completed_unreconciled"] = info
        with open(path, "w", encoding="utf-8") as out:
            for request_id in sorted(requests):
                out.write(json.dumps(
                    {"request_id": request_id,
                     "layers": requests[request_id]},
                    sort_keys=True) + "\n")

    def _start_batch(self, delta: dict) -> None:
        self._profile_batch = {"scanned_entries": 0, "full_scan_entries": 0}
        self._batch = {
            "delivery_sequence": delta["delivery_sequence"],
            "tick": delta["tick"],
            "reasons": list(delta["reasons"]),
            "nodes": [],
            "parent_edges": [],
            "watches": [],
            "assignments": [],
            "kv_actions": [],
            "future_alarms": [],
        }
        # 阶段 3 感知:本批次的发射记录槽(commit ack 层转移凭据)。
        self._emitted_by_delivery[delta["delivery_sequence"]] = {
            "tick": delta["tick"],
            "requests": [],
        }
        # 重建构图器(GraphBatchBuilder)的批次累加器:begin_batch() 之前
        # graph.batch 为 None,发射会在 _collect 处崩溃(步骤 1-8 闭环首跑
        # 暴露)。基类对 graph 的存在与否无感知(getattr),strategy 变体
        # (步骤 1-9)持有自己的构图器时同样生效。
        self._begin_graph_batch()

    def _begin_graph_batch(self) -> None:
        """变体若持有构图器(如 replay 的 GraphBatchBuilder),每批发射前
        重建其批次累加器;没有则跳过(基类自身不构图)。"""
        graph = getattr(self, "graph", None)
        if graph is not None and hasattr(graph, "begin_batch"):
            graph.begin_batch()

    def _graph_batch(self) -> dict:
        """构图器累加器(节点/边);没有构图器时返回空批次模板。"""
        graph = getattr(self, "graph", None)
        if graph is not None and graph.batch is not None:
            return graph.batch
        return {
            "nodes": [],
            "parent_edges": [],
            "watches": [],
            "assignments": [],
            "kv_actions": [],
            "future_alarms": [],
        }

    def _settle_completions(self, delta: dict) -> None:
        """核销 completion:每个 completed_groups 条目把对应 stage 从该请求的
        待办集合移除;REQUEST_COMPLETE(stage == "")核销整个请求。

        阶段 4 §7.3:经 pending fence 索引(in_flight)直接定位受影响
        request/stage(O(1) 字典,无全量扫描);重复完成事件 fail-closed
        (同 request 同 stage 完成两次即 abort;交付层去重保证 v1 不出现)。
        """
        for group in delta["completed_groups"]:
            self._profile_scan()  # §7.3:受影响条目(直接定位,非扫描)
            request_id = group["request_id"]
            stage = group["stage"]
            pending = self.in_flight.get(request_id)
            if pending is None:
                raise ValueError(
                    "completion for unknown request {!r} (stage {!r})".format(
                        request_id, stage))
            if stage == STAGE_REQUEST:
                if request_id in self.completed_request_ids:
                    raise ValueError("request {!r} completed twice".format(request_id))
                self.completed_request_ids.add(request_id)
                del self.in_flight[request_id]
                # 阶段 3 感知:REQUSET_COMPLETE 边界核销 -> completed-unreconciled。
                self._ledger_complete(request_id, delta["tick"])
            else:
                if stage not in pending:
                    raise ValueError(
                        "completion for request {!r} stage {!r} with no pending "
                        "stage (pending={!r})".format(request_id, stage, pending))
                pending.discard(stage)
            # 阶段 7 §10.1:issued 层移除与 completion 核销同步(在策略完成
            # 处理前完成,保证边界查询时刻的 issued 集与 C++ 同边界的
            # injected-unfinished per_request 键集对齐——C++ 在 watch 命中
            # 时已 finish 节点,本边界摘要不含已完成 request)。
            self._ledger_unissue(request_id, stage)

    def _process_arrivals(self, delta: dict) -> None:
        """处理 arrivals:到达请求登记 in-flight(prefill+decode 两段待办)。
        §7.3:直接登记(O(1));重复到达 fail-closed(交付层同 tick 去重 +
        冻结队列序保证 v1 不出现)。"""
        for arrival in delta["arrivals"]:
            self._profile_scan()  # §7.3:受影响条目(直接定位,非扫描)
            request_id = arrival["request_id"]
            if request_id in self.in_flight:
                raise ValueError("request {!r} arrived twice".format(request_id))
            self.in_flight[request_id] = {STAGE_PREFILL, STAGE_DECODE}

    # ------------------------------------------------------------- 决策日志 --

    def log_decision(self, record: dict, tick: int, *, decision: dict = None) -> None:
        """把一次实际执行的决策写入 online_decision_log(与离线
        decision_log.jsonl 同构: seq/tick/priority/kind/request_id/decision)。

        replay 模式默认沿用离线记录的 decision 内容(重放的就是离线决策);
        变体可传入自产 decision。seq 用在线决策序(1 起),tick 用在线边界 tick。
        """
        row = {
            "seq": len(self.online_log_rows) + 1,
            "tick": tick,
            "priority": record.get("priority", 0),
            "kind": record["kind"],
            "request_id": record["request_id"],
            "decision": (
                dict(record["decision"]) if decision is None else decision),
        }
        self.online_log_rows.append(row)

    # --------------------------------------------------------------- digest --

    def _digest_row(self, batch: dict) -> dict:
        payload = json.dumps(
            {
                "nodes": batch["nodes"],
                "parent_edges": batch["parent_edges"],
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        ranks = sorted({
            int(node["rank"])
            for node in batch["nodes"]
        })
        return {
            "delivery_sequence": batch["batch_id"],
            "tick": self._batch["tick"],
            "reasons": self._batch["reasons"],
            "node_count": len(batch["nodes"]),
            "edge_count": len(batch["parent_edges"]),
            "watch_count": len(batch["watches"]),
            "future_alarm_count": len(batch["future_alarms"]),
            "ranks": ranks,
            "content_sha256": hashlib.sha256(payload.encode()).hexdigest(),
        }

    # --------------------------------------------------------------- 收尾 --

    def on_commit_ack(self, ack: dict) -> None:
        """commit ack 到达(provisional 账本 finalize 挂接点;replay 模式只
        计数)。阶段 3:ack 的 batch_id == 请求批次 delivery_sequence;该批次
        发射过的 request/stage 转移入 committed 层(分层账本最小子集;
        簿记免费,感知关闭时同样记账)。

        阶段 4 §7.2 ack 通道 fail-closed:
          - schema_version 必须为 v1;
          - batch_id == delivery_sequence(v1 恒等,禁发散);
          - delivery_sequence 必须是已应用过的交付(<= last_applied_sequence;
            C++ 在读完 response 后才发 ack,不可能领先,领先即违例);
          - 重复 delivery_sequence 幂等忽略(不重复层转移;与协议层文件去重
            一致,防御性)。
        """
        if not isinstance(ack, dict):
            raise ValueError("ack must be a dict, got {!r}".format(type(ack)))
        if ack.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(
                "ack schema_version {!r} != {}".format(
                    ack.get("schema_version"), SCHEMA_VERSION))
        seq = ack.get("delivery_sequence")
        if not isinstance(seq, int) or seq < 0:
            raise ValueError(
                "ack delivery_sequence must be a non-negative int, got {!r}"
                .format(seq))
        if ack.get("batch_id") != seq:
            raise ValueError(
                "ack batch_id {} != delivery_sequence {} (v1 forbids "
                "divergence)".format(ack.get("batch_id"), seq))
        if seq > self.last_applied_sequence:
            raise ValueError(
                "ack for delivery {} not yet applied (last_applied={})"
                .format(seq, self.last_applied_sequence))
        if seq in self._seen_ack_delivery_seqs:
            return  # 幂等:重复 ack 不重复层转移
        self._seen_ack_delivery_seqs.add(seq)
        self.ack_count += 1
        # 阶段 5 §8.2:provisional KV 账本 finalize -- C++ 端已把本批
        # kv_actions 随 GraphBatch 原子提交(validate 通过才 commit),ack
        # 即确认凭据;暂存条目转入 committed 层。幂等 ack 在上面的去重门
        # 直接返回,不会重复转移。
        if seq in self._provisional_kv_actions:
            self._committed_kv_actions[seq] = self._provisional_kv_actions.pop(
                seq)
        emitted = self._emitted_by_delivery.get(seq)
        if emitted is None:
            return
        for request_id, stage in emitted["requests"]:
            self._ledger_commit(request_id, emitted["tick"], seq, stage)

    def verify_run_end(self) -> None:
        """serve_forever 返回(EOF)后的协议一致性校验;失败抛异常(fail-closed)。"""
        if self.ack_count != self.delivery_count:
            raise RuntimeError(
                "protocol mismatch at run end: ack_count={} != delivery_count={}"
                .format(self.ack_count, self.delivery_count))
        # seq 从 0 起(合同 §3.2 修正记录):N 次应用后 last_applied == N-1。
        if self.last_applied_sequence != self.delivery_count - 1:
            raise RuntimeError(
                "protocol mismatch at run end: last_applied_sequence={} != "
                "delivery_count-1={}".format(
                    self.last_applied_sequence, self.delivery_count - 1))
        cached_seq = -1 if self._delivery_reply_cache is None else (
            self._delivery_reply_cache["seq"])
        if cached_seq != self.last_applied_sequence:
            raise RuntimeError(
                "protocol mismatch at run end: reply cache covers delivery "
                "{} but last applied is {}".format(
                    cached_seq, self.last_applied_sequence))
        if self.in_flight:
            raise RuntimeError(
                "run ended with un-settled in-flight requests: {!r}"
                .format(sorted(self.in_flight)))
        # 阶段 5 §8.2:provisional KV 账本结束审计 -- 每笔已入暂存的
        # kv_actions 都必须等到 commit ack 确认;非空 = 有 delivery 未确认
        # (C++ 端 GraphBatch 提交失败才会出现;fail-closed)。
        if self._provisional_kv_actions:
            raise RuntimeError(
                "run ended with un-committed provisional KV actions: "
                "deliveries {} (every kv_actions journal entry must be "
                "finalized by its commit ack)".format(
                    sorted(self._provisional_kv_actions)))
        if self.sensing_enabled:
            # 阶段 3 + 阶段 7 §10.1:分层账本结束态——所有请求已核销,排队
            # 账本 / committed / issued 层清空,completed-unreconciled 覆盖
            # 全部完成请求。
            if self.ledger_admitted or self.ledger_committed or \
                    self.ledger_issued:
                raise RuntimeError(
                    "sensing ledger: run ended with un-settled layers "
                    "(admitted={!r} committed={!r} issued={!r})".format(
                        sorted(self.ledger_admitted),
                        sorted(self.ledger_committed),
                        sorted(self.ledger_issued)))
            if set(self.completed_request_ids) != set(
                    self.ledger_completed_unreconciled):
                raise RuntimeError(
                    "sensing ledger: completed/unreconciled mismatch "
                    "(completed={} unreconciled={})".format(
                        len(self.completed_request_ids),
                        len(self.ledger_completed_unreconciled)))
            if not self.injected_unfinished_history:
                raise RuntimeError(
                    "sensing ledger: no injected-unfinished history recorded")
