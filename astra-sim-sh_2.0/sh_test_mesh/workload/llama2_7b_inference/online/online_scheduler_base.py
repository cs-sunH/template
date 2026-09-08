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
import os
import tempfile
import time

# ---------------------------------------------------------------- B4 在途尾部
# 方案 §4-B4 "PropagatingTail"(真实在途尾部):基类中三个"已产生、等待
# 下游核销/确认"的在途容器属于真实在途工作,禁止截断(红线 §2.1)。本
# 观测器为它们提供当前长度、历史峰值、按来源(增长点)计数与可配置
# fail-closed 上限;超限只 raise(桥层转 error response + 非 0 退出),
# 在途条目原样保留,绝不静默丢弃。
_PROPAGATING_TAIL_SOURCE_ARRIVALS = "arrivals"        # in_flight(到达未完成)
_PROPAGATING_TAIL_SOURCE_DELIVERY = "delivery_start"  # _emitted_by_delivery(未 ack)
_PROPAGATING_TAIL_SOURCE_KV = "kv_provisional"        # _provisional_kv_actions(未确认)

# delivery 类来源(delivery_start / kv_provisional)默认上限推导:
# FileDecisionBridge v0 背压 = 一轮至多 1 个在途 request(C++ 不收到
# response 不发下一个;见 decision_bridge.py 协议头),ack 序列严格 +1
# 单调(_BoundedSequenceTracker),故最大合法未确认交付数 = 1。默认上限
# 取 8 = 8x 合法窗口,余量覆盖:响应文件原子发布与 ack 文件写入之间的
# 时序窗口、ack 与下一 request 同轮到达的排序差,以及未来受控多在途
# 扩展(届时应显式配置)。SH_PROPAGATING_TAIL_LIMIT=<正整数> 覆盖。
_PROPAGATING_TAIL_DELIVERY_LIMIT_DEFAULT = 8
_PROPAGATING_TAIL_LIMIT_ENV = "SH_PROPAGATING_TAIL_LIMIT"


class PropagatingTailTracker:
    """真实在途尾部观测 + fail-closed 上限(方案 §4-B4;纯记账,零决策影响)。

    每个来源 = (标签, 在途容器 getter, 上限)。observe_growth() 由增长点在
    登记完成后调用:重读容器真实长度(不自行加减,与容器永不漂移),更新
    历史峰值与累计增长计数,再做上限检查。超限 raise RuntimeError —— 异常
    经 on_decision_batch 向上传播,由 BridgeServer 写 error response 并以
    非 0 退出(整轮失败重来);超限时刻容器内容完整保留,本类不提供任何
    截断/丢弃手段(红线:真实在途工作禁止截断)。
    """

    def __init__(self):
        self._sources = {}

    def register(self, name, container_getter, limit) -> None:
        if name in self._sources:
            raise ValueError(
                "propagating tail source {!r} registered twice".format(name))
        if int(limit) <= 0:
            raise ValueError(
                "propagating tail limit for {!r} must be positive, got {!r}"
                .format(name, limit))
        self._sources[name] = {
            "getter": container_getter,
            "limit": int(limit),
            "peak": 0,
            "grow_count": 0,
        }

    def configure_limit(self, name, limit) -> None:
        """显式覆盖某来源上限(测试/运维;必须为正整数)。"""
        self._require(name)["limit"] = int(limit)

    def limit(self, name) -> int:
        return self._require(name)["limit"]

    def _require(self, name) -> dict:
        source = self._sources.get(name)
        if source is None:
            raise KeyError(
                "unknown propagating tail source {!r} (registered: {!r})"
                .format(name, sorted(self._sources)))
        return source

    def observe_growth(self, name) -> None:
        """增长点记账:当前长度经 getter 重读真实容器;峰值单调不减;
        累计增长计数 +1;超限 fail-closed(检查在登记之后——报错里的
        current 即真实长度,且包含刚登记的条目:不回滚、不截断)。"""
        source = self._require(name)
        current = len(source["getter"]())
        source["grow_count"] += 1
        if current > source["peak"]:
            source["peak"] = current
        if current > source["limit"]:
            raise RuntimeError(
                "propagating tail {!r} exceeded its fail-closed limit: "
                "current={} peak={} grow_count={} limit={} (real in-flight "
                "work is never truncated; the run fails closed)".format(
                    name, current, source["peak"], source["grow_count"],
                    source["limit"]))

    def snapshot(self) -> dict:
        """随时可读的观测快照:{来源: {current, peak, grow_count, limit}}。
        current 动态重读容器(缩减点不挂钩,快照仍准确)。"""
        return {
            name: {
                "current": len(source["getter"]()),
                "peak": source["peak"],
                "grow_count": source["grow_count"],
                "limit": source["limit"],
            }
            for name, source in self._sources.items()
        }

SCHEMA_VERSION = 1  # 阶段 4 §7.1:StateDelta schema v1(契约文档已删除,本校验器与 C++ 序列化为权威)

# 拼 batch 改造(2026-08-22):列车哨兵 watch 的批命名空间前缀。哨兵事件
# (T_max 截断列车 / 逐迭代 oracle 的完成信号)不属于任何请求,不经
# in-flight 核销,由变体的列车核销逻辑按 train_id 路由。
BATCH_TRAIN_PREFIX = "batch_train_"

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
        decision_log_sink=None,
        profile_sink=None,
        defensive_reply_cache: bool = False,
        sensing_query_sink=None,
        online_stats_sink=None,
        ledger_sink=None,
    ):
        self.mode = mode
        self.config = config
        # 运行期不保留完整 manifest 或 request_id -> manifest record 副本。
        # 唯一需要跨 delivery 保存的输入身份状态是尚未到达的 request ID：
        # arrival 后立即删除，故它单调缩小，不会随完成请求永久增长。
        requests = manifest["requests"]
        self.expected_request_count = len(requests)
        self.unseen_request_ids = {
            record["request_id"] for record in requests
        }
        if len(self.unseen_request_ids) != self.expected_request_count:
            raise ValueError("manifest request_id values must be unique")
        self.replay = replay
        self.digest_sink = digest_sink  # callable(dict) 或 None(不写 digest)
        # M3 流式落盘(2026-08-23):decision_log_sink / profile_sink 提供
        # 时逐行 append+flush 写出,调度器不再驻留行列表(online_log_count
        # 保留供 seq 编号与结束计数);缺省 None = 兼容旧路径(行仍缓冲在
        # 下列 rows 列表,供测试/夹具直读)。
        self.decision_log_sink = decision_log_sink
        self.profile_sink = profile_sink
        # B3 流式落盘(2026-08-28):sensing_query_log / online_stats 行在
        # 产出时即完整,提供 sink 时逐行流式写出(缺省 None = 兼容旧路径,
        # 行仍缓冲在下列 rows 列表,供测试/夹具直读)。online_stats
        # 的 sink 写出的是未合并桥接层 processing_ns 的基础行,由
        # online_service 结束期两遍合并成终文件(字节与改前一致)。
        self.sensing_query_sink = sensing_query_sink
        self.online_stats_sink = online_stats_sink
        # request-neutral 簿记。
        # pending fence 索引(阶段 4 §7.3):request_id -> set[str] 待办
        # stage(已完成 stage 从集合移除;REQUEST_COMPLETE 整体核销)。
        # 完成事件经该索引直接定位受影响 request/stage(O(1),非全量扫描)。
        self.in_flight = {}
        self.arrived_request_count = 0
        self.completed_request_count = 0
        self.delivery_count = 0   # 已应用决策批次数(幂等重放不计)
        self.ack_count = 0        # 已收到 commit ack 数(结束时应 == delivery_count)
        # 阶段 4 §7.2:delivery sequence/ack/幂等。
        # 最近一次已应用交付的 delivery_sequence。C++ 从 0 起(0 = 首个
        # tick-end 交付纪元,阶段 3 存档协议,合同 §3.2 修正记录),故 -1 =
        # 尚未应用任何交付(首个合法 seq 为 0)。
        self.last_applied_sequence = -1
        # 最近一次已应用交付的 (seq, delta, batch) 副本(幂等重放的凭据;
        # 重复 delivery 深度比对 delta 后返回缓存 batch 的 digest)。
        # B3(2026-08-23):生产路径默认引用缓存(不深拷)——C++ 单在途
        # 背压下无重复交付,重放返回路径(_gate_delivery_sequence)仍逐次
        # deepcopy,幂等语义不变;仅幂等 fixture(defensive_reply_cache=
        # True,显式开关)保留改前的双深拷防御姿态。
        self.defensive_reply_cache = bool(defensive_reply_cache)
        self._delivery_reply_cache = None
        # Ack 去重采用连续水位 + 少量乱序洞，而不是已见 seq 的永久集合。
        # _acked_out_of_order 至多覆盖仍在 _emitted_by_delivery 中等待确认的
        # delivery；当洞补齐时立即折叠入水位。
        self._acked_through = -1
        self._acked_out_of_order = set()
        self._batch = None        # 本批次累加器(每次 on_decision_batch 重建)
        # online_decision_log.jsonl 行(replay 模式)。M3 流式落盘起,
        # 生产路径(decision_log_sink 提供)行即写即弃,不驻留本列表;
        # online_log_count 恒维护(行号 seq 与结束计数用)。
        self.online_log_rows = []
        self.online_log_count = 0
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
        #   completed-unreconciled: REQUEST_COMPLETE 边界即流式写出；内存只
        #             保留计数，避免随完成请求数永久增长。
        self.ledger_admitted = {}
        self.ledger_committed = {}
        self.ledger_issued = {}
        self.ledger_completed_unreconciled_count = 0
        self.ledger_sink = ledger_sink
        # 测试/嵌入调用未提供 sink 时仍可在 dump_ledger() 导出完整审计
        # 产物，但记录写入临时文件而不驻留 Python 堆。
        self._ledger_spool = (
            tempfile.TemporaryFile(mode="w+t", encoding="utf-8")
            if self.sensing_enabled and ledger_sink is None else None)
        # 最近一次交付批次的 C++ injected-unfinished 摘要(ledger_summary;
        # 空列表 = 该批次无未完成注入节点或感知关闭)。
        self.last_injected_unfinished = []
        # injected-unfinished 仅保留最近一笔供边界查询；历史只需要一个
        # 记录数来证明至少收到过交付，不能按 delivery 累积完整快照。
        self.injected_unfinished_record_count = 0
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
        # B2(2026-08-28):committed 层从驻留 dict 改为计数器——原 dict 在
        # ack 后只增不删且全仓唯一读者是幂等 fixture 的 len(),纯死重;
        # 计数器语义等价(每笔确认 +1,幂等 ack 在去重门返回不重复计)。
        self.committed_kv_action_batches = 0
        # ---------------------------------------------------------------- 阶段 6
        # §9.1 分项计数器统一采集(Python 侧;写 online_stats.jsonl)。
        #   python_callback_count_by_reason: 按 reason 统计已应用交付所携带
        #     decision 事件数(每个 delta.reasons 条目计 1;合计 = 交付的决策
        #     事件总数)。每次 on_decision_batch 进入 = 一次 Python callback,
        #     其总数 = delivery_count(独立进程架构下无 engine-idle callback)。
        #   scheduler_self_ns_total: on_decision_batch 纯 Python 墙钟累计
        #     (含幂等重放路径;官方路径无重放,二者相等)。
        #   online_stats_rows: 每已应用交付一行 {delivery_sequence, tick,
        #     reasons, scheduler_self_ns, node_count}。M3 残留:行需结束期
        #     合并桥接层 processing_ns,保留缓冲(online_service 落盘)。
        self.python_callback_count_by_reason = {}
        self.scheduler_self_ns_total = 0
        # B3(2026-08-28):提供 online_stats_sink 时行即写即弃(online_service
        # 落盘),本列表不再驻留;缺省缓冲(测试/夹具路径)。
        self.online_stats_rows = []
        # ---------------------------------------------------------------- B4
        # 真实在途尾部(方案 §4-B4 "PropagatingTail")观测注册。上限推导:
        #   arrivals = expected_request_count —— manifest request_id 唯一
        #     (构造期校验)+ 重复到达 fail-closed ⇒ in_flight 条目数 ≤ 全部
        #     请求同时到达且无一完成,即最大合法并发的精确上界;超限只可能
        #     是 unseen/arrival 校验逻辑漂移,属结构性断言;
        #   delivery 类 = _PROPAGATING_TAIL_DELIVERY_LIMIT_DEFAULT(8x 桥
        #     背压窗口,推导见模块头),SH_PROPAGATING_TAIL_LIMIT 覆盖。
        self.propagating_tail = PropagatingTailTracker()
        _delivery_tail_limit = os.environ.get(
            _PROPAGATING_TAIL_LIMIT_ENV, "")
        if not _delivery_tail_limit:
            _delivery_tail_limit = _PROPAGATING_TAIL_DELIVERY_LIMIT_DEFAULT
        else:
            _delivery_tail_limit = int(_delivery_tail_limit)
            if _delivery_tail_limit <= 0:
                raise ValueError(
                    "{} must be a positive integer, got {!r}".format(
                        _PROPAGATING_TAIL_LIMIT_ENV,
                        os.environ[_PROPAGATING_TAIL_LIMIT_ENV]))
        self.propagating_tail.register(
            _PROPAGATING_TAIL_SOURCE_ARRIVALS,
            lambda: self.in_flight, self.expected_request_count)
        self.propagating_tail.register(
            _PROPAGATING_TAIL_SOURCE_DELIVERY,
            lambda: self._emitted_by_delivery, _delivery_tail_limit)
        self.propagating_tail.register(
            _PROPAGATING_TAIL_SOURCE_KV,
            lambda: self._provisional_kv_actions, _delivery_tail_limit)

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
        # §7.2:缓存本次已应用交付(幂等重放凭据)。B3(2026-08-23):生产
        # 路径引用缓存(批对象/下一批 begin_batch 重建累加器,旧引用不被
        # 改写;delta 桥读后即弃);defensive_reply_cache=True(幂等
        # fixture)保留双深拷防御姿态——副本防对端 setdefault 污染缓存。
        if self.defensive_reply_cache:
            self._delivery_reply_cache = {
                "seq": delta["delivery_sequence"],
                "delta": copy.deepcopy(delta),
                "batch": copy.deepcopy(batch),
            }
        else:
            self._delivery_reply_cache = {
                "seq": delta["delivery_sequence"],
                "delta": delta,
                "batch": batch,
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
        row = {
            "delivery_sequence": delta["delivery_sequence"],
            "tick": delta["tick"],
            "reasons": list(delta["reasons"]),
            "scheduler_self_ns": scheduler_self_ns,
            "node_count": len(batch["nodes"]),
        }
        # B3:流式落盘(基础行,online_service 结束期合并 processing_ns)。
        if self.online_stats_sink is not None:
            self.online_stats_sink(row)
        else:
            self.online_stats_rows.append(row)

    def run_variant_policy(self, delta) -> None:  # noqa: D401 -- 变体挂钩
        raise NotImplementedError

    @staticmethod
    def _sorted_touched_ranks(graph_batch: dict) -> list[int]:
        """Return the exact batch rank set without rescanning node payloads.

        GraphBatchBuilder records this private ledger at the same append point
        as every node.  The fallback keeps custom/legacy graph adapters
        byte-for-byte compatible until they adopt the ledger.
        """
        rank_ledger = graph_batch.get("_touched_ranks")
        if rank_ledger is not None:
            return sorted({int(rank) for rank in rank_ledger})
        return sorted({
            int(node["rank"]) for node in graph_batch["nodes"]
        })

    def build_graph_batch(self) -> dict:
        """组装本批次 GraphBatch dict(节点/边/watch/未来 alarm 来自累加器),
        顺带写 graph_batch_digests.jsonl 一行。

        节点/边由构图器(GraphBatchBuilder)经 begin_batch/_collect 写入
        其自身累加器(self.graph.batch);watch/assignment/kv_action/未来
        alarm 由变体策略写入本类的 self._batch。组装时两边合并。
        """
        delivery_sequence = self._batch["delivery_sequence"]
        graph_batch = self._graph_batch()
        touched_ranks = self._sorted_touched_ranks(graph_batch)
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
            # 批 = []);由构图收集时的精确 rank 账本给出，C++ 仍会校验。
            "touched_ranks": touched_ranks,
        }
        # 阶段 5 §8.2:本批 kv_actions 先入 provisional 暂存区(非空才记账),
        # commit ack 到达后 finalize(见 on_commit_ack);幂等重放不经过这里
        # (sequence 门直接返回缓存 digest),不会重复入账。
        if self._batch["kv_actions"]:
            self._provisional_kv_actions[delivery_sequence] = copy.deepcopy(
                self._batch["kv_actions"])
            # B4 在途尾部观测:provisional KV 动作增长点(build_graph_batch)。
            self.propagating_tail.observe_growth(
                _PROPAGATING_TAIL_SOURCE_KV)
        if self.digest_sink is not None:
            self.digest_sink(self._digest_row(batch))
        # 阶段 4 §7.3:本决策批扫描条目数入 profile(M3:行在产出时即
        # 完整,提供 profile_sink 时逐行流式写出;缺省缓冲,供 dump_profile)。
        profile_row = {
            "delivery_sequence": delivery_sequence,
            "tick": self._batch["tick"],
            "scanned_entries": self._profile_batch["scanned_entries"],
            "full_scan_entries": self._profile_batch["full_scan_entries"],
        }
        if self.profile_sink is not None:
            self.profile_sink(profile_row)
        else:
            self.profile_rows.append(profile_row)
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
        """把每决策批扫描条目数写为 profile.jsonl(阶段 4 §7.3 验收输入)。
        M3:生产路径由 profile_sink 逐行流式写出(online_service),本方法
        仅服务缺省缓冲模式(测试/夹具)——流式模式下 rows 为空,不覆写。"""
        with open(path, "w", encoding="utf-8") as out:
            for row in self.profile_rows:
                out.write(json.dumps(row, sort_keys=True) + "\n")

    def _validate_schema(self, delta: dict) -> None:
        """schema v1 校验器(阶段 4 §7.1;原契约 §6 各条强制,契约文档已
        删除)。任何不满足即抛异常 -> BridgeServer 写 error
        response 并以非 0 退出(fail-closed,绝不静默降级)。"""
        if not isinstance(delta, dict):
            raise ValueError("delta must be a dict, got {!r}".format(type(delta)))
        if delta.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(
                "delta schema_version {!r} != {}".format(
                    delta.get("schema_version"), SCHEMA_VERSION))
        for field in ("delivery_sequence", "delivery_epoch", "tick",
                      "deferred_from_tick", "reasons", "arrivals",
                      "completed_groups", "completed_nodes",
                      "affected_ranks", "snapshot_handle"):
            if field not in delta:
                raise ValueError("delta missing field {!r}".format(field))
        if not isinstance(delta["reasons"], list):
            raise ValueError("delta reasons must be a list")
        if not isinstance(delta["arrivals"], list):
            raise ValueError("delta arrivals must be a list")
        if not isinstance(delta["completed_groups"], list):
            raise ValueError("delta completed_groups must be a list")
        # v1 新增字段类型 + 语义检查(原 §6 各条)。
        if not isinstance(delta["completed_nodes"], list):
            raise ValueError("delta completed_nodes must be a list")
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
        self.injected_unfinished_record_count += 1

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
        """REQUEST_COMPLETE 边界:移出活动层并流式输出 completed 层。"""
        admitted = self.ledger_admitted.get(request_id, {})
        committed = self.ledger_committed.get(request_id, {})
        completed = {
            "completed_tick": tick,
            "admitted_tick": admitted.get("admitted_tick"),
            "first_commit_tick": committed.get("first_commit_tick"),
            "stages": list(committed.get("stages", [])),
        }
        self.ledger_admitted.pop(request_id, None)
        self.ledger_committed.pop(request_id, None)
        if not self.sensing_enabled:
            return
        row = {
            "request_id": request_id,
            "layers": {"completed_unreconciled": completed},
        }
        self.ledger_completed_unreconciled_count += 1
        if self.ledger_sink is not None:
            self.ledger_sink(row)
            return
        if self._ledger_spool is None:
            raise RuntimeError("sensing ledger has neither sink nor spool")
        self._ledger_spool.write(json.dumps(row, sort_keys=True) + "\n")
        self._ledger_spool.flush()

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
        row = {
            "delivery_sequence": delta["delivery_sequence"],
            "tick": delta["tick"],
            "injected_unfinished": self.last_injected_unfinished,
            "admitted_not_injected_queued": self.admitted_not_injected_queued(),
            # 阶段 7 §10.1:issued(已发射未完成)与 ready(就绪可服务)层
            # 边界视图(与 C++ 摘要同一 delivery 同 tick,逐层对账输入)。
            "issued": self._sensing_issued_view(),
            "ready": self._sensing_ready_view(),
        }
        # B3:流式落盘(行在产出时即完整;缺省缓冲,供夹具直读)。
        if self.sensing_query_sink is not None:
            self.sensing_query_sink(row)
        else:
            self.sensing_query_rows.append(row)

    def dump_ledger(self, path: str) -> None:
        """完成账本导出。

        生产服务提供 ``ledger_sink`` 时，完成行已按 completion 顺序写入
        ``path``，此方法只校验无活动层；无 sink 的嵌入/测试路径从临时
        spool 复制，仍不重建全量 request 映射。
        """
        if self.ledger_admitted or self.ledger_committed or self.ledger_issued:
            raise RuntimeError("cannot dump ledger with active ledger layers")
        if self.ledger_sink is not None:
            return
        with open(path, "w", encoding="utf-8") as out:
            if self._ledger_spool is None:
                return
            self._ledger_spool.seek(0)
            for line in self._ledger_spool:
                out.write(line)

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
        # B4 在途尾部观测:未 ack 交付增长点(_start_batch;正常背压窗口
        # 内 current 恒 1,ack 丢失/协议违例才会累积并在此 fail-closed)。
        self.propagating_tail.observe_growth(
            _PROPAGATING_TAIL_SOURCE_DELIVERY)
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
            if request_id.startswith(BATCH_TRAIN_PREFIX):
                continue  # 列车哨兵事件:变体侧按 train_id 核销
            stage = group["stage"]
            pending = self.in_flight.get(request_id)
            if pending is None:
                raise ValueError(
                    "completion for unknown request {!r} (stage {!r})".format(
                        request_id, stage))
            if stage == STAGE_REQUEST:
                del self.in_flight[request_id]
                self.completed_request_count += 1
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
            if request_id not in self.unseen_request_ids:
                raise ValueError(
                    "arrival for unknown or already-arrived request {!r}"
                    .format(request_id))
            if request_id in self.in_flight:
                raise ValueError("request {!r} arrived twice".format(request_id))
            self.unseen_request_ids.remove(request_id)
            self.arrived_request_count += 1
            self.in_flight[request_id] = {STAGE_PREFILL, STAGE_DECODE}
            # B4 在途尾部观测:在途请求增长点(_process_arrivals)。
            self.propagating_tail.observe_growth(
                _PROPAGATING_TAIL_SOURCE_ARRIVALS)

    # ------------------------------------------------------------- 决策日志 --

    def log_decision(self, record: dict, tick: int, *, decision: dict = None) -> None:
        """把一次实际执行的决策写入 online_decision_log(与离线
        decision_log.jsonl 同构: seq/tick/priority/kind/request_id/decision)。

        replay 模式默认沿用离线记录的 decision 内容(重放的就是离线决策);
        变体可传入自产 decision。seq 用在线决策序(1 起),tick 用在线边界 tick。

        M3 流式落盘:提供 decision_log_sink 时行即写即弃(online_log_count
        恒维护行号);缺省缓冲在 online_log_rows(兼容测试/夹具直读)。
        """
        row = {
            "seq": self.online_log_count + 1,
            "tick": tick,
            "priority": record.get("priority", 0),
            "kind": record["kind"],
            "request_id": record["request_id"],
            "decision": (
                dict(record["decision"]) if decision is None else decision),
        }
        self.online_log_count += 1
        if self.decision_log_sink is not None:
            self.decision_log_sink(row)
        else:
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
        # build_graph_batch 已把构图器的精确 rank 账本冻结到公开字段；
        # digest 复用它，避免对同一批 nodes 再做一遍全量 rank 扫描。
        ranks = (list(batch["touched_ranks"])
                 if "touched_ranks" in batch
                 else self._sorted_touched_ranks(batch))
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
        if seq <= self._acked_through or seq in self._acked_out_of_order:
            return  # 幂等:已被水位或乱序洞记录的 ack 不重复层转移
        # 每个已应用 delivery 都在 _start_batch 创建一条未确认记录。缺失
        # 说明 ack 已被消费或协议状态损坏；不能像旧实现那样静默加计数。
        emitted = self._emitted_by_delivery.get(seq)
        if emitted is None:
            raise ValueError(
                "ack for delivery {} has no pending emitted record".format(seq))
        self.ack_count += 1
        # 阶段 5 §8.2:provisional KV 账本 finalize -- C++ 端已把本批
        # kv_actions 随 GraphBatch 原子提交(validate 通过才 commit),ack
        # 即确认凭据;暂存条目转入 committed 层。幂等 ack 在上面的去重门
        # 直接返回,不会重复转移。
        if seq in self._provisional_kv_actions:
            self._provisional_kv_actions.pop(seq)
            self.committed_kv_action_batches += 1
        # M4 核销即删(2026-08-23):_emitted_by_delivery 条目在 ack 层转移
        # 后即死重——get 改 pop 当场弹出(重复 ack 已被上方去重门拦截,
        # 每 seq 至多弹出一次;条目创建于 _start_batch,消费于此,无其他
        # 读者)。
        emitted = self._emitted_by_delivery.pop(seq)
        for request_id, stage in emitted["requests"]:
            self._ledger_commit(request_id, emitted["tick"], seq, stage)
        if seq == self._acked_through + 1:
            self._acked_through = seq
            while self._acked_through + 1 in self._acked_out_of_order:
                self._acked_out_of_order.remove(self._acked_through + 1)
                self._acked_through += 1
        else:
            self._acked_out_of_order.add(seq)
        self._release_acknowledged_reply(seq)

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
        if (self._delivery_reply_cache is not None
                and self._delivery_reply_cache["seq"]
                != self.last_applied_sequence):
            raise RuntimeError(
                "protocol mismatch at run end: reply cache covers delivery "
                "{} but last applied is {}".format(
                    self._delivery_reply_cache["seq"],
                    self.last_applied_sequence))
        if self.in_flight:
            raise RuntimeError(
                "run ended with un-settled in-flight requests: {!r}"
                .format(sorted(self.in_flight)))
        if (self.arrived_request_count != self.expected_request_count
                or self.completed_request_count != self.expected_request_count
                or self.unseen_request_ids):
            raise RuntimeError(
                "request lifecycle mismatch at run end: expected={} arrived={} "
                "completed={} unseen={}".format(
                    self.expected_request_count, self.arrived_request_count,
                    self.completed_request_count, len(self.unseen_request_ids)))
        if self._emitted_by_delivery:
            raise RuntimeError(
                "run ended with unacknowledged emitted deliveries: {}"
                .format(sorted(self._emitted_by_delivery)[:5]))
        if (self._acked_through != self.last_applied_sequence
                or self._acked_out_of_order):
            raise RuntimeError(
                "ack watermark mismatch at run end: through={} last_applied={} "
                "out_of_order={}".format(
                    self._acked_through, self.last_applied_sequence,
                    sorted(self._acked_out_of_order)[:5]))
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
            # 账本 / committed / issued 层清空,流式 completed-unreconciled
            # 行数覆盖全部完成请求。
            if self.ledger_admitted or self.ledger_committed or \
                    self.ledger_issued:
                raise RuntimeError(
                    "sensing ledger: run ended with un-settled layers "
                    "(admitted={!r} committed={!r} issued={!r})".format(
                        sorted(self.ledger_admitted),
                        sorted(self.ledger_committed),
                        sorted(self.ledger_issued)))
            if self.completed_request_count != \
                    self.ledger_completed_unreconciled_count:
                raise RuntimeError(
                "sensing ledger: completed/unreconciled mismatch "
                "(completed={} unreconciled={})".format(
                        self.completed_request_count,
                        self.ledger_completed_unreconciled_count))
            if self.injected_unfinished_record_count == 0:
                raise RuntimeError(
                    "sensing ledger: no injected-unfinished history recorded")

    def _release_acknowledged_reply(self, seq: int) -> None:
        """Drop the last response as soon as C++ confirms receipt.

        The reply cache is intentionally retained only until the corresponding
        commit ack so that a transport retry before acknowledgement remains
        idempotent.  Once the ack arrives, the protocol forbids replaying that
        delivery and retaining its delta/batch would pin every graph node and
        edge in the Python heap for an entire long-running service.
        """
        cached = self._delivery_reply_cache
        if cached is None or cached["seq"] != seq:
            return
        current_batch = (
            self._batch is not None
            and self._batch["delivery_sequence"] == seq)
        self._delivery_reply_cache = None
        if not current_batch:
            return
        self._batch = None
        graph = getattr(self, "graph", None)
        if graph is not None and hasattr(graph, "batch"):
            graph.batch = None
            if hasattr(graph, "batch_first_step"):
                graph.batch_first_step = False
