#!/usr/bin/env python3
"""test_propagating_tail.py -- B4(方案 §4-B4)真实在途尾部("PropagatingTail")
观测 + fail-closed 上限的聚焦回归。

被测逻辑 = online_scheduler_base.PropagatingTailTracker 及其三个挂点:
  arrivals       -> in_flight            (增长点 _process_arrivals)
  delivery_start -> _emitted_by_delivery (增长点 _start_batch)
  kv_provisional -> _provisional_kv_actions (增长点 build_graph_batch)

断言三类(任务纪律):正常增长与按来源计数 / 峰值记录(回落后保持) /
超限 fail-closed(raise 且在途条目完整保留 = 绝不截断真实在途工作)。

运行:cd sh_test_mesh/workload/llama2_7b_inference &&
      python3 online/test_propagating_tail.py   （或 pytest 同路径）
"""
import os
import sys
from types import SimpleNamespace

_ONLINE_DIR = os.path.dirname(os.path.abspath(__file__))
_WORKLOAD_DIR = os.path.dirname(_ONLINE_DIR)
for _path in (_ONLINE_DIR, _WORKLOAD_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from online.online_scheduler_base import (  # noqa: E402
    OnlineSchedulerBase,
    PropagatingTailTracker,
    _PROPAGATING_TAIL_SOURCE_ARRIVALS,
    _PROPAGATING_TAIL_SOURCE_DELIVERY,
    _PROPAGATING_TAIL_SOURCE_KV,
)


def _delta(seq, tick, arrivals=(), completed=()):
    """最小合法 StateDelta v1(通过 _validate_schema 的全部强制项)。"""
    return {
        "schema_version": 1,
        "delivery_sequence": seq,
        "delivery_epoch": seq,
        "tick": tick,
        "deferred_from_tick": tick,
        "reasons": ["ARRIVAL"],
        "arrivals": [
            {"request_id": request_id, "queue_index": index, "ingress_seq": 0}
            for index, request_id in enumerate(arrivals)
        ],
        "completed_groups": [
            {"request_id": request_id, "stage": stage}
            for request_id, stage in completed
        ],
        "completed_nodes": [],
        "retry_items": [],
        "affected_ranks": [],
        "snapshot_handle": {"epoch": seq, "tick": tick},
    }


def _ack(seq):
    return {"schema_version": 1, "delivery_sequence": seq, "batch_id": seq}


class _TailScheduler(OnlineSchedulerBase):
    """最小变体:策略无动作(纯基类簿记路径)。"""

    def run_variant_policy(self, delta):
        pass


class _KvTailScheduler(OnlineSchedulerBase):
    """带 kv_actions 的变体:每个含 arrivals 的交付产生一笔 provisional
    KV 动作(驱动 kv_provisional 增长点)。"""

    def run_variant_policy(self, delta):
        if delta["arrivals"]:
            self._batch["kv_actions"].append({"op": "reserve", "seq":
                                              delta["delivery_sequence"]})


def _scheduler(cls=_TailScheduler, count=4, **kwargs):
    return cls(
        manifest={"requests": [
            {"request_id": "r{}".format(index)} for index in range(count)
        ]},
        config=SimpleNamespace(),
        **kwargs)


# ---------------------------------------------------------------- 正常增长 --

def test_normal_growth_records_current_peak_and_source_counts():
    s = _scheduler()
    assert s.propagating_tail.snapshot() == {
        _PROPAGATING_TAIL_SOURCE_ARRIVALS: {
            "current": 0, "peak": 0, "grow_count": 0,
            "limit": s.expected_request_count},
        _PROPAGATING_TAIL_SOURCE_DELIVERY: {
            "current": 0, "peak": 0, "grow_count": 0, "limit": 8},
        _PROPAGATING_TAIL_SOURCE_KV: {
            "current": 0, "peak": 0, "grow_count": 0, "limit": 8},
    }

    s.on_decision_batch(_delta(0, 1000, arrivals=("r0", "r1")))
    s.on_commit_ack(_ack(0))
    assert set(s.in_flight) == {"r0", "r1"}
    assert s.propagating_tail.snapshot()[
        _PROPAGATING_TAIL_SOURCE_ARRIVALS] == {
        "current": 2, "peak": 2, "grow_count": 2, "limit": 4}
    # delivery_start:背压窗口内未确认交付恒 1(ack 后清零),累计增长 1 次。
    assert s.propagating_tail.snapshot()[
        _PROPAGATING_TAIL_SOURCE_DELIVERY] == {
        "current": 0, "peak": 1, "grow_count": 1, "limit": 8}
    # 本变体无 kv_actions:kv_provisional 零增长。
    assert s.propagating_tail.snapshot()[
        _PROPAGATING_TAIL_SOURCE_KV]["grow_count"] == 0

    s.on_decision_batch(_delta(1, 2000, completed=(("r0", ""),)))
    s.on_commit_ack(_ack(1))
    arrivals_view = s.propagating_tail.snapshot()[
        _PROPAGATING_TAIL_SOURCE_ARRIVALS]
    assert arrivals_view["current"] == 1          # current 动态重读容器
    assert arrivals_view["peak"] == 2             # 峰值不随回落下降
    assert arrivals_view["grow_count"] == 2       # 缩减点不计入增长

    s.on_decision_batch(_delta(2, 3000, arrivals=("r2",)))
    s.on_commit_ack(_ack(2))
    s.on_decision_batch(_delta(3, 4000, arrivals=("r3",)))
    s.on_commit_ack(_ack(3))
    s.on_decision_batch(_delta(4, 5000,
                               completed=(("r1", ""), ("r2", ""), ("r3", ""))))
    s.on_commit_ack(_ack(4))
    s.verify_run_end()
    final = s.propagating_tail.snapshot()
    # delta 3 后 {r1, r2, r3} 并存为峰值 3(随后全部核销,current=0)。
    assert final[_PROPAGATING_TAIL_SOURCE_ARRIVALS] == {
        "current": 0, "peak": 3, "grow_count": 4, "limit": 4}
    assert final[_PROPAGATING_TAIL_SOURCE_DELIVERY] == {
        "current": 0, "peak": 1, "grow_count": 5, "limit": 8}
    print("[propagating-tail] case 1 PASS: growth/current/peak/per-source "
          "counts across arrivals, completions and acks")


def test_kv_provisional_source_counts_grow_and_settle():
    s = _scheduler(_KvTailScheduler)
    s.on_decision_batch(_delta(0, 1000, arrivals=("r0",)))
    s.on_commit_ack(_ack(0))
    s.on_decision_batch(_delta(1, 2000, arrivals=("r1",)))
    view = s.propagating_tail.snapshot()[_PROPAGATING_TAIL_SOURCE_KV]
    assert view == {"current": 1, "peak": 1, "grow_count": 2, "limit": 8}
    assert list(s._provisional_kv_actions) == [1]   # seq 0 已被 ack 确认
    print("[propagating-tail] case 2 PASS: kv_provisional grows at "
          "build_graph_batch and settles on commit ack")


# ------------------------------------------------------------- 峰值记录 --

def test_peak_persists_after_full_drain_and_regrow():
    s = _scheduler(count=4)
    s.on_decision_batch(_delta(0, 1000, arrivals=("r0", "r1", "r2")))
    s.on_commit_ack(_ack(0))
    s.on_decision_batch(_delta(1, 2000,
                               completed=(("r0", ""), ("r1", ""), ("r2", ""))))
    s.on_commit_ack(_ack(1))
    arrivals_view = s.propagating_tail.snapshot()[
        _PROPAGATING_TAIL_SOURCE_ARRIVALS]
    assert arrivals_view == {"current": 0, "peak": 3, "grow_count": 3,
                             "limit": 4}
    # 排空后重新增长到更低水位:current 跟随,peak 保持历史最大。
    s.on_decision_batch(_delta(2, 3000, arrivals=("r3",)))
    s.on_commit_ack(_ack(2))
    arrivals_view = s.propagating_tail.snapshot()[
        _PROPAGATING_TAIL_SOURCE_ARRIVALS]
    assert arrivals_view["current"] == 1
    assert arrivals_view["peak"] == 3
    print("[propagating-tail] case 3 PASS: peak survives drain and "
          "partial regrowth")


# --------------------------------------------------------- 超限 fail-closed --

def test_fail_closed_arrivals_over_limit_keeps_work_untruncated():
    s = _scheduler(count=4)
    # 显式收紧上限(模拟结构性断言被打破:arrivals 来源超过配置并发)。
    s.propagating_tail.configure_limit(_PROPAGATING_TAIL_SOURCE_ARRIVALS, 1)
    # 第 1 个到达:current=1 == limit,合法。
    s.on_decision_batch(_delta(0, 1000, arrivals=("r0",)))
    s.on_commit_ack(_ack(0))
    # 第 2 个到达使 current=2 > 1:fail-closed raise。
    try:
        s.on_decision_batch(_delta(1, 2000, arrivals=("r1",)))
    except RuntimeError as exc:
        assert "arrivals" in str(exc), exc
        assert "fail-closed" in str(exc), exc
    else:
        raise AssertionError("arrivals over-limit must fail closed")
    # 红线:超限 raise 不截断真实在途工作 —— 已登记条目(含触发者)完整保留。
    assert set(s.in_flight) == {"r0", "r1"}
    arrivals_view = s.propagating_tail.snapshot()[
        _PROPAGATING_TAIL_SOURCE_ARRIVALS]
    assert arrivals_view["peak"] == 2
    assert arrivals_view["grow_count"] == 2
    print("[propagating-tail] case 4 PASS: arrivals over-limit raises with "
          "in-flight work fully retained (no truncation)")


def test_fail_closed_unacked_deliveries_over_limit():
    s = _scheduler(count=6)
    s.propagating_tail.configure_limit(_PROPAGATING_TAIL_SOURCE_DELIVERY, 2)
    # 连续 3 个已应用交付不 ack(协议违例:背压窗口=1):第 3 个增长点
    # _start_batch 处 fail-closed。
    s.on_decision_batch(_delta(0, 1000, arrivals=("r0",)))
    s.on_decision_batch(_delta(1, 2000, arrivals=("r1",)))
    assert s.propagating_tail.snapshot()[
        _PROPAGATING_TAIL_SOURCE_DELIVERY] == {
        "current": 2, "peak": 2, "grow_count": 2, "limit": 2}
    try:
        s.on_decision_batch(_delta(2, 3000, arrivals=("r2",)))
    except RuntimeError as exc:
        assert "delivery_start" in str(exc), exc
    else:
        raise AssertionError("unacked delivery over-limit must fail closed")
    # 不截断:3 条未确认发射记录(含触发者)原样保留。
    assert sorted(s._emitted_by_delivery) == [0, 1, 2]
    assert s.propagating_tail.snapshot()[
        _PROPAGATING_TAIL_SOURCE_DELIVERY]["peak"] == 3
    print("[propagating-tail] case 5 PASS: unacked deliveries over-limit "
          "raise with the emitted ledger intact")


def test_fail_closed_kv_provisional_over_limit():
    s = _scheduler(_KvTailScheduler, count=6)
    s.propagating_tail.configure_limit(_PROPAGATING_TAIL_SOURCE_KV, 2)
    s.on_decision_batch(_delta(0, 1000, arrivals=("r0",)))
    s.on_decision_batch(_delta(1, 2000, arrivals=("r1",)))
    try:
        s.on_decision_batch(_delta(2, 3000, arrivals=("r2",)))
    except RuntimeError as exc:
        assert "kv_provisional" in str(exc), exc
    else:
        raise AssertionError("kv provisional over-limit must fail closed")
    assert sorted(s._provisional_kv_actions) == [0, 1, 2]
    print("[propagating-tail] case 6 PASS: kv_provisional over-limit raises "
          "with provisional actions intact")


def test_env_limit_override_applies_to_delivery_sources():
    os.environ["SH_PROPAGATING_TAIL_LIMIT"] = "3"
    try:
        s = _scheduler()
    finally:
        del os.environ["SH_PROPAGATING_TAIL_LIMIT"]
    snapshot = s.propagating_tail.snapshot()
    assert snapshot[_PROPAGATING_TAIL_SOURCE_DELIVERY]["limit"] == 3
    assert snapshot[_PROPAGATING_TAIL_SOURCE_KV]["limit"] == 3
    # arrivals 上限不受 env 影响:恒为最大合法并发(expected_request_count)。
    assert snapshot[_PROPAGATING_TAIL_SOURCE_ARRIVALS]["limit"] == \
        s.expected_request_count
    print("[propagating-tail] case 7 PASS: SH_PROPAGATING_TAIL_LIMIT "
          "overrides delivery-class limits only")


# ------------------------------------------------------------- tracker 单元 --

def test_tracker_rejects_bad_registration_and_unknown_source():
    tracker = PropagatingTailTracker()
    container = []
    tracker.register("src", lambda: container, 4)
    for bad in (0, -1):
        try:
            tracker.register("other", lambda: container, bad)
        except ValueError:
            pass
        else:
            raise AssertionError("non-positive limit must be rejected")
    try:
        tracker.register("src", lambda: container, 4)
    except ValueError:
        pass
    else:
        raise AssertionError("duplicate source must be rejected")
    try:
        tracker.observe_growth("missing")
    except KeyError:
        pass
    else:
        raise AssertionError("unknown source must be rejected")
    container.extend([1, 2, 3, 4, 5])
    try:
        tracker.observe_growth("src")
    except RuntimeError:
        pass
    else:
        raise AssertionError("over-limit growth must raise")
    assert tracker.snapshot()["src"] == {
        "current": 5, "peak": 5, "grow_count": 1, "limit": 4}
    print("[propagating-tail] case 8 PASS: tracker registration/validation "
          "guards")


def _run_all():
    test_normal_growth_records_current_peak_and_source_counts()
    test_kv_provisional_source_counts_grow_and_settle()
    test_peak_persists_after_full_drain_and_regrow()
    test_fail_closed_arrivals_over_limit_keeps_work_untruncated()
    test_fail_closed_unacked_deliveries_over_limit()
    test_fail_closed_kv_provisional_over_limit()
    test_env_limit_override_applies_to_delivery_sources()
    test_tracker_rejects_bad_registration_and_unknown_source()
    print("[propagating-tail] all cases PASS")


if __name__ == "__main__":
    _run_all()
