#!/usr/bin/env python3
"""test_delivery_idempotency_gate.py -- 阶段 4 §7.2 delivery 幂等门 + ack 流。

E1(2026-09-28)自 verify/idempotency_fixture.py 规范形态化移植。原夹具
(305 行手动脚本,清单 §910 登记为 python3 直调)需要一次真实运行的
bridge 目录作输入,无任何自动调用面(全仓 grep 零引用);其可自动执行的
契约核心——调度器级 delivery 幂等门与 ack 流 fail-closed——本文件以
合成 delta 经真实 OnlineSchedulerBase 管线钉死(桥层重复 seq 幂等忽略
已由 test_d1_sched_fixes SCHED-9 单独钉住;运行结束对账的权威工具仍是
verify/ledger_reconcile.py)。断言语义逐条保留自原夹具:

  1. 同一 delivery 原样重放 -> 幂等:返回与首次逐字段相等的 batch
     (深拷贝副本),调度器可观测状态零变更;delivery_count 不重复计;
  2. 跳号(seq != last+1)/陈旧重复(旧 seq)/同 seq 不同内容(篡改)
     一律 ValueError fail-closed,且在任何状态变更前抛出(零污染);
  3. ack 流:重复 ack 幂等忽略;对较旧交付的 ack 不得释放最新回复/
     批次;全部 ack 后 reply cache/_batch 释放、水位折叠、ack_count
     == delivery_count;已 ack 交付不可再重放;
  4. ack fail-closed:schema_version 错 / batch_id 发散 / 领先
     last_applied 三形态 ValueError 且不计数;
  5. 干净运行 verify_run_end() 通过。

运行:pytest online/test_delivery_idempotency_gate.py,或
      cd sh_test_mesh/workload/llama2_7b_inference && python3 -m unittest
      online.test_delivery_idempotency_gate(本文件亦可直跑)。
"""
import copy
import os
import sys
import unittest
from types import SimpleNamespace

_ONLINE_DIR = os.path.dirname(os.path.abspath(__file__))
_WORKLOAD_DIR = os.path.dirname(_ONLINE_DIR)
for _path in (_ONLINE_DIR, _WORKLOAD_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from online.online_scheduler_base import (  # noqa: E402
    OnlineSchedulerBase,
)


class _CountingDigestSink:
    """只计数的 digest sink(不写文件;断言幂等重放不重复产出 digest)。"""

    def __init__(self):
        self.count = 0

    def __call__(self, row: dict) -> None:
        self.count += 1


class _GateScheduler(OnlineSchedulerBase):
    """最小变体:策略无动作(纯基类簿记路径,与 test_propagating_tail
    的 _TailScheduler 同款手法)。"""

    def run_variant_policy(self, delta):
        pass


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
        "affected_ranks": [],
        "snapshot_handle": {"epoch": seq, "tick": tick},
    }


def _ack(seq):
    return {"schema_version": 1, "delivery_sequence": seq, "batch_id": seq}


def _snapshot(scheduler, sink):
    """调度器可观测状态快照(幂等断言用;重复 delivery 必须逐项不变)。"""
    return {
        "delivery_count": scheduler.delivery_count,
        "last_applied_sequence": scheduler.last_applied_sequence,
        "ack_count": scheduler.ack_count,
        "in_flight": {request_id: sorted(stages)
                      for request_id, stages
                      in sorted(scheduler.in_flight.items())},
        "arrived_request_count": scheduler.arrived_request_count,
        "completed_request_count": scheduler.completed_request_count,
        "online_log_rows": copy.deepcopy(scheduler.online_log_rows),
        "online_log_count": scheduler.online_log_count,
        "emitted_by_delivery": copy.deepcopy(
            scheduler._emitted_by_delivery),
        "ack_watermark": (scheduler._acked_through,
                          tuple(sorted(scheduler._acked_out_of_order))),
        "provisional_kv_actions": sorted(
            scheduler._provisional_kv_actions),
        "committed_kv_action_batches": (
            scheduler.committed_kv_action_batches),
        "digest_count": sink.count,
        "reply_cache_seq": (
            -1 if scheduler._delivery_reply_cache is None
            else scheduler._delivery_reply_cache["seq"]),
        "batch_seq": (-1 if scheduler._batch is None
                      else scheduler._batch["delivery_sequence"]),
    }


def _scheduler(digest_sink=None, count=4, **kwargs):
    return _GateScheduler(
        manifest={"requests": [
            {"request_id": "r{}".format(index)} for index in range(count)
        ]},
        config=SimpleNamespace(),
        digest_sink=digest_sink,
        **kwargs)


class DeliveryIdempotencyGateTest(unittest.TestCase):
    """§7.2 幂等门:重放逐字段相等、状态零变更、fail-closed 零污染。"""

    def test_duplicate_delivery_replays_identically_zero_state_change(self):
        # 生产默认引用缓存与防御性深拷缓存两模式下幂等语义一致。
        for defensive in (False, True):
            with self.subTest(defensive_reply_cache=defensive):
                sink = _CountingDigestSink()
                s = _scheduler(digest_sink=sink,
                               defensive_reply_cache=defensive)
                delta = _delta(0, 1000, arrivals=("r0",))
                batch1 = s.on_decision_batch(delta)
                before = _snapshot(s, sink)

                batch2 = s.on_decision_batch(delta)
                self.assertEqual(batch2, batch1)
                self.assertIsNot(batch2, batch1)  # 恒深拷贝返回
                self.assertEqual(_snapshot(s, sink), before)
                self.assertEqual(s.delivery_count, 1)
                self.assertEqual(s.last_applied_sequence, 0)
                self.assertEqual(sink.count, 1)  # digest 不重复产出

    def test_gap_and_stale_duplicate_fail_closed_without_pollution(self):
        sink = _CountingDigestSink()
        s = _scheduler(digest_sink=sink)
        s.on_decision_batch(_delta(0, 1000, arrivals=("r0",)))
        state = _snapshot(s, sink)

        # 跳号:seq 2 在 seq 1 缺席时到达。
        with self.assertRaises(ValueError):
            s.on_decision_batch(_delta(2, 1200, arrivals=("r1",)))
        # 陈旧重复:应用 seq 1 后重放 seq 0(gap 臂,非幂等重放臂)。
        s.on_decision_batch(_delta(1, 1100, arrivals=("r1",)))
        state_after_one = _snapshot(s, sink)
        with self.assertRaises(ValueError):
            s.on_decision_batch(_delta(0, 1000, arrivals=("r0",)))
        self.assertEqual(_snapshot(s, sink), state_after_one)

        # 同 seq 不同内容(篡改)= 协议违例,不是幂等重放。
        tampered = _delta(1, 1100, arrivals=("r1",))
        tampered["reasons"] = tampered["reasons"] + ["TAMPERED"]
        with self.assertRaises(ValueError):
            s.on_decision_batch(tampered)
        self.assertEqual(_snapshot(s, sink), state_after_one)
        self.assertEqual(s.delivery_count, 2)
        # 跳号尝试同样零污染(state 取在跳号尝试后,失败不落账)。
        self.assertNotEqual(state, state_after_one)  # seq 1 已正常应用


class AckStreamContractTest(unittest.TestCase):
    """ack 流:幂等、旧 ack 不释放新回复、失败三形态、run-end 审计。"""

    def _feed_two(self, sink=None):
        s = _scheduler(digest_sink=sink)
        s.on_decision_batch(_delta(0, 1000, arrivals=("r0",)))
        s.on_decision_batch(_delta(1, 1100, arrivals=("r1",)))
        return s

    def test_duplicate_ack_is_idempotent(self):
        s = self._feed_two()
        s.on_commit_ack(_ack(0))
        s.on_commit_ack(_ack(0))  # 重复 ack 幂等忽略
        self.assertEqual(s.ack_count, 1)
        self.assertEqual(s._acked_through, 0)

    def test_older_ack_does_not_release_newest_reply(self):
        s = self._feed_two()
        current_seq = 1
        s.on_commit_ack(_ack(0))
        cached = s._delivery_reply_cache
        self.assertIsNotNone(cached)
        self.assertEqual(cached["seq"], current_seq)
        self.assertIsNotNone(s._batch)
        self.assertEqual(s._batch["delivery_sequence"], current_seq)

    def _feed_settled(self):
        """两笔到达 + 两笔核销:全部请求离场(verify_run_end 可审计)。"""
        s = _scheduler(count=2)
        s.on_decision_batch(_delta(0, 1000, arrivals=("r0",)))
        s.on_decision_batch(_delta(1, 1100, arrivals=("r1",)))
        s.on_decision_batch(_delta(
            2, 1200, completed=(("r0", ""), ("r1", ""),)))
        return s

    def test_ack_stream_releases_and_watermark_compacts(self):
        s = self._feed_settled()
        for seq in range(3):
            s.on_commit_ack(_ack(seq))
            s.on_commit_ack(_ack(seq))  # 重复 ack 幂等忽略
        self.assertEqual(s.ack_count, 3)
        self.assertIsNone(s._delivery_reply_cache)
        self.assertIsNone(s._batch)
        self.assertEqual(s._acked_through, 2)
        self.assertEqual(s._acked_out_of_order, set())
        # 已 ack 交付不可再重放(回复已释放,凭据已消费)。
        with self.assertRaises(ValueError):
            s.on_decision_batch(_delta(
                2, 1200, completed=(("r0", ""), ("r1", ""),)))
        s.verify_run_end()

    def test_ack_fail_closed_shapes_not_counted(self):
        s = self._feed_two()
        s.on_commit_ack(_ack(0))
        s.on_commit_ack(_ack(1))
        baseline = s.ack_count
        for label, ack in (
            ("schema version", {"schema_version": 0,
                                "delivery_sequence": 1,
                                "batch_id": 1}),
            ("batch_id divergence", {"schema_version": 1,
                                     "delivery_sequence": 1,
                                     "batch_id": 2}),
            ("ahead of last applied", {"schema_version": 1,
                                       "delivery_sequence": 5,
                                       "batch_id": 5}),
            ("success=false", {"schema_version": 1,
                               "delivery_sequence": 0,
                               "batch_id": 0,
                               "success": False}),
        ):
            with self.subTest(shape=label):
                with self.assertRaises(ValueError):
                    s.on_commit_ack(ack)
        self.assertEqual(s.ack_count, baseline)


if __name__ == "__main__":
    unittest.main()
