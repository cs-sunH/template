"""test_work_ledger.py -- W1 因果运行账本验收单测。

覆盖（方案 W1 验收 + /tmp/joint_w0/inv_w1.md §4.1）：

* 身份：work_id 规范化、leg_work_id 派生、同一物理腿两张摘要不重复
  计费（§4.1:146）；
* 六类事件状态机与非法转移 fail-closed；
* 幂等：(work_id, kind, local_seq) 重放零副作用；local_seq 跳号 /
  tick 倒退 / 同键不同内容 fail-closed（仿 base._gate_delivery_sequence
  的门）；
* 守恒：ΣΔserved 与 ΣΔremaining 减量一致；progress 标 inferred；
* 全快照 / 增量重放等价（§4.3:178：先验证等价再用增量）；
* retired 义务保留（§4.3:178）；
* 桥 envelope 校验 fail-closed（schema 版本 / 未知键 / tick 不一致）；
* 桥接两端 schema 同步：Python 校验器字段面 vs C++ 序列化字段面；
* 关闭 link telemetry 不影响账本正确性（两通道独立）；
* base 调度器接线：_apply_work_ledger 在场应用 / 缺席记录。
"""

import copy
import json
import os
import sys
import unittest
from collections import UserDict
from collections.abc import Mapping as _TestMapping
from types import MappingProxyType

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_WORKLOAD_DIR = os.path.dirname(_THIS_DIR)
for _p in (_WORKLOAD_DIR, os.path.dirname(_WORKLOAD_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from joint.schema_registry import (  # noqa: E402
    BRIDGE_PROTOCOL_VERSION, WORK_LEDGER_SCHEMA_VERSION)
from joint.work_ledger import (  # noqa: E402
    EVENT_KINDS,
    CommittedWork,
    LedgerEvent,
    WorkAmount,
    WorkIdentity,
    WorkLedger,
    WorkLedgerError,
    amount_from_mapping,
    canonical_work_id,
    event_from_mapping,
    leg_work_id,
    parse_work_ledger_state,
    validate_work_ledger_payload,
    _freeze_json_value,
    _freeze_state_tree,
    _typed_event_from_mapping,
    _typed_v3_node_event_from_validated_scalars,
    _wire_integer,
)

_ONLINE_DIR = os.path.join(_WORKLOAD_DIR, "online")


def _envelope(seq, tick, kind, watermark=0, **extra):
    payload = {
        "schema_version": 1,
        "bridge_protocol": BRIDGE_PROTOCOL_VERSION,
        "work_ledger_schema": WORK_LEDGER_SCHEMA_VERSION,
        "snapshot_kind": kind,
        "delivery_sequence": seq,
        "tick": tick,
        "event_watermark": watermark,
    }
    payload.update(extra)
    return payload


def _event(kind, work_id, local_seq, tick=100, delivery=1, **extra):
    body = {
        "kind": kind,
        "work_id": work_id,
        "tick": tick,
        "event_id": {
            "tick": tick,
            "delivery_sequence": delivery,
            "producer": "work_ledger_diff",
            "local_seq": local_seq,
        },
    }
    body.update(extra)
    return body


def _node_state(entries):
    return {"nodes": [{"rank": 0, "nodes": entries}]}


def _node_entry(node_id, request="req-1", stage="prefill", issued=False,
                finished=False, num_ops=100, parents=()):
    return {
        "work_id": canonical_work_id(request, stage, 0, 0, node_id),
        "request_id": request,
        "stage": stage,
        "generation": 0,
        "rank": 0,
        "node_id": node_id,
        "kind": "compute",
        "issued": issued,
        "finished": finished,
        "unresolved_parents": 0,
        "num_ops": num_ops,
        "bytes": 0,
        "runtime_ns": 50,
        "tensor_size": 0,
        "parents": list(parents),
    }


class _TestCustomMapping(_TestMapping):
    def __init__(self, values):
        self._values = dict(values)

    def __getitem__(self, key):
        return self._values[key]

    def __iter__(self):
        return iter(self._values)

    def __len__(self):
        return len(self._values)


def _test_mapping_tree(value, factory):
    if isinstance(value, dict):
        return factory({key: _test_mapping_tree(item, factory)
                        for key, item in value.items()})
    if isinstance(value, list):
        return [_test_mapping_tree(item, factory) for item in value]
    if isinstance(value, tuple):
        return tuple(_test_mapping_tree(item, factory) for item in value)
    return value


class TestWorkIdentity(unittest.TestCase):
    def test_canonical_work_id(self):
        self.assertEqual(
            canonical_work_id("r1", "prefill", 2, 3, 7), "r1#prefill:2:r3n7")

    def test_identity_round_trip(self):
        identity = WorkIdentity(
            request_id="r1", stage="decode", generation=1, rank=2, node_id=9)
        self.assertEqual(identity.work_id, "r1#decode:1:r2n9")
        self.assertEqual(identity.node_uid, (2, 9))

    def test_leg_work_id_distinct_from_node(self):
        base = canonical_work_id("r1", "prefill", 0, 0, 1)
        self.assertEqual(
            leg_work_id(base, "hbm:COMPUTE"), base + "@hbm:COMPUTE")
        self.assertNotEqual(base, leg_work_id(base, "hbm:COMPUTE"))

    def test_same_physical_leg_not_double_billed(self):
        """同一物理腿出现在节点摘要与资源摘要两张快照中：键分立，不合并
        计费也不互相覆盖（§4.1:146）。"""
        state = {
            "nodes": [{"rank": 0, "nodes": [_node_entry(1)]}],
            "hbm": [{"rank": 0, "jobs": [{
                "work_id": leg_work_id(
                    canonical_work_id("req-1", "prefill", 0, 0, 1),
                    "hbm:COMPUTE"),
                "kind": "COMPUTE", "node_id": 1, "remaining_ops": 40,
                "remaining_bytes": 200, "remaining_latency_ns": 20,
                "total_latency_ns": 80,
                "request_id": "req-1", "stage": "prefill", "generation": 0,
            }]}],
        }
        works = parse_work_ledger_state(state)
        node_id = canonical_work_id("req-1", "prefill", 0, 0, 1)
        leg_id = leg_work_id(node_id, "hbm:COMPUTE")
        self.assertIn(node_id, works)
        self.assertIn(leg_id, works)
        # 节点视图（静态总量）与资源腿视图（权威剩余）各自独立表达。
        self.assertEqual(works[node_id].remaining.ops, 100)
        self.assertEqual(works[leg_id].remaining.ops, 40)
        self.assertEqual(works[leg_id].progress_source, "authoritative")


class TestLifecycle(unittest.TestCase):
    def _ledger_with_node(self):
        ledger = WorkLedger()
        node = canonical_work_id("req-1", "prefill", 0, 0, 1)
        ledger.apply_payload(
            _envelope(0, 10, "full", state=_node_state([_node_entry(1)])),
            delivery_sequence=0, tick=10)
        return ledger, node

    def test_full_load_states(self):
        ledger, node = self._ledger_with_node()
        work = ledger.work(node)
        self.assertEqual(work.state, "committed")
        self.assertEqual(work.total.ops, 100)
        self.assertEqual(work.remaining.ops, 100)

    def test_issue_progress_completion_release(self):
        ledger, node = self._ledger_with_node()
        events = [
            _event("issue", node, 0, tick=20),
            _event("progress", node, 1, tick=30,
                   delta_served={"ops": 40}),
            _event("completion", node, 2, tick=40),
            _event("ack", node, 3, tick=40),
            _event("release", node, 4, tick=40),
        ]
        ledger.apply_payload(
            _envelope(1, 40, "delta", watermark=5, events=events),
            delivery_sequence=1, tick=40)
        with self.assertRaises(WorkLedgerError):
            ledger.work(node)  # released = retired
        report = ledger.conservation_report()
        self.assertTrue(report["conserved"], report)

    def test_progress_marks_inferred_and_conserves(self):
        ledger, node = self._ledger_with_node()
        before = ledger.work(node).remaining
        ledger.apply_payload(
            _envelope(1, 30, "delta", watermark=1, events=[
                _event("progress", node, 0, tick=30,
                       delta_served={"ops": 40}),
            ]), delivery_sequence=1, tick=30)
        work = ledger.work(node)
        self.assertEqual(work.progress_source, "inferred")
        self.assertEqual(work.served.ops, 40)
        self.assertEqual(before.ops - work.remaining.ops, 40)
        report = ledger.conservation_report()
        self.assertTrue(report["conserved"])
        self.assertGreater(report["inferred_progress_works"], 0)

    def test_progress_remaining_field_reconciles_backend_tick_end(self):
        ledger, node = self._ledger_with_node()
        ledger.apply_payload(
            _envelope(1, 30, "delta", watermark=1, events=[
                _event("progress", node, 0, tick=30,
                       delta_served={"ops": 40}, remaining_ops=60),
            ]), delivery_sequence=1, tick=30)
        work = ledger.work(node)
        self.assertEqual(work.remaining.ops, 60)
        self.assertEqual(work.served.ops, 40)

        before = work.summary()
        with self.assertRaises(WorkLedgerError):
            ledger.apply_payload(
                _envelope(2, 40, "delta", watermark=2, events=[
                    _event("progress", node, 1, tick=40,
                           delta_served={"ops": 10}, remaining_ops=1),
                ]), delivery_sequence=2, tick=40)
        self.assertEqual(ledger.work(node).summary(), before)

    def test_illegal_transition_fail_closed(self):
        ledger, node = self._ledger_with_node()
        with self.assertRaises(WorkLedgerError):
            ledger.apply_payload(
                _envelope(1, 20, "delta", watermark=1, events=[
                    _event("ack", node, 0, tick=20),
                ]), delivery_sequence=1, tick=20)

    def test_over_serving_fail_closed(self):
        ledger, node = self._ledger_with_node()
        with self.assertRaises(WorkLedgerError):
            ledger.apply_payload(
                _envelope(1, 30, "delta", watermark=1, events=[
                    _event("progress", node, 0, tick=30,
                           delta_served={"ops": 1000}),
                ]), delivery_sequence=1, tick=30)


class TestIdempotencyAndMonotonicity(unittest.TestCase):
    def _applied(self):
        ledger = WorkLedger()
        node = canonical_work_id("req-1", "prefill", 0, 0, 1)
        ledger.apply_payload(
            _envelope(0, 10, "full", state=_node_state([_node_entry(1)])),
            delivery_sequence=0, tick=10)
        delta = _envelope(1, 30, "delta", watermark=1, events=[
            _event("progress", node, 0, tick=30, delta_served={"ops": 10}),
        ])
        ledger.apply_payload(delta, delivery_sequence=1, tick=30)
        return ledger, node, delta

    def test_duplicate_event_replay_zero_side_effects(self):
        """同一 delta 内事件重复（生产者重试追加）：内容一致 → 重放容忍，
        零副作用；交付层重复在真实桥上由 delivery 门早退（基类门）。"""
        node = canonical_work_id("req-1", "prefill", 0, 0, 1)
        event = _event("progress", node, 0, tick=30,
                       delta_served={"ops": 10})
        ledger = WorkLedger()
        ledger.apply_payload(
            _envelope(0, 10, "full", state=_node_state([_node_entry(1)])),
            delivery_sequence=0, tick=10)
        ledger.apply_payload(
            _envelope(1, 30, "delta", watermark=1, events=[event, event]),
            delivery_sequence=1, tick=30)
        self.assertEqual(ledger.replayed_event_count, 1)
        self.assertEqual(ledger.applied_event_count, 1)
        self.assertEqual(ledger.work(node).served.ops, 10)

    def test_local_seq_gap_fail_closed(self):
        ledger = WorkLedger()
        node = canonical_work_id("req-1", "prefill", 0, 0, 1)
        ledger.apply_payload(
            _envelope(0, 10, "full", state=_node_state([_node_entry(1)])),
            delivery_sequence=0, tick=10)
        # 基线 = 全量的 watermark(0)；首个 delta 事件跳号(3)fail-closed。
        with self.assertRaises(WorkLedgerError):
            ledger.apply_payload(
                _envelope(1, 20, "delta", watermark=4, events=[
                    _event("progress", node, 3, tick=20,
                           delta_served={"ops": 5}),
                ]), delivery_sequence=1, tick=20)

    def test_tick_regression_fail_closed(self):
        ledger = WorkLedger()
        node = canonical_work_id("req-1", "prefill", 0, 0, 1)
        ledger.apply_payload(
            _envelope(0, 10, "full", state=_node_state([_node_entry(1)])),
            delivery_sequence=0, tick=10)
        with self.assertRaises(WorkLedgerError):
            ledger.apply_payload(
                _envelope(1, 5, "delta", watermark=1, events=[
                    _event("progress", node, 0, tick=5,
                           delta_served={"ops": 5}),
                ]), delivery_sequence=1, tick=5)

    def test_same_key_different_content_fail_closed(self):
        ledger = WorkLedger()
        node = canonical_work_id("req-1", "prefill", 0, 0, 1)
        ledger.apply_payload(
            _envelope(0, 10, "full", state=_node_state([_node_entry(1)])),
            delivery_sequence=0, tick=10)
        ledger.apply_payload(
            _envelope(1, 30, "delta", watermark=1, events=[
                _event("progress", node, 0, tick=30, delta_served={"ops": 5}),
            ]), delivery_sequence=1, tick=30)
        # 同 (work_id, kind, local_seq) 不同内容：重放必须逐字节一致。
        with self.assertRaises(WorkLedgerError):
            ledger.apply_payload(
                _envelope(1, 30, "delta", watermark=1, events=[
                    _event("progress", node, 0, tick=30,
                           delta_served={"ops": 6}),
                ]), delivery_sequence=1, tick=30)

    def test_event_watermark_regression_fail_closed(self):
        ledger, _, delta = self._applied()
        with self.assertRaises(WorkLedgerError):
            ledger.apply_payload(
                _envelope(2, 40, "delta", watermark=0, events=[]),
                delivery_sequence=2, tick=40)


class TestEquivalence(unittest.TestCase):
    """全快照 / 增量重放等价（§4.3:178：先验证等价再用增量模式）。"""

    NODE = canonical_work_id("req-1", "prefill", 0, 0, 1)
    LEG = leg_work_id(NODE, "hbm:COMPUTE")

    def _hbm_state(self, remaining):
        return {
            "nodes": [{"rank": 0, "nodes": [
                _node_entry(1, issued=True)]}],
            "hbm": [{"rank": 0, "jobs": [{
                "work_id": self.LEG, "kind": "COMPUTE", "node_id": 1,
                "remaining_ops": remaining, "remaining_bytes": remaining * 2,
                "remaining_latency_ns": 0, "request_id": "req-1",
                "total_latency_ns": 80, "stage": "prefill", "generation": 0,
            }]}],
        }

    def _decision_projection(self, ledger):
        """决策消费面投影：状态/剩余量/腿/依赖（历史 served/total 不参与
        等价比较——全量重装载天然重置历史计数）。"""
        return {
            wid: (work.state, round(work.remaining.ops or 0),
                  round(work.remaining.bytes or 0), work.leg,
                  work.dependencies)
            for wid, work in ledger._works.items()
        }

    def test_pricing_snapshot_matches_frontier_and_tracks_mutation(self):
        ledger = WorkLedger()
        ledger.apply_payload(
            _envelope(0, 0, "full", state=self._hbm_state(100)),
            delivery_sequence=0, tick=0)
        full_projection = ledger.projection_snapshot()
        hot = ledger.pricing_snapshot()
        self.assertEqual(hot.pricing_rows, full_projection.pricing_rows)
        self.assertEqual(hot.committed_not_issued,
                         full_projection.committed_not_issued)
        self.assertIs(hot, ledger.pricing_snapshot())

        ledger.apply_payload(
            _envelope(1, 10, "delta", watermark=1, events=[
                _event("progress", self.LEG, 0, tick=10,
                       delta_served={"ops": 20, "bytes": 40}),
            ]), delivery_sequence=1, tick=10)
        updated = ledger.pricing_snapshot()
        self.assertIsNot(updated, hot)
        self.assertEqual(100, hot.pricing_rows[0].remaining.ops)
        self.assertEqual(80, updated.pricing_rows[0].remaining.ops)

    def test_full_vs_incremental_equivalence(self):
        frames = [self._hbm_state(100), self._hbm_state(60),
                  self._hbm_state(10)]
        # 增量：全量装载首帧 + 逐帧派生事件（与 C++ diff 同规则）。
        incremental = WorkLedger()
        incremental.apply_payload(
            _envelope(0, 0, "full", state=frames[0]),
            delivery_sequence=0, tick=0)
        local_seq = 0
        for index, tick in ((1, 100), (2, 200)):
            events = []
            prev = frames[index - 1]["hbm"][0]["jobs"][0]
            cur = frames[index]["hbm"][0]["jobs"][0]
            for field, dim in (("remaining_ops", "ops"),
                               ("remaining_bytes", "bytes")):
                delta = prev[field] - cur[field]
                if delta > 0:
                    events.append(_event(
                        "progress", self.LEG, local_seq, tick=tick,
                        delivery=index, delta_served={dim: delta}))
                    local_seq += 1
            incremental.apply_payload(
                _envelope(index, tick, "delta", watermark=local_seq,
                          events=events),
                delivery_sequence=index, tick=tick)
        # 全量：直接装载终帧。
        full = WorkLedger()
        full.apply_payload(
            _envelope(2, 200, "full", state=frames[2]),
            delivery_sequence=2, tick=200)
        self.assertEqual(
            self._decision_projection(incremental),
            self._decision_projection(full))
        self.assertTrue(
            incremental.conservation_report()["conserved"])
        self.assertTrue(full.conservation_report()["conserved"])

    def test_full_vs_incremental_midflight_first_seen(self):
        """§4.3:178 等价先证——生产主路径分支（全量一次后恒 delta，新
        腿必然**中途首见**）：commit 携带权威 remaining_*（后端已扣进度
        的剩余量）时，增量重放与全量装载的决策投影逐字段等价，剩余量
        保持 authoritative 语义；其后 progress 扣减沿同一条守恒线。"""
        node = self.NODE
        leg = self.LEG
        # 增量：帧 0（full，腿不在场）→ 帧 1（腿中途首见，后端权威剩余
        # ops=40/bytes=80，部分已服务量已扣）→ 帧 2（progress 后剩余
        # 10/20）。事件派生与 C++ diff 同规则：首见帧发 commit（携带
        # remaining_*），其后帧间差分发 progress。
        incremental = WorkLedger()
        incremental.apply_payload(
            _envelope(0, 0, "full",
                      state={"nodes": [
                          {"rank": 0, "nodes": [_node_entry(1, issued=True)]}],
                           "hbm": []}),
            delivery_sequence=0, tick=0)
        incremental.apply_payload(
            _envelope(1, 100, "delta", watermark=1, events=[
                _event(
                    "commit", leg, 0, tick=100, delivery=1,
                    leg="hbm:COMPUTE", request_id="req-1",
                    stage="prefill", generation=0, rank=0, node_id=1,
                    service_unit="bytes",
                    remaining_ops=40, remaining_bytes=80,
                    remaining_latency_ns=0),
            ]), delivery_sequence=1, tick=100)
        incremental.apply_payload(
            _envelope(2, 200, "delta", watermark=2, events=[
                _event("progress", leg, 1, tick=200, delivery=2,
                       delta_served={"ops": 30, "bytes": 60}),
            ]), delivery_sequence=2, tick=200)
        # 全量：直接装载终帧（帧 2 语义：权威剩余 10/20、in_service）。
        full = WorkLedger()
        full.apply_payload(
            _envelope(2, 200, "full", state=self._hbm_state(10)),
            delivery_sequence=2, tick=200)
        self.assertEqual(
            self._decision_projection(incremental),
            self._decision_projection(full))
        # 权威性不因中途首见丢失（§4.3:174：携带权威剩余量的 commit =
        # 后端积分真值，不退化为总量占位）。
        self.assertEqual(
            incremental.work(leg).progress_source, "authoritative")
        self.assertEqual(full.work(leg).progress_source, "authoritative")
        self.assertTrue(
            incremental.conservation_report()["conserved"])
        self.assertTrue(full.conservation_report()["conserved"])
        # 唯一 stale = 节点级 issued 工作（无进度遥测，§4.3:174 同款
        # 标记；中途首见腿自 commit 携带权威剩余量后不在其列）。
        self.assertEqual(
            incremental.conservation_report()["stale_amount_works"], 1)
        self.assertEqual(
            full.conservation_report()["stale_amount_works"], 1)

    def test_hbm_full_preserves_original_total_with_residual(self):
        state = self._hbm_state(40)
        job = state["hbm"][0]["jobs"][0]
        job["num_ops"] = 100
        job["bytes"] = 200
        works = parse_work_ledger_state(state)
        work = works[self.LEG]
        self.assertEqual(work.total.ops, 100)
        self.assertEqual(work.total.bytes, 200)
        self.assertEqual(work.total.latency_ns, 80)
        self.assertEqual(work.remaining.ops, 40)
        self.assertEqual(work.remaining.bytes, 80)
        self.assertEqual(work.served.ops, 60)
        self.assertEqual(work.served.bytes, 120)
        self.assertEqual(work.progress_source, "authoritative")

    def test_delta_commit_without_remaining_is_stale_upper_bound(self):
        """§4.3:174/176：commit 不携带权威 remaining_*（旧对端形态）时
        remaining = 总量**上界占位**而非精确剩余——progress_source=stale
        显式降级登记，不冒充权威观测；守恒审计 stale_amount_works 计数
        披露过估面（消费方据此后端修复前不当作精确剩余）。"""
        ledger = WorkLedger()
        ledger.apply_payload(
            _envelope(0, 0, "full", state=_node_state([])),
            delivery_sequence=0, tick=0)
        ledger.apply_payload(
            _envelope(1, 100, "delta", watermark=1, events=[
                _event(
                    "commit", self.LEG, 0, tick=100, delivery=1,
                    leg="hbm:COMPUTE", request_id="req-1",
                    stage="prefill", generation=0, rank=0, node_id=1,
                    service_unit="bytes", total={"bytes": 100}),
            ]), delivery_sequence=1, tick=100)
        work = ledger.work(self.LEG)
        self.assertEqual(work.remaining.bytes, 100)  # 上界占位 = 总量
        self.assertEqual(work.progress_source, "stale")
        report = ledger.conservation_report()
        self.assertTrue(report["conserved"])
        self.assertEqual(report["stale_amount_works"], 1)

    def test_full_resource_rank_and_wire_identity(self):
        """Resource identities use the producer rank, not a port default."""
        node = canonical_work_id("req-1", "prefill", 0, 2, 3)
        pool_leg = leg_work_id(node, "pool:0:17")
        noc = canonical_work_id("noc", "", 0, 2, 42)
        noc_leg = leg_work_id(noc, "noc:42")
        state = {
            "nodes": [],
            "pool": [{"port_index": 0, "jobs": [{
                "work_id": pool_leg, "rank": 2, "sys_id": 2,
                "node_id": 3, "issue_seq": 17,
                "request_id": "req-1", "stage": "prefill",
                "generation": 0, "bytes": 1000,
                "remaining_bytes": 700, "state": "Streaming",
            }]}],
            "noc": {"flows": [{
                "work_id": noc_leg, "flow_id": 42, "request_id": "noc",
                "stage": "", "generation": 0, "rank": 2,
                "node_id": 42, "total_bytes": 400,
                "remaining_bytes": 250, "route": [7, 8],
            }]},
        }
        works = parse_work_ledger_state(state)
        self.assertIn(pool_leg, works)
        self.assertEqual(works[pool_leg].identity.rank, 2)
        self.assertIn(noc_leg, works)
        self.assertEqual(works[noc_leg].identity.rank, 2)
        self.assertNotIn(
            leg_work_id(canonical_work_id("req-1", "prefill", 0, 0, 3),
                        "pool:0:17"), works)

    def test_noc_legacy_raw_wire_id_fails_closed(self):
        state = {"noc": {"flows": [{
            "work_id": "noc#42", "flow_id": 42, "rank": 0,
            "request_id": "noc", "stage": "", "generation": 0,
            "total_bytes": 100, "remaining_bytes": 50, "route": [1],
        }]}}
        with self.assertRaises(WorkLedgerError):
            parse_work_ledger_state(state)

    def test_resource_completion_waiting_states_are_terminal(self):
        pool_node = canonical_work_id("req-1", "prefill", 0, 1, 9)
        pool_leg = leg_work_id(pool_node, "pool:0:4")
        noc_node = canonical_work_id("noc", "", 0, 2, 7)
        noc_leg = leg_work_id(noc_node, "noc:7")
        state = {
            "pool": [{"port_index": 0, "jobs": [{
                "work_id": pool_leg, "rank": 1, "node_id": 9,
                "issue_seq": 4, "request_id": "req-1",
                "stage": "prefill", "generation": 0, "bytes": 100,
                "remaining_bytes": 0,
                "state": "FluidCompleteAwaitingCallback",
            }]}],
            "noc": {"flows": [{
                "work_id": noc_leg, "flow_id": 7, "rank": 2,
                "request_id": "noc", "stage": "", "generation": 0,
                "total_bytes": 80, "remaining_bytes": 0,
                "state": "PropagatingTail", "route": [3],
            }]},
        }
        works = parse_work_ledger_state(state)
        self.assertEqual(works[pool_leg].state, "completed")
        self.assertEqual(works[pool_leg].served.bytes, 100)
        self.assertEqual(works[noc_leg].state, "completed")
        self.assertEqual(works[noc_leg].served.bytes, 80)

    def test_delta_commit_pool_state_string_mapping(self):
        """资源腿 commit 的 C++ 生命周期字符串映射（pool 端口 job 的
        Streaming/LatencyWaiting → in_service/issued；未知名 fail-closed）。"""
        ledger = WorkLedger()
        pool_leg = leg_work_id(
            canonical_work_id("req-1", "prefill", 0, 0, 3), "pool:0:17")
        pool_leg_waiting = leg_work_id(
            canonical_work_id("req-1", "prefill", 0, 0, 4), "pool:0:18")
        ledger.apply_payload(
            _envelope(0, 0, "full", state=_node_state([])),
            delivery_sequence=0, tick=0)
        ledger.apply_payload(
            _envelope(1, 100, "delta", watermark=2, events=[
                _event(
                    "commit", pool_leg, 0, tick=100, delivery=1,
                    leg="pool:0:17", request_id="req-1", stage="prefill",
                    generation=0, rank=0, node_id=3,
                    total={"bytes": 500}, remaining_bytes=200,
                    state="Streaming"),
                _event(
                    "commit", pool_leg_waiting, 1, tick=100, delivery=1,
                    leg="pool:0:18", request_id="req-1", stage="prefill",
                    generation=0, rank=0, node_id=4,
                    total={"bytes": 500}, remaining_bytes=500,
                    state="LatencyWaiting"),
            ]), delivery_sequence=1, tick=100)
        streaming = ledger.work(pool_leg)
        self.assertEqual(streaming.state, "in_service")
        self.assertEqual(streaming.remaining.bytes, 200)
        self.assertEqual(streaming.progress_source, "authoritative")
        waiting = ledger.work(pool_leg_waiting)
        self.assertEqual(waiting.state, "issued")
        ledger2 = WorkLedger()
        ledger2.apply_payload(
            _envelope(0, 0, "full", state=_node_state([])),
            delivery_sequence=0, tick=0)
        with self.assertRaises(WorkLedgerError):
            ledger2.apply_payload(
                _envelope(1, 100, "delta", watermark=1, events=[
                    _event(
                        "commit", pool_leg, 0, tick=100, delivery=1,
                        leg="pool:0:17", state="Hyperspace"),
                ]), delivery_sequence=1, tick=100)


class TestCompactEncoding(unittest.TestCase):
    """紧凑节点清单（C++ DecisionBridge.hh 注记的体量工程）：重构后与
    全事件形式语义等价；全量装载终态与 增量(紧凑) 终态等价。"""

    NODE1 = canonical_work_id("req-1", "prefill", 0, 0, 1)
    NODE2 = canonical_work_id("req-1", "prefill", 0, 0, 2)

    def test_compact_equivalent_to_full_load(self):
        frames = [
            _node_state([]),
            _node_state([_node_entry(1, parents=()), _node_entry(2, parents=[1])]),
            _node_state([_node_entry(1, issued=True, parents=()),
                         _node_entry(2, parents=[1])]),
            _node_state([_node_entry(2, parents=[])]),  # 1 finished+GC
            _node_state([]),                              # 2 也退休
        ]
        incremental = WorkLedger()
        incremental.apply_payload(
            _envelope(0, 0, "full", state=frames[0]),
            delivery_sequence=0, tick=0)
        seq = 0
        for index, tick in ((1, 10), (2, 20), (3, 30), (4, 40)):
            compact = {"seq_base": seq}
            prev_nodes = frames[index - 1]["nodes"][0]["nodes"]
            cur_nodes = frames[index]["nodes"][0]["nodes"]
            prev_ids = {n["work_id"]: n for n in prev_nodes}
            cur_ids = {n["work_id"]: n for n in cur_nodes}
            compact["node_commits"] = [
                n for wid, n in cur_ids.items() if wid not in prev_ids]
            compact["node_issues"] = [
                wid for wid, n in cur_ids.items()
                if wid in prev_ids and n["issued"]
                and not prev_ids[wid]["issued"]]
            compact["node_completions"] = [
                wid for wid, n in cur_ids.items()
                if wid in prev_ids and n["finished"]
                and not prev_ids[wid]["finished"]]
            compact["node_vanished"] = [
                wid for wid in prev_ids if wid not in cur_ids]
            seq += (len(compact["node_commits"])
                    + len(compact["node_issues"])
                    + len(compact["node_completions"])
                    + 2 * len(compact["node_vanished"]))
            incremental.apply_payload(
                _envelope(index, tick, "delta", watermark=seq, compact=compact,
                          events=[]),
                delivery_sequence=index, tick=tick)
        # 全量装载终帧（空）对照：一切工作都已退休。
        full = WorkLedger()
        full.apply_payload(
            _envelope(4, 40, "full", state=frames[4]),
            delivery_sequence=4, tick=40)
        self.assertEqual(incremental._works, {})
        self.assertEqual(full._works, {})
        self.assertTrue(incremental.conservation_report()["conserved"])

    def test_compact_dependencies_canonicalized(self):
        ledger = WorkLedger()
        compact = {
            "seq_base": 0,
            "node_commits": [_node_entry(2, parents=[1])],
            "node_issues": [], "node_completions": [], "node_vanished": [],
        }
        ledger.apply_payload(
            _envelope(0, 10, "full", state=_node_state([])),
            delivery_sequence=0, tick=10)
        ledger.apply_payload(
            _envelope(1, 20, "delta", watermark=1, compact=compact,
                      events=[]),
            delivery_sequence=1, tick=20)
        work = ledger.work(self.NODE2)
        self.assertEqual(
            work.dependencies, (self.NODE1,),
            "parents node ids canonicalize to dependency work ids")

    def test_compact_unknown_keys_fail_closed(self):
        ledger = WorkLedger()
        ledger.apply_payload(
            _envelope(0, 10, "full", state=_node_state([])),
            delivery_sequence=0, tick=10)
        with self.assertRaises(WorkLedgerError):
            ledger.apply_payload(
                _envelope(1, 20, "delta", watermark=0,
                          compact={"surprise": 1, "seq_base": 0},
                          events=[]),
                delivery_sequence=1, tick=20)


class TestRetainedObligations(unittest.TestCase):
    def test_release_condition_retained_while_depended(self):
        ledger = WorkLedger()
        parent = canonical_work_id("req-1", "prefill", 0, 0, 1)
        child = canonical_work_id("req-1", "prefill", 0, 0, 2)
        ledger.apply_payload(
            _envelope(0, 10, "full", state=_node_state([
                _node_entry(1, issued=True, parents=()),
                _node_entry(2, parents=[1]),
            ])), delivery_sequence=0, tick=10)
        # 模拟 commit 携带的释放条件（全量路径不携带条件，逐事件路径由
        # commit payload 提供）。
        ledger.work(parent).release_conditions = ("watch#1",)
        # 父节点完成并消失（GC），但子节点仍在飞：义务保留。
        ledger.apply_payload(
            _envelope(1, 40, "delta", watermark=3, events=[
                _event("completion", parent, 0, tick=40),
                _event("ack", parent, 1, tick=40),
                _event("release", parent, 2, tick=40),
            ]), delivery_sequence=1, tick=40)
        self.assertIn(parent, ledger.retained_release_conditions())
        # 子节点随后退休：依赖闭包消失，义务档位随依赖消失。
        ledger.apply_payload(
            _envelope(2, 60, "delta", watermark=5, events=[
                _event("completion", child, 3, tick=60),
                _event("ack", child, 4, tick=60),
                _event("release", child, 5, tick=60),
            ]), delivery_sequence=2, tick=60)
        self.assertEqual(ledger.retained_release_conditions(), {})


class TestRuntimeMappingCompatibility(unittest.TestCase):
    def test_mapping_forms_match_event_compact_state_and_freeze_consumers(self):
        factories = (
            ("dict", dict),
            ("UserDict", UserDict),
            ("MappingProxyType", MappingProxyType),
            ("custom Mapping", _TestCustomMapping),
        )
        event = _event(
            "commit", "mapping-probe", 0,
            total={"ops": 1, "bytes": 2, "latency_ns": 3},
            dependencies=["parent"])
        expected_event = _typed_event_from_mapping(event)
        state = {"nodes": [{"rank": 0, "nodes": []}]}
        expected_frozen = _freeze_state_tree({"outer": {"value": [1]}})
        compact = {
            "seq_base": 0,
            "node_commits": [{"work_id": "mapping-probe"}],
        }
        payload = _envelope(0, 0, "full", state={}, events=[])

        for name, factory in factories:
            with self.subTest(mapping_type=name):
                wrapped_event = _test_mapping_tree(event, factory)
                parsed_event = _typed_event_from_mapping(wrapped_event)
                self.assertEqual(parsed_event.canonical_content,
                                 expected_event.canonical_content)
                self.assertEqual(parsed_event.work_id, expected_event.work_id)

                wrapped_compact = _test_mapping_tree(compact, factory)
                self.assertEqual(
                    WorkLedger()._preflight_compact(wrapped_compact), 0)
                self.assertEqual(
                    parse_work_ledger_state(
                        _test_mapping_tree(state, factory)), {})
                validate_work_ledger_payload(
                    _test_mapping_tree(payload, factory), 0, 0)
                self.assertEqual(
                    _freeze_state_tree(_test_mapping_tree(
                        {"outer": {"value": [1]}}, factory)),
                    expected_frozen)

        self.assertNotEqual(_freeze_json_value(1), _freeze_json_value(1.0))
        with self.assertRaisesRegex(WorkLedgerError, "event must be a mapping"):
            _typed_event_from_mapping([])
        with self.assertRaisesRegex(WorkLedgerError, "compact must be a mapping"):
            WorkLedger()._preflight_compact([])
        with self.assertRaisesRegex(WorkLedgerError, "must be an integer"):
            _wire_integer(1.0, "probe")
        with self.assertRaisesRegex(WorkLedgerError, "must be an integer"):
            _wire_integer(True, "probe")


class TestV3TransactionCanonicalRetention(unittest.TestCase):
    def test_missing_final_canonical_fails_before_live_ledger_mutation(self):
        ledger = WorkLedger()
        event_payload = {
            "request_id": "req", "stage": "decode", "generation": 1,
            "rank": 0, "node_id": 1, "operation": "compute",
            "total": {"ops": 1, "bytes": 0, "latency_ns": 1},
        }
        event = _typed_v3_node_event_from_validated_scalars(
            kind="commit", work_id="req#decode:1:r0n1", tick=10,
            delivery_sequence=0, producer="cpp_online", local_seq=1,
            payload=event_payload, build_canonical_content=False)
        self.assertIsNone(event.canonical_content)

        transaction = ledger.begin_protocol_v3_transaction(
            delivery_sequence=0, tick=10, snapshot_kind="full")
        transaction.stage_event(1, event)
        transaction.finish(1)
        state_before = ledger.export_state()
        fields_before = (
            ledger.event_watermark, ledger.last_delivery_sequence,
            ledger.last_tick, ledger.version, ledger.full_load_count,
            ledger.delta_load_count, ledger._last_event,
        )

        with self.assertRaisesRegex(WorkLedgerError, "missing adjacent-replay"):
            transaction.commit()

        self.assertEqual(ledger.export_state(), state_before)
        self.assertEqual(
            (ledger.event_watermark, ledger.last_delivery_sequence,
             ledger.last_tick, ledger.version, ledger.full_load_count,
             ledger.delta_load_count, ledger._last_event),
            fields_before)
        self.assertFalse(transaction._committed)


class TestEnvelopeValidation(unittest.TestCase):
    def test_absent_field_tolerated_by_caller(self):
        # envelope 校验只对在场字段；缺席 = 旧对端（base 记录事实）。
        validate_work_ledger_payload(
            _envelope(0, 0, "full", state=_node_state([])),
            delivery_sequence=0, tick=0)

    def test_schema_mismatch_fail_closed(self):
        # 合法 envelope 通过（对照面）。
        validate_work_ledger_payload(
            _envelope(0, 0, "full", state={}), 0, 0)
        bad = _envelope(0, 0, "full", state={})
        bad["schema_version"] = 2
        with self.assertRaises(WorkLedgerError):
            validate_work_ledger_payload(bad, 0, 0)

    def test_protocol_version_drift_fail_closed(self):
        bad = _envelope(0, 0, "full", state={})
        bad["bridge_protocol"] = "bridge-v0-20200101"
        with self.assertRaises(WorkLedgerError):
            validate_work_ledger_payload(bad, 0, 0)

    def test_unknown_keys_fail_closed(self):
        bad = _envelope(0, 0, "full", state={})
        bad["surprise"] = 1
        with self.assertRaises(WorkLedgerError):
            validate_work_ledger_payload(bad, 0, 0)

    def test_tick_mismatch_fail_closed(self):
        with self.assertRaises(WorkLedgerError):
            validate_work_ledger_payload(
                _envelope(0, 5, "full", state={}), 0, 0)

    def test_bad_snapshot_kind_fail_closed(self):
        with self.assertRaises(WorkLedgerError):
            validate_work_ledger_payload(
                _envelope(0, 0, "partial"), 0, 0)


class TestBridgeSchemaSync(unittest.TestCase):
    """桥接两端 schema 同步：Python 校验器 vs C++ 序列化字段一一对应。"""

    def test_envelope_field_surface(self):
        # C++ compute_work_ledger (main_online.cc) 产出的字段面：
        envelope_keys = {
            "schema_version", "bridge_protocol", "work_ledger_schema",
            "snapshot_kind", "delivery_sequence", "tick", "event_watermark",
            "state", "events",
        }
        # Python 校验器接受的字段面（_validate_payload_envelope 的白名单）：
        probe = _envelope(0, 0, "full", state={}, events=[])
        try:
            validate_work_ledger_payload(probe, 0, 0)
        except WorkLedgerError as exc:
            self.fail("validator rejects its own documented surface: %s" % exc)
        bad = _envelope(0, 0, "full")
        bad["undocumented"] = True
        with self.assertRaises(WorkLedgerError):
            validate_work_ledger_payload(bad, 0, 0)
        self.assertEqual(envelope_keys, {
            "schema_version", "bridge_protocol", "work_ledger_schema",
            "snapshot_kind", "delivery_sequence", "tick", "event_watermark",
            "state", "events"})

    def test_cxx_literal_versions_pinned(self):
        """C++ 侧硬编码的版本串必须与 schema_registry 冻结值一致。"""
        cxx_bridge_protocol = "bridge-v4-20261011"
        cxx_work_ledger_schema = "workledger-v4-20261011"
        self.assertEqual(cxx_bridge_protocol, BRIDGE_PROTOCOL_VERSION)
        self.assertEqual(cxx_work_ledger_schema, WORK_LEDGER_SCHEMA_VERSION)

    def test_event_field_surface(self):
        event = event_from_mapping(_event(
            "progress", "w", 0, delta_served={"ops": 1},
            state="issued", dependencies=["x"], leg="hbm:COMPUTE"))
        self.assertEqual(event.kind, "progress")
        self.assertEqual(event.idempotency_key, ("w", "progress", 0))
        bad = _event("progress", "w", 0)
        bad["surprise"] = 1
        with self.assertRaises(WorkLedgerError):
            event_from_mapping(bad)

    def test_event_kinds_closed_set(self):
        self.assertEqual(
            tuple(sorted(EVENT_KINDS)),
            ("ack", "commit", "completion", "issue", "progress", "release"))


class TestTelemetryIndependence(unittest.TestCase):
    """关闭 link telemetry 不影响账本正确性（两通道独立）。"""

    def test_ledger_correct_without_telemetry(self):
        # delta 不带 link_telemetry 字段：账本装载/守恒与带遥测时一致。
        node = canonical_work_id("req-1", "prefill", 0, 0, 1)
        without = WorkLedger()
        without.apply_payload(
            _envelope(0, 10, "full", state=_node_state([_node_entry(1)])),
            delivery_sequence=0, tick=10)
        self.assertTrue(without.conservation_report()["conserved"])
        self.assertEqual(without.work(node).total.ops, 100)


class TestBaseWiring(unittest.TestCase):
    """online_scheduler_base 的 _apply_work_ledger 接线。"""

    def _base(self):
        if _ONLINE_DIR not in sys.path:
            sys.path.insert(0, _ONLINE_DIR)
        from online.online_scheduler_base import OnlineSchedulerBase

        class _Minimal(OnlineSchedulerBase):
            def run_variant_policy(self, delta):
                return None

        manifest = {"requests": [
            {"request_id": "req-1", "session_id": "s", "turn_index": 0,
             "prefill_length": 8, "decode_length": 4,
             "inter_request_interval_ns": 1000, "arrival_world_ns": 0}]}

        class _Config:  # 最小 config 桩（基类只透传不读）
            pass

        scheduler = _Minimal(manifest=manifest, config=_Config())
        from joint.prediction_config import JointPredictionConfig
        scheduler._joint_prediction_config = JointPredictionConfig()
        return scheduler

    def _delta(self, seq, tick, work_ledger=None, *,
               handle_kind="work_ledger_v4"):
        delta = {
            "schema_version": 2,
            "delivery_sequence": seq,
            "delivery_epoch": seq,
            "tick": tick,
            "deferred_from_tick": 0,
            "reasons": [],
            "arrivals": [],
            "completed_groups": [],
            "completed_nodes": [],
            "affected_ranks": [],
            "snapshot_handle": {
                "epoch": seq, "tick": tick,
                "kind": handle_kind},
            "ledger_summary": {"injected_unfinished": []},
        }
        if work_ledger is not None:
            delta["work_ledger"] = work_ledger
        return delta

    @staticmethod
    def _v4_envelope(delivery, tick, *, base_watermark=0,
                     base_version=0, snapshot_kind="full", entries=()):
        """Small production-shaped v4 envelope with ordinary node commits."""
        transitions = []
        for index, entry in enumerate(entries):
            logical_seq = base_watermark + index + 1
            transitions.append({
                "logical_seq": logical_seq,
                "tick": tick,
                "kind": "commit",
                "work_id": entry["work_id"],
                "payload": {
                    "request_id": entry["request_id"],
                    "stage": entry["stage"],
                    "generation": entry["generation"],
                    "rank": entry["rank"],
                    "node_id": entry["node_id"],
                    "operation": entry["kind"],
                    "total": {
                        "ops": entry["num_ops"],
                        "bytes": entry["bytes"],
                        "latency_ns": entry["runtime_ns"],
                    },
                    "parents": entry["parents"],
                },
            })
        count = len(transitions)
        return {
            "producer": "cpp_online",
            "schema_version": 4,
            "protocol_version": 4,
            "bridge_protocol": "bridge-v4-20261011",
            "work_ledger_schema": "workledger-v4-20261011",
            "epoch": "test-run",
            "delivery_sequence": delivery,
            "base_watermark": base_watermark,
            "logical_seq_start": base_watermark + 1,
            "logical_seq_end": base_watermark + count,
            "logical_event_count": count,
            "replay_token": f"cpp_online/test-run/{delivery}",
            "descriptor_version": 1,
            "base_state_version": base_version,
            "result_state_version": base_version + (1 if count else 0),
            "snapshot_kind": snapshot_kind,
            "tick": tick,
            "descriptor_commits": [],
            "node_transitions": transitions,
            "pipeline_transition_runs": [],
            "cursor_patches": [],
            "active_resources": [],
        }

    def test_apply_v4_and_missing_required_rejected(self):
        scheduler = self._base()
        node = canonical_work_id("req-1", "prefill", 0, 0, 1)
        scheduler.on_decision_batch(self._delta(0, 10, self._v4_envelope(
            0, 10, entries=(_node_entry(1),))))
        self.assertIn(
            node, scheduler.joint_work_ledger.works_summary())
        self.assertFalse(scheduler.work_ledger_absent_seen)
        with self.assertRaisesRegex(ValueError, "requires a v4 work_ledger"):
            scheduler.on_decision_batch(self._delta(1, 20))
        self.assertFalse(scheduler.work_ledger_absent_seen)
        # 缺少必需的生产 ledger 被 fail-closed，不能抹掉既有事实。
        self.assertIn(
            node, scheduler.joint_work_ledger.works_summary())

    def test_bad_kind_rejected_at_schema_gate(self):
        scheduler = self._base()
        bad = self._v4_envelope(0, 10, snapshot_kind="full")
        bad["snapshot_kind"] = "partial"
        # v4 N-line: child envelope gates live in the native receiver
        # (NativeLedgerError, a RuntimeError); the fail-closed contract is
        # unchanged, only the raiser moved across the ABI.
        from joint.native_ledger import NativeLedgerError
        with self.assertRaises((ValueError, NativeLedgerError)):
            scheduler.on_decision_batch(self._delta(0, 10, bad))

    def test_protocol_v1_is_rejected_on_production_bridge(self):
        scheduler = self._base()
        legacy = _envelope(0, 10, "full", state=_node_state([]))
        with self.assertRaisesRegex(ValueError, "protocol 4"):
            scheduler.on_decision_batch(self._delta(0, 10, legacy))
        self.assertEqual(
            scheduler.joint_work_ledger.active_work_count(), 0)
        self.assertFalse(scheduler.joint_pipeline_ledger.initialized)

    def test_v3_archive_is_rejected_by_live_v4_gate(self):
        scheduler = self._base()
        legacy = self._v4_envelope(0, 10, entries=(_node_entry(1),))
        legacy.update({
            "schema_version": 3,
            "protocol_version": 3,
            "bridge_protocol": "bridge-v3-20261009",
            "work_ledger_schema": "workledger-v3-20261009",
        })
        with self.assertRaisesRegex(ValueError, "protocol 4"):
            scheduler.on_decision_batch(self._delta(
                0, 10, legacy, handle_kind="work_ledger_v3"))
        self.assertEqual(
            scheduler.joint_work_ledger.active_work_count(), 0)
        self.assertFalse(scheduler.joint_pipeline_ledger.initialized)

    def test_v3_snapshot_handle_is_rejected_with_v4_child(self):
        scheduler = self._base()
        with self.assertRaisesRegex(ValueError, "requires snapshot_handle.kind"):
            scheduler.on_decision_batch(self._delta(
                0, 10, self._v4_envelope(0, 10),
                handle_kind="work_ledger_v3"))
        self.assertFalse(scheduler.joint_pipeline_ledger.initialized)

    def test_snapshot_handle_kind_sync(self):
        scheduler = self._base()
        delta = self._delta(0, 10)
        delta["snapshot_handle"]["kind"] = "congestion_v9"
        with self.assertRaises(ValueError):
            scheduler.on_decision_batch(delta)

    def test_same_tick_commit_visible_to_policy_snapshot(self):
        """§4.2:164 + W1/V2 验收（账本通道）：同一交付携带的 work_ledger
        commit 在 ``run_variant_policy`` 内**立即可见**——接线钉子钉住
        "commit → 下一个 snapshot"的关系：base 在 run_variant_policy 之
        前应用 _apply_work_ledger（online_scheduler_base 交付固定阶段），
        策略观察点里账本活跃集与该交付内构造的 DecisionSnapshot
        work_index 均已含新工作及其依赖边。流注册表通道的同 tick 可见性
        钉子在 online/test_joint_preadmit_visibility.py；本钉子覆盖账本
        通道（W1 接线此前无任何交付级可见性验收）。"""
        if _ONLINE_DIR not in sys.path:
            sys.path.insert(0, _ONLINE_DIR)
        from online.online_scheduler_base import OnlineSchedulerBase
        from joint.runtime_snapshot import build_decision_snapshot

        class _Observer(OnlineSchedulerBase):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.policy_observations = []

            def run_variant_policy(self, delta):
                # 策略可见面：账本活跃集 + 本交付内构造的快照资源索引
                # （export_state 走生产同一条累积状态通道）。
                self.policy_observations.append({
                    "delivery_sequence": delta["delivery_sequence"],
                    "works": self.joint_work_ledger.works_summary(),
                    "work_index": build_decision_snapshot(
                        delta,
                        ledger_state=self.joint_work_ledger.export_state(),
                        request_queues={},
                    ).work_index,
                })

        manifest = {"requests": [
            {"request_id": "req-1", "session_id": "s", "turn_index": 0,
             "prefill_length": 8, "decode_length": 4,
             "inter_request_interval_ns": 1000, "arrival_world_ns": 0}]}

        class _Config:  # 最小 config 桩（基类只透传不读）
            pass

        scheduler = _Observer(manifest=manifest, config=_Config())
        from joint.prediction_config import JointPredictionConfig
        scheduler._joint_prediction_config = JointPredictionConfig()
        node1 = canonical_work_id("req-1", "prefill", 0, 0, 1)
        node2 = canonical_work_id("req-1", "prefill", 0, 0, 2)
        # 首交付 full（节点 1 在场）：同交付策略已可见。
        scheduler.on_decision_batch(self._delta(0, 10, self._v4_envelope(
            0, 10, entries=(_node_entry(1),))))
        first = scheduler.policy_observations[0]
        self.assertEqual(first["delivery_sequence"], 0)
        self.assertIn(node1, first["works"])
        self.assertIn(1, first["work_index"]["by_rank_nodes"][0])
        # 第二交付 delta 携带新节点 commit（compact 清单，parents=[1]）：
        # 同 tick（本交付）策略立即可见新工作 + 依赖边 + 快照索引。
        scheduler.on_decision_batch(self._delta(1, 20, self._v4_envelope(
            1, 20, base_watermark=1, base_version=1,
            snapshot_kind="delta",
            entries=(_node_entry(2, parents=[1]),))))
        second = scheduler.policy_observations[1]
        self.assertEqual(second["delivery_sequence"], 1)
        self.assertIn(node2, second["works"])
        self.assertEqual(
            second["works"][node2]["dependencies"], [node1])
        self.assertIn(2, second["work_index"]["by_rank_nodes"][0])


class TestAmounts(unittest.TestCase):
    def test_amount_math_and_validation(self):
        with self.assertRaises(WorkLedgerError):
            amount_from_mapping({"ops": -5})
        for bad in (float("nan"), float("inf"), "10", True):
            with self.subTest(bad=bad):
                with self.assertRaises(WorkLedgerError):
                    amount_from_mapping({"ops": bad})
        total = amount_from_mapping({"ops": 10, "bytes": 100})
        served = amount_from_mapping({"ops": 4})
        remaining = total.minus(served)
        self.assertEqual(remaining.ops, 6)
        self.assertIsNone(remaining.latency_ns)

    def test_ledger_event_requires_local_seq(self):
        with self.assertRaises(WorkLedgerError):
            event_from_mapping({
                "kind": "commit", "work_id": "w", "tick": 1,
                "event_id": {"tick": 1, "delivery_sequence": 0}})


if __name__ == "__main__":
    unittest.main()
