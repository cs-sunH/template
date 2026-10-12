#!/usr/bin/env python3
"""test_ledger_world_source.py -- W5 A 部接线单元测试。

覆盖（G-B 门 V2/V1 组的账本接线面）：
  1. LedgerWorldSource 投影：账本资源腿（hbm/pool/noc）→ 精确剩余量
     in-service TaskSpec 腿；issued-not-streaming → 非 in-service 流腿；
     committed-not-issued 节点工作 → 不虚构资源腿（计数披露）；
  2. 逐键降级：账本无权威腿的资源键回退注册表计数腿（unknown-ETA，
     approximation 披露）；有权威腿的键不与注册表双计；
  3. 无账本裸构造 fail-closed（不静默空背景）；
  4. 决策快照版本失效：账本版本/事件水位入 cache_key（版本推进 ⇒ 快
     照失效）；
  5. journal 重放保真（W1 INTEGRATION_NOTES §2 红线）：canonical 桥
     journal 剥离 work_ledger 字段 + gzip 侧车承载重放源——
     iter_request_records(attach_work_ledger=True) 重放前缀得到的账本
     状态与原跑逐字段一致；缺侧车行的重放 fail-closed。

运行：cd sh_test_mesh/workload/llama2_7b_inference &&
      python3 joint/test_ledger_world_source.py （或 pytest 同路径）
"""
import json
import os
import shutil
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

_HERE = os.path.dirname(os.path.abspath(__file__))
_WL = os.path.dirname(_HERE)
for _p in (_HERE, _WL, os.path.join(_WL, "online")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from joint.action_work_graph import (  # noqa: E402
    ActionWorkGraphError,
    LedgerWorldSource,
    LoadsWorldSource,
)
import joint.action_work_graph as action_work_graph_module  # noqa: E402
from joint.work_ledger import (  # noqa: E402
    WorkIdentity,
    WorkAmount,
    WorkLedger,
    CommittedWork,
    leg_work_id,
)
from joint.runtime_snapshot import build_decision_snapshot  # noqa: E402
from joint.pipeline_ledger import PipelineLedger  # noqa: E402


class _CountingMembershipSet(set):
    """Count the real membership and insertion operations in one LWS set."""

    def __init__(self, values=()):
        super().__init__(values)
        self.contains_calls = 0
        self.add_calls = 0

    def __contains__(self, value):
        self.contains_calls += 1
        return super().__contains__(value)

    def add(self, value):
        self.add_calls += 1
        return super().add(value)


def _leg_work(leg, *, request_id="r1", rank=0, node_id=1,
              state="in_service", remaining_bytes=1000.0, route=None,
              total_bytes=None):
    identity = WorkIdentity(
        request_id=request_id, stage="decode", generation=0,
        rank=rank, node_id=node_id, operation="comm_send")
    total = WorkAmount(bytes=(remaining_bytes if total_bytes is None
                              else total_bytes))
    return CommittedWork(
        identity=identity, state=state, total=total,
        served=WorkAmount(), remaining=WorkAmount(bytes=remaining_bytes),
        progress_source="authoritative", service_unit="bytes",
        leg=leg, route=route)


class _Model:
    """LoadsWorldSource/LedgerWorldSource.build_context 的最小 model 面。"""

    class _Rates:
        noc_link_bytes_per_ns = 10.0
        pool_port_bytes_per_ns = 5.0
        local_hbm_bytes_per_ns = 20.0

    rates = _Rates()

    def _exec_ranks(self, instance_index):
        return (0, 1)

    def _hbm_background_divisor(self, rank):
        return 0

    def _background_links(self):
        return []

    def _background_pool_ports(self):
        return []

    def _link_background_members(self, link):
        return 0

    def _pool_background_members(self, port):
        return 0

    loads = {0: None}


class LedgerProjectionTest(unittest.TestCase):
    """账本 → 前推背景投影。"""

    def test_hbm_and_pool_legs_project_authoritative_remaining(self):
        ledger = WorkLedger()
        work = _leg_work("hbm:COMM_READ", rank=3, remaining_bytes=750.0)
        ledger._register_work(work)
        pool = _leg_work("pool:2:17", remaining_bytes=500.0)
        ledger._register_work(pool)
        world = LedgerWorldSource(ledger=ledger).build_context(_Model())
        self.assertEqual(world.progress_source, "ledger")
        legs = {leg.leg_id: leg for item in world.background_works
                for leg in item.legs}
        self.assertTrue(any("hbm:3" in key for key in legs))
        self.assertTrue(any("pool:2" in key for key in legs))
        hbm_leg = next(leg for leg in legs.values()
                       if "hbm:3" in str(leg.route))
        self.assertEqual(hbm_leg.remaining_bytes, 750)
        self.assertTrue(hbm_leg.in_service)
        # 资源规格随腿携带（容量 = rates 标量）。
        keys = {spec.key for spec in world.resources}
        self.assertIn("hbm:3", keys)
        self.assertIn("pool:2", keys)

    def test_authoritative_resource_key_set_scales_against_old_list_scan(self):
        for count in (8, 32, 128):
            with self.subTest(count=count):
                ledger = WorkLedger()
                works = []
                for rank in range(count):
                    work = _leg_work(
                        "hbm:COMM_READ", rank=rank, node_id=rank + 1,
                        remaining_bytes=1000.0)
                    ledger._register_work(work)
                    works.append(work)
                model = _Model()
                snapshot = SimpleNamespace(
                    ledger_projection=ledger.projection_snapshot(),
                    pipeline_ledger_state=None, event_watermark=0)
                source = LedgerWorldSource(ledger, snapshot=snapshot)

                # Independent old list-scan oracle. Every key is unique, so
                # this executes exactly 0+1+...+(N-1) real comparisons.
                expected_specs = []
                old_key_comparisons = 0
                expected_background = []
                for work in works:
                    key = action_work_graph_module.hbm_resource_key(
                        work.identity.rank)
                    found = False
                    for prior in expected_specs:
                        old_key_comparisons += 1
                        if prior.key == key:
                            found = True
                            break
                    if not found:
                        expected_specs.append(
                            action_work_graph_module.ResourceSpec(
                                key=key,
                                kind=action_work_graph_module.RESOURCE_KIND_HBM,
                                capacity_per_ns=float(
                                    model.rates.local_hbm_bytes_per_ns)))
                    remaining = int(work.remaining.bytes)
                    expected_background.append(
                        action_work_graph_module.WorkItem(
                            work_id="bg-ledger:" + work.work_id,
                            legs=(action_work_graph_module._stream_leg(
                                "bg-ledger:{}#{}".format(work.work_id, key),
                                route=(key,), bytes_amount=remaining,
                                remaining_bytes=remaining,
                                in_service=True),)))

                expected_world = action_work_graph_module.WorldContext(
                    now_ns=0,
                    resources=tuple(expected_specs),
                    background_works=tuple(expected_background),
                    efficiency=action_work_graph_module
                    ._world_projection_efficiency(model),
                    approximations=(
                        "ledger_snapshot_watermark=0",
                        "ledger_source=ledger_v1",),
                    progress_source=(
                        action_work_graph_module.PROGRESS_SOURCE_LEDGER),
                )
                constructed_sets = []

                def counted_set(values=()):
                    result = _CountingMembershipSet(values)
                    constructed_sets.append(result)
                    return result

                # This call-site factory observes the production set itself;
                # there is one ledger_keys set and no side membership table.
                with mock.patch.object(
                        action_work_graph_module, "set", counted_set,
                        create=True):
                    world = source.build_context(model)

                self.assertEqual(world, expected_world)
                self.assertEqual(len(constructed_sets), 1)
                actual_membership = constructed_sets[0]
                self.assertEqual(actual_membership.contains_calls, count)
                self.assertEqual(actual_membership.add_calls, count)
                self.assertEqual(len(actual_membership), count)
                fallback_membership_calls = (
                    actual_membership.contains_calls - count)
                self.assertEqual(fallback_membership_calls, 0)
                self.assertEqual(
                    old_key_comparisons, count * (count - 1) // 2)

                # Candidate-specific additions still see the same complete
                # world. Exercise P/D and each action against the independent
                # list-oracle context without running a projection simulation.
                for phase in ("P", "D"):
                    for action in (
                            "stay", "recompute", "copy", "remote_read"):
                        tag = f"candidate:{phase}:{action}"
                        candidate = action_work_graph_module.CandidateGraph(
                            work=action_work_graph_module.WorkItem(
                                work_id=tag,
                                legs=(action_work_graph_module._fixed_leg(
                                    tag + "#compute", 0.0),)),
                            resources=(
                                action_work_graph_module.ResourceSpec(
                                    key=action_work_graph_module.hbm_resource_key(
                                        count + 32),
                                    kind=action_work_graph_module.RESOURCE_KIND_HBM,
                                    capacity_per_ns=1.0),),
                        )
                        self.assertEqual(
                            world.projection_input(candidate),
                            expected_world.projection_input(candidate))
                print(
                    "[ledger-world-resource-key-scale] "
                    f"N={count} old_list_comparisons="
                    f"{old_key_comparisons} new_contains="
                    f"{actual_membership.contains_calls} new_adds="
                    f"{actual_membership.add_calls} "
                    f"stored_keys={len(actual_membership)} "
                    f"fallback_contains={fallback_membership_calls}")

    def test_issued_not_streaming_leg_is_non_in_service(self):
        ledger = WorkLedger()
        work = _leg_work("hbm:COMM_READ", rank=1, state="issued")
        ledger._register_work(work)
        world = LedgerWorldSource(ledger=ledger).build_context(_Model())
        leg = next(leg for item in world.background_works
                   for leg in item.legs)
        self.assertFalse(leg.in_service)

    def test_committed_not_issued_counts_disclosure_only(self):
        ledger = WorkLedger()
        node = CommittedWork(
            identity=WorkIdentity(
                request_id="r2", stage="prefill", generation=0,
                rank=0, node_id=9, operation="compute"),
            state="committed",
            total=WorkAmount(ops=100),
            remaining=WorkAmount(ops=100))
        ledger._register_work(node)
        world = LedgerWorldSource(ledger=ledger).build_context(_Model())
        # 不虚构资源腿：背景无该工作的腿。
        self.assertFalse(any(node.work_id in item.work_id
                             for item in world.background_works))
        self.assertIn("ledger_committed_not_issued=1",
                      world.approximations)

    def test_per_key_registry_fallback_without_double_count(self):
        ledger = WorkLedger()
        work = _leg_work("hbm:COMM_READ", rank=0, remaining_bytes=800.0)
        ledger._register_work(work)

        class _RegistryModel(_Model):
            def _hbm_background_divisor(self, rank):
                return 3 if rank == 0 else 0

            def _background_links(self):
                return [(2, 3)]

            def _link_background_members(self, link):
                return 2

        world = LedgerWorldSource(
            ledger=ledger).build_context(_RegistryModel())
        # hbm:0 有权威腿 ⇒ 无注册表回退腿；hbm:0 回退披露缺席。
        hbm_legs = [leg for item in world.background_works
                    for leg in item.legs
                    if leg.route == ("hbm:0",)]
        self.assertEqual(len(hbm_legs), 1)
        self.assertNotIn("ledger_leg_absent_registry_counts:hbm:0",
                         world.approximations)
        # link:2->3 无账本腿 ⇒ 注册表计数回退（2 条 unknown-ETA 腿）。
        self.assertIn("ledger_leg_absent_registry_counts:link:2->3",
                      world.approximations)
        link_legs = [leg for item in world.background_works
                     for leg in item.legs
                     if leg.route == ("link:2->3",)]
        self.assertEqual(len(link_legs), 2)

    def test_multihop_pipeline_overlay_and_fallback_keep_all_candidate_contexts(self):
        class _MixedRegistryModel(_Model):
            def _hbm_background_divisor(self, rank):
                return 2 if rank == 0 else 0

            def _background_links(self):
                return [(8, 9), (20, 21)]

            def _link_background_members(self, link):
                return 2 if tuple(link) == (8, 9) else 1

            def _background_pool_ports(self):
                return [4]

            def _pool_background_members(self, port):
                return 1

        model = _MixedRegistryModel()
        ledger = WorkLedger()
        multihop = _leg_work(
            "noc:0", request_id="pipe", rank=0, node_id=1,
            remaining_bytes=900.0, route=(7, 8, 7), total_bytes=1000.0)
        shared = _leg_work(
            "noc:0", request_id="shared", rank=1, node_id=2,
            remaining_bytes=700.0, route=(8, 9), total_bytes=800.0)
        hbm = _leg_work(
            "hbm:COMM_READ", request_id="hbm", rank=0, node_id=3,
            remaining_bytes=600.0, total_bytes=700.0)
        for work in (multihop, shared, hbm):
            ledger._register_work(work)

        active_noc = {
            "resource_type": "noc_flow",
            "work_id": multihop.work_id,
            "request_id": "pipe", "stage": "decode", "generation": 0,
            "rank": 0, "node_id": 1, "operation": "comm_send",
            "leg": "noc:0", "state": "Active",
            "total": {"ops": None, "bytes": 1000.0,
                      "latency_ns": None},
            "remaining": {"ops": None, "bytes": 500.0,
                           "latency_ns": None},
            "service_unit": "bytes", "route": [7, 8, 7],
            "as_of_tick": 12, "issued_tick": 2,
        }
        snapshot = SimpleNamespace(
            ledger_projection=ledger.projection_snapshot(),
            pipeline_ledger_state={"active_resources": (active_noc,)},
            event_watermark=12)
        link_map = lambda link: (link, link + 1)
        source = LedgerWorldSource(
            ledger, snapshot=snapshot, link_id_map_fn=link_map)

        # Independent old list-scan oracle over the authoritative route
        # occurrences. It preserves the first ResourceSpec and encounter order.
        replacement = CommittedWork(
            identity=WorkIdentity(
                request_id="pipe", stage="decode", generation=0, rank=0,
                node_id=1, operation="comm_send", resource_legs=("noc:0",)),
            state="in_service", total=WorkAmount(bytes=1000.0),
            served=WorkAmount(), remaining=WorkAmount(bytes=500.0),
            progress_source="authoritative", service_unit="bytes",
            route=(7, 8, 7), leg="noc:0", issued_tick=2, as_of_tick=12)
        authoritative_works = (replacement, shared, hbm)
        expected_ledger_keys = []
        expected_resources = []
        expected_background = []
        old_key_comparisons = 0
        authoritative_key_occurrences = 0
        for work in authoritative_works:
            if work.leg.startswith("noc:"):
                route = tuple(dict.fromkeys(
                    action_work_graph_module.link_resource_key(*link_map(link))
                    for link in (work.route or ())))
                kind = action_work_graph_module.RESOURCE_KIND_LINK
                capacity = float(model.rates.noc_link_bytes_per_ns)
            else:
                route = (action_work_graph_module.hbm_resource_key(
                    work.identity.rank),)
                kind = action_work_graph_module.RESOURCE_KIND_HBM
                capacity = float(model.rates.local_hbm_bytes_per_ns)
            for key in route:
                authoritative_key_occurrences += 1
                found = False
                for prior in expected_resources:
                    old_key_comparisons += 1
                    if prior.key == key:
                        found = True
                        break
                if not found:
                    expected_resources.append(
                        action_work_graph_module.ResourceSpec(
                            key=key, kind=kind, capacity_per_ns=capacity))
                if key not in expected_ledger_keys:
                    expected_ledger_keys.append(key)
            remaining = int(work.remaining.bytes or work.total.bytes)
            expected_background.append(
                action_work_graph_module.WorkItem(
                    work_id="bg-ledger:" + work.work_id,
                    legs=(action_work_graph_module._stream_leg(
                        "bg-ledger:{}#{}".format(
                            work.work_id, "|".join(route)),
                        route=route, bytes_amount=remaining,
                        remaining_bytes=remaining,
                        in_service=work.state == "in_service"),)))

        fallback = LoadsWorldSource().build_context(model)
        expected_approximations = ["ledger_snapshot_watermark=12"]
        for spec in fallback.resources:
            if spec.key in expected_ledger_keys:
                continue
            expected_resources.append(spec)
            expected_approximations.append(
                "ledger_leg_absent_registry_counts:{}".format(spec.key))
        for work in fallback.background_works:
            if any(key in expected_ledger_keys
                   for leg in work.legs for key in leg.route):
                continue
            expected_background.append(work)
        expected_approximations.append("ledger_source=ledger_v1")
        expected_world = action_work_graph_module.WorldContext(
            now_ns=0, resources=tuple(expected_resources),
            background_works=tuple(expected_background),
            efficiency=fallback.efficiency,
            approximations=tuple(expected_approximations),
            progress_source=action_work_graph_module.PROGRESS_SOURCE_LEDGER)

        observed_sets = []

        def counted_set(values=()):
            result = _CountingMembershipSet(values)
            observed_sets.append(result)
            return result

        with mock.patch.object(
                action_work_graph_module, "set", counted_set, create=True):
            world = source.build_context(model)
        self.assertEqual(world, expected_world)
        self.assertEqual(tuple(spec.key for spec in world.resources), (
            "link:7->8", "link:8->9", "link:9->10", "hbm:0",
            "link:20->21", "pool:4"))
        self.assertEqual(old_key_comparisons, 8)
        # The first set is the single ledger_keys set. Its fixed fallback work
        # is reported separately from five authoritative route occurrences.
        self.assertEqual(len(observed_sets), 2)  # ledger_keys + replaced_ids
        ledger_keys = observed_sets[0]
        self.assertEqual(ledger_keys.contains_calls, 13)
        self.assertEqual(ledger_keys.add_calls, 5)
        self.assertEqual(len(ledger_keys), 4)
        fallback_membership_calls = (
            ledger_keys.contains_calls - authoritative_key_occurrences)
        self.assertEqual(authoritative_key_occurrences, 5)
        self.assertEqual(fallback_membership_calls, 8)

        for phase in ("P", "D"):
            for action in ("stay", "recompute", "copy", "remote_read"):
                tag = f"mixed-candidate:{phase}:{action}"
                candidate = action_work_graph_module.CandidateGraph(
                    work=action_work_graph_module.WorkItem(
                        work_id=tag,
                        legs=(action_work_graph_module._fixed_leg(
                            tag + "#compute", 0.0),)),
                    resources=(action_work_graph_module.ResourceSpec(
                        key="link:8->9",
                        kind=action_work_graph_module.RESOURCE_KIND_LINK,
                        capacity_per_ns=999.0),),
                )
                self.assertEqual(
                    world.projection_input(candidate),
                    expected_world.projection_input(candidate))
        print(
            "[ledger-world-resource-key-mixed] "
            f"old_list_comparisons={old_key_comparisons} "
            f"new_authoritative_contains={authoritative_key_occurrences} "
            f"fallback_contains={fallback_membership_calls} "
            f"new_adds={ledger_keys.add_calls} "
            f"stored_keys={len(ledger_keys)} ordered_resources="
            f"{len(world.resources)} candidates=8")

    def test_v2_pipeline_noc_overlays_old_leg_without_request_filter(self):
        # The old WorkLedger row and v2 active resource describe one physical
        # pipeline leg.  The authoritative frame replaces its stale amount,
        # while an unrelated request's flow remains on the same shared link.
        ledger = WorkLedger()
        pipeline_identity = _leg_work(
            "noc:0", request_id="r1", rank=0, node_id=3,
            remaining_bytes=900.0, route=(7,), total_bytes=1000.0)
        pipeline_hbm = _leg_work(
            "hbm:COMM_READ", request_id="r1", rank=0, node_id=3,
            remaining_bytes=900.0, total_bytes=1000.0)
        cross_request_flow = _leg_work(
            "noc:0", request_id="noc", rank=0, node_id=4,
            remaining_bytes=400.0, route=(7,))
        ledger._register_work(pipeline_identity)
        ledger._register_work(pipeline_hbm)
        ledger._register_work(cross_request_flow)
        active_noc = {
            "resource_type": "noc_flow",
            "work_id": pipeline_identity.work_id,
            "request_id": "r1", "stage": "decode", "generation": 0,
            "rank": 0, "node_id": 3, "operation": "comm_send",
            "leg": "noc:0", "state": "Active",
            "total": {"ops": None, "bytes": 1000.0,
                      "latency_ns": None},
            "remaining": {"ops": None, "bytes": 600.0,
                           "latency_ns": None},
            "progress_source": "authoritative", "service_unit": "bytes",
            "route": [7], "as_of_tick": 8, "issued_tick": 2,
            "descriptor_segment_id": "pp1/example", "endpoint_selector": {
                "group_ordinal": 0, "tile_ordinal": 1,
                "shard_ordinal": 0, "endpoint": "data_send"},
            "flow_id": 0, "current_rate_Bpns": 4.0,
            "last_rate_update_time": 8.0,
        }
        active_hbm = {
            "resource_type": "hbm_job",
            "work_id": pipeline_hbm.work_id,
            "request_id": "r1", "stage": "decode", "generation": 0,
            "rank": 0, "node_id": 3, "operation": "comm_send",
            "leg": "hbm:COMM_READ", "state": "in_service",
            "total": {"ops": None, "bytes": 1000.0,
                      "latency_ns": None},
            "remaining": {"ops": None, "bytes": 600.0,
                           "latency_ns": None},
            "progress_source": "authoritative", "service_unit": "bytes",
            "route": [], "as_of_tick": 8, "issued_tick": 2,
            "descriptor_segment_id": "pp1/example", "endpoint_selector": {
                "group_ordinal": 0, "tile_ordinal": 1,
                "shard_ordinal": 0, "endpoint": "data_recv"},
            "kind": "COMM_READ", "start_tick": 2,
        }
        # LWS consumes the typed runtime view after C4 descriptor validation;
        # this fixture isolates the price-input merge rule itself.
        snapshot = type("Snapshot", (), {
            "ledger_projection": ledger.projection_snapshot(),
            "pipeline_ledger_state": {
                "active_resources": (active_hbm, active_noc)},
            "event_watermark": 12,
        })()
        link_map = lambda _link: (0, 1)
        projected = LedgerWorldSource(
            ledger, snapshot=snapshot,
            link_id_map_fn=link_map).build_context(_Model())

        baseline = WorkLedger()
        baseline._register_work(_leg_work(
            "noc:0", request_id="r1", rank=0, node_id=3,
            remaining_bytes=600.0, route=(7,), total_bytes=1000.0))
        baseline._register_work(_leg_work(
            "hbm:COMM_READ", request_id="r1", rank=0, node_id=3,
            remaining_bytes=600.0, total_bytes=1000.0))
        baseline._register_work(cross_request_flow)
        expected = LedgerWorldSource(
            baseline, link_id_map_fn=link_map).build_context(_Model())
        self.assertEqual(projected.resources, expected.resources)
        self.assertEqual(projected.background_works, expected.background_works)
        self.assertEqual(len(projected.background_works), 3)
        remaining_by_work = {
            item.work_id.removeprefix("bg-ledger:"): item.legs[0].remaining_bytes
            for item in projected.background_works
        }
        self.assertEqual(remaining_by_work[pipeline_identity.work_id], 600.0)
        self.assertEqual(remaining_by_work[pipeline_hbm.work_id], 600.0)
        self.assertEqual(remaining_by_work[cross_request_flow.work_id], 400.0)

    def test_bare_construction_fails_closed(self):
        with self.assertRaises(ActionWorkGraphError):
            LedgerWorldSource().build_context(_Model())
        with self.assertRaises(ActionWorkGraphError):
            LedgerWorldSource(snapshot=object()).build_context(_Model())


class SnapshotVersionInvalidationTest(unittest.TestCase):
    """快照版本失效（cache_key 纳入账本版本）。"""

    @staticmethod
    def _snapshot(ledger_version, watermark):
        return build_decision_snapshot(
            {"delivery_sequence": 7, "tick": 100, "arrivals": []},
            ledger_versions={
                "version": ledger_version, "event_watermark": watermark},
            now_ns=100)

    def test_ledger_version_advances_invalidate_cache_key(self):
        first = self._snapshot(3, 30)
        same = self._snapshot(3, 30)
        self.assertEqual(first.cache_key(), same.cache_key())
        bumped_version = self._snapshot(4, 30)
        self.assertNotEqual(first.cache_key(), bumped_version.cache_key())
        bumped_watermark = self._snapshot(3, 31)
        self.assertNotEqual(first.cache_key(), bumped_watermark.cache_key())


class JournalReplayFidelityTest(unittest.TestCase):
    """journal 重放保真（W1 §2 红线；A2 侧车实现）。"""

    def _run_bridge(self, bridge_dir, requests):
        from decision_bridge import BridgeServer
        active = {"canonical": None}
        server = BridgeServer(
            bridge_dir, canonical_request_producer=True,
            work_ledger_bytes_source=lambda _seq: (
                active["canonical"],
                {"snapshot_kind": "full", "event_watermark": None}))
        for seq, payload in enumerate(requests):
            active["canonical"] = json.dumps(
                payload["work_ledger"], sort_keys=True, separators=(",", ":"),
                ensure_ascii=False, allow_nan=False).encode("utf-8")
            with open(os.path.join(bridge_dir, f"request_{seq}.json"),
                      "w", encoding="utf-8") as handle:
                json.dump(payload, handle, separators=(",", ":"))
            server._handle_request(seq, lambda _request, _segment: {
                "pipeline_descriptors": [], "kernel_contract": {}})
        server._close_audit_streams()
        return server

    def test_sidecar_roundtrip_rebuilds_identical_ledger_state(self):
        work_dir = tempfile.mkdtemp(prefix="wl_replay_")
        self.addCleanup(shutil.rmtree, work_dir, True)
        from decision_bridge import BridgeServer
        active = {"canonical": None}
        server = BridgeServer(
            work_dir, canonical_request_producer=True,
            work_ledger_bytes_source=lambda _seq: (
                active["canonical"],
                {"snapshot_kind": "full", "event_watermark": None}))
        # 首次v2 full + 后续v2 delta。嵌套子信封与外层delivery同号；
        # journal侧车必须逐字保存此inner contract，不得降回WorkLedger v1。
        work_id = "r1#decode:0:r0n1"

        def envelope(kind, delivery, tick, base_watermark, base_version,
                     event):
            count = 1
            seq = base_watermark + 1
            return {
                "producer": "cpp_online", "schema_version": 3,
                "protocol_version": 3,
                "bridge_protocol": "bridge-v3-20261009",
                "work_ledger_schema": "workledger-v3-20261009",
                "epoch": "journal-test", "delivery_sequence": delivery,
                "base_watermark": base_watermark,
                "logical_seq_start": seq, "logical_seq_end": seq,
                "logical_event_count": count,
                "replay_token": f"cpp_online/journal-test/{delivery}",
                "descriptor_version": 1,
                "base_state_version": base_version,
                "result_state_version": base_version + 1,
                "snapshot_kind": kind, "tick": tick,
                "descriptor_commits": [],
                "node_transitions": [event],
                "pipeline_transition_runs": [],
                "cursor_patches": [], "active_resources": [],
            }

        full = envelope(
            "full", 0, 10, 0, 0, {
                "logical_seq": 1, "tick": 10, "kind": "commit",
                "work_id": work_id,
                "payload": {
                    "request_id": "r1", "stage": "decode",
                    "generation": 0, "rank": 0, "node_id": 1,
                    "operation": "compute", "state": "in_service",
                    "total": {"ops": 10, "bytes": 0,
                              "latency_ns": 50},
                    "remaining_ops": 10, "remaining_bytes": 0,
                    "remaining_latency_ns": 50,
                    "parents": [], "service_unit": "job",
                },
            })
        delta = envelope(
            "delta", 1, 50, 1, 1, {
                "logical_seq": 2, "tick": 50, "kind": "progress",
                "work_id": work_id,
                "payload": {
                    "delta_served": {"ops": 4, "bytes": 0,
                                     "latency_ns": 20},
                    "remaining_ops": 6, "remaining_bytes": 0,
                    "remaining_latency_ns": 30,
                },
            })
        requests = [
            {"schema_version": 2, "delivery_sequence": 0, "tick": 10,
             "reasons": [], "arrivals": [], "completed_nodes": [],
             "completed_groups": [], "affected_ranks": [],
             "event_seqs": [],
             "snapshot_handle": {"kind": "work_ledger_v3"},
             "work_ledger": full},
            {"schema_version": 2, "delivery_sequence": 1, "tick": 50,
             "reasons": [], "arrivals": [], "completed_nodes": [],
             "completed_groups": [], "affected_ranks": [],
             "event_seqs": [],
             "snapshot_handle": {"kind": "work_ledger_v3"},
             "work_ledger": delta},
        ]

        def apply_v3(work_ledger, pipeline_ledger, delivery):
            payload = delivery["work_ledger"]
            transaction = work_ledger.begin_protocol_v3_transaction(
                delivery_sequence=payload["delivery_sequence"],
                tick=payload["tick"],
                snapshot_kind=payload["snapshot_kind"])
            prepared = pipeline_ledger.prepare_envelope(
                payload, work_transaction=transaction)
            transaction.commit(notify=False)
            pipeline_ledger.commit_prepared(prepared)
            transaction.notify()

        # 原跑：逐交付应用到账本（canonical handler 路径）。
        original = WorkLedger()
        original_pipeline = PipelineLedger()
        for seq, request in enumerate(requests):
            path = os.path.join(work_dir, "request_{}.json".format(seq))
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(request, handle)
            active["canonical"] = json.dumps(
                request["work_ledger"], sort_keys=True,
                separators=(",", ":"), ensure_ascii=False,
                allow_nan=False).encode("utf-8")
            server._handle_request(seq, lambda req, _segment: {
                "pipeline_descriptors": [], "kernel_contract": {}})
            apply_v3(original, original_pipeline, request)
        server._close_audit_streams()

        # journal 剥离核验：journal 行不携带 work_ledger，带 stripped 标记。
        journal_path = os.path.join(work_dir, "request_journal.jsonl")
        with open(journal_path, encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle]
        self.assertEqual(len(rows), 2)
        for row in rows:
            self.assertNotIn("work_ledger", row)
            self.assertIn("work_ledger_journal_stripped", row)
            self.assertEqual(
                row["work_ledger_journal_stripped"]["sidecar"],
                "request_journal_work_ledger.jsonl.gz")

        # 重放（前缀 = 全量）：attach_work_ledger=True 还原账本状态。
        from bridge_request_journal import iter_request_records
        replayed = WorkLedger()
        replayed_pipeline = PipelineLedger()
        for seq, request in iter_request_records(
                work_dir, attach_work_ledger=True):
            self.assertEqual(request["work_ledger"]["delivery_sequence"], seq)
            apply_v3(replayed, replayed_pipeline, request)
        self.assertEqual(
            replayed.works_summary(), original.works_summary())
        self.assertEqual(replayed.version, original.version)
        self.assertEqual(
            replayed.event_watermark, original.event_watermark)
        self.assertEqual(replayed.applied_event_count,
                         original.applied_event_count)
        self.assertEqual(replayed_pipeline.snapshot_state(),
                         original_pipeline.snapshot_state())

    def test_replay_without_sidecar_payload_fails_closed(self):
        work_dir = tempfile.mkdtemp(prefix="wl_replay_missing_")
        self.addCleanup(shutil.rmtree, work_dir, True)
        journal = os.path.join(work_dir, "request_journal.jsonl")
        with open(journal, "w", encoding="utf-8") as handle:
            handle.write(json.dumps({
                "schema_version": 2, "delivery_sequence": 0, "tick": 1,
                "work_ledger_journal_stripped": {"snapshot_kind": "full"},
            }) + "\n")
        from bridge_request_journal import iter_request_records
        with self.assertRaisesRegex(
                ValueError, "replay would degrade to an empty ledger"):
            list(iter_request_records(work_dir, attach_work_ledger=True))


if __name__ == "__main__":
    unittest.main()
