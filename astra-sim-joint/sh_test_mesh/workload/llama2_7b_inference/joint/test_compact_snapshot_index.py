"""Incremental immutable ledger rows and snapshot invalidation coverage."""

import json
import unittest
from types import SimpleNamespace

from joint.schema_registry import (
    BRIDGE_PROTOCOL_VERSION, WORK_LEDGER_SCHEMA_VERSION)
from joint.work_ledger import (
    CommittedWork, WorkAmount, WorkIdentity, WorkLedger,
    canonical_work_id, leg_work_id)
from joint.runtime_snapshot import build_decision_snapshot
from joint.action_work_graph import LedgerWorldSource


def _envelope(sequence, tick, kind, watermark=0, **extra):
    payload = {
        "schema_version": 1,
        "bridge_protocol": BRIDGE_PROTOCOL_VERSION,
        "work_ledger_schema": WORK_LEDGER_SCHEMA_VERSION,
        "snapshot_kind": kind,
        "delivery_sequence": sequence,
        "tick": tick,
        "event_watermark": watermark,
    }
    payload.update(extra)
    return payload


def _state(remaining_bytes=90):
    node_id = canonical_work_id("req", "prefill", 0, 0, 1)
    hbm_id = leg_work_id(node_id, "hbm:COMM_READ")
    return {
        "nodes": [{"rank": 0, "nodes": [{
            "node_id": 1, "request_id": "req", "stage": "prefill",
            "generation": 0, "kind": "comm_send", "issued": True,
            "finished": False, "num_ops": 10, "bytes": 100,
            "runtime_ns": 5, "parents": [],
        }, {
            "node_id": 2, "request_id": "req", "stage": "prefill",
            "generation": 0, "kind": "compute", "issued": False,
            "finished": False, "num_ops": 10, "bytes": 0,
            "runtime_ns": 5, "parents": [1],
        }]}],
        "hbm": [{"rank": 0, "as_of_tick": 10, "jobs": [{
            "work_id": hbm_id, "node_id": 1, "kind": "COMM_READ",
            "request_id": "req", "stage": "prefill", "generation": 0,
            "bytes": 100, "remaining_bytes": remaining_bytes,
            "remaining_ops": 0, "remaining_latency_ns": 0,
            "total_latency_ns": 0, "state": "in_service",
        }]}],
        "pool": [],
        "noc": {},
        "slots": [],
    }


class TestProjectionRowsAndIndexes(unittest.TestCase):
    class _Model:
        rates = SimpleNamespace(
            local_hbm_bytes_per_ns=20.0,
            pool_port_bytes_per_ns=5.0,
            noc_link_bytes_per_ns=10.0)
        service_factors = SimpleNamespace(transfer_factor=1.0)
        loads = {0: None}

        def _exec_ranks(self, instance_index):
            return (0,)

        def _hbm_background_divisor(self, rank):
            return 0

        def _background_links(self):
            return []

        def _link_background_members(self, link):
            return 0

        def _background_pool_ports(self):
            return []

        def _pool_background_members(self, port):
            return 0

    def test_resource_version_vector_does_not_materialize_full_projection(self):
        ledger = WorkLedger()
        ledger._register_work(CommittedWork(
            identity=WorkIdentity(
                request_id="r", stage="decode", generation=0,
                rank=0, node_id=1),
            state="in_service", total=WorkAmount(bytes=5),
            remaining=WorkAmount(bytes=5),
            progress_source="authoritative", leg="hbm:READ"))
        self.assertIsNone(ledger._projection_cache)
        vector = ledger.resource_version_vector()
        self.assertTrue(vector)
        self.assertIsNone(ledger._projection_cache)

    def test_incremental_progress_preserves_old_snapshot_rows(self):
        ledger = WorkLedger()
        ledger.apply_payload(
            _envelope(0, 10, "full", watermark=5, state=_state()),
            delivery_sequence=0, tick=10)
        hbm_id = leg_work_id(
            canonical_work_id("req", "prefill", 0, 0, 1),
            "hbm:COMM_READ")
        first = ledger.projection_snapshot()
        first_row = next(row for row in first.rows if row.work_id == hbm_id)
        self.assertEqual(first_row.remaining.bytes, 90)
        self.assertEqual(
            first.resource_index["hbm_rank:0"][0].work_id, hbm_id)
        self.assertEqual(
            len(first.request_stage_index[("req", "prefill")]), 3)
        self.assertEqual(len(first.rank_index[0]), 3)
        parent = canonical_work_id("req", "prefill", 0, 0, 1)
        dependent = canonical_work_id("req", "prefill", 0, 0, 2)
        self.assertEqual(
            first.dependency_index[parent][0].work_id, dependent)

        receipt = ledger.apply_payload(
            _envelope(
                1, 20, "delta", watermark=6,
                events=[{
                    "kind": "progress", "work_id": hbm_id, "tick": 20,
                    "event_id": {
                        "tick": 20, "delivery_sequence": 1,
                        "producer": "work_ledger_diff", "local_seq": 5,
                    },
                    "delta_served": {"bytes": 10},
                    "remaining_bytes": 80,
                }]),
            delivery_sequence=1, tick=20)
        second = ledger.projection_snapshot()
        second_row = next(row for row in second.rows if row.work_id == hbm_id)
        self.assertEqual(receipt.applied_events, 1)
        self.assertEqual(first_row.remaining.bytes, 90)
        self.assertEqual(second_row.remaining.bytes, 80)
        self.assertIsNot(first_row, second_row)
        self.assertGreater(
            dict(second.resource_version_vector)["hbm_rank:0"],
            dict(first.resource_version_vector)["hbm_rank:0"])
        with self.assertRaises((AttributeError, TypeError)):
            first.resource_index["hbm_rank:0"] = ()
        with self.assertRaises((AttributeError, TypeError)):
            first_row.state = "released"

    def test_dependency_and_release_condition_indexes_are_incremental(self):
        ledger = WorkLedger()
        work = CommittedWork(
            identity=WorkIdentity(
                request_id="r2", stage="decode", generation=1,
                rank=3, node_id=8, operation="remote_read"),
            state="in_service",
            total=WorkAmount(bytes=100),
            served=WorkAmount(bytes=10),
            remaining=WorkAmount(bytes=90),
            progress_source="authoritative",
            dependencies=("parent-work",),
            release_conditions=("watch:5",),
            leg="pool:4:9")
        ledger._register_work(work)
        projection = ledger.projection_snapshot()
        row = projection.rows[0]
        self.assertEqual(
            projection.request_stage_index[("r2", "decode")], (row,))
        self.assertEqual(projection.rank_index[3], (row,))
        self.assertEqual(projection.dependency_index["parent-work"], (row,))
        self.assertEqual(
            projection.release_condition_index["watch:5"], (row,))
        self.assertEqual(projection.unfinished_state_index["in_service"], (row,))
        self.assertEqual(projection.resource_index["pool_port:4"], (row,))

    def test_snapshot_shares_projection_and_keys_every_cache_dimension(self):
        ledger = WorkLedger()
        ledger.apply_payload(
            _envelope(0, 10, "full", watermark=5, state=_state()),
            delivery_sequence=0, tick=10)
        projection = ledger.projection_snapshot()
        kwargs = dict(
            ledger_state=ledger.snapshot_state(),
            ledger_projection=projection,
            resource_version_vector=projection.resource_version_vector,
            directory_version=7,
            capacity_version=11,
            now_ns=20,
        )
        delta = {"delivery_sequence": 1, "tick": 20, "arrivals": []}
        first = build_decision_snapshot(delta, **kwargs)
        same = build_decision_snapshot(delta, **kwargs)
        self.assertIs(first.ledger_projection, projection)
        self.assertEqual(first.cache_key(), same.cache_key())
        self.assertNotEqual(
            first.cache_key(),
            build_decision_snapshot(
                delta, **dict(kwargs, directory_version=8)).cache_key())
        self.assertNotEqual(
            first.cache_key(),
            build_decision_snapshot(
                delta, **dict(kwargs, capacity_version=12)).cache_key())
        self.assertNotEqual(
            first.cache_key(),
            build_decision_snapshot(
                delta, **dict(kwargs, now_ns=21)).cache_key())
        serialized = json.loads(first.to_json())
        self.assertNotIn("ledger_projection", serialized)
        self.assertNotIn("cache_dimensions", serialized)
        frozen_state = ledger.snapshot_state()
        self.assertIs(frozen_state, ledger.snapshot_state())
        with self.assertRaises(TypeError):
            frozen_state["nodes"] = ()

    def test_world_source_consumes_cached_pricing_rows_in_ledger_order(self):
        ledger = WorkLedger()
        ledger.apply_payload(
            _envelope(0, 10, "full", watermark=5, state=_state()),
            delivery_sequence=0, tick=10)
        projection = ledger.projection_snapshot()
        snapshot = build_decision_snapshot(
            {"delivery_sequence": 0, "tick": 10, "arrivals": []},
            ledger_state=ledger.snapshot_state(),
            ledger_projection=projection,
            event_watermark=5,
            now_ns=10)
        indexed = LedgerWorldSource(
            ledger=ledger, snapshot=snapshot).build_context(self._Model())
        direct = LedgerWorldSource(ledger=ledger).build_context(self._Model())
        self.assertEqual(indexed.resources, direct.resources)
        self.assertEqual(indexed.background_works, direct.background_works)
        self.assertIn("ledger_committed_not_issued=1", indexed.approximations)
        self.assertIn("ledger_snapshot_watermark=5", indexed.approximations)

    def test_same_tick_compact_commit_updates_request_and_unfinished_indexes(self):
        ledger = WorkLedger()
        ledger.apply_payload(
            _envelope(0, 10, "full", watermark=5, state=_state()),
            delivery_sequence=0, tick=10)
        node3 = canonical_work_id("req", "prefill", 0, 0, 3)
        before = ledger.projection_snapshot()
        ledger.apply_payload(
            _envelope(
                1, 10, "delta", watermark=6,
                compact={"seq_base": 5,
                         "node_commits": [{
                             "work_id": node3, "request_id": "req",
                             "stage": "prefill", "generation": 0,
                             "rank": 0, "node_id": 3, "operation": "compute",
                             "num_ops": 12, "bytes": 0, "runtime_ns": 4,
                             "parents": [],
                         }]},
                events=[]),
            delivery_sequence=1, tick=10)
        after = ledger.projection_snapshot()
        self.assertEqual(len(before.request_stage_index[("req", "prefill")]), 3)
        self.assertEqual(len(after.request_stage_index[("req", "prefill")]), 4)
        self.assertEqual(
            after.unfinished_state_index["committed"][-1].work_id, node3)


if __name__ == "__main__":
    unittest.main()
