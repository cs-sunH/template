"""Direct compact application preserves the old event-prefix contract."""

import unittest
from unittest import mock

from joint.schema_registry import (
    BRIDGE_PROTOCOL_VERSION, WORK_LEDGER_SCHEMA_VERSION)
from joint.work_ledger import (
    ApplyReceipt, WorkLedger, WorkLedgerError, canonical_work_id)


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


def _node(node_id, *, issued=False, finished=False):
    return {
        "work_id": canonical_work_id("req", "prefill", 0, 0, node_id),
        "node_id": node_id,
        "request_id": "req",
        "stage": "prefill",
        "generation": 0,
        "kind": "compute",
        "issued": issued,
        "finished": finished,
        "num_ops": 10,
        "bytes": 0,
        "runtime_ns": 4,
        "parents": [],
    }


def _state(*nodes):
    return {"nodes": [{"rank": 0, "nodes": list(nodes)}]}


def _event(kind, work_id, sequence, *, value=None, **extra):
    event = {
        "kind": kind,
        "work_id": work_id,
        "tick": 20,
        "event_id": {
            "tick": 20,
            "delivery_sequence": 1,
            "producer": "work_ledger_diff",
            "local_seq": sequence,
        },
    }
    if value is not None:
        event["total_bytes"] = value
    event.update(extra)
    return event


class TestDirectCompactApplication(unittest.TestCase):
    def test_normal_compact_path_materializes_no_wire_event_or_json_digest(self):
        ledger = WorkLedger()
        ledger.apply_payload(
            _envelope(0, 10, "full", watermark=10,
                      state=_state(_node(1))),
            delivery_sequence=0, tick=10)
        entry = _node(2)
        entry["operation"] = "compute"
        with mock.patch(
                "joint.work_ledger.event_from_mapping") as event_parser, \
                mock.patch(
                    "joint.work_ledger._ledger_event_for_observer") as event_view, \
                mock.patch(
                    "joint.work_ledger._event_digest") as json_digest, \
                mock.patch(
                    "joint.work_ledger._legacy_canonical_json") as replay_json:
            receipt = ledger.apply_payload(
                _envelope(
                    1, 20, "delta", watermark=11,
                    compact={"seq_base": 10, "node_commits": [entry]},
                    events=[]),
                delivery_sequence=1, tick=20)
        self.assertEqual(receipt.virtual_count, 1)
        event_parser.assert_not_called()
        event_view.assert_not_called()
        json_digest.assert_not_called()
        replay_json.assert_not_called()

    def test_receipt_and_compact_sequence_are_streamed_in_order(self):
        ledger = WorkLedger()
        ledger.apply_payload(
            _envelope(0, 10, "full", watermark=10,
                      state=_state(_node(1))),
            delivery_sequence=0, tick=10)
        work1 = canonical_work_id("req", "prefill", 0, 0, 1)
        work2 = canonical_work_id("req", "prefill", 0, 0, 2)
        observed = []
        receipt = ledger.apply_payload(
            _envelope(
                1, 20, "delta", watermark=15,
                compact={
                    "seq_base": 10,
                    "node_commits": [_node(2)],
                    "node_issues": [work1],
                    "node_completions": [work1],
                    "node_vanished": [work2],
                },
                events=[]),
            delivery_sequence=1, tick=20,
            observer=lambda event, applied: observed.append(
                (event.as_dict(), applied)))

        self.assertIsInstance(receipt, ApplyReceipt)
        self.assertFalse(hasattr(receipt, "events"))
        self.assertEqual(receipt.applied_events, 5)
        self.assertEqual(receipt.replayed_events, 0)
        self.assertEqual(
            [(item["kind"], item["event_id"]["local_seq"])
             for item, _ in observed],
            [("commit", 10), ("issue", 11), ("completion", 12),
             ("completion", 13), ("release", 14)])
        self.assertEqual(ledger.event_watermark, 15)
        self.assertNotIn(work2, ledger._works)

    def test_explicit_event_error_keeps_applied_prefix_and_payload_watermark(self):
        ledger = WorkLedger()
        ledger.apply_payload(
            _envelope(0, 10, "full", watermark=10,
                      state=_state(_node(1))),
            delivery_sequence=0, tick=10)
        work1 = canonical_work_id("req", "prefill", 0, 0, 1)
        unknown = canonical_work_id("req", "prefill", 0, 0, 99)
        observed = []
        with self.assertRaisesRegex(WorkLedgerError, "unknown work"):
            ledger.apply_payload(
                _envelope(
                    1, 20, "delta", watermark=12,
                    events=[_event("issue", work1, 10),
                            _event("completion", unknown, 11)]),
                delivery_sequence=1, tick=20,
                observer=lambda event, applied: observed.append(
                    (event.work_id, applied)))
        self.assertEqual(observed, [(work1, True)])
        self.assertEqual(ledger.work(work1).state, "issued")
        self.assertEqual(ledger.applied_event_count, 1)
        self.assertEqual(ledger.event_watermark, 10)
        self.assertEqual(ledger.last_delivery_sequence, 0)
        self.assertEqual(ledger.version, 1)
        self.assertEqual(ledger.delta_load_count, 0)

    def test_compact_structure_is_preflighted_before_explicit_prefix(self):
        ledger = WorkLedger()
        ledger.apply_payload(
            _envelope(0, 10, "full", watermark=10,
                      state=_state(_node(1))),
            delivery_sequence=0, tick=10)
        work1 = canonical_work_id("req", "prefill", 0, 0, 1)
        with self.assertRaisesRegex(WorkLedgerError, "carry work_id"):
            ledger.apply_payload(
                _envelope(
                    1, 20, "delta", watermark=11,
                    compact={"seq_base": 10,
                             "node_commits": [_node(2), {"bad": True}]},
                    events=[_event("issue", work1, 10)]),
                delivery_sequence=1, tick=20)
        self.assertEqual(ledger.work(work1).state, "committed")
        self.assertEqual(ledger.applied_event_count, 0)
        self.assertEqual(ledger.last_delivery_sequence, 0)

    def test_replay_comparison_remains_json_type_sensitive(self):
        work1 = canonical_work_id("req", "prefill", 0, 0, 1)
        differing_pairs = (
            ({"total_bytes": 1}, {"total_bytes": 1.0}),
            ({"total": {"ops": [1, None]}},
             {"total": {"ops": [None, 1]}}),
        )
        for first_extra, duplicate_extra in differing_pairs:
            with self.subTest(first_extra=first_extra):
                ledger = WorkLedger()
                ledger.apply_payload(
                    _envelope(0, 10, "full", watermark=10,
                              state=_state(_node(1))),
                    delivery_sequence=0, tick=10)
                observed = []
                with self.assertRaisesRegex(
                        WorkLedgerError, "different content"):
                    ledger.apply_payload(
                        _envelope(
                            1, 20, "delta", watermark=10,
                            events=[
                                _event("issue", work1, 10, **first_extra),
                                _event("issue", work1, 10,
                                       **duplicate_extra)]),
                        delivery_sequence=1, tick=20,
                        observer=lambda event, applied: observed.append(
                            applied))
                self.assertEqual(observed, [True])
                self.assertEqual(ledger.work(work1).state, "issued")
                self.assertEqual(ledger.replayed_event_count, 0)
                self.assertEqual(ledger.last_delivery_sequence, 0)
                self.assertEqual(ledger.version, 1)


if __name__ == "__main__":
    unittest.main()
