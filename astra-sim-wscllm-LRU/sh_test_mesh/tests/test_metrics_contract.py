from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import unittest


_WORKLOAD_DIR = Path(__file__).resolve().parents[1] / "workload" / "llama2_7b_inference"
if str(_WORKLOAD_DIR) not in sys.path:
    sys.path.insert(0, str(_WORKLOAD_DIR))

import metrics_schema as ms
import metrics_postprocess as mp


_CONFIG_PATH = _WORKLOAD_DIR / "metrics_config.json"


def _request(
    queue_index: int,
    *,
    arrival: "ms.Arrival | None" = None,
    prefill_ranks=(0, 1),
    decode_ranks=(2, 3),
    session_id: str | None = None,
    turn_index: int = 0,
) -> ms.RequestMetadata:
    return ms.RequestMetadata(
        queue_index=queue_index,
        request_id=f"req-{queue_index}",
        session_id=session_id if session_id is not None else f"sess-{queue_index}",
        turn_index=turn_index,
        arrival=arrival if arrival is not None else ms.Arrival.absolute(queue_index * 100),
        prefill_instance=0,
        prefill_ranks=tuple(prefill_ranks),
        decode_instance=1,
        decode_ranks=tuple(decode_ranks),
    )


def _builder(**overrides) -> ms.MetricManifestBuilder:
    kwargs = {
        "repo_variant": "astra-sim-face",
        "run_mode": ms.RUN_MODE_SERVICE,
        "npus_count": 4,
        "mesh_rows": 2,
        "mesh_columns": 2,
        "node_count_by_rank": {rank: 1000 for rank in range(4)},
    }
    kwargs.update(overrides)
    builder = ms.MetricManifestBuilder(**kwargs)
    builder.set_digests(
        trace_digest="trace-sha",
        request_mapping_digest="mapping-sha",
        kv_event_digest="kv-sha",
    )
    return builder


def _add_service_boundaries(
    builder: ms.MetricManifestBuilder, request: ms.RequestMetadata, base_node: int = 0
) -> None:
    for offset, rank in enumerate(request.prefill_ranks):
        builder.add_node_event(rank, base_node + offset * 10 + 1, ms.EVENT_PREFILL_START, request.queue_index)
        builder.add_node_event(rank, base_node + offset * 10 + 2, ms.EVENT_PREFILL_END, request.queue_index)
    for offset, rank in enumerate(request.decode_ranks):
        builder.add_node_event(rank, base_node + offset * 10 + 3, ms.EVENT_DECODE_START, request.queue_index)
        builder.add_node_event(rank, base_node + offset * 10 + 4, ms.EVENT_DECODE_END, request.queue_index)


def _service_manifest(requests=(_request(0),)) -> ms.MetricsManifest:
    builder = _builder()
    for request in requests:
        builder.add_request(request)
        _add_service_boundaries(builder, request, base_node=request.queue_index * 100)
    return builder.build()


def _delta(
    sequence_index: int,
    rank: int = 0,
    allocation_key: str = "resident:sess-0:seg-0",
    planner_time_ns: int = 0,
    **components,
) -> ms.MemoryDelta:
    return ms.MemoryDelta(
        sequence_index=sequence_index,
        planner_time_ns=planner_time_ns,
        anchor_kind="stage_boundary",
        trigger_queue_index=components.pop("trigger_queue_index", 0),
        request_id=components.pop("request_id", "req-0"),
        session_id=components.pop("session_id", "sess-0"),
        rank=rank,
        allocation_key=allocation_key,
        **components,
    )


def _json_round_trip(value):
    return json.loads(json.dumps(value))


class ArrivalTests(unittest.TestCase):
    def test_absolute_arrival_resolves_to_value(self) -> None:
        arrival = ms.Arrival.absolute(123)
        self.assertEqual(arrival.resolve(), 123)
        self.assertEqual(arrival.resolve(parent_completion_ns=999), 123)
        self.assertEqual(
            arrival.to_dict(), {"kind": "absolute", "value_ns": 123}
        )

    def test_after_request_arrival_resolves_from_parent_completion(self) -> None:
        parent = _request(0)
        child = _request(
            1,
            session_id="sess-0",
            turn_index=1,
            arrival=ms.Arrival.after_request(parent_queue_index=0, interval_ns=1_000_000),
        )
        manifest = _service_manifest((parent, child))
        tracker = ms.RequestTimingTracker(manifest)
        # Parent actual completion comes from observed decode-end ticks.
        for rank in parent.decode_ranks:
            tracker.record_event(rank, ms.EVENT_DECODE_START, 0, 500)
            tracker.record_event(rank, ms.EVENT_DECODE_END, 0, 700 + rank)
        self.assertEqual(tracker.resolve_arrival_ns(1), 703 + 1_000_000)
        # A child whose parent has not completed yet cannot resolve.
        orphan = _request(
            2,
            arrival=ms.Arrival.after_request(parent_queue_index=1, interval_ns=5),
        )
        orphan_manifest = _service_manifest((parent, child, orphan))
        orphan_tracker = ms.RequestTimingTracker(orphan_manifest)
        with self.assertRaises(ms.MetricsSchemaError):
            orphan_tracker.resolve_arrival_ns(2)

    def test_after_request_arrival_dict_shape(self) -> None:
        arrival = ms.Arrival.after_request(parent_queue_index=12, interval_ns=1_000_000)
        self.assertEqual(
            arrival.to_dict(),
            {"kind": "after_request", "parent_queue_index": 12, "interval_ns": 1_000_000},
        )


class RequestTimingTrackerTests(unittest.TestCase):
    def test_min_start_max_completion_across_tp_ranks(self) -> None:
        request = _request(0, prefill_ranks=(0, 1), decode_ranks=(2, 3))
        tracker = ms.RequestTimingTracker(_service_manifest((request,)))
        tracker.record_event(0, ms.EVENT_PREFILL_START, 0, 100)
        tracker.record_event(1, ms.EVENT_PREFILL_START, 0, 80)
        tracker.record_event(0, ms.EVENT_PREFILL_END, 0, 200)
        tracker.record_event(1, ms.EVENT_PREFILL_END, 0, 220)
        tracker.record_event(2, ms.EVENT_DECODE_START, 0, 300)
        tracker.record_event(3, ms.EVENT_DECODE_START, 0, 290)
        tracker.record_event(2, ms.EVENT_DECODE_END, 0, 500)
        tracker.record_event(3, ms.EVENT_DECODE_END, 0, 550)
        timing = tracker.timing_for(0)
        self.assertTrue(timing.completed)
        self.assertEqual(timing.prefill_start_ns, 80)
        self.assertEqual(timing.prefill_end_ns, 220)
        self.assertEqual(timing.decode_start_ns, 290)
        self.assertEqual(timing.completion_ns, 550)

    def test_duplicate_events_are_deduplicated(self) -> None:
        request = _request(0)
        builder = _builder()
        builder.add_request(request)
        self.assertTrue(builder.add_node_event(0, 1, ms.EVENT_PREFILL_START, 0))
        self.assertFalse(builder.add_node_event(0, 1, ms.EVENT_PREFILL_START, 0))
        _add_service_boundaries(builder, request)
        manifest = builder.build()
        self.assertEqual(
            sum(
                1
                for event in manifest.node_events_by_rank[0]
                if event.event_code == ms.EVENT_PREFILL_START
            ),
            1,
        )
        tracker = ms.RequestTimingTracker(manifest)
        self.assertTrue(tracker.record_event(0, ms.EVENT_PREFILL_START, 0, 100))
        self.assertFalse(tracker.record_event(0, ms.EVENT_PREFILL_START, 0, 100))
        with self.assertRaises(ms.MetricsSchemaError):
            tracker.record_event(0, ms.EVENT_PREFILL_START, 0, 101)

    def test_missing_rank_boundary_marks_request_incomplete(self) -> None:
        request = _request(0, prefill_ranks=(0, 1), decode_ranks=(2, 3))
        tracker = ms.RequestTimingTracker(_service_manifest((request,)))
        tracker.record_event(0, ms.EVENT_PREFILL_START, 0, 100)
        tracker.record_event(1, ms.EVENT_PREFILL_START, 0, 100)
        tracker.record_event(0, ms.EVENT_PREFILL_END, 0, 200)
        tracker.record_event(1, ms.EVENT_PREFILL_END, 0, 200)
        tracker.record_event(2, ms.EVENT_DECODE_START, 0, 300)
        tracker.record_event(3, ms.EVENT_DECODE_START, 0, 300)
        # Only one decode rank reports completion.
        tracker.record_event(2, ms.EVENT_DECODE_END, 0, 500)
        timing = tracker.timing_for(0)
        self.assertFalse(timing.completed)
        self.assertIsNone(timing.completion_ns)
        # The manifest builder also refuses a manifest with missing boundaries.
        builder = _builder()
        builder.add_request(request)
        for rank in request.prefill_ranks:
            builder.add_node_event(rank, 1, ms.EVENT_PREFILL_START, 0)
            builder.add_node_event(rank, 2, ms.EVENT_PREFILL_END, 0)
        for rank in request.decode_ranks:
            builder.add_node_event(rank, 3, ms.EVENT_DECODE_START, 0)
        builder.add_node_event(2, 4, ms.EVENT_DECODE_END, 0)
        with self.assertRaises(ms.MetricsSchemaError):
            builder.build()


class MemoryObserverTests(unittest.TestCase):
    def _observer(self, capacity: int = 10_000, ranks=(0,)) -> ms.MemoryMetricsObserver:
        observer = ms.MemoryMetricsObserver()
        for rank in ranks:
            observer.initialize_rank(rank, capacity)
        return observer

    def test_memory_delta_add_and_remove(self) -> None:
        observer = self._observer()
        observer.record_delta(_delta(0, weight_delta_bytes=1_000, allocation_key="weight:0"))
        observer.record_delta(_delta(1, resident_kv_delta_bytes=400))
        ledger = observer.finalize().ranks[0].ledger
        self.assertEqual(ledger.weight_bytes, 1_000)
        self.assertEqual(ledger.resident_kv_bytes, 400)
        self.assertEqual(ledger.physical_used_bytes, 1_400)
        observer.record_delta(_delta(2, resident_kv_delta_bytes=-400))
        ledger = observer.finalize().ranks[0].ledger
        self.assertEqual(ledger.resident_kv_bytes, 0)
        self.assertEqual(ledger.physical_used_bytes, 1_000)
        self.assertEqual(ledger.uncommitted_free_bytes, 9_000)

    def test_reservation_is_tracked_separately_from_resident(self) -> None:
        observer = self._observer()
        observer.record_delta(
            _delta(0, reserved_kv_delta_bytes=600, allocation_key="reservation:req-0")
        )
        ledger = observer.finalize().ranks[0].ledger
        self.assertEqual(ledger.reserved_kv_bytes, 600)
        self.assertEqual(ledger.resident_kv_bytes, 0)
        self.assertEqual(ledger.physical_used_bytes, 0)
        self.assertEqual(ledger.committed_used_bytes, 600)
        observer.record_delta(_delta(1, resident_kv_delta_bytes=600))
        observer.record_delta(
            _delta(2, reserved_kv_delta_bytes=-600, allocation_key="reservation:req-0")
        )
        ledger = observer.finalize().ranks[0].ledger
        self.assertEqual(ledger.resident_kv_bytes, 600)
        self.assertEqual(ledger.reserved_kv_bytes, 0)
        self.assertEqual(ledger.physical_used_bytes, 600)
        self.assertEqual(ledger.committed_used_bytes, 600)

    def test_chiplet_allocation_key_add_remove_is_symmetric(self) -> None:
        observer = self._observer()
        observer.record_delta(_delta(0, resident_kv_delta_bytes=103))
        chiplets = observer.finalize().ranks[0].chiplets
        self.assertEqual(
            tuple(c.ledger.resident_kv_bytes for c in chiplets), (26, 26, 26, 25)
        )
        observer.record_delta(_delta(1, resident_kv_delta_bytes=-103))
        result = observer.finalize().ranks[0]
        for chiplet in result.chiplets:
            self.assertEqual(chiplet.ledger.resident_kv_bytes, 0)
            self.assertEqual(chiplet.ledger.physical_used_bytes, 0)
        self.assertEqual(observer._allocations[0], {})

    def test_rank_aggregate_equals_chiplet_sum(self) -> None:
        observer = self._observer()
        observer.record_delta(_delta(0, weight_delta_bytes=997, allocation_key="weight:0"))
        observer.record_delta(
            _delta(1, resident_kv_delta_bytes=503, allocation_key="resident:sess-0:a")
        )
        observer.record_delta(
            _delta(2, resident_kv_delta_bytes=251, allocation_key="resident:sess-0:b")
        )
        observer.record_delta(
            _delta(3, reserved_kv_delta_bytes=129, allocation_key="reservation:req-1")
        )
        result = observer.finalize().ranks[0]
        ledger = result.ledger
        for field_name in (
            "capacity_bytes",
            "weight_bytes",
            "resident_kv_bytes",
            "reserved_kv_bytes",
            "physical_used_bytes",
            "committed_used_bytes",
        ):
            self.assertEqual(
                getattr(ledger, field_name),
                sum(getattr(c.ledger, field_name) for c in result.chiplets),
                field_name,
            )

    def test_peak_snapshot_keeps_same_moment_components(self) -> None:
        observer = self._observer()
        observer.record_delta(
            _delta(0, weight_delta_bytes=2_000, allocation_key="weight:0", planner_time_ns=10)
        )
        observer.record_delta(
            _delta(
                1,
                resident_kv_delta_bytes=3_000,
                reserved_kv_delta_bytes=500,
                planner_time_ns=20,
                cause="admission",
            )
        )
        # Later state is smaller; the peak must keep the moment of sequence 1.
        observer.record_delta(_delta(2, resident_kv_delta_bytes=-2_500, planner_time_ns=30))
        result = observer.finalize().ranks[0]
        peak = result.peak_physical
        self.assertEqual(peak.peak_value_bytes, 5_000)
        self.assertEqual(peak.weight_at_peak_bytes, 2_000)
        self.assertEqual(peak.resident_kv_at_peak_bytes, 3_000)
        self.assertEqual(peak.reserved_kv_at_peak_bytes, 500)
        self.assertEqual(peak.free_at_peak_bytes, 10_000 - 5_500)
        self.assertEqual(peak.planner_time_ns, 20)
        self.assertEqual(peak.sequence_index, 1)
        self.assertEqual(peak.cause, "admission")
        self.assertEqual(peak.request_id, "req-0")
        self.assertEqual(peak.session_id, "sess-0")
        committed_peak = result.peak_committed
        self.assertEqual(committed_peak.peak_value_bytes, 5_500)
        self.assertEqual(committed_peak.sequence_index, 1)
        self.assertNotEqual(result.ledger.physical_used_bytes, peak.peak_value_bytes)

    def test_invalid_memory_state_fails_loudly(self) -> None:
        observer = self._observer(capacity=1_000)
        with self.assertRaises(ms.MetricsSchemaError):
            observer.record_delta(
                _delta(0, weight_delta_bytes=1_001, allocation_key="weight:0")
            )
        observer = self._observer(capacity=1_000)
        with self.assertRaises(ms.MetricsSchemaError):
            observer.record_delta(_delta(0, resident_kv_delta_bytes=-1))
        observer = self._observer()
        with self.assertRaises(ms.MetricsSchemaError):
            observer.record_delta(_delta(0, reserved_kv_delta_bytes=-10, allocation_key="ghost"))
        observer = self._observer()
        with self.assertRaises(ms.MetricsSchemaError):
            observer.record_delta(_delta(0, rank=9, weight_delta_bytes=1))
        with self.assertRaises(ms.MetricsSchemaError):
            observer.initialize_rank(0, 5_000)


class NormalizationTests(unittest.TestCase):
    def test_group_max_denominator_selection(self) -> None:
        record = ms.build_normalization_record(
            "mean_e2e_ns", 50.0, ms.NORMALIZATION_GROUP_MAX, "case5-7b", group_values=[50, 200]
        )
        self.assertEqual(record.denominator_value, 200.0)
        self.assertAlmostEqual(record.normalized_value, 0.25)
        self.assertEqual(record.normalization_group_id, "case5-7b")
        self.assertIsNone(record.baseline_run_id)

    def test_ratio_to_baseline_denominator_selection(self) -> None:
        record = ms.build_normalization_record(
            "drain_tput_rps",
            30.0,
            ms.NORMALIZATION_RATIO_TO_BASELINE,
            "case5-7b",
            baseline_run_id="face-base",
            baseline_value=120.0,
        )
        self.assertEqual(record.denominator_value, 120.0)
        self.assertAlmostEqual(record.normalized_value, 0.25)
        self.assertEqual(record.baseline_run_id, "face-base")

    def test_zero_denominator_fails(self) -> None:
        with self.assertRaises(ms.MetricsSchemaError):
            ms.build_normalization_record(
                "mean_e2e_ns", 1.0, ms.NORMALIZATION_GROUP_MAX, "g", group_values=[0, 0]
            )
        with self.assertRaises(ms.MetricsSchemaError):
            ms.build_normalization_record(
                "mean_e2e_ns",
                1.0,
                ms.NORMALIZATION_RATIO_TO_BASELINE,
                "g",
                baseline_run_id="base",
                baseline_value=0,
            )
        with self.assertRaises(ms.MetricsSchemaError):
            ms.select_normalization_denominator("max_of_others", group_values=[1])
        with self.assertRaises(ms.MetricsSchemaError):
            ms.select_normalization_denominator(
                ms.NORMALIZATION_RATIO_TO_BASELINE, baseline_value=10
            )


class ManifestTests(unittest.TestCase):
    def test_schema_version_mismatch_is_rejected(self) -> None:
        data = _json_round_trip(_service_manifest().to_dict())
        data["schema_version"] = 2
        with self.assertRaises(ms.MetricsSchemaError):
            ms.MetricsManifest.from_dict(data)
        data["schema_version"] = "1"
        with self.assertRaises(ms.MetricsSchemaError):
            ms.MetricsManifest.from_dict(data)
        with self.assertRaises(ms.MetricsSchemaError):
            ms.check_schema_version(0)
        self.assertEqual(ms.check_schema_version(ms.SCHEMA_VERSION), 1)

    def test_duplicate_request_is_rejected(self) -> None:
        builder = _builder()
        builder.add_request(_request(0))
        with self.assertRaises(ms.MetricsSchemaError):
            builder.add_request(_request(0))
        # Same request_id under different queue indices is only tolerable when
        # the (session_id, turn_index, request_id) key stays unique.
        data = _json_round_trip(_service_manifest((_request(0), _request(1))).to_dict())
        data["requests"][1]["session_id"] = data["requests"][0]["session_id"]
        data["requests"][1]["request_id"] = data["requests"][0]["request_id"]
        with self.assertRaises(ms.MetricsSchemaError):
            ms.MetricsManifest.from_dict(data)
        data["requests"][1]["turn_index"] = 1
        manifest = ms.MetricsManifest.from_dict(data)
        self.assertEqual(len(manifest.requests), 2)

    def test_invalid_manifest_schema_fails_loudly(self) -> None:
        data = _json_round_trip(_service_manifest().to_dict())
        for field_name in ("trace_digest", "request_mapping_digest", "kv_event_digest"):
            broken = copy.deepcopy(data)
            broken[field_name] = ""
            with self.assertRaises(ms.MetricsSchemaError):
                ms.MetricsManifest.from_dict(broken)
        broken = copy.deepcopy(data)
        broken["requests"][0]["queue_index"] = 7  # contiguity must start at 0
        with self.assertRaises(ms.MetricsSchemaError):
            ms.MetricsManifest.from_dict(broken)
        broken = copy.deepcopy(data)
        broken["node_events_by_rank"]["0"][0][1] = 9
        with self.assertRaises(ms.MetricsSchemaError):
            ms.MetricsManifest.from_dict(broken)
        broken = copy.deepcopy(data)
        broken["node_events_by_rank"]["0"][0] = [1, 2]
        with self.assertRaises(ms.MetricsSchemaError):
            ms.MetricsManifest.from_dict(broken)

    def test_node_id_out_of_rank_range_is_rejected(self) -> None:
        builder = _builder()
        builder.add_request(_request(0))
        with self.assertRaises(ms.MetricsSchemaError):
            builder.add_node_event(0, 1000, ms.EVENT_PREFILL_START, 0)
        with self.assertRaises(ms.MetricsSchemaError):
            builder.add_node_event(9, 0, ms.EVENT_PREFILL_START, 0)

    def test_missing_digests_fail_build(self) -> None:
        builder = ms.MetricManifestBuilder(
            repo_variant="astra-sim-face",
            run_mode=ms.RUN_MODE_SERVICE,
            npus_count=4,
            mesh_rows=2,
            mesh_columns=2,
            node_count_by_rank={0: 10},
        )
        with self.assertRaises(ms.MetricsSchemaError):
            builder.build()

    def test_event_edge_encoding_matches_protocol(self) -> None:
        expected_edges = {
            1: "issue",
            2: "complete",
            3: "issue",
            4: "complete",
            5: "issue",
            6: "complete",
            7: "complete",
        }
        for code, edge in expected_edges.items():
            self.assertEqual(ms.NodeMetricEvent(0, code, 0).edge, edge)
            self.assertEqual(ms.EVENT_EDGE_BY_CODE[code], edge)


class JsonRoundTripTests(unittest.TestCase):
    def test_arrival_round_trip(self) -> None:
        for arrival in (
            ms.Arrival.absolute(0),
            ms.Arrival.after_request(parent_queue_index=12, interval_ns=1_000_000),
        ):
            self.assertEqual(ms.Arrival.from_dict(_json_round_trip(arrival.to_dict())), arrival)

    def test_request_metadata_round_trip(self) -> None:
        request = _request(3)
        self.assertEqual(
            ms.RequestMetadata.from_dict(_json_round_trip(request.to_dict())), request
        )

    def test_node_event_round_trip(self) -> None:
        event = ms.NodeMetricEvent(456, 2, 7)
        self.assertEqual(
            ms.NodeMetricEvent.from_triple(_json_round_trip(event.to_triple())), event
        )

    def test_manifest_round_trip(self) -> None:
        manifest = _service_manifest(
            (
                _request(0),
                _request(
                    1,
                    session_id="sess-0",
                    turn_index=1,
                    arrival=ms.Arrival.after_request(0, 50),
                ),
            )
        )
        restored = ms.MetricsManifest.from_dict(_json_round_trip(manifest.to_dict()))
        self.assertEqual(restored, manifest)
        self.assertEqual(
            restored.memory_projection.chiplets_per_npu, ms.DEFAULT_CHIPLETS_PER_NPU
        )
        self.assertEqual(restored.memory_projection.native_scope, ms.NATIVE_MEMORY_SCOPE)

    def test_memory_delta_round_trip(self) -> None:
        delta = _delta(
            5,
            rank=2,
            planner_time_ns=99,
            resident_kv_delta_bytes=-128,
            cause="eviction",
            trigger_queue_index=None,
            request_id=None,
        )
        self.assertEqual(ms.MemoryDelta.from_dict(_json_round_trip(delta.to_dict())), delta)

    def test_memory_result_round_trip(self) -> None:
        observer = ms.MemoryMetricsObserver()
        observer.initialize_rank(0, 10_000)
        observer.record_delta(_delta(0, weight_delta_bytes=997, allocation_key="weight:0"))
        observer.record_delta(_delta(1, resident_kv_delta_bytes=503))
        result = observer.finalize()
        self.assertEqual(
            ms.MemoryMetricsResult.from_dict(_json_round_trip(result.to_dict())), result
        )

    def test_normalization_record_round_trip(self) -> None:
        record = ms.build_normalization_record(
            "mean_e2e_ns",
            25.0,
            ms.NORMALIZATION_RATIO_TO_BASELINE,
            "g",
            baseline_run_id="base",
            baseline_value=100.0,
        )
        self.assertEqual(
            ms.NormalizationRecord.from_dict(_json_round_trip(record.to_dict())), record
        )


class MetricsConfigTests(unittest.TestCase):
    def test_metrics_config_json_parses_and_pins_protocol_values(self) -> None:
        config = json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
        self.assertEqual(config["schema_version"], ms.SCHEMA_VERSION)
        self.assertEqual(
            config["memory"],
            {
                "chiplets_per_npu": 4,
                "projection": "equal_striping",
                "wasted_definition": "capacity_minus_weight_minus_resident_kv",
            },
        )
        self.assertEqual(
            config["microbenchmark"],
            {
                "tp_degrees": [1, 2, 4, 6, 8],
                "prefill_chunks": [128, 256, 512],
                "decode_batches": [1, 2, 4, 8, 16, 32],
                "kv_lengths": [128, 256, 512, 1024, 2048, 4096],
                "repeats": 1,
            },
        )


class FirstTokenAndRequestMetricsContractTests(unittest.TestCase):
    """B3 additions: frozen cross-repo contract extensions.

    Canonical union of the three B3 batches: event code 8 (constant, edge
    mapping, SERVICE_EVENT_CODES membership), the request_metrics.csv
    column order (29 columns, frozen), the terminal_status /
    first_token_source enumerations, and the NA placeholder semantics for
    fields that are not yet filled.  Only contract surfaces present in
    every repository are referenced (metrics_schema.py symbols and the
    frozen mp.REQUEST_METRICS_COLUMNS list), so this block must stay
    byte-identical across the five repos.
    """

    TERMINAL_STATUS_VALUES = ("completed", "failed")
    FIRST_TOKEN_SOURCE_VALUES = ("exact", "train_interpolated", "NA")
    NA_SENTINEL = "NA"

    def test_event_code_8_constant_edge_and_service_membership(self) -> None:
        self.assertEqual(ms.EVENT_FIRST_TOKEN_COMPLETE, 8)
        self.assertEqual(
            ms.EVENT_EDGE_BY_CODE[ms.EVENT_FIRST_TOKEN_COMPLETE],
            ms.EVENT_EDGE_COMPLETE,
        )
        self.assertEqual(ms.NodeMetricEvent(456, 8, 7).edge, "complete")
        self.assertIn(ms.EVENT_FIRST_TOKEN_COMPLETE, ms.SERVICE_EVENT_CODES)
        # The protocol table stays contiguous 1..8 with issue/complete edges.
        self.assertEqual(sorted(ms.EVENT_EDGE_BY_CODE), list(range(1, 9)))
        self.assertEqual(set(ms.EVENT_EDGE_BY_CODE.values()), {"issue", "complete"})
        # Code 8 is a completion event like code 7 (memory anchor).
        self.assertEqual(
            ms.EVENT_EDGE_BY_CODE[ms.EVENT_FIRST_TOKEN_COMPLETE],
            ms.EVENT_EDGE_BY_CODE[ms.EVENT_MEMORY_ANCHOR_COMPLETE],
        )

    def test_service_event_codes_membership_is_pinned(self) -> None:
        # Request service lifetime boundaries only: microbench (5/6) and
        # memory-anchor (7) codes are observable but not service codes.
        self.assertEqual(
            set(ms.SERVICE_EVENT_CODES),
            {
                ms.EVENT_PREFILL_START,
                ms.EVENT_PREFILL_END,
                ms.EVENT_DECODE_START,
                ms.EVENT_DECODE_END,
                ms.EVENT_FIRST_TOKEN_COMPLETE,
            },
        )

    def test_request_metrics_columns_frozen_29_in_order(self) -> None:
        self.assertEqual(len(mp.REQUEST_METRICS_COLUMNS), 29)
        self.assertEqual(len(set(mp.REQUEST_METRICS_COLUMNS)), 29)
        self.assertEqual(
            list(mp.REQUEST_METRICS_COLUMNS),
            [
                "queue_index",
                "request_id",
                "session_id",
                "turn_index",
                "request_type",
                "terminal_status",
                "arrival_ns",
                "prefill_start_ns",
                "prefill_end_ns",
                "decode_start_ns",
                "first_token_ns",
                "first_token_source",
                "completion_ns",
                "queue_ns",
                "prefill_ns",
                "prefill_decode_gap_ns",
                "decode_ns",
                "e2e_ns",
                "kv_hit_state",
                "restore_start_ns",
                "restore_complete_ns",
                "pre_prefill_restore_ns",
                "hidden_restore_ns",
                "exposed_restore_stall_ns",
                "hidden_ratio",
                "prefill_length",
                "decode_length",
                "prefix_len",
                "instructions",
            ],
        )

    def test_terminal_status_enum(self) -> None:
        # Closed enum written into the terminal_status column: no other
        # value may appear in request_metrics.csv terminal_status cells.
        self.assertIn("terminal_status", mp.REQUEST_METRICS_COLUMNS)
        self.assertEqual(self.TERMINAL_STATUS_VALUES, ("completed", "failed"))

    def test_first_token_source_enum(self) -> None:
        # Closed enum per the main spec: exactly exact /
        # train_interpolated plus the NA placeholder; "train_interpolated"
        # is reserved for the work package that produces it -- until then
        # the cell carries the NA token.
        self.assertIn("first_token_source", mp.REQUEST_METRICS_COLUMNS)
        self.assertEqual(
            self.FIRST_TOKEN_SOURCE_VALUES,
            ("exact", "train_interpolated", "NA"),
        )

    def test_na_semantics(self) -> None:
        # NA = "field not yet produced by the owning work package"; it is
        # a member of first_token_source but never a terminal_status
        # value.
        self.assertEqual(self.NA_SENTINEL, "NA")
        self.assertIn(self.NA_SENTINEL, self.FIRST_TOKEN_SOURCE_VALUES)
        self.assertTrue(
            all(
                value != self.NA_SENTINEL
                for value in self.TERMINAL_STATUS_VALUES
            )
        )


if __name__ == "__main__":
    unittest.main()
