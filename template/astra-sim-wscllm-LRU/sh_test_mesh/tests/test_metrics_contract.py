from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import tempfile
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


def _service_manifest_dict(requests=(_request(0),)) -> dict:
    """Raw ``metrics_manifest.json`` fixture dict.

    M6-09 (2026-09-26): the ET-generator ``MetricManifestBuilder`` fixture
    path is gone; manifests are assembled as plain dicts and validated
    through ``MetricsManifest.from_dict`` like real sidecars.
    """

    events: dict[str, list[list[int]]] = {}
    for request in requests:
        base = request.queue_index * 100
        for offset, rank in enumerate(request.prefill_ranks):
            events.setdefault(str(rank), []).append(
                [base + offset * 10 + 1, ms.EVENT_PREFILL_START, request.queue_index]
            )
            events.setdefault(str(rank), []).append(
                [base + offset * 10 + 2, ms.EVENT_PREFILL_END, request.queue_index]
            )
        for offset, rank in enumerate(request.decode_ranks):
            events.setdefault(str(rank), []).append(
                [base + offset * 10 + 3, ms.EVENT_DECODE_START, request.queue_index]
            )
            events.setdefault(str(rank), []).append(
                [base + offset * 10 + 4, ms.EVENT_DECODE_END, request.queue_index]
            )
    return {
        "schema_version": ms.SCHEMA_VERSION,
        # E6（2026-09-30）：通用夹具的 repo_variant 用本仓真值（旧值
        # "astra-sim-face" 是跨仓拷贝残留；schema 不钉常量故测试仍绿，
        # 但误导排障）。
        "repo_variant": "astra-sim-wscllm",
        "run_mode": ms.RUN_MODE_SERVICE,
        "npus_count": 4,
        "mesh": {"rows": 2, "columns": 2},
        "trace_digest": "trace-sha",
        "request_mapping_digest": "mapping-sha",
        "kv_event_digest": "kv-sha",
        "requests": [request.to_dict() for request in requests],
        "node_events_by_rank": events,
    }


def _service_manifest(requests=(_request(0),)) -> ms.MetricsManifest:
    return ms.MetricsManifest.from_dict(_service_manifest_dict(requests))


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
        child = _request(
            1,
            session_id="sess-0",
            turn_index=1,
            arrival=ms.Arrival.after_request(parent_queue_index=0, interval_ns=1_000_000),
        )
        # In production the parent completion tick is the max decode-end
        # tick across the parent's decode ranks (rank 3 of queue 0 observed
        # at 703 in the old RequestTimingTracker fixture -- the tracker is
        # gone, M6-09; resolve() only needs the tick itself).
        self.assertEqual(child.arrival.resolve(parent_completion_ns=703), 703 + 1_000_000)
        # A child whose parent has not completed yet cannot resolve.
        with self.assertRaises(ms.MetricsSchemaError):
            child.arrival.resolve()

    def test_after_request_arrival_dict_shape(self) -> None:
        arrival = ms.Arrival.after_request(parent_queue_index=12, interval_ns=1_000_000)
        self.assertEqual(
            arrival.to_dict(),
            {"kind": "after_request", "parent_queue_index": 12, "interval_ns": 1_000_000},
        )


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

    def test_failed_delta_leaves_observer_unchanged(self) -> None:
        """MEM-05 (2026-09-26): a rejected delta must not leave a half-
        applied ledger.  Both failure modes below used to mutate the rank
        state BEFORE raising (invariant check after +=; projection raise
        after the rank commit) -- validate-then-commit keeps the whole
        RankMemoryResult (rank ledger, chiplets, peaks) bit-identical."""

        observer = self._observer(capacity=1_000)
        observer.record_delta(_delta(0, weight_delta_bytes=600, allocation_key="weight:0"))
        baseline = observer.finalize().ranks[0]
        # Rank-invariant violation: physical 600+500 exceeds capacity 1_000.
        with self.assertRaises(ms.MetricsSchemaError):
            observer.record_delta(_delta(1, resident_kv_delta_bytes=500))
        # Projection rejection: net-negative components on an unknown
        # allocation key (the rank invariant itself would hold here).
        with self.assertRaises(ms.MetricsSchemaError):
            observer.record_delta(
                _delta(2, weight_delta_bytes=100, resident_kv_delta_bytes=-50,
                       allocation_key="ghost")
            )
        self.assertEqual(observer.finalize().ranks[0], baseline)
        # The observer still accepts a valid follow-up delta afterwards.
        observer.record_delta(_delta(3, resident_kv_delta_bytes=400))
        ledger = observer.finalize().ranks[0].ledger
        self.assertEqual(ledger.weight_bytes, 600)
        self.assertEqual(ledger.resident_kv_bytes, 400)
        self.assertEqual(ledger.physical_used_bytes, 1_000)


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
        # Duplicate queue_index is rejected by the manifest reader (the
        # ET-generator builder that used to guard this is gone, M6-09).
        data = _service_manifest_dict((_request(0), _request(1)))
        data["requests"][1]["queue_index"] = 0
        with self.assertRaises(ms.MetricsSchemaError):
            ms.MetricsManifest.from_dict(data)
        # Same request_id under different queue indices is only tolerable when
        # the (session_id, turn_index, request_id) key stays unique.
        data = _service_manifest_dict((_request(0), _request(1)))
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
        # Events listed for a rank beyond npus_count are rejected by the
        # manifest reader (the rank-range guard moved here from the deleted
        # ET-generator builder, M6-09).
        data = _service_manifest_dict()
        data["node_events_by_rank"]["9"] = [[0, ms.EVENT_PREFILL_START, 0]]
        with self.assertRaises(ms.MetricsSchemaError):
            ms.MetricsManifest.from_dict(data)

    def test_missing_digests_parse_aligned_with_protocol(self) -> None:
        """M-22 (2026-09-30): the current manifest protocol (plan_
        materializer writer + C++ load_manifest reader) carries no digest
        keys at all, so their absence must parse; only a *present* empty
        string is rejected (test_invalid_manifest_schema_fails_loudly)."""
        data = _service_manifest_dict()
        for field_name in ("trace_digest", "request_mapping_digest",
                           "kv_event_digest"):
            del data[field_name]
        manifest = ms.MetricsManifest.from_dict(data)
        self.assertIsNone(manifest.trace_digest)
        self.assertIsNone(manifest.request_mapping_digest)
        self.assertIsNone(manifest.kv_event_digest)

    def test_event_edge_encoding_matches_protocol(self) -> None:
        # M-22 (2026-09-30): microbenchmark iteration codes 5/6 are retired
        # from the legal domain (no producer exists in this repo anymore).
        expected_edges = {
            1: "issue",
            2: "complete",
            3: "issue",
            4: "complete",
            7: "complete",
        }
        for code, edge in expected_edges.items():
            self.assertEqual(ms.NodeMetricEvent(0, code, 0).edge, edge)
            self.assertEqual(ms.EVENT_EDGE_BY_CODE[code], edge)
        for retired in (5, 6):
            with self.assertRaises(ms.MetricsSchemaError):
                ms.NodeMetricEvent(0, retired, 0)


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
        self.assertEqual(config["memory"], {"chiplets_per_npu": 4})


class FirstTokenAndRequestMetricsContractTests(unittest.TestCase):
    """B3 additions: frozen cross-repo contract extensions.

    Canonical union of the three B3 batches: event code 8 (constant, edge
    mapping, SERVICE_EVENT_CODES membership), the request_metrics.csv
    column order (29 columns, frozen), the terminal_status /
    first_token_source enumerations, and the NA placeholder semantics for
    fields that are not yet filled.  Only contract surfaces present in
    every repository are referenced (metrics_schema.py symbols and the
    frozen mp.REQUEST_METRICS_COLUMNS list).  E6（2026-09-30 注记修正）：
    M-22 起事件码域已仓特有（本仓退役微基准码 5/6，joint 等姊妹仓保留
    全 8 码），本块与姊妹仓同名测试块是**语义等价**而非字节一致——按
    旧注记"逐字节同步"会必然弄坏其中一仓；同步时以各仓自身码域为准。
    """

    # G7-2 修复 2026-09-27:枚举集改为生产模块真实导出(metrics_schema
    # 冻结常量),不再是"测试类局部常量 vs 硬编码元组"的重言自验——生产
    # 改枚举/拼写错误时本类用例真实变红。
    TERMINAL_STATUS_VALUES = ms.TERMINAL_STATUS_VALUES
    FIRST_TOKEN_SOURCE_VALUES = ms.FIRST_TOKEN_SOURCE_VALUES
    NA_SENTINEL = ms.NA

    def test_event_code_8_constant_edge_and_service_membership(self) -> None:
        self.assertEqual(ms.EVENT_FIRST_TOKEN_COMPLETE, 8)
        self.assertEqual(
            ms.EVENT_EDGE_BY_CODE[ms.EVENT_FIRST_TOKEN_COMPLETE],
            ms.EVENT_EDGE_COMPLETE,
        )
        self.assertEqual(ms.NodeMetricEvent(456, 8, 7).edge, "complete")
        self.assertIn(ms.EVENT_FIRST_TOKEN_COMPLETE, ms.SERVICE_EVENT_CODES)
        # The retired microbenchmark codes 5/6 are out of the domain
        # (M-22); the table stays contiguous over the surviving codes.
        self.assertEqual(sorted(ms.EVENT_EDGE_BY_CODE), [1, 2, 3, 4, 7, 8])
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


class PostprocessGateTests(unittest.TestCase):
    """POST-01..05 / M6-03 回归：postprocess 解析与门禁 fail-closed 语义。

    每个用例在 /tmp 临时目录自建最小 [METRIC] 日志，直接驱动
    metrics_postprocess 的解析与门禁函数；main() 级用例的 CSV 产物也
    全部落临时目录。
    """

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="metrics_gate_test_")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _log(self, name: str, items) -> Path:
        path = Path(self._tmp.name) / name
        with path.open("w", encoding="utf-8") as sink:
            for item in items:
                sink.write(
                    item if isinstance(item, str) else "[METRIC] " + json.dumps(item)
                )
                sink.write("\n")
        return path

    def _out_paths(self) -> list[str]:
        base = Path(self._tmp.name)
        return [
            "--out-raw", str(base / "raw.csv"),
            "--out-normalized", str(base / "normalized.csv"),
            "--out-requests", str(base / "request.csv"),
        ]

    @staticmethod
    def _run_records(run_id="r1", detail_level="summary", consistency=True):
        records = [
            {"type": "init", "schema": 1, "run_id": run_id,
             "run_mode": "service", "detail_level": detail_level},
            {"type": "summary", "schema": 1,
             "completed_unique_requests": 0, "incomplete_requests": 0,
             "mean_e2e_ns": 100.0, "drain_tput_rps": 10.0},
        ]
        if consistency:
            records.append(
                {"type": "consistency", "schema": 1, "ok": True,
                 "violations": [], "dropped_events": 0}
            )
        return records

    def test_run_without_consistency_record_fails(self) -> None:
        """POST-01/F4/M6-01：零条 consistency 记录的最小 run 必须失败。"""
        path = self._log("no_consistency.log",
                         self._run_records(consistency=False))
        with self.assertRaises(mp.PostprocessError):
            mp._check_consistency(mp._parse_logs([path]))
        # 对照：同一 run 带 consistency 时门禁放行。
        path = self._log("clean.log", self._run_records())
        mp._check_consistency(mp._parse_logs([path]))

    def test_main_fails_without_consistency_and_writes_no_products(self) -> None:
        path = self._log("no_consistency.log",
                         self._run_records(consistency=False))
        rc = mp.main([str(path)] + self._out_paths())
        self.assertEqual(rc, 1)
        base = Path(self._tmp.name)
        self.assertFalse((base / "raw.csv").exists())
        self.assertFalse((base / "normalized.csv").exists())

    def test_minimal_clean_run_exits_zero(self) -> None:
        path = self._log("clean.log", self._run_records())
        rc = mp.main([str(path)] + self._out_paths())
        self.assertEqual(rc, 0)

    def test_duplicate_run_id_rejected_even_with_same_summary(self) -> None:
        """POST-02：相同 run_id + 相同 summary 的重复记录明确拒绝。"""
        records = self._run_records()
        path_a = self._log("a.log", records)
        path_b = self._log("b.log", records)
        with self.assertRaises(mp.PostprocessError):
            mp._parse_logs([path_a, path_b])
        # 同一日志内重复 init 同样拒绝。
        path_c = self._log("c.log", records + records)
        with self.assertRaises(mp.PostprocessError):
            mp._parse_logs([path_c])

    def test_nan_and_infinity_rejected_at_parse_layer(self) -> None:
        """POST-03：解析层拒绝 NaN/Infinity 常量。"""
        payloads = (
            '[METRIC] {"type": "summary", "schema": 1, "mean_e2e_ns": NaN}',
            '[METRIC] {"type": "summary", "schema": 1, "mean_e2e_ns": Infinity}',
            '[METRIC] {"type": "summary", "schema": 1, "mean_e2e_ns": -Infinity}',
        )
        for index, payload in enumerate(payloads):
            path = self._log(f"constant_{index}.log", [payload])
            with self.assertRaises(mp.PostprocessError):
                mp._parse_logs([path])

    def test_scalar_and_array_top_level_rejected(self) -> None:
        """POST-04：顶层 scalar/array 结构化失败，不再裸抛 AttributeError。"""
        for index, payload in enumerate(
            ('[METRIC] 42', '[METRIC] [1, 2]', '[METRIC] "x"', '[METRIC] null')
        ):
            path = self._log(f"top_{index}.log", [payload])
            with self.assertRaises(mp.PostprocessError) as context:
                mp._parse_logs([path])
            self.assertIn("must be an object", str(context.exception))

    def test_schema_rejects_non_finite_normalization_values(self) -> None:
        """POST-03：schema 数值校验拒绝非有限值（NaN 曾因比较恒假漏过）。"""
        with self.assertRaises(ms.MetricsSchemaError):
            ms.build_normalization_record(
                "mean_e2e_ns", float("nan"), ms.NORMALIZATION_GROUP_MAX, "g",
                group_values=[1.0],
            )
        with self.assertRaises(ms.MetricsSchemaError):
            ms.build_normalization_record(
                "mean_e2e_ns", 1.0, ms.NORMALIZATION_GROUP_MAX, "g",
                group_values=[float("inf")],
            )
        with self.assertRaises(ms.MetricsSchemaError):
            ms.build_normalization_record(
                "mean_e2e_ns", 1.0, ms.NORMALIZATION_RATIO_TO_BASELINE, "g",
                baseline_run_id="base", baseline_value=float("nan"),
            )
        # 有限值相除溢出为 inf 的归一化结果同样拒绝。
        with self.assertRaises(ms.MetricsSchemaError):
            ms.build_normalization_record(
                "mean_e2e_ns", 1e308, ms.NORMALIZATION_RATIO_TO_BASELINE, "g",
                baseline_run_id="base", baseline_value=1e-308,
            )

    def test_duplicate_queue_index_in_one_manifest_rejected(self) -> None:
        """POST-05：单 manifest 内重复 queue_index fail-closed；跨两个
        sidecar 的同名 queue_index 仍按字段互补合并（既有 join 口径）。"""
        run = mp.Run(run_id="r1", log_path=Path(self._tmp.name) / "a.log", init={})
        duplicate = {
            "requests": [
                {"queue_index": 0, "request_id": "req-0"},
                {"queue_index": 0, "request_id": "req-0"},
            ]
        }
        with self.assertRaises(mp.PostprocessError):
            mp._manifest_request_entries(run, {}, duplicate)
        merged, counts = mp._manifest_request_entries(
            run,
            {"requests": [{"queue_index": 0, "request_id": "req-0"}]},
            {"requests": [{"queue_index": 0, "request_id": "req-0",
                           "request_type": "human"}]},
        )
        self.assertEqual(merged[0]["request_id"], "req-0")
        self.assertEqual(merged[0]["request_type"], "human")
        self.assertEqual(
            counts, {"metrics_manifest.json": 1, "manifest.json": 1}
        )

    def test_full_detail_run_without_request_records_fails(self) -> None:
        """M6-03：full 档缺 request 记录即失败；summary 档/legacy init
        （无 detail_level 字段）照旧放行。"""
        path = self._log("full.log",
                         self._run_records(detail_level="full",
                                           consistency=False))
        with self.assertRaises(mp.PostprocessError):
            mp._check_request_completion(mp._parse_logs([path])[0])
        path = self._log("summary.log",
                         self._run_records(detail_level="summary",
                                           consistency=False))
        mp._check_request_completion(mp._parse_logs([path])[0])
        legacy = self._run_records(consistency=False)
        del legacy[0]["detail_level"]
        path = self._log("legacy.log", legacy)
        mp._check_request_completion(mp._parse_logs([path])[0])


class SnapshotFromDictTests(unittest.TestCase):
    """M6-12：两个 Snapshot 的 from_dict 与 RequestMetadata/MemoryDelta
    同严格度——字段集必须精确匹配，缺键/多键拒绝，不再静默 None。"""

    def test_ledger_snapshot_field_set_enforced(self) -> None:
        observer = ms.MemoryMetricsObserver()
        observer.initialize_rank(0, 10_000)
        ledger = observer.finalize().ranks[0].ledger
        data = _json_round_trip(ledger.to_dict())
        self.assertEqual(ms.MemoryLedgerSnapshot.from_dict(data), ledger)
        missing = dict(data)
        del missing["weight_bytes"]
        with self.assertRaises(ms.MetricsSchemaError):
            ms.MemoryLedgerSnapshot.from_dict(missing)
        unknown = dict(data)
        unknown["ghost_field"] = 1
        with self.assertRaises(ms.MetricsSchemaError):
            ms.MemoryLedgerSnapshot.from_dict(unknown)

    def test_peak_snapshot_field_set_enforced(self) -> None:
        observer = ms.MemoryMetricsObserver()
        observer.initialize_rank(0, 10_000)
        observer.record_delta(_delta(0, weight_delta_bytes=100, allocation_key="weight:0"))
        peak = observer.finalize().ranks[0].peak_physical
        data = _json_round_trip(peak.to_dict())
        self.assertEqual(ms.MemoryPeakSnapshot.from_dict(data), peak)
        missing = dict(data)
        del missing["planner_time_ns"]
        with self.assertRaises(ms.MetricsSchemaError):
            ms.MemoryPeakSnapshot.from_dict(missing)
        unknown = dict(data)
        unknown["ghost_field"] = 1
        with self.assertRaises(ms.MetricsSchemaError):
            ms.MemoryPeakSnapshot.from_dict(unknown)


class MetricsIntegrationExportTests(unittest.TestCase):
    """M6-10：__all__ 与模块 docstring 声称的两件公共件一致。"""

    def test_all_lists_both_public_pieces(self) -> None:
        import metrics_integration as mi

        self.assertEqual(
            sorted(mi.__all__),
            ["MemoryActionRecorder", "load_chiplets_per_npu"],
        )
        self.assertEqual(mi.load_chiplets_per_npu(), 4)


def _real_shape_metrics_manifest_dict() -> dict:
    """M-22 真实形态 fixture：plan_materializer 写出的 metrics_manifest.json
    全字段（逐字段照抄 plan_materializer.main/_derive_metrics_requests，
    含 schema_version/repo_variant/manifest_source/requests[]（带
    decode_length）/slo_sampling；无 run_mode/npus_count/mesh/digests/
    node_events_by_rank/memory_projection）。"""

    return {
        "schema_version": 1,
        "repo_variant": "astra-sim-wscllm",
        "manifest_source": "synthetic-prerun",
        "requests": [
            {
                "queue_index": 0,
                "request_id": "session_s1_request_0",
                "session_id": "s1",
                "turn_index": 0,
                "arrival": {"kind": "absolute", "value_ns": 1_000_000_000},
                "decode_length": 32,
                "prefill_instance": 0,
                "prefill_ranks": [0, 1, 2],
                "decode_instance": 0,
                "decode_ranks": [3, 4, 5],
            },
            {
                "queue_index": 1,
                "request_id": "session_s1_request_1",
                "session_id": "s1",
                "turn_index": 1,
                "arrival": {
                    "kind": "after_request",
                    "parent_queue_index": 0,
                    "interval_ns": 500_000_000,
                },
                "decode_length": 32,
                "prefill_instance": 0,
                "prefill_ranks": [0, 1, 2],
                "decode_instance": 0,
                "decode_ranks": [3, 4, 5],
            },
        ],
        "slo_sampling": {
            "watermark_period_ns": 5_000_000,
            "link_bucket_ns": 5_000_000,
            "provisional": True,
        },
    }


class ManifestProtocolAlignmentTests(unittest.TestCase):
    """M-22（2026-09-30）：frozen schema 与现行 manifest 协议对齐回归。

    双验证：真实形态 fixture（plan_materializer 产物形态）无损解析 +
    自产完整 fixture 往返（test_manifest_round_trip 承担）。
    """

    def test_real_shape_manifest_parses_losslessly(self) -> None:
        data = _real_shape_metrics_manifest_dict()
        manifest = ms.MetricsManifest.from_dict(data)
        # 缺省字段取 C++ 读侧同款缺省（run_mode）；未写 digest 保持 None。
        self.assertEqual(manifest.run_mode, ms.RUN_MODE_SERVICE)
        self.assertIsNone(manifest.trace_digest)
        self.assertIsNone(manifest.npus_count)
        self.assertIsNone(manifest.mesh_rows)
        self.assertEqual(manifest.manifest_source, "synthetic-prerun")
        self.assertEqual(
            manifest.slo_sampling,
            {"watermark_period_ns": 5_000_000, "link_bucket_ns": 5_000_000,
             "provisional": True},
        )
        # 现行协议字段逐一无损保留（含此前被拒的 decode_length）。
        self.assertEqual(
            [request.decode_length for request in manifest.requests], [32, 32]
        )
        self.assertEqual(
            manifest.requests[1].arrival.to_dict(),
            {"kind": "after_request", "parent_queue_index": 0,
             "interval_ns": 500_000_000},
        )
        self.assertEqual(
            manifest.requests[0].prefill_ranks, (0, 1, 2)
        )

    def test_request_metadata_with_decode_length_round_trips(self) -> None:
        entry = _real_shape_metrics_manifest_dict()["requests"][0]
        request = ms.RequestMetadata.from_dict(entry)
        self.assertEqual(request.decode_length, 32)
        self.assertEqual(
            ms.RequestMetadata.from_dict(_json_round_trip(request.to_dict())),
            request,
        )

    def test_empty_session_id_is_the_cpp_default(self) -> None:
        entry = _real_shape_metrics_manifest_dict()["requests"][0]
        entry["session_id"] = ""
        request = ms.RequestMetadata.from_dict(entry)
        self.assertEqual(request.session_id, "")
        del entry["session_id"]
        request = ms.RequestMetadata.from_dict(entry)
        self.assertEqual(request.session_id, "")

    def test_unknown_top_level_keys_stay_tolerated_like_cpp(self) -> None:
        data = _real_shape_metrics_manifest_dict()
        data["some_future_protocol_field"] = {"a": 1}
        ms.MetricsManifest.from_dict(data)

    def test_optional_fields_still_type_checked_when_present(self) -> None:
        data = _real_shape_metrics_manifest_dict()
        data["npus_count"] = 0
        with self.assertRaises(ms.MetricsSchemaError):
            ms.MetricsManifest.from_dict(data)
        data = _real_shape_metrics_manifest_dict()
        data["trace_digest"] = ""
        with self.assertRaises(ms.MetricsSchemaError):
            ms.MetricsManifest.from_dict(data)
        data = _real_shape_metrics_manifest_dict()
        data["run_mode"] = "benchmark"
        with self.assertRaises(ms.MetricsSchemaError):
            ms.MetricsManifest.from_dict(data)
        data = _real_shape_metrics_manifest_dict()
        data["slo_sampling"] = "fast"
        with self.assertRaises(ms.MetricsSchemaError):
            ms.MetricsManifest.from_dict(data)

    def test_request_collection_rejects_non_request_metadata(self) -> None:
        manifest = _service_manifest()
        with self.assertRaises(ms.MetricsSchemaError):
            ms.MetricsManifest(
                repo_variant=manifest.repo_variant,
                requests=manifest.requests + (object(),),  # type: ignore[list-item]
            )

    def test_memory_metrics_result_rejects_foreign_rank_values(self) -> None:
        with self.assertRaises(ms.MetricsSchemaError):
            ms.MemoryMetricsResult(chiplets_per_npu=4, ranks={0: "nope"})  # type: ignore[dict-item]
        with self.assertRaises(ms.MetricsSchemaError):
            ms.MemoryMetricsResult(chiplets_per_npu=4, ranks={"x": "nope"})  # type: ignore[dict-item]


class M21ComponentSignTests(unittest.TestCase):
    """M-21（2026-09-30）：负分量 fail-closed 单一口径（observer + 写侧）。"""

    def _observer(self, capacity: int = 10_000) -> ms.MemoryMetricsObserver:
        observer = ms.MemoryMetricsObserver()
        observer.initialize_rank(0, capacity)
        return observer

    def test_compensating_negative_component_rejected(self) -> None:
        # 聚合不变量单独存在时（weight=-100, resident=+150 → physical=50
        # 在容量内）旧口径放行并落 journal；新口径与
        # hbm_watermark/C++ 重放同判：负分量=账本损坏。
        observer = self._observer()
        with self.assertRaises(ms.MetricsSchemaError) as context:
            observer.record_delta(
                _delta(0, weight_delta_bytes=-100,
                       resident_kv_delta_bytes=150,
                       allocation_key="ghost")
            )
        self.assertIn("underflowed", str(context.exception))
        # 拒绝后账本零残留（validate-then-commit）。
        ledger = observer.finalize().ranks[0].ledger
        self.assertEqual(
            (ledger.weight_bytes, ledger.resident_kv_bytes), (0, 0)
        )

    def test_ledger_snapshot_reader_rejects_negative_components(self) -> None:
        with self.assertRaises(ms.MetricsSchemaError):
            ms.MemoryLedgerSnapshot(
                capacity_bytes=100,
                weight_bytes=-1,
                resident_kv_bytes=0,
                reserved_kv_bytes=0,
                physical_used_bytes=-1,
                committed_used_bytes=-1,
                uncommitted_free_bytes=101,
            )
        # chiplet 余量剥离设计下的 free<0（分量恒非负）仍合法。
        ms.MemoryLedgerSnapshot(
            capacity_bytes=100,
            weight_bytes=0,
            resident_kv_bytes=0,
            reserved_kv_bytes=0,
            physical_used_bytes=0,
            committed_used_bytes=0,
            uncommitted_free_bytes=-5,
        )

    def test_peak_snapshot_reader_rejects_negative_components(self) -> None:
        # R5-C5-S1：MemoryPeakSnapshot 的 at-peak 三组件（weight/resident/
        # reserved）与姊妹类 MemoryLedgerSnapshot 同受 M-21 显式非负门，
        # 构造侧负值 fail-closed（镜像上一测试结构）。
        with self.assertRaises(ms.MetricsSchemaError):
            ms.MemoryPeakSnapshot(
                scope="rank",
                rank=0,
                chiplet_index=None,
                peak_kind="physical_used",
                peak_value_bytes=50,
                capacity_bytes=100,
                weight_at_peak_bytes=-1,
                resident_kv_at_peak_bytes=0,
                reserved_kv_at_peak_bytes=0,
                free_at_peak_bytes=100,
                planner_time_ns=0,
                sequence_index=0,
                cause="test",
                request_id=None,
                session_id=None,
            )
        # sign-free 的 free_at_peak_bytes<0（chiplet 余量剥离设计）仍合法。
        ms.MemoryPeakSnapshot(
            scope="rank",
            rank=0,
            chiplet_index=None,
            peak_kind="physical_used",
            peak_value_bytes=50,
            capacity_bytes=100,
            weight_at_peak_bytes=0,
            resident_kv_at_peak_bytes=0,
            reserved_kv_at_peak_bytes=0,
            free_at_peak_bytes=-5,
            planner_time_ns=0,
            sequence_index=0,
            cause="test",
            request_id=None,
            session_id=None,
        )

    def _recorder(self, tmpdir: str) -> "tuple[Any, Any]":
        import metrics_integration as mi

        # 裸 observer（rank 由 recorder.initialize_rank 统一初始化）。
        observer = ms.MemoryMetricsObserver()
        recorder = mi.MemoryActionRecorder(
            observer, journal_path=Path(tmpdir) / "kv_delta_journal.jsonl"
        )
        recorder.initialize_rank(0, 10_000)
        return recorder, observer

    def test_journal_writer_rejects_negative_remote_component(self) -> None:
        with tempfile.TemporaryDirectory(prefix="m21_journal_") as tmpdir:
            recorder, _ = self._recorder(tmpdir)
            with self.assertRaises(RuntimeError) as context:
                recorder.record(
                    planner_time_ns=1,
                    anchor_kind="stage_boundary",
                    request_id="req-0",
                    session_id="sess-0",
                    rank=0,
                    allocation_key="remote:sess-0",
                    remote_delta_bytes=-5,
                    cause="writeoff",
                )
            self.assertIn("remote", str(context.exception))
            recorder.close_journal()
            # 被拒行不落盘（fail-closed 写侧门在 journal 行写出之前）。
            self.assertEqual(
                Path(recorder.journal_path).read_text(encoding="utf-8"), ""
            )

    def test_journal_writer_requires_integer_instance_index(self) -> None:
        for bad in (None, "3", 1.5, True):
            with tempfile.TemporaryDirectory(prefix="m21_journal_") as tmpdir:
                recorder, _ = self._recorder(tmpdir)
                with self.assertRaises(RuntimeError):
                    recorder.record(
                        planner_time_ns=1,
                        anchor_kind="stage_boundary",
                        request_id="req-0",
                        session_id="sess-0",
                        rank=0,
                        allocation_key="resident:sess-0",
                        resident_kv_delta_bytes=8,
                        cause="admission",
                        instance_index=bad,
                    )
                recorder.close_journal()

    def test_replay_transaction_count_excludes_preload_group(self) -> None:
        # 权重预载组（transaction_id=0）不计入 transaction_count——与
        # recorder.begin_transaction 计数器同口径（此前恒 +1）。
        with tempfile.TemporaryDirectory(prefix="m21_journal_") as tmpdir:
            recorder, _ = self._recorder(tmpdir)
            recorder.record(
                planner_time_ns=0,
                anchor_kind="weight_preload",
                request_id=None,
                session_id=None,
                rank=0,
                allocation_key="weight:model",
                weight_delta_bytes=100,
                cause="model_weight_preload",
                instance_index=0,
            )
            transaction_id = recorder.begin_transaction()
            recorder.record(
                planner_time_ns=1,
                anchor_kind="stage_boundary",
                request_id="req-0",
                session_id="sess-0",
                rank=0,
                allocation_key="resident:sess-0",
                resident_kv_delta_bytes=8,
                cause="admission",
                instance_index=0,
            )
            recorder.finish_transaction()
            replay = recorder.replay_journal()
            recorder.close_journal()
        self.assertEqual(replay["transaction_count"], 1)
        self.assertEqual(replay["transaction_count"],
                         recorder.transaction_count)
        self.assertEqual(replay["max_transaction_id"], transaction_id)
        self.assertEqual(replay["max_transaction_id"], 1)


class M37ProvenanceGateTests(unittest.TestCase):
    """M-37（2026-09-30）：manifest 读取失败 fail-closed + 空 digest 门
    显式 SKIP（不得静默 PASS）+ 无 provenance run 不落空标签组。"""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="m37_test_")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _run(self, run_id: str, init: dict) -> mp.Run:
        return mp.Run(
            run_id=run_id,
            log_path=Path(self._tmp.name) / f"{run_id}.log",
            init=init,
            schema_versions={mp.SUPPORTED_SCHEMA},
        )

    def test_declared_but_missing_metrics_manifest_fails_closed(self) -> None:
        run = self._run("r1", {"manifest_path":
                               str(Path(self._tmp.name) / "gone.json")})
        with self.assertRaises(mp.PostprocessError) as context:
            mp._load_manifest_sidecars(run)
        self.assertIn("cannot be read", str(context.exception))

    def test_declared_but_missing_service_manifest_fails_closed(self) -> None:
        metrics = Path(self._tmp.name) / "metrics_manifest.json"
        metrics.write_text("{}", encoding="utf-8")
        run = self._run("r1", {"manifest_path": str(metrics)})
        with self.assertRaises(mp.PostprocessError) as context:
            mp._load_manifest_sidecars(run)
        self.assertIn("service manifest", str(context.exception))

    def test_runs_without_manifest_are_separated_with_a_note(self) -> None:
        runs = [self._run("r1", {}), self._run("r2", {})]
        notes: list[str] = []
        groups, _ = mp.build_groups(runs, None, notes)
        self.assertEqual(len(groups), 1)
        self.assertEqual(len(groups[0].runs), 2)
        self.assertTrue(
            any("no manifest_path" in note for note in notes), notes
        )

    def test_digest_gate_skip_note_on_all_empty_digests(self) -> None:
        runs = [self._run("r1", {}), self._run("r2", {})]
        notes: list[str] = []
        config = {"groups": [{"group_id": "g",
                              "runs": [{"run_id": "r1"}, {"run_id": "r2"}]}]}
        mp.build_groups(runs, config, notes)
        self.assertTrue(
            any("digest gate SKIPPED" in note for note in notes), notes
        )

    def test_digest_gate_skip_note_on_auto_groups_too(self) -> None:
        runs = [self._run("r1", {}), self._run("r2", {})]
        notes: list[str] = []
        mp.build_groups(runs, None, notes)
        self.assertTrue(
            any("digest gate SKIPPED" in note for note in notes), notes
        )

    def test_digest_mismatch_still_fails(self) -> None:
        runs = [
            self._run("r1", {"trace_digest": "sha-a"}),
            self._run("r2", {"trace_digest": "sha-b"}),
        ]
        config = {"groups": [{"group_id": "g",
                              "runs": [{"run_id": "r1"}, {"run_id": "r2"}]}]}
        with self.assertRaises(mp.PostprocessError) as context:
            mp.build_groups(runs, config, [])
        self.assertIn("digest mismatch", str(context.exception))


class M38StagingTests(unittest.TestCase):
    """M-38（2026-09-30）：_stage_csv 唯一 tmp 名 + 不互踩。"""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="m38_test_")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _path(self) -> Path:
        return Path(self._tmp.name) / "raw_metrics.csv"

    def test_staging_names_are_unique_and_content_isolated(self) -> None:
        # 并发同目录互踩回归：两个 staged 文件并存、互不覆盖（固定
        # .tmp 名时第二个 stage 会静默重写第一个的字节）。
        staged_a = mp._stage_csv(self._path(), ["a"], [{"a": 1}])
        staged_b = mp._stage_csv(self._path(), ["a"], [{"a": 2}, {"a": 3}])
        try:
            self.assertNotEqual(staged_a, staged_b)
            self.assertTrue(staged_a.exists() and staged_b.exists())
            self.assertIn(".tmp.", staged_a.name)
            self.assertEqual(
                staged_a.read_text(encoding="utf-8").strip().splitlines()[1],
                "1",
            )
            self.assertEqual(
                staged_b.read_text(encoding="utf-8").strip().splitlines()[2],
                "3",
            )
            # 提交走 os.replace 原子落位。
            mp._commit_staged(staged_b, self._path())
            self.assertTrue(self._path().exists())
            self.assertFalse(staged_b.exists())
            self.assertTrue(staged_a.exists())
        finally:
            staged_a.unlink(missing_ok=True)

    def test_failed_stage_leaves_no_tmp_file(self) -> None:
        class _Unstringable:
            def __str__(self) -> str:
                raise RuntimeError("unstringable cell")

        with self.assertRaises(RuntimeError):
            mp._stage_csv(
                self._path(), ["a"],
                [{"a": 1}, {"a": _Unstringable()}],
            )
        leftovers = list(Path(self._tmp.name).glob("*.tmp.*"))
        self.assertEqual(leftovers, [])


class M20CapacityTimeavgTests(unittest.TestCase):
    """M-20（2026-09-30）：collector null（不可用）单元格显式 NA，不被
    当成测量值；重复 rank fail-closed（与 peaks 侧同 gate）。"""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="m20_test_")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _run(self, capacity_timeavg, planner_peaks=()) -> mp.Run:
        return mp.Run(
            run_id="r1",
            log_path=Path(self._tmp.name) / "r1.log",
            init={},
            capacity_timeavg=list(capacity_timeavg),
            planner_peaks=list(planner_peaks),
        )

    def test_null_utilization_cells_render_na(self) -> None:
        run = self._run([
            {"type": "capacity_timeavg", "rank": 0, "capacity_bytes": None,
             "resident_capacity_timeavg_util": None,
             "committed_capacity_timeavg_util": None,
             "actions_total": 0, "actions_replayed": 0,
             "actions_unresolved": 0},
        ])
        rows = mp._memory_ledger_rows(run, {"model": "", "dataset": "",
                                            "hardware": "", "kv_policy": ""})
        self.assertEqual(rows[0]["capacity_bytes"], mp.NA)
        self.assertEqual(rows[0]["resident_capacity_timeavg_util"], mp.NA)
        self.assertEqual(rows[0]["committed_capacity_timeavg_util"], mp.NA)

    def test_zero_utilization_from_real_replay_stays_numeric(self) -> None:
        # 真实重放的合法 0.0 利用率不得被吞成 NA。
        run = self._run([
            {"type": "capacity_timeavg", "rank": 0, "capacity_bytes": 100,
             "resident_capacity_timeavg_util": 0.0,
             "committed_capacity_timeavg_util": 0.0,
             "actions_total": 1, "actions_replayed": 1,
             "actions_unresolved": 0},
        ])
        rows = mp._memory_ledger_rows(run, {"model": "", "dataset": "",
                                            "hardware": "", "kv_policy": ""})
        self.assertEqual(rows[0]["resident_capacity_timeavg_util"], 0.0)
        self.assertEqual(rows[0]["capacity_bytes"], 100)

    def test_duplicate_timeavg_rank_fails_closed(self) -> None:
        record = {"type": "capacity_timeavg", "rank": 0}
        run = self._run([record, dict(record)])
        with self.assertRaises(mp.PostprocessError):
            mp._memory_ledger_rows(run, {"model": "", "dataset": "",
                                         "hardware": "", "kv_policy": ""})

    def test_non_integer_rank_fails_closed(self) -> None:
        run = self._run([{"type": "capacity_timeavg", "rank": "0"}])
        with self.assertRaises(mp.PostprocessError):
            mp._memory_ledger_rows(run, {"model": "", "dataset": "",
                                         "hardware": "", "kv_policy": ""})


class RequestRowInvariantTests(unittest.TestCase):
    """Minor（2026-09-30）：冻结列前的阶段非负/边界单调门 +
    _int_or_na 非整数 fail-closed + 服务路径数值守卫。"""

    def test_negative_stage_duration_rejected(self) -> None:
        for column in ("queue_ns", "prefill_ns", "prefill_decode_gap_ns",
                       "decode_ns"):
            with self.assertRaises(mp.PostprocessError):
                mp._check_request_row_invariants(
                    "r", {column: -1, "arrival_ns": 5, "completion_ns": 10},
                    0)

    def test_boundary_chain_monotonicity_enforced(self) -> None:
        for record in (
            {"arrival_ns": 10, "prefill_start_ns": 5},
            {"prefill_start_ns": 10, "prefill_end_ns": 5},
            {"prefill_end_ns": 10, "decode_start_ns": 5},
            {"decode_start_ns": 10, "completion_ns": 5},
        ):
            with self.assertRaises(mp.PostprocessError):
                mp._check_request_row_invariants("r", record, 0)
        # C++ 完成链同款合法形态放行。
        mp._check_request_row_invariants(
            "r",
            {"arrival_ns": 0, "prefill_start_ns": 1, "prefill_end_ns": 2,
             "decode_start_ns": 3, "completion_ns": 4, "queue_ns": 1,
             "prefill_ns": 1, "prefill_decode_gap_ns": 1, "decode_ns": 1,
             "e2e_ns": 4},
            0,
        )

    def test_negative_stage_caught_without_completion(self) -> None:
        # 截断行（无 arrival/completion）同样不得把负阶段写入冻结列。
        with self.assertRaises(mp.PostprocessError):
            mp._check_request_row_invariants("r", {"queue_ns": -7}, 0)

    def test_int_or_na_rejects_non_integer_values(self) -> None:
        self.assertEqual(
            mp._int_or_na(None, run_id="r", queue_index=0,
                          column="queue_ns"),
            mp.NA,
        )
        self.assertEqual(
            mp._int_or_na(True, run_id="r", queue_index=0,
                          column="queue_ns"),
            mp.NA,
        )
        self.assertEqual(
            mp._int_or_na(5, run_id="r", queue_index=0, column="queue_ns"), 5
        )
        for bad in (1.5, "3"):
            with self.assertRaises(mp.PostprocessError):
                mp._int_or_na(bad, run_id="r", queue_index=0,
                              column="queue_ns")

    def test_service_normalization_rejects_non_numeric_summary(self) -> None:
        run = mp.Run(
            run_id="r1",
            log_path=Path("/tmp/does-not-matter.log"),
            init={"run_mode": "service"},
            summaries=[{"mean_e2e_ns": "fast", "drain_tput_rps": 10.0}],
        )
        group = mp.Group(group_id="g", runs=[run], baseline_run_id=None,
                         labels_by_run={})
        with self.assertRaises(mp.PostprocessError) as context:
            mp.build_normalized_rows(
                [group], ms.NORMALIZATION_GROUP_MAX, None)
        self.assertIn("not numeric", str(context.exception))


class GroupConfigGuardTests(unittest.TestCase):
    """Minor（2026-09-30）：group config 不可哈希/非对象成员结构化失败。"""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="gcfg_test_")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _run(self, run_id: str) -> mp.Run:
        return mp.Run(
            run_id=run_id,
            log_path=Path(self._tmp.name) / f"{run_id}.log",
            init={},
        )

    def test_unhashable_group_id_rejected(self) -> None:
        config = {"groups": [{"group_id": ["x"], "runs": []}]}
        with self.assertRaises(mp.PostprocessError):
            mp.build_groups([self._run("r1")], config, [])

    def test_non_list_runs_field_rejected(self) -> None:
        config = {"groups": [{"group_id": "g", "runs": {"r1": True}}]}
        with self.assertRaises(mp.PostprocessError):
            mp.build_groups([self._run("r1")], config, [])

    def test_non_object_member_entry_rejected(self) -> None:
        config = {"groups": [{"group_id": "g", "runs": ["r1"]}]}
        with self.assertRaises(mp.PostprocessError):
            mp.build_groups([self._run("r1")], config, [])

    def test_unhashable_member_run_id_rejected(self) -> None:
        config = {"groups": [{"group_id": "g",
                              "runs": [{"run_id": ["r1"]}]}]}
        with self.assertRaises(mp.PostprocessError):
            mp.build_groups([self._run("r1")], config, [])


class Derive20AtomicPublishTests(unittest.TestCase):
    """M-16 Python 半边（2026-09-30）：derive 物化器原子发布 + 网格门
    fail-closed（端到端 CLI 驱动，产物全落 /tmp）。"""

    SOURCE_HEADER = ("session_id,arrival_time,prefix_len,prefill_length,"
                     "decode_length,human_time,tool_time\n")

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="derive20_test_")
        self.script = (
            Path(__file__).resolve().parents[1] / "workload" /
            "llama2_7b_inference" / "traces" / "derive_20_first_30_seconds.py"
        )

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _run_script(self, args: list[str]) -> "subprocess.CompletedProcess":
        import subprocess

        return subprocess.run(
            [sys.executable, str(self.script), *args],
            capture_output=True, text=True, cwd=self._tmp.name,
        )

    def test_atomic_publish_and_no_debris(self) -> None:
        source = Path(self._tmp.name) / "src.csv"
        source.write_text(
            self.SOURCE_HEADER
            + "s1,1000000000,0,128,32,500000000,\n"
            + "s1,,,128,32,,\n",
            encoding="utf-8",
        )
        result = self._run_script(["src.csv", "q.csv", "sc.csv"])
        self.assertEqual(result.returncode, 0, result.stderr)
        base = Path(self._tmp.name)
        for name in ("q.csv", "sc.csv", "q.csv.provenance.json"):
            self.assertTrue((base / name).exists(), name)
        self.assertEqual(list(base.glob("*.tmp.*")), [])
        # provenance 与队列字节一致（C++ REQUIRED 门的 Python 侧前提）。
        provenance = json.loads(
            (base / "q.csv.provenance.json").read_text(encoding="utf-8"))
        self.assertEqual(provenance["csv_bytes"],
                         (base / "q.csv").stat().st_size)
        self.assertEqual(provenance["data_rows"], 2)

    def test_off_grid_timing_fails_closed_without_products(self) -> None:
        source = Path(self._tmp.name) / "src.csv"
        source.write_text(
            self.SOURCE_HEADER
            + "s1,1000000500,0,128,32,500000000,\n"
            + "s1,,,128,32,,\n",
            encoding="utf-8",
        )
        result = self._run_script(["src.csv", "q.csv", "sc.csv"])
        self.assertEqual(result.returncode, 2)
        self.assertIn("queue NOT materialized", result.stderr)
        base = Path(self._tmp.name)
        for name in ("q.csv", "sc.csv", "q.csv.provenance.json"):
            self.assertFalse((base / name).exists(), name)
        self.assertEqual(list(base.glob("*.tmp.*")), [])


class PlanMaterializerQueueGuardTests(unittest.TestCase):
    """Minor（2026-09-30）：plan_materializer 物化边界的队列良构性
    fail-closed（同 session 重复 turn-0 / 重复 (session, turn)）。"""

    @staticmethod
    def _spec(request_id: str, session_id: str, turn_index: int,
              prefill_length: int = 10, decode_length: int = 4,
              session_arrival: "int | None" = None,
              interval: "int | None" = None):
        from types import SimpleNamespace

        return SimpleNamespace(
            request_id=request_id,
            session_id=session_id,
            turn_index=turn_index,
            prefill_length=prefill_length,
            decode_length=decode_length,
            session_arrival_time_ns=session_arrival,
            inter_request_interval_ns=interval,
        )

    def _config(self, queue):
        from types import SimpleNamespace

        return SimpleNamespace(
            request_queue=queue,
            inference_groups=[SimpleNamespace(ranks=(0, 1))],
        )

    def test_duplicate_turn0_fails_closed(self) -> None:
        import plan_materializer as pm

        config = self._config([
            self._spec("r0", "s1", 0, session_arrival=0),
            self._spec("r1", "s1", 0, session_arrival=5),
        ])
        with self.assertRaises(RuntimeError) as context:
            pm._derive_manifest_requests(config, {})
        self.assertIn("second turn-0", str(context.exception))
        with self.assertRaises(RuntimeError):
            pm._derive_metrics_requests(config, {})

    def test_duplicate_session_turn_pair_fails_closed(self) -> None:
        import plan_materializer as pm

        config = self._config([
            self._spec("r0", "s1", 0, session_arrival=0),
            self._spec("r1", "s1", 1, interval=10),
            self._spec("r2", "s1", 1, interval=10),
        ])
        with self.assertRaises(RuntimeError) as context:
            pm._derive_metrics_requests(config, {})
        self.assertIn("must be unique", str(context.exception))
        with self.assertRaises(RuntimeError):
            pm._derive_manifest_requests(config, {})

    def test_wellformed_two_turn_session_still_materializes(self) -> None:
        import plan_materializer as pm

        config = self._config([
            self._spec("r0", "s1", 0, session_arrival=0),
            self._spec("r1", "s1", 1, interval=10),
        ])
        requests = pm._derive_manifest_requests(config, {})
        self.assertEqual(len(requests), 2)
        records = pm._derive_metrics_requests(config, {})
        self.assertEqual(records[1]["arrival"]["kind"], "after_request")
        self.assertEqual(records[1]["arrival"]["parent_queue_index"], 0)


if __name__ == "__main__":
    unittest.main()
