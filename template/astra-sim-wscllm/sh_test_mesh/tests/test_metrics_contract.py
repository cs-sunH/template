from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest


_WORKLOAD_DIR = Path(__file__).resolve().parents[1] / "workload" / "llama2_7b_inference"
_SLO_TOOLS_DIR = Path(__file__).resolve().parents[1] / "slo_tools"
for _p in (_WORKLOAD_DIR, _SLO_TOOLS_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import metrics_schema as ms
import metrics_postprocess as mp
import slo_common


_CONFIG_PATH = _WORKLOAD_DIR / "metrics_config.json"


def _request(
    queue_index: int,
    *,
    arrival: "ms.Arrival | None" = None,
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
        prefill_ranks=(0, 1),
        decode_instance=1,
        decode_ranks=(2, 3),
    )


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
        arrival = ms.Arrival.after_request(
            parent_queue_index=0, interval_ns=1_000_000
        )
        # The parent's ACTUAL completion tick is required; a planner
        # prediction must never be substituted (schema doc sec.3.1).
        self.assertEqual(arrival.resolve(parent_completion_ns=703), 703 + 1_000_000)
        # A child whose parent has not completed yet cannot resolve.
        with self.assertRaises(ms.MetricsSchemaError):
            arrival.resolve()

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

    def test_chiplet_allocation_add_remove_is_symmetric(self) -> None:
        # Equal-striping chiplet projection of one allocation, and the
        # add/remove symmetry of its allocation-key bookkeeping (asserted
        # against the observer's own ledger state; the finalized result
        # surface no longer exists -- the observer is a shadow ledger).
        observer = self._observer()
        observer.record_delta(_delta(0, resident_kv_delta_bytes=103))
        self.assertEqual(
            tuple(c.resident_kv_bytes for c in observer._chiplets[0]),
            (26, 26, 26, 25),
        )
        observer.record_delta(_delta(1, resident_kv_delta_bytes=-103))
        for chiplet in observer._chiplets[0]:
            self.assertEqual(chiplet.resident_kv_bytes, 0)
            self.assertEqual(chiplet.physical, 0)
        self.assertEqual(observer._allocations[0], {})

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
    def test_schema_version_check(self) -> None:
        with self.assertRaises(ms.MetricsSchemaError):
            ms.check_schema_version(0)
        with self.assertRaises(ms.MetricsSchemaError):
            ms.check_schema_version("1")
        self.assertEqual(ms.check_schema_version(ms.SCHEMA_VERSION), 1)

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
    fields that are not yet filled.

    A4-16 (2026-09-26, face 仓同源移植): the terminal_status /
    first_token_source / NA assertions are no longer self-proving constant
    comparisons -- they are checked against the values the implementation
    actually writes into request_metrics rows (mp._request_metric_rows over
    a synthetic run covering completed/failed, exact/proxy/NA first-token
    sources).  This couples the enum assertions to this repository's own
    metrics_postprocess implementation, so the former "byte-identical
    across the five repos" property holds only for the schema-symbol
    assertions (ms.*) and the frozen column list; the enum assertions are
    per-repo by design.
    """

    TERMINAL_STATUS_VALUES = ("completed", "failed")
    FIRST_TOKEN_SOURCE_VALUES = ("exact", "train_interpolated", "NA")
    NA_SENTINEL = "NA"

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="wscllm_contract_rows_")
        run_dir = Path(self._tmp.name) / "run"
        (run_dir / "results").mkdir(parents=True)
        # Run-dir sidecars: metrics_manifest.json carries no requests here,
        # so the join is served by the sibling service manifest.json.
        (run_dir / "metrics_manifest.json").write_text("{}", encoding="utf-8")
        (run_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "requests": [
                        {
                            "queue_index": 0,
                            "request_id": "req-exact",
                            "session_id": "s0",
                            "turn_index": 0,
                            "request_type": "human",
                            "prefill_length": 128,
                            "decode_length": 8,
                        },
                        {
                            "queue_index": 1,
                            "request_id": "req-proxy",
                            "session_id": "s1",
                            "turn_index": 0,
                            "request_type": "tool",
                            "prefill_length": 128,
                            "decode_length": 8,
                            "prefill_context_tokens": 128,
                            "final_context_tokens": 136,
                        },
                        {
                            "queue_index": 2,
                            "request_id": "req-na",
                            "session_id": "s2",
                            "turn_index": 0,
                            "request_type": "human",
                            "prefill_length": 128,
                            "decode_length": 8,
                        },
                    ]
                }
            ),
            encoding="utf-8",
        )

        def _request_record(queue_index, request_id, session_id, **overrides):
            decode_start = 2_000_000 + queue_index
            record = {
                "type": "request",
                "schema": 1,
                "queue_index": queue_index,
                "request_id": request_id,
                "session_id": session_id,
                "turn_index": 0,
                "completed": True,
                "arrival_ns": 100 + queue_index,
                "prefill_start_ns": 200 + queue_index,
                "prefill_end_ns": 300 + queue_index,
                "decode_start_ns": decode_start,
                "completion_ns": decode_start + 40_000_000,
                "e2e_ns": decode_start + 40_000_000 - (100 + queue_index),
                "queue_ns": 100,
                "prefill_ns": 100,
                "prefill_decode_gap_ns": 0,
                "decode_ns": decode_start + 40_000_000 - (100 + queue_index)
                - 100 - 100,
            }
            record.update(overrides)
            return record

        records = [
            {
                "type": "init",
                "schema": 1,
                "run_id": "contract-enum",
                "run_mode": "service",
                "detail_level": "full",
                "manifest_path": str(run_dir / "metrics_manifest.json"),
            },
            # exact first token (event code 8 value).
            _request_record(
                0, "req-exact", "s0", first_token_ns=2_100_000
            ),
            # no exact value; the archived train ledger below yields the
            # train-interpolated proxy.
            _request_record(1, "req-proxy", "s1"),
            # incomplete request: terminal_status must be "failed".
            _request_record(
                2,
                "req-na",
                "s2",
                completed=False,
                incomplete_reason="missing stage boundary events",
                arrival_ns=None,
                prefill_start_ns=None,
                prefill_end_ns=None,
                decode_start_ns=None,
                completion_ns=None,
                e2e_ns=None,
                queue_ns=None,
                prefill_ns=None,
                prefill_decode_gap_ns=None,
                decode_ns=None,
            ),
        ]
        log_path = run_dir / "cpp.log"
        with log_path.open("w", encoding="utf-8") as sink:
            for record in records:
                sink.write("[METRIC] " + json.dumps(record) + "\n")
        # Train ledger: one train carrying req-proxy (iterations=8, tick at
        # decode_start) plus the next boundary row on the same instance so
        # the train-end tick exists.
        with (run_dir / "results" / "train_ledger.jsonl").open(
            "w", encoding="utf-8"
        ) as sink:
            sink.write(
                json.dumps(
                    {
                        "train_id": "batch_train_i0_1",
                        "instance_index": 0,
                        "tick": 2_000_000,
                        "iterations": 8,
                        "joiners": ["req-proxy"],
                    }
                )
                + "\n"
            )
            sink.write(
                json.dumps(
                    {
                        "train_id": "batch_train_i0_2",
                        "instance_index": 0,
                        "tick": 9_000_000,
                        "iterations": 8,
                        "joiners": [],
                    }
                )
                + "\n"
            )
        runs = mp._parse_logs([log_path])
        self.assertEqual(len(runs), 1)
        metrics_manifest, service_manifest = mp._load_manifest_sidecars(runs[0])
        entries, _counts = mp._manifest_request_entries(
            runs[0], metrics_manifest, service_manifest
        )
        self.rows = {
            row["request_id"]: row
            for row in mp._request_metric_rows(runs[0], entries)
        }

    def tearDown(self) -> None:
        self._tmp.cleanup()

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
        # (2026-10-01) Cross-repo-of-truth tripwire: the producer
        # (metrics_postprocess.REQUEST_METRICS_COLUMNS) and the consumer
        # (slo_common.REQUEST_METRICS_COLUMNS) each pin their own literal
        # copy; without this equality assertion a one-sided edit could drift
        # both self-tests green until a real run was rejected by slo_tools.
        self.assertEqual(tuple(mp.REQUEST_METRICS_COLUMNS),
                         slo_common.REQUEST_METRICS_COLUMNS)
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
        # Closed enum written into the terminal_status column, asserted
        # against the implementation's actual output: a completed record
        # maps to "completed", an incomplete one to "failed", and nothing
        # else may appear (A4-16: real output, not constant self-comparison).
        self.assertIn("terminal_status", mp.REQUEST_METRICS_COLUMNS)
        written = {row["terminal_status"] for row in self.rows.values()}
        self.assertEqual(
            written, set(self.TERMINAL_STATUS_VALUES), sorted(written)
        )
        self.assertEqual(self.rows["req-exact"]["terminal_status"], "completed")
        self.assertEqual(self.rows["req-na"]["terminal_status"], "failed")
        self.assertIn(
            "incomplete_reason=", self.rows["req-na"]["instructions"]
        )
        # A4-9 对拍: the closed SLO-side word list (slo_common
        # TERMINAL_STATUSES, fail-closed CSV validation) must freeze the
        # exact set the producer writes here.
        self.assertEqual(
            set(slo_common.TERMINAL_STATUSES),
            set(self.TERMINAL_STATUS_VALUES),
            sorted(slo_common.TERMINAL_STATUSES),
        )

    def test_first_token_source_enum(self) -> None:
        # Closed enum per the main spec, asserted against the values the
        # implementation actually writes: exact event wins, missing exact +
        # available train ledger -> train_interpolated proxy, neither -> NA
        # (A4-16: real output, not constant self-comparison).
        self.assertIn("first_token_source", mp.REQUEST_METRICS_COLUMNS)
        self.assertEqual(
            self.rows["req-exact"]["first_token_source"], "exact"
        )
        self.assertEqual(
            self.rows["req-proxy"]["first_token_source"],
            mp.FIRST_TOKEN_SOURCE_TRAIN_INTERPOLATED,
        )
        self.assertEqual(self.rows["req-na"]["first_token_source"], mp.NA)
        written = {row["first_token_source"] for row in self.rows.values()}
        self.assertEqual(
            written, set(self.FIRST_TOKEN_SOURCE_VALUES), sorted(written)
        )

    def test_na_semantics(self) -> None:
        # NA = "field not yet produced by the owning work package"; it is
        # the actual cell value the implementation writes for req-na's
        # first_token_ns/first_token_source, it is never a terminal_status
        # value, and it never parses as a number.
        self.assertEqual(mp.NA, self.NA_SENTINEL)
        self.assertEqual(self.rows["req-na"]["first_token_ns"], mp.NA)
        self.assertIn(self.NA_SENTINEL, self.FIRST_TOKEN_SOURCE_VALUES)
        self.assertTrue(
            all(
                value != self.NA_SENTINEL
                for value in self.TERMINAL_STATUS_VALUES
            )
        )
        with self.assertRaises(ValueError):
            int(self.NA_SENTINEL)
        with self.assertRaises(ValueError):
            float(self.NA_SENTINEL)


class NaNFailClosedTests(unittest.TestCase):
    """P1 fix (2026-09-28): NaN must never slip the normalization gates.

    ``nan <= 0`` is False and max() propagates a NaN in the first slot, so
    the former zero-denominator gates let a NaN summary poison whole
    normalization groups (healthy runs included) while main() still exited 0.
    """

    def test_schema_group_values_must_be_finite(self) -> None:
        for values in ([float("nan"), 5.0], [5.0, float("nan")],
                       [float("inf")], [1.0, float("-inf")]):
            with self.assertRaises(ms.MetricsSchemaError):
                ms.select_normalization_denominator(
                    ms.NORMALIZATION_GROUP_MAX, group_values=values)
            with self.assertRaises(ms.MetricsSchemaError):
                ms.build_normalization_record(
                    "mean_e2e_ns", 1.0, ms.NORMALIZATION_GROUP_MAX, "g",
                    group_values=values)

    def test_schema_baseline_value_must_be_finite(self) -> None:
        for bad in (float("nan"), float("inf"), float("-inf")):
            with self.assertRaises(ms.MetricsSchemaError):
                ms.select_normalization_denominator(
                    ms.NORMALIZATION_RATIO_TO_BASELINE,
                    baseline_run_id="base", baseline_value=bad)

    def test_record_numeric_fields_must_be_finite(self) -> None:
        for field in ("raw_value", "denominator_value", "normalized_value"):
            kwargs = {
                "metric_name": "mean_e2e_ns",
                "raw_value": 2.0,
                "normalization_method": ms.NORMALIZATION_RATIO_TO_BASELINE,
                "normalization_group_id": "g",
                "baseline_run_id": "base",
                "denominator_value": 4.0,
                "normalized_value": 0.5,
            }
            kwargs[field] = float("nan")
            with self.assertRaises(ms.MetricsSchemaError):
                ms.NormalizationRecord(**kwargs)

    def test_zero_and_negative_denominators_still_fail(self) -> None:
        # 修复不得放松原零/负分母门。
        for bad in (0, -1.0):
            with self.assertRaises(ms.MetricsSchemaError):
                ms.select_normalization_denominator(
                    ms.NORMALIZATION_RATIO_TO_BASELINE,
                    baseline_run_id="base", baseline_value=bad)

    def test_nan_summary_run_main_exits_nonzero(self) -> None:
        # 端到端失败路径钉死：NaN 汇总值所在的 run 让 main() 以非零退出
        # （修复前：全组 normalized_time 落盘 nan、退出码 0）。
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "cpp.log"
            lines = []
            for run_id, mean_e2e in (("run_nan", "NaN"), ("run_ok", "1000")):
                lines.append(json.dumps({
                    "schema": 1, "type": "init", "run_id": run_id,
                    "run_mode": "service"}))
                lines.append(json.dumps({
                    "schema": 1, "type": "summary", "run_mode": "service",
                    "mean_e2e_ns": mean_e2e, "drain_tput_rps": "10"}))
            log.write_text("\n".join(lines) + "\n", encoding="utf-8")
            code = mp.main([
                str(log),
                "--out-raw", str(Path(tmp) / "raw.csv"),
                "--out-normalized", str(Path(tmp) / "norm.csv"),
                "--out-requests", str(Path(tmp) / "req.csv"),
            ])
            self.assertEqual(code, 1, "NaN summary must fail the run "
                                      "(exit code must be non-zero)")
            # 失败路径不得落任何含 nan 的产物。
            for name in ("raw.csv", "norm.csv"):
                path = Path(tmp) / name
                if path.exists():
                    self.assertNotIn(
                        "nan", path.read_text(encoding="utf-8").lower())


class MicrobenchPointAlignmentTests(unittest.TestCase):
    """P2-22（2026-09-28）：microbench 归一化分母按 benchmark_point_id
    同点对齐——group_max 不再跨点取 max、ratio_to_baseline 不再取
    baseline 首条 iteration（跨点倍率无意义，实测 50"倍"废值）；
    auto 分组键加入 point 身份（不同点集的 run 不同组）。记录无
    benchmark_point_id 时维持单桶现行为（本仓无 iteration 生产者，
    契约性潜伏）。"""

    @staticmethod
    def _micro_run(run_id: str, iterations: list[dict]) -> mp.Run:
        return mp.Run(
            run_id=run_id,
            log_path=Path("<synthetic>"),
            init={"repo_variant": "astra-sim-wscllm",
                  "run_mode": "microbenchmark"},
            iterations=iterations,
        )

    POINTED_A = [
        {"benchmark_point_id": "small", "iteration_time_ns": 100},
        {"benchmark_point_id": "large", "iteration_time_ns": 10000},
    ]
    POINTED_B = [
        {"benchmark_point_id": "small", "iteration_time_ns": 110},
        {"benchmark_point_id": "large", "iteration_time_ns": 10500},
    ]

    def test_auto_key_splits_disjoint_point_sets(self) -> None:
        run_a = self._micro_run("run_a", self.POINTED_A)
        run_b = self._micro_run("run_b", [
            {"benchmark_point_id": "huge", "iteration_time_ns": 999}])
        groups, _ = mp.build_groups([run_a, run_b], None)
        self.assertEqual(len(groups), 2,
                         "不同 point 集的 run 不得共享归一化组")

    def test_auto_key_keeps_same_point_set_together(self) -> None:
        run_a = self._micro_run("run_a", self.POINTED_A)
        run_b = self._micro_run("run_b", self.POINTED_B)
        groups, _ = mp.build_groups([run_a, run_b], None)
        self.assertEqual(len(groups), 1)

    def test_group_max_normalized_per_point(self) -> None:
        run_a = self._micro_run("run_a", self.POINTED_A)
        run_b = self._micro_run("run_b", self.POINTED_B)
        groups, _ = mp.build_groups([run_a, run_b], None)
        rows = mp.build_normalized_rows(
            groups, mp.NORMALIZATION_GROUP_MAX, None)
        by_point: dict[str, list[dict]] = {}
        for row in (r for r in rows
                    if r["source"] == mp.SOURCE_MICROBENCHMARK):
            by_point.setdefault(row["benchmark_point_id"], []).append(row)
        self.assertEqual(sorted(by_point), ["large", "small"])
        # 同点分母：small 组内 max=110，large 组内 max=10500（跨点 max
        # 10500 不再污染 small 点的归一化——旧口径 small=100/10500≈0.0095）。
        self.assertEqual(
            {r["denominator_iteration_time"] for r in by_point["small"]},
            {110})
        self.assertEqual(
            {r["denominator_iteration_time"] for r in by_point["large"]},
            {10500})
        small_norm = sorted(
            r["normalized_iteration_time"] for r in by_point["small"])
        self.assertAlmostEqual(small_norm[0], 100 / 110)
        self.assertAlmostEqual(small_norm[1], 110 / 110)

    def test_ratio_to_baseline_uses_same_point_baseline(self) -> None:
        run_a = self._micro_run("run_a", self.POINTED_A)
        run_b = self._micro_run("run_b", self.POINTED_B)
        group = mp.Group(group_id="g", runs=[run_a, run_b],
                         baseline_run_id="run_a", labels_by_run={})
        rows = mp.build_normalized_rows(
            [group], mp.NORMALIZATION_RATIO_TO_BASELINE, None)
        by_point: dict[str, list[dict]] = {}
        for row in (r for r in rows
                    if r["source"] == mp.SOURCE_MICROBENCHMARK
                    and r["run_id"] == "run_b"):
            by_point.setdefault(row["benchmark_point_id"], []).append(row)
        # baseline 取同点 baseline iteration（run_a 的 small=100 /
        # large=10000），不再无条件取"首条"（首条恰为 small=100 时，
        # large 的跨点倍率 10500/100=105 即旧口径废值形态）。
        self.assertAlmostEqual(
            by_point["small"][0]["normalized_iteration_time"], 110 / 100)
        self.assertAlmostEqual(
            by_point["large"][0]["normalized_iteration_time"],
            10500 / 10000)
        self.assertEqual(
            by_point["small"][0]["denominator_iteration_time"], 100)
        self.assertEqual(
            by_point["large"][0]["denominator_iteration_time"], 10000)

    def test_no_point_id_keeps_legacy_single_bucket(self) -> None:
        run_a = self._micro_run("run_a", [
            {"iteration_time_ns": 100}, {"iteration_time_ns": 10000}])
        run_b = self._micro_run("run_b", [
            {"iteration_time_ns": 110}, {"iteration_time_ns": 10500}])
        groups, _ = mp.build_groups([run_a, run_b], None)
        self.assertEqual(len(groups), 1, "无 point id 维持单组现行为")
        rows = mp.build_normalized_rows(
            groups, mp.NORMALIZATION_GROUP_MAX, None)
        micro_rows = [r for r in rows
                      if r["source"] == mp.SOURCE_MICROBENCHMARK]
        self.assertEqual(len(micro_rows), 4)
        # 单桶：全部行共享全组 max=10500（现行为，注释钉死）。
        self.assertEqual(
            {r["denominator_iteration_time"] for r in micro_rows}, {10500})


class D2SchemaContractTests(unittest.TestCase):
    """D2（2026-09-28）：chiplet 投影移除路径复核 + 死代码删除钉子。"""

    def _observer(self) -> ms.MemoryMetricsObserver:
        observer = ms.MemoryMetricsObserver()
        observer.initialize_rank(0, 1_000_000)
        return observer

    def test_chiplet_multisegment_grow_then_whole_release_walks_back(self):
        # 审查报告 D2 条目（原 metrics_schema.py:722-740）声称"移除按总量
        # 重新均分而非按记录分布 → 多段增长后整段释放产生静默负占用 +
        # 幽灵分配记录"。现实现按 delta 逐片 stripe 记账（stripe 不可
        # 加：余数堆低片，stripe(103)+stripe(57) != stripe(160)，见
        # metrics_schema._apply_chiplet_projection 的登记），移除走
        # _removal_distribution 按记录分布逐片精确回退：多段增长 +
        # 整段释放后，4 个 chiplet 的三个分量全零、分配记录回收，且
        # 全程无负占用、无异常——旧"总量重均分"法在此数据上会得
        # (1,0,0,-1) 的负占用；该结论由本测试钉死。
        observer = self._observer()
        observer.record_delta(_delta(
            0, resident_kv_delta_bytes=103, allocation_key="k"))
        observer.record_delta(_delta(
            1, resident_kv_delta_bytes=57, allocation_key="k"))
        # 中途另一把钥匙交错（多段 + 多键混合形态）。
        observer.record_delta(_delta(
            2, weight_delta_bytes=40, allocation_key="other"))
        observer.record_delta(_delta(
            3, resident_kv_delta_bytes=-160, allocation_key="k"))
        for chiplet in observer._chiplets[0]:
            self.assertEqual(chiplet.resident_kv_bytes, 0)
            self.assertGreaterEqual(chiplet.weight_bytes, 0)
        self.assertNotIn("k", observer._allocations[0])
        # "other" 键完好（幽灵记录不残留、健康记录不受扰）。
        self.assertEqual(len(observer._allocations[0]), 1)
        self.assertEqual(
            sum(observer._allocations[0]["other"][0]), 40)

    def test_d2_dead_symbols_removed(self):
        # 死类/死常量删除（grep 证全仓零引用后删）：MemoryProjection 与
        # RUN_MODES 族 + chiplet 投影声明三常量（DEFAULT_CHIPLETS_PER_NPU
        # 仍被 MemoryMetricsObserver 使用，保留）。
        for name in ("MemoryProjection", "RUN_MODES", "RUN_MODE_SERVICE",
                     "RUN_MODE_MICROBENCHMARK", "NATIVE_MEMORY_SCOPE",
                     "CHIPLET_PROJECTION_SCOPE", "CHIPLET_PROJECTION_METHOD"):
            self.assertFalse(
                hasattr(ms, name),
                f"metrics_schema.{name} 应已随 D2 死代码批删除")
        self.assertTrue(hasattr(ms, "DEFAULT_CHIPLETS_PER_NPU"))


class D2PostprocessGateTests(unittest.TestCase):
    """D2（2026-09-28）：metrics_postprocess 门面修复——manifest 计数
    单侧容忍、组内重复 run_id、裸 float/int 转换逃逸。"""

    @staticmethod
    def _request_record(queue_index=0, request_id="req-0", **overrides):
        record = {
            "type": "request", "schema": 1,
            "queue_index": queue_index, "request_id": request_id,
            "session_id": "s0", "turn_index": 0, "completed": True,
            "arrival_ns": 100, "prefill_start_ns": 200,
            "prefill_end_ns": 300, "decode_start_ns": 400,
            "first_token_ns": 500, "completion_ns": 900,
            "queue_ns": 100, "prefill_ns": 100, "prefill_decode_gap_ns": 0,
            "decode_ns": 600, "e2e_ns": 800,
        }
        record.update(overrides)
        return record

    def _write_run(self, run_dir: Path, *,
                   metrics_manifest: dict, service_manifest: dict,
                   record: dict) -> Path:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "metrics_manifest.json").write_text(
            json.dumps(metrics_manifest), encoding="utf-8")
        (run_dir / "manifest.json").write_text(
            json.dumps(service_manifest), encoding="utf-8")
        log = run_dir / "cpp.log"
        init = {"type": "init", "schema": 1, "run_id": "d2run",
                "run_mode": "service", "detail_level": "full",
                "manifest_path": str(run_dir / "metrics_manifest.json")}
        log.write_text(
            "[METRIC] " + json.dumps(init) + "\n"
            + "[METRIC] " + json.dumps(record) + "\n",
            encoding="utf-8")
        return log

    _ENTRY = {"queue_index": 0, "request_id": "req-0", "session_id": "s0",
              "turn_index": 0, "request_type": "human",
              "prefill_length": 128, "decode_length": 8, "prefix_len": 0}

    def test_manifest_without_requests_key_is_not_declared_zero(self):
        # D2 修复：单侧容忍语义——service manifest.json 未声明 requests
        # 时不得被当成"声明 0 条"而 fail-closed（旧 counts[name]=0 与
        # _manifest_request_entries 的容忍语义自相矛盾，真实路径即失败）。
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            log = self._write_run(
                run_dir,
                metrics_manifest={"schema_version": 1,
                                  "requests": [dict(self._ENTRY)]},
                service_manifest={"model": {"name": "m"}},
                record=self._request_record())
            runs = mp._parse_logs([log])
            summary = mp.write_request_metrics(
                runs, run_dir / "request_metrics.csv")
            self.assertEqual(summary["request_rows"], 1)

    def test_manifest_declaring_empty_requests_still_fails(self):
        # 声明为空列表 = "声明 0 条"，与日志的 1 条 request 记录矛盾 →
        # 仍 fail-closed（修复不得放松真正的计数对账）。
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            log = self._write_run(
                run_dir,
                metrics_manifest={"schema_version": 1, "requests": []},
                service_manifest={"model": {"name": "m"},
                                  "requests": [dict(self._ENTRY)]},
                record=self._request_record())
            runs = mp._parse_logs([log])
            with self.assertRaisesRegex(mp.PostprocessError, "declares 0"):
                mp.write_request_metrics(runs, run_dir / "request_metrics.csv")

    def test_malformed_request_field_fails_via_postprocess_error(self):
        # 裸 int() ValueError/TypeError（如 "oops"）不再逃逸出
        # PostprocessError 通道。
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            log = self._write_run(
                run_dir,
                metrics_manifest={"schema_version": 1,
                                  "requests": [dict(self._ENTRY)]},
                service_manifest={"model": {"name": "m"}},
                record=self._request_record(arrival_ns="oops"))
            runs = mp._parse_logs([log])
            metrics_manifest, service_manifest = mp._load_manifest_sidecars(
                runs[0])
            entries, _ = mp._manifest_request_entries(
                runs[0], metrics_manifest, service_manifest)
            with self.assertRaises(mp.PostprocessError):
                mp._request_metric_rows(runs[0], entries)

    def test_duplicate_run_id_in_configured_group_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "cpp.log"
            lines = []
            for run_id in ("run_a", "run_b"):
                lines.append("[METRIC] " + json.dumps({
                    "schema": 1, "type": "init", "run_id": run_id,
                    "run_mode": "service"}))
                lines.append("[METRIC] " + json.dumps({
                    "schema": 1, "type": "summary", "run_mode": "service",
                    "mean_e2e_ns": 1000, "drain_tput_rps": 10}))
            log.write_text("\n".join(lines) + "\n", encoding="utf-8")
            runs = mp._parse_logs([log])
            self.assertEqual(len(runs), 2)
            group_config = {"groups": [{
                "group_id": "g", "runs": [
                    {"run_id": "run_a"}, {"run_id": "run_b"},
                    {"run_id": "run_a"}]}]}
            with self.assertRaisesRegex(mp.PostprocessError,
                                        "more than once"):
                mp.build_groups(runs, group_config)

    def test_non_numeric_iteration_time_fails_via_channel(self):
        # float("NA")/float("oops") 裸 ValueError 逃逸 → PostprocessError。
        run = mp.Run(
            run_id="m1", log_path=Path("<synthetic>"),
            init={"repo_variant": "astra-sim-wscllm",
                  "run_mode": "microbenchmark"},
            iterations=[{"iteration_time_ns": "oops"}])
        group = mp.Group(group_id="g", runs=[run], baseline_run_id=None,
                         labels_by_run={})
        with self.assertRaises(mp.PostprocessError):
            mp.build_normalized_rows([group], mp.NORMALIZATION_GROUP_MAX,
                                     None)

    def test_non_numeric_summary_field_fails_closed_main(self):
        # 端到端：summary mean_e2e_ns="NA"（非数值）→ main() 干净退出 1，
        # 不再裸 traceback。
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "cpp.log"
            lines = ["[METRIC] " + json.dumps({
                "schema": 1, "type": "init", "run_id": "r",
                "run_mode": "service"}), "[METRIC] " + json.dumps({
                    "schema": 1, "type": "summary", "run_mode": "service",
                    "mean_e2e_ns": "NA", "drain_tput_rps": 10})]
            log.write_text("\n".join(lines) + "\n", encoding="utf-8")
            code = mp.main([
                str(log),
                "--out-raw", str(Path(tmp) / "raw.csv"),
                "--out-normalized", str(Path(tmp) / "norm.csv"),
                "--out-requests", str(Path(tmp) / "req.csv")])
            self.assertEqual(code, 1)


class F2PyPostprocessChannelTests(unittest.TestCase):
    """F2_PY（2026-09-29）：RV5-F2/F3/F4/F10——D2 MP-3/SC-1 与 A1 P1-4
    同族门在 metrics_postprocess 的批间遗漏点：

      * ``_microbench_rows`` 两处裸 ``int()``（total_num_ops /
        total_local_bytes）与 ``iteration_time_ns`` 除法的裸
        ValueError/TypeError 逃逸 PostprocessError 通道（MP-3 漏点）；
      * ``_labels`` 对形状损坏（合法 JSON 但块非对象）的 manifest.json
        裸 AttributeError（MP-3 同族；与 slo_common.load_request_manifest
        的 fail-closed 姿态对齐）；
      * schema_versions 集合等值吞 bool/浮点（``{True}=={1}``、
        ``{1.0}=={1}``，SC-1 等值陷阱未扩面到 run 级/组级双门）；
      * JSON NaN 字面量经展示列（p50/p95/p99 等）原样落
        raw_metrics.csv 且退出码 0（A1 P1-4 只门归一化临界字段）——
        parse 层即拒绝（NaN 不进入任何列，落盘前有限性成立）。
    """

    @staticmethod
    def _micro_run(rank_compute, iterations):
        return mp.Run(
            run_id="mb1", log_path=Path("<synthetic>"),
            init={"repo_variant": "astra-sim-wscllm",
                  "run_mode": "microbenchmark"},
            rank_compute=rank_compute, iterations=iterations,
        )

    _LABELS = {"model": "m", "dataset": "d", "hardware": "h",
               "kv_policy": ""}

    # ---------------------------------------- MP-3 漏点：_microbench_rows --

    def test_microbench_non_int_ops_field_fails_via_channel(self):
        run = self._micro_run(
            [{"rank": 0, "total_num_ops": "oops", "total_local_bytes": 0}],
            [])
        with self.assertRaisesRegex(mp.PostprocessError, "total_num_ops"):
            mp._microbench_rows(run, self._LABELS, {})

    def test_microbench_non_int_bytes_field_fails_via_channel(self):
        run = self._micro_run(
            [{"rank": 0, "total_num_ops": 5, "total_local_bytes": "oops"}],
            [])
        with self.assertRaisesRegex(mp.PostprocessError,
                                    "total_local_bytes"):
            mp._microbench_rows(run, self._LABELS, {})

    def test_microbench_string_iteration_time_fails_via_channel(self):
        # truthy 字符串 iteration_time_ns 进得来分母分支（旧：``"1000" /
        # 1e9`` 裸 TypeError）；合法数值路径行为不变。
        run = self._micro_run(
            [{"rank": 0, "total_num_ops": 5, "total_local_bytes": 100,
              "peak_flops_per_second": 1e12,
              "local_hbm_bw_bytes_per_second": 1e12}],
            [{"iteration_time_ns": "1000"}])
        with self.assertRaisesRegex(mp.PostprocessError,
                                    "iteration_time_ns"):
            mp._microbench_rows(
                run, self._LABELS,
                {"microbenchmark": {"tp_degree": 2, "ranks": [0]}})
        good = self._micro_run(
            [{"rank": 0, "total_num_ops": 5, "total_local_bytes": 100,
              "peak_flops_per_second": 1e12,
              "local_hbm_bw_bytes_per_second": 1e12}],
            [{"iteration_time_ns": 1000}])
        rows = mp._microbench_rows(
            good, self._LABELS, {"microbenchmark": {"tp_degree": 2,
                                                    "ranks": [0]}})
        self.assertAlmostEqual(rows[0]["compute_util"],
                               5 / (2 * 1e12 * 1e-6))

    # ------------------------------------------- MP-3 同族：_labels 形状门 --

    def test_labels_malformed_manifest_blocks_fail_via_channel(self):
        # 合法 JSON 但块非对象（如 "model": "llama2" 字符串）：链式 .get
        # 曾裸 AttributeError 逃逸通道；现按 fail-closed 拒绝（与
        # kv_policy 未知枚举的既有 PostprocessError 姿态一致——标签列是
        # 比较分组键，静默降级为空串会隐匿分组破损）。
        run = mp.Run(run_id="r", log_path=Path("<synthetic>"), init={})
        for bad in ({"model": "llama2"}, {"hardware": "x"},
                    {"kv_management": "session_lru_recompute"}):
            with self.subTest(bad=bad):
                with self.assertRaisesRegex(mp.PostprocessError,
                                            "must be an object"):
                    mp._labels(run, bad, {})
        # 良构 manifest 行为不变（含缺块的既有容忍）。
        self.assertEqual(
            mp._labels(run, {"model": {"name": "m"}}, {})["model"], "m")
        self.assertEqual(mp._labels(run, {}, {})["model"], "")

    # ------------------------------------------- SC-1 同族：schema 门 --

    def test_schema_bool_masquerade_fails_run_gate(self):
        # {True}=={1}：写端谎报 "schema": true 曾直过 run 级门。
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "cpp.log"
            log.write_text(
                "[METRIC] " + json.dumps({
                    "schema": True, "type": "init", "run_id": "r",
                    "run_mode": "service"}) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(mp.PostprocessError, "schema"):
                mp._parse_logs([log])

    def test_schema_float_masquerade_fails_run_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "cpp.log"
            log.write_text(
                "[METRIC] " + json.dumps({
                    "schema": 1.0, "type": "init", "run_id": "r",
                    "run_mode": "service"}) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(mp.PostprocessError, "schema"):
                mp._parse_logs([log])

    def test_schema_bool_masquerade_fails_group_gate(self):
        # 组级门（hand-built Run 注入 {True}）同款收紧。
        run = mp.Run(run_id="r", log_path=Path("<synthetic>"), init={})
        run.schema_versions = {True}
        group_config = {"groups": [{"group_id": "g",
                                    "runs": [{"run_id": "r"}]}]}
        with self.assertRaisesRegex(mp.PostprocessError, "schema"):
            mp.build_groups([run], group_config)

    # --------------------------------- F10：JSON NaN 展示列 parse 层拒绝 --

    def test_json_nan_literal_in_summary_fails_at_parse(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "cpp.log"
            payload = json.dumps({
                "schema": 1, "type": "summary", "run_mode": "service",
                "mean_e2e_ns": 1000, "drain_tput_rps": 10,
                "p50_e2e_ns": float("nan")})
            self.assertIn("NaN", payload,
                          "构造必须是 JSON NaN 字面量而非字符串")
            log.write_text(
                "[METRIC] " + json.dumps({
                    "schema": 1, "type": "init", "run_id": "r",
                    "run_mode": "service"}) + "\n"
                + "[METRIC] " + payload + "\n", encoding="utf-8")
            with self.assertRaisesRegex(mp.PostprocessError, "NaN"):
                mp._parse_logs([log])
            # 端到端：main() 干净退出 1，不落任何含 nan 的产物。
            raw_path = Path(tmp) / "raw.csv"
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                code = mp.main([
                    str(log), "--out-raw", str(raw_path),
                    "--out-normalized", str(Path(tmp) / "norm.csv"),
                    "--out-requests", str(Path(tmp) / "req.csv")])
            self.assertEqual(code, 1)
            self.assertIn("postprocess error", err.getvalue())
            if raw_path.exists():
                self.assertNotIn("nan", raw_path.read_text().lower())

    def test_json_nan_literal_in_manifest_sidecar_fails_at_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "metrics_manifest.json").write_text(
                '{"requests": [], "note": NaN}', encoding="utf-8")
            run = mp.Run(run_id="r", log_path=Path("<synthetic>"),
                         init={"manifest_path":
                               str(run_dir / "metrics_manifest.json")})
            with self.assertRaisesRegex(mp.PostprocessError, "NaN"):
                mp._load_manifest_sidecars(run)


class DieUtilizationPipelineContractTests(unittest.TestCase):
    """Stage E2: die-backend active_kernel_die_* triplet pipeline gate.

    A run declaring the die compute backend must carry, on every
    type=rank_compute record, the three utilization values non-null and
    within [0, 1+eps]; the only legal null is the collector's no-comp-nodes
    note.  Legacy runs (no declaration / not-die note) are not asserted.
    """

    @staticmethod
    def _die_record(rank: int, *, compute=0.5, memory=0.25, vector=0.125,
                    note=None) -> dict:
        record = {
            "type": "rank_compute",
            "schema": 1,
            "rank": rank,
            "active_kernel_die_compute_util": compute,
            "active_kernel_die_memory_util": memory,
            "active_kernel_die_vector_util": vector,
        }
        if note is not None:
            record["active_kernel_die_note"] = note
        return record

    def _run_in_dir(self, tmp: str, *, metrics_manifest: "dict | None",
                    records: "list[dict]") -> "mp.Run":
        run_dir = Path(tmp) / "run"
        run_dir.mkdir(parents=True, exist_ok=True)
        if metrics_manifest is not None:
            (run_dir / "metrics_manifest.json").write_text(
                json.dumps(metrics_manifest), encoding="utf-8")
        log = run_dir / "cpp.log"
        lines = [json.dumps({"schema": 1, "type": "init", "run_id": "die-run",
                             "detail_level": "summary"})]
        lines += [json.dumps(r) for r in records]
        log.write_text(
            "".join("[METRIC] " + line + "\n" for line in lines),
            encoding="utf-8")
        runs = mp._parse_logs([log])
        self.assertEqual(len(runs), 1)
        return runs[0]

    def test_valid_die_triplet_passes(self) -> None:
        with tempfile.TemporaryDirectory(prefix="die_util_ok_") as tmp:
            run = self._run_in_dir(
                tmp,
                metrics_manifest={"schema_version": 1,
                                  "compute-cost-model": "die"},
                records=[self._die_record(0),
                         self._die_record(1, compute=0.0, memory=1.0,
                                          vector=1.0 + 1e-9)])
            mp._check_die_utilization_records([run])  # must not raise

    def test_null_with_no_comp_nodes_note_passes(self) -> None:
        with tempfile.TemporaryDirectory(prefix="die_util_note_") as tmp:
            run = self._run_in_dir(
                tmp,
                metrics_manifest={"compute-cost-model": "die"},
                records=[self._die_record(
                    0, compute=None, memory=None, vector=None,
                    note="unavailable: no comp nodes in the utilization "
                         "window")])
            mp._check_die_utilization_records([run])  # must not raise

    def test_null_with_unknown_note_fails(self) -> None:
        with tempfile.TemporaryDirectory(prefix="die_util_badnote_") as tmp:
            run = self._run_in_dir(
                tmp,
                metrics_manifest={"compute-cost-model": "die"},
                records=[self._die_record(0, vector=None,
                                          note="unavailable: something else")])
            with self.assertRaises(mp.PostprocessError):
                mp._check_die_utilization_records([run])

    def test_out_of_range_fails(self) -> None:
        with tempfile.TemporaryDirectory(prefix="die_util_range_") as tmp:
            run = self._run_in_dir(
                tmp,
                metrics_manifest={"compute-cost-model": "die"},
                records=[self._die_record(0, compute=1.5)])
            with self.assertRaises(mp.PostprocessError):
                mp._check_die_utilization_records([run])

    def test_non_numeric_fails(self) -> None:
        with tempfile.TemporaryDirectory(prefix="die_util_type_") as tmp:
            run = self._run_in_dir(
                tmp,
                metrics_manifest={"compute-cost-model": "die"},
                records=[self._die_record(0, memory="0.5")])
            with self.assertRaises(mp.PostprocessError):
                mp._check_die_utilization_records([run])

    def test_die_run_without_rank_compute_records_fails(self) -> None:
        with tempfile.TemporaryDirectory(prefix="die_util_empty_") as tmp:
            run = self._run_in_dir(
                tmp,
                metrics_manifest={"compute-cost-model": "die"},
                records=[])
            with self.assertRaises(mp.PostprocessError):
                mp._check_die_utilization_records([run])

    def test_legacy_run_is_not_asserted(self) -> None:
        with tempfile.TemporaryDirectory(prefix="die_util_legacy_") as tmp:
            run = self._run_in_dir(
                tmp,
                metrics_manifest={},
                records=[self._die_record(
                    0, compute=None, memory=None, vector=None,
                    note="unavailable: compute-cost-model is not die")])
            mp._check_die_utilization_records([run])  # must not raise

    def test_service_manifest_declaration_enforced(self) -> None:
        # metrics_manifest without the key, service manifest.json declaring
        # die (the stage-C service manifest path).
        with tempfile.TemporaryDirectory(prefix="die_util_svc_") as tmp:
            run_dir = Path(tmp) / "run"
            run_dir.mkdir(parents=True)
            (run_dir / "metrics_manifest.json").write_text("{}", encoding="utf-8")
            (run_dir / "manifest.json").write_text(
                json.dumps({"compute-cost-model": "die"}), encoding="utf-8")
            log = run_dir / "cpp.log"
            lines = [json.dumps({"schema": 1, "type": "init",
                                 "run_id": "svc-die",
                                 "detail_level": "summary"}),
                     json.dumps(self._die_record(0, compute=2.0))]
            log.write_text(
                "".join("[METRIC] " + line + "\n" for line in lines),
                encoding="utf-8")
            runs = mp._parse_logs([log])
            with self.assertRaises(mp.PostprocessError):
                mp._check_die_utilization_records(runs)

    def test_record_inference_when_no_manifest_declares(self) -> None:
        # No manifest declaration at all: the records themselves identify a
        # die run (non-null die utils) and must be range-checked.
        with tempfile.TemporaryDirectory(prefix="die_util_infer_") as tmp:
            run = self._run_in_dir(
                tmp, metrics_manifest=None,
                records=[self._die_record(0, compute=-0.75)])
            with self.assertRaises(mp.PostprocessError):
                mp._check_die_utilization_records([run])


if __name__ == "__main__":
    unittest.main()
