from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest


_WORKLOAD_DIR = Path(__file__).resolve().parents[1] / "workload" / "llama2_7b_inference"
if str(_WORKLOAD_DIR) not in sys.path:
    sys.path.insert(0, str(_WORKLOAD_DIR))

_SLO_TOOLS_DIR = Path(__file__).resolve().parents[1] / "slo_tools"
if str(_SLO_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_SLO_TOOLS_DIR))

import metrics_schema as ms
import metrics_postprocess as mp
import slo_common


_CONFIG_PATH = _WORKLOAD_DIR / "metrics_config.json"


# (2026-09-29) 原 manifest/记忆层夹具助手(_request/_builder/
# _add_service_boundaries/_service_manifest/_delta)随 metrics_schema
# 死代码层删除一并移除;仅保留仍被存活测试使用的 _json_round_trip。
def _json_round_trip(value):
    return json.loads(json.dumps(value))


# (2026-09-29) ArrivalTests / RequestTimingTrackerTests /
# MemoryObserverTests 已随 metrics_schema 死代码层(Arrival /
# RequestTimingTracker / MemoryMetricsObserver)同步删除。
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


# (2026-09-29) ManifestTests 已随 metrics_schema 死代码层(MetricsManifest
# / MetricManifestBuilder / check_schema_version)同步删除;其中仍有效的
# 事件边编码钉住用例改写并入 FirstTokenAndRequestMetricsContractTests。
class JsonRoundTripTests(unittest.TestCase):

    def test_node_event_round_trip(self) -> None:
        event = ms.NodeMetricEvent(456, 2, 7)
        self.assertEqual(
            ms.NodeMetricEvent.from_triple(_json_round_trip(event.to_triple())), event
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
    def test_metrics_config_json_parses_and_pins_only_live_keys(self) -> None:
        # A4-8 (2026-09-26): the dead "memory" (chiplets_per_npu value was
        # never read; the chiplet count travels in the cross-repo manifest's
        # chiplets array -- this repo carries no built-in default) and
        # "microbenchmark"
        # sections are deleted -- no code consumer existed for either.  The
        # config now carries exactly the two consumed keys.
        config = json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
        self.assertEqual(
            set(config), {"schema_version", "detail_level"}
        )
        self.assertEqual(config["schema_version"], ms.SCHEMA_VERSION)
        self.assertIn(config["detail_level"], ("off", "summary", "full"))


class FirstTokenAndRequestMetricsContractTests(unittest.TestCase):
    """B3 additions: frozen cross-repo contract extensions.

    Canonical union of the three B3 batches: event code 8 (constant, edge
    mapping, SERVICE_EVENT_CODES membership), the request_metrics.csv
    column order (29 columns, frozen), the terminal_status /
    first_token_source enumerations, and the NA placeholder semantics for
    fields that are not yet filled.

    A4-16 (2026-09-26): the terminal_status / first_token_source / NA
    assertions are no longer self-proving constant comparisons -- they are
    checked against the values the implementation actually writes into
    request_metrics rows (mp._request_metric_rows over a synthetic run
    covering completed/failed, exact/proxy/NA first-token sources).  This
    couples the block to each repository's own metrics_postprocess
    implementation, so the former "byte-identical across the five repos"
    property holds only for the schema-symbol assertions (ms.*) and the
    frozen column list; the enum assertions are per-repo by design.
    """

    TERMINAL_STATUS_VALUES = ("completed", "failed")
    FIRST_TOKEN_SOURCE_VALUES = ("exact", "train_interpolated", "NA")
    NA_SENTINEL = "NA"

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="face_contract_rows_")
        run_dir = Path(self._tmp.name) / "run"
        (run_dir / "results").mkdir(parents=True)
        # Run-dir sidecars (A4-4 resolution: the copy next to the log wins).
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
        # A4-9 (2026-09-26): single authoritative vocabulary across the
        # pipeline -- the SLO consumer side (slo_common.TERMINAL_STATUSES,
        # fail-closed CSV validation) must freeze the exact set the
        # producer writes here.
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


class GateContractTests(unittest.TestCase):
    """C10 (2026-09-26): regression coverage for the fail-closed gates.

    Covers the A4-1 consistency hard gate, the M7-2 duplicate-run_id dedup,
    and the A4-18 full-detail empty-request gate -- all of which changed
    postprocess behavior without any test assertion at fix time.
    """

    @staticmethod
    def _run(
        run_id: str = "r1",
        *,
        consistency: "list[dict] | None" = None,
        summary: "dict | None" = None,
        detail_level: "str | None" = "full",
    ) -> "mp.Run":
        init: dict = {"detail_level": detail_level}
        if detail_level is None:
            init.pop("detail_level")
        return mp.Run(
            run_id=run_id,
            log_path=Path("/tmp") / f"{run_id}.log",
            init=init,
            summaries=[summary or {}] if summary is not None else [],
            consistency=list(consistency or []),
        )

    # -- A4-1 consistency hard gate ------------------------------------

    def test_missing_consistency_record_fails(self) -> None:
        run = self._run(consistency=[])
        with self.assertRaises(mp.PostprocessError):
            mp._check_consistency_records([run])

    def test_ok_false_consistency_record_fails(self) -> None:
        run = self._run(consistency=[{"ok": False, "violations": [],
                                      "dropped_events": 0}])
        with self.assertRaises(mp.PostprocessError):
            mp._check_consistency_records([run])

    def test_nonempty_violations_fail(self) -> None:
        run = self._run(consistency=[{"ok": True,
                                      "violations": ["v0"],
                                      "dropped_events": 0}])
        with self.assertRaises(mp.PostprocessError):
            mp._check_consistency_records([run])

    def test_dropped_events_fail(self) -> None:
        run = self._run(consistency=[{"ok": True, "violations": [],
                                      "dropped_events": 2}])
        with self.assertRaises(mp.PostprocessError):
            mp._check_consistency_records([run])

    def test_clean_consistency_record_passes(self) -> None:
        run = self._run(consistency=[{"ok": True, "violations": [],
                                      "dropped_events": 0}])
        mp._check_consistency_records([run])  # must not raise

    # -- M7-2 duplicate run_id dedup ------------------------------------

    def _write_log(self, path: Path, run_id: str, summary: dict) -> None:
        lines = [
            json.dumps({"schema": 1, "type": "init", "run_id": run_id,
                        "detail_level": "summary"}),
            json.dumps({"schema": 1, "type": "summary", **summary}),
        ]
        path.write_text(
            "".join("[METRIC] " + line + "\n" for line in lines),
            encoding="utf-8",
        )

    def test_duplicate_run_id_with_same_summary_dedups(self) -> None:
        with tempfile.TemporaryDirectory(prefix="gate_dedup_") as tmp:
            log_a = Path(tmp) / "a.log"
            log_b = Path(tmp) / "b.log"
            self._write_log(log_a, "r1", {"completed_unique_requests": 3})
            self._write_log(log_b, "r1", {"completed_unique_requests": 3})
            runs = mp._parse_logs([log_a, log_b])
            self.assertEqual(len(runs), 1)
            self.assertEqual(runs[0].summary["completed_unique_requests"], 3)

    def test_duplicate_run_id_with_conflicting_summary_fails(self) -> None:
        with tempfile.TemporaryDirectory(prefix="gate_conflict_") as tmp:
            log_a = Path(tmp) / "a.log"
            log_b = Path(tmp) / "b.log"
            self._write_log(log_a, "r1", {"completed_unique_requests": 3})
            self._write_log(log_b, "r1", {"completed_unique_requests": 4})
            with self.assertRaises(mp.PostprocessError):
                mp._parse_logs([log_a, log_b])

    # -- A4-18 full-detail empty-request gate ----------------------------

    def test_full_detail_claims_without_records_fail(self) -> None:
        run = self._run(
            summary={"completed_unique_requests": 2,
                     "incomplete_requests": 0},
            detail_level="full",
        )
        with self.assertRaises(mp.PostprocessError):
            mp._check_request_completion(run)

    def test_full_detail_empty_queue_passes(self) -> None:
        run = self._run(
            summary={"completed_unique_requests": 0,
                     "incomplete_requests": 0},
            detail_level="full",
        )
        mp._check_request_completion(run)  # must not raise

    def test_summary_detail_empty_requests_pass(self) -> None:
        run = self._run(detail_level="summary", summary={})
        mp._check_request_completion(run)  # legal below full detail

    def test_unrecognized_detail_level_fails(self) -> None:
        run = self._run(detail_level=None, summary={})
        with self.assertRaises(mp.PostprocessError):
            mp._check_request_completion(run)


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
