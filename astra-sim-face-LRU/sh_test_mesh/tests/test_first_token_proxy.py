#!/usr/bin/env python3
"""B4/WP9 fallback: train-interpolated first-token proxy unit tests (face).

Covers the B3_FACE 60s gate-2 failure fallback implemented in
workload/llama2_7b_inference/metrics_postprocess.py:

  1. authoritative W_bytes / KV-bytes-per-token conversions (frozen
     face_scheduler formulas fed by trace_config.csv -- no invented values);
  2. the proxy formula hand-check on a synthetic train ledger index;
  3. end-to-end ``_request_metric_rows``: exact value wins; missing exact
     + joiner ledger row -> train_interpolated; ledger absent -> NA;
  4. the proxy never leaks into the SLO judgment path is proven separately
     in slo_tools/tests/test_slo_contract.py (assert_no_proxy_columns +
     violation fail-closed on proxy columns).

Run: python3 sh_test_mesh/tests/test_first_token_proxy.py
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

_TESTS_DIR = Path(__file__).resolve().parent
_SH_TEST_DIR = _TESTS_DIR.parents[0]
_WORKLOAD_DIR = _SH_TEST_DIR / "workload" / "llama2_7b_inference"
for _p in (str(_WORKLOAD_DIR), str(_TESTS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import metrics_postprocess as mp  # noqa: E402


def _ledger_row(**overrides):
    row = {
        "train_id": "batch_train_i0_1",
        "instance_index": 0,
        "tick": 1_000_000,
        "iterations": 8,
        "member_count": 1,
        "member_iterations": 8,
        "joiners": [],
        "drains": [],
        "exits": [],
        "prefill_chunks": 8,
        "pass_spans": 16,
    }
    row.update(overrides)
    return row


def _fake_index(first_train_by_request, next_tick_by_position):
    index = mp._TrainProxyIndex.__new__(mp._TrainProxyIndex)
    index._run = None
    index._loaded = True
    index._available = True
    index.first_train_by_request = first_train_by_request
    index.next_tick_by_position = next_tick_by_position
    return index


class ModelBytesTests(unittest.TestCase):
    def test_authoritative_conversions(self):
        """trace_config.csv (llama2_7b, swiglu) -> frozen face_scheduler
        numbers (test_face_scheduler.py:371 pins the same value)."""

        weight_bytes, kv_per_token = mp._load_model_bytes()
        self.assertEqual(weight_bytes, 13_476_831_232)
        self.assertEqual(kv_per_token, 2 * 32 * 4096 * 2)


class ProxyFormulaTests(unittest.TestCase):
    def _entry(self, **overrides):
        entry = {
            "request_id": "session_0_request_0",
            "queue_index": 0,
            "session_id": "session_0",
            "turn_index": 0,
            "request_type": "human",
            "prefill_length": 4096,
            "decode_length": 16,
            "history_tokens_before": 0,
            "prefill_context_tokens": 4096,
            "final_context_tokens": 4112,
        }
        entry.update(overrides)
        return entry

    def _record(self, **overrides):
        record = {
            "queue_index": 0,
            "request_id": "session_0_request_0",
            "session_id": "session_0",
            "turn_index": 0,
            "completed": True,
            "arrival_ns": 100,
            "prefill_start_ns": 200,
            "prefill_end_ns": 300,
            "decode_start_ns": 1_000_000,
            "first_token_ns": None,
            "completion_ns": 50_000_000,
        }
        record.update(overrides)
        return record

    def test_hand_check_formula(self):
        """w_i = W + KV(ctx+0+i), i=1..N (train iterations); proxy =
        decode_start + (w1/sum w_i)*(train_end - decode_start).  Hand-
        computed.  decode_length=16 > iterations=8 -> debut rides the
        whole train; P == N."""

        index = _fake_index(
            {"session_0_request_0": (0, _ledger_row(tick=1_000_000))},
            {0: 9_000_000},
        )
        weight_bytes, kv_per_token = mp._load_model_bytes()
        context, decode_length, iterations = 4096, 16, 8
        self.assertEqual(min(decode_length, iterations), 8)
        weights = [
            weight_bytes + kv_per_token * (context + step)
            for step in range(1, iterations + 1)
        ]
        expected = int(round(
            1_000_000
            + (weights[0] / sum(weights)) * (9_000_000 - 1_000_000)))
        value, note = mp._first_token_proxy_value(
            index, self._record(), self._entry())
        self.assertEqual(value, expected)
        self.assertIn("train_interpolated", note)
        self.assertIn("N=8", note)
        self.assertIn("P=8", note)

    def test_short_debut_interpolates_within_train(self):
        """decode_length=4 (P=4) inside an 8-iteration train: the weight
        sum still runs over the TRAIN's N=8 iterations (first token lands
        with iteration 1), NOT the debut's participation -- summing over
        P would overstate the share whenever the debut exits before the
        train boundary (S3's 60s reference run: 7/1454 rows before this
        fix).  decode_length==1 is no longer interpolated but pinned to
        completion -- see test_dl1_pinned_to_completion."""
        index = _fake_index(
            {"r": (0, _ledger_row(tick=1_000_000, iterations=8))},
            {0: 9_000_000},
        )
        weight_bytes, kv_per_token = mp._load_model_bytes()
        context = 4096
        weights = [
            weight_bytes + kv_per_token * (context + step)
            for step in range(1, 8 + 1)
        ]
        expected = int(round(
            1_000_000 + (weights[0] / sum(weights)) * 8_000_000))
        value, note = mp._first_token_proxy_value(
            index,
            self._record(request_id="r"),
            self._entry(request_id="r", decode_length=4,
                        final_context_tokens=4100))
        self.assertEqual(value, expected)
        self.assertLess(value, 9_000_000)
        self.assertIn("N=8", note)
        self.assertIn("P=4", note)

    def test_dl1_pinned_to_completion(self):
        """decode_length==1 debut inside a MULTI-iteration train: share<1
        interpolation lands BEFORE the recorded completion tick, but
        exact-mode semantics for dl1 is first_token==completion -- pin and
        say so (face 0902 full_rt: 18/268 dl1 rows, up to 4.9s early)."""
        index = _fake_index(
            {"r": (0, _ledger_row(tick=1_000_000, iterations=8))},
            {0: 9_000_000},
        )
        value, note = mp._first_token_proxy_value(
            index,
            self._record(request_id="r", completion_ns=5_000_000),
            self._entry(request_id="r", decode_length=1,
                        final_context_tokens=4097))
        # raw interpolation would land at ~1M + share*8M << 5M
        self.assertEqual(value, 5_000_000)
        self.assertIn("pinned_to_completion", note)

    def test_clamped_to_completion_for_one_iteration_train(self):
        """N=1 退化列车：share=1 -> proxy=下一发射边界 > completion（边界晚
        于 end barrier）——钳到 completion 并留痕。"""
        index = _fake_index(
            {"r": (0, _ledger_row(tick=1_000_000, iterations=1))},
            {0: 9_000_000},
        )
        value, note = mp._first_token_proxy_value(
            index,
            self._record(request_id="r", completion_ns=5_000_000),
            self._entry(request_id="r", decode_length=1))
        self.assertEqual(value, 5_000_000)
        self.assertIn("clamped_to_completion", note)

    def test_last_train_on_instance_not_obtainable(self):
        index = _fake_index(
            {"r": (0, _ledger_row(tick=1_000_000))}, {})
        value, note = mp._first_token_proxy_value(
            index, self._record(request_id="r"), self._entry(request_id="r"))
        self.assertIsNone(value)
        self.assertIn("no_next_train_on_instance", note)

    def test_no_joiner_row_not_obtainable(self):
        index = _fake_index({}, {})
        value, note = mp._first_token_proxy_value(
            index, self._record(), self._entry())
        self.assertIsNone(value)
        self.assertIn("no_train_ledger_joiner_row", note)


class EndToEndRowsTests(unittest.TestCase):
    """_request_metric_rows over a synthetic log + manifest + ledger."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(
            prefix="slo_wps_proxy_test_")
        self.run_dir = Path(self._tmp.name)
        self.log_path = self.run_dir / "cpp.log"
        self.results_dir = self.run_dir / "results"
        self.results_dir.mkdir(parents=True)
        manifest_path = self.run_dir / "generated" / "metrics_manifest.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text("{}", encoding="utf-8")
        # manifest.json sidecar: the request_metrics join input.
        (manifest_path.parent / "manifest.json").write_text(json.dumps({
            "requests": [
                {
                    "queue_index": 0,
                    "request_id": "session_0_request_0",
                    "session_id": "session_0",
                    "turn_index": 0,
                    "request_type": "human",
                    "prefill_length": 4096,
                    "decode_length": 16,
                    "prefill_context_tokens": 4096,
                    "final_context_tokens": 4112,
                },
                {
                    "queue_index": 1,
                    "request_id": "session_1_request_0",
                    "session_id": "session_1",
                    "turn_index": 0,
                    "request_type": "tool",
                    "prefill_length": 2048,
                    "decode_length": 4,
                    "prefill_context_tokens": 2048,
                    "final_context_tokens": 2052,
                },
            ],
        }), encoding="utf-8")

        def request_record(queue_index, request_id, session_id,
                           decode_start, first_token=None):
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
                "prefill_decode_gap_ns": (
                    decode_start - (300 + queue_index) - 200),
                "decode_ns": (
                    decode_start + 40_000_000 - (100 + queue_index)
                    - 100 - 100 - (decode_start - (300 + queue_index) - 200)
                ),
            }
            if first_token is not None:
                record["first_token_ns"] = first_token
            return record

        self.records = [
            {"type": "init", "schema": 1, "run_id": "proxytest",
             "manifest_path": str(manifest_path), "detail_level": "full",
             "run_mode": "service"},
            request_record(0, "session_0_request_0", "session_0",
                           1_000_000),
            request_record(1, "session_1_request_0", "session_1",
                           2_000_000, first_token=2_500_000),
        ]

    def tearDown(self):
        self._tmp.cleanup()

    def _write_log(self):
        with self.log_path.open("w", encoding="utf-8") as sink:
            for record in self.records:
                sink.write("[METRIC] " + json.dumps(record) + "\n")

    def _rows(self):
        runs = mp._parse_logs([self.log_path])
        entries, _counts = mp._manifest_request_entries(runs[0], {}, {
            "requests": [
                entry for entry in json.loads(
                    (Path(self.records[0]["manifest_path"]).parent /
                     "manifest.json").read_text(encoding="utf-8")
                )["requests"]
            ]
        })
        return {row["request_id"]: row
                for row in mp._request_metric_rows(runs[0], entries)}

    def test_exact_wins_proxy_fills_missing_ledger_absent_means_na(self):
        self._write_log()  # no results/train_ledger.jsonl at all
        rows = self._rows()
        # exact record: value + source untouched.
        self.assertEqual(rows["session_1_request_0"]["first_token_ns"],
                         2_500_000)
        self.assertEqual(
            rows["session_1_request_0"]["first_token_source"], "exact")
        # no exact + no ledger -> NA with the run-level reason note.
        self.assertEqual(rows["session_0_request_0"]["first_token_ns"], "NA")
        self.assertEqual(rows["session_0_request_0"]["first_token_source"],
                         "NA")
        self.assertIn("proxy_unavailable:no_train_ledger",
                      rows["session_0_request_0"]["instructions"])

    def test_proxy_filled_from_ledger_and_tagged(self):
        self._write_log()
        with (self.results_dir / "train_ledger.jsonl").open(
                "w", encoding="utf-8") as sink:
            sink.write(json.dumps(_ledger_row(
                train_id="batch_train_i0_1", tick=1_000_000,
                joiners=["session_0_request_0"])) + "\n")
            sink.write(json.dumps(_ledger_row(
                train_id="batch_train_i0_2", tick=9_000_000)) + "\n")
        rows = self._rows()
        row = rows["session_0_request_0"]
        weight_bytes, kv_per_token = mp._load_model_bytes()
        weights = [weight_bytes + kv_per_token * (4096 + step)
                   for step in range(1, 9)]
        expected = int(round(
            1_000_000 + (weights[0] / sum(weights)) * 8_000_000))
        self.assertEqual(row["first_token_ns"], expected)
        self.assertEqual(row["first_token_source"], "train_interpolated")
        # first_step rows (split-ON runs) are skipped for joiner matching.
        with (self.results_dir / "train_ledger.jsonl").open(
                "w", encoding="utf-8") as sink:
            sink.write(json.dumps(_ledger_row(
                train_id="batch_train_i0_1", tick=1_000_000, first_step=True,
                joiners=["session_0_request_0"])) + "\n")
            sink.write(json.dumps(_ledger_row(
                train_id="batch_train_i0_1", tick=1_000_000,
                joiners=["session_0_request_0"])) + "\n")
            sink.write(json.dumps(_ledger_row(
                train_id="batch_train_i0_2", tick=9_000_000)) + "\n")
        rows = self._rows()
        self.assertEqual(rows["session_0_request_0"]["first_token_ns"],
                         expected)


if __name__ == "__main__":
    unittest.main()
