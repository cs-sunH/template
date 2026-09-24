"""The ledger exit lists members at launch; joint drains at completion."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

SLO_TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SLO_TOOLS))

import load_imbalance  # noqa: E402
from slo_common import SloToolError  # noqa: E402


class JointCompletionDrainTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.run_dir = Path(self.temp.name)
        (self.run_dir / "results").mkdir()

    def _write(self, completions):
        decisions = [
            {"kind": "prefill", "request_id": "r0", "tick": 100,
             "decision": {}},
            {"kind": "decode", "request_id": "r0", "tick": 120,
             "decision": {"decode_instance_index": 2}},
        ] + [
            {"kind": "completion", "request_id": "r0", "tick": tick,
             "decision": {}} for tick in completions
        ]
        ledger = [{"train_id": "batch_train_i2_1", "instance_index": 2,
                   "tick": 150, "first_step": False, "exits": ["r0"]}]
        for name, rows in (("online_decision_log.jsonl", decisions),
                           ("train_ledger.jsonl", ledger)):
            with (self.run_dir / "results" / name).open("w") as sink:
                for row in rows:
                    sink.write(json.dumps(row) + "\n")

    def test_joint_uses_completion_after_ledger_launch(self):
        self._write([200])
        intervals = load_imbalance.collect_intervals(
            self.run_dir, "astra-sim-joint")
        self.assertEqual(intervals, [{
            "request_id": "r0", "instance": 2, "admission_ns": 100,
            "drain_ns": 200}])

    def test_joint_missing_completion_fails_closed(self):
        self._write([])
        with self.assertRaisesRegex(SloToolError, "缺 completion"):
            load_imbalance.collect_intervals(self.run_dir,
                                             "astra-sim-joint")

    def test_legacy_ledger_proxy_is_labeled(self):
        self._write([])
        intervals = load_imbalance.collect_intervals(
            self.run_dir, "astra-sim-face")
        self.assertEqual(intervals[0]["drain_ns"], 150)

    def test_joint_completion_before_launch_is_rejected(self):
        self._write([140])
        with self.assertRaisesRegex(SloToolError, "早于 train_ledger"):
            load_imbalance.collect_intervals(self.run_dir,
                                             "astra-sim-joint")


if __name__ == "__main__":
    unittest.main()
