#!/usr/bin/env python3
"""test_derive_source_guards.py -- D3 (2026-09-28) derive_20_first_30_seconds
fail-closed source guards, pinned end-to-end through the script's real main().

Contract under test (each shape must abort with exit 2, print the offending
source line on stderr, and leave NO queue/sidecar/provenance output behind --
the P1 atomic-materialization staging must clean up on the fail-closed exit):

  - absolute-arrival row re-opening the CURRENT session block (mid-session
    restart): would silently reset turn_index to 0 and emit a duplicate
    (session, turn-0) row -- the pre-fix hole this batch closes;
  - absolute-arrival row re-opening an already finished session
    (non-contiguous block -- the same shape the C++ reader fail-closes
    on deep inside a run);
  - absolute-arrival row with an EMPTY session_id (materialized a literal
    "session_" pseudo-session pre-fix);
  - session_id containing comma / double quote / CR / LF (cannot round-trip
    the positionally comma-split 8-column queue contract);
  - prefill_length / decode_length <= 0 (zero-length three-port lexicon:
    loader and C++ reader reject, the writer must not admit);
  - continuation row switching session id (the pre-existing 105-② guard,
    kept green by this suite).

Plus the behavior pins that must NOT change: the frozen arrival chain
(turn i arrival = turn i-1 arrival + turn i-1 row's gap) and the banker's
rounding comment fix (a half-grid point deviates by EXACTLY 500 ns, never
more -- round-half-to-even lands on the even neighbour).
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

MODULE_DIR = Path(__file__).resolve().parent
DERIVE = MODULE_DIR / "traces" / "derive_20_first_30_seconds.py"

SOURCE_HEADER = (
    "arrival_time,session_id,request_id,prefix_len,prefill_length,"
    "decode_length,human_time,tool_time"
)


class _DeriveCase:
    """Runs the real derive script against a synthetic source in a temp dir."""

    def __init__(self, rows, window_ns="30000000000", scale=None):
        self.tmp = tempfile.TemporaryDirectory()
        self.source = Path(self.tmp.name) / "source.csv"
        self.source.write_text(
            SOURCE_HEADER + "\n" + "\n".join(rows) + "\n", encoding="utf-8")
        self.queue = Path(self.tmp.name) / "queue.csv"
        self.sidecar = Path(self.tmp.name) / "sidecar.csv"
        self.provenance = Path(str(self.queue) + ".provenance.json")
        argv = [sys.executable, str(DERIVE), str(self.source),
                str(self.queue), str(self.sidecar), window_ns]
        if scale is not None:
            argv.append(scale)
        self.result = subprocess.run(
            argv, capture_output=True, text=True, timeout=120)

    def outputs_absent(self):
        return (not self.queue.exists() and not self.sidecar.exists()
                and not self.provenance.exists())

    def cleanup(self):
        self.tmp.cleanup()


class DeriveFailClosedGuards(unittest.TestCase):
    def _run(self, rows, **kwargs):
        case = _DeriveCase(rows, **kwargs)
        self.addCleanup(case.cleanup)
        return case

    def test_happy_path_publishes_all_three_outputs(self):
        case = self._run([
            "1000000,s0,0,0,100,50,500000,",
            ",s0,1,0,100,50,,",
            "2000000,s1,0,0,100,50,,300000",
        ])
        self.assertEqual(case.result.returncode, 0, case.result.stderr)
        self.assertTrue(case.queue.exists())
        self.assertTrue(case.sidecar.exists())
        self.assertTrue(case.provenance.exists())
        lines = case.queue.read_text(encoding="utf-8").splitlines()
        self.assertEqual(
            lines[0],
            "session_id,turn_index,request_id,prefill_length,decode_length,"
            "session_arrival_time_ns,inter_request_interval_ns,description")
        self.assertEqual(len(lines), 4)  # header + 3 data rows
        provenance = json.loads(case.provenance.read_text(encoding="utf-8"))
        self.assertEqual(provenance["schema"], 1)
        self.assertEqual(provenance["data_rows"], 3)
        self.assertEqual(provenance["sessions"], 2)
        self.assertTrue(provenance["session_blocks_contiguous"])

    def test_mid_session_restart_fail_closed(self):
        """The pre-fix hole: an absolute-arrival row inside the current
        session silently reset turn_index to 0 (duplicate (session, turn-0)
        row, zero detection)."""
        case = self._run([
            "1000000,s0,0,0,100,50,500000,",
            "2000000,s0,0,0,100,50,,",  # same session, new absolute arrival
        ])
        self.assertEqual(case.result.returncode, 2)
        self.assertIn("re-opens the CURRENT session block", case.result.stderr)
        self.assertIn("line 3", case.result.stderr)
        self.assertTrue(case.outputs_absent())

    def test_noncontiguous_session_reappearance_fail_closed(self):
        case = self._run([
            "1000000,s0,0,0,100,50,500000,",
            "2000000,s1,0,0,100,50,,",
            "3000000,s0,0,0,100,50,,",  # finished session re-opened
        ])
        self.assertEqual(case.result.returncode, 2)
        self.assertIn("re-opens the already finished session",
                      case.result.stderr)
        self.assertTrue(case.outputs_absent())

    def test_empty_session_id_with_arrival_fail_closed(self):
        """Pre-fix this materialized a literal "session_" pseudo-session."""
        case = self._run([
            "1000000,,0,0,100,50,500000,",
        ])
        self.assertEqual(case.result.returncode, 2)
        self.assertIn("empty session_id", case.result.stderr)
        self.assertIn("session_", case.result.stderr)
        self.assertTrue(case.outputs_absent())

    def test_session_id_with_comma_fail_closed(self):
        case = self._run(['1000000,"s,0",0,0,100,50,,'])
        self.assertEqual(case.result.returncode, 2)
        self.assertIn("cannot round-trip the queue contract",
                      case.result.stderr)
        self.assertTrue(case.outputs_absent())

    def test_session_id_with_quote_fail_closed(self):
        case = self._run(['1000000,s"0,0,0,100,50,,'])
        self.assertEqual(case.result.returncode, 2)
        self.assertIn("cannot round-trip the queue contract",
                      case.result.stderr)
        self.assertTrue(case.outputs_absent())

    def test_zero_length_rows_fail_closed(self):
        for prefill, decode in (("0", "50"), ("100", "0")):
            case = self._run([f"1000000,s0,0,0,{prefill},{decode},,"])
            with self.subTest(prefill=prefill, decode=decode):
                self.assertEqual(case.result.returncode, 2)
                self.assertIn("zero or negative request lengths",
                              case.result.stderr)
                self.assertTrue(case.outputs_absent())

    def test_continuation_row_session_switch_still_fail_closed(self):
        """Regression pin: the pre-existing 105-② guard survives the D3
        refactor (now routed through the shared _die channel)."""
        case = self._run([
            "1000000,s0,0,0,100,50,500000,",
            ",s1,0,0,100,50,,",  # continuation row switching session id
        ])
        self.assertEqual(case.result.returncode, 2)
        self.assertIn("not grouped by session", case.result.stderr)
        self.assertTrue(case.outputs_absent())


class DeriveBehaviorPins(unittest.TestCase):
    def test_arrival_chain_and_intervals_unchanged(self):
        """Frozen semantics pin: turn i arrival = turn i-1 arrival + the gap
        on turn i-1's row; turn i's queue interval = turn i-1's row gap."""
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        source = Path(tmp.name) / "source.csv"
        source.write_text(
            SOURCE_HEADER + "\n"
            "1000000,s0,0,0,100,50,2000000,\n"
            ",s0,1,0,100,50,,\n"
            ",s0,2,0,100,50,,\n",
            encoding="utf-8")
        queue = Path(tmp.name) / "queue.csv"
        result = subprocess.run(
            [sys.executable, str(DERIVE), str(source), str(queue),
             str(Path(tmp.name) / "sidecar.csv"), "30000000000"],
            capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stderr)
        lines = queue.read_text(encoding="utf-8").splitlines()[1:]
        self.assertEqual(lines[0].split(",")[5], "1000000")   # t0 absolute
        self.assertEqual(lines[0].split(",")[6], "")          # turn-0 no gap
        self.assertEqual(lines[1].split(",")[6], "2000000")   # s0 row-0 gap
        self.assertEqual(lines[2].split(",")[6], "0")         # empty gap = 0

    def test_bankers_rounding_half_grid_point_deviation_exactly_500ns(self):
        """D3 comment fix pin: scale=2.0 over t0=1000 ns lands on the exact
        half-grid point 500 ns; round-half-to-even floors it to 0, so the
        deviation is EXACTLY 500 ns (the old "< 500 ns" comment was wrong)."""
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        source = Path(tmp.name) / "source.csv"
        source.write_text(
            SOURCE_HEADER + "\n1000,s0,0,0,100,50,,\n", encoding="utf-8")
        queue = Path(tmp.name) / "queue.csv"
        result = subprocess.run(
            [sys.executable, str(DERIVE), str(source), str(queue),
             str(Path(tmp.name) / "sidecar.csv"), "30000000000", "2.0"],
            capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stderr)
        row = queue.read_text(encoding="utf-8").splitlines()[1]
        fields = row.split(",")
        self.assertEqual(fields[5], "0")   # 500 ns rounded to the even grid 0
        self.assertIn("turn-0 scaled arrivals rounded to the 1000 ns grid "
                      "(loader timing%1000 gate): 1", result.stdout)


if __name__ == "__main__":
    unittest.main()
