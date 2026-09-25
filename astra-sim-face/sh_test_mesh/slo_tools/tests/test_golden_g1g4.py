#!/usr/bin/env python3
"""T2 golden case 骨架：G1/G2/G3/G4（合成 request_metrics/manifest/anchor
样本断言手算值）。

本批（B2-线4）交付脚本+合成样本断言；**仿真侧 fixture 留待 B3 运行**
（真实 2s 窗 full 档产物接入后，用同一断言口径复跑）。

运行：python3 sh_test_mesh/slo_tools/tests/test_golden_g1g4.py
"""

from __future__ import annotations

import json
import math
import sys
import unittest
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
SLO_TOOLS_DIR = TESTS_DIR.parent
sys.path.insert(0, str(SLO_TOOLS_DIR))
# pytest prepend 模式下本目录含 __init__.py，用例以 tests.* 包方式导入，
# tests/ 本身不在 sys.path，`import synthetic` 需显式补上本目录
# （unittest discover / 直跑模式本就以本目录解析 synthetic，再插一次无害）。
sys.path.insert(0, str(TESTS_DIR))

import synthetic  # noqa: E402
from slo_common import SloToolError  # noqa: E402
import slo_stats  # noqa: E402
import load_imbalance  # noqa: E402
import restore_decomposition  # noqa: E402
import kv_cache_adapter  # noqa: E402
import hopbytes  # noqa: E402


def slo_argv(argv):
    return slo_stats.build_parser().parse_args(argv)


def adapter_args(run_dir, **kw):
    import argparse
    return argparse.Namespace(
        run_dir=run_dir,
        output=kw.get("output", str(run_dir / "cache_events.csv")),
        hit_states=kw.get("hit_states", str(run_dir / "kv_hit_states.csv")),
        json=kw.get("json", ""),
        reconcile=kw.get("reconcile", False),
        repo_variant=kw.get("repo_variant"),
        request_manifest=kw.get("request_manifest"))


def li_args(run_dir, manifest):
    import argparse
    return argparse.Namespace(run_dir=run_dir, manifest=manifest,
                              output="-", json="", repo_variant=None)


def hb_args(run_dir, **kw):
    import argparse
    return argparse.Namespace(
        run_dir=run_dir,
        output=kw.get("output", str(run_dir / "hb_total.csv")),
        per_request=kw.get("per_request",
                           str(run_dir / "hb_per_req.csv")),
        json="", repo_variant=kw.get("repo_variant"))


# ---------------------------------------------------------------------------
# G1 单请求无竞争
# ---------------------------------------------------------------------------

class G1SingleRequestNoContention(unittest.TestCase):
    """单请求：P50=P99=E2E；violation 手算；单实例不均衡=0；无 KV 事件。"""

    def setUp(self):
        self.run_dir = synthetic.make_run_dir("g1")
        # arrival=1000 prefill [2000,4000) decode [4000,8000) completion=8000
        self.e2e = 7000
        synthetic.write_request_metrics(self.run_dir, [
            synthetic.request_row(
                request_id="s0_r0", session_id="s0",
                arrival_ns="1000", prefill_start_ns="2000",
                prefill_end_ns="4000", decode_start_ns="4000",
                completion_ns="8000", queue_ns="1000", prefill_ns="2000",
                prefill_decode_gap_ns="0", decode_ns="4000",
                e2e_ns=str(self.e2e), prefill_length="500",
                decode_length="100"),
        ])
        self.manifest = synthetic.write_slo_manifest(self.run_dir, {
            "alpha_main": 2.0,
            "bucket_percentiles": synthetic.TEST_BUCKET_EDGES,
        })
        self.t_iso = synthetic.write_t_isolated(self.run_dir, [(0, 0, 3000)])
        synthetic.write_cpp_log(self.run_dir, [], repo_variant="astra-sim-face")
        synthetic.write_metrics_manifest(
            self.run_dir, [synthetic.manifest_request("s0_r0", "s0", 0, 0)])

    def test_e2e_stats(self):
        out = self.run_dir / "e2e.csv"
        args = slo_argv(["e2e-stats", str(self.run_dir), "-o", str(out)])
        self.assertEqual(slo_stats.cmd_e2e_stats(args), 0)
        header, rows = synthetic.read_csv(out)
        by_metric = {r["metric"]: r for r in rows}
        self.assertEqual(by_metric["e2e_p50"]["value_ns"], "7000")
        self.assertEqual(by_metric["e2e_p99"]["value_ns"], "7000")
        self.assertEqual(by_metric["n_completed"]["value_ns"], "1")

    def test_violation_hand_value(self):
        # Deadline = 2.0 × 3000 = 6000 < E2E 7000 → 违约 1/1。
        out = self.run_dir / "v.json"
        args = slo_argv(["violation", str(self.run_dir), "--t-isolated",
                         str(self.t_iso), "--manifest", str(self.manifest),
                         "-o", str(out)])
        self.assertEqual(slo_stats.cmd_violation(args), 0)
        payload = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual(payload["violation_numerator"], 1)
        self.assertEqual(payload["violation_denominator"], 1)
        self.assertEqual(payload["violation_rate"], 1.0)

    def test_bucket_stats_slowdown(self):
        out = self.run_dir / "b.csv"
        args = slo_argv(["bucket-stats", str(self.run_dir), "--t-isolated",
                         str(self.t_iso), "--manifest", str(self.manifest),
                         "-o", str(out)])
        self.assertEqual(slo_stats.cmd_bucket_stats(args), 0)
        header, rows = synthetic.read_csv(out)
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["n_requests"], "1")
        self.assertEqual(row["t_isolated_ns"], "3000")
        self.assertAlmostEqual(float(row["p50_e2e_ns"]), 7000.0)
        self.assertAlmostEqual(float(row["median_slowdown"]),
                               7000 / 3000, places=5)

    def test_load_imbalance_single_instance(self):
        synthetic.write_jsonl(
            self.run_dir, "online_decision_log.jsonl", [
                {"kind": "prefill", "request_id": "s0_r0", "tick": 2000,
                 "decision": {}},
                {"kind": "decode", "request_id": "s0_r0", "tick": 4000,
                 "decision": {"decode_instance_index": 3}},
                {"kind": "completion", "request_id": "s0_r0", "tick": 8000,
                 "decision": {}},
            ])
        synthetic.write_jsonl(
            self.run_dir, "train_ledger.jsonl", [
                {"train_id": "t", "instance_index": 3, "tick": 8000,
                 "drains": [], "exits": ["s0_r0"], "joiners": []},
            ])
        manifest = synthetic.write_slo_manifest(
            self.run_dir, {"imbalance_bucket_ns": 1000})
        intervals = load_imbalance.collect_intervals(self.run_dir,
                                                     "astra-sim-face")
        self.assertEqual(intervals, [{
            "request_id": "s0_r0", "instance": 3,
            "admission_ns": 2000, "drain_ns": 8000}])
        self.assertEqual(
            load_imbalance.cmd_load_imbalance(
                li_args(self.run_dir, manifest)), 0)

    def test_kv_adapter_no_events(self):
        synthetic.write_jsonl(self.run_dir, "online_decision_log.jsonl", [
            {"kind": "prefill", "request_id": "s0_r0", "tick": 2000,
             "decision": {"history_action": "NO_HISTORY",
                          "history_transfer_bytes": 0,
                          "history_cache_state_before": "ABSENT"}},
        ])
        self.assertEqual(
            kv_cache_adapter.cmd_adapter(adapter_args(self.run_dir)), 0)
        header, rows = synthetic.read_csv(
            self.run_dir / "cache_events.csv")
        self.assertEqual(rows, [])
        _, hits = synthetic.read_csv(self.run_dir / "kv_hit_states.csv")
        self.assertEqual(hits[0]["kv_hit_state"], "no_history")

    def test_hopbytes_empty(self):
        synthetic.write_jsonl(self.run_dir, "online_decision_log.jsonl", [])
        self.assertEqual(
            hopbytes.cmd_hopbytes(hb_args(self.run_dir)), 0)
        _, rows = synthetic.read_csv(self.run_dir / "hb_total.csv")
        self.assertEqual(rows[0]["hop_bytes_total"], "0")
        self.assertEqual(rows[0]["coverage_bytes_ratio"], "0.000000")


# ---------------------------------------------------------------------------
# G2 双请求排队
# ---------------------------------------------------------------------------

class G2TwoRequestsQueued(unittest.TestCase):
    def setUp(self):
        self.run_dir = synthetic.make_run_dir("g2")
        # 同刻到达 100：r0 completion 500（E2E 400），r1 completion 900
        #（E2E 800，排队 400）。
        synthetic.write_request_metrics(self.run_dir, [
            synthetic.request_row(
                request_id="r0", queue_index="0", arrival_ns="100",
                prefill_start_ns="100", prefill_end_ns="200",
                decode_start_ns="200", completion_ns="500",
                queue_ns="0", prefill_ns="100",
                prefill_decode_gap_ns="0", decode_ns="300", e2e_ns="400"),
            synthetic.request_row(
                request_id="r1", queue_index="1", arrival_ns="100",
                prefill_start_ns="500", prefill_end_ns="600",
                decode_start_ns="600", completion_ns="900",
                queue_ns="400", prefill_ns="100",
                prefill_decode_gap_ns="0", decode_ns="300", e2e_ns="800"),
        ])
        synthetic.write_cpp_log(self.run_dir, [], repo_variant="astra-sim-face")
        synthetic.write_metrics_manifest(self.run_dir, [
            synthetic.manifest_request("r0", "s0", 0, 0),
            synthetic.manifest_request("r1", "s1", 0, 1),
        ])

    def test_backlog_series_hand(self):
        out = self.run_dir / "bl.csv"
        args = slo_argv(["backlog", str(self.run_dir), "-o", str(out)])
        self.assertEqual(slo_stats.cmd_backlog(args), 0)
        _, rows = synthetic.read_csv(out)
        self.assertEqual([(r["time_ns"], r["in_flight"]) for r in rows],
                         [("100", "2"), ("500", "1"), ("900", "0")])

    def test_e2e_ordering_and_percentiles(self):
        out = self.run_dir / "e2e.csv"
        args = slo_argv(["e2e-stats", str(self.run_dir),
                         "--extra-pct", "90", "-o", str(out)])
        self.assertEqual(slo_stats.cmd_e2e_stats(args), 0)
        _, rows = synthetic.read_csv(out)
        by_metric = {r["metric"]: r for r in rows}
        self.assertEqual(by_metric["e2e_p50"]["value_ns"], "400")
        self.assertEqual(by_metric["e2e_p99"]["value_ns"], "800")
        self.assertEqual(by_metric["e2e_p90"]["value_ns"], "800")

    def test_load_imbalance_same_instance_cv_zero(self):
        synthetic.write_jsonl(
            self.run_dir, "online_decision_log.jsonl", [
                {"kind": "prefill", "request_id": "r0", "tick": 100,
                 "decision": {}},
                {"kind": "decode", "request_id": "r0", "tick": 200,
                 "decision": {"decode_instance_index": 0}},
                {"kind": "prefill", "request_id": "r1", "tick": 100,
                 "decision": {}},
                {"kind": "decode", "request_id": "r1", "tick": 600,
                 "decision": {"decode_instance_index": 0}},
            ])
        synthetic.write_jsonl(
            self.run_dir, "train_ledger.jsonl", [
                {"train_id": "t0", "instance_index": 0, "tick": 500,
                 "drains": [], "exits": ["r0"], "joiners": []},
                {"train_id": "t1", "instance_index": 0, "tick": 900,
                 "drains": [], "exits": ["r1"], "joiners": []},
            ])
        manifest = synthetic.write_slo_manifest(
            self.run_dir, {"imbalance_bucket_ns": 100})
        intervals = load_imbalance.collect_intervals(self.run_dir,
                                                     "astra-sim-face")
        self.assertEqual(len(intervals), 2)
        stats = load_imbalance.instance_timeavg_backlog(intervals, 100,
                                                        100, 900)
        # 两请求同实例：b̄_0 = (400+800)/800 = 1.5；单实例 → CV=0。
        self.assertAlmostEqual(stats[0][0], 1.5)
        values = [stats[i][0] for i in sorted(stats)]
        mean = sum(values) / len(values)
        cv = math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))
        self.assertAlmostEqual(cv, 0.0)


# ---------------------------------------------------------------------------
# G3 session 两轮
# ---------------------------------------------------------------------------

class G3SessionTwoTurns(unittest.TestCase):
    def setUp(self):
        self.run_dir = synthetic.make_run_dir("g3")
        # turn0: arrival 0 → completion 1000；turn1: arrival 5000 →
        # completion 7000；human 1000 + tool 500。
        # T_session = 7000 − 0 − 1500 = 5500。
        synthetic.write_request_metrics(self.run_dir, [
            synthetic.request_row(
                request_id="s0_r0", session_id="s0", turn_index="0",
                queue_index="0", arrival_ns="0", completion_ns="1000",
                e2e_ns="1000"),
            synthetic.request_row(
                request_id="s0_r1", session_id="s0", turn_index="1",
                queue_index="1", arrival_ns="5000", completion_ns="7000",
                e2e_ns="2000"),
        ])
        synthetic.write_cpp_log(self.run_dir, [], repo_variant="astra-sim-face")

    def test_t_session_hand_value(self):
        synthetic.write_metrics_manifest(self.run_dir, [
            synthetic.manifest_request("s0_r0", "s0", 0, 0,
                                       human_time_ns=1000),
            synthetic.manifest_request("s0_r1", "s0", 1, 1,
                                       tool_time_ns=500),
        ])
        out = self.run_dir / "sess.csv"
        args = slo_argv(["session", str(self.run_dir), "-o", str(out)])
        self.assertEqual(slo_stats.cmd_session(args), 0)
        header, rows = synthetic.read_csv(out)
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["session_id"], "s0")
        self.assertEqual(row["n_turns"], "2")
        self.assertEqual(row["first_arrival_ns"], "0")
        self.assertEqual(row["last_completion_ns"], "7000")
        self.assertEqual(row["sum_human_tool_ns"], "1500")
        self.assertEqual(row["t_session_ns"], "5500")

    def test_missing_transparent_fields_na_and_counted(self):
        # manifest 缺 human/tool 透传字段（WP2 之前的状态）→ NA 行+计数。
        synthetic.write_metrics_manifest(self.run_dir, [
            synthetic.manifest_request("s0_r0", "s0", 0, 0),
            synthetic.manifest_request("s0_r1", "s0", 1, 1),
        ])
        out = self.run_dir / "sess.csv"
        args = slo_argv(["session", str(self.run_dir), "-o", str(out)])
        self.assertEqual(slo_stats.cmd_session(args), 0)
        _, rows = synthetic.read_csv(out)
        row = rows[0]
        self.assertEqual(row["t_session_ns"], "NA")
        self.assertEqual(row["missing_field_count"], "4")
        self.assertIn("s0_r0:human_time_ns", row["missing_fields_sample"])


# ---------------------------------------------------------------------------
# G4 restore 三段
# ---------------------------------------------------------------------------

class G4RestoreThreeSegments(unittest.TestCase):
    def setUp(self):
        self.run_dir = synthetic.make_run_dir("g4")
        # 手算场景（S1 式 restore 在准入批发射、prefill 主体后置）：
        #   request q0（有 history restore）：
        #     anchors(min=500, max=2500) prefill [2000, 4000)
        #     pre = min(2500,2000)−500      = 1500
        #     hidden = min(2500,4000)−2000  = 500
        #     exposed = max(0,2500−4000)    = 0
        #     ratio = 500/2000              = 0.25
        #   request q1（无锚点 → 四字段 NA）：
        #   request q2（锚点横跨 prefill 全窗）：
        #     anchors(1000, 6000) prefill [2000, 4000)
        #     pre = 1000；hidden = 2000；exposed = 2000；ratio = 0.4
        records = [
            {"type": "request", "queue_index": 0, "request_id": "q0",
             "prefill_start_ns": 2000, "prefill_end_ns": 4000},
            {"type": "request", "queue_index": 1, "request_id": "q1",
             "prefill_start_ns": 2000, "prefill_end_ns": 4000},
            {"type": "request", "queue_index": 2, "request_id": "q2",
             "prefill_start_ns": 2000, "prefill_end_ns": 4000},
            {"type": "memory_anchor", "subject_id": 0, "rank": 20,
             "node_id": 11, "tick_ns": 500},
            {"type": "memory_anchor", "subject_id": 0, "rank": 21,
             "node_id": 12, "tick_ns": 2500},
            {"type": "memory_anchor", "subject_id": 2, "rank": 20,
             "node_id": 13, "tick_ns": 1000},
            {"type": "memory_anchor", "subject_id": 2, "rank": 21,
             "node_id": 14, "tick_ns": 6000},
        ]
        synthetic.write_cpp_log(self.run_dir, records,
                                repo_variant="astra-sim-sh_1.0")

    def test_restore_decomposition_hand_values(self):
        out = self.run_dir / "restore.csv"
        import argparse
        ns = argparse.Namespace(run_dir=self.run_dir, output=str(out),
                                json="")
        self.assertEqual(
            restore_decomposition.cmd_restore(ns), 0)
        header, rows = synthetic.read_csv(out)
        self.assertEqual(
            header, ["queue_index", "request_id", "restore_start_ns",
                     "restore_complete_ns", "pre_prefill_restore_ns",
                     "hidden_restore_ns", "exposed_restore_stall_ns",
                     "hidden_ratio"])
        by_qi = {r["queue_index"]: r for r in rows}
        q0 = by_qi["0"]
        self.assertEqual(q0["restore_start_ns"], "500")
        self.assertEqual(q0["restore_complete_ns"], "2500")
        self.assertEqual(q0["pre_prefill_restore_ns"], "1500")
        self.assertEqual(q0["hidden_restore_ns"], "500")
        self.assertEqual(q0["exposed_restore_stall_ns"], "0")
        self.assertAlmostEqual(float(q0["hidden_ratio"]), 0.25)
        q1 = by_qi["1"]
        for column in ("restore_start_ns", "restore_complete_ns",
                       "pre_prefill_restore_ns", "hidden_restore_ns",
                       "exposed_restore_stall_ns", "hidden_ratio"):
            self.assertEqual(q1[column], "NA",
                             f"无锚点请求的 {column} 必须 NA")
        q2 = by_qi["2"]
        self.assertEqual(q2["pre_prefill_restore_ns"], "1000")
        self.assertEqual(q2["hidden_restore_ns"], "2000")
        self.assertEqual(q2["exposed_restore_stall_ns"], "2000")
        self.assertAlmostEqual(float(q2["hidden_ratio"]), 0.4)

    def test_summary_counts(self):
        rows, summary = restore_decomposition.collect(self.run_dir)
        self.assertEqual(summary["n_requests"], 3)
        self.assertEqual(summary["n_with_anchors"], 2)
        self.assertEqual(summary["n_no_anchor_na"], 1)

    def test_summary_detail_log_rejected(self):
        # summary 档 cpp.log（无 type=request 行）→ fail-closed。
        run_dir = synthetic.make_run_dir("g4b")
        run_dir.joinpath("cpp.log").write_text(
            "[METRIC] " + json.dumps(
                {"type": "summary", "p50_e2e_ns": 1}) + "\n",
            encoding="utf-8")
        with self.assertRaises(SloToolError):
            restore_decomposition.collect(run_dir)


# ---------------------------------------------------------------------------
# scan-export / normalized 骨架
# ---------------------------------------------------------------------------

class ScanExportSkeletonTests(unittest.TestCase):
    def test_drain_assertion_fails_open_point(self):
        run_dir = synthetic.make_run_dir("sc1")
        synthetic.write_cpp_log(run_dir, [], repo_variant="astra-sim-face")
        synthetic.write_request_metrics(run_dir, [
            synthetic.request_row(request_id="r0", queue_index="0",
                                  arrival_ns="0", completion_ns="10",
                                  e2e_ns="10"),
            synthetic.request_row(request_id="r1", queue_index="1",
                                  terminal_status="dropped"),
        ])
        args = slo_argv(["scan-export", "--point", f"{run_dir}=1.0",
                         "-o", "-"])
        with self.assertRaises(SloToolError) as ctx:
            slo_stats.cmd_scan_export(args)
        self.assertIn("drain", str(ctx.exception))

    def test_drained_point_emits_row(self):
        run_dir = synthetic.make_run_dir("sc2")
        synthetic.write_cpp_log(run_dir, [], repo_variant="astra-sim-face")
        synthetic.write_request_metrics(run_dir, [
            synthetic.request_row(request_id="r0", queue_index="0",
                                  arrival_ns="100",
                                  completion_ns="1100", e2e_ns="1000"),
        ])
        out = run_dir / "scan.csv"
        args = slo_argv(["scan-export", "--point", f"{run_dir}=2.0",
                         "-o", str(out)])
        self.assertEqual(slo_stats.cmd_scan_export(args), 0)
        header, rows = synthetic.read_csv(out)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["lambda_scale"], "2.000000")
        self.assertEqual(rows[0]["drain_ok_input_eq_completed"], "true")
        self.assertAlmostEqual(float(rows[0]["tput_rps"]), 1e9 / 1000)

    def test_normalized_group_max(self):
        run_dirs = []
        for tag, e2e in (("a", "1000"), ("b", "4000")):
            run_dir = synthetic.make_run_dir(f"nm_{tag}")
            synthetic.write_request_metrics(run_dir, [
                synthetic.request_row(request_id="r0", queue_index="0",
                                      arrival_ns="0",
                                      completion_ns=e2e, e2e_ns=e2e),
            ])
            synthetic.write_cpp_log(run_dir, [],
                                    repo_variant="astra-sim-face")
            run_dirs.append(run_dir)
        # 输出落独立临时目录：run_dirs 的 /tmp 根为多仓并行测试共享，
        # 固定名 norm.csv 会竞态互踩（mkdtemp 隔离，任务2修复）。
        out = synthetic.make_run_dir("nm_out") / "norm.csv"
        args = slo_argv(["normalized", *[str(d) for d in run_dirs],
                         "-o", str(out)])
        self.assertEqual(slo_stats.cmd_normalized(args), 0)
        header, rows = synthetic.read_csv(out)
        by_run = {Path(r["run_dir"]).name: r for r in rows}
        name_a = run_dirs[0].name
        name_b = run_dirs[1].name
        self.assertAlmostEqual(float(by_run[name_a]["normalized_time_p99"]),
                               0.25)
        self.assertAlmostEqual(float(by_run[name_b]["normalized_time_p99"]),
                               1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
