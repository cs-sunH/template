#!/usr/bin/env python3
"""T0 契约测试：CSV 列序/schema 常量、NA 语义、fail-closed 行为、公式单测。

运行：python3 sh_test_mesh/slo_tools/tests/test_slo_contract.py
  或  python3 -m unittest discover -s sh_test_mesh/slo_tools/tests -v
"""

from __future__ import annotations

import csv
import json
import math
import subprocess
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
import slo_common  # noqa: E402
from slo_common import (  # noqa: E402
    REQUEST_METRICS_COLUMNS, SloToolError, assert_no_proxy_columns,
    bucket_index, default_manifest_path, nearest_rank_percentile,
    parse_ns, read_request_metrics, require_bucket_edges, require_param,
)
import slo_stats  # noqa: E402
import restore_decomposition  # noqa: E402
import load_imbalance  # noqa: E402
import kv_cache_adapter  # noqa: E402
import hopbytes  # noqa: E402


def slo_argv(argv: list[str]) -> object:
    return slo_stats.build_parser().parse_args(argv)


class FrozenSchemaTests(unittest.TestCase):
    def test_request_metrics_columns_frozen(self):
        expected = (
            "queue_index,request_id,session_id,turn_index,request_type,"
            "terminal_status,arrival_ns,prefill_start_ns,prefill_end_ns,"
            "decode_start_ns,first_token_ns,first_token_source,"
            "completion_ns,queue_ns,prefill_ns,prefill_decode_gap_ns,"
            "decode_ns,e2e_ns,kv_hit_state,restore_start_ns,"
            "restore_complete_ns,pre_prefill_restore_ns,hidden_restore_ns,"
            "exposed_restore_stall_ns,hidden_ratio,prefill_length,"
            "decode_length,prefix_len,instructions")
        self.assertEqual(",".join(REQUEST_METRICS_COLUMNS), expected)

    def test_cache_event_columns_frozen(self):
        self.assertEqual(
            ",".join(kv_cache_adapter.CACHE_EVENT_COLUMNS),
            "action_id,request_id,start_ns,end_ns,bytes,source,target,cause")
        self.assertEqual(
            ",".join(kv_cache_adapter.KV_HIT_STATE_COLUMNS),
            "request_id,turn_index,kv_hit_state,evidence")

    def test_restore_output_columns_match_execution_plan(self):
        self.assertEqual(
            ",".join(restore_decomposition.OUTPUT_COLUMNS),
            "queue_index,request_id,restore_start_ns,restore_complete_ns,"
            "pre_prefill_restore_ns,hidden_restore_ns,"
            "exposed_restore_stall_ns,hidden_ratio")

    def test_enums(self):
        self.assertEqual(
            sorted(slo_common.TERMINAL_STATUSES),
            ["completed", "dropped", "failed", "rejected", "timed_out"])
        self.assertEqual(
            sorted(slo_common.KV_HIT_STATES),
            ["full", "miss", "no_history", "not_supported", "partial"])
        self.assertEqual(
            sorted(slo_common.FIRST_TOKEN_SOURCES),
            ["NA", "exact", "train_interpolated"])


class NaSemanticsTests(unittest.TestCase):
    def test_parse_ns(self):
        self.assertIsNone(parse_ns("NA", "f", "w"))
        self.assertIsNone(parse_ns(" NA ", "f", "w"))
        self.assertEqual(parse_ns("123", "f", "w"), 123)
        self.assertEqual(parse_ns(123, "f", "w"), 123)
        with self.assertRaises(SloToolError):
            parse_ns("12.5", "f", "w")
        with self.assertRaises(SloToolError):
            parse_ns("abc", "f", "w")
        with self.assertRaises(SloToolError):
            parse_ns("-3", "f", "w")
        with self.assertRaises(SloToolError):
            parse_ns("", "f", "w")

    def test_missing_column_fails(self):
        run_dir = synthetic.make_run_dir("na1")
        bad = run_dir / "request_metrics.csv"
        bad.write_text("queue_index,request_id\n0,r0\n", encoding="utf-8")
        with self.assertRaises(SloToolError):
            read_request_metrics(bad)

    def test_missing_file_fails(self):
        run_dir = synthetic.make_run_dir("na2")
        with self.assertRaises(SloToolError):
            read_request_metrics(run_dir / "request_metrics.csv")

    def test_iter_jsonl_rejects_nan_infinity_literals(self):
        """O9：NaN/Infinity/-Infinity 字面量 fail-closed（默认 json.loads
        静默收 nan/inf 进统计，下游 float('nan') 中毒聚合）。"""
        run_dir = synthetic.make_run_dir("jsonlconst")
        path = run_dir / "records.jsonl"
        for literal in ("NaN", "Infinity", "-Infinity"):
            path.write_text('{"a": 1}\n{"b": %s}\n' % literal,
                            encoding="utf-8")
            with self.assertRaises(SloToolError):
                list(slo_common.iter_jsonl(path))

    def test_iter_jsonl_normal_rows_unchanged(self):
        run_dir = synthetic.make_run_dir("jsonlok")
        path = run_dir / "records.jsonl"
        path.write_text('{"a": 1}\n\n{"b": null, "c": 2.5, "d": "NA"}\n',
                        encoding="utf-8")
        self.assertEqual(list(slo_common.iter_jsonl(path)),
                         [{"a": 1}, {"b": None, "c": 2.5, "d": "NA"}])

    def test_fmt_ratio_non_finite_is_na(self):
        self.assertEqual(slo_common.fmt_ratio(float("nan")), "NA")
        self.assertEqual(slo_common.fmt_ratio(float("inf")), "NA")
        self.assertEqual(slo_common.fmt_ratio(float("-inf")), "NA")
        self.assertEqual(slo_common.fmt_ratio(None), "NA")
        self.assertEqual(slo_common.fmt_ratio(0.5), "0.500000")


class FailClosedTests(unittest.TestCase):
    def test_repo_manifest_params_filled_b4_frozen(self):
        manifest = json.loads(
            default_manifest_path().read_text(encoding="utf-8"))
        self.assertEqual(manifest["schema_version"], 1)
        for name, entry in manifest["params"].items():
            self.assertIsNotNone(
                entry["value"],
                f"生产 manifest 的 {name}.value 必须已推导"
                f"（B4b 填充后不得回退 null）")
            self.assertIsNotNone(
                entry.get("evidence"),
                f"生产 manifest 的 {name}.evidence 必须留证（B4b）")
            self.assertIsNotNone(
                entry.get("rationale"),
                f"生产 manifest 的 {name}.rationale 必须留理（B4b）")
            self.assertIn("derivation_program", entry)

    def test_require_param_null_fails_closed(self):
        base = json.loads(
            default_manifest_path().read_text(encoding="utf-8"))
        base["params"]["alpha_main"]["value"] = None  # 合成未推导态
        with self.assertRaises(SloToolError) as ctx:
            require_param(base, "alpha_main")
        self.assertIn("参数未推导", str(ctx.exception))

    def test_require_param_missing_entry(self):
        with self.assertRaises(SloToolError):
            require_param({"params": {}}, "alpha_main")

    def test_bucket_edges_null_fails(self):
        base = json.loads(
            default_manifest_path().read_text(encoding="utf-8"))
        base["params"]["bucket_percentiles"]["value"] = None  # 合成未推导态
        with self.assertRaises(SloToolError):
            require_bucket_edges(base)

    def test_bucket_edges_validation(self):
        manifest = {"params": {"bucket_percentiles": {"value": {
            "percentiles": [50],
            "prefill_edges_tokens": [0, 1000, 999],
            "decode_edges_tokens": [0, 200, 400],
        }}}}
        with self.assertRaises(SloToolError):  # 非递增
            require_bucket_edges(manifest)

    def test_bucket_index_semantics(self):
        # 2026-09-05 口径裁决：interior edges 为左桶闭上界（边界值归左桶），
        # 末桶无上限（x > edges[-1] 归末桶，不再报错）；x < edges[0] 仍报错。
        edges = [0, 100, 200]
        self.assertEqual(bucket_index(edges, 0), 0)
        self.assertEqual(bucket_index(edges, 99), 0)
        self.assertEqual(bucket_index(edges, 100), 0)  # interior 边界归左桶
        self.assertEqual(bucket_index(edges, 101), 1)
        self.assertEqual(bucket_index(edges, 200), 1)
        self.assertEqual(bucket_index(edges, 201), 1)  # 末桶吸收越界
        self.assertEqual(bucket_index(edges, 10**9), 1)
        with self.assertRaises(SloToolError):
            bucket_index(edges, -1)

    def test_violation_on_null_manifest_fails(self):
        run_dir = synthetic.make_run_dir("fc1")
        synthetic.write_request_metrics(
            run_dir, [synthetic.request_row(e2e_ns="1000",
                                            completion_ns="1100",
                                            arrival_ns="100")])
        t_iso = synthetic.write_t_isolated(run_dir, [(0, 0, 400)])
        args = slo_argv(["violation", str(run_dir),
                         "--t-isolated", str(t_iso),
                         "--manifest", str(default_manifest_path()),
                         "-o", "-"])
        with self.assertRaises(SloToolError):
            slo_stats.cmd_violation(args)

    def test_assert_no_proxy_columns(self):
        assert_no_proxy_columns(["e2e_ns", "terminal_status"])
        with self.assertRaises(SloToolError):
            assert_no_proxy_columns(["e2e_ns", "first_token_ns"])
        with self.assertRaises(SloToolError):
            assert_no_proxy_columns(["first_token_source"])

    def test_t_isolated_with_proxy_column_rejected(self):
        run_dir = synthetic.make_run_dir("fc2")
        manifest = synthetic.write_slo_manifest(run_dir, {
            "alpha_main": 2.0,
            "bucket_percentiles": synthetic.TEST_BUCKET_EDGES,
        })
        synthetic.write_request_metrics(
            run_dir, [synthetic.request_row(e2e_ns="1000",
                                            arrival_ns="100",
                                            completion_ns="1100")])
        bad = run_dir / "bad_t_iso.csv"
        bad.write_text("prefill_bucket_idx,decode_bucket_idx,t_isolated_ns,"
                       "first_token_ns\n0,0,400,999\n", encoding="utf-8")
        args = slo_argv(["violation", str(run_dir), "--t-isolated", str(bad),
                         "--manifest", str(manifest), "-o", "-"])
        with self.assertRaises(SloToolError) as ctx:
            slo_stats.cmd_violation(args)
        self.assertIn("proxy", str(ctx.exception))


class PercentileTests(unittest.TestCase):
    def test_nearest_rank_matches_cpp_collector(self):
        # index = ceil(p*N) - 1（MetricCollector.cc nearest_rank_percentile）
        self.assertEqual(nearest_rank_percentile(list(range(1, 101)), 0.50), 50)
        self.assertEqual(nearest_rank_percentile(list(range(1, 101)), 0.99), 99)
        self.assertEqual(nearest_rank_percentile(list(range(1, 11)), 0.50), 5)
        self.assertEqual(nearest_rank_percentile(list(range(1, 11)), 0.99), 10)
        self.assertEqual(nearest_rank_percentile([7], 0.99), 7)
        with self.assertRaises(SloToolError):
            nearest_rank_percentile([], 0.5)
        with self.assertRaises(SloToolError):
            nearest_rank_percentile([1, 2], 0.0)


class RestoreFormulaTests(unittest.TestCase):
    """§3.2 固定公式手算 fixture（A 类，逐字符实现）。"""

    def f(self, rs, rc, ps, pe):
        return restore_decomposition.restore_segments(rs, rc, ps, pe)

    def test_fully_before_prefill(self):
        # rs=100 rc=200 ps=300 pe=400
        pre, hidden, exposed, ratio = self.f(100, 200, 300, 400)
        self.assertEqual((pre, hidden, exposed), (100, 0, 0))
        self.assertEqual(ratio, 0.0)

    def test_overlapping_prefill_start(self):
        # min(350,300)-100=200；min(350,400)-max(100,300)=50；max(0,350-400)=0
        pre, hidden, exposed, ratio = self.f(100, 350, 300, 400)
        self.assertEqual((pre, hidden, exposed), (200, 50, 0))
        self.assertAlmostEqual(ratio, 50 / 250)

    def test_fully_after_prefill(self):
        pre, hidden, exposed, ratio = self.f(450, 500, 300, 400)
        self.assertEqual((pre, hidden, exposed), (0, 0, 100))
        self.assertEqual(ratio, 0.0)

    def test_spanning_whole_prefill(self):
        # pre=min(500,300)-100=200；hidden=min(500,400)-300=100；
        # exposed=500-400=100；ratio=100/400
        pre, hidden, exposed, ratio = self.f(100, 500, 300, 400)
        self.assertEqual((pre, hidden, exposed), (200, 100, 100))
        self.assertAlmostEqual(ratio, 0.25)

    def test_zero_duration_ratio_na(self):
        pre, hidden, exposed, ratio = self.f(300, 300, 100, 400)
        self.assertEqual((pre, hidden, exposed), (0, 0, 0))
        self.assertIsNone(ratio)

    def test_missing_inputs_all_na(self):
        self.assertEqual(self.f(None, 100, 100, 200), (None,) * 4)
        self.assertEqual(self.f(100, 200, None, 400), (None,) * 4)


class ViolationDenominatorTests(unittest.TestCase):
    def _setup(self):
        run_dir = synthetic.make_run_dir("vio")
        manifest = synthetic.write_slo_manifest(run_dir, {
            "alpha_main": 2.0,
            "bucket_percentiles": synthetic.TEST_BUCKET_EDGES,
        })
        # prefill_length=500→bucket0, decode_length=100→bucket0；T_iso=400
        # → Deadline=800。
        rows = [
            synthetic.request_row(
                queue_index="0", request_id="r0", e2e_ns="900",
                arrival_ns="0", completion_ns="900"),      # 900 > 800 违约
            synthetic.request_row(
                queue_index="1", request_id="r1", e2e_ns="700",
                arrival_ns="0", completion_ns="700"),      # 未违约
            synthetic.request_row(
                queue_index="2", request_id="r2",
                terminal_status="rejected", e2e_ns="NA"),  # 计分母不计分子
            synthetic.request_row(
                queue_index="3", request_id="r3",
                terminal_status="dropped", e2e_ns="NA"),
        ]
        synthetic.write_request_metrics(run_dir, rows)
        t_iso = synthetic.write_t_isolated(run_dir, [(0, 0, 400)])
        return run_dir, manifest, t_iso

    def test_denominator_includes_all_terminal_states(self):
        run_dir, manifest, t_iso = self._setup()
        out = run_dir / "violation.json"
        args = slo_argv(["violation", str(run_dir), "--t-isolated",
                         str(t_iso), "--manifest", str(manifest),
                         "-o", str(out)])
        self.assertEqual(slo_stats.cmd_violation(args), 0)
        payload = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual(payload["violation_numerator"], 1)
        self.assertEqual(payload["violation_denominator"], 4)
        self.assertAlmostEqual(payload["violation_rate"], 0.25)
        self.assertEqual(payload["per_terminal_status"]["rejected"],
                         {"total": 1, "violating": 0})

    def test_observation_window_keeps_unfinished_and_unreleased_in_denominator(self):
        run_dir, manifest, t_iso = self._setup()
        synthetic.write_request_metrics(run_dir, [
            synthetic.request_row(queue_index="0", request_id="r0", e2e_ns="900",
                                  arrival_ns="0", completion_ns="900"),
            synthetic.request_row(queue_index="1", request_id="r1",
                                  terminal_status="not_arrived", e2e_ns="NA"),
            synthetic.request_row(queue_index="2", request_id="r2",
                                  terminal_status="in_flight", e2e_ns="NA"),
        ])
        out = run_dir / "window_violation.json"
        args = slo_argv(["violation", str(run_dir), "--t-isolated", str(t_iso),
                         "--manifest", str(manifest), "-o", str(out)])
        self.assertEqual(slo_stats.cmd_violation(args), 0)
        payload = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual(payload["violation_denominator"], 3)
        self.assertEqual(payload["not_arrived_requests"], 1)
        self.assertEqual(payload["in_flight_requests"], 1)
        self.assertAlmostEqual(payload["completion_rate"], 1 / 3)

    def test_missing_bucket_in_t_isolated_fails_closed(self):
        run_dir, manifest, _ = self._setup()
        t_iso = synthetic.write_t_isolated(run_dir, [(1, 1, 400)])
        args = slo_argv(["violation", str(run_dir), "--t-isolated",
                         str(t_iso), "--manifest", str(manifest),
                         "-o", "-"])
        with self.assertRaises(SloToolError):
            slo_stats.cmd_violation(args)

    def test_train_interpolated_proxy_cannot_enter_judgment(self):
        """B4/WP9 fallback 专项：request_metrics 携带 train_interpolated
        proxy 值时，violation 判定只消费 E2E——proxy 值无论多么"违约"
        都不得改变 verdict；判定输入侧的 proxy 列仍被 fail-closed 拒绝。
        """
        run_dir, manifest, t_iso = self._setup()
        # 同一 E2E 数据，附上夸张的 proxy first_token（若判定误读 proxy，
        # 这些值全部远超 Deadline=800，verdict 必然翻转）。十进制整数字
        # 面量不带下划线（R8-F4，2026-09-29 face 移植：parse_ns 拒绝
        # "1_000" 型字面量）。
        rows = [
            synthetic.request_row(
                queue_index="0", request_id="r0", e2e_ns="900",
                arrival_ns="0", completion_ns="900",
                first_token_ns="10000000000",
                first_token_source="train_interpolated"),
            synthetic.request_row(
                queue_index="1", request_id="r1", e2e_ns="700",
                arrival_ns="0", completion_ns="700",
                first_token_ns="999999999999",
                first_token_source="train_interpolated"),
            synthetic.request_row(
                queue_index="2", request_id="r2",
                terminal_status="rejected", e2e_ns="NA"),
            synthetic.request_row(
                queue_index="3", request_id="r3",
                terminal_status="dropped", e2e_ns="NA"),
        ]
        synthetic.write_request_metrics(run_dir, rows)
        out = run_dir / "violation_proxy.json"
        args = slo_argv(["violation", str(run_dir), "--t-isolated",
                         str(t_iso), "--manifest", str(manifest),
                         "-o", str(out)])
        self.assertEqual(slo_stats.cmd_violation(args), 0)
        payload = json.loads(out.read_text(encoding="utf-8"))
        # 与无 proxy 列的同一数据（test_denominator_includes_all_terminal_
        # states）完全一致：proxy 未进分子/分母。
        self.assertEqual(payload["violation_numerator"], 1)
        self.assertEqual(payload["violation_denominator"], 4)
        self.assertAlmostEqual(payload["violation_rate"], 0.25)
        # 判定列集自证 + 输入侧拒绝（first_token_source 同属禁用列）。
        assert_no_proxy_columns(
            ("e2e_ns", "prefill_length", "decode_length", "terminal_status"),
            "violation 判定列集")
        with self.assertRaises(SloToolError):
            assert_no_proxy_columns(
                ("e2e_ns", "first_token_source"), "violation 判定列集")
        bad = run_dir / "bad_t_iso_source.csv"
        bad.write_text("prefill_bucket_idx,decode_bucket_idx,t_isolated_ns,"
                       "first_token_source\n0,0,400,train_interpolated\n",
                       encoding="utf-8")
        args = slo_argv(["violation", str(run_dir), "--t-isolated",
                         str(bad), "--manifest", str(manifest), "-o", "-"])
        with self.assertRaises(SloToolError) as ctx:
            slo_stats.cmd_violation(args)
        self.assertIn("proxy", str(ctx.exception))


class KvHitStateMappingTests(unittest.TestCase):
    """逐仓 native→canonical 映射（no_history 不计入 miss）。"""

    def test_face_wscllm(self):
        f = kv_cache_adapter._hit_state_face_wscllm
        self.assertEqual(f({"history_action": "NO_HISTORY"}, 0, "r")[0],
                         "no_history")
        self.assertEqual(f({"history_action": "LOCAL_HIT"}, 1, "r")[0], "full")
        self.assertEqual(f({"history_action": "NOC_MIGRATE"}, 1, "r")[0],
                         "full")
        self.assertEqual(f({"history_action": "RECOMPUTE"}, 1, "r")[0], "miss")
        self.assertEqual(f({"history_action": "WEIRD"}, 1, "r")[0],
                         "not_supported")

    def test_sh10(self):
        f = kv_cache_adapter._hit_state_sh10
        self.assertEqual(f({"history_transfer": None}, 0, "r")[0],
                         "no_history")
        self.assertEqual(f({"history_transfer": None}, 2, "r")[0],
                         "not_supported")
        for kind in ("local_hit", "noc_migrate", "remote_load"):
            self.assertEqual(
                f({"history_transfer": {"kind": kind, "reason": "x"}}, 1,
                  "r")[0], "full")
        self.assertEqual(
            f({"history_transfer": {"kind": "weird"}}, 1, "r")[0],
            "not_supported")

    def test_sh20(self):
        f = kv_cache_adapter._hit_state_sh20
        self.assertEqual(f({}, 0, "r")[0], "no_history")
        self.assertEqual(f({}, 3, "r")[0], "not_supported")
        self.assertEqual(f({}, None, "r")[0], "not_supported")

    def test_sh30(self):
        f = kv_cache_adapter._hit_state_sh30
        cases = {
            "first_request_non_edge": "no_history",
            "first_request_edge_fallback": "no_history",
            "resident_local_hbm": "full",
            "resident_prefix_layers": "partial",
            "remote_edge_load_balance": "full",
        }
        for reason, expected in cases.items():
            self.assertEqual(
                f({"prefill_affinity_reason": reason}, 1, "r")[0], expected)
        self.assertEqual(f({"prefill_affinity_reason": "?"}, 1, "r")[0],
                         "not_supported")

    def test_joint(self):
        # 词表键 = joint_action 裸值（ALL_ACTIONS 二动作），非
        # prefill_affinity_reason 的 joint_ 前缀串——2026-10-18 键口径
        # 修正的回归锚（此前前缀错配使带历史请求恒 not_supported）。
        f = kv_cache_adapter._hit_state_joint
        self.assertEqual(
            f({"joint_action": "stay", "history_tokens_before": 586,
               "prefill_affinity_reason": "joint_stay"}, 1, "r")[0], "full")
        self.assertEqual(
            f({"joint_action": "copy", "history_tokens_before": 586,
               "prefill_affinity_reason": "joint_copy"}, 1, "r")[0], "full")
        self.assertEqual(
            f({"joint_action": "stay", "history_tokens_before": 0,
               "prefill_affinity_reason": "joint_stay"}, 0, "r")[0],
            "no_history")
        self.assertEqual(
            f({"joint_action": "recompute", "history_tokens_before": 586,
               "prefill_affinity_reason": "joint_recompute"}, 1, "r")[0],
            "not_supported")
        self.assertEqual(f({}, 1, "r")[0], "not_supported")

    def test_no_history_never_counted_as_miss(self):
        run_dir = synthetic.make_run_dir("kv1")
        synthetic.write_cpp_log(run_dir, [], repo_variant="astra-sim-sh_3.0")
        records = [
            {"kind": "prefill", "request_id": "r0", "tick": 1,
             "decision": {"prefill_affinity_reason":
                          "first_request_non_edge"}},
            {"kind": "prefill", "request_id": "r1", "tick": 2,
             "decision": {"prefill_affinity_reason": "resident_local_hbm"}},
        ]
        synthetic.write_jsonl(run_dir, "online_decision_log.jsonl", records)
        synthetic.write_metrics_manifest(
            run_dir,
            [synthetic.manifest_request("r0", "s0", 0, 0),
             synthetic.manifest_request("r1", "s0", 1, 1)],
            repo_variant="astra-sim-sh_3.0")
        out = run_dir / "hits.csv"
        import argparse
        ns = argparse.Namespace(
            run_dir=run_dir, output=str(run_dir / "events.csv"),
            hit_states=str(out), json="", reconcile=False,
            repo_variant=None, request_manifest=None)
        self.assertEqual(kv_cache_adapter.cmd_adapter(ns), 0)
        header, rows = synthetic.read_csv(out)
        self.assertEqual(header,
                         ["request_id", "turn_index", "kv_hit_state",
                          "evidence"])
        states = {r["request_id"]: r["kv_hit_state"] for r in rows}
        self.assertEqual(states, {"r0": "no_history", "r1": "full"})


class LoadImbalanceHandTests(unittest.TestCase):
    def test_two_instances_hand_computed(self):
        run_dir = synthetic.make_run_dir("li1")
        synthetic.write_cpp_log(run_dir, [], repo_variant="astra-sim-face")
        decisions = [
            {"kind": "prefill", "request_id": "r0", "tick": 0,
             "decision": {}},
            {"kind": "decode", "request_id": "r0", "tick": 0,
             "decision": {"decode_instance_index": 0}},
            {"kind": "completion", "request_id": "r0", "tick": 100,
             "decision": {}},
            {"kind": "prefill", "request_id": "r1", "tick": 0,
             "decision": {}},
            {"kind": "decode", "request_id": "r1", "tick": 0,
             "decision": {"decode_instance_index": 0}},
            {"kind": "completion", "request_id": "r1", "tick": 100,
             "decision": {}},
            {"kind": "prefill", "request_id": "r2", "tick": 0,
             "decision": {}},
            {"kind": "decode", "request_id": "r2", "tick": 0,
             "decision": {"decode_instance_index": 1}},
            {"kind": "completion", "request_id": "r2", "tick": 100,
             "decision": {}},
        ]
        # O13（2026-09-23）：2026-09-04 口径修正 drain 源 drains→exits
        # （与 drains 时刻同语义迁移；形态对齐 test_golden_g1g4.py）。
        ledger = [
            {"train_id": "t0", "instance_index": 0, "tick": 100,
             "drains": [], "exits": ["r0", "r1"], "joiners": []},
            {"train_id": "t1", "instance_index": 1, "tick": 100,
             "drains": [], "exits": ["r2"], "joiners": []},
        ]
        synthetic.write_jsonl(run_dir, "online_decision_log.jsonl",
                              decisions)
        synthetic.write_jsonl(run_dir, "train_ledger.jsonl", ledger)
        manifest = synthetic.write_slo_manifest(
            run_dir, {"imbalance_bucket_ns": 50})
        import argparse
        ns = argparse.Namespace(
            run_dir=run_dir, manifest=manifest, output="-", json="",
            repo_variant=None)
        # 手算：instance0 区间 [0,100)×2 请求 → b̄=2；instance1 b̄=1；
        # mean=1.5；population std=0.5；CV=1/3；Max/Mean=4/3。
        intervals = load_imbalance.collect_intervals(run_dir,
                                                     "astra-sim-face")
        self.assertEqual(len(intervals), 3)
        stats = load_imbalance.instance_timeavg_backlog(
            intervals, 50, 0, 100)
        self.assertAlmostEqual(stats[0][0], 2.0)
        self.assertAlmostEqual(stats[1][0], 1.0)
        values = [stats[i][0] for i in sorted(stats)]
        mean = sum(values) / len(values)
        std = math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))
        self.assertAlmostEqual(std, 0.5)
        self.assertAlmostEqual(std / mean, 1 / 3)
        self.assertAlmostEqual(max(values) / mean, 4 / 3)

    def test_partial_overlap_interval_not_inflated(self):
        # L19（2026-09-29）：精确端点积分——[10,20) 在 span [0,100) 的
        # b̄ = 10/100 = 0.1（旧"重叠桶计数×桶长"把部分重叠的边缘桶整桶
        # 计入 → bucket=50 时虚增为 50/100=0.5，请求时长 ≪ 桶长时 CV
        # 退化为每实例请求数度量）；零长度区间 [5,5) 恒贡献 0。
        intervals = [
            {"request_id": "a", "instance": 0, "admission_ns": 10,
             "drain_ns": 20},
            {"request_id": "b", "instance": 1, "admission_ns": 0,
             "drain_ns": 100},
        ]
        stats = load_imbalance.instance_timeavg_backlog(
            intervals, 50, 0, 100)
        self.assertAlmostEqual(stats[0][0], 0.1)
        self.assertAlmostEqual(stats[1][0], 1.0)
        stats = load_imbalance.instance_timeavg_backlog(
            [{"request_id": "z", "instance": 0, "admission_ns": 5,
              "drain_ns": 5}], 50, 0, 100)
        self.assertAlmostEqual(stats[0][0], 0.0)

    def test_missing_manifest_param_fails(self):
        run_dir = synthetic.make_run_dir("li2")
        synthetic.write_cpp_log(run_dir, [], repo_variant="astra-sim-face")
        import argparse
        ns = argparse.Namespace(
            run_dir=run_dir, manifest=None, output="-", json="",
            repo_variant=None)
        with self.assertRaises(SloToolError):
            load_imbalance.cmd_load_imbalance(ns)


class HopbytesTests(unittest.TestCase):
    def test_shard_hops_hand_computed(self):
        # shard1: bytes=1000, path len4 → 3 hops → 3000
        # shard2: bytes=500, path len2 → 1 hop → 500；合计 3500。
        acc = {"actions_with_hops": 0, "hop_bytes_total": 0,
               "bytes_with_hops": 0, "bytes_without_hops": 0}
        per_request: dict = {}
        record = {"kind": "prefill", "request_id": "r0", "tick": 1,
                  "decision": {"history_transfer": {
                      "kind": "noc_migrate", "total_bytes": 1500,
                      "shards": [
                          {"bytes": 1000, "noc_path": [20, 14, 8, 2]},
                          {"bytes": 500, "noc_path": [21, 15]}]}}}
        hopbytes.collect_sh10(record, acc, per_request)
        self.assertEqual(acc["hop_bytes_total"], 3500)
        self.assertEqual(acc["bytes_with_hops"], 1500)
        self.assertEqual(acc["bytes_without_hops"], 0)
        self.assertEqual(acc["actions_with_hops"], 2)
        self.assertEqual(per_request["r0"]["hop_bytes"], 3500)

    def test_shard_hops_prefers_explicit_field(self):
        self.assertEqual(hopbytes._shard_hops({"noc_hops": 5}), 5)
        self.assertEqual(hopbytes._shard_hops(
            {"noc_path": [1, 2, 3]}), 2)
        self.assertIsNone(hopbytes._shard_hops({"bytes": 1}))

    def test_sh30_consumes_new_shards_field(self):
        """B4 升级：S3 决策日志 completion_evictions[].shards[].noc_hops
        （B3 60s 证据字段）进入 Hop-Bytes；无 shards 的聚合字段不计。"""
        acc = {"actions_with_hops": 0, "hop_bytes_total": 0,
               "bytes_with_hops": 0, "bytes_without_hops": 0}
        per_request: dict = {}
        record = {"kind": "completion", "request_id": "r0", "tick": 9,
                  "decision": {
                      "history_transfer_bytes": 4096,
                      "completion_evictions": [{
                          "kind": "remote_store", "total_bytes": 3000,
                          "shards": [
                              {"bytes": 1000, "noc_hops": 2,
                               "noc_path": [20, 19, 18]},
                              {"bytes": 500, "noc_hops": 2,
                               "noc_path": [21, 22, 23]},
                              {"bytes": 500, "noc_path": [26, 25]}],
                      }]}}
        hopbytes.collect_sh30(record, acc, per_request)
        # 1000*2 + 500*2 + 500*1 = 3500；noc_path 兜底推导（len-1）。
        self.assertEqual(acc["hop_bytes_total"], 3500)
        self.assertEqual(acc["bytes_with_hops"], 2000)
        self.assertEqual(acc["actions_with_hops"], 3)
        # 聚合 bytes 字段（无路由）不进本账（B4 语义，见 notes）。
        self.assertEqual(acc["bytes_without_hops"], 0)
        # 无 shards 的 shard 条目（bytes 无 hops）→ bytes_without_hops。
        record2 = {"kind": "completion", "request_id": "r1", "tick": 10,
                   "decision": {"completion_evictions": [{
                       "total_bytes": 700,
                       "shards": [{"bytes": 700}]}]}}
        hopbytes.collect_sh30(record2, acc, per_request)
        self.assertEqual(acc["bytes_without_hops"], 700)

    def test_wscllm_static_route_hand_computed(self):
        acc = {"actions_with_hops": 0, "hop_bytes_total": 0,
               "bytes_with_hops": 0, "bytes_without_hops": 0}
        per_request: dict = {}
        record = {"kind": "decode", "request_id": "r0", "tick": 1,
                  "decision": {
                      "prefill_decode_transfer": {"total_bytes": 100},
                      "static_route": {"hop_count": 2}}}
        hopbytes.collect_wscllm(record, acc, per_request)
        self.assertEqual(acc["hop_bytes_total"], 200)
        record2 = {"kind": "prefill", "request_id": "r0", "tick": 2,
                   "decision": {"history_transfer_bytes": 50}}
        hopbytes.collect_wscllm(record2, acc, per_request)
        self.assertEqual(acc["bytes_without_hops"], 50)


class FaceHopsUniformityTests(unittest.TestCase):
    """M18（2026-09-29）：collect_face 的 hops[0]×Σbytes 聚合前提 =
    per-shard hops 全等——非全等（非同构实例块形状）fail-closed 并列
    各 shard 值，不再静默取 hops[0] 失真（joint collect_face 的
    hops[0] 消费面 = prefill 分支；decode 为逐 shard 精确、无此前提）。"""

    def test_face_prefill_nonuniform_hops_fails_closed(self):
        acc = {"actions_with_hops": 0, "hop_bytes_total": 0,
               "bytes_with_hops": 0, "bytes_without_hops": 0}
        per_request: dict = {}
        record = {"kind": "prefill", "request_id": "r0", "tick": 1,
                  "decision": {"history_transfer_bytes": 600,
                               "history_noc_hops": [2, 3]}}
        with self.assertRaises(SloToolError) as ctx:
            hopbytes.collect_face(record, acc, per_request)
        self.assertIn("非全等", str(ctx.exception))
        self.assertIn("[2, 3]", str(ctx.exception))
        self.assertEqual(acc["hop_bytes_total"], 0)
        self.assertEqual(acc["bytes_with_hops"], 0)
        # 全等 hops 行为不变（对照）。
        record = {"kind": "prefill", "request_id": "r1", "tick": 2,
                  "decision": {"history_transfer_bytes": 600,
                               "history_noc_hops": [2, 2]}}
        hopbytes.collect_face(record, acc, per_request)
        self.assertEqual(acc["hop_bytes_total"], 1200)


class CliSurfaceTests(unittest.TestCase):
    """--help 可用 + 退出码纪律（fail-closed=2）。"""

    def test_help_for_every_script(self):
        scripts = ["slo_stats.py", "load_imbalance.py",
                   "restore_decomposition.py", "kv_cache_adapter.py",
                   "hopbytes.py"]
        for script in scripts:
            proc = subprocess.run(
                [sys.executable, str(SLO_TOOLS_DIR / script), "--help"],
                capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, f"{script} --help 失败")
        for sub in ["e2e-stats", "violation", "bucket-stats", "session",
                    "backlog", "warmup", "normalized", "scan-export"]:
            proc = subprocess.run(
                [sys.executable, str(SLO_TOOLS_DIR / "slo_stats.py"),
                 sub, "--help"], capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0,
                             f"slo_stats.py {sub} --help 失败")

    def test_missing_run_dir_exits_fail_closed(self):
        proc = subprocess.run(
            [sys.executable, str(SLO_TOOLS_DIR / "slo_stats.py"),
             "e2e-stats", "/nonexistent/run_dir"],
            capture_output=True, text=True)
        self.assertEqual(proc.returncode, slo_common.EXIT_FAIL_CLOSED)
        self.assertIn("FAIL-CLOSED", proc.stderr)

    def test_load_imbalance_null_param_exit_code(self):
        run_dir = synthetic.make_run_dir("cli1")
        synthetic.write_cpp_log(run_dir, [], repo_variant="astra-sim-face")
        null_manifest = synthetic.write_slo_manifest(
            run_dir, {"imbalance_bucket_ns": None})  # 合成未推导态
        proc = subprocess.run(
            [sys.executable, str(SLO_TOOLS_DIR / "load_imbalance.py"),
             str(run_dir), "--manifest", str(null_manifest)],
            capture_output=True, text=True)
        self.assertEqual(proc.returncode, slo_common.EXIT_FAIL_CLOSED)
        self.assertIn("参数未推导", proc.stderr)


class SessionPassthroughTests(unittest.TestCase):
    """B4（S3 异常③）+ H5（2026-09-29）：metrics_manifest 透传
    human_time_ns/tool_time_ns 后 T_session 可算；initiator 口径 = 行上
    等待是该轮之后的 gap（[completion(j), arrival(j+1)]），Σ 只累加除
    末轮外的 turns；非末轮 turn-0/0-gap 双侧 null=0 贡献（非缺数据）；
    键缺席（非末轮）才 NA。metrics_manifest 缺透传字段时从姊妹
    manifest.json / run_dir/manifest.json 按 request_id 补齐（H5②）。

    旧口径（等待记在到达方=触发 gap）钉住的全 turns 求和已废：
    sidecar 冻结口径是 initiator（derive_20_first_30_seconds.py "the gap
    after turn j is the human_time or tool_time recorded on turn j's own
    row"），全 turns 求和系统性多扣恰一个末轮等待（快响应+长思考场景
    T_session<0 → fail-closed 误中止）。
    """

    def _write_run(self, manifest_requests):
        run_dir = synthetic.make_run_dir("sess")
        synthetic.write_request_metrics(run_dir, [
            synthetic.request_row(
                queue_index="0", request_id="s0t0", session_id="s0",
                turn_index="0", arrival_ns="1000", completion_ns="5000"),
            synthetic.request_row(
                queue_index="1", request_id="s0t1", session_id="s0",
                turn_index="1", arrival_ns="5200", completion_ns="9000"),
        ])
        synthetic.write_metrics_manifest(run_dir, manifest_requests)
        return run_dir

    def test_t_session_with_passthrough_keys(self):
        # initiator 口径：turn0 human=300 计入（[5000,5200] 前的窗口内
        # 等待——手算 fixture 中 300 与 200 的物理间隙不必一致，只验
        # 求和口径）；turn1（末轮）human=200 在窗口外不计、不留痕。
        # Σ(human+tool) = 300；T_session = 9000 − 1000 − 300 = 7700。
        run_dir = self._write_run([
            synthetic.manifest_request(
                "s0t0", "s0", 0, 0, human_time_ns=300, tool_time_ns=None),
            synthetic.manifest_request(
                "s0t1", "s0", 1, 1, human_time_ns=200, tool_time_ns=None),
        ])
        out = run_dir / "slo_session.csv"
        args = slo_argv(["session", str(run_dir), "-o", str(out)])
        self.assertEqual(slo_stats.cmd_session(args), 0)
        with out.open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["t_session_ns"], "7700")
        self.assertEqual(rows[0]["sum_human_tool_ns"], "300")
        # 末轮（s0t1）的 human=200 不进求和也不留 soft_missing；非末轮
        # turn0 的 tool 缺侧按 0 计留痕。
        self.assertEqual(rows[0]["missing_field_count"], "1")
        self.assertIn("s0t0:tool_time_ns=0",
                      rows[0]["missing_fields_sample"])
        self.assertNotIn("s0t1", rows[0]["missing_fields_sample"])

    def test_t_session_na_when_keys_absent(self):
        # 键整体缺席（WP2 之前的 manifest）→ NA；只有非末轮 turn0 被
        # 核对（末轮字段与 T_session 无关）→ 计数 2（human+tool 各 1）。
        run_dir = self._write_run([
            synthetic.manifest_request("s0t0", "s0", 0, 0),
            synthetic.manifest_request("s0t1", "s0", 1, 1),
        ])
        out = run_dir / "slo_session.csv"
        args = slo_argv(["session", str(run_dir), "-o", str(out)])
        self.assertEqual(slo_stats.cmd_session(args), 0)
        with out.open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(rows[0]["t_session_ns"], "NA")
        self.assertEqual(rows[0]["missing_field_count"], "2")

    def test_t_session_supplemented_from_manifest_json(self):
        # H5②：face-LRU 物化器只把透传字段写进 manifest.json —— 
        # metrics_manifest.json 缺字段时从姊妹 manifest.json 按
        # request_id 补齐（只补缺键），t_session 不再恒 NA。
        run_dir = self._write_run([
            synthetic.manifest_request("s0t0", "s0", 0, 0),
            synthetic.manifest_request("s0t1", "s0", 1, 1),
        ])
        (run_dir / "manifest.json").write_text(json.dumps({
            "requests": [
                {"request_id": "s0t0", "human_time_ns": 300},
                {"request_id": "s0t1", "human_time_ns": 200},
            ]}), encoding="utf-8")
        out = run_dir / "slo_session.csv"
        args = slo_argv(["session", str(run_dir), "-o", str(out)])
        self.assertEqual(slo_stats.cmd_session(args), 0)
        with out.open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(rows[0]["t_session_ns"], "7700")
        self.assertEqual(rows[0]["sum_human_tool_ns"], "300")

    def test_t_session_single_turn_ignores_own_fields(self):
        # 单轮会话：唯一 turn 即末轮 → Σ=0，T_session=completion−arrival
        # （行上等待定义在窗口外，字段缺席不构成 NA）。
        run_dir = synthetic.make_run_dir("sess1")
        synthetic.write_request_metrics(run_dir, [
            synthetic.request_row(
                queue_index="0", request_id="s0t0", session_id="s0",
                turn_index="0", arrival_ns="1000", completion_ns="5000"),
        ])
        synthetic.write_metrics_manifest(run_dir, [
            synthetic.manifest_request("s0t0", "s0", 0, 0),
        ])
        out = run_dir / "slo_session.csv"
        args = slo_argv(["session", str(run_dir), "-o", str(out)])
        self.assertEqual(slo_stats.cmd_session(args), 0)
        with out.open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(rows[0]["t_session_ns"], "4000")
        self.assertEqual(rows[0]["sum_human_tool_ns"], "0")


class BacklogTests(unittest.TestCase):
    def test_backlog_reconstruction(self):
        run_dir = synthetic.make_run_dir("bl1")
        synthetic.write_request_metrics(run_dir, [
            synthetic.request_row(request_id="r0", queue_index="0",
                                  arrival_ns="100", completion_ns="500"),
            synthetic.request_row(request_id="r1", queue_index="1",
                                  arrival_ns="100", completion_ns="900"),
        ])
        out = run_dir / "backlog.csv"
        args = slo_argv(["backlog", str(run_dir), "-o", str(out)])
        self.assertEqual(slo_stats.cmd_backlog(args), 0)
        header, rows = synthetic.read_csv(out)
        self.assertEqual(header, ["time_ns", "in_flight"])
        self.assertEqual([(r["time_ns"], r["in_flight"]) for r in rows],
                         [("100", "2"), ("500", "1"), ("900", "0")])

    def test_backlog_bucket_fills_zero_event_buckets(self):
        # L17（2026-09-29）：--bucket-ns 分桶补发零事件桶。事件 100(+2)、
        # 500(−1)、900(−1)，bucket=200 → 桶 0:[0,200) 事件值 max=2；
        # 桶 1:[200,400) 零事件 → 上一事件延续 2；桶 2:[400,600) 只发生
        # 完成事件，真实峰值=桶起点携入值 2（C-4 批次 D 镜像 2026-09-30：
        # 桶值口径补齐「桶起点携入值」max 项——只对桶内事件取 max 会把
        # 该桶系统性低估为 1）；桶 3:[600,800) 零事件 → 延续 1；
        # 桶 4:[800,1000) 的完成事件在 t=900，[800,900) 段携入 1
        # → 桶峰值 1（事件后归 0）。无事件区段不再整桶丢失。
        run_dir = synthetic.make_run_dir("bl2")
        synthetic.write_request_metrics(run_dir, [
            synthetic.request_row(request_id="r0", queue_index="0",
                                  arrival_ns="100", completion_ns="500"),
            synthetic.request_row(request_id="r1", queue_index="1",
                                  arrival_ns="100", completion_ns="900"),
        ])
        out = run_dir / "backlog_b.csv"
        args = slo_argv(["backlog", str(run_dir), "--bucket-ns", "200",
                         "-o", str(out)])
        self.assertEqual(slo_stats.cmd_backlog(args), 0)
        header, rows = synthetic.read_csv(out)
        self.assertEqual(header, ["time_ns", "in_flight"])
        self.assertEqual([(r["time_ns"], r["in_flight"]) for r in rows],
                         [("0", "2"), ("200", "2"), ("400", "2"),
                          ("600", "1"), ("800", "1")])


class WarmupTests(unittest.TestCase):
    def test_warmup_judgment(self):
        run_dir = synthetic.make_run_dir("wu1")
        rows = []
        for i in range(1, 21):
            rows.append(synthetic.request_row(
                request_id=f"r{i}", queue_index=str(i),
                arrival_ns=str(i * 100),
                completion_ns=str(i * 100 + (5000 if i <= 2 else 1000)),
                e2e_ns=str(5000 if i <= 2 else 1000)))
        synthetic.write_request_metrics(run_dir, rows)
        manifest = synthetic.write_slo_manifest(run_dir, {
            "warmup_window": 250, "warmup_change_threshold": 0.05})
        out = run_dir / "warmup.json"
        args = slo_argv(["warmup", str(run_dir), "--manifest", str(manifest),
                         "-o", str(out)])
        self.assertEqual(slo_stats.cmd_warmup(args), 0)
        payload = json.loads(out.read_text(encoding="utf-8"))
        # 全样本 P99=5000（nearest-rank: ceil(.99*20)=20 → max）；
        # 剔除 arrival<250 的 2 行后全为 1000 → P99=1000。
        self.assertEqual(payload["p99_full_ns"], 5000)
        self.assertEqual(payload["p99_after_warmup_ns"], 1000)
        self.assertAlmostEqual(payload["relative_change"], 0.8)
        self.assertFalse(payload["warmup_stable"])



class CppLogFallbackTests(unittest.TestCase):
    """P1 契约：cpp.log → metrics.log → cpp.log.gz 统一回退。

    归档（archive_run_outputs.sh）后 run_dir 只剩 metrics.log（[METRIC]
    无损抽取）与 cpp.log.gz（全量原文）；三条路径读到的记录集必须同一。
    """

    RECORDS = [
        {"type": "request", "queue_index": 0, "request_id": "r0",
         "prefill_start_ns": 100, "prefill_end_ns": 200},
        {"type": "memory_anchor", "subject_id": 0, "rank": 1,
         "node_id": "n1", "tick_ns": 150},
    ]

    def _metric_lines(self, run_dir: Path) -> str:
        cpp_log = run_dir / "cpp.log"
        return "".join(line + "\n" for line in
                       cpp_log.read_text(encoding="utf-8").splitlines()
                       if line.startswith("[METRIC] "))

    def _read_all(self, run_dir: Path) -> list[dict]:
        return list(slo_common.read_cpp_metric_records(
            slo_common.resolve_cpp_metric_log(run_dir)))

    def test_plain_cpp_log(self):
        run_dir = synthetic.make_run_dir("fb0")
        synthetic.write_cpp_log(run_dir, self.RECORDS)
        records = self._read_all(run_dir)
        self.assertEqual([r.get("type") for r in records],
                         ["init", "request", "memory_anchor"])
        self.assertEqual(slo_common.read_init_record(run_dir)["repo_variant"],
                         "astra-sim-face")

    def test_cpp_metric_rejects_non_finite_json_numbers(self):
        run_dir = synthetic.make_run_dir("cpp_nonfinite")
        path = run_dir / "cpp.log"
        for literal in ("NaN", "Infinity", "-Infinity", "1e999", "-1e999"):
            path.write_text(
                '[METRIC] {"type":"request","value":%s}\n' % literal,
                encoding="utf-8")
            with self.subTest(literal=literal):
                with self.assertRaises(SloToolError):
                    list(slo_common.read_cpp_metric_records(path))

    def test_cpp_metric_keeps_finite_json_numbers(self):
        run_dir = synthetic.make_run_dir("cpp_finite")
        path = run_dir / "cpp.log"
        path.write_text(
            '[METRIC] {"i":1,"f":2.5,"exp":1e300,"null":null}\n',
            encoding="utf-8")
        self.assertEqual(
            list(slo_common.read_cpp_metric_records(path)),
            [{"i": 1, "f": 2.5, "exp": 1e300, "null": None}])

    def test_metrics_log_only_matches_cpp_log_records(self):
        plain = synthetic.make_run_dir("fb1a")
        synthetic.write_cpp_log(plain, self.RECORDS)
        archived = synthetic.make_run_dir("fb1b")
        (archived / "metrics.log").write_text(self._metric_lines(plain),
                                              encoding="utf-8")
        self.assertEqual(self._read_all(archived), self._read_all(plain))
        self.assertEqual(slo_common.read_init_record(archived),
                         slo_common.read_init_record(plain))

    def test_cpp_log_gz_only_matches_cpp_log_records(self):
        import gzip
        plain = synthetic.make_run_dir("fb2a")
        synthetic.write_cpp_log(plain, self.RECORDS)
        archived = synthetic.make_run_dir("fb2b")
        raw = plain.joinpath("cpp.log").read_text(encoding="utf-8")
        # 混入非 [METRIC] 行验证过滤不受压缩形态影响。
        raw = "[sim] some non-metric line\n" + raw + "[sim] tail\n"
        with gzip.open(archived / "cpp.log.gz", "wt",
                       encoding="utf-8") as handle:
            handle.write(raw)
        self.assertEqual(self._read_all(archived), self._read_all(plain))
        self.assertEqual(slo_common.read_init_record(archived),
                         slo_common.read_init_record(plain))

    def test_fallback_order_prefers_cpp_log_then_metrics_log(self):
        run_dir = synthetic.make_run_dir("fb3")
        synthetic.write_cpp_log(run_dir, self.RECORDS)
        (run_dir / "metrics.log").write_text(
            "[METRIC] {\"type\": \"init\", \"repo_variant\": \"decoy\"}\n",
            encoding="utf-8")
        self.assertEqual(slo_common.resolve_cpp_metric_log(run_dir).name,
                         "cpp.log")
        (run_dir / "cpp.log").unlink()
        self.assertEqual(slo_common.resolve_cpp_metric_log(run_dir).name,
                         "metrics.log")
        self.assertEqual(slo_common.read_init_record(run_dir)
                         ["repo_variant"], "decoy")

    def test_missing_all_three_fails_closed(self):
        run_dir = synthetic.make_run_dir("fb4")
        with self.assertRaises(SloToolError):
            slo_common.resolve_cpp_metric_log(run_dir)

    def test_restore_decomposition_entry_reads_archived_run_dir(self):
        plain = synthetic.make_run_dir("fb5a")
        synthetic.write_cpp_log(plain, self.RECORDS)
        archived = synthetic.make_run_dir("fb5b")
        (archived / "metrics.log").write_text(self._metric_lines(plain),
                                              encoding="utf-8")
        rows_plain, summary_plain = restore_decomposition.collect(plain)
        rows_archived, summary_archived = restore_decomposition.collect(
            archived)
        self.assertEqual((rows_plain, summary_plain),
                         (rows_archived, summary_archived))

if __name__ == "__main__":
    unittest.main(verbosity=2)
