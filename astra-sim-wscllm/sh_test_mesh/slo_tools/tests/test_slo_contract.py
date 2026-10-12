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
# pytest 的 prepend 导入（tests/ 带 __init__.py，模块名变 tests.test_*）只把
# slo_tools/ 放上 sys.path；本模块目录（tests/）须自行入径，`import synthetic`
# 才与直跑 / unittest discover 两条口径一致。
sys.path.insert(0, str(TESTS_DIR))

import synthetic  # noqa: E402
import slo_common  # noqa: E402
from slo_common import (  # noqa: E402
    REQUEST_METRICS_COLUMNS, SloToolError, assert_no_proxy_columns,
    bucket_index, default_manifest_path, load_json_strict,
    load_slo_manifest, nearest_rank_percentile,
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
            ["completed", "failed"])
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

    def test_bucket_edges_nan_fails_closed(self):
        # P1 修复（2026-09-28）钉死：NaN/Inf 边界在 NaN 上任何 "<=" 比较
        # 恒 False，非递增门会静默放行并让 bucket_index 错桶——有限性门
        # 必须先于递增门拒绝。基线两侧 edges 均合法递增，保证唯一违例
        # 就是被注入的非有限边界。
        for bad_value in (float("nan"), float("inf"), float("-inf")):
            for slot in ("prefill_edges_tokens", "decode_edges_tokens"):
                manifest = {"params": {"bucket_percentiles": {"value": {
                    "percentiles": [50],
                    "prefill_edges_tokens": [0, 1000, 2000],
                    "decode_edges_tokens": [0, 200, 400],
                }}}}
                manifest["params"]["bucket_percentiles"]["value"][slot] = \
                    [0, bad_value, 999]
                with self.assertRaises(SloToolError) as ctx:
                    require_bucket_edges(manifest)
                self.assertIn(slot, str(ctx.exception))
                self.assertIn("非有限数值", str(ctx.exception))

    def test_bucket_edges_nan_reaches_no_comparison(self):
        # 全 NaN 边界（递增门完全失明的形态）也必须被有限性门拒绝。
        manifest = {"params": {"bucket_percentiles": {"value": {
            "percentiles": [50],
            "prefill_edges_tokens": [float("nan")] * 3,
            "decode_edges_tokens": [0, 200, 400],
        }}}}
        with self.assertRaises(SloToolError):
            require_bucket_edges(manifest)

    def test_bucket_index_nonfinite_edges_fails_closed(self):
        # P1 修复（2026-09-28）钉死：即便绕过 require_bucket_edges 直调
        # bucket_index（合成输入），非有限边界/输入也不得静默错桶。
        edges = [1, float("nan"), 100]
        with self.assertRaises(SloToolError) as ctx:
            bucket_index(edges, 5)  # 修复前此处静默返回桶 1（应为桶 0）
        self.assertIn("非有限", str(ctx.exception))
        with self.assertRaises(SloToolError):
            bucket_index([1, float("inf"), 100], 5)
        with self.assertRaises(SloToolError):
            bucket_index([1, 50, 100], float("nan"))

    def test_load_slo_manifest_rejects_nan_literal(self):
        # P1 修复（2026-09-28）钉死：json.loads 默认接受 NaN 字面量；
        # manifest 类输入必须经 load_json_strict 在 parse 层拒绝。
        with self.assertRaises(SloToolError) as ctx:
            load_json_strict(
                '{"a": [1, NaN, 3]}', "synthetic.json")
        self.assertIn("NaN", str(ctx.exception))
        with self.assertRaises(SloToolError):
            load_json_strict('{"a": Infinity}', "synthetic.json")
        with self.assertRaises(SloToolError):
            load_json_strict('{"a": -Infinity}', "synthetic.json")
        # 合法数值不受影响。
        self.assertEqual(load_json_strict('{"a": [1, 2.5]}', "s.json"),
                         {"a": [1, 2.5]})

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
        # 合成 value=null 清单断言 null fail-closed（原版误传生产 manifest，
        # SloToolError 实由 T_isolated 缺桶触发——假绿，2026-09-27 修）。
        null_manifest = synthetic.write_slo_manifest(
            run_dir, {"alpha_main": None})
        args = slo_argv(["violation", str(run_dir),
                         "--t-isolated", str(t_iso),
                         "--manifest", str(null_manifest),
                         "-o", "-"])
        with self.assertRaises(SloToolError) as ctx:
            slo_stats.cmd_violation(args)
        self.assertIn("参数未推导", str(ctx.exception))

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
                terminal_status="failed", e2e_ns="NA"),    # 计分母不计分子
            synthetic.request_row(
                queue_index="3", request_id="r3",
                terminal_status="failed", e2e_ns="NA"),
        ]
        # A4-9（2026-09-26）：原 r2/r3 用 rejected/dropped 演示"非完成
        # 终态计分母不进分子"；词表收紧后 failed 承担同一角色（语义
        # 不变：E2E 为 NA 的终态行计入分母、不进分子）。
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
        self.assertEqual(payload["per_terminal_status"]["failed"],
                         {"total": 2, "violating": 0})

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
        # 这些值全部远超 Deadline=800，verdict 必然翻转）。
        rows = [
            synthetic.request_row(
                queue_index="0", request_id="r0", e2e_ns="900",
                arrival_ns="0", completion_ns="900",
                first_token_ns="10_000_000_000",
                first_token_source="train_interpolated"),
            synthetic.request_row(
                queue_index="1", request_id="r1", e2e_ns="700",
                arrival_ns="0", completion_ns="700",
                first_token_ns="999_999_999_999",
                first_token_source="train_interpolated"),
            synthetic.request_row(
                queue_index="2", request_id="r2",
                terminal_status="failed", e2e_ns="NA"),
            synthetic.request_row(
                queue_index="3", request_id="r3",
                terminal_status="failed", e2e_ns="NA"),
        ]
        # A4-9（2026-09-26）：原 rejected/dropped 行改为 failed（词表
        # 收紧后同角色非完成终态），本用例语义不变。
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
        # 2026-09-04 exits 口径：drains/exits 键齐备，终态归属在 exits
        # （load_imbalance.li_collect_drains 仅从 exits 归集）。
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
        # D2: collect_intervals 已删（仅测试引用的死包装）——直接组合
        # 生产原语（与 CLI/driver 同一循环体）。
        _state = load_imbalance.li_prepare()
        for _record in load_imbalance.iter_jsonl(
                run_dir / load_imbalance.DECISION_LOG_RELPATH):
            load_imbalance.li_consume_decision(
                _record, "astra-sim-face", _state)
        _drains, _skipped = load_imbalance.li_collect_drains(run_dir)
        intervals = load_imbalance.li_assemble_intervals(
            _state, _drains, _skipped)
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

    def test_missing_manifest_param_fails(self):
        run_dir = synthetic.make_run_dir("li2")
        synthetic.write_cpp_log(run_dir, [], repo_variant="astra-sim-face")
        # 合成 value=null 清单并显式传路径（原版 manifest=None 落到生产
        # 清单，SloToolError 实由缺 decision log 触发——假绿，2026-09-27 修）。
        null_manifest = synthetic.write_slo_manifest(
            run_dir, {"imbalance_bucket_ns": None})
        import argparse
        ns = argparse.Namespace(
            run_dir=run_dir, manifest=null_manifest, output="-", json="",
            repo_variant="astra-sim-face")
        with self.assertRaises(SloToolError) as ctx:
            load_imbalance.cmd_load_imbalance(ns)
        self.assertIn("参数未推导", str(ctx.exception))


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

    def test_wscllm_decode_shards_aggregate_per_shard(self):
        """问题 4b（2026-09-05）：decode 行 shards[] 携带 noc_hops/
        noc_path 时逐 shard bytes×hops（100B×2 + 50B×3 = 350），
        legacy 聚合口径（total_bytes×hop_count）不再叠加（防双计）。"""
        acc = {"actions_with_hops": 0, "hop_bytes_total": 0,
               "bytes_with_hops": 0, "bytes_without_hops": 0}
        per_request: dict = {}
        record = {"kind": "decode", "request_id": "r0", "tick": 1,
                  "decision": {
                      "static_route": {"hop_count": 7},
                      "prefill_decode_transfer": {
                          "total_bytes": 150,
                          "shards": [
                              {"relative_tp_rank": 0, "bytes": 100,
                               "noc_hops": 2, "noc_path": [0, 1, 5]},
                              {"relative_tp_rank": 1, "bytes": 50,
                               "noc_hops": 3, "noc_path": [2, 3, 6, 7]},
                          ]}}}
        hopbytes.collect_wscllm(record, acc, per_request)
        self.assertEqual(acc["hop_bytes_total"], 350)
        self.assertEqual(acc["actions_with_hops"], 2)
        self.assertEqual(acc["bytes_with_hops"], 150)
        self.assertEqual(acc["bytes_without_hops"], 0)
        self.assertEqual(per_request["r0"]["hop_bytes"], 350)

    def test_wscllm_decode_shards_without_routes_fall_back(self):
        """旧产物（shards 无 noc 字段）必须走 legacy 回退——0902 控制
        变量回归的读取端前提：total_bytes×static_route.hop_count。"""
        acc = {"actions_with_hops": 0, "hop_bytes_total": 0,
               "bytes_with_hops": 0, "bytes_without_hops": 0}
        per_request: dict = {}
        record = {"kind": "decode", "request_id": "r0", "tick": 1,
                  "decision": {
                      "static_route": {"hop_count": 2},
                      "prefill_decode_transfer": {
                          "total_bytes": 100,
                          "shards": [
                              {"relative_tp_rank": 0, "bytes": 60},
                              {"relative_tp_rank": 1, "bytes": 40},
                          ]}}}
        hopbytes.collect_wscllm(record, acc, per_request)
        self.assertEqual(acc["hop_bytes_total"], 200)
        self.assertEqual(acc["actions_with_hops"], 1)
        self.assertEqual(acc["bytes_with_hops"], 100)
        self.assertEqual(acc["bytes_without_hops"], 0)

    def test_wscllm_prefill_history_transfer_shards(self):
        """问题 4b：prefill 行 history_transfer_shards[] 逐 shard 聚合
        （100B×2 + 50B×3 = 350）；legacy 聚合字段不再叠加（防双计）；
        无路由 shard 计 bytes_without_hops（宁缺勿造）。"""
        acc = {"actions_with_hops": 0, "hop_bytes_total": 0,
               "bytes_with_hops": 0, "bytes_without_hops": 0}
        per_request: dict = {}
        record = {"kind": "prefill", "request_id": "r0", "tick": 1,
                  "decision": {
                      "history_transfer_bytes": 250,
                      "noc_hops": 9,
                      "history_transfer_shards": [
                          {"relative_tp_rank": 0, "source_rank": 0,
                           "target_rank": 5, "bytes": 100,
                           "noc_hops": 2, "noc_path": [0, 1, 5]},
                          {"relative_tp_rank": 1, "source_rank": 2,
                           "target_rank": 7, "bytes": 50,
                           "noc_path": [2, 3, 6, 7]},
                          {"relative_tp_rank": 2, "source_rank": 4,
                           "target_rank": 4, "bytes": 100},
                      ]}}
        hopbytes.collect_wscllm(record, acc, per_request)
        # 100*2 + 50*(len-1=3) = 350；第三 shard 无路由 → bytes_without。
        self.assertEqual(acc["hop_bytes_total"], 350)
        self.assertEqual(acc["actions_with_hops"], 2)
        self.assertEqual(acc["bytes_with_hops"], 150)
        self.assertEqual(acc["bytes_without_hops"], 100)
        self.assertEqual(per_request["r0"]["hop_bytes"], 350)
        self.assertEqual(per_request["r0"]["bytes_without"], 100)

    def test_face_decode_without_kv_noc_hops_falls_back(self):
        """P2-20（2026-09-28）：face decode 行 kv_noc_hops 缺席（旧产物/
        None）→ 整条 shard 字节按 bytes_without_hops 回退计入 coverage
        分母，不再蒸发（对齐同函数 prefill 分支与 docstring 回退承诺；
        旧 isinstance 门两分支都不过 → 整条传输字节从分母消失，
        coverage_bytes_ratio 虚高）。"""
        for absent_hops in (None, "missing"):
            acc = {"actions_with_hops": 0, "hop_bytes_total": 0,
                   "bytes_with_hops": 0, "bytes_without_hops": 0}
            per_request: dict = {}
            record = {"kind": "decode", "request_id": "r0", "tick": 1,
                      "decision": {"prefill_decode_transfer": {
                          "total_bytes": 150,
                          "shards": [{"bytes": 100}, {"bytes": 50}]}}}
            if absent_hops != "missing":
                record["decision"]["kv_noc_hops"] = absent_hops
            hopbytes.collect_face(record, acc, per_request)
            self.assertEqual(acc["hop_bytes_total"], 0)
            self.assertEqual(acc["actions_with_hops"], 0)
            self.assertEqual(acc["bytes_with_hops"], 0)
            self.assertEqual(acc["bytes_without_hops"], 150,
                             f"hops={absent_hops!r}: 分母回退失效")
            self.assertEqual(per_request["r0"]["bytes_without"], 150)
            self.assertEqual(per_request["r0"]["bytes_with"], 0)

    def test_face_decode_with_kv_noc_hops_per_shard(self):
        """face decode 正常路径（B2wp9py 观测字段）：逐 shard bytes×hops。"""
        acc = {"actions_with_hops": 0, "hop_bytes_total": 0,
               "bytes_with_hops": 0, "bytes_without_hops": 0}
        per_request: dict = {}
        record = {"kind": "decode", "request_id": "r0", "tick": 1,
                  "decision": {
                      "kv_noc_hops": [2, 3],
                      "prefill_decode_transfer": {
                          "total_bytes": 150,
                          "shards": [{"bytes": 100}, {"bytes": 50}]}}}
        hopbytes.collect_face(record, acc, per_request)
        self.assertEqual(acc["hop_bytes_total"], 100 * 2 + 50 * 3)
        self.assertEqual(acc["actions_with_hops"], 2)
        self.assertEqual(acc["bytes_with_hops"], 150)
        self.assertEqual(acc["bytes_without_hops"], 0)
        self.assertEqual(per_request["r0"]["hop_bytes"], 350)

    def test_face_prefill_without_hops_falls_back(self):
        """face prefill 行 history_noc_hops 缺席 → bytes_without（既有
        回退承诺的对照面，P2-20 修复不得波及）。"""
        acc = {"actions_with_hops": 0, "hop_bytes_total": 0,
               "bytes_with_hops": 0, "bytes_without_hops": 0}
        per_request: dict = {}
        record = {"kind": "prefill", "request_id": "r0", "tick": 1,
                  "decision": {"history_transfer_bytes": 80}}
        hopbytes.collect_face(record, acc, per_request)
        self.assertEqual(acc["bytes_without_hops"], 80)
        self.assertEqual(acc["bytes_with_hops"], 0)


class WscllmDecodeEvictionExtractionTests(unittest.TestCase):
    """问题 2A 收口（2026-09-27）：wscllm canonical 提取只消费 decode 行
    decode_target_evictions 全集；prefill 行子集不再产事件（防双计）；
    face 维持 prefill 行口径。"""

    EVICTION = {"time_ns": 40, "reason": "synthetic",
                "victim_session_id": "sA", "victim_instance_index": 0,
                "shard_bytes": [40]}

    def test_decode_row_evictions_extracted(self):
        record = {"kind": "decode", "request_id": "sB_r0", "tick": 40,
                  "decision": {
                      "decode_instance_index": 1,
                      "prefill_decode_transfer": {"total_bytes": 100,
                                                  "source_instance_index": 0,
                                                  "target_instance_index": 1},
                      "decode_target_evictions": [dict(self.EVICTION)]}}
        events = kv_cache_adapter.extract_events_wscllm(record)
        self.assertEqual(len(events), 2)  # PD 迁移 + decode 侧逐出
        evict = [e for e in events
                 if e["cause"].startswith(kv_cache_adapter.FAMILY_EVICTION)]
        self.assertEqual(len(evict), 1)
        self.assertEqual(evict[0]["bytes"], 40)
        self.assertEqual(evict[0]["start_ns"], 40)

    def test_prefill_row_subset_not_extracted(self):
        record = {"kind": "prefill", "request_id": "sB_r0", "tick": 30,
                  "decision": {
                      "prefill_instance_index": 0,
                      "history_action": "LOCAL_HIT",
                      "history_transfer_bytes": 0,
                      "admission_evictions": [],
                      # prefill 行子集（写端发射时点快照）不产事件。
                      "decode_target_evictions": [dict(self.EVICTION)]}}
        self.assertEqual(kv_cache_adapter.extract_events_wscllm(record), [])

    def test_face_still_extracts_prefill_subset(self):
        record = {"kind": "prefill", "request_id": "sB_r0", "tick": 30,
                  "decision": {
                      "prefill_instance_index": 0,
                      "history_action": "LOCAL_HIT",
                      "history_transfer_bytes": 0,
                      "admission_evictions": [],
                      "decode_target_evictions": [dict(self.EVICTION)]}}
        events = kv_cache_adapter.extract_events_face_wscllm(record)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["bytes"], 40)


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


class OutputDefaultContractTests(unittest.TestCase):
    """D2 SS-4 / E1（2026-09-28）：-o 缺省去向契约双侧钉死。

    行为侧：单 run_dir 子命令（e2e-stats 代表）缺省写 run_dir/<name>
    文件、stdout 不见 CSV；多点聚合命令（normalized/scan-export）缺省写
    stdout、不落文件；显式 '-' 恒 stdout。help 侧：两族 help 文本如实
    区分（add_output_arg default_target），多点命令不再谎称写 run_dir。
    """

    @staticmethod
    def _run(*argv):
        return subprocess.run(
            [sys.executable, str(SLO_TOOLS_DIR / "slo_stats.py"), *argv],
            capture_output=True, text=True)

    def test_single_run_dir_command_defaults_to_file(self):
        run_dir = synthetic.make_run_dir("odef_file")
        synthetic.write_request_metrics(run_dir, [
            synthetic.request_row(request_id="r0", e2e_ns="100")])
        synthetic.write_cpp_log(run_dir, [], repo_variant="astra-sim-face")
        proc = self._run("e2e-stats", run_dir.as_posix())
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertNotIn("queue_index", proc.stdout,
                         "单 run_dir 命令缺省不得写 stdout")
        self.assertTrue((run_dir / "slo_e2e_stats.csv").is_file(),
                        "缺省产物应落在 run_dir 旁")

    def test_multi_point_commands_default_to_stdout(self):
        run_dir = synthetic.make_run_dir("odef_stdout")
        synthetic.write_request_metrics(run_dir, [
            synthetic.request_row(request_id="r0", e2e_ns="100")])
        synthetic.write_cpp_log(run_dir, [], repo_variant="astra-sim-face")
        # scan-export：缺省 = stdout，不落文件
        proc = self._run("scan-export",
                         "--point", run_dir.as_posix() + "=1.0")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("run_dir,repo_variant", proc.stdout,
                      "多点命令缺省须写 stdout")
        self.assertFalse((run_dir / "slo_scan_export.csv").exists(),
                         "多点命令缺省不得落 run_dir 文件")
        # normalized：缺省 = stdout，不落文件
        proc = self._run("normalized", run_dir.as_posix())
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("normalized_time_p50", proc.stdout)
        self.assertFalse((run_dir / "slo_normalized.csv").exists())

    def test_explicit_dash_is_stdout_for_single_run_command(self):
        run_dir = synthetic.make_run_dir("odef_dash")
        synthetic.write_request_metrics(run_dir, [
            synthetic.request_row(request_id="r0", e2e_ns="100")])
        synthetic.write_cpp_log(run_dir, [], repo_variant="astra-sim-face")
        proc = self._run("e2e-stats", run_dir.as_posix(), "-o", "-")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("metric,unit", proc.stdout)
        self.assertIn("e2e_p50", proc.stdout)
        self.assertFalse((run_dir / "slo_e2e_stats.csv").exists(),
                         "显式 '-' 不得再落 run_dir 文件")

    def test_help_text_matches_default_target(self):
        proc = self._run("e2e-stats", "--help")
        self.assertEqual(proc.returncode, 0)
        self.assertIn("缺省写 run_dir/slo_e2e_stats.csv", proc.stdout)
        for sub, name in (("normalized", "slo_normalized.csv"),
                          ("scan-export", "slo_scan_export.csv")):
            proc = self._run(sub, "--help")
            self.assertEqual(proc.returncode, 0)
            self.assertIn(f"缺省写 stdout/{name}", proc.stdout,
                          f"{sub} help 未如实登记 stdout 缺省")
            self.assertNotIn(f"缺省写 run_dir/{name}", proc.stdout,
                             f"{sub} help 仍谎称写 run_dir")


class SessionPassthroughTests(unittest.TestCase):
    """B4（S3 异常③）：metrics_manifest 透传 human_time_ns/tool_time_ns 后
    T_session 可算；turn-0 双侧 null=0 贡献（非缺数据）；键缺席才 NA。"""

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
        # Σ(human+tool) = 0(turn-0) + 200(turn-1) = 200；
        # T_session = 9000 - 1000 - 200 = 7800。
        run_dir = self._write_run([
            synthetic.manifest_request(
                "s0t0", "s0", 0, 0, human_time_ns=None, tool_time_ns=None),
            synthetic.manifest_request(
                "s0t1", "s0", 1, 1, human_time_ns=200, tool_time_ns=None),
        ])
        out = run_dir / "slo_session.csv"
        args = slo_argv(["session", str(run_dir), "-o", str(out)])
        self.assertEqual(slo_stats.cmd_session(args), 0)
        with out.open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["t_session_ns"], "7800")
        self.assertEqual(rows[0]["sum_human_tool_ns"], "200")
        self.assertIn("turn0/0-gap", rows[0]["missing_fields_sample"])

    def test_t_session_na_when_keys_absent(self):
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
        self.assertEqual(rows[0]["missing_field_count"], "4")


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

    def test_backlog_bucket_ns_carries_step_value(self):
        """--bucket-ns 桶内取 max 须含上一事件值延伸进本桶的平台（series
        是阶跃函数）：桶 1 内唯一事件 (250, 1→0) 递减，桶 max=携带值 1
        而非 0（原实现漏携带 → 低估，M10-B3，2026-09-27 修）。"""
        run_dir = synthetic.make_run_dir("bl2")
        synthetic.write_request_metrics(run_dir, [
            synthetic.request_row(request_id="r0", queue_index="0",
                                  arrival_ns="100", completion_ns="150"),
            synthetic.request_row(request_id="r1", queue_index="1",
                                  arrival_ns="100", completion_ns="250"),
        ])
        out = run_dir / "backlog_bucket.csv"
        args = slo_argv(["backlog", str(run_dir), "--bucket-ns", "100",
                         "-o", str(out)])
        self.assertEqual(slo_stats.cmd_backlog(args), 0)
        header, rows = synthetic.read_csv(out)
        self.assertEqual([(r["time_ns"], r["in_flight"]) for r in rows],
                         [("100", "2"), ("200", "1")])

    def test_backlog_bucket_ns_fills_gap_buckets(self):
        """--bucket-ns 无事件桶整桶恒为携入值，携入值>0 须产行（分桶时序
        不得静默丢桶，M10-B3，2026-09-27 修；与 face 实现同语义）：
        r0 [100,600) 与 r1 [320,350) 使桶 200/400/500 无事件——
        桶 200 携入 1、桶 400/500 携入 1；桶 300 首事件递增，桶内
        max=2。携入=0 的空桶（600 之后）不产行，与逐事件口径一致。"""
        run_dir = synthetic.make_run_dir("bl3")
        synthetic.write_request_metrics(run_dir, [
            synthetic.request_row(request_id="r0", queue_index="0",
                                  arrival_ns="100", completion_ns="600"),
            synthetic.request_row(request_id="r1", queue_index="1",
                                  arrival_ns="320", completion_ns="350"),
        ])
        out = run_dir / "backlog_bucket_gap.csv"
        args = slo_argv(["backlog", str(run_dir), "--bucket-ns", "100",
                         "-o", str(out)])
        self.assertEqual(slo_stats.cmd_backlog(args), 0)
        header, rows = synthetic.read_csv(out)
        self.assertEqual([(r["time_ns"], r["in_flight"]) for r in rows],
                         [("100", "1"), ("200", "1"), ("300", "2"),
                          ("400", "1"), ("500", "1"), ("600", "1")])


class NormalizedZeroSampleTests(unittest.TestCase):
    def test_all_zero_e2e_does_not_divide_by_zero(self):
        """组内 max P50/P99 为 0（全零耗时样本）→ 归一化列 NA，不抛
        ZeroDivisionError（原版仅 tput 有零值保护，R3:G7-F1）。

        E1（2026-09-28）补断言：此前只钉退出码 0——归一化列退化为
        "inf"/"0.000000" 或空串的回归同样 exit 0（半边契约无钉）。
        现断言 stdout 产物：normalized_time_p50/p99/normalized_tps 三列
        双 run 全为字面 "NA"（分母=组内 max=0 → 归一化无定义），原始
        p50_e2e_ns/p99_e2e_ns 仍携带 0 值（分子半边不丢）。"""
        import contextlib
        import io
        run_dirs = []
        for tag in ("nz1", "nz2"):
            rd = synthetic.make_run_dir(tag)
            synthetic.write_request_metrics(rd, [
                synthetic.request_row(request_id="r0", queue_index="0",
                                      arrival_ns="100", completion_ns="100",
                                      e2e_ns="0"),
            ])
            synthetic.write_cpp_log(rd, [], repo_variant="astra-sim-face")
            run_dirs.append(rd)
        args = slo_argv(["normalized", *[str(r) for r in run_dirs],
                         "-o", "-"])
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            self.assertEqual(slo_stats.cmd_normalized(args), 0)
        table = list(csv.reader(io.StringIO(buffer.getvalue())))
        header, rows = table[0], table[1:]
        columns = dict(zip(header, range(len(header))))
        self.assertEqual(len(rows), 2)
        for row in rows:
            for column in ("normalized_time_p50", "normalized_time_p99",
                           "normalized_tps"):
                self.assertEqual(row[columns[column]], "NA",
                                 f"{column}: 归一化列须为字面 NA")
            self.assertEqual(row[columns["p50_e2e_ns"]], "0")
            self.assertEqual(row[columns["p99_e2e_ns"]], "0")


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


class ReadErrorFailClosedTests(unittest.TestCase):
    """P2-16/P2-17（2026-09-28）：

    * P2-16 —— 坏 UTF-8/IO/CSV/gzip 读取错误与 SloToolError 同出口
      （run_main → EXIT_FAIL_CLOSED=2），不再裸 traceback + 退出码 1。
    * P2-17 —— manifest_requests 对非 dict 条目 fail-closed，不再静默
      丢弃（四调用方静默缺数据）。
    """

    def test_run_main_input_read_errors_exit_fail_closed(self):
        import zlib
        cases = [
            UnicodeDecodeError("utf-8", b"\xff", 0, 1,
                               "invalid start byte"),
            OSError("I/O error"),
            csv.Error("bad field"),
            EOFError("truncated gzip"),
            zlib.error("Error -3 while decompressing"),
        ]
        for exc in cases:
            def boom(_exc=exc):
                raise _exc
            with self.subTest(error_type=type(exc).__name__):
                self.assertEqual(
                    slo_common.run_main(boom),
                    slo_common.EXIT_FAIL_CLOSED)

    def test_bad_utf8_request_metrics_exit_fail_closed(self):
        run_dir = synthetic.make_run_dir("badbytes")
        (run_dir / "request_metrics.csv").write_bytes(
            b"queue_index,request_id\n0,\xff\xfe\n")
        proc = subprocess.run(
            [sys.executable, str(SLO_TOOLS_DIR / "slo_stats.py"),
             "e2e-stats", run_dir.as_posix()],
            capture_output=True, text=True)
        self.assertEqual(proc.returncode, slo_common.EXIT_FAIL_CLOSED)
        self.assertIn("FAIL-CLOSED", proc.stderr)
        self.assertNotIn("Traceback", proc.stderr)

    def test_manifest_requests_non_dict_fails_closed(self):
        for bad in ("oops", None, 7, [1, 2]):
            with self.subTest(entry=repr(bad)):
                with self.assertRaises(SloToolError):
                    slo_common.manifest_requests({"requests": [{"a": 1},
                                                               bad]})
        # 合法条目原样返回（含子 dict），不改变既有行为。
        self.assertEqual(
            slo_common.manifest_requests(
                {"requests": [{"a": 1}, {"b": {"c": 2}}]}),
            [{"a": 1}, {"b": {"c": 2}}])

class D2ToolGateTests(unittest.TestCase):
    """D2（2026-09-28）批次：slo_tools 门面修复逐条钉死。"""

    # ------------------------------------------------ schema_version 门 --

    def test_slo_manifest_schema_version_type_gate(self):
        # bool/浮点不再冒充整数 1（True == 1、1.0 == 1 的 Python 等值
        # 陷阱）；缺 schema_version 同拒。
        import tempfile
        for bad in (True, 1.0, "1", None):
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "manifest.json"
                path.write_text(json.dumps(
                    {"schema_version": bad, "params": {}}),
                    encoding="utf-8")
                with self.assertRaises(SloToolError):
                    load_slo_manifest(path)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.json"
            path.write_text(json.dumps(
                {"schema_version": 1, "params": {}}), encoding="utf-8")
            self.assertEqual(load_slo_manifest(path)["schema_version"], 1)

    # ---------------------------------------- violation 分子单一实现 --

    def test_violation_counts_excludes_non_completed_from_numerator(self):
        # D2 修复：cmd_violation 与 cmd_scan_export 共用 violation_counts
        # ——failed 但携带 e2e 的行计入分母、绝不进分子（旧 scan-export
        # 裸循环只认"e2e 非 NA"，靠 drain 断言兜底、潜伏）。
        # F2_PY（2026-09-29）RV5-F6 裁定：no_e2e 恢复旧口径——只数 e2e
        # 为 NA 的行（D2 SS-1 抽取时把"非 completed 行"也计入，字段名
        # rows_without_e2e 失实；D2 报告自己宣称"no_e2e 语义原样保留"）。
        # 分子排除非 completed 的修复本身保持不变（本例 failed+e2e 行
        # e2e=9999 > deadline=1000，若分子泄漏 num 即 2→1 的差）。
        rows = [
            {"queue_index": 0, "terminal_status": "completed",
             "prefill_length": 500, "decode_length": 100, "e2e_ns": 9999},
            {"queue_index": 1, "terminal_status": "failed",
             "prefill_length": 500, "decode_length": 100, "e2e_ns": 9999},
            {"queue_index": 2, "terminal_status": "completed",
             "prefill_length": 500, "decode_length": 100, "e2e_ns": 10},
        ]
        t_isolated = {(0, 0): 1000}
        num, den, no_e2e, per_status = slo_stats.violation_counts(
            rows, 1.0, [0, 1000, 2000], [0, 200, 400], t_isolated)
        self.assertEqual((num, den, no_e2e), (1, 3, 0))
        self.assertEqual(per_status["completed"],
                         {"total": 2, "violating": 1})
        self.assertEqual(per_status["failed"], {"total": 1, "violating": 0})

    # ------------------------------------------------ hopbytes 字节门 --

    def test_byte_count_strict_parsing(self):
        self.assertIsNone(hopbytes._byte_count(True))     # bool 不按 1 字节
        self.assertIsNone(hopbytes._byte_count(False))
        self.assertEqual(hopbytes._byte_count(150), 150)
        self.assertEqual(hopbytes._byte_count(150.0), 150)  # 整值浮点接受
        self.assertIsNone(hopbytes._byte_count(150.5))    # 非整值浮点
        self.assertIsNone(hopbytes._byte_count("150"))
        self.assertIsNone(hopbytes._byte_count(0))
        self.assertIsNone(hopbytes._byte_count(-5))
        self.assertIsNone(hopbytes._byte_count(None))

    def test_sh20_missing_hop_bytes_goes_to_uncovered_side(self):
        # D2 修复：shard_count>0 但 noc_hop_bytes 缺席的行不再"字节计入
        # 覆盖侧宣称覆盖、hop_bytes 贡献 0"——按 bytes_without_hops 计
        # （对齐 face 分支口径）；合法 int>=0（含 0 跳）仍计覆盖侧。
        acc = {"actions_with_hops": 0, "hop_bytes_total": 0,
               "bytes_with_hops": 0, "bytes_without_hops": 0,
               "hops_length_mismatch": 0}
        per_request = {}
        hopbytes.collect_sh20(
            {"kind": "decode", "request_id": "r", "decision": {
                "transfer_hop_bytes": [
                    {"total_bytes": 100, "noc_hop_bytes": 4,
                     "shard_count": 2},
                    {"total_bytes": 50, "shard_count": 2},
                    {"total_bytes": 30, "noc_hop_bytes": 0,
                     "shard_count": 1},
                ]}},
            acc, per_request)
        self.assertEqual(acc["bytes_with_hops"], 130)   # 100 + 0 跳的 30
        self.assertEqual(acc["hop_bytes_total"], 4)     # 行内预乘 noc_hop_bytes：4 + 0
        self.assertEqual(acc["bytes_without_hops"], 50)
        self.assertEqual(acc["actions_with_hops"], 2)

    # ------------------------------------ load_imbalance 无 admission --

    def test_li_decode_without_admission_is_counted(self):
        # D2 修复：有 decode 决策、无 prefill 准入的请求不再静默消失
        # （四类偏差有计数唯独此路径没有）。
        import io
        from contextlib import redirect_stderr
        state = load_imbalance.li_prepare()
        load_imbalance.li_consume_decision(
            {"kind": "prefill", "request_id": "r1", "tick": 100,
             "decision": {}},
            "astra-sim-face", state)
        load_imbalance.li_consume_decision(
            {"kind": "decode", "request_id": "r1", "tick": 150,
             "decision": {"decode_instance_index": 0}},
            "astra-sim-face", state)
        load_imbalance.li_consume_decision(
            {"kind": "decode", "request_id": "r2", "tick": 200,
             "decision": {"decode_instance_index": 1}},
            "astra-sim-face", state)
        drains = {"r1": (400, 0), "r2": (400, 1)}
        buffer = io.StringIO()
        with redirect_stderr(buffer):
            intervals = load_imbalance.li_assemble_intervals(
                state, drains, 0)
        self.assertEqual(len(intervals), 1)
        self.assertEqual(intervals[0]["request_id"], "r1")
        self.assertIn("无 admission 决策", buffer.getvalue())
        self.assertIn("r2", buffer.getvalue())

    # --------------------------------- restore_decomposition 锚点门 --

    def test_memory_anchor_rejects_bool_and_negative_tick(self):
        import tempfile
        for bad_tick in (True, -5):
            with tempfile.TemporaryDirectory() as tmp:
                log = Path(tmp) / "cpp.log"
                log.write_text("[METRIC] " + json.dumps({
                    "type": "memory_anchor", "subject_id": 0,
                    "tick_ns": bad_tick}) + "\n", encoding="utf-8")
                with self.assertRaises(SloToolError):
                    restore_decomposition.scan_metric_anchors_requests(log)

    # -------------------------------------- kv_cache_adapter time_ns --

    def test_eviction_entry_time_ns_null_falls_back_to_tick(self):
        # 显式 null（或 bool）不再让 start_ns 携 None 落盘为 NA——回退
        # 决策发射 tick（本层文档口径）。
        self.assertEqual(
            kv_cache_adapter._entry_time_ns({"time_ns": None}, 77), 77)
        self.assertEqual(
            kv_cache_adapter._entry_time_ns({"time_ns": True}, 77), 77)
        self.assertEqual(
            kv_cache_adapter._entry_time_ns({}, 77), 77)
        self.assertEqual(
            kv_cache_adapter._entry_time_ns({"time_ns": 42}, 77), 42)


class F2PyToolGateTests(unittest.TestCase):
    """F2_PY（2026-09-29）：RV5-F6/F8——批间纪律贯穿修复的钉子。

    * F6：violation_counts 的 rows_without_e2e 恢复旧口径（只数 e2e 为
      NA 的行）；D2 SS-1 的分子排除修复保持不变。
    * F8：hops 判定族补 bool 排除（与 _byte_count 同口径，bool 不冒充
      1 跳）；_shard_hops/_route_hops 孪生助手合并为单一实现（双入口名
      保留，与 collect_sh30=collect_sh10 先例一致）。
    """

    # ------------------------------------- F6：no_e2e 旧口径恢复 --

    def test_violation_counts_no_e2e_counts_only_missing_e2e(self):
        # 旧口径（git HEAD cmd_violation）：no_e2e 只在 e2e is None 时计。
        # failed+e2e 行不计（e2e 在场）；completed e2e=NA 计 1；
        # failed e2e=NA 计 1。分子排除非 completed 与此独立、保持不变。
        rows = [
            {"queue_index": 0, "terminal_status": "failed",
             "prefill_length": 500, "decode_length": 100, "e2e_ns": 9999},
            {"queue_index": 1, "terminal_status": "completed",
             "prefill_length": 500, "decode_length": 100, "e2e_ns": None},
            {"queue_index": 2, "terminal_status": "failed",
             "prefill_length": 500, "decode_length": 100, "e2e_ns": None},
            {"queue_index": 3, "terminal_status": "completed",
             "prefill_length": 500, "decode_length": 100, "e2e_ns": 10},
        ]
        num, den, no_e2e, per_status = slo_stats.violation_counts(
            rows, 1.0, [0, 1000, 2000], [0, 200, 400], {(0, 0): 1000})
        self.assertEqual((num, den, no_e2e), (0, 4, 2))
        self.assertEqual(per_status["completed"],
                         {"total": 2, "violating": 0})
        self.assertEqual(per_status["failed"],
                         {"total": 2, "violating": 0})

    # --------------------------------- F8：hops 族 bool 冒充 + 孪生合并 --

    def test_route_hops_helpers_reject_bool_and_share_implementation(self):
        self.assertIsNone(hopbytes._shard_hops({"noc_hops": True}))
        self.assertIsNone(hopbytes._route_hops({"noc_hops": True}))
        self.assertIsNone(hopbytes._shard_hops({"noc_hops": False}))
        # 合法值不变：显式字段优先、noc_path 推导兜底、皆缺 None。
        self.assertEqual(hopbytes._route_hops({"noc_hops": 3}), 3)
        self.assertEqual(hopbytes._route_hops({"noc_path": [0, 5, 1]}), 2)
        self.assertIsNone(hopbytes._route_hops({}))
        # 孪生助手已合并为单一实现（双入口名保留）。
        self.assertIs(hopbytes._shard_hops, hopbytes._route_hops)

    @staticmethod
    def _acc():
        return {"actions_with_hops": 0, "hop_bytes_total": 0,
                "bytes_with_hops": 0, "bytes_without_hops": 0,
                "hops_length_mismatch": 0}

    def _assert_bool_hops_uncovered(self, collect, record):
        acc = self._acc()
        per_request = {}
        collect(record, acc, per_request)
        self.assertEqual(acc["bytes_without_hops"], 100,
                         "bool hops 冒充的传输必须整体进无覆盖侧")
        self.assertEqual(acc["hop_bytes_total"], 0,
                         "bool hops 不得按 True==1 计 1 跳")
        self.assertEqual(acc["bytes_with_hops"], 0)
        self.assertEqual(acc["actions_with_hops"], 0)

    def test_sh10_shard_bool_hops_uncovered(self):
        self._assert_bool_hops_uncovered(
            hopbytes.collect_sh10,
            {"kind": "decode", "request_id": "r", "decision": {
                "prefill_decode_transfer": {"total_bytes": 100, "shards": [
                    {"bytes": 100, "noc_hops": True}]}}})

    def test_face_prefill_bool_hop_element_uncovered(self):
        self._assert_bool_hops_uncovered(
            hopbytes.collect_face,
            {"kind": "prefill", "request_id": "r", "decision": {
                "history_transfer_bytes": 100,
                "history_noc_hops": [True]}})

    def test_face_decode_bool_hop_element_uncovered(self):
        self._assert_bool_hops_uncovered(
            hopbytes.collect_face,
            {"kind": "decode", "request_id": "r", "decision": {
                "prefill_decode_transfer": {"shards": [{"bytes": 100}]},
                 "kv_noc_hops": [True]}})

    def test_wscllm_prefill_aggregate_bool_hops_uncovered(self):
        self._assert_bool_hops_uncovered(
            hopbytes.collect_wscllm,
            {"kind": "prefill", "request_id": "r", "decision": {
                "history_transfer_bytes": 100, "noc_hops": True}})

    def test_wscllm_decode_static_route_bool_hop_uncovered(self):
        self._assert_bool_hops_uncovered(
            hopbytes.collect_wscllm,
            {"kind": "decode", "request_id": "r", "decision": {
                "prefill_decode_transfer": {"total_bytes": 100},
                "static_route": {"hop_count": True}}})


if __name__ == "__main__":
    unittest.main(verbosity=2)
