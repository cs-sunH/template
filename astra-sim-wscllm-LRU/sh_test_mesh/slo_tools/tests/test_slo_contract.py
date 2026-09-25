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

    def test_face_wscllm_tiered_location_level(self):
        """B4（-LRU 方案 A 双级）：history_location_before 三态优先。

        session 级 Tiered-LRU 改造后新运行值域 = {local_hbm, remote_memory}
        → 恒 full；partial_hbm_remote → partial 分支仅旧产物可达（本用例
        即 legacy 解析证据：适配器只读旧日志、不反驱调度）。"""
        f = kv_cache_adapter._hit_state_face_wscllm
        # legacy 解析证据：partial_hbm_remote 行（旧产物）仍可读。
        for location, expected in (("local_hbm", "full"),
                                   ("partial_hbm_remote", "partial"),
                                   ("remote_memory", "full")):
            state, evidence = f({
                "history_action": "REMOTE_RESTORE",
                "history_location_before": location,
                "history_location_before_instance_index": 3,
                "history_resident_prefix_layers": 16}, 1, "r")
            self.assertEqual(state, expected)
            self.assertIn(f"history_location_before={location}", evidence)
        # remote_memory → full（非 miss），evidence 区分 full_local/full_remote。
        state, evidence = f({
            "history_location_before": "remote_memory",
            "history_location_before_instance_index": None,
            "history_resident_prefix_layers": 0}, 2, "r")
        self.assertEqual(state, "full")
        self.assertIn("remote_memory", evidence)
        # 未知位置 fail 到 not_supported（值域封闭，不猜测）。
        self.assertEqual(f({"history_location_before": "mars"}, 1, "r")[0],
                         "not_supported")
        # location 缺失（旧产物 / 新产物 NO_HISTORY 恒 None）→ action 回退。
        self.assertEqual(f({"history_action": "NO_HISTORY",
                            "history_location_before": None}, 0, "r")[0],
                         "no_history")

    def test_wscllm_tiered_extraction_and_reconcile(self):
        """B4（-LRU wscllm）：契约字段优先（逐段传输/契约逐出行/PD dict），
        legacy 镜像行不消费（防双计）；--reconcile canonical==native。

        legacy 解析证据：本用例喂的是旧产物形状（PARTIAL_MIGRATE 两段
        恢复 / suffix_half 逐出 / partial_hbm_remote 位置）；适配器只读
        旧日志的能力必须保留。新产物形状见
        test_wscllm_whole_session_new_output。"""
        run_dir = synthetic.make_run_dir("kvw")
        synthetic.write_cpp_log(run_dir, [], repo_variant="astra-sim-wscllm")
        records = [
            # r0：旧式行（无契约字段）→ 回退级。
            {"kind": "prefill", "request_id": "r0", "tick": 1,
             "decision": {"history_action": "NO_HISTORY",
                          "history_transfer_bytes": 0,
                          "history_cache_state_before": "ABSENT"}},
            # r1：partial 两段式恢复 + 契约逐出行 + legacy 镜像（不双计）。
            {"kind": "prefill", "request_id": "r1", "tick": 2,
             "decision": {
                 "history_action": "PARTIAL_MIGRATE",
                 "history_location_before": "partial_hbm_remote",
                 "history_location_before_instance_index": 0,
                 "history_resident_prefix_layers": 16,
                 "history_transfer_bytes": 200,
                 "history_transfers": [
                     {"kind": "noc_migrate",
                      "reason": "history_partial_prefix_migrate",
                      "session_id": "s1", "total_bytes": 100,
                      "source_instance_index": 0,
                      "target_instance_index": 2,
                      "layer_start": 0, "layer_end": 16},
                     {"kind": "remote_load",
                      "reason": "history_remote_suffix_restore",
                      "session_id": "s1", "total_bytes": 100,
                      "source_instance_index": None,
                      "target_instance_index": 2,
                      "layer_start": 16, "layer_end": 32}],
                 "history_evictions": [
                     {"kind": "remote_store",
                      "reason": "static_decode_final_kv_reservation_"
                                "suffix_half",
                      "session_id": "s9", "total_bytes": 40,
                      "source_instance_index": 2,
                      "target_instance_index": None,
                      "layer_start": 16, "layer_end": 32}],
                 "prefill_evictions": [],
                 "admission_evictions": [{
                     "time_ns": 2, "phase": "prefill", "reason": "synthetic",
                     "trigger_request_id": "r1", "victim_session_id": "s9",
                     "victim_instance_index": 2,
                     "victim_last_completion_ns": 1, "context_tokens": 1,
                     "shard_bytes": [40]}],
                 "decode_target_evictions": []}},
            # decode：契约逐出行 decode_evictions + legacy 同值镜像
            # decode_target_evictions（不双计）+ PD dict（wscllm 契约形状）。
            {"kind": "decode", "request_id": "r1", "tick": 4,
             "decision": {
                 "decode_instance_index": 5,
                 "prefill_decode_transfer": {
                     "kind": "noc_migrate",
                     "reason": "prefill_decode_instance_migrate",
                     "session_id": "s1", "total_bytes": 300,
                     "source_instance_index": 2,
                     "target_instance_index": 5,
                     "layer_start": 0, "layer_end": 32},
                 "decode_evictions": [
                     {"kind": "remote_store",
                      "reason": "static_decode_final_kv_reservation_"
                                "suffix_half",
                      "session_id": "s8", "total_bytes": 60,
                      "source_instance_index": 5,
                      "target_instance_index": None,
                      "layer_start": 16, "layer_end": 32}],
                 "decode_target_evictions": [
                     {"kind": "remote_store",
                      "reason": "static_decode_final_kv_reservation_"
                                "suffix_half",
                      "session_id": "s8", "total_bytes": 60,
                      "source_instance_index": 5,
                      "target_instance_index": None,
                      "layer_start": 16, "layer_end": 32}]}},
            # completion：契约行（kind 在场）走 _transfer_object_events，
            # legacy 8 字段行走 shard_bytes 回退口径。
            {"kind": "completion", "request_id": "r1", "tick": 5,
             "decision": {"completion_evictions": [
                 {"kind": "remote_store",
                  "reason": "terminal_session_retire",
                  "session_id": "s2", "total_bytes": 50,
                  "source_instance_index": 5,
                  "target_instance_index": None,
                  "layer_start": 0, "layer_end": 32}]}},
        ]
        synthetic.write_jsonl(run_dir, "online_decision_log.jsonl", records)
        synthetic.write_metrics_manifest(
            run_dir,
            [synthetic.manifest_request("r0", "s0", 0, 0),
             synthetic.manifest_request("r1", "s1", 1, 1)],
            repo_variant="astra-sim-wscllm")
        import argparse
        ns = argparse.Namespace(
            run_dir=run_dir, output=str(run_dir / "events.csv"),
            hit_states=str(run_dir / "hits.csv"), json="",
            reconcile=True, repo_variant=None, request_manifest=None)
        self.assertEqual(kv_cache_adapter.cmd_adapter(ns), 0)
        _, rows = synthetic.read_csv(run_dir / "events.csv")
        causes = [(r["bytes"], r["cause"]) for r in rows]
        # history 两段（noc 100 + remote_load 100）+ 契约逐出 40（legacy
        # 镜像不双计）+ decode 契约逐出 60（镜像不双计）+ PD dict 300 +
        # completion 契约 50；local_hit 无。decode 行内逐出先于迁移
        # （记录内「逐出→迁移→增长」固定次序）。
        self.assertEqual(causes, [
            ("100", "history_transfer:noc_migrate:"
                    "history_partial_prefix_migrate"),
            ("100", "history_transfer:remote_load:"
                    "history_remote_suffix_restore"),
            ("40", "eviction:remote_store:"
                   "static_decode_final_kv_reservation_suffix_half"),
            ("60", "eviction:remote_store:"
                   "static_decode_final_kv_reservation_suffix_half"),
            ("300", "prefill_decode_transfer:"
                    "prefill_decode_instance_migrate"),
            ("50", "eviction:remote_store:terminal_session_retire")])
        _, hits = synthetic.read_csv(run_dir / "hits.csv")
        states = {r["request_id"]: r["kv_hit_state"] for r in hits}
        self.assertEqual(states, {"r0": "no_history", "r1": "partial"})
        evidence = {r["request_id"]: r["evidence"] for r in hits}
        self.assertIn("history_location_before=partial_hbm_remote",
                      evidence["r1"])

    def test_wscllm_whole_session_new_output(self):
        """session 级 Tiered-LRU 新产物：history_location_before ∈
        {local_hbm, remote_memory} → 恒 full；逐出为整会话 reason、层域
        [0, L)；输出不出现 partial/suffix 词素（新运行不写 partial 状态）。"""
        run_dir = synthetic.make_run_dir("kvwhole")
        synthetic.write_cpp_log(run_dir, [], repo_variant="astra-sim-wscllm")
        records = [
            # r0：无历史。
            {"kind": "prefill", "request_id": "r0", "tick": 1,
             "decision": {"history_action": "NO_HISTORY",
                          "history_location_before": None,
                          "history_resident_prefix_layers": None,
                          "history_transfer_bytes": 0,
                          "history_cache_state_before": "ABSENT"}},
            # r1：被逐会话远端全量恢复（REMOTE_RESTORE 单段）+ 整会话逐出。
            {"kind": "prefill", "request_id": "r1", "tick": 2,
             "decision": {
                 "history_action": "REMOTE_RESTORE",
                 "history_location_before": "remote_memory",
                 "history_location_before_instance_index": None,
                 "history_resident_prefix_layers": 0,
                 "history_transfer_bytes": 300,
                 "history_transfers": [
                     {"kind": "remote_load",
                      "reason": "history_remote_restore",
                      "session_id": "s1", "total_bytes": 300,
                      "source_instance_index": None,
                      "target_instance_index": 2,
                      "layer_start": 0, "layer_end": 32}],
                 "history_evictions": [
                     {"kind": "remote_store",
                      "reason": "history_and_prefill_admission_session",
                      "session_id": "s9", "total_bytes": 40,
                      "source_instance_index": 2,
                      "target_instance_index": None,
                      "layer_start": 0, "layer_end": 32}],
                 "prefill_evictions": [],
                 "admission_evictions": [{
                     "time_ns": 2, "phase": "prefill", "reason": "synthetic",
                     "trigger_request_id": "r1", "victim_session_id": "s9",
                     "victim_instance_index": 2,
                     "victim_last_completion_ns": 1, "context_tokens": 1,
                     "shard_bytes": [40]}],
                 "decode_target_evictions": []}},
            # r2：完整本地命中。
            {"kind": "prefill", "request_id": "r2", "tick": 3,
             "decision": {
                 "history_action": "LOCAL_HIT",
                 "history_location_before": "local_hbm",
                 "history_location_before_instance_index": 2,
                 "history_resident_prefix_layers": 32,
                 "history_transfer_bytes": 0,
                 "history_transfers": [],
                 "history_evictions": [],
                 "prefill_evictions": [],
                 "admission_evictions": [],
                 "decode_target_evictions": []}},
        ]
        synthetic.write_jsonl(run_dir, "online_decision_log.jsonl", records)
        synthetic.write_metrics_manifest(
            run_dir,
            [synthetic.manifest_request("r0", "s0", 0, 0),
             synthetic.manifest_request("r1", "s1", 1, 1),
             synthetic.manifest_request("r2", "s2", 2, 2)],
            repo_variant="astra-sim-wscllm")
        import argparse
        ns = argparse.Namespace(
            run_dir=run_dir, output=str(run_dir / "events.csv"),
            hit_states=str(run_dir / "hits.csv"), json="",
            reconcile=True, repo_variant=None, request_manifest=None)
        self.assertEqual(kv_cache_adapter.cmd_adapter(ns), 0)
        _, rows = synthetic.read_csv(run_dir / "events.csv")
        causes = [(r["bytes"], r["cause"]) for r in rows]
        # 全量恢复单段 300 + 整会话逐出 40（单层域 [0,32)）。
        self.assertEqual(causes, [
            ("300", "history_transfer:remote_load:history_remote_restore"),
            ("40", "eviction:remote_store:"
                   "history_and_prefill_admission_session")])
        _, hits = synthetic.read_csv(run_dir / "hits.csv")
        states = {r["request_id"]: r["kv_hit_state"] for r in hits}
        # 二态位置 → 恒 full；partial 仅旧产物可达。
        self.assertEqual(
            states, {"r0": "no_history", "r1": "full", "r2": "full"})
        # 新产物不出现 partial eviction/suffix eviction/partial restore 词素。
        dumped = "\n".join(cause for _, cause in causes)
        for banned in ("suffix_half", "partial", "PARTIAL", "full_fallback",
                       "history_remote_suffix_restore",
                       "history_partial_prefix_migrate"):
            self.assertNotIn(banned, dumped, banned)


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
        # 终态 drain 按生产 schema 只认 batch_train 行的 exits 数组
        # （prefill_train 行的 drains 字段是 P 侧发射记录、读取端显式
        # 跳过，wsc_llm_online_scheduler.py:_emit_train 权威口径）。
        ledger = [
            {"train_id": "batch_train_i0_1", "instance_index": 0,
             "tick": 100, "drains": [], "joiners": [],
             "exits": ["r0", "r1"]},
            {"train_id": "batch_train_i1_1", "instance_index": 1,
             "tick": 100, "drains": [], "joiners": [],
             "exits": ["r2"]},
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

    def test_wscllm_evictions_enter_denominator(self):
        """M29（2026-09-24）：逐出/回迁契约行计入 bytes_without_hops
        （此前整类不进账，coverage 分母缺逐出字节）。新旧产物单键取用：
        契约分段字段在场（B3+）读 history_evictions+prefill_evictions
        （prefill 行）与 decode_evictions（decode 行）；旧产物读
        admission_evictions+decode_target_evictions（prefill 行准入
        快照）——B3 起双序列化，同批消费会双计。"""
        acc = {"actions_with_hops": 0, "hop_bytes_total": 0,
               "bytes_with_hops": 0, "bytes_without_hops": 0}
        per_request: dict = {}
        record = {"kind": "prefill", "request_id": "r0", "tick": 1,
                  "decision": {
                      "history_transfers": [],
                      "history_transfer_bytes": 250, "noc_hops": 2,
                      "history_evictions": [
                          {"kind": "remote_store", "total_bytes": 300}],
                      "prefill_evictions": [
                          {"kind": "remote_store", "total_bytes": 100}],
                      # B3 双序列化：union 字段在场但单键取用不消费
                      # （取 999 与分段和 400 区分，防误读 union 也过测）。
                      "admission_evictions": [
                          {"kind": "remote_store", "total_bytes": 999}],
                      "decode_target_evictions": []}}
        hopbytes.collect_wscllm(record, acc, per_request)
        # 250×2 进分子；逐出 300+100 进分母（union 的 999 不消费不双计）。
        self.assertEqual(acc["hop_bytes_total"], 500)
        self.assertEqual(acc["bytes_with_hops"], 250)
        self.assertEqual(acc["bytes_without_hops"], 400)
        decode_record = {"kind": "decode", "request_id": "r0", "tick": 2,
                         "decision": {
                             "decode_evictions": [
                                 {"kind": "remote_store",
                                  "total_bytes": 700}]}}
        hopbytes.collect_wscllm(decode_record, acc, per_request)
        self.assertEqual(acc["bytes_without_hops"], 1100)
        # 旧产物（history_transfers 缺席）：prefill 行准入快照口径。
        old_record = {"kind": "prefill", "request_id": "r1", "tick": 3,
                      "decision": {
                          "admission_evictions": [
                              {"kind": "remote_store", "total_bytes": 90}],
                          "decode_target_evictions": [
                              {"kind": "remote_store",
                               "total_bytes": 10}]}}
        hopbytes.collect_wscllm(old_record, acc, per_request)
        self.assertEqual(acc["bytes_without_hops"], 1200)
        self.assertEqual(per_request["r1"]["bytes_without"], 100)


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

if __name__ == "__main__":
    unittest.main(verbosity=2)
