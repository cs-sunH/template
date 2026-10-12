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
        # A4-9（2026-09-26）：terminal_status 权威词表收紧为生产侧实际
        # 写出的 2 值（completed/failed，由 C++ completed 布尔派生）；
        # rejected/dropped/timed_out 不可构造，消费侧 fail-closed 拒绝。
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
    def test_repo_manifest_params_all_consumed_in_repo(self):
        """(2026-09-30, R2 audit) 静态钉子：manifest 的参数集必须与仓内
        消费引用名集合一致（双向）。上一轮裁掉 11 个零消费 campaign 参数
        后，没有任何测试钉住"现存参数全部有消费者"——防止再引入孤儿参数
        或死等（消费了不存在的参数名会在 fail-closed 读取处炸，反向的
        孤儿参数则完全静默）。"""
        import re
        from pathlib import Path

        root = Path(__file__).resolve().parents[2]  # sh_test_mesh/
        manifest = json.loads(
            (root / "slo_tools" / "slo_params_manifest.json")
            .read_text(encoding="utf-8"))
        param_names = set(manifest["params"].keys())
        # 静态扫描消费引用：slo_tools + workload + run_scripts 下的
        # Python 源码里出现的参数名字符串。
        consumed = set()
        for sub in ("slo_tools", "workload", "run_scripts", "tests"):
            base = root / sub
            if not base.exists():
                continue
            for py in base.rglob("*.py"):
                text = py.read_text(encoding="utf-8")
                for name in param_names:
                    if name in text:
                        consumed.add(name)
        unconsumed = param_names - consumed
        self.assertFalse(
            unconsumed,
            f"manifest 参数在仓内零消费引用（孤儿参数）: {sorted(unconsumed)}")

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

    def test_violation_missing_t_iso_bucket_fails_closed(self):
        # 命名口径（2026-09-28 复核）：生产 manifest 自 B4b 起全量非 null
        # （test_repo_manifest_params_filled_b4_frozen 钉死），本例实际触发
        # 的是合成行 (500,100) 落桶 (1,1) 而 T_isolated 表仅 (0,0,400) 的
        # 缺桶 fail-closed；null-param 路径由
        # test_require_param_null_fails_closed 单独覆盖。
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
        # 这些值全部远超 Deadline=800，verdict 必然翻转）。十进制整数字
        # 面量不带下划线（R8-F4：parse_ns 拒绝 "1_000" 型字面量）。
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


class KvAdapterKindGateTests(unittest.TestCase):
    """B5（2026-09-28）：kind 白名单闸——合法 admission_probe 观测行登记
    通过（跳过但计数留痕），未知 kind 在 canonical 与 native（--reconcile
    重放）双侧 fail-closed，不得静默（否则对账恒 pass）。"""

    @staticmethod
    def _face_ns(run_dir):
        import argparse
        return argparse.Namespace(
            run_dir=run_dir, output=str(run_dir / "events.csv"),
            hit_states=str(run_dir / "hits.csv"), json="", reconcile=True,
            repo_variant=None, request_manifest=None)

    def _make_face_run(self, run_dir, records):
        synthetic.write_cpp_log(run_dir, [], repo_variant="astra-sim-face")
        synthetic.write_jsonl(run_dir, "online_decision_log.jsonl", records)
        synthetic.write_metrics_manifest(
            run_dir, [synthetic.manifest_request("r0", "s0", 0, 0)],
            repo_variant="astra-sim-face")

    def test_admission_probe_kinds_pass(self):
        # 生产方（face_online_scheduler log_decision）的合法观测 kind：
        # 无映射负载，跳过但计数留痕，--reconcile 照常 pass。
        run_dir = synthetic.make_run_dir("kvkind1")
        self._make_face_run(run_dir, [
            {"kind": "prefill", "request_id": "r0", "seq": 1, "tick": 1,
             "decision": {"history_action": "NO_HISTORY"}},
            {"kind": "prefill_admission_probe", "request_id": "r0",
             "seq": 2, "tick": 2, "decision": {}},
            {"kind": "decode", "request_id": "r0", "seq": 3, "tick": 3,
             "decision": {}},
            {"kind": "decode_admission_probe", "request_id": "r0",
             "seq": 4, "tick": 4, "decision": {}},
            {"kind": "completion", "request_id": "r0", "seq": 5, "tick": 9,
             "decision": {}},
        ])
        self.assertEqual(
            kv_cache_adapter.cmd_adapter(self._face_ns(run_dir)), 0)
        # 闸把 probe 行如数计入审计（不静默）。
        variant, turns, state = kv_cache_adapter.adapter_prepare(
            "astra-sim-face", json.loads(
                (run_dir / "metrics_manifest.json").read_text()))
        for record in slo_common.iter_jsonl(
                run_dir / "results" / "online_decision_log.jsonl"):
            kv_cache_adapter.adapter_consume(record, variant, turns, state)
        self.assertEqual(state.skipped_admission_probe, 2)

    def test_unknown_kind_fail_closed(self):
        # 拼写漂移 kind（"prefil"）：canonical 侧（adapter_consume，CLI 与
        # driver 共用闸）fail-closed，不再静默跳过。
        run_dir = synthetic.make_run_dir("kvkind2")
        self._make_face_run(run_dir, [
            {"kind": "prefil", "request_id": "r0", "seq": 1, "tick": 1,
             "decision": {}},
        ])
        with self.assertRaises(SloToolError) as ctx:
            kv_cache_adapter.cmd_adapter(self._face_ns(run_dir))
        self.assertIn("未知 kind", str(ctx.exception))
        self.assertIn("prefil", str(ctx.exception))

    def test_unknown_kind_fail_closed_reconcile_native(self):
        # --reconcile 的 native 重放侧同闸：未知 kind 在对账重放即 fail
        # （不依赖 canonical 侧先撞上）。
        run_dir = synthetic.make_run_dir("kvkind3")
        self._make_face_run(run_dir, [
            {"kind": "future_kind", "request_id": "r0", "seq": 1, "tick": 1,
             "decision": {}},
        ])
        with self.assertRaises(SloToolError) as ctx:
            kv_cache_adapter.reconcile(
                run_dir=run_dir, repo_variant="astra-sim-face",
                out_path=None, summary={})
        self.assertIn("未知 kind", str(ctx.exception))
        self.assertIn("future_kind", str(ctx.exception))


class FaceEvictionBytesTests(unittest.TestCase):
    """_eviction_bytes_face 校验强度（2026-09-28）：与
    hbm_watermark._sum_shard_bytes 同型——shard_bytes 元素必须为非负
    整数，float（int() 静默截断记账）/str/None（裸 TypeError 崩壳）一律
    fail-closed（SloToolError → exit 2）。canonical 与 --reconcile native
    重放两侧共用本函数，此处收紧即双侧同封。"""

    def test_valid_shard_bytes_summed(self):
        entry = {"shard_bytes": [100, 200], "reason": "x"}
        self.assertEqual(kv_cache_adapter._eviction_bytes_face(entry), 300)
        # 缺失/空列表维持既有 None 语义（不产事件，不报错）。
        self.assertIsNone(kv_cache_adapter._eviction_bytes_face({}))
        self.assertIsNone(
            kv_cache_adapter._eviction_bytes_face({"shard_bytes": []}))

    def test_invalid_element_types_fail_closed(self):
        # 校验强度逐句镜像 hbm_watermark._sum_shard_bytes（isinstance int
        # + >=0，bool 不特判——与 hbm 侧完全同强度，不私自加严）。
        for bad in (40.5, "x", None, -1):
            entry = {"shard_bytes": [100, bad], "reason": "r",
                     "trigger_request_id": "req"}
            with self.assertRaises(SloToolError) as ctx:
                kv_cache_adapter._eviction_bytes_face(entry)
            self.assertIn("shard_bytes", str(ctx.exception))
            self.assertIn("非负整数", str(ctx.exception))


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
        # collect_intervals 的读取方向：exits 数组携带终态 request_id
        # （li_collect_drains），drains 键不被工具消费。r0/r1 终态在
        # instance0 列车、r2 在 instance1 列车，与 decode 实例一致。
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

    def test_missing_manifest_param_fails(self):
        """缺参数必须以"参数未推导"为失败原因（R2-F5：旧断言因缺
        decision log 等旁路原因先行失败，属假覆盖）——显式提供 manifest
        （imbalance_bucket_ns=null）与 decision log，钉死失败来自
        require_param 的 value=null fail-closed 出口。
        """
        run_dir = synthetic.make_run_dir("li2")
        synthetic.write_cpp_log(run_dir, [], repo_variant="astra-sim-face")
        synthetic.write_jsonl(run_dir, "online_decision_log.jsonl", [
            {"kind": "prefill", "request_id": "r0", "tick": 0,
             "decision": {}},
        ])
        manifest = synthetic.write_slo_manifest(
            run_dir, {"imbalance_bucket_ns": None})
        import argparse
        ns = argparse.Namespace(
            run_dir=run_dir, manifest=manifest, output="-", json="",
            repo_variant=None)
        with self.assertRaises(SloToolError) as ctx:
            load_imbalance.cmd_load_imbalance(ns)
        self.assertIn("imbalance_bucket_ns", str(ctx.exception))
        self.assertIn("value=null", str(ctx.exception))

    def test_duplicate_decode_decision_fails(self):
        """2026-09-28 查重补齐：同一请求的 decode 决策重复原先静默后者
        覆盖——现与同文件 prefill 查重（准入应恰一次）及
        hbm_watermark.seen_kinds 的恰一次不变量同型 fail-closed。"""
        state = load_imbalance.li_prepare()
        load_imbalance.li_consume_decision(
            {"kind": "decode", "request_id": "r0", "tick": 1,
             "decision": {"decode_instance_index": 0}},
            "astra-sim-face", state)
        with self.assertRaises(SloToolError) as ctx:
            load_imbalance.li_consume_decision(
                {"kind": "decode", "request_id": "r0", "tick": 2,
                 "decision": {"decode_instance_index": 1}},
                "astra-sim-face", state)
        self.assertIn("多次 decode 决策", str(ctx.exception))


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

    def test_holder_without_shards_dropped_accounted(self):
        """A5-3+py-slo-tests-5（2026-09-26 契约改写）：缺 shards 的
        holder 不再被静默丢弃——holder 计入 dropped_holders，可估的
        total_bytes 计入 dropped_bytes 并入 coverage 分母。旧实现（及
        旧期望）是无声消失：字节既不进覆盖侧也不进无覆盖侧，分母缩小、
        coverage 虚高。无 total_bytes 可估的 holder 仍计 dropped_holders
        （dropped_bytes 不变）。"""
        acc = {"actions_with_hops": 0, "hop_bytes_total": 0,
               "bytes_with_hops": 0, "bytes_without_hops": 0,
               "dropped_holders": 0, "dropped_bytes": 0}
        per_request: dict = {}
        record = {"kind": "prefill", "request_id": "r0", "tick": 1,
                  "decision": {"history_transfer": {
                      "kind": "noc_migrate", "total_bytes": 1500}}}
        hopbytes.collect_sh10(record, acc, per_request)
        self.assertEqual(acc["dropped_holders"], 1)
        self.assertEqual(acc["dropped_bytes"], 1500)
        self.assertEqual(acc["bytes_with_hops"], 0)
        self.assertEqual(acc["bytes_without_hops"], 0)
        # coverage 口径（hopbytes_emit）：dropped_bytes 必须留在分母里，
        # 不得让比率虚高。
        total = (acc["bytes_with_hops"] + acc["bytes_without_hops"]
                 + acc["dropped_bytes"])
        self.assertEqual(total, 1500)
        # 无 total_bytes 可估的 holder：仍显式计数，不静默。
        record2 = {"kind": "decode", "request_id": "r0", "tick": 2,
                   "decision": {"prefill_decode_transfer": {
                       "kind": "noc_migrate"}}}
        hopbytes.collect_sh10(record2, acc, per_request)
        self.assertEqual(acc["dropped_holders"], 2)
        self.assertEqual(acc["dropped_bytes"], 1500)

    def test_empty_or_invalid_shards_dropped_accounted(self):
        """B4①（2026-09-28 契约）：shards 为空列表或全无效（行不可计
        字节）时，holder 与 A5-3 缺 shards 分支同型处置——计入
        dropped_holders，可估的 total_bytes 计入 dropped_bytes 并留在
        coverage 分母；不得静默消失抬高 coverage。"""
        acc = {"actions_with_hops": 0, "hop_bytes_total": 0,
               "bytes_with_hops": 0, "bytes_without_hops": 0,
               "dropped_holders": 0, "dropped_bytes": 0}
        per_request: dict = {}
        record = {"kind": "prefill", "request_id": "r0", "tick": 1,
                  "decision": {"history_transfer": {
                      "kind": "noc_migrate", "total_bytes": 1500,
                      "shards": []}}}
        hopbytes.collect_sh10(record, acc, per_request)
        self.assertEqual(acc["dropped_holders"], 1)
        self.assertEqual(acc["dropped_bytes"], 1500)
        self.assertEqual(acc["bytes_with_hops"], 0)
        # 分母守恒：1500 必须留在 coverage 分母里（比率 0 而非虚高）。
        total = (acc["bytes_with_hops"] + acc["bytes_without_hops"]
                 + acc["dropped_bytes"])
        self.assertEqual(total, 1500)
        # 全无效 shard 行（非 dict / bytes 类型不符）同型处置（sh30 同
        # 一 _accumulate_shard_transfers 路径）。
        record2 = {"kind": "completion", "request_id": "r0", "tick": 2,
                   "decision": {"completion_evictions": [{
                       "kind": "remote_store", "total_bytes": 700,
                       "shards": ["junk", {"bytes": "x"}]}]}}
        hopbytes.collect_sh30(record2, acc, per_request)
        self.assertEqual(acc["dropped_holders"], 2)
        self.assertEqual(acc["dropped_bytes"], 2200)
        # 部分 shard 有效时 holder 级不回退（不与 shard 级双计）。
        record3 = {"kind": "completion", "request_id": "r0", "tick": 3,
                   "decision": {"completion_evictions": [{
                       "total_bytes": 1000,
                       "shards": [{"bytes": 400}, {"nope": 1}]}]}}
        hopbytes.collect_sh30(record3, acc, per_request)
        self.assertEqual(acc["dropped_bytes"], 2200)
        self.assertEqual(acc["bytes_without_hops"], 400)

    def test_sh20_mixed_invalid_rows_dropped_accounted(self):
        """B4②（2026-09-28 契约）：sh20 混合无效行在 continue 处不再
        蒸发——可估的 total_bytes 计入 dropped_bytes 并留在 coverage
        分母（2000/(2000+1000)=2/3，而非旧实现的 2000/2000=1）。"""
        acc = {"actions_with_hops": 0, "hop_bytes_total": 0,
               "bytes_with_hops": 0, "bytes_without_hops": 0,
               "dropped_holders": 0, "dropped_bytes": 0}
        per_request: dict = {}
        record = {"kind": "decode", "request_id": "r0", "tick": 1,
                  "decision": {
                      "prefill_decode_transfer_bytes": 3000,
                      "transfer_hop_bytes": [
                          {"total_bytes": 2000, "shard_count": 4,
                           "noc_hop_bytes": 8000},
                          {"total_bytes": 1000, "shard_count": "x"}]}}
        hopbytes.collect_sh20(record, acc, per_request)
        self.assertEqual(acc["bytes_with_hops"], 2000)
        self.assertEqual(acc["dropped_bytes"], 1000)
        total = (acc["bytes_with_hops"] + acc["bytes_without_hops"]
                 + acc["dropped_bytes"])
        self.assertEqual(total, 3000)
        # 行全无效的 completion 记录：无 aggregate 字段可回退 → 可估
        # 字节直接计 dropped_bytes（旧实现回退盲区，字节无声消失）。
        record2 = {"kind": "completion", "request_id": "r0", "tick": 2,
                   "decision": {"transfer_hop_bytes": [
                       {"total_bytes": 500}]}}
        hopbytes.collect_sh20(record2, acc, per_request)
        self.assertEqual(acc["dropped_bytes"], 1500)
        # 空列表 hop_rows 回退盲区：aggregate 字节计入无覆盖侧
        # （A5-4 同口径），不再绕过回退。
        record3 = {"kind": "prefill", "request_id": "r0", "tick": 3,
                   "decision": {"history_transfer_bytes": 4096,
                                "transfer_hop_bytes": []}}
        hopbytes.collect_sh20(record3, acc, per_request)
        self.assertEqual(acc["bytes_without_hops"], 4096)
        # local_hit 名义行（结构可判、零搬运）维持不计入任何桶。
        record4 = {"kind": "prefill", "request_id": "r0", "tick": 4,
                   "decision": {"transfer_hop_bytes": [
                       {"total_bytes": 0, "shard_count": 0}]}}
        hopbytes.collect_sh20(record4, acc, per_request)
        self.assertEqual(acc["dropped_bytes"], 1500)
        self.assertEqual(acc["bytes_without_hops"], 4096)

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

    def test_face_decode_all_invalid_shards_dropped_accounted(self):
        """B4① 同型（2026-09-28，face 兄弟分支）：face decode 的
        shards/kv_noc_hops 均合法 list 且等长、但全部 shard bytes 非法
        时，holder 不再静默消失——计 dropped_holders，可估 total_bytes
        计 dropped_bytes 留在 coverage 分母（旧实现字节蒸发 → coverage
        虚高）；空 shards 同型；混合场景不回退 holder 级 total_bytes
        （与有效 shard 字节双计，同 sh10 口径）。"""
        acc = {"actions_with_hops": 0, "hop_bytes_total": 0,
               "bytes_with_hops": 0, "bytes_without_hops": 0,
               "dropped_holders": 0, "dropped_bytes": 0}
        per_request: dict = {}
        record = {"kind": "decode", "request_id": "r0", "tick": 1,
                  "decision": {
                      "prefill_decode_transfer": {
                          "kind": "noc_migrate", "total_bytes": 1000,
                          "shards": [{"bytes": "x"}]},
                      "kv_noc_hops": [2]}}
        hopbytes.collect_face(record, acc, per_request)
        self.assertEqual(acc["dropped_holders"], 1)
        self.assertEqual(acc["dropped_bytes"], 1000)
        self.assertEqual(acc["bytes_with_hops"], 0)
        self.assertEqual(acc["bytes_without_hops"], 0)
        # 分母守恒：1000 必须留在 coverage 分母里（比率 0 而非虚高）。
        total = (acc["bytes_with_hops"] + acc["bytes_without_hops"]
                 + acc["dropped_bytes"])
        self.assertEqual(total, 1000)
        # 空 shards + 空 hops 同型处置。
        record2 = {"kind": "decode", "request_id": "r0", "tick": 2,
                   "decision": {
                       "prefill_decode_transfer": {
                           "kind": "noc_migrate", "total_bytes": 500,
                           "shards": []},
                       "kv_noc_hops": []}}
        hopbytes.collect_face(record2, acc, per_request)
        self.assertEqual(acc["dropped_holders"], 2)
        self.assertEqual(acc["dropped_bytes"], 1500)
        # 混合场景（部分 shard 有效）：有效字节入账，holder 级不回退
        # （不双计）。
        record3 = {"kind": "decode", "request_id": "r0", "tick": 3,
                   "decision": {
                       "prefill_decode_transfer": {
                           "kind": "noc_migrate", "total_bytes": 1000,
                           "shards": [{"bytes": 400}, {"bytes": "x"}]},
                       "kv_noc_hops": [2, 2]}}
        hopbytes.collect_face(record3, acc, per_request)
        self.assertEqual(acc["dropped_bytes"], 1500)
        self.assertEqual(acc["hop_bytes_total"], 800)
        self.assertEqual(acc["bytes_with_hops"], 400)
        # 正常路径回归：全有效 shards 不受影响。
        record4 = {"kind": "decode", "request_id": "r0", "tick": 4,
                   "decision": {
                       "prefill_decode_transfer": {
                           "kind": "noc_migrate", "total_bytes": 300,
                           "shards": [{"bytes": 100}, {"bytes": 200}]},
                       "kv_noc_hops": [2, 3]}}
        hopbytes.collect_face(record4, acc, per_request)
        self.assertEqual(acc["dropped_holders"], 2)
        self.assertEqual(acc["dropped_bytes"], 1500)
        self.assertEqual(acc["hop_bytes_total"], 800 + 100 * 2 + 200 * 3)


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
T_session 可算；turn-0 双侧 null=0 贡献（非缺数据）。

py-slo-tests-9（2026-09-26 合并去重）：『键缺席 → NA 行 + 计数』分支的
断言与 tests/test_golden_g1g4.py G3 的
test_missing_transparent_fields_na_and_counted 等价重复（后者更全，另验
missing_fields_sample），合并后只在 golden G3 一处固化，本类不再重复。"""

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
        # 剔除 arrival < t0+window(=100+250=350) 的 3 行（100/200/300）后全为 1000 → P99=1000。
        self.assertEqual(payload["p99_full_ns"], 5000)
        self.assertEqual(payload["p99_after_warmup_ns"], 1000)
        self.assertAlmostEqual(payload["relative_change"], 0.8)
        self.assertFalse(payload["warmup_stable"])

    def test_warmup_not_evaluable_when_span_le_window(self):
        """到达跨度 <= 剔除窗口 → NA 降级产物 + exit 0(2026-10-01 修复锚:
        旧版 fail() exit 2,给 2s 冒烟窗的健康 run 盖 FAIL 根标记)。"""
        run_dir = synthetic.make_run_dir("wu2")
        rows = []
        for i in range(1, 6):
            rows.append(synthetic.request_row(
                request_id=f"r{i}", queue_index=str(i),
                arrival_ns=str(i * 100),
                completion_ns=str(i * 100 + 1000),
                e2e_ns="1000"))
        synthetic.write_request_metrics(run_dir, rows)
        manifest = synthetic.write_slo_manifest(run_dir, {
            "warmup_window": 15_000_000_000,
            "warmup_change_threshold": 0.05})
        out = run_dir / "warmup.json"
        args = slo_argv(["warmup", str(run_dir), "--manifest", str(manifest),
                         "-o", str(out)])
        self.assertEqual(slo_stats.cmd_warmup(args), 0)
        payload = json.loads(out.read_text(encoding="utf-8"))
        self.assertIsNone(payload["p99_after_warmup_ns"])
        self.assertIsNone(payload["warmup_stable"])
        self.assertIn("arrival_span_le_window",
                      payload["not_evaluable_reason"])
        self.assertEqual(payload["n_after_warmup"], 0)



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
