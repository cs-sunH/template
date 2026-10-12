#!/usr/bin/env python3
"""批次 D（2026-09-30，wscllm-LRU 深度审查报告 C-4/M-30~M-36 + Minor 组）
修复的针对性回归面。

覆盖：
  * C-4  backlog --bucket-ns 桶化：桶起点携入值 + 零事件桶补行；
  * M-30 hbm_watermark 容量违规 exit=3 经 driver 与 run_slo_postprocess.sh
    原样透传（不再折叠为 1）；
  * M-31 journal evict_events 会话口径（同会话跨 rank 多行去重；bytes
    不虚增）；
  * M-32 certified 第三道门分量字段严格类型校验（缺失/str/float/bool
    拒收）+ physical 纳入校验；
  * M-33 孤儿 checksum 诚实标注（orphan_checksum_upper_bound_only）；
  * M-35 BoundedSorter 半消费迭代 spill 清理契约；
  * M-36 collect_sh20 覆盖口径（noc_hop_bytes 缺席/谎报 →
    bytes_without_hops）+ bool 谎报排除；
  * Minor 组：violation 非 completed 状态门、scan-export λ 有限性、
    session 空 session_id 排除与 manifest 透传值类型门、parse_ns/parse_int
    下划线字面量拒绝、load_json_strict、kv_cache_adapter（sh_1.0 非 dict
    history_transfer / 重复 prefill / 空 decision log）、load_imbalance
    （重复 decode / decode-only 请求留痕）、restore memory_anchor
    bool/负值拒收。

运行：python3 -m pytest -q sh_test_mesh/slo_tools/tests/（与其余套件同跑）。
"""

from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
SLO_TOOLS_DIR = TESTS_DIR.parent
sys.path.insert(0, str(SLO_TOOLS_DIR))
sys.path.insert(0, str(TESTS_DIR))

import synthetic  # noqa: E402
import test_hbm_watermark as H  # noqa: E402
import hopbytes  # noqa: E402
import kv_cache_adapter  # noqa: E402
import load_imbalance  # noqa: E402
import restore_decomposition  # noqa: E402
import slo_common  # noqa: E402
import slo_stats  # noqa: E402
from slo_common import SloToolError  # noqa: E402

REPO_ROOT = SLO_TOOLS_DIR.parent.parent


def slo_argv(argv: list[str]) -> object:
    return slo_stats.build_parser().parse_args(argv)


def csv_rows(path: Path):
    import csv as _csv
    with path.open(newline="", encoding="utf-8") as handle:
        return list(_csv.DictReader(handle))


def _acc() -> dict:
    return {"actions_with_hops": 0, "hop_bytes_total": 0,
            "bytes_with_hops": 0, "bytes_without_hops": 0}


# ---------------------------------------------------------------------------
# C-4：backlog --bucket-ns 桶化（桶起点携入值 + 零事件桶补行）
# ---------------------------------------------------------------------------

class BacklogBucketCarryTests(unittest.TestCase):
    """报告 C-4 实测场景：+1@50 / +1@80 / −1@210 / −1@250，bucket=100ns。

    在途数是事件间阶梯常数：桶 [100,200) 无事件，真实在途=携入 2（旧行
    为整桶丢行）；桶 200 只发生完成事件，真实峰值=携入 2（旧行为只对桶
    内事件取 max 记 1）。
    """

    def _write_run(self, tag: str) -> Path:
        run_dir = synthetic.make_run_dir(tag)
        synthetic.write_request_metrics(run_dir, [
            synthetic.request_row(queue_index="0", request_id="r0",
                                  arrival_ns="50", completion_ns="210"),
            synthetic.request_row(queue_index="1", request_id="r1",
                                  arrival_ns="80", completion_ns="250"),
        ])
        return run_dir

    def test_bucket_carries_in_value_and_refills_empty_buckets(self):
        run_dir = self._write_run("blc1")
        out = run_dir / "backlog.csv"
        args = slo_argv(["backlog", str(run_dir), "--bucket-ns", "100",
                         "-o", str(out)])
        self.assertEqual(slo_stats.cmd_backlog(args), 0)
        header, rows = synthetic.read_csv(out)
        self.assertEqual(header, ["time_ns", "in_flight"])
        self.assertEqual([(r["time_ns"], r["in_flight"]) for r in rows],
                         [("0", "2"), ("100", "2"), ("200", "2")])

    def test_per_event_output_unchanged(self):
        run_dir = self._write_run("blc2")
        out = run_dir / "backlog.csv"
        args = slo_argv(["backlog", str(run_dir), "-o", str(out)])
        self.assertEqual(slo_stats.cmd_backlog(args), 0)
        _, rows = synthetic.read_csv(out)
        self.assertEqual([(r["time_ns"], r["in_flight"]) for r in rows],
                         [("50", "1"), ("80", "2"), ("210", "1"),
                          ("250", "0")])


# ---------------------------------------------------------------------------
# Minor（D2 移植）：violation 分子显式排除非 completed 行
# ---------------------------------------------------------------------------

class ViolationStatusGateTests(unittest.TestCase):
    def _setup_with_noncompleted_carrying_e2e(self):
        run_dir = synthetic.make_run_dir("viogate")
        manifest = synthetic.write_slo_manifest(run_dir, {
            "alpha_main": 2.0,
            "bucket_percentiles": synthetic.TEST_BUCKET_EDGES,
        })
        rows = [
            synthetic.request_row(queue_index="0", request_id="r0",
                                  e2e_ns="900", arrival_ns="0",
                                  completion_ns="900"),  # completed 违约
            # rejected/dropped 行携带 e2e（写端不变式的合成反例）：不得进分子。
            synthetic.request_row(queue_index="1", request_id="r1",
                                  terminal_status="rejected", e2e_ns="9900",
                                  arrival_ns="0", completion_ns="9900"),
            synthetic.request_row(queue_index="2", request_id="r2",
                                  terminal_status="dropped", e2e_ns="9900",
                                  arrival_ns="0", completion_ns="9900"),
        ]
        synthetic.write_request_metrics(run_dir, rows)
        t_iso = synthetic.write_t_isolated(run_dir, [(0, 0, 400)])
        synthetic.write_cpp_log(run_dir, [], repo_variant="astra-sim-wscllm")
        return run_dir, manifest, t_iso

    def test_violation_excludes_non_completed_from_numerator(self):
        run_dir, manifest, t_iso = self._setup_with_noncompleted_carrying_e2e()
        out = run_dir / "violation.json"
        args = slo_argv(["violation", str(run_dir), "--t-isolated",
                         str(t_iso), "--manifest", str(manifest),
                         "-o", str(out)])
        self.assertEqual(slo_stats.cmd_violation(args), 0)
        payload = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual(payload["violation_numerator"], 1)
        self.assertEqual(payload["violation_denominator"], 3)
        self.assertEqual(payload["per_terminal_status"]["rejected"],
                         {"total": 1, "violating": 0})

    def test_scan_export_point_rejects_nonfinite_lambda(self):
        run_dir = synthetic.make_run_dir("violam")
        for bad in ("inf", "nan", "1e999", "-inf"):
            args = slo_argv(["scan-export", "--point", f"{run_dir}={bad}"])
            with self.assertRaises(SloToolError):
                slo_stats.cmd_scan_export(args)


# ---------------------------------------------------------------------------
# Minor：session 空 session_id 排除 + manifest 透传值类型门
# ---------------------------------------------------------------------------

class SessionGuardTests(unittest.TestCase):
    def test_empty_session_id_rows_excluded_not_merged(self):
        run_dir = synthetic.make_run_dir("sessempty")
        synthetic.write_request_metrics(run_dir, [
            synthetic.request_row(queue_index="0", request_id="s0t0",
                                  session_id="s0", turn_index="0",
                                  arrival_ns="1000", completion_ns="5000"),
            # 空 session_id 行：旧行为并入 "NA" 伪会话（跨会话合并污染分位）。
            synthetic.request_row(queue_index="1", request_id="orphan",
                                  session_id="", turn_index="0",
                                  arrival_ns="2000", completion_ns="9000"),
        ])
        synthetic.write_metrics_manifest(run_dir, [
            synthetic.manifest_request("s0t0", "s0", 0, 0,
                                       human_time_ns=None, tool_time_ns=None),
            synthetic.manifest_request("orphan", "", 0, 1),
        ])
        out = run_dir / "slo_session.csv"
        args = slo_argv(["session", str(run_dir), "-o", str(out)])
        self.assertEqual(slo_stats.cmd_session(args), 0)
        rows = csv_rows(out)
        self.assertEqual([r["session_id"] for r in rows], ["s0"])
        self.assertEqual(rows[0]["t_session_ns"], "4000")

    def test_manifest_passthrough_value_type_gate(self):
        # H5①（2026-09-30）：Σ 核对域 = 前 n−1 轮——类型门的坏值须落在
        # 受检轮次（旧单轮夹具的末轮=唯一轮，H5① 起不在核对域）。
        run_dir = synthetic.make_run_dir("sesstype")
        synthetic.write_request_metrics(run_dir, [
            synthetic.request_row(queue_index="0", request_id="s0t0",
                                  session_id="s0", turn_index="0",
                                  arrival_ns="1000", completion_ns="5000"),
            synthetic.request_row(queue_index="1", request_id="s0t1",
                                  session_id="s0", turn_index="1",
                                  arrival_ns="5200", completion_ns="9000"),
        ])
        synthetic.write_metrics_manifest(run_dir, [
            synthetic.manifest_request("s0t0", "s0", 0, 0,
                                       human_time_ns=1.5, tool_time_ns=None),
            synthetic.manifest_request("s0t1", "s0", 1, 1,
                                       human_time_ns=200, tool_time_ns=None),
        ])
        args = slo_argv(["session", str(run_dir), "-o", "-"])
        with self.assertRaises(SloToolError):
            slo_stats.cmd_session(args)


# ---------------------------------------------------------------------------
# Minor：slo_common 下划线字面量 / load_json_strict / INPUT_READ_ERRORS
# ---------------------------------------------------------------------------

class SloCommonGateTests(unittest.TestCase):
    def test_parse_rejects_underscore_literal(self):
        for bad in ("1_000", "10_000_000_000"):
            with self.assertRaises(SloToolError):
                slo_common.parse_ns(bad, "f", "w")
            with self.assertRaises(SloToolError):
                slo_common.parse_int(bad, "f", "w")
        self.assertEqual(slo_common.parse_ns("1000", "f", "w"), 1000)

    def test_load_json_strict_rejects_nonfinite_literals(self):
        for bad in ('{"a": NaN}', '{"a": Infinity}', '{"a": -Infinity}'):
            with self.assertRaises(SloToolError):
                slo_common.load_json_strict(bad, "w")
        self.assertEqual(slo_common.load_json_strict('{"a": 1}', "w"),
                         {"a": 1})

    def test_run_main_maps_unicode_decode_error_to_fail_closed(self):
        def broken():
            raise UnicodeDecodeError("utf-8", b"\xff", 0, 1, "bad")
        self.assertEqual(
            slo_common.run_main(broken), slo_common.EXIT_FAIL_CLOSED)


# ---------------------------------------------------------------------------
# M-35：BoundedSorter 半消费迭代 spill 清理契约
# ---------------------------------------------------------------------------

class BoundedSorterCleanupTests(unittest.TestCase):
    def test_half_consumed_iteration_cleans_spills_on_close(self):
        sorter = slo_common.BoundedSorter(chunk_size=2)
        for value in range(50):
            sorter.add(value)
        it = sorter.sorted_iter()
        next(it)  # 半消费（P50/P95 游标提前停的生产形态）
        spill_paths = tuple(sorter.spill_paths)
        self.assertTrue(spill_paths)
        it.close()  # 显式关闭（GC 前的确定性释放路径）
        self.assertEqual(sorter.spill_paths, ())
        self.assertTrue(all(not p.exists() for p in spill_paths))


# ---------------------------------------------------------------------------
# M-36：collect_sh20 覆盖口径（对齐 collect_wscllm 的 G7-15 形态）
# ---------------------------------------------------------------------------

class CollectSh20CoverageTests(unittest.TestCase):
    def test_missing_hop_bytes_goes_to_uncovered_side(self):
        acc = _acc()
        per_request: dict = {}
        record = {"kind": "prefill", "request_id": "r0", "tick": 1,
                  "decision": {"transfer_hop_bytes": [
                      {"total_bytes": 100, "shard_count": 2}]}}
        hopbytes.collect_sh20(record, acc, per_request)
        self.assertEqual(acc["bytes_without_hops"], 100)
        self.assertEqual(acc["bytes_with_hops"], 0)
        self.assertEqual(acc["actions_with_hops"], 0)
        self.assertEqual(acc["hop_bytes_total"], 0)

    def test_nonfinite_type_hop_bytes_goes_to_uncovered_side(self):
        acc = _acc()
        per_request: dict = {}
        record = {"kind": "prefill", "request_id": "r0", "tick": 1,
                  "decision": {"transfer_hop_bytes": [
                      {"total_bytes": 100, "shard_count": 2,
                       "noc_hop_bytes": "30"}]}}
        hopbytes.collect_sh20(record, acc, per_request)
        self.assertEqual(acc["bytes_without_hops"], 100)
        self.assertEqual(acc["bytes_with_hops"], 0)

    def test_zero_hop_counts_as_covered(self):
        acc = _acc()
        per_request: dict = {}
        record = {"kind": "prefill", "request_id": "r0", "tick": 1,
                  "decision": {"transfer_hop_bytes": [
                      {"total_bytes": 100, "shard_count": 1,
                       "noc_hop_bytes": 0}]}}
        hopbytes.collect_sh20(record, acc, per_request)
        self.assertEqual(acc["bytes_with_hops"], 100)
        self.assertEqual(acc["hop_bytes_total"], 0)
        self.assertEqual(acc["actions_with_hops"], 1)

    def test_bool_lies_rejected(self):
        acc = _acc()
        per_request: dict = {}
        record = {"kind": "prefill", "request_id": "r0", "tick": 1,
                  "decision": {"transfer_hop_bytes": [
                      {"total_bytes": True, "shard_count": 1,
                       "noc_hop_bytes": 5}]}}
        hopbytes.collect_sh20(record, acc, per_request)
        self.assertEqual(acc, {"actions_with_hops": 0, "hop_bytes_total": 0,
                               "bytes_with_hops": 0, "bytes_without_hops": 0})
        record2 = {"kind": "prefill", "request_id": "r1", "tick": 1,
                   "decision": {"transfer_hop_bytes": [
                       {"total_bytes": 100, "shard_count": 1,
                        "noc_hop_bytes": True}]}}
        hopbytes.collect_sh20(record2, acc, per_request)
        self.assertEqual(acc["bytes_without_hops"], 100)
        self.assertEqual(acc["hop_bytes_total"], 0)

    def test_shard_hops_route_helpers_reject_bool(self):
        # True 冒充 1 跳：_shard_hops/_route_hops 落 noc_path 推导/None。
        self.assertIsNone(hopbytes._shard_hops({"noc_hops": True}))
        self.assertIsNone(hopbytes._route_hops({"noc_hops": True}))


# ---------------------------------------------------------------------------
# Minor：kv_cache_adapter（sh_1.0 非 dict / 重复 prefill / 空 log）
# ---------------------------------------------------------------------------

class KvAdapterGuardTests(unittest.TestCase):
    def test_sh10_non_dict_history_transfer_fails_closed(self):
        state = kv_cache_adapter.AdapterState()
        variant = kv_cache_adapter.REPO_VARIANTS["astra-sim-sh_1.0"]
        record = {"kind": "prefill", "request_id": "r0", "tick": 1,
                  "decision": {"history_transfer": "local_hit"}}
        with self.assertRaises(SloToolError):
            kv_cache_adapter.adapter_consume(record, variant, {}, state)

    def test_duplicate_prefill_decision_fails_closed(self):
        state = kv_cache_adapter.AdapterState()
        variant = kv_cache_adapter.REPO_VARIANTS["astra-sim-wscllm"]
        record = {"kind": "prefill", "request_id": "r0", "tick": 1,
                  "decision": {}}
        kv_cache_adapter.adapter_consume(record, variant, {}, state)
        with self.assertRaises(SloToolError):
            kv_cache_adapter.adapter_consume(record, variant, {}, state)

    def test_empty_decision_log_fails_closed_on_emit(self):
        import argparse
        state = kv_cache_adapter.AdapterState()
        # emit 首检即拒：空账本不产出对账表（与 hbm_watermark 同姿态）。
        with self.assertRaises(SloToolError):
            kv_cache_adapter.adapter_emit(
                argparse.Namespace(output="-", reconcile=False, json="",
                                   hit_states=""),
                "astra-sim-wscllm", variant=None, manifest={},
                turns={}, adapter_state=state)


# ---------------------------------------------------------------------------
# Minor：load_imbalance（重复 decode / decode-only 请求留痕 / 默认
# trace_config 回退关闭）
# ---------------------------------------------------------------------------

class LoadImbalanceGuardTests(unittest.TestCase):
    def test_duplicate_decode_decision_fails_closed(self):
        state = load_imbalance.li_prepare()
        record = {"kind": "decode", "request_id": "r0", "tick": 1,
                  "decision": {"decode_instance_index": 0}}
        load_imbalance.li_consume_decision(record, "astra-sim-wscllm", state)
        with self.assertRaises(SloToolError):
            load_imbalance.li_consume_decision(record, "astra-sim-wscllm",
                                               state)

    def test_decode_only_request_accounted_not_silent(self):
        import contextlib
        import io
        state = load_imbalance.li_prepare()
        load_imbalance.li_consume_decision(
            {"kind": "decode", "request_id": "d0", "tick": 5,
             "decision": {"decode_instance_index": 1}},
            "astra-sim-wscllm", state)
        # 无 admission 的 decode 请求不再从遍历域静默消失——进显式跳过
        # 清单并在 stderr 留痕（区间全空仍 fail-closed，原契约不变）。
        buffer = io.StringIO()
        with self.assertRaises(SloToolError), \
                contextlib.redirect_stderr(buffer):
            load_imbalance.li_assemble_intervals(state, {}, 0)
        self.assertIn("无 prefill 准入", buffer.getvalue())

    def test_cv_denominator_rejects_repo_default_trace_config(self):
        # 无本地 trace_config 的 run_dir：分母候选链不回退仓内默认配置。
        run_dir = synthetic.make_run_dir("linodenom")
        from hbm_watermark import count_configured_instances
        with self.assertRaises(SloToolError):
            count_configured_instances(run_dir, None,
                                       allow_repo_default=False)


# ---------------------------------------------------------------------------
# Minor：restore_decomposition memory_anchor bool/负值拒收
# ---------------------------------------------------------------------------

class RestoreAnchorGuardTests(unittest.TestCase):
    def test_bool_or_negative_anchor_fields_rejected(self):
        for subject, tick in ((True, 5), (0, True), (-1, 5), (0, -5)):
            cpp_log = synthetic.make_run_dir("anchorbad") / "cpp.log"
            records = [
                {"type": "memory_anchor", "subject_id": subject,
                 "rank": 0, "node_id": 1, "tick_ns": tick},
            ]
            cpp_log.write_text(
                "".join("[METRIC] " + json.dumps(r, sort_keys=True) + "\n"
                        for r in records), encoding="utf-8")
            with self.assertRaises(SloToolError):
                restore_decomposition.scan_metric_anchors_requests(cpp_log)


# ---------------------------------------------------------------------------
# M-31 / M-32 / M-33 / M-30：journal 路径（subprocess 级，复用
# test_hbm_watermark 的合成 fixture 面）
# ---------------------------------------------------------------------------

class JournalBatchDTests(unittest.TestCase):
    def _fixture(self, tag: str, rows, ranks_final, checksum=None):
        records, tokens = H.base_handcalc_records()
        run_dir = H.make_run_dir(tag)
        H.write_fixture(run_dir, repo_variant="astra-sim-wscllm",
                        records=records, token_requests=tokens,
                        model_rows=True)
        H.write_journal(run_dir, rows,
                        checksum if checksum is not None
                        else H.default_checksum(rows, ranks_final))
        return run_dir

    # -- M-31：evict_events 会话口径 ---------------------------------------

    def test_tp_ranks_single_session_evict_dedup(self):
        # 同会话一次逐出跨 2 rank（同 transaction_id、同 session_id）：
        # 事件数=1（旧逐行计数=2），bytes=700 不虚增。
        rows = [
            H.journal_row(0, 0, 0, 0, 1000, d_weight=300),
            H.journal_row(1, 0, 1, 0, 1000, d_weight=200),
            H.journal_row(2, 10, 0, 0, 1000, weight=300, d_resident=400),
            H.journal_row(3, 10, 1, 0, 1000, weight=200, d_resident=300),
            H.journal_row(4, 30, 0, 0, 1000, weight=300, resident=400,
                          d_resident=-400,
                          cause="evict_static_decode_final_kv_reservation_"
                                "session:layers0-32"),
            H.journal_row(5, 30, 1, 0, 1000, weight=200, resident=300,
                          d_resident=-300,
                          cause="evict_static_decode_final_kv_reservation_"
                                "session:layers0-32"),
        ]
        for row in rows[4:]:
            row["session_id"] = "session_A"
        rows[4]["transaction_id"] = 7
        rows[5]["transaction_id"] = 7
        ranks_final = {
            "0": {"capacity_bytes": 1000, "weight": 300, "resident": 0,
                  "reserved": 0, "physical": 300},
            "1": {"capacity_bytes": 1000, "weight": 200, "resident": 0,
                  "reserved": 0, "physical": 200},
        }
        run_dir = self._fixture("m31dedup", rows, ranks_final)
        proc = H.run_tool(run_dir)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        summary = json.loads((run_dir / "summary.json").read_text())
        self.assertEqual(summary["trust_tier"],
                         "per_rank_total_hbm_certified")
        self.assertEqual(summary["total_evict_events"], 1)
        self.assertEqual(summary["total_evict_bytes"], 700)
        self.assertEqual(summary["instances"]["0"]["evict_events"], 1)

    def test_distinct_sessions_in_distinct_transactions_counted(self):
        rows = [
            H.journal_row(0, 0, 0, 0, 1000, d_weight=300),
            H.journal_row(1, 10, 0, 0, 1000, weight=300, d_resident=100),
            H.journal_row(2, 20, 0, 0, 1000, weight=300, resident=100,
                          d_resident=-100,
                          cause="evict_static_decode_final_kv_reservation_"
                                "session:layers0-32"),
            H.journal_row(3, 30, 0, 0, 1000, weight=300, d_resident=100),
            H.journal_row(4, 40, 0, 0, 1000, weight=300, resident=100,
                          d_resident=-100,
                          cause="evict_static_decode_final_kv_reservation_"
                                "session:layers0-32"),
        ]
        rows[2].update(transaction_id=3, session_id="session_A")
        rows[4].update(transaction_id=5, session_id="session_B")
        ranks_final = {
            "0": {"capacity_bytes": 1000, "weight": 300, "resident": 0,
                  "reserved": 0, "physical": 300},
        }
        run_dir = self._fixture("m31two", rows, ranks_final)
        proc = H.run_tool(run_dir)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        summary = json.loads((run_dir / "summary.json").read_text())
        self.assertEqual(summary["total_evict_events"], 2)
        self.assertEqual(summary["total_evict_bytes"], 200)

    # -- M-32：第三道门严格类型校验 + physical 纳入 ------------------------

    def _good_journal(self):
        rows = [
            H.journal_row(0, 0, 0, 0, 1000, d_weight=300),
            H.journal_row(1, 10, 0, 0, 1000, weight=300, d_resident=100),
            H.journal_row(2, 20, 0, 0, 1000, weight=300, resident=100,
                          d_resident=-100, cause="terminal_session_retire"),
        ]
        ranks_final = {
            "0": {"capacity_bytes": 1000, "weight": 300, "resident": 0,
                  "reserved": 0, "physical": 300},
        }
        return rows, ranks_final

    def test_wellformed_certificate_still_certified(self):
        rows, ranks_final = self._good_journal()
        run_dir = self._fixture("m32ok", rows, ranks_final)
        proc = H.run_tool(run_dir)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        summary = json.loads((run_dir / "summary.json").read_text())
        self.assertEqual(summary["trust_tier"],
                         "per_rank_total_hbm_certified")

    def _assert_malformed_cert_rejected(self, tag, mutate, marker):
        rows, ranks_final = self._good_journal()
        checksum = H.default_checksum(rows, ranks_final)
        mutate(ranks_final)
        run_dir = self._fixture(tag, rows, ranks_final, checksum)
        proc = H.run_tool(run_dir)
        self.assertEqual(proc.returncode, 2, proc.stderr)
        self.assertIn(marker, proc.stderr)

    def test_str_weight_rejected(self):
        self._assert_malformed_cert_rejected(
            "m32str", lambda r: r["0"].__setitem__("weight", "300"),
            "证书残缺")

    def test_float_resident_rejected(self):
        self._assert_malformed_cert_rejected(
            "m32float", lambda r: r["0"].__setitem__("resident", 0.0),
            "证书残缺")

    def test_missing_reserved_rejected(self):
        self._assert_malformed_cert_rejected(
            "m32miss", lambda r: r["0"].pop("reserved"), "证书残缺")

    def test_bool_reserved_rejected(self):
        # bool False 冒充 0（isinstance(int) 恒真）：显式拒收。
        self._assert_malformed_cert_rejected(
            "m32bool", lambda r: r["0"].__setitem__("reserved", False),
            "证书残缺")

    def test_physical_mismatch_rejected(self):
        self._assert_malformed_cert_rejected(
            "m32phys", lambda r: r["0"].__setitem__("physical", 301),
            "physical")

    def test_physical_missing_rejected(self):
        self._assert_malformed_cert_rejected(
            "m32physmiss", lambda r: r["0"].pop("physical"), "physical")

    def test_capacity_bytes_wrong_type_rejected(self):
        self._assert_malformed_cert_rejected(
            "m32cap", lambda r: r["0"].__setitem__("capacity_bytes", "1000"),
            "capacity_bytes")

    # -- M-33：孤儿证书诚实标注 ---------------------------------------------

    def test_orphan_checksum_labeled_not_old_run(self):
        records, tokens = H.base_handcalc_records()
        run_dir = H.make_run_dir("m33orphan")
        H.write_fixture(run_dir, repo_variant="astra-sim-wscllm",
                        records=records, token_requests=tokens,
                        model_rows=True)
        rows, ranks_final = self._good_journal()
        H.write_journal(run_dir, rows, H.default_checksum(rows, ranks_final))
        # 孤儿化：journal 删除、checksum 留存。
        (run_dir / "results" / "kv_delta_journal.jsonl").unlink()
        proc = H.run_tool(run_dir)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        summary = json.loads((run_dir / "summary.json").read_text())
        self.assertEqual(summary["violation_check"],
                         "orphan_checksum_upper_bound_only")
        self.assertIn("孤儿证书", proc.stderr)
        self.assertNotIn("阶段2 前旧 run", proc.stderr)

    # -- M-30：exit=3 经 driver 与 shell 透传 -------------------------------

    def _violation_fixture(self, tag: str) -> Path:
        records, tokens = H.base_handcalc_records()
        run_dir = H.make_run_dir(tag)
        H.write_fixture(run_dir, repo_variant="astra-sim-wscllm",
                        records=records, token_requests=tokens,
                        model_rows=True)
        rows = [
            H.journal_row(0, 0, 0, 0, 1000, d_weight=300),
            H.journal_row(1, 0, 1, 0, 1000, d_weight=200),
            # rank0 physical=1800 > capacity=1000：certified 层正式判决。
            H.journal_row(2, 10, 0, 0, 1000, weight=300, d_resident=1500),
            H.journal_row(3, 20, 1, 0, 1000, weight=200, d_resident=300),
            H.journal_row(4, 30, 0, 0, 1000, weight=300, resident=1500,
                          cause="terminal_session_retire"),
            H.journal_row(5, 30, 1, 0, 1000, weight=200, resident=300,
                          d_resident=-300, cause="terminal_session_retire"),
        ]
        ranks_final = {
            "0": {"capacity_bytes": 1000, "weight": 300, "resident": 1500,
                  "reserved": 0, "physical": 1800},
            "1": {"capacity_bytes": 1000, "weight": 200, "resident": 0,
                  "reserved": 0, "physical": 200},
        }
        H.write_journal(run_dir, rows, H.default_checksum(rows, ranks_final))
        return run_dir

    def test_tool_exit3_driver_exit3_shell_exit3(self):
        run_dir = self._violation_fixture("m30exit3")
        proc = H.run_tool(run_dir)
        self.assertEqual(proc.returncode, 3, proc.stderr)
        driver = subprocess.run(
            [sys.executable,
             str(SLO_TOOLS_DIR / "slo_postprocess_driver.py"), str(run_dir)],
            capture_output=True, text=True, check=False)
        self.assertEqual(driver.returncode, 3,
                         f"driver 退出码折叠回归：{driver.stderr[-800:]}")
        shell = subprocess.run(
            ["bash", str(REPO_ROOT / "sh_test_mesh" / "run_scripts" /
                         "run_slo_postprocess.sh"), str(run_dir)],
            capture_output=True, text=True, check=False)
        self.assertEqual(shell.returncode, 3,
                         f"shell 退出码折叠回归：{shell.stderr[-800:]}")

    def test_clean_run_driver_and_shell_exit0(self):
        rows, ranks_final = self._good_journal()
        run_dir = self._fixture("m30clean", rows, ranks_final)
        driver = subprocess.run(
            [sys.executable,
             str(SLO_TOOLS_DIR / "slo_postprocess_driver.py"), str(run_dir)],
            capture_output=True, text=True, check=False)
        self.assertEqual(driver.returncode, 0, driver.stderr[-800:])
        shell = subprocess.run(
            ["bash", str(REPO_ROOT / "sh_test_mesh" / "run_scripts" /
                         "run_slo_postprocess.sh"), str(run_dir)],
            capture_output=True, text=True, check=False)
        self.assertEqual(shell.returncode, 0, shell.stderr[-800:])

    # -- M-34：非 UTF-8 decision log 不再裸崩 -------------------------------

    def test_non_utf8_decision_log_fails_closed_not_crash(self):
        rows, ranks_final = self._good_journal()
        run_dir = self._fixture("m34utf", rows, ranks_final)
        log = run_dir / "results" / "online_decision_log.jsonl"
        payload = log.read_bytes()
        log.write_bytes(payload[:len(payload) // 2] + b"\xff\xfe garbage\n")
        driver = subprocess.run(
            [sys.executable,
             str(SLO_TOOLS_DIR / "slo_postprocess_driver.py"), str(run_dir)],
            capture_output=True, text=True, check=False)
        # 不裸崩（退出码 1 + traceback 丢失 FAIL 记账）：fail-closed
        # 记账后整体非零（任一步失败语义），且 stderr 带 FAIL-CLOSED、
        # FAIL 条目落盘。
        self.assertIn("FAIL-CLOSED", driver.stderr)
        self.assertNotIn("Traceback", driver.stderr)
        self.assertTrue((run_dir / "slo_postprocess.FAIL").exists())


if __name__ == "__main__":
    unittest.main()
