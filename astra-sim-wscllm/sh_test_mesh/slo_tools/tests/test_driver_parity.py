#!/usr/bin/env python3
"""A4 契约测试：BoundedSorter 有界外部排序 + 单遍 driver 与逐工具链的
逐字节对拍（合成 fixture）。

运行：python3 sh_test_mesh/slo_tools/tests/test_driver_parity.py
  或  python3 -m unittest discover -s sh_test_mesh/slo_tools/tests -v

三组用例：
  * BoundedSorter：随机整数/元组 vs sorted() 逐元素相等（chunk=1 强制
    spill 路径）；带序号键的稳定序等价；env SH_SLO_SORT_CHUNK 解析。
  * nearest_rank_percentile_many：与逐次 nearest_rank_percentile 在随机
    样本/多分位（含重复 p、乱序索引）上取值相等。
  * driver parity：合成 sh_1.0 run_dir 上，"旧链复刻"（9 个工具子命令
    串行 + run_slo_postprocess.sh 原日志语义）vs 当前
    run_slo_postprocess.sh（内部走 slo_postprocess_driver.py 单遍）——
    13 个 SLO 产物 + slo_postprocess.log 逐字节一致；含 G3（缺
    request_metrics）与失败路径（缺 token manifest）变体。P1（2026-
    08-30）起 hbm_watermark 产物集 = slo_hbm_intervals.csv（权威 RLE）
    + slo_hbm_plot_series.csv（行预算绘图，旧 slo_hbm_watermark_
    series.csv 退役）+ slo_hbm_watermark_instances.csv；无 journal 的
    fixture 恒 upper_bound_only 层（超限只诊断、exit 0）。
"""

from __future__ import annotations

import json
import os
import random
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
SLO_TOOLS_DIR = TESTS_DIR.parent
sys.path.insert(0, str(SLO_TOOLS_DIR))

import synthetic  # noqa: E402
from slo_common import (  # noqa: E402
    BoundedSorter, bounded_sort_chunk_size, nearest_rank_percentile,
    nearest_rank_percentile_many,
)

SLO_PRODUCTS = (
    "slo_e2e_stats.csv", "slo_backlog.csv", "slo_session.csv",
    "slo_warmup.json", "cache_events.csv", "kv_hit_states.csv",
    "slo_load_imbalance.csv", "slo_restore_decomposition.csv",
    "slo_hopbytes_total.csv", "slo_hopbytes_per_request.csv",
    "slo_hbm_intervals.csv", "slo_hbm_plot_series.csv",
    "slo_hbm_watermark_instances.csv",
    "slo_postprocess.log", "slo_postprocess.FAIL",
)


# ---------------------------------------------------------------------------
# BoundedSorter / 分位共享排序
# ---------------------------------------------------------------------------

class BoundedSorterTests(unittest.TestCase):
    """外部归并通道 ≡ sorted()（整数/整数元组全序，无稳定性可见差）。"""

    def test_random_ints_default_chunk(self):
        rng = random.Random(20260829)
        values = [rng.randrange(-10**12, 10**12) for _ in range(5000)]
        sorter = BoundedSorter()
        for value in values:
            sorter.add(value)
        self.assertEqual(list(sorter.sorted_iter()), sorted(values))

    def test_random_ints_forced_spill_chunk1(self):
        rng = random.Random(42)
        values = [rng.randrange(10**6) for _ in range(300)]
        sorter = BoundedSorter(chunk_size=1)
        for value in values:
            sorter.add(value)
        self.assertEqual(list(sorter.sorted_iter()), sorted(values))

    def test_tuples_with_position_key_match_stable_sort(self):
        """(key, seq, payload) 全序键 ≡ sorted(items, key=key) 稳定序。"""
        rng = random.Random(7)
        items = [(rng.randrange(20), rng.randrange(5), "payload")
                 for _ in range(400)]
        sorter = BoundedSorter(chunk_size=7)
        for position, item in enumerate(items):
            sorter.add((item[0], position, item))
        merged = [item for _, _, item in sorter.sorted_iter()]
        self.assertEqual(merged, sorted(items, key=lambda it: it[0]))

    def test_small_input_no_spill(self):
        sorter = BoundedSorter(chunk_size=1000)
        for value in (3, 1, 2):
            sorter.add(value)
        self.assertEqual(list(sorter.sorted_iter()), [1, 2, 3])
        self.assertEqual(sorter._spills, [])

    def test_chunk_envParsing(self):
        old = os.environ.get("SH_SLO_SORT_CHUNK")
        try:
            os.environ["SH_SLO_SORT_CHUNK"] = "3"
            self.assertEqual(bounded_sort_chunk_size(), 3)
            os.environ["SH_SLO_SORT_CHUNK"] = "not-a-number"
            self.assertEqual(bounded_sort_chunk_size(), 65536)
            os.environ["SH_SLO_SORT_CHUNK"] = "-5"
            self.assertEqual(bounded_sort_chunk_size(), 1)
            os.environ.pop("SH_SLO_SORT_CHUNK")
            self.assertEqual(bounded_sort_chunk_size(), 65536)
        finally:
            if old is not None:
                os.environ["SH_SLO_SORT_CHUNK"] = old
            else:
                os.environ.pop("SH_SLO_SORT_CHUNK", None)

    def test_spill_files_cleaned_up(self):
        sorter = BoundedSorter(chunk_size=2)
        for value in range(50):
            sorter.add(value)
        consumed = list(sorter.sorted_iter())
        self.assertEqual(consumed, list(range(50)))
        leftovers = list(Path(tempfile.gettempdir()).glob(
            "slo_bounded_sort_*"))
        self.assertEqual(leftovers, [])


class PercentileManyTests(unittest.TestCase):
    """多分位共享一次有界排序 ≡ 逐次 nearest_rank_percentile。"""

    def test_matches_single_call_random(self):
        rng = random.Random(99)
        for _ in range(30):
            values = [rng.randrange(10**9) for _ in range(rng.randrange(1, 60))]
            p_list = [p / 100.0 for p in rng.sample(
                range(1, 101), rng.randrange(1, 6))]
            expected = [nearest_rank_percentile(values, p)
                        for p in p_list]
            self.assertEqual(
                nearest_rank_percentile_many(values, p_list), expected)

    def test_out_of_order_indices(self):
        # p=99 的索引 > p=90 的索引：游标单调推进仍须返回各自正确值。
        values = list(range(1, 101))
        self.assertEqual(
            nearest_rank_percentile_many(values, [0.99, 0.50, 0.90, 0.99]),
            [99, 50, 90, 99])

    def test_empty_fails_closed(self):
        with self.assertRaises(Exception):
            nearest_rank_percentile_many([], [0.5])

    def test_invalid_p_fails_closed(self):
        with self.assertRaises(Exception):
            nearest_rank_percentile_many([1, 2, 3], [1.5])


# ---------------------------------------------------------------------------
# 合成 sh_1.0 run_dir（driver parity 用）
# ---------------------------------------------------------------------------

def build_sh10_run_dir() -> Path:
    run_dir = synthetic.make_run_dir("drv_parity")
    results = run_dir / "results"
    results.mkdir(parents=True, exist_ok=True)

    # -- [METRIC] 流：init + request 边界 + memory_anchor ------------------
    init = {"type": "init", "repo_variant": "astra-sim-sh_1.0",
            "manifest_path": str(run_dir / "metrics_manifest.json"),
            "detail_level": "full", "schema_version": 1}
    metric_records = [init]
    metric_records.append({
        "type": "request", "queue_index": 0,
        "request_id": "session_A_request_0",
        "prefill_start_ns": 150, "prefill_end_ns": 190})
    metric_records.append({
        "type": "memory_anchor", "subject_id": 0, "tick_ns": 120})
    metric_records.append({
        "type": "memory_anchor", "subject_id": 0, "tick_ns": 145})
    metric_records.append({
        "type": "request", "queue_index": 1,
        "request_id": "session_B_request_0",
        "prefill_start_ns": 20_000_000_150, "prefill_end_ns": 20_000_000_190})
    metric_records.append({
        "type": "memory_anchor", "subject_id": 1, "tick_ns": 20_000_000_120})
    lines = ["[METRIC] " + json.dumps(record, sort_keys=True)
             for record in metric_records]
    (run_dir / "cpp.log").write_text("\n".join(lines) + "\n",
                                     encoding="utf-8")

    # -- request_metrics.csv ----------------------------------------------
    synthetic.write_request_metrics(run_dir, [
        synthetic.request_row(
            queue_index="0", request_id="session_A_request_0",
            session_id="session_A", turn_index="0", request_type="human",
            arrival_ns="100", completion_ns="300", queue_ns="10",
            prefill_ns="40", decode_ns="150", e2e_ns="200",
            prefill_length="1", decode_length="1", prefix_len="0",
            kv_hit_state="no_history"),
        synthetic.request_row(
            queue_index="1", request_id="session_B_request_0",
            session_id="session_B", turn_index="1", request_type="tool",
            arrival_ns="20000000100", completion_ns="20000001300",
            queue_ns="10", prefill_ns="40", decode_ns="100", e2e_ns="200",
            prefill_length="1", decode_length="0", prefix_len="1",
            kv_hit_state="full"),
    ])

    # -- metrics_manifest.json（session 透传 + prefill_ranks）--------------
    (run_dir / "metrics_manifest.json").write_text(json.dumps({
        "schema_version": 1, "repo_variant": "astra-sim-sh_1.0",
        "manifest_source": "synthetic",
        "requests": [
            synthetic.manifest_request(
                "session_A_request_0", "session_A", 0, 0,
                prefill_ranks=[0, 1, 2, 3, 4, 5],
                human_time_ns=50),
            synthetic.manifest_request(
                "session_B_request_0", "session_B", 1, 1,
                prefill_ranks=[0, 1, 2, 3, 4, 5],
                tool_time_ns=30),
        ]}), encoding="utf-8")

    # -- token manifest（manifest.json）------------------------------------
    (run_dir / "manifest.json").write_text(json.dumps({
        "manifest_source": "synthetic", "repo_variant": "astra-sim-sh_1.0",
        "requests": [
            {"request_id": "session_A_request_0", "session_id": "session_A",
             "turn_index": 0, "queue_index": 0,
             "prefill_context_tokens": 1, "final_context_tokens": 2,
             "history_tokens_before": 0},
            {"request_id": "session_B_request_0", "session_id": "session_B",
             "turn_index": 1, "queue_index": 1,
             "prefill_context_tokens": 2, "final_context_tokens": 2,
             "history_tokens_before": 1},
        ]}), encoding="utf-8")

    # -- decision log（sh_1.0 字段族）--------------------------------------
    noc_migrate = {
        "kind": "noc_migrate", "reason": "synthetic",
        "trigger_request_id": "session_B_request_0",
        "source_instance_index": 0, "target_instance_index": 1,
        "total_bytes": 100,
        "shards": [{"bytes": 100, "noc_path": [0, 5, 1]}],
    }
    pd_transfer = {
        "kind": "noc_migrate", "reason": "synthetic",
        "source_instance_index": 0, "target_instance_index": 1,
        "total_bytes": 100,
        "shards": [{"bytes": 100, "noc_path": [0, 3, 1]}],
    }
    decision_log = [
        {"kind": "prefill", "request_id": "session_A_request_0",
         "seq": 1, "tick": 100, "decision": {
             "prefill_instance_index": 0, "history_transfer": None}},
        {"kind": "decode", "request_id": "session_A_request_0",
         "seq": 2, "tick": 200, "decision": {
             "decode_instance_index": 0,
             "prefill_decode_transfer": None}},
        {"kind": "completion", "request_id": "session_A_request_0",
         "seq": 3, "tick": 300, "decision": {"completion_evictions": []}},
        {"kind": "prefill", "request_id": "session_B_request_0",
         "seq": 4, "tick": 20_000_000_100, "decision": {
             "prefill_instance_index": 1,
             "history_transfer": noc_migrate}},
        {"kind": "decode", "request_id": "session_B_request_0",
         "seq": 5, "tick": 20_000_001_200, "decision": {
             "decode_instance_index": 1,
             "prefill_decode_transfer": pd_transfer}},
        {"kind": "completion", "request_id": "session_B_request_0",
         "seq": 6, "tick": 20_000_001_300, "decision": {"completion_evictions": []}},
    ]
    (results / "online_decision_log.jsonl").write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n"
                for record in decision_log), encoding="utf-8")

    # -- train_ledger（含 first_step 行以覆盖跳过打印）---------------------
    (results / "train_ledger.jsonl").write_text(
        json.dumps({"tick": 1100, "instance_index": 0, "first_step": True,
                    "train_id": "t0", "drains": []}, sort_keys=True) + "\n"
        + json.dumps({"tick": 20_000_001_400, "instance_index": 0, "first_step": False,
                      "train_id": "t0",
                      "drains": ["session_A_request_0",
                                 "session_B_request_0"]},
                     sort_keys=True) + "\n", encoding="utf-8")

    # -- trace_config（run_dir 本地优先；hardware 用仓内 validation 档）----
    (run_dir / "trace_config.csv").write_text(
        "kind,key,value,group_name,pg_name,ranks,description\n"
        "config,layers,1,,,,synthetic\n"
        "config,hidden_size,50,,,,synthetic\n"
        "config,bytes_per_elem,1,,,,synthetic\n"
        "config,local_hbm_capacity_profile,validation-160gib,,,,synthetic\n",
        encoding="utf-8")
    return run_dir


def _clear_slo_products(run_dir: Path) -> None:
    for name in SLO_PRODUCTS:
        path = run_dir / name
        if path.exists():
            path.unlink()


def _snapshot(run_dir: Path, target: Path) -> None:
    # 目标目录先清空（历史运行的陈旧产物会让 diff 出现幽灵差异——
    # SLO_PRODUCTS 产物名变更后尤其如此）。
    if target.is_dir():
        for stale in target.iterdir():
            if stale.is_file():
                stale.unlink()
    target.mkdir(parents=True, exist_ok=True)
    for name in SLO_PRODUCTS:
        path = run_dir / name
        if path.is_file():
            data = path.read_bytes()
            (target / name).write_bytes(data)


def _count_metric_lines(source: Path) -> int:
    count = 0
    with source.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("[METRIC] "):
                count += 1
    return count


def run_old_chain_replica(run_dir: Path) -> int:
    """旧 run_slo_postprocess.sh（9 子命令版）的等价复刻，产物/日志同构。

    逐字符对齐原 shell：G1-G5 门控 + run/ok/FAIL 行 + slo_postprocess.log
    组装（log 行/命令输出顺序/warn tee 语义）。
    """
    log_path = run_dir / "slo_postprocess.log"
    fail_path = run_dir / "slo_postprocess.FAIL"
    if fail_path.exists():
        fail_path.unlink()
    log_lines: list[str] = []

    def log(message: str) -> None:
        log_lines.append(f"[slo-postprocess] {message}")

    def flush_log() -> None:
        log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")

    def resolve_metric_source() -> Path | None:
        for name in ("cpp.log", "metrics.log", "cpp.log.gz"):
            candidate = run_dir / name
            if candidate.is_file():
                return candidate
        return None

    metric_src = resolve_metric_source()
    if metric_src is None:
        log(f"FAIL: no cpp.log/metrics.log/cpp.log.gz under {run_dir}")
        flush_log()
        with fail_path.open("a", encoding="utf-8") as handle:
            handle.write("input-resolve: no cpp.log/metrics.log/"
                         "cpp.log.gz (exit=1)\n")
        return 1
    if _count_metric_lines(metric_src) == 0:
        log("no [METRIC] lines (metrics detail=off?) — SLO extraction "
            "skipped (by design)")
        flush_log()
        return 0
    log(f"metric source: {metric_src} "
        f"({_count_metric_lines(metric_src)} [METRIC] lines)")

    have_rm = (run_dir / "request_metrics.csv").is_file()
    if not have_rm:
        log("request_metrics.csv absent (metrics detail=summary/off) — "
            "e2e-stats/backlog/session/warmup/restore_decomposition "
            "skipped (by design)")
    have_tl = (run_dir / "results" / "train_ledger.jsonl").is_file()
    if not have_tl:
        log("results/train_ledger.jsonl absent — load_imbalance.py "
            "skipped (by design; legacy variant produces no train ledger)")
    skip_hbm = (run_dir / "results" /
                "kv_event_payload_legacy.json").is_file()
    if skip_hbm:
        log("legacy variant detected "
            "(results/kv_event_payload_legacy.json) — hbm_watermark.py "
            "skipped (legacy allocator semantics not covered by "
            "session-KV replay)")

    steps = [
        ("slo_stats.py e2e-stats --extra-pct 90,95", have_rm,
         ["slo_stats.py", "e2e-stats", str(run_dir),
          "--extra-pct", "90,95"]),
        ("slo_stats.py backlog (per-event, no --bucket-ns)", have_rm,
         ["slo_stats.py", "backlog", str(run_dir)]),
        ("slo_stats.py session", have_rm,
         ["slo_stats.py", "session", str(run_dir)]),
        ("slo_stats.py warmup", have_rm,
         ["slo_stats.py", "warmup", str(run_dir)]),
        ("kv_cache_adapter.py", True,
         ["kv_cache_adapter.py", str(run_dir)]),
        ("load_imbalance.py", have_tl,
         ["load_imbalance.py", str(run_dir)]),
        ("restore_decomposition.py (detail=full)", have_rm,
         ["restore_decomposition.py", str(run_dir)]),
        ("hopbytes.py", True, ["hopbytes.py", str(run_dir)]),
        ("hbm_watermark.py", not skip_hbm,
         ["hbm_watermark.py", str(run_dir)]),
    ]
    slo_fail = False
    for label, enabled, argv in steps:
        if not enabled:
            continue
        log(f"run: {label}")
        proc = subprocess.run(
            [sys.executable, str(SLO_TOOLS_DIR / argv[0])] + argv[1:],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if proc.stdout:
            log_lines.append(proc.stdout.rstrip("\n"))
        if proc.returncode == 0:
            log(f"ok: {label}")
        else:
            log(f"FAIL: {label} (exit={proc.returncode}) — marked in "
                f"slo_postprocess.FAIL (simulation result NOT overturned)")
            with fail_path.open("a", encoding="utf-8") as handle:
                handle.write(f"{label} (exit={proc.returncode})\n")
            slo_fail = True
    if slo_fail:
        log("done WITH FAILURES (default=warn; see slo_postprocess.FAIL)")
        flush_log()
        return 1
    log("done: all steps passed")
    flush_log()
    return 0


def run_new_chain(run_dir: Path) -> int:
    proc = subprocess.run(
        ["bash", str(SLO_TOOLS_DIR.parent / "run_scripts" /
                     "run_slo_postprocess.sh"), str(run_dir)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return proc.returncode


class DriverParityTests(unittest.TestCase):
    """合成 sh_1.0 fixture：旧链复刻 vs 当前链（单遍 driver）逐字节。"""

    def _compare(self, tag: str, mutate=None, expect_rc: int = 0):
        run_dir = build_sh10_run_dir()
        try:
            if mutate is not None:
                mutate(run_dir)
            rc_old = run_old_chain_replica(run_dir)
            self.assertEqual(rc_old, expect_rc)
            old_out = run_dir.parent / f"{tag}_old_out"
            _snapshot(run_dir, old_out)
            _clear_slo_products(run_dir)
            rc_new = run_new_chain(run_dir)
            self.assertEqual(rc_new, expect_rc)
            new_out = run_dir.parent / f"{tag}_new_out"
            _snapshot(run_dir, new_out)
            diffs = []
            for name in sorted(set(os.listdir(old_out))
                               | set(os.listdir(new_out))):
                old_bytes = ((old_out / name).read_bytes()
                             if (old_out / name).is_file() else None)
                new_bytes = ((new_out / name).read_bytes()
                             if (new_out / name).is_file() else None)
                if old_bytes != new_bytes:
                    diffs.append(name)
            self.assertEqual(diffs, [], f"产物字节差: {diffs}")
            return old_out, new_out
        finally:
            pass

    def test_happy_path_byte_identical(self):
        old_out, _ = self._compare("happy")
        produced = sorted(os.listdir(old_out))
        self.assertEqual(produced, sorted(list(SLO_PRODUCTS[:13])
                                          + ["slo_postprocess.log"]))

    def test_g3_missing_request_metrics(self):
        self._compare("g3", lambda rd: (rd / "request_metrics.csv").unlink())

    def test_failure_token_manifest_missing(self):
        # 仅 hbm_watermark 失败（exit=3 链语义：整链 rc=1 + FAIL 条目）。
        def mutate(rd):
            (rd / "manifest.json").unlink()
        self._compare("notoken", mutate, expect_rc=1)


if __name__ == "__main__":
    unittest.main()
