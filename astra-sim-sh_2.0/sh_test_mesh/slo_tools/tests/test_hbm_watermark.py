#!/usr/bin/env python3
"""hbm_watermark.py 单测：合成 ledger 手算断言（B2 线5）。

运行：python3 sh_test_mesh/slo_tools/tests/test_hbm_watermark.py
  或  python3 -m unittest discover -s sh_test_mesh/slo_tools/tests -v

合成口径（全仓一致、手算可验）：
  * trace_config：layers=1、hidden_size=50、bytes_per_elem=1 →
    COEF = 2·1·50·1 = 100 B/token（实例合计）；
  * hardware 容量档 test-64b = 1000 B/NPU × prefill_ranks 6 = 6000 B/实例；
  * slo_params_manifest：watermark_sample_period_ns = 10 ns（B4 未推导的
    生产 manifest 保持 null → 5,000,000 ns 临时锚点，另有专测）。

主手算场景（任务书「admit 100B@t0、evict 40B@t1」对应）：
  会话 A：prefill@t0 增至 f(1)=100B；decode@t10 增至 f(2)=200B；
  会话 B：prefill@t30 准入，触发对 A 的部分逐出 40B（A 余 160B）+ B 100B
  → 时间线 100 → 200 → 260；桶长 10ns 手算各桶末占用/峰值/逐出叠加。
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
SLO_TOOLS_DIR = TESTS_DIR.parent
sys.path.insert(0, str(SLO_TOOLS_DIR))

import hbm_watermark as hw  # noqa: E402
from slo_common import SloToolError  # noqa: E402

COEF = 100  # B/token（layers=1, hidden=50, bytes_per_elem=1）
CAPACITY_PER_INSTANCE = 6000  # 1000 B/NPU × 6 NPU


def make_run_dir(tag: str) -> Path:
    return Path(tempfile.mkdtemp(prefix=f"slo_hbm_{tag}_"))


def write_fixture(run_dir: Path, *, repo_variant: str, records: list[dict],
                  token_requests: list[dict], bucket_ns=10,
                  bucket_null: bool = False, npu_bytes: int = 1000) -> Path:
    """落一个最小可重放 run_dir；返回 run_dir。

    records 为 online_decision_log.jsonl 的行（dict）；token_requests 为
    manifest.json 的 requests 数组成员。
    """
    results = run_dir / "results"
    results.mkdir(parents=True, exist_ok=True)
    (run_dir / "cpp.log").write_text(
        '[METRIC] {"type":"init","repo_variant":"%s",'
        '"manifest_path":"%s","detail_level":"summary","schema_version":1}\n'
        % (repo_variant, run_dir / "metrics_manifest.json"),
        encoding="utf-8")
    with (results / "online_decision_log.jsonl").open(
            "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    (run_dir / "manifest.json").write_text(json.dumps({
        "manifest_source": "synthetic", "repo_variant": repo_variant,
        "requests": token_requests}), encoding="utf-8")
    (run_dir / "metrics_manifest.json").write_text(json.dumps({
        "manifest_source": "synthetic", "repo_variant": repo_variant,
        "schema_version": 1,
        "requests": [{
            "queue_index": index,
            "request_id": entry["request_id"],
            "session_id": entry["session_id"],
            "turn_index": entry.get("turn_index", 0),
            "arrival": {"kind": "absolute", "value_ns": 0},
            "prefill_instance": 0,
            "prefill_ranks": [0, 1, 2, 3, 4, 5],
            "decode_instance": 0,
            "decode_ranks": [0, 1, 2, 3, 4, 5],
        } for index, entry in enumerate(token_requests)]}),
        encoding="utf-8")
    (run_dir / "trace_config.csv").write_text(
        "kind,key,value,group_name,pg_name,ranks,description\n"
        "config,layers,1,,,,synthetic\n"
        "config,hidden_size,50,,,,synthetic\n"
        "config,num_heads,2,,,,synthetic\n"
        "config,bytes_per_elem,1,,,,synthetic\n"
        "config,local_hbm_capacity_profile,test-64b,,,,synthetic\n",
        encoding="utf-8")
    hardware = run_dir / "hardware.json"
    hardware.write_text(json.dumps({
        "schema-version": 1,
        "local-hbm": {"capacity-profiles": {
            "test-64b": {"bytes": npu_bytes, "label": "synthetic",
                         "note": "synthetic"}}}}), encoding="utf-8")
    entry_value = None if bucket_null else bucket_ns
    manifest = run_dir / "slo_params_manifest.json"
    manifest.write_text(json.dumps({
        "schema_version": 1,
        "params": {"watermark_sample_period_ns": {
            "value": entry_value, "unit": "ns",
            "derivation_program": "synthetic test",
            "evidence": None, "rationale": None}}}), encoding="utf-8")
    return run_dir


def token_row(request_id: str, session_id: str, history: int, prefill: int,
              final: int) -> dict:
    return {"request_id": request_id, "session_id": session_id,
            "turn_index": 0, "queue_index": 0,
            "prefill_length": prefill - history,
            "decode_length": final - prefill,
            "history_tokens_before": history,
            "prefill_context_tokens": prefill,
            "final_context_tokens": final}


def prefill_record(request_id: str, tick: int, instance: int = 0,
                   decision: dict | None = None) -> dict:
    return {"kind": "prefill", "request_id": request_id, "priority": 0,
            "seq": prefill_record.next_seq, "tick": tick,
            "decision": {"prefill_instance_index": instance,
                         "history_action": "NO_HISTORY",
                         "history_transfer_bytes": 0,
                         "history_source_instance_index": None,
                         "history_cache_state_before": "ABSENT",
                         "admission_evictions": [],
                         "decode_target_evictions": [],
                         "effective_prefill_tokens": 1, **(decision or {})}}


prefill_record.next_seq = 1


def decode_record(request_id: str, tick: int, instance: int = 0,
                  decision: dict | None = None) -> dict:
    decode_record.next_seq += 1
    return {"kind": "decode", "request_id": request_id, "priority": 0,
            "seq": decode_record.next_seq, "tick": tick,
            "decision": {"decode_instance_index": instance,
                         "prefill_decode_transfer": None,
                         **(decision or {})}}


decode_record.next_seq = 1


def completion_record(request_id: str, tick: int,
                      decision: dict | None = None) -> dict:
    completion_record.next_seq += 1
    return {"kind": "completion", "request_id": request_id, "priority": 0,
            "seq": completion_record.next_seq, "tick": tick,
            "decision": {"completion_evictions": [],
                         "kv_instance_after_completion": 0,
                         "kv_state_after_completion": "RESIDENT",
                         **(decision or {})}}


completion_record.next_seq = 1


def reset_seq() -> None:
    prefill_record.next_seq = 1
    decode_record.next_seq = 1
    completion_record.next_seq = 1


def base_handcalc_records() -> tuple[list[dict], list[dict]]:
    """手算场景：A 100B@t0 → 200B@t10；B@t30 部分逐出 A 40B + 自身 100B。"""
    reset_seq()
    records = [
        prefill_record("session_A_request_0", 0,
                       decision={"effective_prefill_tokens": 1}),
        decode_record("session_A_request_0", 10,
                     decision={"effective_prefill_tokens": 1}),
        completion_record("session_A_request_0", 20),
        prefill_record("session_B_request_0", 30, decision={
            "admission_evictions": [{
                "time_ns": 30, "phase": "prefill", "reason": "synthetic",
                "trigger_request_id": "session_B_request_0",
                "victim_session_id": "session_A", "victim_instance_index": 0,
                "victim_last_completion_ns": 20, "context_tokens": 2,
                "shard_bytes": [40]}]}),
        decode_record("session_B_request_0", 40),
        completion_record("session_B_request_0", 50),
    ]
    tokens = [
        token_row("session_A_request_0", "session_A", 0, 1, 2),
        token_row("session_B_request_0", "session_B", 0, 1, 1),
    ]
    return records, tokens


def run_tool(run_dir: Path, *extra: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SLO_TOOLS_DIR / "hbm_watermark.py"),
         str(run_dir), "--manifest", str(run_dir / "slo_params_manifest.json"),
         "--hardware-config", str(run_dir / "hardware.json"),
         "--json", str(run_dir / "summary.json"),
         "-o", str(run_dir / "series.csv"),
         "--instances-csv", str(run_dir / "instances.csv"), *extra],
        capture_output=True, text=True, check=False)


class HandCalcReplayTests(unittest.TestCase):
    """桶时序/峰值/均值/逐出/违规计数的手算断言（任务书口径）。"""

    def test_face_handcalc_series_and_summary(self):
        records, tokens = base_handcalc_records()
        run_dir = make_run_dir("face")
        write_fixture(run_dir, repo_variant="astra-sim-face",
                      records=records, token_requests=tokens)
        proc = run_tool(run_dir)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        summary = json.loads((run_dir / "summary.json").read_text())
        # 手算：占用时间线 t0=100（A f(1)），t10=200（A 长到 f(2)），
        # t30 = 200 − 40（A 部分逐出）+ 100（B f(1)）= 260。
        self.assertEqual(summary["violation_events"], 0)
        self.assertEqual(summary["coef_bytes_per_token"], COEF)
        self.assertEqual(summary["capacity_bytes_per_instance"],
                         CAPACITY_PER_INSTANCE)
        self.assertEqual(summary["total_evict_events"], 1)
        self.assertEqual(summary["total_evict_bytes"], 40)
        self.assertEqual(summary["actions"]["partial_evictions"], 1)
        instance = summary["instances"]["0"]
        self.assertEqual(instance["peak_occupancy_bytes"], 260)
        self.assertEqual(instance["residual_occupancy_bytes"], 260)
        # 活动窗 [first_event, last_event] = [0,30]；面积 = 100×10 +
        # 200×20 = 5000 → 均值 5000/30 ≈ 166.667（先整数后除）。
        self.assertAlmostEqual(instance["mean_occupancy_bytes"],
                               5000 / 30, places=6)
        self.assertEqual(instance["duration_ns"], 30)
        # 桶时序（桶长 10ns，事件跨度 [0,30] → 4 个网格桶，t30 入桶 3）：
        #   b0[0,10) 末 100；b1[10,20) 末 200；b2[20,30) 末 200；
        #   b3[30,40) 末 260，逐出 1 次/40B。
        with (run_dir / "series.csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        by_bucket = {int(r["bucket_index"]): r for r in rows}
        self.assertEqual(sorted(by_bucket), [0, 1, 2, 3])
        self.assertEqual(int(by_bucket[0]["occupancy_end_bytes"]), 100)
        self.assertEqual(int(by_bucket[1]["occupancy_end_bytes"]), 200)
        self.assertEqual(int(by_bucket[2]["occupancy_end_bytes"]), 200)
        self.assertEqual(int(by_bucket[3]["occupancy_end_bytes"]), 260)
        self.assertEqual(by_bucket[3]["evict_events"], "1")
        self.assertEqual(by_bucket[3]["evict_bytes"], "40")
        for row in rows:
            self.assertEqual(row["repo_variant"], "astra-sim-face")
        self.assertEqual(summary["bucket_ns"], 10)
        self.assertFalse(summary["bucket_ns_provisional"])

    def test_violation_exits_nonzero_and_marks_red(self):
        records, tokens = base_handcalc_records()
        run_dir = make_run_dir("viol")
        # 每 NPU 容量 10B → 实例 60B：t0 占用 100B > 60B → 违规。
        write_fixture(run_dir, repo_variant="astra-sim-face",
                      records=records, token_requests=tokens, npu_bytes=10)
        proc = run_tool(run_dir)
        self.assertEqual(proc.returncode, hw.EXIT_VIOLATION)
        self.assertIn("容量违规", proc.stderr)
        summary = json.loads((run_dir / "summary.json").read_text())
        self.assertGreater(summary["violation_events"], 0)
        self.assertEqual(summary["violation_check"],
                         "occupancy_gt_capacity")

    def test_provisional_bucket_anchor(self):
        records, tokens = base_handcalc_records()
        run_dir = make_run_dir("prov")
        write_fixture(run_dir, repo_variant="astra-sim-face",
                      records=records, token_requests=tokens,
                      bucket_null=True)
        proc = run_tool(run_dir)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("临时锚点", proc.stderr)
        summary = json.loads((run_dir / "summary.json").read_text())
        self.assertEqual(summary["bucket_ns"], hw.PROVISIONAL_BUCKET_NS)
        self.assertTrue(summary["bucket_ns_provisional"])
        with (run_dir / "instances.csv").open(newline="") as handle:
            row = next(csv.DictReader(handle))
        self.assertEqual(row["bucket_ns"], str(hw.PROVISIONAL_BUCKET_NS))
        self.assertEqual(row["bucket_ns_provisional"], "true")


class PerRepoSemanticsTests(unittest.TestCase):
    """逐仓字段映射与账本断言对账（local_hit 信息量 / S3 前缀 / S2 降级）。"""

    def test_s1_local_hit_bytes_are_informational(self):
        """S1 local_hit.total_bytes = f(h) 是复用大小而非搬移——不得翻倍。"""
        reset_seq()
        records = [
            prefill_record("s_r0", 0, decision={}),
            decode_record("s_r0", 10),
            completion_record("s_r0", 20),
            # 下一轮：local_hit（同实例复用），total_bytes=f(2)=200 信息量。
            {"kind": "prefill", "request_id": "s_r1", "priority": 0,
             "seq": 4, "tick": 30,
             "decision": {"prefill_instance_index": 0,
                          "admission_time_ns": 30,
                          "history_transfer": {
                              "kind": "local_hit", "phase": "history",
                              "reason": "history_local_reuse",
                              "session_id": "s", "shards": [],
                              "source_instance_index": 0,
                              "target_instance_index": 0,
                              "total_bytes": 200},
                          "history_evictions": [], "prefill_evictions": []}},
            decode_record("s_r1", 40,
                          decision={"prefill_decode_transfer": None,
                                    "decode_evictions": []}),
            completion_record("s_r1", 50),
        ]
        records[3]["decision"]["prefill_instance_index"] = 0
        tokens = [token_row("s_r0", "s", 0, 1, 2),
                  token_row("s_r1", "s", 2, 3, 3)]
        run_dir = make_run_dir("s1hit")
        write_fixture(run_dir, repo_variant="astra-sim-sh_1.0",
                      records=records, token_requests=tokens)
        proc = run_tool(run_dir)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        summary = json.loads((run_dir / "summary.json").read_text())
        # r0: +100 → +100(decode grow f(2)) = 200；r1 local_hit 不加，
        # prefill grow 到 f(3)=300（+100）→ 峰值 300（若误当搬移会 >300）。
        self.assertEqual(summary["instances"]["0"]["peak_occupancy_bytes"],
                         300)
        # restore_none = r0 无历史（0 bytes）+ r1 local_hit，共 2。
        self.assertEqual(summary["actions"]["restore_none"], 2)
        self.assertEqual(summary["anomalies"]
                         ["local_hit_location_mismatch"], 0)

    def test_s3_prefix_reconcile(self):
        """S3 半层恢复：恢复前本地应持有 f(h)−bytes，超出部分对账扣减。"""
        reset_seq()
        records = [
            prefill_record("s_r0", 0),
            decode_record("s_r0", 10),
            # completion 标 partial_hbm_remote（suffix 半层静默释放，不落盘）
            completion_record("s_r0", 20, decision={
                "kv_location_after_completion": "partial_hbm_remote"}),
            # 半层恢复：bytes = f(2)/2 = 100（suffix 回载）
            prefill_record("s_r1", 30, decision={
                "history_transfer_bytes": 100,
                "history_source_instance_index": 0,
                "history_tokens_before": 2, "history_tokens_discarded": 0,
                "prefill_context_tokens": 2}),
            decode_record("s_r1", 40),
            completion_record("s_r1", 50),
        ]
        tokens = [token_row("s_r0", "s", 0, 1, 2),
                  token_row("s_r1", "s", 2, 2, 2)]
        run_dir = make_run_dir("s3p")
        write_fixture(run_dir, repo_variant="astra-sim-sh_3.0",
                      records=records, token_requests=tokens)
        proc = run_tool(run_dir)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        summary = json.loads((run_dir / "summary.json").read_text())
        # t0 +100 → t10 +100 = 200；t20 静默半层释放（无事件）；t30 恢复：
        # 期望恢复前本地 = f(2)−100 = 100，跟踪 200 → 对账扣 100，
        # 再回载 +100 → 200（与真实 manager 的 prefix+suffix=200 一致）。
        self.assertEqual(
            summary["actions"]["restore_prefix_reconciled"], 1)
        self.assertEqual(
            summary["actions"]["restore_prefix_reconcile_bytes"], 100)
        self.assertEqual(summary["instances"]["0"]
                         ["peak_occupancy_bytes"], 200)
        self.assertEqual(summary["instances"]["0"]
                         ["residual_occupancy_bytes"], 200)

    def test_s2_count_only_degraded(self):
        """S2 只有 *_eviction_count：未归因逐出计数 + 上界重建 + 检查降级。"""
        reset_seq()
        records = [
            prefill_record("s_r0", 0),
            decode_record("s_r0", 10, decision={
                "prefill_instance_index": 0,
                "prefill_decode_transfer_bytes": 100,
                "decode_eviction_count": 2}),
            completion_record("s_r0", 20, decision={
                "completion_eviction_count": 1,
                "kv_location_after_completion": "remote_memory",
                "reserve_unmet_ranks": []}),
        ]
        tokens = [token_row("s_r0", "s", 0, 1, 1)]
        run_dir = make_run_dir("s2c")
        write_fixture(run_dir, repo_variant="astra-sim-sh_2.0",
                      records=records, token_requests=tokens)
        proc = run_tool(run_dir)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        summary = json.loads((run_dir / "summary.json").read_text())
        self.assertEqual(summary["eviction_coverage"], "count_only")
        self.assertFalse(summary["occupancy_valid"])
        self.assertEqual(summary["violation_check"],
                         "degraded_upper_bound_no_certification")
        self.assertEqual(summary["actions"]["unattributed_evictions"], 3)
        # 自身 remote 迁移可归因：completion 时扣 100。
        self.assertEqual(summary["actions"]["own_relocation_remote"], 1)
        self.assertEqual(summary["instances"]["0"]
                         ["residual_occupancy_bytes"], 0)

    def test_s2_full_coverage_new_fields(self):
        """B3-6：history_transfers/*_evictions 在场 → full 口径。

        手算（COEF=100）：A f(1)=100@t0 → f(2)=200@t10；B 准入@t30 逐出
        A 40（余 160）+ remote_load 恢复 100 + 增长到 f(2)=200（+100）；
        B decode@t40 增至 f(3)=300；completion 的 kv_location_after_
        completion=remote_memory 在 full 口径下【忽略】（completion_
        evictions 已含自身释放，归因会双重扣减）→ 末态 160+300=460。
        """
        reset_seq()
        records = [
            prefill_record("a_r0", 0, decision={
                "history_transfer_bytes": 0, "history_eviction_count": 0,
                "history_evictions": [], "prefill_evictions": [],
                "history_transfers": []}),
            decode_record("a_r0", 10, decision={
                "prefill_instance_index": 0,
                "prefill_decode_transfer_bytes": 100,
                "decode_eviction_count": 0, "decode_evictions": []}),
            completion_record("a_r0", 20, decision={
                "completion_eviction_count": 0, "completion_evictions": [],
                "kv_location_after_completion": "local_hbm"}),
            prefill_record("b_r0", 30, decision={
                "history_transfer_bytes": 100, "history_eviction_count": 1,
                "history_evictions": [{
                    "kind": "remote_store", "reason": "reserve_full_fallback",
                    "session_id": "session_A", "total_bytes": 40,
                    "source_instance_index": 0,
                    "target_instance_index": None,
                    "layer_start": 0, "layer_end": 1}],
                "prefill_evictions": [],
                "history_transfers": [{
                    "kind": "remote_load", "reason": "history_restore",
                    "session_id": "session_B", "total_bytes": 100,
                    "source_instance_index": None,
                    "target_instance_index": 0,
                    "layer_start": 0, "layer_end": 1}]}),
            decode_record("b_r0", 40, decision={
                "prefill_instance_index": 0,
                "prefill_decode_transfer_bytes": 200,
                "decode_eviction_count": 0, "decode_evictions": []}),
            completion_record("b_r0", 50, decision={
                "completion_eviction_count": 0, "completion_evictions": [],
                "kv_location_after_completion": "remote_memory"}),
        ]
        tokens = [token_row("a_r0", "session_A", 0, 1, 2),
                  token_row("b_r0", "session_B", 1, 2, 3)]
        run_dir = make_run_dir("s2f")
        write_fixture(run_dir, repo_variant="astra-sim-sh_2.0",
                      records=records, token_requests=tokens)
        proc = run_tool(run_dir)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        summary = json.loads((run_dir / "summary.json").read_text())
        self.assertEqual(summary["eviction_coverage"], "full")
        self.assertTrue(summary["occupancy_valid"])
        self.assertEqual(summary["violation_check"], "occupancy_gt_capacity")
        self.assertEqual(summary["violation_events"], 0)
        self.assertEqual(summary["actions"]["unattributed_evictions"], 0)
        self.assertEqual(summary["actions"]["evictions"], 1)
        self.assertEqual(summary["actions"]["evict_bytes"], 40)
        self.assertEqual(summary["actions"]["partial_evictions"], 1)
        self.assertEqual(summary["actions"]["restore_remote_add"], 1)
        # completion 归因字段被忽略：无双重扣减。
        self.assertEqual(summary["actions"]["own_relocation_remote"], 0)
        self.assertEqual(summary["instances"]["0"]
                         ["residual_occupancy_bytes"], 460)
        self.assertEqual(summary["instances"]["0"]
                         ["peak_occupancy_bytes"], 460)

    def test_s2_full_partial_two_stage_migration(self):
        """B3-6：partial 两段式迁移逐段对账（标量聚合会击穿下界）。

        手算：B turn0 f(1)=100@t5(i1) → f(2)=200@t6；completion@t7 自身
        suffix 逐出 100（i1 余 100）；turn1@t30 跨实例迁 i0：prefix
        noc_migrate 100（i1 -100 / i0 +100）+ suffix remote_load 100（i0
        +100，ratio 0.5）→ i0 200；decode@t40 增至 f(3)=300。
        """
        reset_seq()
        records = [
            prefill_record("b_r0", 5, instance=1, decision={
                "history_transfer_bytes": 0, "history_eviction_count": 0,
                "history_evictions": [], "prefill_evictions": [],
                "history_transfers": []}),
            decode_record("b_r0", 6, decision={
                "decode_instance_index": 1,
                "prefill_instance_index": 1,
                "prefill_decode_transfer_bytes": 100,
                "decode_eviction_count": 0, "decode_evictions": []}),
            completion_record("b_r0", 7, decision={
                "completion_eviction_count": 1,
                "completion_evictions": [{
                    "kind": "remote_store",
                    "reason": "enforce_reserve_suffix_half",
                    "session_id": "session_B", "total_bytes": 100,
                    "source_instance_index": 1,
                    "target_instance_index": None,
                    "layer_start": 0, "layer_end": 1}],
                "kv_location_after_completion": "partial_hbm_remote"}),
            prefill_record("b_r1", 30, decision={
                "history_transfer_bytes": 200, "history_eviction_count": 0,
                "history_evictions": [], "prefill_evictions": [],
                "history_transfers": [{
                    "kind": "noc_migrate",
                    "reason": "history_prefix_target_capacity",
                    "session_id": "session_B", "total_bytes": 100,
                    "source_instance_index": 1,
                    "target_instance_index": 0,
                    "layer_start": 0, "layer_end": 1},
                    {"kind": "remote_load", "reason": "history_restore",
                     "session_id": "session_B", "total_bytes": 100,
                     "source_instance_index": None,
                     "target_instance_index": 0,
                     "layer_start": 0, "layer_end": 1}]}),
            decode_record("b_r1", 40, decision={
                "prefill_instance_index": 0,
                "prefill_decode_transfer_bytes": 200,
                "decode_eviction_count": 0, "decode_evictions": []}),
            completion_record("b_r1", 50, decision={
                "completion_eviction_count": 0, "completion_evictions": [],
                "kv_location_after_completion": "local_hbm"}),
        ]
        tokens = [token_row("b_r0", "session_B", 0, 1, 2),
                  token_row("b_r1", "session_B", 2, 2, 3)]
        tokens[1]["turn_index"] = 1
        run_dir = make_run_dir("s2p")
        write_fixture(run_dir, repo_variant="astra-sim-sh_2.0",
                      records=records, token_requests=tokens)
        proc = run_tool(run_dir)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        summary = json.loads((run_dir / "summary.json").read_text())
        self.assertEqual(summary["eviction_coverage"], "full")
        self.assertEqual(summary["violation_events"], 0)
        self.assertEqual(summary["actions"]["evictions"], 1)
        self.assertEqual(summary["actions"]["partial_evictions"], 1)
        self.assertEqual(summary["actions"]["restore_move"], 1)
        self.assertEqual(summary["actions"]["restore_local_add"], 1)
        self.assertEqual(summary["actions"]["restore_prefix_reconciled"], 0)
        self.assertEqual(sum(summary["anomalies"].values()), 0)
        self.assertEqual(summary["restore_bytes_ratio_histogram"]
                         .get("0.500"), 2)
        self.assertEqual(summary["instances"]["0"]
                         ["residual_occupancy_bytes"], 300)
        self.assertEqual(summary["instances"]["1"]
                         ["residual_occupancy_bytes"], 0)

    def test_w_recompute_state_reconciles_silent_eviction(self):
        """W RECOMPUTE + state_before=EVICTED：跟踪驻留须在恢复点对账清零。"""
        reset_seq()
        records = [
            prefill_record("s_r0", 0),
            decode_record("s_r0", 10),
            completion_record("s_r0", 20, decision={
                "terminal_kv_release_at_completion": False,
                "kv_state_after_completion": "RESIDENT"}),
            # 静默逐出（真实 manager 已清）+ RECOMPUTE 重建
            prefill_record("s_r1", 30, decision={
                "history_action": "RECOMPUTE",
                "history_cache_state_before": "EVICTED",
                "history_transfer_bytes": 0,
                "history_recompute_tokens": 2}),
            decode_record("s_r1", 40),
            completion_record("s_r1", 50),
        ]
        tokens = [token_row("s_r0", "s", 0, 1, 2),
                  token_row("s_r1", "s", 2, 3, 3)]
        run_dir = make_run_dir("wrec")
        write_fixture(run_dir, repo_variant="astra-sim-wscllm",
                      records=records, token_requests=tokens)
        proc = run_tool(run_dir)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        summary = json.loads((run_dir / "summary.json").read_text())
        self.assertEqual(
            summary["actions"]["silent_evictions_reconciled"], 1)
        self.assertEqual(summary["actions"]["silent_eviction_bytes"], 200)
        # r1 重算后 grow 到 f(3)=300；无 RECOMPUTE 对账时会先 fail（200>残留）。
        self.assertEqual(summary["instances"]["0"]
                         ["peak_occupancy_bytes"], 300)


class FailClosedTests(unittest.TestCase):
    def test_missing_token_manifest_fails(self):
        records, tokens = base_handcalc_records()
        run_dir = make_run_dir("fc1")
        write_fixture(run_dir, repo_variant="astra-sim-face",
                      records=records, token_requests=tokens)
        (run_dir / "manifest.json").unlink()
        proc = run_tool(run_dir)
        self.assertEqual(proc.returncode, 2)
        self.assertIn("token manifest", proc.stderr)

    def test_duplicate_prefill_decision_fails(self):
        records, tokens = base_handcalc_records()
        records.append(dict(records[0], seq=99))
        run_dir = make_run_dir("fc2")
        write_fixture(run_dir, repo_variant="astra-sim-face",
                      records=records, token_requests=tokens)
        proc = run_tool(run_dir)
        self.assertEqual(proc.returncode, 2)
        self.assertIn("两次", proc.stderr)

    def test_evicting_untracked_session_fails(self):
        """逐出对象不在跟踪态（漏记进入/重复逐出）→ 结构错，非零退出。"""
        reset_seq()
        records = [
            prefill_record("s_r0", 0, decision={
                "admission_evictions": [{
                    "time_ns": 0, "phase": "prefill", "reason": "synthetic",
                    "trigger_request_id": "s_r0", "victim_session_id": "ghost",
                    "victim_instance_index": 0,
                    "victim_last_completion_ns": 0, "context_tokens": 1,
                    "shard_bytes": [50]}]}),
            decode_record("s_r0", 10),
            completion_record("s_r0", 20),
        ]
        tokens = [token_row("s_r0", "s", 0, 1, 1)]
        run_dir = make_run_dir("fc3")
        write_fixture(run_dir, repo_variant="astra-sim-face",
                      records=records, token_requests=tokens)
        proc = run_tool(run_dir)
        self.assertEqual(proc.returncode, 2)
        self.assertIn("不在跟踪态", proc.stderr)

    def test_negative_occupancy_fails(self):
        """多扣（会话跟踪实例与实际占用错位）→ 负占用 fail-closed。

        构造：A 驻留实例 0；下一轮 prefill 决策谎报 LOCAL_HIT 到实例 1
        （local_hit_location_mismatch），会话归属改到 1 而 bytes 仍挂在
        实例 0 的占用上；随后 completion 从实例 1 逐出 A 全量 → 实例 1
        占用转负 → 结构错拒绝输出。
        """
        reset_seq()
        records = [
            prefill_record("sA_r0", 0, instance=0),
            decode_record("sA_r0", 10, instance=0),
            completion_record("sA_r0", 20),
            prefill_record("sA_r1", 30, instance=1, decision={
                "history_action": "LOCAL_HIT",
                "history_cache_state_before": "RESIDENT",
                "history_transfer_bytes": 0,
                "history_source_instance_index": 0,
                "effective_prefill_tokens": 1}),
            decode_record("sA_r1", 40, instance=1),
            completion_record("sA_r1", 50, decision={
                "completion_evictions": [{
                    "time_ns": 50, "phase": "completion",
                    "reason": "synthetic", "trigger_request_id": "sA_r1",
                    "victim_session_id": "sA", "victim_instance_index": 1,
                    "victim_last_completion_ns": 20,
                    "context_tokens": 1, "shard_bytes": [100]}]}),
        ]
        tokens = [token_row("sA_r0", "sA", 0, 1, 1),
                  token_row("sA_r1", "sA", 1, 1, 1)]
        run_dir = make_run_dir("fc4")
        write_fixture(run_dir, repo_variant="astra-sim-face",
                      records=records, token_requests=tokens)
        proc = run_tool(run_dir)
        self.assertEqual(proc.returncode, 2)
        self.assertIn("负占用", proc.stderr)

    def test_unknown_repo_variant_fails(self):
        records, tokens = base_handcalc_records()
        run_dir = make_run_dir("fc5")
        write_fixture(run_dir, repo_variant="astra-sim-unknown",
                      records=records, token_requests=tokens)
        proc = run_tool(run_dir)
        self.assertEqual(proc.returncode, 2)
        self.assertIn("未登记的 repo_variant", proc.stderr)

    def test_capacity_missing_degrades_to_peak_record(self):
        records, tokens = base_handcalc_records()
        run_dir = make_run_dir("fc6")
        write_fixture(run_dir, repo_variant="astra-sim-face",
                      records=records, token_requests=tokens)
        (run_dir / "hardware.json").unlink()
        proc = run_tool(run_dir, "--trace-config",
                        str(run_dir / "trace_config.csv"))
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("capacity=NA", proc.stderr)
        summary = json.loads((run_dir / "summary.json").read_text())
        self.assertIsNone(summary["capacity_bytes_per_instance"])
        self.assertEqual(summary["violation_check"],
                         "degraded_peak_recorded")
        self.assertEqual(summary["instances"]["0"]["peak_occupancy_bytes"],
                         260)


class UnitHelperTests(unittest.TestCase):
    def test_series_columns_frozen(self):
        self.assertEqual(
            ",".join(hw.SERIES_COLUMNS),
            "repo_variant,instance_index,bucket_index,bucket_start_ns,"
            "bucket_end_ns,occupancy_end_bytes,"
            "occupancy_peak_in_bucket_bytes,evict_events,evict_bytes")
        self.assertEqual(
            ",".join(hw.INSTANCE_COLUMNS),
            "repo_variant,instance_index,eviction_coverage,occupancy_valid,"
            "capacity_bytes,capacity_source,bucket_ns,"
            "bucket_ns_provisional,first_event_ns,last_event_ns,"
            "duration_ns,peak_occupancy_bytes,mean_occupancy_bytes,"
            "residual_occupancy_bytes,evict_events,evict_bytes,"
            "violation_events")

    def test_repo_variants_registered(self):
        self.assertEqual(
            sorted(hw.REPO_VARIANTS),
            ["astra-sim-face", "astra-sim-sh_1.0", "astra-sim-sh_2.0",
             "astra-sim-sh_3.0", "astra-sim-wscllm"])

    def test_load_bucket_ns_null_anchor(self):
        manifest = {"params": {"watermark_sample_period_ns": {"value": None}}}
        bucket_ns, provisional = hw.load_bucket_ns(manifest)
        self.assertEqual(bucket_ns, 5_000_000)
        self.assertTrue(provisional)
        manifest["params"]["watermark_sample_period_ns"]["value"] = 123
        bucket_ns, provisional = hw.load_bucket_ns(manifest)
        self.assertEqual(bucket_ns, 123)
        self.assertFalse(provisional)
        manifest["params"]["watermark_sample_period_ns"]["value"] = -5
        with self.assertRaises(SloToolError):
            hw.load_bucket_ns(manifest)


if __name__ == "__main__":
    unittest.main(verbosity=2)
