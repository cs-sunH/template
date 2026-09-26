#!/usr/bin/env python3
"""hbm_watermark.py 单测：合成 ledger 手算断言（B2 线5）+ P1 语义（四层
可信度/事件流 stats/RLE 无损/行预算/三口径锚定数值）。

运行：python3 sh_test_mesh/slo_tools/tests/test_hbm_watermark.py
  或  python3 -m unittest discover -s sh_test_mesh/slo_tools/tests -v

合成口径（全仓一致、手算可验）：
  * trace_config：layers=1、hidden_size=50、bytes_per_elem=1 →
    COEF = 2·1·50·1 = 100 B/token（实例合计）；
  * hardware 容量档 test-64b = 1000 B/NPU × prefill_ranks 6 = 6000 B/实例；
  * slo_params_manifest：watermark_sample_period_ns = 10 ns（生产 manifest
    已填推导值 5,000,000 ns 且不得回退 null——test_slo_contract.
    test_repo_manifest_params_filled_b4_frozen 钉住；null→5,000,000 ns
    临时锚点语义仅存于合成 fixture 路径，另有专测）；
    watermark_series_row_budget = 5,000,000（P1-④ 绘图行预算，另有小预算
    专测）。

主手算场景（任务书「admit 100B@t0、evict 40B@t1」对应）：
  会话 A：prefill@t0 增至 f(1)=100B；decode@t10 增至 f(2)=200B；
  会话 B：prefill@t30 准入，触发对 A 的部分逐出 40B（A 余 160B）+ B 100B
  → 时间线 100 → 200 → 260；桶长 10ns 手算各桶末占用/峰值/逐出叠加。

P1（2026-08-30）新增断言面：
  * 四层可信度：无 journal 的合成 run_dir 恒 upper_bound_only（无物理
    违规认证、退出码不为 3）；合成 journal+checksum → per_rank_total_
    hbm_certified（违规 exit 3）；checksum 缺失 → resident_kv_exact；
    守恒 checks 有 false → lifecycle_replay_exact；sha 不符 → fail-closed。
  * 容量三口径锚定（llama2_7b/swiglu/TP6/160GiB）：resident 硬上限
    1,723,864 token = 903,801,208,832 B；权重分片合计 13,476,831,232 B；
    水位目标 723,864 token = 379,513,208,832 B。
  * 事件流 stats 多桶长不变；RLE intervals 无损恢复任意桶长；绘图 series
    行数 ≤ 全局行预算；B_eff 公式；实例聚合未超但单 rank 超限必被抓到。
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
                  bucket_null: bool = False, npu_bytes: int = 1000,
                  row_budget: int = 5_000_000,
                  model_rows: bool = False) -> Path:
    """落一个最小可重放 run_dir；返回 run_dir。

    records 为 online_decision_log.jsonl 的行（dict）；token_requests 为
    manifest.json 的 requests 数组成员。model_rows=True 时 trace_config
    追加 ffn_size/num_heads/vocab_size/mlp_variant 行（三口径剖面用）。
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
    config_rows = (
        "config,layers,1,,,,synthetic\n"
        "config,hidden_size,50,,,,synthetic\n"
        "config,num_heads,2,,,,synthetic\n"
        "config,bytes_per_elem,1,,,,synthetic\n")
    if model_rows:
        config_rows += (
            "config,ffn_size,8,,,,synthetic\n"
            "config,vocab_size,100,,,,synthetic\n"
            "config,mlp_variant,swiglu,,,,synthetic\n")
    config_rows += "config,local_hbm_capacity_profile,test-64b,,,,synthetic\n"
    (run_dir / "trace_config.csv").write_text(
        "kind,key,value,group_name,pg_name,ranks,description\n" + config_rows,
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
        "params": {
            "watermark_sample_period_ns": {
                "value": entry_value, "unit": "ns",
                "derivation_program": "synthetic test",
                "evidence": None, "rationale": None},
            "watermark_series_row_budget": {
                "value": row_budget, "unit": "rows",
                "derivation_program": "synthetic test",
                "evidence": None, "rationale": None}}}),
        encoding="utf-8")
    return run_dir


def write_journal(run_dir: Path, rows: list[dict],
                  checksum: dict | None = None) -> None:
    """落合成 kv_delta_journal.jsonl（+可选 checksum json）。"""
    results = run_dir / "results"
    results.mkdir(parents=True, exist_ok=True)
    import hashlib
    payload = "".join(json.dumps(row, sort_keys=True) + "\n"
                      for row in rows)
    (results / "kv_delta_journal.jsonl").write_text(payload,
                                                    encoding="utf-8")
    if checksum is None:
        (results / "kv_delta_journal_checksum.json").unlink(missing_ok=True)
        return
    if checksum.get("sha256") == "auto":
        checksum = dict(checksum)
        checksum["sha256"] = hashlib.sha256(
            payload.encode("utf-8")).hexdigest()
    (results / "kv_delta_journal_checksum.json").write_text(
        json.dumps(checksum), encoding="utf-8")


def journal_row(sequence: int, tick: int, rank: int, instance: int,
                capacity: int, *, weight: int = 0, resident: int = 0,
                reserved: int = 0, d_weight: int = 0, d_resident: int = 0,
                d_reserved: int = 0, cause: str = "synthetic",
                transaction_id: int = 0) -> dict:
    """合成 journal 行（before/after 由 delta 推导）。"""
    return {
        "schema_version": 1, "sequence": sequence,
        "transaction_id": transaction_id, "planner_time_ns": tick,
        "rank": rank, "instance_index": instance,
        "capacity_bytes": capacity, "cause": cause,
        "request_id": None, "session_id": None,
        "allocation_key": f"synthetic:{sequence}",
        "weight_delta_bytes": d_weight,
        "resident_kv_delta_bytes": d_resident,
        "reserved_kv_delta_bytes": d_reserved,
        "before_bytes": {"weight": weight, "resident": resident,
                         "reserved": reserved},
        "after_bytes": {"weight": weight + d_weight,
                        "resident": resident + d_resident,
                        "reserved": reserved + d_reserved},
    }


def default_checksum(rows: list[dict], ranks_final: dict, *,
                     checks_true: bool = True) -> dict:
    return {
        "schema_version": 1,
        "sha256": "auto",
        "line_count": len(rows),
        "checks": {
            "manager_state_match": True,
            "physical_equals_weight": checks_true,
            "residual_reserved_zero": checks_true,
            "residual_resident_zero": checks_true,
        },
        "ranks": ranks_final,
    }


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
         "--intervals-csv", str(run_dir / "intervals.csv"),
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
        # P1-①：无 journal 的合成 run_dir 恒 upper_bound_only。
        self.assertEqual(summary["trust_tier"], "upper_bound_only")
        self.assertFalse(summary["occupancy_valid"])
        self.assertEqual(summary["violation_check"],
                         "diagnostic_upper_bound_only")
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
        self.assertFalse(summary["plot_series"]["resolution_adjusted"])
        self.assertEqual(summary["plot_series"]["effective_bucket_ns"], 10)
        # RLE 权威区间：变点 tick = {0,10,30}（t20 completion 无 KV 动作）。
        with (run_dir / "intervals.csv").open(newline="") as handle:
            intervals = list(csv.DictReader(handle))
        self.assertEqual([int(r["interval_start_ns"]) for r in intervals],
                         [0, 10, 30])
        self.assertEqual(int(intervals[0]["occupancy_start_bytes"]), 0)
        self.assertEqual(int(intervals[0]["occupancy_end_bytes"]), 100)
        self.assertEqual(int(intervals[1]["occupancy_end_bytes"]), 200)
        self.assertEqual(int(intervals[2]["occupancy_end_bytes"]), 260)
        self.assertEqual(int(intervals[2]["occupancy_peak_in_interval_bytes"]),
                         260)

    def test_violation_upper_bound_only_diagnostic_no_exit3(self):
        """P1-①：无 journal 的超限 run → 诊断报告 + 退出码 0（不再 exit 3）。

        旧语义（occupancy_valid=true + exit 3 的物理违规认证）自相矛盾
        （上界口径无资格认证），P1 起废除：violation 计数保留、tier 标注。
        """
        records, tokens = base_handcalc_records()
        run_dir = make_run_dir("viol")
        # 每 NPU 容量 10B → 实例 60B：t0 占用 100B > 60B → 违规。
        write_fixture(run_dir, repo_variant="astra-sim-face",
                      records=records, token_requests=tokens, npu_bytes=10)
        proc = run_tool(run_dir)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        summary = json.loads((run_dir / "summary.json").read_text())
        self.assertGreater(summary["violation_events"], 0)
        self.assertEqual(summary["trust_tier"], "upper_bound_only")
        self.assertFalse(summary["occupancy_valid"])
        self.assertEqual(summary["violation_check"],
                         "diagnostic_upper_bound_only")
        self.assertIn("超限诊断", proc.stderr)
        self.assertNotIn("!! [hbm-watermark] 容量违规", proc.stderr)
        with (run_dir / "instances.csv").open(newline="") as handle:
            row = next(csv.DictReader(handle))
        self.assertEqual(row["trust_tier"], "upper_bound_only")
        self.assertEqual(row["occupancy_valid"], "false")
        self.assertGreater(int(row["violation_events"]), 0)

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
        self.assertEqual(summary["trust_tier"], "upper_bound_only")
        self.assertEqual(summary["violation_check"],
                         "diagnostic_upper_bound_only")
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
        # P1-①：coverage=full 只说明 decision-log 口径；无 journal 的 run
        # tier 仍为 upper_bound_only，occupancy_valid=false。
        self.assertEqual(summary["trust_tier"], "upper_bound_only")
        self.assertFalse(summary["occupancy_valid"])
        self.assertEqual(summary["violation_check"],
                         "diagnostic_upper_bound_only")
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
            ",".join(hw.INTERVAL_COLUMNS),
            "repo_variant,instance_index,interval_start_ns,interval_end_ns,"
            "occupancy_start_bytes,occupancy_end_bytes,"
            "occupancy_peak_in_interval_bytes,evict_events,evict_bytes")
        self.assertEqual(
            ",".join(hw.INSTANCE_COLUMNS),
            "repo_variant,instance_index,trust_tier,eviction_coverage,"
            "occupancy_valid,capacity_bytes,capacity_source,bucket_ns,"
            "bucket_ns_provisional,effective_bucket_ns,resolution_adjusted,"
            "first_event_ns,last_event_ns,duration_ns,peak_occupancy_bytes,"
            "mean_occupancy_bytes,residual_occupancy_bytes,evict_events,"
            "evict_bytes,violation_events,upper_bound_peak_occupancy_bytes")

    def test_repo_variants_registered(self):
        # 2026-09-25 注册批：astra-sim-face-LRU 入表（映射复制 face 条目，
        # 见 hbm_watermark.REPO_VARIANTS 与 plan_materializer.REPO_VARIANT）。
        self.assertEqual(
            sorted(hw.REPO_VARIANTS),
            ["astra-sim-face", "astra-sim-face-LRU", "astra-sim-sh_1.0",
             "astra-sim-sh_2.0", "astra-sim-sh_3.0", "astra-sim-wscllm"])

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


class TrustTierJournalTests(unittest.TestCase):
    """P1-① 四层可信度：合成 journal（阶段2 schema）逐层判定。"""

    def _fixture_with_journal(self, tag: str, rows: list[dict],
                              checksum: dict | None,
                              npu_bytes: int = 1000):
        records, tokens = base_handcalc_records()
        run_dir = make_run_dir(tag)
        write_fixture(run_dir, repo_variant="astra-sim-wscllm",
                      records=records, token_requests=tokens,
                      npu_bytes=npu_bytes,
                      model_rows=True)
        write_journal(run_dir, rows, checksum)
        return run_dir

    def test_certified_tier_per_rank_ok(self):
        """journal+checksum 四门全过 → certified；终态与 checksum 一致。"""
        rows = [
            journal_row(0, 0, 0, 0, 1000, d_weight=300),
            journal_row(1, 0, 1, 0, 1000, d_weight=200),
            journal_row(2, 10, 0, 0, 1000, weight=300, d_resident=400),
            journal_row(3, 20, 1, 0, 1000, weight=200, d_resident=300),
            journal_row(4, 30, 0, 0, 1000, weight=300, resident=400,
                        d_resident=-400, cause="terminal_session_retire"),
            journal_row(5, 30, 1, 0, 1000, weight=200, resident=300,
                        d_resident=-300, cause="terminal_session_retire"),
        ]
        ranks_final = {
            "0": {"capacity_bytes": 1000, "weight": 300, "resident": 0,
                  "reserved": 0, "physical": 300},
            "1": {"capacity_bytes": 1000, "weight": 200, "resident": 0,
                  "reserved": 0, "physical": 200},
        }
        run_dir = self._fixture_with_journal(
            "cert", rows, default_checksum(rows, ranks_final))
        proc = run_tool(run_dir)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        summary = json.loads((run_dir / "summary.json").read_text())
        self.assertEqual(summary["trust_tier"],
                         "per_rank_total_hbm_certified")
        self.assertTrue(summary["occupancy_valid"])
        self.assertEqual(summary["violation_check"],
                         "per_rank_physical_gt_capacity")
        self.assertEqual(summary["violation_events"], 0)
        self.assertIn("trust_tier=per_rank_total_hbm_certified", proc.stderr)
        journal = summary["journal_replay"]
        self.assertEqual(journal["rows"], 6)
        self.assertEqual(journal["per_rank_final"]["0"]["resident"], 0)
        self.assertEqual(journal["per_rank_final"]["0"]["weight"], 300)
        # 权威 stats 来自 journal resident 变点：rank0/1 同实例聚合。
        self.assertEqual(summary["instances"]["0"]["peak_occupancy_bytes"],
                         700)
        self.assertEqual(summary["instances"]["0"]
                         ["residual_occupancy_bytes"], 0)
        # 对照列：decision-log 上界重放同 run 的峰值（260）与 gap 可视化。
        comparison = summary["decision_log_upper_bound_comparison"]["0"]
        self.assertEqual(comparison["upper_bound_peak_occupancy_bytes"], 260)
        self.assertEqual(comparison["authoritative_peak_occupancy_bytes"],
                         700)
        self.assertEqual(comparison["peak_gap_bytes"], -440)
        with (run_dir / "instances.csv").open(newline="") as handle:
            row = next(csv.DictReader(handle))
        self.assertEqual(row["trust_tier"], "per_rank_total_hbm_certified")
        self.assertEqual(row["eviction_coverage"], "journal_exact")
        self.assertEqual(row["occupancy_valid"], "true")
        self.assertEqual(row["upper_bound_peak_occupancy_bytes"], "260")

    def test_certified_tier_rank_over_aggregate_ok_exit3(self):
        """锚定用例：实例聚合未超但某 rank 超限必须被抓到（exit 3）。

        rank0 physical=1200 > capacity 1000（单 rank 违规），而实例聚合
        1200+800=2000 ≤ 2×1000（聚合口径通过）——旧聚合检查放过、逐 rank
        认证必须抓到（TP shard 不均场景的判决面）。
        """
        rows = [
            journal_row(0, 0, 0, 0, 1000, d_weight=300),
            journal_row(1, 0, 1, 0, 1000, d_weight=200),
            journal_row(2, 10, 0, 0, 1000, weight=300, d_resident=500),
            journal_row(3, 10, 1, 0, 1000, weight=200, d_resident=300),
            journal_row(4, 20, 0, 0, 1000, weight=300, resident=500,
                        d_resident=400),
            journal_row(5, 20, 1, 0, 1000, weight=200, resident=300,
                        d_resident=300),
        ]
        ranks_final = {
            "0": {"capacity_bytes": 1000, "weight": 300, "resident": 900,
                  "reserved": 0, "physical": 1200},
            "1": {"capacity_bytes": 1000, "weight": 200, "resident": 600,
                  "reserved": 0, "physical": 800},
        }
        run_dir = self._fixture_with_journal(
            "rankviol", rows, default_checksum(rows, ranks_final))
        proc = run_tool(run_dir)
        self.assertEqual(proc.returncode, hw.EXIT_VIOLATION)
        self.assertIn("正式判决", proc.stderr)
        summary = json.loads((run_dir / "summary.json").read_text())
        self.assertEqual(summary["trust_tier"],
                         "per_rank_total_hbm_certified")
        self.assertEqual(summary["violation_events"], 1)
        self.assertEqual(summary["journal_replay"]
                         ["per_rank_physical_violations"], {"0": 1})
        self.assertEqual(summary["max_exceed_bytes"], 200)

    def test_resident_kv_exact_when_checksum_missing(self):
        """journal 在场、checksum 缺失 → resident_kv_exact（仅报告）。"""
        rows = [
            journal_row(0, 0, 0, 0, 1000, d_weight=300),
            journal_row(1, 10, 0, 0, 1000, weight=300, d_resident=1500),
            journal_row(2, 20, 0, 0, 1000, weight=300, resident=1500,
                        d_resident=-200),
        ]
        run_dir = self._fixture_with_journal("resid", rows, None)
        proc = run_tool(run_dir)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        summary = json.loads((run_dir / "summary.json").read_text())
        self.assertEqual(summary["trust_tier"], "resident_kv_exact")
        # 同样的逐 rank 超限（两行 physical 均超 1000）在无证书层只报
        # 告，不出 exit 3；occupancy 序列本身精确（journal 行级自证）。
        self.assertEqual(summary["violation_events"], 2)
        self.assertIn("超限诊断", proc.stderr)
        self.assertTrue(summary["occupancy_valid"])

    def test_lifecycle_exact_when_conservation_checks_fail(self):
        """checksum 在场但守恒 checks 有 false → lifecycle_replay_exact。"""
        rows = [
            journal_row(0, 0, 0, 0, 1000, d_weight=300),
            journal_row(1, 10, 0, 0, 1000, weight=300, d_resident=100),
            journal_row(2, 20, 0, 0, 1000, weight=300, resident=100,
                        d_resident=-100),
        ]
        ranks_final = {
            "0": {"capacity_bytes": 1000, "weight": 300, "resident": 0,
                  "reserved": 0, "physical": 300},
        }
        # 守恒检查谎报失败（resident 实为 0 但 checks 记 false）→ 层降级。
        checksum = default_checksum(rows, ranks_final, checks_true=False)
        run_dir = self._fixture_with_journal("lc", rows, checksum)
        proc = run_tool(run_dir)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        summary = json.loads((run_dir / "summary.json").read_text())
        self.assertEqual(summary["trust_tier"], "lifecycle_replay_exact")
        self.assertEqual(summary["violation_events"], 0)

    def test_journal_sha_mismatch_fails_closed(self):
        rows = [journal_row(0, 0, 0, 0, 1000, d_weight=300)]
        checksum = default_checksum(
            rows, {"0": {"capacity_bytes": 1000, "weight": 300,
                         "resident": 0, "reserved": 0, "physical": 300}})
        checksum["sha256"] = "0" * 64  # 篡改
        run_dir = self._fixture_with_journal("shabad", rows, checksum)
        proc = run_tool(run_dir)
        self.assertEqual(proc.returncode, 2)
        self.assertIn("sha256", proc.stderr)

    def test_journal_chain_break_fails_closed(self):
        rows = [
            journal_row(0, 0, 0, 0, 1000, d_weight=300),
            # 链断裂：before.resident 谎报 50（前行 after=0）。
            journal_row(1, 10, 0, 0, 1000, weight=300, resident=50,
                        d_resident=50),
        ]
        run_dir = self._fixture_with_journal("chain", rows, None)
        proc = run_tool(run_dir)
        self.assertEqual(proc.returncode, 2)
        self.assertIn("链断裂", proc.stderr)

    def test_hard_limit_diagnostic_per_rank(self):
        """resident 硬上限口径：TP shard 不均下逐 rank resident 超限计数。

        本合成模型（layers=1/hidden=50/heads=5/ffn=8/vocab=100/swiglu/
        bpe=1、TP6）权重分片 = [4025,4025,3875,3875,3775,1775]、KV 分片
        = [20,20,20,20,20,0] B/token（前五 rank 各持 1 头）。容量 10000
        B/rank：hard tokens = min(⌊(10000−4025)/20⌋, ⌊(10000−3875)/20⌋,
        ⌊(10000−3775)/20⌋) = min(298,306,311) = 298 → KV rank 限 20×298
        = 5960 B。rank2 resident 到 5961 → 硬上限超限 1 例，而 physical =
        3875+5961 = 9836 ≤ 10000（正式认证不违规）——"硬上限比正式认证
        更紧"的分层语义（限值取全实例最紧 rank 的 token 数，TP shard 不
        均下低权重 rank 被压得更紧）。
        """
        rows = [
            journal_row(0, 0, 0, 0, 10000, d_weight=4025),
            journal_row(1, 0, 2, 0, 10000, d_weight=3875),
            journal_row(2, 10, 2, 0, 10000, weight=3875, d_resident=3000),
            journal_row(3, 20, 2, 0, 10000, weight=3875, resident=3000,
                        d_resident=2961),
            journal_row(4, 30, 0, 0, 10000, weight=4025, d_resident=1000),
        ]
        ranks_final = {
            "0": {"capacity_bytes": 10000, "weight": 4025, "resident": 1000,
                  "reserved": 0, "physical": 5025},
            "2": {"capacity_bytes": 10000, "weight": 3875, "resident": 5961,
                  "reserved": 0, "physical": 9836},
        }
        run_dir = self._fixture_with_journal(
            "hard", rows, default_checksum(rows, ranks_final),
            npu_bytes=10000)
        # num_heads=5（基础 fixture 为 2）：前五 KV rank 各持 1 头、权重不均。
        (run_dir / "trace_config.csv").write_text(
            "kind,key,value,group_name,pg_name,ranks,description\n"
            "config,layers,1,,,,synthetic\n"
            "config,hidden_size,50,,,,synthetic\n"
            "config,ffn_size,8,,,,synthetic\n"
            "config,num_heads,5,,,,synthetic\n"
            "config,vocab_size,100,,,,synthetic\n"
            "config,bytes_per_elem,1,,,,synthetic\n"
            "config,mlp_variant,swiglu,,,,synthetic\n"
            "config,local_hbm_capacity_profile,test-64b,,,,synthetic\n",
            encoding="utf-8")
        proc = run_tool(run_dir)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        summary = json.loads((run_dir / "summary.json").read_text())
        calibers = summary["capacity_calibers"]
        self.assertEqual(calibers["weight_shard_bytes_by_relative_rank"],
                         [4025, 4025, 3875, 3875, 3775, 1775])
        self.assertEqual(calibers["kv_shard_bytes_per_token_by_relative_rank"],
                         [20, 20, 20, 20, 20, 0])
        hard = calibers["resident_kv_hard_limit"]
        self.assertEqual(hard["limit_tokens"], 298)
        self.assertEqual(hard["limit_bytes_per_instance"], 298 * 100)
        journal = summary["journal_replay"]
        # 正式认证（physical ≤ capacity）无违规；硬上限口径抓到 rank2。
        self.assertEqual(summary["violation_events"], 0)
        self.assertEqual(journal["resident_hard_limit_exceed_events"], 1)
        self.assertEqual(journal["resident_hard_limit_exceed_ranks"],
                         {"2": 1})


class JournalParityRealTests(unittest.TestCase):
    """对拍 /var/tmp/phase2_journal/runs/on_2s 的真 journal+checksum（V1）。"""

    FIXTURE = Path("/var/tmp/phase2_journal/runs/on_2s")

    @unittest.skipUnless(FIXTURE.is_dir(), "phase2 journal fixture 缺失")
    def test_real_journal_certified_and_matches_checksum(self):
        out = Path(tempfile.mkdtemp(prefix="slo_hbm_parity_"))
        proc = subprocess.run(
            [sys.executable, str(SLO_TOOLS_DIR / "hbm_watermark.py"),
             str(self.FIXTURE), "--json", str(out / "summary.json"),
             "--intervals-csv", str(out / "intervals.csv"),
             "-o", str(out / "plot.csv"),
             "--instances-csv", str(out / "instances.csv")],
            capture_output=True, text=True, check=False)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        summary = json.loads((out / "summary.json").read_text())
        self.assertEqual(summary["trust_tier"],
                         "per_rank_total_hbm_certified")
        self.assertEqual(summary["violation_events"], 0)
        journal = summary["journal_replay"]
        checksum = json.loads(
            (self.FIXTURE / "results" / "kv_delta_journal_checksum.json")
            .read_text())
        self.assertEqual(journal["rows"], checksum["line_count"])
        self.assertEqual(journal["sha256"], checksum["sha256"])
        # 逐 rank 终态与 checksum json 全等（weight/resident/reserved/
        # capacity 四字段）。
        for rank_key, expected in checksum["ranks"].items():
            final = journal["per_rank_final"][rank_key]
            self.assertEqual(final["weight"], expected["weight"])
            self.assertEqual(final["resident"], expected["resident"])
            self.assertEqual(final["reserved"], expected["reserved"])
            self.assertEqual(final["capacity_bytes"],
                             expected["capacity_bytes"])
        # 守恒：run 末 resident/reserved 全 0、physical=weight。
        self.assertTrue(all(v["resident"] == 0 and v["reserved"] == 0
                            and v["physical"] == v["weight"]
                            for v in journal["per_rank_final"].values()))
        # 对照列在场：decision-log 上界重放峰值（该 run 无缺口=两值相等）。
        self.assertIn("decision_log_upper_bound_comparison", summary)


class CapacityCaliberTests(unittest.TestCase):
    """P1-③ 三口径锚定数值（full_tracelab 配置：swiglu/TP6/160GiB）。"""

    def test_llama2_7b_tp6_160gib_anchors(self):
        run_dir = make_run_dir("calib")
        records, tokens = base_handcalc_records()
        # 复刻 full_tracelab 的配置链：llama2_7b（swiglu）+ validation-160gib。
        write_fixture(run_dir, repo_variant="astra-sim-wscllm",
                      records=records, token_requests=tokens,
                      npu_bytes=171798691840)
        config = run_dir / "trace_config.csv"
        config.write_text(
            "kind,key,value,group_name,pg_name,ranks,description\n"
            "config,layers,32,,,,synthetic\n"
            "config,hidden_size,4096,,,,synthetic\n"
            "config,ffn_size,11008,,,,synthetic\n"
            "config,num_heads,32,,,,synthetic\n"
            "config,vocab_size,32000,,,,synthetic\n"
            "config,bytes_per_elem,2,,,,synthetic\n"
            "config,mlp_variant,swiglu,,,,synthetic\n"
            "config,kv_reserve_context_tokens,1000000,,,,synthetic\n"
            "config,local_hbm_capacity_profile,test-64b,,,,synthetic\n",
            encoding="utf-8")
        proc = run_tool(run_dir)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        summary = json.loads((run_dir / "summary.json").read_text())
        calibers = summary["capacity_calibers"]
        self.assertNotEqual(calibers, "NA")
        # 锚定值一：权重分片合计 = 13,476,831,232 B（manager 原函数）。
        self.assertEqual(calibers["weight_total_bytes_per_instance"],
                         13_476_831_232)
        self.assertEqual(
            calibers["weight_shard_bytes_by_relative_rank"],
            [2_335_890_091, 2_335_890_091, 2_201_655_979, 2_201_655_979,
             2_200_869_546, 2_200_869_546])
        self.assertEqual(
            calibers["kv_shard_bytes_per_token_by_relative_rank"],
            [98_304, 98_304, 81_920, 81_920, 81_920, 81_920])
        # 锚定值二：resident 硬上限 = 1,723,864 token → 903,801,208,832 B。
        hard = calibers["resident_kv_hard_limit"]
        self.assertEqual(hard["limit_tokens"], 1_723_864)
        self.assertEqual(hard["limit_bytes_per_instance"],
                         903_801_208_832)
        self.assertEqual(hard["limit_resident_bytes_by_relative_rank"],
                         [98_304 * 1_723_864, 98_304 * 1_723_864]
                         + [81_920 * 1_723_864] * 4)
        # 锚定值三：水位目标 = 723,864 token → 379,513,208,832 B（仅报告）。
        target = calibers["watermark_reserve_target"]
        self.assertEqual(target["reserve_context_tokens"], 1_000_000)
        self.assertEqual(target["target_tokens"], 723_864)
        self.assertEqual(target["target_bytes_per_instance"],
                         379_513_208_832)
        self.assertIn("不作判决", target["semantics"])
        # 实例聚合容量链同旧（160GiB × 6）。
        self.assertEqual(summary["capacity_bytes_per_instance"],
                         171798691840 * 6)


class EventStreamStatsTests(unittest.TestCase):
    """P1-② stats 多桶长不变 + P1-④ RLE 无损恢复/行预算。"""

    STAT_FIELDS = ("first_event_ns", "last_event_ns", "duration_ns",
                   "peak_occupancy_bytes", "mean_occupancy_bytes",
                   "residual_occupancy_bytes", "evict_events",
                   "evict_bytes", "violation_events")

    def _stats_for_bucket(self, bucket_ns: int) -> dict:
        records, tokens = base_handcalc_records()
        run_dir = make_run_dir(f"inv{bucket_ns}")
        write_fixture(run_dir, repo_variant="astra-sim-face",
                      records=records, token_requests=tokens, npu_bytes=10,
                      bucket_ns=bucket_ns)
        proc = run_tool(run_dir)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        summary = json.loads((run_dir / "summary.json").read_text())
        instances = {}
        with (run_dir / "instances.csv").open(newline="") as handle:
            for row in csv.DictReader(handle):
                instances[int(row["instance_index"])] = row
        return summary, instances

    def test_stats_invariant_across_bucket_lengths(self):
        reference = None
        for bucket_ns in (10, 7, 3, 25):
            _, instances = self._stats_for_bucket(bucket_ns)
            if reference is None:
                reference = instances
                continue
            self.assertEqual(set(instances), set(reference))
            for instance, row in instances.items():
                for field in self.STAT_FIELDS:
                    if field == "violation_events":
                        self.assertEqual(row[field],
                                         reference[instance][field],
                                         f"bucket={bucket_ns} {field}")
                    else:
                        self.assertEqual(
                            float(row[field]),
                            float(reference[instance][field]),
                            f"bucket={bucket_ns} {field}")

    def test_intervals_lossless_reconstruction(self):
        """intervals.csv → 任意桶长序列恢复 == 直接产出的 plot_series。"""
        records, tokens = base_handcalc_records()
        run_dir = make_run_dir("rle")
        write_fixture(run_dir, repo_variant="astra-sim-face",
                      records=records, token_requests=tokens)
        proc = run_tool(run_dir)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        summary = json.loads((run_dir / "summary.json").read_text())
        origin = summary["plot_series"]["bucket_origin_ns"]
        span_end = summary["span_end_ns"]
        # 读回 intervals.csv，重建 B=10 桶行，与 plot_series 全等。
        by_unit = {}
        with (run_dir / "intervals.csv").open(newline="") as handle:
            for row in csv.DictReader(handle):
                by_unit.setdefault(int(row["instance_index"]), []).append(
                    (int(row["interval_start_ns"]),
                     int(row["occupancy_start_bytes"]),
                     int(row["occupancy_end_bytes"]),
                     int(row["occupancy_peak_in_interval_bytes"]),
                     int(row["evict_events"]),
                     int(row["evict_bytes"])))
        expected_rows = []
        with (run_dir / "series.csv").open(newline="") as handle:
            for row in csv.DictReader(handle):
                expected_rows.append(
                    (int(row["instance_index"]), int(row["bucket_index"]),
                     int(row["bucket_start_ns"]), int(row["bucket_end_ns"]),
                     int(row["occupancy_end_bytes"]),
                     int(row["occupancy_peak_in_bucket_bytes"]),
                     int(row["evict_events"]), int(row["evict_bytes"])))
        rebuilt = []
        for instance, change_points in sorted(by_unit.items()):
            for row in hw.bucket_row_sweep(
                    lambda items=change_points: iter(items), origin,
                    span_end, 10):
                rebuilt.append((instance,) + row)
        self.assertEqual(rebuilt, sorted(expected_rows))

    def test_row_budget_bounds_dense_rows(self):
        """小行预算：B_eff 放粗（或拒绝稠密），行数 ≤ 预算。"""
        records, tokens = base_handcalc_records()
        run_dir = make_run_dir("budget")
        write_fixture(run_dir, repo_variant="astra-sim-face",
                      records=records, token_requests=tokens, bucket_ns=3,
                      row_budget=2)  # N=1 实例、R=2 > N → B_eff 放粗
        proc = run_tool(run_dir)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        summary = json.loads((run_dir / "summary.json").read_text())
        plot = summary["plot_series"]
        self.assertTrue(plot["resolution_adjusted"])
        self.assertEqual(plot["adjustment_reason"], "row_budget_coarsened")
        self.assertGreater(plot["effective_bucket_ns"], 3)
        self.assertLessEqual(plot["dense_rows_written"], plot["row_budget"])
        # span=30、N=1、R=2 → B_eff = ceil(30·1/(2−1)) = 30 → 2 行（b0+b1）。
        self.assertEqual(plot["effective_bucket_ns"], 30)
        self.assertEqual(plot["dense_rows_written"], 2)

    def test_row_budget_refuses_dense_when_r_le_n(self):
        records, tokens = base_handcalc_records()
        run_dir = make_run_dir("refuse")
        write_fixture(run_dir, repo_variant="astra-sim-face",
                      records=records, token_requests=tokens, bucket_ns=10,
                      row_budget=1)  # R=1 ≤ N=1 → 拒绝稠密只给 RLE
        proc = run_tool(run_dir)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("拒绝稠密输出", proc.stderr)
        summary = json.loads((run_dir / "summary.json").read_text())
        plot = summary["plot_series"]
        self.assertTrue(plot["dense_output_refused"])
        self.assertEqual(plot["adjustment_reason"],
                         "row_budget_exceeded_dense_refused")
        self.assertEqual(plot["dense_rows_written"], 0)
        self.assertGreater(summary["interval_rows"], 0)
        with (run_dir / "series.csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(rows, [])

    def test_b_eff_formula_unit(self):
        # ceil(S·N/(R−N)) 放粗；请求桶已够大 → 不调整；R≤N → 拒绝。
        self.assertEqual(hw.resolve_effective_bucket_ns(5, 1000, 4, 100),
                         (42, True, "row_budget_coarsened"))
        self.assertEqual(hw.resolve_effective_bucket_ns(5, 1000, 1, 100),
                         (11, True, "row_budget_coarsened"))
        self.assertEqual(hw.resolve_effective_bucket_ns(50, 1000, 4, 100),
                         (50, False, "none"))
        self.assertEqual(hw.resolve_effective_bucket_ns(5, 1000, 4, 4),
                         (5, True, "row_budget_exceeded_dense_refused"))


class EvictBytesCoverageTests(unittest.TestCase):
    """P1-⑤：evict_bytes 在 full_reconciled 也输出真实值（NA 只留 count_only）。"""

    def test_wscllm_full_reconciled_evict_bytes_real(self):
        reset_seq()
        records = [
            prefill_record("sA_r0", 0, instance=0),
            decode_record("sA_r0", 10, instance=0),
            completion_record("sA_r0", 20, decision={
                "terminal_kv_release_at_completion": False,
                "completion_evictions": [{
                    "time_ns": 20, "phase": "completion",
                    "reason": "synthetic", "trigger_request_id": "sA_r0",
                    "victim_session_id": "sA", "victim_instance_index": 0,
                    "victim_last_completion_ns": 0, "context_tokens": 1,
                    "shard_bytes": [100]}]}),
        ]
        tokens = [token_row("sA_r0", "sA", 0, 1, 1)]
        run_dir = make_run_dir("evb")
        write_fixture(run_dir, repo_variant="astra-sim-wscllm",
                      records=records, token_requests=tokens)
        proc = run_tool(run_dir)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        summary = json.loads((run_dir / "summary.json").read_text())
        self.assertEqual(summary["eviction_coverage"], "full_reconciled")
        with (run_dir / "instances.csv").open(newline="") as handle:
            row = next(csv.DictReader(handle))
        self.assertEqual(row["evict_bytes"], "100")
        self.assertEqual(summary["instances"]["0"]["evict_bytes"], 100)
        self.assertEqual(summary["instances"]["0"]["residual_occupancy_bytes"],
                         0)

    def test_s2_count_only_evict_bytes_still_na(self):
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
        run_dir = make_run_dir("evna")
        write_fixture(run_dir, repo_variant="astra-sim-sh_2.0",
                      records=records, token_requests=tokens)
        proc = run_tool(run_dir)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        summary = json.loads((run_dir / "summary.json").read_text())
        self.assertEqual(summary["eviction_coverage"], "count_only")
        with (run_dir / "instances.csv").open(newline="") as handle:
            row = next(csv.DictReader(handle))
        self.assertEqual(row["evict_bytes"], "NA")


class FaceTieredUpgradeTests(unittest.TestCase):
    """B4（-LRU face，2026-09-07）：history_transfers 在场 → face 映射升级
    full 口径（契约逐出行/逐段 restore/PD 列表行）。

    手算（COEF=100）：A f(1)=100@t0 → f(2)=200@t10；B 准入@t30 逐出 A 40
    （余 160，legacy admission_evictions 同值镜像【不得双计】）+ remote_load
    恢复 100 + 增长到 f(2)=200（+100）→ inst0=360；B decode@t40 PD 契约行
    noc_migrate 200 → inst0=160 / inst1=200，decode 增至 f(3)=300；B turn1
    @t50 noc 300（inst1=0 / inst2=300）——PD 列表行不消费时 B 字节滞留
    inst0 而归属 inst1，本步对 inst1 扣 300 即负占用（B3 冒烟 paper64
    tick=2473790463 复现路径的回归钉）。
    """

    def _records_tokens(self):
        reset_seq()
        records = [
            prefill_record("a_r0", 0, decision={
                "history_action": "NO_HISTORY",
                "history_location_before": None,
                "history_transfer_bytes": 0,
                "history_evictions": [], "prefill_evictions": [],
                "history_transfers": [],
                "admission_evictions": [],
                "decode_target_evictions": []}),
            decode_record("a_r0", 10, decision={
                "prefill_decode_transfer": [{
                    "kind": "local_hit",
                    "reason": "prefill_decode_local_reuse",
                    "session_id": "session_A", "total_bytes": 200,
                    "source_instance_index": 0,
                    "target_instance_index": 0,
                    "layer_start": 0, "layer_end": 1}],
                "decode_evictions": [], "decode_target_evictions": []}),
            completion_record("a_r0", 20),
            prefill_record("b_r0", 30, decision={
                "history_action": "REMOTE_RESTORE",
                "history_location_before": "remote_memory",
                "history_location_before_instance_index": None,
                "history_resident_prefix_layers": 0,
                "history_transfer_bytes": 100,
                "history_evictions": [{
                    "kind": "remote_store",
                    "reason": "history_and_prefill_admission_suffix_half",
                    "session_id": "session_A", "total_bytes": 40,
                    "source_instance_index": 0,
                    "target_instance_index": None,
                    "layer_start": 0, "layer_end": 1}],
                "prefill_evictions": [],
                "history_transfers": [{
                    "kind": "remote_load", "reason": "history_remote_restore",
                    "session_id": "session_B", "total_bytes": 100,
                    "source_instance_index": None,
                    "target_instance_index": 0,
                    "layer_start": 0, "layer_end": 1}],
                # legacy 同值镜像（B3 双序列化）——两级并存消费会双计。
                "admission_evictions": [{
                    "time_ns": 30, "phase": "prefill", "reason": "synthetic",
                    "trigger_request_id": "b_r0",
                    "victim_session_id": "session_A",
                    "victim_instance_index": 0,
                    "victim_last_completion_ns": 20,
                    "context_tokens": 2, "shard_bytes": [40]}],
                "decode_target_evictions": []}),
            decode_record("b_r0", 40, instance=1, decision={
                "prefill_decode_transfer": [{
                    "kind": "noc_migrate",
                    "reason": "prefill_decode_instance_migrate",
                    "session_id": "session_B", "total_bytes": 200,
                    "source_instance_index": 0,
                    "target_instance_index": 1,
                    "layer_start": 0, "layer_end": 1}],
                "decode_evictions": [], "decode_target_evictions": []}),
            completion_record("b_r0", 45),
            prefill_record("b_r1", 50, instance=2, decision={
                "history_action": "NOC_MIGRATE",
                "history_location_before": "local_hbm",
                "history_location_before_instance_index": 1,
                "history_resident_prefix_layers": 1,
                "history_transfer_bytes": 300,
                "history_evictions": [], "prefill_evictions": [],
                "history_transfers": [{
                    "kind": "noc_migrate", "reason": "history_other_instance",
                    "session_id": "session_B", "total_bytes": 300,
                    "source_instance_index": 1,
                    "target_instance_index": 2,
                    "layer_start": 0, "layer_end": 1}],
                "admission_evictions": [], "decode_target_evictions": []}),
            decode_record("b_r1", 60, instance=2, decision={
                "prefill_decode_transfer": [{
                    "kind": "local_hit",
                    "reason": "prefill_decode_local_reuse",
                    "session_id": "session_B", "total_bytes": 300,
                    "source_instance_index": 2,
                    "target_instance_index": 2,
                    "layer_start": 0, "layer_end": 1}],
                "decode_evictions": [], "decode_target_evictions": []}),
            completion_record("b_r1", 70),
        ]
        tokens = [token_row("a_r0", "session_A", 0, 1, 2),
                  token_row("b_r0", "session_B", 1, 2, 3),
                  token_row("b_r1", "session_B", 3, 3, 3)]
        tokens[1]["turn_index"] = 1
        tokens[2]["turn_index"] = 2
        return records, tokens

    def test_face_tiered_full_coverage_no_double_count(self):
        records, tokens = self._records_tokens()
        run_dir = make_run_dir("faceT")
        write_fixture(run_dir, repo_variant="astra-sim-face",
                      records=records, token_requests=tokens)
        proc = run_tool(run_dir)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        summary = json.loads((run_dir / "summary.json").read_text())
        self.assertEqual(summary["eviction_coverage"], "full")
        self.assertEqual(summary["violation_events"], 0)
        # legacy admission_evictions 是契约 history_evictions 的同值镜像：
        # 只消费契约行——40 而非 80（双计即翻倍）。
        self.assertEqual(summary["actions"]["evictions"], 1)
        self.assertEqual(summary["actions"]["evict_bytes"], 40)
        self.assertEqual(summary["actions"]["partial_evictions"], 1)
        self.assertEqual(summary["actions"]["restore_remote_add"], 1)
        self.assertEqual(summary["actions"]["restore_move"], 1)
        # PD 契约行列表：noc_migrate 段真实搬移、local_hit 段仅归属切换。
        self.assertEqual(summary["actions"]["decode_move"], 1)
        self.assertEqual(summary["actions"]["decode_stay"], 2)
        self.assertEqual(sum(summary["anomalies"].values()), 0)
        self.assertEqual(summary["instances"]["0"]
                         ["residual_occupancy_bytes"], 160)
        self.assertEqual(summary["instances"]["1"]
                         ["residual_occupancy_bytes"], 0)
        self.assertEqual(summary["instances"]["2"]
                         ["residual_occupancy_bytes"], 300)

    def test_face_old_product_keeps_legacy_mapping(self):
        """旧产物（无 history_transfers 字段）→ legacy 列表口径原样。"""
        records, tokens = base_handcalc_records()
        run_dir = make_run_dir("faceO")
        write_fixture(run_dir, repo_variant="astra-sim-face",
                      records=records, token_requests=tokens)
        proc = run_tool(run_dir)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        summary = json.loads((run_dir / "summary.json").read_text())
        self.assertEqual(summary["eviction_coverage"], "full")
        self.assertEqual(summary["actions"]["evict_bytes"], 40)
        self.assertEqual(summary["instances"]["0"]
                         ["residual_occupancy_bytes"], 260)

    def test_face_lru_new_run_whole_eviction_restore_ratio_one(self):
        """face-LRU 新运行形状（session 级 Tiered-LRU，2026-09-25）：
        逐出恒为整体 store（evict bytes == session tracked bytes →
        partial_evictions == 0），恢复恒全量（ratio 恒 1.0）；reason/
        layer 域为全量域（layers0-1、before/after = 1/0）。"""

        def _records_tokens():
            reset_seq()
            records = [
                prefill_record("a_r0", 0, decision={
                    "history_action": "NO_HISTORY",
                    "history_location_before": None,
                    "history_transfer_bytes": 0,
                    "history_evictions": [], "prefill_evictions": [],
                    "history_transfers": [],
                    "admission_evictions": [],
                    "decode_target_evictions": []}),
                decode_record("a_r0", 10, decision={
                    "prefill_decode_transfer": None,
                    "decode_evictions": [], "decode_target_evictions": []}),
                completion_record("a_r0", 20, decision={
                    "kv_state_after_completion": "local_hbm"}),
                # b_r0 准入:整体逐出 a(evict bytes == session bytes 200,
                # layer 全量域 [0,1))。
                prefill_record("b_r0", 30, decision={
                    "history_action": "NO_HISTORY",
                    "history_location_before": None,
                    "history_transfer_bytes": 0,
                    "history_evictions": [{
                        "kind": "remote_store",
                        "reason":
                            "history_and_prefill_admission_full:layers0-1",
                        "session_id": "session_A", "total_bytes": 200,
                        "source_instance_index": 0,
                        "target_instance_index": None,
                        "layer_start": 0, "layer_end": 1}],
                    "prefill_evictions": [],
                    "history_transfers": [],
                    "admission_evictions": [],
                    "decode_target_evictions": []}),
                decode_record("b_r0", 40, decision={
                    "prefill_decode_transfer": None,
                    "decode_evictions": [], "decode_target_evictions": []}),
                completion_record("b_r0", 45, decision={
                    "kv_state_after_completion": "local_hbm"}),
                # a_r1 回迁:REMOTE_RESTORE 全量(nbytes == f(history) →
                # ratio 1.0)。
                prefill_record("a_r1", 50, decision={
                    "history_action": "REMOTE_RESTORE",
                    "history_location_before": "remote_memory",
                    "history_location_before_instance_index": None,
                    "history_resident_prefix_layers": 0,
                    "history_transfer_bytes": 200,
                    "history_evictions": [], "prefill_evictions": [],
                    "history_transfers": [{
                        "kind": "remote_load",
                        "reason": "history_remote_restore",
                        "session_id": "session_A", "total_bytes": 200,
                        "source_instance_index": None,
                        "target_instance_index": 0,
                        "layer_start": 0, "layer_end": 1}],
                    "admission_evictions": [],
                    "decode_target_evictions": []}),
                decode_record("a_r1", 60, decision={
                    "prefill_decode_transfer": None,
                    "decode_evictions": [], "decode_target_evictions": []}),
                completion_record("a_r1", 70, decision={
                    "kv_state_after_completion": "local_hbm"}),
            ]
            tokens = [token_row("a_r0", "session_A", 0, 1, 2),
                      token_row("b_r0", "session_B", 0, 1, 1)]
            a_r1 = token_row("a_r1", "session_A", 2, 3, 3)
            a_r1["turn_index"] = 1
            a_r1["queue_index"] = 2
            tokens.append(a_r1)
            return records, tokens

        records, tokens = _records_tokens()
        run_dir = make_run_dir("faceL")
        write_fixture(run_dir, repo_variant="astra-sim-face-LRU",
                      records=records, token_requests=tokens)
        proc = run_tool(run_dir)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        summary = json.loads((run_dir / "summary.json").read_text())
        self.assertEqual(summary["eviction_coverage"], "full")
        self.assertEqual(summary["violation_events"], 0)
        # 新运行形状:整体逐出 → partial 计数恒 0;逐出字节 = 全 session。
        self.assertEqual(summary["actions"]["evictions"], 1)
        self.assertEqual(summary["actions"]["evict_bytes"], 200)
        self.assertEqual(summary["actions"]["partial_evictions"], 0)
        # 恢复恒全量:ratio 桶只有 1.000、无 0.5 半层桶。
        self.assertEqual(
            summary["restore_bytes_ratio_histogram"], {"1.000": 1})
        self.assertEqual(summary["actions"]["restore_remote_add"], 1)
        self.assertEqual(sum(summary["anomalies"].values()), 0)
        # 台账守恒:i0 残留 = b(100) + a 回迁(200) + a_r1 prefill 增长
        # (f(3)-f(2)=100)。
        self.assertEqual(summary["instances"]["0"]
                         ["residual_occupancy_bytes"], 400)


class AdmissionProbeSkipTests(unittest.TestCase):
    """P0-1/P1 probe 观测行适配（收尾批 2026-09-01）：
    decode_admission_probe / prefill_admission_probe 行跳过且计数审计
    （stderr 打印"跳过 N 条"，不静默；stdout 保持 CSV 流纯净——`-o -`
    时 series 直接写 stdout，审计行混入会破坏下游解析）；其余未知 kind
    依旧 fail-closed。
    """

    @staticmethod
    def _probe_record(request_id: str, tick: int, kind: str) -> dict:
        return {
            "seq": -1,  # write_fixture 按行序重排 seq
            "tick": tick,
            "priority": 0,
            "kind": kind,
            "request_id": request_id,
            "decision": {
                "admission_probe": {
                    "original_instance_index": 1,
                    "feasible_candidates": [0, 2],
                    "outcome": "reselected_infeasible_first",
                    "selected_instance_index": 2,
                }
            },
        }

    def test_probe_rows_skipped_with_audit_line(self):
        records, tokens = base_handcalc_records()
        # 在合法三件套之间插入 2 条 probe 观测行(tick 单调)。
        records.insert(1, self._probe_record(
            "session_A_request_0", 5, "decode_admission_probe"))
        records.insert(4, self._probe_record(
            "session_B_request_0", 35, "prefill_admission_probe"))
        run_dir = make_run_dir("probe1")
        write_fixture(run_dir, repo_variant="astra-sim-face",
                      records=records, token_requests=tokens)
        proc = run_tool(run_dir)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("跳过 2 条 admission_probe 观测行", proc.stderr)
        self.assertNotIn("admission_probe 观测行", proc.stdout)
        # 无 probe 行时不打印审计行(噪声零增量)——基线场景自证。
        records2, tokens2 = base_handcalc_records()
        run_dir2 = make_run_dir("probe2")
        write_fixture(run_dir2, repo_variant="astra-sim-face",
                      records=records2, token_requests=tokens2)
        proc2 = run_tool(run_dir2)
        self.assertEqual(proc2.returncode, 0, proc2.stderr)
        self.assertNotIn("admission_probe 观测行", proc2.stderr)
        self.assertNotIn("admission_probe 观测行", proc2.stdout)

    def test_unknown_kind_still_fail_closed(self):
        """非枚举内的未知 kind 依旧 fail-closed（exit 2，防白名单滥用）。"""
        records, tokens = base_handcalc_records()
        records.insert(1, self._probe_record(
            "session_A_request_0", 5, "bogus_probe_kind"))
        run_dir = make_run_dir("probe3")
        write_fixture(run_dir, repo_variant="astra-sim-face",
                      records=records, token_requests=tokens)
        proc = run_tool(run_dir)
        self.assertEqual(proc.returncode, 2)
        self.assertIn("未知 kind", proc.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
