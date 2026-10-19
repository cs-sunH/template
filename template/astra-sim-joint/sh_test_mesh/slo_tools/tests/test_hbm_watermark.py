#!/usr/bin/env python3
"""hbm_watermark.py 单测：合成 ledger 手算断言（B2 线5）+ P1 语义（四层
可信度/事件流 stats/RLE 无损/行预算/三口径锚定数值）。

运行：python3 sh_test_mesh/slo_tools/tests/test_hbm_watermark.py
  或  python3 -m unittest discover -s sh_test_mesh/slo_tools/tests -v

合成口径（全仓一致、手算可验）：
  * trace_config：layers=1、hidden_size=50、bytes_per_elem=1 →
    COEF = 2·1·50·1 = 100 B/token（实例合计）；
  * hardware 容量档 test-64b = 1000 B/NPU × prefill_ranks 6 = 6000 B/实例；
  * slo_params_manifest：watermark_sample_period_ns = 10 ns（B4 未推导的
    生产 manifest 保持 null → 5,000,000 ns 临时锚点，另有专测）；
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
import atexit
import shutil
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
    d = Path(tempfile.mkdtemp(prefix=f"slo_hbm_{tag}_"))
    atexit.register(shutil.rmtree, d, ignore_errors=True)
    return d


def write_fixture(run_dir: Path, *, repo_variant: str, records: list[dict],
                  token_requests: list[dict], bucket_ns=10,
                  bucket_null: bool = False, npu_bytes: int = 1000,
                  row_budget: int = 5_000_000,
                  model_rows: bool = False, layers: int = 1,
                  hidden_size: int = 50, bytes_per_elem: int = 1) -> Path:
    """落一个最小可重放 run_dir；返回 run_dir。

    records 为 online_decision_log.jsonl 的行（dict）；token_requests 为
    manifest.json 的 requests 数组成员。model_rows=True 时 trace_config
    追加 ffn_size/num_heads/vocab_size/mlp_variant 行（三口径剖面用）。
    layers/hidden_size/bytes_per_elem 可覆写（R17 joint 切片用
    32/4096/2 → coef=524,288 B/token 对齐在盘真实 run）。
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
        f"config,layers,{layers},,,,synthetic\n"
        f"config,hidden_size,{hidden_size},,,,synthetic\n"
        "config,num_heads,2,,,,synthetic\n"
        f"config,bytes_per_elem,{bytes_per_elem},,,,synthetic\n")
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
    """落合成 kv_delta_journal.jsonl（+可选 checksum json）。

    newline 显式固定 LF（2026-09-29，报告 3.18.1）：journal sha256 证书对
    文件字节计算，write_text 缺省 newline 会在 Windows CRLF 盘把 \n 翻译成
    \r\n → 10 例 journal-tier 测试假红；显式 LF 保证字节与证书一致。
    """
    results = run_dir / "results"
    results.mkdir(parents=True, exist_ok=True)
    import hashlib
    payload = "".join(json.dumps(row, sort_keys=True) + "\n"
                      for row in rows)
    with (results / "kv_delta_journal.jsonl").open(
            "w", encoding="utf-8", newline="\n") as handle:
        handle.write(payload)
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
        # M17（2026-09-29）：认证层显式披露——本仓自产 run 无 per-rank
        # journal 生产者，certification_layer 恒 upper_bound_only。
        self.assertEqual(summary["certification_layer"], "upper_bound_only")
        # L18（2026-09-29）：顶层逐出计数改名 _upper_bound。
        self.assertEqual(summary["total_evict_events_upper_bound"], 1)
        self.assertEqual(summary["total_evict_bytes_upper_bound"], 40)
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

    def test_negative_occupancy_guard_fails_unit(self):
        """负占用守卫（多扣 → 结构错拒绝输出）由重放基元直接钉住。

        R6-U4（2026-09-29 face 移植）修复后，LOCAL_HIT source≠target
        （账本谎报位置）路径已在上游随迁占用、不再借末段负占用崩溃兜底
        （见 test_local_hit_relocation_keeps_occupancy_consistent）；
        守卫本身仍须对任何多扣保持 fail-closed。
        """
        replay = hw.WatermarkReplay("astra-sim-face", {}, {}, COEF, None)
        replay._apply(0, 0, 50)
        with self.assertRaises(SloToolError) as ctx:
            replay._apply(1, 0, -100)
        self.assertIn("负占用", str(ctx.exception))

    def test_local_hit_relocation_keeps_occupancy_consistent(self):
        """R6-U4（2026-09-29 face 移植）：LOCAL_HIT source≠target（账本
        谎报位置）→ 会话归属切换时占用随迁——instances 占用与 instance
        指向保持一致，不再出现"占用滞留 source + 指向 target"的跟踪态
        分裂（旧分裂在此场景的唯一可见症状是末段 completion 逐出把实例 1
        占用打成负数崩溃）；位置矛盾由 local_hit_location_mismatch +
        restore_occupancy_relocated 计数留痕。

        手算（COEF=100）：A +100@t0（实例0）→ decode grow 零增量；
        LOCAL_HIT@t30 谎报目标实例 1 → 100B 随迁（实例0 归零、实例1
        100B）；completion@t50 逐出 A 100B → 实例1 归零。
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
        self.assertEqual(proc.returncode, 0, proc.stderr)
        summary = json.loads((run_dir / "summary.json").read_text())
        self.assertEqual(
            summary["anomalies"]["local_hit_location_mismatch"], 1)
        self.assertEqual(
            summary["actions"]["restore_occupancy_relocated"], 1)
        self.assertEqual(
            summary["actions"]["restore_occupancy_relocated_bytes"], 100)
        # 占用账目一致：迁移后实例0 归零、逐出后实例1 归零（无幽灵滞留）。
        self.assertEqual(
            summary["instances"]["0"]["residual_occupancy_bytes"], 0)
        self.assertEqual(
            summary["instances"]["1"]["residual_occupancy_bytes"], 0)
        # joint summary 无 face 的 total_evict_* 聚合键，按 actions 计数
        # 与逐实例 evict 列对账（口径一致）。
        self.assertEqual(summary["actions"]["evictions"], 1)
        self.assertEqual(summary["actions"]["evict_bytes"], 100)
        self.assertEqual(
            summary["instances"]["1"]["evict_events"], 1)
        self.assertEqual(
            summary["instances"]["1"]["evict_bytes"], 100)

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

    def test_scalar_restore_type_lie_fails_closed(self):
        # F2_PY（2026-09-29）RV5-F7 等价移植：标量 restore 分支的 source
        # 类型谎报（bytes 臂已由 R8-F1 严格化）——字符串静默降 None 曾抑制
        # 下游 restore_source_mismatch 诊断、bool 冒充 1 实例。在场非 int
        # （含 bool/浮点/字符串）→ fail；键缺席/None 保留缺省语义。
        scalar_mapping = {"restore": {
            "bytes_path": ["history_transfer_bytes"],
            "kind_field": "history_action",
            "source_field": "history_source_instance_index"}}

        def scalar_restore(decision):
            record = {"kind": "prefill", "tick": 5, "decision": decision}
            return hw._prefill_restore_actions(record, scalar_mapping,
                                               "w:seq=?")

        # source_field 类型谎报：字符串数字/bool/浮点。
        for bad in ("3", True, 1.0):
            with self.subTest(source=repr(bad)):
                with self.assertRaisesRegex(SloToolError,
                                            "source_instance_index"):
                    scalar_restore({"history_transfer_bytes": 100,
                                    "history_source_instance_index": bad})
        # 合法值与缺席语义不变。
        self.assertEqual(scalar_restore({
            "history_transfer_bytes": 100,
            "history_source_instance_index": 3})[0]["logged_source"], 3)
        self.assertIsNone(scalar_restore({
            "history_transfer_bytes": 100})[0]["logged_source"])
        # 对象臂（S1 history_transfer）：source 类型谎报（total_bytes 已由
        # R8-F1 严格化，此处一并钉住防回退）。
        obj_mapping = {"restore": {"bytes_path": ["history_transfer"]}}
        with self.assertRaisesRegex(SloToolError, "total_bytes"):
            hw._prefill_restore_actions(
                {"kind": "prefill", "tick": 5, "decision": {
                    "history_transfer": {"kind": "remote_load",
                                         "total_bytes": True}}},
                obj_mapping, "w:seq=?")
        with self.assertRaisesRegex(SloToolError, "source_instance_index"):
            hw._prefill_restore_actions(
                {"kind": "prefill", "tick": 5, "decision": {
                    "history_transfer": {"kind": "noc_migrate",
                                         "total_bytes": 100,
                                         "source_instance_index": "0"}}},
                obj_mapping, "w:seq=?")
        ok = hw._prefill_restore_actions(
            {"kind": "prefill", "tick": 5, "decision": {
                "history_transfer": {"kind": "noc_migrate",
                                     "total_bytes": 100,
                                     "source_instance_index": 0}}},
            obj_mapping, "w:seq=?")
        self.assertEqual(ok[0]["bytes"], 100)
        self.assertEqual(ok[0]["logged_source"], 0)

    def test_decode_move_type_lie_fails_closed(self):
        # F2_PY（2026-09-29）RV5-F7 等价移植：_decode_move_actions 的
        # source isinstance 宽松门补齐（bytes 臂已由 R8-F1 严格化）——
        # 对象臂/source_field（S2 形态）三处；缺席/None 语义不变。
        move_mapping = {"decode_move": {
            "bytes_path": ["prefill_decode_transfer"]}}
        s2_mapping = {"decode_move": {
            "bytes_path": ["prefill_decode_transfer_bytes"],
            "source_field": "prefill_instance_index"}}

        def decode_move(decision, mapping=move_mapping):
            record = {"kind": "decode", "tick": 5, "decision": decision}
            return hw._decode_move_actions(record, mapping, "w:seq=?")

        # 对象臂：source 字符串曾静默降 None。
        with self.assertRaisesRegex(SloToolError, "source_instance_index"):
            decode_move({"prefill_decode_transfer": {
                "total_bytes": 100, "source_instance_index": "0"}})
        # S2 source_field：bool 曾冒充实例 1。
        with self.assertRaisesRegex(SloToolError, "prefill_instance_index"):
            decode_move({"prefill_decode_transfer_bytes": 100,
                         "prefill_instance_index": True}, s2_mapping)
        # 合法值与缺席语义不变。
        ok = decode_move({"prefill_decode_transfer": {
            "total_bytes": 100, "source_instance_index": 0}})
        self.assertEqual(ok[0]["bytes"], 100)
        self.assertEqual(ok[0]["logged_source"], 0)
        absent = decode_move({"prefill_decode_transfer": None})
        self.assertEqual(absent[0]["bytes"], 0)
        self.assertIsNone(absent[0]["logged_source"])


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
        # astra-sim-joint（本仓）2026-09-14 登记：S3 基线映射 + joint
        # 覆盖注记（见 hbm_watermark.REPO_VARIANTS 词条）。
        self.assertEqual(
            sorted(hw.REPO_VARIANTS),
            ["astra-sim-face", "astra-sim-joint", "astra-sim-sh_1.0",
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
        # num_heads=3（基础 fixture 为 2）：三 KV rank 权重不均。
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
        atexit.register(shutil.rmtree, out, ignore_errors=True)
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


# ---------------------------------------------------------------------------
# R17（2026-09-16）golden：tracelab_session_000411 生命周期最小化切片。
#
# 素材抽自在盘真实 run（experiment/0914/4instance/runs/astra-sim-joint/
# full/results/online_decision_log.jsonl，96,189 行）＋该 run 的
# manifest.json token 事实：turn10（驻留建立）→ 四次披露逐出（他人
# prefill 行 history_evictions，victim=session_000411，合计
# 57,365,397,504 B）→ turn11/turn12 池恢复（remote_load 工作副本）。
# 真实缺口：披露逐出之间的静默段 [7,9) 与 [2,5)（10,623,221,760 B =
# 5 层 × 2,124,644,352 B/层）不落盘——旧日志重放把幻影 bytes 结转到
# turn12 prefill（remote_load 后 tracked=144,288,874,496 > 目标
# 133,984,419,840 静默保留）并在 tick=19310999728506072 的 decode_grow
# 爆"增长为负"（目标 134,913,982,464 < 跟踪 144,288,874,496）。
# 本切片手工注入两根 kind=kv_eviction 决策行（模拟 R17 仿真侧修复后
# 形态，条目带 R17-1d 新键 resident_prefix_layers_before/after），静默
# 段按层扣回——重放应零异常闭合。
#
# 切片缩减（保真度注记）：seq 35561 的 history_evictions 还含
# session_000143 victim 条目（该会话在切片内无更早生命周期行，逐出
# 未跟踪会话会先炸）——剔除；他人 stay-prefill 行的 history_transfers
# （local_hit，watermark 对 stay 行不消费）与 prefill_decode_transfer
# （joint 映射 decode_move=None 不消费）同样剔除以压缩夹具。关键数字
# （57,365,397,504 披露合计、10,623,221,760 静默、144,288,874,496 vs
# 134,913,982,464、coef=524,288）全部保真。
# ---------------------------------------------------------------------------

# joint 切片 trace_config：layers=32, hidden=4096, bytes_per_elem=2
# → coef = 2*32*4096*2 = 524,288 B/token（与真实 run 的 f(tokens) 对齐）。
JOINT411_COEF = 524288
# 逐出时点每层字节 = f(129,678)/32 = 67,988,619,264/32（turn10 终态）。
JOINT411_LAYER_BYTES = 2124644352

S411 = "tracelab_session_000411"
S333 = "tracelab_session_333"
S143 = "tracelab_session_143"


def _evict_entry(session_id: str, layer_start: int, layer_end: int,
                 total_bytes: int, reason: str, *, phase: str = "history",
                 prefix_before: int | None = None,
                 prefix_after: int | None = None) -> dict:
    """逐出条目（真实日志形态：4 shard，total_bytes 为 4 倍数）。"""
    entry = {"kind": "remote_store", "reason": reason, "phase": phase,
             "session_id": session_id, "layer_start": layer_start,
             "layer_end": layer_end, "total_bytes": total_bytes,
             "shards": [{"bytes": total_bytes // 4, "noc_hops": 0,
                         "noc_path": [16], "source_rank": 16,
                         "target_rank": 16}] * 4}
    if prefix_before is not None:
        # R17-1d 新序列化键（旧日志无——披露条目不带，注入条目带）。
        entry["resident_prefix_layers_before"] = prefix_before
        entry["resident_prefix_layers_after"] = prefix_after
    return entry


def _kv_eviction_row(seq: int, tick: int, request_id: str,
                     entry: dict) -> dict:
    """注入的 kind=kv_eviction 决策行（R17 仿真侧修复后形态）。"""
    return {"kind": "kv_eviction", "request_id": request_id,
            "priority": 0, "seq": seq, "tick": tick,
            "decision": {"evictions": [entry]}}


def joint_411_token_requests() -> list[dict]:
    """真实 run manifest.json 的 token 事实（逐字段抄录）。"""
    return [
        token_row(f"{S411}_request_000010", S411, 124958, 127293, 129678),
        token_row(f"{S333}_request_000011", S333, 374835, 504811, 511104),
        token_row(f"{S333}_request_000012", S333, 511104, 517844, 518335),
        token_row(f"{S333}_request_000018", S333, 529359, 544129, 547231),
        token_row(f"{S143}_request_000290", S143, 745128, 745407, 745434),
        token_row(f"{S411}_request_000011", S411, 129678, 254376, 254947),
        token_row(f"{S411}_request_000012", S411, 254947, 255555, 257328),
    ]


def joint_411_records(*, drop_kv: str | None = None,
                      corrupt_prefix_before: int | None = None,
                      with_kv: bool = True) -> list[dict]:
    """切片决策行（tick/bytes 逐字段抄自真实日志；seq 重排为切片序）。"""
    seq = 0

    def nxt() -> int:
        nonlocal seq
        seq += 1
        return seq

    def audit(request_id: str, tick: int) -> dict:
        return {"kind": "joint_admission", "request_id": request_id,
                "priority": 0, "seq": nxt(), "tick": tick,
                "decision": {"admission_time_ns": tick,
                             "candidates": []}}

    def prefill(request_id: str, tick: int, instance: int,
                action: str, *, evictions: list[dict] | None = None,
                transfers: list[dict] | None = None,
                discarded: int = 0) -> dict:
        return {"kind": "prefill", "request_id": request_id,
                "priority": 0, "seq": nxt(), "tick": tick,
                "decision": {
                    "prefill_instance_index": instance,
                    "joint_action": action,
                    "history_tokens_discarded": discarded,
                    "history_evictions": evictions or [],
                    # R17-2a 后 joint 映射不再读取 prefill_evictions——
                    # 字段在场（旧日志形态）但零行为差。
                    "prefill_evictions": [],
                    "history_transfers": transfers or [],
                    "history_transfer_bytes": 0,
                    "history_source_instance_index": None}}

    def decode(request_id: str, tick: int, instance: int,
               action: str) -> dict:
        return {"kind": "decode", "request_id": request_id,
                "priority": 0, "seq": nxt(), "tick": tick,
                "decision": {"decode_instance_index": instance,
                             "joint_action": action,
                             "decode_evictions": []}}

    def completion(request_id: str, tick: int, action: str,
                   *, working_copy: bool, home: int | None,
                   location: str, merges: list[dict] | None = None) -> dict:
        return {"kind": "completion", "request_id": request_id,
                "priority": 0, "seq": nxt(), "tick": tick,
                "decision": {"joint_action": action,
                             "joint_working_copy": working_copy,
                             "origin_home_instance": home,
                             "kv_location_after_completion": location,
                             "merge_transfers": merges or [],
                             "completion_evictions": []}}

    def remote_load(total: int, reason: str) -> dict:
        return {"kind": "remote_load", "reason": reason, "phase": "history",
                "session_id": S411, "layer_start": 0, "layer_end": 32,
                "total_bytes": total,
                "shards": [{"bytes": total // 4} for _ in range(4)]}

    def pool_store(total: int) -> dict:
        return {"kind": "remote_store", "reason": "merge_increment_pool_store",
                "phase": "completion", "session_id": S411,
                "layer_start": 0, "layer_end": 32, "total_bytes": total,
                "shards": [{"bytes": total // 4} for _ in range(4)]}

    # 静默段注入条目（R17-1d 新键：before/after 显式披露；tick 落在披露
    # #2（18875809191471704）与 #3（18876212627993193）之间，单调）。
    kv_a_before = 5 if corrupt_prefix_before is None \
        else corrupt_prefix_before

    records = [
        # -- turn10：session_000411 home 驻留建立（stay，local_hit [0,32)
        #    65,513,979,904 信息量；终态 f(129,678)=67,988,619,264）。
        audit(f"{S411}_request_000010", 18761909404184225),
        prefill(f"{S411}_request_000010", 18761909404184225, 0, "stay",
                transfers=[{"kind": "local_hit",
                            "reason": "history_local_reuse",
                            "phase": "history", "session_id": S411,
                            "layer_start": 0, "layer_end": 32,
                            "total_bytes": 65513979904, "shards": []}]),
        decode(f"{S411}_request_000010", 18761910200091720, 0, "stay"),
        completion(f"{S411}_request_000010", 18761961073255981, "stay",
                   working_copy=False, home=0, location="local_hbm"),
        # -- 披露逐出 #1：[9,32) = 48,866,820,096（他人 prefill 行）。
        prefill(f"{S333}_request_000011", 18875314303521902, 0, "stay",
                evictions=[_evict_entry(S411, 9, 32, 23 * JOINT411_LAYER_BYTES,
                                        "request_admission_capacity_"
                                        "suffix_half")]),
        decode(f"{S333}_request_000011", 18875450304339222, 0, "stay"),
        completion(f"{S333}_request_000011", 18875725479471704, "stay",
                   working_copy=False, home=0, location="local_hbm"),
        # -- 披露逐出 #2：[5,7) = 4,249,288,704。
        prefill(f"{S333}_request_000012", 18875809191471704, 0, "stay",
                evictions=[_evict_entry(S411, 5, 7, 2 * JOINT411_LAYER_BYTES,
                                        "request_admission_capacity_"
                                        "suffix_half")]),
        decode(f"{S333}_request_000012", 18875817457114317, 0, "stay"),
        completion(f"{S333}_request_000012", 18875839335583669, "stay",
                   working_copy=False, home=0, location="local_hbm"),
    ]
    if with_kv and drop_kv != "a":
        records.append(_kv_eviction_row(
            nxt(), 18875900000000000, f"{S333}_request_000015",
            _evict_entry(S411, 7, 9, 2 * JOINT411_LAYER_BYTES,
                         "decode_growth_capacity_suffix_half",
                         prefix_before=kv_a_before, prefix_after=5)))
    if with_kv and drop_kv != "b":
        records.append(_kv_eviction_row(
            nxt(), 18876000000000000, f"{S333}_request_000017",
            _evict_entry(S411, 2, 5, 3 * JOINT411_LAYER_BYTES,
                         "admission_failure_capacity_suffix_half",
                         prefix_before=5, prefix_after=2)))
    records.extend([
        # -- 披露逐出 #3：[1,2) = 2,124,644,352（真实行还含 session_000143
        #    victim 条目，切片内该会话无跟踪态——剔除，见类注记）。
        prefill(f"{S333}_request_000018", 18876212627993193, 0, "stay",
                evictions=[_evict_entry(S411, 1, 2, JOINT411_LAYER_BYTES,
                                        "request_admission_capacity_"
                                        "suffix_half")]),
        decode(f"{S333}_request_000018", 18876231497333192, 0, "stay"),
        completion(f"{S333}_request_000018", 18876376810949872, "stay",
                   working_copy=False, home=0, location="local_hbm"),
        # -- 披露逐出 #4：[0,1) = 2,124,644,352（full_fallback；同行还有
        #    session_000333 的 [27,32) = 51,606,077,440，保真保留）。
        prefill(f"{S143}_request_000290", 18878632243660262, 0, "stay",
                evictions=[
                    _evict_entry(S411, 0, 1, JOINT411_LAYER_BYTES,
                                 "request_admission_capacity_full_fallback"),
                    _evict_entry(S333, 27, 32, 51606077440,
                                 "request_admission_capacity_suffix_half")]),
        decode(f"{S143}_request_000290", 18878632896545399, 0, "stay"),
        completion(f"{S143}_request_000290", 18878634603614922, "stay",
                   working_copy=False, home=0, location="local_hbm"),
        # -- turn11：池恢复工作副本（copy→实例 2，remote_load
        #    67,988,619,264 = 32 层全量；完成后 remote_memory 结算、增量
        #    写池 65,677,033,472）。
        audit(f"{S411}_request_000011", 19310813084255981),
        prefill(f"{S411}_request_000011", 19310813084255981, 2, "copy",
                transfers=[remote_load(67988619264,
                                       "history_pool_restore_working_copy")]),
        decode(f"{S411}_request_000011", 19310871486054290, 2, "copy"),
        completion(f"{S411}_request_000011", 19310884578874328, "copy",
                   working_copy=True, home=0, location="remote_memory",
                   merges=[pool_store(65677033472)]),
        # -- turn12：池恢复回 home（copy→实例 0，remote_load
        #    133,665,652,736；原失败点 = 本轮 decode tick
        #    19310999728506072 的 decode_grow 增长为负）。
        audit(f"{S411}_request_000012", 19310999022999956),
        prefill(f"{S411}_request_000012", 19310999022999956, 0, "copy",
                transfers=[remote_load(133665652736,
                                       "history_pool_restore_working_copy")]),
        decode(f"{S411}_request_000012", 19310999728506072, 0, "copy"),
        completion(f"{S411}_request_000012", 19311045548518466, "copy",
                   working_copy=True, home=0, location="remote_memory",
                   merges=[pool_store(1248329728)]),
    ])
    return records


def run_joint_411(tag: str, **builder_kwargs) -> tuple[
        subprocess.CompletedProcess, dict]:
    run_dir = make_run_dir(tag)
    write_fixture(run_dir, repo_variant="astra-sim-joint",
                  records=joint_411_records(**builder_kwargs),
                  token_requests=joint_411_token_requests(),
                  layers=32, hidden_size=4096, bytes_per_elem=2,
                  npu_bytes=1 << 40)
    proc = run_tool(run_dir)
    summary_path = run_dir / "summary.json"
    summary = json.loads(summary_path.read_text()) \
        if summary_path.exists() else {}
    return proc, summary


class JointSession411GoldenTests(unittest.TestCase):
    """R17 golden：session_000411 生命周期切片（重放闭合/水位精确/敏感
    性 fail-closed）。"""

    def test_replay_closes_and_watermark_exact(self):
        """断言①：原失败点（tick=19310999728506072 decode_grow）不再爆，
        重放零异常闭合；断言②：水位终值/峰值/逐出账精确断言。"""
        proc, summary = run_joint_411("j411g")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(summary["trust_tier"], "upper_bound_only")
        self.assertEqual(summary["coef_bytes_per_token"], JOINT411_COEF)
        # 零异常：anomalies 全零（无 bytes/source/kind/location 错配）。
        self.assertEqual({k: v for k, v in summary["anomalies"].items()
                          if v}, {})
        # session_000411 生命周期闭合：终态无驻留（turn12 remote_memory
        # 结算），实例 2 工作副本释放后归零。
        self.assertEqual(summary["instances"]["2"]
                         ["residual_occupancy_bytes"], 0)
        self.assertEqual(summary["instances"]["2"]
                         ["peak_occupancy_bytes"], 133665652736)
        # 实例 0 终值 = 残留三方（000411=0；000333=f(547231)−[27,32)；
        # 000143=f(745434)）：
        #   286,906,646,528 − 51,606,077,440 + 390,822,100,992。
        self.assertEqual(summary["instances"]["0"]
                         ["residual_occupancy_bytes"], 626122670080)
        # 实例 0 峰值 = turn12 decode 后（000411 工作副本
        # f(257,328)=134,913,982,464 叠加残留 626,122,670,080）。
        self.assertEqual(summary["instances"]["0"]
                         ["peak_occupancy_bytes"], 761036652544)
        # 逐出账：披露 4 条（57,365,397,504）+ 注入 2 条（10,623,221,760）
        # + session_000333 victim 1 条（51,606,077,440）。
        # L18（2026-09-29）：顶层逐出计数改名 _upper_bound（来源 =
        # decision-log 重放对照列，journal 权威路径下不混用）。
        self.assertEqual(summary["total_evict_events_upper_bound"], 7)
        self.assertEqual(summary["total_evict_bytes_upper_bound"],
                         119594696704)
        self.assertIn("eviction_counters_source", summary)
        # R17-2d 哨兵：kv_eviction 逐条计数 + reason 分桶随 summary 落盘。
        self.assertEqual(summary["actions"]["evict_entries_kv_eviction"], 2)
        self.assertEqual(summary["actions"][
            "evict_reason_kv_eviction:"
            "decode_growth_capacity_suffix_half"], 1)
        self.assertEqual(summary["actions"][
            "evict_reason_kv_eviction:"
            "admission_failure_capacity_suffix_half"], 1)

    def test_sensitivity_dropping_either_kv_eviction_row_fails(self):
        """断言③（敏感性）：挖空任一根注入 kv_eviction 行 → 幻影 bytes
        结转到 turn12 prefill（remote_load 后 tracked > 目标）→
        R17-3 prefill_shrink fail-closed（非静默），判据不是空转。"""
        for drop in ("a", "b"):
            with self.subTest(drop=drop):
                proc, _ = run_joint_411(f"j411s{drop}", drop_kv=drop)
                self.assertEqual(proc.returncode, 2, proc.stderr)
                self.assertIn("prefill_shrink", proc.stderr)

    def test_kv_eviction_prefix_mismatch_fails_closed(self):
        """R17-4 连锁不变量：注入条目 resident_prefix_layers_before 与
        跟踪前缀不符（披露 #2 后应=5，条目谎报 7）→ fail-closed。"""
        proc, _ = run_joint_411("j411p", corrupt_prefix_before=7)
        self.assertEqual(proc.returncode, 2, proc.stderr)
        self.assertIn("逐出前缀不符", proc.stderr)

    def test_kv_eviction_tick_regression_fails_closed(self):
        """R17-2b 次序纪律：第二根 kv_eviction 行 tick 早于第一根（仍晚于
        前一决策行）→ 分支自行承担的 tick 单调检查 fail-closed（消息含
        kind=kv_eviction 与两侧 tick），不得绕过主循环判罚语义。"""
        run_dir = make_run_dir("j411t")
        records = []
        for record in joint_411_records():
            if record.get("kind") == "kv_eviction" \
                    and record["tick"] == 18876000000000000:
                record = dict(record)
                record["tick"] = 18875850000000000  # 回退到第一根之前
            records.append(record)
        write_fixture(run_dir, repo_variant="astra-sim-joint",
                      records=records,
                      token_requests=joint_411_token_requests(),
                      layers=32, hidden_size=4096, bytes_per_elem=2,
                      npu_bytes=1 << 40)
        proc = run_tool(run_dir)
        self.assertEqual(proc.returncode, 2, proc.stderr)
        self.assertIn("tick 回退", proc.stderr)
        self.assertIn("kv_eviction", proc.stderr)

    def test_prefill_history_tokens_discarded_rejected(self):
        """R17-3：history_tokens_discarded > 0（丢弃历史词表）显式拒收。"""
        run_dir = make_run_dir("j411d")
        records = joint_411_records()
        for record in records:
            if record["request_id"] == f"{S411}_request_000010" \
                    and record["kind"] == "prefill":
                record["decision"]["history_tokens_discarded"] = 7
        write_fixture(run_dir, repo_variant="astra-sim-joint",
                      records=records,
                      token_requests=joint_411_token_requests(),
                      layers=32, hidden_size=4096, bytes_per_elem=2,
                      npu_bytes=1 << 40)
        proc = run_tool(run_dir)
        self.assertEqual(proc.returncode, 2, proc.stderr)
        self.assertIn("history_tokens_discarded", proc.stderr)


# ---------------------------------------------------------------------------
# W1（2026-09-27）：P→D 跨实例迁移重放钉测。
#
# 素材原抽自 b4 冒烟 combo_T 臂 session_3 的 request_0/request_1 生命
# 周期（2026-10-14 远读退役：request_1 改 recompute 全量工作副本形态，
# 字节按三动作口径重算；2026-10-15 recompute 退役：request_1 承载动作
# 改 copy 同几何——重放侧仅按 stay/非 stay 分流，迁移/合并几何不变）：
# request_0 stay-prefill@1 → decode 迁 1→5
# （f(11747)）；request_1 copy 工作副本@0（f(18271)）→ decode 迁
# 0→4 → 完成 forward 合并 4→home 5（f(18272)）。修复前 _joint_decode
# 只改 instance 不搬 P 工作副本：源实例残留幻影、完成结算在目标实例
# 多扣 → 负占用 fail-closed（b4 五臂 exit=2 的根因）。另附同实例
# （P==D + local_hit 披露）对照请求：交割字节为信息量，占用不变。
# coef=524,288（layers=32/hidden=4096/bpe=2，与在盘真实 run 对齐）。
# ---------------------------------------------------------------------------

MIG_COEF = 524288
MIG_F_U0 = 11747 * MIG_COEF          # 6,158,811,136（request_0 P→D 搬移）
MIG_F_D0 = 11837 * MIG_COEF          # 6,205,997,056（request_0 终态）
MIG_WORK1 = 18271 * MIG_COEF         # 9,579,266,048（request_1 P 末工作副本）
MIG_MERGE1 = 18272 * MIG_COEF        # 9,579,790,336（request_1 完成合并）
MIG_F_FINAL = MIG_F_D0 + MIG_MERGE1  # 15,785,787,392（session_3 终态）


def _mig_move_transfer(source: int, total_bytes: int, session: str,
                       kind: str = "noc_migrate",
                       reason: str = "prefill_decode_instance_migrate") -> dict:
    """decode 行 prefill_decode_transfer 披露（真实日志形态，shards 略）。"""
    return {"kind": kind, "phase": "prefill_decode", "reason": reason,
            "session_id": session, "source_instance_index": source,
            "target_instance_index": source,
            "layer_start": 0, "layer_end": 32,
            "resident_prefix_layers_before": 0,
            "resident_prefix_layers_after": 32,
            "total_bytes": total_bytes, "shards": []}


def migration_records() -> list[dict]:
    """combo_T session_3 切片 + 同实例对照请求（seq/tick 合成单调）。"""
    seq = 0

    def nxt() -> int:
        nonlocal seq
        seq += 1
        return seq

    def prefill(request_id, tick, instance, action):
        return {"kind": "prefill", "request_id": request_id,
                "priority": 0, "seq": nxt(), "tick": tick,
                "decision": {"prefill_instance_index": instance,
                             "joint_action": action,
                             "history_tokens_discarded": 0,
                             "history_evictions": [],
                             "history_transfers": [],
                             "history_transfer_bytes": 0,
                             "history_source_instance_index": None}}

    def decode(request_id, tick, instance, action, transfer):
        return {"kind": "decode", "request_id": request_id,
                "priority": 0, "seq": nxt(), "tick": tick,
                "decision": {"decode_instance_index": instance,
                             "joint_action": action,
                             "prefill_decode_transfer": transfer,
                             "decode_evictions": []}}

    def completion(request_id, tick, *, direction, working_copy, merges,
                   home_flipped=None, origin_home=None):
        return {"kind": "completion", "request_id": request_id,
                "priority": 0, "seq": nxt(), "tick": tick,
                "decision": {"joint_action": "copy",
                             "merge_direction": direction,
                             "joint_working_copy": working_copy,
                             "merge_transfers": merges,
                             "home_flipped_to": home_flipped,
                             "origin_home_instance": origin_home,
                             "kv_location_after_completion": "local_hbm",
                             "completion_evictions": []}}

    def merge_noc(source, total):
        return {"kind": "noc_migrate", "phase": "completion",
                "reason": "merge_working_copy_to_home",
                "session_id": "session_3",
                "source_instance_index": source,
                "layer_start": 0, "layer_end": 32, "total_bytes": total,
                "shards": []}

    return [
        # request_0：stay-prefill@1 → P→D 迁 1→5 → 本地提交（stay）。
        prefill("s3_r0", 100, 1, "stay"),
        decode("s3_r0", 200, 5, "stay",
               _mig_move_transfer(1, MIG_F_U0, "session_3")),
        completion("s3_r0", 300, direction="stay", working_copy=False,
                   merges=[]),
        # request_1：copy 工作副本@0（f(18271) 全量，无 history 腿起步——
        # 合成口径）→ P→D 迁 0→4 → forward 合并 4→home 5（home_flipped_to=5）。
        prefill("s3_r1", 400, 0, "copy"),
        decode("s3_r1", 500, 4, "copy",
               _mig_move_transfer(0, MIG_WORK1, "session_3")),
        completion("s3_r1", 600, direction="forward", working_copy=True,
                   merges=[merge_noc(4, MIG_MERGE1)],
                   home_flipped=5, origin_home=1),
        # 同实例对照：stay-prefill@3 → P==D local_hit 交割（字节为信息量，
        # 占用不变）→ 本地提交。
        prefill("s9_r0", 700, 3, "stay"),
        decode("s9_r0", 800, 3, "stay",
               _mig_move_transfer(3, 100 * MIG_COEF, "session_9",
                                  kind="local_hit",
                                  reason="prefill_decode_local_reuse")),
        completion("s9_r0", 900, direction="stay", working_copy=False,
                   merges=[]),
    ]


def migration_token_requests() -> list[dict]:
    """token 事实：request_0 (0,11747,11837)、request_1
    (11837,18271,18272)、对照 (0,100,101)——逐字段对应真实 manifest。"""
    return [
        token_row("s3_r0", "session_3", 0, 11747, 11837),
        token_row("s3_r1", "session_3", 11837, 18271, 18272),
        token_row("s9_r0", "session_9", 0, 100, 101),
    ]


def run_migration(tag: str, mutate=None) -> tuple[subprocess.CompletedProcess,
                                                  dict]:
    """跑迁移切片；mutate(records) 供负例篡改（深拷贝后注入）。"""
    import copy
    records = migration_records()
    if mutate is not None:
        records = mutate(copy.deepcopy(records))
    run_dir = make_run_dir(tag)
    write_fixture(run_dir, repo_variant="astra-sim-joint",
                  records=records,
                  token_requests=migration_token_requests(),
                  layers=32, hidden_size=4096, bytes_per_elem=2,
                  npu_bytes=1 << 40)
    proc = run_tool(run_dir)
    summary_path = run_dir / "summary.json"
    summary = json.loads(summary_path.read_text()) \
        if summary_path.exists() else {}
    return proc, summary


class JointPrefillDecodeMigrationTests(unittest.TestCase):
    """W1 钉测：joint decode 行 P→D 迁移重放（源扣/目标加）+ fail-closed。"""

    def test_cross_instance_migration_replay_closes(self):
        """修复主线：跨实例 P→D 迁移逐字节搬移——重放零异常闭合，工作
        副本精确落位（实例 4 峰值=合并前工作副本），无幻影残留、无负
        占用（b4 combo_T 型 exit=2 不再复现）。"""
        proc, summary = run_migration("wmig1")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual({k: v for k, v in summary["anomalies"].items()
                          if v}, {})
        actions = summary["actions"]
        # 两次跨实例搬移，字节精确（f(11747) + f(6434)）。
        self.assertEqual(actions["joint_decode_move"], 2)
        self.assertEqual(actions["joint_decode_move_bytes"],
                         MIG_F_U0 + MIG_WORK1)
        # 实例 4：峰值=decode 增长后的 f(18272)，结算后残留归零——修复
        # 前此处只加过增量、被合并整扣成负占用。
        self.assertEqual(summary["instances"]["4"]["peak_occupancy_bytes"],
                         MIG_MERGE1)
        self.assertEqual(summary["instances"]["4"]["residual_occupancy_bytes"],
                         0)
        # 源实例 0/1 幻影清零；home 5 终态 = base f(11837) + 合并
        # f(18272)（copy 全量工作副本整份并入）。
        self.assertEqual(summary["instances"]["0"]["residual_occupancy_bytes"],
                         0)
        self.assertEqual(summary["instances"]["1"]["residual_occupancy_bytes"],
                         0)
        self.assertEqual(summary["instances"]["5"]["residual_occupancy_bytes"],
                         MIG_F_FINAL)
        self.assertEqual(summary["instances"]["5"]["peak_occupancy_bytes"],
                         MIG_F_FINAL)
        # 同实例对照：local_hit 交割不改占用，会话驻留 f(101) 不动。
        self.assertEqual(summary["instances"]["3"]["residual_occupancy_bytes"],
                         101 * MIG_COEF)
        # 两个 stay-prefill 行各自计数（request_0/request_1 会话对照）。
        self.assertEqual(actions["joint_stay_prefill"], 2)

    def test_same_instance_disclosure_is_occupancy_noop(self):
        """同实例路径回归：P==D + local_hit 披露行不产生任何搬移/增量
        计数（行为与修复前逐字段一致）。"""
        proc, summary = run_migration("wmig2")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        actions = summary["actions"]
        self.assertNotIn("joint_decode_stay", actions)
        # 对照请求唯一增长 = decode 增量 1 token（f(101)−f(100)）。
        self.assertEqual(summary["instances"]["3"]["peak_occupancy_bytes"],
                         101 * MIG_COEF)

    def test_migration_bytes_mismatch_fails_closed(self):
        """迁移 bytes ≠ 跟踪工作副本（整份搬移口径）→ fail-closed。"""
        def mutate(records):
            for row in records:
                if row["request_id"] == "s3_r1" and row["kind"] == "decode":
                    row["decision"]["prefill_decode_transfer"]["total_bytes"] \
                        += MIG_COEF
            return records
        proc, _ = run_migration("wmig3", mutate)
        self.assertEqual(proc.returncode, 2, proc.stderr)
        self.assertIn("整份搬移", proc.stderr)

    def test_migration_source_mismatch_fails_closed(self):
        """披露源端 ≠ 跟踪工作副本实例 → fail-closed。"""
        def mutate(records):
            for row in records:
                if row["request_id"] == "s3_r1" and row["kind"] == "decode":
                    row["decision"]["prefill_decode_transfer"][
                        "source_instance_index"] = 3
            return records
        proc, _ = run_migration("wmig4", mutate)
        self.assertEqual(proc.returncode, 2, proc.stderr)
        self.assertIn("跟踪工作副本实例", proc.stderr)

    def test_cross_instance_without_disclosure_fails_closed(self):
        """跨实例 decode 行无 prefill_decode_transfer 披露 → fail-closed
        （迁移字节必须落盘，拒绝静默漏记）。"""
        def mutate(records):
            for row in records:
                if row["request_id"] == "s3_r1" and row["kind"] == "decode":
                    row["decision"]["prefill_decode_transfer"] = None
            return records
        proc, _ = run_migration("wmig5", mutate)
        self.assertEqual(proc.returncode, 2, proc.stderr)
        self.assertIn("prefill_decode_transfer", proc.stderr)

    def test_noc_transfer_with_same_instance_fails_closed(self):
        """noc_migrate 披露但源==目标（manager 不产此形态）→ fail-closed。"""
        def mutate(records):
            for row in records:
                if row["request_id"] == "s3_r0" and row["kind"] == "decode":
                    row["decision"]["decode_instance_index"] = 1
            return records
        proc, _ = run_migration("wmig6", mutate)
        self.assertEqual(proc.returncode, 2, proc.stderr)
        self.assertIn("同实例交割", proc.stderr)

    def test_local_hit_across_instances_fails_closed(self):
        """local_hit 披露但源≠目标（交割/搬移形态互斥）→ fail-closed。"""
        def mutate(records):
            for row in records:
                if row["request_id"] == "s9_r0" and row["kind"] == "decode":
                    row["decision"]["decode_instance_index"] = 2
                    row["decision"]["prefill_decode_transfer"][
                        "source_instance_index"] = 3
            return records
        proc, _ = run_migration("wmig7", mutate)
        self.assertEqual(proc.returncode, 2, proc.stderr)
        self.assertIn("与搬移语义矛盾", proc.stderr)

    def test_non_integer_decode_instance_fails_closed(self):
        """joint decode 决策缺整数 decode_instance_index → fail-closed
        （与 _joint_prefill 同级容错，不产 None 假实例桶）。"""
        def mutate(records):
            for row in records:
                if row["request_id"] == "s3_r1" and row["kind"] == "decode":
                    row["decision"]["decode_instance_index"] = None
            return records
        proc, _ = run_migration("wmig8", mutate)
        self.assertEqual(proc.returncode, 2, proc.stderr)
        self.assertIn("decode_instance_index", proc.stderr)


# ---------------------------------------------------------------------------
# G1（2026-09-27）钉测：sh30 D 边界三决策行（boundary_duplicate_ignored /
# boundary_commit_fallback / boundary_terminal）入 joint_audit_kinds 词表
# ——混插后重放零异常闭合（ALL-STEPS-PASSED 等价口径：exit 0 且 stderr
# 空），且主账字节守恒逐位不变（对照不插行基线对拍 instances/actions/
# anomalies 三表 + instances.csv/series.csv 字节级一致）。生产端无条件落
# 行的写点符号锚：_on_prefill_drain（sh30 的 _on_prefill_drain 换址提交段）、
# _settle_terminal_at_boundary（sh30）。
# ---------------------------------------------------------------------------

class BoundaryAuditKindTests(unittest.TestCase):
    """G1 词表补全钉测（审计行混插对拍形态）：boundary_* 行零重放
    语义差 + 未知 kind 门不误红。"""

    COEF_G1 = 100  # B/token（layers=1, hidden=50, bytes_per_elem=1）

    @staticmethod
    def _seq() -> "_SeqCounter":
        return _SeqCounter()

    @staticmethod
    def _records(seq: "_SeqCounter") -> list[dict]:
        """单请求 joint stay 生命周期（coef=100：f(10)=1000、f(12)=1200
        → 残差 1200、峰 1200）。"""
        request_id = "s0_r0"
        return [
            {"kind": "prefill", "request_id": request_id, "priority": 0,
             "seq": seq.nxt(), "tick": 100,
             "decision": {"prefill_instance_index": 0,
                          "joint_action": "stay",
                          "history_tokens_discarded": 0,
                          "history_evictions": [],
                          "history_transfers": [],
                          "history_transfer_bytes": 0,
                          "history_source_instance_index": None}},
            {"kind": "decode", "request_id": request_id, "priority": 0,
             "seq": seq.nxt(), "tick": 110,
             "decision": {"decode_instance_index": 0,
                          "joint_action": "stay",
                          "decode_evictions": []}},
            {"kind": "completion", "request_id": request_id, "priority": 0,
             "seq": seq.nxt(), "tick": 190,
             "decision": {"joint_action": "stay",
                          "merge_direction": "stay",
                          "joint_working_copy": False,
                          "merge_transfers": [],
                          "home_flipped_to": None,
                          "origin_home_instance": 0,
                          "kv_location_after_completion": "local_hbm",
                          "completion_evictions": []}},
        ]

    @staticmethod
    def _boundary_rows(seq: "_SeqCounter") -> list[dict]:
        """三种 boundary 决策行（字段面逐符号对齐生产写点；消费端在
        request_id/tick 门之前登记跳过，不载运任何账本事件）。"""
        request_id = "s0_r0"
        return [
            # sh30 _on_prefill_drain 幂等门（D20 幂等门重复 P_COMPUTE_DONE
            # 通知披露）。
            {"kind": "boundary_duplicate_ignored", "request_id": request_id,
             "priority": 0, "seq": seq.nxt(), "tick": 104,
             "decision": {"joint_stage": "P_COMPUTE_DONE",
                          "joint_boundary_epoch": 0,
                          "rule": "duplicate P_COMPUTE_DONE notification "
                                  "is idempotent (v2.0 sec.16.4 stage "
                                  "epoch)"}},
            # sh30 _on_prefill_drain 换址回退段（Eq.1c 换址提交失败回退披露）。
            {"kind": "boundary_commit_fallback", "request_id": request_id,
             "priority": 0, "seq": seq.nxt(), "tick": 107,
             "decision": {"failed_instance_index": 1,
                          "failed_action": "remote",
                          "fallback_instance_index": 0,
                          "stage_epoch": 0, "reason": "synthetic commit "
                          "failure",
                          "rule": "Eq.1c fallback: continuation stays "
                                  "legal when the switch commit fails "
                                  "without irreversible work"}},
            # sh30 _settle_terminal_at_boundary（已知终止直接结算
            # 披露）。
            {"kind": "boundary_terminal", "request_id": request_id,
             "priority": 0, "seq": seq.nxt(), "tick": 108,
             "decision": {"instance_index": 0,
                          "rule": "observed terminal at P_COMPUTE_DONE: "
                                  "skip D selection, settle directly "
                                  "(v2.0 sec.16.4)"}},
        ]

    def _run(self, tag: str, records: list[dict]) -> tuple[
            subprocess.CompletedProcess, dict, Path]:
        run_dir = make_run_dir(tag)
        write_fixture(run_dir, repo_variant="astra-sim-joint",
                      records=records,
                      token_requests=[token_row("s0_r0", "session_0",
                                                0, 10, 12)],
                      layers=1, hidden_size=50, bytes_per_elem=1,
                      npu_bytes=1 << 40)
        proc = run_tool(run_dir)
        summary = json.loads(
            (run_dir / "summary.json").read_text()) \
            if (run_dir / "summary.json").exists() else {}
        return proc, summary, run_dir

    def test_boundary_rows_mixed_replay_still_closes(self):
        """钉测①：三种 boundary 行混插（两两夹在主账行之间）后重放零
        异常闭合（exit 0、stderr 空），残差/峰与纯主账场景一致——缺词
        表时未知-kind 门 fail-closed 误红的主证。"""
        seq = self._seq()
        records = self._records(seq)
        mixed = records[:1] + self._boundary_rows(seq) + records[1:]
        proc, summary, _ = self._run("g1mix", mixed)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertNotIn("fail-closed", proc.stderr)
        inst = summary["instances"]["0"]
        self.assertEqual(inst["residual_occupancy_bytes"], 12 * self.COEF_G1)
        self.assertEqual(inst["peak_occupancy_bytes"], 12 * self.COEF_G1)

    def test_w5_settle_done_row_mixed_replay_zero_delta(self):
        """W5（§7.4）钉测：settle_done 统一结算披露行混插后重放零异常
        闭合 + 主账字节守恒不变（audit 词表纳新 = 零重放语义差；KV 账
        本走 completion 行重放，本行仅审计——与 merge_done 行同款）。"""
        seq = self._seq()
        records = self._records(seq)
        settle_row = {
            "kind": "settle_done", "request_id": "s0_r0",
            "priority": 0, "seq": seq.nxt(), "tick": 195,
            "decision": {
                "settle_done_ns": 195, "service_done_ns": 190,
                "settle_committed_ns": 190, "merge_done_ns": None,
                "merge_payload_bytes": 0,
                "settlement_policy": "consolidate",
                "settle_state": "SETTLE_DONE"},
        }
        mixed = records + [settle_row]
        base_proc, base_summary, base_dir = self._run("w5s0", records)
        mixed_proc, mixed_summary, mixed_dir = self._run("w5s1", mixed)
        self.assertEqual(base_proc.returncode, 0, base_proc.stderr)
        self.assertEqual(mixed_proc.returncode, 0, mixed_proc.stderr)
        self.assertEqual(base_summary["instances"],
                         mixed_summary["instances"])
        self.assertEqual(base_summary["actions"], mixed_summary["actions"])

    def test_boundary_rows_zero_ledger_delta_bitwise(self):
        """钉测②：主账字节守恒逐位不变——混插行与不插行基线对拍，
        instances/actions/anomalies 三表相等 + instances.csv/series.csv
        字节级一致（boundary 行跳过 = 零重放语义差）。"""
        base_seq = self._seq()
        base_records = self._records(base_seq)
        mixed_seq = self._seq()
        mixed_records = (self._records(mixed_seq)[:1]
                         + self._boundary_rows(mixed_seq)
                         + self._records(mixed_seq)[1:])
        base_proc, base_summary, base_dir = self._run("g1bit0", base_records)
        mixed_proc, mixed_summary, mixed_dir = self._run(
            "g1bit1", mixed_records)
        self.assertEqual(base_proc.returncode, 0, base_proc.stderr)
        self.assertEqual(mixed_proc.returncode, 0, mixed_proc.stderr)
        self.assertEqual(base_summary["instances"],
                         mixed_summary["instances"])
        self.assertEqual(base_summary["actions"], mixed_summary["actions"])
        self.assertEqual(base_summary["anomalies"],
                         mixed_summary["anomalies"])
        # CSV provenance 列内嵌 run_dir 绝对路径，剥离后字节级对拍。
        def strip_path(data: bytes, run_dir: Path) -> bytes:
            return data.replace(str(run_dir).encode(), b"")

        self.assertEqual(
            strip_path((base_dir / "instances.csv").read_bytes(), base_dir),
            strip_path((mixed_dir / "instances.csv").read_bytes(), mixed_dir))
        self.assertEqual(
            strip_path((base_dir / "series.csv").read_bytes(), base_dir),
            strip_path((mixed_dir / "series.csv").read_bytes(), mixed_dir))

    def test_boundary_rows_need_not_bypass_fail_closed(self):
        """词表外边界同形行（篡改 kind 后缀）仍被未知-kind 门拒收：
        本修复是纳新入词表，不是放开校验。"""
        seq = self._seq()
        records = self._records(seq)
        row = self._boundary_rows(seq)[0]
        row["kind"] = "boundary_duplicate_ignored_x"
        mixed = records[:1] + [row] + records[1:]
        proc, _, _ = self._run("g1unk", mixed)
        self.assertEqual(proc.returncode, 2, proc.stderr)
        self.assertIn("未知 kind='boundary_duplicate_ignored_x'",
                      proc.stderr)


class _SeqCounter:
    """局部 seq 计数器（不触碰模块级 *_record.next_seq 共享状态）。"""

    def __init__(self) -> None:
        self.value = 100

    def nxt(self) -> int:
        self.value += 1
        return self.value


# ---------------------------------------------------------------------------
# W3/C19 词表对账（自远读退役批整删的 lease 钉测文件迁入的幸存面：
# 纯审计行零占用差 + 未知 kind 门不放松——lease 账本行重放用例随
# staging lease 水位退役删除，2026-10-14）。
# ---------------------------------------------------------------------------

class AuditRowVocabularyTests(unittest.TestCase):
    """新 kind 审计行混插对账：零占用差钉测 + fail-closed 门不放松。"""

    COEF_V = 100  # B/token（layers=1, hidden=50, bytes_per_elem=1）

    @staticmethod
    def _seq() -> "_SeqCounter":
        return _SeqCounter()

    @classmethod
    def _records(cls, seq: "_SeqCounter") -> list[dict]:
        """单请求 joint stay 生命周期（f(10)=1000、f(12)=1200 → 残差
        1200、峰 1200）。"""
        request_id = "s0_r0"
        return [
            {"kind": "prefill", "request_id": request_id, "priority": 0,
             "seq": seq.nxt(), "tick": 100,
             "decision": {"prefill_instance_index": 0,
                          "joint_action": "stay",
                          "history_tokens_discarded": 0,
                          "history_evictions": [],
                          "history_transfers": [],
                          "history_transfer_bytes": 0,
                          "history_source_instance_index": None}},
            {"kind": "decode", "request_id": request_id, "priority": 0,
             "seq": seq.nxt(), "tick": 110,
             "decision": {"decode_instance_index": 0,
                          "joint_action": "stay",
                          "decode_evictions": []}},
            {"kind": "completion", "request_id": request_id, "priority": 0,
             "seq": seq.nxt(), "tick": 190,
             "decision": {"joint_action": "stay",
                          "merge_direction": "stay",
                          "joint_working_copy": False,
                          "merge_transfers": [],
                          "home_flipped_to": None,
                          "origin_home_instance": 0,
                          "kv_location_after_completion": "local_hbm",
                          "completion_evictions": []}},
        ]

    @classmethod
    def _audit_rows(cls, seq: "_SeqCounter") -> list[dict]:
        """混插全部纯审计 kind（C6/C8/C14/run 级收尾；readplan 两行为
        历史 run 词表——生产端 2026-10-14 起不再发射，零占用语义不变）。"""
        return [
            {"kind": "joint_sample_contended", "request_id": "s0_r0",
             "priority": 0, "seq": seq.nxt(), "tick": 113,
             "decision": {"sample": "decode", "contention": 3,
                          "next_turn_request_id": None}},
            {"kind": "joint_sample_contended", "request_id": "",
             "priority": 0, "seq": seq.nxt(), "tick": 116,
             "decision": {"sample": "decode", "contention": 2,
                          "next_turn_request_id": None}},
            {"kind": "readplan_reconcile", "request_id": "",
             "priority": 0, "seq": seq.nxt(), "tick": 118,
             "decision": {"reconciled_units": 4, "residual_units": 0}},
            {"kind": "readplan_settle", "request_id": "",
             "priority": 0, "seq": seq.nxt(), "tick": 305,
             "decision": {"settled_units": 4, "carry_over_units": 0}},
            {"kind": "merge_done", "request_id": "s0_r0",
             "priority": 0, "seq": seq.nxt(), "tick": 301,
             "decision": {"merge_done_ns": 301}},
            {"kind": "link_telemetry_coverage", "request_id": "",
             "priority": 0, "seq": seq.nxt(), "tick": 0,
             "decision": {"collective_coverage": True,
                          "telemetry_degradation": 0,
                          "telemetry_epoch_count": 3,
                          "telemetry_key_absent_seen": False,
                          "telemetry_rate_entries": 128,
                          "telemetry_sample_count": 4096,
                          "telemetry_window_broken": False,
                          "telemetry_zero_rate_samples_dropped": 0}},
            {"kind": "joint_decision_metrics", "request_id": "",
             "priority": 0, "seq": seq.nxt(), "tick": 0,
             "decision": {"quota_mode": "aimd",
                          "action_selection_counts": {"stay": 2},
                          "quota_deferred_wait_counts": {"capacity": 0},
                          "quota_deferred_dwell": {"count": 0},
                          "admission_decision_wall_ns": {
                              "count": 2, "total_ns": 1000, "avg_ns": 500,
                              "max_ns": 600},
                          "quota_verdict_layer": {
                              "wall_ns_total": 800, "candidate_checks": 6},
                          "quota_events": {"admits": 2, "releases": 2,
                                           "merge_reserves_created": 0,
                                           "oneshot_overflows": 0},
                          "aimd_action_counts": {}}},
        ]

    def _run(self, tag: str, *, with_audit: bool
             ) -> tuple[subprocess.CompletedProcess, dict]:
        seq = self._seq()
        records = self._records(seq)
        if with_audit:
            records = records + self._audit_rows(seq)
        tokens = [token_row("s0_r0", "session_0", 0, 10, 12)]
        run_dir = make_run_dir(tag)
        write_fixture(run_dir, repo_variant="astra-sim-joint",
                      records=records, token_requests=tokens,
                      layers=1, hidden_size=50, bytes_per_elem=1,
                      npu_bytes=1 << 40)
        proc = run_tool(run_dir)
        summary = json.loads((run_dir / "summary.json").read_text()) \
            if (run_dir / "summary.json").exists() else {}
        return proc, summary

    def test_mixed_sequence_full_closure(self):
        """审计行全量混插后重放零异常闭合，残差/峰与纯账本场景一致
        （词表修复主证：缺词表时这些行会触发未知 kind fail-closed）。"""
        proc, summary = self._run("w3mix", with_audit=True)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        inst = summary["instances"]["0"]
        self.assertEqual(inst["residual_occupancy_bytes"], 12 * self.COEF_V)
        self.assertEqual(inst["peak_occupancy_bytes"], 12 * self.COEF_V)

    def test_audit_rows_prove_zero_occupancy_delta(self):
        """钉测：readplan_reconcile/settle（历史 run 词表）、
        joint_sample_contended（含 request_id="" 形态）、merge_done、
        link_telemetry_coverage、joint_decision_metrics 全部不产生占用
        ——同账本加插前后 instances/actions 逐字节相等。"""
        base_proc, base_summary = self._run("w3aud0", with_audit=False)
        mixed_proc, mixed_summary = self._run("w3aud1", with_audit=True)
        self.assertEqual(base_proc.returncode, 0, base_proc.stderr)
        self.assertEqual(mixed_proc.returncode, 0, mixed_proc.stderr)
        self.assertEqual(base_summary["instances"],
                         mixed_summary["instances"])
        self.assertEqual(base_summary["actions"], mixed_summary["actions"])
        self.assertEqual(base_summary["anomalies"],
                         mixed_summary["anomalies"])

    def test_unknown_kind_still_fails_closed(self):
        """词表外 kind 维持 fail-closed（纳新入词表不是放开校验）。"""

        def mutate(records):
            records.append({"kind": "mystery_probe_row",
                            "request_id": "s0_r0", "priority": 0,
                            "seq": 500, "tick": 400,
                            "decision": {"payload": True}})
            return records

        seq = self._seq()
        records = mutate(self._records(seq))
        tokens = [token_row("s0_r0", "session_0", 0, 10, 12)]
        run_dir = make_run_dir("w3unk")
        write_fixture(run_dir, repo_variant="astra-sim-joint",
                      records=records, token_requests=tokens,
                      layers=1, hidden_size=50, bytes_per_elem=1,
                      npu_bytes=1 << 40)
        proc = run_tool(run_dir)
        self.assertEqual(proc.returncode, 2, proc.stderr)
        self.assertIn("未知 kind='mystery_probe_row'", proc.stderr)
