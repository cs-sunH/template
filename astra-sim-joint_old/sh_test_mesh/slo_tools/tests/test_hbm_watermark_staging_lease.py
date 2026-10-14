#!/usr/bin/env python3
"""test_hbm_watermark_staging_lease.py -- W3（2026-09-27）钉测：joint
决策日志 staging lease 账本行与其余新 kind 的重放词表全量对账。

素材口径（schema 逐符号锚定发射面，字段名不发明新口径）：
  * staging_lease_acquire（sh30_online_scheduler 准入相 带
    phase="prefill_read"/scans_per_window；drain 相（无 phase 键））：
    decision={lease_id, boundary_epoch, exec_instance_index,
    staging_shard_bytes（逐 rank）, staging_tile_tokens[, phase[,
    scans_per_window]]}；lease_id = "staging_lease:{rid}#{epoch}"
    （face_scheduler.acquire_staging_lease）。
  * staging_lease_release（_release_staging_lease，仅对真删落行）：
    decision={reason}（审计字段，不载运字节）。
  * joint_sample_contended（C6 纯度门拒收披露，request_id 可为 ""）
    / readplan_reconcile / readplan_settle（C8 读计划审计）/
    merge_done（C14）/ link_telemetry_coverage / joint_decision_metrics
    （run 级收尾，request_id=""、tick 可回 0）：纯审计行，登记跳过。

重放语义钉测：acquire 时 exec 实例加逐 rank 字节合计（执行期 HBM 占用
入账）、release 时减；acquire/release 配平、run 末悬挂 fail-closed；
同 lease_id 同条款幂等重取零 delta（manager 幂等镜像）；失配双持/无册
release/身份破损/字段破损/tick 回退各 fail-closed；审计行零占用差
（同账本混插前后 instances/actions 逐字节相等）；未知 kind 维持
fail-closed（本修复是纳新入词表，不是放开校验）。

运行：python3 -m pytest slo_tools/tests/test_hbm_watermark_staging_lease.py
      -q -p no:cacheprovider
"""

from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
SLO_TOOLS_DIR = TESTS_DIR.parent
sys.path.insert(0, str(SLO_TOOLS_DIR))
sys.path.insert(0, str(TESTS_DIR))

from test_hbm_watermark import (  # noqa: E402
    make_run_dir, run_tool, token_row, write_fixture)

COEF = 100  # B/token（layers=1, hidden=50, bytes_per_elem=1）
S0, S1 = "session_0", "session_1"
R0, R1 = "s0_r0", "s1_r0"


# ---------------------------------------------------------------------------
# 记录构造（joint stay 生命周期 + lease 行 + 新 kind 审计行）
# ---------------------------------------------------------------------------

class _Seq:
    def __init__(self) -> None:
        self.value = 0

    def nxt(self) -> int:
        self.value += 1
        return self.value


def prefill(request_id: str, tick: int, instance: int, seq: _Seq) -> dict:
    return {"kind": "prefill", "request_id": request_id, "priority": 0,
            "seq": seq.nxt(), "tick": tick,
            "decision": {"prefill_instance_index": instance,
                         "joint_action": "stay",
                         "history_tokens_discarded": 0,
                         "history_evictions": [],
                         "history_transfers": [],
                         "history_transfer_bytes": 0,
                         "history_source_instance_index": None}}


def decode(request_id: str, tick: int, instance: int, seq: _Seq) -> dict:
    return {"kind": "decode", "request_id": request_id, "priority": 0,
            "seq": seq.nxt(), "tick": tick,
            "decision": {"decode_instance_index": instance,
                         "joint_action": "stay",
                         "decode_evictions": []}}


def completion(request_id: str, tick: int, seq: _Seq) -> dict:
    return {"kind": "completion", "request_id": request_id, "priority": 0,
            "seq": seq.nxt(), "tick": tick,
            "decision": {"joint_action": "stay",
                         "merge_direction": "stay",
                         "joint_working_copy": False,
                         "merge_transfers": [],
                         "home_flipped_to": None,
                         "origin_home_instance": 0,
                         "kv_location_after_completion": "local_hbm",
                         "completion_evictions": []}}


def lease_acquire(request_id: str, tick: int, epoch: int, instance: int,
                  per_rank: int, tile: int, seq: _Seq, *,
                  ranks: int = 6, prefill_phase: bool = False) -> dict:
    decision = {
        "lease_id": f"staging_lease:{request_id}#{epoch}",
        "boundary_epoch": epoch,
        "exec_instance_index": instance,
        "staging_shard_bytes": [per_rank] * ranks,
        "staging_tile_tokens": tile,
    }
    if prefill_phase:
        decision["phase"] = "prefill_read"
        decision["scans_per_window"] = 8
    return {"kind": "staging_lease_acquire", "request_id": request_id,
            "priority": 0, "seq": seq.nxt(), "tick": tick,
            "decision": decision}


def lease_release(request_id: str, tick: int, reason: str,
                  seq: _Seq) -> dict:
    return {"kind": "staging_lease_release", "request_id": request_id,
            "priority": 0, "seq": seq.nxt(), "tick": tick,
            "decision": {"reason": reason}}


def sample_contended(tick: int, seq: _Seq, *, request_id: str) -> dict:
    """C6 纯度门拒收披露（发射面字段；own_request_ids 空时
    request_id 落 ""——钉测两形态各一）。"""
    return {"kind": "joint_sample_contended", "request_id": request_id,
            "priority": 0, "seq": seq.nxt(), "tick": tick,
            "decision": {"instance_index": 0, "factor": "decode_step_ns",
                         "window_start_ns": tick - 5, "window_end_ns": tick,
                         "own_request_ids": (
                             [request_id] if request_id else []),
                         "foreign_hbm_flows": {"s9_r0#decode#0": [0, 5]},
                         "port_occupancy": {"0": 2, "5": 1}}}


def readplan_reconcile(tick: int, seq: _Seq) -> dict:
    return {"kind": "readplan_reconcile", "request_id": R0, "priority": 0,
            "seq": seq.nxt(), "tick": tick,
            "decision": {"est_steps": 32, "actual_steps": 30,
                         "est_total_bytes": 9000, "actual_total_bytes": 8800,
                         "est_flow_units": 12, "truth_flow_units": 11,
                         "delta_bytes": -200,
                         "est_uncertainty": "causal_horizon_estimate"}}


def readplan_settle(tick: int, seq: _Seq) -> dict:
    return {"kind": "readplan_settle", "request_id": R0, "priority": 0,
            "seq": seq.nxt(), "tick": tick,
            "decision": {"residual_units_released": 1,
                         "truth_flow_units": 11, "consumed_units": 10,
                         "variance_units": -1,
                         "note": "train fragmentation variance"}}


def merge_done(tick: int, seq: _Seq) -> dict:
    return {"kind": "merge_done", "request_id": R0, "priority": 0,
            "seq": seq.nxt(), "tick": tick,
            "decision": {"merge_done_ns": tick, "session_id": S0,
                         "next_arrival_world_ns": tick + 50,
                         "next_turn_arrived_before_merge_done": False,
                         "next_turn_request_id": None}}


def telemetry_row(kind: str, seq: _Seq) -> dict:
    """run 级收尾披露行（request_id=""、tick=0——生产形态：skip 位于
    request_id/tick 门之前）。"""
    if kind == "link_telemetry_coverage":
        decision = {"collective_coverage": True,
                    "telemetry_degradation": 0, "telemetry_epoch_count": 3,
                    "telemetry_key_absent_seen": False,
                    "telemetry_rate_entries": 128,
                    "telemetry_sample_count": 4096,
                    "telemetry_window_broken": False,
                    "telemetry_zero_rate_samples_dropped": 0}
    else:
        decision = {"quota_mode": "aimd",
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
                    "aimd_action_counts": {}}
    return {"kind": kind, "request_id": "", "priority": 0,
            "seq": seq.nxt(), "tick": 0, "decision": decision}


# ---------------------------------------------------------------------------
# 场景
# ---------------------------------------------------------------------------

def two_phase_records() -> list[dict]:
    """单请求两阶段 lease（prefill_read 相 → decode 相）骑跨 stay KV
    生命周期（I0；coef=100，f(10)=1000、f(12)=1200）：
      t100 prefill +1000 → 1000；t105 lease#0 +1800 → 2800；t108 释放
      −1800 → 1000；t110 decode +200 → 1200；t120 lease#1 +600 → 1800；
      t150 释放 −600 → 1200；t190 completion stay → 残差 1200、峰 2800。
    """
    seq = _Seq()
    return [
        prefill(R0, 100, 0, seq),
        lease_acquire(R0, 105, 0, 0, 300, 128, seq, prefill_phase=True),
        lease_release(R0, 108, "prefill_read_stream_consumed", seq),
        decode(R0, 110, 0, seq),
        lease_acquire(R0, 120, 1, 0, 100, 256, seq),
        lease_release(R0, 150, "credit_stream_fully_consumed", seq),
        completion(R0, 190, seq),
    ]


def coexist_records() -> list[dict]:
    """两请求 prefill 相（r1）与 decode 相（r0）lease 在 I0 共存：
      t100 r0 prefill +1000；t105 r0 lease#1 +900 → 1900；t110 r0 decode
      +200 → 2100；t115 r0 释放 −900 → 1200；t200 r1 prefill +1000 →
      2200；t205 r1 lease#0(prefill_read) +1500 → 3700；t210 r1 decode
      +200 → 3900；t215 r1 释放 −1500 → 2400；t300/t310 完成结算 →
      残差 2400、峰 3900。
    """
    seq = _Seq()
    return [
        prefill(R0, 100, 0, seq),
        lease_acquire(R0, 105, 1, 0, 150, 256, seq),
        decode(R0, 110, 0, seq),
        lease_release(R0, 115, "credit_stream_fully_consumed", seq),
        prefill(R1, 200, 0, seq),
        lease_acquire(R1, 205, 0, 0, 250, 128, seq, prefill_phase=True),
        decode(R1, 210, 0, seq),
        lease_release(R1, 215, "prefill_read_stream_consumed", seq),
        completion(R0, 300, seq),
        completion(R1, 310, seq),
    ]


def run_scenario(tag: str, records: list[dict], *, mutate=None,
                 tokens=None):
    if mutate is not None:
        records = mutate(copy.deepcopy(records))
    if tokens is None:
        tokens = [token_row(R0, S0, 0, 10, 12)]
    run_dir = make_run_dir(tag)
    write_fixture(run_dir, repo_variant="astra-sim-joint",
                  records=records, token_requests=tokens,
                  layers=1, hidden_size=50, bytes_per_elem=1,
                  npu_bytes=1 << 40)
    proc = run_tool(run_dir)
    summary_path = run_dir / "summary.json"
    summary = json.loads(summary_path.read_text()) \
        if summary_path.exists() else {}
    return proc, summary


class StagingLeaseReplayTests(unittest.TestCase):
    """W3 钉测：lease 两阶段生命周期重放 + fail-closed 面 + 词表对账。"""

    def test_two_phase_lifecycle_replay_closes(self):
        """主线①：prefill_read 相与 decode 相各一对 acquire/release；
        占用增减逐字节入账（峰 2800 含 lease 1800）、末态归零（残差 =
        纯 KV 1200）、动作计数精确。"""
        proc, summary = run_scenario("w3sl1", two_phase_records())
        self.assertEqual(proc.returncode, 0, proc.stderr)
        inst = summary["instances"]["0"]
        self.assertEqual(inst["residual_occupancy_bytes"], 12 * COEF)
        self.assertEqual(inst["peak_occupancy_bytes"], 28 * COEF)
        actions = summary["actions"]
        self.assertEqual(actions["staging_lease_acquire"], 2)
        self.assertEqual(actions["staging_lease_acquire_bytes"], 2400)
        self.assertEqual(actions["staging_lease_release"], 2)
        self.assertEqual(actions["staging_lease_release_bytes"], 2400)
        self.assertEqual(actions.get("staging_lease_acquire_idempotent", 0),
                         0)

    def test_prefill_and_decode_sources_coexist(self):
        """主线②：两请求的 prefill 相 lease 与 decode 相 lease 同实例
        并存（r0 decode 相 +900 与 r1 prefill 相 +1500 前后交叠），
        峰 3900 / 残差 2400 手算断言。"""
        proc, summary = run_scenario(
            "w3sl2", coexist_records(),
            tokens=[token_row(R0, S0, 0, 10, 12),
                    token_row(R1, S1, 0, 10, 12)])
        self.assertEqual(proc.returncode, 0, proc.stderr)
        inst = summary["instances"]["0"]
        self.assertEqual(inst["residual_occupancy_bytes"], 24 * COEF)
        self.assertEqual(inst["peak_occupancy_bytes"], 39 * COEF)
        actions = summary["actions"]
        self.assertEqual(actions["staging_lease_acquire"], 2)
        self.assertEqual(actions["staging_lease_release"], 2)

    def test_dangling_lease_fails_closed(self):
        """末条 release 删除 → run 末悬挂 lease fail-closed（exit 2）。"""

        def mutate(records):
            return [row for row in records
                    if not (row["kind"] == "staging_lease_release"
                            and row["tick"] == 150)]

        proc, _ = run_scenario("w3sl3", two_phase_records(), mutate=mutate)
        self.assertEqual(proc.returncode, 2, proc.stderr)
        self.assertIn("悬挂 staging lease", proc.stderr)
        self.assertIn("staging_lease:s0_r0#1", proc.stderr)

    def test_release_without_acquire_fails_closed(self):
        """无在册 lease 的 release 行（发射面对幂等空放不落行）→
        fail-closed。"""

        def mutate(records):
            records.append(lease_release(R1, 400, "request_completion_"
                                         "settlement", _Seq()))
            return records

        proc, _ = run_scenario("w3sl4", two_phase_records(), mutate=mutate)
        self.assertEqual(proc.returncode, 2, proc.stderr)
        self.assertIn("无在册 lease", proc.stderr)

    def test_conflicting_double_acquire_fails_closed(self):
        """在册 lease 未释放又取不同 epoch（manager 双持 RuntimeError 的
        镜像）→ fail-closed。"""

        def mutate(records):
            records.insert(4, lease_acquire(R0, 118, 2, 0, 100, 256, _Seq()))
            return records

        proc, _ = run_scenario("w3sl5", two_phase_records(), mutate=mutate)
        self.assertEqual(proc.returncode, 2, proc.stderr)
        self.assertIn("双持/条款冲突", proc.stderr)

    def test_idempotent_reacquire_zero_delta(self):
        """同 lease_id 同条款二次 acquire（boundary 在途 ack 重放的发射
        面）→ manager 幂等镜像：零 delta，占用只计一次。"""

        def mutate(records):
            duplicate = copy.deepcopy(records[1])
            duplicate["seq"] = 99
            records.insert(2, duplicate)
            return records

        proc, summary = run_scenario("w3sl6", two_phase_records(),
                                     mutate=mutate)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        inst = summary["instances"]["0"]
        self.assertEqual(inst["residual_occupancy_bytes"], 12 * COEF)
        self.assertEqual(inst["peak_occupancy_bytes"], 28 * COEF)
        # acquire 计数只计入账事件（2：lease#0/lease#1 各一）；幂等重取
        # 单独分桶（1）——零 delta 不重复计字节。
        self.assertEqual(summary["actions"]["staging_lease_acquire"], 2)
        self.assertEqual(
            summary["actions"]["staging_lease_acquire_idempotent"], 1)
        self.assertEqual(
            summary["actions"]["staging_lease_acquire_bytes"], 2400)
        self.assertEqual(summary["actions"]["staging_lease_release"], 2)

    def test_lease_id_tamper_fails_closed(self):
        """lease_id 与 (request_id, epoch) 复合身份失配 → fail-closed。"""

        def mutate(records):
            records[1]["decision"]["lease_id"] = "staging_lease:s0_r0#9"
            return records

        proc, _ = run_scenario("w3sl7", two_phase_records(), mutate=mutate)
        self.assertEqual(proc.returncode, 2, proc.stderr)
        self.assertIn("复合身份", proc.stderr)

    def test_nonpositive_shard_bytes_fails_closed(self):
        """逐 rank 字节非正（manager 侧同判 ValueError）→ fail-closed。"""

        def mutate(records):
            records[1]["decision"]["staging_shard_bytes"][3] = 0
            return records

        proc, _ = run_scenario("w3sl8", two_phase_records(), mutate=mutate)
        self.assertEqual(proc.returncode, 2, proc.stderr)
        self.assertIn("逐 rank 字节", proc.stderr)

    def test_lease_row_tick_regression_fails_closed(self):
        """lease 行 tick 回退（次序纪律不因新 kind 分流而绕过）→
        fail-closed。"""

        def mutate(records):
            records.insert(3, lease_acquire(R1, 90, 0, 0, 100, 64, _Seq()))
            return records

        proc, _ = run_scenario("w3sl9", two_phase_records(), mutate=mutate)
        self.assertEqual(proc.returncode, 2, proc.stderr)
        self.assertIn("tick 回退", proc.stderr)
        self.assertIn("staging_lease_acquire", proc.stderr)


class NewKindVocabularyTests(unittest.TestCase):
    """W3 词表对账：新 kind 混合序列全量闭合 + 审计行零占用钉测。"""

    @staticmethod
    def _with_audit_rows(records: list[dict]) -> list[dict]:
        """同账本混插全部新 kind 审计行（C6/C8/C14/run 级收尾），tick
        保持体单调、收尾两行 tick=0（生产形态）。"""
        seq = _Seq()
        extra = [
            sample_contended(113, seq, request_id=R0),
            sample_contended(116, seq, request_id=""),
            readplan_reconcile(118, seq),
            readplan_settle(305, seq),
            merge_done(301, seq),
            telemetry_row("link_telemetry_coverage", seq),
            telemetry_row("joint_decision_metrics", seq),
        ]
        return records + extra

    def test_mixed_sequence_full_closure(self):
        """新 kind 全量混插（lease 两对 + 六类审计行）后重放零异常闭合，
        残差/峰与纯 lease 场景一致——exit=2 → 0 的词表修复主证。"""
        proc, summary = run_scenario(
            "w3mix", self._with_audit_rows(two_phase_records()))
        self.assertEqual(proc.returncode, 0, proc.stderr)
        inst = summary["instances"]["0"]
        self.assertEqual(inst["residual_occupancy_bytes"], 12 * COEF)
        self.assertEqual(inst["peak_occupancy_bytes"], 28 * COEF)

    def test_audit_rows_prove_zero_occupancy_delta(self):
        """钉测：readplan_reconcile/settle、joint_sample_contended（含
        request_id="" 形态）、merge_done、link_telemetry_coverage、
        joint_decision_metrics 全部不产生占用——同账本加插前后
        instances/actions 逐字节相等（纯计划审计/采样纯度/run 级指标行
        的零重放语义差）。"""
        base_proc, base_summary = run_scenario(
            "w3aud0", two_phase_records())
        mixed_proc, mixed_summary = run_scenario(
            "w3aud1", self._with_audit_rows(two_phase_records()))
        self.assertEqual(base_proc.returncode, 0, base_proc.stderr)
        self.assertEqual(mixed_proc.returncode, 0, mixed_proc.stderr)
        self.assertEqual(base_summary["instances"],
                         mixed_summary["instances"])
        self.assertEqual(base_summary["actions"], mixed_summary["actions"])
        self.assertEqual(base_summary["anomalies"],
                         mixed_summary["anomalies"])

    def test_unknown_kind_still_fails_closed(self):
        """词表外 kind 维持 fail-closed（本修复 = 纳新入词表，不是放开
        校验）：构造 kind="mystery_probe_row" 行必须 exit 2。"""

        def mutate(records):
            records.append({"kind": "mystery_probe_row", "request_id": R0,
                            "priority": 0, "seq": 500, "tick": 400,
                            "decision": {"payload": True}})
            return records

        proc, _ = run_scenario("w3unk", two_phase_records(), mutate=mutate)
        self.assertEqual(proc.returncode, 2, proc.stderr)
        self.assertIn("未知 kind='mystery_probe_row'", proc.stderr)

    def test_lease_rows_on_non_joint_repo_fail_closed(self):
        """lease 词表仅挂 astra-sim-joint 映射：非 joint 仓出现同 kind
        仍按未知 kind 拒收（跨仓零影响钉测）。"""
        records = two_phase_records()
        tokens = [token_row(R0, S0, 0, 10, 12), token_row(R1, S1, 0, 10, 12)]
        run_dir = make_run_dir("w3nonj")
        write_fixture(run_dir, repo_variant="astra-sim-face",
                      records=records, token_requests=tokens,
                      layers=1, hidden_size=50, bytes_per_elem=1,
                      npu_bytes=1 << 40)
        proc = run_tool(run_dir)
        self.assertEqual(proc.returncode, 2, proc.stderr)
        self.assertIn("未知 kind='staging_lease_acquire'", proc.stderr)


class RemoteBaseThreePositionTests(unittest.TestCase):
    """W3 补充钉测：REMOTE 基三位置 forward 胜者裁决（combo_none
    session_1_request_0 实测形态——新会话 remote-read 无 home 驻留，
    base 不成位置，跟踪面 = {source, exec}，forward ⟹ 胜者 = source 位）。

    手算（coef=100）：s9 新会话 prefill remote-read@I1 → f(10)=1000；
    D 边界无迁移（boundary_remote_no_migration）→ U 留 1、exec@I2 仅 decode
    增长 f(12)−f(10)=200；completion forward（home_flipped_to=None、腿
    src=2 bytes=200 merge_working_copy_to_home）→ 胜者=1，终态
    1000+200=1200=f(12)；后继轮 stay 在 I1 续长到 f(16)=1600。
    """

    def _records(self) -> list[dict]:
        seq = _Seq()
        return [
            prefill("s9_r0", 100, 1, seq),
            {"kind": "decode", "request_id": "s9_r0", "priority": 0,
             "seq": seq.nxt(), "tick": 110,
             "decision": {
                 "decode_instance_index": 2, "joint_action": "remote-read",
                 "joint_decode_action": "remote-read",
                 "is_continuation": False, "decode_evictions": [],
                 "prefill_decode_transfer": {
                     "kind": "local_hit", "phase": "prefill_decode",
                     "reason": "boundary_remote_no_migration",
                     "session_id": "session_9",
                     "source_instance_index": 2,
                     "target_instance_index": 2,
                     "layer_start": 0, "layer_end": 1,
                     "resident_prefix_layers_before": 1,
                     "resident_prefix_layers_after": 1,
                     "total_bytes": 0, "shards": []}}},
            {"kind": "completion", "request_id": "s9_r0", "priority": 0,
             "seq": seq.nxt(), "tick": 120,
             "decision": {
                 "joint_action": "remote-read",
                 "merge_direction": "forward", "joint_working_copy": True,
                 "merge_transfers": [
                     {"kind": "noc_migrate", "phase": "completion",
                      "reason": "merge_working_copy_to_home",
                      "session_id": "session_9",
                      "source_instance_index": 2,
                      "layer_start": 0, "layer_end": 1,
                      "total_bytes": 200, "shards": []}],
                 "merge_transferred_bytes": 200,
                 "home_flipped_to": None, "origin_home_instance": None,
                 "kv_location_after_completion": "local_hbm",
                 "completion_evictions": []}},
            # 后继轮：胜者位（I1）续 stay 两步长到 f(16)=1600。
            prefill("s9_r1", 200, 1, seq),
            decode("s9_r1", 210, 1, seq),
            completion("s9_r1", 220, seq),
        ]

    def test_remote_base_forward_winner_is_source(self):
        proc, summary = run_scenario(
            "w3rb1", self._records(),
            tokens=[token_row("s9_r0", "session_9", 0, 10, 12),
                    token_row("s9_r1", "session_9", 12, 14, 16)])
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(summary["instances"]["1"]
                         ["residual_occupancy_bytes"], 16 * COEF)
        self.assertEqual(summary["instances"]["1"]["peak_occupancy_bytes"],
                         16 * COEF)
        self.assertEqual(summary["instances"]["2"]
                         ["residual_occupancy_bytes"], 0)
        self.assertEqual(summary["instances"]["2"]["peak_occupancy_bytes"],
                         2 * COEF)
        self.assertEqual(summary["actions"]["joint_merge_three_position"], 1)
        self.assertEqual(
            summary["actions"]["joint_merge_three_position_bytes"], 200)

    def test_remote_base_forward_home_flip_to_exec_fails_closed(self):
        """REMOTE 基 forward 篡改 home_flipped_to=执行端 → forward ⟺
        胜者≠执行端 的方向交叉校验命中。"""

        def mutate(records):
            for row in records:
                if row["request_id"] == "s9_r0" \
                        and row["kind"] == "completion":
                    row["decision"]["home_flipped_to"] = 2
            return records

        proc, _ = run_scenario(
            "w3rb2", self._records(), mutate=mutate,
            tokens=[token_row("s9_r0", "session_9", 0, 10, 12),
                    token_row("s9_r1", "session_9", 12, 14, 16)])
        self.assertEqual(proc.returncode, 2, proc.stderr)
        self.assertIn("forward ⟺ 胜者≠执行端", proc.stderr)

    def test_remote_base_forward_leg_bytes_mismatch_fails_closed(self):
        """腿字节 != exec 位跟踪全量（少搬）→ 败者整份搬移对账命中。"""

        def mutate(records):
            for row in records:
                if row["request_id"] == "s9_r0" \
                        and row["kind"] == "completion":
                    row["decision"]["merge_transfers"][0][
                        "total_bytes"] = 100
                    row["decision"]["merge_transferred_bytes"] = 100
            return records

        proc, _ = run_scenario(
            "w3rb3", self._records(), mutate=mutate,
            tokens=[token_row("s9_r0", "session_9", 0, 10, 12),
                    token_row("s9_r1", "session_9", 12, 14, 16)])
        self.assertEqual(proc.returncode, 2, proc.stderr)
        self.assertIn("整份搬移", proc.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
