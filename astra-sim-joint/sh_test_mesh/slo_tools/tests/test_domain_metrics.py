#!/usr/bin/env python3
"""domain_metrics 契约测试（C16/WP6b，2026-09-22；零后端，仅标准库）。

运行：python3 sh_test_mesh/slo_tools/tests/test_domain_metrics.py
  或  python3 -m unittest discover -s sh_test_mesh/slo_tools/tests -v

覆盖面（对应 C16 卡验收）：
* 三口径分列：D_feed（供给集合）/ D_econ（C_alt* 必含 copy、δ=0 主臂）/
  实际选择（四动作计数、recompute elected/forced 分列不得混报）；
* 派生指标：|D|、hop 分位、方向分布（分方向半径方差）、瓶颈资源
  argmax、约束不满足原因、预测/实测差异（σ̂ 三级链）；
* δ 敏感性重定价（A3'）：δ∈{σ̂,2σ̂,4σ̂} 翻转谓词 C_remote+δ≤C_alt*、
  平局裁决 = SH ACTION_ORDER 偏好（δ=0 重放与在线选择一致）；
* 边界对拍：观测 remote/local 分界 vs ρ 锚点（诊断非验收）、V_copy/
  V_remote 静态参考 b_c/b_r 缺席 → NA；
* 双域同图：配额域 WP3 前恒 NA 的兼容与如实标注；
* home 迁移：completion 披露轨迹 + kv_delta_journal sidecar 消费面
  （C14 字段；F3 四层可信度分级：full/partial/empty/decision_only，
  行链断裂 fail-closed）；
* 纪律位：非 joint run（0 决策行）完全静默；manifest B 参数 fail-closed；
  selected_action 枚举/冗余断言位 fail-closed。

fixture：合成 2 实例 C5-schema 决策日志（candidates 2×4 + load_view +
port_snapshot NA + flow_snapshot 空）+ 合成 trace_config/hardware/
plan manifest + 仓内真实 slo_params_manifest.json（domain_* 两参数）。
"""

from __future__ import annotations

import argparse
import csv
import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
SLO_TOOLS_DIR = TESTS_DIR.parent
sys.path.insert(0, str(SLO_TOOLS_DIR))

import domain_metrics as dm  # noqa: E402
from slo_common import SloToolError  # noqa: E402

COEF = 32  # 2·layers(2)·hidden(4)·bytes_per_elem(2)——合成 trace_config


def _seg(compute=0, remote=0, merge=0, target_wait=0, history_prep=0,
         eviction_wait=0):
    return {
        "target_wait_ns": target_wait, "history_prep_ns": history_prep,
        "eviction_wait_ns": eviction_wait, "compute_ns": compute,
        "remote_read_ns": remote, "merge_ns": merge,
        "notes": [], "remote_read_first_credit_ns": 0,
        "remote_read_stream_ns": 0, "contention_divisor": 1, "hops": 0,
    }


def _cand(action, instance, applicable, cost=None, hops=0, reason=None,
          breakdown=None):
    row = {
        "action": action, "instance_index": instance,
        "applicable": applicable, "cost_ns": cost, "hops": hops,
        "inapplicable_reason": reason, "notes": [],
        "breakdown": breakdown if applicable else None,
    }
    return row


def _load_view(instance, remaining):
    return {"instance_index": instance,
            "queued_task_load_ns": 0, "running_task_load_ns": 0,
            "active_decode_task_load_ns": 0,
            "hbm_remaining_bytes_by_tp_rank": [remaining] * 2,
            "reclaimable_bytes_by_tp_rank": [0, 0]}


def _port_snapshot():
    return {"instances": [
        {"instance_index": index, "u_port_active_decode_streams": "NA",
         "u_port_registered_transfer_flows": "NA", "u_port_total": "NA",
         "bulk_slots_used": "NA", "bulk_slots_cap": "NA",
         "parity_gate_headroom": "NA"}
        for index in (0, 1)]}


def _admission(request_id, tick, candidates, selected_action,
               selected_instance, applicable_actions,
               recompute_selection=None, origin_home=None, cost_ns=None):
    return {
        "kind": "joint_admission", "request_id": request_id,
        "tick": tick, "priority": 0, "seq": tick,
        "decision": {
            "joint_mode": "joint", "category_mode": "typed",
            "layer_policy": "adaptive", "remote_enabled": True,
            "contention_coverage": "cold_start",
            "estimated_decode_tokens": 1, "horizon_source": "unit_test",
            "service_factors": {"decode_factor": 1.0, "prefill_factor": 1.0,
                                "transfer_factor": 1.0, "updates": {},
                                "rejected": {}},
            "instance_rule_note": "joint instance x action argmin",
            "selected_action": selected_action, "joint_action": selected_action,
            "joint_instance_index": selected_instance,
            "joint_cost_ns": cost_ns,
            "applicable_actions": applicable_actions,
            "recompute_selection": recompute_selection,
            "origin_home_instance": origin_home,
            "load_view": [_load_view(0, 10_000), _load_view(1, 5_000)],
            "flow_snapshot": {},
            "port_snapshot": _port_snapshot(),
            "candidates": candidates,
        },
    }


def _completion(request_id, tick, merge_direction="stay",
                home_flipped_to=None, transferred=0,
                kv_instance_after=0):
    return {
        "kind": "completion", "request_id": request_id, "tick": tick,
        "priority": 0, "seq": tick,
        "decision": {
            "joint_action": "stay", "joint_working_copy": False,
            "merge_direction": merge_direction,
            "home_flipped_to": home_flipped_to,
            "merge_transferred_bytes": transferred,
            "kv_instance_after_completion": kv_instance_after,
            "origin_home_instance": None,
            "completion_evictions": [], "merge_transfers": [],
            "remote_read_slices": [],
        },
    }


def _decision_rows():
    """四条决策：turn-0 stay / remote 选中 / δ 翻转样本 / recompute forced。"""
    rows = []
    # R1 session_a_request_0（turn-0）：无历史——stay/recompute 双适用平价
    # （stay 按 ACTION_ORDER 胜），copy/remote 不适用；completion 使
    # measured−pred = +100（σ̂ 样本 1）。
    rows.append(_admission(
        "session_a_request_0", 100,
        [_cand("stay", 0, True, 1000, 0, breakdown=_seg(compute=1000)),
         _cand("recompute", 0, True, 1000, 0,
               breakdown=_seg(compute=1000)),
         _cand("copy", 0, False, hops=0, reason="no history to copy"),
         _cand("remote-read", 0, False, hops=0,
               reason="no resident remote history"),
         _cand("stay", 1, True, 1000, 1, breakdown=_seg(compute=1000)),
         _cand("recompute", 1, True, 1000, 1, breakdown=_seg(compute=1000)),
         _cand("copy", 1, False, hops=1, reason="no history to copy"),
         _cand("remote-read", 1, False, hops=1,
               reason="no resident remote history")],
        "stay", 0, ["stay", "recompute"], cost_ns=1000))
    rows.append(_completion("session_a_request_0", 1200, "stay",
                            kv_instance_after=0))
    # R2 session_b_request_1（turn-1，历史在 inst0）：remote@1 = 395 <
    # copy@1 = 400（C_alt* 必含 copy）→ D_econ(δ=0) = {1} 且在线选中
    # remote-read@1；err = +105（σ̂ 样本 2）。
    rows.append(_admission(
        "session_b_request_1", 200,
        [_cand("stay", 0, True, 500, 0, breakdown=_seg(compute=500)),
         _cand("recompute", 0, True, 900, 0, breakdown=_seg(compute=900)),
         _cand("copy", 0, False, hops=0, reason="no history to copy"),
         _cand("remote-read", 0, False, hops=0,
               reason="base resident at target instance"),
         _cand("stay", 1, False, hops=1,
               reason="history not resident at target"),
         _cand("recompute", 1, True, 900, 1, breakdown=_seg(compute=900)),
         _cand("copy", 1, True, 400, 1, breakdown=_seg(compute=400)),
         _cand("remote-read", 1, True, 395, 1,
               breakdown=_seg(compute=90, remote=300, merge=5))],
        "remote-read", 1, ["stay", "recompute", "copy", "remote-read"],
        origin_home=0, cost_ns=395))
    rows.append(_completion("session_b_request_1", 700, "reverse",
                            kv_instance_after=0))
    # R3 session_c_request_0：remote@1 = 850、C_alt* = stay@0 = 1000 →
    # margin 350：δ=0 重放应选 remote（在线记 stay ⇒ replay_mismatch=1）；
    # 敏感档 σ̂=100/200 翻转、400 不翻转；err = −100（σ̂ 样本 3，
    # completion tick 1200 → measured 900）。
    rows.append(_admission(
        "session_c_request_0", 300,
        [_cand("stay", 0, True, 1000, 0, breakdown=_seg(compute=1000)),
         _cand("recompute", 0, True, 1000, 0, breakdown=_seg(compute=1000)),
         _cand("copy", 0, False, hops=0, reason="no history to copy"),
         _cand("remote-read", 0, False, hops=0,
               reason="no resident remote history"),
         _cand("stay", 1, False, hops=1,
               reason="history not resident at target"),
         _cand("recompute", 1, True, 2000, 1, breakdown=_seg(compute=2000)),
         _cand("copy", 1, True, 1200, 1, breakdown=_seg(compute=1200)),
         _cand("remote-read", 1, True, 850, 1,
               breakdown=_seg(compute=800, remote=50))],
        "stay", 0, ["stay", "recompute", "copy", "remote-read"],
        cost_ns=1000))
    rows.append(_completion("session_c_request_0", 1200, "stay",
                            kv_instance_after=0))
    # R4 session_d_request_0：唯一适用动作 = recompute → forced
    # (no_history)；无 completion（measured NA 路径）。
    rows.append(_admission(
        "session_d_request_0", 400,
        [_cand("stay", 0, False, hops=0,
               reason="history not resident at target"),
         _cand("recompute", 0, True, 700, 0, breakdown=_seg(compute=700)),
         _cand("copy", 0, False, hops=0, reason="no history to copy"),
         _cand("remote-read", 0, False, hops=0,
               reason="no resident remote history"),
         _cand("stay", 1, False, hops=1,
               reason="history not resident at target"),
         _cand("recompute", 1, True, 750, 1, breakdown=_seg(compute=750)),
         _cand("copy", 1, False, hops=1, reason="no history to copy"),
         _cand("remote-read", 1, False, hops=1,
               reason="no resident remote history")],
        "recompute", 0, ["recompute"],
        recompute_selection={"tier": "forced", "forced_reason": "no_history"},
        cost_ns=700))
    rows.append(_completion("session_d_request_0", 2000, "forward",
                            home_flipped_to=1, transferred=12345,
                            kv_instance_after=1))
    # N9：有 merge 字节的 completion 必须有 merge_done 披露行（缺行 ⇒
    # 端点配对不可靠 → measured NA + 告警——缺行分支测试另钉）；此处
    # 同刻 2000（合成夹具 merge 尾段零时长），维持 measured=1600 断言。
    rows.append({
        "kind": "merge_done", "request_id": "session_d_request_0",
        "tick": 2000, "priority": 0, "seq": 2001,
        "decision": {"merge_done_ns": 2000}})
    return rows


TRACE_CONFIG_ROWS = (
    "kind,key,value,group_name,pg_name,ranks,description\n"
    "config,layers,2,,,,unit test\n"
    "config,hidden_size,4,,,,unit test\n"
    "config,bytes_per_elem,2,,,,unit test\n"
    "config,local_hbm_capacity_profile,unit-gib,,,,unit test\n"
    "inference_group,,,inst_west,1,\"0,2,4\",column 0 (ranks 0/2/4)\n"
    "inference_group,,,inst_east,2,\"1,3,5\",column 1 (ranks 1/3/5)\n"
)

HARDWARE = {
    "schema-version": 1, "slug": "unit",
    "mesh": {"rows": 3, "columns": 2, "topology-by-network-dimension":
             ["Line", "Line"]},
    "local-hbm": {"bandwidth-gbps": 1640.0, "latency-ns": 100},
    "d2d": {"bandwidth-gbps": 4050.0, "latency-ns": 5},
    "remote-memory": {"memory-type": "PER_NPU_MEMORY_EXPANSION",
                      "bandwidth-gbps": 512.0, "latency-ns": 100},
    "compute": {"peak-perf-tflops": 100.0},
}

PLAN_MANIFEST = {
    "manifest_source": "unit",
    "requests": [
        {"request_id": "session_a_request_0", "history_tokens_before": 0,
         "final_context_tokens": 10},
        {"request_id": "session_b_request_1", "history_tokens_before": 10,
         "final_context_tokens": 20},
        {"request_id": "session_c_request_0", "history_tokens_before": 0,
         "final_context_tokens": 5},
        {"request_id": "session_d_request_0", "history_tokens_before": 0,
         "final_context_tokens": 5},
    ],
}


def build_run_dir(tag: str = "domain_metrics") -> Path:
    root = Path(tempfile.mkdtemp(prefix=f"slo_t_{tag}_"))
    (root / "results").mkdir()
    with (root / "results" / "online_decision_log.jsonl").open(
            "w", encoding="utf-8") as handle:
        for row in _decision_rows():
            handle.write(json.dumps(row) + "\n")
    (root / "trace_config.csv").write_text(
        TRACE_CONFIG_ROWS, encoding="utf-8")
    (root / "hardware.json").write_text(
        json.dumps(HARDWARE), encoding="utf-8")
    (root / "manifest.json").write_text(
        json.dumps(PLAN_MANIFEST), encoding="utf-8")
    return root


def run_tool(run_dir: Path, *, use_explicit_topology=True,
             trace_config_arg=None, hardware_config_arg=None,
             stderr_buffer=None):
    """CLI 全链（绕开 cpp.log init：--repo-variant 显式）。"""
    scan = dm.domain_prepare()
    for record in dm.iter_jsonl(run_dir / dm.DECISION_LOG_RELPATH):
        dm.domain_consume(scan, record)
    args = argparse.Namespace(
        run_dir=run_dir, output="", instances_csv="", json="",
        manifest=None, request_manifest=None,
        trace_config=(str(trace_config_arg or run_dir / "trace_config.csv")
                      if use_explicit_topology else None),
        hardware_config=(str(hardware_config_arg or run_dir / "hardware.json")
                         if use_explicit_topology else None),
        repo_variant="astra-sim-joint", quiet_when_empty=False)
    stderr_context = (contextlib.redirect_stderr(stderr_buffer)
                      if stderr_buffer is not None
                      else contextlib.nullcontext())
    with stderr_context:
        rc = dm.domain_emit(args, "astra-sim-joint", scan)
    with (run_dir / "slo_domain_summary.json").open(
            encoding="utf-8") as handle:
        summary = json.loads(handle.read())
    with (run_dir / "slo_domain_requests.csv").open(
            encoding="utf-8") as handle:
        requests_rows = list(csv.DictReader(handle))
    with (run_dir / "slo_domain_instances.csv").open(
            encoding="utf-8") as handle:
        instances_rows = list(csv.DictReader(handle))
    return rc, summary, requests_rows, instances_rows


class DomainMetricsTests(unittest.TestCase):
    """合成 C5-schema fixture 上的三口径/派生/纪律断言。"""

    def setUp(self):
        self.run_dir = build_run_dir()
        self.rc, self.summary, self.requests, self.instances = run_tool(
            self.run_dir)

    def tearDown(self):
        # 删除纪律：先解析绝对路径并断言位于本测试自建的临时目录内
        # （INCIDENT-C16 教训条款——绝不 rmtree 父目录）。
        resolved = Path(self.run_dir).resolve()
        assert str(resolved).startswith(tempfile.gettempdir())
        assert resolved.name.startswith("slo_t_domain_metrics")
        import shutil
        shutil.rmtree(resolved)

    def test_exit_and_products(self):
        self.assertEqual(self.rc, 0)
        for name in ("slo_domain_requests.csv",
                     "slo_domain_instances.csv",
                     "slo_domain_summary.json"):
            self.assertTrue((self.run_dir / name).is_file(), name)
        self.assertEqual(len(self.requests), 4)
        self.assertEqual(len(self.instances), 8)  # 4 决策 × 2 实例

    def test_three_calibers_separated(self):
        by_id = {row["request_id"]: row for row in self.requests}
        # D_feed：R1/R4 无 remote 适用 → 0；R2/R3 = {1}。
        self.assertEqual(by_id["session_a_request_0"]["d_feed_size"], "0")
        self.assertEqual(by_id["session_b_request_1"]["d_feed_size"], "1")
        self.assertIn("1:1", by_id["session_b_request_1"]["d_feed_members"])
        # D_econ(δ=0)：R2 = {1}（395 ≤ C_alt*=copy@1 400——必含 copy）；
        # R3 = {1}（850 ≤ stay@0 1000）；R1/R4 = 0。
        self.assertEqual(by_id["session_b_request_1"]["d_econ_size_delta0"],
                         "1")
        self.assertEqual(by_id["session_b_request_1"]["c_alt_star_action"],
                         "copy")
        self.assertEqual(by_id["session_c_request_0"]["d_econ_size_delta0"],
                         "1")
        self.assertEqual(by_id["session_a_request_0"]["d_econ_size_delta0"],
                         "0")
        # 实际选择与 measured 分列：R4 completion 在场（home 迁移夹具）
        # → measured = 2000−400 = 1600、err = +900（σ̂ 第 4 样本）。
        self.assertEqual(by_id["session_d_request_0"]["measured_ns"], "1600")
        self.assertEqual(by_id["session_d_request_0"]["pred_ns"], "700")
        self.assertEqual(by_id["session_d_request_0"]
                         ["pred_measured_err_ns"], "900")
        # 聚合窗口在 summary 内注明。
        window = self.summary["calibers"]["actual_selection"][
            "aggregation_window_ns"]
        self.assertEqual(window[0], 100)
        self.assertEqual(window[1], 2000)

    def test_four_action_counts_recompute_split(self):
        selection = self.summary["calibers"]["actual_selection"]
        self.assertEqual(selection["four_action_counts"],
                         {"stay": 2, "recompute": 0, "remote-read": 1,
                          "copy": 0})
        # recompute 仅 elected 入选中数；forced 单列（§19.2 两口径不混报）。
        self.assertEqual(selection["recompute_elected"], 0)
        self.assertEqual(selection["recompute_forced_by_reason"],
                         {"no_history": 1})
        self.assertEqual(selection["recompute_rows_total"], 1)
        self.assertTrue(selection["elected_plus_forced_equals_rows"])

    def test_action_order_tie_break_and_replay(self):
        # R1 stay/recompute 同价 → C_alt* 动作 = stay（SH ACTION_ORDER 偏好，
        # 非字典序）；R1/R2/R4 的 δ=0 重放与在线一致，R3（记 stay、重放
        # remote）恰贡献 1 个 replay_mismatch。
        by_id = {row["request_id"]: row for row in self.requests}
        self.assertEqual(by_id["session_a_request_0"]["c_alt_star_action"],
                         "stay")
        scan = self.summary["delta_epsilon_registry"]["sensitivity_scan"]
        self.assertEqual(scan["flip_repricing"]["replay_mismatch_at_delta0"],
                         1)

    def test_delta_sensitivity_repricing(self):
        tiers = self.summary["delta_epsilon_registry"]["sensitivity_scan"][
            "flip_repricing"]["per_tier"]
        # σ̂ = |{100,105,100,900}| 的 nearest-rank p50（n=4 → index 1）
        # = 100（观测 B 级链）。
        sigma = self.summary["delta_epsilon_registry"]["sensitivity_scan"]
        self.assertEqual(sigma["sigma_hat_ns"], 100)
        self.assertEqual(sigma["sigma_hat_source"],
                         "observational_pred_vs_measured_p50_abs")
        # R2 margin=5（400−395）、R3 margin=150（1000−850）：δ=100 → R2
        # 翻回 copy、R3 翻成 remote（2 翻转）；δ=200/400 → 仅 R2 翻回
        # copy（R3 的 150 < 200 不再翻转）。位移：R2 同实例 0、R3 +1。
        self.assertEqual(tiers["1.0"]["flips_vs_actual"], 2)
        self.assertEqual(tiers["2.0"]["flips_vs_actual"], 1)
        self.assertEqual(tiers["4.0"]["flips_vs_actual"], 1)
        self.assertEqual(tiers["1.0"]["displacement_hop"]["min"], 0)
        self.assertEqual(tiers["1.0"]["displacement_hop"]["max"], 1)
        # A3' 修订落字：撤销 δ=0 空转对拍臂的注释在场。
        self.assertIn("空转对拍臂已撤销", sigma["note"])

    def test_hop_quantiles_and_direction(self):
        feed = self.summary["calibers"]["d_feed"]
        self.assertEqual(feed["size"]["max"], 1)
        self.assertEqual(feed["hop_quantiles"]["p50"], 1)
        # 锚点：inst0=ranks[0]=0→(0,0)、inst1=ranks[0]=3→(0,1) → 方向 E。
        self.assertEqual(feed["direction_distribution"],
                         {"E": {"count": 2, "hop_variance": 0.0}})
        inst = [row for row in self.instances
                if row["request_id"] == "session_b_request_1"
                and row["instance_index"] == "1"][0]
        self.assertEqual(inst["anchor_row"], "0")
        self.assertEqual(inst["anchor_col"], "1")
        self.assertEqual(inst["direction"], "E")

    def test_bottleneck_and_reasons(self):
        bottleneck = self.summary["derived"]["bottleneck_resource"]
        # R2 remote@1 六段 argmax = remote_read_ns(300)；R3 = compute_ns(800)。
        self.assertEqual(bottleneck["d_feed_remote_members"],
                         {"remote_read_ns": 1, "compute_ns": 1})
        reasons = self.summary["derived"]["constraint_unsatisfied_reasons"]
        # R1×2 + R3×1 + R4×2 = 5；R2 的 inst0 是 base-resident 另计。
        self.assertEqual(reasons["remote-read"]
                         ["no resident remote history"], 5)
        self.assertEqual(reasons["remote-read"]
                         ["base resident at target instance"], 1)
        # R2/R3 各 1 + R4×2 = 4。
        self.assertEqual(reasons["stay"]["history not resident at target"],
                         4)

    def test_quota_domain_derived_column(self):
        """K7（P2-11，2026-09-23）：quota_admissible 不再恒 NA——从候选
        inapplicable_reason 派生（quota-off run ⇒ 结构可行动作存在性；
        port_* 配额字段仍 NA——WP3 前 C5 冻结占位）。M2（2026-09-23
        验收审计）：摘要 quota_domain_status 从数据派生（quota-off run
        无 quota_ 理由 ⇒ structural_only——与实例 CSV 0/1 一致，修前
        固定 NA 自相矛盾）。"""
        chart = self.summary["derived"]["dual_domain_chart"]
        self.assertEqual(chart["quota_domain_status"], "structural_only")
        inst = self.instances[0]
        # quota-off 夹具：无 quota_ 理由 ⇒ 本例（有适用动作）= 1
        #（CSV 行字符串形态）。
        self.assertEqual(inst["quota_admissible"], "1")
        # M2：remote 特异单列（quota-off 下 = remote 结构适用性）。
        self.assertEqual(inst["quota_admissible_remote"], "0")
        self.assertEqual(inst["port_u_port_total"], "NA")

    def test_quota_admissible_zero_branch_at_csv_level(self):
        """L8（2026-09-23 复核审计补钉）：全候选结构性不可行（非 quota_
        前缀理由）的实例行 ⇒ quota_admissible = "0"——派生三态的 0 分支
        CSV 级钉（K7-⑥ 单测只钉了 1/NA 与 flag 函数，0 未走到 CSV）。"""
        run_dir = build_run_dir(tag="domain_metrics_qa0")
        log = run_dir / "results" / "online_decision_log.jsonl"
        with log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(_admission(
                "session_e_request_0", 400,
                [_cand("stay", 0, True, 900, 0,
                       breakdown=_seg(compute=900)),
                 _cand("stay", 1, False, hops=1,
                       reason="history not resident at target"),
                 _cand("copy", 1, False, hops=1,
                       reason="no history to copy"),
                 _cand("remote-read", 1, False, hops=1,
                       reason="no resident remote history"),
                 _cand("recompute", 1, False, hops=1,
                       reason="instance unavailable for recompute")],
                "stay", 0, ["stay"], cost_ns=900)) + "\n")
        rc, _summary, _requests, instances = run_tool(run_dir)
        self.assertEqual(rc, 0)
        by_inst = {(row["request_id"], row["instance_index"]): row
                   for row in instances}
        zero_row = by_inst[("session_e_request_0", "1")]
        # 实例 1 全候选结构性不可行（无 quota_ 理由）⇒ 0；实例 0 有
        # 适用 stay ⇒ 1。
        self.assertEqual(zero_row["quota_admissible"], "0")
        self.assertEqual(
            by_inst[("session_e_request_0", "0")]["quota_admissible"], "1")
        import shutil
        resolved = Path(run_dir).resolve()
        assert str(resolved).startswith(tempfile.gettempdir())
        shutil.rmtree(resolved)

    def test_quota_deferred_domain_rebuild_and_remote_column(self):
        """M2（2026-09-23 验收审计②）：配额拒候选（quota_ 前缀 + 保留
        的配额前成本）纳入 D_feed/D_econ——quota-on run 的经济域可从
        日志重建；quota_admissible_remote 单列（stay 可用即 1 的四动作
        any 列对 remote 被配额裁不敏感）；摘要 status 派生非 NA。"""
        run_dir = build_run_dir(tag="domain_metrics_qon")
        log = run_dir / "results" / "online_decision_log.jsonl"
        with log.open("a", encoding="utf-8") as handle:
            # R5：remote-read@1 被配额拒（quota_link 理由、保留成本
            # 350 < C_alt*=copy@1 400）——修前 D_econ 面临空（成本被
            # SH 清 None 且工具双条件滤除），重建后 D_econ(δ=0)={1}。
            handle.write(json.dumps(_admission(
                "session_f_request_0", 500,
                [_cand("stay", 0, True, 500, 0,
                       breakdown=_seg(compute=500)),
                 _cand("recompute", 0, True, 900, 0,
                       breakdown=_seg(compute=900)),
                 _cand("stay", 1, False, hops=1,
                       reason="history not resident at target"),
                 _cand("recompute", 1, True, 900, 1,
                       breakdown=_seg(compute=900)),
                 _cand("copy", 1, True, 400, 1,
                       breakdown=_seg(compute=400)),
                 _cand("remote-read", 1, False, 350, hops=1,
                       reason="quota_link: link=(0,1) remaining=0")],
                "copy", 1, ["stay", "recompute", "copy"], cost_ns=400)) + "\n")
        rc, summary, requests, instances = run_tool(run_dir)
        self.assertEqual(rc, 0)
        chart = summary["derived"]["dual_domain_chart"]
        # 摘要 status 从数据派生（run 内出现过 quota_ 理由）。
        self.assertEqual(chart["quota_domain_status"],
                         "derived_from_candidates")
        by_id = {row["request_id"]: row for row in requests}
        row = by_id["session_f_request_0"]
        # 配额拒 remote（保留成本 350）进 D_feed/D_econ——经济域重建。
        self.assertEqual(row["d_feed_size"], "1")
        self.assertEqual(row["d_econ_size_delta0"], "1")
        self.assertEqual(row["c_remote_min_ns"], "350")
        self.assertEqual(row["c_alt_star_action"], "copy")
        inst = {(r["request_id"], r["instance_index"]): r
                for r in instances}
        # 实例 1：stay 结构不可行、remote 被配额拒 ⇒ 四动作 any=1
        #（copy 可用）但 remote 特异列=1（配额拒=配额移除后可行）。
        self.assertEqual(inst[("session_f_request_0", "1")]
                         ["quota_admissible"], "1")
        self.assertEqual(inst[("session_f_request_0", "1")]
                         ["quota_admissible_remote"], "1")
        # 对照：实例 0 remote 结构不可行（非配额）⇒ remote 列=0。
        self.assertEqual(inst[("session_f_request_0", "0")]
                         ["quota_admissible_remote"], "0")
        import shutil
        resolved = Path(run_dir).resolve()
        assert str(resolved).startswith(tempfile.gettempdir())
        shutil.rmtree(resolved)

    def test_cross_family_tie_replay_no_false_flip(self):
        """O8①：跨族平局镜像在线 argmin 全键序——stay@0=100 vs
        remote@1=100 在线选 stay@0，重放不再恒翻 remote（replay_
        mismatch/quota_counterfactual_flips 假阳性消除）；反向
        remote 实例号小仍选 remote。"""
        run_dir = build_run_dir(tag="domain_metrics_tie")
        log = run_dir / "results" / "online_decision_log.jsonl"
        with log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(_admission(
                "session_tie_stay_wins", 500,
                [_cand("stay", 0, True, 100, 0,
                       breakdown=_seg(compute=100)),
                 _cand("remote-read", 0, False, hops=0,
                       reason="no resident remote history"),
                 _cand("stay", 1, False, hops=1,
                       reason="history not resident at target"),
                 _cand("remote-read", 1, True, 100, 1,
                       breakdown=_seg(compute=10, remote=90))],
                "stay", 0, ["stay", "remote-read"], cost_ns=100)) + "\n")
            handle.write(json.dumps(_admission(
                "session_tie_remote_wins", 600,
                [_cand("stay", 0, False, hops=0,
                       reason="history not resident at target"),
                 _cand("remote-read", 0, True, 100, 0,
                       breakdown=_seg(compute=10, remote=90)),
                 _cand("stay", 1, True, 100, 1,
                       breakdown=_seg(compute=100)),
                 _cand("remote-read", 1, False, hops=1,
                       reason="no resident remote history")],
                "remote-read", 0, ["stay", "remote-read"],
                cost_ns=100)) + "\n")
        rc, summary, _requests, _instances = run_tool(run_dir)
        self.assertEqual(rc, 0)
        flip = summary["delta_epsilon_registry"]["sensitivity_scan"][
            "flip_repricing"]
        # 基线夹具 replay_mismatch=1/quota_counterfactual_flips=1（R3 真
        # 翻转）——两条平局行各贡献 0（修前 stay_wins 行各计 +1 假翻转）。
        self.assertEqual(flip["replay_mismatch_at_delta0"], 1)
        self.assertEqual(flip["quota_counterfactual_flips"], 1)
        import shutil
        resolved = Path(run_dir).resolve()
        assert str(resolved).startswith(tempfile.gettempdir())
        shutil.rmtree(resolved)

    def test_hops_missing_sentinel_excluded_from_aggregation(self):
        """O8②：hops=None → -1 哨兵不进 build_summary 聚合——hop
        分位 min/n 与方向分布计数同边界计算口径（-1 成员被滤除）。"""
        run_dir = build_run_dir(tag="domain_metrics_hops")
        log = run_dir / "results" / "online_decision_log.jsonl"
        with log.open("a", encoding="utf-8") as handle:
            # H1：remote@1 hops=None（-1 哨兵）、remote@2 hops=2——
            # D_feed={1,2}，聚合只计 hop=2 的成员。
            handle.write(json.dumps(_admission(
                "session_hops_gap_request_0", 700,
                [_cand("stay", 0, True, 50, 0,
                       breakdown=_seg(compute=50)),
                 _cand("remote-read", 0, False, hops=0,
                       reason="no resident remote history"),
                 _cand("stay", 1, False, hops=1,
                       reason="history not resident at target"),
                 _cand("remote-read", 1, True, 40, hops=None,
                       breakdown=_seg(remote=40)),
                 _cand("stay", 2, False, hops=2,
                       reason="history not resident at target"),
                 _cand("remote-read", 2, True, 30, 2,
                       breakdown=_seg(remote=30))],
                "remote-read", 2, ["remote-read", "stay"],
                cost_ns=30)) + "\n")
        rc, summary, _requests, _instances = run_tool(run_dir)
        self.assertEqual(rc, 0)
        feed = summary["calibers"]["d_feed"]
        # 基线 R2/R3 各贡献 hop=1（n=2）+ H1 的 hop=2 ⇒ n=3、min=1；
        # 修前 -1 混入 ⇒ n=4、min=-1。
        self.assertEqual(feed["hop_quantiles"]["n"], 3)
        self.assertEqual(feed["hop_quantiles"]["min"], 1)
        # 方向分布：基线 ("E",1)×2 = count 2——H1 的 hop=2 成员在实例 2
        # （trace_config 无该实例锚点）本就不进方向对；被滤的 -1 成员在
        # 实例 1（有锚点），修前会混入 ("E",-1) 对 ⇒ count=3。
        self.assertEqual(feed["direction_distribution"]["E"]["count"], 2)
        self.assertEqual(feed["direction_distribution"]["E"]["hop_variance"],
                         0.0)
        econ = summary["calibers"]["d_econ"]
        self.assertEqual(econ["hop_quantiles_delta0"]["n"], 3)
        self.assertEqual(econ["hop_quantiles_delta0"]["min"], 1)
        import shutil
        resolved = Path(run_dir).resolve()
        assert str(resolved).startswith(tempfile.gettempdir())
        shutil.rmtree(resolved)

    def test_sigma_none_csv_tier_columns_na(self):
        """O8③：σ̂=None（无 completion 且无 merge_ns>0）时 instances CSV
        in_d_econ_m{1,2,4} 列如实 NA（与 summary 侧 delta_ns=NA 同口径；
        修前只看档数输出 0）。"""
        run_dir = build_run_dir("domain_metrics_sigma_na")
        try:
            for name in ("slo_domain_requests.csv",
                         "slo_domain_instances.csv",
                         "slo_domain_summary.json"):
                path = run_dir / name
                if path.exists():
                    path.unlink()
            rows = [_admission(
                "session_sigma_na_request_0", 10,
                [_cand("stay", 0, True, 100, 0,
                       breakdown=_seg(compute=100)),
                 _cand("remote-read", 1, True, 50, 1,
                       breakdown=_seg(remote=50))],
                "remote-read", 1, ["stay", "remote-read"], cost_ns=50)]
            with (run_dir / "results" / "online_decision_log.jsonl").open(
                    "w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row) + "\n")
            rc, summary, _requests, instances = run_tool(run_dir)
            self.assertEqual(rc, 0)
            scan = summary["delta_epsilon_registry"]["sensitivity_scan"]
            self.assertIsNone(scan["sigma_hat_ns"])
            membership = summary["calibers"]["d_econ"][
                "sensitivity_tiers_membership"]
            for key in ("1.0", "2.0", "4.0"):
                self.assertEqual(membership[key]["delta_ns"], "NA")
            by_inst = {row["instance_index"]: row for row in instances}
            # δ=0 主臂不受 σ̂ 影响（remote 50 ≤ C_alt*=stay 100 ⇒ 成员=1；
            # 实例 0 无 remote 候选 ⇒ 0）。
            self.assertEqual(by_inst["1"]["in_d_econ_delta0"], "1")
            self.assertEqual(by_inst["0"]["in_d_econ_delta0"], "0")
            for row in instances:
                self.assertEqual(row["in_d_econ_m1"], "NA")
                self.assertEqual(row["in_d_econ_m2"], "NA")
                self.assertEqual(row["in_d_econ_m4"], "NA")
        finally:
            resolved = Path(run_dir).resolve()
            assert str(resolved).startswith(tempfile.gettempdir())
            assert resolved.name.startswith("slo_t_domain_metrics_sigma_na")
            import shutil
            shutil.rmtree(resolved)

    def test_merge_done_same_endpoint_pairing(self):
        """M3（2026-09-23 验收审计③）：σ̂ 同终点配对——merge_done 披露
        行在场时 err 用 merge 终点（预测口径），service 延迟单列；
        异终点 −M 偏差不进误差。"""
        run_dir = build_run_dir(tag="domain_metrics_md")
        log = run_dir / "results" / "online_decision_log.jsonl"
        with log.open("a", encoding="utf-8") as handle:
            # R6：cost=600（merge_done 终点预测）、completion（service）
            # = 1000、merge_done = 1600（merge 尾段 600ns 在 service 后）。
            handle.write(json.dumps(_admission(
                "session_g_request_0", 100,
                [_cand("stay", 0, True, 600, 0,
                       breakdown=_seg(compute=600)),
                 _cand("recompute", 0, True, 900, 0,
                       breakdown=_seg(compute=900)),
                 _cand("copy", 0, False, hops=0,
                       reason="no history to copy"),
                 _cand("remote-read", 0, False, hops=0,
                       reason="no resident remote history"),
                 _cand("stay", 1, True, 900, 1,
                       breakdown=_seg(compute=900)),
                 _cand("recompute", 1, True, 950, 1,
                       breakdown=_seg(compute=950)),
                 _cand("copy", 1, False, hops=1,
                       reason="no history to copy"),
                 _cand("remote-read", 1, False, hops=1,
                       reason="no resident remote history")],
                "stay", 0, ["stay", "recompute"], cost_ns=600)) + "\n")
            handle.write(json.dumps(_completion(
                "session_g_request_0", 1100, "stay",
                kv_instance_after=0)) + "\n")
            handle.write(json.dumps({
                "kind": "merge_done", "request_id": "session_g_request_0",
                "tick": 1600, "priority": 0, "seq": 1600,
                "decision": {"merge_done_ns": 1600}}) + "\n")
        rc, summary, requests, instances = run_tool(run_dir)
        self.assertEqual(rc, 0)
        by_id = {row["request_id"]: row for row in requests}
        row = by_id["session_g_request_0"]
        # measured（配对终点）= merge_done 1600−100 = 1500；err =
        # 1500−600 = +900（修前混终点：1000−100−600 = +300 偏 −M）。
        self.assertEqual(row["measured_ns"], "1500")
        self.assertEqual(row["measured_service_ns"], "1000")
        self.assertEqual(row["pred_measured_err_ns"], "900")
        # σ̂ 用同终点配对样本。
        obs = summary["derived"]["prediction_vs_measured"]
        # 夹具共 4 个可配对样本（基础 3 + R6）；R6 的 +900（同终点）
        # 是最大绝对误差——混终点时它只会是 +300（−M 偏差）。
        self.assertGreaterEqual(obs["n"], 4)
        self.assertEqual(obs["signed"]["max"], 900)
        self.assertGreaterEqual(obs["abs"]["p90"], 900)
        self.assertIn("merge_done", obs["measured_anchor"])
        import shutil
        resolved = Path(run_dir).resolve()
        assert str(resolved).startswith(tempfile.gettempdir())
        shutil.rmtree(resolved)

    def test_n9_merge_done_missing_row_reports_na_and_warning(self):
        """N9（2026-09-23 复核·M3 缺行分支）：completion 行显示 merge
        字节而 merge_done 行缺失（丢行/被过滤日志）⇒ measured_ns = NA
        + 告警——不静默退 service 终点（−M 系统偏差）。"""
        run_dir = build_run_dir(tag="domain_metrics_n9")
        log = run_dir / "results" / "online_decision_log.jsonl"
        with log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(_admission(
                "session_h_request_0", 100,
                [_cand("stay", 0, True, 600, 0,
                       breakdown=_seg(compute=600)),
                 _cand("recompute", 0, True, 900, 0,
                       breakdown=_seg(compute=900)),
                 _cand("copy", 0, False, hops=0,
                       reason="no history to copy"),
                 _cand("remote-read", 0, False, hops=0,
                       reason="no resident remote history"),
                 _cand("stay", 1, True, 900, 1,
                       breakdown=_seg(compute=900)),
                 _cand("recompute", 1, True, 950, 1,
                       breakdown=_seg(compute=950)),
                 _cand("copy", 1, False, hops=1,
                       reason="no history to copy"),
                 _cand("remote-read", 1, False, hops=1,
                       reason="no resident remote history")],
                "stay", 0, ["stay", "recompute"], cost_ns=600)) + "\n")
            # completion 显示 merge 字节（transferred=9999）但无
            # merge_done 披露行——修前静默退 service 终点（1100−100
            # =1000 ⇒ err=+400 −M 偏差），修后 NA + 告警。
            handle.write(json.dumps(_completion(
                "session_h_request_0", 1100, "forward",
                home_flipped_to=1, transferred=9999,
                kv_instance_after=1)) + "\n")
        rc, summary, requests, instances = run_tool(run_dir)
        self.assertEqual(rc, 0)
        by_id = {row["request_id"]: row for row in requests}
        row = by_id["session_h_request_0"]
        self.assertEqual(row["measured_ns"], "NA")
        self.assertEqual(row["measured_service_ns"], "1000")
        self.assertEqual(row["pred_measured_err_ns"], "NA")
        self.assertTrue(any(
            "merge_done row missing" in warning
            for warning in summary["warnings"]))
        import shutil
        resolved = Path(run_dir).resolve()
        assert str(resolved).startswith(tempfile.gettempdir())
        shutil.rmtree(resolved)

    def test_n6_pre_quota_cost_column_and_dual_replay(self):
        """N6（2026-09-23 复核·生产6）：配额拒远读保留成本单列
        （remote_cost_pre_quota_ns）+ replay 双口径——被拒远读更便宜
        时实际域重放与在线同选（mismatch=0），差异计入配额反事实
        翻转（quota_counterfactual_flips=1），不再混入 replay_mismatch。"""
        run_dir = build_run_dir(tag="domain_metrics_n6")
        log = run_dir / "results" / "online_decision_log.jsonl"
        with log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(_admission(
                "session_i_request_0", 100,
                [_cand("stay", 0, True, 1000, 0,
                       breakdown=_seg(compute=1000)),
                 _cand("recompute", 0, True, 1200, 0,
                       breakdown=_seg(compute=1200)),
                 _cand("copy", 0, False, hops=0,
                       reason="no history to copy"),
                 _cand("remote-read", 0, False, hops=0,
                       reason="no resident remote history"),
                 _cand("stay", 1, True, 1100, 1,
                       breakdown=_seg(compute=1100)),
                 _cand("recompute", 1, True, 1300, 1,
                       breakdown=_seg(compute=1300)),
                 _cand("copy", 1, False, hops=1,
                       reason="no history to copy"),
                 # 配额拒（quota_ 前缀）但配额前成本保留（M2）且比
                 # stay@0 的 1000 便宜 ⇒ 无配额时选择会翻转。
                 _cand("remote-read", 1, False, 300, 1,
                       reason="quota_link: link=(0,1) remaining=0 of "
                              "Q=2, need=3")],
                "stay", 0, ["stay", "recompute"], cost_ns=1000)) + "\n")
            handle.write(json.dumps(_completion(
                "session_i_request_0", 2100, "stay",
                kv_instance_after=0)) + "\n")
        rc, summary, requests, instances = run_tool(run_dir)
        self.assertEqual(rc, 0)
        by_inst = {(row["request_id"], row["instance_index"]): row
                   for row in instances}
        remote_row = by_inst[("session_i_request_0", "1")]
        # 实际域：配额拒 ⇒ applicable=0、实际成本列 NA。
        self.assertEqual(remote_row["remote_applicable"], "0")
        self.assertEqual(remote_row["remote_cost_ns"], "NA")
        # 配额前结构域：可行（quota_ 拒）+ 配额前成本 300 单列——
        # 实例层配额前成本等值线可画。
        self.assertEqual(remote_row["quota_admissible_remote"], "1")
        self.assertEqual(remote_row["remote_cost_pre_quota_ns"], "300")
        # D_feed 的 feed/内外等值线 margin 按移除 quota gate 后的供给成本，
        # 实际 applicable-only remote_cost_ns 则继续保持 NA。
        self.assertEqual(remote_row["feed_margin_ns"], "-700")
        self.assertAlmostEqual(float(remote_row["feed_margin_ratio"]), -0.7)
        self.assertEqual(remote_row["c_remote_minus_stay_ref_ns"], "-700")
        self.assertEqual(remote_row["c_remote_minus_alt_star_ns"], "-700")
        # 双口径 replay：R5（本夹具行）实际域重放与在线同选 stay（
        # 配额拒远读不进实际可行集——其更便宜不影响 mismatch）；配额
        # 前集重放选 remote@1 ≠ 在线 stay ⇒ 反事实翻转。基础夹具 R3
        # （session_c，δ 翻转演示样本）两口径均贡献 1——期望 = 基础
        # 1 + R5（mismatch 0 / counterfactual 1）。
        flip = summary["delta_epsilon_registry"]["sensitivity_scan"][
            "flip_repricing"]
        self.assertEqual(flip["replay_mismatch_at_delta0"], 1)
        self.assertEqual(flip["quota_counterfactual_flips"], 2)
        import shutil
        resolved = Path(run_dir).resolve()
        assert str(resolved).startswith(tempfile.gettempdir())
        shutil.rmtree(resolved)

    def test_quota_static_96_feed_margins_are_not_na(self):
        """O quota-static 96 个 quota_ remote 行均有配额前 feed/内外 margin。"""
        run_dir = build_run_dir(tag="domain_metrics_quota96")
        log = run_dir / "results" / "online_decision_log.jsonl"
        with log.open("a", encoding="utf-8") as handle:
            for index in range(96):
                request_id = f"session_o{index}_request_0"
                handle.write(json.dumps(_admission(
                    request_id, 3000 + index,
                    [_cand("stay", 0, True, 1000, 0,
                           breakdown=_seg(compute=1000)),
                     _cand("recompute", 0, True, 1200, 0,
                           breakdown=_seg(compute=1200)),
                     _cand("copy", 0, False, hops=0,
                           reason="no history to copy"),
                     _cand("stay", 1, True, 1100, 1,
                           breakdown=_seg(compute=1100)),
                     _cand("recompute", 1, True, 1300, 1,
                           breakdown=_seg(compute=1300)),
                     _cand("copy", 1, False, hops=1,
                           reason="no history to copy"),
                     _cand("remote-read", 1, False, 300, 1,
                           reason="quota_link: synthetic quota rejection")],
                    "stay", 0, ["stay", "recompute"], cost_ns=1000))
                                     + "\n")
        try:
            rc, _summary, _requests, instances = run_tool(run_dir)
            self.assertEqual(rc, 0)
            remote_rows = [
                row for row in instances
                if row["request_id"].startswith("session_o")
                and row["instance_index"] == "1"]
            self.assertEqual(len(remote_rows), 96)
            self.assertTrue(all(
                row["remote_applicable"] == "0"
                and row["remote_cost_ns"] == "NA"
                and row["remote_cost_pre_quota_ns"] == "300"
                and row["feed_margin_ns"] == "-700"
                and row["c_remote_minus_stay_ref_ns"] == "-700"
                and row["c_remote_minus_alt_star_ns"] == "-700"
                for row in remote_rows))
        finally:
            resolved = Path(run_dir).resolve()
            assert str(resolved).startswith(tempfile.gettempdir())
            assert resolved.name.startswith("slo_t_domain_metrics_quota96")
            import shutil
            shutil.rmtree(resolved)

    def test_boundary_cross_check(self):
        cross = self.summary["derived"]["boundary_cross_check"]
        # ρ = 4050/1640；R2/R3 的观测内分界 = hop 1 → deviation = 1−ρ → −1。
        self.assertEqual(cross["rho_anchor"], "2.469512")
        self.assertEqual(cross["observed_remote_local_boundary_hop"]["p50"],
                         1)
        self.assertEqual(cross["boundary_deviation_hop"]["p50"], -1)
        static_ref = cross["remote_copy_static_reference"]
        self.assertEqual(static_ref["b_c_b_r"], "NA")
        by_id = {row["request_id"]: row for row in self.requests}
        # H = 32×10、I = 32×(20−10)、F = min rank remaining @选中实例 1。
        self.assertEqual(by_id["session_b_request_1"]["h_bytes_per_rank"],
                         str(COEF * 10))
        self.assertEqual(by_id["session_b_request_1"]["i_bytes_per_rank"],
                         str(COEF * 10))
        self.assertEqual(by_id["session_b_request_1"]["f_min_bytes_at_selected"],
                         "5000")
        self.assertEqual(by_id["session_b_request_1"]["v_copy_anchor"], "NA")
        self.assertEqual(by_id["session_b_request_1"]["v_remote_anchor"],
                         "NA")
        # 三条对拍纪律注释在场（仅历史扫描项平价 / 诊断非验收 / 无 n* 式）。
        joined = " ".join(cross["disclaimers"])
        self.assertIn("仅历史扫描项平价", joined)
        self.assertIn("诊断不是验收", joined)
        self.assertIn("n* = d'", joined)
        self.assertIn("不设单变量理论线", static_ref["note"])

    def test_run_snapshot_replay_precedes_legacy_local_inputs(self):
        run_dir = build_run_dir("domain_metrics_snapshot")
        try:
            snapshot_trace = TRACE_CONFIG_ROWS.replace(
                'inst_west,1,"0,2,4"', 'inst_west,1,"1,3,5"').replace(
                'inst_east,2,"1,3,5"', 'inst_east,2,"0,2,4"')
            snapshot_hardware = json.loads(json.dumps(HARDWARE))
            snapshot_hardware["d2d"]["bandwidth-gbps"] = 2025.0
            (run_dir / "trace_config.csv.snapshot").write_text(
                snapshot_trace, encoding="utf-8")
            (run_dir / "hardware_config.json.snapshot").write_text(
                json.dumps(snapshot_hardware), encoding="utf-8")

            rc, summary, _requests, instances = run_tool(
                run_dir, use_explicit_topology=False)
            self.assertEqual(rc, 0)
            cross = summary["derived"]["boundary_cross_check"]
            self.assertEqual(cross["rho_anchor"], "1.234756")
            self.assertIn("hardware_config.json.snapshot",
                          cross["rho_source"])
            instance_zero = next(row for row in instances
                                 if row["instance_index"] == "0")
            self.assertEqual(instance_zero["anchor_col"], "1")
        finally:
            resolved = Path(run_dir).resolve()
            assert str(resolved).startswith(tempfile.gettempdir())
            assert resolved.name.startswith("slo_t_domain_metrics_snapshot")
            import shutil
            shutil.rmtree(resolved)

    def test_legacy_run_local_inputs_remain_replayable(self):
        run_dir = build_run_dir("domain_metrics_legacy_local")
        try:
            rc, summary, _requests, instances = run_tool(
                run_dir, use_explicit_topology=False)
            self.assertEqual(rc, 0)
            cross = summary["derived"]["boundary_cross_check"]
            self.assertEqual(cross["rho_anchor"], "2.469512")
            self.assertIn("hardware.json#d2d.bandwidth-gbps=4050.0",
                          cross["rho_source"])
            instance_zero = next(row for row in instances
                                 if row["instance_index"] == "0")
            self.assertEqual(instance_zero["anchor_col"], "0")
        finally:
            resolved = Path(run_dir).resolve()
            assert str(resolved).startswith(tempfile.gettempdir())
            assert resolved.name.startswith(
                "slo_t_domain_metrics_legacy_local")
            import shutil
            shutil.rmtree(resolved)

    def test_explicit_topology_arguments_override_run_snapshots(self):
        run_dir = build_run_dir("domain_metrics_explicit_override")
        try:
            snapshot_hardware = json.loads(json.dumps(HARDWARE))
            snapshot_hardware["d2d"]["bandwidth-gbps"] = 2025.0
            (run_dir / "trace_config.csv.snapshot").write_text(
                TRACE_CONFIG_ROWS.replace(
                    'inst_west,1,"0,2,4"', 'inst_west,1,"1,3,5"').replace(
                    'inst_east,2,"1,3,5"', 'inst_east,2,"0,2,4"'),
                encoding="utf-8")
            (run_dir / "hardware_config.json.snapshot").write_text(
                json.dumps(snapshot_hardware), encoding="utf-8")

            rc, summary, _requests, instances = run_tool(
                run_dir, use_explicit_topology=True)
            self.assertEqual(rc, 0)
            cross = summary["derived"]["boundary_cross_check"]
            self.assertEqual(cross["rho_anchor"], "2.469512")
            self.assertIn("hardware.json#d2d.bandwidth-gbps=4050.0",
                          cross["rho_source"])
            instance_zero = next(row for row in instances
                                 if row["instance_index"] == "0")
            self.assertEqual(instance_zero["anchor_col"], "0")
        finally:
            resolved = Path(run_dir).resolve()
            assert str(resolved).startswith(tempfile.gettempdir())
            assert resolved.name.startswith(
                "slo_t_domain_metrics_explicit_override")
            import shutil
            shutil.rmtree(resolved)

    def test_missing_run_inputs_do_not_fall_back_to_current_checkout(self):
        run_dir = build_run_dir("domain_metrics_missing_inputs")
        stderr = io.StringIO()
        try:
            for name in ("trace_config.csv", "hardware.json",
                         "trace_config.csv.snapshot",
                         "hardware_config.json.snapshot",
                         "hardware.json.snapshot"):
                (run_dir / name).unlink(missing_ok=True)

            rc, summary, _requests, instances = run_tool(
                run_dir, use_explicit_topology=False, stderr_buffer=stderr)
            self.assertEqual(rc, 0)
            cross = summary["derived"]["boundary_cross_check"]
            self.assertEqual(cross["rho_anchor"], "NA")
            self.assertIn("未回退当前 checkout", stderr.getvalue())
            self.assertIn("缺 trace_config 快照", stderr.getvalue())
            self.assertIn("缺 hardware JSON 快照", stderr.getvalue())
            self.assertTrue(all(row["anchor_row"] == "NA"
                                and row["anchor_col"] == "NA"
                                for row in instances))
        finally:
            resolved = Path(run_dir).resolve()
            assert str(resolved).startswith(tempfile.gettempdir())
            assert resolved.name.startswith(
                "slo_t_domain_metrics_missing_inputs")
            import shutil
            shutil.rmtree(resolved)

    def test_home_migration(self):
        block = self.summary["home_migration"]["decision_log_completion"]
        self.assertEqual(block["merge_direction_counts"],
                         {"stay": 2, "reverse": 1, "forward": 1})
        self.assertEqual(block["home_migration_count"], 1)
        self.assertEqual(block["merge_transferred_bytes_total"], 12345)
        # kv_delta sidecar 不在 fixture → 四层可信度最低层
        # decision_log_only（F3 接通后 TODO 态退役）。
        kv = self.summary["home_migration"]["kv_delta_journal"]
        self.assertEqual(kv["status"], dm.KV_DELTA_TIER_DECISION_ONLY)
        self.assertEqual(kv["rows"], 0)

    def test_contour_columns_present(self):
        inst = [row for row in self.instances
                if row["request_id"] == "session_b_request_1"
                and row["instance_index"] == "1"][0]
        # 内等值线差值 = 395−500 = −105；外 = 395−400 = −5；同位置胜。
        self.assertEqual(inst["c_remote_minus_stay_ref_ns"], "-105")
        self.assertEqual(inst["c_remote_minus_alt_star_ns"], "-5")
        self.assertEqual(inst["same_pos_best_alt_ns"], "400")
        self.assertEqual(inst["same_pos_remote_pref"], "1")
        self.assertEqual(inst["in_d_econ_delta0"], "1")
        self.assertEqual(inst["remote_bottleneck_component"],
                         "remote_read_ns")


class DomainMetricsDisciplineTests(unittest.TestCase):
    """纪律位：静默跳过 / fail-closed / schema 断言 / kv_delta sidecar。"""

    def test_silent_noop_without_joint_rows(self):
        run_dir = build_run_dir("domain_silent")
        try:
            (run_dir / "results" / "online_decision_log.jsonl").write_text(
                json.dumps({"kind": "prefill", "request_id": "x",
                            "tick": 1, "decision": {}}) + "\n",
                encoding="utf-8")
            args = argparse.Namespace(
                run_dir=run_dir, output="", instances_csv="", json="",
                manifest=None, request_manifest=None, trace_config="",
                hardware_config="", repo_variant="astra-sim-joint",
                quiet_when_empty=True)
            scan = dm.domain_prepare()
            for record in dm.iter_jsonl(
                    run_dir / dm.DECISION_LOG_RELPATH):
                dm.domain_consume(scan, record)
            self.assertEqual(dm.domain_emit(args, "astra-sim-joint",
                                            scan), 0)
            self.assertFalse(
                (run_dir / "slo_domain_requests.csv").exists())
        finally:
            resolved = Path(run_dir).resolve()
            assert str(resolved).startswith(tempfile.gettempdir())
            assert resolved.name.startswith("slo_t_domain_silent")
            import shutil
            shutil.rmtree(resolved)

    def test_manifest_params_fail_closed(self):
        run_dir = build_run_dir("domain_fc")
        try:
            stub = run_dir / "stub_manifest.json"
            entry = json.loads(
                (SLO_TOOLS_DIR / "slo_params_manifest.json")
                .read_text(encoding="utf-8"))
            del entry["params"]["domain_delta_adm_ns"]
            stub.write_text(json.dumps(entry), encoding="utf-8")
            scan = dm.domain_prepare()
            for record in dm.iter_jsonl(
                    run_dir / dm.DECISION_LOG_RELPATH):
                dm.domain_consume(scan, record)
            args = argparse.Namespace(
                run_dir=run_dir, output="", instances_csv="", json="",
                manifest=str(stub), request_manifest=None,
                trace_config="", hardware_config="",
                repo_variant="astra-sim-joint", quiet_when_empty=False)
            with self.assertRaises(SloToolError) as caught:
                dm.domain_emit(args, "astra-sim-joint", scan)
            self.assertIn("domain_delta_adm_ns", str(caught.exception))
        finally:
            resolved = Path(run_dir).resolve()
            assert str(resolved).startswith(tempfile.gettempdir())
            assert resolved.name.startswith("slo_t_domain_fc")
            import shutil
            shutil.rmtree(resolved)

    def test_selected_action_enum_fail_closed(self):
        scan = dm.DomainScan()
        record = _admission(
            "session_x_request_0", 1,
            [_cand("stay", 0, True, 10, 0, breakdown=_seg())],
            "teleport", 0, ["stay"])
        with self.assertRaises(SloToolError):
            scan.consume(record)

    def test_redundant_action_assertion(self):
        scan = dm.DomainScan()
        record = _admission(
            "session_x_request_0", 1,
            [_cand("stay", 0, True, 10, 0, breakdown=_seg())],
            "stay", 0, ["stay"])
        record["decision"]["joint_action"] = "copy"  # 与 selected_action 矛盾
        with self.assertRaises(SloToolError):
            scan.consume(record)

    def test_kv_delta_sidecar_consumption(self):
        run_dir = build_run_dir("domain_kvdelta")
        try:
            sidecar = {
                "merge_degrade_events": [],
                "deep_gap_events": [],
                "kv_delta_journal": [
                    {"seq": 1, "session_id": "session_b",
                     "trigger_request_id": "session_b_request_1",
                     "working_kind": "REMOTE",
                     "direction": "reverse", "zero_byte_flip": False,
                     "winner_instance": 0, "loser_instance": 1,
                     "home_before": 0, "home_after": 0,
                     "home_migration": False, "transferred_bytes": 111,
                     "home_side_retained_bytes": 50,
                     "exec_side_retained_bytes": 40, "new_tokens": 10,
                     "staging_return_bytes": 0},
                    {"seq": 2, "session_id": "session_d",
                     "trigger_request_id": "session_d_request_0",
                     "working_kind": "PARTIAL",
                     "direction": "forward", "zero_byte_flip": True,
                     "winner_instance": 1, "loser_instance": 0,
                     "home_before": 0, "home_after": 1,
                     "home_migration": True, "transferred_bytes": 222,
                     "home_side_retained_bytes": 0,
                     "exec_side_retained_bytes": 60, "new_tokens": 5,
                     "staging_return_bytes": 0},
                ],
            }
            (run_dir / "bridge").mkdir()
            (run_dir / "bridge" / "joint_kv_ledgers.json").write_text(
                json.dumps(sidecar), encoding="utf-8")
            rc, summary, _, _ = run_tool(run_dir)
            self.assertEqual(rc, 0)
            kv = summary["home_migration"]["kv_delta_journal"]
            # 两行均可联 completion tick（session_b_request_1@700、
            # session_d_request_0@2000 在 fixture）→ 顶层 full_join。
            self.assertEqual(kv["status"], dm.KV_DELTA_TIER_FULL)
            self.assertEqual(kv["rows"], 2)
            self.assertEqual(kv["rows_unjoined"], 0)
            self.assertEqual(kv["direction_counts"],
                             {"reverse": 1, "forward": 1})
            self.assertEqual(kv["zero_byte_flip_count"], 1)
            self.assertEqual(kv["home_migration_count"], 1)
            self.assertEqual(kv["transferred_bytes_total"], 333)
            # 驻留时间 = completion tick 联 join 的近似（session_d 单点 →
            # 无样本；dwell n=0）。
            self.assertEqual(kv["home_dwell_ns"]["n"], 0)
            # 顶层缺口如实披露：水印 certified 层的守恒证书本仓不产出。
            self.assertIn("不冒认 certified", kv["tier_note"])
        finally:
            resolved = Path(run_dir).resolve()
            assert str(resolved).startswith(tempfile.gettempdir())
            assert resolved.name.startswith("slo_t_domain_kvdelta")
            import shutil
            shutil.rmtree(resolved)


class KVDeltaTierTests(unittest.TestCase):
    """F3：kv_delta_journal 四层可信度分级的消费位（domain 读取器）。

    四层 = settlement_full_join > settlement_partial_join >
    settlement_empty > decision_log_only（tier 由 run_dir 内容自动判定，
    status 如实标注；行链断裂 fail-closed）。全部零后端合成 sidecar。
    """

    @staticmethod
    def _write_sidecar(run_dir: Path, payload: dict) -> None:
        (run_dir / "bridge").mkdir(exist_ok=True)
        (run_dir / "bridge" / "joint_kv_ledgers.json").write_text(
            json.dumps(payload), encoding="utf-8")

    @staticmethod
    def _row(seq, request_id, direction="forward", **overrides):
        row = {
            "seq": seq, "session_id": request_id.rsplit("_request_", 1)[0],
            "trigger_request_id": request_id, "working_kind": "REMOTE",
            "direction": direction, "zero_byte_flip": False,
            "winner_instance": 0, "loser_instance": 1,
            "home_before": 0, "home_after": 1, "home_migration": True,
            "transferred_bytes": 100,
            "home_side_retained_bytes": 0,
            "exec_side_retained_bytes": 0, "new_tokens": 1,
            "staging_return_bytes": 0,
        }
        row.update(overrides)
        return row

    def _cleanup(self, run_dir: Path, tag: str):
        resolved = Path(run_dir).resolve()
        assert str(resolved).startswith(tempfile.gettempdir())
        assert resolved.name.startswith(f"slo_t_{tag}")
        import shutil
        shutil.rmtree(resolved)

    def test_partial_join_tier(self):
        # 行在案且链自洽，但一行联不上 completion tick（fixture 无该
        # 请求的 completion 行）→ 降一级 settlement_partial_join；
        # 结算事实（方向/字节）照常全量披露。
        run_dir = build_run_dir("domain_kvtier_p")
        try:
            self._write_sidecar(run_dir, {
                "kv_delta_journal": [
                    self._row(0, "session_b_request_1"),
                    self._row(1, "session_zz_request_0"),
                ]})
            rc, summary, _, _ = run_tool(run_dir)
            self.assertEqual(rc, 0)
            kv = summary["home_migration"]["kv_delta_journal"]
            self.assertEqual(kv["status"], dm.KV_DELTA_TIER_PARTIAL)
            self.assertEqual(kv["rows"], 2)
            self.assertEqual(kv["rows_unjoined"], 1)
            self.assertEqual(kv["transferred_bytes_total"], 200)
        finally:
            self._cleanup(run_dir, "domain_kvtier_p")

    def test_empty_key_tier(self):
        # sidecar 在场但 kv_delta_journal 空列表（零结算 run）→
        # settlement_empty；读不到键的行、如实披露 sidecar 键清单。
        run_dir = build_run_dir("domain_kvtier_e")
        try:
            self._write_sidecar(run_dir, {
                "merge_degrade_events": [], "deep_gap_events": [],
                "copy_handoff_events": [], "kv_delta_journal": []})
            rc, summary, _, _ = run_tool(run_dir)
            self.assertEqual(rc, 0)
            kv = summary["home_migration"]["kv_delta_journal"]
            self.assertEqual(kv["status"], dm.KV_DELTA_TIER_EMPTY)
            self.assertEqual(kv["rows"], 0)
            self.assertEqual(kv["sidecar_keys"], [
                "copy_handoff_events", "deep_gap_events",
                "kv_delta_journal", "merge_degrade_events"])
        finally:
            self._cleanup(run_dir, "domain_kvtier_e")

    def test_missing_key_tier_is_empty_not_decision_only(self):
        # F3 前旧 sidecar schema（三键、无 kv_delta_journal 键）：sidecar
        # 在场 → settlement_empty（区别于文件缺席的 decision_log_only）。
        run_dir = build_run_dir("domain_kvtier_m")
        try:
            self._write_sidecar(run_dir, {
                "merge_degrade_events": [], "deep_gap_events": [],
                "copy_handoff_events": []})
            rc, summary, _, _ = run_tool(run_dir)
            self.assertEqual(rc, 0)
            kv = summary["home_migration"]["kv_delta_journal"]
            self.assertEqual(kv["status"], dm.KV_DELTA_TIER_EMPTY)
            self.assertNotIn("kv_delta_journal", kv["sidecar_keys"])
        finally:
            self._cleanup(run_dir, "domain_kvtier_m")

    def test_export_error_sentinel_tier_is_decision_only(self):
        # A12'：生产端逐键哨兵——kv_delta_journal 导出失败（seq 链
        # fail-closed 等）⇒ decision_log_only 层 + 明示注记；其余三键
        # 已按各自通道落盘消费（旧行为：兜底 except 吞成四键尽失）。
        run_dir = build_run_dir("domain_kvtier_s")
        try:
            self._write_sidecar(run_dir, {
                "merge_degrade_events": [], "deep_gap_events": [],
                "copy_handoff_events": [],
                "kv_delta_journal_export_error":
                    "RuntimeError: kv_delta_journal seq chain broken"})
            rc, summary, _, _ = run_tool(run_dir)
            self.assertEqual(rc, 0)
            kv = summary["home_migration"]["kv_delta_journal"]
            self.assertEqual(kv["status"], dm.KV_DELTA_TIER_DECISION_ONLY)
            self.assertIn("导出失败", kv["note"])
            self.assertIn("seq chain broken", kv["note"])
        finally:
            self._cleanup(run_dir, "domain_kvtier_s")

    def test_non_dict_top_level_is_decision_only_not_empty(self):
        # A12'：顶层非对象 = schema 损坏，非"零结算 run"——按最弱层退
        # 决策时刻披露，证据等级不虚标 settlement_empty。
        run_dir = build_run_dir("domain_kvtier_n")
        try:
            (run_dir / "bridge").mkdir(exist_ok=True)
            (run_dir / "bridge" / "joint_kv_ledgers.json").write_text(
                "[1, 2, 3]", encoding="utf-8")
            rc, summary, _, _ = run_tool(run_dir)
            self.assertEqual(rc, 0)
            kv = summary["home_migration"]["kv_delta_journal"]
            self.assertEqual(kv["status"], dm.KV_DELTA_TIER_DECISION_ONLY)
            self.assertIn("顶层结构非对象", kv["note"])
        finally:
            self._cleanup(run_dir, "domain_kvtier_n")

    def test_non_list_journal_value_is_decision_only_not_empty(self):
        # A14'（H7，2026-09-22 第三轮复审）：键在而值非列表 = 值级
        # schema 损坏——顶层 A12' 原则（损坏不得虚标 settlement_empty
        # 证据等级）贯彻到值级：decision_log_only；键缺席仍走 EMPTY
        # （F3 前旧 schema 兼容契约，另测钉住）。
        run_dir = build_run_dir("domain_kvtier_v")
        try:
            self._write_sidecar(run_dir, {
                "merge_degrade_events": [], "deep_gap_events": [],
                "copy_handoff_events": [],
                "kv_delta_journal": {"bogus": 1}})
            rc, summary, _, _ = run_tool(run_dir)
            self.assertEqual(rc, 0)
            kv = summary["home_migration"]["kv_delta_journal"]
            self.assertEqual(kv["status"], dm.KV_DELTA_TIER_DECISION_ONLY)
            self.assertIn("键值非列表", kv["note"])
        finally:
            self._cleanup(run_dir, "domain_kvtier_v")

    def test_stay_home_after_semantics_no_ping_pong_false_positive(self):
        # A12'：stay 行 home_after = home_before（新 schema）——同
        # session"无迁移 forward + stay"的 homes 集合不虚增；旧 sidecar
        # 的 None 形态由消费端过滤兜底（两种形态均零假阳性）。
        for stay_after in (0, None):
            run_dir = build_run_dir("domain_kvtier_pp")
            try:
                self._write_sidecar(run_dir, {"kv_delta_journal": [
                    self._row(0, "session_a_request_0",
                              home_before=0, home_after=0,
                              home_migration=False, winner_instance=0),
                    self._row(1, "session_a_request_1",
                              direction="stay", winner_instance=None,
                              home_before=0, home_after=stay_after,
                              home_migration=False, transferred_bytes=0),
                ]})
                rc, summary, _, _ = run_tool(run_dir)
                self.assertEqual(rc, 0)
                kv = summary["home_migration"]["kv_delta_journal"]
                self.assertEqual(kv["sessions_with_multiple_homes"], 0)
            finally:
                self._cleanup(run_dir, "domain_kvtier_pp")

    def test_all_none_home_session_counts_but_no_ping_pong(self):
        # A14'（H9，2026-09-22 第三轮复审）：单会话全 None home（旧
        # sidecar 无胜者行且无 home_before）——计入 sessions，homes
        # 集合为空 ⇒ 乒乓代理指标不可见（不崩溃、不假阳性）。
        run_dir = build_run_dir("domain_kvtier_an")
        try:
            self._write_sidecar(run_dir, {"kv_delta_journal": [
                self._row(0, "session_a_request_0",
                          direction="stay", winner_instance=None,
                          home_before=None, home_after=None,
                          home_migration=False, transferred_bytes=0),
                self._row(1, "session_a_request_1",
                          direction="stay", winner_instance=None,
                          home_before=None, home_after=None,
                          home_migration=False, transferred_bytes=0),
            ]})
            rc, summary, _, _ = run_tool(run_dir)
            self.assertEqual(rc, 0)
            kv = summary["home_migration"]["kv_delta_journal"]
            self.assertEqual(kv["sessions"], 1)
            self.assertEqual(kv["sessions_with_multiple_homes"], 0)
        finally:
            self._cleanup(run_dir, "domain_kvtier_an")

    def test_corrupt_sidecar_degrades_to_decision_only(self):
        # 半截 JSON → decision_log_only（诊断注记在场，不 fail 整个
        # domain 步骤——sidecar 是披露通道而非判决输入）。
        run_dir = build_run_dir("domain_kvtier_c")
        try:
            (run_dir / "bridge").mkdir()
            (run_dir / "bridge" / "joint_kv_ledgers.json").write_text(
                '{"kv_delta_journal": [ {"seq": 0,', encoding="utf-8")
            rc, summary, _, _ = run_tool(run_dir)
            self.assertEqual(rc, 0)
            kv = summary["home_migration"]["kv_delta_journal"]
            self.assertEqual(kv["status"], dm.KV_DELTA_TIER_DECISION_ONLY)
            self.assertIn("不可解析", kv["note"])
        finally:
            self._cleanup(run_dir, "domain_kvtier_c")

    def test_seq_chain_break_fails_closed(self):
        # seq 非严格增 = 行链断裂 → fail-closed（账本破损不得静默降级，
        # 水印同款纪律）；不落产物。
        run_dir = build_run_dir("domain_kvtier_b")
        try:
            self._write_sidecar(run_dir, {
                "kv_delta_journal": [
                    self._row(0, "session_b_request_1"),
                    self._row(0, "session_d_request_0"),
                ]})
            with self.assertRaises(SloToolError) as caught:
                run_tool(run_dir)
            self.assertIn("seq", str(caught.exception))
        finally:
            self._cleanup(run_dir, "domain_kvtier_b")


class PureFunctionTests(unittest.TestCase):
    """纯函数：方向罗盘 / σ̂ 冷启动锚点 / 选择谓词。"""

    def test_direction_bucket_compass(self):
        self.assertEqual(dm.direction_bucket((1, 1), (1, 1)), "local")
        self.assertEqual(dm.direction_bucket((1, 1), (0, 1)), "N")
        self.assertEqual(dm.direction_bucket((1, 1), (2, 1)), "S")
        self.assertEqual(dm.direction_bucket((1, 1), (1, 0)), "W")
        self.assertEqual(dm.direction_bucket((1, 1), (1, 2)), "E")
        self.assertEqual(dm.direction_bucket((1, 1), (0, 0)), "NW")
        self.assertEqual(dm.direction_bucket((1, 1), (0, 2)), "NE")
        self.assertEqual(dm.direction_bucket((1, 1), (2, 0)), "SW")
        self.assertEqual(dm.direction_bucket((1, 1), (2, 2)), "SE")

    def test_selection_under_delta_predicate(self):
        # 谓词：C_remote+δ ≤ C_alt* 胜；严格大于则 alt* 胜。等号成立时
        # 按全键序 (cost+δ, instance, ACTION_ORDER 秩) 裁决（O8①）——
        # 修前"等号成立即翻转"恒翻 remote，与在线平局取小实例分歧。
        self.assertEqual(
            dm._selection_under_delta(0, (90, 1), (100, 0, "stay")),
            ("remote-read", 1))
        self.assertEqual(
            dm._selection_under_delta(10, (90, 1), (100, 0, "stay")),
            ("stay", 0))
        self.assertEqual(
            dm._selection_under_delta(11, (90, 1), (100, 0, "stay")),
            ("stay", 0))
        self.assertEqual(
            dm._selection_under_delta(0, None, (100, 0, "stay")),
            ("stay", 0))
        self.assertEqual(
            dm._selection_under_delta(0, (90, 1), None), ("NA", None))

    def test_selection_under_delta_cross_family_tie(self):
        """O8①：跨族平局镜像在线 argmin 全键序 (cost, (instance,
        priority))——δ=0 同价时实例号小者胜，不再恒翻 remote。"""
        # stay@0=100 vs remote@1=100：在线取 stay@0（实例小者优先）。
        self.assertEqual(
            dm._selection_under_delta(0, (100, 1), (100, 0, "stay")),
            ("stay", 0))
        # 反向：remote@0=100 vs stay@1=100 ⇒ 翻 remote@0。
        self.assertEqual(
            dm._selection_under_delta(0, (100, 0), (100, 1, "stay")),
            ("remote-read", 0))
        # 同实例同价：rank 分量兜底（stay 秩 0 < remote 秩 3）。
        self.assertEqual(
            dm._selection_under_delta(0, (100, 0), (100, 0, "stay")),
            ("stay", 0))
        # δ>0 且 remote+δ==C_alt*：同价平局同样按全键序裁决。
        self.assertEqual(
            dm._selection_under_delta(10, (90, 1), (100, 0, "stay")),
            ("stay", 0))
        self.assertEqual(
            dm._selection_under_delta(10, (90, 0), (100, 1, "stay")),
            ("remote-read", 0))

    def test_sigma_hat_cold_start_anchor(self):
        # 无配对样本（completion 缺失）→ C 级：merge_ns>0 的 p50。
        run_dir = build_run_dir("domain_sigma")
        try:
            for name in ("slo_domain_requests.csv",
                         "slo_domain_instances.csv",
                         "slo_domain_summary.json"):
                path = run_dir / name
                if path.exists():
                    path.unlink()
            (run_dir / "results" / "online_decision_log.jsonl").write_text(
                "", encoding="utf-8")
            # 重新填充只含 merge_ns 样本的决策（无 completion）。
            rows = [_admission(
                "session_z_request_0", 10,
                [_cand("stay", 0, True, 50, 0,
                       breakdown=_seg(compute=10, merge=7)),
                 _cand("remote-read", 1, True, 40, 1,
                       breakdown=_seg(compute=1, merge=9))],
                "remote-read", 1, ["stay", "remote-read"], cost_ns=40)]
            with (run_dir / "results" / "online_decision_log.jsonl").open(
                    "w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row) + "\n")
            rc, summary, _, _ = run_tool(run_dir)
            self.assertEqual(rc, 0)
            scan = summary["delta_epsilon_registry"]["sensitivity_scan"]
            self.assertEqual(scan["sigma_hat_ns"], 7)  # |{7,9}| 近邻秩 p50
            self.assertEqual(scan["sigma_hat_source"],
                             "merge_leg_cold_start_anchor")
            self.assertIn("冷启动替身", scan["sigma_hat_identity"])
        finally:
            resolved = Path(run_dir).resolve()
            assert str(resolved).startswith(tempfile.gettempdir())
            assert resolved.name.startswith("slo_t_domain_sigma")
            import shutil
            shutil.rmtree(resolved)


if __name__ == "__main__":
    unittest.main()
