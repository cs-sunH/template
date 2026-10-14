#!/usr/bin/env python3
"""test_acc_predictor_cost_accounting.py -- 验收纪律第 3 类统计接口契约
测试（ACC；joint修改文档「验收纪律（v2.0 §19 三类，全部纳入）」/
v2.0 §19.3 仿真可行性）。

钉面：
  1. 接口契约（joint/predictor_cost_accounting.py）：P/D 两阶段决策
     墙时逐次样本 → P50/P95/P99（最近秩法、确定性整数）、total/mean/
     max、候选面宽度 candidates_max；摘要维护费；后端事件量（图批
     节点/边/watch/未来 alarm）；run 末快照的预测器合计、宿主总占比
     （整数 ppm）、峰值 RSS（ru_maxrss 口径披露）、决策日志大小回填；
  2. fail-closed：未知决策阶段、负值/布尔值、非法日志字节一律拒绝；
  3. 接线冒烟：sh30 真实调度器壳的准入（P）与边界（D）决策记账、
     容量摘要维护费记账、图批事件量记账——run 末快照可直接产出
     §19.3 全部证据字段。

宿主 Python 墙时不自动等于真实片上决策时延（§19.3 末段）——本接口
只如实披露测量，测试不外推部署结论。

运行：cd sh_test_mesh/workload/llama2_7b_inference &&
      python3 joint/test_acc_predictor_cost_accounting.py   （或 pytest）
"""
import os
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_WL = os.path.dirname(_HERE)
for _p in (_HERE, _WL):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from joint.predictor_cost_accounting import (  # noqa: E402
    DECISION_STAGES,
    PredictorCostAccounting,
)


class PercentileMathTest(unittest.TestCase):
    """最近秩分位：确定性、整数、空集安全。"""

    def test_nearest_rank_percentiles_are_deterministic(self):
        accounting = PredictorCostAccounting()
        for sample in (10, 20, 30, 40, 50, 60, 70, 80, 90, 100):
            accounting.note_decision("P", sample)
        snapshot = accounting.snapshot(host_wall_ns=10 ** 9)
        stage = snapshot["decision_stages"]["P"]
        self.assertEqual(stage["count"], 10)
        self.assertEqual(stage["total_ns"], 550)
        self.assertEqual(stage["mean_ns"], 55)
        # 最近秩：P50 = 第 ceil(0.5·10)=5 个 = 50；P95 = 第 10 个 =
        # 100；P99 = 第 10 个 = 100（ceil(9.5)）。
        self.assertEqual(stage["p50_ns"], 50)
        self.assertEqual(stage["p95_ns"], 100)
        self.assertEqual(stage["p99_ns"], 100)
        self.assertEqual(stage["max_ns"], 100)

    def test_single_sample_all_percentiles_converge(self):
        accounting = PredictorCostAccounting()
        accounting.note_decision("D", 42)
        stage = accounting.snapshot(host_wall_ns=1)["decision_stages"]["D"]
        self.assertEqual(
            (stage["p50_ns"], stage["p95_ns"], stage["p99_ns"],
             stage["max_ns"], stage["mean_ns"]),
            (42, 42, 42, 42, 42))

    def test_empty_stage_reports_zero_block(self):
        snapshot = PredictorCostAccounting().snapshot(host_wall_ns=1)
        for stage in DECISION_STAGES:
            block = snapshot["decision_stages"][stage]
            self.assertEqual(
                (block["count"], block["total_ns"], block["p50_ns"],
                 block["p99_ns"], block["candidates_max"]),
                (0, 0, 0, 0, 0))


class FailClosedValidationTest(unittest.TestCase):
    """账本 fail-closed：未知阶段/负值/布尔值/非法类型一律拒绝。"""

    def test_unknown_stage_rejected(self):
        accounting = PredictorCostAccounting()
        with self.assertRaisesRegex(ValueError, "unknown predictor "
                                                "decision stage"):
            accounting.note_decision("X", 1)

    def test_negative_and_bool_rejected(self):
        accounting = PredictorCostAccounting()
        for kwargs in ({"wall_ns": -1}, {"candidates": -1},
                       {"wall_ns": True}, {"candidates": False}):
            with self.assertRaises(ValueError):
                accounting.note_decision("P", kwargs.pop("wall_ns", 1),
                                         **kwargs)
        with self.assertRaises(ValueError):
            accounting.note_summary_maintenance(-5)
        with self.assertRaises(ValueError):
            accounting.note_log_bytes(-1)
        with self.assertRaises(ValueError):
            accounting.note_backend_batch(nodes=-1, parent_edges=0,
                                          watches=0, future_alarms=0)
        with self.assertRaises(ValueError):
            accounting.snapshot(host_wall_ns=-1)
        with self.assertRaises(ValueError):
            accounting.snapshot(host_wall_ns=1, decision_log_bytes=-1)

    def test_zero_host_wall_snapshot_reports_none_share(self):
        """退化夹具（构造即快照，host 锚差 0）：占比 None，不造 0/0。"""
        snapshot = PredictorCostAccounting().snapshot(host_wall_ns=0)
        self.assertIsNone(snapshot["host_share_ppm"])


class SnapshotContentTest(unittest.TestCase):
    """run 末快照字段完备性（§19.3 证据面）。"""

    def _populated(self):
        accounting = PredictorCostAccounting()
        accounting.note_decision("P", 100, candidates=8)
        accounting.note_decision("P", 300, candidates=8)
        accounting.note_decision("D", 25, candidates=8)
        accounting.note_summary_maintenance(7)
        accounting.note_summary_maintenance(3)
        accounting.note_backend_batch(
            nodes=4, parent_edges=3, watches=2, future_alarms=1)
        accounting.note_backend_batch(
            nodes=1, parent_edges=0, watches=0, future_alarms=0)
        return accounting

    def test_snapshot_covers_all_section19_3_fields(self):
        snapshot = self._populated().snapshot(
            host_wall_ns=1_000_000, decision_log_bytes=4096)
        # 预测器合计 = 决策（P+D）+ 摘要维护。
        self.assertEqual(snapshot["decision_wall_ns"],
                         {"count": 3, "total_ns": 425})
        self.assertEqual(snapshot["summary_maintenance"],
                         {"count": 2, "total_ns": 10, "max_ns": 7})
        self.assertEqual(snapshot["predictor_total_ns"], 435)
        self.assertEqual(snapshot["host_wall_ns"], 1_000_000)
        # 总占比整数 ppm：435 / 1e6 → 435 ppm。
        self.assertEqual(snapshot["host_share_ppm"], 435)
        # 峰值 RSS：Linux ru_maxrss（KiB→字节）口径如实披露；无
        # resource 模块的平台（Windows）按设计回落 "unavailable"（
        # 2026-10-01 R2：无条件断言在无 resource 平台恒红）。
        try:
            import resource as _resource  # noqa: F401
            self.assertEqual(snapshot["peak_rss"]["source"], "ru_maxrss")
            self.assertGreater(snapshot["peak_rss"]["bytes"], 0)
        except ImportError:
            self.assertEqual(snapshot["peak_rss"]["source"], "unavailable")
        self.assertEqual(snapshot["decision_log_bytes"], 4096)
        # 后端事件量逐批累加。
        self.assertEqual(snapshot["backend_events"], {
            "batches": 2, "nodes": 5, "parent_edges": 3,
            "watches": 2, "future_alarms": 1})
        # 候选面宽度逐阶段披露（合成输入的逐决策证据）。
        self.assertEqual(
            snapshot["decision_stages"]["P"]["candidates_max"], 8)
        self.assertEqual(
            snapshot["decision_stages"]["D"]["candidates_max"], 8)

    def test_snapshot_is_json_serializable(self):
        import json
        snapshot = self._populated().snapshot(host_wall_ns=10)
        json.dumps(snapshot, sort_keys=True)  # 不抛即过


class Sh30WiringSmokeTest(unittest.TestCase):
    """sh30 接线冒烟：真实调度器壳的 P/D 决策与发射路径喂入账本，
    run 末快照可直接产出（不跑仿真——纯调度器壳单测）。"""

    def test_drain_records_boundary_sample_and_summary_fee(self):
        from joint.test_d18_boundary_decision import (
            _drain_ready_scheduler)
        scheduler, kv, runtime = _drain_ready_scheduler(decode_length=8)
        before = scheduler.predictor_cost.snapshot(host_wall_ns=1)
        self.assertEqual(
            before["decision_stages"]["D"]["count"], 0)
        scheduler._on_prefill_drain("r0", 1000)
        self.assertEqual(runtime.joint_stage, "D_COMMITTED")
        after = scheduler.predictor_cost.snapshot(host_wall_ns=10 ** 12)
        # 边界二次决策样本恰一（D 阶段），候选面 = 4 动作 × 2 实例。
        d_stage = after["decision_stages"]["D"]
        self.assertEqual(d_stage["count"], 1)
        self.assertEqual(d_stage["candidates_max"], 6)
        self.assertGreater(d_stage["total_ns"], 0)
        # 容量摘要查询费样本恰一（_joint_cost_model 构造器单点）。
        self.assertEqual(after["summary_maintenance"]["count"], 1)
        # 占比分母取显式宿主墙时（10^12 ns）→ ppm 有界。
        self.assertGreaterEqual(after["host_share_ppm"], 0)

    def test_backend_batch_hook_counts_events(self):
        scheduler = _wired_shell()
        batch = {"nodes": [1, 2, 3], "parent_edges": [(1, 2)],
                 "watches": [1], "future_alarms": []}
        scheduler._note_backend_batch(batch)
        scheduler._note_backend_batch(batch)
        snapshot = scheduler.predictor_cost.snapshot(host_wall_ns=1)
        self.assertEqual(snapshot["backend_events"],
                         {"batches": 2, "nodes": 6, "parent_edges": 2,
                          "watches": 2, "future_alarms": 0})


def _wired_shell():
    from online.sh30_online_scheduler import Sh30OnlineScheduler
    scheduler = Sh30OnlineScheduler.__new__(Sh30OnlineScheduler)
    scheduler.predictor_cost = PredictorCostAccounting()
    scheduler._predictor_cost_host_anchor_ns = 0
    return scheduler


if __name__ == "__main__":
    unittest.main()
