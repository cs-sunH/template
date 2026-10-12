#!/usr/bin/env python3
"""test_d17_rounding.py -- D17 修复轮二：transfer_factor 链路窗口样本
base_ns 的向上取整钉测（joint_cost_model.observe_transfer_from_link_
window，2026-09-27）。

规格：``base_ns = ⌈served_bytes / nominal_rate_bytes_per_ns⌉``（旧
``int()`` 向下取整废除——基准时长不为少报方向截断）。样本方向：base
变大 ⇒ η_sample = base/actual 变大（更保守/拥胀方向——规格方向，
不回转）。恰整除值零漂移；零字节仍按 ``_record`` 零分母拒收披露
（ceil(0)=0 ⇒ base_ns<=0 拒收路径不变）。

与既有 joint/test_d17_schema_version_rounding.py（计价侧传输取整）
互补：本文件钉**观测样本侧**的取整与 η 方向。

运行：cd sh_test_mesh/workload/llama2_7b_inference &&
      python3 joint/test_d17_rounding.py   （或 pytest 同路径）
"""
import os
import sys
import unittest

_JOINT_DIR = os.path.dirname(os.path.abspath(__file__))
_WORKLOAD_DIR = os.path.dirname(_JOINT_DIR)
for _p in (_JOINT_DIR, _WORKLOAD_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from joint.joint_cost_model import ServiceFactors  # noqa: E402


class D17SampleBaseRoundingTest(unittest.TestCase):
    """observe_transfer_from_link_window：base 向上取整 + η 方向。"""

    def test_base_ns_ceils_non_divisible_bytes(self):
        # 1001 B @ 10 B/ns = 100.1 ns：旧 int() = 100（少报），D17
        # ceil = 101。经 flush 后的样本路径核账：actual=202 ns ⇒
        # η_sample = 101/202；factor = 1/η_new，η_new = 15/16·1 +
        # 1/16·(101/202)。
        factors = ServiceFactors()
        factors.observe_transfer_from_link_window(
            served_bytes=1001, active_ns=202,
            nominal_rate_bytes_per_ns=10.0, tick_ns=1000)
        factors.flush()
        self.assertEqual(factors.updates.get("transfer_factor"), 1)
        eta_sample = 101 / 202
        eta_new = (15.0 / 16.0) * 1.0 + (1.0 / 16.0) * eta_sample
        self.assertAlmostEqual(
            factors.transfer_factor, 1.0 / eta_new, places=12,
            msg="ceil(100.1)=101 must be the sample base (old floor=100 "
                "would give a different factor)")

    def test_base_ns_exact_division_undrifted(self):
        # 恰整除（1000 B @ 10 B/ns = 100 ns）：ceil 零漂移锚。
        factors = ServiceFactors()
        factors.observe_transfer_from_link_window(
            served_bytes=1000, active_ns=200,
            nominal_rate_bytes_per_ns=10.0, tick_ns=1000)
        factors.flush()
        # η_sample = 100/200 = 0.5（恰好；旧口径同值——整除样本零漂移）。
        eta_new = (15.0 / 16.0) * 1.0 + (1.0 / 16.0) * 0.5
        self.assertAlmostEqual(
            factors.transfer_factor, 1.0 / eta_new, places=12)

    def test_ceil_direction_grows_eta_not_shrinks(self):
        # 规格方向：同 actual 下 base 从 floor 变 ceil ⇒ η_sample =
        # base/actual 变大（η 向 1 靠近——实测相对基准的效率不再被取整
        # 少报夸大成更慢；factor = 1/η 相应变小）。方向属规格、不回转。
        ceil_factors = ServiceFactors()
        ceil_factors.observe_transfer_from_link_window(
            served_bytes=1001, active_ns=150,
            nominal_rate_bytes_per_ns=10.0, tick_ns=1000)
        ceil_factors.flush()
        floor_factors = ServiceFactors()
        floor_factors.observe_transfer(
            actual_ns=150, base_ns=100, tick_ns=1000)  # 旧 floor 口径
        floor_factors.flush()
        # η 样本方向：ceil(100.1)=101 > floor=100。
        self.assertGreater(101 / 150, 100 / 150)
        # 因子 = 1/η_new 随 η_new 变大而变小（非回转，是同方向的代数
        # 推论；断言两侧口径差异确实源于 base 的取整）。
        self.assertLess(
            ceil_factors.transfer_factor, floor_factors.transfer_factor)

    def test_tiny_bytes_no_longer_rejected_by_zero_base(self):
        # 1 B @ 10 B/ns = 0.1 ns：旧 int() = 0 ⇒ 零分母拒收（样本被
        # 无声丢弃）；D17 ceil = 1 ⇒ 正常落样（小样本不再因取整消失）。
        factors = ServiceFactors()
        factors.observe_transfer_from_link_window(
            served_bytes=1, active_ns=2,
            nominal_rate_bytes_per_ns=10.0, tick_ns=1000)
        factors.flush()
        self.assertEqual(factors.updates.get("transfer_factor"), 1)
        self.assertNotIn("transfer_factor", factors.rejected)

    def test_zero_bytes_still_rejected_with_disclosure(self):
        # 零字节：ceil(0)=0 ⇒ base_ns=0 走 ``_record`` 零分母拒收计数
        # （披露不变，不静默）。
        factors = ServiceFactors()
        factors.observe_transfer_from_link_window(
            served_bytes=0, active_ns=5,
            nominal_rate_bytes_per_ns=10.0, tick_ns=1000)
        factors.flush()
        self.assertNotIn("transfer_factor", factors.updates)
        self.assertEqual(factors.rejected.get("transfer_factor"), 1)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
