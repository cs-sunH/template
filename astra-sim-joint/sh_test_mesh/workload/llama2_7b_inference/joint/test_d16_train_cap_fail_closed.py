#!/usr/bin/env python3
"""test_d16_train_cap_fail_closed.py -- D16 钉测：列车上限 T_max 显式 ≤0
fail-closed（v2.0 §17.1 有界性）。

背景（审查复现）：SH_TRAIN_MAX_ITER=0（无界诊断配置）的旧解析
（``int(env or 0)``）会让列车折叠/预建面扩张到整轮节点数（2s 窗实测
3712 节点）——违反有界性合同。修正 = ``parse_train_max_iter`` 单一
裁决点（joint_config.py）：

* 缺省/未设置（None）→ 按协议推导合法值（交付默认 8，§3.2.8/§7.4
  A2 对拍裁决）——该路径不受影响；
* 显式 ≤0（0 = 无界、负值）→ JointConfigError，错误信息说约束；
* 非整数/带空白/空值 → 同拒（数值合同）；
* 合法正整数照常解析。

运行：cd sh_test_mesh/workload/llama2_7b_inference &&
      python3 -m pytest joint/test_d16_train_cap_fail_closed.py -q
"""
import inspect
import os
import sys
import unittest

_JOINT_DIR = os.path.dirname(os.path.abspath(__file__))
_WORKLOAD_DIR = os.path.dirname(_JOINT_DIR)
for _p in (_JOINT_DIR, _WORKLOAD_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from joint.joint_config import (  # noqa: E402
    TRAIN_MAX_ITER_DEFAULT,
    JointConfigError,
    parse_train_max_iter,
)


class TrainMaxIterParseTest(unittest.TestCase):
    """parse_train_max_iter 单一裁决点合同。"""

    def test_unset_derives_protocol_default(self):
        # 缺省/未设置 = 按协议推导合法值（不是 0/无界）。
        self.assertEqual(parse_train_max_iter(None), TRAIN_MAX_ITER_DEFAULT)
        self.assertEqual(TRAIN_MAX_ITER_DEFAULT, 8)

    def test_legal_values_unaffected(self):
        # 合法正整数照常解析（含交付默认同值与更大治理值）。
        self.assertEqual(parse_train_max_iter("8"), 8)
        self.assertEqual(parse_train_max_iter("16"), 16)
        self.assertEqual(parse_train_max_iter("1"), 1)

    def test_explicit_zero_rejected_with_constraint(self):
        # 0 = 无界诊断配置，违反 v2.0 §17.1 有界性——fail-closed，错误
        # 信息须说约束（正整数上限 + 有界性 + 拒绝语义）。
        with self.assertRaises(JointConfigError) as ctx:
            parse_train_max_iter("0")
        message = str(ctx.exception)
        self.assertIn("SH_TRAIN_MAX_ITER", message)
        self.assertIn("positive integer", message)
        self.assertIn("boundedness", message)

    def test_explicit_negative_rejected(self):
        with self.assertRaises(JointConfigError) as ctx:
            parse_train_max_iter("-3")
        self.assertIn("boundedness", str(ctx.exception))

    def test_malformed_values_rejected(self):
        # 数值合同：空值/带空白/非整数一律拒绝（不静默取 0）。
        for raw in ("", " 8", "8 ", "8.5", "eight", "+"):
            with self.assertRaises(JointConfigError):
                parse_train_max_iter(raw)

    def test_env_unset_maps_to_none_semantics(self):
        # 生产接线语义：os.environ.get 缺省 None -> 协议推导值（钉住
        # 调度器侧不得再带 "8" 字面缺省或 or-0 回退）。
        self.assertEqual(
            parse_train_max_iter(os.environ.get("SH_TRAIN_MAX_ITER")),
            TRAIN_MAX_ITER_DEFAULT)


class TrainMaxIterProductionWiringTest(unittest.TestCase):
    """sh30 生产接线：_train_max_iter 供给源 = parse_train_max_iter。"""

    def test_scheduler_init_consumes_single_adjudication_point(self):
        from online.sh30_online_scheduler import Sh30OnlineScheduler
        source = inspect.getsource(Sh30OnlineScheduler.__init__)
        self.assertIn("parse_train_max_iter", source)
        self.assertIn("SH_TRAIN_MAX_ITER", source)
        # 旧解析形态（字面 "8" 缺省 + or-0 无界回退）不得残留。
        self.assertNotIn(
            'os.environ.get("SH_TRAIN_MAX_ITER", "8")', source)


if __name__ == "__main__":
    unittest.main()
