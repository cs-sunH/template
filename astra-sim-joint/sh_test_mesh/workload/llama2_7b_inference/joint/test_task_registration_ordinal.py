#!/usr/bin/env python3
"""test_task_registration_ordinal.py -- 任务注册序号（uint64）与完成
序解耦的 wire 钉测。

2026-10-14 远读退役同步：pipeline_identity 分配器三用例（preview/
commit 原子性、P/D 共域、状态恢复）随 joint.pipeline_identity 模块
整删（§7.1；方案 §7.2 原记 2 用例，实测该模块在本文共有 3 个消费
用例，全部随被测对象删除）。

运行：cd sh_test_mesh/workload/llama2_7b_inference &&
      python3 joint/test_task_registration_ordinal.py
"""
import os
import sys
import unittest

_JOINT_DIR = os.path.dirname(os.path.abspath(__file__))
_WORKLOAD_DIR = os.path.dirname(_JOINT_DIR)
for _path in (_JOINT_DIR, _WORKLOAD_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from joint.native_projection import _task_wire  # noqa: E402
from joint.resource_projection import (  # noqa: E402
    ResourceProjectionError,
    TaskSpec,
)


class TaskRegistrationOrdinalTest(unittest.TestCase):
    def test_uint64_registration_is_separate_from_completion_order(self):
        task = TaskSpec("leg", order=-1, registration_ordinal=0)
        wire = _task_wire(task)
        self.assertEqual(wire["order"], -1)
        self.assertEqual(wire["registration_ordinal"], 0)

    def test_registration_ordinal_rejects_bool_and_out_of_range_values(self):
        for invalid in (True, -1, 1 << 64):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ResourceProjectionError):
                    TaskSpec("leg", registration_ordinal=invalid)


if __name__ == "__main__":
    unittest.main()
