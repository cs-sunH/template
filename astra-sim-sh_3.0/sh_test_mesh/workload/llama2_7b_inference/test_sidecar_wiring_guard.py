#!/usr/bin/env python3
"""test_sidecar_wiring_guard.py -- sidecar 接线 fail-closed 守卫单元测试
（《sidecar装配链路修复执行方案.md》修复 A；背景：排查报告高-4①）。"""
import os
import sys
import unittest
from pathlib import Path

_WORKLOAD_DIR = os.path.dirname(os.path.abspath(__file__))
if _WORKLOAD_DIR not in sys.path:
    sys.path.insert(0, _WORKLOAD_DIR)

import tempfile

from generate_face_trace import _require_sidecar_wiring  # noqa: E402


class SidecarWiringGuardTest(unittest.TestCase):

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)

    def test_plain_queue_with_sibling_and_no_context_fails(self):
        queue = self.dir / "astra_compute_20_first_30_seconds_request_queue.csv"
        queue.write_text("header\n", encoding="utf-8")
        (self.dir / "astra_compute_20_first_30_seconds_request_context.csv"
         ).write_text("header\n", encoding="utf-8")
        with self.assertRaises(SystemExit):
            _require_sidecar_wiring(queue, None)

    def test_recompute_queue_with_sibling_passes(self):
        queue = (self.dir /
                 "astra_compute_20_first_30_seconds_request_queue_recompute.csv")
        queue.write_text("header\n", encoding="utf-8")
        (self.dir / "astra_compute_20_first_30_seconds_request_context.csv"
         ).write_text("header\n", encoding="utf-8")
        _require_sidecar_wiring(queue, None)  # 不抛即通过

    def test_plain_queue_without_sibling_passes(self):
        queue = self.dir / "synthetic_request_queue.csv"
        queue.write_text("header\n", encoding="utf-8")
        _require_sidecar_wiring(queue, None)  # 合成/测试队列不适用

    def test_plain_queue_with_sibling_and_context_passes(self):
        queue = self.dir / "astra_compute_20_first_30_seconds_request_queue.csv"
        queue.write_text("header\n", encoding="utf-8")
        (self.dir / "astra_compute_20_first_30_seconds_request_context.csv"
         ).write_text("header\n", encoding="utf-8")
        _require_sidecar_wiring(queue, "traces/ctx.csv")  # 已接线


if __name__ == "__main__":
    unittest.main()
