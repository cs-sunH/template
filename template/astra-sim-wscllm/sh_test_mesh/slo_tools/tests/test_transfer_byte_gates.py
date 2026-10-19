#!/usr/bin/env python3
"""test_transfer_byte_gates.py -- H2 (2026-10-01) 回归钉死。

钉死 kv_cache_adapter 提取链的 total_bytes 字节门语义（F1/G1/H1 批）：

  1. total_bytes 显式 0、回退 shards 全 0、回退 shards 空列表三种形态
     一律**静默跳过**（与 native 侧 ">0 才入桶"惯例和提取链改前的
     `if not nbytes: return []` 语义一致；不得以 fail 阻断对账）；
  2. 正常正整数产恰一条 canonical 事件，bytes 原值透传；
  3. bool（isinstance(True, int) 冒充）、浮点、字符串、负值一律
     fail-closed（SloToolError，driver 归一为 rc=2），绝不强制转换。

背景：F1 引入 _strict_total_bytes 扼流点时第一版对 0 也 fail（G1 指认），
且修复一度未落地（H1 指认）——本文件使这两类回归此后可被 pytest 证伪。
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

SLO_TOOLS_DIR = Path(__file__).resolve().parent.parent
if str(SLO_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(SLO_TOOLS_DIR))

import kv_cache_adapter  # noqa: E402
from slo_common import SloToolError  # noqa: E402


def _run_transfer(total_bytes, shards=None):
    transfer = {"kind": "pd_transfer", "reason": "unit"}
    if total_bytes is not None:
        transfer["total_bytes"] = total_bytes
    if shards is not None:
        transfer["shards"] = shards
    return kv_cache_adapter._transfer_object_events(
        transfer, "req_x", 7, "S1")


class TransferByteGateTests(unittest.TestCase):
    """_transfer_object_events 四形态 + 扼流点负值门。"""

    def test_positive_total_yields_exact_event(self):
        events = _run_transfer(512)
        self.assertEqual(len(events), 1)
        self.assertEqual(int(events[0]["bytes"]), 512)

    def test_zero_total_is_silent_skip(self):
        self.assertEqual(_run_transfer(0), [])

    def test_all_zero_shards_fallback_is_silent_skip(self):
        self.assertEqual(
            _run_transfer(None, [{"bytes": 0}, {"bytes": 0}]), [])

    def test_empty_shards_fallback_is_silent_skip(self):
        self.assertEqual(_run_transfer(None, []), [])

    def test_negative_total_fails_closed(self):
        with self.assertRaises(SloToolError):
            _run_transfer(-5)

    def test_bool_total_neutralized_not_counted_as_one(self):
        # True 曾以 isinstance(int) 冒充 1 字节事件；现在走 shards 回退，
        # 无 shards → None → 静默跳过。
        self.assertEqual(_run_transfer(True), [])

    def test_float_and_string_total_take_shards_fallback(self):
        # 浮点/字符串不是 int → 进 shards 回退臂；无 shards → None →
        # 静默跳过（与 F1 前语义一致）。扼流点自身的 float/string 拒绝
        # 由下一条用例直接钉在 _strict_total_bytes 上。
        self.assertEqual(_run_transfer(3.5), [])
        self.assertEqual(_run_transfer("12"), [])

    def test_strict_helper_rejects_bool_float_string_negative(self):
        for bad in (True, 3.5, "12", -5):
            with self.subTest(bad=bad):
                with self.assertRaises(SloToolError):
                    kv_cache_adapter._strict_total_bytes(bad, "unit")

    def test_strict_helper_accepts_zero(self):
        # 扼流点本身容 0（负值才 fail）——0 的跳过决策在调用方。
        self.assertEqual(
            kv_cache_adapter._strict_total_bytes(0, "unit"), 0)


if __name__ == "__main__":
    unittest.main()
