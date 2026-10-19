#!/usr/bin/env python3
"""test_bridge_seq_ascii_guard.py -- 桥序号文件名 ASCII 数字校验钉子
(Minor 2026-09-30:str.isdigit() 放行 Unicode 数字——全角序号经 int()
归一化与 ASCII 同号碰撞静默跳过、上标数字直接 int() 崩溃;桥文件名
序号只会由本侧与 C++ 以 ASCII 十进制写出)。

Run: python3 online/test_bridge_seq_ascii_guard.py
"""
import os
import sys
import tempfile
import unittest

_ONLINE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ONLINE_DIR)

from bridge_request_journal import _loose_request_sequences  # noqa: E402
from decision_bridge import BridgeServer  # noqa: E402


class BridgeSeqAsciiGuardTests(unittest.TestCase):
    def test_seq_of_accepts_ascii_and_rejects_unicode_digits(self):
        self.assertEqual(BridgeServer._seq_of("request_12.json", "request_"), 12)
        self.assertIsNone(
            BridgeServer._seq_of("request_１２.json", "request_"))
        self.assertIsNone(
            BridgeServer._seq_of("request_².json", "request_"))
        self.assertIsNone(BridgeServer._seq_of("request_abc.json", "request_"))

    def test_loose_request_sequences_skips_unicode_filenames(self):
        bridge_dir = tempfile.mkdtemp(prefix="wscllm_bridge_ascii_")
        try:
            for name in ("request_3.json", "request_４.json",
                         "request_Ⅴ.json", "other.json"):
                with open(os.path.join(bridge_dir, name), "w",
                          encoding="utf-8") as handle:
                    handle.write("{}")
            self.assertEqual(_loose_request_sequences(bridge_dir), [3])
        finally:
            for name in os.listdir(bridge_dir):
                os.unlink(os.path.join(bridge_dir, name))
            os.rmdir(bridge_dir)


if __name__ == "__main__":
    unittest.main()
