"""Fail-closed read contracts for config_resolver (RV6-F2, 2026-09-29).

E2 (2026-09-28) added context-wrapping handlers for UnicodeDecodeError and
OSError to config_resolver._read_json: a non-UTF-8 byte sequence used to
escape as a bare UnicodeDecodeError with no filename context, and a read
OSError escaped without description/path. Both now surface as
ValueError/OSError carrying "description + path (+ byte offset)". These two
cases pin that contract at the public entry load_hardware_config so a
regression back to the bare exceptions turns the suite red.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path


_SH_TEST_DIR = Path(__file__).resolve().parents[1]
if str(_SH_TEST_DIR) not in sys.path:
    sys.path.insert(0, str(_SH_TEST_DIR))

from config_resolver import load_hardware_config


_HARDWARE_SOURCE = _SH_TEST_DIR / "hardware" / "face_case5_config_c.json"


class ConfigResolverReadFailClosedTests(unittest.TestCase):
    def setUp(self) -> None:
        source = json.loads(_HARDWARE_SOURCE.read_text(encoding="utf-8"))
        self.profile_name = next(iter(source["local-hbm"]["capacity-profiles"]))

    def test_non_utf8_hardware_source_reports_description_path_and_byte_offset(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            bad_path = Path(temporary) / "hardware_bad_bytes.json"
            bad_path.write_bytes(b"\xff\xfe{}")
            with self.assertRaises(ValueError) as caught:
                load_hardware_config(bad_path, self.profile_name)
            message = str(caught.exception)
            self.assertIn("is not valid UTF-8", message)
            self.assertIn("Hardware configuration", message)
            self.assertIn(str(bad_path), message)
            self.assertRegex(message, r"at byte 0\b")

    # (2026-10-01, I2 复审) geteuid 在 win32 不存在:门先探平台,POSIX 才
    # 求值(与仓内其它 POSIX-only 用例的登记口径一致;官方目标环境为
    # Linux/WSL,win32 下整用例跳过而非收集期崩溃)。
    @unittest.skipIf(os.name == "nt", "POSIX permission semantics only")
    @unittest.skipIf(os.name != "nt" and os.geteuid() == 0,
                     "permission bits do not bind root")
    def test_unreadable_hardware_source_reports_description_and_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            locked_path = Path(temporary) / "hardware_unreadable.json"
            locked_path.write_text("{}", encoding="utf-8")
            locked_path.chmod(0o000)
            with self.assertRaises(OSError) as caught:
                load_hardware_config(locked_path, self.profile_name)
            message = str(caught.exception)
            self.assertIn("Cannot read Hardware configuration", message)
            self.assertIn(str(locked_path), message)


if __name__ == "__main__":
    unittest.main()
