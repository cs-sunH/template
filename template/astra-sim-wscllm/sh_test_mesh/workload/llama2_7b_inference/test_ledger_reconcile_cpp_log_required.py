#!/usr/bin/env python3
"""test_ledger_reconcile_cpp_log_required.py -- P2-19 钉死测试。

契约（2026-09-28 修复）：ledger_reconcile 的 ``--cpp-log`` 必填——
R0d/R0e（service counters）与 R2c/R3a/R3c（phase-5/4 audit）全以
cpp.log 审计计数器为权威数据源；缺省 None 时全部 ``is not None`` 门
跳过 = R0d/R0e 完全空洞通过、R2a/R3a/R3c 部分削弱，报告仍打 PASS
（vacuous pass），违反文件头"cpp.log 审计计数器为权威数据源"契约。

钉死两类拒绝路径：
  1. 缺 --cpp-log → argparse required 报错、退出码 2（usage 至 stderr）；
  2. --cpp-log 指向不存在的文件（含 .gz 回退后仍无）→ FAIL-CLOSED
     exit 2（防拼写错路径退化回空洞通过）。
  3.（2026-10-01 补口）文件在场但三条权威计数器行缺失/不完整（截断、
     归档损坏、措辞漂移、未跑到 phase-4/5 的日志）→ FAIL-CLOSED
     exit 2——原先 load_* 返回 None 后各 ``is not None`` 门跳过 =
     空洞通过，与"文件缺失"同罪但路径不同。
"""

import io
import sys
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

MODULE_DIR = Path(__file__).resolve().parent
VERIFY_DIR = MODULE_DIR / "online" / "verify"
for _p in (str(VERIFY_DIR), str(MODULE_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ledger_reconcile  # noqa: E402


class CppLogRequiredTests(unittest.TestCase):
    """--cpp-log 缺参必红；坏路径必红（不得退化回空洞通过）。"""

    def test_missing_cpp_log_flag_rejected(self):
        stderr = io.StringIO()
        with redirect_stderr(stderr), redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit) as ctx:
                ledger_reconcile.main([
                    "--bridge-dir", "/tmp/nonexistent_bridge",
                    "--manifest", "/tmp/nonexistent_manifest.json",
                ])
        self.assertEqual(ctx.exception.code, 2)
        self.assertIn("--cpp-log", stderr.getvalue())

    def test_nonexistent_cpp_log_path_fails_closed(self):
        stderr = io.StringIO()
        with redirect_stderr(stderr), redirect_stdout(io.StringIO()):
            rc = ledger_reconcile.main([
                "--bridge-dir", "/tmp/nonexistent_bridge",
                "--manifest", "/tmp/nonexistent_manifest.json",
                "--cpp-log", "/tmp/nonexistent_cpp.log",
            ])
        self.assertEqual(rc, 2)
        self.assertIn("FAIL-CLOSED: cpp.log 不存在", stderr.getvalue())

    def test_present_but_counterless_cpp_log_fails_closed(self):
        """2026-10-01 补口：文件在场但 [online] 计数器行缺失 → exit 2。

        截断/代际不符的 cpp.log 原先让 load_cpp_counters/
        load_cpp_audit 返回 None，R0d/R0e/R2c/R3a/R3c 的
        ``is not None`` 门全部跳过 = 空洞通过。
        """
        import os
        import tempfile
        tmpdir = tempfile.mkdtemp(prefix="lr_cpplog2_")
        log = os.path.join(tmpdir, "cpp.log")
        with open(log, "w", encoding="utf-8") as handle:
            handle.write("[online] node gc: erased=0 retained=0\n",)
            # truncated: no service counters / phase-5 / phase-4 lines
        stderr = io.StringIO()
        with redirect_stderr(stderr), redirect_stdout(io.StringIO()):
            import shutil
            self.addCleanup(shutil.rmtree, tmpdir, True)
            rc = ledger_reconcile.main([
                "--bridge-dir", "/tmp/nonexistent_bridge",
                "--manifest", "/tmp/nonexistent_manifest.json",
                "--cpp-log", log,
            ])
        self.assertEqual(rc, 2)
        self.assertIn("FAIL-CLOSED: cpp.log 在场但缺少", stderr.getvalue())

    def test_present_but_partial_audit_cpp_log_fails_closed(self):
        """同一门的另一半：service counters 行在场、phase-4/phase-5
        审计行缺失（load_cpp_audit 只含 m5 或只含 m4）同样 exit 2。"""
        import os
        import tempfile
        tmpdir = tempfile.mkdtemp(prefix="lr_cpplog3_")
        log = os.path.join(tmpdir, "cpp.log")
        with open(log, "w", encoding="utf-8") as handle:
            handle.write("[online] service counters: accepted=1 "
                         "completed=1 active=0 pending_alarm=0\n",)
            handle.write("[online] phase-5 commit counters: "
                         "graph_batch_count=1 single_node_bridge_count=0 "
                         "total_nodes=1 avg_nodes_per_batch=1.0 "
                         "max_nodes_per_batch=1 total_watches=1 "
                         "total_assignments=0 total_kv_actions=0 "
                         "total_future_alarms=0\n",)
            # missing phase-4 end audit line
        stderr = io.StringIO()
        with redirect_stderr(stderr), redirect_stdout(io.StringIO()):
            import shutil
            self.addCleanup(shutil.rmtree, tmpdir, True)
            rc = ledger_reconcile.main([
                "--bridge-dir", "/tmp/nonexistent_bridge",
                "--manifest", "/tmp/nonexistent_manifest.json",
                "--cpp-log", log,
            ])
        self.assertEqual(rc, 2)
        self.assertIn("FAIL-CLOSED: cpp.log 在场但缺少", stderr.getvalue())

    def test_resolved_cpp_log_helper(self):
        """归档 .gz 透明回退解析：存在 .gz 即解析成功；全缺返回 None。"""
        import os
        import tempfile
        tmpdir = tempfile.mkdtemp(prefix="lr_cpplog_")
        gz = os.path.join(tmpdir, "cpp.log.gz")
        import gzip
        with gzip.open(gz, "wt", encoding="utf-8") as handle:
            handle.write("[online] service counters: accepted=1 completed=1"
                         " active=0 pending_alarm=0\n")
        self.assertEqual(
            ledger_reconcile._resolve_cpp_log_path(
                os.path.join(tmpdir, "cpp.log")), gz)
        self.assertEqual(
            ledger_reconcile._resolve_cpp_log_path(gz), gz)
        self.assertIsNone(
            ledger_reconcile._resolve_cpp_log_path(
                os.path.join(tmpdir, "missing.log")))
        self.assertIsNone(ledger_reconcile._resolve_cpp_log_path(None))


if __name__ == "__main__":
    unittest.main()
