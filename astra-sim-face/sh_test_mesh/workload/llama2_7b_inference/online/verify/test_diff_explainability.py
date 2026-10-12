#!/usr/bin/env python3
"""test_diff_explainability.py -- diff_explainability 本地自测。

覆盖(2026-09-29 加固):
  1. 门计数器正则命中 C++ 真实单行格式 "[online] gate counters:
     event_count=N delivery_count=M ..."(旧正则前缀紧邻永不命中,
     delivery_count/event_count 对平静默失效假绿);
  2. 决策行一致但载体摘要不一致 → 契约缺陷级,exit 1(旧写法恒 exit 0);
  3. 两侧完全一致 → exit 0(基线语义保持)。
"""
import json
import os
import sys
import tempfile
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_ONLINE_DIR = os.path.dirname(_HERE)
for _p in (_HERE, _ONLINE_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import diff_explainability  # noqa: E402


# 与 main_online.cc 的 std::cout 输出逐字节同构的样例日志行。
_GATE_LINE = ("[online] gate counters: event_count=42 delivery_count=7 "
              "coalescing_ratio=0.125 tick_end_without_decision_count=3")
_SERVICE_LINE = ("[online] service counters: accepted=9 completed=2 "
                 "active=4 pending_alarm=1")


def _write_run(root, *, completed_group=None, delivery_count=7):
    """构造最小合法运行目录(bridge/ + cpp.log)。"""
    bridge = os.path.join(root, "bridge")
    os.makedirs(bridge)
    record = {"delivery_sequence": 0, "request_id": "r0", "tick": 11}
    if completed_group is not None:
        record["completed_groups"] = [
            {"request_id": "r0", "stage": completed_group}]
    with open(os.path.join(bridge, "online_decision_log.jsonl"), "w",
              encoding="utf-8") as out:
        out.write(json.dumps({
            "seq": 0, "tick": 11, "kind": "prefill", "request_id": "r0",
            "decision": {"action": "prefill", "instance": 0}}) + "\n")
    with open(os.path.join(bridge, "graph_batch_digests.jsonl"), "w",
              encoding="utf-8") as out:
        out.write(json.dumps({
            "delivery_sequence": 0, "tick": 11, "reasons": ["head"],
            "ranks": [0], "node_count": 1, "edge_count": 0,
            "watch_count": 1, "content_sha256": "ab" * 32}) + "\n")
    with open(os.path.join(bridge, "request_journal.jsonl"), "w",
              encoding="utf-8") as out:
        out.write(json.dumps(record) + "\n")
    with open(os.path.join(root, "cpp.log"), "w", encoding="utf-8") as out:
        out.write(_GATE_LINE.replace("delivery_count=7",
                                     "delivery_count={}".format(
                                         delivery_count)) + "\n")
        out.write(_SERVICE_LINE + "\n")


class GateCounterRegexTest(unittest.TestCase):
    """门计数器正则按 C++ 实际格式命中(H2/R3-K1)。"""

    def test_gate_counter_line_is_parsed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "cpp.log")
            with open(path, "w", encoding="utf-8") as out:
                out.write(_GATE_LINE + "\n")
                out.write(_SERVICE_LINE + "\n")
            counters = diff_explainability._load_cpp_counters(path)
        self.assertIsNotNone(counters)
        self.assertEqual(counters["event_count"], 42)
        self.assertEqual(counters["delivery_count"], 7)
        self.assertEqual(counters["accepted"], 9)
        self.assertEqual(counters["completed"], 2)

    def test_gz_fallback_still_parses(self):
        import gzip
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "cpp.log.gz")
            with gzip.open(path, "wt", encoding="utf-8") as out:
                out.write(_GATE_LINE + "\n")
                out.write(_SERVICE_LINE + "\n")
            counters = diff_explainability._load_cpp_counters(
                path[:-len(".gz")])
        self.assertIsNotNone(counters)
        self.assertEqual(counters["delivery_count"], 7)

    def test_missing_cpp_log_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(diff_explainability._load_cpp_counters(
                os.path.join(tmp, "cpp.log")))


class VerdictGatingTest(unittest.TestCase):
    """载体不一致必须走差异出口(R3-K4)。"""

    def _run_main(self, baseline_root, sensing_root, report_path):
        return diff_explainability.main([
            "--baseline-run", baseline_root,
            "--sensing-run", sensing_root,
            "--report", report_path])

    def test_identical_runs_exit_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = os.path.join(tmp, "base")
            sens = os.path.join(tmp, "sens")
            _write_run(base)
            _write_run(sens)
            report = os.path.join(tmp, "report.md")
            self.assertEqual(self._run_main(base, sens, report), 0)

    def test_identical_decisions_but_carrier_diff_exits_nonzero(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = os.path.join(tmp, "base")
            sens = os.path.join(tmp, "sens")
            # 决策日志逐字节一致;完成事实载体不同(sensing 侧多一条
            # prefill 完成边)→ 无法归类 = 契约缺陷级。
            _write_run(base)
            _write_run(sens, completed_group="prefill", delivery_count=7)
            report = os.path.join(tmp, "report.md")
            self.assertEqual(self._run_main(base, sens, report), 1)
            with open(report, "r", encoding="utf-8") as inp:
                text = inp.read()
            self.assertIn("载体摘要不一致", text)

    def test_carrier_only_gate_counter_diff_exits_nonzero(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = os.path.join(tmp, "base")
            sens = os.path.join(tmp, "sens")
            # 唯一差异 = 门计数器 delivery_count(修复前该维度恒缺失,
            # 检不出;修复后必须进差异出口)。
            _write_run(base, delivery_count=7)
            _write_run(sens, delivery_count=8)
            report = os.path.join(tmp, "report.md")
            self.assertEqual(self._run_main(base, sens, report), 1)
            with open(report, "r", encoding="utf-8") as inp:
                text = inp.read()
            self.assertIn("门计数器不一致", text)
            self.assertIn("'delivery_count': 8", text)


if __name__ == "__main__":
    unittest.main()
