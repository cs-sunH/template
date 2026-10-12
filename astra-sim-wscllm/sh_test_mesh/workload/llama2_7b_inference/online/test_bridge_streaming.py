#!/usr/bin/env python3
"""test_bridge_streaming.py -- 桥请求簿记有界性 + 请求日志流契约。

E1(2026-09-28)自 verify/bridge_streaming_fixture.py 规范形态化移植:原文件为
模块级裸断言脚本,全仓零引用、直跑零测试退出 0(假绿)、unittest discover
(官方镜像 pytest 缺失降级 runner)不可见(清单 §910 W 口径;J 仓同款七件
已于 2026-09-27 物理删除)。本文件把其断言本体接进 unittest/pytest 双
runner,断言语义逐条保留:

  1. _BoundedSequenceTracker:O(1) 严格单调重复过滤——mark 新 seq 推进
     next_seq、重复返回 False、跳号 BridgeError fail-closed、实例状态纯
     (只有 label/next_seq 两个字段,无随 seq 增长的集合);
  2. BridgeServer 有界簿记:已处理 request 文件流式退役,任意时刻松散尾
     <= 255 个,audit 流关闭后零松散;journal 行序 == seq 序;
  3. per_request_stats 行序 == seq 序,discard 后暂存流删除;
  4. canonical_request_producer 逐字节保留 C++ 请求字节 + 仅追加 JSONL
     分隔;非规范 pretty 输入走容忍归一化(紧凑重序列化,不盲信单行);
  5. iter_request_records 对 journal 跳号 fail-closed(ValueError)。

运行:pytest online/test_bridge_streaming.py,或
      cd sh_test_mesh/workload/llama2_7b_inference && python3 -m unittest
      online.test_bridge_streaming(本文件亦可直跑)。
"""
import json
import os
import sys
import tempfile
import unittest

_ONLINE_DIR = os.path.dirname(os.path.abspath(__file__))
_WORKLOAD_DIR = os.path.dirname(_ONLINE_DIR)
for _path in (_ONLINE_DIR, _WORKLOAD_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from bridge_request_journal import iter_request_records  # noqa: E402
from decision_bridge import (  # noqa: E402
    BridgeError,
    BridgeServer,
    _BoundedSequenceTracker,
)


def _request(seq):
    return {"schema_version": 1, "delivery_sequence": seq}


def _response(seq):
    return {"batch_id": seq, "source_delivery_sequence": seq}


class BoundedSequenceTrackerTest(unittest.TestCase):
    """O(1) 重复过滤:严格单调 + 跳号 fail-closed + 状态纯度。"""

    def test_strict_monotonic_mark_validate_and_purity(self):
        tracker = _BoundedSequenceTracker("fixture")
        self.assertTrue(tracker.mark(0))
        self.assertTrue(tracker.contains(0))
        self.assertFalse(tracker.mark(0))  # 重复
        with self.assertRaises(BridgeError):
            tracker.validate_new(2)  # 跳号(seq 1 缺席)
        self.assertTrue(tracker.mark(1))
        for seq in range(2, 100_000):
            self.assertTrue(tracker.mark(seq))
        self.assertEqual(tracker.next_seq, 100_000)
        # 状态纯度:无随 seq 增长的成员集合(O(1) 过滤器,不是 seen-set)。
        self.assertEqual(set(vars(tracker)), {"label", "next_seq"})


class BridgeServerStreamingTest(unittest.TestCase):
    """有界簿记 + journal 流:松散尾上限、行序、canonical 字节保真。"""

    def _server(self, bridge, **kwargs):
        return BridgeServer(bridge, **kwargs)

    def test_loose_tail_bounded_and_journal_order(self):
        record_count = 1025
        with tempfile.TemporaryDirectory(
                prefix="bridge_streaming_test.") as bridge:
            server = self._server(bridge)
            for seq in range(record_count):
                path = os.path.join(bridge, "request_{}.json".format(seq))
                with open(path, "w", encoding="utf-8") as target:
                    json.dump(_request(seq), target, separators=(",", ":"))
                server._handle_request(
                    seq,
                    lambda request: _response(request["delivery_sequence"]))
                loose = sum(
                    name.startswith("request_") and name.endswith(".json")
                    for name in os.listdir(bridge))
                # 已处理请求文件流式退役:任意时刻松散尾有界。
                self.assertLessEqual(loose, 255)
            server._close_audit_streams()
            self.assertFalse(
                any(name.startswith("request_") and name.endswith(".json")
                    for name in os.listdir(bridge)))
            records = list(iter_request_records(bridge))
            self.assertEqual([seq for seq, _ in records],
                             list(range(record_count)))
            stats = list(server.per_request_stats())
            self.assertEqual([row["seq"] for row in stats],
                             list(range(record_count)))
            server.discard_per_request_stats()

    def test_canonical_producer_preserves_request_bytes(self):
        # 官方 C++ 生产者路径必须逐字节保留请求字节,只追加 JSONL 分隔符。
        with tempfile.TemporaryDirectory(
                prefix="bridge_canonical_journal_test.") as bridge:
            raw = json.dumps(_request(0), separators=(",", ":"))
            path = os.path.join(bridge, "request_0.json")
            with open(path, "w", encoding="utf-8") as target:
                target.write(raw)
            server = self._server(bridge, canonical_request_producer=True)
            server._handle_request(0, lambda request: _response(0))
            server._close_audit_streams()
            with open(os.path.join(bridge, "request_journal.jsonl"),
                      "r", encoding="utf-8", newline="") as source:
                self.assertEqual(source.read(), raw + "\n")

    def test_non_canonical_input_keeps_normalization(self):
        # 非规范/pretty 夹具输入保留容忍归一化行为(紧凑重序列化),
        # 不被当作可信单行记录。
        with tempfile.TemporaryDirectory(
                prefix="bridge_fallback_journal_test.") as bridge:
            request = _request(0)
            raw = "  " + json.dumps(request, indent=2) + " \r\n"
            path = os.path.join(bridge, "request_0.json")
            with open(path, "w", encoding="utf-8", newline="") as target:
                target.write(raw)
            server = self._server(bridge)
            server._handle_request(0, lambda parsed: _response(0))
            server._close_audit_streams()
            expected = json.dumps(request, separators=(",", ":")) + "\n"
            with open(os.path.join(bridge, "request_journal.jsonl"),
                      "r", encoding="utf-8", newline="") as source:
                self.assertEqual(source.read(), expected)


class JournalGapTest(unittest.TestCase):
    """journal 跳号 fail-closed(iter_request_records)。"""

    def test_journal_delivery_gap_fails_closed(self):
        with tempfile.TemporaryDirectory(
                prefix="bridge_journal_gap_test.") as bridge:
            journal = os.path.join(bridge, "request_journal.jsonl")
            with open(journal, "w", encoding="utf-8") as target:
                target.write(json.dumps(_request(0)) + "\n")
                target.write(json.dumps(_request(2)) + "\n")
            with self.assertRaises(ValueError):
                list(iter_request_records(bridge))


if __name__ == "__main__":
    unittest.main()
