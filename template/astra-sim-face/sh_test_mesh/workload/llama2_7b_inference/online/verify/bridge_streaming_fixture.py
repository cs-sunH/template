#!/usr/bin/env python3
"""Regression fixture for bounded bridge bookkeeping and request journaling."""

import json
import os
import sys
import tempfile
import threading

_ONLINE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ONLINE_DIR not in sys.path:
    sys.path.insert(0, _ONLINE_DIR)

from bridge_request_journal import iter_request_records  # noqa: E402
from decision_bridge import (  # noqa: E402
    BridgeError,
    BridgeServer,
    _BoundedSequenceTracker,
)


def _require(condition, message):
    """显式失败(不用裸 assert:python -O 下会被剥掉,校验静默通过)。"""
    if not condition:
        raise RuntimeError("bridge_streaming_fixture: {}".format(message))


def _request(seq):
    return {"schema_version": 1, "delivery_sequence": seq}


def _response(seq):
    return {"batch_id": seq, "source_delivery_sequence": seq}


def main():
    tracker = _BoundedSequenceTracker("fixture")
    _require(tracker.mark(0), "first mark(0) must succeed")
    _require(tracker.contains(0), "seq 0 must be contained after mark")
    _require(not tracker.mark(0), "duplicate mark(0) must be rejected")
    try:
        tracker.validate_new(2)
    except BridgeError:
        pass
    else:
        raise RuntimeError("future seq must fail before missing seq 1")
    _require(tracker.mark(1), "mark(1) after gap fill must succeed")
    for seq in range(2, 100_000):
        _require(tracker.mark(seq), "mark({}) must succeed".format(seq))
    _require(tracker.next_seq == 100_000,
             "next_seq must advance to 100000, got {}".format(tracker.next_seq))
    _require(set(vars(tracker)) == {"label", "next_seq"},
             "tracker must stay bounded to {{label, next_seq}}, got {}".format(
                 sorted(vars(tracker))))

    record_count = 1025
    with tempfile.TemporaryDirectory(prefix="bridge_streaming_fixture.") as bridge:
        server = BridgeServer(bridge)
        for seq in range(record_count):
            path = os.path.join(bridge, "request_{}.json".format(seq))
            with open(path, "w", encoding="utf-8") as target:
                json.dump(_request(seq), target, separators=(",", ":"))
            server._handle_request(seq, lambda request: _response(
                request["delivery_sequence"]))
            loose = sum(name.startswith("request_") and name.endswith(".json")
                        for name in os.listdir(bridge))
            _require(loose <= 255,
                     "loose request files {} exceed 255 at seq {}".format(
                         loose, seq))
        server._close_audit_streams()
        _require(not any(name.startswith("request_")
                         and name.endswith(".json")
                         for name in os.listdir(bridge)),
                 "loose request files must be gone after audit close")
        records = list(iter_request_records(bridge))
        _require([seq for seq, _ in records] == list(range(record_count)),
                 "journal rows must be 0..{}".format(record_count - 1))
        stats = list(server.per_request_stats())
        _require([row["seq"] for row in stats] == list(range(record_count)),
                 "per-request stats must cover 0..{}".format(record_count - 1))
        server.discard_per_request_stats()

    # The official C++ producer path must preserve its canonical request
    # bytes exactly and add only the JSONL record delimiter.
    with tempfile.TemporaryDirectory(
            prefix="bridge_canonical_journal_fixture.") as bridge:
        raw = json.dumps(_request(0), separators=(",", ":"))
        path = os.path.join(bridge, "request_0.json")
        with open(path, "w", encoding="utf-8") as target:
            target.write(raw)
        server = BridgeServer(bridge, canonical_request_producer=True)
        server._handle_request(0, lambda request: _response(0))
        server._close_audit_streams()
        with open(os.path.join(bridge, "request_journal.jsonl"),
                  "r", encoding="utf-8", newline="") as source:
            _require(source.read() == raw + "\n",
                     "canonical producer must preserve request bytes")

    # Non-canonical/pretty fixture input keeps the legacy normalization
    # behavior instead of being trusted as a one-line record.
    with tempfile.TemporaryDirectory(
            prefix="bridge_fallback_journal_fixture.") as bridge:
        request = _request(0)
        raw = "  " + json.dumps(request, indent=2) + " \r\n"
        path = os.path.join(bridge, "request_0.json")
        with open(path, "w", encoding="utf-8", newline="") as target:
            target.write(raw)
        server = BridgeServer(bridge)
        server._handle_request(0, lambda parsed: _response(0))
        server._close_audit_streams()
        expected = json.dumps(request, separators=(",", ":")) + "\n"
        with open(os.path.join(bridge, "request_journal.jsonl"),
                  "r", encoding="utf-8", newline="") as source:
            _require(source.read() == expected,
                     "non-canonical input must be normalized to one-line")

    with tempfile.TemporaryDirectory(prefix="bridge_journal_gap_fixture.") as bridge:
        journal = os.path.join(bridge, "request_journal.jsonl")
        with open(journal, "w", encoding="utf-8") as target:
            target.write(json.dumps(_request(0)) + "\n")
            target.write(json.dumps(_request(2)) + "\n")
        try:
            list(iter_request_records(bridge))
        except ValueError:
            pass
        else:
            raise AssertionError("journal delivery gap must fail closed")

    # B13(2026-09-28)钉住:on_commit_ack 的协议违例(普通 ValueError,账本
    # finalize 挂接点的真实异常形态)不得穿透 serve_forever——ack 循环须与
    # request 循环同型兜底:写 error response 到当前在途 request 序号
    # (_request_sequences.next_seq,A2-1 口径,非 ack 序号)+ resp_notify
    # 通知 + sys.exit(1)。假 C++ 用线程复刻 fd 语义(同
    # bridge_cpp_death_fixture 的通道操作序列,单交付即可达目标路径)。
    with tempfile.TemporaryDirectory(
            prefix="bridge_ack_blast_fixture.") as bridge:
        for name in ("req_notify.fifo", "resp_notify.fifo"):
            os.mkfifo(os.path.join(bridge, name), 0o600)
        with open(os.path.join(bridge, "commit_ack_0.json"),
                  "w", encoding="utf-8") as target:
            json.dump({"batch_id": 0}, target)

        def fake_cpp():
            wfd = os.open(os.path.join(bridge, "req_notify.fifo"),
                          os.O_WRONLY)  # 阻塞至 serve 侧读端就绪
            os.write(wfd, b"\n")
            rfd = os.open(os.path.join(bridge, "resp_notify.fifo"),
                          os.O_RDONLY)
            notify = os.read(rfd, 1)  # 阻塞至 fail-closed 通知字节
            _require(notify == b"\n", "ack blast: expected resp notify byte")
            os.close(wfd)
            os.close(rfd)

        peer = threading.Thread(target=fake_cpp, daemon=True)
        peer.start()

        def bad_ack(_ack):
            raise ValueError("ledger protocol violation")

        server = BridgeServer(bridge)
        exit_codes = []
        try:
            server.serve_forever(lambda request: _response(0), bad_ack)
        except SystemExit as exc:
            exit_codes.append(exc.code)
        peer.join(timeout=10.0)
        _require(not peer.is_alive(), "ack blast: fake cpp peer hung")
        _require(exit_codes == [1],
                 "ack loop must fail-closed with exit 1, got {!r}".format(
                     exit_codes))
        with open(os.path.join(bridge, "response_0.json"),
                  "r", encoding="utf-8") as source:
            error = json.load(source).get("error", "")
        _require("ValueError" in error
                 and "ledger protocol violation" in error,
                 "ack failure must write error response at the in-flight "
                 "request seq, got {!r}".format(error))

    print("bridge streaming fixture PASS: {} journal rows, bounded loose tail"
          .format(record_count))
    return 0


if __name__ == "__main__":
    sys.exit(main())
