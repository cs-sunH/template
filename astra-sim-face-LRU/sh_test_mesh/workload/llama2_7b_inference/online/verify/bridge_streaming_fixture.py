#!/usr/bin/env python3
"""Regression fixture for bounded bridge bookkeeping and request journaling."""

import json
import os
import sys
import tempfile

_ONLINE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ONLINE_DIR not in sys.path:
    sys.path.insert(0, _ONLINE_DIR)

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


def main():
    tracker = _BoundedSequenceTracker("fixture")
    assert tracker.mark(0)
    assert tracker.contains(0)
    assert not tracker.mark(0)
    try:
        tracker.validate_new(2)
    except BridgeError:
        pass
    else:
        raise AssertionError("future seq must fail before missing seq 1")
    assert tracker.mark(1)
    for seq in range(2, 100_000):
        assert tracker.mark(seq)
    assert tracker.next_seq == 100_000
    assert set(vars(tracker)) == {"label", "next_seq"}

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
            assert loose <= 255
        server._close_audit_streams()
        assert not any(name.startswith("request_") and name.endswith(".json")
                       for name in os.listdir(bridge))
        records = list(iter_request_records(bridge))
        assert [seq for seq, _ in records] == list(range(record_count))
        stats = list(server.per_request_stats())
        assert [row["seq"] for row in stats] == list(range(record_count))
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
            assert source.read() == raw + "\n"

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
            assert source.read() == expected

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

    print("bridge streaming fixture PASS: {} journal rows, bounded loose tail"
          .format(record_count))
    return 0


if __name__ == "__main__":
    sys.exit(main())
