#!/usr/bin/env python3
"""test_decision_bridge_shm.py -- v11 §8 保底：shm 帧传输层 Python 单测。

覆盖（不起仿真、不依赖 C++ 进程；合成帧驱动 transport 层）：
  1. 帧编解码往返：二进制负载（含 NUL、MB 级）逐字节相等；重复 seq
     幂等忽略；空槽 None；未来 seq 原样上交（tracker 语义）。
  2. fail-closed：超槽容量、crc 破坏、世代不符（陈旧段拒绝）、段头
     魔数损坏、指针文件缺失/畸形。
  3. 生产端对等（ensure_bridge_shm）：重复 ensure 幂等；陈旧段按前缀
     清除；指针文件原子写出。
  4. BridgeServer journal 字节保持：_handle_request 收到的帧负载原样
     append 进 request_journal.jsonl（宽容路径 = 负载 + "\\n"）；
     response 帧负载字节 == 旧 _write_json_atomic 的盘上字节
     （separators=(",", ":") 的 dumps 输出）。

运行：cd sh_test_mesh/workload/llama2_7b_inference &&
      python3 -m pytest online/test_decision_bridge_shm.py -q
（或 pytest 同路径；模块自行把 online/ 加进 sys.path）
"""

import json
import os
import struct
import sys
import tempfile
import zlib

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import bridge_shm  # noqa: E402
from bridge_shm import (  # noqa: E402
    ACK_SLOT_BYTES,
    ACK_SLOT_COUNT,
    BridgeShmError,
    DEFAULT_SLOT_CAPACITY,
    SEGMENT_HEADER_BYTES,
    ShmAckRing,
    ShmFrameChannel,
    ensure_bridge_shm,
    load_bridge_shm_info,
)
from decision_bridge import BridgeServer  # noqa: E402


@pytest.fixture()
def bridge_dir(tmp_path):
    root = str(tmp_path / "bridge")
    yield root
    # Test hygiene: reclaim this bridge dir's /dev/shm segments (tmp_path
    # cleans the directory tree, but nothing else unlinks the segments the
    # production C++ bridge destructor would own).
    try:
        bridge_shm.clean_stale_bridge_shm(root)
    except OSError:
        pass


def _binary_payload(size):
    return bytes((i * 13 + (i >> 5)) & 0xFF for i in range(size))


# ------------------------------------------------------------ frame codec --

def test_frame_roundtrip_binary_payload(bridge_dir):
    info = ensure_bridge_shm(bridge_dir, slot_capacity=1 << 22)
    payload = _binary_payload((3 << 20) + 17)  # MB-scale, NULs, high bytes
    writer = ShmFrameChannel.open(
        info["response_segment"], info["generation"], info["slot_capacity"])
    writer.publish(7, payload)
    reader = ShmFrameChannel.open(
        info["response_segment"], info["generation"], info["slot_capacity"])
    frame = reader.read_frame(7)
    assert frame is not None
    seq, got = frame
    assert seq == 7
    assert got == payload
    assert reader.read_frame(8) is None      # duplicate seq ignored
    writer.publish(8, b"next")
    seq, got = reader.read_frame(8)
    assert (seq, got) == (8, b"next")
    writer.close()
    reader.close()


def test_frame_future_seq_surfaces_to_caller(bridge_dir):
    info = ensure_bridge_shm(bridge_dir, slot_capacity=4096)
    channel = ShmFrameChannel.open(
        info["response_segment"], info["generation"], info["slot_capacity"])
    channel.publish(9, b"future")
    # A future seq returns to the caller: the strict-monotonic tracker
    # (BridgeServer._BoundedSequenceTracker) fail-closes it, mirroring the
    # old listdir fallback that surfaced future request_<seq>.json files.
    frame = channel.read_frame(5)
    assert frame is not None and frame[0] == 9 and frame[1] == b"future"
    channel.close()


def test_empty_slot_reads_none(bridge_dir):
    info = ensure_bridge_shm(bridge_dir, slot_capacity=4096)
    channel = ShmFrameChannel.open(
        info["request_segment"], info["generation"], info["slot_capacity"])
    assert channel.read_frame(0) is None
    channel.close()


# ------------------------------------------------------------ fail-closed --

def test_oversize_frame_fails_closed(bridge_dir):
    info = ensure_bridge_shm(bridge_dir, slot_capacity=1024)
    channel = ShmFrameChannel.open(
        info["response_segment"], info["generation"], info["slot_capacity"])
    with pytest.raises(BridgeShmError, match="exceeds"):
        channel.publish(0, b"x" * 1025)
    channel.close()


def test_crc_corruption_fails_closed(bridge_dir):
    info = ensure_bridge_shm(bridge_dir, slot_capacity=4096)
    channel = ShmFrameChannel.open(
        info["response_segment"], info["generation"], info["slot_capacity"])
    channel.publish(3, b"checksum me")
    channel.close()
    # Flip one payload byte behind the channel's back.
    _flip_byte(os.path.join("/dev/shm", info["response_segment"]),
               SEGMENT_HEADER_BYTES + 16 + 2)
    reader = ShmFrameChannel.open(
        info["response_segment"], info["generation"], info["slot_capacity"])
    with pytest.raises(BridgeShmError, match="checksum"):
        reader.read_frame(3)
    reader.close()


def _flip_byte(path, offset):
    import mmap as mmap_mod
    with open(path, "r+b", buffering=0) as handle:
        mm = mmap_mod.mmap(handle.fileno(), os.path.getsize(path))
        mm[offset] = mm[offset] ^ 0xFF
        mm.close()


def test_stale_generation_rejected(bridge_dir):
    info = ensure_bridge_shm(bridge_dir, slot_capacity=4096)
    with pytest.raises(BridgeShmError, match="generation"):
        ShmFrameChannel.open(
            info["response_segment"], info["generation"] + 1,
            info["slot_capacity"])


def test_corrupt_segment_header_rejected(bridge_dir):
    info = ensure_bridge_shm(bridge_dir, slot_capacity=4096)
    _flip_byte(os.path.join("/dev/shm", info["response_segment"]), 0)
    with pytest.raises(BridgeShmError, match="not a bridge shm segment"):
        ShmFrameChannel.open(
            info["response_segment"], info["generation"],
            info["slot_capacity"])


def test_missing_pointer_file_fails_closed(bridge_dir):
    os.makedirs(bridge_dir, exist_ok=True)
    with pytest.raises(BridgeShmError, match="cannot load"):
        load_bridge_shm_info(bridge_dir)


def test_malformed_pointer_file_fails_closed(bridge_dir):
    os.makedirs(bridge_dir, exist_ok=True)
    with open(os.path.join(bridge_dir, bridge_shm.INFO_NAME), "w") as sink:
        sink.write("{not json")
    with pytest.raises(BridgeShmError):
        load_bridge_shm_info(bridge_dir)


# ------------------------------------------------------- producer parity --

def test_ensure_idempotent_and_stale_cleanup(bridge_dir):
    first = ensure_bridge_shm(bridge_dir, slot_capacity=1 << 20)
    second = ensure_bridge_shm(bridge_dir, slot_capacity=1 << 20)
    assert second == first  # adopt path: same generation/names
    # A stale segment from a "crashed previous run" is reclaimed by prefix.
    stem = bridge_shm.bridge_shm_stem(bridge_dir)
    stale = os.path.join("/dev/shm", stem + "req_deadbeefdeadbeef")
    with open(stale, "wb") as sink:
        sink.write(b"\x00" * 128)
    third = ensure_bridge_shm(bridge_dir, slot_capacity=2048)
    assert not os.path.exists(stale)
    assert third["generation"] != first["generation"]
    assert third["slot_capacity"] == 2048
    # pointer file is valid JSON with the frozen key set
    with open(os.path.join(bridge_dir, bridge_shm.INFO_NAME)) as source:
        info = json.load(source)
    assert info["version"] == 1
    assert set(info) == {"version", "generation", "request_segment",
                         "commit_ack_segment", "response_segment",
                         "slot_capacity"}


def test_default_capacity_is_256mib():
    assert DEFAULT_SLOT_CAPACITY == 256 * 1024 * 1024


# --------------------------------------------------------- ack ring tests --

def _ring_name(tag):
    import time
    return "astra_brg_test_ring_{}_{}_{}".format(
        os.getpid(), tag, time.time_ns())


def test_ack_ring_coalesces_back_to_back_frames():
    """ack 流无交换门控:连发(含幂等重发)必须一次 drain 全部按序取回。

    这是单槽设计丢失中间帧的真实场景(旧文件协议一次唤醒可合并交付
    多个 ack;历史 fixture 即 ack0/ack1/ack0-dup 三连发)。
    """
    name = _ring_name("coalesce")
    gen = bridge_shm.new_generation()
    writer = ShmAckRing.create(name, gen)
    ack0 = b'{"schema_version":3,"delivery_sequence":0,"batch_id":0}'
    ack1 = b'{"schema_version":3,"delivery_sequence":1,"batch_id":1}'
    writer.publish(0, ack0)
    writer.publish(1, ack1)
    writer.publish(0, ack0)  # idempotent duplicate, appended again
    reader = ShmAckRing.open(name, gen)
    frames = reader.drain(0)
    assert frames == [(0, ack0), (1, ack1), (0, ack0)]
    # duplicate filtering is the caller's policy (seq < tracker next_seq)
    delivered = [(seq, raw) for seq, raw in frames if seq >= 1]
    assert [seq for seq, _raw in delivered] == [1]
    # tail watermark advanced in the segment for the writer's reuse check
    writer.publish(1, ack1)  # ring not full; append after drain works
    writer.close()
    reader.close()
    bridge_shm.os.unlink("/dev/shm/" + name)


def test_ack_ring_oversize_fails_closed():
    name = _ring_name("oversize")
    gen = bridge_shm.new_generation()
    ring = ShmAckRing.create(name, gen)
    limit = ACK_SLOT_BYTES - bridge_shm._FRAME_HEADER.size - 4
    with pytest.raises(BridgeShmError, match="exceeds the ack slot size"):
        ring.publish(0, b"x" * (limit + 1))
    ring.close()
    bridge_shm.os.unlink("/dev/shm/" + name)


def test_ack_ring_stale_generation_rejected():
    name = _ring_name("stale")
    gen = bridge_shm.new_generation()
    ShmAckRing.create(name, gen).close()
    with pytest.raises(BridgeShmError, match="generation"):
        ShmAckRing.open(name, gen + 1)
    bridge_shm.os.unlink("/dev/shm/" + name)


def test_ack_ring_crc_corruption_fails_closed():
    name = _ring_name("crc")
    gen = bridge_shm.new_generation()
    ring = ShmAckRing.create(name, gen)
    ring.publish(3, b"checksum me")
    ring.close()
    _flip_byte(os.path.join("/dev/shm", name),
               SEGMENT_HEADER_BYTES + bridge_shm._FRAME_HEADER.size + 2)
    reader = ShmAckRing.open(name, gen)
    with pytest.raises(BridgeShmError, match="checksum"):
        reader.drain(0)
    reader.close()
    bridge_shm.os.unlink("/dev/shm/" + name)


# ------------------------------------------- journal / response byte fixes --

def _minimal_request(seq):
    return json.dumps({
        "schema_version": 3,
        "delivery_sequence": seq,
        "tick": 12345 + seq,
        "reasons": ["ARRIVAL"],
        "arrivals": [],
        "completed_nodes": [],
        "completed_groups": [],
        "affected_ranks": [],
        "event_seqs": [],
        "snapshot_handle": {"epoch": 0, "tick": 1, "kind": ""},
        "snapshot": {},
        "ledger_summary": {"injected_unfinished": []},
    }, separators=(",", ":")).encode("utf-8")


def _handler(request):
    return {"kernel_contract": {}}


def test_journal_bytes_preserved(bridge_dir):
    ensure_bridge_shm(bridge_dir, slot_capacity=1 << 20)
    server = BridgeServer(bridge_dir)
    server.open_transport()
    raw0 = _minimal_request(0)
    raw1 = _minimal_request(1)
    server._handle_request(0, _handler, raw0)
    server._handle_request(1, _handler, raw1)
    server._close_audit_streams()
    # Tolerant path: journal line == raw frame payload + "\n" -- the exact
    # bytes the old loose-file producer wrote into request_<seq>.json.
    with open(os.path.join(bridge_dir, "request_journal.jsonl"), "rb") as src:
        journal = src.read()
    assert journal == raw0 + b"\n" + raw1 + b"\n"
    # ... and no loose frame files exist anywhere in the bridge dir (the
    # journal + sidecars + the shm pointer are the only artifacts).
    names = sorted(os.listdir(bridge_dir))
    assert names == sorted([
        ".bridge_processing_stats.partial.jsonl",
        "bridge_shm.json",
        "request_journal.jsonl",
    ]), names


def test_response_frame_bytes_match_legacy_file_format(bridge_dir):
    ensure_bridge_shm(bridge_dir, slot_capacity=1 << 20)
    server = BridgeServer(bridge_dir)
    server.open_transport()
    response = {"kernel_contract": {},
                "nodes": [{"id": 0}], "中文": "值"}
    try:
        server._handle_request(0, lambda _request: response,
                               _minimal_request(0))
    finally:
        server._close_audit_streams()
    info = load_bridge_shm_info(bridge_dir)
    reader = ShmFrameChannel.open(
        info["response_segment"], info["generation"], info["slot_capacity"])
    try:
        frame = reader.read_frame(0)
    finally:
        reader.close()
    assert frame is not None
    seq, payload = frame
    assert seq == 0
    # C3/A1 regression pin: the frame payload is the compact-separator
    # UTF-8 form (json.loads -> compact re-dump reproduces the payload
    # byte-for-byte; any spaced/newline form would not round-trip), and
    # the handler contract's default fields are present.
    assert payload == json.dumps(
        json.loads(payload), separators=(",", ":")).encode("utf-8")
    decoded = json.loads(payload)
    assert decoded["schema_version"] == 3
    assert decoded["batch_id"] == 0
    assert decoded["source_delivery_sequence"] == 0
    for field in ("nodes", "parent_edges", "watches", "assignments",
                  "kv_actions"):
        assert isinstance(decoded[field], list)
    assert decoded["nodes"] == [{"id": 0}] and decoded["中文"] == "值"
    stats = server.stats()
    assert stats["forced_flush_count"] == 1
    assert stats["channel_bytes"] == len(_minimal_request(0)) + len(payload)


def test_stats_channel_bytes_counts_frame_payloads(bridge_dir):
    ensure_bridge_shm(bridge_dir, slot_capacity=1 << 20)
    server = BridgeServer(bridge_dir)
    server.open_transport()
    raw = _minimal_request(0)
    ack_raw = b'{"schema_version":3}'
    server._handle_request(0, _handler, raw)
    server._handle_ack(0, lambda _ack: None, ack_raw)
    server._close_audit_streams()
    stats = server.stats()
    # request + response + ack payload bytes, Python view; the response
    # length is read back from the published frame itself.
    info = load_bridge_shm_info(bridge_dir)
    reader = ShmFrameChannel.open(
        info["response_segment"], info["generation"], info["slot_capacity"])
    try:
        _seq, resp_payload = reader.read_frame(0)
    finally:
        reader.close()
    assert stats["channel_bytes"] == (
        len(raw) + len(resp_payload) + len(ack_raw))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
