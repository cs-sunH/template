#!/usr/bin/env python3
"""Stream readers for the bridge request audit journal.

Successful online runs keep one ``request_journal.jsonl`` instead of one
``request_<seq>.json`` inode per delivery.  Readers also accept the legacy
loose-file layout and merge the bounded loose tail left by an interrupted run.
"""

import json
import os


REQUEST_JOURNAL_NAME = "request_journal.jsonl"
_REQUEST_PREFIX = "request_"
_JSON_SUFFIX = ".json"


def _journal_path(bridge_dir):
    """Return the live or runner-archived journal path, or ``None``."""
    candidates = [os.path.join(bridge_dir, REQUEST_JOURNAL_NAME)]
    run_dir = os.path.dirname(os.path.abspath(bridge_dir))
    candidates.append(os.path.join(run_dir, "results", REQUEST_JOURNAL_NAME))
    for path in candidates:
        if os.path.isfile(path):
            return path
    return None


def _decode_record(raw, source, line_number=None):
    try:
        request = json.loads(raw)
    except (TypeError, ValueError) as exc:
        where = source if line_number is None else "{}:{}".format(
            source, line_number)
        raise ValueError("invalid bridge request JSON at {}: {}".format(
            where, exc)) from exc
    if not isinstance(request, dict):
        raise ValueError("bridge request record in {} is not an object".format(
            source))
    seq = request.get("delivery_sequence")
    if isinstance(seq, bool) or not isinstance(seq, int) or seq < 0:
        raise ValueError(
            "bridge request record in {} has invalid delivery_sequence {!r}"
            .format(source, seq))
    return seq, request


def _loose_request_sequences(bridge_dir):
    seqs = []
    try:
        names = os.listdir(bridge_dir)
    except FileNotFoundError:
        return seqs
    for name in names:
        if not (name.startswith(_REQUEST_PREFIX)
                and name.endswith(_JSON_SUFFIX)):
            continue
        body = name[len(_REQUEST_PREFIX):-len(_JSON_SUFFIX)]
        # ASCII 数字严格校验(与 decision_bridge._seq_of 同款):str.isdigit()
        # 放行 Unicode 数字——全角序号经 int() 归一化与 ASCII 同号碰撞
        # (静默跳过)、上标数字直接 int() 崩溃。散装文件名只会由本侧与
        # C++ 以 ASCII 十进制写出。
        if body.isascii() and body.isdigit():
            seqs.append(int(body))
    return sorted(seqs)


def iter_request_records(bridge_dir):
    """Yield ``(delivery_sequence, request)`` in sequence order.

    The journal is authoritative.  Loose files whose sequence is newer than
    its last row are appended as an interrupted-run tail; duplicate loose
    files are ignored without constructing an all-history ``set``.
    """
    last_seq = -1
    path = _journal_path(bridge_dir)
    if path is not None:
        # 末行 torn write(断电/kill 落在行中间)在此整份 fail-closed 拒用
        # (下方 _decode_record 抛 ValueError)。评估结论=维持现状:盘上
        # 未 unlink 的散装 request 尾部(flush 批窗口 ≤256 行)理论上可
        # 拼回,但散装冗余不含 journal 已核销行的顺序证据,自动容错拼回
        # 会把"审计流末行写了一半"静默升格为"改写后的审计流"——方向
        # 更差。窗口极窄(fsync 后 unlink 前),fail-closed 拒用 + 整跑
        # 重跑是更廉价的诚实处置。
        with open(path, "r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, 1):
                if not line.strip():
                    raise ValueError("blank bridge request journal row at {}:{}"
                                     .format(path, line_number))
                seq, request = _decode_record(line, path, line_number)
                expected_seq = last_seq + 1
                if seq != expected_seq:
                    raise ValueError(
                        "bridge request journal is not contiguous: got {} "
                        "after {}, expected {} at {}:{}".format(
                            seq, last_seq, expected_seq, path, line_number))
                last_seq = seq
                yield seq, request

    for seq in _loose_request_sequences(bridge_dir):
        if seq <= last_seq:
            continue
        expected_seq = last_seq + 1
        if seq != expected_seq:
            raise ValueError(
                "bridge request loose tail is not contiguous: got {}, "
                "expected {} in {}".format(seq, expected_seq, bridge_dir))
        loose_path = os.path.join(
            bridge_dir, _REQUEST_PREFIX + str(seq) + _JSON_SUFFIX)
        with open(loose_path, "r", encoding="utf-8") as source:
            record_seq, request = _decode_record(source.read(), loose_path)
        if record_seq != seq:
            raise ValueError(
                "bridge request filename seq {} != payload seq {} ({})".format(
                    seq, record_seq, loose_path))
        last_seq = seq
        yield seq, request
