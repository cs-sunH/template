#!/usr/bin/env python3
"""Stream readers for the bridge request audit journal.

Successful online runs keep one ``request_journal.jsonl`` instead of one
``request_<seq>.json`` inode per delivery.  Readers also accept the legacy
loose-file layout and merge the bounded loose tail left by an interrupted run.

W5 (2026-09-29, journal replay fidelity): the official producer's journal
records carry ``work_ledger_journal_stripped`` and the stripped
``work_ledger`` payloads live in the gzipped sidecar
``request_journal_work_ledger.jsonl.gz`` (one row per delivery, written by
the bridge at journal time).  ``iter_request_records(bridge_dir,
attach_work_ledger=True)`` merges the sidecar payloads back onto the journal
records so a prefix replay rebuilds exactly the ledger state of the original
run (fail-closed on a stripped row without a sidecar payload).
"""

import gzip
import json
import os


REQUEST_JOURNAL_NAME = "request_journal.jsonl"
WORK_LEDGER_REPLAY_SIDECAR_NAME = "request_journal_work_ledger.jsonl.gz"
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
        if body.isdigit():
            seqs.append(int(body))
    return sorted(seqs)


def _work_ledger_sidecar_path(bridge_dir):
    """Return the live or runner-archived sidecar path, or ``None``."""
    candidates = [os.path.join(
        bridge_dir, WORK_LEDGER_REPLAY_SIDECAR_NAME)]
    run_dir = os.path.dirname(os.path.abspath(bridge_dir))
    candidates.append(os.path.join(
        run_dir, "results", WORK_LEDGER_REPLAY_SIDECAR_NAME))
    for path in candidates:
        if os.path.isfile(path):
            return path
    return None


def iter_work_ledger_payloads(bridge_dir):
    """Yield ``(delivery_sequence, work_ledger_payload)`` from the sidecar.

    Rows are yielded in file order; non-increasing sequences fail closed. The
    reader retains only the current row so journal replay can merge the two
    ordered streams without an all-history payload dictionary.
    """
    path = _work_ledger_sidecar_path(bridge_dir)
    if path is None:
        return
    previous_seq = -1
    try:
        with gzip.open(path, "rt", encoding="utf-8") as source:
            for line_number, line in enumerate(source, 1):
                if not line.strip():
                    raise ValueError(
                        "blank work_ledger sidecar row at {}:{}".format(
                            path, line_number))
                try:
                    row = json.loads(line)
                except ValueError as exc:
                    raise ValueError(
                        "invalid work_ledger sidecar JSON at {}:{}: {}"
                        .format(path, line_number, exc)) from exc
                if not isinstance(row, dict):
                    raise ValueError(
                        "work_ledger sidecar row at {}:{} is not an object"
                        .format(path, line_number))
                seq = row.get("delivery_sequence")
                payload = row.get("work_ledger")
                if (isinstance(seq, bool) or not isinstance(seq, int)
                        or seq < 0 or not isinstance(payload, dict)):
                    raise ValueError(
                        "malformed work_ledger sidecar row at {}:{} (need "
                        "integer delivery_sequence and object work_ledger)"
                        .format(path, line_number))
                if seq <= previous_seq:
                    raise ValueError(
                        "work_ledger sidecar sequence {} is not greater "
                        "than previous sequence {} at {}:{}".format(
                            seq, previous_seq, path, line_number))
                previous_seq = seq
                yield seq, payload
    except (EOFError, gzip.BadGzipFile, OSError) as exc:
        raise ValueError(
            "invalid or truncated work_ledger sidecar {}: {}".format(
                path, exc)) from exc


def iter_request_records(bridge_dir, *, attach_work_ledger=False):
    """Yield ``(delivery_sequence, request)`` in sequence order.

    The journal is authoritative.  Loose files whose sequence is newer than
    its last row are appended as an interrupted-run tail; duplicate loose
    files are ignored without constructing an all-history ``set``.

    ``attach_work_ledger=True`` (W5 replay fidelity) re-attaches the
    ``work_ledger`` field onto journal records that carry the
    ``work_ledger_journal_stripped`` marker, using the gzipped sidecar as the
    payload source; a stripped record without a matching sidecar payload
    fails closed (replay must not silently degrade to an empty ledger).
    Loose tail files still carry the original field and pass through
    unchanged; records that already have ``work_ledger`` are left untouched
    (legacy journals written before the strip).  Records with neither the
    field nor the marker are legacy-peer requests (the absent field is the
    native shape) and pass through unchanged.
    """
    payloads = iter_work_ledger_payloads(bridge_dir) if attach_work_ledger else None
    end_of_payloads = object()
    payload_row = (next(payloads, end_of_payloads)
                   if payloads is not None else end_of_payloads)

    def _attach(seq, request):
        nonlocal payload_row
        if not attach_work_ledger:
            return request
        while payload_row is not end_of_payloads and payload_row[0] < seq:
            raise ValueError(
                "work_ledger sidecar payload {} has no matching stripped "
                "journal record".format(payload_row[0]))
        has_marker = "work_ledger_journal_stripped" in request
        if payload_row is not end_of_payloads and payload_row[0] == seq:
            if not has_marker or "work_ledger" in request:
                raise ValueError(
                    "work_ledger sidecar payload {} has no unique stripped "
                    "journal record".format(seq))
            payload = payload_row[1]
            payload_row = next(payloads, end_of_payloads)
            merged = dict(request)
            merged["work_ledger"] = payload
            return merged
        if not has_marker:
            return request
        if payload_row is end_of_payloads or payload_row[0] != seq:
            raise ValueError(
                "journal record {} carries work_ledger_journal_stripped but "
                "the replay sidecar has no payload for it (replay would "
                "degrade to an empty ledger; refusing)".format(seq))
        raise AssertionError("unreachable sidecar merge state")

    last_seq = -1
    path = _journal_path(bridge_dir)
    if path is not None:
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
                yield seq, _attach(seq, request)

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
        yield seq, _attach(seq, request)

    if (attach_work_ledger and payload_row is not end_of_payloads):
        raise ValueError(
            "work_ledger sidecar payload {} has no matching journal record"
            .format(payload_row[0]))
