#!/usr/bin/env python3
"""Materialize the astra_compute_20.csv first-30-seconds simulation input
(folded recompute single-queue form) for the sh_2.0 repository.

2026-08-21 (KV 账本统一维护改造): the sidecar_restore double-file form was
removed repo-wide; the turn-0 prefix is now FOLDED into turn-0
prefill_length (full recompute caliber), the request-context sidecar
product is gone, and the canonical digest's prefix_mode reads "recompute".
Session KV history size/location is maintained dynamically by the runtime
KV ledger (KVCacheManager) from turn-0 prefill accounting onward.

Simulation-input constraint (user directive 2026-08-15): ONLY
`agent-traces/tracelab/astra_compute_20.csv` rows with
`arrival_time < 30,000,000,000 ns` are permitted. No other csv, no longer
window.

Derivation rules (frozen 3-minute lineage, template_back/history/20_3mins;
single pass, folded queue + canonical digest from one traversal):
  1. window: a session is included iff its first row's arrival_time < 30e9;
     turn arrival = previous arrival + previous gap (gap = human_time if
     non-empty else tool_time if non-empty else 0); a session's later turns
     are truncated once the derived arrival >= 30e9.
  2. queue turn-0 rows FOLD the source prefix into prefill_length
     (prefill_length = source prefix_len + source prefill_length, full
     recompute); turn>=1 rows keep source prefill_length (NEW tokens only).
     Prefix must not be double counted. No request-context sidecar is
     produced: turn-0 recompute accounts the full prefix into the local
     HBM KV ledger at prefill completion, and later turns resolve history
     from the ledger (LOCAL_HIT / migrate / restore / recompute).
     NOTE: this repo's trace_config.csv is the request-neutral
     placeholder -- after materializing, the operator must point the
     config row request_queue_csv at the queue file above.
  3. session_arrival_time_ns uses the source absolute time (turn-0 only).
  4. interval semantics = per-row human_time||tool_time gap.
  5. next_trigger_type (added 2026-08-18, typed KV eviction): per-row
     classification of the return path AFTER this row's request completes,
     derived from the row's own human_time/tool_time -- "human" if
     human_time is non-empty, "tool" if tool_time is non-empty; boundary
     rulings (user, 2026-08-18): a session's final source row (both fields
     empty, no successor) classifies "human"; a non-final row with both
     fields empty (0-interval successor) classifies "tool". The column is
     appended after inter_request_interval_ns; the C++ WindowedTraceReader
     parses only the first 7 columns by position, so it is transparent to
     the simulator.

CLI (WP2, SLO B1, 2026-08-26): materialize_first_30s.py [source] [queue_out]
[window_ns] [arrival_scale] -- all optional; the defaults reproduce the
frozen 30 s / unscaled behavior byte-for-byte (native fixed output names in
this directory).  window_ns parameterizes the [0, window) cut; the queue
description column stays the frozen "first-30-seconds" string so products
of the same window stay byte-identical to the B0 lineage (callers rename
per-window copies).  arrival_scale (float > 0, default 1.0) divides ONLY
each session's turn-0 session_arrival_time_ns (nearest ns, then quantized
onto the 1000 ns grid -- nearest -- per B3-6, so the scaled t0 keeps the
whole-microsecond invariant the unscaled lineage has); intervals are
untouched and window membership is still decided on the SOURCE timeline, so
scaling never adds or drops requests; scale=1.0 is byte-identical.

The canonical digest sidecar gains appended audit columns
(human_time_ns / tool_time_ns / request_type / decode_length) carrying the
source row's own trigger fields.  request_type follows the SLO B1 rule
(next_trigger_type keeps its own 2026-08-18 boundary rulings): human_time
non-empty -> "human"; tool_time non-empty -> "tool"; both empty ->
turn_index 0 "human", otherwise "unknown" (count printed).

Runtime footprint: the source is scanned once with only the current session
buffered, and queue/digest rows are written immediately.  Peak memory is thus
bounded by the largest session rather than the 512 MB source expanded into
Python dictionaries.  The existing flock remains to serialize writers that
target the same native output names.
"""

import csv
import fcntl
import hashlib
import json
import math
import os
import sys

SOURCE_DEFAULT = "/home/sunhao/wsc-simulator/agent-traces/tracelab/astra_compute_20.csv"
WINDOW_NS = 30_000_000_000
HEAVY_LOCK = "/tmp/slo_wps/locks/heavy_mat.lock"
HERE = os.path.dirname(os.path.abspath(__file__))

QUEUE_HEADER = [
    "session_id", "turn_index", "request_id", "prefill_length",
    "decode_length", "session_arrival_time_ns", "inter_request_interval_ns",
    "next_trigger_type",
    "description",
]
DIGEST_HEADER = [
    "request_id", "raw_prefix_tokens", "raw_new_prefill_tokens",
    "effective_prompt_tokens", "digest", "prefix_mode",
    # WP2 (SLO B1, 2026-08-26): appended audit columns for the manifest
    # enrichment (plan_materializer.py) and request_metrics.csv.  Appended
    # after the frozen six so pre-B1 sidecars stay prefix-compatible.
    "human_time_ns", "tool_time_ns", "request_type", "decode_length",
]
QUEUE_DESCRIPTION = (
    "compute_20 first-30-seconds window; turn-0 prefix folded "
    "into prefill_length (recompute caliber)"
)


def _gap_ns(human_time, tool_time):
    h = (human_time or "").strip()
    t = (tool_time or "").strip()
    raw = h if h else (t if t else "")
    return int(raw) if raw else 0


def _next_trigger_type(human_time, tool_time, is_last_row):
    """Trigger type of the request that follows this row (typed eviction).

    human_time non-empty -> "human" (next request comes from a human reply);
    tool_time non-empty -> "tool" (tool-call return). Boundary rulings
    (user, 2026-08-18): the session's final source row (no successor)
    classifies "human"; a non-final row with both fields empty has a
    0-interval successor and classifies "tool".
    """
    h = (human_time or "").strip()
    t = (tool_time or "").strip()
    if h:
        return "human"
    if t:
        return "tool"
    return "human" if is_last_row else "tool"


def _request_type(human_text, tool_text, turn):
    """SLO B1 request_type rule (distinct from next_trigger_type)."""
    if human_text:
        return "human", False
    if tool_text:
        return "tool", False
    if turn == 0:
        return "human", False
    return "unknown", True


def _scaled_t0(arrival_ns, scale):
    # B3-6 (SLO B3, 2026-08-27): quantize the scaled t0 onto the 1000 ns
    # grid (nearest).  Pure division can emit half-us values, which breaks
    # the downstream whole-microsecond invariants (graph_batch_builder
    # timer_gate requires duration % 1000 == 0; source arrivals themselves
    # sit on the 1000 ns grid, matching the S1 derive-script caliber).
    # scale == 1.0 passes the value through untouched (byte-identical
    # contract; for on-grid sources the quantization would be identity
    # anyway -- pass-through keeps non-grid inputs unmodified).
    if scale == 1.0:
        return arrival_ns
    scaled = max(0, int(round(arrival_ns / scale)))
    return int(round(scaled / 1000.0)) * 1000


def _acquire_heavy_lock():
    """Serialize materializers that may target the same native outputs."""
    try:
        os.makedirs(os.path.dirname(HEAVY_LOCK), exist_ok=True)
        handle = open(HEAVY_LOCK, "a")
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        return handle
    except OSError as exc:
        print("warning: %s unavailable (%s); running unlocked"
              % (HEAVY_LOCK, exc), file=sys.stderr)
        return None


def _digest_path_for(queue_path):
    """Canonical digest sidecar next to the queue output (native naming)."""
    base = os.path.basename(queue_path)
    if base.endswith("_request_queue.csv"):
        sibling = base[: -len("_request_queue.csv")] + "_canonical_digest.csv"
    else:
        sibling = base + ".canonical_digest.csv"
    return os.path.join(os.path.dirname(os.path.abspath(queue_path)), sibling)


def _iter_sessions(reader):
    """Yield one contiguous source session at a time (source-order contract)."""
    current_sid = None
    current_rows = []
    for row in reader:
        sid = "session_%s" % row["session_id"]
        if current_sid is None:
            current_sid = sid
        elif sid != current_sid:
            yield current_sid, current_rows
            current_sid = sid
            current_rows = []
        current_rows.append(row)
    if current_sid is not None:
        yield current_sid, current_rows


def _md5_file(path):
    digest = hashlib.md5()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fnv1a64_cont(h: int, data: bytes) -> int:
    """FNV-1a 64 continuation over one chunk of raw bytes."""
    for byte in data:
        h = ((h ^ byte) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return h


def queue_provenance(queue_path: str, generator_version: str) -> dict:
    """Provenance record of a materialized queue, computed from the WRITTEN
    file bytes (exactly what the C++ reader re-derives and compares).

    Definitions mirror the reader's index pass byte-for-byte: lines split on
    b"\n" (a trailing "\r" from the CSV writer stays on the line and rides
    on the unparsed description column, exactly as in C++); the first
    non-empty line is the header; every later non-empty line is a data row;
    turn-0 = non-empty session_arrival_time_ns (6th comma field); adjacent
    inversions are counted over turn-0 arrivals in file order; session
    blocks must be contiguous (a session id may never reappear).
    """
    sha = hashlib.sha256()
    fnv = 14695981039346656037
    csv_bytes = 0
    data_rows = 0
    turn0_count = 0
    turn0_min = None
    turn0_max = None
    inversions = 0
    prev_turn0 = None
    header_seen = False
    current_session = b""
    seen_sessions = set()
    contiguous = True
    pending = b""

    def consume(line: bytes) -> None:
        nonlocal data_rows, turn0_count, turn0_min, turn0_max, inversions
        nonlocal prev_turn0, header_seen, current_session, contiguous
        if not line:
            return
        if not header_seen:
            header_seen = True
            return
        data_rows += 1
        fields = line.split(b",")
        session = fields[0]
        arrival_text = fields[5] if len(fields) > 5 else b""
        if session != current_session:
            if session in seen_sessions:
                contiguous = False
            seen_sessions.add(session)
            current_session = session
        if arrival_text:
            turn0_count += 1
            arrival = int(arrival_text)
            turn0_min = arrival if turn0_min is None else min(turn0_min, arrival)
            turn0_max = arrival if turn0_max is None else max(turn0_max, arrival)
            if prev_turn0 is not None and arrival < prev_turn0:
                inversions += 1
            prev_turn0 = arrival

    with open(queue_path, "rb") as fin:
        while True:
            chunk = fin.read(1 << 20)
            if not chunk:
                break
            csv_bytes += len(chunk)
            sha.update(chunk)
            fnv = fnv1a64_cont(fnv, chunk)
            pending += chunk
            *complete, pending = pending.split(b"\n")
            for line in complete:
                consume(line)
        if pending:
            consume(pending)
    return {
        "schema": 1,
        "generator_version": generator_version,
        "csv_sha256": sha.hexdigest(),
        "csv_fnv1a64": fnv,
        "csv_bytes": csv_bytes,
        "data_rows": data_rows,
        "sessions": len(seen_sessions),
        "turn0_count": turn0_count,
        "turn0_arrival_min_ns": turn0_min if turn0_min is not None else 0,
        "turn0_arrival_max_ns": turn0_max if turn0_max is not None else 0,
        "turn0_adjacent_inversions": inversions,
        "session_blocks_contiguous": contiguous,
    }


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else SOURCE_DEFAULT
    out_queue = (
        sys.argv[2] if len(sys.argv) > 2
        else os.path.join(HERE, "astra_compute_20_first_30_seconds_request_queue.csv")
    )
    window_ns = int(sys.argv[3]) if len(sys.argv) > 3 else WINDOW_NS
    arrival_scale = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
    if not (arrival_scale > 0.0 and math.isfinite(arrival_scale)):
        raise SystemExit(
            "arrival_scale must be a finite float > 0, got %r" % (arrival_scale,))
    if window_ns <= 0:
        raise SystemExit("window_ns must be positive, got %r" % (window_ns,))
    out_digest = _digest_path_for(out_queue)
    description = QUEUE_DESCRIPTION
    if arrival_scale != 1.0:
        description += "; session t0 scaled by 1/%g" % arrival_scale

    lock_handle = _acquire_heavy_lock()
    try:
        session_count = 0
        request_count = 0
        decode_length_sum = 0
        turn0_prefix_rows = 0
        max_total = 0
        unknown_type_rows = 0
        with open(src, newline="") as source, \
                open(out_queue, "w", newline="") as queue_output, \
                open(out_digest, "w", newline="") as digest_output:
            queue_writer = csv.writer(queue_output)
            digest_writer = csv.writer(digest_output)
            queue_writer.writerow(QUEUE_HEADER)
            digest_writer.writerow(DIGEST_HEADER)
            for sid, srows in _iter_sessions(csv.DictReader(source)):
                first_arrival = int(srows[0]["arrival_time"])
                if first_arrival >= window_ns:
                    continue
                session_count += 1
                arrival = first_arrival
                for turn, r in enumerate(srows):
                    if turn > 0:
                        prev = srows[turn - 1]
                        arrival += _gap_ns(prev["human_time"], prev["tool_time"])
                        if arrival >= window_ns:
                            break
                    rid = "%s_request_%d" % (sid, turn)
                    new_prefill = int(r["prefill_length"])
                    prefix = int(r["prefix_len"])
                    prefill = new_prefill + prefix if turn == 0 else new_prefill
                    total = prefix + new_prefill
                    if turn == 0 and prefix > 0:
                        turn0_prefix_rows += 1
                    max_total = max(max_total, total)
                    if turn == 0:
                        session_ns = str(_scaled_t0(arrival, arrival_scale))
                        interval_ns = ""
                    else:
                        session_ns = ""
                        interval_ns = str(_gap_ns(
                            srows[turn - 1]["human_time"],
                            srows[turn - 1]["tool_time"]))
                    trigger_type = _next_trigger_type(
                        r["human_time"], r["tool_time"],
                        is_last_row=(turn == len(srows) - 1),
                    )
                    human_text = (r["human_time"] or "").strip()
                    tool_text = (r["tool_time"] or "").strip()
                    request_type, type_unknown = _request_type(
                        human_text, tool_text, turn)
                    if type_unknown:
                        unknown_type_rows += 1
                    queue_writer.writerow([
                        sid, turn, rid, prefill, r["decode_length"], session_ns,
                        interval_ns, trigger_type, description,
                    ])
                    digest = hashlib.sha256(
                        ("%s|%d|%d|%d" %
                         (rid, prefix, new_prefill, total)).encode()
                    ).hexdigest()
                    digest_writer.writerow([
                        rid, prefix, new_prefill, total, digest, "recompute",
                        human_text, tool_text, request_type, r["decode_length"],
                    ])
                    request_count += 1
                    decode_length_sum += int(r["decode_length"])
    finally:
        if lock_handle is not None:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
            lock_handle.close()

    src_md5 = _md5_file(src)
    queue_md5 = _md5_file(out_queue)
    digest_md5 = _md5_file(out_digest)
    print("source_md5=%s" % src_md5)
    # Frozen references registered 2026-08-21 (KV 账本统一维护改造):
    # sidecar_restore double-file form removed; turn-0 prefix folded into
    # prefill_length; digest prefix_mode="recompute". The queue product is
    # unchanged by WP2 (byte-identical at default window/scale).
    # 2026-08-26 (SLO B1/WP2): the digest sidecar gained four appended audit
    # columns (human_time_ns/tool_time_ns/request_type/decode_length), so its
    # md5 moved to the new frozen value below; the first six columns are
    # unchanged.
    print("queue_md5=%s (frozen: ff18f42df40e5f35582fe3e356c214e8)" % queue_md5)
    print("digest_md5=%s (frozen post-B1: 1430b64ce0fb38b4304bb67082b025b3; "
          "pre-B1 six-column form: a049d7308b4c2c39c13f8a2da55d95ff)" % digest_md5)
    print("window_ns=%d arrival_scale=%g" % (window_ns, arrival_scale))
    print("sessions=%d requests=%d" % (session_count, request_count))
    print("turn0_prefix_gt0_rows=%d max_input_tokens_total=%d" % (turn0_prefix_rows, max_total))
    print("average_decode_length=%s"
          % (decode_length_sum / float(request_count)))
    print("request_type_unknown_rows=%d" % unknown_type_rows)
    # P0 turn-0 fix (2026-08-30): provenance sidecar next to the queue; the
    # C++ reader fail-closes on any mismatch with these stats.
    generator_version = (
        "materialize_first_30s.py window_ns=%d arrival_scale=%g"
        % (window_ns, arrival_scale)
    )
    provenance = queue_provenance(out_queue, generator_version)
    provenance_path = out_queue + ".provenance.json"
    with open(provenance_path, "w", encoding="utf-8") as pout:
        json.dump(provenance, pout, indent=2, sort_keys=True)
        pout.write("\n")
    print(
        "provenance: " + json.dumps(
            {k: provenance[k] for k in (
                "csv_fnv1a64", "csv_bytes", "data_rows", "sessions",
                "turn0_count", "turn0_arrival_min_ns", "turn0_arrival_max_ns",
                "turn0_adjacent_inversions", "session_blocks_contiguous")},
            sort_keys=True))
    print("provenance sidecar: %s" % provenance_path)
    print("[next-steps] 1) trace_config.csv:12 request_queue_csv -> %s" % out_queue)


if __name__ == "__main__":
    main()
