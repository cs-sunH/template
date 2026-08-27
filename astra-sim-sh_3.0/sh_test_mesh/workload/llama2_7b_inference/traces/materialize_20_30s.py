#!/usr/bin/env python3
"""Materialize the astra_compute_20.csv first-30-seconds simulation input
(folded recompute single-queue form) for the sh_3.0 repository.

B1/WP2 CLI (SLO 指标改造): ``materialize_20_30s.py [source] [queue_out]
[window_ns] [arrival_scale]`` -- all optional; the defaults (source trace,
native first_30_seconds output names next to this script, window 30e9,
scale 1.0) reproduce the frozen behavior byte-exactly.  ``window_ns``
applies the same session-inclusion / turn-truncation rule to the new
bound.  ``arrival_scale`` divides ONLY the emitted turn-0
``session_arrival_time_ns`` (arrivals compress by 1/scale);
``inter_request_interval_ns`` and the window selection are untouched.
Runs are serialized through flock /tmp/slo_wps/locks/heavy_mat.lock (the
full-source load peaks ~15-20 GB RSS).

Rebuild entry (backport re-verification 2026-08-16): the phase-0
materializer + PROVENANCE were removed by the phase-7 bare-repo restore;
this reconstruction reproduces the frozen artifacts BYTE-EXACTLY.
2026-08-18: the typed-eviction format change (next_trigger_type column)
superseded the 2026-08-16 queue md5. 2026-08-21 (KV 账本统一维护改造):
the sidecar_restore double-file form was removed repo-wide; the turn-0
prefix is now FOLDED into turn-0 prefill_length (full recompute caliber),
the request-context sidecar product is gone, and the canonical digest's
prefix_mode reads "recompute". Session KV history size/location is
maintained dynamically by the runtime KV ledger (KVCacheManager) from
turn-0 prefill accounting onward. The frozen queue/digest md5s below were
re-registered from an actual rerun after this change.

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
"""

import csv
import fcntl
import os
import hashlib
import sys

SOURCE_DEFAULT = "/home/sunhao/wsc-simulator/agent-traces/tracelab/astra_compute_20.csv"
WINDOW_NS = 30_000_000_000
ARRIVAL_SCALE = 1.0
# B3-6 (SLO B3): the source timing grid is whole microseconds; scaled
# turn-0 arrivals are snapped back onto this grid (nearest rounding) so the
# divisibility property survives arrival_scale != 1.0.
ARRIVAL_GRID_NS = 1000
HERE = os.path.dirname(os.path.abspath(__file__))
# B1/WP2: the full-source CSV load peaks ~15-20 GB RSS; serialize concurrent
# materializations repo-wide through this lock (execution plan §1).
HEAVY_MAT_LOCK = "/tmp/slo_wps/locks/heavy_mat.lock"

QUEUE_HEADER = [
    "session_id", "turn_index", "request_id", "prefill_length",
    "decode_length", "session_arrival_time_ns", "inter_request_interval_ns",
    "next_trigger_type",
    "description",
]
DIGEST_HEADER = [
    "request_id", "raw_prefix_tokens", "raw_new_prefill_tokens",
    "effective_prompt_tokens", "digest", "prefix_mode",
]


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


def _parse_args(argv):
    """Usage: materialize_20_30s.py [source] [queue_out] [window_ns] [arrival_scale]

    All four arguments are optional; defaults reproduce the frozen
    first-30-seconds behavior byte-exactly (B1/WP2 backward-compatibility
    gate).  ``queue_out`` selects the queue CSV path (default: the native
    first_30_seconds name next to this script); the canonical digest is
    written next to it (same stem, ``_canonical_digest.csv``).
    ``window_ns`` reuses the same session-inclusion/turn-truncation rule
    against the new bound.  ``arrival_scale`` divides ONLY the emitted
    turn-0 ``session_arrival_time_ns`` (session arrivals compress by 1/scale;
    ``inter_request_interval_ns`` and the window selection itself are NOT
    touched -- same request population, denser arrivals).
    """

    source = argv[1] if len(argv) > 1 else SOURCE_DEFAULT
    queue_out = (
        argv[2] if len(argv) > 2
        else os.path.join(HERE, "astra_compute_20_first_30_seconds_request_queue.csv")
    )
    window_arg = argv[3] if len(argv) > 3 else str(WINDOW_NS)
    scale_arg = argv[4] if len(argv) > 4 else repr(ARRIVAL_SCALE)
    try:
        window_ns = int(window_arg)
    except ValueError:
        raise SystemExit("window_ns must be an integer nanosecond bound, got %r" % window_arg)
    if window_ns <= 0:
        raise SystemExit("window_ns must be positive, got %d" % window_ns)
    try:
        arrival_scale = float(scale_arg)
    except ValueError:
        raise SystemExit("arrival_scale must be a positive float, got %r" % scale_arg)
    if not (arrival_scale > 0) or arrival_scale != arrival_scale:
        raise SystemExit("arrival_scale must be a positive float, got %r" % scale_arg)
    return source, queue_out, window_ns, arrival_scale


def _scaled_session_ns(arrival, scale):
    """Turn-0 session arrival under scaling: value/scale snapped to the
    1000 ns grid with nearest rounding (B3-6, SLO B3).

    scale == 1.0 keeps the integer untouched (no float round-trip) so the
    default output stays byte-identical to the frozen artifact.  Under
    scale != 1.0 a plain division can leave t0 off the source's
    whole-microsecond grid and break the timing divisibility property (the
    S1 derive script guards the same 1000 ns invariant); the scaled value
    is therefore quantized to the nearest grid point.
    """

    if scale == 1.0:
        return arrival
    return max(0, int(round(arrival / scale / ARRIVAL_GRID_NS)) * ARRIVAL_GRID_NS)


def main():
    source, out_queue, window_ns, arrival_scale = _parse_args(sys.argv)
    if out_queue.endswith("_request_queue.csv"):
        out_digest = out_queue[: -len("_request_queue.csv")] + "_canonical_digest.csv"
    else:
        out_digest = out_queue + ".canonical_digest.csv"

    lock_handle = None
    os.makedirs(os.path.dirname(HEAVY_MAT_LOCK), exist_ok=True)
    lock_handle = open(HEAVY_MAT_LOCK, "a+")
    fcntl.flock(lock_handle, fcntl.LOCK_EX)
    try:
        _materialize(source, out_queue, out_digest, window_ns, arrival_scale)
    finally:
        if lock_handle is not None:
            fcntl.flock(lock_handle, fcntl.LOCK_UN)
            lock_handle.close()


def _materialize(src, out_queue, out_digest, window_ns, arrival_scale):
    with open(src, newline="") as f:
        rows = list(csv.DictReader(f))

    # single grouping pass (source is already ordered by session/turn)
    sessions = {}
    order = []
    for r in rows:
        sid = "session_%s" % r["session_id"]
        if sid not in sessions:
            sessions[sid] = []
            order.append(sid)
        sessions[sid].append(r)

    queue_rows, digest_rows = [], []
    session_count = 0
    turn0_prefix_rows = 0
    t0_grid_snapped = 0
    max_total = 0
    for sid in order:
        srows = sessions[sid]
        first_arrival = int(srows[0]["arrival_time"])
        if first_arrival >= window_ns:
            continue
        session_count += 1
        arrival = first_arrival
        for turn, r in enumerate(srows):
            if turn > 0:
                prev = srows[turn - 1]
                arrival = arrival + _gap_ns(prev["human_time"], prev["tool_time"])
                if arrival >= window_ns:
                    break
            rid = "%s_request_%d" % (sid, turn)
            new_prefill = int(r["prefill_length"])
            prefix = int(r["prefix_len"])
            # Folded recompute caliber: turn-0 prefill carries the full
            # source prefix (prefix + new tokens); later turns keep only
            # their new tokens (history comes from the runtime KV ledger).
            prefill = new_prefill + prefix if turn == 0 else new_prefill
            total = prefix + new_prefill
            if turn == 0 and prefix > 0:
                turn0_prefix_rows += 1
            max_total = max(max_total, total)
            if turn == 0:
                scaled_ns = _scaled_session_ns(arrival, arrival_scale)
                if arrival_scale != 1.0 and scaled_ns != int(arrival / arrival_scale):
                    t0_grid_snapped += 1
                session_ns = str(scaled_ns)
                interval_ns = ""
            else:
                session_ns = ""
                interval_ns = str(_gap_ns(srows[turn - 1]["human_time"], srows[turn - 1]["tool_time"]))
            trigger_type = _next_trigger_type(
                r["human_time"], r["tool_time"],
                is_last_row=(turn == len(srows) - 1),
            )
            queue_rows.append([
                sid, turn, rid, prefill, r["decode_length"], session_ns,
                interval_ns, trigger_type,
                "compute_20 first-30-seconds window; turn-0 prefix folded "
                "into prefill_length (recompute caliber)",
            ])
            digest = hashlib.sha256(
                ("%s|%d|%d|%d" % (rid, prefix, new_prefill, total)).encode()
            ).hexdigest()
            digest_rows.append([rid, prefix, new_prefill, total, digest, "recompute"])

    def write(path, header, data):
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(data)

    write(out_queue, QUEUE_HEADER, queue_rows)
    write(out_digest, DIGEST_HEADER, digest_rows)

    src_md5 = hashlib.md5(open(src, "rb").read()).hexdigest()
    queue_md5 = hashlib.md5(open(out_queue, "rb").read()).hexdigest()
    digest_md5 = hashlib.md5(open(out_digest, "rb").read()).hexdigest()
    print("source=%s" % src)
    print("source_md5=%s" % src_md5)
    print("window_ns=%d arrival_scale=%s" % (window_ns, arrival_scale))
    # Frozen references re-registered 2026-08-21 (KV 账本统一维护改造):
    # sidecar_restore double-file form removed; turn-0 prefix folded into
    # prefill_length; digest prefix_mode="recompute". Values below are the
    # actual product md5s of this script's rerun after the change (the
    # defaults window=30e9/scale=1.0 reproduce them byte-exactly).
    print("queue_md5=%s (frozen: ff18f42df40e5f35582fe3e356c214e8)" % queue_md5)
    print("digest_md5=%s (frozen: a049d7308b4c2c39c13f8a2da55d95ff)" % digest_md5)
    print("sessions=%d requests=%d" % (session_count, len(queue_rows)))
    print("turn0_t0_grid_snapped_rows=%d (scale=%s grid=%dns)"
          % (t0_grid_snapped, arrival_scale, ARRIVAL_GRID_NS))
    print("turn0_prefix_gt0_rows=%d max_input_tokens_total=%d" % (turn0_prefix_rows, max_total))
    print("average_decode_length=%s"
          % (sum(int(r[4]) for r in queue_rows) / float(len(queue_rows))))
    print("[next-steps] 1) trace_config.csv:12 request_queue_csv -> %s" % out_queue)


if __name__ == "__main__":
    main()
