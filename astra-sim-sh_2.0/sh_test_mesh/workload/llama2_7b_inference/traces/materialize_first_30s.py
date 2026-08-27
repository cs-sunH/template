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

Runtime footprint: the full source CSV is still loaded into memory, so the
script serializes against other heavy materializers via flock on
/tmp/slo_wps/locks/heavy_mat.lock (best effort; if the lock file cannot be
created the script warns and proceeds unlocked).
"""

import csv
import fcntl
import hashlib
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
    """Serialize heavy in-memory materialization (EXECUTION_PLAN sec.1)."""
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
        max_total = 0
        unknown_type_rows = 0
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
                    # WP2 scaling: only the session t0 is divided by the
                    # scale; intervals and window membership are untouched.
                    session_ns = str(_scaled_t0(arrival, arrival_scale))
                    interval_ns = ""
                else:
                    session_ns = ""
                    interval_ns = str(_gap_ns(srows[turn - 1]["human_time"], srows[turn - 1]["tool_time"]))
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
                queue_rows.append([
                    sid, turn, rid, prefill, r["decode_length"], session_ns,
                    interval_ns, trigger_type,
                    description,
                ])
                digest = hashlib.sha256(
                    ("%s|%d|%d|%d" % (rid, prefix, new_prefill, total)).encode()
                ).hexdigest()
                digest_rows.append([
                    rid, prefix, new_prefill, total, digest, "recompute",
                    human_text, tool_text, request_type, r["decode_length"],
                ])

        def write(path, header, data):
            with open(path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(header)
                w.writerows(data)

        write(out_queue, QUEUE_HEADER, queue_rows)
        write(out_digest, DIGEST_HEADER, digest_rows)
    finally:
        if lock_handle is not None:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
            lock_handle.close()

    src_md5 = hashlib.md5(open(src, "rb").read()).hexdigest()
    queue_md5 = hashlib.md5(open(out_queue, "rb").read()).hexdigest()
    digest_md5 = hashlib.md5(open(out_digest, "rb").read()).hexdigest()
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
    print("sessions=%d requests=%d" % (session_count, len(queue_rows)))
    print("turn0_prefix_gt0_rows=%d max_input_tokens_total=%d" % (turn0_prefix_rows, max_total))
    print("average_decode_length=%s"
          % (sum(int(r[4]) for r in queue_rows) / float(len(queue_rows))))
    print("request_type_unknown_rows=%d" % unknown_type_rows)
    print("[next-steps] 1) trace_config.csv:12 request_queue_csv -> %s" % out_queue)


if __name__ == "__main__":
    main()
