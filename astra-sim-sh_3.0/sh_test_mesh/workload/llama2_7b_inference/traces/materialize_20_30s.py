#!/usr/bin/env python3
"""Materialize the astra_compute_20.csv first-30-seconds simulation input
(sidecar_restore double-file form) for the sh_3.0 repository.

Rebuild entry (backport re-verification 2026-08-16): the phase-0
materializer + PROVENANCE were removed by the phase-7 bare-repo restore;
this reconstruction reproduces the frozen artifacts BYTE-EXACTLY --
queue md5 8b615b5a12b15143a10bbbf13f31a6bb, context sidecar md5
25b7da357cac249c14bf4a3ae26ddfd4 (both registered in sh_3.0改造执行实录
步骤 0-1; the queue's description column is the sh_3.0 wording, without
sh_2.0's "(sidecar_restore)" suffix -- that is the only inter-repo text
difference and it is what the frozen md5s attest).

Simulation-input constraint (user directive 2026-08-15): ONLY
`agent-traces/tracelab/astra_compute_20.csv` rows with
`arrival_time < 30,000,000,000 ns` are permitted. No other csv, no longer
window.

Derivation rules (frozen 3-minute lineage, template_back/history/20_3mins;
identical to the sh_2.0 materializer -- single pass, queue + sidecar +
canonical digest from one traversal):
  1. window: a session is included iff its first row's arrival_time < 30e9;
     turn arrival = previous arrival + previous gap (gap = human_time if
     non-empty else tool_time if non-empty else 0); a session's later turns
     are truncated once the derived arrival >= 30e9.
  2. queue keeps source prefill_length (NEW tokens only, prefix NOT folded
     in) and emits the context sidecar delivering prefix_tokens = source
     prefix_len with input_tokens_total = prefix_tokens + prefill_length for
     every row (sidecar_restore variant; sh_3.0 loads the sidecar --
     trace_config :12/:13 both point into traces/). Prefix must not be
     double counted.
  3. session_arrival_time_ns uses the source absolute time (turn-0 only).
  4. interval semantics = per-row human_time||tool_time gap.
"""

import csv
import hashlib
import os
import sys

SOURCE_DEFAULT = "/home/sunhao/wsc-simulator/agent-traces/tracelab/astra_compute_20.csv"
WINDOW_NS = 30_000_000_000
HERE = os.path.dirname(os.path.abspath(__file__))

QUEUE_HEADER = [
    "session_id", "turn_index", "request_id", "prefill_length",
    "decode_length", "session_arrival_time_ns", "inter_request_interval_ns",
    "description",
]
CONTEXT_HEADER = [
    "session_id", "turn_index", "request_id", "prefix_tokens",
    "input_tokens_total",
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


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else SOURCE_DEFAULT
    out_queue = os.path.join(HERE, "astra_compute_20_first_30_seconds_request_queue.csv")
    out_ctx = os.path.join(HERE, "astra_compute_20_first_30_seconds_request_context.csv")
    out_digest = os.path.join(HERE, "astra_compute_20_first_30_seconds_canonical_digest.csv")

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

    queue_rows, ctx_rows, digest_rows = [], [], []
    session_count = 0
    for sid in order:
        srows = sessions[sid]
        first_arrival = int(srows[0]["arrival_time"])
        if first_arrival >= WINDOW_NS:
            continue
        session_count += 1
        arrival = first_arrival
        for turn, r in enumerate(srows):
            if turn > 0:
                prev = srows[turn - 1]
                arrival = arrival + _gap_ns(prev["human_time"], prev["tool_time"])
                if arrival >= WINDOW_NS:
                    break
            rid = "%s_request_%d" % (sid, turn)
            prefill = int(r["prefill_length"])
            prefix = int(r["prefix_len"])
            if turn == 0:
                session_ns, interval_ns = str(arrival), ""
            else:
                session_ns = ""
                interval_ns = str(_gap_ns(srows[turn - 1]["human_time"], srows[turn - 1]["tool_time"]))
            queue_rows.append([
                sid, turn, rid, prefill, r["decode_length"], session_ns,
                interval_ns,
                "compute_20 first-30-seconds window; turn-0 prefix kept as "
                "remote-resident history KV (see context sidecar)",
            ])
            ctx_rows.append([sid, turn, rid, prefix, prefix + prefill])
            digest = hashlib.sha256(
                ("%s|%d|%d|%d" % (rid, prefix, prefill, prefix + prefill)).encode()
            ).hexdigest()
            digest_rows.append([rid, prefix, prefill, prefix + prefill, digest, "sidecar_restore"])

    def write(path, header, data):
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(data)

    write(out_queue, QUEUE_HEADER, queue_rows)
    write(out_ctx, CONTEXT_HEADER, ctx_rows)
    write(out_digest, DIGEST_HEADER, digest_rows)

    src_md5 = hashlib.md5(open(src, "rb").read()).hexdigest()
    queue_md5 = hashlib.md5(open(out_queue, "rb").read()).hexdigest()
    ctx_md5 = hashlib.md5(open(out_ctx, "rb").read()).hexdigest()
    turn0_prefix_rows = sum(1 for row in ctx_rows if int(row[1]) == 0 and int(row[3]) > 0)
    max_total = max(int(row[4]) for row in ctx_rows)
    print("source_md5=%s" % src_md5)
    print("queue_md5=%s (frozen: 8b615b5a12b15143a10bbbf13f31a6bb)" % queue_md5)
    print("context_md5=%s (frozen: 25b7da357cac249c14bf4a3ae26ddfd4)" % ctx_md5)
    print("sessions=%d requests=%d" % (session_count, len(queue_rows)))
    print("turn0_prefix_gt0_rows=%d max_input_tokens_total=%d" % (turn0_prefix_rows, max_total))
    print("average_decode_length=%s"
          % (sum(int(r[4]) for r in queue_rows) / float(len(queue_rows))))


if __name__ == "__main__":
    main()
