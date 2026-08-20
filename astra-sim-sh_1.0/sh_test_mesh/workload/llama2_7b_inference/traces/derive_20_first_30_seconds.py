#!/usr/bin/env python3
"""Derive the first-30-seconds request queue (recompute variant) from
agent-traces/tracelab/astra_compute_20.csv.

Semantics (wscllm recompute variant, frozen 2026-08-15 by main controller,
cross-checked against template_back/history/20_3mins/traces/
astra_compute_20_first_3_minutes_request_queue_recompute.csv lineage):
- Window: [0, 30 s] measured on source absolute arrival times (ns).
  A session is included iff its first-row arrival_time < 30_000_000_000 ns.
- Within an included session, a turn is kept iff its derived arrival time
  < 30_000_000_000 ns, where derived arrival of turn i is t0 + sum of gaps of
  all previous turns; the gap after turn j is the human_time or tool_time
  recorded on turn j's own row (empty means zero wait).
- recompute prefix semantics: turn-0 prefill = prefix_len + prefill_length
  (turn-0 prefix recomputed as prefill work); later turns prefill =
  prefill_length (prefix already resident after turn-0 recompute).
- session_arrival_time_ns keeps the source absolute time (NOT normalized to 0).

Context sidecar (sh_1.0 sidecar_restore variant, 2026-08-19 改造; sh_3.0
materialize_20_30s.py 同款双件): single traversal additionally emits
- the plain queue `*_request_queue.csv` (prefill_length = source NEW tokens
  only, turn-0 prefix NOT folded in), and
- the context sidecar `*_request_context.csv` (prefix_tokens = source
  prefix_len, input_tokens_total = prefix_tokens + prefill_length, every
  row). Prefix must not be double counted: the simulator loads the sidecar
  via trace_config request_queue_context_csv and treats prefix_tokens as
  already-processed history KV (truncated to the resident ledger).

Output columns match the simulator request-queue format:
session_id,turn_index,request_id,prefill_length,decode_length,
session_arrival_time_ns,inter_request_interval_ns,description

Usage: derive_20_first_30_seconds.py [source] [recompute_queue]
                                      [canonical_sidecar] [window_ns]
(window_ns defaults to 30e9; the plain queue and context sidecar are written
next to the recompute queue, sharing its name prefix.)
"""

import csv
import hashlib
import sys

SOURCE = "/home/sunhao/wsc-simulator/agent-traces/tracelab/astra_compute_20.csv"
OUTPUT = "astra_compute_20_first_30_seconds_request_queue_recompute.csv"
SIDECAR = "astra_compute_20_first_30_seconds_canonical_sidecar.csv"
WINDOW_NS = 30_000_000_000  # 30 seconds (user directive 2026-08-15, plan 0.3)
DESCRIPTION = "compute_20 first-30-seconds window; turn-0 prefix recomputed as prefill work"
PLAIN_DESCRIPTION = (
    "compute_20 first-30-seconds window; turn-0 prefix kept as "
    "historical KV (see context sidecar)"
)

SIDECAR_COLUMNS = [
    "request_id",
    "turn_index",
    "session_id",
    "raw_prefix_tokens",
    "raw_new_prefill_tokens",
    "effective_prompt_tokens",
    "prefill_length",
    "decode_length",
    "session_arrival_time_ns",
    "inter_request_interval_ns",
    "prefix_mode",
    "digest",
]

CONTEXT_COLUMNS = [
    "session_id",
    "turn_index",
    "request_id",
    "prefix_tokens",
    "input_tokens_total",
]


def row_digest(row: list[str]) -> str:
    """Deterministic digest of the canonical 8-column queue row (B0 input
    equivalence check key)."""
    return hashlib.sha256(",".join(row).encode("utf-8")).hexdigest()


def main() -> None:
    source = sys.argv[1] if len(sys.argv) > 1 else SOURCE
    output = sys.argv[2] if len(sys.argv) > 2 else OUTPUT
    sidecar = sys.argv[3] if len(sys.argv) > 3 else SIDECAR
    window_ns = int(sys.argv[4]) if len(sys.argv) > 4 else WINDOW_NS
    # sidecar_restore 双件与 recompute 队列同目录、共享名称前缀。
    stem = output[:-len(".csv")] if output.endswith(".csv") else output
    if stem.endswith("_request_queue_recompute"):
        stem = stem[: -len("_request_queue_recompute")]
    plain_output = stem + "_request_queue.csv"
    context_output = stem + "_request_context.csv"
    if window_ns == WINDOW_NS:
        description = DESCRIPTION
        plain_description = PLAIN_DESCRIPTION
    else:
        window_s = window_ns // 1_000_000_000
        description = (
            f"compute_20 first-{window_s}-seconds window; "
            "turn-0 prefix recomputed as prefill work"
        )
        plain_description = (
            f"compute_20 first-{window_s}-seconds window; "
            "turn-0 prefix kept as historical KV (see context sidecar)"
        )

    n_sessions = 0
    n_requests = 0
    max_prefill = 0
    min_prefill = None
    max_decode = 0
    min_decode = None
    max_arrival = 0
    non_1000 = 0

    with open(source, newline="") as fin, \
            open(output, "w", newline="") as fout, \
            open(sidecar, "w", newline="") as scout, \
            open(plain_output, "w", newline="") as pout, \
            open(context_output, "w", newline="") as cout:
        reader = csv.DictReader(fin)
        writer = csv.writer(fout)
        sidecar_writer = csv.writer(scout)
        plain_writer = csv.writer(pout)
        context_writer = csv.writer(cout)
        queue_header = [
            "session_id",
            "turn_index",
            "request_id",
            "prefill_length",
            "decode_length",
            "session_arrival_time_ns",
            "inter_request_interval_ns",
            "description",
        ]
        writer.writerow(queue_header)
        plain_writer.writerow(queue_header)
        sidecar_writer.writerow(SIDECAR_COLUMNS)
        context_writer.writerow(CONTEXT_COLUMNS)

        current_sid = None
        turn_index = 0
        prev_arrival = 0
        prev_gap = 0  # wait before the current turn, recorded on the previous row
        active = False
        for line_number, row in enumerate(reader, start=2):
            if row["arrival_time"]:
                # First row of a session: absolute arrival in ns.
                current_sid = row["session_id"]
                turn_index = 0
                prev_arrival = int(row["arrival_time"])
                active = prev_arrival < window_ns
                if not active:
                    continue
                n_sessions += 1
            elif active:
                arrival = prev_arrival + prev_gap
                if arrival >= window_ns:
                    active = False
                    continue
                prev_arrival = arrival
            else:
                continue

            prefix = int(row["prefix_len"]) if row["prefix_len"] else 0
            prefill = int(row["prefill_length"])
            decode = int(row["decode_length"])
            if turn_index == 0:
                effective_prefill = prefix + prefill
            else:
                effective_prefill = prefill
            if prev_arrival % 1000 != 0 or (prev_gap and prev_gap % 1000 != 0):
                non_1000 += 1
            max_prefill = max(max_prefill, effective_prefill)
            min_prefill = (
                effective_prefill
                if min_prefill is None
                else min(min_prefill, effective_prefill)
            )
            max_decode = max(max_decode, decode)
            min_decode = decode if min_decode is None else min(min_decode, decode)
            max_arrival = max(max_arrival, prev_arrival)
            n_requests += 1
            if turn_index == 0:
                session_arrival = prev_arrival
                interval = ""
            else:
                session_arrival = ""
                interval = prev_gap
            request_id = f"session_{current_sid}_request_{turn_index}"
            row_out = [
                f"session_{current_sid}",
                turn_index,
                request_id,
                effective_prefill,
                decode,
                session_arrival,
                interval,
                description,
            ]
            writer.writerow(row_out)
            sidecar_writer.writerow([
                request_id,
                turn_index,
                f"session_{current_sid}",
                prefix,
                prefill,
                effective_prefill,
                effective_prefill,
                decode,
                session_arrival,
                interval,
                "recompute",
                row_digest([str(v) for v in row_out]),
            ])
            # sidecar_restore 双件:plain 队列不折前缀(全部 turn 的
            # prefill = 源新 token);context sidecar 逐行交割 prefix。
            plain_writer.writerow([
                f"session_{current_sid}",
                turn_index,
                request_id,
                prefill,
                decode,
                session_arrival,
                interval,
                plain_description,
            ])
            context_writer.writerow([
                f"session_{current_sid}",
                turn_index,
                request_id,
                prefix,
                prefix + prefill,
            ])
            turn_index += 1
            gap_text = row["human_time"] or row["tool_time"]
            prev_gap = int(gap_text) if gap_text else 0

    print(f"sessions: {n_sessions}")
    print(f"requests: {n_requests}")
    print(f"prefill range: {min_prefill}-{max_prefill}")
    print(f"decode range: {min_decode}-{max_decode}")
    print(f"max in-window arrival: {max_arrival / 1e9:.6f} s")
    print(f"rows with timing not multiple of 1000 ns: {non_1000}")
    print(f"plain queue: {plain_output}")
    print(f"context sidecar: {context_output}")
    print("[next-steps] plain 队列须将 trace_config 的 "
          "request_queue_context_csv 指向伴生 %s（漏接将 fail-closed）；"
          "recompute 口径使用 *_request_queue_recompute.csv 且 context 留空"
          % context_output)


if __name__ == "__main__":
    main()
