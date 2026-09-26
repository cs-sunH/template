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

Output columns match the simulator request-queue format:
session_id,turn_index,request_id,prefill_length,decode_length,
session_arrival_time_ns,inter_request_interval_ns,description

P0 turn-0 late-discovery fix (2026-08-30): besides the queue and the canonical
sidecar, the script writes <queue>.provenance.json next to the output queue --
the provenance record the C++ WindowedTraceReader fail-closes on (any field
mismatch aborts the run before a single Submit). Fields: schema=1,
generator_version, csv_sha256 (human audit only; the C++ gate does not hash
sha256), csv_fnv1a64, csv_bytes, data_rows, sessions, turn0_count,
turn0_arrival_min_ns/max_ns, turn0_adjacent_inversions,
session_blocks_contiguous. The FNV-1a 64 digest is byte-for-byte the same
algorithm as the C++ reader (offset basis 14695981039346656037, prime
1099511628211; h = (h ^ byte) * prime over the raw file bytes, mod 2^64).

Usage: derive_20_first_30_seconds.py [source] [recompute_queue]
                                      [canonical_sidecar] [window_ns]
                                      [arrival_scale]
(window_ns defaults to 30e9; the canonical sidecar is written next to the
recompute queue.  arrival_scale defaults to 1.0 and must be a positive
finite float: it divides ONLY the turn-0 session_arrival_time_ns column
(t0 / arrival_scale, i.e. load x arrival_scale).  The exact quotient is
then rounded to the nearest 1000 ns grid point, because the official
loader (generate_trace.load_request_queue) fail-closes on any timing not
divisible by 1000 ns; the deviation vs the exact quotient is < 500 ns,
and the step is a no-op for scale=1.0 on the microsecond-grid source.
Quantized turn-0 rows are counted on stdout.  inter_request_interval_ns
-- the human/tool exogenous waits -- is never scaled, and the window gate
plus all statistics stay on unscaled source times, so scale=1.0
reproduces the frozen 8-column queue byte-for-byte and a scaled run
selects exactly the same sessions/requests.  New scale/type provenance
goes only into the canonical sidecar and stdout, never into the queue.)
"""

import csv
import hashlib
import json
import math
import sys

SOURCE = "/home/sunhao/wsc-simulator/agent-traces/tracelab/astra_compute_20.csv"
OUTPUT = "astra_compute_20_first_30_seconds_request_queue_recompute.csv"
SIDECAR = "astra_compute_20_first_30_seconds_canonical_sidecar.csv"
WINDOW_NS = 30_000_000_000  # 30 seconds (user directive 2026-08-15, plan 0.3)
DESCRIPTION = "compute_20 first-30-seconds window; turn-0 prefix recomputed as prefill work"

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
    "human_time_ns",
    "tool_time_ns",
    "request_type",
]

def row_digest(row: list[str]) -> str:
    """Deterministic digest of the canonical 8-column queue row (B0 input
    equivalence check key)."""
    return hashlib.sha256(",".join(row).encode("utf-8")).hexdigest()


def fnv1a64(data: bytes) -> int:
    """FNV-1a 64 over raw bytes (provenance gate digest).

    Byte-for-byte identical to the C++ reader implementation
    (WindowedTraceReader.cc): offset basis 14695981039346656037, prime
    1099511628211, per byte h = (h ^ byte) * prime, all mod 2^64.
    """
    h = 14695981039346656037
    for byte in data:
        h = ((h ^ byte) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return h


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


def main() -> None:
    source = sys.argv[1] if len(sys.argv) > 1 else SOURCE
    output = sys.argv[2] if len(sys.argv) > 2 else OUTPUT
    sidecar = sys.argv[3] if len(sys.argv) > 3 else SIDECAR
    window_ns = int(sys.argv[4]) if len(sys.argv) > 4 else WINDOW_NS
    try:
        arrival_scale = float(sys.argv[5]) if len(sys.argv) > 5 else 1.0
    except ValueError:
        print(
            f"arrival_scale must be a positive finite float, got "
            f"{sys.argv[5]!r}",
            file=sys.stderr,
        )
        sys.exit(2)
    if not math.isfinite(arrival_scale) or arrival_scale <= 0:
        print(
            f"arrival_scale must be a positive finite float, got "
            f"{arrival_scale!r}",
            file=sys.stderr,
        )
        sys.exit(2)
    if window_ns == WINDOW_NS:
        description = DESCRIPTION
    else:
        window_s = window_ns // 1_000_000_000
        description = (
            f"compute_20 first-{window_s}-seconds window; "
            "turn-0 prefix recomputed as prefill work"
        )

    n_sessions = 0
    n_requests = 0
    max_prefill = 0
    min_prefill = None
    max_decode = 0
    min_decode = None
    max_arrival = 0
    non_1000 = 0
    turn0_scaled_quantized = 0
    request_type_counts = {"human": 0, "tool": 0, "unknown": 0}

    with open(source, newline="") as fin, \
            open(output, "w", newline="") as fout, \
            open(sidecar, "w", newline="") as scout:
        reader = csv.DictReader(fin)
        writer = csv.writer(fout)
        sidecar_writer = csv.writer(scout)
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
        sidecar_writer.writerow(SIDECAR_COLUMNS)

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
                # arrival_scale applies ONLY here (t0 / arrival_scale);
                # window gating above stays on the unscaled source time.
                # The exact quotient is rounded to the nearest 1000 ns grid
                # point: the official loader (generate_trace.load_request_
                # queue) fail-closes on timing % 1000 != 0, and an arbitrary
                # scale (e.g. 1.3 over float division) almost never keeps
                # the quotient on the grid.  Deviation vs the exact quotient
                # is < 500 ns; a no-op for scale=1.0 on microsecond-grid
                # source times.
                scaled_arrival = prev_arrival / arrival_scale
                session_arrival = int(round(scaled_arrival / 1000)) * 1000
                if session_arrival != int(round(scaled_arrival)):
                    turn0_scaled_quantized += 1
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
            human_time_text = row["human_time"]
            tool_time_text = row["tool_time"]
            if human_time_text:
                request_type = "human"
            elif tool_time_text:
                request_type = "tool"
            elif turn_index == 0:
                request_type = "human"
            else:
                request_type = "unknown"
            request_type_counts[request_type] += 1
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
                human_time_text,
                tool_time_text,
                request_type,
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
    print(f"arrival_scale: {arrival_scale} (only turn-0 "
          "session_arrival_time_ns divided; intervals untouched)")
    print(f"turn-0 scaled arrivals rounded to the 1000 ns grid "
          f"(loader timing%1000 gate): {turn0_scaled_quantized}")
    print("request_type counts: human={human} tool={tool} "
          "unknown={unknown}".format(**request_type_counts))
    print(f"recompute queue: {output}")
    print(f"canonical sidecar: {sidecar}")

    # P0 turn-0 fix (2026-08-30): provenance sidecar next to the queue; the
    # C++ reader fail-closes on any mismatch with these stats.
    generator_version = (
        f"derive_20_first_30_seconds.py window_ns={window_ns} "
        f"arrival_scale={arrival_scale}"
    )
    provenance = queue_provenance(output, generator_version)
    provenance_path = output + ".provenance.json"
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
    print(f"provenance sidecar: {provenance_path}")
    print("[next-steps] 将 trace_config 的 request_queue_csv 指向 "
          f"{output}（recompute 单口径，turn-0 前缀已折入 prefill）")


if __name__ == "__main__":
    main()
