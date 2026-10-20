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

P1 atomic-materialization fix (2026-09-28): the queue, the canonical sidecar
and the provenance record are each written to a sibling .tmp file and
published by rename ONLY after all three are complete; any mid-derivation
abort (fail-closed exit(2), malformed row, interrupt) removes the temps and
leaves the final paths untouched (the C++ reader now fail-closes on an
absent provenance record, so a truncated queue can never reach a run).

D3 arrival-chain semantics registration (2026-09-28): the static chain
implemented here ("turn i arrives at turn i-1's DERIVED ARRIVAL + the gap
recorded on turn i-1's row") is the input-selection/materialization
semantics frozen 2026-08-15 and pinned by the ledger_reconcile acceptance
values (1177 requests / 112 sessions for the 30 s window).  The runtime
replay uses a DIFFERENT, causal chain ("turn i is scheduled at the parent
turn's REQUEST_COMPLETE tick + turn i's own inter_request_interval_ns",
online/wsc_llm_online_scheduler.py _on_request_complete -> future_alarms):
completion times do not exist at materialization time, so the runtime
arrival of turn i is always >= the static derived arrival by the
accumulated per-turn service time.  The two chains answer different
questions (which rows are IN the frozen window vs when the replay fires
them) and are intentionally not unified; do not "fix" one to match the
other without re-freezing the window contract (ledger_reconcile defaults
1177/112 and the byte-exact 30 s smoke baseline move with it).

D3 fail-closed source guards (2026-09-28): an absolute-arrival row whose
session_id is empty, re-opens the CURRENT session block (mid-session
restart -- it would silently reset turn_index to 0 and emit a duplicate
(session, turn-0) row), or re-opens any EARLIER session (non-contiguous
block -- the C++ reader aborts on exactly this shape deep inside the run)
now aborts the derivation with exit 2 before anything is published, so the
error points at the source line instead of a downstream reader.  session_id
values containing comma/quote/CR/LF are likewise rejected: the 8-column
queue is parsed by raw comma splitting on the C++ side (no CSV quoting),
so such a session could not round-trip the queue contract.  A row with
prefill_length/decode_length <= 0 is rejected: the Python loader
(generate_trace.parse_int) and the C++ reader both require positive
lengths, and the writer accepting 0 was the only port still admitting the
shape.  All opens are pinned to encoding='utf-8' (an unpinned text open
writes the locale codec, which on Windows is not UTF-8 and would corrupt
the queue for the byte-exact C++ reader).

non-grid fail-closed publication gate (2026-09-29, report 3.14): the
authoritative materializer now refuses to publish any timing value the
Python loader would reject.  Of the published columns, only a continuation
row's inter_request_interval_ns (= the previous row's human/tool wait) can
leave the 1000 ns grid -- the turn-0 session_arrival_time_ns is forced onto
the grid by its quantized division (deviation counted by
turn0_scaled_quantized) -- and a non-grid interval now aborts with the
source line and value (the former non_1000 count-and-publish diagnostic
published a queue that generate_trace.load_request_queue rejects wholesale
while the C++ reader accepts it: a three-port divergence).  Source timing
fields (arrival_time, human_time, tool_time) are parsed with the same
strict ASCII-digits-only lexicon as the loader and the C++ reader: sign,
whitespace, underscore separators and Unicode digits abort with the source
line (a negative arrival previously survived the turn-0 quantization as a
fabricated 0 ns via banker's rounding).

Usage: derive_20_first_30_seconds.py [source] [recompute_queue]
                                      [canonical_sidecar] [window_ns]
                                      [arrival_scale]
(window_ns defaults to 30e9; the canonical sidecar path is taken verbatim
from argv[3] -- a bare filename lands in the process CWD, NOT next to the
recompute queue.  arrival_scale defaults to 1.0 and must be a positive
finite float: it divides ONLY the turn-0 session_arrival_time_ns column
(t0 / arrival_scale, i.e. load x arrival_scale); inter_request_interval_ns
-- the human/tool exogenous waits -- is never scaled, and the window gate
plus all statistics stay on unscaled source times, so scale=1.0
reproduces the frozen 8-column queue byte-for-byte and a scaled run
selects exactly the same sessions/requests.  Scale/type provenance never
enters the queue: request_type is a canonical-sidecar column and is also
summarized on stdout; arrival_scale goes to stdout and the provenance
record (generator_version field of <queue>.provenance.json) only -- the
canonical sidecar has no scale column.)
"""

import csv
import hashlib
import json
import math
import os
import sys

# infra-8 同步(2026-09-27,对齐 face 2026-09-26 修复批):源 CSV 必填、无
# 内建默认路径(request-neutral fail-closed 入口)——原硬编码作者机器
# 绝对路径已删除,缺失 source 时 main() 打印用法并以 exit 2 退出。
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


def _die(source: str, line_number: int, why: str) -> None:
    """Fail-closed abort for a malformed source row (no partial output)."""
    print(
        f"source {source} line {line_number}: {why}; "
        "queue NOT materialized",
        file=sys.stderr,
    )
    sys.exit(2)


def _validate_session_id(source: str, line_number: int, session_id: str) -> None:
    """D3 fix (2026-09-28): the 8-column queue contract is positionally
    comma-split by the C++ reader with NO CSV quoting, and the request_id is
    synthesized from the session_id -- a session_id containing any of the
    comma/quote/CR/LF characters cannot round-trip the contract (the csv
    writer would QUOTE_MINIMAL-escape it, the C++ splitter would then see
    extra/misaligned fields). Reject at the only port that knows the source
    row, instead of crashing or mis-parsing inside the simulator."""
    if "first_token" in session_id:
        _die(
            source,
            line_number,
            f"session_id {session_id!r} contains the reserved marker token "
            "'first_token' (the WP9 node-name anchor substring; the Python "
            "loader and the C++ reader reject the same shape -- three-port "
            "lexicon closure, 2026-09-30)",
        )
    for character in session_id:
        if character in {",", '"', "\r", "\n"}:
            _die(
                source,
                line_number,
                f"session_id {session_id!r} contains "
                f"{character!r}: the 8-column queue is parsed by raw comma "
                "splitting on the C++ side (no CSV quoting), so comma, "
                "quote, CR and LF cannot round-trip the queue contract",
            )


def _parse_nonnegative_ns(source: str, line_number: int, field: str, text: str) -> int:
    """Report 3.14 fix (2026-09-29): strict parse for source timing fields
    (absolute arrivals and human/tool gap waits).  Python's int() silently
    accepts "+1", " 1", "1_0" and Unicode digits, and a negative arrival even
    survived the turn-0 quantization below as a fabricated 0 ns arrival
    (round(-0.4) == 0); the loader (generate_trace._require_ascii_digits)
    and the C++ reader (parse_u64_field) already enforce plain ASCII digits
    only, so a malformed timing value aborts at the authoritative
    materializer with the offending source line instead of diverging
    downstream."""
    if not text or any(not ("0" <= character <= "9") for character in text):
        _die(
            source,
            line_number,
            f"{field}={text!r}: timing fields must be non-negative integers "
            "written as plain ASCII digits (the strict lexicon shared by "
            "generate_trace.load_request_queue and the C++ reader)",
        )
    return int(text)


def _parse_source_int(source: str, line_number: int, field: str,
                      text: str) -> int:
    """Same plain-ASCII-digit lexicon as _parse_nonnegative_ns, for the
    non-timing source fields (lengths): int() silently accepts "+1", " 1",
    "1_0" and Unicode digits, which would normalize into the published
    queue while the loader/C++ lexicons reject them (2026-09-30)."""
    if not text or any(not ("0" <= character <= "9") for character in text):
        _die(
            source,
            line_number,
            f"{field}={text!r}: source fields must be non-negative "
            "integers written as plain ASCII digits (the strict lexicon "
            "shared by generate_trace.load_request_queue and the C++ "
            "reader)",
        )
    return int(text)


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
    if len(sys.argv) < 2:
        # infra-8 同步(2026-09-27,对齐 face 2026-09-26 修复批): the former
        # hardcoded absolute default source path was removed -- the source
        # CSV is required and must be passed explicitly (request-neutral
        # fail-closed entry).
        print(
            "usage: derive_20_first_30_seconds.py <source> [recompute_queue] "
            "[canonical_sidecar] [window_ns] [arrival_scale]\n"
            "source is required (no built-in default path): pass the "
            "normalized agent-trace CSV to derive, e.g. "
            "agent-traces/tracelab/astra_compute_20.csv",
            file=sys.stderr,
        )
        sys.exit(2)
    source = sys.argv[1]
    output = sys.argv[2] if len(sys.argv) > 2 else OUTPUT
    sidecar = sys.argv[3] if len(sys.argv) > 3 else SIDECAR
    window_ns = WINDOW_NS
    if len(sys.argv) > 4:
        # README documents the scientific-notation form (e.g. 2e9), so parse
        # through float and require a positive integral value; anything else
        # exits 2 with the usage (2026-09-30), same channel as arrival_scale.
        try:
            window_candidate = float(sys.argv[4])
        except ValueError:
            window_candidate = -1.0
        if (not math.isfinite(window_candidate) or window_candidate <= 0
                or window_candidate != int(window_candidate)):
            print(
                f"window_ns must be a positive integer number of ns "
                f"(2e9-style notation accepted), got {sys.argv[4]!r}",
                file=sys.stderr,
            )
            sys.exit(2)
        window_ns = int(window_candidate)
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
        # (2026-10-01) Non-integral windows used to floor the seconds count
        # (2.5e9 -> "first-2-seconds"); render the exact value instead. The
        # canonical 30s window divides evenly and keeps its frozen bytes.
        window_s = window_ns / 1_000_000_000
        window_s_str = (
            f"{int(window_s)}" if window_s == int(window_s)
            else f"{window_s:g}")
        description = (
            f"compute_20 first-{window_s_str}-seconds window; "
            "turn-0 prefix recomputed as prefill work"
        )

    n_sessions = 0
    n_requests = 0
    max_prefill = 0
    min_prefill = None
    max_decode = 0
    min_decode = None
    max_arrival = 0
    turn0_scaled_quantized = 0
    request_type_counts = {"human": 0, "tool": 0, "unknown": 0}

    # P1 atomic-materialization fix (2026-09-28): stage every output in a
    # sibling .tmp file; publish by rename only after queue + sidecar +
    # provenance are ALL complete. On any failure (fail-closed exit(2),
    # malformed row, interrupt, disk full) the temps are removed and the
    # final paths keep their previous complete materialization (or stay
    # absent) -- a truncated queue can never appear at a final path again.
    tmp_output = output + ".tmp"
    tmp_sidecar = sidecar + ".tmp"
    provenance_path = output + ".provenance.json"
    tmp_provenance = provenance_path + ".tmp"
    try:
        with open(source, newline="", encoding="utf-8") as fin, \
                open(tmp_output, "w", newline="", encoding="utf-8") as fout, \
                open(tmp_sidecar, "w", newline="", encoding="utf-8") as scout:
            reader = csv.DictReader(fin)
            source_columns = (
                "session_id", "arrival_time", "prefix_len",
                "prefill_length", "decode_length", "human_time",
                "tool_time")
            if reader.fieldnames is None:
                _die(source, 2, "source CSV has no header row")
            missing_columns = [
                column for column in source_columns
                if column not in reader.fieldnames]
            if missing_columns:
                _die(
                    source, 2,
                    "source CSV is missing columns: "
                    + ", ".join(missing_columns))
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
            # D3 fix (2026-09-28): every session that has already opened an
            # absolute-arrival block. Session blocks must be contiguous and
            # open exactly once -- the same invariant the C++ reader enforces
            # in-row (reader_fatal on a re-opened session id); enforcing it
            # here makes the derivation abort point at the source line
            # instead of a confusing mid-run reader error.
            seen_sessions: set[str] = set()
            turn_index = 0
            prev_arrival = 0
            prev_gap = 0  # wait before the current turn, recorded on the previous row
            active = False
            for line_number, row in enumerate(reader, start=2):
                if not row["arrival_time"] and row["session_id"] != current_sid:
                    # fail-closed (105②): the derivation groups source rows by
                    # session blocks -- an empty-arrival row is a continuation
                    # turn of the CURRENT session (turn_index keeps incrementing
                    # under it).  A session-id switch on such a row would
                    # silently mis-attribute the turn, so abort HERE
                    # (authoritative materializer) instead of writing a wrong
                    # queue.
                    _die(
                        source,
                        line_number,
                        f"empty arrival_time with session_id "
                        f"{row['session_id']!r} after session "
                        f"{current_sid!r}: source rows are not grouped by "
                        "session",
                    )
                if row["arrival_time"]:
                    # D3 fix (2026-09-28), guard order matters: an
                    # absolute-arrival row OPENS a session block, so the
                    # incoming session id must be non-empty, unseen, and
                    # different from the current block. The former code took
                    # any absolute-arrival row as a fresh session and silently
                    # reset turn_index to 0: a mid-session restart emitted a
                    # duplicate (session, turn-0) row with zero detection
                    # (the C++ side then tripped over the duplicate turn
                    # shape far from the root cause), and an empty session_id
                    # materialized a literal "session_" pseudo-session.
                    session_id = row["session_id"]
                    if not session_id:
                        _die(
                            source,
                            line_number,
                            "empty session_id on an absolute-arrival "
                            "(turn-0) row: it would materialize a "
                            "\"session_\" pseudo-session",
                        )
                    if session_id == current_sid:
                        _die(
                            source,
                            line_number,
                            f"absolute-arrival row re-opens the CURRENT "
                            f"session block {session_id!r}: a mid-session "
                            "restart would silently reset turn_index to 0 "
                            "and emit a duplicate (session, turn-0) row",
                        )
                    if session_id in seen_sessions:
                        _die(
                            source,
                            line_number,
                            f"absolute-arrival row re-opens the already "
                            f"finished session {session_id!r}: session "
                            "blocks must be contiguous (the C++ reader "
                            "fail-closes on the same shape)",
                        )
                    _validate_session_id(source, line_number, session_id)
                    seen_sessions.add(session_id)
                    # First row of a session: absolute arrival in ns.
                    current_sid = session_id
                    turn_index = 0
                    prev_arrival = _parse_nonnegative_ns(
                        source, line_number, "arrival_time",
                        row["arrival_time"])
                    # Session boundary: a turn-0 row has no previous turn, so the
                    # gap must never carry over from the previous session's last
                    # row (it would pollute the non-1000 diagnostic counter).
                    prev_gap = 0
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

                prefix = (
                    _parse_source_int(source, line_number,
                                      "prefix_len", row["prefix_len"])
                    if row["prefix_len"] else 0)
                prefill = _parse_source_int(
                    source, line_number, "prefill_length",
                    row["prefill_length"])
                decode = _parse_source_int(
                    source, line_number, "decode_length",
                    row["decode_length"])
                # D3 fix (2026-09-28): the three ports of the trace chain now
                # share ONE length lexicon -- the Python loader
                # (generate_trace.parse_request_length -> parse_int) has always
                # required a positive length and the C++ reader rejects
                # non-positive lengths since the same fix batch, so a zero (or
                # negative) length row must fail-closed HERE too instead of
                # materializing a queue the other two ports refuse.
                if prefill <= 0 or decode <= 0:
                    _die(
                        source,
                        line_number,
                        f"prefill_length={row['prefill_length']!r} "
                        f"decode_length={row['decode_length']!r}: zero or "
                        "negative request lengths are rejected by the Python "
                        "loader and the C++ reader alike",
                    )
                if turn_index == 0:
                    effective_prefill = prefix + prefill
                else:
                    effective_prefill = prefill
                # Report 3.14 fix (2026-09-29): fail-closed non-grid gate,
                # replacing the former `non_1000` count-and-publish
                # diagnostic.  Of the values this row publishes, only a
                # continuation row's inter_request_interval_ns (= prev_gap,
                # the raw human/tool wait of the previous row) can leave the
                # 1000 ns grid: the turn-0 arrival is forced onto the grid
                # by the quantized division below (deviation still counted
                # by turn0_scaled_quantized).  generate_trace.load_request_
                # queue rejects the WHOLE queue on timing % 1000 != 0 while
                # the C++ reader accepts it, so publishing a non-grid
                # interval is a three-port divergence -- abort with the
                # source line and value instead.
                if turn_index != 0 and prev_gap % 1000 != 0:
                    _die(
                        source,
                        line_number,
                        f"inter_request_interval_ns {prev_gap} (the "
                        "human_time/tool_time wait of the previous row) is "
                        "not a multiple of 1000 ns: the Python loader "
                        "(generate_trace.load_request_queue) rejects the "
                        "whole queue on a non-grid timing value",
                    )
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
                    # is <= 500 ns -- Python's round() is banker's rounding
                    # (half-to-even), so an exact half-grid point (500 ns) can
                    # round to the even neighbour and land at EXACTLY 500 ns,
                    # never more; a no-op for scale=1.0 on microsecond-grid
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
                prev_gap = (
                    _parse_nonnegative_ns(
                        source, line_number, "human_time/tool_time",
                        gap_text)
                    if gap_text
                    else 0
                )

        # Zero-request gate (2026-09-30): an empty source or an all-out-of-
        # window window used to publish a header-only queue with a matching
        # data_rows=0 provenance and exit 0 -- the C++ reader's run-end
        # gates all key off expected_requests > 0, so a direct feed of that
        # queue was a silent zero-request green run. Refuse to publish (the
        # Python loader rejects the same shape with "must contain at least
        # one request").
        if n_requests == 0:
            print(
                "derive aborted: zero requests in the window -- refusing "
                "to publish a header-only queue (the Python loader and the "
                "C++ reader both reject a zero-request queue)",
                file=sys.stderr,
            )
            sys.exit(2)

        # P0 turn-0 fix (2026-08-30): provenance sidecar next to the queue; the
        # C++ reader fail-closes on any mismatch with these stats. Computed
        # from the STAGED queue bytes (identical to the published ones).
        generator_version = (
            f"derive_20_first_30_seconds.py window_ns={window_ns} "
            f"arrival_scale={arrival_scale}"
        )
        provenance = queue_provenance(tmp_output, generator_version)
        with open(tmp_provenance, "w", encoding="utf-8") as pout:
            json.dump(provenance, pout, indent=2, sort_keys=True)
            pout.write("\n")

        # All three outputs complete: atomic publish.
        os.replace(tmp_output, output)
        os.replace(tmp_sidecar, sidecar)
        os.replace(tmp_provenance, provenance_path)
    except BaseException:
        for tmp_path in (tmp_output, tmp_sidecar, tmp_provenance):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        raise

    print(f"sessions: {n_sessions}")
    print(f"requests: {n_requests}")
    print(f"prefill range: {min_prefill}-{max_prefill}")
    print(f"decode range: {min_decode}-{max_decode}")
    print(f"max in-window arrival: {max_arrival / 1e9:.6f} s")
    print(f"arrival_scale: {arrival_scale} (only turn-0 "
          "session_arrival_time_ns divided; intervals untouched)")
    print(f"turn-0 scaled arrivals rounded to the 1000 ns grid "
          f"(loader timing%1000 gate): {turn0_scaled_quantized}")
    print("request_type counts: human={human} tool={tool} "
          "unknown={unknown}".format(**request_type_counts))
    print(f"recompute queue: {output}")
    print(f"canonical sidecar: {sidecar}")
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
