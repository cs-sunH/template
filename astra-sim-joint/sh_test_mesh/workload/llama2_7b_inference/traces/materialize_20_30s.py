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
The source is scanned with only one contiguous session buffered and output is
written incrementally, so peak memory is bounded by the largest session rather
than a full-source Python object expansion.  Runs remain serialized through
flock /tmp/slo_wps/locks/heavy_mat.lock to protect native output names.

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

Non-grid fail-closed publication gate (2026-09-29, report 3.14): the
materializer refuses to publish a timing value the Python loader would
reject.  At scale == 1.0 the turn-0 ``session_arrival_time_ns`` goes out
UNQUANTIZED, so BOTH published timing columns are gated before the row is
written: a session_arrival_time_ns / inter_request_interval_ns that is not
a multiple of 1000 ns aborts with the source line and value (the loader
rejects the whole queue on it while the C++ reader accepts it -- a
three-port divergence).  Source timing fields (arrival_time, human_time,
tool_time) are parsed with the strict ASCII-digits-only lexicon (sign,
whitespace, underscore separators and Unicode digits abort with the source
line).

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
import json
import sys

SOURCE_DEFAULT = "/home/sunhao/wsc-simulator/agent-traces/tracelab/astra_compute_20.csv"
WINDOW_NS = 30_000_000_000
ARRIVAL_SCALE = 1.0
# B3-6 (SLO B3): the source timing grid is whole microseconds; scaled
# turn-0 arrivals are snapped back onto this grid (nearest rounding) so the
# divisibility property survives arrival_scale != 1.0.
ARRIVAL_GRID_NS = 1000
HERE = os.path.dirname(os.path.abspath(__file__))
# Serialize concurrent materializations that may target the native outputs.
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


def _parse_nonnegative_ns(field, text, line_num):
    """Report 3.14 fix (2026-09-29): strict parse for source timing fields
    (absolute arrivals and human/tool gap waits).  Python's int() silently
    accepts "+1", " 1", "1_0" and Unicode digits, and a negative arrival
    even publishes verbatim at scale == 1.0; the loader
    (generate_trace.load_request_queue) and the C++ reader enforce plain
    ASCII digits only, so a malformed timing value aborts the authoritative
    materializer with the offending source line instead of publishing a
    queue the downstream ports disagree on."""
    if not text or any(not ("0" <= character <= "9") for character in text):
        raise SystemExit(
            "source line %d: %s=%r: timing fields must be non-negative "
            "integers written as plain ASCII digits (the strict lexicon "
            "shared by generate_trace.load_request_queue and the C++ "
            "reader); queue NOT materialized" % (line_num, field, text))
    return int(text)


def _gap_ns(human_time, tool_time, line_num):
    """Strict non-negative gap of one source row (report 3.14, 2026-09-29):
    no strip(), so a whitespace-wrapped value is rejected by the strict
    lexicon instead of being silently accepted."""
    h = human_time or ""
    t = tool_time or ""
    raw = h if h else (t if t else "")
    if not raw:
        return 0
    return _parse_nonnegative_ns("human_time/tool_time", raw, line_num)


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


def _iter_sessions(reader):
    """Yield one contiguous source session at a time (source-order contract).
    Each row travels with its 1-based source line number (reader.line_num)
    so the fail-closed gates below can point at the offending source line."""
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
        current_rows.append((reader.line_num, row))
    if current_sid is not None:
        yield current_sid, current_rows


def _md5_file(path):
    digest = hashlib.md5()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fnv1a64_cont(h, data):
    """FNV-1a 64 continuation over one chunk of raw bytes.

    Byte-for-byte identical to the C++ reader implementation
    (WindowedTraceReader.cc): offset basis 14695981039346656037, prime
    1099511628211, per byte h = (h ^ byte) * prime, all mod 2^64.
    (Ported from the wscllm mother derive_20_first_30_seconds.py, P0
    turn-0 fix 2026-08-30 lineage.)
    """
    for byte in data:
        h = ((h ^ byte) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return h


def queue_provenance(queue_path, generator_version):
    """Provenance record of a materialized queue, computed from the WRITTEN
    file bytes (exactly what the C++ reader re-derives and compares).

    Definitions mirror the reader's index pass byte-for-byte: lines split on
    b"\\n" (a trailing "\\r" from the CSV writer stays on the line and rides
    on the unparsed description column, exactly as in C++); the first
    non-empty line is the header; every later non-empty line is a data row;
    turn-0 = non-empty session_arrival_time_ns (6th comma field); adjacent
    inversions are counted over turn-0 arrivals in file order; session
    blocks must be contiguous (a session id may never reappear).
    (Ported from the wscllm mother derive_20_first_30_seconds.py.)
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

    def consume(line):
        nonlocal data_rows, turn0_count, turn0_min, turn0_max, inversions
        nonlocal prev_turn0, header_seen, current_session, contiguous
        if not line:
            return
        if not header_seen:
            header_seen = True
            return
        fields = line.split(b",")
        # '#' comment rows (request_id cell starts with '#') are skipped by
        # the C++ index pass (L13) and do not count as data rows; mirror
        # that here so a hand-edited queue with comment rows cannot make
        # the sidecar's data_rows disagree with the reader.
        if len(fields) > 2 and fields[2][:1] == b"#":
            return
        data_rows += 1
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
    session_count = 0
    request_count = 0
    decode_length_sum = 0
    turn0_prefix_rows = 0
    t0_grid_snapped = 0
    max_total = 0
    with open(src, newline="", encoding="utf-8") as source, \
            open(out_queue + ".tmp", "w", newline="",
                 encoding="utf-8") as queue_output, \
            open(out_digest + ".tmp", "w", newline="",
                 encoding="utf-8") as digest_output:
        # R5-S3（2026-09-29 face 移植）：三处文本 open() 补 encoding=
        # "utf-8"（缺省随 locale 漂移）。R3-J2（2026-09-29 face 移植）：
        # 队列/canonical digest 先写 .tmp、流式结束后 os.replace 原子发布
        # （对齐仓内既有原子写范式）——并发读者/C++ 门禁绝不读到半截
        # 产物；provenance sidecar 同款。
        queue_writer = csv.writer(queue_output)
        digest_writer = csv.writer(digest_output)
        queue_writer.writerow(QUEUE_HEADER)
        digest_writer.writerow(DIGEST_HEADER)
        for sid, srows in _iter_sessions(csv.DictReader(source)):
            first_line_num, first_row = srows[0]
            first_arrival = _parse_nonnegative_ns(
                "arrival_time", first_row["arrival_time"], first_line_num)
            if first_arrival >= window_ns:
                continue
            session_count += 1
            arrival = first_arrival
            for turn, entry in enumerate(srows):
                line_num, r = entry
                if turn > 0:
                    prev_line_num, prev = srows[turn - 1]
                    arrival += _gap_ns(
                        prev["human_time"], prev["tool_time"], prev_line_num)
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
                    scaled_ns = _scaled_session_ns(arrival, arrival_scale)
                    if (arrival_scale != 1.0 and
                            scaled_ns != int(arrival / arrival_scale)):
                        t0_grid_snapped += 1
                    session_ns = str(scaled_ns)
                    interval_ns = ""
                else:
                    session_ns = ""
                    interval_ns = str(_gap_ns(
                        srows[turn - 1][1]["human_time"],
                        srows[turn - 1][1]["tool_time"],
                        srows[turn - 1][0]))
                # Report 3.14 fix (2026-09-29): fail-closed non-grid gate.
                # The Python loader (generate_trace.load_request_queue)
                # rejects the WHOLE queue on timing % 1000 != 0 while the
                # C++ reader accepts it, so publishing a non-grid timing
                # value is a three-port divergence.  Unlike the wscllm
                # derive script (which quantizes every turn-0 arrival), this
                # materializer publishes turn-0 arrivals UNQUANTIZED at
                # scale == 1.0, so BOTH published timing columns can leave
                # the 1000 ns grid and both are gated here before the row is
                # written.
                off_grid = None
                if session_ns and int(session_ns) % ARRIVAL_GRID_NS != 0:
                    off_grid = "session_arrival_time_ns %s" % session_ns
                elif interval_ns and \
                        int(interval_ns) % ARRIVAL_GRID_NS != 0:
                    off_grid = (
                        "inter_request_interval_ns %s (the human_time/"
                        "tool_time wait of the previous row)" % interval_ns)
                if off_grid is not None:
                    raise SystemExit(
                        "source line %d: %s is not a multiple of %d ns: the "
                        "Python loader (generate_trace.load_request_queue) "
                        "rejects the whole queue on a non-grid timing value; "
                        "queue NOT materialized"
                        % (line_num, off_grid, ARRIVAL_GRID_NS))
                trigger_type = _next_trigger_type(
                    r["human_time"], r["tool_time"],
                    is_last_row=(turn == len(srows) - 1),
                )
                if window_ns == WINDOW_NS:
                    row_description = (
                        "compute_20 first-30-seconds window; turn-0 prefix "
                        "folded into prefill_length (recompute caliber)")
                else:
                    # R3-J6（2026-09-29 face 移植）：非整秒窗口不再截断
                    # 描述——旧写法恒写 "first-30-seconds"（window_ns 可
                    # 配而描述硬编码 30），描述与实际窗口不符。整秒仍产
                    # 整数标签；非整秒用最短精确小数（%g）。
                    if window_ns % 1_000_000_000 == 0:
                        window_label = str(window_ns // 1_000_000_000)
                    else:
                        window_label = "%g" % (window_ns / 1_000_000_000)
                    row_description = (
                        "compute_20 first-%s-seconds window; turn-0 prefix "
                        "folded into prefill_length (recompute caliber)"
                        % window_label)
                queue_writer.writerow([
                    sid, turn, rid, prefill, r["decode_length"], session_ns,
                    interval_ns, trigger_type,
                    row_description,
                ])
                digest = hashlib.sha256(
                    ("%s|%d|%d|%d" %
                     (rid, prefix, new_prefill, total)).encode()
                ).hexdigest()
                digest_writer.writerow([
                    rid, prefix, new_prefill, total, digest, "recompute",
                ])
                request_count += 1
                decode_length_sum += int(r["decode_length"])

    # R3-J2（2026-09-29 face 移植）：流式结束即原子发布（并发读者/C++
    # 门禁绝不读到半截产物）；md5/ provenance 均在发布后的终版上计算。
    os.replace(out_queue + ".tmp", out_queue)
    os.replace(out_digest + ".tmp", out_digest)

    src_md5 = _md5_file(src)
    queue_md5 = _md5_file(out_queue)
    digest_md5 = _md5_file(out_digest)

    # P0 turn-0 fix (2026-08-30, ported from the wscllm mother): provenance
    # sidecar next to the queue; the C++ calendar reader fail-closes on any
    # mismatch with these stats (absent sidecar = no gate).
    generator_version = (
        "materialize_20_30s.py window_ns=%d arrival_scale=%s"
        % (window_ns, arrival_scale)
    )
    provenance = queue_provenance(out_queue, generator_version)
    provenance_path = out_queue + ".provenance.json"
    # R3-J2: provenance sidecar 原子发布（临时文件 + os.replace）——
    # C++ 门禁按它 fail-closed，绝不读到半截 JSON。
    with open(provenance_path + ".tmp", "w", encoding="utf-8") as pout:
        json.dump(provenance, pout, indent=2, sort_keys=True)
        pout.write("\n")
    os.replace(provenance_path + ".tmp", provenance_path)

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
    print("sessions=%d requests=%d" % (session_count, request_count))
    print("turn0_t0_grid_snapped_rows=%d (scale=%s grid=%dns)"
          % (t0_grid_snapped, arrival_scale, ARRIVAL_GRID_NS))
    print("turn0_prefix_gt0_rows=%d max_input_tokens_total=%d" % (turn0_prefix_rows, max_total))
    print("average_decode_length=%s"
          % (decode_length_sum / float(request_count)
             if request_count else "n/a (zero-request window)"))
    print("provenance: " + json.dumps(
        {k: provenance[k] for k in (
            "csv_fnv1a64", "csv_bytes", "data_rows", "sessions",
            "turn0_count", "turn0_arrival_min_ns", "turn0_arrival_max_ns",
            "turn0_adjacent_inversions", "session_blocks_contiguous")},
        sort_keys=True))
    print("provenance sidecar: %s" % provenance_path)
    print("[next-steps] 1) trace_config.csv request_queue_csv -> %s" % out_queue)


if __name__ == "__main__":
    main()
