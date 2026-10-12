
"""Derive the first-30-seconds request queue from
agent-traces/tracelab/astra_compute_20.csv.

命名说明(py-metrics-C4,2026-09-26):本仓 RECOMPUTE 机制已物理删除,现行
语义即 KV 复用——turn-0 前缀折入 prefill 一次性算毕,后续 turn 复用常驻
前缀、只算增量;派生公式不变,仅"recompute"字样为历史命名。输出文件名与
sidecar 的 prefix_mode=recompute 值是跨文件冻结接口(trace_config 物化
约定 + C++ provenance 门),不可改动。

Semantics (frozen 2026-08-15 by main controller,
cross-checked against template_back/history/20_3mins/traces/
astra_compute_20_first_3_minutes_request_queue_recompute.csv lineage):
- Window: [0, 30 s] measured on source absolute arrival times (ns).
  A session is included iff its first-row arrival_time < 30_000_000_000 ns.
- Within an included session, a turn is kept iff its derived arrival time
  < 30_000_000_000 ns, where derived arrival of turn i is t0 + sum of gaps of
  all previous turns; the gap after turn j is the human_time or tool_time
  recorded on turn j's own row (empty means zero wait).
- prefix semantics (KV 复用口径):turn-0 prefill = prefix_len +
  prefill_length (turn-0 prefix folded into the one-shot prefill);
  later turns prefill = prefill_length (prefix already resident and
  reused).
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
(window_ns defaults to 30e9; the canonical sidecar path is taken verbatim
from argv[3] -- a bare filename lands in the process CWD, NOT next to the
recompute queue.  arrival_scale defaults to 1.0 and must be a positive
finite float: it divides ONLY the turn-0 session_arrival_time_ns column
(t0 / arrival_scale, i.e. load x arrival_scale); inter_request_interval_ns
-- the human/tool exogenous waits -- is never scaled, and the window gate
plus all statistics stay on unscaled source times, so scale=1.0
reproduces the frozen 8-column queue byte-for-byte and a scaled run
selects exactly the same sessions/requests.  New scale/type provenance
goes only into the canonical sidecar and stdout, never into the queue.)

Publishing (2026-09-30): the queue, the canonical sidecar and the
provenance record are staged under unique .tmp names and published with
os.replace (queue -> sidecar -> provenance, the C++ REQUIRED gate
artifact lands last) -- a concurrent reader never sees a half-written
queue or a torn provenance sidecar, and a failed derivation leaves no
staged files behind.  Any timing that leaves the 1000 ns grid fail-closes
the derivation (both loaders reject such timings, so a "successful"
off-grid queue is unpublishable by construction).
"""

import csv
import hashlib
import json
import math
import os
import sys
import uuid




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


def fnv1a64_cont(h: int, data: bytes) -> int:
    """FNV-1a 64 over raw bytes, chunked for streaming. Byte-for-byte
identical to the C++ reader implementation (WindowedTraceReader.cc):
offset basis 14695981039346656037, prime 1099511628211, per byte
h = (h ^ byte) * prime, all mod 2^64. Callers start from the offset
basis and feed the file in chunks."""
    for byte in data:
        h = ((h ^ byte) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return h


def staging_path(path: str) -> str:
    """Unique staging name next to the target (pid + uuid4).

    M-38 同族（2026-09-30）：固定 .tmp 名是并发同目录派生的共享槽位，
    两个进程互踩会静默发布数据混血的产物；唯一名保证各进程各写各的
    stage 文件，os.replace 原子发布（同目录同文件系统）。
    """
    return f"{path}.tmp.{os.getpid()}.{uuid.uuid4().hex}"


def queue_provenance(queue_path: str, generator_version: str) -> dict:
    """Provenance record of a materialized queue, computed from the WRITTEN
    file bytes (exactly what the C++ reader re-derives and compares).

    Definitions mirror the reader's index pass byte-for-byte (aligned
    2026-10-01 with the C++ reader's 2026-09-29 terminator/comment handling):
    lines split on b"\n"; a single trailing b"\r" is stripped BEFORE the
    empty-line check (a CRLF blank row is not a data row, exactly as in the
    C++ process_indexed_row); a non-empty later line whose request_id (3rd
    comma field) starts with '#' is a comment row and is skipped BEFORE
    counting (the C++ reader's M11 classification point -- it consumes
    neither a data-row count nor a queue index); the first
    non-empty line is the header; every other later non-empty line is a
    data row;
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
        # 对齐 C++ process_indexed_row 的处理次序（R2 复核补正）：先剥
        # 行尾单个 '\r' 再做空行判定（CRLF 空行不是数据行、也不吞表头
        # 位）——CRLF 空行作文件首行时 C++ 仍以首个真非空行为表头。
        if line.endswith(b"\r"):
            line = line[:-1]
        if not line:
            return
        if not header_seen:
            header_seen = True
            return
        fields = line.split(b",")
        # 对齐 C++ M11 分类点：request_id（第 3 列）以 '#' 开头的注释行
        # 在计数之前跳过（不占数据行、不占 queue index）。
        if len(fields) > 2 and fields[2].startswith(b"#"):
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
    if len(sys.argv) > 4:
        try:
            window_ns = int(sys.argv[4])
        except ValueError:
            print(
                f"window_ns must be an integer nanosecond count, got "
                f"{sys.argv[4]!r}",
                file=sys.stderr,
            )
            sys.exit(2)
    else:
        window_ns = WINDOW_NS
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
        window_s = window_ns / 1000000000
        description = (
            f"compute_20 first-{window_s:g}-seconds window; "
            "turn-0 prefix recomputed as prefill work"
        )

    n_sessions = 0
    n_requests = 0
    turn0_scaled_quantized = 0
    max_prefill = 0
    min_prefill = None
    max_decode = 0
    min_decode = None
    max_arrival = 0
    request_type_counts = {"human": 0, "tool": 0, "unknown": 0}

    # P1 atomic-materialization fix (2026-09-28) + M-38 同族 unique staging
    # (2026-09-30): stage every output under a unique sibling temp name and
    # publish by os.replace only after queue + sidecar + provenance are ALL
    # complete (the C++ REQUIRED gate artifact lands last). Any failure
    # (fail-closed exit(2), malformed row, interrupt, disk full) removes the
    # staged files and leaves the final paths untouched.
    queue_staged = staging_path(output)
    sidecar_staged = staging_path(sidecar)
    provenance_path = output + ".provenance.json"
    provenance_staged = staging_path(provenance_path)
    fin = fout = scout = None
    try:
        fin = open(source, newline="", encoding="utf-8")
        fout = open(queue_staged, "w", newline="", encoding="utf-8")
        scout = open(sidecar_staged, "w", newline="", encoding="utf-8")
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
        prev_gap = 0
        active = False
        for line_number, row in enumerate(reader, start=2):
            if (
                not row["arrival_time"]
                and row["session_id"] != current_sid
            ):
                print(
                    f"source {source} line {line_number}: "
                    f"empty arrival_time with session_id "
                    f"{row['session_id']!r} after session "
                    f"{current_sid!r}: source rows are not grouped by "
                    "session; queue NOT materialized",
                    file=sys.stderr,
                )
                sys.exit(2)
            if row["arrival_time"]:
                # turn-0 of a (possibly new) session block
                current_sid = row["session_id"]
                turn_index = 0
                prev_arrival = int(row["arrival_time"])

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

            prefix = int(row["prefix_len"]) if row["prefix_len"] else 0
            prefill = int(row["prefill_length"])
            decode = int(row["decode_length"])
            if turn_index == 0:
                effective_prefill = prefix + prefill
            else:
                effective_prefill = prefill
            if prev_arrival % 1000 != 0 or prev_gap and prev_gap % 1000 != 0:
                print(
                    f"source {source} line {line_number}: "
                    f"timing off the 1000 ns grid (turn arrival "
                    f"{prev_arrival} ns, previous-turn wait "
                    f"{prev_gap} ns): both request-queue loaders reject "
                    "non-grid timings -- queue NOT materialized",
                    file=sys.stderr,
                )
                sys.exit(2)
            max_prefill = max(max_prefill, effective_prefill)
            min_prefill = (
                effective_prefill
                if min_prefill is None
                else min(min_prefill, effective_prefill)
            )
            max_decode = max(max_decode, decode)
            min_decode = (
                decode if min_decode is None else min(min_decode, decode)
            )
            max_arrival = max(max_arrival, prev_arrival)
            n_requests += 1
            if turn_index == 0:
                # WP2/derive: only the turn-0 absolute arrival is scaled;
                # the scaled value must stay on the 1000 ns grid both
                # loaders enforce.
                # 2026-10-11 移植 face 仓 half-up 网格量化（偏差 ≤500ns，
                # verify_arrival_scaling bound_ns=500 口径验收），不再
                # fail-closed（原行为使 λ>1 与非二幂中点在本窗不可物化）。
                scaled_arrival = prev_arrival / arrival_scale
                session_arrival = (
                    math.floor(scaled_arrival / 1000 + 0.5) * 1000)
                if session_arrival != math.floor(scaled_arrival + 0.5):
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
            sidecar_writer.writerow(
                [
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
                ]
            )
            turn_index += 1
            gap_text = row["human_time"] or row["tool_time"]
            prev_gap = int(gap_text) if gap_text else 0
        fout.close()
        scout.close()
        fin.close()

        print(f"sessions: {n_sessions}")
        print(f"requests: {n_requests}")
        print(f"prefill range: {min_prefill}-{max_prefill}")
        print(f"decode range: {min_decode}-{max_decode}")
        print(f"max in-window arrival: {max_arrival / 1000000000.0:.6f} s")
        print(
            f"arrival_scale: {arrival_scale}"
            " (only turn-0 session_arrival_time_ns divided; intervals "
            "untouched)"
        )
        print(
            "request_type counts: human={human} tool={tool} "
            "unknown={unknown}".format(**request_type_counts)
        )
        print(f"recompute queue: {output}")
        print(f"canonical sidecar: {sidecar}")

        generator_version = (
            f"derive_20_first_30_seconds.py window_ns={window_ns} "
            f"arrival_scale={arrival_scale}"
        )
        provenance = queue_provenance(queue_staged, generator_version)
        with open(provenance_staged, "w", encoding="utf-8") as pout:
            json.dump(provenance, pout, indent=2, sort_keys=True)
            pout.write("\n")
        print(
            "provenance: "
            + json.dumps(
                {
                    k: provenance[k]
                    for k in (
                        "csv_fnv1a64",
                        "csv_bytes",
                        "data_rows",
                        "sessions",
                        "turn0_count",
                        "turn0_arrival_min_ns",
                        "turn0_arrival_max_ns",
                        "turn0_adjacent_inversions",
                        "session_blocks_contiguous",
                    )
                },
                sort_keys=True,
            )
        )

        os.replace(queue_staged, output)
        os.replace(sidecar_staged, sidecar)
        os.replace(provenance_staged, provenance_path)
        print(f"provenance sidecar: {provenance_path}")
        print(
            f"[next-steps] 将 trace_config 的 request_queue_csv 指向 "
            f"{output}"
            "（turn-0 前缀已折入 prefill 的 KV 复用单口径；文件名 recompute "
            "字样为跨文件冻结接口，历史命名）"
        )
    except BaseException:
        for handle in (fin, fout, scout):
            if handle is not None and not handle.closed:
                handle.close()
        for staged in (queue_staged, sidecar_staged, provenance_staged):
            if os.path.exists(staged):
                os.unlink(staged)
        raise



if __name__ == "__main__":
    main()
