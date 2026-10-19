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

Usage: derive_20_first_30_seconds.py <source> [recompute_queue]
                                      [canonical_sidecar] [window_ns]
                                      [arrival_scale]
(source is required and must be passed explicitly -- there is no built-in
default source path; window_ns defaults to 30e9; the canonical sidecar is
written next to the recompute queue -- derived from the queue path via the
_request_queue_recompute.csv -> _canonical_sidecar.csv sibling-name
convention (2026-10-01), or pass it explicitly.  arrival_scale defaults
to 1.0 and must be a positive
finite float: it divides ONLY the turn-0 session_arrival_time_ns column
(t0 / arrival_scale, i.e. load x arrival_scale).  The exact quotient is
then rounded half-up to the nearest 1000 ns grid point, because the
official
loader (generate_trace.load_request_queue) fail-closes on any timing not
divisible by 1000 ns; the deviation vs the exact quotient is <= 500 ns
(== 500 only at exact midpoints, which round up per 四舍五入),
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
import os
import sys
import tempfile

OUTPUT = "astra_compute_20_first_30_seconds_request_queue_recompute.csv"
# (2026-10-01) the bare SIDECAR constant is removed: the default sidecar is
# derived from the queue path via the sibling-name convention (see main),
# which yields this same name for the default OUTPUT.
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


def _staging_path(target: str) -> str:
    """S-5 (2026-10-01): 目标文件旁的唯一临时名（同目录隐藏前缀 + 随机后缀，
    对齐 config_resolver._write_text_atomically 的 NamedTemporaryFile 做法）。

    取代原先确定性的 `<name>.tmp`：两个物化进程并发写同一目录时旧名必然
    互踩——B 进程的 open("w") 截断发生在 A 进程 os.replace 之前时，A 可把
    B 的半截内容发布成"完整"产物。唯一名后写者间不再共享临时 inode，读者
    侧 os.replace 原子性不变。mkstemp 只负责占名建空文件，正文仍由调用方
    以文本模式重开写入（csv 流式 writer 需要 newline="" 语义）。"""
    directory = os.path.dirname(os.path.abspath(target))
    fd, staged = tempfile.mkstemp(
        dir=directory, prefix=f".{os.path.basename(target)}.", suffix=".tmp")
    os.close(fd)
    return staged


def fnv1a64_cont(h: int, data: bytes) -> int:
    """FNV-1a 64 continuation over one chunk of raw bytes.

    Byte-for-byte identical to the C++ reader implementation
    (WindowedTraceReader.cc): offset basis 14695981039346656037, prime
    1099511628211, per byte h = (h ^ byte) * prime, all mod 2^64.
    """
    for byte in data:
        h = ((h ^ byte) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return h


def queue_provenance(queue_path: str, generator_version: str) -> dict:
    """Provenance record of a materialized queue, computed from the WRITTEN
    file bytes (exactly what the C++ reader re-derives and compares).

    Definitions mirror the reader's index pass byte-for-byte: lines split on
    b"\n" (a trailing "\r" from the CSV writer stays in the raw bytes for
    hashing/counting here; the C++ reader strips it from parsed data rows
    and column semantics since 2026-09-30); the first
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
        # (2026-10-01) mirror the C++ reader's entry normalization: strip
        # one trailing '\r' BEFORE the blank-line check so a CRLF blank
        # line ("\r") is skipped here too; hash/byte counters above stay
        # on the raw bytes.
        if line.endswith(b"\r"):
            line = line[:-1]
        if not line:
            return
        if not header_seen:
            header_seen = True
            return
        fields = line.split(b",")
        # 与 C++ 索引趟同口径（WindowedTraceReader.cc）：request_id 首字符
        # 为 '#' 的注释行不计数——否则含注释行的队列会使 sidecar 的
        # data_rows/turn0_count 与 C++ 重算值不等，provenance 门误杀。
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


def main() -> None:
    if len(sys.argv) < 2:
        # infra-8 (2026-09-26): the former hardcoded absolute default source
        # path was removed -- the source CSV is required and must be passed
        # explicitly (request-neutral fail-closed entry).
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
    # (2026-10-01) 输出路径实参加固:显式传入的空串不再是"回落默认常量"
    # 的静默信号——sidecar="" 曾在 queue 已原子发布后才崩(FileNotFoundError
    # 裸 traceback),留下无 sidecar/无 provenance 的半套产物;空串现在
    # 是 usage 错误,在任何文件创建之前 exit 2。
    def _explicit_path(index: int, what: str):
        if len(sys.argv) <= index:
            return None
        value = sys.argv[index]
        if not value:
            print(
                f"{what} was passed as an empty string; pass an explicit "
                f"path or omit the argument",
                file=sys.stderr,
            )
            sys.exit(2)
        return value

    output = _explicit_path(2, "recompute_queue") or OUTPUT
    sidecar_explicit = _explicit_path(3, "canonical_sidecar")
    if sidecar_explicit is not None:
        sidecar = sidecar_explicit
        # (2026-10-01) plan_materializer joins the sidecar by the sibling
        # name convention; an explicit path that breaks it silently degrades
        # the manifest request_type/prefix_len passthrough to unknown/NA.
        # A loud hint keeps that failure mode visible at materialization
        # time instead of only in the postprocess instructions.
        if output.endswith("_request_queue_recompute.csv"):
            derived = (output[:-len("_request_queue_recompute.csv")] +
                       "_canonical_sidecar.csv")
            if os.path.abspath(sidecar) != os.path.abspath(derived):
                print(
                    "[derive] note: explicit canonical_sidecar {} breaks the "
                    "sibling-name convention ({}); plan_materializer joins "
                    "the sidecar by that convention and the manifest "
                    "request_type/prefix_len passthrough will degrade".format(
                        sidecar, derived),
                    file=sys.stderr)
    elif output.endswith("_request_queue_recompute.csv"):
        # (2026-10-01) sidecar 缺省改为按 queue 路径推导(兄弟命名约定,
        # plan_materializer._sidecar_path_for_queue 的 load-bearing join):
        # 自定义 queue 路径此前把 sidecar 写到 CWD 的 30s 缺省名上,断掉
        # manifest 的 request_type/prefix_len 透传链且污染仓根。默认
        # OUTPUT 推导出的名字与旧常量逐字节相同。
        sidecar = (output[:-len("_request_queue_recompute.csv")] +
                   "_canonical_sidecar.csv")
    else:
        print(
            "custom recompute_queue path does not carry the "
            "_request_queue_recompute.csv suffix; pass canonical_sidecar "
            "explicitly (plan_materializer joins the sidecar by that "
            "sibling-name convention)",
            file=sys.stderr,
        )
        sys.exit(2)
    if len(sys.argv) > 4:
        # (2026-10-01) 与 arrival_scale/路径实参同款 exit-2 加固:坏值
        # 此前以裸 ValueError traceback(exit 1)收场。
        try:
            window_ns = int(sys.argv[4])
        except ValueError:
            print(
                f"window_ns must be a plain decimal integer, got "
                f"{sys.argv[4]!r}",
                file=sys.stderr,
            )
            sys.exit(2)
        if window_ns < 0:
            print(
                "window_ns must be a non-negative integer, got "
                f"{window_ns}",
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
        # R3-J6 (2026-09-29): 非整秒窗口不再截断描述——旧写法 window_ns //
        # 1e9 把 2.5s 窗口写成 "first-2-seconds"(描述与实际窗口不符)。
        # 整秒仍产整数标签;非整秒用最短精确小数(:g)。
        if window_ns % 1_000_000_000 == 0:
            window_label = str(window_ns // 1_000_000_000)
        else:
            window_label = f"{window_ns / 1_000_000_000:g}"
        description = (
            f"compute_20 first-{window_label}-seconds window; "
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

    # R5-S3 (2026-09-29): 三处文本 open() 补 encoding="utf-8"(缺省随
    # locale 漂移)。R3-J2 (2026-09-29) Python 侧: 队列/canonical sidecar
    # 先写临时文件、流式结束后 os.replace 原子发布(对齐仓内既有原子写范式)
    # ——并发读者/C++ 门禁绝不读到半截产物;provenance sidecar 同款。
    # S-5 (2026-10-01): 临时名唯一化(_staging_path,对齐 config_resolver 的
    # NamedTemporaryFile 做法),取代原先确定性的 `<name>.tmp`——并发两次
    # 物化输入时同名临时文件互踩,半截内容可被对方的 os.replace 原子发布;
    # 异常路径统一 unlink 半截临时文件(不劣于旧实现——旧实现写半途失败
    # 残留 .tmp 垃圾,靠下次运行同名覆盖才消)。
    output_tmp = _staging_path(output)
    sidecar_tmp = _staging_path(sidecar)
    try:
        with open(source, newline="", encoding="utf-8") as fin, \
                open(output_tmp, "w", newline="", encoding="utf-8") as fout, \
                open(sidecar_tmp, "w", newline="", encoding="utf-8") as scout:
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
            for row in reader:
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
                # (2026-10-01) only the columns THIS row writes are
                # checked: turn-0 rows write session_arrival only (the
                # interval cell is empty), so the previous session's stale
                # prev_gap must not be counted against them.
                if prev_arrival % 1000 != 0 or (
                        turn_index > 0 and prev_gap
                        and prev_gap % 1000 != 0):
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
                    # The exact quotient is rounded HALF-UP to the nearest
                    # 1000 ns grid point (2026-10-01: Python 3 round() is
                    # banker's rounding, which at exact midpoints rounded
                    # DOWN, opposite to the documented 四舍五入): the
                    # official loader (generate_trace.load_request_queue)
                    # fail-closes on timing % 1000 != 0, and an arbitrary
                    # scale (e.g. 1.3 over float division) almost never
                    # keeps the quotient on the grid.  Deviation vs the
                    # exact quotient is <= 500 ns (== 500 only at exact
                    # midpoints, which round up); a no-op for scale=1.0 on
                    # microsecond-grid source times.
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
        # (2026-10-01) sidecar 先于 queue 发布:消费方(plan_materializer
        # 的兄弟名 join、C++ provenance 门)以 queue 文件在场为信号,
        # "queue 在而 sidecar 缺"是 fail-open 方向,"sidecar 在而 queue 缺"
        # 无人消费、无害。
        os.replace(sidecar_tmp, sidecar)
        os.replace(output_tmp, output)
    except BaseException:
        for leftover in (output_tmp, sidecar_tmp):
            try:
                os.unlink(leftover)
            except OSError:
                pass
        raise

    print(f"sessions: {n_sessions}")
    print(f"requests: {n_requests}")
    print(f"prefill range: {min_prefill}-{max_prefill}")
    print(f"decode range: {min_decode}-{max_decode}")
    print(f"max in-window arrival: {max_arrival / 1e9:.6f} s")
    # (2026-10-01 标签澄清:turn-0 分量度量"源时间离 1000ns 栅"(写出值
    # 恒经量化在栅上);turn>0 的 interval 列原样写出,离栅值会被 loader
    # 拒载——两类语义都由该计数覆盖。
    print(f"rows with timing not multiple of 1000 ns "
          f"(turn-0 = source grid, turn>0 = written interval): "
          f"{non_1000}")
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
    # R3-J2 (2026-09-29): provenance sidecar 原子发布(临时文件 +
    # os.replace)——C++ 门禁按它 fail-closed,绝不读到半截 JSON。
    # S-5 (2026-10-01): 临时名唯一化(_staging_path,同上),异常路径
    # unlink 半截临时文件。
    provenance_tmp = _staging_path(provenance_path)
    try:
        with open(provenance_tmp, "w", encoding="utf-8") as pout:
            json.dump(provenance, pout, indent=2, sort_keys=True)
            pout.write("\n")
        os.replace(provenance_tmp, provenance_path)
    except BaseException:
        try:
            os.unlink(provenance_tmp)
        except OSError:
            pass
        raise
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
