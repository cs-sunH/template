#!/usr/bin/env python3
"""WP3 SLO 统计后处理（纯离线脚本，五仓逐字节相同，标准库实现）。

输入 = run_dir（含 request_metrics.csv / cpp.log / metrics_manifest.json）。
所有数值参数（α、ε、分桶边界、warm-up 窗口等）一律从
sh_test_mesh/slo_tools/slo_params_manifest.json 读取；value=null（B4 未
推导）即 fail-closed 非零退出，禁止内置示例值。

子命令：
  e2e-stats     P50/P99 主档 E2E（--extra-pct 追加附录档；整数纳秒、
                先分位后转单位）
  violation     T_isolated 表 + α → Deadline(r)=α×T_isolated(bucket(r))，
                violation_rate=分子/分母（分母=全部终态）；proxy/TTFT 字段
                显式断言拒绝进入判定
  bucket-stats  长度分桶（边界=manifest bucket_percentiles）+ slowdown
                =E2E/T_isolated(bucket)
  session       T_session=末轮 completion−首轮 arrival−Σ(human+tool)，
                P50/P95；缺字段→NA 行+计数
  backlog       arrival/completion 事件重建在途请求数时序（CSV）
  warmup        剔除前缀前后 P99 E2E 对比判定（窗口/门限从 manifest）
  normalized    Normalized Time/TPS（0-1，对齐比较组内 max）
  scan-export   多点 run_dir 聚合 → Goodput/Capacity 表（λ,
                violation_rate, tput, drain 完备断言 input==completed）
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

from slo_common import (  # noqa: E402
    NA, REQUEST_METRICS_FILENAME, TERMINAL_STATUSES,
    assert_no_proxy_columns, bucket_index, default_manifest_path,
    detect_repo_variant, emit_json, fail, fmt_ratio, load_request_manifest,
    load_slo_manifest, manifest_requests, nearest_rank_percentile,
    nearest_rank_percentile_many, open_output, parse_int, parse_ns,
    read_request_metrics, require_bucket_edges, require_param_number,
    run_main, validated_request_rows, write_csv, BoundedSorter,
)

MAIN_PCTS = (50, 99)


def add_run_dir_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("run_dir", type=Path,
                        help="运行目录（含 request_metrics.csv/cpp.log）")


def load_rows(run_dir: Path) -> list[dict]:
    records = read_request_metrics(run_dir / REQUEST_METRICS_FILENAME)
    return validated_request_rows(records, run_dir / REQUEST_METRICS_FILENAME)


def completed_e2e(rows: list[dict]) -> list[int]:
    values = [row["e2e_ns"] for row in rows
              if row["terminal_status"] == "completed"
              and row["e2e_ns"] is not None]
    if not values:
        fail("没有可统计的 completed e2e_ns（全部为 NA/非 completed）")
    return values


def add_manifest_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--manifest", type=Path, default=None,
                        help="slo_params_manifest.json（默认：脚本同目录）")


def add_output_arg(parser: argparse.ArgumentParser, name: str) -> None:
    parser.add_argument("-o", "--output", default="",
                        help=f"输出路径（'-'=stdout；缺省写 run_dir/{name}）")


# ---------------------------------------------------------------------------
# e2e-stats
# ---------------------------------------------------------------------------

def cmd_e2e_stats(args: argparse.Namespace, ctx=None) -> int:
    rows = ctx.rows if ctx is not None else load_rows(args.run_dir)
    values = completed_e2e(rows)
    pcts = list(MAIN_PCTS)
    if args.extra_pct:
        try:
            extras = [int(tok) for tok in args.extra_pct.split(",") if tok]
        except ValueError:
            fail(f"--extra-pct 解析失败：{args.extra_pct!r}（示例：90,95）")
        for pct in extras:
            if not 0 < pct <= 100:
                fail(f"--extra-pct 百分位非法：{pct}")
        pcts.extend(extras)
    # 先分位后转单位：分位在整数纳秒样本上取值，ms 只作展示换算。
    # 多分位共享一次有界排序（A4；公式与逐次 nearest_rank_percentile
    # 完全一致——同一 sorted 序列上的同一 index 公式）。
    pct_values = nearest_rank_percentile_many(
        values, [pct / 100.0 for pct in pcts])
    rows_out = []
    for pct, value in zip(pcts, pct_values):
        rows_out.append((f"e2e_p{pct}", "ns", len(values), value,
                         f"{value / 1e6:.3f}"))
    rows_out.append(("n_completed", "requests", len(values), len(values), NA))
    rows_out.append(("n_all_rows", "requests", len(rows), len(rows), NA))
    stream, close = open_output(args.output, "slo_e2e_stats.csv", args.run_dir)
    try:
        write_csv(stream, ("metric", "unit", "n", "value_ns", "value_ms"),
                  rows_out)
        print(f"[slo-stats] percentile_method=nearest_rank; "
              f"main={list(MAIN_PCTS)}; unit conversion after percentile",
              file=sys.stderr)
    finally:
        if close:
            stream.close()
    return 0


# ---------------------------------------------------------------------------
# T_isolated 表
# ---------------------------------------------------------------------------

T_ISOLATED_COLUMNS = ("prefill_bucket_idx", "decode_bucket_idx",
                      "t_isolated_ns")


def load_t_isolated(path: Path) -> dict[tuple[int, int], int]:
    import csv as _csv
    if not path.is_file():
        fail(f"T_isolated 表不存在：{path}（B4 批次 isolated_baseline 产物，"
             f"列：{','.join(T_ISOLATED_COLUMNS)}）")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = _csv.DictReader(handle)
        if reader.fieldnames is None:
            fail(f"{path}: 空文件")
        assert_no_proxy_columns(reader.fieldnames,
                                "T_isolated 表（SLO 判定输入）")
        missing = [c for c in T_ISOLATED_COLUMNS
                   if c not in (reader.fieldnames or [])]
        if missing:
            fail(f"{path}: T_isolated 表缺列 {missing}，要求列序 "
                 f"{list(T_ISOLATED_COLUMNS)}（允许附非 proxy 列）")
        table: dict[tuple[int, int], int] = {}
        for lineno, row in enumerate(reader, start=2):
            where = f"{path}:{lineno}"
            pb = parse_int(row.get("prefill_bucket_idx"),
                           "prefill_bucket_idx", where)
            db = parse_int(row.get("decode_bucket_idx"),
                           "decode_bucket_idx", where)
            t_iso = parse_ns(row.get("t_isolated_ns"), "t_isolated_ns", where)
            if pb is None or db is None or t_iso is None:
                fail(f"{where}: 桶索引/t_isolated_ns 不允许为 NA")
            key = (pb, db)
            if key in table:
                fail(f"{where}: 桶 {key} 重复")
            if t_iso <= 0:
                fail(f"{where}: t_isolated_ns 必须为正（{t_iso}）")
            table[key] = t_iso
    if not table:
        fail(f"{path}: T_isolated 表无数据行")
    return table


def request_deadline(row: dict, alpha: float,
                     prefill_edges: list[float], decode_edges: list[float],
                     t_isolated: dict[tuple[int, int], int]) -> Optional[int]:
    """Deadline(r) = α × T_isolated(bucket(r))；桶缺失 → fail-closed。"""
    prefill_len = row.get("prefill_length")
    decode_len = row.get("decode_length")
    if prefill_len is None or decode_len is None:
        fail(f"queue_index={row.get('queue_index')}: prefill_length/"
             f"decode_length 为 NA，无法分桶（WP1 产物需携带长度列）")
    pb = bucket_index(prefill_edges, prefill_len)
    db = bucket_index(decode_edges, decode_len)
    t_iso = t_isolated.get((pb, db))
    if t_iso is None:
        fail(f"queue_index={row.get('queue_index')}: T_isolated 表缺桶 "
             f"(prefill_bucket={pb}, decode_bucket={db})——先补全孤立基线"
             f"矩阵（B4），不得跳过")
    return int(alpha * t_iso)


def cmd_violation(args: argparse.Namespace) -> int:
    manifest = load_slo_manifest(args.manifest or default_manifest_path())
    alpha = require_param_number(manifest, args.alpha_name)
    prefill_edges, decode_edges = require_bucket_edges(manifest)
    t_isolated = load_t_isolated(Path(args.t_isolated))
    rows = load_rows(args.run_dir)
    # 显式自证：判定只消费 E2E/长度/终态列。
    assert_no_proxy_columns(
        ("e2e_ns", "prefill_length", "decode_length", "terminal_status"),
        "violation 判定列集")

    denominator = 0
    numerator = 0
    no_e2e = 0
    per_status: dict[str, dict[str, int]] = {}
    deadlines: list[int] = []
    for row in rows:
        status = row["terminal_status"]
        if status not in TERMINAL_STATUSES:
            fail(f"queue_index={row.get('queue_index')}: 非法终态 {status!r}")
        denominator += 1
        bucket = per_status.setdefault(
            status, {"total": 0, "violating": 0})
        bucket["total"] += 1
        e2e = row.get("e2e_ns")
        deadline = request_deadline(row, alpha, prefill_edges,
                                    decode_edges, t_isolated)
        deadlines.append(deadline)
        if e2e is None:
            no_e2e += 1
            continue  # 非完成终态：计入分母；E2E 不可得，不进分子
        if e2e > deadline:
            numerator += 1
            bucket["violating"] += 1
    if denominator == 0:
        fail("violation 分母为 0（request_metrics 无终态行）")
    rate = numerator / denominator
    payload = {
        "command": "violation",
        "alpha_name": args.alpha_name,
        "alpha": alpha,
        "violation_numerator": numerator,
        "violation_denominator": denominator,
        "violation_rate": rate,
        "rows_without_e2e": no_e2e,
        "per_terminal_status": per_status,
        "deadline_rule": "Deadline(r)=alpha*T_isolated(bucket(r)); "
                         "bucket=(prefill_length,decode_length) by frozen "
                         "manifest edges",
        "denominator_rule": "全部终态 completed+rejected+dropped+"
                            "timed_out+failed",
        "proxy_guard": "first_token_ns/first_token_source 显式断言"
                       "拒绝进入判定（主规格 §1.2-A）",
    }
    stream, close = open_output(args.output, "slo_violation.json", args.run_dir)
    try:
        emit_json(stream, payload)
    finally:
        if close:
            stream.close()
    return 0


# ---------------------------------------------------------------------------
# bucket-stats
# ---------------------------------------------------------------------------

def cmd_bucket_stats(args: argparse.Namespace) -> int:
    manifest = load_slo_manifest(args.manifest or default_manifest_path())
    prefill_edges, decode_edges = require_bucket_edges(manifest)
    t_isolated = load_t_isolated(Path(args.t_isolated))
    rows = load_rows(args.run_dir)
    buckets: dict[tuple[int, int], dict[str, list]] = {}
    skipped = 0
    for row in rows:
        if row["terminal_status"] != "completed" or row["e2e_ns"] is None:
            skipped += 1
            continue
        prefill_len = row.get("prefill_length")
        decode_len = row.get("decode_length")
        if prefill_len is None or decode_len is None:
            fail(f"queue_index={row.get('queue_index')}: 长度列为 NA")
        pb = bucket_index(prefill_edges, prefill_len)
        db = bucket_index(decode_edges, decode_len)
        entry = buckets.setdefault((pb, db), {"e2e": [], "slowdown": []})
        entry["e2e"].append(row["e2e_ns"])
        t_iso = t_isolated.get((pb, db))
        if t_iso is None:
            fail(f"queue_index={row.get('queue_index')}: T_isolated 表缺桶 "
                 f"({pb},{db})")
        entry["slowdown"].append(row["e2e_ns"] / t_iso)
    if not buckets:
        fail("bucket-stats：没有可分桶的 completed 行")
    out_rows = []
    for (pb, db) in sorted(buckets):
        entry = buckets[(pb, db)]
        n = len(entry["e2e"])
        t_iso = t_isolated[(pb, db)]
        out_rows.append((
            pb, db,
            # 2026-09-05 口径裁决：interior edges 为左桶闭上界（右闭），
            # 末桶无上限（吸收 x > edges[-1]）——与 campaign BucketGrid 对齐
            (f"[{prefill_edges[pb]:g},{prefill_edges[pb + 1]:g} inclusive]"
             if pb < len(prefill_edges) - 2
             else f"[{prefill_edges[pb]:g},+inf)"),
            (f"[{decode_edges[db]:g},{decode_edges[db + 1]:g} inclusive]"
             if db < len(decode_edges) - 2
             else f"[{decode_edges[db]:g},+inf)"),
            n, t_iso,
            nearest_rank_percentile(entry["e2e"], 0.50),
            nearest_rank_percentile(entry["e2e"], 0.99),
            fmt_ratio(nearest_rank_percentile(entry["slowdown"], 0.50)),
            fmt_ratio(nearest_rank_percentile(entry["slowdown"], 0.99)),
            fmt_ratio(sum(entry["slowdown"]) / n),
        ))
    stream, close = open_output(args.output, "slo_bucket_stats.csv",
                                args.run_dir)
    try:
        write_csv(
            stream,
            ("prefill_bucket_idx", "decode_bucket_idx",
             "prefill_range_tokens", "decode_range_tokens", "n_requests",
             "t_isolated_ns", "p50_e2e_ns", "p99_e2e_ns",
             "median_slowdown", "p99_slowdown", "mean_slowdown"),
            out_rows)
        print(f"[slo-stats] bucket-stats: {len(out_rows)} 桶，"
              f"跳过非 completed/无 e2e 行 {skipped}", file=sys.stderr)
    finally:
        if close:
            stream.close()
    return 0


# ---------------------------------------------------------------------------
# session
# ---------------------------------------------------------------------------

def cmd_session(args: argparse.Namespace, ctx=None) -> int:
    if ctx is not None:
        rows = ctx.rows
        manifest = ctx.request_manifest
    else:
        rows = load_rows(args.run_dir)
        manifest = load_request_manifest(args.run_dir, args.request_manifest)
    per_request = {str(r.get("request_id")): r
                   for r in manifest_requests(manifest) if r.get("request_id")}
    sessions: dict[str, list[dict]] = {}
    for row in rows:
        sessions.setdefault(row.get("session_id") or NA, []).append(row)
    if not sessions:
        fail("session：request_metrics 无行")
    out_rows = []
    t_sessions: list[int] = []
    na_sessions = 0
    missing_field_rows = 0
    for session_id in sorted(sessions):
        turns = sorted(sessions[session_id],
                       key=lambda r: (r.get("turn_index") is None,
                                      r.get("turn_index") or 0,
                                      r.get("arrival_ns") or 0))
        missing: list[str] = []
        arrivals = [t["arrival_ns"] for t in turns if t["arrival_ns"] is not None]
        completions = [t["completion_ns"] for t in turns
                       if t["completion_ns"] is not None]
        if not arrivals:
            missing.append("arrival_ns")
        if not completions:
            missing.append("completion_ns")
        if len(arrivals) != len(turns):
            missing.append("arrival_ns(partial)")
        if len(completions) != len(turns):
            missing.append("completion_ns(partial)")
        human_tool_total = 0
        soft_missing: list[str] = []
        for turn in turns:
            entry = per_request.get(str(turn.get("request_id")), {})
            human = entry.get("human_time_ns")
            tool = entry.get("tool_time_ns")
            if "human_time_ns" not in entry and "tool_time_ns" not in entry:
                # 透传字段整体缺席（WP2 之前的 manifest）→ 无法计算 T_session。
                missing.append(f"{turn.get('request_id')}:human_time_ns")
                missing.append(f"{turn.get('request_id')}:tool_time_ns")
                continue
            if human is None and tool is None:
                # B4 语义修正（2026-08-26，S3 异常③）：字段已在场而双侧
                # null = turn-0（会话首行，窗口内无前向 gap）或 0-gap 无
                # 类型延续——对本会话 Σ(human+tool) 的贡献为 0，不是缺数据
                # （缺数据=键缺席，上一分支）。计入 soft_missing 留痕。
                soft_missing.append(
                    f"{turn.get('request_id')}:human/tool=0(turn0/0-gap)")
                continue
            # trace 中 human_time/tool_time 逐行互斥：缺侧按 0 计。
            if human is None:
                soft_missing.append(f"{turn.get('request_id')}:human_time_ns=0")
            else:
                human_tool_total += int(human)
            if tool is None:
                soft_missing.append(f"{turn.get('request_id')}:tool_time_ns=0")
            else:
                human_tool_total += int(tool)
        missing_field_rows += len(missing)
        if missing:
            na_sessions += 1
            out_rows.append((session_id, len(turns),
                             arrivals[0] if arrivals else NA,
                             completions[-1] if completions else NA,
                             NA, NA, len(missing), ";".join(missing[:5])))
            continue
        first_arrival = min(arrivals)
        last_completion = max(completions)
        t_session = last_completion - first_arrival - human_tool_total
        if t_session < 0:
            fail(f"session {session_id}: T_session 为负"
                 f"（{last_completion}-{first_arrival}-{human_tool_total}）"
                 f"——completion/arrival 或 human/tool 字段可疑")
        t_sessions.append(t_session)
        out_rows.append((session_id, len(turns), first_arrival,
                         last_completion, human_tool_total, t_session,
                         len(soft_missing), ";".join(soft_missing[:5])))
    stream, close = open_output(args.output, "slo_session.csv", args.run_dir)
    try:
        write_csv(
            stream,
            ("session_id", "n_turns", "first_arrival_ns",
             "last_completion_ns", "sum_human_tool_ns", "t_session_ns",
             "missing_field_count", "missing_fields_sample"),
            out_rows)
    finally:
        if close:
            stream.close()
    if t_sessions:
        print(f"[slo-stats] session: n={len(t_sessions)} "
              f"P50={nearest_rank_percentile(t_sessions, 0.50)}ns "
              f"P95={nearest_rank_percentile(t_sessions, 0.95)}ns",
              file=sys.stderr)
    print(f"[slo-stats] session: NA 会话 {na_sessions}（缺字段行 "
          f"{missing_field_rows}）——缺 manifest 透传字段"
          f"（human_time_ns/tool_time_ns，WP2）时输出 NA+计数",
          file=sys.stderr)
    return 0


# ---------------------------------------------------------------------------
# backlog
# ---------------------------------------------------------------------------

def cmd_backlog(args: argparse.Namespace, ctx=None) -> int:
    rows = ctx.rows if ctx is not None else load_rows(args.run_dir)
    events: list[tuple[int, int]] = []  # (time, delta)
    missing = 0
    for row in rows:
        arrival = row.get("arrival_ns")
        if arrival is None:
            missing += 1
            continue
        events.append((arrival, +1))
        completion = row.get("completion_ns")
        if completion is not None:
            events.append((completion, -1))
    if not events:
        fail("backlog：没有任何 arrival/completion 事件（arrival_ns 全 NA？）")
    # 同刻先减后加：完成事件先落地，再到达（确定性约定）。
    # A4：排序走 BoundedSorter（比较键 (time, delta) 原样——整数元组
    # 全序，外部归并输出 ≡ list.sort(key=(e[0], e[1]))）。
    sorter = BoundedSorter()
    for event in events:
        sorter.add(event)
    events = list(sorter.sorted_iter())
    series: list[tuple[int, int]] = []
    current = 0
    last_time = None
    for time, delta in events:
        current += delta
        if time == last_time:
            series[-1] = (time, current)
        else:
            series.append((time, current))
            last_time = time
    if missing:
        print(f"[slo-stats] backlog: {missing} 行 arrival_ns=NA 被跳过（计数）",
              file=sys.stderr)
    if current != 0:
        print(f"[slo-stats] backlog: 事件重放后 in_flight={current}≠0"
              f"（未 drain 或事件缺失）", file=sys.stderr)
    bucket = args.bucket_ns
    if bucket:
        bucket = int(bucket)
        if bucket <= 0:
            fail("--bucket-ns 必须为正整数")
        bucketed: list[tuple[int, int]] = []
        for time, inflight in series:
            slot = (time // bucket) * bucket
            if bucketed and bucketed[-1][0] == slot:
                bucketed[-1] = (slot, max(bucketed[-1][1], inflight))
            else:
                bucketed.append((slot, inflight))
        series = bucketed
    stream, close = open_output(args.output, "slo_backlog.csv", args.run_dir)
    try:
        write_csv(stream, ("time_ns", "in_flight"), series)
    finally:
        if close:
            stream.close()
    return 0


# ---------------------------------------------------------------------------
# warmup
# ---------------------------------------------------------------------------

def cmd_warmup(args: argparse.Namespace, ctx=None) -> int:
    if ctx is not None:
        manifest = ctx.slo_manifest
        rows = ctx.rows
    else:
        manifest = load_slo_manifest(args.manifest or default_manifest_path())
        rows = load_rows(args.run_dir)
    window_ns = require_param_number(manifest, "warmup_window")
    threshold = require_param_number(manifest, "warmup_change_threshold")
    if window_ns <= 0 or threshold < 0:
        fail("warmup_window/warmup_change_threshold 必须为正/非负")
    completed = [row for row in rows
                 if row["terminal_status"] == "completed"
                 and row["e2e_ns"] is not None]
    arrivals = [row["arrival_ns"] for row in completed
                if row["arrival_ns"] is not None]
    if not completed or not arrivals:
        fail("warmup：没有 completed+e2e+arrival 的行")
    t0 = min(arrivals)
    full_values = [row["e2e_ns"] for row in completed]
    trimmed = [row["e2e_ns"] for row in completed
               if row["arrival_ns"] is not None
               and row["arrival_ns"] >= t0 + window_ns]
    if not trimmed:
        fail(f"warmup：剔除窗口 {window_ns:.0f}ns 后无剩余样本")
    # A4：两组 P99 各走一次有界排序（公式不变）。
    p99_full = nearest_rank_percentile_many(full_values, [0.99])[0]
    p99_trimmed = nearest_rank_percentile_many(trimmed, [0.99])[0]
    relative = abs(p99_full - p99_trimmed) / p99_full if p99_full else 0.0
    payload = {
        "command": "warmup",
        "window_ns": window_ns,
        "threshold": threshold,
        "p99_full_ns": p99_full,
        "p99_after_warmup_ns": p99_trimmed,
        "abs_delta_ns": abs(p99_full - p99_trimmed),
        "relative_change": relative,
        "n_full": len(full_values),
        "n_after_warmup": len(trimmed),
        "warmup_stable": relative <= threshold,
        "judgment_form": "剔除前缀前后比较 P99 E2E（主规格 §1.4-A）",
    }
    stream, close = open_output(args.output, "slo_warmup.json", args.run_dir)
    try:
        emit_json(stream, payload)
    finally:
        if close:
            stream.close()
    return 0


# ---------------------------------------------------------------------------
# normalized
# ---------------------------------------------------------------------------

def run_throughput(rows: list[dict]) -> tuple[Optional[int], Optional[int],
                                              Optional[float], int, int]:
    completed = [row for row in rows if row["terminal_status"] == "completed"
                 and row["completion_ns"] is not None]
    arrivals = [row["arrival_ns"] for row in rows
                if row["arrival_ns"] is not None]
    completions = [row["completion_ns"] for row in completed]
    n_completed = len(completed)
    if not completions or not arrivals:
        return None, None, None, n_completed, len(rows)
    span = max(completions) - min(arrivals)
    tput = (n_completed / span * 1e9) if span > 0 else None
    return min(arrivals), max(completions), tput, n_completed, len(rows)


def cmd_normalized(args: argparse.Namespace) -> int:
    runs = []
    for run_dir in args.run_dir:
        rows = load_rows(run_dir)
        values = [row["e2e_ns"] for row in rows
                  if row["terminal_status"] == "completed"
                  and row["e2e_ns"] is not None]
        if not values:
            fail(f"{run_dir}: 无 completed e2e_ns，无法归一化")
        p50 = nearest_rank_percentile(values, 0.50)
        p99 = nearest_rank_percentile(values, 0.99)
        t0, t1, tput, n_completed, n_rows = run_throughput(rows)
        variant = detect_repo_variant(run_dir, args.repo_variant)
        runs.append({
            "run_dir": str(run_dir), "repo_variant": variant,
            "p50": p50, "p99": p99, "tput": tput,
            "n_completed": n_completed, "n_rows": n_rows,
            "first_arrival_ns": t0, "last_completion_ns": t1,
        })
    max_p50 = max(r["p50"] for r in runs)
    max_p99 = max(r["p99"] for r in runs)
    tputs = [r["tput"] for r in runs if r["tput"] is not None]
    max_tput = max(tputs) if tputs else None
    out_rows = []
    for r in runs:
        out_rows.append((
            r["run_dir"], r["repo_variant"], r["n_completed"], r["n_rows"],
            r["p50"], r["p99"], fmt_ratio(r["tput"], 9),
            fmt_ratio(r["p50"] / max_p50),
            fmt_ratio(r["p99"] / max_p99),
            fmt_ratio(r["tput"] / max_tput
                      if (r["tput"] is not None and max_tput) else None),
        ))
    stream, close = open_output(args.output, "slo_normalized.csv", None)
    try:
        write_csv(
            stream,
            ("run_dir", "repo_variant", "n_completed", "n_rows",
             "p50_e2e_ns", "p99_e2e_ns", "tput_rps",
             "normalized_time_p50", "normalized_time_p99",
             "normalized_tps"),
            out_rows)
        print("[slo-stats] normalized: 0-1 归一化，对齐比较组内 max"
              "（Normalized Time=分位 E2E/组内 max，Normalized TPS=吞吐/组内 max）",
              file=sys.stderr)
    finally:
        if close:
            stream.close()
    return 0


# ---------------------------------------------------------------------------
# scan-export
# ---------------------------------------------------------------------------

def cmd_scan_export(args: argparse.Namespace) -> int:
    points = []
    for spec in args.point:
        if "=" not in spec:
            fail(f"--point 格式：RUN_DIR=LAMBDA（实得 {spec!r}）")
        run_dir_text, lambda_text = spec.split("=", 1)
        try:
            lam = float(lambda_text)
        except ValueError:
            fail(f"--point λ 非数值：{spec!r}")
        if lam <= 0:
            fail(f"--point λ 必须为正：{spec!r}")
        points.append((Path(run_dir_text), lam))
    if not points:
        fail("scan-export：至少需要一个 --point RUN_DIR=LAMBDA")
    manifest = None
    t_isolated = None
    alpha = None
    prefill_edges = decode_edges = None
    if args.t_isolated:
        manifest = load_slo_manifest(args.manifest or default_manifest_path())
        alpha = require_param_number(manifest, args.alpha_name)
        prefill_edges, decode_edges = require_bucket_edges(manifest)
        t_isolated = load_t_isolated(Path(args.t_isolated))

    out_rows = []
    drain_failures = []
    for run_dir, lam in points:
        rows = load_rows(run_dir)
        variant = detect_repo_variant(run_dir, args.repo_variant)
        n_input = len(rows)
        n_completed = sum(1 for row in rows
                          if row["terminal_status"] == "completed")
        drain_ok = n_input == n_completed
        if not drain_ok:
            drain_failures.append((str(run_dir), n_input, n_completed))
        _, _, tput, _, _ = run_throughput(rows)
        num = den = None
        rate = None
        goodput = None
        if t_isolated is not None:
            den = 0
            num = 0
            for row in rows:
                den += 1
                e2e = row.get("e2e_ns")
                deadline = request_deadline(row, alpha, prefill_edges,
                                            decode_edges, t_isolated)
                if e2e is not None and e2e > deadline:
                    num += 1
            rate = num / den
            goodput = ((n_completed / den) * (1 - rate)) if den else None
        out_rows.append((
            str(run_dir), variant, fmt_ratio(lam), n_input, n_completed,
            "true" if drain_ok else "false",
            num if num is not None else NA,
            den if den is not None else NA,
            fmt_ratio(rate), fmt_ratio(tput, 9), fmt_ratio(goodput, 9),
        ))
    if drain_failures:
        # drain 完备断言（input==completed）先行：不满足则不产出表，
        # 该表不可用于 Capacity 判定。
        for run_dir, n_input, n_completed in drain_failures:
            print(f"[slo-stats] scan-export: drain 完备断言失败 "
                  f"{run_dir}: input={n_input} != completed={n_completed}",
                  file=sys.stderr)
        fail(f"scan-export：{len(drain_failures)} 个点未满足 drain 完备断言"
             f"（input==completed）——拒绝输出聚合表")
    stream, close = open_output(args.output, "slo_scan_export.csv", None)
    try:
        write_csv(
            stream,
            ("run_dir", "repo_variant", "lambda_scale", "input_requests",
             "completed_requests", "drain_ok_input_eq_completed",
             "violation_numerator", "violation_denominator",
             "violation_rate", "tput_rps", "goodput_rps"),
            out_rows)
    finally:
        if close:
            stream.close()
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="slo_stats.py",
        description="WP3 SLO 统计后处理（离线；参数取 slo_params_manifest.json，"
                    "value=null 一律 fail-closed）")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("e2e-stats", help="P50/P99 E2E 主档（--extra-pct 附录档）")
    add_run_dir_arg(p)
    add_output_arg(p, "slo_e2e_stats.csv")
    p.add_argument("--extra-pct", default="",
                   help="附录分位点，逗号分隔（示例：90,95；主文只用 P50/P99）")
    p.set_defaults(func=cmd_e2e_stats)

    p = sub.add_parser("violation", help="violation_rate=分子/分母（分母=全部终态）")
    add_run_dir_arg(p)
    add_manifest_arg(p)
    add_output_arg(p, "slo_violation.json")
    p.add_argument("--t-isolated", required=True,
                   help="T_isolated CSV（B4 产物；列 prefill_bucket_idx,"
                        "decode_bucket_idx,t_isolated_ns）")
    p.add_argument("--alpha-name", default="alpha_main",
                   choices=["alpha_main", "alpha_side_low", "alpha_side_high"],
                   help="α 档位（manifest 参数名）")
    p.set_defaults(func=cmd_violation)

    p = sub.add_parser("bucket-stats", help="长度分桶 + slowdown=E2E/T_isolated")
    add_run_dir_arg(p)
    add_manifest_arg(p)
    add_output_arg(p, "slo_bucket_stats.csv")
    p.add_argument("--t-isolated", required=True,
                   help="T_isolated CSV（B4 产物）")
    p.set_defaults(func=cmd_bucket_stats)

    p = sub.add_parser("session", help="T_session=末轮 completion−首轮 arrival"
                                       "−Σ(human+tool)；P50/P95")
    add_run_dir_arg(p)
    add_output_arg(p, "slo_session.csv")
    p.add_argument("--request-manifest", type=Path, default=None,
                   help="per-request manifest（默认 run_dir/metrics_manifest.json"
                        " 或 cpp.log init 行 manifest_path）")
    p.set_defaults(func=cmd_session)

    p = sub.add_parser("backlog", help="arrival/completion 事件重建在途时序（CSV）")
    add_run_dir_arg(p)
    add_output_arg(p, "slo_backlog.csv")
    p.add_argument("--bucket-ns", default=None,
                   help="可选时序分桶（桶内取 max；默认逐事件）")
    p.set_defaults(func=cmd_backlog)

    p = sub.add_parser("warmup", help="剔除前缀前后 P99 E2E 对比判定")
    add_run_dir_arg(p)
    add_manifest_arg(p)
    add_output_arg(p, "slo_warmup.json")
    p.set_defaults(func=cmd_warmup)

    p = sub.add_parser("normalized", help="Normalized Time/TPS（0-1，组内 max）")
    p.add_argument("run_dir", type=Path, nargs="+",
                   help="参与同一比较组的 run_dir（>=1 个）")
    add_output_arg(p, "slo_normalized.csv")
    p.add_argument("--repo-variant", default=None,
                   help="显式指定 repo_variant（默认读 cpp.log init 行）")
    p.set_defaults(func=cmd_normalized)

    p = sub.add_parser("scan-export", help="多点聚合 → Goodput/Capacity 表")
    add_manifest_arg(p)
    add_output_arg(p, "slo_scan_export.csv")
    p.add_argument("--point", action="append", required=True,
                   metavar="RUN_DIR=LAMBDA",
                   help="扫描点（可重复）：run_dir 与到达率缩放 λ")
    p.add_argument("--t-isolated", default=None, help="T_isolated CSV（可选；"
                   "给出则输出 violation_rate/goodput）")
    p.add_argument("--alpha-name", default="alpha_main",
                   choices=["alpha_main", "alpha_side_low", "alpha_side_high"])
    p.add_argument("--repo-variant", default=None)
    p.set_defaults(func=cmd_scan_export)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(run_main(main))
