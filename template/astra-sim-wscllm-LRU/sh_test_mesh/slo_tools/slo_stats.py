#!/usr/bin/env python3
"""WP3 SLO 统计后处理（纯离线脚本，标准库实现）。

同步基线注记（2026-09-26）：本文件与 face/wscllm 家族同名脚本已按仓分叉，
不再跨仓逐字节相同；跨仓同步义务仅覆盖 slo_common 框架契约的调用面。

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
import math
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

def _range_label_text(edges: list[float], idx: int) -> str:
    """range_label 的纯实现（格式化溢出由调用侧统一包裹）。"""
    if idx == 0:
        if len(edges) <= 2:
            return f"[{edges[0]:g},+inf)"
        return f"[{edges[0]:g},{edges[1]:g}]"
    if idx >= len(edges) - 2:
        return f"({edges[idx]:g},+inf)"
    return f"({edges[idx]:g},{edges[idx + 1]:g}]"


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
        # 与主档去重（保持顺序）——重复分位会重复排序取样并输出同名列。
        seen = set(pcts)
        for pct in extras:
            if pct not in seen:
                seen.add(pct)
                pcts.append(pct)
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


def violation_counts(rows: list[dict], alpha: float,
                     prefill_edges: list[float], decode_edges: list[float],
                     t_isolated: dict[tuple[int, int], int]
                     ) -> tuple[int, int, int, dict[str, dict[str, int]]]:
    """violation 分子/分母的单一实现（cmd_violation 与 cmd_scan_export
    共用；D2 修复：scan-export 链此前自带一份无 terminal_status 门的
    循环——非 completed 且携带 e2e 的行会被计入分子，靠 drain 断言事后
    拦截、潜伏）。口径：分母 = 全部终态行；分子 = completed 且 e2e 超
    Deadline(r)；非 completed 终态计入分母不进分子（显式判定，不依赖
    写端 failed⇒e2e_ns 为 NA 的巧合）。rows_without_e2e = e2e 为 NA 的
    行数。返回 (numerator, denominator, rows_without_e2e,
    per_terminal_status)。
    """
    den = 0
    num = 0
    no_e2e = 0
    per_status: dict[str, dict[str, int]] = {}
    for row in rows:
        status = row["terminal_status"]
        if status not in TERMINAL_STATUSES:
            fail(f"queue_index={row.get('queue_index')}: 非法终态 {status!r}")
        den += 1
        bucket = per_status.setdefault(status, {"total": 0, "violating": 0})
        bucket["total"] += 1
        e2e = row.get("e2e_ns")
        deadline = request_deadline(row, alpha, prefill_edges,
                                    decode_edges, t_isolated)
        if e2e is None:
            no_e2e += 1
            continue  # e2e 不可得：计入分母；不进分子
        if status != "completed":
            continue  # 非 completed 终态计入分母；不进分子（显式判定）
        if e2e > deadline:
            num += 1
            bucket["violating"] += 1
    return num, den, no_e2e, per_status


def cmd_violation(args: argparse.Namespace) -> int:
    manifest = load_slo_manifest(args.manifest or default_manifest_path())
    alpha = require_param_number(manifest, args.alpha_name)
    prefill_edges, decode_edges = require_bucket_edges(manifest)
    t_isolated = load_t_isolated(Path(args.t_isolated))
    rows = load_rows(args.run_dir)
    # 判定列代理守卫：数据驱动门（assert_no_proxy_columns）仅在
    # load_t_isolated（T_isolated 表 fieldnames）；request_metrics 为
    # 冻结 schema、恒含 first_token_ns/first_token_source 列，判定路径
    # 只消费 e2e/长度/终态列，不读 proxy 列（2026-09-30 审查清理：原在
    # 此处对源码常量元组复检一遍属恒真仪式，删除）。

    numerator, denominator, no_e2e, per_status = violation_counts(
        rows, alpha, prefill_edges, decode_edges, t_isolated)
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
                            "timed_out+failed；非 completed 终态计入"
                            "分母不进分子",
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
        # 溢出守卫（R2 复核补齐，与 range_label 同族）：畸形巨整数的
        # int/int 真除即 OverflowError，走 fail-closed 出口而非裸栈。
        try:
            _ = float(t_iso) if t_iso else 0.0
        except OverflowError:
            fail("T_isolated 值超出浮点可表示范围"
                 f"（bucket=({pb},{db}), t_iso={t_iso!r}）——manifest 含"
                 f"畸形巨值，拒绝产出 bucket-stats")

        def range_label(edges: list[float], idx: int) -> str:
            # 溢出守卫（2026-10-01 对齐）：巨整数边界（如 10**400）的
            # f"{edge:g}" 格式化即 OverflowError——包裹为 fail-closed，
            # 不让 bucket-stats 以裸栈/退出码 1 崩溃。
            try:
                return _range_label_text(edges, idx)
            except OverflowError:
                fail(f"分桶边界超出浮点可表示范围"
                     f"（edges={edges!r}, idx={idx}）——manifest 分桶参数"
                     f"含畸形巨值，拒绝产出 bucket-stats")

        out_rows.append((
            pb, db,
            range_label(prefill_edges, pb),
            range_label(decode_edges, db),
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
    # R6-U6（2026-09-29 face 移植，2026-09-30 本仓同步）：request_id 是
    # 与 request_metrics.csv 连接的键：重复时拒绝静默取后值（静默
    # last-wins 会把 human/tool 透传记到错误轮次上、污染 T_session 且无
    # 留痕——与 validated_request_rows 对 queue_index 重复即 fail 同口径）。
    per_request: dict[str, dict] = {}
    for entry in manifest_requests(manifest):
        key = str(entry.get("request_id")) if entry.get("request_id") else None
        if key is None:
            continue
        if key in per_request:
            fail(f"per-request manifest request_id 重复：{key}")
        per_request[key] = entry
    sessions: dict[str, list[dict]] = {}
    empty_id_rows = 0
    for row in rows:
        session_id = row.get("session_id")
        if not session_id:
            # 空/缺失 session_id 的行不并入任何会话（旧行为全部并入 "NA"
            # 伪会话——跨真实会话合并，把不同会话的轮次串进同一 T_session
            # 样本、污染分位）；显式排除 + 计数留痕。
            empty_id_rows += 1
            continue
        sessions.setdefault(session_id, []).append(row)
    if not sessions:
        fail("session：request_metrics 无行（或全部行 session_id 为空）")
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
        # H5①（2026-09-29 face 修正，2026-09-30 本仓同步；initiator 口径）
        # ：trace 冻结语义 = "turn j 的 human/tool 等待记在 turn j 自己
        # 行上"（derive_20_first_30_seconds.py: "the gap after turn j is
        # the human_time or tool_time recorded on turn j's own row"），
        # 物理上落在 [completion(j), arrival(j+1)]——只有 j ≤ n−2 的等待
        # 落在会话窗口 [first_arrival, last_completion] 内。Σ(human+tool)
        # 因此只累加除末轮外的 turns（按 turn_index 排序后排除末轮）；
        # 末轮行上字段无论在场/缺席/取值如何都不进求和、不计
        # soft_missing 也不构成缺数据（其等待定义在窗口外，对 T_session
        # 无影响）。单轮会话 turns[:-1] 为空 → Σ=0，T_session=
        # 末轮 completion−首轮 arrival。
        for turn in turns[:-1]:
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
            # manifest 透传值必须为整数（bool 是 int 子类须显式排除）：
            # float 静默截断、str 裸崩、bool 计 1ns 都会污染 T_session
            # ——fail-closed 拒绝。
            for name, value in (("human_time_ns", human),
                                ("tool_time_ns", tool)):
                if value is None:
                    continue
                if isinstance(value, bool) or not isinstance(value, int):
                    fail(f"request {turn.get('request_id')}: manifest "
                         f"{name} 必须为整数纳秒（实得 {value!r}，"
                         f"类型 {type(value).__name__}）")
            if human is None:
                soft_missing.append(f"{turn.get('request_id')}:human_time_ns=0")
            else:
                human_tool_total += human
            if tool is None:
                soft_missing.append(f"{turn.get('request_id')}:tool_time_ns=0")
            else:
                human_tool_total += tool
        missing_field_rows += len(missing)
        if missing:
            na_sessions += 1
            out_rows.append((session_id, len(turns),
                             arrivals[0] if arrivals else NA,
                             completions[-1] if completions else NA,
                             NA, NA, len(missing), ";".join(missing[:5])))
            continue
        # M6-06：统一为 README/NA 分支口径——首轮 arrival/末轮 completion
        # （轮序）。此前正常分支用 min/max，乱序会话下两分支偏差。
        first_arrival = arrivals[0]
        last_completion = completions[-1]
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
    if empty_id_rows:
        print(f"[slo-stats] session: {empty_id_rows} 行 session_id 为空"
              f"——显式排除（不并入任何会话、不进分位）", file=sys.stderr)
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
    # M6-05：末态非零/负在途此前仅告警且照常写出（负值可入
    # slo_backlog.csv）——与同目录 violation/scan 的 fail(...) 口径对齐，
    # 事件不一致的时序拒绝输出。
    if any(inflight < 0 for _, inflight in series):
        fail(f"backlog：时序出现负在途（completion 早于 arrival？）——"
             f"事件不一致，拒绝输出")
    if current != 0:
        fail(f"backlog：事件重放末态 in_flight={current}≠0（存在未 drain "
             f"的 arrival 或缺失 completion）——事件不一致，拒绝输出")
    bucket = args.bucket_ns
    if bucket:
        try:
            bucket = int(bucket)
        except (TypeError, ValueError):
            # 非 "NA" 字符串（如 abc）在此裸 ValueError traceback——包成
            # fail-closed SloToolError，走 run_main 统一退出码。
            fail(f"--bucket-ns 必须为整数（实得 {args.bucket_ns!r}）")
        if bucket <= 0:
            fail("--bucket-ns 必须为正整数")
        # 分桶输出补发零事件桶。事件间在途值分段常值，无事件桶的在途值 =
        # 上一事件值延续（carry）——旧行为整桶丢弃使下游"按行求均值"把
        # 无事件区段漏计为零样本。桶值口径：桶内事件峰值与桶起点延续值的
        # 较大者——桶内只发生完成事件时真实峰值=携入值，只对桶内事件取
        # max 会系统性低估。仅补齐 [首事件桶, 末事件桶] 内的全部无事件桶；
        # 生成器流式产出，内存 O(事件桶数)，与 span/桶长解耦。
        idx_values: dict[int, int] = {}  # 桶号 -> 事件桶代表值
        idx_tail: dict[int, int] = {}    # 桶号 -> 桶内末事件后的在途值
        carry = 0  # 上一锚点行值延续到本桶起点的携带在途值（跨桶区段）
        for time, inflight in series:
            idx = time // bucket
            if idx in idx_values:
                idx_values[idx] = max(idx_values[idx], inflight)
            else:
                idx_values[idx] = max(carry, inflight)
            idx_tail[idx] = inflight
            carry = inflight
        first_idx, last_idx = min(idx_values), max(idx_values)

        def _bucketed():
            # 零事件桶代表值 = 上一事件值的跨桶延续（事件间在途分段常值；
            # 注意取末事件后的在途值，而非前一事件桶的峰值代表值）。
            tail = 0
            for idx in range(first_idx, last_idx + 1):
                if idx in idx_values:
                    yield (idx * bucket, idx_values[idx])
                    tail = idx_tail[idx]
                else:
                    yield (idx * bucket, tail)
        series = _bucketed()
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
    # Report 4.3 fix（2026-09-29 face 修正，2026-09-30 本仓同步）：旧
    # `... / p99_full if p99_full else 0.0` 折叠分支把全零 p99_full 变成
    # relative=0.0 并判 warmup_stable=true 静默放行。全零 P99 E2E 物理
    # 不可达（每个 completed 请求 e2e 恒正），只能来自损坏的指标输入，
    # 稳定性判定不可判定 → 经本模块 fail() 通道 fail-closed，不静默通过。
    if not p99_full:
        fail("warmup：p99_full_ns == 0——全零 P99 E2E 物理不可达，只能来自"
             " 损坏的指标输入（fail-closed，不判 warmup_stable）")
    relative = abs(p99_full - p99_trimmed) / p99_full
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
            fmt_ratio(r["p50"] / max_p50 if max_p50 else None),
            fmt_ratio(r["p99"] / max_p99 if max_p99 else None),
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
        if not math.isfinite(lam):
            # INPUT-01：float() 接受 "inf"/"nan"/"1e999"——NaN 参与比较
            # 恒假、Inf 使 λ 缩放列失义，一律 fail-closed。
            fail(f"--point λ 必须为有限数值：{spec!r}")
        if lam <= 0:
            fail(f"--point λ 必须为正：{spec!r}")
        points.append((Path(run_dir_text), lam))
    if not points:
        fail("scan-export：至少需要一个 --point RUN_DIR=LAMBDA")
    if args.manifest and not args.t_isolated:
        # 显式报错(2026-09-30 审查清理:此前 --manifest 在缺 --t-isolated
        # 时被静默忽略,调用方误以为 manifest 参与了计算)。
        fail("scan-export：--manifest 仅与 --t-isolated 联合使用（violation/"
             "goodput 列的参数来源）；缺 --t-isolated 时本命令不读任何 "
             "manifest——请去掉 --manifest 或补齐 --t-isolated")
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
            # D2 修复：与 cmd_violation 共用 violation_counts 单一实现
            # （分子显式排除非 completed；此前本链的裸循环只认
            # "e2e 非 NA 即可进分子"，与 violation 口径分叉）。
            num, den, _, _ = violation_counts(
                rows, alpha, prefill_edges, decode_edges, t_isolated)
            rate = num / den
            # F7/M6-02：goodput = tput×(1−violation_rate)（README 公式，
            # 单位 rps）。旧实现 (n_completed/den)×(1−rate) 在 drain 完备
            # 断言（input==completed）下完成率因子恒 1，tput 从未参与，
            # 列名 goodput_rps 名不副实。tput 不可得（span≤0/无完成）→ NA。
            goodput = (tput * (1 - rate)) if (tput is not None and den) \
                else None
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
                   help="可选时序分桶（桶值=桶内事件峰值与桶起点携入值的"
                        "较大者，零事件桶补行；默认逐事件）")
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
