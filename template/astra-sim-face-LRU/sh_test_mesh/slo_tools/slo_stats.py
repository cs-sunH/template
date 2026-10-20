#!/usr/bin/env python3
"""WP3 SLO 统计后处理（纯离线脚本，标准库实现）。

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
  session       T_session=末轮 completion−首轮 arrival−Σ(human+tool)
                （initiator 口径：行上等待=该轮之后的 gap，Σ 只累加除
                末轮外的 turns——末轮等待物理在窗口外），P50/P95；
                缺字段→NA 行+计数
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
    # (2026-09-30, ported from the joint repo) R6-U2：ceil 使 deadline 覆盖到 α×T 的上取整格
    # （判定式为 e2e > deadline：恰等于格值不算违规）；int() 截断会
    # 在边界样本上把 deadline 压低一格而翻转判定。冻结 α 为整值（本仓
    # 4/6/8）时 ceil==int、官方产物数值不变；非整 α（如 0.5×奇数 t）
    # 会系统性抬高 deadline 一格（恰等于边界不计违规）。
    return math.ceil(alpha * t_iso)


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
    for row in rows:
        status = row["terminal_status"]
        if status not in TERMINAL_STATUSES:
            fail(f"queue_index={row.get('queue_index')}: 非法终态 {status!r}")
        denominator += 1
        bucket = per_status.setdefault(
            status, {"total": 0, "violating": 0})
        bucket["total"] += 1
        e2e = row.get("e2e_ns")
        if e2e is None:
            no_e2e += 1
            # (2026-09-30) py-slo-E8：先判 e2e 不可得再 continue——E2E
            # 不可得的行不进入判定，也不触发缺桶 fail（request_deadline
            # 的分桶/查表只对进入判定的行执行；缺桶 fail 的对象是
            # "有 e2e 而无基线桶"的行）。
            continue  # 非完成终态：计入分母；E2E 不可得，不进分子
        if status != "completed":
            # (2026-09-30, ported from the joint repo) D2：非 completed
            # 终态显式不进分子——不依赖写端 failed⇒e2e_ns 为 NA 的巧合
            # （rejected/dropped 行携带 e2e 时旧实现会误计分子）。
            continue
        deadline = request_deadline(row, alpha, prefill_edges,
                                    decode_edges, t_isolated)
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

def _bucket_label(edges: list[float], idx: int) -> str:
    """桶 i 的区间标签（真实覆盖口径，L17 2026-09-29 标签修正）。

    实现口径（bucket_index，2026-09-05 裁决）不变：桶 i（非末桶）覆盖
    edges[i] < x <= edges[i+1]——内边界值归左桶，即除首桶外左端**开**；
    首桶左端含 edges[0]（x < edges[0] 报错）。旧标签 "[a,b] inclusive"
    声称左端含入，与实现不符——本函数按真实口径出标签：
    首桶 [a,b]、内部桶 (a,b]、末桶 (a,+inf)（单桶表即 [a,+inf)）。
    """
    last = idx == len(edges) - 2
    left = "[" if idx == 0 else "("
    if last:
        return f"{left}{edges[idx]:g},+inf)"
    return f"{left}{edges[idx]:g},{edges[idx + 1]:g}]"


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
            # 末桶无上限（吸收 x > edges[-1]）——与 campaign BucketGrid 对齐；
            # 标签按真实覆盖口径出（L17，2026-09-29：内边界左端排他，
            # 见 _bucket_label）。
            _bucket_label(prefill_edges, pb),
            _bucket_label(decode_edges, db),
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

def _strict_manifest_ns(value, field: str, where: str) -> int:
    """manifest 透传 ns 字段的严格解析（human_time_ns/tool_time_ns）。

    (2026-09-30, ported from the joint repo) R6-U1：manifest 是类型化
    JSON——bool 一律拒绝（True 会按 1ns 计入 Σ(human+tool)）；非整数
    float 拒绝（int() 静默截断记账）；字符串拒绝（与 token manifest/
    turn_index 的 isinstance int 口径一致，非 CSV 文本通道）。负值由
    parse_ns 同款下界口径拒绝（manifest 透传的等待时长不允许为负）。
    非法值走 SloToolError（fail-closed RC=2，带字段名与定位），不得裸
    traceback 以退出码 1 崩出。整数值 float 接受（与 require_param_int
    同型）。
    """
    if isinstance(value, bool):
        fail(f"{where}: 字段 {field} 为布尔值（{value!r}）——ns 字段拒绝 bool")
    if isinstance(value, int):
        if value < 0:
            fail(f"{where}: 字段 {field} 为负数（{value}）——ns 字段不允许负值")
        return value
    if (isinstance(value, float) and math.isfinite(value)
            and value == int(value)):
        if value < 0:
            fail(f"{where}: 字段 {field} 为负数（{value}）——ns 字段不允许负值")
        return int(value)
    fail(f"{where}: 字段 {field} 不是整数纳秒（实得 {value!r}）")


def cmd_session(args: argparse.Namespace, ctx=None) -> int:
    if ctx is not None:
        rows = ctx.rows
        manifest = ctx.request_manifest
    else:
        rows = load_rows(args.run_dir)
        manifest = load_request_manifest(args.run_dir, args.request_manifest)
    per_request: dict[str, dict] = {}
    for entry in manifest_requests(manifest):
        if not entry.get("request_id"):
            continue
        key = str(entry["request_id"])
        # (2026-09-30, ported from the joint repo) R6-U6：request_id 是与
        # request_metrics.csv 连接的键：重复时拒绝静默取后值（与
        # load_token_manifest 的唯一性 fail-closed 同口径）。
        if key in per_request:
            fail(f"per-request manifest request_id 重复：{key}")
        per_request[key] = entry
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
        # H5①（2026-09-29，initiator 口径修正）：trace 冻结语义 = "turn j
        # 的 human/tool 等待记在 turn j 自己行上"（derive_20_first_30_
        # seconds.py: "the gap after turn j is the human_time or tool_time
        # recorded on turn j's own row"），物理上落在
        # [completion(j), arrival(j+1)]——只有 j ≤ n−2 的等待落在会话窗口
        # [first_arrival, last_completion] 内。Σ(human+tool) 因此只累加
        # 除末轮外的 turns（按 turn_index 排序后排除末轮）；末轮行上字段
        # 无论在场/缺席/取值如何都不进求和、不计 soft_missing 也不构成
        # 缺数据（其等待定义在窗口外，对 T_session 无影响）。单轮会话
        # turns[:-1] 为空 → Σ=0，T_session=末轮 completion−首轮 arrival。
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
                # null = 0-gap（该轮之后无等待，属真实零贡献）——对本会话
                # Σ(human+tool) 的贡献为 0，不是缺数据（缺数据=键缺席，
                # 上一分支）。计入 soft_missing 留痕。initiator 口径下
                # 旧注记"turn-0 窗口内无前向 gap"不再成立（turn-0 行上的
                # 等待是 turn-0 之后的 gap，在窗口内）；0-gap 语义不变。
                soft_missing.append(
                    f"{turn.get('request_id')}:human/tool=0(0-gap)")
                continue
            # trace 中 human_time/tool_time 逐行互斥：缺侧按 0 计。
            # (2026-09-30, ported from the joint repo) R6-U1：透传值严格
            # 解析，拒 bool/非整值 float/str/负值（见 _strict_manifest_ns）。
            turn_where = (f"session={session_id} "
                          f"request={turn.get('request_id')}")
            if human is None:
                soft_missing.append(f"{turn.get('request_id')}:human_time_ns=0")
            else:
                human_tool_total += _strict_manifest_ns(
                    human, "human_time_ns", turn_where)
            if tool is None:
                soft_missing.append(f"{turn.get('request_id')}:tool_time_ns=0")
            else:
                human_tool_total += _strict_manifest_ns(
                    tool, "tool_time_ns", turn_where)
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
            # H5① 新口径下的负值判据不变（窗口内净等待不可能为负），
            # 文案与 initiator 口径一致：Σ 只含前 n−1 轮的窗口内等待。
            fail(f"session {session_id}: T_session 为负"
                 f"（{last_completion}-{first_arrival}-{human_tool_total}；"
                 f"Σ(human+tool) 只含除末轮外的窗口内等待）"
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
    print(f"[slo-stats] session: NA 会话 {na_sessions}（缺字段标记 "
          f"{missing_field_rows} 条，即 CSV 列 missing_field_count——"
          f"按标记计非按行计，一行可计多条）——缺 manifest 透传字段"
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
    # (2026-09-30) py-slo-E2 流式化 + R6-U7：归并结果单遍流式消费
    # （sorter.sorted_iter() 直读，同刻归并/去重经一步 lookahead 生成器
    # 完成——每个变点延迟到下一个不同时刻出现才 yield，等价旧
    # series[-1]=(time, current) 的替换写法），不再
    # `events = list(sorter.sorted_iter())` 整读物化——10 万行输入下全量
    # list() 物化峰值 ~158MB，变点数恒 ≤ 事件数。无 --bucket-ns 时本
    # 生成器直接喂 write_csv（逐行直写输出，峰值内存 = 一个 pending 元组）。
    replay = {"current": 0}

    def _change_points():
        pending = None  # (time, inflight)：同刻事件尚未结束时暂不产出
        for time, delta in sorter.sorted_iter():
            replay["current"] += delta
            if pending is not None and pending[0] == time:
                pending = (time, replay["current"])
            else:
                if pending is not None:
                    yield pending
                pending = (time, replay["current"])
        if pending is not None:
            yield pending

    series = _change_points()
    bucket = args.bucket_ns
    if bucket:
        # (2026-10-01, ported from the face repo) 值校验族纪律
        # （EXIT_FAIL_CLOSED=2）：裸 int() 的 traceback 退出码是 1，与本
        # 族"值错误恒 exit 2"不一致。
        try:
            bucket = int(bucket)
        except ValueError:
            fail("--bucket-ns 必须为正整数")
        if bucket <= 0:
            fail("--bucket-ns 必须为正整数")
        # L17（2026-09-29）：分桶输出补发零事件桶。事件间在途值分段常值，
        # 无事件桶的在途值 = 上一事件值延续（carry）——旧行为整桶丢弃使
        # 下游"按行求均值"把无事件区段漏计为零样本。桶值口径不变（桶内
        # 事件峰值与桶起点延续值的较大者，见下），仅补齐
        # [首事件桶, 末事件桶] 内的全部无事件桶；输入侧与产出侧均为
        # 生成器流式（py-slo-E2）：桶字典 O(事件桶数)，与 span/桶长解耦。
        idx_values: dict[int, int] = {}  # 桶号 -> 事件桶代表值
        idx_tail: dict[int, int] = {}    # 桶号 -> 桶内末事件后的在途值
        carry = 0  # 上一锚点行值延续到本桶起点的携带在途值（跨桶区段）
        for time, inflight in series:
            idx = time // bucket
            if idx in idx_values:
                # 桶内真实峰值含「上一桶末事件值延续到本桶的区段」：
                # 携带值大于桶内事件值时只对桶内事件取 max 会系统性低估。
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
    # 重放终值在写流耗尽后才可得（流式化后 stderr 留痕移至输出之后，
    # CSV 字节序列与旧物化实现逐字节一致）。
    if missing:
        print(f"[slo-stats] backlog: {missing} 行 arrival_ns=NA 被跳过（计数）",
              file=sys.stderr)
    if replay["current"] != 0:
        print(f"[slo-stats] backlog: 事件重放后 in_flight={replay['current']}≠0"
              f"（未 drain 或事件缺失）", file=sys.stderr)
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
    if max_p50 == 0 or max_p99 == 0:
        # E6（2026-09-26）：归一化分母（组内 max 分位）为零时按
        # SloToolError 语义 fail-closed（exit 2）——此前 ZeroDivisionError
        # 以 traceback 冒泡为 exit 1，非本工具集的失败语义。
        fail(f"normalized: 归一化分母为零（组内 max p50={max_p50}，"
             f"max p99={max_p99}）——组内分位全零/退化，无法归一化")
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
            if not rows:
                fail(f"scan-export：{run_dir} 的 request_metrics 无数据行，"
                     f"violation 分母为 0")
            den = 0
            num = 0
            for row in rows:
                den += 1
                e2e = row.get("e2e_ns")
                # (2026-10-01) 与 cmd_violation 的 E8 修复同语义：deadline
                # 的分桶/查表只对 E2E 可得（进入判定）的行执行——无 e2e
                # 行不得触发缺桶 fail（缺桶 fail 的对象是"有 e2e 而无基
                # 线桶"的行）。
                if e2e is None:
                    continue
                # (2026-10-01, D2 对齐)非 completed 终态不进 violation
                # 分子——cmd_violation 已显式排除,本导出路径此前漏同步
                # (产出侧 drain 完备断言兜底:全部行 completed 才出表,
                # 此行把两路径口径钉成同文,防御未来断言松动)。
                if row.get("terminal_status") != "completed":
                    continue
                deadline = request_deadline(row, alpha, prefill_edges,
                                            decode_edges, t_isolated)
                if e2e > deadline:
                    num += 1
            rate = num / den
            # 语义权威（slo_tools/README.md）：goodput = tput ×
            # (1 − violation_rate)；tput 不可得（span≤0）时如实 NA，
            # 不得以 drain 断言下的无量纲分数 (1−rate) 冒充 rps 量纲。
            goodput = (tput * (1 - rate)) \
                if (tput is not None and rate is not None) else None
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
                                       "−Σ(human+tool)（initiator 口径：Σ 不含"
                                       "末轮行上等待）；P50/P95")
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
