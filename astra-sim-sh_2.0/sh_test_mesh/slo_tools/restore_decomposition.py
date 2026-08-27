#!/usr/bin/env python3
"""WP5 restore 三段分解（五仓逐字节相同，标准库实现）。

输入 = cpp.log（full 档）：``[METRIC]`` 行中的
  * type=memory_anchor 记录（subject_id=queue_index, rank, node_id,
    tick_ns；由 C++ 侧 KV transfer 锚点注册——main_online.cc 的
    last_transfer（名字含 "kv" 的最后节点）→ MetricCollector 事件码 7）；
  * type=request 记录（prefill_start_ns/prefill_end_ns 等请求边界）。

每请求：
  restore_start_ns    = min(锚点 tick)   （逐请求归属：subject_id→请求）
  restore_complete_ns = max(锚点 tick)
无任何锚点归属的请求：四个推导字段全 NA 并计数（主规格 WP5 失败处置：
论文须用弱化主张），restore_start/complete 亦为 NA。

三段分解公式 = 主规格 §3.2 逐字符实现（A 类，禁止改动）：

    overlap_window = [prefill_start_ns, prefill_end_ns]
    pre_prefill_restore_ns   = max(0, min(restore_complete_ns, prefill_start_ns) − restore_start_ns)
    hidden_restore_ns        = max(0, min(restore_complete_ns, prefill_end_ns)
                                   − max(restore_start_ns, prefill_start_ns))
    exposed_restore_stall_ns = max(0, restore_complete_ns − prefill_end_ns)
    hidden_ratio             = hidden_restore_ns / (restore_complete_ns − restore_start_ns)

输出 CSV 列名 = EXECUTION_PLAN §2 对应列。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

from slo_common import (  # noqa: E402
    CPP_LOG_NAME, emit_json, fail, fmt_ratio, open_output, parse_ns,
    read_cpp_metric_records, run_main, write_csv,
)

OUTPUT_COLUMNS = ("queue_index", "request_id", "restore_start_ns",
                  "restore_complete_ns", "pre_prefill_restore_ns",
                  "hidden_restore_ns", "exposed_restore_stall_ns",
                  "hidden_ratio")


def restore_segments(restore_start_ns: Optional[int],
                     restore_complete_ns: Optional[int],
                     prefill_start_ns: Optional[int],
                     prefill_end_ns: Optional[int]
                     ) -> tuple[Optional[int], Optional[int], Optional[int],
                                Optional[float]]:
    """§3.2 固定公式（逐字符实现；None → 全 NA 由调用方计数）。"""
    if (restore_start_ns is None or restore_complete_ns is None
            or prefill_start_ns is None or prefill_end_ns is None):
        return None, None, None, None
    pre_prefill = max(
        0, min(restore_complete_ns, prefill_start_ns) - restore_start_ns)
    hidden = max(
        0,
        min(restore_complete_ns, prefill_end_ns)
        - max(restore_start_ns, prefill_start_ns))
    exposed = max(0, restore_complete_ns - prefill_end_ns)
    duration = restore_complete_ns - restore_start_ns
    ratio = (hidden / duration) if duration > 0 else None
    return pre_prefill, hidden, exposed, ratio


def collect(run_dir: Path) -> tuple[list[dict], dict]:
    cpp_log = run_dir / CPP_LOG_NAME
    anchors: dict[int, list[int]] = {}
    requests: dict[int, dict] = {}
    for record in read_cpp_metric_records(cpp_log):
        rtype = record.get("type")
        if rtype == "memory_anchor":
            subject = record.get("subject_id")
            tick = record.get("tick_ns")
            if not isinstance(subject, int) or not isinstance(tick, int):
                fail(f"{cpp_log}: memory_anchor 记录缺整数 subject_id/"
                     f"tick_ns（{record}）")
            anchors.setdefault(subject, []).append(tick)
        elif rtype == "request":
            queue_index = record.get("queue_index")
            if not isinstance(queue_index, int):
                fail(f"{cpp_log}: request 记录缺整数 queue_index（{record}）")
            requests[queue_index] = {
                "queue_index": queue_index,
                "request_id": record.get("request_id"),
                "prefill_start": parse_ns(
                    record.get("prefill_start_ns"), "prefill_start_ns",
                    f"cpp.log:request#{queue_index}"),
                "prefill_end": parse_ns(
                    record.get("prefill_end_ns"), "prefill_end_ns",
                    f"cpp.log:request#{queue_index}"),
            }
    if not requests:
        fail(f"{cpp_log}: 未找到 type=request 记录——restore 分解需要 full "
             f"档 cpp.log（B1 批次 SH_METRICS_DETAIL=full；summary 档不落"
             f"逐请求记录）")
    rows = []
    n_no_anchor = 0
    n_no_boundary = 0
    n_zero_duration = 0
    for queue_index in sorted(requests):
        info = requests[queue_index]
        ticks = anchors.get(queue_index)
        if not ticks:
            n_no_anchor += 1
            rows.append({
                "queue_index": queue_index,
                "request_id": info.get("request_id"),
                "restore_start_ns": None,
                "restore_complete_ns": None,
                "pre_prefill_restore_ns": None,
                "hidden_restore_ns": None,
                "exposed_restore_stall_ns": None,
                "hidden_ratio": None,
            })
            continue
        restore_start = min(ticks)
        restore_complete = max(ticks)
        if info["prefill_start"] is None or info["prefill_end"] is None:
            n_no_boundary += 1
            rows.append({
                "queue_index": queue_index,
                "request_id": info.get("request_id"),
                "restore_start_ns": restore_start,
                "restore_complete_ns": restore_complete,
                "pre_prefill_restore_ns": None,
                "hidden_restore_ns": None,
                "exposed_restore_stall_ns": None,
                "hidden_ratio": None,
            })
            continue
        pre_prefill, hidden, exposed, ratio = restore_segments(
            restore_start, restore_complete, info["prefill_start"],
            info["prefill_end"])
        if restore_complete == restore_start:
            n_zero_duration += 1
        rows.append({
            "queue_index": queue_index,
            "request_id": info.get("request_id"),
            "restore_start_ns": restore_start,
            "restore_complete_ns": restore_complete,
            "pre_prefill_restore_ns": pre_prefill,
            "hidden_restore_ns": hidden,
            "exposed_restore_stall_ns": exposed,
            "hidden_ratio": ratio,
        })
    summary = {
        "n_requests": len(rows),
        "n_with_anchors": sum(1 for r in rows
                              if r["restore_complete_ns"] is not None),
        "n_no_anchor_na": n_no_anchor,
        "n_missing_prefill_boundary_na": n_no_boundary,
        "n_zero_restore_duration_na_ratio": n_zero_duration,
        "formula_source": "主规格 §3.2（A 类固定公式，逐字符实现）",
        "anchor_semantics": "restore_start=min(anchor ticks), "
                            "restore_complete=max(anchor ticks), "
                            "per-request attribution by subject_id"
                            "(=queue_index)",
    }
    return rows, summary


def cmd_restore(args: argparse.Namespace) -> int:
    rows, summary = collect(args.run_dir)
    stream, close = open_output(args.output, "slo_restore_decomposition.csv",
                                args.run_dir)
    try:
        write_csv(
            stream, OUTPUT_COLUMNS,
            ((r["queue_index"], r["request_id"], r["restore_start_ns"],
              r["restore_complete_ns"], r["pre_prefill_restore_ns"],
              r["hidden_restore_ns"], r["exposed_restore_stall_ns"],
              fmt_ratio(r["hidden_ratio"])) for r in rows))
    finally:
        if close:
            stream.close()
    emit_json(sys.stderr, {"command": "restore_decomposition", **summary})
    if args.json:
        jstream, jclose = open_output(
            args.json, "slo_restore_decomposition.json", args.run_dir)
        try:
            emit_json(jstream, {"command": "restore_decomposition",
                                **summary})
        finally:
            if jclose:
                jstream.close()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="restore_decomposition.py",
        description="WP5 restore 三段分解（full 档 cpp.log 的 memory_anchor "
                    "+ request 边界 → §3.2 固定公式）")
    parser.add_argument("run_dir", type=Path,
                        help="运行目录（含 full 档 cpp.log）")
    parser.add_argument("-o", "--output", default="",
                        help="CSV 输出（'-'=stdout；缺省写 "
                             "run_dir/slo_restore_decomposition.csv）")
    parser.add_argument("--json", default="",
                        help="汇总 JSON 输出路径（可选）")
    args = parser.parse_args()
    return int(cmd_restore(args) or 0)


if __name__ == "__main__":
    sys.exit(run_main(main))
