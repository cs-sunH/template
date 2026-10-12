#!/usr/bin/env python3
"""profile_scan_audit.py -- 阶段 4 §7.3 事件索引队列 profile 审计。

输入:一次真实运行产出的 profile.jsonl(online_service.py 在运行结束写出,
每决策批一行 {delivery_sequence, tick, scanned_entries})。

断言(fail-closed,任一不满足即非 0 退出):
  每批 scanned_entries <= total_requests —— 单批复杂度与总 request 数
  无关(20.csv 前 30s 全量 = 1177 请求;到期事件 + 受影响条目 + 就绪
  frontier,任何单批都远小于总量;上限可经 --total-requests 覆盖)。

  (2026-09-26 A2-x 合并处置:原第 1 条断言"每批 full_scan_entries == 0"
  随生产端 _profile_full_scan 死记账删除一并移除——该字段自始恒 0,
  新版 profile 行不再携带;旧产物多出的键在此处被自然忽略。)

输出:profile_report_20.md(与 profile.jsonl 同目录;逐批统计 + 结论)。
"""

import argparse
import json
import math
import os
import sys

_REPORT_NAME = "profile_report_20.md"


def _read_rows(path: str) -> list:
    rows = []
    with open(path, "r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            # (2026-10-01) bool is an int subclass and would slip through
            # as a 1-entry count; reject it like ledger_reconcile's row_int.
            if (not isinstance(row, dict)
                    or isinstance(row.get("scanned_entries"), bool)
                    or not isinstance(row.get("scanned_entries"), int)
                    or row.get("scanned_entries") < 0):
                raise ValueError(
                    "profile row {} is not a scan-count dict: {!r}".format(
                        line_number, row))
            rows.append(row)
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="face phase-4 §7.3 profile scan audit")
    parser.add_argument("--profile", required=True,
                        help="profile.jsonl 路径(真实运行产出)")
    parser.add_argument("--total-requests", type=int, default=1177,
                        help="输入总 request 数(20.csv 前 30s = 1177)")
    args = parser.parse_args(argv)

    rows = _read_rows(args.profile)
    if not rows:
        raise RuntimeError("profile.jsonl is empty: {}".format(args.profile))
    total = args.total_requests

    failures = []
    max_scanned = 0
    max_row = None
    for row in rows:
        if row["scanned_entries"] > max_scanned:
            max_scanned = row["scanned_entries"]
            max_row = row
        if row["scanned_entries"] > total:
            failures.append(
                "delivery {} scanned {} entries > total requests {}"
                .format(row["delivery_sequence"], row["scanned_entries"],
                        total))

    scanned_values = [row["scanned_entries"] for row in rows]
    scanned_values.sort()
    mean = sum(scanned_values) / len(scanned_values)
    # R5-S4(2026-09-29):p99 改全仓 nearest-rank 口径(rank = ceil(p*N),
    # index 夹取——与 slo_common.nearest_rank_percentile / C++
    # MetricCollector 同法;旧 int(N*0.99)-1 截断口径在 N<100 时系统性
    # 偏低甚至负回绕)。display-only 统计列。
    p99_rank = max(1, min(math.ceil(0.99 * len(scanned_values)),
                          len(scanned_values)))
    p99 = scanned_values[p99_rank - 1]
    # 零保护:--total-requests 0 且全零扫描时 total==0,百分比不可算
    # (旧写法 100.0*0/0 直接 ZeroDivisionError);此分支仅在无 failures
    # 时被消费,而无 failures + total<=0 蕴含 max_scanned==0,置 0.0 保
    # 持口径一致。
    max_pct = (100.0 * max_scanned / total) if total > 0 else 0.0

    report_dir = os.path.dirname(os.path.abspath(args.profile))
    report_path = os.path.join(report_dir, _REPORT_NAME)
    with open(report_path, "w", encoding="utf-8") as out:
        out.write("# 阶段 4 §7.3 每决策批扫描条目数 profile 报告\n\n")
        out.write("- 数据源: {}\n".format(args.profile))
        out.write("- 决策批数: {}\n".format(len(rows)))
        out.write("- 输入总 request 数: {}\n".format(total))
        out.write("\n## 统计\n\n")
        out.write("| 指标 | 值 |\n|---|---|\n")
        out.write("| 单批最大扫描条目数 | {} |\n".format(max_scanned))
        out.write("| 单批平均扫描条目数 | {:.1f} |\n".format(mean))
        out.write("| 单批 p99 扫描条目数 | {} |\n".format(p99))
        out.write("\n## 结论\n\n")
        if failures:
            out.write("- **FAIL**: {}\n".format("; ".join(failures)))
        else:
            out.write("- PASS: 每批扫描条目数上限 {} << 总 request 数 {}"
                      "({:.1f}%),与总 request 规模无关。\n"
                      .format(max_scanned, total, max_pct))
        if max_row is not None:
            out.write("- 最大批次: delivery {} (tick {}, scanned {})\n"
                      .format(max_row["delivery_sequence"], max_row["tick"],
                              max_row["scanned_entries"]))
        out.write("\n生成时间: phase-4 §7.3 profile audit\n")

    if failures:
        raise RuntimeError(
            "profile scan audit FAILED: {}".format("; ".join(failures)))
    print(
        "profile scan audit PASS: {} batches, max {} scanned/batch "
        "(< total {}), report {}"
        .format(len(rows), max_scanned, total, report_path),
        flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # noqa: BLE001 -- CLI 顶层 fail-closed
        print("profile_scan_audit: fatal: {}: {}".format(
            type(exc).__name__, exc),
            file=sys.stderr)
        sys.exit(1)
