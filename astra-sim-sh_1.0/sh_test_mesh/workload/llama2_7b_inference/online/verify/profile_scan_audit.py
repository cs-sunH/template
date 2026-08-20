#!/usr/bin/env python3
"""profile_scan_audit.py -- 阶段 4 §7.3 事件索引队列 profile 审计。

输入:一次真实运行产出的 profile.jsonl(online_service.py 在运行结束写出,
每决策批一行 {delivery_sequence, tick, scanned_entries, full_scan_entries})。

断言(fail-closed,任一不满足即非 0 退出):
  1. 每批 full_scan_entries == 0 —— §7.3 之后不存在 O(总规模) 的全量扫描
     (任何 > 0 都意味着索引队列被绕过);
  2. 每批 scanned_entries <= total_requests —— 单批复杂度与总 request 数
     无关(20.csv 前 30s 全量 = 1177 请求;到期事件 + 受影响条目 + 就绪
     frontier,任何单批都远小于总量;上限可经 --total-requests 覆盖)。

输出:profile_report_20.md(与 profile.jsonl 同目录;逐批统计 + 结论)。
"""

import argparse
import json
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
            if not isinstance(row, dict) or not isinstance(
                    row.get("scanned_entries"), int):
                raise ValueError(
                    "profile row {} is not a scan-count dict: {!r}".format(
                        line_number, row))
            rows.append(row)
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="sh_1.0 phase-4 §7.3 profile scan audit")
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
    full_scan_rows = []
    for row in rows:
        if row["full_scan_entries"] != 0:
            full_scan_rows.append(row)
        if row["scanned_entries"] > max_scanned:
            max_scanned = row["scanned_entries"]
            max_row = row
        if row["scanned_entries"] > total:
            failures.append(
                "delivery {} scanned {} entries > total requests {}"
                .format(row["delivery_sequence"], row["scanned_entries"],
                        total))
    if full_scan_rows:
        failures.append(
            "full-scan entries > 0 in {} batches (e.g. delivery {}: {})"
            .format(len(full_scan_rows),
                    full_scan_rows[0]["delivery_sequence"],
                    full_scan_rows[0]["full_scan_entries"]))

    scanned_values = [row["scanned_entries"] for row in rows]
    scanned_values.sort()
    mean = sum(scanned_values) / len(scanned_values)
    p99 = scanned_values[int(len(scanned_values) * 0.99) - 1]

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
        out.write("| full_scan_entries > 0 的批数 | {} |\n".format(
            len(full_scan_rows)))
        out.write("\n## 结论\n\n")
        if failures:
            out.write("- **FAIL**: {}\n".format("; ".join(failures)))
        else:
            out.write("- PASS: 每批扫描条目数上限 {} << 总 request 数 {}"
                      "({:.1f}%),与总 request 规模无关;"
                      "full_scan_entries 全批为 0(无 O(总规模) 全量扫描)"
                      "。\n".format(max_scanned, total,
                                    100.0 * max_scanned / total))
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
        "(< total {}), full-scan rows {}, report {}"
        .format(len(rows), max_scanned, total, len(full_scan_rows),
                report_path),
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
