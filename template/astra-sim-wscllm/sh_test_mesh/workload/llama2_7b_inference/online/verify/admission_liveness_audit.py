#!/usr/bin/env python3
"""Offline admission-liveness auditor for online decision logs.

方案一(2026-10-20)风险一守卫:扫描 run 目录的 online_decision_log.jsonl,
当同一请求连续 --max-blocked-streak(缺省 3)个决策批处于 admission_blocked,
且这些阻塞行自带的观测字段显示"实例 KV 空闲字节逐 rank 都 >= 本请求所需
字节"时判 FAIL——空闲足够却持续阻塞,意味着某条容量释放路径可能漏发了
重试信号(2026-08-31 死端家族的日志可判定形态),fail-closed 拒绝交付。

阻塞行的观测字段(kv_free_bytes_by_rank / required_bytes_by_rank)由调度器
在每次 admission_blocked 决策时写入;无该字段的旧构建日志明确报"字段
缺失,跳过",绝不误判。本工具只消费落盘日志,不读内存状态,可对任意
历史 run 离线重放。

用法:
    python3 admission_liveness_audit.py <run_dir 或 jsonl 路径> \
        [--max-blocked-streak N]

退出码:0 = 通过或明确跳过;1 = 活性违例或输入不可用。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DECISION_LOG_NAME = "online_decision_log.jsonl"
BLOCKED_KIND = "admission_blocked"
# 这些 kind 标志该请求取得进展(准入成功或完成),切断阻塞 streak 计数。
PROGRESS_KINDS = {"prefill", "decode", "completion"}
_LOG_CANDIDATES = (
    Path("bridge") / DECISION_LOG_NAME,
    Path("results") / DECISION_LOG_NAME,
    Path(DECISION_LOG_NAME),
)


def resolve_log_path(target: Path) -> Path:
    if target.is_file():
        return target
    if target.is_dir():
        for candidate in _LOG_CANDIDATES:
            path = target / candidate
            if path.is_file():
                return path
        raise SystemExit(
            "admission_liveness_audit: no {} under {}".format(
                DECISION_LOG_NAME, target))
    raise SystemExit(
        "admission_liveness_audit: input not found: {}".format(target))


def _rank_map(value) -> dict | None:
    """Normalize a {rank: bytes} JSON object to {int: int}; None if absent."""
    if not isinstance(value, dict):
        return None
    try:
        return {int(rank): int(bytes_) for rank, bytes_ in value.items()}
    except (TypeError, ValueError):
        return None


def _feasible(entry: dict) -> bool | None:
    """True/False when the row's free-vs-required verdict is decidable."""
    decision = entry.get("decision")
    if not isinstance(decision, dict):
        return None
    free = _rank_map(decision.get("kv_free_bytes_by_rank"))
    required = _rank_map(decision.get("required_bytes_by_rank"))
    if free is None or required is None:
        return None
    ranks = set(free) | set(required)
    if not ranks:
        return None
    return all(free.get(rank, 0) >= required.get(rank, 0) for rank in ranks)


def audit_log(path: Path, max_streak: int) -> int:
    streaks: dict[str, list[dict]] = {}
    violations = []
    missing_field_streaks = 0
    total_blocked = 0
    with path.open("r", encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                print("admission_liveness_audit: FAIL: "
                      "undecodable decision log line: {}".format(exc))
                return 1
            request_id = row.get("request_id")
            kind = row.get("kind")
            if kind == BLOCKED_KIND:
                total_blocked += 1
                if not isinstance(request_id, str):
                    print("admission_liveness_audit: FAIL: "
                          "admission_blocked row without request_id")
                    return 1
                streaks.setdefault(request_id, []).append(row)
                continue
            if kind in PROGRESS_KINDS and request_id in streaks:
                streaks.pop(request_id)
    # 日志结束后仍挂着的 streak 一并参与判定(结束态阻塞本身已由
    # verify_run_end 断言,这里只按"空闲足够却阻塞"口径复核)。
    for request_id, rows in sorted(streaks.items()):
        if len(rows) < max_streak:
            continue
        verdicts = [_feasible(row) for row in rows]
        if any(verdict is None for verdict in verdicts):
            missing_field_streaks += 1
            continue
        if all(verdict is True for verdict in verdicts):
            violations.append((request_id, rows))
    if total_blocked == 0:
        print("admission_liveness_audit: PASS: no admission_blocked rows")
        return 0
    if violations:
        print("admission_liveness_audit: FAIL: request(s) stayed blocked "
              "with enough free KV bytes for >= {} decision batches:"
              .format(max_streak))
        for request_id, rows in violations:
            for row in rows:
                decision = row.get("decision", {})
                print("  seq={} tick={} request={} stage={} instance={} "
                      "free_by_rank={} required_by_rank={}".format(
                          row.get("seq"), row.get("tick"), request_id,
                          decision.get("stage"),
                          decision.get("instance_index"),
                          json.dumps(decision.get("kv_free_bytes_by_rank"),
                                     sort_keys=True),
                          json.dumps(decision.get("required_bytes_by_rank"),
                                     sort_keys=True)))
        return 1
    if missing_field_streaks:
        print("admission_liveness_audit: SKIP: long blocked streak without "
              "enrichment fields (pre-enrichment build), not judgeable")
        return 0
    print("admission_liveness_audit: PASS: {} blocked row(s), no "
          "free-capacity deadlock streak".format(total_blocked))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="offline admission-liveness audit of "
                    "online_decision_log.jsonl (fail-closed)")
    parser.add_argument("target", type=Path,
                        help="run directory or decision log jsonl path")
    parser.add_argument("--max-blocked-streak", type=int, default=3,
                        help="consecutive blocked decision batches with "
                             "enough free bytes that trip a failure "
                             "(default: %(default)s)")
    args = parser.parse_args(argv)
    if args.max_blocked_streak < 1:
        parser.error("--max-blocked-streak must be >= 1")
    log_path = resolve_log_path(args.target)
    return audit_log(log_path, args.max_blocked_streak)


if __name__ == "__main__":
    sys.exit(main())
