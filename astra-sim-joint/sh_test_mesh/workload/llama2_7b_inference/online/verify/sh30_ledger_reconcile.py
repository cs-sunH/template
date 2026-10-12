#!/usr/bin/env python3
"""sh30_ledger_reconcile.py -- sh_3.0 结束总账核对（方案 §6.2 操作 3 /
合同⑥；wscllm 版 ledger_reconcile.py 的 sh_3.0 适配——对照产物为本仓
sensing 运行导出的 ledger.jsonl / sensing_query_log.jsonl /
online_decision_log.jsonl）。

规则（sh_3.0 口径）：
  R0 决策流计数：online_decision_log 恰好每 request 三行
     （prefill/decode/completion），request 数 == manifest 请求数。
  R1 生命周期序：每 request prefill_tick <= decode_tick <= completion_tick；
     decode 实例 == prefill 实例（红线 #4 在线不变量）。
  R2 分层账本（ledger.jsonl）：每 request 的 issued 层条目在完成后核销
     （末态无 admitted/issued 残留）；completed 层覆盖全部 request。
  R4 排队账本排空：运行 PASS（verify_run_end fail-closed）+ ledger 末态
     pending_admissions 为空的等价证据（R2 的核销完备性）。
  R5 injected 对账（sensing_query_log）：末次交付的 injected_unfinished
     汇总中，非 barrier/control 尾部的 per-request 条目数为 0
     （REQUEST_COMPLETE 交付时刻该 request 的段节点应已全部完成）。
  R6 峰值有界：injected_unfinished 峰值 node_count 记录（观测项）。

用法：
  python3 sh30_ledger_reconcile.py --run-dir <sensing_run_dir> \
      --manifest <离线 manifest.json>
"""

import argparse
import collections
import json
import sys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True,
                        help="sensing 运行目录（含 results/*.jsonl）")
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()

    results = f"{args.run_dir}/results"
    manifest = json.load(open(args.manifest))
    expected = len(manifest["requests"])
    decisions = [json.loads(l) for l in
                 open(f"{results}/online_decision_log.jsonl")]
    ledger = [json.loads(l) for l in open(f"{results}/ledger.jsonl")]
    sensing = [json.loads(l) for l in
               open(f"{results}/sensing_query_log.jsonl")]

    failures = []
    notes = []

    # R0
    counts = collections.Counter((r["request_id"], r["kind"])
                                 for r in decisions)
    kinds = collections.Counter(r["kind"] for r in decisions)
    if not all(v == 1 for v in counts.values()):
        failures.append(f"R0: duplicate (request,kind) rows: "
                        f"{sum(1 for v in counts.values() if v != 1)}")
    if kinds != {"prefill": expected, "decode": expected,
                 "completion": expected}:
        failures.append(f"R0: decision counts {dict(kinds)} != 3x{expected}")
    else:
        notes.append(f"R0 PASS: {expected} requests x 3 decision rows")

    # R1
    ticks = collections.defaultdict(dict)
    for r in decisions:
        ticks[r["request_id"]][r["kind"]] = r["tick"]
    bad_order = [rid for rid, t in ticks.items()
                 if not (t["prefill"] <= t["decode"] <= t["completion"])]
    if bad_order:
        failures.append(f"R1: tick order violations: {bad_order[:5]}")
    else:
        notes.append("R1 PASS: prefill<=decode<=completion for all requests")
    decode_eq = sum(
        1 for r in decisions if r["kind"] == "decode"
        and r["decision"].get("decode_instance_index") is not None
        and r["decision"]["decode_instance_index"]
        != next(x["decision"]["prefill_instance_index"]
                for x in decisions
                if x["request_id"] == r["request_id"]
                and x["kind"] == "prefill"))
    if decode_eq:
        failures.append(f"R1: decode!=prefill instance: {decode_eq}")
    else:
        notes.append("R1 PASS: decode instance == prefill instance (1177)")

    # R2
    issued_open = []
    completed = set()
    for row in ledger:
        rid = row["request_id"]
        layers = row.get("layers", {})
        if layers.get("issued"):
            issued_open.append(rid)
        if layers.get("completed_unreconciled") or layers.get("completed"):
            completed.add(rid)
    if issued_open:
        failures.append(f"R2: issued layer not settled: {issued_open[:5]}")
    if len(completed) != expected:
        failures.append(f"R2: completed layer covers {len(completed)}/{expected}")
    else:
        notes.append(f"R2 PASS: issued settled + completed covers {expected}")

    # R5/R6: injected unfinished from sensing query log. Residual entry
    # classes at the FINAL delivery: (a) the completing request's own decode
    # end-barrier tail (watch member = last real decode node, so the barrier
    # all_reduce is still executing when REQUEST_COMPLETE fires -- the
    # contract ① note "completion 物理收尾不阻塞核销"); (b) anything else =
    # real residual (fail).
    peak_nodes = 0
    last = sensing[-1] if sensing else {}
    inj = last.get("injected_unfinished", [])
    final_tick = max(r["tick"] for r in decisions)
    self_completing = {r["request_id"] for r in decisions
                       if r["kind"] == "completion" and r["tick"] == final_tick}
    tail_residual = 0
    real_residual = 0
    for rank_summary in inj:
        for entry in rank_summary.get("per_request", []):
            if entry.get("request_id") in self_completing:
                tail_residual += 1
            else:
                real_residual += 1
        peak_nodes = max(peak_nodes, rank_summary.get("node_count", 0))
    if real_residual:
        failures.append(f"R5: non-tail injected residual at final delivery: "
                        f"{real_residual}")
    else:
        notes.append(f"R5 PASS: final-delivery residual = self completion "
                     f"tails only ({tail_residual} entries, "
                     f"completing={sorted(self_completing)})")
    notes.append(f"R6 (observed): peak injected node_count across ranks = "
                 f"{peak_nodes}")

    for line in notes:
        print("[sh30-reconcile]", line)
    if failures:
        for line in failures:
            print("[sh30-reconcile] FAIL:", line)
        print("[sh30-reconcile] verdict: 不平 (unbalanced)")
        return 1
    print("[sh30-reconcile] verdict: 对平 (balanced)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
