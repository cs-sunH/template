#!/usr/bin/env python3
"""ledger_reconcile_sh10.py -- sh_1.0 分层账本对账器（阶段 4 门槛项适配）。

sh_2.0 的 ledger_reconcile_sh20.py 为模板；本仓差异（合同⑥/两态 KV/三段式）：
  - KV 权威 = 决策日志六类转移元组（B2 oracle 已逐字节验证）；
  - remote FIFO 为实账本层（AnalyticalRemoteMemory 26 端口 FIFO，C++ 侧
    Workload 计数采集，remote_fifo_ledger.jsonl 逐交付快照）——本对账器的
    核心对平项 RF1-RF4：逐交付覆盖 / drain 守恒 / Python 决策字节 vs C++
    实发字节逐 rank 对平（残差逐条归因）/ 峰值占用登记；
  - local HBM job 层 = 显式"不适用"占位（本仓无 LocalHbmBandwidthModel）。

用法: python3 online/verify/ledger_reconcile_sh10.py --run-dir <sensing_run_dir>
                                          [--expected 1177]
退出码: 全部对平 -> 0;任何不平 -> 1(fail-closed)。
"""
import argparse, collections, json, os, re, sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--expected", type=int, default=1177)
    args = p.parse_args()
    results = os.path.join(args.run_dir, "results")
    cpp = ""
    path = os.path.join(args.run_dir, "cpp.log")
    if os.path.exists(path):
        cpp = open(path, encoding="utf-8", errors="replace").read()

    def jsonl(name):
        rows = []
        q = os.path.join(results, name)
        if os.path.exists(q):
            for line in open(q, encoding="utf-8"):
                if line.strip():
                    rows.append(json.loads(line))
        return rows

    ledger = {r["request_id"]: r["layers"] for r in jsonl("ledger.jsonl")}
    decisions = jsonl("online_decision_log.jsonl")
    digests = jsonl("graph_batch_digests.jsonl")
    fifo_rows = jsonl("remote_fifo_ledger.jsonl")

    checks, attrs = [], []

    def check(name, ok, note=""):
        checks.append((name, bool(ok), note))

    m = re.search(r"\[online\] service counters: .*completed=(\d+)", cpp)
    cpp_completed = int(m.group(1)) if m else None
    kinds = collections.Counter((r["kind"], r["request_id"]) for r in decisions)
    completions = {rid for (k, rid) in kinds if k == "completion"}
    prefills = {rid for (k, rid) in kinds if k == "prefill"}
    decodes = {rid for (k, rid) in kinds if k == "decode"}
    dup = sum(1 for (k, _), c in kinds.items() if c > 1)

    check("R0a C++ completed == expected", cpp_completed == args.expected,
          f"cpp={cpp_completed}")
    check("R0b ledger completed 层 == expected", len(ledger) == args.expected,
          f"ledger={len(ledger)}")
    check("R0c 决策三流集合相等且无重复",
          completions == set(ledger) and prefills == set(ledger)
          and decodes == set(ledger) and dup == 0, f"dup={dup}")

    r1_bad = [rid for rid, L in ledger.items()
              if L["completed_unreconciled"]["completed_tick"]
              < L["completed_unreconciled"]["admitted_tick"]]
    check("R1 admitted <= completed（全部请求）", not r1_bad,
          f"violations={len(r1_bad)}")
    commit_null = sum(1 for L in ledger.values()
                      if L["completed_unreconciled"]["first_commit_tick"]
                      is None)
    if commit_null:
        attrs.append(
            f"R1-residual: first_commit_tick=null ×{commit_null}——sh10 调度器 "
            f"_note_emitted 仅段 1/段 2 发射调用（段 3 completion 边界无新 "
            f"watch，不产生发射凭据），观测面接线口径与 sh_2.0 同款；提交层由 "
            f"C++ graph_batch==delivery=={len(digests)} 对平（graph_batch_audit）")

    # R2 hardened (sh_2.0 reconcile-fix 同款, 2026-08-16): three layers.
    # digest.ranks semantics measured on this repo's artifacts first
    # (sh_2.0对账工具同型缺陷排查报告 §1.3 判据): sh_1.0 ranks = the
    # batch's delivered-graph node rank set, and the wscllm emptiness
    # equivalence (wc==0 <=> ranks empty) does NOT hold here -- 1042/3531
    # legitimate violations on the 30s sensing run (DECODE_COMPLETION+
    # REQUEST_COMPLETE completion-tail batches carry segment-3 nodes but no
    # new watch; 实录 mechanism ruling (a) one-way coverage). Emptiness
    # assertions are therefore NOT ported; the portable invariants are the
    # per-batch upper bound and the rank-set legality.
    total_watches = sum(d["watch_count"] for d in digests)
    m5 = re.search(r"phase-5 commit counters:.*total_watches=(\d+)", cpp)
    cpp_watches = int(m5.group(1)) if m5 else None
    check("R2a watch 总量三方对平（digest 和 == cpp phase-5 == 2×请求）",
          total_watches == 2 * args.expected
          and (cpp_watches is None or cpp_watches == total_watches),
          f"digest={total_watches} cpp={cpp_watches} 2N={2 * args.expected}")
    bound_bad = [d["delivery_sequence"] for d in digests
                 if not (0 <= d["watch_count"] <= len(d.get("ranks") or []))]
    ranks_bad = [d["delivery_sequence"] for d in digests
                 if any(r < 0 or r > 53 for r in (d.get("ranks") or []))]
    check("R2b 批级上界 0<=watch_count<=len(ranks) + ranks⊆[0,53]",
          not bound_bad and not ranks_bad,
          f"bound_bad={len(bound_bad)} ranks_bad={len(ranks_bad)}"
          f"（空性等价不移植：实测 1042 合法违例）")
    mn = re.search(r"phase-5 commit counters:.*total_nodes=(\d+)", cpp)
    cpp_nodes = int(mn.group(1)) if mn else None
    digest_nodes = sum(d["node_count"] for d in digests)
    check("R2c digest node 和 == cpp phase-5 total_nodes",
          cpp_nodes is None or cpp_nodes == digest_nodes,
          f"digest={digest_nodes} cpp={cpp_nodes}")

    # R3: sh_1.0 两态 KV（LOCAL_HBM/REMOTE_MEMORY）。六类转移元组在决策载荷
    # 的五个字段里;R3a = 每请求 prefill 决策存在;R3b = completion 携带合法
    # 两态终态位置。
    def has_kv(d):
        dec = d.get("decision", {})
        return bool(
            dec.get("history_transfer")
            or dec.get("history_evictions")
            or dec.get("prefill_evictions")
            or dec.get("prefill_decode_transfer")
            or dec.get("decode_evictions")
            or dec.get("completion_evictions"))

    kv_requests = {r["request_id"] for r in decisions
                   if r["kind"] == "prefill"}
    kv_facts = {r["request_id"] for r in decisions
                if r["kind"] != "completion" and has_kv(r)}
    check("R3a 每请求有 prefill 决策行", len(kv_requests) == args.expected,
          f"requests={len(kv_requests)}")
    loc = sum(1 for r in decisions if r["kind"] == "completion"
              and r["decision"].get("kv_location_after_completion")
              in ("local_hbm", "remote_memory"))
    check("R3b completion 决策携带合法两态 KV 终态位置",
          loc == args.expected, f"known={loc}")
    # R3c hardened (sh_2.0 reconcile-fix 同款): the completion decision tick
    # must equal the ledger's completed_tick per request (zero mismatch).
    # The sh_2.0 sub-check "cpp total_kv_actions >= requests" is NOT
    # applicable here: sh_1.0 encodes KV traffic as graph MEM nodes (cpp
    # total_kv_actions is 0 by design on every official run); the KV-side
    # accounting closure is RF3's per-rank FIFO byte balance + the B2
    # oracle's 2510-row transfer-tuple equality.
    comp_ticks = {r["request_id"]: r["tick"] for r in decisions
                  if r["kind"] == "completion"}
    tick_bad = [(rid, comp_ticks.get(rid),
                 ledger[rid]["completed_unreconciled"]["completed_tick"])
                for rid in ledger
                if comp_ticks.get(rid)
                != ledger[rid]["completed_unreconciled"]["completed_tick"]]
    check("R3c completion 决策 tick == ledger completed_tick（逐对零失配）",
          not tick_bad, f"mismatches={len(tick_bad)}"
          f"{tick_bad[:3] if tick_bad else ''}")
    attrs.append(
        "R3-residual: KV 逐事件流（wscllm kv_cache_events.csv 对应物）不适用"
        "——KV 等价由 B2 oracle（决策内容 exact + 六类转移元组 2510 行逐条"
        "全等）与 R3a/R3b 决策事实 + RF3 FIFO 字节对平闭环承担；cpp "
        "total_kv_actions==0（KV 编码为图 MEM 节点，本仓设计），sh_2.0 的 "
        "kv_actions 计数下界不适用（登记）")

    adm = sum(1 for L in ledger.values() if "admitted" in L)
    com = sum(1 for L in ledger.values() if "committed" in L)
    iss = sum(1 for L in ledger.values() if "issued" in L)
    check("R4 结束 admitted/committed/issued 层清空",
          adm == 0 and com == 0 and iss == 0,
          f"admitted={adm} committed={com} issued={iss}")

    sensing = len(jsonl("sensing_query_log.jsonl"))
    check("R5 感知查询日志产出（--sensing，逐交付）", sensing == len(digests),
          f"rows={sensing} deliveries={len(digests)}")
    attrs.append(
        "R6/R7-residual: injected-unfinished 末边界残差 = 终请求 decode "
        "end-barrier 控制尾（最终 delivery 后自完成，无后续交付消费）——"
        "自完成尾部类残差，比照 sh_3.0 合同① 口径归因（C++ phase-4 end "
        "audit 的 completed_facts_residual 即该尾量）；local HBM job 层 = "
        "不适用占位（无 LocalHbmBandwidthModel，合同⑥）")

    # ---- remote FIFO 实账本层（本仓核心差异项）RF1-RF4 ----
    fm = re.search(
        r"\[online\] remote fifo ledger: ports=(\d+) issued=(\d+) "
        r"issued_bytes=(\d+) completed=(\d+) completed_bytes=(\d+) "
        r"peak_pending=(\d+) peak_in_flight_bytes=(\d+)", cpp)
    if not fm:
        check("RF0 cpp.log remote fifo ledger 行存在", False, "missing")
    else:
        cpp_ports, cpp_issued, cpp_issued_bytes, cpp_completed, \
            cpp_completed_bytes, cpp_peak_pending, cpp_peak_bytes = (
                int(fm.group(i)) for i in range(1, 8))
        check("RF0 cpp.log remote fifo ledger 行存在", True,
              f"ports={cpp_ports} issued={cpp_issued}")

        # RF1: per-epoch sidecar covers exactly the delivery sequence.
        seqs = [r["delivery_sequence"] for r in fifo_rows]
        check("RF1 FIFO 快照逐交付覆盖（无跳号）",
              seqs == list(range(len(seqs))) and len(fifo_rows) == len(digests),
              f"rows={len(fifo_rows)} deliveries={len(digests)}")

        # RF2: conservation -- every port drained; sidecar last-state counters
        # equal the cpp.log run-end totals (single source, two views).
        last = {}
        for row in fifo_rows:
            for port in row["ports"]:
                last[port["rank"]] = port
        undrained = {r: c for r, c in last.items()
                     if c["issued_count"] != c["completed_count"]}
        sidecar_issued = sum(c["issued_count"] for c in last.values())
        sidecar_issued_bytes = sum(c["issued_bytes"] for c in last.values())
        check("RF2a 全端口 drain（issued==completed，C++ 侧 fail-closed 复核）",
              not undrained and cpp_issued == cpp_completed,
              f"undrained={sorted(undrained)}")
        check("RF2b 快照累计 issue 计数/字节 == cpp.log 运行末总账",
              sidecar_issued == cpp_issued
              and sidecar_issued_bytes == cpp_issued_bytes,
              f"sidecar=({sidecar_issued},{sidecar_issued_bytes}) "
              f"cpp=({cpp_issued},{cpp_issued_bytes})")
        check("RF2c 字节守恒（issued_bytes==completed_bytes）",
              cpp_issued_bytes == cpp_completed_bytes,
              f"issued={cpp_issued_bytes} completed={cpp_completed_bytes}")

        # RF3: Python 决策字节 vs C++ 实发字节，逐 edge rank 对平。
        # 每个 remote_store/remote_load 转移分片 = edge_rank 上一个
        # MEM_LOAD/MEM_STORE 节点（generate_face_trace._emit_kv_transfer）
        # -> 一次 AnalyticalRemoteMemory issue（tensor_size = shard bytes）。
        expect_counts = collections.Counter()
        expect_bytes = collections.Counter()
        transfer_kinds = collections.Counter()
        for r in decisions:
            dec = r.get("decision", {})
            transfers = []
            transfers.append(dec.get("history_transfer"))
            for key in ("history_evictions", "prefill_evictions",
                        "decode_evictions", "completion_evictions"):
                transfers.extend(dec.get(key) or [])
            transfers.append(dec.get("prefill_decode_transfer"))
            for t in transfers:
                if not t:
                    continue
                transfer_kinds[t["kind"]] += 1
                if t["kind"] not in ("remote_store", "remote_load"):
                    continue
                for s in t["shards"]:
                    if s.get("edge_rank") is None:
                        continue
                    expect_counts[s["edge_rank"]] += 1
                    expect_bytes[s["edge_rank"]] += s["bytes"]
        got_counts = {r: c["issued_count"] for r, c in last.items()}
        got_bytes = {r: c["issued_bytes"] for r, c in last.items()}
        count_diff = {rank: (expect_counts[rank], got_counts.get(rank))
                      for rank in sorted(set(expect_counts) | set(got_counts))
                      if expect_counts[rank] != got_counts.get(rank)}
        byte_diff = {rank: (expect_bytes[rank], got_bytes.get(rank))
                     for rank in sorted(set(expect_bytes) | set(got_bytes))
                     if expect_bytes[rank] != got_bytes.get(rank)}
        check("RF3a Python 决策 MEM 分片数 vs C++ 实发 issue 数（逐 rank）",
              not count_diff, f"diff_ranks={count_diff}")
        check("RF3b Python 决策 MEM 字节 vs C++ issued_bytes（逐 rank）",
              not byte_diff, f"diff_ranks={byte_diff}")
        check("RF3c C++ 端口数 == 决策中出现过的 edge rank 数",
              len(last) == len(expect_counts),
              f"cpp_ports={len(last)} expect={len(expect_counts)}")
        check("RF3d remote 转移族存在（六类中 remote_store/remote_load 覆盖）",
              transfer_kinds.get("remote_store", 0)
              + transfer_kinds.get("remote_load", 0) > 0,
              f"kinds={dict(transfer_kinds)}")
        attrs.append(
            f"RF-residual: peak_pending={cpp_peak_pending}（26 端口单服务队列"
            f"实测未排队——远端访问被依赖链串行化，物理事实观测非缺陷）；"
            f"peak_in_flight_bytes={cpp_peak_bytes}；replay 模式 MEM 即时完成"
            f"（合同⑦第④项）不走真实 FIFO，其 FIFO 账本恒零——合同语义非泄漏")

    print("== sh_1.0 分层账本对账（R0-R5 + RF1-RF4 remote FIFO 实账本层）==")
    ok_all = True
    for name, ok, note in checks:
        print(f"[{'PASS' if ok else 'FAIL'}] {name} {note}")
        ok_all = ok_all and ok
    print("\n== 残差归因（登记）==")
    for a in attrs:
        print(f"[ATTR] {a}")
    print(f"\nRESULT: {'BALANCED' if ok_all else 'UNBALANCED'}")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
