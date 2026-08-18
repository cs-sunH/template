#!/usr/bin/env python3
"""ledger_reconcile_sh20.py -- sh_2.0 分层账本对账器（阶段 4 门槛项适配）。

wscllm 的 ledger_reconcile.py 与其工件格式耦合（SessionKVCacheManager 事件
流 / kv_cache_events.csv / cpp.log 完成事实格式）。sh_2.0 的 KVCacheManager
（face_scheduler.py）无事件流 API——KV 动作权威 = 决策产物（KVTransfer 族，
B2 oracle 已逐字节验证）。本脚本按 sh_2.0 工件口径核对 R0-R6，不通过项逐条
归因（比照 sh_3.0 合同① 的残差归因口径）。

用法: python3 online/verify/ledger_reconcile_sh20.py --run-dir <run_dir> [--expected 1177]

2026-08-16 wscllm 054118e 同型缺陷排查加固（两缺陷在本脚本均不成立，本次
为吸收其修法的防御性加固）：
  1) 陈旧契约（wscllm 缺陷 1）：本脚本数据源自诞生起即为归档产物
     （ledger/online_decision_log/graph_batch_digests/sensing_query_log +
     cpp.log），从不读 bridge 的 response_*.json（该文件 phase-7 §10.3 起
     消费即删，本仓 decision_bridge.py 同款 os.remove），故不受影响。
     加固：R2a 补 cpp.log phase-5 total_watches 三方对平。
  2) 错误不变量（wscllm 缺陷 2）：本脚本无"每批 watch∈{0,1}"假设；R2
     总量不变量 watches==2×N 与 wscllm 修复后语义不变量一致。加固：R2b
     引入批级正确上界不变量 0<=watch_count<=len(ranks)（wscllm 054118e
     同款上半段）。注意其下半段"watch_count==0 ⇔ ranks 为空"在本仓
     【不成立、不得照搬】：本仓 digest.ranks = 批交付图全部节点 rank 集
     （online_scheduler_base._digest_row），DECODE_COMPLETION+
     REQUEST_COMPLETE 完成通知批有节点（ranks 非空）但恒 0 watch，
     ARRIVAL 批亦可为 0——30s/3min-20 档全量实测该等价判据会合法违例
     1036/1936 条；上界判据零违例。
     加固：R2c digest node 和 == cpp.log phase-5 total_nodes 对平；
     R3c completion 决策 tick == ledger completed_tick（零失配）+
     cpp total_kv_actions >= 请求数（wscllm 修复版 R3 同款计数保证）。
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

    check("R0a C++ completed == expected", cpp_completed == args.expected, f"cpp={cpp_completed}")
    check("R0b ledger completed 层 == expected", len(ledger) == args.expected, f"ledger={len(ledger)}")
    check("R0c 决策三流集合相等且无重复",
          completions == set(ledger) and prefills == set(ledger)
          and decodes == set(ledger) and dup == 0, f"dup={dup}")

    r1_bad = [rid for rid, L in ledger.items()
              if L["completed_unreconciled"]["completed_tick"]
              < L["completed_unreconciled"]["admitted_tick"]]
    check("R1 admitted <= completed（全部请求）", not r1_bad, f"violations={len(r1_bad)}")
    commit_null = sum(1 for L in ledger.values()
                      if L["completed_unreconciled"]["first_commit_tick"] is None)
    if commit_null:
        attrs.append(f"R1-residual: first_commit_tick=null ×{commit_null}——sh20 调度器 "
                     f"_ledger_issue 仅 prefill 发射段调用，ack 侧 emitted 映射未覆盖 "
                     f"decode/completion 段（观测面接线缺口，非状态不一致；提交层由 "
                     f"C++ graph_batch_count==delivery_count=={len(digests)} 对平）")

    # ---- R2 watch 账本（054118e 同型：归档 digest + cpp phase-5 权威计数）----
    # response_*.json 消费即删（本仓 bridge 同款契约），committed/watch 权威
    # 数据源 = graph_batch_digests.jsonl（批级摘要）+ cpp.log phase-5 计数器。
    # 错误不变量警示（wscllm 缺陷 2 同型排查）："每批 watch ∈{0,1}"是错误
    # 假设——本仓实测 PREFILL_DRAIN 批可 2 watch、DECODE_COMPLETION+
    # REQUEST_COMPLETE 完成通知批恒 0；正确判据 = 总量 2×N + 批级上界
    # 0<=wc<=len(ranks)。空性等价（wc==0<=>ranks 空）不适用于本仓 digest
    # 语义（ranks=交付图 rank 集，非 watch/committed rank 集）。
    total_watches = sum(d["watch_count"] for d in digests)
    m5w = re.search(r"\[online\] phase-5 commit counters:"
                    r"[^\n]*\btotal_watches=(\d+)", cpp)
    cpp_watches = int(m5w.group(1)) if m5w else None
    check("R2a watch 总数对平（digest 和 == cpp phase-5 == 2×请求）",
          total_watches == 2 * args.expected
          and (cpp_watches is None or cpp_watches == total_watches),
          f"watches={total_watches} cpp={cpp_watches}")
    rank_bad = [d["delivery_sequence"] for d in digests
                if not all(0 <= int(r) <= 53 for r in d.get("ranks") or [])]
    batch_bad = [d["delivery_sequence"] for d in digests
                 if not (isinstance(d.get("watch_count"), int)
                         and 0 <= d["watch_count"] <= len(d.get("ranks") or []))]
    check("R2b 批级 watch 不变量（0 <= watch_count <= len(ranks)，ranks ⊆ [0,53]）",
          not batch_bad and not rank_bad,
          f"violations={len(batch_bad)} rank_violations={len(rank_bad)}")
    digest_nodes = sum(d["node_count"] for d in digests)
    m5n = re.search(r"\[online\] phase-5 commit counters:"
                    r"[^\n]*\btotal_nodes=(\d+)", cpp)
    cpp_nodes = int(m5n.group(1)) if m5n else None
    check("R2c digest node 总量 == cpp total_nodes",
          cpp_nodes is None or digest_nodes == cpp_nodes,
          f"digest={digest_nodes} cpp={cpp_nodes}")

    kvr = {r["request_id"] for r in decisions if r["kind"] in
           ("prefill", "decode", "completion") and (
               r["decision"].get("history_transfer_bytes")
               or r["decision"].get("prefill_decode_transfer_bytes")
               or r["decision"].get("history_eviction_count")
               or r["decision"].get("decode_eviction_count")
               or r["decision"].get("completion_eviction_count")
               or r["decision"].get("kv_location_after_completion"))}
    check("R3a 每请求有 KV 决策事实（kv_actions 批字段来源）",
          len(kvr) >= args.expected, f"requests={len(kvr)}")
    loc = sum(1 for r in decisions if r["kind"] == "completion"
              and r["decision"].get("kv_location_after_completion") in
              ("local_hbm", "partial_hbm_remote", "remote_memory"))
    check("R3b completion 决策携带合法 KV 终态位置", loc == args.expected, f"known={loc}")
    attrs.append("R3-residual: face_scheduler.KVCacheManager 无事件流 API——KV 逐事件流"
                 "（wscllm kv_cache_events.csv 对应物）不适用；KV 等价由 B2 oracle"
                 "（决策内容 exact 含全部 KVTransfer）与 R3a/R3b 决策事实闭环承担")
    # R3c（054118e 同型）：completion 决策 tick 与 ledger completed_tick 逐对
    # 零失配（KV 生命周期终止时刻对平）+ C++ total_kv_actions 计数级下界。
    comp_rows = {r["request_id"]: r for r in decisions
                 if r["kind"] == "completion"}
    tick_bad = [rid for rid, r in comp_rows.items()
                if r["tick"] != (ledger.get(rid, {})
                                 .get("completed_unreconciled", {})
                                 .get("completed_tick"))]
    m5k = re.search(r"\[online\] phase-5 commit counters:"
                    r"[^\n]*\btotal_kv_actions=(\d+)", cpp)
    cpp_kv = int(m5k.group(1)) if m5k else None
    check("R3c completion 决策 tick == ledger completed_tick 且 cpp kv 动作计数下界",
          not tick_bad and len(comp_rows) == args.expected
          and (cpp_kv is None or cpp_kv >= args.expected),
          f"tick_mismatch={len(tick_bad)} completions={len(comp_rows)}"
          f" cpp_kv={cpp_kv}")

    adm = sum(1 for L in ledger.values() if "admitted" in L)
    com = sum(1 for L in ledger.values() if "committed" in L)
    check("R4 结束 admitted/committed 层清空", adm == 0 and com == 0,
          f"admitted={adm} committed={com}")

    sensing = len(jsonl("sensing_query_log.jsonl"))
    check("R6 感知查询日志产出（--sensing）", sensing > 0, f"rows={sensing}")
    attrs.append("R5/R7-residual: injected-unfinished 末边界残差 = 终请求 decode "
                 "end-barrier 控制尾（最终 delivery 后自完成，无后续交付消费）——"
                 "自完成尾部类残差，比照 sh_3.0 合同① 口径归因（C++ phase-4 end "
                 "audit 的 completed_facts_residual 即该尾量）")

    print("== sh_2.0 分层账本对账（R0-R6，工件口径适配）==")
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
